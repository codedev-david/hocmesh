use crate::client::MeshClient;
use anyhow::{Context, Result, ensure};
use mesh_ai::{InferenceAssignment, PromptOutput};
use mesh_core::compute::execute_work;
use mesh_core::proximity::Vivaldi;
use mesh_gpu::{InferenceBackend, InferenceRequest, LlamaCppBackend};
use mesh_model::{ChunkStore, ModelRegistry};
use mesh_protocol::NodeCapabilities;
use mesh_transport::{
    HttpPeerSource, ProbeState, SeedServerState, probe_peer, probe_router, seed_from_peer,
    seed_router,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
    time::Duration,
};
use tokio::task::JoinSet;

pub struct AiWorkerConfig {
    pub home: PathBuf,
    pub runtime: PathBuf,
    pub gpu_layers: u32,
    pub seed_listen: Option<String>,
}

/// How often a node re-measures where it sits in the network.
///
/// Latency space moves on the scale of routing changes, not seconds, and every
/// round costs real round trips on other people's machines. Measuring rarely
/// and keeping the fitted position across restarts beats measuring often.
const PROXIMITY_INTERVAL: Duration = Duration::from_secs(60);

pub struct ProximityConfig {
    pub home: PathBuf,
    /// Address this node serves probes on, if the operator opted in.
    pub probe_listen: Option<String>,
}

pub async fn run(
    client: MeshClient,
    capabilities: NodeCapabilities,
    workers: usize,
    poll_ms: u64,
    ai: Option<AiWorkerConfig>,
    proximity: ProximityConfig,
) -> Result<()> {
    let registered = client.register(&capabilities).await?;
    println!("MESH node {} registered", registered.node_id);
    println!(
        "Available balance: {:.3} CU",
        registered.balance_mcu as f64 / 1000.0
    );
    println!("Contribution workers: {}", workers);

    let capabilities = Arc::new(RwLock::new(capabilities));
    let heartbeat_client = client.clone();
    let heartbeat_caps = capabilities.clone();
    let heartbeat = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        loop {
            interval.tick().await;
            let Some(snapshot) = heartbeat_caps.read().ok().map(|caps| caps.clone()) else {
                tracing::error!("capability lock poisoned; stopping heartbeat");
                return;
            };
            if let Err(error) = heartbeat_client.heartbeat(&snapshot).await {
                tracing::warn!(%error, "heartbeat failed");
            }
        }
    });

    let tracker = Arc::new(Mutex::new(Vivaldi::load_or_seeded(
        &proximity.home,
        client.node_id().as_bytes(),
    )));

    let mut probe_server = None;
    if let Some(listen) = proximity.probe_listen.as_ref() {
        let listener = tokio::net::TcpListener::bind(listen).await?;
        let state = ProbeState::new(client.node_id(), tracker.clone());
        println!("Serving latency probes on {listen}");
        probe_server = Some(tokio::spawn(async move {
            axum::serve(listener, probe_router(state)).await
        }));
    }

    let proximity_task = tokio::spawn(proximity_loop(
        client.clone(),
        tracker,
        capabilities.clone(),
        proximity.home,
    ));

    let mut workers_set = JoinSet::new();
    for worker_id in 0..workers.max(1) {
        let worker_client = client.clone();
        workers_set
            .spawn(async move { worker_loop(worker_client, worker_id, poll_ms.max(100)).await });
    }

    let mut seed_server = None;
    if let Some(config) = ai.as_ref()
        && let Some(listen) = config.seed_listen.as_ref()
    {
        let store = Arc::new(ChunkStore::open(config.home.join("model-cache"))?);
        let manifests = ModelRegistry::open(config.home.join("model-registry.db"))?.list()?;
        let state = SeedServerState::new(store, manifests)?;
        let listener = tokio::net::TcpListener::bind(listen).await?;
        seed_server = Some(tokio::spawn(async move {
            axum::serve(listener, seed_router(state)).await
        }));
    }
    if let Some(config) = ai {
        let ai_client = client.clone();
        workers_set.spawn(async move { ai_worker_loop(ai_client, config, poll_ms.max(100)).await });
    }

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            println!("Shutdown requested; stopping MESH node.");
        }
        result = workers_set.join_next() => {
            match result {
                Some(Ok(Ok(()))) => tracing::warn!("worker exited unexpectedly"),
                Some(Ok(Err(error))) => tracing::error!(%error, "worker failed"),
                Some(Err(error)) => tracing::error!(%error, "worker task panicked"),
                None => tracing::warn!("all workers exited"),
            }
        }
    }

    heartbeat.abort();
    proximity_task.abort();
    if let Some(server) = probe_server {
        server.abort();
    }
    if let Some(server) = seed_server {
        server.abort();
    }
    workers_set.abort_all();
    Ok(())
}

async fn ai_worker_loop(client: MeshClient, config: AiWorkerConfig, poll_ms: u64) -> Result<()> {
    let idle_delay = Duration::from_millis(poll_ms);
    loop {
        let poll = match client.poll_inference().await {
            Ok(poll) => poll,
            Err(error) => {
                tracing::warn!(%error, "AI poll failed");
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
        };
        let Some(assignment) = poll.assignment else {
            tokio::time::sleep(idle_delay).await;
            continue;
        };
        let assignment_id = assignment.assignment_id.clone();
        match execute_inference_assignment(&config, assignment).await {
            Ok(outputs) => match client
                .report_inference(assignment_id.clone(), outputs)
                .await
            {
                Ok(response) => {
                    tracing::info!(%assignment_id, job_completed=response.job_completed, "AI assignment completed")
                }
                Err(error) => tracing::warn!(%assignment_id, %error, "AI result submission failed"),
            },
            Err(error) => {
                tracing::warn!(%assignment_id, %error, "AI assignment failed; requesting reroute");
                if let Err(report_error) = client
                    .fail_inference(assignment_id, error.to_string())
                    .await
                {
                    tracing::warn!(%report_error, "AI failure report was not accepted");
                }
            }
        }
    }
}

async fn execute_inference_assignment(
    config: &AiWorkerConfig,
    assignment: InferenceAssignment,
) -> Result<Vec<PromptOutput>> {
    assignment.manifest.validate()?;
    let store = ChunkStore::open(config.home.join("model-cache"))?;
    if assignment
        .manifest
        .chunks
        .iter()
        .any(|chunk| !store.contains(chunk))
    {
        let mut seeded = false;
        for peer in &assignment.seed_peers {
            let source = HttpPeerSource::new(peer)?;
            if seed_from_peer(
                &source,
                &store,
                &assignment.manifest.model_id,
                &assignment.manifest.revision,
            )
            .await
            .is_ok()
            {
                seeded = true;
                break;
            }
        }
        ensure!(
            seeded,
            "model chunks are missing and no seed peer completed transfer"
        );
    }
    ModelRegistry::open(config.home.join("model-registry.db"))?.register(&assignment.manifest)?;
    let models = config.home.join("models");
    std::fs::create_dir_all(&models)?;
    let model = models.join(format!("{}.gguf", assignment.manifest.digest()?));
    if !model.exists() {
        store.materialize(&assignment.manifest, &model)?;
    }
    let device = mesh_gpu::discover_devices()
        .into_iter()
        .find(|device| device.stable_id == assignment.device_id)
        .context("assigned accelerator is no longer available")?;
    let mut outputs = Vec::with_capacity(assignment.prompts.len());
    for (prompt_index, prompt) in assignment.prompts {
        let runtime = config.runtime.clone();
        let model = model.clone();
        let device = device.clone();
        let max_tokens = assignment.max_tokens;
        let temperature_milli = assignment.temperature_milli;
        let seed = assignment.seed.wrapping_add(prompt_index as u64);
        let gpu_layers = config.gpu_layers;
        let output = tokio::task::spawn_blocking(move || {
            LlamaCppBackend::new(runtime, device, gpu_layers)?.infer(
                &model,
                &InferenceRequest {
                    prompt,
                    max_tokens,
                    temperature_milli,
                    seed,
                },
            )
        })
        .await??;
        outputs.push(PromptOutput {
            prompt_index,
            output_sha256: mesh_protocol::hash_bytes(output.text.as_bytes()),
            text: output.text,
            duration_ms: output.elapsed_ms,
        });
    }
    Ok(outputs)
}

async fn worker_loop(client: MeshClient, worker_id: usize, poll_ms: u64) -> Result<()> {
    let idle_delay = Duration::from_millis(poll_ms);
    loop {
        let poll = match client.poll().await {
            Ok(poll) => poll,
            Err(error) => {
                tracing::warn!(worker_id = worker_id, error = %error, "poll failed");
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
        };

        let Some(assignment) = poll.assignment else {
            tokio::time::sleep(idle_delay).await;
            continue;
        };

        tracing::info!(
            worker_id = worker_id,
            assignment_id = %assignment.assignment_id,
            job_id = %assignment.job_id,
            shard = assignment.shard_index,
            reward_mcu = assignment.reward_mcu,
            "executing contribution assignment"
        );

        let work = assignment.work.clone();
        let result = tokio::task::spawn_blocking(move || execute_work(&work)).await?;

        match client.report_result(&assignment, &result).await {
            Ok(settlement) => {
                tracing::info!(
                    worker_id = worker_id,
                    assignment_id = %assignment.assignment_id,
                    earned_cu = settlement.reward_mcu as f64 / 1000.0,
                    balance_cu = settlement.balance_mcu as f64 / 1000.0,
                    "verified contribution settled"
                );
            }
            Err(error) => {
                tracing::warn!(worker_id = worker_id, assignment_id = %assignment.assignment_id, error = %error, "result submission failed");
                // The coordinator lease will expire and requeue the work if the result was not accepted.
            }
        }
    }
}

/// Keep this node's position in latency space current.
///
/// The loop only ever *measures*: it asks the coordinator who is reachable,
/// times the round trip to a few of them, and folds what it saw into its own
/// coordinate. Nothing it is told is taken as a position - a peer's claim is
/// only ever an anchor to measure against, weighted by the confidence the peer
/// itself reports.
async fn proximity_loop(
    client: MeshClient,
    tracker: Arc<Mutex<Vivaldi>>,
    capabilities: Arc<RwLock<NodeCapabilities>>,
    home: PathBuf,
) {
    let http = reqwest::Client::new();
    // What we last measured to each peer, so the peer can fit itself against
    // us on the next exchange without spending a probe of its own.
    let mut measured: HashMap<String, u64> = HashMap::new();
    let mut interval = tokio::time::interval(PROXIMITY_INTERVAL);

    loop {
        interval.tick().await;
        if !proximity_round(
            &http,
            &client,
            &tracker,
            &capabilities,
            &home,
            &mut measured,
        )
        .await
        {
            return;
        }
    }
}

/// One measurement pass: sample peers, time a probe to each, fold what came
/// back into this node's own position.
///
/// Returns `false` only when a poisoned lock has made further measurement
/// meaningless. Every other failure - an unreachable peer, a coordinator that
/// cannot be asked - is ordinary and simply costs this pass.
async fn proximity_round(
    http: &reqwest::Client,
    client: &MeshClient,
    tracker: &Arc<Mutex<Vivaldi>>,
    capabilities: &Arc<RwLock<NodeCapabilities>>,
    home: &Path,
    measured: &mut HashMap<String, u64>,
) -> bool {
    let self_id = client.node_id();
    let peers = match client.peers().await {
        Ok(response) => response.peers,
        Err(error) => {
            tracing::debug!(%error, "peer sample unavailable");
            return true;
        }
    };

    let mut observed = 0usize;
    for peer in peers {
        if peer.node_id == self_id {
            continue;
        }
        let ours = tracker.lock().ok().map(|t| t.provisional_coordinate());
        let outcome = match probe_peer(
            http,
            &peer.probe_endpoint,
            ours,
            measured.get(&peer.node_id).copied(),
        )
        .await
        {
            Ok(outcome) => outcome,
            Err(error) => {
                // An unreachable peer is normal on a home network; it is
                // not an error worth waking an operator for.
                tracing::debug!(peer=%peer.node_id, %error, "probe failed");
                measured.remove(&peer.node_id);
                continue;
            }
        };
        measured.insert(peer.node_id.clone(), outcome.rtt_micros);
        let Some(remote) = outcome.coordinate else {
            continue;
        };
        if let Ok(mut t) = tracker.lock()
            && t.observe(&remote, outcome.rtt_micros)
        {
            observed += 1;
        }
    }

    if observed == 0 {
        return true;
    }

    let Ok(tracker_guard) = tracker.lock() else {
        tracing::error!("proximity lock poisoned; stopping measurement");
        return false;
    };
    let coordinate = tracker_guard.coordinate();
    let observations = tracker_guard.observations();
    if let Err(error) = tracker_guard.save(home) {
        tracing::warn!(%error, "could not persist network coordinate");
    }
    drop(tracker_guard);

    match capabilities.write() {
        Ok(mut caps) => caps.network_coordinate = coordinate,
        Err(_) => {
            tracing::error!("capability lock poisoned; stopping measurement");
            return false;
        }
    }
    tracing::debug!(observations, "network coordinate updated");
    true
}

#[cfg(test)]
mod proximity_tests {
    use super::*;
    use mesh_core::identity::NodeIdentity;
    use mesh_protocol::{NetworkCoordinate, PeerSample, PeerSampleResponse};
    use mesh_transport::{ProbeState, probe_router};

    /// A node arrives with no place in the network and has to earn one by
    /// measuring. This is that whole path: ask who is out there, time a real
    /// round trip to them, and end up with a coordinate worth advertising.
    #[tokio::test]
    async fn a_node_measures_its_way_onto_the_map() {
        let root = std::env::temp_dir().join(format!("mesh-prox-e2e-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        // A peer that already knows roughly where it sits and will serve probes.
        let mut anchor = Vivaldi::seeded(b"anchor");
        let far = NetworkCoordinate {
            vector_micros: [40_000, 0, 0],
            height_micros: 1_500,
            error_permille: 150,
        };
        for _ in 0..25 {
            anchor.observe(&far, 60_000);
        }
        let anchor_id = "anchor-node".to_string();
        let probe_state = ProbeState::new(anchor_id.clone(), Arc::new(Mutex::new(anchor)));
        let probe_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let probe_address = probe_listener.local_addr().unwrap();
        let probe_server = tokio::spawn(async move {
            axum::serve(probe_listener, probe_router(probe_state))
                .await
                .unwrap()
        });

        // A stand-in for the coordinator's directory lookup, which is the only
        // thing a measurement pass asks of it.
        let sample = PeerSampleResponse {
            peers: vec![PeerSample {
                node_id: anchor_id,
                probe_endpoint: format!("http://{probe_address}"),
                coordinate: None,
            }],
        };
        let directory = axum::Router::new().route(
            "/v1/network/peers",
            axum::routing::get(move || {
                let sample = sample.clone();
                async move { axum::Json(sample) }
            }),
        );
        let dir_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let dir_address = dir_listener.local_addr().unwrap();
        let dir_server =
            tokio::spawn(async move { axum::serve(dir_listener, directory).await.unwrap() });

        let home = root.join("node");
        let identity = NodeIdentity::load_or_create(&home).unwrap();
        let client = MeshClient::new(format!("http://{dir_address}"), identity);
        let tracker = Arc::new(Mutex::new(Vivaldi::load_or_seeded(&home, b"node")));
        let capabilities = Arc::new(RwLock::new(mesh_core::hardware::detect_capabilities(false)));
        assert!(
            capabilities.read().unwrap().network_coordinate.is_none(),
            "a node that has measured nothing must not claim a position"
        );

        let http = reqwest::Client::new();
        let mut measured = HashMap::new();
        for _ in 0..5 {
            assert!(
                proximity_round(
                    &http,
                    &client,
                    &tracker,
                    &capabilities,
                    &home,
                    &mut measured
                )
                .await,
                "an ordinary measurement pass must not stop the loop"
            );
        }

        let placed = capabilities.read().unwrap().network_coordinate;
        assert!(
            placed.is_some(),
            "measuring a reachable peer must put this node on the map"
        );
        assert_eq!(
            Vivaldi::load_or_seeded(&home, b"node").coordinate(),
            placed,
            "the position that was advertised is the one a restart resumes from"
        );
        assert!(
            measured.contains_key("anchor-node"),
            "the round trip we measured is what the peer fits itself against next time"
        );

        probe_server.abort();
        dir_server.abort();
        std::fs::remove_dir_all(&root).unwrap();
    }
}
