use crate::client::HocMeshClient;
use anyhow::{Context, Result, ensure};
use hocmesh_ai::{InferenceAssignment, PromptOutput};
use hocmesh_core::compute::execute_work;
use hocmesh_core::proximity::Vivaldi;
use hocmesh_gpu::{InferenceBackend, InferenceRequest, LlamaCppBackend};
use hocmesh_model::{ChunkStore, ModelRegistry};
use hocmesh_protocol::NodeCapabilities;
use hocmesh_transport::{
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

/// Run the node until the operator interrupts it.
pub async fn run(
    client: HocMeshClient,
    capabilities: NodeCapabilities,
    workers: usize,
    poll_ms: u64,
    ai: Option<AiWorkerConfig>,
    proximity: ProximityConfig,
) -> Result<()> {
    run_until(
        client,
        capabilities,
        workers,
        poll_ms,
        ai,
        proximity,
        async {
            let _ = tokio::signal::ctrl_c().await;
        },
    )
    .await
}

/// The daemon itself, stopping when `shutdown` completes.
///
/// Production waits on ctrl-c. Taking the signal as an argument is what makes
/// the teardown below observable: a test can end the run on demand and then
/// check that the servers this spawned are really gone.
async fn run_until<S: std::future::Future<Output = ()>>(
    client: HocMeshClient,
    capabilities: NodeCapabilities,
    workers: usize,
    poll_ms: u64,
    ai: Option<AiWorkerConfig>,
    proximity: ProximityConfig,
    shutdown: S,
) -> Result<()> {
    let registered = client.register(&capabilities).await?;
    println!("hocMESH node {} registered", registered.node_id);
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
        PROXIMITY_INTERVAL,
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
        _ = shutdown => {
            println!("Shutdown requested; stopping hocMESH node.");
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

async fn ai_worker_loop(client: HocMeshClient, config: AiWorkerConfig, poll_ms: u64) -> Result<()> {
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
    let device = hocmesh_gpu::discover_devices()
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
            output_sha256: hocmesh_protocol::hash_bytes(output.text.as_bytes()),
            text: output.text,
            duration_ms: output.elapsed_ms,
        });
    }
    Ok(outputs)
}

async fn worker_loop(client: HocMeshClient, worker_id: usize, poll_ms: u64) -> Result<()> {
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
///
/// `period` is `PROXIMITY_INTERVAL` in production; taking it as an argument
/// lets a test watch the loop tick instead of taking the schedule on trust.
async fn proximity_loop(
    client: HocMeshClient,
    tracker: Arc<Mutex<Vivaldi>>,
    capabilities: Arc<RwLock<NodeCapabilities>>,
    home: PathBuf,
    period: Duration,
) {
    let http = reqwest::Client::new();
    // What we last measured to each peer, so the peer can fit itself against
    // us on the next exchange without spending a probe of its own.
    let mut measured: HashMap<String, u64> = HashMap::new();
    let mut interval = tokio::time::interval(period);

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
    client: &HocMeshClient,
    tracker: &Arc<Mutex<Vivaldi>>,
    capabilities: &Arc<RwLock<NodeCapabilities>>,
    home: &Path,
    measured: &mut HashMap<String, u64>,
) -> bool {
    // A poisoned position can never be read or written again, so there is
    // nothing left for this pass to measure into. Checking once, here, keeps
    // that failure loud instead of letting it look like a quiet empty round.
    if tracker.is_poisoned() {
        tracing::error!("proximity lock poisoned; stopping measurement");
        return false;
    }
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
    use hocmesh_core::identity::NodeIdentity;
    use hocmesh_protocol::{NetworkCoordinate, PeerSample, PeerSampleResponse, ProbeResponse};
    use hocmesh_transport::{ProbeState, probe_router};
    use std::net::SocketAddr;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::task::JoinHandle;

    static SCRATCH: AtomicUsize = AtomicUsize::new(0);

    /// A scratch home no other test shares.
    fn scratch(tag: &str) -> PathBuf {
        let slot = SCRATCH.fetch_add(1, Ordering::SeqCst);
        let root =
            std::env::temp_dir().join(format!("hocmesh-{tag}-{}-{slot}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    /// A peer that has already fitted itself, so probing it teaches us
    /// something instead of two unplaced nodes staring at each other.
    fn fitted_anchor() -> Vivaldi {
        let mut anchor = Vivaldi::seeded(b"anchor");
        let far = NetworkCoordinate {
            vector_micros: [40_000, 0, 0],
            height_micros: 1_500,
            error_permille: 150,
        };
        for _ in 0..25 {
            anchor.observe(&far, 60_000);
        }
        anchor
    }

    /// Serve the real probe endpoint on an ephemeral port.
    async fn serve_probes(node_id: &str, position: Vivaldi) -> (SocketAddr, JoinHandle<()>) {
        let state = ProbeState::new(node_id.to_string(), Arc::new(Mutex::new(position)));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server =
            tokio::spawn(async move { axum::serve(listener, probe_router(state)).await.unwrap() });
        (address, server)
    }

    /// A probe endpoint that answers without offering a position, and counts
    /// its callers - so a test can show who was, and was not, asked.
    async fn counting_probes(node_id: &str) -> (SocketAddr, Arc<AtomicUsize>, JoinHandle<()>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let id = node_id.to_string();
        let router = axum::Router::new().route(
            "/v1/proximity/probe",
            axum::routing::post(move |_: axum::Json<serde_json::Value>| {
                let counter = counter.clone();
                let id = id.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    axum::Json(ProbeResponse {
                        node_id: id,
                        coordinate: None,
                    })
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        (address, calls, server)
    }

    /// A stand-in for the coordinator's directory lookup, the only thing a
    /// measurement pass asks of it. Counting the calls is what tells a single
    /// pass apart from a loop that keeps running.
    async fn serve_directory(peers: Vec<PeerSample>) -> (String, Arc<AtomicUsize>, JoinHandle<()>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let router = axum::Router::new().route(
            "/v1/network/peers",
            axum::routing::get(move || {
                let peers = peers.clone();
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    axum::Json(PeerSampleResponse { peers })
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        (format!("http://{address}"), calls, server)
    }

    /// The measuring side: a fresh identity with no position and nothing yet
    /// worth advertising, pointed at `directory`.
    fn measuring_node(
        home: &Path,
        directory: String,
    ) -> (
        HocMeshClient,
        Arc<Mutex<Vivaldi>>,
        Arc<RwLock<NodeCapabilities>>,
    ) {
        let identity = NodeIdentity::load_or_create(home).unwrap();
        let client = HocMeshClient::new(directory, identity);
        let tracker = Arc::new(Mutex::new(Vivaldi::load_or_seeded(home, b"node")));
        let mut caps = hocmesh_core::hardware::detect_capabilities(false);
        caps.network_coordinate = None;
        (client, tracker, Arc::new(RwLock::new(caps)))
    }

    /// One measurement pass, with the plumbing every test shares.
    async fn round(
        client: &HocMeshClient,
        tracker: &Arc<Mutex<Vivaldi>>,
        capabilities: &Arc<RwLock<NodeCapabilities>>,
        home: &Path,
        measured: &mut HashMap<String, u64>,
    ) -> bool {
        proximity_round(
            &reqwest::Client::new(),
            client,
            tracker,
            capabilities,
            home,
            measured,
        )
        .await
    }

    /// Leave a lock in the state a panicking holder would leave it in.
    fn poison<T: Send + 'static>(lock: &Arc<Mutex<T>>) {
        let held = lock.clone();
        let _ = std::thread::spawn(move || {
            let _guard = held.lock().unwrap();
            panic!("poisoned on purpose");
        })
        .join();
        assert!(lock.is_poisoned());
    }

    /// The same, for the capability snapshot the daemon advertises from.
    fn poison_rw<T: Send + Sync + 'static>(lock: &Arc<RwLock<T>>) {
        let held = lock.clone();
        let _ = std::thread::spawn(move || {
            let _guard = held.write().unwrap();
            panic!("poisoned on purpose");
        })
        .join();
        assert!(lock.is_poisoned());
    }

    /// A node arrives with no place in the network and has to earn one by
    /// measuring. This is that whole path: ask who is out there, time a real
    /// round trip to them, and end up with a coordinate worth advertising.
    #[tokio::test]
    async fn a_node_measures_its_way_onto_the_map() {
        let root = scratch("prox-e2e");
        let (probe_address, probe_server) = serve_probes("anchor-node", fitted_anchor()).await;
        let (directory, lookups, dir_server) = serve_directory(vec![PeerSample {
            node_id: "anchor-node".to_string(),
            probe_endpoint: format!("http://{probe_address}"),
            coordinate: None,
        }])
        .await;
        let home = root.join("node");
        let (client, tracker, capabilities) = measuring_node(&home, directory);

        let mut measured = HashMap::new();
        for _ in 0..5 {
            assert!(
                round(&client, &tracker, &capabilities, &home, &mut measured).await,
                "an ordinary measurement pass must not stop the loop"
            );
        }
        assert_eq!(lookups.load(Ordering::SeqCst), 5);

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

    /// The coordinator is not the authority on where a node sits, so losing it
    /// costs this pass and nothing more. The loop has to survive to try again.
    #[tokio::test]
    async fn a_coordinator_that_cannot_be_asked_costs_only_this_pass() {
        let root = scratch("prox-nodir");
        let dead = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = dead.local_addr().unwrap();
        drop(dead);
        let home = root.join("node");
        let (client, tracker, capabilities) = measuring_node(&home, format!("http://{address}"));

        let mut measured = HashMap::new();
        measured.insert("someone".to_string(), 9_000);
        assert!(
            round(&client, &tracker, &capabilities, &home, &mut measured).await,
            "a coordinator that is down must not end the measurement loop"
        );
        assert!(
            capabilities.read().unwrap().network_coordinate.is_none(),
            "a pass that measured nothing must not advertise a position"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Probing yourself would fold your own position back into itself and read
    /// as confirmation. The sample can name us, so the pass has to skip us.
    #[tokio::test]
    async fn a_node_never_probes_itself() {
        let root = scratch("prox-self");
        let home = root.join("node");
        let self_id = NodeIdentity::load_or_create(&home).unwrap().node_id();
        let (probe_address, probes, probe_server) = counting_probes(&self_id).await;
        let (directory, _, dir_server) = serve_directory(vec![PeerSample {
            node_id: self_id.clone(),
            probe_endpoint: format!("http://{probe_address}"),
            coordinate: None,
        }])
        .await;
        let (client, tracker, capabilities) = measuring_node(&home, directory);
        assert_eq!(client.node_id(), self_id);

        let mut measured = HashMap::new();
        assert!(round(&client, &tracker, &capabilities, &home, &mut measured).await);
        assert_eq!(
            probes.load(Ordering::SeqCst),
            0,
            "a node must never spend a probe on itself"
        );
        assert!(measured.is_empty());
        assert!(capabilities.read().unwrap().network_coordinate.is_none());

        probe_server.abort();
        dir_server.abort();
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// What we last measured to a peer is reported back to it, so it can fit
    /// itself without spending a probe. If the peer stops answering, that
    /// number has to go: a remembered round trip is not evidence of a live one.
    #[tokio::test]
    async fn a_peer_that_stops_answering_is_forgotten_not_guessed() {
        let root = scratch("prox-gone");
        let dead = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = dead.local_addr().unwrap();
        drop(dead);
        let (directory, _, dir_server) = serve_directory(vec![PeerSample {
            node_id: "gone".to_string(),
            probe_endpoint: format!("http://{address}"),
            coordinate: None,
        }])
        .await;
        let home = root.join("node");
        let (client, tracker, capabilities) = measuring_node(&home, directory);

        let mut measured = HashMap::from([("gone".to_string(), 12_345u64)]);
        assert!(
            round(&client, &tracker, &capabilities, &home, &mut measured).await,
            "an unreachable peer costs the pass, not the loop"
        );
        assert!(
            !measured.contains_key("gone"),
            "a failed probe must retract the round trip, not keep reporting it"
        );
        assert!(
            capabilities.read().unwrap().network_coordinate.is_none(),
            "a peer we could not reach cannot place us"
        );

        dir_server.abort();
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A peer may answer a probe without offering a position - it may not have
    /// one yet. That is a usable round trip and an unusable anchor, and the
    /// pass has to take the first without inventing the second.
    #[tokio::test]
    async fn a_peer_with_no_position_yet_is_timed_but_not_fitted_against() {
        let root = scratch("prox-nopos");
        let (probe_address, probes, probe_server) = counting_probes("shy").await;
        let (directory, _, dir_server) = serve_directory(vec![PeerSample {
            node_id: "shy".to_string(),
            probe_endpoint: format!("http://{probe_address}"),
            coordinate: None,
        }])
        .await;
        let home = root.join("node");
        let (client, tracker, capabilities) = measuring_node(&home, directory);

        let mut measured = HashMap::new();
        assert!(round(&client, &tracker, &capabilities, &home, &mut measured).await);
        assert!(
            probes.load(Ordering::SeqCst) > 0,
            "the peer was reachable, so it must have been probed"
        );
        assert!(
            measured.contains_key("shy"),
            "the round trip stands on its own, position or not"
        );
        assert!(
            capabilities.read().unwrap().network_coordinate.is_none(),
            "a peer that gave no position cannot move us"
        );
        assert_eq!(tracker.lock().unwrap().observations(), 0);

        probe_server.abort();
        dir_server.abort();
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A poisoned position can never be read or written again, so measuring
    /// into it is pointless work that would look like a quiet empty round.
    /// The loop has to stop, loudly, rather than spin.
    #[tokio::test]
    async fn a_poisoned_position_stops_the_measurement() {
        let root = scratch("prox-poison");
        let (probe_address, probe_server) = serve_probes("anchor-node", fitted_anchor()).await;
        let (directory, lookups, dir_server) = serve_directory(vec![PeerSample {
            node_id: "anchor-node".to_string(),
            probe_endpoint: format!("http://{probe_address}"),
            coordinate: None,
        }])
        .await;
        let home = root.join("node");
        let (client, tracker, capabilities) = measuring_node(&home, directory);
        poison(&tracker);

        let mut measured = HashMap::new();
        assert!(
            !round(&client, &tracker, &capabilities, &home, &mut measured).await,
            "there is nothing left to measure into, so the loop must end"
        );
        assert_eq!(
            lookups.load(Ordering::SeqCst),
            0,
            "it must give up before spending a request on anyone else"
        );

        probe_server.abort();
        dir_server.abort();
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// If the snapshot the daemon advertises from is poisoned, the measurement
    /// still happened and is still saved - what is lost is the ability to
    /// publish it, and that is what ends the loop.
    #[tokio::test]
    async fn a_poisoned_capability_snapshot_stops_the_measurement() {
        let root = scratch("prox-caps");
        let (probe_address, probe_server) = serve_probes("anchor-node", fitted_anchor()).await;
        let (directory, _, dir_server) = serve_directory(vec![PeerSample {
            node_id: "anchor-node".to_string(),
            probe_endpoint: format!("http://{probe_address}"),
            coordinate: None,
        }])
        .await;
        let home = root.join("node");
        let (client, tracker, capabilities) = measuring_node(&home, directory);
        poison_rw(&capabilities);

        let mut measured = HashMap::new();
        assert!(
            !round(&client, &tracker, &capabilities, &home, &mut measured).await,
            "a position nobody can publish is not worth measuring for"
        );
        assert!(
            Vivaldi::load_or_seeded(&home, b"node").observations() > 0,
            "the round trip was real, so it must survive on disk for the restart"
        );

        probe_server.abort();
        dir_server.abort();
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// One pass is not a position. The loop has to keep re-measuring on its
    /// own schedule, because latency moves and a coordinate fitted once and
    /// never revisited is only true of the network it was measured on.
    #[tokio::test]
    async fn the_measurement_loop_keeps_measuring() {
        let root = scratch("prox-loop");
        let (probe_address, probe_server) = serve_probes("anchor-node", fitted_anchor()).await;
        let (directory, lookups, dir_server) = serve_directory(vec![PeerSample {
            node_id: "anchor-node".to_string(),
            probe_endpoint: format!("http://{probe_address}"),
            coordinate: None,
        }])
        .await;
        let home = root.join("node");
        let (client, tracker, capabilities) = measuring_node(&home, directory);

        let task = tokio::spawn(proximity_loop(
            client,
            tracker,
            capabilities.clone(),
            home.clone(),
            Duration::from_millis(10),
        ));
        for _ in 0..300 {
            if lookups.load(Ordering::SeqCst) >= 3 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        task.abort();

        assert!(
            lookups.load(Ordering::SeqCst) >= 3,
            "the loop measured {} times; it is meant to keep going, not run once",
            lookups.load(Ordering::SeqCst)
        );
        assert!(
            capabilities.read().unwrap().network_coordinate.is_some(),
            "a loop that ran against a reachable anchor must have placed us"
        );

        probe_server.abort();
        dir_server.abort();
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Serving probes is something the operator opted into for as long as the
    /// daemon runs. When the daemon stops, that port has to stop with it -
    /// a listener outliving its node is a promise nothing is left to keep.
    #[tokio::test]
    async fn shutting_down_takes_the_probe_server_with_it() {
        let root = scratch("prox-shutdown");
        // The coordinator only has to get the node as far as running.
        let coordinator = axum::Router::new().route(
            "/v1/nodes/register",
            axum::routing::post(|_: axum::Json<serde_json::Value>| async {
                axum::Json(hocmesh_protocol::RegisterResponse {
                    node_id: "node-under-test".to_string(),
                    balance_mcu: 0,
                    protocol_version: hocmesh_protocol::PROTOCOL_VERSION,
                    ledger_mode: "coordinator".to_string(),
                })
            }),
        );
        let dir_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let dir_address = dir_listener.local_addr().unwrap();
        let dir_server =
            tokio::spawn(async move { axum::serve(dir_listener, coordinator).await.unwrap() });

        // Pick a free port for the probe server the way an operator would.
        let reserved = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let probe_address = reserved.local_addr().unwrap();
        drop(reserved);

        let home = root.join("node");
        let identity = NodeIdentity::load_or_create(&home).unwrap();
        let client = HocMeshClient::new(format!("http://{dir_address}"), identity);
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let node = tokio::spawn(run_until(
            client,
            hocmesh_core::hardware::detect_capabilities(false),
            1,
            1_000,
            None,
            ProximityConfig {
                home: home.clone(),
                probe_listen: Some(probe_address.to_string()),
            },
            async move {
                let _ = stopped.await;
            },
        ));

        assert!(
            reachable(probe_address, true).await,
            "the daemon opted into serving probes, so the port must be open"
        );

        stop.send(()).unwrap();
        node.await
            .expect("the daemon must not panic on the way out")
            .expect("a clean shutdown is not an error");
        assert!(
            reachable(probe_address, false).await,
            "the probe server must not outlive the node that promised to answer"
        );

        dir_server.abort();
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Wait, briefly, for a port to start or stop accepting connections.
    async fn reachable(address: SocketAddr, want: bool) -> bool {
        for _ in 0..300 {
            if tokio::net::TcpStream::connect(address).await.is_ok() == want {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        false
    }
}
