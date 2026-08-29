use crate::client::HocMeshClient;
use crate::control::{ControlSeed, ControlServer, DaemonMetrics, note_exchange};
use anyhow::{Context, Result, ensure};
use hocmesh_ai::{InferenceAssignment, PromptOutput};
use hocmesh_core::bandwidth::UplinkMeter;
use hocmesh_core::compute::{declared_memory_bytes, execute_work};
use hocmesh_core::proximity::Vivaldi;
use hocmesh_core::resources::{Claim, Lent, ResourcePool};
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
use tokio::{sync::Notify, task::JoinSet};

/// What the daemon needs in order to offer a control surface.
///
/// Optional as a whole, because a daemon started from a script on a headless
/// box has nobody to control it and no reason to open a port. The desktop app
/// always asks for one.
pub struct ControlConfig {
    pub home: PathBuf,
    /// `0` asks the OS for a free port, which is the default. The real one is
    /// published in the endpoint file, so nothing has to agree in advance.
    pub port: u16,
    /// The machine as detected, before limits. Held so that a limit raised
    /// through the UI can give back what lowering it took away.
    pub detected: Arc<NodeCapabilities>,
    pub runtime_available: bool,
}

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

/// Everything a run is configured with, decided before it starts.
///
/// One struct rather than a long parameter list because these are read
/// together and only ever set together -- the CLI builds exactly one of these,
/// and a test that wants a daemon with no AI and no control surface says so by
/// leaving two fields `None` rather than by counting positions.
pub struct DaemonConfig {
    pub capabilities: NodeCapabilities,
    pub workers: usize,
    pub poll_ms: u64,
    pub ai: Option<AiWorkerConfig>,
    pub proximity: ProximityConfig,
    pub control: Option<ControlConfig>,
}

/// Run the node until the operator interrupts it.
pub async fn run(client: HocMeshClient, config: DaemonConfig) -> Result<()> {
    run_until(client, config, async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await
}

/// The daemon itself, stopping when `shutdown` completes.
///
/// Production waits on ctrl-c. Taking the signal as an argument is what makes
/// the teardown below observable: a test can end the run on demand and then
/// check that the servers this spawned are really gone.
async fn run_until<S: std::future::Future<Output = ()>>(
    client: HocMeshClient,
    config: DaemonConfig,
    shutdown: S,
) -> Result<()> {
    let DaemonConfig {
        capabilities,
        workers,
        poll_ms,
        ai,
        proximity,
        control,
    } = config;
    let registered = client.register(&capabilities).await?;
    println!("hocMESH node {} registered", registered.node_id);
    println!(
        "Available balance: {:.3} CU",
        registered.balance_mcu as f64 / 1000.0
    );
    println!("Contribution workers: {}", workers);

    // Built from the capabilities as advertised, so the budget enforced here
    // and the budget published to the coordinator are the same numbers. A pool
    // derived from the detected hardware instead would let the node accept work
    // up to what the machine can do rather than up to what its operator lent.
    let pool = ResourcePool::new(Lent::from_capabilities(&capabilities));
    let lent = pool.lent();
    println!(
        "Lending {} worker(s), {} MiB of memory and {} MiB of GPU memory",
        lent.logical_cpus,
        lent.memory_bytes >> 20,
        lent.device_memory_bytes >> 20
    );

    let capabilities = Arc::new(RwLock::new(capabilities));
    let metrics = DaemonMetrics::new();
    // Registration succeeded, so the coordinator was reachable a moment ago.
    // Starting from "unknown" instead would show a disconnected light on a
    // daemon that had just proved otherwise.
    metrics.record_contact();

    // Created here rather than inside the seed server because the heartbeat
    // starts first and has to advertise whatever the serving has measured so
    // far -- which, for a node that has not served yet, is nothing.
    let uplink = Arc::new(UplinkMeter::new());

    let heartbeat_client = client.clone();
    let heartbeat_caps = capabilities.clone();
    let heartbeat_metrics = metrics.clone();
    let heartbeat_uplink = Arc::clone(&uplink);
    let heartbeat_pool = pool.clone();
    let heartbeat = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        loop {
            interval.tick().await;
            // Re-read every tick rather than capturing once: this is what makes
            // a limits change through the control surface reach the coordinator
            // without a restart. The uplink is folded in on the same tick, so
            // the coordinator and the local dashboard read one figure rather
            // than two that drift apart.
            let measured = heartbeat_uplink.kbps().unwrap_or(0);
            // Placement divides by this. It was a constant zero, so the term
            // that was meant to steer work away from a busy node steered
            // nothing, and the busiest machine in the mesh looked as inviting
            // as an idle one.
            let load = heartbeat_pool.load_permille();
            let snapshot = {
                let Ok(mut caps) = heartbeat_caps.write() else {
                    tracing::error!("capability lock poisoned; stopping heartbeat");
                    return;
                };
                if caps.model_bandwidth_kbps != measured {
                    caps.model_bandwidth_kbps = measured;
                }
                if caps.load_permille != load {
                    caps.load_permille = load;
                }
                caps.clone()
            };
            let outcome = heartbeat_client.heartbeat(&snapshot).await;
            note_exchange(&heartbeat_metrics, &outcome);
            if let Err(error) = outcome {
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

    // Bound before any worker starts, so a desktop app that launched this
    // process can attach the moment the endpoint file appears rather than
    // polling for one that may never come.
    let control_shutdown = Arc::new(Notify::new());
    let mut control_home = None;
    let mut control_server = None;
    if let Some(config) = control {
        let seed = ControlSeed {
            home: config.home.clone(),
            node_id: client.node_id(),
            coordinator: client.coordinator().to_string(),
            workers: workers.max(1),
            version: env!("CARGO_PKG_VERSION").to_string(),
            capabilities: capabilities.clone(),
            detected: config.detected,
            runtime_available: config.runtime_available,
            metrics: metrics.clone(),
            shutdown: control_shutdown.clone(),
        };
        let (server, _state) = ControlServer::bind(seed, config.port).await?;
        println!(
            "Control surface on 127.0.0.1:{} (token in {})",
            server.endpoint.port,
            crate::control::ControlEndpoint::path(&config.home).display()
        );
        control_home = Some(config.home);
        control_server = Some(tokio::spawn(async move { server.serve().await }));
    }

    let mut workers_set = JoinSet::new();
    for worker_id in 0..workers.max(1) {
        let worker_client = client.clone();
        let worker_metrics = metrics.clone();
        let worker_pool = pool.clone();
        workers_set.spawn(async move {
            worker_loop(
                worker_client,
                worker_id,
                poll_ms.max(100),
                worker_metrics,
                worker_pool,
            )
            .await
        });
    }

    let mut seed_server = None;
    if let Some(config) = ai.as_ref()
        && let Some(listen) = config.seed_listen.as_ref()
    {
        let store = Arc::new(ChunkStore::open(config.home.join("model-cache"))?);
        let manifests = ModelRegistry::open(config.home.join("model-registry.db"))?.list()?;
        let state = SeedServerState::measuring(store, manifests, Arc::clone(&uplink))?;
        let listener = tokio::net::TcpListener::bind(listen).await?;
        seed_server = Some(tokio::spawn(async move {
            axum::serve(listener, seed_router(state)).await
        }));
    }
    if let Some(config) = ai {
        let ai_client = client.clone();
        let ai_metrics = metrics.clone();
        let ai_pool = pool.clone();
        workers_set.spawn(async move {
            ai_worker_loop(ai_client, config, poll_ms.max(100), ai_metrics, ai_pool).await
        });
    }

    tokio::select! {
        _ = shutdown => {
            println!("Shutdown requested; stopping hocMESH node.");
        }
        _ = control_shutdown.notified() => {
            println!("Shutdown requested through the control surface; stopping hocMESH node.");
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
    if let Some(server) = control_server {
        server.abort();
    }
    // Withdraw the advertisement last and unconditionally. A file left behind
    // tells the next dashboard that a daemon which has just stopped is running,
    // and leaves a spent token on disk.
    if let Some(home) = control_home {
        ControlServer::retire(&home);
    }
    workers_set.abort_all();
    Ok(())
}

async fn ai_worker_loop(
    client: HocMeshClient,
    config: AiWorkerConfig,
    poll_ms: u64,
    metrics: Arc<DaemonMetrics>,
    pool: ResourcePool,
) -> Result<()> {
    let idle_delay = Duration::from_millis(poll_ms);
    loop {
        let outcome = client.poll_inference().await;
        note_exchange(&metrics, &outcome);
        let poll = match outcome {
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
        let job_id = assignment.job_id.clone();
        // The provider prices its own batch. It signs the amount, so it is the
        // one party that must not be taking somebody else's word for it.
        let Some((batch_start, batch_end, reward_mcu)) = hocmesh_ai::assignment_claim(&assignment)
        else {
            tracing::warn!(%assignment_id, "assignment cannot be priced; skipping");
            continue;
        };
        match execute_inference_assignment(&config, assignment, pool.clone()).await {
            Ok(outputs) => match client
                .report_inference(
                    assignment_id.clone(),
                    job_id,
                    batch_start,
                    batch_end,
                    reward_mcu,
                    outputs,
                )
                .await
            {
                Ok(response) => {
                    metrics.record_inference();
                    tracing::info!(%assignment_id, job_completed=response.job_completed, "AI assignment completed")
                }
                Err(error) => {
                    metrics.record_failure(&error);
                    tracing::warn!(%assignment_id, %error, "AI result submission failed")
                }
            },
            Err(error) => {
                metrics.record_failure(&error);
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
    pool: ResourcePool,
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
    // Resolve rather than search. A node that lent its CPU advertised a device
    // no discovery probe returns, and searching `discover_devices` directly is
    // what made a CPU-only node accept inference and then fail every batch.
    let device = hocmesh_gpu::resolve_device(&assignment.device_id)
        .context("assigned accelerator is no longer available")?;

    // The manifest states the model's size exactly, which is the whole reason
    // this is the right place to ask: llama.cpp maps the file, so the weights
    // are the working set and no estimate is involved. The claim is taken
    // before the first prompt and held across all of them, because the model
    // stays resident between them -- claiming per prompt would count the same
    // bytes once and then release them while they were still occupied.
    //
    // Layers pushed onto the GPU come out of the device budget. Layers left on
    // the host stay in the host budget, which is why the claim splits rather
    // than counting the file twice or, worse, once.
    let weights = assignment.manifest.total_size_bytes;
    let gpu_layers = hocmesh_gpu::gpu_layers_for(&device, config.gpu_layers);
    let offloaded = offloaded_bytes(weights, gpu_layers, assignment.manifest.layer_count());
    let lease = {
        let pool = pool.clone();
        // The host claim is the whole file whether or not layers are offloaded:
        // llama.cpp maps it, so those bytes are addressable on the host either
        // way. The device claim is the part that is additionally resident on
        // the card.
        let claim = Claim::host(1, weights).with_device(offloaded);
        tokio::task::spawn_blocking(move || pool.claim(claim))
            .await?
            .map_err(|too_large| {
                anyhow::anyhow!("this model does not fit inside this node's limits: {too_large}")
            })?
    };
    // Other people's work gets the share the operator lent, not the machine --
    // and specifically the part of that share nobody is currently using.
    //
    // Sizing this from the whole lent share instead would double-count: the
    // contribution workers hold permits for the cores they are on, and handing
    // llama.cpp the full share on top of that spends the operator's four cores
    // twice. Sizing it from what is free keeps the two paths adding up to what
    // was lent, and it adapts -- an idle node gives inference everything, a busy
    // one gives it what is spare.
    //
    // Floored at one because zero threads is not a smaller share, it is a
    // process that does not run.
    let threads = inference_threads(pool.lent().logical_cpus, pool.usage().logical_cpus);

    let mut outputs = Vec::with_capacity(assignment.prompts.len());
    for (prompt_index, prompt) in assignment.prompts {
        let runtime = config.runtime.clone();
        let model = model.clone();
        let device = device.clone();
        let max_tokens = assignment.max_tokens;
        let temperature_milli = assignment.temperature_milli;
        let seed = assignment.seed.wrapping_add(prompt_index as u64);
        let output = tokio::task::spawn_blocking(move || {
            LlamaCppBackend::new(runtime, device, gpu_layers)?
                .with_threads(threads)
                .infer(
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
    drop(lease);
    Ok(outputs)
}

/// Threads to hand llama.cpp for one assignment.
///
/// The share the operator lent, less what the contribution workers are holding
/// right now. Handing over the whole lent share regardless would double-count
/// it: those workers hold permits for the cores they are running on, so a node
/// lending four cores would spend them twice and the operator would get eight
/// cores' worth of load from a four-core promise.
///
/// `used` includes this assignment's own permit, which *is* the inference
/// process, so it is discounted rather than charged -- without that, an idle
/// node lending four cores would run llama.cpp on three.
///
/// Floored at one, because zero threads is not a smaller share; it is a process
/// that does not run.
fn inference_threads(lent: usize, used: usize) -> usize {
    lent.saturating_sub(used.saturating_sub(1)).max(1)
}

/// How much of a model's weights `gpu_layers` puts on the device.
///
/// Proportional to the layer count, because layers are the granularity
/// llama.cpp offloads at.
///
/// A limit stated rather than papered over: no importer records
/// [`hocmesh_model::LAYER_COUNT`] yet, so in practice this returns zero today
/// and the device budget is under-counted for offloaded work. The alternative
/// -- charging the whole file to the card whenever any offload is asked for --
/// would refuse a 30 GiB model on a 24 GiB card that was only ever going to
/// hold ten of its sixty layers, and [`TooLarge`] refusals are permanent. An
/// under-count admits work that should have been queued; an over-count refuses
/// work that would have run. Between a budget that is slightly too generous and
/// one that permanently rejects valid work, the generous one is recoverable.
///
/// [`TooLarge`]: hocmesh_core::resources::TooLarge
fn offloaded_bytes(weights: u64, gpu_layers: u32, layers: Option<u32>) -> u64 {
    if gpu_layers == 0 {
        return 0;
    }
    let Some(layers) = layers.filter(|layers| *layers > 0) else {
        return 0;
    };
    if gpu_layers >= layers {
        return weights;
    }
    (u128::from(weights) * u128::from(gpu_layers) / u128::from(layers)) as u64
}

async fn worker_loop(
    client: HocMeshClient,
    worker_id: usize,
    poll_ms: u64,
    metrics: Arc<DaemonMetrics>,
    pool: ResourcePool,
) -> Result<()> {
    let idle_delay = Duration::from_millis(poll_ms);
    loop {
        let outcome = client.poll().await;
        // Every worker reports the same coordinator, so any one of them
        // succeeding is enough to call the daemon connected -- which is what a
        // dashboard is asking about.
        note_exchange(&metrics, &outcome);
        let poll = match outcome {
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
        // Claimed before the work starts and released when the lease drops at
        // the end of this iteration. One worker is one lent core, which the
        // worker count already bounds; the memory is what that bound never
        // covered -- N workers each holding a shard's working set is N times a
        // number nothing was checking against what the operator lent.
        let claim = Claim::host(1, declared_memory_bytes(&work));
        let lease = {
            let pool = pool.clone();
            match tokio::task::spawn_blocking(move || pool.claim(claim)).await? {
                Ok(lease) => lease,
                Err(too_large) => {
                    // Not a transient shortage: this shard will never fit here,
                    // so waiting for capacity would be waiting forever. There
                    // is no channel to decline a contribution assignment, so it
                    // is left to expire and be reassigned -- the same path a
                    // node that stops answering already takes. Logged loudly
                    // because an operator whose limits are too tight for the
                    // work on offer needs to be told, not left to wonder why
                    // this node never earns anything.
                    tracing::error!(
                        worker_id = worker_id,
                        assignment_id = %assignment.assignment_id,
                        reason = %too_large,
                        "assignment does not fit inside this node's limits; \
                         leaving it to be reassigned"
                    );
                    // Waiting before asking again, because the coordinator
                    // still has this assignment leased to this node and will
                    // hand back the same one until it expires. Without the
                    // pause that is a hot loop against the coordinator that
                    // writes an error line per turn.
                    tokio::time::sleep(idle_delay).await;
                    continue;
                }
            }
        };
        let result = tokio::task::spawn_blocking(move || execute_work(&work)).await?;
        drop(lease);

        match client.report_result(&assignment, &result).await {
            Ok(settlement) => {
                // Counted on settlement rather than on execution. Work the
                // coordinator did not accept earned nothing, and a dashboard
                // that counted it would drift from the ledger.
                metrics.record_completion(settlement.reward_mcu);
                tracing::info!(
                    worker_id = worker_id,
                    assignment_id = %assignment.assignment_id,
                    earned_cu = settlement.reward_mcu as f64 / 1000.0,
                    balance_cu = settlement.balance_mcu as f64 / 1000.0,
                    "verified contribution settled"
                );
            }
            Err(error) => {
                metrics.record_failure(&error);
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
            DaemonConfig {
                capabilities: hocmesh_core::hardware::detect_capabilities(false),
                workers: 1,
                poll_ms: 1_000,
                ai: None,
                proximity: ProximityConfig {
                    home: home.clone(),
                    probe_listen: Some(probe_address.to_string()),
                },
                control: None,
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

    #[test]
    fn an_idle_node_gives_inference_every_core_it_lent() {
        // `used` is 1: the inference assignment's own permit and nothing else.
        assert_eq!(inference_threads(4, 1), 4);
    }

    #[test]
    fn a_busy_node_gives_inference_what_is_left_rather_than_the_whole_share() {
        // Two contribution workers running, plus this assignment.
        assert_eq!(
            inference_threads(4, 3),
            2,
            "llama.cpp was handed the full lent share while other work held \
             cores, so a four-core promise produced six cores of load"
        );
    }

    #[test]
    fn a_fully_committed_node_still_runs_the_work_it_accepted() {
        // Zero threads is not a smaller share, it is a process that does not
        // run -- and the assignment has already been accepted by this point.
        assert_eq!(inference_threads(4, 9), 1);
        assert_eq!(inference_threads(0, 0), 1);
    }

    #[test]
    fn offloading_half_the_layers_charges_half_the_weights_to_the_card() {
        assert_eq!(offloaded_bytes(1000, 30, Some(60)), 500);
    }

    #[test]
    fn offloading_every_layer_charges_the_whole_model_to_the_card() {
        assert_eq!(offloaded_bytes(1000, 60, Some(60)), 1000);
        // llama.cpp accepts a number past the end and offloads everything.
        assert_eq!(offloaded_bytes(1000, 999, Some(60)), 1000);
    }

    #[test]
    fn a_cpu_assignment_charges_nothing_to_a_card() {
        assert_eq!(offloaded_bytes(1000, 0, Some(60)), 0);
    }

    #[test]
    fn an_unknown_layer_count_is_not_guessed_at() {
        // Charging the whole file to the card here is what would refuse a model
        // that was only ever going to put ten of its sixty layers there -- and
        // that refusal is permanent, where an under-count is not.
        assert_eq!(offloaded_bytes(1000, 20, None), 0);
        assert_eq!(offloaded_bytes(1000, 20, Some(0)), 0);
    }
}
