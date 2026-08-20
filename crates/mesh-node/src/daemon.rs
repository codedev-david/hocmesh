use crate::client::MeshClient;
use anyhow::{Context, Result, ensure};
use mesh_ai::{InferenceAssignment, PromptOutput};
use mesh_core::compute::execute_work;
use mesh_gpu::{InferenceBackend, InferenceRequest, LlamaCppBackend};
use mesh_model::{ChunkStore, ModelRegistry};
use mesh_protocol::NodeCapabilities;
use mesh_transport::{HttpPeerSource, SeedServerState, seed_from_peer, seed_router};
use std::{path::PathBuf, sync::Arc, time::Duration};
use tokio::task::JoinSet;

pub struct AiWorkerConfig {
    pub home: PathBuf,
    pub runtime: PathBuf,
    pub gpu_layers: u32,
    pub seed_listen: Option<String>,
}

pub async fn run(
    client: MeshClient,
    capabilities: NodeCapabilities,
    workers: usize,
    poll_ms: u64,
    ai: Option<AiWorkerConfig>,
) -> Result<()> {
    let registered = client.register(&capabilities).await?;
    println!("MESH node {} registered", registered.node_id);
    println!(
        "Available balance: {:.3} CU",
        registered.balance_mcu as f64 / 1000.0
    );
    println!("Contribution workers: {}", workers);

    let capabilities = Arc::new(capabilities);
    let heartbeat_client = client.clone();
    let heartbeat_caps = capabilities.clone();
    let heartbeat = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        loop {
            interval.tick().await;
            if let Err(error) = heartbeat_client.heartbeat(&heartbeat_caps).await {
                tracing::warn!(%error, "heartbeat failed");
            }
        }
    });

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
