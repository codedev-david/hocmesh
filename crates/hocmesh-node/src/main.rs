mod client;
mod daemon;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use client::HocMeshClient;
use hocmesh_ai::{InferenceRequirements, PlanRequest, SubmitInferenceRequest};
use hocmesh_core::{hardware, identity::NodeIdentity, limits::ResourceLimits, proximity::Vivaldi};
use hocmesh_gpu::{InferenceBackend, InferenceRequest, LlamaCppBackend};
use hocmesh_ledger::{
    network::LedgerNetwork, store::LedgerStore, types::ValidatorSet,
    validate::validate_validator_set,
};
use hocmesh_model::{ChunkStore, ModelFormat, ModelRegistry, manifest_for_file};
use hocmesh_protocol::WorkSpec;
use hocmesh_transport::{HttpPeerSource, SeedServerState, seed_from_peer, seed_router};
use std::{collections::BTreeSet, fs, path::PathBuf, sync::Arc};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "hocmesh",
    version,
    about = "hocMESH — Mutual Exchange of Shared Hardware"
)]
struct Cli {
    #[arg(long, global = true, default_value = ".hocmesh")]
    home: PathBuf,
    #[arg(long, global = true, default_value = "http://127.0.0.1:8080")]
    coordinator: String,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Init,
    Daemon {
        #[arg(long)]
        workers: Option<usize>,
        #[arg(long, default_value_t = 750)]
        poll_ms: u64,
        #[arg(long)]
        ai_runtime: Option<PathBuf>,
        #[arg(long, default_value_t = 999)]
        gpu_layers: u32,
        #[arg(long)]
        model_seed_listen: Option<String>,
        #[arg(long)]
        model_seed_url: Option<String>,
        /// Serve latency probes here so other nodes can measure the path to
        /// this one. Requires a reachable address; measuring others does not.
        #[arg(long)]
        probe_listen: Option<String>,
    },
    /// Show where this node sits in the network's latency space.
    Proximity,
    /// Show or set the share of this machine lent to the hocmesh.
    Limits {
        #[arg(long)]
        cpu_percent: Option<u8>,
        #[arg(long)]
        gpu_percent: Option<u8>,
        #[arg(long)]
        memory_percent: Option<u8>,
    },
    Status,
    Balance,
    Benchmark,
    GpuInfo,
    ModelImport {
        path: PathBuf,
        #[arg(long)]
        model_id: String,
        #[arg(long, default_value = "main")]
        revision: String,
        #[arg(long)]
        format: String,
        #[arg(long)]
        architecture: String,
        #[arg(long, default_value_t = hocmesh_model::DEFAULT_CHUNK_SIZE)]
        chunk_size: usize,
    },
    ModelList,
    ModelPublish {
        #[arg(long)]
        model_id: String,
        #[arg(long, default_value = "main")]
        revision: String,
    },
    ModelSeed {
        #[arg(long)]
        peer: String,
        #[arg(long)]
        model_id: String,
        #[arg(long, default_value = "main")]
        revision: String,
    },
    ModelServe {
        #[arg(long, default_value = "127.0.0.1:8090")]
        listen: String,
    },
    Infer {
        #[arg(long)]
        model_id: String,
        #[arg(long, default_value = "main")]
        revision: String,
        #[arg(long)]
        runtime: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 128)]
        max_tokens: u32,
        #[arg(long, default_value_t = 0)]
        gpu_layers: u32,
    },
    AiPlan {
        #[arg(long)]
        model_id: String,
        #[arg(long, default_value = "main")]
        revision: String,
        #[arg(long, default_value = "cuda")]
        backend: String,
        #[arg(long, default_value_t = 1)]
        minimum_memory_mb: u64,
        #[arg(long, default_value_t = 1)]
        batch_size: u32,
        #[arg(long, default_value_t = 1)]
        pipeline_stages: u32,
        #[arg(long, default_value_t = 1)]
        tensor_parallelism: u32,
        #[arg(long)]
        layers: u32,
    },
    AiSubmit {
        #[arg(long)]
        model_id: String,
        #[arg(long, default_value = "main")]
        revision: String,
        #[arg(long, required = true)]
        prompt: Vec<String>,
        #[arg(long, default_value = "cuda")]
        backend: String,
        #[arg(long, default_value_t = 1)]
        minimum_memory_mb: u64,
        #[arg(long, default_value_t = 1)]
        pipeline_stages: u32,
        #[arg(long, default_value_t = 1)]
        tensor_parallelism: u32,
        #[arg(long)]
        layers: u32,
        #[arg(long, default_value_t = 128)]
        max_tokens: u32,
        #[arg(long, default_value_t = 0)]
        temperature_milli: u32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    AiJob {
        job_id: String,
    },
    SubmitPrime {
        #[arg(long)]
        start: u64,
        #[arg(long)]
        end: u64,
        #[arg(long, default_value_t = 8)]
        shards: u32,
    },
    Job {
        job_id: String,
    },
    Network,
    Id,
    /// Ask validators directly for a quorum-agreed head and this node's signed balance proof.
    LedgerStatus {
        #[arg(long)]
        validators: String,
    },
    /// Mirror the quorum-certified ledger into a local SQLite file and audit it from genesis.
    LedgerSync {
        #[arg(long)]
        validators: String,
        #[arg(long, default_value = ".hocmesh/ledger-mirror.db")]
        db: String,
        #[arg(long, default_value_t = 500)]
        batch: u64,
    },
    /// Offline audit of a previously mirrored ledger.
    LedgerAudit {
        #[arg(long)]
        validators: String,
        #[arg(long, default_value = ".hocmesh/ledger-mirror.db")]
        db: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    let cli = Cli::parse();
    let identity = NodeIdentity::load_or_create(&cli.home)?;
    let client = HocMeshClient::new(cli.coordinator, identity.clone());
    match cli.command {
        Command::Init => {
            let mut caps = hardware::detect_capabilities(true);
            // Register with the same share the daemon will advertise, so an
            // operator's limits hold from the very first contact.
            hardware::apply_limits(&mut caps, &ResourceLimits::load_or_default(&cli.home)?);
            let r = client.register(&caps).await?;
            print_registration(&r, &caps);
        }
        Command::Daemon {
            workers,
            poll_ms,
            ai_runtime,
            gpu_layers,
            model_seed_listen,
            model_seed_url,
            probe_listen,
        } => {
            let cached_model_manifests = model_registry(&cli.home)?
                .list()?
                .iter()
                .map(hocmesh_model::ModelManifest::digest)
                .collect::<Result<Vec<_>>>()?;
            let mut caps = hardware::detect_capabilities_with_models(
                true,
                model_seed_url,
                cached_model_manifests,
            );
            let limits = ResourceLimits::load_or_default(&cli.home)?;
            hardware::apply_limits(&mut caps, &limits);
            caps.ai_runtime_ready = ai_runtime.is_some() && !caps.gpus.is_empty();
            caps.probe_endpoint = probe_listen.as_ref().map(|listen| probe_url(listen));
            // --workers may lower the ceiling the operator set, never raise it.
            let workers = limits.clamp_requested_workers(workers, caps.logical_cpus);
            println!(
                "Sharing {} of {} logical CPUs (cpu {}%, gpu {}%, memory {}%)",
                workers,
                caps.logical_cpus,
                limits.cpu_percent,
                limits.gpu_percent,
                limits.memory_percent
            );
            let ai = ai_runtime.map(|runtime| daemon::AiWorkerConfig {
                home: cli.home.clone(),
                runtime,
                gpu_layers,
                seed_listen: model_seed_listen,
            });
            let proximity = daemon::ProximityConfig {
                home: cli.home.clone(),
                probe_listen,
            };
            daemon::run(client, caps, workers, poll_ms, ai, proximity).await?;
        }
        Command::Proximity => {
            let tracker = Vivaldi::load_or_seeded(&cli.home, client.node_id().as_bytes());
            match tracker.coordinate() {
                Some(coordinate) => println!(
                    "position: {:?} height {}us\nconfidence: {}%  (from {} measurements)\nstored at: {}",
                    coordinate.vector_micros,
                    coordinate.height_micros,
                    100 - (coordinate.error_permille / 10).min(100),
                    tracker.observations(),
                    Vivaldi::path(&cli.home).display()
                ),
                None => println!(
                    "Not placed yet: this node has measured nothing.\nRun `hocmesh daemon` and give it a minute, or check that the coordinator\nknows peers advertising --probe-listen."
                ),
            }
        }
        Command::Limits {
            cpu_percent,
            gpu_percent,
            memory_percent,
        } => {
            let mut limits = ResourceLimits::load_or_default(&cli.home)?;
            let changed =
                cpu_percent.is_some() || gpu_percent.is_some() || memory_percent.is_some();
            if let Some(v) = cpu_percent {
                limits.cpu_percent = v;
            }
            if let Some(v) = gpu_percent {
                limits.gpu_percent = v;
            }
            if let Some(v) = memory_percent {
                limits.memory_percent = v;
            }
            if changed {
                limits.save(&cli.home)?;
            }
            let cpus = hardware::detect_capabilities(false).logical_cpus;
            println!(
                "cpu {}%  gpu {}%  memory {}%
workers when running: {} of {} logical CPUs
stored at: {}",
                limits.cpu_percent,
                limits.gpu_percent,
                limits.memory_percent,
                limits.effective_workers(cpus),
                cpus,
                ResourceLimits::path(&cli.home).display()
            );
        }
        Command::Status => {
            let s = client.node_status().await?;
            println!(
                "Node: {}\nOnline: {}\nOS/arch: {}/{}\nCPU: {}\nLogical CPUs: {}\nRAM: {:.2} GiB",
                s.node_id,
                s.online,
                s.capabilities.os,
                s.capabilities.arch,
                s.capabilities.cpu_brand,
                s.capabilities.logical_cpus,
                s.capabilities.total_memory_bytes as f64 / 1024.0 / 1024.0 / 1024.0
            );
            for g in s.capabilities.gpus {
                println!(
                    "GPU: {} {} via {} ({:?} MiB)",
                    g.vendor, g.name, g.backend, g.memory_mb
                );
            }
        }
        Command::Balance => {
            let b = client.balance().await?;
            println!(
                "Node: {}\nBanked: {:.3} CU\nLifetime earned: {:.3} CU\nLifetime consumed: {:.3} CU",
                b.node_id,
                b.balance_mcu as f64 / 1000.0,
                b.earned_mcu as f64 / 1000.0,
                b.spent_mcu as f64 / 1000.0
            );
            if let (Some(h), Some(hash)) = (b.ledger_height, b.ledger_head) {
                println!("Ledger: height {h} head {hash}");
            }
        }
        Command::Benchmark => println!(
            "hocMESH CPU benchmark: {} candidate integers/second",
            hardware::benchmark_cpu()
        ),
        Command::GpuInfo => {
            let devices = hocmesh_gpu::discover_devices();
            if devices.is_empty() {
                println!("No CUDA, ROCm, or Metal devices detected.");
            }
            for device in devices {
                let report = hocmesh_gpu::benchmark_memory(&device, 32 * 1024 * 1024, 16);
                println!(
                    "{}: {} via {:?}, memory={:?} MiB, throughput={:.2} GiB/s, p95={:.3} ms",
                    device.stable_id,
                    device.name,
                    device.backend,
                    device.memory_bytes.map(|bytes| bytes / 1024 / 1024),
                    report.throughput_units_per_second / 1024.0 / 1024.0 / 1024.0,
                    report.latency_p95_ms
                );
            }
        }
        Command::ModelImport {
            path,
            model_id,
            revision,
            format,
            architecture,
            chunk_size,
        } => {
            let format = parse_model_format(&format)?;
            let store = model_store(&cli.home)?;
            let manifest = manifest_for_file(
                &store,
                &path,
                model_id,
                revision,
                format,
                architecture,
                chunk_size,
            )?;
            let first = store.read(&manifest.chunks[0].sha256)?;
            hocmesh_model::validate_format_header(format, &first)?;
            let registry = model_registry(&cli.home)?;
            let digest = registry.register(&manifest)?;
            println!(
                "Imported {}@{}: {} bytes, {} chunks, manifest {}",
                manifest.model_id,
                manifest.revision,
                manifest.total_size_bytes,
                manifest.chunks.len(),
                digest
            );
        }
        Command::ModelList => {
            for manifest in model_registry(&cli.home)?.list()? {
                println!(
                    "{}@{} {:?} {} bytes {}",
                    manifest.model_id,
                    manifest.revision,
                    manifest.format,
                    manifest.total_size_bytes,
                    manifest.digest()?
                );
            }
        }
        Command::ModelPublish { model_id, revision } => {
            client
                .register(&hardware::detect_capabilities(false))
                .await?;
            let manifest = model_registry(&cli.home)?
                .get(&model_id, &revision)?
                .context("model revision is not registered")?;
            let response = client.register_model(&manifest).await?;
            println!(
                "Published {}@{} as {}",
                response.model_id, response.revision, response.manifest_digest
            );
        }
        Command::ModelSeed {
            peer,
            model_id,
            revision,
        } => {
            let source = HttpPeerSource::new(peer)?;
            let store = model_store(&cli.home)?;
            let manifest = seed_from_peer(&source, &store, &model_id, &revision).await?;
            let digest = model_registry(&cli.home)?.register(&manifest)?;
            println!(
                "Seeded {}@{} with verified manifest {}",
                model_id, revision, digest
            );
        }
        Command::ModelServe { listen } => {
            let store = Arc::new(model_store(&cli.home)?);
            let manifests = model_registry(&cli.home)?.list()?;
            let state = SeedServerState::new(store, manifests)?;
            let listener = tokio::net::TcpListener::bind(&listen).await?;
            println!("Serving verified model chunks on http://{listen}");
            axum::serve(listener, seed_router(state)).await?;
        }
        Command::Infer {
            model_id,
            revision,
            runtime,
            prompt,
            max_tokens,
            gpu_layers,
        } => {
            let registry = model_registry(&cli.home)?;
            let manifest = registry
                .get(&model_id, &revision)?
                .context("model revision is not registered")?;
            anyhow::ensure!(
                manifest.format == ModelFormat::Gguf,
                "llama.cpp runtime currently requires GGUF"
            );
            let materialized = cli
                .home
                .join("models")
                .join(format!("{}.gguf", manifest.digest()?));
            if !materialized.exists() {
                fs::create_dir_all(materialized.parent().unwrap())?;
                model_store(&cli.home)?.materialize(&manifest, &materialized)?;
            }
            let device = hocmesh_gpu::discover_devices()
                .into_iter()
                .next()
                .unwrap_or(hocmesh_gpu::DeviceCapability {
                    stable_id: "cpu".into(),
                    backend: hocmesh_gpu::BackendKind::Cpu,
                    vendor: "cpu".into(),
                    name: "CPU".into(),
                    memory_bytes: Some(hardware::detect_capabilities(false).total_memory_bytes),
                    driver_version: None,
                    compute_version: None,
                    supports_fp16: false,
                    supports_bf16: false,
                    supports_int8: true,
                });
            let backend = LlamaCppBackend::new(runtime, device, gpu_layers)?;
            let output = backend.infer(
                &materialized,
                &InferenceRequest {
                    prompt,
                    max_tokens,
                    temperature_milli: 0,
                    seed: 0,
                },
            )?;
            print!("{}", output.text);
        }
        Command::AiPlan {
            model_id,
            revision,
            backend,
            minimum_memory_mb,
            batch_size,
            pipeline_stages,
            tensor_parallelism,
            layers,
        } => {
            client
                .register(&hardware::detect_capabilities(false))
                .await?;
            let request = PlanRequest {
                auth: hocmesh_protocol::AuthProof {
                    node_id: String::new(),
                    timestamp: 0,
                    nonce_b64: String::new(),
                    signature_b64: String::new(),
                },
                model_id,
                revision,
                requirements: InferenceRequirements {
                    required_backends: [parse_backend(&backend)?].into_iter().collect(),
                    minimum_memory_bytes: minimum_memory_mb * 1024 * 1024,
                    needs_fp16: false,
                    needs_bf16: false,
                    needs_int8: false,
                    batch_size,
                    pipeline_stages,
                    tensor_parallelism,
                },
                layer_count: layers,
                excluded_nodes: BTreeSet::new(),
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&client.plan_ai(request).await?)?
            );
        }
        Command::AiSubmit {
            model_id,
            revision,
            prompt,
            backend,
            minimum_memory_mb,
            pipeline_stages,
            tensor_parallelism,
            layers,
            max_tokens,
            temperature_milli,
            seed,
        } => {
            client
                .register(&hardware::detect_capabilities(false))
                .await?;
            let request = SubmitInferenceRequest {
                auth: hocmesh_protocol::AuthProof {
                    node_id: String::new(),
                    timestamp: 0,
                    nonce_b64: String::new(),
                    signature_b64: String::new(),
                },
                model_id,
                revision,
                requirements: InferenceRequirements {
                    required_backends: [parse_backend(&backend)?].into_iter().collect(),
                    minimum_memory_bytes: minimum_memory_mb * 1024 * 1024,
                    needs_fp16: false,
                    needs_bf16: false,
                    needs_int8: false,
                    batch_size: prompt.len() as u32,
                    pipeline_stages,
                    tensor_parallelism,
                },
                prompts: prompt,
                max_tokens,
                temperature_milli,
                seed,
                layer_count: layers,
            };
            let response = client.submit_inference(request).await?;
            println!(
                "Submitted AI job {} with {} assignments using manifest {}\n{}",
                response.job_id,
                response.assignments,
                response.manifest_digest,
                serde_json::to_string_pretty(&response.plan)?
            );
        }
        Command::AiJob { job_id } => {
            let status = client.inference_status(&job_id).await?;
            println!("{}", serde_json::to_string_pretty(&status)?);
        }
        Command::SubmitPrime { start, end, shards } => {
            let r = client
                .submit(WorkSpec::PrimeCount { start, end }, shards)
                .await?;
            println!(
                "Submitted job: {}\nParallel shards: {}\nReserved: {:.3} CU\nRemaining balance: {:.3} CU",
                r.job_id,
                r.assignments,
                r.reserved_mcu as f64 / 1000.0,
                r.balance_mcu as f64 / 1000.0
            );
            if let Some(h) = r.ledger_entry_hash {
                println!("Reservation ledger entry: {h}");
            }
        }
        Command::Job { job_id } => {
            let j = client.job_status(&job_id).await?;
            println!(
                "Job: {}\nStatus: {}\nSystem funded: {}\nProgress: {}/{} shards\nCompute: {:.3} CU",
                j.job_id,
                j.status,
                j.system_funded,
                j.completed_assignments,
                j.total_assignments,
                j.reserved_mcu as f64 / 1000.0
            );
            if let Some(t) = j.prime_count_total {
                println!("Current prime-count result: {t}");
            }
        }
        Command::Network => {
            let n = client.network_stats().await?;
            println!(
                "Ledger mode: {}\nRegistered nodes: {}\nOnline nodes: {}\nPending assignments: {}\nLeased assignments: {}\nCompleted assignments: {}\nCached banked community credit: {:.3} CU",
                n.ledger_mode,
                n.registered_nodes,
                n.online_nodes,
                n.pending_assignments,
                n.leased_assignments,
                n.completed_assignments,
                n.total_available_mcu as f64 / 1000.0
            );
        }
        Command::Id => {
            println!(
                "Node ID: {}\nPublic key: {}",
                identity.node_id(),
                identity.public_key_b64()
            );
        }
        Command::LedgerStatus { validators } => {
            let net = load_network(&validators)?;
            let h = net.head_quorum().await?;
            let b = net.balance_quorum(&identity.node_id()).await?;
            println!(
                "Quorum ledger height: {}\nQuorum ledger head: {}\nNode balance: {:.3} CU\nEarned: {:.3} CU\nSpent: {:.3} CU",
                h.sequence,
                h.entry_hash,
                b.balance_mcu as f64 / 1000.0,
                b.earned_mcu as f64 / 1000.0,
                b.spent_mcu as f64 / 1000.0
            );
        }
        Command::LedgerSync {
            validators,
            db,
            batch,
        } => {
            let net = load_network(&validators)?;
            let mut store = LedgerStore::open(&db)?;
            loop {
                let local = store.head(&net.set)?;
                let remote = net.head_quorum().await?;
                if local.sequence >= remote.sequence {
                    let audited = store.audit(&net.set)?;
                    println!(
                        "Ledger mirror synchronized and audited: height={} head={}",
                        audited.sequence, audited.entry_hash
                    );
                    break;
                }
                let certs = net
                    .fetch_certificates(local.sequence + 1, batch.max(1))
                    .await?;
                if certs.is_empty() {
                    anyhow::bail!("validators report a newer head but returned no certificates")
                };
                for cert in certs {
                    store.apply(&cert, &net.set)?;
                }
            }
        }
        Command::LedgerAudit { validators, db } => {
            let set = load_set(&validators)?;
            let store = LedgerStore::open(&db)?;
            let h = store.audit(&set)?;
            println!(
                "LEDGER AUDIT OK: height={} head={}",
                h.sequence, h.entry_hash
            );
        }
    }
    Ok(())
}

fn model_store(home: &std::path::Path) -> Result<ChunkStore> {
    ChunkStore::open(home.join("model-cache"))
}

fn model_registry(home: &std::path::Path) -> Result<ModelRegistry> {
    fs::create_dir_all(home)?;
    ModelRegistry::open(home.join("model-registry.db"))
}

fn parse_model_format(value: &str) -> Result<ModelFormat> {
    match value.to_ascii_lowercase().as_str() {
        "gguf" => Ok(ModelFormat::Gguf),
        "safetensors" => Ok(ModelFormat::Safetensors),
        _ => anyhow::bail!("format must be gguf or safetensors"),
    }
}

fn parse_backend(value: &str) -> Result<hocmesh_gpu::BackendKind> {
    match value.to_ascii_lowercase().as_str() {
        "cuda" => Ok(hocmesh_gpu::BackendKind::Cuda),
        "rocm" | "hip" => Ok(hocmesh_gpu::BackendKind::Rocm),
        "metal" => Ok(hocmesh_gpu::BackendKind::Metal),
        "cpu" => Ok(hocmesh_gpu::BackendKind::Cpu),
        _ => anyhow::bail!("backend must be cuda, rocm, metal, or cpu"),
    }
}

fn load_set(path: &str) -> Result<ValidatorSet> {
    let set: ValidatorSet =
        serde_json::from_str(&fs::read_to_string(path).with_context(|| format!("reading {path}"))?)
            .context("parsing validator set")?;
    validate_validator_set(&set)?;
    Ok(set)
}
fn load_network(path: &str) -> Result<LedgerNetwork> {
    LedgerNetwork::new(load_set(path)?)
}
fn print_registration(
    r: &hocmesh_protocol::RegisterResponse,
    c: &hocmesh_protocol::NodeCapabilities,
) {
    println!(
        "hocMESH — Mutual Exchange of Shared Hardware\nNode ID: {}\nProtocol: v{}\nLedger mode: {}\nCPU: {}\nLogical CPUs: {}\nCPU benchmark: {} candidates/sec\nRAM: {:.2} GiB\nDetected GPUs: {}\nStarting balance: {:.3} CU",
        r.node_id,
        r.protocol_version,
        r.ledger_mode,
        c.cpu_brand,
        c.logical_cpus,
        c.cpu_benchmark_score,
        c.total_memory_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
        c.gpus.len(),
        r.balance_mcu as f64 / 1000.0
    );
    if r.balance_mcu == 0 {
        println!(
            "Contribution-first rule active: run `hocmesh daemon` and complete community work before submitting jobs."
        );
    }
}

/// Turn a listen address into the URL peers should probe.
///
/// A wildcard bind says where to *listen*, not where to be *found*, so it is
/// rewritten to loopback rather than advertised as-is: `0.0.0.0` reaches
/// nothing from another machine, and publishing it would put an endpoint in
/// the directory that every peer wastes a probe timeout on. An operator with a
/// real public address should pass it explicitly.
fn probe_url(listen: &str) -> String {
    let address = listen.trim();
    if let Some(port) = address.strip_prefix("0.0.0.0:") {
        return format!("http://127.0.0.1:{port}");
    }
    if let Some(port) = address.strip_prefix("[::]:") {
        return format!("http://127.0.0.1:{port}");
    }
    if address.starts_with("http://") || address.starts_with("https://") {
        return address.to_string();
    }
    format!("http://{address}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `--probe-listen` is a *bind* address; what peers need is somewhere to
    /// send to. A wildcard bind is not an address anyone can dial, so it has
    /// to be turned into one before it is advertised.
    #[test]
    fn a_wildcard_bind_is_advertised_as_a_dialable_address() {
        assert_eq!(probe_url("0.0.0.0:8646"), "http://127.0.0.1:8646");
        assert_eq!(probe_url("[::]:8646"), "http://127.0.0.1:8646");
    }

    /// Anything already specific is left alone, scheme and all.
    #[test]
    fn an_address_that_is_already_reachable_is_left_alone() {
        assert_eq!(probe_url("127.0.0.1:8646"), "http://127.0.0.1:8646");
        assert_eq!(
            probe_url("hocmesh.example.org:8646"),
            "http://hocmesh.example.org:8646"
        );
        assert_eq!(
            probe_url("http://hocmesh.example.org"),
            "http://hocmesh.example.org"
        );
        assert_eq!(
            probe_url("https://hocmesh.example.org"),
            "https://hocmesh.example.org"
        );
    }

    /// An operator's shell will hand us whitespace sooner or later, and a URL
    /// with a stray space in it fails at the far end, where it is unhelpful.
    #[test]
    fn surrounding_whitespace_never_reaches_the_advertised_url() {
        assert_eq!(probe_url("  0.0.0.0:8646\n"), "http://127.0.0.1:8646");
        assert_eq!(probe_url(" example.org:1 "), "http://example.org:1");
    }

    /// Operators type these by hand, so case is not a signal.
    #[test]
    fn model_formats_and_backends_are_matched_case_insensitively() {
        assert!(matches!(
            parse_model_format("GGUF").unwrap(),
            ModelFormat::Gguf
        ));
        assert!(matches!(
            parse_model_format("SafeTensors").unwrap(),
            ModelFormat::Safetensors
        ));
        assert!(matches!(
            parse_backend("CUDA").unwrap(),
            hocmesh_gpu::BackendKind::Cuda
        ));
        // `hip` is what AMD's own tooling calls it; both names mean ROCm.
        for name in ["rocm", "hip"] {
            assert!(matches!(
                parse_backend(name).unwrap(),
                hocmesh_gpu::BackendKind::Rocm
            ));
        }
    }

    /// A typo has to be a refusal, not a silent fallback to some default that
    /// then fails much later with an error about the wrong thing.
    #[test]
    fn an_unknown_format_or_backend_is_refused_with_the_choices() {
        let error = parse_model_format("onnx").unwrap_err().to_string();
        assert!(error.contains("gguf"), "{error}");
        assert!(error.contains("safetensors"), "{error}");

        let error = parse_backend("opencl").unwrap_err().to_string();
        for choice in ["cuda", "rocm", "metal", "cpu"] {
            assert!(error.contains(choice), "{error} should mention {choice}");
        }
    }

    /// The whole CLI is the operator's only interface. clap can only check
    /// this at runtime, so it has to be checked somewhere.
    #[test]
    fn the_command_line_is_well_formed() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
