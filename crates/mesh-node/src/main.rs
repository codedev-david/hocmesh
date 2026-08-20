mod client;
mod daemon;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use client::MeshClient;
use mesh_ai::{InferenceRequirements, PlanRequest, SubmitInferenceRequest};
use mesh_core::{hardware, identity::NodeIdentity};
use mesh_gpu::{InferenceBackend, InferenceRequest, LlamaCppBackend};
use mesh_ledger::{
    network::LedgerNetwork, store::LedgerStore, types::ValidatorSet,
    validate::validate_validator_set,
};
use mesh_model::{ChunkStore, ModelFormat, ModelRegistry, manifest_for_file};
use mesh_protocol::WorkSpec;
use mesh_transport::{HttpPeerSource, SeedServerState, seed_from_peer, seed_router};
use std::{collections::BTreeSet, fs, path::PathBuf, sync::Arc};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "mesh",
    version,
    about = "MESH — Mutual Exchange of Shared Hardware"
)]
struct Cli {
    #[arg(long, global = true, default_value = ".mesh")]
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
        #[arg(long, default_value_t = mesh_model::DEFAULT_CHUNK_SIZE)]
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
        #[arg(long, default_value = ".mesh/ledger-mirror.db")]
        db: String,
        #[arg(long, default_value_t = 500)]
        batch: u64,
    },
    /// Offline audit of a previously mirrored ledger.
    LedgerAudit {
        #[arg(long)]
        validators: String,
        #[arg(long, default_value = ".mesh/ledger-mirror.db")]
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
    let client = MeshClient::new(cli.coordinator, identity.clone());
    match cli.command {
        Command::Init => {
            let caps = hardware::detect_capabilities(true);
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
        } => {
            let cached_model_manifests = model_registry(&cli.home)?
                .list()?
                .iter()
                .map(mesh_model::ModelManifest::digest)
                .collect::<Result<Vec<_>>>()?;
            let mut caps = hardware::detect_capabilities_with_models(
                true,
                model_seed_url,
                cached_model_manifests,
            );
            caps.ai_runtime_ready = ai_runtime.is_some() && !caps.gpus.is_empty();
            let workers = workers.unwrap_or_else(|| caps.logical_cpus.saturating_sub(1).max(1));
            let ai = ai_runtime.map(|runtime| daemon::AiWorkerConfig {
                home: cli.home.clone(),
                runtime,
                gpu_layers,
                seed_listen: model_seed_listen,
            });
            daemon::run(client, caps, workers, poll_ms, ai).await?;
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
            "MESH CPU benchmark: {} candidate integers/second",
            hardware::benchmark_cpu()
        ),
        Command::GpuInfo => {
            let devices = mesh_gpu::discover_devices();
            if devices.is_empty() {
                println!("No CUDA, ROCm, or Metal devices detected.");
            }
            for device in devices {
                let report = mesh_gpu::benchmark_memory(&device, 32 * 1024 * 1024, 16);
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
            mesh_model::validate_format_header(format, &first)?;
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
            let device = mesh_gpu::discover_devices().into_iter().next().unwrap_or(
                mesh_gpu::DeviceCapability {
                    stable_id: "cpu".into(),
                    backend: mesh_gpu::BackendKind::Cpu,
                    vendor: "cpu".into(),
                    name: "CPU".into(),
                    memory_bytes: Some(hardware::detect_capabilities(false).total_memory_bytes),
                    driver_version: None,
                    compute_version: None,
                    supports_fp16: false,
                    supports_bf16: false,
                    supports_int8: true,
                },
            );
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
                auth: mesh_protocol::AuthProof {
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
                auth: mesh_protocol::AuthProof {
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

fn parse_backend(value: &str) -> Result<mesh_gpu::BackendKind> {
    match value.to_ascii_lowercase().as_str() {
        "cuda" => Ok(mesh_gpu::BackendKind::Cuda),
        "rocm" | "hip" => Ok(mesh_gpu::BackendKind::Rocm),
        "metal" => Ok(mesh_gpu::BackendKind::Metal),
        "cpu" => Ok(mesh_gpu::BackendKind::Cpu),
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
fn print_registration(r: &mesh_protocol::RegisterResponse, c: &mesh_protocol::NodeCapabilities) {
    println!(
        "MESH — Mutual Exchange of Shared Hardware\nNode ID: {}\nProtocol: v{}\nLedger mode: {}\nCPU: {}\nLogical CPUs: {}\nCPU benchmark: {} candidates/sec\nRAM: {:.2} GiB\nDetected GPUs: {}\nStarting balance: {:.3} CU",
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
            "Contribution-first rule active: run `mesh daemon` and complete community work before submitting jobs."
        );
    }
}
