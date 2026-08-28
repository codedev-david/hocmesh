use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use hocmesh::client::HocMeshClient;
use hocmesh::loadtest::{LoadPlan, Workload};
use hocmesh::{control, daemon, install, loadtest};
use hocmesh_ai::{InferenceRequirements, PlanRequest, SubmitInferenceRequest};
use hocmesh_core::compute::{split_work, work_cost_mcu};
use hocmesh_core::{
    hardware,
    identity::{
        self, IDENTITY_EXPORT_PASSPHRASE_ENV, IDENTITY_PASSPHRASE_ENV, NodeIdentity, identity_path,
    },
    limits::ResourceLimits,
    proximity::Vivaldi,
};
use hocmesh_gpu::{InferenceBackend, InferenceRequest, LlamaCppBackend};
use hocmesh_ledger::{
    network::LedgerNetwork,
    store::{LedgerSnapshot, LedgerStore},
    types::{
        LedgerTransaction, MembershipAction, MembershipChangeEvidence, TransactionEvidence,
        TransactionKind, ValidatorMember, ValidatorSet, ValidatorSignature,
    },
    validate::{
        community_reserve_signing_message, membership_hash, membership_result,
        validate_validator_set, verify_membership_change, vouch_signing_message,
    },
};
use hocmesh_model::{ChunkStore, ModelFormat, ModelRegistry, manifest_for_file};
use hocmesh_protocol::WorkSpec;
use hocmesh_transport::{HttpPeerSource, SeedServerState, seed_from_peer, seed_router};
use std::{
    collections::BTreeSet,
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
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
        /// Path to a llama.cpp executable. Defaults to the runtime installed
        /// by `hocmesh runtime-install`; without either, this node advertises
        /// no AI capability and simply never claims inference work.
        #[arg(long)]
        ai_runtime: Option<PathBuf>,
        /// Do not offer AI work even when a runtime is installed.
        #[arg(long)]
        no_ai: bool,
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
        /// Port for the local control surface the desktop app and `hocmesh
        /// stop` talk to. Always loopback-only. `0` takes a free port, which
        /// is what lets two homes run on one machine.
        #[arg(long, default_value_t = 0)]
        control_port: u16,
        /// Run without a control surface. Nothing local can then change this
        /// node's limits or stop it politely -- only a signal will.
        #[arg(long)]
        no_control: bool,
    },
    /// Ask a running daemon on this machine to stop.
    Stop,
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
        /// Whether to run other people's inference on this machine.
        #[arg(long, value_enum)]
        ai: Option<AiSharing>,
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
    /// Read a GGUF file's tensor directory and report what each pipeline stage
    /// would have to hold.
    ///
    /// Nothing is imported and nothing is run: this reads the header and does
    /// arithmetic, so it works on a file a peer has only partly fetched.
    ModelInspect {
        path: PathBuf,
        /// How many pipeline stages to divide the transformer blocks between.
        #[arg(long, default_value_t = 1)]
        stages: u32,
        #[arg(long, default_value_t = hocmesh_model::DEFAULT_CHUNK_SIZE)]
        chunk_size: usize,
    },
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
    /// Download a model, verify its digest, and import it into the chunk store.
    ///
    /// Give a catalogue id (`hocmesh model-catalog` lists them), or point at any
    /// Hugging Face repository with --repository, or at a file with --url. The
    /// bytes are checked against a SHA-256 before anything is imported.
    ModelPull {
        /// A catalogue id. Omit when using --repository or --url.
        id: Option<String>,
        /// Any Hugging Face repository holding GGUF weights, as owner/name.
        #[arg(long, conflicts_with_all = ["id", "url"])]
        repository: Option<String>,
        /// A direct URL to a .gguf file. Requires --sha256.
        #[arg(long, conflicts_with_all = ["id", "repository"])]
        url: Option<String>,
        /// Which quantisation to prefer, e.g. q4_k_m. Ignored with --url.
        #[arg(long)]
        quantisation: Option<String>,
        /// Branch, tag or commit in the repository.
        #[arg(long)]
        revision: Option<String>,
        /// Pin the expected digest yourself. Overrides the repository's, and is
        /// required with --url.
        #[arg(long)]
        sha256: Option<String>,
        /// Register under this id instead of the derived one.
        #[arg(long)]
        model_id: Option<String>,
        /// Override the architecture; by default it is read from the GGUF header.
        #[arg(long)]
        architecture: Option<String>,
        #[arg(long, default_value_t = hocmesh_model::DEFAULT_CHUNK_SIZE)]
        chunk_size: usize,
        /// Keep the downloaded file. Off by default: the chunk store already
        /// holds the bytes, so it would be a second copy.
        #[arg(long)]
        keep_download: bool,
    },
    /// List the models `model-pull` knows by name.
    ModelCatalog,
    /// Download the pinned llama.cpp build for this machine.
    ///
    /// The archive must match a SHA-256 compiled into this binary before any of
    /// it is unpacked. No flag can widen what is acceptable.
    RuntimeInstall {
        /// Reinstall even if a runtime is already present.
        #[arg(long)]
        force: bool,
        /// Keep the downloaded archive after unpacking it.
        #[arg(long)]
        keep_download: bool,
    },
    /// Show which inference runtime this node would use, and what it expects.
    RuntimeStatus,
    Infer {
        #[arg(long)]
        model_id: String,
        #[arg(long, default_value = "main")]
        revision: String,
        /// Path to a llama.cpp executable. Defaults to the runtime installed
        /// by `hocmesh runtime-install`.
        #[arg(long)]
        runtime: Option<PathBuf>,
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
    /// Take back the escrow on batches of your AI job that nobody delivered.
    AiReclaim {
        job_id: String,
    },
    /// Take delivery of a finished batch: pay its escrow into holding and get
    /// the text back. Nothing else on the network will hand it to you.
    AiReceipt {
        job_id: String,
        assignment_id: String,
    },
    /// Say what a delivered batch was worth. Accepting pays the provider;
    /// disputing returns the same CU to the commons, so neither answer is
    /// cheaper than the other.
    AiSettle {
        job_id: String,
        assignment_id: String,
        #[arg(long)]
        dispute: bool,
        #[arg(long, default_value = "")]
        reason: String,
    },
    /// Search a range of starting values for the longest Collatz trajectory.
    SubmitCollatz {
        #[arg(long)]
        start: u64,
        #[arg(long)]
        end: u64,
        #[arg(long, default_value_t = 8)]
        shards: u32,
    },
    SubmitPrime {
        #[arg(long)]
        start: u64,
        #[arg(long)]
        end: u64,
        #[arg(long, default_value_t = 8)]
        shards: u32,
    },
    /// Put artificial load on a coordinator, then check the economy survived it.
    ///
    /// Reports latency and throughput like any load test, and then does the
    /// part that makes it worth shipping: it re-adds the CU. Exit status is
    /// about whether the work settled and the numbers agree, never about how
    /// fast the machine happened to be.
    Loadtest {
        /// Jobs to submit. `0` runs until `--duration-secs` instead.
        #[arg(long, default_value_t = 20)]
        jobs: u64,
        /// Jobs in flight at once. This is the contention knob.
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
        /// Shards per job.
        #[arg(long, default_value_t = 4)]
        shards: u32,
        #[arg(long, value_enum, default_value_t = Workload::Collatz)]
        workload: Workload,
        /// Range width per job, or matrix dimension for `--workload matrix`.
        #[arg(long, default_value_t = 200_000)]
        size: u64,
        /// Stop submitting after this long, whatever `--jobs` says.
        #[arg(long)]
        duration_secs: Option<u64>,
        /// How long one job may take before it counts as a timeout.
        #[arg(long, default_value_t = 120)]
        timeout_secs: u64,
        #[arg(long, default_value_t = 250)]
        poll_ms: u64,
        /// Also write the whole report here as JSON, for a pipeline to keep.
        #[arg(long)]
        json: Option<PathBuf>,
        /// Print what this run would cost and submit nothing.
        ///
        /// The price is deterministic and knowable up front, so a script can
        /// wait until the account can afford the run instead of finding out
        /// halfway through that it is broke -- which would look exactly like
        /// the settlement failure this command exists to detect.
        #[arg(long)]
        dry_run: bool,
    },
    /// Back up, move, or look at the keypair this account *is*.
    ///
    /// Nothing about an account is tied to the machine it was made on. The
    /// balance follows the key, so a new laptop is a copied key and not a
    /// support request -- there is nobody to ask, because nobody ever held it.
    Identity {
        #[command(subcommand)]
        action: IdentityAction,
    },
    /// Take back the escrow on shards of your job that nobody ever delivered.
    Reclaim {
        job_id: String,
    },
    Job {
        job_id: String,
    },
    Network,
    /// Show what the coordinator and the ledger still disagree about.
    ///
    /// Intents the coordinator persisted but has not managed to settle, plus
    /// the work it parked waiting on funding that nothing is chasing any more.
    Reconciliation,
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
    ///
    /// Starts from the newest stored checkpoint when there is one, so the cost
    /// tracks how much has happened since rather than the whole history.
    LedgerAudit {
        #[arg(long)]
        validators: String,
        #[arg(long, default_value = ".hocmesh/ledger-mirror.db")]
        db: String,
        /// Ignore any checkpoint and replay every entry from genesis.
        #[arg(long)]
        full: bool,
    },
    /// Ask the validators for a quorum-signed statement of the whole ledger
    /// state and record it locally as an audit starting point.
    LedgerCheckpoint {
        #[arg(long)]
        validators: String,
        #[arg(long, default_value = ".hocmesh/ledger-mirror.db")]
        db: String,
    },
    /// Drop certificates the newest stored checkpoint already vouches for.
    LedgerPrune {
        #[arg(long)]
        validators: String,
        #[arg(long, default_value = ".hocmesh/ledger-mirror.db")]
        db: String,
    },
    /// Write the newest stored checkpoint, and the state it vouches for, to a
    /// file another operator can start a node from.
    LedgerSnapshot {
        #[arg(long)]
        validators: String,
        #[arg(long, default_value = ".hocmesh/ledger-mirror.db")]
        db: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Start an empty mirror from a snapshot rather than from genesis.
    LedgerRestore {
        #[arg(long)]
        validators: String,
        #[arg(long, default_value = ".hocmesh/ledger-mirror.db")]
        db: String,
        #[arg(long)]
        snapshot: PathBuf,
    },
    /// Page through one account's postings, newest first.
    ///
    /// Reads the local mirror by default. `--validators` asks the network
    /// instead, which is what an operator without a mirror -- or with one
    /// pruned below the entry they are chasing -- actually has to do.
    LedgerHistory {
        #[arg(long)]
        account: String,
        #[arg(long, default_value = ".hocmesh/ledger-mirror.db")]
        db: String,
        #[arg(long)]
        validators: Option<String>,
        #[arg(long)]
        before: Option<u64>,
        #[arg(long, default_value_t = 50)]
        limit: u32,
    },
    /// Sign a sponsorship for a community-funded job.
    ///
    /// The mint is the one place CU comes from nothing, so it is an operator
    /// action for the same reason admission is: a validator that signed
    /// whatever it was handed would make the shared budget free to spend.
    CommunityVouch {
        #[arg(long)]
        validators: String,
        #[arg(long)]
        job_id: String,
        #[arg(long, default_value_t = 2)]
        start: u64,
        #[arg(long, default_value_t = 5_000_000)]
        end: u64,
        #[arg(long, default_value_t = 32)]
        shards: u32,
    },
    /// Sign a sponsorship for a change to the validator set.
    ///
    /// Deliberately an operator action rather than an endpoint. A validator
    /// that vouched for whoever asked would make admission free, which is the
    /// whole thing the vouch exists to stop.
    MembershipVouch {
        #[arg(long)]
        validators: String,
        #[arg(long, value_enum)]
        action: MembershipActionArg,
        /// JSON file describing the validator joining or leaving.
        #[arg(long)]
        member: String,
        /// Consensus threshold the set should carry afterwards.
        #[arg(long)]
        threshold: usize,
    },
    /// Submit a set change once enough sitting validators have sponsored it.
    MembershipCommit {
        #[arg(long)]
        validators: String,
        #[arg(long, value_enum)]
        action: MembershipActionArg,
        #[arg(long)]
        member: String,
        #[arg(long)]
        threshold: usize,
        /// JSON file holding the collected vouch signatures.
        #[arg(long)]
        vouches: String,
        /// Where to write the set the change produces.
        #[arg(long)]
        out: Option<String>,
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
    // Taken out before `load_or_create`, on purpose: these are the commands you
    // reach for when the key on this machine is the thing in question, and
    // minting one as a side effect of asking about it is the wrong answer.
    let command = match cli.command {
        Command::Identity { action } => return run_identity(&cli.home, &action),
        other => other,
    };
    let identity = NodeIdentity::load_or_create(&cli.home)?;
    let client = HocMeshClient::new(cli.coordinator, identity.clone());
    match command {
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
            no_ai,
            gpu_layers,
            model_seed_listen,
            model_seed_url,
            probe_listen,
            control_port,
            no_control,
        } => {
            let cached_model_manifests = model_registry(&cli.home)?
                .list()?
                .iter()
                .map(hocmesh_model::ModelManifest::digest)
                .collect::<Result<Vec<_>>>()?;
            let mut detected = hardware::detect_capabilities_with_models(
                true,
                model_seed_url,
                cached_model_manifests,
            );
            let limits = ResourceLimits::load_or_default(&cli.home)?;
            // An installed runtime is used without having to be named, so
            // `runtime-install` followed by `daemon` is enough. --ai-runtime
            // still overrides it; --no-ai declines the work entirely.
            let ai_runtime = if no_ai {
                None
            } else {
                ai_runtime.or_else(|| hocmesh_gpu::runtime::installed_runtime(&cli.home))
            };
            let runtime_available = ai_runtime.is_some();
            detected.probe_endpoint = probe_listen.as_ref().map(|listen| probe_url(listen));
            // Kept whole and unshrunk. Limits describe how much of this machine
            // is lent, so raising one through the control surface has to be
            // able to give back what lowering it took away -- which needs the
            // machine as detected, not the last share of it.
            let detected = Arc::new(detected);
            let caps = control::advertised_capabilities(&detected, &limits, runtime_available);
            if ai_runtime.is_some() && !caps.ai_runtime_ready {
                println!(
                    "An inference runtime is available but this node is not offering AI work. \
                     Run `hocmesh limits --ai on` to run other people's inference here."
                );
            } else if let Some(cpu) = caps
                .gpus
                .iter()
                .find(|gpu| gpu.stable_id == hardware::SHARED_CPU_DEVICE_ID)
            {
                println!(
                    "Offering AI work on CPU ({}, {} MiB shared). Inference will run and will be \
                     slow; for acceleration build llama.cpp for the local backend and pass \
                     --ai-runtime.",
                    cpu.name,
                    cpu.memory_mb.unwrap_or(0)
                );
            }
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
            let control = (!no_control).then(|| daemon::ControlConfig {
                home: cli.home.clone(),
                port: control_port,
                detected,
                runtime_available,
            });
            daemon::run(
                client,
                daemon::DaemonConfig {
                    capabilities: caps,
                    workers,
                    poll_ms,
                    ai,
                    proximity,
                    control,
                },
            )
            .await?;
        }
        Command::Stop => {
            if control::request_shutdown(&cli.home).await? {
                println!("Asked the hocMESH daemon to stop.");
            } else {
                println!("No hocMESH daemon is running for {}.", cli.home.display());
            }
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
            ai,
        } => {
            let mut limits = ResourceLimits::load_or_default(&cli.home)?;
            let changed = cpu_percent.is_some()
                || gpu_percent.is_some()
                || memory_percent.is_some()
                || ai.is_some();
            if let Some(v) = cpu_percent {
                limits.cpu_percent = v;
            }
            if let Some(v) = gpu_percent {
                limits.gpu_percent = v;
            }
            if let Some(v) = memory_percent {
                limits.memory_percent = v;
            }
            if let Some(v) = ai {
                limits.ai = v.into();
            }
            if changed {
                limits.save(&cli.home)?;
            }
            let cpus = hardware::detect_capabilities(false).logical_cpus;
            println!(
                "cpu {}%  gpu {}%  memory {}%
ai: {}
workers when running: {} of {} logical CPUs
stored at: {}",
                limits.cpu_percent,
                limits.gpu_percent,
                limits.memory_percent,
                describe_ai_sharing(&limits),
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
        Command::ModelInspect {
            path,
            stages,
            chunk_size,
        } => {
            inspect_gguf(&path, stages, chunk_size)?;
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
        Command::ModelCatalog => {
            println!(
                "{:<24} {:>8} {:>9}  {:<10} REPOSITORY",
                "ID", "PARAMS", "SIZE", "LICENCE"
            );
            for entry in hocmesh_model::catalog::CATALOG {
                println!(
                    "{:<24} {:>8} {:>9}  {:<10} {}",
                    entry.id,
                    entry.parameters,
                    install::human_bytes(entry.approx_bytes),
                    entry.license,
                    entry.repository
                );
                println!("{:<24} {}", "", entry.summary);
            }
            println!(
                "\nPull one with `hocmesh model-pull <id>`. Sizes are approximate, and the \
                 exact file and its digest are resolved from the repository at pull time."
            );
        }
        Command::ModelPull {
            id,
            repository,
            url,
            quantisation,
            revision,
            sha256,
            model_id,
            architecture,
            chunk_size,
            keep_download,
        } => {
            let source = match (id, repository, url) {
                (Some(id), None, None) => {
                    let entry = hocmesh_model::catalog::lookup(&id).with_context(|| {
                        let close = hocmesh_model::catalog::suggestions(&id);
                        if close.is_empty() {
                            format!(
                                "{id} is not in the catalogue. Run `hocmesh model-catalog` to \
                                 see it, or pass --repository owner/name for anything else"
                            )
                        } else {
                            format!(
                                "{id} is not in the catalogue. Did you mean: {}",
                                close.join(", ")
                            )
                        }
                    })?;
                    install::PullSource::Catalogued(entry)
                }
                (None, Some(repository), None) => install::PullSource::Repository {
                    repository,
                    quantisation,
                },
                (None, None, Some(url)) => install::PullSource::Url(url),
                _ => {
                    bail!("give exactly one of: a catalogue id, --repository owner/name, or --url")
                }
            };
            let pulled = install::pull_model(
                &cli.home,
                install::PullRequest {
                    source,
                    revision,
                    sha256,
                    model_id,
                    architecture,
                    chunk_size,
                    keep_download,
                },
            )
            .await?;
            if !pulled.downloaded {
                println!("Already downloaded and verified; re-importing.");
            }
            println!(
                "Imported {}@{}: {} ({} bytes) in {} chunks, {} architecture, manifest {}",
                pulled.model_id,
                pulled.revision,
                install::human_bytes(pulled.size_bytes),
                pulled.size_bytes,
                pulled.chunks,
                pulled.architecture,
                pulled.manifest_digest
            );
            println!("sha256 {}", pulled.sha256);
            println!("from   {}", pulled.source_url);
            println!(
                "Run it with `hocmesh infer --model-id {} --prompt \"...\"`.",
                pulled.model_id
            );
        }
        Command::RuntimeInstall {
            force,
            keep_download,
        } => {
            let installed = install::install_runtime(&cli.home, force, keep_download).await?;
            if installed.installed_now {
                println!(
                    "Installed llama.cpp {} ({})",
                    installed.build, installed.asset
                );
                println!("sha256 {} (verified)", installed.sha256);
            } else {
                println!(
                    "llama.cpp {} is already installed; pass --force to reinstall.",
                    installed.build
                );
            }
            println!("runtime {}", installed.executable.display());
            println!("`hocmesh infer` and `hocmesh daemon` now use it without a --runtime flag.");
        }
        Command::RuntimeStatus => {
            match hocmesh_gpu::runtime::asset_for_host() {
                Ok(asset) => {
                    println!(
                        "Pinned  llama.cpp {} for {}/{}",
                        hocmesh_gpu::runtime::PINNED_BUILD,
                        asset.os,
                        asset.arch
                    );
                    println!(
                        "Asset   {} ({})",
                        asset.asset,
                        install::human_bytes(asset.size_bytes)
                    );
                    println!("Expects sha256 {}", asset.sha256);
                    println!("URL     {}", asset.url());
                }
                Err(error) => println!("Pinned  {error}"),
            }
            match hocmesh_gpu::runtime::installed_runtime(&cli.home) {
                Some(path) => println!("Runtime {}", path.display()),
                None => println!(
                    "Runtime not installed. Run `hocmesh runtime-install`, or point --runtime \
                     at a llama.cpp build you already have."
                ),
            }
            let devices = hocmesh_gpu::discover_devices();
            if devices.is_empty() {
                println!("Devices none detected; inference will run on the CPU.");
            } else {
                for device in devices {
                    let memory = match device.memory_bytes {
                        Some(bytes) => install::human_bytes(bytes),
                        None => "unknown memory".to_string(),
                    };
                    println!(
                        "Device  {} ({:?}, {memory})",
                        device.stable_id, device.backend
                    );
                }
            }
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
                    memory_bandwidth_bytes_per_second: None,
                });
            let runtime = match runtime {
                Some(explicit) => explicit,
                None => hocmesh_gpu::runtime::require_runtime(&cli.home)?,
            };
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
            // The requester prices its own job from the published manifest, and
            // signs that price. Nothing downstream gets to raise it.
            let manifest = client.get_model(&model_id, &revision).await?;
            let digest = manifest.digest()?;
            let parameter_count = manifest.parameter_count.ok_or_else(|| {
                anyhow::anyhow!("model does not declare a parameter count, so it cannot be priced")
            })?;
            let billing = hocmesh_ai::bill_for_prompts(
                &digest,
                parameter_count,
                manifest.total_size_bytes,
                &prompt,
                max_tokens,
            )?;
            println!(
                "job priced at {:.3} CU",
                billing.max_cost_mcu as f64 / 1000.0
            );
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
                billing,
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
        Command::AiReclaim { job_id } => {
            let refunds = client.reclaim_inference(&job_id).await?;
            if refunds.is_empty() {
                println!(
                    "Nothing to reclaim: every batch of {job_id} either settled or is still inside its window."
                );
            }
            for r in &refunds {
                println!(
                    "Returned {:.3} CU - balance now {:.3} CU",
                    r.refunded_mcu as f64 / 1000.0,
                    r.balance_mcu as f64 / 1000.0
                );
            }
        }
        Command::AiReceipt {
            job_id,
            assignment_id,
        } => {
            let batch = find_delivered(&client, &job_id, &assignment_id).await?;
            let taken = client.receipt_inference(&job_id, &batch).await?;
            println!(
                "Took delivery of {} ({} prompts) for {:.3} CU, held pending settlement.",
                taken.assignment_id,
                taken.outputs.len(),
                taken.price_mcu as f64 / 1000.0
            );
            println!("{}", serde_json::to_string_pretty(&taken.outputs)?);
            println!(
                "Now say what it was worth: hocmesh ai-settle {job_id} {} [--dispute --reason ...]",
                taken.assignment_id
            );
        }
        Command::AiSettle {
            job_id,
            assignment_id,
            dispute,
            reason,
        } => {
            let batch = find_delivered(&client, &job_id, &assignment_id).await?;
            let reason = if dispute && reason.trim().is_empty() {
                "the answer was not usable".to_string()
            } else {
                reason
            };
            let settled = client
                .settle_inference(&job_id, &batch, !dispute, &reason)
                .await?;
            if settled.accepted {
                println!(
                    "Paid {:.3} CU to the provider of {}.",
                    settled.paid_mcu as f64 / 1000.0,
                    settled.assignment_id
                );
            } else {
                println!(
                    "Disputed {}. The {:.3} CU went to the commons, not back to you.",
                    settled.assignment_id,
                    batch.price_mcu as f64 / 1000.0
                );
                println!("Reason recorded: {reason}");
            }
            if settled.job_completed {
                println!("Every batch of {job_id} is now settled.");
            }
        }
        Command::SubmitCollatz { start, end, shards } => {
            let r = client
                .submit(WorkSpec::CollatzPeak { start, end }, shards)
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
            if let Some(c) = j.collatz_peak {
                println!(
                    "Current Collatz peak: {} steps from seed {}",
                    c.steps, c.seed
                );
            }
            if !j.refundable.is_empty() {
                println!(
                    "Reclaimable: {} shard(s) worth {:.3} CU - run `hocmesh reclaim {}`",
                    j.refundable.len(),
                    j.refundable.iter().map(|s| s.refund_mcu).sum::<i64>() as f64 / 1000.0,
                    j.job_id
                );
            }
        }
        Command::Reclaim { job_id } => {
            let refunds = client.reclaim(&job_id).await?;
            if refunds.is_empty() {
                println!(
                    "Nothing to reclaim: no shard of {job_id} has outlived its settlement window."
                );
            }
            let total: i64 = refunds.iter().map(|r| r.refund_mcu).sum();
            for r in &refunds {
                println!(
                    "Returned {:.3} CU to {}{}",
                    r.refund_mcu as f64 / 1000.0,
                    r.paid_to,
                    r.ledger_entry_hash
                        .as_deref()
                        .map(|h| format!(" (ledger entry {h})"))
                        .unwrap_or_default()
                );
            }
            if !refunds.is_empty() {
                println!("Reclaimed {:.3} CU in total.", total as f64 / 1000.0);
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
        Command::Reconciliation => {
            let r = client.reconciliation().await?;
            if r.unsettled.is_empty() {
                println!("Nothing unsettled: every persisted intent reached the chain.");
            }
            for i in &r.unsettled {
                println!(
                    "{:<14} {:<12} {} attempts={} {}",
                    i.status, i.intent_kind, i.object_id, i.attempts, i.claim_key
                );
                if let Some(why) = &i.last_error {
                    println!("               last error: {why}");
                }
            }
            // Reported and not repaired: a coordinator that closed this gap on
            // its own would be minting CU, so an operator has to look.
            if r.orphaned_objects > 0 {
                println!(
                    "{} job(s) or assignment(s) are waiting on funding no intent covers.",
                    r.orphaned_objects
                );
            }
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
                // The set a mirror verifies against is whatever the chain has
                // last handed it, not the bootstrap file. A mirror that kept
                // using the file would stop dead at the first admission.
                let set = store.current_set()?.unwrap_or_else(|| net.set());
                let local = store.head(&set)?;
                net.refresh_set().await?;
                let remote = net.head_quorum().await?;
                if local.sequence >= remote.sequence {
                    // A mirror that was pruned, or that started from a
                    // snapshot, holds no history below its checkpoint, so
                    // demanding a genesis replay here would make sync
                    // impossible for the very nodes bootstrap exists for.
                    let from = store.latest_checkpoint()?;
                    let genesis = load_set(&validators)?;
                    let audit_set = match &from {
                        Some(cp) => store.set_at(cp.head.sequence)?.unwrap_or(genesis),
                        None => genesis,
                    };
                    let audited = store.audit_from(&audit_set, from.as_ref())?;
                    println!(
                        "Ledger mirror synchronized and audited: height={} head={}",
                        audited.sequence, audited.entry_hash
                    );
                    break;
                }
                let certs = net
                    .fetch_certificates(local.sequence + 1, batch.max(1), &set)
                    .await?;
                if certs.is_empty() {
                    anyhow::bail!("validators report a newer head but returned no certificates")
                };
                let mut set = set;
                for cert in certs {
                    store.apply(&cert, &set)?;
                    if let Some(next) = store.current_set()? {
                        set = next;
                    }
                }
            }
        }
        Command::LedgerAudit {
            validators,
            db,
            full,
        } => {
            let genesis = load_set(&validators)?;
            let store = LedgerStore::open(&db)?;
            let from = if full {
                None
            } else {
                store.latest_checkpoint()?
            };
            // A full replay starts at genesis and evolves the set as the chain
            // changed it; one that resumes from a checkpoint needs the set that
            // was sitting when that checkpoint was signed.
            let set = match &from {
                Some(cp) => store.set_at(cp.head.sequence)?.unwrap_or(genesis),
                None => genesis,
            };
            let h = store.audit_from(&set, from.as_ref())?;
            let start = from.as_ref().map_or(0, |c| c.head.sequence);
            println!(
                "LEDGER AUDIT OK: height={} head={} replayed_from={}",
                h.sequence, h.entry_hash, start
            );
        }
        Command::LedgerCheckpoint { validators, db } => {
            let net = load_network(&validators)?;
            let store = LedgerStore::open(&db)?;
            // A checkpoint is only meaningful against the set that signed
            // it, so follow the chain forward before storing one.
            net.refresh_set().await?;
            let cp = net.checkpoint_quorum().await?;
            store.store_checkpoint(&cp, &net.set())?;
            println!(
                "CHECKPOINT STORED: height={} head={} state={} signatures={}",
                cp.head.sequence,
                cp.head.entry_hash,
                cp.state_hash,
                cp.signatures.len()
            );
        }
        Command::LedgerPrune { validators, db } => {
            let store = LedgerStore::open(&db)?;
            let set = store.current_set()?.unwrap_or(load_set(&validators)?);
            let removed = store.prune_below_checkpoint(&set)?;
            println!("PRUNED {removed} certificates already covered by a checkpoint");
        }
        Command::LedgerSnapshot {
            validators,
            db,
            out,
        } => {
            let store = LedgerStore::open(&db)?;
            let set = store.current_set()?.unwrap_or(load_set(&validators)?);
            let snap = store.snapshot(&set)?;
            fs::write(&out, serde_json::to_string_pretty(&snap)?)?;
            println!(
                "SNAPSHOT WRITTEN: {} height={} state={} accounts={}",
                out.display(),
                snap.checkpoint.head.sequence,
                snap.checkpoint.state_hash,
                snap.state.balances.len()
            );
        }
        Command::LedgerRestore {
            validators,
            db,
            snapshot,
        } => {
            let set = load_set(&validators)?;
            let snap: LedgerSnapshot = serde_json::from_str(
                &fs::read_to_string(&snapshot)
                    .with_context(|| format!("reading {}", snapshot.display()))?,
            )?;
            let mut store = LedgerStore::open(&db)?;
            store.install_snapshot(&snap, &set)?;
            let head = store.audit_from(&set, Some(&snap.checkpoint))?;
            println!(
                "RESTORED: height={} head={} — sync from here rather than genesis",
                head.sequence, head.entry_hash
            );
        }
        Command::LedgerHistory {
            account,
            db,
            validators,
            before,
            limit,
        } => {
            // A pruned mirror is not a broken one -- it simply stopped holding
            // what it no longer needs -- so the network is the fallback rather
            // than an error.
            let page = match &validators {
                Some(v) => {
                    load_network(v)?
                        .fetch_history(&account, before, limit)
                        .await?
                }
                None => LedgerStore::open(&db)?.history(&account, before, limit)?,
            };
            println!("History for {} (newest first)", page.account_id);
            for e in &page.entries {
                println!(
                    "  seq={:<6} #{:<2} {:>10.3} CU  tx={}  at={}",
                    e.sequence,
                    e.posting_index,
                    e.delta_mcu as f64 / 1000.0,
                    e.transaction_id,
                    e.created_at
                );
            }
            match page.next_before {
                Some(b) => println!("-- older postings remain: rerun with --before {b}"),
                None => println!("-- start of this ledger's history for {account}"),
            }
        }
        Command::CommunityVouch {
            validators,
            job_id,
            start,
            end,
            shards,
        } => {
            let set = load_set(&validators)?;
            let work = WorkSpec::PrimeCount { start, end };
            let shards = shards.clamp(1, 256);
            let cost: i64 = split_work(&work, shards).iter().map(work_cost_mcu).sum();
            let message = community_reserve_signing_message(&job_id, &work, shards, cost)?;
            let vouch = ValidatorSignature {
                validator_id: identity.node_id(),
                signature_b64: identity.sign_bytes_b64(message.as_bytes()),
            };
            if !set
                .members
                .iter()
                .any(|m| m.validator_id == vouch.validator_id)
            {
                bail!(
                    "this node is not in the sitting validator set, so its sponsorship counts for nothing"
                )
            }
            println!("{}", serde_json::to_string(&vouch)?);
        }
        Command::MembershipVouch {
            validators,
            action,
            member,
            threshold,
        } => {
            let set = load_set(&validators)?;
            let member: ValidatorMember = read_json(&member)?;
            let action: MembershipAction = action.into();
            let evidence = membership_evidence(&set, action, member, threshold, Vec::new())?;
            let message = vouch_signing_message(
                &membership_hash(&set)?,
                action,
                &evidence.member,
                &evidence.resulting_set_hash,
            );
            let vouch = ValidatorSignature {
                validator_id: identity.node_id(),
                signature_b64: identity.sign_bytes_b64(message.as_bytes()),
            };
            if !set
                .members
                .iter()
                .any(|m| m.validator_id == vouch.validator_id)
            {
                bail!(
                    "this node is not in the sitting validator set, so its vouch counts for nothing"
                )
            }
            println!("{}", serde_json::to_string(&vouch)?);
        }
        Command::MembershipCommit {
            validators,
            action,
            member,
            threshold,
            vouches,
            out,
        } => {
            let set = load_set(&validators)?;
            let member: ValidatorMember = read_json(&member)?;
            let vouches: Vec<ValidatorSignature> = read_json(&vouches)?;
            let action: MembershipAction = action.into();
            let evidence = membership_evidence(&set, action, member, threshold, vouches)?;
            let next = verify_membership_change(&set, &evidence)?;
            let tx = LedgerTransaction {
                transaction_id: format!("membership_{}", uuid::Uuid::new_v4().simple()),
                kind: TransactionKind::MembershipChange,
                postings: Vec::new(),
                evidence: TransactionEvidence::MembershipChange(evidence),
                created_at: hocmesh_protocol::now_unix(),
            };
            let cert = LedgerNetwork::new(set)?.transact(tx).await?;
            if let Some(path) = out {
                std::fs::write(&path, serde_json::to_string_pretty(&next)?)?;
                println!("Wrote the new validator set to {path}");
            }
            println!(
                "MEMBERSHIP CHANGE CERTIFIED at sequence {} ({})",
                cert.entry.sequence, cert.entry.entry_hash
            );
        }
        Command::Loadtest {
            jobs,
            concurrency,
            shards,
            workload,
            size,
            duration_secs,
            timeout_secs,
            poll_ms,
            json,
            dry_run,
        } => {
            let plan = LoadPlan {
                jobs,
                concurrency,
                shards,
                workload,
                size,
                duration: duration_secs.map(Duration::from_secs),
                timeout: Duration::from_secs(timeout_secs),
                poll: Duration::from_millis(poll_ms),
            };
            if dry_run {
                plan.validate()?;
                // One machine-readable line, because the caller most likely to
                // want this is a shell script deciding whether to wait.
                println!("total_mcu={}", plan.cost_mcu());
                println!("per_job_mcu={}", plan.job_cost_mcu(0));
                println!(
                    "{} jobs x {} shards of {:?}, size {} -> {:.3} CU",
                    plan.jobs,
                    plan.shards,
                    plan.workload,
                    plan.size,
                    plan.cost_mcu() as f64 / 1000.0
                );
                return Ok(());
            }
            let report = loadtest::run(&client, plan).await?;
            report.print();
            if let Some(path) = &json {
                std::fs::write(path, serde_json::to_string_pretty(&report)?)
                    .with_context(|| format!("writing {}", path.display()))?;
                println!(
                    "
Wrote {}",
                    path.display()
                );
            }
            // Printed first, then failed: a pipeline that only keeps the exit
            // code still gets the numbers in its log.
            if !report.passed() {
                bail!("load test failed")
            }
        }
        // Taken out above, before an identity could be created. Left here doing
        // the real work rather than panicking, so that if that dispatch is ever
        // dropped these commands still behave -- they just lose the guarantee
        // that looking at an account cannot bring one into existence.
        Command::Identity { action } => run_identity(&cli.home, &action)?,
    }
    Ok(())
}

/// Resolves one batch of a job that a provider has answered but the requester
/// has not yet taken delivery of.
///
/// The list comes from the coordinator's job status, which shows the digest and
/// the price of every answered batch but never the text. That is the point: a
/// requester decides whether to pay for delivery before it can read anything.
async fn find_delivered(
    client: &HocMeshClient,
    job_id: &str,
    assignment_id: &str,
) -> Result<hocmesh_ai::DeliveredBatchSummary> {
    let status = client.inference_status(job_id).await?;
    let Some(batch) = status
        .delivered
        .into_iter()
        .find(|b| b.assignment_id == assignment_id)
    else {
        anyhow::bail!("job {job_id} has no answered batch called {assignment_id}");
    };
    if let Some(verdict) = &batch.settled {
        anyhow::bail!("batch {assignment_id} was already settled as {verdict}");
    }
    Ok(batch)
}

fn model_store(home: &std::path::Path) -> Result<ChunkStore> {
    ChunkStore::open(home.join("model-cache"))
}

fn model_registry(home: &std::path::Path) -> Result<ModelRegistry> {
    fs::create_dir_all(home)?;
    ModelRegistry::open(home.join("model-registry.db"))
}

/// The most of a GGUF file this will read to find the header. Real headers are
/// a few megabytes at worst; the bound is what stops a hostile file from being
/// pulled into memory whole.
const MAX_GGUF_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// Report a GGUF file's layout, and what a pipeline of `stages` would cost each
/// stage to hold.
///
/// This is the arithmetic behind fetching only the layers a stage will run: the
/// chunk counts are the chunks that stage would have to pull, out of the chunks
/// in the whole file.
fn inspect_gguf(path: &Path, stages: u32, chunk_size: usize) -> Result<()> {
    use hocmesh_model::gguf::{self, ByteExtent, TensorDirectory};

    ensure!(
        stages >= 1,
        "a model has to be split into at least one stage"
    );
    ensure!(chunk_size > 0, "chunk size must be greater than zero");
    let chunk_size = chunk_size as u64;

    let file_len = fs::metadata(path)
        .with_context(|| format!("cannot read {}", path.display()))?
        .len();
    let mut head = Vec::new();
    fs::File::open(path)?
        .take(MAX_GGUF_HEADER_BYTES)
        .read_to_end(&mut head)?;

    let directory = gguf::tensor_directory(&head)?.with_context(|| {
        format!(
            "the tensor directory does not end within the first {MAX_GGUF_HEADER_BYTES} bytes of {}",
            path.display()
        )
    })?;
    directory.validate(file_len)?;

    let architecture = gguf::architecture(&head)?.unwrap_or_else(|| "unknown".to_string());
    let name = gguf::model_name(&head)?.unwrap_or_else(|| "unnamed".to_string());
    let blocks = directory.layer_count();
    let total_chunks = file_len.div_ceil(chunk_size);

    println!("{name} ({architecture})");
    println!(
        "  {file_len} bytes, {} tensors, {blocks} transformer blocks",
        directory.tensors.len()
    );
    println!(
        "  tensor data starts at {}, aligned to {}",
        directory.data_start, directory.alignment
    );

    let shared = directory.extents_of(&directory.shared_tensors(), file_len);
    let shared_bytes: u64 = shared.iter().map(ByteExtent::len).sum();
    println!(
        "  shared (embeddings, final norm, output head): {shared_bytes} bytes; the first and last stage need these"
    );

    ensure!(
        blocks > 0,
        "this file names no transformer blocks, so it cannot be split by layer"
    );
    ensure!(
        stages <= blocks,
        "{stages} stages asked for but the model has only {blocks} blocks"
    );

    for stage in 0..stages {
        let first = block_boundary(blocks, stages, stage);
        let last = block_boundary(blocks, stages, stage + 1);
        let extents = directory.extents_for_layers(first..last, file_len);
        let bytes: u64 = extents.iter().map(ByteExtent::len).sum();
        let chunks = TensorDirectory::chunks_for_extents(&extents, chunk_size)?;
        println!(
            "  stage {}/{stages}: blocks {first}..{last}, {bytes} bytes in {} span(s), {} of {total_chunks} chunks",
            stage + 1,
            extents.len(),
            chunks.len()
        );
    }

    Ok(())
}

/// Where stage `stage` begins when `blocks` are divided as evenly as they go
/// between `stages`. Computed in u64 so a large block count cannot wrap.
fn block_boundary(blocks: u32, stages: u32, stage: u32) -> u32 {
    let boundary = u64::from(blocks) * u64::from(stage) / u64::from(stages);
    u32::try_from(boundary).unwrap_or(blocks)
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

/// Whether this node runs other people's inference.
///
/// `Auto` is not the same as `Off`: it means the operator has not said, and the
/// node falls back to "offer it when a GPU is lent", which is what every node
/// did before there was a flag. Keeping it distinct is what lets an upgrade
/// leave existing machines exactly where they were.
/// The passphrase sealing this machine's own key, if there is one.
fn node_passphrase() -> Option<String> {
    std::env::var(IDENTITY_PASSPHRASE_ENV)
        .ok()
        .filter(|p| !p.is_empty())
}

/// The passphrase for a backup file.
///
/// Falls back to the node's own, so one passphrase is enough for anyone who
/// wants it to be, while still letting an operator run a node unsealed on a
/// machine only they can reach and refuse to let that key travel in the clear.
fn backup_passphrase() -> Option<String> {
    std::env::var(IDENTITY_EXPORT_PASSPHRASE_ENV)
        .ok()
        .filter(|p| !p.is_empty())
        .or_else(node_passphrase)
}

fn open_this_machine(home: &Path) -> Result<NodeIdentity> {
    let local = node_passphrase();
    NodeIdentity::load_existing(home, local.as_deref())?.ok_or_else(|| {
        anyhow::anyhow!(
            "no account in {} yet -- run `hocmesh init`, or `hocmesh identity import` to \
             bring an existing one here",
            home.display()
        )
    })
}

fn run_identity(home: &Path, action: &IdentityAction) -> Result<()> {
    match action {
        IdentityAction::Show => {
            let path = identity_path(home);
            match NodeIdentity::load_existing(home, node_passphrase().as_deref()) {
                Ok(Some(id)) => {
                    let sealed = std::fs::read_to_string(&path)
                        .map(|r| !r.contains("secret_key_b64"))
                        .unwrap_or(false);
                    println!(
                        "Account: {}\nPublic key: {}\nKey file: {}\nAt rest: {}",
                        id.node_id(),
                        id.public_key_b64(),
                        path.display(),
                        if sealed {
                            "sealed with a passphrase".to_string()
                        } else {
                            format!("stored unsealed -- set {IDENTITY_PASSPHRASE_ENV} to seal it")
                        }
                    );
                    println!(
                        "\nThis key is the account. Your balance follows it, not this machine,\n\
                         and no part of the network holds a copy: back it up with\n\
                         `hocmesh identity export --out <file>`."
                    );
                }
                Ok(None) => println!(
                    "No account in {} yet. One is created the first time this node runs,\n\
                     or `hocmesh identity import --from <file>` brings an existing one here.",
                    home.display()
                ),
                Err(e) => return Err(e),
            }
        }
        IdentityAction::Export { out, force } => {
            let Some(pass) = backup_passphrase() else {
                anyhow::bail!(
                    "a backup is always encrypted; set {IDENTITY_EXPORT_PASSPHRASE_ENV} (or \
                     {IDENTITY_PASSPHRASE_ENV}) to the passphrase that should seal it"
                )
            };
            let id = open_this_machine(home)?;
            identity::write_backup(out, &id.export_backup(&pass)?, *force)?;
            println!(
                "Wrote a sealed backup of {} to {}.\n\n\
                 Keep it somewhere you will still have after this machine is gone, and keep\n\
                 the passphrase somewhere else. Losing both loses the account: there is no\n\
                 reset, because nobody but you ever had the key.",
                id.node_id(),
                out.display()
            );
        }
        IdentityAction::Import { from, force } => {
            let Some(pass) = backup_passphrase() else {
                anyhow::bail!(
                    "set {IDENTITY_EXPORT_PASSPHRASE_ENV} (or {IDENTITY_PASSPHRASE_ENV}) to \
                     the passphrase this backup was sealed with"
                )
            };
            let backup = identity::read_backup(from)?;
            let local = node_passphrase();
            let id = identity::import_backup(home, &backup, &pass, local.as_deref(), *force)?;
            println!(
                "This machine now signs as {}.\nKey file: {}",
                id.node_id(),
                identity_path(home).display()
            );
            println!(
                "\nThe balance was never stored here -- it is what the ledger implies for this\n\
                 account, so `hocmesh balance` reads the same number it read on the old machine.\n\
                 Do not run both machines on this key at once."
            );
        }
        IdentityAction::Inspect { from } => {
            let b = identity::read_backup(from)?;
            println!(
                "Backup of: {}\nPublic key: {}\nFormat: {} v{}\nCreated: {}\nSealed: yes",
                b.node_id, b.public_key_b64, b.format, b.version, b.created_at
            );
        }
    }
    Ok(())
}

#[derive(Debug, Subcommand)]
enum IdentityAction {
    /// Show which account this machine signs as, and where the key lives.
    Show,
    /// Write a sealed copy of this account that another machine can adopt.
    ///
    /// Always encrypted, so it can be kept somewhere a raw key should never go.
    Export {
        /// Where to write the backup.
        #[arg(long)]
        out: PathBuf,
        /// Replace a backup file that is already there.
        #[arg(long)]
        force: bool,
    },
    /// Adopt an exported account on this machine.
    Import {
        /// The backup to restore.
        #[arg(long)]
        from: PathBuf,
        /// Replace the account already on this machine. The key it displaces is
        /// renamed, never deleted.
        #[arg(long)]
        force: bool,
    },
    /// Say whose account a backup holds, without opening it.
    Inspect {
        #[arg(long)]
        from: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum AiSharing {
    On,
    Off,
    Auto,
}

impl From<AiSharing> for Option<bool> {
    fn from(value: AiSharing) -> Self {
        match value {
            AiSharing::On => Some(true),
            AiSharing::Off => Some(false),
            AiSharing::Auto => None,
        }
    }
}

/// How `limits` reports the AI share, spelling out what the fallback resolves to.
fn describe_ai_sharing(limits: &ResourceLimits) -> String {
    match limits.ai {
        Some(true) => "on".to_string(),
        Some(false) => "off".to_string(),
        None if limits.offers_gpu() => "auto (on: a GPU is lent)".to_string(),
        None => "auto (off: no GPU is lent -- use --ai on to offer CPU inference)".to_string(),
    }
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum MembershipActionArg {
    Join,
    Leave,
}
impl From<MembershipActionArg> for MembershipAction {
    fn from(a: MembershipActionArg) -> Self {
        match a {
            MembershipActionArg::Join => MembershipAction::Join,
            MembershipActionArg::Leave => MembershipAction::Leave,
        }
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &str) -> Result<T> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {path}"))
}

/// Builds the change so a vouch and the commit describe the same thing.
///
/// The resulting set hash is derived on both sides rather than passed along,
/// so a sponsor signs for the set its own copy of the rules produces and not
/// for a hash somebody handed it.
fn membership_evidence(
    set: &ValidatorSet,
    action: MembershipAction,
    member: ValidatorMember,
    threshold: usize,
    vouches: Vec<ValidatorSignature>,
) -> Result<MembershipChangeEvidence> {
    let next = membership_result(set, action, &member, threshold)?;
    Ok(MembershipChangeEvidence {
        action,
        member,
        threshold,
        vouches,
        resulting_set_hash: membership_hash(&next)?,
    })
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
