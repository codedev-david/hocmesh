mod client;
mod daemon;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use client::MeshClient;
use mesh_core::{hardware, identity::NodeIdentity};
use mesh_ledger::{
    network::LedgerNetwork, store::LedgerStore, types::ValidatorSet,
    validate::validate_validator_set,
};
use mesh_protocol::WorkSpec;
use std::{fs, path::PathBuf};
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
    },
    Status,
    Balance,
    Benchmark,
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
        Command::Daemon { workers, poll_ms } => {
            let caps = hardware::detect_capabilities(true);
            let workers = workers.unwrap_or_else(|| caps.logical_cpus.saturating_sub(1).max(1));
            daemon::run(client, caps, workers, poll_ms).await?;
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
