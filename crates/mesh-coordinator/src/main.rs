mod api;
mod db;
mod error;

use anyhow::{Context, Result, bail};
use api::AppState;
use clap::{Parser, Subcommand};
use mesh_core::compute::{split_work, work_cost_mcu};
use mesh_ledger::{
    network::LedgerNetwork,
    types::{
        COMMUNITY_ISSUANCE_ACCOUNT, LedgerTransaction, Posting, TransactionEvidence,
        TransactionKind, ValidatorSet, escrow_account,
    },
    validate::claim_key,
};
use mesh_protocol::{WorkSpec, now_unix};
use rusqlite::{OptionalExtension, params};
use std::{
    fs,
    sync::{Arc, Mutex},
};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(
    name = "mesh-coordinator",
    version,
    about = "MESH scheduling/control plane"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Debug, Subcommand)]
enum Command {
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: String,
        #[arg(long, default_value = "mesh.db")]
        db: String,
        #[arg(long)]
        validators: Option<String>,
    },
    Seed {
        #[arg(long, default_value = "mesh.db")]
        db: String,
        #[arg(long, default_value_t = 2)]
        start: u64,
        #[arg(long, default_value_t = 5_000_000)]
        end: u64,
        #[arg(long, default_value_t = 32)]
        shards: u32,
        #[arg(long)]
        validators: Option<String>,
    },
    Recover {
        #[arg(long, default_value = "mesh.db")]
        db: String,
        #[arg(long)]
        validators: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    match Cli::parse().command {
        Command::Serve {
            listen,
            db,
            validators,
        } => serve(&listen, &db, validators.as_deref()).await,
        Command::Seed {
            db,
            start,
            end,
            shards,
            validators,
        } => {
            seed(
                &db,
                WorkSpec::PrimeCount { start, end },
                shards,
                validators.as_deref(),
            )
            .await
        }
        Command::Recover { db, validators } => {
            let net = load_network(&validators)?;
            recover_pending(&db, &net).await
        }
    }
}
fn load_network(path: &str) -> Result<LedgerNetwork> {
    let set: ValidatorSet = serde_json::from_str(
        &fs::read_to_string(path).with_context(|| format!("reading validator set {path}"))?,
    )?;
    LedgerNetwork::new(set)
}

async fn seed(db_path: &str, work: WorkSpec, shards: u32, validators: Option<&str>) -> Result<()> {
    work.validate().map_err(anyhow::Error::msg)?;
    let shards = shards.clamp(1, 256);
    let job_id = format!("job_community_{}", Uuid::new_v4().simple());
    let network = validators.map(load_network).transpose()?;
    let ledger_tx = network.as_ref().map(|_| {
        let cost: i64 = split_work(&work, shards).iter().map(work_cost_mcu).sum();
        LedgerTransaction {
            transaction_id: format!("community_reserve_{job_id}"),
            kind: TransactionKind::CommunityReserve,
            postings: vec![
                Posting {
                    account_id: COMMUNITY_ISSUANCE_ACCOUNT.into(),
                    delta_mcu: -cost,
                },
                Posting {
                    account_id: escrow_account(&job_id),
                    delta_mcu: cost,
                },
            ],
            evidence: TransactionEvidence::CommunityReserve {
                job_id: job_id.clone(),
                work: work.clone(),
                shards,
            },
            created_at: now_unix(),
        }
    });
    let mut conn = db::open(db_path)?;
    db::seed_system_job_with_id(&mut conn, &job_id, work, shards, ledger_tx.is_none())?;
    if let Some(tx) = ledger_tx {
        let ck = claim_key(&tx);
        db::persist_ledger_intent(
            &conn,
            &ck,
            "community_reserve",
            &job_id,
            &serde_json::to_string(&tx)?,
        )?;
        match network.as_ref().unwrap().transact(tx).await {
            Ok(cert) => {
                finalize_reservation_db(&mut conn, &job_id, &ck, &cert.entry.entry_hash)?;
                println!("Community reservation certified: {}", cert.entry.entry_hash);
            }
            Err(e) => {
                println!(
                    "Community job persisted in funding state; run recover after validators are available: {e}"
                );
                return Err(e);
            }
        }
    }
    println!("Seeded system-funded community job: {job_id}");
    Ok(())
}

async fn serve(listen: &str, db_path: &str, validators: Option<&str>) -> Result<()> {
    let ledger = match validators {
        Some(p) => Some(load_network(p)?),
        None => None,
    };
    if let Some(net) = &ledger
        && let Err(e) = recover_pending(db_path, net).await
    {
        tracing::warn!(error=%e,"coordinator recovery incomplete; serving with unresolved intents blocked")
    }
    if let Some(net) = ledger.clone() {
        let recovery_db = db_path.to_string();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(15));
            loop {
                interval.tick().await;
                if let Err(e) = recover_pending(&recovery_db, &net).await {
                    tracing::warn!(error=%e,"background ledger intent recovery incomplete")
                }
            }
        });
    }
    let conn = db::open(db_path)?;
    let mode = if ledger.is_some() {
        "quorum"
    } else {
        "local-mvp"
    };
    let state = AppState {
        db: Arc::new(Mutex::new(conn)),
        ledger,
    };
    let app = api::router(state);
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .with_context(|| format!("binding coordinator to {listen}"))?;
    tracing::info!(listen=%listen,db=%db_path,ledger_mode=mode,"MESH coordinator started");
    axum::serve(listener, app)
        .await
        .context("coordinator server failed")?;
    Ok(())
}

async fn recover_pending(db_path: &str, net: &LedgerNetwork) -> Result<()> {
    let mut conn = db::open(db_path)?;
    let intents = {
        let mut st=conn.prepare("SELECT claim_key,intent_kind,object_id,transaction_json FROM ledger_intents WHERE status='pending' ORDER BY created_at,claim_key")?;
        let rows = st.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
            ))
        })?;
        let mut v = Vec::new();
        for row in rows {
            v.push(row?)
        }
        v
    };
    for (ck, kind, object_id, tx_json) in intents {
        let tx: LedgerTransaction = serde_json::from_str(&tx_json)?;
        if claim_key(&tx) != ck {
            bail!("persisted intent claim mismatch for {ck}")
        }
        let existing = net.claim_quorum(&ck).await?;
        let entry_hash = if let Some(hash) = existing.entry_hash {
            hash
        } else {
            net.transact(tx).await?.entry.entry_hash
        };
        match kind.as_str() {
            "job_reserve" | "community_reserve" => {
                finalize_reservation_db(&mut conn, &object_id, &ck, &entry_hash)?
            }
            "provider_reward" => finalize_reward_db(&mut conn, &object_id, &ck, &entry_hash)?,
            other => bail!("unknown ledger intent kind {other}"),
        }
        println!("Recovered {kind} {object_id} at ledger entry {entry_hash}");
    }
    Ok(())
}

fn finalize_reservation_db(
    conn: &mut rusqlite::Connection,
    job_id: &str,
    claim: &str,
    entry_hash: &str,
) -> Result<()> {
    let tx = conn.transaction()?;
    db::certify_ledger_intent(&tx, claim, entry_hash)?;
    tx.execute(
        "UPDATE jobs SET status='pending' WHERE job_id=?1 AND status='funding'",
        params![job_id],
    )?;
    tx.execute(
        "UPDATE assignments SET status='pending' WHERE job_id=?1 AND status='blocked'",
        params![job_id],
    )?;
    tx.commit()?;
    Ok(())
}
fn finalize_reward_db(
    conn: &mut rusqlite::Connection,
    assignment_id: &str,
    claim: &str,
    entry_hash: &str,
) -> Result<()> {
    let tx = conn.transaction()?;
    db::certify_ledger_intent(&tx, claim, entry_hash)?;
    let job_id: Option<String> = tx
        .query_row(
            "SELECT job_id FROM assignments WHERE assignment_id=?1",
            params![assignment_id],
            |r| r.get(0),
        )
        .optional()?;
    let Some(job_id) = job_id else {
        bail!("recovery assignment missing: {assignment_id}")
    };
    tx.execute("UPDATE assignments SET status='completed',completed_at=?2,lease_until=NULL WHERE assignment_id=?1 AND status='settling'",params![assignment_id,now_unix()])?;
    let remaining: i64 = tx.query_row(
        "SELECT COUNT(*) FROM assignments WHERE job_id=?1 AND status!='completed'",
        params![job_id],
        |r| r.get(0),
    )?;
    if remaining == 0 {
        tx.execute(
            "UPDATE jobs SET status='completed',completed_at=?2 WHERE job_id=?1",
            params![job_id, now_unix()],
        )?;
    }
    tx.commit()?;
    Ok(())
}
