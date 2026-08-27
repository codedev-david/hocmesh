mod api;
mod db;
mod error;
mod rebuild;

use anyhow::{Context, Result, bail};
use api::AppState;
use clap::{Parser, Subcommand};
use hocmesh_core::compute::{split_work, work_cost_mcu};
use hocmesh_ledger::{
    network::LedgerNetwork,
    types::{
        COMMUNITY_ISSUANCE_ACCOUNT, LedgerTransaction, Posting, TransactionEvidence,
        TransactionKind, ValidatorSet, ValidatorSignature, escrow_account,
    },
    validate::claim_key,
};
use hocmesh_protocol::{WorkSpec, now_unix};
use rusqlite::{OptionalExtension, params};
use std::{fs, sync::Arc};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

#[derive(Debug, Parser)]
#[command(
    name = "hocmesh-coordinator",
    version,
    about = "hocMESH scheduling/control plane"
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
        #[arg(long, default_value = "hocmesh.db")]
        db: String,
        #[arg(long)]
        validators: Option<String>,
    },
    Seed {
        #[arg(long, default_value = "hocmesh.db")]
        db: String,
        #[arg(long, default_value_t = 2)]
        start: u64,
        #[arg(long, default_value_t = 5_000_000)]
        end: u64,
        #[arg(long, default_value_t = 32)]
        shards: u32,
        #[arg(long)]
        validators: Option<String>,
        /// Sponsorships from the sitting validator set, as a JSON array.
        ///
        /// Required whenever a ledger is configured: minting is the set's
        /// decision, not the coordinator's, so the coordinator can only carry
        /// signatures it was handed.
        #[arg(long)]
        sponsors: Option<String>,
        /// Fix the job id so sponsors can sign it before it exists.
        #[arg(long)]
        job_id: Option<String>,
    },
    /// Rebuild scheduling state from the chain into a fresh database.
    ///
    /// This is how a coordinator is replaced. Nothing here is authoritative:
    /// the reservations, rewards and refunds on the ledger already say what
    /// every job is and which of its shards are settled.
    Rebuild {
        #[arg(long, default_value = "hocmesh.db")]
        db: String,
        #[arg(long)]
        validators: String,
        #[arg(long, default_value_t = 256)]
        batch: u64,
    },
    Recover {
        #[arg(long, default_value = "hocmesh.db")]
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
            sponsors,
            job_id,
        } => {
            seed(
                &db,
                WorkSpec::PrimeCount { start, end },
                shards,
                validators.as_deref(),
                sponsors.as_deref(),
                job_id.as_deref(),
            )
            .await
        }
        Command::Recover { db, validators } => {
            let net = load_network(&validators)?;
            recover_pending(&db, &net).await
        }
        Command::Rebuild {
            db,
            validators,
            batch,
        } => {
            let net = load_network(&validators)?;
            let mut conn = db::open(&db)?;
            let report = rebuild::rebuild_from_ledger(&mut conn, &net, batch).await?;
            println!(
                "Replayed {} entries: {} jobs, {} shards already settled, {} still open",
                report.entries, report.jobs, report.settled_shards, report.open_shards
            );
            Ok(())
        }
    }
}
fn load_network(path: &str) -> Result<LedgerNetwork> {
    let set: ValidatorSet = serde_json::from_str(
        &fs::read_to_string(path).with_context(|| format!("reading validator set {path}"))?,
    )?;
    LedgerNetwork::new(set)
}

async fn seed(
    db_path: &str,
    work: WorkSpec,
    shards: u32,
    validators: Option<&str>,
    sponsors: Option<&str>,
    job_id: Option<&str>,
) -> Result<()> {
    work.validate().map_err(anyhow::Error::msg)?;
    let shards = shards.clamp(1, 256);
    let job_id = job_id
        .map(str::to_string)
        .unwrap_or_else(|| format!("job_community_{}", Uuid::new_v4().simple()));
    let network = validators.map(load_network).transpose()?;
    // A mint with no sponsors would be rejected by every validator anyway, so
    // say so here rather than let the operator find out from a quorum failure.
    let sponsors: Vec<ValidatorSignature> = match sponsors {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None if network.is_some() => bail!(
            "a community mint needs sponsorships from the sitting validator set; collect them with `hocmesh community-vouch` and pass --sponsors"
        ),
        None => Vec::new(),
    };
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
                sponsors: sponsors.clone(),
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
                // Pick the set up before settling against it rather than
                // after. The ledger client recovers on its own failure path
                // too, but only once a settlement has already been held up.
                match net.refresh_set().await {
                    Ok(true) => tracing::info!(
                        validators = net.set().members.len(),
                        "validator set advanced by a certified membership change"
                    ),
                    Ok(false) => {}
                    Err(e) => tracing::warn!(error=%e, "validator set refresh failed"),
                }
                if let Err(e) = recover_pending(&recovery_db, &net).await {
                    tracing::warn!(error=%e,"background ledger intent recovery incomplete")
                }
            }
        });
    }
    let pool = db::Pool::open(db_path)?;
    let mode = if ledger.is_some() {
        "quorum"
    } else {
        "local-mvp"
    };
    let state = AppState {
        db: Arc::new(pool),
        ledger,
    };
    let app = api::router(state);
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .with_context(|| format!("binding coordinator to {listen}"))?;
    tracing::info!(listen=%listen,db=%db_path,ledger_mode=mode,"hocMESH coordinator started");
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
        "SELECT COUNT(*) FROM assignments WHERE job_id=?1 AND status NOT IN ('completed','refunded')",
        params![job_id],
        |r| r.get(0),
    )?;
    if remaining == 0 {
        // A job that had a shard refunded closed short of what it asked for,
        // and the recovery path has to say so for the same reason the live
        // one does: a partial result set should never read as a whole one.
        let refunded: i64 = tx.query_row(
            "SELECT COUNT(*) FROM assignments WHERE job_id=?1 AND status='refunded'",
            params![job_id],
            |r| r.get(0),
        )?;
        let status = if refunded > 0 { "closed" } else { "completed" };
        tx.execute(
            "UPDATE jobs SET status=?2,completed_at=?3 WHERE job_id=?1",
            params![job_id, status, now_unix()],
        )?;
    }
    tx.commit()?;
    Ok(())
}
