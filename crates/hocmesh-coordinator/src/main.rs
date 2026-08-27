mod api;
mod db;
mod error;
mod federation;
mod rebuild;
mod schedule;

use anyhow::{Context, Result, bail};
use api::AppState;
use clap::{Parser, Subcommand};
use hocmesh_core::compute::{split_work, work_cost_mcu};
use hocmesh_ledger::{
    network::LedgerNetwork,
    types::{
        COMMUNITY_ISSUANCE_ACCOUNT, InferenceExpiryEvidence, LedgerTransaction, Posting,
        TransactionEvidence, TransactionKind, ValidatorSet, ValidatorSignature, escrow_account,
        inference_holding_account,
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
        /// Path to a federation config, when this is one coordinator of
        /// several over the same ledger.
        #[arg(long)]
        federation: Option<String>,
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
            federation,
        } => serve(&listen, &db, validators.as_deref(), federation.as_deref()).await,
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
            let report = recover_pending(&db, &net).await?;
            println!("{report}");
            Ok(())
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
    // Build the intent before the job exists, so the two land together or not
    // at all. A job seeded without one is stranded: the reconciliation pass can
    // report it and nothing more.
    let intent = ledger_tx
        .as_ref()
        .map(|tx| Ok::<_, anyhow::Error>((claim_key(tx), serde_json::to_string(tx)?)))
        .transpose()?;
    db::seed_system_job_with_id(
        &mut conn,
        &job_id,
        work,
        shards,
        intent
            .as_ref()
            .map(|(ck, json)| (ck.as_str(), "community_reserve", json.as_str())),
    )?;
    if let Some(tx) = ledger_tx {
        let ck = claim_key(&tx);
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

async fn serve(
    listen: &str,
    db_path: &str,
    validators: Option<&str>,
    federation_path: Option<&str>,
) -> Result<()> {
    let ledger = match validators {
        Some(p) => Some(load_network(p)?),
        None => None,
    };
    let federation = match federation_path {
        Some(p) => Some(federation::Federation::load(p)?),
        None => None,
    };
    // A first pass before the door opens, so an operator sees what the previous
    // process left behind rather than discovering it a tick later.
    if let Some(net) = &ledger
        && let Err(e) = recover_pending(db_path, net).await
    {
        tracing::warn!(error=%e,"startup reconciliation could not read the coordinator database")
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
                // Only an unusable database gets here now: individual intents
                // are judged, logged, and left behind by the pass itself.
                if let Err(e) = recover_pending(&recovery_db, &net).await {
                    tracing::warn!(error=%e,"reconciliation pass could not read the coordinator database")
                }
            }
        });
    }
    // Two more loops when the deployment is federated. Neither one moves CU or
    // decides anything a peer could dispute: the first watches whether peers
    // answer, the second reads the chain that already holds the shared job
    // state. A coordinator that is wrong about either wastes effort; it cannot
    // pay twice, because the ledger's claim key rejects the second settlement
    // of a shard whichever coordinator sent it.
    if let Some(fed) = federation.clone() {
        let probe_db = db_path.to_string();
        tokio::spawn(async move {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .unwrap_or_default();
            let mut interval = tokio::time::interval(fed.probe_interval());
            loop {
                interval.tick().await;
                for (peer, transition) in fed.probe_all(&client, now_unix()).await {
                    match transition {
                        federation::Transition::WentDown => {
                            tracing::warn!(peer=%peer, "federation peer stopped answering");
                            // Ownership of that peer's jobs has just moved
                            // here. Its outstanding leases are shortened, not
                            // cancelled: a worker mid-shard is not at fault for
                            // a coordinator-to-coordinator link failing, so it
                            // keeps a grace window to report in before the
                            // shard is offered to anyone else.
                            match shorten_peer_leases(&probe_db, &peer) {
                                Ok(0) => {}
                                Ok(n) => tracing::info!(
                                    peer=%peer,
                                    leases=n,
                                    "shortened leases held by the unreachable peer"
                                ),
                                Err(e) => tracing::warn!(
                                    peer=%peer, error=%e,
                                    "could not shorten leases for the unreachable peer"
                                ),
                            }
                        }
                        federation::Transition::CameUp => {
                            tracing::info!(peer=%peer, "federation peer is answering again")
                        }
                    }
                }
            }
        });
    }
    if let (Some(net), Some(_)) = (ledger.clone(), federation.as_ref()) {
        let sync_db = db_path.to_string();
        // The replay holds a SQLite connection open across the network calls
        // that fetch each batch of certificates, so its future is not `Send`
        // and cannot be handed to the shared runtime. One thread with its own
        // single-threaded runtime says that plainly instead of contorting the
        // replay to hide it.
        std::thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(error=%e, "ledger sync thread could not start a runtime");
                    return;
                }
            };
            runtime.block_on(async move {
                let pool = match db::Pool::open(&sync_db) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!(error=%e, "ledger sync could not open the database");
                        return;
                    }
                };
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(20));
                loop {
                    interval.tick().await;
                    let mut conn = match pool.get() {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::warn!(error=%e, "ledger sync could not take a connection");
                            continue;
                        }
                    };
                    match rebuild::sync_from_ledger(&mut conn, &net, 256).await {
                        Ok(report) if report.entries > 0 => tracing::info!(
                            entries = report.entries,
                            jobs = report.jobs,
                            "adopted federated job state from the ledger"
                        ),
                        Ok(_) => {}
                        Err(e) => tracing::warn!(error=%e, "ledger sync pass failed"),
                    }
                }
            });
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
        federation,
    };
    let state_coordinator = state
        .federation
        .as_ref()
        .map(|f| f.coordinator_id().to_string());
    let app = api::router(state);
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .with_context(|| format!("binding coordinator to {listen}"))?;
    tracing::info!(
        listen=%listen,
        db=%db_path,
        ledger_mode=mode,
        coordinator=state_coordinator.as_deref().unwrap_or("standalone"),
        "hocMESH coordinator started"
    );
    axum::serve(listener, app)
        .await
        .context("coordinator server failed")?;
    Ok(())
}

/// Cut short the leases an unreachable peer handed out.
///
/// Opens its own connection because it runs from the probe loop, which holds
/// no pool. The deadline is a grace window rather than zero, so a worker that
/// is still running the shard can report in and be paid before the shard is
/// offered again.
fn shorten_peer_leases(db_path: &str, peer: &str) -> Result<usize> {
    /// How long a worker keeps a lease whose coordinator has gone away.
    const GRACE_SECONDS: i64 = 60;
    let pool = db::Pool::open(db_path)?;
    let conn = pool.get()?;
    db::shorten_leases_from(&conn, peer, now_unix() + GRACE_SECONDS)
}

/// What one reconciliation pass did.
///
/// Returned rather than only logged so the startup path, the tests, and the
/// operator view all read the same numbers. A pass that touched nothing and a
/// pass where everything is wedged look identical in a log line; they do not
/// look identical here.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct ReconcileReport {
    recovered: usize,
    deferred: usize,
    abandoned: usize,
    /// Coordinator work waiting on funding that no pending intent covers.
    ///
    /// Counted, never repaired: closing this gap locally would mean the
    /// coordinator deciding CU exists, which it has no standing to do.
    orphaned: usize,
    /// Received batches whose requester never judged them, returned to the
    /// commons now that their window has closed.
    expired: usize,
}

impl std::fmt::Display for ReconcileReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "reconciled: {} settled, {} deferred, {} parked, {} orphaned, {} expired",
            self.recovered, self.deferred, self.abandoned, self.orphaned, self.expired
        )
    }
}

/// Why one intent did not settle on this pass.
///
/// The distinction is the whole point of the daemon: a transient fault costs
/// one intent one tick, while a structural one would otherwise cost every
/// intent behind it, every tick, forever.
enum IntentFault {
    /// The network or the quorum was not ready. Nothing is wrong with the
    /// intent; it just needs another pass.
    Transient(String),
    /// The intent cannot settle under its own claim key no matter how long
    /// anyone waits.
    Terminal(String),
}

/// Settle one persisted intent, or say why it could not be settled.
///
/// Split out of the pass so that one broken intent costs exactly itself:
/// every early return here is one intent's verdict, not the daemon's.
async fn recover_one(
    conn: &mut rusqlite::Connection,
    net: &LedgerNetwork,
    ck: &str,
    kind: &str,
    object_id: &str,
    tx_json: &str,
) -> Result<String, IntentFault> {
    let tx: LedgerTransaction = serde_json::from_str(tx_json)
        .map_err(|e| IntentFault::Terminal(format!("persisted transaction is unreadable: {e}")))?;
    if claim_key(&tx) != ck {
        // The claim key is derived from the transaction, so a mismatch means the
        // two stopped describing the same thing. Retrying cannot make them agree
        // again, and settling anyway would file the CU under a key nobody looks
        // for.
        return Err(IntentFault::Terminal(format!(
            "persisted intent claim mismatch for {ck}"
        )));
    }
    if !matches!(
        kind,
        "job_reserve" | "community_reserve" | "provider_reward"
    ) {
        return Err(IntentFault::Terminal(format!(
            "unknown ledger intent kind {kind}"
        )));
    }
    let existing = net
        .claim_quorum(ck)
        .await
        .map_err(|e| IntentFault::Transient(format!("claim not confirmable yet: {e}")))?;
    let entry_hash = match existing.entry_hash {
        Some(hash) => hash,
        None => {
            net.transact(tx)
                .await
                .map_err(|e| IntentFault::Transient(format!("settlement did not go through: {e}")))?
                .entry
                .entry_hash
        }
    };
    // The entry exists on the chain by now either way, so a local write that
    // will not apply -- the assignment was pruned, say -- must not send us back
    // to propose it a second time. Retry the bookkeeping, not the settlement.
    let recorded = match kind {
        "provider_reward" => finalize_reward_db(conn, object_id, ck, &entry_hash),
        _ => finalize_reservation_db(conn, object_id, ck, &entry_hash),
    };
    recorded.map_err(|e| IntentFault::Transient(format!("settled but not recorded: {e}")))?;
    Ok(entry_hash)
}

/// One reconciliation pass over everything the coordinator has not settled.
///
/// Every intent is judged on its own. A pass never stops early, because the
/// intent that fails is rarely the intent that matters most, and the ones
/// queued behind it have done nothing wrong. Only a database the daemon
/// cannot read at all ends the pass.
async fn recover_pending(db_path: &str, net: &LedgerNetwork) -> Result<ReconcileReport> {
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
    let mut report = ReconcileReport::default();
    for (ck, kind, object_id, tx_json) in intents {
        match recover_one(&mut conn, net, &ck, &kind, &object_id, &tx_json).await {
            Ok(entry_hash) => {
                report.recovered += 1;
                println!("Recovered {kind} {object_id} at ledger entry {entry_hash}");
            }
            Err(IntentFault::Transient(why)) => {
                // Deferring forever is its own kind of stuck, so a fault that
                // never clears eventually gets parked like a structural one.
                let attempts = db::defer_ledger_intent(&conn, &ck, &why)?;
                if attempts >= db::MAX_INTENT_ATTEMPTS {
                    let why = format!("gave up after {attempts} attempts: {why}");
                    db::abandon_ledger_intent(&conn, &ck, &why)?;
                    report.abandoned += 1;
                    tracing::error!(claim=%ck, kind=%kind, object=%object_id, reason=%why, "ledger intent parked after repeated failures");
                } else {
                    report.deferred += 1;
                    tracing::debug!(claim=%ck, kind=%kind, attempts, reason=%why, "ledger intent deferred to a later pass");
                }
            }
            Err(IntentFault::Terminal(why)) => {
                db::abandon_ledger_intent(&conn, &ck, &why)?;
                report.abandoned += 1;
                tracing::error!(claim=%ck, kind=%kind, object=%object_id, reason=%why, "ledger intent cannot settle and was parked for an operator");
            }
        }
    }
    report.expired = sweep_expired_batches(&mut conn, net).await?;
    report.orphaned = db::orphaned_funding_objects(&conn)? as usize;
    if report.abandoned > 0 || report.orphaned > 0 {
        tracing::warn!(
            abandoned = report.abandoned,
            orphaned = report.orphaned,
            "coordinator and ledger disagree; run `hocmesh reconciliation` for the detail"
        );
    }
    Ok(report)
}

/// Return batches nobody ever judged to the commons.
///
/// A receipt is deliberately a one-way door: after it, the CU sits in a
/// holding account that only an acceptance or a dispute can empty, and both
/// need the requester's signature. That is what stops a requester reading an
/// answer and then refunding itself - and it is also why a requester that
/// simply stops answering freezes the provider's CU permanently.
///
/// This is the way out, and the reason it is safe is that it decides nothing.
/// The coordinator proposes; every validator re-derives the requester's own
/// receipt signature and measures the window against the timestamp inside it.
/// A coordinator that swept early, or swept a batch that was never received,
/// or pointed the CU anywhere but the commons, would simply be refused. The
/// same pass run by anybody else produces the same transaction.
async fn sweep_expired_batches(
    conn: &mut rusqlite::Connection,
    net: &LedgerNetwork,
) -> Result<usize> {
    let now = now_unix();
    let candidates = {
        let mut st = conn.prepare(
            "SELECT a.assignment_id, a.job_id, a.report_json, a.outputs_digest,
                    a.receipt_auth_json, n.public_key_b64
             FROM ai_assignments a
             JOIN ai_jobs j ON j.job_id = a.job_id
             JOIN nodes n ON n.node_id = j.requester_node_id
             WHERE a.receipted = 1 AND a.settled IS NULL
               AND a.report_json IS NOT NULL AND a.receipt_auth_json IS NOT NULL",
        )?;
        let rows = st.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, String>(5)?,
            ))
        })?;
        let mut v = Vec::new();
        for row in rows {
            v.push(row?)
        }
        v
    };
    let mut swept = 0usize;
    for (assignment_id, job_id, report_json, outputs_digest, receipt_json, requester_pk) in
        candidates
    {
        let report: hocmesh_ai::ReportInferenceRequest = match serde_json::from_str(&report_json) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(assignment=%assignment_id, error=%e, "unreadable batch report; not sweeping");
                continue;
            }
        };
        let receipt: hocmesh_protocol::AuthProof = match serde_json::from_str(&receipt_json) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(assignment=%assignment_id, error=%e, "unreadable receipt proof; not sweeping");
                continue;
            }
        };
        // The same arithmetic the validators will do. Checking it here saves a
        // round trip per unripe batch on every pass, and there will be far
        // more unripe batches than ripe ones.
        if now < receipt.timestamp + hocmesh_protocol::SETTLEMENT_WINDOW_SECS {
            continue;
        }
        let price = report.reward_mcu;
        let tx = LedgerTransaction {
            transaction_id: format!("expire_{assignment_id}"),
            kind: TransactionKind::InferenceExpiry,
            postings: vec![
                Posting {
                    account_id: inference_holding_account(&assignment_id),
                    delta_mcu: -price,
                },
                Posting {
                    account_id: COMMUNITY_ISSUANCE_ACCOUNT.to_string(),
                    delta_mcu: price,
                },
            ],
            evidence: TransactionEvidence::InferenceExpiry(InferenceExpiryEvidence {
                job_id: job_id.clone(),
                assignment_id: assignment_id.clone(),
                batch_start: report.batch_start,
                batch_end: report.batch_end,
                price_mcu: price,
                outputs_digest,
                requester_public_key_b64: requester_pk,
                requester_receipt: receipt,
            }),
            created_at: now,
        };
        // A batch already settled under this claim key by an acceptance or a
        // dispute is not an error here: it is the requester having come back
        // in the gap between the query and the proposal. Record what the chain
        // says and move on.
        let ck = claim_key(&tx);
        let settled = match net.claim_quorum(&ck).await {
            Ok(existing) if existing.entry_hash.is_some() => false,
            Ok(_) => match net.transact(tx).await {
                Ok(_) => true,
                Err(e) => {
                    tracing::debug!(assignment=%assignment_id, error=%e, "expiry sweep deferred to a later pass");
                    continue;
                }
            },
            Err(e) => {
                tracing::debug!(assignment=%assignment_id, error=%e, "expiry claim not confirmable yet");
                continue;
            }
        };
        conn.execute(
            "UPDATE ai_assignments SET settled='expired' WHERE assignment_id=?1 AND settled IS NULL",
            params![assignment_id],
        )?;
        if settled {
            swept += 1;
            tracing::info!(assignment=%assignment_id, job=%job_id, price_mcu=price, "batch abandoned by its requester returned to the commons");
        }
    }
    Ok(swept)
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
