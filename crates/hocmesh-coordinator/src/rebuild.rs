//! Rebuilding coordinator state from the ledger.
//!
//! The coordinator is a cache. Every fact it needs to schedule work is
//! already on the chain: a reservation names the job, its spec and its shard
//! count, and a reward or a refund names the shard it settled. Nothing here
//! is a source of truth, which is why a coordinator can be replaced.
//!
//! Rebuilding cannot mint CU or pay a shard twice. Assignment ids are derived
//! (`assignment_id(job_id, shard_index)`), so a rebuilt coordinator produces
//! exactly the ids the old one did, and the ledger already refuses a second
//! `reward:{assignment_id}` claim. The worst a rebuild can do is hand out work
//! that was already done -- wasted effort, never wasted CU.

use anyhow::{Context, Result};
use hocmesh_core::compute::{split_work, work_cost_mcu};
use hocmesh_ledger::network::LedgerNetwork;
use hocmesh_ledger::types::{TransactionEvidence, ValidatorSet};
use hocmesh_ledger::validate::{
    verify_certificate, verify_historical_evidence, verify_membership_change,
};
use hocmesh_protocol::{WorkSpec, assignment_id, node_id_from_public_key_b64, now_unix};
use rusqlite::{Connection, params};

/// What one rebuild put back.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RebuildReport {
    pub entries: u64,
    pub jobs: u64,
    pub settled_shards: u64,
    pub open_shards: u64,
}

/// Replay the chain into a coordinator database.
///
/// Idempotent by construction: every write is an upsert keyed on an id the
/// chain fixes, so running it twice, or against a database that is already
/// half-populated, converges on the same state.
pub async fn rebuild_from_ledger(
    conn: &mut Connection,
    net: &LedgerNetwork,
    batch: u64,
) -> Result<RebuildReport> {
    net.refresh_set().await.ok();
    let head = net
        .head_quorum()
        .await
        .context("asking the validator quorum for the head")?;
    let mut set: ValidatorSet = net.set();
    let mut report = RebuildReport::default();
    let mut next = 1u64;
    let batch = batch.max(1);
    while next <= head.sequence {
        let certs = net
            .fetch_certificates(next, batch, &set)
            .await
            .with_context(|| format!("fetching ledger entries from {next}"))?;
        if certs.is_empty() {
            anyhow::bail!(
                "validators report head {} but returned nothing at {next}",
                head.sequence
            );
        }
        for cert in certs {
            if cert.entry.sequence != next {
                anyhow::bail!(
                    "ledger returned entry {} where {next} was expected",
                    cert.entry.sequence
                );
            }
            // The chain is verified before it is believed, exactly as a
            // mirroring client verifies it. A coordinator that trusted an
            // unverified rebuild would be trusting whichever validator
            // answered first.
            verify_certificate(&cert, &set).context("verifying a rebuilt entry")?;
            let tx = conn.transaction()?;
            let mut advance: Option<ValidatorSet> = None;
            for t in &cert.entry.transactions {
                verify_historical_evidence(t, &cert.entry.previous_hash, &cert.signatures)?;
                if let TransactionEvidence::MembershipChange(e) = &t.evidence {
                    advance = Some(verify_membership_change(
                        advance.as_ref().unwrap_or(&set),
                        e,
                    )?);
                }
                apply_transaction(&tx, t, &mut report)?;
            }
            tx.commit()?;
            if let Some(s) = advance {
                set = s;
            }
            report.entries += 1;
            next = cert.entry.sequence + 1;
        }
    }
    finalize_jobs(conn)?;
    report.open_shards = count(
        conn,
        "SELECT COUNT(*) FROM assignments WHERE status='pending'",
    )?;
    Ok(report)
}

fn count(conn: &Connection, sql: &str) -> Result<u64> {
    Ok(conn.query_row(sql, [], |r| r.get::<_, i64>(0))? as u64)
}

/// Put one settled transaction back into the scheduling tables.
///
/// Balances are deliberately absent: in quorum mode the coordinator answers
/// balance queries from the validators, so replaying postings here would be
/// building a second, weaker copy of the thing the ledger already is.
fn apply_transaction(
    conn: &Connection,
    t: &hocmesh_ledger::types::LedgerTransaction,
    report: &mut RebuildReport,
) -> Result<()> {
    match &t.evidence {
        TransactionEvidence::JobReserve(e) => {
            let requester = requester_node(conn, &e.requester_public_key_b64)?;
            insert_job(
                conn,
                &e.job_id,
                Some(&requester),
                false,
                &e.work,
                e.shards,
                t.created_at,
            )?;
            report.jobs += 1;
        }
        TransactionEvidence::CommunityReserve {
            job_id,
            work,
            shards,
            ..
        } => {
            insert_job(conn, job_id, None, true, work, *shards, t.created_at)?;
            report.jobs += 1;
        }
        TransactionEvidence::ProviderReward(e) => {
            settle_shard(
                conn,
                &e.assignment_id,
                "completed",
                Some(&e.result),
                t.created_at,
            )?;
            report.settled_shards += 1;
        }
        TransactionEvidence::JobRefund(e) => {
            settle_shard(conn, &e.assignment_id, "refunded", None, t.created_at)?;
            report.settled_shards += 1;
        }
        // Inference jobs live in their own tables and settle against the
        // requester rather than a shard schedule, and membership changes are
        // the validators' business. Neither is coordinator scheduling state.
        _ => {}
    }
    Ok(())
}

/// Make sure the requester exists as a node row, since jobs reference one.
///
/// The row is inserted with `last_seen = 0` and placeholder capabilities: the
/// coordinator picks workers by freshness, so a rebuilt row is present for the
/// foreign key and invisible to the scheduler until the machine itself comes
/// back and re-advertises what it can actually do.
fn requester_node(conn: &Connection, public_key_b64: &str) -> Result<String> {
    let node_id = node_id_from_public_key_b64(public_key_b64)
        .context("a requester public key on the chain is not a valid Ed25519 key")?;
    let placeholder = serde_json::json!({
        "protocol_version": hocmesh_protocol::PROTOCOL_VERSION,
        "hostname": "",
        "os": "",
        "arch": "",
        "cpu_brand": "",
        "logical_cpus": 0,
        "total_memory_bytes": 0,
        "cpu_benchmark_score": 0,
        "gpus": [],
    })
    .to_string();
    conn.execute(
        "INSERT INTO nodes(node_id,public_key_b64,capabilities_json,registered_at,last_seen) \
         VALUES(?1,?2,?3,?4,0) ON CONFLICT(node_id) DO NOTHING",
        params![node_id, public_key_b64, placeholder, now_unix()],
    )?;
    Ok(node_id)
}

/// Recreate a job and its shard schedule from one reservation.
///
/// The shards come from `split_work`, the same call the original coordinator
/// made, so the recreated shard boundaries are the original ones. If they were
/// not, an assignment id would still match while the work behind it differed --
/// which is how a rebuild could pay for work nobody asked for.
fn insert_job(
    conn: &Connection,
    job_id: &str,
    requester: Option<&str>,
    system_funded: bool,
    work: &WorkSpec,
    shards: u32,
    created_at: i64,
) -> Result<()> {
    let parts = split_work(work, shards);
    let total: i64 = parts.iter().map(work_cost_mcu).sum();
    conn.execute(
        "INSERT INTO jobs(job_id,requester_node_id,system_funded,work_json,status,reserved_mcu,created_at) \
         VALUES(?1,?2,?3,?4,'pending',?5,?6) ON CONFLICT(job_id) DO NOTHING",
        params![
            job_id,
            requester,
            i64::from(system_funded),
            serde_json::to_string(work)?,
            total,
            created_at
        ],
    )?;
    for (index, part) in parts.iter().enumerate() {
        let index = index as u32;
        conn.execute(
            "INSERT INTO assignments(assignment_id,job_id,shard_index,work_json,status,reward_mcu) \
             VALUES(?1,?2,?3,?4,'pending',?5) ON CONFLICT(assignment_id) DO NOTHING",
            params![
                assignment_id(job_id, index),
                job_id,
                index,
                serde_json::to_string(part)?,
                work_cost_mcu(part)
            ],
        )?;
    }
    Ok(())
}

/// Mark one shard as already settled on the chain.
///
/// A rebuild can see a reward for a shard whose reservation it has not reached
/// yet only if entries arrive out of order, which the sequence check above
/// forbids. So an update that touches nothing means the chain named a shard no
/// reservation created, and that is worth failing on rather than papering over.
fn settle_shard(
    conn: &Connection,
    assignment_id: &str,
    status: &str,
    result: Option<&hocmesh_protocol::WorkResult>,
    completed_at: i64,
) -> Result<()> {
    let result_json = match result {
        Some(r) => Some(serde_json::to_string(r)?),
        None => None,
    };
    let touched = conn.execute(
        "UPDATE assignments SET status=?2,result_json=COALESCE(?3,result_json),\
         completed_at=?4,leased_to=NULL,lease_until=NULL WHERE assignment_id=?1",
        params![assignment_id, status, result_json, completed_at],
    )?;
    if touched == 0 {
        anyhow::bail!("ledger settles {assignment_id}, which no reservation created");
    }
    Ok(())
}

/// Close out every job whose shards have all settled.
///
/// Status is derived, not remembered: a job is finished when nothing about it
/// is still open, which is a question the assignment rows answer directly.
fn finalize_jobs(conn: &Connection) -> Result<()> {
    conn.execute(
        "UPDATE jobs SET status='completed', completed_at=(SELECT MAX(completed_at) FROM assignments a \
         WHERE a.job_id=jobs.job_id) WHERE NOT EXISTS (SELECT 1 FROM assignments a \
         WHERE a.job_id=jobs.job_id AND a.status NOT IN ('completed','refunded'))",
        [],
    )?;
    Ok(())
}
