use crate::error::ApiError;
use axum::{
    Json, Router,
    extract::{Path, State},
    routing::{get, post},
};
use mesh_core::compute::{split_work, verify_work, work_cost_mcu};
use mesh_ledger::{
    network::LedgerNetwork,
    types::{
        JobReserveEvidence, LedgerTransaction, Posting, ProviderRewardEvidence,
        TransactionEvidence, TransactionKind, escrow_account,
    },
    validate::claim_key,
};
use mesh_protocol::{
    BalanceResponse, DEFAULT_LEASE_SECONDS, HeartbeatRequest, JobStatusResponse,
    NetworkStatsResponse, NodeCapabilities, NodeStatusResponse, PollRequest, PollResponse,
    RegisterRequest, RegisterResponse, ResultRequest, ResultResponse, SubmitJobRequest,
    SubmitJobResponse, WorkAssignment, WorkResult, WorkSpec, empty_body_hash, heartbeat_body_hash,
    job_id_from_auth, now_unix, register_body_hash, result_body_hash, submit_body_hash,
    verify_auth,
};
use rusqlite::{Connection, OptionalExtension, params};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Mutex<Connection>>,
    pub ledger: Option<LedgerNetwork>,
}

type AssignmentSettlementRow = (String, String, i64, i64, String, Option<String>, i64);

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/nodes/register", post(register))
        .route("/v1/nodes/heartbeat", post(heartbeat))
        .route("/v1/work/poll", post(poll_work))
        .route("/v1/work/result", post(report_result))
        .route("/v1/jobs/submit", post(submit_job))
        .route("/v1/jobs/{id}", get(job_status))
        .route("/v1/nodes/{id}/balance", get(balance))
        .route("/v1/nodes/{id}", get(node_status))
        .route("/v1/network/stats", get(network_stats))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

async fn register(
    State(state): State<AppState>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<RegisterResponse>, ApiError> {
    if req.capabilities.protocol_version != mesh_protocol::PROTOCOL_VERSION {
        return Err(ApiError::bad_request("unsupported protocol version"));
    }
    let body_hash =
        register_body_hash(&req.public_key_b64, &req.capabilities).map_err(ApiError::internal)?;
    verify_auth(&req.public_key_b64, &req.auth, "register", &body_hash)
        .map_err(ApiError::unauthorized)?;
    let caps_json = serde_json::to_string(&req.capabilities).map_err(ApiError::internal)?;
    let now = now_unix();
    {
        let mut conn = state.db.lock().map_err(ApiError::internal)?;
        consume_nonce(&conn, &req.auth)?;
        let tx = conn.transaction().map_err(ApiError::internal)?;
        tx.execute(r#"INSERT INTO nodes(node_id,public_key_b64,capabilities_json,registered_at,last_seen)
            VALUES(?1,?2,?3,?4,?4) ON CONFLICT(node_id) DO UPDATE SET public_key_b64=excluded.public_key_b64,capabilities_json=excluded.capabilities_json,last_seen=excluded.last_seen"#,
            params![req.auth.node_id, req.public_key_b64, caps_json, now]).map_err(ApiError::internal)?;
        tx.execute(
            "INSERT OR IGNORE INTO balances(node_id,balance_mcu) VALUES(?1,0)",
            params![req.auth.node_id],
        )
        .map_err(ApiError::internal)?;
        tx.commit().map_err(ApiError::internal)?;
    }
    let balance = authoritative_balance(&state, &req.auth.node_id)
        .await?
        .balance_mcu;
    cache_balance(&state, &req.auth.node_id, balance)?;
    Ok(Json(RegisterResponse {
        node_id: req.auth.node_id,
        balance_mcu: balance,
        protocol_version: mesh_protocol::PROTOCOL_VERSION,
        ledger_mode: ledger_mode(&state).into(),
    }))
}

async fn heartbeat(
    State(state): State<AppState>,
    Json(req): Json<HeartbeatRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let body_hash = heartbeat_body_hash(&req.capabilities).map_err(ApiError::internal)?;
    let conn = state.db.lock().map_err(ApiError::internal)?;
    authenticate_known_node(&conn, &req.auth, "heartbeat", &body_hash)?;
    let caps_json = serde_json::to_string(&req.capabilities).map_err(ApiError::internal)?;
    conn.execute(
        "UPDATE nodes SET last_seen=?2,capabilities_json=?3 WHERE node_id=?1",
        params![req.auth.node_id, now_unix(), caps_json],
    )
    .map_err(ApiError::internal)?;
    Ok(Json(serde_json::json!({"ok":true})))
}

async fn poll_work(
    State(state): State<AppState>,
    Json(req): Json<PollRequest>,
) -> Result<Json<PollResponse>, ApiError> {
    let conn = state.db.lock().map_err(ApiError::internal)?;
    authenticate_known_node(&conn, &req.auth, "poll", &empty_body_hash())?;
    let now = now_unix();
    conn.execute("UPDATE assignments SET status='pending',leased_to=NULL,lease_until=NULL WHERE status='leased' AND lease_until < ?1", params![now]).map_err(ApiError::internal)?;
    conn.execute(
        "UPDATE nodes SET last_seen=?2 WHERE node_id=?1",
        params![req.auth.node_id, now],
    )
    .map_err(ApiError::internal)?;
    let candidate: Option<(String,String,i64,String,i64,i64)> = conn.query_row(
        "SELECT a.assignment_id,a.job_id,a.shard_index,a.work_json,a.reward_mcu,j.system_funded FROM assignments a JOIN jobs j ON j.job_id=a.job_id WHERE a.status='pending' AND (j.requester_node_id IS NULL OR j.requester_node_id != ?1) ORDER BY a.rowid LIMIT 1",
        params![req.auth.node_id], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?))).optional().map_err(ApiError::internal)?;
    let Some((assignment_id, job_id, shard_index, work_json, reward_mcu, system_funded)) =
        candidate
    else {
        return Ok(Json(PollResponse { assignment: None }));
    };
    let lease_until = now + DEFAULT_LEASE_SECONDS;
    let updated=conn.execute("UPDATE assignments SET status='leased',leased_to=?2,lease_until=?3 WHERE assignment_id=?1 AND status='pending'",params![assignment_id,req.auth.node_id,lease_until]).map_err(ApiError::internal)?;
    if updated == 0 {
        return Ok(Json(PollResponse { assignment: None }));
    }
    let work: WorkSpec = serde_json::from_str(&work_json).map_err(ApiError::internal)?;
    Ok(Json(PollResponse {
        assignment: Some(WorkAssignment {
            assignment_id,
            job_id,
            shard_index: shard_index as u32,
            work,
            reward_mcu,
            lease_seconds: DEFAULT_LEASE_SECONDS,
            system_funded: system_funded != 0,
        }),
    }))
}

async fn report_result(
    State(state): State<AppState>,
    Json(req): Json<ResultRequest>,
) -> Result<Json<ResultResponse>, ApiError> {
    let body_hash = result_body_hash(
        &req.assignment_id,
        &req.job_id,
        req.shard_index,
        &req.work,
        req.reward_mcu,
        req.system_funded,
        &req.result,
    )
    .map_err(ApiError::internal)?;
    let (work, job_id, reward_mcu, system_funded, provider_pk) = {
        let conn = state.db.lock().map_err(ApiError::internal)?;
        authenticate_known_node(&conn, &req.auth, "result", &body_hash)?;
        let provider_pk: String = conn
            .query_row(
                "SELECT public_key_b64 FROM nodes WHERE node_id=?1",
                params![req.auth.node_id],
                |r| r.get(0),
            )
            .map_err(ApiError::internal)?;
        let row: Option<AssignmentSettlementRow> = conn
            .query_row(
                "SELECT a.work_json,a.job_id,a.shard_index,a.reward_mcu,a.status,a.leased_to,j.system_funded FROM assignments a JOIN jobs j ON j.job_id=a.job_id WHERE a.assignment_id=?1",
                params![req.assignment_id],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                    ))
                },
            )
            .optional()
            .map_err(ApiError::internal)?;
        let Some((work_json, job_id, db_shard_index, reward_mcu, status, leased_to, system_funded)) =
            row
        else {
            return Err(ApiError::not_found("assignment not found"));
        };
        if status != "leased" || leased_to.as_deref() != Some(req.auth.node_id.as_str()) {
            return Err(ApiError::conflict("assignment is not leased to this node"));
        }
        let db_work: WorkSpec = serde_json::from_str(&work_json).map_err(ApiError::internal)?;
        let sf = system_funded != 0;
        if req.job_id != job_id
            || req.shard_index != db_shard_index as u32
            || req.work != db_work
            || req.reward_mcu != reward_mcu
            || req.system_funded != sf
        {
            return Err(ApiError::conflict(
                "signed result metadata does not match leased assignment",
            ));
        };
        (db_work, job_id, reward_mcu, sf, provider_pk)
    };
    if !verify_work(&work, &req.result) {
        let conn = state.db.lock().map_err(ApiError::internal)?;
        conn.execute("UPDATE assignments SET status='pending',leased_to=NULL,lease_until=NULL WHERE assignment_id=?1",params![req.assignment_id]).map_err(ApiError::internal)?;
        return Err(ApiError::conflict(
            "work verification failed; assignment returned to queue",
        ));
    }

    let mut ledger_hash = None;
    if let Some(ledger) = &state.ledger {
        let tx_record = LedgerTransaction {
            transaction_id: format!("reward_{}", req.assignment_id),
            kind: TransactionKind::ProviderReward,
            postings: vec![
                Posting {
                    account_id: escrow_account(&job_id),
                    delta_mcu: -reward_mcu,
                },
                Posting {
                    account_id: req.auth.node_id.clone(),
                    delta_mcu: reward_mcu,
                },
            ],
            evidence: TransactionEvidence::ProviderReward(ProviderRewardEvidence {
                job_id: job_id.clone(),
                assignment_id: req.assignment_id.clone(),
                shard_index: req.shard_index,
                reward_mcu,
                provider_public_key_b64: provider_pk,
                provider_auth: req.auth.clone(),
                work: work.clone(),
                result: req.result.clone(),
                system_funded,
            }),
            created_at: now_unix(),
        };
        let ck = claim_key(&tx_record);
        let tx_json = serde_json::to_string(&tx_record).map_err(ApiError::internal)?;
        let result_json = serde_json::to_string(&req.result).map_err(ApiError::internal)?;
        {
            let mut conn = state.db.lock().map_err(ApiError::internal)?;
            let local = conn.transaction().map_err(ApiError::internal)?;
            let updated=local.execute("UPDATE assignments SET status='settling',result_json=?2,lease_until=NULL WHERE assignment_id=?1 AND status='leased' AND leased_to=?3",params![req.assignment_id,result_json,req.auth.node_id]).map_err(ApiError::internal)?;
            if updated == 0 {
                return Err(ApiError::conflict(
                    "assignment changed before settlement intent was persisted",
                ));
            }
            crate::db::persist_ledger_intent(
                &local,
                &ck,
                "provider_reward",
                &req.assignment_id,
                &tx_json,
            )
            .map_err(ApiError::internal)?;
            local.commit().map_err(ApiError::internal)?;
        }
        let cert = ledger.transact(tx_record).await.map_err(|e| {
            ApiError::conflict(format!(
                "reward settlement pending recovery after ledger error: {e}"
            ))
        })?;
        ledger_hash = Some(cert.entry.entry_hash.clone());
        finalize_reward(&state, &req.assignment_id, &ck, &cert.entry.entry_hash)?;
    } else {
        let mut conn = state.db.lock().map_err(ApiError::internal)?;
        let tx = conn.transaction().map_err(ApiError::internal)?;
        let updated=tx.execute("UPDATE assignments SET status='completed',result_json=?2,completed_at=?3,lease_until=NULL WHERE assignment_id=?1 AND status='leased' AND leased_to=?4",params![req.assignment_id,serde_json::to_string(&req.result).map_err(ApiError::internal)?,now_unix(),req.auth.node_id]).map_err(ApiError::internal)?;
        if updated == 0 {
            return Err(ApiError::conflict(
                "assignment changed before result settlement",
            ));
        }
        apply_ledger_delta(
            &tx,
            &req.auth.node_id,
            reward_mcu,
            "contribution",
            Some(&job_id),
            Some(&req.assignment_id),
        )?;
        let remaining: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM assignments WHERE job_id=?1 AND status!='completed'",
                params![job_id],
                |r| r.get(0),
            )
            .map_err(ApiError::internal)?;
        if remaining == 0 {
            tx.execute(
                "UPDATE jobs SET status='completed',completed_at=?2 WHERE job_id=?1",
                params![job_id, now_unix()],
            )
            .map_err(ApiError::internal)?;
        }
        tx.commit().map_err(ApiError::internal)?;
    }
    let job_completed = {
        let conn = state.db.lock().map_err(ApiError::internal)?;
        conn.query_row(
            "SELECT status='completed' FROM jobs WHERE job_id=?1",
            params![job_id],
            |r| r.get::<_, bool>(0),
        )
        .map_err(ApiError::internal)?
    };
    let bal = authoritative_balance(&state, &req.auth.node_id).await?;
    cache_balance(&state, &req.auth.node_id, bal.balance_mcu)?;
    Ok(Json(ResultResponse {
        accepted: true,
        reward_mcu,
        balance_mcu: bal.balance_mcu,
        job_completed,
        ledger_entry_hash: ledger_hash,
    }))
}
async fn submit_job(
    State(state): State<AppState>,
    Json(req): Json<SubmitJobRequest>,
) -> Result<Json<SubmitJobResponse>, ApiError> {
    req.work.validate().map_err(ApiError::bad_request)?;
    if !(1..=256).contains(&req.shards) {
        return Err(ApiError::bad_request("shards must be between 1 and 256"));
    }
    let body_hash = submit_body_hash(&req.work, req.shards).map_err(ApiError::internal)?;
    let parts = split_work(&req.work, req.shards);
    let total_cost_mcu: i64 = parts.iter().map(work_cost_mcu).sum();
    let job_id = job_id_from_auth(&req.auth);
    let requester_pk = {
        let conn = state.db.lock().map_err(ApiError::internal)?;
        authenticate_known_node(&conn, &req.auth, "submit", &body_hash)?;
        if conn
            .query_row("SELECT 1 FROM jobs WHERE job_id=?1", params![job_id], |r| {
                r.get::<_, i64>(0)
            })
            .optional()
            .map_err(ApiError::internal)?
            .is_some()
        {
            return Err(ApiError::conflict(
                "this signed submit request was already used",
            ));
        }
        conn.query_row(
            "SELECT public_key_b64 FROM nodes WHERE node_id=?1",
            params![req.auth.node_id],
            |r| r.get::<_, String>(0),
        )
        .map_err(ApiError::internal)?
    };
    let bal = authoritative_balance(&state, &req.auth.node_id).await?;
    if bal.balance_mcu < total_cost_mcu {
        return Err(ApiError::conflict(format!(
            "insufficient compute credit: need {:.3} CU, have {:.3} CU; contribute first",
            total_cost_mcu as f64 / 1000.0,
            bal.balance_mcu as f64 / 1000.0
        )));
    }

    let ledger_tx = state.ledger.as_ref().map(|_| LedgerTransaction {
        transaction_id: format!("reserve_{job_id}"),
        kind: TransactionKind::JobReserve,
        postings: vec![
            Posting {
                account_id: req.auth.node_id.clone(),
                delta_mcu: -total_cost_mcu,
            },
            Posting {
                account_id: escrow_account(&job_id),
                delta_mcu: total_cost_mcu,
            },
        ],
        evidence: TransactionEvidence::JobReserve(JobReserveEvidence {
            job_id: job_id.clone(),
            requester_public_key_b64: requester_pk,
            requester_auth: req.auth.clone(),
            work: req.work.clone(),
            shards: req.shards,
        }),
        created_at: now_unix(),
    });
    let mut ledger_hash = None;
    {
        let mut conn = state.db.lock().map_err(ApiError::internal)?;
        let local = conn.transaction().map_err(ApiError::internal)?;
        let now = now_unix();
        let ready = ledger_tx.is_none();
        let job_status = if ready { "pending" } else { "funding" };
        let assignment_status = if ready { "pending" } else { "blocked" };
        local.execute("INSERT INTO jobs(job_id,requester_node_id,system_funded,work_json,status,reserved_mcu,created_at) VALUES(?1,?2,0,?3,?4,?5,?6)",params![job_id,req.auth.node_id,serde_json::to_string(&req.work).map_err(ApiError::internal)?,job_status,total_cost_mcu,now]).map_err(ApiError::internal)?;
        if ledger_tx.is_none() {
            apply_ledger_delta(
                &local,
                &req.auth.node_id,
                -total_cost_mcu,
                "consumption_reservation",
                Some(&job_id),
                None,
            )?;
        }
        for (index, part) in parts.iter().enumerate() {
            let assignment_id = mesh_protocol::assignment_id(&job_id, index as u32);
            local.execute("INSERT INTO assignments(assignment_id,job_id,shard_index,work_json,status,reward_mcu) VALUES(?1,?2,?3,?4,?5,?6)",params![assignment_id,job_id,index as i64,serde_json::to_string(part).map_err(ApiError::internal)?,assignment_status,work_cost_mcu(part)]).map_err(ApiError::internal)?;
        }
        if let Some(tx_record) = &ledger_tx {
            let ck = claim_key(tx_record);
            crate::db::persist_ledger_intent(
                &local,
                &ck,
                "job_reserve",
                &job_id,
                &serde_json::to_string(tx_record).map_err(ApiError::internal)?,
            )
            .map_err(ApiError::internal)?;
        }
        local.commit().map_err(ApiError::internal)?;
    }
    if let (Some(ledger), Some(tx_record)) = (&state.ledger, ledger_tx) {
        let ck = claim_key(&tx_record);
        let cert = ledger.transact(tx_record).await.map_err(|e| {
            ApiError::conflict(format!(
                "job funding pending recovery after ledger error: {e}"
            ))
        })?;
        ledger_hash = Some(cert.entry.entry_hash.clone());
        finalize_reservation(&state, &job_id, &ck, &cert.entry.entry_hash)?;
    }
    let new_bal = authoritative_balance(&state, &req.auth.node_id).await?;
    cache_balance(&state, &req.auth.node_id, new_bal.balance_mcu)?;
    Ok(Json(SubmitJobResponse {
        job_id,
        reserved_mcu: total_cost_mcu,
        balance_mcu: new_bal.balance_mcu,
        assignments: parts.len() as u32,
        ledger_entry_hash: ledger_hash,
    }))
}
async fn balance(
    State(state): State<AppState>,
    Path(node_id): Path<String>,
) -> Result<Json<BalanceResponse>, ApiError> {
    {
        let conn = state.db.lock().map_err(ApiError::internal)?;
        if conn
            .query_row(
                "SELECT 1 FROM nodes WHERE node_id=?1",
                params![node_id],
                |r| r.get::<_, i64>(0),
            )
            .optional()
            .map_err(ApiError::internal)?
            .is_none()
        {
            return Err(ApiError::not_found("node not found"));
        }
    }
    authoritative_balance(&state, &node_id).await.map(Json)
}

async fn job_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobStatusResponse>, ApiError> {
    let conn = state.db.lock().map_err(ApiError::internal)?;
    let job: Option<(Option<String>, i64, String, i64)> = conn
        .query_row(
            "SELECT requester_node_id,system_funded,status,reserved_mcu FROM jobs WHERE job_id=?1",
            params![job_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .optional()
        .map_err(ApiError::internal)?;
    let Some((requester_node_id, system_funded, status, reserved_mcu)) = job else {
        return Err(ApiError::not_found("job not found"));
    };
    let total: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM assignments WHERE job_id=?1",
            params![job_id],
            |r| r.get(0),
        )
        .map_err(ApiError::internal)?;
    let completed: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM assignments WHERE job_id=?1 AND status='completed'",
            params![job_id],
            |r| r.get(0),
        )
        .map_err(ApiError::internal)?;
    let mut prime_total = 0u64;
    let mut has = false;
    let mut stmt=conn.prepare("SELECT result_json FROM assignments WHERE job_id=?1 AND status='completed' ORDER BY shard_index").map_err(ApiError::internal)?;
    let rows = stmt
        .query_map(params![job_id], |r| r.get::<_, String>(0))
        .map_err(ApiError::internal)?;
    for row in rows {
        match serde_json::from_str::<WorkResult>(&row.map_err(ApiError::internal)?)
            .map_err(ApiError::internal)?
        {
            WorkResult::PrimeCount { count, .. } => {
                has = true;
                prime_total = prime_total.saturating_add(count)
            }
        }
    }
    Ok(Json(JobStatusResponse {
        job_id,
        requester_node_id,
        system_funded: system_funded != 0,
        status,
        total_assignments: total as u32,
        completed_assignments: completed as u32,
        reserved_mcu,
        prime_count_total: has.then_some(prime_total),
    }))
}

async fn node_status(
    State(state): State<AppState>,
    Path(node_id): Path<String>,
) -> Result<Json<NodeStatusResponse>, ApiError> {
    let conn = state.db.lock().map_err(ApiError::internal)?;
    let row: Option<(i64, String)> = conn
        .query_row(
            "SELECT last_seen,capabilities_json FROM nodes WHERE node_id=?1",
            params![node_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
        .map_err(ApiError::internal)?;
    let Some((last_seen_unix, caps_json)) = row else {
        return Err(ApiError::not_found("node not found"));
    };
    let capabilities: NodeCapabilities =
        serde_json::from_str(&caps_json).map_err(ApiError::internal)?;
    Ok(Json(NodeStatusResponse {
        node_id,
        last_seen_unix,
        online: now_unix() - last_seen_unix <= 30,
        capabilities,
    }))
}

async fn network_stats(
    State(state): State<AppState>,
) -> Result<Json<NetworkStatsResponse>, ApiError> {
    let conn = state.db.lock().map_err(ApiError::internal)?;
    let registered_nodes = scalar_u64(&conn, "SELECT COUNT(*) FROM nodes")?;
    let online_nodes = conn
        .query_row(
            "SELECT COUNT(*) FROM nodes WHERE last_seen>=?1",
            params![now_unix() - 30],
            |r| r.get::<_, i64>(0),
        )
        .map_err(ApiError::internal)? as u64;
    let pending_assignments = scalar_u64(
        &conn,
        "SELECT COUNT(*) FROM assignments WHERE status='pending'",
    )?;
    let leased_assignments = scalar_u64(
        &conn,
        "SELECT COUNT(*) FROM assignments WHERE status='leased'",
    )?;
    let completed_assignments = scalar_u64(
        &conn,
        "SELECT COUNT(*) FROM assignments WHERE status='completed'",
    )?;
    let total_available_mcu: i64 = conn
        .query_row(
            "SELECT COALESCE(SUM(balance_mcu),0) FROM balances",
            [],
            |r| r.get(0),
        )
        .map_err(ApiError::internal)?;
    Ok(Json(NetworkStatsResponse {
        registered_nodes,
        online_nodes,
        pending_assignments,
        leased_assignments,
        completed_assignments,
        total_available_mcu,
        ledger_mode: ledger_mode(&state).into(),
    }))
}

fn finalize_reservation(
    state: &AppState,
    job_id: &str,
    claim: &str,
    entry_hash: &str,
) -> Result<(), ApiError> {
    let mut conn = state.db.lock().map_err(ApiError::internal)?;
    let tx = conn.transaction().map_err(ApiError::internal)?;
    crate::db::certify_ledger_intent(&tx, claim, entry_hash).map_err(ApiError::internal)?;
    tx.execute(
        "UPDATE jobs SET status='pending' WHERE job_id=?1 AND status='funding'",
        params![job_id],
    )
    .map_err(ApiError::internal)?;
    tx.execute(
        "UPDATE assignments SET status='pending' WHERE job_id=?1 AND status='blocked'",
        params![job_id],
    )
    .map_err(ApiError::internal)?;
    tx.commit().map_err(ApiError::internal)?;
    Ok(())
}
fn finalize_reward(
    state: &AppState,
    assignment_id: &str,
    claim: &str,
    entry_hash: &str,
) -> Result<(), ApiError> {
    let mut conn = state.db.lock().map_err(ApiError::internal)?;
    let tx = conn.transaction().map_err(ApiError::internal)?;
    crate::db::certify_ledger_intent(&tx, claim, entry_hash).map_err(ApiError::internal)?;
    let job_id: String = tx
        .query_row(
            "SELECT job_id FROM assignments WHERE assignment_id=?1",
            params![assignment_id],
            |r| r.get(0),
        )
        .map_err(ApiError::internal)?;
    tx.execute("UPDATE assignments SET status='completed',completed_at=?2,lease_until=NULL WHERE assignment_id=?1 AND status='settling'",params![assignment_id,now_unix()]).map_err(ApiError::internal)?;
    let remaining: i64 = tx
        .query_row(
            "SELECT COUNT(*) FROM assignments WHERE job_id=?1 AND status!='completed'",
            params![job_id],
            |r| r.get(0),
        )
        .map_err(ApiError::internal)?;
    if remaining == 0 {
        tx.execute(
            "UPDATE jobs SET status='completed',completed_at=?2 WHERE job_id=?1",
            params![job_id, now_unix()],
        )
        .map_err(ApiError::internal)?;
    }
    tx.commit().map_err(ApiError::internal)?;
    Ok(())
}

fn scalar_u64(conn: &Connection, sql: &str) -> Result<u64, ApiError> {
    conn.query_row(sql, [], |r| r.get::<_, i64>(0))
        .map(|v| v as u64)
        .map_err(ApiError::internal)
}
fn ledger_mode(state: &AppState) -> &'static str {
    if state.ledger.is_some() {
        "quorum"
    } else {
        "local-mvp"
    }
}

fn authenticate_known_node(
    conn: &Connection,
    auth: &mesh_protocol::AuthProof,
    action: &str,
    body_hash: &str,
) -> Result<(), ApiError> {
    let pk: Option<String> = conn
        .query_row(
            "SELECT public_key_b64 FROM nodes WHERE node_id=?1",
            params![auth.node_id],
            |r| r.get(0),
        )
        .optional()
        .map_err(ApiError::internal)?;
    let Some(pk) = pk else {
        return Err(ApiError::unauthorized("node is not registered"));
    };
    verify_auth(&pk, auth, action, body_hash).map_err(ApiError::unauthorized)?;
    consume_nonce(conn, auth)
}
fn consume_nonce(conn: &Connection, auth: &mesh_protocol::AuthProof) -> Result<(), ApiError> {
    let now = now_unix();
    conn.execute(
        "DELETE FROM auth_nonces WHERE expires_at < ?1",
        params![now],
    )
    .map_err(ApiError::internal)?;
    let r = conn
        .execute(
            "INSERT OR IGNORE INTO auth_nonces(node_id,nonce_b64,expires_at) VALUES(?1,?2,?3)",
            params![
                auth.node_id,
                auth.nonce_b64,
                now + mesh_protocol::AUTH_MAX_CLOCK_SKEW_SECS * 2
            ],
        )
        .map_err(ApiError::internal)?;
    if r != 1 {
        return Err(ApiError::unauthorized("replayed authentication nonce"));
    }
    Ok(())
}
fn apply_ledger_delta(
    tx: &rusqlite::Transaction<'_>,
    node_id: &str,
    delta_mcu: i64,
    category: &str,
    job_id: Option<&str>,
    assignment_id: Option<&str>,
) -> Result<(), ApiError> {
    let updated = tx
        .execute(
            "UPDATE balances SET balance_mcu=balance_mcu+?2 WHERE node_id=?1",
            params![node_id, delta_mcu],
        )
        .map_err(ApiError::internal)?;
    if updated != 1 {
        return Err(ApiError::not_found("balance record not found"));
    }
    tx.execute("INSERT INTO ledger(node_id,delta_mcu,category,job_id,assignment_id,created_at) VALUES(?1,?2,?3,?4,?5,?6)",params![node_id,delta_mcu,category,job_id,assignment_id,now_unix()]).map_err(ApiError::internal)?;
    Ok(())
}
fn cache_balance(state: &AppState, node_id: &str, balance: i64) -> Result<(), ApiError> {
    let conn = state.db.lock().map_err(ApiError::internal)?;
    conn.execute(
        "UPDATE balances SET balance_mcu=?2 WHERE node_id=?1",
        params![node_id, balance],
    )
    .map_err(ApiError::internal)?;
    Ok(())
}
async fn authoritative_balance(
    state: &AppState,
    node_id: &str,
) -> Result<BalanceResponse, ApiError> {
    if let Some(l) = &state.ledger {
        let p = l
            .balance_quorum(node_id)
            .await
            .map_err(|e| ApiError::conflict(format!("validator quorum unavailable: {e}")))?;
        Ok(BalanceResponse {
            node_id: node_id.into(),
            balance_mcu: p.balance_mcu,
            earned_mcu: p.earned_mcu,
            spent_mcu: p.spent_mcu,
            ledger_height: Some(p.head.sequence),
            ledger_head: Some(p.head.entry_hash),
        })
    } else {
        let conn = state.db.lock().map_err(ApiError::internal)?;
        let b: Option<i64> = conn
            .query_row(
                "SELECT balance_mcu FROM balances WHERE node_id=?1",
                params![node_id],
                |r| r.get(0),
            )
            .optional()
            .map_err(ApiError::internal)?;
        let Some(b) = b else {
            return Err(ApiError::not_found("node not found"));
        };
        let e: i64 = conn
            .query_row(
                "SELECT COALESCE(SUM(delta_mcu),0) FROM ledger WHERE node_id=?1 AND delta_mcu>0",
                params![node_id],
                |r| r.get(0),
            )
            .map_err(ApiError::internal)?;
        let s: i64 = conn
            .query_row(
                "SELECT COALESCE(-SUM(delta_mcu),0) FROM ledger WHERE node_id=?1 AND delta_mcu<0",
                params![node_id],
                |r| r.get(0),
            )
            .map_err(ApiError::internal)?;
        Ok(BalanceResponse {
            node_id: node_id.into(),
            balance_mcu: b,
            earned_mcu: e,
            spent_mcu: s,
            ledger_height: None,
            ledger_head: None,
        })
    }
}
