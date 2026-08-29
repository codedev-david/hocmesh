use crate::error::ApiError;
use crate::federation::Federation;
use crate::schedule::{self, ResourceGraph, ShardCandidate, Vertex, WorkerProfile};
use axum::{
    Json, Router,
    extract::{Path, Query, State},
    routing::{get, post},
};
use hocmesh_ai::{
    DeliveredBatchSummary, FailInferenceRequest, FailInferenceResponse, InferenceAssignment,
    InferenceJobStatus, NodeProfile, PlanRequest, PlanResponse, PollInferenceRequest,
    PollInferenceResponse, PromptOutput, ReceiptInferenceRequest, ReceiptInferenceResponse,
    RefundInferenceRequest, RefundInferenceResponse, RegisterModelRequest, RegisterModelResponse,
    ReportInferenceRequest, ReportInferenceResponse, SettleInferenceRequest,
    SettleInferenceResponse, SubmitInferenceRequest, SubmitInferenceResponse,
    fail_inference_body_hash, inference_settings_digest, plan_body_hash, plan_parallelism,
    rank_candidates, refund_inference_body_hash, register_model_body_hash,
    report_inference_body_hash, submit_inference_body_hash, validate_plan,
};
use hocmesh_core::bandwidth;
use hocmesh_core::compute::{split_work, work_cost_mcu};
use hocmesh_core::proximity;
use hocmesh_core::reputation::Reputation;
use hocmesh_core::roles::{self, NodeRole};
use hocmesh_core::verify::{self, AuditNonce, Verdict};
use hocmesh_gpu::{BackendKind, DeviceCapability};
use hocmesh_ledger::{
    network::LedgerNetwork,
    types::{
        COMMUNITY_ISSUANCE_ACCOUNT, InferenceDisputeEvidence, InferenceReceiptEvidence,
        InferenceRefundEvidence, InferenceReserveEvidence, InferenceRewardEvidence,
        JobRefundEvidence, JobReserveEvidence, LedgerTransaction, Posting, ProviderRewardEvidence,
        TransactionEvidence, TransactionKind, escrow_account, inference_holding_account,
    },
    validate::claim_key,
};
use hocmesh_model::ModelManifest;
use hocmesh_protocol::{
    BalanceResponse, CollatzPeakTotal, DEFAULT_LEASE_SECONDS, HeartbeatRequest, JobStatusResponse,
    LedgerEntry, LedgerHistoryResponse, NetworkCoordinate, NetworkStatsResponse, NodeCapabilities,
    NodeStatusResponse, PeerSample, PeerSampleResponse, PollRequest, PollResponse, PricedBatch,
    ReconciliationResponse, RefundRequest, RefundResponse, RefundableShard, RegisterRequest,
    RegisterResponse, ResultRequest, ResultResponse, SETTLEMENT_WINDOW_SECS, SubmitJobRequest,
    SubmitJobResponse, WorkAssignment, WorkResult, WorkSpec, empty_body_hash, heartbeat_body_hash,
    job_id_from_auth, now_unix, refund_body_hash, register_body_hash, result_body_hash,
    submit_body_hash, verify_auth,
};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
#[cfg(test)]
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<crate::db::Pool>,
    pub ledger: Option<LedgerNetwork>,
    /// Set when this coordinator is one of several over the same ledger.
    /// `None` is a deployment of one, which owns every job by definition.
    pub federation: Option<Federation>,
}

type AssignmentSettlementRow = (String, String, i64, i64, String, Option<String>, i64);

/// How recently a node must have been heard from to count as online.
const NODE_ONLINE_SECS: i64 = 30;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/nodes/register", post(register))
        .route("/v1/nodes/heartbeat", post(heartbeat))
        .route("/v1/work/poll", post(poll_work))
        .route("/v1/work/result", post(report_result))
        .route("/v1/work/refund", post(refund_shard))
        .route("/v1/jobs/submit", post(submit_job))
        .route("/v1/jobs/{id}", get(job_status))
        .route("/v1/nodes/{id}/balance", get(balance))
        .route("/v1/nodes/{id}/history", get(ledger_history))
        .route("/v1/nodes/{id}", get(node_status))
        .route("/v1/network/stats", get(network_stats))
        .route("/v1/network/peers", get(network_peers))
        .route("/v1/ai/models", get(list_models))
        .route("/v1/ai/models/register", post(register_model))
        .route("/v1/ai/models/{model}/{revision}", get(get_model))
        .route("/v1/ai/plan", post(plan_ai))
        .route("/v1/ai/jobs/submit", post(submit_inference))
        .route("/v1/ai/jobs/refund", post(refund_inference))
        .route("/v1/ai/jobs/receipt", post(receipt_inference))
        .route("/v1/ai/jobs/settle", post(settle_inference))
        .route("/v1/ai/jobs/{id}", get(inference_status))
        .route("/v1/ai/work/poll", post(poll_inference))
        .route("/v1/ai/work/result", post(report_inference))
        .route("/v1/ai/work/fail", post(fail_inference))
        .route("/v1/ledger/reconciliation", get(reconciliation))
        .route("/v1/topology", get(topology))
        .route("/v1/federation/status", get(federation_status))
        .route("/v1/federation/jobs/{id}", get(federation_job_owner))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

/// What the coordinator and the ledger still disagree about.
///
/// Read-only on purpose. There is no companion endpoint to force an intent
/// through or write one off, because either would be the coordinator ruling on
/// CU, and the whole design turns on it never doing that. The daemon settles
/// what can be settled; this says what is left.
async fn reconciliation(
    State(state): State<AppState>,
) -> Result<Json<ReconciliationResponse>, ApiError> {
    let conn = state.db.get().map_err(ApiError::internal)?;
    Ok(Json(ReconciliationResponse {
        unsettled: crate::db::unsettled_ledger_intents(&conn).map_err(ApiError::internal)?,
        orphaned_objects: crate::db::orphaned_funding_objects(&conn).map_err(ApiError::internal)?,
    }))
}

async fn register_model(
    State(state): State<AppState>,
    Json(req): Json<RegisterModelRequest>,
) -> Result<Json<RegisterModelResponse>, ApiError> {
    req.manifest
        .validate()
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let body_hash = register_model_body_hash(&req.manifest).map_err(ApiError::internal)?;
    let digest = req
        .manifest
        .digest()
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let json = serde_json::to_string(&req.manifest).map_err(ApiError::internal)?;
    let conn = state.db.get().map_err(ApiError::internal)?;
    authenticate_known_node(&conn, &req.auth, "register_model", &body_hash)?;
    conn.execute(
        "INSERT INTO model_manifests(manifest_digest,model_id,revision,manifest_json,publisher_node_id,created_at)
         VALUES(?1,?2,?3,?4,?5,?6)
         ON CONFLICT(model_id,revision) DO UPDATE SET manifest_digest=excluded.manifest_digest,
           manifest_json=excluded.manifest_json,publisher_node_id=excluded.publisher_node_id,created_at=excluded.created_at",
        params![digest, req.manifest.model_id, req.manifest.revision, json, req.auth.node_id, now_unix()],
    ).map_err(ApiError::internal)?;
    Ok(Json(RegisterModelResponse {
        manifest_digest: digest,
        model_id: req.manifest.model_id,
        revision: req.manifest.revision,
    }))
}

async fn list_models(State(state): State<AppState>) -> Result<Json<Vec<ModelManifest>>, ApiError> {
    let conn = state.db.get().map_err(ApiError::internal)?;
    let mut statement = conn
        .prepare("SELECT manifest_json FROM model_manifests ORDER BY model_id,revision")
        .map_err(ApiError::internal)?;
    let rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(ApiError::internal)?;
    let mut manifests = Vec::new();
    for row in rows {
        manifests.push(
            serde_json::from_str(&row.map_err(ApiError::internal)?).map_err(ApiError::internal)?,
        );
    }
    Ok(Json(manifests))
}

async fn get_model(
    State(state): State<AppState>,
    Path((model, revision)): Path<(String, String)>,
) -> Result<Json<ModelManifest>, ApiError> {
    let conn = state.db.get().map_err(ApiError::internal)?;
    let json: Option<String> = conn
        .query_row(
            "SELECT manifest_json FROM model_manifests WHERE model_id=?1 AND revision=?2",
            params![model, revision],
            |row| row.get(0),
        )
        .optional()
        .map_err(ApiError::internal)?;
    let json = json.ok_or_else(|| ApiError::not_found("model revision not found"))?;
    Ok(Json(
        serde_json::from_str(&json).map_err(ApiError::internal)?,
    ))
}

async fn plan_ai(
    State(state): State<AppState>,
    Json(req): Json<PlanRequest>,
) -> Result<Json<PlanResponse>, ApiError> {
    let body_hash = plan_body_hash(&req).map_err(ApiError::internal)?;
    let (manifest, nodes) = {
        let conn = state.db.get().map_err(ApiError::internal)?;
        authenticate_known_node(&conn, &req.auth, "plan_ai", &body_hash)?;
        let json: Option<String> = conn
            .query_row(
                "SELECT manifest_json FROM model_manifests WHERE model_id=?1 AND revision=?2",
                params![req.model_id, req.revision],
                |row| row.get(0),
            )
            .optional()
            .map_err(ApiError::internal)?;
        let manifest: ModelManifest = serde_json::from_str(
            &json.ok_or_else(|| ApiError::not_found("model revision not found"))?,
        )
        .map_err(ApiError::internal)?;
        let digest = manifest.digest().map_err(ApiError::internal)?;
        let all_chunks: std::collections::BTreeSet<_> = manifest
            .chunks
            .iter()
            .map(|chunk| chunk.sha256.clone())
            .collect();
        let requester = stored_coordinate(&conn, &req.auth.node_id);
        let mut statement = conn
            .prepare(
                "SELECT node_id,capabilities_json FROM nodes WHERE last_seen>=?1 AND node_id!=?2",
            )
            .map_err(ApiError::internal)?;
        let rows = statement
            .query_map(
                params![now_unix() - NODE_ONLINE_SECS, req.auth.node_id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .map_err(ApiError::internal)?;
        let mut nodes = Vec::new();
        for row in rows {
            let (node_id, capabilities_json) = row.map_err(ApiError::internal)?;
            let capabilities: NodeCapabilities =
                serde_json::from_str(&capabilities_json).map_err(ApiError::internal)?;
            if !capabilities.ai_runtime_ready {
                continue;
            }
            let devices = capabilities
                .gpus
                .iter()
                .filter_map(protocol_gpu_to_device)
                .collect();
            let cached_chunks = if capabilities.cached_model_manifests.contains(&digest) {
                all_chunks.clone()
            } else {
                Default::default()
            };
            nodes.push(NodeProfile {
                node_id,
                devices,
                cached_chunks,
                network_latency_ms: scoring_latency_ms(requester.as_ref(), &capabilities),
                bandwidth_mbps: ranking_bandwidth_mbps(&capabilities),
                load_fraction: (capabilities.load_permille.min(1000) as f64) / 1000.0,
                recent_failures: 0,
                online: true,
                memory_bandwidth_bytes_per_second: capabilities.memory_bandwidth_bytes_per_second,
                coordinate: capabilities.network_coordinate,
                prefill_eligible: roles::can_serve(&capabilities, NodeRole::Prefill),
            });
        }
        (manifest, nodes)
    };
    let candidates = rank_candidates(&manifest, &req.requirements, &nodes, &req.excluded_nodes);
    if candidates.is_empty() {
        return Err(ApiError::conflict("no eligible AI devices are online"));
    }
    let plan = plan_parallelism(&candidates, req.layer_count, &req.requirements)
        .map_err(|error| ApiError::conflict(error.to_string()))?;
    validate_plan(&plan, req.layer_count, req.requirements.batch_size)
        .map_err(ApiError::internal)?;
    Ok(Json(PlanResponse {
        manifest_digest: manifest.digest().map_err(ApiError::internal)?,
        candidates,
        plan,
    }))
}

fn protocol_gpu_to_device(gpu: &hocmesh_protocol::GpuCapability) -> Option<DeviceCapability> {
    let backend = match gpu.backend.to_ascii_lowercase().as_str() {
        "cuda" => BackendKind::Cuda,
        "rocm" | "hip" => BackendKind::Rocm,
        "metal" => BackendKind::Metal,
        "cpu" => BackendKind::Cpu,
        _ => return None,
    };
    Some(DeviceCapability {
        stable_id: gpu.stable_id.clone(),
        backend,
        vendor: gpu.vendor.clone(),
        name: gpu.name.clone(),
        memory_bytes: gpu.memory_mb.map(|mb| mb * 1024 * 1024),
        driver_version: gpu.driver_version.clone(),
        compute_version: gpu.compute_version.clone(),
        supports_fp16: gpu.supports_fp16,
        supports_bf16: gpu.supports_bf16,
        supports_int8: gpu.supports_int8,
        // Whatever the node measured on the device itself, which today is
        // nothing: no caller of `benchmark_llama_cpp` exists yet, and the host
        // memcpy that used to be reported here has been withdrawn because it
        // was not a device measurement. `None` is the honest answer, and the
        // planner already knows to fall back to the node's own main-memory
        // figure when a device has not been measured.
        memory_bandwidth_bytes_per_second: gpu.benchmark_bytes_per_second,
    })
}

async fn submit_inference(
    State(state): State<AppState>,
    Json(req): Json<SubmitInferenceRequest>,
) -> Result<Json<SubmitInferenceResponse>, ApiError> {
    req.validate()
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let body_hash = submit_inference_body_hash(&req).map_err(ApiError::internal)?;
    // The requester fixes the job id, not the coordinator. An id nobody can
    // choose after the fact is what stops an escrow being re-pointed later.
    let job_id = hocmesh_protocol::inference_job_id_from_auth(&req.auth);
    let (manifest, digest, plan, seed_peers, requester_pk, total_cost_mcu) = {
        let conn = state.db.get().map_err(ApiError::internal)?;
        authenticate_known_node(&conn, &req.auth, "submit_inference", &body_hash)?;
        let (manifest, nodes, seed_peers) =
            ai_context(&conn, &req.auth.node_id, &req.model_id, &req.revision)?;
        let digest = manifest.digest().map_err(ApiError::internal)?;
        let total_cost_mcu = check_inference_bill(&req, &manifest, &digest)?;
        let candidates = rank_candidates(&manifest, &req.requirements, &nodes, &Default::default());
        if candidates.is_empty() {
            return Err(ApiError::conflict("no eligible AI devices are online"));
        }
        let plan = plan_parallelism(&candidates, req.layer_count, &req.requirements)
            .map_err(|error| ApiError::conflict(error.to_string()))?;
        // The planner is the only producer today, so this should never fire.
        // It is here because of how it would fail if it ever did: a stage
        // discovering it has no layers to run, hours into a job whose escrow
        // is already committed and whose plan is already in the database.
        validate_plan(&plan, req.layer_count, req.requirements.batch_size)
            .map_err(ApiError::internal)?;
        let requester_pk = conn
            .query_row(
                "SELECT public_key_b64 FROM nodes WHERE node_id=?1",
                params![req.auth.node_id],
                |r| r.get::<_, String>(0),
            )
            .map_err(ApiError::internal)?;
        (
            manifest,
            digest,
            plan,
            seed_peers,
            requester_pk,
            total_cost_mcu,
        )
    };
    // Inference is paid for out of the same balance, priced against the same
    // constant as CPU work. That is the whole point of the unit: a machine that
    // counted primes overnight can spend what it earned on somebody else's GPU.
    let bal = authoritative_balance(&state, &req.auth.node_id).await?;
    if bal.balance_mcu < total_cost_mcu {
        return Err(ApiError::conflict(format!(
            "insufficient compute credit: need {:.3} CU, have {:.3} CU; contribute first",
            total_cost_mcu as f64 / 1000.0,
            bal.balance_mcu as f64 / 1000.0
        )));
    }
    // The batch plan is certified with the escrow. Which machines are online is
    // not a fact a validator can reproduce, so who was promised what has to be
    // part of what gets signed rather than something recomputed later.
    let batches: Vec<PricedBatch> = plan
        .batches
        .iter()
        .map(|b| PricedBatch {
            batch_start: b.batch_start,
            batch_end: b.batch_end,
            node_id: b.node_id.clone(),
        })
        .collect();
    let settings_digest = inference_settings_digest(&req).map_err(ApiError::internal)?;
    // One timestamp for the escrow and for the local row. The settlement window
    // is measured from the certified reservation, so the coordinator must not
    // hold a different idea of when the job started.
    let now = now_unix();
    let ledger_tx = state.ledger.as_ref().map(|_| LedgerTransaction {
        transaction_id: format!("reserve_{job_id}"),
        kind: TransactionKind::InferenceReserve,
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
        evidence: TransactionEvidence::InferenceReserve(InferenceReserveEvidence {
            job_id: job_id.clone(),
            requester_public_key_b64: requester_pk.clone(),
            requester_auth: req.auth.clone(),
            billing: req.billing.clone(),
            settings_digest: settings_digest.clone(),
            batches,
        }),
        created_at: now,
    });
    let ready = ledger_tx.is_none();
    {
        let mut conn = state.db.get().map_err(ApiError::internal)?;
        let transaction = conn.transaction().map_err(ApiError::internal)?;
        transaction.execute(
            "INSERT INTO ai_jobs(job_id,requester_node_id,request_json,plan_json,manifest_digest,status,created_at)
             VALUES(?1,?2,?3,?4,?5,?6,?7)",
            params![job_id, req.auth.node_id, serde_json::to_string(&req).map_err(ApiError::internal)?,
                serde_json::to_string(&plan).map_err(ApiError::internal)?, digest,
                if ready { "pending" } else { "funding" }, now],
        ).map_err(ApiError::internal)?;
        if ready {
            apply_ledger_delta(
                &transaction,
                &req.auth.node_id,
                -total_cost_mcu,
                "inference_reservation",
                Some(&job_id),
                None,
            )?;
        }
        for (index, batch) in plan.batches.iter().enumerate() {
            let prompts = (batch.batch_start..batch.batch_end)
                .map(|prompt_index| (prompt_index, req.prompts[prompt_index as usize].clone()))
                .collect();
            let assignment = InferenceAssignment {
                assignment_id: hocmesh_protocol::inference_assignment_id(&job_id, index as u32),
                job_id: job_id.clone(),
                manifest: manifest.clone(),
                seed_peers: seed_peers.clone(),
                prompts,
                max_tokens: req.max_tokens,
                temperature_milli: req.temperature_milli,
                seed: req.seed,
                device_id: batch.device_id.clone(),
                lease_seconds: DEFAULT_LEASE_SECONDS,
            };
            transaction.execute(
                "INSERT INTO ai_assignments(assignment_id,job_id,assigned_node_id,assignment_json,status)
                 VALUES(?1,?2,?3,?4,?5)",
                params![assignment.assignment_id, job_id, batch.node_id,
                    serde_json::to_string(&assignment).map_err(ApiError::internal)?,
                    if ready { "pending" } else { "blocked" }],
            ).map_err(ApiError::internal)?;
        }
        if let Some(tx_record) = &ledger_tx {
            crate::db::persist_ledger_intent(
                &transaction,
                &claim_key(tx_record),
                "inference_reserve",
                &job_id,
                &serde_json::to_string(tx_record).map_err(ApiError::internal)?,
            )
            .map_err(ApiError::internal)?;
        }
        transaction.commit().map_err(ApiError::internal)?;
    }
    if let (Some(ledger), Some(tx_record)) = (&state.ledger, ledger_tx) {
        let ck = claim_key(&tx_record);
        let cert = ledger.transact(tx_record).await.map_err(|e| {
            ApiError::conflict(format!(
                "inference funding pending recovery after ledger error: {e}"
            ))
        })?;
        finalize_inference_reservation(&state, &job_id, &ck, &cert.entry.entry_hash)?;
    }
    Ok(Json(SubmitInferenceResponse {
        job_id,
        manifest_digest: digest,
        assignments: plan.batches.len() as u32,
        plan,
    }))
}

/// Check the bill a requester signed against the request it arrived with.
///
/// A signature over a cheap bill is worthless if the prompts sent alongside it
/// are expensive ones, so every term the price depends on is compared with what
/// actually turned up. The coordinator is not trusted to price the job - it is
/// only checking that the requester priced the job it really sent.
fn check_inference_bill(
    req: &SubmitInferenceRequest,
    manifest: &ModelManifest,
    digest: &str,
) -> Result<i64, ApiError> {
    let billing = &req.billing;
    if billing.manifest_digest != digest {
        return Err(ApiError::bad_request(
            "the bill was written for a different model revision",
        ));
    }
    let Some(parameter_count) = manifest.parameter_count else {
        return Err(ApiError::bad_request(
            "a model that does not declare its parameter count cannot be priced",
        ));
    };
    if billing.parameter_count != parameter_count
        || billing.total_size_bytes != manifest.total_size_bytes
    {
        return Err(ApiError::bad_request(
            "the bill does not match the model it names",
        ));
    }
    // A publisher that overstates a model size could overcharge every requester
    // and pay itself as the provider. Four-bit weights are the densest thing
    // anyone ships, so twice the file size is a hard ceiling on the parameters.
    if !hocmesh_protocol::parameter_count_is_plausible(
        billing.parameter_count,
        billing.total_size_bytes,
    ) {
        return Err(ApiError::bad_request(
            "declared parameter count does not fit the bytes of the model",
        ));
    }
    if billing.max_tokens != req.max_tokens
        || billing.prompt_bytes != hocmesh_ai::prompt_bytes(&req.prompts)
        || billing.prompts_digest
            != hocmesh_ai::prompts_digest(&req.prompts).map_err(ApiError::internal)?
    {
        return Err(ApiError::bad_request(
            "the bill was written for different prompts",
        ));
    }
    let cost = hocmesh_core::compute::inference_cost_mcu(
        &billing.prompt_bytes,
        billing.max_tokens,
        billing.parameter_count,
    );
    if cost > billing.max_cost_mcu {
        return Err(ApiError::bad_request(
            "inference costs more than the requester authorised",
        ));
    }
    Ok(cost)
}

/// Unblock an inference job once its escrow is certified.
fn finalize_inference_reservation(
    state: &AppState,
    job_id: &str,
    claim: &str,
    entry_hash: &str,
) -> Result<(), ApiError> {
    let mut conn = state.db.get().map_err(ApiError::internal)?;
    let tx = conn.transaction().map_err(ApiError::internal)?;
    crate::db::certify_ledger_intent(&tx, claim, entry_hash).map_err(ApiError::internal)?;
    tx.execute(
        "UPDATE ai_jobs SET status=?2 WHERE job_id=?1 AND status=?3",
        params![job_id, "pending", "funding"],
    )
    .map_err(ApiError::internal)?;
    tx.execute(
        "UPDATE ai_assignments SET status=?2 WHERE job_id=?1 AND status=?3",
        params![job_id, "pending", "blocked"],
    )
    .map_err(ApiError::internal)?;
    tx.commit().map_err(ApiError::internal)?;
    Ok(())
}

async fn poll_inference(
    State(state): State<AppState>,
    Json(req): Json<PollInferenceRequest>,
) -> Result<Json<PollInferenceResponse>, ApiError> {
    let conn = state.db.get().map_err(ApiError::internal)?;
    authenticate_known_node(&conn, &req.auth, "poll_inference", &empty_body_hash())?;
    let now = now_unix();
    conn.execute("UPDATE ai_assignments SET status='pending',lease_until=NULL WHERE status='leased' AND lease_until<?1", params![now]).map_err(ApiError::internal)?;
    let row: Option<(String, String)> = conn.query_row(
        "SELECT assignment_id,assignment_json FROM ai_assignments WHERE assigned_node_id=?1 AND status='pending' ORDER BY rowid LIMIT 1",
        params![req.auth.node_id], |row| Ok((row.get(0)?, row.get(1)?)),
    ).optional().map_err(ApiError::internal)?;
    let Some((assignment_id, json)) = row else {
        return Ok(Json(PollInferenceResponse { assignment: None }));
    };
    let updated = conn.execute(
        "UPDATE ai_assignments SET status='leased',lease_until=?2 WHERE assignment_id=?1 AND status='pending'",
        params![assignment_id, now + DEFAULT_LEASE_SECONDS],
    ).map_err(ApiError::internal)?;
    if updated == 0 {
        return Ok(Json(PollInferenceResponse { assignment: None }));
    }
    Ok(Json(PollInferenceResponse {
        assignment: Some(serde_json::from_str(&json).map_err(ApiError::internal)?),
    }))
}

async fn report_inference(
    State(state): State<AppState>,
    Json(req): Json<ReportInferenceRequest>,
) -> Result<Json<ReportInferenceResponse>, ApiError> {
    if req.outputs.is_empty() {
        return Err(ApiError::bad_request("outputs are empty"));
    }
    for output in &req.outputs {
        output
            .validate()
            .map_err(|error| ApiError::bad_request(error.to_string()))?;
    }
    let body_hash = report_inference_body_hash(&req).map_err(ApiError::internal)?;
    let job_id = {
        let conn = state.db.get().map_err(ApiError::internal)?;
        authenticate_known_node(&conn, &req.auth, "report_inference", &body_hash)?;
        let row: Option<(String, String, String)> = conn.query_row(
            "SELECT job_id,assignment_json,status FROM ai_assignments WHERE assignment_id=?1 AND assigned_node_id=?2",
            params![req.assignment_id, req.auth.node_id], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        ).optional().map_err(ApiError::internal)?;
        let Some((job_id, assignment_json, status)) = row else {
            return Err(ApiError::not_found("AI assignment not found"));
        };
        if status != "leased" {
            return Err(ApiError::conflict(
                "AI assignment is not leased to this node",
            ));
        }
        let assignment: InferenceAssignment =
            serde_json::from_str(&assignment_json).map_err(ApiError::internal)?;
        let expected: std::collections::BTreeSet<_> =
            assignment.prompts.iter().map(|(index, _)| *index).collect();
        let actual: std::collections::BTreeSet<_> = req
            .outputs
            .iter()
            .map(|output| output.prompt_index)
            .collect();
        if actual != expected || actual.len() != req.outputs.len() {
            return Err(ApiError::conflict(
                "output indexes do not match the assignment",
            ));
        }
        // The claim a provider signed has to be the claim its own assignment
        // implies: it cannot ask for more than the batch prices at, and it
        // cannot move the claim onto a batch it was never given.
        let Some((batch_start, batch_end, reward_mcu)) = hocmesh_ai::assignment_claim(&assignment)
        else {
            return Err(ApiError::conflict(
                "this assignment carries no priceable batch",
            ));
        };
        if req.job_id != job_id
            || req.batch_start != batch_start
            || req.batch_end != batch_end
            || req.reward_mcu != reward_mcu
        {
            return Err(ApiError::conflict(
                "the signed claim does not match the assignment it names",
            ));
        }
        let requester: String = conn
            .query_row(
                "SELECT requester_node_id FROM ai_jobs WHERE job_id=?1",
                params![job_id],
                |row| row.get(0),
            )
            .map_err(ApiError::internal)?;
        if requester == req.auth.node_id {
            return Err(ApiError::conflict(
                "requester cannot receive a reward from its own paid job",
            ));
        }
        job_id
    };
    // Delivery is not payment. The provider's signed claim is kept here until
    // the requester takes the answer and says what it is worth; nothing moves
    // on the ledger until then, so a provider that returns arbitrary bytes for
    // a real assignment gets a row in a table and nothing else.
    let outputs_digest = hocmesh_protocol::hash_json(&req.outputs).map_err(ApiError::internal)?;
    let job_completed = {
        let mut conn = state.db.get().map_err(ApiError::internal)?;
        let transaction = conn.transaction().map_err(ApiError::internal)?;
        transaction.execute(
        "UPDATE ai_assignments SET status=?4,outputs_json=?2,lease_until=NULL,completed_at=?3,report_json=?5,outputs_digest=?6 WHERE assignment_id=?1",
        params![
            req.assignment_id,
            serde_json::to_string(&req.outputs).map_err(ApiError::internal)?,
            now_unix(),
            "completed",
            serde_json::to_string(&req).map_err(ApiError::internal)?,
            outputs_digest,
        ],
    ).map_err(ApiError::internal)?;
        let remaining: i64 = transaction
            .query_row(
                "SELECT COUNT(*) FROM ai_assignments WHERE job_id=?1 AND status!=?2",
                params![job_id, "completed"],
                |row| row.get(0),
            )
            .map_err(ApiError::internal)?;
        if remaining == 0 {
            transaction
                .execute(
                    "UPDATE ai_jobs SET status=?3,completed_at=?2 WHERE job_id=?1",
                    params![job_id, now_unix(), "completed"],
                )
                .map_err(ApiError::internal)?;
        }
        transaction.commit().map_err(ApiError::internal)?;
        remaining == 0
    };
    let balance = authoritative_balance(&state, &req.auth.node_id).await?;
    cache_balance(&state, &req.auth.node_id, balance.balance_mcu)?;
    Ok(Json(ReportInferenceResponse {
        accepted: true,
        reward_mcu: req.reward_mcu,
        balance_mcu: balance.balance_mcu,
        job_completed,
    }))
}

/// One delivered batch, as the coordinator remembers it between delivery and
/// settlement.
struct DeliveredBatch {
    job_id: String,
    requester: String,
    report: ReportInferenceRequest,
    outputs: Vec<PromptOutput>,
    outputs_digest: String,
    provider_pk: String,
    requester_pk: String,
    receipted: bool,
    settled: Option<String>,
}

/// One row of the settlement view of an assignment: the job it belongs to, the
/// provider's stored report and outputs, the digest they hash to, and how far
/// the batch has got through delivery and settlement.
type SettlementRow = (
    String,
    Option<String>,
    Option<String>,
    Option<String>,
    i64,
    Option<String>,
);

/// Reads back everything a settlement needs about one delivered batch.
fn delivered_batch(conn: &Connection, assignment_id: &str) -> Result<DeliveredBatch, ApiError> {
    let row: Option<SettlementRow> = conn
        .query_row(
            "SELECT job_id,report_json,outputs_json,outputs_digest,receipted,settled
             FROM ai_assignments WHERE assignment_id=?1",
            params![assignment_id],
            |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                ))
            },
        )
        .optional()
        .map_err(ApiError::internal)?;
    let Some((job_id, report_json, outputs_json, digest, receipted, settled)) = row else {
        return Err(ApiError::not_found("AI assignment not found"));
    };
    let (Some(report_json), Some(outputs_json), Some(outputs_digest)) =
        (report_json, outputs_json, digest)
    else {
        return Err(ApiError::conflict("this batch has not been delivered yet"));
    };
    let report: ReportInferenceRequest =
        serde_json::from_str(&report_json).map_err(ApiError::internal)?;
    let outputs: Vec<PromptOutput> =
        serde_json::from_str(&outputs_json).map_err(ApiError::internal)?;
    let requester: String = conn
        .query_row(
            "SELECT requester_node_id FROM ai_jobs WHERE job_id=?1",
            params![job_id],
            |r| r.get(0),
        )
        .map_err(ApiError::internal)?;
    let requester_pk: String = conn
        .query_row(
            "SELECT public_key_b64 FROM nodes WHERE node_id=?1",
            params![requester],
            |r| r.get(0),
        )
        .map_err(ApiError::internal)?;
    let provider_pk: String = conn
        .query_row(
            "SELECT public_key_b64 FROM nodes WHERE node_id=?1",
            params![report.auth.node_id],
            |r| r.get(0),
        )
        .map_err(ApiError::internal)?;
    Ok(DeliveredBatch {
        job_id,
        requester,
        report,
        outputs,
        outputs_digest,
        provider_pk,
        requester_pk,
        receipted: receipted != 0,
        settled,
    })
}

/// A requester taking delivery of a batch.
///
/// The answer is handed over here and nowhere else, and only against a receipt
/// that moves the batch's escrow into a holding account. That is what makes the
/// exchange even: the requester cannot read the answer and then reclaim the CU,
/// and the provider cannot be paid for bytes the requester never asked to see.
async fn receipt_inference(
    State(state): State<AppState>,
    Json(req): Json<ReceiptInferenceRequest>,
) -> Result<Json<ReceiptInferenceResponse>, ApiError> {
    let batch = {
        let conn = state.db.get().map_err(ApiError::internal)?;
        let batch = delivered_batch(&conn, &req.assignment_id)?;
        if batch.requester != req.auth.node_id {
            return Err(ApiError::unauthorized(
                "only the requester that funded a batch can take delivery of it",
            ));
        }
        let body_hash = hocmesh_protocol::inference_receipt_body_hash(
            &req.assignment_id,
            &batch.job_id,
            batch.report.batch_start,
            batch.report.batch_end,
            batch.report.reward_mcu,
            &batch.outputs_digest,
        )
        .map_err(ApiError::internal)?;
        authenticate_known_node(&conn, &req.auth, "receipt_inference", &body_hash)?;
        batch
    };
    if !batch.receipted {
        let ledger_tx = state.ledger.as_ref().map(|_| LedgerTransaction {
            transaction_id: format!("receipt_{}", req.assignment_id),
            kind: TransactionKind::InferenceReceipt,
            postings: vec![
                Posting {
                    account_id: escrow_account(&batch.job_id),
                    delta_mcu: -batch.report.reward_mcu,
                },
                Posting {
                    account_id: inference_holding_account(&req.assignment_id),
                    delta_mcu: batch.report.reward_mcu,
                },
            ],
            evidence: TransactionEvidence::InferenceReceipt(InferenceReceiptEvidence {
                job_id: batch.job_id.clone(),
                assignment_id: req.assignment_id.clone(),
                batch_start: batch.report.batch_start,
                batch_end: batch.report.batch_end,
                price_mcu: batch.report.reward_mcu,
                outputs_digest: batch.outputs_digest.clone(),
                requester_public_key_b64: batch.requester_pk.clone(),
                requester_auth: req.auth.clone(),
            }),
            created_at: now_unix(),
        });
        if let (Some(ledger), Some(tx_record)) = (&state.ledger, ledger_tx) {
            ledger.transact(tx_record).await.map_err(|e| {
                ApiError::conflict(format!("inference receipt rejected by the ledger: {e}"))
            })?;
        }
        let conn = state.db.get().map_err(ApiError::internal)?;
        // The receipt is kept, not just counted. If this requester never comes
        // back to accept or dispute, this proof is the only thing that will
        // let the commons collect the batch once the window closes.
        conn.execute(
            "UPDATE ai_assignments SET receipted=1, receipt_auth_json=?2 WHERE assignment_id=?1",
            params![
                req.assignment_id,
                serde_json::to_string(&req.auth).map_err(ApiError::internal)?
            ],
        )
        .map_err(ApiError::internal)?;
    }
    Ok(Json(ReceiptInferenceResponse {
        assignment_id: req.assignment_id.clone(),
        batch_start: batch.report.batch_start,
        batch_end: batch.report.batch_end,
        price_mcu: batch.report.reward_mcu,
        outputs_digest: batch.outputs_digest,
        outputs: batch.outputs,
    }))
}

/// A requester saying what the answer it took was worth.
///
/// Accepting pays the provider out of the holding account. Disputing sends the
/// same CU to the commons instead. Neither outcome returns it to the requester,
/// so a dispute is a statement about the work rather than a way to get the
/// money back, and there is nothing to gain by lying in either direction.
async fn settle_inference(
    State(state): State<AppState>,
    Json(req): Json<SettleInferenceRequest>,
) -> Result<Json<SettleInferenceResponse>, ApiError> {
    let batch = {
        let conn = state.db.get().map_err(ApiError::internal)?;
        let batch = delivered_batch(&conn, &req.assignment_id)?;
        if batch.requester != req.auth.node_id {
            return Err(ApiError::unauthorized(
                "only the requester that funded a batch can settle it",
            ));
        }
        if !batch.receipted {
            return Err(ApiError::conflict(
                "take delivery of this batch before settling it",
            ));
        }
        if let Some(settled) = &batch.settled {
            return Err(ApiError::conflict(format!(
                "this batch was already settled as {settled}"
            )));
        }
        let body_hash = hocmesh_protocol::inference_verdict_body_hash(
            req.accepted,
            &req.assignment_id,
            &batch.job_id,
            batch.report.batch_start,
            batch.report.batch_end,
            batch.report.reward_mcu,
            &batch.outputs_digest,
        )
        .map_err(ApiError::internal)?;
        let action = if req.accepted {
            "accept_inference"
        } else {
            "dispute_inference"
        };
        authenticate_known_node(&conn, &req.auth, action, &body_hash)?;
        batch
    };
    let requester_pk = batch.requester_pk.clone();
    let held = inference_holding_account(&req.assignment_id);
    let price = batch.report.reward_mcu;
    let (kind, credit, evidence) = if req.accepted {
        (
            TransactionKind::InferenceReward,
            batch.report.auth.node_id.clone(),
            TransactionEvidence::InferenceReward(InferenceRewardEvidence {
                job_id: batch.job_id.clone(),
                assignment_id: req.assignment_id.clone(),
                batch_start: batch.report.batch_start,
                batch_end: batch.report.batch_end,
                reward_mcu: price,
                outputs_digest: batch.outputs_digest.clone(),
                provider_public_key_b64: batch.provider_pk.clone(),
                provider_auth: batch.report.auth.clone(),
                requester_public_key_b64: requester_pk,
                requester_acceptance: req.auth.clone(),
            }),
        )
    } else {
        (
            TransactionKind::InferenceDispute,
            COMMUNITY_ISSUANCE_ACCOUNT.to_string(),
            TransactionEvidence::InferenceDispute(InferenceDisputeEvidence {
                job_id: batch.job_id.clone(),
                assignment_id: req.assignment_id.clone(),
                batch_start: batch.report.batch_start,
                batch_end: batch.report.batch_end,
                price_mcu: price,
                outputs_digest: batch.outputs_digest.clone(),
                reason: req.reason.clone(),
                requester_public_key_b64: requester_pk,
                requester_auth: req.auth.clone(),
            }),
        )
    };
    let ledger_tx = state.ledger.as_ref().map(|_| LedgerTransaction {
        transaction_id: format!("settle_{}", req.assignment_id),
        kind,
        postings: vec![
            Posting {
                account_id: held,
                delta_mcu: -price,
            },
            Posting {
                account_id: credit.clone(),
                delta_mcu: price,
            },
        ],
        evidence,
        created_at: now_unix(),
    });
    // Same order as every other settlement here: the ledger decides first, and
    // only then does the coordinator write down what it decided.
    if let (Some(ledger), Some(tx_record)) = (&state.ledger, ledger_tx) {
        ledger.transact(tx_record).await.map_err(|e| {
            ApiError::conflict(format!("inference settlement rejected by the ledger: {e}"))
        })?;
    }
    let job_completed =
        {
            let mut conn = state.db.get().map_err(ApiError::internal)?;
            let transaction = conn.transaction().map_err(ApiError::internal)?;
            let updated = transaction
            .execute(
                "UPDATE ai_assignments SET settled=?2 WHERE assignment_id=?1 AND settled IS NULL",
                params![req.assignment_id, if req.accepted { "paid" } else { "disputed" }],
            )
            .map_err(ApiError::internal)?;
            if updated == 0 {
                return Err(ApiError::conflict("this batch was already settled"));
            }
            // Without a ledger there are no escrow or holding accounts to move CU
            // between, so the local mirror only records the half a node can see:
            // a paid provider. A disputed batch simply never reaches anyone.
            if state.ledger.is_none() && req.accepted {
                apply_ledger_delta(
                    &transaction,
                    &credit,
                    price,
                    "inference_reward",
                    Some(&batch.job_id),
                    Some(&req.assignment_id),
                )?;
            }
            let unsettled: i64 = transaction
                .query_row(
                    "SELECT COUNT(*) FROM ai_assignments
                 WHERE job_id=?1 AND settled IS NULL AND status<>?2",
                    params![batch.job_id, "refunded"],
                    |row| row.get(0),
                )
                .map_err(ApiError::internal)?;
            transaction.commit().map_err(ApiError::internal)?;
            unsettled == 0
        };
    if req.accepted {
        let balance = authoritative_balance(&state, &credit).await?;
        cache_balance(&state, &credit, balance.balance_mcu)?;
    }
    Ok(Json(SettleInferenceResponse {
        assignment_id: req.assignment_id,
        accepted: req.accepted,
        paid_mcu: price,
        job_completed,
    }))
}

/// Take back the escrow on a batch nobody delivered.
///
/// Without this an escrow is a one-way valve: a provider that crashes or never
/// answers takes the CU that funded it with it. The refund shares a claim key
/// with the reward, so a batch settles exactly once and in one direction.
async fn refund_inference(
    State(state): State<AppState>,
    Json(req): Json<RefundInferenceRequest>,
) -> Result<Json<RefundInferenceResponse>, ApiError> {
    let body_hash = refund_inference_body_hash(&req).map_err(ApiError::internal)?;
    let (requester_pk, reserved_at) = {
        let conn = state.db.get().map_err(ApiError::internal)?;
        authenticate_known_node(&conn, &req.auth, "refund_inference", &body_hash)?;
        let row: Option<(String, i64)> = conn
            .query_row(
                "SELECT requester_node_id,created_at FROM ai_jobs WHERE job_id=?1",
                params![req.job_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(ApiError::internal)?;
        let Some((requester, reserved_at)) = row else {
            return Err(ApiError::not_found("AI job not found"));
        };
        // The escrow returns where it came from, never to whoever asks.
        if requester != req.auth.node_id {
            return Err(ApiError::unauthorized(
                "only the requester who reserved a job can reclaim its escrow",
            ));
        }
        let row: Option<(String, String)> = conn
            .query_row(
                "SELECT assignment_json,status FROM ai_assignments WHERE assignment_id=?1 AND job_id=?2",
                params![req.assignment_id, req.job_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(ApiError::internal)?;
        let Some((assignment_json, status)) = row else {
            return Err(ApiError::not_found("AI assignment not found"));
        };
        if status == "completed" || status == "refunded" {
            return Err(ApiError::conflict("this batch has already settled"));
        }
        let assignment: InferenceAssignment =
            serde_json::from_str(&assignment_json).map_err(ApiError::internal)?;
        let Some((batch_start, batch_end, amount)) = hocmesh_ai::assignment_claim(&assignment)
        else {
            return Err(ApiError::conflict(
                "this assignment carries no priceable batch",
            ));
        };
        // A refund is priced exactly like the reward it replaces, so the escrow
        // drains to zero either way and nothing is stranded or conjured.
        if req.batch_start != batch_start || req.batch_end != batch_end || req.refund_mcu != amount
        {
            return Err(ApiError::conflict(
                "the signed refund does not match the batch it names",
            ));
        }
        let requester_pk: String = conn
            .query_row(
                "SELECT public_key_b64 FROM nodes WHERE node_id=?1",
                params![req.auth.node_id],
                |r| r.get::<_, String>(0),
            )
            .map_err(ApiError::internal)?;
        (requester_pk, reserved_at)
    };
    // Reward and refund windows are disjoint, so a provider and a requester can
    // never race for the same escrow.
    let now = now_unix();
    if now <= reserved_at + SETTLEMENT_WINDOW_SECS {
        return Err(ApiError::conflict(
            "this batch is still inside its settlement window",
        ));
    }
    let ledger_tx = state.ledger.as_ref().map(|_| LedgerTransaction {
        transaction_id: format!("refund_{}", req.assignment_id),
        kind: TransactionKind::InferenceRefund,
        postings: vec![
            Posting {
                account_id: escrow_account(&req.job_id),
                delta_mcu: -req.refund_mcu,
            },
            Posting {
                account_id: req.auth.node_id.clone(),
                delta_mcu: req.refund_mcu,
            },
        ],
        evidence: TransactionEvidence::InferenceRefund(InferenceRefundEvidence {
            job_id: req.job_id.clone(),
            assignment_id: req.assignment_id.clone(),
            batch_start: req.batch_start,
            batch_end: req.batch_end,
            refund_mcu: req.refund_mcu,
            requester_public_key_b64: requester_pk,
            requester_auth: req.auth.clone(),
        }),
        created_at: now,
    });
    if let (Some(ledger), Some(tx_record)) = (&state.ledger, ledger_tx) {
        ledger.transact(tx_record).await.map_err(|e| {
            ApiError::conflict(format!("inference refund rejected by the ledger: {e}"))
        })?;
    }
    {
        let mut conn = state.db.get().map_err(ApiError::internal)?;
        let transaction = conn.transaction().map_err(ApiError::internal)?;
        transaction
            .execute(
                "UPDATE ai_assignments SET status=?2,lease_until=NULL WHERE assignment_id=?1",
                params![req.assignment_id, "refunded"],
            )
            .map_err(ApiError::internal)?;
        if state.ledger.is_none() {
            apply_ledger_delta(
                &transaction,
                &req.auth.node_id,
                req.refund_mcu,
                "inference_refund",
                Some(&req.job_id),
                Some(&req.assignment_id),
            )?;
        }
        transaction.commit().map_err(ApiError::internal)?;
    }
    let balance = authoritative_balance(&state, &req.auth.node_id).await?;
    cache_balance(&state, &req.auth.node_id, balance.balance_mcu)?;
    Ok(Json(RefundInferenceResponse {
        refunded_mcu: req.refund_mcu,
        balance_mcu: balance.balance_mcu,
    }))
}

async fn fail_inference(
    State(state): State<AppState>,
    Json(req): Json<FailInferenceRequest>,
) -> Result<Json<FailInferenceResponse>, ApiError> {
    if req.reason.is_empty() || req.reason.len() > 4096 {
        return Err(ApiError::bad_request("invalid failure reason"));
    }
    let body_hash = fail_inference_body_hash(&req).map_err(ApiError::internal)?;
    let conn = state.db.get().map_err(ApiError::internal)?;
    authenticate_known_node(&conn, &req.auth, "fail_inference", &body_hash)?;
    let row: Option<(String, String, String, String, i64)> = conn.query_row(
        "SELECT a.job_id,j.request_json,a.assignment_json,a.failed_nodes_json,a.failure_count FROM ai_assignments a JOIN ai_jobs j ON j.job_id=a.job_id WHERE a.assignment_id=?1 AND a.assigned_node_id=?2",
        params![req.assignment_id, req.auth.node_id], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
    ).optional().map_err(ApiError::internal)?;
    let Some((_job_id, request_json, assignment_json, failed_nodes_json, failures)) = row else {
        return Err(ApiError::not_found("AI assignment not found"));
    };
    let original: SubmitInferenceRequest =
        serde_json::from_str(&request_json).map_err(ApiError::internal)?;
    let (manifest, nodes, _) = ai_context(
        &conn,
        &original.auth.node_id,
        &original.model_id,
        &original.revision,
    )?;
    let mut excluded: std::collections::BTreeSet<String> =
        serde_json::from_str(&failed_nodes_json).map_err(ApiError::internal)?;
    excluded.insert(req.auth.node_id.clone());
    let next = rank_candidates(&manifest, &original.requirements, &nodes, &excluded)
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::conflict("no failover AI device is available"))?;
    let mut assignment: InferenceAssignment =
        serde_json::from_str(&assignment_json).map_err(ApiError::internal)?;
    assignment.device_id.clone_from(&next.device_id);
    conn.execute(
        "UPDATE ai_assignments SET assigned_node_id=?2,assignment_json=?3,failed_nodes_json=?4,status='pending',lease_until=NULL,failure_count=failure_count+1 WHERE assignment_id=?1",
        params![req.assignment_id, next.node_id, serde_json::to_string(&assignment).map_err(ApiError::internal)?, serde_json::to_string(&excluded).map_err(ApiError::internal)?],
    ).map_err(ApiError::internal)?;
    Ok(Json(FailInferenceResponse {
        rerouted_to: next.node_id,
        attempt: failures as u32 + 2,
    }))
}

async fn inference_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<InferenceJobStatus>, ApiError> {
    let conn = state.db.get().map_err(ApiError::internal)?;
    let status: Option<String> = conn
        .query_row(
            "SELECT status FROM ai_jobs WHERE job_id=?1",
            params![job_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(ApiError::internal)?;
    let status = status.ok_or_else(|| ApiError::not_found("AI job not found"))?;
    let total: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM ai_assignments WHERE job_id=?1",
            params![job_id],
            |row| row.get(0),
        )
        .map_err(ApiError::internal)?;
    let completed: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM ai_assignments WHERE job_id=?1 AND status='completed'",
            params![job_id],
            |row| row.get(0),
        )
        .map_err(ApiError::internal)?;
    // Only receipted batches read out here. This endpoint takes no
    // authentication, so an ungated answer would be a way to read generated
    // text without ever paying for it -- and, worse, a way for the requester
    // itself to skip the receipt that makes the exchange fair.
    let mut statement = conn
        .prepare(
            "SELECT outputs_json FROM ai_assignments
             WHERE job_id=?1 AND outputs_json IS NOT NULL AND receipted=1",
        )
        .map_err(ApiError::internal)?;
    let rows = statement
        .query_map(params![job_id], |row| row.get::<_, String>(0))
        .map_err(ApiError::internal)?;
    let mut outputs: Vec<PromptOutput> = Vec::new();
    for row in rows {
        outputs.extend(
            serde_json::from_str::<Vec<PromptOutput>>(&row.map_err(ApiError::internal)?)
                .map_err(ApiError::internal)?,
        );
    }
    outputs.sort_by_key(|output| output.prompt_index);
    let refundable = refundable_batches(&conn, &job_id)?;
    let delivered = delivered_batches(&conn, &job_id)?;
    Ok(Json(InferenceJobStatus {
        job_id,
        status,
        total_assignments: total as u32,
        completed_assignments: completed as u32,
        outputs,
        refundable,
        delivered,
    }))
}

/// Every batch of a job that has an answer waiting behind a receipt.
///
/// Listed for anyone, because none of it is the answer: an assignment id, the
/// range it covers, what it costs and a digest. A requester reads this to
/// decide what to take delivery of; a bystander learns only that work happened.
fn delivered_batches(
    conn: &Connection,
    job_id: &str,
) -> Result<Vec<DeliveredBatchSummary>, ApiError> {
    let mut statement = conn
        .prepare(
            "SELECT assignment_id,report_json,outputs_digest,receipted,settled
             FROM ai_assignments
             WHERE job_id=?1 AND report_json IS NOT NULL AND outputs_digest IS NOT NULL
             ORDER BY assignment_id",
        )
        .map_err(ApiError::internal)?;
    let rows = statement
        .query_map(params![job_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, Option<String>>(4)?,
            ))
        })
        .map_err(ApiError::internal)?;
    let mut delivered = Vec::new();
    for row in rows {
        let (assignment_id, report_json, outputs_digest, receipted, settled) =
            row.map_err(ApiError::internal)?;
        let report: ReportInferenceRequest =
            serde_json::from_str(&report_json).map_err(ApiError::internal)?;
        delivered.push(DeliveredBatchSummary {
            assignment_id,
            batch_start: report.batch_start,
            batch_end: report.batch_end,
            price_mcu: report.reward_mcu,
            outputs_digest,
            receipted: receipted != 0,
            settled,
        });
    }
    Ok(delivered)
}

/// Batches whose settlement window has closed with nothing delivered.
///
/// Listed, not settled: the coordinator is telling the requester what it can
/// go and claim, and every number here is re-derived by the ledger from the
/// certified billing before any CU moves. A batch already paid or already
/// reclaimed never appears, because its claim key is spent.
fn refundable_batches(
    conn: &Connection,
    job_id: &str,
) -> Result<Vec<hocmesh_ai::RefundableBatch>, ApiError> {
    let reserved_at: Option<i64> = conn
        .query_row(
            "SELECT created_at FROM ai_jobs WHERE job_id=?1 AND status<>'funding'",
            params![job_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(ApiError::internal)?;
    let Some(reserved_at) = reserved_at else {
        return Ok(Vec::new());
    };
    if now_unix() <= reserved_at + SETTLEMENT_WINDOW_SECS {
        return Ok(Vec::new());
    }
    let mut statement = conn
        .prepare(
            "SELECT assignment_json FROM ai_assignments WHERE job_id=?1 AND status NOT IN ('completed','refunded','blocked')",
        )
        .map_err(ApiError::internal)?;
    let rows = statement
        .query_map(params![job_id], |row| row.get::<_, String>(0))
        .map_err(ApiError::internal)?;
    let mut out = Vec::new();
    for row in rows {
        let assignment: InferenceAssignment =
            serde_json::from_str(&row.map_err(ApiError::internal)?).map_err(ApiError::internal)?;
        if let Some((batch_start, batch_end, refund_mcu)) =
            hocmesh_ai::assignment_claim(&assignment)
        {
            out.push(hocmesh_ai::RefundableBatch {
                assignment_id: assignment.assignment_id,
                batch_start,
                batch_end,
                refund_mcu,
            });
        }
    }
    out.sort_by_key(|batch| batch.batch_start);
    Ok(out)
}

/// The uplink to assume for ranking, in Mbit/s.
///
/// A machine that has never served a byte has no measurement, and ranking
/// still has to put a number on what moving a model to it would cost. It
/// assumes an ordinary broadband link rather than refusing, because refusing
/// would be a deadlock and not a safeguard: a node earns its measurement by
/// serving, it can only serve a model it was sent, and it is only sent one if
/// it ranked well enough to be picked.
///
/// The prefill gate does not get this courtesy. A bad guess here costs one
/// slow transfer; a bad guess there costs every request routed through that
/// stage for as long as the job runs.
fn ranking_bandwidth_mbps(caps: &NodeCapabilities) -> f64 {
    let kbps = roles::measured_uplink_kbps(caps).unwrap_or(bandwidth::ASSUMED_KBPS);
    (kbps as f64 / 1000.0).max(0.001)
}

fn ai_context(
    conn: &Connection,
    requester_node_id: &str,
    model_id: &str,
    revision: &str,
) -> Result<(ModelManifest, Vec<NodeProfile>, Vec<String>), ApiError> {
    let json: Option<String> = conn
        .query_row(
            "SELECT manifest_json FROM model_manifests WHERE model_id=?1 AND revision=?2",
            params![model_id, revision],
            |row| row.get(0),
        )
        .optional()
        .map_err(ApiError::internal)?;
    let manifest: ModelManifest =
        serde_json::from_str(&json.ok_or_else(|| ApiError::not_found("model revision not found"))?)
            .map_err(ApiError::internal)?;
    let digest = manifest.digest().map_err(ApiError::internal)?;
    let all_chunks: std::collections::BTreeSet<_> = manifest
        .chunks
        .iter()
        .map(|chunk| chunk.sha256.clone())
        .collect();
    let requester = stored_coordinate(conn, requester_node_id);
    let mut statement = conn
        .prepare("SELECT node_id,capabilities_json FROM nodes WHERE last_seen>=?1 AND node_id!=?2")
        .map_err(ApiError::internal)?;
    let rows = statement
        .query_map(
            params![now_unix() - NODE_ONLINE_SECS, requester_node_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .map_err(ApiError::internal)?;
    let mut nodes = Vec::new();
    let mut seed_peers = Vec::new();
    for row in rows {
        let (node_id, json) = row.map_err(ApiError::internal)?;
        let capabilities: NodeCapabilities =
            serde_json::from_str(&json).map_err(ApiError::internal)?;
        if !capabilities.ai_runtime_ready {
            continue;
        }
        let cached = capabilities.cached_model_manifests.contains(&digest);
        if cached && let Some(url) = &capabilities.model_seed_url {
            seed_peers.push(url.clone());
        }
        nodes.push(NodeProfile {
            node_id,
            devices: capabilities
                .gpus
                .iter()
                .filter_map(protocol_gpu_to_device)
                .collect(),
            cached_chunks: if cached {
                all_chunks.clone()
            } else {
                Default::default()
            },
            network_latency_ms: scoring_latency_ms(requester.as_ref(), &capabilities),
            bandwidth_mbps: ranking_bandwidth_mbps(&capabilities),
            load_fraction: capabilities.load_permille.min(1000) as f64 / 1000.0,
            recent_failures: 0,
            online: true,
            memory_bandwidth_bytes_per_second: capabilities.memory_bandwidth_bytes_per_second,
            coordinate: capabilities.network_coordinate,
            prefill_eligible: roles::can_serve(&capabilities, NodeRole::Prefill),
        });
    }
    Ok((manifest, nodes, seed_peers))
}

async fn register(
    State(state): State<AppState>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<RegisterResponse>, ApiError> {
    if req.capabilities.protocol_version != hocmesh_protocol::PROTOCOL_VERSION {
        return Err(ApiError::bad_request("unsupported protocol version"));
    }
    let body_hash =
        register_body_hash(&req.public_key_b64, &req.capabilities).map_err(ApiError::internal)?;
    verify_auth(&req.public_key_b64, &req.auth, "register", &body_hash)
        .map_err(ApiError::unauthorized)?;
    let caps_json = serde_json::to_string(&req.capabilities).map_err(ApiError::internal)?;
    let now = now_unix();
    {
        let mut conn = state.db.get().map_err(ApiError::internal)?;
        consume_nonce(&conn, &req.auth)?;
        let tx = conn.transaction().map_err(ApiError::internal)?;
        tx.execute(r#"INSERT INTO nodes(node_id,public_key_b64,capabilities_json,registered_at,last_seen,region)
            VALUES(?1,?2,?3,?4,?4,?5) ON CONFLICT(node_id) DO UPDATE SET public_key_b64=excluded.public_key_b64,capabilities_json=excluded.capabilities_json,last_seen=excluded.last_seen,region=excluded.region"#,
            params![
                req.auth.node_id,
                req.public_key_b64,
                caps_json,
                now,
                state.federation.as_ref().and_then(|f| f.region())
            ]).map_err(ApiError::internal)?;
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
        protocol_version: hocmesh_protocol::PROTOCOL_VERSION,
        ledger_mode: ledger_mode(&state).into(),
    }))
}

async fn heartbeat(
    State(state): State<AppState>,
    Json(req): Json<HeartbeatRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let body_hash = heartbeat_body_hash(&req.capabilities).map_err(ApiError::internal)?;
    let conn = state.db.get().map_err(ApiError::internal)?;
    authenticate_known_node(&conn, &req.auth, "heartbeat", &body_hash)?;
    let caps_json = serde_json::to_string(&req.capabilities).map_err(ApiError::internal)?;
    conn.execute(
        "UPDATE nodes SET last_seen=?2,capabilities_json=?3,region=?4 WHERE node_id=?1",
        params![
            req.auth.node_id,
            now_unix(),
            caps_json,
            state.federation.as_ref().and_then(|f| f.region())
        ],
    )
    .map_err(ApiError::internal)?;
    Ok(Json(serde_json::json!({"ok":true})))
}

/// How many pending shards one poll scores.
///
/// Scoring is cheap but not free, and a coordinator with a long backlog should
/// not walk all of it on every poll. The window is bounded and ordered by age,
/// so the shards nearest to starving are always the ones inside it.
const POLL_WINDOW: i64 = 256;

/// The pending shards this coordinator would consider handing to one node,
/// paired with the specs needed to answer with them.
struct PendingWork {
    candidates: Vec<ShardCandidate>,
    /// `assignment_id` -> the parsed spec and whether the job is system-funded.
    specs: HashMap<String, (WorkSpec, bool)>,
}

/// Working-set estimate for a shard, used only to keep a shard off a node that
/// cannot hold it.
///
/// Deliberately an upper bound rather than a measurement: being wrong high
/// costs a node one shard it could have run, while being wrong low hands a node
/// work that will die part-way through and take the lease with it.
fn shard_memory_bytes(work: &WorkSpec) -> u64 {
    // Interpreter, buffers, and the result entry, none of which scale with the
    // spec.
    const BASELINE: u64 = 1 << 20;
    match work {
        // Both are a running counter over a range; nothing is materialised.
        WorkSpec::PrimeCount { .. } | WorkSpec::CollatzPeak { .. } => BASELINE,
        WorkSpec::MatrixMultiply {
            dim,
            row_start,
            row_end,
            ..
        } => {
            // The whole of B is generated from its seed and held, plus the row
            // block of the product. Both are u32 elements.
            let dim = u64::from(*dim);
            let span = u64::from(row_end.saturating_sub(*row_start));
            let elements = dim
                .saturating_mul(dim)
                .saturating_add(span.saturating_mul(dim));
            BASELINE.saturating_add(elements.saturating_mul(4))
        }
    }
}

/// What this coordinator knows about a node, in the shape the scheduler wants.
fn worker_profile(conn: &Connection, node_id: &str) -> Result<WorkerProfile, ApiError> {
    let (caps_json, region): (String, Option<String>) = conn
        .query_row(
            "SELECT capabilities_json,region FROM nodes WHERE node_id=?1",
            params![node_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(ApiError::internal)?;
    let caps: NodeCapabilities = serde_json::from_str(&caps_json).map_err(ApiError::internal)?;
    Ok(WorkerProfile {
        node_id: node_id.to_string(),
        region,
        caps,
        reputation: reputation_row(conn, node_id)?,
    })
}

/// The shards this node has already finished, per job.
///
/// This is the whole of the scheduler's cache-locality evidence: a node that
/// has done neighbouring shards of the same job holds the same generated
/// operands, and giving it another shard of that job avoids regenerating them.
fn completed_shards_by_job(
    conn: &Connection,
    node_id: &str,
) -> Result<HashMap<String, Vec<u32>>, ApiError> {
    let mut stmt = conn
        .prepare(
            "SELECT job_id,shard_index FROM assignments
             WHERE leased_to=?1 AND status IN ('completed','settling')",
        )
        .map_err(ApiError::internal)?;
    let rows = stmt
        .query_map(params![node_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
        })
        .map_err(ApiError::internal)?;
    let mut history: HashMap<String, Vec<u32>> = HashMap::new();
    for row in rows {
        let (job_id, shard_index) = row.map_err(ApiError::internal)?;
        history
            .entry(job_id)
            .or_default()
            .push(shard_index.clamp(0, i64::from(u32::MAX)) as u32);
    }
    Ok(history)
}

/// Collect the pending shards this coordinator may offer to `node_id`.
///
/// Two filters apply before any scoring. A node is never offered a shard of a
/// job it requested itself, because self-serving a job would let a requester
/// audit its own work. And in a federated deployment a coordinator only offers
/// the jobs it owns, so two coordinators never hand the same shard to two
/// nodes.
fn pending_work(
    conn: &Connection,
    node_id: &str,
    federation: Option<&Federation>,
) -> Result<PendingWork, ApiError> {
    let mut stmt = conn
        .prepare(
            "SELECT a.assignment_id,a.job_id,a.shard_index,a.work_json,a.reward_mcu,
                    j.system_funded,j.created_at
             FROM assignments a JOIN jobs j ON j.job_id=a.job_id
             WHERE a.status='pending' AND (j.requester_node_id IS NULL OR j.requester_node_id != ?1)
             ORDER BY j.created_at ASC, a.rowid ASC LIMIT ?2",
        )
        .map_err(ApiError::internal)?;
    let rows = stmt
        .query_map(params![node_id, POLL_WINDOW], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, i64>(4)?,
                r.get::<_, i64>(5)?,
                r.get::<_, i64>(6)?,
            ))
        })
        .map_err(ApiError::internal)?;

    let history = completed_shards_by_job(conn, node_id)?;
    let mut candidates = Vec::new();
    let mut specs = HashMap::new();
    for row in rows {
        let (assignment_id, job_id, shard_index, work_json, reward_mcu, system_funded, created_at) =
            row.map_err(ApiError::internal)?;
        if let Some(fed) = federation
            && !fed.owns(&job_id)
        {
            continue;
        }
        let work: WorkSpec = serde_json::from_str(&work_json).map_err(ApiError::internal)?;
        let shard_index = shard_index.clamp(0, i64::from(u32::MAX)) as u32;
        let done = history.get(&job_id);
        candidates.push(ShardCandidate {
            assignment_id: assignment_id.clone(),
            job_id,
            shard_index,
            reward_mcu,
            memory_bytes: shard_memory_bytes(&work),
            // A shard's wait starts when its job was submitted: shards are
            // created with the job, so this is the age of the shard too.
            created_at,
            shards_done_here: done.map_or(0, |v| v.len() as u32),
            nearest_done_shard: done
                .and_then(|v| v.iter().copied().min_by_key(|s| s.abs_diff(shard_index))),
            // CPU workloads need nothing on disk. Model-backed shards will
            // populate this when they gain a coordinator-side spec.
            required_manifests: Vec::new(),
        });
        specs.insert(assignment_id, (work, system_funded != 0));
    }
    Ok(PendingWork { candidates, specs })
}

async fn poll_work(
    State(state): State<AppState>,
    Json(req): Json<PollRequest>,
) -> Result<Json<PollResponse>, ApiError> {
    let conn = state.db.get().map_err(ApiError::internal)?;
    authenticate_known_node(&conn, &req.auth, "poll", &empty_body_hash())?;
    let now = now_unix();
    conn.execute("UPDATE assignments SET status='pending',leased_to=NULL,lease_until=NULL,leased_by=NULL WHERE status='leased' AND lease_until < ?1", params![now]).map_err(ApiError::internal)?;
    conn.execute(
        "UPDATE nodes SET last_seen=?2 WHERE node_id=?1",
        params![req.auth.node_id, now],
    )
    .map_err(ApiError::internal)?;

    let worker = worker_profile(&conn, &req.auth.node_id)?;
    let pending = pending_work(&conn, &req.auth.node_id, state.federation.as_ref())?;
    let region = state.federation.as_ref().and_then(|f| f.region());
    // How many nodes are competing for this work. Polling is the only way a
    // node asks, so a node seen inside the head-start window is one that either
    // wanted a shard or is about to. This is a count and not a comparison --
    // the scheduler needs to know whether demand exceeds supply, not who is
    // faster than whom -- which is what keeps it one indexed range scan rather
    // than a walk over every peer's capabilities on every poll.
    let recent_pollers: usize = conn
        .query_row(
            "SELECT COUNT(*) FROM nodes WHERE last_seen >= ?1",
            params![now - schedule::HEAD_START_SECONDS],
            |r| r.get::<_, i64>(0),
        )
        .map_err(ApiError::internal)?
        .max(0) as usize;
    let Some((chosen, fit)) = schedule::best(
        &worker,
        &pending.candidates,
        now,
        region,
        &schedule::Weights::default(),
        recent_pollers,
    ) else {
        return Ok(Json(PollResponse { assignment: None }));
    };
    let Some((work, system_funded)) = pending.specs.get(&chosen.assignment_id).cloned() else {
        return Ok(Json(PollResponse { assignment: None }));
    };
    // Emitted per decision so a placement that looks wrong can be explained by
    // the axis that drove it rather than guessed at.
    tracing::debug!(
        node = %worker.node_id,
        assignment = %chosen.assignment_id,
        considered = pending.candidates.len(),
        hardware = fit.hardware,
        network = fit.network,
        reliability = fit.reliability,
        locality = fit.locality,
        starvation = fit.starvation,
        total = fit.total,
        "shard offered"
    );

    // Sized to the node that is taking it. `fit` has already refused anyone no
    // lease would cover, so this only ever decides how much longer than the
    // default a slower machine gets -- and the shard is worth the same mCU
    // either way, because the price is in the work and not in the clock.
    let lease_seconds = schedule::lease_seconds_for(&worker.caps, chosen.reward_mcu);
    let lease_until = now + lease_seconds;
    // `leased_by` records which coordinator is responsible for this lease, so a
    // peer that goes unreachable can have its in-flight leases cut short
    // without touching anyone else's.
    let updated = conn.execute(
        "UPDATE assignments SET status='leased',leased_to=?2,lease_until=?3,leased_by=?4 WHERE assignment_id=?1 AND status='pending'",
        params![
            chosen.assignment_id,
            req.auth.node_id,
            lease_until,
            state.federation.as_ref().map(|f| f.coordinator_id().to_string())
        ],
    ).map_err(ApiError::internal)?;
    if updated == 0 {
        return Ok(Json(PollResponse { assignment: None }));
    }
    Ok(Json(PollResponse {
        assignment: Some(WorkAssignment {
            assignment_id: chosen.assignment_id.clone(),
            job_id: chosen.job_id.clone(),
            shard_index: chosen.shard_index,
            work,
            reward_mcu: chosen.reward_mcu,
            // The same number the row was written with, so the node's own idea
            // of its deadline matches the one it will actually be held to.
            lease_seconds,
            system_funded,
        }),
    }))
}

/// A request for a set of machines that are near one another.
#[derive(Debug, Deserialize)]
struct TopologyQuery {
    /// How many machines the caller wants held together. Absent means "just
    /// describe the graph".
    #[serde(default)]
    cluster: Option<usize>,
    /// Restrict the cluster to one region.
    #[serde(default)]
    region: Option<String>,
    /// Machines with less than this much shareable memory are not considered.
    #[serde(default)]
    min_memory_bytes: Option<u64>,
    /// Exclude machines below this standing, in `[0, 1]`.
    ///
    /// Worth setting for tightly coupled work: a collective finishes when its
    /// last member does, so one unreliable machine costs the whole set, which
    /// is not true of independent shards.
    #[serde(default)]
    min_standing: Option<f64>,
}

/// One machine, as the topology view sees it.
#[derive(Serialize)]
struct TopologyNode {
    node_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    region: Option<String>,
    /// Sustained throughput in mCU/s, or `None` when the node has not
    /// benchmarked itself. Absent is not zero, and must not be read as slow.
    #[serde(skip_serializing_if = "Option::is_none")]
    mcu_per_second: Option<f64>,
    shared_memory_bytes: u64,
    shared_logical_cpus: usize,
    gpus: usize,
    /// Standing in `[0, 1]`, from the same audit history the scheduler uses.
    standing: f64,
    /// Whether this node's coordinate is usable for distance at all.
    located: bool,
}

/// The machines currently offering capacity and how far apart they are.
///
/// A read-only view: nothing here reserves anything or moves CU. It exists so
/// that work which must run on several machines at once can be placed on
/// machines that are actually near each other, and so an operator can see the
/// shape of the network the scheduler is choosing from.
#[derive(Serialize)]
struct TopologyReport {
    online: usize,
    /// Round trip assumed between two machines that have measured nothing,
    /// reported so a caller can tell a real distance from a placeholder.
    unknown_edge_micros: u64,
    nodes: Vec<TopologyNode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cluster: Option<schedule::Cluster>,
    /// Set when a cluster was asked for and could not be formed.
    #[serde(skip_serializing_if = "Option::is_none")]
    cluster_unavailable: Option<String>,
}

/// Build the resource graph from every node the coordinator has heard from.
///
/// Offline nodes stay in the graph as vertices but are never chosen: they are
/// still useful context for an operator, and `cluster` gates on `online`
/// itself.
fn resource_graph(conn: &Connection) -> Result<ResourceGraph, ApiError> {
    let mut stmt = conn
        .prepare("SELECT node_id,capabilities_json,region,last_seen FROM nodes")
        .map_err(ApiError::internal)?;
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, i64>(3)?,
            ))
        })
        .map_err(ApiError::internal)?;
    let cutoff = now_unix() - NODE_ONLINE_SECS;
    let mut vertices = Vec::new();
    for row in rows {
        let (node_id, caps_json, region, last_seen) = row.map_err(ApiError::internal)?;
        let Ok(caps) = serde_json::from_str::<NodeCapabilities>(&caps_json) else {
            continue;
        };
        let reputation = reputation_row(conn, &node_id)?;
        vertices.push(Vertex {
            node_id,
            region,
            caps,
            reputation,
            online: last_seen >= cutoff,
        });
    }
    Ok(ResourceGraph::new(vertices))
}

async fn topology(
    State(state): State<AppState>,
    Query(query): Query<TopologyQuery>,
) -> Result<Json<TopologyReport>, ApiError> {
    let conn = state.db.get().map_err(ApiError::internal)?;
    let graph = resource_graph(&conn)?;
    let nodes: Vec<TopologyNode> = graph
        .vertices()
        .iter()
        .map(|v| TopologyNode {
            node_id: v.node_id.clone(),
            region: v.region.clone(),
            mcu_per_second: schedule::mcu_per_second(&v.caps),
            shared_memory_bytes: v.caps.shared_memory_bytes,
            shared_logical_cpus: v.caps.shared_logical_cpus,
            gpus: v.caps.gpus.len(),
            standing: schedule::standing(&v.reputation),
            located: v
                .caps
                .network_coordinate
                .as_ref()
                .is_some_and(hocmesh_core::proximity::is_plausible),
        })
        .collect();
    let online = graph.vertices().iter().filter(|v| v.online).count();

    let (cluster, cluster_unavailable) = match query.cluster {
        None => (None, None),
        Some(size) => {
            let region = query.region.clone();
            let min_memory = query.min_memory_bytes.unwrap_or(0);
            let min_standing = query.min_standing.unwrap_or(0.0);
            let found = graph.cluster(size, |v| {
                v.caps.shared_memory_bytes >= min_memory
                    && schedule::standing(&v.reputation) >= min_standing
                    && region
                        .as_deref()
                        .is_none_or(|r| v.region.as_deref() == Some(r))
            });
            match found {
                Some(c) => (Some(c), None),
                None => (
                    None,
                    Some(format!(
                        "no {size} online machines satisfy the requested constraints"
                    )),
                ),
            }
        }
    };

    Ok(Json(TopologyReport {
        online,
        unknown_edge_micros: schedule::UNKNOWN_EDGE_MICROS,
        nodes,
        cluster,
        cluster_unavailable,
    }))
}

/// What this coordinator believes about its peers.
#[derive(Serialize)]
struct FederationReport {
    federated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<crate::federation::FederationStatus>,
}

/// Which coordinator is responsible for a job, and where to reach it.
///
/// A client that lands on the wrong coordinator can use this to find the right
/// one rather than being told no.
#[derive(Serialize)]
struct JobOwnerReport {
    job_id: String,
    federated: bool,
    /// `None` when unfederated: there is one coordinator and it owns
    /// everything.
    #[serde(skip_serializing_if = "Option::is_none")]
    owner: Option<String>,
    mine: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
}

async fn federation_status(State(state): State<AppState>) -> Json<FederationReport> {
    Json(FederationReport {
        federated: state.federation.is_some(),
        status: state.federation.as_ref().map(|f| f.status()),
    })
}

async fn federation_job_owner(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Json<JobOwnerReport> {
    let Some(fed) = state.federation.as_ref() else {
        return Json(JobOwnerReport {
            job_id,
            federated: false,
            owner: None,
            mine: true,
            url: None,
        });
    };
    let owner = fed.owner_of(&job_id);
    let mine = owner == fed.coordinator_id();
    let url = fed.peer_url(&owner);
    Json(JobOwnerReport {
        job_id,
        federated: true,
        owner: Some(owner),
        mine,
        url,
    })
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
        let conn = state.db.get().map_err(ApiError::internal)?;
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
    let nonce = AuditNonce::draw(draw_randomness());
    let reputation = load_reputation(&state, &req.auth.node_id)?;
    let settlement = verify::settle(&work, &req.result, &reputation, nonce);
    if settlement.verdict.is_rejected() || settlement.verdict == Verdict::Inconclusive {
        record_reputation(&state, &req.auth.node_id, false)?;
        let conn = state.db.get().map_err(ApiError::internal)?;
        conn.execute("UPDATE assignments SET status='pending',leased_to=NULL,lease_until=NULL WHERE assignment_id=?1",params![req.assignment_id]).map_err(ApiError::internal)?;
        return Err(ApiError::conflict(
            "work verification failed; assignment returned to queue",
        ));
    }
    record_reputation(&state, &req.auth.node_id, true)?;

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
                provisional_audit_nonce: settlement.nonce,
            }),
            created_at: now_unix(),
        };
        let ck = claim_key(&tx_record);
        let tx_json = serde_json::to_string(&tx_record).map_err(ApiError::internal)?;
        let result_json = serde_json::to_string(&req.result).map_err(ApiError::internal)?;
        {
            let mut conn = state.db.get().map_err(ApiError::internal)?;
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
        let mut conn = state.db.get().map_err(ApiError::internal)?;
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
        close_settled_job(&tx, &job_id)?;
        tx.commit().map_err(ApiError::internal)?;
    }
    let job_completed = {
        let conn = state.db.get().map_err(ApiError::internal)?;
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
/// Returns one shard's escrow to whoever funded it. Without this an escrow is
/// a one-way valve: a provider that cheats, or that simply never answers,
/// leaves the requester's CU locked in the job forever, and catching a cheat
/// never becomes a settlement. The coordinator only proposes here - the ledger
/// is what decides whether the settlement window has really closed.
async fn refund_shard(
    State(state): State<AppState>,
    Json(req): Json<RefundRequest>,
) -> Result<Json<RefundResponse>, ApiError> {
    let row = {
        let conn = state.db.get().map_err(ApiError::internal)?;
        conn.query_row(
            "SELECT a.job_id,a.shard_index,a.work_json,a.status,j.system_funded,j.requester_node_id,j.created_at FROM assignments a JOIN jobs j ON j.job_id=a.job_id WHERE a.assignment_id=?1",
            params![req.assignment_id],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, i64>(4)?,
                    r.get::<_, Option<String>>(5)?,
                    r.get::<_, i64>(6)?,
                ))
            },
        )
        .optional()
        .map_err(ApiError::internal)?
    };
    let Some((job_id, shard_index, work_json, status, system_funded, requester, job_created_at)) =
        row
    else {
        return Err(ApiError::not_found("unknown assignment"));
    };
    // Escrow moves once, and only escrow that exists can move at all. A shard
    // that already settled has nothing left to give back, and a blocked shard
    // belongs to a job whose reserve was never certified, so it is holding
    // nothing to return.
    if status != "pending" && status != "leased" {
        return Err(ApiError::conflict(format!(
            "a shard in state '{status}' holds no reclaimable escrow"
        )));
    }
    // The ledger is the authority on when a window closes, and it measures
    // from the reserve it certified, which is never earlier than the job row
    // read here. This clock therefore only ever refuses what the ledger would
    // refuse too, and it turns a rejection into an answer worth reading.
    if now_unix() <= job_created_at + SETTLEMENT_WINDOW_SECS {
        return Err(ApiError::conflict(
            "the settlement window for this shard has not closed yet",
        ));
    }
    let system_funded = system_funded != 0;
    let work: WorkSpec = serde_json::from_str(&work_json).map_err(ApiError::internal)?;
    let shard_index = u32::try_from(shard_index).map_err(ApiError::internal)?;
    let refund_mcu = work_cost_mcu(&work);
    let body_hash = refund_body_hash(
        &req.assignment_id,
        &job_id,
        shard_index,
        &work,
        refund_mcu,
        system_funded,
    )
    .map_err(ApiError::internal)?;

    // Who may claim is decided by who funded, not by who is asking.
    let (paid_to, requester_public_key_b64) = match (&req.auth, system_funded) {
        (Some(_), true) => {
            return Err(ApiError::unauthorized(
                "community escrow returns to the issuance account, not to a node",
            ));
        }
        (None, false) => {
            return Err(ApiError::unauthorized(
                "escrow somebody paid for needs that requester's signature",
            ));
        }
        (None, true) => (COMMUNITY_ISSUANCE_ACCOUNT.to_string(), None),
        (Some(auth), false) => {
            if requester.as_deref() != Some(auth.node_id.as_str()) {
                return Err(ApiError::unauthorized(
                    "only the requester who funded this job can reclaim its escrow",
                ));
            }
            let conn = state.db.get().map_err(ApiError::internal)?;
            authenticate_known_node(&conn, auth, "refund", &body_hash)?;
            let pk: String = conn
                .query_row(
                    "SELECT public_key_b64 FROM nodes WHERE node_id=?1",
                    params![auth.node_id],
                    |r| r.get(0),
                )
                .map_err(ApiError::internal)?;
            (auth.node_id.clone(), Some(pk))
        }
    };

    let mut ledger_entry_hash = None;
    if let Some(ledger) = &state.ledger {
        let tx_record = LedgerTransaction {
            transaction_id: format!("refund_{}", req.assignment_id),
            kind: TransactionKind::JobRefund,
            postings: vec![
                Posting {
                    account_id: escrow_account(&job_id),
                    delta_mcu: -refund_mcu,
                },
                Posting {
                    account_id: paid_to.clone(),
                    delta_mcu: refund_mcu,
                },
            ],
            evidence: TransactionEvidence::JobRefund(JobRefundEvidence {
                job_id: job_id.clone(),
                assignment_id: req.assignment_id.clone(),
                shard_index,
                refund_mcu,
                work: work.clone(),
                system_funded,
                requester_public_key_b64,
                requester_auth: req.auth.clone(),
            }),
            created_at: now_unix(),
        };
        let ck = claim_key(&tx_record);
        let tx_json = serde_json::to_string(&tx_record).map_err(ApiError::internal)?;
        // The ledger decides before the coordinator writes anything down. A
        // refund and a reward race for the same escrow through the same claim
        // key, so exactly one of them can win, and marking the shard refunded
        // first would turn away a provider whose reward the ledger accepted.
        let cert = ledger
            .transact(tx_record)
            .await
            .map_err(|e| ApiError::conflict(format!("refund refused by the ledger: {e}")))?;
        ledger_entry_hash = Some(cert.entry.entry_hash.clone());
        {
            let mut conn = state.db.get().map_err(ApiError::internal)?;
            let local = conn.transaction().map_err(ApiError::internal)?;
            crate::db::persist_ledger_intent(
                &local,
                &ck,
                "job_refund",
                &req.assignment_id,
                &tx_json,
            )
            .map_err(ApiError::internal)?;
            crate::db::certify_ledger_intent(&local, &ck, &cert.entry.entry_hash)
                .map_err(ApiError::internal)?;
            settle_refunded_shard(&local, &req.assignment_id, &job_id)?;
            local.commit().map_err(ApiError::internal)?;
        }
    } else {
        let mut conn = state.db.get().map_err(ApiError::internal)?;
        let tx = conn.transaction().map_err(ApiError::internal)?;
        // Without a ledger the coordinator keeps the balances itself, so it
        // hands the CU back by hand. Minted escrow has no node to return to:
        // it simply stops existing, which is what unminting it against the
        // issuance account amounts to on the real ledger.
        if !system_funded {
            apply_ledger_delta(
                &tx,
                &paid_to,
                refund_mcu,
                "refund",
                Some(&job_id),
                Some(&req.assignment_id),
            )?;
        }
        settle_refunded_shard(&tx, &req.assignment_id, &job_id)?;
        tx.commit().map_err(ApiError::internal)?;
    }

    Ok(Json(RefundResponse {
        refund_mcu,
        paid_to,
        ledger_entry_hash,
    }))
}

/// Marks a shard refunded and closes its job if nothing is left to settle.
/// A refunded shard is as final as a completed one - it is never leased
/// again - so it is written down here without a status guard: by the time
/// this runs the ledger has already decided who the escrow belongs to.
fn settle_refunded_shard(
    conn: &Connection,
    assignment_id: &str,
    job_id: &str,
) -> Result<(), ApiError> {
    conn.execute("UPDATE assignments SET status='refunded',completed_at=?2,lease_until=NULL WHERE assignment_id=?1",params![assignment_id,now_unix()]).map_err(ApiError::internal)?;
    close_settled_job(conn, job_id)?;
    Ok(())
}

/// A job is over once no shard is still waiting on a settlement. It only
/// counts as completed if every shard actually produced work: one refunded
/// shard means the job closed short of what it asked for, and a caller that
/// reads a partial result set deserves to be told which of the two it got.
fn close_settled_job(conn: &Connection, job_id: &str) -> Result<(), ApiError> {
    let open: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM assignments WHERE job_id=?1 AND status NOT IN ('completed','refunded')",
            params![job_id],
            |r| r.get(0),
        )
        .map_err(ApiError::internal)?;
    if open > 0 {
        return Ok(());
    }
    let refunded: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM assignments WHERE job_id=?1 AND status='refunded'",
            params![job_id],
            |r| r.get(0),
        )
        .map_err(ApiError::internal)?;
    let status = if refunded > 0 { "closed" } else { "completed" };
    conn.execute(
        "UPDATE jobs SET status=?2,completed_at=?3 WHERE job_id=?1",
        params![job_id, status, now_unix()],
    )
    .map_err(ApiError::internal)?;
    Ok(())
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
        let conn = state.db.get().map_err(ApiError::internal)?;
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
        let mut conn = state.db.get().map_err(ApiError::internal)?;
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
            let assignment_id = hocmesh_protocol::assignment_id(&job_id, index as u32);
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
        let conn = state.db.get().map_err(ApiError::internal)?;
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

/// How far back a single page may reach.
///
/// A dashboard scrolls; it does not need the whole chain at once, and an
/// unbounded `limit` would let one request pull a coordinator's entire ledger
/// into memory.
const MAX_HISTORY_PAGE: u32 = 500;

#[derive(Debug, Deserialize)]
struct HistoryQuery {
    /// Return entries strictly older than this position. Absent means "start
    /// at the newest".
    before: Option<u64>,
    limit: Option<u32>,
}

/// One node's ledger history, newest first.
///
/// Reads from the validator quorum when there is one and from the
/// coordinator's own table when there is not, and says which in the response.
/// A coordinator's table is a convenience mirror -- the chain is the authority
/// -- so a dashboard must be able to tell the two apart rather than presenting
/// a local row as a settled fact.
async fn ledger_history(
    State(state): State<AppState>,
    Path(node_id): Path<String>,
    Query(query): Query<HistoryQuery>,
) -> Result<Json<LedgerHistoryResponse>, ApiError> {
    let limit = query.limit.unwrap_or(50).clamp(1, MAX_HISTORY_PAGE);
    {
        let conn = state.db.get().map_err(ApiError::internal)?;
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

    if let Some(ledger) = &state.ledger {
        let page = ledger
            .fetch_history(&node_id, query.before, limit)
            .await
            .map_err(|e| ApiError::conflict(format!("validator history unavailable: {e}")))?;
        return Ok(Json(LedgerHistoryResponse {
            node_id,
            authoritative: true,
            entries: page
                .entries
                .into_iter()
                .map(|entry| LedgerEntry {
                    delta_mcu: entry.delta_mcu,
                    // The chain records transactions, not the coordinator's
                    // categories, so these stay empty rather than guessed.
                    category: None,
                    job_id: None,
                    assignment_id: None,
                    sequence: Some(entry.sequence),
                    transaction_id: Some(entry.transaction_id),
                    created_at: entry.created_at,
                })
                .collect(),
            next_before: page.next_before,
        }));
    }

    let conn = state.db.get().map_err(ApiError::internal)?;
    // One more than asked for: if it comes back, there is another page, and
    // the extra row is dropped rather than shown.
    let probe = i64::from(limit) + 1;
    let mut statement = conn
        .prepare(
            "SELECT id,delta_mcu,category,job_id,assignment_id,created_at \
             FROM ledger WHERE node_id=?1 AND (?2 IS NULL OR id < ?2) \
             ORDER BY id DESC LIMIT ?3",
        )
        .map_err(ApiError::internal)?;
    let mut rows: Vec<(i64, LedgerEntry)> = statement
        .query_map(
            params![node_id, query.before.map(|b| b as i64), probe],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    LedgerEntry {
                        delta_mcu: row.get(1)?,
                        category: Some(row.get(2)?),
                        job_id: row.get(3)?,
                        assignment_id: row.get(4)?,
                        // A coordinator without validators has no chain
                        // position to report, and inventing one would make a
                        // local row look checkable when it is not.
                        sequence: None,
                        transaction_id: None,
                        created_at: row.get(5)?,
                    },
                ))
            },
        )
        .map_err(ApiError::internal)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(ApiError::internal)?;
    let next_before = if rows.len() > limit as usize {
        rows.truncate(limit as usize);
        rows.last().map(|(id, _)| *id as u64)
    } else {
        None
    };
    Ok(Json(LedgerHistoryResponse {
        node_id,
        authoritative: false,
        entries: rows.into_iter().map(|(_, entry)| entry).collect(),
        next_before,
    }))
}

async fn job_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<JobStatusResponse>, ApiError> {
    let conn = state.db.get().map_err(ApiError::internal)?;
    let job: Option<(Option<String>, i64, String, i64, i64)> = conn
        .query_row(
            "SELECT requester_node_id,system_funded,status,reserved_mcu,created_at FROM jobs WHERE job_id=?1",
            params![job_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
        .optional()
        .map_err(ApiError::internal)?;
    let Some((requester_node_id, system_funded, status, reserved_mcu, created_at)) = job else {
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
    let mut collatz: Option<CollatzPeakTotal> = None;
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
            // Matrix shards have no scalar total to roll up; the answer is the
            // rows themselves, which the caller fetches per shard.
            WorkResult::MatrixMultiply { .. } => {}
            WorkResult::CollatzPeak {
                peak_steps,
                peak_seed,
                ..
            } => {
                // Ties resolve to the smaller seed, the same rule a shard uses
                // to combine its own buckets, so the rollup is order-free.
                let better = match collatz {
                    None => true,
                    Some(CollatzPeakTotal { steps, seed }) => {
                        peak_steps > steps || (peak_steps == steps && peak_seed < seed)
                    }
                };
                if better {
                    collatz = Some(CollatzPeakTotal {
                        steps: peak_steps,
                        seed: peak_seed,
                    });
                }
            }
        }
    }
    // Only a shard that is still waiting and whose window has closed can be
    // reclaimed, which is the same pair of conditions the refund endpoint
    // applies; listing anything else would only produce refusals.
    let mut refundable = Vec::new();
    if now_unix() > created_at + SETTLEMENT_WINDOW_SECS {
        let mut stmt = conn
            .prepare(
                "SELECT assignment_id,shard_index,work_json FROM assignments WHERE job_id=?1 AND status IN ('pending','leased') ORDER BY shard_index",
            )
            .map_err(ApiError::internal)?;
        let rows = stmt
            .query_map(params![job_id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .map_err(ApiError::internal)?;
        for row in rows {
            let (assignment_id, shard_index, work_json) = row.map_err(ApiError::internal)?;
            let work: WorkSpec = serde_json::from_str(&work_json).map_err(ApiError::internal)?;
            refundable.push(RefundableShard {
                assignment_id,
                shard_index: shard_index as u32,
                refund_mcu: work_cost_mcu(&work),
                work,
            });
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
        collatz_peak: collatz,
        refundable,
    }))
}

async fn node_status(
    State(state): State<AppState>,
    Path(node_id): Path<String>,
) -> Result<Json<NodeStatusResponse>, ApiError> {
    let conn = state.db.get().map_err(ApiError::internal)?;
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
    let conn = state.db.get().map_err(ApiError::internal)?;
    let registered_nodes = scalar_u64(&conn, "SELECT COUNT(*) FROM nodes")?;
    let online_nodes = conn
        .query_row(
            "SELECT COUNT(*) FROM nodes WHERE last_seen>=?1",
            params![now_unix() - NODE_ONLINE_SECS],
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

/// How many probe targets a node is handed at once.
///
/// Vivaldi needs a handful of well-spread peers, not the whole network: the
/// fit converges on a few, and every extra target is a real round trip on
/// someone else's machine.
const PEER_SAMPLE_SIZE: usize = 8;

/// Hand out a sample of reachable peers to measure against.
///
/// This is a directory lookup, not an authority: the coordinator never times
/// anything, never sees a round trip, and cannot influence the fit beyond
/// choosing who gets measured. Gossip can replace this endpoint without
/// changing how a single coordinate is computed.
///
/// The sample is random rather than "closest" or "most recent" on purpose.
/// Handing every node the same peers would fit them all against the same few
/// points, and a coordinate space built from one clique describes that clique
/// rather than the network.
async fn network_peers(
    State(state): State<AppState>,
) -> Result<Json<PeerSampleResponse>, ApiError> {
    let conn = state.db.get().map_err(ApiError::internal)?;
    let mut stmt = conn
        .prepare(
            "SELECT node_id, capabilities_json FROM nodes \
             WHERE last_seen>=?1 ORDER BY RANDOM()",
        )
        .map_err(ApiError::internal)?;
    let rows = stmt
        .query_map(params![now_unix() - NODE_ONLINE_SECS], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(ApiError::internal)?;

    let mut peers = Vec::new();
    for row in rows {
        let (node_id, capabilities_json) = row.map_err(ApiError::internal)?;
        let Ok(caps) = serde_json::from_str::<NodeCapabilities>(&capabilities_json) else {
            continue;
        };
        // A node that does not serve probes is not a target, even though it
        // may well be measuring others.
        let Some(probe_endpoint) = caps.probe_endpoint else {
            continue;
        };
        peers.push(PeerSample {
            node_id,
            probe_endpoint,
            coordinate: caps.network_coordinate,
        });
        if peers.len() >= PEER_SAMPLE_SIZE {
            break;
        }
    }
    Ok(Json(PeerSampleResponse { peers }))
}

fn finalize_reservation(
    state: &AppState,
    job_id: &str,
    claim: &str,
    entry_hash: &str,
) -> Result<(), ApiError> {
    let mut conn = state.db.get().map_err(ApiError::internal)?;
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
    let mut conn = state.db.get().map_err(ApiError::internal)?;
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
    close_settled_job(&tx, &job_id)?;
    tx.commit().map_err(ApiError::internal)?;
    Ok(())
}

/// The coordinate a node last advertised, if it has one.
fn stored_coordinate(conn: &Connection, node_id: &str) -> Option<NetworkCoordinate> {
    let json: Option<String> = conn
        .query_row(
            "SELECT capabilities_json FROM nodes WHERE node_id=?1",
            params![node_id],
            |row| row.get(0),
        )
        .optional()
        .ok()
        .flatten();
    serde_json::from_str::<NodeCapabilities>(&json?)
        .ok()?
        .network_coordinate
        .filter(proximity::is_plausible)
}

/// Latency to score a worker at, in milliseconds.
///
/// Prefers the requester-to-worker distance. `coordinator_latency_micros` is
/// the wrong quantity here: it measures the worker's distance to the
/// coordinator, but the payload travels between requester and worker, and
/// under a coordinator-free design there is no coordinator to be near.
fn scoring_latency_ms(requester: Option<&NetworkCoordinate>, worker: &NodeCapabilities) -> f64 {
    let coords = requester.zip(worker.network_coordinate.as_ref());
    match coords {
        Some((from, to)) if proximity::is_plausible(to) => {
            (proximity::predicted_rtt_micros(from, to) as f64 / 1000.0).max(0.1)
        }
        _ => (worker.coordinator_latency_micros as f64 / 1000.0).max(0.1),
    }
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
    auth: &hocmesh_protocol::AuthProof,
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
fn consume_nonce(conn: &Connection, auth: &hocmesh_protocol::AuthProof) -> Result<(), ApiError> {
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
                now + hocmesh_protocol::AUTH_MAX_CLOCK_SKEW_SECS * 2
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

/// Entropy for an audit challenge.
///
/// Drawn only after a worker's signed result is already in hand, so a worker
/// cannot know which of its submissions will be checked, nor which part of one.
/// A deployment facing a coordinator that may collude with a worker wants a
/// verifiable random beacon here instead; see `verify`'s module docs.
fn draw_randomness() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ COUNTER.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed)
}

fn load_reputation(state: &AppState, node_id: &str) -> Result<Reputation, ApiError> {
    let conn = state.db.get().map_err(ApiError::internal)?;
    reputation_row(&conn, node_id)
}

/// A node's standing, read from a connection the caller already holds.
fn reputation_row(conn: &Connection, node_id: &str) -> Result<Reputation, ApiError> {
    let row = conn
        .query_row(
            "SELECT accepted,rejected,streak FROM reputation WHERE node_id=?1",
            params![node_id],
            |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                ))
            },
        )
        .optional()
        .map_err(ApiError::internal)?;
    Ok(
        row.map_or_else(Reputation::new, |(accepted, rejected, streak)| Reputation {
            accepted: accepted.max(0) as u64,
            rejected: rejected.max(0) as u64,
            streak: streak.clamp(0, i64::from(u32::MAX)) as u32,
        }),
    )
}

/// Fold one settled result into a node's standing.
///
/// A rejection zeroes the streak, so trust is re-earned from nothing and the
/// next results from that node are audited at the full rate.
fn record_reputation(state: &AppState, node_id: &str, accepted: bool) -> Result<(), ApiError> {
    let mut current = load_reputation(state, node_id)?;
    if accepted {
        current.record_accepted();
    } else {
        current.record_rejected();
    }
    let conn = state.db.get().map_err(ApiError::internal)?;
    conn.execute(
        "INSERT INTO reputation(node_id,accepted,rejected,streak) VALUES(?1,?2,?3,?4)
         ON CONFLICT(node_id) DO UPDATE SET accepted=?2,rejected=?3,streak=?4",
        params![
            node_id,
            current.accepted as i64,
            current.rejected as i64,
            i64::from(current.streak)
        ],
    )
    .map_err(ApiError::internal)?;
    Ok(())
}
fn cache_balance(state: &AppState, node_id: &str, balance: i64) -> Result<(), ApiError> {
    let conn = state.db.get().map_err(ApiError::internal)?;
    conn.execute(
        "UPDATE balances SET balance_mcu=?2 WHERE node_id=?1",
        params![node_id, balance],
    )
    .map_err(ApiError::internal)?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod ai_api_tests {
    use super::*;
    use hocmesh_ai::{
        FailInferenceRequest, InferenceRequirements, PollInferenceRequest, PromptOutput,
        RegisterModelRequest, ReportInferenceRequest, SubmitInferenceRequest,
        fail_inference_body_hash, register_model_body_hash, report_inference_body_hash,
        submit_inference_body_hash,
    };
    use hocmesh_core::identity::NodeIdentity;
    use hocmesh_model::{ChunkRef, ModelFormat, sha256};
    use hocmesh_protocol::{RegisterRequest, register_body_hash};
    use serde::{Serialize, de::DeserializeOwned};
    use std::collections::HashSet;
    use std::fs;

    #[tokio::test]
    async fn distributed_ai_job_reroutes_and_completes() {
        let root = std::env::temp_dir().join(format!("hocmesh-ai-api-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let db_path = root.join("coordinator.db");
        let state = AppState {
            db: Arc::new(crate::db::Pool::open(db_path.to_str().unwrap()).unwrap()),
            ledger: None,
            federation: None,
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server =
            tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });
        let base = format!("http://{address}");
        let requester = NodeIdentity::load_or_create(&root.join("requester")).unwrap();
        let worker_a = NodeIdentity::load_or_create(&root.join("worker-a")).unwrap();
        let worker_b = NodeIdentity::load_or_create(&root.join("worker-b")).unwrap();
        register_test_node(&base, &requester, test_capabilities(false, 0)).await;
        register_test_node(&base, &worker_a, test_capabilities(true, 1_000)).await;
        register_test_node(&base, &worker_b, test_capabilities(true, 2_000)).await;
        // Inference is bought, not free: the requester has to be holding CU
        // before it can reserve a job, exactly like any other workload.
        Connection::open(&db_path)
            .unwrap()
            .execute(
                "UPDATE balances SET balance_mcu=?2 WHERE node_id=?1",
                params![requester.node_id(), 1_000_000i64],
            )
            .unwrap();

        let manifest = ModelManifest {
            schema_version: 1,
            model_id: "tiny".into(),
            revision: "v1".into(),
            format: ModelFormat::Gguf,
            architecture: "llama".into(),
            parameter_count: Some(1),
            tensor_dtype: Some("q4".into()),
            total_size_bytes: 1,
            chunks: vec![ChunkRef {
                index: 0,
                sha256: sha256(b"x"),
                size_bytes: 1,
            }],
            metadata: Default::default(),
        };
        let hash = register_model_body_hash(&manifest).unwrap();
        let publish = RegisterModelRequest {
            auth: requester.auth("register_model", &hash),
            manifest: manifest.clone(),
        };
        let _: RegisterModelResponse = post(&base, "/v1/ai/models/register", &publish).await;

        let mut submit = SubmitInferenceRequest {
            auth: requester.auth("unused", &empty_body_hash()),
            model_id: "tiny".into(),
            revision: "v1".into(),
            prompts: vec!["hello".into()],
            max_tokens: 4,
            temperature_milli: 0,
            seed: 7,
            requirements: InferenceRequirements {
                required_backends: [BackendKind::Cuda].into_iter().collect(),
                minimum_memory_bytes: 1,
                needs_fp16: true,
                needs_bf16: false,
                needs_int8: false,
                batch_size: 1,
                pipeline_stages: 1,
                tensor_parallelism: 1,
            },
            layer_count: 2,
            billing: hocmesh_ai::bill_for_prompts(
                &manifest.digest().unwrap(),
                1,
                1,
                &["hello".into()],
                4,
            )
            .unwrap(),
        };
        submit.auth = requester.auth(
            "submit_inference",
            &submit_inference_body_hash(&submit).unwrap(),
        );
        let submitted: SubmitInferenceResponse = post(&base, "/v1/ai/jobs/submit", &submit).await;
        assert_eq!(submitted.assignments, 1);

        let poll_a = PollInferenceRequest {
            auth: worker_a.auth("poll_inference", &empty_body_hash()),
        };
        let leased: PollInferenceResponse = post(&base, "/v1/ai/work/poll", &poll_a).await;
        let assignment = leased.assignment.unwrap();
        let mut failure = FailInferenceRequest {
            auth: worker_a.auth("unused", &empty_body_hash()),
            assignment_id: assignment.assignment_id.clone(),
            reason: "device lost".into(),
        };
        failure.auth = worker_a.auth(
            "fail_inference",
            &fail_inference_body_hash(&failure).unwrap(),
        );
        let rerouted: FailInferenceResponse = post(&base, "/v1/ai/work/fail", &failure).await;
        assert_eq!(rerouted.rerouted_to, worker_b.node_id());

        let poll_b = PollInferenceRequest {
            auth: worker_b.auth("poll_inference", &empty_body_hash()),
        };
        let leased: PollInferenceResponse = post(&base, "/v1/ai/work/poll", &poll_b).await;
        assert_eq!(
            leased.assignment.unwrap().assignment_id,
            assignment.assignment_id
        );
        let output = PromptOutput {
            prompt_index: 0,
            text: "world".into(),
            output_sha256: hocmesh_protocol::hash_bytes(b"world"),
            duration_ms: 1,
        };
        let (batch_start, batch_end, reward_mcu) =
            hocmesh_ai::assignment_claim(&assignment).unwrap();
        let mut report = ReportInferenceRequest {
            auth: worker_b.auth("unused", &empty_body_hash()),
            assignment_id: assignment.assignment_id.clone(),
            job_id: submitted.job_id.clone(),
            batch_start,
            batch_end,
            reward_mcu,
            outputs: vec![output.clone()],
        };
        report.auth = worker_b.auth(
            "report_inference",
            &report_inference_body_hash(&report).unwrap(),
        );
        let accepted: ReportInferenceResponse = post(&base, "/v1/ai/work/result", &report).await;
        assert!(accepted.job_completed);

        // Delivered but not yet taken: the job reads as complete and the text
        // is still nobody's to read.
        let status: InferenceJobStatus =
            get(&base, &format!("/v1/ai/jobs/{}", submitted.job_id)).await;
        assert!(status.outputs.is_empty());
        let receipt_req = ReceiptInferenceRequest {
            auth: requester.auth(
                "receipt_inference",
                &hocmesh_protocol::inference_receipt_body_hash(
                    &assignment.assignment_id,
                    &submitted.job_id,
                    batch_start,
                    batch_end,
                    reward_mcu,
                    &hocmesh_protocol::hash_json(&vec![output.clone()]).unwrap(),
                )
                .unwrap(),
            ),
            assignment_id: assignment.assignment_id.clone(),
        };
        let taken: ReceiptInferenceResponse =
            post(&base, "/v1/ai/jobs/receipt", &receipt_req).await;
        assert_eq!(taken.outputs, vec![output.clone()]);
        let status: InferenceJobStatus =
            get(&base, &format!("/v1/ai/jobs/{}", submitted.job_id)).await;
        assert_eq!(status.outputs, vec![output]);

        server.abort();
        let _ = server.await;
        fs::remove_dir_all(root).unwrap();
    }

    /// A dashboard's ledger view is only as honest as its paging, so this
    /// walks the whole thing: newest first, a cursor that reaches the older
    /// page, and no row served twice or skipped between them.
    #[tokio::test]
    async fn history_pages_backwards_without_repeating_or_losing_a_row() {
        let root = std::env::temp_dir().join(format!("hocmesh-history-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let db_path = root.join("coordinator.db");
        let state = AppState {
            db: Arc::new(crate::db::Pool::open(db_path.to_str().unwrap()).unwrap()),
            ledger: None,
            federation: None,
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server =
            tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });
        let base = format!("http://{address}");
        let node = NodeIdentity::load_or_create(&root.join("node")).unwrap();
        register_test_node(&base, &node, test_capabilities(false, 0)).await;

        {
            let conn = Connection::open(&db_path).unwrap();
            for index in 0..7i64 {
                conn.execute(
                    "INSERT INTO ledger(node_id,delta_mcu,category,job_id,assignment_id,created_at) \
                     VALUES(?1,?2,'reward',?3,NULL,?4)",
                    params![node.node_id(), index + 1, format!("job-{index}"), index],
                )
                .unwrap();
            }
        }

        let first: LedgerHistoryResponse = get(
            &base,
            &format!("/v1/nodes/{}/history?limit=4", node.node_id()),
        )
        .await;
        assert!(
            !first.authoritative,
            "a coordinator with no validators must not present its own table as settled"
        );
        assert_eq!(first.entries.len(), 4);
        assert_eq!(
            first.entries[0].delta_mcu, 7,
            "newest first is what a dashboard shows at the top"
        );
        assert_eq!(first.entries[0].category.as_deref(), Some("reward"));
        assert_eq!(first.entries[0].job_id.as_deref(), Some("job-6"));
        assert!(
            first.entries[0].sequence.is_none(),
            "a local row has no chain position, and inventing one would make it look checkable"
        );
        let cursor = first.next_before.expect("three entries are still older");

        let second: LedgerHistoryResponse = get(
            &base,
            &format!(
                "/v1/nodes/{}/history?limit=4&before={cursor}",
                node.node_id()
            ),
        )
        .await;
        assert_eq!(second.entries.len(), 3);
        assert_eq!(
            second.next_before, None,
            "the start of history is reported as such, not as another page"
        );

        let seen: Vec<i64> = first
            .entries
            .iter()
            .chain(second.entries.iter())
            .map(|entry| entry.delta_mcu)
            .collect();
        assert_eq!(
            seen,
            vec![7, 6, 5, 4, 3, 2, 1],
            "every posting appears exactly once, in order"
        );

        server.abort();
        let _ = server.await;
        fs::remove_dir_all(root).unwrap();
    }

    /// One node's dashboard must not show another node's earnings, and asking
    /// after a node that was never registered is a 404 rather than an empty
    /// page that reads as "you have earned nothing".
    #[tokio::test]
    async fn history_is_scoped_to_one_node_and_unknown_nodes_are_not_found() {
        let root = std::env::temp_dir().join(format!("hocmesh-history-scope-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let db_path = root.join("coordinator.db");
        let state = AppState {
            db: Arc::new(crate::db::Pool::open(db_path.to_str().unwrap()).unwrap()),
            ledger: None,
            federation: None,
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server =
            tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });
        let base = format!("http://{address}");
        let mine = NodeIdentity::load_or_create(&root.join("mine")).unwrap();
        let theirs = NodeIdentity::load_or_create(&root.join("theirs")).unwrap();
        register_test_node(&base, &mine, test_capabilities(false, 0)).await;
        register_test_node(&base, &theirs, test_capabilities(false, 0)).await;

        {
            let conn = Connection::open(&db_path).unwrap();
            for (owner, delta) in [(mine.node_id(), 10i64), (theirs.node_id(), 99)] {
                conn.execute(
                    "INSERT INTO ledger(node_id,delta_mcu,category,job_id,assignment_id,created_at) \
                     VALUES(?1,?2,'reward',NULL,NULL,1)",
                    params![owner, delta],
                )
                .unwrap();
            }
        }

        let page: LedgerHistoryResponse =
            get(&base, &format!("/v1/nodes/{}/history", mine.node_id())).await;
        assert_eq!(page.entries.len(), 1);
        assert_eq!(page.entries[0].delta_mcu, 10);

        let response = reqwest::Client::new()
            .get(format!("{base}/v1/nodes/node-that-never-was/history"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), 404);

        server.abort();
        let _ = server.await;
        fs::remove_dir_all(root).unwrap();
    }

    /// An unbounded page would let one request pull a whole ledger into
    /// memory, so the ceiling has to hold whatever the caller asks for.
    #[tokio::test]
    async fn an_oversized_page_request_is_capped_rather_than_honoured() {
        let root = std::env::temp_dir().join(format!("hocmesh-history-cap-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let db_path = root.join("coordinator.db");
        let state = AppState {
            db: Arc::new(crate::db::Pool::open(db_path.to_str().unwrap()).unwrap()),
            ledger: None,
            federation: None,
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server =
            tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });
        let base = format!("http://{address}");
        let node = NodeIdentity::load_or_create(&root.join("node")).unwrap();
        register_test_node(&base, &node, test_capabilities(false, 0)).await;

        {
            let conn = Connection::open(&db_path).unwrap();
            for index in 0..(MAX_HISTORY_PAGE as i64 + 20) {
                conn.execute(
                    "INSERT INTO ledger(node_id,delta_mcu,category,job_id,assignment_id,created_at) \
                     VALUES(?1,1,'reward',NULL,NULL,?2)",
                    params![node.node_id(), index],
                )
                .unwrap();
            }
        }

        let page: LedgerHistoryResponse = get(
            &base,
            &format!("/v1/nodes/{}/history?limit=100000", node.node_id()),
        )
        .await;
        assert_eq!(page.entries.len(), MAX_HISTORY_PAGE as usize);
        assert!(
            page.next_before.is_some(),
            "capping a page must still say that more remains"
        );

        server.abort();
        let _ = server.await;
        fs::remove_dir_all(root).unwrap();
    }

    async fn register_test_node(
        base: &str,
        identity: &NodeIdentity,
        capabilities: NodeCapabilities,
    ) {
        let public_key_b64 = identity.public_key_b64();
        let hash = register_body_hash(&public_key_b64, &capabilities).unwrap();
        let request = RegisterRequest {
            auth: identity.auth("register", &hash),
            public_key_b64,
            capabilities,
        };
        let _: RegisterResponse = post(base, "/v1/nodes/register", &request).await;
    }

    fn test_capabilities(ai_ready: bool, latency: u64) -> NodeCapabilities {
        NodeCapabilities {
            protocol_version: hocmesh_protocol::PROTOCOL_VERSION,
            hostname: "test".into(),
            os: "test".into(),
            arch: "test".into(),
            cpu_brand: "test".into(),
            logical_cpus: 1,
            total_memory_bytes: 1024,
            cpu_benchmark_score: 1,
            memory_bandwidth_bytes_per_second: None,
            gpus: if ai_ready {
                vec![hocmesh_protocol::GpuCapability {
                    stable_id: format!("gpu-{latency}"),
                    vendor: "nvidia".into(),
                    name: "test".into(),
                    backend: "cuda".into(),
                    memory_mb: Some(1024),
                    driver_version: None,
                    compute_version: Some("8.0".into()),
                    supports_fp16: true,
                    supports_bf16: true,
                    supports_int8: true,
                    benchmark_bytes_per_second: Some(1),
                    benchmark_p95_micros: Some(1),
                }]
            } else {
                Vec::new()
            },
            model_seed_url: ai_ready.then(|| format!("http://seed-{latency}")),
            cached_model_manifests: Vec::new(),
            coordinator_latency_micros: latency,
            model_bandwidth_kbps: 100_000,
            load_permille: 0,
            ai_runtime_ready: ai_ready,
            shared_logical_cpus: 4,
            shared_memory_bytes: 8 * 1024 * 1024 * 1024,
            shared_gpu_percent: if ai_ready { 100 } else { 0 },
            network_coordinate: None,
            probe_endpoint: None,
        }
    }

    async fn post<T: Serialize + ?Sized, R: DeserializeOwned>(
        base: &str,
        path: &str,
        body: &T,
    ) -> R {
        let response = reqwest::Client::new()
            .post(format!("{base}{path}"))
            .json(body)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let text = response.text().await.unwrap();
        assert!(status.is_success(), "{status}: {text}");
        serde_json::from_str(&text).unwrap()
    }

    /// Refusals are half of what a refund endpoint promises, so this reads
    /// the status and the reason rather than unwrapping its way past them.
    /// Contributing a GPU has to earn what using one costs.
    ///
    /// Inference used to run entirely outside the ledger: a provider burned
    /// real electricity and earned nothing, a requester consumed somebody
    /// else's hardware and paid nothing. That made the whole point of the
    /// network - trade my idle GPU for your idle GPU - unenforceable. This
    /// test walks one job end to end and checks the CU actually moved, in the
    /// right amounts, and that nobody can move more than the request priced.
    #[tokio::test]
    async fn an_inference_job_is_bought_and_paid_for() {
        let root = std::env::temp_dir().join(format!("hocmesh-ai-econ-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let db_path = root.join("coordinator.db");
        let state = AppState {
            db: Arc::new(crate::db::Pool::open(db_path.to_str().unwrap()).unwrap()),
            ledger: None,
            federation: None,
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server =
            tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });
        let base = format!("http://{address}");
        let requester = NodeIdentity::load_or_create(&root.join("requester")).unwrap();
        let provider = NodeIdentity::load_or_create(&root.join("provider")).unwrap();
        register_test_node(&base, &requester, test_capabilities(false, 0)).await;
        register_test_node(&base, &provider, test_capabilities(true, 1_000)).await;

        // A 7B model is not cheap: one prompt runs into six figures of CU.
        let opening_mcu = 1_000_000_000i64;
        Connection::open(&db_path)
            .unwrap()
            .execute(
                "UPDATE balances SET balance_mcu=?2 WHERE node_id=?1",
                params![requester.node_id(), opening_mcu],
            )
            .unwrap();

        let manifest = ModelManifest {
            schema_version: 1,
            model_id: "priced".into(),
            revision: "v1".into(),
            format: ModelFormat::Gguf,
            architecture: "llama".into(),
            parameter_count: Some(7_000_000_000),
            tensor_dtype: Some("q4".into()),
            total_size_bytes: 4_000_000_000,
            chunks: vec![ChunkRef {
                index: 0,
                sha256: sha256(b"weights"),
                size_bytes: 4_000_000_000,
            }],
            metadata: Default::default(),
        };
        let hash = register_model_body_hash(&manifest).unwrap();
        let publish = RegisterModelRequest {
            auth: requester.auth("register_model", &hash),
            manifest: manifest.clone(),
        };
        let _: RegisterModelResponse = post(&base, "/v1/ai/models/register", &publish).await;

        // The requester prices its own job from the published manifest.
        let digest = manifest.digest().unwrap();
        let prompts = vec!["explain compute units".to_string()];
        let billing = hocmesh_ai::bill_for_prompts(
            &digest,
            manifest.parameter_count.unwrap(),
            manifest.total_size_bytes,
            &prompts,
            64,
        )
        .unwrap();
        let price = billing.max_cost_mcu;
        assert!(price > 0, "a real model must cost real CU");

        let mut submit = SubmitInferenceRequest {
            auth: requester.auth("unused", &empty_body_hash()),
            model_id: "priced".into(),
            revision: "v1".into(),
            prompts: prompts.clone(),
            max_tokens: 64,
            temperature_milli: 0,
            seed: 7,
            requirements: InferenceRequirements {
                required_backends: [BackendKind::Cuda].into_iter().collect(),
                minimum_memory_bytes: 1,
                needs_fp16: true,
                needs_bf16: false,
                needs_int8: false,
                batch_size: 1,
                pipeline_stages: 1,
                tensor_parallelism: 1,
            },
            layer_count: 2,
            billing: billing.clone(),
        };
        submit.auth = requester.auth(
            "submit_inference",
            &submit_inference_body_hash(&submit).unwrap(),
        );
        let submitted: SubmitInferenceResponse = post(&base, "/v1/ai/jobs/submit", &submit).await;

        // Escrow is funded out of the requester, to the exact number it signed.
        let after_submit = read_balance(&db_path, &requester.node_id());
        assert_eq!(after_submit, opening_mcu - price);

        let poll = PollInferenceRequest {
            auth: provider.auth("poll_inference", &empty_body_hash()),
        };
        let leased: PollInferenceResponse = post(&base, "/v1/ai/work/poll", &poll).await;
        let assignment = leased.assignment.unwrap();
        let (batch_start, batch_end, reward_mcu) =
            hocmesh_ai::assignment_claim(&assignment).unwrap();

        // One batch covers the whole job here, so it is worth the whole price:
        // batch prices tile the job exactly, with nothing stranded in escrow.
        assert_eq!(reward_mcu, price);

        let output = PromptOutput {
            prompt_index: 0,
            text: "a unit of machine work".into(),
            output_sha256: hocmesh_protocol::hash_bytes(b"a unit of machine work"),
            duration_ms: 1,
        };

        // A provider that signs for more than its batch is worth is refused.
        // The price is closed form from the request, so the coordinator does
        // not have to take the claim on faith.
        let mut greedy = ReportInferenceRequest {
            auth: provider.auth("unused", &empty_body_hash()),
            assignment_id: assignment.assignment_id.clone(),
            job_id: submitted.job_id.clone(),
            batch_start,
            batch_end,
            reward_mcu: reward_mcu * 10,
            outputs: vec![output.clone()],
        };
        greedy.auth = provider.auth(
            "report_inference",
            &report_inference_body_hash(&greedy).unwrap(),
        );
        let (status, body) = post_raw(&base, "/v1/ai/work/result", &greedy).await;
        assert_eq!(status, 409, "inflated reward accepted: {body}");

        // The honest claim settles.
        let mut report = ReportInferenceRequest {
            auth: provider.auth("unused", &empty_body_hash()),
            assignment_id: assignment.assignment_id.clone(),
            job_id: submitted.job_id.clone(),
            batch_start,
            batch_end,
            reward_mcu,
            outputs: vec![output.clone()],
        };
        report.auth = provider.auth(
            "report_inference",
            &report_inference_body_hash(&report).unwrap(),
        );
        let paid: ReportInferenceResponse = post(&base, "/v1/ai/work/result", &report).await;
        assert_eq!(paid.reward_mcu, price);

        // Delivery is not payment. Until the requester takes the answer and
        // says what it is worth, the provider has earned nothing at all.
        assert_eq!(read_balance(&db_path, &provider.node_id()), 0);

        // The status endpoint hands out digests, never text, before a receipt.
        let pending: InferenceJobStatus =
            get(&base, &format!("/v1/ai/jobs/{}", submitted.job_id)).await;
        assert!(pending.outputs.is_empty());
        assert_eq!(pending.delivered.len(), 1);
        assert!(!pending.delivered[0].receipted);

        // Taking delivery moves the escrow into holding and returns the text.
        let receipt_req = ReceiptInferenceRequest {
            auth: requester.auth(
                "receipt_inference",
                &hocmesh_protocol::inference_receipt_body_hash(
                    &assignment.assignment_id,
                    &submitted.job_id,
                    batch_start,
                    batch_end,
                    price,
                    &hocmesh_protocol::hash_json(&vec![output.clone()]).unwrap(),
                )
                .unwrap(),
            ),
            assignment_id: assignment.assignment_id.clone(),
        };
        let taken: ReceiptInferenceResponse =
            post(&base, "/v1/ai/jobs/receipt", &receipt_req).await;
        assert_eq!(taken.outputs, vec![output.clone()]);
        assert_eq!(taken.price_mcu, price);
        assert_eq!(read_balance(&db_path, &provider.node_id()), 0);

        // Only the requester's signed acceptance pays the provider.
        let settle_req = SettleInferenceRequest {
            auth: requester.auth(
                "accept_inference",
                &hocmesh_protocol::inference_verdict_body_hash(
                    true,
                    &assignment.assignment_id,
                    &submitted.job_id,
                    batch_start,
                    batch_end,
                    price,
                    &taken.outputs_digest,
                )
                .unwrap(),
            ),
            assignment_id: assignment.assignment_id.clone(),
            accepted: true,
            reason: String::new(),
        };
        let settled: SettleInferenceResponse = post(&base, "/v1/ai/jobs/settle", &settle_req).await;
        assert_eq!(settled.paid_mcu, price);
        assert!(settled.job_completed);

        // The GPU earned exactly what the requester spent. Nothing was minted
        // and nothing evaporated: this is the trade the whole network is for.
        let provider_balance = read_balance(&db_path, &provider.node_id());
        let requester_balance = read_balance(&db_path, &requester.node_id());
        assert_eq!(provider_balance, price);
        assert_eq!(requester_balance, opening_mcu - price);
        assert_eq!(
            (requester_balance - opening_mcu) + provider_balance,
            0,
            "CU was created or destroyed by an inference job"
        );

        // Paid once, paid never again. Re-signed with a fresh nonce so the
        // replay guard cannot answer for the settlement rule: what refuses the
        // second payment is the batch itself already being settled.
        report.auth = provider.auth(
            "report_inference",
            &report_inference_body_hash(&report).unwrap(),
        );
        let (status, body) = post_raw(&base, "/v1/ai/work/result", &report).await;
        assert_eq!(status, 409, "batch paid twice: {body}");
        assert_eq!(read_balance(&db_path, &provider.node_id()), price);

        server.abort();
        let _ = fs::remove_dir_all(&root);
    }

    /// The other half of the exchange: a real assignment, arbitrary bytes, and
    /// a requester that refuses to pay for them.
    #[tokio::test]
    async fn a_disputed_answer_pays_the_provider_nothing() {
        let root = std::env::temp_dir().join(format!("hocmesh-ai-dispute-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let db_path = root.join("coordinator.db");
        let state = AppState {
            db: Arc::new(crate::db::Pool::open(db_path.to_str().unwrap()).unwrap()),
            ledger: None,
            federation: None,
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server =
            tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });
        let base = format!("http://{address}");
        let requester = NodeIdentity::load_or_create(&root.join("requester")).unwrap();
        let provider = NodeIdentity::load_or_create(&root.join("provider")).unwrap();
        register_test_node(&base, &requester, test_capabilities(false, 0)).await;
        register_test_node(&base, &provider, test_capabilities(true, 1_000)).await;

        // A 7B model is not cheap: one prompt runs into six figures of CU.
        let opening_mcu = 1_000_000_000i64;
        Connection::open(&db_path)
            .unwrap()
            .execute(
                "UPDATE balances SET balance_mcu=?2 WHERE node_id=?1",
                params![requester.node_id(), opening_mcu],
            )
            .unwrap();

        let manifest = ModelManifest {
            schema_version: 1,
            model_id: "priced".into(),
            revision: "v1".into(),
            format: ModelFormat::Gguf,
            architecture: "llama".into(),
            parameter_count: Some(7_000_000_000),
            tensor_dtype: Some("q4".into()),
            total_size_bytes: 4_000_000_000,
            chunks: vec![ChunkRef {
                index: 0,
                sha256: sha256(b"weights"),
                size_bytes: 4_000_000_000,
            }],
            metadata: Default::default(),
        };
        let hash = register_model_body_hash(&manifest).unwrap();
        let publish = RegisterModelRequest {
            auth: requester.auth("register_model", &hash),
            manifest: manifest.clone(),
        };
        let _: RegisterModelResponse = post(&base, "/v1/ai/models/register", &publish).await;

        // The requester prices its own job from the published manifest.
        let digest = manifest.digest().unwrap();
        let prompts = vec!["explain compute units".to_string()];
        let billing = hocmesh_ai::bill_for_prompts(
            &digest,
            manifest.parameter_count.unwrap(),
            manifest.total_size_bytes,
            &prompts,
            64,
        )
        .unwrap();
        let price = billing.max_cost_mcu;
        assert!(price > 0, "a real model must cost real CU");

        let mut submit = SubmitInferenceRequest {
            auth: requester.auth("unused", &empty_body_hash()),
            model_id: "priced".into(),
            revision: "v1".into(),
            prompts: prompts.clone(),
            max_tokens: 64,
            temperature_milli: 0,
            seed: 7,
            requirements: InferenceRequirements {
                required_backends: [BackendKind::Cuda].into_iter().collect(),
                minimum_memory_bytes: 1,
                needs_fp16: true,
                needs_bf16: false,
                needs_int8: false,
                batch_size: 1,
                pipeline_stages: 1,
                tensor_parallelism: 1,
            },
            layer_count: 2,
            billing: billing.clone(),
        };
        submit.auth = requester.auth(
            "submit_inference",
            &submit_inference_body_hash(&submit).unwrap(),
        );
        let submitted: SubmitInferenceResponse = post(&base, "/v1/ai/jobs/submit", &submit).await;

        // Escrow is funded out of the requester, to the exact number it signed.
        let after_submit = read_balance(&db_path, &requester.node_id());
        assert_eq!(after_submit, opening_mcu - price);

        let poll = PollInferenceRequest {
            auth: provider.auth("poll_inference", &empty_body_hash()),
        };
        let leased: PollInferenceResponse = post(&base, "/v1/ai/work/poll", &poll).await;
        let assignment = leased.assignment.unwrap();
        let (batch_start, batch_end, reward_mcu) =
            hocmesh_ai::assignment_claim(&assignment).unwrap();

        // One batch covers the whole job here, so it is worth the whole price:
        // batch prices tile the job exactly, with nothing stranded in escrow.
        assert_eq!(reward_mcu, price);

        let output = PromptOutput {
            prompt_index: 0,
            text: "not an answer to anything".into(),
            output_sha256: hocmesh_protocol::hash_bytes(b"not an answer to anything"),
            duration_ms: 1,
        };
        let mut report = ReportInferenceRequest {
            auth: provider.auth("unused", &empty_body_hash()),
            assignment_id: assignment.assignment_id.clone(),
            job_id: submitted.job_id.clone(),
            batch_start,
            batch_end,
            reward_mcu,
            outputs: vec![output.clone()],
        };
        report.auth = provider.auth(
            "report_inference",
            &report_inference_body_hash(&report).unwrap(),
        );
        let delivered: ReportInferenceResponse = post(&base, "/v1/ai/work/result", &report).await;
        assert!(delivered.accepted);
        assert_eq!(read_balance(&db_path, &provider.node_id()), 0);

        // The requester takes delivery - it has to, to see what it bought -
        // and finds the answer is nothing of the kind.
        let receipt_req = ReceiptInferenceRequest {
            auth: requester.auth(
                "receipt_inference",
                &hocmesh_protocol::inference_receipt_body_hash(
                    &assignment.assignment_id,
                    &submitted.job_id,
                    batch_start,
                    batch_end,
                    price,
                    &hocmesh_protocol::hash_json(&vec![output.clone()]).unwrap(),
                )
                .unwrap(),
            ),
            assignment_id: assignment.assignment_id.clone(),
        };
        let taken: ReceiptInferenceResponse =
            post(&base, "/v1/ai/jobs/receipt", &receipt_req).await;
        assert_eq!(taken.outputs, vec![output.clone()]);
        let dispute = SettleInferenceRequest {
            auth: requester.auth(
                "dispute_inference",
                &hocmesh_protocol::inference_verdict_body_hash(
                    false,
                    &assignment.assignment_id,
                    &submitted.job_id,
                    batch_start,
                    batch_end,
                    price,
                    &taken.outputs_digest,
                )
                .unwrap(),
            ),
            assignment_id: assignment.assignment_id.clone(),
            accepted: false,
            reason: "the model was never run".into(),
        };
        let rejected: SettleInferenceResponse = post(&base, "/v1/ai/jobs/settle", &dispute).await;
        assert!(!rejected.accepted);

        // Returning junk earned nothing, and the requester is no better off for
        // having said so: the CU left its account either way. Neither side can
        // profit by lying about what the answer was worth.
        assert_eq!(read_balance(&db_path, &provider.node_id()), 0);
        assert_eq!(
            read_balance(&db_path, &requester.node_id()),
            opening_mcu - price
        );

        // One payout per batch: having disputed it, the requester cannot turn
        // round and pay for it after all.
        let (status, body) = post_raw(&base, "/v1/ai/jobs/settle", &dispute).await;
        assert_eq!(status, 409, "a batch settled twice: {body}");

        server.abort();
        let _ = fs::remove_dir_all(&root);
    }

    /// Escrow that can only pay out is a one-way valve.
    ///
    /// A GPU that takes a batch and never answers would otherwise keep the CU
    /// that funded it locked away forever, and a requester would learn to
    /// never submit a second job. The refund is the other direction, and it
    /// shares a claim key with the reward so a batch settles exactly once.
    #[tokio::test]
    async fn an_undelivered_batch_returns_its_escrow() {
        let root = std::env::temp_dir().join(format!("hocmesh-ai-refund-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let db_path = root.join("coordinator.db");
        let state = AppState {
            db: Arc::new(crate::db::Pool::open(db_path.to_str().unwrap()).unwrap()),
            ledger: None,
            federation: None,
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server =
            tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });
        let base = format!("http://{address}");
        let requester = NodeIdentity::load_or_create(&root.join("requester")).unwrap();
        let provider = NodeIdentity::load_or_create(&root.join("provider")).unwrap();
        register_test_node(&base, &requester, test_capabilities(false, 0)).await;
        register_test_node(&base, &provider, test_capabilities(true, 1_000)).await;

        let opening_mcu = 1_000_000_000i64;
        Connection::open(&db_path)
            .unwrap()
            .execute(
                "UPDATE balances SET balance_mcu=?2 WHERE node_id=?1",
                params![requester.node_id(), opening_mcu],
            )
            .unwrap();

        let manifest = ModelManifest {
            schema_version: 1,
            model_id: "priced".into(),
            revision: "v1".into(),
            format: ModelFormat::Gguf,
            architecture: "llama".into(),
            parameter_count: Some(7_000_000_000),
            tensor_dtype: Some("q4".into()),
            total_size_bytes: 4_000_000_000,
            chunks: vec![ChunkRef {
                index: 0,
                sha256: sha256(b"weights"),
                size_bytes: 4_000_000_000,
            }],
            metadata: Default::default(),
        };
        let hash = register_model_body_hash(&manifest).unwrap();
        let publish = RegisterModelRequest {
            auth: requester.auth("register_model", &hash),
            manifest: manifest.clone(),
        };
        let _: RegisterModelResponse = post(&base, "/v1/ai/models/register", &publish).await;

        let prompts = vec!["never answered".to_string()];
        let billing = hocmesh_ai::bill_for_prompts(
            &manifest.digest().unwrap(),
            manifest.parameter_count.unwrap(),
            manifest.total_size_bytes,
            &prompts,
            64,
        )
        .unwrap();
        let price = billing.max_cost_mcu;

        let mut submit = SubmitInferenceRequest {
            auth: requester.auth("unused", &empty_body_hash()),
            model_id: "priced".into(),
            revision: "v1".into(),
            prompts: prompts.clone(),
            max_tokens: 64,
            temperature_milli: 0,
            seed: 7,
            requirements: InferenceRequirements {
                required_backends: [BackendKind::Cuda].into_iter().collect(),
                minimum_memory_bytes: 1,
                needs_fp16: true,
                needs_bf16: false,
                needs_int8: false,
                batch_size: 1,
                pipeline_stages: 1,
                tensor_parallelism: 1,
            },
            layer_count: 2,
            billing: billing.clone(),
        };
        submit.auth = requester.auth(
            "submit_inference",
            &submit_inference_body_hash(&submit).unwrap(),
        );
        let submitted: SubmitInferenceResponse = post(&base, "/v1/ai/jobs/submit", &submit).await;
        assert_eq!(
            read_balance(&db_path, &requester.node_id()),
            opening_mcu - price
        );

        // The provider takes the batch and is never heard from again.
        let poll = PollInferenceRequest {
            auth: provider.auth("poll_inference", &empty_body_hash()),
        };
        let leased: PollInferenceResponse = post(&base, "/v1/ai/work/poll", &poll).await;
        // The lease is taken; what happens next is nothing at all.
        assert!(leased.assignment.is_some());

        // Inside the window the escrow is the provider's to earn, so there is
        // nothing to reclaim and the coordinator says so.
        let fresh: hocmesh_ai::InferenceJobStatus =
            reqwest::get(format!("{base}/v1/ai/jobs/{}", submitted.job_id))
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
        assert!(fresh.refundable.is_empty());

        // Wind the reservation back past its settlement window.
        Connection::open(&db_path)
            .unwrap()
            .execute(
                "UPDATE ai_jobs SET created_at=?1 WHERE job_id=?2",
                params![now_unix() - SETTLEMENT_WINDOW_SECS - 1, submitted.job_id],
            )
            .unwrap();

        let stale: hocmesh_ai::InferenceJobStatus =
            reqwest::get(format!("{base}/v1/ai/jobs/{}", submitted.job_id))
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
        assert_eq!(stale.refundable.len(), 1);
        let batch = stale.refundable[0].clone();
        assert_eq!(batch.refund_mcu, price);

        let mut refund = RefundInferenceRequest {
            auth: requester.auth("unused", &empty_body_hash()),
            job_id: submitted.job_id.clone(),
            assignment_id: batch.assignment_id.clone(),
            batch_start: batch.batch_start,
            batch_end: batch.batch_end,
            refund_mcu: batch.refund_mcu,
        };

        // A bystander cannot reclaim somebody else's escrow, even with the
        // right numbers: the CU goes back where it came from or nowhere.
        let mut thief = refund.clone();
        thief.auth = provider.auth(
            "refund_inference",
            &refund_inference_body_hash(&thief).unwrap(),
        );
        let (status, body) = post_raw(&base, "/v1/ai/jobs/refund", &thief).await;
        assert_eq!(status, 401, "escrow refunded to a stranger: {body}");

        refund.auth = requester.auth(
            "refund_inference",
            &refund_inference_body_hash(&refund).unwrap(),
        );
        let returned: RefundInferenceResponse = post(&base, "/v1/ai/jobs/refund", &refund).await;
        assert_eq!(returned.refunded_mcu, price);
        assert_eq!(read_balance(&db_path, &requester.node_id()), opening_mcu);

        // Reward and refund share a claim key, so the late provider cannot
        // race the requester for escrow that has already gone home.
        let output = PromptOutput {
            prompt_index: 0,
            text: "too late".into(),
            output_sha256: hocmesh_protocol::hash_bytes(b"too late"),
            duration_ms: 1,
        };
        let mut late = ReportInferenceRequest {
            auth: provider.auth("unused", &empty_body_hash()),
            assignment_id: batch.assignment_id.clone(),
            job_id: submitted.job_id.clone(),
            batch_start: batch.batch_start,
            batch_end: batch.batch_end,
            reward_mcu: batch.refund_mcu,
            outputs: vec![output],
        };
        late.auth = provider.auth(
            "report_inference",
            &report_inference_body_hash(&late).unwrap(),
        );
        let (status, body) = post_raw(&base, "/v1/ai/work/result", &late).await;
        assert_eq!(status, 409, "refunded batch paid out anyway: {body}");
        assert_eq!(read_balance(&db_path, &requester.node_id()), opening_mcu);
        assert_eq!(read_balance(&db_path, &provider.node_id()), 0);

        server.abort();
        let _ = fs::remove_dir_all(&root);
    }

    fn read_balance(db_path: &std::path::Path, node_id: &str) -> i64 {
        Connection::open(db_path)
            .unwrap()
            .query_row(
                "SELECT balance_mcu FROM balances WHERE node_id=?1",
                params![node_id],
                |row| row.get(0),
            )
            .unwrap()
    }

    async fn post_raw<T: Serialize + ?Sized>(base: &str, path: &str, body: &T) -> (u16, String) {
        let response = reqwest::Client::new()
            .post(format!("{base}{path}"))
            .json(body)
            .send()
            .await
            .unwrap();
        let status = response.status().as_u16();
        (status, response.text().await.unwrap())
    }

    /// An escrow that can only ever pay out is a trap: a shard nobody
    /// delivers takes the CU that funded it with it. The refund turns that
    /// one-way valve into a loop, so this walks the whole way round it -
    /// pay in, let the window close on nothing, take it back - and checks
    /// that what comes out is exactly what went in.
    #[tokio::test]
    async fn escrow_for_a_shard_nobody_delivers_comes_back_to_the_requester() {
        let root = std::env::temp_dir().join(format!("hocmesh-refund-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let db_path = root.join("coordinator.db");
        let state = AppState {
            db: Arc::new(crate::db::Pool::open(db_path.to_str().unwrap()).unwrap()),
            ledger: None,
            federation: None,
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server =
            tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });
        let base = format!("http://{address}");

        let requester = NodeIdentity::load_or_create(&root.join("requester")).unwrap();
        register_test_node(&base, &requester, test_capabilities(false, 0)).await;
        // Earning CU is a different story; this one starts with a node that
        // already has some to spend.
        let opening_mcu = 1_000_000i64;
        let conn = Connection::open(&db_path).unwrap();
        conn.execute(
            "UPDATE balances SET balance_mcu=?2 WHERE node_id=?1",
            params![requester.node_id(), opening_mcu],
        )
        .unwrap();
        drop(conn);

        let work = WorkSpec::PrimeCount {
            start: 2,
            end: 40_000,
        };
        let body_hash = submit_body_hash(&work, 2).unwrap();
        let submitted: SubmitJobResponse = post(
            &base,
            "/v1/jobs/submit",
            &SubmitJobRequest {
                auth: requester.auth("submit", &body_hash),
                work: work.clone(),
                shards: 2,
            },
        )
        .await;
        assert_eq!(submitted.assignments, 2);
        assert_eq!(
            submitted.balance_mcu,
            opening_mcu - submitted.reserved_mcu,
            "submitting has to take the whole reservation out of the requester"
        );

        let shards = split_work(&work, 2);
        let claim = |index: u32| {
            let assignment_id = hocmesh_protocol::assignment_id(&submitted.job_id, index);
            let hash = refund_body_hash(
                &assignment_id,
                &submitted.job_id,
                index,
                &shards[index as usize],
                work_cost_mcu(&shards[index as usize]),
                false,
            )
            .unwrap();
            RefundRequest {
                assignment_id,
                auth: Some(requester.auth("refund", &hash)),
            }
        };

        // The window is the whole protection. While it is open the shard is
        // still somebody else's to finish, and taking the CU back would be
        // taking it out from under them.
        let (status, body) = post_raw(&base, "/v1/work/refund", &claim(0)).await;
        assert_eq!(status, 409, "{body}");
        assert!(
            body.contains("settlement window for this shard has not closed"),
            "{body}"
        );

        // Age the job past its window rather than waiting an hour for it.
        let conn = Connection::open(&db_path).unwrap();
        conn.execute(
            "UPDATE jobs SET created_at=?1 WHERE job_id=?2",
            params![now_unix() - SETTLEMENT_WINDOW_SECS - 1, submitted.job_id],
        )
        .unwrap();
        drop(conn);

        let refunded: RefundResponse = post(&base, "/v1/work/refund", &claim(0)).await;
        assert_eq!(refunded.paid_to, requester.node_id());
        assert_eq!(refunded.refund_mcu, work_cost_mcu(&shards[0]));

        // Escrow moves once and only once, whichever direction it moves in.
        let (status, body) = post_raw(&base, "/v1/work/refund", &claim(0)).await;
        assert_eq!(status, 409, "{body}");
        assert!(body.contains("holds no reclaimable escrow"), "{body}");

        // One shard back does not end the job; the other is still open.
        let midway: JobStatusResponse = get(&base, &format!("/v1/jobs/{}", submitted.job_id)).await;
        assert_eq!(midway.status, "pending", "one live shard keeps a job open");
        // A requester should not have to keep the work spec from the day they
        // submitted to sign for its escrow back, so the coordinator names the
        // shard that is still reclaimable and what it is worth.
        assert_eq!(
            midway.refundable.len(),
            1,
            "the one undelivered shard is the one still on offer"
        );
        assert_eq!(midway.refundable[0].shard_index, 1);
        assert_eq!(
            midway.refundable[0].refund_mcu,
            work_cost_mcu(&shards[1]),
            "what is offered back is what that shard cost"
        );

        let _: RefundResponse = post(&base, "/v1/work/refund", &claim(1)).await;
        let closed: JobStatusResponse = get(&base, &format!("/v1/jobs/{}", submitted.job_id)).await;
        assert_eq!(
            closed.status, "closed",
            "a job that gave every shard back never completed anything"
        );
        assert!(
            closed.refundable.is_empty(),
            "a settled shard is never offered back a second time"
        );

        let ending: BalanceResponse =
            get(&base, &format!("/v1/nodes/{}/balance", requester.node_id())).await;
        assert_eq!(
            ending.balance_mcu, opening_mcu,
            "every unit the job reserved has to come back, to the millicu"
        );

        server.abort();
        let _ = server.await;
        fs::remove_dir_all(root).unwrap();
    }

    /// The peer sample is a bootstrap directory, so it must only name nodes a
    /// probe can actually reach: no endpoint, or gone quiet, means no entry.
    #[tokio::test]
    async fn the_peer_sample_lists_only_nodes_that_can_answer_a_probe() {
        let root = std::env::temp_dir().join(format!("hocmesh-peer-sample-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let db_path = root.join("coordinator.db");
        let state = AppState {
            db: Arc::new(crate::db::Pool::open(db_path.to_str().unwrap()).unwrap()),
            ledger: None,
            federation: None,
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server =
            tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });
        let base = format!("http://{address}");

        // Serving probes is opt-in, so a node that never passed --probe-listen
        // is online and schedulable but useless as a probe target.
        let silent = NodeIdentity::load_or_create(&root.join("silent")).unwrap();
        register_test_node(&base, &silent, test_capabilities(false, 0)).await;

        let stale = NodeIdentity::load_or_create(&root.join("stale")).unwrap();
        register_test_node(
            &base,
            &stale,
            probeable_capabilities("http://127.0.0.1:7001"),
        )
        .await;

        let reachable = NodeIdentity::load_or_create(&root.join("reachable")).unwrap();
        register_test_node(
            &base,
            &reachable,
            probeable_capabilities("http://127.0.0.1:7002"),
        )
        .await;

        // Age the stale node past the online window without waiting for it.
        let conn = Connection::open(&db_path).unwrap();
        conn.execute(
            "UPDATE nodes SET last_seen = ?1 WHERE node_id = ?2",
            params![now_unix() - NODE_ONLINE_SECS - 1, stale.node_id()],
        )
        .unwrap();
        drop(conn);

        let sample: PeerSampleResponse = get(&base, "/v1/network/peers").await;
        let ids: Vec<&str> = sample.peers.iter().map(|p| p.node_id.as_str()).collect();
        let expected = reachable.node_id();
        assert_eq!(
            ids,
            vec![expected.as_str()],
            "only an online node that offered a probe endpoint belongs in the sample"
        );
        let peer = &sample.peers[0];
        assert_eq!(peer.probe_endpoint, "http://127.0.0.1:7002");
        assert_eq!(
            peer.coordinate.as_ref().unwrap().error_permille,
            250,
            "an already-fitted peer must arrive with its confidence intact"
        );

        server.abort();
        let _ = server.await;
        fs::remove_dir_all(root).unwrap();
    }

    /// Every extra target is a real round trip on someone else's machine, so
    /// the sample stays bounded however large the hocmesh grows.
    #[tokio::test]
    async fn the_peer_sample_stays_bounded_as_the_mesh_grows() {
        let root = std::env::temp_dir().join(format!("hocmesh-peer-cap-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let state = AppState {
            db: Arc::new(
                crate::db::Pool::open(root.join("coordinator.db").to_str().unwrap()).unwrap(),
            ),
            ledger: None,
            federation: None,
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server =
            tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });
        let base = format!("http://{address}");

        for index in 0..PEER_SAMPLE_SIZE + 4 {
            let home = root.join(format!("node-{index}"));
            let identity = NodeIdentity::load_or_create(&home).unwrap();
            let endpoint = format!("http://127.0.0.1:{}", 7100 + index);
            register_test_node(&base, &identity, probeable_capabilities(&endpoint)).await;
        }

        let sample: PeerSampleResponse = get(&base, "/v1/network/peers").await;
        assert_eq!(sample.peers.len(), PEER_SAMPLE_SIZE);
        let unique: HashSet<&str> = sample.peers.iter().map(|p| p.node_id.as_str()).collect();
        assert_eq!(
            unique.len(),
            PEER_SAMPLE_SIZE,
            "a peer must not be sampled twice"
        );

        server.abort();
        let _ = server.await;
        fs::remove_dir_all(root).unwrap();
    }

    fn probeable_capabilities(endpoint: &str) -> NodeCapabilities {
        let mut capabilities = test_capabilities(false, 0);
        capabilities.probe_endpoint = Some(endpoint.to_string());
        capabilities.network_coordinate = Some(NetworkCoordinate {
            vector_micros: [1_000, 2_000, 3_000],
            height_micros: 400,
            error_permille: 250,
        });
        capabilities
    }

    async fn get<R: DeserializeOwned>(base: &str, path: &str) -> R {
        let response = reqwest::Client::new()
            .get(format!("{base}{path}"))
            .send()
            .await
            .unwrap();
        let status = response.status();
        let text = response.text().await.unwrap();
        assert!(status.is_success(), "{status}: {text}");
        serde_json::from_str(&text).unwrap()
    }

    /// Scheduling on the coordinator's own view of latency sends work to
    /// whoever is nearest the coordinator. When both ends are placed, the
    /// distance that matters is requester-to-worker, and nothing else.
    #[test]
    fn a_placed_pair_is_scored_on_the_distance_between_them() {
        let requester = NetworkCoordinate {
            vector_micros: [0, 0, 0],
            height_micros: 0,
            error_permille: 100,
        };
        // Far from the requester, but the coordinator sees it as next door.
        let mut worker = test_capabilities(false, 1_000);
        worker.network_coordinate = Some(NetworkCoordinate {
            vector_micros: [40_000, 0, 0],
            height_micros: 2_000,
            error_permille: 200,
        });

        let scored = scoring_latency_ms(Some(&requester), &worker);
        assert!((scored - 42.0).abs() < 0.001, "{scored}");
    }

    /// Most of the hocmesh is unplaced most of the time - a node that has never
    /// probed anyone, a requester that never will. The coordinator's own
    /// measurement is the honest fallback, not a refusal to schedule.
    #[test]
    fn an_unplaced_end_falls_back_to_what_the_coordinator_measured() {
        let placed = NetworkCoordinate {
            vector_micros: [0, 0, 0],
            height_micros: 0,
            error_permille: 100,
        };
        let mut worker = test_capabilities(false, 25_000);
        worker.network_coordinate = Some(placed);

        // A requester who has never been placed.
        assert!((scoring_latency_ms(None, &worker) - 25.0).abs() < 0.001);

        // A worker who has never been placed.
        let unplaced = test_capabilities(false, 25_000);
        assert!((scoring_latency_ms(Some(&placed), &unplaced) - 25.0).abs() < 0.001);
    }

    /// A coordinate is a number a node chose for itself. One that is out of
    /// range would let a worker claim to be next to everybody, so scheduling
    /// has to fall back to what the coordinator measured itself.
    #[test]
    fn a_worker_cannot_score_itself_with_an_implausible_coordinate() {
        let requester = NetworkCoordinate {
            vector_micros: [0, 0, 0],
            height_micros: 0,
            error_permille: 100,
        };
        let claimed = [
            // Right on top of the requester, but claiming impossible confidence.
            NetworkCoordinate {
                vector_micros: [0, 0, 0],
                height_micros: 0,
                error_permille: 1_001,
            },
            // Inside the confidence bound, but off the edge of the map.
            NetworkCoordinate {
                vector_micros: [90_000_000, 0, 0],
                height_micros: 0,
                error_permille: 10,
            },
        ];

        for coordinate in claimed {
            let mut worker = test_capabilities(false, 25_000);
            worker.network_coordinate = Some(coordinate);
            let scored = scoring_latency_ms(Some(&requester), &worker);
            assert!(
                (scored - 25.0).abs() < 0.001,
                "{coordinate:?} scored {scored}, but must not be trusted"
            );
        }
    }

    /// Scoring divides by latency, so a zero would make one node infinitely
    /// attractive and starve every other. Nothing is ever free.
    #[test]
    fn latency_never_scores_as_zero() {
        let origin = NetworkCoordinate {
            vector_micros: [0, 0, 0],
            height_micros: 0,
            error_permille: 0,
        };
        let mut worker = test_capabilities(false, 0);
        assert_eq!(scoring_latency_ms(None, &worker), 0.1);

        worker.network_coordinate = Some(origin);
        assert_eq!(
            scoring_latency_ms(Some(&origin), &worker),
            0.1,
            "two nodes in the same place are still not the same node"
        );
    }
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
        let conn = state.db.get().map_err(ApiError::internal)?;
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
