use crate::error::ApiError;
use axum::{
    Json, Router,
    extract::{Path, State},
    routing::{get, post},
};
use mesh_ai::{
    FailInferenceRequest, FailInferenceResponse, InferenceAssignment, InferenceJobStatus,
    NodeProfile, PlanRequest, PlanResponse, PollInferenceRequest, PollInferenceResponse,
    PromptOutput, RegisterModelRequest, RegisterModelResponse, ReportInferenceRequest,
    ReportInferenceResponse, SubmitInferenceRequest, SubmitInferenceResponse,
    fail_inference_body_hash, plan_body_hash, plan_parallelism, rank_candidates,
    register_model_body_hash, report_inference_body_hash, submit_inference_body_hash,
};
use mesh_core::compute::{split_work, verify_work, work_cost_mcu};
use mesh_core::proximity;
use mesh_gpu::{BackendKind, DeviceCapability};
use mesh_ledger::{
    network::LedgerNetwork,
    types::{
        JobReserveEvidence, LedgerTransaction, Posting, ProviderRewardEvidence,
        TransactionEvidence, TransactionKind, escrow_account,
    },
    validate::claim_key,
};
use mesh_model::ModelManifest;
use mesh_protocol::{
    BalanceResponse, DEFAULT_LEASE_SECONDS, HeartbeatRequest, JobStatusResponse, NetworkCoordinate,
    NetworkStatsResponse, NodeCapabilities, NodeStatusResponse, PeerSample, PeerSampleResponse,
    PollRequest, PollResponse, RegisterRequest, RegisterResponse, ResultRequest, ResultResponse,
    SubmitJobRequest, SubmitJobResponse, WorkAssignment, WorkResult, WorkSpec, empty_body_hash,
    heartbeat_body_hash, job_id_from_auth, now_unix, register_body_hash, result_body_hash,
    submit_body_hash, verify_auth,
};
use rusqlite::{Connection, OptionalExtension, params};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Mutex<Connection>>,
    pub ledger: Option<LedgerNetwork>,
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
        .route("/v1/jobs/submit", post(submit_job))
        .route("/v1/jobs/{id}", get(job_status))
        .route("/v1/nodes/{id}/balance", get(balance))
        .route("/v1/nodes/{id}", get(node_status))
        .route("/v1/network/stats", get(network_stats))
        .route("/v1/network/peers", get(network_peers))
        .route("/v1/ai/models", get(list_models))
        .route("/v1/ai/models/register", post(register_model))
        .route("/v1/ai/models/{model}/{revision}", get(get_model))
        .route("/v1/ai/plan", post(plan_ai))
        .route("/v1/ai/jobs/submit", post(submit_inference))
        .route("/v1/ai/jobs/{id}", get(inference_status))
        .route("/v1/ai/work/poll", post(poll_inference))
        .route("/v1/ai/work/result", post(report_inference))
        .route("/v1/ai/work/fail", post(fail_inference))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
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
    let conn = state.db.lock().map_err(ApiError::internal)?;
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
    let conn = state.db.lock().map_err(ApiError::internal)?;
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
    let conn = state.db.lock().map_err(ApiError::internal)?;
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
        let conn = state.db.lock().map_err(ApiError::internal)?;
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
                bandwidth_mbps: (capabilities.model_bandwidth_kbps as f64 / 1000.0).max(0.001),
                load_fraction: (capabilities.accelerator_load_permille.min(1000) as f64) / 1000.0,
                recent_failures: 0,
                online: true,
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
    Ok(Json(PlanResponse {
        manifest_digest: manifest.digest().map_err(ApiError::internal)?,
        candidates,
        plan,
    }))
}

fn protocol_gpu_to_device(gpu: &mesh_protocol::GpuCapability) -> Option<DeviceCapability> {
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
    })
}

async fn submit_inference(
    State(state): State<AppState>,
    Json(req): Json<SubmitInferenceRequest>,
) -> Result<Json<SubmitInferenceResponse>, ApiError> {
    req.validate()
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let body_hash = submit_inference_body_hash(&req).map_err(ApiError::internal)?;
    let mut conn = state.db.lock().map_err(ApiError::internal)?;
    authenticate_known_node(&conn, &req.auth, "submit_inference", &body_hash)?;
    let (manifest, nodes, seed_peers) =
        ai_context(&conn, &req.auth.node_id, &req.model_id, &req.revision)?;
    let digest = manifest.digest().map_err(ApiError::internal)?;
    let candidates = rank_candidates(&manifest, &req.requirements, &nodes, &Default::default());
    if candidates.is_empty() {
        return Err(ApiError::conflict("no eligible AI devices are online"));
    }
    let plan = plan_parallelism(&candidates, req.layer_count, &req.requirements)
        .map_err(|error| ApiError::conflict(error.to_string()))?;
    let job_id = format!("ai_{}", Uuid::new_v4().simple());
    let transaction = conn.transaction().map_err(ApiError::internal)?;
    transaction.execute(
        "INSERT INTO ai_jobs(job_id,requester_node_id,request_json,plan_json,manifest_digest,status,created_at)
         VALUES(?1,?2,?3,?4,?5,'pending',?6)",
        params![job_id, req.auth.node_id, serde_json::to_string(&req).map_err(ApiError::internal)?,
            serde_json::to_string(&plan).map_err(ApiError::internal)?, digest, now_unix()],
    ).map_err(ApiError::internal)?;
    for (index, batch) in plan.batches.iter().enumerate() {
        let prompts = (batch.batch_start..batch.batch_end)
            .map(|prompt_index| (prompt_index, req.prompts[prompt_index as usize].clone()))
            .collect();
        let assignment = InferenceAssignment {
            assignment_id: format!("aiasg_{}_{}", &job_id[3..19], index),
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
             VALUES(?1,?2,?3,?4,'pending')",
            params![assignment.assignment_id, job_id, batch.node_id, serde_json::to_string(&assignment).map_err(ApiError::internal)?],
        ).map_err(ApiError::internal)?;
    }
    transaction.commit().map_err(ApiError::internal)?;
    Ok(Json(SubmitInferenceResponse {
        job_id,
        manifest_digest: digest,
        assignments: plan.batches.len() as u32,
        plan,
    }))
}

async fn poll_inference(
    State(state): State<AppState>,
    Json(req): Json<PollInferenceRequest>,
) -> Result<Json<PollInferenceResponse>, ApiError> {
    let conn = state.db.lock().map_err(ApiError::internal)?;
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
    let mut conn = state.db.lock().map_err(ApiError::internal)?;
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
    let transaction = conn.transaction().map_err(ApiError::internal)?;
    transaction.execute(
        "UPDATE ai_assignments SET status='completed',outputs_json=?2,lease_until=NULL,completed_at=?3 WHERE assignment_id=?1",
        params![req.assignment_id, serde_json::to_string(&req.outputs).map_err(ApiError::internal)?, now_unix()],
    ).map_err(ApiError::internal)?;
    let remaining: i64 = transaction
        .query_row(
            "SELECT COUNT(*) FROM ai_assignments WHERE job_id=?1 AND status!='completed'",
            params![job_id],
            |row| row.get(0),
        )
        .map_err(ApiError::internal)?;
    if remaining == 0 {
        transaction
            .execute(
                "UPDATE ai_jobs SET status='completed',completed_at=?2 WHERE job_id=?1",
                params![job_id, now_unix()],
            )
            .map_err(ApiError::internal)?;
    }
    transaction.commit().map_err(ApiError::internal)?;
    Ok(Json(ReportInferenceResponse {
        accepted: true,
        job_completed: remaining == 0,
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
    let conn = state.db.lock().map_err(ApiError::internal)?;
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
    let conn = state.db.lock().map_err(ApiError::internal)?;
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
    let mut statement = conn
        .prepare(
            "SELECT outputs_json FROM ai_assignments WHERE job_id=?1 AND outputs_json IS NOT NULL",
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
    Ok(Json(InferenceJobStatus {
        job_id,
        status,
        total_assignments: total as u32,
        completed_assignments: completed as u32,
        outputs,
    }))
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
            bandwidth_mbps: (capabilities.model_bandwidth_kbps as f64 / 1000.0).max(0.001),
            load_fraction: capabilities.accelerator_load_permille.min(1000) as f64 / 1000.0,
            recent_failures: 0,
            online: true,
        });
    }
    Ok((manifest, nodes, seed_peers))
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
    let conn = state.db.lock().map_err(ApiError::internal)?;
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

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod ai_api_tests {
    use super::*;
    use mesh_ai::{
        FailInferenceRequest, InferenceRequirements, PollInferenceRequest, PromptOutput,
        RegisterModelRequest, ReportInferenceRequest, SubmitInferenceRequest,
        fail_inference_body_hash, register_model_body_hash, report_inference_body_hash,
        submit_inference_body_hash,
    };
    use mesh_core::identity::NodeIdentity;
    use mesh_model::{ChunkRef, ModelFormat, sha256};
    use mesh_protocol::{RegisterRequest, register_body_hash};
    use serde::{Serialize, de::DeserializeOwned};
    use std::collections::HashSet;
    use std::fs;

    #[tokio::test]
    async fn distributed_ai_job_reroutes_and_completes() {
        let root = std::env::temp_dir().join(format!("mesh-ai-api-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let db_path = root.join("coordinator.db");
        let state = AppState {
            db: Arc::new(Mutex::new(
                crate::db::open(db_path.to_str().unwrap()).unwrap(),
            )),
            ledger: None,
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
            output_sha256: mesh_protocol::hash_bytes(b"world"),
            duration_ms: 1,
        };
        let mut report = ReportInferenceRequest {
            auth: worker_b.auth("unused", &empty_body_hash()),
            assignment_id: assignment.assignment_id,
            outputs: vec![output.clone()],
        };
        report.auth = worker_b.auth(
            "report_inference",
            &report_inference_body_hash(&report).unwrap(),
        );
        let accepted: ReportInferenceResponse = post(&base, "/v1/ai/work/result", &report).await;
        assert!(accepted.job_completed);
        let status: InferenceJobStatus =
            reqwest::get(format!("{base}/v1/ai/jobs/{}", submitted.job_id))
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
        assert_eq!(status.outputs, vec![output]);

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
            protocol_version: mesh_protocol::PROTOCOL_VERSION,
            hostname: "test".into(),
            os: "test".into(),
            arch: "test".into(),
            cpu_brand: "test".into(),
            logical_cpus: 1,
            total_memory_bytes: 1024,
            cpu_benchmark_score: 1,
            gpus: if ai_ready {
                vec![mesh_protocol::GpuCapability {
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
            accelerator_load_permille: 0,
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

    /// The peer sample is a bootstrap directory, so it must only name nodes a
    /// probe can actually reach: no endpoint, or gone quiet, means no entry.
    #[tokio::test]
    async fn the_peer_sample_lists_only_nodes_that_can_answer_a_probe() {
        let root = std::env::temp_dir().join(format!("mesh-peer-sample-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let db_path = root.join("coordinator.db");
        let state = AppState {
            db: Arc::new(Mutex::new(
                crate::db::open(db_path.to_str().unwrap()).unwrap(),
            )),
            ledger: None,
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
    /// the sample stays bounded however large the mesh grows.
    #[tokio::test]
    async fn the_peer_sample_stays_bounded_as_the_mesh_grows() {
        let root = std::env::temp_dir().join(format!("mesh-peer-cap-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let state = AppState {
            db: Arc::new(Mutex::new(
                crate::db::open(root.join("coordinator.db").to_str().unwrap()).unwrap(),
            )),
            ledger: None,
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
