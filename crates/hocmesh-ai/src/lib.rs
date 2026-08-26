use anyhow::{Result, bail, ensure};
use hocmesh_gpu::{BackendKind, DeviceCapability};
use hocmesh_model::ModelManifest;
use hocmesh_protocol::AuthProof;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterModelRequest {
    pub auth: AuthProof,
    pub manifest: ModelManifest,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegisterModelResponse {
    pub manifest_digest: String,
    pub model_id: String,
    pub revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanRequest {
    pub auth: AuthProof,
    pub model_id: String,
    pub revision: String,
    pub requirements: InferenceRequirements,
    pub layer_count: u32,
    #[serde(default)]
    pub excluded_nodes: BTreeSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanResponse {
    pub manifest_digest: String,
    pub candidates: Vec<CandidateScore>,
    pub plan: ParallelismPlan,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitInferenceRequest {
    pub auth: AuthProof,
    pub model_id: String,
    pub revision: String,
    pub prompts: Vec<String>,
    pub max_tokens: u32,
    pub temperature_milli: u32,
    pub seed: u64,
    pub requirements: InferenceRequirements,
    pub layer_count: u32,
}

impl SubmitInferenceRequest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.model_id.is_empty() && !self.revision.is_empty(),
            "model reference is empty"
        );
        ensure!(
            !self.prompts.is_empty() && self.prompts.len() <= 256,
            "prompt batch must contain 1..=256 prompts"
        );
        ensure!(
            self.prompts
                .iter()
                .all(|prompt| !prompt.is_empty() && prompt.len() <= 1_000_000),
            "invalid prompt length"
        );
        ensure!(
            self.max_tokens > 0 && self.max_tokens <= 32_768,
            "max_tokens is out of range"
        );
        ensure!(
            self.temperature_milli <= 5_000,
            "temperature is out of range"
        );
        ensure!(
            self.requirements.batch_size as usize == self.prompts.len(),
            "batch_size must match prompts"
        );
        ensure!(self.layer_count > 0, "layer_count must be positive");
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SubmitInferenceResponse {
    pub job_id: String,
    pub manifest_digest: String,
    pub plan: ParallelismPlan,
    pub assignments: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollInferenceRequest {
    pub auth: AuthProof,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceAssignment {
    pub assignment_id: String,
    pub job_id: String,
    pub manifest: ModelManifest,
    pub seed_peers: Vec<String>,
    pub prompts: Vec<(u32, String)>,
    pub max_tokens: u32,
    pub temperature_milli: u32,
    pub seed: u64,
    pub device_id: String,
    pub lease_seconds: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PollInferenceResponse {
    pub assignment: Option<InferenceAssignment>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PromptOutput {
    pub prompt_index: u32,
    pub text: String,
    pub output_sha256: String,
    pub duration_ms: u64,
}

impl PromptOutput {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            hocmesh_protocol::hash_bytes(self.text.as_bytes()) == self.output_sha256,
            "output digest mismatch"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportInferenceRequest {
    pub auth: AuthProof,
    pub assignment_id: String,
    pub outputs: Vec<PromptOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReportInferenceResponse {
    pub accepted: bool,
    pub job_completed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailInferenceRequest {
    pub auth: AuthProof,
    pub assignment_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FailInferenceResponse {
    pub rerouted_to: String,
    pub attempt: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceJobStatus {
    pub job_id: String,
    pub status: String,
    pub total_assignments: u32,
    pub completed_assignments: u32,
    pub outputs: Vec<PromptOutput>,
}

pub fn register_model_body_hash(manifest: &ModelManifest) -> Result<String, serde_json::Error> {
    hocmesh_protocol::hash_json(manifest)
}

pub fn plan_body_hash(request: &PlanRequest) -> Result<String, serde_json::Error> {
    hocmesh_protocol::hash_json(&(
        &request.model_id,
        &request.revision,
        &request.requirements,
        request.layer_count,
        &request.excluded_nodes,
    ))
}

pub fn submit_inference_body_hash(
    request: &SubmitInferenceRequest,
) -> Result<String, serde_json::Error> {
    hocmesh_protocol::hash_json(&(
        &request.model_id,
        &request.revision,
        &request.prompts,
        request.max_tokens,
        request.temperature_milli,
        request.seed,
        &request.requirements,
        request.layer_count,
    ))
}

pub fn report_inference_body_hash(
    request: &ReportInferenceRequest,
) -> Result<String, serde_json::Error> {
    hocmesh_protocol::hash_json(&(&request.assignment_id, &request.outputs))
}

pub fn fail_inference_body_hash(
    request: &FailInferenceRequest,
) -> Result<String, serde_json::Error> {
    hocmesh_protocol::hash_json(&(&request.assignment_id, &request.reason))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeProfile {
    pub node_id: String,
    pub devices: Vec<DeviceCapability>,
    pub cached_chunks: BTreeSet<String>,
    pub network_latency_ms: f64,
    pub bandwidth_mbps: f64,
    pub load_fraction: f64,
    pub recent_failures: u32,
    pub online: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceRequirements {
    pub required_backends: BTreeSet<BackendKind>,
    pub minimum_memory_bytes: u64,
    pub needs_fp16: bool,
    pub needs_bf16: bool,
    pub needs_int8: bool,
    pub batch_size: u32,
    pub pipeline_stages: u32,
    pub tensor_parallelism: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidateScore {
    pub node_id: String,
    pub device_id: String,
    pub score: f64,
    pub estimated_transfer_bytes: u64,
}

pub fn rank_candidates(
    manifest: &ModelManifest,
    requirements: &InferenceRequirements,
    nodes: &[NodeProfile],
    excluded_nodes: &BTreeSet<String>,
) -> Vec<CandidateScore> {
    let all_chunks: BTreeMap<_, _> = manifest
        .chunks
        .iter()
        .map(|chunk| (chunk.sha256.as_str(), chunk.size_bytes))
        .collect();
    let mut candidates = Vec::new();
    for node in nodes {
        if !node.online || excluded_nodes.contains(&node.node_id) || !valid_metric(node) {
            continue;
        }
        let missing_bytes = all_chunks
            .iter()
            .filter(|(hash, _)| !node.cached_chunks.contains(**hash))
            .map(|(_, size)| *size)
            .sum();
        for device in &node.devices {
            if !device_matches(device, requirements) {
                continue;
            }
            let transfer_ms = missing_bytes as f64 * 8.0 / (node.bandwidth_mbps * 1_000.0);
            let memory_headroom = device
                .memory_bytes
                .unwrap_or(0)
                .saturating_sub(requirements.minimum_memory_bytes);
            let locality = 1.0 - missing_bytes as f64 / manifest.total_size_bytes.max(1) as f64;
            let score = node.network_latency_ms
                + transfer_ms
                + node.load_fraction * 1_000.0
                + node.recent_failures as f64 * 500.0
                - locality * 100.0
                - memory_headroom as f64 / (1024.0 * 1024.0 * 1024.0);
            candidates.push(CandidateScore {
                node_id: node.node_id.clone(),
                device_id: device.stable_id.clone(),
                score,
                estimated_transfer_bytes: missing_bytes,
            });
        }
    }
    candidates.sort_by(|a, b| {
        a.score
            .total_cmp(&b.score)
            .then_with(|| a.node_id.cmp(&b.node_id))
    });
    candidates
}

fn valid_metric(node: &NodeProfile) -> bool {
    node.network_latency_ms.is_finite()
        && node.network_latency_ms >= 0.0
        && node.bandwidth_mbps.is_finite()
        && node.bandwidth_mbps > 0.0
        && node.load_fraction.is_finite()
        && (0.0..=1.0).contains(&node.load_fraction)
}

fn device_matches(device: &DeviceCapability, requirements: &InferenceRequirements) -> bool {
    (requirements.required_backends.is_empty()
        || requirements.required_backends.contains(&device.backend))
        && device.memory_bytes.unwrap_or(0) >= requirements.minimum_memory_bytes
        && (!requirements.needs_fp16 || device.supports_fp16)
        && (!requirements.needs_bf16 || device.supports_bf16)
        && (!requirements.needs_int8 || device.supports_int8)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ParallelismKind {
    Batch,
    Pipeline,
    ModelTensor,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipelineStage {
    pub stage_index: u32,
    pub node_id: String,
    pub device_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorGroup {
    pub rank: u32,
    pub node_id: String,
    pub device_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BatchShard {
    pub batch_start: u32,
    pub batch_end: u32,
    pub node_id: String,
    pub device_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParallelismPlan {
    pub kinds: BTreeSet<ParallelismKind>,
    pub pipeline: Vec<PipelineStage>,
    pub tensor_group: Vec<TensorGroup>,
    pub batches: Vec<BatchShard>,
}

pub fn plan_parallelism(
    candidates: &[CandidateScore],
    layer_count: u32,
    requirements: &InferenceRequirements,
) -> Result<ParallelismPlan> {
    ensure!(layer_count > 0, "model must contain layers");
    ensure!(requirements.batch_size > 0, "batch size must be positive");
    let pipeline_count = requirements.pipeline_stages.max(1) as usize;
    let tensor_count = requirements.tensor_parallelism.max(1) as usize;
    let required = pipeline_count.max(tensor_count);
    ensure!(
        candidates.len() >= required,
        "insufficient eligible devices"
    );
    let mut kinds = BTreeSet::new();
    let mut pipeline = Vec::new();
    if pipeline_count > 1 {
        kinds.insert(ParallelismKind::Pipeline);
        for (index, candidate) in candidates.iter().take(pipeline_count).enumerate() {
            let start = layer_count * index as u32 / pipeline_count as u32;
            let end = layer_count * (index as u32 + 1) / pipeline_count as u32;
            pipeline.push(PipelineStage {
                stage_index: index as u32,
                node_id: candidate.node_id.clone(),
                device_id: candidate.device_id.clone(),
                layer_start: start,
                layer_end: end,
            });
        }
    }
    let mut tensor_group = Vec::new();
    if tensor_count > 1 {
        kinds.insert(ParallelismKind::ModelTensor);
        for (rank, candidate) in candidates.iter().take(tensor_count).enumerate() {
            tensor_group.push(TensorGroup {
                rank: rank as u32,
                node_id: candidate.node_id.clone(),
                device_id: candidate.device_id.clone(),
            });
        }
    }
    let mut batches = Vec::new();
    kinds.insert(ParallelismKind::Batch);
    let workers = candidates.len().min(requirements.batch_size as usize);
    for (index, candidate) in candidates.iter().take(workers).enumerate() {
        let start = requirements.batch_size * index as u32 / workers as u32;
        let end = requirements.batch_size * (index as u32 + 1) / workers as u32;
        batches.push(BatchShard {
            batch_start: start,
            batch_end: end,
            node_id: candidate.node_id.clone(),
            device_id: candidate.device_id.clone(),
        });
    }
    Ok(ParallelismPlan {
        kinds,
        pipeline,
        tensor_group,
        batches,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FailureRecord {
    pub node_id: String,
    pub attempt: u32,
    pub reason: String,
}

pub fn reroute(
    failed: &FailureRecord,
    prior_assignments: &BTreeSet<String>,
    candidates: &[CandidateScore],
) -> Result<CandidateScore> {
    candidates
        .iter()
        .find(|candidate| {
            candidate.node_id != failed.node_id && !prior_assignments.contains(&candidate.node_id)
        })
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no unused failover candidate available"))
}

pub fn validate_plan(plan: &ParallelismPlan, layer_count: u32, batch_size: u32) -> Result<()> {
    if !plan.pipeline.is_empty() {
        ensure!(
            plan.pipeline.first().unwrap().layer_start == 0,
            "pipeline does not start at layer zero"
        );
        ensure!(
            plan.pipeline.last().unwrap().layer_end == layer_count,
            "pipeline does not cover final layer"
        );
        for pair in plan.pipeline.windows(2) {
            ensure!(
                pair[0].layer_end == pair[1].layer_start,
                "pipeline has a gap or overlap"
            );
        }
    }
    if !plan.tensor_group.is_empty() {
        for (rank, member) in plan.tensor_group.iter().enumerate() {
            ensure!(
                member.rank as usize == rank,
                "tensor ranks are not contiguous"
            );
        }
    }
    if !plan.batches.is_empty() {
        ensure!(
            plan.batches.first().unwrap().batch_start == 0,
            "batch plan does not start at zero"
        );
        ensure!(
            plan.batches.last().unwrap().batch_end == batch_size,
            "batch plan is incomplete"
        );
        for pair in plan.batches.windows(2) {
            ensure!(
                pair[0].batch_end == pair[1].batch_start,
                "batch plan has a gap or overlap"
            );
        }
    }
    if plan.pipeline.is_empty() && plan.tensor_group.is_empty() && plan.batches.is_empty() {
        bail!("parallelism plan has no assignments");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hocmesh_model::{ChunkRef, ModelFormat, sha256};

    fn manifest() -> ModelManifest {
        ModelManifest {
            schema_version: 1,
            model_id: "m".into(),
            revision: "1".into(),
            format: ModelFormat::Gguf,
            architecture: "llama".into(),
            parameter_count: None,
            tensor_dtype: Some("f16".into()),
            total_size_bytes: 100,
            chunks: vec![ChunkRef {
                index: 0,
                sha256: sha256(b"x"),
                size_bytes: 100,
            }],
            metadata: Default::default(),
        }
    }
    fn node(id: &str, latency: f64, cached: bool) -> NodeProfile {
        NodeProfile {
            node_id: id.into(),
            devices: vec![DeviceCapability {
                stable_id: format!("gpu-{id}"),
                backend: BackendKind::Cuda,
                vendor: "nvidia".into(),
                name: "gpu".into(),
                memory_bytes: Some(16_000),
                driver_version: None,
                compute_version: None,
                supports_fp16: true,
                supports_bf16: true,
                supports_int8: true,
            }],
            cached_chunks: if cached {
                [sha256(b"x")].into_iter().collect()
            } else {
                BTreeSet::new()
            },
            network_latency_ms: latency,
            bandwidth_mbps: 1000.0,
            load_fraction: 0.0,
            recent_failures: 0,
            online: true,
        }
    }
    fn requirements() -> InferenceRequirements {
        InferenceRequirements {
            required_backends: [BackendKind::Cuda].into_iter().collect(),
            minimum_memory_bytes: 1,
            needs_fp16: true,
            needs_bf16: false,
            needs_int8: false,
            batch_size: 4,
            pipeline_stages: 2,
            tensor_parallelism: 2,
        }
    }

    fn auth() -> AuthProof {
        AuthProof {
            node_id: "node".into(),
            timestamp: 0,
            nonce_b64: "nonce".into(),
            signature_b64: "signature".into(),
        }
    }

    #[test]
    fn scheduler_accounts_for_latency_and_cache_locality() {
        let ranked = rank_candidates(
            &manifest(),
            &requirements(),
            &[node("remote", 1.0, false), node("cached", 20.0, true)],
            &BTreeSet::new(),
        );
        assert_eq!(ranked[0].node_id, "cached");
    }

    #[test]
    fn planner_covers_layers_tensor_ranks_and_batches() {
        let ranked = rank_candidates(
            &manifest(),
            &requirements(),
            &[node("a", 1.0, true), node("b", 2.0, true)],
            &BTreeSet::new(),
        );
        let plan = plan_parallelism(&ranked, 17, &requirements()).unwrap();
        validate_plan(&plan, 17, 4).unwrap();
        assert_eq!(plan.tensor_group.len(), 2);
        assert_eq!(plan.pipeline.len(), 2);
        assert_eq!(plan.batches.len(), 2);
    }

    #[test]
    fn reroute_never_reuses_failed_or_prior_node() {
        let candidates = vec![
            CandidateScore {
                node_id: "a".into(),
                device_id: "1".into(),
                score: 1.0,
                estimated_transfer_bytes: 0,
            },
            CandidateScore {
                node_id: "b".into(),
                device_id: "2".into(),
                score: 2.0,
                estimated_transfer_bytes: 0,
            },
            CandidateScore {
                node_id: "c".into(),
                device_id: "3".into(),
                score: 3.0,
                estimated_transfer_bytes: 0,
            },
        ];
        let next = reroute(
            &FailureRecord {
                node_id: "a".into(),
                attempt: 1,
                reason: "lost".into(),
            },
            &["b".into()].into_iter().collect(),
            &candidates,
        )
        .unwrap();
        assert_eq!(next.node_id, "c");
    }

    #[test]
    fn inference_request_enforces_every_public_bound() {
        let valid = SubmitInferenceRequest {
            auth: auth(),
            model_id: "m".into(),
            revision: "1".into(),
            prompts: vec!["hello".into()],
            max_tokens: 1,
            temperature_milli: 5_000,
            seed: 7,
            requirements: InferenceRequirements {
                batch_size: 1,
                pipeline_stages: 1,
                tensor_parallelism: 1,
                ..requirements()
            },
            layer_count: 1,
        };
        valid.validate().unwrap();
        let mut invalid = valid.clone();
        invalid.prompts.clear();
        assert!(invalid.validate().is_err());
        let mut invalid = valid.clone();
        invalid.max_tokens = 32_769;
        assert!(invalid.validate().is_err());
        let mut invalid = valid.clone();
        invalid.temperature_milli = 5_001;
        assert!(invalid.validate().is_err());
        let mut invalid = valid.clone();
        invalid.requirements.batch_size = 2;
        assert!(invalid.validate().is_err());
        let mut invalid = valid;
        invalid.layer_count = 0;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn scheduler_rejects_unhealthy_incompatible_and_excluded_devices() {
        let mut offline = node("offline", 1.0, true);
        offline.online = false;
        let mut invalid_metric = node("nan", 1.0, true);
        invalid_metric.load_fraction = f64::NAN;
        let mut too_small = node("small", 1.0, true);
        too_small.devices[0].memory_bytes = Some(0);
        let eligible = node("eligible", 2.0, true);
        let ranked = rank_candidates(
            &manifest(),
            &requirements(),
            &[offline, invalid_metric, too_small, eligible],
            &["eligible".to_string()].into_iter().collect(),
        );
        assert!(ranked.is_empty());
    }

    #[test]
    fn invalid_and_exhausted_plans_are_rejected() {
        assert!(plan_parallelism(&[], 1, &requirements()).is_err());
        let empty = ParallelismPlan {
            kinds: BTreeSet::new(),
            pipeline: vec![],
            tensor_group: vec![],
            batches: vec![],
        };
        assert!(validate_plan(&empty, 1, 1).is_err());
        let candidates = vec![CandidateScore {
            node_id: "a".into(),
            device_id: "gpu-a".into(),
            score: 0.0,
            estimated_transfer_bytes: 0,
        }];
        assert!(
            reroute(
                &FailureRecord {
                    node_id: "a".into(),
                    attempt: 2,
                    reason: "lost".into(),
                },
                &BTreeSet::new(),
                &candidates,
            )
            .is_err()
        );
    }
}
