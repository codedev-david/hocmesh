use anyhow::{Result, bail, ensure};
use hocmesh_core::proximity::{self, UNKNOWN_EDGE_MICROS};
use hocmesh_gpu::{BackendKind, DeviceCapability};
use hocmesh_model::ModelManifest;
use hocmesh_protocol::{AuthProof, InferenceBilling, NetworkCoordinate};
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
    pub billing: InferenceBilling,
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
    pub job_id: String,
    pub batch_start: u32,
    pub batch_end: u32,
    pub reward_mcu: i64,
    pub outputs: Vec<PromptOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReportInferenceResponse {
    pub accepted: bool,
    pub reward_mcu: i64,
    pub balance_mcu: i64,
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
    pub refundable: Vec<RefundableBatch>,
    pub delivered: Vec<DeliveredBatchSummary>,
}

/// A batch nobody delivered, and what its escrow is worth back.
///
/// The requester cannot price a batch it never saw assigned, so the
/// coordinator lists what is reclaimable - but it lists the amount the
/// assignment already committed to, and the ledger recomputes that amount
/// from the certified billing before it moves anything.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RefundableBatch {
    pub assignment_id: String,
    pub batch_start: u32,
    pub batch_end: u32,
    pub refund_mcu: i64,
}

/// A batch a provider has answered, seen from the requester's side.
///
/// The digest is here and the text is not. That is the whole point of the
/// two-stage settlement: a requester can see that an answer exists, and what
/// it will cost, before deciding to take delivery of it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeliveredBatchSummary {
    pub assignment_id: String,
    pub batch_start: u32,
    pub batch_end: u32,
    pub price_mcu: i64,
    pub outputs_digest: String,
    pub receipted: bool,
    pub settled: Option<String>,
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

/// What a requester signs to ask for inference.
///
/// The prompts are not in it. Their digest and their sizes are, carried in
/// the billing, and those are what decide the price - so the ledger can hold
/// a signature it can still check without ever holding the prompt text.
pub fn submit_inference_body_hash(
    request: &SubmitInferenceRequest,
) -> Result<String, serde_json::Error> {
    let billing = hocmesh_protocol::inference_billing_hash(&request.billing)?;
    let settings = inference_settings_digest(request)?;
    hocmesh_protocol::inference_submit_body_hash(&billing, &settings)
}

/// Everything about an inference request except what it costs.
///
/// Kept separate from the billing so a validator can check the signature while
/// holding only a digest of these settings - it has no business knowing which
/// model or seed somebody chose, but it does have to know the bill was signed.
pub fn inference_settings_digest(
    request: &SubmitInferenceRequest,
) -> Result<String, serde_json::Error> {
    hocmesh_protocol::hash_json(&(
        &request.model_id,
        &request.revision,
        request.temperature_milli,
        request.seed,
        &request.requirements,
        request.layer_count,
    ))
}

/// The sizes a bill is computed from, one entry per prompt.
pub fn prompt_bytes(prompts: &[String]) -> Vec<u64> {
    prompts.iter().map(|p| p.len() as u64).collect()
}

/// A digest that binds a bill to the exact prompts it was written for,
/// without putting any of them on the ledger.
pub fn prompts_digest(prompts: &[String]) -> Result<String, serde_json::Error> {
    hocmesh_protocol::hash_json(&prompts)
}

/// Write the bill a requester signs for a set of prompts.
///
/// The requester computes its own price rather than being told one: the whole
/// point of a closed-form cost is that nobody has to take the coordinator at
/// its word. `max_cost_mcu` is the ceiling the requester consents to, so the
/// price is agreed in the same round trip that asks for the work.
pub fn bill_for_prompts(
    manifest_digest: &str,
    parameter_count: u64,
    total_size_bytes: u64,
    prompts: &[String],
    max_tokens: u32,
) -> Result<InferenceBilling, serde_json::Error> {
    let bytes = prompt_bytes(prompts);
    let cost = hocmesh_core::compute::inference_cost_mcu(&bytes, max_tokens, parameter_count);
    Ok(InferenceBilling {
        manifest_digest: manifest_digest.to_string(),
        parameter_count,
        total_size_bytes,
        prompts_digest: prompts_digest(prompts)?,
        prompt_bytes: bytes,
        max_tokens,
        max_cost_mcu: cost,
    })
}

/// What a provider should claim for the batch it was handed.
///
/// Derived from the assignment itself rather than from anything the
/// coordinator says the batch is worth. The provider signs this number, so it
/// had better be one the provider worked out - and because the price is closed
/// form, the number it arrives at is the same one the ledger will check.
pub fn assignment_claim(assignment: &InferenceAssignment) -> Option<(u32, u32, i64)> {
    let first = assignment.prompts.first()?.0;
    let last = assignment.prompts.last()?.0;
    let parameter_count = assignment.manifest.parameter_count?;
    let bytes: Vec<u64> = assignment
        .prompts
        .iter()
        .map(|(_, prompt)| prompt.len() as u64)
        .collect();
    let reward =
        hocmesh_core::compute::inference_cost_mcu(&bytes, assignment.max_tokens, parameter_count);
    Some((first, last + 1, reward))
}

/// What a requester sends to take delivery of a batch it paid for.
///
/// Signing this is what moves the CU out of the job escrow and into a holding
/// account it can never come back from, so the requester cannot read the
/// answer and then quietly reclaim the money.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptInferenceRequest {
    pub auth: AuthProof,
    pub assignment_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptInferenceResponse {
    pub assignment_id: String,
    pub batch_start: u32,
    pub batch_end: u32,
    pub price_mcu: i64,
    pub outputs_digest: String,
    /// Handed over only now, in exchange for the receipt.
    pub outputs: Vec<PromptOutput>,
}

/// What a requester sends once it has looked at what it was given.
///
/// Accepting pays the provider. Disputing does not pay the requester back: the
/// CU goes to the commons instead, so refusing good work costs exactly what
/// accepting it would have, and returning junk earns nothing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettleInferenceRequest {
    pub auth: AuthProof,
    pub assignment_id: String,
    pub accepted: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettleInferenceResponse {
    pub assignment_id: String,
    pub accepted: bool,
    pub paid_mcu: i64,
    pub job_completed: bool,
}

/// What a provider signs to claim one batch.
///
/// The coordinator only relays this. The provider signs the amount, the
/// batch, and a digest of what it produced, so a coordinator cannot inflate a
/// reward or move one onto a different batch after the fact.
pub fn report_inference_body_hash(
    request: &ReportInferenceRequest,
) -> Result<String, serde_json::Error> {
    hocmesh_protocol::inference_reward_body_hash(
        &request.assignment_id,
        &request.job_id,
        request.batch_start,
        request.batch_end,
        request.reward_mcu,
        &hocmesh_protocol::hash_json(&request.outputs)?,
    )
}

/// A requester reclaiming the escrow on a batch nobody delivered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefundInferenceRequest {
    pub auth: AuthProof,
    pub job_id: String,
    pub assignment_id: String,
    pub batch_start: u32,
    pub batch_end: u32,
    pub refund_mcu: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RefundInferenceResponse {
    pub refunded_mcu: i64,
    pub balance_mcu: i64,
}

/// What a requester signs to reclaim one batch.
pub fn refund_inference_body_hash(
    request: &RefundInferenceRequest,
) -> Result<String, serde_json::Error> {
    hocmesh_protocol::inference_refund_body_hash(
        &request.assignment_id,
        &request.job_id,
        request.batch_start,
        request.batch_end,
        request.refund_mcu,
    )
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
    /// Main-memory bandwidth measured on this node, used for any device of its
    /// that has no measurement of its own -- in practice a CPU-only machine,
    /// which is the case where this matters most and where nothing else in the
    /// profile answers the question.
    #[serde(default)]
    pub memory_bandwidth_bytes_per_second: Option<u64>,
    /// Where this node sits in latency space, once it has fitted a position.
    ///
    /// `None` means unknown, which is not the same as nearby: a node that has
    /// measured nothing is assumed far from everything, so that not measuring
    /// is never the cheap way into a tight pipeline.
    #[serde(default)]
    pub coordinate: Option<NetworkCoordinate>,
    /// Whether this node has shown an uplink fast enough to hold the first
    /// stage of a pipeline. See `hocmesh_core::roles`.
    #[serde(default)]
    pub prefill_eligible: bool,
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
    /// How fast this particular device streams memory, if anything measured it.
    ///
    /// The device's own figure when it has one, the node's otherwise. `None`
    /// when neither exists, and it stays `None` rather than becoming a default:
    /// `plan_parallelism` needs to be able to tell "slow" from "unmeasured",
    /// because a default would let one unbenchmarked machine quietly decide the
    /// shape of the whole pipeline.
    #[serde(default)]
    pub memory_bandwidth_bytes_per_second: Option<u64>,
    /// This node's position in latency space, copied from its profile so the
    /// planner can measure the hops between stages without a second lookup.
    #[serde(default)]
    pub coordinate: Option<NetworkCoordinate>,
    /// Whether this node may hold stage zero. Copied from its profile for the
    /// same reason.
    #[serde(default)]
    pub prefill_eligible: bool,
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
                // The device's own measurement wins over the node's, because a
                // machine can hold a fast accelerator behind slow main memory
                // and the stage runs on the accelerator. Falling back to the
                // node covers the CPU-only case, where there is no device
                // benchmark and never will be.
                memory_bandwidth_bytes_per_second: device
                    .memory_bandwidth_bytes_per_second
                    .or(node.memory_bandwidth_bytes_per_second),
                coordinate: node.coordinate,
                prefill_eligible: node.prefill_eligible,
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

/// Cut `layer_count` layers into one contiguous span per stage, in order.
fn uniform_spans(stages: usize, layer_count: u32) -> Vec<(u32, u32)> {
    let (n, total) = (stages as u64, u64::from(layer_count));
    (0..n)
        .map(|i| ((total * i / n) as u32, (total * (i + 1) / n) as u32))
        .collect()
}

/// How the model's layers are divided between the stages of a pipeline.
///
/// In pipeline parallelism a token passes through every stage in turn, so the
/// time to produce one is the sum of the stage times and a stage that holds too
/// much for the memory behind it holds up everything downstream. Generating a
/// token re-reads every weight in the stage, which makes the stage time
/// `bytes / bandwidth` -- so the division that finishes soonest is the one where
/// every stage takes the same time, and that means **layers in proportion to
/// bandwidth**, not layers in equal counts.
///
/// An even split is the right answer only when every stage is equally fast.
/// Applied to a mixed set it paces the whole pipeline at its slowest machine
/// while the fast ones idle -- and it is exactly a mixed set that a network of
/// donated hardware produces.
///
/// Two deliberate refusals:
///
/// - **If any stage's bandwidth is unmeasured, every stage is split evenly.**
///   Substituting a default for the one unknown machine would not be a smaller
///   error than an even split, it would be an unpredictable one: the default
///   decides that stage's share of the model, and nothing downstream could tell
///   the guess from a measurement.
/// - **Every stage keeps at least one layer.** A stage with none is a network
///   hop that computes nothing. That floor is a repair applied after the fact,
///   not a reservation taken before: handing every stage a layer up front and
///   sharing out only what is left would pull every split back towards even,
///   including the ones that needed no floor at all.
fn layer_spans(stages: &[CandidateScore], layer_count: u32) -> Vec<(u32, u32)> {
    let n = stages.len();
    // Not enough layers to give each stage one: proportion is meaningless here
    // and the caller has already been told how many devices it needs.
    if n == 0 || (layer_count as usize) <= n {
        return uniform_spans(n, layer_count);
    }
    let Some(weights) = stages
        .iter()
        .map(|s| {
            s.memory_bandwidth_bytes_per_second
                .map(|b| b as f64)
                .filter(|b| b.is_finite() && *b > 0.0)
        })
        .collect::<Option<Vec<_>>>()
    else {
        return uniform_spans(n, layer_count);
    };
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return uniform_spans(n, layer_count);
    }

    // Largest remainder. Each stage's exact share is a fraction of a layer, and
    // the layers left over by rounding all of them down go to the stages that
    // were rounded down hardest. Ties break on stage index so two coordinators
    // planning the same job produce the same plan.
    let total_layers = layer_count as usize;
    let ideal: Vec<f64> = weights
        .iter()
        .map(|w| total_layers as f64 * w / total)
        .collect();
    let mut counts: Vec<usize> = ideal.iter().map(|x| x.floor() as usize).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        (ideal[b] - ideal[b].floor())
            .total_cmp(&(ideal[a] - ideal[a].floor()))
            .then(a.cmp(&b))
    });
    // Flooring `n` fractions can lose at most `n - 1` layers, so one pass over
    // the stages in remainder order always places every last one of them.
    let left = total_layers.saturating_sub(counts.iter().sum::<usize>());
    for &i in order.iter().take(left) {
        counts[i] += 1;
    }

    // Lift any stage that rounded down to nothing, taking the layer from the
    // largest stage -- the one whose share changes least by losing it. There
    // are more layers than stages, so a stage holding none means another holds
    // at least two, and this cannot empty the stage it takes from.
    for i in 0..n {
        if counts[i] > 0 {
            continue;
        }
        let Some(donor) = (0..n).max_by_key(|&j| counts[j]).filter(|&j| counts[j] > 1) else {
            break;
        };
        counts[donor] -= 1;
        counts[i] += 1;
    }

    let mut spans = Vec::with_capacity(n);
    let mut cursor = 0u32;
    for count in counts {
        let end = cursor + count as u32;
        spans.push((cursor, end));
        cursor = end;
    }
    spans
}

/// Round-trip microseconds between two candidates, or the unmeasured cost.
///
/// Both ends have to have fitted a plausible position for the prediction to
/// mean anything. When either has not, the pair is charged
/// [`UNKNOWN_EDGE_MICROS`], which is worse than any real link -- so a node that
/// has never measured its position cannot win a place in a pipeline by being
/// unmeasured, only by being measured and close.
fn edge_micros(a: &CandidateScore, b: &CandidateScore) -> u64 {
    if a.node_id == b.node_id {
        return 0;
    }
    match (a.coordinate.as_ref(), b.coordinate.as_ref()) {
        (Some(x), Some(y)) if proximity::is_plausible(x) && proximity::is_plausible(y) => {
            proximity::predicted_rtt_micros(x, y)
        }
        _ => UNKNOWN_EDGE_MICROS,
    }
}

/// What one chain of stages costs, lower being better.
///
/// Every token crosses every hop, so the sum of the adjacent edges is what a
/// request actually pays -- not the diameter of the set, and not the average.
/// The worst single hop breaks ties, because two chains costing the same in
/// total are not equally good if one of them has a stall in the middle. The
/// head's rank breaks what is left, so the answer does not depend on iteration
/// order.
fn chain_cost(chain: &[&CandidateScore], head_rank: usize) -> (u64, u64, usize) {
    let hops: Vec<u64> = chain
        .windows(2)
        .map(|pair| edge_micros(pair[0], pair[1]))
        .collect();
    (
        hops.iter().sum(),
        hops.iter().copied().max().unwrap_or(0),
        head_rank,
    )
}

/// Choose and order the machines that will hold the stages.
///
/// Ranking scores each machine on its own merits -- how much of the model it
/// already holds, how loaded it is, how far it is from the requester. That is
/// the right question for a batch, where the machines never speak to each
/// other. It is the wrong question for a pipeline: the five best machines
/// individually can be the five worst as a chain, because a token has to cross
/// every hop between them and a hop across an ocean costs more than anything
/// the ranking was measuring.
///
/// So the head is chosen from the machines allowed to hold it, and the rest of
/// the chain is grown from that head by nearest neighbour -- every eligible
/// head tried, the cheapest whole chain kept. That is a heuristic, not an
/// optimum; the exact answer is a shortest Hamiltonian path, and paying for it
/// would cost more than the hops it saved.
///
/// Returns an error rather than a slow pipeline when no machine may hold the
/// head. A pipeline whose first stage cannot push its activations out is a
/// pipeline that will disappoint quietly, in a way that looks like the model
/// being slow rather than the placement being wrong.
fn order_pipeline(candidates: &[CandidateScore], count: usize) -> Result<Vec<CandidateScore>> {
    ensure!(
        candidates.len() >= count,
        "insufficient eligible devices for {count} pipeline stages"
    );
    // One device per machine: two stages on one box is not a pipeline, it is
    // the same memory bus twice, and it would let a single failure take out
    // two stages of a chain that exists to spread them.
    let mut first_per_node: Vec<usize> = Vec::new();
    let mut seen = BTreeSet::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if seen.insert(candidate.node_id.clone()) {
            first_per_node.push(index);
        }
    }
    ensure!(
        first_per_node.len() >= count,
        "a pipeline of {count} stages needs {count} distinct machines, and only          {} offered a device",
        first_per_node.len()
    );
    let heads: Vec<usize> = first_per_node
        .iter()
        .copied()
        .filter(|&i| candidates[i].prefill_eligible)
        .collect();
    ensure!(
        !heads.is_empty(),
        "no eligible machine has shown an uplink of at least {} kbit/s, which is          what holding the first stage of a pipeline asks for; the later stages          have no such requirement, so a plan without pipeline parallelism can          still be served",
        hocmesh_core::roles::PREFILL_UPLINK_KBPS
    );

    let mut best: Option<((u64, u64, usize), Vec<usize>)> = None;
    for &head in &heads {
        let mut chain = vec![head];
        let mut rest: Vec<usize> = first_per_node
            .iter()
            .copied()
            .filter(|&i| i != head)
            .collect();
        while chain.len() < count {
            let tail = *chain.last().expect("chain starts with the head");
            let Some(position) = (0..rest.len()).min_by_key(|&k| {
                let next = rest[k];
                (edge_micros(&candidates[tail], &candidates[next]), next)
            }) else {
                break;
            };
            chain.push(rest.remove(position));
        }
        if chain.len() < count {
            continue;
        }
        let borrowed: Vec<&CandidateScore> = chain.iter().map(|&i| &candidates[i]).collect();
        let cost = chain_cost(&borrowed, head);
        if best.as_ref().is_none_or(|(seen, _)| cost < *seen) {
            best = Some((cost, chain));
        }
    }
    let (_, chain) = best.expect("at least one eligible head produced a full chain");
    Ok(chain.into_iter().map(|i| candidates[i].clone()).collect())
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
        let ordered = order_pipeline(candidates, pipeline_count)?;
        let stages = &ordered[..];
        let spans = layer_spans(stages, layer_count);
        for (index, (candidate, (start, end))) in stages.iter().zip(spans).enumerate() {
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
                memory_bandwidth_bytes_per_second: None,
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
            memory_bandwidth_bytes_per_second: None,
            coordinate: None,
            prefill_eligible: true,
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

    fn billing() -> InferenceBilling {
        bill_for_prompts("digest", 1_000, 500, &["hello".into()], 1).unwrap()
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
                memory_bandwidth_bytes_per_second: None,
                coordinate: None,
                prefill_eligible: true,
            },
            CandidateScore {
                node_id: "b".into(),
                device_id: "2".into(),
                score: 2.0,
                estimated_transfer_bytes: 0,
                memory_bandwidth_bytes_per_second: None,
                coordinate: None,
                prefill_eligible: true,
            },
            CandidateScore {
                node_id: "c".into(),
                device_id: "3".into(),
                score: 3.0,
                estimated_transfer_bytes: 0,
                memory_bandwidth_bytes_per_second: None,
                coordinate: None,
                prefill_eligible: true,
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
            billing: billing(),
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
            memory_bandwidth_bytes_per_second: None,
            coordinate: None,
            prefill_eligible: true,
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

    // -- Splitting a model between unequal machines ------------------------

    fn stage(id: &str, bandwidth: Option<u64>) -> CandidateScore {
        CandidateScore {
            node_id: id.into(),
            device_id: format!("dev-{id}"),
            score: 0.0,
            estimated_transfer_bytes: 0,
            memory_bandwidth_bytes_per_second: bandwidth,
            coordinate: None,
            prefill_eligible: true,
        }
    }

    fn widths(spans: &[(u32, u32)]) -> Vec<u32> {
        spans.iter().map(|(a, b)| b - a).collect()
    }

    /// The property that matters: a stage twice as fast holds about twice the
    /// model, so every stage takes about the same time and none of them waits.
    #[test]
    fn layers_follow_bandwidth_rather_than_headcount() {
        let stages = [
            stage("fast", Some(40_000_000_000)),
            stage("slow", Some(10_000_000_000)),
        ];
        assert_eq!(widths(&layer_spans(&stages, 40)), vec![32, 8]);
    }

    /// Balanced is what an even split was always trying to be; on equal
    /// machines the two agree, so nothing changes for a uniform cluster.
    #[test]
    fn equal_machines_still_get_equal_shares() {
        let stages = [
            stage("a", Some(20_000_000_000)),
            stage("b", Some(20_000_000_000)),
            stage("c", Some(20_000_000_000)),
        ];
        assert_eq!(widths(&layer_spans(&stages, 33)), vec![11, 11, 11]);
    }

    /// One unmeasured machine sends the whole split back to even. A default
    /// would be a number nothing downstream could tell from a measurement.
    #[test]
    fn one_unmeasured_stage_makes_the_whole_split_even() {
        let stages = [stage("fast", Some(40_000_000_000)), stage("unknown", None)];
        assert_eq!(
            layer_spans(&stages, 40),
            uniform_spans(2, 40),
            "an unmeasured stage must not be scored as if it were measured"
        );
    }

    /// A stage holding nothing is a network hop that computes nothing, so even
    /// the slowest machine in a lopsided set keeps a layer.
    #[test]
    fn every_stage_keeps_at_least_one_layer() {
        let stages = [
            stage("fast", Some(1_000_000_000_000)),
            stage("crawling", Some(1)),
        ];
        let spans = layer_spans(&stages, 8);
        assert_eq!(widths(&spans), vec![7, 1]);
    }

    /// Whatever the weights, the spans have to be a partition of the model:
    /// contiguous, in order, covering every layer exactly once. This is the
    /// invariant `validate_plan` enforces downstream.
    #[test]
    fn the_split_always_covers_the_model_exactly_once() {
        let cases: Vec<Vec<Option<u64>>> = vec![
            vec![Some(7), Some(11), Some(13)],
            vec![
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
            ],
            vec![Some(u64::MAX), Some(1)],
            vec![Some(3), None, Some(5)],
            vec![Some(0), Some(4)],
        ];
        for widths_in in cases {
            let stages: Vec<_> = widths_in
                .iter()
                .enumerate()
                .map(|(i, b)| stage(&i.to_string(), *b))
                .collect();
            for layer_count in [1u32, 2, 7, 8, 32, 33, 80] {
                let spans = layer_spans(&stages, layer_count);
                assert_eq!(spans.len(), stages.len());
                assert_eq!(spans[0].0, 0, "{widths_in:?} over {layer_count}");
                assert_eq!(
                    spans.last().unwrap().1,
                    layer_count,
                    "{widths_in:?} over {layer_count}"
                );
                for pair in spans.windows(2) {
                    assert_eq!(pair[0].1, pair[1].0, "{widths_in:?} over {layer_count}");
                }
            }
        }
    }

    /// The plan a coordinator hands out has to be the plan any other
    /// coordinator would have produced from the same inputs.
    #[test]
    fn the_same_machines_always_produce_the_same_split() {
        let stages = [
            stage("a", Some(3_000_000_000)),
            stage("b", Some(3_000_000_000)),
            stage("c", Some(3_000_000_001)),
        ];
        let once = layer_spans(&stages, 41);
        for _ in 0..8 {
            assert_eq!(layer_spans(&stages, 41), once);
        }
    }

    /// End to end: the planner's output is a valid plan and the fast machine
    /// really did get the larger share of the model.
    #[test]
    fn a_planned_pipeline_is_weighted_and_still_valid() {
        let candidates = [
            stage("fast", Some(48_000_000_000)),
            stage("slow", Some(16_000_000_000)),
        ];
        let mut requirements = requirements();
        requirements.pipeline_stages = 2;
        requirements.batch_size = 1;
        let plan = plan_parallelism(&candidates, 32, &requirements).unwrap();
        validate_plan(&plan, 32, 1).unwrap();
        let held: Vec<u32> = plan
            .pipeline
            .iter()
            .map(|s| s.layer_end - s.layer_start)
            .collect();
        assert_eq!(held, vec![24, 8]);
    }

    // -- Choosing the chain, not just the machines -------------------------

    /// A coordinate at `(x, 0, 0)` with no access-link cost and full
    /// confidence. Distance along one axis is then just the difference in `x`,
    /// which keeps these tests readable as a line of machines.
    fn at(x: i64) -> Option<NetworkCoordinate> {
        Some(NetworkCoordinate {
            vector_micros: [x, 0, 0],
            height_micros: 0,
            error_permille: 0,
        })
    }

    fn placed(id: &str, x: Option<i64>, prefill: bool) -> CandidateScore {
        CandidateScore {
            node_id: id.into(),
            device_id: format!("dev-{id}"),
            score: 0.0,
            estimated_transfer_bytes: 0,
            memory_bandwidth_bytes_per_second: None,
            coordinate: x.and_then(at),
            prefill_eligible: prefill,
        }
    }

    /// The whole point of ordering separately from ranking. Ranking put the
    /// two machines on the far side of the world second and third because each
    /// is individually excellent; as a chain they cost two ocean crossings,
    /// and the neighbours ranked below them cost almost nothing.
    #[test]
    fn a_chain_of_near_machines_beats_a_chain_of_better_distant_ones() {
        let candidates = vec![
            placed("head", Some(0), true),
            placed("far-a", Some(150_000), true),
            placed("far-b", Some(150_100), true),
            placed("near-a", Some(500), true),
            placed("near-b", Some(900), true),
        ];
        let chain = order_pipeline(&candidates, 3).unwrap();
        let ids: Vec<&str> = chain.iter().map(|c| c.node_id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["head", "near-a", "near-b"],
            "the chain was picked by individual rank, so every token now crosses \
             an ocean twice to reach machines that were only better in isolation"
        );
    }

    /// Refusing loudly rather than planning a pipeline that will be slow for a
    /// reason nobody can see. The message has to name the threshold, because
    /// "no eligible machine" without a number is not something an operator can
    /// act on.
    #[test]
    fn a_pipeline_is_refused_when_no_machine_may_hold_the_head() {
        let candidates = vec![
            placed("a", Some(0), false),
            placed("b", Some(100), false),
            placed("c", Some(200), false),
        ];
        let error = order_pipeline(&candidates, 3).unwrap_err().to_string();
        assert!(
            error.contains(&hocmesh_core::roles::PREFILL_UPLINK_KBPS.to_string()),
            "the refusal does not say how fast a link would have to be: {error}"
        );
        assert!(
            error.contains("without pipeline parallelism"),
            "the refusal does not say the request is still servable: {error}"
        );
    }

    /// Two stages on one box share a memory bus and a power supply. That is
    /// not a pipeline, and one failure would take out two stages of it.
    #[test]
    fn two_stages_never_land_on_the_same_machine() {
        let mut candidates = vec![placed("one", Some(0), true)];
        // The same machine offering a second and third accelerator.
        for device in 1..3 {
            let mut extra = placed("one", Some(0), true);
            extra.device_id = format!("dev-one-{device}");
            candidates.push(extra);
        }
        candidates.push(placed("two", Some(10), true));
        let error = order_pipeline(&candidates, 3).unwrap_err().to_string();
        assert!(
            error.contains("distinct machines"),
            "three devices on two machines were accepted as a three-stage \
             pipeline: {error}"
        );

        candidates.push(placed("three", Some(20), true));
        let chain = order_pipeline(&candidates, 3).unwrap();
        let machines: BTreeSet<&str> = chain.iter().map(|c| c.node_id.as_str()).collect();
        assert_eq!(machines.len(), 3);
    }

    /// A machine that has never fitted a position cannot win a place by being
    /// unmeasured. Unknown is charged more than any real link, not less.
    #[test]
    fn an_unplaced_machine_does_not_look_nearby() {
        let candidates = vec![
            placed("head", Some(0), true),
            placed("nowhere", None, true),
            placed("near", Some(1_000), true),
        ];
        let chain = order_pipeline(&candidates, 2).unwrap();
        assert_eq!(
            chain[1].node_id, "near",
            "a machine with no coordinate was preferred over a measured \
             neighbour, so not reporting a position is now an advantage"
        );
    }

    /// Ranking may guess about an unmeasured link; the head of a pipeline may
    /// not. This is the seam between the two, exercised end to end.
    #[test]
    fn only_a_measured_head_is_planned_into_a_pipeline() {
        let candidates = [
            placed("slow-but-close", Some(0), false),
            placed("fast", Some(80_000), true),
            placed("fast-neighbour", Some(80_100), true),
        ];
        let plan = plan_parallelism(&candidates, 12, &requirements()).unwrap();
        validate_plan(&plan, 12, requirements().batch_size).unwrap();
        assert_eq!(
            plan.pipeline
                .first()
                .expect("a pipeline was planned")
                .node_id,
            "fast",
            "the first stage went to a machine that has not shown it can push \
             activations out"
        );
    }
}
