use base64::{Engine as _, engine::general_purpose::STANDARD_NO_PAD};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

/// Bumped to 6 when prime shards started pricing by the divisions they really
/// cost instead of a flat rate per candidate. Prices are consensus-visible, so
/// history certified under the old rate must not validate under the new one.
pub const PROTOCOL_VERSION: u32 = 6;
pub const AUTH_MAX_CLOCK_SKEW_SECS: i64 = 300;
pub const DEFAULT_LEASE_SECONDS: i64 = 900;

/// The longest a shard may be leased to a node that needs longer than most.
///
/// A lease is a timeout, not a price: nothing on the chain reads it, and a
/// shard is worth the same mCU however long its holder took. So the ceiling
/// exists only to bound how long a shard can be parked before the mesh gives
/// up on it -- and pinning every node to `DEFAULT_LEASE_SECONDS` did something
/// quite different. It excluded slower machines outright, because a node
/// predicted to overrun the flat lease was refused the shard rather than
/// given longer to finish it. That is backwards for a network whose whole
/// premise is that modest hardware still has something to contribute.
///
/// Three times the default, which is the spread between a current laptop and
/// a workstation a decade older -- roughly the widest gap worth scheduling
/// across before waiting on the slow node costs more than the work is worth.
pub const MAX_LEASE_SECONDS: i64 = DEFAULT_LEASE_SECONDS * 3;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuCapability {
    pub stable_id: String,
    pub vendor: String,
    pub name: String,
    pub backend: String,
    pub memory_mb: Option<u64>,
    pub driver_version: Option<String>,
    pub compute_version: Option<String>,
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub supports_int8: bool,
    pub benchmark_bytes_per_second: Option<u64>,
    pub benchmark_p95_micros: Option<u64>,
}

/// Number of axes in the synthetic latency space.
pub const COORDINATE_DIMENSIONS: usize = 3;

/// A node's position in latency space, carried with its capabilities.
///
/// Fixed-point microseconds rather than floats: this record is signed and
/// compared byte-for-byte, and two honest peers must never disagree about
/// its encoding because their platforms round differently.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct NetworkCoordinate {
    /// Position along each axis, in microseconds.
    pub vector_micros: [i64; COORDINATE_DIMENSIONS],
    /// Access-link cost crossed at both ends of any path, in microseconds.
    pub height_micros: i64,
    /// Confidence in this position, per mille; 1000 means "no confidence".
    pub error_permille: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NodeCapabilities {
    pub protocol_version: u32,
    pub hostname: String,
    pub os: String,
    pub arch: String,
    pub cpu_brand: String,
    pub logical_cpus: usize,
    pub total_memory_bytes: u64,
    pub cpu_benchmark_score: u64,
    pub gpus: Vec<GpuCapability>,
    #[serde(default)]
    pub model_seed_url: Option<String>,
    #[serde(default)]
    pub cached_model_manifests: Vec<String>,
    #[serde(default)]
    pub coordinator_latency_micros: u64,
    #[serde(default)]
    pub model_bandwidth_kbps: u64,
    #[serde(default)]
    pub accelerator_load_permille: u16,
    #[serde(default)]
    pub ai_runtime_ready: bool,
    /// Workers this node will actually run, after applying operator limits.
    #[serde(default)]
    pub shared_logical_cpus: usize,
    /// RAM this node is willing to let a workload occupy.
    #[serde(default)]
    pub shared_memory_bytes: u64,
    /// Share of GPU the operator lends; 0 means the GPU is not offered.
    #[serde(default)]
    pub shared_gpu_percent: u8,
    /// Where this node sits in latency space, once it has measured enough
    /// peers to have a position. `None` means "unknown" and must never be
    /// scored as "nearby".
    #[serde(default)]
    pub network_coordinate: Option<NetworkCoordinate>,
    /// Base URL at which this node answers latency probes, if it is reachable.
    ///
    /// Opt-in and independent of measuring: probing is outbound, so a node
    /// behind NAT still fits its own coordinate. It just cannot be a target,
    /// and so leaves this `None`.
    #[serde(default)]
    pub probe_endpoint: Option<String>,
}

/// Replay-resistant request authentication. The nonce must be unique per node
/// during the server replay window and is included in the Ed25519 signature.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthProof {
    pub node_id: String,
    pub timestamp: i64,
    pub nonce_b64: String,
    pub signature_b64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterRequest {
    pub auth: AuthProof,
    pub public_key_b64: String,
    pub capabilities: NodeCapabilities,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterResponse {
    pub node_id: String,
    pub balance_mcu: i64,
    pub protocol_version: u32,
    pub ledger_mode: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    pub auth: AuthProof,
    pub capabilities: NodeCapabilities,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollRequest {
    pub auth: AuthProof,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkSpec {
    PrimeCount {
        start: u64,
        end: u64,
    },
    /// Rows `[row_start, row_end)` of `C = A x B` over the field of integers
    /// modulo 2^31-1. `A` and `B` are not transmitted: both are generated
    /// element-wise from their seed, so the spec stays a handful of bytes no
    /// matter how large the multiplication is. That is what makes this workload
    /// worth distributing -- the compute-to-bytes ratio scales with `dim`.
    MatrixMultiply {
        seed_a: u64,
        seed_b: u64,
        dim: u32,
        row_start: u32,
        row_end: u32,
    },
    /// The longest Collatz trajectory starting anywhere in `[start, end)`.
    ///
    /// A distributed search that people really run, and the cheapest possible
    /// spec: two integers describe arbitrarily much work. Every step is integer
    /// arithmetic on `u128`, so two machines agree exactly.
    CollatzPeak {
        start: u64,
        end: u64,
    },
}
/// How much of a result a validator can check from the ledger entry alone.
///
/// This is the property that decides whether a workload may be paid for with
/// newly issued CU. A shard whose answer fits in its entry can be audited by
/// anyone holding the chain. A shard whose answer is a matrix cannot: the
/// audit needs the provider to reveal sampled blocks after the challenge is
/// drawn, and until that exchange exists a validator has nothing to check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditClass {
    /// The entry carries the whole answer, so any validator can redraw the
    /// challenge and recheck a sample without talking to anyone.
    SelfContained,
    /// The answer is too large for an entry. Auditing it needs a reveal the
    /// provider supplies after the challenge is drawn.
    RevealRequired,
}

impl WorkSpec {
    /// What a validator can check about this workload on its own.
    ///
    /// Every current workload answers in a few dozen integers, so all three
    /// are self-contained. The match is exhaustive on purpose: adding a
    /// workload forces the author to state which side of the line it falls on.
    pub fn audit_class(&self) -> AuditClass {
        match self {
            WorkSpec::PrimeCount { .. }
            | WorkSpec::MatrixMultiply { .. }
            | WorkSpec::CollatzPeak { .. } => AuditClass::SelfContained,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match self {
            WorkSpec::PrimeCount { start, end } if start >= end => {
                Err("prime_count requires start < end".into())
            }
            WorkSpec::PrimeCount { start, end } if end.saturating_sub(*start) > 2_000_000_000 => {
                Err("prime_count range is too large for one submitted job".into())
            }
            WorkSpec::MatrixMultiply { dim, .. } if *dim == 0 || *dim > 512 => {
                Err("matrix_multiply dim must be between 1 and 512".into())
            }
            WorkSpec::MatrixMultiply {
                row_start, row_end, ..
            } if row_start >= row_end => Err("matrix_multiply requires row_start < row_end".into()),
            WorkSpec::MatrixMultiply { dim, row_end, .. } if row_end > dim => {
                Err("matrix_multiply row_end must not exceed dim".into())
            }
            WorkSpec::CollatzPeak { start, end } if start >= end => {
                Err("collatz_peak requires start < end".into())
            }
            WorkSpec::CollatzPeak { start, .. } if *start == 0 => {
                Err("collatz_peak has no trajectory from zero".into())
            }
            WorkSpec::CollatzPeak { start, end } if end.saturating_sub(*start) > 2_000_000_000 => {
                Err("collatz_peak range is too large for one submitted job".into())
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkResult {
    PrimeCount {
        count: u64,
        bucket_counts: Vec<u64>,
        duration_ms: u64,
    },
    MatrixMultiply {
        rows: Vec<u32>,
        duration_ms: u64,
    },
    /// The peak of one Collatz shard, bucketed the same way a prime count is.
    ///
    /// The per-bucket peak and the seed that reached it are both carried, so an
    /// audit that redraws a bucket can check the whole claim rather than only
    /// the arithmetic that combines the buckets.
    CollatzPeak {
        peak_steps: u32,
        peak_seed: u64,
        bucket_peaks: Vec<u32>,
        bucket_seeds: Vec<u64>,
        duration_ms: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkAssignment {
    pub assignment_id: String,
    pub job_id: String,
    pub shard_index: u32,
    pub work: WorkSpec,
    pub reward_mcu: i64,
    pub lease_seconds: i64,
    pub system_funded: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PollResponse {
    pub assignment: Option<WorkAssignment>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultRequest {
    pub auth: AuthProof,
    pub assignment_id: String,
    pub job_id: String,
    pub shard_index: u32,
    pub work: WorkSpec,
    pub reward_mcu: i64,
    pub system_funded: bool,
    pub result: WorkResult,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultResponse {
    pub accepted: bool,
    pub reward_mcu: i64,
    pub balance_mcu: i64,
    pub job_completed: bool,
    pub ledger_entry_hash: Option<String>,
}
/// Asks for one shard's escrow back after its settlement window closed with no
/// result. `auth` signs the refund body for paid work and is absent for
/// community work, where the CU returns to the issuance account it was minted
/// against and there is nobody to authorise on its behalf.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefundRequest {
    pub assignment_id: String,
    pub auth: Option<AuthProof>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefundResponse {
    pub refund_mcu: i64,
    pub paid_to: String,
    pub ledger_entry_hash: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitJobRequest {
    pub auth: AuthProof,
    pub work: WorkSpec,
    pub shards: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitJobResponse {
    pub job_id: String,
    pub reserved_mcu: i64,
    pub balance_mcu: i64,
    pub assignments: u32,
    pub ledger_entry_hash: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceResponse {
    pub node_id: String,
    pub balance_mcu: i64,
    pub earned_mcu: i64,
    pub spent_mcu: i64,
    pub ledger_height: Option<u64>,
    pub ledger_head: Option<String>,
}
/// One movement of CU into or out of a node's account.
///
/// A balance says where a node stands; this says how it got there, which is
/// what an operator looking at a dashboard actually wants to see. It is
/// deliberately not the ledger's own `AccountHistoryEntry`: a coordinator
/// running without validators has no sequence or transaction to report, and a
/// federated one has no local category. Both can fill this, and a reader can
/// tell which by which fields are present.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// Positive for CU earned, negative for CU spent.
    pub delta_mcu: i64,
    /// Why it moved, in the coordinator's own words -- `reward`, `reserve`,
    /// `refund` and so on. Absent when the entry came from the chain, which
    /// records transactions rather than categories.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assignment_id: Option<String>,
    /// The ledger height this posting landed at, when there is a chain behind
    /// the coordinator. This is what makes a row checkable against a
    /// certificate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transaction_id: Option<String>,
    pub created_at: i64,
}

/// A page of one node's history, newest first.
///
/// `next_before` is the cursor for the page after this one and is absent at
/// the start of history. Paging on the position of the last row rather than on
/// an offset keeps a page correct while new entries land above it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerHistoryResponse {
    pub node_id: String,
    /// Whether these rows came from the validator quorum or from the
    /// coordinator's own table. A dashboard should say which it is showing:
    /// only the first is authoritative.
    pub authoritative: bool,
    pub entries: Vec<LedgerEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_before: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStatusResponse {
    pub job_id: String,
    pub requester_node_id: Option<String>,
    pub system_funded: bool,
    pub status: String,
    pub total_assignments: u32,
    pub completed_assignments: u32,
    pub reserved_mcu: i64,
    pub prime_count_total: Option<u64>,
    /// The longest Collatz trajectory across the shards that have finished,
    /// and the smallest seed that reached it. Absent until one completes.
    #[serde(default)]
    pub collatz_peak: Option<CollatzPeakTotal>,
    /// Shards whose settlement window has closed with nothing delivered, so
    /// the requester can sign for their escrow back without having kept the
    /// work spec from the day they submitted it. What is named here is only
    /// a suggestion: the ledger checks every field against the reservation
    /// it certified, so a coordinator that lies gets a refused refund.
    #[serde(default)]
    pub refundable: Vec<RefundableShard>,
}

/// A whole job's Collatz answer, rolled up from its shards.
///
/// Two shards can tie on length, so the seed is carried and the smaller one
/// wins. Without that rule two coordinators could report different seeds for
/// the same finished job and both be telling the truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollatzPeakTotal {
    pub steps: u32,
    pub seed: u64,
}

/// One shard a requester may reclaim, with everything needed to sign for it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefundableShard {
    pub assignment_id: String,
    pub shard_index: u32,
    pub work: WorkSpec,
    pub refund_mcu: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatusResponse {
    pub node_id: String,
    pub last_seen_unix: i64,
    pub online: bool,
    pub capabilities: NodeCapabilities,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatsResponse {
    pub registered_nodes: u64,
    pub online_nodes: u64,
    pub pending_assignments: u64,
    pub leased_assignments: u64,
    pub completed_assignments: u64,
    pub total_available_mcu: i64,
    pub ledger_mode: String,
}
/// One intent the coordinator persisted but has not finished settling.
///
/// Reported, never acted on by a reader: the coordinator is not the authority
/// for CU, so this view exists to say what is stuck and why, not to let anyone
/// nudge it along out of band.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerIntentState {
    pub claim_key: String,
    pub intent_kind: String,
    pub object_id: String,
    pub status: String,
    pub attempts: i64,
    pub last_error: Option<String>,
    pub entry_hash: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// What the coordinator still owes the ledger, or the ledger still owes it.
///
/// `orphaned_objects` is the half no daemon can repair: work the coordinator
/// parked waiting on funding that no pending intent covers any more. Settling
/// it would mean the coordinator writing CU into existence on its own say-so,
/// which it is never permitted to do, so it is surfaced and left alone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconciliationResponse {
    pub unsettled: Vec<LedgerIntentState>,
    pub orphaned_objects: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// One reachable peer a node may measure itself against.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PeerSample {
    pub node_id: String,
    /// Base URL of the peer's probe endpoint.
    pub probe_endpoint: String,
    /// The peer's last advertised position, if it has fitted one.
    pub coordinate: Option<NetworkCoordinate>,
}

/// A bootstrap list of probe targets.
///
/// The coordinator is a convenient directory, not an authority: it never
/// measures anything and never sees a round trip. A gossip layer can replace
/// this endpoint without touching how coordinates are fitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerSampleResponse {
    pub peers: Vec<PeerSample>,
}

/// A latency probe. The body is deliberately tiny so that what is timed is
/// the round trip rather than the transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeRequest {
    /// The caller's position, so the peer can fit against us as we fit
    /// against it. `None` from a node that has not yet been placed.
    #[serde(default)]
    pub coordinate: Option<NetworkCoordinate>,
    /// A round trip the caller already measured to this peer.
    ///
    /// The responder cannot time the exchange itself, so this is the only way
    /// it learns the distance. It is an untrusted number, bounded on arrival
    /// by the same limits as any other observation.
    #[serde(default)]
    pub measured_rtt_micros: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResponse {
    pub node_id: String,
    /// The responder's position as it stands, whether or not it is yet
    /// confident enough to advertise it for scheduling. `error_permille`
    /// carries how much of it to believe. `None` means the responder declined
    /// to give one, which the caller must treat as unusable rather than as a
    /// position at the origin.
    pub coordinate: Option<NetworkCoordinate>,
}

pub fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

pub fn canonical_auth_message(
    action: &str,
    node_id: &str,
    timestamp: i64,
    nonce_b64: &str,
    body_hash: &str,
) -> String {
    format!("hocmesh-v{PROTOCOL_VERSION}|{action}|{node_id}|{timestamp}|{nonce_b64}|{body_hash}")
}

pub fn hash_bytes(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex_lower(&h.finalize())
}
/// What an inference job costs, as the requester declares it.
///
/// The prompts never reach the ledger: only their digest and their sizes,
/// which is all a price depends on. A validator can check the bill exactly
/// without reading anybody's text, and a requester does not have to publish a
/// private prompt to prove they were charged correctly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceBilling {
    pub manifest_digest: String,
    pub parameter_count: u64,
    pub total_size_bytes: u64,
    pub prompts_digest: String,
    pub prompt_bytes: Vec<u64>,
    pub max_tokens: u32,
    pub max_cost_mcu: i64,
}

/// One contiguous run of prompts, and the node that agreed to run it.
///
/// Position in the reservation's list is the batch index, so an assignment id
/// is derivable rather than assignable: a coordinator cannot invent a batch
/// after the fact and it cannot rename one it already certified.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PricedBatch {
    pub batch_start: u32,
    pub batch_end: u32,
    pub node_id: String,
}

/// A model cannot hold more parameters than its bytes can store.
///
/// Four-bit quantisation is the densest thing anyone ships, at half a byte a
/// parameter, so twice the file size is a hard ceiling. Without it a publisher
/// could declare a toy model as a frontier one and bill inference on it at a
/// hundred times what it costs to run - and the file size is the one number in
/// a manifest that content-addressed chunks already pin down.
pub fn parameter_count_is_plausible(parameter_count: u64, total_size_bytes: u64) -> bool {
    parameter_count > 0 && parameter_count <= total_size_bytes.saturating_mul(2)
}

/// An inference job's id, bound to the nonce its requester signed.
///
/// The requester fixes the id, not the coordinator: an id nobody can choose
/// after the fact is what stops a coordinator quietly re-pointing an escrow.
pub fn inference_job_id_from_auth(proof: &AuthProof) -> String {
    format!("ai_{}", &hash_bytes(proof.nonce_b64.as_bytes())[..32])
}

/// The deterministic id of one batch of an inference job.
pub fn inference_assignment_id(job_id: &str, index: u32) -> String {
    format!("aiasg_{}_{}", &hash_bytes(job_id.as_bytes())[..20], index)
}

pub fn inference_billing_hash(billing: &InferenceBilling) -> Result<String, serde_json::Error> {
    hash_json(billing)
}

/// What a requester signs to authorise an inference job.
///
/// Split in two so that a validator holding only the billing and a digest of
/// the settings can recompute it. That is what keeps prompt text off the
/// ledger while leaving the signature checkable.
pub fn inference_submit_body_hash(
    billing_hash: &str,
    settings_digest: &str,
) -> Result<String, serde_json::Error> {
    hash_json(&("inference_submit", billing_hash, settings_digest))
}

/// What a provider signs to claim one batch.
pub fn inference_reward_body_hash(
    assignment_id: &str,
    job_id: &str,
    batch_start: u32,
    batch_end: u32,
    reward_mcu: i64,
    outputs_digest: &str,
) -> Result<String, serde_json::Error> {
    hash_json(&(
        "inference_reward",
        assignment_id,
        job_id,
        batch_start,
        batch_end,
        reward_mcu,
        outputs_digest,
    ))
}

/// What a requester signs to reclaim one batch nobody delivered.
pub fn inference_refund_body_hash(
    assignment_id: &str,
    job_id: &str,
    batch_start: u32,
    batch_end: u32,
    refund_mcu: i64,
) -> Result<String, serde_json::Error> {
    hash_json(&(
        "inference_refund",
        assignment_id,
        job_id,
        batch_start,
        batch_end,
        refund_mcu,
    ))
}
/// What a requester signs to admit that a batch reached it.
///
/// A receipt is not approval. It says only that the requester now holds the
/// bytes behind `outputs_digest`, and that is the moment its escrow stops
/// being refundable: from here the batch settles either to the provider or
/// back to the commons, but never quietly back to the party that already has
/// the answer.
pub fn inference_receipt_body_hash(
    assignment_id: &str,
    job_id: &str,
    batch_start: u32,
    batch_end: u32,
    price_mcu: i64,
    outputs_digest: &str,
) -> Result<String, serde_json::Error> {
    hash_json(&(
        "inference_receipt",
        assignment_id,
        job_id,
        batch_start,
        batch_end,
        price_mcu,
        outputs_digest,
    ))
}

/// What a requester signs to accept - or to reject - what it was delivered.
///
/// The verdict is the only judgement of an inference answer anybody can make,
/// because no validator can re-run a model to see whether the answer was real.
/// Both directions are signed over the same digest, so a requester cannot
/// accept one set of bytes and dispute another, and the two hash differently
/// so an acceptance can never be replayed as a rejection.
pub fn inference_verdict_body_hash(
    accepted: bool,
    assignment_id: &str,
    job_id: &str,
    batch_start: u32,
    batch_end: u32,
    price_mcu: i64,
    outputs_digest: &str,
) -> Result<String, serde_json::Error> {
    hash_json(&(
        if accepted {
            "inference_accept"
        } else {
            "inference_dispute"
        },
        assignment_id,
        job_id,
        batch_start,
        batch_end,
        price_mcu,
        outputs_digest,
    ))
}

pub fn hash_json<T: Serialize + ?Sized>(value: &T) -> Result<String, serde_json::Error> {
    Ok(hash_bytes(&serde_json::to_vec(value)?))
}
pub fn node_id_from_public_key(public_key: &[u8; 32]) -> String {
    let d = Sha256::digest(public_key);
    format!("hocmesh_{}", hex_lower(&d[..16]))
}
/// The node id behind a base64 public key, or `None` if it is not one.
///
/// Callers that hold a key as it appears on the wire or in ledger evidence
/// should not have to know the encoding or repeat the length check.
pub fn node_id_from_public_key_b64(public_key_b64: &str) -> Option<String> {
    let raw = STANDARD_NO_PAD.decode(public_key_b64).ok()?;
    let key: [u8; 32] = raw.try_into().ok()?;
    Some(node_id_from_public_key(&key))
}

pub fn job_id_from_auth(proof: &AuthProof) -> String {
    format!("job_{}", &hash_bytes(proof.nonce_b64.as_bytes())[..32])
}
pub fn assignment_id(job_id: &str, shard_index: u32) -> String {
    format!(
        "asg_{}_{}",
        &hash_bytes(job_id.as_bytes())[..20],
        shard_index
    )
}

/// Verifies identity + signature only. Use this for historical ledger audit.
pub fn verify_auth_signature(
    public_key_b64: &str,
    proof: &AuthProof,
    action: &str,
    body_hash: &str,
) -> Result<(), String> {
    let pk = STANDARD_NO_PAD
        .decode(public_key_b64)
        .map_err(|_| "invalid public key encoding".to_string())?;
    let pk: [u8; 32] = pk
        .try_into()
        .map_err(|_| "public key must be 32 bytes".to_string())?;
    if node_id_from_public_key(&pk) != proof.node_id {
        return Err("node id does not match public key".into());
    }
    let sb = STANDARD_NO_PAD
        .decode(&proof.signature_b64)
        .map_err(|_| "invalid signature encoding".to_string())?;
    let sig = Signature::from_slice(&sb).map_err(|_| "invalid Ed25519 signature".to_string())?;
    let vk = VerifyingKey::from_bytes(&pk).map_err(|_| "invalid Ed25519 public key".to_string())?;
    let msg = canonical_auth_message(
        action,
        &proof.node_id,
        proof.timestamp,
        &proof.nonce_b64,
        body_hash,
    );
    vk.verify(msg.as_bytes(), &sig)
        .map_err(|_| "signature verification failed".to_string())
}

/// Live API verification additionally enforces a short clock-skew window.
pub fn verify_auth(
    public_key_b64: &str,
    proof: &AuthProof,
    action: &str,
    body_hash: &str,
) -> Result<(), String> {
    if (now_unix() - proof.timestamp).abs() > AUTH_MAX_CLOCK_SKEW_SECS {
        return Err("authentication timestamp is outside the allowed clock skew".into());
    }
    if proof.nonce_b64.len() < 16 {
        return Err("authentication nonce is missing or too short".into());
    }
    verify_auth_signature(public_key_b64, proof, action, body_hash)
}

pub fn empty_body_hash() -> String {
    hash_bytes(b"")
}
pub fn register_body_hash(pk: &str, c: &NodeCapabilities) -> Result<String, serde_json::Error> {
    hash_json(&(pk, c))
}
pub fn heartbeat_body_hash(c: &NodeCapabilities) -> Result<String, serde_json::Error> {
    hash_json(c)
}
pub fn result_body_hash(
    id: &str,
    job_id: &str,
    shard_index: u32,
    work: &WorkSpec,
    reward_mcu: i64,
    system_funded: bool,
    r: &WorkResult,
) -> Result<String, serde_json::Error> {
    hash_json(&(id, job_id, shard_index, work, reward_mcu, system_funded, r))
}
pub fn submit_body_hash(w: &WorkSpec, s: u32) -> Result<String, serde_json::Error> {
    hash_json(&(w, s))
}

/// How long a reserved shard has to settle before its CU may be refunded.
///
/// A reward and a refund for one shard are deliberately disjoint in time:
/// inside the window only a reward is valid, outside it only a refund. That
/// removes the race rather than adjudicating it: a requester cannot pull CU
/// out from under a provider still inside the window it agreed to, and a
/// provider cannot claim a shard the mesh has already written off.
///
/// It is measured from the reserve's own `created_at`, which is on the chain,
/// and never from a coordinator's lease, which is not. The coordinator is
/// never the authority for CU.
pub const SETTLEMENT_WINDOW_SECS: i64 = DEFAULT_LEASE_SECONDS * 4;

/// The window has to outlast the longest lease inside it, or a provider still
/// working under a lease the coordinator granted would find the CU already
/// reclaimed when it delivered. The slack left over is what a shard may spend
/// queued before it is handed out, since the window runs from the reserve and
/// the lease only starts at assignment.
///
/// Checked here rather than left to prose: this constant is consensus-visible,
/// so widening the lease past it is a protocol change and must be seen to be
/// one instead of quietly breaking settlement for slow nodes.
const _: () = assert!(MAX_LEASE_SECONDS < SETTLEMENT_WINDOW_SECS);

/// The body a requester signs to ask for an unsettled shard's CU back.
///
/// The leading tag keeps a signature over a result body from ever being
/// replayed as a refund request.
pub fn refund_body_hash(
    id: &str,
    job_id: &str,
    shard_index: u32,
    work: &WorkSpec,
    refund_mcu: i64,
    system_funded: bool,
) -> Result<String, serde_json::Error> {
    hash_json(&(
        "refund",
        id,
        job_id,
        shard_index,
        work,
        refund_mcu,
        system_funded,
    ))
}

fn hex_lower(bytes: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut o = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        o.push(H[(b >> 4) as usize] as char);
        o.push(H[(b & 15) as usize] as char);
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::STANDARD_NO_PAD;
    use ed25519_dalek::{Signer, SigningKey};

    #[test]
    fn ids_stable() {
        let k = [7u8; 32];
        assert_eq!(node_id_from_public_key(&k), node_id_from_public_key(&k));
    }
    #[test]
    fn invalid_range() {
        assert!(
            WorkSpec::PrimeCount { start: 5, end: 5 }
                .validate()
                .is_err()
        );
    }

    #[test]
    fn expired_live_signature_is_rejected_but_historical_signature_verifies() {
        let key = SigningKey::from_bytes(&[42u8; 32]);
        let public_key_b64 = STANDARD_NO_PAD.encode(key.verifying_key().to_bytes());
        let node_id = node_id_from_public_key(&key.verifying_key().to_bytes());
        let timestamp = now_unix() - AUTH_MAX_CLOCK_SKEW_SECS - 1;
        let nonce_b64 = "old-nonce-1234567890".to_string();
        let body_hash = empty_body_hash();
        let message = canonical_auth_message("poll", &node_id, timestamp, &nonce_b64, &body_hash);
        let proof = AuthProof {
            node_id,
            timestamp,
            nonce_b64,
            signature_b64: STANDARD_NO_PAD.encode(key.sign(message.as_bytes()).to_bytes()),
        };

        assert!(verify_auth(&public_key_b64, &proof, "poll", &body_hash).is_err());
        verify_auth_signature(&public_key_b64, &proof, "poll", &body_hash).unwrap();
    }
}
