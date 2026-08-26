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
    /// Both current workloads answer in a few dozen integers, so both are
    /// self-contained. The match is exhaustive on purpose: adding a workload
    /// forces the author to state which side of the line it falls on.
    pub fn audit_class(&self) -> AuditClass {
        match self {
            WorkSpec::PrimeCount { .. } | WorkSpec::MatrixMultiply { .. } => {
                AuditClass::SelfContained
            }
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
    /// Shards whose settlement window has closed with nothing delivered, so
    /// the requester can sign for their escrow back without having kept the
    /// work spec from the day they submitted it. What is named here is only
    /// a suggestion: the ledger checks every field against the reservation
    /// it certified, so a coordinator that lies gets a refused refund.
    #[serde(default)]
    pub refundable: Vec<RefundableShard>,
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
pub fn hash_json<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    Ok(hash_bytes(&serde_json::to_vec(value)?))
}
pub fn node_id_from_public_key(public_key: &[u8; 32]) -> String {
    let d = Sha256::digest(public_key);
    format!("hocmesh_{}", hex_lower(&d[..16]))
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
