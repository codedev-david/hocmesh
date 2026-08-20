use base64::{Engine as _, engine::general_purpose::STANDARD_NO_PAD};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

pub const PROTOCOL_VERSION: u32 = 2;
pub const AUTH_MAX_CLOCK_SKEW_SECS: i64 = 300;
pub const DEFAULT_LEASE_SECONDS: i64 = 900;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuCapability {
    pub vendor: String,
    pub name: String,
    pub backend: String,
    pub memory_mb: Option<u64>,
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
    PrimeCount { start: u64, end: u64 },
}
impl WorkSpec {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            WorkSpec::PrimeCount { start, end } if start >= end => {
                Err("prime_count requires start < end".into())
            }
            WorkSpec::PrimeCount { start, end } if end.saturating_sub(*start) > 2_000_000_000 => {
                Err("prime_count range is too large for one submitted job".into())
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkResult {
    PrimeCount { count: u64, duration_ms: u64 },
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
    format!("mesh-v{PROTOCOL_VERSION}|{action}|{node_id}|{timestamp}|{nonce_b64}|{body_hash}")
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
    format!("mesh_{}", hex_lower(&d[..16]))
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
