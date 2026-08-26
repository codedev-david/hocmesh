use hocmesh_protocol::{AuthProof, WorkResult, WorkSpec};
use serde::{Deserialize, Serialize};

pub const COMMUNITY_ISSUANCE_ACCOUNT: &str = "hocmesh:community:issuance";
pub fn escrow_account(job_id: &str) -> String {
    format!("hocmesh:escrow:{job_id}")
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Posting {
    pub account_id: String,
    pub delta_mcu: i64,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TransactionKind {
    JobReserve,
    CommunityReserve,
    ProviderReward,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobReserveEvidence {
    pub job_id: String,
    pub requester_public_key_b64: String,
    pub requester_auth: AuthProof,
    pub work: WorkSpec,
    pub shards: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderRewardEvidence {
    pub job_id: String,
    pub assignment_id: String,
    pub shard_index: u32,
    pub reward_mcu: i64,
    pub provider_public_key_b64: String,
    pub provider_auth: AuthProof,
    pub work: WorkSpec,
    pub result: WorkResult,
    pub system_funded: bool,
    /// The challenge the coordinator drew for its own provisional check.
    ///
    /// Advisory only. Validators derive the authoritative challenge from the
    /// entry's chain position instead, because a coordinator colluding with a
    /// provider would otherwise be free to choose an audit that finds nothing.
    /// Kept as an audit trail of what the coordinator claims it checked.
    #[serde(default)]
    pub provisional_audit_nonce: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TransactionEvidence {
    JobReserve(JobReserveEvidence),
    CommunityReserve {
        job_id: String,
        work: WorkSpec,
        shards: u32,
    },
    ProviderReward(ProviderRewardEvidence),
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerTransaction {
    pub transaction_id: String,
    pub kind: TransactionKind,
    pub postings: Vec<Posting>,
    pub evidence: TransactionEvidence,
    pub created_at: i64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub sequence: u64,
    pub previous_hash: String,
    pub transaction: LedgerTransaction,
    pub transaction_hash: String,
    pub entry_hash: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorSignature {
    pub validator_id: String,
    pub signature_b64: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuorumCertificate {
    pub entry: LedgerEntry,
    pub membership_hash: String,
    pub signatures: Vec<ValidatorSignature>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidatorMember {
    pub validator_id: String,
    pub url: String,
    pub public_key_b64: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidatorSet {
    pub threshold: usize,
    pub community_issuance_limit_mcu: i64,
    pub members: Vec<ValidatorMember>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerHead {
    pub sequence: u64,
    pub entry_hash: String,
    pub membership_hash: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadProof {
    pub head: LedgerHead,
    pub validator_id: String,
    pub signature_b64: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceProof {
    pub account_id: String,
    pub balance_mcu: i64,
    pub earned_mcu: i64,
    pub spent_mcu: i64,
    pub head: LedgerHead,
    pub validator_id: String,
    pub signature_b64: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimProof {
    pub claim_key: String,
    pub sequence: Option<u64>,
    pub entry_hash: Option<String>,
    pub certificate: Option<QuorumCertificate>,
    pub head: LedgerHead,
    pub validator_id: String,
    pub signature_b64: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalRequest {
    pub transaction: LedgerTransaction,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalVote {
    pub accepted: bool,
    pub validator_id: String,
    pub sequence: u64,
    pub previous_hash: String,
    pub entry_hash: String,
    pub signature_b64: Option<String>,
    pub error: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitResponse {
    pub committed: bool,
    pub head: LedgerHead,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntriesResponse {
    pub certificates: Vec<QuorumCertificate>,
}
