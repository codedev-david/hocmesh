use hocmesh_protocol::{AuthProof, InferenceBilling, PricedBatch, WorkResult, WorkSpec};
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
    JobRefund,
    InferenceReserve,
    InferenceReward,
    InferenceRefund,
    MembershipChange,
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
    JobRefund(JobRefundEvidence),
    InferenceReserve(InferenceReserveEvidence),
    InferenceReward(InferenceRewardEvidence),
    InferenceRefund(InferenceRefundEvidence),
    MembershipChange(MembershipChangeEvidence),
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRefundEvidence {
    pub job_id: String,
    pub assignment_id: String,
    pub shard_index: u32,
    pub refund_mcu: i64,
    pub work: WorkSpec,
    pub system_funded: bool,
    /// Who the CU goes back to, and their authorisation to ask for it.
    ///
    /// Absent exactly when `system_funded`. Minted CU has no requester to
    /// return to: it goes back to the issuance account it was minted against,
    /// and nobody signs for it because nobody receives it.
    pub requester_public_key_b64: Option<String>,
    pub requester_auth: Option<AuthProof>,
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
    /// Every settlement that shares this slot in the chain.
    ///
    /// One entry per transaction meant one full consensus round - three
    /// network phases - per CU movement, which capped the whole network at a
    /// couple of settlements a second. A round costs the same whether it
    /// carries one transaction or five hundred, so entries carry a batch.
    pub transactions: Vec<LedgerTransaction>,
    pub transactions_hash: String,
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
    /// Every transaction the proposer wants settled in this one entry.
    pub transactions: Vec<LedgerTransaction>,
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

/// An inference job's escrow, and the batch partition it was certified against.
///
/// The plan is fixed here rather than recomputed later: which machines are
/// online is not a fact a validator can reproduce, so the partition has to be
/// part of what gets certified. Everything about the *price*, though, is
/// derivable from the billing alone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceReserveEvidence {
    pub job_id: String,
    pub requester_public_key_b64: String,
    pub requester_auth: AuthProof,
    pub billing: InferenceBilling,
    pub settings_digest: String,
    pub batches: Vec<PricedBatch>,
}

/// One provider's claim on one batch of an inference job.
///
/// Only a digest of the outputs is recorded. The ledger is not the place to
/// publish somebody's generated text, and the digest is enough to bind the
/// provider to what it actually returned.
/// What the ledger remembers about a certified inference job.
///
/// The reservation is the requester's own signed statement of which machine
/// takes which batch and what the whole thing costs. That makes it the only
/// honest answer to "was this the claim that was actually agreed?", which is
/// why rewards and refunds are checked against it rather than taken on trust.
#[derive(Debug, Clone)]
pub struct InferenceReservation {
    pub job_id: String,
    pub billing: InferenceBilling,
    pub batches: Vec<PricedBatch>,
    pub requester: String,
    pub reserved_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRewardEvidence {
    pub job_id: String,
    pub assignment_id: String,
    pub batch_start: u32,
    pub batch_end: u32,
    pub reward_mcu: i64,
    pub outputs_digest: String,
    pub provider_public_key_b64: String,
    pub provider_auth: AuthProof,
}

/// A requester taking back the escrow on a batch nobody delivered.
///
/// Shares a claim key with the reward for the same batch, so a batch settles
/// once and in one direction, exactly as a prime shard does.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRefundEvidence {
    pub job_id: String,
    pub assignment_id: String,
    pub batch_start: u32,
    pub batch_end: u32,
    pub refund_mcu: i64,
    pub requester_public_key_b64: String,
    pub requester_auth: AuthProof,
}

/// One validator's signed statement about the state it holds right now.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateProof {
    pub head: LedgerHead,
    pub state_hash: String,
    pub validator_id: String,
    pub signature_b64: String,
}

/// A quorum's agreement about the whole ledger state at one height.
///
/// An audit that always replays from genesis costs more every day the network
/// runs, until eventually nobody can afford to check anything. A checkpoint
/// gives an auditor a starting point a quorum has vouched for, so the work is
/// bounded by how much has happened since rather than by the whole history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerCheckpoint {
    pub head: LedgerHead,
    pub state_hash: String,
    pub signatures: Vec<ValidatorSignature>,
}

/// Whether a membership change lets a validator in or puts one out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MembershipAction {
    Join,
    Leave,
}

/// A change to who is allowed to certify entries, recorded in the chain itself.
///
/// Earning CU is already Sybil-proof: a fake node's results fail the
/// recompute. What spinning up machines could still buy is a seat at the
/// quorum, and a captured quorum can certify anything at all. So the validator
/// set is the one part of hocMESH that is deliberately not open: a joiner
/// needs existing members to sign for it by name, and the whole history of who
/// admitted whom replays out of the ledger instead of living in a file that
/// every operator has to be trusted to have edited identically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembershipChangeEvidence {
    pub action: MembershipAction,
    pub member: ValidatorMember,
    /// The consensus threshold the set carries once this change lands.
    pub threshold: usize,
    /// Named sponsors from the set as it stands before the change.
    ///
    /// Separate from the quorum certificate on purpose. The certificate says
    /// the entry is agreed; a vouch says a particular validator put its name
    /// to this particular admission, and that stays legible in the evidence
    /// long after the set has rotated past everyone involved.
    pub vouches: Vec<ValidatorSignature>,
    pub resulting_set_hash: String,
}
