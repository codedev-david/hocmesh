use hocmesh_protocol::{AuthProof, InferenceBilling, PricedBatch, WorkResult, WorkSpec};
use serde::{Deserialize, Serialize};

pub const COMMUNITY_ISSUANCE_ACCOUNT: &str = "hocmesh:community:issuance";
pub fn escrow_account(job_id: &str) -> String {
    format!("hocmesh:escrow:{job_id}")
}

/// Where one delivered batch waits while the requester makes up its mind.
///
/// Escrow that has reached this account can no longer be refunded: the
/// requester already holds the answer, so the only honest destinations left
/// are the provider that produced it or the commons.
pub fn inference_holding_account(assignment_id: &str) -> String {
    format!("hocmesh:holding:{assignment_id}")
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
    InferenceReceipt,
    InferenceReward,
    InferenceDispute,
    InferenceExpiry,
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
        /// Named validators putting their own keys behind this mint.
        ///
        /// Community issuance is the one place CU comes from nothing. The
        /// quorum certificate says the entry was agreed; a sponsorship says a
        /// particular validator chose to spend the shared budget on this
        /// particular job, and that stays legible long after the set rotates.
        sponsors: Vec<ValidatorSignature>,
    },
    ProviderReward(ProviderRewardEvidence),
    JobRefund(JobRefundEvidence),
    InferenceReserve(InferenceReserveEvidence),
    InferenceReceipt(InferenceReceiptEvidence),
    InferenceReward(InferenceRewardEvidence),
    InferenceDispute(InferenceDisputeEvidence),
    InferenceExpiry(InferenceExpiryEvidence),
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
/// One movement of CU into or out of an account, as the ledger recorded it.
///
/// A balance says where an account stands; this says how it got there, which
/// is what an operator reconciling a bill or a dispute actually needs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountHistoryEntry {
    pub sequence: u64,
    pub posting_index: u32,
    pub transaction_id: String,
    pub delta_mcu: i64,
    pub created_at: i64,
}

/// A page of an account's history, newest first.
///
/// `next_before` is the cursor for the page after this one, and is absent when
/// the ledger holds nothing older. Paging on the sequence the entry landed at
/// rather than on an offset means a page stays correct while the chain grows
/// underneath the reader.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountHistory {
    pub account_id: String,
    pub entries: Vec<AccountHistoryEntry>,
    pub next_before: Option<u64>,
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
/// One proposer's claim on one height.
///
/// Two clients can reach for the same sequence at the same moment, and a
/// validator will only ever sign one entry there. Ordering the attempts is
/// what lets the later one take the height back instead of both halves of the
/// set sitting on votes that will never add up to a certificate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ballot {
    pub number: u64,
    /// Breaks ties between proposers that picked the same number.
    pub proposer: String,
}
impl Ballot {
    /// Total order over attempts. Equal numbers fall back to the proposer's
    /// name so no two live ballots ever compare equal.
    pub fn outranks(&self, other: &Ballot) -> bool {
        (self.number, &self.proposer) > (other.number, &other.proposer)
    }
}
impl std::fmt::Display for Ballot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.number, self.proposer)
    }
}
/// A proposer asking the set to reserve a height for it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareRequest {
    pub sequence: u64,
    pub ballot: Ballot,
}
/// A validator's answer to a prepare.
///
/// `accepted` is the load-bearing half: a validator that has already signed
/// something at this height hands it back, and the new proposer is obliged to
/// finish that entry rather than its own. That is what stops a superseded
/// round from turning into a fork.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareVote {
    pub promised: bool,
    pub validator_id: String,
    pub sequence: u64,
    pub accepted: Option<AcceptedProposal>,
    /// Whatever ballot this validator is currently holding the height for,
    /// so a proposer that lost knows exactly how high it has to climb.
    pub promised_ballot: Option<Ballot>,
    pub error: Option<String>,
}
/// An entry a validator has already put its signature behind at some height.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptedProposal {
    pub ballot: Ballot,
    pub entry_hash: String,
    pub transactions: Vec<LedgerTransaction>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalRequest {
    /// Every transaction the proposer wants settled in this one entry.
    pub transactions: Vec<LedgerTransaction>,
    /// The height this batch is for, and the attempt it belongs to.
    pub sequence: u64,
    pub ballot: Ballot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalVote {
    pub accepted: bool,
    pub validator_id: String,
    pub sequence: u64,
    pub previous_hash: String,
    pub entry_hash: String,
    pub signature_b64: Option<String>,
    /// The ballot this validator is holding the height for. A vote that fell
    /// short because somebody outbid us is a different problem from one that
    /// fell short because the set disliked the batch.
    pub promised_ballot: Option<Ballot>,
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

/// A requester admitting that one batch of generated output reached it.
///
/// This is the hinge of the whole exchange. Before it, the escrow is still the
/// requester's and an undelivered batch can be reclaimed. After it, the money
/// has left the job and can only reach the provider or the commons - so a
/// requester cannot read an answer and then quietly take its CU back, and a
/// provider cannot be paid for an answer nobody ever saw.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceReceiptEvidence {
    pub job_id: String,
    pub assignment_id: String,
    pub batch_start: u32,
    pub batch_end: u32,
    pub price_mcu: i64,
    /// Binds the receipt to the exact bytes that were handed over.
    pub outputs_digest: String,
    pub requester_public_key_b64: String,
    pub requester_auth: AuthProof,
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
    /// The requester's signature over the same digest.
    ///
    /// Nothing a validator can do proves that a generated answer is the one
    /// the model would have produced, so the ledger does not pretend to. What
    /// it can insist on is that the party out of pocket said, on the record,
    /// that this is the answer it is paying for. Without that a provider with
    /// a real assignment could return any bytes at all and still be paid out
    /// of somebody else's escrow.
    pub requester_public_key_b64: String,
    pub requester_acceptance: AuthProof,
}

/// A requester rejecting what it was given.
///
/// The escrow does not come home. Held CU goes back to community issuance, so
/// rejecting costs the requester exactly what accepting would have - it buys
/// nothing by disputing honest work - while a provider that fabricated an
/// answer collects none of it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceDisputeEvidence {
    pub job_id: String,
    pub assignment_id: String,
    pub batch_start: u32,
    pub batch_end: u32,
    pub price_mcu: i64,
    pub outputs_digest: String,
    pub reason: String,
    pub requester_public_key_b64: String,
    pub requester_auth: AuthProof,
}

/// The commons collecting a batch the requester took and then never judged.
///
/// A receipt moves CU somewhere neither party can reach alone: only an
/// acceptance or a dispute empties a holding account, and both need the
/// requester's signature. A requester that reads its answer and then goes
/// quiet - or loses its key, or simply stops caring - therefore strands that
/// CU forever, and the provider waits forever with it.
///
/// So the passage of time is made a third verdict. Nobody signs an expiry;
/// there is no one left to sign it. What makes it checkable instead is the
/// receipt the requester already signed: it names the batch and the price, and
/// its `AuthProof` carries the moment the requester claimed delivery. Any
/// validator can re-derive that signature and compare the two timestamps, so
/// an expiry is exactly as verifiable years later, replaying from genesis, as
/// it was on the day it settled.
///
/// The CU goes to the commons, never to the sweeper: a permissionless
/// transaction that paid whoever submitted it would be a race, and the race
/// would be the point. Here submitting one is pure cost, which is why the
/// coordinator does it on a timer and anybody else may.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceExpiryEvidence {
    pub job_id: String,
    pub assignment_id: String,
    pub batch_start: u32,
    pub batch_end: u32,
    pub price_mcu: i64,
    pub outputs_digest: String,
    pub requester_public_key_b64: String,
    /// The receipt that filled the holding account, replayed verbatim.
    ///
    /// Both halves of the rule live in here: the signature proves the
    /// requester really took this batch at this price, and `timestamp` is when
    /// it said so, which is what the settlement window is measured from. A
    /// requester could backdate its own receipt to shorten its own window, and
    /// gains nothing by it - disputing was always free to it - while nobody
    /// else can move that timestamp at all without the requester's key.
    pub requester_receipt: AuthProof,
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
