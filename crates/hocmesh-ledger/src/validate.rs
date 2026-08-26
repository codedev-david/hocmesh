use crate::types::*;
use anyhow::{Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD_NO_PAD};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use hocmesh_core::compute::{split_work, work_cost_mcu};
use hocmesh_core::verify::{self, AuditNonce};
use hocmesh_protocol::{
    PricedBatch, hash_json, refund_body_hash, result_body_hash, submit_body_hash,
    verify_auth_signature,
};

pub fn validate_validator_set(set: &ValidatorSet) -> Result<()> {
    let n = set.members.len();
    if n == 0 {
        bail!("validator set is empty")
    };
    if set.threshold == 0 || set.threshold > n {
        bail!("invalid validator threshold")
    };
    if set.threshold * 3 <= n * 2 {
        bail!("validator threshold must be greater than two thirds of membership")
    };
    if set.community_issuance_limit_mcu <= 0 {
        bail!("community issuance limit must be positive")
    };
    let mut ids = std::collections::HashSet::new();
    let mut urls = std::collections::HashSet::new();
    let mut keys = std::collections::HashSet::new();
    for m in &set.members {
        if !ids.insert(m.validator_id.clone()) {
            bail!("duplicate validator id")
        };
        if !urls.insert(m.url.clone()) {
            bail!("duplicate validator url")
        };
        if !keys.insert(m.public_key_b64.clone()) {
            bail!("duplicate validator public key")
        };
        if m.public_key_b64.is_empty() {
            bail!("validator public key is empty")
        }
    }
    Ok(())
}
pub fn membership_hash(set: &ValidatorSet) -> Result<String> {
    Ok(hash_json(set)?)
}
pub fn transaction_hash(tx: &LedgerTransaction) -> Result<String> {
    Ok(hash_json(tx)?)
}
pub fn transactions_hash(txs: &[LedgerTransaction]) -> Result<String> {
    Ok(hash_json(txs)?)
}
pub fn entry_hash(sequence: u64, previous_hash: &str, tx_hash: &str) -> Result<String> {
    Ok(hash_json(&(sequence, previous_hash, tx_hash))?)
}
pub fn build_entry(
    sequence: u64,
    previous_hash: String,
    transactions: Vec<LedgerTransaction>,
) -> Result<LedgerEntry> {
    let th = transactions_hash(&transactions)?;
    let eh = entry_hash(sequence, &previous_hash, &th)?;
    Ok(LedgerEntry {
        sequence,
        previous_hash,
        transactions,
        transactions_hash: th,
        entry_hash: eh,
    })
}

pub fn claim_key(tx: &LedgerTransaction) -> String {
    match &tx.evidence {
        TransactionEvidence::JobReserve(e) => format!("reserve:{}", e.job_id),
        TransactionEvidence::CommunityReserve { job_id, .. } => format!("reserve:{}", job_id),
        TransactionEvidence::ProviderReward(e) => format!("reward:{}", e.assignment_id),
        TransactionEvidence::JobRefund(e) => format!("reward:{}", e.assignment_id),
        TransactionEvidence::InferenceReserve(e) => format!("reserve:{}", e.job_id),
        TransactionEvidence::InferenceReward(e) => {
            inference_claim_key(&e.job_id, e.batch_start, e.batch_end)
        }
        TransactionEvidence::InferenceRefund(e) => {
            inference_claim_key(&e.job_id, e.batch_start, e.batch_end)
        }
        TransactionEvidence::MembershipChange(_) => format!("membership:{}", tx.transaction_id),
    }
}

/// One batch of one job settles once, in one direction.
///
/// The key is the batch itself, not the assignment id the coordinator handed
/// out. An assignment id is the coordinator's to choose, so keying on it would
/// let the same batch be claimed twice under two names; the batch is the thing
/// the requester actually paid for, so it is the thing that can only go once.
fn inference_claim_key(job_id: &str, batch_start: u32, batch_end: u32) -> String {
    format!("reward:{job_id}:{batch_start}:{batch_end}")
}

/// What the ledger will remember about a job, read straight off the reserve.
fn reservation_of(tx: &LedgerTransaction, e: &InferenceReserveEvidence) -> InferenceReservation {
    InferenceReservation {
        job_id: e.job_id.clone(),
        billing: e.billing.clone(),
        batches: e.batches.clone(),
        requester: e.requester_auth.node_id.clone(),
        reserved_at: tx.created_at,
    }
}

/// The batch a claim names, and what the requester agreed it would cost.
///
/// Price is recomputed from the bill the requester signed rather than read off
/// the claim, so the amount is never the claimant's to choose. A batch that is
/// not in the reservation has no price at all, which is the point: it was
/// never bought.
fn reserved_batch(
    r: &InferenceReservation,
    assignment_id: &str,
    start: u32,
    end: u32,
) -> Result<(PricedBatch, i64)> {
    // Found by the assignment id the job itself determines, then checked
    // against the bounds being claimed - the same order a replay uses, so a
    // claim the validators take cannot be one an auditor later throws out.
    let Some(index) = (0..r.batches.len() as u32)
        .find(|i| hocmesh_protocol::inference_assignment_id(&r.job_id, *i) == assignment_id)
    else {
        bail!("inference claim names an assignment this job never had")
    };
    let b = &r.batches[index as usize];
    if b.batch_start != start || b.batch_end != end {
        bail!("inference claim changes the bounds of the batch it names")
    }
    let price = hocmesh_core::compute::inference_batch_cost_mcu(
        &r.billing.prompt_bytes,
        start,
        end,
        r.billing.max_tokens,
        r.billing.parameter_count,
    );
    Ok((b.clone(), price))
}

/// Checks a whole entry's worth of transactions the way they will be applied.
///
/// Each transaction is judged against the balances its predecessors in the
/// batch left behind. Checking them all against the same opening balances
/// would let two separately affordable settlements jointly overdraw an
/// account, which is exactly what batching would otherwise make possible.
pub fn validate_batch(
    transactions: &[LedgerTransaction],
    previous_hash: &str,
    balance: impl Fn(&str) -> Result<i64>,
    reserved: impl Fn(&str) -> Result<Option<InferenceReservation>>,
    community_issuance_limit_mcu: i64,
) -> Result<()> {
    let mut overlay = std::collections::HashMap::<String, i64>::new();
    // A job reserved earlier in this same entry has to be visible to a claim
    // made later in it, for the same reason balances are: the entry applies as
    // one step, so validating against only what was committed before it would
    // reject a reserve-and-settle pair that is perfectly legal once applied.
    let mut reserved_here = std::collections::HashMap::<String, InferenceReservation>::new();
    for tx in transactions {
        validate_transaction(
            tx,
            previous_hash,
            |a| Ok(balance(a)? + overlay.get(a).copied().unwrap_or(0)),
            |j| match reserved_here.get(j) {
                Some(r) => Ok(Some(r.clone())),
                None => reserved(j),
            },
            community_issuance_limit_mcu,
        )?;
        if let TransactionEvidence::InferenceReserve(e) = &tx.evidence {
            reserved_here.insert(e.job_id.clone(), reservation_of(tx, e));
        }
        for p in &tx.postings {
            *overlay.entry(p.account_id.clone()).or_default() += p.delta_mcu;
        }
    }
    Ok(())
}

/// The reservation an inference reserve establishes, as the store would record it.
pub fn validate_transaction(
    tx: &LedgerTransaction,
    previous_hash: &str,
    balance: impl Fn(&str) -> Result<i64>,
    reserved: impl Fn(&str) -> Result<Option<InferenceReservation>>,
    community_issuance_limit_mcu: i64,
) -> Result<()> {
    if let TransactionEvidence::MembershipChange(e) = &tx.evidence {
        return validate_membership_shape(tx, e);
    }
    if tx.postings.len() < 2 {
        bail!("transaction needs at least two postings")
    }
    let sum: i128 = tx.postings.iter().map(|p| p.delta_mcu as i128).sum();
    if sum != 0 {
        bail!("CU conservation violated: posting sum is {sum}")
    }
    if tx.postings.iter().any(|p| p.delta_mcu == 0) {
        bail!("zero-value postings are not allowed")
    }
    for p in &tx.postings {
        if p.delta_mcu < 0 && p.account_id != COMMUNITY_ISSUANCE_ACCOUNT {
            let b = balance(&p.account_id)?;
            if b < -p.delta_mcu {
                bail!("insufficient balance for {}", p.account_id)
            }
        }
    }
    if let Some(p) = tx
        .postings
        .iter()
        .find(|p| p.account_id == COMMUNITY_ISSUANCE_ACCOUNT && p.delta_mcu < 0)
    {
        let current = balance(COMMUNITY_ISSUANCE_ACCOUNT)?;
        let next = current
            .checked_add(p.delta_mcu)
            .ok_or_else(|| anyhow::anyhow!("community issuance overflow"))?;
        if next < -community_issuance_limit_mcu.abs() {
            bail!("community issuance limit exceeded")
        }
    }
    match (&tx.kind, &tx.evidence) {
        (TransactionKind::JobReserve, TransactionEvidence::JobReserve(e)) => {
            validate_reserve(tx, e)?
        }
        (
            TransactionKind::CommunityReserve,
            TransactionEvidence::CommunityReserve {
                job_id,
                work,
                shards,
            },
        ) => validate_community_reserve(tx, job_id, work, *shards)?,
        (TransactionKind::ProviderReward, TransactionEvidence::ProviderReward(e)) => {
            validate_reward(tx, e, previous_hash)?
        }
        (TransactionKind::JobRefund, TransactionEvidence::JobRefund(e)) => validate_refund(tx, e)?,
        (TransactionKind::InferenceReserve, TransactionEvidence::InferenceReserve(e)) => {
            validate_inference_reserve(tx, e)?
        }
        (TransactionKind::InferenceReward, TransactionEvidence::InferenceReward(e)) => {
            validate_inference_reward(tx, e, reserved(&e.job_id)?.as_ref())?
        }
        (TransactionKind::InferenceRefund, TransactionEvidence::InferenceRefund(e)) => {
            validate_inference_refund(tx, e, reserved(&e.job_id)?.as_ref())?
        }
        _ => bail!("transaction kind/evidence mismatch"),
    }
    Ok(())
}
fn validate_reserve(tx: &LedgerTransaction, e: &JobReserveEvidence) -> Result<()> {
    e.work.validate().map_err(anyhow::Error::msg)?;
    if e.job_id != hocmesh_protocol::job_id_from_auth(&e.requester_auth) {
        bail!("job id is not bound to requester nonce")
    };
    if !(1..=256).contains(&e.shards) {
        bail!("invalid shard count")
    }
    let bh = submit_body_hash(&e.work, e.shards)?;
    verify_auth_signature(
        &e.requester_public_key_b64,
        &e.requester_auth,
        "submit",
        &bh,
    )
    .map_err(anyhow::Error::msg)?;
    let cost: i64 = split_work(&e.work, e.shards)
        .iter()
        .map(work_cost_mcu)
        .sum();
    let escrow = escrow_account(&e.job_id);
    if tx.postings.len() != 2
        || !tx
            .postings
            .iter()
            .any(|p| p.account_id == e.requester_auth.node_id && p.delta_mcu == -cost)
        || !tx
            .postings
            .iter()
            .any(|p| p.account_id == escrow && p.delta_mcu == cost)
    {
        bail!("job reserve postings do not match authorized workload cost")
    }
    Ok(())
}

/// An inference job's escrow is only as sound as the bill behind it.
///
/// Three things have to hold before CU moves: the requester signed this exact
/// bill, the bill prices out to what is being escrowed, and the batches cover
/// the prompts once each. The last one is what makes the escrow drain to zero.
fn validate_inference_reserve(tx: &LedgerTransaction, e: &InferenceReserveEvidence) -> Result<()> {
    if e.job_id != hocmesh_protocol::inference_job_id_from_auth(&e.requester_auth) {
        bail!("inference job id is not bound to requester nonce")
    }
    if !hocmesh_protocol::parameter_count_is_plausible(
        e.billing.parameter_count,
        e.billing.total_size_bytes,
    ) {
        bail!("declared parameter count does not fit the model's own bytes")
    }
    let billing_hash = hocmesh_protocol::inference_billing_hash(&e.billing)?;
    let bh = hocmesh_protocol::inference_submit_body_hash(&billing_hash, &e.settings_digest)?;
    verify_auth_signature(
        &e.requester_public_key_b64,
        &e.requester_auth,
        "submit_inference",
        &bh,
    )
    .map_err(anyhow::Error::msg)?;
    batches_partition_prompts(&e.batches, e.billing.prompt_bytes.len())?;
    let cost = hocmesh_core::compute::inference_cost_mcu(
        &e.billing.prompt_bytes,
        e.billing.max_tokens,
        e.billing.parameter_count,
    );
    if cost > e.billing.max_cost_mcu {
        bail!("inference costs more than the requester authorised")
    }
    let escrow = escrow_account(&e.job_id);
    if tx.postings.len() != 2
        || !tx
            .postings
            .iter()
            .any(|p| p.account_id == e.requester_auth.node_id && p.delta_mcu == -cost)
        || !tx
            .postings
            .iter()
            .any(|p| p.account_id == escrow && p.delta_mcu == cost)
    {
        bail!("inference reserve postings do not match the authorised bill")
    }
    Ok(())
}

/// Batches must cover every prompt exactly once, in order, with no gaps.
///
/// Gaps strand CU in an escrow nobody can claim; overlaps let two providers
/// both get paid for the same prompt. Either one breaks the promise that a
/// job's escrow is exactly the sum of its batches.
fn batches_partition_prompts(batches: &[PricedBatch], prompts: usize) -> Result<()> {
    if batches.is_empty() {
        bail!("an inference job with no batches escrows CU nobody can claim")
    }
    let mut cursor = 0_u32;
    for batch in batches {
        if batch.batch_start != cursor || batch.batch_end <= batch.batch_start {
            bail!("inference batches do not tile the prompt list in order")
        }
        cursor = batch.batch_end;
    }
    if cursor as usize != prompts {
        bail!("inference batches cover {cursor} prompts but the job has {prompts}")
    }
    Ok(())
}

/// A provider claiming one batch of an inference job.
///
/// Inference cannot be re-run by a validator - that is the whole reason it is
/// worth paying for - so what the ledger checks here is the bill and the
/// binding, not the answer. Whether the answer was any good is the requester's
/// judgement, and the requester is the party out of pocket if it was not.
fn validate_inference_reward(
    tx: &LedgerTransaction,
    e: &InferenceRewardEvidence,
    reserved: Option<&InferenceReservation>,
) -> Result<()> {
    let bh = hocmesh_protocol::inference_reward_body_hash(
        &e.assignment_id,
        &e.job_id,
        e.batch_start,
        e.batch_end,
        e.reward_mcu,
        &e.outputs_digest,
    )?;
    verify_auth_signature(
        &e.provider_public_key_b64,
        &e.provider_auth,
        "report_inference",
        &bh,
    )
    .map_err(anyhow::Error::msg)?;
    if e.reward_mcu <= 0 {
        bail!("an inference reward must move CU")
    }
    // A claim is only worth what the requester reserved for it. Without this
    // the ledger is taking the claimant's word for who did the work and what
    // it was worth, which leaves the coordinator as the only thing standing
    // between a provider and somebody else's escrow - and the coordinator is
    // deliberately not the authority for CU.
    let Some(r) = reserved else {
        bail!("an inference reward has no reservation to be paid out of")
    };
    let (batch, price) = reserved_batch(r, &e.assignment_id, e.batch_start, e.batch_end)?;
    if batch.node_id != e.provider_auth.node_id {
        bail!(
            "batch {}..{} was assigned to {}, not to the claimant",
            e.batch_start,
            e.batch_end,
            batch.node_id
        )
    }
    if e.reward_mcu != price {
        bail!("inference reward does not equal the price the requester signed for the batch")
    }
    if r.requester == e.provider_auth.node_id {
        bail!("a requester cannot pay itself for its own inference batch")
    }
    let escrow = escrow_account(&e.job_id);
    if tx.postings.len() != 2
        || !tx
            .postings
            .iter()
            .any(|p| p.account_id == escrow && p.delta_mcu == -e.reward_mcu)
        || !tx
            .postings
            .iter()
            .any(|p| p.account_id == e.provider_auth.node_id && p.delta_mcu == e.reward_mcu)
    {
        bail!("inference reward postings do not match the claimed batch")
    }
    Ok(())
}

/// A requester taking back the escrow on a batch nobody delivered.
///
/// The mirror of the reward, down to the claim key, so the two race for one
/// settlement and exactly one of them wins.
fn validate_inference_refund(
    tx: &LedgerTransaction,
    e: &InferenceRefundEvidence,
    reserved: Option<&InferenceReservation>,
) -> Result<()> {
    let bh = hocmesh_protocol::inference_refund_body_hash(
        &e.assignment_id,
        &e.job_id,
        e.batch_start,
        e.batch_end,
        e.refund_mcu,
    )?;
    verify_auth_signature(
        &e.requester_public_key_b64,
        &e.requester_auth,
        "refund_inference",
        &bh,
    )
    .map_err(anyhow::Error::msg)?;
    if e.refund_mcu <= 0 {
        bail!("an inference refund must move CU")
    }
    // A refund is the other direction out of the same escrow, so it needs the
    // same binding: only the requester who reserved the batch may take it
    // back, and only for what that batch actually cost.
    let Some(r) = reserved else {
        bail!("an inference refund has no reservation to be taken back")
    };
    let (_, price) = reserved_batch(r, &e.assignment_id, e.batch_start, e.batch_end)?;
    if r.requester != e.requester_auth.node_id {
        bail!("only the node that reserved this job may take its escrow back")
    }
    if e.refund_mcu != price {
        bail!("inference refund does not equal the price the requester signed for the batch")
    }
    let escrow = escrow_account(&e.job_id);
    if tx.postings.len() != 2
        || !tx
            .postings
            .iter()
            .any(|p| p.account_id == escrow && p.delta_mcu == -e.refund_mcu)
        || !tx
            .postings
            .iter()
            .any(|p| p.account_id == e.requester_auth.node_id && p.delta_mcu == e.refund_mcu)
    {
        bail!("inference refund postings do not match the reclaimed batch")
    }
    Ok(())
}

/// CU may only be issued for work the ledger can check on its own.
///
/// A requester-funded shard moves CU between two accounts. A provider that
/// cheats there robs the requester, who holds the answer and has every reason
/// to check it. A community-funded shard mints CU against the issuance limit,
/// so work nobody can audit is a mint nobody can question - and that is the
/// one place where "the requester will notice" is not an answer.
///
/// The rule loosens the day the reveal exchange exists, not before.
fn issuable(work: &hocmesh_protocol::WorkSpec) -> Result<()> {
    issuable_class(work.audit_class())
}

fn issuable_class(class: hocmesh_protocol::AuditClass) -> Result<()> {
    if class != hocmesh_protocol::AuditClass::SelfContained {
        bail!("community-funded work must be auditable from the ledger entry")
    }
    Ok(())
}

fn validate_community_reserve(
    tx: &LedgerTransaction,
    job_id: &str,
    work: &hocmesh_protocol::WorkSpec,
    shards: u32,
) -> Result<()> {
    work.validate().map_err(anyhow::Error::msg)?;
    issuable(work)?;
    if !(1..=256).contains(&shards) {
        bail!("invalid community shard count")
    };
    let cost: i64 = split_work(work, shards).iter().map(work_cost_mcu).sum();
    let escrow = escrow_account(job_id);
    if tx.postings.len() != 2
        || !tx
            .postings
            .iter()
            .any(|p| p.account_id == COMMUNITY_ISSUANCE_ACCOUNT && p.delta_mcu == -cost)
        || !tx
            .postings
            .iter()
            .any(|p| p.account_id == escrow && p.delta_mcu == cost)
    {
        bail!("community reserve postings do not match workload cost")
    };
    Ok(())
}
fn validate_reward(
    tx: &LedgerTransaction,
    e: &ProviderRewardEvidence,
    previous_hash: &str,
) -> Result<()> {
    let bh = result_body_hash(
        &e.assignment_id,
        &e.job_id,
        e.shard_index,
        &e.work,
        e.reward_mcu,
        e.system_funded,
        &e.result,
    )?;
    verify_auth_signature(&e.provider_public_key_b64, &e.provider_auth, "result", &bh)
        .map_err(anyhow::Error::msg)?;
    if e.system_funded {
        issuable(&e.work)?;
    }
    if !witnessed(&e.work, &e.result, previous_hash, &tx.transaction_id) {
        bail!("provider result does not verify")
    };
    let reward = work_cost_mcu(&e.work);
    if e.reward_mcu != reward {
        bail!("declared reward does not equal deterministic work cost")
    };
    if e.assignment_id != hocmesh_protocol::assignment_id(&e.job_id, e.shard_index) {
        bail!("assignment id is not deterministic for job/shard")
    };
    let source = escrow_account(&e.job_id);
    if tx.postings.len() != 2
        || !tx
            .postings
            .iter()
            .any(|p| p.account_id == source && p.delta_mcu == -reward)
        || !tx
            .postings
            .iter()
            .any(|p| p.account_id == e.provider_auth.node_id && p.delta_mcu == reward)
    {
        bail!("reward postings do not match verified work")
    }
    Ok(())
}

pub fn validate_historical_transaction(
    tx: &LedgerTransaction,
    previous_hash: &str,
    signatures: &[ValidatorSignature],
    balance: impl Fn(&str) -> Result<i64>,
    community_issuance_limit_mcu: i64,
) -> Result<()> {
    verify_historical_evidence(tx, previous_hash, signatures)?;
    if let TransactionEvidence::MembershipChange(e) = &tx.evidence {
        return validate_membership_shape(tx, e);
    }
    if tx.postings.len() < 2 {
        bail!("transaction needs at least two postings")
    }
    let sum: i128 = tx.postings.iter().map(|p| p.delta_mcu as i128).sum();
    if sum != 0 {
        bail!("CU conservation violated: posting sum is {sum}")
    }
    if tx.postings.iter().any(|p| p.delta_mcu == 0) {
        bail!("zero-value postings are not allowed")
    }
    for p in &tx.postings {
        if p.delta_mcu < 0 && p.account_id != COMMUNITY_ISSUANCE_ACCOUNT {
            let b = balance(&p.account_id)?;
            if b < -p.delta_mcu {
                bail!("insufficient historical balance for {}", p.account_id)
            }
        }
    }
    if let Some(p) = tx
        .postings
        .iter()
        .find(|p| p.account_id == COMMUNITY_ISSUANCE_ACCOUNT && p.delta_mcu < 0)
    {
        let next = balance(COMMUNITY_ISSUANCE_ACCOUNT)?
            .checked_add(p.delta_mcu)
            .ok_or_else(|| anyhow::anyhow!("community issuance overflow"))?;
        if next < -community_issuance_limit_mcu.abs() {
            bail!("community issuance limit exceeded")
        }
    }
    match (&tx.kind, &tx.evidence) {
        (TransactionKind::JobReserve, TransactionEvidence::JobReserve(e)) => {
            if e.job_id != hocmesh_protocol::job_id_from_auth(&e.requester_auth) {
                bail!("historical job id not bound to requester nonce")
            };
            let cost: i64 = split_work(&e.work, e.shards)
                .iter()
                .map(work_cost_mcu)
                .sum();
            let escrow = escrow_account(&e.job_id);
            if tx.postings.len() != 2
                || !tx
                    .postings
                    .iter()
                    .any(|p| p.account_id == e.requester_auth.node_id && p.delta_mcu == -cost)
                || !tx
                    .postings
                    .iter()
                    .any(|p| p.account_id == escrow && p.delta_mcu == cost)
            {
                bail!("historical reserve postings invalid")
            }
        }
        (
            TransactionKind::CommunityReserve,
            TransactionEvidence::CommunityReserve {
                job_id,
                work,
                shards,
            },
        ) => validate_community_reserve(tx, job_id, work, *shards)?,
        (TransactionKind::ProviderReward, TransactionEvidence::ProviderReward(e)) => {
            let reward = work_cost_mcu(&e.work);
            if e.reward_mcu != reward
                || e.assignment_id != hocmesh_protocol::assignment_id(&e.job_id, e.shard_index)
            {
                bail!("historical reward metadata invalid")
            };
            let source = escrow_account(&e.job_id);
            if tx.postings.len() != 2
                || !tx
                    .postings
                    .iter()
                    .any(|p| p.account_id == source && p.delta_mcu == -reward)
                || !tx
                    .postings
                    .iter()
                    .any(|p| p.account_id == e.provider_auth.node_id && p.delta_mcu == reward)
            {
                bail!("historical reward postings invalid")
            }
        }
        (TransactionKind::JobRefund, TransactionEvidence::JobRefund(e)) => {
            let refund = work_cost_mcu(&e.work);
            if e.refund_mcu != refund
                || e.assignment_id != hocmesh_protocol::assignment_id(&e.job_id, e.shard_index)
            {
                bail!("historical refund metadata invalid")
            };
            // Who the escrow returns to is decided by how it was funded, not
            // by whichever account the postings happen to name.
            let payee = refund_payee(e)?;
            let source = escrow_account(&e.job_id);
            if tx.postings.len() != 2
                || !tx
                    .postings
                    .iter()
                    .any(|p| p.account_id == source && p.delta_mcu == -refund)
                || !tx
                    .postings
                    .iter()
                    .any(|p| p.account_id == payee && p.delta_mcu == refund)
            {
                bail!("historical refund postings invalid")
            }
        }
        (TransactionKind::InferenceReserve, TransactionEvidence::InferenceReserve(e)) => {
            // What an old inference reservation can be checked against, years
            // later, with nothing but this entry: the job id is the requester's
            // own nonce, the price is the closed form of the billing it signed,
            // and the batch plan tiles the prompts it paid for.
            if e.job_id != hocmesh_protocol::inference_job_id_from_auth(&e.requester_auth) {
                bail!("historical inference job id is not bound to its requester")
            }
            if !hocmesh_protocol::parameter_count_is_plausible(
                e.billing.parameter_count,
                e.billing.total_size_bytes,
            ) {
                bail!("historical inference billing declares an impossible model")
            }
            let cost = hocmesh_core::compute::inference_cost_mcu(
                &e.billing.prompt_bytes,
                e.billing.max_tokens,
                e.billing.parameter_count,
            );
            if cost > e.billing.max_cost_mcu {
                bail!("historical inference reservation exceeded its own ceiling")
            }
            batches_partition_prompts(&e.batches, e.billing.prompt_bytes.len())?;
            let escrow = escrow_account(&e.job_id);
            if tx.postings.len() != 2
                || !tx
                    .postings
                    .iter()
                    .any(|p| p.account_id == e.requester_auth.node_id && p.delta_mcu == -cost)
                || !tx
                    .postings
                    .iter()
                    .any(|p| p.account_id == escrow && p.delta_mcu == cost)
            {
                bail!("historical inference reserve postings invalid")
            }
        }
        (TransactionKind::InferenceReward, TransactionEvidence::InferenceReward(e)) => {
            // A batch price needs the prompt sizes, which live in the
            // reservation, not here - so the cross-binding is the replay's job.
            // What this entry proves on its own is that the provider signed for
            // this exact batch of this exact job, and that the CU moved from
            // that escrow to that provider and nowhere else.
            if e.job_id == e.provider_auth.node_id {
                bail!("historical inference reward pays a job id, not a node")
            }
            if e.batch_end <= e.batch_start {
                bail!("historical inference reward claims an empty batch")
            }
            if e.reward_mcu <= 0 {
                bail!("historical inference reward claims a non-positive amount")
            }
            let bh = hocmesh_protocol::inference_reward_body_hash(
                &e.assignment_id,
                &e.job_id,
                e.batch_start,
                e.batch_end,
                e.reward_mcu,
                &e.outputs_digest,
            )?;
            verify_auth_signature(
                &e.provider_public_key_b64,
                &e.provider_auth,
                "report_inference",
                &bh,
            )
            .map_err(anyhow::Error::msg)?;
            let source = escrow_account(&e.job_id);
            if tx.postings.len() != 2
                || !tx
                    .postings
                    .iter()
                    .any(|p| p.account_id == source && p.delta_mcu == -e.reward_mcu)
                || !tx
                    .postings
                    .iter()
                    .any(|p| p.account_id == e.provider_auth.node_id && p.delta_mcu == e.reward_mcu)
            {
                bail!("historical inference reward postings invalid")
            }
        }
        (TransactionKind::InferenceRefund, TransactionEvidence::InferenceRefund(e)) => {
            // The mirror image, and the reason escrow is not a one-way valve.
            // The CU goes back to the node that signed for it or nowhere.
            if e.batch_end <= e.batch_start || e.refund_mcu <= 0 {
                bail!("historical inference refund claims an empty or empty-valued batch")
            }
            let bh = hocmesh_protocol::inference_refund_body_hash(
                &e.assignment_id,
                &e.job_id,
                e.batch_start,
                e.batch_end,
                e.refund_mcu,
            )?;
            verify_auth_signature(
                &e.requester_public_key_b64,
                &e.requester_auth,
                "refund_inference",
                &bh,
            )
            .map_err(anyhow::Error::msg)?;
            let source = escrow_account(&e.job_id);
            if tx.postings.len() != 2
                || !tx
                    .postings
                    .iter()
                    .any(|p| p.account_id == source && p.delta_mcu == -e.refund_mcu)
                || !tx.postings.iter().any(|p| {
                    p.account_id == e.requester_auth.node_id && p.delta_mcu == e.refund_mcu
                })
            {
                bail!("historical inference refund postings invalid")
            }
        }
        _ => bail!("historical transaction kind/evidence mismatch"),
    }
    Ok(())
}

pub fn ledger_entry_signing_message(membership_hash: &str, entry_hash: &str) -> String {
    format!("hocmesh-ledger-v1|{}|{}", membership_hash, entry_hash)
}

pub fn verify_validator_signature(
    member: &ValidatorMember,
    message: &str,
    sig_b64: &str,
) -> Result<()> {
    let pk = STANDARD_NO_PAD.decode(&member.public_key_b64)?;
    let pk: [u8; 32] = pk
        .try_into()
        .map_err(|_| anyhow::anyhow!("validator key must be 32 bytes"))?;
    let vk = VerifyingKey::from_bytes(&pk)?;
    let sb = STANDARD_NO_PAD.decode(sig_b64)?;
    let sig = Signature::from_slice(&sb)?;
    vk.verify(message.as_bytes(), &sig)?;
    Ok(())
}
pub fn verify_certificate(cert: &QuorumCertificate, set: &ValidatorSet) -> Result<()> {
    if cert.membership_hash != membership_hash(set)? {
        bail!("membership hash mismatch")
    };
    if cert.entry.transactions.is_empty() {
        bail!("ledger entry carries no transactions")
    };
    // A batch must not settle the same claim twice. Across entries the claim
    // table enforces exactly-once; within one entry it has to be enforced
    // here, before any of the postings are applied.
    let mut seen = std::collections::HashSet::new();
    for t in &cert.entry.transactions {
        let ck = claim_key(t);
        if !seen.insert(ck.clone()) {
            bail!("ledger entry settles claim {ck} twice")
        }
    }
    let th = transactions_hash(&cert.entry.transactions)?;
    if th != cert.entry.transactions_hash {
        bail!("transaction hash mismatch")
    };
    if entry_hash(cert.entry.sequence, &cert.entry.previous_hash, &th)? != cert.entry.entry_hash {
        bail!("entry hash mismatch")
    };
    let mut ids = std::collections::HashSet::new();
    let mut good = 0usize;
    for s in &cert.signatures {
        if !ids.insert(&s.validator_id) {
            continue;
        }
        if let Some(m) = set
            .members
            .iter()
            .find(|m| m.validator_id == s.validator_id)
            && verify_validator_signature(
                m,
                &ledger_entry_signing_message(&cert.membership_hash, &cert.entry.entry_hash),
                &s.signature_b64,
            )
            .is_ok()
        {
            good += 1
        }
    }
    if good < set.threshold {
        bail!(
            "certificate has {good} valid signatures; threshold is {}",
            set.threshold
        )
    }
    Ok(())
}

pub fn checkpoint_signing_message(
    membership_hash: &str,
    sequence: u64,
    entry_hash: &str,
    state_hash: &str,
) -> String {
    format!("hocmesh-checkpoint-v1|{membership_hash}|{sequence}|{entry_hash}|{state_hash}")
}

/// Checks that a quorum really did sign for this state at this height.
///
/// Same shape as `verify_certificate`, and for the same reason: a checkpoint
/// is only worth starting an audit from if enough validators independently
/// staked their key on the state being exactly this.
pub fn verify_checkpoint(cp: &LedgerCheckpoint, set: &ValidatorSet) -> Result<()> {
    if cp.head.membership_hash != membership_hash(set)? {
        bail!("checkpoint membership hash mismatch")
    };
    let message = checkpoint_signing_message(
        &cp.head.membership_hash,
        cp.head.sequence,
        &cp.head.entry_hash,
        &cp.state_hash,
    );
    let mut ids = std::collections::HashSet::new();
    let mut good = 0usize;
    for s in &cp.signatures {
        if !ids.insert(&s.validator_id) {
            continue;
        }
        if let Some(m) = set
            .members
            .iter()
            .find(|m| m.validator_id == s.validator_id)
            && verify_validator_signature(m, &message, &s.signature_b64).is_ok()
        {
            good += 1
        }
    }
    if good < set.threshold {
        bail!(
            "checkpoint has {good} valid signatures; threshold is {}",
            set.threshold
        )
    }
    Ok(())
}

/// A shard's CU has exactly two ways out of escrow: a reward, or this.
///
/// Without this an escrow is a one-way valve. A provider that cheats, or that
/// simply never answers, leaves the requester's CU locked in the job's escrow
/// forever. That punishes the requester for someone else's failure and takes
/// the CU out of circulation for good. Catching a cheat is not a settlement
/// rule until the funds can move afterwards.
///
/// Two things keep the release safe. A refund carries the same claim key as
/// the reward it replaces, so the claims table - not a new rule - makes a
/// shard settle exactly once, as one or the other. And minted CU never
/// refunds to a node: it goes back to the issuance account it was minted
/// against, or "reserve community work, let it fail, keep the CU" would be
/// free minting.
///
/// Whether the window has passed, and whether this really is the job's shard,
/// need the reserve entry. Those live with the history, in `Store::audit` and
/// in the validator's propose path.
fn validate_refund(tx: &LedgerTransaction, e: &JobRefundEvidence) -> Result<()> {
    e.work.validate().map_err(anyhow::Error::msg)?;
    if e.assignment_id != hocmesh_protocol::assignment_id(&e.job_id, e.shard_index) {
        bail!("assignment id is not deterministic for job/shard")
    }
    let refund = work_cost_mcu(&e.work);
    if e.refund_mcu != refund {
        bail!("declared refund does not equal deterministic work cost")
    }
    let payee = refund_payee(e)?;
    let escrow = escrow_account(&e.job_id);
    if tx.postings.len() != 2
        || !tx
            .postings
            .iter()
            .any(|p| p.account_id == escrow && p.delta_mcu == -refund)
        || !tx
            .postings
            .iter()
            .any(|p| p.account_id == payee && p.delta_mcu == refund)
    {
        bail!("job refund postings do not return the shard's escrow to its funder")
    }
    Ok(())
}

/// Who a refund pays, and the proof they are entitled to ask.
///
/// The authorisation is present exactly when there is somebody to pay. A
/// half-filled pair is refused rather than interpreted, so neither branch can
/// be reached with the other one's data.
fn refund_payee(e: &JobRefundEvidence) -> Result<String> {
    match (&e.requester_public_key_b64, &e.requester_auth) {
        (Some(pk), Some(auth)) => {
            if e.system_funded {
                bail!("a community-funded shard has no requester to refund")
            }
            let bh = refund_body_hash(
                &e.assignment_id,
                &e.job_id,
                e.shard_index,
                &e.work,
                e.refund_mcu,
                e.system_funded,
            )?;
            verify_auth_signature(pk, auth, "refund", &bh).map_err(anyhow::Error::msg)?;
            Ok(auth.node_id.clone())
        }
        (None, None) => {
            if !e.system_funded {
                bail!("a requester-funded refund needs the requester's authorisation")
            }
            Ok(COMMUNITY_ISSUANCE_ACCOUNT.to_string())
        }
        _ => bail!("refund carries half a requester authorisation"),
    }
}

/// Audit-time evidence signature validation without rejecting old timestamps.
pub fn verify_historical_evidence(
    tx: &LedgerTransaction,
    previous_hash: &str,
    signatures: &[ValidatorSignature],
) -> Result<()> {
    match &tx.evidence {
        TransactionEvidence::JobReserve(e) => {
            let bh = submit_body_hash(&e.work, e.shards)?;
            verify_auth_signature(
                &e.requester_public_key_b64,
                &e.requester_auth,
                "submit",
                &bh,
            )
            .map_err(anyhow::Error::msg)?
        }
        TransactionEvidence::CommunityReserve { work, shards, .. } => {
            work.validate().map_err(anyhow::Error::msg)?;
            issuable(work)?;
            if !(1..=256).contains(shards) {
                bail!("invalid historical community reservation")
            }
        }
        TransactionEvidence::ProviderReward(e) => {
            let bh = result_body_hash(
                &e.assignment_id,
                &e.job_id,
                e.shard_index,
                &e.work,
                e.reward_mcu,
                e.system_funded,
                &e.result,
            )?;
            verify_auth_signature(&e.provider_public_key_b64, &e.provider_auth, "result", &bh)
                .map_err(anyhow::Error::msg)?;
            if e.system_funded {
                issuable(&e.work)?;
            }
            if !certified_witness(
                &e.work,
                &e.result,
                previous_hash,
                &tx.transaction_id,
                signatures,
            ) {
                bail!("historical work result invalid")
            }
        }
        TransactionEvidence::JobRefund(e) => {
            e.work.validate().map_err(anyhow::Error::msg)?;
            refund_payee(e)?;
        }
        TransactionEvidence::InferenceReserve(e) => {
            let billing_hash = hocmesh_protocol::inference_billing_hash(&e.billing)?;
            let bh =
                hocmesh_protocol::inference_submit_body_hash(&billing_hash, &e.settings_digest)?;
            verify_auth_signature(
                &e.requester_public_key_b64,
                &e.requester_auth,
                "submit_inference",
                &bh,
            )
            .map_err(anyhow::Error::msg)?;
            batches_partition_prompts(&e.batches, e.billing.prompt_bytes.len())?;
        }
        TransactionEvidence::InferenceReward(e) => {
            let bh = hocmesh_protocol::inference_reward_body_hash(
                &e.assignment_id,
                &e.job_id,
                e.batch_start,
                e.batch_end,
                e.reward_mcu,
                &e.outputs_digest,
            )?;
            verify_auth_signature(
                &e.provider_public_key_b64,
                &e.provider_auth,
                "report_inference",
                &bh,
            )
            .map_err(anyhow::Error::msg)?;
        }
        TransactionEvidence::InferenceRefund(e) => {
            let bh = hocmesh_protocol::inference_refund_body_hash(
                &e.assignment_id,
                &e.job_id,
                e.batch_start,
                e.batch_end,
                e.refund_mcu,
            )?;
            verify_auth_signature(
                &e.requester_public_key_b64,
                &e.requester_auth,
                "refund_inference",
                &bh,
            )
            .map_err(anyhow::Error::msg)?;
        }
        TransactionEvidence::MembershipChange(e) => {
            if e.member.validator_id.is_empty() || e.member.public_key_b64.is_empty() {
                bail!("membership change names no validator")
            }
            if e.vouches.is_empty() {
                bail!("membership change carries no vouches")
            }
        }
    };
    Ok(())
}

/// Check a historical result the cheap way.
///
/// Every validator runs this on every entry, which is affordable precisely
/// because it is a witness check and not a recomputation: a validator spends a
/// few percent of what the worker spent instead of all of it. That single
/// change is what stops `V` validators from costing the network `V` times the
/// work they are validating.
///
/// The nonce is the one the coordinator recorded, so this replays the audit
/// that actually happened rather than inventing a fresh one. A workload with
/// no witness falls back to recomputation: sound, but expensive enough to be a
/// standing argument for designing workloads that can be checked cheaply.
/// Check a reward entry's work against the challenge its chain position fixes.
///
/// The nonce is derived here, not read from the evidence, so a coordinator that
/// colludes with a provider cannot hand the pair a challenge of its choosing.
fn witnessed(
    work: &hocmesh_protocol::WorkSpec,
    result: &hocmesh_protocol::WorkResult,
    previous_hash: &str,
    transaction_id: &str,
) -> bool {
    let nonce = AuditNonce::for_entry(previous_hash, transaction_id);
    let verdict = verify::witness_check(work, result, nonce);
    if verdict == verify::Verdict::Inconclusive {
        return verify::adjudicate(work, result).is_accepted();
    }
    verdict.is_accepted()
}

/// The authoritative audit: the challenge comes from the quorum's signatures,
/// so neither the provider nor the coordinator had any say in choosing it.
fn certified_witness(
    work: &hocmesh_protocol::WorkSpec,
    result: &hocmesh_protocol::WorkResult,
    previous_hash: &str,
    transaction_id: &str,
    signatures: &[ValidatorSignature],
) -> bool {
    let beacon: Vec<&str> = signatures
        .iter()
        .map(|s| s.signature_b64.as_str())
        .collect();
    let nonce = AuditNonce::for_certified_entry(previous_hash, transaction_id, &beacon);
    let verdict = verify::witness_check(work, result, nonce);
    if verdict == verify::Verdict::Inconclusive {
        return verify::adjudicate(work, result).is_accepted();
    }
    verdict.is_accepted()
}

/// A membership change settles nothing, so it must move no CU at all.
///
/// Written as a requirement rather than an exemption. If an admission were
/// allowed to carry postings it would be the one transaction kind able to
/// move CU while presenting evidence that says nothing about balances.
fn validate_membership_shape(tx: &LedgerTransaction, e: &MembershipChangeEvidence) -> Result<()> {
    if !matches!(tx.kind, TransactionKind::MembershipChange) {
        bail!("transaction kind/evidence mismatch")
    }
    if !tx.postings.is_empty() {
        bail!("a membership change must move no CU")
    }
    if e.threshold == 0 {
        bail!("membership change sets a zero threshold")
    }
    Ok(())
}

/// What a sitting validator signs to sponsor a change to the set.
///
/// The set it was signed against is part of the message, so a vouch
/// collected for one transition cannot be re-presented against a set that
/// has moved on since.
pub fn vouch_signing_message(
    previous_set_hash: &str,
    action: MembershipAction,
    member: &ValidatorMember,
    resulting_set_hash: &str,
) -> String {
    let verb = match action {
        MembershipAction::Join => "join",
        MembershipAction::Leave => "leave",
    };
    format!(
        "hocmesh-vouch-v1|{previous_set_hash}|{verb}|{}|{}|{resulting_set_hash}",
        member.validator_id, member.public_key_b64
    )
}

/// Works out the set a change produces, refusing one that makes no sense.
pub fn membership_result(
    set: &ValidatorSet,
    action: MembershipAction,
    member: &ValidatorMember,
    threshold: usize,
) -> Result<ValidatorSet> {
    let mut next = set.clone();
    match action {
        MembershipAction::Join => {
            if next
                .members
                .iter()
                .any(|m| m.validator_id == member.validator_id)
            {
                bail!("{} is already a validator", member.validator_id)
            }
            next.members.push(member.clone());
        }
        MembershipAction::Leave => {
            let Some(pos) = next
                .members
                .iter()
                .position(|m| m.validator_id == member.validator_id)
            else {
                bail!("{} is not a validator", member.validator_id)
            };
            if next.members[pos] != *member {
                bail!(
                    "membership change describes {} differently from the sitting set",
                    member.validator_id
                )
            }
            next.members.remove(pos);
        }
    }
    next.threshold = threshold;
    validate_validator_set(&next)?;
    Ok(next)
}

/// Applies a change and checks it produces the set it says it will.
///
/// The set hash is re-derived rather than trusted, because it is what every
/// vouch was signed over: if the evidence could claim one set and produce
/// another, a sponsor's signature would authorise something it never saw.
pub fn apply_membership_change(
    set: &ValidatorSet,
    e: &MembershipChangeEvidence,
) -> Result<ValidatorSet> {
    let next = membership_result(set, e.action, &e.member, e.threshold)?;
    let produced = membership_hash(&next)?;
    if produced != e.resulting_set_hash {
        bail!(
            "membership change claims it produces set {} but produces {produced}",
            e.resulting_set_hash
        )
    }
    Ok(next)
}

/// Checks that enough sitting validators put their names to a change.
///
/// The bar is the set's own consensus threshold, deliberately. Eviction that
/// is easier than agreement would itself be the attack: a minority able to
/// vote out the majority captures the quorum without ever holding it. The
/// cost is that a set which has already lost the ability to certify entries
/// cannot repair itself either, which is the same liveness limit the ledger
/// already has rather than a new one.
pub fn verify_membership_change(
    set: &ValidatorSet,
    e: &MembershipChangeEvidence,
) -> Result<ValidatorSet> {
    let previous = membership_hash(set)?;
    let next = apply_membership_change(set, e)?;
    let message = vouch_signing_message(&previous, e.action, &e.member, &e.resulting_set_hash);
    let mut seen = std::collections::HashSet::new();
    let mut good = 0usize;
    for v in &e.vouches {
        if !seen.insert(&v.validator_id) {
            continue;
        }
        if let Some(m) = set
            .members
            .iter()
            .find(|m| m.validator_id == v.validator_id)
            && verify_validator_signature(m, &message, &v.signature_b64).is_ok()
        {
            good += 1
        }
    }
    if good < set.threshold {
        bail!(
            "membership change carries {good} valid vouches from the sitting set; {} are required",
            set.threshold
        )
    }
    Ok(next)
}
#[cfg(test)]
mod tests {
    use super::*;

    /// The four-argument shape the tests that are not about inference use.
    ///
    /// A job nobody ever reserved is the honest default for them: it is what
    /// the ledger would really see, and it keeps the inference binding tested
    /// where it belongs rather than restated in every unrelated case.
    fn validate_transaction(
        tx: &LedgerTransaction,
        previous_hash: &str,
        balance: impl Fn(&str) -> Result<i64>,
        community_issuance_limit_mcu: i64,
    ) -> Result<()> {
        super::validate_transaction(
            tx,
            previous_hash,
            balance,
            |_| Ok(None),
            community_issuance_limit_mcu,
        )
    }

    /// The same, for whole entries.
    fn validate_batch(
        transactions: &[LedgerTransaction],
        previous_hash: &str,
        balance: impl Fn(&str) -> Result<i64>,
        community_issuance_limit_mcu: i64,
    ) -> Result<()> {
        super::validate_batch(
            transactions,
            previous_hash,
            balance,
            |_| Ok(None),
            community_issuance_limit_mcu,
        )
    }

    /// Stands in for the ledger head an entry chains onto. Honest work has to
    /// pass whatever challenge this produces, so its value is arbitrary.
    const TEST_PREVIOUS_HASH: &str =
        "0f8b1c7d2e3a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f9012345678";
    use hocmesh_core::identity::NodeIdentity;
    use hocmesh_protocol::{
        AuthProof, InferenceBilling, PricedBatch, WorkResult, WorkSpec, canonical_auth_message,
    };
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn community_reserve_conserves_credit() {
        let work = WorkSpec::PrimeCount { start: 2, end: 102 };
        let shards = 2;
        let cost: i64 = split_work(&work, shards).iter().map(work_cost_mcu).sum();
        let tx = LedgerTransaction {
            transaction_id: "t1".into(),
            kind: TransactionKind::CommunityReserve,
            postings: vec![
                Posting {
                    account_id: COMMUNITY_ISSUANCE_ACCOUNT.into(),
                    delta_mcu: -cost,
                },
                Posting {
                    account_id: escrow_account("job_test"),
                    delta_mcu: cost,
                },
            ],
            evidence: TransactionEvidence::CommunityReserve {
                job_id: "job_test".into(),
                work,
                shards,
            },
            created_at: 1,
        };
        validate_transaction(&tx, TEST_PREVIOUS_HASH, |_| Ok(0), 1_000_000).unwrap();
    }

    #[test]
    fn conservation_violation_is_rejected() {
        let work = WorkSpec::PrimeCount { start: 2, end: 102 };
        let tx = LedgerTransaction {
            transaction_id: "t2".into(),
            kind: TransactionKind::CommunityReserve,
            postings: vec![
                Posting {
                    account_id: COMMUNITY_ISSUANCE_ACCOUNT.into(),
                    delta_mcu: -10,
                },
                Posting {
                    account_id: escrow_account("job_bad"),
                    delta_mcu: 9,
                },
            ],
            evidence: TransactionEvidence::CommunityReserve {
                job_id: "job_bad".into(),
                work,
                shards: 1,
            },
            created_at: 1,
        };
        assert!(validate_transaction(&tx, TEST_PREVIOUS_HASH, |_| Ok(0), 1_000_000).is_err());
    }

    #[test]
    fn issuance_limit_is_enforced() {
        let work = WorkSpec::PrimeCount {
            start: 2,
            end: 10_002,
        };
        let shards = 1;
        let cost: i64 = split_work(&work, shards).iter().map(work_cost_mcu).sum();
        let tx = LedgerTransaction {
            transaction_id: "t3".into(),
            kind: TransactionKind::CommunityReserve,
            postings: vec![
                Posting {
                    account_id: COMMUNITY_ISSUANCE_ACCOUNT.into(),
                    delta_mcu: -cost,
                },
                Posting {
                    account_id: escrow_account("job_limit"),
                    delta_mcu: cost,
                },
            ],
            evidence: TransactionEvidence::CommunityReserve {
                job_id: "job_limit".into(),
                work,
                shards,
            },
            created_at: 1,
        };
        assert!(validate_transaction(&tx, TEST_PREVIOUS_HASH, |_| Ok(0), 1).is_err());
    }

    #[test]
    fn forged_requester_signature_is_rejected() {
        let requester = test_identity("forged-requester");
        let attacker = test_identity("forged-attacker");
        let work = WorkSpec::PrimeCount { start: 2, end: 20 };
        let attacker_auth = attacker.auth("submit", &submit_body_hash(&work, 1).unwrap());
        let tx = reserve_from_auth(
            "forged-job",
            &requester.public_key_b64(),
            attacker_auth,
            &work,
            1,
        );

        assert!(validate_transaction(&tx, TEST_PREVIOUS_HASH, |_| Ok(10_000), 1_000_000).is_err());
    }

    #[test]
    fn old_historical_requester_signature_remains_auditable() {
        let requester = test_identity("old-requester");
        let work = WorkSpec::PrimeCount { start: 2, end: 20 };
        let body_hash = submit_body_hash(&work, 1).unwrap();
        let old_auth = auth_at(&requester, "submit", &body_hash, 1);
        let tx = reserve_from_auth("old-job", &requester.public_key_b64(), old_auth, &work, 1);

        validate_historical_transaction(&tx, TEST_PREVIOUS_HASH, &[], |_| Ok(10_000), 1_000_000)
            .unwrap();
    }

    #[test]
    fn provider_reward_signature_binds_exact_metadata() {
        let provider = test_identity("provider-metadata");
        let original = WorkSpec::PrimeCount { start: 2, end: 20 };
        let changed = WorkSpec::PrimeCount { start: 2, end: 30 };
        let mut tx = signed_reward("job-metadata", 0, &provider, &original, true);
        if let TransactionEvidence::ProviderReward(e) = &mut tx.evidence {
            e.work = changed;
        }

        assert!(validate_transaction(&tx, TEST_PREVIOUS_HASH, |_| Ok(10_000), 1_000_000).is_err());
    }

    #[test]
    fn provider_reward_rejects_changed_reward_and_assignment_id() {
        let provider = test_identity("provider-reward");
        let work = WorkSpec::PrimeCount { start: 2, end: 20 };
        let mut changed_reward = signed_reward("job-reward", 0, &provider, &work, true);
        if let TransactionEvidence::ProviderReward(e) = &mut changed_reward.evidence {
            e.reward_mcu += 1;
        }
        assert!(
            validate_transaction(
                &changed_reward,
                TEST_PREVIOUS_HASH,
                |_| Ok(10_000),
                1_000_000
            )
            .is_err()
        );

        let mut changed_assignment = signed_reward("job-assignment", 0, &provider, &work, true);
        if let TransactionEvidence::ProviderReward(e) = &mut changed_assignment.evidence {
            e.assignment_id = "asg_not_the_deterministic_id".into();
        }
        assert!(
            validate_transaction(
                &changed_assignment,
                TEST_PREVIOUS_HASH,
                |_| Ok(10_000),
                1_000_000
            )
            .is_err()
        );
    }

    #[test]
    fn duplicate_ledger_claim_is_rejected_on_apply() {
        let set = validator_set("duplicate-claim");
        let mut store = crate::store::LedgerStore::open(":memory:").unwrap();
        let first = certified_community_reserve(&set, "job-dupe", 1, "GENESIS");
        store.apply(&first, &set.set).unwrap();
        let second = certified_community_reserve(&set, "job-dupe", 2, &first.entry.entry_hash);

        assert!(store.apply(&second, &set.set).is_err());
    }

    #[test]
    fn tampered_certificate_entry_is_rejected() {
        let set = validator_set("tampered-cert");
        let mut cert = certified_community_reserve(&set, "job-cert", 1, "GENESIS");
        verify_certificate(&cert, &set.set).unwrap();
        cert.entry.transactions[0].postings[1].delta_mcu += 1;

        assert!(verify_certificate(&cert, &set.set).is_err());
    }

    struct TestValidatorSet {
        set: ValidatorSet,
        identities: Vec<NodeIdentity>,
        _dir: PathBuf,
    }

    fn reserve_from_auth(
        job_suffix: &str,
        requester_public_key_b64: &str,
        auth: AuthProof,
        work: &WorkSpec,
        shards: u32,
    ) -> LedgerTransaction {
        let job_id = hocmesh_protocol::job_id_from_auth(&auth);
        let cost: i64 = split_work(work, shards).iter().map(work_cost_mcu).sum();
        LedgerTransaction {
            transaction_id: format!("reserve-{job_suffix}"),
            kind: TransactionKind::JobReserve,
            postings: vec![
                Posting {
                    account_id: auth.node_id.clone(),
                    delta_mcu: -cost,
                },
                Posting {
                    account_id: escrow_account(&job_id),
                    delta_mcu: cost,
                },
            ],
            evidence: TransactionEvidence::JobReserve(JobReserveEvidence {
                job_id,
                requester_public_key_b64: requester_public_key_b64.into(),
                requester_auth: auth,
                work: work.clone(),
                shards,
            }),
            created_at: 1,
        }
    }

    fn signed_reward(
        job_id: &str,
        shard_index: u32,
        provider: &NodeIdentity,
        work: &WorkSpec,
        system_funded: bool,
    ) -> LedgerTransaction {
        let result = hocmesh_core::compute::execute_work(work);
        let reward_mcu = work_cost_mcu(work);
        let assignment_id = hocmesh_protocol::assignment_id(job_id, shard_index);
        let body_hash = result_body_hash(
            &assignment_id,
            job_id,
            shard_index,
            work,
            reward_mcu,
            system_funded,
            &result,
        )
        .unwrap();
        let auth = provider.auth("result", &body_hash);
        LedgerTransaction {
            transaction_id: format!("reward-{assignment_id}"),
            kind: TransactionKind::ProviderReward,
            postings: vec![
                Posting {
                    account_id: escrow_account(job_id),
                    delta_mcu: -reward_mcu,
                },
                Posting {
                    account_id: auth.node_id.clone(),
                    delta_mcu: reward_mcu,
                },
            ],
            evidence: TransactionEvidence::ProviderReward(ProviderRewardEvidence {
                job_id: job_id.into(),
                assignment_id,
                shard_index,
                reward_mcu,
                provider_public_key_b64: provider.public_key_b64(),
                provider_auth: auth,
                work: work.clone(),
                result,
                system_funded,
                provisional_audit_nonce: 0,
            }),
            created_at: 1,
        }
    }

    fn certified_community_reserve(
        validators: &TestValidatorSet,
        job_id: &str,
        sequence: u64,
        previous_hash: &str,
    ) -> QuorumCertificate {
        let work = WorkSpec::PrimeCount { start: 2, end: 20 };
        let shards = 1;
        let cost: i64 = split_work(&work, shards).iter().map(work_cost_mcu).sum();
        let tx = LedgerTransaction {
            transaction_id: format!("community-{job_id}-{sequence}"),
            kind: TransactionKind::CommunityReserve,
            postings: vec![
                Posting {
                    account_id: COMMUNITY_ISSUANCE_ACCOUNT.into(),
                    delta_mcu: -cost,
                },
                Posting {
                    account_id: escrow_account(job_id),
                    delta_mcu: cost,
                },
            ],
            evidence: TransactionEvidence::CommunityReserve {
                job_id: job_id.into(),
                work,
                shards,
            },
            created_at: 1,
        };
        certify(validators, tx, sequence, previous_hash)
    }
    /// Wraps any transaction in a quorum certificate. Every stateful test needs
    /// this, so it lives apart from the reservation helper that first grew it.
    fn certify(
        validators: &TestValidatorSet,
        tx: LedgerTransaction,
        sequence: u64,
        previous_hash: &str,
    ) -> QuorumCertificate {
        certify_batch(validators, vec![tx], sequence, previous_hash)
    }

    /// Certifies several transactions into a single entry, the way the batching
    /// proposer does.
    fn certify_batch(
        validators: &TestValidatorSet,
        transactions: Vec<LedgerTransaction>,
        sequence: u64,
        previous_hash: &str,
    ) -> QuorumCertificate {
        let entry = build_entry(sequence, previous_hash.into(), transactions).unwrap();
        let membership_hash = membership_hash(&validators.set).unwrap();
        let message = ledger_entry_signing_message(&membership_hash, &entry.entry_hash);
        let signatures = validators
            .identities
            .iter()
            .take(validators.set.threshold)
            .map(|identity| ValidatorSignature {
                validator_id: identity.node_id(),
                signature_b64: identity.sign_bytes_b64(message.as_bytes()),
            })
            .collect();
        QuorumCertificate {
            entry,
            membership_hash,
            signatures,
        }
    }

    /// A refund of one shard's escrow. `requester` signs for paid work; for
    /// community work it is absent, because minted CU goes back to the account
    /// it was minted against and nobody receives it personally.
    fn refund_tx(
        job_id: &str,
        shard_index: u32,
        work: &WorkSpec,
        system_funded: bool,
        requester: Option<&NodeIdentity>,
        created_at: i64,
    ) -> LedgerTransaction {
        let refund_mcu = work_cost_mcu(work);
        let assignment_id = hocmesh_protocol::assignment_id(job_id, shard_index);
        let auth = requester.map(|identity| {
            let body_hash = refund_body_hash(
                &assignment_id,
                job_id,
                shard_index,
                work,
                refund_mcu,
                system_funded,
            )
            .unwrap();
            identity.auth("refund", &body_hash)
        });
        let payee = match &auth {
            Some(a) => a.node_id.clone(),
            None => COMMUNITY_ISSUANCE_ACCOUNT.into(),
        };
        LedgerTransaction {
            transaction_id: format!("refund-{assignment_id}"),
            kind: TransactionKind::JobRefund,
            postings: vec![
                Posting {
                    account_id: escrow_account(job_id),
                    delta_mcu: -refund_mcu,
                },
                Posting {
                    account_id: payee,
                    delta_mcu: refund_mcu,
                },
            ],
            evidence: TransactionEvidence::JobRefund(JobRefundEvidence {
                job_id: job_id.into(),
                assignment_id,
                shard_index,
                refund_mcu,
                work: work.clone(),
                system_funded,
                requester_public_key_b64: requester.map(|i| i.public_key_b64()),
                requester_auth: auth,
            }),
            created_at,
        }
    }

    fn validator_set(name: &str) -> TestValidatorSet {
        let dir = test_dir(name);
        let identities: Vec<_> = (0..4)
            .map(|index| NodeIdentity::load_or_create(&dir.join(format!("v{index}"))).unwrap())
            .collect();
        let members = identities
            .iter()
            .enumerate()
            .map(|(index, identity)| ValidatorMember {
                validator_id: identity.node_id(),
                url: format!("http://127.0.0.1:{}", 9100 + index),
                public_key_b64: identity.public_key_b64(),
            })
            .collect();
        TestValidatorSet {
            set: ValidatorSet {
                threshold: 3,
                community_issuance_limit_mcu: 1_000_000,
                members,
            },
            identities,
            _dir: dir,
        }
    }

    fn auth_at(
        identity: &NodeIdentity,
        action: &str,
        body_hash: &str,
        timestamp: i64,
    ) -> AuthProof {
        let node_id = identity.node_id();
        let nonce_b64 = format!("nonce-{timestamp}-1234567890");
        let message = canonical_auth_message(action, &node_id, timestamp, &nonce_b64, body_hash);
        AuthProof {
            node_id,
            timestamp,
            nonce_b64,
            signature_b64: identity.sign_bytes_b64(message.as_bytes()),
        }
    }

    fn test_identity(name: &str) -> NodeIdentity {
        NodeIdentity::load_or_create(&test_dir(name)).unwrap()
    }

    fn test_dir(name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("hocmesh-ledger-test-{name}-{suffix}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    /// A result that skips every bucket and fills it from its neighbour. The
    /// total still adds up exactly, so the free arithmetic check finds nothing
    /// and only a recount can tell it apart from real work.
    fn rotated(work: &WorkSpec) -> WorkResult {
        let WorkResult::PrimeCount { bucket_counts, .. } =
            hocmesh_core::compute::execute_work(work)
        else {
            unreachable!("prime work returns prime results")
        };
        let width = bucket_counts.len();
        let counts: Vec<u64> = (0..width).map(|i| bucket_counts[(i + 1) % width]).collect();
        WorkResult::PrimeCount {
            count: counts.iter().sum(),
            bucket_counts: counts,
            duration_ms: 0,
        }
    }

    /// Rebuilds a reward around a fabricated result, re-signed by the same
    /// provider, so the only thing wrong with the entry is the work itself.
    fn resigned(
        tx: &LedgerTransaction,
        provider: &NodeIdentity,
        result: WorkResult,
    ) -> LedgerTransaction {
        let mut out = tx.clone();
        let TransactionEvidence::ProviderReward(e) = &mut out.evidence else {
            unreachable!("this helper rewrites provider rewards")
        };
        e.result = result;
        let body_hash = result_body_hash(
            &e.assignment_id,
            &e.job_id,
            e.shard_index,
            &e.work,
            e.reward_mcu,
            e.system_funded,
            &e.result,
        )
        .unwrap();
        e.provider_auth = provider.auth("result", &body_hash);
        out
    }

    /// Stamps the nonce a coordinator claims it audited with.
    fn stamp_nonce(tx: &mut LedgerTransaction, nonce: u64) {
        let TransactionEvidence::ProviderReward(e) = &mut tx.evidence else {
            unreachable!("this helper rewrites provider rewards")
        };
        e.provisional_audit_nonce = nonce;
    }

    fn test_quorum() -> Vec<ValidatorSignature> {
        ["alpha", "bravo", "charlie"]
            .iter()
            .map(|v| ValidatorSignature {
                validator_id: (*v).to_string(),
                signature_b64: format!("sig-{v}"),
            })
            .collect()
    }

    /// A coordinator colluding with a provider gets no say in the audit. It can
    /// stamp any nonce it likes on the entry - including one hand-picked out of
    /// hundreds of thousands because it audits nothing but honest buckets. Even
    /// so, validators still draw their own challenge from the quorum that signed
    /// the entry. The stamp is an audit trail, never an authority.
    #[test]
    fn a_coordinator_chosen_nonce_cannot_excuse_a_lazy_result() {
        let provider = test_identity("provider-collusion");
        let work = WorkSpec::PrimeCount {
            start: 1,
            end: 20_000,
        };
        let honest = signed_reward("job-collusion", 0, &provider, &work, true);
        let lazy = resigned(&honest, &provider, rotated(&work));
        let quorum = test_quorum();
        let TransactionEvidence::ProviderReward(e) = &lazy.evidence else {
            unreachable!("the helper built a provider reward")
        };
        let flattering = (0..400_000u64)
            .find(|n| {
                verify::witness_check(&e.work, &e.result, AuditNonce::replay(*n)).is_accepted()
            })
            .expect("a colluding coordinator can always find a nonce that flatters");
        let mut colluded = lazy.clone();
        stamp_nonce(&mut colluded, flattering);
        let verdict = verify_historical_evidence(&colluded, TEST_PREVIOUS_HASH, &quorum);
        assert!(
            verdict.is_err(),
            "the coordinator's own nonce must carry no weight at settlement"
        );

        // The same stamp on honest work is harmless: the field is an audit
        // trail of what the coordinator claims it checked, nothing more.
        let mut stamped = honest.clone();
        stamp_nonce(&mut stamped, flattering.wrapping_add(1));
        verify_historical_evidence(&stamped, TEST_PREVIOUS_HASH, &quorum).unwrap();
    }

    /// Both layers must reject fabricated work on their own: the propose-time
    /// check a validator runs before it votes, and the apply-time beacon it
    /// runs when the signed certificate comes back to be applied.
    #[test]
    fn both_the_propose_and_apply_checks_reject_fabricated_work() {
        let provider = test_identity("provider-layers");
        let work = WorkSpec::PrimeCount {
            start: 1,
            end: 20_000,
        };
        let honest = signed_reward("job-layers", 0, &provider, &work, true);
        let lazy = resigned(&honest, &provider, rotated(&work));
        let propose = validate_transaction(&lazy, TEST_PREVIOUS_HASH, |_| Ok(10_000), 1_000_000);
        assert!(
            propose.is_err(),
            "a validator must not vote for fabricated work"
        );
        let apply = verify_historical_evidence(&lazy, TEST_PREVIOUS_HASH, &test_quorum());
        assert!(apply.is_err(), "the beacon must catch what the vote missed");
    }

    /// Issuance is the only place a cheat creates CU rather than moving it, so
    /// a workload the ledger cannot audit on its own must never reach it.
    #[test]
    fn unauditable_work_can_never_be_paid_for_with_issued_cu() {
        use hocmesh_protocol::AuditClass;
        assert!(issuable_class(AuditClass::SelfContained).is_ok());
        let refused = issuable_class(AuditClass::RevealRequired);
        assert!(
            refused.is_err(),
            "a shard nobody can check must not mint CU"
        );
    }

    /// Both shipping workloads answer in a few dozen integers, so both stay on
    /// the issuable side. If this ever fails, community funding broke.
    #[test]
    fn todays_workloads_are_auditable_from_the_entry() {
        use hocmesh_protocol::AuditClass;
        let prime = WorkSpec::PrimeCount {
            start: 0,
            end: 100_000,
        };
        let matmul = WorkSpec::MatrixMultiply {
            seed_a: 1,
            seed_b: 2,
            dim: 64,
            row_start: 0,
            row_end: 64,
        };
        for work in [prime, matmul] {
            assert_eq!(work.audit_class(), AuditClass::SelfContained);
            assert!(issuable(&work).is_ok());
        }
    }

    /// Escrow that was minted has no requester to return to. Paying it to a
    /// node would make "reserve community work, let it fail, keep the CU" a
    /// free mint, so it goes back to the account it was issued against.
    #[test]
    fn minted_escrow_can_only_be_refunded_to_the_issuance_account() {
        let work = WorkSpec::PrimeCount { start: 2, end: 20 };
        let honest = refund_tx("job-minted", 0, &work, true, None, 100);
        validate_transaction(&honest, TEST_PREVIOUS_HASH, |_| Ok(10_000), 1_000_000).unwrap();

        let thief = test_identity("refund-thief");
        let stolen = refund_tx("job-minted", 0, &work, true, Some(&thief), 100);
        let refused = refused(validate_transaction(
            &stolen,
            TEST_PREVIOUS_HASH,
            |_| Ok(10_000),
            1_000_000,
        ));
        assert!(
            refused.contains("no requester to refund"),
            "a signature on minted escrow must not redirect it to a node: {refused}"
        );
    }

    /// An unsigned refund pays the issuance account, so letting one settle
    /// paid escrow would mint CU out of a requester's own money.
    #[test]
    fn a_refund_of_paid_escrow_without_the_requesters_signature_is_refused() {
        let work = WorkSpec::PrimeCount { start: 2, end: 20 };
        let unsigned = refund_tx("job-paid", 0, &work, false, None, 100);
        let refused = refused(validate_transaction(
            &unsigned,
            TEST_PREVIOUS_HASH,
            |_| Ok(10_000),
            1_000_000,
        ));
        assert!(
            refused.contains("needs the requester's authorisation"),
            "escrow somebody paid for must not be returned to issuance: {refused}"
        );
    }

    /// The refund is the shard's price. Signing honestly for a larger number
    /// still has to fail, or a requester could withdraw more than they put in.
    #[test]
    fn a_refund_cannot_take_more_than_the_shard_reserved() {
        let requester = test_identity("refund-greedy");
        let work = WorkSpec::PrimeCount { start: 2, end: 20 };
        let mut tx = refund_tx("job-greedy", 0, &work, false, Some(&requester), 100);
        tx.postings[0].delta_mcu -= 1;
        tx.postings[1].delta_mcu += 1;
        let TransactionEvidence::JobRefund(e) = &mut tx.evidence else {
            unreachable!("refund_tx builds a refund")
        };
        e.refund_mcu += 1;
        let body = refund_body_hash(
            &e.assignment_id,
            &e.job_id,
            e.shard_index,
            &e.work,
            e.refund_mcu,
            e.system_funded,
        )
        .unwrap();
        e.requester_auth = Some(requester.auth("refund", &body));
        let refused = refused(validate_transaction(
            &tx,
            TEST_PREVIOUS_HASH,
            |_| Ok(10_000),
            1_000_000,
        ));
        assert!(
            refused.contains("declared refund does not equal deterministic work cost"),
            "a valid signature over the wrong amount is still the wrong amount: {refused}"
        );
    }

    /// Reserves one community shard, settles it the given way, and replays the
    /// whole chain. Replay is where the stateful rules live, so this is what
    /// says whether a settlement is really allowed to stand.
    fn settle_and_replay(
        set: &TestValidatorSet,
        job_id: &str,
        settlement: LedgerTransaction,
    ) -> Result<crate::store::LedgerStore> {
        let mut store = crate::store::LedgerStore::open(":memory:").unwrap();
        let reserve = certified_community_reserve(set, job_id, 1, "GENESIS");
        store.apply(&reserve, &set.set).unwrap();
        let cert = certify(set, settlement, 2, &reserve.entry.entry_hash);
        store.apply(&cert, &set.set)?;
        store.audit(&set.set)?;
        Ok(store)
    }

    /// The reason a settlement was refused matters as much as the refusal, so
    /// every negative case here reads the message rather than trusting `is_err`.
    fn refused<T>(outcome: Result<T>) -> String {
        outcome.map(|_| ()).unwrap_err().to_string()
    }
    /// One shard settles once. If both windows were open at the same moment
    /// the requester and the provider would race for the same escrow, and the
    /// ledger would have to pick a winner after the fact.
    #[test]
    fn a_reward_and_a_refund_never_have_the_same_shard_open_at_once() {
        let set = validator_set("refund-window");
        let work = WorkSpec::PrimeCount { start: 2, end: 20 };
        let provider = test_identity("refund-window-provider");
        // The reservation is stamped at 1, so the window closes here.
        let deadline = 1 + hocmesh_protocol::SETTLEMENT_WINDOW_SECS;

        let early = refused(settle_and_replay(
            &set,
            "early",
            refund_tx("early", 0, &work, true, None, deadline),
        ));
        assert!(
            early.contains("refund inside the shard's settlement window"),
            "a refund must not open while the provider still has time to answer: {early}"
        );
        let late = refund_tx("late", 0, &work, true, None, deadline + 1);
        settle_and_replay(&set, "late", late).expect("a refund one second past the window stands");

        let mut ontime = signed_reward("ontime", 0, &provider, &work, true);
        ontime.created_at = deadline;
        settle_and_replay(&set, "ontime", ontime).expect("a reward inside the window stands");

        let mut stale = signed_reward("stale", 0, &provider, &work, true);
        stale.created_at = deadline + 1;
        let stale = refused(settle_and_replay(&set, "stale", stale));
        assert!(
            stale.contains("reward arrived after the shard's settlement window"),
            "a reward must not land once a refund may already have settled: {stale}"
        );
    }

    /// The refund carries the reward's claim key, so a shard settling twice is
    /// refused by the store rather than by a rule somebody has to remember.
    #[test]
    fn a_shard_that_was_rewarded_cannot_also_be_refunded() {
        let set = validator_set("refund-exclusive");
        let work = WorkSpec::PrimeCount { start: 2, end: 20 };
        let provider = test_identity("refund-exclusive-provider");
        let mut store = crate::store::LedgerStore::open(":memory:").unwrap();
        let reserve = certified_community_reserve(&set, "job-once", 1, "GENESIS");
        store.apply(&reserve, &set.set).unwrap();
        let reward = certify(
            &set,
            signed_reward("job-once", 0, &provider, &work, true),
            2,
            &reserve.entry.entry_hash,
        );
        store.apply(&reward, &set.set).unwrap();

        let late = 1 + hocmesh_protocol::SETTLEMENT_WINDOW_SECS + 1;
        let refund = certify(
            &set,
            refund_tx("job-once", 0, &work, true, None, late),
            3,
            &reward.entry.entry_hash,
        );
        let second = refused(store.apply(&refund, &set.set));
        assert!(
            second.contains("claim already settled"),
            "the reward already spent this shard's one settlement: {second}"
        );
    }

    /// The point of the whole transaction: escrow that would otherwise be
    /// locked forever comes back, and no CU is created on the way.
    #[test]
    fn refunding_a_dead_shard_returns_every_unit_it_reserved() {
        let set = validator_set("refund-conserved");
        let work = WorkSpec::PrimeCount { start: 2, end: 20 };
        let late = 1 + hocmesh_protocol::SETTLEMENT_WINDOW_SECS + 1;
        let refund = refund_tx("job-dead", 0, &work, true, None, late);
        let store = settle_and_replay(&set, "job-dead", refund).unwrap();
        assert_eq!(store.balance(&escrow_account("job-dead")).unwrap(), 0);
        assert_eq!(
            store.balance(COMMUNITY_ISSUANCE_ACCOUNT).unwrap(),
            0,
            "CU minted for work nobody did is unminted"
        );
    }

    /// A batched entry is a real entry: it hashes, verifies and carries every
    /// settlement it was given.
    #[test]
    fn batched_entry_carries_every_transaction() {
        let v = validator_set("batch-verify");
        let a = certified_community_reserve(&v, "batch-a", 1, "GENESIS")
            .entry
            .transactions[0]
            .clone();
        let b = certified_community_reserve(&v, "batch-b", 1, "GENESIS")
            .entry
            .transactions[0]
            .clone();
        let cert = certify_batch(&v, vec![a, b], 1, "GENESIS");
        verify_certificate(&cert, &v.set).unwrap();
        assert_eq!(cert.entry.transactions.len(), 2);
    }

    /// Across entries the claims table enforces exactly-once. Inside one entry
    /// nothing else will, so the certificate check has to.
    #[test]
    fn batched_entry_cannot_settle_one_claim_twice() {
        let v = validator_set("batch-double");
        let tx = certified_community_reserve(&v, "batch-dup", 1, "GENESIS")
            .entry
            .transactions[0]
            .clone();
        let cert = certify_batch(&v, vec![tx.clone(), tx], 1, "GENESIS");
        let err = verify_certificate(&cert, &v.set).unwrap_err().to_string();
        assert!(err.contains("twice"), "{err}");
    }

    /// An entry with nothing in it would still move the head and chain a hash,
    /// so an empty batch has to be refused outright.
    #[test]
    fn empty_entry_is_rejected() {
        let v = validator_set("batch-empty");
        let cert = certify_batch(&v, vec![], 1, "GENESIS");
        let err = verify_certificate(&cert, &v.set).unwrap_err().to_string();
        assert!(err.contains("no transactions"), "{err}");
    }

    /// The reason batching needs its own check: two settlements that each fit
    /// under the issuance limit can breach it together, and a batch that
    /// judged both against the same opening balance would let them.
    #[test]
    fn batch_cannot_overdraw_across_transactions() {
        let v = validator_set("batch-overdraw");
        let a = certified_community_reserve(&v, "over-a", 1, "GENESIS")
            .entry
            .transactions[0]
            .clone();
        let b = certified_community_reserve(&v, "over-b", 1, "GENESIS")
            .entry
            .transactions[0]
            .clone();
        let cost = -a.postings[0].delta_mcu;
        let limit = cost + cost / 2;
        // Each alone leaves the issuance account inside the limit.
        validate_transaction(&a, "GENESIS", |_| Ok(0), limit).unwrap();
        validate_transaction(&b, "GENESIS", |_| Ok(0), limit).unwrap();
        let err = validate_batch(&[a, b], "GENESIS", |_| Ok(0), limit)
            .unwrap_err()
            .to_string();
        assert!(err.contains("issuance limit"), "{err}");
    }

    fn checkpoint(validators: &TestValidatorSet, signers: usize) -> LedgerCheckpoint {
        let head = LedgerHead {
            sequence: 7,
            entry_hash: "entry7".into(),
            membership_hash: membership_hash(&validators.set).unwrap(),
        };
        let state_hash = "state-digest".to_string();
        let message = checkpoint_signing_message(
            &head.membership_hash,
            head.sequence,
            &head.entry_hash,
            &state_hash,
        );
        LedgerCheckpoint {
            head,
            state_hash,
            signatures: validators
                .identities
                .iter()
                .take(signers)
                .map(|i| ValidatorSignature {
                    validator_id: i.node_id(),
                    signature_b64: i.sign_bytes_b64(message.as_bytes()),
                })
                .collect(),
        }
    }

    #[test]
    fn checkpoint_needs_a_quorum_of_signatures() {
        let v = validator_set("cp-quorum");
        assert!(verify_checkpoint(&checkpoint(&v, v.set.threshold), &v.set).is_ok());
        assert!(verify_checkpoint(&checkpoint(&v, v.set.threshold - 1), &v.set).is_err());
    }

    /// The state hash is the whole point of a checkpoint: if it could be
    /// changed after signing, an auditor could be handed any starting state.
    #[test]
    fn checkpoint_state_hash_cannot_be_swapped_after_signing() {
        let v = validator_set("cp-swap");
        let mut cp = checkpoint(&v, v.set.threshold);
        cp.state_hash = "some-other-state".into();
        assert!(verify_checkpoint(&cp, &v.set).is_err());
    }

    #[test]
    fn checkpoint_signatures_do_not_count_twice() {
        let v = validator_set("cp-dup");
        let mut cp = checkpoint(&v, 1);
        let only = cp.signatures[0].clone();
        cp.signatures = vec![only.clone(), only.clone(), only];
        assert!(verify_checkpoint(&cp, &v.set).is_err());
    }

    /// A checkpoint signed against a different validator set says nothing
    /// about this one, however many valid-looking signatures it carries.
    #[test]
    fn checkpoint_from_another_validator_set_is_rejected() {
        let (a, b) = (validator_set("cp-set-a"), validator_set("cp-set-b"));
        assert!(verify_checkpoint(&checkpoint(&a, a.set.threshold), &b.set).is_err());
    }

    /// Signs whatever the store currently holds, the way a validator answering
    /// `/v1/ledger/state` does.
    fn signed_checkpoint(
        v: &TestValidatorSet,
        store: &crate::store::LedgerStore,
    ) -> LedgerCheckpoint {
        signed_checkpoint_at(v, store, None)
    }

    /// The same, but able to sign for a state the store does not actually
    /// hold, which is the only way to test that the store checks.
    fn signed_checkpoint_at(
        v: &TestValidatorSet,
        store: &crate::store::LedgerStore,
        state_hash: Option<&str>,
    ) -> LedgerCheckpoint {
        let head = store.head(&v.set).unwrap();
        let state_hash = state_hash
            .map(str::to_string)
            .unwrap_or_else(|| store.state().unwrap().digest().unwrap());
        let message = checkpoint_signing_message(
            &head.membership_hash,
            head.sequence,
            &head.entry_hash,
            &state_hash,
        );
        LedgerCheckpoint {
            head,
            state_hash,
            signatures: v
                .identities
                .iter()
                .take(v.set.threshold)
                .map(|i| ValidatorSignature {
                    validator_id: i.node_id(),
                    signature_b64: i.sign_bytes_b64(message.as_bytes()),
                })
                .collect(),
        }
    }

    /// A file rather than `:memory:`, so a test can close the store and open it
    /// again - which is where a pruned ledger has to prove it stayed intact.
    fn checkpoint_db(v: &TestValidatorSet) -> String {
        v._dir.join("ledger.db").to_string_lossy().to_string()
    }

    /// Builds a ledger deep enough for a checkpoint to have history both
    /// below and above it: a reservation, its refund, a checkpoint there, and
    /// a second reservation on top.
    fn checkpointed_ledger(v: &TestValidatorSet) -> (crate::store::LedgerStore, LedgerCheckpoint) {
        let work = WorkSpec::PrimeCount { start: 2, end: 20 };
        let at = 1 + hocmesh_protocol::SETTLEMENT_WINDOW_SECS + 1;
        let mut store = crate::store::LedgerStore::open(&checkpoint_db(v)).unwrap();
        let reserve = certified_community_reserve(v, "job-cp", 1, "GENESIS");
        store.apply(&reserve, &v.set).unwrap();
        let refund = refund_tx("job-cp", 0, &work, true, None, at);
        let cert = certify(v, refund, 2, &reserve.entry.entry_hash);
        store.apply(&cert, &v.set).unwrap();
        let cp = signed_checkpoint(v, &store);
        store.store_checkpoint(&cp, &v.set).unwrap();
        let later = certified_community_reserve(v, "job-cp-later", 3, &cert.entry.entry_hash);
        store.apply(&later, &v.set).unwrap();
        (store, cp)
    }

    /// An audit that starts from a checkpoint has to reach exactly the same
    /// place as one that replays everything, or the shortcut is not one.
    #[test]
    fn audit_from_a_checkpoint_lands_where_a_full_audit_does() {
        let v = validator_set("cp-audit");
        let (store, cp) = checkpointed_ledger(&v);
        let full = store.audit(&v.set).unwrap();
        let short = store.audit_from(&v.set, Some(&cp)).unwrap();
        assert_eq!(
            (full.sequence, full.entry_hash),
            (short.sequence, short.entry_hash)
        );
    }

    /// Pruning is only safe if what it removes was never needed again. After
    /// it, the checkpoint route still works and the genesis route says plainly
    /// that it cannot rather than auditing a shortened history.
    #[test]
    fn pruning_below_a_checkpoint_keeps_the_ledger_auditable() {
        let v = validator_set("cp-prune");
        let (store, cp) = checkpointed_ledger(&v);
        store.prune_below_checkpoint(&v.set).unwrap();
        store
            .audit_from(&v.set, Some(&cp))
            .expect("a pruned ledger still audits from its checkpoint");
        let err = store.audit(&v.set).unwrap_err().to_string();
        assert!(err.contains("has been pruned"), "{err}");
    }

    /// A checkpoint the store cannot reproduce from its own tables is a
    /// disagreement, not a shortcut, and has to be refused before it is kept.
    #[test]
    fn a_checkpoint_the_store_disagrees_with_is_refused() {
        let v = validator_set("cp-disagree");
        let (store, _) = checkpointed_ledger(&v);
        // Properly signed by a quorum, and still wrong about the state.
        let wrong = signed_checkpoint_at(&v, &store, Some("not-the-state-this-store-holds"));
        verify_checkpoint(&wrong, &v.set).expect("the signatures themselves are valid");
        let err = store
            .store_checkpoint(&wrong, &v.set)
            .unwrap_err()
            .to_string();
        assert!(err.contains("checkpoint mismatch"), "{err}");
    }

    /// Reopening is where pruning nearly went wrong: the store rebuilds its
    /// derived tables from the certificates on every open, and once the ones
    /// below the checkpoint are gone that rebuild would erase state the
    /// quorum has already signed for.
    #[test]
    fn a_pruned_ledger_survives_being_reopened() {
        let v = validator_set("cp-reopen");
        let (store, cp) = checkpointed_ledger(&v);
        store.prune_below_checkpoint(&v.set).unwrap();
        let before = store.state().unwrap().digest().unwrap();
        drop(store);
        let reopened = crate::store::LedgerStore::open(&checkpoint_db(&v)).unwrap();
        assert_eq!(
            before,
            reopened.state().unwrap().digest().unwrap(),
            "reopening a pruned ledger must not change the state it holds"
        );
        reopened.audit_from(&v.set, Some(&cp)).unwrap();
    }

    /// A validator that is not in the set yet, ready to be sponsored in.
    fn outsider(v: &TestValidatorSet, name: &str) -> (NodeIdentity, ValidatorMember) {
        let id = NodeIdentity::load_or_create(&v._dir.join(name)).unwrap();
        let member = ValidatorMember {
            validator_id: id.node_id(),
            url: format!("http://127.0.0.1:9999/{name}"),
            public_key_b64: id.public_key_b64(),
        };
        (id, member)
    }

    /// A change sponsored by the first `signers` sitting validators.
    fn change(
        v: &TestValidatorSet,
        action: MembershipAction,
        member: &ValidatorMember,
        threshold: usize,
        signers: &[&NodeIdentity],
    ) -> MembershipChangeEvidence {
        let next = membership_result(&v.set, action, member, threshold).unwrap();
        let resulting_set_hash = membership_hash(&next).unwrap();
        let message = vouch_signing_message(
            &membership_hash(&v.set).unwrap(),
            action,
            member,
            &resulting_set_hash,
        );
        let vouches = signers
            .iter()
            .map(|id| ValidatorSignature {
                validator_id: id.node_id(),
                signature_b64: id.sign_bytes_b64(message.as_bytes()),
            })
            .collect();
        MembershipChangeEvidence {
            action,
            member: member.clone(),
            threshold,
            vouches,
            resulting_set_hash,
        }
    }

    #[test]
    fn a_join_needs_a_quorum_of_sitting_validators() {
        let v = validator_set("membership_join_quorum");
        let (_, member) = outsider(&v, "joiner");
        let ids: Vec<&NodeIdentity> = v.identities.iter().collect();

        let short = change(&v, MembershipAction::Join, &member, 4, &ids[..2]);
        assert!(verify_membership_change(&v.set, &short).is_err());

        let enough = change(&v, MembershipAction::Join, &member, 4, &ids[..3]);
        let next = verify_membership_change(&v.set, &enough).unwrap();
        assert_eq!(next.members.len(), 5);
        assert!(
            next.members
                .iter()
                .any(|m| m.validator_id == member.validator_id)
        );
    }

    #[test]
    fn an_outsider_cannot_vouch_itself_in() {
        let v = validator_set("membership_self_vouch");
        let (joiner, member) = outsider(&v, "joiner");
        let mut signers: Vec<&NodeIdentity> = v.identities.iter().take(2).collect();
        signers.push(&joiner);
        let e = change(&v, MembershipAction::Join, &member, 4, &signers);
        assert_eq!(e.vouches.len(), 3);
        assert!(verify_membership_change(&v.set, &e).is_err());
    }

    #[test]
    fn a_vouch_cannot_be_replayed_against_a_set_that_has_moved() {
        let v = validator_set("membership_replay");
        let (_, first) = outsider(&v, "first");
        let (_, second) = outsider(&v, "second");
        let ids: Vec<&NodeIdentity> = v.identities.iter().collect();

        let admit = change(&v, MembershipAction::Join, &first, 4, &ids[..3]);
        let moved = verify_membership_change(&v.set, &admit).unwrap();

        let mut stale = change(&v, MembershipAction::Join, &second, 4, &ids[..3]);
        stale.threshold = 5;
        // Repointed at the set it is now presented against, so the only thing
        // left wrong is what the sponsors actually put their names to.
        stale.resulting_set_hash = membership_hash(
            &membership_result(&moved, MembershipAction::Join, &second, 5).unwrap(),
        )
        .unwrap();
        assert!(verify_membership_change(&moved, &stale).is_err());
    }

    #[test]
    fn a_membership_change_cannot_claim_a_set_it_does_not_produce() {
        let v = validator_set("membership_claimed_set");
        let (_, member) = outsider(&v, "joiner");
        let ids: Vec<&NodeIdentity> = v.identities.iter().collect();
        let mut e = change(&v, MembershipAction::Join, &member, 4, &ids[..3]);
        e.resulting_set_hash = membership_hash(&v.set).unwrap();
        assert!(verify_membership_change(&v.set, &e).is_err());
    }

    #[test]
    fn a_membership_change_must_move_no_credit() {
        let v = validator_set("membership_no_credit");
        let (_, member) = outsider(&v, "joiner");
        let ids: Vec<&NodeIdentity> = v.identities.iter().collect();
        let e = change(&v, MembershipAction::Join, &member, 4, &ids[..3]);
        let mut tx = LedgerTransaction {
            transaction_id: "membership_test".into(),
            kind: TransactionKind::MembershipChange,
            postings: Vec::new(),
            evidence: TransactionEvidence::MembershipChange(e),
            created_at: 0,
        };
        validate_transaction(&tx, TEST_PREVIOUS_HASH, |_| Ok(0), 1_000_000).unwrap();
        tx.postings = vec![
            Posting {
                account_id: COMMUNITY_ISSUANCE_ACCOUNT.into(),
                delta_mcu: -5,
            },
            Posting {
                account_id: "hocmesh:node:attacker".into(),
                delta_mcu: 5,
            },
        ];
        assert!(validate_transaction(&tx, TEST_PREVIOUS_HASH, |_| Ok(0), 1_000_000).is_err());
    }

    #[test]
    fn a_leave_must_describe_the_member_the_set_actually_holds() {
        let v = validator_set("membership_leave");
        let sitting = v.set.members[3].clone();
        let ids: Vec<&NodeIdentity> = v.identities.iter().collect();

        let out = change(&v, MembershipAction::Leave, &sitting, 3, &ids[..3]);
        let next = verify_membership_change(&v.set, &out).unwrap();
        assert_eq!(next.members.len(), 3);
        assert!(
            !next
                .members
                .iter()
                .any(|m| m.validator_id == sitting.validator_id)
        );

        let mut impostor = sitting.clone();
        impostor.public_key_b64 = v.set.members[0].public_key_b64.clone();
        let mut forged = out;
        forged.member = impostor;
        assert!(verify_membership_change(&v.set, &forged).is_err());
    }

    /// A job somebody reserved: two prompts, one batch, priced by the bill.
    fn reservation_for(requester: &str, provider: &str) -> InferenceReservation {
        InferenceReservation {
            job_id: "job1".into(),
            billing: InferenceBilling {
                manifest_digest: "manifest".into(),
                parameter_count: 1_000_000,
                total_size_bytes: 4_000_000,
                prompts_digest: "prompts".into(),
                prompt_bytes: vec![40, 60],
                max_tokens: 128,
                max_cost_mcu: 1_000_000,
            },
            batches: vec![PricedBatch {
                batch_start: 0,
                batch_end: 2,
                node_id: provider.to_string(),
            }],
            requester: requester.to_string(),
            reserved_at: 1_700_000_000,
        }
    }

    /// The assignment id a job determines for one of its batches.
    fn det(job_id: &str, index: u32) -> String {
        hocmesh_protocol::inference_assignment_id(job_id, index)
    }

    /// The price that reservation implies, recomputed the way the ledger does.
    fn reserved_price(r: &InferenceReservation) -> i64 {
        reserved_batch(
            r,
            &hocmesh_protocol::inference_assignment_id(&r.job_id, 0),
            0,
            2,
        )
        .unwrap()
        .1
    }

    /// A provider's claim on a batch, signed the way a provider signs one.
    fn signed_inference_reward(
        provider: &NodeIdentity,
        job_id: &str,
        assignment_id: &str,
        start: u32,
        end: u32,
        reward_mcu: i64,
    ) -> LedgerTransaction {
        let outputs_digest = "outputs".to_string();
        let bh = hocmesh_protocol::inference_reward_body_hash(
            assignment_id,
            job_id,
            start,
            end,
            reward_mcu,
            &outputs_digest,
        )
        .unwrap();
        let auth = provider.auth("report_inference", &bh);
        LedgerTransaction {
            transaction_id: format!("reward-{assignment_id}"),
            kind: TransactionKind::InferenceReward,
            postings: vec![
                Posting {
                    account_id: escrow_account(job_id),
                    delta_mcu: -reward_mcu,
                },
                Posting {
                    account_id: auth.node_id.clone(),
                    delta_mcu: reward_mcu,
                },
            ],
            evidence: TransactionEvidence::InferenceReward(InferenceRewardEvidence {
                job_id: job_id.to_string(),
                assignment_id: assignment_id.to_string(),
                batch_start: start,
                batch_end: end,
                reward_mcu,
                outputs_digest,
                provider_public_key_b64: provider.public_key_b64(),
                provider_auth: auth,
            }),
            created_at: 1_700_000_100,
        }
    }

    /// The honest claim: the assigned node, the reserved batch, the agreed price.
    #[test]
    fn an_assigned_provider_is_paid_the_reserved_price() {
        let provider = test_identity("inf-provider");
        let r = reservation_for("hocmesh:node:requester", &provider.node_id());
        let price = reserved_price(&r);
        let tx = signed_inference_reward(&provider, "job1", &det("job1", 0), 0, 2, price);
        super::validate_transaction(
            &tx,
            TEST_PREVIOUS_HASH,
            |_| Ok(1_000_000),
            |_| Ok(Some(r.clone())),
            0,
        )
        .unwrap();
    }

    /// Somebody else's batch. Nothing about the claim is forged - the signature
    /// is real - it is simply a claim on work that was never assigned to them.
    #[test]
    fn a_node_cannot_claim_a_batch_assigned_to_another() {
        let thief = test_identity("inf-thief");
        let r = reservation_for("hocmesh:node:requester", "hocmesh:node:assigned");
        let price = reserved_price(&r);
        let tx = signed_inference_reward(&thief, "job1", &det("job1", 0), 0, 2, price);
        let err = super::validate_transaction(
            &tx,
            TEST_PREVIOUS_HASH,
            |_| Ok(1_000_000),
            |_| Ok(Some(r.clone())),
            0,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("not to the claimant"), "{err}");
    }

    /// The right node, the right batch, a price it made up. The signature covers
    /// the amount, so this is the provider's own honest signature over a lie.
    #[test]
    fn a_provider_cannot_price_its_own_batch() {
        let provider = test_identity("inf-greedy");
        let r = reservation_for("hocmesh:node:requester", &provider.node_id());
        let price = reserved_price(&r);
        let tx = signed_inference_reward(&provider, "job1", &det("job1", 0), 0, 2, price * 10);
        let err = super::validate_transaction(
            &tx,
            TEST_PREVIOUS_HASH,
            |_| Ok(100_000_000),
            |_| Ok(Some(r.clone())),
            0,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("price the requester signed"), "{err}");
    }

    /// A batch nobody bought. Without the reservation to check against, this is
    /// indistinguishable from an honest claim.
    #[test]
    fn an_unreserved_batch_cannot_be_claimed() {
        let provider = test_identity("inf-ghost");
        let r = reservation_for("hocmesh:node:requester", &provider.node_id());
        let tx = signed_inference_reward(&provider, "job1", &det("job1", 1), 2, 4, 1_000);
        let err = super::validate_transaction(
            &tx,
            TEST_PREVIOUS_HASH,
            |_| Ok(1_000_000),
            |_| Ok(Some(r.clone())),
            0,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("assignment this job never had"), "{err}");
    }

    /// A job with no reservation at all - the escrow was funded some other way,
    /// or the reserve never certified. There is nothing to pay out of.
    #[test]
    fn a_reward_without_a_reservation_is_rejected() {
        let provider = test_identity("inf-orphan");
        let tx = signed_inference_reward(&provider, "job1", "a1", 0, 2, 1_000);
        let err = super::validate_transaction(
            &tx,
            TEST_PREVIOUS_HASH,
            |_| Ok(1_000_000),
            |_| Ok(None),
            0,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no reservation"), "{err}");
    }

    /// Two assignment ids, one batch. The claim key is the batch, so the second
    /// claim collides with the first instead of drawing the escrow down twice.
    #[test]
    fn one_batch_settles_once_whatever_the_assignment_is_called() {
        let provider = test_identity("inf-double");
        let a = signed_inference_reward(&provider, "job1", "a1", 0, 2, 1_000);
        let b = signed_inference_reward(&provider, "job1", "a2", 0, 2, 1_000);
        assert_eq!(claim_key(&a), claim_key(&b));
    }
}
