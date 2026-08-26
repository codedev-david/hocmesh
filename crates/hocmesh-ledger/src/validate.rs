use crate::types::*;
use anyhow::{Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD_NO_PAD};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use hocmesh_core::compute::{split_work, work_cost_mcu};
use hocmesh_core::verify::{self, AuditNonce};
use hocmesh_protocol::{hash_json, result_body_hash, submit_body_hash, verify_auth_signature};

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
pub fn entry_hash(sequence: u64, previous_hash: &str, tx_hash: &str) -> Result<String> {
    Ok(hash_json(&(sequence, previous_hash, tx_hash))?)
}
pub fn build_entry(
    sequence: u64,
    previous_hash: String,
    tx: LedgerTransaction,
) -> Result<LedgerEntry> {
    let th = transaction_hash(&tx)?;
    let eh = entry_hash(sequence, &previous_hash, &th)?;
    Ok(LedgerEntry {
        sequence,
        previous_hash,
        transaction: tx,
        transaction_hash: th,
        entry_hash: eh,
    })
}

pub fn claim_key(tx: &LedgerTransaction) -> String {
    match &tx.evidence {
        TransactionEvidence::JobReserve(e) => format!("reserve:{}", e.job_id),
        TransactionEvidence::CommunityReserve { job_id, .. } => format!("reserve:{}", job_id),
        TransactionEvidence::ProviderReward(e) => format!("reward:{}", e.assignment_id),
    }
}

pub fn validate_transaction(
    tx: &LedgerTransaction,
    balance: impl Fn(&str) -> Result<i64>,
    community_issuance_limit_mcu: i64,
) -> Result<()> {
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
            validate_reward(tx, e)?
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

fn validate_community_reserve(
    tx: &LedgerTransaction,
    job_id: &str,
    work: &hocmesh_protocol::WorkSpec,
    shards: u32,
) -> Result<()> {
    work.validate().map_err(anyhow::Error::msg)?;
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
fn validate_reward(tx: &LedgerTransaction, e: &ProviderRewardEvidence) -> Result<()> {
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
    if !witnessed(&e.work, &e.result, e.audit_nonce) {
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
    balance: impl Fn(&str) -> Result<i64>,
    community_issuance_limit_mcu: i64,
) -> Result<()> {
    verify_historical_evidence(tx)?;
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
    let th = transaction_hash(&cert.entry.transaction)?;
    if th != cert.entry.transaction_hash {
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

/// Audit-time evidence signature validation without rejecting old timestamps.
pub fn verify_historical_evidence(tx: &LedgerTransaction) -> Result<()> {
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
            if !witnessed(&e.work, &e.result, e.audit_nonce) {
                bail!("historical work result invalid")
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
fn witnessed(
    work: &hocmesh_protocol::WorkSpec,
    result: &hocmesh_protocol::WorkResult,
    nonce: u64,
) -> bool {
    let verdict = verify::witness_check(work, result, AuditNonce::replay(nonce));
    if verdict == verify::Verdict::Inconclusive {
        return verify::adjudicate(work, result).is_accepted();
    }
    verdict.is_accepted()
}
#[cfg(test)]
mod tests {
    use super::*;
    use hocmesh_core::identity::NodeIdentity;
    use hocmesh_protocol::{AuthProof, WorkSpec, canonical_auth_message};
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
        validate_transaction(&tx, |_| Ok(0), 1_000_000).unwrap();
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
        assert!(validate_transaction(&tx, |_| Ok(0), 1_000_000).is_err());
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
        assert!(validate_transaction(&tx, |_| Ok(0), 1).is_err());
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

        assert!(validate_transaction(&tx, |_| Ok(10_000), 1_000_000).is_err());
    }

    #[test]
    fn old_historical_requester_signature_remains_auditable() {
        let requester = test_identity("old-requester");
        let work = WorkSpec::PrimeCount { start: 2, end: 20 };
        let body_hash = submit_body_hash(&work, 1).unwrap();
        let old_auth = auth_at(&requester, "submit", &body_hash, 1);
        let tx = reserve_from_auth("old-job", &requester.public_key_b64(), old_auth, &work, 1);

        validate_historical_transaction(&tx, |_| Ok(10_000), 1_000_000).unwrap();
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

        assert!(validate_transaction(&tx, |_| Ok(10_000), 1_000_000).is_err());
    }

    #[test]
    fn provider_reward_rejects_changed_reward_and_assignment_id() {
        let provider = test_identity("provider-reward");
        let work = WorkSpec::PrimeCount { start: 2, end: 20 };
        let mut changed_reward = signed_reward("job-reward", 0, &provider, &work, true);
        if let TransactionEvidence::ProviderReward(e) = &mut changed_reward.evidence {
            e.reward_mcu += 1;
        }
        assert!(validate_transaction(&changed_reward, |_| Ok(10_000), 1_000_000).is_err());

        let mut changed_assignment = signed_reward("job-assignment", 0, &provider, &work, true);
        if let TransactionEvidence::ProviderReward(e) = &mut changed_assignment.evidence {
            e.assignment_id = "asg_not_the_deterministic_id".into();
        }
        assert!(validate_transaction(&changed_assignment, |_| Ok(10_000), 1_000_000).is_err());
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
        cert.entry.transaction.postings[1].delta_mcu += 1;

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
                audit_nonce: 0,
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
        let entry = build_entry(sequence, previous_hash.into(), tx).unwrap();
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
}
