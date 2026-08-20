use crate::{
    types::*,
    validate::{
        build_entry, ledger_entry_signing_message, membership_hash, validate_validator_set,
        verify_certificate, verify_validator_signature,
    },
};
use anyhow::{Result, bail};
use reqwest::Client;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct LedgerNetwork {
    http: Client,
    pub set: ValidatorSet,
    gate: Arc<Mutex<()>>,
}
impl LedgerNetwork {
    pub fn new(set: ValidatorSet) -> Result<Self> {
        validate_validator_set(&set)?;
        Ok(Self {
            http: Client::new(),
            set,
            gate: Arc::new(Mutex::new(())),
        })
    }
    pub async fn head_quorum(&self) -> Result<LedgerHead> {
        let mh = membership_hash(&self.set)?;
        let mut heads = Vec::new();
        for m in &self.set.members {
            if let Ok(r) = self
                .http
                .get(format!("{}/v1/ledger/head", m.url.trim_end_matches('/')))
                .send()
                .await
                && let Ok(p) = r.json::<HeadProof>().await
            {
                let msg = format!(
                    "mesh-head-v1|{}|{}|{}",
                    p.head.membership_hash, p.head.sequence, p.head.entry_hash
                );
                if p.validator_id == m.validator_id
                    && p.head.membership_hash == mh
                    && verify_validator_signature(m, &msg, &p.signature_b64).is_ok()
                {
                    heads.push(p.head)
                }
            }
        }
        let mut counts = std::collections::HashMap::<(u64, String), usize>::new();
        for h in &heads {
            *counts
                .entry((h.sequence, h.entry_hash.clone()))
                .or_default() += 1;
        }
        let Some(((s, hash), n)) = counts.into_iter().max_by_key(|x| x.1) else {
            bail!("no signed validator heads available")
        };
        if n < self.set.threshold {
            bail!("no quorum-agreed ledger head")
        };
        Ok(LedgerHead {
            sequence: s,
            entry_hash: hash,
            membership_hash: mh,
        })
    }
    pub async fn balance_quorum(&self, account: &str) -> Result<BalanceProof> {
        let mut proofs = Vec::new();
        for m in &self.set.members {
            if let Ok(r) = self
                .http
                .get(format!(
                    "{}/v1/ledger/balance/{}",
                    m.url.trim_end_matches('/'),
                    account
                ))
                .send()
                .await
                && let Ok(p) = r.json::<BalanceProof>().await
                && p.validator_id == m.validator_id
                && p.head.membership_hash == membership_hash(&self.set)?
                && verify_validator_signature(
                    m,
                    &format!(
                        "mesh-balance-v1|{}|{}|{}|{}|{}|{}|{}",
                        p.head.membership_hash,
                        p.account_id,
                        p.balance_mcu,
                        p.earned_mcu,
                        p.spent_mcu,
                        p.head.sequence,
                        p.head.entry_hash
                    ),
                    &p.signature_b64,
                )
                .is_ok()
            {
                proofs.push(p)
            }
        }
        let mut counts = std::collections::HashMap::<(i64, i64, i64, u64, String), usize>::new();
        for p in &proofs {
            *counts
                .entry((
                    p.balance_mcu,
                    p.earned_mcu,
                    p.spent_mcu,
                    p.head.sequence,
                    p.head.entry_hash.clone(),
                ))
                .or_default() += 1;
        }
        let Some(((b, e, sp, s, h), n)) = counts.into_iter().max_by_key(|x| x.1) else {
            bail!("no validator balance proofs available")
        };
        if n < self.set.threshold {
            bail!("no quorum-agreed balance")
        };
        Ok(proofs
            .into_iter()
            .find(|p| {
                p.balance_mcu == b
                    && p.earned_mcu == e
                    && p.spent_mcu == sp
                    && p.head.sequence == s
                    && p.head.entry_hash == h
            })
            .unwrap())
    }

    pub async fn claim_quorum(&self, claim: &str) -> Result<ClaimProof> {
        let mh = membership_hash(&self.set)?;
        let mut proofs = Vec::new();
        for m in &self.set.members {
            if let Ok(r) = self
                .http
                .get(format!(
                    "{}/v1/ledger/claim/{}",
                    m.url.trim_end_matches('/'),
                    claim
                ))
                .send()
                .await
                && let Ok(p) = r.json::<ClaimProof>().await
            {
                let msg = format!(
                    "mesh-claim-v1|{}|{}|{:?}|{:?}|{}|{}",
                    p.head.membership_hash,
                    p.claim_key,
                    p.sequence,
                    p.entry_hash,
                    p.head.sequence,
                    p.head.entry_hash
                );
                if p.validator_id == m.validator_id
                    && p.head.membership_hash == mh
                    && p.claim_key == claim
                    && verify_validator_signature(m, &msg, &p.signature_b64).is_ok()
                {
                    if let Some(cert) = &p.certificate
                        && verify_certificate(cert, &self.set).is_ok()
                        && crate::validate::claim_key(&cert.entry.transaction) == claim
                        && Some(cert.entry.sequence) == p.sequence
                        && Some(cert.entry.entry_hash.clone()) == p.entry_hash
                    {
                        return Ok(p);
                    }
                    proofs.push(p)
                }
            }
        }
        let mut counts =
            std::collections::HashMap::<(Option<u64>, Option<String>, u64, String), usize>::new();
        for p in &proofs {
            *counts
                .entry((
                    p.sequence,
                    p.entry_hash.clone(),
                    p.head.sequence,
                    p.head.entry_hash.clone(),
                ))
                .or_default() += 1;
        }
        let Some(((seq, eh, hs, hh), n)) = counts.into_iter().max_by_key(|x| x.1) else {
            bail!("no validator claim proofs available")
        };
        if n < self.set.threshold {
            bail!(
                "no quorum-agreed absent claim status and no verifiable quorum certificate was returned"
            )
        };
        Ok(proofs
            .into_iter()
            .find(|p| {
                p.sequence == seq
                    && p.entry_hash == eh
                    && p.head.sequence == hs
                    && p.head.entry_hash == hh
            })
            .unwrap())
    }
    pub async fn transact(&self, tx: LedgerTransaction) -> Result<QuorumCertificate> {
        let _guard = self.gate.lock().await;
        let head = self.head_quorum().await?;
        let expected = build_entry(head.sequence + 1, head.entry_hash.clone(), tx.clone())?;
        let req = ProposalRequest { transaction: tx };
        let mut sigs = Vec::new();
        for m in &self.set.members {
            if let Ok(r) = self
                .http
                .post(format!("{}/v1/ledger/propose", m.url.trim_end_matches('/')))
                .json(&req)
                .send()
                .await
                && let Ok(v) = r.json::<ProposalVote>().await
                && v.accepted
                && v.entry_hash == expected.entry_hash
                && v.sequence == expected.sequence
                && v.previous_hash == expected.previous_hash
                && let Some(sig) = v.signature_b64
                && verify_validator_signature(
                    m,
                    &ledger_entry_signing_message(&head.membership_hash, &expected.entry_hash),
                    &sig,
                )
                .is_ok()
            {
                sigs.push(ValidatorSignature {
                    validator_id: m.validator_id.clone(),
                    signature_b64: sig,
                })
            }
        }
        if sigs.len() < self.set.threshold {
            bail!(
                "ledger proposal received only {} valid votes; threshold is {}",
                sigs.len(),
                self.set.threshold
            )
        }
        let cert = QuorumCertificate {
            entry: expected,
            membership_hash: membership_hash(&self.set)?,
            signatures: sigs,
        };
        verify_certificate(&cert, &self.set)?;
        let mut committed = 0usize;
        for m in &self.set.members {
            if let Ok(r) = self
                .http
                .post(format!("{}/v1/ledger/commit", m.url.trim_end_matches('/')))
                .json(&cert)
                .send()
                .await
                && r.status().is_success()
            {
                committed += 1
            }
        }
        if committed < self.set.threshold {
            bail!(
                "certificate formed but committed on only {committed} validators; run validator sync/recovery"
            )
        };
        Ok(cert)
    }
    pub async fn fetch_certificates(
        &self,
        from: u64,
        limit: u64,
    ) -> Result<Vec<QuorumCertificate>> {
        for m in &self.set.members {
            let url = format!(
                "{}/v1/ledger/entries?from={}&limit={}",
                m.url.trim_end_matches('/'),
                from,
                limit
            );
            if let Ok(r) = self.http.get(url).send().await
                && r.status().is_success()
            {
                let e = r.json::<EntriesResponse>().await?;
                for c in &e.certificates {
                    verify_certificate(c, &self.set)?;
                }
                return Ok(e.certificates);
            }
        }
        bail!("no validator could provide ledger entries")
    }
}
