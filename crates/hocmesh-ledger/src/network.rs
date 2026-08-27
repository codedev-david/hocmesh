use crate::{
    types::*,
    validate::{
        build_entry, ledger_entry_signing_message, membership_hash, validate_validator_set,
        verify_certificate, verify_membership_change, verify_validator_signature,
    },
};
use anyhow::{Result, bail};
use futures::future::join_all;
use reqwest::Client;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{Mutex, oneshot};

#[derive(Clone)]
pub struct LedgerNetwork {
    http: Client,
    set: Arc<std::sync::RwLock<ValidatorSet>>,
    /// The height at which the current set was last established.
    set_sequence: Arc<std::sync::RwLock<u64>>,
    gate: Arc<Mutex<()>>,
    queue: Arc<std::sync::Mutex<VecDeque<Pending>>>,
    /// Identifies this client's ballots. Two proposers reaching for the same
    /// height need to be distinguishable, and nothing else about a client is.
    proposer: String,
    /// Climbs past whatever ballot last took a height away from us.
    ballot: Arc<std::sync::atomic::AtomicU64>,
}
/// The most transactions one entry will carry.
///
/// A round costs the same however much it settles, but an entry still has to
/// be verified, hashed and applied as a single unit, so the batch is capped
/// rather than left to grow without limit.
const MAX_BATCH: usize = 512;

/// A transaction waiting for a round, and where to send the outcome.
struct Pending {
    tx: LedgerTransaction,
    reply: oneshot::Sender<std::result::Result<QuorumCertificate, String>>,
}

/// Why a round failed, and whether its transactions are still safe to retry.
enum RoundError {
    /// No certificate formed. Nothing was applied anywhere, so every
    /// transaction in the batch is still unspent.
    Rejected(String),
    /// A certificate exists and some validators may already hold it. Resending
    /// these transactions risks applying them twice.
    Committed(String),
}
/// What one attempt at a height achieved.
enum Attempt {
    /// The batch we were asked to settle is now certified.
    Settled(QuorumCertificate),
    /// The height went to another proposer's entry - which we may well have
    /// finished for them. Nothing of ours was applied, so it belongs at the
    /// next height instead.
    Deferred(String),
}
/// How many contested heights a client will walk past before giving up.
///
/// Contention is other people settling, so losing repeatedly is progress for
/// somebody. Still, a caller waiting on a reply deserves an answer eventually.
const ROUND_ATTEMPTS: u32 = 6;

/// A validator that stops answering must not be able to hold a proposer open
/// forever. Without this a single hung socket blocks the round loop past every
/// backoff it has, and the caller waits on a reply that is never coming.
const VALIDATOR_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
impl RoundError {
    fn rejected(e: anyhow::Error) -> Self {
        Self::Rejected(e.to_string())
    }
}
impl std::fmt::Display for RoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (Self::Rejected(m) | Self::Committed(m)) = self;
        f.write_str(m)
    }
}
impl LedgerNetwork {
    pub fn new(set: ValidatorSet) -> Result<Self> {
        validate_validator_set(&set)?;
        Ok(Self {
            http: Client::builder()
                .timeout(VALIDATOR_REQUEST_TIMEOUT)
                .build()?,
            set: Arc::new(std::sync::RwLock::new(set)),
            set_sequence: Arc::default(),
            gate: Arc::new(Mutex::new(())),
            queue: Arc::default(),
            proposer: uuid::Uuid::new_v4().simple().to_string(),
            ballot: Arc::default(),
        })
    }

    /// The validator set this client currently believes in.
    ///
    /// Behind a lock because a certified membership change moves the set under
    /// a running coordinator, and one still addressing the seats it booted with
    /// would be asking a quorum that no longer exists.
    pub fn set(&self) -> ValidatorSet {
        self.set.read().expect("validator set lock").clone()
    }

    /// Follows membership changes the chain has certified since this client
    /// last looked, and adopts the set they produce.
    ///
    /// The entries come from whichever validator answers, which is not a
    /// trusted source and does not need to be. Nothing is adopted that was not
    /// certified by the set already held, so the only way to move this client
    /// onto a set is to have been admitted by the one before it - the same rule
    /// an auditor replaying from genesis follows.
    pub async fn refresh_set(&self) -> Result<bool> {
        let mut active = self.set();
        let mut at = *self.set_sequence.read().expect("set sequence lock");
        let start = at;
        let mut cursor = at;
        loop {
            let certs = self.fetch_certificates(cursor + 1, 256, &active).await?;
            if certs.is_empty() {
                break;
            }
            for c in &certs {
                // Only an entry that changes the set has to prove who signed
                // it. The rest cannot move membership, so walking past them
                // unverified costs nothing - a full audit still checks them.
                let changes: Vec<_> = c
                    .entry
                    .transactions
                    .iter()
                    .filter_map(|t| match &t.evidence {
                        TransactionEvidence::MembershipChange(e) => Some(e),
                        _ => None,
                    })
                    .collect();
                if !changes.is_empty() {
                    for e in changes {
                        active = verify_membership_change(&active, e)?;
                    }
                    at = c.entry.sequence;
                }
                cursor = c.entry.sequence;
            }
        }
        if at == start {
            return Ok(false);
        }
        *self.set.write().expect("validator set lock") = active;
        *self.set_sequence.write().expect("set sequence lock") = at;
        Ok(true)
    }
    pub async fn head_quorum(&self) -> Result<LedgerHead> {
        let mh = membership_hash(&self.set())?;
        // Ask every validator at once. Awaiting them one at a time made a round
        // of consensus cost N round trips instead of one, so adding validators
        // slowed the whole network down - the opposite of what it should do.
        let heads: Vec<LedgerHead> = join_all(self.set().members.iter().map(|m| {
            let (http, mh) = (&self.http, &mh);
            async move {
                let r = http
                    .get(format!("{}/v1/ledger/head", m.url.trim_end_matches('/')))
                    .send()
                    .await
                    .ok()?;
                let p = r.json::<HeadProof>().await.ok()?;
                let msg = format!(
                    "hocmesh-head-v1|{}|{}|{}",
                    p.head.membership_hash, p.head.sequence, p.head.entry_hash
                );
                (p.validator_id == m.validator_id
                    && p.head.membership_hash == *mh
                    && verify_validator_signature(m, &msg, &p.signature_b64).is_ok())
                .then_some(p.head)
            }
        }))
        .await
        .into_iter()
        .flatten()
        .collect();
        let mut counts = std::collections::HashMap::<(u64, String), usize>::new();
        for h in &heads {
            *counts
                .entry((h.sequence, h.entry_hash.clone()))
                .or_default() += 1;
        }
        let Some(((s, hash), n)) = counts.into_iter().max_by_key(|x| x.1) else {
            bail!("no signed validator heads available")
        };
        if n < self.set().threshold {
            bail!("no quorum-agreed ledger head")
        };
        Ok(LedgerHead {
            sequence: s,
            entry_hash: hash,
            membership_hash: mh,
        })
    }
    /// Collects a quorum of validators that agree on the whole ledger state.
    ///
    /// Every validator signs the message a checkpoint is verified against, so
    /// the signatures gathered here need no translation: agreement is just
    /// enough of them saying the same (height, entry, state) triple.
    pub async fn checkpoint_quorum(&self) -> Result<LedgerCheckpoint> {
        let mh = membership_hash(&self.set())?;
        let proofs: Vec<StateProof> = join_all(self.set().members.iter().map(|m| {
            let (http, mh) = (&self.http, &mh);
            async move {
                let r = http
                    .get(format!("{}/v1/ledger/state", m.url.trim_end_matches('/')))
                    .send()
                    .await
                    .ok()?;
                let p = r.json::<StateProof>().await.ok()?;
                let msg = crate::validate::checkpoint_signing_message(
                    &p.head.membership_hash,
                    p.head.sequence,
                    &p.head.entry_hash,
                    &p.state_hash,
                );
                (p.validator_id == m.validator_id
                    && p.head.membership_hash == *mh
                    && verify_validator_signature(m, &msg, &p.signature_b64).is_ok())
                .then_some(p)
            }
        }))
        .await
        .into_iter()
        .flatten()
        .collect();
        let mut groups =
            std::collections::HashMap::<(u64, String, String), Vec<&StateProof>>::new();
        for p in &proofs {
            groups
                .entry((
                    p.head.sequence,
                    p.head.entry_hash.clone(),
                    p.state_hash.clone(),
                ))
                .or_default()
                .push(p);
        }
        let Some(((sequence, entry_hash, state_hash), agreed)) =
            groups.into_iter().max_by_key(|x| x.1.len())
        else {
            bail!("no signed validator state available")
        };
        if agreed.len() < self.set().threshold {
            bail!("no quorum-agreed ledger state")
        };
        Ok(LedgerCheckpoint {
            head: LedgerHead {
                sequence,
                entry_hash,
                membership_hash: mh,
            },
            state_hash,
            signatures: agreed
                .into_iter()
                .map(|p| ValidatorSignature {
                    validator_id: p.validator_id.clone(),
                    signature_b64: p.signature_b64.clone(),
                })
                .collect(),
        })
    }
    pub async fn balance_quorum(&self, account: &str) -> Result<BalanceProof> {
        let mh = membership_hash(&self.set())?;
        let proofs: Vec<BalanceProof> = join_all(self.set().members.iter().map(|m| {
            let (http, mh) = (&self.http, &mh);
            async move {
                let r = http
                    .get(format!(
                        "{}/v1/ledger/balance/{}",
                        m.url.trim_end_matches('/'),
                        account
                    ))
                    .send()
                    .await
                    .ok()?;
                let p = r.json::<BalanceProof>().await.ok()?;
                (p.validator_id == m.validator_id
                    && p.head.membership_hash == *mh
                    && verify_validator_signature(
                        m,
                        &format!(
                            "hocmesh-balance-v1|{}|{}|{}|{}|{}|{}|{}",
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
                    .is_ok())
                .then_some(p)
            }
        }))
        .await
        .into_iter()
        .flatten()
        .collect();
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
        if n < self.set().threshold {
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
        let mh = membership_hash(&self.set())?;
        let all: Vec<ClaimProof> = join_all(self.set().members.iter().map(|m| {
            let (http, mh) = (&self.http, &mh);
            async move {
                let r = http
                    .get(format!(
                        "{}/v1/ledger/claim/{}",
                        m.url.trim_end_matches('/'),
                        claim
                    ))
                    .send()
                    .await
                    .ok()?;
                let p = r.json::<ClaimProof>().await.ok()?;
                let msg = format!(
                    "hocmesh-claim-v1|{}|{}|{:?}|{:?}|{}|{}",
                    p.head.membership_hash,
                    p.claim_key,
                    p.sequence,
                    p.entry_hash,
                    p.head.sequence,
                    p.head.entry_hash
                );
                (p.validator_id == m.validator_id
                    && p.head.membership_hash == *mh
                    && p.claim_key == claim
                    && verify_validator_signature(m, &msg, &p.signature_b64).is_ok())
                .then_some(p)
            }
        }))
        .await
        .into_iter()
        .flatten()
        .collect();
        // One verifiable certificate settles the question by itself: it proves
        // the claim was already spent, whatever the rest of the set reports.
        if let Some(i) = all.iter().position(|p| {
            p.certificate.as_ref().is_some_and(|cert| {
                verify_certificate(cert, &self.set()).is_ok()
                    && cert
                        .entry
                        .transactions
                        .iter()
                        .any(|t| crate::validate::claim_key(t) == claim)
                    && Some(cert.entry.sequence) == p.sequence
                    && Some(cert.entry.entry_hash.clone()) == p.entry_hash
            })
        }) {
            return Ok(all.into_iter().nth(i).unwrap());
        }
        let proofs = all;
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
        if n < self.set().threshold {
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
    /// Settle one transaction, sharing a consensus round with whatever else is
    /// waiting.
    ///
    /// Callers queue up and the first one through the gate settles the whole
    /// queue. Giving every transaction its own round capped the network at a
    /// couple of settlements a second no matter how much hardware joined it,
    /// because a round is three network phases and they cost the same whether
    /// the entry carries one transaction or five hundred.
    pub async fn transact(&self, tx: LedgerTransaction) -> Result<QuorumCertificate> {
        let (reply, result) = oneshot::channel();
        self.queue.lock().unwrap().push_back(Pending { tx, reply });
        {
            let _guard = self.gate.lock().await;
            let batch: Vec<Pending> = {
                let mut q = self.queue.lock().unwrap();
                let n = q.len().min(MAX_BATCH);
                q.drain(..n).collect()
            };
            if !batch.is_empty() {
                self.settle(batch).await;
            }
        }
        // Correct whether or not this caller was the one that led the round:
        // if another did, the answer is already waiting here.
        result
            .await
            .map_err(|_| anyhow::anyhow!("ledger batch was dropped before it settled"))?
            .map_err(anyhow::Error::msg)
    }

    /// Run one round for a batch, then hand each caller its own answer.
    async fn settle(&self, batch: Vec<Pending>) {
        let (txs, replies): (Vec<_>, Vec<_>) = batch.into_iter().map(|p| (p.tx, p.reply)).unzip();
        let mut outcome = self.round(txs.clone()).await;
        // A client addressing seats that have since been replaced fails the
        // same way one the validators merely disagree with fails, and a
        // rejected round applied nothing anywhere. So follow the chain
        // forward once and, when it really does hand back a different set,
        // put the same batch to that one. Without this a single admission
        // strands every running client until somebody restarts it by hand.
        if matches!(outcome, Err(RoundError::Rejected(_)))
            && self.refresh_set().await.unwrap_or(false)
        {
            outcome = self.round(txs.clone()).await;
        }
        match outcome {
            Ok(cert) => {
                for r in replies {
                    let _ = r.send(Ok(cert.clone()));
                }
            }
            Err(RoundError::Rejected(_)) if txs.len() > 1 => {
                // A rejected proposal commits nothing, so the transactions in
                // it are still unspent and can be retried. One transaction the
                // validators dislike must not sink every unrelated settlement
                // that happened to share its round.
                for (tx, reply) in txs.into_iter().zip(replies) {
                    let r = self.round(vec![tx]).await.map_err(|e| e.to_string());
                    let _ = reply.send(r);
                }
            }
            Err(e) => {
                // A formed certificate may already be applied somewhere. There
                // is no safe retry from here; the caller has to resolve it.
                let msg = e.to_string();
                for r in replies {
                    let _ = r.send(Err(msg.clone()));
                }
            }
        }
    }

    /// The next ballot this client will propose under.
    fn next_ballot(&self) -> Ballot {
        Ballot {
            number: self
                .ballot
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1,
            proposer: self.proposer.clone(),
        }
    }
    /// Climb above a ballot that beat ours, so the next attempt can win.
    fn outbid(&self, seen: &Ballot) {
        self.ballot
            .fetch_max(seen.number, std::sync::atomic::Ordering::SeqCst);
    }
    /// Wait before reaching for the next height.
    ///
    /// The jitter comes from this client's own name, so two proposers that
    /// lost the same race do not come back at the same instant and lose it
    /// again the same way.
    async fn backoff(&self, attempt: u32) {
        let jitter = self.proposer.bytes().map(u64::from).sum::<u64>() % 25;
        let millis = (15 + jitter) << attempt.min(5);
        tokio::time::sleep(std::time::Duration::from_millis(millis)).await;
    }
    /// Phase one: take the height, and learn what is already signed there.
    ///
    /// A promise from a validator is a commitment not to sign for any older
    /// ballot, so a threshold of them means no earlier proposer can still
    /// gather a certificate. Whatever those promises report as accepted is
    /// then the only entry this height may carry.
    async fn claim_height(
        &self,
        sequence: u64,
        ballot: &Ballot,
    ) -> std::result::Result<Option<AcceptedProposal>, String> {
        let req = PrepareRequest {
            sequence,
            ballot: ballot.clone(),
        };
        let votes: Vec<PrepareVote> = join_all(self.set().members.iter().map(|m| {
            let (http, req) = (&self.http, &req);
            async move {
                let r = http
                    .post(format!("{}/v1/ledger/prepare", m.url.trim_end_matches('/')))
                    .json(req)
                    .send()
                    .await
                    .ok()?;
                let v = r.json::<PrepareVote>().await.ok()?;
                (v.validator_id == m.validator_id && v.sequence == sequence).then_some(v)
            }
        }))
        .await
        .into_iter()
        .flatten()
        .collect();
        // Anybody who refused is holding a newer ballot; climb past the
        // highest of them so the next attempt is not doomed the same way.
        for v in &votes {
            if let Some(seen) = &v.promised_ballot {
                self.outbid(seen);
            }
        }
        let promises: Vec<&PrepareVote> = votes.iter().filter(|v| v.promised).collect();
        if promises.len() < self.set().threshold {
            return Err(format!(
                "height {sequence} promised to only {} of {} needed validators",
                promises.len(),
                self.set().threshold
            ));
        }
        // Of everything the promising validators have already signed here,
        // the newest ballot is the only one that could have been certified.
        Ok(promises
            .into_iter()
            .filter_map(|v| v.accepted.clone())
            .max_by(|a, b| {
                if a.ballot.outranks(&b.ballot) {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            }))
    }

    /// Settle a batch, retrying while other proposers keep taking the height.
    ///
    /// Losing a height is not a failure: the batch was never applied anywhere,
    /// and the entry that won may well be one this client finished on the
    /// previous owner's behalf. So climb to a higher ballot and try the next
    /// height, backing off by an amount peculiar to this client so two
    /// proposers do not keep stepping on each other in lockstep.
    async fn round(
        &self,
        transactions: Vec<LedgerTransaction>,
    ) -> std::result::Result<QuorumCertificate, RoundError> {
        let mut last = String::from("no round was attempted");
        for attempt in 0..ROUND_ATTEMPTS {
            match self.attempt(transactions.clone()).await {
                Ok(Attempt::Settled(cert)) => return Ok(cert),
                Ok(Attempt::Deferred(why)) => last = why,
                Err(e) => return Err(e),
            }
            self.backoff(attempt).await;
        }
        Err(RoundError::Rejected(format!(
            "gave up after {ROUND_ATTEMPTS} attempts at a contested height: {last}"
        )))
    }

    /// One attempt at one height.
    ///
    /// Two clients can reach for the same sequence at once, and a validator
    /// signs one entry there and no other. So claim the height first: whoever
    /// holds the newest ballot gets to drive it, and a validator that already
    /// signed something hands that entry back for the new proposer to finish.
    /// A round that dies halfway is then just a round somebody else completes,
    /// rather than a height nobody can ever fill.
    async fn attempt(
        &self,
        transactions: Vec<LedgerTransaction>,
    ) -> std::result::Result<Attempt, RoundError> {
        let head = self.head_quorum().await.map_err(RoundError::rejected)?;
        let sequence = head.sequence + 1;
        let ballot = self.next_ballot();
        let adopted = match self.claim_height(sequence, &ballot).await {
            Ok(a) => a,
            Err(why) => return Ok(Attempt::Deferred(why)),
        };
        let deferred = adopted
            .as_ref()
            .map(|a| format!("height {sequence} went to ballot {}", a.ballot));
        // A validator that has already signed something here hands it back,
        // and finishing that entry is not optional: it may already be one vote
        // short of a certificate, and no two entries may exist at one height.
        let transactions = match adopted {
            Some(a) => a.transactions,
            None => transactions,
        };
        let expected = build_entry(sequence, head.entry_hash.clone(), transactions.clone())
            .map_err(RoundError::rejected)?;
        let req = ProposalRequest {
            transactions,
            sequence,
            ballot,
        };
        let votes: Vec<(ValidatorMember, ProposalVote)> =
            join_all(self.set().members.iter().map(|m| {
                let (http, req) = (&self.http, &req);
                async move {
                    let r = http
                        .post(format!("{}/v1/ledger/propose", m.url.trim_end_matches('/')))
                        .json(req)
                        .send()
                        .await
                        .ok()?;
                    Some((m.clone(), r.json::<ProposalVote>().await.ok()?))
                }
            }))
            .await
            .into_iter()
            .flatten()
            .collect();
        // Anyone who has moved on to a newer ballot tells us so, and the next
        // attempt starts above it rather than losing the same race again.
        let outbid = votes
            .iter()
            .filter_map(|(_, v)| v.promised_ballot.as_ref())
            .find(|b| b.outranks(&req.ballot));
        if let Some(b) = outbid {
            self.outbid(b);
            return Ok(Attempt::Deferred(format!(
                "height {sequence} was taken by ballot {b} mid-round"
            )));
        }
        let answered = votes.len();
        let sigs: Vec<ValidatorSignature> = votes
            .into_iter()
            .filter_map(|(m, v)| {
                let sig = v.signature_b64?;
                (v.accepted
                    && v.entry_hash == expected.entry_hash
                    && v.sequence == expected.sequence
                    && v.previous_hash == expected.previous_hash
                    && verify_validator_signature(
                        &m,
                        &ledger_entry_signing_message(&head.membership_hash, &expected.entry_hash),
                        &sig,
                    )
                    .is_ok())
                .then_some(ValidatorSignature {
                    validator_id: m.validator_id,
                    signature_b64: sig,
                })
            })
            .collect();
        if sigs.len() < self.set().threshold {
            // A round falls short for two very different reasons: the set
            // disliked the batch, or somebody else filled this height while we
            // were collecting votes. The first is final; the second is a race
            // that a fresh head settles. Ask the chain which one happened
            // rather than reading it out of a refusal message. A round that
            // fell short applied nothing anywhere, so re-proposing the same
            // batch on top of the new head is safe.
            if self
                .head_quorum()
                .await
                .is_ok_and(|h| h.sequence >= sequence)
            {
                return Ok(Attempt::Deferred(format!(
                    "height {sequence} was filled by another proposer"
                )));
            }
            // Fewer answers than the threshold is an availability problem, not
            // a refusal: nobody said no, they just did not say anything. That
            // is what the backoff is for, so defer instead of failing the
            // caller on a link that may already be coming back.
            if answered < self.set().threshold {
                return Ok(Attempt::Deferred(format!(
                    "only {answered} of {} validators answered at height {sequence}",
                    self.set().members.len()
                )));
            }
            return Err(RoundError::Rejected(format!(
                "ledger proposal received only {} valid votes; threshold is {}",
                sigs.len(),
                self.set().threshold
            )));
        }
        let cert = QuorumCertificate {
            entry: expected,
            membership_hash: membership_hash(&self.set()).map_err(RoundError::rejected)?,
            signatures: sigs,
        };
        verify_certificate(&cert, &self.set()).map_err(RoundError::rejected)?;
        let committed = join_all(self.set().members.iter().map(|m| {
            let (http, cert) = (&self.http, &cert);
            async move {
                http.post(format!("{}/v1/ledger/commit", m.url.trim_end_matches('/')))
                    .json(cert)
                    .send()
                    .await
                    .is_ok_and(|r| r.status().is_success())
            }
        }))
        .await
        .into_iter()
        .filter(|ok| *ok)
        .count();
        if committed < self.set().threshold {
            return Err(RoundError::Committed(format!(
                "certificate formed but committed on only {committed} validators; run validator sync/recovery"
            )));
        };
        Ok(match deferred {
            Some(why) => Attempt::Deferred(why),
            None => Attempt::Settled(cert),
        })
    }
    /// Fetch a run of certificates, checked against the set that governed each.
    ///
    /// `at` is the set the caller believes was sitting when entry `from` was
    /// certified - not the set sitting now. History was signed by whoever held
    /// a seat at the time, so checking an old entry against today's members
    /// rejects the entire chain the moment anybody has ever joined or left.
    /// The set is walked forward here for the same reason an audit walks it:
    /// a membership change governs everything after it and nothing before.
    pub async fn fetch_certificates(
        &self,
        from: u64,
        limit: u64,
        at: &ValidatorSet,
    ) -> Result<Vec<QuorumCertificate>> {
        for m in &self.set().members {
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
                let mut active = at.clone();
                for c in &e.certificates {
                    verify_certificate(c, &active)?;
                    for t in &c.entry.transactions {
                        if let TransactionEvidence::MembershipChange(mc) = &t.evidence {
                            active = verify_membership_change(&active, mc)?;
                        }
                    }
                }
                return Ok(e.certificates);
            }
        }
        bail!("no validator could provide ledger entries")
    }
}
