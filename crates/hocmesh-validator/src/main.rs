use anyhow::{Context, Result, bail};
use axum::{
    Json, Router,
    extract::{Path, Query, State},
    routing::{get, post},
};
use clap::{Parser, Subcommand};
use hocmesh_core::identity::NodeIdentity;
use hocmesh_ledger::{
    network::LedgerNetwork,
    store::{LedgerSnapshot, LedgerStore},
    types::*,
    validate::{
        build_entry, checkpoint_signing_message, claim_key, ledger_entry_signing_message,
        membership_hash, validate_batch, validate_validator_set, verify_certificate,
    },
};
use serde::Deserialize;
use std::{
    fs,
    path::{Path as FsPath, PathBuf},
    sync::{Arc, Mutex, RwLock},
};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "hocmesh-validator",
    version,
    about = "hocMESH replicated CU ledger validator"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}
#[derive(Subcommand)]
enum Cmd {
    Serve {
        #[arg(long, default_value = "127.0.0.1:9101")]
        listen: String,
        #[arg(long, default_value = "validator.db")]
        db: String,
        #[arg(long, default_value = ".hocmesh-validator")]
        home: PathBuf,
        #[arg(long)]
        validators: String,
    },
    Id {
        #[arg(long, default_value = ".hocmesh-validator")]
        home: PathBuf,
    },
    Audit {
        #[arg(long, default_value = "validator.db")]
        db: String,
        #[arg(long)]
        validators: String,
    },
    Sync {
        #[arg(long, default_value = "validator.db")]
        db: String,
        #[arg(long)]
        validators: String,
        #[arg(long, default_value_t = 500)]
        batch: u64,
    },
    /// Ask the sitting set for a signed statement of the whole state and keep
    /// it, so audits and snapshots have a starting point later.
    Checkpoint {
        #[arg(long, default_value = "validator.db")]
        db: String,
        #[arg(long)]
        validators: String,
    },
    /// Writes the latest checkpoint and the state it vouches for to a file.
    ///
    /// The file is self-checking against the validator set, so it can be
    /// published anywhere without the route being part of the trust.
    Snapshot {
        #[arg(long, default_value = "validator.db")]
        db: String,
        #[arg(long)]
        validators: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Starts a fresh database from a snapshot instead of from genesis.
    ///
    /// `sync` then has only the entries above the checkpoint left to fetch,
    /// which is what keeps joining a long-running network affordable.
    Restore {
        #[arg(long, default_value = "validator.db")]
        db: String,
        #[arg(long)]
        validators: String,
        #[arg(long)]
        snapshot: PathBuf,
    },
}
#[derive(Clone)]
struct App {
    store: Arc<Mutex<LedgerStore>>,
    id: NodeIdentity,
    set: Arc<RwLock<ValidatorSet>>,
}
impl App {
    /// The set as the chain last left it.
    ///
    /// Read through a lock rather than held as a plain field because a
    /// certified membership change moves the set underneath a running
    /// validator, and one still signing against the set it booted with would
    /// be producing heads the rest of the quorum cannot verify.
    fn set(&self) -> ValidatorSet {
        self.set.read().expect("validator set lock").clone()
    }
}
#[derive(Deserialize)]
struct EntriesQ {
    from: Option<u64>,
    limit: Option<u64>,
}
#[derive(Deserialize)]
struct HistoryQ {
    before: Option<u64>,
    limit: Option<u32>,
}
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    match Cli::parse().cmd {
        Cmd::Serve {
            listen,
            db,
            home,
            validators,
        } => serve(&listen, &db, &home, &validators).await,
        Cmd::Id { home } => {
            let i = NodeIdentity::load_or_create(&home)?;
            println!("validator_id={}", i.node_id());
            println!("public_key_b64={}", i.public_key_b64());
            Ok(())
        }
        Cmd::Audit { db, validators } => {
            let set = load_set(&validators)?;
            let s = LedgerStore::open(&db)?;
            let h = s.audit(&set)?;
            println!("AUDIT OK height={} head={}", h.sequence, h.entry_hash);
            Ok(())
        }
        Cmd::Sync {
            db,
            validators,
            batch,
        } => sync(&db, &validators, batch).await,
        Cmd::Checkpoint { db, validators } => {
            let net = LedgerNetwork::new(load_set(&validators)?)?;
            // Against the set that is sitting now, not the one in the file:
            // a checkpoint signed by yesterday's membership proves nothing.
            net.refresh_set().await?;
            let store = LedgerStore::open(&db)?;
            let cp = net.checkpoint_quorum().await?;
            store.store_checkpoint(&cp, &net.set())?;
            println!(
                "CHECKPOINT OK height={} head={} state={} signatures={}",
                cp.head.sequence,
                cp.head.entry_hash,
                cp.state_hash,
                cp.signatures.len()
            );
            Ok(())
        }
        Cmd::Snapshot {
            db,
            validators,
            out,
        } => {
            let set = load_set(&validators)?;
            let store = LedgerStore::open(&db)?;
            let snap = store.snapshot(&set)?;
            fs::write(&out, serde_json::to_string_pretty(&snap)?)?;
            println!(
                "SNAPSHOT OK height={} head={} state={} accounts={}",
                snap.checkpoint.head.sequence,
                snap.checkpoint.head.entry_hash,
                snap.checkpoint.state_hash,
                snap.state.balances.len()
            );
            Ok(())
        }
        Cmd::Restore {
            db,
            validators,
            snapshot,
        } => {
            let set = load_set(&validators)?;
            let snap: LedgerSnapshot = serde_json::from_str(
                &fs::read_to_string(&snapshot)
                    .with_context(|| format!("reading {}", snapshot.display()))?,
            )?;
            let mut store = LedgerStore::open(&db)?;
            store.install_snapshot(&snap, &set)?;
            // Proving it lands where the signatures say costs one audit of
            // nothing, and it is the difference between a restore that worked
            // and a restore that only appeared to.
            let head = store.audit_from(&set, Some(&snap.checkpoint))?;
            println!(
                "RESTORE OK height={} head={}",
                head.sequence, head.entry_hash
            );
            Ok(())
        }
    }
}
fn load_set(path: &str) -> Result<ValidatorSet> {
    let set: ValidatorSet = serde_json::from_str(
        &fs::read_to_string(path).with_context(|| format!("reading {path}"))?,
    )?;
    validate_validator_set(&set)?;
    Ok(set)
}
async fn serve(listen: &str, db: &str, home: &FsPath, validators: &str) -> Result<()> {
    let file_set = load_set(validators)?;
    let store = LedgerStore::open(db)?;
    // The bootstrap file is the genesis set and nothing more. Once the quorum
    // has certified a change to membership, that is the set this node is
    // bound by, whatever the operator still has sitting on disk.
    let set = store.current_set()?.unwrap_or(file_set);
    let id = NodeIdentity::load_or_create(home)?;
    if !set
        .members
        .iter()
        .any(|m| m.validator_id == id.node_id() && m.public_key_b64 == id.public_key_b64())
    {
        bail!("this validator identity is not in the set the ledger currently recognises")
    };
    let app = App {
        store: Arc::new(Mutex::new(store)),
        id,
        set: Arc::new(RwLock::new(set)),
    };
    let r = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/ledger/head", get(head))
        .route("/v1/ledger/state", get(ledger_state))
        .route("/v1/ledger/balance/{account}", get(balance))
        .route("/v1/ledger/history/{account}", get(history))
        .route("/v1/ledger/claim/{claim}", get(claim))
        .route("/v1/ledger/prepare", post(prepare))
        .route("/v1/ledger/propose", post(propose))
        .route("/v1/ledger/commit", post(commit))
        .route("/v1/ledger/entries", get(entries))
        .with_state(app);
    let l = tokio::net::TcpListener::bind(listen).await?;
    tracing::info!(%listen,"hocMESH validator started");
    axum::serve(l, r).await?;
    Ok(())
}
/// A refusal that actually reads as one over HTTP.
///
/// Axum renders a bare `String` error as 200 OK with the message as the body,
/// which makes every refusal in this file indistinguishable from success to
/// anything that checks the status line - and the ledger client checks the
/// status line when it counts commits. A validator that rejected a
/// certificate was therefore reported back to the proposer as having stored
/// it. The wrapper exists so that "no" is transmitted as "no".
struct Refusal(String);

impl<E: std::fmt::Display> From<E> for Refusal {
    fn from(e: E) -> Self {
        Self(e.to_string())
    }
}

impl axum::response::IntoResponse for Refusal {
    fn into_response(self) -> axum::response::Response {
        (axum::http::StatusCode::BAD_REQUEST, self.0).into_response()
    }
}

type Answer<T> = Result<Json<T>, Refusal>;

async fn head(State(a): State<App>) -> Answer<HeadProof> {
    let h = a
        .store
        .lock()
        .map_err(|_| "lock".to_string())?
        .head(&a.set())
        .map_err(|e| e.to_string())?;
    let msg = format!(
        "hocmesh-head-v1|{}|{}|{}",
        h.membership_hash, h.sequence, h.entry_hash
    );
    Ok(Json(HeadProof {
        head: h,
        validator_id: a.id.node_id(),
        signature_b64: a.id.sign_bytes_b64(msg.as_bytes()),
    }))
}
/// This validator's own view of the whole ledger state, signed.
///
/// The signature is over exactly the message a checkpoint is checked against,
/// so a caller that collects a quorum of these has a checkpoint already.
async fn ledger_state(State(a): State<App>) -> Answer<StateProof> {
    let s = a.store.lock().map_err(|_| "lock".to_string())?;
    let head = s.head(&a.set()).map_err(|e| e.to_string())?;
    let state_hash = s
        .state()
        .and_then(|st| st.digest())
        .map_err(|e| e.to_string())?;
    drop(s);
    let msg = checkpoint_signing_message(
        &head.membership_hash,
        head.sequence,
        &head.entry_hash,
        &state_hash,
    );
    Ok(Json(StateProof {
        head,
        state_hash,
        validator_id: a.id.node_id(),
        signature_b64: a.id.sign_bytes_b64(msg.as_bytes()),
    }))
}
async fn balance(State(a): State<App>, Path(account): Path<String>) -> Answer<BalanceProof> {
    let s = a.store.lock().map_err(|_| "lock".to_string())?;
    let b = s.balance(&account).map_err(|e| e.to_string())?;
    let (earned, spent) = s.activity(&account).map_err(|e| e.to_string())?;
    let h = s.head(&a.set()).map_err(|e| e.to_string())?;
    let msg = format!(
        "hocmesh-balance-v1|{}|{}|{}|{}|{}|{}|{}",
        h.membership_hash, account, b, earned, spent, h.sequence, h.entry_hash
    );
    Ok(Json(BalanceProof {
        account_id: account,
        balance_mcu: b,
        earned_mcu: earned,
        spent_mcu: spent,
        head: h,
        validator_id: a.id.node_id(),
        signature_b64: a.id.sign_bytes_b64(msg.as_bytes()),
    }))
}

async fn claim(State(a): State<App>, Path(claim): Path<String>) -> Answer<ClaimProof> {
    let s = a.store.lock().map_err(|_| "lock".to_string())?;
    let detail = s.claim_detail(&claim).map_err(|e| e.to_string())?;
    let h = s.head(&a.set()).map_err(|e| e.to_string())?;
    let (sequence, entry_hash, certificate) = match detail {
        Some((seq, hash)) => {
            let cert = s.certificate_at(seq).map_err(|e| e.to_string())?;
            (Some(seq), Some(hash), cert)
        }
        None => (None, None, None),
    };
    let msg = format!(
        "hocmesh-claim-v1|{}|{}|{:?}|{:?}|{}|{}",
        h.membership_hash, claim, sequence, entry_hash, h.sequence, h.entry_hash
    );
    Ok(Json(ClaimProof {
        claim_key: claim,
        sequence,
        entry_hash,
        certificate,
        head: h,
        validator_id: a.id.node_id(),
        signature_b64: a.id.sign_bytes_b64(msg.as_bytes()),
    }))
}
/// Reserve a height for a proposer, and tell it what is already signed there.
///
/// This is the half that makes a contested height recoverable. A validator
/// that has already put its name behind an entry hands that entry back, and
/// the new proposer is then obliged to finish it - so taking a height away
/// from a stalled round can never change what the round was deciding.
async fn prepare(State(a): State<App>, Json(r): Json<PrepareRequest>) -> Json<PrepareVote> {
    let vote = |promised, accepted, promised_ballot, error| PrepareVote {
        promised,
        validator_id: a.id.node_id(),
        sequence: r.sequence,
        accepted,
        promised_ballot,
        error,
    };
    let s = match a.store.lock() {
        Ok(s) => s,
        Err(_) => return Json(vote(false, None, None, Some("ledger lock poisoned".into()))),
    };
    let result = (|| -> Result<Option<AcceptedProposal>> {
        let h = s.head(&a.set())?;
        if r.sequence != h.sequence + 1 {
            bail!(
                "prepare is for sequence {} but this validator is at {}",
                r.sequence,
                h.sequence
            )
        }
        s.promise(r.sequence, &r.ballot)
    })();
    // Either way the proposer is told which ballot holds this height, so a
    // client that lost knows what it has to beat rather than guessing.
    let held = s.promised_ballot(r.sequence).ok().flatten();
    Json(match result {
        Ok(accepted) => vote(true, accepted, held, None),
        Err(e) => vote(false, None, held, Some(e.to_string())),
    })
}

async fn propose(State(a): State<App>, Json(r): Json<ProposalRequest>) -> Json<ProposalVote> {
    let result = (|| -> Result<ProposalVote> {
        let s = a
            .store
            .lock()
            .map_err(|_| anyhow::anyhow!("ledger lock poisoned"))?;
        let h = s.head(&a.set())?;
        if r.sequence != h.sequence + 1 {
            bail!(
                "proposal is for sequence {} but this validator is at {}",
                r.sequence,
                h.sequence
            )
        }
        let mut settled = std::collections::HashSet::new();
        for t in &r.transactions {
            let ck = claim_key(t);
            if s.has_claim(&ck)? || !settled.insert(ck.clone()) {
                bail!("claim already settled: {ck}")
            };
            validate_settlement_membership(&s, t)?;
        }
        validate_batch(
            &r.transactions,
            &h.entry_hash,
            |x| s.balance(x),
            |j| s.inference_reservation(j),
            &a.set(),
        )?;
        let e = build_entry(h.sequence + 1, h.entry_hash.clone(), r.transactions)?;
        s.accept_ballot(e.sequence, &r.ballot, &e.entry_hash, &e.transactions)?;
        let mh = membership_hash(&a.set())?;
        let sig =
            a.id.sign_bytes_b64(ledger_entry_signing_message(&mh, &e.entry_hash).as_bytes());
        Ok(ProposalVote {
            accepted: true,
            validator_id: a.id.node_id(),
            sequence: e.sequence,
            previous_hash: e.previous_hash,
            entry_hash: e.entry_hash,
            signature_b64: Some(sig),
            promised_ballot: s.promised_ballot(r.sequence).ok().flatten(),
            head_sequence: Some(h.sequence),
            error: None,
        })
    })();
    Json(result.unwrap_or_else(|e| {
        // Refusals are where a proposer learns why it fell short, so say more
        // here than anywhere else: which ballot holds the height, and where
        // this chain actually ends. Both under one lock, because a proposer
        // reading them against each other wants them from one moment.
        let (promised_ballot, head_sequence) = a.store.lock().map_or((None, None), |s| {
            (
                s.promised_ballot(r.sequence).ok().flatten(),
                s.head(&a.set()).ok().map(|h| h.sequence),
            )
        });
        ProposalVote {
            accepted: false,
            validator_id: a.id.node_id(),
            sequence: 0,
            previous_hash: String::new(),
            entry_hash: String::new(),
            signature_b64: None,
            promised_ballot,
            head_sequence,
            error: Some(e.to_string()),
        }
    }))
}
async fn commit(State(a): State<App>, Json(c): Json<QuorumCertificate>) -> Answer<CommitResponse> {
    verify_certificate(&c, &a.set()).map_err(|e| e.to_string())?;
    let mut s = a.store.lock().map_err(|_| "lock".to_string())?;
    let local = s.head(&a.set()).map_err(|e| e.to_string())?;
    if c.entry.sequence <= local.sequence {
        if c.entry.sequence == local.sequence && c.entry.entry_hash == local.entry_hash {
            return Ok(Json(CommitResponse {
                committed: true,
                head: local,
            }));
        }
        return Err("conflicting/stale certificate".into());
    }
    for t in &c.entry.transactions {
        validate_settlement_membership(&s, t).map_err(|e| e.to_string())?;
    }
    validate_batch(
        &c.entry.transactions,
        &c.entry.previous_hash,
        |x| s.balance(x),
        |j| s.inference_reservation(j),
        &a.set(),
    )
    .map_err(|e| e.to_string())?;
    s.apply(&c, &a.set()).map_err(|e| e.to_string())?;
    // A change to the set takes effect the moment it is certified, so pick it
    // up before signing the head this same request is about to return.
    if let Some(next) = s.current_set().map_err(|e| e.to_string())? {
        *a.set.write().expect("validator set lock") = next;
    }
    let h = s.head(&a.set()).map_err(|e| e.to_string())?;
    Ok(Json(CommitResponse {
        committed: true,
        head: h,
    }))
}
async fn entries(State(a): State<App>, Query(q): Query<EntriesQ>) -> Answer<EntriesResponse> {
    let s = a.store.lock().map_err(|_| "lock".to_string())?;
    let certs = s
        .certificates_from(q.from.unwrap_or(1), q.limit.unwrap_or(500).min(5000))
        .map_err(|e| e.to_string())?;
    Ok(Json(EntriesResponse {
        certificates: certs,
    }))
}

/// An account's own postings, newest first.
///
/// Unsigned, unlike a balance: this is a convenience for reading, and anything
/// resting on it can be checked against the entries the chain already serves.
async fn history(
    State(a): State<App>,
    Path(account): Path<String>,
    Query(q): Query<HistoryQ>,
) -> Answer<AccountHistory> {
    let s = a.store.lock().map_err(|_| "lock".to_string())?;
    let page = s
        .history(&account, q.before, q.limit.unwrap_or(100))
        .map_err(|e| e.to_string())?;
    Ok(Json(page))
}

/// The reward and the refund for one shard are one claim seen from two sides,
/// so both are checked against the same reservation and the same window.
/// Replay in `hocmesh-ledger` enforces exactly these rules; this is the
/// propose-time copy, so a validator declines to sign what it would later
/// have to reject.
fn validate_settlement_membership(store: &LedgerStore, tx: &LedgerTransaction) -> Result<()> {
    let (job_id, shard_index, work, system_funded, payee, is_refund) = match &tx.evidence {
        TransactionEvidence::ProviderReward(e) => (
            &e.job_id,
            e.shard_index,
            &e.work,
            e.system_funded,
            Some(e.provider_auth.node_id.as_str()),
            false,
        ),
        TransactionEvidence::JobRefund(e) => (
            &e.job_id,
            e.shard_index,
            &e.work,
            e.system_funded,
            e.requester_auth.as_ref().map(|a| a.node_id.as_str()),
            true,
        ),
        TransactionEvidence::InferenceReward(_) | TransactionEvidence::InferenceRefund(_) => {
            return validate_inference_membership(store, tx);
        }
        _ => return Ok(()),
    };
    let Some((root_work, shards, reserved_funding, requester, reserved_at)) =
        store.reservation(job_id)?
    else {
        bail!("settlement references a job with no certified reservation")
    };
    if reserved_funding != system_funded {
        bail!("settlement funding type does not match reservation")
    };
    if is_refund {
        // The escrow returns where it came from, never to whoever asks.
        if requester.as_deref() != payee {
            bail!("refund pays someone other than the requester who reserved")
        }
    } else if requester.as_deref() == payee {
        bail!("requester cannot receive a reward from its own paid job")
    };
    let parts = hocmesh_core::compute::split_work(&root_work, shards);
    let Some(expected) = parts.get(shard_index as usize) else {
        bail!("settlement shard index is outside reservation")
    };
    if expected != work {
        bail!("settlement work is not the reserved shard")
    };
    // A shard settles once, and which way it settles is decided by the clock
    // rather than by whoever reaches the validators first.
    let deadline = reserved_at + hocmesh_protocol::SETTLEMENT_WINDOW_SECS;
    if is_refund && tx.created_at <= deadline {
        bail!("refund inside the shard's settlement window")
    }
    if !is_refund && tx.created_at > deadline {
        bail!("reward arrived after the shard's settlement window")
    }
    Ok(())
}
/// The same membership check as a shard settlement, for a batch of inference.
///
/// A validator cannot re-run the model, so what it checks is everything
/// *around* the answer: that the batch is one the reservation certified, that
/// the node claiming it is the node that was given it, that the amount is what
/// the signed billing prices the batch at, and that a reward and a refund for
/// one batch never share a moment.
fn validate_inference_membership(store: &LedgerStore, tx: &LedgerTransaction) -> Result<()> {
    let (job_id, assignment_id, batch_start, batch_end, amount, payee, is_refund) =
        match &tx.evidence {
            TransactionEvidence::InferenceReward(e) => (
                &e.job_id,
                &e.assignment_id,
                e.batch_start,
                e.batch_end,
                e.reward_mcu,
                e.provider_auth.node_id.as_str(),
                false,
            ),
            TransactionEvidence::InferenceRefund(e) => (
                &e.job_id,
                &e.assignment_id,
                e.batch_start,
                e.batch_end,
                e.refund_mcu,
                e.requester_auth.node_id.as_str(),
                true,
            ),
            _ => return Ok(()),
        };
    let Some(reservation) = store.inference_reservation(job_id)? else {
        bail!("inference settlement references a job with no certified reservation")
    };
    let Some(index) = (0..reservation.batches.len() as u32)
        .find(|i| hocmesh_protocol::inference_assignment_id(job_id, *i) == *assignment_id)
    else {
        bail!("inference settlement names a batch that was never certified")
    };
    let batch = &reservation.batches[index as usize];
    if batch.batch_start != batch_start || batch.batch_end != batch_end {
        bail!("inference settlement changes the bounds of the batch it claims")
    }
    if is_refund {
        // The escrow returns where it came from, never to whoever asks.
        if reservation.requester != payee {
            bail!("inference refund pays someone other than the requester")
        }
    } else {
        if batch.node_id != payee {
            bail!("inference reward paid to a node the batch was not assigned to")
        }
        if reservation.requester == payee {
            bail!("requester cannot receive a reward from its own paid job")
        }
    }
    let expected = hocmesh_core::compute::inference_batch_cost_mcu(
        &reservation.billing.prompt_bytes,
        batch_start,
        batch_end,
        reservation.billing.max_tokens,
        reservation.billing.parameter_count,
    );
    if expected != amount {
        bail!("inference settlement is not what the batch prices at")
    }
    // A batch settles once, and which way is decided by the clock.
    let deadline = reservation.reserved_at + hocmesh_protocol::SETTLEMENT_WINDOW_SECS;
    if is_refund && tx.created_at <= deadline {
        bail!("inference refund inside the batch settlement window")
    }
    if !is_refund && tx.created_at > deadline {
        bail!("inference reward arrived after the batch settlement window")
    }
    Ok(())
}
async fn sync(db: &str, validators: &str, batch: u64) -> Result<()> {
    let net = LedgerNetwork::new(load_set(validators)?)?;
    let mut store = LedgerStore::open(db)?;
    loop {
        // Whatever the chain has last handed over, not the bootstrap file. A
        // validator catching up across an admission has to check the entries
        // after it against the set that admitted, exactly as an audit does.
        let set = store.current_set()?.unwrap_or_else(|| net.set());
        let h = store.head(&set)?;
        net.refresh_set().await?;
        let remote = net.head_quorum().await?;
        if h.sequence >= remote.sequence {
            println!("SYNC OK height={} head={}", h.sequence, h.entry_hash);
            break;
        }
        let certs = net
            .fetch_certificates(h.sequence + 1, batch.max(1), &set)
            .await?;
        if certs.is_empty() {
            bail!("remote head is ahead but no entries returned")
        };
        let mut set = set;
        for c in certs {
            store.apply(&c, &set)?;
            if let Some(next) = store.current_set()? {
                set = next;
            }
        }
    }
    Ok(())
}
