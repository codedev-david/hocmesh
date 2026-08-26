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
    store::LedgerStore,
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
        .route("/v1/ledger/claim/{claim}", get(claim))
        .route("/v1/ledger/propose", post(propose))
        .route("/v1/ledger/commit", post(commit))
        .route("/v1/ledger/entries", get(entries))
        .with_state(app);
    let l = tokio::net::TcpListener::bind(listen).await?;
    tracing::info!(%listen,"hocMESH validator started");
    axum::serve(l, r).await?;
    Ok(())
}
async fn head(State(a): State<App>) -> Result<Json<HeadProof>, String> {
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
async fn ledger_state(State(a): State<App>) -> Result<Json<StateProof>, String> {
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
async fn balance(
    State(a): State<App>,
    Path(account): Path<String>,
) -> Result<Json<BalanceProof>, String> {
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

async fn claim(
    State(a): State<App>,
    Path(claim): Path<String>,
) -> Result<Json<ClaimProof>, String> {
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
async fn propose(State(a): State<App>, Json(r): Json<ProposalRequest>) -> Json<ProposalVote> {
    let result = (|| -> Result<ProposalVote> {
        let s = a
            .store
            .lock()
            .map_err(|_| anyhow::anyhow!("ledger lock poisoned"))?;
        let h = s.head(&a.set())?;
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
        s.lock_vote(e.sequence, &e.entry_hash)?;
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
            error: None,
        })
    })();
    Json(result.unwrap_or_else(|e| ProposalVote {
        accepted: false,
        validator_id: a.id.node_id(),
        sequence: 0,
        previous_hash: String::new(),
        entry_hash: String::new(),
        signature_b64: None,
        error: Some(e.to_string()),
    }))
}
async fn commit(
    State(a): State<App>,
    Json(c): Json<QuorumCertificate>,
) -> Result<Json<CommitResponse>, String> {
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
async fn entries(
    State(a): State<App>,
    Query(q): Query<EntriesQ>,
) -> Result<Json<EntriesResponse>, String> {
    let s = a.store.lock().map_err(|_| "lock".to_string())?;
    let certs = s
        .certificates_from(q.from.unwrap_or(1), q.limit.unwrap_or(500).min(5000))
        .map_err(|e| e.to_string())?;
    Ok(Json(EntriesResponse {
        certificates: certs,
    }))
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
