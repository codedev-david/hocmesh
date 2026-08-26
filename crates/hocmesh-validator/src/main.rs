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
        build_entry, claim_key, ledger_entry_signing_message, membership_hash,
        validate_transaction, validate_validator_set, verify_certificate,
    },
};
use serde::Deserialize;
use std::{
    fs,
    path::{Path as FsPath, PathBuf},
    sync::{Arc, Mutex},
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
    set: ValidatorSet,
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
    let set = load_set(validators)?;
    let id = NodeIdentity::load_or_create(home)?;
    if !set
        .members
        .iter()
        .any(|m| m.validator_id == id.node_id() && m.public_key_b64 == id.public_key_b64())
    {
        bail!("this validator identity is not in membership file")
    };
    let app = App {
        store: Arc::new(Mutex::new(LedgerStore::open(db)?)),
        id,
        set,
    };
    let r = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/ledger/head", get(head))
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
        .head(&a.set)
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
async fn balance(
    State(a): State<App>,
    Path(account): Path<String>,
) -> Result<Json<BalanceProof>, String> {
    let s = a.store.lock().map_err(|_| "lock".to_string())?;
    let b = s.balance(&account).map_err(|e| e.to_string())?;
    let (earned, spent) = s.activity(&account).map_err(|e| e.to_string())?;
    let h = s.head(&a.set).map_err(|e| e.to_string())?;
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
    let h = s.head(&a.set).map_err(|e| e.to_string())?;
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
        let ck = claim_key(&r.transaction);
        if s.has_claim(&ck)? {
            bail!("claim already settled: {ck}")
        };
        validate_reward_membership(&s, &r.transaction)?;
        validate_transaction(
            &r.transaction,
            |x| s.balance(x),
            a.set.community_issuance_limit_mcu,
        )?;
        let h = s.head(&a.set)?;
        let e = build_entry(h.sequence + 1, h.entry_hash.clone(), r.transaction)?;
        s.lock_vote(e.sequence, &e.entry_hash)?;
        let mh = membership_hash(&a.set)?;
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
    verify_certificate(&c, &a.set).map_err(|e| e.to_string())?;
    let mut s = a.store.lock().map_err(|_| "lock".to_string())?;
    let local = s.head(&a.set).map_err(|e| e.to_string())?;
    if c.entry.sequence <= local.sequence {
        if c.entry.sequence == local.sequence && c.entry.entry_hash == local.entry_hash {
            return Ok(Json(CommitResponse {
                committed: true,
                head: local,
            }));
        }
        return Err("conflicting/stale certificate".into());
    }
    validate_reward_membership(&s, &c.entry.transaction).map_err(|e| e.to_string())?;
    validate_transaction(
        &c.entry.transaction,
        |x| s.balance(x),
        a.set.community_issuance_limit_mcu,
    )
    .map_err(|e| e.to_string())?;
    s.apply(&c, &a.set).map_err(|e| e.to_string())?;
    let h = s.head(&a.set).map_err(|e| e.to_string())?;
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

fn validate_reward_membership(store: &LedgerStore, tx: &LedgerTransaction) -> Result<()> {
    if let TransactionEvidence::ProviderReward(e) = &tx.evidence {
        let Some((root_work, shards, system_funded, requester)) = store.reservation(&e.job_id)?
        else {
            bail!("reward references a job with no certified reservation")
        };
        if system_funded != e.system_funded {
            bail!("reward funding type does not match reservation")
        };
        if requester.as_deref() == Some(e.provider_auth.node_id.as_str()) {
            bail!("requester cannot receive a reward from its own paid job")
        };
        let parts = hocmesh_core::compute::split_work(&root_work, shards);
        let Some(expected) = parts.get(e.shard_index as usize) else {
            bail!("reward shard index is outside reservation")
        };
        if expected != &e.work {
            bail!("reward work is not the reserved shard")
        };
    }
    Ok(())
}
async fn sync(db: &str, validators: &str, batch: u64) -> Result<()> {
    let set = load_set(validators)?;
    let net = LedgerNetwork::new(set.clone())?;
    let mut store = LedgerStore::open(db)?;
    loop {
        let h = store.head(&set)?;
        let remote = net.head_quorum().await?;
        if h.sequence >= remote.sequence {
            println!("SYNC OK height={} head={}", h.sequence, h.entry_hash);
            break;
        }
        let certs = net.fetch_certificates(h.sequence + 1, batch.max(1)).await?;
        if certs.is_empty() {
            bail!("remote head is ahead but no entries returned")
        };
        for c in certs {
            store.apply(&c, &set)?;
        }
    }
    Ok(())
}
