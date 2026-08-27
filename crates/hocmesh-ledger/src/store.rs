use crate::{
    types::*,
    validate::{
        claim_key, membership_hash, validate_historical_transaction, verify_certificate,
        verify_checkpoint, verify_historical_evidence, verify_membership_change,
    },
};
use anyhow::{Context, Result, bail};
use hocmesh_protocol::{InferenceBilling, PricedBatch};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub struct LedgerStore {
    conn: Connection,
}

type ReservationRecord = (hocmesh_protocol::WorkSpec, u32, bool, Option<String>, i64);
pub use crate::types::InferenceReservation;

type InferenceRecord = (InferenceBilling, Vec<PricedBatch>, String, i64);

/// Everything an audit needs in order to start somewhere other than genesis.
///
/// Balances and settled claims are the obvious part; the open reservations
/// matter just as much, because a reward or a refund is only valid against
/// the reservation it draws on.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerState {
    pub balances: BTreeMap<String, i64>,
    pub claims: BTreeSet<String>,
    pub reservations: BTreeMap<String, ReservationRecord>,
    pub inference: BTreeMap<String, InferenceRecord>,
    /// Lifetime earned and spent, per account, which validators compare when
    /// they answer a balance query. It lives in the state because two nodes
    /// that agree on every balance can still disagree here — a node that
    /// started from a snapshot has no postings below it to add up.
    pub activity: BTreeMap<String, (i64, i64)>,
}

impl LedgerState {
    /// A hash that two nodes holding the same ledger state will agree on.
    ///
    /// Ordered containers throughout, and reservations reduced to the JSON
    /// they were indexed from, so the digest depends on the state itself and
    /// not on the order anybody happened to load it in.
    pub fn digest(&self) -> Result<String> {
        let balances: Vec<(&String, &i64)> =
            self.balances.iter().filter(|(_, v)| **v != 0).collect();
        let reservations = self
            .reservations
            .iter()
            .map(|(k, v)| Ok((k, serde_json::to_string(v)?)))
            .collect::<Result<Vec<_>>>()?;
        let inference = self
            .inference
            .iter()
            .map(|(k, v)| Ok((k, serde_json::to_string(v)?)))
            .collect::<Result<Vec<_>>>()?;
        let activity: Vec<(&String, &(i64, i64))> = self
            .activity
            .iter()
            .filter(|(_, (e, s))| *e != 0 || *s != 0)
            .collect();
        Ok(hocmesh_protocol::hash_json(&(
            "hocmesh-state-v2",
            &balances,
            &self.claims,
            &reservations,
            &inference,
            &activity,
        ))?)
    }
}

/// A ledger a newcomer can adopt without replaying the chain that produced it.
///
/// Nothing in the file is taken on trust. The certificate proves the head
/// entry was agreed; the checkpoint proves a quorum signed that head together
/// with a hash of the state it left behind; and the state has to hash to
/// exactly that. A file failing any of the three is refused, which is what
/// lets one travel by any route at all — a web server, a mirror, a USB stick
/// — without the route having to be trusted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerSnapshot {
    /// The entry the state stops at, so a bootstrapped store has a head to
    /// extend and peers have something to chain their next entry onto.
    pub certificate: QuorumCertificate,
    pub checkpoint: LedgerCheckpoint,
    pub state: LedgerState,
}
impl LedgerSnapshot {
    /// Checks the file against a validator set the reader already trusts.
    ///
    /// The set is supplied by the operator, not read out of the snapshot: a
    /// file that carried its own list of who to believe would prove nothing.
    pub fn verify(&self, set: &ValidatorSet) -> Result<()> {
        verify_certificate(&self.certificate, set)?;
        verify_checkpoint(&self.checkpoint, set)?;
        let entry = &self.certificate.entry;
        if entry.sequence != self.checkpoint.head.sequence
            || entry.entry_hash != self.checkpoint.head.entry_hash
        {
            bail!("snapshot certificate is for a different entry than its checkpoint")
        }
        let hash = self.state.digest()?;
        if hash != self.checkpoint.state_hash {
            bail!(
                "snapshot state hashes to {hash}, but the quorum signed {}",
                self.checkpoint.state_hash
            )
        }
        Ok(())
    }
}

fn sqlite_sequence(sequence: u64) -> Result<i64> {
    i64::try_from(sequence).context("ledger sequence exceeds SQLite INTEGER range")
}

fn ledger_sequence(sequence: i64) -> Result<u64> {
    u64::try_from(sequence).context("ledger sequence stored as negative INTEGER")
}

impl LedgerStore {
    pub fn open(path: &str) -> Result<Self> {
        let c = Connection::open(path).with_context(|| format!("opening ledger {path}"))?;
        c.pragma_update(None, "journal_mode", "WAL")?;
        c.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS certificates(
                sequence INTEGER PRIMARY KEY,
                entry_hash TEXT UNIQUE NOT NULL,
                certificate_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS balances(
                account_id TEXT PRIMARY KEY,
                balance_mcu INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS claims(
                claim_key TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS activity_baseline(
                account_id TEXT PRIMARY KEY,
                earned_mcu INTEGER NOT NULL,
                spent_mcu INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checkpoints(
                sequence INTEGER PRIMARY KEY,
                checkpoint_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS validator_set(
                sequence INTEGER PRIMARY KEY,
                set_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS votes(
                sequence INTEGER PRIMARY KEY,
                entry_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ballots(
                sequence INTEGER PRIMARY KEY,
                promised_json TEXT NOT NULL,
                accepted_json TEXT
            );
            CREATE TABLE IF NOT EXISTS account_activity(
                account_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                posting_index INTEGER NOT NULL,
                transaction_id TEXT NOT NULL,
                delta_mcu INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY(account_id, sequence, posting_index)
            );
            CREATE INDEX IF NOT EXISTS idx_account_activity_sequence
                ON account_activity(sequence);
            CREATE TABLE IF NOT EXISTS job_reservations(
                job_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL,
                work_json TEXT NOT NULL,
                shards INTEGER NOT NULL,
                system_funded INTEGER NOT NULL,
                requester_node_id TEXT,
                reserved_at INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS inference_reservations(
                job_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL,
                billing_json TEXT NOT NULL,
                batches_json TEXT NOT NULL,
                requester_node_id TEXT NOT NULL,
                reserved_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS assignment_rewards(
                assignment_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL,
                job_id TEXT NOT NULL,
                provider_node_id TEXT NOT NULL,
                reward_mcu INTEGER NOT NULL,
                system_funded INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_assignment_rewards_job
                ON assignment_rewards(job_id);
            "#,
        )?;
        // job_reservations is a rebuildable index, but a database written
        // before refunds existed has no reserved_at column and CREATE TABLE
        // IF NOT EXISTS will not add one. Adding it is a no-op the second
        // time, so the error for "already there" is the expected case.
        let _ = c.execute(
            "ALTER TABLE job_reservations ADD COLUMN reserved_at INTEGER NOT NULL DEFAULT 0",
            [],
        );
        let mut store = Self { conn: c };
        store.rebuild_indexes()?;
        Ok(store)
    }
    pub fn head(&self, set: &ValidatorSet) -> Result<LedgerHead> {
        let row: Option<(i64, String)> = self
            .conn
            .query_row(
                "SELECT sequence,entry_hash FROM certificates ORDER BY sequence DESC LIMIT 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let (sequence, entry_hash) = row.unwrap_or((0, "GENESIS".into()));
        Ok(LedgerHead {
            sequence: ledger_sequence(sequence)?,
            entry_hash,
            membership_hash: membership_hash(set)?,
        })
    }
    /// Reserve a height for one proposer and report what is already signed there.
    ///
    /// Handing back the accepted entry is the whole point: whoever takes the
    /// height next has to finish that entry rather than its own, so a round
    /// that was interrupted halfway can never be replaced by a different one.
    pub fn promise(&self, sequence: u64, ballot: &Ballot) -> Result<Option<AcceptedProposal>> {
        let sqlite_sequence = sqlite_sequence(sequence)?;
        let row: Option<(String, Option<String>)> = self
            .conn
            .query_row(
                "SELECT promised_json,accepted_json FROM ballots WHERE sequence=?1",
                params![sqlite_sequence],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let accepted = match &row {
            Some((promised_json, accepted_json)) => {
                let promised: Ballot = serde_json::from_str(promised_json)?;
                if promised.outranks(ballot) {
                    bail!("sequence {sequence} is promised to ballot {promised}")
                }
                accepted_json
                    .as_deref()
                    .map(serde_json::from_str)
                    .transpose()?
            }
            None => None,
        };
        self.conn.execute(
            "INSERT INTO ballots(sequence,promised_json) VALUES(?1,?2)
             ON CONFLICT(sequence) DO UPDATE SET promised_json=excluded.promised_json",
            params![sqlite_sequence, serde_json::to_string(ballot)?],
        )?;
        Ok(accepted)
    }
    /// The ballot currently holding a height, if any.
    pub fn promised_ballot(&self, sequence: u64) -> Result<Option<Ballot>> {
        let promised: Option<String> = self
            .conn
            .query_row(
                "SELECT promised_json FROM ballots WHERE sequence=?1",
                params![sqlite_sequence(sequence)?],
                |r| r.get(0),
            )
            .optional()?;
        promised.map(|p| Ok(serde_json::from_str(&p)?)).transpose()
    }
    /// Put this validator's name behind one batch at one height.
    ///
    /// Refused once a later proposer has taken the height, so the entry the
    /// set signs is always the one the newest ballot drove.
    pub fn accept_ballot(
        &self,
        sequence: u64,
        ballot: &Ballot,
        entry_hash: &str,
        transactions: &[LedgerTransaction],
    ) -> Result<()> {
        let sqlite_sequence = sqlite_sequence(sequence)?;
        let promised: Option<String> = self
            .conn
            .query_row(
                "SELECT promised_json FROM ballots WHERE sequence=?1",
                params![sqlite_sequence],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(promised) = promised {
            let promised: Ballot = serde_json::from_str(&promised)?;
            if promised.outranks(ballot) {
                bail!("sequence {sequence} has moved on to ballot {promised}")
            }
        }
        // A height that already carries a committed entry is settled; nothing
        // any proposer says can move it.
        let committed: Option<String> = self
            .conn
            .query_row(
                "SELECT entry_hash FROM votes WHERE sequence=?1",
                params![sqlite_sequence],
                |r| r.get(0),
            )
            .optional()?;
        if committed.is_some_and(|h| h != entry_hash) {
            bail!("sequence {sequence} already carries a different committed entry")
        }
        let accepted = AcceptedProposal {
            ballot: ballot.clone(),
            entry_hash: entry_hash.to_string(),
            transactions: transactions.to_vec(),
        };
        self.conn.execute(
            "INSERT INTO ballots(sequence,promised_json,accepted_json) VALUES(?1,?2,?3)
             ON CONFLICT(sequence) DO UPDATE SET
                 promised_json=excluded.promised_json,
                 accepted_json=excluded.accepted_json",
            params![
                sqlite_sequence,
                serde_json::to_string(ballot)?,
                serde_json::to_string(&accepted)?
            ],
        )?;
        Ok(())
    }
    pub fn claim_detail(&self, key: &str) -> Result<Option<(u64, String)>> {
        self.conn
            .query_row(
                "SELECT c.sequence,c.entry_hash FROM claims x JOIN certificates c ON c.sequence=x.sequence WHERE x.claim_key=?1",
                params![key],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)),
            )
            .optional()?
            .map(|(sequence, entry_hash)| Ok((ledger_sequence(sequence)?, entry_hash)))
            .transpose()
    }
    pub fn certificate_at(&self, sequence: u64) -> Result<Option<QuorumCertificate>> {
        let sqlite_sequence = sqlite_sequence(sequence)?;
        let raw: Option<String> = self
            .conn
            .query_row(
                "SELECT certificate_json FROM certificates WHERE sequence=?1",
                params![sqlite_sequence],
                |r| r.get(0),
            )
            .optional()?;
        raw.map(|v| serde_json::from_str(&v).map_err(anyhow::Error::from))
            .transpose()
    }
    pub fn has_claim(&self, key: &str) -> Result<bool> {
        Ok(self
            .conn
            .query_row(
                "SELECT 1 FROM claims WHERE claim_key=?1",
                params![key],
                |r| r.get::<_, i64>(0),
            )
            .optional()?
            .is_some())
    }
    /// The certified facts about an inference job: what it was billed for and
    /// which batch the coordinator promised to whom.
    pub fn inference_reservation(&self, job_id: &str) -> Result<Option<InferenceReservation>> {
        self.conn
            .query_row(
                "SELECT billing_json,batches_json,requester_node_id,reserved_at FROM inference_reservations WHERE job_id=?1",
                params![job_id],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, i64>(3)?,
                    ))
                },
            )
            .optional()?
            .map(|(billing, batches, requester, reserved_at)| {
                Ok(InferenceReservation {
                    job_id: job_id.to_string(),
                    billing: serde_json::from_str(&billing)?,
                    batches: serde_json::from_str(&batches)?,
                    requester,
                    reserved_at,
                })
            })
            .transpose()
    }

    pub fn reservation(&self, job_id: &str) -> Result<Option<ReservationRecord>> {
        self.conn
            .query_row(
                "SELECT work_json,shards,system_funded,requester_node_id,reserved_at FROM job_reservations WHERE job_id=?1",
                params![job_id],
                |r| {
                    let work_json: String = r.get(0)?;
                    Ok((
                        work_json,
                        r.get::<_, i64>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, i64>(4)?,
                    ))
                },
            )
            .optional()?
            .map(|(work_json, shards, system_funded, requester, reserved_at)| {
                let shards = u32::try_from(shards).context("stored shard count is outside u32")?;
                let work = serde_json::from_str(&work_json)?;
                Ok((work, shards, system_funded != 0, requester, reserved_at))
            })
            .transpose()
    }
    pub fn activity(&self, a: &str) -> Result<(i64, i64)> {
        let (earned, spent): (i64, i64) = self.conn.query_row(
            "SELECT COALESCE(SUM(CASE WHEN delta_mcu > 0 THEN delta_mcu ELSE 0 END),0),
                    COALESCE(SUM(CASE WHEN delta_mcu < 0 THEN -delta_mcu ELSE 0 END),0)
             FROM account_activity WHERE account_id=?1",
            params![a],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        let (base_e, base_s) = self.activity_baseline(a)?;
        Ok((earned + base_e, spent + base_s))
    }
    /// A page of an account's postings, newest first.
    ///
    /// `before` excludes everything at or above that sequence, so paging walks
    /// backwards through the chain. The primary key on
    /// `(account_id, sequence, posting_index)` is what makes this a seek rather
    /// than a scan of every posting the ledger has ever written.
    pub fn history(
        &self,
        account: &str,
        before: Option<u64>,
        limit: u32,
    ) -> Result<AccountHistory> {
        let limit = limit.clamp(1, 500);
        let ceiling = before.unwrap_or(u64::MAX).min(i64::MAX as u64) as i64;
        let mut q = self.conn.prepare(
            "SELECT sequence,posting_index,transaction_id,delta_mcu,created_at
             FROM account_activity WHERE account_id=?1 AND sequence<?2
             ORDER BY sequence DESC, posting_index DESC LIMIT ?3",
        )?;
        let mut entries: Vec<AccountHistoryEntry> = q
            .query_map(params![account, ceiling, limit], |r| {
                Ok(AccountHistoryEntry {
                    sequence: r.get::<_, i64>(0)? as u64,
                    posting_index: r.get::<_, i64>(1)? as u32,
                    transaction_id: r.get(2)?,
                    delta_mcu: r.get(3)?,
                    created_at: r.get(4)?,
                })
            })?
            .collect::<std::result::Result<_, _>>()?;

        // A page must never stop in the middle of one entry's postings, or the
        // cursor -- which is a sequence -- would step over whatever was left.
        // Either the page holds more than one sequence, in which case the last
        // one is dropped and comes back whole next time, or it holds exactly
        // one and the rest of it is fetched now.
        let more = entries.len() as u32 == limit;
        if more {
            let last = entries[entries.len() - 1].clone();
            if entries[0].sequence == last.sequence {
                entries.extend(self.postings_at(account, last.sequence, last.posting_index)?);
            } else {
                entries.retain(|e| e.sequence != last.sequence);
            }
        }
        // The cursor is whatever survived, not whatever the page happened to
        // hold: dropping a partial entry shortens the page below `limit`, and
        // reading fullness off the trimmed page would end history right there.
        let next_before = match entries.last() {
            Some(e) if more && self.has_history_below(account, e.sequence)? => Some(e.sequence),
            _ => None,
        };
        Ok(AccountHistory {
            account_id: account.to_string(),
            entries,
            next_before,
        })
    }

    /// The postings for one account inside one entry, below a given index.
    ///
    /// Only ever called to finish a page that landed inside an entry, so the
    /// count is whatever that single transaction posted, not a page's worth.
    fn postings_at(&self, account: &str, seq: u64, below: u32) -> Result<Vec<AccountHistoryEntry>> {
        let mut q = self.conn.prepare(
            "SELECT sequence,posting_index,transaction_id,delta_mcu,created_at
             FROM account_activity
             WHERE account_id=?1 AND sequence=?2 AND posting_index<?3
             ORDER BY posting_index DESC",
        )?;
        Ok(q.query_map(params![account, seq as i64, below], |r| {
            Ok(AccountHistoryEntry {
                sequence: r.get::<_, i64>(0)? as u64,
                posting_index: r.get::<_, i64>(1)? as u32,
                transaction_id: r.get(2)?,
                delta_mcu: r.get(3)?,
                created_at: r.get(4)?,
            })
        })?
        .collect::<std::result::Result<_, _>>()?)
    }

    /// Whether anything older than `before` is still on this node.
    ///
    /// A full page is not evidence that more exists, and a cursor that leads
    /// to an empty page makes a caller walk one more round for nothing.
    fn has_history_below(&self, account: &str, before: u64) -> Result<bool> {
        let mut q = self.conn.prepare(
            "SELECT 1 FROM account_activity WHERE account_id=?1 AND sequence<?2 LIMIT 1",
        )?;
        Ok(q.exists(params![account, before.min(i64::MAX as u64) as i64])?)
    }
    /// What a snapshot carried in for this account before the first posting
    /// this store holds. Zero for anyone who replayed the chain from genesis.
    fn activity_baseline(&self, a: &str) -> Result<(i64, i64)> {
        Ok(self
            .conn
            .query_row(
                "SELECT earned_mcu,spent_mcu FROM activity_baseline WHERE account_id=?1",
                params![a],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?
            .unwrap_or((0, 0)))
    }
    pub fn balance(&self, a: &str) -> Result<i64> {
        Ok(self
            .conn
            .query_row(
                "SELECT balance_mcu FROM balances WHERE account_id=?1",
                params![a],
                |r| r.get(0),
            )
            .optional()?
            .unwrap_or(0))
    }
    pub fn apply(&mut self, cert: &QuorumCertificate, set: &ValidatorSet) -> Result<()> {
        verify_certificate(cert, set)?;
        let mut next_set: Option<ValidatorSet> = None;
        for t in &cert.entry.transactions {
            verify_historical_evidence(t, &cert.entry.previous_hash, &cert.signatures)?;
            if let TransactionEvidence::MembershipChange(e) = &t.evidence {
                next_set = Some(verify_membership_change(
                    next_set.as_ref().unwrap_or(set),
                    e,
                )?);
            }
        }
        let h = self.head(set)?;
        if cert.entry.sequence != h.sequence + 1 || cert.entry.previous_hash != h.entry_hash {
            bail!("certificate does not extend local head")
        };
        for t in &cert.entry.transactions {
            let ck = claim_key(t);
            if self.has_claim(&ck)? {
                bail!("ledger claim already settled: {ck}")
            };
        }
        let tx = self.conn.transaction()?;
        for t in &cert.entry.transactions {
            for p in &t.postings {
                let cur: i64 = tx
                    .query_row(
                        "SELECT balance_mcu FROM balances WHERE account_id=?1",
                        params![p.account_id],
                        |r| r.get(0),
                    )
                    .optional()?
                    .unwrap_or(0);
                let next = cur
                    .checked_add(p.delta_mcu)
                    .ok_or_else(|| anyhow::anyhow!("balance overflow"))?;
                if next < 0
                    && p.account_id != COMMUNITY_ISSUANCE_ACCOUNT
                    && !p.account_id.starts_with("hocmesh:escrow:")
                {
                    bail!("negative user balance")
                };
                if next < 0 && p.account_id.starts_with("hocmesh:escrow:") {
                    bail!("negative escrow balance")
                };
                tx.execute("INSERT INTO balances(account_id,balance_mcu) VALUES(?1,?2) ON CONFLICT(account_id) DO UPDATE SET balance_mcu=excluded.balance_mcu",params![p.account_id,next])?;
            }
        }
        tx.execute(
            "INSERT INTO certificates(sequence,entry_hash,certificate_json) VALUES(?1,?2,?3)",
            params![
                sqlite_sequence(cert.entry.sequence)?,
                cert.entry.entry_hash,
                serde_json::to_string(cert)?
            ],
        )?;
        for t in &cert.entry.transactions {
            tx.execute(
                "INSERT INTO claims(claim_key,sequence) VALUES(?1,?2)",
                params![claim_key(t), sqlite_sequence(cert.entry.sequence)?],
            )?;
        }
        tx.execute(
            "INSERT OR IGNORE INTO votes(sequence,entry_hash) VALUES(?1,?2)",
            params![sqlite_sequence(cert.entry.sequence)?, cert.entry.entry_hash],
        )?;
        // The height is decided, so the attempts that fought over it are only
        // history now - and the batches they carried are not small.
        tx.execute(
            "DELETE FROM ballots WHERE sequence<=?1",
            params![sqlite_sequence(cert.entry.sequence)?],
        )?;
        index_certificate(&tx, cert)?;
        if let Some(next) = &next_set {
            tx.execute(
                "INSERT OR REPLACE INTO validator_set(sequence,set_json) VALUES(?1,?2)",
                params![
                    sqlite_sequence(cert.entry.sequence)?,
                    serde_json::to_string(next)?
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }
    /// The set the chain recognised at a given height.
    ///
    /// An audit that starts from a checkpoint has to verify that checkpoint's
    /// signatures against the set that was sitting when it was signed, not
    /// against whoever holds the seats today.
    pub fn set_at(&self, sequence: u64) -> Result<Option<ValidatorSet>> {
        let json: Option<String> = self
            .conn
            .query_row(
                "SELECT set_json FROM validator_set WHERE sequence<=?1 ORDER BY sequence DESC LIMIT 1",
                params![sqlite_sequence(sequence)?],
                |r| r.get(0),
            )
            .optional()?;
        Ok(match json {
            Some(j) => Some(serde_json::from_str(&j)?),
            None => None,
        })
    }
    /// The validator set as the chain last changed it, if it ever has.
    ///
    /// A node that has been running is bound by what the quorum certified,
    /// not by whatever the operator last wrote in the bootstrap file. That
    /// file is the genesis set and nothing more: it stops being the authority
    /// the moment a membership change is agreed.
    pub fn current_set(&self) -> Result<Option<ValidatorSet>> {
        let json: Option<String> = self
            .conn
            .query_row(
                "SELECT set_json FROM validator_set ORDER BY sequence DESC LIMIT 1",
                [],
                |r| r.get(0),
            )
            .optional()?;
        Ok(match json {
            Some(j) => Some(serde_json::from_str(&j)?),
            None => None,
        })
    }
    pub fn certificates_from(&self, from: u64, limit: u64) -> Result<Vec<QuorumCertificate>> {
        let from = sqlite_sequence(from)?;
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        let mut st = self.conn.prepare(
            "SELECT certificate_json FROM certificates WHERE sequence>=?1 ORDER BY sequence LIMIT ?2",
        )?;
        let rows = st.query_map(params![from, limit], |r| r.get::<_, String>(0))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(serde_json::from_str(&r?)?)
        }
        Ok(out)
    }

    /// Rebuilds the derived tables from the certificates that produced them.
    ///
    /// Only above the newest checkpoint. Below one the certificates may have
    /// been pruned, and these rows are then the only surviving record of them:
    /// rebuilding from a shortened history would quietly erase state the
    /// quorum has already signed for.
    fn rebuild_indexes(&mut self) -> Result<()> {
        let floor = self.latest_checkpoint()?.map_or(0, |c| c.head.sequence);
        let certs = self.certificates_from(floor + 1, u64::MAX)?;
        let floor = sqlite_sequence(floor)?;
        let tx = self.conn.transaction()?;
        for table in [
            "account_activity",
            "job_reservations",
            "inference_reservations",
            "assignment_rewards",
        ] {
            tx.execute(
                &format!("DELETE FROM {table} WHERE sequence>?1"),
                params![floor],
            )?;
        }
        for cert in &certs {
            index_certificate(&tx, cert)?;
        }
        tx.commit()?;
        Ok(())
    }
    /// The whole state this store currently holds, as an audit would build it.
    ///
    /// Read from the tables the store maintains incrementally, so it can be
    /// compared against a replay - if the two ever disagree, one of the two
    /// paths has a bug, and the digest is what makes that visible.
    pub fn state(&self) -> Result<LedgerState> {
        let mut state = LedgerState::default();
        let mut q = self
            .conn
            .prepare("SELECT account_id,balance_mcu FROM balances")?;
        for row in q.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))? {
            let (a, b) = row?;
            state.balances.insert(a, b);
        }
        let mut q = self.conn.prepare("SELECT claim_key FROM claims")?;
        for row in q.query_map([], |r| r.get::<_, String>(0))? {
            state.claims.insert(row?);
        }
        let mut q = self.conn.prepare(
            "SELECT job_id,work_json,shards,system_funded,requester_node_id,reserved_at
             FROM job_reservations",
        )?;
        for row in q.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, i64>(5)?,
            ))
        })? {
            let (job, work_json, shards, funded, requester, at) = row?;
            let shards = u32::try_from(shards).context("stored shard count is outside u32")?;
            state.reservations.insert(
                job,
                (
                    serde_json::from_str(&work_json)?,
                    shards,
                    funded != 0,
                    requester,
                    at,
                ),
            );
        }
        let mut q = self.conn.prepare(
            "SELECT job_id,billing_json,batches_json,requester_node_id,reserved_at
             FROM inference_reservations",
        )?;
        for row in q.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, i64>(4)?,
            ))
        })? {
            let (job, billing, batches, requester, at) = row?;
            state.inference.insert(
                job,
                (
                    serde_json::from_str(&billing)?,
                    serde_json::from_str(&batches)?,
                    requester,
                    at,
                ),
            );
        }
        let mut q = self.conn.prepare(
            "SELECT account_id,
                    COALESCE(SUM(CASE WHEN delta_mcu>0 THEN delta_mcu ELSE 0 END),0),
                    COALESCE(SUM(CASE WHEN delta_mcu<0 THEN -delta_mcu ELSE 0 END),0)
             FROM account_activity GROUP BY account_id",
        )?;
        for row in q.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
            ))
        })? {
            let (a, e, s) = row?;
            state.activity.insert(a, (e, s));
        }
        // Whatever a snapshot handed over sits underneath the postings this
        // store actually holds, so it has to be added in here too or a
        // bootstrapped node would disagree with its peers about the same
        // account and split every balance quorum it took part in.
        let mut q = self
            .conn
            .prepare("SELECT account_id,earned_mcu,spent_mcu FROM activity_baseline")?;
        for row in q.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
            ))
        })? {
            let (a, e, s) = row?;
            let slot = state.activity.entry(a).or_default();
            slot.0 += e;
            slot.1 += s;
        }
        Ok(state)
    }
    /// The state as it stood at `sequence`, undone from the state held now.
    ///
    /// Rewinding needs only the certificates above that height, which is the
    /// whole point: it lets an audit begin at a checkpoint without ever
    /// holding the history below it.
    fn rewind_to(&self, sequence: u64) -> Result<LedgerState> {
        let mut state = self.state()?;
        for c in self.certificates_from(sequence + 1, u64::MAX)? {
            for t in &c.entry.transactions {
                state.claims.remove(&claim_key(t));
                for p in &t.postings {
                    *state.balances.entry(p.account_id.clone()).or_default() -= p.delta_mcu;
                    let a = state.activity.entry(p.account_id.clone()).or_default();
                    if p.delta_mcu > 0 {
                        a.0 -= p.delta_mcu;
                    } else {
                        a.1 += p.delta_mcu;
                    }
                }
            }
        }
        // The reservation tables carry the height they were created at, so
        // undoing them is a matter of dropping the ones that came later.
        let at = sqlite_sequence(sequence)?;
        for job in self.jobs_reserved_after("job_reservations", at)? {
            state.reservations.remove(&job);
        }
        for job in self.jobs_reserved_after("inference_reservations", at)? {
            state.inference.remove(&job);
        }
        Ok(state)
    }
    fn jobs_reserved_after(&self, table: &str, sequence: i64) -> Result<Vec<String>> {
        // `table` is one of two literals chosen here, never anything a caller
        // supplies, so there is nothing for an injected string to reach.
        let mut q = self
            .conn
            .prepare(&format!("SELECT job_id FROM {table} WHERE sequence>?1"))?;
        let rows = q.query_map(params![sequence], |r| r.get::<_, String>(0))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }
    /// Records a checkpoint, but only one this store can reproduce itself.
    ///
    /// A quorum signature says the network agrees; replaying against it says
    /// this node agrees too. Storing one without the second check would mean
    /// trusting the signatures to paper over local corruption.
    pub fn store_checkpoint(&self, cp: &LedgerCheckpoint, set: &ValidatorSet) -> Result<()> {
        self.audit_from(set, Some(cp))?;
        self.conn.execute(
            "INSERT OR REPLACE INTO checkpoints(sequence,checkpoint_json) VALUES(?1,?2)",
            params![
                sqlite_sequence(cp.head.sequence)?,
                serde_json::to_string(cp)?
            ],
        )?;
        Ok(())
    }
    /// The highest checkpoint this store has accepted, if any.
    pub fn latest_checkpoint(&self) -> Result<Option<LedgerCheckpoint>> {
        let json: Option<String> = self
            .conn
            .query_row(
                "SELECT checkpoint_json FROM checkpoints ORDER BY sequence DESC LIMIT 1",
                [],
                |r| r.get(0),
            )
            .optional()?;
        json.map(|j| Ok(serde_json::from_str(&j)?)).transpose()
    }
    /// Packages the highest checkpoint this store holds, with the state it
    /// vouches for, into something another operator can start from.
    ///
    /// The state is rewound to the checkpoint rather than taken as it stands,
    /// so a store that has kept running past its last checkpoint still exports
    /// exactly what the quorum put its name to and nothing after it.
    pub fn snapshot(&self, set: &ValidatorSet) -> Result<LedgerSnapshot> {
        let Some(checkpoint) = self.latest_checkpoint()? else {
            bail!("no checkpoint to build a snapshot from")
        };
        let Some(certificate) = self.certificate_at(checkpoint.head.sequence)? else {
            bail!(
                "the certificate for checkpoint height {} is not held here",
                checkpoint.head.sequence
            )
        };
        let state = self.rewind_to(checkpoint.head.sequence)?;
        let snapshot = LedgerSnapshot {
            certificate,
            checkpoint,
            state,
        };
        // Refusing to hand out a file that would not survive being read is
        // cheaper than every reader discovering it separately.
        snapshot.verify(set)?;
        Ok(snapshot)
    }
    /// Adopts a verified snapshot into a store that has no chain of its own.
    ///
    /// This is deliberately not a merge. A store that already holds entries
    /// has its own head and its own history, and quietly replacing either
    /// would turn a bootstrap tool into a way of rewriting a ledger, so it is
    /// refused instead.
    pub fn install_snapshot(&mut self, snap: &LedgerSnapshot, set: &ValidatorSet) -> Result<()> {
        snap.verify(set)?;
        let held = self.head(set)?;
        if held.sequence != 0 {
            bail!(
                "this store already holds a chain up to height {}; a snapshot is for a store with none",
                held.sequence
            )
        }
        let at = sqlite_sequence(snap.checkpoint.head.sequence)?;
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO certificates(sequence,entry_hash,certificate_json) VALUES(?1,?2,?3)",
            params![
                at,
                snap.certificate.entry.entry_hash,
                serde_json::to_string(&snap.certificate)?
            ],
        )?;
        for (account, balance) in &snap.state.balances {
            tx.execute(
                "INSERT OR REPLACE INTO balances(account_id,balance_mcu) VALUES(?1,?2)",
                params![account, balance],
            )?;
        }
        for claim in &snap.state.claims {
            tx.execute(
                "INSERT OR REPLACE INTO claims(claim_key,sequence) VALUES(?1,?2)",
                params![claim, at],
            )?;
        }
        for (account, (earned, spent)) in &snap.state.activity {
            tx.execute(
                "INSERT OR REPLACE INTO activity_baseline(account_id,earned_mcu,spent_mcu)
                 VALUES(?1,?2,?3)",
                params![account, earned, spent],
            )?;
        }
        for (job, (work, shards, funded, requester, at_ms)) in &snap.state.reservations {
            tx.execute(
                "INSERT OR REPLACE INTO job_reservations(job_id,sequence,work_json,shards,system_funded,requester_node_id,reserved_at)
                 VALUES(?1,?2,?3,?4,?5,?6,?7)",
                params![
                    job,
                    at,
                    serde_json::to_string(work)?,
                    shards,
                    *funded as i64,
                    requester,
                    at_ms
                ],
            )?;
        }
        for (job, (billing, batches, requester, at_ms)) in &snap.state.inference {
            tx.execute(
                "INSERT OR REPLACE INTO inference_reservations(job_id,sequence,billing_json,batches_json,requester_node_id,reserved_at)
                 VALUES(?1,?2,?3,?4,?5,?6)",
                params![
                    job,
                    at,
                    serde_json::to_string(billing)?,
                    serde_json::to_string(batches)?,
                    requester,
                    at_ms
                ],
            )?;
        }
        tx.execute(
            "INSERT OR REPLACE INTO checkpoints(sequence,checkpoint_json) VALUES(?1,?2)",
            params![at, serde_json::to_string(&snap.checkpoint)?],
        )?;
        tx.commit()?;
        Ok(())
    }
    /// Discards the certificates a checkpoint has made unnecessary.
    ///
    /// The state they produced is kept - it is the state the checkpoint
    /// vouches for. What goes is the evidence for how it was reached, which
    /// is what makes a long-lived ledger's disk use bounded.
    pub fn prune_below_checkpoint(&self, set: &ValidatorSet) -> Result<usize> {
        let Some(cp) = self.latest_checkpoint()? else {
            bail!("no checkpoint to prune to")
        };
        self.audit_from(set, Some(&cp))?;
        // The checkpoint's own entry stays, so `head` still has a certificate
        // to read and a pruned node keeps reporting the height it is really
        // at. `account_activity` stays too: earned and spent totals are part
        // of the balance proofs validators have to agree on, and a pruned node
        // whose totals had been reset would disagree with every node that had
        // not pruned.
        let n = self.conn.execute(
            "DELETE FROM certificates WHERE sequence<?1",
            params![sqlite_sequence(cp.head.sequence)?],
        )?;
        Ok(n)
    }
    /// Replays the whole ledger from genesis and checks every rule again.
    pub fn audit(&self, set: &ValidatorSet) -> Result<LedgerHead> {
        self.audit_from(set, None)
    }
    /// Replays the ledger, optionally starting from a quorum-signed checkpoint.
    ///
    /// From genesis this is the strongest check there is and the only one
    /// available on day one. From a checkpoint it costs whatever has happened
    /// since, which is what keeps auditing possible on a ledger that has been
    /// running for years.
    pub fn audit_from(
        &self,
        set: &ValidatorSet,
        from: Option<&LedgerCheckpoint>,
    ) -> Result<LedgerHead> {
        let (mut seq, mut prev, mut state) = match from {
            None => (0, "GENESIS".to_string(), LedgerState::default()),
            Some(cp) => {
                verify_checkpoint(cp, set)?;
                let state = self.rewind_to(cp.head.sequence)?;
                let local = state.digest()?;
                if local != cp.state_hash {
                    bail!(
                        "checkpoint mismatch at sequence {}: local state {local}, checkpoint state {}, local head {}",
                        cp.head.sequence,
                        cp.state_hash,
                        self.head(set)?.sequence
                    )
                };
                (cp.head.sequence, cp.head.entry_hash.clone(), state)
            }
        };
        let certs = self.certificates_from(seq + 1, u64::MAX)?;
        if let (None, Some(first)) = (from, certs.first())
            && first.entry.sequence > 1
        {
            bail!(
                "this ledger has been pruned below sequence {}; audit from a checkpoint instead",
                first.entry.sequence
            )
        };
        let LedgerState {
            balances,
            claims,
            reservations,
            inference,
            activity,
        } = &mut state;
        let mut active = set.clone();
        for c in certs {
            verify_certificate(&c, &active)?;
            if c.entry.sequence != seq + 1 || c.entry.previous_hash != prev {
                bail!("broken chain at sequence {}", c.entry.sequence)
            };
            for txn in &c.entry.transactions {
                let ck = claim_key(txn);
                if !claims.insert(ck.clone()) {
                    bail!("duplicate ledger claim during audit: {ck}")
                };
                validate_historical_transaction(
                    txn,
                    &c.entry.previous_hash,
                    &c.signatures,
                    |a| Ok(*balances.get(a).unwrap_or(&0)),
                    &active,
                )?;
                match &txn.evidence {
                    TransactionEvidence::JobReserve(e) => {
                        if reservations
                            .insert(
                                e.job_id.clone(),
                                (
                                    e.work.clone(),
                                    e.shards,
                                    false,
                                    Some(e.requester_auth.node_id.clone()),
                                    txn.created_at,
                                ),
                            )
                            .is_some()
                        {
                            bail!("duplicate job reservation: {}", e.job_id)
                        }
                    }
                    TransactionEvidence::CommunityReserve {
                        job_id,
                        work,
                        shards,
                        ..
                    } => {
                        if reservations
                            .insert(
                                job_id.clone(),
                                (work.clone(), *shards, true, None, txn.created_at),
                            )
                            .is_some()
                        {
                            bail!("duplicate community job reservation: {job_id}")
                        }
                    }
                    TransactionEvidence::InferenceReserve(e) => {
                        if inference
                            .insert(
                                e.job_id.clone(),
                                (
                                    e.billing.clone(),
                                    e.batches.clone(),
                                    e.requester_auth.node_id.clone(),
                                    txn.created_at,
                                ),
                            )
                            .is_some()
                        {
                            bail!("duplicate inference reservation: {}", e.job_id)
                        }
                    }
                    TransactionEvidence::InferenceReward(e) => {
                        let Some((billing, batches, requester, reserved_at)) =
                            inference.get(&e.job_id)
                        else {
                            bail!(
                                "inference reward references missing reservation: {}",
                                e.job_id
                            )
                        };
                        // The batch has to be one the coordinator certified, at
                        // the index its own assignment id names. Anything else is
                        // a batch invented after the escrow was funded.
                        let Some(index) = (0..batches.len() as u32).find(|i| {
                            hocmesh_protocol::inference_assignment_id(&e.job_id, *i)
                                == e.assignment_id
                        }) else {
                            bail!("inference reward names a batch that was never certified")
                        };
                        let batch = &batches[index as usize];
                        if batch.batch_start != e.batch_start || batch.batch_end != e.batch_end {
                            bail!("inference reward changes the bounds of the batch it claims")
                        }
                        if batch.node_id != e.provider_auth.node_id {
                            bail!("inference reward paid to a node the batch was not assigned to")
                        }
                        if requester == &e.provider_auth.node_id {
                            bail!("requester cannot receive reward from its own paid job")
                        }
                        if inference_batch_price(billing, e.batch_start, e.batch_end)
                            != e.reward_mcu
                        {
                            bail!("inference reward is not what the batch prices at")
                        }
                        // A reward and a refund for one batch are disjoint in
                        // time, exactly as they are for a prime shard.
                        if txn.created_at > reserved_at + hocmesh_protocol::SETTLEMENT_WINDOW_SECS {
                            bail!("inference reward arrived after the settlement window")
                        }
                    }
                    TransactionEvidence::InferenceReceipt(e) => {
                        let Some((billing, batches, requester, reserved_at)) =
                            inference.get(&e.job_id)
                        else {
                            bail!(
                                "inference receipt references missing reservation: {}",
                                e.job_id
                            )
                        };
                        if requester != &e.requester_auth.node_id {
                            bail!("inference receipt signed by someone other than the requester")
                        }
                        if !batches
                            .iter()
                            .any(|b| b.batch_start == e.batch_start && b.batch_end == e.batch_end)
                        {
                            bail!("inference receipt names a batch the job never reserved")
                        }
                        if inference_batch_price(billing, e.batch_start, e.batch_end) != e.price_mcu
                        {
                            bail!("inference receipt is not what the batch prices at")
                        }
                        // Delivery only counts while the escrow is still live.
                        if txn.created_at > reserved_at + hocmesh_protocol::SETTLEMENT_WINDOW_SECS {
                            bail!("inference receipt arrived after the settlement window")
                        }
                    }
                    TransactionEvidence::InferenceDispute(e) => {
                        let Some((billing, _, requester, _)) = inference.get(&e.job_id) else {
                            bail!(
                                "inference dispute references missing reservation: {}",
                                e.job_id
                            )
                        };
                        if requester != &e.requester_auth.node_id {
                            bail!("inference dispute signed by someone other than the requester")
                        }
                        if inference_batch_price(billing, e.batch_start, e.batch_end) != e.price_mcu
                        {
                            bail!("inference dispute is not what the batch prices at")
                        }
                    }
                    TransactionEvidence::InferenceRefund(e) => {
                        let Some((billing, batches, requester, reserved_at)) =
                            inference.get(&e.job_id)
                        else {
                            bail!(
                                "inference refund references missing reservation: {}",
                                e.job_id
                            )
                        };
                        let Some(index) = (0..batches.len() as u32).find(|i| {
                            hocmesh_protocol::inference_assignment_id(&e.job_id, *i)
                                == e.assignment_id
                        }) else {
                            bail!("inference refund names a batch that was never certified")
                        };
                        let batch = &batches[index as usize];
                        if batch.batch_start != e.batch_start || batch.batch_end != e.batch_end {
                            bail!("inference refund changes the bounds of the batch it reclaims")
                        }
                        // The escrow goes back where it came from.
                        if requester != &e.requester_auth.node_id {
                            bail!("inference refund pays someone other than the requester")
                        }
                        if inference_batch_price(billing, e.batch_start, e.batch_end)
                            != e.refund_mcu
                        {
                            bail!("inference refund is not what the batch prices at")
                        }
                        if txn.created_at <= reserved_at + hocmesh_protocol::SETTLEMENT_WINDOW_SECS
                        {
                            bail!("inference refund inside the settlement window")
                        }
                    }
                    TransactionEvidence::ProviderReward(e) => {
                        let Some((root, shards, system, requester, reserved_at)) =
                            reservations.get(&e.job_id)
                        else {
                            bail!("reward references missing reservation: {}", e.job_id)
                        };
                        if *system != e.system_funded {
                            bail!("reward funding type differs from reservation")
                        };
                        if requester.as_deref() == Some(e.provider_auth.node_id.as_str()) {
                            bail!("requester cannot receive reward from its own paid job")
                        };
                        let parts = hocmesh_core::compute::split_work(root, *shards);
                        let Some(expected) = parts.get(e.shard_index as usize) else {
                            bail!("reward shard index outside reservation")
                        };
                        if expected != &e.work {
                            bail!("reward work differs from reserved shard")
                        }
                        // A reward and a refund for one shard are disjoint in
                        // time, so a provider that misses the window it agreed to
                        // cannot race the requester's refund for the same escrow.
                        if txn.created_at > reserved_at + hocmesh_protocol::SETTLEMENT_WINDOW_SECS {
                            bail!("reward arrived after the shard's settlement window")
                        }
                    }
                    TransactionEvidence::JobRefund(e) => {
                        let Some((root, shards, system, requester, reserved_at)) =
                            reservations.get(&e.job_id)
                        else {
                            bail!("refund references missing reservation: {}", e.job_id)
                        };
                        if *system != e.system_funded {
                            bail!("refund funding type differs from reservation")
                        }
                        // The escrow goes back where it came from, never to
                        // whoever happened to ask.
                        let claimant = e.requester_auth.as_ref().map(|a| a.node_id.as_str());
                        if requester.as_deref() != claimant {
                            bail!("refund pays someone other than the requester who reserved")
                        }
                        let parts = hocmesh_core::compute::split_work(root, *shards);
                        let Some(expected) = parts.get(e.shard_index as usize) else {
                            bail!("refund shard index outside reservation")
                        };
                        // Without this a refund could name a fatter shard than it
                        // reserved and drain escrow that is not its own.
                        if expected != &e.work {
                            bail!("refund work differs from reserved shard")
                        }
                        // A reward and a refund for one shard never share a moment.
                        if txn.created_at <= reserved_at + hocmesh_protocol::SETTLEMENT_WINDOW_SECS
                        {
                            bail!("refund inside the shard's settlement window")
                        }
                    }
                    TransactionEvidence::MembershipChange(e) => {
                        active = verify_membership_change(&active, e)?
                    }
                }
                for p in &txn.postings {
                    let next = balances
                        .get(&p.account_id)
                        .copied()
                        .unwrap_or(0)
                        .checked_add(p.delta_mcu)
                        .ok_or_else(|| anyhow::anyhow!("audit balance overflow"))?;
                    if next < 0 && p.account_id != COMMUNITY_ISSUANCE_ACCOUNT {
                        bail!(
                            "negative non-issuance balance during audit: {}",
                            p.account_id
                        )
                    };
                    balances.insert(p.account_id.clone(), next);
                    // Lifetime totals are part of the signed state, so an
                    // audit has to carry them forward with the balances or it
                    // would land on a different hash than the store it checks.
                    let seen = activity.entry(p.account_id.clone()).or_default();
                    if p.delta_mcu > 0 {
                        seen.0 += p.delta_mcu;
                    } else {
                        seen.1 -= p.delta_mcu;
                    }
                }
            }
            seq = c.entry.sequence;
            prev = c.entry.entry_hash;
        }
        // The replay and the tables the store keeps as it goes are two
        // independent accounts of the same thing. If they ever disagree, one
        // of them is wrong, and finding that out is most of what an audit is
        // for.
        if state.digest()? != self.state()?.digest()? {
            bail!("replayed ledger state does not match the state this store has been keeping")
        };
        Ok(LedgerHead {
            sequence: seq,
            entry_hash: prev,
            membership_hash: membership_hash(set)?,
        })
    }
}

/// What one batch of a certified inference job is worth.
///
/// Derived from the billing every time rather than trusted from the claim: the
/// price of a batch is a fact about the signed request, not something a
/// provider or a coordinator gets to assert.
fn inference_batch_price(billing: &InferenceBilling, start: u32, end: u32) -> i64 {
    hocmesh_core::compute::inference_batch_cost_mcu(
        &billing.prompt_bytes,
        start,
        end,
        billing.max_tokens,
        billing.parameter_count,
    )
}
fn index_certificate(tx: &rusqlite::Transaction<'_>, cert: &QuorumCertificate) -> Result<()> {
    let sequence = sqlite_sequence(cert.entry.sequence)?;
    // posting_index runs across the whole entry rather than restarting inside
    // each transaction: account_activity is keyed on
    // (account_id, sequence, posting_index), and a batched entry holds many
    // transactions whose postings all start at zero.
    for (posting_index, (txn, posting)) in cert
        .entry
        .transactions
        .iter()
        .flat_map(|t| t.postings.iter().map(move |p| (t, p)))
        .enumerate()
    {
        tx.execute(
            "INSERT OR REPLACE INTO account_activity(account_id,sequence,posting_index,transaction_id,delta_mcu,created_at)
             VALUES(?1,?2,?3,?4,?5,?6)",
            params![
                posting.account_id,
                sequence,
                i64::try_from(posting_index).context("posting index exceeds SQLite INTEGER range")?,
                txn.transaction_id,
                posting.delta_mcu,
                txn.created_at,
            ],
        )?;
    }
    for txn in &cert.entry.transactions {
        match &txn.evidence {
            TransactionEvidence::JobReserve(e) => {
                tx.execute(
                "INSERT OR REPLACE INTO job_reservations(job_id,sequence,work_json,shards,system_funded,requester_node_id,reserved_at)
                 VALUES(?1,?2,?3,?4,0,?5,?6)",
                params![
                    e.job_id,
                    sequence,
                    serde_json::to_string(&e.work)?,
                    i64::from(e.shards),
                    e.requester_auth.node_id,
                    txn.created_at,
                ],
            )?;
            }
            TransactionEvidence::CommunityReserve {
                job_id,
                work,
                shards,
                ..
            } => {
                tx.execute(
                "INSERT OR REPLACE INTO job_reservations(job_id,sequence,work_json,shards,system_funded,requester_node_id,reserved_at)
                 VALUES(?1,?2,?3,?4,1,NULL,?5)",
                params![
                    job_id,
                    sequence,
                    serde_json::to_string(work)?,
                    i64::from(*shards),
                    txn.created_at,
                ],
            )?;
            }
            TransactionEvidence::InferenceReserve(e) => {
                tx.execute(
                "INSERT OR REPLACE INTO inference_reservations(job_id,sequence,billing_json,batches_json,requester_node_id,reserved_at)
                 VALUES(?1,?2,?3,?4,?5,?6)",
                params![
                    e.job_id,
                    sequence,
                    serde_json::to_string(&e.billing)?,
                    serde_json::to_string(&e.batches)?,
                    e.requester_auth.node_id,
                    txn.created_at,
                ],
            )?;
            }
            TransactionEvidence::InferenceReward(e) => {
                tx.execute(
                "INSERT OR REPLACE INTO assignment_rewards(assignment_id,sequence,job_id,provider_node_id,reward_mcu,system_funded)
                 VALUES(?1,?2,?3,?4,?5,0)",
                params![
                    e.assignment_id,
                    sequence,
                    e.job_id,
                    e.provider_auth.node_id,
                    e.reward_mcu,
                ],
            )?;
            }
            TransactionEvidence::InferenceReceipt(_) | TransactionEvidence::InferenceDispute(_) => {
                // Neither pays a provider, so there is no reward to index. The
                // claim keys already record that the batch was received and
                // that it was settled away from the provider.
            }
            TransactionEvidence::InferenceRefund(_) => {
                // Nothing to index, for the same reason a shard refund indexes
                // nothing: the claim key is already recorded and the escrow it
                // empties is visible in the postings.
            }
            TransactionEvidence::ProviderReward(e) => {
                tx.execute(
                "INSERT OR REPLACE INTO assignment_rewards(assignment_id,sequence,job_id,provider_node_id,reward_mcu,system_funded)
                 VALUES(?1,?2,?3,?4,?5,?6)",
                params![
                    e.assignment_id,
                    sequence,
                    e.job_id,
                    e.provider_auth.node_id,
                    e.reward_mcu,
                    if e.system_funded { 1_i64 } else { 0_i64 },
                ],
            )?;
            }
            TransactionEvidence::JobRefund(_) => {
                // Nothing is indexed. A refund shares the claim key of the reward
                // it replaces, so the claims table has already recorded that this
                // shard is settled, and the escrow it empties is visible in the
                // postings like every other movement of CU.
            }
            TransactionEvidence::MembershipChange(_) => {
                // Nothing is indexed. A membership change moves no CU and
                // reserves nothing; the set it produces is applied by the
                // caller, which is the only place the previous set is known.
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::{
        build_entry, ledger_entry_signing_message, membership_result, verify_certificate,
        vouch_signing_message,
    };
    use hocmesh_core::identity::NodeIdentity;

    fn ballot(number: u64, proposer: &str) -> Ballot {
        Ballot {
            number,
            proposer: proposer.into(),
        }
    }

    /// A height belongs to the newest attempt, not the first one.
    #[test]
    fn a_later_ballot_takes_a_height_from_an_earlier_one() {
        let store = LedgerStore::open(":memory:").unwrap();
        assert!(store.promise(1, &ballot(1, "a")).unwrap().is_none());
        store.accept_ballot(1, &ballot(1, "a"), "aaa", &[]).unwrap();
        // The next proposer is told what is already signed here, and is
        // expected to carry that same entry rather than one of its own.
        let seen = store.promise(1, &ballot(2, "b")).unwrap().unwrap();
        assert_eq!(seen.entry_hash, "aaa");
        store.accept_ballot(1, &ballot(2, "b"), "aaa", &[]).unwrap();
    }

    /// Once the set has moved on, a stale proposer cannot sneak its entry in
    /// behind the one that took the height.
    #[test]
    fn a_superseded_proposer_cannot_still_be_signed_for() {
        let store = LedgerStore::open(":memory:").unwrap();
        store.promise(1, &ballot(1, "a")).unwrap();
        store.promise(1, &ballot(2, "b")).unwrap();
        assert!(store.promise(1, &ballot(1, "a")).is_err());
        assert!(store.accept_ballot(1, &ballot(1, "a"), "aaa", &[]).is_err());
    }

    fn store_dir(name: &str) -> std::path::PathBuf {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("hocmesh-store-test-{name}-{suffix}"));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn member_of(id: &NodeIdentity, index: usize) -> ValidatorMember {
        ValidatorMember {
            validator_id: id.node_id(),
            url: format!("http://127.0.0.1:{}", 9200 + index),
            public_key_b64: id.public_key_b64(),
        }
    }

    /// Signs an entry the way a validator does when it votes for it.
    fn certify(
        entry: LedgerEntry,
        set: &ValidatorSet,
        signers: &[&NodeIdentity],
    ) -> QuorumCertificate {
        let mh = membership_hash(set).unwrap();
        let message = ledger_entry_signing_message(&mh, &entry.entry_hash);
        QuorumCertificate {
            entry,
            membership_hash: mh,
            signatures: signers
                .iter()
                .map(|id| ValidatorSignature {
                    validator_id: id.node_id(),
                    signature_b64: id.sign_bytes_b64(message.as_bytes()),
                })
                .collect(),
        }
    }

    /// A membership transaction sponsored by the given sitting validators.
    fn vouched(
        set: &ValidatorSet,
        action: MembershipAction,
        member: &ValidatorMember,
        threshold: usize,
        signers: &[&NodeIdentity],
    ) -> LedgerTransaction {
        let next = membership_result(set, action, member, threshold).unwrap();
        let resulting_set_hash = membership_hash(&next).unwrap();
        let message = vouch_signing_message(
            &membership_hash(set).unwrap(),
            action,
            member,
            &resulting_set_hash,
        );
        LedgerTransaction {
            transaction_id: format!("membership_{}", member.validator_id),
            kind: TransactionKind::MembershipChange,
            postings: Vec::new(),
            evidence: TransactionEvidence::MembershipChange(MembershipChangeEvidence {
                action,
                member: member.clone(),
                threshold,
                vouches: signers
                    .iter()
                    .map(|id| ValidatorSignature {
                        validator_id: id.node_id(),
                        signature_b64: id.sign_bytes_b64(message.as_bytes()),
                    })
                    .collect(),
                resulting_set_hash,
            }),
            created_at: 0,
        }
    }

    /// The chain, not the bootstrap file, decides who may certify.
    ///
    /// An auditor that starts from the genesis set has to end up accepting
    /// signatures from a validator that set has never heard of, purely because
    /// the history says a quorum admitted it - and has to stop accepting the
    /// one the same history says left. That is the whole reason membership is
    /// a ledger event rather than a file every operator is trusted to match.
    #[test]
    fn an_audit_follows_the_set_the_chain_admits() {
        let dir = store_dir("membership_audit");
        let ids: Vec<NodeIdentity> = (0..5)
            .map(|i| NodeIdentity::load_or_create(&dir.join(format!("v{i}"))).unwrap())
            .collect();
        let genesis = ValidatorSet {
            threshold: 3,
            community_issuance_limit_mcu: 1_000_000,
            members: ids[..4]
                .iter()
                .enumerate()
                .map(|(i, id)| member_of(id, i))
                .collect(),
        };
        let newcomer = member_of(&ids[4], 4);
        let mut store = LedgerStore::open(":memory:").unwrap();

        // Entry 1: the sitting four admit a fifth, and the threshold rises with
        // the set. Sponsored and certified by the set as it stands.
        let join = vouched(
            &genesis,
            MembershipAction::Join,
            &newcomer,
            4,
            &[&ids[0], &ids[1], &ids[2]],
        );
        let e1 = build_entry(1, "GENESIS".into(), vec![join]).unwrap();
        let c1 = certify(e1, &genesis, &[&ids[0], &ids[1], &ids[2]]);
        store.apply(&c1, &genesis).unwrap();

        let admitted = store.current_set().unwrap().unwrap();
        assert_eq!(admitted.members.len(), 5);
        assert_eq!(admitted.threshold, 4);

        // Entry 2: a change the genesis set could not have certified. It is
        // sponsored and signed by four validators, one of whom the genesis set
        // has never heard of, and it carries the membership hash of a set that
        // only exists because entry 1 said so.
        let quorum = [&ids[1], &ids[2], &ids[3], &ids[4]];
        let departing = admitted.members[0].clone();
        let leave = vouched(&admitted, MembershipAction::Leave, &departing, 3, &quorum);
        let e2 = build_entry(2, c1.entry.entry_hash.clone(), vec![leave]).unwrap();
        let c2 = certify(e2, &admitted, &quorum);
        store.apply(&c2, &admitted).unwrap();

        // The genesis set cannot verify it. Nothing about entry 2 is legible
        // without the history that produced the set which signed it.
        assert!(verify_certificate(&c2, &genesis).is_err());

        // A full replay from the genesis set nevertheless accepts both, because
        // it follows the set the chain itself hands forward.
        let head = store.audit_from(&genesis, None).unwrap();
        assert_eq!(head.sequence, 2);
        assert_eq!(head.entry_hash, c2.entry.entry_hash);

        // And the set is queryable at height, which is what a checkpoint-
        // resumed audit needs: the seats as they were, not as they are.
        assert_eq!(store.set_at(1).unwrap().unwrap(), admitted);
        let after = store.current_set().unwrap().unwrap();
        assert_eq!(after.members.len(), 4);
        assert_eq!(after.threshold, 3);
        assert!(
            !after
                .members
                .iter()
                .any(|m| m.validator_id == departing.validator_id)
        );
        assert!(
            after
                .members
                .iter()
                .any(|m| m.validator_id == newcomer.validator_id)
        );
    }
}
