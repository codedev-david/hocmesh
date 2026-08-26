use crate::{
    types::*,
    validate::{
        claim_key, membership_hash, validate_historical_transaction, verify_certificate,
        verify_checkpoint, verify_historical_evidence,
    },
};
use anyhow::{Context, Result, bail};
use hocmesh_protocol::{InferenceBilling, PricedBatch};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::{BTreeMap, BTreeSet};

pub struct LedgerStore {
    conn: Connection,
}

type ReservationRecord = (hocmesh_protocol::WorkSpec, u32, bool, Option<String>, i64);
/// What the ledger remembers about a certified inference job.
#[derive(Debug, Clone)]
pub struct InferenceReservation {
    pub billing: InferenceBilling,
    pub batches: Vec<PricedBatch>,
    pub requester: String,
    pub reserved_at: i64,
}

type InferenceRecord = (InferenceBilling, Vec<PricedBatch>, String, i64);

/// Everything an audit needs in order to start somewhere other than genesis.
///
/// Balances and settled claims are the obvious part; the open reservations
/// matter just as much, because a reward or a refund is only valid against
/// the reservation it draws on.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct LedgerState {
    pub balances: BTreeMap<String, i64>,
    pub claims: BTreeSet<String>,
    pub reservations: BTreeMap<String, ReservationRecord>,
    pub inference: BTreeMap<String, InferenceRecord>,
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
        Ok(hocmesh_protocol::hash_json(&(
            "hocmesh-state-v1",
            &balances,
            &self.claims,
            &reservations,
            &inference,
        ))?)
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
            CREATE TABLE IF NOT EXISTS checkpoints(
                sequence INTEGER PRIMARY KEY,
                checkpoint_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS votes(
                sequence INTEGER PRIMARY KEY,
                entry_hash TEXT NOT NULL
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
    pub fn lock_vote(&self, sequence: u64, entry_hash: &str) -> Result<()> {
        let sqlite_sequence = sqlite_sequence(sequence)?;
        let existing: Option<String> = self
            .conn
            .query_row(
                "SELECT entry_hash FROM votes WHERE sequence=?1",
                params![sqlite_sequence],
                |r| r.get(0),
            )
            .optional()?;
        match existing {
            Some(h) if h != entry_hash => {
                bail!("validator already voted for a conflicting entry at sequence {sequence}")
            }
            Some(_) => Ok(()),
            None => {
                self.conn.execute(
                    "INSERT INTO votes(sequence,entry_hash) VALUES(?1,?2)",
                    params![sqlite_sequence, entry_hash],
                )?;
                Ok(())
            }
        }
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
        let (earned, spent) = self.conn.query_row(
            "SELECT COALESCE(SUM(CASE WHEN delta_mcu > 0 THEN delta_mcu ELSE 0 END),0),
                    COALESCE(SUM(CASE WHEN delta_mcu < 0 THEN -delta_mcu ELSE 0 END),0)
             FROM account_activity WHERE account_id=?1",
            params![a],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        Ok((earned, spent))
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
        for t in &cert.entry.transactions {
            verify_historical_evidence(t, &cert.entry.previous_hash, &cert.signatures)?;
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
        index_certificate(&tx, cert)?;
        tx.commit()?;
        Ok(())
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
        } = &mut state;
        for c in certs {
            verify_certificate(&c, set)?;
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
                    set.community_issuance_limit_mcu,
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
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_cannot_double_vote_at_same_height() {
        let store = LedgerStore::open(":memory:").unwrap();
        store.lock_vote(1, "aaa").unwrap();
        store.lock_vote(1, "aaa").unwrap();
        assert!(store.lock_vote(1, "bbb").is_err());
    }
}
