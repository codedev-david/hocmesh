use crate::{
    types::*,
    validate::{
        claim_key, membership_hash, validate_historical_transaction, verify_certificate,
        verify_historical_evidence,
    },
};
use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, params};

pub struct LedgerStore {
    conn: Connection,
}

type ReservationRecord = (hocmesh_protocol::WorkSpec, u32, bool, Option<String>);

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
                requester_node_id TEXT
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
    pub fn reservation(&self, job_id: &str) -> Result<Option<ReservationRecord>> {
        self.conn
            .query_row(
                "SELECT work_json,shards,system_funded,requester_node_id FROM job_reservations WHERE job_id=?1",
                params![job_id],
                |r| {
                    let work_json: String = r.get(0)?;
                    Ok((
                        work_json,
                        r.get::<_, i64>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, Option<String>>(3)?,
                    ))
                },
            )
            .optional()?
            .map(|(work_json, shards, system_funded, requester)| {
                let shards = u32::try_from(shards).context("stored shard count is outside u32")?;
                let work = serde_json::from_str(&work_json)?;
                Ok((work, shards, system_funded != 0, requester))
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
        verify_historical_evidence(&cert.entry.transaction)?;
        let h = self.head(set)?;
        if cert.entry.sequence != h.sequence + 1 || cert.entry.previous_hash != h.entry_hash {
            bail!("certificate does not extend local head")
        };
        let ck = claim_key(&cert.entry.transaction);
        if self.has_claim(&ck)? {
            bail!("ledger claim already settled: {ck}")
        };
        let tx = self.conn.transaction()?;
        for p in &cert.entry.transaction.postings {
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
        tx.execute(
            "INSERT INTO certificates(sequence,entry_hash,certificate_json) VALUES(?1,?2,?3)",
            params![
                sqlite_sequence(cert.entry.sequence)?,
                cert.entry.entry_hash,
                serde_json::to_string(cert)?
            ],
        )?;
        tx.execute(
            "INSERT INTO claims(claim_key,sequence) VALUES(?1,?2)",
            params![ck, sqlite_sequence(cert.entry.sequence)?],
        )?;
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

    fn rebuild_indexes(&mut self) -> Result<()> {
        let certs = self.certificates_from(1, u64::MAX)?;
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM account_activity", [])?;
        tx.execute("DELETE FROM job_reservations", [])?;
        tx.execute("DELETE FROM assignment_rewards", [])?;
        for cert in &certs {
            index_certificate(&tx, cert)?;
        }
        tx.commit()?;
        Ok(())
    }
    pub fn audit(&self, set: &ValidatorSet) -> Result<LedgerHead> {
        let certs = self.certificates_from(1, u64::MAX)?;
        let mut seq = 0;
        let mut prev = "GENESIS".to_string();
        let mut balances = std::collections::HashMap::<String, i64>::new();
        let mut claims = std::collections::HashSet::<String>::new();
        let mut reservations = std::collections::HashMap::<
            String,
            (hocmesh_protocol::WorkSpec, u32, bool, Option<String>),
        >::new();
        for c in certs {
            verify_certificate(&c, set)?;
            if c.entry.sequence != seq + 1 || c.entry.previous_hash != prev {
                bail!("broken chain at sequence {}", c.entry.sequence)
            };
            let ck = claim_key(&c.entry.transaction);
            if !claims.insert(ck.clone()) {
                bail!("duplicate ledger claim during audit: {ck}")
            };
            validate_historical_transaction(
                &c.entry.transaction,
                |a| Ok(*balances.get(a).unwrap_or(&0)),
                set.community_issuance_limit_mcu,
            )?;
            match &c.entry.transaction.evidence {
                TransactionEvidence::JobReserve(e) => {
                    if reservations
                        .insert(
                            e.job_id.clone(),
                            (
                                e.work.clone(),
                                e.shards,
                                false,
                                Some(e.requester_auth.node_id.clone()),
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
                        .insert(job_id.clone(), (work.clone(), *shards, true, None))
                        .is_some()
                    {
                        bail!("duplicate community job reservation: {job_id}")
                    }
                }
                TransactionEvidence::ProviderReward(e) => {
                    let Some((root, shards, system, requester)) = reservations.get(&e.job_id)
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
                }
            }
            for p in &c.entry.transaction.postings {
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
            seq = c.entry.sequence;
            prev = c.entry.entry_hash;
        }
        Ok(LedgerHead {
            sequence: seq,
            entry_hash: prev,
            membership_hash: membership_hash(set)?,
        })
    }
}

fn index_certificate(tx: &rusqlite::Transaction<'_>, cert: &QuorumCertificate) -> Result<()> {
    let sequence = sqlite_sequence(cert.entry.sequence)?;
    for (posting_index, posting) in cert.entry.transaction.postings.iter().enumerate() {
        tx.execute(
            "INSERT OR REPLACE INTO account_activity(account_id,sequence,posting_index,transaction_id,delta_mcu,created_at)
             VALUES(?1,?2,?3,?4,?5,?6)",
            params![
                posting.account_id,
                sequence,
                i64::try_from(posting_index).context("posting index exceeds SQLite INTEGER range")?,
                cert.entry.transaction.transaction_id,
                posting.delta_mcu,
                cert.entry.transaction.created_at,
            ],
        )?;
    }
    match &cert.entry.transaction.evidence {
        TransactionEvidence::JobReserve(e) => {
            tx.execute(
                "INSERT OR REPLACE INTO job_reservations(job_id,sequence,work_json,shards,system_funded,requester_node_id)
                 VALUES(?1,?2,?3,?4,0,?5)",
                params![
                    e.job_id,
                    sequence,
                    serde_json::to_string(&e.work)?,
                    i64::from(e.shards),
                    e.requester_auth.node_id,
                ],
            )?;
        }
        TransactionEvidence::CommunityReserve {
            job_id,
            work,
            shards,
        } => {
            tx.execute(
                "INSERT OR REPLACE INTO job_reservations(job_id,sequence,work_json,shards,system_funded,requester_node_id)
                 VALUES(?1,?2,?3,?4,1,NULL)",
                params![job_id, sequence, serde_json::to_string(work)?, i64::from(*shards)],
            )?;
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
