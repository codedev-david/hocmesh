use anyhow::{Context, Result, anyhow};
use hocmesh_core::compute::{split_work, work_cost_mcu};
use hocmesh_protocol::{WorkSpec, now_unix};
use rusqlite::{Connection, params};
use std::marker::PhantomData;
use std::sync::{Condvar, Mutex, MutexGuard};

/// Opens a standalone connection with the schema in place.
///
/// For the one-shot paths - seeding, recovery - that run outside the server
/// and have no pool to borrow from.
pub fn open(path: &str) -> Result<Connection> {
    let conn = connect(path)?;
    init_schema(&conn)?;
    Ok(conn)
}

fn connect(path: &str) -> Result<Connection> {
    let conn =
        Connection::open(path).with_context(|| format!("opening coordinator database {path}"))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    // Writers serialise no matter how many connections there are, so a reader
    // that arrives mid-write should wait for it rather than fail outright.
    conn.busy_timeout(std::time::Duration::from_secs(10))?;
    Ok(conn)
}

fn init_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            public_key_b64 TEXT NOT NULL,
            capabilities_json TEXT NOT NULL,
            registered_at INTEGER NOT NULL,
            last_seen INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS balances (
            node_id TEXT PRIMARY KEY REFERENCES nodes(node_id) ON DELETE CASCADE,
            balance_mcu INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL REFERENCES nodes(node_id),
            delta_mcu INTEGER NOT NULL,
            category TEXT NOT NULL,
            job_id TEXT,
            assignment_id TEXT,
            created_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            requester_node_id TEXT REFERENCES nodes(node_id),
            system_funded INTEGER NOT NULL DEFAULT 0,
            work_json TEXT NOT NULL,
            status TEXT NOT NULL,
            reserved_mcu INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            completed_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS assignments (
            assignment_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
            shard_index INTEGER NOT NULL,
            work_json TEXT NOT NULL,
            status TEXT NOT NULL,
            leased_to TEXT REFERENCES nodes(node_id),
            lease_until INTEGER,
            result_json TEXT,
            reward_mcu INTEGER NOT NULL,
            completed_at INTEGER,
            UNIQUE(job_id, shard_index)
        );

        CREATE INDEX IF NOT EXISTS idx_assignments_status ON assignments(status);
        CREATE INDEX IF NOT EXISTS idx_assignments_lease_until ON assignments(lease_until);
        CREATE INDEX IF NOT EXISTS idx_ledger_node ON ledger(node_id);
        CREATE TABLE IF NOT EXISTS auth_nonces (
            node_id TEXT NOT NULL,
            nonce_b64 TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            PRIMARY KEY(node_id, nonce_b64)
        );
        CREATE INDEX IF NOT EXISTS idx_auth_nonces_expiry ON auth_nonces(expires_at);

        CREATE TABLE IF NOT EXISTS ledger_intents (
            claim_key TEXT PRIMARY KEY,
            intent_kind TEXT NOT NULL,
            object_id TEXT NOT NULL,
            transaction_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            entry_hash TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ledger_intents_status ON ledger_intents(status);

        CREATE TABLE IF NOT EXISTS model_manifests (
            manifest_digest TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            revision TEXT NOT NULL,
            manifest_json TEXT NOT NULL,
            publisher_node_id TEXT NOT NULL REFERENCES nodes(node_id),
            created_at INTEGER NOT NULL,
            UNIQUE(model_id, revision)
        );
        CREATE INDEX IF NOT EXISTS idx_model_manifests_model ON model_manifests(model_id);

        CREATE TABLE IF NOT EXISTS ai_jobs (
            job_id TEXT PRIMARY KEY,
            requester_node_id TEXT NOT NULL REFERENCES nodes(node_id),
            request_json TEXT NOT NULL,
            plan_json TEXT NOT NULL,
            manifest_digest TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            completed_at INTEGER
        );
        CREATE TABLE IF NOT EXISTS ai_assignments (
            assignment_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES ai_jobs(job_id) ON DELETE CASCADE,
            assigned_node_id TEXT NOT NULL REFERENCES nodes(node_id),
            assignment_json TEXT NOT NULL,
            status TEXT NOT NULL,
            lease_until INTEGER,
            outputs_json TEXT,
            failure_count INTEGER NOT NULL DEFAULT 0,
            failed_nodes_json TEXT NOT NULL DEFAULT '[]',
            completed_at INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_ai_assignments_node_status ON ai_assignments(assigned_node_id,status);
        CREATE INDEX IF NOT EXISTS idx_ai_assignments_lease ON ai_assignments(lease_until);

        CREATE TABLE IF NOT EXISTS reputation (
            node_id TEXT PRIMARY KEY REFERENCES nodes(node_id) ON DELETE CASCADE,
            accepted INTEGER NOT NULL DEFAULT 0,
            rejected INTEGER NOT NULL DEFAULT 0,
            streak INTEGER NOT NULL DEFAULT 0
        );
        "#,
    )?;
    // Added after v0.3.0: a delivered batch is no longer paid on arrival, so
    // the coordinator has to remember the provider's signed report until the
    // requester takes delivery and settles. Added by ALTER so an existing
    // coordinator database picks the columns up without being rebuilt.
    for column in [
        "report_json TEXT",
        "outputs_digest TEXT",
        "receipted INTEGER NOT NULL DEFAULT 0",
        "settled TEXT",
    ] {
        // Fails only when the column is already there, which is the steady state.
        let _ = conn.execute(
            &format!("ALTER TABLE ai_assignments ADD COLUMN {column}"),
            [],
        );
    }
    Ok(())
}

pub fn persist_ledger_intent(
    conn: &Connection,
    claim_key: &str,
    intent_kind: &str,
    object_id: &str,
    tx_json: &str,
) -> Result<()> {
    let now = now_unix();
    conn.execute("INSERT INTO ledger_intents(claim_key,intent_kind,object_id,transaction_json,status,created_at,updated_at) VALUES(?1,?2,?3,?4,'pending',?5,?5) ON CONFLICT(claim_key) DO UPDATE SET transaction_json=excluded.transaction_json,updated_at=excluded.updated_at", params![claim_key,intent_kind,object_id,tx_json,now])?;
    Ok(())
}

pub fn certify_ledger_intent(conn: &Connection, claim_key: &str, entry_hash: &str) -> Result<()> {
    conn.execute("UPDATE ledger_intents SET status='certified',entry_hash=?2,updated_at=?3 WHERE claim_key=?1", params![claim_key,entry_hash,now_unix()])?;
    Ok(())
}

pub fn seed_system_job_with_id(
    conn: &mut Connection,
    job_id: &str,
    work: WorkSpec,
    shards: u32,
    ready: bool,
) -> Result<()> {
    work.validate().map_err(anyhow::Error::msg)?;
    let parts = split_work(&work, shards.clamp(1, 256));
    let reserved_mcu: i64 = parts.iter().map(work_cost_mcu).sum();
    let now = now_unix();
    let tx = conn.transaction()?;
    let job_status = if ready { "pending" } else { "funding" };
    let assignment_status = if ready { "pending" } else { "blocked" };
    tx.execute("INSERT INTO jobs(job_id,requester_node_id,system_funded,work_json,status,reserved_mcu,created_at) VALUES(?1,NULL,1,?2,?3,?4,?5)",params![job_id,serde_json::to_string(&work)?,job_status,reserved_mcu,now])?;
    for (index, part) in parts.iter().enumerate() {
        let assignment_id = hocmesh_protocol::assignment_id(job_id, index as u32);
        tx.execute("INSERT INTO assignments(assignment_id,job_id,shard_index,work_json,status,reward_mcu) VALUES(?1,?2,?3,?4,?5,?6)",params![assignment_id,job_id,index as i64,serde_json::to_string(part)?,assignment_status,work_cost_mcu(part)])?;
    }
    tx.commit()?;
    Ok(())
}

/// A small pool of SQLite connections.
///
/// The coordinator used to share a single connection behind a mutex, so every
/// request queued behind every other one however unrelated: a status poll
/// waited on somebody else's settlement write. SQLite in WAL mode reads
/// concurrently and serialises only writers, so the answer is more
/// connections rather than a longer queue.
pub struct Pool {
    path: String,
    state: Mutex<PoolState>,
    returned: Condvar,
    max: usize,
}

struct PoolState {
    idle: Vec<Connection>,
    /// Connections that exist, whether idle or lent out. Never above `max`.
    live: usize,
}

impl Pool {
    /// Opens the database, creating the schema, and sizes the pool.
    ///
    /// The cap is the machine's parallelism because a connection is only ever
    /// held by a thread that is actively using it - never across an await -
    /// so more connections than threads could not be in use at once.
    pub fn open(path: &str) -> Result<Self> {
        let conn = connect(path)?;
        init_schema(&conn)?;
        let max = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(4)
            .clamp(4, 32);
        Ok(Self {
            path: path.into(),
            state: Mutex::new(PoolState {
                idle: vec![conn],
                live: 1,
            }),
            returned: Condvar::new(),
            max,
        })
    }

    /// Borrows a connection, opening one if the pool has room to grow.
    ///
    /// Blocks only when every connection is already lent out, which cannot
    /// outlast the request holding one, because a borrowed connection can
    /// never be held across an await point.
    pub fn get(&self) -> Result<PooledConnection<'_>> {
        let mut state = self.lock()?;
        loop {
            if let Some(conn) = state.idle.pop() {
                return Ok(self.lend(conn));
            }
            if state.live < self.max {
                state.live += 1;
                drop(state);
                // Opened outside the lock: a new file handle is slow enough
                // that holding the pool shut for it would defeat the point.
                return connect(&self.path).map(|c| self.lend(c)).inspect_err(|_| {
                    if let Ok(mut s) = self.lock() {
                        s.live -= 1;
                    }
                });
            }
            state = self
                .returned
                .wait(state)
                .map_err(|_| anyhow!("coordinator database pool poisoned"))?;
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, PoolState>> {
        self.state
            .lock()
            .map_err(|_| anyhow!("coordinator database pool poisoned"))
    }
    fn lend(&self, conn: Connection) -> PooledConnection<'_> {
        PooledConnection {
            pool: self,
            conn: Some(conn),
            not_send: PhantomData,
        }
    }
}

/// A borrowed connection, returned to the pool when it goes out of scope.
///
/// Deliberately not `Send`. That is what the old `MutexGuard` gave for free,
/// and it is what keeps a connection from being held across an await point -
/// which is in turn what bounds how many can be in use at once.
pub struct PooledConnection<'a> {
    pool: &'a Pool,
    conn: Option<Connection>,
    not_send: PhantomData<*const ()>,
}
impl Drop for PooledConnection<'_> {
    fn drop(&mut self) {
        let Some(conn) = self.conn.take() else { return };
        if let Ok(mut state) = self.pool.lock() {
            state.idle.push(conn);
            self.pool.returned.notify_one();
        }
    }
}
impl std::ops::Deref for PooledConnection<'_> {
    type Target = Connection;
    fn deref(&self) -> &Connection {
        self.conn.as_ref().expect("connection taken before drop")
    }
}
impl std::ops::DerefMut for PooledConnection<'_> {
    fn deref_mut(&mut self) -> &mut Connection {
        self.conn.as_mut().expect("connection taken before drop")
    }
}
