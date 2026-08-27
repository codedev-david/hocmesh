use anyhow::{Context, Result, anyhow};
use hocmesh_core::compute::{split_work, work_cost_mcu};
use hocmesh_protocol::{LedgerIntentState, WorkSpec, now_unix};
use rusqlite::{Connection, OptionalExtension, params};
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
    // `receipt_auth_json` arrived with the expiry sweep. A batch whose
    // requester takes delivery and then never judges it would otherwise strand
    // its holding account forever, and the only thing that can authorise the
    // commons to collect it is the receipt the requester signed on the way in.
    // The coordinator has to keep that proof to be able to hand it back.
    for column in [
        "report_json TEXT",
        "outputs_digest TEXT",
        "receipted INTEGER NOT NULL DEFAULT 0",
        "settled TEXT",
        "receipt_auth_json TEXT",
    ] {
        // Fails only when the column is already there, which is the steady state.
        let _ = conn.execute(
            &format!("ALTER TABLE ai_assignments ADD COLUMN {column}"),
            [],
        );
    }
    // The reconciliation daemon writes its own working state back here so that
    // "stuck" is a thing an operator can read rather than infer from logs. Older
    // coordinator databases predate the columns and pick them up in place.
    for column in ["attempts INTEGER NOT NULL DEFAULT 0", "last_error TEXT"] {
        // Fails only when the column is already there, which is the steady state.
        let _ = conn.execute(
            &format!("ALTER TABLE ledger_intents ADD COLUMN {column}"),
            [],
        );
    }
    // Federation. `leased_by` names the coordinator that handed a shard out,
    // which is what lets a survivor tell its own leases from those of a
    // coordinator that has since died. `region` on a node is the region of the
    // coordinator it registered with -- the only region anything knows about
    // it -- and is what the scheduler's network axis compares against.
    let _ = conn.execute("ALTER TABLE assignments ADD COLUMN leased_by TEXT", []);
    let _ = conn.execute("ALTER TABLE nodes ADD COLUMN region TEXT", []);
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS coordinator_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_assignments_leased_by ON assignments(leased_by);
        "#,
    )?;
    Ok(())
}

/// Read one row of the coordinator's own working state.
///
/// Deliberately a key/value table and deliberately not on the ledger: what is
/// kept here is how far this replica has read, which is a fact about this
/// process and nobody else's business.
pub fn coordinator_state(conn: &Connection, key: &str) -> Result<Option<String>> {
    Ok(conn
        .query_row(
            "SELECT value FROM coordinator_state WHERE key=?1",
            params![key],
            |r| r.get(0),
        )
        .optional()?)
}

pub fn set_coordinator_state(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO coordinator_state(key,value) VALUES(?1,?2)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        params![key, value],
    )?;
    Ok(())
}

/// Bring forward the expiry of every lease a departed coordinator granted.
///
/// Not a straight release. The worker on the other end may be perfectly
/// healthy and halfway through the shard -- what failed might be the link
/// between the coordinators rather than the coordinator itself -- and
/// cancelling its lease outright would throw that work away and reject the
/// result when it arrives. Shortening the lease instead means the adopting
/// coordinator waits a grace period rather than the full fifteen minutes, and
/// an in-flight worker still gets to finish. Leases already expiring sooner
/// are left alone.
pub fn shorten_leases_from(
    conn: &Connection,
    coordinator_id: &str,
    deadline: i64,
) -> Result<usize> {
    Ok(conn.execute(
        "UPDATE assignments SET lease_until=?2
         WHERE status='leased' AND leased_by=?1 AND (lease_until IS NULL OR lease_until > ?2)",
        params![coordinator_id, deadline],
    )?)
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
    conn.execute("UPDATE ledger_intents SET status='certified',entry_hash=?2,last_error=NULL,updated_at=?3 WHERE claim_key=?1", params![claim_key,entry_hash,now_unix()])?;
    Ok(())
}

/// How many consecutive passes an intent may fail before it is parked.
///
/// A transient fault that never clears is indistinguishable from a permanent
/// one after long enough, and an intent retried forever is an intent nobody
/// ever looks at. At the daemon's tick this is roughly an hour of trying.
pub const MAX_INTENT_ATTEMPTS: i64 = 240;

/// Note that an intent could not be settled this pass, and say how many passes
/// it has now cost. The intent itself is untouched: same transaction, same
/// claim key, still `pending`, and no CU has moved either way.
pub fn defer_ledger_intent(conn: &Connection, claim_key: &str, reason: &str) -> Result<i64> {
    conn.execute(
        "UPDATE ledger_intents SET attempts=attempts+1,last_error=?2,updated_at=?3 WHERE claim_key=?1 AND status='pending'",
        params![claim_key, reason, now_unix()],
    )?;
    Ok(conn
        .query_row(
            "SELECT attempts FROM ledger_intents WHERE claim_key=?1",
            params![claim_key],
            |r| r.get::<_, i64>(0),
        )
        .optional()?
        .unwrap_or(0))
}

/// Stop retrying an intent that cannot settle under its own claim key.
///
/// Bookkeeping only. No ledger entry is written, none is withdrawn, and no CU
/// moves -- the coordinator has no standing to do any of that. Parking it just
/// takes it out of the retry rotation and leaves the reason attached where an
/// operator will find it.
pub fn abandon_ledger_intent(conn: &Connection, claim_key: &str, reason: &str) -> Result<()> {
    conn.execute(
        "UPDATE ledger_intents SET status='unrecoverable',attempts=attempts+1,last_error=?2,updated_at=?3 WHERE claim_key=?1 AND status='pending'",
        params![claim_key, reason, now_unix()],
    )?;
    Ok(())
}

/// Every intent the coordinator has not finished settling, oldest first.
///
/// Both the ones still being retried and the ones that were given up on: the
/// second group is the one an operator has to act on, and hiding it would make
/// the view look healthier than the coordinator actually is.
pub fn unsettled_ledger_intents(conn: &Connection) -> Result<Vec<LedgerIntentState>> {
    let mut st = conn.prepare(
        "SELECT claim_key,intent_kind,object_id,status,attempts,last_error,entry_hash,created_at,updated_at \
         FROM ledger_intents WHERE status IN ('pending','unrecoverable') ORDER BY created_at,claim_key",
    )?;
    let rows = st.query_map([], |r| {
        Ok(LedgerIntentState {
            claim_key: r.get(0)?,
            intent_kind: r.get(1)?,
            object_id: r.get(2)?,
            status: r.get(3)?,
            attempts: r.get(4)?,
            last_error: r.get(5)?,
            entry_hash: r.get(6)?,
            created_at: r.get(7)?,
            updated_at: r.get(8)?,
        })
    })?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?)
    }
    Ok(out)
}

/// Coordinator work still waiting on a ledger entry that nothing is chasing.
///
/// A job parked in `funding`, or an assignment parked in `settling`, with no
/// pending intent naming it: the two sides drifted apart and the intent that
/// would have joined them back up is gone. The daemon counts these and stops
/// there. Filling the gap would mean the coordinator deciding on its own that
/// CU exists, which is the one thing it is never allowed to do.
pub fn orphaned_funding_objects(conn: &Connection) -> Result<u64> {
    Ok(conn.query_row(
        "SELECT (SELECT COUNT(*) FROM jobs j WHERE j.status='funding' \
              AND NOT EXISTS (SELECT 1 FROM ledger_intents i \
                              WHERE i.status='pending' AND i.object_id=j.job_id)) \
              + (SELECT COUNT(*) FROM assignments a WHERE a.status='settling' \
              AND NOT EXISTS (SELECT 1 FROM ledger_intents i \
                              WHERE i.status='pending' AND i.object_id=a.assignment_id))",
        [],
        |r| r.get::<_, i64>(0),
    )? as u64)
}

/// Seed a system-funded job, and the intent that will pay for it, together.
///
/// One transaction on purpose. A job written without its intent is work the
/// reconciliation pass can only report, never finish: nothing names it, so
/// nothing will ever propose the entry it is waiting on, and the coordinator
/// is not allowed to write that entry on its own say-so.
pub fn seed_system_job_with_id(
    conn: &mut Connection,
    job_id: &str,
    work: WorkSpec,
    shards: u32,
    intent: Option<(&str, &str, &str)>,
) -> Result<()> {
    work.validate().map_err(anyhow::Error::msg)?;
    let parts = split_work(&work, shards.clamp(1, 256));
    let reserved_mcu: i64 = parts.iter().map(work_cost_mcu).sum();
    let now = now_unix();
    let tx = conn.transaction()?;
    // No intent means nothing has to be paid for before the work can run.
    let ready = intent.is_none();
    let job_status = if ready { "pending" } else { "funding" };
    let assignment_status = if ready { "pending" } else { "blocked" };
    tx.execute("INSERT INTO jobs(job_id,requester_node_id,system_funded,work_json,status,reserved_mcu,created_at) VALUES(?1,NULL,1,?2,?3,?4,?5)",params![job_id,serde_json::to_string(&work)?,job_status,reserved_mcu,now])?;
    for (index, part) in parts.iter().enumerate() {
        let assignment_id = hocmesh_protocol::assignment_id(job_id, index as u32);
        tx.execute("INSERT INTO assignments(assignment_id,job_id,shard_index,work_json,status,reward_mcu) VALUES(?1,?2,?3,?4,?5,?6)",params![assignment_id,job_id,index as i64,serde_json::to_string(part)?,assignment_status,work_cost_mcu(part)])?;
    }
    if let Some((claim_key, kind, transaction_json)) = intent {
        persist_ledger_intent(&tx, claim_key, kind, job_id, transaction_json)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A job with four leases, spread across two coordinators and two states.
    fn lease_db() -> Connection {
        let conn = open(":memory:").expect("in-memory coordinator database");
        conn.execute(
            "INSERT INTO jobs(job_id,system_funded,work_json,status,reserved_mcu,created_at) \
             VALUES('job1',1,'{}','running',0,0)",
            [],
        )
        .expect("seeding a job");
        // (id, status, leased_by, lease_until)
        let rows = [
            ("a-long", "leased", Some("a"), Some(10_000i64)),
            ("a-short", "leased", Some("a"), Some(50i64)),
            ("a-done", "completed", Some("a"), Some(10_000i64)),
            ("b-long", "leased", Some("b"), Some(10_000i64)),
            ("orphan", "leased", None, Some(10_000i64)),
        ];
        for (index, (id, status, by, until)) in rows.iter().enumerate() {
            conn.execute(
                "INSERT INTO assignments(assignment_id,job_id,shard_index,work_json,status,lease_until,leased_by,reward_mcu) \
                 VALUES(?1,'job1',?2,'{}',?3,?4,?5,1)",
                params![id, index as i64, status, until, by],
            )
            .expect("seeding an assignment");
        }
        conn
    }

    fn lease_until(conn: &Connection, id: &str) -> Option<i64> {
        conn.query_row(
            "SELECT lease_until FROM assignments WHERE assignment_id=?1",
            params![id],
            |row| row.get(0),
        )
        .expect("reading a lease")
    }

    /// Cutting a dead coordinator's leases short must not touch anyone else's.
    ///
    /// This runs when a peer stops answering, which is exactly when the mesh
    /// is least sure what is going on -- so it has to be surgical. Reclaiming
    /// a live coordinator's leases here would turn one coordinator's outage
    /// into everybody's.
    #[test]
    fn shortening_a_dead_peers_leases_leaves_every_other_lease_alone() {
        let conn = lease_db();
        assert_eq!(shorten_leases_from(&conn, "a", 100).unwrap(), 1);
        assert_eq!(lease_until(&conn, "a-long"), Some(100));
        // Already inside the deadline, so there was nothing to cut.
        assert_eq!(lease_until(&conn, "a-short"), Some(50));
        // A finished shard is not a lease, whoever handed it out.
        assert_eq!(lease_until(&conn, "a-done"), Some(10_000));
        // The live coordinator keeps its work.
        assert_eq!(lease_until(&conn, "b-long"), Some(10_000));
        // So does a lease from before the column existed.
        assert_eq!(lease_until(&conn, "orphan"), Some(10_000));
    }

    /// The deadline is a floor, never an extension.
    ///
    /// A second pass over the same peer -- the probe loop is a loop -- must not
    /// hand a worker back time that the first pass took away.
    #[test]
    fn shortening_never_lengthens_a_lease() {
        let conn = lease_db();
        shorten_leases_from(&conn, "a", 100).unwrap();
        assert_eq!(shorten_leases_from(&conn, "a", 9_000).unwrap(), 0);
        assert_eq!(lease_until(&conn, "a-long"), Some(100));
    }

    /// The watermark has to survive a restart and has to move forward, because
    /// a coordinator that forgets it replays the chain from genesis and one
    /// that cannot overwrite it never advances past its first pass.
    #[test]
    fn coordinator_state_round_trips_and_overwrites() {
        let conn = open(":memory:").expect("in-memory coordinator database");
        assert_eq!(coordinator_state(&conn, "watermark").unwrap(), None);
        set_coordinator_state(&conn, "watermark", "17").unwrap();
        assert_eq!(
            coordinator_state(&conn, "watermark").unwrap().as_deref(),
            Some("17")
        );
        set_coordinator_state(&conn, "watermark", "42").unwrap();
        assert_eq!(
            coordinator_state(&conn, "watermark").unwrap().as_deref(),
            Some("42")
        );
        assert_eq!(coordinator_state(&conn, "other").unwrap(), None);
    }

    /// A pending intent with nothing else in the database around it.
    fn intent_db(claim: &str) -> Connection {
        let conn = open(":memory:").expect("in-memory coordinator database");
        conn.execute(
            "INSERT INTO ledger_intents(claim_key,intent_kind,object_id,transaction_json,status,created_at,updated_at) \
             VALUES(?1,'community_reserve','job_x','{}','pending',0,0)",
            params![claim],
        )
        .expect("seeding a pending intent");
        conn
    }

    #[test]
    fn deferring_counts_attempts_without_settling_anything() {
        let conn = intent_db("ck1");
        assert_eq!(
            defer_ledger_intent(&conn, "ck1", "network down").unwrap(),
            1
        );
        assert_eq!(
            defer_ledger_intent(&conn, "ck1", "network down").unwrap(),
            2
        );
        let rows = unsettled_ledger_intents(&conn).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, "pending");
        assert_eq!(rows[0].attempts, 2);
        assert_eq!(rows[0].last_error.as_deref(), Some("network down"));
        assert!(rows[0].entry_hash.is_none());
    }

    #[test]
    fn a_parked_intent_stops_being_retried_but_stays_visible() {
        let conn = intent_db("ck2");
        abandon_ledger_intent(&conn, "ck2", "claim mismatch").unwrap();
        // Nothing to hand back to a later pass...
        assert_eq!(defer_ledger_intent(&conn, "ck2", "again").unwrap(), 1);
        let rows = unsettled_ledger_intents(&conn).unwrap();
        assert_eq!(rows[0].status, "unrecoverable");
        // ...and the reason it was parked survives the attempt.
        assert_eq!(rows[0].last_error.as_deref(), Some("claim mismatch"));
    }

    #[test]
    fn certifying_clears_the_failure_and_the_backlog() {
        let conn = intent_db("ck3");
        defer_ledger_intent(&conn, "ck3", "no quorum yet").unwrap();
        certify_ledger_intent(&conn, "ck3", "hash3").unwrap();
        assert!(unsettled_ledger_intents(&conn).unwrap().is_empty());
    }

    #[test]
    fn work_waiting_on_funding_nothing_is_chasing_is_counted_not_fixed() {
        let conn = intent_db("ck4");
        conn.execute(
            "INSERT INTO jobs(job_id,work_json,status,reserved_mcu,created_at) \
             VALUES('job_x','{}','funding',10,0)",
            [],
        )
        .unwrap();
        // A pending intent names it, so the daemon is still on the case.
        assert_eq!(orphaned_funding_objects(&conn).unwrap(), 0);
        // Park that intent and the job is left waiting on a ledger entry that
        // nothing will ever propose. The count is the whole remedy: writing the
        // entry locally would be the coordinator deciding CU into existence.
        abandon_ledger_intent(&conn, "ck4", "claim mismatch").unwrap();
        assert_eq!(orphaned_funding_objects(&conn).unwrap(), 1);
    }

    #[test]
    fn a_seeded_job_and_the_intent_that_pays_for_it_land_together() {
        let mut conn = open(":memory:").expect("in-memory coordinator database");
        let work = WorkSpec::PrimeCount { start: 2, end: 100 };
        seed_system_job_with_id(
            &mut conn,
            "job_seeded",
            work,
            1,
            Some(("ck_seed", "community_reserve", "{}")),
        )
        .unwrap();
        // The job is parked waiting on funding, and the intent that will pay
        // for it exists in the same breath -- so nothing is stranded.
        assert_eq!(orphaned_funding_objects(&conn).unwrap(), 0);
        let rows = unsettled_ledger_intents(&conn).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].object_id, "job_seeded");
    }
}
