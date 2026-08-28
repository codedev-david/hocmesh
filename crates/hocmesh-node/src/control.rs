//! The loopback control surface a desktop app drives the daemon through.
//!
//! Everything here is already readable by the operator: the limits are a file
//! in their own home, the capabilities are their own hardware, and the counters
//! describe their own machine. The surface exists because a *running* daemon
//! holds state that is nowhere on disk -- how long it has been up, what it has
//! finished, whether the coordinator answered the last poll -- and a UI that
//! could not see that would be reduced to guessing from log files.
//!
//! Two things keep it from being a new way into the machine. It binds loopback
//! only, so nothing off-host can reach it at all; and every route but `/health`
//! demands a bearer token written to a file inside the node home when the
//! daemon starts. Holding the token therefore means being able to read that
//! file, which means already being the same OS user -- who could have read
//! `limits.json` directly. The token adds no authority; it only stops another
//! user's process on the same host from borrowing this one's.

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use hocmesh_core::{hardware, limits::ResourceLimits};
use hocmesh_protocol::NodeCapabilities;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::Notify;

/// Where a running daemon publishes how to reach it.
///
/// The desktop app finds a daemon by reading this, which is also how it tells
/// "a daemon is running" from "a daemon is not": the file is written after the
/// listener is bound and removed when the daemon stops.
pub const ENDPOINT_FILE: &str = "control.json";

/// Seconds without a successful coordinator exchange before the daemon calls
/// itself disconnected.
///
/// Polls happen on the order of a second and heartbeats every ten, so a minute
/// of silence is well past "the network hiccuped" and into "tell the operator".
const STALE_AFTER_SECS: i64 = 60;

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// The address and credential of a running daemon control surface.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ControlEndpoint {
    pub port: u16,
    pub token: String,
    pub pid: u32,
    pub version: String,
    pub started_unix: i64,
}

impl ControlEndpoint {
    pub fn path(home: &Path) -> PathBuf {
        home.join(ENDPOINT_FILE)
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Read the endpoint a daemon published, if there is one.
    ///
    /// A missing file is `None` rather than an error: "no daemon is running" is
    /// an ordinary answer to give a dashboard, not a failure. Unparsable
    /// content is also `None` -- a half-written or stale file from a daemon
    /// that was killed says nothing useful, and treating it as fatal would
    /// leave the UI stuck on an error it cannot clear.
    pub fn load(home: &Path) -> Option<Self> {
        let raw = std::fs::read_to_string(Self::path(home)).ok()?;
        serde_json::from_str(&raw).ok()
    }

    /// Publish this endpoint so anything on this machine can find the daemon.
    ///
    /// Public so a front end's tests can stage a home that looks like one with
    /// a daemon in it. Writing the file is not a way to fake a running node:
    /// every caller that reads it then has to connect to the port and present
    /// the token, and a file pointing at nothing simply fails to attach.
    pub fn write(&self, home: &Path) -> Result<()> {
        std::fs::create_dir_all(home).with_context(|| format!("creating {}", home.display()))?;
        let path = Self::path(home);
        std::fs::write(&path, serde_json::to_vec_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))?;
        restrict_to_owner(&path);
        Ok(())
    }

    fn remove(home: &Path) {
        let _ = std::fs::remove_file(Self::path(home));
    }
}

/// Narrow the endpoint file to its owner where the OS expresses that in a mode.
///
/// The token is the whole access control, so a world-readable file would hand
/// it to every account on a shared host. On Windows the equivalent is inherited
/// ACLs on the user profile, which the node home already sits inside, and there
/// is no mode to set.
fn restrict_to_owner(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    #[cfg(not(unix))]
    let _ = path;
}

/// A fresh bearer token.
///
/// 256 bits from the OS generator. It lives for one daemon run and is never
/// reused, so it does not need to be memorable or derivable -- only unguessable
/// by a process that cannot read the file it is written to.
fn new_token() -> String {
    use rand_core::RngCore;
    let mut bytes = [0u8; 32];
    rand_core::OsRng.fill_bytes(&mut bytes);
    hex::encode(bytes)
}

/// Compare two tokens without leaking how far they matched.
///
/// A byte-by-byte `==` returns sooner for a wrong first character than for a
/// wrong last one, which is enough to recover a secret one character at a time
/// from a local caller who can time it precisely. Folding every byte into the
/// same accumulator gives every comparison the same shape.
fn tokens_match(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Counters a running daemon keeps that exist nowhere else.
///
/// Atomics rather than a lock because the writers are the worker loops on their
/// hot path and the only reader is a human refreshing a window; making workers
/// queue behind a dashboard would be exactly backwards.
#[derive(Debug, Default)]
pub struct DaemonMetrics {
    jobs_completed: AtomicU64,
    jobs_failed: AtomicU64,
    inferences_completed: AtomicU64,
    mcu_earned: AtomicI64,
    last_contact_unix: AtomicI64,
    coordinator_reachable: AtomicBool,
    last_error: Mutex<Option<String>>,
}

impl DaemonMetrics {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record a shard finished and what it paid.
    pub fn record_completion(&self, mcu: i64) {
        self.jobs_completed.fetch_add(1, Ordering::Relaxed);
        self.mcu_earned.fetch_add(mcu, Ordering::Relaxed);
        self.record_contact();
    }

    /// Record a shard this node took and could not finish.
    pub fn record_failure(&self, error: impl std::fmt::Display) {
        self.jobs_failed.fetch_add(1, Ordering::Relaxed);
        self.set_last_error(error);
    }

    /// Record an inference batch delivered.
    pub fn record_inference(&self) {
        self.inferences_completed.fetch_add(1, Ordering::Relaxed);
        self.record_contact();
    }

    /// Note that the coordinator answered just now.
    pub fn record_contact(&self) {
        self.last_contact_unix.store(now_unix(), Ordering::Relaxed);
        self.coordinator_reachable.store(true, Ordering::Relaxed);
    }

    /// Note that the coordinator did not answer, and why.
    pub fn record_unreachable(&self, error: impl std::fmt::Display) {
        self.coordinator_reachable.store(false, Ordering::Relaxed);
        self.set_last_error(error);
    }

    fn set_last_error(&self, error: impl std::fmt::Display) {
        if let Ok(mut slot) = self.last_error.lock() {
            *slot = Some(error.to_string());
        }
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.lock().ok().and_then(|slot| slot.clone())
    }

    /// Whether the coordinator is answering, allowing for a quiet stretch.
    ///
    /// A daemon that has simply had nothing to do still heartbeats, so silence
    /// past [`STALE_AFTER_SECS`] is a real disconnection rather than an idle
    /// mesh -- but a daemon that only just started has not had a chance to talk
    /// to anyone yet, and calling that "disconnected" would flash a red light
    /// at the operator on every single start.
    fn connected(&self, started_unix: i64) -> bool {
        if !self.coordinator_reachable.load(Ordering::Relaxed) {
            return false;
        }
        let last = self.last_contact_unix.load(Ordering::Relaxed);
        let reference = last.max(started_unix);
        now_unix() - reference < STALE_AFTER_SECS
    }
}

/// Everything the control surface can answer from, shared with the daemon.
#[derive(Clone)]
pub struct ControlState {
    pub home: PathBuf,
    pub node_id: String,
    pub coordinator: String,
    pub workers: usize,
    pub started_unix: i64,
    pub version: String,
    /// What the daemon is currently advertising, which the heartbeat re-reads
    /// every tick -- so writing here is what makes a limits change take effect
    /// without a restart.
    pub capabilities: Arc<RwLock<NodeCapabilities>>,
    /// The machine as detected, before any limit was applied. Kept so that
    /// raising a limit can give back what lowering it took away, without
    /// re-running detection on the request path.
    pub detected: Arc<NodeCapabilities>,
    pub runtime_available: bool,
    pub metrics: Arc<DaemonMetrics>,
    pub shutdown: Arc<Notify>,
    token: String,
}

/// A running daemon, as a dashboard sees it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DaemonState {
    pub node_id: String,
    pub coordinator: String,
    pub version: String,
    pub started_unix: i64,
    pub uptime_secs: i64,
    pub workers: usize,
    pub connected: bool,
    pub jobs_completed: u64,
    pub jobs_failed: u64,
    pub inferences_completed: u64,
    pub mcu_earned: i64,
    pub last_contact_unix: i64,
    pub last_error: Option<String>,
    pub ai_offered: bool,
}

/// What a limits change did, including what it could not do yet.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LimitsUpdate {
    pub limits: ResourceLimits,
    pub capabilities: NodeCapabilities,
    /// Set when the new limits imply a different worker count than the one
    /// currently running. Workers are spawned at start and a live daemon cannot
    /// grow or shrink the pool, so saying "saved" without saying "not yet in
    /// force" would be a lie the operator only discovers by measuring.
    pub restart_required: bool,
}

/// The limits an operator is asking for. Every field is optional so a UI can
/// send only what the operator touched.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LimitsRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_percent: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_percent: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_percent: Option<u8>,
    /// `Some(None)` is "auto" and `None` is "leave it alone", which are
    /// different answers -- so this is deliberately doubly wrapped rather than
    /// collapsed into one `Option`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai: Option<Option<bool>>,
}

impl LimitsRequest {
    /// Fold this request onto stored limits.
    /// Fold this request onto an existing consent record.
    ///
    /// Public so a front end can apply the same partial-update rules when no
    /// daemon is running as the daemon applies when one is. Two
    /// implementations of "what does an absent field mean" would eventually
    /// disagree, and the disagreement would be about how much of a machine is
    /// lent.
    pub fn apply_to(&self, limits: &mut ResourceLimits) {
        if let Some(v) = self.cpu_percent {
            limits.cpu_percent = v;
        }
        if let Some(v) = self.gpu_percent {
            limits.gpu_percent = v;
        }
        if let Some(v) = self.memory_percent {
            limits.memory_percent = v;
        }
        if let Some(v) = self.ai {
            limits.ai = v;
        }
    }
}

/// Recompute what a node advertises from a detection and stored limits.
///
/// This is the one place the three-step rule lives -- detect, apply limits,
/// settle AI readiness -- so a limits change made through the UI lands on
/// exactly the capabilities a restart would have produced.
pub fn advertised_capabilities(
    detected: &NodeCapabilities,
    limits: &ResourceLimits,
    runtime_available: bool,
) -> NodeCapabilities {
    let mut caps = detected.clone();
    hardware::apply_limits(&mut caps, limits);
    hardware::apply_ai_readiness(&mut caps, limits, runtime_available);
    caps
}

#[derive(Debug)]
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

fn internal(error: impl std::fmt::Display) -> ApiError {
    ApiError(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
}

pub fn router(state: ControlState) -> Router {
    let guarded = Router::new()
        .route("/v1/state", get(state_handler))
        .route("/v1/capabilities", get(capabilities_handler))
        .route("/v1/limits", get(limits_handler).put(set_limits_handler))
        .route("/v1/shutdown", post(shutdown_handler))
        .route_layer(middleware::from_fn_with_state(state.clone(), authorize));

    Router::new()
        .route("/health", get(health_handler))
        .merge(guarded)
        .with_state(state)
}

/// `/health` is deliberately outside the token check and deliberately empty of
/// facts. A caller needs to know the port is a hocMESH daemon before it can
/// sensibly present a stale-token error, and answering that discloses nothing
/// that scanning the port did not already.
async fn health_handler(State(state): State<ControlState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "service": "hocmesh-node", "version": state.version }))
}

async fn authorize(
    State(state): State<ControlState>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let presented = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .unwrap_or_default();
    if !tokens_match(presented, &state.token) {
        return Err(ApiError(
            StatusCode::UNAUTHORIZED,
            "control token missing or wrong".into(),
        ));
    }
    Ok(next.run(request).await)
}

async fn state_handler(State(state): State<ControlState>) -> Json<DaemonState> {
    let ai_offered = state
        .capabilities
        .read()
        .map(|caps| caps.ai_runtime_ready)
        .unwrap_or(false);
    Json(DaemonState {
        node_id: state.node_id.clone(),
        coordinator: state.coordinator.clone(),
        version: state.version.clone(),
        started_unix: state.started_unix,
        uptime_secs: (now_unix() - state.started_unix).max(0),
        workers: state.workers,
        connected: state.metrics.connected(state.started_unix),
        jobs_completed: state.metrics.jobs_completed.load(Ordering::Relaxed),
        jobs_failed: state.metrics.jobs_failed.load(Ordering::Relaxed),
        inferences_completed: state.metrics.inferences_completed.load(Ordering::Relaxed),
        mcu_earned: state.metrics.mcu_earned.load(Ordering::Relaxed),
        last_contact_unix: state.metrics.last_contact_unix.load(Ordering::Relaxed),
        last_error: state.metrics.last_error(),
        ai_offered,
    })
}

async fn capabilities_handler(
    State(state): State<ControlState>,
) -> Result<Json<NodeCapabilities>, ApiError> {
    let caps = state
        .capabilities
        .read()
        .map_err(|_| internal("capability lock poisoned"))?
        .clone();
    Ok(Json(caps))
}

async fn limits_handler(
    State(state): State<ControlState>,
) -> Result<Json<ResourceLimits>, ApiError> {
    ResourceLimits::load_or_default(&state.home)
        .map(Json)
        .map_err(internal)
}

async fn set_limits_handler(
    State(state): State<ControlState>,
    Json(request): Json<LimitsRequest>,
) -> Result<Json<LimitsUpdate>, ApiError> {
    let mut limits = ResourceLimits::load_or_default(&state.home).map_err(internal)?;
    request.apply_to(&mut limits);
    // Refuse before writing. A rejected request must leave the stored consent
    // exactly as it was, or a typo in a dashboard could widen a share nobody
    // agreed to widen.
    limits
        .validate()
        .map_err(|error| ApiError(StatusCode::BAD_REQUEST, error.to_string()))?;
    limits.save(&state.home).map_err(internal)?;

    let caps = advertised_capabilities(&state.detected, &limits, state.runtime_available);
    let restart_required = caps.shared_logical_cpus != state.workers;
    *state
        .capabilities
        .write()
        .map_err(|_| internal("capability lock poisoned"))? = caps.clone();

    Ok(Json(LimitsUpdate {
        limits,
        capabilities: caps,
        restart_required,
    }))
}

async fn shutdown_handler(State(state): State<ControlState>) -> StatusCode {
    // `notify_one` rather than `notify_waiters` so the request wins even if it
    // arrives before the daemon parks on the signal; a stored permit is
    // collected whenever the waiter gets there.
    state.shutdown.notify_one();
    StatusCode::ACCEPTED
}

/// The parts of [`ControlState`] the daemon knows before the port is bound.
pub struct ControlSeed {
    pub home: PathBuf,
    pub node_id: String,
    pub coordinator: String,
    pub workers: usize,
    pub version: String,
    pub capabilities: Arc<RwLock<NodeCapabilities>>,
    pub detected: Arc<NodeCapabilities>,
    pub runtime_available: bool,
    pub metrics: Arc<DaemonMetrics>,
    pub shutdown: Arc<Notify>,
}

/// A bound control surface, and the file that advertises it.
pub struct ControlServer {
    pub endpoint: ControlEndpoint,
    home: PathBuf,
    listener: tokio::net::TcpListener,
    router: Router,
}

impl ControlServer {
    /// Bind the control surface and publish how to reach it.
    ///
    /// `port` of 0 asks the OS for a free one, which is the default: a fixed
    /// port would collide between two homes on one machine, and the desktop app
    /// reads the real port out of the endpoint file anyway.
    pub async fn bind(seed: ControlSeed, port: u16) -> Result<(Self, ControlState)> {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
            .await
            .with_context(|| format!("binding the control surface on 127.0.0.1:{port}"))?;
        let bound = listener.local_addr()?.port();
        let token = new_token();
        let started_unix = now_unix();
        let state = ControlState {
            home: seed.home.clone(),
            node_id: seed.node_id,
            coordinator: seed.coordinator,
            workers: seed.workers,
            started_unix,
            version: seed.version.clone(),
            capabilities: seed.capabilities,
            detected: seed.detected,
            runtime_available: seed.runtime_available,
            metrics: seed.metrics,
            shutdown: seed.shutdown,
            token: token.clone(),
        };
        let endpoint = ControlEndpoint {
            port: bound,
            token,
            pid: std::process::id(),
            version: seed.version,
            started_unix,
        };
        endpoint.write(&seed.home)?;
        Ok((
            Self {
                endpoint,
                home: seed.home,
                listener,
                router: router(state.clone()),
            },
            state,
        ))
    }

    /// Serve until the task is dropped. The daemon aborts it at shutdown.
    pub async fn serve(self) -> std::io::Result<()> {
        axum::serve(self.listener, self.router).await
    }

    /// Withdraw the advertisement.
    ///
    /// Leaving the file behind would tell the next dashboard that a dead daemon
    /// is running, and would keep a spent token on disk.
    pub fn retire(home: &Path) {
        ControlEndpoint::remove(home);
    }

    pub fn home(&self) -> &Path {
        &self.home
    }
}

/// Ask a running daemon to stop, if one is running.
///
/// Returns whether a daemon was actually reached. A stale endpoint file is not
/// an error -- it is the ordinary result of a daemon that was killed rather
/// than asked -- so the caller learns "nothing to stop" instead of a failure.
pub async fn request_shutdown(home: &Path) -> Result<bool> {
    let Some(endpoint) = ControlEndpoint::load(home) else {
        return Ok(false);
    };
    let response = reqwest::Client::new()
        .post(format!("{}/v1/shutdown", endpoint.base_url()))
        .bearer_auth(&endpoint.token)
        .send()
        .await;
    match response {
        Ok(response) if response.status().is_success() => Ok(true),
        Ok(response) => anyhow::bail!("daemon refused the shutdown request: {}", response.status()),
        // Unreachable means the daemon behind this file is gone. Clear it so
        // the next reader is not told about a daemon that no longer exists.
        Err(_) => {
            ControlEndpoint::remove(home);
            Ok(false)
        }
    }
}

/// A typed client for the control surface, used by the desktop app.
pub struct ControlClient {
    http: reqwest::Client,
    endpoint: ControlEndpoint,
}

impl ControlClient {
    /// Attach to whatever daemon this home is running, if any.
    pub fn attach(home: &Path) -> Option<Self> {
        ControlEndpoint::load(home).map(Self::for_endpoint)
    }

    pub fn for_endpoint(endpoint: ControlEndpoint) -> Self {
        Self {
            http: reqwest::Client::new(),
            endpoint,
        }
    }

    pub fn endpoint(&self) -> &ControlEndpoint {
        &self.endpoint
    }

    async fn get<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T> {
        let response = self
            .http
            .get(format!("{}{path}", self.endpoint.base_url()))
            .bearer_auth(&self.endpoint.token)
            .send()
            .await
            .with_context(|| format!("asking the daemon for {path}"))?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("daemon answered {status} for {path}: {body}");
        }
        serde_json::from_str(&body).with_context(|| format!("reading the daemon answer for {path}"))
    }

    pub async fn state(&self) -> Result<DaemonState> {
        self.get("/v1/state").await
    }

    pub async fn capabilities(&self) -> Result<NodeCapabilities> {
        self.get("/v1/capabilities").await
    }

    pub async fn limits(&self) -> Result<ResourceLimits> {
        self.get("/v1/limits").await
    }

    pub async fn set_limits(&self, request: &LimitsRequest) -> Result<LimitsUpdate> {
        let response = self
            .http
            .put(format!("{}/v1/limits", self.endpoint.base_url()))
            .bearer_auth(&self.endpoint.token)
            .json(request)
            .send()
            .await
            .context("sending new limits to the daemon")?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("daemon refused the limits: {body}");
        }
        serde_json::from_str(&body).context("reading the daemon answer to a limits change")
    }
}

/// Record the outcome of a coordinator exchange on the shared counters.
///
/// The worker loops call this instead of touching the metrics directly so that
/// "a request failed" always both marks the daemon disconnected and stores the
/// reason -- forgetting the second is how a dashboard ends up showing a red
/// light with no explanation.
pub fn note_exchange<T, E: std::fmt::Display>(
    metrics: &DaemonMetrics,
    outcome: &std::result::Result<T, E>,
) {
    match outcome {
        Ok(_) => metrics.record_contact(),
        Err(error) => metrics.record_unreachable(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hocmesh_core::hardware;

    /// A home directory under this run of the test binary.
    ///
    /// The limits routes write a real file, so each test needs its own home or
    /// they overwrite each other when cargo runs them in parallel.
    fn scratch_home(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hocmesh-control-{}-{}-{tag}",
            std::process::id(),
            now_unix()
        ));
        std::fs::create_dir_all(&dir).expect("scratch home");
        dir
    }

    fn a_machine() -> NodeCapabilities {
        let mut caps = hardware::detect_capabilities(false);
        // Detection reads the real machine, which differs between CI runners
        // and desks. Pinning the fields the assertions depend on keeps the test
        // about the control surface rather than about the host it runs on.
        caps.logical_cpus = 8;
        caps.total_memory_bytes = 32 * 1024 * 1024 * 1024;
        caps.gpus.clear();
        caps
    }

    fn a_state(home: PathBuf, workers: usize) -> ControlState {
        let detected = a_machine();
        let limits = ResourceLimits::load_or_default(&home).expect("limits");
        let advertised = advertised_capabilities(&detected, &limits, false);
        ControlState {
            home,
            node_id: "node-under-test".into(),
            coordinator: "http://127.0.0.1:9999".into(),
            workers,
            started_unix: now_unix(),
            version: "0.0.0-test".into(),
            capabilities: Arc::new(RwLock::new(advertised)),
            detected: Arc::new(detected),
            runtime_available: false,
            metrics: DaemonMetrics::new(),
            shutdown: Arc::new(Notify::new()),
            token: "the-expected-token".into(),
        }
    }

    /// Serve a router on a loopback port for the duration of one test.
    async fn serve(state: ControlState) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind");
        let base = format!("http://{}", listener.local_addr().expect("addr"));
        let app = router(state);
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (base, task)
    }

    #[test]
    fn a_token_comparison_rejects_every_way_of_being_wrong() {
        assert!(tokens_match("abc", "abc"));
        assert!(!tokens_match("abc", "abd"), "a differing last byte");
        assert!(!tokens_match("abc", "bbc"), "a differing first byte");
        assert!(!tokens_match("abc", "abcd"), "a prefix is not a match");
        assert!(!tokens_match("", "abc"), "an absent token is not a match");
        assert!(!tokens_match("abc", ""), "and neither is an empty secret");
    }

    #[test]
    fn a_fresh_token_is_long_and_never_repeats() {
        let first = new_token();
        let second = new_token();
        assert_eq!(first.len(), 64, "32 bytes as hex");
        assert_ne!(first, second);
        assert!(first.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn an_endpoint_round_trips_through_the_file_it_advertises() {
        let home = scratch_home("endpoint");
        assert!(
            ControlEndpoint::load(&home).is_none(),
            "no daemon has published here yet"
        );
        let endpoint = ControlEndpoint {
            port: 41234,
            token: "t".into(),
            pid: 7,
            version: "0.3.0".into(),
            started_unix: 1_700_000_000,
        };
        endpoint.write(&home).expect("write");
        assert_eq!(ControlEndpoint::load(&home).as_ref(), Some(&endpoint));
        assert_eq!(endpoint.base_url(), "http://127.0.0.1:41234");
        ControlEndpoint::remove(&home);
        assert!(ControlEndpoint::load(&home).is_none(), "retired");
    }

    #[test]
    fn a_corrupt_endpoint_file_reads_as_no_daemon_rather_than_an_error() {
        // A daemon killed mid-write, or a file truncated by a crash, must not
        // wedge the dashboard on an error the operator cannot clear.
        let home = scratch_home("corrupt");
        std::fs::write(ControlEndpoint::path(&home), b"{not json").expect("write");
        assert!(ControlEndpoint::load(&home).is_none());
    }

    #[test]
    fn a_partial_request_changes_only_what_it_names() {
        let mut limits = ResourceLimits {
            cpu_percent: 50,
            gpu_percent: 60,
            memory_percent: 70,
            ai: Some(true),
        };
        LimitsRequest {
            cpu_percent: Some(10),
            ..Default::default()
        }
        .apply_to(&mut limits);
        assert_eq!(limits.cpu_percent, 10);
        assert_eq!(limits.gpu_percent, 60, "untouched");
        assert_eq!(limits.memory_percent, 70, "untouched");
        assert_eq!(limits.ai, Some(true), "untouched");
    }

    #[test]
    fn asking_for_auto_is_distinguishable_from_asking_for_nothing() {
        // This is the whole reason the field is doubly wrapped. If they
        // collapsed, a dashboard saving the CPU slider would silently reset an
        // operator AI decision back to auto.
        let mut asked = ResourceLimits {
            ai: Some(false),
            ..Default::default()
        };
        LimitsRequest {
            ai: Some(None),
            ..Default::default()
        }
        .apply_to(&mut asked);
        assert_eq!(asked.ai, None, "auto was explicitly requested");

        let mut untouched = ResourceLimits {
            ai: Some(false),
            ..Default::default()
        };
        LimitsRequest::default().apply_to(&mut untouched);
        assert_eq!(untouched.ai, Some(false), "no opinion was expressed");
    }

    #[test]
    fn a_limits_request_serialises_without_the_fields_it_did_not_set() {
        // The wire form is what the desktop app sends, and an absent field is
        // the only way to say "leave this alone".
        let json = serde_json::to_string(&LimitsRequest {
            cpu_percent: Some(25),
            ..Default::default()
        })
        .expect("serialise");
        assert_eq!(json, r#"{"cpu_percent":25}"#);
    }

    #[test]
    fn advertising_follows_detect_then_limit_then_ai() {
        let detected = a_machine();
        let limits = ResourceLimits {
            cpu_percent: 50,
            gpu_percent: 0,
            memory_percent: 25,
            ai: Some(true),
        };
        let caps = advertised_capabilities(&detected, &limits, true);
        assert_eq!(caps.shared_logical_cpus, 4, "half of eight");
        // A quarter of the machine as `ResourceLimits` rounds it: it divides
        // before multiplying so a very large machine cannot overflow, and that
        // costs a few bytes of exactness on the way.
        assert_eq!(caps.shared_memory_bytes, 32 * 1024 * 1024 * 1024 / 100 * 25);
        assert!(
            caps.ai_runtime_ready,
            "the operator opted in with a runtime"
        );
        assert_eq!(
            caps.gpus.len(),
            1,
            "a CPU-only node that opts in still advertises a device"
        );
        assert_eq!(caps.gpus[0].stable_id, hardware::SHARED_CPU_DEVICE_ID);
        assert_eq!(
            caps.gpus[0].memory_mb,
            Some(caps.shared_memory_bytes / (1024 * 1024)),
            "the device reports the lent slice, not the machine"
        );
        assert!(
            caps.gpus[0].memory_mb < Some(caps.total_memory_bytes / (1024 * 1024)),
            "and the slice is smaller than the machine"
        );
    }

    #[test]
    fn a_daemon_that_just_started_is_not_reported_as_disconnected() {
        // Nothing has been polled yet, so `last_contact_unix` is zero. Reading
        // that literally would say "disconnected" for the first minute of every
        // run.
        let metrics = DaemonMetrics::default();
        metrics.record_contact();
        assert!(metrics.connected(now_unix()));
    }

    #[test]
    fn a_long_silence_is_reported_as_disconnected() {
        let metrics = DaemonMetrics::default();
        metrics.record_contact();
        let started_long_ago = now_unix() - (STALE_AFTER_SECS * 3);
        metrics
            .last_contact_unix
            .store(started_long_ago, Ordering::Relaxed);
        assert!(!metrics.connected(started_long_ago));
    }

    #[test]
    fn a_failed_exchange_records_both_the_state_and_the_reason() {
        let metrics = DaemonMetrics::default();
        metrics.record_contact();
        note_exchange::<(), _>(&metrics, &Err("connection refused"));
        assert!(!metrics.connected(now_unix()));
        assert_eq!(metrics.last_error().as_deref(), Some("connection refused"));

        note_exchange::<(), &str>(&metrics, &Ok(()));
        assert!(metrics.connected(now_unix()), "recovery is observable");
    }

    #[test]
    fn counters_add_up_the_way_a_dashboard_reports_them() {
        let metrics = DaemonMetrics::default();
        metrics.record_completion(1_500);
        metrics.record_completion(2_500);
        metrics.record_failure("shard exploded");
        metrics.record_inference();
        assert_eq!(metrics.jobs_completed.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.jobs_failed.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.inferences_completed.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.mcu_earned.load(Ordering::Relaxed), 4_000);
        assert_eq!(metrics.last_error().as_deref(), Some("shard exploded"));
    }

    #[tokio::test]
    async fn health_answers_without_a_token_and_discloses_nothing() {
        let state = a_state(scratch_home("health"), 4);
        let (base, task) = serve(state).await;
        let body: serde_json::Value = reqwest::get(format!("{base}/health"))
            .await
            .expect("request")
            .json()
            .await
            .expect("json");
        assert_eq!(body["service"], "hocmesh-node");
        assert!(
            body.get("node_id").is_none(),
            "identity is behind the token"
        );
        task.abort();
    }

    #[tokio::test]
    async fn every_other_route_refuses_a_caller_without_the_token() {
        let state = a_state(scratch_home("unauth"), 4);
        let (base, task) = serve(state).await;
        let http = reqwest::Client::new();
        for path in ["/v1/state", "/v1/capabilities", "/v1/limits"] {
            let status = http
                .get(format!("{base}{path}"))
                .send()
                .await
                .expect("request")
                .status();
            assert_eq!(status, 401, "{path} must not answer an untokened caller");
        }
        let status = http
            .post(format!("{base}/v1/shutdown"))
            .send()
            .await
            .expect("request")
            .status();
        assert_eq!(status, 401, "and least of all shutdown");
        task.abort();
    }

    #[tokio::test]
    async fn a_wrong_token_is_refused_as_firmly_as_none() {
        let state = a_state(scratch_home("wrongtoken"), 4);
        let (base, task) = serve(state).await;
        let status = reqwest::Client::new()
            .get(format!("{base}/v1/state"))
            .bearer_auth("not-the-expected-token")
            .send()
            .await
            .expect("request")
            .status();
        assert_eq!(status, 401);
        task.abort();
    }

    #[tokio::test]
    async fn state_reports_the_running_daemon() {
        let state = a_state(scratch_home("state"), 3);
        state.metrics.record_completion(7_000);
        let (base, task) = serve(state).await;
        let body: DaemonState = reqwest::Client::new()
            .get(format!("{base}/v1/state"))
            .bearer_auth("the-expected-token")
            .send()
            .await
            .expect("request")
            .json()
            .await
            .expect("json");
        assert_eq!(body.node_id, "node-under-test");
        assert_eq!(body.workers, 3);
        assert_eq!(body.jobs_completed, 1);
        assert_eq!(body.mcu_earned, 7_000);
        assert!(body.connected);
        assert!(body.uptime_secs >= 0);
        task.abort();
    }

    #[tokio::test]
    async fn capabilities_report_the_lent_share_not_the_machine() {
        let home = scratch_home("caps");
        ResourceLimits {
            cpu_percent: 25,
            gpu_percent: 0,
            memory_percent: 25,
            ai: None,
        }
        .save(&home)
        .expect("save");
        let state = a_state(home, 2);
        let (base, task) = serve(state).await;
        let caps: NodeCapabilities = reqwest::Client::new()
            .get(format!("{base}/v1/capabilities"))
            .bearer_auth("the-expected-token")
            .send()
            .await
            .expect("request")
            .json()
            .await
            .expect("json");
        assert_eq!(caps.logical_cpus, 8, "the machine is still described");
        assert_eq!(caps.shared_logical_cpus, 2, "but only a quarter is lent");
        assert_eq!(caps.shared_memory_bytes, 32 * 1024 * 1024 * 1024 / 100 * 25);
        task.abort();
    }

    #[tokio::test]
    async fn changing_limits_saves_them_and_re_advertises_without_a_restart() {
        let home = scratch_home("setlimits");
        let state = a_state(home.clone(), 8);
        let capabilities = state.capabilities.clone();
        let (base, task) = serve(state).await;

        let update: LimitsUpdate = reqwest::Client::new()
            .put(format!("{base}/v1/limits"))
            .bearer_auth("the-expected-token")
            .json(&LimitsRequest {
                cpu_percent: Some(50),
                memory_percent: Some(10),
                ..Default::default()
            })
            .send()
            .await
            .expect("request")
            .json()
            .await
            .expect("json");

        assert_eq!(update.limits.cpu_percent, 50);
        assert_eq!(update.capabilities.shared_logical_cpus, 4);
        assert!(
            update.restart_required,
            "eight workers are running but only four are now lent"
        );
        assert_eq!(
            ResourceLimits::load_or_default(&home)
                .expect("reload")
                .cpu_percent,
            50,
            "the consent record on disk is what actually changed"
        );
        assert_eq!(
            capabilities.read().expect("lock").shared_logical_cpus,
            4,
            "the next heartbeat advertises the new share"
        );
        task.abort();
    }

    #[tokio::test]
    async fn an_impossible_limit_is_refused_and_changes_nothing() {
        let home = scratch_home("badlimits");
        ResourceLimits {
            cpu_percent: 40,
            gpu_percent: 40,
            memory_percent: 40,
            ai: None,
        }
        .save(&home)
        .expect("save");
        let state = a_state(home.clone(), 4);
        let (base, task) = serve(state).await;

        let status = reqwest::Client::new()
            .put(format!("{base}/v1/limits"))
            .bearer_auth("the-expected-token")
            .json(&LimitsRequest {
                cpu_percent: Some(200),
                ..Default::default()
            })
            .send()
            .await
            .expect("request")
            .status();
        assert_eq!(status, 400);
        assert_eq!(
            ResourceLimits::load_or_default(&home)
                .expect("reload")
                .cpu_percent,
            40,
            "a refused request must not have written anything"
        );
        task.abort();
    }

    #[tokio::test]
    async fn shutdown_wakes_the_daemon_waiting_on_it() {
        let state = a_state(scratch_home("shutdown"), 1);
        let shutdown = state.shutdown.clone();
        let (base, task) = serve(state).await;

        let status = reqwest::Client::new()
            .post(format!("{base}/v1/shutdown"))
            .bearer_auth("the-expected-token")
            .send()
            .await
            .expect("request")
            .status();
        assert_eq!(status, 202);

        // The permit outlives the request, so a daemon that parks on the signal
        // after the call still stops.
        tokio::time::timeout(std::time::Duration::from_secs(5), shutdown.notified())
            .await
            .expect("a daemon waiting on the signal is woken");
        task.abort();
    }

    #[tokio::test]
    async fn a_client_drives_the_surface_the_way_the_desktop_app_will() {
        let home = scratch_home("client");
        let seed = ControlSeed {
            home: home.clone(),
            node_id: "node-under-test".into(),
            coordinator: "http://127.0.0.1:9999".into(),
            workers: 4,
            version: "0.0.0-test".into(),
            capabilities: Arc::new(RwLock::new(a_machine())),
            detected: Arc::new(a_machine()),
            runtime_available: false,
            metrics: DaemonMetrics::new(),
            shutdown: Arc::new(Notify::new()),
        };
        let (server, _state) = ControlServer::bind(seed, 0).await.expect("bind");
        let task = tokio::spawn(async move {
            let _ = server.serve().await;
        });

        let client = ControlClient::attach(&home).expect("the endpoint file advertises the daemon");
        assert_eq!(
            client.state().await.expect("state").node_id,
            "node-under-test"
        );
        assert_eq!(client.limits().await.expect("limits").cpu_percent, 50);
        let update = client
            .set_limits(&LimitsRequest {
                gpu_percent: Some(0),
                ai: Some(Some(true)),
                ..Default::default()
            })
            .await
            .expect("set limits");
        assert_eq!(update.limits.gpu_percent, 0);
        assert_eq!(update.limits.ai, Some(true));
        assert!(client.capabilities().await.is_ok());

        assert!(
            request_shutdown(&home).await.expect("shutdown"),
            "a running daemon was reached"
        );

        ControlServer::retire(&home);
        assert!(
            ControlClient::attach(&home).is_none(),
            "a retired daemon cannot be attached to"
        );
        task.abort();
    }

    #[tokio::test]
    async fn asking_a_dead_daemon_to_stop_clears_its_stale_advertisement() {
        let home = scratch_home("stale");
        // A port nothing is listening on, which is what a killed daemon leaves
        // behind: the file survives the process.
        ControlEndpoint {
            port: 1,
            token: "spent".into(),
            pid: 999_999,
            version: "0.0.0-test".into(),
            started_unix: 0,
        }
        .write(&home)
        .expect("write");

        assert!(
            !request_shutdown(&home).await.expect("no error"),
            "nothing was there to stop"
        );
        assert!(
            ControlEndpoint::load(&home).is_none(),
            "and the dead advertisement was cleared"
        );
    }

    #[tokio::test]
    async fn asking_a_home_with_no_daemon_to_stop_is_not_an_error() {
        let home = scratch_home("nodaemon");
        assert!(!request_shutdown(&home).await.expect("no error"));
    }
}
