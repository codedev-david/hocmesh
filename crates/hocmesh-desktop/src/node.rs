//! The app's connection to one node: read it, steer it, start and stop it.
//!
//! This is where the reading actually happens. [`crate::dashboard`] decides
//! what a reading *means*; this decides where each one comes from, and its
//! central rule is that a refresh is best-effort. Four sources answer
//! independently -- the endpoint file, the daemon's control surface, the disk,
//! and the coordinator -- and any of them can be down while the others are
//! fine. A refresh therefore never fails as a whole: it returns what it got
//! and reports what it did not.
//!
//! That matters because the moments an operator opens this window are exactly
//! the moments when something is not answering.

use crate::dashboard::{Readings, Snapshot};
use crate::settings::Settings;
use crate::supervisor::{LaunchOptions, RunState, Supervisor};
use anyhow::{Context, Result};
use hocmesh::client::HocMeshClient;
use hocmesh::control::{ControlClient, LimitsRequest, LimitsUpdate};
use hocmesh_core::hardware;
use hocmesh_core::identity::{NodeIdentity, identity_path};
use hocmesh_core::limits::ResourceLimits;
use hocmesh_protocol::NodeCapabilities;
use std::path::Path;

/// How many ledger rows one refresh pulls.
///
/// Enough to fill the panel and scroll a little, not enough to make the
/// refresh that runs every few seconds expensive.
pub const LEDGER_PAGE: u32 = 50;

/// The unix clock, in the one place that reads it.
///
/// Everything downstream takes `now` as an argument, which is what lets the
/// formatting and snapshot rules be tested without a clock.
pub fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Detect this machine, without asking the network anything.
///
/// The desktop app describes hardware even when no daemon is running, so it
/// does its own detection rather than depending on one. Benchmarks are skipped:
/// a window that spun the CPU every refresh to redraw a number would be lending
/// the operator's machine to itself.
pub fn detect() -> NodeCapabilities {
    hardware::detect_capabilities(false)
}

/// Whether an AI runtime is installed for this home.
pub fn runtime_available(home: &Path) -> bool {
    hocmesh_gpu::runtime::installed_runtime(home).is_some()
}

/// A client for the coordinator, if this home has an identity yet.
///
/// `load_or_create` is deliberately not used. A home that has never run a node
/// has no identity, and minting one because a window was opened would create a
/// node that never asked to exist -- and would then show a 404 balance for it.
pub fn coordinator_client(home: &Path, coordinator: &str) -> Option<HocMeshClient> {
    if !identity_path(home).exists() {
        return None;
    }
    let identity = NodeIdentity::load_or_create(home).ok()?;
    Some(HocMeshClient::new(coordinator, identity))
}

/// One node, as this app sees it.
pub struct Node {
    settings: Settings,
    supervisor: Supervisor,
    capabilities: NodeCapabilities,
}

impl Node {
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            supervisor: Supervisor::new(),
            capabilities: detect(),
        }
    }

    /// A node wired to a known executable, for tests.
    pub fn with_supervisor(settings: Settings, supervisor: Supervisor) -> Self {
        Self {
            settings,
            supervisor,
            capabilities: detect(),
        }
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    pub fn set_settings(&mut self, settings: Settings) {
        self.settings = settings.normalised();
    }

    pub fn supervisor_mut(&mut self) -> &mut Supervisor {
        &mut self.supervisor
    }

    pub fn run_state(&mut self) -> RunState {
        self.supervisor.state(&self.settings.home)
    }

    pub fn launch_options(&self) -> LaunchOptions {
        LaunchOptions {
            home: self.settings.home.clone(),
            coordinator: self.settings.coordinator.clone(),
            workers: self.settings.workers,
            no_ai: self.settings.no_ai,
            control_port: self.settings.control_port,
        }
    }

    pub fn start(&mut self) -> Result<RunState> {
        crate::supervisor::check_home(&self.settings.home)?;
        self.supervisor.start(&self.launch_options())
    }

    pub async fn stop(&mut self) -> Result<bool> {
        let home = self.settings.home.clone();
        self.supervisor.stop(&home).await
    }

    /// Change what this machine lends.
    ///
    /// Routed through the running daemon when there is one, so the change
    /// takes effect on the next heartbeat and the daemon -- not the app -- is
    /// the one that validates and writes the consent record. With no daemon
    /// running the app writes the file itself, after the same validation, so
    /// an operator can set limits before ever starting a node.
    pub async fn set_limits(&mut self, request: LimitsRequest) -> Result<LimitsUpdate> {
        let home = self.settings.home.clone();
        if let Some(client) = ControlClient::attach(&home) {
            return client.set_limits(&request).await;
        }
        let mut limits = ResourceLimits::load_or_default(&home)?;
        request.apply_to(&mut limits);
        limits
            .validate()
            .context("those limits would not be a valid share of this machine")?;
        limits.save(&home)?;
        let capabilities = hocmesh::control::advertised_capabilities(
            &self.capabilities,
            &limits,
            runtime_available(&home),
        );
        Ok(LimitsUpdate {
            limits,
            capabilities,
            // Nothing is running, so nothing is working to the old numbers.
            restart_required: false,
        })
    }

    /// Read everything the window shows, tolerating whatever is down.
    pub async fn snapshot(&mut self, before: Option<u64>) -> Snapshot {
        let home = self.settings.home.clone();
        let run_state = self.run_state();
        let stored_limits = ResourceLimits::load_or_default(&home).unwrap_or_default();

        let mut daemon = None;
        let mut live_limits = None;
        let mut live_capabilities = None;
        if let Some(client) = ControlClient::attach(&home) {
            daemon = client.state().await.ok();
            live_limits = client.limits().await.ok();
            live_capabilities = client.capabilities().await.ok();
        }

        // The daemon's own view of what it advertises is preferred: it is the
        // process that answered the coordinator, and this window may have been
        // opened long after it started.
        let capabilities = live_capabilities.unwrap_or_else(|| {
            hocmesh::control::advertised_capabilities(
                &self.capabilities,
                live_limits.as_ref().unwrap_or(&stored_limits),
                runtime_available(&home),
            )
        });

        let coordinator = daemon
            .as_ref()
            .map(|state| state.coordinator.clone())
            .unwrap_or_else(|| self.settings.coordinator.clone());

        let mut balance = None;
        let mut history = None;
        let mut ledger_error = None;
        match coordinator_client(&home, &coordinator) {
            None => {
                ledger_error = Some(
                    "this home has no node identity yet -- start the node once to create one"
                        .into(),
                )
            }
            Some(client) => match client.balance().await {
                Err(error) => ledger_error = Some(format!("{error}")),
                Ok(found) => {
                    balance = Some(found);
                    match client.history(before, LEDGER_PAGE).await {
                        // A balance without its history is still worth
                        // showing: the number an operator came for is the
                        // balance, and the table is the detail behind it.
                        Err(error) => ledger_error = Some(format!("{error}")),
                        Ok(page) => history = Some(page),
                    }
                }
            },
        }

        Snapshot::build(Readings {
            run_state: &run_state,
            coordinator: &coordinator,
            daemon: daemon.as_ref(),
            live_limits: live_limits.as_ref(),
            stored_limits: &stored_limits,
            capabilities: &capabilities,
            balance: balance.as_ref(),
            history: history.as_ref().map(|page| {
                (
                    page.entries.as_slice(),
                    page.authoritative,
                    page.next_before,
                )
            }),
            ledger_error,
            now: now_unix(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hocmesh_core::limits::ResourceLimits;
    use std::fs;
    use std::path::PathBuf;

    fn scratch(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("hocmesh-desktop-{tag}-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn node_for(home: &Path) -> Node {
        let settings = Settings {
            home: home.to_path_buf(),
            coordinator: "http://127.0.0.1:1".into(),
            ..Settings::default()
        };
        Node::with_supervisor(settings, Supervisor::with_binary(None))
    }

    #[test]
    fn launch_options_carry_the_settings_the_operator_chose() {
        let home = scratch("launch-options");
        let mut node = node_for(&home);
        node.set_settings(Settings {
            home: home.clone(),
            coordinator: "https://mesh.example/".into(),
            workers: Some(4),
            no_ai: true,
            control_port: 7799,
            start_node_with_app: true,
        });
        let options = node.launch_options();
        assert_eq!(
            options.coordinator, "https://mesh.example",
            "the trailing slash is trimmed on the way in, not on every request"
        );
        assert_eq!(options.workers, Some(4));
        assert!(options.no_ai);
        assert_eq!(options.control_port, 7799);
        fs::remove_dir_all(home).unwrap();
    }

    #[tokio::test]
    async fn limits_can_be_set_before_a_node_has_ever_run() {
        // An operator should be able to decide what they lend and only then
        // join a mesh. Requiring a running daemon to record consent would put
        // the two in the wrong order.
        let home = scratch("limits-offline");
        let mut node = node_for(&home);
        let update = node
            .set_limits(LimitsRequest {
                cpu_percent: Some(25),
                memory_percent: Some(30),
                gpu_percent: None,
                ai: None,
            })
            .await
            .unwrap();
        assert_eq!(update.limits.cpu_percent, 25);
        assert_eq!(update.limits.memory_percent, 30);
        assert!(!update.restart_required, "nothing was running to restart");

        let on_disk = ResourceLimits::load_or_default(&home).unwrap();
        assert_eq!(on_disk.cpu_percent, 25);
        fs::remove_dir_all(home).unwrap();
    }

    #[tokio::test]
    async fn an_invalid_share_is_refused_and_nothing_is_written() {
        let home = scratch("limits-invalid");
        let mut node = node_for(&home);
        let before = ResourceLimits::load_or_default(&home).unwrap();
        let error = node
            .set_limits(LimitsRequest {
                cpu_percent: Some(200),
                memory_percent: None,
                gpu_percent: None,
                ai: None,
            })
            .await
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("valid share"),
            "unhelpful error: {error:#}"
        );
        assert_eq!(
            ResourceLimits::load_or_default(&home).unwrap(),
            before,
            "a refused change must not leave a partial one behind"
        );
        fs::remove_dir_all(home).unwrap();
    }

    #[tokio::test]
    async fn a_home_with_no_identity_says_so_instead_of_showing_an_empty_ledger() {
        // An empty table would read as "you have earned nothing". The truth is
        // that there is no account to ask about yet.
        let home = scratch("snapshot-no-identity");
        let mut node = node_for(&home);
        let snapshot = node.snapshot(None).await;
        assert!(snapshot.ledger.is_none());
        assert!(
            snapshot
                .ledger_error
                .as_deref()
                .unwrap()
                .contains("no node identity"),
            "unexpected: {:?}",
            snapshot.ledger_error
        );
        fs::remove_dir_all(home).unwrap();
    }

    #[tokio::test]
    async fn an_unreachable_coordinator_still_leaves_the_machine_readable() {
        // This is the state an operator opens the window *in*. The hardware
        // and the limits must still be on screen.
        let home = scratch("snapshot-unreachable");
        NodeIdentity::load_or_create(&home).unwrap();
        let mut node = node_for(&home);
        let snapshot = node.snapshot(None).await;
        assert!(snapshot.ledger.is_none());
        assert!(snapshot.ledger_error.is_some());
        assert!(!snapshot.resources.cpu_brand.is_empty());
        assert!(snapshot.resources.logical_cpus > 0);
        assert!(!snapshot.overview.running);
        fs::remove_dir_all(home).unwrap();
    }

    #[tokio::test]
    async fn a_snapshot_of_a_stopped_node_shows_the_limits_on_disk() {
        let home = scratch("snapshot-stored-limits");
        ResourceLimits {
            cpu_percent: 33,
            memory_percent: 44,
            gpu_percent: 0,
            ai: None,
        }
        .save(&home)
        .unwrap();
        let mut node = node_for(&home);
        let snapshot = node.snapshot(None).await;
        assert_eq!(snapshot.resources.cpu_percent, 33);
        assert_eq!(snapshot.resources.memory_percent, 44);
        fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn a_home_that_never_ran_has_no_client_rather_than_a_freshly_minted_identity() {
        let home = scratch("client-no-identity");
        assert!(coordinator_client(&home, "http://127.0.0.1:1").is_none());
        assert!(
            !identity_path(&home).exists(),
            "opening a window must not create a node that never asked to exist"
        );
        fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn a_home_that_has_run_gets_a_client_for_the_coordinator_it_was_given() {
        let home = scratch("client-identity");
        let identity = NodeIdentity::load_or_create(&home).unwrap();
        let client = coordinator_client(&home, "http://mesh.example/").unwrap();
        assert_eq!(client.node_id(), identity.node_id());
        assert_eq!(client.coordinator(), "http://mesh.example");
        fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn detection_describes_this_machine_without_touching_the_network() {
        let caps = detect();
        assert!(caps.logical_cpus > 0);
        assert!(caps.total_memory_bytes > 0);
    }
}
