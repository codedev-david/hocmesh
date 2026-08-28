//! Starting, finding, and stopping the node this machine runs.
//!
//! The desktop app is not the node. The node is `hocmesh daemon`, a separate
//! process that keeps running when the window is closed, and the app is a way
//! to watch and steer it -- the same split Docker Desktop draws between its
//! window and its engine. That split is what lets a machine contribute from
//! boot without anyone logged in, and it is why this module deals in *finding*
//! and *attaching* as much as in spawning.
//!
//! Two rules fall out of it, and both are enforced here rather than left to
//! the UI:
//!
//! * A daemon this app did not start is not this app's to kill on quit. An
//!   operator who launched the node from a service manager or a terminal did
//!   not ask a window to take it down with it.
//! * A daemon that is already running is attached to, never duplicated. Two
//!   daemons on one home would fight over the same identity file and the same
//!   control endpoint.

use anyhow::{Context, Result, bail};
use hocmesh::control::{ControlEndpoint, request_shutdown};
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// The node executable's name on this platform.
pub const NODE_BINARY: &str = if cfg!(windows) {
    "hocmesh.exe"
} else {
    "hocmesh"
};

/// What the operator asked the node to do, before it is turned into argv.
///
/// This is the app's copy of the daemon's command line, kept as data so the
/// exact flags a launch will use can be shown, stored, and tested rather than
/// assembled inline at the moment of spawning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchOptions {
    pub home: PathBuf,
    pub coordinator: String,
    /// `None` leaves the count to the daemon, which derives it from the lent
    /// CPU share. A number here is a ceiling the operator chose.
    pub workers: Option<u32>,
    /// Decline inference work for this run regardless of what is installed.
    pub no_ai: bool,
    /// `0` takes a free port, which is what lets two homes coexist.
    pub control_port: u16,
}

impl LaunchOptions {
    /// The argv this launch would use, `hocmesh` itself excluded.
    ///
    /// Kept separate from spawning so the exact command can be asserted in a
    /// test and shown to an operator who wants to run it themselves. A flag
    /// that widens what the machine lends must never appear here by accident,
    /// so absent options emit nothing rather than a default.
    pub fn to_args(&self) -> Vec<OsString> {
        let mut args: Vec<OsString> = vec![
            "--home".into(),
            self.home.clone().into_os_string(),
            "daemon".into(),
            "--coordinator".into(),
            self.coordinator.clone().into(),
            "--control-port".into(),
            self.control_port.to_string().into(),
        ];
        if let Some(workers) = self.workers {
            args.push("--workers".into());
            args.push(workers.to_string().into());
        }
        if self.no_ai {
            args.push("--no-ai".into());
        }
        args
    }
}

/// How the node this app is showing came to be running.
///
/// The distinction is not cosmetic: it decides whether quitting the app takes
/// the node down with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ownership {
    /// This app spawned it, so this app stops it on quit.
    Supervised,
    /// It was already running -- a service, a terminal, another window. The
    /// app watches and steers it but never ends it uninvited.
    Attached,
}

/// Whether a node is running for this home, and on whose behalf.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum RunState {
    Stopped,
    Running {
        ownership: Ownership,
        pid: u32,
        port: u16,
        version: String,
    },
}

impl RunState {
    pub fn is_running(&self) -> bool {
        matches!(self, RunState::Running { .. })
    }
}

/// Where the node executable might be, in the order worth trying.
///
/// Beside the app first: an installer lays both binaries down together, and
/// that copy is the one whose version matches this window. Only then the
/// operator's `PATH`, which may hold an older build from a different install.
pub fn candidate_paths(app_dir: Option<&Path>, path_var: Option<&str>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(dir) = app_dir {
        candidates.push(dir.join(NODE_BINARY));
    }
    if let Some(path) = path_var {
        for entry in std::env::split_paths(path) {
            if entry.as_os_str().is_empty() {
                continue;
            }
            candidates.push(entry.join(NODE_BINARY));
        }
    }
    candidates
}

/// The first candidate that is actually a file.
pub fn locate_node(app_dir: Option<&Path>, path_var: Option<&str>) -> Option<PathBuf> {
    candidate_paths(app_dir, path_var)
        .into_iter()
        .find(|candidate| candidate.is_file())
}

/// Find the node next to this executable, then on the `PATH`.
pub fn discover_node() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok();
    let app_dir = exe.as_deref().and_then(Path::parent);
    let path_var = std::env::var("PATH").ok();
    locate_node(app_dir, path_var.as_deref())
}

/// The node process for one home.
///
/// Holds the child handle when this app started it and nothing when it did
/// not, which is the whole of the ownership rule.
#[derive(Debug)]
pub struct Supervisor {
    node_binary: Option<PathBuf>,
    child: Option<Child>,
}

impl Default for Supervisor {
    fn default() -> Self {
        Self::new()
    }
}

impl Supervisor {
    pub fn new() -> Self {
        Self {
            node_binary: discover_node(),
            child: None,
        }
    }

    /// A supervisor pointed at a known executable, for tests and for an
    /// operator who put the node somewhere unusual.
    pub fn with_binary(node_binary: Option<PathBuf>) -> Self {
        Self {
            node_binary,
            child: None,
        }
    }

    pub fn node_binary(&self) -> Option<&Path> {
        self.node_binary.as_deref()
    }

    /// Whether this app is holding a child handle for a live process.
    ///
    /// Reaps a child that has already exited, so a daemon that crashed stops
    /// being reported as supervised. Without this, a crashed node would look
    /// like one the app still owns and quitting would try to stop a process
    /// that is already gone.
    pub fn supervised_child_alive(&mut self) -> bool {
        let Some(child) = self.child.as_mut() else {
            return false;
        };
        match child.try_wait() {
            Ok(Some(_)) => {
                self.child = None;
                false
            }
            Ok(None) => true,
            // The handle is unusable, so it cannot be relied on to stop
            // anything; treating it as gone is the honest reading.
            Err(_) => {
                self.child = None;
                false
            }
        }
    }

    /// What is running for this home, read from the endpoint file the daemon
    /// writes.
    ///
    /// The file is the shared fact between the two processes: the app does not
    /// have to have started the node to see it, which is what makes attaching
    /// work at all.
    pub fn state(&mut self, home: &Path) -> RunState {
        let supervised = self.supervised_child_alive();
        match ControlEndpoint::load(home) {
            Some(endpoint) => RunState::Running {
                ownership: if supervised {
                    Ownership::Supervised
                } else {
                    Ownership::Attached
                },
                pid: endpoint.pid,
                port: endpoint.port,
                version: endpoint.version,
            },
            None => RunState::Stopped,
        }
    }

    /// Start a node for this home, or do nothing if one is already up.
    ///
    /// Returning the existing state rather than an error is deliberate: the
    /// tray's Start item and an autostart at login can race, and the right
    /// outcome of that race is one daemon, not a dialog.
    pub fn start(&mut self, options: &LaunchOptions) -> Result<RunState> {
        let existing = self.state(&options.home);
        if existing.is_running() {
            return Ok(existing);
        }
        let binary = self
            .node_binary
            .clone()
            .context("the hocmesh node executable was not found beside this app or on PATH")?;
        std::fs::create_dir_all(&options.home)
            .with_context(|| format!("could not create {}", options.home.display()))?;
        let mut command = Command::new(&binary);
        command
            .args(options.to_args())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        detach(&mut command);
        let child = command
            .spawn()
            .with_context(|| format!("could not start {}", binary.display()))?;
        self.child = Some(child);
        Ok(RunState::Running {
            ownership: Ownership::Supervised,
            pid: self.child.as_ref().map(Child::id).unwrap_or_default(),
            port: 0,
            version: env!("CARGO_PKG_VERSION").into(),
        })
    }

    /// Ask the node for this home to stop.
    ///
    /// Politely, through the control surface, so the daemon finishes what it
    /// is holding and clears its endpoint file rather than leaving a stale one
    /// behind. A node this app did not start is stopped the same way -- the
    /// operator asked for it here and now, which is different from taking it
    /// down as a side effect of closing a window.
    pub async fn stop(&mut self, home: &Path) -> Result<bool> {
        let asked = request_shutdown(home).await?;
        self.child = None;
        Ok(asked)
    }

    /// What to do with the node when the app quits.
    ///
    /// Only a node this app started is stopped. This is the rule the whole
    /// ownership distinction exists for, so it is a named method with a test
    /// rather than an `if` in the quit handler.
    pub fn should_stop_on_quit(&mut self, home: &Path) -> bool {
        matches!(
            self.state(home),
            RunState::Running {
                ownership: Ownership::Supervised,
                ..
            }
        )
    }
}

/// Keep the node out of the app's console and lifetime where the platform
/// needs telling.
#[cfg(windows)]
fn detach(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    // CREATE_NO_WINDOW: the node is a console program, and spawning it from a
    // GUI app would otherwise flash a console window on every start.
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn detach(_command: &mut Command) {}

/// A one-line reason a start could not happen, or nothing if it could.
///
/// Checked before spawning so the tray can grey out Start and say why, rather
/// than letting the operator press a button that will fail.
pub fn start_blocker(node_binary: Option<&Path>, coordinator: &str) -> Option<String> {
    if node_binary.is_none() {
        return Some(format!(
            "{NODE_BINARY} was not found beside this app or on PATH"
        ));
    }
    if coordinator.trim().is_empty() {
        return Some("no coordinator is configured".into());
    }
    if !coordinator.starts_with("http://") && !coordinator.starts_with("https://") {
        return Some(format!("{coordinator} is not an http(s) address"));
    }
    None
}

/// Reject a home that is not usable before anything tries to write to it.
pub fn check_home(home: &Path) -> Result<()> {
    if home.as_os_str().is_empty() {
        bail!("no home directory is configured");
    }
    if home.exists() && !home.is_dir() {
        bail!("{} exists and is not a directory", home.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn options(home: &Path) -> LaunchOptions {
        LaunchOptions {
            home: home.to_path_buf(),
            coordinator: "http://127.0.0.1:8080".into(),
            workers: None,
            no_ai: false,
            control_port: 0,
        }
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("hocmesh-desktop-{tag}-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_launch_names_the_home_before_the_subcommand() {
        // `--home` is a global flag on the node's CLI: putting it after
        // `daemon` makes clap reject the whole command line, so the order here
        // is load-bearing rather than stylistic.
        let args = options(Path::new("/tmp/home")).to_args();
        let home_at = args.iter().position(|a| a == "--home").unwrap();
        let daemon_at = args.iter().position(|a| a == "daemon").unwrap();
        assert!(home_at < daemon_at);
    }

    #[test]
    fn an_unset_option_contributes_no_flag_at_all() {
        // A default worker count emitted here would override the ceiling the
        // operator set in limits.json, which is exactly the kind of quiet
        // widening the consent record exists to prevent.
        let args = options(Path::new("/tmp/home")).to_args();
        assert!(!args.iter().any(|a| a == "--workers"));
        assert!(!args.iter().any(|a| a == "--no-ai"));
    }

    #[test]
    fn a_chosen_worker_count_and_a_declined_ai_share_both_reach_the_command() {
        let mut opts = options(Path::new("/tmp/home"));
        opts.workers = Some(3);
        opts.no_ai = true;
        let args = opts.to_args();
        let at = args.iter().position(|a| a == "--workers").unwrap();
        assert_eq!(args[at + 1], OsString::from("3"));
        assert!(args.iter().any(|a| a == "--no-ai"));
    }

    #[test]
    fn a_home_containing_a_space_survives_as_one_argument() {
        // Windows homes live under "C:\Users\...\AppData\Local", but an
        // operator's may not be so tidy. Building argv as a vector rather than
        // a string is what keeps this whole.
        let home = Path::new("/tmp/two words/home");
        let args = options(home).to_args();
        let at = args.iter().position(|a| a == "--home").unwrap();
        assert_eq!(args[at + 1], home.as_os_str());
    }

    #[test]
    fn the_control_port_is_always_stated_so_a_default_never_collides() {
        let args = options(Path::new("/tmp/home")).to_args();
        let at = args.iter().position(|a| a == "--control-port").unwrap();
        assert_eq!(args[at + 1], OsString::from("0"));
    }

    #[test]
    fn the_copy_beside_the_app_is_preferred_over_one_on_the_path() {
        let root = scratch("locate");
        let beside = root.join("beside");
        let on_path = root.join("on-path");
        fs::create_dir_all(&beside).unwrap();
        fs::create_dir_all(&on_path).unwrap();
        fs::write(beside.join(NODE_BINARY), b"x").unwrap();
        fs::write(on_path.join(NODE_BINARY), b"x").unwrap();
        let path_var = on_path.to_str().unwrap();

        let found = locate_node(Some(&beside), Some(path_var)).unwrap();
        assert_eq!(
            found,
            beside.join(NODE_BINARY),
            "the installed pair must be kept together, or the window and the node can disagree on version"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_path_is_the_fallback_when_nothing_sits_beside_the_app() {
        let root = scratch("locate-path");
        let empty = root.join("empty");
        let on_path = root.join("on-path");
        fs::create_dir_all(&empty).unwrap();
        fs::create_dir_all(&on_path).unwrap();
        fs::write(on_path.join(NODE_BINARY), b"x").unwrap();

        let found = locate_node(Some(&empty), Some(on_path.to_str().unwrap())).unwrap();
        assert_eq!(found, on_path.join(NODE_BINARY));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_missing_node_is_none_rather_than_a_path_that_does_not_exist() {
        let root = scratch("locate-none");
        assert!(locate_node(Some(&root), Some("")).is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn an_empty_path_entry_is_skipped_rather_than_resolving_to_the_cwd() {
        // A trailing separator leaves an empty entry, and joining onto it
        // would look for the node in whatever directory the app happens to be
        // running from.
        let candidates = candidate_paths(None, Some(""));
        assert!(candidates.is_empty());
    }

    #[test]
    fn no_daemon_for_a_home_reads_as_stopped() {
        let root = scratch("state-stopped");
        let mut supervisor = Supervisor::with_binary(None);
        assert_eq!(supervisor.state(&root), RunState::Stopped);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_daemon_this_app_did_not_start_is_reported_as_attached() {
        let root = scratch("state-attached");
        ControlEndpoint {
            port: 7788,
            token: "t".into(),
            pid: 4242,
            version: "9.9.9".into(),
            started_unix: 1,
        }
        .write(&root)
        .unwrap();

        let mut supervisor = Supervisor::with_binary(None);
        match supervisor.state(&root) {
            RunState::Running {
                ownership,
                pid,
                port,
                version,
            } => {
                assert_eq!(ownership, Ownership::Attached);
                assert_eq!(pid, 4242);
                assert_eq!(port, 7788);
                assert_eq!(version, "9.9.9");
            }
            other => panic!("expected an attached daemon, got {other:?}"),
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn quitting_leaves_a_node_this_app_did_not_start_running() {
        let root = scratch("quit-attached");
        ControlEndpoint {
            port: 7788,
            token: "t".into(),
            pid: 4242,
            version: "9.9.9".into(),
            started_unix: 1,
        }
        .write(&root)
        .unwrap();

        let mut supervisor = Supervisor::with_binary(None);
        assert!(
            !supervisor.should_stop_on_quit(&root),
            "closing a window must not take down a node the operator started elsewhere"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn quitting_with_no_node_running_stops_nothing() {
        let root = scratch("quit-stopped");
        let mut supervisor = Supervisor::with_binary(None);
        assert!(!supervisor.should_stop_on_quit(&root));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn starting_when_a_daemon_is_already_up_attaches_instead_of_duplicating() {
        let root = scratch("start-existing");
        ControlEndpoint {
            port: 7788,
            token: "t".into(),
            pid: 4242,
            version: "9.9.9".into(),
            started_unix: 1,
        }
        .write(&root)
        .unwrap();

        // No executable is configured, so a real spawn would fail outright.
        // Reaching a Running state proves the existing daemon short-circuited
        // the launch rather than a second one being started.
        let mut supervisor = Supervisor::with_binary(None);
        let state = supervisor.start(&options(&root)).unwrap();
        assert!(matches!(
            state,
            RunState::Running {
                ownership: Ownership::Attached,
                ..
            }
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn starting_without_an_executable_says_so_rather_than_failing_silently() {
        let root = scratch("start-missing");
        let mut supervisor = Supervisor::with_binary(None);
        let error = supervisor.start(&options(&root)).unwrap_err().to_string();
        assert!(error.contains("was not found"), "unhelpful error: {error}");
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn stopping_a_home_with_no_daemon_is_not_an_error() {
        let root = scratch("stop-none");
        let mut supervisor = Supervisor::with_binary(None);
        assert!(!supervisor.stop(&root).await.unwrap());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_missing_executable_blocks_start_with_a_reason_the_operator_can_act_on() {
        let reason = start_blocker(None, "http://127.0.0.1:8080").unwrap();
        assert!(reason.contains(NODE_BINARY));
    }

    #[test]
    fn an_unconfigured_or_malformed_coordinator_blocks_start() {
        let binary = PathBuf::from("/somewhere/hocmesh");
        assert!(start_blocker(Some(&binary), "   ").is_some());
        assert!(start_blocker(Some(&binary), "127.0.0.1:8080").is_some());
        assert!(start_blocker(Some(&binary), "https://mesh.example").is_none());
    }

    #[test]
    fn a_home_that_is_a_file_is_refused_before_anything_writes_to_it() {
        let root = scratch("home-file");
        let file = root.join("not-a-dir");
        fs::write(&file, b"x").unwrap();
        assert!(check_home(&file).is_err());
        assert!(check_home(&root).is_ok());
        assert!(check_home(Path::new("")).is_err());
        // A home that does not exist yet is fine -- starting creates it.
        assert!(check_home(&root.join("fresh")).is_ok());
        fs::remove_dir_all(root).unwrap();
    }
}
