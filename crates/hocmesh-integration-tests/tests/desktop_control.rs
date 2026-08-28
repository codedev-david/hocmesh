//! The desktop app, driving a real daemon against a real coordinator.
//!
//! Everything the window does that a unit test cannot reach is here: starting
//! a node as a supervised child, finding it again through the file it
//! publishes, reading counters out of a process this test did not write, and
//! changing how much of the machine is lent while it runs. The unit tests in
//! `hocmesh-desktop` prove the rules; this proves the rules are wired to a
//! node that is actually doing work.
//!
//! No webview is linked. The crate's `gui` feature is off here, so these tests
//! run on a headless machine -- which is where CI lives.

use anyhow::{Context, Result, bail};
use hocmesh::control::LimitsRequest;
use hocmesh_core::limits::ResourceLimits;
use hocmesh_desktop::dashboard::{Health, Snapshot};
use hocmesh_desktop::node::Node;
use hocmesh_desktop::settings::Settings;
use hocmesh_desktop::supervisor::{Ownership, RunState, Supervisor};
use std::env;
use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// How long any "wait until" below is allowed to take.
///
/// Generous because CI machines are slow and shared, and a flaky integration
/// test teaches people to re-run rather than to read.
const PATIENCE: Duration = Duration::from_secs(90);

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_app_starts_a_node_watches_it_earn_changes_its_limits_and_stops_it() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let node_bin = bin_dir.join(exe("hocmesh"));
    let coordinator_bin = bin_dir.join(exe("hocmesh-coordinator"));

    let tmp = TestDir::new("desktop-control")?;
    let db = tmp.path.join("coordinator.db");
    // Community work, so the node has something to earn from. Without a
    // ledger configured the coordinator mints this itself, which is the
    // single-machine shape an operator installing the desktop app first sees.
    run_ok(
        Command::new(&coordinator_bin)
            .arg("seed")
            .arg("--db")
            .arg(&db)
            .arg("--start")
            .arg("2")
            .arg("--end")
            .arg("120000")
            .arg("--shards")
            .arg("4"),
        "seed community work",
    )?;

    let port = free_port()?;
    let _coordinator = ProcessGuard::spawn(
        Command::new(&coordinator_bin)
            .arg("serve")
            .arg("--db")
            .arg(&db)
            .arg("--listen")
            .arg(format!("127.0.0.1:{port}")),
    )?;
    let coordinator = format!("http://127.0.0.1:{port}");
    wait_health(port).await?;

    let home = tmp.path.join("node-home");
    let settings = Settings {
        home: home.clone(),
        coordinator: coordinator.clone(),
        start_node_with_app: false,
        workers: Some(2),
        // The point of this test is the control path, and pulling a model in
        // would make it depend on a download.
        no_ai: true,
        // A free port, which is what the app asks for so that two homes can
        // run on one machine.
        control_port: 0,
    };
    let mut node = Node::with_supervisor(settings, Supervisor::with_binary(Some(node_bin.clone())));

    // Nothing has ever run here. The window still has to be useful.
    let cold = node.snapshot(None).await;
    assert_eq!(cold.overview.health, Health::Stopped);
    assert!(!cold.overview.running);
    assert!(
        cold.resources.logical_cpus > 0,
        "the Resources tab has to describe the machine before anything runs, \
         or limits cannot be set until after joining a mesh"
    );
    assert!(
        cold.ledger.is_none() && cold.ledger_error.is_some(),
        "a home with no identity has no balance to show, and says why rather \
         than showing a zero that looks like a real one"
    );

    let started = node.start().context("starting the node from the app")?;
    let RunState::Running { ownership, pid, .. } = started else {
        bail!("the app started a node and did not get a running state back");
    };
    assert_eq!(
        ownership,
        Ownership::Supervised,
        "a node this app spawned has to be recorded as ours, or quitting the \
         app would leave it running forever"
    );
    assert!(pid > 0);

    // Registered, polling, and the coordinator answered.
    let working = wait_for(&mut node, "the node to reach the mesh", |snap| {
        snap.overview.health == Health::Working
    })
    .await?;
    assert!(working.overview.node_id.is_some());
    assert_eq!(working.overview.coordinator, coordinator);
    // Not a flat `Some(2)`: a two-core CI runner lending the default half of
    // itself is allowed one worker, and asking for two does not raise that
    // ceiling. What has to hold on every machine is that the daemon reports
    // the count the operator's own share permits, so the expectation is
    // computed from the same rule the daemon applies rather than assumed.
    let permitted =
        ResourceLimits::default().clamp_requested_workers(Some(2), cold.resources.logical_cpus);
    assert_eq!(
        working.overview.workers,
        Some(permitted),
        "the worker count the operator chose has to be the one the daemon \
         reports, clamped by the share they lent"
    );
    assert!(
        working.overview.node_version.is_some(),
        "the app shows the node's version next to its own so a mismatched pair is visible"
    );
    assert!(working.overview.last_error.is_none());

    // The seeded job is small, so this is the dashboard watching real work
    // finish rather than a contrived counter.
    let earning = wait_for(&mut node, "the node to finish a shard", |snap| {
        snap.overview.jobs_completed >= 1
    })
    .await?;
    assert_eq!(earning.overview.jobs_failed, 0);

    let paid = wait_for(&mut node, "the coordinator to pay for the work", |snap| {
        snap.ledger
            .as_ref()
            .is_some_and(|ledger| !ledger.entries.is_empty())
    })
    .await?;
    let ledger = paid.ledger.as_ref().expect("checked above");
    assert!(
        ledger.balance.parse::<f64>().unwrap_or(0.0) > 0.0,
        "work was finished, so the balance shown cannot still be zero: {}",
        ledger.balance
    );
    assert!(
        ledger.entries.iter().any(|row| row.positive),
        "the Ledger tab has to show the entry that paid for the work"
    );
    assert!(
        !ledger.authoritative,
        "this coordinator has no validators, so its table is a mirror and the \
         window must not present it as settled by a quorum"
    );

    // Changing consent while the node runs. The daemon is the one that has to
    // agree, and the file is what a restart would read back.
    let before = ResourceLimits::load_or_default(&home)?;
    assert_ne!(before.cpu_percent, 37, "pick a value the default is not");
    let update = node
        .set_limits(LimitsRequest {
            cpu_percent: Some(37),
            memory_percent: None,
            gpu_percent: None,
            ai: None,
        })
        .await
        .context("setting limits through the running daemon")?;
    assert_eq!(update.limits.cpu_percent, 37);
    assert_eq!(
        update.limits.memory_percent, before.memory_percent,
        "an absent field means unchanged, not defaulted -- anything else would \
         silently move a share the operator did not touch"
    );
    let on_disk = ResourceLimits::load_or_default(&home)?;
    assert_eq!(
        on_disk.cpu_percent, 37,
        "the consent record on disk has to carry the change, or a restart \
         would quietly undo it"
    );

    let after = node.snapshot(None).await;
    assert_eq!(after.resources.cpu_percent, 37);
    assert!(
        !after.resources.restart_required,
        "the daemon accepted this change live, so the window must not ask for \
         a restart it does not need"
    );

    // Stopping. The endpoint file is how everything on this machine finds the
    // daemon, so it going away is the observable part of a clean exit.
    assert!(
        node.stop().await?,
        "the app started this node, so it must be able to stop it"
    );
    let stopped = wait_for(&mut node, "the node to go away", |snap| {
        snap.overview.health == Health::Stopped
    })
    .await?;
    assert!(!stopped.overview.running);
    assert!(
        !home.join("control.json").exists(),
        "a stopped daemon must not leave an endpoint behind for the next app \
         to attach to"
    );
    assert!(
        !node.stop().await?,
        "stopping a node that is already stopped is not an error"
    );
    Ok(())
}

/// A daemon the operator started themselves is not the app's to end.
///
/// This is the rule that stops the desktop app from being a hazard on a
/// machine that also runs the node from a terminal or a service manager: it
/// attaches, it shows, it can be told to stop -- but quitting the window does
/// not take a node down that the window did not put up.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_node_the_app_found_is_not_taken_down_when_the_app_quits() -> Result<()> {
    let workspace = workspace_root()?;
    build_bins(&workspace)?;
    let bin_dir = workspace.join("target").join("debug");
    let node_bin = bin_dir.join(exe("hocmesh"));
    let coordinator_bin = bin_dir.join(exe("hocmesh-coordinator"));

    let tmp = TestDir::new("desktop-attach")?;
    let port = free_port()?;
    let _coordinator = ProcessGuard::spawn(
        Command::new(&coordinator_bin)
            .arg("serve")
            .arg("--db")
            .arg(tmp.path.join("coordinator.db"))
            .arg("--listen")
            .arg(format!("127.0.0.1:{port}")),
    )?;
    let coordinator = format!("http://127.0.0.1:{port}");
    wait_health(port).await?;

    // Started the way an operator would, with no app involved.
    let home = tmp.path.join("node-home");
    let mut daemon = ProcessGuard::spawn(
        Command::new(&node_bin)
            .arg("--home")
            .arg(&home)
            .arg("daemon")
            .arg("--coordinator")
            .arg(&coordinator)
            .arg("--control-port")
            .arg("0")
            .arg("--no-ai"),
    )?;

    let settings = Settings {
        home: home.clone(),
        coordinator: coordinator.clone(),
        start_node_with_app: false,
        workers: None,
        no_ai: true,
        control_port: 0,
    };
    let mut node = Node::with_supervisor(settings, Supervisor::with_binary(Some(node_bin.clone())));

    let found = wait_for(&mut node, "the app to find the running node", |snap| {
        snap.overview.running
    })
    .await?;
    assert_eq!(
        found.overview.health,
        Health::Working,
        "the app attached to a healthy node, so it has to say so"
    );
    assert!(
        !found.overview.supervised,
        "this node is not the app's child and the window must not claim it is"
    );
    assert!(
        !node.supervisor_mut().should_stop_on_quit(&home),
        "quitting the app must leave a node it did not start alone"
    );

    // An explicit Stop is different from quitting, and still works.
    assert!(node.stop().await?);
    wait_for(&mut node, "the attached node to stop", |snap| {
        snap.overview.health == Health::Stopped
    })
    .await?;
    daemon.kill();
    Ok(())
}

/// Poll the app's own snapshot until it says what we are waiting for.
///
/// Going through `Node::snapshot` rather than querying the daemon directly is
/// deliberate: what is under test is what the window would draw, so a fact the
/// daemon knows but the snapshot drops must fail here.
async fn wait_for(
    node: &mut Node,
    what: &str,
    ready: impl Fn(&Snapshot) -> bool,
) -> Result<Snapshot> {
    let deadline = std::time::Instant::now() + PATIENCE;
    let mut last = node.snapshot(None).await;
    while std::time::Instant::now() < deadline {
        if ready(&last) {
            return Ok(last);
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        last = node.snapshot(None).await;
    }
    if ready(&last) {
        return Ok(last);
    }
    bail!(
        "waited {}s for {what}; last saw health={:?} running={} jobs={} error={:?} ledger_error={:?}",
        PATIENCE.as_secs(),
        last.overview.health,
        last.overview.running,
        last.overview.jobs_completed,
        last.overview.last_error,
        last.ledger_error
    )
}

async fn wait_health(port: u16) -> Result<()> {
    let http = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/health");
    for _ in 0..200 {
        if let Ok(response) = http.get(&url).send().await
            && response.status().is_success()
        {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    bail!("the coordinator on port {port} never became healthy")
}

/// Build only what these tests launch.
///
/// Named packages rather than the whole workspace: the desktop binary needs a
/// webview toolchain to link, and nothing here opens a window.
fn build_bins(workspace: &Path) -> Result<()> {
    run_ok(
        Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
            .arg("build")
            .arg("--ignore-rust-version")
            .arg("--bins")
            .arg("-p")
            .arg("hocmesh")
            .arg("-p")
            .arg("hocmesh-coordinator")
            .current_dir(workspace),
        "build the node and coordinator binaries",
    )
}

fn run_ok(command: &mut Command, label: &str) -> Result<()> {
    let output = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("running {label}"))?;
    if !output.status.success() {
        bail!(
            "{label} failed. stdout: {} stderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

fn workspace_root() -> Result<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .context("cannot resolve the workspace root")
}

fn free_port() -> Result<u16> {
    Ok(TcpListener::bind(("127.0.0.1", 0))?.local_addr()?.port())
}

fn exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new(label: &str) -> Result<Self> {
        let suffix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = env::temp_dir().join(format!("hocmesh-{label}-{suffix}"));
        fs::create_dir_all(&path)?;
        Ok(Self { path })
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

struct ProcessGuard {
    child: Option<Child>,
}

impl ProcessGuard {
    fn spawn(command: &mut Command) -> Result<Self> {
        let child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("spawning a test process")?;
        Ok(Self { child: Some(child) })
    }

    fn kill(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        self.kill();
    }
}
