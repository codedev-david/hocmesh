//! Everything the window renders, assembled once per refresh.
//!
//! The page is given one object and draws it. There is no second source of
//! truth in the JavaScript and no formatting decided there, which is what
//! keeps the numbers on screen testable: the whole of what an operator sees
//! is produced by [`Snapshot::build`], a pure function over what was read.
//!
//! Reading is deliberately partial. A node that is stopped still has limits on
//! disk and hardware to describe, so the Resources tab works with nothing
//! running; a coordinator that is unreachable still leaves the daemon's own
//! counters readable. Each section therefore carries its own error rather than
//! one failure blanking the window -- an operator diagnosing a machine needs
//! the parts that *do* answer.

use crate::format;
use crate::supervisor::{Ownership, RunState};
use hocmesh::control::DaemonState;
use hocmesh_core::hardware;
use hocmesh_core::limits::ResourceLimits;
use hocmesh_protocol::{BalanceResponse, LedgerEntry, NodeCapabilities};
use serde::{Deserialize, Serialize};

/// The colour of the tray icon, and the one-word answer to "is it working?".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Health {
    /// Running and in touch with its coordinator.
    Working,
    /// Running, but not reaching the mesh. Work is not flowing.
    Degraded,
    /// Not running.
    Stopped,
}

impl Health {
    pub fn label(self) -> &'static str {
        match self {
            Health::Working => "Contributing",
            Health::Degraded => "Not reaching the mesh",
            Health::Stopped => "Stopped",
        }
    }
}

/// The Overview tab.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Overview {
    pub health: Health,
    pub health_label: String,
    pub running: bool,
    pub supervised: bool,
    pub node_id: Option<String>,
    pub coordinator: String,
    pub node_version: Option<String>,
    pub app_version: String,
    pub uptime: Option<String>,
    pub workers: Option<usize>,
    pub jobs_completed: u64,
    pub jobs_failed: u64,
    pub inferences_completed: u64,
    /// Earned during this run, which is not the same as the balance: the
    /// balance is the coordinator's authoritative figure and this is what this
    /// process has seen settle since it started.
    pub earned_this_run: String,
    pub last_contact: String,
    pub last_error: Option<String>,
    pub ai_offered: bool,
}

/// The Resources tab: what this machine is lending, against what it has.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resources {
    pub cpu_percent: u8,
    pub memory_percent: u8,
    pub gpu_percent: u8,
    /// `None` is "auto": lend the accelerator to inference when one is lent at
    /// all and a runtime is installed.
    pub ai: Option<bool>,
    pub ai_effective: bool,
    pub logical_cpus: usize,
    pub shared_logical_cpus: usize,
    pub total_memory: String,
    pub shared_memory: String,
    pub shared_memory_percent_of_machine: u32,
    pub cpu_brand: String,
    pub hostname: String,
    pub os: String,
    pub arch: String,
    pub accelerators: Vec<Accelerator>,
    /// Set when the running daemon is working to older limits than the ones on
    /// disk, which happens after a change that needs a different worker pool.
    pub restart_required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Accelerator {
    pub stable_id: String,
    pub name: String,
    pub vendor: String,
    pub backend: String,
    pub memory: Option<String>,
    /// True when this entry is the CPU standing in for an accelerator, which
    /// is what a CPU-only node lends to inference.
    pub is_cpu: bool,
}

/// The Ledger tab.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ledger {
    pub balance: String,
    pub earned: String,
    pub spent: String,
    pub ledger_height: Option<u64>,
    /// Whether the entries below came from the validator quorum. A
    /// coordinator's own table is a mirror, and a dashboard that presented it
    /// as settled would be overstating what it knows.
    pub authoritative: bool,
    pub entries: Vec<LedgerRow>,
    pub next_before: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerRow {
    pub delta: String,
    pub positive: bool,
    pub reason: String,
    pub reference: Option<String>,
    pub when: String,
    pub sequence: Option<u64>,
}

/// One refresh of the whole window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    pub overview: Overview,
    pub resources: Resources,
    pub ledger: Option<Ledger>,
    /// Why the ledger is missing, when it is. Shown in place of the table
    /// rather than swallowed.
    pub ledger_error: Option<String>,
    /// A tooltip short enough for a tray icon.
    pub tray_tooltip: String,
}

/// What was read, before any of it is turned into what is shown.
///
/// Passing this in rather than doing the I/O inside `build` is what makes
/// every rule below testable without a running daemon, a coordinator, or a
/// network.
pub struct Readings<'a> {
    pub run_state: &'a RunState,
    pub coordinator: &'a str,
    pub daemon: Option<&'a DaemonState>,
    /// The limits the daemon is enforcing, when one is running.
    pub live_limits: Option<&'a ResourceLimits>,
    /// The limits on disk, which is what a stopped node would start with.
    pub stored_limits: &'a ResourceLimits,
    /// What this node *advertises*, not what the machine has: limits already
    /// applied and AI readiness already settled, by
    /// [`hocmesh::control::advertised_capabilities`]. Whether inference is
    /// actually offered is read from here rather than recomputed, so the
    /// window cannot claim a readiness the coordinator was never told about.
    pub capabilities: &'a NodeCapabilities,
    pub balance: Option<&'a BalanceResponse>,
    pub history: Option<(&'a [LedgerEntry], bool, Option<u64>)>,
    pub ledger_error: Option<String>,
    pub now: i64,
}

impl Snapshot {
    pub fn build(readings: Readings<'_>) -> Self {
        let running = readings.run_state.is_running();
        let supervised = matches!(
            readings.run_state,
            RunState::Running {
                ownership: Ownership::Supervised,
                ..
            }
        );
        let health = match (running, readings.daemon) {
            (false, _) => Health::Stopped,
            // Running but the control surface did not answer: something is
            // wrong with the node itself, which is a degraded node rather than
            // a healthy one.
            (true, None) => Health::Degraded,
            (true, Some(state)) if state.connected => Health::Working,
            (true, Some(_)) => Health::Degraded,
        };

        // The daemon's live view wins where it exists: a change made through
        // the CLI while this window was open is real, and showing the file the
        // app last wrote would be showing a stale consent record.
        let limits = readings.live_limits.unwrap_or(readings.stored_limits);
        let caps = readings.capabilities;

        let overview = Overview {
            health,
            health_label: health.label().into(),
            running,
            supervised,
            node_id: readings.daemon.map(|s| s.node_id.clone()),
            coordinator: readings
                .daemon
                .map(|s| s.coordinator.clone())
                .unwrap_or_else(|| readings.coordinator.to_string()),
            node_version: readings.daemon.map(|s| s.version.clone()),
            app_version: env!("CARGO_PKG_VERSION").into(),
            uptime: readings.daemon.map(|s| format::duration(s.uptime_secs)),
            workers: readings.daemon.map(|s| s.workers),
            jobs_completed: readings.daemon.map(|s| s.jobs_completed).unwrap_or(0),
            jobs_failed: readings.daemon.map(|s| s.jobs_failed).unwrap_or(0),
            inferences_completed: readings.daemon.map(|s| s.inferences_completed).unwrap_or(0),
            earned_this_run: format::cu(readings.daemon.map(|s| s.mcu_earned).unwrap_or(0)),
            last_contact: readings
                .daemon
                .map(|s| format::since(s.last_contact_unix, readings.now))
                .unwrap_or_else(|| "never".into()),
            last_error: readings.daemon.and_then(|s| s.last_error.clone()),
            ai_offered: readings.daemon.map(|s| s.ai_offered).unwrap_or(false),
        };

        let accelerators = caps
            .gpus
            .iter()
            .map(|gpu| Accelerator {
                stable_id: gpu.stable_id.clone(),
                name: gpu.name.clone(),
                vendor: gpu.vendor.clone(),
                backend: gpu.backend.clone(),
                memory: gpu
                    .memory_mb
                    .map(|mb| format::bytes(mb.saturating_mul(1024 * 1024))),
                is_cpu: gpu.stable_id == hardware::SHARED_CPU_DEVICE_ID,
            })
            .collect();

        let resources = Resources {
            cpu_percent: limits.cpu_percent,
            memory_percent: limits.memory_percent,
            gpu_percent: limits.gpu_percent,
            ai: limits.ai,
            // Consent is `limits.ai`; this is whether it reached the mesh.
            // The two differ when a runtime is missing or no accelerator is
            // lent, and saying so is the whole point of showing both.
            ai_effective: caps.ai_runtime_ready,
            logical_cpus: caps.logical_cpus,
            shared_logical_cpus: caps.shared_logical_cpus,
            total_memory: format::bytes(caps.total_memory_bytes),
            shared_memory: format::bytes(caps.shared_memory_bytes),
            shared_memory_percent_of_machine: format::percent_of(
                caps.shared_memory_bytes,
                caps.total_memory_bytes,
            ),
            cpu_brand: caps.cpu_brand.clone(),
            hostname: caps.hostname.clone(),
            os: caps.os.clone(),
            arch: caps.arch.clone(),
            accelerators,
            // Only a running daemon can be behind: a stopped node will read
            // the file when it starts, so there is nothing to warn about.
            restart_required: running
                && readings
                    .live_limits
                    .is_some_and(|live| live != readings.stored_limits),
        };

        let ledger = readings.balance.map(|balance| {
            // A balance with no history page is an ordinary state -- the
            // balance endpoint answered and the history one did not -- and it
            // shows as a total over an empty table rather than as no ledger.
            let (entries, authoritative, next_before) =
                readings.history.unwrap_or((&[], false, None));
            Ledger {
                balance: format::cu(balance.balance_mcu),
                earned: format::cu(balance.earned_mcu),
                spent: format::cu(balance.spent_mcu),
                ledger_height: balance.ledger_height,
                authoritative,
                entries: entries
                    .iter()
                    .map(|entry| ledger_row(entry, readings.now))
                    .collect(),
                next_before,
            }
        });

        let tray_tooltip = tray_tooltip(&overview, ledger.as_ref());

        Self {
            overview,
            resources,
            ledger,
            ledger_error: readings.ledger_error,
            tray_tooltip,
        }
    }
}

fn ledger_row(entry: &LedgerEntry, now: i64) -> LedgerRow {
    LedgerRow {
        delta: format::signed_cu(entry.delta_mcu),
        positive: entry.delta_mcu >= 0,
        reason: entry
            .category
            .clone()
            // A chain entry carries no category, and calling it "unknown"
            // would read as a fault. It is a settled posting; what it lacks is
            // the coordinator's local label.
            .unwrap_or_else(|| "settled".into()),
        reference: entry
            .job_id
            .clone()
            .or_else(|| entry.assignment_id.clone())
            .or_else(|| entry.transaction_id.clone()),
        when: format::since(entry.created_at, now),
        sequence: entry.sequence,
    }
}

/// A tray tooltip: state first, then the number an operator hovers to see.
///
/// Kept short because several platforms truncate a long one, and the truncated
/// half would be the half that matters.
fn tray_tooltip(overview: &Overview, ledger: Option<&Ledger>) -> String {
    let mut text = format!("hocMESH — {}", overview.health_label);
    if overview.running {
        text.push_str(&format!(
            "\n{} jobs · {} CU this run",
            overview.jobs_completed, overview.earned_this_run
        ));
    }
    if let Some(ledger) = ledger {
        text.push_str(&format!("\nBalance {} CU", ledger.balance));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use hocmesh_protocol::GpuCapability;

    fn caps() -> NodeCapabilities {
        NodeCapabilities {
            protocol_version: hocmesh_protocol::PROTOCOL_VERSION,
            hostname: "desk".into(),
            os: "windows".into(),
            arch: "x86_64".into(),
            cpu_brand: "Test CPU".into(),
            logical_cpus: 16,
            total_memory_bytes: 32 * 1024 * 1024 * 1024,
            cpu_benchmark_score: 1,
            memory_bandwidth_bytes_per_second: None,
            gpus: Vec::new(),
            model_seed_url: None,
            cached_model_manifests: Vec::new(),
            coordinator_latency_micros: 0,
            model_bandwidth_kbps: 0,
            load_permille: 0,
            ai_runtime_ready: false,
            shared_logical_cpus: 8,
            shared_memory_bytes: 16 * 1024 * 1024 * 1024,
            shared_gpu_percent: 0,
            network_coordinate: None,
            probe_endpoint: None,
        }
    }

    fn daemon() -> DaemonState {
        DaemonState {
            node_id: "node-1".into(),
            coordinator: "http://mesh.example".into(),
            version: "0.3.0".into(),
            started_unix: 1_000_000,
            uptime_secs: 3_700,
            workers: 8,
            connected: true,
            jobs_completed: 12,
            jobs_failed: 1,
            inferences_completed: 3,
            mcu_earned: 4_500,
            last_contact_unix: 1_003_640,
            last_error: None,
            ai_offered: false,
        }
    }

    fn running() -> RunState {
        RunState::Running {
            ownership: Ownership::Supervised,
            pid: 1,
            port: 1,
            version: "0.3.0".into(),
        }
    }

    fn readings<'a>(
        run_state: &'a RunState,
        daemon: Option<&'a DaemonState>,
        limits: &'a ResourceLimits,
        caps: &'a NodeCapabilities,
    ) -> Readings<'a> {
        Readings {
            run_state,
            coordinator: "http://configured.example",
            daemon,
            live_limits: None,
            stored_limits: limits,
            capabilities: caps,
            balance: None,
            history: None,
            ledger_error: None,
            now: 1_003_700,
        }
    }

    #[test]
    fn a_stopped_node_still_describes_what_the_machine_would_lend() {
        // The Resources tab has to work before anything is running, or an
        // operator cannot set limits until they have already joined a mesh.
        let state = RunState::Stopped;
        let limits = ResourceLimits::default();
        let caps = caps();
        let snapshot = Snapshot::build(readings(&state, None, &limits, &caps));
        assert_eq!(snapshot.overview.health, Health::Stopped);
        assert!(!snapshot.overview.running);
        assert_eq!(snapshot.resources.logical_cpus, 16);
        assert_eq!(snapshot.resources.shared_memory, "16.0 GiB");
        assert_eq!(snapshot.resources.total_memory, "32.0 GiB");
        assert_eq!(snapshot.resources.shared_memory_percent_of_machine, 50);
    }

    #[test]
    fn a_stopped_node_shows_the_configured_coordinator_not_a_blank() {
        let state = RunState::Stopped;
        let limits = ResourceLimits::default();
        let caps = caps();
        let snapshot = Snapshot::build(readings(&state, None, &limits, &caps));
        assert_eq!(snapshot.overview.coordinator, "http://configured.example");
        assert_eq!(snapshot.overview.node_id, None);
    }

    #[test]
    fn a_running_node_in_touch_with_the_mesh_is_working() {
        let state = running();
        let limits = ResourceLimits::default();
        let caps = caps();
        let daemon = daemon();
        let snapshot = Snapshot::build(readings(&state, Some(&daemon), &limits, &caps));
        assert_eq!(snapshot.overview.health, Health::Working);
        assert_eq!(snapshot.overview.health_label, "Contributing");
        assert_eq!(snapshot.overview.uptime.as_deref(), Some("1h 1m"));
        assert_eq!(snapshot.overview.earned_this_run, "4.500");
        assert_eq!(snapshot.overview.last_contact, "1m 0s ago");
        assert!(snapshot.overview.supervised);
    }

    #[test]
    fn a_running_node_out_of_touch_is_degraded_rather_than_healthy() {
        // "Running" is not the question an operator is asking. A node that is
        // up but not reaching its coordinator is earning nothing, and a green
        // icon over that would be the dashboard's worst failure.
        let state = running();
        let limits = ResourceLimits::default();
        let caps = caps();
        let mut daemon = daemon();
        daemon.connected = false;
        daemon.last_error = Some("connection refused".into());
        let snapshot = Snapshot::build(readings(&state, Some(&daemon), &limits, &caps));
        assert_eq!(snapshot.overview.health, Health::Degraded);
        assert_eq!(
            snapshot.overview.last_error.as_deref(),
            Some("connection refused")
        );
    }

    #[test]
    fn a_process_that_is_up_but_will_not_answer_its_own_control_surface_is_degraded() {
        let state = running();
        let limits = ResourceLimits::default();
        let caps = caps();
        let snapshot = Snapshot::build(readings(&state, None, &limits, &caps));
        assert_eq!(snapshot.overview.health, Health::Degraded);
    }

    #[test]
    fn the_daemons_own_limits_win_over_the_file_the_app_last_saw() {
        // A change made through the CLI while this window was open is real.
        // Showing the stored file over the live one would show a consent
        // record the node is not enforcing.
        let state = running();
        let stored = ResourceLimits {
            cpu_percent: 50,
            ..ResourceLimits::default()
        };
        let live = ResourceLimits {
            cpu_percent: 20,
            ..ResourceLimits::default()
        };
        let caps = caps();
        let daemon = daemon();
        let mut reads = readings(&state, Some(&daemon), &stored, &caps);
        reads.live_limits = Some(&live);
        let snapshot = Snapshot::build(reads);
        assert_eq!(snapshot.resources.cpu_percent, 20);
    }

    #[test]
    fn limits_that_differ_from_the_running_ones_raise_the_restart_notice() {
        let state = running();
        let stored = ResourceLimits {
            cpu_percent: 90,
            ..ResourceLimits::default()
        };
        let live = ResourceLimits::default();
        let caps = caps();
        let daemon = daemon();
        let mut reads = readings(&state, Some(&daemon), &stored, &caps);
        reads.live_limits = Some(&live);
        assert!(Snapshot::build(reads).resources.restart_required);
    }

    #[test]
    fn a_stopped_node_never_asks_to_be_restarted_for_limits_it_has_not_read() {
        let state = RunState::Stopped;
        let stored = ResourceLimits {
            cpu_percent: 90,
            ..ResourceLimits::default()
        };
        let caps = caps();
        let snapshot = Snapshot::build(readings(&state, None, &stored, &caps));
        assert!(!snapshot.resources.restart_required);
    }

    #[test]
    fn the_cpu_standing_in_for_an_accelerator_is_labelled_as_one() {
        // A CPU-only node that opted into inference advertises a device whose
        // id is the shared-CPU one. Showing it as a GPU would misrepresent the
        // machine; hiding it would leave the operator unable to see what they
        // lent.
        let state = RunState::Stopped;
        let limits = ResourceLimits::default();
        let mut caps = caps();
        caps.gpus = vec![GpuCapability {
            stable_id: hardware::SHARED_CPU_DEVICE_ID.into(),
            vendor: "x86_64".into(),
            name: "Test CPU".into(),
            backend: "cpu".into(),
            memory_mb: Some(8 * 1024),
            driver_version: None,
            compute_version: None,
            supports_fp16: true,
            supports_bf16: true,
            supports_int8: true,
            benchmark_bytes_per_second: None,
            benchmark_p95_micros: None,
        }];
        let snapshot = Snapshot::build(readings(&state, None, &limits, &caps));
        let accelerator = &snapshot.resources.accelerators[0];
        assert!(accelerator.is_cpu);
        assert_eq!(accelerator.memory.as_deref(), Some("8.0 GiB"));
    }

    #[test]
    fn a_real_gpu_is_not_mistaken_for_the_cpu_stand_in() {
        let state = RunState::Stopped;
        let limits = ResourceLimits::default();
        let mut caps = caps();
        caps.gpus = vec![GpuCapability {
            stable_id: "cuda-0".into(),
            vendor: "nvidia".into(),
            name: "Test GPU".into(),
            backend: "cuda".into(),
            memory_mb: Some(24 * 1024),
            driver_version: None,
            compute_version: Some("8.9".into()),
            supports_fp16: true,
            supports_bf16: true,
            supports_int8: true,
            benchmark_bytes_per_second: None,
            benchmark_p95_micros: None,
        }];
        let snapshot = Snapshot::build(readings(&state, None, &limits, &caps));
        assert!(!snapshot.resources.accelerators[0].is_cpu);
    }

    #[test]
    fn no_balance_means_no_ledger_section_rather_than_a_row_of_zeroes() {
        // Zeroes would read as "you have earned nothing", which is a claim.
        // Absence reads as "not known", which is the truth when the
        // coordinator did not answer.
        let state = running();
        let limits = ResourceLimits::default();
        let caps = caps();
        let daemon = daemon();
        let mut reads = readings(&state, Some(&daemon), &limits, &caps);
        reads.ledger_error = Some("coordinator unreachable".into());
        let snapshot = Snapshot::build(reads);
        assert!(snapshot.ledger.is_none());
        assert_eq!(
            snapshot.ledger_error.as_deref(),
            Some("coordinator unreachable")
        );
    }

    #[test]
    fn ledger_rows_carry_direction_reason_and_a_reference_to_chase() {
        let state = running();
        let limits = ResourceLimits::default();
        let caps = caps();
        let daemon = daemon();
        let balance = BalanceResponse {
            node_id: "node-1".into(),
            balance_mcu: 12_500,
            earned_mcu: 20_000,
            spent_mcu: 7_500,
            ledger_height: Some(42),
            ledger_head: Some("abc".into()),
        };
        let entries = vec![
            LedgerEntry {
                delta_mcu: 2_500,
                category: Some("reward".into()),
                job_id: Some("job-7".into()),
                assignment_id: None,
                sequence: None,
                transaction_id: None,
                created_at: 1_003_640,
            },
            LedgerEntry {
                delta_mcu: -1_000,
                category: None,
                job_id: None,
                assignment_id: None,
                sequence: Some(41),
                transaction_id: Some("tx-9".into()),
                created_at: 1_003_600,
            },
        ];
        let mut reads = readings(&state, Some(&daemon), &limits, &caps);
        reads.balance = Some(&balance);
        reads.history = Some((&entries, true, Some(40)));
        let snapshot = Snapshot::build(reads);
        let ledger = snapshot.ledger.unwrap();
        assert_eq!(ledger.balance, "12.500");
        assert_eq!(ledger.earned, "20.000");
        assert_eq!(ledger.spent, "7.500");
        assert!(ledger.authoritative);
        assert_eq!(ledger.next_before, Some(40));
        assert_eq!(ledger.entries[0].delta, "+2.500");
        assert!(ledger.entries[0].positive);
        assert_eq!(ledger.entries[0].reason, "reward");
        assert_eq!(ledger.entries[0].reference.as_deref(), Some("job-7"));
        assert_eq!(ledger.entries[1].delta, "-1.000");
        assert!(!ledger.entries[1].positive);
        assert_eq!(
            ledger.entries[1].reason, "settled",
            "a chain posting has no local category, and 'unknown' would read as a fault"
        );
        assert_eq!(ledger.entries[1].reference.as_deref(), Some("tx-9"));
        assert_eq!(ledger.entries[1].sequence, Some(41));
    }

    #[test]
    fn a_balance_with_no_history_yet_still_shows_the_balance() {
        let state = running();
        let limits = ResourceLimits::default();
        let caps = caps();
        let daemon = daemon();
        let balance = BalanceResponse {
            node_id: "node-1".into(),
            balance_mcu: 0,
            earned_mcu: 0,
            spent_mcu: 0,
            ledger_height: None,
            ledger_head: None,
        };
        let mut reads = readings(&state, Some(&daemon), &limits, &caps);
        reads.balance = Some(&balance);
        let ledger = Snapshot::build(reads).ledger.unwrap();
        assert!(ledger.entries.is_empty());
        assert_eq!(ledger.balance, "0.000");
    }

    #[test]
    fn the_tray_tooltip_leads_with_the_state_because_platforms_truncate_it() {
        let state = running();
        let limits = ResourceLimits::default();
        let caps = caps();
        let daemon = daemon();
        let snapshot = Snapshot::build(readings(&state, Some(&daemon), &limits, &caps));
        assert!(snapshot.tray_tooltip.starts_with("hocMESH — Contributing"));
        assert!(snapshot.tray_tooltip.contains("12 jobs"));
    }

    #[test]
    fn a_stopped_tray_tooltip_does_not_claim_a_run_that_is_not_happening() {
        let state = RunState::Stopped;
        let limits = ResourceLimits::default();
        let caps = caps();
        let snapshot = Snapshot::build(readings(&state, None, &limits, &caps));
        assert_eq!(snapshot.tray_tooltip, "hocMESH — Stopped");
    }

    #[test]
    fn ai_reads_as_effective_only_when_a_runtime_is_actually_installed() {
        // `ai: Some(true)` is consent, not capability. Reporting it as
        // effective with no runtime present would tell the operator work is
        // being served that cannot be.
        //
        // Both halves go through `advertised_capabilities`, which is the same
        // function the daemon advertises with, so this test fails if the
        // window ever starts deciding readiness for itself.
        let state = RunState::Stopped;
        let limits = ResourceLimits {
            gpu_percent: 100,
            ai: Some(true),
            ..ResourceLimits::default()
        };
        let detected = caps();

        let without = hocmesh::control::advertised_capabilities(&detected, &limits, false);
        assert!(
            !Snapshot::build(readings(&state, None, &limits, &without))
                .resources
                .ai_effective
        );

        let with = hocmesh::control::advertised_capabilities(&detected, &limits, true);
        assert!(
            Snapshot::build(readings(&state, None, &limits, &with))
                .resources
                .ai_effective
        );
    }

    #[test]
    fn consent_to_ai_is_still_shown_when_it_has_not_reached_the_mesh() {
        // The operator said yes and the mesh is not being offered inference.
        // Hiding either half would make the Resources tab a liar: one way it
        // forgets a consent that was given, the other it claims work is being
        // served that is not.
        let state = RunState::Stopped;
        let limits = ResourceLimits {
            gpu_percent: 100,
            ai: Some(true),
            ..ResourceLimits::default()
        };
        let detected = caps();
        let advertised = hocmesh::control::advertised_capabilities(&detected, &limits, false);
        let resources = Snapshot::build(readings(&state, None, &limits, &advertised)).resources;
        assert_eq!(resources.ai, Some(true));
        assert!(!resources.ai_effective);
    }

    #[test]
    fn a_snapshot_survives_the_trip_to_the_page_as_json() {
        // The window renders this object and nothing else, so a field that
        // will not serialise is a blank panel rather than a compile error.
        let state = running();
        let limits = ResourceLimits::default();
        let caps = caps();
        let daemon = daemon();
        let snapshot = Snapshot::build(readings(&state, Some(&daemon), &limits, &caps));
        let text = serde_json::to_string(&snapshot).unwrap();
        let back: Snapshot = serde_json::from_str(&text).unwrap();
        assert_eq!(back, snapshot);
    }
}
