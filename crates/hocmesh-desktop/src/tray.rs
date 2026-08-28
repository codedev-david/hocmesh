//! The system tray: what the icon says and which menu items are live.
//!
//! The tray is the app most of the time. An operator who has set this machine
//! up looks at the icon and nothing else for weeks, so what the icon claims
//! has to be exactly true -- a green icon over a node that is not reaching its
//! coordinator would be worse than no icon at all.
//!
//! The menu's *shape* is decided here, as data, so the rules about which items
//! are enabled can be tested. The Tauri wiring that turns this into a real
//! menu lives in [`crate::app`] and is the only part a test cannot reach.

use crate::dashboard::{Health, Snapshot};
use serde::{Deserialize, Serialize};

/// Menu item ids, shared between the model here and the Tauri menu built from
/// it. Constants rather than literals because a typo in a handler's match arm
/// is a menu item that silently does nothing.
pub const ID_STATUS: &str = "status";
pub const ID_DASHBOARD: &str = "dashboard";
pub const ID_START: &str = "start";
pub const ID_STOP: &str = "stop";
pub const ID_RESTART: &str = "restart";
pub const ID_QUIT: &str = "quit";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MenuItem {
    pub id: String,
    pub label: String,
    pub enabled: bool,
}

/// The whole tray, as data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrayModel {
    pub tooltip: String,
    pub health: Health,
    pub items: Vec<MenuItem>,
}

impl TrayModel {
    /// Build the menu for a snapshot.
    ///
    /// `blocker` is the reason the node cannot be started, if there is one. It
    /// is shown in place of Start rather than left to a failed click: a
    /// disabled item that says why is a better answer than an error dialog
    /// after the fact.
    pub fn from_snapshot(snapshot: &Snapshot, blocker: Option<&str>) -> Self {
        let running = snapshot.overview.running;
        let status = match blocker {
            Some(reason) if !running => format!("Cannot start: {reason}"),
            _ => snapshot.overview.health_label.clone(),
        };
        let can_start = !running && blocker.is_none();
        Self {
            tooltip: snapshot.tray_tooltip.clone(),
            health: snapshot.overview.health,
            items: vec![
                MenuItem {
                    id: ID_STATUS.into(),
                    label: status,
                    // The status line is a label, not a button. Leaving it
                    // clickable would invite a click that does nothing.
                    enabled: false,
                },
                MenuItem {
                    id: ID_DASHBOARD.into(),
                    label: "Open Dashboard".into(),
                    enabled: true,
                },
                MenuItem {
                    id: ID_START.into(),
                    label: "Start Node".into(),
                    enabled: can_start,
                },
                MenuItem {
                    id: ID_STOP.into(),
                    label: "Stop Node".into(),
                    enabled: running,
                },
                MenuItem {
                    id: ID_RESTART.into(),
                    label: "Restart Node".into(),
                    enabled: running && blocker.is_none(),
                },
                MenuItem {
                    id: ID_QUIT.into(),
                    label: "Quit hocMESH Desktop".into(),
                    enabled: true,
                },
            ],
        }
    }

    pub fn item(&self, id: &str) -> Option<&MenuItem> {
        self.items.iter().find(|item| item.id == id)
    }

    /// Whether an item is live. Unknown ids are not enabled, so a typo in a
    /// caller reads as "greyed out" rather than as "clickable and inert".
    pub fn enabled(&self, id: &str) -> bool {
        self.item(id).is_some_and(|item| item.enabled)
    }
}

/// The icon file for a health state.
///
/// Three icons rather than one with an overlay: an overlay drawn at tray size
/// is a few pixels and reads as nothing on a busy taskbar.
pub fn icon_name(health: Health) -> &'static str {
    match health {
        Health::Working => "tray-working.png",
        Health::Degraded => "tray-degraded.png",
        Health::Stopped => "tray-stopped.png",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dashboard::{Health, Overview, Resources, Snapshot};

    fn snapshot(health: Health, running: bool) -> Snapshot {
        Snapshot {
            overview: Overview {
                health,
                health_label: health.label().into(),
                running,
                supervised: false,
                node_id: None,
                coordinator: "http://mesh.example".into(),
                node_version: None,
                app_version: "0.3.0".into(),
                uptime: None,
                workers: None,
                jobs_completed: 0,
                jobs_failed: 0,
                inferences_completed: 0,
                earned_this_run: "0.000".into(),
                last_contact: "never".into(),
                last_error: None,
                ai_offered: false,
            },
            resources: Resources {
                cpu_percent: 50,
                memory_percent: 50,
                gpu_percent: 0,
                ai: None,
                ai_effective: false,
                logical_cpus: 8,
                shared_logical_cpus: 4,
                total_memory: "16.0 GiB".into(),
                shared_memory: "8.0 GiB".into(),
                shared_memory_percent_of_machine: 50,
                cpu_brand: "Test".into(),
                hostname: "desk".into(),
                os: "test".into(),
                arch: "test".into(),
                accelerators: Vec::new(),
                restart_required: false,
            },
            ledger: None,
            ledger_error: None,
            tray_tooltip: format!("hocMESH — {}", health.label()),
        }
    }

    #[test]
    fn a_stopped_node_offers_start_and_nothing_that_needs_it_running() {
        let model = TrayModel::from_snapshot(&snapshot(Health::Stopped, false), None);
        assert!(model.enabled(ID_START));
        assert!(!model.enabled(ID_STOP));
        assert!(!model.enabled(ID_RESTART));
    }

    #[test]
    fn a_running_node_offers_stop_and_restart_but_not_a_second_start() {
        let model = TrayModel::from_snapshot(&snapshot(Health::Working, true), None);
        assert!(
            !model.enabled(ID_START),
            "a second daemon would fight the first over the same home"
        );
        assert!(model.enabled(ID_STOP));
        assert!(model.enabled(ID_RESTART));
    }

    #[test]
    fn something_that_blocks_a_start_is_said_out_loud_rather_than_failing_on_click() {
        let model = TrayModel::from_snapshot(
            &snapshot(Health::Stopped, false),
            Some("hocmesh.exe was not found beside this app or on PATH"),
        );
        assert!(!model.enabled(ID_START));
        assert_eq!(
            model.item(ID_STATUS).unwrap().label,
            "Cannot start: hocmesh.exe was not found beside this app or on PATH"
        );
    }

    #[test]
    fn a_blocker_does_not_hide_the_state_of_a_node_that_is_already_running() {
        // A misconfigured coordinator setting must not make a healthy running
        // node read as "cannot start" -- it is not trying to start.
        let model =
            TrayModel::from_snapshot(&snapshot(Health::Working, true), Some("no coordinator"));
        assert_eq!(model.item(ID_STATUS).unwrap().label, "Contributing");
    }

    #[test]
    fn the_dashboard_and_quit_are_always_reachable() {
        // These are the two ways out of any bad state, including one where the
        // node cannot start at all.
        for (health, running) in [
            (Health::Stopped, false),
            (Health::Degraded, true),
            (Health::Working, true),
        ] {
            let model = TrayModel::from_snapshot(&snapshot(health, running), Some("blocked"));
            assert!(model.enabled(ID_DASHBOARD));
            assert!(model.enabled(ID_QUIT));
        }
    }

    #[test]
    fn the_status_line_is_a_label_and_never_a_button() {
        let model = TrayModel::from_snapshot(&snapshot(Health::Working, true), None);
        assert!(!model.enabled(ID_STATUS));
    }

    #[test]
    fn every_health_state_has_its_own_icon() {
        let names = [
            icon_name(Health::Working),
            icon_name(Health::Degraded),
            icon_name(Health::Stopped),
        ];
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(
            unique.len(),
            3,
            "two states sharing an icon would make the tray unable to say which one it is in"
        );
    }

    #[test]
    fn a_degraded_node_is_not_shown_with_the_working_icon() {
        // The whole value of the tray is that this distinction survives to the
        // pixel an operator glances at.
        assert_ne!(icon_name(Health::Degraded), icon_name(Health::Working));
    }

    #[test]
    fn the_tooltip_comes_from_the_snapshot_rather_than_being_reassembled() {
        let snapshot = snapshot(Health::Degraded, true);
        let model = TrayModel::from_snapshot(&snapshot, None);
        assert_eq!(model.tooltip, snapshot.tray_tooltip);
    }

    #[test]
    fn every_menu_id_is_distinct_so_a_handler_cannot_match_the_wrong_item() {
        let model = TrayModel::from_snapshot(&snapshot(Health::Working, true), None);
        let ids: std::collections::HashSet<_> = model.items.iter().map(|i| &i.id).collect();
        assert_eq!(ids.len(), model.items.len());
    }
}
