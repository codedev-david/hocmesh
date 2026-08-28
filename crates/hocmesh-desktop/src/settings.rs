//! What the app remembers between launches.
//!
//! Deliberately small, and deliberately *not* the place where the operator's
//! consent lives. How much of this machine is lent is recorded in the node's
//! own `limits.json`, because the node enforces it whether or not a window is
//! open. What is stored here is only how to find that node and how to launch
//! it: the home directory, the mesh it joins, and a couple of launch
//! preferences.
//!
//! Keeping the split this way round means uninstalling the desktop app cannot
//! change what a machine shares, and editing settings here cannot widen a
//! share behind the daemon's back.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The coordinator a fresh install points at.
pub const DEFAULT_COORDINATOR: &str = "http://127.0.0.1:8080";

/// The app's own preferences.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// The node home this app watches. One window, one home -- an operator
    /// running two nodes runs two windows, which keeps every reading on
    /// screen unambiguous.
    pub home: PathBuf,
    pub coordinator: String,
    /// Start the node when the app starts, the way Docker Desktop brings its
    /// engine up with the window.
    pub start_node_with_app: bool,
    /// A worker ceiling the operator chose. `None` leaves it to the daemon,
    /// which derives it from the lent CPU share.
    pub workers: Option<u32>,
    /// Decline inference work when launching from here.
    pub no_ai: bool,
    /// Fixed control port, for an operator who wants a predictable one. `0`
    /// takes a free port.
    pub control_port: u16,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            home: default_home(),
            coordinator: DEFAULT_COORDINATOR.into(),
            start_node_with_app: false,
            workers: None,
            no_ai: false,
            control_port: 0,
        }
    }
}

/// The node's default home, matching the CLI's.
///
/// The app and the CLI must agree about where a node lives, or an operator who
/// set limits with one would find the other watching a different node
/// entirely.
pub fn default_home() -> PathBuf {
    if let Ok(explicit) = std::env::var("HOCMESH_HOME")
        && !explicit.is_empty()
    {
        return PathBuf::from(explicit);
    }
    home_root().join(".hocmesh")
}

fn home_root() -> PathBuf {
    #[cfg(windows)]
    let candidates = ["USERPROFILE", "HOME"];
    #[cfg(not(windows))]
    let candidates = ["HOME"];
    for key in candidates {
        if let Ok(value) = std::env::var(key)
            && !value.is_empty()
        {
            return PathBuf::from(value);
        }
    }
    PathBuf::from(".")
}

impl Settings {
    pub fn path(config_dir: &Path) -> PathBuf {
        config_dir.join("desktop-settings.json")
    }

    /// Read the stored settings, falling back to defaults.
    ///
    /// A settings file that will not parse is replaced by defaults rather than
    /// refused: the app is how an operator fixes a broken machine, so it has
    /// to open even when its own preferences are damaged. Nothing here is
    /// authoritative over what the machine shares, so defaulting is safe --
    /// the consent record is untouched either way.
    pub fn load(config_dir: &Path) -> Self {
        let path = Self::path(config_dir);
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        serde_json::from_str(&text).unwrap_or_default()
    }

    pub fn save(&self, config_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(config_dir)
            .with_context(|| format!("could not create {}", config_dir.display()))?;
        let path = Self::path(config_dir);
        let text = serde_json::to_string_pretty(self).context("could not encode settings")?;
        std::fs::write(&path, text).with_context(|| format!("could not write {}", path.display()))
    }

    /// Normalise what a form sent before it is stored.
    ///
    /// A trailing slash on the coordinator would produce `//v1/...` on every
    /// request, and a blank coordinator would make every call fail with a
    /// confusing error rather than an obvious one.
    pub fn normalised(mut self) -> Self {
        self.coordinator = self.coordinator.trim().trim_end_matches('/').to_string();
        if self.coordinator.is_empty() {
            self.coordinator = DEFAULT_COORDINATOR.into();
        }
        if self.home.as_os_str().is_empty() {
            self.home = default_home();
        }
        // Zero workers would be a node that joined the mesh and did nothing.
        // An operator who wants that stops the node instead.
        self.workers = self.workers.filter(|count| *count > 0);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn scratch(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("hocmesh-desktop-{tag}-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn settings_survive_a_round_trip_through_disk() {
        let dir = scratch("settings-round-trip");
        let settings = Settings {
            home: dir.join("node"),
            coordinator: "https://mesh.example".into(),
            start_node_with_app: true,
            workers: Some(6),
            no_ai: true,
            control_port: 7788,
        };
        settings.save(&dir).unwrap();
        assert_eq!(Settings::load(&dir), settings);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_missing_settings_file_reads_as_defaults_rather_than_an_error() {
        let dir = scratch("settings-missing");
        assert_eq!(Settings::load(&dir), Settings::default());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_damaged_settings_file_still_lets_the_app_open() {
        // The app is how an operator fixes a machine. Refusing to start
        // because its own preferences file is corrupt would take away the tool
        // at the moment it is needed.
        let dir = scratch("settings-corrupt");
        fs::write(Settings::path(&dir), b"{ this is not json").unwrap();
        assert_eq!(Settings::load(&dir), Settings::default());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_settings_file_missing_newer_fields_keeps_its_old_answers() {
        // An older install's file must not reset the operator's coordinator
        // just because a field was added since.
        let dir = scratch("settings-partial");
        fs::write(
            Settings::path(&dir),
            br#"{"coordinator":"https://mesh.example"}"#,
        )
        .unwrap();
        let loaded = Settings::load(&dir);
        assert_eq!(loaded.coordinator, "https://mesh.example");
        assert_eq!(loaded.home, default_home());
        assert_eq!(loaded.control_port, 0);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_trailing_slash_on_the_coordinator_is_trimmed_before_it_doubles_a_path() {
        let settings = Settings {
            coordinator: "  https://mesh.example/  ".into(),
            ..Settings::default()
        }
        .normalised();
        assert_eq!(settings.coordinator, "https://mesh.example");
    }

    #[test]
    fn a_blank_coordinator_falls_back_rather_than_failing_every_later_call() {
        let settings = Settings {
            coordinator: "   ".into(),
            ..Settings::default()
        }
        .normalised();
        assert_eq!(settings.coordinator, DEFAULT_COORDINATOR);
    }

    #[test]
    fn a_zero_worker_ceiling_is_dropped_rather_than_joining_a_mesh_to_do_nothing() {
        let settings = Settings {
            workers: Some(0),
            ..Settings::default()
        }
        .normalised();
        assert_eq!(settings.workers, None);
    }

    #[test]
    fn a_blank_home_falls_back_to_the_one_the_cli_uses() {
        let settings = Settings {
            home: PathBuf::new(),
            ..Settings::default()
        }
        .normalised();
        assert_eq!(settings.home, default_home());
    }

    #[test]
    fn saving_creates_the_config_directory_it_was_given() {
        let dir = scratch("settings-mkdir");
        let nested = dir.join("a").join("b");
        Settings::default().save(&nested).unwrap();
        assert!(Settings::path(&nested).is_file());
        fs::remove_dir_all(dir).unwrap();
    }
}
