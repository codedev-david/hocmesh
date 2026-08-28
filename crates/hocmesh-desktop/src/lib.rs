//! hocMESH Desktop -- the tray application and dashboard for a node.
//!
//! The desktop app is a *front end*, not a second implementation of the node.
//! It reads the same `limits.json` the daemon reads, asks a running daemon for
//! its live state over the loopback control surface, and asks the coordinator
//! for the ledger. It never computes a number the node could have computed,
//! because two implementations of "how much of this machine is lent" would
//! eventually disagree, and the disagreement would be about consent.
//!
//! The crate is laid out so that almost everything is testable without a
//! display server:
//!
//! - [`format`] turns raw numbers into the strings the window shows.
//! - [`settings`] is the app's own preferences -- never consent, which lives
//!   in the node's `limits.json`.
//! - [`supervisor`] starts, finds and stops the node process.
//! - [`dashboard`] folds every reading into one [`dashboard::Snapshot`].
//! - [`tray`] decides what the tray icon says and which items are live.
//! - [`node`] gathers the readings.
//! - [`app`] is the Tauri wiring, and is the only part a test cannot reach.

/// The window, the tray and the event loop. Behind the `gui` feature so the
/// rest of this crate -- which is where the rules live -- can be built and
/// tested on a machine with no display stack at all, and so the integration
/// tests can drive a real daemon through [`node::Node`] without linking a
/// webview.
#[cfg(feature = "gui")]
pub mod app;
pub mod dashboard;
pub mod format;
pub mod node;
pub mod settings;
pub mod supervisor;
pub mod tray;

#[cfg(feature = "gui")]
pub use app::run;

/// The app's own version, shown next to the node's so a mismatch is visible.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn the_app_reports_a_real_version() {
        assert!(!super::VERSION.is_empty());
        assert!(
            super::VERSION.split('.').count() >= 3,
            "the window shows this next to the node's version, so it has to be comparable"
        );
    }
}
