use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Share of local hardware the operator is willing to lend to the hocmesh.
///
/// This is a *user consent* record. Nothing in the hocmesh may cause a node to
/// exceed it: the node derives its own worker count from these values and
/// advertises only the shared slice, so a coordinator cannot request capacity
/// that was never offered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Percentage of logical CPUs offered to the hocmesh (0 = contribute no CPU).
    pub cpu_percent: u8,
    /// Percentage of GPU capacity offered to the hocmesh (0 = contribute no GPU).
    pub gpu_percent: u8,
    /// Ceiling on the share of system memory a workload may occupy.
    pub memory_percent: u8,
    /// Whether this node offers inference to the hocmesh.
    ///
    /// `None` means the operator has not said, and the node falls back to
    /// "offer it when a GPU is lent" -- which is exactly what every node did
    /// before this field existed, so an existing `limits.json` keeps its
    /// behaviour on upgrade.
    ///
    /// It has to be sayable separately because a CPU-only machine has no GPU
    /// share to read consent out of, and reading it out of the CPU share
    /// instead would treat "run allow-listed arithmetic for me" as "run a
    /// stranger's prompt through a stranger's weights on my desktop". Those
    /// are different questions, so the operator gets asked the second one.
    #[serde(default)]
    pub ai: Option<bool>,
}

impl Default for ResourceLimits {
    /// Conservative default: half the CPU, no GPU, half of RAM, and no
    /// stated position on inference.
    ///
    /// GPU defaults to zero because lending a GPU is far more disruptive to an
    /// interactive desktop than lending a couple of cores, so it is opt-in.
    /// `ai` defaults to unstated rather than to `false` so that the fallback,
    /// not this constant, decides -- see [`ResourceLimits::offers_ai`].
    fn default() -> Self {
        Self {
            cpu_percent: 50,
            gpu_percent: 0,
            memory_percent: 50,
            ai: None,
        }
    }
}

impl ResourceLimits {
    /// Reject values that are out of range or that would offer nothing.
    pub fn validate(&self) -> Result<()> {
        if self.cpu_percent == 0 || self.cpu_percent > 100 {
            bail!(
                "cpu_percent must be between 1 and 100 (got {})",
                self.cpu_percent
            );
        }
        if self.gpu_percent > 100 {
            bail!(
                "gpu_percent must be between 0 and 100 (got {})",
                self.gpu_percent
            );
        }
        if self.memory_percent == 0 || self.memory_percent > 100 {
            bail!(
                "memory_percent must be between 1 and 100 (got {})",
                self.memory_percent
            );
        }
        Ok(())
    }

    /// Number of concurrent workers this node may run.
    ///
    /// Rounds down so the operator always keeps at least the share they held
    /// back, but never returns 0 for a valid configuration - a node that
    /// advertises itself must be able to honour at least one assignment.
    pub fn effective_workers(&self, logical_cpus: usize) -> usize {
        let cpus = logical_cpus.max(1);
        let share = (cpus as u64 * self.cpu_percent as u64) / 100;
        (share as usize).clamp(1, cpus)
    }

    /// Bytes of RAM a workload may occupy on this node.
    pub fn shared_memory_bytes(&self, total_memory_bytes: u64) -> u64 {
        total_memory_bytes / 100 * self.memory_percent as u64
    }

    /// Whether the operator lends the GPU at all.
    pub fn offers_gpu(&self) -> bool {
        self.gpu_percent > 0
    }

    /// Whether this node offers inference, given what it still advertises.
    ///
    /// `accelerator_shared` is whether any GPU survived [`Self::offers_gpu`],
    /// which is the signal AI readiness was inferred from before there was a
    /// field for it. An operator who never touched `--ai` therefore gets the
    /// old answer, and one who set it gets theirs.
    pub fn offers_ai(&self, accelerator_shared: bool) -> bool {
        self.ai.unwrap_or(accelerator_shared)
    }

    /// Resolve a requested worker count against the operator's ceiling.
    ///
    /// An explicit request may only ever *lower* the ceiling. This is the
    /// single point where a worker count is decided, so a caller cannot
    /// accidentally opt out of the limit by computing its own.
    pub fn clamp_requested_workers(&self, requested: Option<usize>, logical_cpus: usize) -> usize {
        let allowed = self.effective_workers(logical_cpus);
        requested.map(|w| w.min(allowed)).unwrap_or(allowed).max(1)
    }

    /// Location of the consent record inside the node home directory.
    pub fn path(home: &Path) -> PathBuf {
        home.join("limits.json")
    }

    /// Load the operator's limits, falling back to the conservative default.
    ///
    /// A malformed file is an error rather than a silent reset: silently
    /// widening what a machine lends is exactly the failure we must not have.
    pub fn load_or_default(home: &Path) -> Result<Self> {
        let path = Self::path(home);
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let limits: Self =
            serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
        limits.validate()?;
        Ok(limits)
    }

    /// Persist the operator's limits, validating before anything is written.
    pub fn save(&self, home: &Path) -> Result<()> {
        self.validate()?;
        fs::create_dir_all(home).with_context(|| format!("creating {}", home.display()))?;
        let path = Self::path(home);
        let body = serde_json::to_string_pretty(self)?;
        fs::write(&path, body).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_conservative_and_valid() {
        let limits = ResourceLimits::default();
        limits.validate().unwrap();
        assert_eq!(limits.cpu_percent, 50);
        assert_eq!(limits.gpu_percent, 0);
        assert!(!limits.offers_gpu());
    }

    #[test]
    fn workers_scale_with_cpu_share() {
        let half = ResourceLimits {
            cpu_percent: 50,
            ..Default::default()
        };
        assert_eq!(half.effective_workers(8), 4);
        assert_eq!(half.effective_workers(16), 8);

        let quarter = ResourceLimits {
            cpu_percent: 25,
            ..Default::default()
        };
        assert_eq!(quarter.effective_workers(8), 2);

        let all = ResourceLimits {
            cpu_percent: 100,
            ..Default::default()
        };
        assert_eq!(all.effective_workers(8), 8);
    }

    #[test]
    fn worker_count_never_exceeds_the_machine_or_drops_to_zero() {
        let tiny = ResourceLimits {
            cpu_percent: 1,
            ..Default::default()
        };
        assert_eq!(
            tiny.effective_workers(2),
            1,
            "must still honour one assignment"
        );

        let all = ResourceLimits {
            cpu_percent: 100,
            ..Default::default()
        };
        assert_eq!(all.effective_workers(1), 1, "cannot exceed logical cpus");
        assert_eq!(
            all.effective_workers(0),
            1,
            "degenerate cpu count is clamped"
        );
    }

    #[test]
    fn out_of_range_values_are_rejected() {
        assert!(
            ResourceLimits {
                cpu_percent: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            ResourceLimits {
                cpu_percent: 101,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            ResourceLimits {
                gpu_percent: 101,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            ResourceLimits {
                memory_percent: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn a_small_machine_lending_half_of_itself_allows_one_worker() {
        // Two logical CPUs at the default half share round down to one. An
        // operator asking for two does not get two, and a node that
        // advertises itself must still be able to take one assignment -- so
        // the answer is 1, never 0 and never the 2 that was asked for. This
        // is the size of a stock CI runner, which is why it is pinned.
        let default = ResourceLimits::default();
        assert_eq!(default.cpu_percent, 50);
        assert_eq!(default.effective_workers(2), 1);
        assert_eq!(default.clamp_requested_workers(Some(2), 2), 1);
        assert_eq!(default.clamp_requested_workers(None, 2), 1);
        // Three cores round down the same way; four is the first size that
        // honours a request for two.
        assert_eq!(default.clamp_requested_workers(Some(2), 3), 1);
        assert_eq!(default.clamp_requested_workers(Some(2), 4), 2);
    }

    #[test]
    fn an_explicit_worker_request_can_only_lower_the_ceiling() {
        let half = ResourceLimits {
            cpu_percent: 50,
            ..Default::default()
        };
        // Ceiling on an 8-core box is 4.
        assert_eq!(half.clamp_requested_workers(None, 8), 4);
        assert_eq!(
            half.clamp_requested_workers(Some(2), 8),
            2,
            "may ask for less"
        );
        assert_eq!(
            half.clamp_requested_workers(Some(8), 8),
            4,
            "may not ask for more"
        );
        assert_eq!(
            half.clamp_requested_workers(Some(999), 8),
            4,
            "absurd request clamped"
        );
        assert_eq!(
            half.clamp_requested_workers(Some(0), 8),
            1,
            "zero still yields one"
        );
    }

    #[test]
    fn memory_share_is_proportional() {
        let limits = ResourceLimits {
            memory_percent: 25,
            ..Default::default()
        };
        assert_eq!(limits.shared_memory_bytes(16_000_000_000), 4_000_000_000);
    }

    fn scratch_home() -> PathBuf {
        std::env::temp_dir().join(format!("hocmesh-limits-{}", uuid::Uuid::new_v4()))
    }

    #[test]
    fn save_then_load_round_trips() {
        let home = scratch_home();
        let limits = ResourceLimits {
            cpu_percent: 30,
            gpu_percent: 70,
            memory_percent: 40,
            ai: None,
        };
        limits.save(&home).unwrap();
        assert_eq!(ResourceLimits::load_or_default(&home).unwrap(), limits);
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn missing_file_yields_the_conservative_default() {
        let home = scratch_home();
        assert_eq!(
            ResourceLimits::load_or_default(&home).unwrap(),
            ResourceLimits::default()
        );
    }

    #[test]
    fn an_unstated_ai_share_keeps_the_pre_existing_rule() {
        // Every `limits.json` written before the field existed deserialises to
        // this, so this case is what "upgrading changes nobody's behaviour"
        // actually means. Lending a GPU offered inference; not lending one did
        // not.
        let unstated = ResourceLimits::default();
        assert!(unstated.ai.is_none());
        assert!(unstated.offers_ai(true));
        assert!(!unstated.offers_ai(false));
    }

    #[test]
    fn a_stated_ai_share_overrides_the_gpu_signal_in_both_directions() {
        let on = ResourceLimits {
            ai: Some(true),
            ..Default::default()
        };
        // The whole point: a machine with nothing to accelerate on can still
        // say yes.
        assert!(on.offers_ai(false));
        assert!(on.offers_ai(true));

        let off = ResourceLimits {
            ai: Some(false),
            ..Default::default()
        };
        // And lending a GPU no longer conscripts the node into inference.
        assert!(!off.offers_ai(true));
        assert!(!off.offers_ai(false));
    }

    #[test]
    fn a_limits_file_written_before_the_ai_field_existed_still_loads() {
        let home = scratch_home();
        fs::create_dir_all(&home).unwrap();
        fs::write(
            ResourceLimits::path(&home),
            r#"{"cpu_percent":50,"gpu_percent":25,"memory_percent":50}"#,
        )
        .unwrap();
        let loaded = ResourceLimits::load_or_default(&home).unwrap();
        assert_eq!(loaded.ai, None);
        assert!(loaded.offers_ai(true));
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn a_corrupt_or_over_permissive_file_fails_loudly() {
        // Silently falling back to a default here would widen what the machine
        // lends without the operator ever asking for it.
        let home = scratch_home();
        fs::create_dir_all(&home).unwrap();
        fs::write(ResourceLimits::path(&home), "{ not json").unwrap();
        assert!(ResourceLimits::load_or_default(&home).is_err());

        fs::write(
            ResourceLimits::path(&home),
            r#"{"cpu_percent":250,"gpu_percent":0,"memory_percent":50}"#,
        )
        .unwrap();
        assert!(ResourceLimits::load_or_default(&home).is_err());
        let _ = fs::remove_dir_all(&home);
    }
}
