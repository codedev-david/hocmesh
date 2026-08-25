//! Latency-space network coordinates.
//!
//! Every node keeps a small synthetic coordinate that travels with its
//! capabilities. The distance between two coordinates predicts the round-trip
//! time between those two machines, so "who is nearest?" becomes local
//! arithmetic instead of an O(n^2) probe mesh or a question for a server.
//!
//! This is Vivaldi (Dabek et al., SIGCOMM 2004) with a height term: the vector
//! models the shared internet core, the height models the private access link
//! that traffic must cross at both ends of a path regardless of direction.
//!
//! A coordinate is an untrusted input: it arrives from whichever peer we just
//! spoke to. Every bound in this module exists so that a peer reporting an
//! absurd or adversarial position cannot drag an honest node's coordinate
//! anywhere useful to the attacker.
//!
//! Sample source: a coordinate is only meaningful once it has been fitted
//! against node-to-node round trips. Distance to a single landmark (the
//! coordinator) fits nothing useful - it would place every node on a sphere
//! and call opposite sides of the planet adjacent - so nodes advertise `None`
//! until a peer-probe source exists, and schedulers fall back accordingly.

use anyhow::{Context, Result};
use mesh_protocol::{COORDINATE_DIMENSIONS, NetworkCoordinate};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Fraction of the error estimate corrected per observation.
const ERROR_SENSITIVITY: f64 = 0.25;
/// Fraction of the positional error corrected per observation.
const POSITION_SENSITIVITY: f64 = 0.25;
/// A coordinate that has never been fitted is maximally unsure of itself.
const INITIAL_ERROR: f64 = 1.0;
/// Floor on confidence, so a peer claiming certainty cannot dominate the fit.
const MIN_ERROR: f64 = 0.05;
/// Positions beyond a minute of round trip are nonsense; reject them.
const MAX_COORDINATE_MICROS: i64 = 60_000_000;
/// No single observation may move us further than this, in microseconds.
const MAX_STEP_MICROS: f64 = 5_000_000.0;
/// Below this vector magnitude the direction is meaningless.
const ZERO_THRESHOLD: f64 = 1.0e-6;
/// Spread of the identity-derived starting jitter, in microseconds either way.
const SEED_JITTER_MICROS: u64 = 1_000;
/// FNV-1a, used only to scatter starting positions. Not a security primitive.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Predicted round-trip time between two coordinates, in microseconds.
///
/// Heights add because each end of the path crosses its own access link.
pub fn predicted_rtt_micros(a: &NetworkCoordinate, b: &NetworkCoordinate) -> u64 {
    let mut sum = 0.0f64;
    for axis in 0..COORDINATE_DIMENSIONS {
        let delta = (a.vector_micros[axis] - b.vector_micros[axis]) as f64;
        sum += delta * delta;
    }
    let distance = sum.sqrt() + a.height_micros.max(0) as f64 + b.height_micros.max(0) as f64;
    distance.max(0.0) as u64
}

/// Whether a coordinate reported by a peer is inside the range we will accept.
///
/// A coordinate is a free-form number from an untrusted party. Refusing the
/// out-of-range ones here is what stops a peer from teleporting our position.
pub fn is_plausible(coord: &NetworkCoordinate) -> bool {
    coord.error_permille <= 1000
        && coord.height_micros >= 0
        && coord.height_micros <= MAX_COORDINATE_MICROS
        && coord
            .vector_micros
            .iter()
            .all(|axis| axis.abs() <= MAX_COORDINATE_MICROS)
}

/// A node's own position, refined by every latency measurement it makes.
#[derive(Debug, Clone)]
pub struct Vivaldi {
    vector: [f64; COORDINATE_DIMENSIONS],
    height: f64,
    error: f64,
    observations: u64,
}

impl Default for Vivaldi {
    fn default() -> Self {
        Self {
            vector: [0.0; COORDINATE_DIMENSIONS],
            height: 0.0,
            error: INITIAL_ERROR,
            observations: 0,
        }
    }
}

impl Vivaldi {
    /// A fresh position at the origin.
    ///
    /// Prefer [`Vivaldi::seeded`] on a real node: two coordinates that sit on
    /// exactly the same point have no direction to separate along, so a pair
    /// of unseeded nodes measuring only each other never spreads apart.
    pub fn new() -> Self {
        Self::default()
    }

    /// A fresh position jittered deterministically from `seed`.
    ///
    /// Pass the node's public key. Vivaldi breaks ties with a random direction;
    /// deriving it from identity instead keeps startup reproducible, which
    /// matters when a coordinate has to be explained after the fact.
    pub fn seeded(seed: &[u8]) -> Self {
        let mut node = Self::default();
        let mut hash = FNV_OFFSET;
        for (axis, slot) in node.vector.iter_mut().enumerate() {
            for byte in seed.iter().chain(std::iter::once(&(axis as u8))) {
                hash = (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME);
            }
            let spread = (hash % (2 * SEED_JITTER_MICROS + 1)) as f64;
            *slot = spread - SEED_JITTER_MICROS as f64;
        }
        node
    }

    /// How many measurements have been folded in so far.
    pub fn observations(&self) -> u64 {
        self.observations
    }

    /// The wire form of this position, or `None` until it has been fitted.
    ///
    /// Advertising an unfitted coordinate would place this node at the origin,
    /// which every other node would read as "adjacent to me".
    pub fn coordinate(&self) -> Option<NetworkCoordinate> {
        if self.observations == 0 {
            return None;
        }
        Some(self.provisional_coordinate())
    }

    /// This position as it stands, fitted or not, for peers to measure against.
    ///
    /// Two nodes that have both never measured anything would otherwise have
    /// nothing to fit against and the network could never start. The way out
    /// is not to pretend an unfitted node is placed, but to separate the two
    /// audiences: a *peer* gets the raw position carrying `error_permille` at
    /// its maximum, which [`Vivaldi::observe`] weights down to almost nothing,
    /// while the *scheduler* keeps seeing `None` from [`Vivaldi::coordinate`]
    /// until the position means something. Confidence travels with the
    /// coordinate, so an unfitted peer is used but barely trusted.
    pub fn provisional_coordinate(&self) -> NetworkCoordinate {
        let mut vector_micros = [0i64; COORDINATE_DIMENSIONS];
        for (axis, slot) in vector_micros.iter_mut().enumerate() {
            *slot = clamp_axis(self.vector[axis]);
        }
        NetworkCoordinate {
            vector_micros,
            height_micros: clamp_axis(self.height.max(0.0)),
            error_permille: (self.error.clamp(0.0, 1.0) * 1000.0).round() as u16,
        }
    }

    /// Fold one measured round trip to `remote` into this node's position.
    ///
    /// Returns `false` when the observation was refused. A refusal leaves the
    /// coordinate exactly as it was, so a peer that lies loudly is simply
    /// ignored rather than being allowed to steer us.
    pub fn observe(&mut self, remote: &NetworkCoordinate, rtt_micros: u64) -> bool {
        if rtt_micros == 0 || !is_plausible(remote) {
            return false;
        }
        let measured = rtt_micros as f64;
        let remote_error = (f64::from(remote.error_permille) / 1000.0).clamp(MIN_ERROR, 1.0);
        let weight = self.error / (self.error + remote_error);
        let predicted = self.predicted_to(remote);
        let relative_error = (predicted - measured).abs() / measured;
        let alpha = ERROR_SENSITIVITY * weight;
        self.error =
            (relative_error * alpha + self.error * (1.0 - alpha)).clamp(MIN_ERROR, INITIAL_ERROR);

        let force = (POSITION_SENSITIVITY * weight * (measured - predicted))
            .clamp(-MAX_STEP_MICROS, MAX_STEP_MICROS);
        self.apply_force(remote, force);
        self.observations = self.observations.saturating_add(1);
        true
    }

    /// Round trip this node currently expects to `remote`, in microseconds.
    fn predicted_to(&self, remote: &NetworkCoordinate) -> f64 {
        let mut sum = 0.0f64;
        for axis in 0..COORDINATE_DIMENSIONS {
            let delta = self.vector[axis] - remote.vector_micros[axis] as f64;
            sum += delta * delta;
        }
        sum.sqrt() + self.height.max(0.0) + remote.height_micros.max(0) as f64
    }

    /// Push this coordinate away from (or toward) `remote` by `force`.
    fn apply_force(&mut self, remote: &NetworkCoordinate, force: f64) {
        let mut unit = [0.0f64; COORDINATE_DIMENSIONS];
        let mut magnitude = 0.0f64;
        for (axis, slot) in unit.iter_mut().enumerate() {
            *slot = self.vector[axis] - remote.vector_micros[axis] as f64;
            magnitude += *slot * *slot;
        }
        magnitude = magnitude.sqrt();
        if magnitude > ZERO_THRESHOLD {
            for slot in unit.iter_mut() {
                *slot /= magnitude;
            }
        } else {
            // Co-located coordinates have no direction to push along. Step
            // along one axis, rotating deterministically so that repeated
            // collisions do not pile onto the same one.
            unit[(self.observations as usize) % COORDINATE_DIMENSIONS] = 1.0;
        }

        for (axis, step) in unit.iter().enumerate() {
            self.vector[axis] = (self.vector[axis] + step * force)
                .clamp(-MAX_COORDINATE_MICROS as f64, MAX_COORDINATE_MICROS as f64);
        }
        if magnitude > ZERO_THRESHOLD {
            let height = (self.height + remote.height_micros.max(0) as f64) * force / magnitude
                + self.height;
            self.height = height.clamp(0.0, MAX_COORDINATE_MICROS as f64);
        }
    }
}

/// On-disk form of a fitted position.
///
/// The wire type is fixed point because peers must agree byte-for-byte; this
/// one keeps full precision because it is only ever read back by the node that
/// wrote it, and rounding a position on every restart would slowly walk it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProximityState {
    vector_micros: [f64; COORDINATE_DIMENSIONS],
    height_micros: f64,
    error: f64,
    observations: u64,
}

impl Vivaldi {
    /// Location of the fitted position inside the node home directory.
    pub fn path(home: &Path) -> PathBuf {
        home.join("proximity.json")
    }

    /// The position fitted by earlier runs, or a fresh one seeded from `seed`.
    ///
    /// A missing or unreadable file is not an error. Unlike the operator's
    /// resource limits - where a silent default would widen what a machine
    /// lends - losing a coordinate costs only the time it takes to re-fit,
    /// and refusing to start over a corrupt cache would be worse.
    pub fn load_or_seeded(home: &Path, seed: &[u8]) -> Self {
        fs::read_to_string(Self::path(home))
            .ok()
            .and_then(|raw| serde_json::from_str::<ProximityState>(&raw).ok())
            .and_then(Self::from_state)
            .unwrap_or_else(|| Self::seeded(seed))
    }

    /// Persist the fitted position so a restart does not start from nowhere.
    pub fn save(&self, home: &Path) -> Result<()> {
        fs::create_dir_all(home).with_context(|| format!("creating {}", home.display()))?;
        let path = Self::path(home);
        let body = serde_json::to_string_pretty(&self.to_state())?;
        fs::write(&path, body).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    /// Rebuild from a stored state, rejecting anything that is not a position.
    ///
    /// A file containing NaN or an out-of-range axis would poison every
    /// subsequent observation, so it is discarded in favour of a fresh start.
    fn from_state(state: ProximityState) -> Option<Self> {
        let finite = state
            .vector_micros
            .iter()
            .all(|axis| axis.is_finite() && axis.abs() <= MAX_COORDINATE_MICROS as f64);
        if !finite
            || !state.height_micros.is_finite()
            || !(0.0..=MAX_COORDINATE_MICROS as f64).contains(&state.height_micros)
            || !state.error.is_finite()
            || !(MIN_ERROR..=INITIAL_ERROR).contains(&state.error)
        {
            return None;
        }
        Some(Self {
            vector: state.vector_micros,
            height: state.height_micros,
            error: state.error,
            observations: state.observations,
        })
    }

    /// The storable form of this position.
    fn to_state(&self) -> ProximityState {
        ProximityState {
            vector_micros: self.vector,
            height_micros: self.height,
            error: self.error,
            observations: self.observations,
        }
    }
}

/// Convert a float position to the wire's fixed point, inside the sane range.
fn clamp_axis(value: f64) -> i64 {
    if !value.is_finite() {
        return 0;
    }
    value
        .round()
        .clamp(-MAX_COORDINATE_MICROS as f64, MAX_COORDINATE_MICROS as f64) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run two nodes against each other until their coordinates settle.
    fn settle(a: &mut Vivaldi, b: &mut Vivaldi, rtt_micros: u64, rounds: usize) {
        for _ in 0..rounds {
            let (ca, cb) = (a.coordinate(), b.coordinate());
            a.observe(&cb.unwrap_or_default(), rtt_micros);
            b.observe(&ca.unwrap_or_default(), rtt_micros);
        }
    }

    #[test]
    fn an_unfitted_node_advertises_no_coordinate() {
        assert_eq!(Vivaldi::new().coordinate(), None);
    }

    #[test]
    fn a_pair_converges_on_the_measured_round_trip() {
        let (mut a, mut b) = (Vivaldi::seeded(b"node-a"), Vivaldi::seeded(b"node-b"));
        settle(&mut a, &mut b, 100_000, 200);
        let predicted = predicted_rtt_micros(&a.coordinate().unwrap(), &b.coordinate().unwrap());
        assert!(
            predicted.abs_diff(100_000) < 10_000,
            "predicted {predicted}us for a measured 100000us link"
        );
    }

    #[test]
    fn nearby_peers_rank_ahead_of_distant_ones() {
        let (mut near_a, mut near_b) = (Vivaldi::seeded(b"near-a"), Vivaldi::seeded(b"near-b"));
        settle(&mut near_a, &mut near_b, 2_000, 200);
        let (mut far_a, mut far_b) = (Vivaldi::seeded(b"far-a"), Vivaldi::seeded(b"far-b"));
        settle(&mut far_a, &mut far_b, 250_000, 200);
        let close =
            predicted_rtt_micros(&near_a.coordinate().unwrap(), &near_b.coordinate().unwrap());
        let distant =
            predicted_rtt_micros(&far_a.coordinate().unwrap(), &far_b.coordinate().unwrap());
        assert!(
            close < distant,
            "close {close}us should beat distant {distant}us"
        );
    }

    #[test]
    fn confidence_improves_as_observations_agree() {
        let (mut a, mut b) = (Vivaldi::seeded(b"node-a"), Vivaldi::seeded(b"node-b"));
        settle(&mut a, &mut b, 50_000, 200);
        let error = a.coordinate().unwrap().error_permille;
        assert!(error < 500, "error stayed at {error} per mille");
    }

    #[test]
    fn a_zero_round_trip_is_refused() {
        let mut node = Vivaldi::new();
        assert!(!node.observe(&NetworkCoordinate::default(), 0));
        assert_eq!(node.observations(), 0);
    }

    #[test]
    fn an_out_of_range_coordinate_is_refused_without_moving_us() {
        let (mut a, mut b) = (Vivaldi::seeded(b"node-a"), Vivaldi::seeded(b"node-b"));
        settle(&mut a, &mut b, 40_000, 200);
        let before = a.coordinate().unwrap();
        let hostile = NetworkCoordinate {
            vector_micros: [i64::MAX, 0, 0],
            height_micros: 0,
            error_permille: 0,
        };
        assert!(!is_plausible(&hostile));
        assert!(!a.observe(&hostile, 40_000));
        assert_eq!(a.coordinate().unwrap(), before);
    }

    #[test]
    fn no_single_observation_can_teleport_a_settled_node() {
        let (mut a, mut b) = (Vivaldi::seeded(b"node-a"), Vivaldi::seeded(b"node-b"));
        settle(&mut a, &mut b, 30_000, 200);
        let before = a.coordinate().unwrap();
        let liar = NetworkCoordinate {
            vector_micros: [MAX_COORDINATE_MICROS, 0, 0],
            height_micros: 0,
            error_permille: 0,
        };
        assert!(is_plausible(&liar));
        assert!(a.observe(&liar, 59_000_000));
        let moved = distance(&before, &a.coordinate().unwrap());
        assert!(
            moved <= MAX_STEP_MICROS + 1.0,
            "moved {moved}us in one step"
        );
    }

    #[test]
    fn heights_are_charged_at_both_ends_of_a_path() {
        let flat = NetworkCoordinate::default();
        let raised = NetworkCoordinate {
            vector_micros: [0; COORDINATE_DIMENSIONS],
            height_micros: 5_000,
            error_permille: 100,
        };
        assert_eq!(predicted_rtt_micros(&flat, &raised), 5_000);
        assert_eq!(predicted_rtt_micros(&raised, &raised), 10_000);
    }

    /// A restart must not throw away a position that took real probes to fit.
    #[test]
    fn a_fitted_position_survives_a_restart() {
        let home = std::env::temp_dir().join(format!("mesh-px-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let mut node = Vivaldi::seeded(b"node");
        let peer = NetworkCoordinate {
            vector_micros: [30_000, 0, 0],
            height_micros: 2_000,
            error_permille: 100,
        };
        for _ in 0..20 {
            node.observe(&peer, 45_000);
        }
        node.save(&home).unwrap();

        let restored = Vivaldi::load_or_seeded(&home, b"node");
        assert_eq!(restored.observations(), node.observations());
        assert_eq!(
            restored.coordinate(),
            node.coordinate(),
            "a restart must resume from the fitted position, not re-derive one"
        );
        std::fs::remove_dir_all(&home).unwrap();
    }

    /// Losing the cache costs time, not correctness, so it must not be fatal.
    #[test]
    fn a_corrupt_or_absent_position_starts_over_instead_of_failing() {
        let home = std::env::temp_dir().join(format!("mesh-px-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);

        let fresh = Vivaldi::load_or_seeded(&home, b"node");
        assert_eq!(fresh.observations(), 0, "no file means a fresh position");

        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(Vivaldi::path(&home), "{ not json").unwrap();
        assert_eq!(Vivaldi::load_or_seeded(&home, b"node").observations(), 0);

        // A syntactically valid file whose numbers are not a position.
        std::fs::write(
            Vivaldi::path(&home),
            r#"{"vector_micros":[1e300,0,0],"height_micros":0.0,"error":0.5,"observations":9}"#,
        )
        .unwrap();
        assert_eq!(
            Vivaldi::load_or_seeded(&home, b"node").observations(),
            0,
            "an out-of-range stored position must be discarded, not trusted"
        );
        std::fs::remove_dir_all(&home).unwrap();
    }

    /// Straight-line distance between two wire coordinates, heights ignored.
    fn distance(a: &NetworkCoordinate, b: &NetworkCoordinate) -> f64 {
        let mut sum = 0.0f64;
        for axis in 0..COORDINATE_DIMENSIONS {
            let delta = (a.vector_micros[axis] - b.vector_micros[axis]) as f64;
            sum += delta * delta;
        }
        sum.sqrt()
    }
}
