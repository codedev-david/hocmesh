//! Audit rate, and the economics that keep cheating unprofitable.
//!
//! Verification is only affordable if it applies to a *fraction* of results.
//! That is safe exactly when the expected value of cheating stays negative,
//! which is a statement about three numbers: how often work is audited, what a
//! shard pays, and how much a caught node loses.
//!
//! The banked balance is the collateral. That only works because CU cannot be
//! purchased: a balance is proof of work already performed, so destroying it
//! costs the cheater something no amount of money can replace.

use crate::verify::AuditNonce;

/// Every result from an unproven node is audited.
pub const INITIAL_AUDIT_RATE: f64 = 1.0;

/// The lowest rate any node earns, however long its clean record.
pub const FLOOR_AUDIT_RATE: f64 = 0.05;

/// Clean results needed to close half the gap to the floor.
pub const STREAK_HALF_LIFE: f64 = 12.0;

/// One node's standing, as the coordinator and every validator see it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Reputation {
    pub accepted: u64,
    pub rejected: u64,
    pub streak: u32,
}

impl Reputation {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decays from every-result toward the floor as the clean streak grows.
    pub fn audit_rate(&self) -> f64 {
        let decay = 0.5_f64.powf(f64::from(self.streak) / STREAK_HALF_LIFE);
        FLOOR_AUDIT_RATE + (INITIAL_AUDIT_RATE - FLOOR_AUDIT_RATE) * decay
    }

    pub fn record_accepted(&mut self) {
        self.accepted = self.accepted.saturating_add(1);
        self.streak = self.streak.saturating_add(1);
    }

    /// A rejection resets the streak: trust is re-earned from nothing.
    pub fn record_rejected(&mut self) {
        self.rejected = self.rejected.saturating_add(1);
        self.streak = 0;
    }

    /// Whether this particular result draws an audit. The nonce is the
    /// coordinator's, drawn after the result was signed, so a worker cannot
    /// tell in advance which of its submissions will be checked.
    pub fn should_audit(&self, nonce: AuditNonce) -> bool {
        nonce.unit_interval() < self.audit_rate()
    }
}

/// Expected value, in mCU, of submitting one fabricated result.
///
/// `detection` is the chance the fabrication is both audited and caught.
pub fn cheating_expected_value(reward_mcu: i64, slash_mcu: i64, detection: f64) -> f64 {
    (1.0 - detection) * reward_mcu as f64 - detection * slash_mcu as f64
}

/// The smallest slash that makes cheating EV-negative at this detection rate.
///
/// Solves `(1-d)*reward - d*slash < 0` for `slash`.
pub fn minimum_slash_mcu(reward_mcu: i64, detection: f64) -> i64 {
    if detection <= 0.0 {
        return i64::MAX;
    }
    (((1.0 - detection) / detection) * reward_mcu as f64).ceil() as i64 + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A node with no history has earned nothing, so nothing it sends is taken
    /// on trust. This is the property that makes a fresh Sybil worthless.
    #[test]
    fn an_unproven_node_has_every_result_audited() {
        assert_eq!(Reputation::new().audit_rate(), INITIAL_AUDIT_RATE);
    }

    /// Audit cost has to fall as trust accrues, or verification never gets
    /// cheap. It must also stop falling, or a long-lived node buys immunity.
    #[test]
    fn a_clean_streak_lowers_the_audit_rate_but_never_past_the_floor() {
        let mut node = Reputation::new();
        let mut previous = node.audit_rate();
        for _ in 0..500 {
            node.record_accepted();
            let rate = node.audit_rate();
            assert!(
                rate <= previous,
                "audit rate must not rise on clean work"
            );
            assert!(
                rate >= FLOOR_AUDIT_RATE,
                "no streak may buy a node out of auditing"
            );
            previous = rate;
        }
        assert!(
            (previous - FLOOR_AUDIT_RATE).abs() < 1e-6,
            "a long clean record converges on the floor"
        );
    }

    /// Trust is re-earned from nothing, so one caught fabrication costs a node
    /// every discount its whole clean history bought.
    #[test]
    fn a_rejection_puts_a_node_back_under_full_audit() {
        let mut node = Reputation::new();
        for _ in 0..200 {
            node.record_accepted();
        }
        assert!(node.audit_rate() < 0.06);
        node.record_rejected();
        assert_eq!(node.audit_rate(), INITIAL_AUDIT_RATE);
        assert_eq!(node.streak, 0);
    }

    /// The whole scheme rests on this inequality. Below the computed slash a
    /// rational node should cheat; at or above it, cheating loses money.
    #[test]
    fn the_minimum_slash_is_exactly_where_cheating_stops_paying() {
        let reward = 1_000i64;
        for detection in [0.05, 0.2, 0.5, 0.95] {
            let floor = minimum_slash_mcu(reward, detection);
            assert!(cheating_expected_value(reward, floor, detection) < 0.0);
            assert!(cheating_expected_value(reward, floor - 2, detection) > 0.0);
        }
    }

    /// `audit_rate` is a promise about long-run frequency; if the nonce draw
    /// did not honour it the published economics would describe nothing real.
    #[test]
    fn audits_actually_fire_at_the_declared_rate() {
        for streak in [0u32, 6, 12, 24, 96] {
            let node = Reputation {
                accepted: streak as u64,
                rejected: 0,
                streak,
            };
            let fired = (0..40_000u64)
                .filter(|&i| node.should_audit(AuditNonce::draw(i.wrapping_mul(0x9E37_79B9))))
                .count();
            let measured = fired as f64 / 40_000.0;
            assert!(
                (measured - node.audit_rate()).abs() < 0.01,
                "streak {streak}: {measured}"
            );
        }
    }

    /// The end-to-end economic claim: even at the cheapest audit rate MESH
    /// offers, one fabricated shard is still a losing trade.
    #[test]
    fn cheating_loses_money_at_the_cheapest_audit_rate_a_node_can_earn() {
        let veteran = Reputation {
            accepted: 5_000,
            rejected: 0,
            streak: 5_000,
        };
        let caught_if_audited = 0.90;
        let detection = veteran.audit_rate() * caught_if_audited;
        let reward = 20i64;
        let slash = minimum_slash_mcu(reward, detection);
        assert!(
            slash < 1_000,
            "a node can plausibly bank {slash} mCU of collateral"
        );
        assert!(cheating_expected_value(reward, slash, detection) < 0.0);
    }
}
