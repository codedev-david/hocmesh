//! Measuring how fast this machine can actually push bytes to another one.
//!
//! Placement needs to know a node's uplink, and until now every node reported
//! the same hardcoded 100 Mbps. A number every machine reports identically
//! cannot separate any machine from any other, so a policy that gated on it
//! would have been a policy that gated on nothing -- and would have looked, in
//! every test and every dashboard, exactly like a policy that worked.
//!
//! What is measured here is a real transfer to a real peer: the chunks this
//! node serves when it seeds a model. Nothing extra is sent. That has two
//! consequences worth stating plainly.
//!
//! The first is that a machine which has never served anything has no
//! measurement, and says so. `kbps` returns `None` rather than a guess, and a
//! role that needs a fast uplink is refused to it. Unmeasured is not slow and
//! it is certainly not fast; it is unknown, and the only safe reading of
//! unknown is "not yet".
//!
//! The second is that this is a *bootstrapping* path rather than a barrier.
//! Seeding is open to every machine on any connection. A node earns a
//! measurement by doing the work anyone may do, and the measurement is what
//! qualifies it -- or does not -- for the work that needs a fast link.
//!
//! Small transfers are discarded rather than averaged in. A 4 KiB response
//! completes in about one round trip whatever the link underneath it, so
//! timing one measures the distance to the peer and reports it as bandwidth --
//! which would make a nearby dial-up connection look like a fast one.
//!
//! One known bias, stated rather than hidden: the first bytes of a transfer go
//! into the kernel's send buffer at memory speed, so a timed push gets a head
//! start it did not earn and reads slightly high. The bias shrinks as the
//! transfer grows past the buffer, which is why only sizeable transfers count
//! and why the caller should time the drain of a body rather than the handler
//! that produced it. A threshold set at this figure should carry margin for it.

use std::{
    sync::Mutex,
    time::{Duration, Instant},
};

/// What to assume about a link that nothing has measured yet.
///
/// Ranking has to put *some* number on the cost of moving a model to a machine
/// that has never served one, and refusing to rank it at all would be a
/// deadlock rather than a safeguard: a node earns its measurement by serving,
/// it can only serve a model it was sent, and it is only sent one if it ranked
/// well enough to be picked. So ranking assumes an ordinary broadband link and
/// gets on with it.
///
/// This is deliberately *not* what the role gate uses. Ranking may guess,
/// because a bad guess costs one slow transfer; the prefill gate may not,
/// because a bad guess there costs every request routed through that stage.
pub const ASSUMED_KBPS: u64 = 100_000;

/// Transfers below this are timing the round trip, not the link.
const MEANINGFUL_BYTES: u64 = 256 * 1024;

/// How many recent samples decide the answer.
const WINDOW: usize = 16;

/// A transfer that took no measurable time did not measure anything.
const MIN_ELAPSED: Duration = Duration::from_micros(100);

/// Throughput observed while serving real bytes to real peers.
///
/// Cheap to share: one lock held for the length of a push, and readers take
/// the same lock for as long as it takes to copy at most [`WINDOW`] numbers.
#[derive(Debug, Default)]
pub struct UplinkMeter {
    samples: Mutex<Vec<u64>>,
}

impl UplinkMeter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one completed transfer.
    ///
    /// Transfers too small or too quick to say anything about the link are
    /// dropped, so a burst of them cannot flood out the real measurements.
    pub fn record(&self, bytes: u64, elapsed: Duration) {
        if bytes < MEANINGFUL_BYTES || elapsed < MIN_ELAPSED {
            return;
        }
        let kbps = (bytes as f64 * 8.0 / 1000.0 / elapsed.as_secs_f64()) as u64;
        if kbps == 0 {
            return;
        }
        let Ok(mut samples) = self.samples.lock() else {
            return;
        };
        samples.push(kbps);
        let len = samples.len();
        if len > WINDOW {
            samples.drain(..len - WINDOW);
        }
    }

    /// The measured uplink in kbit/s, or `None` if nothing has been measured.
    ///
    /// The median of the window rather than the maximum. A node is asked for
    /// this number so that other people's work can be placed on it, so the
    /// question is what it sustains, not what it once managed -- and a
    /// best-of-window would reward a machine for one lucky transfer.
    #[must_use]
    pub fn kbps(&self) -> Option<u64> {
        let samples = self.samples.lock().ok()?;
        if samples.is_empty() {
            return None;
        }
        let mut sorted = samples.clone();
        sorted.sort_unstable();
        Some(sorted[sorted.len() / 2])
    }

    /// Time `f`, record the transfer, and hand back what it returned.
    pub fn time<T>(&self, bytes: u64, f: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        let value = f();
        self.record(bytes, started.elapsed());
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_measured_reports_nothing_rather_than_a_default() {
        // The whole point. A meter that answered "100 Mbps" here would put this
        // module back where the hardcoded constant was.
        assert_eq!(UplinkMeter::new().kbps(), None);
    }

    #[test]
    fn a_transfer_too_small_to_time_is_not_a_measurement() {
        let meter = UplinkMeter::new();
        meter.record(4096, Duration::from_millis(40));
        assert_eq!(
            meter.kbps(),
            None,
            "a 4 KiB response timed at one round trip was read as a bandwidth \
             measurement, which makes a nearby slow link look fast"
        );
    }

    #[test]
    fn a_transfer_that_took_no_time_is_not_a_measurement() {
        let meter = UplinkMeter::new();
        // A cache hit served from memory in under the clock's resolution
        // divides by nearly zero and reports an impossible number.
        meter.record(8 << 20, Duration::from_nanos(1));
        assert_eq!(meter.kbps(), None);
    }

    #[test]
    fn megabytes_per_second_becomes_kilobits_per_second() {
        let meter = UplinkMeter::new();
        // 8 MiB in one second is 8 MiB * 8 bits / 1000 = 67_108 kbit/s.
        meter.record(8 << 20, Duration::from_secs(1));
        assert_eq!(meter.kbps(), Some(67_108));
    }

    #[test]
    fn one_lucky_transfer_does_not_set_the_reported_speed() {
        let meter = UplinkMeter::new();
        for _ in 0..4 {
            meter.record(1 << 20, Duration::from_secs(1));
        }
        meter.record(1 << 20, Duration::from_millis(10));
        let slow = 1048576 * 8 / 1000; // 1 MiB in a second, in kbit/s.
        assert_eq!(
            meter.kbps(),
            Some(slow),
            "a single fast transfer moved the reported uplink, so a node could \
             advertise a link it does not sustain"
        );
    }

    #[test]
    fn only_the_recent_window_counts() {
        let meter = UplinkMeter::new();
        for _ in 0..WINDOW {
            meter.record(1 << 20, Duration::from_secs(1));
        }
        for _ in 0..WINDOW {
            meter.record(8 << 20, Duration::from_secs(1));
        }
        assert_eq!(
            meter.kbps(),
            Some(67_108),
            "a link that has since got faster is still being reported at its \
             old speed"
        );
    }
}
