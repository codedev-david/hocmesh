//! Turning machine numbers into the words on the dashboard.
//!
//! These live in Rust rather than in the page's JavaScript on purpose. A
//! dashboard that renders `1000` where it means `1.000 CU`, or that rounds a
//! balance the operator is owed, is wrong in a way that matters -- and
//! formatting done in the view is formatting nobody tests. Everything here is
//! a pure function over a number, so every rounding rule below is nailed down
//! by a test rather than by whatever `toFixed` happened to do.

/// CU as the operator reads it.
///
/// The ledger counts in milli-CU because integers are exact and floats are
/// not; the operator reads CU. Three decimals is the whole of the unit, so
/// this is a reformat rather than a rounding -- no earned milli-CU ever
/// disappears into a display.
pub fn cu(mcu: i64) -> String {
    let sign = if mcu < 0 { "-" } else { "" };
    let magnitude = mcu.unsigned_abs();
    format!("{sign}{}.{:03}", magnitude / 1000, magnitude % 1000)
}

/// CU with an explicit sign, for a column of movements.
///
/// A ledger row's whole point is which way the CU went, so `+` is shown as
/// deliberately as `-`. Zero carries no sign: nothing moved.
pub fn signed_cu(mcu: i64) -> String {
    match mcu {
        0 => cu(0),
        m if m > 0 => format!("+{}", cu(m)),
        m => cu(m),
    }
}

/// Bytes at human scale.
///
/// Binary units, because that is what the machine reports and what every other
/// tool on the operator's screen will say about the same memory.
pub fn bytes(value: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut scaled = value as f64;
    let mut unit = 0;
    while scaled >= 1024.0 && unit + 1 < UNITS.len() {
        scaled /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{value} B")
    } else {
        format!("{scaled:.1} {}", UNITS[unit])
    }
}

/// An elapsed span, coarse on purpose.
///
/// Uptime is read to answer "has this been up a while?", so the largest two
/// units are the whole of the answer; seconds of precision on a four-day
/// uptime is noise.
pub fn duration(seconds: i64) -> String {
    if seconds <= 0 {
        return "just now".into();
    }
    let days = seconds / 86_400;
    let hours = (seconds % 86_400) / 3_600;
    let minutes = (seconds % 3_600) / 60;
    let secs = seconds % 60;
    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {minutes}m")
    } else if minutes > 0 {
        format!("{minutes}m {secs}s")
    } else {
        format!("{secs}s")
    }
}

/// How long ago a unix timestamp was, given the clock now.
///
/// `now` is passed rather than read so this stays a pure function and the
/// tests do not have to sleep. A timestamp in the future reads as "just now"
/// rather than as a negative age: a slightly fast peer clock is not worth a
/// nonsense string.
pub fn since(unix: i64, now: i64) -> String {
    if unix <= 0 {
        return "never".into();
    }
    let elapsed = now - unix;
    if elapsed <= 0 {
        return "just now".into();
    }
    format!("{} ago", duration(elapsed))
}

/// A percentage of a whole, guarding the zero-sized whole.
///
/// A machine reporting no memory at all would otherwise divide by zero here;
/// showing 0% is the honest answer to "how much of nothing is lent".
pub fn percent_of(part: u64, whole: u64) -> u32 {
    if whole == 0 {
        return 0;
    }
    ((part as u128 * 100) / whole as u128).min(100) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn milli_cu_reads_as_cu_without_losing_a_thousandth() {
        assert_eq!(cu(0), "0.000");
        assert_eq!(cu(1), "0.001");
        assert_eq!(cu(999), "0.999");
        assert_eq!(cu(1_000), "1.000");
        assert_eq!(cu(1_234_567), "1234.567");
    }

    #[test]
    fn a_debt_keeps_its_sign_and_its_thousandths() {
        assert_eq!(cu(-1), "-0.001");
        assert_eq!(cu(-1_500), "-1.500");
        // The obvious implementation -- dividing a negative by 1000 -- gives
        // "-1.-500". Taking the magnitude first is what stops that.
        assert!(!cu(-1_500).contains(".-"));
    }

    #[test]
    fn the_extreme_of_the_range_does_not_overflow_into_nonsense() {
        // `i64::MIN` has no positive counterpart, so a naive `-mcu` panics in
        // debug and wraps in release. `unsigned_abs` is what makes this safe.
        assert_eq!(cu(i64::MIN), "-9223372036854775.808");
        assert_eq!(cu(i64::MAX), "9223372036854775.807");
    }

    #[test]
    fn a_ledger_column_shows_which_way_the_cu_went() {
        assert_eq!(signed_cu(2_500), "+2.500");
        assert_eq!(signed_cu(-2_500), "-2.500");
        assert_eq!(
            signed_cu(0),
            "0.000",
            "nothing moved, so no sign is claimed"
        );
    }

    #[test]
    fn memory_is_reported_in_the_units_the_rest_of_the_machine_uses() {
        assert_eq!(bytes(0), "0 B");
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(1024), "1.0 KiB");
        assert_eq!(bytes(8 * 1024 * 1024 * 1024), "8.0 GiB");
        assert_eq!(bytes(1536 * 1024 * 1024), "1.5 GiB");
    }

    #[test]
    fn a_machine_larger_than_the_unit_table_still_reads_sensibly() {
        // Stopping at TiB rather than running off the end of the array is the
        // property; a petabyte host reads as four figures of TiB.
        assert!(bytes(u64::MAX).ends_with(" TiB"));
    }

    #[test]
    fn uptime_shows_the_two_units_that_answer_the_question() {
        assert_eq!(duration(0), "just now");
        assert_eq!(duration(45), "45s");
        assert_eq!(duration(90), "1m 30s");
        assert_eq!(duration(3_700), "1h 1m");
        assert_eq!(duration(90_000), "1d 1h");
    }

    #[test]
    fn a_contact_that_never_happened_says_so_rather_than_lying_about_1970() {
        assert_eq!(since(0, 1_000_000), "never");
        assert_eq!(since(-5, 1_000_000), "never");
    }

    #[test]
    fn a_peer_clock_running_fast_does_not_produce_a_negative_age() {
        assert_eq!(since(1_000_100, 1_000_000), "just now");
        assert_eq!(since(1_000_000, 1_000_000), "just now");
    }

    #[test]
    fn an_age_reads_in_the_past_tense() {
        assert_eq!(since(1_000_000, 1_000_090), "1m 30s ago");
    }

    #[test]
    fn a_share_of_nothing_is_zero_rather_than_a_division_by_zero() {
        assert_eq!(percent_of(0, 0), 0);
        assert_eq!(percent_of(5, 0), 0);
    }

    #[test]
    fn a_share_is_clamped_to_the_whole_it_is_a_share_of() {
        assert_eq!(percent_of(50, 100), 50);
        assert_eq!(percent_of(100, 100), 100);
        // A lent slice can only exceed the machine through a bug elsewhere,
        // and a dashboard reading 300% would hide that rather than show it.
        assert_eq!(percent_of(300, 100), 100);
    }

    #[test]
    fn a_share_of_a_very_large_machine_does_not_overflow_the_multiply() {
        // `part * 100` in u64 overflows above ~1.8e17 bytes; the u128 widening
        // is what keeps a large host's percentage correct rather than tiny.
        assert_eq!(percent_of(u64::MAX / 2, u64::MAX), 49);
    }
}
