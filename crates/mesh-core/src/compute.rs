use mesh_protocol::{WorkResult, WorkSpec};
use std::time::Instant;

/// Execute one declarative MESH workload.
///
/// The MVP deliberately supports a small allow-list instead of arbitrary binaries.
pub fn execute_work(work: &WorkSpec) -> WorkResult {
    match work {
        WorkSpec::PrimeCount { start, end } => {
            let started = Instant::now();
            let count = count_primes(*start, *end);
            WorkResult::PrimeCount {
                count,
                duration_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            }
        }
    }
}

/// Verify a result by recomputing the deterministic workload.
///
/// This is intentionally expensive but simple for the MVP. Production MESH should
/// use workload-specific verification, probabilistic spot checks, and redundant
/// execution where appropriate.
pub fn verify_work(work: &WorkSpec, result: &WorkResult) -> bool {
    match (work, result) {
        (WorkSpec::PrimeCount { start, end }, WorkResult::PrimeCount { count, .. }) => {
            count_primes(*start, *end) == *count
        }
    }
}

pub fn count_primes(start: u64, end: u64) -> u64 {
    (start..end).filter(|n| is_prime(*n)).count() as u64
}

pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }

    let mut i = 5_u64;
    while i <= n / i {
        if n.is_multiple_of(i) || n.is_multiple_of(i + 2) {
            return false;
        }
        i += 6;
    }
    true
}

/// MVP pricing: 1 milli-compute-unit (mCU) per 50 candidate integers.
/// 1,000 mCU = 1 CU.
///
/// Pricing is based on deterministic work size, not elapsed wall-clock time,
/// which prevents a slow/misconfigured provider from earning extra credit merely
/// by taking longer.
pub fn work_cost_mcu(work: &WorkSpec) -> i64 {
    match work {
        WorkSpec::PrimeCount { start, end } => {
            let width = end.saturating_sub(*start);
            ((width.saturating_add(49)) / 50)
                .max(1)
                .min(i64::MAX as u64) as i64
        }
    }
}

pub fn split_work(work: &WorkSpec, shards: u32) -> Vec<WorkSpec> {
    let shards = shards.max(1);
    match work {
        WorkSpec::PrimeCount { start, end } => {
            let width = end.saturating_sub(*start);
            let actual_shards = (shards as u64).min(width.max(1)) as u32;
            let base = width / actual_shards as u64;
            let remainder = width % actual_shards as u64;

            let mut out = Vec::with_capacity(actual_shards as usize);
            let mut cursor = *start;
            for i in 0..actual_shards {
                let extra = if (i as u64) < remainder { 1 } else { 0 };
                let next = cursor + base + extra;
                out.push(WorkSpec::PrimeCount {
                    start: cursor,
                    end: next,
                });
                cursor = next;
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prime_count_is_correct() {
        assert_eq!(count_primes(2, 20), 8);
    }

    #[test]
    fn split_preserves_range() {
        let work = WorkSpec::PrimeCount {
            start: 10,
            end: 110,
        };
        let parts = split_work(&work, 3);
        assert_eq!(parts.len(), 3);
        match (&parts[0], &parts[2]) {
            (WorkSpec::PrimeCount { start, .. }, WorkSpec::PrimeCount { end, .. }) => {
                assert_eq!(*start, 10);
                assert_eq!(*end, 110);
            }
        }
    }
}
