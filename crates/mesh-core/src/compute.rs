use crate::matrix;
use crate::verify::{self, BUCKETS, Verdict};
use mesh_protocol::{WorkResult, WorkSpec};
use std::time::Instant;

/// Multiply-adds that one milli-compute-unit buys.
///
/// Every workload prices against this one number, so a mCU means the same
/// amount of machine work whichever workload earned it. Cost is always derived
/// from the *spec*, never measured, so anyone can recompute a price and check
/// it -- which is what keeps the coordinator from being the authority on CU.
pub const REFERENCE_OPS_PER_MCU: u64 = 8_192;

/// Execute one declarative MESH workload.
///
/// A small allow-list, never arbitrary binaries: the code that runs on a
/// contributor's machine is always this code.
pub fn execute_work(work: &WorkSpec) -> WorkResult {
    match work {
        WorkSpec::PrimeCount { start, end } => prime_count(*start, *end),
        WorkSpec::MatrixMultiply {
            seed_a,
            seed_b,
            dim,
            row_start,
            row_end,
        } => matrix_multiply(*seed_a, *seed_b, *dim, *row_start, *row_end),
    }
}

/// Count primes, recording per-bucket sub-counts as we go.
///
/// The buckets are the whole point: they let a verifier recompute a few
/// sixty-fourths of the range and still catch a fabricated total, because the
/// buckets must both sum to the total and individually survive a spot check.
fn prime_count(start: u64, end: u64) -> WorkResult {
    let started = Instant::now();
    let mut bucket_counts = Vec::with_capacity(BUCKETS as usize);
    for index in 0..BUCKETS {
        let (lo, hi) = verify::bucket_bounds(start, end, index, BUCKETS);
        bucket_counts.push(count_primes(lo, hi));
    }
    WorkResult::PrimeCount {
        count: bucket_counts.iter().sum(),
        bucket_counts,
        duration_ms: elapsed_ms(started),
    }
}

fn matrix_multiply(seed_a: u64, seed_b: u64, dim: u32, row_start: u32, row_end: u32) -> WorkResult {
    let started = Instant::now();
    let rows = matrix::multiply_rows(seed_a, seed_b, dim, row_start, row_end);
    WorkResult::MatrixMultiply {
        rows,
        duration_ms: elapsed_ms(started),
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u64::MAX as u128) as u64
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

/// Price a workload from its declaration alone.
///
/// Never from elapsed time: a slow machine must not earn more for identical
/// work, and nobody can prove how long they spent. Anyone holding the spec can
/// recompute this number, which is why the coordinator is not the authority
/// on what a shard is worth.
///
/// Both variants are calibrated against one reference unit: 1 mCU is either 50
/// candidate integers or `REFERENCE_OPS_PER_MCU` field multiply-adds, which
/// cost about the same on the reference machine. Every workload added later
/// must declare its price the same way -- closed form, from the spec.
pub fn work_cost_mcu(work: &WorkSpec) -> i64 {
    let mcu = match work {
        WorkSpec::PrimeCount { start, end } => end.saturating_sub(*start).div_ceil(50),
        other => verify::compute_ops(other).div_ceil(REFERENCE_OPS_PER_MCU),
    };
    mcu.max(1).min(i64::MAX as u64) as i64
}

pub fn split_work(work: &WorkSpec, shards: u32) -> Vec<WorkSpec> {
    let shards = shards.max(1);
    match work {
        WorkSpec::PrimeCount { start, end } => split_primes(*start, *end, shards),
        WorkSpec::MatrixMultiply {
            seed_a,
            seed_b,
            dim,
            row_start,
            row_end,
        } => split_rows(*seed_a, *seed_b, *dim, *row_start, *row_end, shards),
    }
}

fn split_primes(start: u64, end: u64, shards: u32) -> Vec<WorkSpec> {
    let width = end.saturating_sub(start);
    let actual = (shards as u64).min(width.max(1)) as u32;
    let base = width / actual as u64;
    let remainder = width % actual as u64;
    let mut out = Vec::with_capacity(actual as usize);
    let mut cursor = start;
    for i in 0..actual {
        let extra = u64::from((i as u64) < remainder);
        let next = cursor + base + extra;
        out.push(WorkSpec::PrimeCount {
            start: cursor,
            end: next,
        });
        cursor = next;
    }
    out
}

fn split_rows(
    seed_a: u64,
    seed_b: u64,
    dim: u32,
    row_start: u32,
    row_end: u32,
    shards: u32,
) -> Vec<WorkSpec> {
    let span = row_end.saturating_sub(row_start);
    let actual = shards.min(span.max(1));
    let base = span / actual;
    let remainder = span % actual;
    let mut out = Vec::with_capacity(actual as usize);
    let mut cursor = row_start;
    for i in 0..actual {
        let next = cursor + base + u32::from(i < remainder);
        out.push(WorkSpec::MatrixMultiply {
            seed_a,
            seed_b,
            dim,
            row_start: cursor,
            row_end: next,
        });
        cursor = next;
    }
    out
}

/// Full recomputation.
///
/// This is the adjudicator of last resort, not the settlement path: it costs
/// the verifier exactly what it cost the worker, so calling it on every result
/// makes the network burn more compute than it delivers. Settlement goes
/// through `verify::witness_check`; this runs only to break a tie between two
/// replicas that disagree.
pub fn verify_work(work: &WorkSpec, result: &WorkResult) -> bool {
    matches!(verify::adjudicate(work, result), Verdict::Accepted(_))
}

/// Whether two results carry the same answer.
///
/// Deliberately blind to `duration_ms`: two honest machines never agree on how
/// long a shard took, so comparing timings would make every replica pair look
/// like a disagreement.
pub fn results_agree(left: &WorkResult, right: &WorkResult) -> bool {
    match (left, right) {
        (
            WorkResult::PrimeCount {
                count: a,
                bucket_counts: ab,
                ..
            },
            WorkResult::PrimeCount {
                count: b,
                bucket_counts: bb,
                ..
            },
        ) => a == b && ab == bb,
        (
            WorkResult::MatrixMultiply { rows: a, .. },
            WorkResult::MatrixMultiply { rows: b, .. },
        ) => a == b,
        _ => false,
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
            other => panic!("prime shards must stay prime shards, got {other:?}"),
        }
    }
}
