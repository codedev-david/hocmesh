use crate::matrix;
use crate::verify::{self, BUCKETS, Verdict};
use hocmesh_protocol::{WorkResult, WorkSpec};
use std::time::Instant;

/// Multiply-adds that one milli-compute-unit buys.
///
/// Every workload prices against this one number, so a mCU means the same
/// amount of machine work whichever workload earned it. Cost is always derived
/// from the *spec*, never measured, so anyone can recompute a price and check
/// it -- which is what keeps the coordinator from being the authority on CU.
pub const REFERENCE_OPS_PER_MCU: u64 = 8_192;

/// Steps after which a Collatz trajectory is abandoned.
///
/// Nothing below 2^68 is known to need more than about two thousand, so this
/// never fires on a real shard. It exists so that a start that did diverge --
/// which would be a mathematical result, not a bug -- stops a contributor's
/// machine instead of hanging it forever.
pub const COLLATZ_STEP_CEILING: u32 = 100_000;

/// Execute one declarative hocMESH workload.
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
        WorkSpec::CollatzPeak { start, end } => collatz_peak(*start, *end),
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

/// Find the longest Collatz trajectory in `[start, end)`, bucket by bucket.
///
/// Same shape as the prime count and for the same reason: an auditor that
/// redraws a bucket recomputes a sixty-fourth of the shard and compares one
/// number, and a shard whose buckets were invented cannot survive it.
fn collatz_peak(start: u64, end: u64) -> WorkResult {
    let started = Instant::now();
    let mut bucket_peaks = Vec::with_capacity(BUCKETS as usize);
    let mut bucket_seeds = Vec::with_capacity(BUCKETS as usize);
    for index in 0..BUCKETS {
        let (lo, hi) = verify::bucket_bounds(start, end, index, BUCKETS);
        let (steps, seed) = peak_trajectory(lo, hi);
        bucket_peaks.push(steps);
        bucket_seeds.push(seed);
    }
    // The shard's answer is whichever bucket peaked highest, and the smallest
    // seed that reached it, so two shards that tie resolve the same way
    // everywhere.
    let (peak_steps, peak_seed) = combine_peaks(&bucket_peaks, &bucket_seeds);
    WorkResult::CollatzPeak {
        peak_steps,
        peak_seed,
        bucket_peaks,
        bucket_seeds,
        duration_ms: elapsed_ms(started),
    }
}

/// The highest bucket peak, and the smallest seed among the buckets that tie.
///
/// Ties have to break the same way on every machine or two honest nodes
/// disagree about an answer they both computed correctly.
pub fn combine_peaks(bucket_peaks: &[u32], bucket_seeds: &[u64]) -> (u32, u64) {
    let mut best = (0u32, 0u64);
    for (index, &steps) in bucket_peaks.iter().enumerate() {
        let seed = bucket_seeds.get(index).copied().unwrap_or(0);
        if steps > best.0 || (steps == best.0 && steps > 0 && seed < best.1) {
            best = (steps, seed);
        }
    }
    best
}

/// The longest trajectory in `[lo, hi)`, and the smallest seed that reaches it.
pub fn peak_trajectory(lo: u64, hi: u64) -> (u32, u64) {
    let mut best = (0u32, 0u64);
    for n in lo..hi {
        let steps = collatz_steps(n);
        if steps > best.0 {
            best = (steps, n);
        }
    }
    best
}

/// Steps to reach 1, counting each halving and each `3n + 1` as one.
///
/// `u128` because `3n + 1` overflows `u64` for large starts, and the whole
/// point of an integer workload is that nothing depends on which machine ran
/// it. Zero has no trajectory and is reported as zero steps.
pub fn collatz_steps(n: u64) -> u32 {
    if n == 0 {
        return 0;
    }
    let mut value = u128::from(n);
    let mut steps = 0u32;
    // No start below 2^68 is known to escape, and the longest of them stops in
    // under 2000 steps. The ceiling is only here so a hypothetical divergence
    // cannot hang a contributor's machine forever.
    while value != 1 && steps < COLLATZ_STEP_CEILING {
        value = if value % 2 == 0 {
            value / 2
        } else {
            value * 3 + 1
        };
        steps += 1;
    }
    steps
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
/// There is one calibration and every workload goes through it: ops on the
/// reference machine, divided by `REFERENCE_OPS_PER_MCU`. No workload gets its
/// own rate, because a mCU earned counting primes has to buy the same machine
/// work as a mCU earned multiplying matrices -- otherwise the cheapest workload
/// to run becomes the cheapest way to mint, and the unit stops meaning
/// anything. Every workload added later must price the same way: closed form,
/// from the spec, against this one constant.
pub fn work_cost_mcu(work: &WorkSpec) -> i64 {
    let mcu = verify::compute_ops(work).div_ceil(REFERENCE_OPS_PER_MCU);
    mcu.max(1).min(i64::MAX as u64) as i64
}

/// Bytes of prompt text that count as one token.
///
/// A tokeniser lives inside the model, so a coordinator cannot run one and a
/// validator cannot either. Four bytes a token is the standard rule of thumb
/// for English, and using it keeps the price a pure function of the request.
pub const BYTES_PER_TOKEN: u64 = 4;

/// Price an inference batch from the request alone.
///
/// A forward pass costs about two operations per parameter per token, and the
/// job asks for `max_tokens` on top of whatever the prompt already holds. That
/// is the whole formula, and every term of it is in the signed request, so the
/// price does not depend on which machine answers or how long it took.
///
/// Priced against `REFERENCE_OPS_PER_MCU` like every other workload, which is
/// what lets a CPU that counted primes pay for a GPU that ran a model.
pub fn inference_cost_mcu(prompt_bytes: &[u64], max_tokens: u32, parameter_count: u64) -> i64 {
    let tokens = prompt_bytes.iter().fold(0_u64, |sum, bytes| {
        sum.saturating_add(bytes.div_ceil(BYTES_PER_TOKEN))
            .saturating_add(u64::from(max_tokens))
    });
    let ops = tokens.saturating_mul(parameter_count).saturating_mul(2);
    ops.div_ceil(REFERENCE_OPS_PER_MCU)
        .max(1)
        .min(i64::MAX as u64) as i64
}

/// The slice of a job's prompts one batch is responsible for.
///
/// Batches partition the prompt list, so batch prices sum to the job price
/// exactly - which is what lets an escrow drain to zero with nothing stranded
/// and nothing conjured.
pub fn inference_batch_cost_mcu(
    prompt_bytes: &[u64],
    batch_start: u32,
    batch_end: u32,
    max_tokens: u32,
    parameter_count: u64,
) -> i64 {
    let lo = (batch_start as usize).min(prompt_bytes.len());
    let hi = (batch_end as usize).clamp(lo, prompt_bytes.len());
    inference_cost_mcu(&prompt_bytes[lo..hi], max_tokens, parameter_count)
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
        WorkSpec::CollatzPeak { start, end } => split_collatz(*start, *end, shards),
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

/// Split a Collatz search the same way a prime range splits.
///
/// Trajectory length is wildly uneven across a range, so equal-width shards are
/// not equal-cost shards. That is fine and deliberate: the price of a shard is
/// derived from its spec, not from how long it ran, so an unlucky shard is
/// slower without being underpaid relative to the model everyone can recompute.
fn split_collatz(start: u64, end: u64, shards: u32) -> Vec<WorkSpec> {
    let width = end.saturating_sub(start);
    let actual = (shards as u64).min(width.max(1)) as u32;
    let base = width / actual as u64;
    let remainder = width % actual as u64;
    let mut out = Vec::with_capacity(actual as usize);
    let mut cursor = start;
    for i in 0..actual {
        let extra = u64::from((i as u64) < remainder);
        let next = cursor + base + extra;
        out.push(WorkSpec::CollatzPeak {
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
        (
            WorkResult::CollatzPeak {
                peak_steps: a,
                peak_seed: asd,
                bucket_peaks: ap,
                bucket_seeds: aseeds,
                ..
            },
            WorkResult::CollatzPeak {
                peak_steps: b,
                peak_seed: bsd,
                bucket_peaks: bp,
                bucket_seeds: bseeds,
                ..
            },
        ) => a == b && asd == bsd && ap == bp && aseeds == bseeds,
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

    /// The price of a prime shard has to track what the machine really does.
    ///
    /// Counting the divisions a range actually costs and comparing them with
    /// the number the ledger charges for is the only way to know a mCU still
    /// means the same work at 10^8 as at 10^4. The flat rate this replaced was
    /// off by more than a hundredfold across that span, which quietly made the
    /// unit mean whichever range you happened to pick.
    #[test]
    fn a_prime_shard_costs_what_it_is_priced_at() {
        fn divisions(n: u64) -> u64 {
            if n < 5 {
                return 1;
            }
            if n.is_multiple_of(2) || n.is_multiple_of(3) {
                return 2;
            }
            let mut ops = 2;
            let mut i = 5_u64;
            while i <= n / i {
                ops += 2;
                if n.is_multiple_of(i) || n.is_multiple_of(i + 2) {
                    return ops;
                }
                i += 6;
            }
            ops
        }

        for (start, end) in [
            (2_u64, 20_000_u64),
            (1_000_000, 1_020_000),
            (100_000_000, 100_020_000),
        ] {
            let measured: u64 = (start..end).map(divisions).sum();
            let priced = verify::compute_ops(&WorkSpec::PrimeCount { start, end });
            let ratio = priced as f64 / measured as f64;
            assert!(
                (0.6..1.7).contains(&ratio),
                "primes in {start}..{end} really cost {measured} divisions but are \
                 priced at {priced} ({ratio:.2}x) - a mCU has stopped meaning one thing"
            );
        }
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

    #[test]
    fn collatz_steps_match_the_known_trajectories() {
        // Hand-checkable cases: 1 is already home, 6 takes 8 steps, and 27
        // is the small seed famous for taking 111.
        assert_eq!(collatz_steps(1), 0);
        assert_eq!(collatz_steps(6), 8);
        assert_eq!(collatz_steps(27), 111);
        assert_eq!(collatz_steps(0), 0);
    }

    #[test]
    fn a_collatz_shard_reports_the_peak_of_its_own_range() {
        let work = WorkSpec::CollatzPeak {
            start: 1,
            end: 1_000,
        };
        let WorkResult::CollatzPeak {
            peak_steps,
            peak_seed,
            bucket_peaks,
            ..
        } = execute_work(&work)
        else {
            panic!("a collatz spec must produce a collatz result");
        };
        // 871 is the longest trajectory below 1000, at 178 steps.
        assert_eq!(peak_steps, 178);
        assert_eq!(peak_seed, 871);
        assert_eq!(bucket_peaks.len(), verify::BUCKETS as usize);
    }

    #[test]
    fn splitting_a_collatz_range_covers_it_exactly_once() {
        let work = WorkSpec::CollatzPeak { start: 5, end: 305 };
        let parts = split_work(&work, 7);
        assert_eq!(parts.len(), 7);
        let mut cursor = 5;
        for part in &parts {
            let WorkSpec::CollatzPeak { start, end } = part else {
                panic!("collatz shards must stay collatz shards, got {part:?}");
            };
            assert_eq!(*start, cursor);
            assert!(end > start);
            cursor = *end;
        }
        assert_eq!(cursor, 305);
    }
}
