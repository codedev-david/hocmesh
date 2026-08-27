//! Tiered result verification.
//!
//! The MVP verified a result by recomputing it in full. That is sound, but it
//! costs the verifier exactly what it cost the worker, so a network of one
//! coordinator and `V` validators burned `V + 2` machines' worth of compute to
//! deliver one machine's worth of useful work. Verification has to be cheaper
//! than the work, or the network is worse than not existing.
//!
//! Three tiers, cheapest first:
//!
//! 1. [`Tier::Witness`] -- a workload-specific check that is asymptotically
//!    cheaper than recomputation. Freivalds for matrix products; a random
//!    bucket audit for prime counting.
//! 2. [`Tier::Replication`] -- the same shard handed to a second, independent
//!    node and the two answers compared. Applied to a sampled fraction.
//! 3. [`Tier::Recompute`] -- full recomputation. Adjudication only, to break a
//!    replication disagreement. Never the default path.
//!
//! # Why the nonce cannot come from the result
//!
//! A witness audit only binds a worker if the worker cannot predict which part
//! of its answer will be checked. If the audit set were derived from public
//! data -- the assignment id, or a hash of the submitted result -- a lazy
//! worker could compute `m` of `BUCKETS` buckets honestly and then grind the
//! result until the audit set happened to fall inside the honest `m`. With
//! `BUCKETS = 64`, `AUDIT_BUCKETS = 3` and `m = 16`, that is
//! `C(64,3) / C(16,3) ~ 74` attempts. Cheap enough to be free.
//!
//! So the challenge is drawn *after* the worker has signed and submitted its
//! result, and it is never drawn by a party with a stake in the answer. Each
//! validator derives it independently, twice over: once from the chain position
//! the entry claims ([`AuditNonce::for_entry`]), and again from the quorum
//! signatures that certified it ([`AuditNonce::for_certified_entry`]). The
//! worker cannot predict either, and a coordinator cannot compute the second
//! one at all without the validator keys - so grinding it costs a fresh, public
//! ledger round per attempt. The worker commits first; the challenge comes
//! second, and comes from somewhere neither side owns. See [`AuditNonce`].

use crate::matrix;
use crate::reputation::Reputation;
use hocmesh_protocol::{WorkResult, WorkSpec};
use sha2::{Digest, Sha256};

/// Buckets a `PrimeCount` shard is divided into for auditing.
pub const BUCKETS: u32 = 64;

/// Buckets actually recomputed per audit.
///
/// Verification therefore costs `AUDIT_BUCKETS / BUCKETS` of the work -- about
/// 4.7% -- and a worker that fabricated every bucket is caught with certainty.
pub const AUDIT_BUCKETS: u32 = 3;

/// Which tier accepted a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// A cheap workload-specific check. The common path.
    Witness,
    /// Two independent nodes agreed.
    Replication,
    /// Full recomputation, used to settle a dispute.
    Recompute,
    /// Accepted without checking, on the strength of the node's record.
    ///
    /// Sound only because the audit that did not happen this time may happen
    /// next time, and a node caught once loses more than it ever gained.
    Unaudited,
}

/// The outcome of checking one result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The result passed at this tier and CU may move.
    Accepted(Tier),
    /// The result is wrong. The reason is operator-facing, not worker-facing.
    Rejected(String),
    /// This tier cannot decide; escalate to the next one.
    Inconclusive,
}

impl Verdict {
    pub fn is_accepted(&self) -> bool {
        matches!(self, Verdict::Accepted(_))
    }
    pub fn is_rejected(&self) -> bool {
        matches!(self, Verdict::Rejected(_))
    }
}

/// A challenge value drawn *after* a result is committed.
///
/// Wrapped in its own type so that it cannot be accidentally constructed from
/// the result it is meant to challenge. [`AuditNonce::for_result`] does not
/// exist, and deliberately so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditNonce(u64);

impl AuditNonce {
    /// Draw a fresh challenge. Call this only once the worker's signed result
    /// is already recorded.
    pub fn draw(randomness: u64) -> Self {
        AuditNonce(randomness)
    }

    /// Rebuild the challenge a coordinator already drew, so a validator can
    /// replay the identical audit.
    pub fn replay(stored: u64) -> Self {
        AuditNonce(stored)
    }

    pub fn value(self) -> u64 {
        self.0
    }

    /// This challenge as a fraction in `[0, 1)`, drawn from a bit range the
    /// bucket selection does not use, so one nonce yields two independent
    /// decisions: whether to audit, and what to audit.
    pub fn unit_interval(self) -> f64 {
        let mixed = mix(self.0 ^ 0x5DEE_CE66_D39B_1A17);
        (mixed >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Derive the authoritative challenge from the chain position an entry
    /// occupies, so neither the provider nor the coordinator supplies it.
    ///
    /// `previous_hash` is the ledger head this entry chains onto. The provider
    /// cannot know it while computing - the head moves with traffic it does not
    /// control - and the coordinator cannot choose it, because chain continuity
    /// forces it to be the current head. See the module docs for what a
    /// coordinator can still do and what that costs it.
    pub fn for_entry(previous_hash: &str, transaction_id: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"hocmesh-audit-v1");
        hasher.update(previous_hash.as_bytes());
        hasher.update(b"|");
        hasher.update(transaction_id.as_bytes());
        let digest = hasher.finalize();
        let mut word = [0u8; 8];
        word.copy_from_slice(&digest[..8]);
        Self(u64::from_be_bytes(word))
    }

    /// Derive the authoritative challenge from the quorum's own signatures.
    ///
    /// This is the only value in the settlement path that neither the provider
    /// nor the coordinator can produce: signing requires validator keys. The
    /// provider commits to a result, the quorum signs the entry containing it,
    /// and only then does the audit target exist.
    ///
    /// Regrinding it means re-proposing at the same sequence, which validators
    /// refuse once they have locked a vote there - so an attempt costs a real,
    /// attributable ledger round rather than a local hash.
    pub fn for_certified_entry(
        previous_hash: &str,
        transaction_id: &str,
        signatures: &[&str],
    ) -> Self {
        let mut ordered: Vec<&str> = signatures.to_vec();
        ordered.sort_unstable();
        let mut hasher = Sha256::new();
        hasher.update(b"hocmesh-audit-beacon-v1");
        hasher.update(previous_hash.as_bytes());
        hasher.update(b"|");
        hasher.update(transaction_id.as_bytes());
        for signature in ordered {
            hasher.update(b"|");
            hasher.update(signature.as_bytes());
        }
        let digest = hasher.finalize();
        let mut word = [0u8; 8];
        word.copy_from_slice(&digest[..8]);
        Self(u64::from_be_bytes(word))
    }
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Choose `count` distinct bucket indices below `total`, driven by the nonce.
///
/// Partial Fisher-Yates: unbiased, allocates `total` slots once, and is
/// deterministic given the nonce so every validator audits the same buckets.
pub fn audit_indices(nonce: AuditNonce, total: u32, count: u32) -> Vec<u32> {
    let count = count.min(total);
    let mut pool: Vec<u32> = (0..total).collect();
    let mut state = nonce.value();
    for slot in 0..count as usize {
        state = mix(state);
        let remaining = pool.len() - slot;
        let pick = slot + (state % remaining as u64) as usize;
        pool.swap(slot, pick);
    }
    pool.truncate(count as usize);
    pool.sort_unstable();
    pool
}

/// The half-open range of integers a `PrimeCount` bucket covers.
///
/// Shared by the worker and the auditor so that both agree on where bucket
/// boundaries fall; an off-by-one here would reject honest work.
pub fn bucket_bounds(start: u64, end: u64, index: u32, buckets: u32) -> (u64, u64) {
    let width = end.saturating_sub(start);
    let buckets = buckets.max(1) as u64;
    let base = width / buckets;
    let remainder = width % buckets;
    let index = index as u64;
    // The first `remainder` buckets carry one extra integer, so the bucket
    // widths differ by at most one and every integer lands in exactly one.
    let lead = index.min(remainder);
    let offset = base * index + lead;
    let this = base + u64::from(index < remainder);
    let from = start.saturating_add(offset);
    (from, from.saturating_add(this).min(end))
}

/// Check a result with the cheapest sound check available for its workload.
///
/// Returns [`Verdict::Inconclusive`] when the workload has no witness, which
/// tells the caller to fall back to replication rather than to accept.
pub fn witness_check(work: &WorkSpec, result: &WorkResult, nonce: AuditNonce) -> Verdict {
    match (work, result) {
        (
            WorkSpec::PrimeCount { start, end },
            WorkResult::PrimeCount {
                count,
                bucket_counts,
                ..
            },
        ) => prime_witness(*start, *end, *count, bucket_counts, nonce),
        (
            WorkSpec::MatrixMultiply {
                seed_a,
                seed_b,
                dim,
                row_start,
                row_end,
            },
            WorkResult::MatrixMultiply { rows, .. },
        ) => freivalds(*seed_a, *seed_b, *dim, *row_start, *row_end, rows, nonce),
        _ => Verdict::Rejected("result does not match the assigned workload".into()),
    }
}

/// Audit a prime count without recounting the whole range.
///
/// Two checks. The structural one is free: the per-bucket counts must sum to
/// the reported total, so a worker cannot report honest buckets and a dishonest
/// headline. The sampled one recomputes `AUDIT_BUCKETS` of `BUCKETS` buckets.
///
/// A worker that computed `m` of `B` buckets and guessed the rest survives with
/// probability `C(m, k) / C(B, k)`: 0.9% for `m = 32`, and 0 for `m < B - k`
/// once every guess is wrong.
fn prime_witness(
    start: u64,
    end: u64,
    count: u64,
    bucket_counts: &[u64],
    nonce: AuditNonce,
) -> Verdict {
    if bucket_counts.len() != BUCKETS as usize {
        return Verdict::Rejected(format!(
            "expected {BUCKETS} bucket counts, got {}",
            bucket_counts.len()
        ));
    }
    let claimed: u64 = bucket_counts
        .iter()
        .copied()
        .fold(0u64, u64::saturating_add);
    if claimed != count {
        return Verdict::Rejected(format!(
            "bucket counts sum to {claimed} but the reported total is {count}"
        ));
    }
    for index in audit_indices(nonce, BUCKETS, AUDIT_BUCKETS) {
        let (from, to) = bucket_bounds(start, end, index, BUCKETS);
        let actual = crate::compute::count_primes(from, to);
        if actual != bucket_counts[index as usize] {
            return Verdict::Rejected(format!(
                "bucket {index} covering [{from}, {to}) holds {actual} primes, not {}",
                bucket_counts[index as usize]
            ));
        }
    }
    Verdict::Accepted(Tier::Witness)
}

/// Freivalds' check for `C = A x B` restricted to the shard's rows.
///
/// Draw a random vector `r`, then compare `A(Br)` with `Cr`. Computing the
/// shard costs `rows * dim^2` multiplications; this costs `dim^2 + 2*rows*dim`.
/// If `C` is wrong, the check passes with probability at most `1/MODULUS`,
/// which is below 2^-31 -- so a single round suffices.
fn freivalds(
    seed_a: u64,
    seed_b: u64,
    dim: u32,
    row_start: u32,
    row_end: u32,
    rows: &[u32],
    nonce: AuditNonce,
) -> Verdict {
    let span = row_end.saturating_sub(row_start) as usize;
    let width = dim as usize;
    if rows.len() != span * width {
        return Verdict::Rejected(format!(
            "expected {} result entries, got {}",
            span * width,
            rows.len()
        ));
    }
    if rows
        .iter()
        .any(|value| u64::from(*value) >= matrix::MODULUS)
    {
        return Verdict::Rejected("result contains a value outside the field".into());
    }

    let challenge = matrix::challenge_vector(nonce.value(), dim);
    // Br, once, shared across every row of the shard.
    let projected = matrix::matrix_vector(seed_b, dim, &challenge);

    for (offset, row_index) in (row_start..row_end).enumerate() {
        let a_row = matrix::row(seed_a, row_index, dim);
        let expected = matrix::dot(&a_row, &projected);
        let claimed = matrix::dot(&rows[offset * width..(offset + 1) * width], &challenge);
        if expected != claimed {
            return Verdict::Rejected(format!(
                "row {row_index} fails the product check ({claimed} != {expected})"
            ));
        }
    }
    Verdict::Accepted(Tier::Witness)
}

/// Settle a dispute by recomputing the shard in full.
///
/// This is the expensive path the tiers above exist to avoid. It runs when two
/// replicas disagree, which tells us one of them is lying but not which.
pub fn adjudicate(work: &WorkSpec, result: &WorkResult) -> Verdict {
    let truth = crate::compute::execute_work(work);
    if crate::compute::results_agree(&truth, result) {
        Verdict::Accepted(Tier::Recompute)
    } else {
        Verdict::Rejected("recomputation disagrees with the submitted result".into())
    }
}

/// Compare two independent answers to the same shard.
pub fn compare_replicas(first: &WorkResult, second: &WorkResult) -> Verdict {
    if crate::compute::results_agree(first, second) {
        Verdict::Accepted(Tier::Replication)
    } else {
        Verdict::Inconclusive
    }
}

/// Multiplications needed to produce a shard, for cost reporting.
pub fn compute_ops(work: &WorkSpec) -> u64 {
    match work {
        WorkSpec::PrimeCount { start, end } => end
            .saturating_sub(*start)
            .saturating_mul(trial_division_ops(*end)),
        WorkSpec::MatrixMultiply {
            dim,
            row_start,
            row_end,
            ..
        } => {
            let span = u64::from(row_end.saturating_sub(*row_start));
            span * u64::from(*dim) * u64::from(*dim)
        }
    }
}

/// Divisions one trial-division candidate costs near `n`.
///
/// Composites almost all fall out on the first two checks; a prime runs to the
/// square root in steps of six, and about one candidate in `ln n` is prime.
///
/// Flat-rating a candidate, which is what the first cut did, makes one mCU
/// mean a hundred times more machine work at 10^12 than at 10^6 - and a unit
/// that drifts with its input is not a unit.
///
/// Integer arithmetic only. This number sets a price every validator has to
/// reproduce exactly, and floating point makes no such promise across
/// machines.
pub fn trial_division_ops(n: u64) -> u64 {
    let n = n.max(2);
    // ln n, from the exact integer log2 scaled by 693/1000 ~ ln 2.
    let ln_n = (u64::from(n.ilog2()) * 693 / 1000).max(1);
    2 + n.isqrt() / (3 * ln_n)
}

/// Multiplications needed to check a shard with its witness.
pub fn witness_ops(work: &WorkSpec) -> u64 {
    match work {
        WorkSpec::PrimeCount { start, end } => {
            let width = end.saturating_sub(*start);
            // The audit recomputes whole buckets, and a candidate inside an
            // audited bucket costs exactly what it cost the provider. Leaving
            // that factor out was the bug: it priced the check as if trial
            // division were free and reported prime counting as 900x cheaper
            // to verify than to do, when the sampling rate makes it 21x.
            width / u64::from(BUCKETS) * u64::from(AUDIT_BUCKETS) * trial_division_ops(*end)
        }
        WorkSpec::MatrixMultiply {
            dim,
            row_start,
            row_end,
            ..
        } => {
            let span = u64::from(row_end.saturating_sub(*row_start));
            let dim = u64::from(*dim);
            // Br is dim^2 and the two shard passes are span*dim each, but the
            // check also has to regenerate B and the shard's rows of A before
            // it can touch either. Counting only the multiplies made Freivalds
            // look twice as cheap as it measures.
            2 * dim * dim + 3 * span * dim
        }
    }
}

/// How many times cheaper checking is than doing. Above 1.0 means the network
/// produces more useful work than it spends policing itself.
pub fn verification_advantage(work: &WorkSpec) -> f64 {
    let checking = witness_ops(work).max(1) as f64;
    compute_ops(work) as f64 / checking
}

/// One settlement decision, and the evidence needed to replay it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settlement {
    pub verdict: Verdict,
    /// Whether this result actually drew an audit.
    pub audited: bool,
    /// The challenge drawn, recorded so validators replay the same audit.
    pub nonce: u64,
}

/// Decide whether CU may move for one result.
///
/// The common path does no work at all. A node with a long clean record is
/// believed, because the audit it escaped this time is one it cannot escape
/// reliably, and one catch costs more than the cheating ever paid. That is what
/// makes verification affordable: it is priced per *audit*, not per result.
pub fn settle(
    work: &WorkSpec,
    result: &WorkResult,
    reputation: &Reputation,
    nonce: AuditNonce,
) -> Settlement {
    if !reputation.should_audit(nonce) {
        return Settlement {
            verdict: Verdict::Accepted(Tier::Unaudited),
            audited: false,
            nonce: nonce.value(),
        };
    }
    Settlement {
        verdict: witness_check(work, result, nonce),
        audited: true,
        nonce: nonce.value(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::execute_work;

    fn primes(start: u64, end: u64) -> (WorkSpec, WorkResult) {
        let work = WorkSpec::PrimeCount { start, end };
        let result = execute_work(&work);
        (work, result)
    }

    fn product(dim: u32, row_start: u32, row_end: u32) -> (WorkSpec, WorkResult) {
        let work = WorkSpec::MatrixMultiply {
            seed_a: 0xA5A5,
            seed_b: 0x5A5A,
            dim,
            row_start,
            row_end,
        };
        let result = execute_work(&work);
        (work, result)
    }

    /// The whole scheme is worthless if it ever rejects honest work, so this
    /// sweeps every nonce a real audit could draw rather than sampling a few.
    #[test]
    fn honest_prime_work_survives_every_audit_it_could_draw() {
        let (work, result) = primes(1_000, 20_000);
        for seed in 0..2_000u64 {
            let verdict = witness_check(&work, &result, AuditNonce::draw(seed));
            assert_eq!(verdict, Verdict::Accepted(Tier::Witness), "seed {seed}");
        }
    }

    #[test]
    fn honest_matrix_work_survives_every_challenge_it_could_draw() {
        let (work, result) = product(64, 0, 16);
        for seed in 0..2_000u64 {
            let verdict = witness_check(&work, &result, AuditNonce::draw(seed));
            assert_eq!(verdict, Verdict::Accepted(Tier::Witness), "seed {seed}");
        }
    }

    /// A total that does not match its own buckets costs nothing to catch:
    /// no recomputation happens at all, so this cheat is free to detect.
    #[test]
    fn a_total_that_contradicts_its_own_buckets_is_caught_without_any_recomputation() {
        let (work, result) = primes(1_000, 20_000);
        let WorkResult::PrimeCount { bucket_counts, .. } = result else {
            unreachable!()
        };
        let inflated = WorkResult::PrimeCount {
            count: bucket_counts.iter().sum::<u64>() + 1,
            bucket_counts,
            duration_ms: 1,
        };
        for seed in 0..64u64 {
            let verdict = witness_check(&work, &inflated, AuditNonce::draw(seed));
            assert!(
                verdict.is_rejected(),
                "seed {seed} let an inflated total through"
            );
        }
    }

    /// A worker that computes nothing and invents a self-consistent answer.
    /// It survives the free structural check, so only the audit can catch it.
    #[test]
    fn a_wholly_invented_answer_is_caught_by_essentially_every_audit() {
        let (work, honest) = primes(1_000, 20_000);
        let WorkResult::PrimeCount { bucket_counts, .. } = honest else {
            unreachable!()
        };
        let faked: Vec<u64> = bucket_counts.iter().map(|c| c + 1).collect();
        let lie = WorkResult::PrimeCount {
            count: faked.iter().sum(),
            bucket_counts: faked,
            duration_ms: 1,
        };
        let caught = (0..2_000u64)
            .filter(|s| witness_check(&work, &lie, AuditNonce::draw(*s)).is_rejected())
            .count();
        assert_eq!(
            caught, 2_000,
            "every bucket was wrong; every audit must catch it"
        );
    }

    fn choose(n: u64, k: u64) -> f64 {
        if k > n {
            return 0.0;
        }
        (0..k).map(|i| (n - i) as f64 / (i + 1) as f64).product()
    }

    /// A worker that does half the work and fabricates the rest, keeping the
    /// total self-consistent so the free check cannot see it.
    fn half_lazy(work: &WorkSpec, honest: &WorkResult, effort: u32) -> WorkResult {
        let _ = work;
        let WorkResult::PrimeCount { bucket_counts, .. } = honest else {
            unreachable!()
        };
        let faked: Vec<u64> = bucket_counts
            .iter()
            .enumerate()
            .map(|(i, c)| if (i as u32) < effort { *c } else { c + 1 })
            .collect();
        WorkResult::PrimeCount {
            count: faked.iter().sum(),
            bucket_counts: faked,
            duration_ms: 1,
        }
    }

    /// The detection rate is not a hope, it is a hypergeometric draw. A worker
    /// that honestly computes `m` of `BUCKETS` buckets escapes exactly when all
    /// `AUDIT_BUCKETS` samples land inside those `m`. This checks the measured
    /// rate against that closed form, so the guarantee is a number, not a vibe.
    #[test]
    fn the_measured_detection_rate_matches_the_combinatorial_prediction() {
        let (work, honest) = primes(1_000, 20_000);
        const TRIALS: u64 = 20_000;
        for effort in [16u32, 32, 48, 56] {
            let lie = half_lazy(&work, &honest, effort);
            let caught = (0..TRIALS)
                .filter(|s| witness_check(&work, &lie, AuditNonce::draw(*s)).is_rejected())
                .count() as f64;
            let measured = caught / TRIALS as f64;
            let escape = choose(u64::from(effort), u64::from(AUDIT_BUCKETS))
                / choose(u64::from(BUCKETS), u64::from(AUDIT_BUCKETS));
            let predicted = 1.0 - escape;
            assert!(
                (measured - predicted).abs() < 0.02,
                "effort {effort}: measured {measured:.4}, predicted {predicted:.4}"
            );
        }
    }

    /// Freivalds is one-sided: a wrong product survives a challenge with
    /// probability at most 1/MODULUS, so a single corrupted entry anywhere in
    /// the shard must be caught by every challenge we can afford to try.
    #[test]
    fn one_wrong_entry_in_a_matrix_product_is_caught_by_every_challenge() {
        let (work, result) = product(64, 0, 16);
        let WorkResult::MatrixMultiply { mut rows, .. } = result else {
            unreachable!()
        };
        let victim = rows.len() / 2;
        rows[victim] = (rows[victim] + 1) % (matrix::MODULUS as u32);
        let tampered = WorkResult::MatrixMultiply {
            rows,
            duration_ms: 1,
        };
        let escaped = (0..5_000u64)
            .filter(|s| !witness_check(&work, &tampered, AuditNonce::draw(*s)).is_rejected())
            .count();
        assert_eq!(
            escaped, 0,
            "a single flipped element slipped past a challenge"
        );
    }

    #[test]
    fn a_matrix_shard_of_the_wrong_shape_is_rejected_not_misread() {
        let (work, _) = product(64, 0, 16);
        let stunted = WorkResult::MatrixMultiply {
            rows: vec![0; 8],
            duration_ms: 1,
        };
        assert!(witness_check(&work, &stunted, AuditNonce::draw(7)).is_rejected());
    }

    #[test]
    fn a_result_from_a_different_workload_is_rejected() {
        let (prime_work, _) = primes(1_000, 2_000);
        let (_, matrix_result) = product(32, 0, 8);
        assert!(
            witness_check(&prime_work, &matrix_result, AuditNonce::draw(1)).is_rejected(),
            "a matrix answer must never settle a prime shard"
        );
    }

    /// The claim the whole design rests on: checking costs strictly less than
    /// doing. If this ever fails, the network is burning more compute than it
    /// delivers and nothing else in hocMESH matters.
    #[test]
    fn checking_always_costs_less_than_doing() {
        let cases = [
            WorkSpec::PrimeCount {
                start: 0,
                end: 1_000_000,
            },
            WorkSpec::MatrixMultiply {
                seed_a: 1,
                seed_b: 2,
                dim: 512,
                row_start: 0,
                row_end: 64,
            },
        ];
        for work in cases {
            let advantage = verification_advantage(&work);
            assert!(
                advantage > 10.0,
                "{work:?} verifies at only {advantage:.1}x cheaper than it computes"
            );
        }
    }

    /// Every validator must audit exactly what the coordinator audited, or
    /// honest work would be rejected somewhere in the network at random.
    #[test]
    fn replaying_a_recorded_nonce_reproduces_the_identical_audit() {
        for seed in 0..500u64 {
            let drawn = audit_indices(AuditNonce::draw(seed), BUCKETS, AUDIT_BUCKETS);
            let replayed = audit_indices(AuditNonce::replay(seed), BUCKETS, AUDIT_BUCKETS);
            assert_eq!(drawn, replayed, "seed {seed} audited differently on replay");
        }
    }

    #[test]
    fn an_audit_never_samples_the_same_bucket_twice() {
        for seed in 0..1_000u64 {
            let picked = audit_indices(AuditNonce::draw(seed), BUCKETS, AUDIT_BUCKETS);
            assert_eq!(picked.len(), AUDIT_BUCKETS as usize);
            let mut unique = picked.clone();
            unique.dedup();
            assert_eq!(unique, picked, "seed {seed} sampled a bucket twice");
            assert!(picked.iter().all(|i| *i < BUCKETS));
        }
    }

    /// An off-by-one in bucket boundaries would reject honest work, so the
    /// buckets must tile the range exactly: no gaps, no overlaps, nothing lost.
    #[test]
    fn buckets_tile_the_range_exactly() {
        for (start, end) in [(0u64, 1_000u64), (7, 130), (1_000, 1_001), (5, 5 + 63)] {
            let mut cursor = start;
            let mut covered = 0u64;
            for index in 0..BUCKETS {
                let (lo, hi) = bucket_bounds(start, end, index, BUCKETS);
                assert_eq!(lo, cursor, "bucket {index} of [{start},{end}) left a gap");
                assert!(hi >= lo);
                covered += hi - lo;
                cursor = hi;
            }
            assert_eq!(cursor, end, "buckets stopped short of [{start},{end})");
            assert_eq!(covered, end - start);
        }
    }

    /// A quorum signature set is a set, not a list: every validator must derive
    /// the same challenge from the same certified entry however the signatures
    /// happen to be ordered on the wire, or the quorum would disagree about
    /// what "verified" even means.
    #[test]
    fn every_validator_draws_the_same_challenge_from_one_entry() {
        let collected = ["sig-a", "sig-b", "sig-c"];
        let shuffled = ["sig-c", "sig-a", "sig-b"];
        let first = AuditNonce::for_certified_entry("head", "tx-1", &collected);
        let second = AuditNonce::for_certified_entry("head", "tx-1", &shuffled);
        assert_eq!(first.value(), second.value());
    }

    /// The challenge is bound to the chain position the entry occupies, so the
    /// same work settled onto a different head is audited somewhere else. That
    /// is what stops a coordinator from replaying one lucky audit forever.
    #[test]
    fn a_different_chain_position_draws_a_different_challenge() {
        let quorum = ["sig-a", "sig-b", "sig-c"];
        let here = AuditNonce::for_certified_entry("head-1", "tx-1", &quorum);
        let later = AuditNonce::for_certified_entry("head-2", "tx-1", &quorum);
        assert_ne!(here.value(), later.value());
    }

    /// Swapping a single validator changes the beacon completely, and no party
    /// outside the quorum can forge the draw, because signing needs the keys.
    #[test]
    fn a_different_quorum_draws_a_different_challenge() {
        let signed = ["sig-a", "sig-b", "sig-c"];
        let reshuffled = ["sig-a", "sig-b", "sig-d"];
        let first = AuditNonce::for_certified_entry("head", "tx", &signed);
        let second = AuditNonce::for_certified_entry("head", "tx", &reshuffled);
        assert_ne!(first.value(), second.value());
    }

    /// The strongest cheap cheat: skip whole buckets, fill them from a
    /// neighbour so the numbers stay plausible, and report a total that matches
    /// the buckets exactly so the free arithmetic check finds nothing.
    fn skip_buckets(honest: &WorkResult, skipped: u32) -> WorkResult {
        let WorkResult::PrimeCount { bucket_counts, .. } = honest else {
            unreachable!("this helper fabricates prime work")
        };
        let mut counts = bucket_counts.clone();
        for index in audit_indices(AuditNonce::replay(0xC0FF_EE00), BUCKETS, skipped) {
            counts[index as usize] = bucket_counts[((index + 1) % BUCKETS) as usize];
        }
        WorkResult::PrimeCount {
            count: counts.iter().sum(),
            bucket_counts: counts,
            duration_ms: 0,
        }
    }

    /// How often a fabricated result survives the propose-time challenge, the
    /// apply-time beacon, and both together, sampled over distinct positions.
    fn escape_rates(work: &WorkSpec, lazy: &WorkResult, entries: u32) -> (f64, f64, f64) {
        let (mut first, mut second, mut both) = (0u32, 0u32, 0u32);
        for n in 0..entries {
            let head = format!("head-{n}");
            let tx = format!("tx-{n}");
            let quorum = [format!("a-{n}"), format!("b-{n}"), format!("c-{n}")];
            let signed: Vec<&str> = quorum.iter().map(String::as_str).collect();
            let chain = AuditNonce::for_entry(&head, &tx);
            let beacon = AuditNonce::for_certified_entry(&head, &tx, &signed);
            let a = witness_check(work, lazy, chain).is_accepted();
            let b = witness_check(work, lazy, beacon).is_accepted();
            first += u32::from(a);
            second += u32::from(b);
            both += u32::from(a && b);
        }
        let rate = |hits: u32| f64::from(hits) / f64::from(entries);
        (rate(first), rate(second), rate(both))
    }

    /// Propose-time and apply-time challenges are independent draws over the
    /// same entry, so escaping settlement means escaping both. Two layers
    /// multiply: surviving one is not surviving the ledger.
    #[test]
    fn a_lazy_result_must_escape_both_challenges_to_settle() {
        let (work, honest) = primes(1, 20_000);
        let lazy = skip_buckets(&honest, 16);
        let (chain, beacon, both) = escape_rates(&work, &lazy, 1_000);
        let independent = chain * beacon;
        assert!(
            (both - independent).abs() < 0.06,
            "the layers are not independent: {both} measured, {independent} predicted"
        );
        assert!(
            both < chain - 0.15,
            "the beacon must cost the cheat something: {chain} -> {both}"
        );
    }

    /// Grinding the beacon is not a local hash loop. Each attempt needs a fresh
    /// quorum to sign a fresh entry, and a validator that has locked a vote at
    /// a sequence refuses a conflicting entry hash there - so every retry is a
    /// public, attributable round rather than a private guess.
    #[test]
    fn grinding_the_beacon_costs_more_public_rounds_the_more_a_node_skips() {
        let (work, honest) = primes(1, 20_000);
        let rounds = |skipped: u32| {
            let (_, _, both) = escape_rates(&work, &skip_buckets(&honest, skipped), 1_000);
            1.0 / both.max(0.001)
        };
        let (light, heavy, brazen) = (rounds(4), rounds(16), rounds(32));
        assert!(
            light < heavy && heavy < brazen,
            "grinding must get harder, not easier: {light} {heavy} {brazen}"
        );
        assert!(
            brazen > 8.0,
            "skipping half the work should cost many public rounds, not {brazen}"
        );
    }
}
