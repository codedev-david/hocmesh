//! Witnessing floating-point work.
//!
//! Everything in [`crate::verify`] rests on exact arithmetic: two honest nodes
//! computing the same shard produce bit-identical answers, so any difference is
//! a cheat. Neural network inference does not work that way. The same weights
//! and the same prompt give slightly different logits on different hardware,
//! because float addition is not associative and every kernel sums in its own
//! order. Bit-equality would reject every honest node.
//!
//! That is the gap this module closes, and it is the gap between hocMESH
//! verifying a toy prime count and hocMESH verifying the work people actually
//! want to buy. Transformer inference is dominated by matrix products, and
//! Freivalds' test checks a matrix product in O(n^2) instead of O(n^3). The
//! only thing that has to change is the comparison: instead of asking whether
//! the residual is zero, ask whether it is small enough to be rounding.
//!
//! "Small enough" is not a guess. [`TOLERANCE`] sits between two measured
//! quantities: how far two honest kernels drift apart, and how far the
//! cheapest worthwhile cheat moves the answer. The example
//! `float_witness_proof` measures both and asserts the gap is wide.
//!
//! One round of Freivalds over the reals is weaker than one round over a large
//! prime field: a challenge vector of plus and minus ones catches a wrong
//! product with probability at least 1/2, not 1 - 1e-9. So this module runs
//! [`ROUNDS`] independent rounds and requires every one of them to pass, which
//! puts the escape probability below 1 in 250 for the weakest possible cheat
//! and far lower for any cheat large enough to be worth the trouble.
use crate::matrix::splitmix64;
use sha2::{Digest, Sha256};

/// Independent challenge vectors drawn per witness. Each round catches a wrong
/// product with probability at least 1/2, so eight rounds put the escape
/// probability below 1 in 250 even for an adversary who knows the algorithm.
pub const ROUNDS: usize = 8;

/// How far apart two honest results may drift, relative to the magnitude of
/// the answer itself.
///
/// Measured, not chosen. Section 7 of the `float_witness_proof` example walks
/// the inner dimension across a 64x range and reports two populations that
/// never approach each other: an honest f32 kernel drifts 3.5e-7 to 2.9e-6,
/// and the same shard run in fp16 or TF32 drifts 3.1e-3 to 2.3e-2.
///
/// This constant is their geometric centre. Honest work clears it by 35x at
/// the worst shape measured; the cheapest cheat worth running misses it by
/// 31x at the shape most favourable to the cheat. Neither margin closes as
/// the shard grows, which is why one constant serves every job.
pub const TOLERANCE: f32 = 1e-4;

/// Shape of a matrix product `C = A x B`, with `A` as `rows x inner` and `B`
/// as `inner x cols`, both stored row-major.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub rows: usize,
    pub inner: usize,
    pub cols: usize,
}

impl Shape {
    /// Multiply-accumulates in the product itself.
    pub fn compute_ops(self) -> u64 {
        (self.rows * self.inner * self.cols) as u64
    }

    /// Multiply-accumulates in one witness round: `B r`, then `A (B r)`, then
    /// `C r`. Linear in each dimension instead of their product.
    pub fn witness_ops(self) -> u64 {
        (self.inner * self.cols + self.rows * self.inner + self.rows * self.cols) as u64
    }
}

/// The challenge block: one Rademacher vector per round, stacked so that the
/// `ROUNDS` signs for a given index sit next to each other.
///
/// Signs rather than magnitudes: a Rademacher vector keeps the residual on the
/// same scale as the answer, so the tolerance means the same thing whatever
/// the weights look like. Gaussian or uniform draws would let one large
/// component drown the rest.
///
/// The layout is what makes the witness fast: every matrix is read once and
/// all `ROUNDS` challenges ride along in registers, instead of `ROUNDS`
/// separate sweeps through memory.
pub fn challenge_block(nonce: u64, len: usize) -> Vec<f32> {
    let mut state = nonce | 1;
    (0..len * ROUNDS)
        .map(|_| {
            state = splitmix64(state);
            if state & 1 == 0 { 1.0 } else { -1.0 }
        })
        .collect()
}

/// `M R` for a row-major `M` of `rows x cols` and a `cols x ROUNDS` block,
/// accumulated in f64 so the witness is not itself a source of the drift it
/// is measuring.
fn mat_block(m: &[f32], rows: usize, cols: usize, r: &[f32]) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * ROUNDS];
    for i in 0..rows {
        let src = i * cols;
        let mut acc = [0.0f64; ROUNDS];
        for k in 0..cols {
            let value = f64::from(m[src + k]);
            let signs: &[f32; ROUNDS] = r[k * ROUNDS..][..ROUNDS].try_into().unwrap();
            for t in 0..ROUNDS {
                acc[t] += value * f64::from(signs[t]);
            }
        }
        out[i * ROUNDS..][..ROUNDS].copy_from_slice(&acc);
    }
    out
}

/// `M X` where `X` already carries f64 precision from an earlier stage.
fn mat_block64(m: &[f32], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * ROUNDS];
    for i in 0..rows {
        let src = i * cols;
        let mut acc = [0.0f64; ROUNDS];
        for k in 0..cols {
            let value = f64::from(m[src + k]);
            let row: &[f64; ROUNDS] = x[k * ROUNDS..][..ROUNDS].try_into().unwrap();
            for t in 0..ROUNDS {
                acc[t] += value * row[t];
            }
        }
        out[i * ROUNDS..][..ROUNDS].copy_from_slice(&acc);
    }
    out
}

/// The relative residual of every round, in one sweep over each matrix.
///
/// Each entry compares `C r` against `A (B r)` for one challenge and divides by
/// the size of the answer, so the numbers are dimensionless: 0.0 means exactly
/// right, 1.0 means as wrong as it is large.
pub fn round_residuals(a: &[f32], b: &[f32], c: &[f32], shape: Shape, nonce: u64) -> [f32; ROUNDS] {
    let r = challenge_block(nonce, shape.cols);
    let br = mat_block(b, shape.inner, shape.cols, &r);
    let abr = mat_block64(a, shape.rows, shape.inner, &br);
    let cr = mat_block(c, shape.rows, shape.cols, &r);
    let mut gap = [0.0f64; ROUNDS];
    let mut size = [0.0f64; ROUNDS];
    for i in 0..shape.rows {
        for t in 0..ROUNDS {
            let (left, right) = (cr[i * ROUNDS + t], abr[i * ROUNDS + t]);
            gap[t] = gap[t].max((left - right).abs());
            size[t] = size[t].max(left.abs().max(right.abs()));
        }
    }
    let mut out = [0.0f32; ROUNDS];
    for t in 0..ROUNDS {
        out[t] = (gap[t] / size[t].max(f64::from(f32::EPSILON))) as f32;
    }
    out
}

/// The worst relative residual across every round of a witness.
///
/// A caller compares this against [`TOLERANCE`]. Reporting the number rather
/// than a bare verdict lets an adjudicator see how badly a result failed, and
/// lets the proof example measure honest drift and cheating side by side.
pub fn witness_residual(a: &[f32], b: &[f32], c: &[f32], shape: Shape, nonce: u64) -> f32 {
    round_residuals(a, b, c, shape, nonce)
        .into_iter()
        .fold(0.0f32, f32::max)
}

/// Whether a claimed product survives a float witness at the given challenge.
///
/// Every round must pass. One round of Rademacher Freivalds is a coin flip
/// against a determined adversary; [`ROUNDS`] of them is not.
pub fn witnessed(a: &[f32], b: &[f32], c: &[f32], shape: Shape, nonce: u64) -> bool {
    witness_residual(a, b, c, shape, nonce) <= TOLERANCE
}

/// A commitment to a result, over the raw IEEE-754 bits.
///
/// A witness needs the whole product to compute `C r`, but a ledger entry
/// carrying the whole product would be enormous: one 128x512 shard of f32 is
/// 256 KB of payload against 32 bytes of commitment. So the entry commits and
/// the payload travels with the answer the requester already receives.
///
/// Hashing the bit patterns rather than the decimal forms keeps this exact and
/// endian-stable: two nodes that agree on the payload agree on the digest, and
/// no rounding creeps into the binding.
pub fn commit(product: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"hocmesh-tensor-commit-v1");
    hasher.update((product.len() as u64).to_be_bytes());
    for value in product {
        hasher.update(value.to_bits().to_be_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Witness a product the ledger holds only a commitment to.
///
/// The commitment is checked first, before a single multiply is spent on the
/// payload. That ordering is the point: a provider that has seen the challenge
/// cannot swap in a matrix that happens to satisfy it, because the entry it
/// already signed pins the one matrix it is allowed to be judged on.
pub fn witnessed_committed(
    a: &[f32],
    b: &[f32],
    payload: &[f32],
    commitment: &str,
    shape: Shape,
    nonce: u64,
) -> bool {
    commit(payload) == commitment && witnessed(a, b, payload, shape, nonce)
}

/// Why the ledger cannot simply run the witness above.
///
/// Freivalds checks a product you already hold. A validator that never
/// receives `C` cannot run it, and a provider that never computed `C` can
/// answer any challenge about it for free: `C r` and `A (B r)` are the same
/// vector, and the second costs `O(n^2)`. The test detects a wrong answer; it
/// does not prove anyone ever did the work.
///
/// So possession has to be forced before the challenge is drawn, exactly as
/// the prime audit forces it. The shard is committed one row block at a time.
/// The provider publishes `BLOCKS` digests before it knows which the audit
/// will open; the challenge then names `verify::AUDIT_BUCKETS` of them and the
/// provider must reveal those rows. A validator re-executes only those rows.
///
/// A provider that skipped `m` blocks is caught unless every opened block
/// falls in the part it did compute, so it pays the same hypergeometric
/// escape rate the prime audit charges, and the two challenges the beacon
/// draws still compose.
pub const BLOCKS: u32 = crate::verify::BUCKETS;

/// The half-open row range block `index` covers.
pub fn block_rows(shape: Shape, index: u32) -> (usize, usize) {
    let (start, end) = crate::verify::bucket_bounds(0, shape.rows as u64, index, BLOCKS);
    (start as usize, end as usize)
}

/// The rows of `product` that block `index` covers.
pub fn block_payload(product: &[f32], shape: Shape, index: u32) -> &[f32] {
    let (start, end) = block_rows(shape, index);
    &product[start * shape.cols..end * shape.cols]
}

/// One digest per row block, published before the challenge exists.
pub fn block_commitments(product: &[f32], shape: Shape) -> Vec<String> {
    (0..BLOCKS)
        .map(|index| commit(block_payload(product, shape, index)))
        .collect()
}

/// The blocks a challenge opens for reveal.
pub fn opened_blocks(nonce: crate::verify::AuditNonce) -> Vec<u32> {
    crate::verify::audit_indices(nonce, BLOCKS, crate::verify::AUDIT_BUCKETS)
}

/// Re-execute one opened block and check the revealed rows against it.
///
/// The digest binds the reveal to what was committed before the challenge; the
/// re-execution decides whether what was committed is the answer. Both have to
/// hold, and neither alone is worth anything.
pub fn block_reexecuted(
    a: &[f32],
    b: &[f32],
    shape: Shape,
    index: u32,
    revealed: &[f32],
    commitment: &str,
) -> bool {
    let (start, end) = block_rows(shape, index);
    if revealed.len() != (end - start) * shape.cols || commit(revealed) != commitment {
        return false;
    }
    let mut worst = 0.0f32;
    let mut scale = 0.0f32;
    for row in start..end {
        for col in 0..shape.cols {
            let mut acc = 0.0f64;
            for k in 0..shape.inner {
                acc += f64::from(a[row * shape.inner + k]) * f64::from(b[k * shape.cols + col]);
            }
            let honest = acc as f32;
            let claim = revealed[(row - start) * shape.cols + col];
            worst = worst.max((honest - claim).abs());
            scale = scale.max(honest.abs());
        }
    }
    // A block of exact zeros has no scale to be relative to, so it has to
    // match exactly; anything else is compared against its own magnitude.
    if scale == 0.0 {
        return worst == 0.0;
    }
    worst / scale <= TOLERANCE
}

/// The single digest a ledger entry carries for a whole shard.
///
/// The per-block digests are what the audit needs, but 64 of them is 4 KB and
/// an entry should be small. So the entry carries their root, and the block
/// list arrives with the reveal, where it can be checked against the root that
/// was published before anyone knew which blocks would be opened.
pub fn commit_blocks(commitments: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"hocmesh-tensor-blocks-v1");
    hasher.update((commitments.len() as u64).to_be_bytes());
    for digest in commitments {
        hasher.update((digest.len() as u64).to_be_bytes());
        hasher.update(digest.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The whole audit a validator runs on a revealed shard.
///
/// `root` came from the ledger entry, which was signed before `nonce` could be
/// derived. `digests` and `revealed` arrive with the reveal, so neither is
/// trusted: the digest list has to reproduce the root, and the revealed rows
/// have to match both their own digest and a fresh re-execution of those rows.
pub fn reveal_accepted(
    a: &[f32],
    b: &[f32],
    shape: Shape,
    root: &str,
    digests: &[String],
    nonce: crate::verify::AuditNonce,
    revealed: &[&[f32]],
) -> bool {
    if digests.len() != BLOCKS as usize || commit_blocks(digests) != root {
        return false;
    }
    let opened = opened_blocks(nonce);
    if revealed.len() != opened.len() {
        return false;
    }
    opened
        .iter()
        .zip(revealed)
        .all(|(index, rows)| block_reexecuted(a, b, shape, *index, rows, &digests[*index as usize]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random matrix, values in roughly [-1, 1].
    fn matrix(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = splitmix64(state);
                ((state >> 40) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    /// The straightforward kernel: accumulate each dot product in order.
    fn multiply(a: &[f32], b: &[f32], shape: Shape) -> Vec<f32> {
        let mut c = vec![0.0f32; shape.rows * shape.cols];
        for i in 0..shape.rows {
            for k in 0..shape.inner {
                let scale = a[i * shape.inner + k];
                for j in 0..shape.cols {
                    c[i * shape.cols + j] += scale * b[k * shape.cols + j];
                }
            }
        }
        c
    }

    /// A second honest kernel summing in a different order, the way another
    /// vendor's GEMM would. Same answer, different last bits.
    fn multiply_blocked(a: &[f32], b: &[f32], shape: Shape) -> Vec<f32> {
        const BLOCK: usize = 32;
        let mut c = vec![0.0f32; shape.rows * shape.cols];
        for k0 in (0..shape.inner).step_by(BLOCK) {
            let k1 = (k0 + BLOCK).min(shape.inner);
            for i in 0..shape.rows {
                for j in 0..shape.cols {
                    let mut acc = 0.0f32;
                    for k in k0..k1 {
                        acc += a[i * shape.inner + k] * b[k * shape.cols + j];
                    }
                    c[i * shape.cols + j] += acc;
                }
            }
        }
        c
    }

    fn shape() -> Shape {
        Shape {
            rows: 64,
            inner: 256,
            cols: 128,
        }
    }

    /// Two honest kernels that disagree in the last bits must both pass, or
    /// the network would reject every node whose hardware differs from the
    /// coordinator's - which is every node.
    #[test]
    fn honest_kernels_that_differ_in_the_last_bits_both_pass() {
        let s = shape();
        let a = matrix(0xA11CE, s.rows * s.inner);
        let b = matrix(0xB0BB, s.inner * s.cols);
        let plain = multiply(&a, &b, s);
        let blocked = multiply_blocked(&a, &b, s);
        assert_ne!(plain, blocked, "the two kernels must not be bit-identical");
        assert!(witnessed(&a, &b, &plain, s, 0xDEC0DE));
        assert!(witnessed(&a, &b, &blocked, s, 0xDEC0DE));
    }

    /// The tolerance has to sit in a gap, not on a knife edge. Honest drift
    /// must be orders of magnitude below it, or the constant is a guess.
    #[test]
    fn honest_drift_sits_far_below_the_tolerance() {
        let s = shape();
        let a = matrix(0xA11CE, s.rows * s.inner);
        let b = matrix(0xB0BB, s.inner * s.cols);
        let drift = witness_residual(&a, &b, &multiply_blocked(&a, &b, s), s, 0xDEC0DE);
        assert!(
            drift * 20.0 < TOLERANCE,
            "honest drift {drift:e} leaves no headroom under {TOLERANCE}"
        );
    }

    /// The cheapest real cheat: run the accumulation over part of the inner
    /// dimension and stop early. It saves proportional time and moves the
    /// answer far more than rounding ever does.
    #[test]
    fn stopping_the_accumulation_early_is_caught() {
        let s = shape();
        let a = matrix(0xA11CE, s.rows * s.inner);
        let b = matrix(0xB0BB, s.inner * s.cols);
        for kept in [255usize, 248, 224, 128] {
            let short = Shape { inner: kept, ..s };
            let trimmed: Vec<f32> = (0..s.rows)
                .flat_map(|i| a[i * s.inner..i * s.inner + kept].iter().copied())
                .collect();
            let lazy = multiply(&trimmed, &b[..kept * s.cols], short);
            let residual = witness_residual(&a, &b, &lazy, s, 0xDEC0DE);
            assert!(
                residual > TOLERANCE,
                "skipping to {kept}/256 left residual {residual}, under {TOLERANCE}"
            );
        }
    }

    /// Witnessing has to be cheap enough that every validator can afford it on
    /// every shard, which is the whole reason the tier exists.
    #[test]
    fn witnessing_costs_far_less_than_the_product() {
        let s = shape();
        let ratio = s.compute_ops() as f64 / (s.witness_ops() * ROUNDS as u64) as f64;
        assert!(ratio > 4.0, "witness saves only {ratio:.1}x");
    }

    /// A commitment has to bind every bit of the answer, including the ones
    /// too small to change how the answer reads.
    #[test]
    fn the_commitment_binds_every_bit_of_the_product() {
        let s = shape();
        let a = matrix(0xA11CE, s.rows * s.inner);
        let b = matrix(0xB0BB, s.inner * s.cols);
        let product = multiply(&a, &b, s);
        let pinned = commit(&product);
        let mut nudged = product.clone();
        nudged[17] = f32::from_bits(nudged[17].to_bits() ^ 1);
        assert_ne!(commit(&nudged), pinned, "a one-bit change must show");
        assert_eq!(commit(&product), pinned, "the digest must be stable");
    }

    /// The whole point of committing first: once the challenge is public, the
    /// provider is stuck with the matrix it already signed for. A payload that
    /// would sail through the witness is refused because it is the wrong one.
    #[test]
    fn a_payload_swapped_after_the_challenge_is_refused() {
        let s = shape();
        let a = matrix(0xA11CE, s.rows * s.inner);
        let b = matrix(0xB0BB, s.inner * s.cols);
        let honest = multiply(&a, &b, s);
        let other = multiply_blocked(&a, &b, s);
        let pinned = commit(&honest);
        assert!(witnessed_committed(&a, &b, &honest, &pinned, s, 0xDEC0DE));
        assert!(
            witnessed(&a, &b, &other, s, 0xDEC0DE),
            "the swapped payload is itself a correct product"
        );
        assert!(
            !witnessed_committed(&a, &b, &other, &pinned, s, 0xDEC0DE),
            "only the committed payload may be judged"
        );
    }

    /// A ledger that carried whole products would outgrow the work it records.
    /// Committing keeps an entry the same size whatever the shard.
    #[test]
    fn committing_keeps_a_ledger_entry_small() {
        let s = shape();
        let payload = s.rows * s.cols * size_of::<f32>();
        let entry = 64;
        assert!(
            payload / entry > 500,
            "a commitment must be much smaller than {payload} bytes"
        );
    }

    /// The reveal is checked by re-executing it, so a kernel that sums in a
    /// different order than the validator's must still be accepted.
    #[test]
    fn an_honest_reveal_survives_re_execution() {
        let s = shape();
        let a = matrix(0xa11, s.rows * s.inner);
        let b = matrix(0xb22, s.inner * s.cols);
        let product = multiply_blocked(&a, &b, s);
        let commitments = block_commitments(&product, s);
        for index in opened_blocks(crate::verify::AuditNonce::draw(7)) {
            let revealed = block_payload(&product, s, index);
            assert!(
                block_reexecuted(&a, &b, s, index, revealed, &commitments[index as usize]),
                "block {index} was computed honestly and must be accepted"
            );
        }
    }

    /// The whole point of committing block by block: a provider that never did
    /// the work has to commit to something, and whatever it commits to cannot
    /// survive being re-executed.
    #[test]
    fn a_block_that_was_never_computed_cannot_be_revealed() {
        let s = shape();
        let a = matrix(0xa11, s.rows * s.inner);
        let b = matrix(0xb22, s.inner * s.cols);
        let mut product = multiply(&a, &b, s);
        let (start, end) = block_rows(s, 5);
        let invented = matrix(0xdead, (end - start) * s.cols);
        product[start * s.cols..end * s.cols].copy_from_slice(&invented);
        let commitments = block_commitments(&product, s);
        assert!(
            !block_reexecuted(&a, &b, s, 5, block_payload(&product, s, 5), &commitments[5]),
            "an invented block passed re-execution"
        );
    }

    /// Revealing the right answer for the wrong commitment is how a provider
    /// would compute the opened blocks only after seeing the challenge.
    #[test]
    fn a_reveal_that_does_not_match_its_commitment_is_refused() {
        let s = shape();
        let a = matrix(0xa11, s.rows * s.inner);
        let b = matrix(0xb22, s.inner * s.cols);
        let product = multiply(&a, &b, s);
        let honest = block_payload(&product, s, 9);
        let stale = commit(block_payload(&product, s, 10));
        assert!(!block_reexecuted(&a, &b, s, 9, honest, &stale));
    }

    /// Blocks that left a gap would be blocks nobody can be audited on.
    #[test]
    fn the_blocks_cover_every_element_exactly_once() {
        let s = shape();
        let product = matrix(0xc33, s.rows * s.cols);
        let mut seen = Vec::new();
        for index in 0..BLOCKS {
            seen.extend_from_slice(block_payload(&product, s, index));
        }
        assert_eq!(seen, product, "the blocks must reassemble the shard");
        assert_eq!(block_commitments(&product, s).len(), BLOCKS as usize);
    }

    /// Skipping work has to cost the same escape rate the prime audit charges,
    /// or float shards would be the cheap place to cheat.
    #[test]
    fn skipping_blocks_is_caught_at_the_sampled_audit_rate() {
        let s = shape();
        let a = matrix(0xa11, s.rows * s.inner);
        let b = matrix(0xb22, s.inner * s.cols);
        let honest = multiply(&a, &b, s);
        let skipped = 16u32;
        let mut lazy = honest.clone();
        let junk = matrix(0xbad, s.rows * s.cols);
        // The lazy provider computed the first `BLOCKS - skipped` blocks and
        // filled the rest with something it never multiplied.
        for index in (BLOCKS - skipped)..BLOCKS {
            let (start, end) = block_rows(s, index);
            let span = start * s.cols..end * s.cols;
            lazy[span.clone()].copy_from_slice(&junk[span]);
        }
        let commitments = block_commitments(&lazy, s);
        let trials = 4_000u32;
        let mut escapes = 0u32;
        for seed in 0..trials {
            let opened = opened_blocks(crate::verify::AuditNonce::draw(u64::from(seed)));
            let caught = opened.iter().any(|index| {
                let revealed = block_payload(&lazy, s, *index);
                !block_reexecuted(&a, &b, s, *index, revealed, &commitments[*index as usize])
            });
            escapes += u32::from(!caught);
        }
        let measured = f64::from(escapes) / f64::from(trials);
        // C(48,3)/C(64,3): every opened block lands in the honest part.
        let predicted = (48.0 / 64.0) * (47.0 / 63.0) * (46.0 / 62.0);
        assert!(
            (measured - predicted).abs() < 0.03,
            "measured {measured:.4} against predicted {predicted:.4}"
        );
    }

    /// The reveal has to be small, or the audit costs what shipping the whole
    /// product costs and the commitment bought nothing.
    #[test]
    fn the_reveal_is_a_small_slice_of_the_payload() {
        let s = shape();
        let product = matrix(0xc33, s.rows * s.cols);
        let opened = opened_blocks(crate::verify::AuditNonce::draw(11));
        let revealed: usize = opened
            .iter()
            .map(|index| block_payload(&product, s, *index).len())
            .sum();
        assert!(
            revealed * 10 < product.len(),
            "reveal {revealed} of {}",
            product.len()
        );
    }

    /// One helper for the whole reveal round, so the tests below only vary the
    /// thing they are actually testing.
    fn reveal_for(
        product: &[f32],
        s: Shape,
        nonce: crate::verify::AuditNonce,
    ) -> (String, Vec<String>, Vec<&[f32]>) {
        let digests = block_commitments(product, s);
        let root = commit_blocks(&digests);
        let revealed = opened_blocks(nonce)
            .into_iter()
            .map(|index| block_payload(product, s, index))
            .collect();
        (root, digests, revealed)
    }

    /// The round a validator actually runs: a 64-byte root from the entry, a
    /// nonce it derived itself, and a reveal it does not trust.
    #[test]
    fn an_honest_shard_survives_the_whole_reveal_round() {
        let s = shape();
        let a = matrix(0xa11, s.rows * s.inner);
        let b = matrix(0xb22, s.inner * s.cols);
        let product = multiply_blocked(&a, &b, s);
        for seed in [1u64, 2, 3, 99] {
            let nonce = crate::verify::AuditNonce::draw(seed);
            let (root, digests, revealed) = reveal_for(&product, s, nonce);
            assert!(reveal_accepted(
                &a, &b, s, &root, &digests, nonce, &revealed
            ));
        }
    }

    /// Swapping the digest list for one that matches a shard the provider did
    /// compute is the obvious attack on a root that only commits to a list.
    #[test]
    fn a_digest_list_that_does_not_reproduce_the_root_is_refused() {
        let s = shape();
        let a = matrix(0xa11, s.rows * s.inner);
        let b = matrix(0xb22, s.inner * s.cols);
        let product = multiply(&a, &b, s);
        let nonce = crate::verify::AuditNonce::draw(4);
        let (root, mut digests, revealed) = reveal_for(&product, s, nonce);
        digests[0] = commit(&[0.0f32]);
        assert!(!reveal_accepted(
            &a, &b, s, &root, &digests, nonce, &revealed
        ));
    }

    /// A provider that computed the shard for a different nonce cannot reuse
    /// the reveal, because the challenge names different blocks.
    #[test]
    fn a_reveal_for_the_wrong_nonce_is_refused() {
        let s = shape();
        let a = matrix(0xa11, s.rows * s.inner);
        let b = matrix(0xb22, s.inner * s.cols);
        let product = multiply(&a, &b, s);
        let asked = crate::verify::AuditNonce::draw(5);
        let answered = crate::verify::AuditNonce::draw(6);
        assert_ne!(opened_blocks(asked), opened_blocks(answered));
        let (root, digests, stale) = reveal_for(&product, s, answered);
        assert!(!reveal_accepted(&a, &b, s, &root, &digests, asked, &stale));
    }

    /// The entry stays small whatever the shard costs, which is the only
    /// reason committing beats shipping the product.
    #[test]
    fn the_entry_carries_one_digest_however_large_the_shard() {
        let s = shape();
        let product = matrix(0xc33, s.rows * s.cols);
        let root = commit_blocks(&block_commitments(&product, s));
        assert_eq!(root.len(), 64);
        let bigger = Shape {
            rows: s.rows,
            inner: s.inner,
            cols: s.cols * 2,
        };
        let wide = matrix(0xc34, bigger.rows * bigger.cols);
        let wide_root = commit_blocks(&block_commitments(&wide, bigger));
        assert_eq!(wide_root.len(), root.len());
        assert_ne!(wide_root, root);
    }

    /// Round to the mantissa a cheaper format carries, keeping f32's exponent.
    /// bf16 keeps 7 mantissa bits, fp16 and TF32 keep 10.
    fn truncated(value: f32, mantissa_bits: u32) -> f32 {
        let drop = 23 - mantissa_bits;
        let mask = !0u32 << drop;
        f32::from_bits((value.to_bits() + (1 << (drop - 1))) & mask)
    }

    /// A product computed as if the hardware carried a shorter mantissa: the
    /// operands are rounded going in and the running sum is rounded each step.
    fn low_precision(a: &[f32], b: &[f32], s: Shape, mantissa_bits: u32) -> Vec<f32> {
        let mut out = vec![0.0f32; s.rows * s.cols];
        for row in 0..s.rows {
            for col in 0..s.cols {
                let mut acc = 0.0f32;
                for k in 0..s.inner {
                    let left = truncated(a[row * s.inner + k], mantissa_bits);
                    let right = truncated(b[k * s.cols + col], mantissa_bits);
                    acc = truncated(acc + left * right, mantissa_bits);
                }
                out[row * s.cols + col] = acc;
            }
        }
        out
    }

    /// The cheat an inference provider would actually reach for: run the shard
    /// in bf16 or fp16, return it as f32, and pocket the speedup. It has to
    /// land outside the tolerance or the whole threshold is theatre.
    #[test]
    fn quietly_dropping_precision_is_caught() {
        let s = shape();
        let a = matrix(0xa11, s.rows * s.inner);
        let b = matrix(0xb22, s.inner * s.cols);
        let honest = multiply(&a, &b, s);
        // 7 mantissa bits is bf16, 10 is fp16 and TF32.
        for bits in [7u32, 10] {
            let cheap = low_precision(&a, &b, s, bits);
            let digests = block_commitments(&cheap, s);
            let opened: Vec<u32> = (0..BLOCKS).collect();
            let caught = opened.iter().any(|index| {
                let rows = block_payload(&cheap, s, *index);
                !block_reexecuted(&a, &b, s, *index, rows, &digests[*index as usize])
            });
            assert!(caught, "a {bits}-bit mantissa passed the block audit");
            let drift = witness_residual(&a, &b, &cheap, s, 0xDEC0DE);
            assert!(
                drift > TOLERANCE,
                "{bits}-bit mantissa drifted {drift:e}, inside the tolerance"
            );
        }
        // f32 done honestly stays far inside it, or the test above proves
        // nothing except that the threshold is too tight for everyone.
        assert!(witness_residual(&a, &b, &honest, s, 0xDEC0DE) < TOLERANCE);
    }

    /// A threshold is only a threshold if it keeps its margins at every shape
    /// the mesh would run. Honest f32 drift grows with the inner dimension,
    /// and so does the drift from a cheaper format - faster. This walks a 64x
    /// range and asserts the two populations never approach each other, which
    /// is the entire basis for [`TOLERANCE`] being one constant rather than a
    /// function of the shape.
    #[test]
    fn the_tolerance_holds_its_margins_at_every_shape() {
        const MARGIN: f32 = 20.0;
        for inner in [128usize, 1024, 8192] {
            let s = Shape {
                rows: 16,
                inner,
                cols: 64,
            };
            let a = matrix(0xa11, s.rows * s.inner);
            let b = matrix(0xb22, s.inner * s.cols);
            let honest = witness_residual(&a, &b, &multiply(&a, &b, s), s, 7);
            assert!(
                honest * MARGIN <= TOLERANCE,
                "at inner={inner} an honest f32 kernel drifted {honest:e}, \
                 leaving less than {MARGIN}x under {TOLERANCE}"
            );
            // 10 mantissa bits is the narrowest cheat worth running: fp16 and
            // TF32. bf16 keeps 7 and lands further out, so bounding fp16
            // bounds the whole family.
            let cheap = witness_residual(&a, &b, &low_precision(&a, &b, s, 10), s, 7);
            assert!(
                cheap >= TOLERANCE * MARGIN,
                "at inner={inner} an fp16 shard drifted only {cheap:e}, \
                 less than {MARGIN}x above {TOLERANCE}"
            );
        }
    }
}
