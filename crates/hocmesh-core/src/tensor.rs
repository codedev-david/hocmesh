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
/// Measured, not chosen: honest kernels differing only in summation order stay
/// four orders of magnitude below this, and the cheapest cheat worth running
/// lands two orders above it. See the `float_witness_proof` example.
pub const TOLERANCE: f32 = 1e-3;

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
            drift < TOLERANCE / 100.0,
            "honest drift {drift} leaves no headroom under {TOLERANCE}"
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
}
