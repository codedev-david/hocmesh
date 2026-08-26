//! Deterministic modular matrix arithmetic.
//!
//! This exists to give MESH a workload whose answers can be *checked* far more
//! cheaply than they can be *produced*. Prime counting cannot do that: the only
//! way to confirm a prime count is to count the primes again, so every verifier
//! pays what the worker paid. Matrix products can do it, via Freivalds' test, at
//! `O(n^2)` instead of `O(n^3)`.
//!
//! Both operands are generated from a 64-bit seed rather than shipped, so a job
//! spec stays a few dozen bytes while describing billions of operations. That
//! ratio — compute per transmitted byte — is what makes distribution worthwhile
//! in the first place.

/// Arithmetic happens modulo the Mersenne prime `2^31 - 1`.
///
/// A prime modulus is what makes Freivalds' test sound: the verifier's random
/// vector is drawn from a *field*, so a wrong product survives the check with
/// probability at most `1/MODULUS`. A composite modulus would leave zero
/// divisors for a cheater to hide in.
pub const MODULUS: u64 = 2_147_483_647;

/// The largest square matrix a single job may describe.
///
/// Bounded so one shard cannot ask a worker for unbounded memory, and so the
/// result of a shard stays a sane size on the wire.
pub const MAX_DIM: u32 = 512;

/// One round of SplitMix64.
///
/// Chosen because it is a bijection with good avalanche from a single word of
/// state, which is what lets [`element`] address any cell of a virtual matrix
/// independently instead of generating the whole thing in order.
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The cell at (`row`, `col`) of the virtual matrix identified by `seed`.
///
/// Random access matters: a verifier running Freivalds' test touches whole rows
/// of `A` and whole columns of `B` in an order that has nothing to do with how a
/// worker walked them. Deriving each cell independently means neither side has
/// to materialise a matrix it only partly needs.
pub fn element(seed: u64, row: u32, col: u32) -> u32 {
    let mixed = splitmix64(
        seed ^ (row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (col as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F),
    );
    (mixed % MODULUS) as u32
}

/// Materialise one row of the virtual matrix `seed`.
pub fn row(seed: u64, row_index: u32, dim: u32) -> Vec<u32> {
    (0..dim).map(|col| element(seed, row_index, col)).collect()
}

/// Multiply `a * b` modulo [`MODULUS`], accumulating in 64 bits.
///
/// Every factor is below `2^31`, so a product fits in 62 bits and a running sum
/// is reduced each step; nothing here can overflow a `u64`.
#[inline]
fn mul_mod(a: u32, b: u32) -> u64 {
    (a as u64 * b as u64) % MODULUS
}

/// Compute rows `[row_start, row_end)` of `A * B (mod MODULUS)`.
///
/// This is the honest `O(rows * dim^2)` path — the work a provider is actually
/// being paid for.
pub fn multiply_rows(seed_a: u64, seed_b: u64, dim: u32, row_start: u32, row_end: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(((row_end - row_start) as usize) * dim as usize);
    // Hoisting B into memory once turns the inner loop into a stride-1 walk;
    // regenerating each cell inside it would dominate the multiply itself.
    let b: Vec<u32> = (0..dim)
        .flat_map(|k| (0..dim).map(move |j| element(seed_b, k, j)))
        .collect();

    for i in row_start..row_end {
        let a_row = row(seed_a, i, dim);
        let mut acc = vec![0u64; dim as usize];
        for (k, &a_ik) in a_row.iter().enumerate() {
            if a_ik == 0 {
                continue;
            }
            let b_row = &b[k * dim as usize..(k + 1) * dim as usize];
            for (slot, &b_kj) in acc.iter_mut().zip(b_row.iter()) {
                *slot = (*slot + mul_mod(a_ik, b_kj)) % MODULUS;
            }
        }
        out.extend(acc.into_iter().map(|v| v as u32));
    }
    out
}

/// Multiply the virtual matrix `seed` by the dense vector `vector`.
pub fn matrix_vector(seed: u64, dim: u32, vector: &[u32]) -> Vec<u32> {
    (0..dim)
        .map(|r| {
            let mut acc = 0u64;
            for (col, &v) in vector.iter().enumerate() {
                if v != 0 {
                    acc = (acc + mul_mod(element(seed, r, col as u32), v)) % MODULUS;
                }
            }
            acc as u32
        })
        .collect()
}

/// Dot product of two dense vectors modulo [`MODULUS`].
pub fn dot(left: &[u32], right: &[u32]) -> u32 {
    let mut acc = 0u64;
    for (&l, &r) in left.iter().zip(right.iter()) {
        acc = (acc + mul_mod(l, r)) % MODULUS;
    }
    acc as u32
}

/// A pseudorandom challenge vector of `dim` field elements derived from `seed`.
///
/// Freivalds' soundness depends on this vector being unpredictable to whoever
/// produced the product, which is a property of *when* the seed is chosen, not
/// of this function. See `verify::audit_seed`.
pub fn challenge_vector(seed: u64, dim: u32) -> Vec<u32> {
    (0..dim)
        .map(|i| {
            (splitmix64(seed ^ (i as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)) % MODULUS) as u32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both sides of the network derive operands from the seed alone, so the
    /// same coordinates must give the same value on any machine, forever.
    #[test]
    fn a_cell_depends_only_on_its_seed_and_coordinates() {
        assert_eq!(element(7, 3, 9), element(7, 3, 9));
        assert_ne!(element(7, 3, 9), element(8, 3, 9));
        assert_ne!(element(7, 3, 9), element(7, 9, 3));
    }

    /// A cell that fell outside the field would break the soundness argument
    /// for Freivalds' test, which assumes arithmetic in `GF(MODULUS)`.
    #[test]
    fn every_cell_lands_inside_the_field() {
        for row_index in 0..16 {
            for col in 0..16 {
                assert!((element(42, row_index, col) as u64) < MODULUS);
            }
        }
    }

    /// The blocked inner loop in `multiply_rows` is an optimisation, so it has
    /// to agree with the textbook definition it replaced.
    #[test]
    fn the_fast_multiply_agrees_with_the_naive_one() {
        let (seed_a, seed_b, dim) = (11u64, 22u64, 24u32);
        let fast = multiply_rows(seed_a, seed_b, dim, 4, 8);
        let mut naive = Vec::new();
        for i in 4..8u32 {
            for j in 0..dim {
                let mut acc = 0u64;
                for k in 0..dim {
                    acc = (acc + mul_mod(element(seed_a, i, k), element(seed_b, k, j))) % MODULUS;
                }
                naive.push(acc as u32);
            }
        }
        assert_eq!(fast, naive, "the blocked multiply must be exact, not close");
    }

    /// `A(Br) == (AB)r` is the identity Freivalds' test rests on. If matrix and
    /// vector arithmetic ever disagreed here, honest work would be rejected.
    #[test]
    fn multiplication_is_associative_against_a_vector() {
        let (seed_a, seed_b, dim) = (5u64, 6u64, 20u32);
        let product = multiply_rows(seed_a, seed_b, dim, 0, dim);
        let r = challenge_vector(99, dim);
        let br = matrix_vector(seed_b, dim, &r);
        for i in 0..dim as usize {
            let via_product = dot(&product[i * dim as usize..(i + 1) * dim as usize], &r);
            let via_operands = dot(&row(seed_a, i as u32, dim), &br);
            assert_eq!(via_product, via_operands, "row {i} broke associativity");
        }
    }
}
