//! Proof that floating-point work can be witnessed - which is what decides
//! whether hocMESH can verify AI inference or only toy integer workloads.
//!
//! Run it:
//!
//!     cargo run --release -p hocmesh-core --example float_witness_proof
//!
//! Transformer inference is a stack of matrix products, so if a matrix product
//! can be checked in O(n^2) without rejecting honest hardware, the whole
//! workload can. Everything below is measured, and every claim is asserted, so
//! this example fails rather than overstates.

use hocmesh_core::tensor::{self, ROUNDS, Shape, TOLERANCE};
use std::time::Instant;

/// A projection the size of one attention head's output in a small model.
const SHAPE: Shape = Shape {
    rows: 128,
    inner: 512,
    cols: 512,
};

const TRIALS: u64 = 200;

fn main() {
    let a = matrix(0xA11CE, SHAPE.rows * SHAPE.inner);
    let b = matrix(0xB0BB, SHAPE.inner * SHAPE.cols);
    let honest = multiply(&a, &b, SHAPE);
    let other_kernel = multiply_blocked(&a, &b, SHAPE);
    the_gap(&a, &b, &honest, &other_kernel);
    detection(&a, &b, &honest);
    cost(&a, &b, &honest);
}

/// The tolerance has to separate two populations. Show both.
fn the_gap(a: &[f32], b: &[f32], honest: &[f32], other: &[f32]) {
    rule("1. Honest hardware disagrees; cheating disagrees far more");
    let drift = tensor::witness_residual(a, b, other, SHAPE, 0xDEC0DE);
    println!("  two honest kernels, different summation order   {drift:>12.3e}");
    println!("  tolerance                                       {TOLERANCE:>12.3e}");
    for kept in [511usize, 504, 448, 256] {
        let lazy = truncated(a, b, kept);
        let residual = tensor::witness_residual(a, b, &lazy, SHAPE, 0xDEC0DE);
        let saved = 100.0 - (kept as f64 * 100.0 / SHAPE.inner as f64);
        println!(
            "  stopped at {kept:>3}/512 ({saved:>4.1}% of the work skipped) {residual:>12.3e}"
        );
    }
    let headroom = TOLERANCE / drift;
    println!("\n  the tolerance sits {headroom:.0}x above honest drift");
    assert!(
        headroom > 100.0,
        "the tolerance must not sit on a knife edge"
    );
    assert!(tensor::witnessed(a, b, honest, SHAPE, 0xDEC0DE));
    assert!(tensor::witnessed(a, b, other, SHAPE, 0xDEC0DE));
}

/// A witness is worthless if it only catches cheats at one lucky challenge.
/// Draw the challenge fresh every trial, the way the beacon does.
fn detection(a: &[f32], b: &[f32], honest: &[f32]) {
    rule("2. Detection holds across independent challenges");
    println!("     work skipped   caught   honest wrongly rejected");
    for kept in [511usize, 504, 448, 256] {
        let lazy = truncated(a, b, kept);
        let mut caught = 0u64;
        let mut rejected = 0u64;
        for trial in 0..TRIALS {
            let nonce = 0x5EED_0000 ^ trial.wrapping_mul(0x9E37_79B9);
            if !tensor::witnessed(a, b, &lazy, SHAPE, nonce) {
                caught += 1;
            }
            if !tensor::witnessed(a, b, honest, SHAPE, nonce) {
                rejected += 1;
            }
        }
        let skipped = 100.0 - (kept as f64 * 100.0 / SHAPE.inner as f64);
        let rate = caught as f64 * 100.0 / TRIALS as f64;
        println!("     {skipped:>10.1}%   {rate:>5.1}%   {rejected:>10} of {TRIALS}");
        assert_eq!(caught, TRIALS, "a cheat escaped at {kept}/512");
        assert_eq!(rejected, 0, "honest work was rejected at {kept}/512");
    }
    println!("\n  Skipping even one of 512 accumulation steps is caught every time.");
}

/// The tier only works if every validator can afford it on every shard.
fn cost(a: &[f32], b: &[f32], honest: &[f32]) {
    rule("3. Witnessing is cheap enough to run on every shard");
    let start = Instant::now();
    let product = multiply(a, b, SHAPE);
    let compute_ms = start.elapsed().as_secs_f64() * 1000.0;
    std::hint::black_box(&product);
    let start = Instant::now();
    let residual = tensor::witness_residual(a, b, honest, SHAPE, 0xDEC0DE);
    let witness_ms = start.elapsed().as_secs_f64() * 1000.0;
    std::hint::black_box(residual);
    let predicted = SHAPE.compute_ops() as f64 / (SHAPE.witness_ops() * ROUNDS as u64) as f64;
    println!("  product  {compute_ms:>8.2} ms");
    println!("  witness  {witness_ms:>8.2} ms  ({ROUNDS} rounds)");
    println!(
        "  measured {:>8.1}x cheaper   predicted {predicted:.1}x",
        compute_ms / witness_ms
    );
    assert!(compute_ms > witness_ms, "witnessing must be the cheap side");
}

fn rule(title: &str) {
    println!("\n{title}\n{}", "-".repeat(title.len()));
}

/// A product computed over only the first `kept` steps of the inner dimension:
/// proportionally faster, and wrong by more than rounding.
fn truncated(a: &[f32], b: &[f32], kept: usize) -> Vec<f32> {
    let short = Shape {
        inner: kept,
        ..SHAPE
    };
    let trimmed: Vec<f32> = (0..SHAPE.rows)
        .flat_map(|i| a[i * SHAPE.inner..i * SHAPE.inner + kept].iter().copied())
        .collect();
    multiply(&trimmed, &b[..kept * SHAPE.cols], short)
}

fn matrix(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = splitmix64(state);
            ((state >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

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

/// Another vendor's kernel: same maths, different summation order.
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
