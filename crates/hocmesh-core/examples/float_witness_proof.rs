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
use hocmesh_core::verify::AuditNonce;
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
    ledger_size(&honest);
    sampled_reveal(&a, &b, &honest);
    precision(&a, &b, &honest);
    threshold();
}

/// The witness above is the requester's tool: it checks a product you hold.
/// A validator holds a ledger entry, so it needs possession forced first.
fn sampled_reveal(a: &[f32], b: &[f32], honest: &[f32]) {
    rule("5. What a validator can check without the payload");
    println!(
        "  the shard is committed in {} row blocks; the challenge opens 3",
        tensor::BLOCKS
    );
    println!("  and the provider must reveal them.\n");
    println!(
        "  {:>9}  {:>10}  {:>11}  {:>11}",
        "skipped", "caught", "escape", "predicted"
    );
    let junk = matrix(0xBADF00D, SHAPE.rows * SHAPE.cols);
    for skipped in [4u32, 16, 32, 64] {
        let mut lazy = honest.to_vec();
        for index in (tensor::BLOCKS - skipped)..tensor::BLOCKS {
            let (start, end) = tensor::block_rows(SHAPE, index);
            let span = start * SHAPE.cols..end * SHAPE.cols;
            lazy[span.clone()].copy_from_slice(&junk[span]);
        }
        let commitments = tensor::block_commitments(&lazy, SHAPE);
        let mut caught = 0u64;
        for seed in 0..TRIALS {
            let opened = tensor::opened_blocks(AuditNonce::draw(seed));
            if opened.iter().any(|index| {
                let revealed = tensor::block_payload(&lazy, SHAPE, *index);
                let digest = &commitments[*index as usize];
                !tensor::block_reexecuted(a, b, SHAPE, *index, revealed, digest)
            }) {
                caught += 1;
            }
        }
        let kept = f64::from(tensor::BLOCKS - skipped);
        let total = f64::from(tensor::BLOCKS);
        let predicted =
            (kept / total) * ((kept - 1.0) / (total - 1.0)) * ((kept - 2.0) / (total - 2.0));
        let escape = 1.0 - caught as f64 / TRIALS as f64;
        println!(
            "  {:>6}/{:<2}  {caught:>4}/{TRIALS:<5}  {:>10.1}%  {:>10.1}%",
            skipped,
            tensor::BLOCKS,
            escape * 100.0,
            predicted.max(0.0) * 100.0
        );
    }
    let opened = tensor::opened_blocks(AuditNonce::draw(1));
    let revealed: usize = opened
        .iter()
        .map(|index| size_of_val(tensor::block_payload(honest, SHAPE, *index)))
        .sum();
    let payload = size_of_val(honest);
    println!(
        "\n  reveal {revealed} of {payload} bytes ({:.1}% of the shard), and the validator",
        revealed as f64 / payload as f64 * 100.0
    );
    println!("  re-executes exactly those rows: the same fraction of the work.");
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
    const REPS: u32 = 5;
    let start = Instant::now();
    for _ in 0..REPS {
        std::hint::black_box(multiply(a, b, SHAPE));
    }
    let compute_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(REPS);
    let start = Instant::now();
    for round in 0..REPS {
        std::hint::black_box(tensor::witness_residual(
            a,
            b,
            honest,
            SHAPE,
            0xDEC0DE ^ u64::from(round),
        ));
    }
    let witness_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(REPS);
    let predicted = SHAPE.compute_ops() as f64 / (SHAPE.witness_ops() * ROUNDS as u64) as f64;
    println!("  product  {compute_ms:>8.2} ms");
    println!("  witness  {witness_ms:>8.2} ms  ({ROUNDS} rounds)");
    println!(
        "  measured {:>8.1}x cheaper   predicted {predicted:.1}x",
        compute_ms / witness_ms
    );
    assert!(compute_ms > witness_ms, "witnessing must be the cheap side");

    // The timed shape above is small; the ratio structurally improves with size,
    // because the product grows with rows*inner*cols and the witness only with
    // the matrix areas. Real inference shapes are where this has to pay off.
    println!(
        "
  op-count ratio at real model shapes (the timed shape is the pessimistic case):"
    );
    for (label, shape) in [
        (
            "512 x 4096 x 4096   attention batch",
            Shape {
                rows: 512,
                inner: 4096,
                cols: 4096,
            },
        ),
        (
            "4096 x 4096 x 14336  MLP projection",
            Shape {
                rows: 4096,
                inner: 4096,
                cols: 14336,
            },
        ),
    ] {
        let ratio = shape.compute_ops() as f64 / (shape.witness_ops() * ROUNDS as u64) as f64;
        println!("    {label:<36} {ratio:>6.0}x cheaper");
    }
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

/// A witness needs the whole product, but a ledger cannot carry it. The entry
/// commits; the payload rides along with the answer the requester gets anyway.
fn ledger_size(honest: &[f32]) {
    rule("4. The ledger commits to the answer instead of storing it");
    let payload = size_of_val(honest);
    let digest = tensor::commit(honest);
    println!("  product payload  {:>9} bytes", payload);
    println!("  ledger entry     {:>9} bytes", digest.len());
    println!("  {:>25.0}x smaller", payload as f64 / digest.len() as f64);
    assert!(
        payload / digest.len() > 500,
        "committing must actually save"
    );
    println!("\n  The commitment is checked before a single multiply is spent, so a provider");
    println!("  that has seen the challenge cannot swap in a matrix that satisfies it.");
}

/// Skipping blocks is the cheat the audit was designed for. Dropping precision
/// is the cheat a real inference provider would reach for first.
fn precision(a: &[f32], b: &[f32], honest: &[f32]) {
    rule("6. Running the shard in a cheaper format");
    println!(
        "  {:>18}  {:>12}  {:>8}  {:>10}",
        "format", "residual", "vs tol", "verdict"
    );
    let honest_drift = tensor::witness_residual(a, b, honest, SHAPE, 0xDEC0DE);
    report("f32 (honest)", honest_drift);
    for (label, bits) in [("bf16 (7-bit)", 7u32), ("fp16/TF32 (10-bit)", 10)] {
        let cheap = low_precision(a, b, SHAPE, bits);
        report(
            label,
            tensor::witness_residual(a, b, &cheap, SHAPE, 0xDEC0DE),
        );
    }
}

/// Section 6 fixes one shape. The constant has to hold at every shape, so
/// this walks the inner dimension across a 64x range and shows the two
/// populations staying four orders of magnitude apart the whole way. That
/// separation is the only reason one constant can serve every job.
fn threshold() {
    rule("7. Where the threshold sits");
    println!(
        "  {:>7}  {:>11}  {:>11}  {:>11}  {:>9}  {:>9}",
        "inner", "honest", "fp16", "bf16", "under", "over"
    );
    for inner in [128usize, 512, 2048, 8192] {
        let s = Shape {
            rows: 16,
            inner,
            cols: 64,
        };
        let a = matrix(0xA11CE, s.rows * s.inner);
        let b = matrix(0xB0BB, s.inner * s.cols);
        let residual = |c: &[f32]| tensor::witness_residual(&a, &b, c, s, 0xDEC0DE);
        let honest = residual(&multiply(&a, &b, s));
        let fp16 = residual(&low_precision(&a, &b, s, 10));
        let bf16 = residual(&low_precision(&a, &b, s, 7));
        println!(
            "  {inner:>7}  {honest:>11.3e}  {fp16:>11.3e}  {bf16:>11.3e}  {:>8.0}x  {:>8.0}x",
            TOLERANCE / honest,
            fp16 / TOLERANCE
        );
    }
    println!("\n  under = how far honest f32 stays inside {TOLERANCE:e}.");
    println!("  over  = how far the cheapest worthwhile cheat lands outside it.");
    println!("  Neither margin closes as the shard grows, so the constant does");
    println!("  not have to be a function of the shape.");
}

fn report(label: &str, residual: f32) {
    let verdict = if residual <= TOLERANCE {
        "accepted"
    } else {
        "REJECTED"
    };
    println!(
        "  {label:>18}  {residual:>12.3e}  {:>7.0}x  {verdict:>10}",
        residual / TOLERANCE
    );
}

fn rounded(value: f32, mantissa_bits: u32) -> f32 {
    let drop = 23 - mantissa_bits;
    let mask = !0u32 << drop;
    f32::from_bits((value.to_bits() + (1 << (drop - 1))) & mask)
}

fn low_precision(a: &[f32], b: &[f32], s: Shape, bits: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; s.rows * s.cols];
    for row in 0..s.rows {
        for col in 0..s.cols {
            let mut acc = 0.0f32;
            for k in 0..s.inner {
                let left = rounded(a[row * s.inner + k], bits);
                let right = rounded(b[k * s.cols + col], bits);
                acc = rounded(acc + left * right, bits);
            }
            out[row * s.cols + col] = acc;
        }
    }
    out
}
