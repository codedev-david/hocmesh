//! Measured evidence that checking hocMESH work costs less than doing it.
//!
//! `cargo run --release -p hocmesh-core --example verification_proof`
//!
//! Every figure below is measured or simulated on the machine that runs this;
//! none of it is a constant copied out of the design notes.

use std::time::Instant;

use hocmesh_core::compute::{execute_work, work_cost_mcu};
use hocmesh_core::reputation::{Reputation, cheating_expected_value, minimum_slash_mcu};
use hocmesh_core::verify::{self, AUDIT_BUCKETS, AuditNonce, BUCKETS};
use hocmesh_protocol::{WorkResult, WorkSpec};

const PRIME: WorkSpec = WorkSpec::PrimeCount {
    start: 1,
    end: 3_000_000,
};
const MATMUL: WorkSpec = WorkSpec::MatrixMultiply {
    seed_a: 0xA11CE,
    seed_b: 0xB0BB,
    dim: 512,
    row_start: 0,
    row_end: 64,
};
const COLLATZ: WorkSpec = WorkSpec::CollatzPeak {
    start: 1,
    end: 400_000,
};

fn main() {
    rule("1. Checking costs less than computing (measured on this machine)");
    let prime = measure(&PRIME);
    let matmul = measure(&MATMUL);
    let collatz = measure(&COLLATZ);
    network_cost(&[&prime, &matmul, &collatz]);
    detection_sweep();
    collateral_vs_discount();
    grinding_cost();
}

struct Measured {
    label: &'static str,
    compute_ms: f64,
    witness_ms: f64,
}

fn label(work: &WorkSpec) -> &'static str {
    match work {
        WorkSpec::PrimeCount { .. } => "prime count",
        WorkSpec::MatrixMultiply { .. } => "matrix product",
        WorkSpec::CollatzPeak { .. } => "collatz peak",
    }
}

/// Time the honest computation, then time the witness check of its output.
fn measure(work: &WorkSpec) -> Measured {
    let started = Instant::now();
    let result = execute_work(work);
    let compute_ms = started.elapsed().as_secs_f64() * 1e3;

    // Average over several nonces: one audit is a few microseconds and would
    // otherwise be measuring the clock rather than the work.
    let rounds: u32 = 32;
    let started = Instant::now();
    for round in 0..rounds {
        let nonce = AuditNonce::replay(0x5EED_0000_u64 + u64::from(round));
        assert!(
            verify::witness_check(work, &result, nonce).is_accepted(),
            "honest work must survive every audit"
        );
    }
    let witness_ms = started.elapsed().as_secs_f64() * 1e3 / f64::from(rounds);

    // The predicted advantage is a price the whole network quotes; a model that
    // drifts away from the machine is a claim nobody measured. Order of
    // magnitude is all a static model can promise, so that is what is asserted.
    let measured = compute_ms / witness_ms;
    let predicted = verify::verification_advantage(work);
    let ratio = (measured / predicted).max(predicted / measured);
    assert!(
        ratio < 3.0,
        "{}: predicted {predicted:.0}x but measured {measured:.0}x",
        label(work)
    );

    println!(
        "{:<15} compute {:>9.2} ms   witness {:>8.3} ms   measured {:>7.0}x cheaper   \
         predicted {:>7.0}x",
        label(work),
        compute_ms,
        witness_ms,
        measured,
        predicted
    );
    Measured {
        label: label(work),
        compute_ms,
        witness_ms,
    }
}

fn rule(title: &str) {
    println!("\n{title}\n{}", "-".repeat(title.len()));
}

/// What one accepted shard costs the whole network, before and after.
///
/// Before: the coordinator recomputed the shard and so did every validator, so
/// the hocmesh burned `(V + 2)` times the compute it actually delivered.
fn network_cost(shards: &[&Measured]) {
    rule("2. Total network cost of one accepted shard (V = 3 validators)");
    let validators = 3.0;
    for m in shards.iter().copied() {
        let old = m.compute_ms * (validators + 2.0);
        // Now: the worker computes; the coordinator audits at the veteran rate;
        // every validator witnesses the entry it is replaying.
        let audit_rate = Reputation {
            accepted: 5_000,
            rejected: 0,
            streak: 5_000,
        }
        .audit_rate();
        let new = m.compute_ms + m.witness_ms * (audit_rate + validators);
        println!(
            "{:<15} was {:>9.2} ms ({:>4.2}x delivered)   now {:>9.2} ms ({:>5.3}x delivered)   \
             {:>5.1}x less waste",
            m.label,
            old,
            old / m.compute_ms,
            new,
            new / m.compute_ms,
            old / new
        );
    }
}

/// A shard small enough that thousands of audits can be replayed against it.
const SWEEP: WorkSpec = WorkSpec::PrimeCount {
    start: 1,
    end: 100_000,
};

/// How many trials each cheat level gets in the sweep.
const TRIALS: u64 = 2_000;

/// A slash worth this many shard payments is the policy under test.
const POLICY_SLASH: i64 = 25;

/// The result a lazy worker would actually submit.
///
/// It computes the buckets it kept and guesses the rest from a neighbour, then
/// reports the sum of its own buckets — so the free arithmetic check passes and
/// only a recount of a skipped bucket can expose it. This is the strongest
/// cheat available, not a strawman.
fn lazy_prime(honest: &WorkResult, skipped: u32) -> WorkResult {
    let WorkResult::PrimeCount { bucket_counts, .. } = honest else {
        unreachable!("the sweep runs on prime work")
    };
    let mut counts = bucket_counts.clone();
    let nonce = AuditNonce::replay(0xC0FF_EE00 ^ u64::from(skipped));
    for index in verify::audit_indices(nonce, BUCKETS, skipped) {
        let donor = bucket_counts[((index + 1) % BUCKETS) as usize];
        counts[index as usize] = donor;
    }
    let count = counts.iter().sum();
    WorkResult::PrimeCount {
        count,
        bucket_counts: counts,
        duration_ms: 0,
    }
}

/// `C(n, k)` as a float; the sets involved here are far too small to overflow.
fn choose(n: u32, k: u32) -> f64 {
    if k > n {
        return 0.0;
    }
    (0..k)
        .map(|i| f64::from(n - i) / f64::from(i + 1))
        .product()
}

/// Replay thousands of real audits against a committed lazy result.
fn detection_sweep() {
    rule("3. Every level of cheating loses money");
    let honest = execute_work(&SWEEP);
    let reward = work_cost_mcu(&SWEEP);
    let veteran = Reputation {
        accepted: 5_000,
        rejected: 0,
        streak: 5_000,
    };
    let rate = veteran.audit_rate();
    println!(
        "shard pays {reward} mCU, slash = {POLICY_SLASH}x that, and the cheater is the most \
         trusted node hocMESH allows ({:.0}% audit rate)",
        rate * 100.0
    );
    println!(
        "{:>8}{:>7}{:>9}{:>10}{:>10}{:>12}{:>13}",
        "skipped", "wrong", "gain", "caught", "predicted", "min slash", "EV @ policy"
    );
    for skipped in [1u32, 2, 4, 8, 16, 32, 48, 64] {
        let lazy = lazy_prime(&honest, skipped);
        let caught = (0..TRIALS)
            .filter(|trial| {
                let nonce = AuditNonce::replay(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(trial + 1));
                !verify::witness_check(&SWEEP, &lazy, nonce).is_accepted()
            })
            .count();
        let caught_if_audited = caught as f64 / TRIALS as f64;
        // A guessed bucket that happens to match the truth is indistinguishable
        // from honest work, so the prediction is over the buckets it got wrong.
        let wrong = wrong_buckets(&honest, &lazy);
        let predicted =
            1.0 - choose(BUCKETS - wrong, AUDIT_BUCKETS) / choose(BUCKETS, AUDIT_BUCKETS);
        let detection = rate * caught_if_audited;
        let gain = (reward * i64::from(skipped)) / i64::from(BUCKETS);
        let ev = cheating_expected_value(gain, reward * POLICY_SLASH, detection);
        let floor = minimum_slash_mcu(gain, detection);
        println!(
            "{:>6}/64{:>7}{:>9}{:>8.1}%{:>9.1}%{:>8} mCU{:>9.1} mCU",
            skipped,
            wrong,
            gain,
            caught_if_audited * 100.0,
            predicted * 100.0,
            floor,
            ev
        );
        assert!(ev < 0.0, "cheating must never pay");
    }
}

/// The trust discount is only safe if collateral outgrows it.
///
/// A falling audit rate raises the slash needed to deter cheating. The same
/// clean history that earns the discount also banks the CU that pays it.
fn collateral_vs_discount() {
    rule("4. Collateral outgrows the trust discount");
    let reward = work_cost_mcu(&SWEEP);
    let catch = 1.0 - choose(BUCKETS - 1, AUDIT_BUCKETS) / choose(BUCKETS, AUDIT_BUCKETS);
    println!("worst case: the cheat is the smallest one possible, a single skipped bucket");
    println!(
        "{:>10}{:>9}{:>14}{:>14}{:>9}",
        "accepted", "audit", "slash needed", "banked", "margin"
    );
    for accepted in [0u64, 1, 6, 12, 24, 48, 120, 600, 5_000] {
        let node = Reputation {
            accepted,
            rejected: 0,
            streak: accepted as u32,
        };
        let detection = node.audit_rate() * catch;
        let gain = reward / i64::from(BUCKETS);
        let needed = minimum_slash_mcu(gain, detection);
        // The shard being settled is itself held until the audit clears, so a
        // node with no history still has one payment at stake.
        let banked = (accepted as i64 + 1) * reward;
        println!(
            "{accepted:>10}{:>8.0}%{needed:>10} mCU{banked:>10} mCU{:>8.0}x",
            node.audit_rate() * 100.0,
            banked as f64 / needed as f64
        );
        assert!(
            banked >= needed,
            "the discount must never outrun the collateral"
        );
    }
    println!("\nEvery figure above was produced by the code under test, not asserted by hand.");
}

/// How many buckets the lazy result actually got wrong.
fn wrong_buckets(honest: &WorkResult, lazy: &WorkResult) -> u32 {
    let (
        WorkResult::PrimeCount {
            bucket_counts: truth,
            ..
        },
        WorkResult::PrimeCount {
            bucket_counts: claimed,
            ..
        },
    ) = (honest, lazy)
    else {
        unreachable!("the sweep runs on prime work")
    };
    truth.iter().zip(claimed).filter(|(t, c)| t != c).count() as u32
}

/// Two independent challenges, and what it costs to grind past both.
///
/// A cheat must escape the propose-time draw (from the chain head) and the
/// apply-time beacon (from the quorum signatures). Neither is computable by a
/// coordinator in advance, so each retry is a public re-proposal.
fn grinding_cost() {
    rule("4. Grinding the audit costs public rounds, not CPU cycles");
    let honest = execute_work(&SWEEP);
    println!("  skipped   escape 1   escape 2   escape both   predicted   public rounds");
    for skipped in [4u32, 8, 16, 32, 64] {
        let lazy = lazy_prime(&honest, skipped);
        let (mut one, mut two, mut both) = (0u64, 0u64, 0u64);
        for n in 0..TRIALS {
            let head = format!("head-{n}");
            let tx = format!("tx-{n}");
            let quorum = [format!("a-{n}"), format!("b-{n}"), format!("c-{n}")];
            let signed: Vec<&str> = quorum.iter().map(String::as_str).collect();
            let chain = AuditNonce::for_entry(&head, &tx);
            let beacon = AuditNonce::for_certified_entry(&head, &tx, &signed);
            let a = verify::witness_check(&SWEEP, &lazy, chain).is_accepted();
            let b = verify::witness_check(&SWEEP, &lazy, beacon).is_accepted();
            one += u64::from(a);
            two += u64::from(b);
            both += u64::from(a && b);
        }
        let rate = |hits: u64| hits as f64 / TRIALS as f64;
        let (e1, e2, ej) = (rate(one), rate(two), rate(both));
        let rounds = 1.0 / ej.max(1.0 / TRIALS as f64);
        println!(
            "  {skipped:>4}/64   {:>7.1}%   {:>7.1}%   {:>10.1}%   {:>8.1}%   {rounds:>12.1}",
            e1 * 100.0,
            e2 * 100.0,
            ej * 100.0,
            e1 * e2 * 100.0
        );
        assert!(
            (ej - e1 * e2).abs() < 0.05,
            "the two challenges must be independent draws"
        );
    }
    println!("\n  A coordinator cannot compute the beacon without validator keys, so every");
    println!("  extra attempt is a fresh quorum round - and a validator that has voted at a");
    println!("  sequence refuses a conflicting entry hash there. Grinding happens in public.");
}
