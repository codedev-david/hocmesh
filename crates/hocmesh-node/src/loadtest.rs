//! Artificial load, and why it is a shipped command rather than a benchmark.
//!
//! Throughput is the least interesting thing this measures. Every genuinely
//! hard bug this ledger has had was a race -- a stale head read, a winner whose
//! entry landed before its signed head was readable, a proposer refused by its
//! own committed work -- and none of them are reachable by one client doing one
//! thing at a time. They need contention, and contention is exactly what a
//! single developer machine never produces by hand.
//!
//! So a run here does two jobs at once. It reports latency and throughput,
//! which is what anyone expects of a load test. And it *audits the economy it
//! just stressed*: every CU the coordinator said it reserved has to be the CU
//! the account records as spent, and the account's own three numbers -- banked,
//! earned, consumed -- have to agree with each other afterwards. A run that
//! moves fast and loses a CU fails.
//!
//! That is the difference between a benchmark and a test. This is a test.

use crate::client::HocMeshClient;
use anyhow::{Context, Result, bail};
use hocmesh_protocol::WorkSpec;
use serde::Serialize;
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

/// Which of the three verifiable CPU workloads to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Workload {
    /// Longest Collatz trajectory in a range. Cheapest possible spec, pure
    /// integer arithmetic, and the shape most like real distributed search.
    Collatz,
    /// Count primes in a range.
    Prime,
    /// Rows of a matrix product generated from seeds, so the spec stays tiny
    /// while the arithmetic grows with `--size`. The one workload here whose
    /// cost is superlinear in its size knob.
    Matrix,
}

/// What to generate. Separated from the CLI so it can be exercised in tests
/// without a coordinator on the other end.
#[derive(Debug, Clone)]
pub struct LoadPlan {
    /// How many jobs to submit. `0` means "keep going until `duration`".
    pub jobs: u64,
    /// How many jobs may be in flight at once.
    pub concurrency: usize,
    /// Shards per job -- the parallelism the coordinator is asked to find.
    pub shards: u32,
    pub workload: Workload,
    /// The size knob: range width for Collatz and Prime, dimension for Matrix.
    pub size: u64,
    /// Stop submitting once this much time has passed, whatever `jobs` says.
    pub duration: Option<Duration>,
    /// How long any one job may take to finish before it is called a timeout.
    pub timeout: Duration,
    /// How often to ask the coordinator whether a job has finished.
    pub poll: Duration,
}

impl LoadPlan {
    pub fn validate(&self) -> Result<()> {
        if self.jobs == 0 && self.duration.is_none() {
            bail!("a load test needs --jobs, --duration-secs, or both")
        }
        if self.concurrency == 0 {
            bail!("--concurrency must be at least 1")
        }
        if self.shards == 0 {
            bail!("--shards must be at least 1")
        }
        if self.size == 0 {
            bail!("--size must be at least 1")
        }
        if self.workload == Workload::Matrix && self.size > 4096 {
            bail!(
                "--size {} for a matrix multiply is {} multiplications per row; \
                 pick something under 4096",
                self.size,
                self.size.saturating_mul(self.size)
            )
        }
        Ok(())
    }

    /// What one job of this plan costs, in mCU.
    ///
    /// Priced the way `submit` prices it -- split into shards first, then each
    /// shard priced -- rather than pricing the whole range once. Those differ by
    /// rounding, and a cost estimate that is a rounding error under the real
    /// charge is worse than none: it is the estimate that says you can afford a
    /// run you cannot.
    pub fn job_cost_mcu(&self, index: u64) -> i64 {
        hocmesh_core::compute::split_work(&self.spec(index), self.shards)
            .iter()
            .map(hocmesh_core::compute::work_cost_mcu)
            .sum()
    }

    /// What the whole run will cost, so a caller can be sure it can afford it
    /// before it starts. A duration-bounded run has no fixed total, so this
    /// reports one job's price for those.
    pub fn cost_mcu(&self) -> i64 {
        if self.jobs == 0 {
            return self.job_cost_mcu(0);
        }
        (0..self.jobs).map(|i| self.job_cost_mcu(i)).sum()
    }

    /// The work for job `index`.
    ///
    /// Every job gets a *different* range on purpose. Submitting the same spec
    /// repeatedly would measure a cache and settle under one claim key, which
    /// is the opposite of the contention this exists to create.
    pub fn spec(&self, index: u64) -> WorkSpec {
        match self.workload {
            Workload::Collatz => {
                let start = 2 + index.saturating_mul(self.size);
                WorkSpec::CollatzPeak {
                    start,
                    end: start.saturating_add(self.size),
                }
            }
            Workload::Prime => {
                let start = 2 + index.saturating_mul(self.size);
                WorkSpec::PrimeCount {
                    start,
                    end: start.saturating_add(self.size),
                }
            }
            Workload::Matrix => {
                let dim = self.size as u32;
                WorkSpec::MatrixMultiply {
                    seed_a: index.wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1),
                    seed_b: index.wrapping_add(1),
                    dim,
                    row_start: 0,
                    row_end: dim,
                }
            }
        }
    }
}

/// What happened to one job.
#[derive(Debug, Clone, Serialize)]
pub struct JobOutcome {
    pub index: u64,
    pub job_id: Option<String>,
    pub shards: u32,
    pub reserved_mcu: i64,
    /// How long the submit call took. This is not a bookkeeping detail: a
    /// submit is a quorum write, so this number is the ledger's write latency
    /// under whatever concurrency the run is applying.
    pub submit_ms: f64,
    /// Submit start to every shard finished. `None` if it never did.
    pub settle_ms: Option<f64>,
    pub error: Option<String>,
}

impl JobOutcome {
    fn failed(index: u64, submit_ms: f64, error: String) -> Self {
        Self {
            index,
            job_id: None,
            shards: 0,
            reserved_mcu: 0,
            submit_ms,
            settle_ms: None,
            error: Some(error),
        }
    }
}

/// Nearest-rank percentiles over a sample.
///
/// Nearest-rank rather than interpolated because every value here is a real
/// measurement that really happened, and a p99 that no request experienced is
/// harder to reason about than one that some request did.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Percentiles {
    pub count: usize,
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
    pub max: f64,
}

impl Percentiles {
    pub fn of(mut values: Vec<f64>) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let at = |q: f64| {
            let rank = (q * values.len() as f64).ceil().max(1.0) as usize;
            values[rank.min(values.len()) - 1]
        };
        Self {
            count: values.len(),
            mean: values.iter().sum::<f64>() / values.len() as f64,
            p50: at(0.50),
            p90: at(0.90),
            p99: at(0.99),
            max: *values.last().expect("non-empty"),
        }
    }
}

/// The economy, before and after, and whether it still adds up.
#[derive(Debug, Clone, Serialize)]
pub struct Accounting {
    /// What every successful submit said it had reserved.
    pub reserved_mcu: i64,
    pub spent_delta_mcu: i64,
    pub earned_delta_mcu: i64,
    pub balance_delta_mcu: i64,
    pub ledger_height_before: Option<u64>,
    pub ledger_height_after: Option<u64>,
    /// Every way the numbers failed to agree. Empty means they did.
    pub discrepancies: Vec<String>,
}

impl Accounting {
    pub fn check(
        reserved_mcu: i64,
        before: (i64, i64, i64),
        after: (i64, i64, i64),
        ledger_height_before: Option<u64>,
        ledger_height_after: Option<u64>,
    ) -> Self {
        let (bal0, earned0, spent0) = before;
        let (bal1, earned1, spent1) = after;
        let spent_delta_mcu = spent1 - spent0;
        let earned_delta_mcu = earned1 - earned0;
        let balance_delta_mcu = bal1 - bal0;
        let mut discrepancies = Vec::new();

        // Every CU the coordinator said it took has to be a CU the account
        // says it spent. A reservation the ledger dropped, or one it applied
        // twice, shows up here and nowhere else.
        if spent_delta_mcu != reserved_mcu {
            discrepancies.push(format!(
                "submits reserved {reserved_mcu} mCU but the account recorded \
                 {spent_delta_mcu} mCU spent"
            ));
        }
        // And the account's own three numbers have to describe one history.
        if balance_delta_mcu != earned_delta_mcu - spent_delta_mcu {
            discrepancies.push(format!(
                "balance moved {balance_delta_mcu} mCU while earned-minus-spent moved {} mCU",
                earned_delta_mcu - spent_delta_mcu
            ));
        }
        if let (Some(a), Some(b)) = (ledger_height_before, ledger_height_after)
            && b < a
        {
            discrepancies.push(format!("ledger height went backwards: {a} then {b}"));
        }
        Self {
            reserved_mcu,
            spent_delta_mcu,
            earned_delta_mcu,
            balance_delta_mcu,
            ledger_height_before,
            ledger_height_after,
            discrepancies,
        }
    }

    pub fn balanced(&self) -> bool {
        self.discrepancies.is_empty()
    }
}

/// Everything a run found out.
#[derive(Debug, Clone, Serialize)]
pub struct LoadReport {
    pub workload: String,
    pub jobs_requested: u64,
    pub concurrency: usize,
    pub shards_per_job: u32,
    pub size: u64,
    pub submitted: u64,
    pub completed: u64,
    pub failed: u64,
    pub timed_out: u64,
    pub shards_completed: u64,
    pub wall_ms: f64,
    pub jobs_per_second: f64,
    pub shards_per_second: f64,
    pub submit_ms: Percentiles,
    pub settle_ms: Percentiles,
    pub accounting: Accounting,
    /// Distinct failure texts, so a hundred identical errors read as one line.
    pub failures: Vec<String>,
}

impl LoadReport {
    /// Whether this run should fail a pipeline.
    ///
    /// Deliberately not "was it fast enough". Speed varies with the machine CI
    /// happened to allocate, and a threshold on it would be a flaky test that
    /// teaches people to ignore red. What does not vary is whether the work
    /// finished and whether the CU adds up, so that is what the exit code says.
    pub fn passed(&self) -> bool {
        self.failed == 0
            && self.timed_out == 0
            && self.completed == self.submitted
            && self.submitted > 0
            && self.accounting.balanced()
    }

    pub fn print(&self) {
        println!(
            "hocMESH load test: {} x {} shards of {:?}, size {}, {} in flight",
            self.jobs_requested, self.shards_per_job, self.workload, self.size, self.concurrency
        );
        println!(
            "\nSubmitted {} | completed {} | failed {} | timed out {}",
            self.submitted, self.completed, self.failed, self.timed_out
        );
        println!(
            "Wall clock {:.2}s | {:.2} jobs/s | {:.2} shards/s ({} shards)",
            self.wall_ms / 1000.0,
            self.jobs_per_second,
            self.shards_per_second,
            self.shards_completed
        );
        print_percentiles("Submit latency (ledger reservation)", &self.submit_ms);
        print_percentiles("Job settle latency (submit to last shard)", &self.settle_ms);

        let a = &self.accounting;
        println!("\nEconomy");
        println!("  reserved by submits : {:.3} CU", cu(a.reserved_mcu));
        println!(
            "  account recorded    : {:.3} CU spent",
            cu(a.spent_delta_mcu)
        );
        println!("  earned during run   : {:.3} CU", cu(a.earned_delta_mcu));
        println!("  balance moved       : {:.3} CU", cu(a.balance_delta_mcu));
        if let (Some(b), Some(a2)) = (a.ledger_height_before, a.ledger_height_after) {
            println!(
                "  ledger height       : {b} -> {a2} (+{})",
                a2.saturating_sub(b)
            );
        }
        if a.balanced() {
            println!("  CU conserved        : yes");
        } else {
            for d in &a.discrepancies {
                println!("  MISMATCH            : {d}");
            }
        }
        if !self.failures.is_empty() {
            println!("\nDistinct failures");
            for f in &self.failures {
                println!("  {f}");
            }
        }
        println!(
            "\n{}",
            if self.passed() {
                "PASS: every job settled and the CU adds up."
            } else {
                "FAIL: see above."
            }
        );
    }
}

fn cu(mcu: i64) -> f64 {
    mcu as f64 / 1000.0
}

fn print_percentiles(label: &str, p: &Percentiles) {
    if p.count == 0 {
        println!("\n{label}: no samples");
        return;
    }
    println!(
        "\n{label} (n={})\n  mean {:.0} ms | p50 {:.0} | p90 {:.0} | p99 {:.0} | max {:.0}",
        p.count, p.mean, p.p50, p.p90, p.p99, p.max
    );
}

/// Submits one job and waits for every shard of it to finish.
async fn run_one(client: &HocMeshClient, plan: &LoadPlan, index: u64) -> JobOutcome {
    let started = Instant::now();
    let submitted = match client.submit(plan.spec(index), plan.shards).await {
        Ok(r) => r,
        Err(e) => return JobOutcome::failed(index, ms(started.elapsed()), format!("submit: {e}")),
    };
    let submit_ms = ms(started.elapsed());

    let deadline = Instant::now() + plan.timeout;
    loop {
        match client.job_status(&submitted.job_id).await {
            Ok(s) if s.completed_assignments >= s.total_assignments && s.total_assignments > 0 => {
                return JobOutcome {
                    index,
                    job_id: Some(submitted.job_id),
                    shards: s.total_assignments,
                    reserved_mcu: submitted.reserved_mcu,
                    submit_ms,
                    settle_ms: Some(ms(started.elapsed())),
                    error: None,
                };
            }
            Ok(_) => {}
            Err(e) => {
                return JobOutcome {
                    index,
                    job_id: Some(submitted.job_id),
                    shards: submitted.assignments,
                    reserved_mcu: submitted.reserved_mcu,
                    submit_ms,
                    settle_ms: None,
                    error: Some(format!("status: {e}")),
                };
            }
        }
        if Instant::now() >= deadline {
            return JobOutcome {
                index,
                job_id: Some(submitted.job_id),
                shards: submitted.assignments,
                reserved_mcu: submitted.reserved_mcu,
                submit_ms,
                settle_ms: None,
                error: None, // a timeout is counted separately from a failure
            };
        }
        tokio::time::sleep(plan.poll).await;
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Runs the plan against a live coordinator and reports what happened.
pub async fn run(client: &HocMeshClient, plan: LoadPlan) -> Result<LoadReport> {
    plan.validate()?;
    let before = client
        .balance()
        .await
        .context("reading the balance this run will be measured against")?;

    let plan = Arc::new(plan);
    let next = Arc::new(AtomicU64::new(0));
    let started = Instant::now();
    let stop_at = plan.duration.map(|d| started + d);

    let mut workers = tokio::task::JoinSet::new();
    for _ in 0..plan.concurrency {
        let client = client.clone();
        let plan = Arc::clone(&plan);
        let next = Arc::clone(&next);
        workers.spawn(async move {
            let mut mine = Vec::new();
            loop {
                if stop_at.is_some_and(|t| Instant::now() >= t) {
                    break;
                }
                let index = next.fetch_add(1, Ordering::Relaxed);
                if plan.jobs > 0 && index >= plan.jobs {
                    break;
                }
                mine.push(run_one(&client, &plan, index).await);
            }
            mine
        });
    }

    let mut outcomes = Vec::new();
    while let Some(joined) = workers.join_next().await {
        outcomes.extend(joined.context("a load generator task panicked")?);
    }
    let wall_ms = ms(started.elapsed());

    let after = client
        .balance()
        .await
        .context("reading the balance after the run")?;

    Ok(summarize(&plan, outcomes, wall_ms, before, after))
}

/// Turns raw outcomes into the report. Pure, so the arithmetic is testable
/// without standing up a mesh.
fn summarize(
    plan: &LoadPlan,
    outcomes: Vec<JobOutcome>,
    wall_ms: f64,
    before: hocmesh_protocol::BalanceResponse,
    after: hocmesh_protocol::BalanceResponse,
) -> LoadReport {
    let submitted = outcomes.iter().filter(|o| o.job_id.is_some()).count() as u64;
    let failed = outcomes.iter().filter(|o| o.error.is_some()).count() as u64;
    let completed = outcomes.iter().filter(|o| o.settle_ms.is_some()).count() as u64;
    let timed_out = outcomes
        .iter()
        .filter(|o| o.job_id.is_some() && o.settle_ms.is_none() && o.error.is_none())
        .count() as u64;
    let shards_completed: u64 = outcomes
        .iter()
        .filter(|o| o.settle_ms.is_some())
        .map(|o| o.shards as u64)
        .sum();
    // Only what actually reserved: a submit that errored took no CU, and
    // counting it would manufacture the very discrepancy this is checking for.
    let reserved: i64 = outcomes
        .iter()
        .filter(|o| o.job_id.is_some())
        .map(|o| o.reserved_mcu)
        .sum();

    let mut failures: Vec<String> = outcomes.iter().filter_map(|o| o.error.clone()).collect();
    failures.sort();
    failures.dedup();
    failures.truncate(10);

    let seconds = (wall_ms / 1000.0).max(f64::MIN_POSITIVE);
    LoadReport {
        workload: format!("{:?}", plan.workload),
        jobs_requested: plan.jobs,
        concurrency: plan.concurrency,
        shards_per_job: plan.shards,
        size: plan.size,
        submitted,
        completed,
        failed,
        timed_out,
        shards_completed,
        wall_ms,
        jobs_per_second: completed as f64 / seconds,
        shards_per_second: shards_completed as f64 / seconds,
        submit_ms: Percentiles::of(
            outcomes
                .iter()
                .filter(|o| o.job_id.is_some())
                .map(|o| o.submit_ms)
                .collect(),
        ),
        settle_ms: Percentiles::of(outcomes.iter().filter_map(|o| o.settle_ms).collect()),
        accounting: Accounting::check(
            reserved,
            (before.balance_mcu, before.earned_mcu, before.spent_mcu),
            (after.balance_mcu, after.earned_mcu, after.spent_mcu),
            before.ledger_height,
            after.ledger_height,
        ),
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(workload: Workload, size: u64) -> LoadPlan {
        LoadPlan {
            jobs: 4,
            concurrency: 2,
            shards: 2,
            workload,
            size,
            duration: None,
            timeout: Duration::from_secs(30),
            poll: Duration::from_millis(50),
        }
    }

    fn balance(bal: i64, earned: i64, spent: i64) -> hocmesh_protocol::BalanceResponse {
        hocmesh_protocol::BalanceResponse {
            node_id: "hocmesh_test".into(),
            balance_mcu: bal,
            earned_mcu: earned,
            spent_mcu: spent,
            ledger_height: Some(10),
            ledger_head: None,
        }
    }

    fn settled(index: u64, reserved: i64, submit_ms: f64, settle_ms: f64) -> JobOutcome {
        JobOutcome {
            index,
            job_id: Some(format!("job_{index}")),
            shards: 2,
            reserved_mcu: reserved,
            submit_ms,
            settle_ms: Some(settle_ms),
            error: None,
        }
    }

    /// Two jobs asking for the same range would settle under one claim key and
    /// measure a cache instead of the ledger.
    #[test]
    fn every_job_asks_for_different_work() {
        for w in [Workload::Collatz, Workload::Prime, Workload::Matrix] {
            let p = plan(w, 1000);
            let specs: Vec<String> = (0..8).map(|i| format!("{:?}", p.spec(i))).collect();
            let mut unique = specs.clone();
            unique.sort();
            unique.dedup();
            assert_eq!(unique.len(), specs.len(), "{w:?} repeats a work spec");
        }
    }

    #[test]
    fn ranges_do_not_overlap_between_jobs() {
        let p = plan(Workload::Collatz, 100);
        let WorkSpec::CollatzPeak { end: first_end, .. } = p.spec(0) else {
            panic!("expected a collatz spec")
        };
        let WorkSpec::CollatzPeak {
            start: second_start,
            ..
        } = p.spec(1)
        else {
            panic!("expected a collatz spec")
        };
        assert!(second_start >= first_end);
    }

    #[test]
    fn a_plan_with_no_stopping_condition_is_refused() {
        let mut p = plan(Workload::Collatz, 10);
        p.jobs = 0;
        assert!(p.validate().is_err());
        p.duration = Some(Duration::from_secs(1));
        assert!(p.validate().is_ok());
    }

    #[test]
    fn an_absurd_matrix_is_refused_before_it_is_submitted() {
        let mut p = plan(Workload::Matrix, 100_000);
        assert!(p.validate().is_err());
        p.size = 256;
        assert!(p.validate().is_ok());
    }

    #[test]
    fn percentiles_are_values_that_actually_happened() {
        let p = Percentiles::of((1..=100).map(|v| v as f64).collect());
        assert_eq!(p.count, 100);
        assert_eq!(p.p50, 50.0);
        assert_eq!(p.p90, 90.0);
        assert_eq!(p.p99, 99.0);
        assert_eq!(p.max, 100.0);
        assert_eq!(Percentiles::of(vec![]).count, 0);
        assert_eq!(Percentiles::of(vec![7.0]).p99, 7.0);
    }

    /// The single-machine case: this node both spends and earns, so a naive
    /// "balance went down by what we reserved" check would fail on a healthy
    /// run. What must hold is that the three counters describe one history.
    #[test]
    fn a_run_that_earns_back_what_it_spent_still_balances() {
        let p = plan(Workload::Collatz, 10);
        let outcomes = vec![settled(0, 500, 12.0, 300.0), settled(1, 500, 14.0, 320.0)];
        let r = summarize(
            &p,
            outcomes,
            1000.0,
            balance(0, 0, 0),
            balance(700, 1700, 1000),
        );
        assert!(r.accounting.balanced(), "{:?}", r.accounting.discrepancies);
        assert!(r.passed());
        assert_eq!(r.completed, 2);
        assert_eq!(r.shards_completed, 4);
    }

    /// The bug class this exists to catch: the coordinator says it reserved
    /// CU that the account never records as spent.
    #[test]
    fn a_reservation_the_account_never_recorded_fails_the_run() {
        let p = plan(Workload::Collatz, 10);
        let outcomes = vec![settled(0, 500, 10.0, 200.0), settled(1, 500, 10.0, 200.0)];
        let r = summarize(
            &p,
            outcomes,
            1000.0,
            balance(0, 0, 0),
            balance(-500, 0, 500),
        );
        assert!(!r.accounting.balanced());
        assert!(!r.passed());
        assert!(r.accounting.discrepancies[0].contains("1000"));
    }

    /// A balance that does not equal earned minus spent means the account is
    /// telling two stories, whatever the reservations said.
    #[test]
    fn an_account_that_contradicts_itself_fails_the_run() {
        let p = plan(Workload::Collatz, 10);
        let outcomes = vec![settled(0, 1000, 10.0, 200.0)];
        let r = summarize(
            &p,
            outcomes,
            1000.0,
            balance(0, 0, 0),
            balance(999, 0, 1000),
        );
        assert!(!r.accounting.balanced());
        assert!(
            r.accounting
                .discrepancies
                .iter()
                .any(|d| d.contains("balance moved"))
        );
    }

    #[test]
    fn a_failed_submit_reserves_nothing_and_fails_the_run() {
        let p = plan(Workload::Collatz, 10);
        let outcomes = vec![
            settled(0, 500, 10.0, 200.0),
            JobOutcome::failed(1, 5.0, "submit: insufficient balance".into()),
        ];
        let r = summarize(
            &p,
            outcomes,
            1000.0,
            balance(0, 0, 0),
            balance(-500, 0, 500),
        );
        assert_eq!(r.submitted, 1, "a refused submit did not reserve anything");
        assert_eq!(r.failed, 1);
        assert!(r.accounting.balanced(), "the CU still adds up");
        assert!(!r.passed(), "but a run with a failure has not passed");
        assert_eq!(r.failures, vec!["submit: insufficient balance"]);
    }

    #[test]
    fn a_job_that_never_settles_is_a_timeout_not_a_failure() {
        let p = plan(Workload::Collatz, 10);
        let outcomes = vec![JobOutcome {
            index: 0,
            job_id: Some("job_0".into()),
            shards: 2,
            reserved_mcu: 500,
            submit_ms: 10.0,
            settle_ms: None,
            error: None,
        }];
        let r = summarize(
            &p,
            outcomes,
            1000.0,
            balance(0, 0, 0),
            balance(-500, 0, 500),
        );
        assert_eq!(r.timed_out, 1);
        assert_eq!(r.failed, 0);
        assert!(!r.passed());
    }

    #[test]
    fn a_run_that_submitted_nothing_has_not_passed() {
        let p = plan(Workload::Collatz, 10);
        let r = summarize(&p, Vec::new(), 1000.0, balance(0, 0, 0), balance(0, 0, 0));
        assert!(!r.passed(), "an empty run must not report success");
    }

    /// The price a run will pay is knowable before it starts, from the same
    /// function the ledger charges with. That is what lets a pipeline wait for
    /// a balance instead of discovering halfway through that it is broke.
    #[test]
    fn a_plan_can_say_what_it_will_cost_before_it_spends_anything() {
        let p = plan(Workload::Collatz, 50_000);
        let total = p.cost_mcu();
        assert!(total > 0);
        assert_eq!(
            total,
            (0..p.jobs).map(|i| p.job_cost_mcu(i)).sum::<i64>(),
            "the total has to be the sum of the jobs it will actually submit"
        );
        for i in 0..p.jobs {
            assert_eq!(
                p.job_cost_mcu(i),
                hocmesh_core::compute::split_work(&p.spec(i), p.shards)
                    .iter()
                    .map(hocmesh_core::compute::work_cost_mcu)
                    .sum::<i64>(),
                "the price must be the one submit charges, shard by shard"
            );
        }
        let mut open_ended = p.clone();
        open_ended.jobs = 0;
        open_ended.duration = Some(Duration::from_secs(1));
        assert_eq!(open_ended.cost_mcu(), open_ended.job_cost_mcu(0));
    }

    #[test]
    fn identical_failures_collapse_to_one_line() {
        let p = plan(Workload::Collatz, 10);
        let outcomes: Vec<JobOutcome> = (0..50)
            .map(|i| JobOutcome::failed(i, 1.0, "submit: connection refused".into()))
            .collect();
        let r = summarize(&p, outcomes, 100.0, balance(0, 0, 0), balance(0, 0, 0));
        assert_eq!(r.failures.len(), 1);
        assert_eq!(r.failed, 50);
    }
}
