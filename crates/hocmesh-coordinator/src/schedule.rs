//! The resource graph, and how pending work is matched onto it.
//!
//! Until now a polling node got whichever shard had the lowest rowid. That is
//! correct -- every shard is worth what its spec says and no more -- but it is
//! indifferent, and indifference is expensive at scale: it hands a shard to a
//! machine that cannot finish it inside the lease, ignores that a node already
//! has the working set for the job next to it, and treats a node that has
//! never returned a verified result exactly like one with a thousand.
//!
//! Nothing here can move CU. A score decides *who is offered which shard*, and
//! that is all. The reward is still `work_cost_mcu` of the spec, the claim key
//! is still derived from the assignment id, and a validator still recomputes
//! both from evidence the coordinator did not author. A scheduler that scored
//! badly -- or maliciously -- wastes effort. It cannot overpay, underpay, or
//! pay twice. That is the property that lets this file be a heuristic at all.
//!
//! The four axes are the ones the roadmap names: hardware, network,
//! reliability, and cache locality. Each is normalised to `[0, 1]` where 1 is
//! best, and they are combined by a weighted mean, so the fit score is also in
//! `[0, 1]`. The starvation bonus is added *outside* that mean and is weighted
//! above 1, which is what makes waiting a guarantee rather than a preference:
//! a shard that has waited out the window outranks every fresh candidate no
//! matter how well they fit.

use hocmesh_core::compute::REFERENCE_OPS_PER_MCU;
use hocmesh_core::proximity;
use hocmesh_core::reputation::{FLOOR_AUDIT_RATE, Reputation};
use hocmesh_core::verify::trial_division_ops;
use hocmesh_protocol::{DEFAULT_LEASE_SECONDS, MAX_LEASE_SECONDS, NodeCapabilities};
use serde::Serialize;

/// The prime bound `hocmesh_core::hardware::benchmark_cpu` runs to.
///
/// Kept here so the benchmark score can be read back as ops: the benchmark
/// reports candidates per second at this bound, and a candidate at this bound
/// costs `trial_division_ops(BENCHMARK_LIMIT)`.
const BENCHMARK_LIMIT: u64 = 150_000;

/// Score given to an axis whose input the node never measured.
///
/// Deliberately the middle rather than the top. A node that advertises no
/// benchmark and no latency must not be ranked as fast and nearby -- that is
/// the same mistake `NetworkCoordinate` documents for unplaced nodes, and it
/// would make "measure nothing" the winning strategy.
const UNKNOWN: f64 = 0.5;

/// How long a shard may wait before its age decides the matter.
pub const STARVATION_SECS: i64 = 300;

/// Weight of a fully starved shard's age bonus.
///
/// Greater than 1, and the fit score can never exceed 1, so a shard that has
/// waited out `STARVATION_SECS` strictly outranks every candidate that has
/// not. Liveness is not traded against efficiency; it wins.
pub const STARVATION_WEIGHT: f64 = 1.25;

/// The bonus must exceed a perfect fit, or "waited too long" would be a
/// preference rather than a guarantee. Checked at compile time because it is a
/// property of the constant, not of any particular candidate.
const _: () = assert!(STARVATION_WEIGHT > 1.0);

/// How the four axes are combined. They sum to 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weights {
    pub hardware: f64,
    pub network: f64,
    pub reliability: f64,
    pub locality: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            hardware: 0.25,
            network: 0.20,
            reliability: 0.25,
            locality: 0.30,
        }
    }
}

impl Weights {
    fn total(&self) -> f64 {
        self.hardware + self.network + self.reliability + self.locality
    }
}

/// One node, as the scheduler sees it.
#[derive(Debug, Clone)]
pub struct WorkerProfile {
    pub node_id: String,
    /// The region of the coordinator this node registered with, if the
    /// deployment is federated at all.
    pub region: Option<String>,
    pub caps: NodeCapabilities,
    pub reputation: Reputation,
}

/// One pending shard, with the little history that makes it warm or cold for
/// the node being scored.
#[derive(Debug, Clone)]
pub struct ShardCandidate {
    pub assignment_id: String,
    pub job_id: String,
    pub shard_index: u32,
    /// What the shard pays, which is also its size: the reward *is*
    /// `work_cost_mcu` of the spec, so one number serves as both.
    pub reward_mcu: i64,
    /// Working-set estimate, used only as a hard gate against a node that
    /// cannot hold it.
    pub memory_bytes: u64,
    /// When the shard first became schedulable.
    pub created_at: i64,
    /// Shards of this same job this node has already returned.
    pub shards_done_here: u32,
    /// The closest shard index of this job this node has already returned.
    pub nearest_done_shard: Option<u32>,
    /// Model manifests the shard needs on disk. Empty for CPU work; the field
    /// exists so inference placement can score through the same function.
    pub required_manifests: Vec<String>,
}

/// The spread of the candidate set, so relative measures stay meaningful.
///
/// Several axes want "how big is this shard *for this batch of shards*"
/// rather than "how big is this shard in seconds". Absolute measures saturate:
/// today's shards finish in milliseconds against a lease measured in minutes,
/// so anything scaled to the lease is 1.0 for every candidate and decides
/// nothing. Scaled to the field, the same axis discriminates at any size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scale {
    pub max_reward_mcu: i64,
    /// How many shards are on offer at once.
    pub pending_shards: usize,
    /// How many nodes have asked for work recently, or `0` when the caller did
    /// not measure it. Zero disables the head start in `fit` entirely, so a
    /// caller that knows nothing about demand gets exactly the old behaviour.
    pub recent_pollers: usize,
}

impl Scale {
    pub fn of(candidates: &[ShardCandidate]) -> Self {
        Self {
            max_reward_mcu: candidates
                .iter()
                .map(|c| c.reward_mcu)
                .max()
                .unwrap_or(1)
                .max(1),
            pending_shards: candidates.len(),
            recent_pollers: 0,
        }
    }

    /// Record how many nodes are competing for these shards.
    pub fn with_recent_pollers(mut self, pollers: usize) -> Self {
        self.recent_pollers = pollers;
        self
    }

    /// Whether there are more nodes wanting work than there is work.
    ///
    /// Only then does it cost anything to give a shard to a slower machine:
    /// with a shard for everyone, holding one back denies it to a node that was
    /// going to sit idle anyway and delays the job for no gain at all.
    fn contended(&self) -> bool {
        self.recent_pollers > self.pending_shards
    }

    /// This shard's share of the largest on offer, in `[0, 1]`.
    fn exposure(&self, candidate: &ShardCandidate) -> f64 {
        (candidate.reward_mcu.max(0) as f64 / self.max_reward_mcu as f64).clamp(0.0, 1.0)
    }
}

/// Why a candidate scored the way it did.
///
/// Returned whole rather than as a single number so an operator can see which
/// axis moved a decision, and so the tests can assert about one axis without
/// reverse-engineering it out of the total.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Fit {
    pub hardware: f64,
    pub network: f64,
    pub reliability: f64,
    pub locality: f64,
    pub starvation: f64,
    pub total: f64,
}

/// Why a node was refused a shard outright.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unfit {
    /// The operator lent no CPU, so there is nothing to schedule onto.
    NoSharedCpu,
    /// The shard's working set does not fit in what the operator lent.
    NotEnoughMemory,
    /// The node cannot finish this shard before its lease expires. Handing it
    /// over anyway produces a guaranteed expiry and a guaranteed re-lease.
    SlowerThanLease,
    /// Not a refusal: this shard is new, faster machines are competing for it,
    /// and this node's head start has not run out yet. Ask again shortly and
    /// the same shard will be offered.
    StillReservedForFasterNodes,
}

/// The longest a shard is ever held back from the node in front of it.
///
/// Scheduling here is pull-based: a node asks, and the coordinator answers from
/// the shards that exist at that moment. It cannot save a shard for a faster
/// machine that has not asked yet, so once slower machines are eligible -- which
/// is the point of the variable lease -- whoever asks first takes it, and a job
/// can finish later than it needed to for no reason but the order of arrival.
///
/// The head start is the whole of the correction, so it is bounded on purpose
/// and it is short: a fixed, known ceiling on how long any shard can be idle
/// while a faster node is hoped for. Past it every eligible node is equal again,
/// which is what stops this from becoming an exclusion rule wearing a different
/// hat.
pub const HEAD_START_SECONDS: i64 = 30;

/// mCU per second this node's benchmark says it sustains, or `None` when it
/// never ran one.
///
/// The conversion is arithmetic between two definitions this workspace already
/// fixes, not a tuning constant: `benchmark_cpu` reports prime candidates per
/// second at `BENCHMARK_LIMIT`, each such candidate costs
/// `trial_division_ops(BENCHMARK_LIMIT)` reference ops, and `REFERENCE_OPS_PER_MCU`
/// reference ops are one mCU by definition.
///
/// The benchmark is single-threaded and so is one shard; `shared_logical_cpus`
/// says how many shards run at once, not how fast one of them goes, so it is
/// deliberately not a factor here.
pub fn mcu_per_second(caps: &NodeCapabilities) -> Option<f64> {
    if caps.cpu_benchmark_score == 0 {
        return None;
    }
    let ops_per_second =
        caps.cpu_benchmark_score as f64 * trial_division_ops(BENCHMARK_LIMIT) as f64;
    let rate = ops_per_second / REFERENCE_OPS_PER_MCU as f64;
    (rate.is_finite() && rate > 0.0).then_some(rate)
}

/// How long this node should need for this shard, if it can be predicted.
fn predicted_seconds(caps: &NodeCapabilities, reward_mcu: i64) -> Option<f64> {
    let rate = mcu_per_second(caps)?;
    let seconds = reward_mcu.max(0) as f64 / rate;
    seconds.is_finite().then_some(seconds)
}

/// How much longer than predicted a node is given before its lease expires.
///
/// A prediction is made from one benchmark against a reference machine, on a
/// node the coordinator does not control and that is by design only lending a
/// slice of itself. Handing it exactly the time the arithmetic says would make
/// every ordinary fluctuation -- the operator opening something, a thermal cap,
/// a shard on the unlucky side of the split -- into an expiry, and expiries
/// throw away completed work.
const LEASE_HEADROOM: f64 = 1.5;

/// How long this node gets for this shard.
///
/// The point of varying it is that a slower machine finishing the same shard is
/// worth exactly what a faster one finishing it is worth: the mCU is in the
/// work, not in the clock. So the mesh does not need slow nodes to be quick, it
/// only needs to know they are slow -- and it does, because `mcu_per_second`
/// already says so. What it must not do is refuse them, which is what a lease
/// fixed at the default silently did.
///
/// Never shorter than the default, so this can only ever widen the field of
/// machines that qualify, and never take time away from one that qualified
/// before. A node that has not been benchmarked keeps the default.
pub fn lease_seconds_for(caps: &NodeCapabilities, reward_mcu: i64) -> i64 {
    let Some(seconds) = predicted_seconds(caps, reward_mcu) else {
        return DEFAULT_LEASE_SECONDS;
    };
    let wanted = (seconds * LEASE_HEADROOM).ceil();
    if !wanted.is_finite() {
        return DEFAULT_LEASE_SECONDS;
    }
    (wanted as i64).clamp(DEFAULT_LEASE_SECONDS, MAX_LEASE_SECONDS)
}

/// Round-trip time to hand this shard out and take the result back, in
/// seconds, or `None` when the node has never measured it.
fn round_trip_seconds(caps: &NodeCapabilities) -> Option<f64> {
    (caps.coordinator_latency_micros > 0)
        .then(|| caps.coordinator_latency_micros as f64 / 1_000_000.0)
}

/// Standing in `[0, 1]`: 0 for a node nobody has verified, 1 for a long clean
/// record.
///
/// Two independent things are folded together. The audit rate is the system's
/// own opinion of how much checking this node still needs, and decays with the
/// clean streak. The Laplace-smoothed acceptance ratio is the record itself,
/// and unlike the streak it does not forget: a node with fifty rejections
/// behind it cannot buy back a clean reputation with twelve good results.
pub fn standing(rep: &Reputation) -> f64 {
    let settled = rep.accepted + rep.rejected;
    let acceptance = (rep.accepted as f64 + 1.0) / (settled as f64 + 2.0);
    let trust = ((1.0 - rep.audit_rate()) / (1.0 - FLOOR_AUDIT_RATE)).clamp(0.0, 1.0);
    (0.5 * trust + 0.5 * acceptance).clamp(0.0, 1.0)
}

/// Score one shard for one node, or say why the node cannot have it.
///
/// `now` is passed rather than read so the caller controls the clock and the
/// tests do not have to sleep.
pub fn fit(
    worker: &WorkerProfile,
    candidate: &ShardCandidate,
    scale: &Scale,
    now: i64,
    coordinator_region: Option<&str>,
    weights: &Weights,
) -> Result<Fit, Unfit> {
    if worker.caps.shared_logical_cpus == 0 {
        return Err(Unfit::NoSharedCpu);
    }
    if worker.caps.shared_memory_bytes < candidate.memory_bytes {
        return Err(Unfit::NotEnoughMemory);
    }
    // Against the ceiling, not the default. The lease this node will actually
    // be granted stretches to fit it -- `lease_seconds_for` gives it
    // `predicted * LEASE_HEADROOM`, and headroom above 1 means that always
    // covers the prediction until the clamp bites -- so the only nodes left to
    // refuse here are the ones no lease we are willing to grant would cover,
    // which is exactly `predicted > MAX_LEASE_SECONDS`.
    //
    // Judging against the default instead is what turned "slower than most"
    // into "excluded", and a network built on lending hardware people already
    // own cannot afford to read those two as the same thing.
    let seconds = predicted_seconds(&worker.caps, candidate.reward_mcu);
    if seconds.is_some_and(|s| s > MAX_LEASE_SECONDS as f64) {
        return Err(Unfit::SlowerThanLease);
    }

    // Hardware: headroom against the *default* lease, deliberately, and not
    // against the longer one a slow node would actually be granted. The
    // denominator has to be the same for every node or the axis stops being a
    // comparison: scaling each node by its own lease would hand the slowest
    // machines the largest denominators and score them best, which is exactly
    // backwards. A node past the default simply pins at 0 -- ranked last among
    // those eligible, which is what "slower, still welcome" should look like.
    //
    // For today's shard sizes this saturates at 1 for every capable node, which
    // is the truth -- hardware is a filter here, not a preference. It starts
    // discriminating exactly when shards grow large enough for it to matter.
    let hardware = seconds.map_or(UNKNOWN, |s| {
        (1.0 - s / DEFAULT_LEASE_SECONDS as f64).clamp(0.0, 1.0)
    });

    // The head start, and the only place the scheduler prefers a fast machine
    // over a slow one for reasons of time rather than of fit.
    //
    // Every node's wait is `HEAD_START_SECONDS * (1 - hardware)`, so the
    // fastest waits nothing and takes fresh work the instant it appears, and
    // each slower one is behind it by an amount that is a measurement rather
    // than a rank. Nobody is compared against anybody: the node is compared
    // against the clock, which is what keeps this cheap -- no peer lookup, no
    // shared state, the same answer from any coordinator scoring the same row.
    //
    // Three things stop it becoming exclusion. It is capped, so the slowest
    // eligible node waits half a minute and not a second longer. It applies
    // only while more nodes want work than there is work, because otherwise
    // the shard would go to nobody at all. And it is measured from when the
    // shard appeared, so a shard that has been sitting is open to everyone --
    // which means the correction switches itself off in precisely the case it
    // was not built for, a queue nobody fast is draining.
    if scale.contended() {
        let head_start = (HEAD_START_SECONDS as f64 * (1.0 - hardware)).round() as i64;
        if now.saturating_sub(candidate.created_at) < head_start {
            return Err(Unfit::StillReservedForFasterNodes);
        }
    }

    // Network: how much of the exchange is the work rather than the round trip
    // carrying it. A distant node scores better on a larger shard, which is
    // the batching argument stated as arithmetic. Unknown latency is neutral,
    // never near.
    let network = match (round_trip_seconds(&worker.caps), seconds) {
        (Some(rtt), Some(work)) if rtt + work > 0.0 => work / (rtt + work),
        _ => UNKNOWN,
    };
    // Crossing a region is a real cost that the coordinator-to-node latency
    // does not see, because it was measured against a coordinator on the near
    // side. It scales the axis rather than gating: work may always run
    // anywhere, the ledger does not care where it ran.
    let network = match (coordinator_region, worker.region.as_deref()) {
        (Some(here), Some(there)) if here != there => network * 0.75,
        _ => network,
    };

    // Reliability: an unproven node is offered the cheap end of the board.
    // A verified record scores 1 whatever the size; the less of a record there
    // is, the more the shard's value counts against it. This limits what a
    // first fabrication can be worth and what the full-rate audit costs.
    let standing = standing(&worker.reputation);
    let reliability = standing + (1.0 - standing) * (1.0 - scale.exposure(candidate));

    let locality = locality(worker, candidate);

    let fit = (weights.hardware * hardware
        + weights.network * network
        + weights.reliability * reliability
        + weights.locality * locality)
        / weights.total().max(f64::MIN_POSITIVE);

    let waited = (now - candidate.created_at).max(0) as f64;
    let starvation = (waited / STARVATION_SECS as f64).clamp(0.0, 1.0) * STARVATION_WEIGHT;

    Ok(Fit {
        hardware,
        network,
        reliability,
        locality,
        starvation,
        total: fit + starvation,
    })
}

/// Cache locality: how warm this node already is for this shard.
///
/// Three signals, and only the ones that apply are averaged, so a workload
/// with no model to cache is not silently penalised for having nothing to hit.
fn locality(worker: &WorkerProfile, candidate: &ShardCandidate) -> f64 {
    // Having done shards of this job before means the code path, the operand
    // seeds and the page cache are already warm. Saturating: the first shard
    // is most of the benefit.
    let job_affinity = 1.0 - 1.0 / (1.0 + f64::from(candidate.shards_done_here));
    let mut parts = vec![job_affinity];

    // Adjacency to a shard this node already finished. For a row block of a
    // matrix or a range of integers, the neighbouring shard reuses the most.
    if let Some(done) = candidate.nearest_done_shard {
        let distance = (i64::from(candidate.shard_index) - i64::from(done)).unsigned_abs() as f64;
        parts.push(1.0 / distance.max(1.0));
    }

    if !candidate.required_manifests.is_empty() {
        let held = candidate
            .required_manifests
            .iter()
            .filter(|m| worker.caps.cached_model_manifests.contains(m))
            .count();
        parts.push(held as f64 / candidate.required_manifests.len() as f64);
    }

    parts.iter().sum::<f64>() / parts.len() as f64
}

/// The best shard for this node, if any of them will do.
///
/// Ties break on the assignment id so two coordinators replaying the same
/// state make the same offer, and so a test does not depend on row order.
pub fn best<'a>(
    worker: &WorkerProfile,
    candidates: &'a [ShardCandidate],
    now: i64,
    coordinator_region: Option<&str>,
    weights: &Weights,
    recent_pollers: usize,
) -> Option<(&'a ShardCandidate, Fit)> {
    let scale = Scale::of(candidates).with_recent_pollers(recent_pollers);
    candidates
        .iter()
        .filter_map(|c| {
            fit(worker, c, &scale, now, coordinator_region, weights)
                .ok()
                .map(|f| (c, f))
        })
        .max_by(|(lc, lf), (rc, rf)| {
            lf.total
                .total_cmp(&rf.total)
                .then_with(|| rc.assignment_id.cmp(&lc.assignment_id))
        })
}

// ---------------------------------------------------------------------------
// The resource graph
// ---------------------------------------------------------------------------

/// Distance assumed between two nodes that have measured nothing.
///
/// Chosen to be worse than any plausible measurement rather than better, for
/// the same reason `UNKNOWN` is the middle of an axis and not the top: if
/// unmeasured looked close, the cheapest way to be picked for a tight cluster
/// would be to stop measuring.
pub const UNKNOWN_EDGE_MICROS: u64 = 400_000;

/// One vertex of the resource graph.
#[derive(Debug, Clone)]
pub struct Vertex {
    pub node_id: String,
    pub region: Option<String>,
    pub caps: NodeCapabilities,
    pub reputation: Reputation,
    pub online: bool,
}

/// The machines currently offering capacity, and the distances between them.
///
/// Vertex weights are the ones `NodeCapabilities` already carries -- benchmark,
/// shared memory, GPUs, cached manifests, reliability -- and edge weights are
/// predicted round trips. Tightly coupled work (pipeline or tensor
/// parallelism) needs a set of machines that are near *each other*, which is a
/// question about edges and cannot be answered by ranking nodes one at a time.
#[derive(Debug, Clone, Default)]
pub struct ResourceGraph {
    vertices: Vec<Vertex>,
}

/// A set of machines chosen to be near one another.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Cluster {
    pub node_ids: Vec<String>,
    /// The longest edge inside the cluster, which is what a synchronising
    /// collective actually waits on.
    pub worst_edge_micros: u64,
    pub total_edge_micros: u64,
}

impl ResourceGraph {
    pub fn new(vertices: Vec<Vertex>) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    /// Predicted round trip between two vertices.
    ///
    /// Measured coordinates when both nodes have a plausible one. Otherwise
    /// the path through the coordinator, which is a real route and a genuine
    /// upper bound on the direct one. When even that is unmeasured the pair is
    /// treated as far apart rather than close.
    pub fn edge_micros(&self, a: usize, b: usize) -> u64 {
        if a == b {
            return 0;
        }
        let (va, vb) = (&self.vertices[a], &self.vertices[b]);
        match (
            va.caps.network_coordinate.as_ref(),
            vb.caps.network_coordinate.as_ref(),
        ) {
            (Some(x), Some(y)) if proximity::is_plausible(x) && proximity::is_plausible(y) => {
                proximity::predicted_rtt_micros(x, y)
            }
            _ => {
                let via = va
                    .caps
                    .coordinator_latency_micros
                    .saturating_add(vb.caps.coordinator_latency_micros);
                if via == 0 { UNKNOWN_EDGE_MICROS } else { via }
            }
        }
    }

    /// The tightest `size` machines that pass `gate`.
    ///
    /// Grown greedily from every eligible seed and the best result kept, which
    /// is not guaranteed optimal -- the exact problem is the k-clique of
    /// minimum diameter -- but is deterministic, runs in time the coordinator
    /// can spend on a request, and beats picking by node weight alone, which
    /// is what "no graph" means in practice.
    ///
    /// Restricting a cluster to one region is expressed by gating on region;
    /// there is deliberately no region term in the edge weight, because
    /// inventing a latency for a region boundary would be inventing data.
    pub fn cluster(&self, size: usize, gate: impl Fn(&Vertex) -> bool) -> Option<Cluster> {
        if size == 0 {
            return None;
        }
        let eligible: Vec<usize> = (0..self.vertices.len())
            .filter(|&i| self.vertices[i].online && gate(&self.vertices[i]))
            .collect();
        if eligible.len() < size {
            return None;
        }
        let mut best: Option<Cluster> = None;
        for &seed in &eligible {
            let mut chosen = vec![seed];
            let mut rest: Vec<usize> = eligible.iter().copied().filter(|&i| i != seed).collect();
            while chosen.len() < size {
                let Some(&pick) = rest.iter().min_by_key(|&&c| {
                    let worst = chosen
                        .iter()
                        .map(|&m| self.edge_micros(m, c))
                        .max()
                        .unwrap_or(0);
                    let total: u64 = chosen.iter().map(|&m| self.edge_micros(m, c)).sum();
                    (worst, total, self.vertices[c].node_id.clone())
                }) else {
                    break;
                };
                chosen.push(pick);
                rest.retain(|&i| i != pick);
            }
            if chosen.len() < size {
                continue;
            }
            let candidate = self.measure(&chosen);
            let better = best.as_ref().is_none_or(|b| {
                (
                    candidate.worst_edge_micros,
                    candidate.total_edge_micros,
                    &candidate.node_ids,
                ) < (b.worst_edge_micros, b.total_edge_micros, &b.node_ids)
            });
            if better {
                best = Some(candidate);
            }
        }
        best
    }

    fn measure(&self, chosen: &[usize]) -> Cluster {
        let mut worst: u64 = 0;
        let mut total: u64 = 0;
        for (i, &a) in chosen.iter().enumerate() {
            for &b in &chosen[i + 1..] {
                let e = self.edge_micros(a, b);
                worst = worst.max(e);
                total = total.saturating_add(e);
            }
        }
        let mut node_ids: Vec<String> = chosen
            .iter()
            .map(|&i| self.vertices[i].node_id.clone())
            .collect();
        node_ids.sort();
        Cluster {
            node_ids,
            worst_edge_micros: worst,
            total_edge_micros: total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hocmesh_protocol::NetworkCoordinate;

    fn caps(benchmark: u64, latency_micros: u64) -> NodeCapabilities {
        NodeCapabilities {
            protocol_version: hocmesh_protocol::PROTOCOL_VERSION,
            hostname: "test".into(),
            os: "test".into(),
            arch: "test".into(),
            cpu_brand: "test".into(),
            logical_cpus: 8,
            total_memory_bytes: 16 << 30,
            cpu_benchmark_score: benchmark,
            memory_bandwidth_bytes_per_second: None,
            gpus: Vec::new(),
            model_seed_url: None,
            cached_model_manifests: Vec::new(),
            coordinator_latency_micros: latency_micros,
            model_bandwidth_kbps: 100_000,
            accelerator_load_permille: 0,
            ai_runtime_ready: false,
            shared_logical_cpus: 4,
            shared_memory_bytes: 8 << 30,
            shared_gpu_percent: 0,
            network_coordinate: None,
            probe_endpoint: None,
        }
    }

    fn worker(node_id: &str, benchmark: u64, latency_micros: u64) -> WorkerProfile {
        WorkerProfile {
            node_id: node_id.into(),
            region: None,
            caps: caps(benchmark, latency_micros),
            reputation: Reputation::new(),
        }
    }

    fn shard(assignment_id: &str, index: u32, reward_mcu: i64) -> ShardCandidate {
        ShardCandidate {
            assignment_id: assignment_id.into(),
            job_id: "job".into(),
            shard_index: index,
            reward_mcu,
            memory_bytes: 1 << 20,
            created_at: 1_000,
            shards_done_here: 0,
            nearest_done_shard: None,
            required_manifests: Vec::new(),
        }
    }

    /// A shard this node is predicted to spend `seconds` over.
    ///
    /// The head start is a function of predicted time, so a test that wants a
    /// particular wait has to ask for a particular duration. Writing the reward
    /// as a literal instead couples the test to `REFERENCE_OPS_PER_MCU` and
    /// quietly crosses `MAX_LEASE_SECONDS` when either constant moves.
    fn shard_taking(node: &WorkerProfile, id: &str, seconds: f64) -> ShardCandidate {
        let reward = (mcu_per_second(&node.caps).expect("benchmarked node") * seconds) as i64;
        shard(id, 0, reward.max(1))
    }

    fn score(w: &WorkerProfile, c: &ShardCandidate, now: i64) -> Fit {
        let field = [c.clone()];
        fit(w, c, &Scale::of(&field), now, None, &Weights::default())
            .expect("candidate should be schedulable")
    }

    #[test]
    fn a_node_that_lent_nothing_is_offered_nothing() {
        let mut w = worker("idle", 1_000_000, 20_000);
        w.caps.shared_logical_cpus = 0;
        let c = shard("a", 0, 100);
        assert_eq!(
            fit(
                &w,
                &c,
                &Scale::of(std::slice::from_ref(&c)),
                1_000,
                None,
                &Weights::default()
            ),
            Err(Unfit::NoSharedCpu)
        );
    }

    #[test]
    fn a_shard_that_does_not_fit_in_memory_is_refused_rather_than_leased() {
        let mut w = worker("small", 1_000_000, 20_000);
        w.caps.shared_memory_bytes = 1 << 20;
        let mut c = shard("a", 0, 100);
        c.memory_bytes = 1 << 30;
        assert_eq!(
            fit(
                &w,
                &c,
                &Scale::of(std::slice::from_ref(&c)),
                1_000,
                None,
                &Weights::default()
            ),
            Err(Unfit::NotEnoughMemory)
        );
    }

    #[test]
    fn a_shard_the_node_cannot_finish_before_the_lease_expires_is_refused() {
        // One mCU is REFERENCE_OPS_PER_MCU reference ops; a node scoring 1
        // candidate per second retires one candidate's worth of ops per
        // second, so a shard of any size is far past even MAX_LEASE_SECONDS --
        // which is the bar now, since a merely slow node gets a longer lease
        // instead of a refusal. This one is not slow, it is hopeless.
        let w = worker("slow", 1, 20_000);
        let c = shard("a", 0, 1_000_000);
        assert_eq!(
            fit(
                &w,
                &c,
                &Scale::of(std::slice::from_ref(&c)),
                1_000,
                None,
                &Weights::default()
            ),
            Err(Unfit::SlowerThanLease)
        );
        // The same shard on a machine that can actually finish it is fine.
        let fast = worker("fast", 5_000_000, 20_000);
        assert!(
            fit(
                &fast,
                &c,
                &Scale::of(std::slice::from_ref(&c)),
                1_000,
                None,
                &Weights::default()
            )
            .is_ok()
        );
    }

    /// A benchmark this node never ran cannot be used to shorten its lease.
    #[test]
    fn an_unbenchmarked_node_gets_the_default_lease() {
        assert_eq!(
            lease_seconds_for(&caps(0, 20_000), 100),
            DEFAULT_LEASE_SECONDS
        );
    }

    /// The lease only ever grows. A quick machine finishing in a second must
    /// not be handed a one-second deadline: the headroom exists because the
    /// prediction is one benchmark against a reference machine, on hardware
    /// only lending a slice of itself.
    #[test]
    fn a_fast_node_is_never_given_less_than_the_default() {
        let quick = caps(5_000_000, 20_000);
        assert!(predicted_seconds(&quick, 10).unwrap() < DEFAULT_LEASE_SECONDS as f64);
        assert_eq!(lease_seconds_for(&quick, 10), DEFAULT_LEASE_SECONDS);
    }

    /// The point of the whole change: a machine that is merely slower than the
    /// default lease is given longer, in proportion to how much slower it is,
    /// rather than being refused the shard.
    #[test]
    fn a_slower_node_is_given_longer_rather_than_refused() {
        // Derived from the node's own rate rather than hard-coded, so the
        // shard lands squarely in the band that used to be an outright
        // exclusion -- past the default lease, inside the ceiling -- however
        // the benchmark-to-mCU arithmetic is later retuned.
        let modest = worker("modest", 20_000, 20_000);
        let target = DEFAULT_LEASE_SECONDS as f64 * 1.5;
        let reward = (mcu_per_second(&modest.caps).unwrap() * target) as i64;
        let predicted = predicted_seconds(&modest.caps, reward).unwrap();
        assert!(
            predicted > DEFAULT_LEASE_SECONDS as f64,
            "this test is only meaningful for a node the old gate would have refused"
        );

        let lease = lease_seconds_for(&modest.caps, reward);
        assert!(lease > DEFAULT_LEASE_SECONDS);
        assert!(lease <= MAX_LEASE_SECONDS);
        assert!(
            lease as f64 >= predicted,
            "a lease shorter than the prediction it was sized from guarantees the expiry \
             it exists to avoid"
        );

        let c = shard("a", 0, reward);
        assert!(
            fit(
                &modest,
                &c,
                &Scale::of(std::slice::from_ref(&c)),
                1_000,
                None,
                &Weights::default()
            )
            .is_ok(),
            "a node the mesh is willing to wait for must not be refused the work"
        );
    }

    /// Slower still is not a licence to park a shard indefinitely. Past the
    /// ceiling the refusal comes back, because at that point the mesh really
    /// would be better off giving the shard to somebody else.
    #[test]
    fn the_lease_stops_stretching_at_the_ceiling() {
        let glacial = worker("glacial", 1, 20_000);
        let reward = 1_000_000;
        assert_eq!(lease_seconds_for(&glacial.caps, reward), MAX_LEASE_SECONDS);

        let c = shard("a", 0, reward);
        assert_eq!(
            fit(
                &glacial,
                &c,
                &Scale::of(std::slice::from_ref(&c)),
                1_000,
                None,
                &Weights::default()
            ),
            Err(Unfit::SlowerThanLease)
        );
    }

    /// Waiting longer is a scheduling concession and must never become a
    /// pricing one. The reward is the shard's, not the holder's.
    /// The trap in sizing leases per node: if the hardware axis were scaled by
    /// the lease each node is granted rather than by a constant, a slower node
    /// would get a larger denominator and therefore a *better* hardware score.
    /// Being slow would rank you up. This pins the ordering the right way round.
    #[test]
    fn a_slower_node_still_scores_below_a_faster_one_on_hardware() {
        let w = Weights::default();
        let quick = worker("quick", 1_000_000, 20_000);
        let modest = worker("modest", 60_000, 20_000);
        let crawling = worker("crawling", 20_000, 20_000);

        // Sized so even the slowest of the three stays inside the ceiling --
        // the point here is the ordering among eligible nodes, not the gate.
        let reward =
            (mcu_per_second(&crawling.caps).unwrap() * MAX_LEASE_SECONDS as f64 * 0.9) as i64;
        let c = shard("s", 0, reward);
        let scale = Scale::of(std::slice::from_ref(&c));

        let h = |p: &WorkerProfile| fit(p, &c, &scale, 1_000, None, &w).unwrap().hardware;
        assert!(
            h(&quick) > h(&modest),
            "quick {} should out-score modest {}",
            h(&quick),
            h(&modest)
        );
        assert!(h(&modest) >= h(&crawling));
        // All three are still eligible -- ranking is the whole of the penalty.
        for p in [&quick, &modest, &crawling] {
            assert!(fit(p, &c, &scale, 1_000, None, &w).is_ok());
        }
    }

    #[test]
    fn a_longer_lease_does_not_change_what_the_shard_pays() {
        let c = shard("a", 0, 2_000_000);
        let modest = worker("modest", 20_000, 20_000);
        let quick = worker("quick", 5_000_000, 20_000);
        assert!(
            lease_seconds_for(&modest.caps, c.reward_mcu)
                > lease_seconds_for(&quick.caps, c.reward_mcu)
        );
        // `reward_mcu` is the shard's own field and nothing about scheduling
        // reads or rewrites it -- asserted here so a future change that made
        // the lease feed back into the price would fail loudly.
        assert_eq!(c.reward_mcu, 2_000_000);
    }

    #[test]
    fn measuring_nothing_scores_the_middle_and_never_the_top() {
        let blind = worker("blind", 0, 0);
        let f = score(&blind, &shard("a", 0, 100), 1_000);
        assert!((f.hardware - UNKNOWN).abs() < 1e-9);
        assert!((f.network - UNKNOWN).abs() < 1e-9);
        // A node that did measure, and is fast and near, must beat it on both.
        let measured = worker("measured", 5_000_000, 1_000);
        let m = score(&measured, &shard("a", 0, 100), 1_000);
        assert!(m.hardware > f.hardware);
        assert!(m.network > f.network);
    }

    #[test]
    fn a_distant_node_prefers_the_larger_shard() {
        let w = worker("far", 1_000_000, 250_000);
        let small = score(&w, &shard("small", 0, 10), 1_000);
        let large = score(&w, &shard("large", 1, 100_000), 1_000);
        assert!(
            large.network > small.network,
            "a round trip amortises better over more work: {} vs {}",
            large.network,
            small.network
        );
    }

    #[test]
    fn an_unproven_node_is_offered_the_cheaper_end_of_the_board() {
        let candidates = vec![shard("cheap", 0, 10), shard("dear", 1, 10_000)];
        let scale = Scale::of(&candidates);
        let unproven = worker("new", 1_000_000, 20_000);
        let mut proven = worker("old", 1_000_000, 20_000);
        proven.reputation = Reputation {
            accepted: 500,
            rejected: 0,
            streak: 500,
        };
        let weights = Weights::default();
        let cheap_new = fit(&unproven, &candidates[0], &scale, 1_000, None, &weights).unwrap();
        let dear_new = fit(&unproven, &candidates[1], &scale, 1_000, None, &weights).unwrap();
        assert!(
            cheap_new.reliability > dear_new.reliability,
            "an unproven node should be exposed to less"
        );
        let cheap_old = fit(&proven, &candidates[0], &scale, 1_000, None, &weights).unwrap();
        let dear_old = fit(&proven, &candidates[1], &scale, 1_000, None, &weights).unwrap();
        assert!(
            (cheap_old.reliability - dear_old.reliability).abs() < 0.05,
            "a proven record should not care about the size: {cheap_old:?} {dear_old:?}"
        );
        assert!(dear_old.reliability > dear_new.reliability);
    }

    #[test]
    fn a_record_of_rejections_is_not_erased_by_a_clean_streak() {
        // The audit rate forgets -- that is what makes it an audit rate. The
        // acceptance ratio does not, and standing folds in both.
        let fresh = Reputation::new();
        let redeemed = Reputation {
            accepted: 40,
            rejected: 40,
            streak: 40,
        };
        let clean = Reputation {
            accepted: 40,
            rejected: 0,
            streak: 40,
        };
        assert!(standing(&redeemed) > standing(&fresh));
        assert!(
            standing(&clean) > standing(&redeemed),
            "forty rejections have to still count"
        );
    }

    #[test]
    fn warmth_for_a_job_beats_a_cold_start_on_it() {
        let w = worker("warm", 1_000_000, 20_000);
        let cold = shard("cold", 7, 100);
        let mut warm = shard("warm", 8, 100);
        warm.shards_done_here = 3;
        warm.nearest_done_shard = Some(7);
        assert!(score(&w, &warm, 1_000).locality > score(&w, &cold, 1_000).locality);
    }

    #[test]
    fn the_adjacent_shard_beats_the_distant_one_on_the_same_job() {
        let w = worker("warm", 1_000_000, 20_000);
        let mut near = shard("near", 8, 100);
        near.shards_done_here = 1;
        near.nearest_done_shard = Some(7);
        let mut far = shard("far", 90, 100);
        far.shards_done_here = 1;
        far.nearest_done_shard = Some(7);
        assert!(score(&w, &near, 1_000).locality > score(&w, &far, 1_000).locality);
    }

    #[test]
    fn a_workload_with_no_model_is_not_penalised_for_having_no_cache_hit() {
        // The manifest term must be absent, not zero: averaging in a zero for
        // a signal that does not apply would make every CPU shard look cold.
        let w = worker("cpu", 1_000_000, 20_000);
        let mut done = shard("a", 1, 100);
        done.shards_done_here = 1;
        done.nearest_done_shard = Some(0);
        let cpu = score(&w, &done, 1_000).locality;

        let mut with_model = done.clone();
        with_model.required_manifests = vec!["sha256:absent".into()];
        let missing = score(&w, &with_model, 1_000).locality;
        assert!(cpu > missing, "{cpu} should beat {missing}");

        let mut cached = worker("cached", 1_000_000, 20_000);
        cached.caps.cached_model_manifests = vec!["sha256:absent".into()];
        assert!(score(&cached, &with_model, 1_000).locality > missing);
    }

    #[test]
    fn a_shard_that_waited_out_the_window_beats_any_fresh_candidate() {
        let w = worker("any", 1_000_000, 20_000);
        // A candidate constructed to fit perfectly, and a starved one built to
        // fit as badly as the gates allow.
        let mut ideal = shard("ideal", 1, 100);
        ideal.shards_done_here = 100;
        ideal.nearest_done_shard = Some(1);
        ideal.created_at = 10_000;
        let mut starved = shard("starved", 50, 100);
        starved.created_at = 10_000 - STARVATION_SECS;
        let now = 10_000;

        let candidates = vec![ideal.clone(), starved.clone()];
        let (picked, _) = best(&w, &candidates, now, None, &Weights::default(), 0).unwrap();
        assert_eq!(picked.assignment_id, "starved");

        // And the reason is structural, not a coincidence of these numbers:
        // the fit half can never reach the starvation bonus.
        let scale = Scale::of(&candidates);
        let f = fit(&w, &ideal, &scale, now, None, &Weights::default()).unwrap();
        assert!(f.total - f.starvation <= 1.0 + 1e-9);
    }

    #[test]
    fn crossing_a_region_costs_something_but_never_everything() {
        let mut near = worker("near", 1_000_000, 20_000);
        near.region = Some("eu".into());
        let mut far = worker("far", 1_000_000, 20_000);
        far.region = Some("us".into());
        let c = shard("a", 0, 100);
        let scale = Scale::of(std::slice::from_ref(&c));
        let w = Weights::default();
        let here = fit(&near, &c, &scale, 1_000, Some("eu"), &w).unwrap();
        let there = fit(&far, &c, &scale, 1_000, Some("eu"), &w).unwrap();
        assert!(here.network > there.network);
        assert!(
            there.network > 0.0,
            "a region boundary is a cost, not a ban"
        );
    }

    #[test]
    fn choosing_is_deterministic_when_two_shards_are_indistinguishable() {
        let w = worker("any", 1_000_000, 20_000);
        let candidates = vec![shard("bbb", 3, 100), shard("aaa", 3, 100)];
        let first = best(&w, &candidates, 1_000, None, &Weights::default(), 0)
            .unwrap()
            .0
            .assignment_id
            .clone();
        let reversed: Vec<_> = candidates.iter().rev().cloned().collect();
        let second = best(&w, &reversed, 1_000, None, &Weights::default(), 0)
            .unwrap()
            .0
            .assignment_id
            .clone();
        assert_eq!(first, "aaa");
        assert_eq!(first, second);
    }

    #[test]
    fn nothing_schedulable_is_reported_as_nothing_rather_than_a_bad_choice() {
        let mut w = worker("idle", 1_000_000, 20_000);
        w.caps.shared_logical_cpus = 0;
        assert!(
            best(
                &w,
                &[shard("a", 0, 100), shard("b", 1, 100)],
                1_000,
                None,
                &Weights::default(),
                0
            )
            .is_none()
        );
        assert!(
            best(
                &worker("ok", 1_000_000, 20_000),
                &[],
                1_000,
                None,
                &Weights::default(),
                0
            )
            .is_none()
        );
    }

    // -- the resource graph ------------------------------------------------

    fn at(x: i64, y: i64) -> NetworkCoordinate {
        NetworkCoordinate {
            vector_micros: [x, y, 0],
            height_micros: 0,
            error_permille: 100,
        }
    }

    fn vertex(node_id: &str, coordinate: Option<NetworkCoordinate>) -> Vertex {
        let mut c = caps(1_000_000, 20_000);
        c.network_coordinate = coordinate;
        Vertex {
            node_id: node_id.into(),
            region: None,
            caps: c,
            reputation: Reputation::new(),
            online: true,
        }
    }

    #[test]
    fn a_cluster_is_chosen_by_the_edges_and_not_by_the_vertices() {
        // Three machines in one place and two far away. Ranking nodes one at a
        // time cannot see the difference; the graph can.
        let graph = ResourceGraph::new(vec![
            vertex("a", Some(at(0, 0))),
            vertex("b", Some(at(1_000, 0))),
            vertex("c", Some(at(2_000, 0))),
            vertex("d", Some(at(900_000, 0))),
            vertex("e", Some(at(901_000, 0))),
        ]);
        let cluster = graph.cluster(3, |_| true).expect("three of five");
        assert_eq!(cluster.node_ids, vec!["a", "b", "c"]);
        let pair = graph.cluster(2, |_| true).unwrap();
        assert!(pair.worst_edge_micros < cluster.worst_edge_micros);
    }

    #[test]
    fn unmeasured_distance_is_never_mistaken_for_proximity() {
        let graph = ResourceGraph::new(vec![
            vertex("known-a", Some(at(0, 0))),
            vertex("known-b", Some(at(5_000, 0))),
            {
                let mut v = vertex("blind-a", None);
                v.caps.coordinator_latency_micros = 0;
                v
            },
            {
                let mut v = vertex("blind-b", None);
                v.caps.coordinator_latency_micros = 0;
                v
            },
        ]);
        assert_eq!(graph.edge_micros(2, 3), UNKNOWN_EDGE_MICROS);
        assert!(graph.edge_micros(0, 1) < graph.edge_micros(2, 3));
        assert_eq!(
            graph.cluster(2, |_| true).unwrap().node_ids,
            vec!["known-a", "known-b"]
        );
    }

    #[test]
    fn an_unplaced_node_falls_back_to_the_route_through_the_coordinator() {
        let mut graph = ResourceGraph::new(vec![vertex("a", None), vertex("b", None)]);
        graph.vertices[0].caps.coordinator_latency_micros = 5_000;
        graph.vertices[1].caps.coordinator_latency_micros = 7_000;
        assert_eq!(graph.edge_micros(0, 1), 12_000);
        assert_eq!(graph.edge_micros(0, 0), 0);
    }

    #[test]
    fn a_gate_is_how_a_cluster_is_kept_inside_one_region() {
        let mut graph = ResourceGraph::new(vec![
            vertex("eu-1", Some(at(0, 0))),
            vertex("eu-2", Some(at(80_000, 0))),
            vertex("us-1", Some(at(1_000, 0))),
            vertex("us-2", Some(at(1_100, 0))),
        ]);
        graph.vertices[0].region = Some("eu".into());
        graph.vertices[1].region = Some("eu".into());
        graph.vertices[2].region = Some("us".into());
        graph.vertices[3].region = Some("us".into());
        // Unconstrained, the two American machines are closest.
        assert_eq!(
            graph.cluster(2, |_| true).unwrap().node_ids,
            vec!["us-1", "us-2"]
        );
        // Gated, the only European pair is returned even though it is worse.
        let eu = graph
            .cluster(2, |v| v.region.as_deref() == Some("eu"))
            .unwrap();
        assert_eq!(eu.node_ids, vec!["eu-1", "eu-2"]);
    }

    #[test]
    fn a_cluster_larger_than_the_field_is_none_rather_than_a_short_one() {
        let graph = ResourceGraph::new(vec![vertex("a", Some(at(0, 0)))]);
        assert!(graph.cluster(2, |_| true).is_none());
        assert!(graph.cluster(0, |_| true).is_none());
        assert!(ResourceGraph::default().cluster(1, |_| true).is_none());
    }

    #[test]
    fn an_offline_machine_is_not_clustered_onto() {
        let mut graph = ResourceGraph::new(vec![
            vertex("a", Some(at(0, 0))),
            vertex("gone", Some(at(1, 0))),
            vertex("b", Some(at(9_000, 0))),
        ]);
        graph.vertices[1].online = false;
        assert_eq!(graph.cluster(2, |_| true).unwrap().node_ids, vec!["a", "b"]);
    }

    // -- The head start ----------------------------------------------------
    //
    // These lock the property the mechanism exists for: it changes who goes
    // *first*, and never who is allowed to go at all.

    /// The reserve costs nothing when there is work spare, which is the state a
    /// healthy mesh spends most of its time in.
    #[test]
    fn with_a_shard_for_everyone_nobody_waits() {
        let slow = worker("slow", 20_000, 20_000);
        let field = [
            shard_taking(&slow, "a", 1_000.0),
            shard_taking(&slow, "b", 1_000.0),
        ];
        // Two nodes asking, two shards on offer: not contended.
        let scale = Scale::of(&field).with_recent_pollers(2);
        assert!(
            fit(
                &slow,
                &field[0],
                &scale,
                field[0].created_at,
                None,
                &Weights::default()
            )
            .is_ok()
        );
    }

    /// With more nodes than shards, a fresh shard goes to the fastest asker and
    /// the slower one is told to come back.
    #[test]
    fn on_fresh_work_the_faster_node_goes_first() {
        let quick = worker("quick", 100_000_000, 20_000);
        let slow = worker("slow", 20_000, 20_000);
        let field = [shard_taking(&slow, "a", 1_000.0)];
        let scale = Scale::of(&field).with_recent_pollers(4);
        let now = field[0].created_at;
        let w = Weights::default();

        assert!(fit(&quick, &field[0], &scale, now, None, &w).is_ok());
        assert_eq!(
            fit(&slow, &field[0], &scale, now, None, &w),
            Err(Unfit::StillReservedForFasterNodes)
        );
    }

    /// And the wait is over in `HEAD_START_SECONDS` whatever the machine, which
    /// is what keeps a head start from turning into the exclusion it replaced.
    #[test]
    fn no_node_waits_longer_than_the_head_start() {
        let slow = worker("slow", 20_000, 20_000);
        let field = [shard_taking(&slow, "a", 1_000.0)];
        let scale = Scale::of(&field).with_recent_pollers(4);
        let w = Weights::default();
        assert!(
            fit(
                &slow,
                &field[0],
                &scale,
                field[0].created_at + HEAD_START_SECONDS,
                None,
                &w
            )
            .is_ok()
        );
    }

    /// A node fast enough to be at the front of the queue is never held back,
    /// however contended the mesh is.
    #[test]
    fn the_fastest_node_never_waits() {
        let quick = worker("quick", 100_000_000, 20_000);
        let field = [shard("a", 0, 1_000)];
        let scale = Scale::of(&field).with_recent_pollers(1_000);
        assert!(
            fit(
                &quick,
                &field[0],
                &scale,
                field[0].created_at,
                None,
                &Weights::default()
            )
            .is_ok()
        );
    }

    /// The wait is graded rather than binary: a middling machine is behind the
    /// quick one and ahead of the crawling one, by a measured amount.
    #[test]
    fn a_slower_machine_waits_longer_than_a_less_slow_one() {
        let modest = worker("modest", 200_000, 20_000);
        let crawling = worker("crawling", 20_000, 20_000);
        // Sized so the modest machine is part-way down the axis and the
        // crawling one is pinned at the bottom of it.
        let field = [shard_taking(&modest, "a", 200.0)];
        let scale = Scale::of(&field).with_recent_pollers(9);
        let w = Weights::default();
        let waited = |worker: &WorkerProfile| {
            (0..=HEAD_START_SECONDS)
                .find(|d| fit(worker, &field[0], &scale, field[0].created_at + d, None, &w).is_ok())
                .expect("every node is eligible by the end of the head start")
        };
        let modest = waited(&modest);
        let crawling = waited(&crawling);
        assert!(
            modest < crawling,
            "a faster machine should wait less: {modest}s vs {crawling}s"
        );
        assert!(crawling <= HEAD_START_SECONDS);
    }

    /// A caller that measured no demand gets the behaviour that existed before
    /// the head start did.
    #[test]
    fn without_a_demand_reading_the_head_start_is_off() {
        let slow = worker("slow", 20_000, 20_000);
        let field = [shard_taking(&slow, "a", 1_000.0)];
        assert!(
            fit(
                &slow,
                &field[0],
                &Scale::of(&field),
                field[0].created_at,
                None,
                &Weights::default()
            )
            .is_ok()
        );
    }
}
