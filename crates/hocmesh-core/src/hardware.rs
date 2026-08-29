use hocmesh_gpu::benchmark_memory;
use hocmesh_protocol::{GpuCapability, NodeCapabilities, PROTOCOL_VERSION};
use std::time::Instant;
use sysinfo::System;

use crate::compute::count_primes;
use crate::limits::ResourceLimits;

pub fn detect_capabilities(run_benchmark: bool) -> NodeCapabilities {
    detect_capabilities_with_models(run_benchmark, None, Vec::new())
}

pub fn detect_capabilities_with_models(
    run_benchmark: bool,
    model_seed_url: Option<String>,
    cached_model_manifests: Vec<String>,
) -> NodeCapabilities {
    let system = System::new_all();
    let cpu_brand = system
        .cpus()
        .first()
        .map(|cpu| cpu.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    let gpus = hocmesh_gpu::discover_devices()
        .into_iter()
        .map(|device| {
            let benchmark = run_benchmark.then(|| benchmark_memory(&device, 8 * 1024 * 1024, 8));
            GpuCapability {
                stable_id: device.stable_id,
                vendor: device.vendor,
                name: device.name,
                backend: format!("{:?}", device.backend).to_lowercase(),
                memory_mb: device.memory_bytes.map(|bytes| bytes / 1024 / 1024),
                driver_version: device.driver_version,
                compute_version: device.compute_version,
                supports_fp16: device.supports_fp16,
                supports_bf16: device.supports_bf16,
                supports_int8: device.supports_int8,
                benchmark_bytes_per_second: benchmark
                    .as_ref()
                    .map(|report| report.throughput_units_per_second as u64),
                benchmark_p95_micros: benchmark
                    .as_ref()
                    .map(|report| (report.latency_p95_ms * 1000.0) as u64),
            }
        })
        .collect();

    let mut caps = NodeCapabilities {
        protocol_version: PROTOCOL_VERSION,
        hostname: System::host_name().unwrap_or_else(|| "unknown".to_string()),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_brand,
        logical_cpus: system.cpus().len().max(1),
        total_memory_bytes: system.total_memory(),
        cpu_benchmark_score: if run_benchmark { benchmark_cpu() } else { 0 },
        memory_bandwidth_bytes_per_second: run_benchmark.then(benchmark_memory_bandwidth).flatten(),
        gpus,
        model_seed_url,
        cached_model_manifests,
        coordinator_latency_micros: 0,
        // Zero means "nothing has measured this link", which is what is true
        // of a machine that has not yet served a byte. A constant here made
        // every node in the hocmesh report the same uplink, so any policy that
        // gated on it was gating on nothing while looking like it worked.
        model_bandwidth_kbps: 0,
        accelerator_load_permille: 0,
        ai_runtime_ready: false,
        // Fail safe: until an operator's limits are applied, advertise only the
        // conservative default share rather than the whole machine.
        shared_logical_cpus: 0,
        shared_memory_bytes: 0,
        shared_gpu_percent: 0,
        network_coordinate: None,
        probe_endpoint: None,
    };
    apply_limits(&mut caps, &ResourceLimits::default());
    caps
}

/// Returns a deterministic CPU throughput score (candidate integers/second).
pub fn benchmark_cpu() -> u64 {
    const END: u64 = 150_000;
    let started = Instant::now();
    let _ = count_primes(2, END);
    let elapsed = started.elapsed().as_secs_f64().max(0.000_001);
    ((END - 2) as f64 / elapsed) as u64
}

/// Bytes of main memory this machine can stream per second.
///
/// This is the number that decides how fast a machine can generate tokens, and
/// it is not the one `benchmark_cpu` reports. Generating a token re-reads every
/// weight the stage holds, so the step time is `bytes / bandwidth` and the core
/// spends most of it waiting. Two machines with the same arithmetic score and
/// different memory can differ several-fold here, which is why the planner
/// needs this measured rather than inferred.
///
/// Measured over a buffer far larger than any last-level cache, so what is
/// timed is the trip to DRAM rather than a cache that would flatter a machine
/// with a big L3 and slow memory -- the opposite of what the planner needs.
/// Four accumulators keep the loop waiting on memory instead of on the
/// dependency chain between one addition and the next, which would measure the
/// adder. The best pass wins rather than the mean: every source of error here
/// is something else stealing the machine, so the fastest observation is the
/// one least contaminated.
///
/// Returns `None` rather than a guess if the machine cannot spare the buffer or
/// the clock reports nothing usable. A wrong bandwidth is worse than an unknown
/// one -- unknown falls back to an even split, wrong silently overloads a stage.
pub fn benchmark_memory_bandwidth() -> Option<u64> {
    const WORDS: usize = 8 * 1024 * 1024; // 64 MiB of u64
    const PASSES: usize = 3;

    let mut buffer = Vec::new();
    if buffer.try_reserve_exact(WORDS).is_err() {
        return None;
    }
    buffer.extend((0..WORDS).map(|i| i as u64));

    let bytes = std::mem::size_of_val(buffer.as_slice()) as f64;
    let mut best = 0.0f64;
    for _ in 0..PASSES {
        let started = Instant::now();
        let (mut a, mut b, mut c, mut d) = (0u64, 0u64, 0u64, 0u64);
        for chunk in buffer.chunks_exact(4) {
            a = a.wrapping_add(chunk[0]);
            b = b.wrapping_add(chunk[1]);
            c = c.wrapping_add(chunk[2]);
            d = d.wrapping_add(chunk[3]);
        }
        // Without this the whole loop is dead code and the compiler is entitled
        // to delete it, which would time an empty function.
        std::hint::black_box(a ^ b ^ c ^ d);
        let elapsed = started.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            best = best.max(bytes / elapsed);
        }
    }
    (best.is_finite() && best > 0.0).then_some(best as u64)
}

/// Overwrite the advertised shared-capacity fields from the operator's limits.
///
/// Advertising the *shared* slice rather than the whole machine keeps the
/// scheduler honest and avoids disclosing the full hardware profile of a
/// contributor's machine to the rest of the hocmesh.
pub fn apply_limits(caps: &mut NodeCapabilities, limits: &ResourceLimits) {
    caps.shared_logical_cpus = limits.effective_workers(caps.logical_cpus);
    caps.shared_memory_bytes = limits.shared_memory_bytes(caps.total_memory_bytes);
    caps.shared_gpu_percent = limits.gpu_percent;
    if !limits.offers_gpu() {
        caps.gpus.clear();
        caps.ai_runtime_ready = false;
    }
}

/// Identifier for the shared CPU slice when it is advertised as a device.
///
/// Re-exported rather than restated. The scheduler places work on this id and
/// the AI worker resolves an assignment back to a device by it, so the name has
/// to be one constant: two that merely happened to match would be a rename away
/// from a node advertising work it then refused to run.
pub use hocmesh_gpu::SHARED_CPU_DEVICE_ID;

/// Settle whether this node offers inference to the hocmesh, and on what.
///
/// Call this *after* [`apply_limits`], which is what decides whether any
/// accelerator is still advertised.
///
/// Two things have to be true. The node must be able to run inference at all,
/// which is `runtime_available`; and the operator must have agreed to run it
/// for other people, which is [`ResourceLimits::offers_ai`]. Installing a
/// runtime is not consent -- an operator may want `hocmesh infer` for
/// themselves and nothing else -- so the two are kept apart.
///
/// A node that agrees but has no accelerator left to agree *with* advertises
/// its shared CPU slice as a device. The scheduler places work on devices, so
/// a node with none is a node it cannot reach; that is what used to make a
/// CPU-only machine unable to serve the hocmesh however willing its operator
/// was. The device is not a fiction: `--gpu-layers 0` against llama.cpp's CPU
/// backend is how the AI worker already runs, and it is what `runtime-install`
/// installs.
pub fn apply_ai_readiness(
    caps: &mut NodeCapabilities,
    limits: &ResourceLimits,
    runtime_available: bool,
) {
    // Drop a slice this function added on an earlier pass before deciding, so
    // that re-applying lands where applying once did instead of stacking
    // devices and reading its own output back as "an accelerator is shared".
    caps.gpus
        .retain(|device| device.stable_id != SHARED_CPU_DEVICE_ID);
    let accelerator_shared = !caps.gpus.is_empty();
    caps.ai_runtime_ready = runtime_available && limits.offers_ai(accelerator_shared);
    if caps.ai_runtime_ready && !accelerator_shared {
        let device = shared_cpu_device(caps);
        caps.gpus.push(device);
    }
}

/// The lent CPU slice, described the way the scheduler describes a device.
///
/// The identity and the capability flags come from [`hocmesh_gpu::cpu_device`],
/// which is the same description the AI worker will resolve the assignment
/// against. Only the two facts that crate cannot know are added here: what this
/// particular CPU is called, and how much of this machine the operator lent.
fn shared_cpu_device(caps: &NodeCapabilities) -> GpuCapability {
    let device = hocmesh_gpu::cpu_device();
    GpuCapability {
        stable_id: device.stable_id,
        vendor: device.vendor,
        name: caps.cpu_brand.clone(),
        // The coordinator already maps this backend onto `BackendKind::Cpu`,
        // and a request that genuinely needs CUDA still names it in
        // `required_backends` and so still refuses to land here.
        backend: format!("{:?}", device.backend).to_lowercase(),
        // The lent slice, not the machine. `--memory-percent` therefore
        // governs which models may be placed on this node, which is the whole
        // reason the operator was asked for a percentage.
        memory_mb: Some(caps.shared_memory_bytes / (1024 * 1024)),
        driver_version: device.driver_version,
        compute_version: device.compute_version,
        supports_fp16: device.supports_fp16,
        supports_bf16: device.supports_bf16,
        supports_int8: device.supports_int8,
        // Left unmeasured rather than guessed: these are memory-bandwidth
        // numbers from `benchmark_memory`, and inventing one would put a
        // fabricated figure in front of the scheduler.
        benchmark_bytes_per_second: None,
        benchmark_p95_micros: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a_gpu() -> GpuCapability {
        GpuCapability {
            stable_id: "gpu-0".into(),
            vendor: "test".into(),
            name: "Test GPU".into(),
            backend: "cpu".into(),
            memory_mb: Some(8_192),
            driver_version: None,
            compute_version: None,
            supports_fp16: true,
            supports_bf16: false,
            supports_int8: true,
            benchmark_bytes_per_second: None,
            benchmark_p95_micros: None,
        }
    }

    /// A machine with known hardware, before any operator limit is applied.
    fn a_machine() -> NodeCapabilities {
        let mut caps = detect_capabilities(false);
        caps.logical_cpus = 16;
        caps.total_memory_bytes = 32_000_000_000;
        caps.gpus = vec![a_gpu()];
        caps.ai_runtime_ready = true;
        caps
    }

    /// What the hocmesh is told is the operator's share, not the machine. The
    /// detected totals stay put so the share can be recomputed if the limits
    /// change; only the advertised slice moves.
    #[test]
    fn limits_shrink_what_is_advertised_not_what_was_detected() {
        let mut caps = a_machine();
        apply_limits(
            &mut caps,
            &ResourceLimits {
                cpu_percent: 25,
                gpu_percent: 60,
                memory_percent: 50,
                ai: None,
            },
        );

        assert_eq!(caps.shared_logical_cpus, 4);
        assert_eq!(caps.shared_memory_bytes, 16_000_000_000);
        assert_eq!(caps.shared_gpu_percent, 60);
        assert_eq!(caps.logical_cpus, 16, "the machine did not change size");
        assert_eq!(caps.total_memory_bytes, 32_000_000_000);
        assert_eq!(caps.gpus.len(), 1, "a lent GPU is still advertised");
        assert!(caps.ai_runtime_ready);
    }

    /// Lending no GPU has to mean the hocmesh never learns there is one. Leaving
    /// the GPU in the advertisement would keep the node eligible for AI work
    /// it has been told not to do, and would disclose hardware the operator
    /// deliberately kept back.
    #[test]
    fn a_withheld_gpu_is_not_advertised_at_all() {
        let mut caps = a_machine();
        apply_limits(
            &mut caps,
            &ResourceLimits {
                cpu_percent: 50,
                gpu_percent: 0,
                memory_percent: 50,
                ai: None,
            },
        );

        assert!(
            caps.gpus.is_empty(),
            "a GPU nobody may use is nobody's business"
        );
        assert!(
            !caps.ai_runtime_ready,
            "a node with no GPU to lend must not be offered AI work"
        );
        assert_eq!(caps.shared_gpu_percent, 0);
        assert_eq!(caps.shared_logical_cpus, 8, "the CPU share is unaffected");
    }

    #[test]
    fn an_unstated_operator_gets_exactly_the_old_rule() {
        // Sharing a GPU offered inference before this function existed, and
        // still does.
        let with_gpu = ResourceLimits {
            gpu_percent: 50,
            ..Default::default()
        };
        let mut sharing = a_machine();
        apply_limits(&mut sharing, &with_gpu);
        apply_ai_readiness(&mut sharing, &with_gpu, true);
        assert!(sharing.ai_runtime_ready);
        assert_eq!(
            sharing.gpus.len(),
            1,
            "no invented device alongside a real one"
        );
        assert_eq!(sharing.gpus[0].stable_id, "gpu-0");

        // Not sharing one did not, and still does not.
        let no_gpu = ResourceLimits {
            gpu_percent: 0,
            ..Default::default()
        };
        let mut declining = a_machine();
        apply_limits(&mut declining, &no_gpu);
        apply_ai_readiness(&mut declining, &no_gpu, true);
        assert!(
            !declining.ai_runtime_ready,
            "an operator who never said yes must not be volunteered by an upgrade"
        );
        assert!(declining.gpus.is_empty());
    }

    #[test]
    fn a_cpu_only_node_that_opts_in_advertises_its_shared_cpu_slice() {
        let limits = ResourceLimits {
            memory_percent: 50,
            ai: Some(true),
            ..Default::default()
        };
        let mut caps = a_machine();
        caps.gpus.clear();
        apply_limits(&mut caps, &limits);
        apply_ai_readiness(&mut caps, &limits, true);

        assert!(caps.ai_runtime_ready);
        let device = caps
            .gpus
            .first()
            .expect("a node the scheduler can reach has at least one device");
        assert_eq!(device.backend, "cpu");
        // The lent half of 32 GB, not the whole machine.
        assert_eq!(device.memory_mb, Some(16_000_000_000 / (1024 * 1024)));
    }

    #[test]
    fn opting_in_without_a_runtime_offers_nothing() {
        // Willingness is not capability. Advertising readiness here would draw
        // work this node would then fail.
        let limits = ResourceLimits {
            ai: Some(true),
            ..Default::default()
        };
        let mut caps = a_machine();
        caps.gpus.clear();
        apply_limits(&mut caps, &limits);
        apply_ai_readiness(&mut caps, &limits, false);
        assert!(!caps.ai_runtime_ready);
        assert!(caps.gpus.is_empty());
    }

    #[test]
    fn opting_out_declines_even_with_a_gpu_and_a_runtime() {
        let limits = ResourceLimits {
            gpu_percent: 100,
            ai: Some(false),
            ..Default::default()
        };
        let mut caps = a_machine();
        apply_limits(&mut caps, &limits);
        apply_ai_readiness(&mut caps, &limits, true);
        assert!(!caps.ai_runtime_ready);
        assert!(
            !caps.gpus.is_empty(),
            "the GPU is still lent for everything else; only inference was declined"
        );
    }

    #[test]
    fn readiness_does_not_stack_cpu_devices_across_restarts() {
        // A daemon re-detects and re-applies on every start, and an operator
        // may toggle limits between them. Applying twice must land where
        // applying once did.
        let limits = ResourceLimits {
            ai: Some(true),
            ..Default::default()
        };
        let mut caps = a_machine();
        caps.gpus.clear();
        apply_limits(&mut caps, &limits);
        apply_ai_readiness(&mut caps, &limits, true);
        let once = caps.gpus.clone();

        apply_limits(&mut caps, &limits);
        apply_ai_readiness(&mut caps, &limits, true);
        assert_eq!(caps.gpus, once);
    }

    /// `hocmesh init` and the daemon both apply limits to a freshly detected
    /// machine, and an operator may change them and restart. Applying twice
    /// must land in the same place, or the share would ratchet downwards.
    #[test]
    fn applying_the_same_limits_twice_changes_nothing() {
        let limits = ResourceLimits {
            cpu_percent: 30,
            gpu_percent: 40,
            memory_percent: 30,
            ai: None,
        };
        let mut once = a_machine();
        apply_limits(&mut once, &limits);
        let mut twice = once.clone();
        apply_limits(&mut twice, &limits);
        assert_eq!(once, twice);
    }

    /// A tiny share of a big machine still has to be able to do one thing at
    /// a time, and a generous share must never promise more than exists.
    #[test]
    fn the_worker_count_stays_between_one_and_the_whole_machine() {
        for (cpu_percent, cpus, expected) in [(1u8, 16usize, 1usize), (100, 16, 16), (50, 1, 1)] {
            let mut caps = a_machine();
            caps.logical_cpus = cpus;
            apply_limits(
                &mut caps,
                &ResourceLimits {
                    cpu_percent,
                    gpu_percent: 50,
                    memory_percent: 50,
                    ai: None,
                },
            );
            assert_eq!(
                caps.shared_logical_cpus, expected,
                "{cpu_percent}% of {cpus} CPUs"
            );
        }
    }
}
