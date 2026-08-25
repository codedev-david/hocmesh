use mesh_gpu::benchmark_memory;
use mesh_protocol::{GpuCapability, NodeCapabilities, PROTOCOL_VERSION};
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
    let gpus = mesh_gpu::discover_devices()
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
        gpus,
        model_seed_url,
        cached_model_manifests,
        coordinator_latency_micros: 0,
        model_bandwidth_kbps: 100_000,
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

/// Overwrite the advertised shared-capacity fields from the operator's limits.
///
/// Advertising the *shared* slice rather than the whole machine keeps the
/// scheduler honest and avoids disclosing the full hardware profile of a
/// contributor's machine to the rest of the mesh.
pub fn apply_limits(caps: &mut NodeCapabilities, limits: &ResourceLimits) {
    caps.shared_logical_cpus = limits.effective_workers(caps.logical_cpus);
    caps.shared_memory_bytes = limits.shared_memory_bytes(caps.total_memory_bytes);
    caps.shared_gpu_percent = limits.gpu_percent;
    if !limits.offers_gpu() {
        caps.gpus.clear();
        caps.ai_runtime_ready = false;
    }
}
