use mesh_protocol::{GpuCapability, NodeCapabilities, PROTOCOL_VERSION};
use std::{process::Command, time::Instant};
use sysinfo::System;

use crate::compute::count_primes;

pub fn detect_capabilities(run_benchmark: bool) -> NodeCapabilities {
    let system = System::new_all();
    let cpu_brand = system
        .cpus()
        .first()
        .map(|cpu| cpu.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    NodeCapabilities {
        protocol_version: PROTOCOL_VERSION,
        hostname: System::host_name().unwrap_or_else(|| "unknown".to_string()),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_brand,
        logical_cpus: system.cpus().len().max(1),
        total_memory_bytes: system.total_memory(),
        cpu_benchmark_score: if run_benchmark { benchmark_cpu() } else { 0 },
        gpus: detect_gpus(),
    }
}

/// Returns a simple deterministic CPU throughput score (candidate integers/sec).
pub fn benchmark_cpu() -> u64 {
    const END: u64 = 150_000;
    let started = Instant::now();
    let _ = count_primes(2, END);
    let elapsed = started.elapsed().as_secs_f64().max(0.000_001);
    ((END - 2) as f64 / elapsed) as u64
}

fn detect_gpus() -> Vec<GpuCapability> {
    let mut gpus = Vec::new();
    gpus.extend(detect_nvidia());
    if gpus.is_empty() && std::env::consts::OS == "macos" {
        gpus.extend(detect_apple_gpu());
    }
    gpus
}

fn detect_nvidia() -> Vec<GpuCapability> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let mut parts = line.splitn(2, ',');
            let name = parts.next()?.trim();
            let memory = parts.next()?.trim().parse::<u64>().ok();
            if name.is_empty() {
                return None;
            }
            Some(GpuCapability {
                vendor: "nvidia".to_string(),
                name: name.to_string(),
                backend: "cuda".to_string(),
                memory_mb: memory,
            })
        })
        .collect()
}

fn detect_apple_gpu() -> Vec<GpuCapability> {
    let output = Command::new("system_profiler")
        .arg("SPDisplaysDataType")
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let name = text
        .lines()
        .find_map(|line| line.trim().strip_prefix("Chipset Model:"))
        .map(str::trim)
        .filter(|s| !s.is_empty());

    name.map(|name| {
        vec![GpuCapability {
            vendor: "apple".to_string(),
            name: name.to_string(),
            backend: "metal".to_string(),
            memory_mb: None,
        }]
    })
    .unwrap_or_default()
}
