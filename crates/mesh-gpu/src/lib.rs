use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Cuda,
    Rocm,
    Metal,
    Cpu,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeviceCapability {
    pub stable_id: String,
    pub backend: BackendKind,
    pub vendor: String,
    pub name: String,
    pub memory_bytes: Option<u64>,
    pub driver_version: Option<String>,
    pub compute_version: Option<String>,
    pub supports_fp16: bool,
    pub supports_bf16: bool,
    pub supports_int8: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkReport {
    pub device_id: String,
    pub backend: BackendKind,
    pub warmup_iterations: u32,
    pub measured_iterations: u32,
    pub elapsed_ms: u64,
    pub throughput_units_per_second: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub benchmark: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature_milli: u32,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceOutput {
    pub text: String,
    pub backend: BackendKind,
    pub elapsed_ms: u64,
}

pub trait InferenceBackend: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn device(&self) -> &DeviceCapability;
    fn infer(&self, model: &Path, request: &InferenceRequest) -> Result<InferenceOutput>;
}

/// A real llama.cpp process adapter. The selected llama.cpp build determines
/// whether execution uses CUDA, ROCm/HIP, Metal, or CPU. MESH passes only a
/// fixed allow-list of arguments and never executes model-supplied commands.
pub struct LlamaCppBackend {
    executable: PathBuf,
    device: DeviceCapability,
    gpu_layers: u32,
}

impl LlamaCppBackend {
    pub fn new(
        executable: impl Into<PathBuf>,
        device: DeviceCapability,
        gpu_layers: u32,
    ) -> Result<Self> {
        let executable = executable.into();
        ensure!(executable.is_file(), "llama.cpp executable does not exist");
        Ok(Self {
            executable,
            device,
            gpu_layers,
        })
    }

    fn arguments(&self, model: &Path, request: &InferenceRequest) -> Vec<String> {
        vec![
            "--model".into(),
            model.display().to_string(),
            "--prompt".into(),
            request.prompt.clone(),
            "--n-predict".into(),
            request.max_tokens.to_string(),
            "--temp".into(),
            format!("{:.3}", request.temperature_milli as f64 / 1000.0),
            "--seed".into(),
            request.seed.to_string(),
            "--gpu-layers".into(),
            self.gpu_layers.to_string(),
            "--no-display-prompt".into(),
        ]
    }
}

impl InferenceBackend for LlamaCppBackend {
    fn kind(&self) -> BackendKind {
        self.device.backend
    }
    fn device(&self) -> &DeviceCapability {
        &self.device
    }

    fn infer(&self, model: &Path, request: &InferenceRequest) -> Result<InferenceOutput> {
        ensure!(model.is_file(), "model file does not exist");
        ensure!(request.max_tokens > 0, "max_tokens must be positive");
        let started = Instant::now();
        let output = Command::new(&self.executable)
            .args(self.arguments(model, request))
            .output()
            .context("launching llama.cpp runtime")?;
        if !output.status.success() {
            bail!(
                "llama.cpp inference failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        Ok(InferenceOutput {
            text: String::from_utf8(output.stdout).context("runtime returned non-UTF-8 output")?,
            backend: self.device.backend,
            elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        })
    }
}

macro_rules! native_backend {
    ($feature:literal, $name:ident, $kind:expr) => {
        #[cfg(feature = $feature)]
        pub struct $name(LlamaCppBackend);

        #[cfg(feature = $feature)]
        impl $name {
            pub fn new(
                executable: impl Into<PathBuf>,
                device: DeviceCapability,
                gpu_layers: u32,
            ) -> Result<Self> {
                ensure!(
                    device.backend == $kind,
                    "device does not match backend adapter"
                );
                Ok(Self(LlamaCppBackend::new(executable, device, gpu_layers)?))
            }
        }

        #[cfg(feature = $feature)]
        impl InferenceBackend for $name {
            fn kind(&self) -> BackendKind {
                self.0.kind()
            }
            fn device(&self) -> &DeviceCapability {
                self.0.device()
            }
            fn infer(&self, model: &Path, request: &InferenceRequest) -> Result<InferenceOutput> {
                self.0.infer(model, request)
            }
        }
    };
}

native_backend!("cuda", CudaBackend, BackendKind::Cuda);
native_backend!("rocm", RocmBackend, BackendKind::Rocm);
native_backend!("metal", MetalBackend, BackendKind::Metal);

/// Runs llama-bench against a real model/backend and extracts measured token
/// throughput from its JSON output. Unlike `benchmark_memory`, this exercises
/// the selected inference runtime and accelerator.
pub fn benchmark_llama_cpp(
    executable: &Path,
    model: &Path,
    device: &DeviceCapability,
    gpu_layers: u32,
) -> Result<BenchmarkReport> {
    ensure!(
        executable.is_file(),
        "llama-bench executable does not exist"
    );
    ensure!(model.is_file(), "model file does not exist");
    let started = Instant::now();
    let output = Command::new(executable)
        .args([
            "--model",
            &model.display().to_string(),
            "--n-gpu-layers",
            &gpu_layers.to_string(),
            "--output",
            "json",
            "--prompt",
            "128",
            "--generation",
            "32",
            "--repetitions",
            "3",
        ])
        .output()
        .context("launching llama-bench")?;
    if !output.status.success() {
        bail!(
            "llama-bench failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let json: serde_json::Value =
        serde_json::from_slice(&output.stdout).context("parsing llama-bench JSON")?;
    let throughput = find_metric(&json, &["avg_ts", "tokens_per_second", "t_s"])
        .context("llama-bench output has no token throughput metric")?;
    ensure!(
        throughput.is_finite() && throughput > 0.0,
        "invalid benchmark throughput"
    );
    let latency_ms = 1000.0 / throughput;
    Ok(BenchmarkReport {
        device_id: device.stable_id.clone(),
        backend: device.backend,
        warmup_iterations: 0,
        measured_iterations: 3,
        elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
        throughput_units_per_second: throughput,
        latency_p50_ms: latency_ms,
        latency_p95_ms: latency_ms,
        benchmark: "llama_bench_tokens_v1".into(),
    })
}

fn find_metric(value: &serde_json::Value, keys: &[&str]) -> Option<f64> {
    match value {
        serde_json::Value::Object(map) => keys
            .iter()
            .find_map(|key| map.get(*key).and_then(|v| v.as_f64()))
            .or_else(|| map.values().find_map(|value| find_metric(value, keys))),
        serde_json::Value::Array(values) => {
            values.iter().find_map(|value| find_metric(value, keys))
        }
        _ => None,
    }
}

pub fn discover_devices() -> Vec<DeviceCapability> {
    let mut devices = discover_cuda();
    devices.extend(discover_rocm());
    devices.extend(discover_metal());
    devices
}

pub fn discover_cuda() -> Vec<DeviceCapability> {
    let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=uuid,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    parse_cuda_csv(&String::from_utf8_lossy(&output.stdout))
}

pub fn discover_rocm() -> Vec<DeviceCapability> {
    let output = Command::new("rocm-smi")
        .args([
            "--showuniqueid",
            "--showproductname",
            "--showmeminfo",
            "vram",
            "--json",
        ])
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    parse_rocm_json(&String::from_utf8_lossy(&output.stdout)).unwrap_or_default()
}

pub fn discover_metal() -> Vec<DeviceCapability> {
    if !cfg!(target_os = "macos") {
        return Vec::new();
    }
    let Ok(output) = Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    parse_metal_json(&String::from_utf8_lossy(&output.stdout)).unwrap_or_default()
}

pub fn benchmark_memory(
    device: &DeviceCapability,
    bytes: usize,
    iterations: u32,
) -> BenchmarkReport {
    let bytes = bytes.max(1024);
    let iterations = iterations.max(1);
    let source = vec![0x5a_u8; bytes];
    let mut target = vec![0_u8; bytes];
    for _ in 0..2 {
        target.copy_from_slice(&source);
    }
    let mut samples = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let start = Instant::now();
        target.copy_from_slice(&source);
        std::hint::black_box(&target);
        samples.push(start.elapsed());
    }
    samples.sort();
    let elapsed: Duration = samples.iter().copied().sum();
    let seconds = elapsed.as_secs_f64().max(f64::EPSILON);
    BenchmarkReport {
        device_id: device.stable_id.clone(),
        backend: device.backend,
        warmup_iterations: 2,
        measured_iterations: iterations,
        elapsed_ms: elapsed.as_millis().min(u64::MAX as u128) as u64,
        throughput_units_per_second: bytes as f64 * iterations as f64 / seconds,
        latency_p50_ms: percentile_ms(&samples, 0.50),
        latency_p95_ms: percentile_ms(&samples, 0.95),
        benchmark: "host_memory_copy_v1".into(),
    }
}

fn percentile_ms(samples: &[Duration], percentile: f64) -> f64 {
    let index = ((samples.len() - 1) as f64 * percentile).round() as usize;
    samples[index].as_secs_f64() * 1000.0
}

fn parse_cuda_csv(text: &str) -> Vec<DeviceCapability> {
    text.lines()
        .filter_map(|line| {
            let fields: Vec<_> = line.split(',').map(str::trim).collect();
            if fields.len() < 5 {
                return None;
            }
            let compute = fields[4].to_string();
            let major = compute.split('.').next()?.parse::<u32>().ok()?;
            Some(DeviceCapability {
                stable_id: fields[0].into(),
                backend: BackendKind::Cuda,
                vendor: "nvidia".into(),
                name: fields[1].into(),
                memory_bytes: fields[2].parse::<u64>().ok().map(|mb| mb * 1024 * 1024),
                driver_version: Some(fields[3].into()),
                compute_version: Some(compute),
                supports_fp16: major >= 5,
                supports_bf16: major >= 8,
                supports_int8: major >= 6,
            })
        })
        .collect()
}

fn parse_rocm_json(text: &str) -> Result<Vec<DeviceCapability>> {
    let root: serde_json::Value = serde_json::from_str(text)?;
    let Some(map) = root.as_object() else {
        return Ok(Vec::new());
    };
    Ok(map
        .iter()
        .map(|(card, value)| {
            let name = json_string(value, &["Card series", "Card model", "Card SKU"])
                .unwrap_or_else(|| card.clone());
            let id = json_string(value, &["Unique ID"]).unwrap_or_else(|| stable_id("rocm", &name));
            let memory_bytes = json_u64(
                value,
                &["VRAM Total Memory (B)", "VRAM Total Used Memory (B)"],
            );
            DeviceCapability {
                stable_id: id,
                backend: BackendKind::Rocm,
                vendor: "amd".into(),
                name,
                memory_bytes,
                driver_version: None,
                compute_version: None,
                supports_fp16: true,
                supports_bf16: true,
                supports_int8: true,
            }
        })
        .collect())
}

fn parse_metal_json(text: &str) -> Result<Vec<DeviceCapability>> {
    let root: serde_json::Value = serde_json::from_str(text)?;
    let entries = root
        .get("SPDisplaysDataType")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    Ok(entries
        .into_iter()
        .filter_map(|value| {
            let name = value
                .get("sppci_model")
                .and_then(|v| v.as_str())?
                .to_string();
            Some(DeviceCapability {
                stable_id: stable_id("metal", &name),
                backend: BackendKind::Metal,
                vendor: "apple".into(),
                name,
                memory_bytes: None,
                driver_version: None,
                compute_version: None,
                supports_fp16: true,
                supports_bf16: false,
                supports_int8: true,
            })
        })
        .collect())
}

fn json_string(value: &serde_json::Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(key)?.as_str().map(str::to_string))
}
fn json_u64(value: &serde_json::Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| {
        let value = value.get(key)?;
        value
            .as_u64()
            .or_else(|| value.as_str()?.split_whitespace().next()?.parse().ok())
    })
}
fn stable_id(prefix: &str, name: &str) -> String {
    format!("{prefix}-{:x}", Sha256::digest(name.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(kind: BackendKind) -> DeviceCapability {
        DeviceCapability {
            stable_id: "gpu".into(),
            backend: kind,
            vendor: "test".into(),
            name: "test".into(),
            memory_bytes: Some(1),
            driver_version: None,
            compute_version: None,
            supports_fp16: true,
            supports_bf16: true,
            supports_int8: true,
        }
    }

    fn temporary_executable() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mesh-gpu-backend-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, b"test").unwrap();
        path
    }

    #[test]
    fn parses_cuda_capabilities() {
        let devices = parse_cuda_csv("GPU-1, RTX 4090, 24564, 555.42, 8.9\n");
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].memory_bytes, Some(24564 * 1024 * 1024));
        assert!(devices[0].supports_bf16);
    }

    #[test]
    fn parses_rocm_capabilities() {
        let devices = parse_rocm_json(r#"{"card0":{"Unique ID":"abc","Card series":"Radeon","VRAM Total Memory (B)":"1024"}}"#).unwrap();
        assert_eq!(devices[0].stable_id, "abc");
        assert_eq!(devices[0].memory_bytes, Some(1024));
    }

    #[test]
    fn benchmark_produces_finite_metrics() {
        let device = DeviceCapability {
            stable_id: "cpu".into(),
            backend: BackendKind::Cpu,
            vendor: "test".into(),
            name: "test".into(),
            memory_bytes: None,
            driver_version: None,
            compute_version: None,
            supports_fp16: false,
            supports_bf16: false,
            supports_int8: false,
        };
        let report = benchmark_memory(&device, 4096, 3);
        assert!(report.throughput_units_per_second.is_finite());
        assert!(report.latency_p95_ms >= report.latency_p50_ms);
    }

    #[test]
    fn finds_nested_llama_bench_metric() {
        let value = serde_json::json!([{"model":"tiny","avg_ts":42.5}]);
        assert_eq!(find_metric(&value, &["avg_ts"]), Some(42.5));
    }

    #[test]
    fn runtime_arguments_are_fixed_and_deterministic() {
        let backend = LlamaCppBackend {
            executable: PathBuf::from("llama-cli"),
            device: DeviceCapability {
                stable_id: "gpu".into(),
                backend: BackendKind::Cuda,
                vendor: "nvidia".into(),
                name: "test".into(),
                memory_bytes: Some(1),
                driver_version: None,
                compute_version: None,
                supports_fp16: true,
                supports_bf16: true,
                supports_int8: true,
            },
            gpu_layers: 12,
        };
        let args = backend.arguments(
            Path::new("model.gguf"),
            &InferenceRequest {
                prompt: "hello".into(),
                max_tokens: 8,
                temperature_milli: 250,
                seed: 7,
            },
        );
        assert_eq!(
            args,
            vec![
                "--model",
                "model.gguf",
                "--prompt",
                "hello",
                "--n-predict",
                "8",
                "--temp",
                "0.250",
                "--seed",
                "7",
                "--gpu-layers",
                "12",
                "--no-display-prompt"
            ]
        );
    }

    #[test]
    fn parses_metal_and_ignores_entries_without_a_model() {
        let devices = parse_metal_json(
            r#"{"SPDisplaysDataType":[{"sppci_model":"Apple M4 Max"},{"vendor":"unknown"}]}"#,
        )
        .unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].backend, BackendKind::Metal);
        assert_eq!(devices[0].name, "Apple M4 Max");
        assert!(devices[0].stable_id.starts_with("metal-"));
    }

    #[test]
    fn malformed_discovery_output_is_safe() {
        assert!(parse_cuda_csv("missing,fields").is_empty());
        assert!(parse_rocm_json("not json").is_err());
        assert!(parse_metal_json("not json").is_err());
    }

    #[test]
    fn runtime_requires_existing_executable_and_model() {
        assert!(LlamaCppBackend::new("definitely-missing", device(BackendKind::Cpu), 0).is_err());
    }

    macro_rules! backend_contract_test {
        ($test:ident, $backend:ident, $kind:expr) => {
            #[test]
            fn $test() {
                let executable = temporary_executable();
                let backend = $backend::new(&executable, device($kind), 1).unwrap();
                assert_eq!(backend.kind(), $kind);
                assert_eq!(backend.device().backend, $kind);
                assert!($backend::new(&executable, device(BackendKind::Cpu), 1).is_err());
                std::fs::remove_file(executable).unwrap();
            }
        };
    }

    #[cfg(feature = "cuda")]
    backend_contract_test!(
        cuda_backend_enforces_device_kind,
        CudaBackend,
        BackendKind::Cuda
    );
    #[cfg(feature = "rocm")]
    backend_contract_test!(
        rocm_backend_enforces_device_kind,
        RocmBackend,
        BackendKind::Rocm
    );
    #[cfg(feature = "metal")]
    backend_contract_test!(
        metal_backend_enforces_device_kind,
        MetalBackend,
        BackendKind::Metal
    );
}
