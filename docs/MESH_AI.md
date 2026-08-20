# MESH AI

## Architecture

MESH AI is split into four libraries and two executable integration points:

- `mesh-model`: manifests, registry, content-addressed chunks, materialization.
- `mesh-gpu`: CUDA/ROCm/Metal discovery, benchmarks, fixed-argument llama.cpp runtime adapters.
- `mesh-ai`: placement scoring, batch/pipeline/tensor plans, job wire types, rerouting.
- `mesh-transport`: peer seeding and checksum-bound ordered tensor transport.
- `mesh-coordinator`: network registry, authenticated planning, leases, results, job state.
- `mesh`: import/seed/serve/publish/plan/submit commands and daemon inference workers.

No model can supply an executable or shell command. Runtime paths are local operator configuration, and the adapter passes a fixed argument allow-list.

## Model lifecycle

```powershell
mesh model-import .\model.gguf --model-id example --revision v1 --format gguf --architecture llama
mesh model-list
mesh model-serve --listen 0.0.0.0:8090
mesh model-publish --model-id example --revision v1
```

Another peer can retrieve and verify it:

```powershell
mesh model-seed --peer http://peer.example:8090 --model-id example --revision v1
```

Chunks are written atomically beneath `.mesh/model-cache/chunks/<prefix>/<sha256>`. Every read and network transfer is rehashed.

## Worker daemon

Use a llama.cpp binary built for the local backend:

```powershell
mesh daemon `
  --ai-runtime C:\runtimes\llama-cli.exe `
  --model-seed-listen 0.0.0.0:8090 `
  --model-seed-url http://public-host:8090
```

The node advertises AI readiness only when a runtime was configured and a supported accelerator was detected. The coordinator will not place AI work on CPU-only or unconfigured nodes.

## Scheduling and execution

```powershell
mesh ai-plan --model-id example --revision v1 --backend cuda --layers 32 --batch-size 4
mesh ai-submit --model-id example --revision v1 --backend cuda --layers 32 --prompt "one" --prompt "two"
mesh ai-job <job-id>
```

The coordinator filters incompatible devices, scores candidates, creates a durable plan, and leases batch assignments. Workers seed missing chunks, materialize the verified model, invoke the configured runtime, hash outputs, and submit signed results. A reported device/runtime failure persistently excludes that node and rewrites the assignment for the replacement device.

## Parallelism

- Batch plans divide prompt indexes without gaps or overlap and execute end-to-end.
- Pipeline plans divide model layers without gaps or overlap.
- Model/tensor plans assign contiguous ranks.
- Tensor frames bind job, stream, sequence, shape, dtype, payload, and SHA-256. Receivers reorder within a bounded window and reject corruption, duplicates, replay, and oversized gaps. Senders fail over across routes.

Stock llama.cpp provides whole-model inference, not MESH's partial-layer or collective interface. Pipeline/tensor plans and transport are available to runtime plugins, while the built-in llama.cpp path executes independent batches. This is an explicit runtime boundary, not a claim that WAN tensor parallelism is universally useful.

## Hardware verification

`mesh gpu-info` detects devices and prints a host-transfer baseline. `mesh-gpu::benchmark_llama_cpp` runs `llama-bench` against a real model for token throughput. CI proves adapter compilation on appropriate operating systems; real-device validation requires self-hosted CUDA, ROCm, and Metal runners.
