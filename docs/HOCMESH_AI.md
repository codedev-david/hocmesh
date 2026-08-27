# hocMESH AI

## Architecture

hocMESH AI is split into four libraries and two executable integration points:

- `hocmesh-model`: manifests, registry, content-addressed chunks, materialization.
- `hocmesh-gpu`: CUDA/ROCm/Metal discovery, benchmarks, fixed-argument llama.cpp runtime adapters.
- `hocmesh-ai`: placement scoring, batch/pipeline/tensor plans, job wire types, rerouting.
- `hocmesh-transport`: peer seeding and checksum-bound ordered tensor transport.
- `hocmesh-coordinator`: network registry, authenticated planning, leases, results, job state.
- `hocmesh`: import/seed/serve/publish/plan/submit commands and daemon inference workers.

No model can supply an executable or shell command. Runtime paths are local operator configuration, and the adapter passes a fixed argument allow-list.

## Getting a runtime and weights

```powershell
hocmesh runtime-install                      # pinned llama.cpp, verified by SHA-256
hocmesh runtime-status                       # what is pinned, what is installed
hocmesh model-catalog                        # ids model-pull understands
hocmesh model-pull qwen2.5-0.5b-instruct     # resolve, verify, chunk, register
```

`runtime-install` fetches the llama.cpp release pinned in
`crates/hocmesh-gpu/src/runtime.rs` for the host OS and architecture and checks
it against a digest compiled into the binary. Pinning by digest rather than
resolving by name is the point: a node executes allow-listed work and never a
binary somebody sent it, and "fetch the latest build" would give that away. It
installs a CPU build, which runs inference but does not accelerate it; for CUDA
or ROCm, build llama.cpp for the local backend and pass `--ai-runtime`.

`model-pull` resolves a GGUF on Hugging Face, downloads it with resume, verifies
the digest the Hub published for the file, reads `general.architecture` out of
the GGUF header rather than trusting an operator's `--architecture`, and imports
it through the same chunk store as `model-import`. `--url` requires `--sha256`,
because there is otherwise nothing to check the bytes against.

## Model lifecycle

```powershell
hocmesh model-import .\model.gguf --model-id example --revision v1 --format gguf --architecture llama
hocmesh model-list
hocmesh model-serve --listen 0.0.0.0:8090
hocmesh model-publish --model-id example --revision v1
```

Another peer can retrieve and verify it:

```powershell
hocmesh model-seed --peer http://peer.example:8090 --model-id example --revision v1
```

Chunks are written atomically beneath `.hocmesh/model-cache/chunks/<prefix>/<sha256>`. Every read and network transfer is rehashed.

## Worker daemon

An installed runtime is used without being named, so after `runtime-install`
this is enough:

```powershell
hocmesh daemon `
  --model-seed-listen 0.0.0.0:8090 `
  --model-seed-url http://public-host:8090
```

`--ai-runtime` overrides it with a llama.cpp built for the local backend, which
is what to use for CUDA or ROCm. `--no-ai` declines AI work outright even when a
runtime is installed.

```powershell
hocmesh daemon `
  --ai-runtime C:\runtimes\llama-cli.exe `
  --model-seed-listen 0.0.0.0:8090 `
  --model-seed-url http://public-host:8090
```

The node advertises AI readiness only when a runtime is available *and* the operator has agreed to serve inference, which `limits --ai on|off|auto` records. Installing a runtime is not by itself that agreement — an operator may want `hocmesh infer` for themselves and nothing else — so the two are asked separately. `auto`, the default, means "offer it when a GPU is lent" and is what every node did before the switch existed.

A node that agrees but has no accelerator advertises its shared CPU slice as a `cpu` device and is placed on like any other, bounded by `--memory-percent`; a request that names a backend in `required_backends` still will not land there. The coordinator will not place AI work on a node that advertises no device.

## Scheduling and execution

```powershell
hocmesh ai-plan --model-id example --revision v1 --backend cuda --layers 32 --batch-size 4
hocmesh ai-submit --model-id example --revision v1 --backend cuda --layers 32 --prompt "one" --prompt "two"
hocmesh ai-job <job-id>
```

The coordinator filters incompatible devices, scores candidates, creates a durable plan, and leases batch assignments. Workers seed missing chunks, materialize the verified model, invoke the configured runtime, hash outputs, and submit signed results. A reported device/runtime failure persistently excludes that node and rewrites the assignment for the replacement device.

## Taking delivery and paying for it

A finished batch is not paid for on report. `hocmesh ai-job <job-id>` lists every
answered batch with its digest, its price and its size, and no text at all.

```powershell
hocmesh ai-receipt <job-id> <assignment-id>
hocmesh ai-settle  <job-id> <assignment-id>
hocmesh ai-settle  <job-id> <assignment-id> --dispute --reason "truncated"
```

`ai-receipt` moves that batch's escrow into a holding account and returns the
text in exchange. `ai-settle` then says what it was worth: accepting pays the
provider, disputing sends the same CU to the commons. A dispute is not a refund,
which is what stops a requester from reading an answer and then declining to pay
for it. See *Paying for an answer nobody can recompute* in
[SECURITY.md](SECURITY.md).

## Parallelism

- Batch plans divide prompt indexes without gaps or overlap and execute end-to-end.
- Pipeline plans divide model layers without gaps or overlap.
- Model/tensor plans assign contiguous ranks.
- Tensor frames bind job, stream, sequence, shape, dtype, payload, and SHA-256. Receivers reorder within a bounded window and reject corruption, duplicates, replay, and oversized gaps. Senders fail over across routes.

Stock llama.cpp provides whole-model inference, not hocMESH's partial-layer or collective interface. Pipeline/tensor plans and transport are available to runtime plugins, while the built-in llama.cpp path executes independent batches. This is an explicit runtime boundary, not a claim that WAN tensor parallelism is universally useful.

## Hardware verification

`hocmesh gpu-info` detects devices and prints a host-transfer baseline. `hocmesh-gpu::benchmark_llama_cpp` runs `llama-bench` against a real model for token throughput. CI proves adapter compilation on appropriate operating systems; real-device validation requires self-hosted CUDA, ROCm, and Metal runners.
