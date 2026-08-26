# MeshCompute / MeshAI
## Community Distributed AI Compute Mesh — Full Technical Specification

> **Historical document.** This is the original specification, kept verbatim
> for the record — including the naming exploration that eventually produced
> "MESH". The project is now called **hocMESH**; every other document and all
> code use that name. Nothing here was renamed, so that the trail of how the
> design arrived where it did stays readable.


## 1. Project Summary

The proposed platform is a **community-owned distributed compute network** that combines idle CPUs, GPUs, RAM, and network capacity from computers around the world into a shared virtual data center.

There is **no cryptocurrency, no cash payment, no marketplace, and no ability to purchase compute credits**.

The fundamental rule is:

> **Contribute first. Compute later.**

A participant begins with **zero compute credits**. Their machine must successfully contribute processing resources to other users before they accumulate credits. Those credits can then be spent to run their own workloads across resources contributed by other members of the network.

The objective is eventually to allow something like:

```text
User wants to run a 120B parameter model

Their computer:
RTX 4070 12 GB
32 GB RAM

Available community:

Node A   RTX 4090       24 GB
Node B   RTX 3090       24 GB
Node C   RTX 3060       12 GB
Node D   RX 7900 XTX    24 GB
Node E   CPU            128 GB RAM
Node F   RTX 4080       16 GB
Node G   Apple M4 Max   48 GB unified memory
Node H   RTX 4070 Ti    16 GB

            ↓

Compute Mesh Scheduler

            ↓

Model partition / workload distribution

            ↓

A + B + C + D + E + F + G + H

            ↓

Distributed inference

            ↓

Results returned to requester
```

Conceptually, it is closer to a **community compute cooperative** than a cloud provider.

Existing projects such as Petals, Golem, and Akash demonstrate pieces of distributed or decentralized compute, but MeshCompute's defining difference is its **non-monetary, contribution-first cooperative model**.

---

## 2. Working Project Name

For the specification the platform is called:

**MeshCompute**

The AI-specific execution layer is called:

**MeshAI**

Other possible names include:

- OpenCompute Mesh
- ComputeCoop
- SwarmCompute
- NeuralMesh
- CommonCompute
- OpenGPU Mesh
- Community Compute Grid

The architecture should not depend on the final branding.

---

## 3. Core Philosophy

MeshCompute should operate according to several immutable principles.

### Contribution before consumption

A newly registered node receives:

```text
Compute balance: 0
```

It cannot immediately consume community resources.

It must contribute resources first.

Example:

```text
Alice joins.

Alice contributes:

RTX 4070
4 hours available
Average GPU utilization: 82%

Verified contribution:
3.28 equivalent GPU-hours

Alice balance:

+3.28 GPU Compute Units
```

Later:

```text
Alice launches an AI workload.

Network resources consumed:
2.1 GPU Compute Units

Balance:

3.28 - 2.1 = 1.18
```

This prevents a network composed entirely of consumers.

---

## 4. Credits Are Not Currency

Compute credits should deliberately avoid behaving like cryptocurrency.

They are:

```text
Earned through compute
↓
Associated with user/network identity
↓
Consumed through compute
```

They are not:

```text
Bought
Sold
Traded
Transferred
Cashed out
Speculated on
Tokenized
Converted to cryptocurrency
```

The accounting system measures **participation**, not wealth.

---

## 5. The Credit Ledger

A simplistic system such as:

> 1 GPU hour = 1 credit

would be unfair because different hardware provides dramatically different amounts of useful computation.

Instead, the platform should use normalized **Compute Units (CU)**.

A contribution could be calculated from:

```text
Compute Unit =
Execution Time
× Hardware Performance Factor
× Utilization
× Reliability
× Workload Factor
```

Example:

```text
RTX 4090:
4 hours
× 8.0 performance multiplier
× 0.91 utilization
× 0.99 reliability

≈ 28.8 CU
```

while:

```text
GTX 1660:
4 hours
× 1.4 performance multiplier
× 0.86 utilization
× 0.97 reliability

≈ 4.67 CU
```

The exact coefficients would be determined through standardized benchmarking and refined over time.

---

## 6. Hardware Benchmarking

A node should not simply be trusted when it claims to own a particular device.

The client performs standardized benchmarks.

### GPU Benchmarks

```text
FP32 throughput
FP16 throughput
BF16 throughput
INT8 throughput
Memory bandwidth
VRAM capacity
Tensor/matrix throughput
Kernel latency
Host ↔ GPU bandwidth
```

### CPU Benchmarks

```text
Integer throughput
FP32
FP64
AVX
AVX2
AVX-512 if supported
Memory bandwidth
Core count
Thread count
```

Each node receives a dynamic **Compute Capability Profile**.

Example:

```json
{
  "node": "7F93...",
  "cpu": {
    "cores": 16,
    "threads": 32,
    "score": 18420
  },
  "gpu": {
    "vendor": "nvidia",
    "model": "RTX 4090",
    "vram_mb": 24576,
    "fp16_score": 72115,
    "memory_bandwidth_score": 88420
  },
  "ram_mb": 65536,
  "network": {
    "download_mbps": 945,
    "upload_mbps": 842,
    "latency_ms": 18
  }
}
```

---

## 7. Why Rust

The primary MeshCompute client should be written in **Rust**.

Rust is particularly appropriate because the software needs:

```text
High performance
Low overhead
Memory safety
Native binaries
Concurrency
Networking
Cryptography
Hardware interaction
Daemon/service support
CPU scheduling
Cross-platform support
```

The project should target:

```text
Windows x86-64
Linux x86-64
Linux ARM64
macOS ARM64
macOS x86-64
```

There will not literally be one identical executable for every platform. The Rust project will compile platform-specific native binaries:

```text
meshcompute-windows-x86_64.exe
meshcompute-linux-x86_64
meshcompute-linux-aarch64
meshcompute-macos-aarch64
meshcompute-macos-x86_64
```

To the user, each behaves as the same application.

---

## 8. High-Level Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    USER APPLICATION                     │
│                                                         │
│ CLI / Desktop App / Local API / Web Interface           │
└──────────────────────────┬──────────────────────────────┘
                           │
                    MeshCompute Client
                           │
┌──────────────────────────▼──────────────────────────────┐
│                     NODE AGENT                          │
│                                                         │
│ Resource Manager                                        │
│ Hardware Discovery                                      │
│ Benchmark Engine                                        │
│ Runtime Sandbox                                         │
│ GPU Runtime                                             │
│ Networking                                              │
│ Credit Accounting                                       │
│ Model Cache                                             │
│ Telemetry                                               │
└──────────────────────────┬──────────────────────────────┘
                           │
                  Encrypted Mesh Network
                           │
               ┌───────────┴───────────┐
               │                       │
        Scheduler Network        Peer Discovery
               │                       │
               └───────────┬───────────┘
                           │
                  Community Nodes
                           │
       ┌────────────┬──────┼──────┬────────────┐
       ▼            ▼      ▼      ▼            ▼
      GPU          GPU    CPU    GPU          CPU
     Node 1       Node 2 Node 3 Node 4       Node N
```

---

## 9. Rust Project Structure

A clean Rust workspace could look like:

```text
meshcompute/
│
├── Cargo.toml
│
├── crates/
│   ├── mesh-client/
│   ├── mesh-daemon/
│   ├── mesh-cli/
│   ├── mesh-core/
│   ├── mesh-network/
│   ├── mesh-protocol/
│   ├── mesh-identity/
│   ├── mesh-credit/
│   ├── mesh-scheduler/
│   ├── mesh-runtime/
│   ├── mesh-cpu/
│   ├── mesh-gpu/
│   ├── mesh-model/
│   ├── mesh-storage/
│   ├── mesh-security/
│   ├── mesh-benchmark/
│   └── mesh-telemetry/
│
├── runtimes/
│   ├── cuda/
│   ├── rocm/
│   ├── vulkan/
│   ├── metal/
│   └── cpu/
│
├── protocol/
│   └── protobuf/
│
└── tests/
```

---

## 10. The Local Client

Installation could look like:

```bash
meshcompute install
```

The client launches an initial setup wizard:

```text
Welcome to MeshCompute

Available resources detected:

CPU:
AMD Ryzen 9 7950X
16 cores / 32 threads

GPU:
NVIDIA RTX 4090
24 GB VRAM

RAM:
64 GB

Network:
912 Mbps down
748 Mbps up

What may the network use?

GPU:   [80%]
CPU:   [50%]
RAM:   [24 GB]
Disk:  [100 GB]

Idle-only contribution: YES

Contribute while computer is actively used: NO
```

The user remains completely in control.

---

## 11. Resource Limits

A provider can define strict limits.

Example:

```toml
[resources]
cpu_percent = 50
gpu_percent = 80
memory_gb = 24
disk_cache_gb = 100

[availability]
idle_only = true
minimum_idle_minutes = 10
pause_on_user_activity = true

[network]
max_upload_mbps = 100
max_download_mbps = 200
```

A user can therefore use their PC normally during the day and allow MeshCompute to operate overnight.

---

## 12. Node Identity

Every installation generates a cryptographic identity locally.

```text
Private Key
      │
      ▼
Public Key
      │
      ▼
Node ID
```

Example:

```text
node:
mc:3df81b90a95e...
```

The private key never leaves the machine.

Messages are digitally signed.

This enables nodes to verify:

```text
Who submitted a task?
Who executed it?
Who verified it?
Who earned credits?
```

without relying on shared passwords between machines.

---

## 13. Peer Discovery

A node must locate other nodes.

Initially, discovery should not be completely decentralized.

Use bootstrap coordinators:

```text
Node
 ↓
Bootstrap Service
 ↓
Known peers
 ↓
Peer-to-peer connections
```

Later:

```text
Distributed Hash Table
        +
Bootstrap nodes
        +
Peer gossip
```

This gives the project a practical MVP without permanently requiring one centralized service.

---

## 14. Network Protocol

A likely stack is:

```text
QUIC
TLS 1.3
Protocol Buffers
libp2p
Tokio
```

QUIC is attractive because MeshCompute needs:

```text
Encrypted communication
Multiplexed streams
Connection migration
High throughput
Low latency
Internet-friendly transport
```

Control communication should be separated from model data.

### Control Plane

```text
Node registration
Health checks
Scheduling
Credits
Capability advertisements
```

### Data Plane

```text
Tensor transfers
Model blocks
Inference activations
Input datasets
Task results
```

---

## 15. Community Virtual Data Center

The scheduler effectively sees something like:

```text
GLOBAL COMPUTE INVENTORY

12,481 nodes online

NVIDIA GPUs         6,201
AMD GPUs            1,452
Apple Silicon       1,732
CPU-only nodes      3,096

Available VRAM      96.4 TB
Available RAM       448 TB
Available CPU cores 114,821
```

A requester sees this as one pool:

```text
Available Mesh Compute
```

---

## 16. Scheduling Engine

For every job, the scheduler evaluates:

```text
Hardware capability
VRAM
RAM
GPU architecture
Network latency
Upload speed
Download speed
Geographic locality
Current load
Reliability
Expected uptime
Model cache state
Credit cost
```

A conceptual score:

```text
NodeScore =
ComputePerformance
× NetworkQuality
× Reliability
× DataLocality
× Availability
```

---

## 17. Latency-Aware Clustering

Randomly placing a model across computers around the world would often perform terribly.

The scheduler therefore forms temporary **compute neighborhoods**.

Example:

```text
Job originated in Georgia

Potential GPU nodes:

Atlanta       11 ms
Charlotte     24 ms
Orlando       29 ms
Virginia      34 ms
Germany      118 ms
Japan        183 ms
Australia    241 ms
```

The scheduler strongly prefers nearby nodes for tightly coupled workloads.

Farther-away machines can still perform independent workloads.

---

## 18. Task Parallelism

This is the easiest distributed-compute model.

```text
1,000 images

Node A → images 1-100
Node B → images 101-200
Node C → images 201-300
...
```

Very little inter-node communication is needed.

This should be implemented first.

---

## 19. Batch AI Inference

Example:

```text
10,000 prompts
```

Instead of splitting one inference operation:

```text
Node A → prompts 1-500
Node B → prompts 501-1000
Node C → prompts 1001-1500
...
```

This is extremely well suited to Internet-distributed compute.

---

## 20. Model Parallelism

Suppose:

```text
Model = 120 GB

Machine A: 24 GB VRAM
Machine B: 24 GB
Machine C: 16 GB
Machine D: 24 GB
Machine E: 32 GB
```

The scheduler can divide model layers:

```text
Layers 0-14
    ↓
GPU A

Layers 15-29
    ↓
GPU B

Layers 30-39
    ↓
GPU C

Layers 40-54
    ↓
GPU D

Layers 55-79
    ↓
GPU E
```

Inference works approximately:

```text
Prompt
  ↓
A
  ↓ activations
B
  ↓
C
  ↓
D
  ↓
E
  ↓
Token
```

---

## 21. Tensor Parallelism

Later, the system could support tensor parallelism.

Instead of:

```text
Node A = layers 1-10
Node B = layers 11-20
```

the calculation for an individual layer can be distributed:

```text
              Matrix
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
     GPU A     GPU B     GPU C
       │         │         │
       └─────────┼─────────┘
                 ▼
               Result
```

This provides higher parallelism but requires significantly more communication.

Across consumer Internet links it will only be appropriate under favorable network conditions.

---

## 22. Pipeline Parallelism

Pipeline execution is much more promising across the Internet.

```text
GPU A    Stage 1
     ↓
GPU B    Stage 2
     ↓
GPU C    Stage 3
     ↓
GPU D    Stage 4
```

Multiple requests can move through simultaneously:

```text
time →

A:  T1 T2 T3 T4
B:     T1 T2 T3 T4
C:        T1 T2 T3 T4
D:           T1 T2 T3 T4
```

This keeps nodes occupied instead of waiting for one request.

---

## 23. Hybrid Parallelism

The ultimate architecture should combine:

```text
Task parallelism
+
Pipeline parallelism
+
Tensor parallelism
+
Model parallelism
+
Batch parallelism
```

The scheduler decides dynamically which approach best fits each workload.

---

## 24. Heterogeneous Hardware

The system must assume hardware is never uniform.

Example:

```text
RTX 4090
RTX 3060
RX 7900 XTX
Apple M4
Intel Arc
Threadripper CPU
Xeon server
```

The scheduler must understand the relative capabilities and runtime constraints of each node.

---

## 25. GPU Abstraction

The low-level GPU layer needs multiple runtime paths.

```text
NVIDIA
   ↓
CUDA

AMD
   ↓
ROCm

Apple
   ↓
Metal

Intel
   ↓
oneAPI / Vulkan

Fallback
   ↓
Vulkan / CPU
```

The Rust orchestration layer does not need to perform every matrix operation itself.

Instead:

```text
Rust
 ↓
Execution Backend
 ↓
CUDA / ROCm / Metal / Vulkan
```

---

## 26. Model Format

An AI-focused MVP should standardize model representation.

Likely supported formats:

```text
GGUF
safetensors
ONNX
```

The scheduler needs metadata describing:

```text
layers
parameter count
quantization
tensor sizes
VRAM requirement
supported backends
```

---

## 27. Model Block Distribution

Nodes should maintain content-addressed model caches.

Example:

```text
Node A already has:
Layers 0-12

Node B already has:
Layers 13-25

Node C already has:
Layers 26-39
```

When scheduling:

```text
Scheduler sees cached blocks
             ↓
reuses suitable nodes
             ↓
reduces network traffic
```

---

## 28. Content-Addressed Model Storage

Model pieces should be identified by cryptographic hashes.

```text
SHA-256:
47a28f915...

block:
model-x/layer-0042

hash:
47a28f915...
```

Nodes verify downloaded data before execution.

This enables torrent-like peer distribution:

```text
Model Block
    │
 ┌──┼──┬──┐
 ▼  ▼  ▼  ▼
A   B  C   D

Later:

A → E
B → F
C → G
```

---

## 29. Compute and Data Swarms

There are really two meshes.

```text
              MeshCompute

        ┌─────────┴─────────┐
        │                   │
   Data Distribution   Computation
        │                   │
   model blocks         GPU / CPU
   checkpoints          inference
   datasets             training
```

Torrent-like distribution handles large immutable artifacts.

The compute scheduler handles processing.

---

## 30. Security Model

A provider must never give another participant remote shell access.

Absolutely no:

```text
SSH access
Remote desktop
Arbitrary host commands
Host filesystem access
Administrator privileges
```

Jobs execute through a restricted runtime.

```text
Host Computer
│
├── Mesh Agent
│
└── Sandbox
      │
      ├── Job
      ├── restricted RAM
      ├── restricted GPU
      ├── restricted CPU
      └── restricted filesystem
```

---

## 31. Execution Sandbox

Depending on operating system, isolation could use:

### Linux

```text
namespaces
cgroups
seccomp
```

### Windows

```text
Job Objects
AppContainer / sandbox mechanisms
```

### macOS

```text
sandbox facilities
```

A later architecture could use:

```text
WebAssembly / WASI
```

for generic workloads.

AI GPU execution is more complicated because the sandbox requires controlled GPU access.

---

## 32. Jobs Should Initially Be Declarative

Version one should not allow arbitrary executable uploads.

Instead a task describes a supported workload:

```text
Runtime:
llm-inference-v1

Model:
hash xyz

Input:
encrypted task data

Resources:
8 GB VRAM

Maximum execution:
300 seconds
```

The trusted MeshCompute runtime interprets it.

This dramatically reduces attack surface.

---

## 33. Provider Privacy

The requester should not see personally identifying information about providers.

Instead of:

```text
David's computer
Exact street address
RTX 4090
```

they see something like:

```text
Node d874a3

Region:
US Southeast

Latency:
21 ms

Capability:
GPU Tier 8
```

Exact IP exposure should be minimized where possible.

---

## 34. Requester Privacy

Requester privacy is technically harder.

A machine performing inference may inherently see some representation of the data it processes.

The architecture should therefore support privacy levels:

```text
Public jobs
Community-safe jobs
Encrypted transport
Trusted-node jobs
TEE-backed jobs
```

Future work can investigate trusted execution environments and privacy-preserving computation.

The project should never claim ordinary GPU jobs are automatically invisible to the machine performing them.

---

## 35. Work Verification

Verification primarily ensures:

```text
The task actually ran.
The result is not corrupted.
The hardware is functioning properly.
Credits are legitimate.
```

Multiple verification mechanisms should be used.

---

## 36. Signed Work Receipts

A completed task produces:

```text
Job ID
Requester
Provider node
Task hash
Input commitment
Result hash
Start timestamp
End timestamp
Resource usage
Execution signature
```

The provider signs it.

The scheduler and/or requester acknowledges it.

The credit ledger is then updated.

---

## 37. Probabilistic Verification

Running every job twice would waste enormous amounts of compute.

Instead:

```text
Most tasks:
single execution

Small percentage:
verification execution
```

The verification rate can depend on reputation.

New node:

```text
Verification rate: high
```

Reliable established node:

```text
Verification rate: low
```

---

## 38. Node Reliability Score

Every provider develops a reliability history.

```text
Node Reliability

Jobs accepted:      9,281
Jobs completed:     9,244
Timeouts:              21
Incorrect results:      2
Disconnects:           14

Reliability:
99.6%
```

Schedulers prefer reliable nodes for long or critical jobs.

---

## 39. Contribution Accounting

The ledger could maintain:

```text
Account:
mc:user:a814...

Earned:
6,284 CU

Consumed:
4,881 CU

Available:
1,403 CU
```

The fundamental equation is:

```text
AVAILABLE =
VERIFIED CONTRIBUTIONS
-
VERIFIED CONSUMPTION
```

Never:

```text
credit card → CU
```

---

## 40. No Free Initial Compute

This should be a protocol-level rule.

```rust
if requested_compute > account.available_compute {
    reject_job();
}
```

There should be no:

```text
Free trial credits
Starter credits
Credit purchases
Borrow now / contribute later
```

A user earns their first unit of compute by helping somebody else.

That is the defining social contract of the network.

---

## 41. Banking Credits

Credits may be stored.

Example:

```text
January
+800 CU

February
+650 CU

March
No contribution

Balance:
1,450 CU
```

In April the participant could spend those credits on a large workload.

Initial versions should allow banking without expiration.

---

## 42. Preventing Sybil Abuse

Someone could create thousands of fake identities.

That should provide no economic advantage because every identity starts with zero credits.

Fake accounts cannot consume compute unless they first contribute verified work.

---

## 43. Scheduler Credit Reservation

Suppose a workload is expected to consume:

```text
180 CU
```

The scheduler reserves that amount:

```text
Balance:
500 CU

Reserved:
180 CU

Spendable:
320 CU
```

After completion:

```text
Actual cost:
157 CU

Released:
23 CU
```

---

## 44. Failure Handling

Nodes will disappear frequently because community computers are inherently ephemeral.

Participants may:

```text
close a laptop
lose Wi-Fi
reboot
lose power
start gaming
pause MeshCompute
```

The scheduler must treat node failure as normal operation rather than an exceptional event.

---

## 45. Model Replication

Important model blocks should have multiple providers.

Instead of:

```text
Layer 20-30
only Node B
```

maintain:

```text
Layer 20-30

Node B
Node G
Node J
```

If B disappears:

```text
routing shifts → G
```

---

## 46. Heartbeats

Nodes periodically advertise:

```text
alive status
available resources
job progress
temperature
load
network quality
```

If enough heartbeats are missed:

```text
NODE → SUSPECT
      ↓
NODE → OFFLINE
```

Tasks are rescheduled.

---

## 47. Temperature Protection

The client must protect participant hardware.

Example:

```text
GPU maximum configured temperature:
82°C

Current:
79°C

Scheduler:
reduce workload

Current:
83°C

Client:
pause contribution
```

Provider safety always wins over network throughput.

---

## 48. User Activity Detection

If configured:

```text
User starts a game.

GPU utilization increases.

MeshCompute:
drain current task
stop accepting tasks
release GPU
```

After:

```text
10 minutes idle

GPU returns to pool
```

This makes participation practical for ordinary home users.

---

## 49. Minimum Resource Reservations

Users can configure local reservations.

```text
Reserve:

4 CPU cores
8 GB RAM
20% GPU
```

MeshCompute may only use the remainder.

---

## 50. CPU Contribution

CPU nodes should participate fully.

Potential CPU workloads include:

```text
tokenization
data preprocessing
embeddings
compression
model conversion
CPU inference
verification
dataset transformations
simulation
rendering
scientific workloads
```

This prevents the network from becoming GPU-only.

---

## 51. Resource Classes

The ledger could eventually distinguish:

```text
CPU Compute Units
GPU Compute Units
Memory Units
Storage Units
Network Transfer Units
```

However, the first implementation should use one normalized:

```text
Compute Unit (CU)
```

and refine the accounting system later.

---

## 52. Node Capability Advertisement

Nodes periodically advertise signed capability records.

```text
Node 8f20

CPU:
16 threads available

RAM:
22 GB available

GPU:
RTX 4080

VRAM:
13.8 GB available

Latency class:
A

Reliability:
99.8%

Cached model blocks:
42
```

The scheduler builds a live resource graph.

---

## 53. Scheduling as a Graph Problem

Distributed inference can be modeled as a weighted graph.

Nodes represent:

```text
Compute resources
```

Edges represent:

```text
Bandwidth
Latency
Packet loss
```

Example:

```text
A ──8ms── B
│         │
21ms     12ms
│         │
C ──15ms─ D
```

The scheduler seeks clusters with:

```text
enough VRAM
+
enough compute
+
low inter-node latency
+
high reliability
```

rather than simply choosing the fastest individual GPUs.

---

## 54. Temporary Virtual Clusters

A job creates something analogous to an ephemeral cloud cluster.

```text
Virtual Cluster #839182

Node A     RTX 4090
Node B     RTX 3090
Node C     RTX 4070
Node D     RX 7900 XTX

Model:
LargeModel-70B

Pipeline:
A → B → C → D
```

When the job completes:

```text
Cluster destroyed
Nodes returned to mesh
Credits settled
```

---

## 55. Local API

Applications should communicate with MeshCompute through a local API.

Example:

```http
POST /v1/inference
```

Example request:

```json
{
  "model": "community/model-70b",
  "prompt": "Explain quantum entanglement",
  "max_tokens": 500
}
```

From the developer's perspective it behaves like a conventional AI API while execution occurs on the community mesh.

---

## 56. OpenAI-Compatible API

A future compatibility endpoint would be highly useful.

Example:

```text
http://localhost:11455/v1/chat/completions
```

Existing applications could point to MeshCompute without major integration changes.

---

## 57. CLI

Examples:

```bash
mesh status
```

```text
Node: ONLINE

GPU contribution:
Enabled

Current contribution:
72%

Credits:
428.4 CU

Lifetime contributed:
1,882 CU
```

Run a model:

```bash
mesh run llama-large
```

Inspect hardware:

```bash
mesh hardware
```

Enable contribution:

```bash
mesh contribute --gpu 80 --cpu 40
```

Pause:

```bash
mesh pause
```

---

## 58. Desktop Application

The desktop interface should remain simple.

```text
        MESHCOMPUTE

Network: ● Connected

Your Contribution

GPU     ███████░░░  72%
CPU     ███░░░░░░░  31%

Earned today:
+28.4 CU

Available balance:
642.7 CU

[ Run AI Model ]

[ Contribution Settings ]
```

The technical complexity remains underneath.

---

## 59. Control Plane

The first version should use a lightweight centralized control plane.

It handles:

```text
Node discovery
Authentication
Scheduling
Credit ledger
Reputation
Model registry
Task coordination
```

This does not conflict with distributed compute.

Trying to make every subsystem peer-to-peer in version one would dramatically increase complexity.

---

## 60. Long-Term Decentralization

Over time:

```text
Central scheduler
      ↓
Federated schedulers
      ↓
Regional coordinators
      ↓
Distributed coordination
```

Compute can be decentralized from day one while scheduling and governance decentralize gradually.

---

## 61. Regional Schedulers

Eventually:

```text
North America Scheduler
Europe Scheduler
Asia Scheduler
Australia Scheduler
```

Jobs remain region-local when appropriate.

This significantly reduces latency.

---

## 62. Global Scheduler

```text
             Global Coordinator
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
      America     Europe       Asia
        │           │           │
      Nodes       Nodes       Nodes
```

Independent jobs can span continents.

Tightly coupled inference should prefer locality.

---

## 63. Data Locality

Suppose a 300 GB model is already cached across ten machines in Virginia.

The scheduler should not select marginally faster GPUs elsewhere if doing so requires transferring the entire model again.

Scheduling therefore incorporates:

```text
compute cost
+
network cost
+
model transfer cost
```

---

## 64. Model Registry

MeshCompute should maintain a model registry.

Example:

```text
model:
open-model-70b

manifest:
version
architecture
parameter count
block hashes
runtime compatibility
license
quantization
```

The registry contains metadata.

The model chunks themselves can propagate peer-to-peer.

---

## 65. Torrent-Like Model Distribution

Suppose a model is:

```text
140 GB
```

split into:

```text
1,400 × 100 MB chunks
```

A node can retrieve chunks simultaneously:

```text
chunk 1   ← Node A
chunk 2   ← Node B
chunk 3   ← Node C
chunk 4   ← Node D
```

After downloading them, that node can become another source.

Popular models therefore become easier to distribute as more participants cache them.

---

## 66. Community Seeding

Nodes could opt into:

```text
Compute contribution
Model seeding
Both
```

A machine with large storage but limited compute can still help the network by caching model data.

Whether storage contribution earns CU should be decided later.

---

## 67. Phase One — MVP

Do not start with distributed 120B model inference.

The MVP should prove the cooperative compute model:

```text
Rust client
       ↓
Node registration
       ↓
Hardware discovery
       ↓
CPU task queue
       ↓
Sandbox execution
       ↓
Verified work receipt
       ↓
Credit earned
       ↓
Credit consumed
```

---

## 68. MVP Demonstration

Use three computers:

```text
Computer A
Computer B
Computer C
```

A and B contribute compute.

C starts with:

```text
0 CU
```

C cannot submit work.

C contributes a task.

Now:

```text
C = 20 CU
```

C submits a workload.

A and B execute portions simultaneously.

C spends:

```text
8 CU
```

A and B earn credits.

This proves the underlying economy.

---

## 69. Phase Two — GPU Worker

Add:

```text
NVIDIA CUDA worker
```

Run independent GPU tasks such as:

```text
image inference
embedding generation
batch model inference
```

This proves GPU scheduling.

---

## 70. Phase Three — Peer-to-Peer Data

Implement:

```text
content-addressed storage
model chunking
peer discovery
parallel downloads
hash verification
```

The network now begins to behave like a torrent-style data mesh.

---

## 71. Phase Four — Distributed Model Inference

Implement pipeline partitioning.

Start with two nodes:

```text
GPU A
layers 0-15

GPU B
layers 16-31
```

Then:

```text
A → B
```

Expand to three nodes, then five, and finally dynamically scheduled groups.

---

## 72. Phase Five — Fault Tolerance

Add:

```text
replicated layers
automatic failover
rerouting
checkpointing
node reputation
latency prediction
```

Machines can now join and leave without destroying every request.

---

## 73. Phase Six — Heterogeneous GPUs

Support:

```text
CUDA
ROCm
Metal
```

Then optimize scheduling by backend.

---

## 74. Phase Seven — Intelligent Scheduler

The scheduler continuously measures:

```text
throughput
latency
VRAM
queue depth
failure rate
network path
model availability
```

and dynamically changes topology.

---

## 75. Phase Eight — Community Scale

Potential progression:

```text
10 nodes
↓
100
↓
1,000
↓
10,000
↓
100,000+
```

At scale the network becomes a substantial distributed pool of otherwise idle hardware.

---

## 76. Initial Rust Technology Stack

Recommended starting stack:

```text
Language
Rust

Async runtime
Tokio

P2P networking
libp2p

Transport
QUIC

Serialization
Protocol Buffers

RPC
gRPC where appropriate

Local database
SQLite initially

Cryptography
RustCrypto / audited primitives

CLI
clap

Configuration
serde + TOML

Logging
tracing
```

GPU execution should initially use backend-specific bindings rather than creating a GPU framework from scratch.

---

## 77. Basic Node Process Architecture

```text
mesh-daemon
│
├── IdentityManager
├── PeerManager
├── ResourceMonitor
├── ContributionManager
├── SchedulerClient
├── JobManager
├── SandboxManager
├── GPUManager
├── ModelCache
├── CreditManager
└── TelemetryManager
```

Each component communicates asynchronously.

---

## 78. Internal Rust Events

The program should be heavily event-driven.

Conceptually:

```rust
enum MeshEvent {
    PeerConnected,
    PeerDisconnected,
    JobReceived,
    JobStarted,
    JobCompleted,
    JobFailed,
    ResourceChanged,
    CreditUpdated,
    ModelDownloaded,
    NodeOverheated,
}
```

Tokio channels can route these events internally.

---

## 79. Job State Machine

Every distributed task should follow a formal state machine.

```text
Created
   ↓
CreditReserved
   ↓
Scheduled
   ↓
Dispatched
   ↓
Accepted
   ↓
Running
   ↓
Verifying
   ↓
Completed
   ↓
Settled
```

Failure path:

```text
Running
  ↓
NodeLost
  ↓
Rescheduled
```

---

## 80. Protocol Messages

Initial protocol messages might include:

```text
NodeHello
NodeCapabilities
Heartbeat
JobOffer
JobAccept
JobReject
JobStart
JobProgress
JobResult
JobReceipt
CreditSettlement
ModelChunkRequest
ModelChunkResponse
```

Protocol Buffers can provide versioned binary serialization.

---

## 81. Scheduler Database

Core entities:

```text
Users
Nodes
Capabilities
Jobs
JobAssignments
ModelBlocks
NodeCaches
CreditLedger
WorkReceipts
Reputation
Heartbeats
```

---

## 82. Ledger Should Be Append-Only

Do not store only:

```text
User balance = 500
```

Store ledger entries:

```text
+22 contribution
+18 contribution
-9 inference
+34 contribution
-12 inference
```

Balance is calculated from the ledger.

This provides auditability.

---

## 83. No Blockchain Required

Blockchain is not necessary for the initial design.

An append-only signed ledger can provide:

```text
auditability
consistency
traceability
```

without introducing:

```text
tokens
mining
wallet economics
transaction fees
speculation
```

If decentralized ledger replication is eventually required, it can be solved separately.

---

## 84. Governance

Possible long-term principles:

```text
open-source protocol
public technical specifications
transparent credit algorithm
community proposals
versioned protocol changes
no paid priority tier
```

Someone with money should not be able to buy priority over somebody who contributed compute.

---

## 85. Fair Scheduling

Scheduling can prioritize:

```text
credits available
job age
resource suitability
fairness
network efficiency
```

rather than:

```text
highest bidder wins
```

---

## 86. Potential Killer Use Case

The most compelling demonstration is:

> **Run an AI model that none of the participating machines can run individually.**

Example:

```text
10 people

each has:
8-24 GB GPU memory

Nobody can individually run:
150 GB model

Together:
combined accessible capacity >150 GB

MeshCompute partitions model.

One user invokes it.

Model runs across community machines.
```

That demonstrates the entire idea visually and technically.

---

## 87. What MeshCompute Is Not

MeshCompute is not:

```text
AWS with strangers' PCs
A crypto mining platform
A GPU rental marketplace
A cryptocurrency
A remote desktop network
A botnet
A cloud reseller
```

It is:

> **A cooperative distributed computing system in which access to shared computational resources is earned by contributing computational resources to the community.**

---

## 88. Long-Term Vision

Imagine:

```text
183,000 people connected

47,000 GPUs idle

62,000 CPUs available

1.8 PB distributed model cache

Hundreds of TB RAM

Thousands of models
```

Instead of those resources sitting unused:

```text
                       COMMUNITY

Laptop ─────┐
Gaming PC ──┤
Mac Studio ─┤
Server ─────┤
Workstation ┼──► THE hocMESH NETWORK
Home Lab ───┤
Linux Box ──┤
GPU Rig ────┘
```

When one participant needs enormous compute:

```text
Community idle resources
            ↓
temporarily combine
            ↓
execute workload
            ↓
return result
            ↓
dissolve cluster
```

That is the central vision.

---

## 89. Major Architectural Separation

The project should be separated into two major layers.

### MeshCompute Core

A general community distributed-compute protocol:

```text
resource contribution
credit accounting
scheduling
P2P networking
security
verification
fault tolerance
```

### MeshAI

An AI execution layer built on top:

```text
LLM inference
model partitioning
GPU execution
model caching
tensor transport
pipeline scheduling
```

Architecture:

```text
              MeshAI

        AI / LLM execution
               │
               ▼
        MeshCompute Core

       scheduling / credits
       networking / security
       resource federation
               │
               ▼
     Community Hardware Mesh
```

---

## 90. Recommended Repository Architecture

```text
meshcompute/
├── core/
├── networking/
├── protocol/
├── scheduler/
├── ledger/
├── runtime/
├── client/
├── daemon/
└── cli/

meshai/
├── model-runtime/
├── model-registry/
├── partitioner/
├── tensor-transport/
├── inference/
└── gpu-backends/
```

Both should be primarily Rust.

---

## 91. First Development Milestone

The first prototype should not run an LLM.

Build:

```text
Machine A
Machine B
Machine C
```

Each runs:

```bash
meshcompute daemon
```

Then prove:

```text
A contributes CPU.
        ↓
B submits matrix workload.
        ↓
A executes it.
        ↓
Result verified.
        ↓
A receives CU.
        ↓
A submits workload.
        ↓
B/C execute it.
        ↓
A spends CU.
```

Once that works reliably, the fundamental network exists.

Everything else becomes an execution engine built on top.

---

## 92. Second Major Prototype

The next demonstration should be:

```text
          LARGE MODEL

              │
        Model Partitioner
              │
     ┌────────┼────────┐
     ▼        ▼        ▼
   PC A     PC B      PC C
  12 GB     24 GB     16 GB
     │        │        │
     └────────┼────────┘
              │
           Inference
              │
           Response
```

Expand progressively:

```text
3 computers on LAN
        ↓
3 computers across Internet
        ↓
10 computers
        ↓
100 computers
```

---

## 93. Core Design Principle

A concise statement of the project vision is:

> **Your computer becomes part of the world's community data center when you're not using it, and the community's computers become yours when you need them.**

The defining architecture combines:

```text
Contribution-first cooperative economics
+
Heterogeneous CPU/GPU hardware
+
Distributed scheduling
+
Torrent-like model distribution
+
Secure Rust node software
+
Dynamic virtual clusters
+
Fault-tolerant distributed inference
+
Credits that cannot simply be purchased
```

This should be treated as a real software platform composed of **MeshCompute Core + MeshAI**, not merely as an AI application.

---

# 94. Development Priorities

Recommended implementation order:

1. Rust workspace and protocol definitions.
2. Node identity and signed messaging.
3. Hardware discovery.
4. Hardware benchmarking.
5. Resource configuration and limits.
6. Scheduler registration.
7. CPU task execution.
8. Sandboxing.
9. Work receipts.
10. Append-only CU ledger.
11. Contribution-first enforcement.
12. Multi-node task parallelism.
13. Node reputation.
14. Fault recovery.
15. GPU discovery.
16. CUDA worker.
17. GPU benchmarking.
18. GPU task scheduling.
19. Content-addressed storage.
20. P2P model distribution.
21. Model registry.
22. Model layer partitioning.
23. Pipeline inference.
24. Dynamic cluster construction.
25. Heterogeneous GPU support.
26. Distributed fault-tolerant inference.
27. Public local API.
28. OpenAI-compatible API.
29. Desktop client.
30. Regional/federated schedulers.

---

# 95. Initial Success Criteria

The project should consider the first major architecture successful when it can demonstrate all of the following:

```text
✓ A new user begins with zero CU.

✓ The user cannot submit a workload with zero CU.

✓ The user's machine contributes a real task.

✓ The contribution is benchmarked and verified.

✓ A signed work receipt is generated.

✓ CU is credited to the user.

✓ The user submits a workload.

✓ The scheduler finds multiple remote nodes.

✓ The job is divided among those nodes.

✓ The nodes execute in parallel.

✓ The result is reconstructed and returned.

✓ CU is deducted from the requester.

✓ CU is distributed to contributing providers.

✓ One provider can disconnect during a job.

✓ The scheduler can recover or reschedule the failed work.

✓ No provider receives arbitrary remote access to another participant's system.
```

The next major success criterion is:

> A model too large for any single participating computer successfully runs across multiple community GPUs.

---

# 96. Final Product Goal

MeshCompute should ultimately make a geographically distributed community of ordinary machines behave, as much as practical, like a single shared data center.

```text
Thousands of independent computers
              │
              ▼
      MeshCompute Protocol
              │
              ▼
     Global Resource Graph
              │
              ▼
      Dynamic Scheduling
              │
              ▼
     Temporary Clusters
              │
              ▼
          MeshAI
              │
              ▼
       Large AI Models
```

The network's fundamental exchange is simple:

```text
CONTRIBUTE
    ↓
EARN COMPUTE
    ↓
BANK COMPUTE
    ↓
USE COMMUNITY COMPUTE
    ↓
CONTRIBUTE AGAIN
```

There is no financial marketplace required.

The resource being exchanged is **computation itself**.
