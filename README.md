# MESH Compute

**MESH = Mutual Exchange of Shared Hardware**

MESH is an open, contribution-first distributed compute network. Participants contribute idle compute, earn non-monetary Compute Units (CU), bank those units, and later spend them on work executed by other participants.

There is **no payment system, token, cryptocurrency, market, or purchasable credit** in this design.

> Contribute first. Compute later.

This repository is a Rust implementation of the MESH Compute Core plus the security foundation for the future MESH AI layer.

## What is implemented in v0.2

This repository contains working source for three native Rust programs:

- `mesh` — participant/client/worker CLI
- `mesh-coordinator` — workload scheduler and node control plane
- `mesh-validator` — replicated CU ledger validator

Implemented architecture:

- Ed25519 node identities generated locally.
- Replay-resistant signed API requests using timestamp + cryptographic nonce.
- Hardware discovery and CPU benchmarking.
- GPU detection for NVIDIA/CUDA and Apple/Metal-capable systems where detectable by the current hardware adapter.
- Declarative allow-listed work instead of arbitrary remote binaries.
- Deterministic prime-range workload as the first safe distributed workload.
- Multi-worker task parallelism.
- Work leasing and lease expiration/requeue.
- Requesters cannot execute their own paid shards; both scheduler and validators enforce this.
- Contribution-first balance enforcement.
- Local SQLite accounting mode for development.
- **Quorum replicated ledger mode** for the public-network architecture.
- CU-conserving double-entry-style postings.
- Per-job escrow accounts.
- Community bootstrap issuance with a validator-enforced hard cap.
- User job reservation signed by the requester.
- Provider result proofs signed over the exact job, shard, work, reward, funding type, and result.
- Validators independently recompute deterministic work before signing rewards.
- Hash-linked ledger entries.
- Ed25519 quorum certificates.
- Persistent one-vote-per-height validator locks.
- Duplicate reservation/reward claim prevention.
- Full validator ledger replicas in SQLite.
- Validator catch-up/synchronization.
- Ordinary client full-ledger mirroring.
- Offline audit from genesis.
- Direct validator balance/head verification independent of the coordinator.
- Crash-safe coordinator ledger intents with startup/manual recovery.
- Signed validator claim proofs for reservation/reward reconciliation.
- Per-process ledger proposal serialization to reduce same-height races.

## Important scope boundary

MESH Compute Core v0.2 is designed to be a real executable distributed-compute MVP, but **distributed LLM layer/tensor execution is not claimed to be complete**.

The current executable workload is deterministic CPU prime-range computation. The architecture deliberately proves the harder control-plane primitives first:

1. identity,
2. trust,
3. contribution accounting,
4. replicated state,
5. work scheduling,
6. sharding,
7. verification,
8. fault recovery.

The future `MESH AI` layer should add CUDA/ROCm/Metal execution, model manifests, content-addressed model chunks, model/pipeline partitioning, activation transport, GPU-aware scheduling, and distributed inference.

See `docs/FULL_ORIGINAL_SPEC.md` and `docs/ROADMAP.md`.

---

# Repository layout

```text
MESH/
├── Cargo.toml
├── rust-toolchain.toml
├── README.md
├── CODEX_HANDOFF.md
├── LICENSE
│
├── config/
│   └── validators.example.json
│
├── crates/
│   ├── mesh-protocol/
│   │   └── shared wire types, signed request format, hashes, IDs
│   │
│   ├── mesh-core/
│   │   ├── identity.rs
│   │   ├── hardware.rs
│   │   └── compute.rs
│   │
│   ├── mesh-ledger/
│   │   ├── types.rs
│   │   ├── validate.rs
│   │   ├── store.rs
│   │   └── network.rs
│   │
│   ├── mesh-node/
│   │   ├── main.rs
│   │   ├── client.rs
│   │   └── daemon.rs
│   │
│   ├── mesh-coordinator/
│   │   ├── main.rs
│   │   ├── api.rs
│   │   ├── db.rs
│   │   └── error.rs
│   │
│   └── mesh-validator/
│       └── main.rs
│
├── docs/
│   ├── FULL_SYSTEM_SPEC.md
│   ├── ARCHITECTURE.md
│   ├── LEDGER.md
│   ├── PROTOCOL.md
│   ├── SECURITY.md
│   ├── CONSENSUS_INVARIANTS.md
│   ├── CRASH_RECOVERY.md
│   ├── ROADMAP.md
│   └── FULL_ORIGINAL_SPEC.md
│
└── scripts/
    ├── verify.sh
    ├── verify.ps1
    ├── build-release.sh
    ├── build-release.ps1
    ├── demo-local.sh
    └── demo-local.ps1
```

---

# Requirements

## Rust

The repository pins Rust in `rust-toolchain.toml`.

Install Rust using rustup:

### Windows PowerShell

```powershell
winget install Rustlang.Rustup
rustup toolchain install 1.97.1
rustup default 1.97.1
```

Or install rustup from <https://rustup.rs/>.

### Linux/macOS

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup toolchain install 1.97.1
rustup default 1.97.1
```

Verify:

```bash
rustc --version
cargo --version
```

## Native build tools

Windows normally needs the Visual Studio C++ Build Tools if the Rust MSVC target was selected.

Linux may need:

```bash
sudo apt-get install build-essential pkg-config
```

macOS needs Xcode Command Line Tools:

```bash
xcode-select --install
```

SQLite is bundled through the Rust dependency, so a separately installed SQLite development package should not be required.

---

# Compile everything

From the repository root:

```bash
cargo build --release --workspace
```

Expected binaries:

## Windows

```text
target\release\mesh.exe
target\release\mesh-coordinator.exe
target\release\mesh-validator.exe
```

## Linux/macOS

```text
target/release/mesh
target/release/mesh-coordinator
target/release/mesh-validator
```

Run the full verification suite:

### Linux/macOS

```bash
./scripts/verify.sh
```

### Windows PowerShell

```powershell
./scripts/verify.ps1
```

The verification script runs formatting checks, compilation, tests, and Clippy.

---

# Install the participant binary for the current user

Linux/macOS:

```bash
./scripts/install-user.sh
```

Windows PowerShell:

```powershell
./scripts/install-user.ps1
```

These scripts compile only the `mesh` participant client and copy it to a per-user binary directory. They are not MSI/PKG installers; native OS installer packaging remains on the roadmap.

---

# Build a release folder

### Linux/macOS

```bash
./scripts/build-release.sh
```

### Windows PowerShell

```powershell
./scripts/build-release.ps1
```

The script creates `dist/` containing the three native binaries plus the documentation/config files required to deploy them.

For a participant-only machine, only `mesh` is required.

---

# Quick local MVP mode

This mode uses the coordinator's local SQLite ledger and is useful only for development/testing.

Terminal 1:

```bash
mesh-coordinator seed --db mesh.db --start 2 --end 5000000 --shards 32
mesh-coordinator serve --db mesh.db --listen 127.0.0.1:8080
```

Terminal 2:

```bash
mesh --home .mesh-node-a init
mesh --home .mesh-node-a daemon --workers 2
```

Terminal 3:

```bash
mesh --home .mesh-node-b init
mesh --home .mesh-node-b daemon --workers 2
```

After a node completes community-funded work:

```bash
mesh --home .mesh-node-a balance
```

Then submit a paid distributed job:

```bash
mesh --home .mesh-node-a submit-prime --start 2 --end 10000000 --shards 32
```

A requester is excluded from executing its own paid shards.

---

# Recommended quorum-ledger mode

Public/community deployment should use validator mode.

A four-validator lab uses four independent identities and a 3-of-4 threshold.

## 1. Generate validator identities

```bash
mesh-validator id --home .validator-1
mesh-validator id --home .validator-2
mesh-validator id --home .validator-3
mesh-validator id --home .validator-4
```

Each command prints:

```text
validator_id=mesh_...
public_key_b64=...
```

Copy `config/validators.example.json` to `validators.json` and fill in those values.

Example:

```json
{
  "threshold": 3,
  "community_issuance_limit_mcu": 1000000000,
  "members": [
    {
      "validator_id": "mesh_...",
      "url": "http://127.0.0.1:9101",
      "public_key_b64": "..."
    },
    {
      "validator_id": "mesh_...",
      "url": "http://127.0.0.1:9102",
      "public_key_b64": "..."
    },
    {
      "validator_id": "mesh_...",
      "url": "http://127.0.0.1:9103",
      "public_key_b64": "..."
    },
    {
      "validator_id": "mesh_...",
      "url": "http://127.0.0.1:9104",
      "public_key_b64": "..."
    }
  ]
}
```

`1000000000 mCU` is `1,000,000 CU` maximum lifetime community bootstrap issuance for that validator membership file. Choose policy deliberately before a public deployment.

## 2. Start validators

Terminal 1:

```bash
mesh-validator serve \
  --home .validator-1 \
  --db validator-1.db \
  --listen 127.0.0.1:9101 \
  --validators validators.json
```

Terminal 2:

```bash
mesh-validator serve \
  --home .validator-2 \
  --db validator-2.db \
  --listen 127.0.0.1:9102 \
  --validators validators.json
```

Terminal 3:

```bash
mesh-validator serve \
  --home .validator-3 \
  --db validator-3.db \
  --listen 127.0.0.1:9103 \
  --validators validators.json
```

Terminal 4:

```bash
mesh-validator serve \
  --home .validator-4 \
  --db validator-4.db \
  --listen 127.0.0.1:9104 \
  --validators validators.json
```

For a real Internet deployment, validators should be operated by independent parties and exposed through HTTPS/reverse proxies. Do not expose plaintext validator HTTP endpoints directly over the public Internet.

## 3. Reserve community bootstrap work through quorum

```bash
mesh-coordinator seed \
  --db mesh.db \
  --validators validators.json \
  --start 2 \
  --end 5000000 \
  --shards 32
```

This does two things:

1. proposes a `CommunityReserve` ledger transaction,
2. moves CU from the bounded community issuance account into that job's escrow account.

The coordinator cannot simply credit a user balance.

## 4. Start the scheduler

```bash
mesh-coordinator serve \
  --db mesh.db \
  --listen 127.0.0.1:8080 \
  --validators validators.json
```

In this mode the validators are authoritative for balances.

## 5. Start participant nodes

```bash
mesh --home .mesh-a init
mesh --home .mesh-a daemon --workers 4
```

On another machine:

```bash
mesh --coordinator https://coordinator.example.org --home .mesh init
mesh --coordinator https://coordinator.example.org --home .mesh daemon --workers 4
```

Workers only need outbound access to the coordinator. They do not need an inbound listening port.

---

# How clients find work and one another

In v0.2, participant nodes do **not** open random inbound peer ports or directly discover workers.

The scheduler model is:

```text
Worker A ──poll──► Coordinator ◄──poll── Worker B
                    │
                    ├── pending shard queue
                    ├── capability registry
                    └── leases
```

This choice is deliberate because it works through normal home NAT/firewalls and minimizes attack surface.

The end-state architecture will add peer discovery/data paths for model blocks and low-latency compute neighborhoods, while the control plane can remain federated rather than requiring direct arbitrary remote access.

See `docs/ARCHITECTURE.md`.

---

# How CU is stored in quorum mode

CU is not stored as one editable number on the coordinator.

The authoritative state is a replicated, append-only, quorum-certified hash chain.

A paid job reservation is represented as:

```text
Requester             Job Escrow
   -30 CU   ───────►    +30 CU
```

A completed shard is represented as:

```text
Job Escrow             Provider
   -8 CU    ───────►     +8 CU
```

Every normal transaction must sum to zero.

The only issuance source is:

```text
mesh:community:issuance
```

and that account is limited by `community_issuance_limit_mcu` in the pinned validator membership file.

Each validator stores:

- every quorum certificate,
- every transaction,
- every posting,
- derived balances,
- duplicate settlement claims,
- persistent vote locks.

A ledger entry contains the previous entry hash, creating a tamper-evident chain.

A transaction becomes certified only after the configured threshold of independent validators signs the exact entry hash.

See `docs/LEDGER.md`.

---

# Verify the ledger without trusting the coordinator

Participant nodes can query validators directly.

```bash
mesh --home .mesh-a ledger-status --validators validators.json
```

This requires a quorum of validators to independently agree on both:

- the ledger head,
- the participant's balance/activity proof.

Mirror the entire ledger locally:

```bash
mesh --home .mesh-a ledger-sync \
  --validators validators.json \
  --db .mesh-a/ledger-mirror.db
```

Audit it later without trusting the coordinator:

```bash
mesh --home .mesh-a ledger-audit \
  --validators validators.json \
  --db .mesh-a/ledger-mirror.db
```

The audit checks:

- certificate quorum,
- validator signatures,
- membership hash,
- sequence continuity,
- previous-hash continuity,
- transaction hashes,
- CU conservation,
- issuance limit,
- duplicate settlement claims,
- historical requester signatures,
- historical provider signatures,
- deterministic reward size,
- deterministic work result.

---

# Coordinator settlement recovery

In quorum mode, MESH persists the exact ledger transaction locally **before** asking validators to certify it. If the coordinator crashes or loses connectivity during reservation/reward settlement, the local job/shard remains blocked in `funding` or `settling` rather than being double-spent or reissued.

Recovery runs automatically when the coordinator starts with `--validators`. It can also be run explicitly:

```bash
mesh-coordinator recover --db mesh.db --validators validators.json
```

Recovery asks independent validators for a signed quorum claim proof. If the claim is already certified, it finalizes local state. If it is not yet certified, it retries the **same persisted transaction** so existing validator vote locks remain compatible.

# Validator recovery

A validator that was offline can catch up from peers:

```bash
mesh-validator sync \
  --db validator-3.db \
  --validators validators.json
```

Then verify its entire local replica:

```bash
mesh-validator audit \
  --db validator-3.db \
  --validators validators.json
```

---

# Why this is not a blockchain or cryptocurrency

MESH deliberately borrows useful distributed-ledger ideas without introducing a financial token.

Used concepts:

- content hashes,
- append-only history,
- digital signatures,
- independent replicas,
- quorum certificates,
- deterministic state replay.

Not present:

- mining,
- proof-of-work,
- proof-of-stake,
- coins,
- wallets containing financial assets,
- gas fees,
- market pricing,
- exchange listings,
- buying CU.

CU is a non-transferable accounting unit representing previously contributed compute.

---

# Identity storage

The node identity is stored under the selected `--home` directory:

```text
.mesh/identity.json
```

On Unix, the implementation attempts to set the file to mode `0600`.

The private key never needs to be sent to the coordinator or validators.

Back up this identity if you want to retain access to the CU associated with that node identity.

For a production client, the next security step should be OS-native secure key storage:

- Windows DPAPI / CNG,
- macOS Keychain/Secure Enclave where applicable,
- Linux Secret Service/TPM where available.

---

# Security model

MESH workers do not expose SSH, RDP, shell access, or arbitrary host command execution.

The current runtime only accepts a known `WorkSpec` enum.

That is intentional.

Do **not** replace the declarative work protocol with "download a binary and run it" when adding workloads.

For future generalized workloads, prefer:

- WASI/WebAssembly for portable sandboxed CPU tasks,
- dedicated signed runtime adapters for GPU workloads,
- explicit filesystem/network capabilities,
- memory/CPU/GPU quotas,
- OS sandbox primitives.

See `docs/SECURITY.md`.

---

# Networking for a public deployment

The binaries themselves speak HTTP in this repository. A public deployment should place them behind TLS termination such as:

```text
Internet
   │
 HTTPS
   │
Reverse Proxy / Load Balancer
   │
   ├── Coordinator :8080
   └── Validators  :9101...
```

Recommended protections before public exposure:

- TLS 1.3,
- rate limits,
- request body size limits,
- DDoS protection,
- validator network ACLs where practical,
- separate validator hosts/operators,
- monitoring and alerting,
- database backups,
- pinned validator membership distributed with clients.

---

# Current workload

The first workload is:

```rust
WorkSpec::PrimeCount { start, end }
```

The coordinator splits the requested range into deterministic shards.

Example:

```text
2 .. 10,000,000
       │
       ▼
32 shards
       │
 ┌─────┼────────┐
 ▼     ▼        ▼
A      B        C ...
```

The provider returns a deterministic count. Validators recompute the shard before certifying payment.

This is intentionally expensive verification for an MVP. Future workload types need workload-specific verification schemes.

---

# Planned MESH AI layer

The architecture is intended to grow into:

```text
MESH AI
   │
   ├── model registry
   ├── content-addressed model chunks
   ├── peer-to-peer model seeding
   ├── CUDA backend
   ├── ROCm backend
   ├── Metal backend
   ├── GGUF / safetensors manifests
   ├── GPU capability benchmark
   ├── latency-aware scheduler
   ├── pipeline parallelism
   ├── model parallelism
   ├── batch parallelism
   ├── tensor transport
   └── failure-aware rerouting
         │
         ▼
MESH Compute Core
```

The first AI milestone should be **batch inference across independent GPUs**, not WAN tensor parallelism. The second should be pipeline/model partitioning across a small low-latency cluster.

---

# Known limitations before public production

This repository intentionally documents the remaining work instead of disguising it.

1. The code needs a full compile/test/Clippy pass in a Rust-equipped environment after handoff.
2. Validator membership rotation/epochs are not yet implemented.
3. Consensus is a quorum-certified linear log, not a complete BFT view-change protocol.
4. The coordinator is still the centralized scheduler, although accounting is independently replicated.
5. Public TLS is expected to be provided by a reverse proxy rather than the binaries directly.
6. GPU execution is not yet implemented.
7. Work verification currently recomputes deterministic CPU work.
8. Community issuance authorization is bounded by validator policy but does not yet require a separate governance key/proposal process.
9. Key storage is file-based rather than OS hardware-backed.
10. No automatic MSI/PKG/DEB installer is included yet; release scripts produce native binaries and a distribution folder.
11. NAT-friendly P2P model/data transport is a future layer.
12. Coordinator crash recovery for certified reservations/rewards is implemented through durable ledger intents and `mesh-coordinator recover`; a full multi-coordinator BFT view-change protocol remains a production blocker.

These are specifically called out in `CODEX_HANDOFF.md` as next engineering targets.

---

# Development principles

When extending MESH, preserve these invariants:

1. **No purchased CU.**
2. **New identities start at zero.**
3. **Paid work is escrowed before scheduling.**
4. **Normal ledger transactions conserve CU exactly.**
5. **Community issuance is explicit and bounded.**
6. **No arbitrary remote code execution.**
7. **Provider results are signed over the exact work metadata.**
8. **A paid requester cannot process its own shard.**
9. **A reward claim settles at most once.**
10. **Validators do not double-vote at one ledger height.**
11. **Ordinary users can independently verify ledger state.**
12. **The coordinator is not the ultimate authority for CU.**

---

# License

MIT. See `LICENSE`.
