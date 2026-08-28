# hocMESH Compute

**hocMESH = Mutual Exchange of Shared Hardware**

hocMESH is an open, contribution-first distributed compute network. Participants contribute idle compute, earn non-monetary Compute Units (CU), bank those units, and later spend them on work executed by other participants.

There is **no payment system, token, cryptocurrency, market, or purchasable credit** in this design.

> Contribute first. Compute later.

This repository is a Rust implementation of hocMESH Compute Core and the hocMESH AI control/data-plane architecture.

## What is implemented in v0.3

This repository contains working source for three native Rust programs:

- `hocmesh` — the peer CLI: it serves hardware and it spends what that earns
- `hocmesh-coordinator` — workload scheduler and node control plane
- `hocmesh-validator` — replicated CU ledger validator

Implemented architecture:

- Ed25519 node identities generated locally.
- Replay-resistant signed API requests using timestamp + cryptographic nonce.
- Hardware discovery and CPU benchmarking.
- Operator-set resource limits: a node advertises the share of CPU/memory/GPU it lends, never the whole machine.
- Measured network coordinates, so the scheduler ranks workers by distance to the requester rather than to itself.
- GPU detection for NVIDIA/CUDA and Apple/Metal-capable systems where detectable by the current hardware adapter.
- Declarative allow-listed work instead of arbitrary remote binaries.
- Deterministic prime-range workload as the first safe distributed workload.
- One-command inference setup: `runtime-install` fetches a llama.cpp build pinned by SHA-256, `model-pull` fetches and verifies GGUF weights. Neither trusts a name.
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
- Quorum-signed portable snapshots, so a new replica adopts a verified state and syncs from there rather than replaying the chain from genesis.
- Direct validator balance/head verification independent of the coordinator.
- Crash-safe coordinator ledger intents with startup/manual recovery.
- Coordinator rebuild from the chain, so a lost scheduling database is not a lost job.
- Signed validator claim proofs for reservation/reward reconciliation.
- Per-process ledger proposal serialization to reduce same-height races.
- SQLite-backed network model registry.
- GGUF/safetensors manifests and SHA-256 content-addressed model chunks.
- HTTP peer model seeding with rarest-first multi-peer planning.
- CUDA, ROCm, and Metal discovery and feature-gated llama.cpp adapters.
- Latency/cache/capability-aware AI scheduling and all three parallelism planners.
- Checksum-bound tensor framing, ordered delivery, replay rejection, and route failover.
- Distributed batch inference with leases, results, status, and worker rerouting.

## Runtime boundary

hocMESH v0.3 executes independent distributed inference batches through an external llama.cpp runtime. `hocmesh runtime-install` will fetch a pinned build for the host platform and verify it against a SHA-256 compiled into the binary, so no separate llama.cpp setup is required; `--runtime` still accepts a build you compiled yourself, which is the path to take for CUDA or ROCm acceleration. Pipeline and tensor/model plans and transport are implemented; actual partial-layer kernels require a compatible runtime plugin because stock llama.cpp does not expose them.

The current executable workload is deterministic CPU prime-range computation. The architecture deliberately proves the harder control-plane primitives first:

1. identity,
2. trust,
3. contribution accounting,
4. replicated state,
5. work scheduling,
6. sharding,
7. verification,
8. fault recovery.

See `docs/HOCMESH_AI.md` for commands, interfaces, validation, and hardware/runtime boundaries, and `docs/DEPLOYMENT.md` for the runbook that takes this onto two or more real machines.

See `docs/FULL_ORIGINAL_SPEC.md` and `docs/ROADMAP.md`.

---

# Repository layout

```text
hocMESH/
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
│   ├── hocmesh-protocol/
│   │   └── shared wire types, signed request format, hashes, IDs
│   │
│   ├── hocmesh-core/
│   │   ├── identity.rs
│   │   ├── hardware.rs
│   │   └── compute.rs
│   │
│   ├── hocmesh-ledger/
│   │   ├── types.rs
│   │   ├── validate.rs
│   │   ├── store.rs
│   │   └── network.rs
│   │
│   ├── hocmesh-node/
│   │   ├── main.rs
│   │   ├── client.rs
│   │   └── daemon.rs
│   │
│   ├── hocmesh-coordinator/
│   │   ├── main.rs
│   │   ├── api.rs
│   │   ├── db.rs
│   │   └── error.rs
│   │
│   ├── hocmesh-validator/
│   │   └── main.rs
│   │
│   └── hocmesh-desktop/
│       ├── supervisor.rs   starts, finds and stops the node
│       ├── dashboard.rs    what the window is allowed to say
│       ├── tray.rs         the menu and the health icon
│       └── ui/             the window itself
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
target\release\hocmesh.exe
target\release\hocmesh-coordinator.exe
target\release\hocmesh-validator.exe
```

## Linux/macOS

```text
target/release/hocmesh
target/release/hocmesh-coordinator
target/release/hocmesh-validator
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

These scripts compile only the `hocmesh` peer binary and copy it to a per-user binary directory. Tagged GitHub releases additionally provide native Windows MSI, macOS PKG, and Linux DEB installers. To build installers locally from an existing release binary:

```bash
./scripts/package-linux.sh target/release/hocmesh "$(cat VERSION)" dist amd64
./scripts/package-macos.sh target/release/hocmesh "$(cat VERSION)" dist
```

```powershell
dotnet tool install --global wix --version 6.0.2
./scripts/package-windows.ps1 -Binary target/release/hocmesh.exe -Version (Get-Content VERSION -Raw).Trim() -OutputDirectory dist
```

Install a downloaded release package with the native platform tool:

```bash
sudo apt install ./hocmesh_0.3.0_amd64.deb
sudo installer -pkg ./hocmesh-0.3.0.pkg -target /
```

```powershell
Start-Process msiexec.exe -Wait -ArgumentList '/i', '.\hocmesh-0.3.0-x86_64.msi'
```

---

# Desktop app

`hocmesh-desktop` is the window and the system tray: a dashboard showing whether this machine is contributing, exactly how much of it is being lent, and the ledger of compute units it has earned and spent, together with the controls to change any of that.

It is not the node. The node is `hocmesh daemon`, a separate process that keeps running when the window is closed, and the app starts, watches and stops it — the same split Docker Desktop draws between its window and its engine. Two rules follow, and both are enforced in code rather than in the interface: a daemon the app did not start is never stopped when the app quits, and a daemon that is already running is attached to, never duplicated.

Run it from a checkout. The node has to be findable — beside the app binary, or on `PATH`:

```bash
cargo build --release -p hocmesh
cargo run -p hocmesh-desktop
```

Build the installers, which lay the app and the node down together so the app finds the matching daemon beside itself rather than an older build somewhere on `PATH`:

```bash
./scripts/package-desktop.sh target/release/hocmesh "$(cat VERSION)" dist
```

```powershell
./scripts/package-desktop.ps1 -Binary target/release/hocmesh.exe -Version (Get-Content VERSION -Raw).Trim() -OutputDirectory dist
```

That produces an MSI and an NSIS setup executable on Windows, a `.dmg` on macOS, and a `.deb` and an `.AppImage` on Linux; tagged releases carry all of them. `crates/hocmesh-desktop/BUNDLING.md` covers how the node is embedded and what each platform needs installed first.

There is no client build and no server build. Every hocMESH install is a whole peer — node, coordinator and validator — because the model is a torrent swarm run in the other order: you seed first, lending CPU, memory and GPU to other people's work, and what that earns is what lets you later reach for somebody else's hardware. Both installers above carry all three binaries. The only difference is whether the machine has a screen: the desktop installer adds the window over the same peer, the headless installers further up leave it out.

So they replace each other rather than sitting side by side. Both lay down `/usr/bin/hocmesh` as the command an operator types, and each declares that in package metadata — the desktop `.deb` carries `Provides`, `Conflicts` and `Replaces: hocmesh`, the headless one `Conflicts` and `Replaces: hoc-mesh-desktop` — so installing either on a machine that has the other swaps it cleanly instead of failing on a file collision. The app still prefers the node it shipped with, beside its own binary, over whatever is on `PATH`.

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

For a participant-only machine, only `hocmesh` is required.

---

# Run a local model

Two commands, no separate llama.cpp setup and no manual weight downloads.

```bash
hocmesh runtime-install                        # pinned llama.cpp, verified by digest
hocmesh model-pull qwen2.5-0.5b-instruct       # GGUF weights, verified by digest
hocmesh infer --model-id qwen2.5-0.5b-instruct --prompt "hello"
```

`runtime-install` downloads the llama.cpp release pinned in
`crates/hocmesh-gpu/src/runtime.rs` for this OS and architecture and checks it
against a SHA-256 compiled into the binary. It is pinned by digest rather than
resolved by name on purpose: hocMESH's safety property is that a node executes
allow-listed work and never a binary somebody sent it, and "fetch the latest
build" would quietly hand that away. Mismatched bytes are discarded, not
installed. `hocmesh runtime-status` shows what is pinned and what is installed
without downloading anything.

`model-pull` resolves the file on Hugging Face, downloads it with resume,
verifies the SHA-256 the Hub published for it, reads the architecture out of the
GGUF header instead of asking you to assert it, chunks it into the
content-addressed store, and registers it.

```bash
hocmesh model-catalog                                        # ids known by name
hocmesh model-pull --repository Qwen/Qwen2.5-7B-Instruct-GGUF --quantisation q4_k_m
hocmesh model-pull --url https://example.org/m.gguf --sha256 <64 hex>
```

The catalogue maps a memorable id to a repository and a preferred quantisation.
It deliberately carries no digests: those are resolved per pull and checked
against the bytes that arrive, because a digest shipped in the binary that
nobody could re-verify would look like a guarantee it is not. `--sha256` is
optional when the source publishes a digest and required with `--url`.

Once installed, `infer` and `daemon` find the runtime without a flag.
`--runtime` / `--ai-runtime` still override it, and `daemon --no-ai` declines AI
work outright.

Running a model for yourself and running one for strangers are separate
decisions, so serving inference to the mesh is asked for separately:

```bash
hocmesh limits --ai on        # run other people's inference here
hocmesh limits --ai off       # never, whatever hardware is lent
hocmesh limits --ai auto      # the default: on when a GPU is lent
```

`--ai on` works on a machine with no GPU at all. The node then advertises its
shared CPU slice as a device and serves inference on it — slowly, but it serves
it, which is what the pinned CPU runtime is for. `auto` is what an existing
`limits.json` says, so upgrading changes nothing about what a node already
offered.

`docs/DEPLOYMENT.md` has the full two-machine runbook.

---

# Quick local MVP mode

This mode uses the coordinator's local SQLite ledger and is useful only for development/testing.

Terminal 1:

```bash
hocmesh-coordinator seed --db hocmesh.db --start 2 --end 5000000 --shards 32
```

The local mode keeps a coordinator-owned ledger, so this mint answers to nobody
but the operator running it. That is the whole reason it is development-only:
with `--validators`, minting needs sponsorships from the sitting set.

Terminal 2:

```bash
hocmesh --home .hocmesh-node-a init
hocmesh --home .hocmesh-node-a daemon --workers 2
```

Terminal 3:

```bash
hocmesh --home .hocmesh-node-b init
hocmesh --home .hocmesh-node-b daemon --workers 2
```

After a node completes community-funded work:

```bash
hocmesh --home .hocmesh-node-a balance
```

Then submit a paid distributed job:

```bash
hocmesh --home .hocmesh-node-a submit-prime --start 2 --end 10000000 --shards 32
```

A requester is excluded from executing its own paid shards.

---

# Recommended quorum-ledger mode

Public/community deployment should use validator mode.

A four-validator lab uses four independent identities and a 3-of-4 threshold.

## 1. Generate validator identities

```bash
hocmesh-validator id --home .validator-1
hocmesh-validator id --home .validator-2
hocmesh-validator id --home .validator-3
hocmesh-validator id --home .validator-4
```

Each command prints:

```text
validator_id=hocmesh_...
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
      "validator_id": "hocmesh_...",
      "url": "http://127.0.0.1:9101",
      "public_key_b64": "..."
    },
    {
      "validator_id": "hocmesh_...",
      "url": "http://127.0.0.1:9102",
      "public_key_b64": "..."
    },
    {
      "validator_id": "hocmesh_...",
      "url": "http://127.0.0.1:9103",
      "public_key_b64": "..."
    },
    {
      "validator_id": "hocmesh_...",
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
hocmesh-validator serve \
  --home .validator-1 \
  --db validator-1.db \
  --listen 127.0.0.1:9101 \
  --validators validators.json
```

Terminal 2:

```bash
hocmesh-validator serve \
  --home .validator-2 \
  --db validator-2.db \
  --listen 127.0.0.1:9102 \
  --validators validators.json
```

Terminal 3:

```bash
hocmesh-validator serve \
  --home .validator-3 \
  --db validator-3.db \
  --listen 127.0.0.1:9103 \
  --validators validators.json
```

Terminal 4:

```bash
hocmesh-validator serve \
  --home .validator-4 \
  --db validator-4.db \
  --listen 127.0.0.1:9104 \
  --validators validators.json
```

For a real Internet deployment, validators should be operated by independent parties and exposed through HTTPS/reverse proxies. Do not expose plaintext validator HTTP endpoints directly over the public Internet.

## 3. Reserve community bootstrap work through quorum

Minting is the set's decision, not the coordinator's, so the mint has to be
sponsored first. On each validator machine, in turn:

```bash
hocmesh community-vouch \
  --validators validators.json \
  --job-id job_bootstrap_1 \
  --start 2 \
  --end 5000000 \
  --shards 32
```

Each prints one signature line. Collect `threshold` of them into a JSON array
in `sponsors.json`, then seed:

```bash
hocmesh-coordinator seed \
  --db hocmesh.db \
  --validators validators.json \
  --job-id job_bootstrap_1 \
  --sponsors sponsors.json \
  --start 2 \
  --end 5000000 \
  --shards 32
```

This does two things:

1. proposes a `CommunityReserve` ledger transaction carrying those sponsorships,
2. moves CU from the bounded community issuance account into that job's escrow account.

The coordinator cannot simply credit a user balance, and it holds no key that
can mint: without `threshold` valid sponsorships from the sitting set, every
validator rejects the transaction.

## 4. Start the scheduler

```bash
hocmesh-coordinator serve \
  --db hocmesh.db \
  --listen 127.0.0.1:8080 \
  --validators validators.json
```

In this mode the validators are authoritative for balances.

## 5. Start participant nodes

```bash
hocmesh --home .hocmesh-a init
hocmesh --home .hocmesh-a daemon --workers 4
```

On another machine:

```bash
hocmesh --coordinator https://coordinator.example.org --home .hocmesh init
hocmesh --coordinator https://coordinator.example.org --home .hocmesh daemon --workers 4
```

Workers only need outbound access to the coordinator. They do not need an inbound listening port unless they opt into answering other nodes' latency probes.

### Choosing how much of the machine to lend

A contributor lends a share, not the whole box. The share is stored under
`--home` and is what the node advertises; the coordinator never sees the rest.

```bash
hocmesh --home .hocmesh-a limits
hocmesh --home .hocmesh-a limits --cpu-percent 50 --memory-percent 25 --gpu-percent 0
```

Limits apply from the first contact: `hocmesh init` registers the same share the
daemon will advertise. Setting `--gpu-percent 0` withdraws the GPU entirely, and
the node stops advertising accelerators at all. `--ai on|off|auto` is a separate
question from all three, because inference is the one workload that runs a
stranger's prompt through a stranger's weights rather than allow-listed
arithmetic; see "Run a local model".

### Seeing where the node sits

```bash
hocmesh --home .hocmesh-a proximity
```

A node that has not yet measured enough peers reports that it has no place yet
rather than inventing one. To also answer other nodes' probes, give the daemon a
port to listen on:

```bash
hocmesh --home .hocmesh-a daemon --workers 4 --probe-listen 0.0.0.0:8646
```

---

# How clients find work and one another

Participant nodes do **not** open random inbound peer ports and do not discover
workers directly. Work is still handed out by the coordinator.

The scheduler model is:

```text
Worker A ──poll──► Coordinator ◄──poll── Worker B
                    │
                    ├── pending shard queue
                    ├── capability registry
                    └── leases
```

This choice is deliberate because it works through normal home NAT/firewalls and minimizes attack surface.

The one thing nodes do measure between themselves is distance. Every daemon
probes a small sample of peers outbound and fits a network coordinate from the
round trips it observes, which is what lets the scheduler rank workers by their
distance to the requester rather than to the coordinator.

Probing outward needs no inbound port. Answering other nodes' probes does, so it
stays opt-in behind `--probe-listen`; a node that never sets it is still placed
on the map by its own measurements.

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
hocmesh:community:issuance
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
hocmesh --home .hocmesh-a ledger-status --validators validators.json
```

This requires a quorum of validators to independently agree on both:

- the ledger head,
- the participant's balance/activity proof.

Mirror the entire ledger locally:

```bash
hocmesh --home .hocmesh-a ledger-sync \
  --validators validators.json \
  --db .hocmesh-a/ledger-mirror.db
```

Or start from a snapshot instead, so a new mirror does not replay the whole
chain to get to today:

```bash
hocmesh --home .hocmesh-a ledger-restore \
  --validators validators.json \
  --db .hocmesh-a/ledger-mirror.db \
  --snapshot ledger-snapshot.json
```

A validator produces that file with `hocmesh-validator snapshot`. Nothing in
it is trusted: it is refused unless the certificate and the checkpoint both
carry a quorum from `validators.json`, name the same entry, and the state
inside hashes to the digest that quorum signed. So it can be published
anywhere, and a restore over a store that already holds a chain is refused.

Page back through the postings behind a balance:

```bash
hocmesh --home .hocmesh-a ledger-history \
  --db .hocmesh-a/ledger-mirror.db \
  --account <node-id> \
  --limit 20
```

Pass `--validators validators.json` instead of `--db` to read it off the
network rather than a local mirror, and `--before <sequence>` to follow the
cursor the previous page printed. Pages run newest first and never stop inside
one entry's postings, because the cursor is a sequence: a page that split an
entry would leave postings the next page could never ask for.

See what the coordinator and the ledger still disagree about:

```bash
hocmesh --coordinator http://127.0.0.1:8080 reconciliation
```

Intents the coordinator wrote down but has not settled, with the attempt count
and the last failure attached, plus a count of work left waiting on funding
that nothing is chasing any more. A background pass retries the settleable ones
every 15 seconds and once at startup; one that can never settle under its own
claim key is parked with the reason rather than retried forever, and never
blocks the intents behind it. Nothing here moves CU: the orphan count is a
report, because closing that gap locally would be the coordinator deciding CU
into existence.

Audit it later without trusting the coordinator:

```bash
hocmesh --home .hocmesh-a ledger-audit \
  --validators validators.json \
  --db .hocmesh-a/ledger-mirror.db
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

In quorum mode, hocMESH persists the exact ledger transaction locally **before** asking validators to certify it. If the coordinator crashes or loses connectivity during reservation/reward settlement, the local job/shard remains blocked in `funding` or `settling` rather than being double-spent or reissued.

Recovery runs automatically when the coordinator starts with `--validators`. It can also be run explicitly:

```bash
hocmesh-coordinator recover --db hocmesh.db --validators validators.json
```

Recovery asks independent validators for a signed quorum claim proof. If the claim is already certified, it finalizes local state. If it is not yet certified, it retries the **same persisted transaction** so existing validator vote locks remain compatible.

# Rebuilding a lost coordinator

`recover` assumes the coordinator's database survived. If it did not - the disk is gone, or the host is - a replacement can be rebuilt from the chain, because the coordinator caches scheduling state over facts the ledger already keeps:

```bash
hocmesh-coordinator rebuild --db new.db --validators validators.json
```

It verifies every certificate, refuses a gap in the sequence, follows membership changes forward, and turns each settled transaction back into job and shard rows. Shard ids are derived from the job id rather than remembered, so a replacement reconstructs the same ids the dead coordinator issued and finishes a half-done job without re-offering a settled shard. Balances are not replayed: in quorum mode the validators answer those. Running it twice is safe.

# Validator recovery

A validator that was offline can catch up from peers:

```bash
hocmesh-validator sync \
  --db validator-3.db \
  --validators validators.json
```

Then verify its entire local replica:

```bash
hocmesh-validator audit \
  --db validator-3.db \
  --validators validators.json
```

---

# Why this is not a blockchain or cryptocurrency

hocMESH deliberately borrows useful distributed-ledger ideas without introducing a financial token.

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
.hocmesh/identity.json
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

hocMESH workers do not expose SSH, RDP, shell access, or arbitrary host command execution.

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

## What the network tests actually break

The quorum flow suite does not assume a healthy wire. It puts a fault-injecting
TCP relay in front of every validator and advertises the relay addresses in the
signed validator set, so coordinator-to-validator and validator-to-validator
traffic both go through it. A test can then delay or sever any link mid-request:

- **WAN latency.** 45 ms one way on every hop. Seeding, submitting, settling and
  auditing all still complete.
- **Minority partition.** One of four validators is cut off. The threshold is
  three, so settlement continues; the isolated replica falls behind rather than
  inventing entries, and `hocmesh-validator sync` replays it back into line.
- **Majority partition.** Two of four are cut off, one short of the threshold.
  Delivery fails instead of paying from a database the chain never certified,
  and when the link heals `hocmesh-coordinator recover` settles the stranded
  shard exactly once - running it twice does not pay twice.
- **Clock skew.** A request signed with a timestamp outside the 300 second
  window is refused; one inside it still gets work.

What this is not: every process in those tests runs on one machine over
loopback. Nothing here has crossed a NAT, a real WAN, a lossy or reordering
link, or a bandwidth cap, and no test spans two operating systems. Treat the
suite as evidence that the protocol survives delay and partition, not as
evidence that a multi-host deployment works.


## What the ledger tests assume an adversary will try

The network tests break the wire. These break the evidence.

- **Every single edit to a settled reward.** Take a reward the chain would
  accept and change exactly one thing - double a posting, negate one, drop
  one, redirect the credit to another account, inflate the claimed reward,
  renumber the shard, flip who pays, forge the worker's signature, swap the
  work, relabel the kind - and the chain has to refuse it. Four hundred
  randomised rounds, each one first proving the unedited reward is accepted.
- **The issuance ceiling.** However the community budget is drawn on, it can
  never be drawn past the bound the sitting set agreed to. That bound is the
  whole of "no purchased CU": free work exists, but only up to here.
- **Replay.** A chain fed to two databases leaves them byte-identical; fed to
  the same database twice, every certificate is refused the second time; and
  across every account the balances sum to exactly zero.
- **Equivocation.** The same seats signing two different entries at one height
  get at most one of them onto the chain. Both certificates verify on their
  own - a signature is a claim about one entry - but the branch that loses the
  height funds nothing.
- **A quorum of strangers.** A full threshold of correct signatures from
  validators outside the sitting set settles nothing. Membership, not signature
  count, is what makes a certificate binding.

What this is not: a validator that equivocates *while* its peers are
partitioned, or a coordinator that lies about scheduling rather than about
payment, is still untested. So is any of it on more than one machine.

---

# Current workloads

The workloads are a fixed allow-list, never arbitrary binaries:

```rust
WorkSpec::PrimeCount { start, end }
WorkSpec::MatrixMultiply { seed_a, seed_b, dim, row_start, row_end }
WorkSpec::CollatzPeak { start, end }
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

# hocMESH AI layer

The implemented architecture is:

```text
hocMESH AI
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
hocMESH Compute Core
```

Batch inference and authenticated scheduling/rerouting run end to end. Pipeline and model/tensor planning plus ordered tensor transport are implemented as the control/data plane; actual partial-layer and collective kernels remain the responsibility of the configured backend runtime.

---

# Known limitations before public production

This repository intentionally documents the remaining work instead of disguising it.

1. Validator membership rotation/epochs are not yet implemented.
2. Consensus is a quorum-certified linear log, not a complete BFT view-change protocol.
3. The coordinator is still the centralized scheduler, although accounting is independently replicated.
4. Public TLS is expected to be provided by a reverse proxy rather than the binaries directly.
5. CUDA, ROCm, and Metal execution delegates to an external llama.cpp-compatible process; native in-process kernels are not bundled. `runtime-install` fetches a CPU build pinned by digest, which is enough to run inference but not to accelerate it.
6. Work verification currently recomputes deterministic CPU work.
7. Community issuance authorization is bounded by validator policy but does not yet require a separate governance key/proposal process.
8. Key storage is file-based rather than OS hardware-backed.
9. Installers are unsigned until platform signing identities are configured in the release environment.
10. P2P model seeding uses authenticated HTTP peers; NAT traversal and peer discovery remain deployment concerns.
11. Coordinator crash recovery is implemented through durable ledger intents and `hocmesh-coordinator recover`, and a coordinator whose database is lost entirely can be rebuilt from the chain with `hocmesh-coordinator rebuild`. Both are operator-initiated: automatic failover between competing coordinators, and a full multi-coordinator BFT view-change protocol, remain production blockers.
12. The network has been broken deliberately but never crossed. `cargo test -p hocmesh-integration-tests --test quorum_flow` now runs the quorum behind a fault-injecting TCP relay: WAN-scale latency on every link, a minority partition (settlement continues, the isolated validator falls behind and is repaired with `hocmesh-validator sync`), a majority partition (settlement refuses rather than fakes, and the stranded shard pays exactly once when the link heals), and authentication under clock skew on both sides of the tolerated window. Every process in those tests still runs on one machine over loopback. Multi-host deployment, NAT traversal, packet loss and reordering, and bandwidth limits are untested.
13. Adversarial coverage stops at one lie at a time. The ledger crate now proves that no single edit to a settled reward survives validation, that the issuance ceiling holds under any sequence of mints, that a replayed chain is deterministic and idempotent and sums to zero, that an equivocating quorum cannot fill a height twice, and that a quorum of strangers settles nothing. What no test covers is faults in combination - equivocation while the honest peers are partitioned - or a coordinator that lies about scheduling rather than about payment.

These are specifically called out in `CODEX_HANDOFF.md` as next engineering targets.

---

# Development principles

When extending hocMESH, preserve these invariants:

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
