# hocMESH Roadmap

## Already implemented in this repository

- Rust workspace split by responsibility.
- Native participant/coordinator/validator binaries.
- Ed25519 identities.
- Signed protocol v4 requests with nonces and AI capability advertisements.
- Hardware discovery and CPU benchmarking.
- Declarative CPU work: `PrimeCount`, `MatrixMultiply`, `CollatzPeak`.
- Multi-worker task parallelism.
- Work leasing/requeue.
- Contribution-first local ledger mode.
- Quorum ledger mode.
- Escrow reservations.
- Bounded community issuance.
- Hash-linked entries.
- Quorum certificates.
- Persistent validator vote locks.
- Duplicate settlement protection.
- Certified reservation-to-reward binding.
- Full replica sync/audit for validators and participants.
- Operator resource limits, so a contributor shares a slice rather than the machine.
- Measured network coordinates, so scheduling uses requester-to-worker distance.
- Two-stage inference settlement, so generated text is paid for by a signed exchange rather than by trusting whoever produced it.
- Portable signed snapshots, so a newcomer adopts a quorum-signed state and syncs from there instead of replaying the chain from genesis.
- Out-of-band checkpoint distribution, because that snapshot proves itself against a validator set the reader already trusts and so can travel by any untrusted route.
- Indexed account history, so an operator reconciling a bill can page back through the postings behind a balance -- served by validators, readable from a local mirror, and keyed so a page is a seek rather than a scan.

## Priority 0 — handoff validation

1. Run `cargo fmt --check`.
2. Run `cargo check --workspace`.
3. Run `cargo test --workspace`.
4. Run `cargo clippy --workspace --all-targets -- -D warnings`.
5. Fix all compiler/linter findings.
6. Add GitHub/Azure DevOps CI matrix for Windows/Linux/macOS.
7. Add integration test spawning 4 validators (3-of-4 threshold) + coordinator + 3 nodes.

## Priority 1 — production ledger hardening

- Reconciliation daemon for partial coordinator/ledger failures.
- Sweeping stale inference holding accounts to the commons once the settlement window closes, so a requester that takes delivery and never gives a verdict cannot strand CU indefinitely.
- Property tests for conservation and replay invariants. Covered in `hocmesh-ledger`: a settled reward survives no single edit to its postings, evidence, or signature; the community ceiling is never crossed by any sequence of mints; and a chain replayed into two databases leaves them identical, refuses every certificate a second time, and sums to exactly zero.
- Byzantine/fault-injection tests. Network faults are covered: the quorum flow suite runs a fault-injecting relay for WAN latency, minority and majority partitions, and clock skew. Two Byzantine cases are covered too: an equivocating quorum that signs two entries at one height cannot get both onto the chain, and a full quorum of strangers certifies nothing. What is left is the rest of adversarial *behaviour* - a validator that equivocates while its peers are partitioned, a coordinator that lies about scheduling rather than about payment - and a real multi-host run.

## Priority 2 — scheduler federation

- Regional coordinators.
- Coordinator health/failover. Standing a replacement up from the chain works today (`hocmesh-coordinator rebuild`); what is missing is doing it automatically, without an operator having to notice the old one died.
- Shared/federated job state.
- Resource graph.
- Scheduling by hardware, network, reliability, and cache locality.

## Priority 3 — secure runtime

- Per-platform resource quotas.
- Process isolation.
- WASI sandbox for generic CPU workloads. Until it exists the allow-list is
  the safety property: adding a workload means adding a spec, a result and an
  audit rule to this repository, never shipping a binary to a contributor.
- Explicit capability model for filesystem/network.
- Thermal and user-activity controls.

## Priority 4 — hocMESH AI independent GPU jobs

- CUDA backend first.
- GPU benchmark/profile.
- GGUF/safetensors manifest support.
- Independent batch inference.
- Embedding generation.
- Model cache.
- CU pricing based on calibrated accelerator work rather than wall time.

## Priority 5 — distributed model data

- Content-addressed chunk store.
- Signed model manifests.
- P2P chunk seeding.
- Parallel chunk download.
- Cache advertisement.
- License metadata.

## Priority 6 — distributed model execution

- Model layer partitioner.
- Pipeline stages.
- Activation transport.
- Topology-aware routing.
- Replicated critical layers.
- Node departure recovery.
- Micro-batching.

## Priority 7 — heterogeneous accelerators

- ROCm.
- Metal.
- Intel GPU backend.
- Backend capability normalization.

## Priority 8 — consumer application

- Windows service + tray UI.
- macOS daemon/menu app.
- Linux systemd service/UI.
- Idle detection.
- Gaming/activity pause.
- Temperature/power limits.
- Bandwidth caps.
- Signed auto-update.
- RPM packaging (MSI, PKG, and DEB are shipped by the release workflow).

## North-star demonstration

Run an AI model that **none of the participating machines can run individually**, using only compute contributed by community nodes and credits previously earned through contribution.
