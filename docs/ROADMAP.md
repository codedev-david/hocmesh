# MESH Roadmap

## Already implemented in this repository

- Rust workspace split by responsibility.
- Native participant/coordinator/validator binaries.
- Ed25519 identities.
- Signed protocol v2 requests with nonces.
- Hardware discovery and CPU benchmarking.
- Declarative CPU work.
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

## Priority 0 — handoff validation

1. Run `cargo fmt --check`.
2. Run `cargo check --workspace`.
3. Run `cargo test --workspace`.
4. Run `cargo clippy --workspace --all-targets -- -D warnings`.
5. Fix all compiler/linter findings.
6. Add GitHub/Azure DevOps CI matrix for Windows/Linux/macOS.
7. Add integration test spawning 4 validators (3-of-4 threshold) + coordinator + 3 nodes.

## Priority 1 — production ledger hardening

- Consensus/view-change protocol rather than coordinator-driven proposal sequencing.
- Validator membership epochs and rotation.
- Governance authorization for `CommunityReserve`.
- Snapshot/checkpoint format.
- Efficient indexed transaction/account history.
- Reconciliation daemon for partial coordinator/ledger failures.
- Signed checkpoints distributed out-of-band.
- Property tests for conservation and replay invariants.
- Byzantine/fault-injection tests.

## Priority 2 — scheduler federation

- Regional coordinators.
- Coordinator health/failover.
- Shared/federated job state.
- Latency/bandwidth probes.
- Resource graph.
- Scheduling by hardware, network, reliability, and cache locality.

## Priority 3 — secure runtime

- Per-platform resource quotas.
- Process isolation.
- WASI sandbox for generic CPU workloads.
- Explicit capability model for filesystem/network.
- Thermal and user-activity controls.

## Priority 4 — MESH AI independent GPU jobs

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
- MSI/PKG/DEB/RPM packaging.

## North-star demonstration

Run an AI model that **none of the participating machines can run individually**, using only compute contributed by community nodes and credits previously earned through contribution.
