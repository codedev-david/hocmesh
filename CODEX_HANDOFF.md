# Codex CLI Engineering Handoff

## Mission

Take this repository from hocMESH Compute v0.2 to a compile-clean, integration-tested foundation suitable for progressively adding hocMESH AI.

Do not redesign away the contribution-first cooperative model.

## First command sequence

```bash
rustup show
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

The source package was generated in an environment that did not provide a local Rust compiler, so **the first Codex action should be a real compiler-driven repair pass**. Treat compiler output as authoritative.

## Architecture invariants — do not weaken

1. New participant identities begin with zero CU.
2. CU cannot be purchased or transferred as money.
3. Paid jobs reserve CU into job escrow before work is scheduled.
4. Ordinary transactions sum to exactly zero.
5. Community issuance is explicit and bounded.
6. Provider rewards debit certified job escrow.
7. Requester submit authorization is signed.
8. Job ID is bound to the request nonce.
9. Provider result authorization signs exact job/shard/work/reward/result metadata.
10. Reward work must map to the certified reservation shard.
11. Each assignment reward may settle only once.
12. Validators persist one vote per ledger sequence before returning a signature.
13. A quorum certificate is required before distributed-ledger state changes.
14. Participant clients can mirror and audit the entire ledger.
15. Do not add arbitrary remote shell/binary execution.

## Highest-value immediate improvements

### A. Compile and repair

Fix all Rust type/borrow/trait/API errors before feature work.

Prefer readable modules over the intentionally compact generated sections where refactoring improves maintainability.

### B. Integration test harness

Build a Tokio integration test that launches:

```text
4 validators (threshold 3)
1 coordinator
3 participant nodes
```

Test sequence:

1. create validator membership,
2. start empty replicas,
3. reserve one community bootstrap job,
4. node A performs a shard,
5. quorum pays A,
6. verify A balance from all validators,
7. A reserves a paid job,
8. B/C process its shards,
9. verify escrow drains correctly,
10. attempt duplicate result settlement and require rejection,
11. attempt reused submit nonce and require rejection,
12. stop validator C,
13. continue with the remaining three validators (still meeting 3-of-4 threshold),
14. restart C and sync,
15. audit all four ledger heads and require equality.

### C. Coordinator/ledger reconciliation

There is a distributed transaction boundary between certified ledger state and coordinator SQLite scheduling state. The repository now implements:

```text
local durable intent -> ledger certificate/claim proof -> local finalize
```

Review and stress-test the durable `ledger_intents` + signed claim recovery path; extend it to any new ledger-backed transitions and add failure-injection coverage.

### D. Validator membership epochs

Current certificates bind to one static membership hash.

Add:

```text
membership_epoch
previous_epoch_checkpoint
new membership authorization
```

Historical certificates must remain verifiable under the validator set that existed at their epoch.

### E. Consensus hardening

Current persistent vote locks prevent simple double voting but this is not a complete BFT protocol.

Evaluate a mature consensus implementation or implement a clearly specified protocol with:

- leader/view number,
- timeout/view change,
- quorum intersection assumptions,
- commit rules,
- recovery,
- membership change rules.

Do not improvise consensus without tests/property checks.

### F. Ledger indexing

`activity()` and `reservation()` currently scan certificates. Add normalized indexes populated during certificate apply:

```text
account_activity
job_reservations
assignment_rewards
```

The certificate JSON remains authoritative.

### G. Security tests

Add tests for:

- forged node ID,
- forged Ed25519 signature,
- expired live signature,
- old historical signature accepted during audit,
- nonce replay,
- duplicate job reservation,
- duplicate assignment reward,
- mismatched shard index,
- changed WorkSpec,
- changed reward,
- changed job ID,
- changed funding type,
- insufficient balance,
- negative escrow,
- issuance cap exceeded,
- conflicting validator vote same height,
- invalid membership hash,
- insufficient validator signatures,
- tampered old ledger entry.

## Then begin hocMESH AI

Do not start with WAN tensor parallelism.

Recommended implementation order:

1. `hocmesh-gpu` crate with CUDA capability discovery/benchmark.
2. Signed model manifest crate.
3. Content-addressed local model cache.
4. Independent GPU batch inference worker.
5. Scheduler placement by VRAM/model cache/benchmark.
6. P2P model chunk distribution.
7. Pipeline model partitioning on a LAN.
8. Direct QUIC activation transport.
9. WAN latency-aware pipeline scheduling.
10. ROCm and Metal backends.

## Suggested future workspace

```text
crates/
  hocmesh-protocol
  hocmesh-core
  hocmesh-ledger
  hocmesh-node
  hocmesh-coordinator
  hocmesh-validator
  hocmesh-runtime
  hocmesh-gpu
  hocmesh-model
  hocmesh-p2p
  hocmesh-ai
```

## Release engineering

Add CI jobs for:

```text
Windows x86_64
Linux x86_64
Linux aarch64
macOS arm64
```

Add:

- `cargo audit`,
- `cargo deny`,
- SBOM generation,
- release checksums,
- code signing,
- reproducible build notes.

## Definition of done for the next handoff

A successful next milestone should demonstrate, from fresh checkout:

```text
cargo test --workspace   -> green
cargo clippy             -> green
4-validator / 3-of-4 integration -> green
ledger audit             -> green
node A starts at 0 CU
node A earns CU
node A spends CU
other nodes earn exactly the spent CU
no duplicate payout succeeds
validator outage/rejoin recovers
```

Only after that baseline is stable should hocMESH AI GPU execution become the main workstream.


## Newly implemented crash recovery

The coordinator now persists exact ledger intents before quorum submission. Jobs are held in `funding`/`blocked`, and provider results in `settling`, until certification is reconciled. `hocmesh-coordinator recover` (also invoked best-effort on startup) uses signed quorum claim proofs and retries the same transaction if necessary.

Codex should specifically test crashes at these points:

1. after intent persistence, before any validator vote;
2. after some votes, before certificate formation;
3. after certificate formation, before quorum commit response;
4. after quorum commit, before coordinator finalization;
5. after local finalization, before HTTP response.

No case may result in a second reservation or duplicate provider reward.
