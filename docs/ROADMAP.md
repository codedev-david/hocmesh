# hocMESH Roadmap

## Already implemented in this repository

- Rust workspace split by responsibility.
- Native participant/coordinator/validator binaries.
- Ed25519 identities.
- Signed protocol v6 requests with nonces and AI capability advertisements.
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
- Reconciliation of partial coordinator/ledger failures, on a timer and at startup: every persisted intent is judged on its own, so a broken one no longer blocks the ones queued behind it, and one that can never settle under its own claim key is parked with the reason attached instead of retried forever. Work left waiting on funding that no intent covers is reported and not repaired -- filling that gap locally would be the coordinator ruling on CU. Readable at `/v1/ledger/reconciliation` and via `hocmesh reconciliation`.
- Sweeping abandoned inference batches to the commons, so a requester that takes delivery and then never gives a verdict cannot strand a provider's CU behind a holding account forever. Time is the third verdict and it pays neither party.
- Adversarial coverage of scheduling and of a partitioned quorum: a validator that equivocates while its peers cannot see each other still cannot fork the chain, and a coordinator that lies about what work was done cannot claim more than the work was worth.
- Federated coordinators. Several coordinators serve one job store and split ownership of jobs by rendezvous hash over the set that is currently answering, so no coordinator has to be elected and none has to be told what the others are doing.
- Automatic coordinator failover. Peers are probed on a timer and drop out of the live set after a bounded number of misses, at which point ownership of their jobs moves on its own. Leases the departed coordinator handed out are shortened to a grace window rather than cancelled, so a worker still running the shard can report in and be paid.
- Incremental ledger sync for coordinators, resuming from a persisted watermark that carries the validator set the entries were signed under, so a coordinator that has been down catches up without replaying the chain from genesis.
- A resource graph over the registered machines, with distance taken from measured coordinates where they exist and never invented where they do not, and a clustering pass for work that has to land on several machines at once. Readable at `/v1/topology`.
- Scheduling on hardware, network, reliability, and cache locality rather than arrival order, with a starvation guarantee that outranks any fit. Every axis is derived from something already measured or already priced, and a shard nobody can finish inside its lease is refused rather than scored.
- An execution engine that runs a *layer range* rather than a whole model. `hocmesh-engine` loads blocks `[start, end)` out of a GGUF file and runs a forward pass over an activation it was handed, so a machine holding a third of a model can do a third of the work. Split execution is bit-identical to whole execution, asserted rather than assumed.
- Distributed inference end to end: `stage-serve` chains layer ranges across processes, `model-shard` materialises only the bytes a stage needs and records which bytes are real, and three processes each holding about 41% of a file generate output identical to the whole model in one process.
- Scarcity as a scheduling term. A shard's declared working set is scored against the machine offering to hold it, preferring the smallest machine that fits and ranking a GPU node down for work no GPU helps with. It never touches the price, which every validator recomputes from the spec.
- Validators that repair themselves. A seat that missed a commit fetches what it missed from the rest of the set, on a certificate landing above its head and on a heartbeat, instead of refusing every later entry until an operator runs `sync`.

## Priority 0 — handoff validation — done

Formatting, `cargo check`, the test suite and `clippy -D warnings` all run on
Windows, Linux and macOS in `.github/workflows/ci.yml`, and the integration
suite spawns a real 3-of-4 validator quorum with coordinators and nodes as
separate processes rather than mocking them.

## Priority 1 — production ledger hardening

- Sweeping stale inference holding accounts to the commons once the settlement window closes. **Done.** An expiry is a transaction like any other: it carries the receipt the requester signed on the way in, it may not be proposed until the window has closed, and it pays the commons rather than either party.
- Property tests for conservation and replay invariants. **Done.** Covered in `hocmesh-ledger`: a settled reward survives no single edit to its postings, evidence, or signature; the community ceiling is never crossed by any sequence of mints; and a chain replayed into two databases leaves them identical, refuses every certificate a second time, and sums to exactly zero.
- Byzantine/fault-injection tests. **Done in software.** The quorum flow suite runs a fault-injecting relay for WAN latency, minority and majority partitions, and clock skew. The Byzantine cases are covered too: an equivocating quorum that signs two entries at one height cannot get both onto the chain, a full quorum of strangers certifies nothing, a validator that equivocates while its peers are partitioned still cannot fork, and a coordinator that lies about scheduling cannot take more than the work was worth.
- A real multi-host run. **Not done, and not doable from one machine.** Every fault above is injected between processes on one host, which is honest about protocol behaviour and says nothing about NIC drivers, MTU, NAT, asymmetric routes, or clocks that drift independently. This needs two or more real machines on a real network; it is the only Priority 1 item still open.

## Priority 2 — scheduler federation — done

- Regional coordinators. **Done.** A coordinator is given its own id and region in a `--federation` file and advertises both; a worker's region follows the coordinator that registered it, and crossing a region boundary costs a shard part of its network score rather than disqualifying it.
- Coordinator health/failover. **Done.** Peers are probed on their own timer and leave the live set after a bounded number of consecutive misses. Because ownership is a pure function of the job id and the live set, the survivor starts serving the departed coordinator's jobs without an election, a handover, or an operator noticing. Standing a replacement up from the chain (`hocmesh-coordinator rebuild`) still exists for the case where the database is gone as well.
- Shared/federated job state. **Done.** Coordinators share the job store, and the ownership rule is what stops two of them handing out the same shard. State they do not share comes back from the chain: `sync_from_ledger` resumes from a persisted watermark that records which validator set signed the entries, so a returning coordinator verifies against the set that was in force rather than the set in force now.
- Resource graph. **Done.** Machines are vertices; an edge is a predicted round trip from measured coordinates where both ends have placed themselves, the route through the coordinator where they have not, and an explicit placeholder where nothing has been measured at all. `/v1/topology` reads it back, and a clustering pass finds the tightest set of machines that satisfies a gate for work that has to run in several places at once.
- Scheduling by hardware, network, reliability, and cache locality. **Done.** `poll_work` scores a bounded oldest-first window of shards instead of taking the first row. Hardware converts the node's own benchmark into mCU/s through the same reference constant that prices the work, and refuses a shard that cannot finish inside its lease. Network amortises the measured round trip against the size of the shard. Reliability limits how much an unproven node is exposed to at once, using the audit history the ledger already keeps. Locality prefers a node that already holds the job's neighbouring shards or the model manifests it needs. A shard that has waited past the starvation window outranks every fresh candidate by construction.

A scheduler is not trusted with any of this. A score decides who is *offered* a
shard and nothing else: a scheduler that scores badly, or maliciously, wastes
effort. It cannot overpay, underpay, or pay twice, because payment is settled
by the quorum against evidence the worker signed.

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

- Model layer partitioner. **Done.** `plan_parallelism` cuts a model into stages and `model-inspect` gives the byte span each stage needs; `model-shard` writes a file holding only those bytes.
- Pipeline stages. **Done.** `hocmesh-engine` executes one layer range; `stage-serve` puts it behind a port; a stage owns the KV cache for its own blocks and takes its position from the activation rather than counting locally.
- Activation transport. **Done.** Stages chain over HTTP with the hidden state encoded on every hop; the logits return down the chain. Proved by three processes holding disjoint shards producing output identical, to the SHA-256, to the whole model in one process.
- Topology-aware routing. **Partly.** Stage order is given on the command line. The resource graph and the round-trip scoring exist and choose *which* machines hold which shards; nothing yet reorders a chain to shorten it.
- Replicated critical layers. **Not done.**
- Node departure recovery. **Decided, not automated.** A stage that drops takes its KV cache with it; the answer is replay from the prompt on a replacement, and `stage/reset` is the mechanism. Nothing orchestrates the replacement yet.
- Micro-batching. **Not done.** One sequence at a time per chain.
- Parity against another implementation. **Done.** `reference_parity.rs` checks the forward pass against llama.cpp: identical tokens for the unsplit model and for a three-stage split, and bit-identical decoding for every quantised format. What remains of the gap is narrow -- it is held on a generated fixture rather than a downloaded checkpoint, and quantised *generation* is not compared because llama.cpp takes a different arithmetic path there (it quantises activations too), which the fixture's near-tied logits would turn into noise.
- GPU execution of a stage. **Not done.** The engine is CPU. GPU inference still goes through the llama.cpp adapters, which load whole models, so the distributed path and the accelerated path do not yet meet.

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
- RPM packaging. **Done.** `scripts/package-linux-rpm.sh` and an `rpm` target on the desktop bundle, both in the release job with checksums and signatures, alongside MSI, PKG, DEB and AppImage.

## North-star demonstration

Run an AI model that **none of the participating machines can run individually**, using only compute contributed by community nodes and credits previously earned through contribution.

**The mechanism is done and proved by test.** `crates/hocmesh-integration-tests/tests/distributed_inference.rs` runs three separate OS processes, each holding a shard with about 41% of the model's bytes — asserted by reading the shard, not by assuming it — chained together, generating output identical to the same model run whole, down to the SHA-256 of the logits. No process in that run holds enough of the file to answer on its own, and a stage pointed at a shard missing its layers refuses to start rather than reading zeros as weights.

**What that is not.** Three processes on one host is the protocol proved, not a deployment. The remaining Priority 1 item — a run across two or more real machines on a real network — still stands, and this does not close it: loopback says nothing about NIC drivers, MTU, NAT, asymmetric routes, or clocks that drift independently. The claim earned here is that the software can do it; the claim about the world needs the world.
