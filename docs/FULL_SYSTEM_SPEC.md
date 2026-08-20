# MESH Compute / MESH AI — Full System Specification

**MESH = Mutual Exchange of Shared Hardware**

Version: **0.2.0 source architecture**

MESH is a contribution-first cooperative compute network. Participants make idle compute available to the community, earn non-monetary Compute Units (CU) for verified useful work, bank those units, and later spend them on work performed by other participants.

There is no cash-out, token, cryptocurrency, bidding market, purchasable balance, or paid priority tier in the protocol described here.

> **Contribute first. Compute later.**

The long-term objective is a geographically distributed community data center capable of combining heterogeneous CPUs, GPUs, memory, storage, and network paths for AI and general compute. The current source implements the control-plane and accounting foundation with deterministic CPU work; MESH AI GPU/model execution remains the next major layer.

---

## 1. System goals

MESH is designed around these goals:

1. **Contribution first.** A new identity begins with zero CU.
2. **No purchased compute.** CU represents verified community contribution, not money.
3. **No arbitrary remote access.** A participant does not grant SSH/RDP or execute arbitrary user-supplied binaries.
4. **Internet-friendly workers.** Ordinary nodes use outbound connections and can operate behind home NAT/firewalls.
5. **Parallel work.** Jobs are divided into deterministic shards and distributed among available workers.
6. **Independent accounting.** In quorum mode, the scheduler is not authoritative for balances.
7. **Tamper-evident history.** CU history is hash-linked, signed, replicated, and independently auditable.
8. **Exactly-once settlement.** Job reservations and shard rewards have unique ledger claims.
9. **Failure is normal.** Leases, retries, validator sync, and coordinator settlement recovery account for machines going offline.
10. **Extensible runtime.** The control plane is intentionally separate from the future GPU/model runtime.

---

## 2. Current executable roles

The repository builds three Rust programs.

### `mesh`

Participant client and worker.

Responsibilities:

- create/load the participant Ed25519 identity;
- discover CPU/RAM/GPU capabilities;
- register with a coordinator;
- send heartbeats;
- pull declarative work;
- execute supported work locally;
- sign complete result metadata;
- submit verified results;
- submit jobs after earning CU;
- query balances/status;
- directly query the validator set;
- mirror the full certified ledger;
- audit a mirrored ledger from genesis.

### `mesh-coordinator`

Scheduler and control plane.

Responsibilities:

- maintain the node/capability registry;
- maintain the shard queue;
- create deterministic job shards;
- lease work to workers;
- exclude the requester from its own paid shards;
- verify returned deterministic work;
- create ledger reservation/reward transactions;
- persist settlement intents before contacting validators;
- reconcile ambiguous ledger outcomes;
- expose node/job/network APIs.

The coordinator is **not** the authoritative CU database when quorum mode is enabled.

### `mesh-validator`

Independent replicated CU ledger authority.

Responsibilities:

- maintain a complete certified ledger replica;
- independently validate transaction invariants;
- verify participant signatures;
- recompute the current deterministic work type;
- bind rewards to certified job reservations;
- reject requester self-rewards;
- enforce CU conservation and issuance limits;
- enforce unique settlement claims;
- persist one-vote-per-height locks;
- sign proposed ledger entries;
- store quorum certificates;
- provide signed head, balance, and claim proofs;
- synchronize missing certified history from peers;
- audit its own history from genesis.

---

## 3. Current topology

```text
                           ┌──────────────────────┐
                           │   mesh-coordinator   │
                           │ scheduler / leases   │
                           └──────────┬───────────┘
                                      │ outbound worker HTTP(S)
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
                 mesh A            mesh B            mesh C
                 CPU/GPU           CPU/GPU           CPU/GPU

                                      │ ledger proposals / proofs
                                      ▼
               ┌─────────────────────────────────────────────┐
               │           independent validator set          │
               │                                             │
               │ V1          V2          V3          V4      │
               │ full DB     full DB     full DB     full DB │
               └─────────────────────────────────────────────┘

                 Any participant may also mirror/audit ledger
```

The recommended example validator set is 4 validators with a threshold of 3. The code requires the threshold to be strictly greater than two thirds of membership.

This is intended to tolerate at most one Byzantine validator in the 3-of-4 configuration. It is not yet a complete production BFT consensus implementation with view changes and dynamic membership.

---

## 4. How clients find distributed resources

Ordinary workers do not currently discover one another directly.

They register capabilities with one coordinator and repeatedly poll for assignments:

```text
worker ── register/capabilities ─► coordinator
worker ── heartbeat ─────────────► coordinator
worker ── poll ──────────────────► coordinator
worker ◄─ shard assignment ─────── coordinator
worker ── signed result ─────────► coordinator
```

This is deliberate for the first Internet-capable architecture:

- no inbound worker port;
- works behind NAT;
- small attack surface;
- scheduler can reason about leases and availability;
- worker does not expose a general remote execution service.

The end-state data plane should add direct peer paths for model chunks and selected compute traffic using authenticated QUIC/libp2p-style networking, while retaining a federated scheduling/control plane.

---

## 5. Identity and request authentication

Each installation creates an Ed25519 keypair locally.

The private key remains on the participant machine. The public key derives the node identity and is registered with the coordinator.

Protocol v2 signed live requests include:

```text
protocol domain
operation/action
node ID
timestamp
cryptographic nonce
hash of the complete request body
```

Live coordinator authentication checks:

- public-key signature;
- identity binding;
- timestamp skew;
- nonce replay cache.

The nonce also makes a user job ID deterministic from the exact signed submit authorization. Replaying the same signed submission therefore maps to the same unique reservation claim.

Historical ledger audit deliberately verifies the cryptographic signature **without** requiring the old timestamp to be close to the current wall clock. Otherwise certified history would become unauditable after the live authentication window expires.

---

## 6. Safe workload model

MESH v0.2 does not accept arbitrary executables from requesters.

Jobs are declarative allow-listed `WorkSpec` values. The implemented workload is deterministic prime counting over a numeric range.

This workload exists because it provides a real end-to-end primitive for testing:

- deterministic sharding;
- parallel CPU execution;
- independent verification;
- deterministic reward calculation;
- retry after worker loss;
- accounting and settlement.

The runtime boundary is intentionally replaceable. Future MESH AI work types should be explicit protocol objects backed by trusted local runtimes, not unrestricted shell commands.

---

## 7. Sharding and task parallelism

A root work specification is deterministically split into N shards.

Example:

```text
root range
2 .. 100,000,000

             deterministic split
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      shard 0      shard 1      shard N
        │            │            │
      worker A     worker B     worker C
```

Assignment IDs are deterministic from:

```text
(job_id, shard_index)
```

The coordinator leases one pending shard to a worker for a bounded period. Expired leases return to the pending queue.

Member-funded shards are not offered to their requester. Validators independently enforce the same rule during settlement.

---

## 8. Compute Units

MESH stores CU internally as milli-compute-units (mCU):

```text
1000 mCU = 1 CU
```

The current deterministic prime workload uses a work-size cost rather than elapsed wall time. This prevents a slow machine from earning more merely because it took longer.

Future GPU accounting should be based on benchmarked useful work classes and normalized resource factors, not self-reported device names.

---

## 9. Contribution-first rule

A participant starts at:

```text
0 CU
```

A normal job reservation is rejected if the validator quorum says the participant has insufficient available balance.

There are no starter credits and no API to buy CU.

The bootstrap problem is solved by explicitly marked **community-funded work**. The validator-set policy permits a bounded amount of community issuance to fund useful bootstrap jobs. CU reaches a newcomer only after the newcomer performs and proves real work.

---

## 10. Replicated ledger architecture

In quorum mode, authoritative credit is not an editable balance row in the coordinator database.

Each validator maintains the same append-only certified log.

A ledger transaction has:

- transaction ID;
- transaction kind;
- balanced postings;
- evidence;
- creation timestamp.

Each ledger entry has:

- sequence number;
- previous entry hash;
- transaction;
- transaction hash;
- entry hash.

Conceptually:

```text
GENESIS
   │
   ▼
entry 1 ─hash─► A
                 │
entry 2 previous=A ─hash─► B
                           │
entry 3 previous=B ─hash─► C
```

Changing old history breaks later hash links and quorum signatures.

---

## 11. CU conservation and escrow

Every ordinary transaction must satisfy:

```text
Σ posting.delta_mcu = 0
```

A user job reservation:

```text
requester         -30 CU
job escrow        +30 CU
------------------------
net                 0 CU
```

A provider reward:

```text
job escrow         -8 CU
provider           +8 CU
------------------------
net                 0 CU
```

Therefore ordinary compute transfers contribution entitlement; it does not manufacture it.

The only issuance source is:

```text
mesh:community:issuance
```

Its maximum negative balance is bounded by `community_issuance_limit_mcu` in the pinned validator-set policy.

---

## 12. Validator verification

Before voting for a reservation or reward, validators independently check the transaction.

### User reservation

Validators verify:

- requester signature;
- request body hash;
- job ID derived from signed nonce;
- valid WorkSpec;
- shard count;
- deterministic total cost;
- requester has enough CU;
- exact requester-to-escrow postings;
- reservation claim is unused.

### Community reservation

Validators verify:

- valid WorkSpec;
- shard count;
- deterministic total cost;
- exact issuance-to-escrow postings;
- issuance cap;
- reservation claim is unused.

### Provider reward

Validators verify:

- provider signature over all material result metadata;
- deterministic assignment ID;
- deterministic reward amount;
- result correctness;
- a certified root reservation exists;
- shard index belongs to that reservation;
- exact WorkSpec equals the deterministic shard of that reservation;
- funding type matches the reservation;
- provider is not the requester for a paid job;
- escrow has sufficient CU;
- reward claim is unused.

The coordinator alone therefore cannot create a valid provider reward by editing its SQLite database.

---

## 13. Settlement claims

The ledger has unique semantic claim keys:

```text
reserve:<job_id>
reward:<assignment_id>
```

Both user-funded and community-funded reservations use the same `reserve:<job_id>` namespace, preventing two reservation types from claiming the same job ID.

A certified claim can appear only once.

---

## 14. Quorum certification

A proposer reads a quorum-agreed ledger head and constructs the exact next entry.

Each validator that accepts the proposal persists:

```text
(sequence, entry_hash)
```

before signing it.

It will not sign a conflicting entry at the same sequence, even after restart.

Validator signatures bind:

```text
membership_hash + entry_hash
```

Once the configured threshold signs the same entry, those signatures form a portable `QuorumCertificate`.

Any party with the validator membership file can verify that certificate independently.

---

## 15. Signed state proofs

Validators expose signed proofs for:

- current ledger head;
- account balance/activity at a head;
- settlement claim status.

The participant CLI can query validators directly instead of trusting what the coordinator says its balance is.

A claim response may include the complete quorum certificate. One valid certificate is enough to prove that the validator threshold already certified an entry, even when only one surviving replica currently stores that certificate.

---

## 16. Full participant ledger replicas

Any participant can maintain the same full certified history as validators:

```bash
mesh ledger-sync --validators validators.json --db .mesh/ledger-mirror.db
```

Then audit it offline:

```bash
mesh ledger-audit --validators validators.json --db .mesh/ledger-mirror.db
```

The audit replays history from genesis and checks:

- certificate signatures;
- membership hash;
- sequence and previous-hash chain;
- transaction signatures;
- CU conservation;
- balance non-negativity;
- issuance cap;
- duplicate claims;
- deterministic reward structure;
- reward-to-reservation binding;
- requester self-reward prohibition;
- deterministic work results.

This is the Git-like property of MESH: history is content-linked, replicated, and independently checkable. Unlike Git, MESH additionally has a quorum rule to decide which next entry is certified.

---

## 17. Crash-safe coordinator settlement

The coordinator and validator quorum cannot share one ACID database transaction. MESH therefore uses durable settlement intents.

### Job funding

Before contacting validators:

- job row is stored as `funding`;
- shards are stored as `blocked`;
- exact serialized ledger transaction is stored in `ledger_intents`.

Only after certification:

- intent becomes `certified`;
- job becomes `pending`;
- shards become `pending`.

### Provider settlement

Before contacting validators:

- assignment becomes `settling`;
- result is persisted;
- exact reward ledger transaction is persisted.

Only after certification:

- intent becomes `certified`;
- assignment becomes `completed`;
- the job becomes completed if all shards have settled.

### Recovery

Recovery runs:

- once when a quorum-mode coordinator starts;
- periodically while it runs;
- manually with `mesh-coordinator recover`.

Recovery first asks validators whether the semantic claim already has a certificate. If a full valid quorum certificate is returned, local state can safely finalize. If a quorum agrees that the claim is absent, the coordinator retries the **same persisted transaction**.

This protects against double charging or duplicate rewards after ambiguous network/process failures.

---

## 18. Worker failure handling

A worker assignment is leased for a bounded duration.

If a worker disappears before accepted settlement:

```text
leased → lease expires → pending → another worker
```

A result that has entered quorum settlement changes to `settling`, which is not eligible for re-lease while accounting is reconciled.

This is important: once there is ambiguity about whether a reward was certified, MESH resolves the ledger outcome before allowing another provider to perform the same settlement claim.

---

## 19. Data stored on the coordinator

Coordinator SQLite stores operational scheduling state:

```text
nodes
balances              # cache / local-MVP authoritative only
ledger                # local-MVP history only
jobs
assignments
auth_nonces
ledger_intents
```

In quorum mode, `balances` is only a convenience cache for UI/network statistics. The quorum-certified validator history is authoritative.

---

## 20. Data stored on validators

Each validator SQLite replica stores:

```text
certificates
balances       # derived cache
claims
votes
```

The certificate log is the authoritative local history. Balances can be independently reconstructed by replay.

---

## 21. Security boundary

MESH intentionally does not provide:

- SSH access to provider machines;
- remote desktop access;
- requester-supplied shell commands;
- arbitrary requester binaries;
- administrator/root access;
- unrestricted provider filesystem access.

Future runtime integrations must retain the declarative/sandboxed execution model.

Public deployments should place coordinator and validator HTTP services behind authenticated TLS-capable reverse proxies until TLS configuration is directly integrated.

---

## 22. Current consensus limitation

The v0.2 validator implementation provides a quorum-certified linear log, persistent vote locks, full replicas, proof verification, and recovery primitives.

It is **not yet a complete formal BFT state machine**.

Major remaining consensus work includes:

- leader election / proposer ownership;
- view changes when a proposal partially locks a height;
- multi-coordinator contention handling;
- membership epochs and safe validator rotation;
- snapshot/state transfer protocol;
- formal fault/liveness specification;
- adversarial integration/fuzz testing.

A production public deployment should complete or adopt a mature BFT consensus layer before claiming Byzantine-fault-tolerant ledger operation.

---

## 23. MESH AI end-state architecture

MESH AI sits above MESH Compute Core.

```text
                    MESH AI
      model registry / partitioner / runtime
                       │
                       ▼
                 MESH Compute
 identity / scheduling / accounting / trust / network
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      NVIDIA           AMD          Apple
       CUDA            ROCm          Metal
```

The intended AI execution modes are introduced in increasing difficulty:

1. independent GPU batch tasks;
2. batched LLM inference where prompts are distributed among full-model replicas;
3. content-addressed model block distribution;
4. model/pipeline parallelism across nearby nodes;
5. direct authenticated activation transport;
6. latency-aware dynamic virtual GPU clusters;
7. heterogeneous CUDA/ROCm/Metal placement;
8. selected tensor parallelism only where bandwidth/latency makes it practical.

WAN tensor parallelism should not be treated as the first implementation target because tightly synchronized collective operations are highly sensitive to latency and bandwidth.

---

## 24. Model distribution end state

Large immutable model artifacts should use content-addressed chunks:

```text
model manifest
   │
   ├── chunk hash A
   ├── chunk hash B
   ├── chunk hash C
   └── ...
```

Nodes cache chunks and can seed them to peers, creating the torrent-like data plane originally envisioned for MESH.

The scheduler should place work using both compute suitability and data locality so it does not repeatedly move hundreds of gigabytes when useful model blocks are already cached near a job.

---

## 25. Future scheduler graph

For tightly coupled AI work, candidate machines form a graph:

- vertices = compute nodes;
- vertex weights = benchmark, VRAM/RAM, runtime, reliability, cached data;
- edge weights = latency, bandwidth, packet loss.

The scheduler should choose temporary clusters that satisfy model capacity while minimizing expensive inter-node activation movement.

Independent work may span continents; pipeline/tensor stages should strongly prefer low-latency neighborhoods.

---

## 26. Repository ownership boundaries

```text
mesh-protocol
    wire format and signatures

mesh-core
    identity, hardware, deterministic compute

mesh-ledger
    accounting types, validation, storage, quorum client

mesh-node
    participant CLI and worker daemon

mesh-coordinator
    scheduling, leases, jobs, settlement intents

mesh-validator
    independent replicated ledger authority
```

Future suggested crates:

```text
mesh-runtime
mesh-gpu
mesh-model
mesh-p2p
mesh-ai
```

Avoid putting GPU/model implementation into the ledger or coordinator crates. The accounting/control plane should remain usable for other safe distributed workloads.

---

## 27. Build and binaries

The workspace is built with:

```bash
cargo build --release --workspace
```

Expected release programs:

```text
mesh
mesh-coordinator
mesh-validator
```

On Windows the binaries have `.exe` extensions.

See the root `README.md` for platform prerequisites, release scripts, user install helpers, complete local-mode walkthrough, and quorum-mode walkthrough.

---

## 28. Verification before development continuation

Because the source-generation environment for this handoff did not contain Cargo/Rust, the first Codex CLI action must be compiler-driven verification:

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Fix compiler/lint failures before adding features. Do not weaken consensus/accounting invariants merely to make a test pass.

Then run a four-validator 3-of-4 integration environment and verify:

- new user starts at 0 CU;
- community work is certified before becoming runnable;
- user earns CU only after verified provider settlement;
- paid job moves CU into escrow before work;
- requester cannot receive own paid shard;
- providers receive exactly the spent CU;
- duplicate reservation/reward claims fail;
- validator failure and rejoin sync works;
- coordinator crash at every settlement boundary recovers exactly once;
- participant full-ledger audit succeeds.

---

## 29. Production blockers

Before MESH should be exposed as a hostile public Internet network, complete at least:

- compiler/test/clippy clean baseline;
- extensive integration/failure-injection tests;
- mature BFT view-change/leader protocol or established consensus engine;
- validator membership epochs/rotation;
- TLS/mTLS deployment model;
- rate limiting and resource exhaustion protection;
- dependency auditing and supply-chain policy;
- fuzzing of protocol/ledger decoders;
- sandbox hardening for every new workload runtime;
- secure auto-update/code-signing strategy;
- observability without leaking participant workload data;
- GPU runtime isolation and benchmark attestation strategy;
- legal/privacy review for public workload execution.

---

## 30. Core project statement

MESH is intended to make a community of independent machines behave, where technically practical, like a shared distributed data center:

```text
CONTRIBUTE VERIFIED COMPUTE
            ↓
         EARN CU
            ↓
         BANK CU
            ↓
  REQUEST COMMUNITY COMPUTE
            ↓
      CU FLOWS TO PROVIDERS
            ↓
         CONTRIBUTE AGAIN
```

The authoritative resource being exchanged is **computation itself**, not money.

The long-term MESH AI objective is to add a secure, latency-aware heterogeneous GPU/model data plane on top of this contribution, identity, scheduling, verification, and replicated-accounting foundation.
