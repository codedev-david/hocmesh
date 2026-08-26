# hocMESH Quorum Ledger

## Goal

The hocMESH ledger exists to prevent any single scheduler, administrator, database, or participant from being able to manufacture or erase Compute Units undetectably.

It is a replicated accounting log, not a cryptocurrency.

## Core accounting invariant

Every ledger transaction contains postings whose sum is exactly zero.

```text
Σ posting.delta_mcu = 0
```

Paid job reservation:

```text
requester        -30 CU
job escrow       +30 CU
```

Provider settlement:

```text
job escrow        -8 CU
provider           +8 CU
```

Community bootstrap reservation:

```text
community issuance   -100 CU
community job escrow +100 CU
```

The issuance account is the only account allowed to become negative and its magnitude is bounded by `community_issuance_limit_mcu` in the pinned validator set.

## Why escrow exists

Without escrow a scheduler could accept a job, allow providers to work, then discover that the requester spent the same CU elsewhere.

Reservation first makes the available balance unavailable before work begins.

## Hash chain

Each entry contains:

```text
sequence
previous_hash
transactions
transactions_hash
entry_hash
```

An entry carries a *batch* of transactions, not one. A consensus round costs
the same three network phases whether it settles one CU movement or five
hundred, and entries chain by `previous_hash` so rounds are inherently
sequential: batching is what stops the ledger being capped at one settlement
per round. Sixteen concurrent settlements measured against four validators
take two rounds.

Conceptually:

```text
GENESIS
   │
   ▼
Entry 1 hash A
   │ previous=A
   ▼
Entry 2 hash B
   │ previous=B
   ▼
Entry 3 hash C
```

Changing an old transaction changes its transaction hash, its entry's transactions hash and entry hash, and every later previous-hash relationship.

## Quorum certificate

A coordinator proposes one exact batch of transactions to all validators.

Each validator, for every transaction in the batch:

1. verifies policy,
2. verifies signatures/evidence,
3. verifies account balances against a running overlay, so two transactions
   that each pass alone cannot jointly overdraw an account,
4. checks duplicate claims, both against history and within the batch,
5. checks its local ledger head,
6. persists a one-vote-per-height lock,
7. signs the proposed entry hash.

Once the threshold is reached, those signatures form a `QuorumCertificate`.

The certificate is then committed to validator replicas.

## Persistent vote lock

A validator stores:

```text
(sequence, entry_hash)
```

before returning its signature.

It will return the same vote for an identical proposal, but it refuses to sign a different entry hash at the same sequence.

That lock survives process restart because it is stored in SQLite.

## Settlement claims

Validators also keep unique claims.

Examples:

```text
reserve:<job_id>
reserve:<job_id>
reward:<assignment_id>
```

Therefore the same assignment cannot be paid twice even if a signed result is replayed.

## Requester authorization

A normal job reservation includes the original requester Ed25519 proof.

The job ID is deterministically derived from the signed request nonce.

Validators independently recompute:

- request body hash,
- requester signature,
- deterministic job cost,
- expected escrow postings.

## Provider authorization

A provider signs all material result metadata:

```text
assignment_id
job_id
shard_index
exact WorkSpec
reward_mcu
system_funded flag
WorkResult
```

A compromised coordinator therefore cannot take a signed answer and silently change the job, reward, shard, or workload represented by that signature.

## Reward-to-reservation binding

For paid and community jobs, validators retrieve the certified reservation from their own ledger.

They split the reserved root workload deterministically and verify that:

```text
provider WorkSpec == reserved shards[provider shard_index]
```

They also verify the deterministic assignment ID.

## Independent result verification

For the current `PrimeCount` workload, validators recompute the result.

This is intentionally straightforward rather than efficient.

Future workloads should use appropriate mechanisms such as:

- deterministic recomputation,
- random challenge/spot checks,
- redundant execution,
- commitment proofs,
- model-specific verification,
- trusted execution evidence where useful.

## Validator storage

Each validator SQLite database stores:

```text
certificates
balances
claims
votes
```

`balances` is derived/cache state. The certificate log is the authoritative history.

## Client mirroring

Any ordinary participant can execute:

```bash
hocmesh ledger-sync --validators validators.json --db .hocmesh/ledger-mirror.db
```

The client downloads quorum-certified entries and applies them locally.

It can then run:

```bash
hocmesh ledger-audit --validators validators.json --db .hocmesh/ledger-mirror.db
```

This means full replicas are not restricted to validator operators.

## Checkpoints and pruning

An audit that always replays from genesis costs more every day the network
runs. A checkpoint is a quorum-signed statement about the whole ledger state at
one height:

```bash
hocmesh ledger-checkpoint --validators validators.json --db .hocmesh/ledger-mirror.db
```

Each validator answers `GET /v1/ledger/state` with its own head, a digest of
the state it holds, and a signature over exactly the message a checkpoint is
verified against. Enough validators agreeing on the same
`(sequence, entry_hash, state_hash)` triple *is* the checkpoint.

A checkpoint is only stored after the local store proves it can reproduce it:
the store rewinds its own tables back to that height by undoing the entries
above it, and the resulting digest must equal the one the quorum signed. A
checkpoint the store disagrees with is refused rather than kept.

Because rewinding needs only the entries *above* the checkpoint, everything
below it can be discarded:

```bash
hocmesh ledger-prune --validators validators.json --db .hocmesh/ledger-mirror.db
```

`ledger-audit` then starts from the newest stored checkpoint by default, so its
cost tracks how much has happened since rather than the whole history. Pass
`--full` to force a genesis replay; on a pruned mirror that fails outright
rather than quietly auditing a shortened history.

Pruning keeps the checkpoint's own entry, so a pruned node still reports the
height it is really at, and it keeps `account_activity`: earned and spent
totals are part of the balance proofs validators have to agree on, and a node
whose totals had been reset would disagree with every node that had not pruned.

## What makes history difficult to fake

An attacker attempting to rewrite history must contend with:

- content hashes,
- previous-hash links,
- independent validator signatures,
- a quorum threshold,
- persistent double-vote locks,
- independent full replicas,
- participant mirrors,
- deterministic replay,
- duplicate claim rules,
- CU conservation.

A single compromised coordinator database is not sufficient in quorum mode.

## Validator set changes

The set that certifies entries is itself recorded in the chain. A
`MembershipChange` transaction carries the joining or departing member, the
threshold the set takes on afterwards, individually signed vouches from sitting
validators, and the hash of the set it produces.

Both the sponsor and the committer re-derive that hash rather than trusting the
one supplied, so evidence can never claim one set and produce another. The
change moves no CU, and is rejected if it tries to.

Validators persist the resulting set alongside the entry that certified it, in
a `validator_set(sequence, set_json)` table that survives pruning. `set_at()`
answers what the seats were at a height; `current_set()` answers what they are
now. A node that has been running is bound by what the quorum certified, not by
whatever is still sitting in its bootstrap file.

See `docs/SECURITY.md`, *Validator set membership*, for the vouch message, the
threshold rationale, and the operator commands.

## Consensus boundary

This implementation is intentionally a practical v0.2 quorum-certified linear log. It is not yet a complete production BFT state machine with leader election, view changes, membership epochs, and automated fork resolution.

Before hostile Internet-scale deployment, the recommended next step is to either:

1. implement a formally specified BFT consensus protocol around these transaction rules, or
2. embed a mature consensus library/protocol while keeping hocMESH's ledger transaction validation logic.

Do not replace the current voting rules with an ad-hoc "majority wins" implementation that permits double voting.

## Coordinator crash recovery

Before quorum submission, the coordinator persists a `ledger_intents` record containing the exact serialized transaction and leaves affected work in a non-runnable state (`funding` or `settling`).

After a restart, `hocmesh-coordinator recover` or automatic startup recovery asks validators for a signed quorum claim proof. A certified claim is finalized locally; an absent claim causes the coordinator to retry the exact same transaction. This avoids creating a second debit/reward after an ambiguous network failure.

The validator proposal client also serializes proposals within one process to reduce accidental same-height races. This does not replace a full BFT leader/view-change protocol for multiple independent proposers.

## Requester cannot pay itself

For member-funded jobs, the certified reservation records the requester identity. Provider-reward validation rejects a reward when `provider_node_id == requester_node_id`. The same invariant is replayed by offline client audit, so a malicious scheduler cannot bypass it without producing an invalid ledger.
