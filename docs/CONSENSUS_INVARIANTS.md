# MESH Consensus and Accounting Invariants

These rules are security boundaries, not implementation suggestions. Changes that weaken them require an explicit protocol-version change and threat-model review.

## Validator membership

- A validator set is pinned by the exact serialized membership configuration hash.
- The signature threshold must be **strictly greater than two thirds** of configured members.
- Validator IDs, URLs, and public keys must be unique.
- The current static-membership implementation assumes no more Byzantine validators than the threshold geometry allows; for the example 3-of-4 set, the intended fault bound is **one Byzantine validator**.

## Linear history

- Sequence 0 is `GENESIS`.
- Entry `N` must have sequence `N` and `previous_hash == entry[N-1].entry_hash`.
- A validator persists a one-vote-per-height lock before returning a signature.
- Validator entry signatures bind both the membership hash and entry hash.
- A quorum certificate must contain the configured threshold of distinct, valid validator signatures.

## Compute Unit conservation

Every normal ledger transaction must satisfy:

```text
sum(posting.delta_mcu) == 0
```

A user reservation transfers CU from requester to a job escrow. A provider reward transfers CU from that same escrow to the provider. No ordinary job creates CU.

The only account allowed to supply newly issued bootstrap CU is `mesh:community:issuance`, and cumulative issuance may not exceed the validator-set policy limit.

## Exactly-once claims

- Both user and community reservations claim `reserve:<job_id>`.
- Provider settlements claim `reward:<assignment_id>`.
- A claim can appear only once in certified history.
- Job IDs are bound to requester-signed nonces for user jobs.
- Assignment IDs are deterministic from `(job_id, shard_index)`.

## Provider settlement

A provider reward is valid only when:

- the provider signed the exact assignment ID, job ID, shard index, WorkSpec, declared reward, funding type, and WorkResult;
- the result independently verifies for the declarative workload;
- reward equals the deterministic cost of that exact shard;
- the shard is exactly the deterministic split of a previously certified root reservation;
- its funding type matches that reservation;
- for member-funded jobs, provider != requester;
- escrow has enough CU.

## Replay rules

Live coordinator requests enforce timestamp skew and nonce replay prevention. Historical ledger verification checks the cryptographic signature without applying current wall-clock skew; otherwise an old but valid certified history could not be audited later.

## Crash recovery

Before requesting quorum certification, the coordinator persists the exact transaction in `ledger_intents` and puts affected scheduler objects into a blocked state. Recovery must either:

1. prove an existing settlement by validating a returned quorum certificate, or
2. establish quorum agreement that the claim is absent and retry the exact persisted transaction.

Recovery must never synthesize a replacement transaction with different signed metadata.

## Participant audit

A participant full-ledger audit must independently replay:

- membership-bound quorum certificates,
- chain linkage,
- CU conservation,
- nonnegative user/escrow balances,
- issuance cap,
- duplicate claims,
- requester/provider signatures,
- deterministic workload results,
- reward-to-reservation binding,
- requester self-reward prohibition.

A coordinator UI balance is never the source of truth in quorum mode.

## Current consensus limitation

This v0.2 implementation is a quorum-certified replicated log with persistent vote locks. It is not a complete BFT consensus protocol with leader election, view changes, membership epochs, or formally proven liveness under competing proposers. Do not market or deploy it as Byzantine-fault-tolerant production consensus until that milestone is completed and reviewed.
