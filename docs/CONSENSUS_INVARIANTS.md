# hocMESH Consensus and Accounting Invariants

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

The only account allowed to supply newly issued bootstrap CU is `hocmesh:community:issuance`, and cumulative issuance may not exceed the validator-set policy limit.

## What a Compute Unit buys

Every workload prices against one constant, `REFERENCE_OPS_PER_MCU`, so a mCU
means the same machine work whichever workload earned it. The price comes from
the *spec*, never from elapsed time: a slow machine must not earn more for
identical work, and nobody can prove how long they spent.

That rule is only true if the op model matches the work. Prime shards were
first rated at a flat one operation per candidate, which is right at 10^4 and
wrong by 33x at 10^8, because trial division gets more expensive as the numbers
grow. A unit that drifts with its input is not a unit, and the drift is an
arbitrage: pick the range where the mCU is cheapest to earn, spend it where the
mCU is dearest to buy.

A prime candidate near `n` now costs `2 + isqrt(n) / (3 * ln n)` divisions -
composites fall out on the first two checks, primes run to the square root in
steps of six, and about one candidate in `ln n` is prime. Measured against the
divisions the code actually performs, that model holds within 25% from 10^4 to
10^8, where the flat rate drifted 33x over the same span.

The model is integer-only on purpose. A price has to be reproduced exactly by
every validator, and floating point makes no such promise across machines.


## Exactly-once claims

- Both user and community reservations claim `reserve:<job_id>`.
- Provider settlements claim `reward:<assignment_id>`.
- Escrow refunds claim `reward:<assignment_id>` too, so a shard settles in exactly one direction.
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
- escrow has enough CU;
- the reward arrives no later than `reserved_at + SETTLEMENT_WINDOW_SECS`.

## Escrow refunds

Escrow that can only ever pay out is a one-way valve: a shard whose provider
cheats, crashes, or simply never answers takes the CU that funded it with it,
and catching a cheat never becomes a settlement. `JobRefund` is the other
direction, and it is safe because of three rules.

**One claim key.** A refund claims `reward:<assignment_id>` - the same key the
reward claims. Exactly-once is therefore enforced by the claims table rather
than by a rule anyone has to remember: a shard settles once, in one direction,
and no reconciliation pass can double-spend it.

**The escrow returns where it came from.** A member-funded shard refunds only
to the node whose signature reserved it, and only against that node's
signature over the refund body. A community-funded shard has no requester at
all: its CU was minted against `COMMUNITY_ISSUANCE_ACCOUNT` and it is unminted
back to that account. Letting a node claim minted escrow would make "reserve
community work, let it fail, keep the CU" a free mint, so the protocol refuses
a community refund that carries any requester authorisation at all.

**Reward and refund never overlap in time.** A reward is valid only at or
before `reserved_at + SETTLEMENT_WINDOW_SECS`; a refund only strictly after.
The two windows are disjoint, so a provider and a requester can never race for
the same escrow, and a provider that misses the deadline it accepted cannot
outbid the requester who is reclaiming it.

`SETTLEMENT_WINDOW_SECS` is `DEFAULT_LEASE_SECONDS * 4` (3600s), and it is
measured from the `created_at` of the certified reservation - never from a
coordinator lease, which no validator has any reason to trust.

A refund is otherwise validated exactly like a reward: same shard-split check
against the certified root reservation, same funding-type check, same
deterministic amount, and the same two-posting shape.

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
