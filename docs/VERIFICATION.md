# Verification

MESH pays providers in CU for work the network cannot re-do cheaply. The whole
economy therefore rests on one question: **how does anyone know a submitted
result is real?**

Until v0.4 the answer was "recompute it". The coordinator recomputed every
shard before settling it, and every ledger validator recomputed it again while
replaying the block. With `V` validators the mesh burned `V + 2` times the
compute it delivered, and the harder a job was, the worse the multiplier hurt.
That is not a performance bug; it is a design that cannot scale past a toy.

## Three tiers

Verification now escalates, and almost never leaves the cheapest tier.

| Tier | Cost | When |
| --- | --- | --- |
| **Witness** | sublinear in the work | every settled shard, and every ledger entry a validator replays |
| **Replication** | 2x | shards a job marks as high-value |
| **Recompute** | 1x per adjudicator | only to settle a dispute between the tiers above |

The witness tier is what changed. A witness is a workload-specific check that
costs asymptotically less than the work it validates, so a validator can check
*every* entry it replays and still spend a fraction of what recomputing one
would have cost. Sampling was never needed; cheap checking was.

## Witnesses

### Matrix products - Freivalds' test

To check `C = A x B` without computing `A x B`, draw a random vector `r` over
`GF(2^31 - 1)` and test `C r == A (B r)`. Both sides are matrix-vector products:
`O(n^2)` against the `O(n^3)` the worker paid. A wrong `C` survives one round
with probability at most `1 / MODULUS`, about `5e-10`.

Operands are never transmitted. Both matrices are generated from a 64-bit seed,
so a job spec of a few dozen bytes describes billions of operations - which is
what makes shipping the work worth the round trip at all.

### Prime counts - bucketed audit

Prime counting has no algebraic shortcut: the only way to confirm a count is to
count again. So the shard is split into `BUCKETS = 64` contiguous ranges and the
result carries a count per bucket. Two checks follow:

1. The claimed total must equal the sum of the claimed buckets. Free.
2. `AUDIT_BUCKETS = 3` buckets, chosen at random, are recounted. 4.7% of the work.

A worker that skips `m` of the 64 buckets and guesses them escapes only if all
three audited buckets fall outside the `m` it faked, with probability
`C(64 - m, 3) / C(64, 3)`. Skipping a single bucket - the smallest cheat the
scheme can express - is caught 4.7% of the time; skipping half is caught 86%.

## The nonce must come after the commitment

Both witnesses need a challenge the worker could not predict: the vector `r` for
Freivalds, the bucket choice for prime counts. If the challenge were derived
from the result - the obvious, tempting design - a lazy worker could grind it.

Compute 48 of 64 buckets honestly, guess 16, then vary the guesses until the
derived challenge happens to select three buckets you really did compute. At
`C(64,3) / C(48,3)` that is about 74 attempts - minutes of work to steal a
quarter of the shard, forever.

So the nonce is drawn by the coordinator *after* the signed result arrives, and
recorded in the reward evidence so every validator replays the same audit.

**Known gap.** A coordinator colluding with a worker can still grind its own
nonce. Closing that needs a beacon neither side controls - a VRF over the block
hash, or a threshold signature from the validator set. Until then, collusion
between a coordinator and a provider is outside the threat model this closes.

## Audit rate and the economics

Auditing every shard at 4.7% is affordable, but a node with a long clean record
does not need it. The audit rate decays from 1.0 toward a floor of 0.05, halving
the gap every 12 clean results; one rejection resets the streak to zero.

That is only safe while cheating stays EV-negative:

    EV(cheat) = (1 - d) * gain - d * slash

where `d` is the audit rate times the chance the audit catches it. The trust
discount lowers `d`, which raises the slash needed - but the same clean history
that earned the discount also banked the CU that pays it. The margin between
banked balance and required slash *widens* with tenure: about 3x at zero
history, 758x after 5,000 accepted shards.

The collateral is real because **CU cannot be purchased**. A balance is proof of
work already performed, so slashing it destroys something money cannot replace.

## Reproducing the numbers

    cargo run --release -p mesh-core --example verification_proof

Every figure in this document is printed by that example, measured on the
machine that runs it. It asserts its own claims, so it fails rather than lies.

Measured on a Windows 11 laptop, release build:

| workload | compute | witness | cheaper by |
| --- | --- | --- | --- |
| prime count, 3M range | 159.0 ms | 7.08 ms | 22x |
| matrix product, 512-dim, 64 rows | 25.0 ms | 1.11 ms | 23x |

Per accepted shard with three validators, total network cost falls from 5.00x
the work delivered to 1.14x - a 4.4x cut in waste that grows with `V`.
