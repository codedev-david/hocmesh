# Verification

hocMESH pays providers in CU for work the network cannot re-do cheaply. The whole
economy therefore rests on one question: **how does anyone know a submitted
result is real?**

Until v0.4 the answer was "recompute it". The coordinator recomputed every
shard before settling it, and every ledger validator recomputed it again while
replaying the block. With `V` validators the hocmesh burned `V + 2` times the
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

### Neural network inference - float Freivalds

Everything above assumes exact arithmetic: two honest nodes produce identical
answers, so any difference is a cheat. Inference does not work that way. Float
addition is not associative, every GEMM kernel sums in its own order, and the
same weights on different hardware give slightly different logits. Bit-equality
would reject every honest node.

This matters more than the other two tiers put together: prime counting is a
demo, and inference is the workload people actually want to buy. If float work
cannot be witnessed, none of the economics above reaches it.

It can. Transformer inference is a stack of matrix products, so the same
Freivalds test applies - the only change is the comparison. Instead of asking
whether the residual is zero, ask whether it is small enough to be rounding:

    max |C r - A (B r)| / max |C r|  <=  TOLERANCE

That threshold is measured, not chosen. It has to sit in the gap between two
populations, and it does, by a wide margin:

    two honest kernels, different summation order    1.733e-7
    tolerance                                        1.000e-3
    stopped at 511 of 512 accumulation steps         6.609e-2

The tolerance sits about 5,800x above honest hardware drift and 66x below the
smallest cheat the workload can express - skipping one of 512 accumulation
steps, a 0.2% saving. Over 200 independent challenges that cheat was caught 200
times, and honest work was rejected zero times.

One round of Freivalds over the reals is weaker than one round over a large
prime field: a plus-or-minus-one challenge catches a wrong product with
probability at least 1/2, not 1 - 5e-10. So the witness runs `ROUNDS = 8`
independent challenges and requires all of them, which puts the escape
probability below 1 in 250 even for an adversary who knows the algorithm.

The eight challenges are stacked into one block so each matrix is read once
rather than eight times, and each row accumulates in registers. On a small
128x512x512 projection that lands at 5.5x cheaper than the product measured,
against 10.7x predicted from operation counts. The small shape is the
pessimistic case: the product grows with rows x inner x cols while the witness
grows only with the matrix areas, so the margin widens with model size - 51x
on a 512x4096x4096 attention batch, 224x on a 4096x4096x14336 MLP projection.

    cargo run --release -p hocmesh-core --example float_witness_proof

A witness needs the whole product, but a ledger entry cannot carry a 262 KB
matrix. The entry therefore carries a 64-byte SHA-256 commitment over the
IEEE-754 bit pattern of every element (domain tag `hocmesh-tensor-commit-v1`),
and the payload itself rides along with the answer the requester wanted anyway.
The holder of the payload hashes it and compares it against the committed digest
before spending a single multiply, so a provider that has already seen the
challenge cannot swap in a different matrix that satisfies it. That is 4096x
less ledger per shard, and it keeps the commitment-before-nonce rule below
intact for float work.

### Why a validator cannot just run the witness

Freivalds checks a product you already hold, and that is a narrower claim than
it first looks. A validator holds a ledger entry, not a 262 KB matrix. Worse,
a provider that never computed `C` can answer any Freivalds challenge about it
for free, because `C r` and `A (B r)` are the same vector and the second costs
`O(n^2)`. The test detects a wrong answer. It does not prove anyone did the
work. So the witness is the requester's tool - the requester holds the payload
and wants it checked cheaply - and the ledger needs something else.

That something else is the prime audit again, with row blocks in place of
buckets. The shard is committed one row block at a time, so the provider
publishes 64 digests before it knows anything about the challenge. The
authoritative nonce then names three blocks and the provider must reveal those
rows. A validator re-executes only those rows and compares them to the reveal
within the same tolerance, and the digest binds the reveal to what was
committed before the challenge existed.

A provider that skipped `m` blocks cannot answer for them, so it escapes only
when all three opened blocks land in the part it did compute - the same
hypergeometric rate the prime audit charges, which means float work and integer
work cost the same to cheat and compose with the beacon the same way:


| blocks skipped | escape measured | predicted |
|----------------|-----------------|-----------|
| 4/64           | 85.0%           | 82.1%     |
| 16/64          | 43.5%           | 41.5%     |
| 32/64          | 12.5%           | 11.9%     |
| 64/64          | 0.0%            | 0.0%      |

The reveal is 12,288 of 262,144 bytes - 4.7% of the shard - and the validator
re-executes exactly those rows, so it pays 4.7% of the job. Both numbers fall
out of `AUDIT_BUCKETS / BUCKETS`, unchanged from the integer path.

**Not yet wired.** `hocmesh-core::tensor` proves both halves work: the witness
for whoever holds the payload, the block audit for whoever does not. Neither is
reachable from a job yet. There is no `WorkSpec` for inference, so no inference
shard is priced, escrowed or paid through the ledger, and the protocol has no
reveal round: the block audit needs the provider to answer a challenge after
settlement is proposed, which is a message that does not exist today. Both are
engineering jobs rather than open questions, and that is the whole change in
their status.

## The nonce must come after the commitment

All three witnesses need a challenge the worker could not predict: the vector
`r` for Freivalds, the bucket choice for prime counts. If the challenge were
derived from the result - the obvious, tempting design - a lazy worker could
grind it.

Compute 48 of 64 buckets honestly, guess 16, then vary the guesses until the
derived challenge happens to select three buckets you really did compute. At
`C(64,3) / C(48,3)` that is about 74 attempts - minutes of work to steal a
quarter of the shard, forever.

So the challenge is never drawn by the party being checked, and never from the
result. Validators derive it themselves, twice, from things that already exist
on the ledger.

**Layer one - the propose-time challenge.** Before a validator votes for an
entry it derives a challenge from the chain head that entry builds on and the
transaction id:

    AuditNonce::for_entry(previous_hash, transaction_id)

The worker cannot predict it: it commits and signs its result before it knows
which head the entry will land on. The coordinator cannot choose it either
without moving the entry to a different position in the chain.

**Layer two - the apply-time beacon.** When the signed certificate comes back to
be applied, every validator derives a second, independent challenge from the
quorum signatures on it:

    AuditNonce::for_certified_entry(previous_hash, transaction_id, signatures)

This is the part a colluding coordinator cannot walk around. Producing a
signature needs a validator's key, so no amount of local computation tells the
coordinator what the beacon will say before the quorum has actually signed. To
draw again it must re-propose - and `lock_vote` refuses a conflicting entry hash
at a sequence a validator has already voted on. Every retry is therefore a
public, attributable ledger round, not a private hash loop.

The two challenges are independent draws over the same entry, so a fabricated
result has to escape both. `a_lazy_result_must_escape_both_challenges_to_settle`
measures it: the joint escape rate matches the product of the two single-layer
rates. `grinding_the_beacon_costs_more_public_rounds_the_more_a_node_skips`
prices the attack - the more work a node skips, the more rounds it has to run in
the open before one settles.

The nonce the coordinator drew for its own provisional check is still recorded,
as `provisional_audit_nonce`, but only as an audit trail of what it claims it
looked at. `a_coordinator_chosen_nonce_cannot_excuse_a_lazy_result` hand-picks
the luckiest nonce out of 400,000 - one that audits nothing but honest buckets -
stamps it on a fabricated entry, and shows the entry is still rejected.

**Remaining assumption.** The beacon is only as unpredictable as the quorum is
independent. A coordinator holding enough validator keys to sign an entry by
itself controls the beacon too - but that is the same threshold that already
lets it forge the ledger outright, so it is not a new weakness.

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

## Who loses when a cheat gets through

The audit rate above is not the only thing standing between a lazy provider and
a payout, and for most jobs it is not even the main thing. It matters who is out
of pocket.

A requester-funded shard moves CU from one account to another. Nothing is
created. If a provider cheats, the requester paid for an answer it did not get -
and the requester is the one party that holds the payload, wants it to be right,
and can check it for a few percent of what the job cost. That is a party with
both the means and the motive, which is the strongest verifier in the system.

A community-funded shard is different in kind. It mints CU against the issuance
limit, and there is no counterparty out of pocket to notice. A cheat there is
not theft from a requester; it is inflation. So the ledger applies a rule that
has nothing to do with probabilities:

> CU may only be issued for work a validator can audit from the ledger entry
> alone.

`WorkSpec::audit_class` states which side of that line a workload falls on, and
`validate.rs` refuses a community reservation or a system-funded reward for
anything that is not `SelfContained` - at propose time and again at apply time.

Both shipping workloads answer in a few dozen integers, so both qualify today
and nothing is restricted in practice. The rule exists for the workload that is
coming: a tensor shard answers in a matrix, which is `RevealRequired`, and it
would otherwise have walked straight into the issuance path the day it landed.
The match in `audit_class` is exhaustive, so it cannot be added silently.

## Reproducing the numbers

    cargo run --release -p hocmesh-core --example verification_proof

Every figure in this document is printed by that example, measured on the
machine that runs it. It asserts its own claims, so it fails rather than lies.

Measured on a Windows 11 laptop, release build:

| workload | compute | witness | cheaper by |
| --- | --- | --- | --- |
| prime count, 3M range | 159.0 ms | 7.08 ms | 22x |
| matrix product, 512-dim, 64 rows | 25.0 ms | 1.11 ms | 23x |

Per accepted shard with three validators, total network cost falls from 5.00x
the work delivered to 1.14x - a 4.4x cut in waste that grows with `V`.

Section 4 prices the collusion attack. Escaping one challenge is not escaping
settlement, and the joint rate tracks the product of the two:

    skipped   escape 1   escape 2   escape both   predicted   public rounds
      16/64      44.7%      47.4%         21.9%       21.2%            4.6
      32/64      13.7%      15.2%          1.9%        2.1%           51.3
      64/64       0.0%       0.1%          0.0%        0.0%       >= 2000

A node skipping half its work needs about 51 public re-proposals before one
settles, each needing a fresh quorum signature and each refused at any sequence
where a validator has already voted. That is the price of grinding when the
challenge comes from a beacon the grinder cannot compute.
