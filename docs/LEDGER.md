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

## Who is allowed to mint

Community issuance is the only place CU comes from nothing, so it is the only
place the ledger has to answer "says who?" rather than "does this balance?".

A `CommunityReserve` transaction carries `sponsors`: signatures from named
members of the sitting validator set over the job id, the workload, the shard
count and the price it comes to. A mint is valid only if at least `threshold`
distinct sitting members have signed it - the same k-of-n that admits a new
validator. Spending the shared budget is never cheaper than agreeing on who is
allowed to agree.

Sponsorships are produced by an operator on a validator machine with `hocmesh
community-vouch`, and carried - never created - by whoever assembles the
transaction. The coordinator holds no key that can mint.

The limit still applies. Sponsorship says the set chose to spend; the limit
says how much there was to spend. Neither replaces the other.

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

A validator stores, per height:

```text
(sequence, promised_ballot, accepted_ballot, accepted_entry, accepted_transactions)
```

before returning its signature. That lock survives process restart because it is stored in SQLite.

A ballot is a proposer's claim on one height, ordered by `(number, proposer)` so no two live claims compare equal. A validator refuses to promise or accept anything for a ballot older than the one it is holding.

## How a contested height is resolved

Nothing elects a leader. Any client may settle, so two of them can reach for the same sequence at the same instant.

A round therefore runs in two phases:

1. **Prepare.** The proposer asks every validator to hold the height for its ballot. A threshold of promises means no older ballot can still gather a certificate. Each promise reports whatever that validator has already accepted here.
2. **Propose.** If any promise carried an accepted entry, the proposer must drive *that* entry — the newest accepted ballot wins — rather than its own batch. Otherwise it proposes its own.

The second rule is what keeps the chain single-valued. An entry that some validators have already signed may be one vote short of a certificate that somebody else will finish, so it can never be quietly replaced — a later proposer either completes it or does nothing.

The cost is that a proposer which adopts somebody else's entry has not settled its own batch. It retries at the next height, backing off by an interval derived from its own identity so two clients that lost the same race do not collide again the same way.

Without this a split is terminal: half the set signs one entry and half signs another, neither reaches threshold, nothing is applied, and the height can never be filled by anything else. That failure is covered by `a_split_proposal_does_not_wedge_the_height`.

There is one more way to lose the race, and it is not a split. A proposer can
be promised a height, and then have that same height committed by somebody else
before its own votes come back. Every validator has moved on by then, so the
proposal collects nothing at all — not a minority, zero. Reading that as "the
set rejected this batch" is wrong and, worse, final: it fails a settlement that
nothing was ever wrong with.

So a round that falls short asks the chain what happened rather than reading it
out of a refusal message: if the head has advanced past the height it aimed at,
the round was overtaken and is retried on the new head, and only otherwise is it
reported as rejected. A round that fell short applied nothing anywhere, which is
what makes re-proposing the same batch safe.

That question can be asked a moment too early. Reading the heads is another
round trip, and the winner's entry is applied before its new head is readable
everywhere, so a proposer can find every seat refusing and no head yet past the
contested height. The votes themselves settle it. Each one reports where that
validator's own chain ends, and a vote is a verdict on the batch only if the
seat was building on the same head the proposer was: one that has moved past
the height has already applied somebody else's entry there, one that is behind
cannot judge transactions it has not caught up to, and one that signs an entry
other than the one put to it is saying the proposer's head is stale. None of
those is an opinion about the transactions. Below a threshold of seats actually
judging the batch, no quorum was reachable that round at all, so it defers and
re-reads the head instead of failing the caller.

Under a normally-loaded machine these windows are microseconds wide; under
coverage instrumentation they open far enough to fail
`two_independent_proposers_both_settle`, which is where the second one was
found. Timing bugs in this path are worth fixing rather than retrying around,
because a rejection is final: the round loop hands it straight back to the
caller.

A refusal now carries its reason out of the round, too. `received only 0 valid
votes` is true and useless; the message names the validators and quotes what
each of them said, and where every seat accepted but none of the signatures
counted, it says that instead — that is the signature of a stale head, not of a
bad batch.

The same reasoning applies one step earlier, before a proposal is even built.
A round starts by reading a head the quorum agrees on, and while two proposers
are reaching for one height the validators are briefly split across it, so no
head reaches threshold. That is not a refusal either: nothing was proposed and
nothing was applied. A client that treats it as one abandons a transaction the
chain was a backoff away from accepting. So an unagreed head defers the round
into the same retry loop a contested one uses, and only a round that exhausts
that budget is reported as failed. `a_client_waits_out_a_head_the_quorum_has_not_agreed_on`
drops the set one seat below threshold mid-round, heals it, and requires the
transaction to land anyway.

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

For the integer workloads - `PrimeCount`, `MatrixMultiply` and `CollatzPeak` -
validators recompute the result, in whole or through a witness.

This is intentionally straightforward rather than efficient.

Future workloads should use appropriate mechanisms such as:

- deterministic recomputation,
- random challenge/spot checks,
- redundant execution,
- commitment proofs,
- model-specific verification,
- trusted execution evidence where useful.

## Inference settlement

Generated text cannot be recomputed by a validator, so inference does not settle
in one step the way `PrimeCount` does. It settles in two, through a per-batch
holding account:

```text
hocmesh:escrow:<job_id>            the requester's money, still the requester's
hocmesh:holding:<assignment_id>    committed to this batch, owned by nobody
```

`InferenceReceipt` moves one batch's price from the job escrow into that batch's
holding account. It is signed by the requester over the assignment, the batch
range, the price and the outputs digest. Taking delivery is the requester's
statement that it now owns the outcome of this batch, whatever the text turns
out to say.

`InferenceReward` moves the holding account to the provider. It carries two
signatures: the provider's claim over the batch, and the requester's acceptance
over the *same* digest.

`InferenceDispute` moves the holding account to `hocmesh:community-issuance`. It
is signed by the requester and carries a reason. It returns nothing to the
requester, which is the point: a dispute costs the same as an acceptance, so
neither party gains by lying about the quality of an answer.

Validators pin the postings on all three. A receipt may only debit the job
escrow and credit that batch's holding account; a dispute may only debit the
holding account and credit the commons. A requester cannot point a dispute at
its own account, and a provider cannot pay itself out of an escrow directly.

The claim keys stage the two halves without any extra state:

```text
escrow:<job_id>:<start>:<end>    InferenceReceipt, InferenceRefund
payout:<job_id>:<start>:<end>    InferenceReward, InferenceDispute
```

A batch is therefore either taken or reclaimed, never both, and either accepted
or disputed, never both. Nothing needs to record that a receipt happened before
a payout: the holding account is empty until the receipt lands, and the
conservation rule refuses an overdraw.

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

## Bootstrapping from a snapshot

A checkpoint saves an existing replica work. It does nothing for a machine
joining for the first time, which still has to fetch and replay every entry
ever written. A snapshot is that checkpoint made portable:

```bash
hocmesh-validator snapshot --db ledger.db \
  --validators validators.json --out ledger-snapshot.json
hocmesh ledger-restore --validators validators.json \
  --db .hocmesh/ledger-mirror.db --snapshot ledger-snapshot.json
```

The file holds three things: the certificate for the head entry, the
checkpoint a quorum signed over that head, and the state itself. Reading one
checks all three against a validator set the operator already trusts — the
certificate has to carry a quorum, the checkpoint has to carry a quorum, the
two have to name the same entry, and the state has to hash to exactly the
digest the quorum signed. A file failing any of those is refused.

The set is supplied by the operator and never read out of the file: a snapshot
that carried its own list of who to believe would prove nothing. That is what
makes the route irrelevant. The file can arrive over a web server, a mirror,
or a USB stick, because a forged one is caught by the reader rather than by
whoever handed it over.

A restore refuses a store that already holds a chain, so a snapshot can never
be used to overwrite history a node had already verified for itself. From
there `ledger-sync` carries on from the snapshot's height rather than from
genesis, and `ledger-audit` resumes from the checkpoint it arrived with.

Snapshot state commits to lifetime earned and spent as well as to balances,
because those totals are part of what validators compare when they answer a
balance query. A restored node carries them in as a baseline underneath the
postings it later collects, so it agrees with a node that replayed everything
instead of splitting the quorum on every account it is asked about.

## Account history

A balance says where an account stands. An operator reconciling a bill, or an
account disputing one, needs the postings that produced it. Every posting a
certificate carries is indexed as it is applied, keyed on
`(account_id, sequence, posting_index)`, so reading one account's history is a
seek into that key rather than a scan of every posting the ledger ever wrote.

Validators serve it at `GET /v1/ledger/history/{account}?before=&limit=`, and
`hocmesh ledger-history` reads it either from a local mirror or, with
`--validators`, off the network:

```bash
hocmesh ledger-history --db .hocmesh/ledger-mirror.db --account <node-id>
hocmesh ledger-history --validators validators.json --account <node-id> --limit 20
```

Pages run newest first and the cursor is a sequence, not an offset, so a page
stays correct while the chain grows underneath the reader. That choice forces
one rule: a page never stops in the middle of one entry's postings, because the
next page asks for everything strictly below a sequence and would step over
whatever was left. A page that would split an entry either takes the whole
entry or drops it to the next page. A cursor is returned only when older
postings genuinely exist, so following it never lands on an empty page.

History is served unsigned, unlike a balance or a certificate. It is an index
over evidence the chain already holds, not a new claim about it: every row names
the sequence and transaction it came from, so anything resting on a row can be
checked against the certificate at that height. Pruning keeps this index --
`account_activity` survives because lifetime totals are part of the balance
proofs validators compare -- but a node bootstrapped from a snapshot holds no
postings from before its checkpoint, and says so by ending the page rather than
handing out a cursor into history it does not have.

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

Mirroring follows the same rule as auditing. A client fetching a run of
entries checks each one against the set that governed it, walking that set
forward across any change the run contains, because an old entry was signed by
whoever held a seat at the time. Checking history against the seats sitting now
would reject the whole chain the moment anybody joined or left.

See `docs/SECURITY.md`, *Validator set membership*, for the vouch message, the
threshold rationale, and the operator commands.

## Consensus boundary

This implementation is a quorum-certified linear log with ballot-ordered heights and on-ledger membership changes. It is not a Byzantine-fault-tolerant state machine: validators are assumed to follow the protocol, and a validator that signs two entries at one height is detectable but not prevented.

Before hostile Internet-scale deployment, the recommended next step is to either:

1. implement a formally specified BFT consensus protocol around these transaction rules, or
2. embed a mature consensus library/protocol while keeping hocMESH's ledger transaction validation logic.

Do not replace the current voting rules with an ad-hoc "majority wins" implementation that permits double voting.

## Coordinator crash recovery

Before quorum submission, the coordinator persists a `ledger_intents` record containing the exact serialized transaction and leaves affected work in a non-runnable state (`funding` or `settling`).

After a restart, `hocmesh-coordinator recover` or automatic startup recovery asks validators for a signed quorum claim proof. A certified claim is finalized locally; an absent claim causes the coordinator to retry the exact same transaction. This avoids creating a second debit/reward after an ambiguous network failure.

The proposal client also serialises rounds within one process, so a single coordinator batches rather than races itself. Independent processes are handled by ballots rather than by that lock; see "How a contested height is resolved".

Recovery runs on a timer as well as at startup -- every fifteen seconds while the coordinator is serving -- and every intent is judged on its own. A pass never stops early. The intent that fails is rarely the one that matters most, and the ones queued behind it have done nothing wrong; before this was isolated, one row the daemon could not process stopped every later row on every pass, forever.

Two kinds of failure are distinguished. A transient one -- no quorum yet, a validator unreachable, the local write briefly blocked -- costs that intent one tick and nothing else. A structural one cannot settle under its own claim key no matter how long anyone waits: the transaction no longer derives the key it is filed under, or names a kind the coordinator does not issue. Those are parked as `unrecoverable` with the reason attached, and a transient fault that has not cleared after `MAX_INTENT_ATTEMPTS` passes is parked the same way, because a fault that never clears is indistinguishable from a permanent one and an intent retried forever is an intent nobody ever reads.

Parking is bookkeeping and nothing else. No ledger entry is written, none is withdrawn, and no CU moves -- the coordinator has no standing to do any of that. The same pass also counts the other half of a partial failure: work still parked in `funding` or `settling` that no pending intent covers any more. That gap is reported and deliberately not repaired, because filling it locally would be the coordinator deciding on its own that CU exists. Both are readable at `GET /v1/ledger/reconciliation` and through `hocmesh reconciliation`, and neither has a companion that forces an intent through.

## Requester cannot pay itself

For member-funded jobs, the certified reservation records the requester identity. Provider-reward validation rejects a reward when `provider_node_id == requester_node_id`. The same invariant is replayed by offline client audit, so a malicious scheduler cannot bypass it without producing an invalid ledger.
