# Coordinator / Ledger Crash Recovery

hocMESH separates scheduler state from authoritative CU state. That creates an unavoidable distributed transaction boundary: a validator quorum can certify a CU transfer while the coordinator crashes before updating its local SQLite rows.

The v0.2 recovery design uses durable intents and idempotent ledger claims rather than pretending SQLite and the validator quorum share one transaction.

## Reservation path

```text
verify signed submit request
        ↓
persist job = funding
persist shards = blocked
persist exact JobReserve ledger transaction
        ↓
request validator certification
        ↓
quorum certificate
        ↓
mark ledger intent certified
job = pending
shards = pending
```

Workers cannot receive `blocked` shards.

## Provider reward path

```text
verify signed provider result
verify deterministic result
        ↓
assignment = settling
persist result
persist exact ProviderReward ledger transaction
        ↓
request validator certification
        ↓
quorum certificate
        ↓
mark intent certified
assignment = completed
possibly job = completed
```

`settling` assignments are not re-leased.

## Recovery algorithm

`hocmesh-coordinator recover --db hocmesh.db --validators validators.json` scans pending intents.

For each intent it asks validators for the claim (`reserve:<job>` or `reward:<assignment>`).

- If any validator returns the full quorum certificate, the coordinator verifies the certificate and can safely finalize local state even if only that replica received the prior commit.
- Otherwise hocMESH requires a threshold of signed validator proofs agreeing the claim is absent at the same ledger head before retrying the exact persisted transaction.

The same recovery runs best-effort on coordinator startup and periodically while the coordinator is running.

## Losing the coordinator entirely

`recover` finishes settlements a coordinator started. It assumes the database
survived. The harder case is that it did not: the disk is gone, or the host is,
and someone has to stand a replacement up from nothing.

That works because the coordinator holds no authority. It is a cache of
scheduling state over facts the ledger already keeps: a `JobReserve` or
`CommunityReserve` names the job, its requester, its work spec and its shard
count, and a `ProviderReward` or `JobRefund` names the shard it settled.

`hocmesh-coordinator rebuild --db new.db --validators validators.json` replays
the chain from entry 1 into an empty database. It verifies every certificate,
refuses a gap in the sequence, re-checks historical evidence and follows
membership changes forward, then turns each settled transaction back into the
rows a scheduler needs.

Two properties make that safe rather than merely convenient.

- **Shard identity is derived, not remembered.** `assignment_id(job, index)`
  hashes the job id, so a replacement reconstructs the same ids the dead
  coordinator issued. It cannot invent a shard, and it cannot rename one.
- **The ledger, not the coordinator, refuses the second payment.** Both
  `ProviderReward` and `JobRefund` claim `reward:<assignment>`. A rebuilt
  coordinator that got its bookkeeping wrong and re-offered a settled shard
  would have the reward refused at the validators. The worst a bad rebuild
  can cost is wasted compute; it can never mint CU or pay twice.

Balances are deliberately not replayed. In quorum mode the coordinator answers
a balance query by asking the validators, so rebuilding a local ledger here
would be building a second, weaker copy of the thing that is already
authoritative.

Node rows come back as placeholders with `last_seen = 0`, which the scheduler
ignores, so a rebuilt coordinator hands out no work until real machines
re-register and say what they can do.

The rebuild is idempotent: every write is an upsert keyed on an id the chain
fixes, so running it twice, or against a half-populated database, converges.

`a_replacement_coordinator_rebuilds_from_the_chain_and_finishes_the_job` in
`crates/hocmesh-integration-tests/tests/quorum_flow.rs` proves the whole path:
a job is left half done, the coordinator and its database are destroyed, a
replacement is rebuilt from the chain onto an empty file, and it finishes the
job without ever re-offering or re-paying the shard that was already settled.

## Why the full certificate matters

A certificate is already a portable proof that the threshold signed one exact ledger entry. Requiring the certificate to have been copied to a threshold of databases would unnecessarily turn a post-certification network interruption into a deadlock.

## Failure-injection tests Codex should add

Kill the coordinator at each boundary and verify exactly-once accounting:

1. after local intent commit, before proposals;
2. after one validator vote;
3. after threshold votes, before any commit;
4. after one or two replicas commit a 3-of-4 certificate;
5. after threshold replicas commit, before local finalization;
6. after local finalization, before HTTP response.

After every case:

- no requester is charged twice;
- no provider is rewarded twice;
- no runnable shard exists without certified funding;
- no completed shard lacks a certified reward in quorum mode;
- client ledger audit succeeds.
