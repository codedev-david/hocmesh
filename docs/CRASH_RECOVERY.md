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
