# MESH Protocol v4

## Authentication

Every live mutating/worker request contains an `AuthProof`:

```text
node_id
timestamp
nonce_b64
signature_b64
```

Canonical signed message:

```text
mesh-v4|ACTION|NODE_ID|TIMESTAMP|NONCE|BODY_HASH
```

The server verifies:

1. public key maps to `node_id`,
2. Ed25519 signature is valid,
3. timestamp is within allowed skew,
4. nonce has not already been consumed.

Historical ledger audit deliberately verifies the signature without enforcing current clock skew.

## Request replay protection

The coordinator stores recently used `(node_id, nonce)` pairs in SQLite.

A signed request replayed within the normal timestamp window is rejected.

## Deterministic IDs

Normal job ID:

```text
job_<hash(request-auth-nonce)>
```

Assignment ID:

```text
asg_<hash(job_id)>_<shard_index>
```

This prevents a coordinator from replaying one signed user submit authorization under arbitrary new job IDs.

## Work model

The protocol is declarative:

```rust
pub enum WorkSpec {
    PrimeCount { start: u64, end: u64 }
}
```

Do not introduce an `ExecuteShell`, `RunBinary`, or arbitrary command variant.

## Signed result body

Provider result authentication covers:

```text
assignment_id
job_id
shard_index
WorkSpec
reward_mcu
system_funded
WorkResult
```

That exact metadata is also placed into the ledger evidence.

## Main coordinator routes

```text
POST /v1/nodes/register
POST /v1/nodes/heartbeat
POST /v1/work/poll
POST /v1/work/result
POST /v1/jobs/submit
GET  /v1/jobs/{id}
GET  /v1/nodes/{id}/balance
GET  /v1/nodes/{id}
GET  /v1/network/stats
GET  /v1/network/peers
```

`GET /v1/network/peers` is deliberately unauthenticated and returns a small
random sample of nodes that have opted into serving latency probes. It is a
directory lookup, not an authority: nothing in the response is trusted, it only
tells a node who is worth measuring. Gossip can replace it without a wire change.

## Node routes

```text
POST /v1/proximity/probe
```

Served only by nodes started with `--probe-listen`. The request carries the
caller's current position and the round trip it last measured to this node, so
both sides can fit from one exchange; the response carries the responder's
position as it stands, whether or not it is yet confident enough to advertise
it for scheduling.

## Validator routes

```text
GET  /v1/ledger/head
GET  /v1/ledger/balance/{account}
POST /v1/ledger/propose
POST /v1/ledger/commit
GET  /v1/ledger/entries?from=N&limit=M
```

## Versioning

`PROTOCOL_VERSION` is currently `4`.

Breaking wire changes should bump the protocol version and preserve explicit migration/compatibility behavior rather than silently changing serialized structures.

Version `4` added `network_coordinate` and `probe_endpoint` to
`NodeCapabilities`, and with them the two routes above. Because the version is
part of the signed message, a v3 node cannot register against a v4 coordinator:
registration rejects the mismatch before it checks the signature at all.
