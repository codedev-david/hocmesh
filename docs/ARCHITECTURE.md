# hocMESH Architecture

## Product split

```text
hocMESH AI
  model/runtime/inference layer
            │
            ▼
hocMESH Compute Core
  identity, scheduling, ledger, resources, verification
            │
            ▼
Community hardware
```

## Current runtime topology

```text
                    ┌───────────────────┐
                    │ hocMESH Coordinator  │
                    │ scheduler/control │
                    └─────────┬─────────┘
                              │
        outbound HTTP(S) poll │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│ hocmesh Node A │        │ hocmesh Node B │        │ hocmesh Node C │
│ CPU/GPU cap │        │ CPU/GPU cap │        │ CPU/GPU cap │
└─────────────┘        └─────────────┘        └─────────────┘

                  accounting proposals
                           │
               ┌───────────┼───────────┐
               ▼           ▼           ▼
          Validator A  Validator B  Validator C
          full ledger  full ledger  full ledger
```

## Why nodes pull work

Home and consumer networks are often behind NAT, CGNAT, or restrictive firewalls.

A pull model requires only outbound connectivity:

```text
hocmesh -> HTTPS -> coordinator
```

No participant has to expose SSH, RDP, a shell, or an inbound job listener.

## Coordinator responsibilities

The coordinator owns ephemeral scheduling state:

- node registration,
- capability advertisements,
- heartbeats,
- work queues,
- work leases,
- job reconstruction,
- shard assignment.

In quorum mode it does **not** own authoritative CU state.

## Validator responsibilities

Validators own replicated accounting state:

- certified job reservations,
- community reservations,
- provider rewards,
- balances,
- settlement uniqueness,
- voting locks,
- history replication.

## Node responsibilities

A participant node owns:

- private identity key,
- hardware/resource policy,
- worker loops,
- declarative work execution,
- signed result proofs,
- optional full-ledger mirror,
- its own position in the network, measured rather than declared.

## Separation of control and data planes

Current MVP:

```text
Control + small work data -> coordinator HTTP API
Ledger -> validators
```

hocMESH AI:

```text
Control plane -> scheduler/federation
Ledger plane  -> validator consensus
Model data    -> P2P content-addressed swarm
Tensor data   -> direct low-latency worker paths
```

## Peer discovery

Latency probing runs directly between nodes. A node asks the coordinator for a
small random sample of probe-serving peers, times a real round trip to each, and
fits a Vivaldi coordinate from what it measured.

The coordinator is only a directory here. It never asserts where a node sits,
and nothing in its peer sample is trusted: it just says who is worth measuring.
A node that has not measured enough advertises no coordinate at all, and the
scheduler scores it by the worker's coordinator-observed latency instead - a
worse number, honestly labelled, rather than a confident guess.

Probing outward is unconditional; serving probes is opt-in (`--probe-listen`),
so a node behind NAT still earns a position without ever accepting a connection.
Each exchange also carries the round trip the caller measured, which lets the
responder fit itself from the same packets rather than spending its own probe.

What remains is to replace that bootstrap directory with a DHT/gossip layer,
which needs no wire change, and to carry the same discovery to:

- model chunk providers,
- compute neighborhood discovery,
- direct data-path establishment.

The scheduler can still select topology while large model/tensor traffic bypasses the central coordinator.

## Heterogeneous compute

The scheduler should evolve from CPU queueing to a resource graph containing:

```text
CPU architecture / features
GPU vendor/backend
VRAM
RAM
benchmark score
network RTT
bandwidth
reliability
model cache inventory
thermal/power policy
```

## Parallelism implementation order

Recommended order:

1. task parallelism — implemented for deterministic CPU work,
2. batch inference — implemented end-to-end across GPU workers,
3. pipeline/model parallel planning — implemented for low-latency peers,
4. ordered activation/tensor transport with route failover — implemented,
5. partial-layer and collective kernels — supplied by compatible runtime plugins.

WAN tensor parallelism should not be assumed to outperform a single local GPU simply because aggregate FLOPS are larger. Communication is a first-class scheduling constraint.
