# MESH Architecture

## Product split

```text
MESH AI
  model/runtime/inference layer
            │
            ▼
MESH Compute Core
  identity, scheduling, ledger, resources, verification
            │
            ▼
Community hardware
```

## Current runtime topology

```text
                    ┌───────────────────┐
                    │ MESH Coordinator  │
                    │ scheduler/control │
                    └─────────┬─────────┘
                              │
        outbound HTTP(S) poll │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│ mesh Node A │        │ mesh Node B │        │ mesh Node C │
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
mesh -> HTTPS -> coordinator
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
- optional full-ledger mirror.

## Separation of control and data planes

Current MVP:

```text
Control + small work data -> coordinator HTTP API
Ledger -> validators
```

MESH AI:

```text
Control plane -> scheduler/federation
Ledger plane  -> validator consensus
Model data    -> P2P content-addressed swarm
Tensor data   -> direct low-latency worker paths
```

## Future peer discovery

The end state should support a DHT/bootstrap/gossip layer for:

- model chunk providers,
- latency probes,
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
