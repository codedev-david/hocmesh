# hocMESH Security Model

## Primary rule

A participant contributes compute, **not remote control of the computer**.

## Explicitly forbidden worker capabilities

A remote requester must not receive:

- SSH access,
- RDP access,
- shell access,
- arbitrary executable launch,
- unrestricted filesystem access,
- unrestricted network access,
- administrator/root privileges.

## Current execution model

The Rust worker matches an allow-listed `WorkSpec` enum and runs native trusted code shipped with the hocMESH client.

This is significantly safer than accepting arbitrary uploaded programs.

## Identity

Each node uses Ed25519.

The private key stays local.

The node ID is derived from the public key.

Current key persistence is a local JSON file. Public production should move private-key protection into OS secure storage or hardware-backed key facilities.

## Replay resistance

Protocol v4 uses a random nonce in every signed API request.

The coordinator persists recent nonces and rejects reuse.

## Scheduler compromise

In quorum-ledger mode, compromising only the coordinator does not allow silent CU balance rewriting.

Validators independently verify ledger transactions and keep separate replicas.

A compromised coordinator can still attack availability and scheduling quality. Scheduler federation and signed scheduler assignments are future hardening work.

## Validator compromise

Security depends on threshold assumptions.

Example 3-of-4:

- one malicious validator cannot create a conflicting quorum certificate,
- the 3-of-4 configuration is designed to tolerate **at most one Byzantine validator**,
- two Byzantine validators exceed that fault bound and can potentially participate in conflicting quorum intersections with different honest validators,
- the >2/3 threshold is necessary but the current protocol still needs formal leader/view-change rules before production BFT claims.

Real deployments should place validators under genuinely independent administrative control.

## Double-vote protection

Validators persist a vote lock before signing a ledger entry.

An honest validator refuses to sign a conflicting entry at the same sequence even after restart.

## Credit forgery defenses

- requester reservation signature,
- deterministic job ID,
- escrow,
- conservation rule,
- provider signature over exact work metadata,
- deterministic shard ID,
- reservation-to-reward binding,
- independent result recomputation,
- duplicate claim table,
- quorum certificate,
- replicated history,
- participant audit capability.

## Community issuance

Community bootstrap CU is not unrestricted.

Validator policy specifies a maximum cumulative issuance magnitude.

The bootstrap job is reserved into escrow through a certified `CommunityReserve` transaction before providers are paid.

Future production hardening should require a distinct governance authorization for new community reservations.

## Transport security

The Rust services currently expose HTTP listeners suitable for a private network/lab.

For public deployment, use HTTPS/TLS and authenticated infrastructure boundaries.

Do not send signed workload requests through untrusted plaintext networks merely because the payload itself contains signatures; confidentiality still matters.

## Denial of service

Before public release add:

- route-specific rate limits,
- body-size limits,
- connection limits,
- per-node scheduler quotas,
- validator proposal rate limits,
- computational verification budgets,
- abuse reputation/ban controls.

## Data privacy

Transport encryption does not make provider-visible workload data magically private from the provider executing it.

Future privacy tiers may use trusted execution environments or privacy-preserving techniques, but the product must not overclaim confidentiality.

## Supply chain

Before release:

```bash
cargo audit
cargo deny check
```

should be part of CI, with a reviewed dependency policy and reproducible release process.


## Crash ambiguity and exactly-once settlement

The coordinator uses a durable settlement-intent pattern. The exact transaction is stored before contacting validators, and affected scheduler objects are blocked until certification is reconciled. Signed quorum claim proofs let recovery distinguish "not certified" from "already certified" without trusting a single validator.

This supplies idempotent reservation/reward recovery for the current single-scheduler architecture. It is not a substitute for a complete BFT consensus protocol with leader election/view changes across competing proposers.
