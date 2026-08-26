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

Keys persist in a local JSON file, sealed with a passphrase when `HOCMESH_IDENTITY_PASSPHRASE` is set. See *Key custody, and why the ledger is not encrypted* below. Hardware-backed key storage remains the stronger option for a production validator.

## Replay resistance

Protocol v4 uses a random nonce in every signed API request.

The coordinator persists recent nonces and rejects reuse.

## Scheduler compromise

In quorum-ledger mode, compromising only the coordinator does not allow silent CU balance rewriting. It also does not allow minting: a `CommunityReserve` without threshold sponsorships from the sitting set is rejected by every validator.

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

## Validator set membership

Earning CU is already Sybil-proof: a fake node's results fail the recompute, so
spinning up machines buys nothing. What it could still buy is a seat at the
quorum, and a captured quorum can certify anything at all. The validator set is
therefore the one part of hocMESH that is deliberately not open.

Membership changes are ledger transactions (`TransactionKind::MembershipChange`).
They are proposed, voted on, and certified exactly like a settlement, and they
carry the same quorum certificate, so the set is derivable by replaying the
chain rather than by trusting that every operator edited the same JSON file the
same way. Out-of-band edits are not a supported operation and never were: the
membership hash is bound into every entry signature, so a set that differs from
the one the chain agreed on simply cannot verify anything.

### Vouching

A join is not self-service. The evidence must carry individually signed
sponsorships from sitting members over

```
hocmesh-vouch-v1|<previous_set_hash>|join|<validator_id>|<public_key_b64>|<resulting_set_hash>
```

The message names both the set the vouch was made against and the set it
produces, so a sponsorship cannot be replayed against a set that has since
moved on, and cannot be pointed at a different admission. Only signatures from
validators already in the set count, which is what stops a joiner voting itself
in. Duplicates from one validator count once.

The bar is the set's own consensus threshold. Making eviction easier than
agreement would itself be the attack: a minority able to vote out the majority
captures the quorum without ever holding it. The cost is that a set which has
already lost the ability to certify entries also cannot change itself — the
same liveness bound the ledger already lives under, and a safe one.

A membership change must move no CU. That is written as a requirement rather
than an exemption, so an admission can never be the one transaction kind that
shifts balances while presenting evidence that says nothing about them.

### Operating it

Vouching is an operator command, not an HTTP endpoint, and deliberately so: a
validator that automatically signed for whoever asked would make admission free
and destroy the whole defence.

```
hocmesh-node membership-vouch  --validators set.json --action join --member m.json --threshold 4
hocmesh-node membership-commit --validators set.json --action join --member m.json --threshold 4 \
                               --vouches vouches.json --out set.next.json
```

Each sponsor runs `membership-vouch` and returns the signature it prints;
`membership-commit` collects them and submits the transaction. Validators pick
the new set up as soon as the change is certified — they read it from the
store, not from disk.

Clients built from a file follow the same change without being restarted.
`LedgerNetwork::refresh_set` walks the chain forward from the height its set was
last established at, verifies the certificate on any entry that carries a
membership change against the set it already holds, and adopts the result. The
entries come from whichever validator answers, which does not have to be
trusted: nothing is adopted that the set already held did not certify, which is
the same rule an auditor replaying from genesis follows. The coordinator calls
it on its fifteen-second recovery tick, and the ledger client calls it once on a
rejected round before retrying that batch — rejected means nothing was applied
anywhere, so the retry is safe. A client whose *entire* set has rotated out has
nobody left to ask and does still need a new file. Staleness remains fail-safe
rather than fail-open throughout: a client on the wrong set cannot reach quorum,
it never certifies against one.

An audit follows the set the chain hands forward: each entry's certificate is
checked against the set sitting *before* that entry, and membership changes
apply to everything after. A full replay starts from the genesis file; one
resuming from a checkpoint asks the store for `set_at(checkpoint_height)`,
because a checkpoint has to be verified against the seats that signed it rather
than whoever holds them today.

## Double-vote protection

Validators persist a ballot lock before signing a ledger entry, and it survives restart.

A validator will not sign for a ballot older than the one it is holding, and it hands any entry it has already accepted back to the next proposer, which is then obliged to finish that entry rather than propose a different one. So a height carries at most one entry even when several clients reach for it at once.

The earlier design locked a validator to the first entry hash it saw at a height, with no way to release it. That is safe but not live: two proposers could split the set, neither reach threshold, and the height then be unfillable forever. Ordering the attempts is what makes the lock releasable without making it forgeable.
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

A mint also has to be authorized, not merely affordable. Every `CommunityReserve` carries `sponsors`: signatures from named members of the sitting validator set over the job id, the workload, the shard count and the price. Validation rejects it unless at least `threshold` distinct sitting members signed - the same k-of-n that admits a validator.

Sponsorships bind to one job. Lifting a signature off one mint and attaching it to another fails, because the price and the workload are inside what was signed.

The coordinator holds no key that can mint. It carries sponsorships an operator collected with `hocmesh community-vouch`; it cannot produce them.

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

## Key custody, and why the ledger is not encrypted

These two are one decision, so they are written down together.

The ledger is **not** encrypted, at rest or in the entries it serves, and that
is deliberate. Its entire security argument is that anyone can replay it and
arrive at the same balances. A chain nobody can independently check is worth
less than one everybody can read.

Encrypting the validator database would also have bought very little. Entries
are served over the API and clients are expected to mirror them, so a stolen
disk yields nothing an attacker could not have fetched. It would have protected
data that is public by design while making the one property the system depends
on harder to exercise.

Two related ideas were considered and rejected on the same grounds. Rotating
per-epoch account keys would buy weak unlinkability - the network layer leaks
far more than the ledger does - while breaking per-account CU conservation,
`BalanceProof`, escrow addressing, and the requester-cannot-pay-itself check.
Confidential amounts would make CU conservation uncheckable without a
zero-knowledge circuit, which contradicts the invariant outright.

### What does need to stay secret

The signing key. A validator's key is the whole quorum's security, and it lived
in `identity.json` in the clear, protected by a `chmod 0600` that was a silent
no-op on any platform without Unix file modes.

Setting `HOCMESH_IDENTITY_PASSPHRASE` now seals the key with XChaCha20-Poly1305
under an Argon2id-derived key. Setting it on a node that already has a
plaintext identity re-seals that identity in place, keeping the node id the rest
of the network knows it by. Where file modes cannot be enforced, an unsealed key
says so on stderr rather than reporting success.

It is a passphrase from the environment rather than a prompt because a
validator has to come back after a reboot without a human present. Supply it
the way the platform supplies secrets - a systemd credential, a service
environment, a secrets manager - not from a file next to the key it protects.

This does not defend against malware already running as the node's own user;
nothing that starts unattended can. It defends against the ways keys actually
escape: a backup, a synced folder, a copied disk, a repository someone committed
their working directory to. If a validator key does leak, the answer is now
eviction - see *Validator set membership* above.

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
