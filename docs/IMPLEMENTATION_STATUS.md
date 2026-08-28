# Implementation Status

## hocMESH Compute Core

| Area | Status | Evidence |
|---|---|---|
| Identity and replay-resistant authentication | Implemented | Ed25519 signatures, timestamp/nonce checks, replay tests |
| Declarative CPU workload execution | Implemented | `PrimeCount`, `MatrixMultiply` and `CollatzPeak`; a fixed allow-list, sharded and audited by the same rules |
| Work leases and failure requeue | Implemented | Coordinator SQLite state and process tests |
| Contribution-first CU accounting | Implemented | Reservation/escrow/reward invariants |
| Replicated validator ledger | Implemented | Hash-linked entries and quorum certificates |
| Crash-safe settlement intents | Implemented | Recovery integration test |
| Reconciliation of partial coordinator/ledger failures | Implemented | A pass at startup and every 15s judges each persisted intent on its own, so a broken one never blocks the ones behind it; transient faults retry, structural ones are parked `unrecoverable` with the reason; work waiting on funding no intent covers is counted, never repaired. Read at `/v1/ledger/reconciliation` or `hocmesh reconciliation` |
| Coordinator federation | Implemented | `--federation <file>` names this coordinator, its region and its peers; ownership of a job is a rendezvous hash over the coordinators currently answering, so nothing is elected and nothing is handed over. Read at `/v1/federation/status` and `/v1/federation/jobs/{job_id}` |
| Automatic coordinator failover | Implemented | Peers probed on their own interval; a peer leaves the live set after `misses_before_down` consecutive failures and its jobs become the survivors' to hand out. Leases it issued are shortened to a 60s grace rather than cancelled, so a worker mid-shard can still report in and be paid |
| Incremental ledger sync | Implemented | Coordinators resume replay from a persisted `(sequence, validator set)` watermark, so entries are verified against the set that signed them rather than the set sitting now |
| Resource graph | Implemented | Vertices are registered machines; edges are predicted RTT from measured coordinates, else the route through the coordinator, else an explicit placeholder. Greedy min-diameter clustering for co-scheduled work. Read at `/v1/topology` |
| Scheduling by hardware/network/reliability/locality | Implemented | `poll_work` scores a bounded oldest-first window: benchmark converted to mCU/s through `REFERENCE_OPS_PER_MCU` with a hard refusal past the lease, round trip amortised against shard size, exposure limited by audit standing, affinity to shards and manifests already held, and a starvation bonus that outranks any fit |
| Coordinator rebuild from the chain | Implemented | `rebuild` replays certified entries into an empty database; shard ids are derived, so a replacement finishes a half-done job without re-offering or re-paying a settled shard |
| Peer mirror and offline audit | Implemented | Validator quorum sync/audit commands |
| Snapshot bootstrap and out-of-band checkpoints | Implemented | `snapshot`/`ledger-restore` write and adopt a quorum-signed state file; it is refused unless certificate, checkpoint and state hash all agree against the operator's own validator set, and refused again over a store that already holds a chain |
| Indexed account history | Implemented | Every posting is indexed on `(account_id, sequence, posting_index)` as it is applied; validators serve `/v1/ledger/history/{account}` and `ledger-history` reads it from a mirror or off the network, paging newest-first on a sequence cursor that never splits one entry |
| Operator resource limits | Implemented | Persisted share of CPU/memory/GPU plus a separate `--ai on\|off\|auto` consent to serve inference; advertised capacity is the share, not the machine |
| Network coordinates | Implemented | Vivaldi fit from measured probes, persisted across restarts, advertised only once fitted |
| Contested-height recovery | Implemented | Ballot-ordered proposals: any client can take a height, and one that finds an entry already half-signed there is obliged to finish it |
| Community issuance authorization | Implemented | `CommunityReserve` carries threshold sponsorships from the sitting set, bound to job id, workload, shards and price |
| Validator set membership | Implemented | On-ledger join/leave carrying threshold vouches from sitting members; clients follow the chain forward with `refresh_set` |
| Key custody | Implemented | Signing key sealed with XChaCha20-Poly1305 under an Argon2id key from `HOCMESH_IDENTITY_PASSPHRASE` |
| Behaviour under network faults | Partially implemented | Integration tests drive the quorum through a fault-injecting relay: WAN-scale latency, minority partition (settlement continues, laggard repaired by `validator sync`), majority partition (settlement refuses, stranded shard pays once on heal), and clock skew either side of the 300s window. Still one machine over loopback: no multi-host run, NAT traversal, packet loss or reordering |
| Adversarial and property coverage | Partially implemented | `hocmesh-ledger` property tests: no single edit to a settled reward survives validation (10 mutations across postings, evidence, and signature), the community issuance ceiling is never crossed, a replayed chain is deterministic and idempotent and sums to zero. Byzantine cases covered: an equivocating quorum cannot fill a height twice, a quorum of strangers settles nothing. Not covered: equivocation combined with partition, and any run on more than one machine |

## hocMESH AI

| Component | Status | Evidence / boundary |
|---|---|---|
| Model registry | Implemented | Local and coordinator SQLite registries; authenticated publication |
| Content-addressed chunks | Implemented | SHA-256 paths, atomic import, deduplication, read verification |
| Peer-to-peer model seeding | Implemented | Peer HTTP server/client, integrity checks, rarest-first planner |
| CUDA backend | Implemented adapter | NVIDIA discovery; feature-gated llama.cpp adapter; requires a CUDA llama.cpp build |
| ROCm backend | Implemented adapter | ROCm discovery; feature-gated llama.cpp adapter; requires a HIP/ROCm llama.cpp build |
| Metal backend | Implemented adapter | Metal discovery; feature-gated llama.cpp adapter; requires macOS and a Metal llama.cpp build |
| Runtime acquisition | Implemented | `runtime-install` fetches a llama.cpp release pinned by SHA-256 per OS/arch; refuses anything that does not match |
| Model acquisition | Implemented | `model-pull` resolves a GGUF on the Hub or a `--url`, verifies its digest, resumes interrupted transfers, derives the architecture from the GGUF header |
| GGUF manifests | Implemented | Schema and magic-header validation |
| GGUF tensor directory | Implemented | `gguf::tensor_directory` reads name, type, shape and offset for every tensor plus the declared alignment; `tensors_for_layers`/`extents_for_layers`/`chunks_for_extents` turn a layer range into byte spans and chunk indexes; `hocmesh model-inspect` prints them. Reads the header only, so it works on a partly-fetched file |
| Safetensors manifests | Implemented | Schema and header-length validation |
| GPU capability benchmark | Implemented | Capability report, host-transfer baseline, real-model `llama-bench` adapter |
| Latency-aware scheduler | Implemented | Backend/VRAM/dtype/cache/RTT/bandwidth/load/failure scoring |
| Batch parallelism | Implemented end-to-end | Distributed prompt assignments executed by participant daemons |
| Pipeline parallelism | Implemented control/data plane | Complete layer plans and ordered tensor transport; partial-layer kernels require a runtime plugin |
| Model/tensor parallelism | Implemented control/data plane | Contiguous ranks and tensor transport; collective kernels require a runtime plugin |
| Tensor transport | Implemented | Checksums, ordered delivery, replay/window checks, HTTP route failover |
| Failure-aware rerouting | Implemented | Persistent failed-node exclusions and device-correct reassignment |
| Inference settlement | Implemented | Two-stage receipt/verdict: escrow moves to a per-batch holding account on delivery, then to the provider on a signed acceptance or to the commons on a signed dispute |

## Desktop application

| Component | Status | Evidence / boundary |
|---|---|---|
| Tray and window | Implemented | Tauri v2 shell; the tray menu and its health icon are built from a `TrayModel` computed in plain Rust, so what the menu offers is decided and tested away from the event loop |
| Node supervision | Implemented | Start, stop, restart and attach. A daemon the app did not start is never stopped when the app quits, and a daemon already running is attached to rather than duplicated; both rules live in `supervisor.rs` and are covered by unit tests and an end-to-end test that spawns a daemon behind the app's back |
| Live dashboard | Implemented | Poll-and-emit snapshot every 3s: run state, coordinator, worker count, jobs completed/failed, what share of this machine is lent, and whether inference actually reached the mesh. Readiness is read from the same `advertised_capabilities` the daemon registers with, so the window cannot claim a readiness the coordinator was never told about |
| Ledger view | Implemented | Balance and newest-first paged history read through the daemon's control endpoint, each page marked with whether a validator quorum stood behind it; a balance whose history page is missing shows as a total over an empty table rather than as no ledger |
| Settings and limits | Implemented | Coordinator, worker ceiling, AI consent and the CPU/memory/GPU shares, persisted and applied to a running node without a restart. The settings file is a consent record: the app writes what the operator set and never widens a share on its own |
| End-to-end proof | Implemented | An integration test drives the app's own layers against a real coordinator and daemon: cold snapshot, start, work completed, a paid ledger entry, a limit changed and read back off disk, then stop |
| Installers | Implemented | Tauri bundler MSI and NSIS setup on Windows, DMG on macOS, DEB and AppImage on Linux, each carrying node, coordinator and validator as sidecars beside the app; `scripts/package-desktop.*` open what they produced and fail unless all four executables are inside |
| One peer per machine | Implemented | The desktop installer carries the whole peer — node, coordinator and validator — so it is the headless install plus a window rather than a second, different product. The two Linux packages therefore claim the same `/usr/bin/hocmesh` and declare that they replace each other: `Provides`/`Conflicts`/`Replaces: hocmesh` on the desktop `.deb`, `Conflicts`/`Replaces: hoc-mesh-desktop` on the headless one. `package-desktop.sh` writes any of those fields the bundler omitted into the built package and then reads all three back out with `dpkg-deb --field`, so a package that cannot replace its counterpart fails the build. On Windows and macOS nothing enforces this yet: the MSI/NSIS pair share no upgrade code and the `.pkg` and `.dmg` no identifier, so installing both leaves two copies rather than one replacing the other |

## Distribution

| Component | Status | Evidence / boundary |
|---|---|---|
| Windows installer | Implemented | WiX-built MSI, ICE validation, administrative extraction, peer execution smoke test |
| macOS installer | Implemented | Native PKG, payload inspection, extracted peer execution smoke test |
| Linux installer | Implemented | Native DEB, metadata/content validation, extracted peer execution smoke test |
| Desktop installers | Implemented | Built and content-checked on all three platforms in CI, published alongside the headless installers on a tagged release |
| Release integrity | Implemented | Per-artifact SHA-256 checksums and CycloneDX SBOM |
| Platform signing | Deployment configuration required | Installers are unsigned until release signing identities are supplied |

## Verification

The workspace is pinned to Rust 1.97.1. CI runs format, check, tests, and Clippy on Windows, Linux, and macOS; separately compiles/tests CUDA, ROCm, and Metal adapter features; enforces measured coverage; audits dependencies; checks policy; emits per-crate CycloneDX SBOMs; and extracts/runs every native installer.

Hardware claims are deliberately scoped. Unit and process tests prove hocMESH-owned logic. Actual GPU execution also requires the matching device, driver, and backend-enabled llama.cpp binaries. A platform that was not physically exercised is never reported as hardware-validated.
