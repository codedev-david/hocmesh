# Implementation Status

## hocMESH Compute Core

| Area | Status | Evidence |
|---|---|---|
| Identity and replay-resistant authentication | Implemented | Ed25519 signatures, timestamp/nonce checks, replay tests |
| Declarative CPU workload execution | Implemented | `PrimeCount`, sharding, deterministic verification |
| Work leases and failure requeue | Implemented | Coordinator SQLite state and process tests |
| Contribution-first CU accounting | Implemented | Reservation/escrow/reward invariants |
| Replicated validator ledger | Implemented | Hash-linked entries and quorum certificates |
| Crash-safe settlement intents | Implemented | Recovery integration test |
| Client mirror and offline audit | Implemented | Validator quorum sync/audit commands |
| Operator resource limits | Implemented | Persisted share of CPU/memory/GPU; advertised capacity is the share, not the machine |
| Network coordinates | Implemented | Vivaldi fit from measured probes, persisted across restarts, advertised only once fitted |
| Contested-height recovery | Implemented | Ballot-ordered proposals: any client can take a height, and one that finds an entry already half-signed there is obliged to finish it |
| Community issuance authorization | Implemented | `CommunityReserve` carries threshold sponsorships from the sitting set, bound to job id, workload, shards and price |
| Validator set membership | Implemented | On-ledger join/leave carrying threshold vouches from sitting members; clients follow the chain forward with `refresh_set` |
| Key custody | Implemented | Signing key sealed with XChaCha20-Poly1305 under an Argon2id key from `HOCMESH_IDENTITY_PASSPHRASE` |

## hocMESH AI

| Component | Status | Evidence / boundary |
|---|---|---|
| Model registry | Implemented | Local and coordinator SQLite registries; authenticated publication |
| Content-addressed chunks | Implemented | SHA-256 paths, atomic import, deduplication, read verification |
| Peer-to-peer model seeding | Implemented | Peer HTTP server/client, integrity checks, rarest-first planner |
| CUDA backend | Implemented adapter | NVIDIA discovery; feature-gated llama.cpp adapter; requires a CUDA llama.cpp build |
| ROCm backend | Implemented adapter | ROCm discovery; feature-gated llama.cpp adapter; requires a HIP/ROCm llama.cpp build |
| Metal backend | Implemented adapter | Metal discovery; feature-gated llama.cpp adapter; requires macOS and a Metal llama.cpp build |
| GGUF manifests | Implemented | Schema and magic-header validation |
| Safetensors manifests | Implemented | Schema and header-length validation |
| GPU capability benchmark | Implemented | Capability report, host-transfer baseline, real-model `llama-bench` adapter |
| Latency-aware scheduler | Implemented | Backend/VRAM/dtype/cache/RTT/bandwidth/load/failure scoring |
| Batch parallelism | Implemented end-to-end | Distributed prompt assignments executed by participant daemons |
| Pipeline parallelism | Implemented control/data plane | Complete layer plans and ordered tensor transport; partial-layer kernels require a runtime plugin |
| Model/tensor parallelism | Implemented control/data plane | Contiguous ranks and tensor transport; collective kernels require a runtime plugin |
| Tensor transport | Implemented | Checksums, ordered delivery, replay/window checks, HTTP route failover |
| Failure-aware rerouting | Implemented | Persistent failed-node exclusions and device-correct reassignment |

## Distribution

| Component | Status | Evidence / boundary |
|---|---|---|
| Windows installer | Implemented | WiX-built MSI, ICE validation, administrative extraction, client execution smoke test |
| macOS installer | Implemented | Native PKG, payload inspection, extracted client execution smoke test |
| Linux installer | Implemented | Native DEB, metadata/content validation, extracted client execution smoke test |
| Release integrity | Implemented | Per-artifact SHA-256 checksums and CycloneDX SBOM |
| Platform signing | Deployment configuration required | Installers are unsigned until release signing identities are supplied |

## Verification

The workspace is pinned to Rust 1.97.1. CI runs format, check, tests, and Clippy on Windows, Linux, and macOS; separately compiles/tests CUDA, ROCm, and Metal adapter features; enforces measured coverage; audits dependencies; checks policy; emits per-crate CycloneDX SBOMs; and extracts/runs every native installer.

Hardware claims are deliberately scoped. Unit and process tests prove hocMESH-owned logic. Actual GPU execution also requires the matching device, driver, and backend-enabled llama.cpp binaries. A platform that was not physically exercised is never reported as hardware-validated.
