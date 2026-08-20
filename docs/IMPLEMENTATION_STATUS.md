# Implementation Status

## Functional source paths present

| Area | Status | Notes |
|---|---|---|
| Rust workspace | Implemented | 6 crates |
| Node Ed25519 identity | Implemented | file-backed key |
| Replay-resistant API auth | Implemented | timestamp + nonce |
| CPU hardware discovery | Implemented | sysinfo |
| GPU discovery | Partial | NVIDIA + Apple detection only |
| Declarative workload runtime | Implemented | PrimeCount |
| Multi-worker execution | Implemented | Tokio worker loops |
| Scheduler leases/requeue | Implemented | SQLite |
| Requester self-work exclusion | Implemented | scheduler + validator + audit invariant |
| Local CU ledger | Implemented | development fallback |
| Quorum ledger | Implemented | validators + certificates |
| Escrow accounting | Implemented | job escrow accounts |
| Community issuance cap | Implemented | validator-set policy |
| Validator vote lock | Implemented | persistent per-height lock |
| Duplicate reward prevention | Implemented | settlement claim keys |
| Signed provider work metadata | Implemented | protocol v2 |
| Reward-to-reservation binding | Implemented | validators split root workload |
| Validator sync | Implemented | certificate catch-up |
| Participant ledger mirror | Implemented | `mesh ledger-sync` |
| Full offline ledger audit | Implemented | `mesh ledger-audit` |
| TLS in binary | Not implemented | use reverse proxy today |
| Full BFT view-change consensus | Not implemented | next production milestone |
| Membership rotation/epochs | Not implemented | static pinned set today |
| Coordinator/ledger crash saga | Implemented for reservation/reward intents | durable exact transaction + signed claim reconciliation; multi-coordinator BFT liveness remains |
| CUDA compute runtime | Not implemented | discovery only |
| ROCm runtime | Not implemented | roadmap |
| Metal compute runtime | Not implemented | roadmap |
| P2P model distribution | Not implemented | roadmap |
| Distributed LLM inference | Not implemented | roadmap |
| Consumer installers | Not implemented | build-release scripts included |

## Validation status of this archive

Static repository validation was performed while generating the package:

- required files present,
- JSON configuration parses,
- Cargo TOML files parse,
- workspace crate paths exist,
- source/archive checksum generation,
- no generated target/build directory included.

The generation environment did **not** provide `cargo`/`rustc`, so compiler verification is intentionally left as the first Codex CLI task. See `CODEX_HANDOFF.md`.
