# Package Manifest

This archive is a source repository for **hocMESH Compute v0.3.0**.

Primary deliverables:

- `README.md` — compile, install, run, validator, client, and ledger walkthrough.
- `CODEX_HANDOFF.md` — prioritized engineering instructions for Codex CLI.
- `docs/FULL_ORIGINAL_SPEC.md` — complete original end-goal product specification.
- `docs/FULL_SYSTEM_SPEC.md` — consolidated Compute Core architecture and end-state design.
- `docs/HOCMESH_AI.md` — implemented AI architecture, commands, and runtime boundaries.
- `docs/ARCHITECTURE.md` — implemented and future topology.
- `docs/LEDGER.md` — replicated CU accounting design.
- `docs/PROTOCOL.md` — signed wire behavior (current protocol v3).
- `docs/SECURITY.md` — threat model and hardening requirements.
- `docs/CONSENSUS_INVARIANTS.md` — rules that future changes must preserve.
- `docs/CRASH_RECOVERY.md` — durable settlement and failure-reconciliation design.
- `docs/ROADMAP.md` — implementation path to hocMESH AI.
- `docs/IMPLEMENTATION_STATUS.md` — exact implemented/not-implemented matrix.
- `docs/RELEASE_ENGINEERING.md` — release checks, SBOM, signing, and reproducibility notes.
- `crates/` — all Rust source.
- `config/validators.example.json` — quorum validator membership example.
- `scripts/` — validation, release build, local demo, user install, and native package helpers.
- `packaging/` — WiX MSI and Debian package definitions; scripts also build macOS PKG artifacts.
- `deny.toml` — audited dependency license, advisory, duplicate, and source policy.
- `.github/workflows/ci.yml` — cross-platform compiler/test/lint workflow.
- `SHA256SUMS.txt` — integrity hashes for package contents.
