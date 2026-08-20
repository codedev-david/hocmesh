# Release Engineering

MESH release artifacts should be built from a clean checkout using the pinned
Rust toolchain in `rust-toolchain.toml`.

Required pre-release checks:

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo audit
cargo deny check
```

Release folders are produced with:

```bash
./scripts/build-release.sh
```

or on Windows PowerShell:

```powershell
./scripts/build-release.ps1
```

The release folder must include:

- `mesh`
- `mesh-coordinator`
- `mesh-validator`
- `README.md`
- `CODEX_HANDOFF.md`
- `LICENSE`
- `docs/`
- `config/`

Before publishing a public release:

- Generate and publish SHA-256 checksums for every binary and packaged archive.
- Generate a CycloneDX SBOM from the exact source revision used for the build.
- Sign platform binaries or installers with the relevant OS signing mechanism.
- Record the Rust toolchain, target triple, source commit, and build command.
- Keep validator membership files out of generic release artifacts unless they
  are explicitly intended as a public deployment policy.

Reproducible-build notes:

- Do not build from a dirty worktree for public artifacts.
- Use the pinned Rust toolchain instead of local `stable`.
- Avoid embedding local absolute paths, machine-specific timestamps, or private
  validator configuration into release artifacts.
- Preserve `Cargo.lock` for application releases.
