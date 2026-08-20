# Release Engineering

MESH release artifacts should be built from a clean checkout using the pinned
Rust toolchain in `rust-toolchain.toml`.

Required pre-release checks:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-features --locked
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo audit
cargo deny check
cargo llvm-cov --workspace --all-features --locked --lcov --output-path target/mesh-coverage.lcov --fail-under-lines 45
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

Native participant-client installers are produced from the release binary with
`scripts/package-linux.sh`, `scripts/package-macos.sh`, and
`scripts/package-windows.ps1`. Each packager validates the resulting DEB, PKG,
or MSI structure before it is uploaded.

Before publishing a public release:

- Generate and publish SHA-256 checksums for every binary and packaged archive.
- Generate a CycloneDX SBOM from the exact source revision used for the build.
- Sign platform binaries or installers with the relevant OS signing mechanism.
- Record the Rust toolchain, target triple, source commit, and build command.
- Keep validator membership files out of generic release artifacts unless they
  are explicitly intended as a public deployment policy.

GitHub release process:

- Pushing a `v*` tag runs `.github/workflows/release.yml`.
- The workflow builds Windows x86_64, Linux x86_64, and macOS arm64 release
  artifacts.
- Each platform publishes both a full archive and a native participant-client
  installer (DEB, MSI, or PKG), with a `.sha256` checksum for each.
- The workflow creates a draft prerelease so checksums, release notes, and
  signing status can be reviewed before publishing.

Reproducible-build notes:

- Do not build from a dirty worktree for public artifacts.
- Use the pinned Rust toolchain instead of local `stable`.
- Avoid embedding local absolute paths, machine-specific timestamps, or private
  validator configuration into release artifacts.
- Preserve `Cargo.lock` for application releases.
