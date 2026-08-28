# Release Engineering

hocMESH release artifacts should be built from a clean checkout using the pinned
Rust toolchain in `rust-toolchain.toml`.

Required pre-release checks:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-features --locked
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo audit
cargo deny check
cargo llvm-cov --workspace --all-features --locked --lcov --output-path target/hocmesh-coverage.lcov --fail-under-lines 45
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

- `hocmesh`
- `hocmesh-coordinator`
- `hocmesh-validator`
- `README.md`
- `CODEX_HANDOFF.md`
- `LICENSE`
- `docs/`
- `config/`

Headless installers are produced from the release binaries with
`scripts/package-linux.sh`, `scripts/package-macos.sh`, and
`scripts/package-windows.ps1`. Each packager validates the resulting DEB, PKG,
or MSI structure before it is uploaded.

Desktop-app installers are produced from the same release binary with
`scripts/package-desktop.sh` (macOS DMG, Linux DEB and AppImage) and
`scripts/package-desktop.ps1` (Windows MSI and NSIS setup executable). These
carry the window *and* the whole peer it supervises -- node, coordinator and
validator -- so each packager opens what it produced and fails unless all four
executables are inside. Linux hosts need
`libwebkit2gtk-4.1-dev`, `libjavascriptcoregtk-4.1-dev`, `libsoup-3.0-dev`,
`libappindicator3-dev`, `librsvg2-dev`, and `patchelf`;
`crates/hocmesh-desktop/BUNDLING.md` has the details.

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
- Each platform publishes a full archive, a headless installer
  (DEB, MSI, or PKG), and the desktop-app installers for that platform, with a
  `.sha256` checksum for each.
- The workflow creates a draft prerelease so checksums, release notes, and
  signing status can be reviewed before publishing.

Reproducible-build notes:

- Do not build from a dirty worktree for public artifacts.
- Use the pinned Rust toolchain instead of local `stable`.
- Avoid embedding local absolute paths, machine-specific timestamps, or private
  validator configuration into release artifacts.
- Preserve `Cargo.lock` for application releases.
