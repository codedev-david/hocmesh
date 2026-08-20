$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
