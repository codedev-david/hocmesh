$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
cargo fmt --all -- --check
cargo check --workspace --all-features --locked
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
