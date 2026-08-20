#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release --workspace
rm -rf dist
mkdir -p dist/docs dist/config
cp target/release/mesh dist/ 2>/dev/null || true
cp target/release/mesh-coordinator dist/ 2>/dev/null || true
cp target/release/mesh-validator dist/ 2>/dev/null || true
cp README.md CODEX_HANDOFF.md LICENSE dist/
cp docs/*.md dist/docs/
cp config/*.json dist/config/
echo "Release folder: $(pwd)/dist"
