#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release --workspace
rm -rf dist
mkdir -p dist/docs dist/config
cp target/release/hocmesh dist/ 2>/dev/null || true
cp target/release/hocmesh-coordinator dist/ 2>/dev/null || true
cp target/release/hocmesh-validator dist/ 2>/dev/null || true
cp README.md CODEX_HANDOFF.md LICENSE VERSION dist/
cp docs/*.md dist/docs/
cp config/*.json dist/config/
archive="hocmesh-$(tr -d '[:space:]' < VERSION)-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m).tar.gz"
rm -f "$archive" "$archive.sha256"
tar -czf "$archive" dist
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$archive" > "$archive.sha256"
else
    shasum -a 256 "$archive" > "$archive.sha256"
fi
echo "Release folder: $(pwd)/dist"
echo "Release archive: $(pwd)/$archive"
