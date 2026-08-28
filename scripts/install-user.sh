#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release -p hocmesh -p hocmesh-coordinator -p hocmesh-validator
DEST="${HOME}/.local/bin"
mkdir -p "$DEST"
for exe in hocmesh hocmesh-coordinator hocmesh-validator; do
  install -m 0755 "target/release/$exe" "$DEST/$exe"
done
echo "Installed the hocMESH peer to $DEST"
echo "Ensure $DEST is in PATH, then run: hocmesh init"
