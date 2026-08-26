#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release -p hocmesh
DEST="${HOME}/.local/bin"
mkdir -p "$DEST"
cp target/release/hocmesh "$DEST/hocmesh"
chmod 755 "$DEST/hocmesh"
echo "Installed hocMESH participant client to $DEST/hocmesh"
echo "Ensure $DEST is in PATH, then run: hocmesh init"
