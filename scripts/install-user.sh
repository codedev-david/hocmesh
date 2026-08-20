#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release -p mesh
DEST="${HOME}/.local/bin"
mkdir -p "$DEST"
cp target/release/mesh "$DEST/mesh"
chmod 755 "$DEST/mesh"
echo "Installed MESH participant client to $DEST/mesh"
echo "Ensure $DEST is in PATH, then run: mesh init"
