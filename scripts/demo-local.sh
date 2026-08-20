#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cargo build --release --workspace
rm -f mesh-demo.db mesh-demo.db-shm mesh-demo.db-wal
rm -rf node-a node-b node-c

./target/release/mesh-coordinator seed --db mesh-demo.db --start 2 --end 5000000 --shards 64

echo
echo "Next:" 
echo "  Terminal 1: ./target/release/mesh-coordinator serve --db mesh-demo.db --listen 127.0.0.1:8080"
echo "  Terminal 2: ./target/release/mesh --home node-a init && ./target/release/mesh --home node-a daemon --workers 2"
echo "  Terminal 3: ./target/release/mesh --home node-b init && ./target/release/mesh --home node-b daemon --workers 2"
echo "  Terminal 4: ./target/release/mesh --home node-c init && ./target/release/mesh --home node-c daemon --workers 2"
