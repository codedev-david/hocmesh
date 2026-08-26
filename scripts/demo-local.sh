#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cargo build --release --workspace
rm -f hocmesh-demo.db hocmesh-demo.db-shm hocmesh-demo.db-wal
rm -rf node-a node-b node-c

./target/release/hocmesh-coordinator seed --db hocmesh-demo.db --start 2 --end 5000000 --shards 64

echo
echo "Next:" 
echo "  Terminal 1: ./target/release/hocmesh-coordinator serve --db hocmesh-demo.db --listen 127.0.0.1:8080"
echo "  Terminal 2: ./target/release/hocmesh --home node-a init && ./target/release/hocmesh --home node-a daemon --workers 2"
echo "  Terminal 3: ./target/release/hocmesh --home node-b init && ./target/release/hocmesh --home node-b daemon --workers 2"
echo "  Terminal 4: ./target/release/hocmesh --home node-c init && ./target/release/hocmesh --home node-c daemon --workers 2"
