#!/usr/bin/env bash
#
# Stand up a whole hocMESH -- a validator quorum, a coordinator, and worker
# nodes -- put artificial load through it, and check the economy survived.
#
# This exists because the interesting failures in this system are not slow
# responses, they are races: two proposers on the same head, a reward applied
# before its head is readable, a claim key settled twice. None of them are
# reachable by one person clicking through a demo, and all of them are reachable
# by a dozen jobs landing at once. So the script's job is to *create* the
# contention and then ask the ledger to prove nothing leaked.
#
# What makes it a test rather than a benchmark: the exit status is decided by
# whether the work settled and whether the CU add up, never by how fast this
# machine happened to be. A speed threshold in CI is a flaky test, and a flaky
# test teaches people to ignore red.
#
#   ./scripts/loadtest-local.sh                  # defaults, ~1 minute
#   ./scripts/loadtest-local.sh --jobs 40 --concurrency 8
#   ./scripts/loadtest-local.sh --keep           # leave it running to poke at
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

JOBS=12
CONCURRENCY=4
SHARDS=4
SIZE=50000
WORKLOAD=collatz
WORKERS=2
DURATION=""
JSON=""
KEEP=0
SKIP_BUILD=0
# Bigger than any plausible run needs. Community work is minted by the sitting
# validators and is the only way a new account gets its first CU, so the seed
# has to cover the load test with room to spare -- an account that runs out
# halfway through fails in a way that looks exactly like the settlement bug
# this script exists to detect.
SEED_END=4000000
SEED_SHARDS=64

while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    --workload) WORKLOAD="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --duration-secs) DURATION="$2"; shift 2 ;;
    --json) JSON="$2"; shift 2 ;;
    --seed-end) SEED_END="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help)
      sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

EXE=""
case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*) EXE=".exe" ;;
esac
BIN="$ROOT/target/release"
NODE="$BIN/hocmesh$EXE"
COORD="$BIN/hocmesh-coordinator$EXE"
VALIDATOR="$BIN/hocmesh-validator$EXE"

if [ "$SKIP_BUILD" -eq 0 ]; then
  echo "==> building release binaries"
  cargo build --release -p hocmesh -p hocmesh-coordinator -p hocmesh-validator
fi
for b in "$NODE" "$COORD" "$VALIDATOR"; do
  [ -x "$b" ] || { echo "missing binary: $b (drop --skip-build?)" >&2; exit 1; }
done

WORK="$ROOT/target/loadtest"
rm -rf "$WORK"
mkdir -p "$WORK"

PIDS=()
cleanup() {
  local status=$?
  if [ "$KEEP" -eq 1 ] && [ $status -eq 0 ]; then
    echo
    echo "--keep: leaving the network up. Coordinator: http://127.0.0.1:${COORD_PORT:-?}"
    echo "PIDs: ${PIDS[*]}"
    return
  fi
  # Reverse order, so workers stop before the coordinator they report to and
  # the coordinator stops before the ledger it settles against. Tearing down
  # the other way produces a screenful of connection errors that look like
  # failures and are not.
  for ((i=${#PIDS[@]}-1; i>=0; i--)); do
    kill "${PIDS[$i]}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT

port_free() { ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
pick_port() {
  local p="$1"
  while ! port_free "$p"; do p=$((p + 1)); done
  echo "$p"
}

wait_health() {
  local port="$1" what="$2" tries=0
  until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    tries=$((tries + 1))
    if [ "$tries" -gt 200 ]; then
      echo "$what never became healthy on port $port" >&2
      tail -n 20 "$WORK/$what.log" >&2 2>/dev/null || true
      exit 1
    fi
    sleep 0.1
  done
}

# ------------------------------------------------------------------ ledger --
# Four validators, threshold three. The set refuses any threshold that is not
# more than two thirds of its membership, which is what makes the ledger a
# quorum rather than a database with extra steps: it survives one seat being
# wrong or gone, and no smaller group can move a balance. Four is the smallest
# membership where that leaves a spare seat.
echo "==> creating a 4-validator set (threshold 3)"
MEMBERS=""
VAL_PORTS=()
PORT=$(pick_port 9301)
for i in 0 1 2 3; do
  PORT=$(pick_port "$PORT")
  VAL_PORTS+=("$PORT")
  home="$WORK/validator-$i"
  out="$("$VALIDATOR" id --home "$home")"
  vid="$(echo "$out" | sed -n 's/^validator_id=//p')"
  pk="$(echo "$out" | sed -n 's/^public_key_b64=//p')"
  [ -n "$vid" ] && [ -n "$pk" ] || { echo "could not read validator $i identity" >&2; exit 1; }
  sep=""; [ -n "$MEMBERS" ] && sep=","
  MEMBERS="$MEMBERS$sep{\"validator_id\":\"$vid\",\"url\":\"http://127.0.0.1:$PORT\",\"public_key_b64\":\"$pk\"}"
  PORT=$((PORT + 1))
done

VALIDATORS="$WORK/validators.json"
cat > "$VALIDATORS" <<EOF
{
  "threshold": 3,
  "community_issuance_limit_mcu": 1000000000,
  "members": [$MEMBERS]
}
EOF

for i in 0 1 2 3; do
  "$VALIDATOR" serve \
    --home "$WORK/validator-$i" \
    --db "$WORK/validator-$i.db" \
    --listen "127.0.0.1:${VAL_PORTS[$i]}" \
    --validators "$VALIDATORS" \
    >"$WORK/validator-$i.log" 2>&1 &
  PIDS+=($!)
done
for i in 0 1 2 3; do wait_health "${VAL_PORTS[$i]}" "validator-$i"; done
echo "    validators up on ${VAL_PORTS[*]}"

# ---------------------------------------------------------------- the seed --
# The sponsorships have to come off the keys that actually sit in the
# validators' homes: minting is the set's decision, and the coordinator can
# only carry signatures it was handed. Signing in-process here would prove
# something no operator ever does.
echo "==> minting community work (2..$SEED_END, $SEED_SHARDS shards)"
SEED_JOB="job_loadtest_seed"
SPONSORS="$WORK/sponsors.json"
{
  printf '['
  for i in 0 1 2 3; do
    [ "$i" -gt 0 ] && printf ','
    "$NODE" --home "$WORK/validator-$i" community-vouch \
      --validators "$VALIDATORS" --job-id "$SEED_JOB" \
      --start 2 --end "$SEED_END" --shards "$SEED_SHARDS" | tail -n 1 | tr -d '\r\n'
  done
  printf ']'
} > "$SPONSORS"

COORD_DB="$WORK/coordinator.db"
"$COORD" seed --db "$COORD_DB" --validators "$VALIDATORS" \
  --job-id "$SEED_JOB" --sponsors "$SPONSORS" \
  --start 2 --end "$SEED_END" --shards "$SEED_SHARDS" >"$WORK/seed.log" 2>&1

COORD_PORT=$(pick_port 9401)
"$COORD" serve --db "$COORD_DB" --listen "127.0.0.1:$COORD_PORT" \
  --validators "$VALIDATORS" >"$WORK/coordinator.log" 2>&1 &
PIDS+=($!)
wait_health "$COORD_PORT" "coordinator"
COORDINATOR="http://127.0.0.1:$COORD_PORT"
echo "    coordinator up on $COORD_PORT"

# ----------------------------------------------------------------- funding --
# What the run will cost is knowable before it starts, from the same pricing
# function the ledger charges with, so wait for exactly that much rather than
# guessing a sleep. A run that begins underfunded fails at settlement and looks
# indistinguishable from a real bug.
DRY=$("$NODE" --home "$WORK/node-a" loadtest --coordinator "$COORDINATOR" --dry-run \
  --jobs "$JOBS" --concurrency "$CONCURRENCY" --shards "$SHARDS" \
  --workload "$WORKLOAD" --size "$SIZE" 2>/dev/null)
NEED_MCU=$(echo "$DRY" | sed -n 's/^total_mcu=//p')
[ -n "$NEED_MCU" ] || { echo "could not price the run" >&2; exit 1; }
echo "==> this run will cost ${NEED_MCU} mCU; earning it first"

"$NODE" --home "$WORK/node-a" init --coordinator "$COORDINATOR" >/dev/null
"$NODE" --home "$WORK/node-a" daemon --coordinator "$COORDINATOR" \
  --workers "$WORKERS" --no-control >"$WORK/node-a.log" 2>&1 &
EARNER_PID=$!
PIDS+=("$EARNER_PID")

banked_mcu() {
  "$NODE" --home "$WORK/node-a" balance --coordinator "$COORDINATOR" 2>/dev/null \
    | sed -n 's/^Banked: \([0-9.]*\) CU/\1/p' \
    | awk '{printf "%d", $1 * 1000 + 0.5}'
}

# Only the requester works this stage, so it takes the whole seed and reaches
# solvency in a handful of shards rather than racing two other nodes for it.
tries=0
while :; do
  have=$(banked_mcu)
  have=${have:-0}
  if [ "$have" -ge "$NEED_MCU" ]; then break; fi
  tries=$((tries + 1))
  if [ "$tries" -gt 600 ]; then
    echo "node-a only earned ${have} of the ${NEED_MCU} mCU it needs" >&2
    tail -n 20 "$WORK/node-a.log" >&2
    exit 1
  fi
  sleep 0.5
done
echo "    node-a banked ${have} mCU"

# Stop earning before spending. During the load test node-a is a requester, and
# a requester whose own daemon is draining leftover community work in the
# background makes the accounting harder to read for no benefit -- the CU
# invariants would still hold, but a human staring at the report would have to
# work out why.
kill "$EARNER_PID" 2>/dev/null || true
wait "$EARNER_PID" 2>/dev/null || true

# ----------------------------------------------------------------- workers --
echo "==> starting 2 worker nodes, $WORKERS threads each"
for n in b c; do
  home="$WORK/node-$n"
  "$NODE" --home "$home" init --coordinator "$COORDINATOR" >/dev/null
  "$NODE" --home "$home" daemon --coordinator "$COORDINATOR" \
    --workers "$WORKERS" --no-control >"$WORK/node-$n.log" 2>&1 &
  PIDS+=($!)
done

# -------------------------------------------------------------------- load --
echo "==> running the load test"
echo
ARGS=(--coordinator "$COORDINATOR" --jobs "$JOBS" --concurrency "$CONCURRENCY"
      --shards "$SHARDS" --workload "$WORKLOAD" --size "$SIZE")
[ -n "$DURATION" ] && ARGS+=(--duration-secs "$DURATION")
[ -n "$JSON" ] && ARGS+=(--json "$JSON")

set +e
"$NODE" --home "$WORK/node-a" loadtest "${ARGS[@]}"
STATUS=$?
set -e

echo
if [ $STATUS -ne 0 ]; then
  echo "LOAD TEST FAILED. Logs are in $WORK/"
  exit $STATUS
fi

# The run passing means the coordinator's arithmetic held. Auditing the ledger
# from genesis is a different claim, and the stronger one: every entry the
# quorum certified re-verified from the first, against the validator set that
# was sitting at the time.
echo "==> auditing the ledger from genesis"
"$NODE" --home "$WORK/node-a" ledger-sync --validators "$VALIDATORS" \
  --db "$WORK/mirror.db" --coordinator "$COORDINATOR" || {
    echo "ledger audit FAILED after a passing load test -- that is the bad case" >&2
    exit 1
  }

echo
echo "PASSED: work settled, CU conserved, ledger audits from genesis."
