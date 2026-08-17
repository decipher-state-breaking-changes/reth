#!/usr/bin/env bash
# W5.2 + W5.3 — score the cache windows offline, then estimate what each one would cost to carry.
#
#     run_window_sweep.sh <fixtures-dir> <out-dir> [producer.out ...]
#
# The fixtures come from a capture run (`PS_CAPTURE_DIR`); the producer logs are runs that really
# built proofs, and are what turns a miss count into an estimated byte count. Give at least two
# from *different* windows: the first is fitted, the rest are held out, and a size model with
# nothing held out has no error bar.
#
# Neither half needs a datadir, a node, or a governor. The sweep replays recorded access sets
# through the eviction policy — the same policy object the node runs — and counts. Nothing is
# timed, so this is reproducible from the fixtures alone.
#
# The grid stops at 120 by default because the warm-up is the largest window in it: a 500-block
# capture scored against a 240-block window measures only 260 blocks, and the widest arm is then
# the least trustworthy row in the table. Widen the grid when the capture is long enough to pay
# for it.
set -euo pipefail

FIXTURES=${1:?usage: run_window_sweep.sh <fixtures-dir> <out-dir> [producer.out ...]}
OUT_DIR=${2:?usage: run_window_sweep.sh <fixtures-dir> <out-dir> [producer.out ...]}
shift 2

ACCOUNT_WINDOWS=${ACCOUNT_WINDOWS:-8,15,30,60,90,120}
STORAGE_WINDOWS=${STORAGE_WINDOWS:-4,8,15,30,45,60}
BASELINE=${BASELINE:-60,30}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$OUT_DIR"
echo "==> sweeping $(ls "$FIXTURES" | wc -l) fixtures over ${ACCOUNT_WINDOWS} x ${STORAGE_WINDOWS}"
( cd "$REPO" && cargo run --release -p partial-stateless --bin cache_window_bench -- \
    --fixtures "$FIXTURES" \
    --account-windows "$ACCOUNT_WINDOWS" \
    --storage-windows "$STORAGE_WINDOWS" \
    --baseline "$BASELINE" \
    --out "$OUT_DIR/sweep.csv" ) | tee "$OUT_DIR/sweep.txt"

if [ "$#" -eq 0 ]; then
    echo "==> no producer log given: the sweep stands alone, with no size estimate"
    exit 0
fi

FIT=$1; shift
ESTIMATE_ARGS=(--sweep "$OUT_DIR/sweep.csv" --fit "$FIT" --baseline "$BASELINE")
for held_out in "$@"; do ESTIMATE_ARGS+=(--check "$held_out"); done

echo "==> estimating witness size (fit: $FIT, held out: ${*:-none})"
python3 "$SCRIPT_DIR/estimate_witness_bytes.py" "${ESTIMATE_ARGS[@]}" \
    --json "$OUT_DIR/window-estimate.json" --out "$OUT_DIR/window-estimate.md"
echo "==> $OUT_DIR/sweep.csv, $OUT_DIR/window-estimate.md"
