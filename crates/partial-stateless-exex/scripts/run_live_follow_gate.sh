#!/usr/bin/env bash
# The S3 live equivalence gate, consumer side.
#
# Runs `ps-replay --follow` against a spool a producer is writing RIGHT NOW, and when the follow
# ends, re-replays the very same spool in batch mode — the live stream and the deterministic
# corpus are the same bytes, and this is where that design promise is exercised rather than
# assumed.
#
# The producer is NOT started here. It needs the datadir, so starting it is the operator's move:
#     (stop the vanilla node first — it holds the MDBX writer lock)
#     PS_ENGINE_PAYLOAD=on PS_STREAM_DIR=<spool-dir> \
#         <reth-partial-stateless binary> node <the vanilla node's own flags>
#
# Usage:
#     run_live_follow_gate.sh <spool-dir> <output-dir> [extra ps-replay --follow args...]
#
# The follower waits for the producer's End frame by default (SIGTERM the producer to close the
# stream); pass --max-blocks N or --idle-timeout-secs N to bound the wait. A kill -9 control run
# is judged by the batch pass afterwards reporting closed=false — the follower deliberately
# cannot tell a killed producer from a quiet chain and will wait, so bound it for that run.
#
# Success is all of:
#   - follow exit code 0, with blocks_verified > 0 and zero disagreements/failures
#   - every commit witnessed (admission was checking something on every block)
#   - batch re-replay exit code 0 over the same spool, closed=true after a SIGTERM stop
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
    exit 64
fi

SPOOL_DIR=$1
OUT_DIR=$2
shift 2

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PS_REPLAY_BIN=${PS_REPLAY_BIN:-"$REPO_ROOT/target/release/ps-replay"}
if [ ! -x "$PS_REPLAY_BIN" ]; then
    echo "error: $PS_REPLAY_BIN is not built; run: cargo build --release -p partial-stateless-replay" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
FOLLOW_JSON="$OUT_DIR/follow.jsonl"
ACK_FILE="$OUT_DIR/ack.json"
BATCH_JSON="$OUT_DIR/batch.jsonl"

echo "==> live follow against $SPOOL_DIR (verdicts: $FOLLOW_JSON, ack: $ACK_FILE)"
follow_code=0
"$PS_REPLAY_BIN" --follow "$SPOOL_DIR" \
    --json "$FOLLOW_JSON" --ack "$ACK_FILE" --label s3-live-gate "$@" || follow_code=$?
echo "==> follow exit code: $follow_code (0 clean end, 1 disagreement/fault, 2 NeedsSnapshot, 3 ended before checkpoint, 4 idle timeout)"

echo "==> batch re-replay of the same spool (the live stream and the corpus are the same bytes)"
batch_code=0
"$PS_REPLAY_BIN" "$SPOOL_DIR" --json "$BATCH_JSON" --label s3-live-gate-batch || batch_code=$?
echo "==> batch exit code: $batch_code"

echo "==> results in $OUT_DIR"
exit $(( follow_code != 0 ? follow_code : batch_code ))
