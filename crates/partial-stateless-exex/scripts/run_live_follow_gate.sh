#!/usr/bin/env bash
# The S3 live equivalence gate, consumer side.
#
# Runs `ps-replay --follow` against a spool a producer is writing RIGHT NOW, re-replays the very
# same spool in batch mode, and then ASSERTS on both runs' aggregate records — exit codes alone
# cannot tell "verified 100 witnessed blocks" from "verified nothing at all", and a gate that
# only checked them could pass on an empty, reconstructed, or cut stream.
#
# The producer is NOT started here. It needs the datadir, so starting it is the operator's move:
#     (stop the vanilla node first — it holds the MDBX writer lock)
#     PS_ENGINE_PAYLOAD=on PS_STREAM_DIR=<spool-dir> \
#         <reth-partial-stateless binary> node <the vanilla node's own flags>
#
# Usage:
#     run_live_follow_gate.sh <spool-dir> <output-dir> [extra ps-replay --follow args...]
#
# Environment:
#     GATE_MODE=clean       (default) the SIGTERM run: the stream must close as End(shutdown),
#                           every commit witnessed, batch closed=true.
#     GATE_MODE=truncated   the kill -9 control run: the follower is expected to time out idle
#                           (a killed producer and a quiet chain look identical from files), and
#                           the batch pass must report closed=false. Judged offline, separately
#                           from the success gate. Adds --idle-timeout-secs 60 unless one is
#                           passed.
#     GATE_MIN_BLOCKS=N     minimum verified blocks (default 100 clean, 1 truncated).
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'
    exit 64
fi

SPOOL_DIR=$1
OUT_DIR=$2
shift 2

GATE_MODE=${GATE_MODE:-clean}
case "$GATE_MODE" in
    clean) GATE_MIN_BLOCKS=${GATE_MIN_BLOCKS:-100} ;;
    truncated) GATE_MIN_BLOCKS=${GATE_MIN_BLOCKS:-1} ;;
    *) echo "error: GATE_MODE must be clean or truncated, not '$GATE_MODE'" >&2; exit 64 ;;
esac

command -v jq >/dev/null || { echo "error: jq is required to judge the aggregate records" >&2; exit 1; }

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
if [ -z "${PS_REPLAY_BIN:-}" ]; then
    # The workspace may redirect its target directory; ask cargo rather than guessing.
    TARGET_DIR=$(cd "$REPO_ROOT" && cargo metadata --format-version 1 --no-deps 2>/dev/null |
        sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')
    PS_REPLAY_BIN="${TARGET_DIR:-$REPO_ROOT/target}/release/ps-replay"
fi
if [ ! -x "$PS_REPLAY_BIN" ]; then
    echo "error: $PS_REPLAY_BIN is not built; run: cargo build --release -p partial-stateless-replay" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
FOLLOW_JSON="$OUT_DIR/follow.jsonl"
ACK_FILE="$OUT_DIR/ack.json"
BATCH_JSON="$OUT_DIR/batch.jsonl"

# The truncated run must terminate on its own: the follower deliberately waits forever by
# default, because it cannot tell a killed producer from a quiet chain.
if [ "$GATE_MODE" = "truncated" ] && ! printf '%s\n' "$@" | grep -q -- '--idle-timeout-secs'; then
    set -- "$@" --idle-timeout-secs 60
fi

echo "==> live follow against $SPOOL_DIR (mode: $GATE_MODE, verdicts: $FOLLOW_JSON)"
follow_code=0
"$PS_REPLAY_BIN" --follow "$SPOOL_DIR" \
    --json "$FOLLOW_JSON" --ack "$ACK_FILE" --label "s3-live-gate-$GATE_MODE" "$@" || follow_code=$?
echo "==> follow exit code: $follow_code (0 clean end, 1 disagreement/fault, 2 NeedsSnapshot, 3 ended before checkpoint, 4 idle timeout)"

echo "==> batch re-replay of the same spool (the live stream and the corpus are the same bytes)"
batch_code=0
"$PS_REPLAY_BIN" "$SPOOL_DIR" --json "$BATCH_JSON" --label "s3-live-gate-batch-$GATE_MODE" || batch_code=$?
echo "==> batch exit code: $batch_code"

SUMMARY=$(grep '"kind":"summary"' "$FOLLOW_JSON" 2>/dev/null | tail -1 || true)
if [ -z "$SUMMARY" ]; then
    echo "FAIL: no summary record in $FOLLOW_JSON; the follower never finished a run" >&2
    exit 1
fi
BATCH=$(tail -1 "$BATCH_JSON" 2>/dev/null || true)
if [ -z "$BATCH" ]; then
    echo "FAIL: no batch record in $BATCH_JSON" >&2
    exit 1
fi

FAILED=0
check() { # <record> <label> <jq boolean expression>
    local got
    got=$(printf '%s' "$1" | jq -r "$3" 2>/dev/null || echo error)
    if [ "$got" = "true" ]; then
        echo "  ok: $2"
    else
        echo "  FAIL: $2  [$3]" >&2
        FAILED=1
    fi
}

echo "==> judging the follow summary"
check "$SUMMARY" "verified at least $GATE_MIN_BLOCKS blocks" ".blocks_verified >= $GATE_MIN_BLOCKS"
check "$SUMMARY" "every commit was witnessed" ".witnessed == .blocks_verified"
check "$SUMMARY" "no reconstructed payload" ".reconstructed == 0"
check "$SUMMARY" "no absent payload" ".absent == 0"
check "$SUMMARY" "no disagreement" ".disagreements == 0"
check "$SUMMARY" "no failure" ".failures == 0"
check "$SUMMARY" "exactly one restore, no NeedsSnapshot" ".restores == 1 and .needs_snapshot_entries == 0"
check "$SUMMARY" "the follower itself agreed" ".agreed == true"
case "$GATE_MODE" in
    clean)
        [ "$follow_code" -eq 0 ] && echo "  ok: follow exit 0" || { echo "  FAIL: follow exit $follow_code, wanted 0" >&2; FAILED=1; }
        check "$SUMMARY" "the stream closed as End(shutdown)" \
            '.outcome == "ended" and .end_kind == "shutdown" and .before_checkpoint == false'
        ;;
    truncated)
        [ "$follow_code" -eq 4 ] && echo "  ok: follow exit 4 (idle timeout — a cut stream never says goodbye)" || { echo "  FAIL: follow exit $follow_code, wanted 4" >&2; FAILED=1; }
        check "$SUMMARY" "the follower timed out waiting, not recovering" \
            '.outcome == "idle_timeout" and .waiting_in != "needs_snapshot"'
        ;;
esac

echo "==> judging the batch record"
[ "$batch_code" -eq 0 ] && echo "  ok: batch exit 0" || { echo "  FAIL: batch exit $batch_code, wanted 0" >&2; FAILED=1; }
check "$BATCH" "batch agreed with the recording" ".agreed == true and .terminal == null"
check "$BATCH" "live and batch verified the same commits" ".commits == $(printf '%s' "$SUMMARY" | jq '.blocks_verified')"
check "$BATCH" "batch saw only witnessed payloads" ".witnessed == .commits and .reconstructed == 0 and .absent == 0"
case "$GATE_MODE" in
    clean) check "$BATCH" "the spool is closed (End present)" ".closed == true" ;;
    truncated) check "$BATCH" "the spool is cut (no End) — the point of this control run" ".closed == false" ;;
esac

echo "==> results in $OUT_DIR"
if [ "$FAILED" -ne 0 ]; then
    echo "==> GATE FAILED ($GATE_MODE)" >&2
    exit 1
fi
echo "==> GATE PASSED ($GATE_MODE)"
