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
#                           every commit witnessed, batch closed=true, and nothing in the
#                           lifecycle — a run that undid a reorg is judged under GATE_MODE=reorg.
#     GATE_MODE=reorg       the run held open until mainnet reorgs (roughly hourly): the follower
#                           must undo it in place and keep publishing — no NeedsSnapshot, no
#                           re-bootstrap, continuous. Set GATE_EXPECT_RECHECKPOINT=1 when the
#                           producer runs with PS_STREAM_REORG_CHECKPOINT=always (the default) to
#                           require that its recovery checkpoint was skimmed and agreed with.
#                           Set GATE_FORCE_RESTORE=1 to add the offline proof that the recovery
#                           checkpoint really does rebootstrap a validator: the batch driver
#                           installs it and replays the winning branch against it.
#     GATE_MODE=resume      the restart run: the follower is killed mid-stream and started again
#                           with --resume. It must come back to the checkpoint its ack names,
#                           re-derive the blocks between, land on the same block the ack recorded,
#                           and publish new verdicts only above the watermark it left.
#                           GATE_KILL_AFTER_SECS controls when the first run is killed (default 90).
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
    clean|reorg) GATE_MIN_BLOCKS=${GATE_MIN_BLOCKS:-100} ;;
    resume) GATE_MIN_BLOCKS=${GATE_MIN_BLOCKS:-1} ;;
    truncated) GATE_MIN_BLOCKS=${GATE_MIN_BLOCKS:-1} ;;
    *)
        echo "error: GATE_MODE must be clean, reorg, resume or truncated, not '$GATE_MODE'" >&2
        exit 64
        ;;
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

KILLED_AT_SEQUENCE=""
if [ "$GATE_MODE" = "resume" ]; then
    # The restart is the thing under test, so the gate performs it. SIGKILL rather than SIGTERM:
    # a follower given the chance to close cleanly would prove nothing about crash recovery.
    KILL_AFTER=${GATE_KILL_AFTER_SECS:-90}
    echo "==> live follow against $SPOOL_DIR (first run, killed after ${KILL_AFTER}s)"
    "$PS_REPLAY_BIN" --follow "$SPOOL_DIR" \
        --json "$FOLLOW_JSON" --ack "$ACK_FILE" --label "s3-live-gate-resume-first" "$@" &
    first_pid=$!
    sleep "$KILL_AFTER"
    kill -9 "$first_pid" 2>/dev/null || true
    wait "$first_pid" 2>/dev/null || true
    if [ ! -s "$ACK_FILE" ]; then
        echo "FAIL: the first run left no ack; there is nothing to resume from" >&2
        exit 1
    fi
    KILLED_AT_SEQUENCE=$(jq -r '.last_sequence' "$ACK_FILE")
    echo "==> killed with the ack at sequence $KILLED_AT_SEQUENCE"
    # A separate verdict file, so "which lines did the *resumed* run publish" is answerable.
    FOLLOW_JSON="$OUT_DIR/follow-resumed.jsonl"
    set -- "$@" --resume
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
# The second axis, asserted everywhere: agreeing on every block it looked at is not the same as
# having looked at every canonical block, and only this distinguishes them.
check "$SUMMARY" "no canonical block went unverified" ".continuous == true"
check "$SUMMARY" "every announced branch was delivered in full" ".winning_branches_incomplete == 0"
check "$SUMMARY" "no recovery landed somewhere it did not ask for" ".restores_reset == 0"
check "$SUMMARY" "nothing in the recovery scan had to be refused" ".scan_refusals == 0"
case "$GATE_MODE" in
    clean)
        [ "$follow_code" -eq 0 ] && echo "  ok: follow exit 0" || { echo "  FAIL: follow exit $follow_code, wanted 0" >&2; FAILED=1; }
        check "$SUMMARY" "the stream closed as End(shutdown)" \
            '.outcome == "ended" and .end_kind == "shutdown" and .before_checkpoint == false'
        # What makes "one restore, no NeedsSnapshot" mean something here: the chain did not move
        # under this run at all, so there was nothing for the lifecycle to handle. A run that did
        # undo a reorg is a different claim and is judged under GATE_MODE=reorg.
        check "$SUMMARY" "the chain never moved under this run" \
            '.reorgs_applied == 0 and .reverts_applied == 0'
        ;;
    reorg)
        [ "$follow_code" -eq 0 ] && echo "  ok: follow exit 0" || { echo "  FAIL: follow exit $follow_code, wanted 0" >&2; FAILED=1; }
        check "$SUMMARY" "the stream closed as End(shutdown)" \
            '.outcome == "ended" and .end_kind == "shutdown" and .before_checkpoint == false'
        # The headline. A depth-1 reorg is undone against the retained generation with no database
        # and no rebootstrap, and verdicts never stop — which is what "one restore, no
        # NeedsSnapshot" above is now asserting *through* a real chain reorganisation.
        check "$SUMMARY" "at least one reorg or revert was applied in place" \
            '.reorgs_applied + .reverts_applied >= 1'
        if [ "${GATE_EXPECT_RECHECKPOINT:-0}" = "1" ]; then
            # The producer re-checkpoints at the common ancestor. The follower does not install
            # it — it already recovered — but it compares it field by field, which makes the
            # producer's own recovery a live cross-implementation check rather than an assertion.
            check "$SUMMARY" "the producer's recovery checkpoint was skimmed" \
                '.checkpoints_skimmed >= 1'
            check "$SUMMARY" "and agreed with the generation this follower recovered to" \
                '.disagreements == 0'
        fi
        ;;
    resume)
        [ "$follow_code" -eq 0 ] && echo "  ok: follow exit 0" || { echo "  FAIL: follow exit $follow_code, wanted 0" >&2; FAILED=1; }
        check "$SUMMARY" "the resumed run rebuilt from the checkpoint its ack named" \
            '.resumed_from != null'
        check "$SUMMARY" "and re-derived the blocks between it and the watermark" \
            '.catch_up_blocks >= 1'
        # The claim a restart has to earn: every verdict it published as its own is above the
        # watermark the killed run left. Anything at or below it was re-derived, and re-derived
        # lines say so.
        NEW_BELOW=$(jq -r --argjson at "$KILLED_AT_SEQUENCE" \
            'select(.kind == "verdict" and .catch_up == false and .sequence <= $at) | .sequence' \
            "$FOLLOW_JSON" | wc -l)
        if [ "$NEW_BELOW" -eq 0 ]; then
            echo "  ok: no new verdict was published below the watermark (sequence $KILLED_AT_SEQUENCE)"
        else
            echo "  FAIL: $NEW_BELOW verdicts at or below sequence $KILLED_AT_SEQUENCE were not labelled catch-up" >&2
            FAILED=1
        fi
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
check "$BATCH" "batch verified every canonical block the corpus carried" ".continuous == true"
check "$BATCH" "batch saw only witnessed payloads" ".witnessed == .commits and .reconstructed == 0 and .absent == 0"
case "$GATE_MODE" in
    clean)
        check "$BATCH" "live and batch verified the same commits" \
            ".commits == $(printf '%s' "$SUMMARY" | jq '.blocks_verified')"
        check "$BATCH" "the spool is closed (End present)" ".closed == true"
        ;;
    reorg)
        # Not equal to the follower's count: the batch driver replays the abandoned block too,
        # then undoes it, while the follower verified it during the first run. What has to match
        # is the conclusion, which is what the axes above assert.
        check "$BATCH" "batch replayed the reorg rather than past it" ".reorgs_applied + .reverts_applied >= 1"
        check "$BATCH" "batch reached the end of the corpus" ".complete == true and .closed == true"
        ;;
    resume)
        # The corpus is one stream whichever way it is read, so a batch replay of it has to reach
        # the same conclusion as two follower runs stitched together.
        check "$BATCH" "the spool is closed (End present)" ".closed == true"
        ;;
    truncated) check "$BATCH" "the spool is cut (no End) — the point of this control run" ".closed == false" ;;
esac

if [ "$GATE_MODE" = "reorg" ] && [ "${GATE_FORCE_RESTORE:-0}" = "1" ]; then
    # Skimming a recovery checkpoint proves the two implementations agree about the block. It does
    # not prove the snapshot behind it would restore a validator that has no retained generation
    # to recover with — the one consumer that most needs it. This installs it and replays the
    # winning branch against it, which is the only thing that does prove it.
    RESTORE_AT=$(grep '"kind":"state"' "$FOLLOW_JSON" 2>/dev/null |
        jq -rs 'map(select(.reason == "restored")) | .[-1].sequence // empty' 2>/dev/null || true)
    RESTORE_AT=${RESTORE_AT:-$(jq -r '.restored_from_sequence // empty' "$ACK_FILE" 2>/dev/null || true)}
    if [ -z "$RESTORE_AT" ]; then
        echo "  FAIL: GATE_FORCE_RESTORE=1 but no recovery checkpoint sequence was recorded" >&2
        FAILED=1
    else
        echo "==> forcing a restore from the recovery checkpoint at sequence $RESTORE_AT"
        FORCED_JSON="$OUT_DIR/batch-forced.jsonl"
        forced_code=0
        "$PS_REPLAY_BIN" "$SPOOL_DIR" --force-restore-at "$RESTORE_AT" \
            --json "$FORCED_JSON" --label "s3-live-gate-forced" || forced_code=$?
        FORCED=$(tail -1 "$FORCED_JSON" 2>/dev/null || true)
        [ "$forced_code" -eq 0 ] && echo "  ok: forced-restore batch exit 0" ||
            { echo "  FAIL: forced-restore batch exit $forced_code" >&2; FAILED=1; }
        if [ -n "$FORCED" ]; then
            check "$FORCED" "the recovery checkpoint really does rebootstrap a validator" \
                ".agreed == true and .restores_forced != false"
        fi
    fi
fi

echo "==> results in $OUT_DIR"
if [ "$FAILED" -ne 0 ]; then
    echo "==> GATE FAILED ($GATE_MODE)" >&2
    exit 1
fi
echo "==> GATE PASSED ($GATE_MODE)"
