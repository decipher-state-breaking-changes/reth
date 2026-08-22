#!/usr/bin/env bash
# The live equivalence gate, consumer side.
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
#                           and publish new verdicts only above the watermark it left. The kill
#                           waits for the first run to be streaming and GATE_KILL_AFTER_BLOCKS
#                           verdicts in (default 20, GATE_KILL_TIMEOUT_SECS 900): a fixed delay
#                           can land before the snapshot export finishes, and killing a follower
#                           that has not verified anything proves nothing about resuming.
#     GATE_MODE=epoch       the producer-restart run: the producer is stopped, restarted with
#                           PS_STREAM_RESUME=1 (the operator's move — it holds the datadir), and
#                           the follower is started again with --resume. The boundary is a
#                           *deliberate* reset, so this mode does not assert continuity: an epoch
#                           says the producer's state broke, and the interval across it is
#                           unvalidated by construction. What it does assert is that the crossing
#                           was recognised, checked, and reported as the reset it is.
#     GATE_MODE=truncated   the kill -9 control run: the follower is expected to time out idle
#                           (a killed producer and a quiet chain look identical from files), and
#                           the batch pass must report closed=false. Judged offline, separately
#                           from the success gate. Adds --idle-timeout-secs 60 unless one is
#                           passed.
#     GATE_MODE=long        the multi-hour cohort run: clean's assertions minus the reorg-zero
#                           requirement, because a run held open for many hours will legitimately
#                           meet reorgs (roughly hourly at best) and clean would fail on the first
#                           one. Reorg/revert counts are reported as observations; when any
#                           occurred, every winning branch must have completed and (with
#                           GATE_EXPECT_RECHECKPOINT=1) every recovery checkpoint must have been
#                           skimmed and agreed with. A reorg past the retained generation's depth
#                           is also an observation, not a failure: the follower must ask for a
#                           snapshot and recover *continuously* from the producer's checkpoint —
#                           what is asserted is that every NeedsSnapshot was answered that way,
#                           never that none occurred. Default GATE_MIN_BLOCKS is 6000 — the
#                           matrix-row-5 scale.
#
#                           Long refuses --max-blocks: a follower stopped by a block bound exits
#                           before the producer's End, so the spool never closes and the batch
#                           gate below could not run on a closed corpus. Instead the follower
#                           runs unbounded while this script watches the verdict stream; when
#                           GATE_MIN_BLOCKS live (tail_live) verdicts have been published it
#                           SIGTERMs the producer named by GATE_PRODUCER_PID — or prints the
#                           instruction to, if none was named — and the follower then consumes
#                           the End(shutdown) the producer writes on its way out. On an already
#                           closed corpus (the offline validation of this mode) the follower
#                           simply reads to the End and exits: zero live verdicts, no signal.
#                           GATE_REQUIRE_LIVE=1 additionally asserts the live count reached
#                           GATE_MIN_BLOCKS — set it on the real capture cohort, where "live"
#                           is the claim; leave it unset when validating offline.
#     GATE_PRODUCER_PID=P   long only: the producer to SIGTERM when the live target is reached.
#     GATE_REQUIRE_LIVE=1   long only: fail unless live (tail_live) verdicts >= GATE_MIN_BLOCKS.
#     GATE_WATCH_SECS=N     long only: watcher poll interval (default 15).
#     GATE_END_GRACE_SECS=N long only: how long to keep waiting for the producer's End after
#                           the producer process itself is gone (default 300). Past it the
#                           follower is abandoned and the corpus judged truncated, so a
#                           producer that died without closing fails its arm instead of
#                           holding the gate open indefinitely.
#     GATE_MIN_BLOCKS=N     minimum verified blocks (default 100 clean, 6000 long, 1 truncated).
#     GATE_PHASE=capture    run only the online half: everything up to the follower's exit, then
#                           stop. The datadir is free the moment this returns, so the node that
#                           owns it can be restarted before the offline half runs. Writes
#                           <out>/gate-capture.env, which is the judge's input.
#     GATE_PHASE=judge      run only the offline half against an existing capture: batch
#                           re-replay, every assertion, the forced restore, the analyzer and the
#                           renderer. Opens no database and needs no producer. Refuses a bar or a
#                           policy switch that disagrees with the capture's own.
#     GATE_ALLOW_REJUDGE_OVERRIDE=1
#                           judge only: re-judge an archived capture under a different bar on
#                           purpose, and say so in the log.
#     GATE_PHASE=all        (default) both halves back to back, as before.
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'
    exit 64
fi

SPOOL_DIR=$1
OUT_DIR=$2
shift 2

GATE_MODE=${GATE_MODE:-clean}
# Kept before the defaults below so `judge` can tell "the operator asked for this bar" from
# "the mode's default filled it in": judging a 300-verdict capture under long's default of 6000
# would fail a run that met the target it was actually given.
GATE_MIN_BLOCKS_REQUESTED=${GATE_MIN_BLOCKS:-}
case "$GATE_MODE" in
    clean|reorg) GATE_MIN_BLOCKS=${GATE_MIN_BLOCKS:-100} ;;
    long) GATE_MIN_BLOCKS=${GATE_MIN_BLOCKS:-6000} ;;
    resume|epoch) GATE_MIN_BLOCKS=${GATE_MIN_BLOCKS:-1} ;;
    truncated) GATE_MIN_BLOCKS=${GATE_MIN_BLOCKS:-1} ;;
    *)
        echo "error: GATE_MODE must be clean, reorg, resume, epoch, long or truncated, not '$GATE_MODE'" >&2
        exit 64
        ;;
esac

# Two halves, because the offline one has no business holding the datadir hostage. Everything up
# to the follower's exit needs the producer alive and writing; everything after it reads files
# and nothing else. On a long run the batch re-replay and the forced restore take tens of
# minutes — 56 of them after a 2,890-block reorg run — and whatever holds the datadir next then
# has to backfill that whole gap at full speed with an ExEx building a multiproof per block,
# which is how MDBX times a reader out and panics it. Split the halves and the node goes back up
# first.
GATE_PHASE=${GATE_PHASE:-all}
case "$GATE_PHASE" in
    all|capture|judge) ;;
    *)
        echo "error: GATE_PHASE must be all, capture or judge, not '$GATE_PHASE'" >&2
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

# One invocation, one run. A stale follow.jsonl would satisfy the live-verdict watcher and the
# summary greps with a run that already happened, and a stale ack would turn a fresh phase into
# an accidental resume. The one deliberate exception is epoch mode, whose *input* is the
# previous invocation's ack and whose own phase writes a separate file — only that file must
# be fresh there.
CAPTURE_ENV="$OUT_DIR/gate-capture.env"
if [ "$GATE_PHASE" = "judge" ]; then
    # Mirror image: the capture's outputs are this phase's *input* and must be present, while the
    # judge's own must not — a second judge over the first one's batch record would be asserting
    # on a replay that already happened.
    if [ ! -s "$CAPTURE_ENV" ]; then
        echo "error: $CAPTURE_ENV is missing; GATE_PHASE=judge reads what GATE_PHASE=capture left" >&2
        exit 64
    fi
    for stale in "$BATCH_JSON" "$OUT_DIR/batch-forced.jsonl"; do
        if [ -e "$stale" ]; then
            echo "error: $stale already exists; this capture has already been judged" >&2
            exit 64
        fi
    done
elif [ "$GATE_MODE" = "epoch" ]; then
    if [ -e "$OUT_DIR/follow-epoch.jsonl" ]; then
        echo "error: $OUT_DIR/follow-epoch.jsonl already exists; a previous epoch crossing was already judged here" >&2
        exit 64
    fi
else
    for stale in "$FOLLOW_JSON" "$ACK_FILE" "$BATCH_JSON" "$OUT_DIR/follow-resumed.jsonl"; do
        if [ -e "$stale" ]; then
            echo "error: $stale already exists; give the gate a fresh OUT_DIR — stale outputs corrupt the watcher and the summary selection" >&2
            exit 64
        fi
    done
fi

# ==== the online half: the producer is alive from here to the follower's exit ================
if [ "$GATE_PHASE" = "judge" ]; then
    # Everything the assertions below read out of the run, restored from the capture rather than
    # from this process — which never started a follower.
    # shellcheck disable=SC1090
    . "$CAPTURE_ENV"
    if [ "${CAPTURE_SCHEMA:-0}" != "1" ]; then
        echo "error: $CAPTURE_ENV is schema '${CAPTURE_SCHEMA:-}', which this gate cannot read" >&2
        exit 64
    fi
    if [ "$CAPTURE_GATE_MODE" != "$GATE_MODE" ]; then
        echo "error: the capture was taken under GATE_MODE=$CAPTURE_GATE_MODE, not $GATE_MODE" >&2
        exit 64
    fi
    if [ "$CAPTURE_SPOOL_DIR" != "$SPOOL_DIR" ]; then
        echo "error: the capture was taken against $CAPTURE_SPOOL_DIR, not $SPOOL_DIR" >&2
        exit 64
    fi
    FOLLOW_JSON=$CAPTURE_FOLLOW_JSON
    follow_code=$CAPTURE_FOLLOW_CODE
    LIVE_VERDICTS=$CAPTURE_LIVE_VERDICTS
    KILLED_AT_SEQUENCE=$CAPTURE_KILLED_AT_SEQUENCE
    ACK_EPOCH_BEFORE=$CAPTURE_ACK_EPOCH_BEFORE
    # The capture's bar and policy switches carry over, and an environment variable that disagrees
    # with them stops the run. A gate whose standard can be chosen after the numbers are known is
    # not a gate: splitting it in two must not turn "how many blocks did it have to verify" into a
    # question answered at judging time. Re-judging an archive under a different bar is a real need
    # and a separate, named act.
    reject_override() { # <name> <requested> <captured>
        [ -z "$2" ] && return 0
        [ "$2" = "$3" ] && return 0
        if [ "${GATE_ALLOW_REJUDGE_OVERRIDE:-0}" = "1" ]; then
            echo "warning: re-judging with $1=$2 where the capture was taken under $3" >&2
            return 0
        fi
        echo "error: $1=$2 differs from the capture's $1=$3." >&2
        echo "       The capture was judged against its own bar; changing it now would be" >&2
        echo "       choosing the standard after seeing the run. Set" >&2
        echo "       GATE_ALLOW_REJUDGE_OVERRIDE=1 to re-judge an archived capture deliberately." >&2
        exit 64
    }
    reject_override GATE_MIN_BLOCKS "${GATE_MIN_BLOCKS_REQUESTED:-}" "$CAPTURE_MIN_BLOCKS"
    reject_override GATE_EXPECT_RECHECKPOINT "${GATE_EXPECT_RECHECKPOINT:-}" "$CAPTURE_EXPECT_RECHECKPOINT"
    reject_override GATE_REQUIRE_LIVE "${GATE_REQUIRE_LIVE:-}" "$CAPTURE_REQUIRE_LIVE"
    reject_override GATE_FORCE_RESTORE "${GATE_FORCE_RESTORE:-}" "$CAPTURE_FORCE_RESTORE"
    GATE_MIN_BLOCKS=${GATE_MIN_BLOCKS_REQUESTED:-$CAPTURE_MIN_BLOCKS}
    GATE_EXPECT_RECHECKPOINT=${GATE_EXPECT_RECHECKPOINT:-$CAPTURE_EXPECT_RECHECKPOINT}
    GATE_REQUIRE_LIVE=${GATE_REQUIRE_LIVE:-$CAPTURE_REQUIRE_LIVE}
    GATE_FORCE_RESTORE=${GATE_FORCE_RESTORE:-$CAPTURE_FORCE_RESTORE}
    echo "==> judging the capture in $OUT_DIR (mode: $GATE_MODE, follow exit $follow_code, verdicts: $FOLLOW_JSON)"
else

# The truncated run must terminate on its own: the follower deliberately waits forever by
# default, because it cannot tell a killed producer from a quiet chain.
if [ "$GATE_MODE" = "truncated" ] && ! printf '%s\n' "$@" | grep -q -- '--idle-timeout-secs'; then
    set -- "$@" --idle-timeout-secs 60
fi

# The follower starts only against a structurally ready spool: manifest chain verified, the
# *current* epoch's checkpoint present with every declared chunk hashing to its declaration, and
# an accepted head it can admit H+1 under. Both 1,001-block preflights started against an empty
# pre-checkpoint spool and mislabeled the restore backlog as live latency; a file-existence probe
# would repeat that mistake on a resumed spool, whose previous epochs hold checkpoints of their
# own. Exit 2 means "not written yet" and is polled; exit 3 means the spool is structurally
# wrong — a broken chain, a corrupt chunk, an epoch closed checkpointless — and waiting through
# the deadline would only report the corruption as a timeout, so it fails immediately.
GATE_WAIT_CHECKPOINT_SECS=${GATE_WAIT_CHECKPOINT_SECS:-1200}
READY_DEADLINE=$(( $(date +%s) + GATE_WAIT_CHECKPOINT_SECS ))
while :; do
    set +e
    READY_JSON=$("$PS_REPLAY_BIN" --inspect-ready "$SPOOL_DIR" 2>/dev/null)
    READY_CODE=$?
    set -e
    case "$READY_CODE" in
        0)
            echo "spool ready: $READY_JSON"
            break
            ;;
        2)
            if [ "$(date +%s)" -ge "$READY_DEADLINE" ]; then
                echo "error: the spool at $SPOOL_DIR is not ready after ${GATE_WAIT_CHECKPOINT_SECS}s: $READY_JSON" >&2
                exit 65
            fi
            sleep 5
            ;;
        3)
            echo "error: the spool at $SPOOL_DIR is structurally invalid; polling cannot fix it: $READY_JSON" >&2
            exit 65
            ;;
        *)
            echo "error: spool readiness inspection failed (exit $READY_CODE): $READY_JSON" >&2
            exit 65
            ;;
    esac
done

KILLED_AT_SEQUENCE=""
if [ "$GATE_MODE" = "resume" ]; then
    # The restart is the thing under test, so the gate performs it. SIGKILL rather than SIGTERM:
    # a follower given the chance to close cleanly would prove nothing about crash recovery.
    #
    # Waited for rather than timed. The snapshot export has taken over two and a half minutes on
    # this corpus, and a fixed delay that lands before the first checkpoint kills a follower with
    # no ack at all — a gate failure that says nothing about resuming.
    KILL_AFTER_BLOCKS=${GATE_KILL_AFTER_BLOCKS:-20}
    KILL_TIMEOUT=${GATE_KILL_TIMEOUT_SECS:-900}
    echo "==> live follow against $SPOOL_DIR (first run, killed after $KILL_AFTER_BLOCKS verdicts)"
    "$PS_REPLAY_BIN" --follow "$SPOOL_DIR" \
        --json "$FOLLOW_JSON" --ack "$ACK_FILE" --label "s3-live-gate-resume-first" "$@" &
    first_pid=$!
    waited=0
    while [ "$waited" -lt "$KILL_TIMEOUT" ]; do
        if ! kill -0 "$first_pid" 2>/dev/null; then
            echo "FAIL: the first run exited on its own before it could be killed" >&2
            exit 1
        fi
        verdicts=$(grep -c '"kind":"verdict"' "$FOLLOW_JSON" 2>/dev/null || echo 0)
        state=$(jq -r '.state // ""' "$ACK_FILE" 2>/dev/null || echo "")
        if [ "$state" = "streaming" ] && [ "$verdicts" -ge "$KILL_AFTER_BLOCKS" ]; then
            break
        fi
        sleep 5
        waited=$((waited + 5))
    done
    if [ "$waited" -ge "$KILL_TIMEOUT" ]; then
        echo "FAIL: the first run did not reach $KILL_AFTER_BLOCKS verdicts in ${KILL_TIMEOUT}s" >&2
        kill -9 "$first_pid" 2>/dev/null || true
        exit 1
    fi
    kill -9 "$first_pid" 2>/dev/null || true
    wait "$first_pid" 2>/dev/null || true
    KILLED_AT_SEQUENCE=$(jq -r '.last_sequence' "$ACK_FILE")
    echo "==> killed after $verdicts verdicts, with the ack at sequence $KILLED_AT_SEQUENCE"
    # A separate verdict file, so "which lines did the *resumed* run publish" is answerable.
    FOLLOW_JSON="$OUT_DIR/follow-resumed.jsonl"
    set -- "$@" --resume
fi

if [ "$GATE_MODE" = "epoch" ]; then
    # The producer restart is the operator's move: it holds the datadir. By the time this runs the
    # spool must already carry End, Manifest(epoch 2), and that epoch's checkpoint, and the ack
    # must be the one the epoch-1 follower left behind.
    if [ ! -s "$ACK_FILE" ]; then
        echo "FAIL: GATE_MODE=epoch needs the epoch-1 follower's ack at $ACK_FILE" >&2
        echo "      run the clean gate first, let it consume the producer's End, then restart" >&2
        echo "      the producer with PS_STREAM_RESUME=1 and run this mode" >&2
        exit 1
    fi
    ACK_EPOCH_BEFORE=$(jq -r '.epoch // 1' "$ACK_FILE")
    echo "==> resuming across an epoch boundary (ack was written under epoch $ACK_EPOCH_BEFORE)"
    FOLLOW_JSON="$OUT_DIR/follow-epoch.jsonl"
    set -- "$@" --resume
fi

echo "==> live follow against $SPOOL_DIR (mode: $GATE_MODE, verdicts: $FOLLOW_JSON)"
follow_code=0
LIVE_VERDICTS=0
if [ "$GATE_MODE" = "long" ]; then
    if printf '%s\n' "$@" | grep -q -- '--max-blocks'; then
        echo "error: GATE_MODE=long refuses --max-blocks — a block bound stops the follower" >&2
        echo "       before the producer's End, so the spool never closes and the batch gate" >&2
        echo "       cannot run. The watcher below bounds the run instead." >&2
        exit 64
    fi
    "$PS_REPLAY_BIN" --follow "$SPOOL_DIR" \
        --json "$FOLLOW_JSON" --ack "$ACK_FILE" --label "s3-live-gate-$GATE_MODE" "$@" &
    follow_pid=$!
    signaled=0
    while kill -0 "$follow_pid" 2>/dev/null; do
        LIVE_VERDICTS=$(grep -c '"tail_live":true' "$FOLLOW_JSON" 2>/dev/null || true)
        LIVE_VERDICTS=${LIVE_VERDICTS:-0}
        if [ "$signaled" -eq 0 ] && [ "$LIVE_VERDICTS" -ge "$GATE_MIN_BLOCKS" ]; then
            # Quiesce before the SIGTERM: a producer stopped inside a recovery-checkpoint export
            # loses that checkpoint (write-through commits are on disk, the checkpoint is not),
            # and the skim assertions below would fail a run the chain behaved normally in.
            # Judged by the event-log reducer, not by grep counts: a failed attempt with retries
            # left *counts* as settled but the cause is still pending, and a SIGTERM inside that
            # gap loses the checkpoint the retry owes. Current epoch only, bounded by
            # GATE_QUIESCE_SECS, after which the SIGTERM proceeds and the assertions judge
            # whatever the interrupted export left behind.
            EVENTS_FILE="$(dirname "$SPOOL_DIR")/$(basename "$SPOOL_DIR").producer-events.jsonl"
            if [ -f "$EVENTS_FILE" ]; then
                QUIESCE_STATE=$(python3 "$(dirname "$0")/producer_event_state.py" "$EVENTS_FILE" 2>&1) && quiesced=0 || quiesced=$?
                if [ "$quiesced" -eq 1 ]; then
                    QUIESCE_DEADLINE=${QUIESCE_DEADLINE:-$(( $(date +%s) + ${GATE_QUIESCE_SECS:-600} ))}
                    if [ "$(date +%s)" -lt "$QUIESCE_DEADLINE" ]; then
                        echo "==> $LIVE_VERDICTS live verdicts, but a recovery cause is still pending; waiting to quiesce before SIGTERM: $QUIESCE_STATE"
                        sleep "${GATE_WATCH_SECS:-15}"
                        continue
                    fi
                    echo "warning: a recovery cause never settled within ${GATE_QUIESCE_SECS:-600}s; proceeding with SIGTERM: $QUIESCE_STATE" >&2
                elif [ "$quiesced" -ne 0 ]; then
                    echo "warning: the producer-event reducer failed ($QUIESCE_STATE); proceeding without a quiesce judgement" >&2
                fi
            fi
            signaled=1
            if [ -n "${GATE_PRODUCER_PID:-}" ]; then
                echo "==> $LIVE_VERDICTS live verdicts: SIGTERM to producer $GATE_PRODUCER_PID; waiting for End(shutdown)"
                kill -TERM "$GATE_PRODUCER_PID" 2>/dev/null ||
                    echo "warning: could not signal $GATE_PRODUCER_PID; stop the producer yourself" >&2
            else
                echo "==> $LIVE_VERDICTS live verdicts: target reached — SIGTERM the producer now;"
                echo "    the follower exits when it consumes the End(shutdown) the producer writes"
            fi
        fi
        # An `End` that never comes must not hold this gate open forever.
        #
        # The producer writes `End` from its own shutdown path, but nothing outside this process
        # tree guarantees it ran: reth does not register an ExEx as a graceful task, so on
        # 2026-08-22 a SATA host exited on SIGTERM with the spool's last frame a commit, and the
        # follower — correctly, by long mode's own rules — waited for a terminator that did not
        # exist. It waited nearly four hours, taking the arm, the phase behind it and the node's
        # datadir with it. The producer-side fix removes the race; this removes the dependence.
        #
        # The bound starts only once the producer is gone, so a slow chain still gets all the time
        # it wants. What follows is a real failure — the assertions below refuse an unclosed
        # corpus — and failing an arm is the outcome this is for. Silently hanging is not.
        if [ "$signaled" -eq 1 ] && [ -n "${GATE_PRODUCER_PID:-}" ] &&
           ! kill -0 "$GATE_PRODUCER_PID" 2>/dev/null; then
            END_DEADLINE=${END_DEADLINE:-$(( $(date +%s) + ${GATE_END_GRACE_SECS:-300} ))}
            if [ "$(date +%s)" -ge "$END_DEADLINE" ]; then
                echo "warning: producer $GATE_PRODUCER_PID exited without closing the stream;" >&2
                echo "         no End reached the follower in ${GATE_END_GRACE_SECS:-300}s." >&2
                echo "==> abandoning the follower; this corpus is truncated and the assertions say so"
                kill -TERM "$follow_pid" 2>/dev/null
                for _ in $(seq 1 30); do kill -0 "$follow_pid" 2>/dev/null || break; sleep 1; done
                kill -KILL "$follow_pid" 2>/dev/null
                break
            fi
        fi
        sleep "${GATE_WATCH_SECS:-15}"
    done
    wait "$follow_pid" || follow_code=$?
    LIVE_VERDICTS=$(grep -c '"tail_live":true' "$FOLLOW_JSON" 2>/dev/null || true)
    LIVE_VERDICTS=${LIVE_VERDICTS:-0}
else
    "$PS_REPLAY_BIN" --follow "$SPOOL_DIR" \
        --json "$FOLLOW_JSON" --ack "$ACK_FILE" --label "s3-live-gate-$GATE_MODE" "$@" || follow_code=$?
fi
echo "==> follow exit code: $follow_code (0 clean end, 1 disagreement/fault, 2 NeedsSnapshot, 3 ended before checkpoint, 4 idle timeout)"
fi
# ==== end of the online half =================================================================

if [ "$GATE_PHASE" = "capture" ]; then
    # Written to a temporary file and renamed, which is what makes "whole" true rather than
    # intended: a judge that found a half-written record would restore a run state nobody
    # captured, and every value here is one the assertions cannot re-derive from files.
    {
        echo "CAPTURE_SCHEMA=1"
        printf 'CAPTURE_GATE_MODE=%q\n' "$GATE_MODE"
        printf 'CAPTURE_SPOOL_DIR=%q\n' "$SPOOL_DIR"
        printf 'CAPTURE_FOLLOW_JSON=%q\n' "$FOLLOW_JSON"
        printf 'CAPTURE_FOLLOW_CODE=%q\n' "$follow_code"
        printf 'CAPTURE_LIVE_VERDICTS=%q\n' "$LIVE_VERDICTS"
        printf 'CAPTURE_KILLED_AT_SEQUENCE=%q\n' "${KILLED_AT_SEQUENCE:-}"
        printf 'CAPTURE_ACK_EPOCH_BEFORE=%q\n' "${ACK_EPOCH_BEFORE:-}"
        printf 'CAPTURE_MIN_BLOCKS=%q\n' "$GATE_MIN_BLOCKS"
        printf 'CAPTURE_EXPECT_RECHECKPOINT=%q\n' "${GATE_EXPECT_RECHECKPOINT:-1}"
        printf 'CAPTURE_REQUIRE_LIVE=%q\n' "${GATE_REQUIRE_LIVE:-0}"
        printf 'CAPTURE_FORCE_RESTORE=%q\n' "${GATE_FORCE_RESTORE:-0}"
    } > "$CAPTURE_ENV.tmp"
    mv -f "$CAPTURE_ENV.tmp" "$CAPTURE_ENV"
    echo "==> capture complete: $CAPTURE_ENV"
    echo "    the datadir is free — restart the node that owns it, then judge with:"
    echo "    GATE_MODE=$GATE_MODE GATE_PHASE=judge $0 $SPOOL_DIR $OUT_DIR"
    exit 0
fi

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
# `witnessed` counts every commit this process replayed; `blocks_verified` counts only the ones it
# verified for the first time. A resumed run re-derives the blocks between its checkpoint and the
# watermark it left, and those are counted apart — so the identity has a second term there.
check "$SUMMARY" "every commit was witnessed" \
    ".witnessed == .blocks_verified + .catch_up_blocks"
check "$SUMMARY" "no reconstructed payload" ".reconstructed == 0"
check "$SUMMARY" "no absent payload" ".absent == 0"
check "$SUMMARY" "no disagreement" ".disagreements == 0"
check "$SUMMARY" "no failure" ".failures == 0"
check "$SUMMARY" "the follower itself agreed" ".agreed == true"
check "$SUMMARY" "nothing in the recovery scan had to be refused" ".scan_refusals == 0"
# A late recovery checkpoint that contradicted the follower's own generation is a disagreement
# with its own counter; the run continues past one, so only this assert fails the gate on it.
check "$SUMMARY" "no late recovery checkpoint contradicted the recovered generation" \
    "(.late_skim_mismatches // 0) == 0"
# The rewind bound exists for pathological spools; a healthy gate run never hits it.
check "$SUMMARY" "no rewind window was refused for its size" \
    "(.rewind_windows_refused // 0) == 0"
if [ "${GATE_EXPECT_RECHECKPOINT:-1}" = "1" ]; then
    # Under PS_STREAM_REORG_CHECKPOINT=always every applied reorg owes a checkpoint before the
    # stream closes; one still pending at End means the quiesce failed or the producer lost it.
    check "$SUMMARY" "no recovery checkpoint was still pending when the stream closed" \
        "(.recovery_checkpoints_pending_at_end // 0) == 0"
fi
# Counter-first until a full-scale cohort confirms it; the gate holds it to zero meanwhile.
# 424/424 both-recorded comparisons agreed on the live-follower and reorg-recovery corpora (72 of
# the reorg-recovery commits carried none).
check "$SUMMARY" "the readiness watermarks agreed wherever both sides recorded one" \
    "(.watermark_mismatches // 0) == 0"
if [ "$GATE_MODE" != "epoch" ]; then
    # The second axis: agreeing on every block it looked at is not the same as having looked at
    # every canonical block, and only this distinguishes them. Not asserted in `epoch` mode —
    # there the discontinuity is the thing under test, and is judged as such below.
    check "$SUMMARY" "no canonical block went unverified" ".continuous == true"
    check "$SUMMARY" "every announced branch was delivered in full" ".winning_branches_incomplete == 0"
    check "$SUMMARY" "no recovery landed somewhere it did not ask for" ".restores_reset == 0"
    if [ "$GATE_MODE" = "long" ]; then
        # A cohort held open for many hours may meet a reorg past the retained generation's
        # depth. That is the chain behaving normally and the DB-free design recovering as
        # designed: NeedsSnapshot, then the producer's recovery checkpoint, restored on the
        # exact block it asked for. So long does not require zero entries — it requires every
        # entry answered by a continuous restore (the bootstrap restore is the +1), and none
        # left pending at the end, which the equality catches: an unanswered entry leaves
        # `restores` one short.
        check "$SUMMARY" "every NeedsSnapshot was answered by a continuous restore" \
            ".restores == 1 + .needs_snapshot_entries and .restores_continuous == .needs_snapshot_entries"
    else
        check "$SUMMARY" "exactly one restore, no NeedsSnapshot" \
            ".restores == 1 and .needs_snapshot_entries == 0"
    fi
fi
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
    long)
        [ "$follow_code" -eq 0 ] && echo "  ok: follow exit 0" || { echo "  FAIL: follow exit $follow_code, wanted 0" >&2; FAILED=1; }
        check "$SUMMARY" "the stream closed as End(shutdown)" \
            '.outcome == "ended" and .end_kind == "shutdown" and .before_checkpoint == false'
        # Reorgs are observations here, not failures: a cohort held open for a day meets the
        # chain as it is. What is asserted is that every one it met was handled — in place at
        # depth 1, by a continuous checkpoint restore past it (the common checks above).
        LIFECYCLE=$(printf '%s' "$SUMMARY" | jq -r '"reorgs=\(.reorgs_applied) reverts=\(.reverts_applied) skimmed=\(.checkpoints_skimmed) needs_snapshot=\(.needs_snapshot_entries) latency_anomalies=\(.latency_anomalies // 0) timing_anomalies=\(.timing_anomalies // 0)"')
        echo "  observed: $LIFECYCLE"
        echo "  observed: live(tail_live) verdicts=$LIVE_VERDICTS of $GATE_MIN_BLOCKS wanted"
        if [ "${GATE_REQUIRE_LIVE:-0}" = "1" ]; then
            if [ "$LIVE_VERDICTS" -ge "$GATE_MIN_BLOCKS" ]; then
                echo "  ok: the live capture reached its target"
            else
                echo "  FAIL: only $LIVE_VERDICTS live verdicts; the capture claim needs $GATE_MIN_BLOCKS" >&2
                FAILED=1
            fi
        fi
        if [ "${GATE_EXPECT_RECHECKPOINT:-1}" = "1" ]; then
            check "$SUMMARY" "every applied reorg/revert had its recovery checkpoint skimmed" \
                '.checkpoints_skimmed >= (.reorgs_applied + .reverts_applied)'
        fi
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
        if [ "${GATE_EXPECT_RECHECKPOINT:-1}" = "1" ]; then
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
            'select(.kind == "verdict" and .catch_up == false and (.recovery_replay // false) == false and .sequence <= $at) | .sequence' \
            "$FOLLOW_JSON" | wc -l)
        if [ "$NEW_BELOW" -eq 0 ]; then
            echo "  ok: no new verdict was published below the watermark (sequence $KILLED_AT_SEQUENCE)"
        else
            echo "  FAIL: $NEW_BELOW verdicts at or below sequence $KILLED_AT_SEQUENCE were not labelled catch-up" >&2
            FAILED=1
        fi
        # The counter and the lines it summarises have to be the same claim, or one of them is
        # describing a run that did not happen.
        LABELLED=$(grep -c '"catch_up":true' "$FOLLOW_JSON" 2>/dev/null || echo 0)
        check "$SUMMARY" "the labelled lines and the catch-up count agree" \
            ".catch_up_blocks == $LABELLED"
        ;;
    epoch)
        # Exit 0 means agreed *and* continuous, and a producer restart is honestly neither the
        # second nor pretending to be. Requiring 0 here would be requiring the follower to report
        # a continuity across the boundary that nobody established, so the record is judged
        # instead — the same reason the batch exit code is only noted below.
        echo "  note: follow exit $follow_code (an epoch boundary is a real discontinuity)"
        # The crossing itself: it started at the new epoch's manifest rather than replaying the
        # closed one below it, and rebootstrapped from that epoch's own checkpoint.
        check "$SUMMARY" "the follower crossed into the new epoch" '.resumed_from != null'
        check "$SUMMARY" "and rebootstrapped there" '.restores == 1'
        # Reported as the reset it is. An epoch says the producer's state broke, so the interval
        # across the boundary was validated by nothing, and a run claiming otherwise would be
        # claiming a continuity nobody established.
        check "$SUMMARY" "the boundary is reported as an explicit reset" \
            '.restores_reset == 1 and .restores_continuous == 0'
        check "$SUMMARY" "and the run does not claim continuity across it" '.continuous == false'
        ACK_EPOCH_AFTER=$(jq -r '.epoch // 0' "$ACK_FILE")
        if [ "$ACK_EPOCH_AFTER" -gt "$ACK_EPOCH_BEFORE" ]; then
            echo "  ok: the ack moved from epoch $ACK_EPOCH_BEFORE to $ACK_EPOCH_AFTER"
        else
            echo "  FAIL: the ack is still epoch $ACK_EPOCH_AFTER; the crossing was not recorded" >&2
            echo "        (if it already says the new epoch, this mode has been run once already;" >&2
            echo "         it crosses a boundary, so it needs an ack from below one)" >&2
            FAILED=1
        fi
        ;;
    truncated)
        [ "$follow_code" -eq 4 ] && echo "  ok: follow exit 4 (idle timeout — a cut stream never says goodbye)" || { echo "  FAIL: follow exit $follow_code, wanted 4" >&2; FAILED=1; }
        check "$SUMMARY" "the follower timed out waiting, not recovering" \
            '.outcome == "idle_timeout" and .waiting_in != "needs_snapshot"'
        ;;
esac

if [ "$GATE_MODE" = "clean" ] || [ "$GATE_MODE" = "reorg" ] || [ "$GATE_MODE" = "long" ]; then
    echo "==> judging the verdict instrumentation"
    # The never-null trio on every non-catch-up verdict: the primary, its transport cost, and
    # its phases — valid on backlog frames too, because validation and delivery really ran.
    UNINSTRUMENTED=$(jq -r 'select(.kind == "verdict" and .catch_up == false and
        ((.standalone_validation_us == null) or (.delivery_us == null) or (.phases == null))) |
        .sequence' "$FOLLOW_JSON" 2>/dev/null | wc -l)
    if [ "$UNINSTRUMENTED" -eq 0 ]; then
        echo "  ok: every verdict carries the timing boundaries and phases"
    else
        echo "  FAIL: $UNINSTRUMENTED verdicts are missing timing fields" >&2
        FAILED=1
    fi
    # Latency exists only on live-tail verdicts (backlog mtime distance is history, not a wait),
    # and a clock anomaly nulls it by design and is counted.
    NO_LATENCY=$(jq -r 'select(.kind == "verdict" and .catch_up == false and .tail_live == true
        and .decision_latency_us == null) | .sequence' "$FOLLOW_JSON" 2>/dev/null | wc -l)
    ANOMALIES=$(printf '%s' "$SUMMARY" | jq -r '.latency_anomalies // 0')
    if [ "$NO_LATENCY" -le "$ANOMALIES" ]; then
        echo "  ok: live latency nulls ($NO_LATENCY) are within the counted anomalies ($ANOMALIES)"
    else
        echo "  FAIL: $NO_LATENCY live verdicts lack latency with only $ANOMALIES anomalies counted" >&2
        FAILED=1
    fi
    echo "==> judging the run manifest"
    MANIFEST_LINE=$(grep '"kind":"run_manifest"' "$FOLLOW_JSON" 2>/dev/null | head -1 || true)
    if [ -n "$MANIFEST_LINE" ]; then
        echo "  ok: the follower stamped a run manifest"
        if [ "$GATE_MODE" = "long" ]; then
            # A cohort is code-freeze evidence only if its binary can say what it was built
            # from: a null commit or a dirty tree makes the record unattributable, so long
            # requires the build to have been stamped (export PS_BUILD_COMMIT/PS_BUILD_DIRTY/
            # PS_CARGO_LOCK_SHA256 before cargo build).
            check "$MANIFEST_LINE" "the manifest pins a commit from a clean tree" \
                '.provenance.build_commit != null and .provenance.build_dirty == false'
        fi
    else
        echo "  FAIL: no run_manifest line in $FOLLOW_JSON" >&2
        FAILED=1
    fi
fi

echo "==> judging the batch record"
if [ "$GATE_MODE" = "epoch" ]; then
    # A two-epoch corpus is *not* continuous, and the batch CLI exits non-zero for exactly that
    # reason. Asserting exit 0 here would be asserting that a producer restart left no gap.
    echo "  note: batch exit $batch_code (an epoch boundary is a real discontinuity; judged on the record)"
    check "$BATCH" "batch crossed exactly one epoch boundary" ".epoch_transitions == 1"
    check "$BATCH" "and rebootstrapped there rather than continuing" \
        "([.resyncs[] | select(.continuous == false)] | length) >= 1"
    check "$BATCH" "batch agreed on every block it did verify" ".agreed == true"
    check "$BATCH" "and read the whole corpus" ".terminal == null and .closed == true"
else
    [ "$batch_code" -eq 0 ] && echo "  ok: batch exit 0" || { echo "  FAIL: batch exit $batch_code, wanted 0" >&2; FAILED=1; }
    check "$BATCH" "batch agreed with the recording" ".agreed == true and .terminal == null"
    check "$BATCH" "batch verified every canonical block the corpus carried" ".continuous == true"
fi
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
    long)
        check "$BATCH" "batch reached the end of the corpus" ".complete == true and .closed == true"
        echo "  observed: batch reorgs=$(printf '%s' "$BATCH" | jq -r '.reorgs_applied') reverts=$(printf '%s' "$BATCH" | jq -r '.reverts_applied')"
        ;;
    resume)
        # The corpus is one stream whichever way it is read, so a batch replay of it has to reach
        # the same conclusion as two follower runs stitched together.
        check "$BATCH" "the spool is closed (End present)" ".closed == true"
        ;;
    epoch) ;;
    truncated) check "$BATCH" "the spool is cut (no End) — the point of this control run" ".closed == false" ;;
esac

if [ "$GATE_MODE" = "reorg" ] && [ "${GATE_FORCE_RESTORE:-0}" = "1" ]; then
    # Skimming a recovery checkpoint proves the two implementations agree about the block. It does
    # not prove the snapshot behind it would restore a validator that has no retained generation
    # to recover with — the one consumer that most needs it. This installs it and replays the
    # winning branch against it, which is the only thing that does prove it.
    RESTORE_AT=$(jq -rs 'map(select(.kind == "skimmed")) | .[-1].sequence // empty' \
        "$FOLLOW_JSON" 2>/dev/null || true)
    if [ -z "$RESTORE_AT" ]; then
        echo "  FAIL: GATE_FORCE_RESTORE=1 but the follower skimmed no recovery checkpoint" >&2
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
                "([.resyncs[] | select(.at_sequence == $RESTORE_AT)] | length) == 1"
            # The two assertions above pass on a driver that installs the checkpoint and reads
            # straight on, provided nothing follows the checkpoint to trip over — which is how a
            # driver that replayed none of the branch reported that all of it agreed, for five
            # days, against a corpus whose checkpoint sat at the stream tail. Agreement is only
            # evidence about the branch if the branch was replayed, so that is asserted first.
            check "$FORCED" "the winning branch below the checkpoint was actually replayed" \
                ".rewind_replayed_commits > 0"
            check "$FORCED" "and no window was refused for its size" \
                ".rewind_windows_refused == 0"
            check "$FORCED" "and the winning branch replayed against it agreed throughout" \
                ".agreed == true and .complete == true"
            check "$FORCED" "and the recovery it recorded is continuous" \
                "[.resyncs[] | select(.at_sequence == $RESTORE_AT) | .continuous] == [true]"
        fi
    fi
fi

# The report is generated, never hand-curated: both preflights were written up by hand and both
# pooled the restore backlog into the live latency table. The analyzer emits one stable JSON and
# the renderer consumes only that, so every number in result.md exists in distributions.json
# beside it. A failure here fails the gate — a run whose evidence cannot be rendered is a run
# whose evidence nobody will read.
echo "==> analyzing and rendering"
SCRIPT_DIR=$(dirname "$0")
ANALYZE_ARGS=()
for candidate in "$OUT_DIR/follow.jsonl" "$OUT_DIR/follow-resumed.jsonl" "$OUT_DIR/follow-epoch.jsonl"; do
    [ -s "$candidate" ] && ANALYZE_ARGS+=(--follow "$candidate")
done
[ -s "$BATCH_JSON" ] && ANALYZE_ARGS+=(--batch "$BATCH_JSON")
EVENTS_FILE="$(dirname "$SPOOL_DIR")/$(basename "$SPOOL_DIR").producer-events.jsonl"
[ -s "$EVENTS_FILE" ] && ANALYZE_ARGS+=(--producer-events "$EVENTS_FILE")
PRODUCER_MANIFEST="$(dirname "$SPOOL_DIR")/$(basename "$SPOOL_DIR").run-manifest.jsonl"
[ -s "$PRODUCER_MANIFEST" ] && ANALYZE_ARGS+=(--producer-manifest "$PRODUCER_MANIFEST")
if python3 "$SCRIPT_DIR/analyze_follow_bench.py" "${ANALYZE_ARGS[@]}" --phases \
        --json "$OUT_DIR/distributions.json" > "$OUT_DIR/distributions.txt" 2>&1 &&
   python3 "$SCRIPT_DIR/render_follow_report.py" "$OUT_DIR/distributions.json" \
        --out "$OUT_DIR/result.md"; then
    echo "  ok: $OUT_DIR/distributions.json and $OUT_DIR/result.md"
else
    echo "  FAIL: the analyzer/renderer chain failed; see $OUT_DIR/distributions.txt" >&2
    FAILED=1
fi

echo "==> results in $OUT_DIR"
if [ "$FAILED" -ne 0 ]; then
    echo "==> GATE FAILED ($GATE_MODE)" >&2
    exit 1
fi
echo "==> GATE PASSED ($GATE_MODE)"
