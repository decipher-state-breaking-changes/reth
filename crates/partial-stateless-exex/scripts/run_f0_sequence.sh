#!/usr/bin/env bash
# The final-profile dress rehearsal, unattended: the reorg gate, then the fsync durability arms,
# then the transition-mutation gate if a recorded spool exists for it.
#
#     nohup bash crates/partial-stateless-exex/scripts/run_f0_sequence.sh \
#         > <log-dir>/f0-sequence.log 2>&1 &
#     tail -f <log-dir>/f0-sequence.log
#
# Tracked, because a run's profile is part of its provenance: a cohort whose producer environment
# lives in an untracked file beside the results cannot be reproduced from the commit it names.
#
# All three phases want the datadir, so they cannot overlap — that is why this is one script and
# not three. Worst case is about 28 hours: up to REORG_WATCH_HOURS (default 24) waiting for a
# chain reorganisation, then four hours of ABBA arms. Reorg arrival is not hourly; this host's
# own record is one attempt that ran 3,129 blocks over six hours and met none, so the bound is a
# bound, not an estimate.
#
# A failing phase does not abandon the ones after it: each is independent evidence, and a script
# that stops at the first disappointment turns a 24-hour investment into one data point.
#
# Environment:
#     REORG_WATCH_HOURS=24   outer bound on phase 1's wait for a reorg
#     ABBA_BLOCKS=300        live verdicts per fsync arm (four arms: A B B A)
#     GATE_WAIT_CHECKPOINT_SECS=1800
#                            upper bound for cold 90-block warm-up plus checkpoint export;
#                            readiness is polled, so this is not a fixed delay
#     SKIP_REORG=1           start at phase 2
#     SKIP_ABBA=1            stop after phase 1
#
# Host paths — every one of these is outside the repository, so every one is overridable and the
# two that have no defensible default are required:
#     BASE=<dir>             where this run's arms and results go (default: ./f0-<timestamp>)
#     PRODUCER_BIN, REPLAY_BIN
#                            the two stamped binaries (default: this checkout's release build)
#     NODE_DATADIR=<dir>     the node datadir all three phases take turns holding (required)
#     NODE_JWT=<file>        the engine JWT secret (required)
#     RESTORE_VANILLA=<file> optional script the cleanup runs to put the ordinary node back
set -uo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
BASE=${BASE:-$PWD/f0-$(date +%Y%m%d-%H%M)}
SCRIPTS="$REPO/crates/partial-stateless-exex/scripts"
# Default to the same target directory Cargo would use: on a host that exports
# CARGO_TARGET_DIR, $REPO/target holds nothing and a hardcoded default would fail the
# stamp check on every launch.
PRODUCER_BIN=${PRODUCER_BIN:-${CARGO_TARGET_DIR:-$REPO/target}/release/reth-partial-stateless}
REPLAY_BIN=${REPLAY_BIN:-${CARGO_TARGET_DIR:-$REPO/target}/release/ps-replay}
NODE_DATADIR=${NODE_DATADIR:?set NODE_DATADIR to the node datadir this run may take over}
NODE_JWT=${NODE_JWT:?set NODE_JWT to the engine JWT secret}
REORG_WATCH_HOURS=${REORG_WATCH_HOURS:-24}
ABBA_BLOCKS=${ABBA_BLOCKS:-300}
GATE_WAIT_CHECKPOINT_SECS=${GATE_WAIT_CHECKPOINT_SECS:-1800}

# A policy-dataset capture is a separate job that holds the datadir and spends part of every block
# on a witness nobody measured. Unsetting it for the producer (below) keeps it out of this run's
# processes; refusing here keeps this run from being started in a shell that was configured for it,
# which is a different mistake with the same result.
if [ -n "${PS_POLICY_DATASET_CAPTURE_DIR:-}" ]; then
    echo "PS_POLICY_DATASET_CAPTURE_DIR is set; this sequence measures and cannot run beside a" >&2
    echo "policy replay dataset capture. Unset it, or run the capture on its own." >&2
    exit 2
fi
    # MDBX times a reader out at 5 minutes by default and the ExEx dies with it: its sidecar
    # multiproof runs inside a read transaction, and a node backfilling a gap at full speed holds
    # one open far longer than a node following the tip. 0 disables the timeout.
NODE_FLAGS=(
    node --datadir "$NODE_DATADIR" --minimal --http
    --db.read-transaction-timeout 0
    --http.api "eth,net,web3,debug,trace"
    --authrpc.addr 127.0.0.1 --authrpc.port 8551
    --authrpc.jwtsecret "$NODE_JWT"
    --ws --ws.addr 127.0.0.1 --ws.port 8546
    --ws.api eth,trace,debug,net --ws.origins "localhost"
)

mkdir -p "$BASE"
RESULTS="$BASE/RESULTS.txt"
: > "$RESULTS"
say() { echo "[$(date +%H:%M:%S)] $*"; }
record() { echo "$*" >> "$RESULTS"; say "RESULT: $*"; }

# Optional and host-specific: whatever brings the operator's ordinary node back up after this
# run hands the datadir over. Unset means the cleanup only stops the producer.
RESTORE_VANILLA=${RESTORE_VANILLA:-}
CLEANED=0
cleanup() {
    [ "$CLEANED" = 1 ] && return
    CLEANED=1
    # Whatever happened — finished, failed, killed — the datadir goes back to the node that had
    # it. A run that dies unattended must not leave the node down until somebody notices.
    pkill -f "while sleep 300; do find '" 2>/dev/null
    if pgrep -f "reth-partial-stateless node --datadir" >/dev/null; then
        say "cleanup: SIGTERM to the producer"
        pkill -TERM -f "reth-partial-stateless node --datadir" 2>/dev/null
        for _ in $(seq 1 180); do
            pgrep -f "reth-partial-stateless node --datadir" >/dev/null || break
            sleep 2
        done
    fi
    if [ "${RESTORE:-1}" = 1 ] && [ -x "$RESTORE_VANILLA" ] && ! pgrep -x reth >/dev/null; then
        say "cleanup: putting the vanilla node back"
        bash "$RESTORE_VANILLA" >> "$BASE/restore.out" 2>&1
    fi
}
trap cleanup EXIT INT TERM

if pgrep -x reth >/dev/null; then
    if [ "${STOP_VANILLA:-0}" = 1 ]; then
        say "stopping the vanilla node (pid $(pgrep -x reth)); it holds the MDBX writer lock"
        kill "$(pgrep -x reth)"
        for _ in $(seq 1 300); do pgrep -x reth >/dev/null || break; sleep 2; done
        pgrep -x reth >/dev/null && { say "it did not go down; refusing to fight it"; exit 2; }
        say "vanilla node down"
    else
        say "vanilla reth is up (pid $(pgrep -x reth)); it holds the datadir."
        say "stop it first, or re-run with STOP_VANILLA=1 to let this script do it."
        exit 2
    fi
fi
# Everything that invalidated a previous run, checked before this one starts rather than found in
# its results. A dirty tree, a binary that cannot say what it was built from, two binaries built
# from different commits, or a governor that throttles mid-run each cost a whole cohort.
HEAD_COMMIT=$(cd "$REPO" && git rev-parse HEAD)
DIRTY=$(cd "$REPO" && git status --porcelain | wc -l)
if [ "$DIRTY" -ne 0 ]; then
    say "the working tree has $DIRTY uncommitted changes; a dirty build is not code-freeze evidence"
    exit 2
fi
for bin in "$PRODUCER_BIN" "$REPLAY_BIN"; do
    [ -x "$bin" ] || { say "no binary at $bin"; exit 2; }
    # The stamp is read from the binary, not from its timestamp: cargo hands back a cached
    # unstamped artifact without even changing the mtime, which is how the last arms lost theirs.
    grep -qa "$HEAD_COMMIT" "$bin" || {
        say "$bin does not carry $HEAD_COMMIT — build both in one stamped shell:"
        say "  export PS_BUILD_COMMIT=\$(git rev-parse HEAD) PS_BUILD_DIRTY=0 \\"
        say "         PS_CARGO_LOCK_SHA256=\$(sha256sum Cargo.lock | cut -d' ' -f1)"
        exit 2
    }
done
# The canonical policy is the driver's *dynamic, load-scaling* governor, not a particular name.
# Under the generic cpufreq governors that is `ondemand`. Under `intel_pstate` in active mode --
# which is what recent Intel parts default to -- `ondemand` does not exist and the dynamic governor
# is named `powersave`: it scales with load up to turbo and is emphatically not a pinned low-power
# mode. Demanding the literal string `ondemand` would make this sequence unrunnable on such a host,
# and switching that host to `performance` to satisfy the check would put it on a different policy
# from the other one, which is the thing this gate exists to prevent.
CPU_COUNT=$(getconf _NPROCESSORS_ONLN)
GOVERNOR_COUNT=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | wc -l)
GOVERNORS=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | tr '\n' ' ')
SCALING_DRIVER=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>/dev/null)
AVAILABLE=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors 2>/dev/null)
case " $AVAILABLE " in
    *" ondemand "*) EXPECTED_GOVERNOR=ondemand ;;
    *" powersave "*) EXPECTED_GOVERNOR=powersave ;;
    *) EXPECTED_GOVERNOR="" ;;
esac
if [ -z "$EXPECTED_GOVERNOR" ]; then
    say "no dynamic governor is available on this host"
    say "driver='$SCALING_DRIVER' available='$AVAILABLE'"
    exit 2
fi
if [ "$GOVERNOR_COUNT" -ne "$CPU_COUNT" ] || [ "$GOVERNORS" != "$EXPECTED_GOVERNOR " ]; then
    say "canonical F0 requires $EXPECTED_GOVERNOR on every online CPU (driver '$SCALING_DRIVER')"
    say "observed governors='$GOVERNORS' files=$GOVERNOR_COUNT online_cpus=$CPU_COUNT"
    say "  sudo cpupower frequency-set -g $EXPECTED_GOVERNOR"
    exit 2
fi
say "base=$BASE"
say "producer=$PRODUCER_BIN ($(date -r "$PRODUCER_BIN" '+%F %T'))"
say "repo HEAD=$HEAD_COMMIT  clean  governor=${GOVERNORS:-unknown} driver=${SCALING_DRIVER:-unknown}"

# --- helpers ------------------------------------------------------------------------------

# Starts a producer against a fresh arm directory. Echoes its pid.
# The canonical producer profile, in full and explicitly — including the two switches
# whose absence made both 1,001-verdict preflights measure the re-executing baseline instead, and
# the two that must be absent rather than merely unset in this shell.
# No comments inside the continuation below. A `\` joins the next line whole, so a comment line
# after one ends the command there — `env` then runs with assignments and no command, prints the
# environment, and the *remaining* lines start a producer carrying none of these settings. That is
# a producer that records nothing, and it looks like a slow one rather than a broken one.
start_producer() { # <arm-dir> <fsync>
    local arm=$1 fsync=$2
    mkdir -p "$arm"/{spool,bootstrap,sidecars,out}
    env -u PS_CAPTURE_DIR -u PS_VALIDATION_BENCH -u PS_POLICY_DATASET_CAPTURE_DIR \
    -u PS_WITNESS_BASELINE -u PS_RESOURCE_METRICS -u PS_TRIE_CACHE_DIAGNOSTICS \
    -u PS_FORCE_PREVIOUS_CACHE_SNAPSHOT \
    PS_ENGINE_ACCESS=on \
    PS_SHADOW_SAMPLE=50 \
    PS_ENGINE_PAYLOAD=on \
    PS_STREAM_FSYNC="$fsync" \
    PS_STREAM_REORG_CHECKPOINT=always \
    PS_ACCOUNT_WINDOW=90 \
    PS_STORAGE_WINDOW=60 \
    PS_WITNESS_V3=1 \
    PS_TRIE_REPR=exact \
    PS_PARALLEL_INITIAL_PROOF=0 \
    PS_RETAIN_GENERATION=1 \
    PS_CANONICAL_REBUILD=0 \
    PS_SIDECAR_ROLE=builder \
    PS_STREAM_DIR="$arm/spool" \
    PS_BOOTSTRAP_DIR="$arm/bootstrap" \
    PS_SIDECAR_DIR="$arm/sidecars" \
    PS_SHADOW_OUTPUT="$arm/access-shadow.jsonl" \
    nohup "$PRODUCER_BIN" "${NODE_FLAGS[@]}" > "$arm/producer.out" 2>&1 &
    local pid=$!
    echo "$pid" > "$arm/producer.pid"
    # The builder role writes ~2.4 MiB of sidecar per block that nothing here reads. Over a day
    # that is the disk; thirty minutes of retention is far past any window a depth-1 undo needs.
    nohup bash -c "while sleep 300; do find '$arm/sidecars' -type f -mmin +30 -delete 2>/dev/null; done" \
        > /dev/null 2>&1 &
    echo "$!" > "$arm/janitor.pid"
    echo "$pid"
}

# SIGTERM, never SIGKILL: the End frame is what lets the follower finish and the gate judge a
# complete corpus instead of a truncated one.
stop_producer() { # <arm-dir>
    local arm=$1 pid
    pkill -f "while sleep 300; do find '$arm/sidecars'" 2>/dev/null
    pid=$(cat "$arm/producer.pid" 2>/dev/null)
    [ -n "$pid" ] || return 0
    kill -0 "$pid" 2>/dev/null || return 0
    say "SIGTERM -> producer $pid"
    kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 180); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
}

gate_verdict() { # <arm-dir> -> PASSED | FAILED | UNKNOWN
    grep -qE "GATE PASSED" "$1/gate.out" 2>/dev/null && { echo PASSED; return; }
    grep -qE "GATE FAILED" "$1/gate.out" 2>/dev/null && { echo FAILED; return; }
    echo UNKNOWN
}

# --- phase 1: the reorg gate ---------------------------------------------------------------
#
# The one claim only a live reorg can settle. GATE_MODE=reorg fails on zero reorgs by design, and
# has no watcher of its own — the follower ends when it consumes the producer's End — so the wait
# and the stop are here. GATE_FORCE_RESTORE=1 adds the offline half: skimming proves the two
# implementations agree about the block, installing the checkpoint proves the snapshot behind it
# would rebootstrap a validator that has nothing to recover with.

ARM="$BASE/reorg"
if [ "${SKIP_REORG:-0}" != "1" ]; then
    say "=== phase 1: reorg gate (bound ${REORG_WATCH_HOURS}h) ==="
    PID=$(start_producer "$ARM" 0)
    say "producer pid=$PID  log=$ARM/producer.out"
    # Capture only. This phase's corpus is however many blocks a reorg takes to arrive —
    # 2,890 the last time — and its offline half then re-replays all of them twice, which took 56
    # minutes with the datadir sitting empty behind it. The judging waits until the node is back.
    ( cd "$SCRIPTS" && GATE_MODE=reorg GATE_FORCE_RESTORE=1 GATE_PHASE=capture \
        GATE_WAIT_CHECKPOINT_SECS="$GATE_WAIT_CHECKPOINT_SECS" \
        ./run_live_follow_gate.sh "$ARM/spool" "$ARM/out" > "$ARM/gate.out" 2>&1 ) &
    GATE_PID=$!
    say "gate pid=$GATE_PID  log=$ARM/gate.out (a stale datadir warms cold: ~18 min + export)"

    J="$ARM/out/follow.jsonl"
    count() { local n; n=$(grep -c "$1" "$J" 2>/dev/null); echo "${n:-0}"; }
    DEADLINE=$(( $(date +%s) + REORG_WATCH_HOURS * 3600 ))
    ARMED=timeout
    while [ "$(date +%s)" -lt "$DEADLINE" ]; do
        if ! kill -0 "$PID" 2>/dev/null; then ARMED=producer_died; break; fi
        if ! kill -0 "$GATE_PID" 2>/dev/null; then ARMED=gate_exited; break; fi
        if [ "$(count '"kind":"lifecycle"')" -ge 1 ]; then
            say "reorg applied — $(grep '"kind":"lifecycle"' "$J" | tail -1 | cut -c1-160)"
            # The producer publishes the winning branch first and lands the recovery checkpoint
            # at the stream tail a minute or two later; wait for both halves before stopping
            # anything.
            SKIM_BY=$(( $(date +%s) + 600 ))
            while [ "$(count '"kind":"skimmed"')" -lt 1 ] && [ "$(date +%s)" -lt "$SKIM_BY" ]; do
                sleep 15
            done
            say "skimmed=$(count '"kind":"skimmed"') verdicts=$(count '"kind":"verdict"')"
            ARMED=reorg; break
        fi
        sleep 45
    done
    say "phase 1 watch ended: $ARMED"
    stop_producer "$ARM"
    say "waiting for the follower to consume the End"
    wait "$GATE_PID" 2>/dev/null
    pkill -f "reth-partial-stateless node --datadir" 2>/dev/null
    # The datadir goes back to the node *before* the offline half runs. Both passes read files
    # only, so the node syncs through them instead of waiting an hour to start backfilling.
    if [ "${RESTORE:-1}" = 1 ] && [ -x "$RESTORE_VANILLA" ] && ! pgrep -x reth >/dev/null; then
        say "restarting the vanilla node before judging"
        bash "$RESTORE_VANILLA" >> "$BASE/restore.out" 2>&1
        sleep 10
    fi
    say "judging phase 1 offline (batch re-replay, forced restore, render)"
    ( cd "$SCRIPTS" && GATE_MODE=reorg GATE_PHASE=judge \
        ./run_live_follow_gate.sh "$ARM/spool" "$ARM/out" >> "$ARM/gate.out" 2>&1 )
    record "phase1 reorg: watch=$ARMED gate=$(gate_verdict "$ARM") \
verdicts=$(grep -c '"kind":"verdict"' "$J" 2>/dev/null || echo 0) \
reorgs=$(grep -c '"kind":"lifecycle"' "$J" 2>/dev/null || echo 0) report=$ARM/out/result.md"
    # Phase 2 wants the datadir back: stop the vanilla node again the same way the top of this
    # script did, or the arms below will find the writer lock held.
    if pgrep -x reth >/dev/null; then
        say "stopping the vanilla node again for phase 2"
        kill "$(pgrep -x reth)"
        for _ in $(seq 1 300); do pgrep -x reth >/dev/null || break; sleep 2; done
    fi
    sleep 5
fi

# --- phase 2: the fsync durability arm ------------------------------------------------------
#
# ABBA rather than AB: the two arms run at different times against a live chain, so anything
# that drifts over the session — block sizes, peer set, page cache — lands on both letters
# instead of on whichever went second. A = the profile every other run uses, B = the durability
# profile (producer fsyncs each frame, follower fsyncs each ack).
#
# Each arm bounds itself: long mode's watcher SIGTERMs the producer once the live target is met,
# so no manual stop. Back-to-back arms keep the cache within its 90-block gap, so only the first
# pays the cold warm-up.

if [ "${SKIP_ABBA:-0}" != "1" ]; then
    say "=== phase 2: fsync ABBA, ${ABBA_BLOCKS} live verdicts per arm ==="
    INDEX=0
    for LETTER in A B B A; do
        INDEX=$((INDEX + 1))
        ARM="$BASE/fsync-$INDEX$LETTER"
        if [ "$LETTER" = B ]; then FSYNC=1; EXTRA=(--ack-fsync); else FSYNC=0; EXTRA=(); fi
        say "--- arm $INDEX$LETTER: PS_STREAM_FSYNC=$FSYNC ${EXTRA[*]:-(no ack fsync)} ---"
        PID=$(start_producer "$ARM" "$FSYNC")
        say "producer pid=$PID"
        ( cd "$SCRIPTS" && GATE_MODE=long GATE_MIN_BLOCKS="$ABBA_BLOCKS" GATE_REQUIRE_LIVE=1 \
            GATE_WAIT_CHECKPOINT_SECS="$GATE_WAIT_CHECKPOINT_SECS" \
            GATE_PHASE=capture GATE_PRODUCER_PID="$PID" \
            ./run_live_follow_gate.sh "$ARM/spool" "$ARM/out" "${EXTRA[@]}" \
            > "$ARM/gate.out" 2>&1 )
        stop_producer "$ARM"
        pkill -f "reth-partial-stateless node --datadir" 2>/dev/null
        # Same rule as phase 1, for the same reason and at a smaller scale: the offline half reads
        # files only, so the node holds the datadir through it rather than the chain running away
        # from an empty one.
        if [ "${RESTORE:-1}" = 1 ] && [ -x "$RESTORE_VANILLA" ] && ! pgrep -x reth >/dev/null; then
            bash "$RESTORE_VANILLA" >> "$BASE/restore.out" 2>&1
            sleep 10
        fi
        ( cd "$SCRIPTS" && GATE_MODE=long GATE_PHASE=judge \
            ./run_live_follow_gate.sh "$ARM/spool" "$ARM/out" >> "$ARM/gate.out" 2>&1 )
        record "phase2 arm$INDEX$LETTER: fsync=$FSYNC gate=$(gate_verdict "$ARM") \
report=$ARM/out/result.md"
        if pgrep -x reth >/dev/null; then
            kill "$(pgrep -x reth)"
            for _ in $(seq 1 300); do pgrep -x reth >/dev/null || break; sleep 2; done
        fi
        sleep 5
    done
fi

# --- phase 3: the transition-mutation gate --------------------------------------------------
#
# `TransitionMutation::ReceiptsRoot` rebinds the mutated sidecar to the re-sealed block hash and
# drives recorded blocks through a full EVM execution to a post-execution rejection. Corpus-gated
# and `#[ignore]`d, so it needs both the fixture variable and `--ignored` — without either it
# reports success without having run anything. It needs no producer, so it goes last, with the
# node already back on the datadir, and runs against this session's own recording.

say "=== phase 3: transition-mutation gate ==="
MUTATION_SPOOL=${MUTATION_SPOOL:-$BASE/reorg/spool}
[ -d "$MUTATION_SPOOL" ] || MUTATION_SPOOL=$(ls -d "$BASE"/fsync-*/spool 2>/dev/null | head -1)
if [ "${SKIP_MUTATIONS:-0}" != 1 ] && [ -d "${MUTATION_SPOOL:-}" ]; then
    # cargo test can relink the package binary with test-only feature unification. Stamp that
    # relink from the already-verified clean HEAD so it cannot silently replace ps-replay with
    # an unattributable artifact. Phase 3 is last; F1 still performs the runbook's explicit
    # canonical rebuild to restore its declared feature/hash identity.
    LOCK_SHA256=$(sha256sum "$REPO/Cargo.lock" | cut -d' ' -f1)
    ( cd "$REPO" && PS_BUILD_COMMIT="$HEAD_COMMIT" PS_BUILD_DIRTY=0 \
        PS_CARGO_LOCK_SHA256="$LOCK_SHA256" PS_MUTATION_FIXTURE_SPOOL="$MUTATION_SPOOL" \
        cargo test --release -p partial-stateless-replay --test transition_mutations \
        -- --ignored --test-threads=1 > "$BASE/mutations.log" 2>&1 )
    record "phase3 mutations: $([ $? -eq 0 ] && echo PASSED || echo FAILED) \
spool=$MUTATION_SPOOL log=$BASE/mutations.log"
else
    record "phase3 mutations: SKIPPED (no recorded spool at ${MUTATION_SPOOL:-<none>})"
fi

# --- summary --------------------------------------------------------------------------------

say "=== summary ==="
python3 - "$BASE" <<'PY'
import json, pathlib, sys
base = pathlib.Path(sys.argv[1])
rows = []
for dist in sorted(base.glob("*/out/distributions.json")):
    arm = dist.parent.parent.name
    try:
        data = json.loads(dist.read_text())
    except Exception as err:
        rows.append((arm, f"unreadable: {err}", "", "", "", ""))
        continue
    summary = (data.get("follow") or {}).get("summary") or {}
    steady = ((data.get("follow") or {}).get("populations") or {}).get("phase:steady") or {}
    metrics = steady.get("metrics") or {}
    def p50(name):
        row = metrics.get(name)
        return f"{row['p50']:,.0f}" if row else "—"
    ack = summary.get("ack_write_us") or {}
    rows.append((
        arm,
        str(summary.get("blocks_verified")),
        f"{summary.get('reorgs_applied')}/{summary.get('reverts_applied')}",
        p50("standalone_validation_us"),
        p50("decision_latency_us[mtime]"),
        f"{ack.get('p50', '—')}",
    ))
header = ("arm", "blocks", "reorg/revert", "primary p50 us", "latency p50 us", "ack write p50 us")
widths = [max(len(str(r[i])) for r in ([header] + rows)) for i in range(len(header))]
line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths))
print(line(header)); print("  ".join("-" * w for w in widths))
for row in rows:
    print(line(row))
PY
echo
cat "$RESULTS"
say "done — results in $BASE"
