#!/usr/bin/env bash
# Detached 50-block builder schema-5 preflight supervisor.
#
# One launch performs the preflight's dependent stages: validated vanilla/WAL handoff, concurrent
# builder-producer plus live-follower capture, then vanilla restoration and file-only
# judge/builder/schema checks. It does not run a paper cohort or the offline Weak control. The supervisor
# survives an SSH disconnect, records phase changes in STATUS.json/EVENTS.jsonl, and restores
# vanilla reth on success, failure, INT, or TERM.
#
# Usage:
#   scripts/run_schema5_preflight.sh start
#   scripts/run_schema5_preflight.sh status /data/bench-runs/schema5-preflight-<timestamp>
#   scripts/run_schema5_preflight.sh stop   /data/bench-runs/schema5-preflight-<timestamp>
#
# Relevant overrides:
#   BASE, BENCH, NODE_DATADIR, NODE_JWT, RESTORE_VANILLA
#   PRODUCER_BIN, REPLAY_BIN, PS_TARGET_DIR, GATE_WAIT_CHECKPOINT_SECS
set -Eeuo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
REPO=${REPO:-$(cd "$(dirname "$SCRIPT_PATH")/../../.." && pwd)}
SCRIPTS="$REPO/crates/partial-stateless-exex/scripts"
BENCH=${BENCH:-/data/bench-runs}
NODE_DATADIR=${NODE_DATADIR:-/data/reth_data}
NODE_JWT=${NODE_JWT:-/data/secrets/jwt.hex}
RESTORE_VANILLA=${RESTORE_VANILLA:-/data/bench-runs/w1-smoke/restore-vanilla.sh}
PREFLIGHT_BLOCKS=${PREFLIGHT_BLOCKS:-50}
GATE_WAIT_CHECKPOINT_SECS=${GATE_WAIT_CHECKPOINT_SECS:-3600}

if [ -z "${PS_TARGET_DIR:-}" ]; then
    PS_TARGET_DIR=$(cd "$REPO" && cargo metadata --format-version 1 --no-deps 2>/dev/null |
        sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')
fi
PS_TARGET_DIR=${PS_TARGET_DIR:-$REPO/target}
PRODUCER_BIN=${PRODUCER_BIN:-$PS_TARGET_DIR/release/reth-partial-stateless}
REPLAY_BIN=${REPLAY_BIN:-$PS_TARGET_DIR/release/ps-replay}

BASE=${BASE:-}
STATUS_FILE=
EVENTS_FILE=
WORKER_PID_FILE=
ACTIVE_GROUP_PID=
ACTIVE_PRODUCER_PID=
ACTIVE_PRODUCER_DIR=
WORK_SUCCEEDED=0
CLEANED=0
PHASE=initializing
LAST_ERROR=

say() { printf '[%s] %s\n' "$(date '+%F %T%z')" "$*"; }

write_status() { # <state> <phase> <message>
    local state=$1 phase=$2 message=$3 tmp
    [ -n "$STATUS_FILE" ] || return 0
    tmp="$STATUS_FILE.tmp.$$"
    jq -n \
        --arg state "$state" --arg phase "$phase" --arg message "$message" \
        --arg base "$BASE" --arg updated_at "$(date --iso-8601=seconds)" \
        --argjson pid "$$" \
        '{state:$state,phase:$phase,message:$message,pid:$pid,base:$base,updated_at:$updated_at}' \
        > "$tmp"
    mv -f -- "$tmp" "$STATUS_FILE"
}

record_event() { # <event> <message>
    local event=$1 message=$2
    [ -n "$EVENTS_FILE" ] || return 0
    jq -cn --arg at "$(date --iso-8601=seconds)" --arg event "$event" \
        --arg phase "$PHASE" --arg message "$message" \
        '{at:$at,event:$event,phase:$phase,message:$message}' >> "$EVENTS_FILE"
}

begin_phase() { # <phase> <message>
    PHASE=$1
    say "$2"
    write_status running "$PHASE" "$2"
    record_event phase_started "$2"
}

finish_phase() { # <message>
    say "$1"
    record_event phase_completed "$1"
}

read_pid_file() {
    local pid
    pid=$(sed -n '1p' "$1" 2>/dev/null) || return 1
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$pid"
}

cmdline_for_pid() {
    [ -r "/proc/$1/cmdline" ] || return 1
    tr '\0' ' ' < "/proc/$1/cmdline"
}

vanilla_pid() {
    local pids
    pids=$(pgrep -x reth 2>/dev/null || true)
    [ -n "$pids" ] || return 1
    if [ "$(printf '%s\n' "$pids" | wc -l)" -ne 1 ]; then
        say "refusing ambiguous datadir handoff: multiple reth processes: $pids" >&2
        return 2
    fi
    printf '%s\n' "$pids"
}

vanilla_pid_matches_datadir() { # <pid>
    local cmd
    cmd=$(cmdline_for_pid "$1") || return 1
    [[ "$cmd" == *"reth node "* ]] || return 1
    [[ "$cmd" == *" --datadir $NODE_DATADIR "* ||
       "$cmd" == *" --datadir=$NODE_DATADIR "* ||
       "$cmd" == *" --datadir $NODE_DATADIR" ]] || return 1
}

producer_pid_matches() { # <pid> <run-dir>
    local cmd
    cmd=$(cmdline_for_pid "$1") || return 1
    [[ "$cmd" == *"$PRODUCER_BIN"*"node --datadir $NODE_DATADIR"* ]] || return 1
    [ -r "/proc/$1/environ" ] || return 1
    tr '\0' '\n' < "/proc/$1/environ" | grep -Fqx "PS_STREAM_DIR=$2/spool"
}

producer_pids_for_datadir() {
    local proc pid cmd found=0
    for proc in /proc/[0-9]*/cmdline; do
        [ -r "$proc" ] || continue
        pid=${proc#/proc/}
        pid=${pid%/cmdline}
        cmd=$(tr '\0' ' ' < "$proc" 2>/dev/null) || continue
        if [[ "$cmd" == *"reth-partial-stateless"*"node --datadir $NODE_DATADIR"* ]]; then
            printf '%s\n' "$pid"
            found=1
        fi
    done
    [ "$found" -eq 1 ]
}

terminate_process_group() { # <group-leader>
    local pid=$1 i
    [[ "$pid" =~ ^[0-9]+$ ]] || return 0
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for i in $(seq 1 15); do
        if ! ps -eo pgid=,stat= | awk -v pgid="$pid" \
            '$1 == pgid && $2 !~ /^Z/ { live=1 } END { exit !live }'; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        sleep 2
    done
    say "cleanup: process group $pid ignored TERM for 30 seconds; sending KILL" >&2
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

stop_active_producer() {
    local pid=${ACTIVE_PRODUCER_PID:-} i
    [[ "$pid" =~ ^[0-9]+$ ]] || return 0
    if producer_pid_matches "$pid" "$ACTIVE_PRODUCER_DIR"; then
        say "cleanup: SIGTERM -> producer $pid"
        kill -TERM "$pid" 2>/dev/null || true
        for i in $(seq 1 180); do
            producer_pid_matches "$pid" "$ACTIVE_PRODUCER_DIR" || break
            sleep 2
        done
    fi
    wait "$pid" 2>/dev/null || true
    ACTIVE_PRODUCER_PID=
    ACTIVE_PRODUCER_DIR=
}

stop_vanilla() {
    local pid i rc
    rc=0
    pid=$(vanilla_pid) || rc=$?
    if [ "$rc" -eq 1 ]; then
        say "vanilla reth is already down"
        return 0
    fi
    [ "$rc" -eq 0 ] || return "$rc"
    vanilla_pid_matches_datadir "$pid" || {
        say "refusing to stop reth pid $pid: command line does not own $NODE_DATADIR" >&2
        return 2
    }
    say "stopping vanilla reth pid $pid for datadir handoff"
    kill -TERM "$pid"
    for i in $(seq 1 300); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 2
    done
    kill -0 "$pid" 2>/dev/null && {
        say "vanilla reth pid $pid did not stop within 600 seconds" >&2
        return 2
    }
    say "vanilla reth stopped"
}

restore_vanilla() {
    local pid i rc
    rc=0
    pid=$(vanilla_pid) || rc=$?
    if [ "$rc" -eq 0 ]; then
        vanilla_pid_matches_datadir "$pid" || {
            say "a reth process is up, but it does not own $NODE_DATADIR" >&2
            return 2
        }
        vanilla_ports_ready && return 0
        say "vanilla reth pid $pid exists; waiting for its RPC listeners"
    else
        [ "$rc" -eq 1 ] || return "$rc"
        [ -x "$RESTORE_VANILLA" ] || {
            say "cannot restore vanilla: $RESTORE_VANILLA is not executable" >&2
            return 2
        }
        say "restoring vanilla reth"
        setsid "$RESTORE_VANILLA" >> "$BASE/restore.out" 2>&1
    fi
    for i in $(seq 1 60); do
        rc=0
        pid=$(vanilla_pid) || rc=$?
        if [ "$rc" -eq 0 ] && vanilla_pid_matches_datadir "$pid" && vanilla_ports_ready; then
            # It must survive past the restore wrapper, not merely appear for one poll.
            sleep 2
            kill -0 "$pid" 2>/dev/null && { say "vanilla reth restored as pid $pid"; return 0; }
        fi
        [ "$rc" -le 1 ] || return "$rc"
        sleep 2
    done
    say "vanilla reth did not become stable within 120 seconds" >&2
    return 2
}

vanilla_ports_ready() {
    local port hex
    for port in 8545 8546 8551; do
        printf -v hex '%04X' "$port"
        awk -v port="$hex" \
            '$2 ~ (":" port "$") && $4 == "0A" { found=1 } END { exit !found }' \
            /proc/net/tcp /proc/net/tcp6 || return 1
    done
}

validate_datadir_target() {
    [ "$NODE_DATADIR" = /data/reth_data ] || {
        say "NODE_DATADIR must be the explicit expected path /data/reth_data, got $NODE_DATADIR" >&2
        return 2
    }
    [ -d "$NODE_DATADIR" ] || { say "datadir does not exist: $NODE_DATADIR" >&2; return 2; }
    [ "$NODE_DATADIR" != / ] && [ "$NODE_DATADIR" != /data ]
}

clear_exex_wal() {
    local rc=0 pids
    validate_datadir_target
    vanilla_pid >/dev/null 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then
        say "refusing WAL removal while vanilla reth is running" >&2
        return 2
    fi
    [ "$rc" -eq 1 ] || { say "refusing WAL removal: reth ownership is ambiguous" >&2; return "$rc"; }
    [ -z "${ACTIVE_PRODUCER_PID:-}" ] || {
        say "refusing WAL removal while a producer is recorded active" >&2
        return 2
    }
    pids=$(producer_pids_for_datadir || true)
    [ -z "$pids" ] || {
        say "refusing WAL removal while partial-stateless producer(s) are active: $pids" >&2
        return 2
    }
    if [ -e "$NODE_DATADIR/exex" ]; then
        say "removing stale ExEx WAL: $NODE_DATADIR/exex"
        rm -rf -- "$NODE_DATADIR/exex"
    else
        say "ExEx WAL is already absent"
    fi
    [ ! -e "$NODE_DATADIR/exex" ]
}

cleanup() {
    local cleanup_rc=0
    [ "$CLEANED" -eq 0 ] || return 0
    CLEANED=1
    if [ -n "${ACTIVE_GROUP_PID:-}" ]; then
        say "cleanup: stopping active process group $ACTIVE_GROUP_PID"
        terminate_process_group "$ACTIVE_GROUP_PID"
        ACTIVE_GROUP_PID=
    fi
    stop_active_producer || cleanup_rc=$?
    restore_vanilla || cleanup_rc=$?
    return "$cleanup_rc"
}

on_error() {
    local rc=$1 line=$2 command=$3
    LAST_ERROR="line $line exited $rc: $command"
    say "ERROR: $LAST_ERROR" >&2
    record_event command_failed "$LAST_ERROR" || true
}

on_exit() {
    local rc=$? cleanup_rc=0 message
    trap - EXIT ERR INT TERM
    set +e
    cleanup
    cleanup_rc=$?
    if [ "$rc" -eq 0 ] && [ "$WORK_SUCCEEDED" -eq 1 ] && [ "$cleanup_rc" -eq 0 ]; then
        : > "$BASE/COMPLETE"
        message="50-block schema preflight completed; vanilla restored"
        write_status completed complete "$message"
        record_event preflight_completed "$message"
    else
        [ "$rc" -ne 0 ] || rc=${cleanup_rc:-1}
        [ "$rc" -ne 0 ] || rc=1
        message=${LAST_ERROR:-"preflight exited with status $rc"}
        [ "$cleanup_rc" -eq 0 ] || message="$message; vanilla restore/cleanup status $cleanup_rc"
        printf '%s\n' "$message" > "$BASE/FAILED"
        write_status failed "$PHASE" "$message"
        record_event preflight_failed "$message"
    fi
    exit "$rc"
}

on_signal() {
    local status=$1 name=$2
    LAST_ERROR="received $name"
    say "$LAST_ERROR; cleaning up and stopping" >&2
    exit "$status"
}

assert_dynamic_governor() {
    local cpu_count governor_count governors driver available expected
    cpu_count=$(getconf _NPROCESSORS_ONLN)
    governor_count=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | wc -l)
    governors=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null |
        sort -u | tr '\n' ' ')
    driver=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>/dev/null)
    available=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors 2>/dev/null)
    case " $available " in
        *" ondemand "*) expected=ondemand ;;
        *" powersave "*) expected=powersave ;;
        *) say "no dynamic governor is available (driver='$driver', available='$available')" >&2; return 2 ;;
    esac
    if [ "$governor_count" -ne "$cpu_count" ] || [ "$governors" != "$expected " ]; then
        say "expected $expected on all $cpu_count online CPUs; observed '$governors' in $governor_count files" >&2
        return 2
    fi
    printf '%s\n' "$expected" > "$BASE/governor.txt"
}

assert_prerequisites() {
    local head dirty bin free_kb needed_kb
    [ "$PREFLIGHT_BLOCKS" -eq 50 ] || { say "PREFLIGHT_BLOCKS must remain 50" >&2; return 2; }
    for bin in jq flock setsid timeout sha256sum; do
        command -v "$bin" >/dev/null || { say "required command is missing: $bin" >&2; return 2; }
    done
    validate_datadir_target
    [ -r "$NODE_JWT" ] || { say "JWT is not readable: $NODE_JWT" >&2; return 2; }
    [ -x "$RESTORE_VANILLA" ] || { say "restore script is not executable: $RESTORE_VANILLA" >&2; return 2; }
    [ -x "$SCRIPTS/run_live_follow_gate.sh" ] || return 2
    [ -x "$SCRIPTS/export_replay_bundle.sh" ] || return 2
    head=$(cd "$REPO" && git rev-parse HEAD)
    dirty=$(cd "$REPO" && git status --porcelain | wc -l)
    [ "$dirty" -eq 0 ] || { say "working tree has $dirty uncommitted paths" >&2; return 2; }
    for bin in "$PRODUCER_BIN" "$REPLAY_BIN"; do
        [ -x "$bin" ] || { say "required binary is not executable: $bin" >&2; return 2; }
        grep -qa "$head" "$bin" || { say "$bin does not carry current HEAD $head" >&2; return 2; }
    done
    free_kb=$(df -Pk "$BENCH" | awk 'NR==2 {print $4}')
    needed_kb=$((5 * 1024 * 1024))
    [ "$free_kb" -ge "$needed_kb" ] || {
        say "insufficient free space: need at least 5 GiB under $BENCH" >&2
        return 2
    }
    assert_dynamic_governor
    {
        echo "head=$head"
        echo "preflight_blocks=$PREFLIGHT_BLOCKS"
        echo "producer_sha256=$(sha256sum "$PRODUCER_BIN" | awk '{print $1}')"
        echo "replay_sha256=$(sha256sum "$REPLAY_BIN" | awk '{print $1}')"
    } > "$BASE/provenance.env"
}

start_producer() { # <run-dir>
    local run=$1 pid
    mkdir -p "$run"/{spool,bootstrap,sidecars,out}
    setsid env -u PS_CAPTURE_DIR -u PS_VALIDATION_BENCH -u PS_POLICY_DATASET_CAPTURE_DIR \
        -u PS_WITNESS_BASELINE -u PS_RESOURCE_METRICS -u PS_TRIE_CACHE_DIAGNOSTICS \
        -u PS_FORCE_PREVIOUS_CACHE_SNAPSHOT \
        PS_ENGINE_ACCESS=on \
        PS_SHADOW_SAMPLE=50 \
        PS_ENGINE_PAYLOAD=on \
        PS_STREAM_FSYNC=0 \
        PS_STREAM_REORG_CHECKPOINT=always \
        PS_ACCOUNT_WINDOW=90 \
        PS_STORAGE_WINDOW=60 \
        PS_WITNESS_V3=1 \
        PS_TRIE_REPR=exact \
        PS_PARALLEL_INITIAL_PROOF=0 \
        PS_RETAIN_GENERATION=1 \
        PS_CANONICAL_REBUILD=0 \
        PS_SIDECAR_ROLE=builder \
        PS_STREAM_DIR="$run/spool" \
        PS_BOOTSTRAP_DIR="$run/bootstrap" \
        PS_SIDECAR_DIR="$run/sidecars" \
        PS_SHADOW_OUTPUT="$run/access-shadow.jsonl" \
        PS_BUILDER_BENCH_OUTPUT="$run/builder.jsonl" \
        RUST_LOG=warn,partial_stateless=info,partial_stateless_stream=info \
        "$PRODUCER_BIN" node --datadir "$NODE_DATADIR" --minimal --http \
        --db.read-transaction-timeout 0 \
        --http.api eth,net,web3,debug,trace \
        --authrpc.addr 127.0.0.1 --authrpc.port 8551 --authrpc.jwtsecret "$NODE_JWT" \
        --ws --ws.addr 127.0.0.1 --ws.port 8546 \
        --ws.api eth,trace,debug,net --ws.origins localhost \
        > "$run/producer.out" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" > "$run/producer.pid"
    ACTIVE_PRODUCER_PID=$pid
    ACTIVE_PRODUCER_DIR=$run
}

run_in_group() { # <stdout/stderr file> <command> [args...]
    local output=$1 rc=0
    shift
    setsid "$@" > "$output" 2>&1 &
    ACTIVE_GROUP_PID=$!
    wait "$ACTIVE_GROUP_PID" || rc=$?
    [ "$rc" -eq 0 ] || terminate_process_group "$ACTIVE_GROUP_PID"
    ACTIVE_GROUP_PID=
    return "$rc"
}

wait_for_producer() { # <pid> <run-dir>
    local pid=$1 run=$2 rc=0 close_count banner_count
    wait "$pid" || rc=$?
    ACTIVE_PRODUCER_PID=
    ACTIVE_PRODUCER_DIR=
    [ "$rc" -eq 0 ] || { say "producer exited with status $rc" >&2; return "$rc"; }
    close_count=$(grep -c 'Closed the event stream' "$run/producer.out" || true)
    banner_count=$(grep -c 'Structured builder benchmark output ENABLED' "$run/producer.out" || true)
    [ "$close_count" -eq 1 ] || { say "expected one close summary, found $close_count" >&2; return 2; }
    [ "$banner_count" -eq 1 ] || { say "expected one builder-output banner, found $banner_count" >&2; return 2; }
}

validate_builder_schema5() { # <run-dir> <sample-count>
    local run=$1 samples=$2
    jq -se --argjson n "$samples" '
      [ .[] | select(.cache_parent_synced == true and
                      .sidecar_constructed == true and
                      .sidecar_published == true) ][0:$n] as $r |
      ($r | length) == $n and
      all($r[];
          .schema_version == 5 and .trie_repr == "exact" and
          has("artifact_available") and has("artifact_reused") and
          has("shadow_sampled") and has("fallback_reason") and
          .fallback_reason != "capture_off") and
      any($r[]; .artifact_available == true) and
      any($r[]; .artifact_reused == true)
    ' "$run/builder.jsonl" > "$run/schema5-check.out"
}

run_preflight() {
    local run=$BASE blocks=$PREFLIGHT_BLOCKS pid capture_secs gate_rc=0
    for output in spool bootstrap sidecars out builder.jsonl bundle; do
        [ ! -e "$run/$output" ] || { say "preflight output already exists: $run/$output" >&2; return 2; }
    done

    begin_phase handoff "preflight: stopping vanilla and clearing its ExEx WAL"
    stop_vanilla
    clear_exex_wal

    begin_phase capture "preflight: starting builder and live standalone validator for 50 verdicts"
    start_producer "$run"
    pid=$ACTIVE_PRODUCER_PID
    capture_secs=$((GATE_WAIT_CHECKPOINT_SECS + blocks * 15 + 1800))
    setsid timeout -k 60 "$capture_secs" \
        env PS_REPLAY_BIN="$REPLAY_BIN" GATE_MODE=long GATE_MIN_BLOCKS="$blocks" \
        GATE_REQUIRE_LIVE=1 GATE_WAIT_CHECKPOINT_SECS="$GATE_WAIT_CHECKPOINT_SECS" \
        GATE_PHASE=capture GATE_PRODUCER_PID="$pid" \
        "$SCRIPTS/run_live_follow_gate.sh" "$run/spool" "$run/out" \
        > "$run/capture.out" 2>&1 &
    ACTIVE_GROUP_PID=$!
    wait "$ACTIVE_GROUP_PID" || gate_rc=$?
    if [ "$gate_rc" -ne 0 ]; then
        terminate_process_group "$ACTIVE_GROUP_PID"
        ACTIVE_GROUP_PID=
        say "preflight capture gate exited with status $gate_rc" >&2
        return "$gate_rc"
    fi
    ACTIVE_GROUP_PID=
    wait_for_producer "$pid" "$run"

    begin_phase restore "preflight: capture closed; restoring vanilla before file-only judging"
    restore_vanilla

    begin_phase judge "preflight: judging closed corpus, builder records, schema, and bundle"
    run_in_group "$run/judge.out" env PS_REPLAY_BIN="$REPLAY_BIN" \
        GATE_MODE=long GATE_PHASE=judge \
        "$SCRIPTS/run_live_follow_gate.sh" "$run/spool" "$run/out"
    grep -q 'GATE PASSED' "$run/judge.out"
    run_in_group "$run/builder-analyzer.out" python3 "$SCRIPTS/analyze_builder_bench.py" \
        --records "$run/builder.jsonl" --warmup 0 --samples "$blocks" --require-published \
        --expect-snapshot skipped --output "$run/builder-report.md"
    validate_builder_schema5 "$run" "$blocks"
    run_in_group "$run/bundle-export.out" "$SCRIPTS/export_replay_bundle.sh" \
        "$run/spool" "$run/bundle" schema5-preflight
    run_in_group "$run/bundle-verify.out" "$run/bundle/verify_corpus.sh" "$run/spool"
    finish_phase "50-block schema preflight: PASSED"
}

worker_main() { # <base>
    BASE=$(readlink -m "$1")
    STATUS_FILE="$BASE/STATUS.json"
    EVENTS_FILE="$BASE/EVENTS.jsonl"
    WORKER_PID_FILE="$BASE/worker.pid"
    mkdir -p "$BASE"
    printf '%s\n' "$$" > "$WORKER_PID_FILE"
    exec 9> "$BENCH/.schema5-preflight.lock"
    flock -n 9 || { say "another schema-5 preflight holds the host lock" >&2; return 2; }

    trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
    trap on_exit EXIT
    trap 'on_signal 130 INT' INT
    trap 'on_signal 143 TERM' TERM

    begin_phase prerequisites "checking freeze, binaries, disk, governor, and datadir paths"
    assert_prerequisites
    finish_phase "prerequisites: PASSED"
    run_preflight
    WORK_SUCCEEDED=1
}

launch_detached() {
    local stamp worker_pid
    stamp=$(date +%Y%m%d-%H%M%S)
    BASE=${BASE:-$BENCH/schema5-preflight-$stamp}
    BASE=$(readlink -m "$BASE")
    if [ -e "$BASE" ] && [ -n "$(find "$BASE" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
        say "BASE is not empty: $BASE" >&2
        return 2
    fi
    mkdir -p "$BASE"
    nohup setsid "$SCRIPT_PATH" --worker "$BASE" </dev/null > "$BASE/preflight.log" 2>&1 &
    worker_pid=$!
    printf '%s\n' "$worker_pid" > "$BASE/launcher.pid"
    sleep 1
    if ! kill -0 "$worker_pid" 2>/dev/null; then
        say "preflight failed during launch; see $BASE/preflight.log" >&2
        return 1
    fi
    printf 'preflight: %s\nworker pid: %s\nlog: %s\nstatus: %s status %q\nstop: %s stop %q\n' \
        "$BASE" "$worker_pid" "$BASE/preflight.log" "$SCRIPT_PATH" "$BASE" "$SCRIPT_PATH" "$BASE"
}

show_status() { # <base>
    local base pid
    base=$(readlink -m "$1")
    [ -s "$base/STATUS.json" ] && jq . "$base/STATUS.json" || say "STATUS.json not written yet"
    pid=$(read_pid_file "$base/worker.pid" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        say "worker $pid is running"
    else
        say "worker is not running"
    fi
    [ -f "$base/preflight.log" ] && tail -n 20 "$base/preflight.log"
}

stop_preflight() { # <base>
    local base pid cmd
    base=$(readlink -m "$1")
    pid=$(read_pid_file "$base/worker.pid") || { say "no valid worker pid in $base" >&2; return 2; }
    cmd=$(cmdline_for_pid "$pid") || { say "worker $pid is no longer running"; return 0; }
    [[ "$cmd" == *"$SCRIPT_PATH --worker $base"* ]] || {
        say "refusing to signal pid $pid: it is not this preflight worker" >&2
        return 2
    }
    kill -TERM "$pid"
    say "TERM sent to preflight worker $pid; it will restore vanilla before exiting"
}

usage() {
    sed -n '2,/^set -/p' "$SCRIPT_PATH" | sed '$d; s/^# \{0,1\}//'
}

main() {
    case "${1:-start}" in
        start) [ "$#" -le 1 ] || { usage; return 64; }; launch_detached ;;
        --worker) [ "$#" -eq 2 ] || return 64; worker_main "$2" ;;
        status) [ "$#" -eq 2 ] || { usage; return 64; }; show_status "$2" ;;
        stop) [ "$#" -eq 2 ] || { usage; return 64; }; stop_preflight "$2" ;;
        -h|--help|help) usage ;;
        *) usage >&2; return 64 ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
