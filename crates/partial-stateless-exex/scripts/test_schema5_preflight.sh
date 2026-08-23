#!/usr/bin/env bash
# Datadir-free regression checks for the detached builder schema-5 preflight supervisor.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
LAUNCHER=${LAUNCHER:-$REPO/crates/partial-stateless-exex/scripts/run_schema5_preflight.sh}
WORK=$(mktemp -d)
cleanup_test() {
    local pid
    pid=$(sed -n '1p' "$WORK/run/producer.pid" 2>/dev/null || true)
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -TERM "$pid" 2>/dev/null || true
    rm -rf -- "$WORK"
}
trap cleanup_test EXIT

echo "==> syntax and static lifecycle safety"
bash -n "$LAUNCHER"
if grep -q 'pkill .*-[^ ]*f\|pkill .* -f' "$LAUNCHER"; then
    echo "FAIL: cleanup must never use a caller-matching pkill -f pattern" >&2
    exit 1
fi
grep -q 'nohup setsid.*--worker' "$LAUNCHER"
if grep -qE 'POLICY_BIN|run_offline_weak' "$LAUNCHER"; then
    echo "FAIL: the preflight-only supervisor still contains offline-Weak work" >&2
    exit 1
fi
live_body=$(sed -n '/^run_preflight() {/,/^}/p' "$LAUNCHER")
stop_line=$(printf '%s\n' "$live_body" | grep -n 'stop_vanilla' | head -1 | cut -d: -f1)
wal_line=$(printf '%s\n' "$live_body" | grep -n 'clear_exex_wal' | head -1 | cut -d: -f1)
[ "$stop_line" -lt "$wal_line" ] || {
    echo "FAIL: live phase must stop vanilla before removing the WAL" >&2
    exit 1
}

# Avoid cargo metadata and real release binaries while sourcing the functions under test.
PS_TARGET_DIR="$WORK/target"
# shellcheck disable=SC1090
source "$LAUNCHER"

echo "==> the exact final profile reaches a real child process"
cat > "$WORK/stub-producer" <<'STUB'
#!/usr/bin/env bash
env | grep -E '^(PS_|RUST_LOG=)' | sort
trap 'exit 0' TERM INT
while :; do sleep 1; done
STUB
chmod +x "$WORK/stub-producer"
PRODUCER_BIN="$WORK/stub-producer"
NODE_DATADIR=/data/reth_data
NODE_JWT="$WORK/jwt"
: > "$NODE_JWT"
run="$WORK/run"
start_producer "$run"
pid=$ACTIVE_PRODUCER_PID
sleep 1
[[ "$pid" =~ ^[0-9]+$ ]]
producer_pid_matches "$pid" "$run"

check_assignment() {
    grep -Fqx -- "$1" "$run/producer.out" || {
        echo "FAIL: producer did not receive $1" >&2
        exit 1
    }
}
for assignment in \
    PS_ENGINE_ACCESS=on PS_SHADOW_SAMPLE=50 PS_ENGINE_PAYLOAD=on PS_STREAM_FSYNC=0 \
    PS_STREAM_REORG_CHECKPOINT=always PS_ACCOUNT_WINDOW=90 PS_STORAGE_WINDOW=60 \
    PS_WITNESS_V3=1 PS_TRIE_REPR=exact PS_PARALLEL_INITIAL_PROOF=0 \
    PS_RETAIN_GENERATION=1 PS_CANONICAL_REBUILD=0 PS_SIDECAR_ROLE=builder \
    "PS_STREAM_DIR=$run/spool" "PS_BOOTSTRAP_DIR=$run/bootstrap" \
    "PS_SIDECAR_DIR=$run/sidecars" "PS_SHADOW_OUTPUT=$run/access-shadow.jsonl" \
    "PS_BUILDER_BENCH_OUTPUT=$run/builder.jsonl" \
    RUST_LOG=warn,partial_stateless=info,partial_stateless_stream=info; do
    check_assignment "$assignment"
done
for absent in PS_CAPTURE_DIR PS_VALIDATION_BENCH PS_POLICY_DATASET_CAPTURE_DIR \
              PS_WITNESS_BASELINE PS_RESOURCE_METRICS PS_TRIE_CACHE_DIAGNOSTICS \
              PS_FORCE_PREVIOUS_CACHE_SNAPSHOT; do
    ! grep -q "^$absent=" "$run/producer.out" || {
        echo "FAIL: producer inherited forbidden variable $absent" >&2
        exit 1
    }
done
stop_active_producer

echo "==> detached worker has its own session and ignores a terminal hangup"
mkdir -p "$WORK/detach-bench"
cat > "$WORK/stub-worker" <<'STUB'
#!/usr/bin/env bash
trap 'exit 0' TERM
printf '%s\n' "$$" > "$DETACH_MARKER"
while :; do sleep 1; done
STUB
chmod +x "$WORK/stub-worker"
SCRIPT_PATH="$WORK/stub-worker"
BENCH="$WORK/detach-bench"
BASE="$WORK/detached"
DETACH_MARKER="$WORK/detach-marker"
export DETACH_MARKER
launch_detached > "$WORK/launch.out"
detached_pid=$(read_pid_file "$BASE/launcher.pid")
[ "$(cat "$DETACH_MARKER")" = "$detached_pid" ]
[ "$(ps -o sid= -p "$detached_pid" | tr -d ' ')" = "$detached_pid" ]
kill -HUP "$detached_pid"
sleep 1
kill -0 "$detached_pid"
kill -TERM "$detached_pid"
for _ in $(seq 1 20); do
    kill -0 "$detached_pid" 2>/dev/null || break
    sleep 0.1
done
! kill -0 "$detached_pid" 2>/dev/null
SCRIPT_PATH="$LAUNCHER"

echo "==> schema-5 gate accepts complete records and rejects the old schema"
mkdir -p "$WORK/schema-ok" "$WORK/schema-old"
for i in $(seq 1 50); do
    jq -cn --argjson reused "$([ "$i" -eq 2 ] && echo true || echo false)" \
        --argjson available "$([ "$i" -eq 1 ] && echo true || echo false)" \
        '{cache_parent_synced:true,sidecar_constructed:true,sidecar_published:true,
          schema_version:5,trie_repr:"exact",artifact_available:$available,
          artifact_reused:$reused,shadow_sampled:false,fallback_reason:"not_sampled"}' \
        >> "$WORK/schema-ok/builder.jsonl"
done
validate_builder_schema5 "$WORK/schema-ok" 50
{
    jq -cn '{cache_parent_synced:true,sidecar_constructed:true,sidecar_published:true,
             schema_version:4,trie_repr:"exact",artifact_available:true,
             artifact_reused:true,shadow_sampled:false,fallback_reason:"capture_off"}'
    tail -n +2 "$WORK/schema-ok/builder.jsonl"
} > "$WORK/schema-old/builder.jsonl"
if validate_builder_schema5 "$WORK/schema-old" 50 2>/dev/null; then
    echo "FAIL: schema-4/capture-off record passed the schema-5 gate" >&2
    exit 1
fi

echo "==> TERM exits 143, cleans once, and cannot continue"
marker="$WORK/signal-marker"
set +e
PS_TARGET_DIR="$WORK/target" LAUNCHER="$LAUNCHER" MARKER="$marker" bash -c '
    source "$LAUNCHER"
    BASE=$(dirname "$MARKER")
    STATUS_FILE=
    EVENTS_FILE=
    cleanup() { printf "cleanup\n" >> "$MARKER"; }
    write_status() { :; }
    record_event() { :; }
    trap on_exit EXIT
    trap "on_signal 143 TERM" TERM
    (sleep 0.2; kill -TERM $$) &
    sleep 30 &
    child=$!
    wait "$child"
    printf "continued\n" >> "$MARKER"
'
signal_rc=$?
set -e
[ "$signal_rc" -eq 143 ]
[ "$(grep -c '^cleanup$' "$marker")" -eq 1 ]
! grep -q '^continued$' "$marker"

echo "==> SCHEMA-5 PREFLIGHT REGRESSIONS PASSED"
