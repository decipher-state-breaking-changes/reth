#!/usr/bin/env bash
# The sequence must clean up once and stop immediately on INT/TERM, while preserving an ordinary
# exit status. This exercises the launcher's actual handler definitions without touching a node.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
LAUNCHER=${LAUNCHER:-$REPO/crates/partial-stateless-exex/scripts/run_f0_sequence.sh}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

HANDLERS=$(sed -n '/^on_exit() {/,/^trap .*on_signal 143.*TERM/p' "$LAUNCHER")
if [ -z "$HANDLERS" ]; then
    echo "FAIL: could not extract signal handlers from $LAUNCHER" >&2
    exit 1
fi

echo "==> cleanup never matches an arbitrary caller command line"
if grep -q 'pkill .*-[^ ]*f\|pkill .* -f' "$LAUNCHER"; then
    echo "FAIL: $LAUNCHER still contains pkill -f" >&2
    exit 1
fi

echo "==> every live/offline gate receives the verified replay binary"
replay_wires=$(grep -c 'PS_REPLAY_BIN="\$REPLAY_BIN"' "$LAUNCHER")
if [ "$replay_wires" -ne 4 ]; then
    echo "FAIL: found $replay_wires replay-binary wires, want 4" >&2
    exit 1
fi

echo "==> TERM cleans once and cannot continue into the next arm"
MARKER=$WORK/signal-marker
MARKER="$MARKER" HANDLERS="$HANDLERS" bash -c '
cleanup() {
    printf "cleanup\n" >> "$MARKER"
    [ -n "${CHILD:-}" ] && kill -TERM "$CHILD" 2>/dev/null
}
eval "$HANDLERS"
sleep 30 &
CHILD=$!
printf "ready\n" >> "$MARKER"
wait "$CHILD"
printf "continued\n" >> "$MARKER"
' &
shell_pid=$!
for _ in $(seq 1 100); do
    [ -f "$MARKER" ] && grep -q '^ready$' "$MARKER" && break
    sleep 0.02
done
kill -TERM "$shell_pid"
signal_rc=0
wait "$shell_pid" || signal_rc=$?
if [ "$signal_rc" -ne 143 ]; then
    echo "FAIL: TERM returned $signal_rc, want 143" >&2
    exit 1
fi
if [ "$(grep -c '^cleanup$' "$MARKER")" -ne 1 ] || grep -q '^continued$' "$MARKER"; then
    echo "FAIL: TERM cleanup count/continuation is wrong" >&2
    cat "$MARKER" >&2
    exit 1
fi

echo "==> EXIT cleanup preserves the command's status"
MARKER=$WORK/exit-marker
MARKER="$MARKER" HANDLERS="$HANDLERS" bash -c '
cleanup() { printf "cleanup\n" >> "$MARKER"; }
eval "$HANDLERS"
exit 7
'
exit_rc=$?
if [ "$exit_rc" -ne 7 ]; then
    echo "FAIL: ordinary exit returned $exit_rc, want 7" >&2
    exit 1
fi
if [ "$(grep -c '^cleanup$' "$MARKER")" -ne 1 ]; then
    echo "FAIL: ordinary exit did not clean exactly once" >&2
    exit 1
fi

echo "==> SIGNAL TRAPS OK"
