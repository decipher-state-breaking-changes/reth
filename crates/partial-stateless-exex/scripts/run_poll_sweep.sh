#!/usr/bin/env bash
# The S5c transport-wait measurement: three sequential follower arms against the SAME live
# producer, at --poll-ms 200 / 25 / 5, each measuring live-tail verdicts. What it decides is
# whether a Unix socket is worth building at all — history's rule is that the decision waits on
# the measured delivery term, and this is that measurement.
#
# The population is the whole point. The first arm starts fresh: everything it reads down to the
# live tail is backlog (`tail_live: false` in the records) and is excluded from the analysis; its
# live samples begin at the first empty poll. The later arms `--resume` from the shared ack the
# previous arm left, so they start at the tail and poll live almost immediately. Judged samples
# are `tail_live == true` only.
#
# The producer is NOT started here (it needs the datadir; the operator's move). fsync must be off
# for the sweep — the availability proxy is the frame's mtime, and the power-loss profile moves
# the write's completion without moving the stamp.
#
# Usage:
#     run_poll_sweep.sh <spool-dir> <output-dir> [live-blocks-per-arm]
#
# Judged with the pre-registered relative rule, printed at the end:
#   - build the socket only if, at 5 ms, p95 live queue wait exceeds 10% of the p50 live
#     validation time, or the follower uses more than 0.5 of a core (process CPU seconds over
#     wall seconds);
#   - otherwise the socket closes "measured, not needed", and this report is the record.
# The p95/p95 ratio is printed beside the registered p95/p50 one, and the raw distributions
# stay in the arm files, so the verdict is reproducible under any other threshold.
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
    exit 64
fi

SPOOL_DIR=$1
OUT_DIR=$2
BLOCKS=${3:-300}
mkdir -p "$OUT_DIR"
ACK_FILE="$OUT_DIR/sweep-ack.json"

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
if [ -z "${PS_REPLAY_BIN:-}" ]; then
    TARGET_DIR=$(cd "$REPO_ROOT" && cargo metadata --format-version 1 --no-deps 2>/dev/null |
        sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')
    PS_REPLAY_BIN="${TARGET_DIR:-$REPO_ROOT/target}/release/ps-replay"
fi
[ -x "$PS_REPLAY_BIN" ] || { echo "error: $PS_REPLAY_BIN not built" >&2; exit 1; }

FIRST=1
for MS in 200 25 5; do
    ARM_JSON="$OUT_DIR/arm-${MS}ms.jsonl"
    ARM_TIME="$OUT_DIR/arm-${MS}ms.time"
    if [ "$FIRST" -eq 1 ]; then
        # The fresh arm replays the spool's backlog before it reaches the tail, and --max-blocks
        # counts backlog verdicts too, so its budget is the current commit count plus the live
        # target plus a reorg cushion. The analysis keeps only the tail_live samples.
        BACKLOG=$(find "$SPOOL_DIR" -maxdepth 1 -name '*_commit.frame' 2>/dev/null | wc -l)
        CAP=$((BACKLOG + BLOCKS + 50))
        RESUME_ARGS=()
        echo "==> arm --poll-ms $MS: fresh start, $BACKLOG backlog + $BLOCKS live (cap $CAP)"
    else
        # Resumed from the previous arm's ack: a short catch-up from the restored checkpoint to
        # the watermark (labelled catch_up, not counted by --max-blocks), then live at the tail.
        CAP=$((BLOCKS + 30))
        RESUME_ARGS=(--resume)
        echo "==> arm --poll-ms $MS: --resume from the shared ack, $BLOCKS live (cap $CAP)"
    fi
    /usr/bin/time -v -o "$ARM_TIME" \
        "$PS_REPLAY_BIN" --follow "$SPOOL_DIR" \
        --poll-ms "$MS" --max-blocks "$CAP" \
        --ack "$ACK_FILE" "${RESUME_ARGS[@]}" \
        --json "$ARM_JSON" --label "s5c-sweep-${MS}ms" ||
        { echo "arm $MS failed" >&2; exit 1; }
    FIRST=0
done

echo "==> sweep report (tail_live samples only)"
python3 - "$OUT_DIR" "$BLOCKS" <<'PYEOF'
import json, re, sys

out, target = sys.argv[1], int(sys.argv[2])
print(f"{'arm':>8} {'n_live':>6} {'qw_p50':>9} {'qw_p95':>9} {'val_p50':>9} {'val_p95':>9} "
      f"{'p95qw/p50val':>12} {'p95qw/p95val':>12} {'cores':>7}")
rows = {}
for ms in (200, 25, 5):
    live = []
    for line in open(f"{out}/arm-{ms}ms.jsonl"):
        record = json.loads(line)
        if (record.get("kind") == "verdict" and record.get("tail_live") and
                not record.get("catch_up")):
            live.append(record)
    qw = sorted(v["queue_wait_us"] for v in live if v.get("queue_wait_us") is not None)
    val = sorted(
        v["standalone_validation_us"]
        for v in live
        if v.get("standalone_validation_us") is not None
    )
    pct = lambda xs, f: xs[min(len(xs) - 1, round(f * (len(xs) - 1)))] if xs else None
    timing = open(f"{out}/arm-{ms}ms.time").read()
    def clock(name):
        m = re.search(rf"{name}.*?([\d.]+)$", timing, re.M)
        return float(m.group(1)) if m else None
    user, system = clock("User time"), clock("System time")
    # The Elapsed header itself contains colons ("(h:mm:ss or m:ss)"), so the value is taken as
    # the line's last token and split: h:mm:ss past an hour, m:ss.cs below one.
    wall = None
    for line in timing.splitlines():
        if line.strip().startswith("Elapsed (wall clock)"):
            parts = line.rsplit(" ", 1)[-1].split(":")
            try:
                wall = float(parts[-1])
            except ValueError:
                break
            if len(parts) >= 2:
                wall += int(parts[-2]) * 60
            if len(parts) == 3:
                wall += int(parts[-3]) * 3600
            break
    cores = (user + system) / wall if user is not None and system is not None and wall else None
    ratio_p50 = (pct(qw, 0.95) / pct(val, 0.50)) if qw and val else None
    ratio_p95 = (pct(qw, 0.95) / pct(val, 0.95)) if qw and val else None
    rows[ms] = (ratio_p50, cores, len(live))
    fmt = lambda r: f"{r:.3f}" if r is not None else "n/a"
    print(
        f"{ms:>6}ms {len(live):>6} {pct(qw, 0.50) or 0:>9} {pct(qw, 0.95) or 0:>9} "
        f"{pct(val, 0.50) or 0:>9} {pct(val, 0.95) or 0:>9} {fmt(ratio_p50):>12} "
        f"{fmt(ratio_p95):>12} {f'{cores:.2f}' if cores is not None else 'n/a':>7}"
    )
    if len(live) < target * 0.9:
        print(f"    warning: arm {ms}ms holds {len(live)} live samples, under 90% of the "
              f"{target} target — extend the arm before judging")

ratio, cores, n_live = rows.get(5, (None, None, 0))
print("\npre-registered rule at the 5 ms arm: build the socket only if")
print(f"  p95 live queue wait / p50 live validation > 0.10  -> measured: "
      f"{f'{ratio:.3f}' if ratio is not None else 'n/a'}")
print(f"  or follower cores-used > 0.50                     -> measured: "
      f"{f'{cores:.2f}' if cores is not None else 'n/a'}")
if ratio is not None and cores is not None:
    verdict = "BUILD the hybrid socket" if (ratio > 0.10 or cores > 0.5) else \
              "CLOSE the socket as measured-not-needed (B1/V3 precedent)"
    print(f"  -> {verdict}")
print("\nraw live/backlog/catch-up records are in the arm-*.jsonl files (tail_live labels the")
print("population); the verdict is reproducible under any other threshold from the same records.")
PYEOF
