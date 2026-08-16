#!/usr/bin/env python3
"""Renders `result.md` from the analyzer's JSON. Nothing else is read.

Both 1,001-verdict preflights were written up by hand, and both hand-curations made the same
mistake in the same place — the restore backlog pooled into the live latency table — because the
prose was assembled from whatever the terminal happened to show. This renderer takes exactly one
input, `analyze_follow_bench.py --json`, so a number in the report exists in the JSON beside it or
it does not exist at all. Re-deriving anything here from raw JSONL would reintroduce the second
opinion this file exists to remove.

    render_follow_report.py <distributions.json> [--out result.md]

What it refuses to do:

  - print a latency percentile without the phase label and the sample count it came from
  - print a pooled p99 (the analyzer does not emit one, and this does not invent one)
  - hide a warning, an unclassified verdict, or an RB7 cause that never reached its endpoint —
    coverage is a row in the table, not a footnote
  - silently render a JSON whose `schema_version` it does not know

Exit 0 rendered, 2 unreadable or unknown-schema input.
"""

from __future__ import annotations

import argparse
import json
import sys

SCHEMA_VERSION = 1
#: The order the tables read in, primary first. Metrics absent from a run are simply absent.
PRIMARY_METRICS = (
    "standalone_validation_us",
    "delivery_us",
    "admission_us",
    "transition_us",
    "oracle_compare_us",
    "unattributed_validation_us",
)
PHASE_ORDER = ("phase:steady", "phase:recovery", "phase:startup", "phase:unclassified")


def fmt(value, digits=1):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return f"{value:,}"


def table(rows, header):
    """A markdown table, or nothing at all — an empty table is a claim that there was no data."""
    if not rows:
        return []
    widths = [len(cell) for cell in header]
    for row in rows:
        widths = [max(width, len(str(cell))) for width, cell in zip(widths, row)]
    out = [
        "| " + " | ".join(cell.ljust(width) for cell, width in zip(header, widths)) + " |",
        "|" + "|".join("-" * (width + 2) for width in widths) + "|",
    ]
    for row in rows:
        out.append(
            "| " + " | ".join(str(cell).ljust(width) for cell, width in zip(row, widths)) + " |"
        )
    return out + [""]


def metric_rows(metrics, names=None, digits=1):
    rows = []
    for name in names or sorted(metrics):
        row = metrics.get(name)
        if not row:
            continue
        rows.append([
            name,
            f"{row['count']}/{row['population']}",
            fmt(row["nulls"]),
            fmt(row["p50"], digits),
            fmt(row["p95"], digits),
            fmt(row["p99"], digits),
            fmt(row["max"], digits),
        ])
    return rows


METRIC_HEADER = ["metric", "n/population", "nulls", "p50", "p95", "p99", "max"]


def render_gate(result, lines):
    follow = (result.get("follow") or {}).get("summary") or {}
    batch = (result.get("batch") or {}).get("summary") or {}
    if not follow and not batch:
        return
    lines.append("## Gate status")
    lines.append("")
    rows = []

    def add(label, record, expr):
        if record:
            rows.append([label, str(expr(record))])

    add("follower agreed", follow, lambda r: r.get("agreed"))
    add("follower continuous", follow, lambda r: r.get("continuous"))
    add("blocks verified", follow, lambda r: r.get("blocks_verified"))
    add("catch-up blocks", follow, lambda r: r.get("catch_up_blocks"))
    add("disagreements / failures", follow,
        lambda r: f"{r.get('disagreements')} / {r.get('failures')}")
    add("reorgs / reverts applied", follow,
        lambda r: f"{r.get('reorgs_applied')} / {r.get('reverts_applied')}")
    add("restores (continuous / reset)", follow,
        lambda r: f"{r.get('restores')} ({r.get('restores_continuous')} / {r.get('restores_reset')})")
    add("late-skim mismatches", follow, lambda r: r.get("late_skim_mismatches", 0))
    add("rewind windows refused", follow, lambda r: r.get("rewind_windows_refused", 0))
    add("recovery checkpoints pending at end", follow,
        lambda r: r.get("recovery_checkpoints_pending_at_end", 0))
    add("batch agreed / continuous / closed", batch,
        lambda r: f"{r.get('agreed')} / {r.get('continuous')} / {r.get('closed')}")
    lines.extend(table(rows, ["field", "value"]))


def render_runs(result, lines):
    runs = (result.get("follow") or {}).get("runs") or []
    if not runs:
        return
    lines.append("## Runs analyzed")
    lines.append("")
    rows = [
        [run["file"], str(run["segment"]), str(run.get("label")), str(run["complete"])]
        for run in runs
    ]
    lines.extend(table(rows, ["file", "segment", "label", "closed by a summary"]))


def render_populations(result, lines):
    follow = result.get("follow")
    if not follow:
        return
    populations = follow["populations"]
    lines.append("## Populations")
    lines.append("")
    lines.append(
        "Named and never pooled. `raw_work` is every verdict the process(es) published; "
        "`unique_frames` collapses re-publications of one frame sequence; `delivered` is "
        "first-time verification; `catch_up` is re-derivation on the way back to a watermark."
    )
    lines.append("")
    rows = [
        [name, str(body["verdicts"]), str(body["live_tail_verdicts"])]
        for name, body in populations.items()
    ]
    lines.extend(table(rows, ["population", "verdicts", "live-tail verdicts"]))

    primary = populations.get("delivered", {}).get("metrics", {})
    if primary:
        lines.append("### Primary validation cost — `delivered` (µs)")
        lines.append("")
        lines.extend(table(metric_rows(primary, PRIMARY_METRICS), METRIC_HEADER))

    lines.append("### By phase (µs)")
    lines.append("")
    lines.append(
        "One table per phase because they are different measurements: `steady` is the live tail, "
        "`recovery` is a rewind window replayed out of the spool, `startup` is backlog, and "
        "`unclassified` is a recording that predates the tail flag. A percentile without one of "
        "these labels is not reported."
    )
    lines.append("")
    for name in PHASE_ORDER:
        body = populations.get(name)
        if not body or not body["verdicts"]:
            continue
        rows = metric_rows(body["metrics"])
        lines.append(f"**{name}** — {body['verdicts']} verdicts")
        lines.append("")
        lines.extend(table(rows, METRIC_HEADER))


def render_paired(result, lines):
    paired = result.get("paired")
    if not paired:
        return
    lines.append("## Live versus batch, paired by frame sequence")
    lines.append("")
    lines.append(
        f"{paired['joined']} frames joined; {paired['follow_only']} seen only by the follower, "
        f"{paired['batch_only']} only by the batch replay, {paired['block_mismatched']} refused "
        f"for naming different blocks under one sequence. The ratio is per frame — never a "
        f"quotient of two percentiles."
    )
    lines.append("")
    rows = []
    for name, digits in (("standalone_validation_ratio", 3),
                         ("standalone_validation_delta_us", 1)):
        rows.extend(metric_rows(paired, [name], digits))
    lines.extend(table(rows, METRIC_HEADER))


def render_rb7(result, lines):
    rb7 = result.get("producer_events")
    if not rb7:
        return
    lines.append("## RB7 — recovery lifecycle")
    lines.append("")
    ledger = rb7["causes"]
    lines.append(
        f"Epoch {rb7['epoch']}, {ledger['total']} detected cause(s). Coverage is the denominator "
        f"of every interval below: a cause that was fenced or ran out of retries never reaches "
        f"its endpoint, and the table says so rather than shrinking the population to fit."
    )
    lines.append("")
    rows = []
    for axis in ("by_origin", "by_state", "first_winning"):
        for key, count in (ledger.get(axis) or {}).items():
            rows.append([axis, key, str(count)])
    rows.append(["attempts", "started", str(ledger.get("attempts_started", 0))])
    rows.append(["causes", "with a lifecycle origin", str(ledger.get("origin_anchored", 0))])
    lines.extend(table(rows, ["axis", "value", "causes"]))

    intervals = rb7["intervals"]
    if intervals:
        lines.append("### Intervals")
        lines.append("")
        lines.append(
            "Producer-internal intervals difference the monotonic clock. "
            "`first_winning_to_follower_verdict_ms` is the only cross-process one and carries "
            "whatever offset separates the two wall clocks; it is reported in its own unit and "
            "never mixed into the others."
        )
        lines.append("")
        rows = []
        for name, row in intervals.items():
            rows.append([
                f"{name} ({row['unit']})",
                f"{row['count']}/{row['population']}",
                fmt(row["nulls"]),
                fmt(row["p50"]),
                fmt(row["p95"]),
                fmt(row["p99"]),
                fmt(row["max"]),
            ])
        lines.extend(table(rows, METRIC_HEADER))

    anomalies = rb7["anomalies"]
    rejected, unjoined = anomalies["rejected"], anomalies["unjoined_first_winning"]
    if rejected or unjoined or anomalies.get("follower_join") != "joined":
        lines.append("### Refused samples")
        lines.append("")
        rows = [
            [str(item.get("interval")), str(item.get("cause_id")), str(item.get("why"))]
            for item in rejected
        ]
        rows.extend(
            ["first_winning_to_follower_verdict_ms", str(item["cause_id"]),
             f"no follower verdict at sequence {item['sequence']}"]
            for item in unjoined
        )
        if anomalies.get("follower_join") != "joined":
            rows.append(["first_winning_to_follower_verdict_ms", "—",
                         "no follower input was given"])
        lines.extend(table(rows, ["interval", "cause", "why"]))

    if rb7.get("per_cause"):
        lines.append("### Per cause")
        lines.append("")
        rows = []
        for row in rb7["per_cause"]:
            measured = row["intervals"]
            rows.append([
                str(row["cause_id"]),
                row["origin_kind"],
                row["state"],
                str(row["attempts"]),
                fmt(measured.get("detection_to_publication_us")),
                fmt(measured.get("detection_to_first_winning_us")),
                str(row.get("first_winning_unmeasured") or row.get("first_winning_sequence") or "—"),
            ])
        lines.extend(table(rows, [
            "cause", "origin", "state", "attempts",
            "detection→publication (µs)", "detection→first winning (µs)", "first winning",
        ]))


def render_provenance(result, lines):
    lines.append("## Provenance")
    lines.append("")
    inputs = result.get("inputs", {})
    rows = [[key, str(value)] for key, value in inputs.items()]
    lines.extend(table(rows, ["input", "path"]))
    provenance = result.get("provenance")
    if provenance:
        lines.append("The producer's run manifest, verbatim:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(provenance, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")


def render_warnings(result, lines):
    warnings = result.get("warnings") or []
    if not warnings:
        return
    lines.append("## Warnings")
    lines.append("")
    for warning in warnings:
        lines.append(f"- {warning}")
    lines.append("")


def render(result):
    if result.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(
            f"error: this renderer reads analyzer schema {SCHEMA_VERSION}, "
            f"not {result.get('schema_version')!r}"
        )
    lines = ["# Standalone follow report", ""]
    lines.append(f"Generated from `{result.get('generated_by')}` schema {SCHEMA_VERSION} output.")
    lines.append("")
    render_warnings(result, lines)
    render_gate(result, lines)
    render_runs(result, lines)
    render_populations(result, lines)
    render_paired(result, lines)
    render_rb7(result, lines)
    render_provenance(result, lines)
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("distributions", help="the analyzer's --json output")
    parser.add_argument("--out", metavar="PATH", help="write here instead of stdout")
    args = parser.parse_args()

    try:
        with open(args.distributions, encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError) as err:
        print(f"error: {args.distributions}: {err}", file=sys.stderr)
        return 2
    body = render(result)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(body)
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
