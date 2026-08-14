#!/usr/bin/env python3
"""Distributions for standalone follow/replay JSONL records (schema v2).

Three populations are reported separately, because they answer different questions and pooling
them misattributes real work:

  delivered   every verdict published, in sequence order — the work the process actually did
  canonical   the verdicts still standing at the end — the chain's cost
  abandoned   verdicts for blocks the chain later left — reorg/revert branch work, which is real
              resource consumption and not double counting

Abandonment is read from the follower's own lifecycle records first: a `reorg_applied` or
`revert_applied` line names the abandoned blocks by number *and hash*, which is what catches a
pure revert (a block undone with nothing replacing it at that height). The same-height
supersession rule — a later verdict at the same height abandons the earlier one — remains as the
fallback for batch records, which carry no lifecycle lines.

Catch-up verdicts (a resumed run re-deriving already-acknowledged frames) are their own bucket:
their validation and delivery costs are real, but their mtime-derived latency fields are history
rather than latency and are null in the records by construction. Backlog verdicts
(`tail_live: false` — frames that were already in the spool before the follower reached its live
tail) have real validation costs too, and null latency for the same reason; the latency
distributions therefore only ever hold live-tail samples.

Latency fields are aggregated per `available_at_source` and never pooled across sources: an
"mtime" reading and a socket reading are different clocks with different biases.

Keys are frame sequences, never block heights. Absent (null) fields are excluded per metric and
the surviving sample count is printed beside every row; null is not zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

TIMING_FIELDS = (
    "standalone_validation_us",
    "delivery_us",
    "admission_us",
    "transition_us",
    "oracle_compare_us",
    "unattributed_validation_us",
)
LATENCY_FIELDS = ("queue_wait_us", "decision_latency_us")
PHASE_PREFIX = "phases."


def load_verdicts(paths):
    """Verdict-shaped dicts from follow JSONL lines and batch-record block arrays.

    Lifecycle records are consumed in file order as they are met: each names the blocks a reorg
    or revert removed from the chain, and the most recent verdict for that exact (number, hash)
    is marked abandoned. Order matters — a block abandoned and later re-verdicted on the winning
    branch is abandoned once, not twice.
    """
    verdicts = []
    for path in paths:
        latest_by_id = {}
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as err:
                    print(f"{path}:{line_number}: unreadable line skipped: {err}", file=sys.stderr)
                    continue
                kind = record.get("kind")
                if kind == "verdict":
                    verdicts.append(record)
                    key = (record.get("block"), record.get("block_hash"))
                    latest_by_id[key] = record
                elif kind == "lifecycle":
                    numbers = record.get("abandoned") or []
                    hashes = record.get("abandoned_hashes") or []
                    for number, block_hash in zip(numbers, hashes):
                        gone = latest_by_id.pop((number, block_hash), None)
                        if gone is not None:
                            gone["abandoned_by_lifecycle"] = True
                elif isinstance(record.get("blocks"), list):
                    for block in record["blocks"]:
                        block = dict(block)
                        block.setdefault("block", block.get("number"))
                        block.setdefault("catch_up", False)
                        verdicts.append(block)
    return verdicts


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def summarize(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": values[0],
        "max": values[-1],
    }


def split_populations(verdicts):
    """Sequence-ordered split into live/catch-up, then canonical/abandoned.

    A verdict is abandoned when the follower's lifecycle record named its exact (number, hash)
    as leaving the chain — the authority, and the only signal that sees a pure revert — or, as
    the fallback that batch records need, when a later verdict at the same height superseded it.
    """
    ordered = sorted(verdicts, key=lambda v: v.get("sequence", 0))
    live = [v for v in ordered if not v.get("catch_up")]
    catch_up = [v for v in ordered if v.get("catch_up")]
    last_at_height = {}
    for verdict in live:
        height = verdict.get("block")
        if height is not None:
            last_at_height[height] = verdict.get("sequence")
    canonical, abandoned = [], []
    for verdict in live:
        height = verdict.get("block")
        superseded = height is not None and last_at_height.get(height) != verdict.get("sequence")
        if verdict.get("abandoned_by_lifecycle") or superseded:
            abandoned.append(verdict)
        else:
            canonical.append(verdict)
    return live, canonical, abandoned, catch_up


def collect_metrics(verdicts, include_phases):
    metrics = {}
    for field in TIMING_FIELDS:
        metrics[field] = summarize(v.get(field) for v in verdicts)
    # Latency readings are grouped by the clock they came from and never pooled across clocks.
    by_source = defaultdict(list)
    for verdict in verdicts:
        source = verdict.get("available_at_source")
        if source:
            by_source[source].append(verdict)
    for source, group in sorted(by_source.items()):
        for field in LATENCY_FIELDS:
            metrics[f"{field}[{source}]"] = summarize(v.get(field) for v in group)
    if include_phases:
        names = set()
        for verdict in verdicts:
            names.update((verdict.get("phases") or {}).keys())
        for name in sorted(names):
            metrics[PHASE_PREFIX + name] = summarize(
                (v.get("phases") or {}).get(name) for v in verdicts
            )
    return {name: stats for name, stats in metrics.items() if stats}


def report(populations, include_phases):
    out = {}
    for name, verdicts in populations.items():
        out[name] = {
            "verdicts": len(verdicts),
            "live_tail_verdicts": sum(1 for v in verdicts if v.get("tail_live")),
            "metrics": collect_metrics(verdicts, include_phases),
        }
    return out


def print_human(result):
    for population, body in result.items():
        print(
            f"\n== {population} ({body['verdicts']} verdicts, "
            f"{body['live_tail_verdicts']} live-tail) =="
        )
        for metric, stats in body["metrics"].items():
            print(
                f"  {metric:44s} n={stats['count']:6d}  mean={stats['mean']:12.1f}  "
                f"p50={stats['p50']:10d}  p95={stats['p95']:10d}  p99={stats['p99']:10d}"
            )


def self_check():
    """The attribution rules, run against a synthetic stream that exercises each of them."""
    import os
    import tempfile

    records = [
        # A canonical run with one reorg: height 101 is verdicted twice, the lifecycle names the
        # loser, and the same-height rule would agree — either signal alone abandons it.
        {"kind": "verdict", "sequence": 10, "block": 100, "block_hash": "0xaa",
         "standalone_validation_us": 100, "catch_up": False, "tail_live": True,
         "queue_wait_us": 7, "available_at_source": "mtime", "phases": {"evm_us": 40}},
        {"kind": "verdict", "sequence": 11, "block": 101, "block_hash": "0xb1",
         "standalone_validation_us": 200, "catch_up": False},
        {"kind": "lifecycle", "event": "reorg_applied", "common_ancestor": 100,
         "abandoned": [101], "abandoned_hashes": ["0xb1"]},
        {"kind": "verdict", "sequence": 13, "block": 101, "block_hash": "0xb2",
         "standalone_validation_us": 300, "catch_up": False},
        # A pure revert: block 102 is undone and nothing replaces it at that height, so only the
        # lifecycle record can say it left the chain — the same-height rule never fires.
        {"kind": "verdict", "sequence": 14, "block": 102, "block_hash": "0xcc",
         "standalone_validation_us": 400, "catch_up": False,
         "queue_wait_us": None, "available_at_source": None},
        {"kind": "lifecycle", "event": "revert_applied", "common_ancestor": 101,
         "abandoned": [102], "abandoned_hashes": ["0xcc"]},
        # Catch-up: validation cost is real, latency is absent by construction.
        {"kind": "verdict", "sequence": 5, "block": 99, "block_hash": "0x99",
         "standalone_validation_us": 50, "catch_up": True},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        path = handle.name
    try:
        verdicts = load_verdicts([path])
    finally:
        os.unlink(path)
    live, canonical, abandoned, catch_up = split_populations(verdicts)
    assert [v["sequence"] for v in live] == [10, 11, 13, 14]
    assert [v["sequence"] for v in canonical] == [10, 13], "the survivors are 100 and 101@0xb2"
    assert [v["sequence"] for v in abandoned] == [11, 14], (
        "the reorged verdict AND the purely-reverted one are abandoned work"
    )
    assert [v["sequence"] for v in catch_up] == [5]
    metrics = collect_metrics(live, include_phases=True)
    assert metrics["standalone_validation_us"]["count"] == 4
    assert metrics["queue_wait_us[mtime]"]["count"] == 1, "null latency is excluded, not zeroed"
    assert metrics["phases.evm_us"]["count"] == 1
    print("self-check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="follow/replay JSONL files")
    parser.add_argument("--phases", action="store_true", help="include per-phase distributions")
    parser.add_argument("--json", metavar="PATH", help="write the report as JSON")
    parser.add_argument("--self-check", action="store_true", help="run the built-in checks")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return
    if not args.files:
        parser.error("no input files (or use --self-check)")

    verdicts = list(load_verdicts(args.files))
    live, canonical, abandoned, catch_up = split_populations(verdicts)
    result = report(
        {
            "delivered": live,
            "canonical": canonical,
            "abandoned": abandoned,
            "catch_up": catch_up,
        },
        args.phases,
    )
    print_human(result)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
