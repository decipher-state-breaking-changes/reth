#!/usr/bin/env python3
"""Decide the B3 stage-3 gate from an access-shadow JSONL.

The gate is not "no misses". Artifacts cannot exist for backfilled blocks, for notifications
replayed from the WAL after a restart, or for a sibling already evicted, and the builder falls
back to its own execution in each case. The gate is that among the blocks that *did* hit,
divergence is zero, with enough evidence behind that zero.

"Enough" is about coverage, not block count. The divergence classes this stage exists to close --
prewarming and the cross-block cache, read-only and reverted access survival, system calls,
payload-versus-historical settings -- are systematic: they are exercised by every mainnet block
and would show up in the first tens of them, not the six-thousandth. What block count buys is
only a tighter bound on a hypothetical rare divergence (0 in N bounds the rate at about 3/N),
and what it does *not* buy at any realistic N is a reorg or a restart, which are time-conditional.
So the threshold here is modest, and two coverage facts are reported alongside it:

  - compared keys, the denominator a zero mismatch count is a zero *of*;
  - whether any block observed BLOCKHASH, because `lowest_block_mismatched` is false when
    neither side read it, and a run of such blocks passes that check vacuously.

A sustained miss rate is reported separately. It does not fail the gate -- drop-and-fallback is
the design -- but it bounds how much of the stage 5 win stage 4 can actually collect.

Usage:
    python3 analyze_access_shadow.py <run-dir-or-jsonl> [--required 1000]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

KEY_FIELDS = (
    "accounts_only_simulated",
    "accounts_only_captured",
    "accounts_mismatched",
    "storage_only_simulated",
    "storage_only_captured",
    "storage_mismatched",
    "codes_only_simulated",
    "codes_only_captured",
    "codes_mismatched",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=pathlib.Path, help="access_shadow.jsonl, or a run directory containing it")
    parser.add_argument("--required", type=int, default=1000, help="hits required to close the gate")
    parser.add_argument("--show-samples", type=int, default=5, help="diverging blocks to print in full")
    return parser.parse_args()


def resolve(path: pathlib.Path) -> pathlib.Path:
    if path.is_dir():
        path = path / "access_shadow.jsonl"
    if not path.is_file():
        raise SystemExit(f"no shadow output at {path}")
    return path


def load(path: pathlib.Path):
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A run killed mid-write leaves one truncated final line; everything before it is good.
            print(f"warning: skipping unparsable line {line_number}", file=sys.stderr)
    return records


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main():
    args = parse_args()
    path = resolve(args.path)
    records = load(path)
    if not records:
        raise SystemExit(f"{path} contains no records")

    hits = [r for r in records if r.get("artifact_hit")]
    misses = [r for r in records if not r.get("artifact_hit")]
    diverging = [r for r in hits if not r.get("clean", True)]

    blocks = [r["block_number"] for r in records]
    capture = [r["engine_capture_us"] for r in hits if r.get("engine_capture_us") is not None]
    simulation = [r["builder_simulation_us"] for r in records if r.get("builder_simulation_us")]

    print(f"source              {path}")
    print(f"blocks              {len(records)}  ({min(blocks)}..{max(blocks)})")
    print(f"artifact hits       {len(hits)}  ({len(hits) / len(records):.1%})")
    print(f"artifact misses     {len(misses)}  ({len(misses) / len(records):.1%})")
    print(f"diverging blocks    {len(diverging)}")

    last = records[-1]
    print(
        "handoff             "
        f"inserted={last.get('handoff_inserted')} taken={last.get('handoff_taken')} "
        f"missed={last.get('handoff_missed')} dropped_capacity={last.get('handoff_dropped_capacity')} "
        f"dropped_contended={last.get('handoff_dropped_contended')} replaced={last.get('handoff_replaced')}"
    )
    print(
        "residence           "
        f"mean={last.get('handoff_mean_residence_us')}us depth={last.get('handoff_mean_depth_at_take')} "
        f"queue={last.get('handoff_queue_depth')} resident_bytes={last.get('handoff_resident_bytes')}"
    )
    compared = [r["compared_keys"] for r in hits if r.get("compared_keys") is not None]
    uncovered = [r for r in hits if r.get("compared_keys") is None]
    blockhash_blocks = [r for r in hits if r.get("blockhash_observed")]
    per_category = {
        name: sum(r.get(f"{name}_compared") or 0 for r in hits)
        for name in ("accounts", "storage", "codes")
    }
    if compared:
        print(
            "compared keys       "
            f"{sum(compared):,} total, median {percentile(compared, 0.5):,}/block  "
            f"(accounts {per_category['accounts']:,} / storage {per_category['storage']:,} / "
            f"codes {per_category['codes']:,})"
        )
        print(
            "blockhash coverage  "
            f"{len(blockhash_blocks)} of {len(hits)} blocks read BLOCKHASH"
        )
        if uncovered:
            print(f"coverage gaps       {len(uncovered)} hits carry no denominator")
    else:
        print("compared keys       not recorded (schema 1 run; no denominator for the zero)")

    if misses:
        reasons = Counter(r.get("miss_reason") or "unattributed" for r in misses)
        print(
            "miss attribution    "
            + ", ".join(f"{reason}={count}" for reason, count in reasons.most_common())
        )
    if capture:
        print(
            "engine capture      "
            f"median={percentile(capture, 0.5) / 1000:.2f}ms p95={percentile(capture, 0.95) / 1000:.2f}ms"
        )
    if simulation:
        print(
            "builder simulation  "
            f"median={percentile(simulation, 0.5) / 1000:.1f}ms p95={percentile(simulation, 0.95) / 1000:.1f}ms"
            "   <- what stage 4 removes"
        )

    if diverging:
        totals = Counter()
        for record in diverging:
            for field in KEY_FIELDS:
                totals[field] += record.get(field) or 0
            for field in ("lowest_block_mismatched", "parent_mismatched"):
                totals[field] += 1 if record.get(field) else 0
        print("\ndivergence by category")
        for field, count in totals.most_common():
            if count:
                print(f"  {field:<28} {count}")
        print("\nfirst diverging blocks")
        for record in diverging[: args.show_samples]:
            print(f"  block {record['block_number']} {record['block_hash']}")
            for sample in record.get("samples") or []:
                print(f"    {sample}")

    print()
    if diverging:
        print(f"GATE FAILED: {len(diverging)} of {len(hits)} hits diverged; divergence must be zero")
        return 1
    if len(hits) < args.required:
        print(f"GATE OPEN: 0 divergences, but {len(hits)} hits < {args.required} required")
        return 2
    if not compared:
        print(f"GATE OPEN: {len(hits)} hits, zero divergence, but this run recorded no denominator")
        print("           re-run on schema 2 to record compared keys and BLOCKHASH coverage")
        return 2
    if uncovered:
        # One covered hit is not evidence that the rest were covered. A mixed-schema file, or a
        # bug that skipped the coverage path on some blocks, would otherwise pass on the strength
        # of whichever records happened to carry the field.
        print(f"GATE OPEN: {len(uncovered)} of {len(hits)} hits carry no coverage denominator")
        return 2
    if sum(compared) == 0:
        print(f"GATE OPEN: {len(hits)} hits compared zero keys; the zero divergence is vacuous")
        return 2
    empty_categories = [name for name, total in per_category.items() if total == 0]
    if empty_categories:
        print(
            f"GATE OPEN: no {', '.join(empty_categories)} keys were compared at all; "
            "those categories are unproven, not proven"
        )
        return 2
    if not blockhash_blocks:
        print(f"GATE OPEN: {len(hits)} hits, zero divergence, but no block read BLOCKHASH")
        print("           the range check passed vacuously; it is unproven, not proven")
        return 2
    print(
        f"GATE PASSED: {len(hits)} hits, {sum(compared):,} keys compared, zero divergence, "
        f"BLOCKHASH exercised on {len(blockhash_blocks)} blocks"
    )
    if misses:
        share = len(misses) / len(records)
        print(
            f"note: {len(misses)} misses ({share:.1%}) fell back to re-execution. That is the "
            "design working, not failing; it only means stage 4 collects the win on "
            f"{1 - share:.1%} of blocks."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
