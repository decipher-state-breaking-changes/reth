#!/usr/bin/env python3
"""Reads an alternating replay A/B and reports the paired per-block difference.

The pairing is the point. Every block the control arm replayed is the same block, byte for byte,
that the candidate arm replayed, so the per-block difference has no workload variance in it at all
-- which is exactly what a live paired run cannot offer. What is left is host noise, and the
round-to-round spread within one arm is the honest estimate of it.

Read the two numbers together and in this order:

  1. `within-arm spread` -- how much one arm moved across rounds with nothing changed. Any
     difference between arms smaller than this is not a result.
  2. `paired delta` -- the median per-block candidate-minus-control, over blocks present in both.

Usage: analyze_replay_ab.py <ab.jsonl>
"""
import json
import statistics
import sys
from collections import defaultdict


def load(path):
    runs = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def arm(label):
    return label.split("-r")[0]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    runs = load(sys.argv[1])
    if not runs:
        print("no records")
        return 1

    disagreed = [r["label"] for r in runs if not r.get("agreed", True)]
    if disagreed:
        print(f"REFUSING TO REPORT TIMINGS: {len(disagreed)} run(s) disagreed with the recording")
        for label in disagreed:
            print(f"  {label}")
        return 1

    arms = defaultdict(list)
    for run in runs:
        arms[arm(run["label"])].append(run)

    print(f"corpus: {runs[0]['commits']} commits "
          f"({runs[0]['witnessed']} witnessed, {runs[0]['reconstructed']} reconstructed, "
          f"{runs[0]['absent']} absent)")
    if not runs[0].get("admission_is_load_bearing"):
        print("  WARNING: no witnessed payload in this corpus. The admission checks ran and "
              "proved nothing about the rules.")
    print()

    for name, group in sorted(arms.items()):
        totals = [r["transition_us"] + r["admission_us"] for r in group]
        spread = (max(totals) - min(totals)) / statistics.mean(totals) * 100 if totals else 0
        print(f"{name}: {len(group)} rounds, "
              f"total {statistics.mean(totals) / 1000:.1f} ms mean, "
              f"within-arm spread {spread:.2f}%")

    if len(arms) != 2:
        print("\nonly one arm present; nothing to compare")
        return 0

    control_name, candidate_name = sorted(arms)
    control = per_block(arms[control_name])
    candidate = per_block(arms[candidate_name])
    shared = sorted(set(control) & set(candidate))
    if not shared:
        print("\nno block replayed by both arms")
        return 1

    print(f"\npaired over {len(shared)} blocks, {candidate_name} minus {control_name}:")
    for field in ("admission_us", "transition_us"):
        deltas = [candidate[b][field] - control[b][field] for b in shared]
        base = statistics.median([control[b][field] for b in shared])
        median = statistics.median(deltas)
        pct = median / base * 100 if base else 0
        print(f"  {field:14s} median {median:+.0f} us on a {base:.0f} us base ({pct:+.2f}%)")
    return 0


def per_block(group):
    """Median per-block cost across an arm's rounds, keyed by block number."""
    gathered = defaultdict(lambda: defaultdict(list))
    for run in group:
        for block in run["blocks"]:
            gathered[block["number"]]["admission_us"].append(block["admission_us"])
            gathered[block["number"]]["transition_us"].append(block["transition_us"])
    return {
        number: {field: statistics.median(values) for field, values in fields.items()}
        for number, fields in gathered.items()
    }


if __name__ == "__main__":
    sys.exit(main())
