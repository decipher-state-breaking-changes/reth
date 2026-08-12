#!/usr/bin/env python3
"""Reads an alternating replay A/B and reports what it can and cannot resolve.

The pairing is real: every block the control arm replayed is the same block, byte for byte, that
the candidate arm replayed, so the per-block difference carries no workload variance at all. That
is what a live paired run cannot offer, and it is the whole reason the corpus exists.

**Pairing does not remove host noise, and on a busy host the host noise is larger.** The
interference is per *process*, not per block: when a round is slow, every block in it is slow
together, so averaging blocks within a round does not average the noise away. Read the output in
this order, and stop at the first line that disqualifies the rest:

  1. `per-block spread` -- the same block, same binary, across rounds. This is the noise floor.
     Anything below it is not a result, no matter how the arms compare.
  2. `min statistic` -- per block, the fastest round in each arm. Interference only ever *adds*
     time, so the minimum is the least contaminated estimate available. Prefer it.
  3. `median statistic` -- the same comparison over per-block medians. It inherits the round-level
     noise through whichever arm happened to catch more slow rounds, which is exactly the failure
     mode a five-round run is most exposed to. Reported so the two can be compared, not trusted
     over the minimum.

Usage: analyze_replay_ab.py <ab.jsonl>
"""
import json
import statistics
import sys
from collections import defaultdict

# Arms are named, never inferred from sort order: reporting "A minus B" when the reader asked for
# "candidate minus control" inverts the sign of every conclusion drawn from it.
CONTROL = "control"
CANDIDATE = "candidate"
FIELDS = ("admission_us", "transition_us")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    runs = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
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
        arms[run["label"].split("-r")[0]].append(run)

    first = runs[0]
    print(f"corpus: {first['commits']} commits ({first['witnessed']} witnessed, "
          f"{first['reconstructed']} reconstructed, {first['absent']} absent)")
    if not first.get("admission_is_load_bearing"):
        print("  WARNING: no witnessed payload in this corpus. The admission checks ran and "
              "proved nothing about the rules.")

    print("\nround totals (admission + transition):")
    for name in sorted(arms):
        totals = [(r["admission_us"] + r["transition_us"]) / 1000 for r in arms[name]]
        print(f"  {name:10s} {len(totals)} rounds: "
              + " ".join(f"{t:8.1f}" for t in sorted(totals))
              + f"   median {statistics.median(totals):8.1f} ms")

    if CONTROL not in arms or CANDIDATE not in arms:
        print(f"\nneed both {CONTROL!r} and {CANDIDATE!r} arms; found {sorted(arms)}")
        return 1

    control = per_block(arms[CONTROL])
    candidate = per_block(arms[CANDIDATE])
    blocks = sorted(set(control) & set(candidate))
    if not blocks:
        print("\nno block replayed by both arms")
        return 1

    print(f"\nper-block spread, same block and same binary across {len(arms[CONTROL])} rounds:")
    for field in FIELDS:
        spreads = [
            (max(control[b][field]) - min(control[b][field])) / min(control[b][field]) * 100
            for b in blocks if min(control[b][field]) > 0
        ]
        print(f"  {field:14s} median {statistics.median(spreads):6.2f}%  "
              f"p90 {quantile(spreads, 0.9):6.2f}%   <-- nothing below this is a result")

    for stat, reduce in (("min", min), ("median", statistics.median)):
        print(f"\n{stat} statistic, paired over {len(blocks)} blocks, "
              f"{CANDIDATE} minus {CONTROL}:")
        for field in FIELDS:
            base = [reduce(control[b][field]) for b in blocks]
            deltas = [
                (reduce(candidate[b][field]) - c) / c * 100
                for b, c in zip(blocks, base) if c > 0
            ]
            print(f"  {field:14s} median {statistics.median(deltas):+6.2f}%  "
                  f"(p10 {quantile(deltas, 0.1):+6.2f}%, p90 {quantile(deltas, 0.9):+6.2f}%)"
                  f"   base {statistics.median(base) / 1000:8.2f} ms")
    return 0


def per_block(group):
    """Every round's cost for every block, keyed by block number then field."""
    gathered = defaultdict(lambda: defaultdict(list))
    for run in group:
        for block in run["blocks"]:
            for field in FIELDS:
                gathered[block["number"]][field].append(block[field])
    return gathered


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


if __name__ == "__main__":
    sys.exit(main())
