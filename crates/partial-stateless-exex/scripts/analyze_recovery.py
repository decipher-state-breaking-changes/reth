#!/usr/bin/env python3
"""Link undo recovery to the first reapplied verdict and the announced winning head.

Durations use one follower process's monotonic clock. They include time waiting for later
inputs, so bulk replay and live-paced runs must be reported separately. Missing milestones
remain null; a restart, second lifecycle event or fault ends the observation window.
"""
import argparse
import json
from pathlib import Path


def summarize(rows):
    results = []
    for index, event in enumerate(rows):
        if event.get("event") not in ("reorg_applied", "revert_applied"):
            continue
        if not all(key in event for key in ("run_id", "process_elapsed_us", "recovery_us", "recovery_started_elapsed_us")):
            raise ValueError("recovery event lacks monotonic measurement fields")
        result = {"run_id": event["run_id"], "common_ancestor": event["common_ancestor"],
                  "depth": len(event["abandoned"]), "recovery_us": event["recovery_us"],
                  "first_reapplied_block": None, "first_reapplied_step_us": None,
                  "through_first_verdict_us": None, "through_winning_head_us": None}
        for row in rows[index + 1:]:
            if row.get("run_id") != event["run_id"] or row.get("kind") == "lifecycle":
                break
            if row.get("kind") == "state" and row.get("state") not in ("streaming",):
                break
            if row.get("kind") != "verdict" or row.get("verdict") != "accepted":
                continue
            elapsed = row["process_elapsed_us"] - event["recovery_started_elapsed_us"]
            if elapsed < 0:
                raise ValueError("non-monotonic recovery records")
            if result["first_reapplied_block"] is None:
                result.update(first_reapplied_block=row["block"],
                              first_reapplied_step_us=row["standalone_validation_us"],
                              through_first_verdict_us=elapsed)
            if (row["block"] == event.get("winning_tip") and
                    row["block_hash"] == event.get("winning_tip_hash")):
                result["through_winning_head_us"] = elapsed
                break
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("follow", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.follow.read_text().splitlines() if line]
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
