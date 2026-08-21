#!/usr/bin/env python3
"""Compare two `ps-replay --json` batch summaries for behavioral invariance.

Usage: check_replay_invariance.py <current.json> <baseline.json>

Both inputs are the JSONL files `ps-replay <spool> --json <path>` writes; the last
line is the run summary. The pass bar: every deterministic aggregate field and the
ordered per-block (sequence, number, verdict) chain identical. Wall-clock fields
(`*_us`, `elapsed_ms`), the free-form `label`, and sample lists are excluded —
replays of one corpus differ there by machine and by run without meaning anything.

The intended baseline is a build old enough to predate the two recovery-ledger
counters (`late_skim_mismatches`, `recovery_checkpoints_pending_at_end`) and W4's
`transition_mutations_checked`, so those are exempt from the field comparison — but
their *presence* on the baseline side fails the check: a baseline that reports them
was built from too new a commit, and that failure mode is quieter than a field
mismatch and worth its own message. The mutation counter carries one rule the W1
pair does not: mutation coverage is opt-in and must never run inside an invariance
replay, so a nonzero current-side value is its own failure rather than an exemption.
"""
import json
import sys

CURRENT_ONLY_COUNTERS = {"late_skim_mismatches", "recovery_checkpoints_pending_at_end"}
CURRENT_ONLY_ZERO_COUNTERS = {"transition_mutations_checked"}


def load_summary(path):
    with open(path) as handle:
        record = [json.loads(line) for line in handle][-1]
    aggregates = {
        key: value
        for key, value in record.items()
        if key not in ("blocks", "label")
        and not key.endswith("_us")
        and key != "elapsed_ms"
        and not isinstance(value, list)
    }
    sequence = [(b["sequence"], b["number"], b["verdict"]) for b in record["blocks"]]
    return aggregates, sequence


def compare(current_path, baseline_path):
    (current, current_seq) = load_summary(current_path)
    (baseline, baseline_seq) = load_summary(baseline_path)
    problems = []
    exempt = CURRENT_ONLY_COUNTERS | CURRENT_ONLY_ZERO_COUNTERS
    for key in sorted((set(current) | set(baseline)) - exempt):
        if current.get(key) != baseline.get(key):
            problems.append(
                f"{key}: current={current.get(key)!r} baseline={baseline.get(key)!r}"
            )
    for key in sorted(exempt & set(baseline)):
        problems.append(
            f"{key}: present on the baseline side, so the baseline binary is too new"
        )
    for key in sorted(exempt - set(current)):
        problems.append(
            f"{key}: absent on the current side, so the current binary/schema is too old"
        )
    for key in sorted(CURRENT_ONLY_ZERO_COUNTERS):
        if key in current and current[key] != 0:
            problems.append(
                f"{key}: {current[key]!r} on the current side — opt-in coverage "
                "ran inside an invariance replay"
            )
    if current_seq != baseline_seq:
        problems.append(
            f"block sequences differ ({len(current_seq)} vs {len(baseline_seq)} blocks)"
        )
    compared_fields = len((set(current) | set(baseline)) - exempt)
    return problems, compared_fields, len(current_seq)


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    problems, field_count, block_count = compare(argv[1], argv[2])
    if problems:
        print("INVARIANCE FAILED")
        for problem in problems:
            print("  " + problem)
        return 1
    print(
        f"INVARIANCE OK ({field_count} shared aggregate fields + "
        f"{len(CURRENT_ONLY_COUNTERS | CURRENT_ONLY_ZERO_COUNTERS)} current-only counters, "
        f"{block_count} blocks)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
