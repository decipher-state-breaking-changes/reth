#!/usr/bin/env python3
"""Arm-versus-arm analysis of a `ps-policy-frontier` run, and the network break-even it implies.

The frontier rotates every arm over the **same block inside one process**, so a policy comparison
here is paired within the block rather than across runs. That is a stronger design than the replay
campaign's and this tool exists because nothing was using it: the accepted Weak control was read
off summary totals, which is a ratio of sums over an unpaired population.

Three questions, in one tool because the third needs the first two and would otherwise be computed
by hand from two files that nobody re-derives:

  1. *Arm versus arm on latency*: paired per block on `sidecar_decode_and_commit_us`, with
     `sidecar_decode_us` reported separately because it is the part that scales with witness size
     and therefore the part a bandwidth argument is allowed to touch.
  2. *Arm versus arm on bytes*: the same pairing on `sidecar_bytes`, and on `sidecar_zstd_bytes`
     when a size pass recorded them.
  3. *Break-even bandwidth*: per block, `8 * delta_bytes / delta_seconds` — the link speed at which
     the bytes an arm saves exactly pay for the CPU time it costs. This replaces judging a
     network arm by a latency ratio, which cannot answer the question: what decides a crossover is
     the absolute trade on one block, not how the two latencies compare.

Statistics are `analyze_run_drift`'s, imported rather than reimplemented, so the two tools cannot
drift apart in method: ratio of medians (never a mean of ratios), moving-block bootstrap with the
block length shrunk on short samples, one seeded RNG.

Refusals, each one a failure this tool is here to catch rather than average over:

  * A `rotation_slot` distribution that is not balanced across arms. Position in the rotation is a
    warm-state confound and the rotation exists to cancel it; an unbalanced run has not cancelled
    it and its arm comparison is not a policy comparison.
  * Arms that disagree about a block's hash.
  * A latency question asked of a run whose summary says it compressed sidecars, because that run
     paid an asymmetric zstd tax between arms after each timer closed.
  * A byte question asked across two runs whose measured block sets differ.

Usage:

    analyze_frontier_arms.py RUN [--baseline weak] [--candidate 90/60]
    analyze_frontier_arms.py --timing RUN_A --sizes RUN_B      # two-pass mode
    analyze_frontier_arms.py RUN_A --vs RUN_B --arm 90/60      # A/A floor, one arm, two runs
    analyze_frontier_arms.py --self-check

A RUN is either a `frontier.jsonl` or the directory holding it. In two-pass mode latencies come
from `--timing` and compressed sizes from `--sizes`, which is the intended shape: the size pass is
a separate invocation over the same corpus at the same commit, and this tool checks that claim by
requiring the two passes to agree block for block on `sidecar_digest`.

`--vs` is the A/A floor: one arm against itself in two repetitions of the same run, which bounds
what a cross-run effect on this host may claim. It reuses the arm machinery by presenting the two
repetitions as two arms, so the floor and the effect are computed by the same estimator — a floor
measured a different way from the effect it is supposed to bound is not a floor.
"""

import argparse
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_run_drift import (  # noqa: E402
    RESAMPLES,
    SEED,
    effective_block_len,
    interval,
    moving_block_resample,
)

PRIMARY = "sidecar_decode_and_commit_us"
SECONDARY = "sidecar_decode_us"


def run_paths(target):
    """Resolves a run argument to (jsonl, summary-or-None)."""
    if os.path.isdir(target):
        jsonl = os.path.join(target, "frontier.jsonl")
        summary = os.path.join(target, "frontier-summary.json")
    else:
        jsonl = target
        summary = os.path.join(os.path.dirname(target), "frontier-summary.json")
    if not os.path.exists(jsonl):
        raise SystemExit(f"no frontier.jsonl at {target}")
    return jsonl, (summary if os.path.exists(summary) else None)


def load_run(target):
    """Reads one run into {arm: {block_number: policy_record}} plus block and summary metadata."""
    jsonl, summary_path = run_paths(target)
    arms = {}
    blocks = {}
    with open(jsonl, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Warm-up blocks were replayed so the caches would be warm and are not the population;
            # counting them would report a policy measured before its window was populated.
            if not record.get("measured"):
                continue
            number = record["block_number"]
            blocks[number] = {
                "block_hash": record.get("block_hash"),
                "block_admission_us": record.get("block_admission_us"),
            }
            for policy in record.get("policies", []):
                arms.setdefault(policy["policy"], {})[number] = policy
    summary = None
    if summary_path:
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
    return {"arms": arms, "blocks": blocks, "summary": summary, "path": jsonl}


def as_two_arms(left, right, arm, labels=("rep-a", "rep-b")):
    """Presents one arm from two runs as two arms of one run, for an A/A comparison.

    Blocks are the same corpus, so the pairing is the same pairing. Rotation slots are carried
    through unchanged and therefore agree by construction, which is correct: two repetitions of the
    same run visit the same slots, and there is no position confound between them to cancel.
    """
    for run, label in ((left, labels[0]), (right, labels[1])):
        if arm not in run["arms"]:
            raise SystemExit(f"arm {arm!r} is not in {run['path']}; it holds {sorted(run['arms'])}")
    return {
        "arms": {labels[0]: left["arms"][arm], labels[1]: right["arms"][arm]},
        "blocks": left["blocks"],
        "summary": left["summary"],
        "path": f"{left['path']} vs {right['path']}",
    }


def rotation_balance(arm_records):
    """Slot occupancy per arm, and whether the rotation actually cancelled position.

    Balanced means every arm visited every slot about equally. The tolerance is deliberately loose
    — the rotation is deterministic, so a real imbalance is structural and large, not a rounding
    artefact — but "about equally" still has to be checked rather than assumed, because a run that
    lost blocks to a filter is exactly the case where it silently stops being true.
    """
    slots = {}
    for arm, records in arm_records.items():
        counts = {}
        for policy in records.values():
            counts[policy["rotation_slot"]] = counts.get(policy["rotation_slot"], 0) + 1
        slots[arm] = counts
    arm_count = len(arm_records)
    balanced = True
    for counts in slots.values():
        total = sum(counts.values())
        if not counts:
            balanced = False
            continue
        expected = total / arm_count
        if len(counts) < arm_count or any(
            abs(value - expected) > max(2.0, 0.05 * expected) for value in counts.values()
        ):
            balanced = False
    return {"slots": slots, "balanced": balanced}


def paired_ratio(pairs, rng):
    """Median of candidate/baseline with a moving-block interval, in block order."""
    ratios = [b / a for a, b in pairs if a]
    if not ratios:
        return None
    block_len = effective_block_len(len(ratios))
    replicates = [
        statistics.median(moving_block_resample(ratios, rng, block_len))
        for _ in range(RESAMPLES)
    ]
    return {
        "n": len(ratios),
        "baseline_p50": statistics.median(pair[0] for pair in pairs),
        "candidate_p50": statistics.median(pair[1] for pair in pairs),
        "median_ratio": statistics.median(ratios),
        "interval": interval(replicates),
        "block_len": block_len,
    }


def break_even(shared, base, cand, size_base, size_cand, rng):
    """Per-block break-even bandwidth in Mbit/s, with a moving-block interval.

    The trade is: the candidate arm ships `delta_bytes` fewer bytes and spends `delta_us` more
    microseconds doing it. Below the break-even link speed the bytes dominate and the candidate
    wins; above it the CPU does and the baseline wins. Blocks where both deltas point the same way
    have no crossover at all — one arm is simply better — and they are counted, not folded into a
    median that would then describe a mixture of two different situations.
    """
    values = []
    candidate_dominates = 0
    baseline_dominates = 0
    for number in shared:
        delta_bytes = size_base[number] - size_cand[number]
        delta_us = cand[number] - base[number]
        if delta_bytes > 0 and delta_us > 0:
            # Mbit/s: bits saved divided by seconds spent.
            values.append((delta_bytes * 8.0) / (delta_us * 1e-6) / 1e6)
        elif delta_bytes > 0 and delta_us <= 0:
            candidate_dominates += 1
        elif delta_bytes <= 0 and delta_us > 0:
            baseline_dominates += 1
    if not values:
        return {
            "n": 0,
            "candidate_dominates_blocks": candidate_dominates,
            "baseline_dominates_blocks": baseline_dominates,
        }
    block_len = effective_block_len(len(values))
    replicates = [
        statistics.median(moving_block_resample(values, rng, block_len))
        for _ in range(RESAMPLES)
    ]
    return {
        "n": len(values),
        "median_mbit_s": statistics.median(values),
        "interval_mbit_s": interval(replicates),
        "block_len": block_len,
        "candidate_dominates_blocks": candidate_dominates,
        "baseline_dominates_blocks": baseline_dominates,
    }


def analyze(timing, sizes, baseline, candidate):
    rng = random.Random(SEED)
    report = {"timing_run": timing["path"], "baseline": baseline, "candidate": candidate}

    for name in (baseline, candidate):
        if name not in timing["arms"]:
            raise SystemExit(
                f"arm {name!r} is not in {timing['path']}; it holds "
                f"{sorted(timing['arms'])}"
            )

    summary = timing["summary"] or {}
    report["allocator"] = summary.get("allocator")
    report["generator_build_commit"] = summary.get("generator_build_commit")
    report["measured_block_set_digest"] = summary.get("measured_block_set_digest")
    compressed_timing = bool(summary.get("compressed_sidecars"))
    report["timing_run_compressed_sidecars"] = compressed_timing

    balance = rotation_balance(timing["arms"])
    report["rotation"] = balance

    base_records = timing["arms"][baseline]
    cand_records = timing["arms"][candidate]
    shared = sorted(set(base_records) & set(cand_records))
    refused = [
        number
        for number in shared
        if base_records[number]["witness_commitment"] is None
        or timing["blocks"][number]["block_hash"] is None
    ]
    report["paired_blocks"] = len(shared)
    report["baseline_only"] = len(set(base_records) - set(cand_records))
    report["candidate_only"] = len(set(cand_records) - set(base_records))
    report["refused"] = len(refused)

    if compressed_timing:
        report["latency"] = None
        report["latency_refused_because"] = (
            "the timing run's summary says it compressed sidecars, so its per-arm timings carry an "
            "asymmetric zstd tax and are void"
        )
    elif balance["balanced"]:
        report["latency"] = {
            PRIMARY: paired_ratio(
                [(base_records[n][PRIMARY], cand_records[n][PRIMARY]) for n in shared], rng
            ),
            SECONDARY: paired_ratio(
                [(base_records[n][SECONDARY], cand_records[n][SECONDARY]) for n in shared], rng
            ),
        }
    else:
        report["latency"] = None
        report["latency_refused_because"] = (
            "rotation slots are not balanced across arms, so position in the rotation is not "
            "cancelled and this is not a policy comparison"
        )

    report["bytes"] = {
        "sidecar_bytes": paired_ratio(
            [(base_records[n]["sidecar_bytes"], cand_records[n]["sidecar_bytes"]) for n in shared],
            rng,
        )
    }

    size_run = sizes or timing
    report["size_run"] = size_run["path"]
    size_base = size_run["arms"].get(baseline, {})
    size_cand = size_run["arms"].get(candidate, {})
    size_shared = sorted(set(size_base) & set(size_cand) & set(shared))

    # The premise that licenses reading sizes off one pass and timings off another: the sidecars
    # are the same objects. Checked on the *semantic* digest deliberately. The serialized bytes are
    # not bit-identical across runs — a sidecar carries `stats.computation_time_ms`, which is how
    # long that host took to build it — so a raw-byte comparison would report every block as
    # differing. bincode writes that field at fixed width, so the length is stable and the
    # compressed size moves by at most a byte or two; the semantic digest is what actually says the
    # two passes built the same sidecar.
    mismatched = [
        n
        for n in size_shared
        if size_base[n]["sidecar_digest"] != base_records[n]["sidecar_digest"]
        or size_cand[n]["sidecar_digest"] != cand_records[n]["sidecar_digest"]
    ]
    report["size_pass_digest_mismatches"] = len(mismatched)

    have_zstd = [
        n
        for n in size_shared
        if size_base[n].get("sidecar_zstd_bytes") and size_cand[n].get("sidecar_zstd_bytes")
    ]
    if mismatched:
        report["compressed"] = None
        report["compressed_refused_because"] = (
            f"{len(mismatched)} blocks have a different sidecar in the size pass than in the "
            "timing pass; the two passes are not describing the same objects"
        )
    elif have_zstd:
        report["compressed"] = {
            "sidecar_zstd_bytes": paired_ratio(
                [
                    (
                        size_base[n]["sidecar_zstd_bytes"],
                        size_cand[n]["sidecar_zstd_bytes"],
                    )
                    for n in have_zstd
                ],
                rng,
            ),
            "zstd_level": (size_run["summary"] or {}).get("sidecar_zstd_level"),
            "baseline_compression": statistics.median(
                size_base[n]["sidecar_bytes"] / size_base[n]["sidecar_zstd_bytes"]
                for n in have_zstd
            ),
            "candidate_compression": statistics.median(
                size_cand[n]["sidecar_bytes"] / size_cand[n]["sidecar_zstd_bytes"]
                for n in have_zstd
            ),
        }
    else:
        report["compressed"] = None
        report["compressed_refused_because"] = "no run recorded sidecar_zstd_bytes"

    if report["latency"] and report["compressed"] and not mismatched:
        report["break_even"] = break_even(
            have_zstd,
            {n: base_records[n][PRIMARY] for n in have_zstd},
            {n: cand_records[n][PRIMARY] for n in have_zstd},
            {n: size_base[n]["sidecar_zstd_bytes"] for n in have_zstd},
            {n: size_cand[n]["sidecar_zstd_bytes"] for n in have_zstd},
            rng,
        )
    else:
        report["break_even"] = None
        report["break_even_refused_because"] = (
            "needs both a valid latency pairing and compressed sizes for the same blocks"
        )
    return report


def fmt_ratio(row):
    if not row:
        return "  (none)"
    low, high = row["interval"]
    resolves = "" if low <= 1.0 <= high else "  — resolves an effect"
    return (
        f"  n={row['n']}  {row['baseline_p50']:,.0f} -> {row['candidate_p50']:,.0f}"
        f"   ratio **{row['median_ratio']:.4f}x**  95% [{low:.4f}, {high:.4f}]{resolves}"
    )


def print_human(report):
    print(f"# Frontier arms — {report['candidate']} against {report['baseline']}\n")
    print(f"Timing run `{report['timing_run']}`")
    if report["size_run"] != report["timing_run"]:
        print(f"Size run   `{report['size_run']}`")
    print(
        f"allocator `{report.get('allocator')}`  commit `{report.get('generator_build_commit')}`"
        f"  corpus `{report.get('measured_block_set_digest')}`"
    )
    print(
        f"\npaired on {report['paired_blocks']} blocks "
        f"(+{report['baseline_only']} baseline-only, +{report['candidate_only']} candidate-only)"
    )
    slots = report["rotation"]["slots"]
    verdict = "balanced" if report["rotation"]["balanced"] else "**NOT BALANCED**"
    print(f"rotation slots {verdict}: " + "; ".join(f"{a} {dict(sorted(c.items()))}" for a, c in slots.items()))

    print("\n## Latency")
    if report["latency"]:
        print(f"{PRIMARY}\n{fmt_ratio(report['latency'][PRIMARY])}")
        print(f"{SECONDARY}\n{fmt_ratio(report['latency'][SECONDARY])}")
    else:
        print(f"  refused: {report['latency_refused_because']}")

    print("\n## Bytes")
    print(f"sidecar_bytes (uncompressed)\n{fmt_ratio(report['bytes']['sidecar_bytes'])}")
    if report["compressed"]:
        comp = report["compressed"]
        print(f"sidecar_zstd_bytes (level {comp['zstd_level']})\n{fmt_ratio(comp['sidecar_zstd_bytes'])}")
        print(
            f"  compression achieved: {report['baseline']} {comp['baseline_compression']:.3f}x, "
            f"{report['candidate']} {comp['candidate_compression']:.3f}x"
        )
    else:
        print(f"  compressed: refused — {report['compressed_refused_because']}")

    print("\n## Break-even bandwidth")
    be = report["break_even"]
    if be and be["n"]:
        low, high = be["interval_mbit_s"]
        print(
            f"  n={be['n']} blocks with a crossover  median **{be['median_mbit_s']:,.0f} Mbit/s**"
            f"  95% [{low:,.0f}, {high:,.0f}]"
        )
        print(
            f"  {be['candidate_dominates_blocks']} blocks where {report['candidate']} wins on both"
            f"; {be['baseline_dominates_blocks']} where {report['baseline']} does"
        )
        print(
            "\n  Below this link speed the smaller arm wins; above it the faster one does. Compare"
            "\n  against the deployment range fixed before the run, not against this line."
        )
    else:
        print(f"  refused: {report.get('break_even_refused_because') or 'no crossover blocks'}")


def self_check():
    """Synthetic run with known answers, so the estimators are checked without a corpus."""
    rng = random.Random(1)
    blocks = {}
    arms = {"weak": {}, "90/60": {}}
    for i in range(400):
        number = 1000 + i
        blocks[number] = {"block_hash": f"0x{number:064x}", "block_admission_us": 2000}
        jitter = 1.0 + rng.uniform(-0.01, 0.01)
        # Weak: cheap and large. 90/60: 1.25x the time, a fifth of the bytes.
        arms["weak"][number] = {
            "policy": "weak",
            "rotation_slot": i % 2,
            "sidecar_bytes": 7_500_000,
            "sidecar_zstd_bytes": 5_000_000,
            "sidecar_digest": f"0xw{number:063x}",
            "witness_commitment": "0x00",
            PRIMARY: int(100_000 * jitter),
            SECONDARY: 2_000,
        }
        arms["90/60"][number] = {
            "policy": "90/60",
            "rotation_slot": (i + 1) % 2,
            "sidecar_bytes": 1_500_000,
            "sidecar_zstd_bytes": 1_000_000,
            "sidecar_digest": f"0xc{number:063x}",
            "witness_commitment": "0x00",
            PRIMARY: int(125_000 * jitter),
            SECONDARY: 500,
        }
    run = {"arms": arms, "blocks": blocks, "summary": {"sidecar_zstd_level": 3}, "path": "<synthetic>"}
    report = analyze(run, None, "weak", "90/60")

    failures = []
    if not report["rotation"]["balanced"]:
        failures.append("rotation should read as balanced: each arm takes each slot 200 times")
    lat = report["latency"][PRIMARY]["median_ratio"]
    if not math.isclose(lat, 1.25, rel_tol=0.02):
        failures.append(f"latency ratio {lat:.4f} should be ~1.25")
    byt = report["bytes"]["sidecar_bytes"]["median_ratio"]
    if not math.isclose(byt, 0.2, rel_tol=1e-9):
        failures.append(f"byte ratio {byt:.4f} should be exactly 0.2")
    # 4 MB saved against 25 ms spent = 32e6 bits / 0.025 s = 1280 Mbit/s.
    be = report["break_even"]["median_mbit_s"]
    if not math.isclose(be, 1280.0, rel_tol=0.03):
        failures.append(f"break-even {be:,.0f} Mbit/s should be ~1280")

    # A run whose summary admits it compressed must refuse to answer on time.
    voided = dict(run, summary={"compressed_sidecars": True, "sidecar_zstd_level": 3})
    if analyze(voided, None, "weak", "90/60")["latency"] is not None:
        failures.append("a compressed run must refuse the latency question")

    # An unbalanced rotation must refuse too.
    skewed = {arm: dict(records) for arm, records in arms.items()}
    for number in list(skewed["weak"])[:300]:
        skewed["weak"][number] = dict(skewed["weak"][number], rotation_slot=0)
    if analyze(dict(run, arms=skewed), None, "weak", "90/60")["latency"] is not None:
        failures.append("an unbalanced rotation must refuse the latency question")

    for line in failures:
        print(f"FAIL: {line}")
    if not failures:
        print("self-check ok: latency 1.25x, bytes 0.2x, break-even ~1280 Mbit/s, both refusals fire")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", help="frontier.jsonl or the directory holding it")
    parser.add_argument("--timing", help="run to read latencies from")
    parser.add_argument("--sizes", help="run to read compressed sizes from")
    parser.add_argument("--vs", help="second run, for a same-arm A/A floor")
    parser.add_argument("--arm", default="90/60", help="the arm to compare across runs with --vs")
    parser.add_argument("--baseline", default="weak")
    parser.add_argument("--candidate", default="90/60")
    parser.add_argument("--json", help="write the full report here")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        return self_check()

    target = args.timing or args.run
    if not target:
        parser.error("give a run, or --timing with --sizes")
    timing = load_run(target)
    sizes = load_run(args.sizes) if args.sizes else None

    if sizes and timing["summary"] and sizes["summary"]:
        left = timing["summary"].get("measured_block_set_digest")
        right = sizes["summary"].get("measured_block_set_digest")
        if left and right and left != right:
            raise SystemExit(
                "the timing pass and the size pass measured different block sets "
                f"({left} vs {right}); they are not comparable"
            )

    if args.vs:
        other = load_run(args.vs)
        left = (timing["summary"] or {}).get("measured_block_set_digest")
        right = (other["summary"] or {}).get("measured_block_set_digest")
        if left and right and left != right:
            raise SystemExit(
                f"the two repetitions measured different block sets ({left} vs {right})"
            )
        report = analyze(as_two_arms(timing, other, args.arm), None, "rep-a", "rep-b")
        report["a_a_floor_for_arm"] = args.arm
    else:
        report = analyze(timing, sizes, args.baseline, args.candidate)
    print_human(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
