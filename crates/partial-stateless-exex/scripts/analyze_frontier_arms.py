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
  3. *The deployment question*: at a given link speed, which arm delivers a block sooner, and by
     how much? Modelled as `8 * bytes / B + cpu`, so for the cached arm against the baseline

         delta_total(B) = (cached_cpu - weak_cpu) + 8 * (cached_bytes - weak_bytes) / B

     which is negative — cached wins — while `B` is **below** the break-even
     `B* = 8 * (weak_bytes - cached_bytes) / (cached_cpu - weak_cpu)`, and positive above it.
     Cached ships less and computes more, so a slow link favours it and a fast link does not. This
     replaces judging a network arm by a latency ratio, which cannot answer the question at all:
     what decides a crossover is the absolute trade on one block, not how two latencies compare.

     Reported two ways, because they are different estimands and only the first is a deployment
     answer. `endpoints` evaluates `delta_total` at fixed link speeds over **every** paired block,
     which is the statistic a decision is made on. `break_even` is the median of the per-block
     crossover over the blocks that *have* one, which is a description of the corpus: blocks where
     one arm is both smaller and faster have no crossover and are excluded from it, so it is not a
     population statistic and must not be read as one.

Statistics are `analyze_run_drift`'s, imported rather than reimplemented, so the two tools cannot
drift apart in method: moving-block bootstrap with the block length shrunk on short samples, one
seeded RNG.

The paired estimator is the **median of the per-block ratio**, which is the right one for paired
data and is not the same as either alternative it gets confused with. A *mean* of ratios would
weight a cheap block like an expensive one. A *ratio of totals* answers a different question — what
a link carrying the whole corpus sees, rather than what a typical block sees — and both are
defensible, so this tool reports the paired median and the summary keeps the totals, and neither is
quoted without naming which it is.

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
    exact = True
    for counts in slots.values():
        total = sum(counts.values())
        if not counts or len(counts) < arm_count:
            balanced = False
            exact = False
            continue
        expected = total / arm_count
        # A cyclic rotation over a population that divides evenly by the arm count lands on exactly
        # equal occupancy, so that is what is required rather than a tolerance: any deviation is a
        # block the rotation did not visit as designed, and a tolerance would let a real defect
        # through as rounding.
        if total % arm_count == 0:
            if any(value != total // arm_count for value in counts.values()):
                balanced = False
                exact = False
        else:
            exact = False
            if any(abs(value - expected) > 1.0 for value in counts.values()):
                balanced = False
    return {"slots": slots, "balanced": balanced, "exact": exact}


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


DEFAULT_ENDPOINTS_MBIT = (100.0, 1000.0)


def delta_total_ms(cpu_base, cpu_cand, bytes_base, bytes_cand, bandwidth_mbit):
    """Candidate minus baseline block-ready time at one link speed, in milliseconds.

    Negative means the candidate delivers the block sooner. Transfer and processing are added
    because this is a store-and-forward transport: `decode_frame` verifies a digest over the whole
    payload before anything is deserialized, so nothing is processed while it is still arriving. A
    model that overlapped them would describe a protocol this system does not implement.
    """
    transfer_s = 8.0 * (bytes_cand - bytes_base) / (bandwidth_mbit * 1e6)
    cpu_s = (cpu_cand - cpu_base) * 1e-6
    return (transfer_s + cpu_s) * 1e3


def endpoints(shared, base, cand, size_base, size_cand, rng, bandwidths):
    """The deployment statistic: `delta_total` at fixed link speeds, over every paired block.

    Every block counts, including the ones the per-block crossover has to discard — a block where
    one arm is both smaller and faster has no break-even, but it certainly has a delta at 100
    Mbit/s, and dropping it would answer the deployment question on a subset chosen by the answer.

    The win rate rides along because a median near zero and a 50% win rate mean something very
    different from a median near zero and a 95% win rate.
    """
    rows = []
    for bandwidth in bandwidths:
        deltas = [
            delta_total_ms(base[n], cand[n], size_base[n], size_cand[n], bandwidth)
            for n in shared
        ]
        block_len = effective_block_len(len(deltas))
        replicates = [
            statistics.median(moving_block_resample(deltas, rng, block_len))
            for _ in range(RESAMPLES)
        ]
        rows.append({
            "bandwidth_mbit_s": bandwidth,
            "n": len(deltas),
            "median_delta_ms": statistics.median(deltas),
            "interval_ms": interval(replicates),
            "candidate_win_rate": sum(1 for d in deltas if d < 0) / len(deltas),
        })
    return rows


def population_crossover(shared, base, cand, size_base, size_cand, low=1.0, high=1e6):
    """The link speed where the *median* block's delta is zero, by bisection.

    Not the same number as the median of per-block crossovers, and the difference is the whole
    reason both are reported: one asks where the typical block flips, the other where the
    population's median flips. They coincide only if the two deltas are perfectly rank-correlated.
    Returns `None` when the sign does not change over the bracket — the median block never flips
    at any realistic link speed, which is itself the answer.
    """
    def median_delta(bandwidth):
        return statistics.median(
            delta_total_ms(base[n], cand[n], size_base[n], size_cand[n], bandwidth)
            for n in shared
        )

    if median_delta(low) * median_delta(high) > 0:
        return None
    for _ in range(200):
        mid = (low + high) / 2.0
        if median_delta(low) * median_delta(mid) <= 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


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


def analyze(timing, sizes, baseline, candidate, bandwidths=DEFAULT_ENDPOINTS_MBIT):
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
    baseline_only = sorted(set(base_records) - set(cand_records))
    candidate_only = sorted(set(cand_records) - set(base_records))
    incomplete = [
        number
        for number in shared
        if base_records[number]["witness_commitment"] is None
        or cand_records[number]["witness_commitment"] is None
        or timing["blocks"][number]["block_hash"] is None
    ]
    report["paired_blocks"] = len(shared)
    report["baseline_only"] = len(baseline_only)
    report["candidate_only"] = len(candidate_only)
    report["refused"] = len(incomplete)

    # Integrity is fail-closed and refuses the whole report rather than dropping rows. Every arm
    # runs over every measured block by construction, so a block one arm is missing is not a gap to
    # work around — it means the run did not do what the file says it did, and a population quietly
    # shrunk to the blocks that happened to survive is exactly the kind of number that reads as
    # clean and is not.
    integrity = []
    if baseline_only or candidate_only:
        integrity.append(
            f"{len(baseline_only)} blocks only {baseline} saw and {len(candidate_only)} only "
            f"{candidate} saw; every arm rotates over every measured block, so this run is not "
            "internally consistent"
        )
    if incomplete:
        integrity.append(
            f"{len(incomplete)} blocks are missing a block hash or a witness commitment"
        )
    if not shared:
        integrity.append("the two arms share no measured blocks")
    report["integrity_refusals"] = integrity
    if integrity:
        report["latency"] = None
        report["latency_refused_because"] = "; ".join(integrity)
        report["bytes"] = {"sidecar_bytes": None}
        report["bytes_refused_because"] = "; ".join(integrity)
        report["size_run"] = (sizes or timing)["path"]
        report["compressed"] = None
        report["compressed_refused_because"] = "; ".join(integrity)
        report["break_even"] = None
        report["endpoints"] = None
        report["break_even_refused_because"] = "; ".join(integrity)
        return report

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
    # Partial coverage is refused rather than computed over whatever subset carries values. A
    # sidecar's compressibility varies with its own composition, so a subset chosen by which blocks
    # happen to have been compressed is a subset chosen by something correlated with the answer.
    partial_coverage = have_zstd and len(have_zstd) != len(shared)

    # A size pass is only a stand-in for the timing pass if it was the same build doing the same
    # thing to the same corpus. Checked here rather than trusted to the driver, because the driver
    # is untracked and this file is what a later reader has.
    size_summary = size_run["summary"] or {}
    metadata = []
    if sizes is not None:
        for key in ("allocator", "generator_build_commit", "measured_block_set_digest"):
            left, right = summary.get(key), size_summary.get(key)
            if left != right:
                metadata.append(f"{key}: timing {left!r} vs size {right!r}")
        if not size_summary.get("compressed_sidecars"):
            metadata.append("the size run's summary does not say it compressed sidecars")
    if have_zstd and size_summary.get("sidecar_zstd_level") is None:
        metadata.append("compressed sizes are present but no zstd level was recorded")

    if metadata:
        report["compressed"] = None
        report["compressed_refused_because"] = (
            "the size pass and the timing pass do not describe the same measurement — "
            + "; ".join(metadata)
        )
    elif partial_coverage:
        report["compressed"] = None
        report["compressed_refused_because"] = (
            f"only {len(have_zstd)} of {len(shared)} paired blocks carry compressed sizes; "
            "compressibility varies with a sidecar's composition, so a partial subset is not a "
            "sample of the population"
        )
    elif mismatched:
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

    if report["latency"] and report["compressed"]:
        cpu_base = {n: base_records[n][PRIMARY] for n in have_zstd}
        cpu_cand = {n: cand_records[n][PRIMARY] for n in have_zstd}
        zb = {n: size_base[n]["sidecar_zstd_bytes"] for n in have_zstd}
        zc = {n: size_cand[n]["sidecar_zstd_bytes"] for n in have_zstd}
        # The decision statistic first, over every paired block.
        report["endpoints"] = endpoints(have_zstd, cpu_base, cpu_cand, zb, zc, rng, bandwidths)
        report["population_crossover_mbit_s"] = population_crossover(
            have_zstd, cpu_base, cpu_cand, zb, zc
        )
        # The corpus description second, over the subset that has a crossover at all.
        report["break_even"] = break_even(have_zstd, cpu_base, cpu_cand, zb, zc, rng)
    else:
        report["endpoints"] = None
        report["break_even"] = None
        report["population_crossover_mbit_s"] = None
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
    if report.get("integrity_refusals"):
        print("\n**REFUSED**: " + "; ".join(report["integrity_refusals"]))
    slots = report["rotation"]["slots"]
    verdict = "balanced" if report["rotation"]["balanced"] else "**NOT BALANCED**"
    if report["rotation"]["balanced"] and report["rotation"].get("exact"):
        verdict = "balanced exactly"
    print(f"rotation slots {verdict}: " + "; ".join(f"{a} {dict(sorted(c.items()))}" for a, c in slots.items()))

    print("\n## Latency")
    if report["latency"]:
        print(f"{PRIMARY}\n{fmt_ratio(report['latency'][PRIMARY])}")
        print(f"{SECONDARY}\n{fmt_ratio(report['latency'][SECONDARY])}")
    else:
        print(f"  refused: {report['latency_refused_because']}")

    print("\n## Bytes")
    if report["bytes"].get("sidecar_bytes") is None:
        print(f"  refused: {report.get('bytes_refused_because')}")
    else:
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

    print("\n## Block-ready latency at deployment link speeds")
    rows = report.get("endpoints")
    if rows:
        print(f"  negative = {report['candidate']} delivers the block sooner\n")
        print("  | link | median delta | 95% interval | " + f"{report['candidate']} win rate |")
        print("  | ---: | ---: | :---: | ---: |")
        for row in rows:
            low, high = row["interval_ms"]
            print(
                f"  | {row['bandwidth_mbit_s']:,.0f} Mbit/s | {row['median_delta_ms']:+.2f} ms "
                f"| [{low:+.2f}, {high:+.2f}] | {row['candidate_win_rate']:.1%} |"
            )
        crossover = report.get("population_crossover_mbit_s")
        if crossover:
            print(
                f"\n  Median block flips at **{crossover:,.0f} Mbit/s**: {report['candidate']} is "
                f"sooner below it,\n  {report['baseline']} above it — the cached arm ships less and "
                "computes more, so a slow\n  link favours it and a fast link does not."
            )
        else:
            print(
                f"\n  The median block does not flip at any link speed in the bracket: one arm is "
                "sooner throughout."
            )
    else:
        print(f"  refused: {report.get('break_even_refused_because') or 'not computed'}")

    print("\n## Per-block crossover (corpus description, not a deployment answer)")
    be = report["break_even"]
    if be and be["n"]:
        low, high = be["interval_mbit_s"]
        print(
            f"  n={be['n']} of {report['paired_blocks']} blocks have a crossover  "
            f"median **{be['median_mbit_s']:,.0f} Mbit/s**  95% [{low:,.0f}, {high:,.0f}]"
        )
        print(
            f"  {be['candidate_dominates_blocks']} blocks where {report['candidate']} is both "
            f"smaller and faster; {be['baseline_dominates_blocks']} where {report['baseline']} is"
        )
        print(
            "\n  Excludes the blocks with no crossover, so this is not a population statistic and"
            "\n  the decision is not made on it. Use the table above."
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
    crossover = report["population_crossover_mbit_s"]
    if not math.isclose(crossover, 1280.0, rel_tol=0.03):
        failures.append(f"population crossover {crossover:,.0f} Mbit/s should be ~1280")

    # The direction, which is the thing a sign error would silently invert. Break-even is above
    # both endpoints, so the smaller-and-slower arm must deliver sooner at both.
    rows = {row["bandwidth_mbit_s"]: row for row in report["endpoints"]}
    slow, fast = rows[100.0], rows[1000.0]
    if not slow["median_delta_ms"] < 0:
        failures.append("at 100 Mbit/s the cached arm must deliver sooner: it is far below B*")
    if not fast["median_delta_ms"] < 0:
        failures.append("at 1000 Mbit/s the cached arm must still deliver sooner: B* is 1280")
    if not slow["median_delta_ms"] < fast["median_delta_ms"]:
        failures.append("the cached arm's advantage must shrink as the link gets faster")
    if not math.isclose(slow["median_delta_ms"], -295.0, abs_tol=3.0):
        failures.append(f"100 Mbit/s delta {slow['median_delta_ms']:.1f} ms should be ~-295")
    if not math.isclose(fast["median_delta_ms"], -7.0, abs_tol=3.0):
        failures.append(f"1000 Mbit/s delta {fast['median_delta_ms']:.1f} ms should be ~-7")
    # Above the crossover the sign must flip, or the model is not a model of anything.
    above = analyze(run, None, "weak", "90/60", bandwidths=(4000.0,))["endpoints"][0]
    if not above["median_delta_ms"] > 0:
        failures.append("above B* the baseline must deliver sooner")

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
        print(
            "self-check ok: latency 1.25x, bytes 0.2x, B* ~1280 Mbit/s, cached sooner below it and "
            "later above it, both refusals fire"
        )
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
    parser.add_argument(
        "--endpoints",
        default=",".join(str(int(b)) for b in DEFAULT_ENDPOINTS_MBIT),
        help="comma-separated link speeds in Mbit/s to evaluate the deployment question at",
    )
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

    bandwidths = tuple(float(x) for x in args.endpoints.split(",") if x.strip())
    if not bandwidths:
        parser.error("--endpoints needs at least one link speed")

    if args.vs:
        other = load_run(args.vs)
        left = (timing["summary"] or {}).get("measured_block_set_digest")
        right = (other["summary"] or {}).get("measured_block_set_digest")
        if left and right and left != right:
            raise SystemExit(
                f"the two repetitions measured different block sets ({left} vs {right})"
            )
        report = analyze(
            as_two_arms(timing, other, args.arm), None, "rep-a", "rep-b", bandwidths
        )
        report["a_a_floor_for_arm"] = args.arm
    else:
        report = analyze(timing, sizes, args.baseline, args.candidate, bandwidths)
    print_human(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
