#!/usr/bin/env python3
"""Within-run time-order drift and cross-run allocator/implementation A/B for replay JSONL.

Two questions, deliberately in one tool because answering either alone is misleading:

  1. *Within a run*: does the per-block cost drift over the run's own block order, and which
     phases carry the drift? A validator whose cost climbs 15% across a run has a defect, but it
     also sets a floor: a cross-run A/B on that host cannot resolve an effect smaller than the
     drift it did not control for.
  2. *Across runs of the same corpus*: does a build change move the per-block cost, paired block
     by block? This is the measurement the floor above protects.

Inputs are the JSONL a replay writes, in either shape:

  follow JSONL   one record per verdict, `block` + `standalone_validation_us` + `phases`
  batch JSONL    a `blocks` array inside the terminal summary record, same per-block fields

The shape is detected per file; a batch summary is expanded to its per-block records so the two
are analyzed identically. `run_manifest` records are read for the build stamp and the `allocator`
field, not analyzed, and a file holding several segments uses the last one — the same rule
`analyze_follow_bench.py` applies, for the same reason.

Statistics, and why these and not the obvious alternatives:

  * Within-run drift is the **ratio of half medians** (second half / first half), never the mean
    of per-block ratios: the halves are different blocks, so there is nothing to pair, and a mean
    of ratios would weight cheap blocks like expensive ones.
  * Its interval is a **moving-block bootstrap**. Consecutive blocks are serially correlated —
    they share a warm cache, a heap state, and a workload regime — so an i.i.d. resample would
    report an interval several times too narrow and turn every run into a significant one.
  * Cross-run comparison **pairs by block number** and reports the median of per-block ratios with
    an ordinary bootstrap interval. Pairing is legitimate here and only here: the two runs replayed
    the same recorded corpus, so the same block appears in both.
  * Quarter medians are reported beside the half split because a monotone climb and a single step
    have the same half-split ratio and very different causes.
  * Phase attribution is by **Q4 - Q1 median difference in absolute time**, ranked. A phase that
    doubles from 0.2 ms explains nothing; the ranking answers "where did the milliseconds go",
    which is the question a fix has to target.
  * Where the record carries `trie_storage_tries_copied`, cost **per copy** is reported. This is
    the discriminator between "the run got more work" and "the same work got slower", and those
    two have entirely different fixes.

Nulls are counted, never treated as zero, and every row carries the surviving sample count beside
the population it was drawn from.

Usage:
  analyze_run_drift.py RUN.jsonl [RUN2.jsonl ...] [--json PATH] [--label NAME ...]
  analyze_run_drift.py --self-check
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys

# Metric the drift and the A/B are both reported on. The complete standalone boundary rather than
# a component: a fix that moves work between phases without removing it must not read as a win.
PRIMARY = "standalone_validation_us"

# Moving-block length, in blocks. Long enough to carry the serial correlation the resample has to
# preserve, short enough that a 900-block quarter still contributes many distinct blocks.
BLOCK_LEN = 50

# Fixed so a report is reproducible from its inputs. A seed that varied per run would make two
# readings of the same file disagree in the last digit and invite re-rolling until a number reads
# well.
SEED = 20260825

RESAMPLES = 2000


def load_records(path):
    """Returns (records, manifest) for either JSONL shape, using the file's last run segment."""
    segments = [[]]
    manifest = None
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") == "run_manifest":
                segments.append([])
                manifest = record
                continue
            segments[-1].append(record)
    records = segments[-1]

    # A batch run writes its per-block detail inside the terminal summary rather than as lines.
    expanded = []
    for record in records:
        if isinstance(record.get("blocks"), list):
            expanded.extend(record["blocks"])
        else:
            expanded.append(record)
    return expanded, manifest


def block_number(record):
    """Blocks are keyed by number; `sequence` is the fallback a pre-number recording needs."""
    for key in ("block", "number", "sequence"):
        value = record.get(key)
        if value is not None:
            return value
    return None


def select(records, population):
    """Verdicts carrying the primary metric, in block order, filtered to one population.

    `steady` keeps only the live tail where the record says which it is. A batch replay has no
    tail to be at, so a record with no `tail_live` at all is kept rather than dropped — dropping it
    would silently empty every batch file.
    """
    kept = []
    for record in records:
        if record.get(PRIMARY) is None:
            continue
        if population == "steady" and record.get("tail_live") is False:
            continue
        number = block_number(record)
        if number is None:
            continue
        kept.append((number, record))
    kept.sort(key=lambda pair: pair[0])
    return kept


def phase_value(record, name):
    """Phase times live under `phases`; the copy counters live under `details`."""
    for container in ("phases", "details"):
        section = record.get(container)
        if isinstance(section, dict) and section.get(name) is not None:
            return section[name]
    return None


def phase_names(selected):
    names = set()
    for _, record in selected:
        section = record.get("phases")
        if isinstance(section, dict):
            names.update(key for key, value in section.items() if isinstance(value, (int, float)))
    return sorted(names)


def median_of(values):
    return statistics.median(values) if values else None


def effective_block_len(count):
    """Block length actually used, shrunk when the sample is too short to carry the default.

    A fixed 50 over a 100-value half draws two blocks per replicate, which is so coarse that the
    interval can miss the point estimate entirely — the failure is silent and reads like a
    confident narrow answer. Twenty blocks per replicate is the floor this keeps.
    """
    return max(1, min(BLOCK_LEN, count // 20))


def moving_block_resample(values, rng, block_len):
    """One moving-block bootstrap replicate, preserving local serial structure."""
    if len(values) <= block_len:
        return [values[rng.randrange(len(values))] for _ in values]
    out = []
    while len(out) < len(values):
        start = rng.randrange(0, len(values) - block_len)
        out.extend(values[start : start + block_len])
    return out[: len(values)]


def interval(replicates):
    replicates = sorted(replicates)
    low = replicates[int(0.025 * len(replicates))]
    high = replicates[min(int(0.975 * len(replicates)), len(replicates) - 1)]
    return [low, high]


def drift(selected, rng):
    """Half-split ratio with a moving-block interval, plus quarter medians."""
    values = [record[PRIMARY] for _, record in selected]
    count = len(values)
    if count < 8:
        return None
    half = count // 2
    first, second = values[:half], values[half : 2 * half]
    ratio = statistics.median(second) / statistics.median(first)
    block_len = effective_block_len(half)
    replicates = [
        statistics.median(moving_block_resample(second, rng, block_len))
        / statistics.median(moving_block_resample(first, rng, block_len))
        for _ in range(RESAMPLES)
    ]
    quarter = count // 4
    quarters = [
        statistics.median(values[index * quarter : (index + 1) * quarter]) for index in range(4)
    ]
    return {
        "n": count,
        "block_len": block_len,
        "first_half_p50_us": statistics.median(first),
        "second_half_p50_us": statistics.median(second),
        "ratio": ratio,
        "interval": interval(replicates),
        "resolves_drift": interval(replicates)[0] > 1.0 or interval(replicates)[1] < 1.0,
        "quarter_p50_us": quarters,
        "quarter_ratio": quarters[3] / quarters[0] if quarters[0] else None,
    }


def phase_attribution(selected):
    """Ranked Q4 - Q1 median difference per phase, in absolute time."""
    count = len(selected)
    quarter = count // 4
    if quarter < 2:
        return []
    rows = []
    for name in phase_names(selected):
        medians = []
        for index in range(4):
            window = selected[index * quarter : (index + 1) * quarter]
            values = [
                phase_value(record, name)
                for _, record in window
                if phase_value(record, name) is not None
            ]
            medians.append(median_of(values))
        if any(value is None for value in medians) or not medians[0]:
            continue
        rows.append(
            {
                "phase": name,
                "quarter_p50_us": medians,
                "delta_us": medians[3] - medians[0],
                "ratio": medians[3] / medians[0],
            }
        )
    rows.sort(key=lambda row: row["delta_us"], reverse=True)
    return rows


def per_copy_cost(selected):
    """Cost of the copy-carrying phases per copy-on-write copy, by quarter.

    Discriminates workload growth from unit-cost growth. Rising copies with flat unit cost is a
    heavier corpus; falling copies with rising unit cost is the process getting slower at the same
    work, which is what an allocator effect looks like.
    """
    count = len(selected)
    quarter = count // 4
    if quarter < 2:
        return None
    carriers = ("state_root_us", "trie_retention_us")
    quarters, copies = [], []
    for index in range(4):
        window = selected[index * quarter : (index + 1) * quarter]
        unit, counted = [], []
        for _, record in window:
            copied = phase_value(record, "trie_storage_tries_copied")
            if not copied:
                continue
            total = sum(phase_value(record, name) or 0 for name in carriers)
            unit.append(total / copied)
            counted.append(copied)
        if not unit:
            return None
        quarters.append(statistics.median(unit))
        copies.append(statistics.median(counted))
    return {
        "quarter_us_per_copy": quarters,
        "quarter_p50_copies": copies,
        "unit_cost_ratio": quarters[3] / quarters[0] if quarters[0] else None,
        "copies_ratio": copies[3] / copies[0] if copies[0] else None,
    }


def compare(baseline, candidate, rng):
    """Per-block paired ratio, candidate / baseline, over the blocks both runs verified."""
    left = {number: record for number, record in baseline}
    right = {number: record for number, record in candidate}
    shared = sorted(set(left) & set(right))
    ratios = [right[number][PRIMARY] / left[number][PRIMARY] for number in shared]
    if not ratios:
        return None
    replicates = []
    for _ in range(RESAMPLES):
        sample = [ratios[rng.randrange(len(ratios))] for _ in ratios]
        replicates.append(statistics.median(sample))
    phases = []
    for name in sorted(set(phase_names(baseline)) & set(phase_names(candidate))):
        pairs = [
            (phase_value(left[number], name), phase_value(right[number], name))
            for number in shared
            if phase_value(left[number], name) and phase_value(right[number], name)
        ]
        if len(pairs) < max(8, len(shared) // 10):
            continue
        phases.append(
            {
                "phase": name,
                "baseline_p50_us": statistics.median(pair[0] for pair in pairs),
                "candidate_p50_us": statistics.median(pair[1] for pair in pairs),
                "paired_median_ratio": statistics.median(
                    pair[1] / pair[0] for pair in pairs
                ),
                "n": len(pairs),
            }
        )
    phases.sort(key=lambda row: row["baseline_p50_us"] - row["candidate_p50_us"], reverse=True)
    return {
        "paired_blocks": len(shared),
        "baseline_only": len(set(left) - set(right)),
        "candidate_only": len(set(right) - set(left)),
        "baseline_p50_us": statistics.median(left[number][PRIMARY] for number in shared),
        "candidate_p50_us": statistics.median(right[number][PRIMARY] for number in shared),
        "paired_median_ratio": statistics.median(ratios),
        "interval": interval(replicates),
        "phases": phases,
    }


def analyze(paths, labels, population):
    rng = random.Random(SEED)
    runs = []
    for index, path in enumerate(paths):
        records, manifest = load_records(path)
        selected = select(records, population)
        provenance = (manifest or {}).get("provenance") or {}
        runs.append(
            {
                "label": labels[index] if index < len(labels) else (manifest or {}).get("label")
                or path,
                "path": path,
                "allocator": (manifest or {}).get("allocator"),
                "build_commit": provenance.get("build_commit"),
                "n": len(selected),
                "p50_us": median_of([record[PRIMARY] for _, record in selected]),
                "drift": drift(selected, rng),
                "phase_attribution": phase_attribution(selected),
                "per_copy": per_copy_cost(selected),
                "_selected": selected,
            }
        )
    comparisons = []
    for run in runs[1:]:
        result = compare(runs[0]["_selected"], run["_selected"], rng)
        if result:
            result["baseline"] = runs[0]["label"]
            result["candidate"] = run["label"]
            comparisons.append(result)
    for run in runs:
        del run["_selected"]
    return {
        "schema_version": 1,
        "metric": PRIMARY,
        "population": population,
        "moving_block_len": BLOCK_LEN,
        "resamples": RESAMPLES,
        "seed": SEED,
        "runs": runs,
        "comparisons": comparisons,
    }


def print_human(result):
    print(f"# Run drift and cross-run comparison — {result['metric']}")
    print()
    print(f"Population `{result['population']}`; moving-block length {result['moving_block_len']}, "
          f"{result['resamples']} resamples, seed {result['seed']}.")
    for run in result["runs"]:
        print()
        print(f"## {run['label']}")
        stamp = ", ".join(
            part
            for part in (
                f"allocator {run['allocator']}" if run["allocator"] else None,
                f"commit {run['build_commit'][:10]}" if run["build_commit"] else None,
            )
            if part
        )
        print(f"n={run['n']}  p50={run['p50_us'] / 1000:.2f} ms" + (f"  ({stamp})" if stamp else ""))
        drift_row = run["drift"]
        if drift_row:
            low, high = drift_row["interval"]
            verdict = (
                "resolves a time-order effect"
                if drift_row["resolves_drift"]
                else "no time-order effect resolvable at this sample size"
            )
            print()
            shrunk = (
                f", block length {drift_row['block_len']} (shrunk for the sample size)"
                if drift_row["block_len"] != BLOCK_LEN
                else ""
            )
            print(f"- half split (second/first): **{drift_row['ratio']:.3f}x**  "
                  f"95% moving-block [{low:.3f}, {high:.3f}]{shrunk} — {verdict}")
            quarters = "  ".join(f"{value / 1000:.2f}" for value in drift_row["quarter_p50_us"])
            print(f"- quarter p50 (ms): {quarters}   Q4/Q1 {drift_row['quarter_ratio']:.3f}x")
        if run["per_copy"]:
            copy_row = run["per_copy"]
            print(f"- per CoW copy: {copy_row['quarter_us_per_copy'][0]:.0f} -> "
                  f"{copy_row['quarter_us_per_copy'][3]:.0f} us "
                  f"({copy_row['unit_cost_ratio']:.3f}x) while copies moved "
                  f"{copy_row['copies_ratio']:.3f}x")
        rows = [row for row in run["phase_attribution"] if abs(row["delta_us"]) >= 100][:8]
        if rows:
            print()
            print("| phase | Q1 | Q4 | delta | ratio |")
            print("| --- | ---: | ---: | ---: | ---: |")
            for row in rows:
                print(f"| {row['phase']} | {row['quarter_p50_us'][0] / 1000:.2f} | "
                      f"{row['quarter_p50_us'][3] / 1000:.2f} | "
                      f"{row['delta_us'] / 1000:+.2f} ms | {row['ratio']:.3f}x |")
    for comparison in result["comparisons"]:
        low, high = comparison["interval"]
        print()
        print(f"## {comparison['candidate']} vs {comparison['baseline']}")
        print(f"paired on {comparison['paired_blocks']} blocks "
              f"(+{comparison['baseline_only']} baseline-only, "
              f"+{comparison['candidate_only']} candidate-only)")
        print()
        print(f"- p50 {comparison['baseline_p50_us'] / 1000:.2f} -> "
              f"{comparison['candidate_p50_us'] / 1000:.2f} ms")
        print(f"- paired median ratio **{comparison['paired_median_ratio']:.4f}x**  "
              f"95% [{low:.4f}, {high:.4f}]")
        rows = [
            row
            for row in comparison["phases"]
            if abs(row["baseline_p50_us"] - row["candidate_p50_us"]) >= 100
        ][:8]
        if rows:
            print()
            print("| phase | baseline | candidate | paired ratio |")
            print("| --- | ---: | ---: | ---: |")
            for row in rows:
                print(f"| {row['phase']} | {row['baseline_p50_us'] / 1000:.2f} | "
                      f"{row['candidate_p50_us'] / 1000:.2f} | "
                      f"{row['paired_median_ratio']:.4f}x |")


def self_check():
    """Checks the two statistics against inputs whose answers are known by construction."""
    import tempfile
    import os

    failures = []

    def write(path, records, allocator=None):
        with open(path, "w") as handle:
            handle.write(json.dumps({"kind": "run_manifest", "label": "t",
                                     "allocator": allocator, "provenance": {}}) + "\n")
            for record in records:
                handle.write(json.dumps(record) + "\n")

    with tempfile.TemporaryDirectory() as tmp:
        # A run whose second half costs exactly 20% more, with a phase carrying all of it.
        flat, sloped = [], []
        for index in range(1000):
            base = 100_000
            bump = 20_000 if index >= 500 else 0
            flat.append(
                {
                    "block": index,
                    PRIMARY: base,
                    "phases": {"state_root_us": 40_000, "evm_us": 20_000},
                    "details": {"trie_storage_tries_copied": 100},
                }
            )
            sloped.append(
                {
                    "block": index,
                    PRIMARY: base + bump,
                    "phases": {"state_root_us": 40_000 + bump, "evm_us": 20_000},
                    "details": {"trie_storage_tries_copied": 100},
                }
            )
        sloped_path = os.path.join(tmp, "sloped.jsonl")
        flat_path = os.path.join(tmp, "flat.jsonl")
        write(sloped_path, sloped, "system")
        write(flat_path, flat, "jemalloc")

        result = analyze([sloped_path, flat_path], [], "steady")
        drift_row = result["runs"][0]["drift"]
        if abs(drift_row["ratio"] - 1.2) > 1e-9:
            failures.append(f"half split should be exactly 1.2, got {drift_row['ratio']}")
        if not drift_row["resolves_drift"]:
            failures.append("a clean 20% step must resolve as a time-order effect")
        if result["runs"][1]["drift"]["resolves_drift"]:
            failures.append("a flat run must not resolve a time-order effect")

        top = result["runs"][0]["phase_attribution"][0]
        if top["phase"] != "state_root_us":
            failures.append(f"attribution should name state_root_us, named {top['phase']}")
        if abs(top["delta_us"] - 20_000) > 1e-9:
            failures.append(f"attribution delta should be 20000, got {top['delta_us']}")

        # Cross-run: flat is the candidate. Half the blocks pair 1.0 and half pair 100/120, so an
        # even-sized population puts the median exactly on the crossing between the two halves —
        # (1 + 5/6) / 2. Asserting that rather than either arm's ratio is what catches a median
        # that silently became a mean.
        comparison = result["comparisons"][0]
        if comparison["paired_blocks"] != 1000:
            failures.append(f"expected 1000 paired blocks, got {comparison['paired_blocks']}")
        if abs(comparison["paired_median_ratio"] - 11 / 12) > 1e-9:
            failures.append(
                f"paired ratio should be 11/12, got {comparison['paired_median_ratio']}"
            )

        # A batch-shaped file must expand to the same answer as the line-shaped one.
        batch_path = os.path.join(tmp, "batch.jsonl")
        with open(batch_path, "w") as handle:
            handle.write(json.dumps({"kind": "run_manifest", "label": "b", "provenance": {}}) + "\n")
            handle.write(json.dumps({"agreed": True, "blocks": sloped}) + "\n")
        batch = analyze([batch_path], [], "steady")
        if batch["runs"][0]["n"] != 1000:
            failures.append(f"batch expansion should yield 1000 records, got {batch['runs'][0]['n']}")
        if abs(batch["runs"][0]["drift"]["ratio"] - 1.2) > 1e-9:
            failures.append("batch expansion must reproduce the line-shaped answer")

        # Unit cost must separate from workload: same total, twice the copies, half the unit cost.
        heavier = [
            {
                "block": index,
                PRIMARY: 100_000,
                "phases": {"state_root_us": 40_000, "trie_retention_us": 20_000},
                "details": {"trie_storage_tries_copied": 100 if index < 500 else 200},
            }
            for index in range(1000)
        ]
        heavier_path = os.path.join(tmp, "heavier.jsonl")
        write(heavier_path, heavier)
        copy_row = analyze([heavier_path], [], "steady")["runs"][0]["per_copy"]
        if abs(copy_row["copies_ratio"] - 2.0) > 1e-9:
            failures.append(f"copies ratio should be 2.0, got {copy_row['copies_ratio']}")
        if abs(copy_row["unit_cost_ratio"] - 0.5) > 1e-9:
            failures.append(f"unit cost ratio should be 0.5, got {copy_row['unit_cost_ratio']}")

    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    print("self-check: " + ("FAILED" if failures else "passed"))
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="*", metavar="RUN.jsonl",
                        help="replay JSONL; the first is the baseline every other is compared to")
    # Repeatable alternative to the positionals. Positionals cannot be interleaved with `--label`
    # — argparse stops collecting them at the first option — so a caller naming every run has to
    # pass them as options too, and a caller naming none can keep the shorter positional form.
    parser.add_argument("--run", action="append", default=[], metavar="PATH", dest="run",
                        help="replay JSONL, repeatable; use instead of the positionals when "
                             "pairing each run with a --label")
    parser.add_argument("--label", action="append", default=[], metavar="NAME",
                        help="name for each run, in order; defaults to the manifest label")
    parser.add_argument("--population", choices=("steady", "all"), default="steady",
                        help="steady drops records flagged as backlog; all keeps everything")
    parser.add_argument("--json", metavar="PATH", help="write the report as JSON")
    parser.add_argument("--self-check", action="store_true", help="run the built-in checks")
    args = parser.parse_args()

    if args.self_check:
        return self_check()
    runs = args.runs + args.run
    if not runs:
        parser.error("give at least one run, or --self-check")
    if args.label and len(args.label) != len(runs):
        parser.error(
            f"{len(args.label)} labels for {len(runs)} runs; labels are positional, so a partial "
            "list would silently name the wrong run"
        )

    result = analyze(runs, args.label, args.population)
    print_human(result)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
