#!/usr/bin/env python3
"""Compare two structured builder benchmark runs.

Two join modes, because the environment decides which one is available:

`block-identical` joins on block hash and demands equal witness commitments. It is the strong
comparison and needs both runs to have seen the same blocks, which requires either concurrent runs
or a replayable datadir.

`non-overlapping` is for a single live datadir, where the control and the candidate necessarily
cover different blocks. There is no per-block pairing to be had, so the comparison is between two
distributions and the workload difference between the ranges is reported next to the timing
difference. The rule this mode exists to enforce: a timing difference smaller than the workload
difference is not evidence.
"""

import argparse
import collections
import random
import statistics
from pathlib import Path

from analyze_validation_bench import load_jsonl, percentile


METRICS = (
    "builder_total_us",
    "transition_witness_build_us",
    "initial_provider_us",
    "snapshot_us",
)

# What the initial proof was asked to produce. Reported per side so a timing difference can be
# read against the work that produced it rather than on its own.
WORKLOAD = (
    "initial_targets",
    "distinct_storage_tries",
    "initial_proof_nodes",
    "initial_proof_bytes",
)

# Proof sources that mean the parallel path was asked for and did not deliver. Separate from
# `serial-low-width`, which is the threshold working as designed.
PARALLEL_FAILURE_SOURCES = ("serial-after-parallel-error",)

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_BLOCK = 25


def eligible(records):
    return [
        record
        for record in records
        if record.get("block_hash")
        and record.get("cache_parent_synced", False)
        and record.get("sidecar_constructed", False)
    ]


def eligible_by_hash(records):
    return {record["block_hash"].lower(): record for record in eligible(records)}


def check_run_integrity(records, label):
    """Reject a run that cannot be compared at all, before anything is measured.

    Duplicates and missing commitments are not statistical problems to be reported alongside a
    result — they mean the records do not describe what they claim to, so there is no result.
    """
    selected = eligible(records)
    if not selected:
        raise ValueError(f"{label}: no eligible builder samples")

    hashes = [record["block_hash"].lower() for record in selected]
    duplicates = [h for h, count in collections.Counter(hashes).items() if count > 1]
    if duplicates:
        raise ValueError(
            f"{label}: {len(duplicates)} block hashes appear more than once; "
            "the run was restarted over its own range or the file was concatenated"
        )

    numbers = sorted(record["block_number"] for record in selected)
    gaps = [
        (previous, following)
        for previous, following in zip(numbers, numbers[1:])
        if following != previous + 1
    ]

    missing = [record for record in selected if record.get("witness_commitment") is None]
    if missing:
        raise ValueError(f"{label}: {len(missing)} eligible samples have no witness commitment")

    return selected, gaps


def workload_span(selected):
    numbers = [record["block_number"] for record in selected]
    return min(numbers), max(numbers)


def ratio_of_means(numerator, denominator):
    denominator_mean = statistics.fmean(denominator)
    return statistics.fmean(numerator) / denominator_mean if denominator_mean else float("nan")


def moving_block_bootstrap_ratio(candidate, control, resamples=BOOTSTRAP_RESAMPLES):
    """95% interval for the candidate/control ratio of means, without pairing.

    Consecutive blocks are not independent — page-cache state and workload both persist across
    them — so resampling single records would understate the spread. Resampling contiguous blocks
    keeps that dependence inside the sample.
    """
    if not candidate or not control:
        return float("nan"), float("nan")
    rng = random.Random(0x50C0)
    ratios = []
    for _ in range(resamples):
        resampled_candidate = _resample_blocks(candidate, rng)
        resampled_control = _resample_blocks(control, rng)
        control_mean = statistics.fmean(resampled_control)
        if control_mean:
            ratios.append(statistics.fmean(resampled_candidate) / control_mean)
    if not ratios:
        return float("nan"), float("nan")
    ratios.sort()
    return percentile(ratios, 0.025), percentile(ratios, 0.975)


def _resample_blocks(values, rng):
    block = min(BOOTSTRAP_BLOCK, len(values))
    out = []
    while len(out) < len(values):
        start = rng.randrange(0, len(values) - block + 1)
        out.extend(values[start : start + block])
    return out[: len(values)]


def proof_source_lines(selected, label):
    sources = collections.Counter(
        record.get("initial_proof_source", "missing") for record in selected
    )
    failures = sum(sources[source] for source in PARALLEL_FAILURE_SOURCES)
    lines = [f"- {label}: " + ", ".join(f"`{k}` {v}" for k, v in sorted(sources.items()))]
    if failures:
        lines.append(
            f"  - **{failures} samples fell back to serial after a parallel error**; "
            "section 5.2 requires zero"
        )
    return lines


def compare_block_identical(control, candidate, warmup, samples, candidate_source=None):
    control_by_hash = eligible_by_hash(control)
    candidate_by_hash = eligible_by_hash(candidate)
    hashes = sorted(
        control_by_hash.keys() & candidate_by_hash.keys(),
        key=lambda block_hash: candidate_by_hash[block_hash]["block_number"],
    )
    if candidate_source:
        hashes = [
            block_hash
            for block_hash in hashes
            if candidate_by_hash[block_hash].get("initial_proof_source") == candidate_source
        ]
    hashes = hashes[warmup : warmup + samples]
    if len(hashes) < samples:
        raise ValueError(
            f"only {len(hashes)} block-identical samples; requested {samples} after warm-up {warmup}"
            "\nIf the two runs could not cover the same blocks, use --mode non-overlapping."
        )

    missing_commitments = [
        block_hash
        for block_hash in hashes
        if control_by_hash[block_hash].get("witness_commitment") is None
        or candidate_by_hash[block_hash].get("witness_commitment") is None
    ]
    if missing_commitments:
        raise ValueError(
            f"witness commitment missing for {len(missing_commitments)} joined blocks"
        )

    commitment_mismatches = [
        block_hash
        for block_hash in hashes
        if control_by_hash[block_hash].get("witness_commitment")
        != candidate_by_hash[block_hash].get("witness_commitment")
    ]
    if commitment_mismatches:
        raise ValueError(
            f"witness commitment differs for {len(commitment_mismatches)} joined blocks"
        )

    lines = [
        "# Block-identical builder comparison", "",
        f"Joined samples: **{len(hashes)}**",
        f"Candidate proof-source filter: **{candidate_source or 'any'}**",
        "Witness commitment mismatches: **0**", "",
        "| Metric | Control avg | Candidate avg | Candidate/control | Median paired ratio | Control p95 | Candidate p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        control_values = [control_by_hash[block_hash][metric] for block_hash in hashes]
        candidate_values = [candidate_by_hash[block_hash][metric] for block_hash in hashes]
        paired_ratios = [
            candidate_value / control_value
            for control_value, candidate_value in zip(control_values, candidate_values)
            if control_value
        ]
        median_ratio = statistics.median(paired_ratios) if paired_ratios else float("nan")
        lines.append(
            "| {} | {:.2f} ms | {:.2f} ms | {:.3f}x | {:.3f}x | {:.2f} ms | {:.2f} ms |".format(
                metric,
                statistics.fmean(control_values) / 1000,
                statistics.fmean(candidate_values) / 1000,
                ratio_of_means(candidate_values, control_values),
                median_ratio,
                percentile(control_values, 0.95) / 1000,
                percentile(candidate_values, 0.95) / 1000,
            )
        )
    return "\n".join(lines) + "\n"


def compare_non_overlapping(control, candidate, warmup, samples, repeat=None):
    """One AB comparison, plus the BA repeat when it is supplied.

    The repeat is not optional evidence. Two sequential runs on one datadir differ in time order
    as well as in configuration, and section 5.2.3 asks for both orderings precisely because that
    confound is not otherwise separable. Running only AB produces a number this tool will print
    and label as unseparated.
    """
    report = _one_ordering(control, candidate, warmup, samples, "AB")
    if repeat is None:
        return report + _no_repeat_warning()
    repeat_control, repeat_candidate = repeat
    report += _one_ordering(repeat_control, repeat_candidate, warmup, samples, "BA")
    return report + _pooled_orderings(
        _adjusted_for(control, candidate, warmup, samples),
        _adjusted_for(repeat_control, repeat_candidate, warmup, samples),
    )


def _no_repeat_warning():
    return (
        "\n## Ordering\n\n"
        "- Only one ordering was supplied, so configuration and time order are confounded.\n"
        "- Run `--mode self-check` on either run to measure this host's time-order floor, and\n"
        "  supply `--repeat-control`/`--repeat-candidate` for the reversed ordering before\n"
        "  reading the adjusted ratio as an adoption result.\n"
    )


def _pooled_orderings(first, second):
    """Combine AB and BA, and report their spread as the order effect itself."""
    pooled = (first * second) ** 0.5
    spread = abs(first - second)
    return (
        "\n## Ordering\n\n"
        f"- AB adjusted ratio: **{first:.3f}x**\n"
        f"- BA adjusted ratio: **{second:.3f}x**\n"
        f"- Pooled (geometric mean): **{pooled:.3f}x**\n"
        f"- Spread between orderings: **{spread:.3f}**, which is the order effect this pair of\n"
        "  runs can see. A pooled result closer to 1.0 than this spread is not separable from\n"
        "  time order.\n"
    )


def _adjusted_for(control, candidate, warmup, samples):
    control_selected, _ = check_run_integrity(control, "control")
    candidate_selected, _ = check_run_integrity(candidate, "candidate")
    control_selected = control_selected[warmup : warmup + samples]
    candidate_selected = candidate_selected[warmup : warmup + samples]
    return adjusted_ratio(fit_cost_model(control_selected), candidate_selected)


def compare_self_check(records, warmup, samples):
    """Compare one run against itself, split in half.

    Both halves ran the same configuration, so the adjusted ratio should be 1.0 and whatever it
    actually is, is this host's floor: no A/B result smaller than it distinguishes anything. Two
    sequential runs cannot be closer together in time than two halves of one run, so this is a
    lower bound on the confound, not an estimate of it.
    """
    selected, _ = check_run_integrity(records, "run")
    selected = selected[warmup:]
    half = len(selected) // 2
    if half < samples:
        raise ValueError(
            f"only {len(selected)} eligible samples after warm-up {warmup}; "
            f"a self-check at {samples} per half needs {samples * 2}"
        )
    first = selected[half - samples : half]
    second = selected[half : half + samples]
    ratio = adjusted_ratio(fit_cost_model(first), second)
    low, high = bootstrap_adjusted_ratio(first, second)
    separable = low > 1.0 or high < 1.0
    return "\n".join([
        "# Self-check (A/A)", "",
        "One run split in half. Both halves are the same configuration, so a result other than",
        "1.0 measures time order and page-cache drift rather than any change.", "",
        f"- First half: blocks {workload_span(first)[0]}--{workload_span(first)[1]}",
        f"- Second half: blocks {workload_span(second)[0]}--{workload_span(second)[1]}",
        f"- Adjusted second/first: **{ratio:.3f}x**",
        f"- Moving-block bootstrap 95% interval: **[{low:.3f}, {high:.3f}]**", "",
        (
            f"- **Floor: {abs(1 - ratio) * 100:.1f}%.** An A/B on this host that moves the adjusted "
            "ratio by less than this has not separated the configurations."
            if separable
            else "- The interval spans 1.0: no time-order effect is resolvable at this sample size, "
            "so an A/B result is limited by sampling error rather than by drift."
        ),
        "",
    ])


def _one_ordering(control, candidate, warmup, samples, ordering):
    control_selected, control_gaps = check_run_integrity(control, "control")
    candidate_selected, candidate_gaps = check_run_integrity(candidate, "candidate")
    control_selected = control_selected[warmup : warmup + samples]
    candidate_selected = candidate_selected[warmup : warmup + samples]
    for label, selected in (("control", control_selected), ("candidate", candidate_selected)):
        if len(selected) < samples:
            raise ValueError(
                f"{label}: only {len(selected)} eligible samples after warm-up {warmup}; "
                f"requested {samples}"
            )

    control_span = workload_span(control_selected)
    candidate_span = workload_span(candidate_selected)
    if control_span[1] >= candidate_span[0] and candidate_span[1] >= control_span[0]:
        raise ValueError(
            f"the ranges overlap (control {control_span}, candidate {candidate_span}); "
            "use --mode block-identical, which is the stronger comparison"
        )

    lines = [
        f"# Non-overlapping builder comparison ({ordering})", "",
        "The two runs cover different blocks, so nothing here is a paired measurement. Read the",
        "timing table only against the workload table below it.", "",
        f"- Control samples: **{len(control_selected)}**, blocks {control_span[0]}--{control_span[1]}",
        f"- Candidate samples: **{len(candidate_selected)}**, blocks {candidate_span[0]}--{candidate_span[1]}",
        f"- Control block-number gaps: **{len(control_gaps)}**",
        f"- Candidate block-number gaps: **{len(candidate_gaps)}**", "",
        "## Initial proof selection", "",
        *proof_source_lines(control_selected, "Control"),
        *proof_source_lines(candidate_selected, "Candidate"),
        "",
        "## Workload", "",
        "| Input | Control avg | Candidate avg | Candidate/control |",
        "| --- | ---: | ---: | ---: |",
    ]
    workload_ratios = {}
    for field in WORKLOAD:
        control_values = [record[field] for record in control_selected]
        candidate_values = [record[field] for record in candidate_selected]
        ratio = ratio_of_means(candidate_values, control_values)
        workload_ratios[field] = ratio
        lines.append(
            "| {} | {:.1f} | {:.1f} | {:.3f}x |".format(
                field,
                statistics.fmean(control_values),
                statistics.fmean(candidate_values),
                ratio,
            )
        )

    lines += [
        "",
        "## Timing", "",
        "| Metric | Control avg | Candidate avg | Candidate/control | Control p95 | Candidate p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    timing_ratios = {}
    for metric in METRICS:
        control_values = [record[metric] for record in control_selected]
        candidate_values = [record[metric] for record in candidate_selected]
        ratio = ratio_of_means(candidate_values, control_values)
        timing_ratios[metric] = ratio
        lines.append(
            "| {} | {:.2f} ms | {:.2f} ms | {:.3f}x | {:.2f} ms | {:.2f} ms |".format(
                metric,
                statistics.fmean(control_values) / 1000,
                statistics.fmean(candidate_values) / 1000,
                ratio,
                percentile(control_values, 0.95) / 1000,
                percentile(candidate_values, 0.95) / 1000,
            )
        )

    # Per-node cost is reported but deliberately not used as the gate. It assumes the initial
    # provider costs a fixed amount per proof node, and it does not: a large share of the call is
    # fixed setup, so a range that happens to contain bigger blocks looks cheaper per node without
    # anything having improved. Splitting one run in half reproduces that artifact.
    control_normalized = _per_node(control_selected)
    candidate_normalized = _per_node(candidate_selected)

    # The gate instead uses the adjustment an earlier build-parity run established: fit the
    # control's own cost model, evaluate it at the candidate's composition, and compare the
    # candidate with what the control would have cost on the candidate's blocks.
    fit = fit_cost_model(control_selected)
    adjusted = adjusted_ratio(fit, candidate_selected)
    low, high = bootstrap_adjusted_ratio(control_selected, candidate_selected)

    lines += [
        "",
        "## Workload-normalized initial proof", "",
        f"- Control: **{statistics.fmean(control_normalized):.3f} us/proof node**",
        f"- Candidate: **{statistics.fmean(candidate_normalized):.3f} us/proof node**",
        "- Descriptive only: per-node cost falls as blocks grow, so this favours whichever side",
        "  drew the larger blocks. The composition-adjusted comparison below is the gate.",
        "",
        "## Composition-adjusted initial proof", "",
        "This section models the initial multiproof, so it answers the proof-generation question",
        "only. For the parent-cache snapshot read `snapshot_us` and `builder_total_us` in the",
        "timing table against the workload table; the snapshot scales with cache size, which is",
        "not one of the inputs modelled here.",
        "",
        f"- Control model: **{fit.intercept / 1000:.2f} ms + "
        f"{fit.proof_nodes:.3f} us/node + {fit.storage_tries:.3f} us/storage trie**, "
        f"R-squared **{fit.r_squared:.2f}**",
        f"- Control model evaluated at the candidate's composition: "
        f"**{fit.predict_mean(candidate_selected) / 1000:.2f} ms**",
        f"- Candidate actual: "
        f"**{statistics.fmean(r['initial_provider_us'] for r in candidate_selected) / 1000:.2f} ms**",
        f"- Adjusted candidate/control: **{adjusted:.3f}x**",
        f"- Moving-block bootstrap 95% interval: **[{low:.3f}, {high:.3f}]**",
        "",
        "## Verdict inputs", "",
        _evidence_verdict(timing_ratios["initial_provider_us"], workload_ratios, fit, adjusted, low, high),
        "",
    ]
    return "\n".join(lines) + "\n"


class CostModel:
    """Ordinary least squares for initial-proof cost against the two inputs that drive it.

    Two predictors rather than one because parallel proof generation changes how storage tries are
    walked, so a model that only knows the total node count cannot tell a wider block from a deeper
    one.
    """

    def __init__(self, intercept, proof_nodes, storage_tries, r_squared):
        self.intercept = intercept
        self.proof_nodes = proof_nodes
        self.storage_tries = storage_tries
        self.r_squared = r_squared

    def predict(self, record):
        return (
            self.intercept
            + self.proof_nodes * record["initial_proof_nodes"]
            + self.storage_tries * record["distinct_storage_tries"]
        )

    def predict_mean(self, records):
        return statistics.fmean(self.predict(record) for record in records)


def fit_cost_model(records):
    rows = [
        (1.0, float(r["initial_proof_nodes"]), float(r["distinct_storage_tries"]))
        for r in records
    ]
    observed = [float(r["initial_provider_us"]) for r in records]
    coefficients = _solve_normal_equations(rows, observed)
    if coefficients is None:
        # A degenerate range (every block the same width) has no model to fit. Falling back to the
        # mean makes the adjustment a no-op and the reported R-squared zero, which is the honest
        # description of what was learned.
        return CostModel(statistics.fmean(observed), 0.0, 0.0, 0.0)
    intercept, proof_nodes, storage_tries = coefficients
    model = CostModel(intercept, proof_nodes, storage_tries, 0.0)
    mean = statistics.fmean(observed)
    total = sum((value - mean) ** 2 for value in observed)
    residual = sum(
        (value - model.predict(record)) ** 2 for value, record in zip(observed, records)
    )
    model.r_squared = 1 - residual / total if total else 0.0
    return model


def _solve_normal_equations(rows, observed):
    """Solve X'X b = X'y by Gaussian elimination, for the 3x3 system two predictors produce."""
    width = len(rows[0])
    matrix = [
        [sum(row[i] * row[j] for row in rows) for j in range(width)]
        + [sum(row[i] * value for row, value in zip(rows, observed))]
        for i in range(width)
    ]
    for column in range(width):
        pivot = max(range(column, width), key=lambda r: abs(matrix[r][column]))
        if abs(matrix[pivot][column]) < 1e-9:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        matrix[column] = [value / divisor for value in matrix[column]]
        for other in range(width):
            if other == column:
                continue
            factor = matrix[other][column]
            matrix[other] = [
                value - factor * pivot_value
                for value, pivot_value in zip(matrix[other], matrix[column])
            ]
    return [row[width] for row in matrix]


def adjusted_ratio(fit, candidate_selected):
    predicted = fit.predict_mean(candidate_selected)
    actual = statistics.fmean(r["initial_provider_us"] for r in candidate_selected)
    return actual / predicted if predicted else float("nan")


def bootstrap_adjusted_ratio(control_selected, candidate_selected, resamples=BOOTSTRAP_RESAMPLES):
    """Interval for the adjusted ratio, resampling both sides.

    The control is refit on every resample because the model itself is estimated from a finite
    sample; treating its coefficients as known would report an interval narrower than the evidence
    supports.
    """
    rng = random.Random(0x50C0)
    ratios = []
    for _ in range(resamples):
        control_resample = _resample_blocks(control_selected, rng)
        candidate_resample = _resample_blocks(candidate_selected, rng)
        ratio = adjusted_ratio(fit_cost_model(control_resample), candidate_resample)
        if ratio == ratio:
            ratios.append(ratio)
    if not ratios:
        return float("nan"), float("nan")
    ratios.sort()
    return percentile(ratios, 0.025), percentile(ratios, 0.975)


def _per_node(selected):
    return [
        record["initial_provider_us"] / record["initial_proof_nodes"]
        for record in selected
        if record.get("initial_proof_nodes")
    ]


def _evidence_verdict(timing_ratio, workload_ratios, fit, adjusted, low, high):
    """State whether the result clears section 5.2's stated bars, and nothing more."""
    timing_change = abs(1 - timing_ratio)
    workload_change = max(abs(1 - ratio) for ratio in workload_ratios.values())
    lines = [
        f"- Raw `initial_provider_us` moved **{timing_change * 100:.1f}%**; "
        f"the largest workload input moved **{workload_change * 100:.1f}%**.",
    ]
    if timing_change <= workload_change:
        lines.append(
            "- **The raw table is not evidence.** A timing difference smaller than "
            "the workload difference does not distinguish the change from the blocks it ran on. "
            "Read the adjusted result instead."
        )
    else:
        lines.append("- Timing moved by more than the workload did, so the raw table is readable.")

    if fit.r_squared < 0.30:
        lines.append(
            f"- **The adjustment is weak: control model R-squared is {fit.r_squared:.2f}.** The "
            "model explains too little of the per-block variation to carry an adoption decision; "
            "an earlier build-parity run declined a 0.19 fit for the same reason. Treat the "
            "interval below as "
            "descriptive and get a block-identical comparison before adopting."
        )
    if high != high:
        lines.append("- Adjusted bootstrap interval unavailable.")
    elif high < 1.0:
        lines.append(
            f"- Adjusted ratio **{adjusted:.3f}x**, interval upper bound **{high:.3f} < 1.0**: "
            "the candidate is cheaper than the control's own model predicts for the candidate's "
            "blocks."
        )
    elif low > 1.0:
        lines.append(
            f"- Adjusted ratio **{adjusted:.3f}x**, interval lower bound **{low:.3f} > 1.0**: "
            "the candidate is *more* expensive than the control model predicts."
        )
    else:
        lines.append(
            f"- Adjusted ratio **{adjusted:.3f}x**, interval **[{low:.3f}, {high:.3f}]** spans "
            "1.0: this run does not separate the two configurations."
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path, nargs="?")
    parser.add_argument(
        "--mode",
        choices=("block-identical", "non-overlapping", "self-check"),
        default="block-identical",
        help=(
            "block-identical joins on block hash and is the stronger comparison; "
            "non-overlapping is for two sequential runs on one live datadir; "
            "self-check splits one run in half to measure this host's time-order floor"
        ),
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument(
        "--candidate-source",
        help="block-identical only: keep candidate samples with this initial_proof_source",
    )
    parser.add_argument(
        "--repeat-control",
        type=Path,
        help="non-overlapping only: the control run of the reversed (BA) ordering",
    )
    parser.add_argument(
        "--repeat-candidate",
        type=Path,
        help="non-overlapping only: the candidate run of the reversed (BA) ordering",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode != "self-check" and args.candidate is None:
        parser.error("candidate is required unless --mode self-check")
    if bool(args.repeat_control) != bool(args.repeat_candidate):
        parser.error("--repeat-control and --repeat-candidate must be given together")
    if args.repeat_control and args.mode != "non-overlapping":
        parser.error("the BA repeat applies only to --mode non-overlapping")
    try:
        if args.mode == "block-identical":
            report = compare_block_identical(
                load_jsonl(args.control),
                load_jsonl(args.candidate),
                args.warmup,
                args.samples,
                args.candidate_source,
            )
        elif args.mode == "self-check":
            report = compare_self_check(load_jsonl(args.control), args.warmup, args.samples)
        else:
            if args.candidate_source:
                parser.error("--candidate-source applies only to --mode block-identical")
            repeat = (
                (load_jsonl(args.repeat_control), load_jsonl(args.repeat_candidate))
                if args.repeat_control
                else None
            )
            report = compare_non_overlapping(
                load_jsonl(args.control),
                load_jsonl(args.candidate),
                args.warmup,
                args.samples,
                repeat,
            )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(report, end="")
    if args.output:
        args.output.write_text(report)


if __name__ == "__main__":
    main()
