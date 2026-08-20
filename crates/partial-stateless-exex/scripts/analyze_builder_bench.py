#!/usr/bin/env python3
"""Analyze structured partial-stateless builder benchmark records."""

import argparse
import collections
import random
import statistics
from pathlib import Path

from analyze_validation_bench import (
    format_summary,
    load_jsonl,
    retained_generation_lines,
)


def select_builder_samples(records, warmup, limit=None, require_published=False):
    eligible = [
        record
        for record in records
        if record.get("cache_parent_synced", False)
        and record.get("sidecar_constructed", False)
        and (not require_published or record.get("sidecar_published", False))
    ]
    selected = eligible[warmup:]
    return selected[:limit] if limit is not None else selected


ARTIFACT_SCHEMA = 4
"""Builder schema that declares delivery, sampling, and fallback cause (see `benchmark.rs`)."""

TRIE_REPRS = {"parallel", "exact"}


def build_trie_repr_section(selected):
    """Name the representation behind every footprint, without guessing from old schemas.

    `PS_TRIE_REPR` became runtime-selectable while builder records still declared schema 4, so
    schema 4 exists for both representations. Schema 5 closes that provenance gap. Old records
    remain usable when their external arm manifest or startup log supplies the label, but this
    analyzer cannot manufacture it and must not make cross-representation memory claims from it.
    """
    recorded = [record.get("trie_repr") for record in selected if "trie_repr" in record]
    if not recorded:
        schemas = sorted(
            {record.get("schema_version") for record in selected},
            key=lambda value: (value is None, value),
        )
        return [
            "## Trie representation", "",
            "- **Not recorded in the builder records.** Resolve it from the arm manifest or "
            "startup log; do not infer it from the schema number "
            f"(declared schemas: `{schemas}`).", "",
            "The memory estimates below are valid only within that resolved representation. "
            "They are not a cross-representation memory comparison.", "",
        ]

    if len(recorded) != len(selected):
        raise ValueError(
            "mixed builder records: trie_repr is present on only "
            f"{len(recorded)}/{len(selected)} selected samples"
        )
    unknown = sorted({label for label in recorded if label not in TRIE_REPRS})
    if unknown:
        raise ValueError(f"unknown trie_repr label(s): {unknown}")
    labels = sorted(set(recorded))
    if len(labels) != 1:
        raise ValueError(
            f"mixed trie representations in one builder report: {labels}; analyze each arm alone"
        )

    schemas = sorted(
        {record.get("schema_version") for record in selected},
        key=lambda value: (value is None, value),
    )
    return [
        "## Trie representation", "",
        f"- `{labels[0]}` (recorded on all {len(selected)} selected samples; "
        f"declared schemas: `{schemas}`)", "",
        "Footprint estimates below use this representation's own accounting. The counting-"
        "allocator differential is the cross-representation memory meter.", "",
    ]


def bootstrap_mean_ci(values, iterations=10000, seed=0xB3, confidence=0.95):
    """Percentile bootstrap interval for the mean, seeded so a report regenerates identically.

    Per-block EVM time is heavy-tailed, so a mean over a dozen sampled blocks carries far more
    spread than its own arithmetic suggests. The interval is the width of the extrapolation that
    follows, not decoration: without it, an estimate built from a dozen blocks and multiplied by
    several hundred reads like a measurement of several hundred blocks.
    """
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in range(size)) for _ in range(iterations)
    )
    tail = (1.0 - confidence) / 2.0
    return means[int(tail * (iterations - 1))], means[int((1.0 - tail) * (iterations - 1))]


def build_artifact_section(selected):
    """Engine-access artifact delivery and reuse, over the records that can actually answer for it.

    Delivery and reuse are reported as separate rates. They differ by the sampled blocks, which
    are delivered and deliberately re-executed to keep the differential oracle alive, so reading
    reuse as the delivery rate understates a healthy handoff by exactly the sampling fraction.

    Schema 3 recorded only `artifact_reused`. Treating its missing `artifact_available` as False
    would turn a run with a flawless handoff into a 0% delivery report, so those records are
    reported for reuse alone and say so.

    Which records those are is decided by the fields present, not by the declared version, and
    the two disagree for one window: the delivery fields landed before `schema_version` was
    raised to 4, so files exist that carry every schema-4 field while declaring 3. Trusting the
    declared number there would produce exactly the misreading the bump exists to prevent.
    Presence is authoritative; the version is reported so a mislabelled file stays visible.

    The timing split is by `artifact_reused`, because `historical_full_db_evm_us` is zero on the
    reused blocks: a median over the mixed population describes neither path. Note that the
    comparison groups here are not equally trustworthy -- blocks that missed did so because the
    builder was behind, which correlates with heavy blocks, while sampled blocks are chosen by
    height and are unbiased. Only the sampled group is a fair contrast.
    """
    modern = [r for r in selected if "artifact_available" in r]
    legacy = [r for r in selected if "artifact_available" not in r and "artifact_reused" in r]
    if not modern:
        if not legacy:
            return []
        reused = sum(1 for r in legacy if r.get("artifact_reused"))
        return [
            "## Engine access artifact", "",
            f"- Artifact reused: **{reused}/{len(legacy)}** ({reused / len(legacy):.2%})",
            "- Artifact delivered, sampling, and fallback cause: **not recorded** "
            f"(pre-schema-{ARTIFACT_SCHEMA} records)", "",
            "Reuse below 100% here cannot be separated into sampling, misses, and their causes, "
            "and no delivery rate is recoverable from the file. Rerun against a current binary "
            "if delivery is the question.", "",
        ]

    lines = ["## Engine access artifact", ""]
    if legacy:
        lines.extend([
            f"**Mixed records: {len(legacy)} of {len(selected)} accepted records predate the "
            "delivery fields and are excluded from every rate below.** Rates are over the "
            f"{len(modern)} records that carry them.", "",
        ])
    declared = sorted({r.get("schema_version") for r in modern}, key=lambda v: (v is None, v))
    if any(version is None or version < ARTIFACT_SCHEMA for version in declared):
        lines.extend([
            f"**Schema label {declared} predates the bump to {ARTIFACT_SCHEMA}, but the delivery "
            "fields are present and are what this section is computed from.** Written between "
            "the fields landing and the version being raised; the numbers are unaffected.", "",
        ])

    available = sum(1 for r in modern if r.get("artifact_available"))
    reused = [r for r in modern if r.get("artifact_reused")]
    sampled = [r for r in modern if r.get("fallback_reason") == "shadow_sampled"]
    reasons = collections.Counter(
        r["fallback_reason"] for r in modern if r.get("fallback_reason")
    )

    lines.extend([
        f"- Artifact delivered: **{available}/{len(modern)}** ({available / len(modern):.2%})",
        f"- Artifact reused: **{len(reused)}/{len(modern)}** ({len(reused) / len(modern):.2%})",
    ])
    for reason, count in sorted(reasons.items()):
        lines.append(f"- Not reused, `{reason}`: **{count}**")
    if reused and sampled:
        evm_us = [r["historical_full_db_evm_us"] for r in sampled]
        mean_us = statistics.fmean(evm_us)
        median_us = statistics.median(evm_us)
        interval = bootstrap_mean_ci(evm_us)
        sampled_line = (
            f"- Re-execution measured on the **{len(sampled)}** sampled blocks: mean "
            f"**{mean_us / 1000:.1f} ms**, median **{median_us / 1000:.1f} ms**"
        )
        estimate = f"- Estimated avoided execution CPU: **{mean_us * len(reused) / 1e6:.1f} s**"
        if interval:
            sampled_line += (
                f" (95% bootstrap CI on the mean {interval[0] / 1000:.1f}--"
                f"{interval[1] / 1000:.1f} ms)"
            )
            low, high = (bound * len(reused) / 1e6 for bound in interval)
            estimate += f" (95% bootstrap CI {low:.1f}--{high:.1f} s)"
        lines.extend([
            "",
            sampled_line,
            estimate + f", extrapolating that mean to {len(reused)} reused blocks",
            "",
            "The avoided CPU is an **estimate, not a measurement**. The reused blocks never ran "
            "the EVM, so their re-execution cost does not exist to be measured; this is the "
            "sampled mean scaled by the reused count, and the sampled blocks are the only "
            "unbiased estimator available because they are selected by height rather than by "
            "outcome. The interval is wide because per-block EVM time is heavy-tailed and the "
            "sample is small -- quote it with the point estimate. When forming a ratio against "
            "the Engine's capture cost, keep mean with mean or median with median.",
            "", "| Path | n | p50 | p95 |", "| --- | ---: | ---: | ---: |",
            format_summary_short("Reused (artifact)", [r["builder_total_us"] for r in reused]),
            format_summary_short("Sampled (re-executed)", [r["builder_total_us"] for r in sampled]),
            "",
            "Read the two rows as a sanity check, not as the win. The removed re-execution is a "
            "few percent of builder end-to-end and an order of magnitude below its block-to-block "
            "spread, so it does not surface as a median difference at any sample count.",
        ])
    return lines + [""]


def format_summary_short(label, values):
    ordered = sorted(values)

    def pct(fraction):
        return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))] / 1000

    return f"| {label} | {len(ordered)} | {pct(0.5):.1f} | {pct(0.95):.1f} ms |"


def build_builder_report(
    records,
    warmup,
    requested,
    expect_snapshot=None,
    require_published=False,
):
    selected = select_builder_samples(records, warmup, requested, require_published)
    if len(selected) < requested:
        raise ValueError(
            f"only {len(selected)} eligible builder samples; requested {requested} "
            f"after warm-up {warmup}"
        )

    if expect_snapshot is not None:
        mismatches = [
            record
            for record in selected
            if bool(record.get("snapshot_created", False)) != expect_snapshot
        ]
        if mismatches:
            expected = "created" if expect_snapshot else "skipped"
            raise ValueError(
                f"snapshot invariant failed for {len(mismatches)} samples; expected {expected}"
            )

    proof_sources = collections.Counter(
        record.get("initial_proof_source", "missing") for record in selected
    )
    lines = [
        "# Partial-stateless builder benchmark", "",
        f"Accepted builder samples: **{len(selected)}**",
        f"Post-Ready sample-warm-up records excluded: **{warmup}**", "",
    ]
    lines.extend(build_trie_repr_section(selected))
    lines.extend([
        "## Builder timings", "",
        "| Operation | Average | p50 | p90 | p95 | p99 | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        format_summary("Builder end-to-end", [r["builder_total_us"] for r in selected]),
        format_summary(
            "Historical full-DB EVM", [r["historical_full_db_evm_us"] for r in selected]
        ),
        format_summary(
            "Transition witness build", [r["transition_witness_build_us"] for r in selected]
        ),
        format_summary("Initial provider", [r["initial_provider_us"] for r in selected]),
        format_summary("Previous-cache snapshot", [r["snapshot_us"] for r in selected]), "",
    ])
    lines.extend(build_artifact_section(selected))
    lines.extend([
        "## Initial proof selection", "",
    ])
    for source, count in sorted(proof_sources.items()):
        lines.append(f"- `{source}`: **{count}**")
    lines.extend([
        "",
        f"- Mean initial targets: **{statistics.fmean(r['initial_targets'] for r in selected):.1f}**",
        (
            "- Mean distinct storage tries: "
            f"**{statistics.fmean(r['distinct_storage_tries'] for r in selected):.1f}**"
        ),
        (
            "- Mean parallel storage workers: "
            f"**{statistics.fmean(r['parallel_storage_workers'] for r in selected):.1f}**"
        ),
        (
            "- Mean parallel account workers: "
            f"**{statistics.fmean(r['parallel_account_workers'] for r in selected):.1f}**"
        ),
        f"- Snapshot-created samples: **{sum(bool(r['snapshot_created']) for r in selected)}**",
        f"- Sidecars published: **{sum(bool(r['sidecar_published']) for r in selected)}**",
        f"- Missing witness commitments: **{sum(r.get('witness_commitment') is None for r in selected)}**",
        "",
        "## Builder memory estimates", "",
        "| Structure | Average | p50 | p90 | p95 | p99 | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        format_summary(
            "Snapshot clone", [r["snapshot_estimated_bytes"] for r in selected], 1024 * 1024, "MiB"
        ),
        format_summary(
            "Value cache", [r["value_cache_bytes"] for r in selected], 1024 * 1024, "MiB"
        ),
        format_summary(
            "Trie cache", [r["trie_cache_bytes"] for r in selected], 1024 * 1024, "MiB"
        ),
        "",
    ])
    lines.extend(retained_generation_lines(selected))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument(
        "--warmup",
        type=int,
        required=True,
        help="builder records excluded after Ready; use the value selected by the runner",
    )
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--require-published", action="store_true")
    parser.add_argument(
        "--expect-snapshot",
        choices=("created", "skipped", "any"),
        default="any",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    expected = {"created": True, "skipped": False, "any": None}[args.expect_snapshot]
    try:
        report = build_builder_report(
            load_jsonl(args.records),
            args.warmup,
            args.samples,
            expected,
            args.require_published,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(report, end="")
    if args.output:
        args.output.write_text(report)


if __name__ == "__main__":
    main()
