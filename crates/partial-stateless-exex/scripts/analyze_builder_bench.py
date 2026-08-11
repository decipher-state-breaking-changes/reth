#!/usr/bin/env python3
"""Analyze structured partial-stateless builder benchmark records."""

import argparse
import collections
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


def build_artifact_section(selected):
    """B3 artifact reuse, on records that carry it (schema 3 and later).

    Reports delivery and reuse as separate rates. They differ by the sampled blocks, which are
    delivered and deliberately re-executed to keep the differential oracle alive, so reading
    reuse as the delivery rate understates a healthy handoff by exactly the sampling fraction.

    The timing split is by `artifact_reused`, because `historical_full_db_evm_us` is zero on the
    reused blocks: a median over the mixed population describes neither path. Note that the
    comparison groups here are not equally trustworthy -- blocks that missed did so because the
    builder was behind, which correlates with heavy blocks, while sampled blocks are chosen by
    height and are unbiased. Only the sampled group is a fair contrast.
    """
    carrying = [r for r in selected if r.get("fallback_reason") is not None or r.get("artifact_reused")]
    if not carrying:
        return []

    available = sum(1 for r in carrying if r.get("artifact_available"))
    reused = [r for r in carrying if r.get("artifact_reused")]
    sampled = [r for r in carrying if r.get("fallback_reason") == "shadow_sampled"]
    reasons = collections.Counter(
        r["fallback_reason"] for r in carrying if r.get("fallback_reason")
    )

    lines = [
        "## Engine access artifact (B3)", "",
        f"- Artifact delivered: **{available}/{len(carrying)}** ({available / len(carrying):.2%})",
        f"- Artifact reused: **{len(reused)}/{len(carrying)}** ({len(reused) / len(carrying):.2%})",
    ]
    for reason, count in sorted(reasons.items()):
        lines.append(f"- Not reused, `{reason}`: **{count}**")
    if reused:
        evm_removed = sum(r["historical_full_db_evm_us"] for r in sampled) / len(sampled) if sampled else 0
        lines.extend([
            "",
            f"- Re-execution removed per reused block: **{evm_removed / 1000:.1f} ms** "
            "(mean over the sampled blocks, which still pay it)",
            f"- Total re-execution avoided: **{evm_removed * len(reused) / 1e6:.1f} s** "
            f"over {len(reused)} blocks",
        ])
    if reused and sampled:
        lines.extend([
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
    ]
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
