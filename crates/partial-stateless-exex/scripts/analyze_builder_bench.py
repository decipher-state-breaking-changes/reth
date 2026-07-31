#!/usr/bin/env python3
"""Analyze structured partial-stateless builder benchmark records."""

import argparse
import collections
import statistics
from pathlib import Path

from analyze_validation_bench import format_summary, load_jsonl


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
        f"Warm-up records excluded: **{warmup}**", "",
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
        "## Initial proof selection", "",
    ]
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
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=60)
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
