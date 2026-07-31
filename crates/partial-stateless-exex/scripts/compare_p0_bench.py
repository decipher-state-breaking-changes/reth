#!/usr/bin/env python3
"""Compare two block-identical structured P0 builder benchmark runs."""

import argparse
import statistics
from pathlib import Path

from analyze_validation_bench import load_jsonl, percentile


METRICS = (
    "builder_total_us",
    "transition_witness_build_us",
    "initial_provider_us",
    "snapshot_us",
)


def eligible_by_hash(records):
    return {
        record["block_hash"].lower(): record
        for record in records
        if record.get("block_hash")
        and record.get("cache_parent_synced", False)
        and record.get("sidecar_constructed", False)
    }


def compare(control, candidate, warmup, samples, candidate_source=None):
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
        "# P0 block-identical builder comparison", "",
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
                statistics.fmean(candidate_values) / statistics.fmean(control_values)
                if statistics.fmean(control_values)
                else float("nan"),
                median_ratio,
                percentile(control_values, 0.95) / 1000,
                percentile(candidate_values, 0.95) / 1000,
            )
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--candidate-source")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = compare(
            load_jsonl(args.control),
            load_jsonl(args.candidate),
            args.warmup,
            args.samples,
            args.candidate_source,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(report, end="")
    if args.output:
        args.output.write_text(report)


if __name__ == "__main__":
    main()
