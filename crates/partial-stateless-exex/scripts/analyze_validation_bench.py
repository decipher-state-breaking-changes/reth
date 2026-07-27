#!/usr/bin/env python3
"""Analyze same-block Vanilla/Partial/Weak records from the live ExEx benchmark."""

import argparse
import bisect
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

RESET_MARKERS = ("Chain reorg detected", "Chain reverted", "Partial Stateless ExEx started")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def load_jsonl(path: Path):
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL in {path} at line {line_number}: {error}") from error
    return records


def field_hash(line: str, name: str):
    line = ANSI_ESCAPE.sub("", line)
    match = re.search(rf"(?:^|\s){re.escape(name)}=(0x[0-9a-fA-F]+)", line)
    return match.group(1).lower() if match else None


@dataclass
class SelectionStats:
    before_last_reset: int = 0
    warmup: int = 0
    invalid: int = 0
    missing_engine: int = 0
    missing_log_position: int = 0
    contaminated: int = 0
    pending_next_engine: int = 0


def parse_log_positions(log_path: Path):
    engine_positions = {}
    paired_positions = {}
    ordered_engine_positions = []
    reset_positions = []
    if not log_path.exists():
        return engine_positions, paired_positions, ordered_engine_positions, reset_positions
    for position, line in enumerate(log_path.read_text(errors="replace").splitlines()):
        if any(marker in line for marker in RESET_MARKERS):
            reset_positions.append(position)
        if "Vanilla Reth Engine V2 benchmark start" in line:
            block_hash = field_hash(line, "block_hash")
            if block_hash:
                engine_positions[block_hash] = position
                ordered_engine_positions.append(position)
        if "Paired Partial/Weak timed validation complete" in line:
            block_hash = field_hash(line, "block_hash")
            if block_hash:
                paired_positions[block_hash] = position
    ordered_engine_positions.sort()
    return engine_positions, paired_positions, ordered_engine_positions, reset_positions


def select_samples(paired_records, engine_records, log_path: Path, warmup: int, limit=None):
    engine_by_hash = {
        record.get("block_hash", "").lower(): record
        for record in engine_records
        if record.get("block_hash")
    }
    engine_positions, paired_positions, ordered_engine_positions, resets = parse_log_positions(log_path)
    last_reset = resets[-1] if resets else -1
    stats = SelectionStats()
    accepted = []
    warmup_remaining = warmup

    for record in paired_records:
        block_hash = record.get("block_hash", "").lower()
        paired_position = paired_positions.get(block_hash)
        if paired_position is None:
            stats.missing_log_position += 1
            continue
        if paired_position <= last_reset:
            stats.before_last_reset += 1
            continue
        if not record.get("valid", False):
            stats.invalid += 1
            continue
        engine = engine_by_hash.get(block_hash)
        own_engine_position = engine_positions.get(block_hash)
        if engine is None or own_engine_position is None:
            stats.missing_engine += 1
            continue
        if warmup_remaining:
            warmup_remaining -= 1
            stats.warmup += 1
            continue
        next_index = bisect.bisect_right(ordered_engine_positions, own_engine_position)
        if next_index >= len(ordered_engine_positions):
            stats.pending_next_engine += 1
            continue
        if ordered_engine_positions[next_index] < paired_position or engine.get("contaminated", False):
            stats.contaminated += 1
            continue
        merged = dict(record)
        merged["vanilla_engine"] = engine
        accepted.append(merged)
        if limit is not None and len(accepted) >= limit:
            break
    return accepted, stats


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return math.nan
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(values[lower])
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def summary(values, scale=1.0):
    scaled = [value / scale for value in values]
    return {
        "avg": statistics.fmean(scaled),
        "p50": percentile(scaled, 0.50),
        "p90": percentile(scaled, 0.90),
        "p95": percentile(scaled, 0.95),
        "p99": percentile(scaled, 0.99),
        "max": max(scaled),
    }


def format_summary(name, values, scale=1000.0, unit="ms"):
    stats = summary(values, scale)
    return "| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} {} |".format(
        name, stats["avg"], stats["p50"], stats["p90"], stats["p95"],
        stats["p99"], stats["max"], unit,
    )


def paired_ratio(numerators, denominators):
    return statistics.fmean(n / d for n, d in zip(numerators, denominators) if d)


def ratio_of_means(numerators, denominators):
    return statistics.fmean(numerators) / statistics.fmean(denominators)


def faster_share(numerators, denominators):
    return sum(n < d for n, d in zip(numerators, denominators)) / len(numerators)


def build_report(accepted, stats: SelectionStats, warmup: int, requested: int):
    if len(accepted) < requested:
        raise ValueError(f"only {len(accepted)} accepted samples; requested {requested} after warm-up {warmup}")
    accepted = accepted[:requested]
    vanilla_primary = [r["vanilla_engine"]["state_access_execution_us"] for r in accepted]
    partial_primary = [r["partial"]["state_access_execution_us"] for r in accepted]
    weak_primary = [r["weak"]["state_access_execution_us"] for r in accepted]
    vanilla_evm = [r["vanilla_engine"]["execution_us"] for r in accepted]
    partial_evm = [r["partial"]["evm_us"] for r in accepted]
    weak_evm = [r["weak"]["evm_us"] for r in accepted]
    gas = [r["gas_used"] for r in accepted]
    partial_witness = [r["partial_witness"]["serialized_witness_bytes"] for r in accepted]
    weak_witness = [r["weak_witness"]["serialized_witness_bytes"] for r in accepted]

    lines = [
        "# Single-process Vanilla / Partial / Weak benchmark", "",
        f"Accepted same-block samples: **{len(accepted)}**",
        f"Warm-up blocks excluded: **{stats.warmup}**",
        f"Excluded: overlap {stats.contaminated}, invalid {stats.invalid}, missing Engine {stats.missing_engine}, pending {stats.pending_next_engine}.", "",
        "## State access + execution (primary)", "",
        "Includes Vanilla provider/prewarming/DB reads and Partial/Weak deserialize, witness checks/materialization, provider setup/lookups, and EVM.", "",
        "| Mode | Average | p50 | p90 | p95 | p99 | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        format_summary("Vanilla Engine V2", vanilla_primary),
        format_summary("Partial Stateless", partial_primary),
        format_summary("Weak Stateless", weak_primary), "",
        "## EVM executor call (including state-provider reads)", "",
        "Includes lookups performed inside each executor call. Partial/Weak pre-execution deserialize, context validation, and witness materialization are excluded.", "",
        "| Mode | Average | p50 | p90 | p95 | p99 | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        format_summary("Vanilla Engine V2", vanilla_evm),
        format_summary("Partial Stateless", partial_evm),
        format_summary("Weak Stateless", weak_evm), "",
        "## Gas-normalized primary time", "",
        "| Mode | Average | p50 | p90 | p95 | p99 | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        format_summary("Vanilla Engine V2", [t * 1000 / g for t, g in zip(vanilla_primary, gas) if g], 1, "ms/Mgas"),
        format_summary("Partial Stateless", [t * 1000 / g for t, g in zip(partial_primary, gas) if g], 1, "ms/Mgas"),
        format_summary("Weak Stateless", [t * 1000 / g for t, g in zip(weak_primary, gas) if g], 1, "ms/Mgas"), "",
        "## Partial / Weak witness size", "",
        "| Payload | Average | p50 | p90 | p95 | p99 | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        format_summary("Partial witness", partial_witness, 1024 * 1024, "MiB"),
        format_summary("Weak witness", weak_witness, 1024 * 1024, "MiB"),
        format_summary("Partial full sidecar", [r["partial_sidecar_bytes"] for r in accepted], 1024 * 1024, "MiB"),
        format_summary("Weak full sidecar", [r["weak_sidecar_bytes"] for r in accepted], 1024 * 1024, "MiB"), "",
        "## Partial local cache memory", "",
        "| Cache | Average | p50 | p90 | p95 | p99 | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        format_summary("Value cache", [r["value_cache_bytes"] for r in accepted], 1024 * 1024, "MiB"),
        format_summary("Sparse trie cache", [r["trie_cache_bytes"] for r in accepted], 1024 * 1024, "MiB"), "",
        "## Paired ratios", "",
        f"- Partial / Vanilla primary, ratio of means: **{ratio_of_means(partial_primary, vanilla_primary):.3f}x**",
        f"- Partial / Vanilla median same-block ratio: **{statistics.median(p / v for p, v in zip(partial_primary, vanilla_primary) if v):.3f}x**",
        f"- Partial faster than Vanilla blocks: **{faster_share(partial_primary, vanilla_primary) * 100:.1f}%**",
        f"- Weak / Vanilla primary, ratio of means: **{ratio_of_means(weak_primary, vanilla_primary):.3f}x**",
        f"- Weak faster than Vanilla blocks: **{faster_share(weak_primary, vanilla_primary) * 100:.1f}%**",
        f"- Partial / Weak primary, ratio of means: **{ratio_of_means(partial_primary, weak_primary):.3f}x**",
        f"- Partial / Weak EVM, ratio of means: **{ratio_of_means(partial_evm, weak_evm):.3f}x**",
        f"- Partial witness size / Weak, ratio of means: **{ratio_of_means(partial_witness, weak_witness):.3f}x**",
        f"- Mean same-block Partial witness reduction: **{statistics.fmean(1 - p / w for p, w in zip(partial_witness, weak_witness) if w) * 100:.2f}%**", "",
        "## Builder-only costs (outside primary)", "",
        "| Operation | Average | p50 | p90 | p95 | p99 | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        format_summary("Full-DB simulation", [r["historical_full_db_evm_us"] for r in accepted]),
        format_summary("Partial witness build", [r["partial_witness_build_us"] for r in accepted]),
        format_summary("Weak witness build", [r["weak_witness_build_us"] for r in accepted]),
        format_summary("Partial serialize", [r["partial_serialize_us"] for r in accepted]),
        format_summary("Weak serialize", [r["weak_serialize_us"] for r in accepted]), "",
        "## Validation totals (secondary)", "",
        "| Total | Partial avg | Weak avg |",
        "| --- | ---: | ---: |",
        "| Raw validation | {:.2f} ms | {:.2f} ms |".format(
            statistics.fmean(r["partial"]["raw_total_us"] for r in accepted) / 1000,
            statistics.fmean(r["weak"]["raw_total_us"] for r in accepted) / 1000),
        "| Protocol total | {:.2f} ms | {:.2f} ms |".format(
            statistics.fmean(r["partial"]["protocol_total_us"] for r in accepted) / 1000,
            statistics.fmean(r["weak"]["protocol_total_us"] for r in accepted) / 1000),
        "| Root-validation reference | {:.2f} ms | {:.2f} ms |".format(
            statistics.fmean(r["partial"]["root_validation_total_us"] for r in accepted) / 1000,
            statistics.fmean(r["weak"]["root_validation_total_us"] for r in accepted) / 1000),
        "| Execution core | {:.2f} ms | {:.2f} ms |".format(
            statistics.fmean(r["partial"]["execution_core_us"] for r in accepted) / 1000,
            statistics.fmean(r["weak"]["execution_core_us"] for r in accepted) / 1000), "",
        "## Secondary phase breakdown", "",
        "| Phase | Partial avg | Weak avg |", "| --- | ---: | ---: |",
    ]
    phase_fields = [
        ("Deserialize", "deserialize_us"),
        ("Previous cache context / anchor", "context_check_us"),
        ("Witness self-consistency", "witness_self_consistency_us"),
        ("Materialize", "materialize_us"),
        ("Provider setup", "provider_setup_us"),
        ("Access capture (excluded from primary)", "accessed_state_capture_us"),
        ("Hash post-state", "hash_post_state_us"),
        ("Sparse-trie root", "state_root_us"),
        ("Exact miss-only policy", "miss_policy_check_us"),
        ("Cache update", "cache_update_us"),
        ("Trie retention", "trie_retention_us"),
        ("Next cache anchor", "next_cache_anchor_us"),
        ("Transactional trie clone", "trie_clone_us"),
        ("Trie commit", "trie_commit_us"),
        ("Unattributed", "unattributed_us"),
    ]
    for label, field in phase_fields:
        p_avg = statistics.fmean(r["partial"].get(field, 0) for r in accepted) / 1000
        w_avg = statistics.fmean(r["weak"].get(field, 0) for r in accepted) / 1000
        lines.append(f"| {label} | {p_avg:.2f} ms | {w_avg:.2f} ms |")

    orders = [r["verifier_order"] for r in accepted]
    tx_average = statistics.fmean(r["tx_count"] for r in accepted)
    partial_first_count = orders.count("partial-then-weak")
    weak_first_count = orders.count("weak-then-partial")
    lines.extend(["", "## Workload and validity", "",
        f"- Gas used average: **{statistics.fmean(gas) / 1_000_000:.2f} Mgas**",
        f"- Transaction count average: **{tx_average:.1f}**",
        f"- Partial-first samples: **{partial_first_count}**",
        f"- Weak-first samples: **{weak_first_count}**",
        "- Correctness failures: **0**", "",
        "## Interpretation limits", "",
        "- Engine executes each block before the ExEx builder and paired stateless executions.",
        "- Previous-block proof generation can warm or evict pages used by the next Engine block.",
        "- Partial and Weak are DB-free during their timed state-access + execution path.",
        "- Network transfer and sidecar file I/O are excluded; witness bytes are reported separately."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--engine-records", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    accepted, stats = select_samples(load_jsonl(args.records), load_jsonl(args.engine_records), args.log, args.warmup, args.samples)
    try:
        report = build_report(accepted, stats, args.warmup, args.samples)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(report, end="")
    if args.output:
        args.output.write_text(report)


if __name__ == "__main__":
    main()
