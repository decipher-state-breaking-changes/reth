#!/usr/bin/env python3
"""Compare isolated prepared-input validation passes, paired by identical block hashes.

Each repetition is independent. Repeated observations of one block are reduced to a median
before the moving-block bootstrap; repetitions are also reported separately, not pooled as
independent blocks. The wall boundary includes coordinated disk-undo enqueue and cacheless
cleanup, while background write completion is reported separately.
"""
import argparse
import json
import random
import statistics
from pathlib import Path

from analyze_frontier_arms import paired_ratio


def load_pass(directory):
    directory = Path(directory)
    config = json.loads((directory / "run.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    if not result.get("complete") or config.get("timing_eligible") is not True:
        raise ValueError(f"{directory}: incomplete or diagnostic pass")
    if config.get("schema_version") != 1 or not config.get("binary_keccak256"):
        raise ValueError(f"{directory}: missing schema or binary provenance")
    if config.get("timing_boundary") != "payload_decode_through_coordinated_commit":
        raise ValueError(f"{directory}: incompatible timing boundary")
    partial = config["arm"] != "weak"
    writer = result.get("writer")
    if partial and (not isinstance(writer, dict) or config.get("undo_layout") != "disk-frames"
                    or config.get("undo_recording") is not True or not 1 <= config["retention_depth"] <= 64):
        raise ValueError(f"{directory}: missing disk-undo profile or final writer counters")
    if not partial and (writer is not None or config.get("undo_layout") != "none" or config.get("retention_depth") != 0):
        raise ValueError(f"{directory}: incompatible cacheless profile")
    writer = writer or {}
    if any(writer.get(key, 0) for key in ("failed", "enqueue_failures", "telemetry_failures", "pending")):
        raise ValueError(f"{directory}: writer failure or undrained work")
    rows = [json.loads(line) for line in (directory / "validation.jsonl").read_text().splitlines() if line]
    if any(not row.get("valid") for row in rows):
        raise ValueError(f"{directory}: invalid verdict")
    measured = [row for row in rows if row["measured"]]
    if len(rows) != config["warmup"] + config["samples"] or any(
            row["measured"] != (i >= config["warmup"]) for i, row in enumerate(rows)):
        raise ValueError(f"{directory}: incomplete or reordered warm-up/sample population")
    if len(measured) != result["samples"] or len(measured) != config["samples"]:
        raise ValueError(f"{directory}: sample count mismatch")
    keys = [(row["block_number"], row["block_hash"]) for row in measured]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{directory}: repeated block")
    if any(row["resident_flat_undo_records"] != 0 for row in rows):
        raise ValueError(f"{directory}: accumulated resident flat undo")
    for row in rows:
        if not 0 <= row["verified_us"] <= row["block_step_us"]:
            raise ValueError(f"{directory}: invalid timing bounds")
        disk = row["commit"].get("disk") or {}
        if row.get("active_warm_shrink_blocks") != config.get("warm_shrink_blocks"):
            raise ValueError(f"{directory}: active warm-shrink policy disagrees with run manifest")
        if partial and row["commit"].get("disk") is None:
            raise ValueError(f"{directory}: missing per-block disk counters")
        if partial:
            commit = row["commit"]
            retained = commit["retained_depth"]
            if commit.get("completed_prior_depth", 0) > max(0, retained - 1):
                raise ValueError(f"{directory}: impossible completed prior depth")
            if retained and not isinstance(commit.get("current_undo_written"), bool):
                raise ValueError(f"{directory}: missing current undo completion state")
        if any(disk.get(key, 0) for key in ("failed", "enqueue_failures", "telemetry_failures")):
            raise ValueError(f"{directory}: writer failure during pass")
    if partial and writer.get("submitted") != writer.get("completed", 0) + writer.get("cancelled", 0):
        raise ValueError(f"{directory}: inconsistent writer completion counters")
    return config, result, measured


def analyze(baselines, candidates):
    if not baselines or len(baselines) != len(candidates):
        raise ValueError("supply equally many baseline and candidate repetitions")
    sides = [[load_pass(path) for path in paths] for paths in (baselines, candidates)]
    reference_keys = [(r["block_number"], r["block_hash"]) for r in sides[0][0][2]]
    reference = sides[0][0][0]
    for side in sides:
        first_config, first_result, _ = side[0]
        first_sidecar_bytes = [row["sidecar_bytes"] for row in side[0][2]]
        for config, result, rows in side:
            keys = [(r["block_number"], r["block_hash"]) for r in rows]
            if keys != reference_keys or result["measured_block_set_digest"] != sides[0][0][1]["measured_block_set_digest"]:
                raise ValueError("passes do not contain exactly the same ordered block set")
            for field in ("build_commit", "build_dirty", "binary_keccak256", "allocator", "interval_ms", "timing_boundary", "rayon_num_threads", "malloc_conf", "warmup", "trie_repr", "asm_keccak", "keccak_cache_global"):
                if config.get(field) != reference.get(field):
                    raise ValueError(f"passes disagree on {field}")
            for field in ("input_manifest_digest", "retention_depth", "undo_layout", "undo_recording", "warm_shrink_blocks", "arm"):
                if config.get(field) != first_config.get(field):
                    raise ValueError(f"repetitions disagree on {field}")
            if any(row["arm"] != config["arm"] for row in rows):
                raise ValueError("row arm disagrees with manifest")
            if [row["sidecar_bytes"] for row in rows] != first_sidecar_bytes:
                raise ValueError("repetitions disagree on per-block sidecar bytes")
    rng = random.Random(20260916)
    fields = ("block_step_us", "verified_us", "validation_core_us")
    combined = {}
    repetitions = []
    for left, right in zip(*sides):
        repetitions.append({field: paired_ratio([(a[field], b[field]) for a, b in zip(left[2], right[2])], rng)
                            for field in fields})
    medians = [{field: [statistics.median(run[2][i][field] for run in side)
                        for i in range(len(reference_keys))] for field in fields} for side in sides]
    for field in fields:
        combined[field] = paired_ratio(list(zip(medians[0][field], medians[1][field])), rng)
    network = {}
    for bandwidth in (100, 1000):
        deltas = [(medians[1]["block_step_us"][i] - medians[0]["block_step_us"][i]) / 1000 +
                  8 * (sides[1][0][2][i]["sidecar_bytes"] - sides[0][0][2][i]["sidecar_bytes"]) / (bandwidth * 1000)
                  for i in range(len(reference_keys))]
        network[str(bandwidth)] = {"mean_candidate_minus_baseline_ms": statistics.fmean(deltas),
                                    "candidate_faster_fraction": sum(x < 0 for x in deltas) / len(deltas)}
    return {"blocks": len(reference_keys), "repetitions": repetitions, "combined": combined,
            "network_model_mbit": network, "baseline": sides[0][0][0], "candidate": sides[1][0][0],
            "writer_final": [[run[1].get("writer") for run in side] for side in sides],
            "notes": ["file delivery and background writer completion excluded from block step",
                      "network model uses uncompressed sidecar bytes, not full transport bytes",
                      "within-block phases are nested; do not add percentile rows"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="append", required=True, type=Path)
    parser.add_argument("--candidate", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.baseline, args.candidate)
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(str(error)) from error
    output = json.dumps(result, indent=2) + "\n"
    print(output, end="")
    if args.output:
        args.output.write_text(output)


if __name__ == "__main__":
    main()
