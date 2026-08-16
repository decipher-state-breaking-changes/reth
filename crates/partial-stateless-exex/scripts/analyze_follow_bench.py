#!/usr/bin/env python3
"""Distributions for standalone follow/replay JSONL records (schema v2), run-segmented.

Inputs are role-separated — a follow verdict stream and a batch replay record answer different
questions and are never mixed into one population:

  --follow PATH            live follower JSONL; repeatable, in timeline order (a killed run's
                           file first, its resumed run's file second)
  --batch PATH             batch replay JSONL; repeatable
  --producer-events PATH   the producer's out-of-band lifecycle log; reduced to per-cause
                           states here (interval distributions land with the RB7 ingest)
  --producer-manifest PATH the producer's run-manifest JSON, copied into the report verbatim
                           as provenance

Every file is split into run segments at its `run_manifest` records; the *last* segment is
analyzed unless `--run N` or `--all-runs` says otherwise, and a file holding several runs warns
rather than silently pooling them.

Populations, reported separately because pooling misattributes real work:

  raw_work       every verdict in the selected segments — the work the process(es) did,
                 re-derivations and re-publications included
  unique_frames  one verdict per frame sequence, the latest winning: the evidence base. The
                 dedup key is the *sequence*, never (block, hash) — an epoch restart re-records
                 the same canonical block under a fresh sequence, and those are two distinct
                 verification events
  delivered      unique frames minus catch-up — first-time verification work
  canonical      delivered verdicts still standing at the end — the chain's cost
  abandoned      delivered verdicts for blocks the chain later left (reorg/revert branch work)
  catch_up       re-derivations on the way back to a previous run's watermark — real cost,
                 never latency evidence

Phase labels on delivered verdicts (`recovery_replay` outranks the tail flag; `catch_up` is
orthogonal and already excluded):

  recovery       recovery_replay=true — first verification replayed out of the spool inside a
                 rewind window
  steady         tail_live=true — consumed at the live tail
  startup        tail_live=false — backlog read before this run reached the tail (also anything
                 at or below --startup-through-seq, the escape hatch for pre-fix recordings)
  unclassified   records that carry no tail_live at all (legacy schema)

The primary latency table is the steady population only, keyed by `available_at_source` and
never pooled across sources or phases; a pooled, unlabeled p99 is deliberately not emitted.
Every metric row carries its population size beside the surviving sample count, so a null-heavy
field reads as what it is; null is not zero.

Abandonment is read from the follower's own lifecycle records first (they name the abandoned
blocks by number *and hash*, which is what catches a pure revert); the same-height supersession
rule remains as the fallback batch records need.

Percentiles interpolate linearly between order statistics — the same definition
analyze_validation_bench.py uses, so a number quoted from either analyzer means the same thing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict

TIMING_FIELDS = (
    "standalone_validation_us",
    "delivery_us",
    "admission_us",
    "transition_us",
    "oracle_compare_us",
    "unattributed_validation_us",
)
LATENCY_FIELDS = ("queue_wait_us", "decision_latency_us")
PHASE_PREFIX = "phases."
PHASE_LABELS = ("startup", "recovery", "steady", "unclassified")


def segment_runs(path):
    """Splits one JSONL file into run segments at its `run_manifest` records.

    Records before the first manifest form an implicit leading segment (legacy files), and a
    segment without a closing `summary` is a run that died mid-write — kept, and labelled.
    """
    segments = []
    current = None
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                print(f"{path}:{line_number}: unreadable line skipped: {err}", file=sys.stderr)
                continue
            kind = record.get("kind")
            if kind == "run_manifest":
                current = {"manifest": record, "records": [], "summary": None}
                segments.append(current)
                continue
            if current is None:
                current = {"manifest": None, "records": [], "summary": None}
                segments.append(current)
            if kind == "summary":
                current["summary"] = record
            elif isinstance(record.get("blocks"), list):
                # The batch replay's closing record: counters plus the per-block array.
                current["summary"] = record
            else:
                current["records"].append(record)
    return segments


def select_segments(path, segments, run_index, all_runs, warnings):
    if not segments:
        warnings.append(f"{path}: no records")
        return []
    if all_runs:
        return list(enumerate(segments))
    if run_index is not None:
        if run_index >= len(segments):
            raise SystemExit(
                f"error: {path} holds {len(segments)} run segment(s); --run {run_index} names none"
            )
        return [(run_index, segments[run_index])]
    if len(segments) > 1:
        warnings.append(
            f"{path}: {len(segments)} run segments; analyzing the last — pass --run N or "
            f"--all-runs to choose explicitly"
        )
    return [(len(segments) - 1, segments[-1])]


def follow_verdicts(selected):
    """Verdicts from selected follow segments, lifecycle abandonment applied in record order."""
    verdicts = []
    latest_by_id = {}
    for _, segment in selected:
        for record in segment["records"]:
            kind = record.get("kind")
            if kind == "verdict":
                verdicts.append(record)
                latest_by_id[(record.get("block"), record.get("block_hash"))] = record
            elif kind == "lifecycle":
                numbers = record.get("abandoned") or []
                hashes = record.get("abandoned_hashes") or []
                for number, block_hash in zip(numbers, hashes):
                    gone = latest_by_id.pop((number, block_hash), None)
                    if gone is not None:
                        gone["abandoned_by_lifecycle"] = True
    return verdicts


def batch_verdicts(selected):
    verdicts = []
    for _, segment in selected:
        summary = segment["summary"] or {}
        for block in summary.get("blocks") or []:
            block = dict(block)
            block.setdefault("block", block.get("number"))
            block.setdefault("catch_up", False)
            verdicts.append(block)
    return verdicts


def dedup_by_sequence(verdicts):
    """One verdict per frame sequence, the latest record in input order winning.

    Collapses a resumed run's all-or-nothing re-publications onto the surviving record. A
    verdict with no sequence at all (defensive; the schema always writes one) stays as is.
    """
    latest = {}
    orphans = []
    for verdict in verdicts:
        sequence = verdict.get("sequence")
        if sequence is None:
            orphans.append(verdict)
        else:
            latest[sequence] = verdict
    unique = sorted(latest.values(), key=lambda v: v.get("sequence", 0))
    return unique + orphans


def label_phase(verdict, startup_through):
    if verdict.get("recovery_replay"):
        return "recovery"
    startup = startup_through is not None and (verdict.get("sequence") or 0) <= startup_through
    tail_live = verdict.get("tail_live")
    if startup:
        return "startup"
    if tail_live is True:
        return "steady"
    if tail_live is False:
        return "startup"
    return "unclassified"


def split_populations(verdicts, startup_through=None):
    """The population map: raw work in, labelled evidence out."""
    raw_work = sorted(verdicts, key=lambda v: v.get("sequence", 0))
    unique = dedup_by_sequence(verdicts)
    catch_up = [v for v in unique if v.get("catch_up")]
    delivered = [v for v in unique if not v.get("catch_up")]
    last_at_height = {}
    for verdict in delivered:
        height = verdict.get("block")
        if height is not None:
            last_at_height[height] = (verdict.get("sequence"), verdict.get("block_hash"))
    canonical, abandoned = [], []
    for verdict in delivered:
        height = verdict.get("block")
        last = last_at_height.get(height) if height is not None else None
        # A later verdict at the same height supersedes this one only when it stands for a
        # *different* block: an epoch restart re-records the same (block, hash) under a fresh
        # sequence, and that is re-verification of a block still standing, not a branch
        # replacement. The exemption needs both hashes present — batch block rows carry none,
        # and for them the height-only fallback stays exactly what it was.
        same_hash = (
            last is not None
            and last[1] is not None
            and verdict.get("block_hash") is not None
            and last[1] == verdict.get("block_hash")
        )
        superseded = last is not None and last[0] != verdict.get("sequence") and not same_hash
        if verdict.get("abandoned_by_lifecycle") or superseded:
            abandoned.append(verdict)
        else:
            canonical.append(verdict)
    phases = {label: [] for label in PHASE_LABELS}
    for verdict in delivered:
        phases[label_phase(verdict, startup_through)].append(verdict)
    return {
        "raw_work": raw_work,
        "unique_frames": unique,
        "delivered": delivered,
        "canonical": canonical,
        "abandoned": abandoned,
        "catch_up": catch_up,
        **{f"phase:{label}": population for label, population in phases.items()},
    }


def percentile(values, fraction):
    """Linear interpolation between order statistics — analyze_validation_bench.py's definition."""
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
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def summarize(values, population_size):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return {
        "count": len(values),
        "population": population_size,
        "nulls": population_size - len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": values[0],
        "max": values[-1],
    }


def collect_metrics(verdicts, include_phases):
    size = len(verdicts)
    metrics = {}
    for field in TIMING_FIELDS:
        metrics[field] = summarize((v.get(field) for v in verdicts), size)
    # Latency readings are grouped by the clock they came from and never pooled across clocks.
    by_source = defaultdict(list)
    for verdict in verdicts:
        source = verdict.get("available_at_source")
        if source:
            by_source[source].append(verdict)
    for source, group in sorted(by_source.items()):
        for field in LATENCY_FIELDS:
            metrics[f"{field}[{source}]"] = summarize((v.get(field) for v in group), size)
    if include_phases:
        names = set()
        for verdict in verdicts:
            names.update((verdict.get("phases") or {}).keys())
        for name in sorted(names):
            metrics[PHASE_PREFIX + name] = summarize(
                ((v.get("phases") or {}).get(name) for v in verdicts), size
            )
    return {name: stats for name, stats in metrics.items() if stats}


def report_populations(populations, include_phases):
    out = {}
    for name, verdicts in populations.items():
        out[name] = {
            "verdicts": len(verdicts),
            "live_tail_verdicts": sum(1 for v in verdicts if v.get("tail_live")),
            "metrics": collect_metrics(verdicts, include_phases),
        }
    return out


def reduce_producer_events(path, warnings):
    """Per-cause lifecycle states via the gate's own reducer; RB7 intervals are its successor."""
    try:
        from producer_event_state import reduce_events
    except ImportError:
        import importlib.util
        import os

        spec = importlib.util.spec_from_file_location(
            "producer_event_state",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "producer_event_state.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reduce_events = module.reduce_events
    try:
        with open(path, encoding="utf-8") as handle:
            epoch, causes = reduce_events(handle)
    except (OSError, json.JSONDecodeError) as err:
        warnings.append(f"{path}: producer events unreadable: {err}")
        return None
    return {
        "epoch": epoch,
        "causes": {str(cause): state for cause, state in sorted(causes.items())},
        "note": "interval distributions land with the RB7 ingest",
    }


def print_human(result):
    for section in ("follow", "batch"):
        body = result.get(section)
        if not body:
            continue
        for population, stats in body["populations"].items():
            print(
                f"\n== {section}/{population} ({stats['verdicts']} verdicts, "
                f"{stats['live_tail_verdicts']} live-tail) =="
            )
            for metric, row in stats["metrics"].items():
                print(
                    f"  {metric:44s} n={row['count']:6d}/{row['population']:<6d} "
                    f"mean={row['mean']:12.1f}  p50={row['p50']:10.1f}  "
                    f"p95={row['p95']:10.1f}  p99={row['p99']:10.1f}"
                )
    for warning in result.get("warnings", []):
        print(f"warning: {warning}", file=sys.stderr)


def analyze(args):
    warnings = []
    result = {
        "schema_version": 1,
        "generated_by": "analyze_follow_bench.py",
        "inputs": {
            "follow": args.follow,
            "batch": args.batch,
            "producer_events": args.producer_events,
            "producer_manifest": args.producer_manifest,
        },
        "warnings": warnings,
    }

    if args.follow:
        selected = []
        runs = []
        for path in args.follow:
            for index, segment in select_segments(
                path, segment_runs(path), args.run, args.all_runs, warnings
            ):
                selected.append((index, segment))
                manifest = segment["manifest"] or {}
                runs.append(
                    {
                        "file": path,
                        "segment": index,
                        "label": manifest.get("label"),
                        "complete": segment["summary"] is not None,
                    }
                )
        verdicts = follow_verdicts(selected)
        populations = split_populations(verdicts, args.startup_through_seq)
        summaries = [s["summary"] for _, s in selected if s["summary"]]
        result["follow"] = {
            "runs": runs,
            "populations": report_populations(populations, args.phases),
            # The last selected run's own aggregate record, verbatim: the gate status source.
            "summary": summaries[-1] if summaries else None,
        }

    if args.batch:
        selected = []
        for path in args.batch:
            selected.extend(
                select_segments(path, segment_runs(path), args.run, args.all_runs, warnings)
            )
        verdicts = batch_verdicts(selected)
        populations = split_populations(verdicts)
        # Batch replays carry no tail and no resume, so the phase axis says nothing there;
        # dropping it beats printing four rows of one label.
        populations = {
            name: body for name, body in populations.items() if not name.startswith("phase:")
        }
        summaries = [s["summary"] for _, s in selected if s["summary"]]
        result["batch"] = {
            "populations": report_populations(populations, args.phases),
            "summary": summaries[-1] if summaries else None,
        }

    if args.producer_events:
        result["producer_events"] = reduce_producer_events(args.producer_events, warnings)

    if args.producer_manifest:
        try:
            with open(args.producer_manifest, encoding="utf-8") as handle:
                result["provenance"] = json.load(handle)
        except (OSError, json.JSONDecodeError) as err:
            warnings.append(f"{args.producer_manifest}: provenance unreadable: {err}")

    return result


def self_check():
    """The attribution rules, against a synthetic stream that exercises each of them."""
    import os
    import tempfile

    records = [
        {"kind": "run_manifest", "benchmark": "standalone_follow_v1", "label": "first"},
        # The killed run: two verdicts, then nothing — no summary.
        {"kind": "verdict", "sequence": 10, "block": 100, "block_hash": "0xaa",
         "standalone_validation_us": 100, "catch_up": False, "tail_live": True,
         "queue_wait_us": 7, "available_at_source": "mtime", "phases": {"evm_us": 40}},
        {"kind": "verdict", "sequence": 11, "block": 101, "block_hash": "0xb1",
         "standalone_validation_us": 200, "catch_up": False, "tail_live": False},
        {"kind": "run_manifest", "benchmark": "standalone_follow_v1", "label": "resumed"},
        # The resumed run re-publishes sequence 11 (all-or-nothing window replay) — the dedup
        # keeps this one — then meets a reorg, a rewind, and an epoch restart that re-records
        # height 101 under a fresh sequence, which dedup must NOT collapse.
        {"kind": "verdict", "sequence": 11, "block": 101, "block_hash": "0xb1",
         "standalone_validation_us": 210, "catch_up": False, "recovery_replay": True,
         "tail_live": False},
        {"kind": "lifecycle", "event": "reorg_applied", "common_ancestor": 100,
         "abandoned": [101], "abandoned_hashes": ["0xb1"]},
        {"kind": "verdict", "sequence": 13, "block": 101, "block_hash": "0xb2",
         "standalone_validation_us": 300, "catch_up": False, "tail_live": True},
        {"kind": "verdict", "sequence": 14, "block": 102, "block_hash": "0xcc",
         "standalone_validation_us": 400, "catch_up": False, "tail_live": False,
         "queue_wait_us": None, "available_at_source": None},
        {"kind": "lifecycle", "event": "revert_applied", "common_ancestor": 101,
         "abandoned": [102], "abandoned_hashes": ["0xcc"]},
        # Same block and hash as sequence 13, at a fresh sequence: an epoch restart's
        # re-recording. A (block, hash) dedup would wrongly collapse it into sequence 13.
        {"kind": "verdict", "sequence": 20, "block": 101, "block_hash": "0xb2",
         "standalone_validation_us": 310, "catch_up": False},
        {"kind": "verdict", "sequence": 5, "block": 99, "block_hash": "0x99",
         "standalone_validation_us": 50, "catch_up": True},
        {"kind": "summary", "agreed": True},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        path = handle.name
    try:
        args = argparse.Namespace(
            follow=[path], batch=[], producer_events=None, producer_manifest=None,
            run=None, all_runs=True, startup_through_seq=None, phases=True,
        )
        result = analyze(args)
    finally:
        os.unlink(path)

    populations = result["follow"]["populations"]
    assert populations["raw_work"]["verdicts"] == 7, "both publications of sequence 11 are work"
    assert populations["unique_frames"]["verdicts"] == 6, "sequence 11 collapses to one frame"
    assert populations["delivered"]["verdicts"] == 5
    assert populations["catch_up"]["verdicts"] == 1
    assert populations["canonical"]["verdicts"] == 3, "100, 101@0xb2 (later sequence), 101@20"
    assert populations["abandoned"]["verdicts"] == 2, "the reorged 0xb1 and the reverted 0xcc"
    assert populations["phase:recovery"]["verdicts"] == 1, "recovery_replay outranks tail_live"
    assert populations["phase:steady"]["verdicts"] == 2
    assert populations["phase:startup"]["verdicts"] == 1
    assert populations["phase:unclassified"]["verdicts"] == 1, "sequence 20 carries no tail_live"
    metrics = populations["delivered"]["metrics"]
    assert metrics["standalone_validation_us"]["count"] == 5
    assert metrics["standalone_validation_us"]["population"] == 5
    steady = populations["phase:steady"]["metrics"]
    assert "queue_wait_us[mtime]" not in steady or steady["queue_wait_us[mtime]"]["count"] <= 1
    raw = populations["raw_work"]["metrics"]["standalone_validation_us"]
    assert raw["count"] == 7 and raw["nulls"] == 0
    # The surviving sequence-11 record is the resumed run's (210, recovery_replay).
    assert metrics["standalone_validation_us"]["mean"] == (100 + 210 + 300 + 400 + 310) / 5
    assert result["warnings"] == []
    print("self-check passed")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--follow", action="append", default=[], metavar="PATH",
                        help="live follower JSONL, repeatable in timeline order")
    parser.add_argument("--batch", action="append", default=[], metavar="PATH",
                        help="batch replay JSONL, repeatable")
    parser.add_argument("--producer-events", metavar="PATH",
                        help="producer lifecycle JSONL, reduced to per-cause states")
    parser.add_argument("--producer-manifest", metavar="PATH",
                        help="producer run-manifest JSON, copied into the report as provenance")
    parser.add_argument("--run", type=int, metavar="N",
                        help="analyze run segment N (0-based) of every input file")
    parser.add_argument("--all-runs", action="store_true",
                        help="analyze every run segment, deduplicated by frame sequence")
    parser.add_argument("--startup-through-seq", type=int, metavar="N",
                        help="label verdicts at or below sequence N as startup (pre-fix data)")
    parser.add_argument("--phases", action="store_true", help="include per-phase distributions")
    parser.add_argument("--json", metavar="PATH", help="write the report as JSON")
    parser.add_argument("--self-check", action="store_true", help="run the built-in checks")
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return
    if not args.follow and not args.batch:
        parser.error("no inputs: pass --follow and/or --batch (or --self-check)")

    result = analyze(args)
    print_human(result)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
