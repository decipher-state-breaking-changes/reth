#!/usr/bin/env python3
"""Distributions for standalone follow/replay JSONL records (schema v2), run-segmented.

Inputs are role-separated — a follow verdict stream and a batch replay record answer different
questions and are never mixed into one population:

  --follow PATH            live follower JSONL; repeatable, in timeline order (a killed run's
                           file first, its resumed run's file second)
  --batch PATH             batch replay JSONL; repeatable
  --producer-events PATH   the producer's out-of-band lifecycle log; reduced to the RB7 cause
                           ledger and its interval distributions (joined to --follow by frame
                           sequence, so pass both when the producer↔follower leg matters)
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

RB7 (recovery lifecycle) is reported over the *whole* detected-cause denominator: every cause the
producer opened appears in the ledger with how it ended — published, fenced by a later branch
change, out of retries, skipped by policy, or still pending when the log stopped — so a thin
interval sample reads as the coverage it is. Producer-internal intervals difference the monotonic
clock; the producer→follower leg is the only cross-clock one, is reported in its own unit, and a
negative reading there is dropped and counted rather than clamped to zero.

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


def pair_by_sequence(follow, batch):
    """Per-frame live/batch ratios, joined on the frame sequence.

    Never a ratio of two aggregates: the two runs read the same frames, so the comparison that
    means something is per frame, and a ratio of p50s would hide a live run that was faster on
    the cheap blocks and slower on the expensive ones. Heights are not the key — a reorg repeats
    one — and the two sides must agree on the block behind the sequence or the pair is dropped
    and counted.
    """
    batch_by_sequence = {b.get("sequence"): b for b in batch if b.get("sequence") is not None}
    ratios, deltas = [], []
    joined, mismatched = 0, 0
    for verdict in follow:
        other = batch_by_sequence.get(verdict.get("sequence"))
        if other is None:
            continue
        if verdict.get("block") is not None and other.get("block") is not None and (
            verdict["block"] != other["block"]
        ):
            mismatched += 1
            continue
        joined += 1
        live = verdict.get("standalone_validation_us")
        replay = other.get("standalone_validation_us")
        if live is None or not replay:
            continue
        ratios.append(live / replay)
        deltas.append(live - replay)
    return {
        "joined": joined,
        "follow_only": len(follow) - joined - mismatched,
        "batch_only": len(batch_by_sequence) - joined,
        "block_mismatched": mismatched,
        "standalone_validation_ratio": summarize(ratios, joined),
        "standalone_validation_delta_us": summarize(deltas, joined),
    }


def load_event_module():
    """The gate's own reducer, imported whether or not scripts/ is on the path."""
    try:
        import producer_event_state

        return producer_event_state
    except ImportError:
        import importlib.util
        import os

        spec = importlib.util.spec_from_file_location(
            "producer_event_state",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "producer_event_state.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def mono_interval(start, end, anomalies, name, cause_id):
    """A producer-internal interval, in microseconds off the monotonic clock.

    Refused rather than reported when the two stamps come from different process origins
    (`mono_run`) or when the difference is negative — the log is append-only from one thread,
    so a negative reading means the record is not what this interval assumes it is.
    """
    if start is None or end is None:
        return None
    if start.get("mono_us") is None or end.get("mono_us") is None:
        anomalies.append({"interval": name, "cause_id": cause_id, "why": "no monotonic stamp"})
        return None
    if start.get("mono_run") != end.get("mono_run"):
        anomalies.append({"interval": name, "cause_id": cause_id, "why": "monotonic clock reset"})
        return None
    delta = end["mono_us"] - start["mono_us"]
    if delta < 0:
        anomalies.append(
            {"interval": name, "cause_id": cause_id, "why": "negative", "value_us": delta}
        )
        return None
    return delta


def wall_interval(start, end_ms, anomalies, name, cause_id):
    """A cross-process interval, in milliseconds off two wall clocks.

    The producer and the follower stamp with their own `SystemTime`, so this reading carries
    whatever offset separates them; it is reported as its own metric, never mixed into the
    monotonic ones. A negative difference means the offset exceeded the interval — the sample
    is dropped and counted, because clamping it at zero would quietly report the offset as a
    latency of zero.
    """
    if start is None or end_ms is None or start.get("wall_ms") is None:
        return None
    delta = end_ms - start["wall_ms"]
    if delta < 0:
        anomalies.append(
            {"interval": name, "cause_id": cause_id, "why": "negative cross-clock",
             "value_ms": delta}
        )
        return None
    return delta


def follower_arrivals(verdicts):
    """`{sequence: earliest observed_at_ms}` — when the follower first decided that frame.

    Earliest, not latest: a resumed run re-publishing the same sequence is re-derivation, and
    the RB7 endpoint is the first decision the frame ever received.
    """
    arrivals = {}
    for verdict in verdicts:
        sequence = verdict.get("sequence")
        observed = verdict.get("observed_at_ms")
        if sequence is None or observed is None:
            continue
        if sequence not in arrivals or observed < arrivals[sequence]:
            arrivals[sequence] = observed
    return arrivals


def rb7_report(path, arrivals, warnings):
    """RB7: the recovery lifecycle as intervals, over the full detected-cause denominator.

    Every cause the producer opened is accounted for — published, fenced, out of retries,
    skipped by policy, or still pending at the log's end — so an interval's sample count is
    read against what actually happened rather than against itself. `checkpoint_published` is
    the availability endpoint: the recorder emits it after the announce *and* every chunk is on
    disk, which is the first moment a consumer could restore from it.
    """
    module = load_event_module()
    try:
        with open(path, encoding="utf-8") as handle:
            epoch, causes = module.assemble_causes(handle)
    except (OSError, json.JSONDecodeError) as err:
        warnings.append(f"{path}: producer events unreadable: {err}")
        return None
    if not causes:
        warnings.append(f"{path}: no producer causes in the log")
        return {
            "epoch": epoch,
            "causes": {"total": 0},
            "intervals": {},
            "per_cause": [],
            "anomalies": {
                "rejected": [],
                "unjoined_first_winning": [],
                "follower_join": "unavailable" if arrivals is None else "joined",
            },
        }

    anomalies = []
    unjoined = []
    samples = defaultdict(list)
    rows = []
    attempts_started = 0
    reachable = defaultdict(int)

    for cause_id, cause in sorted(causes.items()):
        # The origin is the cause's own lifecycle event: a detection for a branch change, the
        # arming line for a discontinuity, which has none. The stream-opening export has
        # neither — it was not caused by anything — so it carries no detection-anchored
        # interval, and the denominator says so rather than contributing a zero.
        origin = cause["detection"] or cause["armed"]
        first_start = cause["attempts"][0]["started"] if cause["attempts"] else None
        row = {
            "cause_id": cause_id,
            "origin_kind": cause["origin_kind"],
            "state": cause["state"],
            "attempts": len(cause["attempts"]),
            "intervals": {},
        }

        def record(name, value, unit="us", row=row):
            if value is not None:
                samples[(name, unit)].append(value)
                row["intervals"][name] = value

        if origin is not None:
            reachable["detection_to_export_start_us"] += 1
            reachable["detection_to_publication_us"] += 1
            reachable["detection_to_first_winning_us"] += 1
        record("detection_to_export_start_us",
               mono_interval(origin, first_start, anomalies,
                             "detection_to_export_start_us", cause_id))
        for attempt in cause["attempts"]:
            if attempt["started"] is None:
                continue
            attempts_started += 1
            duration = mono_interval(attempt["started"], attempt["completed"], anomalies,
                                     "export_duration_us", cause_id)
            if duration is not None:
                samples[("export_duration_us", "us")].append(duration)
            reported = (attempt["completed"] or {}).get("export_us")
            if reported is not None:
                samples[("export_reported_us", "us")].append(reported)
        last_completed = next(
            (a["completed"] for a in reversed(cause["attempts"]) if a["completed"]), None
        )
        if last_completed is not None:
            reachable["export_complete_to_publication_us"] += 1
        if cause["published"] is not None:
            reachable["checkpoint_announce_to_complete_us"] += 1
        if cause["first_winning"] is not None:
            reachable["first_winning_to_follower_verdict_ms"] += 1
        record("export_complete_to_publication_us",
               mono_interval(last_completed, cause["published"], anomalies,
                             "export_complete_to_publication_us", cause_id))
        record("detection_to_publication_us",
               mono_interval(origin, cause["published"], anomalies,
                             "detection_to_publication_us", cause_id))
        if cause["published"] is not None:
            announce = cause["published"].get("announce_to_complete_us")
            if announce is not None:
                samples[("checkpoint_announce_to_complete_us", "us")].append(announce)
                row["intervals"]["checkpoint_announce_to_complete_us"] = announce
        record("detection_to_first_winning_us",
               mono_interval(origin, cause["first_winning"], anomalies,
                             "detection_to_first_winning_us", cause_id))

        winning = cause["first_winning"]
        if winning is not None:
            sequence = winning.get("sequence")
            arrival = arrivals.get(sequence) if arrivals is not None else None
            if arrivals is None:
                pass
            elif arrival is None:
                unjoined.append({"cause_id": cause_id, "sequence": sequence})
            else:
                record("first_winning_to_follower_verdict_ms",
                       wall_interval(winning, arrival, anomalies,
                                     "first_winning_to_follower_verdict_ms", cause_id),
                       unit="ms")
            row["first_winning_sequence"] = sequence
        if cause["first_winning_unmeasured"] is not None:
            row["first_winning_unmeasured"] = cause["first_winning_unmeasured"].get("reason")
        rows.append(row)

    by_origin = defaultdict(int)
    by_state = defaultdict(int)
    winning_outcomes = defaultdict(int)
    for cause in causes.values():
        by_origin[cause["origin_kind"]] += 1
        by_state[cause["state"]] += 1
        if cause["first_winning"] is not None:
            winning_outcomes["published"] += 1
        elif cause["first_winning_unmeasured"] is not None:
            winning_outcomes[
                "unmeasured:" + str(cause["first_winning_unmeasured"].get("reason"))
            ] += 1
        else:
            winning_outcomes["none"] += 1

    total = len(causes)
    reachable["export_duration_us"] = attempts_started
    reachable["export_reported_us"] = attempts_started
    intervals = {}
    for (name, unit), values in sorted(samples.items()):
        # The population is what this interval *could* have measured — causes with an origin,
        # attempts that started, publications that happened — never the sample count itself.
        stats = summarize(values, reachable.get(name, len(values)))
        if stats:
            stats["unit"] = unit
            intervals[name] = stats

    return {
        "epoch": epoch,
        "causes": {
            "total": total,
            "by_origin": dict(sorted(by_origin.items())),
            "by_state": dict(sorted(by_state.items())),
            "attempts_started": attempts_started,
            "origin_anchored": reachable["detection_to_export_start_us"],
            "first_winning": dict(sorted(winning_outcomes.items())),
        },
        "intervals": intervals,
        "per_cause": rows,
        "anomalies": {
            "rejected": anomalies,
            "unjoined_first_winning": unjoined,
            "follower_join": "unavailable" if arrivals is None else "joined",
        },
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
    paired = result.get("paired")
    if paired:
        print(
            f"\n== live/batch paired by sequence ({paired['joined']} joined, "
            f"{paired['follow_only']} follow-only, {paired['batch_only']} batch-only, "
            f"{paired['block_mismatched']} block-mismatched) =="
        )
        for metric in ("standalone_validation_ratio", "standalone_validation_delta_us"):
            row = paired.get(metric)
            if row:
                print(
                    f"  {metric:44s} n={row['count']:6d}/{row['population']:<6d} "
                    f"mean={row['mean']:12.3f}  p50={row['p50']:10.3f}  "
                    f"p95={row['p95']:10.3f}  p99={row['p99']:10.3f}"
                )
    rb7 = result.get("producer_events")
    if rb7:
        ledger = rb7["causes"]
        print(f"\n== rb7/causes (epoch {rb7['epoch']}, {ledger['total']} detected) ==")
        for axis in ("by_origin", "by_state", "first_winning"):
            if ledger.get(axis):
                inner = "  ".join(f"{k}={v}" for k, v in ledger[axis].items())
                print(f"  {axis:16s} {inner}")
        for metric, row in rb7["intervals"].items():
            print(
                f"  {metric:44s} n={row['count']:6d}/{row['population']:<6d} "
                f"mean={row['mean']:12.1f}  p50={row['p50']:10.1f}  "
                f"p95={row['p95']:10.1f}  max={row['max']:10.1f} {row['unit']}"
            )
        rejected = rb7["anomalies"]["rejected"]
        if rejected:
            print(f"  {len(rejected)} interval sample(s) refused: "
                  f"{sorted({a['why'] for a in rejected})}")
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

    follow_arrivals = None
    follow_unique, batch_rows = None, None
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
        follow_arrivals = follower_arrivals(verdicts)
        populations = split_populations(verdicts, args.startup_through_seq)
        follow_unique = populations["unique_frames"]
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
        batch_rows = verdicts
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

    if follow_unique is not None and batch_rows is not None:
        result["paired"] = pair_by_sequence(follow_unique, batch_rows)

    if args.producer_events:
        if follow_arrivals is None:
            warnings.append(
                "no --follow input: the RB7 producer→follower interval is not measurable"
            )
        result["producer_events"] = rb7_report(args.producer_events, follow_arrivals, warnings)

    if args.producer_manifest:
        # JSONL: a producer restart appends its own manifest rather than replacing the file. The
        # last line is the run that produced the frames this report is about; earlier ones are
        # counted so a reader can see the spool spans more than one producer process.
        try:
            with open(args.producer_manifest, encoding="utf-8") as handle:
                manifests = [json.loads(line) for line in handle if line.strip()]
            if not manifests:
                warnings.append(f"{args.producer_manifest}: empty provenance file")
            else:
                result["provenance"] = manifests[-1]
                if len(manifests) > 1:
                    warnings.append(
                        f"{args.producer_manifest}: {len(manifests)} producer runs recorded; "
                        f"the provenance block is the last one"
                    )
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
    self_check_rb7()
    print("self-check passed")


def self_check_rb7():
    """The RB7 ledger, against a log holding one of each way a cause can end."""
    import os
    import tempfile

    def event(kind, cause_id, mono, wall, attempt=0, **fields):
        return {"schema_version": 1, "benchmark": "partial_stateless_producer_events",
                "kind": kind, "epoch": 2, "attempt": attempt, "observed_at_ms": wall,
                "mono_elapsed_us": mono, "cause_id": cause_id, **fields}

    base = 1_700_000_000_000
    events = [
        # cause 0 — the stream-opening export: no detection, no arming, so no interval anchored
        # on one, but its export duration and chunk completion are still samples.
        event("export_started", 0, 1_000, base + 1, attempt=1, block=10, write_through=False),
        event("export_completed", 0, 5_000, base + 5, attempt=1, block=10, export_us=3_800),
        event("checkpoint_published", 0, 5_500, base + 6, attempt=1, block=10,
              announce_sequence=3, chunks=2, announce_to_complete_us=400),
        # cause 1 — a reorg that retries once, then publishes. Its winning commit publishes
        # *ahead* of the checkpoint: that ordering is what W1 exists to produce, and the two
        # intervals below are what proves it happened.
        event("reorg_detected", 1, 10_000, base + 10, winning_tip=41, abandoned_from=40),
        event("recheckpoint_armed", 1, 10_050, base + 10, cause="branch_change", armed=True),
        event("export_started", 1, 10_500, base + 11, attempt=2, block=40, write_through=True),
        event("first_winning_commit_published", 1, 10_800, base + 11, block=41, sequence=42),
        event("export_failed", 1, 11_000, base + 11, attempt=2, retries_left=1,
              why="promotion failed", stream_ended=False),
        event("export_started", 1, 12_000, base + 12, attempt=3, block=40, write_through=True),
        event("export_completed", 1, 15_000, base + 15, attempt=3, block=40, export_us=2_900),
        event("checkpoint_published", 1, 15_400, base + 15, attempt=3, block=40,
              announce_sequence=50, chunks=2, announce_to_complete_us=300),
        # cause 2 — a revert fenced by the reorg that followed it, its measurement closed out.
        event("revert_detected", 2, 20_000, base + 20, reverted_from=42),
        event("recheckpoint_armed", 2, 20_050, base + 20, cause="branch_change", armed=True),
        event("export_started", 2, 20_500, base + 20, attempt=4, block=41, write_through=True),
        event("first_winning_commit_unmeasured", 2, 21_000, base + 21,
              reason="superseded_by_branch_change"),
        event("export_fenced", 2, 21_000, base + 21, attempt=4, fenced_by_cause_id=3,
              why="the chain reorged under the export", elapsed_ms=1),
        # cause 3 — the reorg that fenced it, and which then ran out of retries.
        event("reorg_detected", 3, 21_000, base + 21, winning_tip=43),
        event("recheckpoint_armed", 3, 21_050, base + 21, cause="branch_change", armed=True),
        event("export_started", 3, 21_500, base + 21, attempt=5, block=41, write_through=True),
        event("export_failed", 3, 24_000, base + 24, attempt=5, retries_left=0,
              why="no retries left", stream_ended=False),
    ]
    # The follower's own record of the frame cause 1 freed, plus one it never saw.
    follow = [
        {"kind": "run_manifest", "benchmark": "standalone_follow_v1"},
        {"kind": "verdict", "sequence": 42, "block": 41, "block_hash": "0x29",
         "standalone_validation_us": 500, "catch_up": False, "tail_live": True,
         "observed_at_ms": base + 74},
        # A re-derivation of the same frame, later: the join must take the first decision.
        {"kind": "verdict", "sequence": 42, "block": 41, "block_hash": "0x29",
         "standalone_validation_us": 490, "catch_up": True, "observed_at_ms": base + 9_999},
        {"kind": "summary", "agreed": True},
    ]
    paths = []
    for records in (events, follow):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
            paths.append(handle.name)
    try:
        args = argparse.Namespace(
            follow=[paths[1]], batch=[], producer_events=paths[0], producer_manifest=None,
            run=None, all_runs=True, startup_through_seq=None, phases=False,
        )
        result = analyze(args)
    finally:
        for path in paths:
            os.unlink(path)

    rb7 = result["producer_events"]
    ledger = rb7["causes"]
    assert rb7["epoch"] == 2
    assert ledger["total"] == 4, "every cause the producer opened is in the denominator"
    assert ledger["by_origin"] == {"initial": 1, "reorg": 2, "revert": 1}
    assert ledger["by_state"] == {"failed_final": 1, "fenced": 1, "published": 2}
    assert ledger["attempts_started"] == 5, "the retry is its own attempt"
    assert ledger["origin_anchored"] == 3, "the stream-opening export was caused by nothing"
    assert ledger["first_winning"] == {
        "published": 1, "unmeasured:superseded_by_branch_change": 1, "none": 2
    }

    intervals = rb7["intervals"]
    start = intervals["detection_to_export_start_us"]
    assert start["count"] == 3 and start["population"] == 3
    assert start["p50"] == 500.0
    duration = intervals["export_duration_us"]
    assert duration["count"] == 2 and duration["population"] == 5, (
        "three attempts died without completing; the population says so"
    )
    assert sorted([duration["min"], duration["max"]]) == [3_000, 4_000]
    assert intervals["export_reported_us"]["count"] == 2
    assert intervals["export_complete_to_publication_us"]["population"] == 2
    assert intervals["detection_to_publication_us"]["count"] == 1
    assert intervals["detection_to_publication_us"]["p50"] == 5_400.0
    winning = intervals["detection_to_first_winning_us"]
    assert winning["count"] == 1 and winning["p50"] == 800.0
    assert winning["p50"] < intervals["detection_to_publication_us"]["p50"], (
        "the winning branch publishes ahead of its recovery checkpoint — that is W1"
    )
    join = intervals["first_winning_to_follower_verdict_ms"]
    assert join["unit"] == "ms" and join["count"] == 1 and join["population"] == 1
    assert join["p50"] == 63.0, "joined on frame sequence, to the follower's *first* decision"
    assert rb7["anomalies"]["rejected"] == []
    assert rb7["anomalies"]["unjoined_first_winning"] == []
    assert result["warnings"] == []


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
