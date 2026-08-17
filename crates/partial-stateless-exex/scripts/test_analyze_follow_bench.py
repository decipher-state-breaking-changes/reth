#!/usr/bin/env python3
"""Tests for the follow analyzer, its RB7 ingest, and the report renderer.

The analyzer's built-in `--self-check` is a smoke test the gate runs; this is the file that holds
the attribution rules to account one at a time, including the ones that only appear when a run
went wrong — a fenced export, a retry that ran out, a checkpoint that never published, two wall
clocks that disagree about which came first.

    python3 -m unittest test_analyze_follow_bench -v

Two of these tests are regressions against real archives rather than fixtures: the I.1 and I.2
preflights, whose distributions were originally derived by hand. They skip when the archive is not
on this host. The equivalent command by hand is

    ./analyze_follow_bench.py --follow <run>/out/follow.jsonl --batch <run>/out/batch.jsonl \
        --producer-manifest <run>/spool.run-manifest.jsonl --json /tmp/dist.json
    ./render_follow_report.py /tmp/dist.json --out /tmp/result.md
"""

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_follow_bench import analyze, pair_by_sequence, split_populations  # noqa: E402
from producer_event_state import assemble_causes, reduce_events  # noqa: E402
from render_follow_report import render  # noqa: E402

#: The Xeon preflight whose hand-derived p50 the analyzer has to reproduce exactly.
I1_ARCHIVE = Path("/data/bench-runs/s5-preflight-1000-20260815-101319")
I1_EXPECTED_P50_US = 497_586.0
#: The i7/SATA preflight, same shape. Present on the second host only.
I2_ARCHIVE = Path("/data/bench-runs/s5-preflight-1000-i7")


def verdict(sequence, block, block_hash="0xaa", **fields):
    record = {
        "kind": "verdict",
        "sequence": sequence,
        "block": block,
        "block_hash": block_hash,
        "standalone_validation_us": 1_000 + sequence,
        "catch_up": False,
    }
    record.update(fields)
    return record


def event(kind, cause_id, mono, wall, attempt=0, epoch=1, **fields):
    return {
        "schema_version": 1,
        "benchmark": "partial_stateless_producer_events",
        "kind": kind,
        "epoch": epoch,
        "attempt": attempt,
        "observed_at_ms": wall,
        "mono_elapsed_us": mono,
        "cause_id": cause_id,
        **fields,
    }


def write_jsonl(records):
    handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for record in records:
        handle.write(json.dumps(record) + "\n")
    handle.close()
    return handle.name


class AnalyzerCase(unittest.TestCase):
    """Runs the analyzer over temporary JSONL files and cleans them up."""

    def setUp(self):
        self.paths = []

    def tearDown(self):
        for path in self.paths:
            os.unlink(path)

    def file(self, records):
        path = write_jsonl(records)
        self.paths.append(path)
        return path

    def run_analyzer(self, follow=(), batch=(), producer_events=None, **overrides):
        args = argparse.Namespace(
            follow=list(follow),
            batch=list(batch),
            producer_events=producer_events,
            producer_manifest=None,
            run=None,
            all_runs=False,
            startup_through_seq=None,
            phases=False,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return analyze(args)


class RunSegmentationTest(AnalyzerCase):
    def records_of_two_runs(self):
        return [
            {"kind": "run_manifest", "label": "first"},
            verdict(1, 100, tail_live=True),
            verdict(2, 101, tail_live=True),
            {"kind": "summary", "agreed": True, "blocks_verified": 2},
            {"kind": "run_manifest", "label": "second"},
            verdict(3, 102, tail_live=True),
            {"kind": "summary", "agreed": True, "blocks_verified": 1},
        ]

    def test_the_last_segment_is_the_default_and_the_pooling_is_announced(self):
        result = self.run_analyzer(follow=[self.file(self.records_of_two_runs())])
        self.assertEqual(result["follow"]["populations"]["delivered"]["verdicts"], 1)
        self.assertEqual(result["follow"]["runs"][0]["label"], "second")
        self.assertTrue(any("2 run segments" in w for w in result["warnings"]))

    def test_an_explicit_run_index_selects_the_earlier_one_without_warning(self):
        result = self.run_analyzer(follow=[self.file(self.records_of_two_runs())], run=0)
        self.assertEqual(result["follow"]["populations"]["delivered"]["verdicts"], 2)
        self.assertEqual(result["warnings"], [])

    def test_all_runs_pools_them_and_the_summary_is_the_last_one(self):
        result = self.run_analyzer(follow=[self.file(self.records_of_two_runs())], all_runs=True)
        self.assertEqual(result["follow"]["populations"]["delivered"]["verdicts"], 3)
        self.assertEqual(result["follow"]["summary"]["blocks_verified"], 1)

    def test_a_run_that_died_mid_write_is_kept_and_labelled_incomplete(self):
        path = self.file([{"kind": "run_manifest", "label": "killed"}, verdict(1, 100)])
        result = self.run_analyzer(follow=[path])
        self.assertEqual(result["follow"]["runs"][0]["complete"], False)

    def test_two_files_in_timeline_order_are_one_analysis(self):
        killed = self.file([{"kind": "run_manifest", "label": "killed"}, verdict(1, 100)])
        resumed = self.file([
            {"kind": "run_manifest", "label": "resumed"},
            verdict(2, 101),
            {"kind": "summary", "agreed": True},
        ])
        result = self.run_analyzer(follow=[killed, resumed])
        self.assertEqual(result["follow"]["populations"]["delivered"]["verdicts"], 2)
        self.assertEqual([run["label"] for run in result["follow"]["runs"]], ["killed", "resumed"])


class DedupTest(AnalyzerCase):
    def test_a_republished_sequence_is_work_once_and_evidence_once(self):
        # The resumed run replays its whole rewind window, so the same frame is published twice.
        path = self.file([
            {"kind": "run_manifest"},
            verdict(7, 100, standalone_validation_us=500),
            verdict(7, 100, standalone_validation_us=600, recovery_replay=True),
            {"kind": "summary"},
        ])
        populations = self.run_analyzer(follow=[path])["follow"]["populations"]
        self.assertEqual(populations["raw_work"]["verdicts"], 2)
        self.assertEqual(populations["unique_frames"]["verdicts"], 1)
        # The later record wins: it is the one the surviving run stands behind.
        self.assertEqual(
            populations["delivered"]["metrics"]["standalone_validation_us"]["p50"], 600.0
        )

    def test_one_block_recorded_under_two_sequences_stays_two_verifications(self):
        # An epoch restart re-records the same canonical block at a fresh sequence. A (block,
        # hash) dedup key would collapse these into one and lose half the work the run did.
        path = self.file([
            {"kind": "run_manifest"},
            verdict(7, 100, "0xbb"),
            verdict(50, 100, "0xbb"),
            {"kind": "summary"},
        ])
        populations = self.run_analyzer(follow=[path])["follow"]["populations"]
        self.assertEqual(populations["unique_frames"]["verdicts"], 2)
        self.assertEqual(populations["canonical"]["verdicts"], 2)
        self.assertEqual(populations["abandoned"]["verdicts"], 0)


class PhaseLabelTest(AnalyzerCase):
    def test_recovery_replay_outranks_the_tail_flag(self):
        path = self.file([
            {"kind": "run_manifest"},
            verdict(1, 100, recovery_replay=True, tail_live=True),
            {"kind": "summary"},
        ])
        populations = self.run_analyzer(follow=[path])["follow"]["populations"]
        self.assertEqual(populations["phase:recovery"]["verdicts"], 1)
        self.assertEqual(populations["phase:steady"]["verdicts"], 0)

    def test_catch_up_is_its_own_population_and_no_phase_at_all(self):
        path = self.file([
            {"kind": "run_manifest"},
            verdict(1, 100, catch_up=True, tail_live=True),
            verdict(2, 101, tail_live=True),
            {"kind": "summary"},
        ])
        populations = self.run_analyzer(follow=[path])["follow"]["populations"]
        self.assertEqual(populations["catch_up"]["verdicts"], 1)
        self.assertEqual(populations["delivered"]["verdicts"], 1)
        self.assertEqual(
            sum(populations[f"phase:{name}"]["verdicts"]
                for name in ("startup", "recovery", "steady", "unclassified")),
            1,
        )

    def test_a_recording_without_the_tail_flag_is_unclassified_not_steady(self):
        path = self.file([{"kind": "run_manifest"}, verdict(1, 100), {"kind": "summary"}])
        populations = self.run_analyzer(follow=[path])["follow"]["populations"]
        self.assertEqual(populations["phase:unclassified"]["verdicts"], 1)

    def test_the_startup_escape_hatch_relabels_below_a_sequence(self):
        path = self.file([
            {"kind": "run_manifest"},
            verdict(1, 100, tail_live=True),
            verdict(9, 101, tail_live=True),
            {"kind": "summary"},
        ])
        populations = self.run_analyzer(
            follow=[path], startup_through_seq=5
        )["follow"]["populations"]
        self.assertEqual(populations["phase:startup"]["verdicts"], 1)
        self.assertEqual(populations["phase:steady"]["verdicts"], 1)

    def test_a_pooled_unlabelled_percentile_is_not_emitted(self):
        path = self.file([
            {"kind": "run_manifest"},
            verdict(1, 100, tail_live=True, queue_wait_us=5, available_at_source="mtime"),
            {"kind": "summary"},
        ])
        metrics = self.run_analyzer(follow=[path])["follow"]["populations"]["delivered"]["metrics"]
        # Latency is only ever reported under the clock it came from.
        self.assertNotIn("queue_wait_us", metrics)
        self.assertIn("queue_wait_us[mtime]", metrics)


class LineageTest(AnalyzerCase):
    def test_the_followers_own_lifecycle_record_names_the_abandoned_block(self):
        path = self.file([
            {"kind": "run_manifest"},
            verdict(1, 100, "0xa1", tail_live=True),
            verdict(2, 101, "0xb1", tail_live=True),
            {"kind": "lifecycle", "event": "revert_applied", "abandoned": [101],
             "abandoned_hashes": ["0xb1"]},
            {"kind": "summary"},
        ])
        populations = self.run_analyzer(follow=[path])["follow"]["populations"]
        self.assertEqual(populations["abandoned"]["verdicts"], 1)
        self.assertEqual(populations["canonical"]["verdicts"], 1)

    def test_a_later_verdict_at_one_height_supersedes_only_a_different_block(self):
        # The batch fallback: no lifecycle records, so height order is all there is. A different
        # hash at the same height is a branch replacement; the same hash is a re-verification.
        replaced = split_populations([
            {"sequence": 1, "block": 100, "block_hash": "0xa1"},
            {"sequence": 2, "block": 100, "block_hash": "0xa2"},
        ])
        self.assertEqual(len(replaced["abandoned"]), 1)
        rerecorded = split_populations([
            {"sequence": 1, "block": 100, "block_hash": "0xa1"},
            {"sequence": 2, "block": 100, "block_hash": "0xa1"},
        ])
        self.assertEqual(len(rerecorded["abandoned"]), 0)

    def test_hashless_rows_keep_the_height_only_fallback(self):
        # Batch block rows carry no hash at all; the old rule has to survive for them.
        populations = split_populations([
            {"sequence": 1, "block": 100},
            {"sequence": 2, "block": 100},
        ])
        self.assertEqual(len(populations["abandoned"]), 1)


class PairingTest(unittest.TestCase):
    def test_frames_join_on_sequence_not_height(self):
        follow = [
            {"sequence": 1, "block": 100, "standalone_validation_us": 200},
            {"sequence": 2, "block": 100, "standalone_validation_us": 300},
        ]
        batch = [
            {"sequence": 1, "block": 100, "standalone_validation_us": 100},
            {"sequence": 2, "block": 100, "standalone_validation_us": 100},
        ]
        paired = pair_by_sequence(follow, batch)
        self.assertEqual(paired["joined"], 2)
        self.assertEqual(paired["standalone_validation_ratio"]["min"], 2.0)
        self.assertEqual(paired["standalone_validation_delta_us"]["max"], 200)

    def test_a_sequence_naming_two_different_blocks_is_refused_not_averaged(self):
        paired = pair_by_sequence(
            [{"sequence": 1, "block": 100, "standalone_validation_us": 200}],
            [{"sequence": 1, "block": 999, "standalone_validation_us": 100}],
        )
        self.assertEqual(paired["joined"], 0)
        self.assertEqual(paired["block_mismatched"], 1)
        self.assertIsNone(paired["standalone_validation_ratio"])

    def test_unjoined_frames_are_counted_on_both_sides(self):
        paired = pair_by_sequence(
            [{"sequence": 1, "block": 100, "standalone_validation_us": 200}],
            [{"sequence": 2, "block": 101, "standalone_validation_us": 100}],
        )
        self.assertEqual((paired["joined"], paired["follow_only"], paired["batch_only"]), (0, 1, 1))


class Rb7CauseLedgerTest(AnalyzerCase):
    def healthy_log(self):
        return [
            event("reorg_detected", 1, 1_000, 500, winning_tip=41),
            event("recheckpoint_armed", 1, 1_100, 500, cause="branch_change", armed=True),
            event("export_started", 1, 2_000, 501, attempt=1, block=40),
            event("first_winning_commit_published", 1, 3_000, 502, block=41, sequence=77),
            event("export_completed", 1, 9_000, 508, attempt=1, block=40, export_us=6_500),
            event("checkpoint_published", 1, 9_500, 509, attempt=1, block=40,
                  announce_sequence=80, chunks=2, announce_to_complete_us=450),
        ]

    def test_the_write_through_ordering_is_visible_as_two_intervals(self):
        result = self.run_analyzer(producer_events=self.file(self.healthy_log()))
        intervals = result["producer_events"]["intervals"]
        self.assertEqual(intervals["detection_to_first_winning_us"]["p50"], 2_000.0)
        self.assertEqual(intervals["detection_to_publication_us"]["p50"], 8_500.0)
        self.assertEqual(intervals["export_duration_us"]["p50"], 7_000.0)
        self.assertEqual(intervals["export_complete_to_publication_us"]["p50"], 500.0)
        self.assertEqual(intervals["checkpoint_announce_to_complete_us"]["p50"], 450.0)

    def test_a_retry_is_its_own_attempt_and_the_cause_publishes_once(self):
        log = [
            event("reorg_detected", 1, 1_000, 500),
            event("recheckpoint_armed", 1, 1_100, 500, cause="branch_change", armed=True),
            event("export_started", 1, 2_000, 501, attempt=1),
            event("export_failed", 1, 3_000, 502, attempt=1, retries_left=1, why="promote"),
            event("export_started", 1, 4_000, 503, attempt=2),
            event("export_completed", 1, 6_000, 505, attempt=2, export_us=1_900),
            event("checkpoint_published", 1, 6_500, 505, attempt=2, announce_to_complete_us=100),
        ]
        rb7 = self.run_analyzer(producer_events=self.file(log))["producer_events"]
        self.assertEqual(rb7["causes"]["attempts_started"], 2)
        self.assertEqual(rb7["causes"]["by_state"], {"published": 1})
        duration = rb7["intervals"]["export_duration_us"]
        self.assertEqual((duration["count"], duration["population"]), (1, 2))
        # The interval is measured from the *first* attempt: that is when the cause started
        # costing something, and a retry does not restart the clock a reorg started.
        self.assertEqual(rb7["intervals"]["detection_to_export_start_us"]["p50"], 1_000.0)

    def test_a_fenced_cause_and_a_final_failure_stay_in_the_denominator(self):
        log = [
            event("revert_detected", 1, 1_000, 500),
            event("recheckpoint_armed", 1, 1_100, 500, cause="branch_change", armed=True),
            event("export_started", 1, 2_000, 501, attempt=1),
            event("first_winning_commit_unmeasured", 1, 2_500, 501,
                  reason="superseded_by_branch_change"),
            event("export_fenced", 1, 2_500, 501, attempt=1, fenced_by_cause_id=2),
            event("reorg_detected", 2, 2_500, 501),
            event("recheckpoint_armed", 2, 2_600, 501, cause="branch_change", armed=True),
            event("export_started", 2, 3_000, 502, attempt=2),
            event("export_failed", 2, 8_000, 507, attempt=2, retries_left=0, why="out of retries"),
        ]
        rb7 = self.run_analyzer(producer_events=self.file(log))["producer_events"]
        ledger = rb7["causes"]
        self.assertEqual(ledger["total"], 2)
        self.assertEqual(ledger["by_state"], {"failed_final": 1, "fenced": 1})
        self.assertEqual(ledger["by_origin"], {"reorg": 1, "revert": 1})
        self.assertEqual(
            ledger["first_winning"], {"unmeasured:superseded_by_branch_change": 1, "none": 1}
        )
        # Neither cause ever published, so the publication intervals have no samples — and the
        # populations still say two causes could have produced one.
        self.assertNotIn("detection_to_publication_us", rb7["intervals"])
        self.assertEqual(rb7["intervals"]["detection_to_export_start_us"]["population"], 2)

    def test_an_export_that_completed_but_never_published_leaves_the_gap_visible(self):
        log = self.healthy_log()[:-1]
        rb7 = self.run_analyzer(producer_events=self.file(log))["producer_events"]
        self.assertNotIn("checkpoint_announce_to_complete_us", rb7["intervals"])
        self.assertNotIn("export_complete_to_publication_us", rb7["intervals"])
        self.assertEqual(rb7["causes"]["by_state"], {"in_flight": 1})
        self.assertEqual(rb7["intervals"]["export_duration_us"]["count"], 1)

    def test_the_stream_opening_export_is_counted_but_anchors_no_interval(self):
        log = [
            event("export_started", 0, 1_000, 500, attempt=1, block=10),
            event("export_completed", 0, 5_000, 504, attempt=1, block=10, export_us=3_900),
            event("checkpoint_published", 0, 5_500, 505, attempt=1, block=10,
                  announce_to_complete_us=200),
        ]
        rb7 = self.run_analyzer(producer_events=self.file(log))["producer_events"]
        self.assertEqual(rb7["causes"]["total"], 1)
        self.assertEqual(rb7["causes"]["by_origin"], {"initial": 1})
        self.assertEqual(rb7["causes"]["measurable"], 0)
        self.assertNotIn("detection_to_export_start_us", rb7["intervals"])
        self.assertEqual(rb7["intervals"]["export_duration_us"]["p50"], 4_000.0)

    def test_arming_churn_is_counted_but_does_not_dilute_the_denominator(self):
        # A cold cache arms one re-checkpoint per warm-up block; the first live run saw sixty of
        # them before its export. Each was superseded by the successor carrying the same need, so
        # none is a recovery that failed to be measured — and reading the one real interval as
        # "1 of 61" would say sixty recoveries went unmeasured.
        log = []
        for cause in range(1, 4):
            log.append(event("recheckpoint_armed", cause, cause * 100, cause,
                             cause="discontinuity", armed=True))
        log += [
            event("export_started", 3, 400, 4, attempt=1, block=10),
            event("export_completed", 3, 900, 9, attempt=1, block=10, export_us=480),
            event("checkpoint_published", 3, 950, 9, attempt=1, block=10,
                  announce_to_complete_us=40),
        ]
        rb7 = self.run_analyzer(producer_events=self.file(log))["producer_events"]
        ledger = rb7["causes"]
        self.assertEqual(ledger["total"], 3)
        self.assertEqual(ledger["by_state"], {"published": 1, "superseded": 2})
        self.assertEqual(ledger["measurable"], 1)
        start = rb7["intervals"]["detection_to_export_start_us"]
        self.assertEqual((start["count"], start["population"]), (1, 1))
        self.assertEqual([row["cause_id"] for row in rb7["per_cause"] if row["measurable"]], [3])

    def test_only_the_highest_epoch_is_reduced(self):
        log = [
            event("reorg_detected", 1, 1_000, 500, epoch=1),
            event("export_started", 1, 2_000, 501, attempt=1, epoch=1),
            event("reorg_detected", 1, 1_000, 900, epoch=2),
            event("recheckpoint_armed", 1, 1_100, 900, epoch=2, armed=True),
            event("export_started", 1, 2_000, 901, attempt=1, epoch=2),
            event("export_completed", 1, 4_000, 903, attempt=1, epoch=2, export_us=1_900),
        ]
        rb7 = self.run_analyzer(producer_events=self.file(log))["producer_events"]
        self.assertEqual(rb7["epoch"], 2)
        self.assertEqual(rb7["causes"]["total"], 1)
        self.assertEqual(rb7["intervals"]["export_duration_us"]["count"], 1)


class Rb7ClockTest(AnalyzerCase):
    def test_the_producer_follower_leg_joins_on_frame_sequence(self):
        log = [
            event("reorg_detected", 1, 1_000, 10_000),
            event("recheckpoint_armed", 1, 1_100, 10_000, cause="branch_change", armed=True),
            event("export_started", 1, 2_000, 10_001, attempt=1),
            event("first_winning_commit_published", 1, 3_000, 10_002, block=41, sequence=77),
        ]
        follow = [
            {"kind": "run_manifest"},
            # A different sequence at the same height: joining on the block would take this one.
            verdict(76, 41, "0xff", tail_live=True, observed_at_ms=10_500),
            verdict(77, 41, "0x29", tail_live=True, observed_at_ms=10_042),
            {"kind": "summary"},
        ]
        rb7 = self.run_analyzer(
            follow=[self.file(follow)], producer_events=self.file(log)
        )["producer_events"]
        join = rb7["intervals"]["first_winning_to_follower_verdict_ms"]
        self.assertEqual(join["unit"], "ms")
        self.assertEqual(join["p50"], 40.0)

    def test_the_join_takes_the_followers_first_decision_on_the_frame(self):
        log = [
            event("reorg_detected", 1, 1_000, 10_000),
            event("recheckpoint_armed", 1, 1_100, 10_000, armed=True),
            event("export_started", 1, 2_000, 10_001, attempt=1),
            event("first_winning_commit_published", 1, 3_000, 10_002, sequence=77),
        ]
        follow = [
            {"kind": "run_manifest"},
            verdict(77, 41, tail_live=True, observed_at_ms=10_042),
            verdict(77, 41, catch_up=True, observed_at_ms=99_999),
            {"kind": "summary"},
        ]
        rb7 = self.run_analyzer(
            follow=[self.file(follow)], producer_events=self.file(log)
        )["producer_events"]
        self.assertEqual(rb7["intervals"]["first_winning_to_follower_verdict_ms"]["p50"], 40.0)

    def test_a_negative_cross_clock_reading_is_dropped_and_counted(self):
        # The two processes stamp with their own SystemTime. An offset larger than the interval
        # produces a negative reading; clamping it at zero would report the offset as a latency.
        log = [
            event("reorg_detected", 1, 1_000, 10_000),
            event("recheckpoint_armed", 1, 1_100, 10_000, armed=True),
            event("export_started", 1, 2_000, 10_001, attempt=1),
            event("first_winning_commit_published", 1, 3_000, 10_002, sequence=77),
        ]
        follow = [
            {"kind": "run_manifest"},
            verdict(77, 41, tail_live=True, observed_at_ms=9_000),
            {"kind": "summary"},
        ]
        rb7 = self.run_analyzer(
            follow=[self.file(follow)], producer_events=self.file(log)
        )["producer_events"]
        self.assertNotIn("first_winning_to_follower_verdict_ms", rb7["intervals"])
        rejected = rb7["anomalies"]["rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["why"], "negative cross-clock")
        self.assertEqual(rejected[0]["value_ms"], -1_002)

    def test_a_frame_the_follower_never_verified_is_named_not_silently_dropped(self):
        log = [
            event("reorg_detected", 1, 1_000, 10_000),
            event("recheckpoint_armed", 1, 1_100, 10_000, armed=True),
            event("export_started", 1, 2_000, 10_001, attempt=1),
            event("first_winning_commit_published", 1, 3_000, 10_002, sequence=77),
        ]
        follow = [{"kind": "run_manifest"}, verdict(1, 40), {"kind": "summary"}]
        rb7 = self.run_analyzer(
            follow=[self.file(follow)], producer_events=self.file(log)
        )["producer_events"]
        self.assertEqual(
            rb7["anomalies"]["unjoined_first_winning"], [{"cause_id": 1, "sequence": 77}]
        )

    def test_without_a_follower_input_the_cross_clock_leg_says_it_is_unavailable(self):
        result = self.run_analyzer(producer_events=self.file([
            event("reorg_detected", 1, 1_000, 10_000),
            event("first_winning_commit_published", 1, 3_000, 10_002, sequence=77),
        ]))
        self.assertEqual(result["producer_events"]["anomalies"]["follower_join"], "unavailable")
        self.assertTrue(any("not measurable" in w for w in result["warnings"]))

    def test_a_monotonic_reset_inside_one_epoch_refuses_the_interval(self):
        # Two producer processes appending under one epoch number: the second's monotonic origin
        # is its own, so differencing across the boundary would measure nothing real.
        log = [
            event("reorg_detected", 1, 900_000, 10_000),
            event("recheckpoint_armed", 1, 900_100, 10_000, armed=True),
            event("export_started", 1, 5, 20_000, attempt=1),
            event("export_completed", 1, 2_005, 20_002, attempt=1, export_us=1_900),
        ]
        rb7 = self.run_analyzer(producer_events=self.file(log))["producer_events"]
        self.assertNotIn("detection_to_export_start_us", rb7["intervals"])
        self.assertEqual(rb7["intervals"]["export_duration_us"]["p50"], 2_000.0)
        self.assertEqual(
            [item["why"] for item in rb7["anomalies"]["rejected"]], ["monotonic clock reset"]
        )


class EventReducerTest(unittest.TestCase):
    def test_a_failure_with_retries_left_is_pending_not_settled(self):
        _, states = reduce_events(json.dumps(record) for record in [
            event("recheckpoint_armed", 1, 1, 1, armed=True),
            event("export_started", 1, 2, 1, attempt=1),
            event("export_failed", 1, 3, 1, attempt=1, retries_left=1),
        ])
        self.assertEqual(states, {1: "retrying"})

    def test_an_armed_cause_that_never_attempted_is_superseded_by_the_next(self):
        _, states = reduce_events(json.dumps(record) for record in [
            event("recheckpoint_armed", 1, 1, 1, armed=True),
            event("recheckpoint_armed", 2, 2, 1, armed=True),
        ])
        self.assertEqual(states, {1: "superseded", 2: "armed"})

    def test_the_never_policy_is_terminal_on_arrival(self):
        _, states = reduce_events(json.dumps(record) for record in [
            event("recheckpoint_armed", 1, 1, 1, armed=False),
        ])
        self.assertEqual(states, {1: "not_armed"})

    def test_attempts_nest_under_their_cause(self):
        _, causes = assemble_causes(json.dumps(record) for record in [
            event("reorg_detected", 1, 1, 1),
            event("export_started", 1, 2, 1, attempt=1),
            event("export_failed", 1, 3, 1, attempt=1, retries_left=1),
            event("export_started", 1, 4, 1, attempt=2),
        ])
        self.assertEqual(len(causes[1]["attempts"]), 2)
        self.assertEqual([a["attempt"] for a in causes[1]["attempts"]], [1, 2])
        self.assertIsNotNone(causes[1]["attempts"][0]["failed"])
        self.assertIsNone(causes[1]["attempts"][1]["failed"])


class RendererTest(unittest.TestCase):
    def minimal_result(self):
        return {
            "schema_version": 1,
            "generated_by": "analyze_follow_bench.py",
            "inputs": {"follow": ["/runs/follow.jsonl"], "batch": None,
                       "producer_events": None, "producer_manifest": None},
            "warnings": ["two run segments, analyzing the last"],
            "follow": {
                "runs": [{"file": "/runs/follow.jsonl", "segment": 0, "label": "gate",
                          "complete": True}],
                "summary": {"agreed": True, "continuous": True, "blocks_verified": 2,
                            "catch_up_blocks": 0, "disagreements": 0, "failures": 0,
                            "reorgs_applied": 1, "reverts_applied": 0, "restores": 1,
                            "restores_continuous": 1, "restores_reset": 0},
                "populations": {
                    "delivered": {"verdicts": 2, "live_tail_verdicts": 2, "metrics": {
                        "standalone_validation_us": {"count": 2, "population": 2, "nulls": 0,
                                                     "mean": 150.0, "p50": 150.0, "p95": 195.0,
                                                     "p99": 199.0, "min": 100, "max": 200},
                    }},
                    "phase:steady": {"verdicts": 2, "live_tail_verdicts": 2, "metrics": {
                        "standalone_validation_us": {"count": 2, "population": 2, "nulls": 0,
                                                     "mean": 150.0, "p50": 150.0, "p95": 195.0,
                                                     "p99": 199.0, "min": 100, "max": 200},
                    }},
                },
            },
        }

    def test_the_golden_report(self):
        # Pinned verbatim, prose included. The report is the artifact a reader judges the cohort
        # from, and a silent change to what a column means is exactly the kind of drift a
        # property assertion would pass.
        expected = GOLDEN_REPORT
        self.assertEqual(render(self.minimal_result()), expected)

    def test_a_batch_summary_adds_its_own_row_and_its_absence_removes_it(self):
        self.assertNotIn("batch agreed", render(self.minimal_result()))
        result = self.minimal_result()
        result["batch"] = {"summary": {"agreed": True, "continuous": True, "closed": True},
                           "populations": {}}
        row = next(line for line in render(result).splitlines() if "batch agreed" in line)
        self.assertEqual([cell.strip() for cell in row.strip("|").split("|")],
                         ["batch agreed / continuous / closed", "True / True / True"])

    def test_every_markdown_table_is_rectangular(self):
        body = render(self.minimal_result())
        block = []
        for line in body.splitlines() + [""]:
            if line.startswith("|"):
                block.append(line)
                continue
            if block:
                self.assertEqual(len({len(row) for row in block}), 1, block[0])
            block = []

    def test_an_unknown_schema_is_refused_rather_than_rendered(self):
        result = self.minimal_result()
        result["schema_version"] = 99
        with self.assertRaises(SystemExit):
            render(result)

    def test_a_phase_with_no_verdicts_gets_no_table(self):
        result = self.minimal_result()
        result["follow"]["populations"]["phase:recovery"] = {
            "verdicts": 0, "live_tail_verdicts": 0, "metrics": {}
        }
        self.assertNotIn("**phase:recovery**", render(result))

    def test_rb7_refusals_are_rendered_as_rows_not_dropped(self):
        result = self.minimal_result()
        result["producer_events"] = {
            "epoch": 1,
            "causes": {"total": 1, "by_origin": {"reorg": 1}, "by_state": {"fenced": 1},
                       "attempts_started": 1, "measurable": 1, "first_winning": {"none": 1}},
            "intervals": {},
            "per_cause": [{"cause_id": 1, "origin_kind": "reorg", "state": "fenced",
                           "attempts": 1, "intervals": {}}],
            "anomalies": {"rejected": [{"interval": "detection_to_publication_us", "cause_id": 1,
                                        "why": "negative"}],
                          "unjoined_first_winning": [], "follower_join": "joined"},
        }
        body = render(result)
        self.assertIn("### Refused samples", body)
        self.assertIn("negative", body)


@unittest.skipUnless(I1_ARCHIVE.is_dir(), f"{I1_ARCHIVE} is not on this host")
class I1RegressionTest(unittest.TestCase):
    """The hand-derived preflight numbers, reproduced mechanically."""

    def analyze_archive(self, archive):
        args = argparse.Namespace(
            follow=[str(archive / "out/follow.jsonl")],
            batch=[str(archive / "out/batch.jsonl")],
            producer_events=None,
            producer_manifest=str(archive / "spool.run-manifest.jsonl"),
            run=None, all_runs=False, startup_through_seq=None, phases=False,
        )
        return analyze(args)

    def test_the_primary_p50_matches_the_number_the_paper_draft_quotes(self):
        result = self.analyze_archive(I1_ARCHIVE)
        delivered = result["follow"]["populations"]["delivered"]["metrics"]
        self.assertEqual(delivered["standalone_validation_us"]["count"], 1_001)
        self.assertEqual(delivered["standalone_validation_us"]["p50"], I1_EXPECTED_P50_US)

    def test_the_whole_preflight_is_one_steady_population(self):
        populations = self.analyze_archive(I1_ARCHIVE)["follow"]["populations"]
        self.assertEqual(populations["phase:steady"]["verdicts"], 1_001)
        self.assertEqual(populations["catch_up"]["verdicts"], 0)
        self.assertEqual(populations["abandoned"]["verdicts"], 0)

    def test_every_frame_pairs_with_its_batch_replay(self):
        paired = self.analyze_archive(I1_ARCHIVE)["paired"]
        self.assertEqual(paired["joined"], 1_001)
        self.assertEqual((paired["follow_only"], paired["batch_only"]), (0, 0))
        self.assertEqual(paired["block_mismatched"], 0)

    def test_the_report_renders(self):
        body = render(self.analyze_archive(I1_ARCHIVE))
        self.assertIn("497,586.0", body)
        self.assertIn("## Provenance", body)


@unittest.skipUnless(I2_ARCHIVE.is_dir(), f"{I2_ARCHIVE} is not on this host")
class I2RegressionTest(unittest.TestCase):
    def test_the_i7_preflight_analyzes_and_renders(self):
        args = argparse.Namespace(
            follow=[str(I2_ARCHIVE / "out/follow.jsonl")],
            batch=[str(I2_ARCHIVE / "out/batch.jsonl")],
            producer_events=None, producer_manifest=None,
            run=None, all_runs=False, startup_through_seq=None, phases=False,
        )
        result = analyze(args)
        delivered = result["follow"]["populations"]["delivered"]["metrics"]
        self.assertEqual(delivered["standalone_validation_us"]["count"], 1_001)
        self.assertIn("## Populations", render(result))


GOLDEN_REPORT = """\
# Standalone follow report

Generated from `analyze_follow_bench.py` schema 1 output.

## Warnings

- two run segments, analyzing the last

## Gate status

| field                               | value     |
|-------------------------------------|-----------|
| follower agreed                     | True      |
| follower continuous                 | True      |
| blocks verified                     | 2         |
| catch-up blocks                     | 0         |
| disagreements / failures            | 0 / 0     |
| reorgs / reverts applied            | 1 / 0     |
| restores (continuous / reset)       | 1 (1 / 0) |
| late-skim mismatches                | 0         |
| rewind windows refused              | 0         |
| recovery checkpoints pending at end | 0         |

## Runs analyzed

| file               | segment | label | closed by a summary |
|--------------------|---------|-------|---------------------|
| /runs/follow.jsonl | 0       | gate  | True                |

## Populations

Named and never pooled. `raw_work` is every verdict the process(es) published; `unique_frames` collapses re-publications of one frame sequence; `delivered` is first-time verification; `catch_up` is re-derivation on the way back to a watermark.

| population   | verdicts | live-tail verdicts |
|--------------|----------|--------------------|
| delivered    | 2        | 2                  |
| phase:steady | 2        | 2                  |

### Primary validation cost — `delivered` (µs)

| metric                   | n/population | nulls | p50   | p95   | p99   | max |
|--------------------------|--------------|-------|-------|-------|-------|-----|
| standalone_validation_us | 2/2          | 0     | 150.0 | 195.0 | 199.0 | 200 |

### By phase (µs)

One table per phase because they are different measurements: `steady` is the live tail, `recovery` is a rewind window replayed out of the spool, `startup` is backlog, and `unclassified` is a recording that predates the tail flag. A percentile without one of these labels is not reported.

**phase:steady** — 2 verdicts

| metric                   | n/population | nulls | p50   | p95   | p99   | max |
|--------------------------|--------------|-------|-------|-------|-------|-----|
| standalone_validation_us | 2/2          | 0     | 150.0 | 195.0 | 199.0 | 200 |

## Provenance

| input             | path                   |
|-------------------|------------------------|
| follow            | ['/runs/follow.jsonl'] |
| batch             | None                   |
| producer_events   | None                   |
| producer_manifest | None                   |
"""


if __name__ == "__main__":
    unittest.main()
