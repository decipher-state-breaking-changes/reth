#!/usr/bin/env python3
"""Tests for the run-drift analyzer's statistics and its two input shapes.

The analyzer's `--self-check` is the smoke test a campaign runs before it trusts a report; this
file holds the rules that decide whether a number is claimable, one at a time — the ones that only
matter when someone wants to report an effect the data does not support.

    python3 -m unittest test_analyze_run_drift -v

One test is a regression against a real archive rather than a fixture: the accepted 3,600-verdict
cohort, whose drift was originally derived by hand. It lives outside the repository, so point
`PS_DRIFT_ARCHIVE` at the run directory to enable it; unset, it skips rather than passing silently.

    ./analyze_run_drift.py <run>/out/follow.jsonl <run>/out/batch.jsonl \
        --label live --label batch --json /tmp/drift.json
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_run_drift import (  # noqa: E402
    PRIMARY,
    analyze,
    compare,
    drift,
    load_records,
    per_copy_cost,
    select,
)

import random  # noqa: E402


def record(block, total, phases=None, copies=None, tail_live=None):
    out = {"block": block, PRIMARY: total}
    if phases:
        out["phases"] = phases
    if copies is not None:
        out["details"] = {"trie_storage_tries_copied": copies}
    if tail_live is not None:
        out["tail_live"] = tail_live
    return out


def write_follow(path, records, allocator=None, commit=None):
    with open(path, "w") as handle:
        handle.write(
            json.dumps(
                {
                    "kind": "run_manifest",
                    "label": "fixture",
                    "allocator": allocator,
                    "provenance": {"build_commit": commit},
                }
            )
            + "\n"
        )
        for row in records:
            handle.write(json.dumps(row) + "\n")


class DriftStatistics(unittest.TestCase):
    def test_flat_run_does_not_resolve_an_effect(self):
        """A run with no trend must report an interval spanning 1.0.

        This is the guard that stops a campaign reporting noise as a regression.
        """
        rng = random.Random(1)
        rows = select([record(i, 100_000) for i in range(1000)], "steady")
        result = drift(rows, rng)
        self.assertAlmostEqual(result["ratio"], 1.0)
        self.assertFalse(result["resolves_drift"])
        self.assertLessEqual(result["interval"][0], 1.0)
        self.assertGreaterEqual(result["interval"][1], 1.0)

    def test_step_resolves_and_is_reported_exactly(self):
        rng = random.Random(1)
        rows = select(
            [record(i, 100_000 + (25_000 if i >= 500 else 0)) for i in range(1000)], "steady"
        )
        result = drift(rows, rng)
        self.assertAlmostEqual(result["ratio"], 1.25)
        self.assertTrue(result["resolves_drift"])

    def test_quarters_separate_a_step_from_a_climb(self):
        """The half split cannot tell these apart, which is why quarters are reported beside it."""
        rng = random.Random(1)
        step = select(
            [record(i, 100_000 + (20_000 if i >= 500 else 0)) for i in range(1000)], "steady"
        )
        climb = select([record(i, 100_000 + 40 * i) for i in range(1000)], "steady")
        step_result, climb_result = drift(step, rng), drift(climb, rng)
        # Both land near the same half split — which is the whole problem with reporting it alone.
        self.assertAlmostEqual(step_result["ratio"], 1.2)
        self.assertLess(abs(step_result["ratio"] - climb_result["ratio"]), 0.02)
        # The quarters tell them apart: the step's first two are identical, the climb's are not.
        self.assertEqual(step_result["quarter_p50_us"][0], step_result["quarter_p50_us"][1])
        self.assertLess(climb_result["quarter_p50_us"][0], climb_result["quarter_p50_us"][1])

    def test_moving_block_interval_is_wider_than_an_iid_one_would_be(self):
        """Serially correlated data must not be resampled as if it were independent.

        A run alternating between two regimes in long stretches has a large moving-block interval
        and a vanishing i.i.d. one. Reporting the second would make almost every run significant.
        """
        rng = random.Random(1)
        values = []
        for index in range(1000):
            values.append(record(index, 100_000 if (index // 100) % 2 == 0 else 160_000))
        result = drift(select(values, "steady"), rng)
        self.assertGreater(result["interval"][1] - result["interval"][0], 0.2)


class BootstrapBlockLength(unittest.TestCase):
    """The interval has to contain the point estimate; at small n a fixed block length does not.

    With the default 50 over a 100-value half a replicate is two blocks, and the resulting interval
    can sit entirely to one side of the ratio it is supposed to bracket — a silent failure that
    reads like a confident narrow answer.
    """

    def test_interval_brackets_the_point_estimate_at_small_n(self):
        rng = random.Random(1)
        rows = select(
            [record(index, 100_000 + (6_000 if index >= 100 else 0)) for index in range(200)],
            "steady",
        )
        result = drift(rows, rng)
        self.assertLessEqual(result["interval"][0], result["ratio"])
        self.assertGreaterEqual(result["interval"][1], result["ratio"])

    def test_block_length_shrinks_only_when_the_sample_is_short(self):
        from analyze_run_drift import BLOCK_LEN, effective_block_len

        self.assertEqual(effective_block_len(1800), BLOCK_LEN)
        self.assertEqual(effective_block_len(100), 5)
        self.assertEqual(effective_block_len(4), 1)

    def test_full_length_sample_still_uses_the_default(self):
        rng = random.Random(1)
        rows = select([record(index, 100_000) for index in range(4000)], "steady")
        from analyze_run_drift import BLOCK_LEN

        self.assertEqual(drift(rows, rng)["block_len"], BLOCK_LEN)


class PhaseAttribution(unittest.TestCase):
    def test_ranks_by_absolute_time_not_ratio(self):
        """A phase that triples from 0.1 ms must not outrank one that adds 10 ms."""
        rows = []
        for index in range(1000):
            late = index >= 500
            rows.append(
                record(
                    index,
                    100_000,
                    phases={
                        "tiny_us": 300 if late else 100,
                        "state_root_us": 50_000 if late else 40_000,
                    },
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.jsonl")
            write_follow(path, rows)
            attribution = analyze([path], [], "steady")["runs"][0]["phase_attribution"]
        self.assertEqual(attribution[0]["phase"], "state_root_us")
        self.assertAlmostEqual(attribution[0]["delta_us"], 10_000)


class UnitCostVersusWorkload(unittest.TestCase):
    def test_more_work_at_constant_unit_cost_is_not_a_slowdown(self):
        """Twice the copies for twice the time is a heavier corpus, not a slower process."""
        rows = select(
            [
                record(
                    index,
                    100_000,
                    phases={"state_root_us": 20_000 if index < 500 else 40_000},
                    copies=100 if index < 500 else 200,
                )
                for index in range(1000)
            ],
            "steady",
        )
        result = per_copy_cost(rows)
        self.assertAlmostEqual(result["copies_ratio"], 2.0)
        self.assertAlmostEqual(result["unit_cost_ratio"], 1.0)

    def test_same_work_taking_longer_is_a_slowdown(self):
        rows = select(
            [
                record(
                    index,
                    100_000,
                    phases={"state_root_us": 20_000 if index < 500 else 40_000},
                    copies=100,
                )
                for index in range(1000)
            ],
            "steady",
        )
        result = per_copy_cost(rows)
        self.assertAlmostEqual(result["copies_ratio"], 1.0)
        self.assertAlmostEqual(result["unit_cost_ratio"], 2.0)

    def test_absent_counters_report_nothing_rather_than_zero(self):
        rows = select([record(index, 100_000) for index in range(1000)], "steady")
        self.assertIsNone(per_copy_cost(rows))


class CrossRunComparison(unittest.TestCase):
    def test_pairs_by_block_and_ignores_unshared_blocks(self):
        rng = random.Random(1)
        baseline = select([record(index, 100_000) for index in range(100)], "steady")
        candidate = select([record(index, 80_000) for index in range(50, 150)], "steady")
        result = compare(baseline, candidate, rng)
        self.assertEqual(result["paired_blocks"], 50)
        self.assertEqual(result["baseline_only"], 50)
        self.assertEqual(result["candidate_only"], 50)
        self.assertAlmostEqual(result["paired_median_ratio"], 0.8)

    def test_no_shared_blocks_is_no_comparison(self):
        """Two runs of different corpora are not an A/B, and must not be reported as one."""
        rng = random.Random(1)
        baseline = select([record(index, 100_000) for index in range(100)], "steady")
        candidate = select([record(index, 80_000) for index in range(200, 300)], "steady")
        self.assertIsNone(compare(baseline, candidate, rng))

    def test_identical_runs_pair_at_one(self):
        rng = random.Random(1)
        rows = select([record(index, 100_000 + index) for index in range(500)], "steady")
        result = compare(rows, rows, rng)
        self.assertAlmostEqual(result["paired_median_ratio"], 1.0)
        self.assertAlmostEqual(result["interval"][0], 1.0)
        self.assertAlmostEqual(result["interval"][1], 1.0)


class InputShapes(unittest.TestCase):
    def test_batch_summary_expands_to_the_same_records(self):
        rows = [record(index, 100_000 + index) for index in range(200)]
        with tempfile.TemporaryDirectory() as tmp:
            follow_path = os.path.join(tmp, "follow.jsonl")
            batch_path = os.path.join(tmp, "batch.jsonl")
            write_follow(follow_path, rows)
            with open(batch_path, "w") as handle:
                handle.write(json.dumps({"kind": "run_manifest", "provenance": {}}) + "\n")
                handle.write(json.dumps({"agreed": True, "blocks": rows}) + "\n")
            follow_records, _ = load_records(follow_path)
            batch_records, _ = load_records(batch_path)
        self.assertEqual(follow_records, batch_records)

    def test_last_run_segment_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "two.jsonl")
            with open(path, "w") as handle:
                handle.write(json.dumps({"kind": "run_manifest", "provenance": {}}) + "\n")
                handle.write(json.dumps(record(1, 1)) + "\n")
                handle.write(
                    json.dumps({"kind": "run_manifest", "allocator": "jemalloc", "provenance": {}})
                    + "\n"
                )
                handle.write(json.dumps(record(2, 2)) + "\n")
            records, manifest = load_records(path)
        self.assertEqual([row["block"] for row in records], [2])
        self.assertEqual(manifest["allocator"], "jemalloc")

    def test_backlog_is_dropped_only_when_the_record_says_so(self):
        """A batch record carries no tail flag and must survive the steady filter."""
        rows = [
            record(1, 100, tail_live=False),
            record(2, 100, tail_live=True),
            record(3, 100),
        ]
        self.assertEqual([number for number, _ in select(rows, "steady")], [2, 3])
        self.assertEqual([number for number, _ in select(rows, "all")], [1, 2, 3])

    def test_allocator_and_commit_reach_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.jsonl")
            write_follow(
                path,
                [record(index, 100_000) for index in range(100)],
                allocator="jemalloc",
                commit="079cbbc6aa",
            )
            run = analyze([path], [], "steady")["runs"][0]
        self.assertEqual(run["allocator"], "jemalloc")
        self.assertEqual(run["build_commit"], "079cbbc6aa")


class CommandLine(unittest.TestCase):
    """The CLI shape the campaign driver depends on.

    `--run` exists because argparse stops collecting positionals at the first option, so
    `a.jsonl --label a b.jsonl --label b` silently loses every run after the first. That failure
    mode is invisible until a campaign has already spent its replays.
    """

    def run_cli(self, argv):
        import subprocess

        return subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "analyze_run_drift.py")] + argv,
            capture_output=True,
            text=True,
        )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = []
        for name in ("a", "b"):
            path = os.path.join(self.tmp.name, f"{name}.jsonl")
            write_follow(path, [record(index, 100_000) for index in range(100)])
            self.paths.append(path)

    def test_runs_and_labels_pair_when_given_as_options(self):
        result = self.run_cli(
            ["--run", self.paths[0], "--label", "one", "--run", self.paths[1], "--label", "two"]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("## one", result.stdout)
        self.assertIn("## two", result.stdout)
        self.assertIn("## two vs one", result.stdout)

    def test_bare_positionals_still_work(self):
        result = self.run_cli(self.paths)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_label_count_mismatch_is_refused(self):
        result = self.run_cli(["--run", self.paths[0], "--run", self.paths[1], "--label", "one"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("labels are positional", result.stderr)


class Determinism(unittest.TestCase):
    def test_two_readings_of_one_file_agree_exactly(self):
        """A seeded report is reproducible; an unseeded one invites re-rolling until it reads well."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.jsonl")
            write_follow(
                path,
                [record(index, 100_000 + (index * 30)) for index in range(1000)],
            )
            first = analyze([path], [], "steady")
            second = analyze([path], [], "steady")
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))


@unittest.skipUnless(os.environ.get("PS_DRIFT_ARCHIVE"), "PS_DRIFT_ARCHIVE is unset")
class AcceptedCohortRegression(unittest.TestCase):
    """The hand-derived drift of the accepted cohort, as a regression on the real recording."""

    def setUp(self):
        self.root = Path(os.environ["PS_DRIFT_ARCHIVE"])

    def test_live_and_batch_report_the_same_drift(self):
        result = analyze(
            [str(self.root / "out" / "follow.jsonl"), str(self.root / "out" / "batch.jsonl")],
            ["live", "batch"],
            "steady",
        )
        live, batch = result["runs"]
        self.assertEqual(live["n"], 3601)
        self.assertEqual(batch["n"], 3609)
        for run in (live, batch):
            self.assertAlmostEqual(run["drift"]["ratio"], 1.10, places=2)
            self.assertTrue(run["drift"]["resolves_drift"])
            self.assertEqual(run["phase_attribution"][0]["phase"], "state_root_us")
            self.assertAlmostEqual(run["per_copy"]["unit_cost_ratio"], 1.77, places=1)
            self.assertLess(run["per_copy"]["copies_ratio"], 1.0)

    def test_batch_replay_is_slightly_cheaper_than_the_live_tail(self):
        """Co-location and the delivery path, and the only direction that reading can go."""
        result = analyze(
            [str(self.root / "out" / "follow.jsonl"), str(self.root / "out" / "batch.jsonl")],
            ["live", "batch"],
            "steady",
        )
        comparison = result["comparisons"][0]
        self.assertGreater(comparison["paired_blocks"], 3500)
        self.assertLess(comparison["paired_median_ratio"], 1.0)
        self.assertGreater(comparison["paired_median_ratio"], 0.95)


if __name__ == "__main__":
    unittest.main()
