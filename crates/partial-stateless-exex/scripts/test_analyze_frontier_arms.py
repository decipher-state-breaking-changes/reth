#!/usr/bin/env python3
"""Tests for the frontier arm analyzer: the rules that decide whether a number is claimable.

    python3 -m unittest test_analyze_frontier_arms -v

`--self-check` is the smoke test a campaign runs before trusting a report. This file holds the
refusals one at a time, because every one of them exists to stop a specific wrong number from being
reported confidently: an unbalanced rotation reported as a policy effect, timings read off a run
that paid an asymmetric compression tax, compressed sizes paired against sidecars they do not
describe, and a break-even bandwidth computed from blocks that have no crossover.

One test is a regression against the accepted 1,000-block Weak control, which lives outside the
repository. Point `PS_FRONTIER_ARCHIVE` at the run directory to enable it; unset, it skips rather
than passing silently.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_frontier_arms import (  # noqa: E402
    PRIMARY,
    SECONDARY,
    analyze,
    as_two_arms,
    break_even,
    delta_total_ms,
    endpoints,
    load_run,
    population_crossover,
    rotation_balance,
    self_check,
)

import random  # noqa: E402


def policy(arm, slot, *, us=100_000, decode_us=1_000, size=1_000_000, zstd=None, digest=None):
    record = {
        "policy": arm,
        "rotation_slot": slot,
        "sidecar_bytes": size,
        "sidecar_digest": digest or f"0x{arm}-{slot}",
        "witness_commitment": "0x00",
        PRIMARY: us,
        SECONDARY: decode_us,
    }
    if zstd is not None:
        record["sidecar_zstd_bytes"] = zstd
    return record


# The two passes a real campaign produces: one binary, one corpus, one flag apart. Spelled out
# because the analyzer now checks that claim, and a fixture that omits it is testing a refusal it
# did not mean to trigger.
TIMING_SUMMARY = {
    "allocator": "jemalloc",
    "generator_build_commit": "deadbeef",
    "measured_block_set_digest": "0xcorpus",
}
SIZE_SUMMARY = dict(TIMING_SUMMARY, compressed_sidecars=True, sidecar_zstd_level=3)


def synthetic(count=400, *, weak_us=100_000, cand_us=125_000, weak_zstd=None, cand_zstd=None,
              weak_size=7_500_000, cand_size=1_500_000, summary=None, seed=7):
    rng = random.Random(seed)
    arms = {"weak": {}, "90/60": {}}
    blocks = {}
    for i in range(count):
        number = 1000 + i
        blocks[number] = {"block_hash": f"0x{number:064x}", "block_admission_us": 2000}
        jitter = 1.0 + rng.uniform(-0.01, 0.01)
        arms["weak"][number] = policy(
            "weak", i % 2, us=int(weak_us * jitter), decode_us=2_000,
            size=weak_size, zstd=weak_zstd, digest=f"0xw{number}",
        )
        arms["90/60"][number] = policy(
            "90/60", (i + 1) % 2, us=int(cand_us * jitter), decode_us=500,
            size=cand_size, zstd=cand_zstd, digest=f"0xc{number}",
        )
    return {"arms": arms, "blocks": blocks, "summary": summary or {}, "path": "<synthetic>"}


class SelfCheck(unittest.TestCase):
    def test_self_check_passes(self):
        self.assertEqual(self_check(), 0)


class RotationBalance(unittest.TestCase):
    def test_alternating_rotation_is_balanced(self):
        run = synthetic()
        self.assertTrue(rotation_balance(run["arms"])["balanced"])

    def test_one_arm_pinned_to_a_slot_is_not_balanced(self):
        run = synthetic()
        for number in run["arms"]["weak"]:
            run["arms"]["weak"][number]["rotation_slot"] = 0
        self.assertFalse(rotation_balance(run["arms"])["balanced"])

    def test_an_even_population_must_be_exactly_balanced(self):
        """400 blocks over 2 arms lands on exactly 200/200; anything else is a defect."""
        run = synthetic()
        numbers = sorted(run["arms"]["weak"])
        run["arms"]["weak"][numbers[0]]["rotation_slot"] = 0
        run["arms"]["weak"][numbers[1]]["rotation_slot"] = 0
        result = rotation_balance(run["arms"])
        self.assertFalse(result["balanced"])
        self.assertFalse(result["exact"])

    def test_an_exactly_balanced_run_says_so(self):
        self.assertTrue(rotation_balance(synthetic()["arms"])["exact"])


class LatencyRefusals(unittest.TestCase):
    def test_balanced_run_answers_and_recovers_the_planted_ratio(self):
        report = analyze(synthetic(), None, "weak", "90/60")
        self.assertAlmostEqual(report["latency"][PRIMARY]["median_ratio"], 1.25, delta=0.02)

    def test_compressed_timing_run_refuses_latency(self):
        run = synthetic(summary={"compressed_sidecars": True})
        report = analyze(run, None, "weak", "90/60")
        self.assertIsNone(report["latency"])
        self.assertIn("zstd tax", report["latency_refused_because"])

    def test_compressed_timing_run_still_answers_on_bytes(self):
        """Bytes survive a compression pass; only the timings are void."""
        run = synthetic(summary={"compressed_sidecars": True})
        report = analyze(run, None, "weak", "90/60")
        self.assertIsNotNone(report["bytes"]["sidecar_bytes"])

    def test_unbalanced_rotation_refuses_latency(self):
        run = synthetic()
        for number in run["arms"]["weak"]:
            run["arms"]["weak"][number]["rotation_slot"] = 0
        report = analyze(run, None, "weak", "90/60")
        self.assertIsNone(report["latency"])
        self.assertIn("not balanced", report["latency_refused_because"])

    def test_missing_arm_is_an_error_not_an_empty_report(self):
        with self.assertRaises(SystemExit):
            analyze(synthetic(), None, "weak", "120/45")


class Pairing(unittest.TestCase):
    def test_ratio_is_of_medians_not_mean_of_ratios(self):
        """A few enormous blocks must not drag the answer the way a mean of ratios would."""
        run = synthetic(count=200)
        numbers = sorted(run["arms"]["weak"])
        for number in numbers[:5]:
            run["arms"]["weak"][number][PRIMARY] = 1
            run["arms"]["90/60"][number][PRIMARY] = 10_000_000
        report = analyze(run, None, "weak", "90/60")
        self.assertAlmostEqual(report["latency"][PRIMARY]["median_ratio"], 1.25, delta=0.03)

    def test_blocks_only_one_arm_saw_refuse_the_whole_report(self):
        """Every arm rotates over every measured block; a gap means the run is inconsistent."""
        run = synthetic(count=100)
        del run["arms"]["90/60"][1099]
        report = analyze(run, None, "weak", "90/60")
        self.assertEqual(report["baseline_only"], 1)
        self.assertTrue(report["integrity_refusals"])
        self.assertIsNone(report["latency"])
        self.assertIsNone(report["bytes"]["sidecar_bytes"])
        self.assertIsNone(report["break_even"])

    def test_a_block_missing_its_hash_refuses_too(self):
        run = synthetic(count=100)
        run["blocks"][1050]["block_hash"] = None
        report = analyze(run, None, "weak", "90/60")
        self.assertEqual(report["refused"], 1)
        self.assertIsNone(report["latency"])

    def test_interval_is_wider_than_an_iid_one_would_be(self):
        """Serially correlated blocks: the moving-block interval must not read as narrow."""
        arms = {"weak": {}, "90/60": {}}
        blocks = {}
        for i in range(400):
            number = 1000 + i
            blocks[number] = {"block_hash": f"0x{number:064x}", "block_admission_us": 2000}
            # A slow ramp: adjacent blocks are alike, distant ones are not.
            ramp = 1.0 + i / 400.0
            arms["weak"][number] = policy("weak", i % 2, us=100_000, digest=f"0xw{number}")
            arms["90/60"][number] = policy(
                "90/60", (i + 1) % 2, us=int(100_000 * ramp), digest=f"0xc{number}"
            )
        run = {"arms": arms, "blocks": blocks, "summary": {}, "path": "<ramp>"}
        row = analyze(run, None, "weak", "90/60")["latency"][PRIMARY]
        low, high = row["interval"]
        self.assertGreater(high - low, 0.15)


class SizePass(unittest.TestCase):
    def test_two_pass_digest_mismatch_refuses_compressed_numbers(self):
        timing = synthetic(summary=TIMING_SUMMARY)
        sizes = synthetic(weak_zstd=5_000_000, cand_zstd=1_000_000, summary=SIZE_SUMMARY)
        for number in sizes["arms"]["weak"]:
            sizes["arms"]["weak"][number]["sidecar_digest"] = "0xdifferent"
        report = analyze(timing, sizes, "weak", "90/60")
        self.assertIsNone(report["compressed"])
        self.assertIn("not describing the same objects", report["compressed_refused_because"])

    def test_matching_two_pass_reports_compressed_ratio(self):
        timing = synthetic(summary=TIMING_SUMMARY)
        sizes = synthetic(weak_zstd=5_000_000, cand_zstd=1_000_000, summary=SIZE_SUMMARY)
        report = analyze(timing, sizes, "weak", "90/60")
        self.assertAlmostEqual(report["compressed"]["sidecar_zstd_bytes"]["median_ratio"], 0.2)
        self.assertEqual(report["compressed"]["zstd_level"], 3)

    def test_compressed_ratio_differs_from_uncompressed_when_arms_compress_unequally(self):
        """The whole reason to measure this: equal byte ratios do not survive compression."""
        sizes = synthetic(weak_size=7_500_000, cand_size=1_500_000,
                          weak_zstd=3_000_000, cand_zstd=1_000_000,
                          summary={"sidecar_zstd_level": 3})
        report = analyze(sizes, None, "weak", "90/60")
        self.assertAlmostEqual(report["bytes"]["sidecar_bytes"]["median_ratio"], 0.2)
        self.assertAlmostEqual(report["compressed"]["sidecar_zstd_bytes"]["median_ratio"], 1 / 3)

    def test_no_zstd_anywhere_refuses_rather_than_reporting_zero(self):
        report = analyze(synthetic(), None, "weak", "90/60")
        self.assertIsNone(report["compressed"])
        self.assertIn("no run recorded", report["compressed_refused_because"])

    def test_partial_zstd_coverage_refuses_rather_than_using_the_subset(self):
        """Compressibility varies with composition, so a partial subset is not a sample."""
        run = synthetic(weak_zstd=5_000_000, cand_zstd=1_000_000,
                        summary={"sidecar_zstd_level": 3})
        for number in list(run["arms"]["weak"])[:100]:
            del run["arms"]["weak"][number]["sidecar_zstd_bytes"]
        report = analyze(run, None, "weak", "90/60")
        self.assertIsNone(report["compressed"])
        self.assertIn("not a sample of the population", report["compressed_refused_because"])

    def test_compressed_sizes_without_a_recorded_level_are_refused(self):
        run = synthetic(weak_zstd=5_000_000, cand_zstd=1_000_000, summary={})
        report = analyze(run, None, "weak", "90/60")
        self.assertIsNone(report["compressed"])
        self.assertIn("no zstd level", report["compressed_refused_because"])

    def test_two_passes_from_different_builds_are_refused(self):
        timing = synthetic(summary={"generator_build_commit": "aaa", "allocator": "jemalloc"})
        sizes = synthetic(weak_zstd=5_000_000, cand_zstd=1_000_000, summary={
            "generator_build_commit": "bbb", "allocator": "jemalloc",
            "compressed_sidecars": True, "sidecar_zstd_level": 3,
        })
        report = analyze(timing, sizes, "weak", "90/60")
        self.assertIsNone(report["compressed"])
        self.assertIn("generator_build_commit", report["compressed_refused_because"])

    def test_a_size_pass_that_did_not_compress_is_refused(self):
        timing = synthetic(summary={"allocator": "jemalloc"})
        sizes = synthetic(weak_zstd=5_000_000, cand_zstd=1_000_000,
                          summary={"allocator": "jemalloc", "sidecar_zstd_level": 3})
        report = analyze(timing, sizes, "weak", "90/60")
        self.assertIsNone(report["compressed"])
        self.assertIn("does not say it compressed", report["compressed_refused_because"])


class Direction(unittest.TestCase):
    """The sign. A pre-registration in this project once had it backwards on both branches."""

    def test_cached_wins_below_break_even_and_loses_above(self):
        # 4 MB saved, 25 ms of extra CPU -> B* = 1280 Mbit/s.
        args = (100_000, 125_000, 5_000_000, 1_000_000)
        self.assertLess(delta_total_ms(*args, 100), 0)
        self.assertLess(delta_total_ms(*args, 1000), 0)
        self.assertAlmostEqual(delta_total_ms(*args, 1280), 0.0, delta=0.1)
        self.assertGreater(delta_total_ms(*args, 4000), 0)

    def test_a_slower_link_helps_the_smaller_arm_monotonically(self):
        args = (100_000, 125_000, 5_000_000, 1_000_000)
        deltas = [delta_total_ms(*args, b) for b in (50, 100, 500, 1000, 2000)]
        self.assertEqual(deltas, sorted(deltas))

    def test_endpoints_report_the_win_rate_alongside_the_median(self):
        shared = list(range(200))
        rows = endpoints(
            shared,
            {n: 100_000 for n in shared},
            {n: 125_000 for n in shared},
            {n: 5_000_000 for n in shared},
            {n: 1_000_000 for n in shared},
            random.Random(1),
            (100.0, 4000.0),
        )
        self.assertEqual(rows[0]["candidate_win_rate"], 1.0)
        self.assertEqual(rows[1]["candidate_win_rate"], 0.0)


class PopulationVersusPerBlock(unittest.TestCase):
    """The two crossovers are different estimands and the tool must not conflate them."""

    def test_endpoints_use_every_block_while_per_block_crossover_does_not(self):
        shared = list(range(100))
        cpu_base = {n: 100_000 for n in shared}
        cpu_cand = {n: 125_000 for n in shared}
        b_base = {n: 5_000_000 for n in shared}
        b_cand = {n: 1_000_000 for n in shared}
        # A quarter of the blocks have the cached arm both smaller and faster: no crossover.
        for n in shared[:25]:
            cpu_cand[n] = 90_000
        rng = random.Random(1)
        rows = endpoints(shared, cpu_base, cpu_cand, b_base, b_cand, rng, (100.0,))
        crossing = break_even(shared, cpu_base, cpu_cand, b_base, b_cand, rng)
        self.assertEqual(rows[0]["n"], 100)
        self.assertEqual(crossing["n"], 75)
        self.assertEqual(crossing["candidate_dominates_blocks"], 25)

    def test_population_crossover_returns_none_when_the_median_never_flips(self):
        shared = list(range(50))
        result = population_crossover(
            shared,
            {n: 125_000 for n in shared},   # baseline slower
            {n: 100_000 for n in shared},   # candidate faster
            {n: 5_000_000 for n in shared},  # and smaller
            {n: 1_000_000 for n in shared},
        )
        self.assertIsNone(result)


class BreakEven(unittest.TestCase):
    def test_recovers_a_planted_crossover(self):
        run = synthetic(weak_us=100_000, cand_us=125_000,
                        weak_zstd=5_000_000, cand_zstd=1_000_000,
                        summary={"sidecar_zstd_level": 3})
        # Timings must come from a run that did not compress, so build the pairing by hand.
        rng = random.Random(1)
        shared = sorted(run["arms"]["weak"])
        result = break_even(
            shared,
            {n: 100_000 for n in shared},
            {n: 125_000 for n in shared},
            {n: 5_000_000 for n in shared},
            {n: 1_000_000 for n in shared},
            rng,
        )
        # 4 MB saved / 25 ms spent = 32e6 bits / 0.025 s = 1280 Mbit/s.
        self.assertAlmostEqual(result["median_mbit_s"], 1280.0, delta=1.0)

    def test_blocks_with_no_crossover_are_counted_not_folded_in(self):
        """One arm smaller *and* faster has no break-even; averaging it in would invent one."""
        rng = random.Random(1)
        shared = list(range(100))
        result = break_even(
            shared,
            {n: 125_000 for n in shared},   # baseline slower
            {n: 100_000 for n in shared},   # candidate faster
            {n: 5_000_000 for n in shared},  # and smaller
            {n: 1_000_000 for n in shared},
            rng,
        )
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["candidate_dominates_blocks"], 100)

    def test_needs_both_halves_before_it_will_answer(self):
        report = analyze(synthetic(), None, "weak", "90/60")
        self.assertIsNone(report["break_even"])
        self.assertIsNone(report["endpoints"])


class AAFloor(unittest.TestCase):
    def test_identical_repetitions_read_as_a_floor_at_one(self):
        a = synthetic(seed=1)
        b = synthetic(seed=1)
        report = analyze(as_two_arms(a, b, "90/60"), None, "rep-a", "rep-b")
        self.assertAlmostEqual(report["latency"][PRIMARY]["median_ratio"], 1.0, delta=0.001)

    def test_a_drifted_repetition_shows_up_as_floor_not_effect(self):
        """The point of the floor: a host that shifted between reps must say so."""
        a = synthetic(seed=1)
        b = synthetic(seed=1, cand_us=137_500)  # 10% slower on the second run
        report = analyze(as_two_arms(a, b, "90/60"), None, "rep-a", "rep-b")
        row = report["latency"][PRIMARY]
        self.assertAlmostEqual(row["median_ratio"], 1.1, delta=0.02)
        self.assertGreater(row["interval"][0], 1.0)

    def test_a_missing_arm_is_an_error(self):
        a = synthetic()
        with self.assertRaises(SystemExit):
            as_two_arms(a, a, "120/45")

    def test_the_floor_uses_the_same_estimator_as_the_effect(self):
        """Same code path, so a floor cannot be tighter than the effect merely by method."""
        a = synthetic(seed=1)
        floor = analyze(as_two_arms(a, synthetic(seed=1), "90/60"), None, "rep-a", "rep-b")
        effect = analyze(a, None, "weak", "90/60")
        self.assertEqual(
            floor["latency"][PRIMARY]["block_len"], effect["latency"][PRIMARY]["block_len"]
        )


class FileShapes(unittest.TestCase):
    def test_warmup_blocks_are_excluded_from_the_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frontier.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for i in range(10):
                    handle.write(json.dumps({
                        "block_number": 100 + i,
                        "block_hash": f"0x{i:064x}",
                        "measured": i >= 4,
                        "block_admission_us": 2000,
                        "policies": [policy("weak", 0), policy("90/60", 1)],
                    }) + "\n")
            run = load_run(str(path))
            self.assertEqual(len(run["arms"]["weak"]), 6)

    def test_directory_and_file_arguments_are_equivalent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frontier.jsonl"
            path.write_text(json.dumps({
                "block_number": 1,
                "block_hash": "0x01",
                "measured": True,
                "block_admission_us": 1,
                "policies": [policy("weak", 0)],
            }) + "\n", encoding="utf-8")
            (Path(tmp) / "frontier-summary.json").write_text(
                json.dumps({"allocator": "jemalloc"}), encoding="utf-8"
            )
            self.assertEqual(load_run(tmp)["summary"]["allocator"], "jemalloc")
            self.assertEqual(load_run(str(path))["summary"]["allocator"], "jemalloc")


@unittest.skipUnless(os.environ.get("PS_FRONTIER_ARCHIVE"), "set PS_FRONTIER_ARCHIVE to enable")
class AcceptedWeakControl(unittest.TestCase):
    """Regression against the real 1,000-block run the accepted 1.24x came from."""

    def setUp(self):
        self.run = load_run(os.environ["PS_FRONTIER_ARCHIVE"])
        self.report = analyze(self.run, None, "weak", "90/60")

    def test_pairs_the_whole_measured_population(self):
        self.assertEqual(self.report["paired_blocks"], 1000)
        self.assertEqual(self.report["baseline_only"], 0)
        self.assertEqual(self.report["candidate_only"], 0)

    def test_rotation_was_balanced(self):
        self.assertTrue(self.report["rotation"]["balanced"])

    def test_reproduces_the_recorded_latency_penalty(self):
        ratio = self.report["latency"][PRIMARY]["median_ratio"]
        self.assertAlmostEqual(ratio, 1.238, delta=0.01)

    def test_paired_byte_ratio_is_not_the_ratio_of_totals(self):
        """5.27x is a ratio of sums; the paired median is a different estimator and says so."""
        paired = self.report["bytes"]["sidecar_bytes"]["median_ratio"]
        totals = self.run["summary"]["policies"]["90/60"]["total_sidecar_bytes"] / (
            self.run["summary"]["policies"]["weak"]["total_sidecar_bytes"]
        )
        self.assertAlmostEqual(paired, 0.176, delta=0.005)
        self.assertAlmostEqual(totals, 0.190, delta=0.005)
        self.assertNotAlmostEqual(paired, totals, delta=0.005)

    def test_the_run_predates_the_allocator_axis(self):
        """The defect this whole re-measurement is about: no allocator was recorded."""
        self.assertIsNone(self.report["allocator"])


if __name__ == "__main__":
    unittest.main()
