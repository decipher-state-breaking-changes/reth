#!/usr/bin/env python3
"""Tests for the witness-size estimator.

The estimator's whole job is to be honest about a number it cannot measure, so these check the
honesty as much as the arithmetic: that a percentage-only sweep is refused rather than guessed
at, that the shape is chosen on data the fit never saw, and that the pre-registered rule stays
pre-registered — it must be able to say "stay where you are".
"""

import csv
import io
import json
import os
import tempfile
import unittest

import estimate_witness_bytes as est


def producer_log(rows, colour=False):
    """Two lines per block, as the producer really writes them — optionally ANSI-coloured."""
    out = io.StringIO()
    for block, (ma, ms, mc, ap, sp, bb) in rows.items():
        prefix = "\x1b[2m2026-01-01T00:00:00Z\x1b[0m " if colour else ""
        out.write(
            f"{prefix}INFO partial_stateless: witness block={block} "
            f"miss_ratio=\"100.0%\" missed_accounts={ma} missed_storage={ms} "
            f"missed_codes={mc} total_missed={ma + ms + mc}\n"
        )
        out.write(
            f"{prefix}INFO partial_stateless: sizes block={block} "
            f"witness_total_bytes={ap + sp + bb} account_proof_bytes={ap} "
            f"storage_proof_bytes={sp} bytecode_bytes={bb} account_proof_nodes=1 "
            f"storage_proof_nodes=1 target_accounts={ma} target_storage_slots={ms}\n"
        )
    return out.getvalue()


def write(tmpdir, name, text):
    path = os.path.join(tmpdir, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


SWEEP_COLUMNS = [
    "account_window", "storage_window", "measured_blocks", "overall_hit_pct",
    "account_hit_pct", "storage_hit_pct", "code_hit_pct", "acc_accessed", "acc_hit",
    "sto_accessed", "sto_hit", "code_accessed", "code_hit", "avg_cache_accounts",
    "avg_cache_storage", "avg_cache_codes", "avg_cache_mem_bytes", "peak_cache_mem_bytes",
]


def sweep_csv(tmpdir, rows, columns=None):
    path = os.path.join(tmpdir, "sweep.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or SWEEP_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def sweep_row(account, storage, acc_miss, sto_miss, code_miss, mem, blocks=100):
    """A row whose *misses per block* are the three numbers given."""
    return {
        "account_window": account,
        "storage_window": storage,
        "measured_blocks": blocks,
        "overall_hit_pct": 90.0,
        "account_hit_pct": 90.0,
        "storage_hit_pct": 90.0,
        "code_hit_pct": 90.0,
        "acc_accessed": 1000 * blocks,
        "acc_hit": (1000 - acc_miss) * blocks,
        "sto_accessed": 2000 * blocks,
        "sto_hit": (2000 - sto_miss) * blocks,
        "code_accessed": 300 * blocks,
        "code_hit": (300 - code_miss) * blocks,
        "avg_cache_accounts": 1.0,
        "avg_cache_storage": 1.0,
        "avg_cache_codes": 1.0,
        "avg_cache_mem_bytes": mem,
        "peak_cache_mem_bytes": int(mem * 1.2),
    }


class ParsingTest(unittest.TestCase):
    def test_the_two_log_lines_are_joined_on_the_block_they_both_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "p.out", producer_log({10: (5, 7, 2, 500, 700, 200)}))
            observations = est.parse_producer_log(path)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].block, 10)
        self.assertEqual(observations[0].missed_storage, 7)
        self.assertEqual(observations[0].bytecode_bytes, 200)

    def test_the_log_is_read_through_its_colour_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = write(tmp, "a.out", producer_log({1: (2, 3, 4, 5, 6, 7)}))
            coloured = write(tmp, "b.out", producer_log({1: (2, 3, 4, 5, 6, 7)}, colour=True))
            self.assertEqual(est.parse_producer_log(plain), est.parse_producer_log(coloured))

    def test_a_block_with_only_one_of_the_two_lines_is_dropped(self):
        text = producer_log({1: (2, 3, 4, 5, 6, 7)})
        half = text.splitlines()[0] + "\n"          # the miss line, no sizes
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "p.out", half)
            self.assertEqual(est.parse_producer_log(path), [])


class FitTest(unittest.TestCase):
    def test_an_exactly_linear_set_recovers_its_coefficient(self):
        points = [(float(n), 3.0 * n) for n in range(1, 20)]
        self.assertAlmostEqual(est.fit_through_origin(points), 3.0)

    def test_a_power_law_recovers_its_exponent(self):
        points = [(float(n), 5.0 * n ** 0.6) for n in range(1, 50)]
        k, p = est.fit_power(points)
        self.assertAlmostEqual(p, 0.6, places=6)
        self.assertAlmostEqual(k, 5.0, places=5)

    def test_the_shape_is_chosen_on_the_held_out_run_not_the_fitted_one(self):
        # Sublinear truth: the power shape must win, and only the held-out run can say so.
        rows = {n: (n, 1, 1, int(100 * n ** 0.6), 1, 1) for n in range(1, 40)}
        held = {n: (n, 1, 1, int(100 * n ** 0.6), 1, 1) for n in range(40, 60)}
        with tempfile.TemporaryDirectory() as tmp:
            fit = est.parse_producer_log(write(tmp, "fit.out", producer_log(rows)))
            check = est.parse_producer_log(write(tmp, "check.out", producer_log(held)))
        models = est.build_models(fit, check)
        self.assertEqual(models["account"].chosen, "power")
        self.assertLess(models["account"].power_check_error, models["account"].linear_check_error)

    def test_without_a_held_out_run_the_model_says_it_has_no_error_bar(self):
        rows = {n: (n, 1, 1, 100 * n, 1, 1) for n in range(1, 10)}
        with tempfile.TemporaryDirectory() as tmp:
            fit = est.parse_producer_log(write(tmp, "fit.out", producer_log(rows)))
        models = est.build_models(fit, [])
        self.assertIsNone(models["account"].linear_check_error)
        self.assertTrue(any("no held-out run" in note for note in models["account"].notes))

    def test_a_sublinear_exponent_is_called_out_as_overestimating_the_savings(self):
        rows = {n: (n, 1, 1, int(100 * n ** 0.5), 1, 1) for n in range(1, 40)}
        with tempfile.TemporaryDirectory() as tmp:
            fit = est.parse_producer_log(write(tmp, "fit.out", producer_log(rows)))
        models = est.build_models(fit, [])
        self.assertTrue(any("sharing nodes" in note for note in models["account"].notes))


class SweepTest(unittest.TestCase):
    def test_a_percentage_only_sweep_is_refused_rather_than_guessed_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = sweep_csv(
                tmp,
                [{"account_window": 60, "storage_window": 30, "measured_blocks": 10,
                  "overall_hit_pct": 90.0, "avg_cache_mem_bytes": 1.0}],
                columns=["account_window", "storage_window", "measured_blocks",
                         "overall_hit_pct", "avg_cache_mem_bytes"],
            )
            with self.assertRaises(SystemExit) as raised:
                est.read_sweep(path)
        self.assertIn("percentage cannot be turned back", str(raised.exception))

    def test_misses_per_block_come_out_of_the_totals_and_the_block_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = sweep_csv(tmp, [sweep_row(60, 30, 100, 200, 30, 1e9)])
            rows = est.estimate_rows(est.read_sweep(path), fixed_models(1.0, 1.0, 1.0))
        self.assertEqual(rows[0]["misses_per_block"]["account"], 100)
        self.assertEqual(rows[0]["misses_per_block"]["storage"], 200)
        self.assertEqual(rows[0]["misses_per_block"]["code"], 30)
        self.assertEqual(rows[0]["estimated_bytes_total"], 330)


def fixed_models(account_k, storage_k, code_k):
    return {
        name: est.Model(category=name, linear_k=k, power_k=k, power_p=1.0, fit_n=1)
        for name, k in (("account", account_k), ("storage", storage_k), ("code", code_k))
    }


class RuleTest(unittest.TestCase):
    def rows_for(self, specs):
        with tempfile.TemporaryDirectory() as tmp:
            path = sweep_csv(tmp, [sweep_row(*spec) for spec in specs])
            return est.estimate_rows(est.read_sweep(path), fixed_models(1.0, 1.0, 1.0))

    def test_a_candidate_that_saves_too_little_does_not_qualify(self):
        rows = self.rows_for([
            (60, 30, 100, 200, 30, 1e9),          # baseline: 330 bytes
            (120, 60, 95, 190, 29, 1.5e9),        # 314 bytes — only 5% off
        ])
        rule = est.apply_rule(rows, (60, 30))
        self.assertEqual(rule["qualifying"], [])
        self.assertTrue(rule["decision_is_baseline"])

    def test_a_candidate_that_saves_enough_but_costs_too_much_memory_does_not_qualify(self):
        rows = self.rows_for([
            (60, 30, 100, 200, 30, 1e9),
            (240, 120, 50, 100, 15, 4e9),         # half the bytes, four times the memory
        ])
        rule = est.apply_rule(rows, (60, 30))
        self.assertEqual(rule["qualifying"], [])
        self.assertEqual(rule["decision"], [60, 30])

    def test_the_smallest_qualifying_window_wins_not_the_cheapest_in_bytes(self):
        rows = self.rows_for([
            (60, 30, 100, 200, 30, 1e9),
            (90, 45, 70, 140, 21, 1.4e9),         # 231 bytes, 1.4x memory — qualifies
            (120, 60, 60, 120, 18, 1.9e9),        # 198 bytes, 1.9x memory — also qualifies
        ])
        rule = est.apply_rule(rows, (60, 30))
        self.assertEqual(rule["qualifying"], [[90, 45], [120, 60]])
        self.assertEqual(rule["decision"], [90, 45])

    def test_a_sweep_without_the_baseline_row_is_refused(self):
        rows = self.rows_for([(90, 45, 70, 140, 21, 1.4e9)])
        with self.assertRaises(SystemExit) as raised:
            est.apply_rule(rows, (60, 30))
        self.assertIn("no baseline row", str(raised.exception))

    def test_the_knee_is_where_one_more_step_stops_paying(self):
        rows = self.rows_for([
            (60, 30, 100, 200, 30, 1e9),          # 330
            (90, 45, 70, 140, 21, 1.1e9),         # 231 — a 30% step
            (120, 60, 68, 136, 20, 1.2e9),        # 224 — a 2% step, below the 3% threshold
            (240, 120, 67, 134, 20, 1.3e9),       # 221
        ])
        rule = est.apply_rule(rows, (60, 30))
        self.assertEqual(rule["knee"], [90, 45])


class ReportTest(unittest.TestCase):
    def test_the_report_says_what_it_is_and_names_the_decision(self):
        rows = RuleTest().rows_for([
            (60, 30, 100, 200, 30, 1e9),
            (90, 45, 70, 140, 21, 1.4e9),
        ])
        rule = est.apply_rule(rows, (60, 30))
        report = est.render(fixed_models(1.0, 1.0, 1.0), rows, rule, ["a.out"], [])
        self.assertIn("Estimates, not measurements", report)
        self.assertIn("decision: 90/45", report)
        self.assertIn("It is not a result", report)

    def test_the_json_is_stamped_as_an_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = {n: (n, n, n, 100 * n, 100 * n, 100 * n) for n in range(1, 30)}
            fit = write(tmp, "fit.out", producer_log(rows))
            sweep = sweep_csv(tmp, [
                sweep_row(60, 30, 100, 200, 30, 1e9),
                sweep_row(90, 45, 70, 140, 21, 1.4e9),
            ])
            out_json = os.path.join(tmp, "out.json")
            est.main(["--sweep", sweep, "--fit", fit, "--json", out_json,
                      "--out", os.path.join(tmp, "out.md")])
            with open(out_json, encoding="utf-8") as handle:
                payload = json.load(handle)
        self.assertEqual(payload["status"], "exploratory-estimate-never-cite-as-measurement")
        self.assertEqual(payload["schema_version"], 1)
        self.assertIn("rule", payload)


if __name__ == "__main__":
    unittest.main()


class KneeShapeTest(unittest.TestCase):
    """A cross-product grid is not a curve, and only one walk through it is a sequence of steps."""

    def test_a_grid_walked_in_two_dimensions_reports_no_knee(self):
        rows = RuleTest().rows_for([
            (8, 4, 640, 1102, 116, 9.6e6),
            (8, 8, 640, 971, 91, 13.7e6),
            (15, 4, 575, 1102, 116, 10.0e6),
            (15, 8, 575, 971, 91, 14.1e6),
        ])
        rule = est.apply_rule(rows, (15, 8))
        # 8/4 -> 8/8 -> 15/4 is not a sequence of single steps in any direction, and only two of
        # these rows share the baseline's ratio.
        self.assertIsNone(rule["knee"])

    def test_the_knee_follows_the_baseline_ratio_and_ignores_off_ratio_rows(self):
        rows = RuleTest().rows_for([
            (60, 30, 100, 200, 30, 1e9),          # on the 2:1 ratio
            (90, 45, 70, 140, 21, 1.1e9),         # on it
            (90, 10, 70, 400, 21, 1.05e9),        # off it — must not be treated as a step
            (120, 60, 68, 136, 20, 1.2e9),        # on it, a 2% step
            (240, 120, 67, 134, 20, 1.3e9),       # on it
        ])
        rule = est.apply_rule(rows, (60, 30))
        self.assertEqual(rule["knee"], [90, 45])
