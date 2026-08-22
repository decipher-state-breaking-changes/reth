#!/usr/bin/env python3
"""Regression tests for the F0 summary's distribution and producer-total columns."""

import json
import pathlib
import tempfile
import unittest

import summarize_f0


class SummarizeF0Tests(unittest.TestCase):
    def make_arm(self, base: pathlib.Path, name: str = "fsync-1A") -> pathlib.Path:
        arm = base / name
        (arm / "out").mkdir(parents=True)
        data = {
            "follow": {
                "summary": {
                    "blocks_verified": 300,
                    "reorgs_applied": 0,
                    "reverts_applied": 0,
                    "ack_write_us": {"mean": 75_050, "p50": 100, "p95": 150_000},
                },
                "populations": {
                    "phase:steady": {
                        "metrics": {
                            "standalone_validation_us": {"mean": 700_000, "p50": 600_000},
                            "decision_latency_us[mtime]": {"mean": 800_000, "p50": 750_000},
                        }
                    }
                },
            }
        }
        (arm / "out" / "distributions.json").write_text(json.dumps(data))
        return arm

    def test_reports_mean_beside_percentiles_and_producer_totals(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            arm = self.make_arm(pathlib.Path(temp))
            (arm / "producer.out").write_text(
                "\x1b[32mINFO\x1b[0m Closed the event stream kind=\"shutdown\" frames=10 "
                "frame_write_us=40000 frame_fsync_us=120000 dir_syncs=9\n"
            )

            rendered = summarize_f0.summarize(pathlib.Path(temp))

            self.assertIn("ack write mean/p50/p95 us", rendered)
            self.assertIn("75,050/100/150,000", rendered)
            self.assertIn("700,000/600,000", rendered)
            self.assertIn("40,000", rendered)
            self.assertIn("120,000", rendered)
            self.assertIn("4,000", rendered)
            self.assertIn("12,000", rendered)
            self.assertIn("shutdown; dir_syncs=9", rendered)
            self.assertIn("includes serialization plus file write", rendered)

    def test_does_not_choose_between_multiple_close_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            arm = self.make_arm(pathlib.Path(temp))
            line = (
                'Closed the event stream kind="shutdown" frames=10 frame_write_us=40 '
                "frame_fsync_us=20 dir_syncs=1\n"
            )
            (arm / "producer.out").write_text(line + line)

            rendered = summarize_f0.summarize(pathlib.Path(temp))

            self.assertIn("multiple(2)", rendered)
            self.assertNotIn("shutdown; dir_syncs=1", rendered)


if __name__ == "__main__":
    unittest.main()
