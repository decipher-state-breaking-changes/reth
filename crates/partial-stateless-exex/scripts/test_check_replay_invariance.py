import json
import tempfile
import unittest
from pathlib import Path

import check_replay_invariance as inv

MANIFEST = {"kind": "run_manifest", "schema_version": 2}
SUMMARY = {
    "benchmark": "standalone_replay_v1",
    "schema_version": 2,
    "label": "current-arm",
    "commits": 40,
    "witnessed": 30,
    "reconstructed": 10,
    "agreed": True,
    "disagreements": 0,
    "reorgs_applied": 0,
    "late_skim_mismatches": 0,
    "recovery_checkpoints_pending_at_end": 0,
    "elapsed_ms": 68543,
    "transition_us": 17678287,
    "watermark_mismatch_samples": [],
    "blocks": [
        {"sequence": 7, "number": 25737714, "verdict": "agreed", "transition_us": 300},
        {"sequence": 8, "number": 25737715, "verdict": "agreed", "transition_us": 310},
    ],
}


def write(record, directory, name):
    path = Path(directory) / name
    path.write_text(json.dumps(MANIFEST) + "\n" + json.dumps(record) + "\n")
    return str(path)


def baseline_from(summary):
    record = {k: v for k, v in summary.items() if k not in inv.CURRENT_ONLY_COUNTERS}
    record["label"] = "baseline-arm"
    return record


class CheckReplayInvarianceTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.current = write(SUMMARY, self.dir.name, "current.json")

    def compare_against(self, baseline_record):
        baseline = write(baseline_record, self.dir.name, "baseline.json")
        problems, _, _ = inv.compare(self.current, baseline)
        return problems

    def test_identical_behavior_passes_despite_timing_and_label_differences(self):
        baseline = baseline_from(SUMMARY)
        baseline["elapsed_ms"] = 1
        baseline["transition_us"] = 5
        baseline["blocks"] = [dict(b, transition_us=1) for b in SUMMARY["blocks"]]
        self.assertEqual(self.compare_against(baseline), [])

    def test_aggregate_field_difference_fails(self):
        baseline = baseline_from(SUMMARY)
        baseline["commits"] = 39
        problems = self.compare_against(baseline)
        self.assertTrue(any(p.startswith("commits:") for p in problems))

    def test_block_sequence_difference_fails(self):
        baseline = baseline_from(SUMMARY)
        baseline["blocks"] = list(reversed(SUMMARY["blocks"]))
        problems = self.compare_against(baseline)
        self.assertTrue(any("block sequences differ" in p for p in problems))

    def test_current_only_counters_on_baseline_side_fail_as_wrong_binary(self):
        problems = self.compare_against(dict(SUMMARY, label="baseline-arm"))
        self.assertEqual(
            sorted(p.split(":")[0] for p in problems),
            sorted(inv.CURRENT_ONLY_COUNTERS),
        )


if __name__ == "__main__":
    unittest.main()
