#!/usr/bin/env python3
"""Regression tests for measurement eligibility, joining and lifetime resource capture."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_prepared_validation import analyze, load_pass
from analyze_validation_bench import build_standalone_section
from analyze_builder_bench import build_commit_section
from measure_resources import run, sample
from analyze_recovery import summarize
from run_live_paired_bench import configured_undo_dirs


class PreparedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def make_pass(self, name, arm="weak", scale=1):
        directory = self.root / name
        directory.mkdir()
        config = {"schema_version": 1, "binary_keccak256": "test-binary", "warmup": 0,
                  "arm": arm, "timing_boundary": "payload_decode_through_coordinated_commit",
                  "samples": 3, "build_commit": "test", "build_dirty": "true", "allocator": "system",
                  "interval_ms": 0, "input_manifest_digest": arm, "retention_depth": 32 if arm != "weak" else 0,
                  "undo_recording": arm != "weak", "undo_layout": "disk-frames" if arm != "weak" else "none",
                  "warm_shrink_blocks": None, "timing_eligible": True}
        rows = [{"block_number": i, "block_hash": str(i), "arm": arm,
                 "measured": True, "valid": True, "verified_us": 50 * scale,
                 "block_step_us": 100 * scale, "validation_core_us": 40 * scale,
                 "resident_flat_undo_records": 0, "sidecar_bytes": 1000,
                 "active_warm_shrink_blocks": None,
                 "commit": {"disk": {}, "retained_depth": 0 if arm == "weak" else 32,
                            "current_undo_written": False, "completed_prior_depth": 31}}
                for i in range(3)]
        (directory / "run.json").write_text(json.dumps(config))
        (directory / "result.json").write_text(json.dumps({"complete": True, "samples": 3,
            "measured_block_set_digest": "same", "writer": {"pending": 0, "failed": 0, "submitted": 2, "completed": 2} if arm != "weak" else None}))
        (directory / "validation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        return directory

    def change(self, directory, name, field, value):
        path = directory / name
        data = json.loads(path.read_text())
        data[field] = value
        path.write_text(json.dumps(data))

    def test_pairing_uses_block_ratios_and_separates_repetitions(self):
        a = self.make_pass("a")
        b = self.make_pass("b", "90/60", 2)
        with patch("analyze_frontier_arms.RESAMPLES", 20):
            result = analyze([a], [b])
        self.assertEqual(result["combined"]["block_step_us"]["median_ratio"], 2)
        self.assertEqual(result["blocks"], 3)
        self.assertEqual(len(result["repetitions"]), 1)

    def test_rejects_missing_completion_and_writer_failure(self):
        directory = self.make_pass("run", "90/60")
        self.change(directory, "result.json", "writer", {"pending": 1})
        with self.assertRaisesRegex(ValueError, "undrained"):
            load_pass(directory)
        self.change(directory, "result.json", "writer", {"failed": 1})
        with self.assertRaisesRegex(ValueError, "writer"):
            load_pass(directory)
        (directory / "result.json").unlink()
        with self.assertRaises(FileNotFoundError):
            load_pass(directory)

    def test_rejects_missing_warmup_and_missing_disk_profile(self):
        directory = self.make_pass("run", "90/60")
        self.change(directory, "run.json", "warmup", 1)
        with self.assertRaisesRegex(ValueError, "warm-up"):
            load_pass(directory)
        self.change(directory, "run.json", "warmup", 0)
        self.change(directory, "result.json", "writer", None)
        with self.assertRaisesRegex(ValueError, "undo profile"):
            load_pass(directory)

    def test_memory_frames_have_no_writer_and_keep_one_flat_record_per_frame(self):
        directory = self.make_pass("run", "90/60")
        self.change(directory, "run.json", "undo_layout", "memory-frames")
        with self.assertRaisesRegex(ValueError, "undo profile"):
            load_pass(directory)
        self.change(directory, "result.json", "writer", None)
        path = directory / "validation.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["commit"] = {"retained_depth": 32}
            row["resident_flat_undo_records"] = 32
        path.write_text("\n".join(map(json.dumps, rows)))
        load_pass(directory)
        rows[2]["resident_flat_undo_records"] = 33
        path.write_text("\n".join(map(json.dumps, rows)))
        with self.assertRaisesRegex(ValueError, "resident flat undo"):
            load_pass(directory)

    def test_rejects_diagnostic_run_and_mismatched_build(self):
        a, b = self.make_pass("a"), self.make_pass("b")
        self.change(b, "run.json", "timing_eligible", False)
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            analyze([a], [b])
        self.change(b, "run.json", "timing_eligible", True)
        self.change(b, "run.json", "build_commit", "another")
        with self.assertRaisesRegex(ValueError, "build_commit"):
            analyze([a], [b])

    def test_vary_lets_named_settings_differ_between_sides_only(self):
        a1, a2 = self.make_pass("a1"), self.make_pass("a2")
        b1, b2 = self.make_pass("b1"), self.make_pass("b2")
        for directory in (b1, b2):
            self.change(directory, "run.json", "malloc_conf", "dirty_decay_ms:-1")
        with self.assertRaisesRegex(ValueError, "disagree on malloc_conf"):
            analyze([a1, a2], [b1, b2])
        with patch("analyze_frontier_arms.RESAMPLES", 20):
            result = analyze([a1, a2], [b1, b2], vary=["malloc_conf"])
        self.assertEqual(result["varied"], {"malloc_conf": {"baseline": None, "candidate": "dirty_decay_ms:-1"}})
        # A varied setting still has to hold within each side.
        self.change(b2, "run.json", "malloc_conf", "dirty_decay_ms:30000")
        with self.assertRaisesRegex(ValueError, "repetitions disagree on malloc_conf"):
            analyze([a1, a2], [b1, b2], vary=["malloc_conf"])
        # The trie parallelism floor is guarded the same way.
        self.change(b2, "run.json", "malloc_conf", "dirty_decay_ms:-1")
        self.change(b1, "run.json", "trie_parallel_min", [64, 64])
        self.change(b2, "run.json", "trie_parallel_min", [64, 64])
        with self.assertRaisesRegex(ValueError, "disagree on trie_parallel_min"):
            analyze([a1, a2], [b1, b2], vary=["malloc_conf"])
        with patch("analyze_frontier_arms.RESAMPLES", 20):
            result = analyze([a1, a2], [b1, b2], vary=["malloc_conf", "trie_parallel_min"])
        self.assertEqual(result["varied"]["trie_parallel_min"], {"baseline": None, "candidate": [64, 64]})
        # And so is the retention walk: a full-versus-narrowed A/B names it.
        for directory in (b1, b2):
            self.change(directory, "run.json", "delta_retention", "on")
        with self.assertRaisesRegex(ValueError, "disagree on delta_retention"):
            analyze([a1, a2], [b1, b2], vary=["malloc_conf", "trie_parallel_min"])
        with patch("analyze_frontier_arms.RESAMPLES", 20):
            result = analyze([a1, a2], [b1, b2],
                             vary=["malloc_conf", "trie_parallel_min", "delta_retention"])
        self.assertEqual(result["varied"]["delta_retention"], {"baseline": None, "candidate": "on"})
        # Only process settings can be varied; the build and the boundary never can.
        with self.assertRaisesRegex(ValueError, "cannot vary build_commit"):
            analyze([a1], [b1], vary=["build_commit"])

    def test_undo_filesystem_must_hold_within_a_side(self):
        first, second = self.make_pass("first", "90/60"), self.make_pass("second", "90/60")
        self.change(first, "run.json", "undo_filesystem", "tmpfs tmpfs /dev/shm")
        self.change(second, "run.json", "undo_filesystem", "/dev/sdc1 ext4 /data2")
        with self.assertRaisesRegex(ValueError, "repetitions disagree on undo_filesystem"):
            analyze([first, second], [self.make_pass("weak-1"), self.make_pass("weak-2")])

    def test_rejects_different_order_even_with_same_claimed_digest(self):
        a, b = self.make_pass("a"), self.make_pass("b")
        path = b / "validation.jsonl"
        path.write_text("\n".join(reversed(path.read_text().splitlines())))
        with self.assertRaisesRegex(ValueError, "ordered block"):
            analyze([a], [b])

    def test_rejects_sidecar_bytes_that_change_between_repetitions(self):
        first = self.make_pass("first", "90/60")
        second = self.make_pass("second", "90/60")
        path = second / "validation.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["sidecar_bytes"] += 1
        path.write_text("\n".join(map(json.dumps, rows)))
        with self.assertRaisesRegex(ValueError, "sidecar bytes"):
            analyze(
                [first, second],
                [self.make_pass("weak-1"), self.make_pass("weak-2")],
            )

    def test_rejects_warm_shrink_provenance_mismatch(self):
        directory = self.make_pass("run", "90/60")
        self.change(directory, "run.json", "warm_shrink_blocks", 100)
        with self.assertRaisesRegex(ValueError, "warm-shrink"):
            load_pass(directory)

    def test_rejects_resident_flat_history_and_invalid_time(self):
        directory = self.make_pass("run")
        path = directory / "validation.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["resident_flat_undo_records"] = 1
        path.write_text("\n".join(map(json.dumps, rows)))
        with self.assertRaisesRegex(ValueError, "resident flat undo"):
            load_pass(directory)
        rows[0]["resident_flat_undo_records"] = 0
        rows[0]["verified_us"] = 200
        path.write_text("\n".join(map(json.dumps, rows)))
        with self.assertRaisesRegex(ValueError, "timing bounds"):
            load_pass(directory)


class JoinTests(unittest.TestCase):
    def test_recovery_uses_monotonic_time_and_does_not_cross_restart(self):
        event = {"kind": "lifecycle", "event": "reorg_applied", "run_id": "first",
                 "process_elapsed_us": 100, "recovery_us": 20, "common_ancestor": 1,
                 "recovery_started_elapsed_us": 70,
                 "abandoned": [2, 3], "winning_tip": 3, "winning_tip_hash": "c"}
        first = {"kind": "verdict", "verdict": "accepted", "run_id": "first",
                 "process_elapsed_us": 150, "block": 2, "block_hash": "b", "standalone_validation_us": 30}
        head = {**first, "block": 3, "block_hash": "c", "process_elapsed_us": 200}
        result = summarize([event, first, head])[0]
        self.assertEqual(result["through_first_verdict_us"], 80)
        self.assertEqual(result["through_winning_head_us"], 130)
        head["run_id"] = "second"
        self.assertIsNone(summarize([event, first, head])[0]["through_winning_head_us"])

    def test_standalone_requires_actual_live_disk_commit(self):
        accepted = [{"block_number": 1, "block_hash": "a", "vanilla_engine": {"validation_us": 100}}]
        verdict = {"kind": "verdict", "verdict": "accepted", "block": 1, "block_hash": "a",
                   "tail_live": True, "standalone_validation_us": 200,
                   "undo": {"commit": {"disk": {"submitted": 1}}}}
        summary = {"kind": "summary", "failures": 0, "disagreements": 0,
                   "undo_writer": {"pending": 0, "failed": 0, "completed": 1}}
        self.assertIn("2.000x", build_standalone_section(accepted, [verdict, summary]))
        with self.assertRaisesRegex(ValueError, "final writer-drain"):
            build_standalone_section(accepted, [verdict])
        verdict["undo"] = None
        with self.assertRaisesRegex(ValueError, "disk-undo"):
            build_standalone_section(accepted, [verdict, summary])
        verdict["tail_live"] = False
        with self.assertRaisesRegex(ValueError, "missing live"):
            build_standalone_section(accepted, [verdict, summary])

    def test_producer_companion_must_match_creation_boundary(self):
        selected = [{"block_number": 1, "block_hash": "a", "builder_total_us": 100}]
        commit = {"block_number": 1, "block_hash": "a", "sidecar_build_us": 100, "through_commit_us": 150,
                  "commit": {"total_us": 40, "undo": {"spill_call_us": 10}}}
        self.assertIn("Production coordination", build_commit_section(selected, [commit]))
        commit["sidecar_build_us"] = 99
        with self.assertRaisesRegex(ValueError, "boundaries"):
            build_commit_section(selected, [commit])
        with self.assertRaisesRegex(ValueError, "missing producer"):
            build_commit_section(selected, [])


class ResourceTests(unittest.TestCase):
    def test_live_undo_directory_matches_the_exex_default(self):
        self.assertEqual(
            configured_undo_dirs({"PS_SIDECAR_DIR": "/run/sidecars"}),
            [Path("/run/sidecars/undo")],
        )
        self.assertEqual(
            configured_undo_dirs({
                "PS_SIDECAR_DIR": "/run/sidecars",
                "PS_UNDO_DIR": "/separate/undo",
            }),
            [Path("/separate/undo")],
        )

    def test_sampling_process_stat_with_spaces_in_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            pid = proc / "42"
            pid.mkdir()
            (pid / "status").write_text("VmRSS: 1024 kB\nVmHWM: 2048 kB\nRssAnon: 512 kB\n")
            stat = ["S"] + ["0"] * 30
            stat[7], stat[9], stat[11], stat[12] = "7", "2", "100", "50"
            (pid / "stat").write_text("42 (some worker) " + " ".join(stat))
            (pid / "io").write_text("read_bytes: 4096\nwrite_bytes: 8192\n")
            row = sample(42, proc=proc)
            self.assertEqual(row["minor_faults"], 7)
            self.assertEqual(row["major_faults"], 2)
            self.assertEqual(row["cpu_seconds"], 150 / os.sysconf("SC_CLK_TCK"))
            self.assertEqual(row["io"]["write_bytes"], 8192)

    def test_exit_record_retains_actual_exit_status_and_peak(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "resources.jsonl"
            code = run([sys.executable, "-c", "x=bytearray(4000000); raise SystemExit(3)"], out, [], .01)
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(code, 3)
            self.assertEqual(rows[-1]["kind"], "exit")
            self.assertGreater(rows[-1]["maxrss_kib"], 0)
            self.assertEqual(rows[-1]["returncode"], 3)


if __name__ == "__main__":
    unittest.main()
