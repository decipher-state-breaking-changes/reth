#!/usr/bin/env python3
"""Compatibility tests for validation benchmark retention telemetry."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_validation_bench import build_retention_split_section


def partial_record(**updates):
    partial = {
        "trie_retention_us": 10_000,
        "retention_warm_membership_us": 1_000,
        "retention_storage_paths_us": 2_000,
        "retention_account_paths_us": 1_000,
        "retention_account_trie_us": 3_000,
        "retention_storage_tries_us": 3_000,
        "retention_account_paths": 30_000,
        "retention_storage_tries_pruned": 4,
        "retention_storage_tries_skipped": 6,
    }
    partial.update(updates)
    return {"partial": partial}


class RetentionTelemetryCompatibilityTest(unittest.TestCase):
    def test_schema_three_record_without_walk_details_remains_readable(self):
        lines = build_retention_split_section([partial_record()])

        self.assertTrue(lines)
        self.assertFalse(any("Retention walk internals" in line for line in lines))

    def test_missing_schema_four_detail_fields_default_to_zero(self):
        lines = build_retention_split_section(
            [
                partial_record(
                    retention_account_trie_detail={"nodes_visited": 12},
                    retention_storage_trie_detail={},
                )
            ]
        )
        rendered = "\n".join(lines)

        self.assertIn("Retention walk internals", rendered)
        self.assertIn("sorted fallbacks **0**", rendered)
        self.assertIn("unprunable dirty / inline **0 / 0**", rendered)


if __name__ == "__main__":
    unittest.main()
