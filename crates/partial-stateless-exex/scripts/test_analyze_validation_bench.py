#!/usr/bin/env python3
"""Compatibility tests for validation benchmark retention telemetry."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_validation_bench import (
    build_anchor_split_section,
    build_cache_composition_section,
    build_cache_delta_section,
    build_clone_split_section,
    build_retention_split_section,
)


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


def anchor_record(**detail_updates):
    detail = {
        "account_collect_sort_us": 6_000,
        "storage_collect_sort_us": 13_000,
        "code_collect_sort_us": 300,
        "account_leaf_hash_us": 25_000,
        "storage_leaf_hash_us": 26_000,
        "code_leaf_hash_us": 1_800,
        "account_namespace_us": 3_700,
        "storage_namespace_us": 4_200,
        "code_namespace_us": 280,
        "root_us": 2,
        "accounts": 30_100,
        "storage": 32_900,
        "codes": 2_300,
        "memo_hits": 0,
    }
    detail.update(detail_updates)
    return {
        "cache_accounts": detail["accounts"],
        "cache_storage": detail["storage"],
        "cache_codes": detail["codes"],
        "partial": {
            "next_cache_anchor_us": 102_000,
            "cache_update_us": 30_000,
            "trie_retention_us": 82_000,
            "trie_clone_us": 100_000,
            "next_cache_anchor_detail": detail,
            "trie_clone_detail": {
                "account_trie_us": 84_000,
                "storage_tries_us": 5_000,
                "warm_membership_us": 7_000,
                "retained_paths_us": 4_000,
                "storage_tries": 1_820,
                "warm_accounts": 30_100,
                "warm_storage": 32_900,
                "retained_account_paths": 30_500,
            },
        },
    }


class SchemaFiveTelemetryTest(unittest.TestCase):
    """The V5 sections must render, and a V4 record must not make them appear."""

    def test_anchor_split_ranks_the_v2_candidates(self):
        rendered = "\n".join(build_anchor_split_section([anchor_record()]))

        self.assertIn("Next cache anchor split", rendered)
        self.assertIn("the digest index (moved)", rendered)
        self.assertIn("nothing — measured irreducible", rendered)
        # 25.0 + 26.0 + 1.8 = 52.8 ms of the 102 ms phase.
        self.assertIn("**52.80 ms**", rendered)
        self.assertIn("65300 leaves", rendered)

    def test_anchor_split_flags_memo_hits_that_would_dilute_the_mean(self):
        rendered = "\n".join(build_anchor_split_section([anchor_record(memo_hits=1)]))

        self.assertIn("answered from the cache-root memo", rendered)

    def test_clone_split_separates_size_proportional_copies(self):
        rendered = "\n".join(build_clone_split_section([anchor_record()]))

        self.assertIn("Transactional trie clone split", rendered)
        # 5.0 + 7.0 + 4.0 = 16 ms that is not the account trie.
        self.assertIn("**16.00 ms**", rendered)

    def test_composition_section_reports_per_entry_coefficients(self):
        rendered = "\n".join(build_cache_composition_section([anchor_record()]))

        self.assertIn("30100 / 32900 / 2300", rendered)
        # 102_000 us over 63_000 cached entries.
        self.assertIn("1.619 µs", rendered)

    def test_composition_section_combines_the_anchor_with_its_index_maintenance(self):
        """The gate reads the sum, so the sum is what the report has to state.

        Neither phase alone answers whether the index paid for itself: it makes the anchor cheaper
        by making the cache update more expensive. A record predating the field must read as zero
        maintenance rather than break, so an old run stays comparable against a new one.
        """
        record = anchor_record()
        record["partial"]["cache_root_index_maintenance_us"] = 12_600
        rendered = "\n".join(build_cache_composition_section([record]))

        # 102_000 + 12_600 us over 63_000 cached entries.
        self.assertIn("**114.60 ms**", rendered)
        self.assertIn("1.819 µs", rendered)
        # 30_000 us of cache update, less the 12_600 measured inside it.
        self.assertIn("**17.40 ms**", rendered)

        without = "\n".join(build_cache_composition_section([anchor_record()]))
        self.assertIn("**102.00 ms**", without)

    def test_schema_four_records_emit_no_v5_sections(self):
        legacy = [partial_record()]

        self.assertEqual(build_anchor_split_section(legacy), [])
        self.assertEqual(build_clone_split_section(legacy), [])
        self.assertEqual(build_cache_composition_section(legacy), [])


class SchemaSixCacheDeltaTest(unittest.TestCase):
    """The delta section is what sizes a leaf digest memo, so its arithmetic is pinned."""

    def test_reuse_share_counts_refreshes_and_excludes_evictions(self):
        record = anchor_record()
        record["partial"]["cache_delta"] = {
            "accounts_added": 100, "accounts_refreshed": 1_400, "accounts_evicted": 90,
            "storage_added": 300, "storage_refreshed": 3_000, "storage_evicted": 280,
            "codes_added": 10, "codes_refreshed": 120, "codes_evicted": 8,
        }
        rendered = "\n".join(build_cache_delta_section([record]))

        # Accounts: (100 + 1400) of 30_100 invalidated, so 95.0% reuse. Evictions are not in it.
        self.assertIn("1500.0 (5.0%)", rendered)
        self.assertIn("**95.0%**", rendered)
        # Weighted: (1500 + 3300 + 130) of (30_100 + 32_900 + 2_300) = 4930 / 65_300.
        self.assertIn("**92.5%**", rendered)
        self.assertIn("4930 of 65300", rendered)

    def test_schema_five_records_emit_no_delta_section(self):
        self.assertEqual(build_cache_delta_section([anchor_record()]), [])


if __name__ == "__main__":
    unittest.main()
