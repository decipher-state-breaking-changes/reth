#!/usr/bin/env python3
"""Compatibility tests for validation benchmark retention telemetry."""

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_validation_bench import (
    build_account_copy_section,
    build_admission_section,
    build_anchor_split_section,
    build_cache_composition_section,
    build_cache_delta_section,
    build_clone_split_section,
    build_retention_split_section,
    build_storage_prune_split,
    build_walk_frontier_section,
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



class StoragePruneSplitTest(unittest.TestCase):
    """The split is what names the storage time the walk phases never covered."""

    def test_absent_when_the_records_predate_the_fields(self):
        record = {"partial": {"retention_storage_tries_us": 40_000}}
        self.assertEqual(build_storage_prune_split([record]), [])

    def test_residual_is_the_total_less_the_walk_the_copy_and_the_drop(self):
        record = {"partial": {
            "retention_storage_tries_us": 40_000,
            "retention_storage_trie_cow_us": 15_000,
            "retention_storage_trie_cow_copies": 164,
            "retention_storage_trie_drop_us": 3_000,
            "retention_storage_tries_dropped": 12,
            "retention_storage_trie_detail": {
                "input_us": 200, "traversal_us": 15_000,
                "mutation_us": 500, "finalization_us": 7_900,
            },
        }}
        body = "\n".join(build_storage_prune_split([record]))
        self.assertIn("| Copy-on-write copy before the walk | 15.00 ms |", body)
        self.assertIn("| Witness-path walk | 23.60 ms |", body)
        self.assertIn("| Releasing no-longer-retained tries | 3.00 ms |", body)
        # 40.00 - 23.60 - 15.00 - 3.00, clamped at zero rather than reported negative.
        self.assertIn("| Storage-trie map scan (residual) | 0.00 ms |", body)
        self.assertIn("**164 / 12**", body)

    def test_the_copy_outweighing_the_walk_is_called_out(self):
        record = {"partial": {
            "retention_storage_tries_us": 30_000,
            "retention_storage_trie_cow_us": 20_000,
            "retention_storage_trie_detail": {"traversal_us": 5_000},
        }}
        body = "\n".join(build_storage_prune_split([record]))
        self.assertIn("costs more than the walk it precedes", body)


class RetentionYieldTest(unittest.TestCase):
    """The walk's cost-to-output ratio is the case for replacing it, and no timer shows it."""

    def test_nodes_walked_per_node_blinded_is_reported(self):
        record = partial_record(retention_account_trie_detail={
            "nodes_visited": 100_000, "nodes_converted": 200,
            "finalization_branch_masks_scanned": 68_000,
        })
        body = "\n".join(build_retention_split_section([record]))
        self.assertIn("**100000** nodes walked to blind **200** (500:1)", body)
        self.assertIn("scanned **68000** branch masks", body)


class WalkFrontierTest(unittest.TestCase):
    """The frontier's ceiling and the finalization mechanism are both decided by these counts."""

    WALK_DETAILS = [("Account", "retention_account_trie_detail")]

    @staticmethod
    def detail(**updates):
        base = {
            "nodes_visited": 160_000, "nodes_converted": 365,
            "visits_on_productive_path": 2_400, "productive_path_us": 900,
            "finalization_us": 21_710, "finalization_masks_us": 14_000,
            "finalization_maps_us": 7_500, "finalization_subtries_us": 210,
            "finalization_branch_masks_scanned": 100_735,
            "finalization_masks_removed": 400,
            "finalization_masks_removed_without_node": 0,
            "finalization_nodes_removed": 1_200,
            "finalization_upper_roots": 0,
            "finalization_lower_subtries_with_roots": 180,
            "calls": 1,
        }
        base.update(updates)
        return {"partial": {"retention_account_trie_detail": base}}

    def test_partitioning_is_normalized_per_call(self):
        """The storage row aggregates every trie pruned in the block, so a per-block sum lies.

        Against a 256-slot denominator, summing 175 pruned tries reads as more subtries than the
        map has. Per call is the figure that answers what a per-subtrie mask partition would skip.
        """
        one = "\n".join(build_walk_frontier_section([self.detail()], self.WALK_DETAILS))
        self.assertIn("**180.0** of 256 lower subtries", one)
        self.assertIn("(70% of a per-subtrie mask map", one)

        many = "\n".join(build_walk_frontier_section(
            [self.detail(calls=175, finalization_lower_subtries_with_roots=420)],
            self.WALK_DETAILS))
        # 420 across 175 calls is 2.4 subtries per trie, not 420 of 256.
        self.assertIn("**2.4** of 256 lower subtries", many)
        self.assertIn("(1% of a per-subtrie mask map", many)

    def test_the_obligatory_share_is_reported_as_a_ceiling(self):
        body = "\n".join(build_walk_frontier_section([self.detail()], self.WALK_DETAILS))

        # 2,400 of 160,000 visits reached something, leaving 157,600 as the ceiling.
        self.assertIn("**2400** of **160000** visits", body)
        self.assertIn("(1.5%)", body)
        self.assertIn("**157600** are the ceiling", body)
        self.assertIn("those visits also prove exclusion", body)

    def test_finalization_is_split_by_map(self):
        body = "\n".join(build_walk_frontier_section([self.detail()], self.WALK_DETAILS))

        self.assertIn("**14.00 ms** branch-mask scan", body)
        self.assertIn("**7.50 ms** node and value maps", body)
        self.assertIn("of **21.71 ms**", body)

    def test_an_orphaned_mask_rules_out_descendant_enumeration(self):
        clean = "\n".join(build_walk_frontier_section([self.detail()], self.WALK_DETAILS))
        self.assertIn("descendant enumeration is admissible", clean)

        orphaned = "\n".join(build_walk_frontier_section(
            [self.detail(finalization_masks_removed_without_node=12)], self.WALK_DETAILS))
        self.assertIn("would leave them behind", orphaned)
        self.assertIn("needs a mask map with a prefix range", orphaned)

    def test_instrumentation_is_called_out_for_subtraction(self):
        body = "\n".join(build_walk_frontier_section([self.detail()], self.WALK_DETAILS))
        self.assertIn("**0.90 ms** per block", body)

    def test_pre_v9_records_emit_nothing(self):
        self.assertEqual(
            build_walk_frontier_section([self.detail(visits_on_productive_path=0)],
                                        self.WALK_DETAILS),
            [])


class AccountCopyDecompositionTest(unittest.TestCase):
    """The copy is the largest phase, and the report has to say which field it is in."""

    @staticmethod
    def record(**detail):
        base = {
            "nodes_us": 80_000, "values_us": 20_000, "masks_us": 6_000,
            "buffers_us": 1_000, "rest_us": 3_000,
            "accounting_us": 1_500, "branch_hash_probe_us": 24_000,
            "total_bytes": 419_430_400, "nodes_bytes": 314_572_800,
            "values_bytes": 83_886_080, "masks_bytes": 10_485_760,
            "buffers_bytes": 1_048_576, "rest_bytes": 9_437_184,
            "branch_hash_bytes": 209_715_200, "branch_hash_allocs": 409_600,
            "total_allocs": 500_000, "nodes_allocs": 409_665, "values_allocs": 90_335,
            "subtries": 256, "node_entries": 600_000, "branch_nodes": 409_600,
            "extension_nodes": 90_400, "leaf_nodes": 100_000,
            "value_entries": 90_334, "mask_entries": 68_000,
        }
        base.update(detail)
        return {"partial": {"trie_clone_detail": {"account_trie_detail": base}}}

    def test_components_are_ranked_with_bytes_and_allocations(self):
        body = "\n".join(build_account_copy_section([self.record()]))

        self.assertIn("Account-trie copy decomposition", body)
        # 80 + 20 + 6 + 1 + 3 = 110 ms, of which the node maps are 72.7%.
        self.assertIn("| Node maps | 80.00 ms | 72.7%", body)
        self.assertIn("**110.00 ms**", body)
        self.assertIn("409,600", body)

    def test_the_branch_hash_box_is_priced_when_the_probe_ran(self):
        body = "\n".join(build_account_copy_section([self.record()]))

        # 200 MiB of the 400 MiB copy, in 409,600 of its 500,000 allocations.
        self.assertIn("200.00 MiB** (50.0% of the copy's bytes)", body)
        self.assertIn("(81.9% of the copy's allocations)", body)
        self.assertIn("**24.00 ms** per block (21.8% of the copy)", body)
        # 1.5 ms accounting + 24 ms probe, which a cross-run comparison has to remove.
        self.assertIn("**25.50 ms** per block", body)

    def test_a_run_without_the_probe_reports_size_without_cost(self):
        body = "\n".join(build_account_copy_section([self.record(branch_hash_probe_us=0)]))

        self.assertIn("the box has a size but no cost", body)
        self.assertIn("200.00 MiB** (50.0% of the copy's bytes)", body)

    def test_a_run_without_the_census_reports_times_and_says_so(self):
        """The timers are free and always run; the census walks the copy and is opt-in.

        Printing its zeroes in the byte column would read as a trie that holds no bytes rather
        than as a run that did not ask for the walk.
        """
        bare = {field: 0 for field in (
            "total_bytes", "nodes_bytes", "values_bytes", "masks_bytes", "buffers_bytes",
            "rest_bytes", "branch_hash_bytes", "total_allocs", "nodes_allocs", "values_allocs",
            "branch_hash_allocs", "accounting_us", "branch_hash_probe_us", "subtries",
            "node_entries", "branch_nodes", "extension_nodes", "leaf_nodes", "value_entries",
            "mask_entries")}
        body = "\n".join(build_account_copy_section([self.record(**bare)]))

        self.assertIn("| Node maps | 80.00 ms | 72.7% |", body)
        self.assertIn("**110.00 ms**", body)
        self.assertIn("did not request the census", body)
        self.assertIn("PS_TRIE_SHAPE_DIAGNOSTICS=probe", body)
        self.assertNotIn("MiB", body)
        self.assertNotIn("branch-hash box", body)

    def test_pre_v9_records_emit_nothing(self):
        self.assertEqual(build_account_copy_section([anchor_record()]), [])
        self.assertEqual(build_account_copy_section([partial_record()]), [])



class AdmissionSectionTest(unittest.TestCase):
    """A null admission phase is not a zero-cost one, and the report has to keep them apart."""

    @staticmethod
    def record(admission):
        return {"partial": {"admission": admission}, "weak": {"admission": admission}}

    def test_a_recovered_only_run_says_so_instead_of_tabulating_zeros(self):
        recovered = {
            "source": "recovered",
            "input_decode_us": None,
            "payload_validation_us": None,
            "sender_recovery_us": None,
            "pre_execution_consensus_us": None,
        }
        lines = build_admission_section([self.record(recovered)] * 3)

        text = "\n".join(lines)
        self.assertIn("recovered", text)
        self.assertNotIn("| Sender recovery |", text)

    def test_averages_are_taken_only_over_samples_that_performed_the_phase(self):
        performed = {
            "source": "execution_data",
            "input_decode_us": None,
            "payload_validation_us": 2000,
            "sender_recovery_us": 4000,
            "pre_execution_consensus_us": 0,
        }
        skipped = {
            "source": "recovered",
            "input_decode_us": None,
            "payload_validation_us": None,
            "sender_recovery_us": None,
            "pre_execution_consensus_us": None,
        }
        lines = build_admission_section(
            [self.record(performed), self.record(skipped)]
        )

        text = "\n".join(lines)
        # One of two samples recovered senders, at 4 ms. Counting the other as zero would report
        # 2 ms and describe a validator that recovers senders twice as fast as it does.
        self.assertIn("| Sender recovery | 1 | 4.00 ms |", text)
        # Performed and free is a real answer, and must not read as absent.
        self.assertIn("| Pre-execution consensus | 1 | 0.00 ms |", text)
        # Never performed by anything in the corpus.
        self.assertIn("| Input decode | 0 | not performed |", text)


class ReportStructureTest(unittest.TestCase):
    """The report builders must not fall off the end, and must not orphan their own tails.

    Both halves of this failed at once when `build_admission_section` was defined *inside*
    `build_report`: the report function ended at the call and returned `None`, and everything
    after the new helper's `return` — the retention, anchor, clone, composition, and workload
    sections plus the real return — became unreachable. Every unit test still passed, because
    each section builder was exercised directly and none of them goes through `build_report`.
    A run would have failed only at `report_path.write_text(None)`, after collecting samples.
    """

    @staticmethod
    def module_ast():
        source = Path(__file__).resolve().parent / "analyze_validation_bench.py"
        return ast.parse(source.read_text())

    def test_no_report_builder_has_code_after_a_return(self):
        offenders = []
        for node in ast.walk(self.module_ast()):
            if not isinstance(node, ast.FunctionDef):
                continue
            for index, statement in enumerate(node.body[:-1]):
                if isinstance(statement, ast.Return):
                    offenders.append(f"{node.name}: line {node.body[index + 1].lineno}")
        self.assertEqual(offenders, [], f"unreachable code after a return: {offenders}")

    def test_build_report_ends_in_a_return(self):
        report = next(
            node
            for node in ast.walk(self.module_ast())
            if isinstance(node, ast.FunctionDef) and node.name == "build_report"
        )
        self.assertIsInstance(
            report.body[-1],
            ast.Return,
            "build_report fell off the end, so it returns None and the run fails at write time",
        )


if __name__ == "__main__":
    unittest.main()
