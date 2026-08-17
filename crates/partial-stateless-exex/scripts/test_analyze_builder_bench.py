#!/usr/bin/env python3
"""Schema-compatibility tests for the builder benchmark's artifact section."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_builder_bench import build_artifact_section


def record(schema_version, **updates):
    """A builder record, carrying the delivery fields only when the schema is meant to have them.

    The version argument names the *label*; `delivery=` decouples it, because one window of real
    runs declares 3 while carrying every schema-4 field.
    """
    fields = {
        "schema_version": schema_version,
        "builder_total_us": 2_000_000,
        "historical_full_db_evm_us": 0,
        "artifact_reused": True,
    }
    if updates.pop("delivery", schema_version >= 4):
        fields.update(
            {"artifact_available": True, "shadow_sampled": False, "fallback_reason": None}
        )
    fields.update(updates)
    return fields


def sampled(schema_version, evm_us):
    return record(
        schema_version,
        artifact_reused=False,
        shadow_sampled=True,
        fallback_reason="shadow_sampled",
        historical_full_db_evm_us=evm_us,
    )


class ArtifactSectionSchemaTest(unittest.TestCase):
    def test_schema_three_reports_reuse_and_refuses_to_infer_delivery(self):
        # The failure this guards: schema 3 has no `artifact_available`, and reading its absence
        # as False turns a run with a perfect handoff into a 0% delivery report.
        body = "\n".join(build_artifact_section([record(3) for _ in range(4)]))

        self.assertIn("Artifact reused: **4/4**", body)
        self.assertIn("not recorded", body)
        self.assertNotIn("Artifact delivered: ", body)

    def test_schema_four_separates_delivery_from_reuse(self):
        records = [record(4) for _ in range(9)] + [sampled(4, 100_000)]
        body = "\n".join(build_artifact_section(records))

        self.assertIn("Artifact delivered: **10/10** (100.00%)", body)
        self.assertIn("Artifact reused: **9/10** (90.00%)", body)
        self.assertIn("`shadow_sampled`: **1**", body)

    def test_mixed_records_are_declared_and_excluded_from_the_rates(self):
        records = [record(3), record(4), record(4), sampled(4, 100_000)]
        body = "\n".join(build_artifact_section(records))

        self.assertIn("Mixed records: 1 of 4", body)
        self.assertIn("Artifact delivered: **3/3**", body)

    def test_the_pre_bump_window_is_read_from_its_fields_and_flagged(self):
        # Runs written after the delivery fields landed but before the version was raised. The
        # numbers are there; only the label is stale, and refusing them would discard the very
        # measurements the engine-access claim rests on.
        records = [record(3, delivery=True) for _ in range(9)]
        records.append(
            record(
                3,
                delivery=True,
                artifact_reused=False,
                shadow_sampled=True,
                fallback_reason="shadow_sampled",
                historical_full_db_evm_us=100_000,
            )
        )
        body = "\n".join(build_artifact_section(records))

        self.assertIn("Artifact delivered: **10/10** (100.00%)", body)
        self.assertIn("predates the bump to 4", body)
        self.assertNotIn("not recorded", body)

    def test_avoided_cpu_is_labelled_an_extrapolation_and_carries_an_interval(self):
        # Two reused blocks per sampled block, and the sampled costs spread widely enough that
        # the interval cannot collapse: the point estimate alone would hide that spread.
        records = [record(4) for _ in range(6)]
        records += [sampled(4, us) for us in (20_000, 100_000, 480_000)]
        body = "\n".join(build_artifact_section(records))

        self.assertIn("Estimated avoided execution CPU: **1.2 s**", body)
        self.assertIn("95% bootstrap CI", body)
        self.assertIn("extrapolating that mean to 6 reused blocks", body)
        self.assertIn("estimate, not a measurement", body)

    def test_a_run_without_artifact_fields_emits_nothing(self):
        self.assertEqual(build_artifact_section([{"schema_version": 2}]), [])


if __name__ == "__main__":
    unittest.main()
