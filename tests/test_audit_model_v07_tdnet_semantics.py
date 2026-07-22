from __future__ import annotations

import json
from pathlib import Path
import unittest

import pandas as pd

from research.audit_model_v07_tdnet_semantics import (
    DEFAULT_OUTPUT,
    IMPLEMENTATION_PATHS,
    ROOT,
    cache_inventory,
    empty_and_completeness_probe,
    sha256_file,
    stable_frame_sha256,
    synthetic_semantics_probe,
)


class TdnetSemanticAuditUnitTests(unittest.TestCase):
    def test_synthetic_semantics_and_completeness_contracts(self) -> None:
        semantic = synthetic_semantics_probe()
        completeness = empty_and_completeness_probe()

        self.assertTrue(semantic["passed"], semantic["checks"])
        self.assertTrue(completeness["passed"], completeness["checks"])
        self.assertTrue(
            completeness[
                "incomplete_source_timestamp_retained_for_provenance"
            ]
        )

    def test_stable_frame_hash_binds_values_and_dtypes(self) -> None:
        baseline = pd.DataFrame({"value": pd.Series([1.0], dtype="float32")})
        changed_value = pd.DataFrame(
            {"value": pd.Series([2.0], dtype="float32")}
        )
        changed_dtype = pd.DataFrame(
            {"value": pd.Series([1.0], dtype="float64")}
        )

        self.assertEqual(
            stable_frame_sha256(baseline),
            stable_frame_sha256(baseline.copy()),
        )
        self.assertNotEqual(
            stable_frame_sha256(baseline), stable_frame_sha256(changed_value)
        )
        self.assertNotEqual(
            stable_frame_sha256(baseline), stable_frame_sha256(changed_dtype)
        )


class TdnetSemanticAuditArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result_path = Path(DEFAULT_OUTPUT)
        cls.manifest_path = cls.result_path.with_suffix(".manifest.json")
        cls.report = json.loads(cls.result_path.read_text(encoding="utf-8"))
        cls.manifest = json.loads(
            cls.manifest_path.read_text(encoding="utf-8")
        )

    def test_result_is_hash_bound_and_implementation_is_current(self) -> None:
        self.assertEqual(
            self.manifest["result_sha256"], sha256_file(self.result_path)
        )
        self.assertEqual(
            self.manifest["audit_script_sha256"],
            sha256_file(ROOT / self.manifest["audit_script_path"]),
        )
        bound_files = self.report["bindings"]["implementation"]["files"]
        self.assertEqual(
            {record["path"] for record in bound_files},
            {str(path.relative_to(ROOT)) for path in IMPLEMENTATION_PATHS},
        )
        for record in bound_files:
            self.assertEqual(
                record["sha256"], sha256_file(ROOT / record["path"])
            )
        panel = self.report["bindings"]["panel_manifest"]
        self.assertEqual(panel["sha256"], sha256_file(ROOT / panel["path"]))

    def test_frozen_corpus_schema_and_invariants_are_recorded(self) -> None:
        self.assertEqual(self.report["checks"]["failed"], 0)
        self.assertEqual(
            self.report["verdict"]["computational_integrity"], "pass"
        )
        self.assertEqual(self.report["corpus"]["documents"], 97_006)
        self.assertEqual(self.report["corpus"]["bundles"], 66_702)
        self.assertEqual(
            self.report["candidate_schema"]["candidate_column_count"], 63
        )
        self.assertTrue(
            all(
                detail["violation_count"] == 0
                for detail in self.report["invariants"].values()
            )
        )
        bundle = self.report["distributions"]["bundle"]
        self.assertEqual(
            bundle["semantic_indicator_sums"]["tdnet_v07_has_fresh_buyback"],
            1_712,
        )
        self.assertEqual(
            bundle["semantic_indicator_sums"]["tdnet_v07_has_followup_buyback"],
            6_090,
        )
        self.assertEqual(
            bundle["semantic_indicator_sums"][
                "tdnet_v07_fresh_classified_economic_any"
            ],
            29_282,
        )

    def test_reproducibility_and_input_constraints_are_explicit(self) -> None:
        reproducibility = self.report["reproducibility"]
        self.assertTrue(reproducibility["input_shuffle"]["bit_exact"])
        self.assertTrue(reproducibility["legacy_bit_exact"]["bit_exact"])
        self.assertTrue(reproducibility["empty_and_completeness"]["passed"])
        self.assertTrue(reproducibility["synthetic_semantics"]["passed"])
        self.assertEqual(self.report["pit"]["violation_count"], 0)

        inventory = self.report["input_inventory"]
        self.assertEqual(inventory["html_files"], 461)
        self.assertEqual(inventory["metadata_files"], 461)
        self.assertEqual(inventory["total_files"], 922)
        self.assertEqual(inventory["location_kind"], "gitignored_repository_cache")
        self.assertFalse(inventory["temporary_only"])
        self.assertIn("excluded by .gitignore", inventory["reexecution_constraint"])

        cache_path = ROOT / inventory["root"]
        if cache_path.is_dir():
            current = cache_inventory(cache_path)
            self.assertEqual(
                current["aggregate_sha256"], inventory["aggregate_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
