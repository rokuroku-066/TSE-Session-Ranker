from __future__ import annotations

import unittest

from research.audit_model_v08_zero_base import (
    EXPECTED_HYPOTHESIS_COUNTS,
    EXPECTED_SHADOW_VALUES,
    ROOT,
    audit_artifacts,
)


class ModelV08ArtifactAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = audit_artifacts(ROOT)

    def test_canonical_protocol_result_manifest_chains_pass(self) -> None:
        failures = self.report["failures"]
        self.assertTrue(
            self.report["passed"],
            f"model-v0.8 artifact audit failures: {failures}",
        )
        for family in ("target", "feature", "tdnet"):
            with self.subTest(family=family):
                self.assertTrue(
                    self.report["checks"][f"{family}.result_binds_protocol"][
                        "passed"
                    ]
                )
                self.assertTrue(
                    self.report["checks"][f"{family}.manifest_binds_protocol"][
                        "passed"
                    ]
                )
                self.assertTrue(
                    self.report["checks"][f"{family}.manifest_binds_result"][
                        "passed"
                    ]
                )
        for check in (
            "target.stable_registration_set",
            "tdnet.stable_registration_set",
        ):
            with self.subTest(check=check):
                self.assertTrue(self.report["checks"][check]["passed"])

    def test_all_43_new_feature_hypotheses_and_five_universes_are_reported(self) -> None:
        self.assertEqual(
            self.report["hypothesis_counts"], EXPECTED_HYPOTHESIS_COUNTS
        )
        for check in (
            "target.stage1_registered_candidates_reported",
            "target.stage2_registered_candidates_reported",
            "target.stage3_registered_candidates_reported",
            "target.stage4_registered_candidates_reported",
            "feature.registered_groups_reported",
            "feature.registered_universes_reported",
            "tdnet.registered_groups_reported",
            "tdnet.generation4_registered_recipes_reported",
        ):
            with self.subTest(check=check):
                self.assertTrue(self.report["checks"][check]["passed"])

    def test_shadow_values_rejections_and_stop_rules_are_locked(self) -> None:
        for candidate_id, expected in EXPECTED_SHADOW_VALUES.items():
            actual = self.report["shadow_roles"][candidate_id]
            with self.subTest(candidate_id=candidate_id):
                self.assertAlmostEqual(actual["net20"], expected["net20"], places=12)
                self.assertAlmostEqual(actual["net40"], expected["net40"], places=12)
                self.assertAlmostEqual(
                    actual["max_t_lower"], expected["max_t_lower"], places=12
                )
                self.assertLess(actual["max_t_lower"], 0.0)

        for check in (
            "target.protocol_stop_rule",
            "target.result_stop_rule",
            "feature.all_general_max_t_lowers_negative",
            "feature.discovery_feature_survivors_empty",
            "feature.discovery_universe_survivors_empty",
            "feature.alpha1_transfer_rejected",
            "tdnet.all_generation3_candidates_rejected",
            "tdnet.all_generation4_candidates_rejected",
            "tdnet.all_generation3_max_t_lowers_negative",
            "tdnet.coverage_below_qualification_floor",
            "tdnet.stop_rule_triggered",
            "tdnet.title_only_value_not_demonstrated",
        ):
            with self.subTest(check=check):
                self.assertTrue(self.report["checks"][check]["passed"])


if __name__ == "__main__":
    unittest.main()
