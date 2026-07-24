#!/usr/bin/env python3
"""Integrity checks for the amended T02 prospective shadow protocol."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
PROTOCOL_PATH = RESEARCH / "model_v11_t02_forward_protocol_v2.json"
PLAN_PATH = RESEARCH / "model_v11_t02_next_validation.md"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ModelV11T02ForwardProtocolV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = read_json(PROTOCOL_PATH)

    def test_historical_protocol_is_preserved_and_hash_bound(self) -> None:
        authority = self.protocol["authority"]
        superseded = authority["supersedes_for_future_observations_only"]
        old_path = ROOT / superseded["path"]

        self.assertTrue(authority["historical_artifacts_immutable"])
        self.assertTrue(old_path.is_file())
        self.assertEqual(sha256_file(old_path), superseded["sha256"])

    def test_all_bound_candidate_artifacts_are_hash_exact(self) -> None:
        for name, artifact in self.protocol["bound_candidate_artifacts"].items():
            with self.subTest(artifact=name):
                path = ROOT / artifact["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(sha256_file(path), artifact["sha256"])

    def test_outcome_erratum_fixes_direction_without_changing_candidate(self) -> None:
        candidate = self.protocol["candidate"]
        evaluation = self.protocol["evaluation"]
        errata = self.protocol["errata"]

        self.assertEqual(candidate["id"], "v10_t02_char_value_event_top1")
        self.assertEqual(
            candidate["target"]["name"],
            "open_to_close_return_pct",
        )
        self.assertEqual(
            evaluation["outcome"],
            "target-session open-to-close return",
        )
        self.assertEqual(
            evaluation["outcome_field"],
            "open_to_close_return_pct",
        )
        self.assertEqual(len(errata), 1)
        self.assertEqual(
            errata[0]["recorded_value"],
            "target-session close-to-open return",
        )
        self.assertEqual(
            errata[0]["correct_value"],
            "target-session open-to-close return",
        )

    def test_model_and_decision_parameters_remain_frozen(self) -> None:
        candidate = self.protocol["candidate"]
        representation = candidate["representation"]
        model = candidate["model"]
        decision = candidate["decision"]

        self.assertEqual(representation["analyzer"], "char")
        self.assertEqual(representation["ngram_range"], [2, 5])
        self.assertEqual(representation["min_df"], 3)
        self.assertEqual(representation["max_features"], 30000)
        self.assertIs(representation["sublinear_tf"], True)
        self.assertEqual(representation["norm"], "l2")
        self.assertEqual(model, {"type": "Ridge", "alpha": 20.0, "solver": "lsqr"})
        self.assertEqual(decision["portfolio_size"], 1)
        self.assertEqual(decision["no_qualifying_event"], "cash")
        self.assertIsNone(decision["fallback"])

    def test_monthly_expanding_retraining_is_explicit_and_fail_closed(self) -> None:
        training = self.protocol["training_contract"]

        self.assertEqual(training["mode"], "monthly_expanding_walk_forward")
        self.assertIs(training["intramonth_refit_allowed"], False)
        self.assertIs(training["future_or_target_month_outcomes_allowed"], False)
        self.assertIs(training["refit_vectorizer_each_fold"], True)
        self.assertIs(training["refit_estimator_each_fold"], True)
        self.assertIs(training["hyperparameter_changes_allowed"], False)
        self.assertIs(training["changes_reset_counter"], True)
        self.assertIn("Fail closed", training["failure_policy"])

        required = set(training["required_fold_artifacts"])
        self.assertTrue(
            {
                "training_row_identity_sha256",
                "source_manifest_sha256",
                "vectorizer_vocabulary_sha256",
                "vectorizer_idf_sha256",
                "ridge_coef_sha256",
                "prediction_artifact_sha256",
                "scikit_learn_version",
            }.issubset(required)
        )

    def test_activation_cannot_be_retroactive(self) -> None:
        activation = self.protocol["activation"]

        self.assertEqual(activation["status"], "pending_hash_bound_activation")
        self.assertIsNone(activation["start_session"])
        self.assertIs(activation["activation_manifest_required"], True)
        self.assertIn("Never backfill", activation["pre_activation_sessions"])
        self.assertIn("strictly after", activation["eligibility_rule"])

        required = set(activation["activation_manifest_required_fields"])
        self.assertTrue(
            {
                "protocol_sha256",
                "activation_commit_sha",
                "activated_at",
                "first_eligible_session",
                "training_snapshot_sha256",
                "source_registry_sha256",
                "scoring_code_sha256",
                "activation_approved_by",
            }.issubset(required)
        )

    def test_pit_and_decision_logs_are_fail_closed_and_append_only(self) -> None:
        pit = self.protocol["point_in_time_contract"]
        log = self.protocol["decision_log_contract"]

        self.assertIs(pit["published_at_must_be_at_or_before_cutoff"], True)
        self.assertIs(pit["received_at_must_be_at_or_before_cutoff"], True)
        self.assertIs(pit["computed_at_must_be_at_or_before_cutoff"], True)
        self.assertIn("never a no-event day", pit["missing_source_policy"])
        self.assertIs(pit["append_only_prediction_log"], True)
        self.assertIn("rejected", log["immutability"])

        required = set(log["required_fields"])
        self.assertTrue(
            {
                "protocol_sha256",
                "fold_artifact_sha256",
                "source_manifest_sha256",
                "decision_at",
                "source_complete",
                "decision",
                "decision_payload_sha256",
            }.issubset(required)
        )

    def test_protocol_does_not_authorize_production_or_orders(self) -> None:
        authority = self.protocol["authority"]
        gate = self.protocol["promotion_gate_after_window"]

        self.assertIs(authority["production_model_changed"], False)
        self.assertIs(
            authority["production_promotion_allowed_during_run"],
            False,
        )
        self.assertIs(authority["orders_allowed"], False)
        self.assertIn("not an order authorization", gate["production_action"])

    def test_human_readable_plan_points_to_the_amended_protocol(self) -> None:
        plan = PLAN_PATH.read_text(encoding="utf-8")

        self.assertIn("model_v11_t02_forward_protocol_v2.json", plan)
        self.assertIn("始値→終値", plan)
        self.assertIn("monthly expanding walk-forward", plan)
        self.assertIn("過去sessionはbackfillしない", plan)
        self.assertIn("orders_allowed=false", plan)


if __name__ == "__main__":
    unittest.main()
