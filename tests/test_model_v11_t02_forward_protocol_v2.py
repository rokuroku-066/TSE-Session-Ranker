#!/usr/bin/env python3
"""Integrity checks for the amended T02 prospective shadow protocol."""

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
PROTOCOL_PATH = RESEARCH / "model_v11_t02_forward_protocol_v2.json"
PLAN_PATH = RESEARCH / "model_v11_t02_next_validation.md"
RUNTIME_PATH = ROOT / "src" / "tse_session_ranker" / "t02_forward.py"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_decision_payload_keys() -> set[str]:
    """Return literal keys placed in `_decision_payload` before self-hashing."""

    tree = ast.parse(RUNTIME_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_decision_payload"
    )
    for node in ast.walk(function):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "payload"
            and isinstance(node.value, ast.Dict)
        ):
            keys = {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant)
                and isinstance(key.value, str)
            }
            if len(keys) != len(node.value.keys):
                raise AssertionError(
                    "_decision_payload must use literal unique string keys"
                )
            return keys
    raise AssertionError("could not find _decision_payload literal")


class ModelV11T02ForwardProtocolV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = read_json(PROTOCOL_PATH)
        cls.plan = PLAN_PATH.read_text(encoding="utf-8")

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

    def test_outcome_and_oot_source_errata_are_explicit(self) -> None:
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
        by_pointer = {
            (item["source_path"], item["json_pointer"]): item
            for item in errata
        }
        outcome = by_pointer[
            ("research/model_v10_forward_protocol.json", "/evaluation/outcome")
        ]
        self.assertEqual(
            outcome["recorded_value"],
            "target-session close-to-open return",
        )
        self.assertEqual(
            outcome["correct_value"],
            "target-session open-to-close return",
        )
        source = by_pointer[
            ("research/model_v11_t02_oot_protocol.json", "/candidate/decision")
        ]
        self.assertIn("incomplete source means cash", source["recorded_value"])
        self.assertIn("never classified as no-event", source["correct_value"])
        self.assertIn("historical OOT only", source["interpretation"])

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
        self.assertEqual(representation["dtype"], "float32")
        self.assertEqual(model, {"type": "Ridge", "alpha": 20.0, "solver": "lsqr"})
        self.assertEqual(decision["portfolio_size"], 1)
        self.assertEqual(decision["no_qualifying_event"], "cash")
        self.assertIsNone(decision["fallback"])

    def test_frozen_runner_eligibility_masks_and_code_are_hash_bound(self) -> None:
        eligibility = self.protocol["candidate"]["eligibility_contract"]

        self.assertEqual(
            eligibility["scoring_mask"],
            "qualifying overnight event-code bundle AND price_eligible == true",
        )
        self.assertIn(
            "price_training_eligible == true",
            eligibility["training_mask"],
        )
        for key in (
            "retrospective_runner",
            "universe_config",
            "eligibility_code",
        ):
            with self.subTest(reference=key):
                reference = eligibility[key]
                path = ROOT / reference["path"]
                self.assertEqual(sha256_file(path), reference["sha256"])

        runner = (
            ROOT / eligibility["retrospective_runner"]["path"]
        ).read_text(encoding="utf-8")
        self.assertIn('panel["price_training_eligible"].eq(True)', runner)
        self.assertIn('panel["price_eligible"].eq(True)', runner)
        self.assertIn('" [SEP] ".join', runner)

    def test_monthly_fold_is_sealed_before_scoring_and_closed_after_month(self) -> None:
        training = self.protocol["training_contract"]
        hashing = self.protocol["hashing_contract"]

        self.assertEqual(training["mode"], "monthly_expanding_walk_forward")
        self.assertIs(training["intramonth_refit_allowed"], False)
        self.assertIs(training["future_or_target_month_outcomes_allowed"], False)
        self.assertIs(training["refit_vectorizer_each_fold"], True)
        self.assertIs(training["refit_estimator_each_fold"], True)
        self.assertIs(training["hyperparameter_changes_allowed"], False)
        self.assertIs(training["changes_reset_counter"], True)
        self.assertIn("Fail closed", training["failure_policy"])
        self.assertIn("price_training_eligible == true", training["training_sample"])

        pre_score = training["pre_score_fold_manifest"]
        required = set(pre_score["required_fields"])
        self.assertTrue(
            {
                "training_row_identity_sha256",
                "training_bundle_text_sha256",
                "training_target_sha256",
                "training_source_payload_sha256",
                "target_clip_lower_pct",
                "target_clip_upper_pct",
                "source_manifest_sha256",
                "scoring_code_sha256",
                "python_version",
                "canonical_json_contract",
                "vectorizer_vocabulary_sha256",
                "vectorizer_idf_sha256",
                "ridge_coef_sha256",
                "pre_score_fold_manifest_sha256",
                "scipy_version",
                "scikit_learn_version",
            }.issubset(required)
        )
        self.assertNotIn("prediction_artifact_sha256", required)
        self.assertIn(
            "prediction_artifact_sha256",
            pre_score["forbidden_fields"],
        )
        self.assertIn("before the first activated", pre_score["seal_rule"])
        self.assertEqual(hashing["id"], "project_canonical_json_v1")
        self.assertIn("ensure_ascii=False", hashing["serialization"])
        self.assertIn("sort_keys=True", hashing["serialization"])
        self.assertIn("allow_nan=False", hashing["serialization"])
        self.assertIs(hashing["not_rfc8785"], True)
        self.assertIn("project_canonical_json_v1", pre_score["canonical_hashing"])
        self.assertIn(
            "pre_score_fold_manifest_sha256 explicitly omitted",
            pre_score["manifest_hash_rule"],
        )

        closeout = training["month_closeout_manifest"]
        closeout_required = set(closeout["required_fields"])
        self.assertTrue(
            {
                "pre_score_fold_manifest_sha256",
                "python_version",
                "canonical_json_contract",
                "activated_scheduled_session_set_sha256",
                "decision_ledger_head_sha256",
                "prediction_artifact_sha256",
                "outcome_artifact_sha256",
                "month_closeout_manifest_sha256",
            }.issubset(closeout_required)
        )
        self.assertIn("after the score month closes", closeout["creation_rule"])
        self.assertIn(
            "month_closeout_manifest_sha256 explicitly omitted",
            closeout["manifest_hash_rule"],
        )

    def test_activation_uses_non_self_referential_payload_and_receipt(self) -> None:
        activation = self.protocol["activation"]

        self.assertEqual(activation["status"], "pending_payload_and_receipt")
        self.assertIsNone(activation["earliest_calendar_eligible_session"])
        self.assertIsNone(activation["first_counted_session"])
        self.assertIn("Never backfill", activation["pre_activation_sessions"])
        self.assertIn("first_counted_session is separately", activation["eligibility_rule"])
        self.assertIn(
            "must equal earliest_calendar_eligible_session",
            activation["eligibility_rule"],
        )
        self.assertIn("new payload and receipt", activation["eligibility_rule"])

        payload = activation["activation_payload"]
        payload_required = set(payload["required_fields"])
        self.assertTrue(
            {
                "protocol_sha256",
                "not_before_session",
                "training_snapshot_sha256",
                "source_registry_schema_sha256",
                "scoring_code_path",
                "scoring_code_sha256",
                "universe_config_sha256",
                "eligibility_code_sha256",
                "activation_approved_by",
                "python_version",
                "canonical_json_contract",
                "payload_sha256",
            }.issubset(payload_required)
        )
        self.assertNotIn("activation_commit_sha", payload_required)
        self.assertIn("must not contain its own Git commit SHA", payload["self_reference_rule"])

        receipt = activation["activation_receipt"]
        receipt_required = set(receipt["required_fields"])
        self.assertTrue(
            {
                "payload_sha256",
                "payload_default_branch_commit_sha",
                "default_branch_ref_observed_at",
                "default_branch_tip_sha_when_observed",
                "receipt_issued_at",
                "earliest_calendar_eligible_session",
                "canonical_json_contract",
                "canonicalizer_implementation",
                "canonicalizer_version",
                "receipt_sha256",
                "receipt_signature",
            }.issubset(receipt_required)
        )
        self.assertIn("ancestor", receipt["reachability_rule"])
        self.assertIn("not Git author_date", receipt["timestamp_rule"])
        self.assertNotIn(
            '"activation_commit_sha"',
            json.dumps(activation, sort_keys=True),
        )

    def test_live_cutoff_snapshot_and_final_archive_are_separated(self) -> None:
        pit = self.protocol["point_in_time_contract"]

        self.assertIs(pit["published_at_must_be_at_or_before_cutoff"], True)
        self.assertIs(pit["received_at_must_be_at_or_before_cutoff"], True)
        self.assertIs(pit["computed_at_must_be_at_or_before_cutoff"], True)
        live = pit["live_cutoff_snapshot"]
        archive = pit["post_close_final_archive_audit"]
        self.assertIn("only TDnet bytes", live["purpose"])
        self.assertIn("never called end-of-day final", live["finality_semantics"])
        self.assertTrue(
            any("watermark" in item for item in live["requirements"])
        )
        self.assertIn("without feeding final bytes", archive["purpose"])
        self.assertIn("never rewrite", archive["requirements"][-1])
        self.assertIn("PIT-noncompliant", archive["failure_action"])
        self.assertIn("never a no-event day", pit["missing_source_policy"])
        self.assertIn("fail_closed_source", pit["missing_source_policy"])
        self.assertIs(pit["append_only_prediction_log"], True)

    def test_decision_hash_is_canonical_non_self_referential_and_chained(self) -> None:
        log = self.protocol["decision_log_contract"]

        self.assertEqual(
            log["canonical_serialization"],
            "project_canonical_json_v1.",
        )
        self.assertIn("project-canonical JSON object", log["decision_id_rule"])
        self.assertIn("Exactly one decision_id", log["decision_id_rule"])
        self.assertIn(
            "exactly payload_field_allowlist",
            log["decision_payload_hash_rule"],
        )
        self.assertIn("previous_record_sha256", log["chain_rule"])
        self.assertIn("monotonically increasing", log["chain_rule"])
        required = set(log["required_fields"])
        self.assertTrue(
            {
                "schema_version",
                "candidate_id",
                "protocol_sha256",
                "activation_payload_sha256",
                "activation_receipt_sha256",
                "python_version",
                "canonical_json_contract",
                "pre_score_fold_manifest_sha256",
                "source_manifest_sha256",
                "live_source_snapshot_sha256",
                "price_eligibility_sha256",
                "decision_at",
                "source_complete",
                "decision",
                "failure_reason",
                "decision_payload_sha256",
                "sequence_number",
                "previous_record_sha256",
                "record_sha256",
            }.issubset(required)
        )
        self.assertIn("deletion", log["immutability"])
        self.assertIn("truncation", log["immutability"])
        self.assertIn("post-cutoff value", log["preopen_only_rule"])
        payload_fields = set(log["payload_field_allowlist"])
        envelope_fields = set(log["envelope_only_fields"])
        self.assertEqual(payload_fields, runtime_decision_payload_keys())
        self.assertEqual(
            required,
            payload_fields | envelope_fields,
        )
        self.assertTrue(payload_fields.isdisjoint(envelope_fields))
        self.assertNotIn("decision_payload_sha256", payload_fields)
        self.assertNotIn("record_sha256", payload_fields)
        self.assertEqual(
            envelope_fields,
            {
                "decision_payload_sha256",
                "sequence_number",
                "previous_record_sha256",
                "record_sha256",
            },
        )
        self.assertNotIn("gross_return_pct", required)
        self.assertNotIn("cost_bps_charged", required)
        source_rule = log["state_field_rules"]["fail_closed_source"]
        model_rule = log["state_field_rules"]["fail_closed_model"]
        self.assertIn("failure_reason is non-empty", source_rule)
        self.assertIn("price_eligibility_sha256 is null", source_rule)
        self.assertIn("mask itself is missing or invalid", model_rule)

        outcome = self.protocol["post_close_outcome_log_contract"]
        self.assertIn("never modifies", outcome["link_rule"])
        self.assertTrue(
            {
                "decision_payload_sha256",
                "python_version",
                "canonical_json_contract",
                "outcome_source_sha256",
                "official_open",
                "official_close",
                "open_to_close_return_pct",
                "net_return_pct_at_40bps",
                "outcome_payload_sha256",
            }.issubset(set(outcome["required_fields"]))
        )
        self.assertIn("gross return 0", outcome["state_accounting"])
        self.assertIn(
            "all three net return fields are 0",
            outcome["state_field_rules"]["fail_closed_source"],
        )
        self.assertIn(
            "outcome_payload_sha256 explicitly omitted",
            outcome["hash_rule"],
        )

    def test_all_scheduled_sessions_are_primary_denominator(self) -> None:
        candidate = self.protocol["candidate"]
        evaluation = self.protocol["evaluation"]

        self.assertIn(
            "all-scheduled-session primary denominator",
            candidate["decision"]["source_incomplete"],
        )
        self.assertIn(
            "Every activated scheduled JPX session",
            evaluation["scheduled_day_denominator"],
        )
        for state in (
            "cash_no_event",
            "fail_closed_source",
            "fail_closed_model",
        ):
            self.assertIn(state, evaluation["scheduled_day_denominator"])
        self.assertIn("gross return 0", evaluation["scheduled_day_denominator"])
        self.assertIn("diagnostic subset only", evaluation["source_complete_subset"])
        self.assertIn(
            "every activated scheduled JPX session",
            evaluation["primary_metric"],
        )

    def test_historical_oot_gate_is_a_hash_bound_prerequisite(self) -> None:
        gate = self.protocol["promotion_gate_after_window"]
        prerequisite = gate["historical_oot_prerequisite"]

        self.assertIs(prerequisite["required"], True)
        self.assertEqual(
            sha256_file(ROOT / prerequisite["protocol_path"]),
            prerequisite["protocol_sha256"],
        )
        self.assertEqual(
            prerequisite["required_status"],
            "all statistical_promotion_gate requirements passed without tuning",
        )
        self.assertTrue(
            {
                "historical_oot_result_sha256",
                "historical_oot_independent_audit_sha256",
            }.issubset(set(prerequisite["required_evidence"]))
        )
        self.assertTrue(
            any("historical OOT prerequisite" in item for item in gate["requirements"])
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

    def test_human_readable_plan_matches_the_amended_semantics(self) -> None:
        required_phrases = (
            "model_v11_t02_forward_protocol_v2.json",
            "始値→終値",
            "monthly expanding walk-forward",
            "activation payload",
            "timestamp receipt",
            "自己参照",
            "earliest_calendar_eligible_session",
            "first_counted_session",
            "price_training_eligible",
            "price_eligible",
            "pre-score fold",
            "closeout manifest",
            "全activated scheduled session",
            "live snapshot",
            "final archive",
            "project_canonical_json_v1",
            "hash chain",
            "過去sessionはbackfillしない",
            "orders_allowed=false",
        )
        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.plan)


if __name__ == "__main__":
    unittest.main()
