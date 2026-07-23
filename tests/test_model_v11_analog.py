#!/usr/bin/env python3
"""Integrity tests for the preregistered model-v11 analog experiment."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
PROTOCOL = RESEARCH / "model_v11_analog_protocol.json"
RESULT = RESEARCH / "model_v11_analog_result.json"
AUDIT = RESEARCH / "model_v11_analog_audit.json"
PICKS = RESEARCH / "model_v11_analog_picks.csv"
EXPECTED_PROTOCOL_SHA256 = (
    "823012afa89fe9ec982a7ca82c78c3f24697b6969d1d3fcaa73f5ee7d3b82e86"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path.name} must contain an object")
    return value


class ModelV11AnalogTests(unittest.TestCase):
    def test_protocol_was_frozen_before_results(self) -> None:
        protocol = read_json(PROTOCOL)
        sidecar = (
            RESEARCH / "model_v11_analog_protocol.sha256"
        ).read_text(encoding="utf-8").strip()
        self.assertEqual(sha256_file(PROTOCOL), EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(
            sidecar,
            f"{EXPECTED_PROTOCOL_SHA256}  model_v11_analog_protocol.json",
        )
        self.assertEqual(
            protocol["status"],
            "registered_before_any_v11_outcome_computation",
        )
        self.assertFalse(
            protocol["authority"]["production_promotion_allowed_from_this_run"]
        )
        self.assertTrue(protocol["shared_retrieval_contract"]["no_k_grid"])
        self.assertTrue(
            protocol["shared_retrieval_contract"][
                "no_similarity_threshold_grid"
            ]
        )

    def test_hypotheses_are_structurally_distinct_and_complete(self) -> None:
        protocol = read_json(PROTOCOL)
        hypotheses = protocol["registered_hypotheses"]
        self.assertEqual(len(hypotheses), 10)
        self.assertEqual(
            protocol["hypothesis_family_size"], len(hypotheses)
        )
        self.assertEqual(
            len({item["id"] for item in hypotheses}), len(hypotheses)
        )
        self.assertEqual(
            len({item["family"] for item in hypotheses}), len(hypotheses)
        )
        joined = json.dumps(hypotheses, ensure_ascii=False).lower()
        for marker in (
            "issuer-excluded",
            "recency-weighted",
            "cross-code",
            "lower bound",
            "tail probability",
            "prototype",
            "issuer-specific",
            "consensus",
            "hierarchical",
        ):
            self.assertIn(marker, joined)

    def test_result_binding_and_pit_audits(self) -> None:
        result = read_json(RESULT)
        self.assertEqual(result["protocol_sha256"], EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(
            result["artifacts"]["picks_sha256"], sha256_file(PICKS)
        )
        self.assertTrue(
            result["integrity"]["strictly_prior_training_all_folds"]
        )
        self.assertEqual(
            result["integrity"]["future_publication_violations"], 0
        )
        self.assertTrue(
            result["integrity"]["source_missing_never_encoded_no_event"]
        )
        self.assertTrue(
            result["integrity"]["target_session_outcome_mutation"]["passes"]
        )
        self.assertTrue(
            all(
                fold["strictly_prior_training"]
                for fold in result["folds"]
            )
        )
        self.assertEqual(
            result["coverage"]["strict_complete_score_sessions"], 88
        )
        self.assertFalse(
            result["coverage"][
                "historical_cache_observation_metadata_available"
            ]
        )

    def test_metrics_are_complete_for_costs_topk_months_and_slices(
        self,
    ) -> None:
        result = read_json(RESULT)
        self.assertEqual(len(result["implementation"]["candidate_ids"]), 10)
        self.assertFalse(
            result["implementation"][
                "outcome_coefficients_fitted_for_analog_candidates"
            ]
        )
        self.assertFalse(
            result["implementation"]["hyperparameter_grid_searched"]
        )
        for candidate_id, by_top_k in result["metrics"].items():
            with self.subTest(candidate=candidate_id):
                self.assertEqual(set(by_top_k), {"top1", "top2"})
                for top_k in (1, 2):
                    metrics = by_top_k[f"top{top_k}"]
                    self.assertEqual(
                        set(metrics["costs"]), {"0", "20", "40", "60"}
                    )
                    self.assertEqual(metrics["months"], 5)
                    self.assertEqual(
                        set(metrics["slices"]),
                        {"discovery", "confirmation_a", "confirmation_b"},
                    )
                    self.assertIn("net20_mean_pct", metrics["top10_code_cash"])
                    self.assertIn(
                        "net20_mean_pct", metrics["best20_session_removal"]
                    )

    def test_independent_audit_and_decision_are_self_consistent(self) -> None:
        audit = read_json(AUDIT)
        self.assertTrue(audit["integrity"]["passes"])
        self.assertTrue(audit["pnl_reproduction"]["passes"])
        self.assertTrue(
            audit["known_comparator_reproduction"]["passes"]
        )
        self.assertTrue(
            audit["deterministic_reproduction"]["picks_byte_identical"]
        )
        self.assertTrue(
            audit["deterministic_reproduction"]["metrics_identical"]
        )
        passing = [
            candidate_id
            for candidate_id, gate in audit["candidate_gates"].items()
            if gate["passes_all"]
        ]
        self.assertEqual(
            passing, audit["decision"]["passing_candidate_ids"]
        )
        selected = audit["decision"]["selected_forward_shadow_candidate"]
        self.assertEqual(selected is not None, bool(passing))
        self.assertFalse(audit["decision"]["production_gate_passes"])
        self.assertFalse(audit["decision"]["production_model_changed"])
        self.assertFalse(audit["decision"]["orders_allowed"])
        self.assertFalse(
            audit["decision"]["production_checks"][
                "historical_observation_provenance_proven"
            ]
        )
        self.assertFalse(
            audit["decision"]["production_checks"][
                "new_120_session_prospective_gate_passed"
            ]
        )


if __name__ == "__main__":
    unittest.main()
