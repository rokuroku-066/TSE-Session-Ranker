#!/usr/bin/env python3
"""Integrity tests for model-v11 cross-stock graph research."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
PROTOCOL = RESEARCH / "model_v11_graph_protocol.json"
RESULT = RESEARCH / "model_v11_graph_result.json"
AUDIT = RESEARCH / "model_v11_graph_audit.json"
PICKS = RESEARCH / "model_v11_graph_picks.csv"
EXPECTED_PROTOCOL_SHA256 = (
    "2f2382716ba9f7b151988470334b4da93bee5f68e30f7264beedcf16bf790244"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path.name} must contain an object")
    return value


class ModelV11GraphTests(unittest.TestCase):
    def test_protocol_is_frozen_and_zero_base(self) -> None:
        protocol = read_json(PROTOCOL)
        self.assertEqual(sha256_file(PROTOCOL), EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(
            (RESEARCH / "model_v11_graph_protocol.sha256")
            .read_text(encoding="utf-8")
            .strip(),
            f"{EXPECTED_PROTOCOL_SHA256}  model_v11_graph_protocol.json",
        )
        self.assertEqual(
            protocol["status"],
            "registered_before_any_v11_graph_candidate_outcome_computation",
        )
        self.assertFalse(
            protocol["authority"]["production_promotion_allowed_from_this_run"]
        )
        self.assertFalse(
            protocol["separation_from_prior_work"][
                "z17_same_date_peer_mean_residual_target_reimplemented"
            ]
        )
        self.assertTrue(
            protocol["shared_graph_contract"][
                "no_neighbor_or_threshold_grid"
            ]
        )
        self.assertTrue(protocol["shared_graph_contract"]["no_alpha_grid"])

    def test_hypotheses_and_fail_closed_sources_are_complete(self) -> None:
        protocol = read_json(PROTOCOL)
        hypotheses = protocol["registered_hypotheses"]
        self.assertEqual(protocol["hypothesis_family_size"], 10)
        self.assertEqual(len(hypotheses), 10)
        self.assertEqual(
            len({item["id"] for item in hypotheses}), len(hypotheses)
        )
        self.assertEqual(
            len({item["family"] for item in hypotheses}), len(hypotheses)
        )
        blocked = protocol[
            "external_data_limit_and_fail_closed_hypotheses"
        ]
        self.assertEqual(
            {item["id"] for item in blocked},
            {
                "B01_external_economic_causal_chain",
                "B02_japan_us_sector_lead_lag",
                "B03_news_social_sentiment_granger",
            },
        )
        self.assertTrue(
            all(item["decision"] == "fail_closed_no_score" for item in blocked)
        )
        self.assertEqual(len(protocol["prior_research"]), 4)
        self.assertIn(
            "no significant future-return predictive power",
            json.dumps(protocol["prior_research"], ensure_ascii=False),
        )

    def test_result_is_bound_and_all_folds_are_pit(self) -> None:
        result = read_json(RESULT)
        self.assertEqual(result["protocol_sha256"], EXPECTED_PROTOCOL_SHA256)
        self.assertEqual(
            result["artifacts"]["picks_sha256"], sha256_file(PICKS)
        )
        self.assertEqual(result["coverage"]["score_sessions"], 266)
        self.assertEqual(len(result["coverage"]["score_months"]), 13)
        self.assertEqual(len(result["folds"]), 13)
        self.assertTrue(result["integrity"]["strictly_prior_all_folds"])
        self.assertEqual(result["integrity"]["price_source_violations"], 0)
        self.assertTrue(
            result["integrity"]["target_session_outcome_mutation"]["passes"]
        )
        self.assertFalse(
            result["integrity"][
                "z17_same_date_peer_residual_target_reimplemented"
            ]
        )
        self.assertFalse(
            result["integrity"]["external_unavailable_sources_proxied"]
        )
        self.assertTrue(
            all(fold["strictly_prior_all"] for fold in result["folds"])
        )
        for fold in result["folds"]:
            self.assertEqual(
                set(fold["graph"]["community_counts"]),
                {"0", "1", "2", "3"},
            )
            self.assertTrue(
                all(
                    count > 0
                    for count in fold["graph"]["community_counts"].values()
                )
            )

    def test_metrics_cover_required_costs_and_robustness(self) -> None:
        result = read_json(RESULT)
        self.assertEqual(len(result["implementation"]["graph_candidate_ids"]), 10)
        self.assertFalse(
            result["implementation"][
                "hyperparameter_or_threshold_grid_searched"
            ]
        )
        self.assertFalse(
            result["implementation"][
                "graph_candidate_fitted_regression_coefficients"
            ]
        )
        self.assertEqual(
            set(result["implementation"]["fail_closed_external_hypotheses"]),
            {
                "B01_external_economic_causal_chain",
                "B02_japan_us_sector_lead_lag",
                "B03_news_social_sentiment_granger",
            },
        )
        for candidate_id, by_top_k in result["metrics"].items():
            with self.subTest(candidate=candidate_id):
                self.assertEqual(set(by_top_k), {"top1", "top2"})
                for top_k in (1, 2):
                    metrics = by_top_k[f"top{top_k}"]
                    self.assertEqual(
                        set(metrics["costs"]), {"0", "20", "40", "60"}
                    )
                    self.assertEqual(metrics["months"], 13)
                    self.assertEqual(
                        set(metrics["slices"]),
                        {"discovery", "confirmation_a", "confirmation_b"},
                    )
                    self.assertIn(
                        "net20_mean_pct", metrics["best20_session_removal"]
                    )
                    self.assertIn(
                        "net20_mean_pct", metrics["top10_code_cash"]
                    )
                    self.assertIn(
                        "top10_selection_share", metrics["concentration"]
                    )

    def test_audit_and_no_production_decision_are_consistent(self) -> None:
        audit = read_json(AUDIT)
        self.assertTrue(audit["integrity"]["passes"])
        self.assertTrue(audit["pnl_reproduction"]["passes"])
        self.assertTrue(audit["known_control_reproduction"]["passes"])
        self.assertTrue(
            audit["deterministic_reproduction"]["picks_byte_identical"]
        )
        self.assertTrue(
            audit["deterministic_reproduction"][
                "metrics_and_folds_identical"
            ]
        )
        passing = [
            candidate_id
            for candidate_id, gate in audit["candidate_gates"].items()
            if gate["passes_all"]
        ]
        self.assertEqual(passing, audit["decision"]["passing_candidate_ids"])
        self.assertEqual(
            audit["decision"]["selected_forward_shadow_candidate"] is not None,
            bool(passing),
        )
        self.assertFalse(audit["decision"]["production_gate_passes"])
        self.assertFalse(audit["decision"]["production_model_changed"])
        self.assertFalse(audit["decision"]["orders_allowed"])
        self.assertFalse(
            audit["decision"]["production_checks"][
                "new_120_session_prospective_gate_passed"
            ]
        )


if __name__ == "__main__":
    unittest.main()
