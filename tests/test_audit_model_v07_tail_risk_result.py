from __future__ import annotations

import json
from pathlib import Path
import unittest

import pandas as pd

from research.audit_model_v07_tail_risk_result import (
    ROOT,
    bootstrap_pair,
    build_recipe_rows,
    daily_returns,
    recipe_acceptance,
    recompute_flags,
)


class IndependentTailRiskAuditUnitTests(unittest.TestCase):
    @staticmethod
    def _protocol() -> dict:
        path = ROOT / "research/model_v07_tail_risk_protocol.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_missing_count_is_passed_through_and_vetoed_slot_stays_cash(self) -> None:
        protocol = self._protocol()
        candidates = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-04-01", "2025-04-01"]),
                "model_rank": [1, 2],
                "code": ["A", "B"],
                "name": ["A", "B"],
                "model_score": [0.9, 0.8],
                "oc_return_pct": [10.0, -2.0],
                "display_source": ["test", "test"],
                "session_range_ratio_5_20": [1.5, 1.0],
                "cc_vol_ratio_5_20": [1.7, None],
                "xrank_close_momentum_60": [-0.6, 0.0],
                "flat_oc_rate_20": [0.1, 0.0],
                "overnight_last": [1.7, 0.0],
            }
        )
        flagged = recompute_flags(candidates, protocol)
        self.assertEqual(flagged.loc[0, "tail_condition_count"], 5)
        self.assertTrue(pd.isna(flagged.loc[1, "tail_condition_count"]))
        accepted = recipe_acceptance(flagged, "veto_any_1_of_5")
        self.assertEqual(accepted.tolist(), [False, True])

        rows = build_recipe_rows(flagged, protocol)
        selected = rows.loc[rows["recipe_id"].eq("veto_any_1_of_5")]
        self.assertEqual(selected["model_rank"].tolist(), [1, 2])
        self.assertEqual(selected["executed"].tolist(), [False, True])
        daily = daily_returns(rows)
        forced = daily.loc[daily["recipe_id"].eq("forced_top2")].iloc[0]
        veto = daily.loc[daily["recipe_id"].eq("veto_any_1_of_5")].iloc[0]
        self.assertAlmostEqual(forced["net20_return_pct"], 3.8)
        self.assertAlmostEqual(veto["net20_return_pct"], -1.1)
        self.assertEqual(veto["signal_slots"], 2)
        self.assertEqual(veto["executed_slots"], 1)

    def test_paired_block_interval_uses_candidate_minus_baseline(self) -> None:
        protocol = self._protocol()
        protocol["bootstrap"] = {
            "block_length": 3,
            "samples": 100,
            "confidence": 0.8,
            "random_state": 31,
        }
        dates = pd.date_range("2025-01-01", periods=12, freq="D")
        baseline = pd.Series(range(12), index=dates, dtype=float)
        candidate = baseline + 0.25
        result = bootstrap_pair(candidate, baseline, protocol, seed=32)
        self.assertAlmostEqual(result["point_estimate_delta_pct"], 0.25)
        self.assertAlmostEqual(result["one_sided_lower_delta_pct"], 0.25)
        self.assertAlmostEqual(result["two_sided_lower_delta_pct"], 0.25)
        self.assertAlmostEqual(result["two_sided_upper_delta_pct"], 0.25)


class IndependentTailRiskAuditArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "research/model_v07_tail_risk_result.audit.json"
        cls.report = json.loads(path.read_text(encoding="utf-8"))

    def test_all_independent_recomputations_pass(self) -> None:
        self.assertEqual(self.report["checks"]["failed"], 0)
        self.assertEqual(
            self.report["checks"]["passed"], self.report["checks"]["total"]
        )
        self.assertEqual(self.report["issues"]["critical"], [])

    def test_verdict_and_documentation_findings_are_explicit(self) -> None:
        verdict = self.report["verdict"]
        self.assertEqual(verdict["computational_integrity"], "pass")
        self.assertEqual(verdict["production_adoption"], "reject")
        self.assertEqual(verdict["shadow_marker_adoption"], "none_qualified")
        self.assertEqual(
            verdict["loss_mitigation_observation_only"], "veto_any_1_of_5"
        )
        self.assertEqual(verdict["unrestricted_e0"], "reject")
        issue_ids = {item["issue_id"] for item in self.report["issues"]["minor"]}
        self.assertEqual(
            issue_ids,
            {
                "PROTOCOL_ARTIFACT_REGISTRY_OMITS_CONDITION_SLICE",
                "E0_TAIL_EXPOSURE_DENOMINATOR_IMPLICIT",
            },
        )

    def test_e0_complete_case_denominators_are_recorded(self) -> None:
        denominators = self.report["recomputed_summary"]["e0"][
            "tail_exposure_denominators"
        ]
        self.assertEqual(denominators["g0_replay_slots"], 364)
        self.assertEqual(denominators["g0_complete_tail_slots"], 334)
        self.assertEqual(denominators["g0_incomplete_tail_slots"], 30)
        self.assertEqual(denominators["e0_complete_tail_slots"], 335)
        self.assertEqual(denominators["e0_incomplete_tail_slots"], 29)


if __name__ == "__main__":
    unittest.main()
