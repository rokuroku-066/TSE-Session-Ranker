from __future__ import annotations

import unittest
from datetime import datetime, timezone

import pandas as pd

from research.analyze_model_v07_tail_risk import (
    assert_protocol_registered_before_run,
    attach_tail_conditions,
    build_recipe_picks,
    daily_recipe_returns,
    load_protocol,
)
class TailRiskProtocolTests(unittest.TestCase):
    def test_protocol_explicitly_forbids_production_promotion(self) -> None:
        protocol = load_protocol()
        self.assertFalse(protocol["authority"]["production_promotion_allowed"])
        self.assertTrue(
            protocol["authority"]["hypothesis_was_selected_after_outcome_inspection"]
        )
        self.assertEqual(len(protocol["tail_conditions"]), 5)
        self.assertEqual(len(protocol["recipe_registry"]), 9)

    def test_protocol_registration_must_strictly_precede_run(self) -> None:
        protocol = load_protocol()
        assert_protocol_registered_before_run(
            protocol, datetime(2026, 7, 22, 7, 0, tzinfo=timezone.utc)
        )
        with self.assertRaisesRegex(ValueError, "not registered before"):
            assert_protocol_registered_before_run(
                protocol, datetime(2026, 7, 22, 6, 0, tzinfo=timezone.utc)
            )


class TailFlagTests(unittest.TestCase):
    @staticmethod
    def _display() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-04-01", "2025-04-01"]),
                "model_rank": [1, 2],
                "code": ["A", "B"],
                "name": ["A", "B"],
                "model_score": [0.9, 0.8],
                "label": [1.0, 0.0],
                "oc_return_pct": [10.0, -2.0],
                "outcome_observed": [True, True],
                "display_source": ["test", "test"],
            }
        )

    @staticmethod
    def _projection() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-04-01", "2025-04-01"]),
                "code": ["A", "B"],
                "feature_source_max_date": pd.to_datetime(
                    ["2025-03-31", "2025-03-31"]
                ),
                "candidate_price_source_max_date": pd.to_datetime(
                    ["2025-03-31", "2025-03-31"]
                ),
                "session_range_ratio_5_20": [1.5, 1.499],
                "cc_vol_ratio_5_20": [1.7, 1.699],
                "xrank_close_momentum_60": [-0.6, -0.599],
                "flat_oc_rate_20": [0.1, 0.099],
                "overnight_last": [1.7, 1.699],
            }
        )

    def test_registered_boundaries_and_target_invariance(self) -> None:
        protocol = load_protocol()
        baseline = attach_tail_conditions(
            self._display(), self._projection(), protocol
        )
        self.assertEqual(baseline["tail_condition_count"].tolist(), [5, 0])
        changed = self._display()
        changed["label"] = [0.0, 1.0]
        changed["oc_return_pct"] = [-99.0, 99.0]
        rebuilt = attach_tail_conditions(changed, self._projection(), protocol)
        columns = [
            column for column in baseline if column.startswith("tail_flag__")
        ] + ["tail_condition_count"]
        pd.testing.assert_frame_equal(baseline[columns], rebuilt[columns])

    def test_missing_preopen_feature_is_explicit_and_does_not_invent_a_veto(self) -> None:
        protocol = load_protocol()
        projection = self._projection()
        projection.loc[0, "cc_vol_ratio_5_20"] = None
        candidates = attach_tail_conditions(self._display(), projection, protocol)
        self.assertTrue(pd.isna(candidates.loc[0, "tail_condition_count"]))
        self.assertFalse(candidates.loc[0, "tail_all_features_available"])
        recipes = build_recipe_picks(candidates, protocol)
        single = recipes.loc[
            recipes["recipe_id"].eq("veto_close_volatility_expansion")
        ]
        count = recipes.loc[recipes["recipe_id"].eq("veto_any_1_of_5")]
        self.assertTrue(single.iloc[0]["marker_accepted"])
        self.assertTrue(count.iloc[0]["marker_accepted"])

    def test_cash_veto_keeps_two_slots_and_never_replaces_rank_one(self) -> None:
        protocol = load_protocol()
        candidates = attach_tail_conditions(
            self._display(), self._projection(), protocol
        )
        recipes = build_recipe_picks(candidates, protocol)
        count_one = recipes.loc[recipes["recipe_id"].eq("veto_any_1_of_5")]
        self.assertEqual(count_one["model_rank"].tolist(), [1, 2])
        self.assertEqual(count_one["code"].tolist(), ["A", "B"])
        self.assertEqual(count_one["executed"].tolist(), [False, True])
        self.assertEqual(count_one["slot_weight"].tolist(), [0.5, 0.5])
        daily = daily_recipe_returns(recipes)
        forced = daily.loc[daily["recipe_id"].eq("forced_top2")].iloc[0]
        gated = daily.loc[daily["recipe_id"].eq("veto_any_1_of_5")].iloc[0]
        self.assertAlmostEqual(forced["net20_return_pct"], 3.8)
        self.assertAlmostEqual(gated["net20_return_pct"], -1.1)
        self.assertEqual(gated["signal_slots"], 2)
        self.assertEqual(gated["executed_slots"], 1)


if __name__ == "__main__":
    unittest.main()
