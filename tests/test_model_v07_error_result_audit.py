from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.audit_model_v07_error_result import (
    materialize_trades,
    portfolio_metrics,
    recompute_repeat_features,
)


class IndependentTradeAuditTests(unittest.TestCase):
    @staticmethod
    def _display() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2025-01-06", "2025-01-06", "2025-01-07", "2025-01-07"]
                ),
                "model_rank": [1, 2, 1, 2],
                "code": ["A", "B", "C", "D"],
                "outcome_observed": [True] * 4,
                "oc_return_pct": [2.0, -1.0, 4.0, 1.0],
                "anchor_model_score": [0.9, 0.8, 0.7, 0.6],
            }
        )

    def test_cash_sleeve_and_cost_are_independently_materialized(self) -> None:
        dates = pd.bdate_range("2025-01-06", periods=5)
        display = pd.DataFrame(
            {
                "date": np.repeat(dates, 2),
                "model_rank": [1, 2] * 5,
                "code": [f"{rank}-{day}" for day in range(5) for rank in (1, 2)],
                "outcome_observed": [True] * 10,
                "oc_return_pct": [2.0, -1.0] * 5,
                "anchor_model_score": [0.9, 0.8] * 5,
            }
        )
        predictions = display[["date", "model_rank", "code"]].copy()
        predictions["gate_id"] = "gate"
        predictions["trade_decision"] = predictions["model_rank"].eq(1)
        trades = materialize_trades(display, predictions, gate_id="gate")
        self.assertEqual(trades["code"].tolist(), display["code"].tolist())
        self.assertEqual(int(trades["executed"].sum()), 5)
        daily = trades.groupby("date")["net_sleeve_return_pct_20bp"].sum()
        self.assertTrue(np.allclose(daily, 0.5 * (2.0 - 0.2)))
        metrics = portfolio_metrics(trades)
        self.assertEqual(metrics["trade_days"], 5)
        self.assertEqual(metrics["executed_slots"], 5)

    def test_repeat_reconstruction_uses_only_prior_date(self) -> None:
        display = self._display()
        display.loc[2, "code"] = "A"
        rebuilt = recompute_repeat_features(display)
        repeated = rebuilt[
            rebuilt["date"].eq(pd.Timestamp("2025-01-07"))
            & rebuilt["code"].eq("A")
        ].iloc[0]
        self.assertEqual(repeated["prior_selection_count_5"], 1.0)
        self.assertAlmostEqual(repeated["last_selected_net_pct"], 1.8)
        self.assertEqual(
            repeated["repeat_feature_source_max_date"], pd.Timestamp("2025-01-06")
        )
        self.assertTrue(np.isfinite(repeated["anchor_score_change_since_selected"]))


if __name__ == "__main__":
    unittest.main()
