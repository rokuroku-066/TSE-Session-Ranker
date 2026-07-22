from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.audit_model_v06_t1_correction_result import (
    daily_returns,
    pick_change_report,
    profit_metrics,
)


class IndependentT1CorrectionAuditTests(unittest.TestCase):
    @staticmethod
    def _picks() -> pd.DataFrame:
        dates = pd.bdate_range("2024-07-01", periods=6)
        return pd.DataFrame(
            {
                "date": np.repeat(dates, 2),
                "model_rank": [1, 2] * len(dates),
                "code": [f"{day}-{rank}" for day in range(6) for rank in (1, 2)],
                "label": [1.0, 0.0] * len(dates),
                "oc_return_pct": [2.0, -1.0] * len(dates),
            }
        )

    def test_fixed_slot_cost_and_metrics(self) -> None:
        picks = self._picks()
        daily = daily_returns(picks, top_k=2, cost_bps=20.0)
        self.assertTrue(np.allclose(daily["net_return_pct"], 0.3))
        metrics = profit_metrics(picks, top_k=2, cost_bps=20.0)
        self.assertEqual(metrics["n"], 12)
        self.assertEqual(metrics["days"], 6)
        self.assertEqual(metrics["executed"], 12)
        self.assertAlmostEqual(metrics["net_mean_pct_at_cost"], 0.3)

    def test_pick_change_counts_slots_and_distinct_days(self) -> None:
        legacy = self._picks()
        corrected = legacy.copy()
        corrected.loc[0, "code"] = "replacement-a"
        corrected.loc[1, "code"] = "replacement-b"
        corrected.loc[4, "code"] = "replacement-c"
        report = pick_change_report(legacy, corrected)
        self.assertEqual(report["changed_slots"], 3)
        self.assertEqual(report["dates_with_any_changed_slot"], 2)
        self.assertEqual(report["unchanged_slots"], 9)


if __name__ == "__main__":
    unittest.main()
