from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tse_session_ranker.data.common import (
    build_prior_session_universe,
    prepare_modeling_prices,
    session_coverage_report,
)
from tse_session_ranker.exceptions import DataValidationError
from tse_session_ranker.features import build_feature_panel
from tse_session_ranker.io import write_json
from tse_session_ranker.profit import daily_portfolio_returns, profit_metrics


class ProfitTests(unittest.TestCase):
    def test_top2_keeps_unfilled_slot_as_cash(self) -> None:
        picks = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2025-01-06", "2025-01-06", "2025-01-07", "2025-01-07"]
                ),
                "model_rank": [1, 2, 1, 2],
                "label": [1.0, np.nan, 0.0, 1.0],
                "oc_return_pct": [1.0, np.nan, -1.0, 0.6],
            }
        )
        daily = daily_portfolio_returns(picks, top_k=2, cost_bps=20.0)
        first = daily.set_index("date").loc[pd.Timestamp("2025-01-06")]
        self.assertAlmostEqual(first["gross_return_pct"], 0.5)
        self.assertAlmostEqual(first["net_return_pct"], 0.4)
        self.assertEqual(first["executed_slots"], 1)
        metrics = profit_metrics(picks, top_k=2, cost_bps=20.0)
        self.assertEqual(metrics["days"], 2)
        self.assertAlmostEqual(metrics["execution_rate"], 0.75)

    def test_prior_universe_restores_missing_outcome(self) -> None:
        rows = []
        for date in pd.bdate_range("2025-01-06", periods=3):
            for code in ("1001", "1002"):
                rows.append(
                    {
                        "date": date,
                        "code": code,
                        "open": 100.0,
                        "high": 102.0,
                        "low": 99.0,
                        "close": 101.0,
                    }
                )
        prices = pd.DataFrame(rows)
        target = pd.Timestamp("2025-01-07")
        prices = prices[
            ~(prices["date"].eq(target) & prices["code"].eq("1001"))
        ]
        modeled = build_prior_session_universe(prices)
        restored = modeled[
            modeled["date"].eq(target) & modeled["code"].eq("1001")
        ]
        self.assertEqual(len(restored), 1)
        self.assertFalse(bool(restored.iloc[0]["traded"]))
        self.assertTrue(restored[["open", "high", "low", "close"]].isna().all(axis=None))

    def test_rolling_upper_reference_flags_sustained_incomplete_source(self) -> None:
        rows = []
        dates = pd.bdate_range("2025-01-06", periods=50)
        for index, date in enumerate(dates):
            count = 80 if index >= 20 else 100
            for code in range(count):
                rows.append(
                    {
                        "date": date,
                        "code": str(1000 + code),
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.0,
                    }
                )
        report = session_coverage_report(pd.DataFrame(rows))
        outage = report[report["date"].isin(dates[20:])]
        self.assertTrue((~outage["source_complete"]).all())

    def test_source_incomplete_date_is_not_bridged_to_an_older_session(self) -> None:
        rows = []
        dates = pd.bdate_range("2025-01-06", periods=25)
        for index, date in enumerate(dates):
            count = 80 if index == 20 else 100
            for code in range(count):
                rows.append(
                    {
                        "date": date,
                        "code": str(1000 + code),
                        "open": 100.0,
                        "high": 102.0,
                        "low": 99.0,
                        "close": 101.0,
                    }
                )
        modeled, report = prepare_modeling_prices(pd.DataFrame(rows))
        bad_date = dates[20]
        next_date = dates[21]
        day_after = dates[22]
        self.assertFalse(
            bool(report.loc[report["date"].eq(bad_date), "source_complete"].iloc[0])
        )
        next_rows = modeled[modeled["date"].eq(next_date)]
        self.assertTrue(next_rows["universe_source_date"].eq(bad_date).all())
        self.assertTrue((~next_rows["evaluation_ready"]).all())
        self.assertTrue(modeled.loc[modeled["date"].eq(day_after), "evaluation_ready"].all())
        panel = build_feature_panel(modeled)
        recovered = panel[
            panel["date"].eq(day_after) & panel["code"].eq("1099")
        ].iloc[0]
        self.assertTrue(pd.isna(recovered["overnight_last"]))
        self.assertAlmostEqual(recovered["oc_last"], 1.0)

    def test_explicit_calendar_masks_a_fully_missing_session_and_successor(self) -> None:
        dates = pd.bdate_range("2025-01-06", periods=25)
        missing_date = dates[20]
        rows = []
        for date in dates:
            if date == missing_date:
                continue
            for code in range(100):
                rows.append(
                    {
                        "date": date,
                        "code": str(1000 + code),
                        "open": 100.0,
                        "high": 102.0,
                        "low": 99.0,
                        "close": 101.0,
                    }
                )
        modeled, report = prepare_modeling_prices(
            pd.DataFrame(rows), expected_sessions=dates
        )
        self.assertFalse(
            bool(
                report.loc[
                    report["date"].eq(missing_date), "source_complete"
                ].iloc[0]
            )
        )
        missing_rows = modeled[modeled["date"].eq(missing_date)]
        successor_rows = modeled[modeled["date"].eq(dates[21])]
        recovered_rows = modeled[modeled["date"].eq(dates[22])]
        self.assertEqual(len(missing_rows), 100)
        self.assertTrue((~missing_rows["evaluation_ready"]).all())
        self.assertTrue(successor_rows["universe_source_date"].eq(missing_date).all())
        self.assertTrue((~successor_rows["evaluation_ready"]).all())
        self.assertTrue(recovered_rows["evaluation_ready"].all())

    def test_coverage_can_diagnose_a_fully_missing_trailing_session(self) -> None:
        dates = pd.bdate_range("2025-01-06", periods=12)
        rows = []
        for date in dates[:-1]:
            for code in range(20):
                rows.append(
                    {
                        "date": date,
                        "code": str(1000 + code),
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.0,
                    }
                )
        report = session_coverage_report(
            pd.DataFrame(rows),
            expected_sessions=dates,
            expected_through=dates[-1],
        )
        trailing = report[report["date"].eq(dates[-1])].iloc[0]
        self.assertEqual(trailing["session_rows"], 0)
        self.assertFalse(bool(trailing["source_complete"]))
        with self.assertRaisesRegex(DataValidationError, "expected_through"):
            session_coverage_report(
                pd.DataFrame(rows),
                expected_sessions=dates,
                expected_through=dates[-1] + pd.offsets.BDay(1),
            )

    def test_drawdown_includes_initial_nav(self) -> None:
        picks = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-06", "2025-01-07"]),
                "model_rank": [1, 1],
                "label": [0.0, 0.0],
                "oc_return_pct": [-10.0, 0.0],
            }
        )
        metrics = profit_metrics(picks, top_k=1, cost_bps=0.0)
        self.assertAlmostEqual(metrics["max_drawdown_pct"], -10.0)

    def test_costs_must_be_finite_and_non_negative(self) -> None:
        picks = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-06"]),
                "model_rank": [1],
                "label": [1.0],
                "oc_return_pct": [1.0],
            }
        )
        for invalid in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                daily_portfolio_returns(picks, top_k=1, cost_bps=invalid)
            with self.assertRaises(ValueError):
                profit_metrics(picks, top_k=1, cost_bps=invalid)

    def test_json_metrics_replace_non_finite_values_with_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            write_json({"nan": float("nan"), "infinity": float("inf")}, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsNone(payload["nan"])
        self.assertIsNone(payload["infinity"])


if __name__ == "__main__":
    unittest.main()
