from __future__ import annotations

import unittest

import pandas as pd

from tse_session_ranker.backtest import monthly_walk_forward
from tse_session_ranker.config import RankerConfig

from .helpers import synthetic_prices, synthetic_tdnet


class BacktestTests(unittest.TestCase):
    def test_monthly_folds_are_strictly_ordered(self) -> None:
        prices = synthetic_prices(periods=240, codes=4)
        dates = sorted(prices["date"].unique())
        config = RankerConfig(regime_start=str(pd.Timestamp(dates[70]).date()))
        result = monthly_walk_forward(
            prices,
            evaluation_start=pd.Timestamp(dates[180]),
            evaluation_end=pd.Timestamp(dates[230]),
            tdnet_dataset=synthetic_tdnet(prices),
            config=config,
            expected_sessions=dates,
        )
        for row in result.folds.itertuples(index=False):
            self.assertLess(pd.Timestamp(row.train_end), pd.Timestamp(row.score_start))
        self.assertEqual(result.picks_top1["date"].nunique(), len(result.picks_top1))
        self.assertLessEqual(
            result.picks_top2.groupby("date")["code"].size().max(), 2
        )

    def test_no_trade_outcome_is_ranked_before_being_marked_unfilled(self) -> None:
        prices = synthetic_prices(periods=240, codes=4)
        dates = sorted(prices["date"].unique())
        no_trade_date = pd.Timestamp(dates[220])
        mask = prices["date"].eq(no_trade_date) & prices["code"].eq("1001")
        prices.loc[mask, ["open", "high", "low", "close"]] = float("nan")
        prices.loc[mask, "traded"] = False
        config = RankerConfig(regime_start=str(pd.Timestamp(dates[70]).date()))
        result = monthly_walk_forward(
            prices,
            evaluation_start=pd.Timestamp(dates[180]),
            evaluation_end=pd.Timestamp(dates[230]),
            tdnet_dataset=synthetic_tdnet(prices),
            config=config,
            expected_sessions=dates,
        )
        scored = result.scores[
            result.scores["date"].eq(no_trade_date)
            & result.scores["code"].eq("1001")
        ]
        self.assertEqual(len(scored), 1)
        self.assertTrue(scored["label"].isna().all())

    def test_missing_target_row_is_restored_from_prior_universe(self) -> None:
        prices = synthetic_prices(periods=240, codes=4)
        dates = sorted(prices["date"].unique())
        target = pd.Timestamp(dates[220])
        prices = prices[
            ~(prices["date"].eq(target) & prices["code"].eq("1001"))
        ].copy()
        config = RankerConfig(
            regime_start=str(pd.Timestamp(dates[70]).date()),
            minimum_source_coverage=0.50,
        )
        result = monthly_walk_forward(
            prices,
            evaluation_start=pd.Timestamp(dates[180]),
            evaluation_end=pd.Timestamp(dates[230]),
            tdnet_dataset=synthetic_tdnet(prices),
            config=config,
            expected_sessions=dates,
        )
        row = result.scores[
            result.scores["date"].eq(target)
            & result.scores["code"].eq("1001")
        ]
        self.assertEqual(len(row), 1)
        self.assertTrue(row["label"].isna().all())
        self.assertEqual(
            result.summary["primary_objective"],
            "top1.net_mean_pct_at_cost",
        )


if __name__ == "__main__":
    unittest.main()
