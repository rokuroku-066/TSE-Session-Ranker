from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from tse_session_ranker.config import RankerConfig
from tse_session_ranker.exceptions import DataValidationError
from tse_session_ranker.features import (
    FEATURE_COLUMNS,
    build_feature_panel,
    build_inference_frame,
)

from .helpers import synthetic_prices


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prices = synthetic_prices(periods=100, codes=3)
        cls.dates = sorted(cls.prices["date"].unique())

    def test_target_ohlc_cannot_change_target_features(self) -> None:
        target = pd.Timestamp(self.dates[80])
        original = build_feature_panel(self.prices)
        modified_prices = self.prices.copy()
        mask = modified_prices["date"].eq(target) & modified_prices["code"].eq("1001")
        prior = modified_prices.loc[mask, "open"].iloc[0]
        modified_prices.loc[mask, ["open", "high", "low", "close"]] = [
            prior,
            prior * 1.01,
            prior * 0.80,
            prior * 0.81,
        ]
        modified = build_feature_panel(modified_prices)
        columns = [
            *FEATURE_COLUMNS,
            "prior_close",
            "history_count",
            "atr14_pct",
            "zero_oc_20",
            "eligible",
        ]
        left = original.loc[
            original["date"].eq(target) & original["code"].eq("1001"), columns
        ].reset_index(drop=True)
        right = modified.loc[
            modified["date"].eq(target) & modified["code"].eq("1001"), columns
        ].reset_index(drop=True)
        assert_frame_equal(left, right, check_exact=True)
        old_label = original.loc[
            original["date"].eq(target) & original["code"].eq("1001"), "label"
        ].iloc[0]
        new_label = modified.loc[
            modified["date"].eq(target) & modified["code"].eq("1001"), "label"
        ].iloc[0]
        self.assertNotEqual(old_label, new_label)

    def test_training_and_live_feature_parity_and_no_label(self) -> None:
        target = pd.Timestamp(self.dates[80])
        panel = build_feature_panel(self.prices)
        expected = panel[panel["date"].eq(target)].sort_values("code")
        prior_date = pd.Timestamp(self.dates[79])
        live = build_inference_frame(
            self.prices, target, expected_history_date=prior_date
        ).sort_values("code")
        np.testing.assert_allclose(
            expected[list(FEATURE_COLUMNS) + ["prior_close", "atr14_pct"]],
            live[list(FEATURE_COLUMNS) + ["prior_close", "atr14_pct"]],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        self.assertTrue(live["label"].isna().all())
        self.assertTrue(live["oc_return_pct"].isna().all())

    def test_future_mutation_does_not_change_past_features(self) -> None:
        target = pd.Timestamp(self.dates[75])
        original = build_feature_panel(self.prices)
        changed = self.prices.copy()
        future = changed["date"] > target
        changed.loc[future, ["open", "high", "low", "close"]] *= 3.0
        rebuilt = build_feature_panel(changed)
        columns = ["date", "code", *FEATURE_COLUMNS, "eligible"]
        left = original.loc[original["date"].le(target), columns].reset_index(drop=True)
        right = rebuilt.loc[rebuilt["date"].le(target), columns].reset_index(drop=True)
        assert_frame_equal(left, right, check_exact=True)

    def test_rolling_correlation_matches_prior_sixty_pairs(self) -> None:
        panel = build_feature_panel(self.prices)
        code_panel = panel[panel["code"].eq("1001")].reset_index(drop=True)
        row_index = 90
        prior = code_panel.iloc[row_index - 60 : row_index]
        expected = prior["overnight"].corr(prior["oc_return_pct"])
        actual = code_panel.loc[row_index, "night_day_corr_60"]
        self.assertAlmostEqual(actual, expected, places=12)

    def test_inference_uses_latest_session_active_universe(self) -> None:
        target = pd.Timestamp(self.dates[-1]) + pd.offsets.BDay(1)
        changed = self.prices.copy()
        last_date = changed["date"].max()
        changed = changed[
            ~(changed["code"].eq("1003") & changed["date"].eq(last_date))
        ]
        config = RankerConfig(min_latest_session_coverage=0.50)
        live = build_inference_frame(
            changed,
            target,
            config=config,
            expected_history_date=last_date,
        )
        self.assertNotIn("1003", set(live["code"]))

    def test_stale_history_fails_closed(self) -> None:
        target = pd.Timestamp(self.dates[-1]) + pd.Timedelta(days=10)
        with self.assertRaisesRegex(DataValidationError, "stale"):
            build_inference_frame(
                self.prices,
                target,
                expected_history_date=pd.Timestamp(self.dates[-1]),
            )


if __name__ == "__main__":
    unittest.main()
