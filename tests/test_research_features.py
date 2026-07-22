from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from tse_session_ranker.data.common import (
    SESSION_OHLC,
    build_prior_session_universe,
    merge_daily_prices,
    normalize_daily_prices,
)
from tse_session_ranker.exceptions import DataValidationError, LeakageError
from tse_session_ranker.features import build_feature_panel
from tse_session_ranker.research_features import (
    FUTURES_INTERACTION_FEATURE_COLUMNS,
    SESSION_MARKET_FEATURE_COLUMNS,
    add_futures_interactions,
    add_session_market_features,
)


def _session_prices(periods: int = 85, codes: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2024-11-06", periods=periods)
    for code_index in range(codes):
        previous_close = 800.0 + code_index * 180.0
        beta = 0.55 + code_index * 0.35
        for day_index, date in enumerate(dates):
            market_wave = 0.006 * np.sin(day_index * 0.29)
            overnight = 0.001 * np.cos(day_index * 0.17 + code_index)
            am_return = beta * market_wave + 0.002 * np.cos(
                day_index * 0.41 + code_index
            )
            midday_gap = 0.0008 * np.sin(day_index * 0.23 + code_index)
            pm_return = -0.35 * am_return + 0.003 * np.sin(
                day_index * 0.37 + 0.7 * code_index
            )
            am_open = previous_close * (1.0 + overnight)
            am_close = am_open * (1.0 + am_return)
            pm_open = am_close * (1.0 + midday_gap)
            pm_close = pm_open * (1.0 + pm_return)
            am_pad = 0.0025 + 0.0004 * code_index
            pm_pad = 0.0030 + 0.0003 * code_index
            am_high = max(am_open, am_close) * (1.0 + am_pad)
            am_low = min(am_open, am_close) * (1.0 - am_pad)
            pm_high = max(pm_open, pm_close) * (1.0 + pm_pad)
            pm_low = min(pm_open, pm_close) * (1.0 - pm_pad)
            rows.append(
                {
                    "date": date,
                    "code": str(1001 + code_index),
                    "name": f"銘柄{code_index + 1}",
                    "open": am_open,
                    "high": max(am_high, pm_high),
                    "low": min(am_low, pm_low),
                    "close": pm_close,
                    "am_open": am_open,
                    "am_high": am_high,
                    "am_low": am_low,
                    "am_close": am_close,
                    "pm_open": pm_open,
                    "pm_high": pm_high,
                    "pm_low": pm_low,
                    "pm_close": pm_close,
                }
            )
            previous_close = pm_close
    return pd.DataFrame(rows)


class CommonSessionColumnTests(unittest.TestCase):
    def test_session_columns_survive_normalize_merge_and_prior_universe(self) -> None:
        prices = _session_prices(periods=3, codes=1)
        first = normalize_daily_prices(prices.iloc[:2])
        second = normalize_daily_prices(prices.iloc[2:])
        merged = merge_daily_prices([first, second])
        self.assertTrue(merged[list(SESSION_OHLC)].notna().all(axis=None))
        modeled = build_prior_session_universe(merged)
        target = pd.Timestamp(prices.iloc[1]["date"])
        expected = prices.loc[prices["date"].eq(target), "pm_close"].iloc[0]
        actual = modeled.loc[modeled["date"].eq(target), "pm_close"].iloc[0]
        self.assertAlmostEqual(actual, expected)

    def test_session_column_conflict_is_not_silently_deduplicated(self) -> None:
        row = _session_prices(periods=1, codes=1)
        changed = row.copy()
        changed["pm_close"] *= 1.001
        with self.assertRaisesRegex(DataValidationError, "conflicting duplicate"):
            merge_daily_prices([row, changed])

    def test_partial_session_vector_is_rejected(self) -> None:
        row = _session_prices(periods=1, codes=1)
        row.loc[:, "pm_close"] = np.nan
        with self.assertRaisesRegex(DataValidationError, "all eight"):
            normalize_daily_prices(row)


class SessionMarketFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prices = _session_prices()

    def test_prior_session_formulas_and_market_interactions(self) -> None:
        panel = build_feature_panel(self.prices)
        enriched = add_session_market_features(panel)
        target = pd.Timestamp(sorted(self.prices["date"].unique())[75])
        code = "1002"
        actual = enriched[
            enriched["date"].eq(target) & enriched["code"].eq(code)
        ].iloc[0]
        history = self.prices[
            self.prices["code"].eq(code) & self.prices["date"].lt(target)
        ].sort_values("date")
        prior = history.iloc[-1]
        expected_pm = 100.0 * (prior["pm_close"] / prior["pm_open"] - 1.0)
        expected_midday = 100.0 * (prior["pm_open"] / prior["am_close"] - 1.0)
        self.assertAlmostEqual(actual["prior_pm_return_pct"], expected_pm)
        self.assertAlmostEqual(actual["prior_midday_gap_pct"], expected_midday)
        self.assertTrue(np.isfinite(actual["market_beta_60"]))
        self.assertAlmostEqual(
            actual["market_beta_x_prior_market_return"],
            actual["market_beta_60"] * actual["prior_market_oc_return_pct"],
        )
        raw = self.prices.copy()
        raw["stock_oc"] = 100.0 * (raw["close"] / raw["open"] - 1.0)
        raw["market_oc"] = raw.groupby("date")["stock_oc"].transform("mean")
        beta_history = raw[
            raw["code"].eq(code) & raw["date"].lt(target)
        ].sort_values("date").tail(60)
        expected_beta = beta_history["stock_oc"].cov(beta_history["market_oc"]) / (
            beta_history["market_oc"].var()
        )
        self.assertAlmostEqual(actual["market_beta_60"], expected_beta, places=12)
        ranks = enriched.loc[
            enriched["date"].eq(target),
            [
                column
                for column in SESSION_MARKET_FEATURE_COLUMNS
                if column.startswith("xrank_")
            ],
        ]
        self.assertTrue((ranks.max() <= 1.0).all())
        self.assertTrue((ranks.min() >= -1.0).all())

    def test_same_day_and_future_ohlc_cannot_change_features(self) -> None:
        target = pd.Timestamp(sorted(self.prices["date"].unique())[75])
        baseline = add_session_market_features(build_feature_panel(self.prices))
        changed = self.prices.copy()
        mutate = changed["date"].ge(target)
        changed.loc[mutate, ["open", "high", "low", "close", *SESSION_OHLC]] *= 2.5
        rebuilt = add_session_market_features(build_feature_panel(changed))
        columns = ["date", "code", *SESSION_MARKET_FEATURE_COLUMNS]
        left = baseline.loc[baseline["date"].le(target), columns].reset_index(drop=True)
        right = rebuilt.loc[rebuilt["date"].le(target), columns].reset_index(drop=True)
        assert_frame_equal(left, right, check_exact=True)

    def test_missing_session_prices_remain_missing_not_zero(self) -> None:
        panel = build_feature_panel(self.prices.drop(columns=list(SESSION_OHLC)))
        enriched = add_session_market_features(panel)
        self.assertTrue(enriched["prior_pm_return_pct"].isna().all())
        self.assertTrue(enriched["prior_pm_range_atr_clipped"].isna().all())


class FuturesInteractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = pd.Timestamp("2026-07-21")
        self.panel = pd.DataFrame(
            {
                "date": [self.target, self.target, self.target + pd.Timedelta(days=1)],
                "code": ["1001", "1002", "1001"],
                "market_beta_60": [0.5, 1.5, 0.5],
                "atr14_pct": [1.0, 2.0, 1.0],
                "tdnet_has_revision_up": [1.0, 0.0, 1.0],
            }
        )
        self.context = pd.DataFrame(
            {
                "date": [self.target],
                "observed_at": [
                    pd.Timestamp("2026-07-21 08:58:59", tz="Asia/Tokyo")
                ],
                "nikkei_return_pct": [1.2],
                "topix_return_pct": [0.8],
                "return_definition": [
                    "previous_cash_close_to_08:58:59_JST"
                ],
            }
        )

    def test_exact_date_interactions_vary_by_beta_without_past_fill(self) -> None:
        enriched = add_futures_interactions(self.panel, self.context, "08:58:59")
        today = enriched[enriched["date"].eq(self.target)].sort_values("code")
        self.assertAlmostEqual(
            today.iloc[0]["market_beta_x_nikkei_futures_return"], 0.6
        )
        self.assertAlmostEqual(
            today.iloc[1]["market_beta_x_nikkei_futures_return"], 1.8
        )
        self.assertEqual(today["futures_context_available"].tolist(), [1.0, 1.0])
        self.assertAlmostEqual(
            today.iloc[0]["tdnet_material_x_nikkei_futures_return"], 1.2
        )
        tomorrow = enriched[enriched["date"].gt(self.target)].iloc[0]
        self.assertEqual(tomorrow["futures_context_available"], 0.0)
        self.assertTrue(
            pd.isna(tomorrow["market_beta_x_nikkei_futures_return"])
        )
        self.assertTrue(
            pd.isna(tomorrow["tdnet_material_x_nikkei_futures_return"])
        )
        self.assertTrue(
            set(FUTURES_INTERACTION_FEATURE_COLUMNS).issubset(enriched.columns)
        )

    def test_after_cutoff_and_naive_observation_are_rejected(self) -> None:
        late = self.context.copy()
        late["observed_at"] = pd.Timestamp(
            "2026-07-21 08:59:00", tz="Asia/Tokyo"
        )
        with self.assertRaisesRegex(LeakageError, "after cutoff"):
            add_futures_interactions(self.panel, late)
        naive = self.context.copy()
        naive["observed_at"] = pd.Timestamp("2026-07-21 08:58:00")
        with self.assertRaisesRegex(LeakageError, "timezone-aware"):
            add_futures_interactions(self.panel, naive)

    def test_identical_duplicate_context_is_idempotent(self) -> None:
        duplicated = pd.concat([self.context, self.context], ignore_index=True)
        expected = add_futures_interactions(self.panel, self.context)
        actual = add_futures_interactions(self.panel, duplicated)
        assert_frame_equal(actual, expected)

    def test_mixed_return_definitions_are_rejected(self) -> None:
        changed = self.context.copy()
        changed["date"] = pd.Timestamp("2026-07-22")
        changed["observed_at"] = pd.Timestamp(
            "2026-07-22 08:58:00", tz="Asia/Tokyo"
        )
        changed["return_definition"] = "night_close_to_08:58:59_JST"
        mixed = pd.concat([self.context, changed], ignore_index=True)
        with self.assertRaisesRegex(DataValidationError, "cannot mix"):
            add_futures_interactions(self.panel, mixed)

    def test_context_date_and_observation_local_date_must_match(self) -> None:
        changed = self.context.copy()
        changed["observed_at"] = pd.Timestamp(
            "2026-07-20 23:58:00", tz="Asia/Tokyo"
        )
        with self.assertRaisesRegex(LeakageError, "target date"):
            add_futures_interactions(self.panel, changed)


if __name__ == "__main__":
    unittest.main()
