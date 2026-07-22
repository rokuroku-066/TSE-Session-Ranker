from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from research.finalize_logit_v04 import add_bounded_daily_features
from tse_session_ranker.features import build_feature_panel
from tse_session_ranker.research_candidates import (
    G1_SHORT_REVERSAL_COLUMNS,
    G2_GAP_TRAIT_COLUMNS,
    G3_SESSION_DYNAMICS_COLUMNS,
    G4_MARKET_REGIME_COLUMNS,
    G5_LIQUIDITY_PROXY_COLUMNS,
    T0_CLEAN_EVENT_COLUMNS,
    T1_EVENT_STRUCTURE_COLUMNS,
    X0_EVENT_CONTEXT_COLUMNS,
    add_event_context_interactions,
    add_historical_candidate_features,
    attach_clean_tdnet_candidate_features,
    build_clean_tdnet_candidate_features,
)
from tse_session_ranker.research_features import add_session_market_features


def _prices(periods: int = 90, codes: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2024-01-04", periods=periods)
    for code_index in range(codes):
        previous_close = 700.0 + 150.0 * code_index
        for day_index, date in enumerate(dates):
            market = 0.004 * np.sin(day_index * 0.23)
            gap = 0.0015 * np.cos(day_index * 0.17 + code_index)
            am_return = market * (0.7 + code_index * 0.2) + 0.002 * np.sin(
                day_index * 0.31 + code_index
            )
            midday = 0.0007 * np.cos(day_index * 0.29 + code_index)
            pm_return = -0.25 * am_return + 0.0025 * np.cos(
                day_index * 0.37 + code_index
            )
            am_open = previous_close * (1.0 + gap)
            am_close = am_open * (1.0 + am_return)
            pm_open = am_close * (1.0 + midday)
            pm_close = pm_open * (1.0 + pm_return)
            am_high = max(am_open, am_close) * 1.003
            am_low = min(am_open, am_close) * 0.997
            pm_high = max(pm_open, pm_close) * 1.0035
            pm_low = min(pm_open, pm_close) * 0.9965
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


def _candidate_panel(prices: pd.DataFrame) -> pd.DataFrame:
    panel = build_feature_panel(prices)
    panel = add_bounded_daily_features(panel)
    panel = add_session_market_features(panel)
    return add_historical_candidate_features(panel)


def _disclosures() -> pd.DataFrame:
    values = [
        (
            "2026-07-20 16:00:00+09:00",
            "1001",
            "自己株式取得に係る事項の決定に関するお知らせ",
        ),
        (
            "2026-07-20 16:01:00+09:00",
            "1002",
            "譲渡制限付株式報酬としての新株式発行に関するお知らせ",
        ),
        (
            "2026-07-20 16:02:00+09:00",
            "1003",
            "通期業績予想の修正に関するお知らせ",
        ),
        (
            "2026-07-20 16:03:00+09:00",
            "1004",
            "株式会社ABCの株式取得（子会社化）に関するお知らせ",
        ),
        (
            "2026-07-21 08:59:00+09:00",
            "1005",
            "株主優待制度の新設に関するお知らせ",
        ),
    ]
    return pd.DataFrame(
        {
            "published_at": [pd.Timestamp(value[0]) for value in values],
            "code": [value[1] for value in values],
            "name": [value[1] for value in values],
            "title": [value[2] for value in values],
            "url": [f"https://example.test/{index}.pdf" for index in range(len(values))],
        }
    )


class HistoricalCandidateFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prices = _prices()

    def test_all_registered_historical_columns_are_created(self) -> None:
        panel = _candidate_panel(self.prices)
        expected = {
            *G1_SHORT_REVERSAL_COLUMNS,
            *G2_GAP_TRAIT_COLUMNS,
            *G3_SESSION_DYNAMICS_COLUMNS,
            *G4_MARKET_REGIME_COLUMNS,
            *G5_LIQUIDITY_PROXY_COLUMNS,
        }
        self.assertTrue(expected.issubset(panel.columns))
        row = panel[
            panel["date"].eq(sorted(panel["date"].unique())[80])
            & panel["code"].eq("1002")
        ].iloc[0]
        finite_values = pd.to_numeric(row[list(expected)], errors="coerce").dropna()
        self.assertTrue(np.isfinite(finite_values.to_numpy(dtype=float)).all())
        self.assertLess(row["candidate_price_source_max_date"], row["date"])

    def test_target_and_future_ohlc_mutation_cannot_change_target_features(self) -> None:
        target = pd.Timestamp(sorted(self.prices["date"].unique())[80])
        baseline = _candidate_panel(self.prices)
        changed = self.prices.copy()
        mutate = changed["date"].ge(target)
        price_columns = [
            "open",
            "high",
            "low",
            "close",
            "am_open",
            "am_high",
            "am_low",
            "am_close",
            "pm_open",
            "pm_high",
            "pm_low",
            "pm_close",
        ]
        changed.loc[mutate, price_columns] *= 2.75
        rebuilt = _candidate_panel(changed)
        columns = [
            "date",
            "code",
            *G1_SHORT_REVERSAL_COLUMNS,
            *G2_GAP_TRAIT_COLUMNS,
            *G3_SESSION_DYNAMICS_COLUMNS,
            *G4_MARKET_REGIME_COLUMNS,
            *G5_LIQUIDITY_PROXY_COLUMNS,
        ]
        left = baseline.loc[baseline["date"].le(target), columns].reset_index(drop=True)
        right = rebuilt.loc[rebuilt["date"].le(target), columns].reset_index(drop=True)
        assert_frame_equal(left, right, check_exact=True)


class CleanTDnetCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sessions = pd.DatetimeIndex(
            ["2026-07-21", "2026-07-22", "2026-07-23"]
        )
        self.disclosures = _disclosures()

    def test_taxonomy_separates_buyback_ma_compensation_and_unknown_revision(self) -> None:
        features = build_clean_tdnet_candidate_features(
            self.disclosures, self.sessions
        )

        def row(code: str, date: str = "2026-07-21") -> pd.Series:
            return features[
                features["code"].eq(code)
                & features["date"].eq(pd.Timestamp(date))
            ].iloc[0]

        buyback = row("1001")
        self.assertEqual(buyback["tdnet_clean_has_buyback_decision"], 1.0)
        self.assertEqual(buyback["tdnet_clean_has_ma_transaction"], 0.0)
        compensation = row("1002")
        self.assertEqual(compensation["tdnet_clean_has_equity_compensation"], 1.0)
        self.assertEqual(
            compensation["tdnet_clean_has_external_equity_financing"], 0.0
        )
        revision = row("1003")
        self.assertEqual(revision["tdnet_clean_has_revision"], 1.0)
        self.assertEqual(
            revision["tdnet_clean_revision_direction_unknown"], 1.0
        )
        acquisition = row("1004")
        self.assertEqual(acquisition["tdnet_clean_has_ma_transaction"], 1.0)
        self.assertEqual(acquisition["tdnet_clean_has_buyback_decision"], 0.0)
        after_cutoff = row("1005", "2026-07-22")
        self.assertEqual(after_cutoff["tdnet_clean_has_benefit"], 1.0)

    def test_incomplete_source_remains_missing_while_complete_no_event_is_zero(self) -> None:
        panel = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-07-21"), pd.Timestamp("2026-07-21")],
                "code": ["9998", "9999"],
                "tdnet_source_complete": [True, False],
            }
        )
        attached = attach_clean_tdnet_candidate_features(
            panel, self.disclosures, self.sessions
        )
        complete = attached[attached["code"].eq("9998")].iloc[0]
        incomplete = attached[attached["code"].eq("9999")].iloc[0]
        self.assertEqual(complete["tdnet_clean_any"], 0.0)
        self.assertTrue(pd.isna(incomplete["tdnet_clean_any"]))

    def test_registered_interactions_are_exact_and_bounded_by_parent_event(self) -> None:
        panel = pd.DataFrame(
            {
                "xrank_close_momentum_20": [0.4, -0.3],
                "prior_market_tail_balance": [0.2, -0.1],
                "tdnet_clean_any": [1.0, 0.0],
                "tdnet_clean_has_revision": [1.0, 0.0],
                "tdnet_clean_has_dividend": [0.0, 0.0],
                "tdnet_clean_has_buyback_decision": [0.0, 0.0],
                "tdnet_clean_has_external_equity_financing": [0.0, 0.0],
                "tdnet_clean_has_ma_transaction": [0.0, 0.0],
            }
        )
        enriched = add_event_context_interactions(panel)
        self.assertTrue(set(X0_EVENT_CONTEXT_COLUMNS).issubset(enriched.columns))
        self.assertAlmostEqual(
            enriched.loc[0, "tdnet_clean_revision_x_xrank_close_momentum_20"],
            0.4,
        )
        self.assertEqual(
            enriched.loc[1, "tdnet_clean_any_x_prior_market_tail_balance"],
            0.0,
        )


class CandidateCatalogTests(unittest.TestCase):
    def test_catalog_columns_match_implementation_constants(self) -> None:
        root = Path(__file__).resolve().parents[1]
        catalog = json.loads(
            (root / "research/model_v06_feature_catalog.json").read_text(
                encoding="utf-8"
            )
        )["historical_screen"]
        expected = {
            "G1_short_reversal": G1_SHORT_REVERSAL_COLUMNS,
            "G2_gap_trait": G2_GAP_TRAIT_COLUMNS,
            "G3_session_dynamics": G3_SESSION_DYNAMICS_COLUMNS,
            "G4_market_regime": G4_MARKET_REGIME_COLUMNS,
            "G5_liquidity_proxy": G5_LIQUIDITY_PROXY_COLUMNS,
            "T0_clean_event": T0_CLEAN_EVENT_COLUMNS,
            "T1_event_structure": T1_EVENT_STRUCTURE_COLUMNS,
            "X0_event_context": X0_EVENT_CONTEXT_COLUMNS,
        }
        for name, columns in expected.items():
            self.assertEqual(tuple(catalog[name]["features"]), tuple(columns))


if __name__ == "__main__":
    unittest.main()
