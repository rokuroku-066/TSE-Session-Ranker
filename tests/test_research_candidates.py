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
    TDNET_CANDIDATE_COLUMNS,
    V07_EVENT_SEMANTIC_COLUMNS,
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

    def _features_for_titles(self, titles: list[str]) -> pd.DataFrame:
        disclosures = pd.DataFrame(
            {
                "published_at": [
                    pd.Timestamp("2026-07-20 16:00:00+09:00")
                    + pd.Timedelta(minutes=index)
                    for index in range(len(titles))
                ],
                "code": [str(3001 + index) for index in range(len(titles))],
                "name": [str(3001 + index) for index in range(len(titles))],
                "title": titles,
                "url": [
                    f"https://example.test/v07-{index}.pdf"
                    for index in range(len(titles))
                ],
            }
        )
        return build_clean_tdnet_candidate_features(
            disclosures, self.sessions
        ).set_index("code")

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

    def test_family_count_stays_aligned_after_bundle_sort(self) -> None:
        values = [
            ("2026-07-20 16:00:00+09:00", "1001", "2026年3月期 決算短信"),
            ("2026-07-20 16:01:00+09:00", "1002", "2026年3月期 決算短信"),
            ("2026-07-20 16:02:00+09:00", "1002", "株主優待制度の新設"),
            ("2026-07-21 16:00:00+09:00", "1001", "2026年3月期 決算短信"),
            ("2026-07-21 16:01:00+09:00", "1001", "株主優待制度の新設"),
            ("2026-07-21 16:02:00+09:00", "1001", "株式分割のお知らせ"),
            ("2026-07-21 16:03:00+09:00", "1002", "2026年3月期 決算短信"),
        ]
        disclosures = pd.DataFrame(
            {
                "published_at": [pd.Timestamp(value[0]) for value in values],
                "code": [value[1] for value in values],
                "name": [value[1] for value in values],
                "title": [value[2] for value in values],
                "url": [
                    f"https://example.test/family-{index}.pdf"
                    for index in range(len(values))
                ],
            }
        )
        features = build_clean_tdnet_candidate_features(
            disclosures.sample(frac=1.0, random_state=17), self.sessions
        ).set_index(["date", "code"])
        expected = {
            (pd.Timestamp("2026-07-21"), "1001"): 1,
            (pd.Timestamp("2026-07-21"), "1002"): 2,
            (pd.Timestamp("2026-07-22"), "1001"): 3,
            (pd.Timestamp("2026-07-22"), "1002"): 1,
        }
        for key, count in expected.items():
            row = features.loc[key]
            self.assertAlmostEqual(
                row["tdnet_clean_family_count_log1p"], np.log1p(count)
            )
            self.assertEqual(
                row["tdnet_clean_single_family"], float(count == 1)
            )
            self.assertEqual(row["tdnet_v07_economic_family_count"], count)
            self.assertEqual(
                row["tdnet_v07_single_economic_family"], float(count == 1)
            )

    def test_v07_observed_and_classified_events_are_distinct(self) -> None:
        values = [
            (
                "2026-07-20 16:00:00+09:00",
                "2001",
                "コーポレート・ガバナンスに関する報告書",
            ),
            (
                "2026-07-20 16:01:00+09:00",
                "2002",
                "剰余金の配当（増配）に関するお知らせ",
            ),
        ]
        disclosures = pd.DataFrame(
            {
                "published_at": [pd.Timestamp(value[0]) for value in values],
                "code": [value[1] for value in values],
                "name": [value[1] for value in values],
                "title": [value[2] for value in values],
                "url": [
                    f"https://example.test/observed-{index}.pdf"
                    for index in range(len(values))
                ],
            }
        )
        features = build_clean_tdnet_candidate_features(
            disclosures, self.sessions
        ).set_index(["date", "code"])
        unclassified = features.loc[(pd.Timestamp("2026-07-21"), "2001")]
        self.assertEqual(unclassified["tdnet_clean_any"], 1.0)
        self.assertEqual(unclassified["tdnet_v07_observed_any"], 1.0)
        self.assertEqual(
            unclassified["tdnet_v07_fresh_classified_economic_any"], 0.0
        )
        classified = features.loc[(pd.Timestamp("2026-07-21"), "2002")]
        self.assertEqual(classified["tdnet_v07_observed_any"], 1.0)
        self.assertEqual(
            classified["tdnet_v07_fresh_classified_economic_any"], 1.0
        )

    def test_v07_intercompany_dividend_is_not_shareholder_dividend(self) -> None:
        values = [
            (
                "2026-07-20 16:00:00+09:00",
                "2101",
                "連結子会社からの配当金受領に関するお知らせ",
            ),
            (
                "2026-07-20 16:01:00+09:00",
                "2102",
                "剰余金の配当に関するお知らせ",
            ),
        ]
        disclosures = pd.DataFrame(
            {
                "published_at": [pd.Timestamp(value[0]) for value in values],
                "code": [value[1] for value in values],
                "name": [value[1] for value in values],
                "title": [value[2] for value in values],
                "url": [
                    f"https://example.test/dividend-{index}.pdf"
                    for index in range(len(values))
                ],
            }
        )
        features = build_clean_tdnet_candidate_features(
            disclosures, self.sessions
        ).set_index(["date", "code"])
        intercompany = features.loc[(pd.Timestamp("2026-07-21"), "2101")]
        # The legacy broad dividend flag remains unchanged for v0.6 replay.
        self.assertEqual(intercompany["tdnet_clean_has_dividend"], 1.0)
        self.assertEqual(
            intercompany["tdnet_v07_has_shareholder_dividend"], 0.0
        )
        self.assertEqual(
            intercompany["tdnet_v07_has_intercompany_dividend"], 1.0
        )
        shareholder = features.loc[(pd.Timestamp("2026-07-21"), "2102")]
        self.assertEqual(
            shareholder["tdnet_v07_has_shareholder_dividend"], 1.0
        )
        self.assertEqual(
            shareholder["tdnet_v07_has_intercompany_dividend"], 0.0
        )

    def test_v07_dividend_roles_are_separated(self) -> None:
        features = self._features_for_titles(
            [
                "子会社からの特別配当金受領に関するお知らせ",
                "投資先からの配当金受領に関するお知らせ",
                "子会社における業績予想及び配当予想の修正に関するお知らせ",
                "剰余金の配当（増配）に関するお知らせ",
            ]
        )
        received_columns = [
            "tdnet_v07_has_shareholder_dividend",
            "tdnet_v07_has_received_dividend",
            "tdnet_v07_has_intercompany_dividend",
            "tdnet_v07_has_subsidiary_dividend",
        ]
        self.assertEqual(
            features.loc["3001", received_columns].tolist(), [0, 1, 1, 0]
        )
        self.assertEqual(
            features.loc["3002", received_columns].tolist(), [0, 1, 0, 0]
        )
        self.assertEqual(
            features.loc["3003", received_columns].tolist(), [0, 0, 0, 1]
        )
        self.assertEqual(
            features.loc["3004", received_columns].tolist(), [1, 0, 0, 0]
        )

    def test_v07_forecast_initial_revision_and_explanation_are_separated(
        self,
    ) -> None:
        features = self._features_for_titles(
            [
                "業績予想の公表に関するお知らせ",
                "通期業績予想の変更に関するお知らせ",
                "業績予想に関する補足説明資料",
            ]
        )
        forecast_columns = [
            "tdnet_v07_has_forecast_initial",
            "tdnet_v07_has_forecast_revision",
        ]
        self.assertEqual(
            features.loc["3001", forecast_columns].tolist(), [1, 0]
        )
        self.assertEqual(
            features.loc["3002", forecast_columns].tolist(), [0, 1]
        )
        self.assertEqual(
            features.loc["3003", forecast_columns].tolist(), [0, 0]
        )
        self.assertEqual(
            features.loc["3003", "tdnet_v07_economic_family_count"], 0.0
        )

    def test_v07_expanded_progress_phrases_are_not_fresh_economic_families(
        self,
    ) -> None:
        titles = [
            "第1回新株予約権の大量行使に関するお知らせ",
            "第1回新株予約権の月間行使状況に関するお知らせ",
            "第1回新株予約権の行使完了に関するお知らせ",
            "第1回新株予約権の発行内容確定に関するお知らせ",
            "第三者割当による新株式発行の条件決定に関するお知らせ",
        ]
        disclosures = pd.DataFrame(
            {
                "published_at": [
                    pd.Timestamp(f"2026-07-20 16:0{index}:00+09:00")
                    for index in range(len(titles))
                ],
                "code": [str(2201 + index) for index in range(len(titles))],
                "name": [str(2201 + index) for index in range(len(titles))],
                "title": titles,
                "url": [
                    f"https://example.test/progress-{index}.pdf"
                    for index in range(len(titles))
                ],
            }
        )
        features = build_clean_tdnet_candidate_features(
            disclosures, self.sessions
        )
        self.assertTrue(features["tdnet_v07_has_progress_stage"].eq(1.0).all())
        self.assertTrue(
            features["tdnet_v07_economic_family_count"].eq(0.0).all()
        )
        self.assertTrue(
            features["tdnet_v07_fresh_classified_economic_any"].eq(0.0).all()
        )
        massive_exercise = features[features["code"].eq("2201")].iloc[0]
        self.assertEqual(massive_exercise["tdnet_clean_has_progress_stage"], 0.0)

    def test_v07_generic_change_word_does_not_erase_fresh_revision(self) -> None:
        disclosures = pd.DataFrame(
            {
                "published_at": [pd.Timestamp("2026-07-20 16:00:00+09:00")],
                "code": ["2251"],
                "name": ["2251"],
                "title": ["通期業績予想の変更（上方修正）に関するお知らせ"],
                "url": ["https://example.test/fresh-revision.pdf"],
            }
        )
        row = build_clean_tdnet_candidate_features(
            disclosures, self.sessions
        ).iloc[0]
        self.assertEqual(row["tdnet_v07_has_progress_stage"], 0.0)
        self.assertEqual(row["tdnet_v07_fresh_classified_economic_any"], 1.0)
        self.assertEqual(row["tdnet_v07_economic_family_count"], 1.0)

    def test_v07_progress_is_gated_per_family_in_mixed_titles(self) -> None:
        features = self._features_for_titles(
            [
                "自己株式の取得状況及び通期業績予想の修正に関するお知らせ",
                "公開買付けへの応募結果および業績予想の修正に関するお知らせ",
            ]
        )
        buyback = features.loc["3001"]
        self.assertEqual(buyback["tdnet_v07_has_fresh_buyback"], 0.0)
        self.assertEqual(buyback["tdnet_v07_has_followup_buyback"], 1.0)
        self.assertEqual(buyback["tdnet_v07_has_forecast_revision"], 1.0)
        self.assertEqual(buyback["tdnet_v07_economic_family_count"], 1.0)
        self.assertEqual(buyback["tdnet_v07_followup_family_count"], 1.0)
        self.assertEqual(
            buyback["tdnet_v07_fresh_classified_economic_any"], 1.0
        )

        takeover = features.loc["3002"]
        self.assertEqual(takeover["tdnet_v07_has_fresh_ma"], 0.0)
        self.assertEqual(takeover["tdnet_v07_has_followup_ma"], 1.0)
        self.assertEqual(takeover["tdnet_v07_has_forecast_revision"], 1.0)
        self.assertEqual(takeover["tdnet_v07_economic_family_count"], 1.0)
        self.assertEqual(takeover["tdnet_v07_followup_family_count"], 1.0)

    def test_v07_self_tender_offer_is_a_buyback_not_ma(self) -> None:
        features = self._features_for_titles(
            [
                "自己株式の公開買付けに関するお知らせ",
                "自己株式の公開買付けの結果及び取得終了に関するお知らせ",
                "当社株式に対する公開買付けに関する賛同意見表明",
                "トヨタ自動車株式会社による自己株式の公開買付けへの応募に関するお知らせ",
                "当社子会社による自己株式の公開買付けに関するお知らせ",
                "三菱商事株式会社の自己株式公開買付けへの当社子会社による応募のお知らせ",
            ]
        )
        fresh = features.loc["3001"]
        followup = features.loc["3002"]
        external_takeover = features.loc["3003"]
        tender_applicant = features.loc["3004"]
        subsidiary_tender = features.loc["3005"]
        subsidiary_applicant = features.loc["3006"]

        self.assertEqual(fresh["tdnet_v07_has_fresh_buyback"], 1.0)
        self.assertEqual(fresh["tdnet_v07_has_followup_buyback"], 0.0)
        self.assertEqual(fresh["tdnet_v07_has_fresh_ma"], 0.0)
        self.assertEqual(fresh["tdnet_v07_economic_family_count"], 1.0)

        self.assertEqual(followup["tdnet_v07_has_fresh_buyback"], 0.0)
        self.assertEqual(followup["tdnet_v07_has_followup_buyback"], 1.0)
        self.assertEqual(followup["tdnet_v07_has_fresh_ma"], 0.0)
        self.assertEqual(followup["tdnet_v07_followup_family_count"], 1.0)

        self.assertEqual(external_takeover["tdnet_v07_has_fresh_buyback"], 0.0)
        self.assertEqual(external_takeover["tdnet_v07_has_fresh_ma"], 1.0)

        for nonissuer in (
            tender_applicant,
            subsidiary_tender,
            subsidiary_applicant,
        ):
            self.assertEqual(nonissuer["tdnet_v07_has_fresh_buyback"], 0.0)
            self.assertEqual(nonissuer["tdnet_v07_has_followup_buyback"], 0.0)
            self.assertEqual(nonissuer["tdnet_v07_has_fresh_ma"], 0.0)

    def test_v07_standard_buyback_titles_separate_fresh_and_followup(self) -> None:
        features = self._features_for_titles(
            [
                "自己株式の取得に関するお知らせ",
                "自己株式取得に関するお知らせ",
                "自己株式の市場買付けに関するお知らせ",
                "自社株式の取得状況に関するお知らせ",
            ]
        )
        for code in ("3001", "3002"):
            self.assertEqual(
                features.loc[code, "tdnet_v07_has_fresh_buyback"], 1.0
            )
            self.assertEqual(
                features.loc[code, "tdnet_v07_has_followup_buyback"], 0.0
            )
        for code in ("3003", "3004"):
            self.assertEqual(
                features.loc[code, "tdnet_v07_has_fresh_buyback"], 0.0
            )
            self.assertEqual(
                features.loc[code, "tdnet_v07_has_followup_buyback"], 1.0
            )

    def test_v07_nonissuer_buybacks_are_not_issuer_buyback_events(self) -> None:
        features = self._features_for_titles(
            [
                "スカイマーク株式会社による自己株式の取得への応募に関するお知らせ",
                "当社連結子会社（アスクル）による自己株式の取得に関するお知らせ",
                "子会社による自己株式の取得に係る決定",
                "関連会社における自社株式の市場買付けに関するお知らせ",
            ]
        )
        for code in features.index:
            self.assertEqual(
                features.loc[code, "tdnet_v07_has_fresh_buyback"], 0.0
            )
            self.assertEqual(
                features.loc[code, "tdnet_v07_has_followup_buyback"], 0.0
            )

    def test_v07_buyback_changes_explanations_and_investigations_are_followups(
        self,
    ) -> None:
        features = self._features_for_titles(
            [
                "自己株式の取得期間延長に関するお知らせ",
                "自己株式の取得方法追加に関するお知らせ",
                "一括取得型自己株式取得の事後調整完了に関するお知らせ",
                "CB発行及び自己株式の取得に関する補足説明資料",
                "自己株式の取得に関するQ&A",
                "自己株式取得に関する調査委員会の調査結果及び再発防止策",
                "分配可能額を超えた当期の中間配当金と自己株式取得に関するお知らせ",
                "自己株式取得に関する第三者委員会設置のお知らせ",
                "当社の自己株式取得の手法（ファシリティ型自己株式取得）に関するQ&amp;Aについて",
                "自己株式立会外買付取引（ToSTNeT-3）による自己株式買付の結果に関するお知らせ",
            ]
        )
        for code in features.index:
            self.assertEqual(
                features.loc[code, "tdnet_v07_has_fresh_buyback"], 0.0
            )
            self.assertEqual(
                features.loc[code, "tdnet_v07_has_followup_buyback"], 1.0
            )

    def test_v07_equity_and_cancellation_followups_are_not_fresh(self) -> None:
        features = self._features_for_titles(
            [
                "譲渡制限付株式としての自己株式処分に係る払込完了及び一部失権に関するお知らせ",
                "自己株式の消却に関するお知らせ",
                "自己株式の消却完了に関するお知らせ",
            ]
        )
        equity = features.loc["3001"]
        self.assertEqual(equity["tdnet_v07_has_fresh_equity"], 0.0)
        self.assertEqual(equity["tdnet_v07_has_followup_equity"], 1.0)
        self.assertEqual(equity["tdnet_v07_economic_family_count"], 0.0)
        self.assertEqual(equity["tdnet_v07_followup_family_count"], 1.0)

        fresh_cancellation = features.loc["3002"]
        self.assertEqual(
            fresh_cancellation["tdnet_v07_has_fresh_share_cancellation"], 1.0
        )
        self.assertEqual(
            fresh_cancellation["tdnet_v07_has_followup_share_cancellation"],
            0.0,
        )
        self.assertEqual(
            fresh_cancellation["tdnet_v07_economic_family_count"], 1.0
        )

        completed_cancellation = features.loc["3003"]
        self.assertEqual(
            completed_cancellation["tdnet_v07_has_fresh_share_cancellation"],
            0.0,
        )
        self.assertEqual(
            completed_cancellation[
                "tdnet_v07_has_followup_share_cancellation"
            ],
            1.0,
        )
        self.assertEqual(
            completed_cancellation["tdnet_v07_economic_family_count"], 0.0
        )
        self.assertEqual(
            completed_cancellation["tdnet_v07_followup_family_count"], 1.0
        )

    def test_v07_equity_requires_an_equity_instrument_or_compensation(self) -> None:
        features = self._features_for_titles(
            [
                "第３回無担保普通社債（私募債）の発行に係る払込完了に関するお知らせ",
                "当社代表取締役による当社株式の取得に関するお知らせ",
                "取締役に対する譲渡制限付株式報酬としての新株式発行に関するお知らせ",
            ]
        )
        for code in ("3001", "3002"):
            self.assertEqual(features.loc[code, "tdnet_v07_has_fresh_equity"], 0.0)
            self.assertEqual(
                features.loc[code, "tdnet_v07_has_followup_equity"], 0.0
            )
        self.assertEqual(features.loc["3001", "tdnet_v07_has_progress_stage"], 0.0)
        self.assertEqual(features.loc["3002", "tdnet_v07_has_fresh_ma"], 0.0)
        self.assertEqual(features.loc["3003", "tdnet_v07_has_fresh_equity"], 1.0)
        self.assertEqual(
            features.loc["3003", "tdnet_v07_economic_family_count"], 1.0
        )

    def test_v07_ma_subtypes_are_mutually_exclusive(self) -> None:
        values = [
            (
                "2026-07-20 16:00:00+09:00",
                "2301",
                "株式会社ABCの株式取得（子会社化）に関するお知らせ",
            ),
            (
                "2026-07-20 16:01:00+09:00",
                "2302",
                "連結子会社ABCの株式譲渡に関するお知らせ",
            ),
            (
                "2026-07-20 16:02:00+09:00",
                "2303",
                "株式交換による株式会社ABCの完全子会社化に関するお知らせ",
            ),
        ]
        disclosures = pd.DataFrame(
            {
                "published_at": [pd.Timestamp(value[0]) for value in values],
                "code": [value[1] for value in values],
                "name": [value[1] for value in values],
                "title": [value[2] for value in values],
                "url": [
                    f"https://example.test/ma-{index}.pdf"
                    for index in range(len(values))
                ],
            }
        )
        features = build_clean_tdnet_candidate_features(
            disclosures, self.sessions
        ).set_index("code")
        subtype_columns = [
            "tdnet_v07_has_ma_acquisition",
            "tdnet_v07_has_ma_divestiture",
            "tdnet_v07_has_ma_reorganization",
        ]
        self.assertEqual(features.loc["2301", subtype_columns].tolist(), [1, 0, 0])
        self.assertEqual(features.loc["2302", subtype_columns].tolist(), [0, 1, 0])
        self.assertEqual(features.loc["2303", subtype_columns].tolist(), [0, 0, 1])

    def test_v07_ma_patterns_reject_common_false_positives(self) -> None:
        features = self._features_for_titles(
            [
                "買収防衛策の継続に関するお知らせ",
                "固定資産の譲渡に関するお知らせ",
                "譲渡制限付株式報酬としての自己株式の処分に関するお知らせ",
                "代表取締役による当社株式取得に関するお知らせ",
                "株式会社ABCの株式取得（子会社化）に関するお知らせ",
                "連結子会社ABCの株式譲渡に関するお知らせ",
                "吸収分割による事業承継に関するお知らせ",
                "完全子会社間の吸収合併に関するお知らせ",
            ]
        )
        ma_columns = [
            "tdnet_v07_has_ma_acquisition",
            "tdnet_v07_has_ma_divestiture",
            "tdnet_v07_has_ma_reorganization",
            "tdnet_v07_has_ma_internal_reorganization",
        ]
        for code in ("3001", "3002", "3003", "3004"):
            self.assertEqual(features.loc[code, ma_columns].tolist(), [0, 0, 0, 0])
        self.assertEqual(features.loc["3005", ma_columns].tolist(), [1, 0, 0, 0])
        self.assertEqual(features.loc["3006", ma_columns].tolist(), [0, 1, 0, 0])
        self.assertEqual(features.loc["3007", ma_columns].tolist(), [0, 0, 1, 0])
        self.assertEqual(features.loc["3008", ma_columns].tolist(), [0, 0, 1, 1])

    def test_v07_ma_rejects_issuer_share_and_securities_sale_noise(self) -> None:
        features = self._features_for_titles(
            [
                "自己の株式の取得に関するお知らせ",
                "自己株式の公開買付けの結果及び取得終了に関するお知らせ",
                "政策保有株式の売却に関するお知らせ",
                "資産の譲渡完了に関するお知らせ（準共有持分10％の譲渡）",
                "単独株式移転による純粋持株会社体制への移行及び定款の一部変更に関するお知らせ",
            ]
        )
        for code in ("3001", "3002", "3003", "3004"):
            self.assertEqual(features.loc[code, "tdnet_v07_has_fresh_ma"], 0.0)
            self.assertEqual(features.loc[code, "tdnet_v07_has_followup_ma"], 0.0)
        reorganization = features.loc["3005"]
        self.assertEqual(reorganization["tdnet_v07_has_ma_reorganization"], 1.0)
        self.assertEqual(reorganization["tdnet_v07_has_fresh_ma"], 1.0)
        self.assertEqual(reorganization["tdnet_v07_has_followup_ma"], 0.0)

    def test_v07_economic_count_caps_attributes_and_excludes_stages(self) -> None:
        values = [
            (
                "2026-07-20 16:00:00+09:00",
                "2401",
                "自己株式取得に係る事項の決定（ToSTNeT-3による買付け）",
            ),
            (
                "2026-07-20 16:01:00+09:00",
                "2401",
                "第三者割当による新株式発行に関するお知らせ",
            ),
            (
                "2026-07-20 16:02:00+09:00",
                "2401",
                "株式会社ABCの株式取得（子会社化）に関するお知らせ",
            ),
            (
                "2026-07-20 16:03:00+09:00",
                "2401",
                "株式会社ABCの株式取得完了に関するお知らせ",
            ),
            (
                "2026-07-20 16:04:00+09:00",
                "2401",
                "（訂正）通期業績予想の上方修正に関するお知らせ",
            ),
            (
                "2026-07-20 16:05:00+09:00",
                "2402",
                "自己株式取得に係る事項の決定（ToSTNeT-3による買付け）",
            ),
        ]
        disclosures = pd.DataFrame(
            {
                "published_at": [pd.Timestamp(value[0]) for value in values],
                "code": [value[1] for value in values],
                "name": [value[1] for value in values],
                "title": [value[2] for value in values],
                "url": [
                    f"https://example.test/economic-{index}.pdf"
                    for index in range(len(values))
                ],
            }
        )
        features = build_clean_tdnet_candidate_features(
            disclosures, self.sessions
        ).set_index("code")
        mixed = features.loc["2401"]
        self.assertEqual(mixed["tdnet_v07_economic_family_count"], 3.0)
        self.assertAlmostEqual(
            mixed["tdnet_v07_economic_family_count_log1p"], np.log1p(3)
        )
        self.assertEqual(mixed["tdnet_v07_single_economic_family"], 0.0)
        one_family = features.loc["2402"]
        self.assertEqual(one_family["tdnet_v07_economic_family_count"], 1.0)
        self.assertEqual(one_family["tdnet_v07_single_economic_family"], 1.0)

    def test_v07_correction_attributes_do_not_become_fresh_economic_events(
        self,
    ) -> None:
        row = self._features_for_titles(
            ["（訂正）通期業績予想の上方修正に関するお知らせ"]
        ).loc["3001"]
        self.assertEqual(row["tdnet_v07_has_forecast_revision"], 1.0)
        self.assertEqual(row["tdnet_v07_economic_family_count"], 0.0)
        self.assertEqual(
            row["tdnet_v07_fresh_classified_economic_any"], 0.0
        )

    def test_empty_disclosures_return_stable_schema_and_attach_safely(self) -> None:
        disclosures = pd.DataFrame(
            {
                "published_at": pd.Series(
                    dtype="datetime64[ns, Asia/Tokyo]"
                ),
                "code": pd.Series(dtype="object"),
                "title": pd.Series(dtype="object"),
            }
        )
        features = build_clean_tdnet_candidate_features(
            disclosures, self.sessions
        )
        self.assertEqual(
            features.columns.tolist(),
            [
                "date",
                "code",
                *TDNET_CANDIDATE_COLUMNS,
                "tdnet_clean_feature_source_max_timestamp",
            ],
        )
        self.assertTrue(features.empty)
        self.assertEqual(features["date"].dtype, np.dtype("datetime64[ns]"))
        self.assertEqual(features["code"].dtype, np.dtype("object"))
        self.assertTrue(
            all(features[column].dtype == np.dtype("float32") for column in TDNET_CANDIDATE_COLUMNS)
        )
        self.assertEqual(
            str(features["tdnet_clean_feature_source_max_timestamp"].dtype),
            "datetime64[ns, Asia/Tokyo]",
        )

        panel = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-07-21"), pd.Timestamp("2026-07-21")],
                "code": ["9998", "9999"],
                "tdnet_source_complete": [True, False],
            }
        )
        attached = attach_clean_tdnet_candidate_features(
            panel, disclosures, self.sessions
        ).set_index("code")
        self.assertTrue(
            attached.loc["9998", list(TDNET_CANDIDATE_COLUMNS)].eq(0).all()
        )
        self.assertTrue(
            attached.loc["9999", list(TDNET_CANDIDATE_COLUMNS)].isna().all()
        )

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
        self.assertEqual(complete["tdnet_v07_observed_any"], 0.0)
        self.assertEqual(
            complete["tdnet_v07_fresh_classified_economic_any"], 0.0
        )
        self.assertTrue(complete[list(TDNET_CANDIDATE_COLUMNS)].eq(0).all())
        self.assertTrue(pd.isna(incomplete["tdnet_clean_any"]))
        self.assertTrue(pd.isna(incomplete["tdnet_v07_observed_any"]))
        self.assertTrue(incomplete[list(TDNET_CANDIDATE_COLUMNS)].isna().all())

    def test_nullable_missing_completeness_fails_closed(self) -> None:
        panel = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-07-21")],
                "code": ["1001"],
                "tdnet_source_complete": pd.Series(
                    [pd.NA], dtype="boolean"
                ),
            }
        )
        attached = attach_clean_tdnet_candidate_features(
            panel, self.disclosures, self.sessions
        ).iloc[0]
        self.assertTrue(
            attached[list(TDNET_CANDIDATE_COLUMNS)].isna().all()
        )

    def test_v07_semantic_columns_are_registered(self) -> None:
        features = build_clean_tdnet_candidate_features(
            self.disclosures, self.sessions
        )
        self.assertTrue(set(V07_EVENT_SEMANTIC_COLUMNS).issubset(features.columns))

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
