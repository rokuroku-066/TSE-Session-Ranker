from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.finalize_logit_v04 import (
    BOUNDED_DAILY_COLUMNS,
    HOLDOUT_START,
    MODEL_DESIGN_END,
    ModelSpec,
    _periods,
    _verify_blind_dates,
    _verify_parsed_jpx_manifest,
    add_bounded_daily_features,
    evaluate_spec,
    feature_blocks,
    make_estimator as make_research_estimator,
    objective_and_weights,
)


class FrozenFeatureRegistryTests(unittest.TestCase):
    def test_registry_has_six_unique_bounded_blocks(self) -> None:
        blocks = feature_blocks()
        self.assertEqual(
            list(blocks),
            [
                "legacy_v03",
                "bounded_daily",
                "session_shape",
                "session_market",
                "session_tdnet",
                "session_market_tdnet",
            ],
        )
        self.assertEqual(len(blocks["legacy_v03"]), 18)
        self.assertEqual(len(blocks["session_market_tdnet"]), 62)
        for columns in blocks.values():
            self.assertEqual(len(columns), len(set(columns)))
            self.assertLessEqual(len(columns), 64)

    def test_model_spec_id_binds_every_tunable_field(self) -> None:
        first = ModelSpec(
            "session_market",
            "net_positive_20bp",
            c=0.03,
            training_horizon_sessions=504,
            class_weight=None,
        )
        second = ModelSpec(
            "session_market",
            "net_positive_20bp",
            c=0.08,
            training_horizon_sessions=None,
            class_weight="balanced",
        )
        self.assertNotEqual(first.id, second.id)
        self.assertIn("C0.03", first.id)
        self.assertIn("h504", first.id)
        self.assertIn("cwnone", first.id)

    def test_legacy_control_estimator_matches_v03_training_pipeline(self) -> None:
        from tse_session_ranker.config import RankerConfig
        from tse_session_ranker.training import (
            date_equal_weights,
            make_estimator as make_production_estimator,
        )

        rng = np.random.default_rng(7)
        features = feature_blocks()["legacy_v03"]
        frame = pd.DataFrame(rng.normal(size=(60, len(features))), columns=features)
        frame["date"] = np.repeat(pd.bdate_range("2025-01-06", periods=12), 5)
        frame["code"] = np.tile(["1000", "1001", "1002", "1003", "1004"], 12)
        target = pd.Series(np.tile([0, 0, 1, 0, 1], 12), index=frame.index)
        weights = date_equal_weights(frame)
        production = make_production_estimator(RankerConfig())
        production.fit(
            frame[list(features)], target, model__sample_weight=weights
        )
        research = make_research_estimator(
            ModelSpec(
                "legacy_v03",
                "positive_session",
                class_weight="balanced",
                train_start="2024-11-06",
                min_train_sessions=20,
                legacy_estimator_class_weight=True,
            )
        )
        research.fit(
            frame[list(features)], target, model__sample_weight=weights
        )
        np.testing.assert_allclose(
            research.predict_proba(frame[list(features)]),
            production.predict_proba(frame[list(features)]),
            atol=1e-12,
        )


class ProfitTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-04"] * 5 + ["2024-01-05"] * 5
                ),
                "oc_return_pct": [
                    -2.0,
                    -0.1,
                    0.1,
                    0.5,
                    3.0,
                    -1.0,
                    0.0,
                    0.3,
                    0.8,
                    2.0,
                ],
            }
        )

    def test_all_objectives_preserve_equal_total_date_weight(self) -> None:
        objectives = (
            "positive_session",
            "net_positive_20bp",
            "net_positive_magnitude_weighted",
            "same_day_top_quintile",
        )
        for objective in objectives:
            for class_weight in (None, "balanced"):
                target, weights = objective_and_weights(
                    self.frame, objective, class_weight
                )
                self.assertEqual(target.nunique(), 2)
                totals = weights.groupby(self.frame["date"]).sum().to_numpy()
                np.testing.assert_allclose(totals, np.ones(2), atol=1e-12)
                if class_weight == "balanced":
                    class_totals = weights.groupby(target).sum().to_numpy()
                    np.testing.assert_allclose(
                        class_totals,
                        np.repeat(len(totals) / 2.0, 2),
                        atol=1e-8,
                    )

    def test_cost_target_uses_strict_twenty_basis_point_threshold(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-04"] * 4),
                "oc_return_pct": [0.19, 0.20, 0.21, 1.0],
            }
        )
        target, _ = objective_and_weights(
            frame, "net_positive_20bp", None
        )
        self.assertEqual(target.tolist(), [0, 0, 1, 1])

    def test_top_quintile_is_computed_within_each_training_date(self) -> None:
        target, _ = objective_and_weights(
            self.frame, "same_day_top_quintile", None
        )
        counts = target.groupby(self.frame["date"]).sum()
        self.assertEqual(counts.tolist(), [1, 1])


class ChronologyTests(unittest.TestCase):
    def test_model_design_ends_before_sealed_holdout(self) -> None:
        self.assertLess(MODEL_DESIGN_END, HOLDOUT_START)

    def test_quarter_periods_never_cross_requested_boundaries(self) -> None:
        start = pd.Timestamp("2023-02-15")
        end = pd.Timestamp("2023-11-17")
        periods = _periods(start, end, "Q")
        self.assertEqual(periods[0][0], start)
        self.assertEqual(periods[-1][1], end)
        self.assertTrue(all(left <= right for left, right, _ in periods))


class ScheduledDayEvaluationTests(unittest.TestCase):
    def test_no_candidate_days_are_retained_as_zero_return(self) -> None:
        train_dates = pd.bdate_range("2024-01-02", periods=25)
        score_dates = pd.bdate_range(train_dates[-1] + pd.offsets.BDay(), periods=20)
        dates = train_dates.append(score_dates)
        features = feature_blocks()["legacy_v03"]
        panel = pd.DataFrame(
            {
                "date": dates,
                "code": "1000",
                "name": "sample",
                "eligible": [False] * 25 + [True] * 10 + [False] * 10,
                "training_eligible": [True] * 25 + [False] * 20,
                "label": [float(index % 2) for index in range(45)],
                "oc_return_pct": [0.5 if index % 2 else -0.5 for index in range(45)],
                "outcome_observed": True,
                "source_complete": True,
                "universe_source_complete": True,
            }
        )
        for position, column in enumerate(features):
            panel[column] = np.arange(len(panel), dtype=float) + position / 100.0
        spec = ModelSpec(
            "legacy_v03",
            "positive_session",
            class_weight=None,
            train_start=str(train_dates[0].date()),
            min_train_sessions=20,
        )
        result = evaluate_spec(
            panel,
            spec,
            evaluation_start=score_dates[0],
            evaluation_end=score_dates[-1],
            evaluation_sessions=score_dates,
            retrain_frequency="M",
            bootstrap_samples=100,
        )
        self.assertEqual(len(result.daily_20bp), 20)
        self.assertEqual(result.summary["display_days"], 10)
        self.assertEqual(result.summary["candidate_display_rate"], 0.5)
        np.testing.assert_allclose(result.daily_20bp.iloc[-10:], 0.0)


class BoundedTechnicalFeatureTests(unittest.TestCase):
    def test_target_session_prices_cannot_change_target_features(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=70)
        rows = []
        for code, offset in (("1000", 0.0), ("1001", 10.0)):
            for index, date in enumerate(dates):
                close = 100.0 + offset + index * 0.2
                rows.append(
                    {
                        "date": date,
                        "code": code,
                        "open": close - 0.1,
                        "high": close + 1.0,
                        "low": close - 1.0,
                        "close": close,
                        "overnight": 0.1,
                        "eligible": True,
                        "training_eligible": True,
                        "oc_last": 0.2,
                        "oc_mean_5": 0.1,
                        "oc_mean_20": 0.05,
                        "oc_std_20": 1.0,
                        "overnight_last": 0.1,
                        "overnight_mean_20": 0.02,
                        "night_day_corr_60": 0.0,
                        "atr14_pct": 1.5,
                    }
                )
        original = pd.DataFrame(rows)
        changed = original.copy()
        target_mask = changed["date"].eq(dates[-1]) & changed["code"].eq("1000")
        changed.loc[target_mask, ["open", "high", "low", "close"]] = [
            900.0,
            1_100.0,
            800.0,
            1_000.0,
        ]
        first = add_bounded_daily_features(original)
        second = add_bounded_daily_features(changed)
        columns = [*BOUNDED_DAILY_COLUMNS, "price_history_continuous_60"]
        pd.testing.assert_series_equal(
            first.loc[target_mask, columns].iloc[0],
            second.loc[target_mask, columns].iloc[0],
        )


class SealedHoldoutBindingTests(unittest.TestCase):
    def test_parsed_manifest_must_bind_exact_sources_and_parser(self) -> None:
        import hashlib
        import tempfile
        from pathlib import Path

        from tse_session_ranker.data.jpx import PARSER_VERSION

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "stq_20250804.pdf"
            source.write_text("payload", encoding="utf-8")
            source_hash = hashlib.sha256(b"payload").hexdigest()
            parsed = {
                "parser_version": PARSER_VERSION,
                "inputs": [
                    {
                        "path": str(source),
                        "sha256": source_hash,
                        "ordinary_rows": 1,
                        "parsed_rows": 1,
                        "full_session_rows": 1,
                        "partial_session_rows": 0,
                        "no_trade_rows": 0,
                        "rejected_rows": 0,
                        "parser_version": PARSER_VERSION,
                        "source_format": (
                            "jpx_stock_quotations_auction_regular_way_"
                            "domestic_ordinary"
                        ),
                    }
                ],
            }
            blind = pd.DataFrame({"source_file": [str(source)]})
            _verify_parsed_jpx_manifest(parsed, [source], blind)
            parsed["inputs"][0]["sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "parsed JPX"):
                _verify_parsed_jpx_manifest(parsed, [source], blind)

            parsed["inputs"][0]["sha256"] = source_hash
            parsed["inputs"][0]["rejected_rows"] = 1
            with self.assertRaisesRegex(ValueError, "rejected or missing"):
                _verify_parsed_jpx_manifest(parsed, [source], blind)

    def test_blind_rows_must_match_sealed_dates_and_source_files(self) -> None:
        sealed = {
            "files": [
                {"path": "/sealed/stq_20250804.pdf"},
            ]
        }
        blind = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-08-01", "2025-08-04"]),
                "source_file": ["stq_20250801.txt", "stq_20250804.txt"],
            }
        )
        _verify_blind_dates(blind, sealed)
        blind.loc[1, "source_file"] = "stq_20250805.txt"
        with self.assertRaisesRegex(ValueError, "provenance"):
            _verify_blind_dates(blind, sealed)


if __name__ == "__main__":
    unittest.main()
