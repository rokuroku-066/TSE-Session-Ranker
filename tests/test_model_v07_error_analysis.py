from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.analyze_model_v07_errors import (
    add_strictly_prior_repeat_features,
    build_month_meta_features,
    fit_monthly_meta_gates,
    load_protocol,
    make_trade_picks,
)
from tse_session_ranker.exceptions import DataValidationError


class ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_four_gates_and_nonoverlapping_replays(self) -> None:
        protocol = load_protocol()
        self.assertEqual(len(protocol["gate_registry"]), 4)
        self.assertEqual(protocol["decision"]["display_slots"], 2)
        discovery_end = pd.Timestamp(protocol["periods"]["error_discovery"]["end"])
        previous = discovery_end
        for name in ("replay_a", "replay_b", "replay_c"):
            start = pd.Timestamp(protocol["periods"][name]["start"])
            end = pd.Timestamp(protocol["periods"][name]["end"])
            self.assertGreater(start, previous)
            previous = end


class ScoreGeometryTests(unittest.TestCase):
    @staticmethod
    def _scoring() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for date_index, date in enumerate(pd.to_datetime(["2024-09-02", "2024-09-03"])):
            for code_index in range(10):
                rows.append(
                    {
                        "date": date,
                        "code": f"{1000 + code_index}",
                        "name": f"security-{code_index}",
                        "oc_return_pct": float(code_index - date_index),
                        "outcome_observed": True,
                    }
                )
        return pd.DataFrame(rows)

    def test_target_outcomes_cannot_change_scores_margins_or_agreement(self) -> None:
        scoring = self._scoring()
        anchor = np.tile(np.linspace(1.0, 0.1, 10), 2)
        ridge = np.tile(np.linspace(0.1, 1.0, 10), 2)
        baseline = build_month_meta_features(
            scoring,
            {"anchor": anchor, "ridge": ridge},
            anchor_name="anchor",
        )
        changed = scoring.copy()
        changed["oc_return_pct"] = np.linspace(-99.0, 99.0, len(changed))
        rebuilt = build_month_meta_features(
            changed,
            {"anchor": anchor, "ridge": ridge},
            anchor_name="anchor",
        )
        meta = [
            "date",
            "code",
            "model_rank",
            "anchor_score_tail_z",
            "anchor_rank_percentile",
            "local_margin_iqr",
            "top_tail_slope_iqr",
            "family_rank_percentile_mean",
            "family_rank_percentile_std",
            "family_top2_vote_rate",
            "family_top5_vote_rate",
            "family_rank_correlation_mean",
        ]
        pd.testing.assert_frame_equal(baseline[meta], rebuilt[meta])
        self.assertEqual(baseline.groupby("date").size().tolist(), [2, 2])
        self.assertEqual(set(baseline["model_rank"]), {1, 2})

    def test_missing_anchor_or_invalid_family_scores_abort(self) -> None:
        scoring = self._scoring()
        with self.assertRaisesRegex(ValueError, "anchor"):
            build_month_meta_features(
                scoring,
                {"other": np.ones(len(scoring))},
                anchor_name="anchor",
            )
        invalid = np.ones(len(scoring))
        invalid[0] = np.nan
        with self.assertRaises(DataValidationError):
            build_month_meta_features(
                scoring,
                {"anchor": invalid},
                anchor_name="anchor",
            )


class RepeatHistoryTests(unittest.TestCase):
    @staticmethod
    def _display() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2024-09-02",
                        "2024-09-02",
                        "2024-09-03",
                        "2024-09-03",
                        "2024-09-04",
                        "2024-09-04",
                    ]
                ),
                "model_rank": [1, 2, 1, 2, 1, 2],
                "code": ["A", "B", "A", "C", "A", "D"],
                "anchor_model_score": [0.8, 0.7, 0.85, 0.6, 0.83, 0.55],
                "oc_return_pct": [-1.0, 1.0, -2.0, 2.0, 3.0, -3.0],
                "outcome_observed": [True] * 6,
            }
        )

    def test_repeat_features_use_only_strictly_prior_selections(self) -> None:
        baseline = add_strictly_prior_repeat_features(self._display())
        day2_a = baseline[
            baseline["date"].eq(pd.Timestamp("2024-09-03"))
            & baseline["code"].eq("A")
        ].iloc[0]
        self.assertEqual(day2_a["prior_selection_count_5"], 1.0)
        self.assertEqual(day2_a["sessions_since_selected"], 1.0)
        self.assertAlmostEqual(day2_a["last_selected_net_pct"], -1.2)
        self.assertTrue(day2_a["consecutive_selection"])
        self.assertLess(
            pd.Timestamp(day2_a["repeat_feature_source_max_date"]),
            pd.Timestamp(day2_a["date"]),
        )

        changed = self._display()
        changed.loc[
            changed["date"].eq(pd.Timestamp("2024-09-03"))
            & changed["code"].eq("A"),
            "oc_return_pct",
        ] = 99.0
        rebuilt = add_strictly_prior_repeat_features(changed)
        columns = [
            "prior_selection_count_5",
            "prior_selection_count_20",
            "sessions_since_selected",
            "last_selected_net_pct",
            "consecutive_selection",
            "anchor_score_change_since_selected",
        ]
        day2 = baseline[baseline["date"].eq(pd.Timestamp("2024-09-03"))]
        changed_day2 = rebuilt[rebuilt["date"].eq(pd.Timestamp("2024-09-03"))]
        pd.testing.assert_frame_equal(
            day2[columns].reset_index(drop=True),
            changed_day2[columns].reset_index(drop=True),
        )
        day3_original = baseline[
            baseline["date"].eq(pd.Timestamp("2024-09-04"))
            & baseline["code"].eq("A")
        ]["last_selected_net_pct"].iloc[0]
        day3_changed = rebuilt[
            rebuilt["date"].eq(pd.Timestamp("2024-09-04"))
            & rebuilt["code"].eq("A")
        ]["last_selected_net_pct"].iloc[0]
        self.assertNotEqual(day3_original, day3_changed)


class MonthlyMetaGateTests(unittest.TestCase):
    @staticmethod
    def _meta_rows(protocol: dict[str, object]) -> pd.DataFrame:
        dates = pd.bdate_range("2024-07-01", "2024-11-29")
        feature_columns = sorted(
            {
                column
                for values in protocol["meta_features"].values()  # type: ignore[index, union-attr]
                for column in values
            }
        )
        rows: list[dict[str, object]] = []
        for date_position, date in enumerate(dates):
            for model_rank in (1, 2):
                row: dict[str, object] = {
                    "date": date,
                    "model_rank": model_rank,
                    "code": f"{model_rank}-{date_position}",
                    "oc_return_pct": (
                        0.7 if (date_position + model_rank) % 3 else -0.8
                    ),
                    "outcome_observed": True,
                }
                for feature_position, column in enumerate(feature_columns):
                    row[column] = (
                        0.01 * date_position
                        + 0.1 * model_rank
                        + 0.001 * feature_position
                    )
                rows.append(row)
        return pd.DataFrame(rows)

    def test_target_month_outcome_mutation_cannot_change_its_gate(self) -> None:
        protocol = load_protocol()
        baseline_rows = self._meta_rows(protocol)
        baseline = fit_monthly_meta_gates(baseline_rows, protocol)
        changed_rows = baseline_rows.copy()
        september = changed_rows["date"].dt.to_period("M").eq(pd.Period("2024-09"))
        changed_rows.loc[september, "oc_return_pct"] = np.linspace(
            -50.0, 50.0, int(september.sum())
        )
        rebuilt = fit_monthly_meta_gates(changed_rows, protocol)
        compare = [
            "date",
            "model_rank",
            "code",
            "gate_id",
            "predicted_net_mean_pct",
            "predicted_negative_part_pct",
            "trade_utility_pct",
            "trade_decision",
            "meta_train_end",
        ]
        left = baseline[baseline["date"].dt.to_period("M").eq(pd.Period("2024-09"))]
        right = rebuilt[rebuilt["date"].dt.to_period("M").eq(pd.Period("2024-09"))]
        pd.testing.assert_frame_equal(
            left[compare].reset_index(drop=True),
            right[compare].reset_index(drop=True),
        )
        self.assertTrue((baseline["meta_train_end"] < baseline["date"]).all())


class CashSleeveTests(unittest.TestCase):
    @staticmethod
    def _display() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-11-01", "2024-11-01", "2024-11-05", "2024-11-05"]
                ),
                "model_rank": [1, 2, 1, 2],
                "code": ["A", "B", "C", "D"],
                "name": ["A", "B", "C", "D"],
                "oc_return_pct": [2.0, 10.0, -1.0, 4.0],
                "outcome_observed": [True] * 4,
            }
        )

    def test_each_rejected_slot_stays_cash_without_rank_replacement(self) -> None:
        display = self._display()
        predictions = display[["date", "model_rank", "code"]].copy()
        predictions["gate_id"] = "gate"
        predictions["predicted_net_mean_pct"] = [1.0, -1.0, -1.0, 1.0]
        predictions["predicted_negative_part_pct"] = 0.0
        predictions["trade_utility_pct"] = predictions["predicted_net_mean_pct"]
        predictions["trade_decision"] = [True, False, False, True]
        predictions["meta_train_end"] = pd.Timestamp("2024-10-31")
        predictions["meta_train_sessions"] = 60
        trades = make_trade_picks(display, predictions, gate_id="gate")
        self.assertEqual(trades["code"].tolist(), display["code"].tolist())
        self.assertEqual(set(trades["model_rank"]), {1, 2})
        daily = trades.groupby("date")["net_sleeve_return_pct_20bp"].sum()
        self.assertAlmostEqual(daily.loc[pd.Timestamp("2024-11-01")], 0.5 * (2.0 - 0.2))
        self.assertAlmostEqual(daily.loc[pd.Timestamp("2024-11-05")], 0.5 * (4.0 - 0.2))
        self.assertEqual(int(trades.groupby("date")["executed"].any().sum()), 2)
        self.assertEqual(int(trades["executed"].sum()), 2)

    def test_duplicate_gate_slots_are_rejected(self) -> None:
        display = self._display()
        predictions = display[["date", "model_rank", "code"]].copy()
        predictions["gate_id"] = "gate"
        predictions["predicted_net_mean_pct"] = 1.0
        predictions["predicted_negative_part_pct"] = 0.0
        predictions["trade_utility_pct"] = 1.0
        predictions["trade_decision"] = True
        predictions["meta_train_end"] = pd.Timestamp("2024-10-31")
        predictions["meta_train_sessions"] = 60
        duplicated = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
        with self.assertRaises(DataValidationError):
            make_trade_picks(display, duplicated, gate_id="gate")


if __name__ == "__main__":
    unittest.main()
