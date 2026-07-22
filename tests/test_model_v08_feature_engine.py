from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from research.model_v08_feature_engine import (
    G0,
    add_derived_features,
    add_prior_close,
)


def _feature_fixture() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-04", periods=90)
    parts: list[pd.DataFrame] = []
    for code_number, code in enumerate(("1001", "2002")):
        position = np.arange(len(dates), dtype=float)
        phase = position / 7.0 + code_number * 0.4
        realised = 1.2 * np.sin(phase) - 0.25 * np.cos(phase / 2.0)
        prior_dates = pd.Series(dates).shift(1)
        prior_dates.iloc[0] = dates[0] - pd.Timedelta(days=1)
        frame = pd.DataFrame(
            {
                "date": dates,
                "code": code,
                "label": (realised > 0).astype(float),
                "oc_return_pct": realised,
                "price_eligible": True,
                "feature_source_max_date": prior_dates.to_numpy(),
                "strict_prior_close": 500.0 + 20.0 * code_number + position,
                "flat_oc_rate_20": 0.02 + 0.01 * (position % 3),
                "no_trade_rate_60": 0.01 + 0.005 * (position % 2),
                "zero_range_rate_20": 0.01,
                "cc_reversal_1_atr": np.sin(phase / 3.0),
                "range_shock_1_20": 1.0 + 0.2 * np.cos(phase),
                "cc_momentum_3_atr": np.cos(phase / 4.0),
                "prior_session_close_location": 0.5 + 0.3 * np.sin(phase / 5.0),
                "session_range_ratio_5_20": 1.0 + 0.1 * np.cos(phase / 2.0),
                "overnight_std_20": 0.8 + 0.1 * np.sin(phase).clip(-0.5, 0.5),
                "gap_response_beta_60": 0.2 * np.cos(phase / 3.0),
                "gap_fill_rate_20": 0.45 + 0.1 * np.sin(phase / 4.0),
                "gap_frequency_20": 0.35 + 0.1 * np.cos(phase / 5.0),
                "prior_market_breadth": 0.1 * np.sin(position / 9.0),
                "prior_market_tail_balance": 0.1 * np.cos(position / 8.0),
                "prior_market_dispersion": 1.0 + 0.1 * np.sin(position / 6.0),
                "prior_market_overnight_return_pct": 0.2 * np.cos(position / 11.0),
                "market_cc_momentum_5": 0.3 * np.sin(position / 10.0),
                "market_cc_vol_ratio_5_20": 1.0 + 0.1 * np.cos(position / 12.0),
            }
        )
        frame["oc_last"] = 0.5 * np.sin(phase)
        frame["oc_mean_5"] = 0.3 * np.sin(phase / 2.0)
        frame["oc_mean_20"] = 0.2 * np.sin(phase / 3.0)
        frame["oc_mean_60"] = 0.1 * np.sin(phase / 5.0)
        frame["oc_win_20"] = 0.5 + 0.1 * np.sin(phase / 4.0)
        frame["oc_std_20"] = 1.0 + 0.2 * np.cos(phase / 4.0)
        frame["overnight_last"] = 0.4 * np.cos(phase)
        frame["overnight_mean_20"] = 0.2 * np.cos(phase / 3.0)
        frame["overnight_mean_60"] = 0.1 * np.cos(phase / 5.0)
        frame["night_day_corr_60"] = 0.2 * np.sin(phase / 6.0)
        frame["xrank_atr14_pct"] = 0.5 * np.sin(phase / 4.0)
        frame["xrank_close_momentum_5"] = np.sin(phase / 3.0)
        frame["xrank_close_momentum_20"] = np.sin(phase / 5.0)
        frame["xrank_close_momentum_60"] = np.sin(phase / 8.0)
        frame["xrank_prior_close_location_20"] = np.cos(phase / 6.0)
        parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    missing = sorted(set(G0) - set(result.columns))
    if missing:  # pragma: no cover - fixture construction guard
        raise AssertionError(missing)
    return result


class ModelV08FeatureEngineTest(unittest.TestCase):
    def test_prior_close_join_is_positional_for_non_range_index(self) -> None:
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-04", "2024-01-05", "2024-01-04", "2024-01-05"]
                ),
                "code": ["1001", "1001", "2002", "2002"],
                "close": [101.0, 102.0, 201.0, 202.0],
            }
        )
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-08", "2024-01-08", "2024-01-05"]),
                "code": ["2002", "1001", "1001"],
                "feature_source_max_date": pd.to_datetime(
                    ["2024-01-05", "2024-01-05", "2024-01-04"]
                ),
            },
            index=[91, 7, 44],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily.pkl"
            joblib.dump(daily, path)
            add_prior_close(panel, path)
        np.testing.assert_allclose(
            panel["strict_prior_close"].to_numpy(), [202.0, 102.0, 101.0]
        )

    def test_shuffled_non_range_panel_restores_exact_feature_positions(self) -> None:
        reference = _feature_fixture()
        reference_groups, reference_masks = add_derived_features(reference)

        shuffled = _feature_fixture().sample(frac=1.0, random_state=31).copy()
        shuffled.index = np.arange(10_000, 10_000 + len(shuffled)) * 3
        shuffled_groups, shuffled_masks = add_derived_features(shuffled)
        self.assertEqual(reference_groups, shuffled_groups)

        key = ["date", "code"]
        reference_order = reference.sort_values(key, kind="stable").reset_index(drop=True)
        shuffled_order = shuffled.sort_values(key, kind="stable").reset_index(drop=True)
        columns = [
            column
            for values in reference_groups.values()
            for column in values
        ]
        np.testing.assert_allclose(
            reference_order[columns].to_numpy(dtype=float),
            shuffled_order[columns].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
        for mask_id in reference_masks:
            left = pd.DataFrame(
                {
                    "date": reference["date"],
                    "code": reference["code"],
                    "mask": reference_masks[mask_id].to_numpy(),
                }
            ).sort_values(key, kind="stable")
            right = pd.DataFrame(
                {
                    "date": shuffled["date"],
                    "code": shuffled["code"],
                    "mask": shuffled_masks[mask_id].to_numpy(),
                }
            ).sort_values(key, kind="stable")
            np.testing.assert_array_equal(
                left["mask"].to_numpy(), right["mask"].to_numpy()
            )

    def test_future_outcome_mutation_cannot_change_past_features(self) -> None:
        baseline = _feature_fixture()
        mutated = _feature_fixture()
        cutoff = pd.Timestamp("2024-03-15")
        future = mutated["date"].gt(cutoff)
        mutated.loc[future, "oc_return_pct"] = np.linspace(
            -50.0, 50.0, int(future.sum())
        )
        mutated.loc[future, "label"] = (
            mutated.loc[future, "oc_return_pct"] > 0
        ).astype(float)
        baseline_groups, baseline_masks = add_derived_features(baseline)
        mutated_groups, mutated_masks = add_derived_features(mutated)
        past = baseline["date"].le(cutoff)
        columns = [
            column for values in baseline_groups.values() for column in values
        ]
        np.testing.assert_allclose(
            baseline.loc[past, columns].to_numpy(dtype=float),
            mutated.loc[past, columns].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
        for mask_id in baseline_masks:
            np.testing.assert_array_equal(
                baseline_masks[mask_id].loc[past].to_numpy(),
                mutated_masks[mask_id].loc[past].to_numpy(),
            )

    def test_duplicate_date_code_fails_closed(self) -> None:
        panel = _feature_fixture()
        duplicated = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate date/code"):
            add_derived_features(duplicated)


if __name__ == "__main__":
    unittest.main()
