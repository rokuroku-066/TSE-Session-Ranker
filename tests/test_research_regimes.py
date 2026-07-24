from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from tse_session_ranker.exceptions import DataValidationError, LeakageError
from tse_session_ranker.research_regimes import (
    PRIOR_MARKET_OC_BREADTH,
    PRIOR_MARKET_OC_BREADTH_SOURCE_DATE,
    V09_VALID_OC_BREADTH_RANK2_POLICY,
    add_prior_market_oc_breadth,
    select_valid_oc_breadth_rank2,
)


def _panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    returns = {
        "2026-07-17": {"1001": 1.0, "1002": -1.0, "1003": np.nan},
        "2026-07-21": {"1001": 2.0, "1002": 0.0, "1003": -1.0},
        "2026-07-22": {"1001": -2.0, "1002": 1.0, "1003": 3.0},
    }
    for date, by_code in returns.items():
        for code, value in by_code.items():
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "code": code,
                    "oc_return_pct": value,
                    "traded": not pd.isna(value),
                    "outcome_observed": True,
                    "source_complete": True,
                }
            )
    return pd.DataFrame(rows)


def _picks() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date in pd.to_datetime(
        ["2026-07-17", "2026-07-21", "2026-07-22"]
    ):
        for candidate, base in (("L4", 1000), ("L6", 2000)):
            for rank in (1, 2):
                rows.append(
                    {
                        "date": date,
                        "candidate": candidate,
                        "model_rank": rank,
                        "code": str(base + rank),
                    }
                )
    return pd.DataFrame(rows)


class PriorMarketOcBreadthTests(unittest.TestCase):
    def test_feature_excludes_no_trade_and_shifts_one_session(self) -> None:
        result = add_prior_market_oc_breadth(_panel())
        by_date = result.groupby("date", sort=True).agg(
            breadth=(PRIOR_MARKET_OC_BREADTH, "first"),
            source=(PRIOR_MARKET_OC_BREADTH_SOURCE_DATE, "first"),
        )
        self.assertTrue(pd.isna(by_date.loc[pd.Timestamp("2026-07-17"), "breadth"]))
        self.assertAlmostEqual(
            float(by_date.loc[pd.Timestamp("2026-07-21"), "breadth"]),
            0.5,
        )
        self.assertAlmostEqual(
            float(by_date.loc[pd.Timestamp("2026-07-22"), "breadth"]),
            1.0 / 3.0,
            places=7,
        )
        self.assertEqual(
            by_date.loc[pd.Timestamp("2026-07-22"), "source"],
            pd.Timestamp("2026-07-21"),
        )

    def test_target_and_future_outcomes_cannot_change_target_feature(self) -> None:
        target = pd.Timestamp("2026-07-21")
        baseline = add_prior_market_oc_breadth(_panel())
        changed = _panel()
        changed.loc[
            changed["date"].ge(target) & changed["traded"],
            "oc_return_pct",
        ] *= -7.0
        rebuilt = add_prior_market_oc_breadth(changed)
        columns = [
            "date",
            "code",
            PRIOR_MARKET_OC_BREADTH,
            PRIOR_MARKET_OC_BREADTH_SOURCE_DATE,
        ]
        left = baseline.loc[baseline["date"].le(target), columns].reset_index(
            drop=True
        )
        right = rebuilt.loc[rebuilt["date"].le(target), columns].reset_index(
            drop=True
        )
        assert_frame_equal(left, right, check_exact=True)

    def test_row_order_does_not_change_feature_values(self) -> None:
        baseline = add_prior_market_oc_breadth(_panel())
        shuffled = add_prior_market_oc_breadth(
            _panel().sample(frac=1.0, random_state=41)
        )
        columns = [
            "date",
            "code",
            PRIOR_MARKET_OC_BREADTH,
            PRIOR_MARKET_OC_BREADTH_SOURCE_DATE,
        ]
        left = baseline[columns].sort_values(["date", "code"]).reset_index(
            drop=True
        )
        right = shuffled[columns].sort_values(["date", "code"]).reset_index(
            drop=True
        )
        assert_frame_equal(left, right, check_exact=True)

    def test_duplicate_key_and_non_boolean_flags_fail_closed(self) -> None:
        duplicate = pd.concat([_panel(), _panel().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(DataValidationError, "duplicate date/code"):
            add_prior_market_oc_breadth(duplicate)
        invalid = _panel()
        invalid["source_complete"] = invalid["source_complete"].astype(object)
        invalid.loc[0, "source_complete"] = "yes"
        with self.assertRaisesRegex(DataValidationError, "source_complete"):
            add_prior_market_oc_breadth(invalid)


class ValidOcBreadthRank2PolicyTests(unittest.TestCase):
    def _context(self) -> pd.DataFrame:
        return (
            add_prior_market_oc_breadth(_panel())
            .drop_duplicates("date")
            [
                [
                    "date",
                    PRIOR_MARKET_OC_BREADTH,
                    PRIOR_MARKET_OC_BREADTH_SOURCE_DATE,
                ]
            ]
            .reset_index(drop=True)
        )

    def test_policy_uses_low_high_and_missing_fallback(self) -> None:
        selected = select_valid_oc_breadth_rank2(
            _picks(),
            self._context(),
            low_breadth_candidate="L4",
            high_breadth_candidate="L6",
        )
        self.assertEqual(selected["candidate"].tolist(), ["L4", "L6", "L4"])
        self.assertEqual(selected["model_rank"].tolist(), [2, 2, 2])
        self.assertEqual(selected["portfolio_weight"].tolist(), [1.0] * 3)
        self.assertEqual(
            selected["shadow_policy_id"].unique().tolist(),
            [V09_VALID_OC_BREADTH_RANK2_POLICY],
        )

    def test_nonprior_context_and_missing_model_rank_fail_closed(self) -> None:
        context = self._context()
        context.loc[1, PRIOR_MARKET_OC_BREADTH_SOURCE_DATE] = context.loc[
            1, "date"
        ]
        with self.assertRaisesRegex(LeakageError, "non-prior"):
            select_valid_oc_breadth_rank2(
                _picks(),
                context,
                low_breadth_candidate="L4",
                high_breadth_candidate="L6",
            )

        missing = _picks()
        missing = missing[
            ~(
                missing["date"].eq(pd.Timestamp("2026-07-21"))
                & missing["candidate"].eq("L6")
                & missing["model_rank"].eq(2)
            )
        ]
        with self.assertRaisesRegex(DataValidationError, "exactly one"):
            select_valid_oc_breadth_rank2(
                missing,
                self._context(),
                low_breadth_candidate="L4",
                high_breadth_candidate="L6",
            )


if __name__ == "__main__":
    unittest.main()
