from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from tse_session_ranker.api import SessionRanker
from tse_session_ranker.data.market_context import (
    MARKET_CONTEXT_COLUMNS,
    append_market_context,
    normalize_market_context,
)
from tse_session_ranker.exceptions import DataValidationError, LeakageError


RETURN_DEFINITION = "previous_cash_close_to_08:58:59_JST"


def _row(
    date: str,
    observed_at: object,
    nikkei_return: float = 1.2,
    topix_return: float = 0.8,
) -> dict[str, object]:
    return {
        "date": date,
        "observed_at": observed_at,
        "nikkei_return_pct": nikkei_return,
        "topix_return_pct": topix_return,
        "nikkei_level": 42_100.0,
        "topix_level": 3_050.0,
        "source": "forward-snapshot",
        "return_definition": RETURN_DEFINITION,
    }


class NormalizeMarketContextTests(unittest.TestCase):
    def test_timezone_is_converted_to_jst_and_cutoff_is_inclusive(self) -> None:
        context = pd.DataFrame(
            [
                _row(
                    "2026-07-21",
                    pd.Timestamp("2026-07-20 23:58:59", tz="UTC"),
                )
            ]
        )
        result = normalize_market_context(context)
        self.assertEqual(
            result.iloc[0]["observed_at"],
            pd.Timestamp("2026-07-21 08:58:59", tz="Asia/Tokyo"),
        )
        self.assertEqual(result.iloc[0]["date"], pd.Timestamp("2026-07-21"))
        self.assertEqual(tuple(result.columns), MARKET_CONTEXT_COLUMNS)

    def test_after_cutoff_naive_time_and_wrong_local_date_are_rejected(self) -> None:
        late = pd.DataFrame(
            [_row("2026-07-21", "2026-07-21 08:59:00+09:00")]
        )
        with self.assertRaisesRegex(LeakageError, "after cutoff"):
            normalize_market_context(late)
        naive = pd.DataFrame([_row("2026-07-21", "2026-07-21 08:58:00")])
        with self.assertRaisesRegex(LeakageError, "timezone-aware"):
            normalize_market_context(naive)
        wrong_date = pd.DataFrame(
            [_row("2026-07-21", "2026-07-20 08:58:00+09:00")]
        )
        with self.assertRaisesRegex(LeakageError, "target date"):
            normalize_market_context(wrong_date)

    def test_identical_dates_are_idempotent_but_any_difference_conflicts(self) -> None:
        first = _row("2026-07-21", "2026-07-21 08:58:00+09:00")
        same_in_utc = dict(first)
        same_in_utc["observed_at"] = "2026-07-20 23:58:00+00:00"
        identical = normalize_market_context(pd.DataFrame([first, same_in_utc]))
        self.assertEqual(len(identical), 1)

        changed = dict(first)
        changed["topix_return_pct"] = 0.81
        with self.assertRaisesRegex(DataValidationError, "conflicting"):
            normalize_market_context(pd.DataFrame([first, changed]))
        changed_time = dict(first)
        changed_time["observed_at"] = "2026-07-21 08:58:01+09:00"
        with self.assertRaisesRegex(DataValidationError, "conflicting"):
            normalize_market_context(pd.DataFrame([first, changed_time]))

    def test_return_definition_is_required_and_cannot_be_mixed(self) -> None:
        missing = _row("2026-07-21", "2026-07-21 08:58:00+09:00")
        missing["return_definition"] = " "
        with self.assertRaisesRegex(DataValidationError, "non-empty"):
            normalize_market_context(pd.DataFrame([missing]))
        different = _row("2026-07-22", "2026-07-22 08:58:00+09:00")
        different["return_definition"] = "night_close_to_08:58:59_JST"
        with self.assertRaisesRegex(DataValidationError, "cannot mix"):
            normalize_market_context(pd.DataFrame([_row(
                "2026-07-21", "2026-07-21 08:58:00+09:00"
            ), different]))

    def test_returns_and_optional_levels_must_be_finite(self) -> None:
        for column, value in (
            ("nikkei_return_pct", np.nan),
            ("topix_return_pct", np.inf),
            ("nikkei_level", np.inf),
        ):
            with self.subTest(column=column):
                row = _row("2026-07-21", "2026-07-21 08:58:00+09:00")
                row[column] = value
                with self.assertRaises(DataValidationError):
                    normalize_market_context(pd.DataFrame([row]))


class AppendMarketContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = pd.DataFrame(
            [_row("2026-07-21", "2026-07-21 08:58:00+09:00")]
        )
        self.next = pd.DataFrame(
            [_row("2026-07-23", "2026-07-23 08:58:00+09:00", 0.4, 0.2)]
        )

    def test_append_preserves_past_is_idempotent_and_never_fills_dates(self) -> None:
        original = normalize_market_context(self.first)
        appended = append_market_context(original, self.next)
        self.assertEqual(
            appended["date"].tolist(),
            [pd.Timestamp("2026-07-21"), pd.Timestamp("2026-07-23")],
        )
        assert_frame_equal(appended.iloc[[0]].reset_index(drop=True), original)
        repeated = append_market_context(appended, self.next)
        assert_frame_equal(repeated, appended)

    def test_conflicting_overlap_and_retroactive_backfill_are_rejected(self) -> None:
        current = append_market_context(self.first, self.next)
        conflict = self.next.copy()
        conflict["nikkei_return_pct"] = 0.41
        with self.assertRaisesRegex(DataValidationError, "conflicting"):
            append_market_context(current, conflict)
        backfill = pd.DataFrame(
            [_row("2026-07-22", "2026-07-22 08:58:00+09:00", 0.3, 0.1)]
        )
        with self.assertRaisesRegex(DataValidationError, "retroactive"):
            append_market_context(current, backfill)

    def test_append_rejects_a_different_return_definition(self) -> None:
        changed = self.next.copy()
        changed["return_definition"] = "night_close_to_08:58:59_JST"
        with self.assertRaisesRegex(DataValidationError, "different"):
            append_market_context(self.first, changed)

    def test_session_ranker_persists_an_append_only_context(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "market_context.pkl"
            ranker = SessionRanker()
            first = ranker.ingest_market_context(self.first, output=output)
            result = ranker.ingest_market_context(
                self.next, output=output, existing=output
            )
            self.assertTrue(output.is_file())
            self.assertEqual(len(first), 1)
            self.assertEqual(len(result), 2)
            assert_frame_equal(result.iloc[[0]].reset_index(drop=True), first)


if __name__ == "__main__":
    unittest.main()
