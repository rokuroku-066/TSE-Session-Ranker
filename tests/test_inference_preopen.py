from __future__ import annotations

import unittest

import pandas as pd

from tse_session_ranker.exceptions import DataValidationError, LeakageError
from tse_session_ranker.inference import apply_preopen_overlay, rank_candidates
from tse_session_ranker.data.preopen import normalize_preopen_snapshots


class InferencePreopenTests(unittest.TestCase):
    def test_rank_filters_first_and_ties_by_code(self) -> None:
        scored = pd.DataFrame(
            {
                "code": ["1003", "1002", "1001", "9999"],
                "eligible": [True, True, True, False],
                "model_score": [0.7, 0.8, 0.8, 0.99],
            }
        )
        selected = rank_candidates(scored, 2)
        self.assertEqual(selected["code"].tolist(), ["1001", "1002"])
        self.assertEqual(selected["model_rank"].tolist(), [1, 2])
        with self.assertRaises(ValueError):
            rank_candidates(scored, 0)

    def test_overlay_vetoes_without_changing_rank(self) -> None:
        candidates = pd.DataFrame(
            {
                "code": ["1001", "1002"],
                "model_score": [0.8, 0.7],
                "model_rank": [1, 2],
                "prior_close": [100.0, 100.0],
                "atr14_pct": [2.0, 2.0],
            }
        )
        snapshots = pd.DataFrame(
            {
                "observed_at": [
                    "2026-07-21T08:58:00+09:00",
                    "2026-07-21T08:58:00+09:00",
                ],
                "code": ["1001", "1002"],
                "indicative_price": [104.0, 99.0],
                "buy_special": [False, True],
            }
        )
        overlaid = apply_preopen_overlay(
            candidates,
            snapshots,
            target_date="2026-07-21",
            as_of="2026-07-21T08:58:00+09:00",
        )
        self.assertEqual(overlaid["model_score"].tolist(), [0.8, 0.7])
        self.assertEqual(overlaid["model_rank"].tolist(), [1, 2])
        self.assertEqual(overlaid["order_status"].tolist(), ["DISPLAY_ONLY", "DISPLAY_ONLY"])
        self.assertIn("positive_gap_atr", overlaid.loc[0, "veto_reasons"])
        self.assertIn("buy_special_quote", overlaid.loc[1, "veto_reasons"])

    def test_snapshot_after_cutoff_is_never_used(self) -> None:
        candidates = pd.DataFrame(
            {
                "code": ["1001"],
                "model_score": [0.8],
                "model_rank": [1],
                "prior_close": [100.0],
                "atr14_pct": [2.0],
            }
        )
        snapshots = pd.DataFrame(
            {
                "observed_at": [
                    "2026-07-21T08:57:00+09:00",
                    "2026-07-21T08:59:00+09:00",
                ],
                "code": ["1001", "1001"],
                "indicative_price": [101.0, 110.0],
            }
        )
        result = apply_preopen_overlay(
            candidates,
            snapshots,
            target_date="2026-07-21",
            as_of="2026-07-21T08:58:00+09:00",
        )
        self.assertEqual(result.loc[0, "indicative_price"], 101.0)
        self.assertEqual(result.loc[0, "order_status"], "ORDER_ELIGIBLE")

    def test_expected_open_after_0905_is_display_only(self) -> None:
        candidates = pd.DataFrame(
            {
                "code": ["1001"],
                "model_score": [0.8],
                "model_rank": [1],
                "prior_close": [100.0],
                "atr14_pct": [2.0],
            }
        )
        snapshots = pd.DataFrame(
            {
                "observed_at": ["2026-07-21T08:58:00+09:00"],
                "code": ["1001"],
                "indicative_price": [101.0],
                "expected_open_at": ["09:06:00"],
            }
        )
        result = apply_preopen_overlay(
            candidates,
            snapshots,
            target_date="2026-07-21",
            as_of="2026-07-21T08:58:00+09:00",
        )
        self.assertEqual(result.loc[0, "order_status"], "DISPLAY_ONLY")
        self.assertIn("expected_open_after_0905", result.loc[0, "veto_reasons"])

    def test_snapshot_after_decision_time_is_rejected(self) -> None:
        candidates = pd.DataFrame(
            {
                "code": ["1001"],
                "model_score": [0.8],
                "model_rank": [1],
                "selection_role": ["CORE"],
                "prior_close": [100.0],
                "atr14_pct": [2.0],
            }
        )
        snapshots = pd.DataFrame(
            {
                "observed_at": ["2026-07-21T09:20:00+09:00"],
                "code": ["1001"],
                "indicative_price": [101.0],
            }
        )
        with self.assertRaisesRegex(LeakageError, "decision cutoff"):
            apply_preopen_overlay(
                candidates,
                snapshots,
                target_date="2026-07-21",
                as_of="2026-07-21T09:21:00+09:00",
            )

    def test_reserve_rank_is_never_order_eligible(self) -> None:
        candidates = pd.DataFrame(
            {
                "code": ["1002"],
                "model_score": [0.7],
                "model_rank": [2],
                "selection_role": ["RESERVE"],
                "prior_close": [100.0],
                "atr14_pct": [2.0],
            }
        )
        snapshots = pd.DataFrame(
            {
                "observed_at": ["2026-07-21T08:58:00+09:00"],
                "code": ["1002"],
                "indicative_price": [101.0],
            }
        )
        result = apply_preopen_overlay(
            candidates,
            snapshots,
            target_date="2026-07-21",
            as_of="2026-07-21T08:58:00+09:00",
        )
        self.assertEqual(result.loc[0, "order_status"], "DISPLAY_ONLY")
        self.assertIn("reserve_rank", result.loc[0, "veto_reasons"])

    def test_preopen_code_normalisation_and_positive_price(self) -> None:
        frame = normalize_preopen_snapshots(
            pd.DataFrame(
                {
                    "observed_at": ["2026-07-21T08:58:00+09:00"],
                    "code": [1001.0],
                    "indicative_price": [101.0],
                }
            )
        )
        self.assertEqual(frame.loc[0, "code"], "1001")
        with self.assertRaisesRegex(DataValidationError, "positive"):
            normalize_preopen_snapshots(
                pd.DataFrame(
                    {
                        "observed_at": ["2026-07-21T08:58:00+09:00"],
                        "code": ["1001"],
                        "indicative_price": [0],
                    }
                )
            )


if __name__ == "__main__":
    unittest.main()
