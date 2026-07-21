from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from research import finalize_logit_v04 as base
from research import recover_logit_v04_holdout as formal
from research.build_model_v04_diagnostic_panel import (
    EVALUATION_COLUMNS,
    PRICE_INPUT_COLUMNS,
    build_panel_memory_bounded,
)


class DiagnosticPanelEquivalenceTests(unittest.TestCase):
    def test_memory_pipeline_is_exact_for_frozen_metric_columns(self) -> None:
        sessions = pd.bdate_range("2024-01-04", periods=100)
        rows: list[dict[str, object]] = []
        codes = ("1301", "1332", "1605", "7203")
        for day_index, date in enumerate(sessions):
            for code_index, code in enumerate(codes):
                if day_index == 45 and code == "1605":
                    # Missing outcome retained through the prior universe.
                    continue
                level = 900.0 + 3.0 * code_index + day_index
                if code == "1301" and day_index >= 70:
                    level *= 1.5  # Known discontinuity hygiene path.
                open_price = level + (day_index % 3 - 1)
                close_price = level + ((day_index + code_index) % 3 - 1)
                high = max(open_price, close_price) + 3.0
                low = min(open_price, close_price) - 3.0
                row: dict[str, object] = {
                    "date": date,
                    "code": code,
                    "name": f"銘柄{code}",
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close_price,
                    "am_open": open_price,
                    "am_high": max(open_price, level) + 2.0,
                    "am_low": min(open_price, level) - 2.0,
                    "am_close": level,
                    "pm_open": level,
                    "pm_high": high,
                    "pm_low": low,
                    "pm_close": close_price,
                    "traded": True,
                    "partial_session": False,
                    "volume": 100_000.0 + day_index,
                    "turnover": (100_000.0 + day_index) * level,
                    "source_file": "unused-provenance.txt",
                }
                if day_index == 35 and code == "1332":
                    for column in base.SESSION_OHLC:
                        row[column] = np.nan
                    row["partial_session"] = True
                if day_index == 55 and code == "7203":
                    for column in ("open", "high", "low", "close", *base.SESSION_OHLC):
                        row[column] = np.nan
                    row["traded"] = False
                    row["volume"] = 0.0
                    row["turnover"] = 0.0
                rows.append(row)
        prices = pd.DataFrame(rows)
        disclosure_time = pd.Timestamp(sessions[60]).tz_localize(
            "Asia/Tokyo"
        ) + pd.Timedelta(hours=16)
        disclosures = pd.DataFrame(
            {
                "published_at": [disclosure_time],
                "code": ["1301"],
                "name": ["銘柄1301"],
                "title": ["業績予想の上方修正に関するお知らせ"],
                "url": ["https://example.test/disclosure"],
            }
        )
        complete_dates = pd.date_range(sessions.min(), sessions.max(), freq="D")
        complete_dates = complete_dates.delete(50)

        prior_cow = pd.options.mode.copy_on_write
        try:
            pd.options.mode.copy_on_write = False
            expected, expected_coverage = base.build_research_panel(
                prices.copy(), disclosures.copy(), sessions, complete_dates
            )
            formal_lock = json.loads(
                formal.ROOT.joinpath(
                    "research/model_v04_parser_recovery_lock.json"
                ).read_text(encoding="utf-8")
            )
            winner = base.ModelSpec.from_dict(formal_lock["winner"])
            control = formal.baseline_spec()
            frozen_features = tuple(
                dict.fromkeys(
                    [
                        *base.feature_blocks()[winner.feature_block],
                        *base.feature_blocks()[control.feature_block],
                    ]
                )
            )
            keep = list(dict.fromkeys([*EVALUATION_COLUMNS, *frozen_features]))
            expected = expected.loc[
                expected["date"].ge(base.TRAIN_START), keep
            ].copy()

            pd.options.mode.copy_on_write = True
            actual, actual_coverage = build_panel_memory_bounded(
                prices.loc[:, list(PRICE_INPUT_COLUMNS)].copy(),
                sessions,
                disclosures.copy(),
                complete_dates,
            )
        finally:
            pd.options.mode.copy_on_write = prior_cow

        pd.testing.assert_frame_equal(
            expected.reset_index(drop=True),
            actual.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            expected_coverage.reset_index(drop=True),
            actual_coverage.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )


if __name__ == "__main__":
    unittest.main()
