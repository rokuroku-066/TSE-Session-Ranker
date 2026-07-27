from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from research import model_v14_feature_contrast_runner as runner


ROOT = Path(__file__).resolve().parents[1]


def _history_panel(
    *,
    periods: int = 45,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    sessions = pd.bdate_range("2024-05-01", periods=periods)
    rows: list[dict[str, object]] = []
    for code_index, code in enumerate(("1001", "1002", "1003")):
        prior_close = 100.0 + 10.0 * code_index
        for date_index, date in enumerate(sessions):
            overnight = 0.2 * np.sin((date_index + code_index) / 3.0)
            open_price = prior_close * (1.0 + overnight / 100.0)
            am_return = 0.3 * np.sin((date_index + code_index) / 4.0)
            am_close = open_price * (1.0 + am_return / 100.0)
            lunch = 0.05 * np.cos((date_index + code_index) / 2.0)
            pm_open = am_close * (1.0 + lunch / 100.0)
            pm_return = 0.25 * np.cos((date_index + 2 * code_index) / 5.0)
            close = pm_open * (1.0 + pm_return / 100.0)
            am_high = max(open_price, am_close) * 1.002
            am_low = min(open_price, am_close) * 0.998
            pm_high = max(pm_open, close) * 1.0025
            pm_low = min(pm_open, close) * 0.9975
            row: dict[str, object] = {
                "date": date,
                "code": code,
                "name": code,
                "open": open_price,
                "high": max(am_high, pm_high),
                "low": min(am_low, pm_low),
                "close": close,
                "prior_close": prior_close,
                "overnight": overnight,
                "oc_return_pct": 100.0 * (close / open_price - 1.0),
                "am_open": open_price,
                "am_high": am_high,
                "am_low": am_low,
                "am_close": am_close,
                "pm_open": pm_open,
                "pm_high": pm_high,
                "pm_low": pm_low,
                "pm_close": close,
                "traded": True,
                "price_eligible": True,
            }
            for feature_index, feature in enumerate(runner.EXISTING_FEATURES):
                row[feature] = (
                    0.01 * feature_index
                    + 0.001 * date_index
                    + 0.0001 * code_index
                )
            rows.append(row)
            prior_close = close
    return pd.DataFrame(rows), pd.DatetimeIndex(sessions)


def _discovery_dates() -> pd.DatetimeIndex:
    dates = pd.bdate_range(runner.DISCOVERY_START, runner.DISCOVERY_END)
    # Five TSE holidays in this interval are removed to match the frozen count.
    remove = pd.to_datetime(
        ["2024-07-15", "2024-08-12", "2024-09-16", "2024-09-23", "2024-10-14"]
    )
    return dates[~dates.isin(remove)]


def test_protocol_hash_and_catalog_are_exact() -> None:
    path = ROOT / "research/model_v14_feature_contrast_protocol.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == runner.PROTOCOL_SHA256
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == runner.PROTOCOL_ID
    assert protocol["authority"]["confirmation_scoring_allowed_by_this_protocol"] is False
    assert protocol["authority"]["production_promotion_allowed"] is False
    assert protocol["authority"]["orders_allowed"] is False
    assert len(runner.EXISTING_FEATURES) == 15
    assert len(runner.SESSION_FEATURES) == 10
    assert len(runner.SPECTRAL_FEATURES) == 20
    assert len(runner.COMPLEXITY_FEATURES) == 8
    assert len(runner.ALL_FEATURES) == 53
    assert len(set(runner.ALL_FEATURES)) == 53


def test_target_and_future_ohlc_cannot_change_lagged_features() -> None:
    panel, sessions = _history_panel()
    original = runner.add_v14_features(panel, sessions)
    target = sessions[36]
    mutated = panel.copy()
    target_row = mutated["date"].eq(target) & mutated["code"].eq("1001")
    mutated.loc[
        target_row,
        [
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
        ],
    ] *= np.asarray(
        [1.4, 1.8, 0.7, 1.6, 1.4, 1.7, 0.8, 1.5, 1.3, 1.9, 0.6, 1.6]
    )
    mutated.loc[target_row, ["overnight", "oc_return_pct"]] = [30.0, 25.0]
    future_row = mutated["date"].eq(sessions[42]) & mutated["code"].eq("1002")
    mutated.loc[
        future_row,
        [
            "open",
            "high",
            "low",
            "close",
            "am_open",
            "am_close",
            "pm_open",
            "pm_close",
        ],
    ] *= np.asarray([1.3, 1.7, 0.7, 1.5, 1.3, 1.5, 1.4, 1.5])
    mutated.loc[future_row, ["overnight", "oc_return_pct"]] = [20.0, 15.0]
    changed = runner.add_v14_features(mutated, sessions)

    feature_columns = list(runner.ALL_FEATURES)
    pd.testing.assert_frame_equal(
        original.loc[original["date"].le(target), feature_columns].reset_index(
            drop=True
        ),
        changed.loc[changed["date"].le(target), feature_columns].reset_index(
            drop=True
        ),
        check_exact=True,
    )


def test_nonconsecutive_history_fails_closed() -> None:
    panel, sessions = _history_panel()
    target = sessions[36]
    missing = panel[
        ~(panel["date"].eq(sessions[35]) & panel["code"].eq("1001"))
    ].copy()
    featured = runner.add_v14_features(missing, sessions)
    row = featured.loc[
        featured["date"].eq(target) & featured["code"].eq("1001")
    ].iloc[0]
    assert row[list(runner.SESSION_FEATURES[:5])].isna().all()
    assert row[list(runner.SPECTRAL_FEATURES)].isna().all()
    assert row[list(runner.COMPLEXITY_FEATURES)].isna().all()


def test_zero_paths_have_zero_spectrum_and_locked_complexity() -> None:
    spectral = runner._spectral_values(np.zeros((2, 5, 32), dtype=float))
    assert all(np.array_equal(values, np.zeros((2, 5))) for values in spectral)

    channel_windows = np.zeros((2, 5, 20), dtype=float)
    range_windows = np.zeros((2, 20), dtype=float)
    complexity = runner._complexity_values(channel_windows, range_windows)
    assert len(complexity) == 8
    assert np.array_equal(complexity[0], np.zeros(2))
    assert np.array_equal(complexity[1], np.zeros(2))
    assert np.array_equal(complexity[2], np.zeros(2))
    assert np.array_equal(complexity[3], np.zeros(2))
    assert np.array_equal(complexity[4], np.ones(2))
    assert np.array_equal(complexity[5], np.zeros(2))
    assert np.array_equal(complexity[6], np.zeros(2))
    assert np.array_equal(complexity[7], np.zeros(2))


def test_discovery_contrast_reports_bh_slices_and_stable_signals() -> None:
    rows: list[dict[str, object]] = []
    for date in _discovery_dates():
        for index, (label, value) in enumerate(
            ((False, -2.0), (False, -1.0), (True, 1.0), (True, 2.0))
        ):
            row: dict[str, object] = {
                "date": date,
                "open": 100.0,
                "close": 101.0 if label else 99.0,
                "price_eligible": True,
                "code": str(1000 + index),
            }
            row.update({feature: value for feature in runner.ALL_FEATURES})
            rows.append(row)
    result = runner.analyze_feature_contrasts(pd.DataFrame(rows))
    assert result["scheduled_sessions"] == runner.SCHEDULED_DISCOVERY_SESSIONS
    assert result["feature_count"] == 53
    assert len(result["stable_signals"]) == 53
    for feature in result["features"]:
        assert feature["daily_rank_gap"]["mean"] > 0.0
        assert feature["daily_rank_gap"]["normal_p_two_sided"] == 0.0
        assert feature["bh_q"] == 0.0
        assert feature["stable_signal"] is True
        assert set(feature["daily_rank_gap"]["slices"]) == set(runner.SLICES)


def test_confirmation_rows_are_rejected() -> None:
    row: dict[str, object] = {
        "date": pd.Timestamp("2024-11-01"),
        "open": 100.0,
        "close": 101.0,
        "price_eligible": True,
    }
    row.update({feature: 0.0 for feature in runner.ALL_FEATURES})
    with pytest.raises(ValueError, match="forbidden confirmation"):
        runner.analyze_feature_contrasts(pd.DataFrame([row]))


def test_joblib_panel_cache_formats(tmp_path: Path) -> None:
    panel, sessions = _history_panel(periods=10)
    path = tmp_path / "panel.joblib"
    joblib.dump(
        {
            "panel": panel,
            "sessions": sessions,
            "input_checks": {"fixture": True},
        },
        path,
    )
    loaded, loaded_sessions, checks = runner._load_panel_cache(path)
    pd.testing.assert_frame_equal(loaded, panel)
    pd.testing.assert_index_equal(loaded_sessions, sessions)
    assert checks["fixture"] is True
    assert checks["panel_rows"] == len(panel)
    assert checks["panel_cache_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_bh_adjustment_is_monotone_in_sorted_p_values() -> None:
    p_values = [0.04, 0.001, 0.02, 0.5]
    q_values = runner._bh_q_values(p_values)
    ordered = sorted(zip(p_values, q_values))
    assert [item[1] for item in ordered] == sorted(item[1] for item in ordered)
    assert all(p <= q <= 1.0 for p, q in zip(p_values, q_values, strict=True))
