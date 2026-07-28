from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research import model_v16_liquidity_runner as runner


ROOT = Path(__file__).resolve().parents[1]
DAILY_FORMAT = (
    "jpx_stock_quotations_auction_regular_way_domestic_ordinary"
)


def _daily_prices(
    *,
    periods: int = 28,
    codes: tuple[str, ...] = ("1001", "1002", "1003"),
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    sessions = pd.bdate_range("2025-08-01", periods=periods)
    rows: list[dict[str, object]] = []
    for code_index, code in enumerate(codes):
        prior_close = 100.0 + 10.0 * code_index
        for date_index, date in enumerate(sessions):
            open_price = prior_close * (
                1.0 + 0.001 * np.sin((date_index + code_index) / 3.0)
            )
            close = open_price * (
                1.0 + 0.002 * np.cos((date_index + code_index) / 4.0)
            )
            volume = 100_000.0 + 1_000.0 * date_index + 10_000.0 * code_index
            turnover = volume * (open_price + close) / 2.0
            vwap = (open_price + close) / 2.0
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "name": code,
                    "open": open_price,
                    "high": max(open_price, close) * 1.002,
                    "low": min(open_price, close) * 0.998,
                    "close": close,
                    "volume": volume,
                    "turnover": turnover,
                    "vwap": vwap,
                    "trading_unit": 100.0,
                    "traded": True,
                    "partial_session": False,
                    "source_format": DAILY_FORMAT,
                }
            )
            prior_close = close
    return pd.DataFrame(rows), pd.DatetimeIndex(sessions)


def _long_daily_prices() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    sessions = pd.bdate_range("2025-05-01", "2025-12-05")
    rows: list[dict[str, object]] = []
    for code_index in range(12):
        code = str(1001 + code_index)
        prior_close = 100.0 + 2.0 * code_index
        for date_index, date in enumerate(sessions):
            overnight = 0.002 * np.sin((date_index + code_index) / 7.0)
            open_price = prior_close * (1.0 + overnight)
            intraday = 0.004 * np.cos(
                (2 * date_index + 3 * code_index) / 11.0
            )
            close = open_price * (1.0 + intraday)
            volume = (
                2_000_000.0
                + 20_000.0 * code_index
                + 5_000.0 * (date_index % 10)
            )
            vwap = (2.0 * open_price + close) / 3.0
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "name": code,
                    "open": open_price,
                    "high": max(open_price, close) * 1.003,
                    "low": min(open_price, close) * 0.997,
                    "close": close,
                    "volume": volume,
                    "turnover": volume * vwap,
                    "vwap": vwap,
                    "trading_unit": 100.0,
                    "traded": True,
                    "partial_session": False,
                    "source_format": DAILY_FORMAT,
                }
            )
            prior_close = close
    return pd.DataFrame(rows), pd.DatetimeIndex(sessions)


def test_protocol_hash_candidates_and_authority_are_exact() -> None:
    protocol_path = ROOT / "research/model_v16_liquidity_protocol.json"
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == (
        runner.PROTOCOL_SHA256
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == runner.PROTOCOL_ID
    assert tuple(item["id"] for item in protocol["candidates"]) == (
        runner.CANDIDATES
    )
    assert protocol["authority"]["project_level_untouched"] is False
    assert protocol["authority"]["production_promotion_allowed"] is False
    assert protocol["authority"]["orders_allowed"] is False
    assert protocol["evaluation"]["candidate_variants"] == 6
    assert protocol["evaluation"]["costs_bps"] == [20, 40, 60]
    runner._validate_protocol()


def test_exact_liquidity_is_strictly_lagged_and_same_day_invariant() -> None:
    prices, sessions = _daily_prices()
    target = sessions[22]
    original = runner.build_exact_liquidity_features(prices, sessions)
    original_target = original[original["date"].eq(target)].reset_index(drop=True)
    assert original_target["liq_history_complete20"].all()
    assert (
        original_target["feature_source_max_date"] == sessions[21]
    ).all()

    mutated = prices.copy()
    target_rows = mutated["date"].eq(target)
    mutated.loc[
        target_rows,
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
            "vwap",
            "trading_unit",
        ],
    ] *= np.asarray([1.4, 1.8, 0.6, 1.5, 8.0, 9.0, 1.7, 2.0])
    changed = runner.build_exact_liquidity_features(mutated, sessions)
    changed_target = changed[changed["date"].eq(target)].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        original_target,
        changed_target,
        check_exact=True,
    )


def test_future_mutation_cannot_change_earlier_liquidity_features() -> None:
    prices, sessions = _daily_prices()
    cutoff = sessions[22]
    original = runner.build_exact_liquidity_features(prices, sessions)
    mutated = prices.copy()
    future = mutated["date"].eq(sessions[25])
    mutated.loc[future, ["close", "volume", "turnover", "vwap"]] *= [
        2.0,
        7.0,
        9.0,
        1.8,
    ]
    changed = runner.build_exact_liquidity_features(mutated, sessions)
    columns = [
        "date",
        "code",
        "feature_source_max_date",
        "liq_history_complete20",
        "activity_veto_pass",
        *runner.LIQUIDITY_RAW_FEATURES,
        *runner.LIQUIDITY_FEATURES,
    ]
    pd.testing.assert_frame_equal(
        original.loc[original["date"].le(cutoff), columns].reset_index(drop=True),
        changed.loc[changed["date"].le(cutoff), columns].reset_index(drop=True),
        check_exact=True,
    )


def test_missing_session_and_trading_unit_change_fail_closed() -> None:
    prices, sessions = _daily_prices()
    target = sessions[22]
    missing = prices[
        ~(prices["date"].eq(sessions[10]) & prices["code"].eq("1001"))
    ].copy()
    missing_features = runner.build_exact_liquidity_features(missing, sessions)
    missing_row = missing_features.loc[
        missing_features["date"].eq(target)
        & missing_features["code"].eq("1001")
    ].iloc[0]
    assert not bool(missing_row["liq_history_complete20"])
    assert not bool(missing_row["activity_veto_pass"])

    changed_unit = prices.copy()
    changed_unit.loc[
        changed_unit["date"].eq(sessions[10])
        & changed_unit["code"].eq("1001"),
        "trading_unit",
    ] = 200.0
    unit_features = runner.build_exact_liquidity_features(
        changed_unit, sessions
    )
    unit_row = unit_features.loc[
        unit_features["date"].eq(target)
        & unit_features["code"].eq("1001")
    ].iloc[0]
    assert not bool(unit_row["liq_history_complete20"])


def test_duplicate_daily_source_row_fails_closed() -> None:
    prices, sessions = _daily_prices()
    duplicated = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        runner.build_exact_liquidity_features(duplicated, sessions)


def test_official_zero_activity_is_real_but_ineligible() -> None:
    prices, sessions = _daily_prices()
    target = sessions[22]
    zero = prices.copy()
    mask = zero["date"].eq(sessions[10]) & zero["code"].eq("1001")
    zero.loc[mask, ["volume", "turnover"]] = 0.0
    features = runner.build_exact_liquidity_features(zero, sessions)
    row = features.loc[
        features["date"].eq(target) & features["code"].eq("1001")
    ].iloc[0]
    assert not bool(row["liq_history_complete20"])
    assert not bool(row["activity_veto_pass"])


def test_rank_to_unit_has_exact_endpoints_and_ignores_ineligible_rows() -> None:
    values = pd.Series([10.0, 20.0, 30.0, 999.0])
    dates = pd.Series(pd.to_datetime(["2026-01-05"] * 4))
    universe = pd.Series([True, True, True, False])
    ranked = runner._rank_to_unit(values, dates, universe)
    np.testing.assert_allclose(ranked.iloc[:3], [-1.0, 0.0, 1.0])
    assert np.isnan(ranked.iloc[3])


def test_vetoed_top_slot_is_cash_without_rank_three_replacement() -> None:
    scoring = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05"] * 4),
            "code": ["1001", "1002", "1003", "1004"],
            "name": ["a", "b", "c", "d"],
            "activity_veto_pass": [False, True, True, True],
        }
    )
    picks = runner._top_two(
        scoring,
        np.asarray([4.0, 3.0, 2.0, 1.0]),
        "TEST",
        apply_veto=True,
    )
    first = picks[picks["model_rank"].eq(1)].iloc[0]
    second = picks[picks["model_rank"].eq(2)].iloc[0]
    assert first["pre_veto_code"] == "1001"
    assert bool(first["vetoed"])
    assert pd.isna(first["code"])
    assert second["code"] == "1002"
    assert "1003" not in set(picks["code"].dropna())


def test_score_ledger_hash_is_row_order_invariant_and_has_no_outcome() -> None:
    dates = pd.to_datetime(["2026-01-05", "2026-01-05"])
    ledger = pd.DataFrame(
        {
            "candidate_id": ["A", "A"],
            "date": dates,
            "model_rank": [1, 2],
            "code": ["1001", "1002"],
            "name": ["a", "b"],
            "model_score": [2.0, 1.0],
            "pre_veto_code": ["1001", "1002"],
            "vetoed": [False, False],
        }
    )
    assert not {"label", "oc_return_pct"} & set(ledger.columns)
    assert runner.semantic_score_hash(ledger) == runner.semantic_score_hash(
        ledger.sample(frac=1.0, random_state=7)
    )


def test_outcome_join_happens_after_score_and_duplicate_fails() -> None:
    ledger = pd.DataFrame(
        {
            "candidate_id": ["A"],
            "date": pd.to_datetime(["2026-01-05"]),
            "model_rank": [1],
            "code": ["1001"],
            "name": ["a"],
            "model_score": [1.0],
            "pre_veto_code": ["1001"],
            "vetoed": [False],
        }
    )
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05"]),
            "code": ["1001"],
            "label": [1.0],
            "oc_return_pct": [0.8],
        }
    )
    joined = runner.attach_outcomes(ledger, panel)
    assert joined["label"].tolist() == [1.0]
    assert joined["oc_return_pct"].tolist() == [0.8]
    with pytest.raises(ValueError, match="duplicate"):
        runner.attach_outcomes(ledger, pd.concat([panel, panel]))


def test_end_to_end_scores_ignore_target_finality_and_missing_outcomes_are_cash() -> None:
    prices, sessions = _long_daily_prices()
    target = sessions[-1]
    original_panel, _, original_checks = runner.build_model_panel(
        prices, sessions
    )
    scheduled = pd.DatetimeIndex([target])
    original_ledger, folds = runner.build_score_ledger(
        original_panel, scheduled
    )
    assert original_checks["same_day_finality_used_for_scoring"] is False
    assert all(
        pd.Timestamp(item["train_end"])
        < pd.Period(item["period"], freq="M").start_time
        for item in folds
    )

    # Remove every target outcome from the prior-session universe and leave one
    # new right-only code solely to retain the target date in the calendar.
    missing_target = prices[~prices["date"].eq(target)].copy()
    placeholder = prices[prices["date"].eq(target)].iloc[[0]].copy()
    placeholder["code"] = "9999"
    placeholder["name"] = "new-right-only"
    placeholder[
        ["open", "high", "low", "close", "volume", "turnover", "vwap"]
    ] *= 7.0
    missing_target = pd.concat(
        [missing_target, placeholder], ignore_index=True
    )
    changed_panel, changed_coverage, changed_checks = runner.build_model_panel(
        missing_target, sessions
    )
    assert not bool(
        changed_coverage.loc[
            changed_coverage["date"].eq(target), "source_complete"
        ].iloc[0]
    )
    assert changed_checks["same_day_finality_used_for_scoring"] is False
    changed_ledger, _ = runner.build_score_ledger(
        changed_panel, scheduled
    )
    score_columns = [
        "candidate_id",
        "date",
        "model_rank",
        "code",
        "model_score",
        "pre_veto_code",
        "vetoed",
    ]
    pd.testing.assert_frame_equal(
        original_ledger[score_columns].reset_index(drop=True),
        changed_ledger[score_columns].reset_index(drop=True),
        check_exact=True,
    )

    # Input row order cannot alter a semantic score ledger.
    shuffled_panel, _, _ = runner.build_model_panel(
        prices.sample(frac=1.0, random_state=19), sessions
    )
    shuffled_ledger, _ = runner.build_score_ledger(
        shuffled_panel, scheduled
    )
    assert runner.semantic_score_hash(original_ledger) == (
        runner.semantic_score_hash(shuffled_ledger)
    )

    earlier = pd.DatetimeIndex([sessions[-3]])
    earlier_ledger, _ = runner.build_score_ledger(
        original_panel, earlier
    )
    future_mutated = prices.copy()
    future_rows = future_mutated["date"].eq(sessions[-1])
    future_mutated.loc[
        future_rows,
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
            "vwap",
            "trading_unit",
        ],
    ] *= np.asarray([1.4, 1.8, 0.6, 1.5, 8.0, 9.0, 1.7, 2.0])
    future_panel, _, _ = runner.build_model_panel(
        future_mutated, sessions
    )
    future_ledger, _ = runner.build_score_ledger(
        future_panel, earlier
    )
    assert runner.semantic_score_hash(earlier_ledger) == (
        runner.semantic_score_hash(future_ledger)
    )

    changed_picks = runner.attach_outcomes(changed_ledger, changed_panel)
    assert changed_picks["label"].isna().all()
    for candidate_id in runner.ALL_MODELS:
        candidate = changed_picks[
            changed_picks["candidate_id"].eq(candidate_id)
        ]
        daily = runner._daily(
            candidate,
            scheduled,
            capacity=2,
            cost_bps=40.0,
        )
        assert daily.tolist() == [0.0]


def test_no_gate_passer_means_no_winner() -> None:
    variants = [
        {
            "variant_id": "A__top1",
            "candidate_id": "A",
            "capacity": 1,
            "gate_passed": False,
            "paired_vs_control_net40": {
                "one_sided_lower_delta_pct": 1.0
            },
            "top4_days_removed_net40_mean_pct": 1.0,
        }
    ]
    assert runner._choose_winner(variants) is None
