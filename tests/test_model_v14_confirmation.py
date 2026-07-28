from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research import model_v14_confirmation_runner as runner
from tse_session_ranker.validation import PairedBootstrapInterval


ROOT = Path(__file__).resolve().parents[1]


def _confirmation_dates() -> pd.DatetimeIndex:
    dates = pd.bdate_range(
        runner.CONFIRMATION_START, runner.CONFIRMATION_END
    )
    holidays = pd.to_datetime(
        [
            "2024-11-04",
            "2024-12-31",
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-13",
            "2025-02-11",
            "2025-02-24",
            "2025-03-20",
            "2025-04-29",
            "2025-05-05",
            "2025-05-06",
            "2025-07-21",
        ]
    )
    output = dates[~dates.isin(holidays)]
    assert len(output) == runner.SCHEDULED_CONFIRMATION_SESSIONS
    return output


def _dmd_panel() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    sessions = pd.bdate_range("2024-09-02", periods=14)
    rows: list[dict[str, object]] = []
    codes = ("1001", "1002", "1003", "1004")
    for date_index, date in enumerate(sessions):
        for code_index, code in enumerate(codes):
            base = 100.0 + code_index
            oc = (-1.5, -0.5, 0.5, 1.5)[
                (date_index + code_index) % len(codes)
            ]
            am = (-0.8, -0.2, 0.4, 1.0)[
                (2 * date_index + code_index) % len(codes)
            ]
            lunch = (-0.3, -0.1, 0.1, 0.3)[
                (date_index + 2 * code_index) % len(codes)
            ]
            pm = oc - am - lunch
            open_price = base
            am_close = open_price * (1.0 + am / 100.0)
            pm_open = am_close * (1.0 + lunch / 100.0)
            close = open_price * (1.0 + oc / 100.0)
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "name": code,
                    "open": open_price,
                    "close": close,
                    "am_open": open_price,
                    "am_close": am_close,
                    "pm_open": pm_open,
                    "pm_close": close
                    if np.isfinite(pm)
                    else pm_open,
                    "traded": True,
                    "price_eligible": True,
                    "label": float(close > open_price),
                    "oc_return_pct": oc,
                }
            )
    return pd.DataFrame(rows), pd.DatetimeIndex(sessions)


def _monthly_classifier_panel() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    scheduled = _confirmation_dates()
    training_dates = pd.bdate_range("2024-10-01", "2024-10-31")
    dates = training_dates.append(scheduled)
    rows: list[dict[str, object]] = []
    for date_index, date in enumerate(dates):
        for code_index, code in enumerate(("1001", "1002", "1003", "1004")):
            label = float((date_index + code_index) % 2 == 0)
            row: dict[str, object] = {
                "date": date,
                "code": code,
                "name": code,
                "label": label,
                "oc_return_pct": 0.8 if label else -0.6,
                "price_eligible": True,
                "price_training_eligible": True,
            }
            for feature_index, feature in enumerate(runner.SP_FEATURES):
                row[f"_v14_rank_{feature}"] = (
                    -1.0
                    + 2.0
                    * ((date_index + code_index + feature_index) % 7)
                    / 6.0
                )
            for feature_index, (feature, _) in enumerate(
                runner.GN_FEATURE_DIRECTIONS
            ):
                row[f"_v14_rank_{feature}"] = (
                    -1.0
                    + 2.0
                    * ((2 * date_index + code_index + feature_index) % 9)
                    / 8.0
                )
            rows.append(row)
    return pd.DataFrame(rows), scheduled


def _gate_picks(
    scheduled: pd.DatetimeIndex,
    candidate_id: str,
    return_pct: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, date in enumerate(scheduled):
        for model_rank in (1, 2):
            code_number = 1000 + (
                (date_index + 70 * (model_rank - 1)) % 140
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "date": date,
                    "model_rank": model_rank,
                    "code": str(code_number),
                    "name": str(code_number),
                    "model_score": float(3 - model_rank),
                    "label": float(return_pct > 0.0),
                    "oc_return_pct": return_pct,
                }
            )
    return pd.DataFrame(rows)


def test_protocol_hash_erratum_and_exact_discovery_selection() -> None:
    protocol_path = ROOT / "research/model_v14_confirmation_protocol.json"
    observed = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert observed == runner.PROTOCOL_SHA256
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    assert protocol["protocol_id"] == runner.PROTOCOL_ID
    assert protocol["stage_a_binding"]["protocol"]["sha256"] == (
        runner.STAGE_A_PROTOCOL_SHA256
    )
    assert protocol["stage_a_binding"]["result"]["sha256"] == (
        runner.STAGE_A_RESULT_SHA256
    )
    assert protocol["frozen_input"]["v14_input_erratum"]["sha256"] == (
        runner.INPUT_ERRATUM_SHA256
    )
    assert hashlib.sha256(runner.INPUT_ERRATUM.read_bytes()).hexdigest() == (
        runner.INPUT_ERRATUM_SHA256
    )
    selected = tuple(
        (item["feature"], item["direction"])
        for item in protocol["stage_a_binding"][
            "all_stable_features_in_discovery_order"
        ]
    )
    assert selected == runner.STABLE_FEATURES
    candidate_features = protocol["stage_a_binding"][
        "candidate_feature_selections"
    ]
    assert tuple(item["feature"] for item in candidate_features["SP01"]) == (
        runner.SP_FEATURES
    )
    assert tuple(
        (item["feature"], item["direction"])
        for item in candidate_features["GN01"]
    ) == runner.GN_FEATURE_DIRECTIONS
    assert tuple(
        item["id"] for item in protocol["registered_candidates"]
    ) == runner.CANDIDATES
    assert protocol["registered_candidates"][0]["fit_schedule"] == (
        "month-fixed"
    )
    assert protocol["registered_candidates"][0][
        "state_lookback_sessions"
    ] == 60
    assert protocol["registered_candidates"][0]["fixed_rank"] == 8
    assert protocol["registered_candidates"][1][
        "state_lookback_sessions"
    ] == 32
    assert protocol["registered_candidates"][1]["fixed_rank"] == 12
    assert protocol["authority"]["production_promotion_allowed"] is False
    assert protocol["authority"]["orders_allowed"] is False
    runner._validate_protocol()


def test_equal_open_close_is_nonpositive_class() -> None:
    labels = runner.make_binary_label(
        pd.Series([100.0, 100.0, 100.0, np.nan]),
        pd.Series([101.0, 100.0, 99.0, 101.0]),
    )
    assert labels.iloc[:3].tolist() == [1.0, 0.0, 0.0]
    assert np.isnan(labels.iloc[3])


def test_within_date_rank_maps_exactly_to_unit_interval() -> None:
    dates = pd.Series(pd.to_datetime(["2025-01-06"] * 4 + ["2025-01-07"]))
    values = pd.Series([30.0, 10.0, 20.0, 20.0, 7.0])
    ranked = runner._rank_to_unit(values, dates)
    np.testing.assert_allclose(
        ranked.iloc[:4].to_numpy(),
        np.asarray([1.0, -1.0, 0.0, 0.0]),
        rtol=0.0,
        atol=1e-12,
    )
    assert ranked.iloc[4] == 0.0


def test_exact_dmd_math_and_rank_failure() -> None:
    states = np.asarray([[1.0, 2.0, 4.0, 8.0]])
    predicted = runner._exact_dmd_predict(states, rank=1)
    np.testing.assert_allclose(predicted, np.asarray([16.0]), atol=1e-12)
    with pytest.raises(runner.DMDStateError, match="numerical rank"):
        runner._fit_exact_dmd(np.ones((4, 5)), rank=2)


def test_dmd_mapping_month_freeze_and_target_future_mutation() -> None:
    panel, sessions = _dmd_panel()
    target = sessions[9]
    spec = runner.DMDSpec(
        candidate_id="DMD_TEST",
        channels=("oc",),
        lookback=6,
        rank=1,
    )
    states = runner.make_completed_rank_states(panel, sessions)
    operator, codes, history = runner._fit_dmd_month(
        target, sessions, states, spec
    )
    assert history.max() < target
    scoring = panel.loc[panel["date"].eq(target)].copy()
    original = runner._score_dmd_date(
        scoring, target, sessions, states, spec, operator, codes
    )

    mutated = panel.copy()
    target_rows = mutated["date"].eq(target)
    mutated.loc[target_rows, ["open", "close", "am_open", "am_close"]] *= [
        1.3,
        1.7,
        1.2,
        1.6,
    ]
    future_rows = mutated["date"].eq(sessions[12])
    mutated.loc[future_rows, ["open", "close", "pm_open", "pm_close"]] *= [
        1.4,
        1.8,
        1.3,
        1.9,
    ]
    changed_states = runner.make_completed_rank_states(mutated, sessions)
    changed_operator, changed_codes, changed_history = runner._fit_dmd_month(
        target, sessions, changed_states, spec
    )
    changed_scoring = mutated.loc[mutated["date"].eq(target)].copy()
    changed = runner._score_dmd_date(
        changed_scoring,
        target,
        sessions,
        changed_states,
        spec,
        changed_operator,
        changed_codes,
    )

    assert codes == changed_codes
    pd.testing.assert_index_equal(history, changed_history)
    for channel in states:
        pd.testing.assert_frame_equal(
            states[channel].loc[: sessions[8]],
            changed_states[channel].loc[: sessions[8]],
        )
    pd.testing.assert_frame_equal(
        original[["code", "model_score"]].reset_index(drop=True),
        changed[["code", "model_score"]].reset_index(drop=True),
        check_exact=True,
    )
    latest = states["oc"].loc[sessions[8], list(codes)].to_numpy(dtype=float)
    expected = pd.Series(operator.predict(latest), index=list(codes))
    for row in original.itertuples():
        assert row.model_score == pytest.approx(
            expected.loc[str(row.code)], abs=1e-12
        )


def test_monthly_classifier_does_not_use_same_month_outcomes() -> None:
    panel, scheduled = _monthly_classifier_panel()
    original, folds = runner._monthly_classifier_scores(
        panel, scheduled, candidate_id="GN01"
    )
    mutated = panel.copy()
    november = mutated["date"].between("2024-11-01", "2024-11-30")
    mutated.loc[november, "label"] = 1.0 - mutated.loc[november, "label"]
    mutated.loc[november, "oc_return_pct"] *= -20.0
    changed, _ = runner._monthly_classifier_scores(
        mutated, scheduled, candidate_id="GN01"
    )

    november_original = original.loc[
        original["date"].between("2024-11-01", "2024-11-30"),
        ["date", "model_rank", "code", "model_score"],
    ].reset_index(drop=True)
    november_changed = changed.loc[
        changed["date"].between("2024-11-01", "2024-11-30"),
        ["date", "model_rank", "code", "model_score"],
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        november_original, november_changed, check_exact=True
    )
    assert len(folds) == runner.CONFIRMATION_MONTHS
    assert all(item["strictly_prior_training"] for item in folds)
    assert all(item["date_equal_training_weight"] for item in folds)


def test_equal_scores_break_ties_by_code() -> None:
    scoring = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-06"] * 3),
            "code": ["3000", "1000", "2000"],
            "name": ["c", "a", "b"],
            "label": [1.0, 0.0, 1.0],
            "oc_return_pct": [1.0, -1.0, 0.5],
        }
    )
    ranked = runner._top_two(
        scoring, np.zeros(len(scoring)), "candidate"
    )
    assert ranked["code"].tolist() == ["1000", "2000"]
    assert ranked["model_rank"].tolist() == [1, 2]


def test_metrics_apply_every_strict_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled = _confirmation_dates()
    candidate = _gate_picks(scheduled, "DMD01", 1.0)
    control = _gate_picks(scheduled, runner.CONTROL, 0.0)

    interval = PairedBootstrapInterval(
        observations=len(scheduled),
        block_length=runner.BOOTSTRAP_BLOCK_LENGTH,
        samples=runner.BOOTSTRAP_SAMPLES,
        confidence=1.0 - (1.0 - 0.90) / runner.FAMILY_SIZE,
        random_state=runner.BOOTSTRAP_RANDOM_STATE,
        candidate_mean_pct=0.6,
        baseline_mean_pct=-0.4,
        point_estimate_delta_pct=1.0,
        one_sided_lower_delta_pct=0.5,
        two_sided_lower_delta_pct=0.4,
        two_sided_upper_delta_pct=1.6,
        bootstrap_standard_error_delta_pct=0.2,
    )
    monkeypatch.setattr(
        runner,
        "paired_moving_block_bootstrap",
        lambda *args, **kwargs: interval,
    )
    result = runner._variant_metrics(
        candidate,
        control,
        scheduled,
        candidate_id="DMD01",
        capacity=1,
    )

    assert result["gate_passed"] is True
    assert len(result["gate_checks"]) == 12
    assert all(result["gate_checks"].values())
    assert result["cost_metrics"]["40"]["net_mean_pct_at_cost"] == (
        pytest.approx(0.6)
    )
    assert result["cost_metrics"]["60"]["net_mean_pct_at_cost"] == (
        pytest.approx(0.4)
    )
    assert result["positive_months_net40"] == 9
    assert result["unique_selected_codes"] >= 100
    assert result["maximum_code_selection_share"] <= 0.05
    assert result["top10_code_selection_share"] <= 0.25
    assert result["traded_days"] == len(scheduled)
    assert result["executed_slot_fraction"] == 1.0
