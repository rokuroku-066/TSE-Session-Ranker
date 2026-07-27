from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import model_v13_symbolic_context_runner as runner


ROOT = Path(__file__).resolve().parents[1]


def _panel() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    sessions = pd.bdate_range("2025-01-06", periods=9)
    rows: list[dict[str, object]] = []
    for code_index, code in enumerate(("1001", "1002", "1003")):
        prior_close = 100.0 + code_index * 10.0
        for date_index, date in enumerate(sessions):
            gap = (-0.4 + 0.4 * code_index) + 0.05 * date_index
            open_price = prior_close * (1.0 + gap / 100.0)
            oc = (-0.8 + 0.7 * code_index) + 0.1 * ((date_index % 3) - 1)
            close = open_price * (1.0 + oc / 100.0)
            low = min(open_price, close) * 0.995
            high = max(open_price, close) * 1.005
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "name": code,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "prior_close": prior_close,
                    "overnight": gap,
                    "oc_return_pct": oc,
                    "label": float(oc > 0.0),
                    "traded": True,
                    "price_eligible": True,
                    "price_training_eligible": True,
                }
            )
            prior_close = close
    return pd.DataFrame(rows), pd.DatetimeIndex(sessions)


def test_protocol_is_locked_and_non_production() -> None:
    path = ROOT / "research/model_v13_symbolic_context_protocol.json"
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    assert observed == runner.PROTOCOL_SHA256
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "model_v13_symbolic_context_zero_base_20260727"
    assert protocol["authority"]["production_promotion_allowed"] is False
    assert protocol["authority"]["orders_allowed"] is False
    assert protocol["family_size"] == (
        len(runner.CANDIDATES) * len(runner.CAPACITIES)
    )
    erratum_path = (
        ROOT / "research/model_v13_symbolic_context_input_erratum_v2.json"
    )
    assert hashlib.sha256(erratum_path.read_bytes()).hexdigest() == (
        runner.INPUT_ERRATUM_SHA256
    )
    erratum = json.loads(erratum_path.read_text(encoding="utf-8"))
    assert erratum["protocol_id"] == runner.PROTOCOL_ID
    assert erratum["authority"]["input_revision_only"] is True
    assert erratum["authority"]["production_promotion_allowed"] is False
    assert erratum["authority"]["orders_allowed"] is False


def test_target_and_future_ohlc_cannot_change_existing_contexts() -> None:
    panel, sessions = _panel()
    original = runner.add_symbolic_contexts(panel, sessions)
    target = sessions[6]
    mutated = panel.copy()
    same_day = mutated["date"].eq(target) & mutated["code"].eq("1001")
    mutated.loc[same_day, ["open", "high", "low", "close"]] *= [
        1.4,
        1.8,
        0.8,
        1.6,
    ]
    mutated.loc[same_day, "overnight"] = 40.0
    mutated.loc[same_day, "oc_return_pct"] = 25.0
    future = mutated["date"].eq(sessions[8]) & mutated["code"].eq("1002")
    mutated.loc[future, ["open", "high", "low", "close"]] *= [
        1.3,
        1.7,
        0.7,
        1.5,
    ]
    mutated.loc[future, "overnight"] = 30.0
    mutated.loc[future, "oc_return_pct"] = 20.0
    changed = runner.add_symbolic_contexts(mutated, sessions)
    context_columns = [
        column for column in original if "_context_" in column
    ]
    original_target = original.loc[
        original["date"].eq(target) & original["code"].eq("1001"),
        context_columns,
    ].reset_index(drop=True)
    changed_target = changed.loc[
        changed["date"].eq(target) & changed["code"].eq("1001"),
        context_columns,
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(original_target, changed_target)
    cutoff = target
    pd.testing.assert_frame_equal(
        original.loc[original["date"].le(cutoff), context_columns].reset_index(
            drop=True
        ),
        changed.loc[changed["date"].le(cutoff), context_columns].reset_index(
            drop=True
        ),
    )


def test_nonconsecutive_history_fails_closed_for_deeper_context() -> None:
    panel, sessions = _panel()
    missing = panel[
        ~(panel["date"].eq(sessions[4]) & panel["code"].eq("1001"))
    ].copy()
    contexts = runner.add_symbolic_contexts(missing, sessions)
    row = contexts.loc[
        contexts["date"].eq(sessions[5]) & contexts["code"].eq("1001")
    ].iloc[0]
    assert row["absolute_context_1"] == -1
    assert row["absolute_context_2"] == -1
    assert row["relative_context_1"] == -1
    assert row["direction_context_1"] == -1


def test_context_tree_scores_before_same_day_update() -> None:
    panel, sessions = _panel()
    contexts = runner.add_symbolic_contexts(panel, sessions)
    family = "direction"
    tree = runner.DenseContextTree(
        family,
        base=10,
        maximum_depth=6,
        row_prior_strength=2.0,
        date_prior_strength=1.0,
    )
    training = contexts[
        contexts["date"].lt(sessions[6])
        & contexts["oc_return_pct"].notna()
    ]
    scoring = contexts[contexts["date"].eq(sessions[6])].copy()
    tree.add(training)
    before = tree.predict(scoring)
    scoring["oc_return_pct"] = scoring["oc_return_pct"] * -100.0
    still_before = tree.predict(scoring)
    np.testing.assert_array_equal(before, still_before)
    tree.add(scoring)
    following = contexts[contexts["date"].eq(sessions[7])]
    after = tree.predict(following)
    assert np.isfinite(after).all()
    assert tree.global_rows == len(training) + len(scoring)


def test_dense_context_backoff_matches_fixed_formula() -> None:
    tree = runner.DenseContextTree(
        "absolute",
        base=2,
        maximum_depth=1,
        row_prior_strength=2.0,
        date_prior_strength=1.0,
    )
    training = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"]
            ),
            "code": ["1", "2", "1", "2"],
            "oc_return_pct": [1.0, -1.0, 3.0, 1.0],
            "_session_index": [0, 0, 1, 1],
            "absolute_context_1": [0, 1, 0, 1],
        }
    )
    tree.add(training)
    scoring = pd.DataFrame({"absolute_context_1": [0, 1]})
    predicted = tree.predict(scoring)
    global_mean = 1.0
    reliability = min(2.0 / 4.0, 2.0 / 3.0)
    expected = np.asarray(
        [
            reliability * 2.0 + (1.0 - reliability) * global_mean,
            reliability * 0.0 + (1.0 - reliability) * global_mean,
        ]
    )
    np.testing.assert_allclose(predicted, expected, rtol=0.0, atol=1e-12)


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
        scoring,
        np.zeros(len(scoring)),
        "candidate",
    )
    assert ranked["code"].tolist() == ["1000", "2000"]
    assert ranked["model_rank"].tolist() == [1, 2]
