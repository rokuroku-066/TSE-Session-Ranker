from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.model_v11_distributional_audit import independently_join_outcomes
from research.model_v11_distributional_runner import (
    Prediction,
    daily_returns,
    select_policy_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def test_distributional_protocol_is_a_fixed_sixteen_policy_family() -> None:
    protocol = json.loads(
        (ROOT / "research/model_v11_distributional_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(protocol["hypotheses"]) == 8
    assert protocol["candidate_family_size"] == 16
    assert protocol["portfolio"]["capacities"] == [1, 2]
    assert protocol["evaluation"]["multiplicity"].startswith("5000 deterministic")
    assert protocol["production_gate"]["prospective_sessions_at_least"] == 60
    assert (
        protocol["authority"]["production_promotion_allowed_from_this_panel"]
        is False
    )
    assert any(
        "v10 HistGradientBoosting" in value
        for value in protocol["prohibited_reuse"]
    )


def test_cash_slot_is_not_renormalised() -> None:
    sessions = pd.DatetimeIndex(["2025-01-06", "2025-01-07"])
    picks = pd.DataFrame(
        {
            "date": [
                sessions[0],
                sessions[0],
                sessions[1],
                sessions[1],
            ],
            "capacity": [2, 2, 2, 2],
            "code": ["1001", np.nan, np.nan, np.nan],
            "oc_return_pct": [1.0, np.nan, np.nan, np.nan],
        }
    )
    value = daily_returns(picks, sessions, 20.0)
    assert value.loc[sessions[0], "gross"] == 0.5
    assert value.loc[sessions[0], "net"] == 0.4
    assert value.loc[sessions[0], "exposure"] == 0.5
    assert value.loc[sessions[1], "net"] == 0.0


def test_slotwise_stop_does_not_backfill_a_rejected_first_slot() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-06"] * 3),
            "code": ["1001", "1002", "1003"],
            "name": ["a", "b", "c"],
            "label": [1.0, 1.0, 1.0],
            "oc_return_pct": [1.0, 1.0, 1.0],
        }
    )
    prediction = Prediction(
        score=np.array([3.0, 2.0, 1.0]),
        allowed=np.array([False, True, True]),
        native_rank=np.array([1, 2, 3]),
    )
    top1 = select_policy_rows(
        frame, prediction, "D08_SLOTWISE_CASH_STOP", capacity=1
    )
    top2 = select_policy_rows(
        frame, prediction, "D08_SLOTWISE_CASH_STOP", capacity=2
    )
    assert top1.empty
    assert top2["code"].tolist() == ["1002"]
    assert top2["model_rank"].tolist() == [2]


def test_independent_join_ignores_embedded_pick_outcomes() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-06", "2025-01-06"]),
            "code": ["1001", "1002"],
            "oc_return_pct": [1.25, -0.75],
            "label": [1.0, 0.0],
        }
    )
    picks = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-06", "2025-01-06"]),
            "model_rank": [1, 2],
            "code": pd.Series(["1001", pd.NA], dtype="string"),
            "policy_id": ["X_K2", "X_K2"],
            "hypothesis_id": ["X", "X"],
            "capacity": [2, 2],
            "oc_return_pct": [99.0, np.nan],
            "label": [0.0, np.nan],
        }
    )
    audited, embedded_exact = independently_join_outcomes(picks, panel)
    assert embedded_exact is False
    assert audited.loc[0, "oc_return_pct"] == 1.25
    assert audited.loc[0, "label"] == 1.0
    assert pd.isna(audited.loc[1, "oc_return_pct"])
