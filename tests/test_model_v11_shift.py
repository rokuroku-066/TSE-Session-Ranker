from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.model_v11_shift_runner import (
    STATE_FEATURES,
    daily_state_transform,
    normalise_within_date,
)


ROOT = Path(__file__).resolve().parents[1]


def test_shift_protocol_registers_eight_distinct_mechanisms() -> None:
    protocol = json.loads(
        (ROOT / "research/model_v11_shift_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    identifiers = [value["id"] for value in protocol["hypotheses"]]
    assert len(identifiers) == 8
    assert len(set(identifiers)) == 8
    assert protocol["candidate_family_size"] == 16
    assert protocol["portfolio"]["capacities"] == [1, 2]
    assert protocol["authority"]["production_promotion_allowed_from_this_panel"] is False
    prohibited = " ".join(protocol["prohibited_reuse"])
    assert "M09 temporal median" in prohibited
    assert "D07 posterior regime abstention" in prohibited
    assert "grid search" in prohibited


def test_shift_weights_are_renormalised_inside_each_date() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-06", "2025-01-06", "2025-01-07"]
            )
        }
    )
    value = normalise_within_date(frame, np.array([1.0, 3.0, 8.0]))
    assert np.allclose(value, [0.25, 0.75, 1.0])
    totals = pd.Series(value).groupby(frame["date"]).sum()
    assert np.allclose(totals, 1.0)


def test_market_state_transform_uses_one_row_per_date() -> None:
    values: dict[str, object] = {
        "date": pd.to_datetime(
            ["2025-01-06", "2025-01-06", "2025-01-07", "2025-01-07"]
        )
    }
    for index, name in enumerate(STATE_FEATURES):
        values[name] = [float(index), float(index), float(index + 1), np.nan]
    frame = pd.DataFrame(values)
    daily, transformed, _, _ = daily_state_transform(frame)
    assert daily["date"].tolist() == [
        pd.Timestamp("2025-01-06"),
        pd.Timestamp("2025-01-07"),
    ]
    assert transformed.shape == (2, len(STATE_FEATURES))
    assert np.isfinite(transformed).all()
