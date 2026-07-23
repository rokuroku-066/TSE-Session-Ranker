from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.model_v11_calendar_runner import (
    HYPOTHESES,
    annotate_calendar,
    daily_returns,
)


ROOT = Path(__file__).resolve().parents[1]


def test_calendar_protocol_freezes_twelve_mechanisms_and_twenty_four_policies() -> None:
    protocol = json.loads(
        (ROOT / "research/model_v11_calendar_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert [value["id"] for value in protocol["registered_hypotheses"]] == list(
        HYPOTHESES
    )
    assert len(HYPOTHESES) == 12
    assert protocol["candidate_family_size"] == 24
    assert protocol["portfolio"]["capacities"] == [1, 2]
    assert protocol["portfolio"]["scheduled_day_denominator"].startswith(
        "All 266"
    )
    assert (
        protocol["authority"]["production_promotion_allowed_from_this_run"]
        is False
    )


def test_calendar_annotations_are_date_deterministic() -> None:
    sessions = pd.DatetimeIndex(
        [
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
            "2025-01-10",
            "2025-01-14",
            "2025-01-15",
            "2025-01-30",
            "2025-01-31",
            "2025-02-03",
        ]
    )
    value = annotate_calendar(sessions).set_index("date")
    assert value.loc[pd.Timestamp("2025-01-14"), "post_holiday"]
    assert value.loc[pd.Timestamp("2025-01-10"), "pre_holiday"]
    assert value.loc[pd.Timestamp("2025-01-10"), "sq_proxy"]
    assert value.loc[pd.Timestamp("2025-01-31"), "month_end"]
    assert value.loc[pd.Timestamp("2025-02-03"), "month_start"]
    assert value.loc[pd.Timestamp("2025-02-03"), "turn_month"]


def test_calendar_top_two_cash_slot_is_not_renormalised() -> None:
    sessions = pd.DatetimeIndex(["2025-01-06", "2025-01-07"])
    picks = pd.DataFrame(
        {
            "date": [
                "2025-01-06",
                "2025-01-06",
                "2025-01-07",
                "2025-01-07",
            ],
            "capacity": [2, 2, 2, 2],
            "code": ["1001", None, None, None],
            "oc_return_pct": [1.0, np.nan, np.nan, np.nan],
        }
    )
    daily = daily_returns(picks, sessions, 20)
    assert daily.loc[sessions[0], "gross"] == 0.5
    assert daily.loc[sessions[0], "net"] == 0.4
    assert daily.loc[sessions[0], "exposure"] == 0.5
    assert daily.loc[sessions[1], "net"] == 0.0


def test_calendar_artifacts_reject_every_policy_after_integrity_passes() -> None:
    result = json.loads(
        (ROOT / "research/model_v11_calendar_result.json").read_text(
            encoding="utf-8"
        )
    )
    audit = json.loads(
        (ROOT / "research/model_v11_calendar_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["coverage"]["score_sessions"] == 266
    assert result["coverage"]["strict_tdnet_score_sessions"] == 88
    assert len(result["metrics"]) == 24
    assert result["integrity"]["target_session_outcome_mutation"]["passes"]
    assert audit["integrity"]["passes"]
    assert audit["pnl_reproduction"]["passes"]
    assert audit["deterministic_reproduction"]["passes"]
    assert audit["multiplicity"]["family_size"] == 24
    assert not any(
        value["passes_all"] for value in audit["candidate_gates"].values()
    )
    assert audit["decision"]["retrospective_decision"] == (
        "REJECT_ALL_CALENDAR_HYPOTHESES"
    )
    assert audit["decision"]["forward_shadow_finalist"] is None
    assert audit["decision"]["production_ready"] is False
