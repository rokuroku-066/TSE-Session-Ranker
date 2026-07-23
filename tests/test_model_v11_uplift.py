from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.model_v11_uplift_audit import independent_outcome_join
from research.model_v11_uplift_runner import (
    HYPOTHESES,
    daily_returns,
    strict_sessions,
)


ROOT = Path(__file__).resolve().parents[1]


def test_uplift_protocol_freezes_eight_distinct_counterfactual_mechanisms() -> None:
    protocol = json.loads(
        (ROOT / "research/model_v11_uplift_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert [value["id"] for value in protocol["registered_hypotheses"]] == list(
        HYPOTHESES
    )
    assert len(protocol["registered_hypotheses"]) == 8
    assert protocol["candidate_family_size"] == 16
    assert protocol["portfolio"]["capacities"] == [1, 2]
    assert protocol["evaluation"]["multiplicity"].startswith(
        "5000 deterministic circular"
    )
    assert (
        protocol["authority"]["production_promotion_allowed_from_this_run"]
        is False
    )


def test_strict_tdnet_sessions_require_every_intervening_calendar_page() -> None:
    sessions = pd.DatetimeIndex(["2025-01-06", "2025-01-07", "2025-01-10"])
    cache = {
        pd.Timestamp("2025-01-06"),
        pd.Timestamp("2025-01-07"),
        pd.Timestamp("2025-01-08"),
        pd.Timestamp("2025-01-10"),
    }
    accepted, rejected = strict_sessions(sessions, cache)
    assert accepted.equals(pd.DatetimeIndex(["2025-01-07"]))
    assert rejected["2025-01-06"] == ["prior panel session unavailable"]
    assert rejected["2025-01-10"] == ["2025-01-09"]


def test_empty_top_two_slot_remains_cash_without_weight_renormalisation() -> None:
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
    value = daily_returns(picks, sessions, cost_bps=20)
    assert value.loc[sessions[0], "gross"] == 0.5
    assert value.loc[sessions[0], "net"] == 0.4
    assert value.loc[sessions[0], "exposure"] == 0.5
    assert value.loc[sessions[1], "net"] == 0.0


def test_independent_join_detects_embedded_outcome_tampering() -> None:
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
            "code": pd.Series(["1001", pd.NA], dtype="string"),
            "oc_return_pct": [99.0, np.nan],
            "label": [0.0, np.nan],
        }
    )
    joined, exact = independent_outcome_join(picks, panel)
    assert exact is False
    assert joined.loc[0, "oc_return_pct"] == 1.25
    assert joined.loc[0, "label"] == 1.0
    assert pd.isna(joined.loc[1, "oc_return_pct"])


def test_uplift_artifacts_fail_closed_after_all_integrity_audits_pass() -> None:
    result = json.loads(
        (ROOT / "research/model_v11_uplift_result.json").read_text(
            encoding="utf-8"
        )
    )
    audit = json.loads(
        (ROOT / "research/model_v11_uplift_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(result["metrics"]) == 16
    assert result["coverage"]["strict_score_sessions"] == 88
    assert result["integrity"]["target_session_outcome_mutation"]["passes"]
    assert audit["integrity"]["passes"]
    assert audit["pnl_reproduction"]["passes"]
    assert audit["deterministic_reproduction"]["passes"]
    assert audit["multiplicity"]["family_size"] == 16
    assert not any(
        value["passes_all"] for value in audit["candidate_gates"].values()
    )
    assert audit["decision"]["retrospective_decision"] == (
        "REJECT_ALL_UPLIFT_HYPOTHESES"
    )
    assert audit["decision"]["forward_shadow_finalist"] is None
    assert audit["decision"]["production_ready"] is False
