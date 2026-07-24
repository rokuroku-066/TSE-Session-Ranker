#!/usr/bin/env python3
"""Freeze the only valid out-of-time decision path for the v10 T02 policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
PROTOCOL = RESEARCH / "model_v11_t02_oot_protocol.json"
SIDECAR = RESEARCH / "model_v11_t02_oot_protocol.sha256"
EXPECTED_PROTOCOL_SHA256 = (
    "5eceb8bf396a703a6a3bf217837331f6af95a7b303c91c79e639d132f575024d"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_t02_out_of_time_protocol_is_immutable_and_bound() -> None:
    protocol = read_json(PROTOCOL)
    assert sha256_file(PROTOCOL) == EXPECTED_PROTOCOL_SHA256
    assert SIDECAR.read_text(encoding="utf-8").strip() == (
        f"{EXPECTED_PROTOCOL_SHA256}  model_v11_t02_oot_protocol.json"
    )
    assert (
        protocol["authority"]["production_promotion_allowed_before_all_gates_pass"]
        is False
    )

    for binding in protocol["bound_candidate_artifacts"].values():
        path = ROOT / binding["path"]
        assert path.is_file()
        assert sha256_file(path) == binding["sha256"]


def test_t02_window_is_strictly_after_discovery_and_cannot_be_tuned() -> None:
    protocol = read_json(PROTOCOL)
    window = protocol["evaluation_window"]
    assert window["score_start"] == "2025-08-04"
    assert window["score_end"] == "2026-03-31"
    assert window["expected_score_sessions"] == 159
    assert window["minimum_source_complete_sessions"] == 120
    assert protocol["candidate"]["id"] == "v10_t02_char_value_event_top1"
    assert protocol["candidate"]["decision_cutoff"] == "08:58:59"
    assert any(
        "changing T02 vectorizer" in item
        for item in protocol["authority"]["forbidden"]
    )
    assert protocol["candidate"]["estimator"] == {
        "type": "Ridge",
        "alpha": 20.0,
    }


def test_statistical_and_execution_gates_are_both_mandatory() -> None:
    protocol = read_json(PROTOCOL)
    statistical = protocol["statistical_promotion_gate"]
    execution = protocol["execution_promotion_gate"]
    assert statistical["all_required"] is True
    assert execution["all_required_after_statistical_gate"] is True
    assert len(statistical["requirements"]) >= 10
    assert "exact 08:58 indicative price or executable order simulation" in (
        execution["requirements"]
    )
    assert "realized spread and slippage" in execution["requirements"]
    assert "minimum trading unit and order value" in execution["requirements"]
    assert execution["status_before_data"] == (
        "not_evaluable_from_OHLC_and_titles"
    )
