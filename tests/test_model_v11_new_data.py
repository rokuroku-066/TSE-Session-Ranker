#!/usr/bin/env python3
"""Integrity checks for the model-v11 information-space experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
PROTOCOL = RESEARCH / "model_v11_new_data_protocol.json"
RESULT = RESEARCH / "model_v11_new_data_result.json"
AUDIT = RESEARCH / "model_v11_new_data_audit.json"
MANIFEST = RESEARCH / "model_v11_new_data_manifest.json"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registered_information_channels_are_fail_closed() -> None:
    protocol = read_json(PROTOCOL)
    result = read_json(RESULT)

    registered = protocol["registered_data_generation_hypotheses"]
    availability = result["availability"]
    implementable = availability["implementable_hypotheses"]
    blocked = availability["blocked_hypotheses"]

    assert len(registered) == 11
    assert len({item["id"] for item in registered}) == 11
    assert len(implementable) == 2
    assert len(blocked) == 9
    assert len(implementable) + len(blocked) == len(registered)
    assert result["integrity"]["blocked_hypotheses_scored"] == 0
    assert result["integrity"]["URL_treated_as_document_body"] is False
    assert result["integrity"]["title_vectorizer_used"] is False
    assert result["integrity"]["hyperparameter_search_used"] is False


def test_protocol_result_and_manifest_hash_bindings_are_exact() -> None:
    protocol = read_json(PROTOCOL)
    result = read_json(RESULT)
    manifest = read_json(MANIFEST)

    assert protocol["authority"]["production_promotion_allowed"] is False
    assert result["protocol_id"] == protocol["protocol_id"]
    assert result["protocol_sha256"] == sha256_file(PROTOCOL)
    assert result["conclusion_policy"]["production_ready"] is False
    assert result["conclusion_policy"]["promotion_gate_enabled"] is False

    for relative_path, expected in manifest["files"].items():
        path = ROOT / relative_path
        assert path.is_file()
        assert path.stat().st_size == expected["bytes"]
        assert sha256_file(path) == expected["sha256"]


def test_candidate_metrics_keep_cash_cost_and_session_contracts() -> None:
    result = read_json(RESULT)
    assert result["coverage"]["accepted_score_sessions"] == 107
    assert result["coverage"]["future_publication_violations"] == 0
    assert result["coverage"]["pre_prior_close_violations"] == 0
    assert result["integrity"]["strictly_prior_training_all_scored_folds"]
    assert result["integrity"]["source_missing_never_encoded_as_no_event"]

    assert set(result["candidates"]) == {
        "L4_price_control",
        "ND04_simultaneous_adverse_bundle",
        "ND05_release_clock_and_fiscal_horizon",
        "ND04_plus_ND05",
    }
    for candidate_id, by_top_k in result["candidates"].items():
        for top_k in (1, 2):
            metrics = by_top_k[f"top{top_k}"]
            costs = metrics["costs_bps"]
            assert set(costs) == {"0", "20", "40", "60"}
            gross = costs["0"]["mean_pct"]
            exposure = metrics["filled_slots"] / metrics["scheduled_slots"]
            for cost_bps in (20, 40, 60):
                expected = gross - (cost_bps / 100.0) * exposure
                assert abs(costs[str(cost_bps)]["mean_pct"] - expected) < 1e-12
            assert costs["40"]["days"] == 107
            assert metrics["months"] == 6


def test_independent_audit_recomputes_every_saved_metric() -> None:
    result = read_json(RESULT)
    audit = read_json(AUDIT)

    assert audit["audit_status"] == "pass"
    assert audit["independent_of_runner_imports"] is True
    assert audit["assertions"]["all_reported_metrics_match"] is True
    assert audit["assertions"]["familywise_tests_match"] is True
    assert audit["assertions"]["numeric_or_scalar_checks"] == 408
    assert audit["assertions"]["structural_groups_checked"] == 8
    assert audit["hashes"]["protocol_sha256"] == sha256_file(PROTOCOL)
    assert audit["hashes"]["result_sha256"] == sha256_file(RESULT)
    assert set(audit["familywise_recomputed"]) == set(
        result["familywise"]["tests"]
    )
