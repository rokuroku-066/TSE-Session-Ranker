from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from research import model_v16_liquidity_audit as audit


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v16_saved_decision_rejects_the_family_without_replay_authority() -> None:
    result = _json(RESEARCH / "model_v16_liquidity_result.json")

    assert result["status"] == "selection_rejected_all_candidates"
    assert result["selection"]["gate_passers"] == 0
    assert result["selection"]["winner"] is None
    assert result["selection"]["locked_replay_allowed"] is False
    assert result["authority"]["locked_replay_input_opened"] is False
    assert result["authority"]["project_level_untouched"] is False
    assert result["authority"]["production_model_changed"] is False
    assert result["authority"]["production_promotion_allowed"] is False
    assert result["authority"]["orders_allowed"] is False
    assert result["decision"] == {
        "candidate_family_rejected": True,
        "forward_shadow_nominee": None,
        "orders_allowed": False,
        "production_model_changed": False,
    }


def test_v16_saved_audit_is_bound_to_the_exact_selection_artifacts() -> None:
    report = _json(RESEARCH / "model_v16_liquidity_audit.json")
    bindings = report["artifact_hashes"]
    expected_paths = {
        "protocol": RESEARCH / "model_v16_liquidity_protocol.json",
        "result": RESEARCH / "model_v16_liquidity_result.json",
        "picks": RESEARCH / "model_v16_liquidity_picks.csv",
        "runner": RESEARCH / "model_v16_liquidity_runner.py",
        "audit_runner": RESEARCH / "model_v16_liquidity_audit.py",
        "parser_source": ROOT / "src/tse_session_ranker/data/jpx.py",
        "price_manifest": RESEARCH / "model_v05_input_lock.json",
        "parser_audit": RESEARCH / "model_v04_parser_recovery_audit.json",
    }

    assert report["status"] == "pass"
    assert report["discrepancies"] == []
    assert all(report["checks"].values())
    assert report["recomputed"]["gate_passers"] == 0
    assert report["recomputed"]["winner"] is None
    assert report["recomputed"]["maximum_absolute_numeric_difference"] == 0.0
    assert {
        name: _sha256(path) for name, path in expected_paths.items()
    } == bindings


def test_v16_independent_audit_recomputes_without_runner_helpers() -> None:
    recomputed = audit.audit(
        protocol_path=RESEARCH / "model_v16_liquidity_protocol.json",
        result_path=RESEARCH / "model_v16_liquidity_result.json",
        picks_path=RESEARCH / "model_v16_liquidity_picks.csv",
        runner_path=RESEARCH / "model_v16_liquidity_runner.py",
    )

    assert recomputed["status"] == "pass"
    assert recomputed["discrepancies"] == []
    assert recomputed["recomputed"]["gate_passers"] == 0
    assert recomputed["recomputed"]["winner"] is None
    assert recomputed["independence"] == {
        "selection_runner_imported": False,
        "project_profit_helpers_imported": False,
        "project_bootstrap_helpers_imported": False,
        "recomputed_from": [
            "research/model_v16_liquidity_protocol.json",
            "research/model_v16_liquidity_result.json",
            "research/model_v16_liquidity_picks.csv",
        ],
    }


def test_v16_audit_rejects_incomplete_warmup_bindings() -> None:
    protocol = _json(RESEARCH / "model_v16_liquidity_protocol.json")
    result = _json(RESEARCH / "model_v16_liquidity_result.json")
    manifest = _json(RESEARCH / "model_v05_input_lock.json")

    assert audit.warmup_source_bindings_match(protocol, result, manifest)

    shortened = copy.deepcopy(result)
    shortened["input"]["price_warmup_sources"].pop()
    assert not audit.warmup_source_bindings_match(
        protocol, shortened, manifest
    )

    duplicated = copy.deepcopy(result)
    duplicated["input"]["price_warmup_sources"][2] = copy.deepcopy(
        duplicated["input"]["price_warmup_sources"][1]
    )
    assert not audit.warmup_source_bindings_match(
        protocol, duplicated, manifest
    )

    wrong_size = copy.deepcopy(result)
    wrong_size["input"]["price_warmup_sources"][0]["bytes"] += 1
    assert not audit.warmup_source_bindings_match(
        protocol, wrong_size, manifest
    )


def test_v16_audit_rejects_corrupt_outcomes_and_slot_replacement(
    tmp_path: Path,
) -> None:
    picks = pd.read_csv(
        RESEARCH / "model_v16_liquidity_picks.csv",
        dtype=str,
    )

    corrupt_numeric = picks.copy()
    corrupt_numeric.loc[0, "oc_return_pct"] = "not-a-number"
    corrupt_path = tmp_path / "corrupt-numeric.csv"
    corrupt_numeric.to_csv(corrupt_path, index=False)
    with pytest.raises(ValueError, match="non-numeric oc_return_pct"):
        audit.load_picks(corrupt_path)

    mismatched_presence = picks.copy()
    observed_index = mismatched_presence["label"].first_valid_index()
    assert observed_index is not None
    mismatched_presence.loc[observed_index, "oc_return_pct"] = None
    mismatch_path = tmp_path / "mismatched-outcome.csv"
    mismatched_presence.to_csv(mismatch_path, index=False)
    with pytest.raises(ValueError, match="label and return presence"):
        audit.load_picks(mismatch_path)

    loaded = audit.load_picks(
        RESEARCH / "model_v16_liquidity_picks.csv"
    )
    assert audit.outcome_contract_matches(loaded)
    assert audit.fixed_slot_contract_matches(loaded)
    selected_index = loaded.index[
        ~loaded["vetoed"] & loaded["code"].notna()
    ][0]
    loaded.loc[selected_index, "pre_veto_code"] = "9999"
    assert not audit.fixed_slot_contract_matches(loaded)
