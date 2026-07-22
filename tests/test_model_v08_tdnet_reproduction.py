from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from research.evaluate_model_v08_tdnet import (
    EXPECTED_PROTOCOL_SHA256,
    _validate_historical_bindings,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "research/model_v08_tdnet_protocol.json"
RESULT_PATH = ROOT / "research/model_v08_tdnet_result.json"
MANIFEST_PATH = ROOT / "research/model_v08_tdnet_result.manifest.json"
RUNNER_PATH = ROOT / "research/evaluate_model_v08_tdnet.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_stable_historical_registrations_match_bound_bytes() -> None:
    protocol = _json(PROTOCOL_PATH)
    bindings = protocol["historical_precommit_bindings"]
    assert len(bindings) == 4
    for binding in bindings.values():
        stable = ROOT / binding["stable_path"]
        assert stable.is_file()
        assert _sha256(stable) == binding["sha256"]
    _validate_historical_bindings(protocol)


def test_historical_binding_tamper_is_rejected() -> None:
    protocol = _json(PROTOCOL_PATH)
    changed = copy.deepcopy(protocol)
    changed["historical_precommit_bindings"][
        "generation_2_protocol"
    ]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="historical stable artifact changed"):
        _validate_historical_bindings(changed)


def test_canonical_result_manifest_hash_chain_is_complete() -> None:
    protocol = _json(PROTOCOL_PATH)
    result = _json(RESULT_PATH)
    manifest = _json(MANIFEST_PATH)

    protocol_sha256 = _sha256(PROTOCOL_PATH)
    assert protocol_sha256 == EXPECTED_PROTOCOL_SHA256
    assert manifest["protocol_sha256"] == protocol_sha256
    assert manifest["result_sha256"] == _sha256(RESULT_PATH)
    assert manifest["runner_sha256"] == _sha256(RUNNER_PATH)
    assert manifest["production_promotion_allowed"] is False
    assert manifest["production_changed"] is False
    assert result["bindings"]["protocol_sha256"] == protocol_sha256
    assert (
        result["bindings"]["historical_precommit_bindings"]
        == protocol["historical_precommit_bindings"]
    )


def test_canonical_tdnet_economics_are_unchanged() -> None:
    result = _json(RESULT_PATH)
    generation_3 = result["generation_3_alpha10"]
    generation_4 = result["generation_4_alpha1"]

    assert generation_3["qualified"] == []
    assert generation_4["qualified"] == []
    assert result["verdict"]["stop_rule_triggered"] is True
    assert (
        generation_3["baseline"]["periods"]["full"]["net20"]
        ["net_mean_pct_at_cost"]
        == pytest.approx(0.1342002695206938, abs=1e-15)
    )
    assert (
        generation_3["candidates"]["H08_arrival_timing"]["periods"]
        ["full"]["uplift_net20_pct"]
        == pytest.approx(0.00992770497346713, abs=1e-15)
    )
    assert (
        generation_3["candidates"]["H10_stage_linear_runup"]["periods"]
        ["full"]["uplift_net20_pct"]
        == pytest.approx(0.007330573657976446, abs=1e-15)
    )
    assert (
        generation_4["baseline"]["periods"]["full"]["net20"]
        ["net_mean_pct_at_cost"]
        == pytest.approx(0.2190633531227891, abs=1e-15)
    )
    assert (
        generation_4["candidates"]["H08_arrival_timing"]["periods"]
        ["full"]["net20"]["net_mean_pct_at_cost"]
        == pytest.approx(0.15083936602024478, abs=1e-15)
    )
    assert (
        generation_4["candidates"]["H10_stage_linear_runup"]["periods"]
        ["full"]["net20"]["net_mean_pct_at_cost"]
        == pytest.approx(0.21344049974915497, abs=1e-15)
    )
    assert (
        generation_4["candidates"]["H08_H10"]["periods"]["full"]
        ["net20"]["net_mean_pct_at_cost"]
        == pytest.approx(0.19275344029775834, abs=1e-15)
    )
