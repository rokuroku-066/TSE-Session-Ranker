from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
PROTOCOL = RESEARCH / "model_v13_symbolic_context_protocol.json"
ERRATUM = RESEARCH / "model_v13_symbolic_context_input_erratum_v2.json"
RUNNER = RESEARCH / "model_v13_symbolic_context_runner.py"
RESULT = RESEARCH / "model_v13_symbolic_context_result.json"
PICKS = RESEARCH / "model_v13_symbolic_context_picks.csv"
AUDIT_RUNNER = RESEARCH / "model_v13_symbolic_context_audit.py"
AUDIT = RESEARCH / "model_v13_symbolic_context_audit.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_result_is_bound_to_frozen_inputs_and_non_production() -> None:
    result = _json(RESULT)
    audit = _json(AUDIT)

    assert result["base_protocol_sha256"] == _sha256(PROTOCOL)
    assert result["input_erratum_sha256"] == _sha256(ERRATUM)
    assert result["runner_sha256"] == _sha256(RUNNER)
    assert result["integrity"]["picks_semantic_sha256"] == (
        "3d47891da9a18469bce7a38ee4b74952485a6092f25a54e326817b88f8cb8715"
    )
    assert audit["result_sha256"] == _sha256(RESULT)
    assert audit["picks_file_sha256"] == _sha256(PICKS)
    assert audit["audit_runner_sha256"] == _sha256(AUDIT_RUNNER)

    assert result["integrity"]["production_model_changed"] is False
    assert result["integrity"]["orders_allowed"] is False
    assert result["decision"]["forward_shadow_candidate"] is None
    assert result["decision"]["production_candidate"] is None
    assert result["decision"]["orders_allowed"] is False


def test_all_variants_fail_and_best_point_estimate_is_still_negative_net40() -> None:
    result = _json(RESULT)
    variants = {
        item["variant_id"]: item
        for item in result["variants"]
    }

    assert len(variants) == 8
    assert result["decision"]["gate_passers"] == []
    assert all(item["gate_passed"] is False for item in variants.values())

    best = variants["CT02_RELATIVE_PATH__top1"]
    assert best["cost_metrics"]["20"]["net_mean_pct_at_cost"] == pytest.approx(
        0.18117585771090353,
        abs=1e-12,
    )
    assert best["cost_metrics"]["40"]["net_mean_pct_at_cost"] == pytest.approx(
        -0.018824142289096436,
        abs=1e-12,
    )
    assert best["top10_days_removed_net40_mean_pct"] == pytest.approx(
        -0.3262481196230469,
        abs=1e-12,
    )
    assert best["top10_profit_codes_cash_net40_mean_pct"] == pytest.approx(
        -0.31375675112424634,
        abs=1e-12,
    )
    assert best["positive_months_net40"] == 5
    assert best["paired_vs_control_net40"][
        "one_sided_lower_delta_pct"
    ] == pytest.approx(-0.43542810785373276, abs=1e-12)


def test_independent_audit_reproduces_committed_audit(tmp_path: Path) -> None:
    reproduced = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            str(AUDIT_RUNNER),
            "--result",
            str(RESULT),
            "--picks",
            str(PICKS),
            "--output",
            str(reproduced),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    expected = _json(AUDIT)
    observed = _json(reproduced)
    assert observed == expected
    assert observed["decision"]["audit_passed"] is True
    assert observed["gate_passers"] == []
    assert set(observed["checks"].values()) == {"PASS"}
