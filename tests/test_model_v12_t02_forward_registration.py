from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_PATH = (
    ROOT / "research" / "model_v12_t02_forward_registration.json"
)
VALIDATION_PATH = ROOT / "VALIDATION.md"
README_PATH = ROOT / "README.md"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "tests.yml"

ENTRY_016_PREFIX_BYTES = 150_920
ENTRY_016_PREFIX_SHA256 = (
    "33a3e107942dc5b0f128b16aee13cfd318a895059b9600067cf3db7c573ce4a8"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_registration() -> dict[str, object]:
    value = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_registration_binds_every_phase0_artifact_exactly() -> None:
    registration = load_registration()
    artifacts = registration["bound_artifacts"]

    assert isinstance(artifacts, dict)
    assert set(artifacts) == {
        "readme",
        "forward_protocol",
        "next_validation_plan",
        "phase0_runtime",
        "protocol_tests",
        "phase0_runtime_tests",
        "readiness_recomputation_tests",
        "pull_request_ci",
    }
    for artifact in artifacts.values():
        assert isinstance(artifact, dict)
        path = ROOT / str(artifact["path"])
        assert path.is_file()
        assert sha256(path) == artifact["sha256"]


def test_registration_is_explicitly_non_production_and_not_activated() -> None:
    registration = load_registration()
    authority = registration["authority"]
    evidence = registration["phase0_evidence"]
    activation = registration["activation_state"]
    decision = registration["decision"]

    assert authority == {
        "scope": (
            "Protocol correction, fail-closed Phase-0 runtime, and "
            "synthetic fixture validation only."
        ),
        "historical_artifacts_immutable": True,
        "retrospective_or_forward_performance_claimed": False,
        "production_model_changed": False,
        "orders_allowed": False,
    }
    assert evidence["runtime_test_count"] == 18
    assert evidence["uses_synthetic_fixtures_only"] is True
    assert evidence["external_market_data_used"] is False
    assert evidence["market_return_observations"] == 0
    assert activation == {
        "payload_present": False,
        "receipt_present": False,
        "earliest_calendar_eligible_session": None,
        "first_counted_session": None,
        "forward_counter": 0,
    }
    assert decision["phase0_runtime"] == "PASS_SYNTHETIC_ONLY"
    assert decision["activation"] == "NOT_STARTED"
    assert decision["historical_oot"] == "BLOCKED_INPUT_MISSING"
    assert decision["production_candidate"] is False
    assert decision["orders_allowed"] is False


def test_validation_history_preserves_entry_016_byte_for_byte() -> None:
    content = VALIDATION_PATH.read_bytes()
    prefix = content[:ENTRY_016_PREFIX_BYTES]

    assert len(prefix) == ENTRY_016_PREFIX_BYTES
    assert hashlib.sha256(prefix).hexdigest() == ENTRY_016_PREFIX_SHA256
    text = content.decode("utf-8")
    assert text.count("## Entry 017 —") == 1
    assert content.index("## Entry 017 —".encode("utf-8")) >= (
        ENTRY_016_PREFIX_BYTES
    )
    assert "v12_t02_forward_protocol_v2_pr4_20260727" in text
    assert "Phase 0" in text
    assert "forward counter:                 0" in text
    assert "orders allowed:                  false" in text


def test_readme_and_ci_expose_the_current_boundary() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "v1.2 T02 forward protocol v2" in readme
    assert "payload・receiptとも未作成" in readme
    assert "forward counter\n`0`" in readme
    assert "`orders_allowed=false`" in readme
    assert "source watermark" in readme
    assert "python -m compileall -q src research tests" in workflow
    assert "python -m pytest -q" in workflow
