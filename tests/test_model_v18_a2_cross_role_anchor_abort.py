from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import inspect
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

from research import model_v18_shoulder_state_runner as runner


TARGET = "2026-08-06"
RECEIPT = "2026-08-06T15:35:00+09:00"


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def _write_external_object(root: Path, key: str, payload: bytes) -> Path:
    target = root.joinpath(*key.split("/"))
    _private_directory(target.parent)
    target.write_bytes(payload)
    target.chmod(0o600)
    return target


def _tree_snapshot(root: Path) -> tuple[tuple[Any, ...], ...]:
    rows: list[tuple[Any, ...]] = []
    for path in sorted([root, *root.rglob("*")], key=lambda item: item.as_posix()):
        metadata = path.lstat()
        rows.append(
            (
                "." if path == root else path.relative_to(root).as_posix(),
                stat.S_IFMT(metadata.st_mode),
                stat.S_IMODE(metadata.st_mode),
                metadata.st_uid,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                path.read_bytes() if stat.S_ISREG(metadata.st_mode) else None,
            )
        )
    return tuple(rows)


def _cross_role_fixture(
    tmp_path: Path,
    *,
    alias_raw_objects: bool = False,
) -> tuple[
    Path,
    Path,
    Path,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    predictor_root = _private_directory(tmp_path / "predictor-raw")
    derived_root = _private_directory(tmp_path / "predictor-derived")
    outcome_root = _private_directory(tmp_path / "outcome-raw")
    file_name = "stq_20260805.pdf"
    url = f"https://www.jpx.co.jp/markets/statistics-equities/daily/{file_name}"
    payload = b"%PDF-1.7\nseparately retained official bytes\n"
    digest = hashlib.sha256(payload).hexdigest()
    predictor_key = f"{runner.PREDICTOR_OBJECT_PREFIX}daily/{file_name}"
    outcome_key = f"{runner.OUTCOME_OBJECT_PREFIX}2026-08-05.pdf"
    predictor_path = _write_external_object(predictor_root, predictor_key, payload)
    if alias_raw_objects:
        outcome_path = outcome_root.joinpath(*outcome_key.split("/"))
        _private_directory(outcome_path.parent)
        os.link(predictor_path, outcome_path)
    else:
        _write_external_object(outcome_root, outcome_key, payload)
    shard_key = (
        f"{runner.PREDICTOR_SHARD_OBJECT_PREFIX}"
        f"{digest}.manifest.json"
    )
    raw = {
        "file": file_name,
        "url": url,
        "object_key": predictor_key,
        "byte_count": len(payload),
        "sha256": digest,
    }
    binding = {"shard_manifest_object_key": shard_key}
    overlap = {
        "target_session": "2026-08-05",
        "source_file_name": file_name,
        "source_url": url,
        "raw_source_object_key": outcome_key,
        "source_byte_count": len(payload),
        "source_sha256": digest,
        "source_received_at": RECEIPT,
    }
    terminal_only = {
        "target_session": TARGET,
        "source_file_name": "stq_20260806.pdf",
        "source_url": (
            "https://www.jpx.co.jp/markets/statistics-equities/daily/"
            "stq_20260806.pdf"
        ),
        "raw_source_object_key": (
            f"{runner.OUTCOME_OBJECT_PREFIX}2026-08-06.pdf"
        ),
        "source_byte_count": 19,
        "source_sha256": "f" * 64,
        "source_received_at": "2026-08-07T15:35:00+09:00",
    }
    shards = {
        shard_key: {
            "chronology_class": "forward",
            "raw_received_at": RECEIPT,
        }
    }
    return (
        predictor_root,
        derived_root,
        outcome_root,
        [raw],
        [binding],
        [overlap, terminal_only],
        shards,
    )


def _patch_cross_role_shards(
    monkeypatch: pytest.MonkeyPatch,
    shards: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, str]]:
    receipt_checks: list[tuple[str, str]] = []

    def read_shard(
        root: str | Path,
        key: str,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], bytes]:
        assert key in shards
        return dict(shards[key]), b"{}\n"

    def validate_shard(
        manifest: Mapping[str, Any],
        *,
        raw_record: Mapping[str, Any],
        expected_chronology_class: str,
        expected_raw_received_at: Any,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], None, dict[str, Any]]:
        observed = runner._timestamp(
            manifest["raw_received_at"], "test shard raw receipt"
        )
        expected = runner._timestamp(
            expected_raw_received_at, "test expected raw receipt"
        )
        receipt_checks.append((observed.isoformat(), expected.isoformat()))
        if manifest["chronology_class"] != expected_chronology_class:
            raise runner.V18Error("test shard chronology differs")
        if observed != expected:
            raise runner.V18Error("predictor/outcome raw receipt differs")
        return dict(manifest), None, dict(raw_record)

    monkeypatch.setattr(runner, "_read_external_canonical_json", read_shard)
    monkeypatch.setattr(runner, "_validate_parsed_shard_manifest", validate_shard)
    return receipt_checks


def test_prepare_day_cross_role_preflight_requires_exact_bytes_receipt_and_nonalias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (
        predictor_root,
        derived_root,
        outcome_root,
        raw,
        bindings,
        manifests,
        shards,
    ) = _cross_role_fixture(tmp_path)
    receipt_checks = _patch_cross_role_shards(monkeypatch, shards)

    runner._validate_terminal_predictor_outcome_cross_role(
        raw,
        bindings,
        manifests,
        terminal_session=TARGET,
        allow_terminal_outcome_only=True,
        predictor_raw_store_root=predictor_root,
        predictor_derived_store_root=derived_root,
        outcome_raw_store_root=outcome_root,
    )

    assert receipt_checks == [(RECEIPT, RECEIPT)]
    predictor_path = predictor_root.joinpath(*raw[0]["object_key"].split("/"))
    outcome_path = outcome_root.joinpath(
        *manifests[0]["raw_source_object_key"].split("/")
    )
    assert predictor_path.read_bytes() == outcome_path.read_bytes()
    assert not os.path.samefile(predictor_path, outcome_path)
    assert predictor_path.stat().st_nlink == outcome_path.stat().st_nlink == 1


def test_prefinal_cross_role_preflight_forbids_an_outcome_only_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (
        predictor_root,
        derived_root,
        outcome_root,
        raw,
        bindings,
        manifests,
        shards,
    ) = _cross_role_fixture(tmp_path)
    _patch_cross_role_shards(monkeypatch, shards)
    before = (
        _tree_snapshot(predictor_root),
        _tree_snapshot(derived_root),
        _tree_snapshot(outcome_root),
    )

    with pytest.raises(
        runner.V18Error, match="predictor PDF is missing before final session"
    ):
        runner._validate_terminal_predictor_outcome_cross_role(
            raw,
            bindings,
            manifests,
            terminal_session=TARGET,
            allow_terminal_outcome_only=False,
            predictor_raw_store_root=predictor_root,
            predictor_derived_store_root=derived_root,
            outcome_raw_store_root=outcome_root,
        )

    assert (
        _tree_snapshot(predictor_root),
        _tree_snapshot(derived_root),
        _tree_snapshot(outcome_root),
    ) == before


def test_postappend_retry_allows_only_the_exact_terminal_outcome_only_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (
        predictor_root,
        derived_root,
        outcome_root,
        raw,
        bindings,
        manifests,
        shards,
    ) = _cross_role_fixture(tmp_path)
    manifests[-1] = {**manifests[-1], "target_session": "2026-08-07"}
    _patch_cross_role_shards(monkeypatch, shards)

    with pytest.raises(
        runner.V18Error, match="predictor PDF is missing before final session"
    ):
        runner._validate_terminal_predictor_outcome_cross_role(
            raw,
            bindings,
            manifests,
            terminal_session=TARGET,
            allow_terminal_outcome_only=True,
            predictor_raw_store_root=predictor_root,
            predictor_derived_store_root=derived_root,
            outcome_raw_store_root=outcome_root,
        )

    finalize_source = inspect.getsource(runner.finalize_terminal)
    assert (
        "allow_terminal_outcome_only=len(outcomes) == len(decisions)"
        in finalize_source
    )


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("source_file_name", "stq_20260804.pdf"),
        ("source_url", "https://example.invalid/changed.pdf"),
        ("source_byte_count", 1),
        ("source_sha256", "0" * 64),
        ("source_received_at", "2026-08-06T15:36:00+09:00"),
    ],
)
def test_prepare_day_cross_role_mismatch_rejects_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    changed: Any,
) -> None:
    (
        predictor_root,
        derived_root,
        outcome_root,
        raw,
        bindings,
        manifests,
        shards,
    ) = _cross_role_fixture(tmp_path)
    manifests[0] = {**manifests[0], field: changed}
    _patch_cross_role_shards(monkeypatch, shards)
    before = (
        _tree_snapshot(predictor_root),
        _tree_snapshot(derived_root),
        _tree_snapshot(outcome_root),
    )

    with pytest.raises(
        runner.V18Error,
        match="different|missing before final|receipt differs",
    ):
        runner._validate_terminal_predictor_outcome_cross_role(
            raw,
            bindings,
            manifests,
            terminal_session=TARGET,
            allow_terminal_outcome_only=True,
            predictor_raw_store_root=predictor_root,
            predictor_derived_store_root=derived_root,
            outcome_raw_store_root=outcome_root,
        )

    assert (
        _tree_snapshot(predictor_root),
        _tree_snapshot(derived_root),
        _tree_snapshot(outcome_root),
    ) == before


def test_prepare_day_cross_role_rejects_a_physical_raw_alias_without_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (
        predictor_root,
        derived_root,
        outcome_root,
        raw,
        bindings,
        manifests,
        shards,
    ) = _cross_role_fixture(tmp_path, alias_raw_objects=True)
    _patch_cross_role_shards(monkeypatch, shards)
    before = (
        _tree_snapshot(predictor_root),
        _tree_snapshot(derived_root),
        _tree_snapshot(outcome_root),
    )

    with pytest.raises(runner.V18Error, match="hard-link alias|alias one inode"):
        runner._validate_terminal_predictor_outcome_cross_role(
            raw,
            bindings,
            manifests,
            terminal_session=TARGET,
            allow_terminal_outcome_only=True,
            predictor_raw_store_root=predictor_root,
            predictor_derived_store_root=derived_root,
            outcome_raw_store_root=outcome_root,
        )

    assert (
        _tree_snapshot(predictor_root),
        _tree_snapshot(derived_root),
        _tree_snapshot(outcome_root),
    ) == before


def _terminal_cross_role_rows() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    raw: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for index, session in enumerate(
        ("2026-08-03", "2026-08-04", "2026-08-05")
    ):
        token = session.replace("-", "")
        file_name = f"stq_{token}.pdf"
        digest = f"{index + 1:x}" * 64
        url = f"https://example.invalid/{file_name}"
        raw.append(
            {
                "file": file_name,
                "url": url,
                "object_key": f"{runner.PREDICTOR_OBJECT_PREFIX}{file_name}",
                "byte_count": 100 + index,
                "sha256": digest,
            }
        )
        bindings.append(
            {
                "shard_manifest_object_key": (
                    f"{runner.PREDICTOR_SHARD_OBJECT_PREFIX}{token}.manifest.json"
                )
            }
        )
        manifests.append(
            {
                "target_session": session,
                "source_file_name": file_name,
                "source_url": url,
                "raw_source_object_key": f"{runner.OUTCOME_OBJECT_PREFIX}{session}.pdf",
                "source_byte_count": 100 + index,
                "source_sha256": digest,
                "source_received_at": f"{session}T15:35:00+09:00",
            }
        )
    manifests.append(
        {
            "target_session": "2026-08-06",
            "source_file_name": "stq_20260806.pdf",
            "source_url": "https://example.invalid/stq_20260806.pdf",
            "raw_source_object_key": f"{runner.OUTCOME_OBJECT_PREFIX}2026-08-06.pdf",
            "source_byte_count": 103,
            "source_sha256": "4" * 64,
            "source_received_at": "2026-08-06T15:35:00+09:00",
        }
    )
    return raw, bindings, manifests


@pytest.mark.parametrize("overlap_index", [0, 1, 2])
def test_terminal_every_overlap_mismatch_rejects_before_performance_helper(
    monkeypatch: pytest.MonkeyPatch,
    overlap_index: int,
) -> None:
    sessions = pd.DatetimeIndex(
        ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06"]
    )
    decisions = [{"session_date": str(item.date())} for item in sessions]
    outcomes = [{"session_date": str(item.date())} for item in sessions]
    raw, bindings, manifests = _terminal_cross_role_rows()
    manifests[overlap_index] = {
        **manifests[overlap_index],
        "source_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        runner, "validate_decision_records", lambda value: decisions
    )
    monkeypatch.setattr(
        runner,
        "deterministic_terminal_session",
        lambda first, calendar: calendar[-1],
    )
    monkeypatch.setattr(
        runner, "_validate_startup_and_module_closure", lambda **kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "validate_canonical_activation_artifacts",
        lambda: ({}, "a" * 64, {}, "b" * 64),
    )
    monkeypatch.setattr(
        runner, "_validate_deferred_state_evidence", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        runner,
        "_preflight_terminal_completed_month_ledger",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        runner, "validate_completed_month_records", lambda value: [dict(value[0])]
    )
    monkeypatch.setattr(
        runner, "validate_outcome_records", lambda value, decisions: outcomes
    )
    monkeypatch.setattr(
        runner, "validate_completed_month_coverage", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner, "validate_deferred_score_evidence", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner, "validate_outcome_evidence", lambda *args, **kwargs: manifests
    )
    monkeypatch.setattr(
        runner, "_read_external_canonical_json", lambda *args, **kwargs: ({}, b"{}")
    )
    monkeypatch.setattr(
        runner, "_validate_parsed_shard_manifest", lambda *args, **kwargs: ({}, None, {})
    )
    monkeypatch.setattr(
        runner, "_assert_external_objects_nonalias", lambda *args, **kwargs: None
    )
    performance_calls: list[str] = []

    def forbidden_performance(*args: Any, **kwargs: Any) -> Any:
        performance_calls.append("_evaluation_frame")
        pytest.fail("cross-role mismatch reached the first performance helper")

    monkeypatch.setattr(runner, "_evaluation_frame", forbidden_performance)
    predictor_bindings = {
        "_outcome_blind_score_expectations": [],
        "_terminal_predictor_raw_records": raw,
        "_terminal_predictor_shard_bindings": bindings,
    }

    with pytest.raises(runner.V18Error, match="different daily bytes"):
        runner.evaluate(
            decisions,
            outcomes,
            [{"completed_month": "2026-08"}],
            calendar=sessions,
            source_manifest_directory="source-manifests",
            predictor_raw_store_root="predictor-raw",
            predictor_derived_store_root="predictor-derived",
            scores=pd.DataFrame(),
            outcome_manifest_directory="outcome-manifests",
            outcome_raw_store_root="outcome-raw",
            checkpoint_core_store_root="checkpoints",
            prevalidated_predictor_bindings=predictor_bindings,
            prevalidated_checkpoint_bindings={},
        )

    assert performance_calls == []


def _abort_context() -> dict[str, str]:
    return {
        "activation_payload_sha256": "a" * 64,
        "activation_receipt_sha256": "b" * 64,
        "activation_receipt_commit_sha": "c" * 40,
    }


def _patch_abort_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    context = _abort_context()
    protocol = {
        "result_contract": {
            "abort_stage_reason_values": {
                "source_ingestion": ["source_integrity_failure"]
            },
            "required_input_fields": [
                "runtime_lock_sha256",
                "predictor_parser_sha256",
                "v17_c00_protocol_sha256",
                "v17_c00_runner_sha256",
            ],
            "required_artifact_hashes": [
                "decision_ledger_sha256",
                "outcome_ledger_sha256",
                "completed_month_ledger_sha256",
                "score_output_sha256",
                "score_semantic_sha256",
                "picks_output_sha256",
            ],
            "authority_values": {
                "production_model_changed": False,
                "orders_allowed": False,
            },
        }
    }
    monkeypatch.setattr(runner, "_STRICT_RUNTIME_ACTIVE", True)
    monkeypatch.setattr(
        runner,
        "_STRICT_RUNTIME_VERIFIED_AT",
        datetime.fromisoformat("2026-08-05T08:00:00+09:00"),
    )
    monkeypatch.setattr(runner, "RESULT_OUTPUT", tmp_path / "result.json")
    for name in (
        "DECISION_LEDGER",
        "OUTCOME_LEDGER",
        "COMPLETED_MONTH_LEDGER",
        "SCORE_OUTPUT",
        "PICKS_OUTPUT",
    ):
        monkeypatch.setattr(runner, name, tmp_path / f"{name.lower()}.missing")
    monkeypatch.setattr(
        runner, "validate_protocol", lambda *args, **kwargs: (protocol, "d" * 64)
    )
    monkeypatch.setattr(
        runner,
        "validate_runtime_lock",
        lambda **kwargs: (
            {"runtime_lock_self_sha256": "e" * 64},
            runner.RUNTIME_LOCK_SHA256,
        ),
    )

    def validate_context(value: Mapping[str, Any]) -> dict[str, Any]:
        observed = dict(value)
        required = set(context)
        if not required.issubset(observed) or observed[
            "activation_receipt_commit_sha"
        ] != context["activation_receipt_commit_sha"]:
            raise runner.V18Error("integrity abort requires exact activated C context")
        return observed

    monkeypatch.setattr(runner, "validate_activation_context", validate_context)
    monkeypatch.setattr(
        runner,
        "validate_activation_payload",
        lambda *args, **kwargs: ({}, context["activation_payload_sha256"]),
    )
    monkeypatch.setattr(
        runner,
        "validate_activation_receipt",
        lambda *args, **kwargs: ({}, context["activation_receipt_sha256"]),
    )


def test_activated_abort_binds_exact_payload_receipt_and_commit_c(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_abort_contract(monkeypatch, tmp_path)
    context = _abort_context()

    result = runner.build_integrity_abort_result(
        failure_reason="source_integrity_failure",
        integrity_stage="source_ingestion",
        activation_context=context,
    )

    assert result["status"] == "aborted_integrity_failure"
    assert result["activation_payload_sha256"] == context[
        "activation_payload_sha256"
    ]
    assert result["activation_receipt_sha256"] == context[
        "activation_receipt_sha256"
    ]
    assert result["activation_receipt_commit_sha"] == context[
        "activation_receipt_commit_sha"
    ]
    assert result["candidate_gate"] is None
    assert result["models"] is None
    assert result["forward_period"] is None
    assert not runner.RESULT_OUTPUT.exists()


@pytest.mark.parametrize(
    ("field", "changed", "message"),
    [
        ("activation_payload_sha256", "0" * 64, "activation context changed"),
        ("activation_receipt_sha256", "0" * 64, "activation context changed"),
        ("activation_receipt_commit_sha", "0" * 40, "exact activated C context"),
        ("activation_receipt_commit_sha", None, "exact activated C context"),
    ],
)
def test_activated_abort_rejects_missing_or_changed_exact_context_without_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    changed: str | None,
    message: str,
) -> None:
    _patch_abort_contract(monkeypatch, tmp_path)
    context = _abort_context()
    if changed is None:
        context.pop(field)
    else:
        context[field] = changed

    with pytest.raises(runner.V18Error, match=message):
        runner.build_integrity_abort_result(
            failure_reason="source_integrity_failure",
            integrity_stage="source_ingestion",
            activation_context=context,
        )

    assert not runner.RESULT_OUTPUT.exists()


def _operational_command(command: str, tmp_path: Path) -> list[str]:
    common = {
        "prepare-day": [
            "prepare-day",
            "--session",
            TARGET,
            "--predictor-raw-store-root",
            str(tmp_path / "predictor-raw"),
            "--predictor-derived-store-root",
            str(tmp_path / "predictor-derived"),
            "--outcome-raw-store-root",
            str(tmp_path / "outcome-raw"),
            "--checkpoint-core-store-root",
            str(tmp_path / "checkpoint-core"),
            "--activation-context",
            str(tmp_path / "activation.json"),
        ],
        "publish-checkpoint": ["publish-checkpoint", "--session", TARGET],
        "decide": [
            "decide",
            "--session",
            TARGET,
            "--checkpoint-core-store-root",
            str(tmp_path / "checkpoint-core"),
        ],
        "finalize-terminal": [
            "finalize-terminal",
            "--source-pdf",
            str(tmp_path / "outcome.pdf"),
            "--source-file-name",
            "stq_20260806.pdf",
            "--source-url",
            "https://example.invalid/stq_20260806.pdf",
            "--source-received-at",
            "2026-08-07T15:35:00+09:00",
            "--predictor-raw-store-root",
            str(tmp_path / "predictor-raw"),
            "--predictor-derived-store-root",
            str(tmp_path / "predictor-derived"),
            "--outcome-raw-store-root",
            str(tmp_path / "outcome-raw"),
            "--checkpoint-core-store-root",
            str(tmp_path / "checkpoint-core"),
            "--activation-context",
            str(tmp_path / "activation.json"),
        ],
        "evaluate": [
            "evaluate",
            "--predictor-raw-store-root",
            str(tmp_path / "predictor-raw"),
            "--predictor-derived-store-root",
            str(tmp_path / "predictor-derived"),
            "--outcome-raw-store-root",
            str(tmp_path / "outcome-raw"),
            "--checkpoint-core-store-root",
            str(tmp_path / "checkpoint-core"),
        ],
        "abort": [
            "abort",
            "--failure-reason",
            "source_integrity_failure",
            "--integrity-stage",
            "source_ingestion",
            "--activation-context",
            str(tmp_path / "activation.json"),
        ],
    }
    return common[command]


@pytest.mark.parametrize(
    "command",
    [
        "prepare-day",
        "publish-checkpoint",
        "decide",
        "finalize-terminal",
    ],
)
def test_existing_result_blocks_every_operational_mutation_before_any_semantic_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_bytes(b'{"status":"terminal"}\n')
    result_path.chmod(0o600)
    monkeypatch.setattr(runner, "RESULT_OUTPUT", result_path)
    semantic_calls: list[str] = []

    def forbidden_semantic(*args: Any, **kwargs: Any) -> Any:
        semantic_calls.append("semantic")
        raise AssertionError("existing-result guard opened semantic authority")

    for name in (
        "validate_runtime_lock",
        "validate_protocol",
        "validate_canonical_activation_artifacts",
        "read_json",
        "write_json",
    ):
        monkeypatch.setattr(runner, name, forbidden_semantic)
    before = _tree_snapshot(tmp_path)

    with pytest.raises(runner.V18Error, match="post-result work is forbidden"):
        runner.main(_operational_command(command, tmp_path))

    assert semantic_calls == []
    assert _tree_snapshot(tmp_path) == before


def test_prepare_day_source_contains_the_registered_cross_role_preflight() -> None:
    source = inspect.getsource(runner.prepare_day)
    for token in (
        '"file": outcome_manifest["source_file_name"]',
        '"url": outcome_manifest["source_url"]',
        '"byte_count": int(outcome_manifest["source_byte_count"])',
        '"sha256": outcome_manifest["source_sha256"]',
        "expected_raw_received_at=outcome_received",
        'outcome_manifest["source_received_at"]',
        "_assert_external_objects_nonalias(",
    ):
        assert token in source


def _synthetic_full31_prices(
    *, target: str = TARGET, sessions: int = 100
) -> pd.DataFrame:
    target_date = pd.Timestamp(target)
    dates = pd.bdate_range(
        end=target_date - pd.offsets.BDay(1), periods=sessions
    )
    rows: list[dict[str, Any]] = []
    source_format = (
        "jpx_stock_quotations_auction_regular_way_domestic_ordinary"
    )
    for code_index, code in enumerate(("1001", "1002", "1003")):
        for session_index, session in enumerate(dates):
            open_price = 1_000.0 + 25.0 * code_index + 0.5 * session_index
            return_fraction = (
                0.003 if (session_index + code_index) % 2 else -0.003
            )
            close = open_price * (1.0 + return_fraction)
            high = max(open_price, close) * 1.005
            low = min(open_price, close) * 0.995
            volume = 100_000.0 + 1_000.0 * code_index + 100.0 * session_index
            vwap = (open_price + close) / 2.0
            turnover = volume * vwap
            rows.append(
                {
                    "date": session,
                    "code": code,
                    "name": f"name-{code}",
                    "raw_name": f"name-{code}",
                    "trading_unit": 100,
                    "final_special_quote": None,
                    "net_change": close - open_price,
                    "vwap": vwap,
                    "volume": volume,
                    "turnover": turnover,
                    "volume_unit": "shares",
                    "turnover_unit": "yen",
                    "source_volume_unit": "shares",
                    "source_turnover_unit": "yen",
                    "source_file": f"stq_{session:%Y%m%d}.txt",
                    "source_line": code_index + 1,
                    "source_format": source_format,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "am_open": open_price,
                    "am_high": high,
                    "am_low": low,
                    "am_close": vwap,
                    "pm_open": vwap,
                    "pm_high": high,
                    "pm_low": low,
                    "pm_close": close,
                    "traded": True,
                    "partial_session": False,
                }
            )
    return runner._coerce_jsonl_frame(
        pd.DataFrame(rows),
        runner.PARSED_PRICE_COLUMNS,
        label="test synthetic full31 predictor prices",
    )


def _compact12(full31: pd.DataFrame) -> pd.DataFrame:
    return runner._coerce_model_price_frame(
        full31, label="test synthetic compact12 predictor prices"
    )


def _patch_synthetic_calendar(
    monkeypatch: pytest.MonkeyPatch,
    full31: pd.DataFrame,
    *,
    target: str = TARGET,
) -> None:
    sessions = pd.DatetimeIndex(
        sorted(pd.to_datetime(full31["date"]).unique())
    ).append(pd.DatetimeIndex([pd.Timestamp(target)]))
    monkeypatch.setattr(runner, "load_registered_calendar", lambda: sessions)


def test_anchor_receipt_proves_full31_compact12_exact_consumer_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full31 = _synthetic_full31_prices()
    compact12 = _compact12(full31)
    _patch_synthetic_calendar(monkeypatch, full31)

    receipt = runner.build_compact_consumer_equivalence_receipt(
        full31,
        compact12,
        synthetic_target_session=TARGET,
    )

    assert runner.validate_compact_consumer_equivalence_receipt(receipt) == receipt
    assert receipt["consumer_projection_columns"] == list(
        runner.MODEL_PRICE_COLUMNS
    )
    assert receipt["full_g0_exact_digest"] == receipt[
        "compact_g0_exact_digest"
    ]
    digest = receipt["full_g0_exact_digest"]
    assert digest["columns"] == list(runner.G0_PANEL_COLUMNS)
    assert [item["column"] for item in digest["column_digests"]] == list(
        runner.G0_PANEL_COLUMNS
    )
    assert all(
        set(item)
        == {
            "column",
            "dtype",
            "null_bitmap_sha256",
            "value_bytes_sha256",
            "dtype_metadata",
        }
        for item in digest["column_digests"]
    )
    assert receipt["synthetic_target_session"] == TARGET
    assert receipt["latest_feature_source_session"] == str(
        pd.to_datetime(full31["date"]).max().date()
    )
    assert receipt["synthetic_target_row_count"] == 3
    assert receipt["synthetic_target_outcome_nonnull_count"] == 0
    assert receipt["synthetic_target_max_feature_source_date"] == receipt[
        "latest_feature_source_session"
    ]
    assert receipt["exact_columns_order_dtypes_nulls_ieee_strings_bools_equal"]
    assert receipt["receipt_sha256"] == runner.canonical_json_sha256(
        receipt, exclude_fields={"receipt_sha256"}
    )

    panel = runner.build_forward_c00_panel(full31, TARGET)
    target_rows = panel.loc[pd.to_datetime(panel["date"]).eq(pd.Timestamp(TARGET))]
    assert receipt["synthetic_target_exact_digest_sha256"] == (
        runner._exact_g0_frame_digest(target_rows)["exact_digest_sha256"]
    )


def _first_present_float_cell(panel: pd.DataFrame) -> tuple[int, str]:
    for column in runner.G0_PANEL_COLUMNS:
        if not pd.api.types.is_float_dtype(panel[column].dtype):
            continue
        present = panel[column].notna()
        if present.any():
            return int(np.flatnonzero(present.to_numpy())[0]), column
    raise AssertionError("synthetic G0 panel has no present float cell")


@pytest.mark.parametrize(
    "mutation",
    ["dtype", "null", "ieee_negative_zero", "string", "bool"],
)
def test_exact_g0_digest_detects_dtype_null_ieee_string_and_bool_mutations(
    mutation: str,
) -> None:
    full31 = _synthetic_full31_prices()
    panel = runner.build_forward_c00_panel(full31, TARGET)
    left = panel.copy(deep=True)
    right = panel.copy(deep=True)
    row, float_column = _first_present_float_cell(panel)

    if mutation == "dtype":
        right["name"] = right["name"].astype("string")
    elif mutation == "null":
        right.loc[right.index[row], float_column] = np.nan
    elif mutation == "ieee_negative_zero":
        left.loc[left.index[row], float_column] = 0.0
        right.loc[right.index[row], float_column] = -0.0
        assert np.signbit(right.loc[right.index[row], float_column])
    elif mutation == "string":
        right.loc[right.index[row], "name"] = "changed-consumer-string"
    elif mutation == "bool":
        right.loc[right.index[row], "common_score_eligible"] = not bool(
            right.loc[right.index[row], "common_score_eligible"]
        )
    else:  # pragma: no cover - parametrization is the complete mutation set.
        raise AssertionError(mutation)

    left_digest = runner._exact_g0_frame_digest(left)
    right_digest = runner._exact_g0_frame_digest(right)
    assert left_digest["columns"] == right_digest["columns"] == list(
        runner.G0_PANEL_COLUMNS
    )
    assert left_digest["exact_digest_sha256"] != right_digest[
        "exact_digest_sha256"
    ]


@pytest.mark.parametrize(
    "field",
    [
        "synthetic_target_exact_digest_sha256",
        "synthetic_target_row_count",
        "synthetic_target_outcome_nonnull_count",
        "synthetic_target_max_feature_source_date",
    ],
)
def test_anchor_receipt_binds_synthetic_target_hash_counts_and_feature_date(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    full31 = _synthetic_full31_prices()
    compact12 = _compact12(full31)
    _patch_synthetic_calendar(monkeypatch, full31)
    expected = runner.build_compact_consumer_equivalence_receipt(
        full31,
        compact12,
        synthetic_target_session=TARGET,
    )
    changed = copy.deepcopy(expected)
    replacements: dict[str, Any] = {
        "synthetic_target_exact_digest_sha256": "f" * 64,
        "synthetic_target_row_count": expected["synthetic_target_row_count"] + 1,
        "synthetic_target_outcome_nonnull_count": 1,
        "synthetic_target_max_feature_source_date": "2026-08-04",
    }
    changed[field] = replacements[field]

    with pytest.raises(runner.V18Error, match="target|outcome|self-hash"):
        runner.validate_compact_consumer_equivalence_receipt(changed)

    changed["receipt_sha256"] = runner.canonical_json_sha256(
        changed, exclude_fields={"receipt_sha256"}
    )
    if field == "synthetic_target_outcome_nonnull_count":
        with pytest.raises(runner.V18Error, match="contains an outcome"):
            runner.validate_compact_consumer_equivalence_receipt(changed)
    else:
        # A rehashed counterfeit is still unequal to the proof freshly derived
        # from raw31/compact12 and therefore fails the anchor/terminal exact
        # comparison that consumes this receipt.
        assert (
            runner.validate_compact_consumer_equivalence_receipt(changed)
            != runner.build_compact_consumer_equivalence_receipt(
                full31,
                compact12,
                synthetic_target_session=TARGET,
            )
        )


@pytest.mark.parametrize("dropped_column", ["am_open", "pm_close", "partial_session"])
def test_equivalence_proof_detects_a_consumer_dependency_on_dropped_full31_data(
    monkeypatch: pytest.MonkeyPatch,
    dropped_column: str,
) -> None:
    full31 = _synthetic_full31_prices()
    compact12 = _compact12(full31)
    _patch_synthetic_calendar(monkeypatch, full31)
    real_build = runner.build_forward_c00_panel

    def mutated_consumer(prices: pd.DataFrame, target: Any) -> pd.DataFrame:
        panel = real_build(prices, target)
        if dropped_column not in prices.columns:
            return panel
        row, float_column = _first_present_float_cell(panel)
        if dropped_column == "partial_session":
            signal = float(prices[dropped_column].astype(bool).sum() + 1)
        else:
            signal = float(pd.to_numeric(prices[dropped_column]).sum())
        panel.loc[panel.index[row], float_column] = (
            float(panel.loc[panel.index[row], float_column])
            + signal * np.finfo("float64").eps
        )
        return panel

    monkeypatch.setattr(runner, "build_forward_c00_panel", mutated_consumer)

    with pytest.raises(
        runner.V18Error,
        match="full31 and compact12 G0 consumers are not bit-exact",
    ):
        runner.build_compact_consumer_equivalence_receipt(
            full31,
            compact12,
            synthetic_target_session=TARGET,
        )


def _minimal_compact_consumer_receipt() -> dict[str, Any]:
    digest: dict[str, Any] = {
        "columns": list(runner.G0_PANEL_COLUMNS),
        "row_count": 1,
        "row_identity_sha256": "1" * 64,
        "column_digests": [],
    }
    digest["exact_digest_sha256"] = runner.canonical_json_sha256(
        digest, exclude_fields={"exact_digest_sha256"}
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "consumer_projection_columns": list(runner.MODEL_PRICE_COLUMNS),
        "consumer_projection_columns_sha256": runner.canonical_json_sha256(
            list(runner.MODEL_PRICE_COLUMNS)
        ),
        "consumer_union_contract_sha256": runner.canonical_json_sha256(
            {
                key: sorted(columns)
                for key, columns in sorted(
                    runner.MODEL_PRICE_REQUIRED_BY_CONSUMERS.items()
                )
            }
        ),
        "raw_date_code_identity_sha256": "2" * 64,
        "historical_session_registry_sha256": "3" * 64,
        "synthetic_target_session": "2026-08-03",
        "latest_feature_source_session": "2026-07-31",
        "full_g0_exact_digest": digest,
        "compact_g0_exact_digest": copy.deepcopy(digest),
        "synthetic_target_exact_digest_sha256": "4" * 64,
        "synthetic_target_row_count": 1,
        "synthetic_target_outcome_nonnull_count": 0,
        "synthetic_target_max_feature_source_date": "2026-07-31",
        "exact_columns_order_dtypes_nulls_ieee_strings_bools_equal": True,
        "canonical_json_contract": runner.CANONICAL_JSON_CONTRACT,
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = runner.canonical_json_sha256(
        receipt, exclude_fields={"receipt_sha256"}
    )
    return receipt


def _synthetic_cache_anchor_for_real_validator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    shard_sealed_at: str,
) -> tuple[dict[str, Any], list[str], list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]]:
    latest = pd.Timestamp("2026-07-31")
    source_file = "stq_20260731.pdf"
    raw_record = {
        "file": source_file,
        "url": (
            "https://www.jpx.co.jp/markets/statistics-equities/daily/"
            f"{source_file}"
        ),
        "object_key": f"{runner.PREDICTOR_OBJECT_PREFIX}{source_file}",
        "byte_count": 101,
        "sha256": "5" * 64,
    }
    shard_manifest_key = (
        f"{runner.PREDICTOR_SHARD_OBJECT_PREFIX}synthetic.manifest.json"
    )
    binding = {
        "raw_object_key": raw_record["object_key"],
        "raw_file": source_file,
        "raw_url": raw_record["url"],
        "raw_byte_count": raw_record["byte_count"],
        "raw_sha256": raw_record["sha256"],
        "shard_manifest_object_key": shard_manifest_key,
        "shard_manifest_byte_count": 17,
        "shard_manifest_file_sha256": "6" * 64,
        "shard_manifest_sha256": "7" * 64,
        "shard_object_key": (
            f"{runner.PREDICTOR_SHARD_OBJECT_PREFIX}synthetic.jsonl"
        ),
        "shard_byte_count": 103,
        "shard_sha256": "8" * 64,
        "parsed_row_count": 1,
        "parsed_semantic_sha256": "9" * 64,
        "pdftotext_text_byte_count": 107,
        "pdftotext_text_sha256": "a" * 64,
        "parser_report_sha256": "b" * 64,
    }
    raw_records = [raw_record]
    bindings = [binding]
    raw_set_sha256 = runner.canonical_json_sha256(raw_records)
    shard_set_sha256 = runner._parsed_shard_set_sha256(bindings)
    _, snapshot_key, anchor_manifest_key = runner._cache_anchor_identity(
        latest_source_session=latest,
        raw_source_set_sha256=raw_set_sha256,
        ordered_shard_set_sha256=shard_set_sha256,
    )
    model_manifest_key = (
        f"{runner.MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX}"
        "2026-08/synthetic.manifest.json"
    )
    model_manifest_payload = runner._json_file_bytes({"synthetic": True})
    model_snapshot = {
        "created_at": "2026-08-01T11:00:00+09:00",
        "sealed_at": "2026-08-01T11:05:00+09:00",
        "data_object_key": (
            f"{runner.MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX}"
            "2026-08/synthetic.csv"
        ),
        "data_byte_count": 109,
        "data_sha256": "c" * 64,
        "model_price_semantic_sha256": "d" * 64,
        "snapshot_manifest_sha256": "e" * 64,
    }
    compact_receipt = _minimal_compact_consumer_receipt()
    anchor: dict[str, Any] = {
        "schema_version": 1,
        "cache_contract_id": runner.PREDICTOR_CACHE_CONTRACT_ID,
        "latest_source_session": str(latest.date()),
        "raw_source_set_sha256": raw_set_sha256,
        "raw_source_count": 1,
        "raw_sources": raw_records,
        "ordered_shard_set_sha256": shard_set_sha256,
        "ordered_shard_count": 1,
        "parsed_shards": bindings,
        "columns": list(runner.PARSED_PRICE_COLUMNS),
        "columns_sha256": runner.canonical_json_sha256(
            list(runner.PARSED_PRICE_COLUMNS)
        ),
        "cumulative_snapshot_object_key": snapshot_key,
        "cumulative_snapshot_byte_count": 113,
        "cumulative_snapshot_file_sha256": "f" * 64,
        "cumulative_snapshot_semantic_sha256": "0" * 64,
        "model_price_snapshot_target_month": "2026-08",
        "model_price_snapshot_object_key": model_snapshot["data_object_key"],
        "model_price_snapshot_byte_count": model_snapshot["data_byte_count"],
        "model_price_snapshot_file_sha256": model_snapshot["data_sha256"],
        "model_price_snapshot_semantic_sha256": model_snapshot[
            "model_price_semantic_sha256"
        ],
        "model_price_snapshot_manifest_object_key": model_manifest_key,
        "model_price_snapshot_manifest_byte_count": len(model_manifest_payload),
        "model_price_snapshot_manifest_file_sha256": hashlib.sha256(
            model_manifest_payload
        ).hexdigest(),
        "model_price_snapshot_manifest_sha256": model_snapshot[
            "snapshot_manifest_sha256"
        ],
        "snapshot_manifest_object_key": anchor_manifest_key,
        "snapshot_manifest_sha256": "",
        "direct_reparse_started_at": "2026-08-01T12:00:00+09:00",
        "direct_reparse_completed_at": "2026-08-01T12:10:00+09:00",
        "direct_clean_room_verification_receipt_sha256": "",
        "compact_consumer_equivalence_receipt": compact_receipt,
        "runtime_lock_sha256": runner.RUNTIME_LOCK_SHA256,
        "runtime_lock_verified_at": "2026-08-01T09:00:00+09:00",
        "created_at": "2026-08-01T13:00:00+09:00",
        "sealed_at": "2026-08-01T13:05:00+09:00",
        "protocol_sha256": runner.PROTOCOL_SHA256,
        "runner_sha256": runner.sha256_file(runner.__file__),
        "parser_sha256": runner.JPX_PARSER_SHA256,
        "canonical_jsonl_contract": runner.PARSED_SHARD_JSONL_CONTRACT,
        "verified_at": "2026-08-01T12:10:00+09:00",
        "anchor_manifest_sha256": "",
    }
    direct_receipt = {
        "cache_contract_id": runner.PREDICTOR_CACHE_CONTRACT_ID,
        "raw_source_set_sha256": raw_set_sha256,
        "ordered_shard_set_sha256": shard_set_sha256,
        "direct_reparse_started_at": anchor["direct_reparse_started_at"],
        "direct_reparse_completed_at": anchor["direct_reparse_completed_at"],
        "direct_parsed_semantic_sha256": anchor[
            "cumulative_snapshot_semantic_sha256"
        ],
        "snapshot_semantic_sha256": anchor[
            "cumulative_snapshot_semantic_sha256"
        ],
        "model_price_snapshot_semantic_sha256": anchor[
            "model_price_snapshot_semantic_sha256"
        ],
        "exact_frame_and_dtype_equal": True,
        "runtime_lock_sha256": runner.RUNTIME_LOCK_SHA256,
    }
    anchor["direct_clean_room_verification_receipt_sha256"] = (
        runner.canonical_json_sha256(direct_receipt)
    )
    anchor_hash = runner.canonical_json_sha256(
        anchor,
        exclude_fields={"anchor_manifest_sha256", "snapshot_manifest_sha256"},
    )
    anchor["snapshot_manifest_sha256"] = anchor_hash
    anchor["anchor_manifest_sha256"] = anchor_hash

    read_keys: list[str] = []
    bound_sets: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    monkeypatch.setattr(
        runner,
        "_expected_predictor_files",
        lambda observed_latest: (
            [source_file],
            {source_file: "daily"},
        ),
    )
    monkeypatch.setattr(
        runner,
        "_latest_registered_source_before_month",
        lambda target_month: latest,
    )

    def validate_bound(
        observed_raw: list[dict[str, Any]],
        observed_bindings: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        bound_sets.append(
            ([dict(item) for item in observed_raw], [dict(item) for item in observed_bindings])
        )
        return []

    monkeypatch.setattr(
        runner, "_validate_bound_predictor_shard_metadata", validate_bound
    )

    def read_manifest(
        root: str | Path,
        key: str,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], bytes]:
        read_keys.append(key)
        if key == shard_manifest_key:
            return (
                {
                    "chronology_class": "anchor",
                    "raw_received_at": None,
                    "sealed_at": shard_sealed_at,
                },
                b"{}\n",
            )
        assert key == model_manifest_key
        return {"synthetic": True}, model_manifest_payload

    monkeypatch.setattr(runner, "_read_external_canonical_json", read_manifest)

    def object_metadata(
        root: str | Path,
        key: str,
        **kwargs: Any,
    ) -> tuple[int, str]:
        assert key == snapshot_key
        return (
            anchor["cumulative_snapshot_byte_count"],
            anchor["cumulative_snapshot_file_sha256"],
        )

    monkeypatch.setattr(runner, "_external_object_metadata", object_metadata)

    def validate_model_snapshot(
        manifest: Mapping[str, Any],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], pd.DataFrame, str, bytes]:
        assert kwargs["expected_target_month"] == pd.Period("2026-08", freq="M")
        assert kwargs["expected_latest_source_session"] == latest
        assert kwargs["expected_raw_source_set_sha256"] == raw_set_sha256
        assert kwargs["expected_parsed_shard_set_sha256"] == shard_set_sha256
        assert kwargs["decode_data"] is False
        return dict(model_snapshot), pd.DataFrame(), model_manifest_key, model_manifest_payload

    monkeypatch.setattr(
        runner, "_validate_model_price_snapshot_manifest", validate_model_snapshot
    )
    return anchor, read_keys, bound_sets


def test_real_cache_anchor_validator_accepts_one_sealed_bound_shard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    anchor, read_keys, bound_sets = _synthetic_cache_anchor_for_real_validator(
        monkeypatch,
        shard_sealed_at="2026-08-01T10:00:00+09:00",
    )

    validated, summary, manifest_payload = runner.validate_predictor_cache_anchor(
        anchor,
        predictor_raw_store_root=tmp_path / "predictor-raw",
        predictor_derived_store_root=tmp_path / "predictor-derived",
        require_direct_clean_room_reparse=False,
    )

    assert validated == anchor
    assert summary["ordered_shard_count"] == 1
    assert manifest_payload == runner._json_file_bytes(anchor)
    assert bound_sets == [(anchor["raw_sources"], anchor["parsed_shards"])]
    assert read_keys == [
        anchor["parsed_shards"][0]["shard_manifest_object_key"],
        anchor["model_price_snapshot_manifest_object_key"],
    ]


def test_real_cache_anchor_validator_rejects_model_snapshot_backdated_before_shard_seal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    anchor, read_keys, bound_sets = _synthetic_cache_anchor_for_real_validator(
        monkeypatch,
        # This remains before direct_reparse_started_at, so rejection must be
        # the compact-snapshot-vs-shard seal edge and not the earlier edge.
        shard_sealed_at="2026-08-01T11:30:00+09:00",
    )

    with pytest.raises(
        runner.V18Error,
        match="cache anchor compact snapshot timestamp DAG is invalid",
    ):
        runner.validate_predictor_cache_anchor(
            anchor,
            predictor_raw_store_root=tmp_path / "predictor-raw",
            predictor_derived_store_root=tmp_path / "predictor-derived",
            require_direct_clean_room_reparse=False,
        )

    assert bound_sets == [(anchor["raw_sources"], anchor["parsed_shards"])]
    assert read_keys == [
        anchor["parsed_shards"][0]["shard_manifest_object_key"],
        anchor["model_price_snapshot_manifest_object_key"],
    ]
