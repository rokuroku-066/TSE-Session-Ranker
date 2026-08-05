from __future__ import annotations

from datetime import datetime
import errno
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import pandas as pd
import pytest

from research import model_v18_shoulder_state_runner as runner


OLD_RUNTIME_TIMESTAMP = "2026-08-05T08:00:00+09:00"
NEW_RUNTIME_TIMESTAMP = "2026-08-05T08:01:00+09:00"


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def _tree_identity(root: Path) -> tuple[tuple[Any, ...], ...]:
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
                path.read_bytes() if stat.S_ISREG(metadata.st_mode) else None,
            )
        )
    return tuple(rows)


def _publish_crash_pair(final: Path, payload: bytes, *, mode: int = 0o600) -> Path:
    stage = final.with_name(f".{final.name}.staging")
    stage.write_bytes(payload)
    stage.chmod(mode)
    os.link(stage, final)
    return stage


def _terminal_cli_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], list[str]]:
    result_parent = _private_directory(tmp_path / "terminal-result")
    result_path = result_parent / "result.json"
    paths = {
        "RESULT_OUTPUT": result_path,
        "DECISION_LEDGER": tmp_path / "decisions.jsonl",
        "OUTCOME_LEDGER": tmp_path / "outcomes.jsonl",
        "COMPLETED_MONTH_LEDGER": tmp_path / "completed.jsonl",
        "SCORE_OUTPUT": tmp_path / "scores.csv",
        "PICKS_OUTPUT": tmp_path / "picks.csv",
        "SOURCE_MANIFEST_DIR": tmp_path / "sources",
        "OUTCOME_MANIFEST_DIR": tmp_path / "outcomes",
    }
    for name, path in paths.items():
        monkeypatch.setattr(runner, name, path)
    decisions = [{"session_date": "2026-08-05"}]
    outcomes = [{"session_date": "2026-08-05"}]
    completed = [{"completed_month": "2026-08"}]
    events: list[str] = []

    monkeypatch.setattr(
        runner,
        "validate_runtime_lock",
        lambda **kwargs: ({"lock_id": "result-retry"}, runner.RUNTIME_LOCK_SHA256),
    )
    monkeypatch.setattr(
        runner, "validate_protocol", lambda *args, **kwargs: ({}, "a" * 64)
    )
    monkeypatch.setattr(
        runner,
        "validate_canonical_activation_artifacts",
        lambda: ({}, "b" * 64, {}, "c" * 64),
    )

    def load_authority(path: str | Path, **kwargs: Any) -> list[dict[str, Any]]:
        observed = Path(path)
        if observed == paths["DECISION_LEDGER"]:
            return decisions
        if observed == paths["OUTCOME_LEDGER"]:
            return outcomes
        if observed == paths["COMPLETED_MONTH_LEDGER"]:
            return completed
        raise AssertionError(f"unexpected record authority: {observed}")

    monkeypatch.setattr(runner, "load_jsonl_record_authority", load_authority)
    monkeypatch.setattr(
        runner,
        "load_registered_calendar",
        lambda: pd.DatetimeIndex(["2026-08-05"]),
    )
    monkeypatch.setattr(
        runner,
        "deterministic_terminal_session",
        lambda first, scheduled: scheduled[-1],
    )

    def predictor_gate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("predictor-gate")
        return {}

    def checkpoint_gate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("checkpoint-gate")
        return {}

    def performance(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("performance")
        return {"status": "forward_rejected_candidate"}

    monkeypatch.setattr(runner, "validate_predictor_evidence", predictor_gate)
    monkeypatch.setattr(runner, "validate_checkpoint_evidence", checkpoint_gate)
    monkeypatch.setattr(
        runner, "_validate_deferred_state_evidence", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "_preflight_terminal_completed_month_ledger",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(runner, "evaluate", performance)

    picks = pd.DataFrame({"slot": [1]})
    pick_payload = picks.to_csv(index=False, lineterminator="\n").encode()
    monkeypatch.setattr(runner, "materialize_picks", lambda *args: picks)
    monkeypatch.setattr(
        runner,
        "_write_csv_exclusive",
        lambda frame, path: events.append("picks-exact-write"),
    )
    monkeypatch.setattr(
        runner,
        "_plain_file_bytes",
        lambda path, **kwargs: pick_payload,
    )

    retained = {
        "status": "forward_rejected_candidate",
        "result_kind": "terminal",
        "runtime": {"runtime_lock_verified_at": OLD_RUNTIME_TIMESTAMP},
    }

    def rebuild_result(
        *args: Any,
        runtime_lock_verified_at: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        events.append("result-rebuild")
        timestamp = (
            NEW_RUNTIME_TIMESTAMP
            if runtime_lock_verified_at is None
            else str(runtime_lock_verified_at)
        )
        return {
            "status": "forward_rejected_candidate",
            "result_kind": "terminal",
            "runtime": {"runtime_lock_verified_at": timestamp},
        }

    monkeypatch.setattr(runner, "build_result", rebuild_result)
    return result_path, retained, events


@pytest.mark.parametrize("publication", ("completed", "crash-linked"))
def test_terminal_result_exact_retry_rebuilds_after_gates_and_is_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    publication: str,
) -> None:
    result_path, retained, events = _terminal_cli_harness(monkeypatch, tmp_path)
    payload = runner._json_file_bytes(retained)
    if publication == "completed":
        result_path.write_bytes(payload)
        result_path.chmod(0o600)
        stage = result_path.with_name(f".{result_path.name}.staging")
    else:
        stage = _publish_crash_pair(result_path, payload)
    before_inode = result_path.stat().st_ino
    real_read = runner._read_local_authority_bytes

    def observed_read(path: str | Path, *, label: str) -> bytes:
        if Path(path) == result_path:
            events.append("result-read")
        return real_read(path, label=label)

    monkeypatch.setattr(runner, "_read_local_authority_bytes", observed_read)
    monkeypatch.setattr(
        runner,
        "write_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("exact terminal retry must not rewrite its result")
        ),
    )

    assert (
        runner.main(
            [
                "evaluate",
                "--predictor-raw-store-root",
                str(tmp_path / "predictor-raw"),
                "--predictor-derived-store-root",
                str(tmp_path / "predictor-derived"),
                "--outcome-raw-store-root",
                str(tmp_path / "outcome-raw"),
                "--checkpoint-core-store-root",
                str(tmp_path / "checkpoint-core"),
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out) == retained
    assert events.index("result-read") > events.index("performance")
    assert events[-1] == "result-rebuild"
    assert result_path.read_bytes() == payload
    assert result_path.stat().st_ino == before_inode
    assert result_path.stat().st_nlink == 1
    assert not os.path.lexists(stage)


@pytest.mark.parametrize("publication", ("completed", "crash-linked"))
def test_evaluate_rejects_retained_abort_before_any_semantic_or_performance_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    publication: str,
) -> None:
    result_parent = _private_directory(tmp_path / "aborted-evaluate-result")
    result_path = result_parent / "result.json"
    retained_abort = {
        "models": {"forbidden_performance": 123.0},
        "runtime": {"runtime_lock_verified_at": OLD_RUNTIME_TIMESTAMP},
        "status": "aborted_integrity_failure",
    }
    payload = runner._json_file_bytes(retained_abort)
    if publication == "completed":
        result_path.write_bytes(payload)
        result_path.chmod(0o600)
        stage = result_path.with_name(f".{result_path.name}.staging")
    else:
        stage = _publish_crash_pair(result_path, payload)
    monkeypatch.setattr(runner, "RESULT_OUTPUT", result_path)

    forbidden = [
        "validate_runtime_lock",
        "validate_protocol",
        "validate_canonical_activation_artifacts",
        "load_jsonl_record_authority",
        "load_registered_calendar",
        "validate_predictor_evidence",
        "validate_checkpoint_evidence",
        "_validate_deferred_state_evidence",
        "evaluate",
        "materialize_picks",
        "_write_csv_exclusive",
        "write_json",
    ]
    for name in forbidden:
        monkeypatch.setattr(
            runner,
            name,
            lambda *args, _name=name, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"retained abort opened or mutated {_name}")
            ),
        )
    monkeypatch.setattr(
        runner,
        "_read_local_authority_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("retained abort body must remain unopened")
        ),
    )

    before = _tree_identity(result_parent)
    with pytest.raises(runner.V18Error, match="irreversibly terminated"):
        runner.main(
            [
                "evaluate",
                "--predictor-raw-store-root",
                str(tmp_path / "predictor-raw"),
                "--predictor-derived-store-root",
                str(tmp_path / "predictor-derived"),
                "--outcome-raw-store-root",
                str(tmp_path / "outcome-raw"),
                "--checkpoint-core-store-root",
                str(tmp_path / "checkpoint-core"),
            ]
        )
    assert _tree_identity(result_parent) == before
    if publication == "crash-linked":
        assert stage.stat().st_ino == result_path.stat().st_ino
        assert result_path.stat().st_nlink == 2
    else:
        assert not os.path.lexists(stage)


def _patch_abort_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, str]]:
    result_parent = _private_directory(tmp_path / "abort-result")
    result_path = result_parent / "result.json"
    context_path = tmp_path / "activation-context.json"
    context = {
        "activation_payload_sha256": "a" * 64,
        "activation_receipt_sha256": "b" * 64,
        "activation_receipt_commit_sha": "c" * 40,
    }
    context_path.write_bytes(runner._json_file_bytes(context))
    context_path.chmod(0o600)
    monkeypatch.setattr(runner, "RESULT_OUTPUT", result_path)
    monkeypatch.setattr(runner, "ACTIVATION_CONTEXT", context_path)
    for name in (
        "DECISION_LEDGER",
        "OUTCOME_LEDGER",
        "COMPLETED_MONTH_LEDGER",
        "SCORE_OUTPUT",
        "PICKS_OUTPUT",
    ):
        monkeypatch.setattr(runner, name, tmp_path / f"{name.lower()}.missing")
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
        datetime.fromisoformat(OLD_RUNTIME_TIMESTAMP),
    )
    monkeypatch.setattr(
        runner, "validate_protocol", lambda *args, **kwargs: (protocol, "d" * 64)
    )
    monkeypatch.setattr(
        runner,
        "validate_runtime_lock",
        lambda **kwargs: (
            {"lock_id": "abort-retry", "runtime_lock_self_sha256": "e" * 64},
            runner.RUNTIME_LOCK_SHA256,
        ),
    )
    monkeypatch.setattr(runner, "validate_activation_context", lambda value: dict(value))
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
    monkeypatch.setattr(runner, "_stream_plain_file_sha256", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner, "_raw_source_provenance_envelope", lambda: {"kind": "opaque"}
    )
    return result_path, context_path, context


@pytest.mark.parametrize("publication", ("completed", "crash-linked"))
def test_abort_result_exact_retry_is_outcome_blind_and_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    publication: str,
) -> None:
    result_path, _, context = _patch_abort_contract(monkeypatch, tmp_path)
    retained = runner.build_integrity_abort_result(
        failure_reason="source_integrity_failure",
        integrity_stage="source_ingestion",
        activation_context=context,
    )
    payload = runner._json_file_bytes(retained)
    if publication == "completed":
        result_path.write_bytes(payload)
        result_path.chmod(0o600)
        stage = result_path.with_name(f".{result_path.name}.staging")
    else:
        stage = _publish_crash_pair(result_path, payload)
    inode = result_path.stat().st_ino
    monkeypatch.setattr(
        runner,
        "_STRICT_RUNTIME_VERIFIED_AT",
        datetime.fromisoformat(NEW_RUNTIME_TIMESTAMP),
    )
    for forbidden in ("evaluate", "materialize_picks", "pair_history_from_ledgers"):
        monkeypatch.setattr(
            runner,
            forbidden,
            lambda *args, _name=forbidden, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"abort retry opened outcome helper: {_name}")
            ),
        )

    assert (
        runner.main(
            [
                "abort",
                "--failure-reason",
                "source_integrity_failure",
                "--integrity-stage",
                "source_ingestion",
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out) == json.loads(payload)
    assert result_path.read_bytes() == payload
    assert result_path.stat().st_ino == inode
    assert result_path.stat().st_nlink == 1
    assert not os.path.lexists(stage)


@pytest.mark.parametrize(
    "fault",
    ("link", "fsync-after-link", "unlink", "fsync-after-unlink"),
)
@pytest.mark.parametrize(
    "artifact",
    (
        "result",
        "picks",
        "activation-payload",
        "activation-receipt",
        "activation-context",
    ),
)
def test_result_picks_and_preactivation_writers_recover_publication_faults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault: str,
    artifact: str,
) -> None:
    parent = _private_directory(tmp_path / f"{artifact}-{fault}")
    file_names = {
        "result": "result.json",
        "picks": "picks.csv",
        "activation-payload": "activation-payload.json",
        "activation-receipt": "activation-receipt.json",
        "activation-context": "activation-context.json",
    }
    constants = {
        "result": "RESULT_OUTPUT",
        "activation-payload": "ACTIVATION_PAYLOAD",
        "activation-receipt": "ACTIVATION_RECEIPT",
        "activation-context": "ACTIVATION_CONTEXT",
    }
    result_path = parent / file_names[artifact]
    if artifact in constants:
        monkeypatch.setattr(runner, constants[artifact], result_path)
    result = {
        "status": "forward_rejected_candidate",
        "runtime": {"runtime_lock_verified_at": OLD_RUNTIME_TIMESTAMP},
    }
    picks = pd.DataFrame({"slot": [1], "code": ["1001"]})
    payload = (
        runner._json_file_bytes(result)
        if artifact != "picks"
        else picks.to_csv(index=False, lineterminator="\n").encode()
    )
    stage = result_path.with_name(f".{result_path.name}.staging")
    real_link = os.link
    real_fsync = os.fsync
    real_unlink = os.unlink
    state = {"linked": False, "unlinked": False, "failed": False}

    def injected_link(*args: Any, **kwargs: Any) -> None:
        if fault == "link" and not state["failed"]:
            state["failed"] = True
            raise OSError(errno.EIO, "injected result link failure")
        real_link(*args, **kwargs)
        state["linked"] = True

    def injected_unlink(*args: Any, **kwargs: Any) -> None:
        if fault == "unlink" and state["linked"] and not state["failed"]:
            state["failed"] = True
            raise OSError(errno.EIO, "injected result unlink failure")
        real_unlink(*args, **kwargs)
        if state["linked"]:
            state["unlinked"] = True

    def injected_fsync(descriptor: int) -> None:
        directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        if (
            fault == "fsync-after-link"
            and directory
            and state["linked"]
            and not state["unlinked"]
            and not state["failed"]
        ):
            state["failed"] = True
            raise OSError(errno.EIO, "injected result link fsync failure")
        if (
            fault == "fsync-after-unlink"
            and directory
            and state["unlinked"]
            and not state["failed"]
        ):
            state["failed"] = True
            raise OSError(errno.EIO, "injected result unlink fsync failure")
        real_fsync(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(runner.os, "link", injected_link)
        patch.setattr(runner.os, "unlink", injected_unlink)
        patch.setattr(runner.os, "fsync", injected_fsync)
        with pytest.raises(OSError, match="injected result"):
            if artifact != "picks":
                runner.write_json(result, result_path, exclusive=True)
            else:
                runner._write_csv_exclusive(picks, result_path)

    assert state["failed"] is True
    if artifact != "picks":
        runner.write_json(result, result_path, exclusive=True)
    else:
        runner._write_csv_exclusive(picks, result_path)
    assert result_path.read_bytes() == payload
    assert result_path.stat().st_nlink == 1
    assert stat.S_IMODE(result_path.stat().st_mode) == 0o600
    assert not os.path.lexists(stage)


@pytest.mark.parametrize("conflict", ("distinct-stage", "wrong-mode", "different-bytes"))
@pytest.mark.parametrize(
    "artifact",
    (
        "result",
        "picks",
        "activation-payload",
        "activation-receipt",
        "activation-context",
    ),
)
def test_result_picks_and_preactivation_writers_reject_hostile_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    conflict: str,
    artifact: str,
) -> None:
    parent = _private_directory(tmp_path / f"hostile-{artifact}-{conflict}")
    file_names = {
        "result": "result.json",
        "picks": "picks.csv",
        "activation-payload": "activation-payload.json",
        "activation-receipt": "activation-receipt.json",
        "activation-context": "activation-context.json",
    }
    constants = {
        "result": "RESULT_OUTPUT",
        "activation-payload": "ACTIVATION_PAYLOAD",
        "activation-receipt": "ACTIVATION_RECEIPT",
        "activation-context": "ACTIVATION_CONTEXT",
    }
    result_path = parent / file_names[artifact]
    if artifact in constants:
        monkeypatch.setattr(runner, constants[artifact], result_path)
    retained = {"status": "retained", "value": 1}
    candidate = retained if conflict != "different-bytes" else {
        "status": "retained",
        "value": 2,
    }
    retained_picks = pd.DataFrame({"slot": [1], "code": ["1001"]})
    candidate_picks = (
        retained_picks
        if conflict != "different-bytes"
        else pd.DataFrame({"slot": [1], "code": ["1002"]})
    )
    payload = (
        runner._json_file_bytes(retained)
        if artifact != "picks"
        else retained_picks.to_csv(index=False, lineterminator="\n").encode()
    )
    stage = result_path.with_name(f".{result_path.name}.staging")
    if conflict == "distinct-stage":
        result_path.write_bytes(payload)
        result_path.chmod(0o600)
        stage.write_bytes(payload)
        stage.chmod(0o600)
    else:
        _publish_crash_pair(
            result_path,
            payload,
            mode=0o644 if conflict == "wrong-mode" else 0o600,
        )
    before = _tree_identity(parent)

    with pytest.raises(runner.V18Error):
        if artifact != "picks":
            runner.write_json(candidate, result_path, exclusive=True)
        else:
            runner._write_csv_exclusive(candidate_picks, result_path)

    assert _tree_identity(parent) == before


def test_result_presence_preflight_is_stat_only_for_exact_crash_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "presence")
    result_path = parent / "result.json"
    _publish_crash_pair(result_path, b"opaque result bytes\n")
    before = _tree_identity(parent)

    with monkeypatch.context() as patch:
        patch.setattr(
            runner.os,
            "open",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("presence preflight opened result bytes")
            ),
        )
        assert (
            runner._local_authority_presence_state(
                result_path, label="canonical result"
            )
            == "crash_linked"
        )

    assert _tree_identity(parent) == before
