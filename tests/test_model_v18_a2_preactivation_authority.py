from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Any

import pytest

from research import model_v18_shoulder_state_runner as runner


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


def _publish(final: Path, payload: bytes, publication: str) -> Path:
    stage = final.with_name(f".{final.name}.staging")
    if publication == "completed":
        final.write_bytes(payload)
        final.chmod(0o600)
    else:
        stage.write_bytes(payload)
        stage.chmod(0o600)
        os.link(stage, final)
    return stage


def _forbidden_network(name: str) -> Any:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"exact preactivation retry invoked {name}")

    return forbidden


def _materialize_additional_test_tree(root: Path) -> list[dict[str, str]]:
    for index, relative in enumerate(runner.ADDITIONAL_TEST_ARTIFACT_PATHS):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"additional test {index}: {relative}\n", encoding="utf-8")
    return runner._additional_test_artifacts_from_worktree()


def test_protocol_fixes_additional_test_paths_without_cyclic_hashes() -> None:
    protocol = {
        "activation": {
            "preregistration_commit": {
                "additional_test_artifact_paths": list(
                    runner.ADDITIONAL_TEST_ARTIFACT_PATHS
                )
            },
            "payload": {
                "required_fields": ["additional_test_artifacts"],
                "fixed_values": {},
            },
        }
    }
    assert runner._protocol_additional_test_artifact_paths(protocol) == (
        runner.ADDITIONAL_TEST_ARTIFACT_PATHS
    )
    reordered = json.loads(json.dumps(protocol))
    reordered["activation"]["preregistration_commit"][
        "additional_test_artifact_paths"
    ][0:2] = reversed(
        reordered["activation"]["preregistration_commit"][
            "additional_test_artifact_paths"
        ][0:2]
    )
    with pytest.raises(runner.V18Error, match="path registry changed"):
        runner._protocol_additional_test_artifact_paths(reordered)
    cyclic = json.loads(json.dumps(protocol))
    cyclic["activation"]["payload"]["fixed_values"][
        "additional_test_artifacts"
    ] = []
    with pytest.raises(runner.V18Error, match="cyclic additional test hashes"):
        runner._protocol_additional_test_artifact_paths(cyclic)


def test_additional_test_artifacts_bind_worktree_order_and_prereg_git_blobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    records = _materialize_additional_test_tree(tmp_path)
    assert tuple(item["path"] for item in records) == (
        runner.ADDITIONAL_TEST_ARTIFACT_PATHS
    )
    assert runner._validate_additional_test_artifacts(
        records, verify_worktree=True
    ) == tuple(records)

    monkeypatch.setattr(
        runner,
        "_git_file_bytes",
        lambda commit, relative: (tmp_path / relative).read_bytes(),
    )
    runner._validate_additional_test_git_blobs(records, commit_sha="a" * 40)

    changed_path = tmp_path / runner.ADDITIONAL_TEST_ARTIFACT_PATHS[-1]
    changed_path.write_text("mutated after preregistration\n", encoding="utf-8")
    with pytest.raises(runner.V18Error, match="working-tree bytes changed"):
        runner._validate_additional_test_artifacts(records, verify_worktree=True)
    with pytest.raises(runner.V18Error, match="additional test changed"):
        runner._validate_additional_test_git_blobs(records, commit_sha="a" * 40)

    reordered = list(records)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(runner.V18Error, match="path/hash changed"):
        runner._validate_additional_test_artifacts(
            reordered, verify_worktree=False
        )
    with pytest.raises(runner.V18Error, match="construction path registry changed"):
        runner._additional_test_artifacts_from_worktree(
            tuple(reversed(runner.ADDITIONAL_TEST_ARTIFACT_PATHS))
        )


def test_additional_test_paths_join_every_preregistered_immutability_loop() -> None:
    payload = {
        "protocol_path": "research/protocol.json",
        "hypothesis_path": "research/hypothesis.md",
        "runtime_lock_path": "research/runtime.json",
        "runner_path": "research/runner.py",
        "audit_path": "research/audit.py",
        "rehearsal_path": "research/rehearsal.py",
        "tests_path": "tests/test_model_v18_shoulder_state.py",
        "iteration_report_path": "research/iteration-report.md",
        "validation_report_path": "VALIDATION.md",
        "session_calendar_path": "research/calendar.csv",
        "workflow_path": ".github/workflows/tests.yml",
        "additional_test_artifacts": [
            {"path": path, "sha256": f"{index + 1:064x}"}
            for index, path in enumerate(runner.ADDITIONAL_TEST_ARTIFACT_PATHS)
        ],
    }
    observed = runner._preregistered_artifact_paths(payload)
    assert observed[-len(runner.ADDITIONAL_TEST_ARTIFACT_PATHS) :] == (
        runner.ADDITIONAL_TEST_ARTIFACT_PATHS
    )
    assert len(observed) == 11 + len(runner.ADDITIONAL_TEST_ARTIFACT_PATHS)


@pytest.mark.parametrize("publication", ("completed", "crash-linked"))
def test_activation_payload_exact_retry_never_refetches_network_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    publication: str,
) -> None:
    parent = _private_directory(tmp_path / "payload")
    output = parent / "activation-payload.json"
    monkeypatch.setattr(runner, "ACTIVATION_PAYLOAD", output)
    commit = "a" * 40
    anchor_key = "model_v18_shoulder_state/cache-anchor/manifest.json"
    retained = {
        "preregistration_commit_sha": commit,
        "predictor_cache_anchor": {"snapshot_manifest_object_key": anchor_key},
    }
    payload = runner._json_file_bytes(retained)
    stage = _publish(output, payload, publication)
    inode = output.stat().st_ino
    monkeypatch.setattr(
        runner,
        "_protocol_activation",
        lambda: ({}, {"payload": {"fixed_values": {}}}),
    )
    monkeypatch.setattr(runner, "_git_require_commit", lambda value, label: value)
    monkeypatch.setattr(
        runner,
        "validate_activation_payload",
        lambda value, *args, **kwargs: (dict(value), "b" * 64),
    )
    for name in (
        "_github_observation",
        "_github_workflow_run_observation",
        "_read_external_canonical_json",
    ):
        monkeypatch.setattr(runner, name, _forbidden_network(name))

    observed = runner.create_activation_payload(
        preregistration_commit_sha=commit,
        predictor_raw_store_root=tmp_path / "raw",
        predictor_derived_store_root=tmp_path / "derived",
        predictor_cache_anchor_manifest_object_key=anchor_key,
        output=output,
    )

    assert observed == retained
    assert output.read_bytes() == payload
    assert output.stat().st_ino == inode
    assert output.stat().st_nlink == 1
    assert not os.path.lexists(stage)


@pytest.mark.parametrize("publication", ("completed", "crash-linked"))
def test_activation_receipt_exact_retry_never_refetches_network_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    publication: str,
) -> None:
    parent = _private_directory(tmp_path / "receipt")
    output = parent / "activation-receipt.json"
    monkeypatch.setattr(runner, "ACTIVATION_RECEIPT", output)
    commit = "b" * 40
    retained = {"payload_commit_sha": commit, "activation_id": "activation-a2"}
    payload = runner._json_file_bytes(retained)
    stage = _publish(output, payload, publication)
    inode = output.stat().st_ino
    monkeypatch.setattr(
        runner,
        "_protocol_activation",
        lambda: ({}, {"receipt": {"fixed_values": {}}}),
    )
    monkeypatch.setattr(runner, "_git_require_commit", lambda value, label: value)
    monkeypatch.setattr(
        runner,
        "validate_activation_payload",
        lambda value, *args, **kwargs: ({"activation_id": "activation-a2"}, "c" * 64),
    )
    monkeypatch.setattr(
        runner,
        "validate_activation_receipt",
        lambda value, *args, **kwargs: (dict(value), "d" * 64),
    )
    for name in (
        "_github_observation",
        "_github_workflow_run_observation",
    ):
        monkeypatch.setattr(runner, name, _forbidden_network(name))

    observed = runner.create_activation_receipt(
        {"activation_id": "activation-a2"},
        payload_commit_sha=commit,
        output=output,
    )

    assert observed == retained
    assert output.read_bytes() == payload
    assert output.stat().st_ino == inode
    assert output.stat().st_nlink == 1
    assert not os.path.lexists(stage)


@pytest.mark.parametrize("publication", ("completed", "crash-linked"))
def test_preflight_context_exact_retry_heals_without_reactivating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    publication: str,
) -> None:
    parent = _private_directory(tmp_path / "preflight")
    payload_path = parent / "payload.json"
    receipt_path = parent / "receipt.json"
    context_path = parent / "context.json"
    context = {
        "activation_receipt_commit_sha": "c" * 40,
        "activation_payload_sha256": "a" * 64,
        "activation_receipt_sha256": "b" * 64,
    }
    stage = _publish(context_path, runner._json_file_bytes(context), publication)
    inode = context_path.stat().st_ino
    monkeypatch.setattr(runner, "ACTIVATION_PAYLOAD", payload_path)
    monkeypatch.setattr(runner, "ACTIVATION_RECEIPT", receipt_path)
    monkeypatch.setattr(runner, "ACTIVATION_CONTEXT", context_path)
    monkeypatch.setattr(
        runner,
        "validate_runtime_lock",
        lambda **kwargs: ({"lock_id": "preflight"}, runner.RUNTIME_LOCK_SHA256),
    )
    monkeypatch.setattr(
        runner, "validate_protocol", lambda *args, **kwargs: ({}, "d" * 64)
    )
    monkeypatch.setattr(runner, "validate_activation_context", lambda value: dict(value))
    monkeypatch.setattr(runner, "activate", _forbidden_network("activate"))

    assert (
        runner.main(
            [
                "preflight",
                "--receipt-commit-sha",
                context["activation_receipt_commit_sha"],
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out) == context
    assert context_path.stat().st_ino == inode
    assert context_path.stat().st_nlink == 1
    assert not os.path.lexists(stage)


def _patch_first_decide(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    context: dict[str, Any],
) -> Path:
    context_path = tmp_path / "activation-context.json"
    context_path.write_bytes(runner._json_file_bytes(context))
    context_path.chmod(0o600)
    monkeypatch.setattr(runner, "ACTIVATION_CONTEXT", context_path)
    monkeypatch.setattr(runner, "RESULT_OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(runner, "DECISION_LEDGER", tmp_path / "decisions.jsonl")
    monkeypatch.setattr(runner, "validate_activation_context", lambda value: dict(value))
    monkeypatch.setattr(
        runner,
        "validate_runtime_lock",
        lambda **kwargs: ({"lock_id": "first-decide"}, runner.RUNTIME_LOCK_SHA256),
    )
    monkeypatch.setattr(
        runner, "validate_protocol", lambda *args, **kwargs: ({}, "d" * 64)
    )
    monkeypatch.setattr(
        runner,
        "validate_canonical_activation_artifacts",
        lambda: ({}, "a" * 64, {}, "b" * 64),
    )
    monkeypatch.setattr(runner, "activate", _forbidden_network("activate"))
    return context_path


def test_first_decide_uses_only_canonical_activation_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = {
        "activation_payload_sha256": "a" * 64,
        "activation_receipt_sha256": "b" * 64,
        "activation_receipt_commit_sha": "c" * 40,
    }
    _patch_first_decide(monkeypatch, tmp_path, context)
    record = {
        "session_date": "2026-08-05",
        "sequence_number": 0,
        "record_sha256": "e" * 64,
    }
    loads = iter(([], [record]))
    monkeypatch.setattr(runner, "_load_decision_record_authority", lambda **kwargs: next(loads))

    def resolve(*args: Any, activation: dict[str, Any], **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        assert activation == context
        return {"core": "sealed"}, {"binding": "sealed"}

    monkeypatch.setattr(runner, "resolve_checkpoint", resolve)
    monkeypatch.setattr(
        runner,
        "materialize_checkpoint_decision",
        lambda *args, activation, **kwargs: [record] if activation == context else [],
    )
    monkeypatch.setattr(runner, "append_jsonl_record", lambda *args, **kwargs: record)
    monkeypatch.setattr(runner, "semantic_decision_hash", lambda rows: "f" * 64)

    assert (
        runner.main(
            [
                "decide",
                "--session",
                "2026-08-05",
                "--checkpoint-core-store-root",
                str(tmp_path / "checkpoint-core"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["record_sha256"] == record[
        "record_sha256"
    ]


@pytest.mark.parametrize("failure", ("missing", "decision-mismatch"))
def test_activation_context_missing_or_decision_mismatch_is_zero_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    context = {
        "activation_payload_sha256": "a" * 64,
        "activation_receipt_sha256": "b" * 64,
        "activation_receipt_commit_sha": "c" * 40,
    }
    context_path = tmp_path / "activation-context.json"
    monkeypatch.setattr(runner, "ACTIVATION_CONTEXT", context_path)
    monkeypatch.setattr(runner, "validate_activation_context", lambda value: dict(value))
    existing: list[dict[str, Any]] = []
    if failure == "decision-mismatch":
        context_path.write_bytes(runner._json_file_bytes(context))
        context_path.chmod(0o600)
        existing = [{"session_date": "2026-08-05"}]
        monkeypatch.setattr(
            runner,
            "_activation_context_from_decision",
            lambda row: {**context, "activation_payload_sha256": "0" * 64},
        )
    monkeypatch.setattr(runner, "activate", _forbidden_network("activate"))
    before = _tree_identity(tmp_path)

    with pytest.raises(runner.V18Error, match="missing|differs"):
        runner._discover_activation_context(existing)

    assert _tree_identity(tmp_path) == before


def test_discover_activation_context_has_no_legacy_activate_or_git_fallback() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    body = source[
        source.index("def _discover_activation_context(") :
        source.index("def _canonical_target_score_pair(")
    ]

    assert "ACTIVATION_CONTEXT" in body
    assert "read_json(ACTIVATION_CONTEXT)" in body
    assert "activate(" not in body
    assert "_git(" not in body
