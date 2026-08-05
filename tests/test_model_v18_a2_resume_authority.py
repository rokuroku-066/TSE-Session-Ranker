from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping

import pandas as pd
import pytest

from research import model_v18_shoulder_state_runner as runner


TARGET_SESSION = "2026-08-05"


def _private_directory(path: Path, mode: int = 0o700) -> Path:
    path.mkdir(mode=mode, parents=True, exist_ok=True)
    path.chmod(mode)
    return path


def _score_pair(session: str, code_base: int) -> pd.DataFrame:
    target = pd.Timestamp(session)
    previous = target - pd.offsets.BDay(1)
    return pd.DataFrame(
        {
            "session_date": [session, session],
            "source_rank": [1, 2],
            "code": [str(code_base), str(code_base + 1)],
            "name": [f"name-{code_base}", f"name-{code_base + 1}"],
            "model_score": [1.0, 0.5],
            "feature_source_max_date": [str(previous.date())] * 2,
            "score_generated_at": [f"{session}T08:00:00+09:00"] * 2,
            "runtime_lock_sha256": [runner.RUNTIME_LOCK_SHA256] * 2,
            "runtime_lock_verified_at": [f"{session}T07:59:00+09:00"] * 2,
            "source_manifest_sha256": ["a" * 64] * 2,
            "c00_fold_manifest_sha256": ["b" * 64] * 2,
        },
        columns=runner.SCORE_FIELDS,
    )


def _score_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame, bytes]:
    score_output = tmp_path / "scores.csv"
    score_sessions = _private_directory(tmp_path / "score-sessions")
    monkeypatch.setattr(runner, "SCORE_OUTPUT", score_output)
    monkeypatch.setattr(runner, "SCORE_SESSION_DIR", score_sessions)
    first = _score_pair("2026-08-05", 1001)
    second = _score_pair("2026-08-06", 2001)
    runner.append_score_rows(first, score_output)
    runner.append_score_rows(second, score_output)
    ledger = runner.validate_score_ledger(
        pd.concat([first, second], ignore_index=True)
    )
    expected = ledger.to_csv(index=False, lineterminator="\n").encode()
    assert score_output.read_bytes() == expected
    assert runner.validate_score_session_authority(ledger)
    return score_output, score_sessions, first, second, expected


@pytest.mark.parametrize(
    "crash_state",
    [
        "torn_final",
        "torn_final_and_stage",
        "stage_without_final",
        "completed_final_with_stage",
    ],
)
def test_score_session_shards_heal_only_exact_derived_prefixes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    crash_state: str,
) -> None:
    output, sessions, first, second, expected = _score_authority(
        monkeypatch, tmp_path
    )
    stage = output.with_name(f".{output.name}.derived-staging")
    first_payload = runner.validate_score_rows(first).to_csv(
        index=False, lineterminator="\n"
    ).encode()
    prefix = expected[: len(first_payload) + 17]

    if crash_state == "torn_final":
        output.write_bytes(prefix)
    elif crash_state == "torn_final_and_stage":
        output.write_bytes(first_payload)
        stage.write_bytes(prefix)
    elif crash_state == "stage_without_final":
        output.unlink()
        stage.write_bytes(prefix)
    else:
        stage.write_bytes(prefix)

    shard_snapshot = {
        item.name: item.read_bytes() for item in sorted(sessions.iterdir())
    }
    runner.append_score_rows(second, output)

    assert output.read_bytes() == expected
    assert not stage.exists()
    assert {
        item.name: item.read_bytes() for item in sorted(sessions.iterdir())
    } == shard_snapshot
    ledger = pd.read_csv(output, dtype={"code": "string"}, float_precision="round_trip")
    assert runner.validate_score_session_authority(ledger)


def test_score_session_authority_rejects_a_divergent_derived_ledger_without_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, sessions, _, second, _ = _score_authority(monkeypatch, tmp_path)
    corrupt = b"not,an,authority,prefix\n"
    output.write_bytes(corrupt)
    shard_snapshot = {
        item.name: item.read_bytes() for item in sorted(sessions.iterdir())
    }

    with pytest.raises(runner.V18Error, match="exact authority prefix"):
        runner.append_score_rows(second, output)

    assert output.read_bytes() == corrupt
    assert {
        item.name: item.read_bytes() for item in sorted(sessions.iterdir())
    } == shard_snapshot


def test_score_session_authority_rejects_an_extra_shard_name_without_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, sessions, _, second, expected = _score_authority(monkeypatch, tmp_path)
    extra = sessions / "not-a-registered-session.csv"
    extra.write_bytes(b"unregistered\n")

    with pytest.raises(runner.V18Error, match="unregistered name"):
        runner.append_score_rows(second, output)

    assert output.read_bytes() == expected
    assert extra.read_bytes() == b"unregistered\n"


def test_score_session_authority_rejects_a_hard_link_alias_without_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, sessions, _, second, expected = _score_authority(monkeypatch, tmp_path)
    shard = sessions / "2026-08-06.csv"
    alias = tmp_path / "score-shard-alias.csv"
    os.link(shard, alias)

    with pytest.raises(
        runner.V18Error,
        match="hard-link|single-link|alias|completed retry changed|not private",
    ):
        runner.append_score_rows(second, output)

    assert output.read_bytes() == expected
    assert os.path.samefile(shard, alias)
    assert shard.stat().st_nlink == 2


def test_score_session_authority_rejects_divergent_shard_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output, sessions, _, second, expected = _score_authority(monkeypatch, tmp_path)
    shard = sessions / "2026-08-06.csv"
    shard.write_bytes(b"divergent shard\n")

    with pytest.raises(runner.V18Error, match="differs|bytes|canonical"):
        runner.append_score_rows(second, output)

    assert output.read_bytes() == expected
    assert shard.read_bytes() == b"divergent shard\n"


def _checkpoint_payloads(suffix: str = ".bin") -> dict[str, bytes]:
    return {
        f"{role}{suffix}": f"fixed-{role}-payload".encode()
        for role in runner.CHECKPOINT_ROLES
    }


def _call_install(
    parent: Path,
    *,
    payloads: Mapping[str, bytes] | None = None,
    file_mode: int = 0o600,
    directory_mode: int = 0o700,
    label: str = "checkpoint test pair",
) -> dict[str, tuple[int, str]]:
    descriptor = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        return runner._install_checkpoint_session_directory(
            descriptor,
            session_name=TARGET_SESSION,
            payloads=_checkpoint_payloads() if payloads is None else payloads,
            file_mode=file_mode,
            directory_mode=directory_mode,
            label=label,
        )
    finally:
        os.close(descriptor)


def _expected_checkpoint_metadata(
    payloads: Mapping[str, bytes],
) -> dict[str, tuple[int, str]]:
    return {
        name: (len(payload), hashlib.sha256(payload).hexdigest())
        for name, payload in payloads.items()
    }


def _assert_installed_pair(
    parent: Path,
    payloads: Mapping[str, bytes],
    *,
    file_mode: int = 0o600,
    directory_mode: int = 0o700,
) -> None:
    final = parent / TARGET_SESSION
    assert final.is_dir() and not final.is_symlink()
    assert stat.S_IMODE(final.stat().st_mode) == directory_mode
    assert {item.name for item in final.iterdir()} == set(payloads)
    for name, payload in payloads.items():
        child = final / name
        assert child.read_bytes() == payload
        assert not child.is_symlink()
        assert child.stat().st_nlink == 1
        assert stat.S_IMODE(child.stat().st_mode) == file_mode
    assert not (parent / f".{TARGET_SESSION}.staging").exists()


def _tree_snapshot(root: Path) -> tuple[tuple[Any, ...], ...]:
    rows: list[tuple[Any, ...]] = []
    for path in sorted([root, *root.rglob("*")], key=lambda item: item.as_posix()):
        metadata = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        payload = path.read_bytes() if stat.S_ISREG(metadata.st_mode) else None
        target = os.readlink(path) if stat.S_ISLNK(metadata.st_mode) else None
        rows.append(
            (
                relative,
                stat.S_IFMT(metadata.st_mode),
                stat.S_IMODE(metadata.st_mode),
                metadata.st_uid,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                payload,
                target,
            )
        )
    return tuple(rows)


def test_checkpoint_pair_install_exact_completed_retry_is_validation_only(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "checkpoint-parent")
    payloads = _checkpoint_payloads()
    expected = _expected_checkpoint_metadata(payloads)

    assert _call_install(parent, payloads=payloads) == expected
    before = _tree_snapshot(parent)
    assert _call_install(parent, payloads=payloads) == expected

    assert _tree_snapshot(parent) == before
    _assert_installed_pair(parent, payloads)


def test_checkpoint_pair_install_recovers_a_partial_staged_directory(
    tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "checkpoint-parent")
    payloads = _checkpoint_payloads()
    stage = _private_directory(parent / f".{TARGET_SESSION}.staging")
    partial = stage / "safety_cash.bin"
    partial.write_bytes(b"torn unpublished bytes")
    partial.chmod(0o600)

    assert _call_install(parent, payloads=payloads) == _expected_checkpoint_metadata(
        payloads
    )

    _assert_installed_pair(parent, payloads)


@pytest.mark.parametrize(
    "fault",
    [
        "first_write",
        "first_file_fsync",
        "parent_fsync_before_rename",
        "rename",
        "parent_fsync_after_rename",
    ],
)
def test_checkpoint_pair_install_recovers_every_staged_publication_fault(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault: str,
) -> None:
    parent = _private_directory(tmp_path / f"checkpoint-parent-{fault}")
    payloads = _checkpoint_payloads()
    parent_metadata = parent.stat()
    real_write = os.write
    real_fsync = os.fsync
    real_rename = os.rename
    state = {"failed": False, "renamed": False}

    def injected_write(descriptor: int, payload: Any) -> int:
        if fault == "first_write" and not state["failed"]:
            state["failed"] = True
            raise OSError(errno.EIO, "injected first write failure")
        return real_write(descriptor, payload)

    def injected_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        is_parent = (metadata.st_dev, metadata.st_ino) == (
            parent_metadata.st_dev,
            parent_metadata.st_ino,
        )
        should_fail = (
            fault == "first_file_fsync" and stat.S_ISREG(metadata.st_mode)
        ) or (
            fault == "parent_fsync_before_rename"
            and is_parent
            and not state["renamed"]
        ) or (
            fault == "parent_fsync_after_rename" and is_parent and state["renamed"]
        )
        if should_fail and not state["failed"]:
            state["failed"] = True
            raise OSError(errno.EIO, f"injected {fault} failure")
        real_fsync(descriptor)

    def injected_rename(*args: Any, **kwargs: Any) -> None:
        if fault == "rename" and not state["failed"]:
            state["failed"] = True
            raise OSError(errno.EIO, "injected rename failure")
        real_rename(*args, **kwargs)
        state["renamed"] = True

    with monkeypatch.context() as patch:
        patch.setattr(runner.os, "write", injected_write)
        patch.setattr(runner.os, "fsync", injected_fsync)
        patch.setattr(runner.os, "rename", injected_rename)
        with pytest.raises(OSError, match="injected"):
            _call_install(parent, payloads=payloads)

    assert state["failed"] is True
    stage = parent / f".{TARGET_SESSION}.staging"
    final = parent / TARGET_SESSION
    if fault == "parent_fsync_after_rename":
        assert final.is_dir()
        assert not stage.exists()
    else:
        assert stage.is_dir()
        assert not final.exists()

    assert _call_install(parent, payloads=payloads) == _expected_checkpoint_metadata(
        payloads
    )
    _assert_installed_pair(parent, payloads)


@pytest.mark.parametrize(
    "invalid_state",
    ["extra", "symlink", "hardlink", "file_mode", "directory_mode"],
)
def test_checkpoint_pair_install_rejects_unsafe_staged_directory_without_mutation(
    tmp_path: Path, invalid_state: str
) -> None:
    parent = _private_directory(tmp_path / f"checkpoint-parent-{invalid_state}")
    stage = _private_directory(parent / f".{TARGET_SESSION}.staging")
    victim = parent / "victim.bin"
    if invalid_state == "extra":
        extra = stage / "extra.bin"
        extra.write_bytes(b"extra")
        extra.chmod(0o600)
    elif invalid_state == "symlink":
        victim.write_bytes(b"victim")
        (stage / "safety_cash.bin").symlink_to(victim)
    elif invalid_state == "hardlink":
        victim.write_bytes(b"victim")
        victim.chmod(0o600)
        os.link(victim, stage / "safety_cash.bin")
    elif invalid_state == "file_mode":
        child = stage / "safety_cash.bin"
        child.write_bytes(b"wrong mode")
        child.chmod(0o644)
    else:
        stage.chmod(0o755)
    before = _tree_snapshot(parent)

    with pytest.raises(runner.V18Error, match="recoverable|staged"):
        _call_install(parent)

    assert _tree_snapshot(parent) == before
    assert not (parent / TARGET_SESSION).exists()


def test_checkpoint_pair_install_rejects_staged_wrong_uid_where_portable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "checkpoint-parent-wrong-uid")
    stage = _private_directory(parent / f".{TARGET_SESSION}.staging")
    stage_identity = stage.stat()
    real_fstat = os.fstat

    def wrong_stage_owner(descriptor: int) -> os.stat_result:
        observed = real_fstat(descriptor)
        if (observed.st_dev, observed.st_ino) == (
            stage_identity.st_dev,
            stage_identity.st_ino,
        ):
            fields = list(observed)
            fields[4] = observed.st_uid + 1
            return os.stat_result(fields)
        return observed

    before = _tree_snapshot(parent)

    with monkeypatch.context() as patch:
        patch.setattr(runner.os, "fstat", wrong_stage_owner)
        with pytest.raises(runner.V18Error, match="recoverable"):
            _call_install(parent)

    assert _tree_snapshot(parent) == before
    assert not (parent / TARGET_SESSION).exists()


@pytest.mark.parametrize(
    "invalid_state", ["extra", "symlink", "hardlink", "file_mode", "directory_mode"]
)
def test_checkpoint_existing_pair_reader_rejects_extra_alias_and_mode_changes(
    tmp_path: Path, invalid_state: str
) -> None:
    parent = _private_directory(tmp_path / f"checkpoint-final-{invalid_state}")
    payloads = _checkpoint_payloads()
    _call_install(parent, payloads=payloads)
    final = parent / TARGET_SESSION
    victim = parent / "victim.bin"
    if invalid_state == "extra":
        extra = final / "extra.bin"
        extra.write_bytes(b"extra")
        extra.chmod(0o600)
    elif invalid_state == "symlink":
        child = final / "safety_cash.bin"
        child.unlink()
        victim.write_bytes(b"victim")
        child.symlink_to(victim)
    elif invalid_state == "hardlink":
        os.link(final / "safety_cash.bin", victim)
    elif invalid_state == "file_mode":
        (final / "safety_cash.bin").chmod(0o644)
    else:
        final.chmod(0o755)
    before = _tree_snapshot(parent)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        with pytest.raises(runner.V18Error, match="exact pair|metadata"):
            runner._existing_checkpoint_session_payloads(
                descriptor,
                session_name=TARGET_SESSION,
                file_names=tuple(payloads),
                file_mode=0o600,
                directory_mode=0o700,
                label="checkpoint test pair",
            )
    finally:
        os.close(descriptor)

    assert _tree_snapshot(parent) == before


def test_checkpoint_existing_pair_reader_rejects_wrong_uid_where_portable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    parent = _private_directory(tmp_path / "checkpoint-final-wrong-uid")
    payloads = _checkpoint_payloads()
    _call_install(parent, payloads=payloads)
    child = parent / TARGET_SESSION / "safety_cash.bin"
    child_identity = child.stat()
    real_stat = os.stat

    def wrong_child_owner(*args: Any, **kwargs: Any) -> os.stat_result:
        observed = real_stat(*args, **kwargs)
        if (observed.st_dev, observed.st_ino) == (
            child_identity.st_dev,
            child_identity.st_ino,
        ):
            fields = list(observed)
            fields[4] = observed.st_uid + 1
            return os.stat_result(fields)
        return observed

    before = _tree_snapshot(parent)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(runner.os, "stat", wrong_child_owner)
            with pytest.raises(runner.V18Error, match="metadata"):
                runner._existing_checkpoint_session_payloads(
                    descriptor,
                    session_name=TARGET_SESSION,
                    file_names=tuple(payloads),
                    file_mode=0o600,
                    directory_mode=0o700,
                    label="checkpoint test pair",
                )
    finally:
        os.close(descriptor)

    assert _tree_snapshot(parent) == before


def _checkpoint_core() -> dict[str, Any]:
    return {
        "candidate_id": runner.CANDIDATE_ID,
        "runtime_lock_verified_at": f"{TARGET_SESSION}T07:59:00+09:00",
        "source_manifest_sha256": "1" * 64,
        "c00_fold_manifest_sha256": "2" * 64,
        "fold_model_bundle_file_sha256": "3" * 64,
        "state_manifest_sha256": "4" * 64,
        "score_session_file_sha256": "5" * 64,
        "score_session_semantic_sha256": "6" * 64,
        "score_session_set_sha256": "7" * 64,
        "decision_cutoff": runner._cutoff(pd.Timestamp(TARGET_SESSION)).isoformat(),
        "computed_at": f"{TARGET_SESSION}T08:00:00+09:00",
        "source_complete": True,
        "model_complete": True,
        "state_available": True,
        "three_prior_calendar_months": ["2026-05", "2026-06", "2026-07"],
        "three_complete_pair_day_counts": [17, 19, 18],
        "three_month_medians_pct": [0.4, 0.5, 0.1],
        "state_value_pct": 0.4,
        "selected_source_rank": 1,
        "c00_rank1_code": "1001",
        "c00_rank1_score": 1.0,
        "c02_rank2_code": "1002",
        "c02_rank2_score": 0.5,
        "candidate_selected_code": "1001",
        "decision": "selected_rank1",
        "failure_reason": None,
    }


def _prepare_checkpoint_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path, Path, dict[str, Any]]:
    repository = _private_directory(tmp_path / "isolated-repository")
    proposal_root = _private_directory(
        repository / "research" / "checkpoint-proposals"
    )
    external_root = _private_directory(tmp_path / "isolated-external-store")
    core_parent = _private_directory(
        external_root / "model_v18_shoulder_state" / "checkpoint-core"
    )
    activation_path = repository / "research" / "activation-payload.json"
    activation_path.write_text('{"branch":"test-a2-resume"}\n', encoding="utf-8")
    activation = {
        "activation_payload_sha256": "8" * 64,
        "activation_receipt_sha256": "9" * 64,
        "activation_receipt_commit_sha": "a" * 40,
    }
    core = _checkpoint_core()

    monkeypatch.setattr(runner, "ROOT", repository)
    monkeypatch.setattr(runner, "CHECKPOINT_PROPOSAL_DIR", proposal_root)
    monkeypatch.setattr(runner, "ACTIVATION_PAYLOAD", activation_path)
    monkeypatch.setattr(runner, "DECISION_LEDGER", repository / "decisions.jsonl")
    monkeypatch.setattr(runner, "_STRICT_RUNTIME_ACTIVE", True)
    monkeypatch.setattr(
        runner, "_validate_startup_and_module_closure", lambda *, phase: None
    )

    def derive(
        target: pd.Timestamp,
        existing: Any,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert str(target.date()) == TARGET_SESSION
        assert existing == []
        if kwargs.get("computed_at") is not None:
            assert kwargs["computed_at"] == core["computed_at"]
            assert (
                kwargs["runtime_lock_verified_at"]
                == core["runtime_lock_verified_at"]
            )
            assert kwargs["capture_operational_timestamp"] is False
        return dict(core), dict(activation)

    monkeypatch.setattr(runner, "_derive_canonical_checkpoint_core", derive)
    return proposal_root, external_root, core_parent, core


def test_prepare_checkpoint_proposal_without_core_aborts_before_any_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proposal_root, external_root, _, _ = _prepare_checkpoint_harness(
        monkeypatch, tmp_path
    )
    proposal_session = _private_directory(
        proposal_root / TARGET_SESSION, mode=0o755
    )
    for role in runner.CHECKPOINT_ROLES:
        child = proposal_session / f"{role}.json"
        child.write_bytes(b"untrusted retained proposal\n")
        child.chmod(0o644)
    before_proposals = _tree_snapshot(proposal_root)
    before_external = _tree_snapshot(external_root)

    with pytest.raises(runner.V18Error, match="proposal exists without"):
        runner.prepare_checkpoint(
            session_date=TARGET_SESSION,
            checkpoint_core_store_root=external_root,
        )

    assert _tree_snapshot(proposal_root) == before_proposals
    assert _tree_snapshot(external_root) == before_external


def test_prepare_checkpoint_recovers_core_only_then_exact_bilateral_retry_is_no_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proposal_root, external_root, core_parent, _ = _prepare_checkpoint_harness(
        monkeypatch, tmp_path
    )
    real_install: Callable[..., dict[str, tuple[int, str]]] = (
        runner._install_checkpoint_session_directory
    )
    failed = {"proposal": False}

    def fail_before_proposal_install(*args: Any, **kwargs: Any) -> dict[str, tuple[int, str]]:
        if kwargs.get("label") == "checkpoint proposal" and not failed["proposal"]:
            failed["proposal"] = True
            raise OSError(errno.EIO, "injected bilateral proposal failure")
        return real_install(*args, **kwargs)

    monkeypatch.setattr(
        runner, "_install_checkpoint_session_directory", fail_before_proposal_install
    )
    with pytest.raises(OSError, match="injected bilateral"):
        runner.prepare_checkpoint(
            session_date=TARGET_SESSION,
            checkpoint_core_store_root=external_root,
        )

    core_session = core_parent / TARGET_SESSION
    assert core_session.is_dir()
    assert not (proposal_root / TARGET_SESSION).exists()
    retained_core = _tree_snapshot(core_session)

    monkeypatch.setattr(
        runner, "_install_checkpoint_session_directory", real_install
    )
    recovered = runner.prepare_checkpoint(
        session_date=TARGET_SESSION,
        checkpoint_core_store_root=external_root,
    )
    assert recovered["target_session"] == TARGET_SESSION
    assert [
        item["checkpoint_role"] for item in recovered["role_artifacts"]
    ] == list(runner.CHECKPOINT_ROLES)
    assert _tree_snapshot(core_session) == retained_core
    assert (proposal_root / TARGET_SESSION).is_dir()

    before_core = _tree_snapshot(core_parent)
    before_proposals = _tree_snapshot(proposal_root)
    exact_retry = runner.prepare_checkpoint(
        session_date=TARGET_SESSION,
        checkpoint_core_store_root=external_root,
    )

    assert exact_retry == recovered
    assert _tree_snapshot(core_parent) == before_core
    assert _tree_snapshot(proposal_root) == before_proposals
