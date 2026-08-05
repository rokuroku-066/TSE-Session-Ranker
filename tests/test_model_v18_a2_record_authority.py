from __future__ import annotations

import inspect
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping

import pytest

from research import model_v18_shoulder_state_runner as runner


ROLE_CASES = (
    ("decision", "session_date", ("2026-08-05", "2026-08-06", "2026-08-07")),
    ("outcome", "session_date", ("2026-08-05", "2026-08-06", "2026-08-07")),
    ("completed", "completed_month", ("2026-06", "2026-07", "2026-08")),
)


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
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_nlink,
                metadata.st_size,
                path.read_bytes() if stat.S_ISREG(metadata.st_mode) else None,
                os.readlink(path) if stat.S_ISLNK(metadata.st_mode) else None,
            )
        )
    return tuple(rows)


def _role_spec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
    key_field: str,
    *,
    precreate_authority: bool = True,
) -> tuple[Path, Path, tuple[str, ...], Callable[[Any], list[dict[str, Any]]]]:
    root = _private_directory(tmp_path / role)
    ledger = root / f"{role}.jsonl"
    authority = root / "records"
    if role == "decision":
        monkeypatch.setattr(runner, "DECISION_LEDGER", ledger)
        monkeypatch.setattr(runner, "DECISION_RECORD_DIR", authority)
    elif role == "outcome":
        monkeypatch.setattr(runner, "OUTCOME_LEDGER", ledger)
        monkeypatch.setattr(runner, "OUTCOME_RECORD_DIR", authority)
    else:
        monkeypatch.setattr(runner, "COMPLETED_MONTH_LEDGER", ledger)
        monkeypatch.setattr(runner, "COMPLETED_MONTH_RECORD_DIR", authority)
    if precreate_authority:
        _private_directory(authority)
    fields = (
        "schema_version",
        "record_kind",
        key_field,
        "created_at",
        "sequence_number",
        "previous_record_sha256",
        "record_sha256",
    )

    def validator(value: Any) -> list[dict[str, Any]]:
        return runner.validate_hash_chain(value, required_fields=fields)

    return ledger, authority, fields, validator


def _record(
    role: str,
    key_field: str,
    key: str,
    sequence: int,
    previous: str,
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    return runner._sealed_record(
        {
            "schema_version": 1,
            "record_kind": role,
            key_field: key,
            "created_at": created_at or f"2026-08-05T0{sequence}:00:00+09:00",
        },
        sequence_number=sequence,
        previous_record_sha256=previous,
    )


def _chain(
    role: str, key_field: str, keys: tuple[str, ...], count: int = 2
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous = runner.ZERO_SHA256
    for sequence, key in enumerate(keys[:count]):
        row = _record(role, key_field, key, sequence, previous)
        rows.append(row)
        previous = row["record_sha256"]
    return rows


def _derived_bytes(rows: list[Mapping[str, Any]]) -> bytes:
    return b"".join(runner.canonical_json_bytes(dict(item)) + b"\n" for item in rows)


def _append_chain(
    ledger: Path,
    fields: tuple[str, ...],
    validator: Callable[[Any], list[dict[str, Any]]],
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        assert runner.append_jsonl_record(
            ledger,
            row,
            required_fields=fields,
            validator=validator,
        ) == row


@pytest.mark.parametrize(("role", "key_field", "keys"), ROLE_CASES)
def test_record_shards_heal_missing_stale_and_partial_derived_views(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
    key_field: str,
    keys: tuple[str, ...],
) -> None:
    ledger, authority, fields, validator = _role_spec(
        monkeypatch, tmp_path, role, key_field
    )
    rows = _chain(role, key_field, keys)
    _append_chain(ledger, fields, validator, rows)
    expected = _derived_bytes(rows)
    assert ledger.read_bytes() == expected
    assert [item.name for item in sorted(authority.iterdir())] == [
        f"{keys[0]}.json",
        f"{keys[1]}.json",
    ]

    ledger.unlink()
    assert runner.load_jsonl_record_authority(
        ledger,
        required_fields=fields,
        validator=validator,
        heal_derived=True,
    ) == rows
    assert ledger.read_bytes() == expected

    ledger.write_bytes(_derived_bytes(rows[:1]))
    runner.load_jsonl_record_authority(
        ledger,
        required_fields=fields,
        validator=validator,
        heal_derived=True,
    )
    assert ledger.read_bytes() == expected

    ledger.write_bytes(expected[:-11])
    runner.load_jsonl_record_authority(
        ledger,
        required_fields=fields,
        validator=validator,
        heal_derived=True,
    )
    assert ledger.read_bytes() == expected


@pytest.mark.parametrize(("role", "key_field", "keys"), ROLE_CASES)
def test_shard_final_without_ledger_is_an_exact_append_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
    key_field: str,
    keys: tuple[str, ...],
) -> None:
    ledger, authority, fields, validator = _role_spec(
        monkeypatch, tmp_path, role, key_field
    )
    row = _chain(role, key_field, keys, count=1)[0]
    _private_directory(authority)
    final = authority / f"{keys[0]}.json"
    final.write_bytes(runner.canonical_json_bytes(row) + b"\n")
    final.chmod(0o600)
    before_created_at = row["created_at"]

    retained = runner.append_jsonl_record(
        ledger,
        row,
        required_fields=fields,
        validator=validator,
    )

    assert retained == row
    assert retained["created_at"] == before_created_at
    assert ledger.read_bytes() == _derived_bytes([row])
    assert final.read_bytes() == runner.canonical_json_bytes(row) + b"\n"


@pytest.mark.parametrize(("role", "key_field", "keys"), ROLE_CASES)
def test_exact_retained_record_retry_preserves_its_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
    key_field: str,
    keys: tuple[str, ...],
) -> None:
    ledger, _, fields, validator = _role_spec(
        monkeypatch, tmp_path, role, key_field
    )
    row = _record(
        role,
        key_field,
        keys[0],
        0,
        runner.ZERO_SHA256,
        created_at="2026-08-05T01:02:03.456789+09:00",
    )
    runner.append_jsonl_record(
        ledger, row, required_fields=fields, validator=validator
    )

    retained = runner.append_jsonl_record(
        ledger, row, required_fields=fields, validator=validator
    )

    assert retained == row
    assert retained["created_at"] == "2026-08-05T01:02:03.456789+09:00"
    assert ledger.read_bytes() == _derived_bytes([row])


@pytest.mark.parametrize(("role", "key_field", "keys"), ROLE_CASES)
def test_final_plus_stage_link_crash_heals_before_derived_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
    key_field: str,
    keys: tuple[str, ...],
) -> None:
    ledger, authority, fields, validator = _role_spec(
        monkeypatch, tmp_path, role, key_field
    )
    row = _chain(role, key_field, keys, count=1)[0]
    _private_directory(authority)
    final = authority / f"{keys[0]}.json"
    stage = authority / f".{final.name}.staging"
    stage.write_bytes(runner.canonical_json_bytes(row) + b"\n")
    stage.chmod(0o600)
    os.link(stage, final)
    assert os.path.samefile(stage, final)
    assert final.stat().st_nlink == 2

    retained = runner.append_jsonl_record(
        ledger,
        row,
        required_fields=fields,
        validator=validator,
    )

    assert retained == row
    assert not stage.exists()
    assert final.stat().st_nlink == 1
    assert ledger.read_bytes() == _derived_bytes([row])


@pytest.mark.parametrize(("role", "key_field", "keys"), ROLE_CASES)
def test_unpublished_dynamic_stage_is_rebuilt_by_append(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
    key_field: str,
    keys: tuple[str, ...],
) -> None:
    ledger, authority, fields, validator = _role_spec(
        monkeypatch, tmp_path, role, key_field
    )
    row = _chain(role, key_field, keys, count=1)[0]
    _private_directory(authority)
    stage = authority / f".{keys[0]}.json.staging"
    stale = b'{"created_at":"2026-08-05T00:00:00+09:00"'
    stage.write_bytes(stale)
    stage.chmod(0o600)

    retained = runner.append_jsonl_record(
        ledger,
        row,
        required_fields=fields,
        validator=validator,
    )

    assert retained == row
    assert not stage.exists()
    assert (authority / f"{keys[0]}.json").read_bytes() == (
        runner.canonical_json_bytes(row) + b"\n"
    )


@pytest.mark.parametrize(("role", "key_field", "keys"), ROLE_CASES)
@pytest.mark.parametrize("view_fault", ["divergent", "extra_suffix"])
def test_divergent_or_extra_derived_view_is_rejected_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
    key_field: str,
    keys: tuple[str, ...],
    view_fault: str,
) -> None:
    ledger, _, fields, validator = _role_spec(
        monkeypatch, tmp_path, role, key_field
    )
    rows = _chain(role, key_field, keys)
    _append_chain(ledger, fields, validator, rows)
    expected = _derived_bytes(rows)
    if view_fault == "divergent":
        damaged = b"X" + expected[1:]
    else:
        damaged = expected + b'{}\n'
    ledger.write_bytes(damaged)
    before = _tree_identity(tmp_path / role)

    with pytest.raises(
        runner.V18Error, match="diverges|conflict|not an exact authority prefix"
    ):
        runner.load_jsonl_record_authority(
            ledger,
            required_fields=fields,
            validator=validator,
            heal_derived=True,
        )

    assert _tree_identity(tmp_path / role) == before


@pytest.mark.parametrize(("role", "key_field", "keys"), ROLE_CASES)
@pytest.mark.parametrize(
    "authority_fault",
    [
        "unregistered",
        "symlink",
        "hardlink",
        "wrong_mode",
        "wrong_filename",
        "duplicate_sequence",
        "sequence_gap",
    ],
)
def test_authority_extra_alias_key_and_sequence_faults_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
    key_field: str,
    keys: tuple[str, ...],
    authority_fault: str,
) -> None:
    ledger, authority, fields, validator = _role_spec(
        monkeypatch, tmp_path, role, key_field
    )
    rows = _chain(role, key_field, keys)
    runner.append_jsonl_record(
        ledger, rows[0], required_fields=fields, validator=validator
    )
    victim = tmp_path / f"{role}-victim.bin"
    victim.write_bytes(b"victim must remain unchanged")
    candidate_path = authority / f"{keys[1]}.json"
    if authority_fault == "unregistered":
        (authority / "extra.bin").write_bytes(b"extra")
    elif authority_fault == "symlink":
        candidate_path.symlink_to(victim)
    elif authority_fault == "hardlink":
        os.link(victim, candidate_path)
    elif authority_fault == "wrong_mode":
        (authority / f"{keys[0]}.json").chmod(0o644)
    elif authority_fault == "wrong_filename":
        candidate_path.write_bytes(runner.canonical_json_bytes(rows[0]) + b"\n")
        candidate_path.chmod(0o600)
    elif authority_fault == "duplicate_sequence":
        duplicate = _record(
            role, key_field, keys[1], 0, runner.ZERO_SHA256
        )
        candidate_path.write_bytes(runner.canonical_json_bytes(duplicate) + b"\n")
        candidate_path.chmod(0o600)
    elif authority_fault == "sequence_gap":
        gap = _record(
            role, key_field, keys[2], 2, rows[0]["record_sha256"]
        )
        candidate_path = authority / f"{keys[2]}.json"
        candidate_path.write_bytes(runner.canonical_json_bytes(gap) + b"\n")
        candidate_path.chmod(0o600)
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(authority_fault)
    before = _tree_identity(tmp_path)

    with pytest.raises(runner.V18Error):
        runner.load_jsonl_record_authority(
            ledger,
            required_fields=fields,
            validator=validator,
            heal_derived=True,
        )

    assert _tree_identity(tmp_path) == before
    assert victim.read_bytes() == b"victim must remain unchanged"


@pytest.mark.parametrize(("role", "key_field", "keys"), ROLE_CASES)
@pytest.mark.parametrize("stage_fault", ["symlink", "hardlink"])
def test_aliased_unpublished_stage_cannot_mutate_its_victim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
    key_field: str,
    keys: tuple[str, ...],
    stage_fault: str,
) -> None:
    ledger, authority, fields, validator = _role_spec(
        monkeypatch, tmp_path, role, key_field
    )
    row = _chain(role, key_field, keys, count=1)[0]
    _private_directory(authority)
    victim = tmp_path / f"{role}-stage-victim.bin"
    victim_payload = b"unrelated victim exact bytes"
    victim.write_bytes(victim_payload)
    stage = authority / f".{keys[0]}.json.staging"
    if stage_fault == "symlink":
        stage.symlink_to(victim)
    else:
        os.link(victim, stage)
    before = _tree_identity(tmp_path)

    with pytest.raises(runner.V18Error):
        runner.append_jsonl_record(
            ledger,
            row,
            required_fields=fields,
            validator=validator,
        )

    assert _tree_identity(tmp_path) == before
    assert victim.read_bytes() == victim_payload


def test_canonical_ledger_callsites_have_zero_direct_jsonl_reads() -> None:
    source = inspect.getsource(runner)
    forbidden = re.compile(
        r"read_jsonl\((?:DECISION_LEDGER|OUTCOME_LEDGER|COMPLETED_MONTH_LEDGER)\)"
    )
    assert forbidden.search(source) is None
    for token in (
        "_load_decision_record_authority(heal_derived=True)",
        "_load_outcome_record_authority(",
        "_load_completed_month_record_authority(heal_derived=True)",
    ):
        assert token in source


def _patch_readiness_paths(
    monkeypatch: pytest.MonkeyPatch, local_root: Path
) -> tuple[Path, ...]:
    monkeypatch.setattr(runner, "ROOT", local_root)
    names = (
        "CHECKPOINT_PROPOSAL_DIR",
        "DECISION_RECORD_DIR",
        "OUTCOME_RECORD_DIR",
        "COMPLETED_MONTH_RECORD_DIR",
        "SOURCE_MANIFEST_DIR",
        "MONTH_SOURCE_MANIFEST_DIR",
        "OUTCOME_MANIFEST_DIR",
        "FOLD_MANIFEST_DIR",
        "FOLD_MODEL_DIR",
        "STATE_MANIFEST_DIR",
        "SCORE_SESSION_DIR",
    )
    paths: list[Path] = []
    for name in names:
        path = local_root / name.lower()
        monkeypatch.setattr(runner, name, path)
        paths.append(path)
    return tuple(paths)


def test_prepare_operational_stores_precreates_every_registered_local_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_root = _private_directory(tmp_path / "local")
    local_paths = _patch_readiness_paths(monkeypatch, local_root)
    external_roots = [
        _private_directory(tmp_path / name)
        for name in (
            "predictor-raw",
            "predictor-derived",
            "outcome-raw",
            "checkpoint-core",
        )
    ]
    real_fsync = os.fsync
    fsynced_paths: list[Path] = []

    def recording_fsync(descriptor: int) -> None:
        try:
            fsynced_paths.append(
                Path(os.readlink(f"/proc/self/fd/{descriptor}")).resolve()
            )
        except OSError:
            pass
        real_fsync(descriptor)

    monkeypatch.setattr(runner.os, "fsync", recording_fsync)

    result = runner.prepare_operational_stores(
        predictor_raw_store_root=external_roots[0],
        predictor_derived_store_root=external_roots[1],
        outcome_raw_store_root=external_roots[2],
        checkpoint_core_store_root=external_roots[3],
    )

    assert result["external_roots_pairwise_disjoint"] is True
    for path in local_paths:
        metadata = path.stat(follow_symlinks=False)
        assert path.is_dir() and not path.is_symlink()
        assert metadata.st_uid == os.geteuid()
        assert stat.S_IMODE(metadata.st_mode) == 0o700
    assert local_root.resolve() in fsynced_paths


@pytest.mark.parametrize(("role", "key_field", "keys"), ROLE_CASES)
def test_daily_append_requires_readiness_parent_and_is_zero_write_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
    key_field: str,
    keys: tuple[str, ...],
) -> None:
    ledger, authority, fields, validator = _role_spec(
        monkeypatch,
        tmp_path,
        role,
        key_field,
        precreate_authority=False,
    )
    row = _chain(role, key_field, keys, count=1)[0]
    before = _tree_identity(tmp_path / role)

    with pytest.raises(runner.V18Error, match="readiness|missing|authority"):
        runner.append_jsonl_record(
            ledger,
            row,
            required_fields=fields,
            validator=validator,
        )

    assert _tree_identity(tmp_path / role) == before
    assert not authority.exists()
    assert not ledger.exists()
