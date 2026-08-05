from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Any

import pandas as pd
import pytest

from research import model_v18_shoulder_state_runner as runner


ARTIFACT_ROLES = (
    "source",
    "month-source",
    "outcome",
    "state",
    "score",
    "result",
    "activation-payload",
    "activation-receipt",
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
                metadata.st_uid,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_nlink,
                path.read_bytes() if stat.S_ISREG(metadata.st_mode) else None,
            )
        )
    return tuple(rows)


def _artifact_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
) -> tuple[Path, Path, bytes]:
    parent = _private_directory(tmp_path / role)
    if role == "source":
        monkeypatch.setattr(runner, "SOURCE_MANIFEST_DIR", parent)
        final = parent / "2026-08-05.json"
    elif role == "month-source":
        monkeypatch.setattr(runner, "MONTH_SOURCE_MANIFEST_DIR", parent)
        final = parent / "2026-08.json"
    elif role == "outcome":
        monkeypatch.setattr(runner, "OUTCOME_MANIFEST_DIR", parent)
        final = parent / "2026-08-05.json"
    elif role == "state":
        monkeypatch.setattr(runner, "STATE_MANIFEST_DIR", parent)
        final = parent / "2026-08.json"
    elif role == "score":
        monkeypatch.setattr(runner, "SCORE_SESSION_DIR", parent)
        final = parent / "2026-08-05.csv"
    elif role == "result":
        final = parent / "result.json"
        monkeypatch.setattr(runner, "RESULT_OUTPUT", final)
    elif role == "activation-payload":
        final = parent / "activation-payload.json"
        monkeypatch.setattr(runner, "ACTIVATION_PAYLOAD", final)
    elif role == "activation-receipt":
        final = parent / "activation-receipt.json"
        monkeypatch.setattr(runner, "ACTIVATION_RECEIPT", final)
    else:  # pragma: no cover - parametrization is closed
        raise AssertionError(role)
    payload = (
        b"artifact,value\nscore,1\n"
        if role == "score"
        else runner._json_file_bytes({"artifact": role, "value": 1})
    )
    return parent, final, payload


def _read_registered(final: Path, role: str) -> Any:
    if role == "score":
        return runner._read_csv_plain(final, label="canonical score shard")
    return runner.read_json(final)


@pytest.mark.parametrize("role", ARTIFACT_ROLES)
def test_registered_local_authority_read_heals_exact_post_link_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
) -> None:
    parent, final, payload = _artifact_case(monkeypatch, tmp_path, role)
    stage = final.with_name(f".{final.name}.staging")
    stage.write_bytes(payload)
    stage.chmod(0o600)
    os.link(stage, final)
    inode = final.stat().st_ino

    retained = _read_registered(final, role)

    if role == "score":
        assert isinstance(retained, pd.DataFrame)
        assert retained.to_dict("records") == [{"artifact": "score", "value": 1}]
    else:
        assert retained == {"artifact": role, "value": 1}
    assert final.read_bytes() == payload
    assert final.stat().st_ino == inode
    assert final.stat().st_nlink == 1
    assert stat.S_IMODE(final.stat().st_mode) == 0o600
    assert not os.path.lexists(stage)
    assert [item.name for item in parent.iterdir()] == [final.name]


@pytest.mark.parametrize("role", ARTIFACT_ROLES)
def test_registered_local_authority_read_rejects_0644_crash_pair_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
) -> None:
    parent, final, payload = _artifact_case(monkeypatch, tmp_path, role)
    stage = final.with_name(f".{final.name}.staging")
    stage.write_bytes(payload)
    stage.chmod(0o644)
    os.link(stage, final)
    before = _tree_identity(parent)

    with pytest.raises(runner.V18Error, match="crash stage|mode|private"):
        _read_registered(final, role)

    assert _tree_identity(parent) == before
    assert os.path.samefile(stage, final)
    assert stage.stat().st_nlink == 2


@pytest.mark.parametrize("role", ("activation-payload", "activation-receipt"))
def test_git_tracked_activation_final_allows_0644_but_result_does_not(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    role: str,
) -> None:
    _, final, payload = _artifact_case(monkeypatch, tmp_path, role)
    final.write_bytes(payload)
    final.chmod(0o644)

    assert _read_registered(final, role) == {"artifact": role, "value": 1}

    result_parent, result, result_payload = _artifact_case(
        monkeypatch, tmp_path, "result"
    )
    result.write_bytes(result_payload)
    result.chmod(0o644)
    before = _tree_identity(result_parent)
    with pytest.raises(runner.V18Error, match="mode|private"):
        _read_registered(result, "result")
    assert _tree_identity(result_parent) == before


def test_registered_readers_route_every_local_authority_through_healer() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")

    assert "_is_registered_local_authority_path(path)" in source
    assert "_read_local_authority_bytes(path, label=label)" in source
    assert "_registered_local_authority_files()" in source
    file_registry = source[
        source.index("def _registered_local_authority_files") :
        source.index("def _is_registered_local_authority_path")
    ]
    for name in (
        "ACTIVATION_PAYLOAD",
        "ACTIVATION_RECEIPT",
        "ACTIVATION_CONTEXT",
        "RESULT_OUTPUT",
    ):
        assert name in file_registry
    for name in (
        "SOURCE_MANIFEST_DIR",
        "MONTH_SOURCE_MANIFEST_DIR",
        "OUTCOME_MANIFEST_DIR",
        "STATE_MANIFEST_DIR",
        "SCORE_SESSION_DIR",
    ):
        assert name in source[source.index("def _registered_local_authority_directories"):]
