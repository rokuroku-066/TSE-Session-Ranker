from __future__ import annotations

import errno
import os
from pathlib import Path
import stat
from typing import Any

import pytest

from research import model_v18_shoulder_state_runner as runner


PAYLOAD = b'{"created_at":"2026-08-05T09:00:01+09:00","value":1}\n'
OTHER_PAYLOAD = b'{"created_at":"2026-08-05T09:00:02+09:00","value":2}\n'


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True)
    path.chmod(0o700)
    return path


def _case(tmp_path: Path, kind: str, name: str) -> dict[str, Any]:
    root = _private_directory(tmp_path / f"{kind}-{name}")
    if kind == "local":
        final = root / f"{name}.json"
        key = None
    else:
        key = f"test-a2/{name}.json"
        final = root.joinpath(*key.split("/"))
        _private_directory(final.parent)
    return {
        "kind": kind,
        "root": root,
        "key": key,
        "final": final,
        "stage": final.with_name(f".{final.name}.staging"),
    }


def _publish(
    case: dict[str, Any],
    payload: bytes,
    *,
    replace_unpublished_stage: bool = False,
) -> None:
    if case["kind"] == "local":
        runner._atomic_local_bytes_once(case["final"], payload)
        return
    runner._write_external_bytes_once(
        payload,
        store_root=case["root"],
        object_key=case["key"],
        prefix="test-a2/",
        label="A2 atomic test object",
        replace_unpublished_stage=replace_unpublished_stage,
    )


def _assert_published(case: dict[str, Any], payload: bytes) -> None:
    final = case["final"]
    assert final.read_bytes() == payload
    assert not case["stage"].exists()
    assert final.stat().st_nlink == 1


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


@pytest.mark.parametrize("kind", ["local", "external"])
def test_atomic_create_once_resumes_an_exact_staging_prefix(
    tmp_path: Path, kind: str
) -> None:
    case = _case(tmp_path, kind, "prefix-resume")
    case["stage"].write_bytes(PAYLOAD[:23])
    case["stage"].chmod(0o600)

    _publish(case, PAYLOAD)

    _assert_published(case, PAYLOAD)


def test_local_unpublished_dynamic_stage_is_safely_rebuilt(tmp_path: Path) -> None:
    case = _case(tmp_path, "local", "dynamic-restart")
    stale = b'{"created_at":"2026-08-05T09:00:00+09:00"'
    case["stage"].write_bytes(stale)
    case["stage"].chmod(0o600)

    _publish(case, PAYLOAD)

    _assert_published(case, PAYLOAD)


def test_external_dynamic_stage_requires_explicit_restart_authority(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, "external", "dynamic-restart")
    stale = b'{"created_at":"2026-08-05T09:00:00+09:00"'
    case["stage"].write_bytes(stale)
    case["stage"].chmod(0o600)

    with pytest.raises(runner.V18Error, match="staging bytes conflict"):
        _publish(case, PAYLOAD)
    assert case["stage"].read_bytes() == stale
    assert not case["final"].exists()

    _publish(case, PAYLOAD, replace_unpublished_stage=True)

    _assert_published(case, PAYLOAD)


@pytest.mark.parametrize("kind", ["local", "external"])
@pytest.mark.parametrize("conflict", ["symlink", "hardlink"])
def test_atomic_create_once_rejects_nonprivate_staging_aliases_without_mutation(
    tmp_path: Path, kind: str, conflict: str
) -> None:
    case = _case(tmp_path, kind, f"{conflict}-conflict")
    victim = case["stage"].with_name("victim.bin")
    victim_payload = b"must remain untouched"
    victim.write_bytes(victim_payload)
    if conflict == "symlink":
        case["stage"].symlink_to(victim.name)
    else:
        os.link(victim, case["stage"])

    with pytest.raises(runner.V18Error):
        _publish(case, PAYLOAD, replace_unpublished_stage=True)

    assert victim.read_bytes() == victim_payload
    assert not case["final"].exists()
    if conflict == "hardlink":
        assert victim.stat().st_nlink == 2


@pytest.mark.parametrize("kind", ["local", "external"])
def test_atomic_create_once_heals_link_before_unlink_crash_state(
    tmp_path: Path, kind: str
) -> None:
    case = _case(tmp_path, kind, "nlink2-heal")
    case["stage"].write_bytes(PAYLOAD)
    case["stage"].chmod(0o600)
    os.link(case["stage"], case["final"])
    assert case["stage"].stat().st_nlink == 2

    _publish(case, PAYLOAD)

    _assert_published(case, PAYLOAD)


@pytest.mark.parametrize("kind", ["local", "external"])
def test_nlink2_restart_rejects_nonprivate_mode_before_unlink(
    tmp_path: Path,
    kind: str,
) -> None:
    case = _case(tmp_path, kind, "nlink2-hostile-mode")
    case["stage"].write_bytes(PAYLOAD)
    case["stage"].chmod(0o644)
    os.link(case["stage"], case["final"])
    assert os.path.samefile(case["stage"], case["final"])
    assert case["stage"].stat().st_nlink == 2
    before = _tree_identity(case["root"])

    with pytest.raises(runner.V18Error, match="stage|mode|0600|private"):
        _publish(case, PAYLOAD)

    assert _tree_identity(case["root"]) == before
    assert os.path.samefile(case["stage"], case["final"])
    assert case["stage"].stat().st_nlink == 2


def test_local_completed_final_rejects_nonprivate_mode_without_mutation(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, "local", "completed-hostile-mode")
    _publish(case, PAYLOAD)
    case["final"].chmod(0o644)
    before = _tree_identity(case["root"])

    with pytest.raises(runner.V18Error, match="mode|0600|private"):
        _publish(case, PAYLOAD)

    assert _tree_identity(case["root"]) == before


def test_local_unpublished_stage_rejects_nonprivate_mode_without_mutation(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, "local", "stage-hostile-mode")
    case["stage"].write_bytes(PAYLOAD[:23])
    case["stage"].chmod(0o644)
    before = _tree_identity(case["root"])

    with pytest.raises(runner.V18Error, match="stage|mode|0600|private"):
        _publish(case, PAYLOAD)

    assert _tree_identity(case["root"]) == before


@pytest.mark.parametrize("kind", ["local", "external"])
def test_atomic_create_once_rejects_different_bytes_after_publication(
    tmp_path: Path, kind: str
) -> None:
    case = _case(tmp_path, kind, "published-conflict")
    _publish(case, PAYLOAD)

    with pytest.raises(runner.V18Error):
        _publish(case, OTHER_PAYLOAD, replace_unpublished_stage=True)

    _assert_published(case, PAYLOAD)


@pytest.mark.parametrize("kind", ["local", "external"])
@pytest.mark.parametrize("fault", ["link", "parent_fsync_after_link", "unlink"])
def test_atomic_create_once_recovers_from_publication_faults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    fault: str,
) -> None:
    case = _case(tmp_path, kind, f"fault-{fault}")
    real_link = os.link
    real_fsync = os.fsync
    real_unlink = os.unlink
    state = {"linked": False, "failed": False}

    def injected_link(*args: object, **kwargs: object) -> None:
        if fault == "link" and not state["failed"]:
            state["failed"] = True
            raise OSError(errno.EIO, "injected link failure")
        real_link(*args, **kwargs)
        state["linked"] = True

    def injected_fsync(descriptor: int) -> None:
        if (
            fault == "parent_fsync_after_link"
            and state["linked"]
            and not state["failed"]
            and stat.S_ISDIR(os.fstat(descriptor).st_mode)
        ):
            state["failed"] = True
            raise OSError(errno.EIO, "injected parent fsync failure")
        real_fsync(descriptor)

    def injected_unlink(*args: object, **kwargs: object) -> None:
        if fault == "unlink" and state["linked"] and not state["failed"]:
            state["failed"] = True
            raise OSError(errno.EIO, "injected unlink failure")
        real_unlink(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(runner.os, "link", injected_link)
        patch.setattr(runner.os, "fsync", injected_fsync)
        patch.setattr(runner.os, "unlink", injected_unlink)
        with pytest.raises(OSError, match="injected"):
            _publish(case, PAYLOAD)

    assert state["failed"] is True
    if fault == "link":
        assert not case["final"].exists()
        assert case["stage"].read_bytes() == PAYLOAD
        assert case["stage"].stat().st_nlink == 1
    else:
        assert os.path.samefile(case["stage"], case["final"])
        assert case["stage"].stat().st_nlink == 2

    _publish(case, PAYLOAD)

    _assert_published(case, PAYLOAD)
