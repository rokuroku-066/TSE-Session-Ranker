#!/usr/bin/env python3
"""Outcome-blind command planning for the preregistered v1.8 A2 runner.

This helper never imports or executes the model runner.  It accepts only
operator-supplied source files, maintains a source-only convenience inventory,
checks the four disjoint external store roots, and writes deterministic plans
whose argv vectors invoke the canonical runner.  The inventory and plans are
not scientific authority.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "research/model_v18_shoulder_state_runner.py"
RUNTIME_LOCK = ROOT / "research/model_v18_runtime_lock.json"
CALENDAR = ROOT / "research/model_v18_tse_session_calendar.csv"
PRICE_LOCK = ROOT / "research/model_v05_input_lock.json"
LEGACY_DAILY_LOCK = ROOT / "research/model_v04_parser_recovery_audit.json"
REPLAY_DAILY_LOCK = ROOT / "research/model_v17_replay_input_lock.json"

ACTIVATION_PAYLOAD = (
    ROOT / "research/model_v18_shoulder_state_activation_payload.json"
)
ACTIVATION_RECEIPT = (
    ROOT / "research/model_v18_shoulder_state_activation_receipt.json"
)
ACTIVATION_CONTEXT = (
    ROOT / "research/model_v18_shoulder_state_activation_context.json"
)
RESULT_OUTPUT = ROOT / "research/model_v18_shoulder_state_result.json"
SOURCE_MANIFEST_DIR = (
    ROOT / "research/model_v18_shoulder_state_source_manifests"
)
SCORE_SESSION_DIR = ROOT / "research/model_v18_shoulder_state_score_sessions"
DECISION_RECORD_DIR = (
    ROOT / "research/model_v18_shoulder_state_decision_records"
)
CHECKPOINT_PROPOSAL_DIR = (
    ROOT / "research/model_v18_shoulder_state_checkpoint_proposals"
)

RUNTIME_LOCK_SHA256 = (
    "2cd701e0ae5969b3a13236908e260c7f072344f5ca477287b14a3ed22b257a54"
)
CALENDAR_SHA256 = (
    "c5c5908b0e26ebd57eb2e473b9d4ce7f92a7c8b336f6ad70596152de971b77a7"
)
PRICE_LOCK_SHA256 = (
    "02370bda9c5fe73b166c557bcdc837d363f5deafbe91baa2d450b33dfcd45272"
)
LEGACY_DAILY_LOCK_SHA256 = (
    "317d5cde9741429263cc91c11575300660f94f62123725fc8ce2d76cdee02eeb"
)
REPLAY_DAILY_LOCK_SHA256 = (
    "1d9a391c8e6b8d2003498c18ba09dac904e672e1f17c05516998ffb71e11c475"
)

TOKYO = ZoneInfo("Asia/Tokyo")
CUTOFF_TIME = time(8, 58, 59)
ANCHOR_THROUGH = date(2026, 8, 4)
NOT_BEFORE_SESSION = date(2026, 8, 6)
FIXED_FORWARD_DATES = (
    date(2026, 7, 28),
    date(2026, 7, 29),
    date(2026, 7, 30),
    date(2026, 7, 31),
    date(2026, 8, 3),
    date(2026, 8, 4),
)

RAW_SOURCE_PROVENANCE_MODE = "manual_operator_attested_v1"
RAW_SOURCE_PROVENANCE_CAVEAT_ID = (
    "manual_jpx_origin_not_independently_verified_v1"
)
RAW_SOURCE_PROVENANCE_CAVEAT = (
    "The operator attests faithful manual acquisition of the labeled official "
    "JPX PDF without omission, substitution, or pre-seal alteration. The "
    "protocol proves post-seal bytes and computation only; it has no "
    "independent server receipt, availability, acquisition-time, origin, or "
    "authenticity proof."
)
RAW_SOURCE_PROVENANCE_CAVEAT_SHA256 = hashlib.sha256(
    RAW_SOURCE_PROVENANCE_CAVEAT.encode("utf-8")
).hexdigest()

DAILY_RE = re.compile(r"^stq_(\d{8})\.pdf$")
WARMUP_RE = re.compile(r"^(\d{6})\.pdf$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
ANCHOR_OBJECT_KEY_RE = re.compile(
    r"^model_v18_shoulder_state/cache-anchor/[A-Za-z0-9._/-]+$"
)
JPX_URL_LABEL_RE = re.compile(
    r"^https://www\.jpx\.co\.jp(?P<path>/[^?#\\]*)$"
)

LAYOUT_FILE = "layout.json"
ACQUISITION_FILE = "predictor_acquisitions.jsonl"
ACQUISITION_LOCK = ".predictor_acquisitions.lock"
OPS_SCHEMA_VERSION = 2

STORE_ARGUMENTS = (
    ("predictor_raw", "--predictor-raw-store-root"),
    ("predictor_derived", "--predictor-derived-store-root"),
    ("outcome_raw", "--outcome-raw-store-root"),
    ("checkpoint_core", "--checkpoint-core-store-root"),
)

ABORT_STAGE_REASONS: Mapping[str, tuple[str, ...]] = {
    "preregistration": ("protocol_mismatch", "canonical_path_conflict"),
    "activation_payload": (
        "activation_evidence_invalid",
        "workflow_evidence_invalid",
        "canonical_path_conflict",
    ),
    "activation_receipt": (
        "activation_evidence_invalid",
        "workflow_evidence_invalid",
        "canonical_path_conflict",
    ),
    "receipt_workflow_observation": (
        "workflow_evidence_invalid",
        "canonical_path_conflict",
    ),
    "pre_count_runtime": ("runtime_lock_mismatch", "canonical_path_conflict"),
    "source_ingestion": (
        "source_integrity_failure",
        "canonical_path_conflict",
    ),
    "monthly_fold": (
        "fold_integrity_failure",
        "runtime_lock_mismatch",
        "canonical_path_conflict",
    ),
    "state_construction": (
        "state_integrity_failure",
        "canonical_path_conflict",
    ),
    "checkpoint_seal": (
        "checkpoint_integrity_failure",
        "workflow_evidence_invalid",
        "runtime_lock_mismatch",
        "canonical_path_conflict",
    ),
    "decision_seal": (
        "decision_integrity_failure",
        "runtime_lock_mismatch",
        "canonical_path_conflict",
    ),
    "outcome_ingestion": (
        "outcome_integrity_failure",
        "runtime_lock_mismatch",
        "canonical_path_conflict",
    ),
    "completed_month_close": (
        "completed_month_integrity_failure",
        "canonical_path_conflict",
    ),
    "terminal_precondition": (
        "terminal_precondition_failure",
        "runtime_lock_mismatch",
        "canonical_path_conflict",
    ),
    "terminal_reconstruction": (
        "terminal_reconstruction_failure",
        "checkpoint_integrity_failure",
        "runtime_lock_mismatch",
        "canonical_path_conflict",
    ),
    "independent_audit": (
        "audit_integrity_failure",
        "checkpoint_integrity_failure",
        "runtime_lock_mismatch",
        "canonical_path_conflict",
    ),
}
ABORT_FAILURE_REASONS = tuple(
    sorted({reason for reasons in ABORT_STAGE_REASONS.values() for reason in reasons})
)


class OpsError(RuntimeError):
    """A fail-closed operational precondition failed."""


@dataclass(frozen=True)
class RegistryItem:
    file_name: str
    kind: str
    source_date: date
    expected_sha256: str | None = None
    expected_bytes: int | None = None
    bound_url: str | None = None


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(value: Any, *, exclude: Iterable[str] = ()) -> str:
    excluded = frozenset(exclude)
    if isinstance(value, Mapping):
        payload: Any = {
            key: item for key, item in value.items() if key not in excluded
        }
    else:
        payload = value
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _read_plain_bytes(
    path: str | Path,
    *,
    signature: bytes | None = None,
    required_mode: int | None = None,
) -> bytes:
    source = Path(path)
    try:
        descriptor = os.open(
            source,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise OpsError(f"plain file cannot be opened: {source}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise OpsError(f"file is not a single-link regular file: {source}")
        if required_mode is not None and (
            before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != required_mode
        ):
            raise OpsError(
                f"file owner/mode differs from operator {required_mode:04o}: {source}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            any(getattr(before, field) != getattr(after, field) for field in stable_fields)
            or len(payload) != before.st_size
            or after.st_nlink != 1
        ):
            raise OpsError(f"file changed while being read: {source}")
        if signature is not None and not payload.startswith(signature):
            raise OpsError(f"file signature is not {signature!r}: {source}")
        return payload
    finally:
        os.close(descriptor)


def _sha256_plain_file(
    path: str | Path,
    *,
    signature: bytes | None = None,
    required_mode: int | None = None,
) -> tuple[int, str]:
    payload = _read_plain_bytes(
        path, signature=signature, required_mode=required_mode
    )
    return len(payload), hashlib.sha256(payload).hexdigest()


def _read_json_authority(path: Path, expected_sha256: str) -> dict[str, Any]:
    payload = _read_plain_bytes(path)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise OpsError(f"registered authority bytes changed: {path.relative_to(ROOT)}")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpsError(f"registered authority is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise OpsError(f"registered authority must be a JSON object: {path}")
    return value


def _read_json_plain(path: Path, *, required_mode: int | None = None) -> dict[str, Any]:
    payload = _read_plain_bytes(path, required_mode=required_mode)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpsError(f"JSON file is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise OpsError(f"JSON file must contain one object: {path}")
    return value


def _parse_timestamp(value: str, label: str) -> datetime:
    token = str(value)
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(token)
    except ValueError as exc:
        raise OpsError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OpsError(f"{label} must include a UTC offset")
    return parsed


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise OpsError(f"{label} must be YYYY-MM-DD") from exc


def _assert_no_symlink_chain(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if not os.path.lexists(current):
            continue
        try:
            observed = current.lstat()
        except OSError as exc:
            raise OpsError(f"path component cannot be inspected: {current}") from exc
        if stat.S_ISLNK(observed.st_mode):
            raise OpsError(f"symlink path component is forbidden: {current}")


def _secure_directory(path: str | Path, *, create: bool) -> Path:
    target = _absolute(path)
    _assert_no_symlink_chain(target)
    if create:
        try:
            target.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise OpsError(f"directory cannot be created: {target}") from exc
        _assert_no_symlink_chain(target)
    try:
        observed = target.lstat()
    except OSError as exc:
        raise OpsError(f"directory is unavailable: {target}") from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        raise OpsError(f"directory must be owner-operated mode 0700: {target}")
    return target.resolve(strict=True)


def _outside_repository(path: Path, label: str) -> None:
    repository = ROOT.resolve(strict=True)
    if path == repository or repository in path.parents:
        raise OpsError(f"{label} must be outside the repository")


def _assert_disjoint_roots(roots: Mapping[str, Path]) -> None:
    values = list(roots.items())
    for index, (left_label, left) in enumerate(values):
        left_stat = os.stat(left, follow_symlinks=False)
        for right_label, right in values[index + 1 :]:
            right_stat = os.stat(right, follow_symlinks=False)
            if (
                left == right
                or left in right.parents
                or right in left.parents
                or (left_stat.st_dev, left_stat.st_ino)
                == (right_stat.st_dev, right_stat.st_ino)
            ):
                raise OpsError(
                    f"{left_label} and {right_label} roots must be disjoint"
                )


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
    except FileExistsError as exc:
        raise OpsError(f"create-once file already exists: {path}") from exc
    except OSError as exc:
        raise OpsError(f"create-once file cannot be opened: {path}") from exc
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OpsError(f"short write: {path}")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    else:
        os.close(descriptor)
        parent_fd = os.open(
            path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)


def _provenance_envelope() -> dict[str, Any]:
    return {
        "mode": RAW_SOURCE_PROVENANCE_MODE,
        "official_source_verified": False,
        "caveat_id": RAW_SOURCE_PROVENANCE_CAVEAT_ID,
        "caveat_text_sha256": RAW_SOURCE_PROVENANCE_CAVEAT_SHA256,
    }


def _root_record(path: Path) -> dict[str, Any]:
    observed = os.stat(path, follow_symlinks=False)
    return {
        "path": str(path),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "mode": stat.S_IMODE(observed.st_mode),
    }


def initialise_layout(
    *,
    ops_root: str | Path,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
    outcome_raw_store_root: str | Path,
    checkpoint_core_store_root: str | Path,
) -> dict[str, Any]:
    operations = _secure_directory(ops_root, create=True)
    roots = {
        "operations": operations,
        "predictor_raw": _secure_directory(
            predictor_raw_store_root, create=True
        ),
        "predictor_derived": _secure_directory(
            predictor_derived_store_root, create=True
        ),
        "outcome_raw": _secure_directory(outcome_raw_store_root, create=True),
        "checkpoint_core": _secure_directory(
            checkpoint_core_store_root, create=True
        ),
    }
    for label, root in roots.items():
        _outside_repository(root, f"{label} root")
    _assert_disjoint_roots(roots)
    for relative in (
        "inbox/price_warmup",
        "inbox/daily",
        "plans",
    ):
        _secure_directory(operations / relative, create=True)
    layout: dict[str, Any] = {
        "schema_version": OPS_SCHEMA_VERSION,
        "operations_root": _root_record(operations),
        "stores": {
            key: _root_record(roots[key])
            for key in (
                "predictor_raw",
                "predictor_derived",
                "outcome_raw",
                "checkpoint_core",
            )
        },
        "repository": str(ROOT.resolve(strict=True)),
        "raw_source_provenance": _provenance_envelope(),
        "raw_evidence_written_by_this_tool": False,
        "runner_execution_enabled": False,
        "github_token_persisted": False,
    }
    layout["layout_sha256"] = _canonical_hash(
        layout, exclude={"layout_sha256"}
    )
    target = operations / LAYOUT_FILE
    payload = _canonical_json_bytes(layout) + b"\n"
    if os.path.lexists(target):
        retained = _load_layout(operations)
        if retained != layout:
            raise OpsError("existing operations layout differs")
    else:
        _write_exclusive(target, payload)
    return layout


def _validate_root_record(
    value: Mapping[str, Any], *, label: str
) -> Path:
    if set(value) != {"path", "device", "inode", "mode"}:
        raise OpsError(f"{label} root record schema differs")
    root = _secure_directory(str(value["path"]), create=False)
    observed = os.stat(root, follow_symlinks=False)
    if (
        int(value["device"]) != observed.st_dev
        or int(value["inode"]) != observed.st_ino
        or int(value["mode"]) != 0o700
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        raise OpsError(f"{label} root identity changed")
    _outside_repository(root, f"{label} root")
    return root


def _load_layout(ops_root: str | Path) -> dict[str, Any]:
    operations = _secure_directory(ops_root, create=False)
    value = _read_json_plain(operations / LAYOUT_FILE, required_mode=0o600)
    required = {
        "schema_version",
        "operations_root",
        "stores",
        "repository",
        "raw_source_provenance",
        "raw_evidence_written_by_this_tool",
        "runner_execution_enabled",
        "github_token_persisted",
        "layout_sha256",
    }
    if set(value) != required or value["schema_version"] != OPS_SCHEMA_VERSION:
        raise OpsError("operations layout schema differs")
    if value["layout_sha256"] != _canonical_hash(
        value, exclude={"layout_sha256"}
    ):
        raise OpsError("operations layout self-hash differs")
    if value["repository"] != str(ROOT.resolve(strict=True)):
        raise OpsError("operations layout belongs to another repository checkout")
    if value["raw_source_provenance"] != _provenance_envelope():
        raise OpsError("operations layout provenance contract changed")
    if (
        value["raw_evidence_written_by_this_tool"] is not False
        or value["runner_execution_enabled"] is not False
        or value["github_token_persisted"] is not False
    ):
        raise OpsError("operations layout safety flags changed")
    root_value = _validate_root_record(
        value["operations_root"], label="operations"
    )
    if root_value != operations:
        raise OpsError("operations layout root moved")
    stores = value["stores"]
    if not isinstance(stores, dict) or set(stores) != {
        "predictor_raw",
        "predictor_derived",
        "outcome_raw",
        "checkpoint_core",
    }:
        raise OpsError("operations layout store registry differs")
    roots = {"operations": operations}
    for label, record in stores.items():
        if not isinstance(record, Mapping):
            raise OpsError(f"{label} root record is invalid")
        roots[label] = _validate_root_record(record, label=label)
    _assert_disjoint_roots(roots)
    return value


def _store_path(layout: Mapping[str, Any], name: str) -> Path:
    try:
        return Path(str(layout["stores"][name]["path"]))
    except (KeyError, TypeError) as exc:
        raise OpsError(f"layout lacks store root: {name}") from exc


def _calendar() -> list[date]:
    payload = _read_plain_bytes(CALENDAR)
    if hashlib.sha256(payload).hexdigest() != CALENDAR_SHA256:
        raise OpsError("registered v1.8 calendar bytes changed")
    try:
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise OpsError("registered calendar cannot be decoded") from exc
    if (
        not rows
        or set(rows[0])
        != {"session_date", "market", "source_url", "source_retrieved_at"}
    ):
        raise OpsError("registered calendar schema differs")
    sessions = [_parse_date(row["session_date"], "calendar session") for row in rows]
    if sessions != sorted(set(sessions)) or any(
        row["market"] != "TSE" for row in rows
    ):
        raise OpsError("registered calendar order/content differs")
    return sessions


def latest_required_source_session(target_session: str | date) -> date:
    target = (
        target_session
        if isinstance(target_session, date)
        else _parse_date(target_session, "session")
    )
    sessions = _calendar()
    try:
        position = sessions.index(target)
    except ValueError as exc:
        raise OpsError("target session is outside the registered calendar") from exc
    return ANCHOR_THROUGH if position == 0 else sessions[position - 1]


def _source_date(file_name: str, kind: str) -> date:
    pattern = WARMUP_RE if kind == "price_warmup" else DAILY_RE
    match = pattern.fullmatch(file_name)
    if match is None:
        raise OpsError(f"invalid {kind} predictor filename: {file_name}")
    token = match.group(1) + ("01" if kind == "price_warmup" else "")
    try:
        return datetime.strptime(token, "%Y%m%d").date()
    except ValueError as exc:
        raise OpsError(f"invalid date in predictor filename: {file_name}") from exc


def _historical_registry() -> dict[str, RegistryItem]:
    price = _read_json_authority(PRICE_LOCK, PRICE_LOCK_SHA256)
    legacy = _read_json_authority(LEGACY_DAILY_LOCK, LEGACY_DAILY_LOCK_SHA256)
    replay = _read_json_authority(REPLAY_DAILY_LOCK, REPLAY_DAILY_LOCK_SHA256)
    records: dict[str, RegistryItem] = {}
    for item in price.get("jpx", {}).get("sources", []):
        name = str(item.get("filename"))
        if name not in {"202505.pdf", "202506.pdf", "202507.pdf"}:
            continue
        records[name] = RegistryItem(
            name,
            "price_warmup",
            _source_date(name, "price_warmup"),
            str(item["sha256"]),
            int(item["bytes"]),
            str(item["source_url"]),
        )
    for item in legacy.get("per_file", []):
        name = str(item.get("name"))
        if DAILY_RE.fullmatch(name) is None:
            raise OpsError("legacy daily registry contains an invalid filename")
        records[name] = RegistryItem(
            name,
            "daily",
            _source_date(name, "daily"),
            str(item["sha256"]),
        )
    for item in replay.get("files", []):
        name = str(item.get("name"))
        if DAILY_RE.fullmatch(name) is None:
            raise OpsError("replay daily registry contains an invalid filename")
        records[name] = RegistryItem(
            name,
            "daily",
            _source_date(name, "daily"),
            str(item["sha256"]),
            int(item["bytes"]),
            str(item["source_url"]),
        )
    if len(records) != 242:
        raise OpsError("historical predictor registry cardinality changed")
    return records


def expected_registry(latest: date) -> list[RegistryItem]:
    records = dict(_historical_registry())
    sessions = _calendar()
    forward = [item for item in FIXED_FORWARD_DATES if item <= latest]
    if latest >= sessions[0]:
        forward.extend(item for item in sessions if item <= latest)
    for source_session in forward:
        name = f"stq_{source_session:%Y%m%d}.pdf"
        records.setdefault(
            name, RegistryItem(name, "daily", source_session)
        )
    warmup = sorted(
        (item for item in records.values() if item.kind == "price_warmup"),
        key=lambda item: (item.source_date, item.file_name),
    )
    daily = sorted(
        (
            item
            for item in records.values()
            if item.kind == "daily" and item.source_date <= latest
        ),
        key=lambda item: (item.source_date, item.file_name),
    )
    return [*warmup, *daily]


def _allowed_registry_item(file_name: str, kind: str) -> RegistryItem:
    historical = _historical_registry()
    if file_name in historical:
        item = historical[file_name]
        if item.kind != kind:
            raise OpsError("predictor kind conflicts with registered history")
        return item
    source_session = _source_date(file_name, kind)
    if kind != "daily":
        raise OpsError("unregistered price warm-up source")
    if source_session not in set(FIXED_FORWARD_DATES) | set(_calendar()):
        raise OpsError("daily source is outside the preregistered session registry")
    return RegistryItem(file_name, kind, source_session)


def _source_url_label(value: str, *, registry: RegistryItem) -> str:
    token = str(value)
    match = JPX_URL_LABEL_RE.fullmatch(token)
    path = "" if match is None else str(match.group("path"))
    if (
        match is None
        or "%" in path
        or ";" in path
        or "//" in path
        or any(part in {".", ".."} for part in path.split("/"))
        or Path(path).name != registry.file_name
    ):
        raise OpsError("source URL label is not a canonical JPX HTTPS label")
    if registry.kind == "daily":
        expected = re.compile(
            r"^/markets/statistics-equities/daily/"
            r"[a-z0-9]+-att/"
            + re.escape(registry.file_name)
            + r"$"
        )
        if expected.fullmatch(path) is None:
            raise OpsError("daily source URL label has a noncanonical path")
    if registry.bound_url is not None and token != registry.bound_url:
        raise OpsError("source URL label differs from bound historical authority")
    return token


ACQUISITION_FIELDS = frozenset(
    {
        "schema_version",
        "sequence_number",
        "previous_record_sha256",
        "provenance_mode",
        "official_source_verified",
        "provenance_caveat_id",
        "provenance_caveat_text_sha256",
        "kind",
        "file_name",
        "source_date",
        "source_url_label",
        "byte_count",
        "sha256",
        "operator_attested_received_at",
        "staged_relative_path",
        "record_sha256",
    }
)


def _decode_jsonl(payload: bytes) -> list[dict[str, Any]]:
    if not payload:
        return []
    if not payload.endswith(b"\n"):
        raise OpsError("predictor acquisition ledger lacks a final newline")
    records: list[dict[str, Any]] = []
    for raw_line in payload.splitlines():
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise OpsError("predictor acquisition ledger contains invalid JSON") from exc
        if not isinstance(value, dict):
            raise OpsError("predictor acquisition record must be an object")
        records.append(value)
    return records


def _validate_acquisitions(
    records: Sequence[Mapping[str, Any]], ops_root: Path
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous = "0" * 64
    for sequence, raw in enumerate(records):
        value = dict(raw)
        if (
            set(value) != ACQUISITION_FIELDS
            or value.get("schema_version") != OPS_SCHEMA_VERSION
            or value.get("sequence_number") != sequence
            or value.get("previous_record_sha256") != previous
        ):
            raise OpsError("predictor acquisition record schema/hash chain differs")
        if (
            value["provenance_mode"] != RAW_SOURCE_PROVENANCE_MODE
            or value["official_source_verified"] is not False
            or value["provenance_caveat_id"] != RAW_SOURCE_PROVENANCE_CAVEAT_ID
            or value["provenance_caveat_text_sha256"]
            != RAW_SOURCE_PROVENANCE_CAVEAT_SHA256
        ):
            raise OpsError("predictor acquisition provenance changed")
        name = str(value["file_name"])
        kind = str(value["kind"])
        registry = _allowed_registry_item(name, kind)
        if name in seen or value["source_date"] != registry.source_date.isoformat():
            raise OpsError("predictor acquisition file/date registry differs")
        seen.add(name)
        label = _source_url_label(
            str(value["source_url_label"]), registry=registry
        )
        del label
        received = _parse_timestamp(
            str(value["operator_attested_received_at"]),
            "operator-attested receipt",
        )
        if received > datetime.now(timezone.utc):
            raise OpsError("operator-attested receipt is in the future")
        if (
            not isinstance(value["byte_count"], int)
            or isinstance(value["byte_count"], bool)
            or value["byte_count"] <= 0
            or SHA256_RE.fullmatch(str(value["sha256"])) is None
        ):
            raise OpsError("predictor acquisition byte metadata is invalid")
        if (
            registry.expected_sha256 is not None
            and value["sha256"] != registry.expected_sha256
        ):
            raise OpsError("predictor acquisition differs from historical SHA")
        if (
            registry.expected_bytes is not None
            and value["byte_count"] != registry.expected_bytes
        ):
            raise OpsError("predictor acquisition differs from historical size")
        relative = Path(str(value["staged_relative_path"]))
        expected_relative = Path("inbox") / kind / name
        if relative.is_absolute() or relative.parts != expected_relative.parts:
            raise OpsError("predictor staged relative path differs")
        observed = _sha256_plain_file(
            ops_root / relative,
            signature=b"%PDF",
            required_mode=0o600,
        )
        if observed != (value["byte_count"], value["sha256"]):
            raise OpsError("predictor staged bytes differ from acquisition record")
        expected_hash = _canonical_hash(value, exclude={"record_sha256"})
        if value["record_sha256"] != expected_hash:
            raise OpsError("predictor acquisition self-hash differs")
        previous = expected_hash
        validated.append(value)
    return validated


def load_acquisitions(ops_root: str | Path) -> list[dict[str, Any]]:
    operations = _secure_directory(ops_root, create=False)
    _load_layout(operations)
    path = operations / ACQUISITION_FILE
    if not os.path.lexists(path):
        return []
    payload = _read_plain_bytes(path, required_mode=0o600)
    return _validate_acquisitions(_decode_jsonl(payload), operations)


def _copy_create_once(source: Path, destination: Path) -> tuple[int, str]:
    supplied_payload = _read_plain_bytes(source, signature=b"%PDF")
    supplied = (len(supplied_payload), hashlib.sha256(supplied_payload).hexdigest())
    if os.path.lexists(destination):
        retained = _sha256_plain_file(
            destination, signature=b"%PDF", required_mode=0o600
        )
        if retained != supplied:
            raise OpsError("create-once predictor staging object conflicts")
        return retained
    _write_exclusive(destination, supplied_payload)
    retained = _sha256_plain_file(
        destination, signature=b"%PDF", required_mode=0o600
    )
    if retained != supplied:
        raise OpsError("predictor staging bytes changed after publication")
    return retained


def register_source(
    *,
    ops_root: str | Path,
    source_file: str | Path,
    source_url: str,
    received_at: str,
    kind: str,
) -> dict[str, Any]:
    operations = _secure_directory(ops_root, create=False)
    _load_layout(operations)
    if kind not in {"price_warmup", "daily"}:
        raise OpsError("predictor kind must be price_warmup or daily")
    source = _absolute(source_file)
    registry = _allowed_registry_item(source.name, kind)
    url_label = _source_url_label(source_url, registry=registry)
    received = _parse_timestamp(received_at, "received_at")
    if received > datetime.now(timezone.utc):
        raise OpsError("received_at is in the future")
    canonical_received = _timestamp_text(received)
    supplied_count, supplied_digest = _sha256_plain_file(
        source, signature=b"%PDF"
    )
    if (
        registry.expected_sha256 is not None
        and supplied_digest != registry.expected_sha256
    ):
        raise OpsError("source bytes differ from bound historical SHA authority")
    if (
        registry.expected_bytes is not None
        and supplied_count != registry.expected_bytes
    ):
        raise OpsError("source bytes differ from bound historical size authority")
    lock_path = operations / ACQUISITION_LOCK
    lock_fd = os.open(
        lock_path,
        os.O_RDWR
        | os.O_CREAT
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        lock_stat = os.fstat(lock_fd)
        if (
            not stat.S_ISREG(lock_stat.st_mode)
            or lock_stat.st_nlink != 1
            or lock_stat.st_uid != os.geteuid()
            or stat.S_IMODE(lock_stat.st_mode) != 0o600
        ):
            raise OpsError("predictor acquisition lock is unsafe")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        ledger_path = operations / ACQUISITION_FILE
        payload = (
            _read_plain_bytes(ledger_path, required_mode=0o600)
            if os.path.lexists(ledger_path)
            else b""
        )
        records = _validate_acquisitions(
            _decode_jsonl(payload), operations
        )
        by_name = {str(item["file_name"]): item for item in records}
        destination = operations / "inbox" / kind / source.name
        byte_count, digest = _copy_create_once(source, destination)
        relative = destination.relative_to(operations).as_posix()
        existing = by_name.get(source.name)
        if existing is not None:
            expected = {
                "kind": kind,
                "source_url_label": url_label,
                "byte_count": byte_count,
                "sha256": digest,
                "operator_attested_received_at": canonical_received,
                "staged_relative_path": relative,
            }
            if any(existing[field] != value for field, value in expected.items()):
                raise OpsError("existing predictor acquisition record conflicts")
            return existing
        record: dict[str, Any] = {
            "schema_version": OPS_SCHEMA_VERSION,
            "sequence_number": len(records),
            "previous_record_sha256": (
                records[-1]["record_sha256"] if records else "0" * 64
            ),
            "provenance_mode": RAW_SOURCE_PROVENANCE_MODE,
            "official_source_verified": False,
            "provenance_caveat_id": RAW_SOURCE_PROVENANCE_CAVEAT_ID,
            "provenance_caveat_text_sha256": (
                RAW_SOURCE_PROVENANCE_CAVEAT_SHA256
            ),
            "kind": kind,
            "file_name": source.name,
            "source_date": registry.source_date.isoformat(),
            "source_url_label": url_label,
            "byte_count": byte_count,
            "sha256": digest,
            "operator_attested_received_at": canonical_received,
            "staged_relative_path": relative,
        }
        record["record_sha256"] = _canonical_hash(
            record, exclude={"record_sha256"}
        )
        line = _canonical_json_bytes(record) + b"\n"
        ledger_fd = os.open(
            ledger_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_APPEND
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            observed = os.fstat(ledger_fd)
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or observed.st_uid != os.geteuid()
                or stat.S_IMODE(observed.st_mode) != 0o600
            ):
                raise OpsError("predictor acquisition ledger is unsafe")
            remaining = memoryview(line)
            while remaining:
                written = os.write(ledger_fd, remaining)
                if written <= 0:
                    raise OpsError("short predictor acquisition ledger write")
                remaining = remaining[written:]
            os.fsync(ledger_fd)
        finally:
            os.close(ledger_fd)
        return record
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(lock_fd)


def _runtime_launch_contract() -> tuple[str, list[str]]:
    lock = _read_json_authority(RUNTIME_LOCK, RUNTIME_LOCK_SHA256)
    try:
        loader = lock["elf_closure"]["loader_environment"]
        expected_executable_hash = lock["runtime"]["python"]["executable_sha256"]
    except (KeyError, TypeError) as exc:
        raise OpsError("runtime lock launch contract is missing") from exc
    if (
        not isinstance(loader, Mapping)
        or not loader
        or any(value is not None for value in loader.values())
    ):
        raise OpsError("runtime lock loader environment contract changed")
    names = sorted(str(name) for name in loader)
    if len(names) != len(set(names)) or "GITHUB_TOKEN" in names:
        raise OpsError("runtime lock loader environment names are invalid")
    executable = Path(
        str(getattr(sys, "_base_executable", sys.executable))
    ).resolve(strict=True)
    _, digest = _sha256_plain_file(executable)
    if digest != expected_executable_hash:
        raise OpsError("current Python executable differs from runtime lock")
    return str(executable), names


def _runner_argv(*arguments: str) -> list[str]:
    executable, _ = _runtime_launch_contract()
    return [
        executable,
        "-B",
        str(RUNNER.relative_to(ROOT)),
        *arguments,
    ]


def _plan_step(
    step_id: str,
    argv: Sequence[str],
    *,
    local_mutation: bool,
    remote_git_mutation: bool = False,
    retry_policy: str,
) -> dict[str, Any]:
    _, environment_unset = _runtime_launch_contract()
    return {
        "step_id": step_id,
        "working_directory": str(ROOT.resolve(strict=True)),
        "environment_unset": environment_unset,
        "argv": list(argv),
        "executes_when_planned": False,
        "local_authority_mutation": local_mutation,
        "remote_git_mutation": remote_git_mutation,
        "retry_policy": retry_policy,
        "environment_values_persisted": False,
        "github_token_value_persisted": False,
    }


def _store_arguments(layout: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for name, flag in STORE_ARGUMENTS:
        result.extend([flag, str(_store_path(layout, name))])
    return result


def _predictor_store_arguments(layout: Mapping[str, Any]) -> list[str]:
    return [
        "--predictor-raw-store-root",
        str(_store_path(layout, "predictor_raw")),
        "--predictor-derived-store-root",
        str(_store_path(layout, "predictor_derived")),
    ]


def _base_plan(
    *,
    layout: Mapping[str, Any],
    scope: str,
    steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": OPS_SCHEMA_VERSION,
        "scope": scope,
        "layout_sha256": layout["layout_sha256"],
        "runner_path": str(RUNNER.relative_to(ROOT)),
        "runner_sha256": _sha256_plain_file(RUNNER)[1],
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "raw_source_provenance": _provenance_envelope(),
        "steps": [dict(item) for item in steps],
        "runner_execution_enabled": False,
        "scientific_authority": False,
        "github_token_value_persisted": False,
    }


def _persist_plan(
    operations: Path, file_name: str, plan: dict[str, Any]
) -> dict[str, Any]:
    value = dict(plan)
    value["plan_sha256"] = _canonical_hash(value, exclude={"plan_sha256"})
    payload = _canonical_json_bytes(value) + b"\n"
    target = operations / "plans" / file_name
    if os.path.lexists(target):
        retained = _read_plain_bytes(target, required_mode=0o600)
        if retained != payload:
            raise OpsError("existing plan conflicts with exact retry")
    else:
        _write_exclusive(target, payload)
    return value


def build_store_plan(*, ops_root: str | Path) -> dict[str, Any]:
    operations = _secure_directory(ops_root, create=False)
    layout = _load_layout(operations)
    steps = [
        _plan_step(
            "validate_runtime",
            _runner_argv("validate-runtime"),
            local_mutation=False,
            retry_policy="safe_repeat",
        ),
        _plan_step(
            "prepare_operational_stores",
            _runner_argv(
                "prepare-operational-stores", *_store_arguments(layout)
            ),
            local_mutation=True,
            retry_policy="same_roots_preactivation_retry",
        ),
    ]
    return _persist_plan(
        operations,
        "preactivation-stores.json",
        _base_plan(
            layout=layout,
            scope="preactivation_store_readiness_only",
            steps=steps,
        ),
    )


def _records_by_name(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["file_name"]): dict(item) for item in records}


def build_predictor_cache_plan(*, ops_root: str | Path) -> dict[str, Any]:
    operations = _secure_directory(ops_root, create=False)
    layout = _load_layout(operations)
    records = _records_by_name(load_acquisitions(operations))
    expected = expected_registry(ANCHOR_THROUGH)
    missing = [item.file_name for item in expected if item.file_name not in records]
    if missing:
        raise OpsError(
            f"predictor anchor manual inventory is incomplete: {missing}"
        )
    ordered = [records[item.file_name] for item in expected]
    source_arguments: list[str] = []
    for item in ordered:
        source_arguments.extend(
            [
                "--source-pdf",
                str(operations / str(item["staged_relative_path"])),
                "--source-file-name",
                str(item["file_name"]),
                "--source-url",
                str(item["source_url_label"]),
            ]
        )
    steps = [
        _plan_step(
            "validate_runtime",
            _runner_argv("validate-runtime"),
            local_mutation=False,
            retry_policy="safe_repeat",
        ),
        _plan_step(
            "prepare_predictor_cache",
            _runner_argv(
                "prepare-predictor-cache",
                *source_arguments,
                "--through",
                ANCHOR_THROUGH.isoformat(),
                *_predictor_store_arguments(layout),
            ),
            local_mutation=True,
            retry_policy="runner_create_once_recovery_only",
        ),
    ]
    plan = _base_plan(
        layout=layout,
        scope="preactivation_predictor_cache_anchor_only",
        steps=steps,
    )
    plan["anchor_through"] = ANCHOR_THROUGH.isoformat()
    plan["anchor_source_count"] = len(ordered)
    return _persist_plan(operations, "predictor-cache-anchor.json", plan)


def _require_git_sha(value: str | None, label: str) -> str:
    token = "" if value is None else str(value)
    if GIT_SHA_RE.fullmatch(token) is None:
        raise OpsError(f"{label} must be a full 40-lowercase-hex Git SHA")
    return token


def _require_anchor_key(value: str | None) -> str:
    token = "" if value is None else str(value)
    if (
        ANCHOR_OBJECT_KEY_RE.fullmatch(token) is None
        or token.startswith("/")
        or ".." in Path(token).parts
    ):
        raise OpsError("predictor cache anchor object key is invalid")
    return token


def build_activation_plan(
    *,
    ops_root: str | Path,
    stage: str,
    commit_sha: str | None = None,
    predictor_cache_anchor_manifest_object_key: str | None = None,
) -> dict[str, Any]:
    operations = _secure_directory(ops_root, create=False)
    layout = _load_layout(operations)
    if stage == "payload":
        command = _runner_argv(
            "create-activation-payload",
            "--preregistration-commit-sha",
            _require_git_sha(commit_sha, "preregistration commit"),
            *_predictor_store_arguments(layout),
            "--predictor-cache-anchor-manifest-object-key",
            _require_anchor_key(
                predictor_cache_anchor_manifest_object_key
            ),
        )
        step_id = "create_activation_payload"
    elif stage == "receipt":
        command = _runner_argv(
            "create-activation-receipt",
            "--payload-commit-sha",
            _require_git_sha(commit_sha, "payload commit B"),
        )
        step_id = "create_activation_receipt"
    elif stage == "preflight":
        command = _runner_argv(
            "preflight",
            "--receipt-commit-sha",
            _require_git_sha(commit_sha, "receipt commit C"),
        )
        step_id = "create_canonical_activation_context"
    else:
        raise OpsError("activation stage must be payload, receipt, or preflight")
    steps = [
        _plan_step(
            "validate_runtime",
            _runner_argv("validate-runtime"),
            local_mutation=False,
            retry_policy="safe_repeat",
        ),
        _plan_step(
            step_id,
            command,
            local_mutation=True,
            retry_policy="same_commit_and_authority_exact_retry",
        ),
    ]
    plan = _base_plan(
        layout=layout,
        scope=f"activation_{stage}_canonical_paths_only",
        steps=steps,
    )
    plan["activation_stage"] = stage
    plan["canonical_activation_payload"] = str(ACTIVATION_PAYLOAD.relative_to(ROOT))
    plan["canonical_activation_receipt"] = str(ACTIVATION_RECEIPT.relative_to(ROOT))
    plan["canonical_activation_context"] = str(ACTIVATION_CONTEXT.relative_to(ROOT))
    return _persist_plan(operations, f"activation-{stage}.json", plan)


def _load_activation_context() -> dict[str, Any]:
    if _absolute(ACTIVATION_CONTEXT) != ACTIVATION_CONTEXT:
        raise OpsError("canonical activation context constant is not absolute")
    value = _read_json_plain(ACTIVATION_CONTEXT)
    required = {
        "first_counted_session",
        "terminal_session",
        "activation_payload_sha256",
        "activation_receipt_sha256",
    }
    if not required.issubset(value):
        raise OpsError("canonical activation context lacks required fields")
    first = _parse_date(value["first_counted_session"], "first counted session")
    terminal = _parse_date(value["terminal_session"], "terminal session")
    sessions = _calendar()
    if (
        first < NOT_BEFORE_SESSION
        or first not in sessions
        or terminal not in sessions
        or terminal < first
        or SHA256_RE.fullmatch(str(value["activation_payload_sha256"])) is None
        or SHA256_RE.fullmatch(str(value["activation_receipt_sha256"])) is None
    ):
        raise OpsError("canonical activation context calendar/hash fields differ")
    return value


def _incremental_daily_names(
    *, target: date, first_counted: date
) -> list[str]:
    sessions = _calendar()
    if target not in sessions or first_counted not in sessions:
        raise OpsError("daily target/activation is outside the registered calendar")
    if target < first_counted:
        raise OpsError("daily target predates activation")
    latest = latest_required_source_session(target)
    if target == first_counted:
        source_sessions = [
            item for item in sessions if ANCHOR_THROUGH < item <= latest
        ]
    else:
        source_sessions = [latest]
    if not source_sessions:
        raise OpsError("daily incremental suffix is empty")
    return [f"stq_{item:%Y%m%d}.pdf" for item in source_sessions]


def _cutoff(session: date) -> datetime:
    return datetime.combine(session, CUTOFF_TIME, TOKYO)


def _validate_as_of(value: str, *, cutoff: datetime | None) -> datetime:
    observed = _parse_timestamp(value, "as_of")
    if observed > datetime.now(timezone.utc):
        raise OpsError("as_of is in the future")
    if cutoff is not None and observed.astimezone(TOKYO) > cutoff:
        raise OpsError("plan cannot be created after the decision cutoff")
    return observed


def _daily_records(
    operations: Path,
    *,
    target: date,
    context: Mapping[str, Any],
    as_of: datetime,
) -> list[dict[str, Any]]:
    first = _parse_date(context["first_counted_session"], "first counted session")
    names = _incremental_daily_names(target=target, first_counted=first)
    by_name = _records_by_name(load_acquisitions(operations))
    missing = [name for name in names if name not in by_name]
    if missing:
        raise OpsError(
            f"manual daily source suffix is incomplete: {missing}; "
            "do not generate cash/publish/decide"
        )
    records = [by_name[name] for name in names]
    cutoff = _cutoff(target)
    for item in records:
        received = _parse_timestamp(
            item["operator_attested_received_at"],
            "operator-attested source receipt",
        )
        if received > as_of:
            raise OpsError("daily source inventory contains a post-as_of receipt")
        if received.astimezone(TOKYO) > cutoff:
            raise OpsError("daily source receipt is later than the decision cutoff")
    return records


def build_daily_plan(
    *,
    ops_root: str | Path,
    session: str,
    as_of: str,
) -> dict[str, Any]:
    operations = _secure_directory(ops_root, create=False)
    layout = _load_layout(operations)
    context = _load_activation_context()
    target = _parse_date(session, "session")
    first = _parse_date(context["first_counted_session"], "first counted session")
    terminal = _parse_date(context["terminal_session"], "terminal session")
    if target < first or target > terminal:
        raise OpsError("daily target is outside the activated denominator")
    observed = _validate_as_of(as_of, cutoff=_cutoff(target))
    records = _daily_records(
        operations, target=target, context=context, as_of=observed
    )
    source_arguments: list[str] = []
    for item in records:
        source_arguments.extend(
            [
                "--new-predictor-pdf",
                str(operations / str(item["staged_relative_path"])),
                "--new-predictor-file-name",
                str(item["file_name"]),
                "--new-predictor-url",
                str(item["source_url_label"]),
                "--new-predictor-received-at",
                str(item["operator_attested_received_at"]),
            ]
        )
    steps = [
        _plan_step(
            "validate_runtime",
            _runner_argv("validate-runtime"),
            local_mutation=False,
            retry_policy="safe_repeat",
        ),
        _plan_step(
            "prepare_day",
            _runner_argv(
                "prepare-day",
                "--session",
                target.isoformat(),
                *source_arguments,
                *_store_arguments(layout),
            ),
            local_mutation=True,
            retry_policy="same_inputs_exact_retry",
        ),
        _plan_step(
            "publish_checkpoint_once",
            _runner_argv(
                "publish-checkpoint", "--session", target.isoformat()
            ),
            local_mutation=False,
            remote_git_mutation=True,
            retry_policy="no_blind_retry_after_possible_ref_update",
        ),
        _plan_step(
            "materialize_decision",
            _runner_argv(
                "decide",
                "--session",
                target.isoformat(),
                "--checkpoint-core-store-root",
                str(_store_path(layout, "checkpoint_core")),
            ),
            local_mutation=True,
            retry_policy="same_session_exact_retry_or_read_only_remote_recovery",
        ),
    ]
    plan = _base_plan(
        layout=layout,
        scope="activated_daily_prepare_publish_decide_only",
        steps=steps,
    )
    plan.update(
        {
            "target_session": target.isoformat(),
            "first_counted_session": first.isoformat(),
            "terminal_session": terminal.isoformat(),
            "latest_required_source_session": (
                latest_required_source_session(target).isoformat()
            ),
            "incremental_source_files": [
                str(item["file_name"]) for item in records
            ],
            "incremental_source_count": len(records),
            "as_of": _timestamp_text(observed),
            "decision_cutoff": _cutoff(target).isoformat(),
            "source_failure_cash_path_present": False,
            "terminal_operations_included": False,
            "publication_retry_allowed": False,
            "plan_is_not_cutoff_or_execution_authority": True,
        }
    )
    return _persist_plan(operations, f"daily-{target}.json", plan)


def build_terminal_plan(
    *,
    ops_root: str | Path,
    session: str,
    as_of: str,
) -> dict[str, Any]:
    operations = _secure_directory(ops_root, create=False)
    layout = _load_layout(operations)
    context = _load_activation_context()
    target = _parse_date(session, "terminal session")
    terminal = _parse_date(context["terminal_session"], "terminal session")
    if target != terminal:
        raise OpsError("terminal plan session differs from activation context")
    observed = _validate_as_of(as_of, cutoff=None)
    file_name = f"stq_{target:%Y%m%d}.pdf"
    record = _records_by_name(load_acquisitions(operations)).get(file_name)
    if record is None:
        raise OpsError("manually registered terminal outcome source is missing")
    received = _parse_timestamp(
        record["operator_attested_received_at"],
        "terminal outcome receipt",
    )
    if received > observed:
        raise OpsError("terminal source inventory contains a post_as_of receipt")
    steps = [
        _plan_step(
            "validate_runtime",
            _runner_argv("validate-runtime"),
            local_mutation=False,
            retry_policy="safe_repeat",
        ),
        _plan_step(
            "finalize_terminal",
            _runner_argv(
                "finalize-terminal",
                "--source-pdf",
                str(operations / str(record["staged_relative_path"])),
                "--source-file-name",
                file_name,
                "--source-url",
                str(record["source_url_label"]),
                "--source-received-at",
                str(record["operator_attested_received_at"]),
                *_store_arguments(layout),
            ),
            local_mutation=True,
            retry_policy="same_terminal_source_exact_retry",
        ),
        _plan_step(
            "evaluate",
            _runner_argv(
                "evaluate",
                *_store_arguments(layout),
            ),
            local_mutation=True,
            retry_policy="same_canonical_authorities_exact_retry",
        ),
    ]
    plan = _base_plan(
        layout=layout,
        scope="terminal_finalize_then_evaluate_only",
        steps=steps,
    )
    plan.update(
        {
            "terminal_session": target.isoformat(),
            "terminal_source_file": file_name,
            "as_of": _timestamp_text(observed),
            "terminal_finalize_precedes_evaluate": True,
            "activation_context_argument_overridden": False,
        }
    )
    return _persist_plan(operations, f"terminal-{target}.json", plan)


def build_abort_plan(
    *,
    ops_root: str | Path,
    failure_reason: str,
    integrity_stage: str,
) -> dict[str, Any]:
    operations = _secure_directory(ops_root, create=False)
    layout = _load_layout(operations)
    _load_activation_context()
    allowed = ABORT_STAGE_REASONS.get(integrity_stage)
    if allowed is None or failure_reason not in allowed:
        raise OpsError("abort failure reason is not allowed for its integrity stage")
    steps = [
        _plan_step(
            "abort_integrity_failure",
            _runner_argv(
                "abort",
                "--failure-reason",
                failure_reason,
                "--integrity-stage",
                integrity_stage,
            ),
            local_mutation=True,
            retry_policy="same_abort_reason_and_stage_exact_retry",
        )
    ]
    plan = _base_plan(
        layout=layout,
        scope="outcome_blind_global_integrity_abort_only",
        steps=steps,
    )
    plan.update(
        {
            "failure_reason": failure_reason,
            "integrity_stage": integrity_stage,
            "store_arguments_included": False,
            "performance_paths_included": False,
            "activation_context_argument_overridden": False,
        }
    )
    return _persist_plan(
        operations,
        f"abort-{integrity_stage}-{failure_reason}.json",
        plan,
    )


def source_health(
    *,
    ops_root: str | Path,
    phase: str,
    session: str | None = None,
) -> dict[str, Any]:
    operations = _secure_directory(ops_root, create=False)
    _load_layout(operations)
    records = _records_by_name(load_acquisitions(operations))
    cutoff: datetime | None = None
    if phase == "anchor":
        names = [item.file_name for item in expected_registry(ANCHOR_THROUGH)]
        target: str | None = None
    elif phase == "daily":
        if session is None:
            raise OpsError("daily source health requires --session")
        context = _load_activation_context()
        target_date = _parse_date(session, "session")
        names = _incremental_daily_names(
            target=target_date,
            first_counted=_parse_date(
                context["first_counted_session"], "first counted session"
            ),
        )
        cutoff = _cutoff(target_date)
        target = target_date.isoformat()
    elif phase == "terminal":
        context = _load_activation_context()
        terminal = _parse_date(
            context["terminal_session"], "terminal session"
        )
        if session is not None and _parse_date(session, "session") != terminal:
            raise OpsError("terminal health session differs from activation")
        names = [f"stq_{terminal:%Y%m%d}.pdf"]
        target = terminal.isoformat()
    else:
        raise OpsError("source health phase must be anchor, daily, or terminal")
    missing = [name for name in names if name not in records]
    late: list[str] = []
    if cutoff is not None:
        late = [
            name
            for name in names
            if name in records
            and _parse_timestamp(
                records[name]["operator_attested_received_at"], "source receipt"
            ).astimezone(TOKYO)
            > cutoff
        ]
    return {
        "schema_version": OPS_SCHEMA_VERSION,
        "phase": phase,
        "target_session": target,
        "required_source_count": len(names),
        "present_source_count": len(names) - len(missing),
        "missing_source_count": len(missing),
        "late_source_count": len(late),
        "missing_source_files": missing,
        "late_source_files": late,
        "source_complete": not missing and not late,
        "raw_source_provenance": _provenance_envelope(),
        "scientific_authority": False,
        "performance_semantics_read": False,
    }


def _presence(path: Path, *, expected_kind: str) -> dict[str, Any]:
    if not os.path.lexists(path):
        return {
            "exists": False,
            "plain_expected_kind": None,
            "mode": None,
            "nlink": None,
            "size": None,
        }
    try:
        observed = path.lstat()
    except OSError as exc:
        raise OpsError(f"health path cannot be inspected: {path}") from exc
    plain = (
        stat.S_ISREG(observed.st_mode)
        if expected_kind == "file"
        else stat.S_ISDIR(observed.st_mode)
    )
    return {
        "exists": True,
        "plain_expected_kind": plain and not stat.S_ISLNK(observed.st_mode),
        "mode": stat.S_IMODE(observed.st_mode),
        "nlink": int(observed.st_nlink),
        "size": int(observed.st_size),
    }


def health_report(
    *, ops_root: str | Path, session: str
) -> dict[str, Any]:
    operations = _secure_directory(ops_root, create=False)
    layout = _load_layout(operations)
    target = _parse_date(session, "session")
    source = source_health(
        ops_root=operations, phase="daily", session=target.isoformat()
    )
    exact_paths = {
        "activation_payload": (ACTIVATION_PAYLOAD, "file"),
        "activation_receipt": (ACTIVATION_RECEIPT, "file"),
        "activation_context": (ACTIVATION_CONTEXT, "file"),
        "target_source_manifest": (
            SOURCE_MANIFEST_DIR / f"{target}.json",
            "file",
        ),
        "target_score_session": (
            SCORE_SESSION_DIR / f"{target}.csv",
            "file",
        ),
        "target_decision_record": (
            DECISION_RECORD_DIR / f"{target}.json",
            "file",
        ),
        "target_checkpoint_proposal": (
            CHECKPOINT_PROPOSAL_DIR / str(target),
            "directory",
        ),
        "terminal_result": (RESULT_OUTPUT, "file"),
    }
    canonical = {
        name: _presence(path, expected_kind=kind)
        for name, (path, kind) in exact_paths.items()
    }
    stores = {
        name: _presence(
            _store_path(layout, name), expected_kind="directory"
        )
        for name in (
            "predictor_raw",
            "predictor_derived",
            "outcome_raw",
            "checkpoint_core",
        )
    }
    return {
        "schema_version": OPS_SCHEMA_VERSION,
        "scope": "referenced_source_and_exact_path_metadata_only",
        "target_session": target.isoformat(),
        "source_health": source,
        "canonical_exact_path_presence": canonical,
        "external_root_presence": stores,
        "manual_staged_sources_validated": True,
        "external_store_children_enumerated": False,
        "scientific_artifact_bytes_read": False,
        "performance_semantics_read": False,
        "outcome_paths_opened": False,
        "runner_executed": False,
    }


def _shell_plan(plan: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for step in plan["steps"]:
        prefix = ["/usr/bin/env"]
        for name in step["environment_unset"]:
            prefix.extend(["-u", str(name)])
        command = [*prefix, *[str(item) for item in step["argv"]]]
        lines.append(f"# {step['step_id']}")
        lines.append(shlex.join(command))
    return "\n".join(lines) + "\n"


def _add_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("json", "shell"), default="json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialise = subparsers.add_parser(
        "init", help="create the operations root and four disjoint store roots"
    )
    initialise.add_argument("--ops-root", required=True)
    initialise.add_argument("--predictor-raw-store-root", required=True)
    initialise.add_argument("--predictor-derived-store-root", required=True)
    initialise.add_argument("--outcome-raw-store-root", required=True)
    initialise.add_argument("--checkpoint-core-store-root", required=True)

    register = subparsers.add_parser(
        "register-source",
        help="manually stage one operator-attested JPX PDF label/byte binding",
    )
    register.add_argument("--ops-root", required=True)
    register.add_argument("--source-file", required=True)
    register.add_argument("--source-url", required=True)
    register.add_argument("--received-at", required=True)
    register.add_argument(
        "--kind", choices=("price_warmup", "daily"), required=True
    )

    inventory = subparsers.add_parser(
        "source-health", help="check only manually registered source readiness"
    )
    inventory.add_argument("--ops-root", required=True)
    inventory.add_argument(
        "--phase", choices=("anchor", "daily", "terminal"), required=True
    )
    inventory.add_argument("--session")

    health = subparsers.add_parser(
        "health",
        help="inspect referenced manual sources and exact canonical path metadata",
    )
    health.add_argument("--ops-root", required=True)
    health.add_argument("--session", required=True)

    stores = subparsers.add_parser(
        "plan-stores", help="plan validate-runtime and store readiness"
    )
    stores.add_argument("--ops-root", required=True)
    _add_format(stores)

    cache = subparsers.add_parser(
        "plan-cache", help="plan the fixed 2026-08-04 predictor cache anchor"
    )
    cache.add_argument("--ops-root", required=True)
    _add_format(cache)

    activation = subparsers.add_parser(
        "plan-activation", help="plan canonical payload, receipt, or preflight"
    )
    activation.add_argument("--ops-root", required=True)
    activation.add_argument(
        "--stage", choices=("payload", "receipt", "preflight"), required=True
    )
    activation.add_argument("--commit-sha")
    activation.add_argument(
        "--predictor-cache-anchor-manifest-object-key"
    )
    _add_format(activation)

    daily = subparsers.add_parser(
        "plan-day", help="plan prepare-day, publish-checkpoint, and decide"
    )
    daily.add_argument("--ops-root", required=True)
    daily.add_argument("--session", required=True)
    daily.add_argument("--as-of", required=True)
    _add_format(daily)

    terminal = subparsers.add_parser(
        "plan-terminal", help="plan finalize-terminal followed by evaluate"
    )
    terminal.add_argument("--ops-root", required=True)
    terminal.add_argument("--session", required=True)
    terminal.add_argument("--as-of", required=True)
    _add_format(terminal)

    abort = subparsers.add_parser(
        "plan-abort", help="plan one outcome-blind global integrity abort"
    )
    abort.add_argument("--ops-root", required=True)
    abort.add_argument("--failure-reason", choices=ABORT_FAILURE_REASONS, required=True)
    abort.add_argument(
        "--integrity-stage", choices=tuple(ABORT_STAGE_REASONS), required=True
    )
    _add_format(abort)
    return parser


def _emit_plan(plan: Mapping[str, Any], format_name: str) -> None:
    if format_name == "shell":
        print(_shell_plan(plan), end="")
    else:
        print(
            json.dumps(
                plan,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "init":
        result = initialise_layout(
            ops_root=args.ops_root,
            predictor_raw_store_root=args.predictor_raw_store_root,
            predictor_derived_store_root=args.predictor_derived_store_root,
            outcome_raw_store_root=args.outcome_raw_store_root,
            checkpoint_core_store_root=args.checkpoint_core_store_root,
        )
    elif args.command == "register-source":
        result = register_source(
            ops_root=args.ops_root,
            source_file=args.source_file,
            source_url=args.source_url,
            received_at=args.received_at,
            kind=args.kind,
        )
    elif args.command == "source-health":
        result = source_health(
            ops_root=args.ops_root,
            phase=args.phase,
            session=args.session,
        )
    elif args.command == "health":
        result = health_report(
            ops_root=args.ops_root,
            session=args.session,
        )
    elif args.command == "plan-stores":
        result = build_store_plan(ops_root=args.ops_root)
        _emit_plan(result, args.format)
        return 0
    elif args.command == "plan-cache":
        result = build_predictor_cache_plan(ops_root=args.ops_root)
        _emit_plan(result, args.format)
        return 0
    elif args.command == "plan-activation":
        result = build_activation_plan(
            ops_root=args.ops_root,
            stage=args.stage,
            commit_sha=args.commit_sha,
            predictor_cache_anchor_manifest_object_key=(
                args.predictor_cache_anchor_manifest_object_key
            ),
        )
        _emit_plan(result, args.format)
        return 0
    elif args.command == "plan-day":
        result = build_daily_plan(
            ops_root=args.ops_root,
            session=args.session,
            as_of=args.as_of,
        )
        _emit_plan(result, args.format)
        return 0
    elif args.command == "plan-terminal":
        result = build_terminal_plan(
            ops_root=args.ops_root,
            session=args.session,
            as_of=args.as_of,
        )
        _emit_plan(result, args.format)
        return 0
    elif args.command == "plan-abort":
        result = build_abort_plan(
            ops_root=args.ops_root,
            failure_reason=args.failure_reason,
            integrity_stage=args.integrity_stage,
        )
        _emit_plan(result, args.format)
        return 0
    else:  # pragma: no cover
        raise OpsError(f"unknown command: {args.command}")
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OpsError as exc:
        print(f"v1.8 ops fail-closed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
