#!/usr/bin/env python3
"""Bound, local-only A2 reference and cold-rehearsal driver.

This module is deliberately not an operational runner.  It accepts the exact
documentary 248-PDF registry, builds a nonauthority raw/shard reference under a
fresh private ``/tmp`` root, and runs the two protocol-fixed compact cases
against the runner's pure rehearsal seam.  It cannot create or validate an
activation, decision, checkpoint, outcome, result, or publication artifact.

Invoke this file directly with the runtime-locked Python and ``-B``.  Successful
commands print one canonical, health-only JSON object; all retained files are
confined to the explicitly supplied fresh output root.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


# Prevent an incorrectly launched direct command from writing import caches
# before main() can reject the missing -B flag.  Tests import this module and do
# not change their interpreter-wide bytecode policy.
if __name__ == "__main__":
    sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import model_v18_shoulder_state_runner as runner  # noqa: E402


TARGET_SESSION = "2026-08-05"
LATEST_SOURCE_SESSION = "2026-08-04"
JULY_SNAPSHOT_CUTOFF = "2026-07-31"
JUNE_SNAPSHOT_CUTOFF = "2026-06-30"
EXPECTED_SOURCE_COUNT = 248
EXPECTED_SOURCE_BYTE_COUNT = 1_265_218_618
EXPECTED_RAW_SOURCE_SET_SHA256 = (
    "461450f1916440907eddcdac4443d7085954902c51d87b49bff76669c8b0a540"
)
EXPECTED_REGISTRY_SCOPE = "non_authoritative_nonactivation_a2_cache_rehearsal_only"
REFERENCE_SCOPE = "nonauthority_a2_raw_reference_only"
HEALTH_SCOPE = "nonauthority_a2_cold_rehearsal_health_only"
REFERENCE_RELATIVE = Path("reference/reference.json")
JULY_SNAPSHOT_RELATIVE = Path("reference/model-prices-through-2026-07-31.csv")
JUNE_SNAPSHOT_RELATIVE = Path("reference/model-prices-through-2026-06-30.csv")
HEALTH_RELATIVE = Path("health.json")
BOUNDARY_BUDGET_SECONDS = 300.0
INTRAMONTH_BUDGET_SECONDS = 120.0
FIXED_REPLAY_TIMESTAMPS = {
    "runtime_lock_verified_at": "2026-08-05T08:52:00+09:00",
    "month_source_sealed_at": "2026-08-05T08:53:00+09:00",
    "fit_started_at": "2026-08-05T08:54:00+09:00",
    "fit_completed_at": "2026-08-05T08:55:00+09:00",
    "score_generated_at": "2026-08-05T08:56:00+09:00",
}
STARTUP_CONTROL_ENVIRONMENT = frozenset(
    {
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "PYTHONSAFEPATH",
        "PYTHONUSERBASE",
        "SETUPTOOLS_USE_DISTUTILS",
    }
)
CANONICAL_AUTHORITY_PATHS = (
    runner.ACTIVATION_PAYLOAD,
    runner.ACTIVATION_RECEIPT,
    runner.ACTIVATION_CONTEXT,
    runner.DECISION_LEDGER,
    runner.OUTCOME_LEDGER,
    runner.COMPLETED_MONTH_LEDGER,
    runner.SOURCE_MANIFEST_DIR,
    runner.MONTH_SOURCE_MANIFEST_DIR,
    runner.OUTCOME_MANIFEST_DIR,
    runner.CHECKPOINT_PROPOSAL_DIR,
    runner.FOLD_MANIFEST_DIR,
    runner.FOLD_MODEL_DIR,
    runner.STATE_MANIFEST_DIR,
    runner.SCORE_SESSION_DIR,
    runner.SCORE_OUTPUT,
    runner.PICKS_OUTPUT,
    runner.RESULT_OUTPUT,
)

REGISTRY_FIELDS = frozenset(
    {
        "schema_version",
        "scope",
        "through_session",
        "source_count",
        "source_byte_count",
        "raw_source_set_sha256",
        "independent_acquisition_authority",
        "scientific_authority",
        "activation_payload_or_receipt_created",
        "manual_provenance_caveat",
        "repository",
        "output_root",
        "stores",
        "evidence",
        "sources",
        "runner_raw_source_projection",
        "registry_self_sha256",
    }
)
SOURCE_FIELDS = frozenset(
    {
        "sequence_number",
        "kind",
        "source_date",
        "file_name",
        "source_path",
        "source_url_label",
        "byte_count",
        "sha256",
        "content_binding",
        "url_evidence",
        "manual_operator_provenance_caveat_applies",
        "official_source_independently_verified",
    }
)
REFERENCE_FIELDS = frozenset(
    {
        "schema_version",
        "scope",
        "production_authority",
        "canonical_artifact_written",
        "target_session",
        "latest_source_session",
        "input_kind",
        "source_count",
        "source_byte_count",
        "source_registry_file_sha256",
        "source_registry_self_sha256",
        "raw_source_set_sha256",
        "parsed_shard_set_sha256",
        "parsed_shard_count",
        "rehearsal_source_binding_sha256",
        "fixed_replay_timestamps",
        "predictor_raw_store_relative",
        "predictor_derived_store_relative",
        "raw_source_records",
        "parsed_shard_bindings",
        "snapshots",
        "exact_comparison",
        "exact_comparison_sha256",
        "runtime_lock_sha256",
        "protocol_sha256",
        "runner_sha256",
        "rehearsal_sha256",
        "rehearsal_contract_preflight_sha256",
        "rehearsal_contract_postflight_sha256",
        "manual_provenance_caveat",
        "reference_manifest_sha256",
    }
)
COMPARISON_FIELDS = frozenset(
    {
        "target_session",
        "latest_required_source_session",
        "model_price_row_count",
        "model_price_semantic_sha256",
        "model_price_csv_sha256",
        "source_manifest_sha256",
        "source_set_sha256",
        "parsed_shard_set_sha256",
        "g0_panel_exact_digest",
        "g0_training_row_count",
        "g0_training_panel_semantic_sha256",
        "target_cache_byte_count",
        "target_cache_sha256",
        "target_cache_semantic_sha256",
        "fold_manifest_file_sha256",
        "fold_manifest_sha256",
        "fold_model_bundle_file_sha256",
        "fold_model_bundle_sha256",
        "score_file_sha256",
        "score_semantic_sha256",
        "top2_code_score_ieee_sha256",
        "build_forward_c00_panel_call_count",
    }
)
SNAPSHOT_FIELDS = frozenset(
    {
        "relative_path",
        "latest_source_session",
        "row_count",
        "byte_count",
        "file_sha256",
        "semantic_sha256",
    }
)


class RehearsalError(RuntimeError):
    """Fail closed without creating operational authority."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_file_bytes(value: Any) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _canonical_sha256(value: Any, *, exclude: Iterable[str] = ()) -> str:
    excluded = frozenset(exclude)
    projected = (
        {key: child for key, child in value.items() if key not in excluded}
        if isinstance(value, Mapping)
        else value
    )
    return hashlib.sha256(_canonical_bytes(projected)).hexdigest()


def _plain_file_bytes(
    path: str | Path,
    *,
    label: str,
    signature: bytes | None = None,
) -> tuple[bytes, os.stat_result]:
    target = Path(path)
    try:
        descriptor = os.open(
            target,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise RehearsalError(f"{label} is not a readable plain file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RehearsalError(f"{label} is not a single-link regular file")
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise RehearsalError(f"{label} changed while being read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise RehearsalError(f"{label} changed length while being read")
        if signature is not None and not payload.startswith(signature):
            raise RehearsalError(f"{label} signature changed")
        return payload, after
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def _read_json_plain(path: str | Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    payload, _ = _plain_file_bytes(path, label=label)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RehearsalError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RehearsalError(f"{label} is not a JSON object")
    return value, payload


def _under(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def _private_tmp_directory(
    raw_path: str | Path,
    *,
    label: str,
    require_empty: bool,
) -> Path:
    supplied = Path(raw_path)
    if not supplied.is_absolute():
        raise RehearsalError(f"{label} must be an absolute /tmp path")
    try:
        metadata = supplied.lstat()
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise RehearsalError(f"{label} does not exist") from exc
    tmp_root = Path("/tmp").resolve(strict=True)
    repository = ROOT.resolve(strict=True)
    if (
        resolved != supplied
        or resolved == tmp_root
        or tmp_root not in resolved.parents
        or _under(resolved, repository)
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RehearsalError(f"{label} is not a private, physical /tmp directory")
    if require_empty and next(resolved.iterdir(), None) is not None:
        raise RehearsalError(f"{label} is not fresh and empty")
    return resolved


def _private_subdirectory(root: Path, relative: str) -> Path:
    if "/" in relative or relative in {"", ".", ".."}:
        raise RehearsalError("private output subdirectory name is unsafe")
    target = root / relative
    target.mkdir(mode=0o700)
    metadata = target.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise RehearsalError("private output subdirectory mode changed")
    return target


def _confined_target(root: Path, relative: str | Path) -> Path:
    token = Path(relative)
    if token.is_absolute() or any(part in {"", ".", ".."} for part in token.parts):
        raise RehearsalError("output relative path is unsafe")
    target = root.joinpath(*token.parts)
    parent = target.parent.resolve(strict=True)
    if root != parent and root not in parent.parents:
        raise RehearsalError("output path escapes its fresh root")
    return target


def _write_bytes_exclusive(root: Path, relative: str | Path, payload: bytes) -> Path:
    target = _confined_target(root, relative)
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise RehearsalError("exclusive rehearsal output publication failed") from exc
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != len(payload)
        ):
            raise RehearsalError("rehearsal output metadata changed")
    finally:
        os.close(descriptor)
    parent_descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return target


def _write_json_exclusive(root: Path, relative: str | Path, value: Any) -> Path:
    return _write_bytes_exclusive(root, relative, _canonical_file_bytes(value))


@contextmanager
def _temporary_writes_confined(output_root: Path) -> Iterable[None]:
    scratch = _private_subdirectory(output_root, "scratch")
    previous = tempfile.tempdir
    tempfile.tempdir = str(scratch)
    try:
        yield
    finally:
        tempfile.tempdir = previous
        try:
            scratch.rmdir()
        except OSError as exc:
            raise RehearsalError("rehearsal scratch directory retained content") from exc


def _tree_state(root: Path, *, hash_regular_files: bool) -> tuple[tuple[Any, ...], ...]:
    """Return a no-follow tree identity; .git is outside the repository check."""

    base = root.resolve(strict=True)
    rows: list[tuple[Any, ...]] = []
    stack = [base]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise RehearsalError("tree identity enumeration failed") from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(base)
            if relative.parts[0] == ".git":
                continue
            metadata = entry.stat(follow_symlinks=False)
            kind = stat.S_IFMT(metadata.st_mode)
            content_identity: str | None = None
            if stat.S_ISREG(metadata.st_mode) and hash_regular_files:
                payload, stable = _plain_file_bytes(path, label="tree identity file")
                metadata = stable
                content_identity = hashlib.sha256(payload).hexdigest()
            elif stat.S_ISLNK(metadata.st_mode):
                content_identity = os.readlink(path)
            rows.append(
                (
                    relative.as_posix(),
                    kind,
                    stat.S_IMODE(metadata.st_mode),
                    metadata.st_uid,
                    metadata.st_gid,
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_nlink,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                    content_identity,
                )
            )
            if stat.S_ISDIR(metadata.st_mode):
                stack.append(path)
    return tuple(sorted(rows, key=lambda item: item[0]))


def _assert_canonical_authority_absent() -> None:
    present = [str(path.relative_to(ROOT)) for path in CANONICAL_AUTHORITY_PATHS if os.path.lexists(path)]
    if present:
        raise RehearsalError(
            "nonauthority rehearsal refuses existing canonical authority: "
            + ", ".join(present)
        )


def _validate_direct_runtime() -> dict[str, str]:
    expected_self = ROOT / "research/model_v18_a2_rehearsal.py"
    expected_runner = ROOT / "research/model_v18_shoulder_state_runner.py"
    if Path(__file__).resolve(strict=True) != expected_self.resolve(strict=True):
        raise RehearsalError("rehearsal module path differs from its registered path")
    if Path(runner.__file__).resolve(strict=True) != expected_runner.resolve(strict=True):
        raise RehearsalError("runner import did not resolve to the registered module")
    if __name__ == "__main__" and sys.flags.dont_write_bytecode != 1:
        raise RehearsalError("direct rehearsal must use the registered Python with -B")
    dirty_startup = sorted(name for name in STARTUP_CONTROL_ENVIRONMENT if name in os.environ)
    if dirty_startup:
        raise RehearsalError(
            "Python startup-control environment is not clean: " + ", ".join(dirty_startup)
        )
    runtime_value, runtime_hash = runner.validate_runtime_lock(strict_environment=True)
    _, protocol_hash = runner.validate_protocol()
    if runtime_hash != runner.RUNTIME_LOCK_SHA256 or protocol_hash != runner.PROTOCOL_SHA256:
        raise RehearsalError("runtime/protocol fixed hashes differ from the runner")
    return {
        "runtime_lock_sha256": runtime_hash,
        "protocol_sha256": protocol_hash,
        "runner_sha256": runner.sha256_file(expected_runner),
        "rehearsal_sha256": runner.sha256_file(expected_self),
        "runtime_lock_self_sha256": str(runtime_value["runtime_lock_self_sha256"]),
    }


def _validate_documentary_labels(registry: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]) -> None:
    if (
        registry.get("independent_acquisition_authority") is not False
        or registry.get("scientific_authority") is not False
        or registry.get("activation_payload_or_receipt_created") is not False
    ):
        raise RehearsalError("source registry attempts an authority upgrade")
    caveat = str(registry.get("manual_provenance_caveat", ""))
    if not all(token in caveat for token in ("Aug03", "Aug04", "not independent")):
        raise RehearsalError("source registry omits the fixed manual provenance caveat")
    by_name = {str(item.get("file_name")): item for item in sources}
    expected_classes = {
        "stq_20260803.pdf": (
            "documentary_manual_unbound_page_capture_not_independent_acquisition_authority"
        ),
        "stq_20260804.pdf": (
            "documentary_manual_legacy_operator_ledger_not_independent_acquisition_authority"
        ),
    }
    for name, expected in expected_classes.items():
        item = by_name.get(name)
        evidence = None if item is None else item.get("url_evidence")
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("classification") != expected
            or evidence.get("independent_acquisition_authority") is not False
            or item.get("official_source_independently_verified") is not False
            or item.get("manual_operator_provenance_caveat_applies") is not True
        ):
            raise RehearsalError(f"{name} documentary/manual classification changed")


def _validate_registry_runtime_evidence(
    registry: Mapping[str, Any], runtime_info: Mapping[str, str]
) -> None:
    evidence = registry.get("evidence")
    if not isinstance(evidence, Mapping):
        raise RehearsalError("source registry evidence is not an object")
    authorities = {
        "protocol": (runner.PROTOCOL, "protocol_sha256"),
        "runner": (Path(runner.__file__), "runner_sha256"),
        "runtime_lock": (runner.RUNTIME_LOCK, "runtime_lock_sha256"),
    }
    for role, (path, hash_field) in authorities.items():
        canonical_path = path.resolve(strict=True)
        payload, _ = _plain_file_bytes(
            canonical_path, label=f"current {role} evidence authority"
        )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != runtime_info.get(hash_field):
            raise RehearsalError(f"strict runtime info differs from current {role}")
        expected = {
            "path": str(canonical_path),
            "byte_count": len(payload),
            "sha256": digest,
        }
        if evidence.get(role) != expected:
            raise RehearsalError(f"source registry {role} evidence is stale")


def _validate_source_registry(
    path: str | Path, *, runtime_info: Mapping[str, str]
) -> dict[str, Any]:
    registry_path = Path(path)
    registry, payload = _read_json_plain(registry_path, label="source registry")
    if set(registry) != REGISTRY_FIELDS:
        raise RehearsalError("source registry top-level schema changed")
    if (
        registry.get("schema_version") != 1
        or registry.get("scope") != EXPECTED_REGISTRY_SCOPE
        or registry.get("through_session") != LATEST_SOURCE_SESSION
        or registry.get("source_count") != EXPECTED_SOURCE_COUNT
        or registry.get("source_byte_count") != EXPECTED_SOURCE_BYTE_COUNT
        or registry.get("raw_source_set_sha256") != EXPECTED_RAW_SOURCE_SET_SHA256
        or registry.get("registry_self_sha256")
        != _canonical_sha256(registry, exclude={"registry_self_sha256"})
    ):
        raise RehearsalError("source registry fixed identity changed")
    _validate_registry_runtime_evidence(registry, runtime_info)
    sources = registry.get("sources")
    projection = registry.get("runner_raw_source_projection")
    if not isinstance(sources, list) or not isinstance(projection, list):
        raise RehearsalError("source registry vectors are not arrays")
    _validate_documentary_labels(registry, sources)
    expected_names, kinds = runner._expected_predictor_files(
        runner.pd.Timestamp(LATEST_SOURCE_SESSION)
    )
    if len(expected_names) != EXPECTED_SOURCE_COUNT:
        raise RehearsalError("runner predictor registry cardinality changed")
    raw_records: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    identities: set[tuple[int, int]] = set()
    observed_total = 0
    for index, (item, expected_name) in enumerate(zip(sources, expected_names, strict=True)):
        if not isinstance(item, Mapping) or set(item) != SOURCE_FIELDS:
            raise RehearsalError(f"source registry record {index} schema changed")
        name = str(item["file_name"])
        kind = str(item["kind"])
        if (
            item["sequence_number"] != index
            or name != expected_name
            or kind != kinds[name]
            or item["source_date"] != str(runner._source_file_date(name, kind).date())
            or item["manual_operator_provenance_caveat_applies"] is not True
            or item["official_source_independently_verified"] is not False
        ):
            raise RehearsalError(f"source registry ordering/identity changed at {index}")
        source = Path(str(item["source_path"]))
        if not source.is_absolute():
            raise RehearsalError(f"source registry path {index} is not absolute")
        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise RehearsalError(f"source registry path {index} is unavailable") from exc
        if resolved != source or Path("/tmp").resolve() not in resolved.parents or _under(resolved, ROOT):
            raise RehearsalError(f"source registry path {index} is not a physical /tmp file")
        pdf, metadata = _plain_file_bytes(source, label=f"source PDF {index}", signature=b"%PDF")
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in identities:
            raise RehearsalError("source registry aliases two PDFs to one inode")
        identities.add(identity)
        count = len(pdf)
        digest = hashlib.sha256(pdf).hexdigest()
        if count != item["byte_count"] or digest != item["sha256"]:
            raise RehearsalError(f"source PDF {index} differs from its registry binding")
        url = str(item["source_url_label"])
        if kind == "daily":
            runner._official_jpx_daily_url(
                url,
                f"rehearsal source URL {index}",
                file_name=name,
                source_session=runner._source_file_date(name, kind),
            )
        else:
            runner._official_jpx_url(url, f"rehearsal source URL {index}")
        raw_records.append(
            {
                "object_key": runner._predictor_object_key(name, kind),
                "file": name,
                "url": url,
                "byte_count": count,
                "sha256": digest,
            }
        )
        source_paths.append(source)
        observed_total += count
    if len(sources) != len(expected_names):
        raise RehearsalError("source registry contains trailing or missing records")
    if projection != raw_records:
        raise RehearsalError("source registry runner projection changed")
    if (
        observed_total != EXPECTED_SOURCE_BYTE_COUNT
        or runner.canonical_json_sha256(raw_records) != EXPECTED_RAW_SOURCE_SET_SHA256
    ):
        raise RehearsalError("source registry exact raw-set binding changed")
    return {
        "path": registry_path.resolve(strict=True),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "registry": registry,
        "sources": [dict(item) for item in sources],
        "source_paths": source_paths,
        "raw_records": raw_records,
    }


def _source_binding_sha256(registry_info: Mapping[str, Any]) -> str:
    return runner.canonical_json_sha256(
        {
            "scope": "nonauthority_rehearsal_source_binding_only",
            "source_registry_file_sha256": registry_info["file_sha256"],
            "source_registry_self_sha256": registry_info["registry"][
                "registry_self_sha256"
            ],
            "raw_source_set_sha256": EXPECTED_RAW_SOURCE_SET_SHA256,
            "independent_acquisition_authority": False,
        }
    )


def _seam_arguments(
    *,
    registry_info: Mapping[str, Any],
    parsed_shard_set_sha256: str,
) -> dict[str, Any]:
    return {
        "target_session": TARGET_SESSION,
        **FIXED_REPLAY_TIMESTAMPS,
        "source_manifest_sha256": _source_binding_sha256(registry_info),
        "source_set_sha256": EXPECTED_RAW_SOURCE_SET_SHA256,
        "parsed_shard_set_sha256": parsed_shard_set_sha256,
    }


def _snapshot_record(
    payload: bytes,
    *,
    frame: Any,
    relative_path: Path,
    latest_source_session: str,
) -> dict[str, Any]:
    return {
        "relative_path": relative_path.as_posix(),
        "latest_source_session": latest_source_session,
        "row_count": int(len(frame)),
        "byte_count": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "semantic_sha256": runner.model_price_semantic_sha256(frame),
    }


def _validate_seam_envelope(
    value: Mapping[str, Any],
    *,
    input_kind: str,
    seam_arguments: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        value.get("schema_version") != 1
        or value.get("scope") != "nonauthority_rehearsal_only"
        or value.get("input_kind") != input_kind
        or value.get("production_authority") is not False
        or value.get("canonical_artifact_written") is not False
        or value.get("comparison", {}).get("build_forward_c00_panel_call_count") != 1
    ):
        raise RehearsalError(f"runner rehearsal envelope changed: {input_kind}")
    comparison = value.get("comparison")
    if not isinstance(comparison, dict) or set(comparison) != COMPARISON_FIELDS:
        raise RehearsalError("runner rehearsal comparison schema changed")
    if (
        comparison.get("target_session") != TARGET_SESSION
        or comparison.get("latest_required_source_session")
        != LATEST_SOURCE_SESSION
        or comparison.get("source_manifest_sha256")
        != seam_arguments["source_manifest_sha256"]
        or comparison.get("source_set_sha256")
        != seam_arguments["source_set_sha256"]
        or comparison.get("parsed_shard_set_sha256")
        != seam_arguments["parsed_shard_set_sha256"]
        or isinstance(comparison.get("g0_training_row_count"), bool)
        or not isinstance(comparison.get("g0_training_row_count"), int)
        or int(comparison["g0_training_row_count"]) <= 0
    ):
        raise RehearsalError("runner source/training comparison binding changed")
    for field in (
        "source_manifest_sha256",
        "source_set_sha256",
        "parsed_shard_set_sha256",
        "g0_training_panel_semantic_sha256",
    ):
        digest = comparison.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(child not in "0123456789abcdef" for child in digest)
        ):
            raise RehearsalError(f"runner comparison SHA changed: {field}")
    return dict(comparison)


def _prepare_runner_contract() -> Any:
    """Create the runner's opaque preflight capability outside every timer."""

    factory = getattr(runner, "prepare_a2_nonauthority_rehearsal_contract", None)
    if factory is None:
        raise RehearsalError("runner lacks the registered rehearsal preflight capability")
    return factory()


def _contract_preflight_sha256(rehearsal_contract: Any) -> str:
    digest = getattr(rehearsal_contract, "preflight_sha256", None)
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(child not in "0123456789abcdef" for child in digest)
    ):
        raise RehearsalError("runner rehearsal capability lacks its preflight hash")
    return digest


def _postflight_runner_contract(rehearsal_contract: Any) -> str:
    digest = runner.validate_a2_nonauthority_rehearsal_postflight(
        rehearsal_contract
    )
    if digest != _contract_preflight_sha256(rehearsal_contract):
        raise RehearsalError("runner rehearsal preflight/postflight hashes differ")
    return digest


def _build_seam(
    snapshot_prices: Any,
    suffix_prices: Sequence[Any],
    *,
    input_kind: str,
    seam_arguments: Mapping[str, Any],
    rehearsal_contract: Any,
    reuse_fold_token: Any | None = None,
) -> tuple[dict[str, Any], Any]:
    """The only real execution path; it exposes no callable/mock injection."""

    return runner.build_a2_nonauthority_rehearsal_day(
        snapshot_prices,
        suffix_prices,
        input_kind=input_kind,
        **dict(seam_arguments),
        rehearsal_contract=rehearsal_contract,
        reuse_fold_token=reuse_fold_token,
    )


def _prepare_reference(
    *,
    output_root: Path,
    registry_info: Mapping[str, Any],
    runtime_info: Mapping[str, str],
) -> dict[str, Any]:
    raw_root = _private_subdirectory(output_root, "predictor-raw")
    derived_root = _private_subdirectory(output_root, "predictor-derived")
    reference_dir = _private_subdirectory(output_root, "reference")
    del reference_dir
    raw_records = [dict(item) for item in registry_info["raw_records"]]
    bindings: list[dict[str, Any]] = []
    for source_path, raw_record in zip(
        registry_info["source_paths"], raw_records, strict=True
    ):
        observed = runner._seal_predictor_object(
            source_path,
            predictor_raw_store_root=raw_root,
            object_key=str(raw_record["object_key"]),
        )
        if observed != (int(raw_record["byte_count"]), str(raw_record["sha256"])):
            raise RehearsalError("retained raw PDF differs from its registry binding")
        shard, _, binding = runner.ensure_predictor_parsed_shard(
            raw_record,
            predictor_raw_store_root=raw_root,
            predictor_derived_store_root=derived_root,
            chronology_class="anchor",
            raw_received_at=None,
        )
        if shard.get("chronology_class") != "anchor" or shard.get("raw_received_at") is not None:
            raise RehearsalError("reference parsed shard claims forward chronology")
        bindings.append(dict(binding))
    full31 = runner._load_bound_predictor_shards(
        raw_records,
        bindings,
        predictor_raw_store_root=raw_root,
        predictor_derived_store_root=derived_root,
    )
    dates = runner.pd.to_datetime(full31["date"], errors="coerce")
    if dates.isna().any() or dates.max().normalize() != runner.pd.Timestamp(
        LATEST_SOURCE_SESSION
    ):
        raise RehearsalError("full31 reference does not end on the registered D-1")
    compact = runner._coerce_model_price_frame(
        full31, label="A2 nonauthority full31 compact projection"
    )
    july = runner._coerce_model_price_frame(
        compact.loc[runner.pd.to_datetime(compact["date"]).le(JULY_SNAPSHOT_CUTOFF)],
        label="A2 nonauthority July compact snapshot",
    )
    june = runner._coerce_model_price_frame(
        compact.loc[runner.pd.to_datetime(compact["date"]).le(JUNE_SNAPSHOT_CUTOFF)],
        label="A2 nonauthority June compact snapshot",
    )
    july_payload = runner.canonical_model_price_csv_bytes(
        july, label="A2 nonauthority July compact snapshot"
    )
    june_payload = runner.canonical_model_price_csv_bytes(
        june, label="A2 nonauthority June compact snapshot"
    )
    _write_bytes_exclusive(output_root, JULY_SNAPSHOT_RELATIVE, july_payload)
    _write_bytes_exclusive(output_root, JUNE_SNAPSHOT_RELATIVE, june_payload)
    shard_set_hash = runner._parsed_shard_set_sha256(bindings)
    seam_arguments = _seam_arguments(
        registry_info=registry_info,
        parsed_shard_set_sha256=shard_set_hash,
    )
    rehearsal_contract = _prepare_runner_contract()
    contract_preflight_hash = _contract_preflight_sha256(rehearsal_contract)
    try:
        reference_envelope, _ = _build_seam(
            full31,
            [],
            input_kind="full31_reference",
            seam_arguments=seam_arguments,
            rehearsal_contract=rehearsal_contract,
        )
        comparison = _validate_seam_envelope(
            reference_envelope,
            input_kind="full31_reference",
            seam_arguments=seam_arguments,
        )
    finally:
        contract_postflight_hash = _postflight_runner_contract(
            rehearsal_contract
        )
    snapshots = {
        "month_boundary_compact": _snapshot_record(
            july_payload,
            frame=july,
            relative_path=JULY_SNAPSHOT_RELATIVE,
            latest_source_session=JULY_SNAPSHOT_CUTOFF,
        ),
        "intramonth_fold_reuse_upper_bound_proxy": _snapshot_record(
            june_payload,
            frame=june,
            relative_path=JUNE_SNAPSHOT_RELATIVE,
            latest_source_session=JUNE_SNAPSHOT_CUTOFF,
        ),
    }
    reference: dict[str, Any] = {
        "schema_version": 1,
        "scope": REFERENCE_SCOPE,
        "production_authority": False,
        "canonical_artifact_written": False,
        "target_session": TARGET_SESSION,
        "latest_source_session": LATEST_SOURCE_SESSION,
        "input_kind": "full31_reference",
        "source_count": EXPECTED_SOURCE_COUNT,
        "source_byte_count": EXPECTED_SOURCE_BYTE_COUNT,
        "source_registry_file_sha256": registry_info["file_sha256"],
        "source_registry_self_sha256": registry_info["registry"][
            "registry_self_sha256"
        ],
        "raw_source_set_sha256": EXPECTED_RAW_SOURCE_SET_SHA256,
        "parsed_shard_set_sha256": shard_set_hash,
        "parsed_shard_count": len(bindings),
        "rehearsal_source_binding_sha256": _source_binding_sha256(registry_info),
        "fixed_replay_timestamps": dict(FIXED_REPLAY_TIMESTAMPS),
        "predictor_raw_store_relative": "predictor-raw",
        "predictor_derived_store_relative": "predictor-derived",
        "raw_source_records": raw_records,
        "parsed_shard_bindings": bindings,
        "snapshots": snapshots,
        "exact_comparison": comparison,
        "exact_comparison_sha256": runner.canonical_json_sha256(comparison),
        "runtime_lock_sha256": runtime_info["runtime_lock_sha256"],
        "protocol_sha256": runtime_info["protocol_sha256"],
        "runner_sha256": runtime_info["runner_sha256"],
        "rehearsal_sha256": runtime_info["rehearsal_sha256"],
        "rehearsal_contract_preflight_sha256": contract_preflight_hash,
        "rehearsal_contract_postflight_sha256": contract_postflight_hash,
        "manual_provenance_caveat": (
            "The Aug03 page capture and Aug04 legacy-ledger URL are documentary/manual "
            "labels only, never independent acquisition authority. This reference is "
            "not scientific or production authority."
        ),
    }
    reference["reference_manifest_sha256"] = _canonical_sha256(
        reference, exclude={"reference_manifest_sha256"}
    )
    reference_path = _write_json_exclusive(output_root, REFERENCE_RELATIVE, reference)
    reference_payload, _ = _plain_file_bytes(
        reference_path, label="retained nonauthority reference manifest"
    )
    return {
        "schema_version": 1,
        "scope": HEALTH_SCOPE,
        "command": "prepare-reference",
        "status": "pass",
        "authority": False,
        "target_session": TARGET_SESSION,
        "source_count": EXPECTED_SOURCE_COUNT,
        "source_byte_count": EXPECTED_SOURCE_BYTE_COUNT,
        "raw_source_set_sha256": EXPECTED_RAW_SOURCE_SET_SHA256,
        "parsed_shard_set_sha256": shard_set_hash,
        "reference_manifest_file_sha256": hashlib.sha256(reference_payload).hexdigest(),
        "reference_manifest_sha256": reference["reference_manifest_sha256"],
        "exact_comparison_sha256": reference["exact_comparison_sha256"],
        "reference_build_timed_for_activation_gate": False,
        "repository_unchanged": True,
        "output_confined": True,
        "network_used": False,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "runtime_lock_sha256": runtime_info["runtime_lock_sha256"],
        "protocol_sha256": runtime_info["protocol_sha256"],
        "runner_sha256": runtime_info["runner_sha256"],
        "rehearsal_sha256": runtime_info["rehearsal_sha256"],
        "rehearsal_contract_preflight_sha256": contract_preflight_hash,
        "rehearsal_contract_postflight_sha256": contract_postflight_hash,
    }


def _bound_reference_file(reference_root: Path, relative: str, *, label: str) -> bytes:
    token = Path(relative)
    if token.is_absolute() or any(part in {"", ".", ".."} for part in token.parts):
        raise RehearsalError(f"{label} relative path is unsafe")
    path = reference_root.joinpath(*token.parts)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RehearsalError(f"{label} is missing") from exc
    if reference_root not in resolved.parents:
        raise RehearsalError(f"{label} escapes its reference root")
    payload, metadata = _plain_file_bytes(resolved, label=label)
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise RehearsalError(f"{label} is not private mode 0600")
    return payload


def _load_reference(
    *,
    reference_root: Path,
    registry_info: Mapping[str, Any],
    runtime_info: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    payload = _bound_reference_file(
        reference_root, REFERENCE_RELATIVE.as_posix(), label="reference manifest"
    )
    try:
        reference = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RehearsalError("reference manifest is not valid JSON") from exc
    if not isinstance(reference, dict) or set(reference) != REFERENCE_FIELDS:
        raise RehearsalError("reference manifest schema changed")
    if payload != _canonical_file_bytes(reference):
        raise RehearsalError("reference manifest is not canonical JSON/LF")
    expected_fixed = {
        "schema_version": 1,
        "scope": REFERENCE_SCOPE,
        "production_authority": False,
        "canonical_artifact_written": False,
        "target_session": TARGET_SESSION,
        "latest_source_session": LATEST_SOURCE_SESSION,
        "input_kind": "full31_reference",
        "source_count": EXPECTED_SOURCE_COUNT,
        "source_byte_count": EXPECTED_SOURCE_BYTE_COUNT,
        "source_registry_file_sha256": registry_info["file_sha256"],
        "source_registry_self_sha256": registry_info["registry"][
            "registry_self_sha256"
        ],
        "raw_source_set_sha256": EXPECTED_RAW_SOURCE_SET_SHA256,
        "parsed_shard_count": EXPECTED_SOURCE_COUNT,
        "rehearsal_source_binding_sha256": _source_binding_sha256(registry_info),
        "fixed_replay_timestamps": FIXED_REPLAY_TIMESTAMPS,
        "predictor_raw_store_relative": "predictor-raw",
        "predictor_derived_store_relative": "predictor-derived",
        "runtime_lock_sha256": runtime_info["runtime_lock_sha256"],
        "protocol_sha256": runtime_info["protocol_sha256"],
        "runner_sha256": runtime_info["runner_sha256"],
        "rehearsal_sha256": runtime_info["rehearsal_sha256"],
    }
    for field, expected in expected_fixed.items():
        if reference.get(field) != expected:
            raise RehearsalError(f"reference fixed binding changed: {field}")
    contract_hash = reference.get("rehearsal_contract_preflight_sha256")
    if (
        not isinstance(contract_hash, str)
        or len(contract_hash) != 64
        or any(child not in "0123456789abcdef" for child in contract_hash)
        or reference.get("rehearsal_contract_postflight_sha256") != contract_hash
        or not isinstance(reference.get("exact_comparison"), Mapping)
        or set(reference["exact_comparison"]) != COMPARISON_FIELDS
        or reference["reference_manifest_sha256"]
        != _canonical_sha256(reference, exclude={"reference_manifest_sha256"})
        or reference["exact_comparison_sha256"]
        != runner.canonical_json_sha256(reference["exact_comparison"])
        or reference["raw_source_records"] != registry_info["raw_records"]
        or runner._parsed_shard_set_sha256(reference["parsed_shard_bindings"])
        != reference["parsed_shard_set_sha256"]
    ):
        raise RehearsalError("reference manifest self/set binding changed")
    snapshots = reference.get("snapshots")
    expected_snapshot_names = {
        "month_boundary_compact": JULY_SNAPSHOT_CUTOFF,
        "intramonth_fold_reuse_upper_bound_proxy": JUNE_SNAPSHOT_CUTOFF,
    }
    if not isinstance(snapshots, dict) or set(snapshots) != set(expected_snapshot_names):
        raise RehearsalError("reference compact snapshot registry changed")
    payloads: dict[str, bytes] = {}
    for name, cutoff in expected_snapshot_names.items():
        item = snapshots[name]
        if not isinstance(item, Mapping) or set(item) != SNAPSHOT_FIELDS:
            raise RehearsalError(f"reference compact snapshot schema changed: {name}")
        snapshot_payload = _bound_reference_file(
            reference_root, str(item["relative_path"]), label=f"{name} snapshot"
        )
        frame = runner.decode_canonical_model_price_csv(
            snapshot_payload, label=f"{name} snapshot"
        )
        latest = runner.pd.to_datetime(frame["date"], errors="coerce").max().normalize()
        if (
            item["latest_source_session"] != cutoff
            or latest != runner.pd.Timestamp(cutoff)
            or item["row_count"] != len(frame)
            or item["byte_count"] != len(snapshot_payload)
            or item["file_sha256"] != hashlib.sha256(snapshot_payload).hexdigest()
            or item["semantic_sha256"] != runner.model_price_semantic_sha256(frame)
        ):
            raise RehearsalError(f"reference compact snapshot binding changed: {name}")
        payloads[name] = snapshot_payload
    return reference, payloads


def _reference_store(reference_root: Path, relative: str, *, label: str) -> Path:
    token = Path(relative)
    if token.is_absolute() or len(token.parts) != 1:
        raise RehearsalError(f"{label} store relative path changed")
    path = reference_root / token
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as exc:
        raise RehearsalError(f"{label} store is missing") from exc
    if (
        resolved != path
        or reference_root not in resolved.parents
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RehearsalError(f"{label} store identity changed")
    return resolved


def _suffix_frames(
    reference: Mapping[str, Any],
    *,
    reference_root: Path,
    after: str,
) -> list[Any]:
    raw_records = reference["raw_source_records"]
    bindings = reference["parsed_shard_bindings"]
    raw_root = _reference_store(
        reference_root, reference["predictor_raw_store_relative"], label="predictor raw"
    )
    derived_root = _reference_store(
        reference_root,
        reference["predictor_derived_store_relative"],
        label="predictor derived",
    )
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for raw_record, binding in zip(raw_records, bindings, strict=True):
        name = str(raw_record["file"])
        if not name.startswith("stq_"):
            continue
        session = runner._source_file_date(name, "daily")
        if session > runner.pd.Timestamp(after) and session <= runner.pd.Timestamp(
            LATEST_SOURCE_SESSION
        ):
            selected.append((raw_record, binding))
    expected_names = (
        ["stq_20260803.pdf", "stq_20260804.pdf"]
        if after == JULY_SNAPSHOT_CUTOFF
        else [
            str(item[0]["file"])
            for item in selected
            if runner._source_file_date(str(item[0]["file"]), "daily")
            > runner.pd.Timestamp(JUNE_SNAPSHOT_CUTOFF)
        ]
    )
    if [str(item[0]["file"]) for item in selected] != expected_names or not selected:
        raise RehearsalError("reference compact suffix registry changed")
    frames: list[Any] = []
    for raw_record, binding in selected:
        frames.append(
            runner._load_bound_predictor_shards(
                [raw_record],
                [binding],
                predictor_raw_store_root=raw_root,
                predictor_derived_store_root=derived_root,
            )
        )
    return frames


def _timed_seam(
    snapshot: bytes,
    suffix: Sequence[Any],
    *,
    input_kind: str,
    seam_arguments: Mapping[str, Any],
    rehearsal_contract: Any,
    reuse_fold_token: Any | None = None,
) -> tuple[dict[str, Any], Any, int]:
    started = time.perf_counter_ns()
    envelope, token = _build_seam(
        snapshot,
        suffix,
        input_kind=input_kind,
        seam_arguments=seam_arguments,
        rehearsal_contract=rehearsal_contract,
        reuse_fold_token=reuse_fold_token,
    )
    elapsed = time.perf_counter_ns() - started
    if elapsed <= 0:
        raise RehearsalError("monotonic rehearsal timer did not advance")
    return envelope, token, elapsed


def _health_only(value: Any) -> Any:
    forbidden_keys = {
        "code",
        "name",
        "model_score",
        "return",
        "returns",
        "outcome",
        "outcomes",
        "pick",
        "picks",
        "decision",
        "decisions",
    }

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if str(key).lower() in forbidden_keys:
                    raise RehearsalError("health JSON contains a forbidden value field")
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, float) and not math.isfinite(item):
            raise RehearsalError("health JSON contains a non-finite number")
        elif item is not None and not isinstance(item, (str, int, float, bool)):
            raise RehearsalError("health JSON contains a non-JSON value")

    visit(value)
    return value


def _run_cold(
    *,
    reference_root: Path,
    registry_info: Mapping[str, Any],
    runtime_info: Mapping[str, str],
    cold_run_index: int,
) -> dict[str, Any]:
    reference_before = _tree_state(reference_root, hash_regular_files=False)
    reference, snapshots = _load_reference(
        reference_root=reference_root,
        registry_info=registry_info,
        runtime_info=runtime_info,
    )
    boundary_suffix = _suffix_frames(
        reference, reference_root=reference_root, after=JULY_SNAPSHOT_CUTOFF
    )
    intramonth_suffix = _suffix_frames(
        reference, reference_root=reference_root, after=JUNE_SNAPSHOT_CUTOFF
    )
    seam_arguments = _seam_arguments(
        registry_info=registry_info,
        parsed_shard_set_sha256=reference["parsed_shard_set_sha256"],
    )
    rehearsal_contract = _prepare_runner_contract()
    contract_preflight_hash = _contract_preflight_sha256(rehearsal_contract)
    try:
        if contract_preflight_hash != reference[
            "rehearsal_contract_preflight_sha256"
        ]:
            raise RehearsalError("cold-run preflight differs from raw reference")
        boundary_envelope, boundary_token, boundary_ns = _timed_seam(
            snapshots["month_boundary_compact"],
            boundary_suffix,
            input_kind="month_boundary_compact",
            seam_arguments=seam_arguments,
            rehearsal_contract=rehearsal_contract,
        )
        boundary_comparison = _validate_seam_envelope(
            boundary_envelope,
            input_kind="month_boundary_compact",
            seam_arguments=seam_arguments,
        )
        intramonth_envelope, intramonth_token, intramonth_ns = _timed_seam(
            snapshots["intramonth_fold_reuse_upper_bound_proxy"],
            intramonth_suffix,
            input_kind="intramonth_fold_reuse_upper_bound_proxy",
            seam_arguments=seam_arguments,
            rehearsal_contract=rehearsal_contract,
            reuse_fold_token=boundary_token,
        )
        if intramonth_token is not boundary_token:
            raise RehearsalError("intramonth rehearsal replaced its opaque fold token")
        intramonth_comparison = _validate_seam_envelope(
            intramonth_envelope,
            input_kind="intramonth_fold_reuse_upper_bound_proxy",
            seam_arguments=seam_arguments,
        )
        if not (
            reference["exact_comparison"]
            == boundary_comparison
            == intramonth_comparison
        ):
            raise RehearsalError("raw31/compact12 exact rehearsal comparison failed")
    finally:
        contract_postflight_hash = _postflight_runner_contract(
            rehearsal_contract
        )
    if contract_postflight_hash != reference[
        "rehearsal_contract_postflight_sha256"
    ]:
        raise RehearsalError("cold-run postflight differs from raw reference")
    if _tree_state(reference_root, hash_regular_files=False) != reference_before:
        raise RehearsalError("reference root changed during a cold rehearsal")
    boundary_seconds = boundary_ns / 1_000_000_000.0
    intramonth_seconds = intramonth_ns / 1_000_000_000.0
    boundary_pass = boundary_seconds <= BOUNDARY_BUDGET_SECONDS
    intramonth_pass = intramonth_seconds <= INTRAMONTH_BUDGET_SECONDS
    health: dict[str, Any] = {
        "schema_version": 1,
        "scope": HEALTH_SCOPE,
        "command": "run",
        "cold_run_index": cold_run_index,
        "status": "pass" if boundary_pass and intramonth_pass else "fail",
        "authority": False,
        "target_session": TARGET_SESSION,
        "exact_comparison_sha256": reference["exact_comparison_sha256"],
        "reference_manifest_sha256": reference["reference_manifest_sha256"],
        "source_count": EXPECTED_SOURCE_COUNT,
        "raw_source_set_sha256": EXPECTED_RAW_SOURCE_SET_SHA256,
        "parsed_shard_set_sha256": reference["parsed_shard_set_sha256"],
        "scenarios": [
            {
                "input_kind": "month_boundary_compact",
                "elapsed_seconds": boundary_seconds,
                "budget_seconds": BOUNDARY_BUDGET_SECONDS,
                "within_budget": boundary_pass,
                "exact_match": True,
                "comparison_sha256": runner.canonical_json_sha256(
                    boundary_comparison
                ),
                "panel_call_count": 1,
                "fit_reused": False,
            },
            {
                "input_kind": "intramonth_fold_reuse_upper_bound_proxy",
                "elapsed_seconds": intramonth_seconds,
                "budget_seconds": INTRAMONTH_BUDGET_SECONDS,
                "within_budget": intramonth_pass,
                "exact_match": True,
                "comparison_sha256": runner.canonical_json_sha256(
                    intramonth_comparison
                ),
                "panel_call_count": 1,
                "fit_reused": True,
            },
        ],
        "repository_unchanged": True,
        "reference_unchanged": True,
        "output_confined": True,
        "network_used": False,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "runtime_lock_sha256": runtime_info["runtime_lock_sha256"],
        "protocol_sha256": runtime_info["protocol_sha256"],
        "runner_sha256": runtime_info["runner_sha256"],
        "rehearsal_sha256": runtime_info["rehearsal_sha256"],
        "rehearsal_contract_preflight_sha256": contract_preflight_hash,
        "rehearsal_contract_postflight_sha256": contract_postflight_hash,
    }
    return _health_only(health)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bound local-only A2 raw/compact rehearsal"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    reference = subparsers.add_parser(
        "prepare-reference",
        help="untimed real raw31/shard reference build under one fresh /tmp root",
    )
    reference.add_argument("--source-registry", required=True)
    reference.add_argument("--output-root", required=True)
    run = subparsers.add_parser(
        "run", help="one fresh boundary plus intramonth compact timing run"
    )
    run.add_argument("--source-registry", required=True)
    run.add_argument("--reference-root", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--cold-run-index", required=True, type=int, choices=(1, 2, 3))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_root = _private_tmp_directory(
        args.output_root, label="rehearsal output root", require_empty=True
    )
    reference_root: Path | None = None
    if args.command == "run":
        reference_root = _private_tmp_directory(
            args.reference_root, label="rehearsal reference root", require_empty=False
        )
        if reference_root == output_root:
            raise RehearsalError("reference and output roots must be distinct")
    repository_before = _tree_state(ROOT, hash_regular_files=True)
    with _temporary_writes_confined(output_root):
        runtime_info = _validate_direct_runtime()
        _assert_canonical_authority_absent()
        registry_info = _validate_source_registry(
            args.source_registry, runtime_info=runtime_info
        )
        if args.command == "prepare-reference":
            health = _prepare_reference(
                output_root=output_root,
                registry_info=registry_info,
                runtime_info=runtime_info,
            )
        else:
            assert reference_root is not None
            health = _run_cold(
                reference_root=reference_root,
                registry_info=registry_info,
                runtime_info=runtime_info,
                cold_run_index=args.cold_run_index,
            )
        _assert_canonical_authority_absent()
    if _tree_state(ROOT, hash_regular_files=True) != repository_before:
        raise RehearsalError("repository changed during the nonauthority rehearsal")
    health = _health_only(health)
    _write_json_exclusive(output_root, HEALTH_RELATIVE, health)
    retained, _ = _plain_file_bytes(
        output_root / HEALTH_RELATIVE, label="retained rehearsal health JSON"
    )
    if retained != _canonical_file_bytes(health):
        raise RehearsalError("retained rehearsal health JSON changed")
    print(_canonical_bytes(health).decode("utf-8"))
    return 0 if health["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
