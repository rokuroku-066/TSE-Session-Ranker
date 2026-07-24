#!/usr/bin/env python3
"""No-network, fail-closed T02 OOT data-readiness verifier.

This program can prove that required inputs are absent or internally
inconsistent.  It deliberately cannot certify production readiness: a
separate, signed statistical result, execution result, intended-capital
capacity result, and approval record are required for that decision.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse
from typing import Any, Iterable


SCHEMA_VERSION = 3
VERIFIER_ID = "model_v11_t02_data_readiness_no_network_v3"
SKIP_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
}
DATA_EXTENSIONS = {".csv", ".tsv"}
FIELD_KEYWORDS = {
    "preopen",
    "indicative",
    "auction",
    "spread",
    "slippage",
    "execution",
    "futures",
    "pts",
    "turnover",
    "volume",
    "trading_unit",
    "security_master",
    "tick",
}
JPX_FILE_PATTERN = re.compile(r"^stq_(20\d{6})\.pdf$")
TDNET_FILE_PATTERN = re.compile(r"^(20\d{6})\.html$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_EVIDENCE_REGISTRY = "research/model_v11_data_evidence_registry.json"
FROZEN_EVIDENCE_REGISTRY_SHA256 = (
    "742224f1b1d85bbfa5142d86346594f6b54d2719d39531257c2a3a9b392ba9e3"
)

# Header discovery is diagnostic only.  These definitions are used to
# describe unregistered candidate files, never to turn them into gate
# evidence.
FIELD_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "exact_0858_indicative_price": {
        "label": "exact 08:58 indicative price",
        "gate": "execution",
    },
    "executable_order_simulation": {
        "label": "executable order simulation",
        "gate": "execution",
    },
    "bid_ask_spread": {
        "label": "08:58 bid/ask spread",
        "gate": "execution",
    },
    "realized_spread": {
        "label": "realized spread",
        "gate": "execution",
    },
    "slippage": {
        "label": "realized slippage",
        "gate": "execution",
    },
    "special_quote_0858": {
        "label": "08:58 special-quote state",
        "gate": "execution",
    },
    "open_time_or_delayed_open": {
        "label": "actual/expected open time and delayed-open handling",
        "gate": "execution",
    },
    "futures_0858": {
        "label": "exact-date futures observed by 08:58:59",
        "gate": "research_context",
    },
    "pts_price_and_volume": {
        "label": "PTS price and volume",
        "gate": "research_context",
    },
    "volume": {
        "label": "share volume",
        "gate": "research_context",
    },
    "turnover": {
        "label": "JPY turnover",
        "gate": "execution",
    },
    "minimum_trading_unit": {
        "label": "effective minimum trading unit",
        "gate": "execution",
    },
    "tick_size": {
        "label": "effective tick size",
        "gate": "execution",
    },
    "predicted_opening_turnover": {
        "label": "predicted opening turnover",
        "gate": "execution",
    },
}

# The protocol permits either an exact indicative price or an executable
# order simulation.  Every other tuple is an AND requirement.
EXECUTION_EVIDENCE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("exact_0858_indicative_price", "executable_order_simulation"),
    ("bid_ask_spread",),
    ("realized_spread",),
    ("slippage",),
    ("special_quote_0858",),
    ("open_time_or_delayed_open",),
    ("turnover",),
    ("minimum_trading_unit",),
    ("tick_size",),
    ("predicted_opening_turnover",),
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: str | Path, repo_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve())
    except ValueError:
        return str(resolved)
    return relative.as_posix() if relative.parts else "."


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end precedes start")
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def parse_date_token(token: str) -> date:
    return datetime.strptime(token, "%Y%m%d").date()


def aware_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def canonical_roots(
    protocol_path: Path,
    registry: dict[str, Any],
    explicit_roots: Iterable[str | Path] | None,
) -> tuple[Path, list[Path]]:
    repo_root = protocol_path.resolve().parent.parent
    candidates: list[Path] = [repo_root]
    for value in registry.get("registered_source_roots", []):
        raw = Path(str(value))
        candidates.append(
            raw.resolve() if raw.is_absolute() else (repo_root / raw).resolve()
        )
    if explicit_roots is not None:
        candidates.extend(Path(value).resolve() for value in explicit_roots)
    # Never infer search roots from historical absolute paths stored in an
    # audit.  In particular, scanning a shared /tmp makes a no-network result
    # depend on unrelated prior test runs.  Additional acquisition roots must
    # be supplied explicitly with --root.
    existing: list[Path] = []
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        if resolved not in existing:
            existing.append(resolved)
    minimal: list[Path] = []
    for candidate in sorted(existing, key=lambda value: len(value.parts)):
        if any(candidate == root or root in candidate.parents for root in minimal):
            continue
        minimal.append(candidate)
    return repo_root, minimal


def walk_files(roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [
                name for name in dirnames if name not in SKIP_DIRECTORIES
            ]
            base = Path(directory)
            for name in filenames:
                path = base / name
                try:
                    resolved = path.resolve()
                except OSError:
                    continue
                if resolved in seen or not path.is_file():
                    continue
                seen.add(resolved)
                files.append(path)
    return files


def index_by_name(files: Iterable[Path], names: set[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {name: [] for name in names}
    for path in files:
        if path.name in result:
            result[path.name].append(path.resolve())
    return result


def find_hash_matches(
    files: Iterable[Path],
    expected_sha256: str,
    *,
    suffixes: set[str] | None = None,
    maximum_bytes: int = 5_000_000,
) -> list[str]:
    matches: list[str] = []
    for path in files:
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        try:
            if path.stat().st_size > maximum_bytes:
                continue
            if sha256_file(path) == expected_sha256:
                matches.append(str(path.resolve()))
        except OSError:
            continue
    return sorted(matches)


def verify_bound_artifacts(
    protocol: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for name, specification in protocol["bound_candidate_artifacts"].items():
        raw_path = Path(specification["path"])
        path = raw_path if raw_path.is_absolute() else repo_root / raw_path
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        artifacts[name] = {
            "path": portable_path(path, repo_root),
            "exists": exists,
            "expected_sha256": specification["sha256"],
            "actual_sha256": actual,
            "sha256_exact": bool(exists and actual == specification["sha256"]),
        }
    return {
        "artifacts": artifacts,
        "all_present_and_exact": all(
            value["sha256_exact"] for value in artifacts.values()
        ),
    }


def jpx_expected_records(audit: dict[str, Any]) -> list[dict[str, Any]]:
    records = audit.get("per_file")
    if not isinstance(records, list):
        raise ValueError("parser audit per_file must be a list")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in records:
        if not isinstance(value, dict):
            raise ValueError("parser audit per_file entry must be an object")
        name = value.get("name")
        match = JPX_FILE_PATTERN.fullmatch(str(name))
        if match is None or str(name) in seen:
            raise ValueError(f"invalid or duplicate JPX audit filename: {name}")
        expected_sha256 = str(value.get("sha256", ""))
        if SHA256_PATTERN.fullmatch(expected_sha256) is None:
            raise ValueError(f"invalid JPX audit SHA-256 for {name}")
        seen.add(str(name))
        output.append(
            {
                "name": str(name),
                "date": parse_date_token(match.group(1)),
                "expected_sha256": expected_sha256,
                "parsed_rows": int(value.get("new_parsed_rows", 0)),
                "rejected_rows": int(value.get("new_rejected_rows", -1)),
                "ordinary_rows": int(value.get("ordinary_rows", 0)),
                "recovered_rows": int(value.get("recovered_rows", 0)),
            }
        )
    return sorted(output, key=lambda value: value["date"])


def load_jpx_manifest(
    manifest_path: Path | None,
    expected_sha256: str | None,
    registered_files: set[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    status = {
        "path": str(manifest_path) if manifest_path is not None else None,
        "exists": bool(manifest_path is not None and manifest_path.is_file()),
        "registered_search_file": False,
        "expected_sha256": expected_sha256,
        "actual_sha256": None,
        "sha256_exact": False,
        "readable": False,
        "schema_version": None,
        "declared_file_count": None,
        "record_count": 0,
        "duplicate_names": [],
        "contract_exact": False,
    }
    if manifest_path is None:
        return {}, status
    resolved = manifest_path.resolve()
    status["registered_search_file"] = resolved in registered_files
    if not manifest_path.is_file() or resolved not in registered_files:
        return {}, status
    try:
        actual = sha256_file(manifest_path)
        status["actual_sha256"] = actual
        status["sha256_exact"] = bool(
            expected_sha256
            and SHA256_PATTERN.fullmatch(expected_sha256)
            and actual == expected_sha256
        )
        manifest = read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}, status
    status["readable"] = True
    status["schema_version"] = manifest.get("schema_version")
    status["declared_file_count"] = manifest.get("file_count")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raw_files = manifest.get("downloads")
    if not isinstance(raw_files, list):
        return {}, status
    records: dict[str, dict[str, Any]] = {}
    duplicate_names: list[str] = []
    for record in raw_files:
        if not isinstance(record, dict):
            continue
        name = record.get("name", record.get("filename"))
        if isinstance(name, str):
            if name in records:
                duplicate_names.append(name)
            records[name] = record
    status["record_count"] = len(records)
    status["duplicate_names"] = sorted(set(duplicate_names))
    return records, status


def jpx_manifest_provenance_complete(
    record: dict[str, Any] | None,
    expected_name: str,
    expected_sha256: str,
    *,
    expected_parser_version: str,
    expected_parser_sha256: str,
    approved_source_host: str,
    approved_source_path_prefix: str,
    matching_file_sizes: set[int],
) -> tuple[bool, list[str]]:
    required = {
        "source_url": ("source_url", "url"),
        "byte_count": ("byte_count", "bytes"),
        "request_start": ("request_started_at", "request_start", "observed_at"),
        "receipt_completion": (
            "receipt_completed_at",
            "receipt_completion",
            "fetched_at",
        ),
        "parser_version": ("parser_version",),
        "parser_sha256": ("parser_sha256",),
        "parse_completion": ("parse_completed_at", "parse_completion"),
    }
    if record is None:
        return False, list(required)
    missing: list[str] = []
    for label, alternatives in required.items():
        if not any(record.get(key) not in (None, "") for key in alternatives):
            missing.append(label)
    if record.get("sha256") != expected_sha256:
        missing.append("sha256")
    source_url = next(
        (
            str(record.get(key))
            for key in required["source_url"]
            if record.get(key) not in (None, "")
        ),
        "",
    )
    source = urlparse(source_url)
    if not (
        source.scheme == "https"
        and source.hostname == approved_source_host
        and source.path.startswith(approved_source_path_prefix)
        and source.path.lower().endswith(".pdf")
        and Path(source.path).name == expected_name
    ):
        missing.append("source_url_not_approved")
    byte_count = next(
        (
            record.get(key)
            for key in required["byte_count"]
            if record.get(key) not in (None, "")
        ),
        None,
    )
    try:
        byte_count_value = int(byte_count)
    except (TypeError, ValueError):
        byte_count_value = -1
    if byte_count_value <= 0 or byte_count_value not in matching_file_sizes:
        missing.append("byte_count_mismatch")
    if record.get("parser_version") != expected_parser_version:
        missing.append("parser_version_mismatch")
    if record.get("parser_sha256") != expected_parser_sha256:
        missing.append("parser_sha256_mismatch")
    request = next(
        (
            aware_datetime(record.get(key))
            for key in required["request_start"]
            if record.get(key) not in (None, "")
        ),
        None,
    )
    receipt = next(
        (
            aware_datetime(record.get(key))
            for key in required["receipt_completion"]
            if record.get(key) not in (None, "")
        ),
        None,
    )
    parsed = next(
        (
            aware_datetime(record.get(key))
            for key in required["parse_completion"]
            if record.get(key) not in (None, "")
        ),
        None,
    )
    if request is None:
        missing.append("request_start_timestamp")
    if receipt is None:
        missing.append("receipt_completion_timestamp")
    if parsed is None:
        missing.append("parse_completion_timestamp")
    if request is not None and receipt is not None and receipt < request:
        missing.append("receipt_before_request")
    if receipt is not None and parsed is not None and parsed < receipt:
        missing.append("parse_before_receipt")
    return not missing, sorted(set(missing))


def reparse_jpx_pdf(
    path: Path,
    record: dict[str, Any],
    *,
    expected_parser_version: str,
    expected_parser_sha256: str,
    canonical_parser_path: Path,
    required_fields: set[str],
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "attempted": True,
        "valid": False,
        "parsed_rows": 0,
        "rejected_rows": None,
        "date_exact": False,
        "required_fields_present": False,
        "error": None,
    }
    try:
        from tse_session_ranker.data import jpx as jpx_module

        runtime_parser_path = Path(jpx_module.__file__).resolve()
        canonical_parser_path = canonical_parser_path.resolve()
        status["runtime_parser_path"] = str(runtime_parser_path)
        status["runtime_parser_sha256"] = sha256_file(runtime_parser_path)
        if (
            runtime_parser_path != canonical_parser_path
            or status["runtime_parser_sha256"] != expected_parser_sha256
        ):
            raise ValueError("runtime parser is not the frozen canonical parser")

        if path.read_bytes()[:5] != b"%PDF-":
            raise ValueError("raw source does not have PDF magic")
        frame, manifest = jpx_module.collect_jpx(path)
        inputs = manifest.get("inputs")
        if not isinstance(inputs, list) or len(inputs) != 1:
            raise ValueError("parser manifest must contain exactly one input")
        parsed = inputs[0]
        parsed_rows = int(parsed.get("parsed_rows", -1))
        rejected_rows = int(parsed.get("rejected_rows", -1))
        dates = {
            str(value)[:10]
            for value in frame.get("date", [])
        }
        required_present = required_fields.issubset(frame.columns)
        status.update(
            {
                "parsed_rows": parsed_rows,
                "rejected_rows": rejected_rows,
                "date_exact": dates == {record["date"].isoformat()},
                "required_fields_present": required_present,
                "parser_version": manifest.get("parser_version"),
                "input_sha256": parsed.get("sha256"),
            }
        )
        status["valid"] = bool(
            manifest.get("parser_version") == expected_parser_version
            and parsed.get("parser_version") == expected_parser_version
            and parsed.get("sha256") == record["expected_sha256"]
            and parsed_rows == record["parsed_rows"]
            and rejected_rows == record["rejected_rows"] == 0
            and len(frame) == parsed_rows
            and status["date_exact"]
            and required_present
        )
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}:{exc}"
    return status


def verify_jpx(
    protocol: dict[str, Any],
    audit: dict[str, Any],
    registry: dict[str, Any],
    file_index: dict[str, list[Path]],
    all_files: list[Path],
    repo_root: Path,
) -> dict[str, Any]:
    window = protocol["evaluation_window"]
    warmup = date.fromisoformat(window["warmup_session"])
    score_start = date.fromisoformat(window["score_start"])
    score_end = date.fromisoformat(window["score_end"])
    records = jpx_expected_records(audit)
    score_records = [
        value for value in records if score_start <= value["date"] <= score_end
    ]
    record_dates = [value["date"] for value in records]
    parsed_rows_sum = sum(value["parsed_rows"] for value in records)
    rejected_rows_sum = sum(value["rejected_rows"] for value in records)
    recovered_rows_sum = sum(value["recovered_rows"] for value in records)
    session_date_digest = hashlib.sha256(
        "\n".join(value.isoformat() for value in record_dates).encode("ascii")
    ).hexdigest()
    contract = registry["jpx"]
    audit_contract_exact = bool(
        int(audit.get("pdf_count", -1)) == len(records)
        and len(records) == int(contract["expected_session_count_including_warmup"])
        and session_date_digest == str(contract["session_date_sha256"])
        and len(score_records) == int(window["expected_score_sessions"])
        and records
        and records[0]["date"] == warmup
        and records[-1]["date"] == score_end
        and record_dates == sorted(set(record_dates))
        and len({value["expected_sha256"] for value in records}) == len(records)
        and parsed_rows_sum == int(audit.get("new_parsed_rows", -1))
        and rejected_rows_sum == int(audit.get("new_rejected_rows", -1))
        and recovered_rows_sum == int(audit.get("recovered_rows", -1))
        and int(audit.get("new_rejected_rows", -1)) == 0
        and all(
            value["parsed_rows"] > 0 and value["rejected_rows"] == 0
            for value in records
        )
    )
    expected_parser_sha256 = str(contract["parser_sha256"])
    expected_parser_version = str(contract["parser_version"])
    parser_matches = find_hash_matches(
        all_files, expected_parser_sha256, suffixes={".py"}
    )
    canonical_parser_path = (
        repo_root / "src" / "tse_session_ranker" / "data" / "jpx.py"
    )
    canonical_parser_sha256 = (
        sha256_file(canonical_parser_path)
        if canonical_parser_path.is_file()
        else None
    )
    parser_exact = bool(
        canonical_parser_sha256 == expected_parser_sha256
    )
    manifest_contract = contract["registered_acquisition_parse_manifest"]
    raw_manifest_path = manifest_contract.get("path")
    manifest_path = (
        None
        if raw_manifest_path in (None, "")
        else (
            Path(str(raw_manifest_path))
            if Path(str(raw_manifest_path)).is_absolute()
            else Path(str(raw_manifest_path))
        )
    )
    if manifest_path is not None and not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    manifest_records, manifest_status = load_jpx_manifest(
        manifest_path,
        manifest_contract.get("sha256"),
        {value.resolve() for value in all_files},
    )
    # Scan PDF content hashes once so a hash-exact official file is not missed
    # merely because it was renamed.  The sealed manifest remains mandatory
    # for provenance even when raw bytes are found this way.
    expected_hashes = {value["expected_sha256"] for value in records}
    expected_manifest_names = {value["name"] for value in records}
    manifest_status["contract_exact"] = bool(
        manifest_status["sha256_exact"]
        and manifest_status["schema_version"]
        == manifest_contract.get("schema_version")
        and manifest_status["declared_file_count"] == len(records)
        and manifest_status["record_count"] == len(records)
        and not manifest_status["duplicate_names"]
        and set(manifest_records) == expected_manifest_names
    )
    pdf_paths_by_hash: dict[str, list[Path]] = {
        value: [] for value in expected_hashes
    }
    for path in all_files:
        if path.suffix.lower() != ".pdf":
            continue
        try:
            if path.stat().st_size > 20_000_000:
                continue
            actual = sha256_file(path)
        except OSError:
            continue
        if actual in pdf_paths_by_hash:
            pdf_paths_by_hash[actual].append(path.resolve())
    file_status: list[dict[str, Any]] = []
    for record in records:
        basename_candidates = file_index.get(record["name"], [])
        hash_candidates = pdf_paths_by_hash[record["expected_sha256"]]
        candidates = sorted(set(basename_candidates) | set(hash_candidates))
        matching: list[str] = [
            str(value) for value in sorted(set(hash_candidates))
        ]
        mismatched: list[dict[str, str]] = []
        for path in basename_candidates:
            try:
                actual = sha256_file(path)
            except OSError:
                continue
            if actual == record["expected_sha256"]:
                if str(path) not in matching:
                    matching.append(str(path))
            else:
                mismatched.append({"path": str(path), "sha256": actual})
        matching_file_sizes = {
            Path(value).stat().st_size
            for value in matching
            if Path(value).is_file()
        }
        manifest_complete, manifest_missing = jpx_manifest_provenance_complete(
            manifest_records.get(record["name"]),
            record["name"],
            record["expected_sha256"],
            expected_parser_version=expected_parser_version,
            expected_parser_sha256=expected_parser_sha256,
            approved_source_host=str(contract["approved_source_host"]),
            approved_source_path_prefix=str(
                contract["approved_source_path_prefix"]
            ),
            matching_file_sizes=matching_file_sizes,
        )
        raw_validation = (
            reparse_jpx_pdf(
                Path(sorted(matching)[0]),
                record,
                expected_parser_version=expected_parser_version,
                expected_parser_sha256=expected_parser_sha256,
                canonical_parser_path=canonical_parser_path,
                required_fields=set(
                    protocol["required_sources"]["prices"]["required_fields"]
                ),
            )
            if matching
            else {
                "attempted": False,
                "valid": False,
                "error": "raw_file_missing",
            }
        )
        current_complete = bool(
            matching
            and parser_exact
            and raw_validation["valid"]
            and record["parsed_rows"] > 0
            and record["rejected_rows"] == 0
        )
        file_status.append(
            {
                **record,
                "date": record["date"].isoformat(),
                "present_paths": [
                    portable_path(value, repo_root) for value in candidates
                ],
                "matching_paths": [
                    portable_path(value, repo_root) for value in matching
                ],
                "mismatched_paths": [
                    {
                        **value,
                        "path": portable_path(value["path"], repo_root),
                    }
                    for value in mismatched
                ],
                "raw_file_present": bool(candidates),
                "raw_sha256_exact": bool(matching),
                "raw_reparse": raw_validation,
                "audit_source_complete": bool(
                    record["parsed_rows"] > 0 and record["rejected_rows"] == 0
                ),
                "currently_reproducible_source_complete": current_complete,
                "provenance_complete": bool(
                    current_complete
                    and manifest_status["contract_exact"]
                    and manifest_complete
                ),
                "missing_provenance_fields": manifest_missing,
            }
        )
    score_status = [
        value
        for value in file_status
        if score_start <= date.fromisoformat(value["date"]) <= score_end
    ]
    current_complete_dates = [
        value["date"]
        for value in score_status
        if value["currently_reproducible_source_complete"]
    ]
    provenance_dates = [
        value["date"] for value in score_status if value["provenance_complete"]
    ]
    if manifest_status["path"] is not None:
        manifest_status["path"] = portable_path(
            manifest_status["path"], repo_root
        )
    return {
        "publisher": protocol["required_sources"]["prices"]["publisher"],
        "required_fields": protocol["required_sources"]["prices"][
            "required_fields"
        ],
        "expected_files_including_warmup": len(records),
        "expected_score_sessions": len(score_records),
        "audit_contract_exact": audit_contract_exact,
        "audit_contract_checks": {
            "date_set_unique_and_sorted": record_dates == sorted(set(record_dates)),
            "date_set_sha256": session_date_digest,
            "expected_date_set_sha256": contract["session_date_sha256"],
            "calendar_binding": (
                "exact 160-session date-set digest bound by the separate "
                "data-evidence registry"
            ),
            "parsed_rows_sum": parsed_rows_sum,
            "parsed_rows_top_level": audit.get("new_parsed_rows"),
            "rejected_rows_sum": rejected_rows_sum,
            "rejected_rows_top_level": audit.get("new_rejected_rows"),
            "recovered_rows_sum": recovered_rows_sum,
            "recovered_rows_top_level": audit.get("recovered_rows"),
            "all_per_file_sha256_well_formed": all(
                SHA256_PATTERN.fullmatch(value["expected_sha256"]) is not None
                for value in records
            ),
            "all_per_file_sha256_unique": (
                len({value["expected_sha256"] for value in records})
                == len(records)
            ),
        },
        "raw_discovery_method": (
            "exact basename plus content-SHA256 scan of local PDF files; "
            "sealed manifest still required for provenance"
        ),
        "audit_source_complete_files": sum(
            value["audit_source_complete"] for value in file_status
        ),
        "audit_source_complete_score_sessions": sum(
            value["audit_source_complete"] for value in score_status
        ),
        "raw_files_present": sum(value["raw_file_present"] for value in file_status),
        "raw_files_hash_exact": sum(
            value["raw_sha256_exact"] for value in file_status
        ),
        "currently_reproducible_source_complete_files": sum(
            value["currently_reproducible_source_complete"]
            for value in file_status
        ),
        "currently_reproducible_source_complete_score_sessions": len(
            current_complete_dates
        ),
        "provenance_complete_score_sessions": len(provenance_dates),
        "missing_raw_files": [
            value["name"] for value in file_status if not value["raw_file_present"]
        ],
        "hash_mismatch_files": [
            value["name"]
            for value in file_status
            if value["raw_file_present"] and not value["raw_sha256_exact"]
        ],
        "current_source_complete_score_dates": current_complete_dates,
        "provenance_complete_score_dates": provenance_dates,
        "parser": {
            "expected_version": expected_parser_version,
            "expected_sha256": expected_parser_sha256,
            "canonical_path": portable_path(canonical_parser_path, repo_root),
            "canonical_actual_sha256": canonical_parser_sha256,
            "matching_paths": [
                portable_path(value, repo_root) for value in parser_matches
            ],
            "available_and_exact": parser_exact,
        },
        "sealed_manifest": manifest_status,
        "files": file_status,
    }


def tdnet_expected_dates(protocol: dict[str, Any]) -> list[date]:
    start = date.fromisoformat(protocol["evaluation_window"]["warmup_session"])
    end = date.fromisoformat(protocol["evaluation_window"]["score_end"])
    return date_range(start, end)


def validate_tdnet_page(
    page_path: Path,
    index_date: date,
    required_fields: set[str],
    expected_parser_version: str,
    expected_parser_sha256: str,
    approved_index_host: str,
    approved_index_path_prefix: str,
    approved_release_host: str,
    manifest_record: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata_path = page_path.with_suffix(page_path.suffix + ".meta.json")
    status: dict[str, Any] = {
        "page_path": str(page_path),
        "metadata_path": str(metadata_path),
        "page_present": page_path.is_file(),
        "metadata_present": metadata_path.is_file(),
        "page_sha256": None,
        "parser_valid": False,
        "parser_sha256_exact": False,
        "row_semantics_valid": False,
        "content_semantics_valid": False,
        "metadata_integrity_valid": False,
        "manifest_binding_valid": False,
        "finalized": False,
        "source_complete": False,
        "provenance_complete": False,
        "missing_provenance_fields": [],
        "error": None,
    }
    if not page_path.is_file():
        status["error"] = "page_missing"
        return status
    try:
        payload = page_path.read_bytes()
        status["page_sha256"] = hashlib.sha256(payload).hexdigest()
    except OSError as exc:
        status["error"] = f"page_unreadable:{exc}"
        return status
    try:
        from tse_session_ranker.data.tdnet import parse_tdnet_index

        parsed = parse_tdnet_index(payload, index_date.isoformat())
        validation_frame = parsed.rename(columns={"url": "document_url"})
        columns_valid = required_fields.issubset(validation_frame.columns)
        row_semantics_valid = True
        if not validation_frame.empty and columns_valid:
            for row in validation_frame[list(required_fields)].to_dict("records"):
                if str(row.get("index_date", ""))[:10] != index_date.isoformat():
                    row_semantics_valid = False
                    break
                if not all(
                    str(row.get(key, "")).strip()
                    for key in ("code", "name", "title", "document_url")
                ):
                    row_semantics_valid = False
                    break
                document_url = urlparse(str(row["document_url"]))
                published_at = aware_datetime(row.get("published_at"))
                if (
                    document_url.scheme != "https"
                    or document_url.hostname != approved_release_host
                    or not document_url.path.lower().endswith(".pdf")
                    or published_at is None
                ):
                    row_semantics_valid = False
                    break
                published_local = published_at.astimezone(
                    timezone(timedelta(hours=9))
                )
                if published_local.date() != index_date:
                    row_semantics_valid = False
                    break
        decoded = payload.decode("utf-8", errors="replace")
        linked_documents = {
            value.lower()
            for value in re.findall(
                r"https?://www\.release\.tdnet\.info/"
                r"[^\"'<>\s&]+\.pdf",
                decoded,
                flags=re.IGNORECASE,
            )
        }
        parsed_documents = {
            str(value).lower()
            for value in validation_frame.get("document_url", [])
        }
        link_count_exact = linked_documents == parsed_documents
        empty_page_structure = bool(
            re.search(r'<div\s+class=["\']table["\']', decoded)
            and re.search(r"<strong>\s*時刻\s*</strong>", decoded)
            and re.search(r"<strong>\s*銘柄名\s*</strong>", decoded)
            and re.search(r"<strong>\s*表題\s*</strong>", decoded)
            and re.search(
                rf"/contents/tdnet_date/"
                rf"{(index_date - timedelta(days=1)).strftime('%Y%m%d')}",
                decoded,
            )
            and re.search(
                rf"/contents/tdnet_date/"
                rf"{(index_date + timedelta(days=1)).strftime('%Y%m%d')}",
                decoded,
            )
        )
        content_semantics_valid = bool(
            len(payload) >= 100
            and link_count_exact
            and (not validation_frame.empty or empty_page_structure)
        )
        status["row_count"] = int(len(validation_frame))
        status["empty_page_structure_valid"] = empty_page_structure
        status["release_link_count"] = len(linked_documents)
        status["parsed_document_count"] = len(parsed_documents)
        status["release_link_set_exact"] = link_count_exact
        status["row_semantics_valid"] = row_semantics_valid
        status["content_semantics_valid"] = content_semantics_valid
        status["parser_valid"] = bool(
            columns_valid and row_semantics_valid and content_semantics_valid
        )
    except Exception as exc:
        status["error"] = f"parser_invalid:{type(exc).__name__}:{exc}"
    if not metadata_path.is_file():
        status["missing_provenance_fields"] = [
            "source_url",
            "byte_count",
            "sha256",
            "request_start",
            "receipt_completion",
            "parser_version",
            "parser_sha256",
            "parse_completion",
        ]
        return status
    try:
        metadata = read_json(metadata_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        status["error"] = f"metadata_unreadable:{exc}"
        return status
    metadata_sha256 = sha256_file(metadata_path)
    status["metadata_sha256"] = metadata_sha256
    observed = aware_datetime(metadata.get("observed_at"))
    fetched = aware_datetime(metadata.get("fetched_at"))
    parsed_at = aware_datetime(
        metadata.get("parse_completed_at", metadata.get("parse_completion"))
    )
    source_url = str(metadata.get("source_url", metadata.get("url", "")))
    source = urlparse(source_url)
    expected_source_path = (
        f"{approved_index_path_prefix}{index_date.strftime('%Y%m%d')}"
    )
    parser_sha256 = str(metadata.get("parser_sha256", ""))
    status["parser_sha256_exact"] = parser_sha256 == expected_parser_sha256
    integrity_checks = {
        "schema_version": metadata.get("schema_version") == 1,
        "index_date": metadata.get("index_date") == index_date.isoformat(),
        "bytes": metadata.get("bytes") == len(payload),
        "sha256": metadata.get("sha256") == status["page_sha256"],
        "source_url": bool(
            source.scheme == "https"
            and source.hostname == approved_index_host
            and source.path.rstrip("/") == expected_source_path
        ),
        "observed_at": observed is not None,
        "fetched_at": fetched is not None,
        "parse_completed_at": parsed_at is not None,
        "chronology": bool(
            observed is not None
            and fetched is not None
            and parsed_at is not None
            and observed <= fetched <= parsed_at
        ),
        "provenance": metadata.get("provenance")
        == "network_request_start_recorded",
        "parser_version": bool(str(metadata.get("parser_version", "")).strip()),
        "parser_sha256": status["parser_sha256_exact"],
        "http_status": metadata.get("http_status") == 200,
        "content_type": str(metadata.get("content_type", ""))
        .lower()
        .startswith("text/html"),
    }
    status["metadata_integrity_checks"] = integrity_checks
    integrity_checks["parser_version"] = (
        metadata.get("parser_version") == expected_parser_version
    )
    status["metadata_integrity_valid"] = all(integrity_checks.values())
    if manifest_record is not None:
        try:
            manifest_byte_count = int(
                manifest_record.get(
                    "byte_count", manifest_record.get("bytes", -1)
                )
            )
        except (TypeError, ValueError):
            manifest_byte_count = -1
        status["manifest_binding_valid"] = bool(
            manifest_record.get("name", manifest_record.get("filename"))
            == page_path.name
            and manifest_record.get("sha256") == status["page_sha256"]
            and manifest_byte_count == len(payload)
            and manifest_record.get("metadata_name") == metadata_path.name
            and manifest_record.get("metadata_sha256") == metadata_sha256
        )
    status["finalized"] = bool(metadata.get("finalized") is True)
    field_alternatives = {
        "source_url": ("source_url", "url"),
        "byte_count": ("byte_count", "bytes"),
        "sha256": ("sha256",),
        "request_start": ("request_started_at", "observed_at"),
        "receipt_completion": ("receipt_completed_at", "fetched_at"),
        "parser_version": ("parser_version",),
        "parser_sha256": ("parser_sha256",),
        "parse_completion": ("parse_completed_at", "parse_completion"),
    }
    missing = [
        label
        for label, alternatives in field_alternatives.items()
        if not any(metadata.get(key) not in (None, "") for key in alternatives)
    ]
    if parsed_at is None:
        missing.append("parse_completion_timestamp")
    if parser_sha256 != expected_parser_sha256:
        missing.append("parser_sha256_mismatch")
    if metadata.get("parser_version") != expected_parser_version:
        missing.append("parser_version_mismatch")
    if observed is not None and fetched is not None and fetched < observed:
        missing.append("receipt_before_request")
    if fetched is not None and parsed_at is not None and parsed_at < fetched:
        missing.append("parse_before_receipt")
    status["missing_provenance_fields"] = sorted(set(missing))
    status["source_complete"] = bool(
        status["parser_valid"]
        and status["metadata_integrity_valid"]
        and status["manifest_binding_valid"]
        and status["finalized"]
    )
    status["provenance_complete"] = bool(
        status["source_complete"] and not status["missing_provenance_fields"]
    )
    return status


def tdnet_session_coverage(
    jpx_score_dates: list[date],
    warmup: date,
    page_status: dict[date, dict[str, Any]],
    field: str,
) -> list[date]:
    complete: list[date] = []
    previous = warmup
    for session in jpx_score_dates:
        required = date_range(previous, session)
        if all(page_status.get(value, {}).get(field) is True for value in required):
            complete.append(session)
        previous = session
    return complete


def verify_tdnet(
    protocol: dict[str, Any],
    registry: dict[str, Any],
    jpx_records: list[dict[str, Any]],
    file_index: dict[str, list[Path]],
    repo_root: Path,
    all_files: list[Path],
) -> dict[str, Any]:
    required_fields = set(
        protocol["required_sources"]["tdnet"]["required_fields"]
    )
    parser_path = repo_root / "src" / "tse_session_ranker" / "data" / "tdnet.py"
    actual_parser_sha256 = sha256_file(parser_path) if parser_path.is_file() else ""
    contract = registry["tdnet"]
    expected_parser_sha256 = str(contract["parser_sha256"])
    expected_dates = tdnet_expected_dates(protocol)
    manifest_contract = contract["registered_page_manifest"]
    raw_manifest_path = manifest_contract.get("path")
    manifest_path = (
        None
        if raw_manifest_path in (None, "")
        else Path(str(raw_manifest_path))
    )
    if manifest_path is not None and not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    manifest_records, manifest_status = load_jpx_manifest(
        manifest_path,
        manifest_contract.get("sha256"),
        {value.resolve() for value in all_files},
    )
    expected_manifest_names = {
        value.strftime("%Y%m%d.html") for value in expected_dates
    }
    manifest_status["contract_exact"] = bool(
        manifest_status["sha256_exact"]
        and manifest_status["schema_version"]
        == manifest_contract.get("schema_version")
        and manifest_status["declared_file_count"] == len(expected_dates)
        and manifest_status["record_count"] == len(expected_dates)
        and not manifest_status["duplicate_names"]
        and set(manifest_records) == expected_manifest_names
    )
    start = date.fromisoformat(protocol["evaluation_window"]["score_start"])
    end = date.fromisoformat(protocol["evaluation_window"]["score_end"])
    warmup = date.fromisoformat(protocol["evaluation_window"]["warmup_session"])
    score_dates = [
        value["date"]
        for value in jpx_records
        if start <= value["date"] <= end
    ]
    statuses: dict[date, dict[str, Any]] = {}
    duplicate_pages: dict[str, list[str]] = {}
    for value in expected_dates:
        name = value.strftime("%Y%m%d.html")
        candidates = file_index.get(name, [])
        if len(candidates) > 1:
            duplicate_pages[value.isoformat()] = [
                str(candidate) for candidate in candidates
            ]
        if not candidates:
            statuses[value] = {
                "date": value.isoformat(),
                "page_path": None,
                "metadata_path": None,
                "page_present": False,
                "metadata_present": False,
                "parser_valid": False,
                "metadata_integrity_valid": False,
                "finalized": False,
                "source_complete": False,
                "provenance_complete": False,
                "missing_provenance_fields": [
                    "source_url",
                    "byte_count",
                    "sha256",
                    "request_start",
                    "receipt_completion",
                    "parser_version",
                    "parser_sha256",
                    "parse_completion",
                ],
                "error": "page_missing",
            }
            continue
        candidate_results = [
            validate_tdnet_page(
                candidate,
                value,
                required_fields,
                str(contract["parser_contract_version"]),
                expected_parser_sha256,
                str(contract["approved_index_host"]),
                str(contract["approved_index_path_prefix"]),
                str(contract["approved_release_host"]),
                manifest_records.get(name)
                if manifest_status["contract_exact"]
                else None,
            )
            for candidate in candidates
        ]
        candidate_results.sort(
            key=lambda item: (
                item["provenance_complete"],
                item["source_complete"],
                item["metadata_present"],
            ),
            reverse=True,
        )
        statuses[value] = {"date": value.isoformat(), **candidate_results[0]}
        if len(candidates) > 1:
            statuses[value]["source_complete"] = False
            statuses[value]["provenance_complete"] = False
            statuses[value]["error"] = "duplicate_page_candidates"
    source_complete_sessions = tdnet_session_coverage(
        score_dates, warmup, statuses, "source_complete"
    )
    provenance_complete_sessions = tdnet_session_coverage(
        score_dates, warmup, statuses, "provenance_complete"
    )
    ordered = [statuses[value] for value in expected_dates]
    parser_implementation_exact = bool(
        parser_path.is_file()
        and actual_parser_sha256 == expected_parser_sha256
    )
    if not parser_implementation_exact:
        for value in ordered:
            value["source_complete"] = False
            value["provenance_complete"] = False
            missing = list(value.get("missing_provenance_fields", []))
            missing.append("local_parser_sha256_mismatch")
            value["missing_provenance_fields"] = sorted(set(missing))
        source_complete_sessions = []
        provenance_complete_sessions = []
    if manifest_status["path"] is not None:
        manifest_status["path"] = portable_path(
            manifest_status["path"], repo_root
        )
    duplicate_pages = {
        key: [portable_path(value, repo_root) for value in values]
        for key, values in duplicate_pages.items()
    }
    for value in ordered:
        for key in ("page_path", "metadata_path"):
            if value.get(key):
                value[key] = portable_path(value[key], repo_root)
    return {
        "required_fields": protocol["required_sources"]["tdnet"][
            "required_fields"
        ],
        "parser": {
            "path": portable_path(parser_path, repo_root),
            "expected_sha256": expected_parser_sha256,
            "actual_sha256": actual_parser_sha256,
            "available": parser_path.is_file(),
            "available_and_exact": parser_implementation_exact,
            "metadata_must_bind_exact_sha256": True,
        },
        "expected_calendar_pages": len(expected_dates),
        "expected_date_start": expected_dates[0].isoformat(),
        "expected_date_end": expected_dates[-1].isoformat(),
        "pages_present": sum(value["page_present"] for value in ordered),
        "metadata_sidecars_present": sum(
            value["metadata_present"] for value in ordered
        ),
        "parser_valid_pages": sum(value["parser_valid"] for value in ordered),
        "source_complete_pages": sum(
            value["source_complete"] for value in ordered
        ),
        "provenance_complete_pages": sum(
            value["provenance_complete"] for value in ordered
        ),
        "source_complete_score_sessions": len(source_complete_sessions),
        "provenance_complete_score_sessions": len(
            provenance_complete_sessions
        ),
        "source_complete_score_dates": [
            value.isoformat() for value in source_complete_sessions
        ],
        "provenance_complete_score_dates": [
            value.isoformat() for value in provenance_complete_sessions
        ],
        "missing_page_dates": [
            value["date"] for value in ordered if not value["page_present"]
        ],
        "missing_metadata_dates": [
            value["date"] for value in ordered if not value["metadata_present"]
        ],
        "duplicate_pages": duplicate_pages,
        "registered_page_manifest": manifest_status,
        "pages": ordered,
    }


def delimited_header_candidate(path: Path) -> dict[str, Any]:
    """Inspect a small delimited file for diagnostics, never gate evidence."""

    try:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig") as stream:
            lines = [stream.readline(), stream.readline()]
        columns = [
            value.strip().strip('"') for value in lines[0].split(delimiter)
        ] if lines[0] else []
        return {
            "columns": columns or None,
            "has_at_least_one_data_row": bool(lines[1].strip()),
        }
    except (OSError, UnicodeError):
        return {"columns": None, "has_at_least_one_data_row": False}


def load_registered_t02_decisions(
    *,
    registry: dict[str, Any],
    repo_root: Path,
    registered_files: set[Path],
    allowed_sessions: set[date],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    contract = registry["registered_t02_decision_artifact"]
    status: dict[str, Any] = {
        "path": contract.get("path"),
        "sha256_exact": False,
        "rows": 0,
        "sessions": 0,
        "order_decisions": 0,
        "decision_id_sha256": None,
        "decision_rows_sha256": None,
        "valid": False,
        "errors": [],
    }
    raw_path = contract.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        status["errors"].append("decision_artifact_path_missing")
        return {}, status
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    status["path"] = portable_path(path, repo_root)
    if path not in registered_files or not path.is_file():
        status["errors"].append("decision_artifact_not_registered")
        return {}, status
    expected_sha = contract.get("sha256")
    status["sha256_exact"] = bool(
        isinstance(expected_sha, str)
        and SHA256_PATTERN.fullmatch(expected_sha)
        and sha256_file(path) == expected_sha
    )
    if not status["sha256_exact"]:
        status["errors"].append("decision_artifact_sha256")
        return {}, status
    required = {
        "session_date",
        "decision_id",
        "action",
        "code",
        "decision_at",
        "candidate_id",
        "rank",
        "predicted_value",
    }
    rows_by_session: dict[date, dict[str, Any]] = {}
    order_decisions: dict[str, dict[str, Any]] = {}
    seen_decision_ids: set[str] = set()
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not required.issubset(set(reader.fieldnames or [])):
                errors.append("required_columns")
            else:
                for line_number, row in enumerate(reader, start=2):
                    try:
                        session = date.fromisoformat(row["session_date"])
                        decision_at = aware_datetime(row["decision_at"])
                        action = row["action"].strip().lower()
                        decision_id = row["decision_id"].strip()
                        if session not in allowed_sessions:
                            raise ValueError("session_not_joint_source_complete")
                        if session in rows_by_session:
                            raise ValueError("duplicate_session")
                        if not decision_id or decision_id in seen_decision_ids:
                            raise ValueError("duplicate_or_empty_decision_id")
                        if (
                            row["candidate_id"]
                            != "v10_t02_char_value_event_top1"
                        ):
                            raise ValueError("candidate_id")
                        jst = timezone(timedelta(hours=9))
                        cutoff = datetime.combine(
                            session,
                            datetime.strptime("08:58:59", "%H:%M:%S").time(),
                            tzinfo=jst,
                        )
                        if (
                            decision_at is None
                            or decision_at.astimezone(jst).date() != session
                            or decision_at.astimezone(jst) > cutoff
                        ):
                            raise ValueError("decision_pit")
                        if action not in {"order", "cash"}:
                            raise ValueError("action")
                        if action == "order":
                            if (
                                re.fullmatch(r"[0-9A-Z]{4,5}", row["code"])
                                is None
                                or int(row["rank"]) != 1
                            ):
                                raise ValueError("order_identity")
                            predicted_value = float(row["predicted_value"])
                            if not (
                                predicted_value == predicted_value
                                and abs(predicted_value) < float("inf")
                            ):
                                raise ValueError("predicted_value_nonfinite")
                            order_decisions[decision_id] = {
                                **row,
                                "session": session,
                                "decision_at_parsed": decision_at,
                            }
                        elif row["code"].strip() or row["rank"].strip() not in {
                            "",
                            "0",
                        }:
                            raise ValueError("cash_identity")
                        seen_decision_ids.add(decision_id)
                        rows_by_session[session] = row
                    except (KeyError, TypeError, ValueError) as exc:
                        errors.append(
                            f"row_{line_number}:{type(exc).__name__}:{exc}"
                        )
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(f"csv_unreadable:{type(exc).__name__}")
    if set(rows_by_session) != allowed_sessions:
        errors.append("decision_sessions_not_complete")
    ordered_rows = [rows_by_session[value] for value in sorted(rows_by_session)]
    decision_id_sha256 = hashlib.sha256(
        "\n".join(
            sorted(str(value.get("decision_id", "")) for value in ordered_rows)
        ).encode("utf-8")
    ).hexdigest()
    decision_rows_sha256 = hashlib.sha256(
        json.dumps(
            ordered_rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    status.update(
        {
            "rows": len(rows_by_session),
            "sessions": len(rows_by_session),
            "order_decisions": len(order_decisions),
            "decision_id_sha256": decision_id_sha256,
            "decision_rows_sha256": decision_rows_sha256,
            "errors": errors,
            "valid": not errors,
        }
    )
    return order_decisions if not errors else {}, status


def validate_t02_decision_replay_audit(
    *,
    registry: dict[str, Any],
    repo_root: Path,
    registered_files: set[Path],
    decision_artifact: dict[str, Any],
) -> dict[str, Any]:
    contract = registry["registered_t02_decision_replay_audit"]
    status: dict[str, Any] = {
        "path": contract.get("path"),
        "sha256_exact": False,
        "internal_integrity_valid": False,
        "external_origin_authenticated": False,
        "valid": False,
        "errors": [],
    }
    raw_path = contract.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        status["errors"].append("decision_replay_audit_path_missing")
        return status
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    status["path"] = portable_path(path, repo_root)
    if path not in registered_files or not path.is_file():
        status["errors"].append("decision_replay_audit_not_registered")
        return status
    expected_sha = contract.get("sha256")
    actual_sha = sha256_file(path)
    status["sha256_exact"] = bool(
        isinstance(expected_sha, str)
        and SHA256_PATTERN.fullmatch(expected_sha)
        and actual_sha == expected_sha
    )
    if not status["sha256_exact"]:
        status["errors"].append("decision_replay_audit_sha256")
        return status
    try:
        audit = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        status["errors"].append(
            f"decision_replay_audit_unreadable:{type(exc).__name__}"
        )
        return status
    if audit.get("schema_version") != contract.get("schema_version"):
        status["errors"].append("decision_replay_audit_schema")
    if audit.get("validator_id") != "t02_decision_exact_replay_v1":
        status["errors"].append("decision_replay_audit_validator")
    exact_values = {
        "decision_artifact_sha256": registry[
            "registered_t02_decision_artifact"
        ].get("sha256"),
        "protocol_sha256": registry["protocol"].get("sha256"),
        "jpx_manifest_sha256": registry["jpx"][
            "registered_acquisition_parse_manifest"
        ].get("sha256"),
        "tdnet_manifest_sha256": registry["tdnet"][
            "registered_page_manifest"
        ].get("sha256"),
        "decision_id_sha256": decision_artifact.get("decision_id_sha256"),
        "decision_rows_sha256": decision_artifact.get(
            "decision_rows_sha256"
        ),
        "session_count": decision_artifact.get("sessions"),
        "order_count": decision_artifact.get("order_decisions"),
    }
    for key, expected in exact_values.items():
        if expected in (None, "") or audit.get(key) != expected:
            status["errors"].append(f"decision_replay_binding:{key}")
    checks = audit.get("checks")
    required_checks = {
        "source_complete_sessions_exact",
        "frozen_candidate_replayed",
        "orders_and_cash_exact",
        "scores_and_ties_exact",
        "mutation_rejects_code_substitution",
        "mutation_rejects_cash_substitution",
    }
    if not isinstance(checks, dict) or not all(
        checks.get(value) is True for value in required_checks
    ):
        status["errors"].append("decision_replay_checks")
    artifacts = audit.get("bound_artifacts")
    required_roles = {"decision_generator_source", "mutation_tests"}
    roles: list[str] = []
    if not isinstance(artifacts, list):
        status["errors"].append("decision_replay_bound_artifacts")
    else:
        for value in artifacts:
            if not isinstance(value, dict):
                status["errors"].append(
                    "decision_replay_bound_artifact_record"
                )
                continue
            role = str(value.get("role", ""))
            roles.append(role)
            artifact_path = Path(str(value.get("path", "")))
            if not artifact_path.is_absolute():
                artifact_path = repo_root / artifact_path
            artifact_path = artifact_path.resolve()
            declared_sha = value.get("sha256")
            if (
                artifact_path not in registered_files
                or not artifact_path.is_file()
                or not isinstance(declared_sha, str)
                or SHA256_PATTERN.fullmatch(declared_sha) is None
                or declared_sha != sha256_file(artifact_path)
            ):
                status["errors"].append(
                    f"decision_replay_artifact:{role}"
                )
        if set(roles) != required_roles or len(roles) != len(set(roles)):
            status["errors"].append("decision_replay_artifact_roles")
    status["internal_integrity_valid"] = bool(
        decision_artifact.get("valid") and not status["errors"]
    )
    status["errors"].append("external_attestation_not_verified")
    status["valid"] = False
    return status


def validate_execution_policy_audit(
    *,
    registry: dict[str, Any],
    repo_root: Path,
    registered_files: set[Path],
) -> dict[str, Any]:
    contract = registry["registered_execution_policy_audit"]
    status: dict[str, Any] = {
        "path": contract.get("path"),
        "sha256_exact": False,
        "sha256": None,
        "bound_artifact_sha256": {},
        "internal_integrity_valid": False,
        "external_origin_authenticated": False,
        "valid": False,
        "errors": [],
    }
    raw_path = contract.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        status["errors"].append("policy_audit_path_missing")
        return status
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    status["path"] = portable_path(path, repo_root)
    if path not in registered_files or not path.is_file():
        status["errors"].append("policy_audit_not_registered")
        return status
    expected_sha = contract.get("sha256")
    status["sha256_exact"] = bool(
        isinstance(expected_sha, str)
        and SHA256_PATTERN.fullmatch(expected_sha)
        and sha256_file(path) == expected_sha
    )
    status["sha256"] = sha256_file(path)
    if not status["sha256_exact"]:
        status["errors"].append("policy_audit_sha256")
        return status
    try:
        audit = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        status["errors"].append(f"policy_audit_unreadable:{type(exc).__name__}")
        return status
    if audit.get("schema_version") != contract.get("schema_version"):
        status["errors"].append("policy_audit_schema")
    if audit.get("validator_id") != "execution_policy_scenarios_v1":
        status["errors"].append("policy_audit_validator")
    scenarios = audit.get("scenario_results")
    required_scenarios = set(
        registry["execution_evidence_contract"]["required_policy_scenarios"]
    )
    if not isinstance(scenarios, dict) or not all(
        scenarios.get(value) is True for value in required_scenarios
    ):
        status["errors"].append("policy_scenarios")
    artifacts = audit.get("bound_artifacts")
    required_roles = {"simulator_source", "simulator_config", "mutation_tests"}
    roles: set[str] = set()
    role_counts: dict[str, int] = {}
    if not isinstance(artifacts, list):
        status["errors"].append("policy_bound_artifacts")
    else:
        for value in artifacts:
            if not isinstance(value, dict):
                status["errors"].append("policy_bound_artifact_record")
                continue
            artifact_path = Path(str(value.get("path", "")))
            if not artifact_path.is_absolute():
                artifact_path = repo_root / artifact_path
            artifact_path = artifact_path.resolve()
            role = str(value.get("role", ""))
            roles.add(role)
            role_counts[role] = role_counts.get(role, 0) + 1
            declared_sha = value.get("sha256")
            if isinstance(declared_sha, str):
                status["bound_artifact_sha256"][role] = declared_sha
            if (
                artifact_path not in registered_files
                or not artifact_path.is_file()
                or not isinstance(declared_sha, str)
                or SHA256_PATTERN.fullmatch(declared_sha) is None
                or declared_sha != sha256_file(artifact_path)
            ):
                status["errors"].append(f"policy_artifact:{role}")
        if roles != required_roles:
            status["errors"].append("policy_artifact_roles")
        if any(value != 1 for value in role_counts.values()):
            status["errors"].append("policy_artifact_duplicate_role")
    status["internal_integrity_valid"] = not status["errors"]
    status["errors"].append("external_attestation_not_verified")
    status["valid"] = False
    return status


def validate_execution_manifest(
    registration: dict[str, Any],
    *,
    registry: dict[str, Any],
    repo_root: Path,
    registered_files: set[Path],
    allowed_sessions: set[date],
    expected_decisions: dict[str, dict[str, Any]],
    policy_audit: dict[str, Any],
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "manifest_path": registration.get("path"),
        "manifest_sha256_exact": False,
        "data_path": None,
        "data_sha256_exact": False,
        "rows": 0,
        "sessions": 0,
        "filled_rows": 0,
        "filled_sessions": 0,
        "field_content_valid": False,
        "capacity_valid": False,
        "internal_integrity_valid": False,
        "reference_data_recomputed_from_registered_sources": False,
        "external_origin_authenticated": False,
        "valid": False,
        "errors": [],
    }
    raw_path = registration.get("path")
    expected_manifest_sha = registration.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        status["errors"].append("manifest_path_missing")
        return status
    manifest_path = Path(raw_path)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest_path = manifest_path.resolve()
    status["manifest_path"] = portable_path(manifest_path, repo_root)
    if manifest_path not in registered_files or not manifest_path.is_file():
        status["errors"].append("manifest_not_in_registered_search_files")
        return status
    actual_manifest_sha = sha256_file(manifest_path)
    status["manifest_sha256_exact"] = bool(
        isinstance(expected_manifest_sha, str)
        and SHA256_PATTERN.fullmatch(expected_manifest_sha)
        and actual_manifest_sha == expected_manifest_sha
    )
    if not status["manifest_sha256_exact"]:
        status["errors"].append("manifest_sha256_mismatch")
        return status
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        status["errors"].append(f"manifest_unreadable:{type(exc).__name__}")
        return status
    contract = registry["execution_evidence_contract"]
    if manifest.get("schema_version") != 1:
        status["errors"].append("manifest_schema_version")
    if manifest.get("validator_id") != contract["validator_id"]:
        status["errors"].append("validator_id")
    source_id = str(manifest.get("source_id", "")).strip()
    if source_id not in set(registry["approved_execution_source_ids"]):
        status["errors"].append("source_id_not_approved")
    if not policy_audit.get("valid"):
        status["errors"].append("execution_policy_audit")
    if (
        manifest.get("execution_policy_audit_sha256")
        != policy_audit.get("sha256")
        or manifest.get("simulator_source_sha256")
        != policy_audit.get("bound_artifact_sha256", {}).get(
            "simulator_source"
        )
        or manifest.get("simulator_config_sha256")
        != policy_audit.get("bound_artifact_sha256", {}).get(
            "simulator_config"
        )
    ):
        status["errors"].append("execution_policy_binding")
    if not expected_decisions:
        status["errors"].append("expected_t02_decisions")
    exported_at = aware_datetime(manifest.get("exported_at"))
    request_started_at = aware_datetime(manifest.get("request_started_at"))
    receipt_completed_at = aware_datetime(
        manifest.get("receipt_completed_at")
    )
    parse_completed_at = aware_datetime(manifest.get("parse_completed_at"))
    if exported_at is None:
        status["errors"].append("exported_at")
    if not (
        request_started_at is not None
        and receipt_completed_at is not None
        and parse_completed_at is not None
        and exported_at is not None
        and request_started_at
        <= receipt_completed_at
        <= parse_completed_at
        <= exported_at
    ):
        status["errors"].append("acquisition_chronology")
    raw_data_path = manifest.get("data_path")
    if not isinstance(raw_data_path, str) or not raw_data_path:
        status["errors"].append("data_path")
        return status
    data_path = Path(raw_data_path)
    if not data_path.is_absolute():
        data_path = repo_root / data_path
    data_path = data_path.resolve()
    status["data_path"] = portable_path(data_path, repo_root)
    if data_path not in registered_files or not data_path.is_file():
        status["errors"].append("data_not_in_registered_search_files")
        return status
    try:
        data_size = data_path.stat().st_size
        data_sha = sha256_file(data_path)
    except OSError as exc:
        status["errors"].append(f"data_unreadable:{type(exc).__name__}")
        return status
    status["data_sha256_exact"] = bool(
        manifest.get("data_sha256") == data_sha
        and manifest.get("data_bytes") == data_size
    )
    if not status["data_sha256_exact"]:
        status["errors"].append("data_identity_mismatch")
        return status
    required_columns = set(contract["required_columns"])
    score_start = date.fromisoformat(
        registry["protocol_evaluation_window"]["score_start"]
    )
    score_end = date.fromisoformat(
        registry["protocol_evaluation_window"]["score_end"]
    )
    intended_capital = registry.get("intended_capital_jpy")
    try:
        intended_capital_value = float(intended_capital)
    except (TypeError, ValueError):
        intended_capital_value = float("nan")
    row_errors: list[str] = []
    sessions: set[date] = set()
    filled_sessions: set[date] = set()
    seen_decision_ids: set[str] = set()
    seen_order_ids: set[str] = set()
    daily_notional: dict[date, float] = {}
    daily_executed_notional: dict[date, float] = {}
    code_session_notional: dict[tuple[date, str], float] = {}
    code_session_executed_notional: dict[tuple[date, str], float] = {}
    code_session_turnover: dict[tuple[date, str], tuple[float, float]] = {}
    rows = 0
    filled_rows = 0
    capacity_valid = bool(
        intended_capital_value > 0
        and intended_capital_value < float("inf")
    )
    try:
        with data_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            if not required_columns.issubset(columns):
                row_errors.append("required_columns")
            else:
                for line_number, row in enumerate(reader, start=2):
                    rows += 1
                    try:
                        session = date.fromisoformat(row["session_date"])
                        preopen = aware_datetime(row["preopen_observed_at"])
                        decision = aware_datetime(row["decision_at"])
                        policy_action = aware_datetime(row["policy_action_at"])
                        cancel_requested = aware_datetime(
                            row["cancel_requested_at"]
                        )
                        cancel_ack = aware_datetime(row["cancel_ack_at"])
                        actual_open = aware_datetime(row["actual_open_at"])
                        numeric = {
                            key: float(row[key])
                            for key in (
                                "bid_price_0858",
                                "ask_price_0858",
                                "indicative_price",
                                "simulated_fill_price",
                                "actual_fill_price",
                                "realized_spread_bps",
                                "slippage_bps",
                                "turnover_jpy",
                                "minimum_trading_unit",
                                "effective_tick_size",
                                "predicted_opening_turnover_jpy",
                                "order_quantity",
                                "planned_order_notional_jpy",
                                "order_notional_jpy",
                            )
                        }
                        if not score_start <= session <= score_end:
                            raise ValueError("session_outside_window")
                        if session not in allowed_sessions:
                            raise ValueError("session_not_joint_source_complete")
                        if re.fullmatch(r"[0-9A-Z]{4,5}", row["code"]) is None:
                            raise ValueError("code")
                        decision_id = row["decision_id"].strip()
                        order_id = row["order_id"].strip()
                        if (
                            decision_id in seen_decision_ids
                            or order_id in seen_order_ids
                            or not decision_id
                            or not order_id
                        ):
                            raise ValueError("duplicate_or_empty_order_identity")
                        expected = expected_decisions.get(decision_id)
                        if (
                            expected is None
                            or expected["session"] != session
                            or expected["code"] != row["code"]
                            or expected["decision_at_parsed"] != decision
                        ):
                            raise ValueError("decision_artifact_mismatch")
                        if row["side"].strip().lower() != "buy":
                            raise ValueError("side")
                        if (
                            preopen is None
                            or decision is None
                            or policy_action is None
                            or actual_open is None
                        ):
                            raise ValueError("timestamp")
                        jst = timezone(timedelta(hours=9))
                        cutoff = datetime.combine(
                            session,
                            datetime.strptime("08:58:59", "%H:%M:%S").time(),
                            tzinfo=jst,
                        )
                        if not (
                            preopen.astimezone(jst).date() == session
                            and decision.astimezone(jst).date() == session
                            and preopen.astimezone(jst).time()
                            >= datetime.strptime("08:58:00", "%H:%M:%S").time()
                            and preopen.astimezone(jst) <= cutoff
                            and preopen <= decision
                            and decision.astimezone(jst) <= cutoff
                        ):
                            raise ValueError("preopen_pit")
                        if (
                            actual_open.astimezone(jst).date() != session
                            or actual_open.astimezone(jst).time()
                            < datetime.strptime("09:00:00", "%H:%M:%S").time()
                        ):
                            raise ValueError("actual_open")
                        if (
                            policy_action.astimezone(jst).date() != session
                            or policy_action < decision
                            or policy_action > actual_open
                        ):
                            raise ValueError("policy_action")
                        if not all(
                            value == value and abs(value) < float("inf")
                            for value in numeric.values()
                        ):
                            raise ValueError("nonfinite")
                        if not all(
                            numeric[key] > 0
                            for key in (
                                "bid_price_0858",
                                "ask_price_0858",
                                "indicative_price",
                                "turnover_jpy",
                                "minimum_trading_unit",
                                "effective_tick_size",
                                "predicted_opening_turnover_jpy",
                                "order_quantity",
                                "planned_order_notional_jpy",
                            )
                        ):
                            raise ValueError("nonpositive")
                        if any(
                            numeric[key] < 0
                            for key in (
                                "simulated_fill_price",
                                "actual_fill_price",
                                "order_notional_jpy",
                            )
                        ):
                            raise ValueError("negative_fill_value")
                        lot = numeric["minimum_trading_unit"]
                        tick = numeric["effective_tick_size"]
                        quantity = numeric["order_quantity"]
                        if (
                            lot != int(lot)
                            or quantity != int(quantity)
                            or abs(quantity / lot - round(quantity / lot))
                            > 1e-9
                        ):
                            raise ValueError("lot_multiple")
                        if row["special_quote_0858"].strip().lower() not in {
                            "0",
                            "1",
                            "false",
                            "true",
                            "none",
                            "buy",
                            "sell",
                        }:
                            raise ValueError("special_quote")
                        evidence_mode = row["evidence_mode"].strip().lower()
                        if evidence_mode not in {"broker_fill", "simulation"}:
                            raise ValueError("evidence_mode")
                        outcome = row["execution_outcome"].strip().lower()
                        if outcome not in {
                            "filled",
                            "cancelled_special_quote",
                            "cancelled_delayed_open",
                            "cancelled_liquidity",
                            "unfilled",
                        }:
                            raise ValueError("execution_outcome")
                        special_quote = row["special_quote_0858"].strip().lower()
                        notional = numeric["order_notional_jpy"]
                        planned_notional = numeric["planned_order_notional_jpy"]
                        bid = numeric["bid_price_0858"]
                        ask = numeric["ask_price_0858"]
                        indicative = numeric["indicative_price"]
                        simulated_fill = numeric["simulated_fill_price"]
                        actual_fill = numeric["actual_fill_price"]
                        if not (bid <= indicative <= ask and bid <= ask):
                            raise ValueError("price_relationship")
                        tick_prices = [bid, ask, indicative]
                        if outcome == "filled":
                            tick_prices.append(actual_fill)
                            if simulated_fill > 0:
                                tick_prices.append(simulated_fill)
                        if any(
                            abs(value / tick - round(value / tick)) > 1e-6
                            for value in tick_prices
                        ):
                            raise ValueError("tick_multiple")
                        if not (
                            abs(planned_notional - quantity * indicative)
                            <= max(1.0, planned_notional * 1e-6)
                        ):
                            raise ValueError("planned_notional")
                        latest_open = datetime.combine(
                            session,
                            datetime.strptime(
                                contract["latest_eligible_open_time_jst"],
                                "%H:%M:%S",
                            ).time(),
                            tzinfo=jst,
                        )
                        midpoint = (bid + ask) / 2.0
                        if outcome == "filled":
                            if (
                                actual_fill <= 0
                                or (
                                    evidence_mode == "simulation"
                                    and (
                                        simulated_fill <= 0
                                        or abs(simulated_fill - actual_fill) > 1e-9
                                    )
                                )
                                or special_quote not in {"0", "false", "none"}
                                or actual_open > latest_open
                                or policy_action > cutoff
                                or cancel_requested is not None
                                or cancel_ack is not None
                            ):
                                raise ValueError("filled_policy")
                            computed_spread = (
                                2.0
                                * (actual_fill - midpoint)
                                / midpoint
                                * 10_000.0
                            )
                            computed_slippage = (
                                (actual_fill - indicative)
                                / indicative
                                * 10_000.0
                            )
                            if (
                                abs(
                                    numeric["realized_spread_bps"]
                                    - computed_spread
                                )
                                > 1e-6
                                or abs(
                                    numeric["slippage_bps"]
                                    - computed_slippage
                                )
                                > 1e-6
                            ):
                                raise ValueError(
                                    "spread_or_slippage_recalculation"
                                )
                            if not (
                                abs(notional - quantity * actual_fill)
                                <= max(1.0, notional * 1e-6)
                            ):
                                raise ValueError("notional")
                            filled_rows += 1
                            filled_sessions.add(session)
                        else:
                            if not (
                                simulated_fill == 0
                                and actual_fill == 0
                                and numeric["realized_spread_bps"] == 0
                                and numeric["slippage_bps"] == 0
                                and notional == 0
                            ):
                                raise ValueError("nonfill_has_fill_values")
                            if outcome.startswith("cancelled_"):
                                if (
                                    cancel_requested is None
                                    or cancel_ack is None
                                    or not (
                                        decision
                                        <= policy_action
                                        <= cancel_requested
                                        <= cancel_ack
                                        <= actual_open
                                    )
                                ):
                                    raise ValueError("cancel_timestamps")
                            elif cancel_requested is not None or cancel_ack is not None:
                                raise ValueError("unfilled_cancel_timestamps")
                            if (
                                outcome == "cancelled_special_quote"
                                and (
                                    special_quote
                                    not in {"1", "true", "buy", "sell"}
                                    or policy_action > cutoff
                                )
                            ):
                                raise ValueError("special_quote_cancel")
                            if (
                                outcome == "cancelled_delayed_open"
                                and (
                                    special_quote not in {"0", "false", "none"}
                                    or actual_open <= latest_open
                                    or cancel_requested < latest_open
                                )
                            ):
                                raise ValueError("delayed_open_cancel")
                            if (
                                outcome == "cancelled_liquidity"
                                and (
                                    special_quote not in {"0", "false", "none"}
                                    or policy_action > cutoff
                                    or (
                                        planned_notional
                                        <= intended_capital_value
                                        and planned_notional
                                        <= numeric["turnover_jpy"] * 0.005
                                        and planned_notional
                                        <= numeric[
                                            "predicted_opening_turnover_jpy"
                                        ]
                                        * 0.05
                                    )
                                )
                            ):
                                raise ValueError("liquidity_cancel")
                            if outcome == "unfilled" and (
                                special_quote not in {"0", "false", "none"}
                                or actual_open > latest_open
                                or policy_action > cutoff
                            ):
                                raise ValueError("unfilled_policy")
                        key = (session, row["code"])
                        submitted = outcome in {
                            "filled",
                            "cancelled_delayed_open",
                            "unfilled",
                        }
                        if submitted:
                            daily_notional[session] = (
                                daily_notional.get(session, 0.0)
                                + planned_notional
                            )
                            code_session_notional[key] = (
                                code_session_notional.get(key, 0.0)
                                + planned_notional
                            )
                            prior_liquidity = code_session_turnover.get(key)
                            current_liquidity = (
                                numeric["turnover_jpy"],
                                numeric["predicted_opening_turnover_jpy"],
                            )
                            if (
                                prior_liquidity is not None
                                and prior_liquidity != current_liquidity
                            ):
                                raise ValueError("inconsistent_liquidity")
                            code_session_turnover[key] = current_liquidity
                        if outcome == "filled":
                            daily_executed_notional[session] = (
                                daily_executed_notional.get(session, 0.0)
                                + notional
                            )
                            code_session_executed_notional[key] = (
                                code_session_executed_notional.get(key, 0.0)
                                + notional
                            )
                        seen_decision_ids.add(decision_id)
                        seen_order_ids.add(order_id)
                        sessions.add(session)
                    except (KeyError, TypeError, ValueError) as exc:
                        row_errors.append(
                            f"row_{line_number}:{type(exc).__name__}:{exc}"
                        )
    except (OSError, UnicodeError, csv.Error) as exc:
        row_errors.append(f"csv_unreadable:{type(exc).__name__}")
    status["rows"] = rows
    status["sessions"] = len(sessions)
    status["filled_rows"] = filled_rows
    status["filled_sessions"] = len(filled_sessions)
    if rows < int(contract["minimum_rows"]):
        row_errors.append("minimum_rows")
    if len(sessions) < int(contract["minimum_sessions"]):
        row_errors.append("minimum_sessions")
    if filled_rows < int(contract["minimum_filled_rows"]):
        row_errors.append("minimum_filled_rows")
    if len(filled_sessions) < int(contract["minimum_filled_sessions"]):
        row_errors.append("minimum_filled_sessions")
    if seen_decision_ids != set(expected_decisions):
        row_errors.append("decision_set_not_exact")
    if len(seen_order_ids) != rows:
        row_errors.append("order_id_set_not_exact")
    capacity_valid = bool(
        capacity_valid
        and all(
            value <= intended_capital_value
            for value in daily_notional.values()
        )
        and all(
            value <= intended_capital_value
            for value in daily_executed_notional.values()
        )
        and all(
            code_session_notional[key] <= liquidity[0] * 0.005
            and code_session_notional[key] <= liquidity[1] * 0.05
            for key, liquidity in code_session_turnover.items()
        )
        and all(
            code_session_executed_notional.get(key, 0.0)
            <= liquidity[0] * 0.005
            and code_session_executed_notional.get(key, 0.0)
            <= liquidity[1] * 0.05
            for key, liquidity in code_session_turnover.items()
        )
    )
    status["field_content_valid"] = not row_errors
    status["capacity_valid"] = bool(not row_errors and capacity_valid)
    status["errors"].extend(row_errors)
    status["internal_integrity_valid"] = bool(
        not status["errors"]
        and status["manifest_sha256_exact"]
        and status["data_sha256_exact"]
        and status["field_content_valid"]
        and status["capacity_valid"]
    )
    status["valid"] = bool(
        status["internal_integrity_valid"]
        and status["reference_data_recomputed_from_registered_sources"]
        and status["external_origin_authenticated"]
    )
    return status


def discover_field_artifacts(
    files: Iterable[Path],
    repo_root: Path,
    registry: dict[str, Any],
    allowed_sessions: set[date],
) -> dict[str, Any]:
    ignored_roots = {
        (repo_root / "research").resolve(),
        (repo_root / "tests").resolve(),
        (repo_root / "src").resolve(),
    }
    candidates: list[dict[str, Any]] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix not in DATA_EXTENSIONS:
            continue
        resolved = path.resolve()
        if any(root == resolved or root in resolved.parents for root in ignored_roots):
            continue
        lowered = path.name.lower()
        if not any(keyword in lowered for keyword in FIELD_KEYWORDS):
            continue
        header = delimited_header_candidate(path)
        candidates.append(
            {
                "path": portable_path(resolved, repo_root),
                "extension": suffix,
                "bytes": path.stat().st_size,
                **header,
                "classification": "unregistered_header_candidate_only",
                "usable_for_gate": False,
                "rejection_reason": (
                    "No explicit frozen evidence manifest, content reader, "
                    "OOT row/date/PIT/nonmissingness coverage, provenance/hash "
                    "binding, or joint-session coverage was registered."
                ),
            }
        )
    registered_files = {value.resolve() for value in files}
    expected_decisions, decision_artifact = load_registered_t02_decisions(
        registry=registry,
        repo_root=repo_root,
        registered_files=registered_files,
        allowed_sessions=allowed_sessions,
    )
    decision_replay_audit = validate_t02_decision_replay_audit(
        registry=registry,
        repo_root=repo_root,
        registered_files=registered_files,
        decision_artifact=decision_artifact,
    )
    policy_audit = validate_execution_policy_audit(
        registry=registry,
        repo_root=repo_root,
        registered_files=registered_files,
    )
    validated = [
        validate_execution_manifest(
            value,
            registry=registry,
            repo_root=repo_root,
            registered_files=registered_files,
            allowed_sessions=allowed_sessions,
            expected_decisions=expected_decisions,
            policy_audit=policy_audit,
        )
        for value in registry.get(
            "registered_execution_evidence_manifests", []
        )
    ]
    valid_artifacts = [value for value in validated if value["valid"]]
    registry_contract_valid = bool(
        decision_artifact["valid"]
        and decision_replay_audit["valid"]
        and policy_audit["valid"]
        and validated
        and len(valid_artifacts) == len(validated)
    )
    fields = {key: False for key in FIELD_REQUIREMENTS}
    if registry_contract_valid:
        for key, requirement in FIELD_REQUIREMENTS.items():
            if requirement["gate"] == "execution":
                fields[key] = True
    return {
        "search_method": (
            "diagnostic filename-keyword scan of CSV/TSV headers only; "
            "research, tests, and source-code directories excluded"
        ),
        "certification_scope": (
            "registry-bound execution-ledger validation plus diagnostic "
            "header scan; this does not certify production"
        ),
        "positive_evidence_requirements": [
            "explicit frozen evidence manifest and content reader",
            "nonempty rows in the registered OOT date window",
            "field types and nonmissingness coverage",
            "available_at/observed_at PIT cutoff validation",
            "source identity, byte count, SHA-256, request/receipt/parse chronology",
            "joint source-complete session coverage",
            "an exact, hash-bound T02 decision row for every joint-complete session",
            "a hash-bound independent replay audit rejecting code/cash substitution",
            "an approved execution source identity and exact order/decision linkage",
            "a hash-bound simulator/policy audit covering normal, cancel, tick, lot, and liquidity cases",
            "spread, slippage, notional, daily capital, and participation recomputation",
            "dated tick/lot and turnover values rederived from registered raw sources",
            "provider-authenticated origin or independently verified execution attestation",
        ],
        "header_candidates": candidates,
        "header_candidate_count": len(candidates),
        "registered_artifact_results": validated,
        "registered_t02_decision_artifact": decision_artifact,
        "registered_t02_decision_replay_audit": decision_replay_audit,
        "registered_execution_policy_audit": policy_audit,
        "registered_validated_artifacts": valid_artifacts,
        "registered_validated_artifact_count": len(valid_artifacts),
        "registered_execution_contract_valid": registry_contract_valid,
        "intended_capital_capacity_valid": bool(
            registry_contract_valid
            and all(value["capacity_valid"] for value in valid_artifacts)
        ),
        "fields": fields,
    }


def build_blockers(
    protocol: dict[str, Any],
    freeze: dict[str, Any],
    jpx: dict[str, Any],
    tdnet: dict[str, Any],
    field_evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []

    def add(
        identifier: str,
        requirement: str,
        observed: str,
        *,
        gate: str,
    ) -> None:
        blockers.append(
            {
                "id": identifier,
                "severity": "blocking",
                "gate": gate,
                "requirement": requirement,
                "observed": observed,
            }
        )

    if not freeze["all_present_and_exact"]:
        failed = [
            key
            for key, value in freeze["artifacts"].items()
            if not value["sha256_exact"]
        ]
        add(
            "BOUND_CANDIDATE_HASH_MISMATCH",
            "Every frozen T02 artifact must be present with its registered SHA-256.",
            f"non-exact artifacts: {failed}",
            gate="statistical_input",
        )
    if not jpx["audit_contract_exact"]:
        add(
            "JPX_AUDIT_CONTRACT_INCONSISTENT",
            "The parser audit must exactly enumerate the warmup plus registered score sessions with zero rejected rows.",
            "parser audit counts/dates/rejections do not match the OOT protocol",
            gate="statistical_input",
        )
    if jpx["raw_files_hash_exact"] < jpx["expected_files_including_warmup"]:
        add(
            "JPX_RAW_FILES_MISSING_OR_MISMATCHED",
            "Official JPX daily quotation PDFs must be locally present and match the audited per-file SHA-256 values.",
            f"{jpx['raw_files_hash_exact']}/{jpx['expected_files_including_warmup']} hash-exact PDFs; missing={len(jpx['missing_raw_files'])}, mismatched={len(jpx['hash_mismatch_files'])}",
            gate="statistical_input",
        )
    if not jpx["parser"]["available_and_exact"]:
        add(
            "JPX_PARSER_NOT_HASH_EXACT",
            "The parser implementation must match the parser-recovery audit SHA-256.",
            f"expected {jpx['parser']['expected_sha256']}; matches={jpx['parser']['matching_paths']}",
            gate="statistical_input",
        )
    if not jpx["sealed_manifest"]["contract_exact"]:
        add(
            "JPX_REGISTERED_MANIFEST_UNAVAILABLE",
            "The registry-bound acquisition/parse manifest must be present and hash-exact for source and parser provenance.",
            (
                f"path={jpx['sealed_manifest']['path']}; "
                f"exists={jpx['sealed_manifest']['exists']}; "
                f"registered_search_file="
                f"{jpx['sealed_manifest']['registered_search_file']}; "
                f"sha256_exact={jpx['sealed_manifest']['sha256_exact']}; "
                f"contract_exact={jpx['sealed_manifest']['contract_exact']}"
            ),
            gate="statistical_input",
        )
    if (
        jpx["provenance_complete_score_sessions"]
        < protocol["evaluation_window"]["minimum_source_complete_sessions"]
    ):
        add(
            "JPX_PROVENANCE_BELOW_MINIMUM",
            "At least 120 score sessions require source URL, byte count, SHA-256, request/receipt timestamps, parser version, and parse completion.",
            f"{jpx['provenance_complete_score_sessions']} provenance-complete score sessions",
            gate="statistical_input",
        )
    if tdnet["pages_present"] < tdnet["expected_calendar_pages"]:
        add(
            "TDNET_DAILY_PAGES_MISSING",
            "Every calendar-date TDnet index from warmup through score_end must be locally present.",
            f"{tdnet['pages_present']}/{tdnet['expected_calendar_pages']} pages; {len(tdnet['missing_page_dates'])} missing",
            gate="statistical_input",
        )
    if not tdnet["registered_page_manifest"]["contract_exact"]:
        manifest = tdnet["registered_page_manifest"]
        add(
            "TDNET_REGISTERED_MANIFEST_UNAVAILABLE",
            "The registry-bound TDnet page/metadata manifest must be present, hash-exact, unique, and enumerate all 243 dates.",
            (
                f"path={manifest['path']}; exists={manifest['exists']}; "
                f"registered_search_file={manifest['registered_search_file']}; "
                f"sha256_exact={manifest['sha256_exact']}; "
                f"contract_exact={manifest['contract_exact']}"
            ),
            gate="statistical_input",
        )
    if not tdnet["parser"]["available_and_exact"]:
        add(
            "TDNET_PARSER_NOT_HASH_EXACT",
            "The TDnet parser implementation must match the separately frozen SHA-256.",
            (
                f"expected {tdnet['parser']['expected_sha256']}; "
                f"actual {tdnet['parser']['actual_sha256']}"
            ),
            gate="statistical_input",
        )
    if tdnet["source_complete_pages"] < tdnet["expected_calendar_pages"]:
        add(
            "TDNET_SOURCE_COMPLETENESS_MISSING",
            "Every TDnet page must be parser-valid, finalized, and have integrity-valid acquisition metadata; a gap is not a no-event day.",
            f"{tdnet['source_complete_pages']}/{tdnet['expected_calendar_pages']} source-complete pages",
            gate="statistical_input",
        )
    if tdnet["provenance_complete_pages"] < tdnet["expected_calendar_pages"]:
        add(
            "TDNET_PROVENANCE_INCOMPLETE",
            "Every TDnet page needs source URL, byte count, SHA-256, request start, receipt completion, parser version, and parse completion.",
            f"{tdnet['provenance_complete_pages']}/{tdnet['expected_calendar_pages']} provenance-complete pages",
            gate="statistical_input",
        )
    joint_dates = sorted(
        set(jpx["provenance_complete_score_dates"])
        & set(tdnet["provenance_complete_score_dates"])
    )
    minimum_sessions = int(
        protocol["evaluation_window"]["minimum_source_complete_sessions"]
    )
    joint_months = sorted({value[:7] for value in joint_dates})
    minimum_months = int(
        protocol["evaluation_window"]["minimum_calendar_months"]
    )
    if len(joint_dates) < minimum_sessions:
        add(
            "JOINT_COVERAGE_BELOW_MINIMUM",
            "JPX and TDnet must be jointly provenance-complete for at least the registered minimum number of score sessions.",
            f"{len(joint_dates)}/{minimum_sessions} joint score sessions",
            gate="statistical_input",
        )
    if len(joint_months) < minimum_months:
        add(
            "JOINT_MONTHS_BELOW_MINIMUM",
            "Jointly provenance-complete score sessions must span at least the registered minimum calendar months.",
            f"{len(joint_months)}/{minimum_months} months: {joint_months}",
            gate="statistical_input",
        )
    for group in EXECUTION_EVIDENCE_GROUPS:
        if any(field_evidence["fields"][key] for key in group):
            continue
        if len(group) > 1:
            add(
                "FIELD_EXECUTION_PRICE_OR_FILL_MISSING",
                "Either exact 08:58 indicative price or an executable order simulation is required.",
                "Neither field has an explicitly registered and content-validated OOT artifact.",
                gate="execution_input",
            )
            continue
        key = group[0]
        add(
            f"FIELD_{key.upper()}_MISSING",
            FIELD_REQUIREMENTS[key]["label"],
            "No explicitly registered and content-validated OOT artifact was supplied.",
            gate="execution_input",
        )
    if not field_evidence["intended_capital_capacity_valid"]:
        add(
            "CAPACITY_AT_INTENDED_CAPITAL_NOT_EVALUABLE",
            "Capacity must be checked at the intended capital size.",
            "No registry-bound execution ledger passed lot, notional, turnover, predicted-opening-turnover, and intended-capital checks.",
            gate="execution_input",
        )
    return blockers


def build_readiness(
    protocol_path: str | Path,
    parser_audit_path: str | Path,
    *,
    evidence_registry_path: str | Path | None = None,
    roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    protocol_path = Path(protocol_path).resolve()
    parser_audit_path = Path(parser_audit_path).resolve()
    if evidence_registry_path is None:
        repo_root = protocol_path.parent.parent
        evidence_registry_path = repo_root / DEFAULT_EVIDENCE_REGISTRY
    evidence_registry_path = Path(evidence_registry_path).resolve()
    protocol = read_json(protocol_path)
    audit = read_json(parser_audit_path)
    registry = read_json(evidence_registry_path)
    if protocol.get("protocol_id") != "model_v11_t02_out_of_time_202508_202603":
        raise ValueError("unexpected T02 OOT protocol id")
    if audit.get("record_type") != "jpx_parser_recovery_audit":
        raise ValueError("unexpected parser audit record type")
    if (
        registry.get("registry_id")
        != "model_v11_t02_data_evidence_contract_20260723"
    ):
        raise ValueError("unexpected data-evidence registry id")
    if sha256_file(evidence_registry_path) != FROZEN_EVIDENCE_REGISTRY_SHA256:
        raise ValueError("data-evidence registry SHA-256 is not frozen")
    if registry.get("protocol", {}).get("sha256") != sha256_file(protocol_path):
        raise ValueError("data-evidence registry does not bind this protocol")
    if (
        registry.get("parser_audit", {}).get("sha256")
        != sha256_file(parser_audit_path)
    ):
        raise ValueError("data-evidence registry does not bind this parser audit")
    registered_window = registry.get("protocol_evaluation_window", {})
    if (
        registered_window.get("score_start")
        != protocol["evaluation_window"]["score_start"]
        or registered_window.get("score_end")
        != protocol["evaluation_window"]["score_end"]
    ):
        raise ValueError("data-evidence registry evaluation window mismatch")
    repo_root, search_roots = canonical_roots(protocol_path, registry, roots)
    files = walk_files(search_roots)
    jpx_records = jpx_expected_records(audit)
    expected_names = {value["name"] for value in jpx_records}
    expected_tdnet = tdnet_expected_dates(protocol)
    expected_names.update(value.strftime("%Y%m%d.html") for value in expected_tdnet)
    expected_names.update(
        value.strftime("%Y%m%d.html.meta.json") for value in expected_tdnet
    )
    registered_manifest_path = (
        registry["jpx"]["registered_acquisition_parse_manifest"].get("path")
    )
    if isinstance(registered_manifest_path, str) and registered_manifest_path:
        expected_names.add(Path(registered_manifest_path).name)
    tdnet_manifest_path = registry["tdnet"]["registered_page_manifest"].get(
        "path"
    )
    if isinstance(tdnet_manifest_path, str) and tdnet_manifest_path:
        expected_names.add(Path(tdnet_manifest_path).name)
    file_index = index_by_name(files, expected_names)
    freeze = verify_bound_artifacts(protocol, repo_root)
    jpx = verify_jpx(
        protocol, audit, registry, file_index, files, repo_root
    )
    tdnet = verify_tdnet(
        protocol, registry, jpx_records, file_index, repo_root, files
    )
    minimum = int(protocol["evaluation_window"]["minimum_source_complete_sessions"])
    minimum_months = int(protocol["evaluation_window"]["minimum_calendar_months"])
    joint_dates = sorted(
        set(jpx["provenance_complete_score_dates"])
        & set(tdnet["provenance_complete_score_dates"])
    )
    field_evidence = discover_field_artifacts(
        files,
        repo_root,
        registry,
        {date.fromisoformat(value) for value in joint_dates},
    )
    joint_months = sorted({value[:7] for value in joint_dates})
    blockers = build_blockers(protocol, freeze, jpx, tdnet, field_evidence)
    statistical_blockers = [
        value for value in blockers if value["gate"] == "statistical_input"
    ]
    execution_blockers = [
        value for value in blockers if value["gate"] == "execution_input"
    ]
    statistical_input_ready = not statistical_blockers
    full_window_ready = bool(
        statistical_input_ready
        and len(joint_dates)
        == int(protocol["evaluation_window"]["expected_score_sessions"])
    )
    execution_fields_ready = all(
        any(field_evidence["fields"][key] for key in group)
        for group in EXECUTION_EVIDENCE_GROUPS
    )
    execution_input_ready = bool(
        statistical_input_ready
        and execution_fields_ready
        and not execution_blockers
    )
    unavailable_context = [
        key
        for key, requirement in FIELD_REQUIREMENTS.items()
        if requirement["gate"] == "research_context"
        and not field_evidence["fields"][key]
    ]
    source_paths = {
        "protocol": portable_path(protocol_path, repo_root),
        "parser_audit": portable_path(parser_audit_path, repo_root),
        "evidence_registry": portable_path(evidence_registry_path, repo_root),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "verifier_id": VERIFIER_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "network_accessed": False,
        "inputs": {
            **source_paths,
            "protocol_sha256": sha256_file(protocol_path),
            "parser_audit_sha256": sha256_file(parser_audit_path),
            "evidence_registry_sha256": sha256_file(evidence_registry_path),
            "search_roots": [
                portable_path(value, repo_root) for value in search_roots
            ],
            "files_walked": len(files),
            "trust_scope": (
                "repository-hash-bound evidence contract; no external "
                "signature or release attestation"
            ),
        },
        "window": {
            **protocol["evaluation_window"],
            "expected_tdnet_calendar_pages": len(expected_tdnet),
        },
        "candidate_freeze": freeze,
        "jpx": jpx,
        "tdnet": tdnet,
        "execution_and_context_data": {
            **field_evidence,
            "required_execution_evidence_groups": [
                list(value) for value in EXECUTION_EVIDENCE_GROUPS
            ],
            "unavailable_optional_research_context_fields": unavailable_context,
            "jpx_daily_parser_capability_only": {
                "fields": [
                    "final_special_quote",
                    "volume",
                    "turnover",
                    "trading_unit",
                ],
                "evidence": "The hash-exact parser source defines these outputs and the audit reports zero rejected rows.",
                "usable_for_gate": False,
                "reason": "The raw PDFs/parsed export and field-level nonmissing coverage are not locally available; final_special_quote is not an exact 08:58 quote state.",
            },
        },
        "readiness": {
            "scope": "data_readiness_only_not_production_certification",
            "can_certify_production": False,
            "external_evidence_signature_present": False,
            "execution_internal_integrity_only": execution_input_ready,
            "execution_external_origin_authenticated": False,
            "genuinely_untouched_holdout_present": False,
            "fresh_paper_live_or_later_holdout_required": True,
            "joint_provenance_complete_score_sessions": len(joint_dates),
            "joint_provenance_complete_months": joint_months,
            "minimum_source_complete_sessions": minimum,
            "minimum_calendar_months": minimum_months,
            "bound_t02_no_tuning_artifacts_exact": freeze[
                "all_present_and_exact"
            ],
            "full_159_session_window_reproducible": full_window_ready,
            "statistical_input_data_ready": statistical_input_ready,
            "t02_exact_no_tuning_oot_statistical_gate_can_run": statistical_input_ready,
            "execution_fields_ready": execution_fields_ready,
            "execution_input_data_ready": execution_input_ready,
            "t02_execution_gate_can_run": execution_input_ready,
            "statistical_blocker_ids": [
                value["id"] for value in statistical_blockers
            ],
            "execution_blocker_ids": [
                value["id"] for value in execution_blockers
            ],
            "signed_statistical_result_present": False,
            "signed_execution_result_present": False,
            "intended_capital_capacity_result_present": False,
            "independent_production_approval_present": False,
            "production_promotion_evaluable": False,
            "production_ready": False,
            "production_ready_reason": (
                "This is a data-readiness verifier, not a production "
                "certifier. "
                f"statistical_input_data_ready={statistical_input_ready}; "
                f"execution_input_data_ready={execution_input_ready}. "
                "The registered historical window is explicitly not a "
                "genuinely untouched holdout, and repository hashes prove "
                "internal identity rather than external origin. "
                "No signed statistical/execution gate results or independent "
                "production approval record are present."
            ),
        },
        "blockers": blockers,
        "blocker_count": len(blockers),
        "blocker_counts_by_gate": {
            "statistical_input": len(statistical_blockers),
            "execution_input": len(execution_blockers),
        },
        "blocker_count_semantics": (
            "Overlapping failed requirements, not independent missing datasets."
        ),
    }


def write_report(path: str | Path, result: dict[str, Any]) -> None:
    jpx = result["jpx"]
    tdnet = result["tdnet"]
    readiness = result["readiness"]
    freeze = result["candidate_freeze"]
    fields = result["execution_and_context_data"]["fields"]
    lines = [
        "# T02 OOT data-readiness / data-gap verification",
        "",
        "## 結論",
        "",
        "**Fail closed: T02のexact no-tuning OOT統計gateもexecution gateも現在は実行不能です。**",
        "",
        f"- Frozen T02 artifact hashes: "
        f"{'4/4 exact' if freeze['all_present_and_exact'] else 'NOT exact'}",
        f"- Joint provenance-complete score sessions: "
        f"{readiness['joint_provenance_complete_score_sessions']}/"
        f"{readiness['minimum_source_complete_sessions']} minimum",
        f"- Required calendar months: "
        f"{len(readiness['joint_provenance_complete_months'])}/"
        f"{readiness['minimum_calendar_months']} minimum",
        f"- Exact no-tuning statistical gate can run: "
        f"{readiness['t02_exact_no_tuning_oot_statistical_gate_can_run']}",
        f"- Execution gate can run: {readiness['t02_execution_gate_can_run']}",
        f"- Verifier scope: {readiness['scope']}",
        f"- Can certify production: {readiness['can_certify_production']}",
        f"- Genuinely untouched holdout present: "
        f"{readiness['genuinely_untouched_holdout_present']}",
        f"- Fresh paper-live/later holdout required: "
        f"{readiness['fresh_paper_live_or_later_holdout_required']}",
        "",
        "## JPX price window",
        "",
        f"- Expected files: {jpx['expected_files_including_warmup']} "
        f"(warmup 1 + score {jpx['expected_score_sessions']})",
        f"- Prior parser audit reports source complete: "
        f"{jpx['audit_source_complete_files']}/{jpx['expected_files_including_warmup']} "
        f"files, {jpx['audit_source_complete_score_sessions']}/"
        f"{jpx['expected_score_sessions']} score sessions",
        f"- Raw PDFs currently present and hash-exact: "
        f"{jpx['raw_files_hash_exact']}/{jpx['expected_files_including_warmup']}",
        f"- Currently reproducible source-complete score sessions: "
        f"{jpx['currently_reproducible_source_complete_score_sessions']}/"
        f"{jpx['expected_score_sessions']}",
        f"- Provenance-complete score sessions: "
        f"{jpx['provenance_complete_score_sessions']}/"
        f"{jpx['expected_score_sessions']}",
        f"- Parser hash exact: {jpx['parser']['available_and_exact']} "
        f"({', '.join(jpx['parser']['matching_paths']) or 'no match'})",
        f"- Registry-bound acquisition/parse manifest available/contract-exact: "
        f"{jpx['sealed_manifest']['exists']}/"
        f"{jpx['sealed_manifest']['contract_exact']}",
        "",
        "prior監査JSONは160 PDF・622,724 parsed rows・reject 0と記録しています。"
        "ただしraw bytesとsealed acquisition manifestが無いため、現在のworkspaceから"
        "同じ入力を再構成してその記録を再証明することはできません。",
        "",
        "## TDnet window and provenance",
        "",
        f"- Expected calendar-date pages: {tdnet['expected_calendar_pages']} "
        f"({tdnet['expected_date_start']} … {tdnet['expected_date_end']})",
        f"- Pages present: {tdnet['pages_present']}/{tdnet['expected_calendar_pages']}",
        f"- Metadata sidecars present: "
        f"{tdnet['metadata_sidecars_present']}/{tdnet['expected_calendar_pages']}",
        f"- Parser-valid/source-complete/provenance-complete pages: "
        f"{tdnet['parser_valid_pages']}/{tdnet['source_complete_pages']}/"
        f"{tdnet['provenance_complete_pages']}",
        f"- Provenance-complete score sessions: "
        f"{tdnet['provenance_complete_score_sessions']}/"
        f"{jpx['expected_score_sessions']}",
        "",
        "不足pageをno-eventとして扱うことは禁止されているため、TDnet 0件日は生成していません。",
        "",
        "## Execution and context fields",
        "",
        "| Field | Explicitly registered and content-validated OOT evidence |",
        "|---|---:|",
    ]
    labels = {
        "exact_0858_indicative_price": "Exact 08:58 indicative price",
        "executable_order_simulation": "Executable order simulation/fill",
        "bid_ask_spread": "08:58 bid/ask spread",
        "realized_spread": "Realized spread",
        "slippage": "Slippage",
        "special_quote_0858": "08:58 special quote",
        "open_time_or_delayed_open": "Open time / delayed-open handling",
        "futures_0858": "Exact 08:58 futures",
        "pts_price_and_volume": "PTS price and volume",
        "volume": "Volume",
        "turnover": "Turnover",
        "minimum_trading_unit": "Minimum trading unit",
        "tick_size": "Tick size",
        "predicted_opening_turnover": "Predicted opening turnover",
    }
    for key, label in labels.items():
        lines.append(f"| {label} | {'yes' if fields[key] else 'no'} |")
    lines.extend(
        [
            "",
            "JPX parser codeにはfinal special quote、volume、turnover、trading unitの"
            "出力能力がありますが、現物raw/parsed exportとfield-level coverageが無く、"
            "final special quoteも08:58 snapshotではありません。このためgate証拠には"
            "昇格させていません。",
            "",
            "ファイル名・CSV/TSV headerの探索結果はdiagnostic candidateにすぎません。"
            "空のheader-only file、OOT外の行、PIT/provenance/hash未検証fileは"
            "field evidenceとして数えません。positive certificationには明示的に"
            "凍結したmanifestとcontent readerが必要です。",
            "",
            "execution ledgerは、hash-bound T02 decision artifactと独立replay audit、"
            "approved source、policy auditへ結合し、全order decisionをfilled、"
            "special-quote cancel、delayed-open cancel、liquidity cancel、unfilled"
            "のいずれかで完全被覆する必要があります。08:58秒内観測、tick/lot、"
            "spread/slippage、予定注文額の日次合計と参加率は入力値から再計算します。",
            "",
            "## Exact blockers",
            "",
        ]
    )
    for blocker in result["blockers"]:
        lines.append(
            f"- `{blocker['id']}` ({blocker['gate']}) — {blocker['requirement']} "
            f"Observed: {blocker['observed']}"
        )
    lines.extend(
        [
            "",
            "blocker件数は重複し得るfailed requirement数であり、独立した欠測dataset数ではありません。",
            "",
        "## Production boundary",
        "",
        "このverifierはrepository-hash-boundなdata readinessだけを判定し、"
        "外部署名を検証しません。production昇格には、"
        "別成果物として署名・hash bindingされた統計結果、execution結果、"
            "intended-capital capacity結果、独立承認が必要です。",
            "また、登録historical windowはgenuinely untouchedではないため、"
            "事前固定したpaper-liveまたはさらに後年のholdoutが別途必要です。",
        ]
    )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "No network access is used.",
            "",
            "```bash",
            "PYTHONPATH=src:. .venv/bin/python research/model_v11_production_readiness.py",
            "PYTHONPATH=src:. .venv/bin/pytest -q tests/test_model_v11_production_readiness.py",
            "```",
            "",
            f"- Protocol SHA-256: `{result['inputs']['protocol_sha256']}`",
            f"- Parser audit SHA-256: `{result['inputs']['parser_audit_sha256']}`",
            f"- Data-evidence registry SHA-256: "
            f"`{result['inputs']['evidence_registry_sha256']}`",
            f"- Search roots: {', '.join(result['inputs']['search_roots'])}",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default="research/model_v11_t02_oot_protocol.json"
    )
    parser.add_argument(
        "--parser-audit", default="research/model_v04_parser_recovery_audit.json"
    )
    parser.add_argument(
        "--evidence-registry", default=DEFAULT_EVIDENCE_REGISTRY
    )
    parser.add_argument(
        "--output", default="research/model_v11_production_readiness.json"
    )
    parser.add_argument(
        "--report", default="research/model_v11_production_readiness_report.md"
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Optional explicit read-only search root; may be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_readiness(
        args.protocol,
        args.parser_audit,
        evidence_registry_path=args.evidence_registry,
        roots=args.root,
    )
    write_json(args.output, result)
    write_report(args.report, result)
    print(
        "statistical_input_data_ready="
        f"{result['readiness']['statistical_input_data_ready']} "
        "execution_input_data_ready="
        f"{result['readiness']['execution_input_data_ready']} "
        "statistical_gate_can_run="
        f"{result['readiness']['t02_exact_no_tuning_oot_statistical_gate_can_run']} "
        "execution_gate_can_run="
        f"{result['readiness']['t02_execution_gate_can_run']} "
        f"blockers={result['blocker_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
