#!/usr/bin/env python3
"""Audit the v5 -> v6 JPX parser recovery without opening return labels.

The audit converts every sealed PDF once, feeds the identical text to both
parser versions, and proves that v6 preserves every row accepted by v5 while
adding only rows whose Final Special Quote carries JPX's half-width ``ｶ`` or
``ｳ`` side marker.  It never builds features, candidates, or P&L.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from tse_session_ranker.data import jpx as current_jpx


EXPECTED_OLD_PARSER_SHA256 = (
    "9b4a38cb4fd02b4248c3cc5edc8e10b54407ba14eb6cf9c89e02184df2e101e7"
)
EXPECTED_OLD_REJECTED_ROWS = 63
EXPECTED_AFFECTED_FILES = 47
SPECIAL_QUOTE_MARKER = re.compile(
    r"^[ｶｳ][+-]?[0-9][0-9,]*(?:\.[0-9]+)?$"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_old_parser(path: Path) -> ModuleType:
    if sha256_file(path) != EXPECTED_OLD_PARSER_SHA256:
        raise ValueError("old parser copy does not match the selection lock")
    module_name = "tse_session_ranker.data._jpx_recovery_v5"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the old JPX parser")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sealed_paths(manifest_path: Path) -> list[Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 159:
        raise ValueError("sealed manifest must contain 159 scored PDFs")
    paths = [manifest_path.parent / "stq_20250801.pdf"]
    paths.extend(Path(item["path"]) for item in files)
    if len(paths) != 160 or any(not path.is_file() for path in paths):
        raise ValueError("sealed JPX PDF set is incomplete")
    return paths


def keyed(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["date", "code"]
    if frame.duplicated(keys).any():
        raise ValueError("daily JPX output has duplicate date/code rows")
    return frame.sort_values(keys).set_index(keys, drop=False)


def assert_common_rows_identical(old: pd.DataFrame, new: pd.DataFrame) -> None:
    old_keyed = keyed(old)
    new_keyed = keyed(new)
    missing = old_keyed.index.difference(new_keyed.index)
    if len(missing):
        raise AssertionError(f"v6 removed rows accepted by v5: {missing.tolist()}")
    columns = list(old_keyed.columns)
    if columns != list(new_keyed.columns):
        raise AssertionError("v6 changed the canonical JPX frame schema")
    pd.testing.assert_frame_equal(
        old_keyed,
        new_keyed.loc[old_keyed.index, columns],
        check_dtype=True,
        check_exact=True,
        check_names=True,
    )


def marker_for_source_line(text_path: Path, source_line: int) -> str:
    lines = text_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if source_line <= 0 or source_line > len(lines):
        raise AssertionError("new row has invalid source_line provenance")
    tokens = lines[source_line - 1].split()
    if len(tokens) < 13:
        raise AssertionError("new row cannot be traced to 13 JPX value fields")
    values = tokens[-13:]
    marker = values[8]
    if SPECIAL_QUOTE_MARKER.fullmatch(marker) is None:
        raise AssertionError(
            "v6 added a row without a Final Special Quote side marker"
        )
    for index, token in enumerate(values):
        if index != 8 and ("ｶ" in token or "ｳ" in token):
            raise AssertionError("v6 added a row with a marker outside field 9")
    return marker[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    old_jpx = load_old_parser(Path(args.old_parser).resolve())
    manifest_path = Path(args.sealed_manifest).resolve()
    paths = sealed_paths(manifest_path)
    per_file: list[dict[str, Any]] = []
    total_old_rows = 0
    total_new_rows = 0
    total_old_rejected = 0
    total_new_rejected = 0
    affected_files = 0
    marker_counts = {"ｶ": 0, "ｳ": 0}

    with tempfile.TemporaryDirectory(prefix="jpx-parser-recovery-audit-") as raw:
        directory = Path(raw)
        for position, pdf_path in enumerate(paths, start=1):
            text_path = directory / f"{pdf_path.stem}.txt"
            current_jpx.convert_jpx_pdf(pdf_path, text_path)
            old_frame, old_report = old_jpx.parse_jpx_text(text_path)
            new_frame, new_report = current_jpx.parse_jpx_text(text_path)
            assert_common_rows_identical(old_frame, new_frame)

            old_keys = keyed(old_frame).index
            new_keyed = keyed(new_frame)
            added_index = new_keyed.index.difference(old_keys)
            for _, row in new_keyed.loc[added_index].iterrows():
                marker_counts[
                    marker_for_source_line(text_path, int(row["source_line"]))
                ] += 1

            added = int(len(added_index))
            rejected_delta = int(old_report.rejected_rows - new_report.rejected_rows)
            if added != rejected_delta:
                raise AssertionError(
                    f"{pdf_path.name}: added rows do not equal recovered rejects"
                )
            if added:
                affected_files += 1
            total_old_rows += int(old_report.parsed_rows)
            total_new_rows += int(new_report.parsed_rows)
            total_old_rejected += int(old_report.rejected_rows)
            total_new_rejected += int(new_report.rejected_rows)
            per_file.append(
                {
                    "name": pdf_path.name,
                    "sha256": sha256_file(pdf_path),
                    "ordinary_rows": int(new_report.ordinary_rows),
                    "old_parsed_rows": int(old_report.parsed_rows),
                    "old_rejected_rows": int(old_report.rejected_rows),
                    "new_parsed_rows": int(new_report.parsed_rows),
                    "new_rejected_rows": int(new_report.rejected_rows),
                    "recovered_rows": added,
                }
            )
            if position % 20 == 0 or position == len(paths):
                print(f"audited {position}/{len(paths)} PDFs", flush=True)

    recovered = total_new_rows - total_old_rows
    if total_old_rejected != EXPECTED_OLD_REJECTED_ROWS:
        raise AssertionError("v5 rejected-row total changed from the incident record")
    if total_new_rejected != 0:
        raise AssertionError("v6 still rejects sealed ordinary-stock rows")
    if recovered != EXPECTED_OLD_REJECTED_ROWS:
        raise AssertionError("v6 recovered an unexpected number of rows")
    if affected_files != EXPECTED_AFFECTED_FILES:
        raise AssertionError("v6 affected an unexpected number of PDFs")
    if sum(marker_counts.values()) != recovered:
        raise AssertionError("not every recovered row has an audited side marker")

    return {
        "schema_version": 1,
        "record_type": "jpx_parser_recovery_audit",
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "sealed_manifest_path": str(manifest_path),
        "sealed_manifest_sha256": sha256_file(manifest_path),
        "pdf_count": len(paths),
        "old_parser_path": str(Path(args.old_parser).resolve()),
        "old_parser_sha256": sha256_file(args.old_parser),
        "old_parser_version": old_jpx.PARSER_VERSION,
        "new_parser_sha256": sha256_file(Path(current_jpx.__file__)),
        "new_parser_version": current_jpx.PARSER_VERSION,
        "old_parsed_rows": total_old_rows,
        "new_parsed_rows": total_new_rows,
        "old_rejected_rows": total_old_rejected,
        "new_rejected_rows": total_new_rejected,
        "recovered_rows": recovered,
        "affected_files": affected_files,
        "marker_counts": marker_counts,
        "old_accepted_rows_canonically_identical": True,
        "return_labels_or_metrics_accessed": False,
        "per_file": per_file,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-manifest", required=True)
    parser.add_argument("--old-parser", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"audit output already exists: {output}")
    payload = run(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"audit={output}", flush=True)


if __name__ == "__main__":
    main()
