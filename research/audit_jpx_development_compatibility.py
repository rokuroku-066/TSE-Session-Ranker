#!/usr/bin/env python3
"""Prove that JPX parser v6 leaves the development export unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from tse_session_ranker.data.common import SESSION_OHLC
from tse_session_ranker.data.jpx import PARSER_VERSION, collect_jpx
from tse_session_ranker.io import read_frame


EXPECTED_OLD_PARSER = "jpx_daily_text_v5_domestic_ordinary"
EXPECTED_NEW_PARSER = "jpx_daily_text_v6_special_quote_marker"
FEATURE_CONTENT_COLUMNS = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    *SESSION_OHLC,
    "volume",
    "turnover",
    "traded",
    "partial_session",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    values = pd.util.hash_pandas_object(frame[list(columns)], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = Path(args.development_manifest).resolve()
    daily_path = Path(args.development_daily).resolve()
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if old_manifest.get("parser_version") != EXPECTED_OLD_PARSER:
        raise ValueError("development manifest is not the frozen v5 export")
    if old_manifest.get("export_sha256") != sha256_file(daily_path):
        raise ValueError("development manifest does not bind the daily export")
    inputs = old_manifest.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 61:
        raise ValueError("development manifest must contain 61 monthly inputs")
    paths: list[Path] = []
    for item in inputs:
        source = Path(item["path"])
        if not source.is_file() or sha256_file(source) != item.get("sha256"):
            raise ValueError("development JPX source hash changed")
        if int(item.get("rejected_rows", -1)) != 0:
            raise ValueError("frozen development parser had rejected rows")
        paths.append(source)

    print("reparse 61 frozen development inputs with parser v6", flush=True)
    new_frame, new_manifest = collect_jpx(paths)
    if PARSER_VERSION != EXPECTED_NEW_PARSER:
        raise ValueError("unexpected recovery parser version")
    if any(int(item["rejected_rows"]) != 0 for item in new_manifest["inputs"]):
        raise AssertionError("parser v6 rejected a development row")

    old_frame = read_frame(daily_path)
    if list(old_frame.columns) != list(new_frame.columns):
        raise AssertionError("parser v6 changed the development frame schema")
    old_full_hash = frame_hash(old_frame, old_frame.columns)
    new_full_hash = frame_hash(new_frame, new_frame.columns)
    old_feature_hash = frame_hash(old_frame, FEATURE_CONTENT_COLUMNS)
    new_feature_hash = frame_hash(new_frame, FEATURE_CONTENT_COLUMNS)
    checks = {
        "rows_equal": len(old_frame) == len(new_frame),
        "codes_equal": old_frame["code"].nunique() == new_frame["code"].nunique(),
        "min_date_equal": old_frame["date"].min() == new_frame["date"].min(),
        "max_date_equal": old_frame["date"].max() == new_frame["date"].max(),
        "schema_equal": list(old_frame.columns) == list(new_frame.columns),
        "all_canonical_columns_hash_equal": old_full_hash == new_full_hash,
        "model_feature_content_hash_equal": old_feature_hash == new_feature_hash,
        "no_v6_rejections": all(
            int(item["rejected_rows"]) == 0 for item in new_manifest["inputs"]
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"development compatibility failed: {checks}")

    return {
        "schema_version": 1,
        "record_type": "jpx_parser_development_compatibility_audit",
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "old_manifest_path": str(manifest_path),
        "old_manifest_sha256": sha256_file(manifest_path),
        "old_daily_path": str(daily_path),
        "old_daily_sha256": sha256_file(daily_path),
        "old_parser_version": old_manifest["parser_version"],
        "new_parser_version": PARSER_VERSION,
        "source_input_count": len(paths),
        "source_hashes": {str(path): sha256_file(path) for path in paths},
        "rows": int(len(new_frame)),
        "codes": int(new_frame["code"].nunique()),
        "min_date": str(new_frame["date"].min().date()),
        "max_date": str(new_frame["date"].max().date()),
        "old_all_canonical_columns_sha256": old_full_hash,
        "new_all_canonical_columns_sha256": new_full_hash,
        "old_model_feature_content_sha256": old_feature_hash,
        "new_model_feature_content_sha256": new_feature_hash,
        "new_rejected_rows": int(
            sum(int(item["rejected_rows"]) for item in new_manifest["inputs"])
        ),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--development-daily", required=True)
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
