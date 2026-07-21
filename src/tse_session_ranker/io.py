from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import DataValidationError


def read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source, dtype={"code": "string"})
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(source)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(source)
        except ImportError as exc:
            raise DataValidationError(
                "Parquet support requires the optional 'parquet' dependency"
            ) from exc
    raise DataValidationError(f"unsupported table format: {source}")


def write_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    with tempfile.NamedTemporaryFile(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        if suffix == ".csv":
            frame.to_csv(temporary, index=False)
        elif suffix in {".pkl", ".pickle"}:
            frame.to_pickle(temporary)
        elif suffix in {".parquet", ".pq"}:
            try:
                frame.to_parquet(temporary, index=False)
            except ImportError as exc:
                raise DataValidationError(
                    "Parquet support requires the optional 'parquet' dependency"
                ) from exc
        else:
            raise DataValidationError(f"unsupported table format: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def json_dumps(payload: object) -> str:
    return json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        indent=2,
        default=str,
        allow_nan=False,
    )


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json_dumps(payload) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=target.name + ".",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, target)
    return target
