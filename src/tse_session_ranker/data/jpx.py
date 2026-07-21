from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..exceptions import DataValidationError
from .common import merge_daily_prices, normalize_daily_prices


PARSER_VERSION = "jpx_daily_text_v2"
_ORDINARY_ROW = re.compile(
    r"^\s*(\d{8})\s+([0-9A-Z]{5})\s+(.+?)\s+普通株式(?:\s+(.*?))?\s*$"
)


@dataclass
class JPXParseReport:
    path: str
    sha256: str
    ordinary_rows: int = 0
    parsed_rows: int = 0
    full_session_rows: int = 0
    partial_session_rows: int = 0
    no_trade_rows: int = 0
    rejected_rows: int = 0
    parser_version: str = PARSER_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _price(value: str) -> float:
    try:
        return float(value.replace(",", ""))
    except (TypeError, ValueError):
        return float("nan")


def parse_jpx_text(path: str | Path) -> tuple[pd.DataFrame, JPXParseReport]:
    """Parse a JPX daily stock quotation text produced by ``pdftotext -layout``."""

    source = Path(path)
    report = JPXParseReport(path=str(source), sha256=_sha256(source))
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
    ):
        match = _ORDINARY_ROW.match(line)
        if not match:
            continue
        report.ordinary_rows += 1
        raw_date, code5, raw_name, values_text = match.groups()
        tokens = [] if not values_text else values_text.split()
        numeric = [_price(token) for token in tokens[:8]]
        valid = [value for value in numeric if np.isfinite(value)]
        base = {
            "date": raw_date,
            "code": code5[:-1],
            "name": re.sub(r"\s+", " ", raw_name).strip(),
            "raw_name": raw_name.strip(),
            "source_file": source.name,
            "source_line": line_number,
        }
        if len(numeric) >= 8 and len(valid) == 8:
            row = {
                **base,
                "open": numeric[0],
                "high": max(numeric[1], numeric[5]),
                "low": min(numeric[2], numeric[6]),
                "close": numeric[7],
                "traded": True,
                "partial_session": False,
            }
            report.full_session_rows += 1
        elif len(valid) == 4:
            row = {
                **base,
                "open": valid[0],
                "high": valid[1],
                "low": valid[2],
                "close": valid[3],
                "traded": True,
                "partial_session": True,
            }
            report.partial_session_rows += 1
        elif len(valid) == 0:
            row = {
                **base,
                "open": np.nan,
                "high": np.nan,
                "low": np.nan,
                "close": np.nan,
                "traded": False,
                "partial_session": False,
            }
            report.no_trade_rows += 1
        else:
            report.rejected_rows += 1
            continue
        rows.append(row)
        report.parsed_rows += 1
    if not rows:
        raise DataValidationError(f"no ordinary-stock rows found in {source}")
    return normalize_daily_prices(pd.DataFrame(rows)), report


def convert_jpx_pdf(pdf_path: str | Path, text_path: str | Path) -> Path:
    source = Path(pdf_path)
    target = Path(text_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["pdftotext", "-layout", str(source), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise DataValidationError("pdftotext is required to parse JPX PDFs") from exc
    except subprocess.CalledProcessError as exc:
        raise DataValidationError(
            f"pdftotext failed for {source}: {exc.stderr.strip()}"
        ) from exc
    return target


def _expand_inputs(inputs: Iterable[str | Path] | str | Path) -> list[Path]:
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    candidates: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            candidates.extend(sorted(path.glob("*.txt")))
            candidates.extend(sorted(path.glob("*.pdf")))
        elif path.is_file():
            candidates.append(path)
        else:
            raise DataValidationError(f"JPX input does not exist: {path}")
    by_stem: dict[tuple[Path, str], Path] = {}
    for path in candidates:
        if path.suffix.lower() not in {".txt", ".pdf"}:
            continue
        key = (path.parent.resolve(), path.stem)
        current = by_stem.get(key)
        if current is None or path.suffix.lower() == ".txt":
            by_stem[key] = path
    selected = sorted(by_stem.values(), key=lambda path: str(path))
    if not selected:
        raise DataValidationError("no .txt or .pdf JPX inputs were found")
    return selected


def collect_jpx(
    inputs: Iterable[str | Path] | str | Path,
    existing: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Collect one or more JPX quotation files into the canonical daily schema."""

    frames: list[pd.DataFrame] = []
    reports: list[JPXParseReport] = []
    with tempfile.TemporaryDirectory(prefix="tse-ranker-jpx-") as directory:
        for source in _expand_inputs(inputs):
            text_path = source
            if source.suffix.lower() == ".pdf":
                text_path = Path(directory) / f"{source.stem}.txt"
                convert_jpx_pdf(source, text_path)
            frame, report = parse_jpx_text(text_path)
            if source.suffix.lower() == ".pdf":
                report.path = str(source)
                report.sha256 = _sha256(source)
            frames.append(frame)
            reports.append(report)
    if existing is not None and not existing.empty:
        frames.insert(0, existing)
    combined = merge_daily_prices(frames)
    manifest: dict[str, object] = {
        "parser_version": PARSER_VERSION,
        "inputs": [report.to_dict() for report in reports],
        "rows": int(len(combined)),
        "codes": int(combined["code"].nunique()),
        "min_date": str(combined["date"].min().date()),
        "max_date": str(combined["date"].max().date()),
        "no_trade_rows": int((~combined["traded"]).sum()),
        "partial_session_rows": int(combined["partial_session"].sum()),
        "has_volume": bool(combined["volume"].notna().any()),
        "has_turnover": bool(combined["turnover"].notna().any()),
    }
    return combined, manifest


def download_jpx_urls(
    urls: Iterable[str] | str, destination: str | Path, overwrite: bool = False
) -> list[dict[str, object]]:
    """Download explicitly supplied official files; no hidden URL guessing is used."""

    target_dir = Path(destination)
    target_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    if isinstance(urls, str):
        urls = [urls]
    for url in urls:
        name = Path(urllib.parse.urlparse(url).path).name
        if not name:
            raise DataValidationError(f"URL has no filename: {url}")
        target = target_dir / name
        if target.exists() and not overwrite:
            reports.append(
                {"url": url, "path": str(target), "sha256": _sha256(target), "cached": True}
            )
            continue
        request = urllib.request.Request(
            url, headers={"User-Agent": "tse-session-ranker/0.1 (+local research)"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        if name.lower().endswith(".pdf") and not payload.startswith(b"%PDF"):
            raise DataValidationError(f"downloaded content is not a PDF: {url}")
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(target)
        reports.append(
            {"url": url, "path": str(target), "sha256": _sha256(target), "cached": False}
        )
    return reports
