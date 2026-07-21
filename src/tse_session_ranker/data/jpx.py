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


PARSER_VERSION = "jpx_daily_text_v6_special_quote_marker"
SESSION_OHLC = (
    "am_open",
    "am_high",
    "am_low",
    "am_close",
    "pm_open",
    "pm_high",
    "pm_low",
    "pm_close",
)
_ORDINARY_ROW = re.compile(
    r"^\s*(\d{8})\s+([0-9A-Z]{5})\s+(.+?)\s+普通株式(?:\s+(.*?))?\s*$"
)
_DAILY_REPORT_DATE = re.compile(
    r"^\s*\f?\s*(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日"
)
_DAILY_REPORT_TITLE = re.compile(r"Auction\s+Trades\s+Regular\s+Way")
_DAILY_REPORT_ORDINARY_END = re.compile(r"Domestic\s+Preferred\s+Stock")
_DAILY_REPORT_ROW = re.compile(r"^\s*([0-9][0-9A-Z]{3})\s+([0-9,]+)\s+")
_DAILY_REPORT_VALUE = re.compile(r"^[+-]?[0-9][0-9,]*(?:\.[0-9]+)?$")
_DAILY_REPORT_SPECIAL_QUOTE_VALUE = re.compile(
    r"^[ｶｳ]?(?P<price>[+-]?[0-9][0-9,]*(?:\.[0-9]+)?)$"
)
_MISSING_VALUE_TOKENS = frozenset({"-", "－", "−", "–", "—", "―"})
_DAILY_REPORT_FIELD_COUNT = 13
_DAILY_REPORT_SOURCE_SCALE = 1_000.0
_DAILY_REPORT_FORMAT = (
    "jpx_stock_quotations_auction_regular_way_domestic_ordinary"
)
_MONTHLY_TEXT_FORMAT = "jpx_date_prefixed_monthly_text"


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
    source_format: str = _MONTHLY_TEXT_FORMAT
    volume_semantics: str | None = None
    turnover_semantics: str | None = None
    trading_unit_semantics: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _price(value: str) -> float:
    if value in _MISSING_VALUE_TOKENS:
        return float("nan")
    try:
        return float(value.replace(",", "").replace("−", "-"))
    except (TypeError, ValueError):
        return float("nan")


def _scaled_daily_report_value(value: str) -> float:
    parsed = _price(value)
    if not np.isfinite(parsed):
        return float("nan")
    return round(parsed * _DAILY_REPORT_SOURCE_SCALE, 6)


def _daily_report_special_quote(value: str) -> float:
    """Parse JPX's final special quote without discarding its numeric price.

    In the official table this field alone may prefix the quote with half-width
    ``ｶ`` or ``ｳ``.  They identify the special-quote side and are not part of
    the price.  No other numeric field accepts these markers.
    """

    if value in _MISSING_VALUE_TOKENS:
        return float("nan")
    match = _DAILY_REPORT_SPECIAL_QUOTE_VALUE.fullmatch(value)
    if match is None:
        return float("nan")
    return _price(match.group("price"))


def _daily_report_date(lines: list[str], source: Path) -> str:
    dates = {
        f"{int(match.group(1)):04d}{int(match.group(2)):02d}{int(match.group(3)):02d}"
        for line in lines
        if (match := _DAILY_REPORT_DATE.match(line)) is not None
    }
    if not dates:
        raise DataValidationError(f"JPX daily report has no heading date: {source}")
    if len(dates) != 1:
        raise DataValidationError(
            f"JPX daily report contains conflicting heading dates: {source}"
        )
    return dates.pop()


def _daily_report_page_parts(line: str) -> list[int]:
    match = _DAILY_REPORT_DATE.match(line)
    if match is None:
        return []
    return [int(value) for value in re.findall(r"[0-9]+", line[match.end() :])]


def _daily_report_base(
    *,
    raw_date: str,
    code: str,
    name: str,
    trading_unit: int,
    source: Path,
    line_number: int,
    values: list[str],
) -> dict[str, object]:
    return {
        "date": raw_date,
        "code": code,
        "name": re.sub(r"\s+", " ", name).strip(),
        "raw_name": name.strip(),
        "trading_unit": trading_unit,
        "final_special_quote": _daily_report_special_quote(values[8]),
        "net_change": _price(values[9]),
        "vwap": _price(values[10]),
        # The report labels these columns as thousands.  Canonical values are
        # actual shares and actual yen, so both source fields are x1000.
        "volume": _scaled_daily_report_value(values[11]),
        "turnover": _scaled_daily_report_value(values[12]),
        "volume_unit": "shares",
        "turnover_unit": "JPY",
        "source_volume_unit": "thousand_shares",
        "source_turnover_unit": "thousand_JPY",
        "source_file": source.name,
        "source_line": line_number,
        "source_format": _DAILY_REPORT_FORMAT,
    }


def _parse_daily_report(
    lines: list[str], source: Path, report: JPXParseReport
) -> tuple[pd.DataFrame, JPXParseReport]:
    raw_date = _daily_report_date(lines, source)
    title_index = next(
        (
            index
            for index, line in enumerate(lines)
            if _DAILY_REPORT_TITLE.search(line) is not None
        ),
        None,
    )
    if title_index is None:
        raise DataValidationError(
            f"JPX daily report has no Auction Trades Regular Way table: {source}"
        )

    report.source_format = _DAILY_REPORT_FORMAT
    report.volume_semantics = (
        "actual shares; source thousand-shares multiplied by 1000"
    )
    report.turnover_semantics = (
        "actual JPY; source thousand-JPY multiplied by 1000"
    )
    report.trading_unit_semantics = "shares in one trading lot"
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        lines[title_index + 1 :], start=title_index + 2
    ):
        if _DAILY_REPORT_ORDINARY_END.search(line) is not None:
            # Section 1 continues with preferred stock, investment
            # certificates, ETFs/ETNs and foreign securities.  The historical
            # date-prefixed source selects only rows marked 普通株式, so stop at
            # the first explicit boundary to keep both universes equivalent.
            break
        page_parts = _daily_report_page_parts(line)
        if page_parts and (len(page_parts) != 2 or page_parts[0] != 1):
            # A JPX file concatenates regular-way, same-day, ToSTNeT and other
            # tables.  Only top-level section 1 is the auction regular-way table.
            break

        match = _DAILY_REPORT_ROW.match(line)
        if match is None:
            continue
        report.ordinary_rows += 1
        code, raw_trading_unit = match.groups()
        tokens = line.split()
        if len(tokens) < 2 + 1 + _DAILY_REPORT_FIELD_COUNT:
            report.rejected_rows += 1
            continue
        values = tokens[-_DAILY_REPORT_FIELD_COUNT:]
        name_tokens = tokens[2:-_DAILY_REPORT_FIELD_COUNT]
        invalid_values = any(
            token not in _MISSING_VALUE_TOKENS
            and (
                _DAILY_REPORT_SPECIAL_QUOTE_VALUE.fullmatch(token)
                if index == 8
                else _DAILY_REPORT_VALUE.fullmatch(token)
            )
            is None
            for index, token in enumerate(values)
        )
        if not name_tokens or invalid_values:
            report.rejected_rows += 1
            continue
        trading_unit = int(raw_trading_unit.replace(",", ""))
        if trading_unit <= 0:
            report.rejected_rows += 1
            continue

        session_prices = [_price(value) for value in values[:8]]
        finite = np.isfinite(session_prices)
        finite_count = int(finite.sum())
        is_am_only = bool(finite[:4].all() and not finite[4:].any())
        is_pm_only = bool(not finite[:4].any() and finite[4:].all())
        base = _daily_report_base(
            raw_date=raw_date,
            code=code,
            name=" ".join(name_tokens),
            trading_unit=trading_unit,
            source=source,
            line_number=line_number,
            values=values,
        )
        if finite_count == 8:
            row = {
                **base,
                "open": session_prices[0],
                "high": max(session_prices[1], session_prices[5]),
                "low": min(session_prices[2], session_prices[6]),
                "close": session_prices[7],
                **dict(zip(SESSION_OHLC, session_prices, strict=True)),
                "traded": True,
                "partial_session": False,
            }
            report.full_session_rows += 1
        elif finite_count == 4 and (is_am_only or is_pm_only):
            traded_prices = [value for value in session_prices if np.isfinite(value)]
            row = {
                **base,
                "open": traded_prices[0],
                "high": traded_prices[1],
                "low": traded_prices[2],
                "close": traded_prices[3],
                # Do not fabricate the missing half-day.  Downstream features
                # receive session OHLC only when all eight values are present.
                **dict.fromkeys(SESSION_OHLC, np.nan),
                "traded": True,
                "partial_session": True,
            }
            report.partial_session_rows += 1
        elif finite_count == 0:
            volume = base["volume"]
            turnover = base["turnover"]
            if (np.isfinite(volume) and float(volume) > 0) or (
                np.isfinite(turnover) and float(turnover) > 0
            ):
                report.rejected_rows += 1
                continue
            row = {
                **base,
                "open": np.nan,
                "high": np.nan,
                "low": np.nan,
                "close": np.nan,
                **dict.fromkeys(SESSION_OHLC, np.nan),
                "traded": False,
                "partial_session": False,
            }
            report.no_trade_rows += 1
        else:
            # An arbitrary subset of AM/PM fields signals layout drift or a
            # damaged row.  Reject it instead of silently shifting columns.
            report.rejected_rows += 1
            continue
        rows.append(row)
        report.parsed_rows += 1

    if not rows:
        raise DataValidationError(
            f"no auction regular-way quotation rows found in {source}"
        )
    return normalize_daily_prices(pd.DataFrame(rows)), report


def _parse_date_prefixed_monthly_text(
    lines: list[str], source: Path, report: JPXParseReport
) -> tuple[pd.DataFrame, JPXParseReport]:
    rows: list[dict[str, object]] = []
    report.source_format = _MONTHLY_TEXT_FORMAT
    for line_number, line in enumerate(lines, start=1):
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
            "source_format": _MONTHLY_TEXT_FORMAT,
        }
        if len(numeric) >= 8 and len(valid) == 8:
            row = {
                **base,
                "open": numeric[0],
                "high": max(numeric[1], numeric[5]),
                "low": min(numeric[2], numeric[6]),
                "close": numeric[7],
                **dict(zip(SESSION_OHLC, numeric, strict=True)),
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
                **dict.fromkeys(SESSION_OHLC, np.nan),
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
                **dict.fromkeys(SESSION_OHLC, np.nan),
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


def parse_jpx_text(path: str | Path) -> tuple[pd.DataFrame, JPXParseReport]:
    """Parse either supported JPX ``pdftotext -layout`` quotation format.

    The date-prefixed monthly text format is preserved for historical inputs.
    The PDF-derived daily universe stops before ``Domestic Preferred Stock`` so
    it matches monthly rows marked ``普通株式``.  Prices are yen, ``volume`` is
    normalised from thousands to actual shares, ``turnover`` is normalised from
    thousands to actual yen, and ``trading_unit`` retains the reported lot size.
    """

    source = Path(path)
    report = JPXParseReport(path=str(source), sha256=_sha256(source))
    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    if any(_DAILY_REPORT_TITLE.search(line) is not None for line in lines):
        return _parse_daily_report(lines, source, report)
    return _parse_date_prefixed_monthly_text(lines, source, report)


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
        "canonical_units": {
            "price": "JPY_per_share",
            "volume": "shares",
            "turnover": "JPY",
            "trading_unit": "shares_per_lot",
        },
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
            url, headers={"User-Agent": "tse-session-ranker/0.3 (+local research)"}
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
