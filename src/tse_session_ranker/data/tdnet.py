from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import time as clock_time
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..exceptions import DataValidationError
from .common import normalize_code, normalize_expected_sessions


TDNET_INDEX_URL = "https://contents.webapi.yanoshin.jp/contents/tdnet_date/{date}"
TDNET_USER_AGENT = "tse-session-ranker/0.3"
TDNET_CACHE_METADATA_SCHEMA = 1
TDNET_NETWORK_PROVENANCE = "network_request_start_recorded"
TDNET_LEGACY_PROVENANCE = "legacy_historical_file_mtime_assumption"
TDNET_UNTAGGED_PROVENANCE = "schema_v1_untagged_request_start"
TDNET_HISTORICAL_PROVENANCE = frozenset(
    {TDNET_LEGACY_PROVENANCE, TDNET_UNTAGGED_PROVENANCE}
)

TDNET_CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "earnings": (r"決算短信",),
    "revision": (r"業績予想.*修正", r"通期.*予想.*修正"),
    "revision_up": (r"上方修正", r"上方に修正", r"増額修正"),
    "revision_down": (r"下方修正", r"下方に修正", r"減額修正"),
    "dividend": (r"配当",),
    "dividend_up": (r"増配", r"復配"),
    "dividend_down": (r"減配", r"無配"),
    "buyback_status": (r"自己株式.*(?:取得状況|取得結果|取得終了)",),
    "buyback_tostnet": (r"ToSTNeT", r"ＴｏＳＴＮｅＴ", r"立会外"),
    "buyback_decision": (
        r"自己株式.*(?:取得に係る事項|取得の決定|取得枠|取得を行う)",
        r"自己株式取得.*(?:決定|実施)",
    ),
    "equity_financing": (
        r"第三者割当",
        r"公募増資",
        r"新株式.*発行",
        r"新株予約権",
        r"株式の売出し",
        r"転換社債",
    ),
    "equity_financing_status": (
        r"払込完了",
        r"発行結果",
        r"行使状況",
        r"大量行使",
        r"月間行使",
        r"行使完了",
        r"発行中止",
        r"失権",
        r"新株予約権.*消却",
    ),
    "benefit": (r"株主優待",),
    "split": (r"株式分割",),
    "ma_alliance": (
        r"子会社化",
        r"株式取得",
        r"事業譲受",
        r"事業譲渡",
        r"業務提携",
        r"資本業務提携",
        r"会社分割",
        r"合併",
    ),
    "control_transaction": (r"公開買付", r"ＭＢＯ", r"MBO", r"株式交換"),
    "impairment_loss": (r"減損", r"特別損失", r"債権放棄"),
    "audit_problem": (
        r"不適切会計",
        r"継続企業の前提",
        r"内部統制.*不備",
        r"調査報告書",
        r"監査法人.*(?:異動|辞任|退任|意見不表明|限定付)",
        r"(?:異動|辞任|退任).*監査法人",
    ),
    "medium_term_plan": (r"中期経営計画",),
    "monthly": (r"月次",),
    "correction": (r"訂正",),
}

TDNET_FEATURE_COLUMNS: tuple[str, ...] = (
    "tdnet_any",
    "tdnet_count_log1p",
    "tdnet_premarket_count_log1p",
    "tdnet_intraday_count_log1p",
    "tdnet_postclose_count_log1p",
    "tdnet_latest_age_hours_log1p",
    *(f"tdnet_has_{name}" for name in TDNET_CATEGORY_PATTERNS),
    "tdnet_positive",
    "tdnet_negative",
    "tdnet_mixed",
)

TDNET_MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "tdnet_has_revision_up",
    "tdnet_has_revision_down",
    "tdnet_has_dividend_up",
    "tdnet_has_dividend_down",
    "tdnet_has_buyback_decision",
    "tdnet_has_equity_financing",
)


@dataclass
class TDnetDataset:
    disclosures: pd.DataFrame
    complete_dates: pd.DatetimeIndex
    observed_at_by_date: pd.Series
    provenance_by_date: pd.Series
    source_sha256: str
    source_files: int

    def __post_init__(self) -> None:
        self.disclosures = normalize_tdnet_disclosures(self.disclosures)
        complete = pd.DatetimeIndex(
            pd.to_datetime(list(self.complete_dates), errors="coerce")
        )
        if complete.isna().any() or complete.empty:
            raise DataValidationError("TDnet complete_dates are empty or invalid")
        complete = pd.DatetimeIndex(sorted(complete.normalize().unique()))
        observations: dict[pd.Timestamp, pd.Timestamp] = {}
        for key, value in pd.Series(self.observed_at_by_date).items():
            try:
                timestamp = pd.Timestamp(value)
            except (TypeError, ValueError) as exc:
                raise DataValidationError(
                    "TDnet observed_at_by_date contains an invalid timestamp"
                ) from exc
            if pd.isna(timestamp) or timestamp.tzinfo is None:
                raise DataValidationError(
                    "TDnet observation timestamps must be timezone-aware"
                )
            observations[pd.Timestamp(key).normalize()] = timestamp.tz_convert(
                "Asia/Tokyo"
            )
        if set(observations) != set(complete):
            raise DataValidationError(
                "TDnet observation dates must exactly match complete_dates"
            )
        if self.source_files != len(complete):
            raise DataValidationError(
                "TDnet source_files must equal the number of complete_dates"
            )
        if not str(self.source_sha256).strip():
            raise DataValidationError("TDnet source_sha256 must not be empty")
        provenances: dict[pd.Timestamp, str] = {}
        for key, value in pd.Series(self.provenance_by_date).items():
            provenance = str(value).strip()
            if provenance not in {
                TDNET_NETWORK_PROVENANCE,
                *TDNET_HISTORICAL_PROVENANCE,
            }:
                raise DataValidationError(
                    "TDnet provenance contains an unsupported value"
                )
            provenances[pd.Timestamp(key).normalize()] = provenance
        if set(provenances) != set(complete):
            raise DataValidationError(
                "TDnet provenance dates must exactly match complete_dates"
            )
        self.complete_dates = complete
        self.observed_at_by_date = pd.Series(
            observations, name="observed_at"
        ).sort_index()
        self.provenance_by_date = pd.Series(
            provenances, name="provenance", dtype="object"
        ).sort_index()


def require_production_tdnet_provenance(dataset: TDnetDataset) -> None:
    invalid = dataset.provenance_by_date.ne(TDNET_NETWORK_PROVENANCE)
    if invalid.any():
        examples = ", ".join(
            str(pd.Timestamp(value).date())
            for value in dataset.provenance_by_date.index[invalid][:5]
        )
        raise DataValidationError(
            "production TDnet data requires recorded network request starts; "
            f"historical-only provenance was found for {examples}"
        )


def tdnet_dataset_digest(dataset: TDnetDataset) -> str:
    """Canonical digest binding disclosures and every completeness assertion."""

    digest = hashlib.sha256()

    def update(value: object) -> None:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)

    update("tdnet-dataset-v1")
    update(dataset.source_sha256)
    update(dataset.source_files)
    for row in dataset.disclosures.itertuples(index=False):
        update(pd.Timestamp(row.published_at).isoformat())
        update(row.code)
        update(row.name)
        update(row.title)
        update(row.url)
    for date in dataset.complete_dates:
        update(pd.Timestamp(date).strftime("%Y-%m-%d"))
        update(pd.Timestamp(dataset.observed_at_by_date.loc[date]).isoformat())
        update(dataset.provenance_by_date.loc[date])
    return digest.hexdigest()


class _TDnetIndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[set[str]] = []
        self.row_depth: int | None = None
        self.cell: str | None = None
        self.cell_depth: int | None = None
        self.anchor_href: str | None = None
        self.anchor_text: list[str] = []
        self.current: dict[str, str] = {}
        self.rows: list[dict[str, str]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = {key: value or "" for key, value in attrs}
        if tag == "div":
            classes = set(values.get("class", "").split())
            self.stack.append(classes)
            if self.row_depth is None and {"row", "card"}.issubset(classes):
                self.row_depth = len(self.stack)
                self.current = {}
            elif self.row_depth is not None:
                if "ad-box" in classes:
                    # The public mirror inserts an advertising card into the
                    # same ``row card`` stream as disclosures.  Mark that
                    # exact structure so it can be skipped without silently
                    # accepting a malformed disclosure card.
                    self.current["_card_kind"] = "advertisement"
                if {"one", "column"}.issubset(classes):
                    self.cell = "time"
                    self.cell_depth = len(self.stack)
                elif {"three", "columns"}.issubset(classes):
                    self.cell = "company"
                    self.cell_depth = len(self.stack)
                elif {"eight", "columns"}.issubset(classes):
                    self.cell = "title"
                    self.cell_depth = len(self.stack)
        elif tag == "a" and self.row_depth is not None:
            self.anchor_href = values.get("href", "")
            self.anchor_text = []

    def handle_data(self, data: str) -> None:
        if self.row_depth is None or self.cell is None:
            return
        text = " ".join(data.split())
        if text:
            self.current[self.cell] = (
                self.current.get(self.cell, "") + " " + text
            ).strip()
            if self.anchor_href is not None:
                self.anchor_text.append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.anchor_href is not None:
            text = " ".join(self.anchor_text).strip()
            href = self.anchor_href
            if self.cell == "company" and "/contents/tdnet/" in href:
                self.current["company_href"] = href
                self.current["company"] = text or self.current.get("company", "")
            elif self.cell == "title":
                self.current["url"] = href
                self.current["title"] = text or self.current.get("title", "")
            self.anchor_href = None
            self.anchor_text = []
        if tag != "div" or not self.stack:
            return
        depth = len(self.stack)
        if self.cell_depth == depth:
            self.cell = None
            self.cell_depth = None
        if self.row_depth == depth:
            self.rows.append(dict(self.current))
            self.row_depth = None
            self.current = {}
        self.stack.pop()


def _release_url(value: str) -> str:
    marker = "rd.php?"
    return value.split(marker, 1)[1] if marker in value else value


def parse_tdnet_index(html: str | bytes, index_date: object) -> pd.DataFrame:
    """Parse one cached TDnet date index into timestamped disclosures."""

    date = pd.Timestamp(index_date).normalize()
    text = html.decode("utf-8", errors="replace") if isinstance(html, bytes) else html
    parser = _TDnetIndexParser()
    parser.feed(text)
    expected_heading = re.compile(
        rf"<h1>\s*{date.year}年0?{date.month}月0?{date.day}日.*?"
        r"提出された適時開示情報\s*</h1>",
        flags=re.DOTALL,
    )
    if not expected_heading.search(text):
        raise DataValidationError(
            f"TDnet index heading does not match {date.date()}"
        )
    rows: list[dict[str, object]] = []
    for item in parser.rows:
        if item.get("_card_kind") == "advertisement":
            continue
        # A withdrawn document can remain as an empty shell in the public
        # index.  It has no usable title or document URL, so it cannot be
        # classified and is deliberately not represented as a disclosure.
        if "time" not in item or "company_href" not in item:
            raise DataValidationError(
                f"TDnet index contains a malformed disclosure card on {date.date()}"
            )
        if not item.get("title", "").strip():
            continue
        if not item.get("url", "").strip():
            raise DataValidationError(
                f"TDnet disclosure card has no document URL on {date.date()}"
            )
        time_match = re.fullmatch(r"(\d{1,2}):(\d{2})", item["time"].strip())
        code_match = re.search(r"/contents/tdnet/([0-9A-Z]{4,5})", item["company_href"])
        if time_match is None or code_match is None:
            raise DataValidationError(
                f"TDnet index contains an invalid time or code on {date.date()}"
            )
        hour, minute = (int(value) for value in time_match.groups())
        if hour > 23 or minute > 59:
            raise DataValidationError(
                f"TDnet index contains an out-of-range time on {date.date()}"
            )
        published = pd.Timestamp(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=hour,
            minute=minute,
            tz="Asia/Tokyo",
        )
        rows.append(
            {
                "published_at": published,
                "code": normalize_code(code_match.group(1)),
                "name": re.sub(r"\s*（[0-9A-Z]{4,5}）\s*$", "", item["company"]).strip(),
                "title": item["title"].strip(),
                "url": _release_url(item.get("url", "")),
                "index_date": date,
            }
        )
    linked_documents = {
        value.lower()
        for value in re.findall(
            r"https?://www\.release\.tdnet\.info/[^\"'<>\s&]+\.pdf",
            text,
            flags=re.IGNORECASE,
        )
    }
    parsed_documents = {str(row["url"]).lower() for row in rows}
    if linked_documents - parsed_documents:
        raise DataValidationError(
            f"TDnet index contains unparsed disclosure links on {date.date()}"
        )
    columns = ["published_at", "code", "name", "title", "url", "index_date"]
    return normalize_tdnet_disclosures(pd.DataFrame(rows, columns=columns))


def normalize_tdnet_disclosures(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"published_at", "code", "title"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"TDnet data is missing columns: {missing}")
    result = frame.copy()
    timezone_missing = result["published_at"].map(
        lambda value: pd.isna(value) or pd.Timestamp(value).tzinfo is None
    )
    if timezone_missing.any():
        raise DataValidationError(
            "TDnet published_at values must include an explicit timezone"
        )
    parsed = pd.to_datetime(result["published_at"], errors="coerce", utc=True)
    if parsed.isna().any():
        raise DataValidationError("TDnet data contains invalid published_at values")
    result["published_at"] = parsed.dt.tz_convert("Asia/Tokyo")
    result["index_date"] = result["published_at"].dt.tz_localize(None).dt.normalize()
    result["code"] = result["code"].map(normalize_code)
    if result["code"].eq("").any():
        raise DataValidationError("TDnet data contains an empty security code")
    result["title"] = result["title"].fillna("").astype(str).str.strip()
    if result["title"].eq("").any():
        raise DataValidationError("TDnet data contains an empty title")
    if "name" not in result:
        result["name"] = result["code"]
    else:
        result["name"] = result["name"].fillna(result["code"]).astype(str)
    if "url" not in result:
        result["url"] = ""
    else:
        result["url"] = result["url"].fillna("").astype(str)
    return (
        result.drop_duplicates(["published_at", "code", "title", "url"])
        .sort_values(["published_at", "code", "title"], kind="stable")
        .reset_index(drop=True)
    )


def _tdnet_paths(inputs: Iterable[str | Path] | str | Path) -> list[Path]:
    values = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        paths.extend(sorted(path.glob("*.html")) if path.is_dir() else [path])
    return sorted(set(paths))


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _cache_metadata(
    index_date: pd.Timestamp,
    payload: bytes,
    observed_at: pd.Timestamp,
    fetched_at: pd.Timestamp,
) -> dict[str, object]:
    return {
        "schema_version": TDNET_CACHE_METADATA_SCHEMA,
        "provenance": TDNET_NETWORK_PROVENANCE,
        "index_date": str(index_date.date()),
        "observed_at": observed_at.isoformat(),
        "fetched_at": fetched_at.isoformat(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "finalized": _page_is_finalized(index_date, observed_at),
    }


def _write_cache_metadata(path: Path, metadata: dict[str, object]) -> None:
    target = _metadata_path(path)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _read_cache_metadata(
    path: Path,
    payload: bytes,
    index_date: pd.Timestamp,
    *,
    allow_historical_provenance: bool = False,
) -> tuple[pd.Timestamp, dict[str, object]]:
    metadata_path = _metadata_path(path)
    if not metadata_path.exists():
        raise DataValidationError(f"TDnet cache metadata is missing: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(
            f"TDnet cache metadata is unreadable: {metadata_path}"
        ) from exc
    if not isinstance(metadata, dict):
        raise DataValidationError(
            f"TDnet cache metadata must be an object: {metadata_path}"
        )
    if metadata.get("schema_version") != TDNET_CACHE_METADATA_SCHEMA:
        raise DataValidationError(f"unsupported TDnet cache metadata: {metadata_path}")
    raw_provenance = metadata.get("provenance")
    provenance = (
        TDNET_UNTAGGED_PROVENANCE
        if raw_provenance is None
        else str(raw_provenance).strip()
    )
    allowed = {TDNET_NETWORK_PROVENANCE}
    if allow_historical_provenance:
        allowed.update(TDNET_HISTORICAL_PROVENANCE)
    if provenance not in allowed:
        raise DataValidationError(
            f"TDnet cache provenance is not permitted: {metadata_path}"
        )
    try:
        observed_at = pd.Timestamp(metadata.get("observed_at"))
        fetched_at = pd.Timestamp(metadata.get("fetched_at"))
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"TDnet cache timestamps are invalid: {metadata_path}"
        ) from exc
    if (
        pd.isna(observed_at)
        or observed_at.tzinfo is None
        or pd.isna(fetched_at)
        or fetched_at.tzinfo is None
    ):
        raise DataValidationError(
            f"TDnet cache timestamps are invalid: {metadata_path}"
        )
    observed_at = observed_at.tz_convert("Asia/Tokyo")
    fetched_at = fetched_at.tz_convert("Asia/Tokyo")
    if fetched_at < observed_at:
        raise DataValidationError(
            f"TDnet cache fetched_at precedes observed_at: {metadata_path}"
        )
    expected = _cache_metadata(index_date, payload, observed_at, fetched_at)
    for field in (
        "index_date",
        "observed_at",
        "fetched_at",
        "bytes",
        "sha256",
        "finalized",
    ):
        if metadata.get(field) != expected[field]:
            raise DataValidationError(
                f"TDnet cache metadata field {field!r} is invalid: {metadata_path}"
            )
    normalized_metadata = dict(metadata)
    normalized_metadata["provenance"] = provenance
    return observed_at, normalized_metadata


def _page_is_finalized(index_date: pd.Timestamp, observed_at: pd.Timestamp) -> bool:
    observed_date = observed_at.tz_localize(None).normalize()
    return bool(observed_date > index_date.normalize())


def collect_tdnet_dataset(
    inputs: Iterable[str | Path] | str | Path,
    *,
    allow_historical_provenance: bool = False,
) -> TDnetDataset:
    paths = _tdnet_paths(inputs)
    if not paths:
        raise DataValidationError("no TDnet index files were found")
    frames: list[pd.DataFrame] = []
    source_dates: list[pd.Timestamp] = []
    observed_at_by_date: dict[pd.Timestamp, pd.Timestamp] = {}
    provenance_by_date: dict[pd.Timestamp, str] = {}
    seen_dates: dict[pd.Timestamp, str] = {}
    for path in paths:
        match = re.search(r"(20\d{6})", path.stem)
        if match is None:
            raise DataValidationError(f"TDnet cache filename has no YYYYMMDD date: {path}")
        date = pd.to_datetime(match.group(1), format="%Y%m%d")
        payload = path.read_bytes()
        payload_hash = hashlib.sha256(payload).hexdigest()
        previous_hash = seen_dates.get(date)
        if previous_hash is not None:
            if previous_hash != payload_hash:
                raise DataValidationError(
                    f"conflicting TDnet indexes for {date.date()}"
                )
            observed_at, metadata = _read_cache_metadata(
                path,
                payload,
                date,
                allow_historical_provenance=allow_historical_provenance,
            )
            previous_observed = observed_at_by_date[date]
            previous_provenance = provenance_by_date[date]
            provenance = str(metadata["provenance"])
            if observed_at > previous_observed or (
                observed_at == previous_observed
                and provenance == TDNET_NETWORK_PROVENANCE
                and previous_provenance != TDNET_NETWORK_PROVENANCE
            ):
                observed_at_by_date[date] = observed_at
                provenance_by_date[date] = provenance
            continue
        seen_dates[date] = payload_hash
        source_dates.append(date)
        observed_at, metadata = _read_cache_metadata(
            path,
            payload,
            date,
            allow_historical_provenance=allow_historical_provenance,
        )
        observed_at_by_date[date] = observed_at
        provenance_by_date[date] = str(metadata["provenance"])
        frames.append(parse_tdnet_index(payload, date))
    disclosures = normalize_tdnet_disclosures(pd.concat(frames, ignore_index=True))
    source_digest = hashlib.sha256()
    for date in sorted(seen_dates):
        source_digest.update(date.strftime("%Y-%m-%d").encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(bytes.fromhex(seen_dates[date]))
        source_digest.update(b"\0")
        source_digest.update(observed_at_by_date[date].isoformat().encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(provenance_by_date[date].encode("utf-8"))
    return TDnetDataset(
        disclosures=disclosures,
        complete_dates=pd.DatetimeIndex(sorted(source_dates)),
        observed_at_by_date=pd.Series(
            observed_at_by_date, name="observed_at"
        ).sort_index(),
        source_sha256=source_digest.hexdigest(),
        source_files=len(source_dates),
        provenance_by_date=pd.Series(
            provenance_by_date, name="provenance"
        ).sort_index(),
    )


def collect_tdnet_indexes(inputs: Iterable[str | Path] | str | Path) -> pd.DataFrame:
    return collect_tdnet_dataset(inputs).disclosures


def _validate_download(payload: bytes, token: str) -> None:
    lowered = payload[:10_000].lower()
    if len(payload) < 200 or b"<html" not in lowered or token.encode() not in payload:
        raise OSError("downloaded TDnet index did not match the requested date")


def download_tdnet_indexes(
    start_date: object,
    end_date: object,
    destination: str | Path,
    *,
    overwrite: bool = False,
    max_workers: int = 4,
    timeout: float = 30.0,
    retries: int = 3,
) -> list[dict[str, object]]:
    """Download an inclusive calendar-date range of public TDnet index pages."""

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("TDnet end_date precedes start_date")
    if max_workers < 1 or retries < 1 or timeout <= 0:
        raise ValueError("TDnet download limits must be positive")
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    dates = list(pd.date_range(start, end, freq="D"))

    def fetch(date: pd.Timestamp) -> dict[str, object]:
        token = date.strftime("%Y%m%d")
        path = target / f"{token}.html"
        url = TDNET_INDEX_URL.format(date=token)
        cache_was_present = path.exists()
        payload = path.read_bytes() if cache_was_present else b""
        cache_is_finalized = False
        observed_at: pd.Timestamp | None = None
        if cache_was_present and not overwrite:
            try:
                _validate_download(payload, token)
                observed_at, _ = _read_cache_metadata(path, payload, date)
                cache_is_finalized = _page_is_finalized(date, observed_at)
            except (DataValidationError, OSError):
                cache_is_finalized = False
        if cache_was_present and not overwrite and cache_is_finalized:
            status = "cached"
        else:
            error: Exception | None = None
            for attempt in range(retries):
                try:
                    observed_at = pd.Timestamp.now(tz="Asia/Tokyo")
                    request = urllib.request.Request(
                        url, headers={"User-Agent": TDNET_USER_AGENT}
                    )
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        payload = response.read()
                    fetched_at = pd.Timestamp.now(tz="Asia/Tokyo")
                    _validate_download(payload, token)
                    temporary = path.with_suffix(".html.part")
                    temporary.write_bytes(payload)
                    temporary.replace(path)
                    _write_cache_metadata(
                        path,
                        _cache_metadata(
                            date, payload, observed_at, fetched_at
                        ),
                    )
                    status = "refreshed" if cache_was_present else "downloaded"
                    break
                except Exception as exc:  # pragma: no cover - network dependent
                    error = exc
                    if attempt + 1 < retries:
                        time.sleep(0.5 * (attempt + 1))
            else:  # pragma: no cover - network dependent
                raise OSError(f"failed to download {url}: {error}") from error
        if observed_at is None:
            raise OSError(f"TDnet cache observation time is unavailable: {path}")
        return {
            "date": str(date.date()),
            "url": url,
            "path": str(path),
            "status": status,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "observed_at": observed_at.isoformat(),
            "finalized": _page_is_finalized(date, observed_at),
        }

    reports: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch, date): date for date in dates}
        for future in as_completed(futures):
            reports.append(future.result())
    return sorted(reports, key=lambda item: str(item["date"]))


def _title_flags(title: pd.Series) -> pd.DataFrame:
    flags = pd.DataFrame(index=title.index)
    for name, patterns in TDNET_CATEGORY_PATTERNS.items():
        expression = "|".join(f"(?:{pattern})" for pattern in patterns)
        flags[name] = title.str.contains(expression, regex=True, na=False).astype(float)
    # Status/end reports are not new buyback decisions even if a broad title
    # happens to contain one of the decision phrases.
    flags["buyback_decision"] *= 1.0 - flags["buyback_status"]
    flags["buyback_decision"] *= 1.0 - flags["buyback_tostnet"]
    flags["equity_financing"] *= 1.0 - flags["equity_financing_status"]
    # Direction words are meaningful only under their parent disclosure type.
    # For example, "配当予想の上方修正" is not an earnings revision.
    flags["revision_up"] *= flags["revision"]
    flags["revision_down"] *= flags["revision"]
    flags["dividend_up"] *= flags["dividend"]
    flags["dividend_down"] *= flags["dividend"]
    return flags


def build_tdnet_features(
    disclosures: pd.DataFrame,
    expected_sessions: Iterable[object],
    *,
    decision_time: str = "08:58:59",
) -> pd.DataFrame:
    """Aggregate disclosures known at each session's pre-open cutoff by code."""

    events = normalize_tdnet_disclosures(disclosures)
    sessions = normalize_expected_sessions(expected_sessions)
    try:
        cutoff_time = clock_time.fromisoformat(decision_time)
    except ValueError as exc:
        raise ValueError("decision_time must be an ISO local time") from exc
    cutoffs = pd.DatetimeIndex(
        [
            pd.Timestamp.combine(date.date(), cutoff_time).tz_localize("Asia/Tokyo")
            for date in sessions
        ]
    )
    published = pd.DatetimeIndex(events["published_at"])
    target_positions = cutoffs.searchsorted(published, side="left")
    in_range = target_positions < len(cutoffs)
    events = events.loc[in_range].copy()
    target_positions = target_positions[in_range]
    events["date"] = sessions[target_positions].to_numpy()
    local_time = events["published_at"].dt.time
    event_is_session = events["index_date"].isin(sessions)
    same_target_date = events["index_date"].eq(events["date"])
    market_close_minutes = np.where(
        events["index_date"].lt(pd.Timestamp("2024-11-05")), 15 * 60, 15 * 60 + 30
    )
    event_minutes = local_time.map(lambda value: value.hour * 60 + value.minute).to_numpy()
    events["premarket"] = same_target_date
    events["intraday"] = event_is_session & ~same_target_date & (
        event_minutes < market_close_minutes
    )
    events["postclose"] = ~(events["premarket"] | events["intraday"])
    target_cutoff = pd.Series(cutoffs[target_positions], index=events.index)
    events["age_hours"] = (
        target_cutoff - events["published_at"]
    ).dt.total_seconds() / 3600.0
    if (events["age_hours"] < 0).any():
        raise DataValidationError("TDnet event was assigned before its publication")
    flags = _title_flags(events["title"])
    events = pd.concat([events, flags], axis=1)
    keys = ["date", "code"]
    grouped = events.groupby(keys, sort=True)
    output = grouped.size().rename("count").reset_index()
    for name in ("premarket", "intraday", "postclose"):
        counts = grouped[name].sum().rename(name).reset_index()
        output = output.merge(counts, on=keys, how="left", validate="one_to_one")
    latest = grouped["age_hours"].min().rename("latest_age_hours").reset_index()
    output = output.merge(latest, on=keys, how="left", validate="one_to_one")
    latest_source = (
        grouped["published_at"]
        .max()
        .rename("tdnet_feature_source_max_timestamp")
        .reset_index()
    )
    output = output.merge(latest_source, on=keys, how="left", validate="one_to_one")
    for name in TDNET_CATEGORY_PATTERNS:
        values = grouped[name].max().rename(name).reset_index()
        output = output.merge(values, on=keys, how="left", validate="one_to_one")
    result = output[keys].copy()
    result["tdnet_any"] = 1.0
    result["tdnet_count_log1p"] = np.log1p(output["count"])
    result["tdnet_premarket_count_log1p"] = np.log1p(output["premarket"])
    result["tdnet_intraday_count_log1p"] = np.log1p(output["intraday"])
    result["tdnet_postclose_count_log1p"] = np.log1p(output["postclose"])
    result["tdnet_latest_age_hours_log1p"] = np.log1p(output["latest_age_hours"])
    for name in TDNET_CATEGORY_PATTERNS:
        result[f"tdnet_has_{name}"] = output[name].astype(float)
    positive_names = ("revision_up", "dividend_up", "buyback_decision", "benefit", "split")
    negative_names = (
        "revision_down",
        "dividend_down",
        "equity_financing",
        "impairment_loss",
        "audit_problem",
    )
    result["tdnet_positive"] = output[list(positive_names)].max(axis=1)
    result["tdnet_negative"] = output[list(negative_names)].max(axis=1)
    result["tdnet_mixed"] = result["tdnet_positive"] * result["tdnet_negative"]
    result["tdnet_feature_source_max_timestamp"] = output[
        "tdnet_feature_source_max_timestamp"
    ]
    return result[
        [
            "date",
            "code",
            *TDNET_FEATURE_COLUMNS,
            "tdnet_feature_source_max_timestamp",
        ]
    ].sort_values(
        ["date", "code"], kind="stable"
    ).reset_index(drop=True)


def tdnet_target_completeness(
    expected_sessions: Iterable[object],
    complete_dates: Iterable[object],
    observed_at_by_date: pd.Series | dict[object, object] | None = None,
    *,
    decision_time: str = "08:58:59",
) -> pd.Series:
    """Whether every required index was observed late enough for the cutoff.

    A page for a date before the target must have been collected on a later
    calendar date, after that page stopped changing.  The target-date page must
    have been collected at or after the configured pre-open decision cutoff.
    """

    sessions = normalize_expected_sessions(expected_sessions)
    complete = set(pd.to_datetime(list(complete_dates)).normalize())
    try:
        cutoff_time = clock_time.fromisoformat(decision_time)
    except ValueError as exc:
        raise ValueError("decision_time must be an ISO local time") from exc
    observations: dict[pd.Timestamp, pd.Timestamp] = {}
    if observed_at_by_date is not None:
        for key, value in pd.Series(observed_at_by_date).items():
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                continue
            timestamp = (
                timestamp.tz_localize("Asia/Tokyo")
                if timestamp.tzinfo is None
                else timestamp.tz_convert("Asia/Tokyo")
            )
            observations[pd.Timestamp(key).normalize()] = timestamp
    values: dict[pd.Timestamp, bool] = {sessions[0]: False}
    for previous, target in zip(sessions[:-1], sessions[1:], strict=True):
        required = pd.date_range(previous.normalize(), target.normalize(), freq="D")
        target_cutoff = pd.Timestamp.combine(
            target.date(), cutoff_time
        ).tz_localize("Asia/Tokyo")
        complete_for_target = True
        for date in required:
            observed_at = observations.get(date)
            required_observation = (
                target_cutoff
                if date == target
                else (date + pd.Timedelta(days=1)).tz_localize("Asia/Tokyo")
            )
            if (
                date not in complete
                or observed_at is None
                or observed_at < required_observation
            ):
                complete_for_target = False
                break
        values[target] = complete_for_target
    return pd.Series(values, name="tdnet_source_complete", dtype=bool)


def merge_tdnet_features(
    panel: pd.DataFrame,
    dataset: TDnetDataset,
    expected_sessions: Iterable[object],
    *,
    decision_time: str = "08:58:59",
) -> pd.DataFrame:
    """Left-join point-in-time TDnet features and preserve source completeness."""

    sessions = normalize_expected_sessions(expected_sessions)
    features = build_tdnet_features(
        dataset.disclosures, sessions, decision_time=decision_time
    )
    result = panel.merge(
        features,
        on=["date", "code"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    completeness = tdnet_target_completeness(
        sessions,
        dataset.complete_dates,
        dataset.observed_at_by_date,
        decision_time=decision_time,
    )
    result["tdnet_source_complete"] = result["date"].map(completeness).eq(True)
    complete_rows = result["tdnet_source_complete"]
    result.loc[complete_rows, list(TDNET_FEATURE_COLUMNS)] = result.loc[
        complete_rows, list(TDNET_FEATURE_COLUMNS)
    ].fillna(0.0)
    # Missing index pages are unknown, never evidence that no disclosure exists.
    result.loc[~complete_rows, list(TDNET_FEATURE_COLUMNS)] = np.nan
    cutoff_time = clock_time.fromisoformat(decision_time)
    target_cutoff = pd.Series(
        [
            pd.Timestamp.combine(pd.Timestamp(date).date(), cutoff_time).tz_localize(
                "Asia/Tokyo"
            )
            for date in result["date"]
        ],
        index=result.index,
    )
    leaked = result["tdnet_feature_source_max_timestamp"].notna() & (
        result["tdnet_feature_source_max_timestamp"] > target_cutoff
    )
    if leaked.any():
        raise DataValidationError("TDnet features contain a post-cutoff disclosure")
    result["eligible"] &= complete_rows
    result["training_eligible"] &= complete_rows
    return result
