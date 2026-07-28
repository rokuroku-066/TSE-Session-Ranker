"""Official TDnet document acquisition for quantitative material research.

This module is deliberately separate from the title-only production pipeline.
It can acquire current official TDnet index pages and linked PDF documents,
preserve request/receipt provenance, and parse one strict forecast-revision
table shape.  It does not provide historical coverage, silently substitute
titles for document bodies, or authorize a model/order decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time
import hashlib
from html.parser import HTMLParser
import json
from math import isfinite
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Iterable
import unicodedata
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

from ..exceptions import DataValidationError


TOKYO = ZoneInfo("Asia/Tokyo")
OFFICIAL_TDNET_HOSTS = frozenset({"www.release.tdnet.info"})
OFFICIAL_TDNET_BASE_URL = "https://www.release.tdnet.info/inbs/"
OFFICIAL_INDEX_PATH = re.compile(
    r"^/inbs/I_list_(?P<page>[0-9]{3})_(?P<date>20[0-9]{6})\.html$"
)
OFFICIAL_DOCUMENT_PATH = re.compile(
    r"^/inbs/(?P<document_id>[0-9]{18})\.pdf$"
)
OFFICIAL_XBRL_PATH = re.compile(r"^/inbs/[0-9]{18}\.zip$")
FORECAST_REVISION_TITLE = re.compile(r"業績予想.*修正")
MATERIAL_PARSER_VERSION = "tdnet_forecast_revision_table_v1"
DOWNLOAD_PROVENANCE = "network_request_start_recorded"


class TDnetMaterialError(DataValidationError):
    """Raised when official TDnet material violates the strict contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).replace("\u3000", " ")


def _normalise_official_code(value: str) -> str:
    code = _normalise_text(value).strip().upper()
    if len(code) == 5 and code.endswith("0"):
        code = code[:-1]
    if re.fullmatch(r"[0-9A-Z]{4,5}", code) is None:
        raise TDnetMaterialError(f"invalid official TDnet code: {value!r}")
    return code


def _validated_official_url(
    value: str,
    *,
    allow_index: bool,
    allow_document: bool,
) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(str(value))
    if (
        parsed.scheme != "https"
        or parsed.hostname not in OFFICIAL_TDNET_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise TDnetMaterialError("TDnet URL must be an unmodified official HTTPS URL")
    index_match = OFFICIAL_INDEX_PATH.fullmatch(parsed.path)
    document_match = OFFICIAL_DOCUMENT_PATH.fullmatch(parsed.path)
    if (index_match is None or not allow_index) and (
        document_match is None or not allow_document
    ):
        raise TDnetMaterialError("unsupported official TDnet URL path")
    filename = Path(parsed.path).name
    return urllib.parse.urlunparse(parsed), filename


@dataclass(frozen=True)
class OfficialDownloadReceipt:
    """Hash-bound receipt for one official TDnet HTTP response."""

    url: str
    path: str
    requested_at: datetime
    received_at: datetime
    http_status: int
    content_type: str
    last_modified: str | None
    bytes: int
    sha256: str
    provenance: str = DOWNLOAD_PROVENANCE
    downloaded_in_run: bool = True

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["requested_at"] = self.requested_at.isoformat()
        payload["received_at"] = self.received_at.isoformat()
        return payload


def _receipt_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".receipt.json")


def _write_receipt(receipt: OfficialDownloadReceipt) -> None:
    target = _receipt_path(Path(receipt.path))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=target.name + ".",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    ) as handle:
        json.dump(
            receipt.to_dict(),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(target)


def _load_receipt(path: Path, url: str) -> OfficialDownloadReceipt:
    sidecar = _receipt_path(path)
    if not path.is_file():
        raise TDnetMaterialError(f"official TDnet cache is not a file: {path}")
    if not sidecar.is_file():
        raise TDnetMaterialError(f"official TDnet receipt is missing: {sidecar}")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TDnetMaterialError(
            f"official TDnet receipt is unreadable: {sidecar}"
        ) from exc
    required = {
        "url",
        "path",
        "requested_at",
        "received_at",
        "http_status",
        "content_type",
        "last_modified",
        "bytes",
        "sha256",
        "provenance",
        "downloaded_in_run",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise TDnetMaterialError("official TDnet receipt schema is invalid")
    try:
        requested_at = datetime.fromisoformat(str(payload["requested_at"]))
        received_at = datetime.fromisoformat(str(payload["received_at"]))
    except ValueError as exc:
        raise TDnetMaterialError("official TDnet receipt timestamp is invalid") from exc
    if (
        requested_at.tzinfo is None
        or received_at.tzinfo is None
        or requested_at.utcoffset() is None
        or received_at.utcoffset() is None
        or requested_at > received_at
    ):
        raise TDnetMaterialError("official TDnet receipt timestamp order is invalid")
    receipt = OfficialDownloadReceipt(
        url=str(payload["url"]),
        path=str(payload["path"]),
        requested_at=requested_at,
        received_at=received_at,
        http_status=int(payload["http_status"]),
        content_type=str(payload["content_type"]),
        last_modified=(
            None
            if payload["last_modified"] is None
            else str(payload["last_modified"])
        ),
        bytes=int(payload["bytes"]),
        sha256=str(payload["sha256"]),
        provenance=str(payload["provenance"]),
        downloaded_in_run=False,
    )
    if (
        receipt.url != url
        or Path(receipt.path).resolve() != path.resolve()
        or receipt.http_status != 200
        or receipt.provenance != DOWNLOAD_PROVENANCE
        or receipt.bytes != path.stat().st_size
        or receipt.sha256 != _sha256_file(path)
        or re.fullmatch(r"[0-9a-f]{64}", receipt.sha256) is None
    ):
        raise TDnetMaterialError("official TDnet cache does not match its receipt")
    _validate_download_payload(url, receipt.content_type, path.read_bytes())
    return receipt


def _archive_existing_snapshot(path: Path, url: str) -> None:
    """Preserve the current hash-bound raw and receipt before replacement."""

    receipt = _load_receipt(path, url)
    sidecar = _receipt_path(path)
    archive = path.parent / "history" / path.name / receipt.sha256
    archive.mkdir(parents=True, exist_ok=True)
    raw_archive = archive / "payload"
    if raw_archive.exists():
        if (
            not raw_archive.is_file()
            or raw_archive.stat().st_size != receipt.bytes
            or _sha256_file(raw_archive) != receipt.sha256
        ):
            raise TDnetMaterialError(
                "official TDnet history contains a conflicting raw object"
            )
    else:
        shutil.copyfile(path, raw_archive)
    sidecar_sha256 = _sha256_file(sidecar)
    receipt_archive = archive / f"{sidecar_sha256}.receipt.json"
    if receipt_archive.exists():
        if (
            not receipt_archive.is_file()
            or _sha256_file(receipt_archive) != sidecar_sha256
        ):
            raise TDnetMaterialError(
                "official TDnet history contains a conflicting receipt"
            )
    else:
        shutil.copyfile(sidecar, receipt_archive)


def _validate_download_payload(url: str, content_type: str, payload: bytes) -> None:
    path = urllib.parse.urlparse(url).path
    if OFFICIAL_INDEX_PATH.fullmatch(path) is not None:
        prefix = payload[:16_384].lower()
        if (
            len(payload) < 500
            or b"<html" not in prefix
            or "適時開示情報閲覧サービス".encode() not in payload
        ):
            raise TDnetMaterialError("official TDnet index response is invalid")
        if "html" not in content_type.lower():
            raise TDnetMaterialError("official TDnet index content type is invalid")
        return
    if OFFICIAL_DOCUMENT_PATH.fullmatch(path) is not None:
        if len(payload) < 500 or not payload.startswith(b"%PDF"):
            raise TDnetMaterialError("official TDnet document response is not a PDF")
        if "pdf" not in content_type.lower():
            raise TDnetMaterialError("official TDnet PDF content type is invalid")
        return
    raise TDnetMaterialError("unsupported official TDnet response path")


def download_official_tdnet_urls(
    urls: Iterable[str] | str,
    destination: str | Path,
    *,
    overwrite: bool = False,
    timeout: float = 30.0,
) -> list[OfficialDownloadReceipt]:
    """Download explicit official index/PDF URLs with hash-bound receipts."""

    if timeout <= 0:
        raise ValueError("timeout must be positive")
    values = [urls] if isinstance(urls, str) else list(urls)
    if not values:
        raise TDnetMaterialError("at least one official TDnet URL is required")
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    receipts: list[OfficialDownloadReceipt] = []
    for raw_url in values:
        url, filename = _validated_official_url(
            raw_url,
            allow_index=True,
            allow_document=True,
        )
        path = target / filename
        if not path.exists() and _receipt_path(path).exists():
            raise TDnetMaterialError(
                "official TDnet receipt exists without its raw file"
            )
        if path.exists() and not overwrite:
            receipts.append(_load_receipt(path, url))
            continue
        requested_at = datetime.now(tz=TOKYO)
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "tse-session-ranker/0.3 (+material research)",
                "Accept": "text/html,application/pdf",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_url = str(response.geturl())
            if response_url != url:
                raise TDnetMaterialError(
                    "official TDnet request was redirected"
                )
            status = int(response.status)
            content_type = response.headers.get_content_type()
            last_modified = response.headers.get("Last-Modified")
            payload = response.read()
        received_at = datetime.now(tz=TOKYO)
        if status != 200:
            raise TDnetMaterialError(f"official TDnet returned HTTP {status}")
        _validate_download_payload(url, content_type, payload)
        if path.exists():
            _archive_existing_snapshot(path, url)
        with tempfile.NamedTemporaryFile(
            prefix=filename + ".",
            suffix=".part",
            dir=target,
            delete=False,
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        temporary.replace(path)
        receipt = OfficialDownloadReceipt(
            url=url,
            path=str(path.resolve()),
            requested_at=requested_at,
            received_at=received_at,
            http_status=status,
            content_type=content_type,
            last_modified=last_modified,
            bytes=len(payload),
            sha256=_sha256_bytes(payload),
            downloaded_in_run=True,
        )
        _write_receipt(receipt)
        receipts.append(receipt)
    return receipts


@dataclass(frozen=True)
class OfficialTDnetDisclosure:
    """One disclosure row parsed from the official current index."""

    published_at: datetime
    code: str
    name: str
    title: str
    document_url: str
    xbrl_url: str | None
    exchange: str

    @property
    def document_id(self) -> str:
        match = OFFICIAL_DOCUMENT_PATH.fullmatch(
            urllib.parse.urlparse(self.document_url).path
        )
        if match is None:  # pragma: no cover - constructor output invariant
            raise TDnetMaterialError("invalid disclosure document URL")
        return match.group("document_id")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["published_at"] = self.published_at.isoformat()
        payload["document_id"] = self.document_id
        return payload


@dataclass(frozen=True)
class OfficialTDnetIndexPage:
    """Parsed official index page with pagination coverage."""

    index_date: date
    range_start: int
    range_end: int
    total_disclosures: int
    page_urls: tuple[str, ...]
    disclosures: tuple[OfficialTDnetDisclosure, ...]


_TD_CLASS_TO_FIELD = {
    "kjTime": "time",
    "kjCode": "code",
    "kjName": "name",
    "kjTitle": "title",
    "kjXbrl": "xbrl",
    "kjPlace": "exchange",
    "kjHistroy": "history",
}


class _OfficialIndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[dict[str, str]] = []
        self.page_hrefs: set[str] = set()
        self.anchor_hrefs: list[str] = []
        self.div_text_stack: list[list[str]] = []
        self.div_texts: list[str] = []
        self.current_row: dict[str, str] | None = None
        self.current_field: str | None = None
        self.current_text: list[str] = []
        self.current_anchor_href: str | None = None
        self.div_field: str | None = None
        self.div_text: list[str] = []
        self.index_date_text = ""
        self.index_date_count = 0
        self.summary_texts: list[str] = []
        self.disclosure_state_text = ""
        self.disclosure_state_count = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        values = {key: value or "" for key, value in attrs}
        onclick = values.get("onclick", values.get("onClick", ""))
        for href in re.findall(
            r"(?:pagerLink|pager)\('([^']+)'\)",
            onclick,
        ):
            self.page_hrefs.add(href)
        if tag == "tr":
            self.current_row = {}
        elif tag == "td" and self.current_row is not None:
            classes = set(values.get("class", "").split())
            matches = [
                field
                for class_name, field in _TD_CLASS_TO_FIELD.items()
                if class_name in classes
            ]
            if len(matches) == 1:
                self.current_field = matches[0]
                self.current_text = []
        elif tag == "a" and self.current_field in {"title", "xbrl"}:
            self.current_anchor_href = values.get("href", "")
        elif tag == "div":
            self.div_text_stack.append([])
            if values.get("id") == "kaiji-date-1":
                self.div_field = "date"
                self.div_text = []
                self.index_date_count += 1
            elif "kaijiSum" in values.get("class", "").split():
                self.div_field = "summary"
                self.div_text = []
            elif values.get("id") == "kaiji-text-1":
                self.div_field = "disclosure_state"
                self.div_text = []
                self.disclosure_state_count += 1
        if tag == "a" and values.get("href", "").strip():
            self.anchor_hrefs.append(values["href"].strip())

    def handle_data(self, data: str) -> None:
        if self.current_field is not None:
            self.current_text.append(data)
        if self.div_field is not None:
            self.div_text.append(data)
        for div_text in self.div_text_stack:
            div_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.current_anchor_href is not None:
            if self.current_row is not None and self.current_field is not None:
                self.current_row[f"{self.current_field}_href"] = (
                    self.current_anchor_href
                )
            self.current_anchor_href = None
        elif tag == "td" and self.current_field is not None:
            if self.current_row is not None:
                self.current_row[self.current_field] = " ".join(
                    " ".join(self.current_text).split()
                )
            self.current_field = None
            self.current_text = []
        elif tag == "tr" and self.current_row is not None:
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = None
        elif tag == "div":
            if self.div_text_stack:
                value = " ".join(
                    " ".join(self.div_text_stack.pop()).split()
                )
                self.div_texts.append(value)
            if self.div_field is not None:
                value = " ".join(" ".join(self.div_text).split())
                if self.div_field == "date":
                    self.index_date_text = value
                elif self.div_field == "summary":
                    self.summary_texts.append(value)
                else:
                    self.disclosure_state_text = value
                self.div_field = None
                self.div_text = []


def parse_official_tdnet_index(
    html: str | bytes,
    *,
    source_url: str,
) -> OfficialTDnetIndexPage:
    """Parse one official index page and fail on pagination/layout drift."""

    url, _ = _validated_official_url(
        source_url,
        allow_index=True,
        allow_document=False,
    )
    path_match = OFFICIAL_INDEX_PATH.fullmatch(
        urllib.parse.urlparse(url).path
    )
    if path_match is None:  # pragma: no cover - validated above
        raise TDnetMaterialError("invalid official index URL")
    text = (
        html.decode("utf-8", errors="strict")
        if isinstance(html, bytes)
        else str(html)
    )
    parser = _OfficialIndexParser()
    parser.feed(text)
    expected_date = datetime.strptime(
        path_match.group("date"), "%Y%m%d"
    ).date()
    date_match = re.fullmatch(
        r"([0-9]{4})年([0-9]{2})月([0-9]{2})日",
        _normalise_text(parser.index_date_text),
    )
    if date_match is None or parser.index_date_count != 1:
        raise TDnetMaterialError("official index date heading is missing")
    heading_date = date(*(int(value) for value in date_match.groups()))
    if heading_date != expected_date:
        raise TDnetMaterialError("official index date does not match its URL")
    summaries = [
        _normalise_text(value).replace(" ", "")
        for value in parser.summary_texts
    ]
    disclosure_state = _normalise_text(
        parser.disclosure_state_text
    ).replace(" ", "")
    if not summaries:
        if (
            parser.disclosure_state_count != 1
            or disclosure_state != "に開示された情報はありません。"
            or parser.rows
            or parser.page_hrefs
            or parser.anchor_hrefs
            or any(
                re.fullmatch(
                    r"[0-9]+[~～][0-9]+件/全[0-9]+件",
                    _normalise_text(value).replace(" ", ""),
                )
                is not None
                for value in parser.div_texts
            )
        ):
            raise TDnetMaterialError(
                "official index pagination summary is invalid"
            )
        range_start, range_end, total = 0, 0, 0
    else:
        if (
            len(summaries) != 2
            or len(set(summaries)) != 1
            or parser.disclosure_state_count != 1
            or disclosure_state != "に開示された情報"
        ):
            raise TDnetMaterialError(
                "official index pagination summary is invalid"
            )
        summary_match = re.fullmatch(
            r"([0-9]+)[~～]([0-9]+)件/全([0-9]+)件",
            summaries[0],
        )
        if summary_match is None:
            raise TDnetMaterialError(
                "official index pagination summary is invalid"
            )
        range_start, range_end, total = (
            int(value) for value in summary_match.groups()
        )
        if total <= 0 or not (1 <= range_start <= range_end <= total):
            raise TDnetMaterialError(
                "official index pagination range is invalid"
            )
    disclosures: list[OfficialTDnetDisclosure] = []
    for row in parser.rows:
        if "time" not in row:
            continue
        required = {"time", "code", "name", "title", "title_href", "exchange"}
        missing = required - set(row)
        if missing:
            raise TDnetMaterialError(
                f"official index disclosure row is incomplete: {sorted(missing)}"
            )
        time_match = re.fullmatch(r"([0-9]{2}):([0-9]{2})", row["time"])
        if time_match is None:
            raise TDnetMaterialError("official index disclosure time is invalid")
        hour, minute = (int(value) for value in time_match.groups())
        if hour > 23 or minute > 59:
            raise TDnetMaterialError("official index disclosure time is invalid")
        document_url = urllib.parse.urljoin(url, row["title_href"])
        _validated_official_url(
            document_url,
            allow_index=False,
            allow_document=True,
        )
        xbrl_url: str | None = None
        if row.get("xbrl_href", "").strip():
            xbrl_url = urllib.parse.urljoin(url, row["xbrl_href"])
            xbrl_path = urllib.parse.urlparse(xbrl_url)
            if (
                xbrl_path.scheme != "https"
                or xbrl_path.hostname not in OFFICIAL_TDNET_HOSTS
                or xbrl_path.username is not None
                or xbrl_path.password is not None
                or xbrl_path.port not in {None, 443}
                or xbrl_path.params
                or xbrl_path.query
                or xbrl_path.fragment
                or OFFICIAL_XBRL_PATH.fullmatch(xbrl_path.path) is None
            ):
                raise TDnetMaterialError("official index XBRL URL is invalid")
        title = row["title"].strip()
        name = row["name"].strip()
        exchange = row["exchange"].strip()
        if not title or not name or not exchange:
            raise TDnetMaterialError("official index contains an empty field")
        disclosures.append(
            OfficialTDnetDisclosure(
                published_at=datetime.combine(
                    expected_date,
                    time(hour, minute),
                    tzinfo=TOKYO,
                ),
                code=_normalise_official_code(row["code"]),
                name=name,
                title=title,
                document_url=document_url,
                xbrl_url=xbrl_url,
                exchange=exchange,
            )
        )
    expected_rows = 0 if total == 0 else range_end - range_start + 1
    if len(disclosures) != expected_rows:
        raise TDnetMaterialError(
            "official index row count does not match pagination summary"
        )
    linked_documents: set[str] = set()
    linked_xbrl: set[str] = set()
    for href in parser.anchor_hrefs:
        resolved = urllib.parse.urljoin(url, href)
        resolved_path = urllib.parse.urlparse(resolved).path
        if resolved_path.lower().endswith(".pdf"):
            validated, _ = _validated_official_url(
                resolved,
                allow_index=False,
                allow_document=True,
            )
            linked_documents.add(validated)
        elif resolved_path.lower().endswith(".zip"):
            parsed_xbrl = urllib.parse.urlparse(resolved)
            if (
                parsed_xbrl.scheme != "https"
                or parsed_xbrl.hostname not in OFFICIAL_TDNET_HOSTS
                or parsed_xbrl.username is not None
                or parsed_xbrl.password is not None
                or parsed_xbrl.port not in {None, 443}
                or parsed_xbrl.params
                or parsed_xbrl.query
                or parsed_xbrl.fragment
                or OFFICIAL_XBRL_PATH.fullmatch(parsed_xbrl.path) is None
            ):
                raise TDnetMaterialError(
                    "official index XBRL URL is invalid"
                )
            linked_xbrl.add(resolved)
    parsed_documents = {item.document_url for item in disclosures}
    if linked_documents != parsed_documents:
        raise TDnetMaterialError("official index contains unparsed PDF links")
    parsed_xbrl = {
        item.xbrl_url
        for item in disclosures
        if item.xbrl_url is not None
    }
    if linked_xbrl != parsed_xbrl:
        raise TDnetMaterialError("official index contains unparsed XBRL links")
    page_urls = {
        url,
        *(
            urllib.parse.urljoin(url, href)
            for href in parser.page_hrefs
        ),
    }
    for page_url in page_urls:
        validated, _ = _validated_official_url(
            page_url,
            allow_index=True,
            allow_document=False,
        )
        page_match = OFFICIAL_INDEX_PATH.fullmatch(
            urllib.parse.urlparse(validated).path
        )
        if (
            page_match is None
            or page_match.group("date") != path_match.group("date")
        ):
            raise TDnetMaterialError("official pagination crosses index dates")
    return OfficialTDnetIndexPage(
        index_date=expected_date,
        range_start=range_start,
        range_end=range_end,
        total_disclosures=total,
        page_urls=tuple(sorted(page_urls)),
        disclosures=tuple(disclosures),
    )


def extract_tdnet_pdf_text(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    timeout: float = 30.0,
    max_pdf_bytes: int = 20 * 1024 * 1024,
    max_text_bytes: int = 10 * 1024 * 1024,
) -> str:
    """Return layout-preserving UTF-8 text from a verified local PDF."""

    source = Path(path)
    if timeout <= 0 or max_pdf_bytes <= 0 or max_text_bytes <= 0:
        raise ValueError("PDF extraction limits must be positive")
    if (
        not source.is_file()
        or source.stat().st_size > max_pdf_bytes
        or not source.read_bytes()[:4] == b"%PDF"
    ):
        raise TDnetMaterialError("TDnet material input is not a PDF")
    before_sha256 = _sha256_file(source)
    if (
        expected_sha256 is not None
        and before_sha256 != expected_sha256
    ):
        raise TDnetMaterialError(
            "TDnet material input does not match its receipt"
        )
    try:
        completed = subprocess.run(
            ["pdftotext", "-layout", str(source), "-"],
            check=True,
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise TDnetMaterialError(
            "pdftotext is required for TDnet material extraction"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise TDnetMaterialError("pdftotext failed for TDnet material") from exc
    except subprocess.TimeoutExpired as exc:
        raise TDnetMaterialError("pdftotext timed out for TDnet material") from exc
    if _sha256_file(source) != before_sha256:
        raise TDnetMaterialError(
            "TDnet material changed during text extraction"
        )
    if len(completed.stdout) > max_text_bytes:
        raise TDnetMaterialError("TDnet PDF text exceeds the extraction limit")
    try:
        text = completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TDnetMaterialError("TDnet PDF text is not valid UTF-8") from exc
    if len(text.strip()) < 100:
        raise TDnetMaterialError("TDnet PDF produced insufficient text")
    return text


def _numeric_tokens(value: str, expected: int) -> tuple[float, ...]:
    raw_tokens = value.split()
    tokens: list[float] = []
    for raw in raw_tokens:
        token = raw.strip()
        if token in {"-", "--", "―"}:
            raise TDnetMaterialError(
                "forecast revision table contains an unavailable numeric value"
            )
        negative = token.startswith(("△", "▲"))
        token = token.lstrip("△▲")
        if token.startswith("(") and token.endswith(")"):
            negative = True
            token = token[1:-1]
        token = token.replace(",", "")
        if re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?", token) is None:
            raise TDnetMaterialError(
                f"forecast revision row contains an unknown token: {raw!r}"
            )
        numeric = float(token)
        tokens.append(-abs(numeric) if negative else numeric)
    if len(tokens) != expected or not all(isfinite(item) for item in tokens):
        raise TDnetMaterialError(
            f"forecast revision row requires exactly {expected} finite values"
        )
    return tuple(tokens)


def _matched_line(
    text: str,
    pattern: str,
    *,
    name: str,
) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise TDnetMaterialError(
            f"forecast revision requires exactly one {name} row"
        )
    return str(matches[0]).strip()


def _matched_line_span(
    text: str,
    pattern: str,
    *,
    name: str,
) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if len(matches) != 1:
        raise TDnetMaterialError(
            f"forecast revision requires exactly one {name} row"
        )
    return matches[0]


@dataclass(frozen=True)
class ForecastRevision:
    """Strictly parsed full-year forecast revision."""

    company: str
    code: str
    period: str
    scope: str
    unit: str
    prior_sales: float
    prior_operating_profit: float
    prior_ordinary_profit: float
    prior_net_profit: float
    prior_eps: float
    current_sales: float
    current_operating_profit: float
    current_ordinary_profit: float
    current_net_profit: float
    current_eps: float
    reported_sales_revision_pct: float
    reported_operating_revision_pct: float
    reported_ordinary_revision_pct: float
    reported_net_revision_pct: float
    reported_sales_delta: float
    reported_operating_delta: float
    reported_ordinary_delta: float
    reported_net_delta: float
    parser_version: str = MATERIAL_PARSER_VERSION

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_forecast_revision_text(text: str) -> ForecastRevision:
    """Parse one unambiguous Japanese full-year forecast-revision table."""

    normalized = _normalise_text(text)
    if not all(
        label in normalized
        for label in ("売上高", "営業利益", "経常利益", "純利益", "百万円")
    ):
        raise TDnetMaterialError("forecast revision table headers are incomplete")
    company = _matched_line(
        normalized,
        r"^\s*上場会社名\s+(.+?)\s*$",
        name="company",
    )
    code = _normalise_official_code(
        _matched_line(
            normalized,
            r"^\s*[（(]?コード番号\s+([0-9A-Z]{4,5})[）)]?\s*$",
            name="security code",
        )
    )
    period_match = _matched_line_span(
        normalized,
        r"^\s*(.+?通期(?:連結|個別)業績予想数値の修正.+?)\s*$",
        name="full-year period",
    )
    period = str(period_match.group(1)).strip()
    scope = "consolidated" if "連結" in period else "non_consolidated"
    prior_match = _matched_line_span(
        normalized,
        r"^\s*前回発表予想\s*\(A\)\s+(.+?)\s*$",
        name="prior forecast",
    )
    current_match = _matched_line_span(
        normalized,
        r"^\s*今回(?:修正|発表)予想\s*\(B\)\s+(.+?)\s*$",
        name="current forecast",
    )
    delta_match = _matched_line_span(
        normalized,
        r"^\s*増減額\s*\(B-A\)\s+(.+?)\s*$",
        name="revision amount",
    )
    reported_match = _matched_line_span(
        normalized,
        r"^\s*増減率\s*\(%\)\s+(.+?)\s*$",
        name="revision percentage",
    )
    if not (
        period_match.end()
        < prior_match.start()
        < current_match.start()
        < delta_match.start()
        < reported_match.start()
    ):
        raise TDnetMaterialError(
            "forecast revision table rows are out of order"
        )
    header = normalized[period_match.end() : prior_match.start()]
    header_lines = [
        line for line in header.splitlines() if line.strip()
    ]
    column_header_lines = [
        (index, line)
        for index, line in enumerate(header_lines)
        if all(
            label in line
            for label in ("売上高", "営業利益", "経常利益")
        )
    ]
    ownership_header_lines = [
        (index, line)
        for index, line in enumerate(header_lines)
        if "親会社株主に帰属" in line and "1株当たり" in line
    ]
    profit_header_lines = [
        (index, line)
        for index, line in enumerate(header_lines)
        if line.count("当期純利益") == 2
    ]
    unit_lines = [
        (index, line)
        for index, line in enumerate(header_lines)
        if line.count("百万円") == 4
        and re.search(r"円\s*銭", line) is not None
    ]
    if len(column_header_lines) != 1:
        raise TDnetMaterialError(
            "forecast revision column header is ambiguous"
        )
    if (
        len(ownership_header_lines) != 1
        or len(profit_header_lines) != 1
        or len(unit_lines) != 1
    ):
        raise TDnetMaterialError(
            "forecast revision table header and units are not bound"
        )
    column_index, column_header = column_header_lines[0]
    ownership_index, ownership_header = ownership_header_lines[0]
    profit_index, profit_header = profit_header_lines[0]
    unit_index, unit_header = unit_lines[0]
    sales_position = column_header.index("売上高")
    operating_position = column_header.index("営業利益")
    ordinary_position = column_header.index("経常利益")
    net_income_positions = [
        match.start() for match in re.finditer("当期純利益", profit_header)
    ]
    eps_unit_match = re.search(r"円\s*銭", unit_header)
    if eps_unit_match is None:  # pragma: no cover - selected above
        raise TDnetMaterialError(
            "forecast revision table header and units are not bound"
        )
    if (
        len(net_income_positions) != 2
        or header.count("売上高") != 1
        or header.count("営業利益") != 1
        or header.count("経常利益") != 1
        or header.count("親会社株主に帰属") != 1
        or header.count("1株当たり") != 1
        or header.count("当期純利益") != 2
        or header.count("百万円") != 4
        or len(re.findall(r"円\s*銭", header)) != 1
        or column_header.count("売上高") != 1
        or column_header.count("営業利益") != 1
        or column_header.count("経常利益") != 1
        or ownership_header.count("親会社株主に帰属") != 1
        or ownership_header.count("1株当たり") != 1
        or not sales_position < operating_position < ordinary_position
        or ownership_header.index("親会社株主に帰属")
        >= ownership_header.index("1株当たり")
        or net_income_positions[0] >= net_income_positions[1]
        or not ownership_index < column_index < profit_index < unit_index
        or unit_header.rfind("百万円") >= eps_unit_match.start()
    ):
        raise TDnetMaterialError(
            "forecast revision table header and units are not bound"
        )
    prior = _numeric_tokens(
        str(prior_match.group(1)).strip(),
        5,
    )
    current = _numeric_tokens(
        str(current_match.group(1)).strip(),
        5,
    )
    delta = _numeric_tokens(
        str(delta_match.group(1)).strip(),
        4,
    )
    reported = _numeric_tokens(
        str(reported_match.group(1)).strip(),
        4,
    )
    for prior_value, current_value, delta_value, reported_value in zip(
        prior[:4], current[:4], delta, reported, strict=True
    ):
        if abs((current_value - prior_value) - delta_value) > 0.01:
            raise TDnetMaterialError(
                "reported forecast revision amount does not recompute"
            )
        if prior_value == 0:
            raise TDnetMaterialError(
                "forecast revision percentage has a zero prior denominator"
            )
        recomputed = 100.0 * (current_value / prior_value - 1.0)
        if abs(recomputed - reported_value) > 0.11:
            raise TDnetMaterialError(
                "reported forecast revision percentage does not recompute"
            )
    return ForecastRevision(
        company=company,
        code=code,
        period=period,
        scope=scope,
        unit="JPY_million",
        prior_sales=prior[0],
        prior_operating_profit=prior[1],
        prior_ordinary_profit=prior[2],
        prior_net_profit=prior[3],
        prior_eps=prior[4],
        current_sales=current[0],
        current_operating_profit=current[1],
        current_ordinary_profit=current[2],
        current_net_profit=current[3],
        current_eps=current[4],
        reported_sales_revision_pct=reported[0],
        reported_operating_revision_pct=reported[1],
        reported_ordinary_revision_pct=reported[2],
        reported_net_revision_pct=reported[3],
        reported_sales_delta=delta[0],
        reported_operating_delta=delta[1],
        reported_ordinary_delta=delta[2],
        reported_net_delta=delta[3],
    )


def forecast_material_strength(
    revision: ForecastRevision,
    *,
    cap: float = 0.30,
) -> float:
    """Return capped signed operating-profit growth for research arm A."""

    if not isfinite(cap) or cap <= 0 or cap > 1:
        raise ValueError("material strength cap must be in (0, 1]")
    prior = revision.prior_operating_profit
    current = revision.current_operating_profit
    if not isfinite(prior) or not isfinite(current) or prior <= 0:
        raise TDnetMaterialError(
            "operating-profit material strength requires a positive prior forecast"
        )
    if (
        revision.parser_version != MATERIAL_PARSER_VERSION
        or revision.unit != "JPY_million"
        or revision.scope not in {"consolidated", "non_consolidated"}
    ):
        raise TDnetMaterialError(
            "material strength requires the strict parser contract"
        )
    prior_values = (
        revision.prior_sales,
        revision.prior_operating_profit,
        revision.prior_ordinary_profit,
        revision.prior_net_profit,
    )
    current_values = (
        revision.current_sales,
        revision.current_operating_profit,
        revision.current_ordinary_profit,
        revision.current_net_profit,
    )
    deltas = (
        revision.reported_sales_delta,
        revision.reported_operating_delta,
        revision.reported_ordinary_delta,
        revision.reported_net_delta,
    )
    rates = (
        revision.reported_sales_revision_pct,
        revision.reported_operating_revision_pct,
        revision.reported_ordinary_revision_pct,
        revision.reported_net_revision_pct,
    )
    for prior_value, current_value, delta_value, rate in zip(
        prior_values,
        current_values,
        deltas,
        rates,
        strict=True,
    ):
        if not all(
            isfinite(value)
            for value in (prior_value, current_value, delta_value, rate)
        ):
            raise TDnetMaterialError(
                "material strength requires finite revision values"
            )
        if prior_value == 0:
            raise TDnetMaterialError(
                "material strength requires nonzero prior forecasts"
            )
        if abs((current_value - prior_value) - delta_value) > 0.01:
            raise TDnetMaterialError(
                "material strength revision amount does not recompute"
            )
        recomputed = 100.0 * (current_value / prior_value - 1.0)
        if abs(recomputed - rate) > 0.11:
            raise TDnetMaterialError(
                "material strength revision percentage does not recompute"
            )
    growth = current / prior - 1.0
    return max(-cap, min(cap, growth))


def _manifest_receipt(
    receipt: OfficialDownloadReceipt,
    root: Path,
) -> dict[str, object]:
    payload = receipt.to_dict()
    try:
        payload["path"] = str(
            Path(receipt.path).resolve().relative_to(root.resolve())
        )
    except ValueError as exc:  # pragma: no cover - internal invariant
        raise TDnetMaterialError(
            "official TDnet receipt escaped the probe directory"
        ) from exc
    return payload


def collect_official_forecast_revisions(
    index_url: str,
    destination: str | Path,
    *,
    overwrite: bool = False,
    timeout: float = 30.0,
) -> dict[str, object]:
    """Acquire current official pages/PDFs and emit a collectibility manifest.

    This is a source probe, not a historical validation run.  The manifest
    therefore reports source extraction and same-day cutoff status but never
    claims model scores, market outcomes, or order authority.
    """

    first_url, _ = _validated_official_url(
        index_url,
        allow_index=True,
        allow_document=False,
    )
    first_match = OFFICIAL_INDEX_PATH.fullmatch(
        urllib.parse.urlparse(first_url).path
    )
    if first_match is None or first_match.group("page") != "001":
        raise TDnetMaterialError("source probe must start from index page 001")
    destination_path = Path(destination)
    first_receipt = download_official_tdnet_urls(
        first_url,
        destination_path,
        overwrite=overwrite,
        timeout=timeout,
    )[0]
    first_page = parse_official_tdnet_index(
        Path(first_receipt.path).read_bytes(),
        source_url=first_receipt.url,
    )
    if not first_receipt.downloaded_in_run:
        raise TDnetMaterialError(
            "source probe requires fresh index network responses; "
            "rerun with overwrite=True"
        )
    page_receipts_by_url = {first_receipt.url: first_receipt}
    pages_by_url = {first_receipt.url: first_page}
    pending_urls = set(first_page.page_urls) - set(pages_by_url)
    while pending_urls:
        if len(pages_by_url) + len(pending_urls) > max(
            first_page.total_disclosures,
            1,
        ):
            raise TDnetMaterialError(
                "official index advertises an impossible page graph"
            )
        batch_urls = sorted(pending_urls)
        pending_urls.clear()
        batch_receipts = download_official_tdnet_urls(
            batch_urls,
            destination_path,
            overwrite=overwrite,
            timeout=timeout,
        )
        if not all(item.downloaded_in_run for item in batch_receipts):
            raise TDnetMaterialError(
                "source probe requires fresh index network responses; "
                "rerun with overwrite=True"
            )
        for receipt in batch_receipts:
            page = parse_official_tdnet_index(
                Path(receipt.path).read_bytes(),
                source_url=receipt.url,
            )
            if (
                page.index_date != first_page.index_date
                or page.total_disclosures
                != first_page.total_disclosures
            ):
                raise TDnetMaterialError(
                    "official index pages disagree on date or total"
            )
            page_receipts_by_url[receipt.url] = receipt
            pages_by_url[receipt.url] = page
        for receipt in batch_receipts:
            page = pages_by_url[receipt.url]
            pending_urls.update(
                url
                for url in page.page_urls
                if url not in pages_by_url
            )
    page_receipts = [
        page_receipts_by_url[url] for url in sorted(page_receipts_by_url)
    ]
    pages = [pages_by_url[url] for url in sorted(pages_by_url)]
    last_modified = {receipt.last_modified for receipt in page_receipts}
    if None in last_modified or len(last_modified) != 1:
        raise TDnetMaterialError(
            "official index pages lack concordant Last-Modified headers"
        )
    ordered_pages = sorted(pages, key=lambda item: item.range_start)
    disclosures: list[OfficialTDnetDisclosure] = []
    if first_page.total_disclosures == 0:
        if (
            len(ordered_pages) != 1
            or ordered_pages[0].range_start != 0
            or ordered_pages[0].range_end != 0
            or ordered_pages[0].disclosures
        ):
            raise TDnetMaterialError(
                "official zero-disclosure pagination is invalid"
            )
    else:
        expected_start = 1
        for page in ordered_pages:
            if page.range_start != expected_start:
                raise TDnetMaterialError(
                    "official index pagination has a gap or overlap"
                )
            expected_start = page.range_end + 1
            disclosures.extend(page.disclosures)
        if (
            expected_start != first_page.total_disclosures + 1
            or len(disclosures) != first_page.total_disclosures
            or len({item.document_url for item in disclosures})
            != len(disclosures)
        ):
            raise TDnetMaterialError(
                "official index pagination is incomplete"
            )
    material_disclosures = [
        item
        for item in disclosures
        if FORECAST_REVISION_TITLE.search(item.title) is not None
    ]
    document_receipts = download_official_tdnet_urls(
        [item.document_url for item in material_disclosures],
        destination_path,
        overwrite=overwrite,
        timeout=timeout,
    ) if material_disclosures else []
    if not all(receipt.downloaded_in_run for receipt in document_receipts):
        raise TDnetMaterialError(
            "source probe requires fresh document network responses; "
            "rerun with overwrite=True"
        )
    stability_receipts = download_official_tdnet_urls(
        [receipt.url for receipt in page_receipts],
        destination_path / "stability",
        overwrite=True,
        timeout=timeout,
    )
    stability_by_url = {
        receipt.url: receipt for receipt in stability_receipts
    }
    if set(stability_by_url) != set(page_receipts_by_url):
        raise TDnetMaterialError(
            "official index stability recheck is incomplete"
        )
    for url, initial_receipt in page_receipts_by_url.items():
        stability_receipt = stability_by_url[url]
        stability_page = parse_official_tdnet_index(
            Path(stability_receipt.path).read_bytes(),
            source_url=stability_receipt.url,
        )
        if (
            not stability_receipt.downloaded_in_run
            or stability_receipt.sha256 != initial_receipt.sha256
            or stability_receipt.last_modified
            != initial_receipt.last_modified
            or stability_page != pages_by_url[url]
        ):
            raise TDnetMaterialError(
                "official index changed during the source probe"
            )
    receipts_by_url = {item.url: item for item in document_receipts}
    extracted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    cutoff = datetime.combine(
        first_page.index_date,
        time(8, 58, 59),
        tzinfo=TOKYO,
    )
    for disclosure in material_disclosures:
        receipt = receipts_by_url[disclosure.document_url]
        try:
            revision = parse_forecast_revision_text(
                extract_tdnet_pdf_text(
                    receipt.path,
                    expected_sha256=receipt.sha256,
                    timeout=timeout,
                )
            )
            if revision.code != disclosure.code:
                raise TDnetMaterialError(
                    "official index/PDF security codes do not match"
                )
            computed_at = datetime.now(tz=TOKYO)
            available_at = max(
                computed_at,
                receipt.received_at,
                *(item.received_at for item in page_receipts),
                *(item.received_at for item in stability_receipts),
            )
            timestamp_cutoff_check = bool(
                disclosure.published_at <= cutoff
                and all(
                    item.requested_at <= item.received_at <= cutoff
                    for item in page_receipts
                )
                and receipt.requested_at <= receipt.received_at <= cutoff
                and computed_at <= cutoff
                and all(
                    item.requested_at <= item.received_at <= cutoff
                    for item in stability_receipts
                )
            )
            extracted.append(
                {
                    "disclosure": disclosure.to_dict(),
                    "document_receipt": _manifest_receipt(
                        receipt,
                        destination_path,
                    ),
                    "computed_at": computed_at.isoformat(),
                    "available_at": available_at.isoformat(),
                    "revision": revision.to_dict(),
                    "material_strength": forecast_material_strength(revision),
                    "record_timestamp_cutoff_check": timestamp_cutoff_check,
                    "candidate_eligible": False,
                    "quarantine_reason": (
                        "sample_source_probe_is_not_a_source-complete_"
                        "historical_or_prospective_cohort"
                    ),
                }
            )
        except TDnetMaterialError as exc:
            rejected.append(
                {
                    "disclosure": disclosure.to_dict(),
                    "document_receipt": _manifest_receipt(
                        receipt,
                        destination_path,
                    ),
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
    return {
        "schema_version": 1,
        "probe_type": "official_tdnet_current_collectibility",
        "index_date": first_page.index_date.isoformat(),
        "decision_cutoff": cutoff.isoformat(),
        "source": {
            "provider": "Tokyo Stock Exchange TDnet",
            "official_host": "www.release.tdnet.info",
            "first_index_url": first_url,
            "last_modified_header_concordance": next(iter(last_modified)),
            "page_receipts": [
                _manifest_receipt(receipt, destination_path)
                for receipt in page_receipts
            ],
            "stability_receipts": [
                _manifest_receipt(receipt, destination_path)
                for receipt in stability_receipts
            ],
            "document_receipts": [
                _manifest_receipt(receipt, destination_path)
                for receipt in document_receipts
            ],
        },
        "coverage": {
            "index_pages": len(pages),
            "index_disclosures": len(disclosures),
            "forecast_revision_documents": len(material_disclosures),
            "strict_extractions": len(extracted),
            "strict_rejections": len(rejected),
            "historical_validation_sessions": 0,
        },
        "candidate_records": [],
        "quarantined_extractions": extracted,
        "rejected_documents": rejected,
        "readiness": {
            "supported_table_shape_sample_verified": bool(extracted),
            "source_complete": False,
            "same_day_pit_complete_sessions": 0,
            "historical_validation_ready": False,
            "performance_evaluation_allowed": False,
            "reason": (
                "Current-source acquisition does not establish a sealed "
                "08:58 snapshot or historical coverage."
            ),
        },
        "integrity": {
            "fresh_network_acquisition": True,
            "clock_sync_evidence_present": False,
            "title_used_as_document_body": False,
            "proxy_values_used": False,
            "model_scores_computed": 0,
            "market_outcomes_inspected_by_probe": False,
            "same_day_ohlc_predictors_used": False,
            "orders_allowed": False,
        },
    }
