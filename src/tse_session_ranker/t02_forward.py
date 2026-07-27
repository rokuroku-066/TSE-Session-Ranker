"""Pure, fail-closed primitives for the prospective T02 shadow.

This module intentionally does not acquire market data and does not authorize
orders.  It provides the deterministic part of the Phase-0 dry run:

* point-in-time validation for a live TDnet cutoff snapshot;
* a later, audit-only comparison with the finalized archive;
* monthly expanding-fold eligibility and a frozen T02 fit;
* a pre-score, hash-bound fold manifest;
* deterministic selected/cash/fail-closed decisions; and
* an append-only hash-chain ledger.

The finalized archive is never an input to a live decision.  It can only
invalidate a session for later evaluation if its pre-cutoff prefix differs
from the live snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
import hashlib
import json
import platform
import re
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from .exceptions import DataValidationError, LeakageError


TOKYO = ZoneInfo("Asia/Tokyo")
T02_CANDIDATE_ID = "v10_t02_char_value_event_top1"
CANONICAL_JSON_CONTRACT = "project_canonical_json_v1"
DEFAULT_CUTOFF_TIME = time(8, 58, 59)
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class T02ForwardError(DataValidationError):
    """Base validation error for the T02 forward runtime."""


class FoldUnavailableError(T02ForwardError):
    """Raised when a monthly fold cannot be fitted before scoring."""


class DecisionState(str, Enum):
    """Exhaustive Phase-0 decision states."""

    SELECTED = "selected"
    CASH_NO_EVENT = "cash_no_event"
    FAIL_CLOSED_SOURCE = "fail_closed_source"
    FAIL_CLOSED_MODEL = "fail_closed_model"


def session_cutoff(session_date: date) -> datetime:
    """Return the exact 08:58:59 JST decision cutoff."""

    if not isinstance(session_date, date) or isinstance(session_date, datetime):
        raise T02ForwardError("session_date must be a date")
    return datetime.combine(session_date, DEFAULT_CUTOFF_TIME, tzinfo=TOKYO)


def require_jst_timestamp(
    value: datetime,
    name: str,
    *,
    session_date: date | None = None,
    at_or_before_cutoff: bool = False,
) -> datetime:
    """Validate a strict JST timestamp without silently converting timezones.

    An aware UTC value that represents the same instant is deliberately
    rejected.  Source adapters must normalize timestamps to JST before they
    enter the canonical forward ledger.
    """

    if not isinstance(value, datetime):
        raise T02ForwardError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise T02ForwardError(f"{name} must be timezone-aware JST")
    if value.utcoffset() != timedelta(hours=9):
        raise T02ForwardError(f"{name} must use the JST UTC+09:00 offset")
    if session_date is not None and at_or_before_cutoff:
        cutoff = session_cutoff(session_date)
        if value > cutoff:
            raise LeakageError(
                f"{name}={value.isoformat()} is after "
                f"{cutoff.isoformat()}"
            )
    return value


def _canonical_value(value: Any, excluded_fields: frozenset[str]) -> Any:
    if isinstance(value, Enum):
        return _canonical_value(value.value, excluded_fields)
    if isinstance(value, datetime):
        require_jst_timestamp(value, "canonical datetime")
        return value.isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _canonical_value(value.item(), excluded_fields)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise T02ForwardError("canonical JSON object keys must be strings")
            if key in excluded_fields:
                continue
            normalized[key] = _canonical_value(item, excluded_fields)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item, excluded_fields) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise T02ForwardError("canonical JSON rejects non-finite floats")
        return value
    raise T02ForwardError(
        f"unsupported canonical JSON value: {type(value).__name__}"
    )


def canonical_json_bytes(
    value: Any,
    *,
    exclude_fields: Iterable[str] = (),
) -> bytes:
    """Return canonical UTF-8 JSON bytes.

    Hash fields are **not** removed implicitly.  Every excluded field must be
    named explicitly by the caller.  Exclusion applies recursively by exact
    key name; this makes self-hash rules visible at each call site.
    """

    excluded = frozenset(str(field) for field in exclude_fields)
    normalized = _canonical_value(value, excluded)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(
    value: Any,
    *,
    exclude_fields: Iterable[str] = (),
) -> str:
    """Hash canonical JSON, excluding only explicitly named fields."""

    return hashlib.sha256(
        canonical_json_bytes(value, exclude_fields=exclude_fields)
    ).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise T02ForwardError(f"{name} must be a lowercase SHA-256")
    return value


def _array_sha256(value: Any) -> str:
    array = np.asarray(value)
    if array.dtype.kind == "f":
        array = np.asarray(array, dtype="<f8")
    elif array.dtype.kind in {"i", "u"}:
        array = np.asarray(array, dtype="<i8")
    else:
        raise T02ForwardError("only numeric arrays can be hashed")
    payload = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes_sha256": hashlib.sha256(
            np.ascontiguousarray(array).tobytes(order="C")
        ).hexdigest(),
    }
    return canonical_json_sha256(payload)


@dataclass(frozen=True)
class TDnetDisclosure:
    """One title-bearing TDnet disclosure."""

    published_at: datetime
    code: str
    title: str
    document_url: str = ""

    @property
    def identity_payload(self) -> dict[str, Any]:
        return {
            "published_at": self.published_at,
            "code": self.code,
            "title": self.title,
            "document_url": self.document_url,
        }


def _validate_disclosure(
    disclosure: TDnetDisclosure,
    *,
    cutoff: datetime | None,
) -> TDnetDisclosure:
    require_jst_timestamp(disclosure.published_at, "published_at")
    code = str(disclosure.code).strip()
    title = str(disclosure.title).strip()
    if not code or not re.fullmatch(r"[0-9A-Z]{4,5}", code):
        raise T02ForwardError("disclosure code must contain 4-5 digits/A-Z")
    if not title:
        raise T02ForwardError("disclosure title must be non-empty")
    if cutoff is not None and disclosure.published_at > cutoff:
        raise LeakageError(
            "live snapshot contains a disclosure published after cutoff"
        )
    return disclosure


def _sorted_disclosures(
    disclosures: Iterable[TDnetDisclosure],
    *,
    cutoff: datetime | None,
) -> tuple[TDnetDisclosure, ...]:
    validated = [
        _validate_disclosure(item, cutoff=cutoff) for item in disclosures
    ]
    ordered = tuple(
        sorted(
            validated,
            key=lambda item: (
                item.published_at,
                item.code,
                item.title,
                item.document_url,
            ),
        )
    )
    identities = [
        canonical_json_sha256(item.identity_payload) for item in ordered
    ]
    if len(identities) != len(set(identities)):
        raise T02ForwardError("duplicate disclosure in snapshot/archive")
    return ordered


@dataclass(frozen=True)
class LiveTDnetSnapshot:
    """A provider assertion of the titles completely observed at cutoff."""

    session_date: date
    source_id: str
    snapshot_id: str
    request_started_at: datetime
    source_received_at: datetime
    computed_at: datetime
    source_complete: bool
    source_manifest_sha256: str
    raw_bytes_sha256: str
    parser_sha256: str
    source_watermark: str
    disclosures: tuple[TDnetDisclosure, ...] = ()


def validate_live_snapshot(snapshot: LiveTDnetSnapshot) -> LiveTDnetSnapshot:
    """Validate that a live snapshot was fully usable by 08:58:59 JST."""

    if not isinstance(snapshot, LiveTDnetSnapshot):
        raise T02ForwardError("snapshot must be a LiveTDnetSnapshot")
    if not str(snapshot.source_id).strip() or not str(snapshot.snapshot_id).strip():
        raise T02ForwardError("source_id and snapshot_id must be non-empty")
    _require_sha256(
        snapshot.source_manifest_sha256,
        "source_manifest_sha256",
    )
    _require_sha256(snapshot.raw_bytes_sha256, "raw_bytes_sha256")
    _require_sha256(snapshot.parser_sha256, "parser_sha256")
    if not str(snapshot.source_watermark).strip():
        raise T02ForwardError("source_watermark must be non-empty")
    require_jst_timestamp(
        snapshot.request_started_at,
        "request_started_at",
        session_date=snapshot.session_date,
        at_or_before_cutoff=True,
    )
    require_jst_timestamp(
        snapshot.source_received_at,
        "source_received_at",
        session_date=snapshot.session_date,
        at_or_before_cutoff=True,
    )
    require_jst_timestamp(
        snapshot.computed_at,
        "computed_at",
        session_date=snapshot.session_date,
        at_or_before_cutoff=True,
    )
    if not (
        snapshot.request_started_at
        <= snapshot.source_received_at
        <= snapshot.computed_at
    ):
        raise T02ForwardError(
            "snapshot provenance must satisfy "
            "request_started_at <= source_received_at <= computed_at"
        )
    if not isinstance(snapshot.source_complete, bool):
        raise T02ForwardError("source_complete must be boolean")
    cutoff = session_cutoff(snapshot.session_date)
    disclosures = _sorted_disclosures(snapshot.disclosures, cutoff=cutoff)
    if disclosures != snapshot.disclosures:
        raise T02ForwardError(
            "live disclosures must be stored in canonical order"
        )
    if any(
        disclosure.published_at > snapshot.source_received_at
        for disclosure in disclosures
    ):
        raise T02ForwardError(
            "a disclosure cannot be received before it was published"
        )
    return snapshot


def live_snapshot_sha256(snapshot: LiveTDnetSnapshot) -> str:
    """Hash the exact live source assertion used by candidate generation."""

    validate_live_snapshot(snapshot)
    return canonical_json_sha256(
        {
            "session_date": snapshot.session_date,
            "source_id": snapshot.source_id,
            "snapshot_id": snapshot.snapshot_id,
            "request_started_at": snapshot.request_started_at,
            "source_received_at": snapshot.source_received_at,
            "computed_at": snapshot.computed_at,
            "source_complete": snapshot.source_complete,
            "source_manifest_sha256": snapshot.source_manifest_sha256,
            "raw_bytes_sha256": snapshot.raw_bytes_sha256,
            "parser_sha256": snapshot.parser_sha256,
            "source_watermark": snapshot.source_watermark,
            "disclosures": [
                item.identity_payload for item in snapshot.disclosures
            ],
        }
    )


@dataclass(frozen=True)
class FinalTDnetArchive:
    """A later finalized archive used only for post-decision auditing."""

    session_date: date
    source_id: str
    archive_id: str
    request_started_at: datetime
    source_received_at: datetime
    computed_at: datetime
    finalized: bool
    raw_bytes_sha256: str
    parser_sha256: str
    source_watermark: str
    disclosures: tuple[TDnetDisclosure, ...] = ()


@dataclass(frozen=True)
class ArchivePrefixAudit:
    """Comparison of live titles with the finalized pre-cutoff prefix."""

    session_date: date
    matched: bool
    live_prefix_sha256: str
    final_prefix_sha256: str
    live_raw_bytes_sha256: str
    final_raw_bytes_sha256: str
    live_parser_sha256: str
    final_parser_sha256: str
    live_source_watermark: str
    final_source_watermark: str
    missing_from_live: tuple[str, ...]
    unexpected_in_live: tuple[str, ...]
    audit_sha256: str


def audit_final_archive_prefix(
    snapshot: LiveTDnetSnapshot,
    archive: FinalTDnetArchive,
) -> ArchivePrefixAudit:
    """Compare a live snapshot with the final archive's pre-cutoff prefix.

    This function is audit-only.  The archive receipt time is required to be
    after the decision cutoff, making accidental use in live scoring visible.
    """

    validate_live_snapshot(snapshot)
    if archive.session_date != snapshot.session_date:
        raise T02ForwardError("archive and snapshot session dates differ")
    if archive.source_id != snapshot.source_id:
        raise T02ForwardError("archive and snapshot source IDs differ")
    if not str(archive.archive_id).strip():
        raise T02ForwardError("archive_id must be non-empty")
    require_jst_timestamp(archive.request_started_at, "archive request_started_at")
    require_jst_timestamp(archive.source_received_at, "archive source_received_at")
    require_jst_timestamp(archive.computed_at, "archive computed_at")
    cutoff = session_cutoff(snapshot.session_date)
    if archive.request_started_at <= cutoff:
        raise T02ForwardError(
            "final archive must be a post-cutoff audit artifact"
        )
    if not (
        archive.request_started_at
        <= archive.source_received_at
        <= archive.computed_at
    ):
        raise T02ForwardError(
            "archive provenance must satisfy "
            "request_started_at <= source_received_at <= computed_at"
        )
    if archive.finalized is not True:
        raise T02ForwardError("archive must be explicitly finalized")
    _require_sha256(
        archive.raw_bytes_sha256,
        "archive raw_bytes_sha256",
    )
    _require_sha256(archive.parser_sha256, "archive parser_sha256")
    if not str(archive.source_watermark).strip():
        raise T02ForwardError("archive source_watermark must be non-empty")

    all_final = _sorted_disclosures(archive.disclosures, cutoff=None)
    if all_final != archive.disclosures:
        raise T02ForwardError(
            "final disclosures must be stored in canonical order"
        )
    if any(
        item.published_at > archive.source_received_at
        for item in all_final
    ):
        raise T02ForwardError(
            "archive contains a disclosure published after archive receipt"
        )
    final_prefix = tuple(
        item for item in all_final if item.published_at <= cutoff
    )
    live_payloads = [
        item.identity_payload for item in snapshot.disclosures
    ]
    final_payloads = [item.identity_payload for item in final_prefix]
    live_by_hash = {
        canonical_json_sha256(payload): payload for payload in live_payloads
    }
    final_by_hash = {
        canonical_json_sha256(payload): payload for payload in final_payloads
    }
    missing = tuple(sorted(set(final_by_hash) - set(live_by_hash)))
    unexpected = tuple(sorted(set(live_by_hash) - set(final_by_hash)))
    report: dict[str, Any] = {
        "session_date": snapshot.session_date,
        "matched": not missing and not unexpected,
        "live_prefix_sha256": canonical_json_sha256(live_payloads),
        "final_prefix_sha256": canonical_json_sha256(final_payloads),
        "live_raw_bytes_sha256": snapshot.raw_bytes_sha256,
        "final_raw_bytes_sha256": archive.raw_bytes_sha256,
        "live_parser_sha256": snapshot.parser_sha256,
        "final_parser_sha256": archive.parser_sha256,
        "live_source_watermark": snapshot.source_watermark,
        "final_source_watermark": archive.source_watermark,
        "missing_from_live": missing,
        "unexpected_in_live": unexpected,
    }
    audit_sha = canonical_json_sha256(report)
    return ArchivePrefixAudit(
        session_date=snapshot.session_date,
        matched=bool(report["matched"]),
        live_prefix_sha256=str(report["live_prefix_sha256"]),
        final_prefix_sha256=str(report["final_prefix_sha256"]),
        live_raw_bytes_sha256=snapshot.raw_bytes_sha256,
        final_raw_bytes_sha256=archive.raw_bytes_sha256,
        live_parser_sha256=snapshot.parser_sha256,
        final_parser_sha256=archive.parser_sha256,
        live_source_watermark=snapshot.source_watermark,
        final_source_watermark=archive.source_watermark,
        missing_from_live=missing,
        unexpected_in_live=unexpected,
        audit_sha256=audit_sha,
    )


@dataclass(frozen=True)
class FoldTrainingRow:
    """One source-complete event/code bundle with a known outcome."""

    session_date: date
    code: str
    bundle_text: str
    open_to_close_return_pct: float
    source_complete: bool
    price_training_eligible: bool
    outcome_available_at: datetime

    @property
    def identity_payload(self) -> dict[str, Any]:
        return {
            "session_date": self.session_date,
            "code": self.code,
            "bundle_text": self.bundle_text,
            "open_to_close_return_pct": self.open_to_close_return_pct,
            "source_complete": self.source_complete,
            "price_training_eligible": self.price_training_eligible,
            "outcome_available_at": self.outcome_available_at,
        }


def eligible_monthly_training_rows(
    rows: Iterable[FoldTrainingRow],
    *,
    score_session: date,
    fit_started_at: datetime,
) -> tuple[FoldTrainingRow, ...]:
    """Return source-complete rows strictly before the scored month."""

    require_jst_timestamp(fit_started_at, "fit_started_at")
    cutoff = session_cutoff(score_session)
    if fit_started_at > cutoff:
        raise LeakageError("fit_started_at is after the score cutoff")
    month_start = date(score_session.year, score_session.month, 1)
    eligible: list[FoldTrainingRow] = []
    identities: set[tuple[date, str]] = set()
    for row in rows:
        require_jst_timestamp(
            row.outcome_available_at,
            "outcome_available_at",
        )
        if not isinstance(row.session_date, date):
            raise T02ForwardError("training session_date must be a date")
        if not str(row.code).strip() or not str(row.bundle_text).strip():
            raise T02ForwardError("training code and bundle_text must be non-empty")
        if not np.isfinite(float(row.open_to_close_return_pct)):
            raise T02ForwardError("training outcome must be finite")
        if row.session_date >= month_start:
            continue
        if row.outcome_available_at >= fit_started_at:
            continue
        if not isinstance(row.source_complete, bool):
            raise T02ForwardError("source_complete must be boolean")
        if not isinstance(row.price_training_eligible, bool):
            raise T02ForwardError(
                "price_training_eligible must be boolean"
            )
        if not row.source_complete or not row.price_training_eligible:
            continue
        identity = (row.session_date, str(row.code))
        if identity in identities:
            raise T02ForwardError(
                "duplicate session/code in eligible training rows"
            )
        identities.add(identity)
        eligible.append(row)
    return tuple(
        sorted(
            eligible,
            key=lambda row: (row.session_date, str(row.code)),
        )
    )


@dataclass(frozen=True)
class T02MonthlyFold:
    """A fitted T02 model and its immutable pre-score manifest."""

    score_month: str
    vectorizer: TfidfVectorizer
    estimator: Ridge
    manifest: Mapping[str, Any]

    @property
    def manifest_sha256(self) -> str:
        return str(self.manifest["pre_score_fold_manifest_sha256"])


def _manifest_is_exact(manifest: Mapping[str, Any]) -> bool:
    recorded = manifest.get("pre_score_fold_manifest_sha256")
    if not isinstance(recorded, str):
        return False
    if manifest.get("canonical_json_contract") != CANONICAL_JSON_CONTRACT:
        return False
    if not str(manifest.get("python_version", "")).strip():
        return False
    expected = canonical_json_sha256(
        manifest,
        exclude_fields={"pre_score_fold_manifest_sha256"},
    )
    return recorded == expected


def fit_t02_monthly_fold(
    rows: Iterable[FoldTrainingRow],
    *,
    score_session: date,
    fit_started_at: datetime,
    fit_completed_at: datetime,
    source_manifest_sha256: str,
    parser_sha256: str,
    scoring_code_sha256: str,
    universe_config_sha256: str,
    eligibility_code_sha256: str,
    minimum_source_complete_sessions: int = 20,
    minimum_bundles: int = 100,
) -> T02MonthlyFold:
    """Fit the frozen T02 specification and seal a pre-score manifest."""

    _require_sha256(source_manifest_sha256, "source_manifest_sha256")
    _require_sha256(parser_sha256, "parser_sha256")
    _require_sha256(scoring_code_sha256, "scoring_code_sha256")
    _require_sha256(universe_config_sha256, "universe_config_sha256")
    _require_sha256(eligibility_code_sha256, "eligibility_code_sha256")
    require_jst_timestamp(fit_started_at, "fit_started_at")
    require_jst_timestamp(fit_completed_at, "fit_completed_at")
    if fit_completed_at < fit_started_at:
        raise FoldUnavailableError("fit_completed_at precedes fit_started_at")
    if fit_completed_at > session_cutoff(score_session):
        raise FoldUnavailableError("monthly fold was not complete before scoring")
    training = eligible_monthly_training_rows(
        rows,
        score_session=score_session,
        fit_started_at=fit_started_at,
    )
    session_count = len({row.session_date for row in training})
    if session_count < minimum_source_complete_sessions:
        raise FoldUnavailableError(
            "too few source-complete training sessions"
        )
    if len(training) < minimum_bundles:
        raise FoldUnavailableError("too few qualifying event/code bundles")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=3,
        max_features=30_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(
        [row.bundle_text for row in training]
    ).tocsr()
    if matrix.shape[1] == 0:
        raise FoldUnavailableError("T02 vocabulary is empty")
    raw_target = np.asarray(
        [row.open_to_close_return_pct for row in training],
        dtype=float,
    )
    lower = max(float(np.quantile(raw_target, 0.01)), -10.0)
    upper = min(float(np.quantile(raw_target, 0.99)), 10.0)
    target = np.clip(raw_target, lower, upper)
    estimator = Ridge(alpha=20.0, solver="lsqr")
    estimator.fit(matrix, target)

    row_identity = [
        {
            "session_date": row.session_date,
            "code": row.code,
        }
        for row in training
    ]
    bundle_payload = [
        {
            "session_date": row.session_date,
            "code": row.code,
            "bundle_text": row.bundle_text,
        }
        for row in training
    ]
    target_payload = [
        {
            "session_date": row.session_date,
            "code": row.code,
            "open_to_close_return_pct": row.open_to_close_return_pct,
        }
        for row in training
    ]
    source_payload = [
        {
            "session_date": row.session_date,
            "code": row.code,
            "source_complete": row.source_complete,
            "price_training_eligible": row.price_training_eligible,
            "outcome_available_at": row.outcome_available_at,
        }
        for row in training
    ]
    vocabulary = {
        str(key): int(value)
        for key, value in vectorizer.vocabulary_.items()
    }
    coef = np.asarray(estimator.coef_, dtype=float)
    intercept = float(np.asarray(estimator.intercept_).reshape(-1)[0])
    score_month = f"{score_session.year:04d}-{score_session.month:02d}"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "state": "pre_score",
        "candidate_id": T02_CANDIDATE_ID,
        "score_month": score_month,
        "fit_started_at": fit_started_at,
        "fit_completed_at": fit_completed_at,
        "training_session_first": training[0].session_date,
        "training_session_last": training[-1].session_date,
        "training_session_count": session_count,
        "training_bundle_count": len(training),
        "training_row_identity_sha256": canonical_json_sha256(
            row_identity
        ),
        "training_bundle_text_sha256": canonical_json_sha256(
            bundle_payload
        ),
        "training_target_sha256": canonical_json_sha256(target_payload),
        "training_source_payload_sha256": canonical_json_sha256(
            source_payload
        ),
        "source_manifest_sha256": source_manifest_sha256,
        "parser_sha256": parser_sha256,
        "scoring_code_sha256": scoring_code_sha256,
        "universe_config_sha256": universe_config_sha256,
        "eligibility_code_sha256": eligibility_code_sha256,
        "python_version": platform.python_version(),
        "canonical_json_contract": CANONICAL_JSON_CONTRACT,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "vectorizer_vocabulary_sha256": canonical_json_sha256(
            vocabulary
        ),
        "vectorizer_idf_sha256": _array_sha256(vectorizer.idf_),
        "ridge_coef_sha256": _array_sha256(coef),
        "ridge_intercept": intercept,
        "target_clip_lower_pct": lower,
        "target_clip_upper_pct": upper,
    }
    # The self-hash is deliberately excluded by exact field name.  No other
    # field containing "hash" is ignored.
    manifest["pre_score_fold_manifest_sha256"] = canonical_json_sha256(
        manifest,
        exclude_fields={"pre_score_fold_manifest_sha256"},
    )
    return T02MonthlyFold(
        score_month=score_month,
        vectorizer=vectorizer,
        estimator=estimator,
        manifest=manifest,
    )


def _bundles(
    disclosures: Sequence[TDnetDisclosure],
) -> tuple[tuple[str, str], ...]:
    grouped: dict[str, list[TDnetDisclosure]] = {}
    for disclosure in disclosures:
        grouped.setdefault(disclosure.code, []).append(disclosure)
    result: list[tuple[str, str]] = []
    for code in sorted(grouped):
        ordered = sorted(
            grouped[code],
            key=lambda item: (
                item.published_at,
                item.title,
                item.document_url,
            ),
        )
        result.append(
            (code, " [SEP] ".join(item.title for item in ordered))
        )
    return tuple(result)


def decision_id(
    protocol_sha256: str,
    activation_receipt_sha256: str,
    session_date: date,
) -> str:
    """Return the unique ID for one protocol/activation/session tuple."""

    _require_sha256(protocol_sha256, "protocol_sha256")
    _require_sha256(
        activation_receipt_sha256,
        "activation_receipt_sha256",
    )
    return canonical_json_sha256(
        {
            "activation_receipt_sha256": activation_receipt_sha256,
            "protocol_sha256": protocol_sha256,
            "session_date": session_date,
        }
    )


def _price_eligible_code_set(
    codes: Iterable[str] | None,
) -> tuple[str, ...]:
    if codes is None:
        raise T02ForwardError("price eligibility mask is missing")
    if isinstance(codes, (str, bytes)):
        raise T02ForwardError(
            "price eligibility mask must be an iterable of codes"
        )
    normalized = tuple(sorted(str(code).strip() for code in codes))
    if any(
        not code or re.fullmatch(r"[0-9A-Z]{4,5}", code) is None
        for code in normalized
    ):
        raise T02ForwardError(
            "price eligibility mask contains an invalid code"
        )
    if len(normalized) != len(set(normalized)):
        raise T02ForwardError(
            "price eligibility mask contains duplicate codes"
        )
    return normalized


def _decision_payload(
    *,
    protocol_sha256: str,
    activation_payload_sha256: str,
    activation_receipt_sha256: str,
    default_branch_tip_sha_at_decision: str,
    session_date: date,
    state: DecisionState,
    candidate_generation_started_at: datetime,
    decision_at: datetime,
    snapshot: LiveTDnetSnapshot | None,
    fold: T02MonthlyFold | None,
    selected_code: str | None,
    selected_score: float | None,
    failure_reason: str | None,
    price_eligibility_sha256: str | None,
    qualifying_event_count: int,
    live_source_snapshot_sha256: str | None,
) -> dict[str, Any]:
    source_manifest = (
        snapshot.source_manifest_sha256 if snapshot is not None else None
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "session_date": session_date,
        "decision_id": decision_id(
            protocol_sha256,
            activation_receipt_sha256,
            session_date,
        ),
        "candidate_id": T02_CANDIDATE_ID,
        "python_version": platform.python_version(),
        "canonical_json_contract": CANONICAL_JSON_CONTRACT,
        "protocol_sha256": protocol_sha256,
        "activation_payload_sha256": activation_payload_sha256,
        "activation_receipt_sha256": activation_receipt_sha256,
        "default_branch_tip_sha_at_decision": (
            default_branch_tip_sha_at_decision
        ),
        "pre_score_fold_manifest_sha256": (
            fold.manifest_sha256 if fold is not None else None
        ),
        "source_manifest_sha256": source_manifest,
        "live_source_snapshot_sha256": live_source_snapshot_sha256,
        "price_eligibility_sha256": price_eligibility_sha256,
        "candidate_generation_started_at": candidate_generation_started_at,
        "decision_at": decision_at,
        "published_at_max": (
            max(
                (
                    item.published_at
                    for item in snapshot.disclosures
                ),
                default=None,
            )
            if snapshot is not None
            else None
        ),
        "received_at_max": (
            snapshot.source_received_at if snapshot is not None else None
        ),
        "computed_at": snapshot.computed_at if snapshot is not None else None,
        "source_complete": bool(
            snapshot is not None and snapshot.source_complete
        ),
        "qualifying_event_count": qualifying_event_count,
        "selected_code": selected_code,
        "selected_score": selected_score,
        "decision": state.value,
        "failure_reason": failure_reason,
    }
    payload["decision_payload_sha256"] = canonical_json_sha256(
        payload,
        exclude_fields={"decision_payload_sha256"},
    )
    return payload


def make_t02_decision(
    *,
    protocol_sha256: str,
    activation_payload_sha256: str,
    activation_receipt_sha256: str,
    default_branch_tip_sha_at_decision: str,
    session_date: date,
    candidate_generation_started_at: datetime,
    decision_at: datetime,
    snapshot: LiveTDnetSnapshot | None,
    fold: T02MonthlyFold | None,
    price_eligible_codes: Iterable[str] | None,
) -> dict[str, Any]:
    """Create one deterministic T02 shadow decision.

    Invalid/missing source data and unavailable model folds become distinct
    fail-closed states.  A post-cutoff decision timestamp itself is rejected:
    it cannot be made safe by relabeling it as a source failure.
    """

    _require_sha256(protocol_sha256, "protocol_sha256")
    _require_sha256(
        activation_payload_sha256,
        "activation_payload_sha256",
    )
    _require_sha256(
        activation_receipt_sha256,
        "activation_receipt_sha256",
    )
    _require_sha256(
        default_branch_tip_sha_at_decision,
        "default_branch_tip_sha_at_decision",
    )
    require_jst_timestamp(
        candidate_generation_started_at,
        "candidate_generation_started_at",
        session_date=session_date,
        at_or_before_cutoff=True,
    )
    require_jst_timestamp(
        decision_at,
        "decision_at",
        session_date=session_date,
        at_or_before_cutoff=True,
    )
    if decision_at < candidate_generation_started_at:
        raise T02ForwardError(
            "decision_at cannot precede candidate generation"
        )

    if snapshot is None:
        return _decision_payload(
            protocol_sha256=protocol_sha256,
            activation_payload_sha256=activation_payload_sha256,
            activation_receipt_sha256=activation_receipt_sha256,
            default_branch_tip_sha_at_decision=(
                default_branch_tip_sha_at_decision
            ),
            session_date=session_date,
            state=DecisionState.FAIL_CLOSED_SOURCE,
            candidate_generation_started_at=candidate_generation_started_at,
            decision_at=decision_at,
            snapshot=None,
            fold=fold,
            selected_code=None,
            selected_score=None,
            failure_reason="source_snapshot_missing",
            price_eligibility_sha256=None,
            qualifying_event_count=0,
            live_source_snapshot_sha256=None,
        )
    try:
        validate_live_snapshot(snapshot)
        if snapshot.session_date != session_date:
            raise T02ForwardError("snapshot session date differs")
        if not snapshot.source_complete:
            raise T02ForwardError("source snapshot is incomplete")
        snapshot_sha = live_snapshot_sha256(snapshot)
    except (DataValidationError, LeakageError) as exc:
        return _decision_payload(
            protocol_sha256=protocol_sha256,
            activation_payload_sha256=activation_payload_sha256,
            activation_receipt_sha256=activation_receipt_sha256,
            default_branch_tip_sha_at_decision=(
                default_branch_tip_sha_at_decision
            ),
            session_date=session_date,
            state=DecisionState.FAIL_CLOSED_SOURCE,
            candidate_generation_started_at=candidate_generation_started_at,
            decision_at=decision_at,
            snapshot=snapshot,
            fold=fold,
            selected_code=None,
            selected_score=None,
            failure_reason=f"{type(exc).__name__}:{exc}",
            price_eligibility_sha256=None,
            qualifying_event_count=0,
            live_source_snapshot_sha256=None,
        )

    try:
        eligible_codes = _price_eligible_code_set(
            price_eligible_codes
        )
        eligibility_sha = canonical_json_sha256(
            {
                "session_date": session_date,
                "price_eligible_codes": eligible_codes,
            }
        )
    except DataValidationError as exc:
        return _decision_payload(
            protocol_sha256=protocol_sha256,
            activation_payload_sha256=activation_payload_sha256,
            activation_receipt_sha256=activation_receipt_sha256,
            default_branch_tip_sha_at_decision=(
                default_branch_tip_sha_at_decision
            ),
            session_date=session_date,
            state=DecisionState.FAIL_CLOSED_MODEL,
            candidate_generation_started_at=candidate_generation_started_at,
            decision_at=decision_at,
            snapshot=snapshot,
            fold=fold,
            selected_code=None,
            selected_score=None,
            failure_reason=f"{type(exc).__name__}:{exc}",
            price_eligibility_sha256=None,
            qualifying_event_count=0,
            live_source_snapshot_sha256=snapshot_sha,
        )

    eligible = set(eligible_codes)
    bundles = tuple(
        (code, text)
        for code, text in _bundles(snapshot.disclosures)
        if code in eligible
    )
    if not bundles:
        return _decision_payload(
            protocol_sha256=protocol_sha256,
            activation_payload_sha256=activation_payload_sha256,
            activation_receipt_sha256=activation_receipt_sha256,
            default_branch_tip_sha_at_decision=(
                default_branch_tip_sha_at_decision
            ),
            session_date=session_date,
            state=DecisionState.CASH_NO_EVENT,
            candidate_generation_started_at=candidate_generation_started_at,
            decision_at=decision_at,
            snapshot=snapshot,
            fold=fold,
            selected_code=None,
            selected_score=None,
            failure_reason=None,
            price_eligibility_sha256=eligibility_sha,
            qualifying_event_count=0,
            live_source_snapshot_sha256=snapshot_sha,
        )

    expected_month = f"{session_date.year:04d}-{session_date.month:02d}"
    if (
        fold is None
        or fold.score_month != expected_month
        or not _manifest_is_exact(fold.manifest)
    ):
        return _decision_payload(
            protocol_sha256=protocol_sha256,
            activation_payload_sha256=activation_payload_sha256,
            activation_receipt_sha256=activation_receipt_sha256,
            default_branch_tip_sha_at_decision=(
                default_branch_tip_sha_at_decision
            ),
            session_date=session_date,
            state=DecisionState.FAIL_CLOSED_MODEL,
            candidate_generation_started_at=candidate_generation_started_at,
            decision_at=decision_at,
            snapshot=snapshot,
            fold=fold,
            selected_code=None,
            selected_score=None,
            failure_reason="monthly_fold_unavailable_or_invalid",
            price_eligibility_sha256=eligibility_sha,
            qualifying_event_count=len(bundles),
            live_source_snapshot_sha256=snapshot_sha,
        )

    try:
        codes = [code for code, _ in bundles]
        texts = [text for _, text in bundles]
        matrix = fold.vectorizer.transform(texts)
        scores = np.asarray(fold.estimator.predict(matrix), dtype=float)
        if scores.shape != (len(codes),) or not np.isfinite(scores).all():
            raise T02ForwardError("model returned invalid scores")
    except Exception as exc:
        return _decision_payload(
            protocol_sha256=protocol_sha256,
            activation_payload_sha256=activation_payload_sha256,
            activation_receipt_sha256=activation_receipt_sha256,
            default_branch_tip_sha_at_decision=(
                default_branch_tip_sha_at_decision
            ),
            session_date=session_date,
            state=DecisionState.FAIL_CLOSED_MODEL,
            candidate_generation_started_at=candidate_generation_started_at,
            decision_at=decision_at,
            snapshot=snapshot,
            fold=fold,
            selected_code=None,
            selected_score=None,
            failure_reason=f"{type(exc).__name__}:{exc}",
            price_eligibility_sha256=eligibility_sha,
            qualifying_event_count=len(bundles),
            live_source_snapshot_sha256=snapshot_sha,
        )

    winner = min(
        zip(codes, scores, strict=True),
        key=lambda item: (-float(item[1]), item[0]),
    )
    return _decision_payload(
        protocol_sha256=protocol_sha256,
        activation_payload_sha256=activation_payload_sha256,
        activation_receipt_sha256=activation_receipt_sha256,
        default_branch_tip_sha_at_decision=(
            default_branch_tip_sha_at_decision
        ),
        session_date=session_date,
        state=DecisionState.SELECTED,
        candidate_generation_started_at=candidate_generation_started_at,
        decision_at=decision_at,
        snapshot=snapshot,
        fold=fold,
        selected_code=winner[0],
        selected_score=float(winner[1]),
        failure_reason=None,
        price_eligibility_sha256=eligibility_sha,
        qualifying_event_count=len(bundles),
        live_source_snapshot_sha256=snapshot_sha,
    )


def _validate_decision_payload(payload: Mapping[str, Any]) -> str:
    required = {
        "session_date",
        "decision_id",
        "python_version",
        "canonical_json_contract",
        "protocol_sha256",
        "activation_payload_sha256",
        "activation_receipt_sha256",
        "default_branch_tip_sha_at_decision",
        "pre_score_fold_manifest_sha256",
        "live_source_snapshot_sha256",
        "candidate_generation_started_at",
        "decision_at",
        "published_at_max",
        "received_at_max",
        "computed_at",
        "source_complete",
        "qualifying_event_count",
        "selected_code",
        "selected_score",
        "decision",
        "decision_payload_sha256",
    }
    missing = required - set(payload)
    if missing:
        raise T02ForwardError(
            f"decision payload is missing fields: {sorted(missing)}"
        )
    if not str(payload["python_version"]).strip():
        raise T02ForwardError("python_version must be non-empty")
    if payload["canonical_json_contract"] != CANONICAL_JSON_CONTRACT:
        raise T02ForwardError("unknown canonical_json_contract")
    protocol_sha = _require_sha256(
        str(payload["protocol_sha256"]),
        "protocol_sha256",
    )
    _require_sha256(
        str(payload["activation_payload_sha256"]),
        "activation_payload_sha256",
    )
    activation_receipt_sha = _require_sha256(
        str(payload["activation_receipt_sha256"]),
        "activation_receipt_sha256",
    )
    _require_sha256(
        str(payload["default_branch_tip_sha_at_decision"]),
        "default_branch_tip_sha_at_decision",
    )
    for field in (
        "pre_score_fold_manifest_sha256",
        "live_source_snapshot_sha256",
    ):
        if payload[field] is not None:
            _require_sha256(str(payload[field]), field)
    session_value = payload["session_date"]
    if isinstance(session_value, str):
        try:
            session_value = date.fromisoformat(session_value)
        except ValueError as exc:
            raise T02ForwardError("invalid decision session_date") from exc
    if not isinstance(session_value, date) or isinstance(
        session_value, datetime
    ):
        raise T02ForwardError("decision session_date must be a date")
    expected_id = decision_id(
        protocol_sha,
        activation_receipt_sha,
        session_value,
    )
    if payload["decision_id"] != expected_id:
        raise T02ForwardError("decision_id is not protocol/session exact")
    try:
        DecisionState(str(payload["decision"]))
    except ValueError as exc:
        raise T02ForwardError("unknown decision state") from exc
    recorded = str(payload["decision_payload_sha256"])
    expected = canonical_json_sha256(
        payload,
        exclude_fields={
            "decision_payload_sha256",
            "sequence_number",
            "previous_record_sha256",
            "record_sha256",
        },
    )
    if recorded != expected:
        raise T02ForwardError("decision payload SHA-256 mismatch")
    return recorded


class DecisionHashChainLedger:
    """In-memory append-only decision ledger with deterministic hash chaining."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}

    @property
    def entries(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(entry) for entry in self._entries)

    @property
    def head_sha256(self) -> str | None:
        if not self._entries:
            return None
        return str(self._entries[-1]["record_sha256"])

    def append(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Append once; exact retries are idempotent and conflicts fail."""

        payload_copy = dict(payload)
        payload_hash = _validate_decision_payload(payload_copy)
        identifier = str(payload_copy["decision_id"])
        prior = self._by_id.get(identifier)
        if prior is not None:
            if prior["decision_payload_sha256"] != payload_hash:
                raise T02ForwardError(
                    "conflicting payload for protocol/session decision"
                )
            comparable_prior = {
                key: value
                for key, value in prior.items()
                if key
                not in {
                    "sequence_number",
                    "previous_record_sha256",
                    "record_sha256",
                }
            }
            comparable_incoming = {
                key: value
                for key, value in payload_copy.items()
                if key
                not in {
                    "sequence_number",
                    "previous_record_sha256",
                    "record_sha256",
                }
            }
            if canonical_json_bytes(comparable_prior) != canonical_json_bytes(
                comparable_incoming
            ):
                raise T02ForwardError(
                    "conflicting payload for protocol/session decision"
                )
            return dict(prior)

        session_value = payload_copy["session_date"]
        session_date_value = (
            date.fromisoformat(session_value)
            if isinstance(session_value, str)
            else session_value
        )
        if self._entries:
            last_session = self._entries[-1]["session_date"]
            last_session_value = (
                date.fromisoformat(last_session)
                if isinstance(last_session, str)
                else last_session
            )
            if session_date_value <= last_session_value:
                raise T02ForwardError(
                    "new ledger decisions must be appended chronologically"
                )

        sequence = len(self._entries)
        previous = self.head_sha256 or ("0" * 64)
        entry = dict(payload_copy)
        for field in (
            "sequence_number",
            "previous_record_sha256",
            "record_sha256",
        ):
            entry.pop(field, None)
        entry["sequence_number"] = sequence
        entry["previous_record_sha256"] = previous
        entry["record_sha256"] = canonical_json_sha256(
            entry,
            exclude_fields={"record_sha256"},
        )
        self._entries.append(entry)
        self._by_id[identifier] = entry
        return dict(entry)
