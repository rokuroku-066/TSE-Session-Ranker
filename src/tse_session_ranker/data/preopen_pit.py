"""Point-in-time validation primitives for prospective pre-open data.

The module deliberately models *availability* separately from an exchange or
publisher event timestamp. A record is usable only when it had been received
and durably computed by the decision cutoff.

This module defines the canonical, source-neutral ledger contract. Acquisition
adapters should preserve their raw payloads and normalize each event or
snapshot into :class:`PointInTimeRecord`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from ..exceptions import DataValidationError, LeakageError


TOKYO = ZoneInfo("Asia/Tokyo")
DEFAULT_DECISION_TIME = time(8, 58, 59)


class SchemaError(DataValidationError):
    """Raised when a pre-open PIT record violates its semantic contract."""


class FeatureGroup(str, Enum):
    FUTURES = "futures"
    AUCTION = "auction"
    PTS = "pts"
    LIQUIDITY = "liquidity"
    TDNET = "tdnet"


class CoverageStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    NOT_SUPPORTED = "not_supported"


class ActivityStatus(str, Enum):
    ACTIVE = "active"
    NO_ACTIVITY = "no_activity"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class FieldStatus(str, Enum):
    OBSERVED = "observed"
    ZERO = "zero"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class FieldRule:
    """Validation rule for one measurement."""

    unit: str
    positive: bool = False
    non_negative: bool = False


GROUP_FIELD_RULES: Mapping[FeatureGroup, Mapping[str, FieldRule]] = {
    FeatureGroup.FUTURES: {
        "last_price": FieldRule("index_points", positive=True),
        "best_bid": FieldRule("index_points", positive=True),
        "best_ask": FieldRule("index_points", positive=True),
        "session_open": FieldRule("index_points", positive=True),
        "cumulative_volume": FieldRule("contracts", non_negative=True),
        "days_to_expiry": FieldRule("calendar_days", non_negative=True),
    },
    FeatureGroup.AUCTION: {
        "indicative_equilibrium_price": FieldRule("JPY", positive=True),
        "bid_qty_l1": FieldRule("shares", non_negative=True),
        "ask_qty_l1": FieldRule("shares", non_negative=True),
        "bid_qty_l3": FieldRule("shares", non_negative=True),
        "ask_qty_l3": FieldRule("shares", non_negative=True),
        "bid_qty_l10": FieldRule("shares", non_negative=True),
        "ask_qty_l10": FieldRule("shares", non_negative=True),
        "buy_market_qty": FieldRule("shares", non_negative=True),
        "sell_market_qty": FieldRule("shares", non_negative=True),
    },
    FeatureGroup.PTS: {
        "last_price": FieldRule("JPY", positive=True),
        "best_bid": FieldRule("JPY", positive=True),
        "best_ask": FieldRule("JPY", positive=True),
        "volume": FieldRule("shares", non_negative=True),
        "turnover": FieldRule("JPY", non_negative=True),
        "trade_count": FieldRule("trades", non_negative=True),
    },
    FeatureGroup.LIQUIDITY: {
        "prior_close": FieldRule("JPY", positive=True),
        "prior_volume": FieldRule("shares", non_negative=True),
        "prior_turnover": FieldRule("JPY", non_negative=True),
        "median_turnover_20": FieldRule("JPY", non_negative=True),
        "trading_unit": FieldRule("shares", positive=True),
        "tick_size": FieldRule("JPY", positive=True),
        "amihud_20": FieldRule(
            "absolute_return_per_JPY",
            non_negative=True,
        ),
        "high_low_proxy_20": FieldRule("ratio", non_negative=True),
    },
    FeatureGroup.TDNET: {
        "document_count": FieldRule("documents", non_negative=True),
        "forecast_sales_revision_pct": FieldRule("percent"),
        "forecast_operating_revision_pct": FieldRule("percent"),
        "forecast_ordinary_revision_pct": FieldRule("percent"),
        "forecast_net_revision_pct": FieldRule("percent"),
        "dividend_change_yield_pct": FieldRule("percent"),
        "buyback_shares_pct": FieldRule("percent", non_negative=True),
        "buyback_value_market_cap_pct": FieldRule(
            "percent",
            non_negative=True,
        ),
    },
}


def decision_cutoff(
    target_date: date,
    *,
    decision_time: time = DEFAULT_DECISION_TIME,
) -> datetime:
    """Return the Tokyo-aware model cutoff for ``target_date``."""

    if decision_time.tzinfo is not None:
        raise SchemaError("decision_time must be a naive wall-clock time")
    return datetime.combine(target_date, decision_time, tzinfo=TOKYO)


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SchemaError(f"{name} must be timezone-aware")


def _is_numeric_zero(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
        and float(value) == 0.0
    )


@dataclass(frozen=True)
class PointInTimeRecord:
    """One source snapshot or event available to a pre-open decision."""

    target_date: date
    instrument_id: str
    feature_group: FeatureGroup
    venue: str
    session: str
    source: str
    source_record_id: str
    source_event_at: datetime | None
    source_received_at: datetime
    computed_at: datetime
    coverage_status: CoverageStatus
    activity_status: ActivityStatus
    values: Mapping[str, Any]
    field_status: Mapping[str, FieldStatus]
    source_version: str = ""

    @property
    def available_at(self) -> datetime:
        """Return when the fully computed record first became usable."""

        return max(self.source_received_at, self.computed_at)

    @property
    def identity(self) -> tuple[Any, ...]:
        """Return the immutable append-ledger identity."""

        return (
            self.target_date,
            self.instrument_id,
            self.feature_group.value,
            self.venue,
            self.session,
            self.source,
            self.source_record_id,
        )


def validate_record(
    record: PointInTimeRecord,
    *,
    cutoff: datetime | None = None,
    max_age: timedelta | None = None,
) -> PointInTimeRecord:
    """Validate point-in-time availability and missing/zero semantics.

    ``source_event_at`` is the exchange or publisher timestamp. It is never
    used as a substitute for local receipt and computation time.
    """

    cutoff = cutoff or decision_cutoff(record.target_date)
    _require_aware(cutoff, "cutoff")
    _require_aware(record.source_received_at, "source_received_at")
    _require_aware(record.computed_at, "computed_at")
    if record.source_event_at is not None:
        _require_aware(record.source_event_at, "source_event_at")

    if not isinstance(record.feature_group, FeatureGroup):
        raise SchemaError("feature_group must be a FeatureGroup")
    if not isinstance(record.coverage_status, CoverageStatus):
        raise SchemaError("coverage_status must be a CoverageStatus")
    if not isinstance(record.activity_status, ActivityStatus):
        raise SchemaError("activity_status must be an ActivityStatus")
    if cutoff.astimezone(TOKYO).date() != record.target_date:
        raise SchemaError("cutoff Tokyo date must equal target_date")
    if not record.instrument_id.strip():
        raise SchemaError("instrument_id must be non-empty")
    if not record.venue.strip() or not record.session.strip():
        raise SchemaError("venue and session must be non-empty")
    if not record.source.strip() or not record.source_record_id.strip():
        raise SchemaError("source and source_record_id must be non-empty")
    if record.source_event_at is not None:
        # Production feeds with documented clock skew should normalize clocks
        # before constructing the canonical record.
        if record.source_event_at > record.source_received_at:
            raise SchemaError(
                "source_event_at cannot be after source_received_at"
            )
    if record.computed_at < record.source_received_at:
        raise SchemaError("computed_at cannot be before source_received_at")
    if record.available_at > cutoff:
        raise LeakageError(
            f"record available at {record.available_at.isoformat()}, "
            f"after cutoff {cutoff.isoformat()}"
        )
    if max_age is not None and cutoff - record.available_at > max_age:
        raise SchemaError(
            "record is older than the configured freshness window"
        )

    if set(record.values) != set(record.field_status):
        raise SchemaError("values and field_status must have identical keys")

    if (
        record.coverage_status != CoverageStatus.COMPLETE
        and record.activity_status == ActivityStatus.NO_ACTIVITY
    ):
        raise SchemaError(
            "no_activity may be asserted only when source coverage is complete"
        )

    rules = GROUP_FIELD_RULES[record.feature_group]
    unknown = set(record.values) - set(rules)
    if unknown:
        raise SchemaError(
            f"unknown fields for {record.feature_group.value}: "
            f"{sorted(unknown)}"
        )

    for field_name, value in record.values.items():
        try:
            status = FieldStatus(record.field_status[field_name])
        except ValueError as exc:
            raise SchemaError(
                f"{field_name}: invalid field status"
            ) from exc
        rule = rules[field_name]

        if status in {
            FieldStatus.MISSING,
            FieldStatus.NOT_APPLICABLE,
        }:
            if value is not None:
                raise SchemaError(
                    f"{field_name}: {status.value} requires null"
                )
            continue
        if status == FieldStatus.ZERO:
            if not _is_numeric_zero(value):
                raise SchemaError(
                    f"{field_name}: zero requires numeric 0"
                )
            if rule.positive:
                raise SchemaError(
                    f"{field_name}: price/positive field cannot be zero"
                )
            continue
        if value is None:
            raise SchemaError(
                f"{field_name}: observed requires a non-null value"
            )
        if _is_numeric_zero(value):
            raise SchemaError(
                f"{field_name}: numeric zero must use status=zero"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaError(
                f"{field_name}: measurements must be numeric"
            )
        numeric = float(value)
        if not isfinite(numeric):
            raise SchemaError(f"{field_name}: value must be finite")
        if rule.positive and numeric <= 0:
            raise SchemaError(f"{field_name}: value must be positive")
        if rule.non_negative and numeric < 0:
            raise SchemaError(
                f"{field_name}: value must be non-negative"
            )

    return record


def validate_append_batch(
    existing: Sequence[PointInTimeRecord],
    incoming: Sequence[PointInTimeRecord],
) -> tuple[PointInTimeRecord, ...]:
    """Validate immutable append semantics and conflicting duplicates."""

    by_identity = {item.identity: item for item in existing}
    output = list(existing)
    for item in incoming:
        prior = by_identity.get(item.identity)
        if prior is not None:
            if prior != item:
                raise SchemaError(
                    f"conflicting duplicate: {item.identity}"
                )
            continue
        by_identity[item.identity] = item
        output.append(item)
    return tuple(output)


def latest_by_source(
    records: Iterable[PointInTimeRecord],
    *,
    cutoff: datetime,
) -> tuple[PointInTimeRecord, ...]:
    """Return the latest usable record per source slice.

    The function validates rather than silently dropping post-cutoff records.
    Callers should construct their candidate set from the prospective ledger,
    not from a retrospectively complete table.
    """

    selected: dict[
        tuple[str, str, str, str, str],
        PointInTimeRecord,
    ] = {}
    for record in records:
        validate_record(record, cutoff=cutoff)
        key = (
            record.instrument_id,
            record.feature_group.value,
            record.venue,
            record.session,
            record.source,
        )
        prior = selected.get(key)
        if prior is None or record.available_at > prior.available_at:
            selected[key] = record
    return tuple(selected[key] for key in sorted(selected))


def safe_imbalance(
    buy_quantity: float | None,
    sell_quantity: float | None,
) -> float | None:
    """Return ``(buy-sell)/(buy+sell)`` without inventing undefined states."""

    if buy_quantity is None or sell_quantity is None:
        return None
    if buy_quantity < 0 or sell_quantity < 0:
        raise SchemaError("quantities cannot be negative")
    total = buy_quantity + sell_quantity
    if total == 0:
        return None
    return (buy_quantity - sell_quantity) / total
