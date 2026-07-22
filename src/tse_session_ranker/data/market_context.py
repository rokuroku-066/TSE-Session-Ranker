"""Append-only, point-in-time market context for forward shadow research."""

from __future__ import annotations

from datetime import time as clock_time

import numpy as np
import pandas as pd

from ..exceptions import DataValidationError, LeakageError


DEFAULT_DECISION_TIME = "08:58:59"
REQUIRED_MARKET_CONTEXT_COLUMNS: tuple[str, ...] = (
    "date",
    "observed_at",
    "nikkei_return_pct",
    "topix_return_pct",
    "return_definition",
)
OPTIONAL_MARKET_CONTEXT_COLUMNS: tuple[str, ...] = (
    "nikkei_level",
    "topix_level",
    "source",
)
MARKET_CONTEXT_COLUMNS: tuple[str, ...] = (
    *REQUIRED_MARKET_CONTEXT_COLUMNS,
    *OPTIONAL_MARKET_CONTEXT_COLUMNS,
)


def _target_date(value: object) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("market context contains an invalid date") from exc
    if pd.isna(timestamp):
        raise DataValidationError("market context contains an invalid date")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Tokyo").tz_localize(None)
    return timestamp.normalize()


def _observed_at(value: object) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            "market context contains an invalid observed_at"
        ) from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise LeakageError("market context observed_at must be timezone-aware")
    return timestamp.tz_convert("Asia/Tokyo")


def _decision_clock(decision_time: str) -> clock_time:
    try:
        value = clock_time.fromisoformat(decision_time)
    except ValueError as exc:
        raise ValueError("decision_time must be an ISO local time") from exc
    if value.tzinfo is not None:
        raise ValueError("decision_time must not contain a timezone offset")
    return value


def _normalise_optional_text(value: object) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def _assert_one_return_definition(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    definitions = frame["return_definition"].drop_duplicates()
    if len(definitions) != 1:
        raise DataValidationError(
            "market context cannot mix return_definition values"
        )


def _deduplicate_exact_dates(frame: pd.DataFrame) -> pd.DataFrame:
    duplicated = frame.duplicated("date", keep=False)
    if not duplicated.any():
        return frame
    conflicts: list[str] = []
    compare = [column for column in MARKET_CONTEXT_COLUMNS if column != "date"]
    for date, group in frame.loc[duplicated].groupby("date", sort=False):
        if any(group[column].nunique(dropna=False) != 1 for column in compare):
            conflicts.append(str(pd.Timestamp(date).date()))
            if len(conflicts) == 5:
                break
    if conflicts:
        raise DataValidationError(
            "conflicting market context rows for target date: "
            + ", ".join(conflicts)
        )
    return frame.drop_duplicates("date", keep="first")


def normalize_market_context(
    context: pd.DataFrame,
    decision_time: str = DEFAULT_DECISION_TIME,
) -> pd.DataFrame:
    """Validate and canonicalise exact-date 08:58 market observations.

    No value is carried to another date.  The return horizon is explicit in
    ``return_definition`` and one frame may contain only one such definition.
    """

    missing = sorted(set(REQUIRED_MARKET_CONTEXT_COLUMNS) - set(context.columns))
    if missing:
        raise DataValidationError(
            f"market context is missing columns: {missing}"
        )
    cutoff_time = _decision_clock(decision_time)
    frame = context.copy().reset_index(drop=True)
    for column in OPTIONAL_MARKET_CONTEXT_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[list(MARKET_CONTEXT_COLUMNS)].copy()
    if frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])
        return frame

    frame["date"] = frame["date"].map(_target_date)
    frame["observed_at"] = frame["observed_at"].map(_observed_at)
    frame["return_definition"] = frame["return_definition"].map(
        _normalise_optional_text
    )
    if frame["return_definition"].isna().any():
        raise DataValidationError("return_definition must be non-empty")
    frame["source"] = frame["source"].map(_normalise_optional_text)

    for column in (
        "nikkei_return_pct",
        "topix_return_pct",
        "nikkei_level",
        "topix_level",
    ):
        original_nonmissing = frame[column].notna()
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = original_nonmissing & values.isna()
        if invalid.any() or np.isinf(values.dropna()).any():
            raise DataValidationError(f"market context has invalid {column}")
        frame[column] = values
    if frame[["nikkei_return_pct", "topix_return_pct"]].isna().any(axis=None):
        raise DataValidationError("market context returns must be finite")

    for row in frame[["date", "observed_at"]].itertuples(index=False):
        date = pd.Timestamp(row.date)
        observed = pd.Timestamp(row.observed_at)
        if observed.date() != date.date():
            raise LeakageError(
                "market context observed_at local date must equal target date"
            )
        cutoff = pd.Timestamp.combine(date.date(), cutoff_time).tz_localize(
            "Asia/Tokyo"
        )
        if observed > cutoff:
            raise LeakageError(
                f"market context observation {observed.isoformat()} is after "
                f"cutoff {cutoff.isoformat()}"
            )

    _assert_one_return_definition(frame)
    frame = _deduplicate_exact_dates(frame)
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def append_market_context(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    decision_time: str = DEFAULT_DECISION_TIME,
) -> pd.DataFrame:
    """Append new forward observations without changing canonical past rows."""

    if existing.empty:
        return normalize_market_context(new, decision_time=decision_time)
    current = normalize_market_context(existing, decision_time=decision_time)
    if new.empty:
        return current.copy()
    incoming = normalize_market_context(new, decision_time=decision_time)
    if incoming.empty:
        return current.copy()

    current_definition = str(current.iloc[0]["return_definition"])
    incoming_definition = str(incoming.iloc[0]["return_definition"])
    if current_definition != incoming_definition:
        raise DataValidationError(
            "cannot append a different return_definition to market context"
        )

    current_dates = set(current["date"])
    overlap = incoming[incoming["date"].isin(current_dates)]
    if not overlap.empty:
        check = pd.concat(
            [current[current["date"].isin(overlap["date"])], overlap],
            ignore_index=True,
        )
        _deduplicate_exact_dates(check)

    novel = incoming[~incoming["date"].isin(current_dates)].copy()
    if novel.empty:
        return current.copy()
    latest_existing = current["date"].max()
    retroactive = novel["date"].le(latest_existing)
    if retroactive.any():
        examples = ", ".join(
            str(value.date()) for value in novel.loc[retroactive, "date"].head(5)
        )
        raise DataValidationError(
            "append-only market context rejects retroactive target dates: "
            + examples
        )
    result = pd.concat([current, novel], ignore_index=True)
    return normalize_market_context(result, decision_time=decision_time)
