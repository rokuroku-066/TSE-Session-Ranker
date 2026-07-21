from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ..exceptions import DataValidationError, LeakageError
from .common import normalize_code


REQUIRED_PREOPEN = ("observed_at", "code", "indicative_price")
NUMERIC_PREOPEN = (
    "indicative_price",
    "prior_close",
    "buy_market_qty",
    "sell_market_qty",
    "pts_price",
    "pts_volume",
)


def _aware_timestamp(value: object, timezone: str = "Asia/Tokyo") -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise LeakageError("pre-open timestamps and as_of must include a timezone")
    return timestamp.tz_convert(timezone)


def _boolean(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"1", "true", "yes", "y", "買", "buy"}:
            return True
        if normalised in {"0", "false", "no", "n", "", "none"}:
            return False
        raise DataValidationError(f"invalid boolean value: {value}")
    return bool(value)


def normalize_preopen_snapshots(
    snapshots: pd.DataFrame, timezone: str = "Asia/Tokyo"
) -> pd.DataFrame:
    missing = [column for column in REQUIRED_PREOPEN if column not in snapshots]
    if missing:
        raise DataValidationError(f"pre-open data is missing columns: {missing}")
    frame = snapshots.copy()
    frame["observed_at"] = frame["observed_at"].map(
        lambda value: _aware_timestamp(value, timezone)
    )
    frame["code"] = frame["code"].map(normalize_code)
    if frame["code"].eq("").any():
        raise DataValidationError("pre-open data contains an empty security code")
    for column in NUMERIC_PREOPEN:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("indicative_price", "prior_close", "pts_price"):
        invalid = frame[column].notna() & frame[column].le(0)
        if invalid.any():
            raise DataValidationError(f"{column} must be positive when supplied")
    for column in ("buy_market_qty", "sell_market_qty", "pts_volume"):
        invalid = frame[column].notna() & frame[column].lt(0)
        if invalid.any():
            raise DataValidationError(f"{column} must not be negative")
    for column in ("buy_special", "sell_special"):
        if column not in frame:
            frame[column] = False
        frame[column] = frame[column].map(_boolean)
    if "target_date" in frame:
        frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce").dt.date
        if frame["target_date"].isna().any():
            raise DataValidationError("pre-open data contains an invalid target_date")
    else:
        frame["target_date"] = frame["observed_at"].map(lambda value: value.date())
    if "expected_open_at" not in frame:
        frame["expected_open_at"] = pd.NA
    keys = ["target_date", "code", "observed_at"]
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        compare = [column for column in NUMERIC_PREOPEN + ("buy_special", "sell_special", "expected_open_at")]
        for key, group in frame.loc[duplicated].groupby(keys, sort=False):
            if any(group[column].nunique(dropna=False) > 1 for column in compare):
                raise DataValidationError(f"conflicting pre-open snapshots for {key}")
        frame = frame.drop_duplicates(keys, keep="last")
    return frame.sort_values(["target_date", "code", "observed_at"]).reset_index(drop=True)


def latest_preopen_snapshots(
    snapshots: pd.DataFrame,
    target_date: object,
    as_of: datetime | pd.Timestamp | str,
    timezone: str = "Asia/Tokyo",
) -> pd.DataFrame:
    frame = normalize_preopen_snapshots(snapshots, timezone)
    cutoff = _aware_timestamp(as_of, timezone)
    date_value = pd.Timestamp(target_date).date()
    if cutoff.date() != date_value:
        raise LeakageError("as_of and target_date must refer to the same local date")
    eligible = frame[
        frame["target_date"].eq(date_value) & frame["observed_at"].le(cutoff)
    ].copy()
    if eligible.empty:
        return eligible
    return (
        eligible.sort_values(["code", "observed_at"], kind="stable")
        .groupby("code", as_index=False, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
