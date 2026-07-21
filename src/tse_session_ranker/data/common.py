from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..exceptions import DataValidationError


OHLC = ("open", "high", "low", "close")
REQUIRED_DAILY_COLUMNS = ("date", "code", *OHLC)


def normalize_code(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _normalise_bool(value: object, default: bool = False) -> bool:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"1", "true", "yes", "y"}:
            return True
        if normalised in {"0", "false", "no", "n", ""}:
            return False
        raise DataValidationError(f"invalid boolean value: {value}")
    return bool(value)


def _deduplicate_or_raise(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["code", "date"]
    duplicated = frame.duplicated(keys, keep=False)
    if not duplicated.any():
        return frame
    compare = [
        column
        for column in ("name", *OHLC, "volume", "turnover", "traded", "partial_session")
        if column in frame.columns
    ]
    conflicts: list[str] = []
    for key, group in frame.loc[duplicated].groupby(keys, sort=False, dropna=False):
        if any(group[column].nunique(dropna=False) > 1 for column in compare):
            conflicts.append(f"{key[0]}@{pd.Timestamp(key[1]).date()}")
            if len(conflicts) == 5:
                break
    if conflicts:
        raise DataValidationError(
            "conflicting duplicate daily rows: " + ", ".join(conflicts)
        )
    return frame.drop_duplicates(keys, keep="last")


def normalize_daily_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalise daily TSE quotations.

    A no-trade session is represented by ``traded=False`` and four missing OHLC
    values.  Keeping that row prevents a 20-session window from silently
    becoming a 20-trade window.
    """

    missing = [column for column in REQUIRED_DAILY_COLUMNS if column not in prices]
    if missing:
        raise DataValidationError(f"daily data is missing columns: {missing}")
    frame = prices.copy()
    parsed_dates = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    if parsed_dates.isna().any():
        raise DataValidationError("daily data contains invalid dates")
    if getattr(parsed_dates.dt, "tz", None) is not None:
        parsed_dates = parsed_dates.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    frame["date"] = parsed_dates.dt.normalize()
    frame["code"] = frame["code"].map(normalize_code)
    if frame["code"].eq("").any():
        raise DataValidationError("daily data contains an empty security code")
    frame["name"] = frame.get("name", frame["code"]).fillna(frame["code"]).astype(str)
    for column in (*OHLC, "volume", "turnover"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    inferred_traded = frame[list(OHLC)].notna().all(axis=1)
    if "traded" in frame:
        frame["traded"] = [
            _normalise_bool(value, bool(inferred))
            for value, inferred in zip(frame["traded"], inferred_traded, strict=True)
        ]
    else:
        frame["traded"] = inferred_traded
    if "partial_session" not in frame:
        frame["partial_session"] = False
    frame["partial_session"] = frame["partial_session"].map(_normalise_bool)

    incomplete_trade = frame["traded"] & frame[list(OHLC)].isna().any(axis=1)
    populated_no_trade = ~frame["traded"] & frame[list(OHLC)].notna().any(axis=1)
    if incomplete_trade.any() or populated_no_trade.any():
        raise DataValidationError(
            "traded rows need four OHLC values and no-trade rows need none"
        )
    traded = frame[frame["traded"]]
    if (traded[list(OHLC)] <= 0).any(axis=None):
        raise DataValidationError("OHLC values must be positive")
    bad_high = traded["high"] < traded[["open", "close"]].max(axis=1)
    bad_low = traded["low"] > traded[["open", "close"]].min(axis=1)
    if bad_high.any() or bad_low.any() or (traded["high"] < traded["low"]).any():
        raise DataValidationError("daily data violates OHLC high/low bounds")
    if (frame[["volume", "turnover"]].dropna() < 0).any(axis=None):
        raise DataValidationError("volume and turnover must not be negative")

    frame = _deduplicate_or_raise(frame)
    return frame.sort_values(["code", "date"], kind="stable").reset_index(drop=True)


def merge_daily_prices(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    materialised = [frame for frame in frames if frame is not None and not frame.empty]
    if not materialised:
        raise DataValidationError("no daily price rows were collected")
    return normalize_daily_prices(pd.concat(materialised, ignore_index=True, sort=False))
