from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..exceptions import DataValidationError


OHLC = ("open", "high", "low", "close")
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
REQUIRED_DAILY_COLUMNS = ("date", "code", *OHLC)


def normalize_expected_sessions(
    values: Iterable[object] | object,
    *,
    through: object | None = None,
) -> pd.DatetimeIndex:
    """Canonicalise an explicit exchange-session calendar."""

    if isinstance(values, (str, pd.Timestamp, np.datetime64)):
        values = [values]
    parsed = pd.to_datetime(pd.Series(list(values)), errors="coerce", format="mixed")
    if parsed.empty or parsed.isna().any():
        raise DataValidationError("expected session calendar contains invalid dates")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    sessions = pd.DatetimeIndex(parsed.dt.normalize().drop_duplicates().sort_values())
    if through is not None:
        sessions = sessions[sessions <= pd.Timestamp(through).normalize()]
    if sessions.empty:
        raise DataValidationError("expected session calendar is empty")
    return sessions


def session_calendar_hash(
    values: Iterable[object] | object,
    *,
    through: object | None = None,
) -> str:
    sessions = normalize_expected_sessions(values, through=through)
    encoded = "\n".join(session.strftime("%Y-%m-%d") for session in sessions)
    return hashlib.sha256((encoded + "\n").encode("utf-8")).hexdigest()


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
        for column in (
            "name",
            *OHLC,
            *SESSION_OHLC,
            "volume",
            "turnover",
            "traded",
            "partial_session",
        )
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
    for column in (*OHLC, *SESSION_OHLC, "volume", "turnover"):
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
    populated_session = frame[list(SESSION_OHLC)].notna().any(axis=1)
    incomplete_session = populated_session & frame[list(SESSION_OHLC)].isna().any(
        axis=1
    )
    if incomplete_session.any():
        raise DataValidationError(
            "session OHLC must contain all eight AM/PM prices or none"
        )
    invalid_session_state = populated_session & (
        ~frame["traded"] | frame["partial_session"]
    )
    if invalid_session_state.any():
        raise DataValidationError(
            "AM/PM OHLC may be populated only for a full traded session"
        )
    session_rows = frame.loc[populated_session, list(SESSION_OHLC)]
    if (session_rows <= 0).any(axis=None):
        raise DataValidationError("session OHLC values must be positive")
    bad_am_high = session_rows["am_high"] < session_rows[
        ["am_open", "am_close"]
    ].max(axis=1)
    bad_am_low = session_rows["am_low"] > session_rows[
        ["am_open", "am_close"]
    ].min(axis=1)
    bad_pm_high = session_rows["pm_high"] < session_rows[
        ["pm_open", "pm_close"]
    ].max(axis=1)
    bad_pm_low = session_rows["pm_low"] > session_rows[
        ["pm_open", "pm_close"]
    ].min(axis=1)
    if (
        bad_am_high.any()
        or bad_am_low.any()
        or (session_rows["am_high"] < session_rows["am_low"]).any()
        or bad_pm_high.any()
        or bad_pm_low.any()
        or (session_rows["pm_high"] < session_rows["pm_low"]).any()
    ):
        raise DataValidationError("daily data violates AM/PM OHLC bounds")
    if (frame[["volume", "turnover"]].dropna() < 0).any(axis=None):
        raise DataValidationError("volume and turnover must not be negative")

    frame = _deduplicate_or_raise(frame)
    return frame.sort_values(["code", "date"], kind="stable").reset_index(drop=True)


def merge_daily_prices(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    materialised = [frame for frame in frames if frame is not None and not frame.empty]
    if not materialised:
        raise DataValidationError("no daily price rows were collected")
    return normalize_daily_prices(pd.concat(materialised, ignore_index=True, sort=False))


def session_coverage_report(
    prices: pd.DataFrame,
    *,
    lookback: int = 20,
    minimum_coverage: float = 0.90,
    expected_sessions: Iterable[object] | None = None,
    expected_through: object | None = None,
) -> pd.DataFrame:
    """Report unusually incomplete source sessions without using future dates.

    The reference is estimated from previously *accepted* sessions only.  A
    sustained source outage therefore cannot lower its own reference and later
    certify itself as complete.  A genuine structural universe contraction
    requires an explicit baseline reset instead of an automatic fail-open.
    """

    if lookback < 10:
        raise ValueError("coverage lookback must be at least 10 sessions")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    frame = normalize_daily_prices(prices)
    counts = frame.groupby("date", sort=True)["code"].nunique()
    if expected_sessions is not None:
        full_calendar = normalize_expected_sessions(expected_sessions)
        if expected_through is not None:
            calendar_through = pd.Timestamp(expected_through).normalize()
            if calendar_through not in full_calendar:
                raise DataValidationError(
                    "expected_through is not in the exchange session calendar"
                )
        else:
            calendar_through = counts.index.max()
        calendar = full_calendar[full_calendar <= calendar_through]
        calendar = calendar[calendar >= counts.index.min()]
        unexpected = counts.index.difference(calendar)
        if len(unexpected):
            examples = ", ".join(str(value.date()) for value in unexpected[:5])
            raise DataValidationError(
                "daily data contains dates outside expected session calendar: "
                + examples
            )
        counts = counts.reindex(calendar, fill_value=0)
        counts.index.name = "date"
    elif expected_through is not None:
        raise ValueError("expected_through requires expected_sessions")
    bootstrap_sessions = min(10, lookback)
    accepted_counts: list[float] = []
    references: list[float] = []
    ratios: list[float] = []
    complete_flags: list[bool] = []
    for count in counts.astype(float):
        if count <= 0:
            reference = (
                float(np.quantile(accepted_counts[-lookback:], 0.90))
                if accepted_counts
                else float("nan")
            )
            ratio = 0.0
            complete = False
        elif len(accepted_counts) < bootstrap_sessions:
            reference = float("nan")
            ratio = float("nan")
            complete = True
        else:
            reference = float(
                np.quantile(accepted_counts[-lookback:], 0.90)
            )
            ratio = count / reference if reference else 0.0
            complete = ratio >= minimum_coverage
        references.append(reference)
        ratios.append(ratio)
        complete_flags.append(complete)
        if complete:
            accepted_counts.append(float(count))
    report = pd.DataFrame(
        {
            "session_rows": counts,
            "reference_rows": references,
            "coverage_ratio": ratios,
            "source_complete": complete_flags,
        }
    ).reset_index()
    return report


def build_prior_session_universe(
    prices: pd.DataFrame,
    *,
    expected_sessions: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Create each target-day universe solely from the preceding session.

    Target outcomes are joined only after the candidate rows are fixed.  If a
    security was present on the preceding session but its target-day row is
    absent, it remains in the panel as ``traded=False`` with missing OHLC.  An
    outer join also retains observed rows outside that candidate set strictly
    as feature history; ``prior_universe_member`` keeps them out of scoring.
    This prevents historical ranking from conditioning on eventual execution.
    """

    frame = normalize_daily_prices(prices)
    observed_sessions = pd.DatetimeIndex(sorted(frame["date"].unique()))
    if expected_sessions is None:
        sessions = observed_sessions
    else:
        sessions = normalize_expected_sessions(
            expected_sessions, through=observed_sessions.max()
        )
        sessions = sessions[sessions >= observed_sessions.min()]
        unexpected = observed_sessions.difference(sessions)
        if len(unexpected):
            examples = ", ".join(str(value.date()) for value in unexpected[:5])
            raise DataValidationError(
                "daily data contains dates outside expected session calendar: "
                + examples
            )
    if len(sessions) < 2:
        raise DataValidationError(
            "at least two sessions are required to build a prior-session universe"
        )
    next_session = dict(zip(sessions[:-1], sessions[1:], strict=True))
    previous_session = dict(zip(sessions[1:], sessions[:-1], strict=True))
    candidates = frame[["date", "code", "name"]].copy()
    candidates["universe_source_date"] = candidates["date"]
    candidates["date"] = candidates["date"].map(next_session)
    candidates = candidates.dropna(subset=["date"]).rename(
        columns={"name": "prior_name"}
    )
    candidates["prior_universe_member"] = True

    outcome_columns = [
        "date",
        "code",
        "name",
        *OHLC,
        *SESSION_OHLC,
        "volume",
        "turnover",
        "traded",
        "partial_session",
    ]
    outcomes = frame[outcome_columns].copy()
    combined = candidates.merge(
        outcomes,
        on=["date", "code"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    combined["outcome_observed"] = combined["_merge"].ne("left_only")
    combined["prior_universe_member"] = combined[
        "prior_universe_member"
    ].eq(True)
    right_only = combined["_merge"].eq("right_only")
    combined.loc[right_only, "universe_source_date"] = combined.loc[
        right_only, "date"
    ].map(previous_session)
    combined["name"] = combined["name"].fillna(combined["prior_name"])
    combined = combined.drop(columns=["prior_name", "_merge"])
    missing_outcome = combined["traded"].isna()
    combined.loc[missing_outcome, "traded"] = False
    combined.loc[missing_outcome, "partial_session"] = False
    combined["traded"] = combined["traded"].astype(bool)
    combined["partial_session"] = combined["partial_session"].astype(bool)
    return normalize_daily_prices(combined)


def prepare_modeling_prices(
    prices: pd.DataFrame,
    *,
    coverage_lookback: int = 20,
    minimum_source_coverage: float = 0.90,
    expected_sessions: Iterable[object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a prior-session universe while retaining the real session calendar.

    Source-incomplete sessions stay in the history so the next session is never
    reconnected to an older date.  They and their immediate successor are
    marked ineligible for model fitting/scoring, while observed outcomes remain
    available as individual-security feature history.
    """

    canonical = normalize_daily_prices(prices)
    calendar = (
        normalize_expected_sessions(
            expected_sessions, through=canonical["date"].max()
        )
        if expected_sessions is not None
        else None
    )
    coverage = session_coverage_report(
        canonical,
        lookback=coverage_lookback,
        minimum_coverage=minimum_source_coverage,
        expected_sessions=calendar,
    )
    completeness = coverage.set_index("date")["source_complete"]
    modeled = build_prior_session_universe(
        canonical, expected_sessions=calendar
    )
    modeled["source_complete"] = modeled["date"].map(completeness).eq(True)
    modeled["universe_source_complete"] = modeled[
        "universe_source_date"
    ].map(completeness).eq(True)
    modeled["evaluation_ready"] = (
        modeled["prior_universe_member"]
        & modeled["source_complete"]
        & modeled["universe_source_complete"]
    )
    return modeled, coverage
