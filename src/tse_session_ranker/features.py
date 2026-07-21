from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import RankerConfig, UniversePolicy
from .data.common import (
    normalize_daily_prices,
    normalize_expected_sessions,
    prepare_modeling_prices,
    session_coverage_report,
)
from .data.tdnet import (
    TDNET_MODEL_FEATURE_COLUMNS,
    TDnetDataset,
    merge_tdnet_features,
    require_production_tdnet_provenance,
    tdnet_target_completeness,
)
from .exceptions import DataValidationError, LeakageError


PRICE_FEATURE_COLUMNS: tuple[str, ...] = (
    "oc_last",
    "oc_mean_5",
    "oc_mean_20",
    "oc_mean_60",
    "oc_win_5",
    "oc_win_20",
    "oc_win_60",
    "oc_std_20",
    "overnight_last",
    "overnight_mean_20",
    "overnight_mean_60",
    "night_day_corr_60",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    *PRICE_FEATURE_COLUMNS,
    *TDNET_MODEL_FEATURE_COLUMNS,
)

BANNED_MODEL_COLUMNS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "label",
        "win",
        "oc",
        "oc_return_pct",
        "next_open",
        "next_close",
    }
)


def validate_feature_columns(columns: Sequence[str]) -> tuple[str, ...]:
    result = tuple(columns)
    banned = sorted(set(result) & BANNED_MODEL_COLUMNS)
    if banned:
        raise LeakageError(f"target-derived columns cannot be model features: {banned}")
    if result != FEATURE_COLUMNS:
        raise LeakageError(
            "artifact feature order does not match the session_v3 manifest"
        )
    return result


def _prior_rolling(
    frame: pd.DataFrame,
    column: str,
    window: int,
    operation: str,
    min_periods: int | None = None,
) -> pd.Series:
    minimum = min_periods if min_periods is not None else max(5, window // 2)
    shifted = frame.groupby("code", sort=False)[column].shift(1)
    roller = shifted.groupby(frame["code"], sort=False).rolling(
        window, min_periods=minimum
    )
    if operation == "mean":
        result = roller.mean()
    elif operation == "std":
        result = roller.std()
    elif operation == "sum":
        result = roller.sum()
    elif operation == "max":
        result = roller.max()
    elif operation == "min":
        result = roller.min()
    else:
        raise ValueError(f"unsupported rolling operation: {operation}")
    return result.reset_index(level=0, drop=True).sort_index()


def _prior_rolling_corr(
    frame: pd.DataFrame, left: str, right: str, window: int, min_pairs: int
) -> pd.Series:
    valid = frame[left].notna() & frame[right].notna()
    x = frame[left].where(valid, 0.0)
    y = frame[right].where(valid, 0.0)
    temporary = frame[["code"]].copy()
    temporary["n"] = valid.astype(float)
    temporary["x"] = x
    temporary["y"] = y
    temporary["xy"] = x * y
    temporary["xx"] = x * x
    temporary["yy"] = y * y
    sums = {
        column: _prior_rolling(temporary, column, window, "sum", min_periods=1)
        for column in ("n", "x", "y", "xy", "xx", "yy")
    }
    count = sums["n"]
    safe_count = count.where(count > 0)
    mean_x = sums["x"] / safe_count
    mean_y = sums["y"] / safe_count
    covariance = sums["xy"] / safe_count - mean_x * mean_y
    variance_x = (sums["xx"] / safe_count - mean_x * mean_x).clip(lower=0)
    variance_y = (sums["yy"] / safe_count - mean_y * mean_y).clip(lower=0)
    denominator = np.sqrt(variance_x * variance_y)
    correlation = covariance / denominator.where(denominator > 0)
    return correlation.where(count >= min_pairs)


def eligibility_mask(
    frame: pd.DataFrame,
    policy: UniversePolicy | None = None,
    *,
    for_training: bool = False,
) -> pd.Series:
    rules = policy or UniversePolicy()
    atr_max = (
        rules.model_training_atr_max_pct if for_training else rules.atr14_max_pct
    )
    return (
        frame["history_count"].ge(rules.min_history)
        & frame["prior_close"].between(rules.price_min, rules.price_max)
        & frame["atr14_pct"].between(
            rules.atr14_min_pct, atr_max
        )
        & frame["zero_oc_20"].le(rules.max_zero_oc_20)
    ).fillna(False)


def explain_ineligibility(
    frame: pd.DataFrame, policy: UniversePolicy | None = None
) -> pd.Series:
    rules = policy or UniversePolicy()

    def explain(row: pd.Series) -> str:
        reasons: list[str] = []
        if row["history_count"] < rules.min_history:
            reasons.append("history")
        if not rules.price_min <= row["prior_close"] <= rules.price_max:
            reasons.append("price")
        if not rules.atr14_min_pct <= row["atr14_pct"] <= rules.atr14_max_pct:
            reasons.append("atr")
        if pd.isna(row["zero_oc_20"]) or row["zero_oc_20"] > rules.max_zero_oc_20:
            reasons.append("no_trade_or_flat")
        return "eligible" if not reasons else ",".join(reasons)

    return frame.apply(explain, axis=1)


def build_feature_panel(
    prices: pd.DataFrame, config: RankerConfig | None = None
) -> pd.DataFrame:
    """Build leakage-safe features for each observed target session.

    Every model feature and eligibility field is based on rows strictly before
    that row's ``date``.  The current row is used only to create the training
    label and realised open-to-close return.
    """

    settings = config or RankerConfig()
    validate_feature_columns(FEATURE_COLUMNS)
    frame = normalize_daily_prices(prices)
    group = frame.groupby("code", sort=False)
    observed = frame.get(
        "outcome_observed", pd.Series(True, index=frame.index, dtype=bool)
    ).fillna(False).astype(bool)
    frame["history_count"] = observed.groupby(frame["code"], sort=False).cumsum()
    frame["history_count"] = frame["history_count"] - observed.astype(int)
    frame["feature_source_max_date"] = group["date"].shift(1)
    frame["effective_close"] = group["close"].ffill()
    prior_outcome_observed = observed.groupby(
        frame["code"], sort=False
    ).shift(1).eq(True)
    frame["prior_close"] = frame.groupby(
        "code", sort=False
    )["effective_close"].shift(1).where(prior_outcome_observed)

    traded = frame["traded"] & frame[["open", "high", "low", "close"]].notna().all(axis=1)
    frame["oc_return_pct"] = np.nan
    frame.loc[traded, "oc_return_pct"] = 100.0 * (
        frame.loc[traded, "close"] / frame.loc[traded, "open"] - 1.0
    )
    frame["label"] = np.nan
    frame.loc[traded, "label"] = (
        frame.loc[traded, "close"] > frame.loc[traded, "open"]
    ).astype(float)
    frame["overnight"] = np.nan
    valid_overnight = traded & frame["prior_close"].notna()
    frame.loc[valid_overnight, "overnight"] = 100.0 * (
        frame.loc[valid_overnight, "open"]
        / frame.loc[valid_overnight, "prior_close"]
        - 1.0
    )

    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - frame["prior_close"]).abs(),
            (frame["low"] - frame["prior_close"]).abs(),
        ],
        axis=1,
    )
    frame["true_range"] = ranges.max(axis=1).where(traded)

    frame["oc_last"] = group["oc_return_pct"].shift(1)
    frame["overnight_last"] = group["overnight"].shift(1)
    for window in (5, 20, 60):
        frame[f"oc_mean_{window}"] = _prior_rolling(
            frame, "oc_return_pct", window, "mean"
        )
        frame[f"oc_win_{window}"] = _prior_rolling(
            frame, "label", window, "mean"
        )
    frame["oc_std_20"] = _prior_rolling(frame, "oc_return_pct", 20, "std")
    for window in (20, 60):
        frame[f"overnight_mean_{window}"] = _prior_rolling(
            frame, "overnight", window, "mean"
        )
    frame["night_day_corr_60"] = _prior_rolling_corr(
        frame, "overnight", "oc_return_pct", window=60, min_pairs=30
    )

    atr14 = _prior_rolling(frame, "true_range", 14, "mean", min_periods=7)
    frame["atr14_pct"] = 100.0 * atr14 / frame["prior_close"]
    synthetic = frame.get(
        "synthetic_target", pd.Series(False, index=frame.index, dtype=bool)
    ).fillna(False)
    source_complete = frame.get(
        "source_complete", pd.Series(True, index=frame.index, dtype=bool)
    ).fillna(False).astype(bool)
    unobserved_source_gap = ~observed & ~source_complete
    frame["zero_oc"] = np.where(
        synthetic | unobserved_source_gap,
        np.nan,
        np.where(~traded, 1.0, frame["oc_return_pct"].abs().lt(1e-12).astype(float)),
    )
    frame["zero_oc_20"] = _prior_rolling(
        frame, "zero_oc", 20, "mean", min_periods=10
    )
    evaluation_ready = frame.get(
        "evaluation_ready",
        frame.get(
            "prior_universe_member",
            pd.Series(True, index=frame.index, dtype=bool),
        ),
    )
    evaluation_ready = pd.Series(evaluation_ready, index=frame.index).fillna(False)
    frame["eligible"] = (
        eligibility_mask(frame, settings.universe) & evaluation_ready
    )
    frame["training_eligible"] = (
        eligibility_mask(frame, settings.universe, for_training=True)
        & evaluation_ready
    )

    leaked_time = frame["feature_source_max_date"].notna() & (
        frame["feature_source_max_date"] >= frame["date"]
    )
    if leaked_time.any():
        raise LeakageError("feature source date is not strictly before target date")
    return frame.drop(columns=["effective_close"])


def build_inference_frame(
    prices: pd.DataFrame,
    target_date: object,
    tdnet_dataset: TDnetDataset,
    config: RankerConfig | None = None,
    expected_history_date: object | None = None,
    expected_sessions: object | None = None,
) -> pd.DataFrame:
    """Create the next-session feature rows without requiring target OHLC data."""

    settings = config or RankerConfig()
    require_production_tdnet_provenance(tdnet_dataset)
    target = pd.Timestamp(target_date).normalize()
    history = normalize_daily_prices(prices)
    history = history[history["date"] < target].copy()
    if history.empty:
        raise DataValidationError("no price history exists before target_date")
    latest_session = history["date"].max()
    calendar = (
        normalize_expected_sessions(expected_sessions, through=target)
        if expected_sessions is not None
        else None
    )
    age_days = int((target - latest_session).days)
    if age_days > settings.max_history_age_calendar_days:
        raise DataValidationError(
            f"daily history is stale: latest={latest_session.date()}, "
            f"target={target.date()}, age={age_days} calendar days"
        )
    if expected_history_date is None and settings.require_expected_history_date:
        raise DataValidationError(
            "expected_history_date is required for fail-closed live inference"
        )
    if calendar is None:
        raise DataValidationError(
            "session_v3 inference requires an explicit exchange session calendar"
        )
    if calendar is not None:
        if target not in calendar:
            raise DataValidationError(
                "target_date is not in the exchange session calendar"
            )
        prior_sessions = calendar[calendar < target]
        if prior_sessions.empty:
            raise DataValidationError(
                "exchange session calendar has no session before target_date"
            )
        if expected_history_date is None:
            raise DataValidationError(
                "expected_history_date is required with an explicit session calendar"
            )
        calendar_previous = prior_sessions.max()
        expected = pd.Timestamp(expected_history_date).normalize()
        if calendar_previous != expected:
            raise DataValidationError(
                f"calendar previous session is {calendar_previous.date()}, "
                f"expected_history_date is {expected.date()}"
            )
    if expected_history_date is not None:
        expected = pd.Timestamp(expected_history_date).normalize()
        if expected >= target:
            raise DataValidationError("expected_history_date must precede target_date")
        if latest_session != expected:
            raise DataValidationError(
                f"latest daily session is {latest_session.date()}, "
                f"expected {expected.date()}"
            )
    source_coverage = session_coverage_report(
        history,
        lookback=settings.source_coverage_lookback,
        minimum_coverage=max(
            settings.minimum_source_coverage,
            settings.min_latest_session_coverage,
        ),
        expected_sessions=calendar,
    )
    latest_report = source_coverage[
        source_coverage["date"].eq(latest_session)
    ].iloc[0]
    coverage = float(latest_report["coverage_ratio"])
    if not np.isfinite(coverage):
        coverage = 1.0
    if not bool(latest_report["source_complete"]):
        raise DataValidationError(
            f"latest daily session coverage is only {coverage:.1%}; "
            "collection may be incomplete"
        )
    latest = (
        history.sort_values(["code", "date"], kind="stable")
        .groupby("code", as_index=False, sort=False)
        .tail(1)
    )
    # JPX no-trade rows are retained, so presence on the latest complete
    # session is the safest available active-listing signal.  This removes
    # delisted and code-changed securities from live inference.
    latest = latest[latest["date"].eq(latest_session)].copy()
    modeling_history, source_coverage = prepare_modeling_prices(
        history,
        coverage_lookback=settings.source_coverage_lookback,
        minimum_source_coverage=settings.minimum_source_coverage,
        expected_sessions=calendar,
    )
    incomplete_dates = set(
        source_coverage.loc[~source_coverage["source_complete"], "date"]
    )
    if latest_session in incomplete_dates:
        raise DataValidationError(
            f"latest daily session {latest_session.date()} is source-incomplete"
        )
    placeholders = pd.DataFrame(
        {
            "date": target,
            "code": latest["code"].to_numpy(),
            "name": latest["name"].to_numpy(),
            "open": np.nan,
            "high": np.nan,
            "low": np.nan,
            "close": np.nan,
            "volume": np.nan,
            "turnover": np.nan,
            "traded": False,
            "partial_session": False,
            "synthetic_target": True,
            "universe_source_date": latest_session,
            "prior_universe_member": True,
            "outcome_observed": False,
            "source_complete": True,
            "universe_source_complete": True,
            "evaluation_ready": True,
        }
    )
    modeling_history["synthetic_target"] = False
    combined = pd.concat(
        [modeling_history, placeholders], ignore_index=True, sort=False
    )
    tdnet_coverage = tdnet_target_completeness(
        calendar,
        tdnet_dataset.complete_dates,
        tdnet_dataset.observed_at_by_date,
        decision_time=settings.preopen.decision_time,
    )
    if not bool(tdnet_coverage.get(target, False)):
        raise DataValidationError(
            f"TDnet index coverage is incomplete for target session {target.date()}"
        )
    panel = merge_tdnet_features(
        build_feature_panel(combined, settings),
        tdnet_dataset,
        calendar,
        decision_time=settings.preopen.decision_time,
    )
    inference = panel[panel["synthetic_target"].fillna(False)].copy()
    if inference["label"].notna().any() or inference["oc_return_pct"].notna().any():
        raise LeakageError("inference rows unexpectedly contain target labels")
    inference["eligibility_reason"] = explain_ineligibility(
        inference, settings.universe
    )
    inference["daily_data_date"] = latest_session
    inference["latest_session_coverage"] = coverage
    return inference.sort_values("code", kind="stable").reset_index(drop=True)
