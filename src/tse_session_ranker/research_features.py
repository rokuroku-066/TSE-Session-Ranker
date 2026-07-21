"""Point-in-time feature candidates kept outside the production manifest.

The functions in this module are intentionally opt-in.  They may be used by
walk-forward research, but they do not change ``features.FEATURE_COLUMNS`` or
the production artifact schema until a separate validation promotes them.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .data.common import SESSION_OHLC
from .data.market_context import normalize_market_context
from .data.tdnet import TDNET_MODEL_FEATURE_COLUMNS
from .exceptions import DataValidationError, LeakageError


PM_RETURN_ATR_CLIP = 5.0
PM_RANGE_ATR_CLIP = 10.0
PM_SHOCK_HINGE_ATR = 2.0

SESSION_FEATURE_COLUMNS: tuple[str, ...] = (
    "prior_am_return_pct",
    "prior_pm_return_pct",
    "prior_am_range_pct",
    "prior_pm_range_pct",
    "prior_midday_gap_pct",
    "prior_pm_close_location",
    "pm_return_mean_5",
    "pm_return_mean_20",
    "pm_win_rate_20",
    "prior_pm_return_atr_clipped",
    "prior_pm_return_atr_abs",
    "prior_pm_return_atr_positive_hinge_2",
    "prior_pm_return_atr_negative_hinge_2",
    "prior_pm_range_atr_clipped",
    "xrank_prior_pm_return_pct",
    "xrank_prior_pm_return_atr_abs",
    "xrank_prior_pm_range_atr_clipped",
    "xrank_prior_am_range_pct",
)

MARKET_FEATURE_COLUMNS: tuple[str, ...] = (
    "prior_market_oc_return_pct",
    "market_beta_60",
    "market_beta_x_prior_market_return",
    "atr_x_abs_prior_market_return",
    "prior_oc_market_residual_pct",
)

SESSION_MARKET_FEATURE_COLUMNS: tuple[str, ...] = (
    *SESSION_FEATURE_COLUMNS,
    *MARKET_FEATURE_COLUMNS,
)

FUTURES_INTERACTION_FEATURE_COLUMNS: tuple[str, ...] = (
    "futures_context_available",
    "nikkei_futures_available",
    "topix_futures_available",
    "market_beta_x_nikkei_futures_return",
    "market_beta_x_topix_futures_return",
    "market_beta_x_futures_spread",
    "atr_x_abs_nikkei_futures_return",
    "atr_x_abs_topix_futures_return",
    "tdnet_material_x_nikkei_futures_return",
    "tdnet_material_x_topix_futures_return",
)

RESEARCH_FEATURE_COLUMNS: tuple[str, ...] = (
    *SESSION_MARKET_FEATURE_COLUMNS,
    *FUTURES_INTERACTION_FEATURE_COLUMNS,
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DataValidationError(f"{source} is missing columns: {missing}")


def _safe_percent(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = denominator.where(denominator.gt(0)).replace([np.inf, -np.inf], np.nan)
    return 100.0 * (numerator / valid - 1.0)


def _prior_rolling(
    frame: pd.DataFrame,
    values: pd.Series,
    window: int,
    operation: str,
    min_periods: int,
) -> pd.Series:
    shifted = values.groupby(frame["code"], sort=False).shift(1)
    roller = shifted.groupby(frame["code"], sort=False).rolling(
        window, min_periods=min_periods
    )
    if operation == "mean":
        result = roller.mean()
    elif operation == "sum":
        result = roller.sum()
    else:
        raise ValueError(f"unsupported rolling operation: {operation}")
    return result.reset_index(level=0, drop=True).sort_index()


def _prior_rolling_sum(
    frame: pd.DataFrame, values: pd.Series, window: int
) -> pd.Series:
    return _prior_rolling(
        frame, values, window=window, operation="sum", min_periods=1
    )


def _cross_section_rank(
    frame: pd.DataFrame, values: pd.Series, universe: pd.Series
) -> pd.Series:
    """Map each date's finite cross-section onto [-1, 1] without look-ahead."""

    source = values.where(universe & np.isfinite(values))
    ranks = source.groupby(frame["date"], sort=False).rank(method="average")
    counts = source.groupby(frame["date"], sort=False).transform("count")
    result = 2.0 * (ranks - 1.0) / (counts - 1.0) - 1.0
    result = result.where(counts.gt(1), 0.0)
    return result.where(source.notna())


def _market_beta(
    frame: pd.DataFrame,
    stock_return: pd.Series,
    market_return: pd.Series,
    *,
    window: int = 60,
    min_pairs: int = 30,
) -> pd.Series:
    valid = stock_return.notna() & market_return.notna()
    x = market_return.where(valid, 0.0)
    y = stock_return.where(valid, 0.0)
    n = _prior_rolling_sum(frame, valid.astype(float), window)
    sx = _prior_rolling_sum(frame, x, window)
    sy = _prior_rolling_sum(frame, y, window)
    sxx = _prior_rolling_sum(frame, x * x, window)
    sxy = _prior_rolling_sum(frame, x * y, window)
    safe_n = n.where(n.gt(0))
    covariance_numerator = sxy - sx * sy / safe_n
    variance_numerator = sxx - sx * sx / safe_n
    beta = covariance_numerator / variance_numerator.where(
        variance_numerator.gt(1e-12)
    )
    return beta.where(n.ge(min_pairs))


def add_session_market_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add prior-session shape and market-interaction research features.

    Current-session AM/PM prices and current market return are used only as
    source observations for *later* rows.  Every returned feature for date D is
    built from dates strictly before D.
    """

    _require_columns(
        panel,
        ("date", "code", "open", "close", "traded", "atr14_pct"),
        "panel",
    )
    frame = panel.copy().reset_index(drop=True)
    frame["_research_original_order"] = np.arange(len(frame))
    parsed_dates = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    if parsed_dates.isna().any():
        raise DataValidationError("panel contains an invalid date")
    if getattr(parsed_dates.dt, "tz", None) is not None:
        parsed_dates = parsed_dates.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    frame["date"] = parsed_dates.dt.normalize()
    frame["code"] = frame["code"].astype(str).str.strip()
    if frame["code"].eq("").any():
        raise DataValidationError("panel contains an empty code")
    if frame.duplicated(["date", "code"]).any():
        raise DataValidationError("panel contains duplicate date/code rows")
    for column in SESSION_OHLC:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["atr14_pct"] = pd.to_numeric(frame["atr14_pct"], errors="coerce")
    frame = frame.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    group = frame.groupby("code", sort=False)

    am_return = _safe_percent(frame["am_close"], frame["am_open"])
    pm_return = _safe_percent(frame["pm_close"], frame["pm_open"])
    am_range = _safe_percent(frame["am_high"], frame["am_low"])
    pm_range = _safe_percent(frame["pm_high"], frame["pm_low"])
    midday_gap = _safe_percent(frame["pm_open"], frame["am_close"])
    pm_span = frame["pm_high"] - frame["pm_low"]
    pm_close_location = (
        (frame["pm_close"] - frame["pm_low"]) / pm_span.where(pm_span.gt(0))
    ).clip(0.0, 1.0)

    raw_session_features = {
        "prior_am_return_pct": am_return,
        "prior_pm_return_pct": pm_return,
        "prior_am_range_pct": am_range,
        "prior_pm_range_pct": pm_range,
        "prior_midday_gap_pct": midday_gap,
        "prior_pm_close_location": pm_close_location,
    }
    for name, values in raw_session_features.items():
        frame[name] = values.groupby(frame["code"], sort=False).shift(1)

    frame["pm_return_mean_5"] = _prior_rolling(
        frame, pm_return, window=5, operation="mean", min_periods=3
    )
    frame["pm_return_mean_20"] = _prior_rolling(
        frame, pm_return, window=20, operation="mean", min_periods=10
    )
    pm_win = pm_return.gt(0).astype(float).where(pm_return.notna())
    frame["pm_win_rate_20"] = _prior_rolling(
        frame, pm_win, window=20, operation="mean", min_periods=10
    )

    safe_atr = frame["atr14_pct"].where(frame["atr14_pct"].gt(0))
    pm_atr = (frame["prior_pm_return_pct"] / safe_atr).clip(
        -PM_RETURN_ATR_CLIP, PM_RETURN_ATR_CLIP
    )
    frame["prior_pm_return_atr_clipped"] = pm_atr
    frame["prior_pm_return_atr_abs"] = pm_atr.abs()
    frame["prior_pm_return_atr_positive_hinge_2"] = (
        pm_atr - PM_SHOCK_HINGE_ATR
    ).clip(lower=0.0)
    frame["prior_pm_return_atr_negative_hinge_2"] = (
        -pm_atr - PM_SHOCK_HINGE_ATR
    ).clip(lower=0.0)
    frame["prior_pm_range_atr_clipped"] = (
        frame["prior_pm_range_pct"] / safe_atr
    ).clip(0.0, PM_RANGE_ATR_CLIP)

    if "training_eligible" in frame:
        rank_universe = frame["training_eligible"].fillna(False).astype(bool)
    elif "prior_universe_member" in frame:
        rank_universe = frame["prior_universe_member"].fillna(False).astype(bool)
    else:
        rank_universe = pd.Series(True, index=frame.index)
    rank_sources = {
        "xrank_prior_pm_return_pct": frame["prior_pm_return_pct"],
        "xrank_prior_pm_return_atr_abs": frame["prior_pm_return_atr_abs"],
        "xrank_prior_pm_range_atr_clipped": frame[
            "prior_pm_range_atr_clipped"
        ],
        "xrank_prior_am_range_pct": frame["prior_am_range_pct"],
    }
    for name, values in rank_sources.items():
        frame[name] = _cross_section_rank(frame, values, rank_universe)

    traded = frame["traded"].fillna(False).astype(bool)
    stock_oc = _safe_percent(frame["close"], frame["open"]).where(traded)
    market_source = stock_oc
    if "source_complete" in frame:
        market_source = market_source.where(
            frame["source_complete"].fillna(False).astype(bool)
        )
    market_by_date = market_source.groupby(frame["date"], sort=True).mean()
    date_order = pd.DatetimeIndex(sorted(frame["date"].unique()))
    market_by_date = market_by_date.reindex(date_order)
    prior_market_by_date = market_by_date.shift(1)
    current_market = frame["date"].map(market_by_date)
    frame["prior_market_oc_return_pct"] = frame["date"].map(prior_market_by_date)
    frame["market_beta_60"] = _market_beta(frame, stock_oc, current_market)
    frame["market_beta_x_prior_market_return"] = (
        frame["market_beta_60"] * frame["prior_market_oc_return_pct"]
    )
    frame["atr_x_abs_prior_market_return"] = (
        frame["atr14_pct"] * frame["prior_market_oc_return_pct"].abs()
    )
    prior_stock_oc = stock_oc.groupby(frame["code"], sort=False).shift(1)
    frame["prior_oc_market_residual_pct"] = prior_stock_oc - (
        frame["market_beta_60"] * frame["prior_market_oc_return_pct"]
    )

    prior_source_date = group["date"].shift(1)
    if (prior_source_date.notna() & prior_source_date.ge(frame["date"])).any():
        raise LeakageError("research features contain a non-prior price source")
    for column in SESSION_MARKET_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("_research_original_order", kind="stable").drop(
        columns=["_research_original_order"]
    )
    return frame.reset_index(drop=True)


def _normalise_context_date(value: object) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("futures context contains an invalid date") from exc
    if pd.isna(timestamp):
        raise DataValidationError("futures context contains an invalid date")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Tokyo").tz_localize(None)
    return timestamp.normalize()


def add_futures_interactions(
    panel: pd.DataFrame,
    context: pd.DataFrame,
    decision_time: str = "08:58:59",
) -> pd.DataFrame:
    """Attach exact-date pre-open futures interactions without fallback values."""

    _require_columns(panel, ("date", "market_beta_60", "atr14_pct"), "panel")
    _require_columns(
        context,
        (
            "date",
            "observed_at",
            "nikkei_return_pct",
            "topix_return_pct",
            "return_definition",
        ),
        "futures context",
    )
    futures = normalize_market_context(
        context, decision_time=decision_time
    )[
        [
            "date",
            "observed_at",
            "nikkei_return_pct",
            "topix_return_pct",
            "return_definition",
        ]
    ].copy()

    frame = panel.copy().reset_index(drop=True)
    frame["date"] = frame["date"].map(_normalise_context_date)
    if "code" in frame and frame.duplicated(["date", "code"]).any():
        raise DataValidationError("panel contains duplicate date/code rows")
    futures = futures.rename(
        columns={
            "observed_at": "futures_observed_at",
            "nikkei_return_pct": "futures_nikkei_return_pct_context",
            "topix_return_pct": "futures_topix_return_pct_context",
            "return_definition": "futures_return_definition",
        }
    )
    frame = frame.merge(futures, on="date", how="left", validate="many_to_one")
    nikkei = frame["futures_nikkei_return_pct_context"]
    topix = frame["futures_topix_return_pct_context"]
    frame["nikkei_futures_available"] = nikkei.notna().astype(float)
    frame["topix_futures_available"] = topix.notna().astype(float)
    frame["futures_context_available"] = (
        nikkei.notna() & topix.notna()
    ).astype(float)
    beta = pd.to_numeric(frame["market_beta_60"], errors="coerce")
    atr = pd.to_numeric(frame["atr14_pct"], errors="coerce")
    frame["market_beta_x_nikkei_futures_return"] = beta * nikkei
    frame["market_beta_x_topix_futures_return"] = beta * topix
    frame["market_beta_x_futures_spread"] = beta * (nikkei - topix)
    frame["atr_x_abs_nikkei_futures_return"] = atr * nikkei.abs()
    frame["atr_x_abs_topix_futures_return"] = atr * topix.abs()

    available_flags = [
        column for column in TDNET_MODEL_FEATURE_COLUMNS if column in frame
    ]
    if available_flags:
        flags = frame[available_flags].apply(pd.to_numeric, errors="coerce")
        material = flags.max(axis=1, skipna=False)
    else:
        material = pd.Series(np.nan, index=frame.index, dtype=float)
    frame["tdnet_material_x_nikkei_futures_return"] = material * nikkei
    frame["tdnet_material_x_topix_futures_return"] = material * topix
    for column in FUTURES_INTERACTION_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame
