"""Point-in-time derived features for the canonical model-v0.8 research replay."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

G0 = (
    "oc_last",
    "oc_mean_5",
    "oc_mean_20",
    "oc_mean_60",
    "oc_win_20",
    "oc_std_20",
    "overnight_last",
    "overnight_mean_20",
    "overnight_mean_60",
    "night_day_corr_60",
    "xrank_atr14_pct",
    "xrank_close_momentum_5",
    "xrank_close_momentum_20",
    "xrank_close_momentum_60",
    "xrank_prior_close_location_20",
)

PERIODS = {
    "discovery": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-31")),
    "confirmation_a": (pd.Timestamp("2024-11-01"), pd.Timestamp("2025-03-31")),
    "confirmation_b": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_float32(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    if isinstance(values, pd.Series):
        result = pd.to_numeric(values, errors="coerce")
        return result.replace([np.inf, -np.inf], np.nan).astype("float32")
    result = np.asarray(values, dtype=float)
    result = np.where(np.isfinite(result), result, np.nan)
    return result.astype("float32")


def _validate_panel_keys(panel: pd.DataFrame) -> pd.DataFrame:
    """Return position-tagged, validated keys without trusting the index."""

    missing = sorted({"date", "code"} - set(panel.columns))
    if missing:
        raise ValueError(f"feature panel lacks key columns: {missing}")
    dates = pd.to_datetime(panel["date"], errors="coerce")
    codes = panel["code"]
    if dates.isna().any() or codes.isna().any():
        raise ValueError("feature panel contains an invalid date/code key")
    keys = pd.DataFrame(
        {
            "date": dates.to_numpy(),
            "code": codes.astype(str).to_numpy(),
            "_v08_position": np.arange(len(panel), dtype="int64"),
        }
    )
    if keys[["date", "code"]].duplicated().any():
        raise ValueError("feature panel contains duplicate date/code rows")
    return keys


def _stable_sorted_work(panel: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Sort by code/date for lag operations and retain a positional restore map."""

    keys = _validate_panel_keys(panel)
    order = (
        keys.sort_values(["code", "date", "_v08_position"], kind="stable")
        ["_v08_position"]
        .to_numpy(dtype="int64")
    )
    if np.array_equal(order, np.arange(len(panel), dtype="int64")):
        work = panel
    else:
        work = panel.iloc[order].copy()
    work_dates = pd.to_datetime(work["date"], errors="raise")
    if not work_dates.groupby(work["code"].astype(str), sort=False).apply(
        lambda values: values.is_monotonic_increasing
    ).all():
        raise ValueError("feature panel could not be ordered monotonically by code/date")
    return work, order


def prior_rolling_mean(
    values: pd.Series,
    codes: pd.Series,
    window: int,
    minimum: int,
) -> pd.Series:
    shifted = values.groupby(codes, sort=False).shift(1)
    result = (
        shifted.groupby(codes, sort=False)
        .rolling(window, min_periods=minimum)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    return result


def add_prior_close(panel: pd.DataFrame, daily_prices_path: str | Path) -> None:
    """Join the exact strictly-prior close from a caller-supplied daily-price cache."""
    _validate_panel_keys(panel)
    raw = joblib.load(Path(daily_prices_path), mmap_mode="r")
    lookup = raw.loc[:, ["date", "code", "close"]].rename(
        columns={"date": "feature_source_max_date", "close": "strict_prior_close"}
    )
    lookup["code"] = lookup["code"].astype(str)
    keys = panel.loc[:, ["feature_source_max_date", "code"]].copy()
    keys["code"] = keys["code"].astype(str)
    keys["_v08_position"] = np.arange(len(keys), dtype="int64")
    matched = keys.merge(
        lookup,
        on=["feature_source_max_date", "code"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    matched = matched.sort_values("_v08_position", kind="stable")
    if not np.array_equal(
        matched["_v08_position"].to_numpy(dtype="int64"),
        np.arange(len(panel), dtype="int64"),
    ):
        raise AssertionError("strict-prior-close join changed input row positions")
    panel["strict_prior_close"] = as_float32(
        matched["strict_prior_close"].to_numpy(dtype=float)
    )
    del raw, lookup, keys, matched
    gc.collect()


def _add_derived_features_sorted(panel: pd.DataFrame) -> tuple[dict[str, tuple[str, ...]], dict[str, pd.Series]]:
    codes = panel["code"].astype(str)
    dates = pd.to_datetime(panel["date"])
    ret = pd.to_numeric(panel["oc_return_pct"], errors="coerce")
    valid = ret.notna()
    ret0 = ret.fillna(0.0)

    # Strictly-prior expanding stock moments.  Subtracting the current row is
    # algebraically identical to shift(1), while avoiding a second large copy.
    prior_count = valid.astype("int32").groupby(codes, sort=False).cumsum() - valid.astype("int32")
    prior_sum = ret0.groupby(codes, sort=False).cumsum() - ret0
    prior_sq_sum = ret0.pow(2).groupby(codes, sort=False).cumsum() - ret0.pow(2)
    prior_mean = prior_sum / prior_count.where(prior_count.gt(0))
    prior_var = (
        prior_sq_sum - prior_sum.pow(2) / prior_count.where(prior_count.gt(0))
    ) / (prior_count - 1).where(prior_count.gt(1))
    prior_std = np.sqrt(prior_var.clip(lower=0.0))
    prior_win_sum = panel["label"].fillna(0.0).groupby(codes, sort=False).cumsum() - panel["label"].fillna(0.0)
    prior_win = prior_win_sum / prior_count.where(prior_count.gt(0))
    shrink = prior_count / (prior_count + 60.0)

    panel["h02_prior_count_log1p"] = as_float32(np.log1p(prior_count))
    panel["h02_stock_alpha_shrunk"] = as_float32(prior_mean * shrink)
    panel["h02_stock_win_shrunk"] = as_float32((prior_win - 0.5) * shrink)
    panel["h02_stock_alpha_t"] = as_float32(
        (prior_mean / (prior_std / np.sqrt(prior_count.clip(lower=1)) + 0.10)).clip(-10, 10)
    )

    # H01: daily cross-sectional ranks of raw G0 state.  All ranked inputs are
    # already sourced from sessions strictly before `date`.
    h01: list[str] = []
    for source in (
        "oc_last",
        "oc_mean_5",
        "oc_mean_20",
        "oc_mean_60",
        "oc_win_20",
        "oc_std_20",
        "overnight_last",
        "overnight_mean_20",
        "overnight_mean_60",
        "night_day_corr_60",
    ):
        target = f"h01_xsec_{source}"
        panel[target] = as_float32(panel[source].groupby(dates, sort=False).rank(pct=True).sub(0.5).mul(2.0))
        h01.append(target)

    # H03: asymmetric 60-session realised tails, shifted by one session.
    lag_ret = ret.groupby(codes, sort=False).shift(1)
    tail_inputs = pd.DataFrame(
        {
            "negative": lag_ret.clip(upper=0.0),
            "negative_sq": lag_ret.clip(upper=0.0).pow(2),
            "loss2": lag_ret.le(-2.0).where(lag_ret.notna()).astype(float),
            "gain2": lag_ret.ge(2.0).where(lag_ret.notna()).astype(float),
        },
        index=panel.index,
    )
    tail_roll = (
        tail_inputs.groupby(codes, sort=False)
        .rolling(60, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    panel["h03_downside_mean_60"] = as_float32(tail_roll["negative"])
    panel["h03_downside_semidev_60"] = as_float32(np.sqrt(tail_roll["negative_sq"]))
    panel["h03_loss2_rate_60"] = as_float32(tail_roll["loss2"])
    panel["h03_gain2_rate_60"] = as_float32(tail_roll["gain2"])
    panel["h03_tail_balance_60"] = as_float32(tail_roll["gain2"] - tail_roll["loss2"])

    # H04: changes of state across fixed, economically distinct horizons.
    panel["h04_oc_accel_5_20"] = as_float32(panel["oc_mean_5"] - panel["oc_mean_20"])
    panel["h04_oc_accel_20_60"] = as_float32(panel["oc_mean_20"] - panel["oc_mean_60"])
    panel["h04_overnight_accel_20_60"] = as_float32(panel["overnight_mean_20"] - panel["overnight_mean_60"])
    panel["h04_last_minus_mean5"] = as_float32(panel["oc_last"] - panel["oc_mean_5"])
    panel["h04_oc_accel_volnorm"] = as_float32(
        (panel["oc_mean_5"] - panel["oc_mean_20"]) / panel["oc_std_20"].clip(lower=0.10)
    )

    # H05: nonlinear regime switching using only prior-market state.
    panel["h05_alpha_x_breadth"] = as_float32(panel["oc_mean_20"] * panel["prior_market_breadth"])
    panel["h05_reversal_x_market_tail"] = as_float32(panel["cc_reversal_1_atr"] * panel["prior_market_tail_balance"])
    panel["h05_momentum_x_market_trend"] = as_float32(panel["xrank_close_momentum_20"] * panel["market_cc_momentum_5"])
    panel["h05_alpha_x_dispersion"] = as_float32(panel["oc_mean_20"] * panel["prior_market_dispersion"])
    panel["h05_alpha_x_vol_regime"] = as_float32(panel["oc_mean_20"] * panel["market_cc_vol_ratio_5_20"])
    panel["h05_gaptrait_x_market_gap"] = as_float32(panel["gap_response_beta_60"] * panel["prior_market_overnight_return_pct"])

    # H06: deterministic calendar interactions plus strictly-prior same-weekday alpha.
    dow = dates.dt.dayofweek.astype(float)
    dow_sin = np.sin(2.0 * np.pi * dow / 5.0)
    dow_cos = np.cos(2.0 * np.pi * dow / 5.0)
    month_end = dates.dt.is_month_end.astype(float)
    panel["h06_dowsin_x_alpha"] = as_float32(dow_sin * panel["oc_mean_20"])
    panel["h06_dowcos_x_alpha"] = as_float32(dow_cos * panel["oc_mean_20"])
    panel["h06_dowsin_x_reversal"] = as_float32(dow_sin * panel["oc_last"])
    panel["h06_monthend_x_momentum"] = as_float32(month_end * panel["xrank_close_momentum_20"])
    panel["h06_monthend_x_reversal"] = as_float32(month_end * panel["oc_last"])
    weekday_key = pd.Series(codes.astype(str) + "_" + dow.astype(int).astype(str), index=panel.index)
    wd_count = valid.astype("int32").groupby(weekday_key, sort=False).cumsum() - valid.astype("int32")
    wd_sum = ret0.groupby(weekday_key, sort=False).cumsum() - ret0
    wd_mean = wd_sum / wd_count.where(wd_count.gt(0))
    panel["h06_same_weekday_alpha_shrunk"] = as_float32(wd_mean * (wd_count / (wd_count + 12.0)))

    # H07: continuous tradability/reliability; no post-hoc cutoff.
    bad_activity = (
        panel["flat_oc_rate_20"].fillna(1.0)
        + panel["no_trade_rate_60"].fillna(1.0)
        + panel["zero_range_rate_20"].fillna(1.0)
    ) / 3.0
    activity = (1.0 - bad_activity).clip(0.0, 1.0)
    panel["h07_activity_score"] = as_float32(activity)
    panel["h07_alpha_x_activity"] = as_float32(panel["oc_mean_20"] * activity)
    panel["h07_vol_per_activity"] = as_float32(panel["oc_std_20"] / activity.clip(lower=0.10))
    panel["h07_liquidity_disagreement"] = as_float32(
        panel["flat_oc_rate_20"] - panel["no_trade_rate_60"]
    )

    # H08: minimum-lot affordability proxy from the exact prior close.
    log_price = np.log(panel["strict_prior_close"].where(panel["strict_prior_close"].gt(0)))
    panel["h08_log_prior_close"] = as_float32(log_price)
    panel["h08_xsec_price_rank"] = as_float32(
        panel["strict_prior_close"].groupby(dates, sort=False).rank(pct=True).sub(0.5).mul(2.0)
    )
    panel["h08_price_x_atr"] = as_float32(log_price * panel["xrank_atr14_pct"])
    panel["h08_log_distance_1000"] = as_float32((log_price - np.log(1000.0)).abs())

    # H09: lagged path exhaustion versus continuation.
    panel["h09_shock_x_range"] = as_float32(panel["cc_reversal_1_atr"] * panel["range_shock_1_20"])
    panel["h09_location_x_range"] = as_float32((panel["prior_session_close_location"] - 0.5) * panel["range_shock_1_20"])
    panel["h09_lastoc_x_sessionrange"] = as_float32(panel["oc_last"] * panel["session_range_ratio_5_20"])
    panel["h09_mom3_x_range"] = as_float32(panel["cc_momentum_3_atr"] * panel["range_shock_1_20"])

    # H10: historical gap response, not the target-session unknown gap.
    gap_std = panel["overnight_std_20"].clip(lower=0.10)
    panel["h10_lag_gap_z"] = as_float32(panel["overnight_last"] / gap_std)
    panel["h10_gap_x_response"] = as_float32(panel["overnight_last"] * panel["gap_response_beta_60"])
    panel["h10_gap_x_fill"] = as_float32(-panel["overnight_last"] * panel["gap_fill_rate_20"])
    panel["h10_gap_novelty_x_frequency"] = as_float32(
        panel["overnight_last"].abs() / gap_std * (1.0 - panel["gap_frequency_20"])
    )

    # H11: daily crowding summaries are label-blind and interacted with stock state.
    eligible = panel["price_eligible"].eq(True)
    crowd = panel.loc[eligible, ["date", "xrank_close_momentum_20", "xrank_atr14_pct"]].copy()
    crowd["mom_tail"] = crowd["xrank_close_momentum_20"].abs().ge(1.0).astype(float)
    crowd["atr_tail"] = crowd["xrank_atr14_pct"].abs().ge(1.0).astype(float)
    daily = crowd.groupby("date", sort=False).agg(
        mom_std=("xrank_close_momentum_20", "std"),
        mom_tail=("mom_tail", "mean"),
        atr_std=("xrank_atr14_pct", "std"),
        atr_tail=("atr_tail", "mean"),
    )
    # Map each label-blind daily summary back to the stock rows.
    mom_std = dates.map(daily["mom_std"])
    mom_tail = dates.map(daily["mom_tail"])
    atr_std = dates.map(daily["atr_std"])
    atr_tail = dates.map(daily["atr_tail"])
    panel["h11_momentum_x_crowd_std"] = as_float32(panel["xrank_close_momentum_20"] * mom_std)
    panel["h11_momentum_x_crowd_tail"] = as_float32(panel["xrank_close_momentum_20"] * mom_tail)
    panel["h11_alpha_x_atr_crowd_std"] = as_float32(panel["oc_mean_20"] * atr_std)
    panel["h11_alpha_x_atr_crowd_tail"] = as_float32(panel["oc_mean_20"] * atr_tail)

    # H12: make uncertainty/missingness explicit.
    missing_fraction = panel.loc[:, list(G0)].isna().mean(axis=1)
    feature_age = (dates - pd.to_datetime(panel["feature_source_max_date"])).dt.days
    panel["h12_prior_observed_log1p"] = as_float32(np.log1p(prior_count))
    panel["h12_g0_missing_fraction"] = as_float32(missing_fraction)
    panel["h12_feature_age_days"] = as_float32(feature_age)
    panel["h12_history_x_alpha"] = as_float32(np.log1p(prior_count) * panel["oc_mean_20"])

    groups = {
        "H01_cross_sectional_state": tuple(h01),
        "H02_expanding_stock_alpha": (
            "h02_prior_count_log1p", "h02_stock_alpha_shrunk",
            "h02_stock_win_shrunk", "h02_stock_alpha_t",
        ),
        "H03_downside_asymmetry": (
            "h03_downside_mean_60", "h03_downside_semidev_60",
            "h03_loss2_rate_60", "h03_gain2_rate_60", "h03_tail_balance_60",
        ),
        "H04_horizon_transition": (
            "h04_oc_accel_5_20", "h04_oc_accel_20_60",
            "h04_overnight_accel_20_60", "h04_last_minus_mean5",
            "h04_oc_accel_volnorm",
        ),
        "H05_market_regime_switch": (
            "h05_alpha_x_breadth", "h05_reversal_x_market_tail",
            "h05_momentum_x_market_trend", "h05_alpha_x_dispersion",
            "h05_alpha_x_vol_regime", "h05_gaptrait_x_market_gap",
        ),
        "H06_calendar_stock_interaction": (
            "h06_dowsin_x_alpha", "h06_dowcos_x_alpha",
            "h06_dowsin_x_reversal", "h06_monthend_x_momentum",
            "h06_monthend_x_reversal", "h06_same_weekday_alpha_shrunk",
        ),
        "H07_tradability_quality": (
            "h07_activity_score", "h07_alpha_x_activity",
            "h07_vol_per_activity", "h07_liquidity_disagreement",
        ),
        "H08_lot_price_state": (
            "h08_log_prior_close", "h08_xsec_price_rank",
            "h08_price_x_atr", "h08_log_distance_1000",
        ),
        "H09_range_exhaustion": (
            "h09_shock_x_range", "h09_location_x_range",
            "h09_lastoc_x_sessionrange", "h09_mom3_x_range",
        ),
        "H10_gap_response_state": (
            "h10_lag_gap_z", "h10_gap_x_response",
            "h10_gap_x_fill", "h10_gap_novelty_x_frequency",
        ),
        "H11_cross_sectional_crowding": (
            "h11_momentum_x_crowd_std", "h11_momentum_x_crowd_tail",
            "h11_alpha_x_atr_crowd_std", "h11_alpha_x_atr_crowd_tail",
        ),
        "H12_history_source_quality": (
            "h12_prior_observed_log1p", "h12_g0_missing_fraction",
            "h12_feature_age_days", "h12_history_x_alpha",
        ),
    }
    universe_masks = {
        "U01_active_history": (
            panel["flat_oc_rate_20"].le(0.10)
            & panel["no_trade_rate_60"].le(0.05)
            & panel["zero_range_rate_20"].le(0.05)
        ),
        "U02_middle_atr": panel["xrank_atr14_pct"].between(-0.80, 0.80),
        "U03_affordable_lot": panel["strict_prior_close"].between(200.0, 5000.0),
        "U04_mature_history": prior_count.ge(180),
        "U05_tail_clean": panel["h03_loss2_rate_60"].le(0.05),
    }
    for columns in groups.values():
        bad = ~np.isfinite(panel.loc[:, list(columns)].to_numpy(dtype=float)) & panel.loc[:, list(columns)].notna().to_numpy()
        if bad.any():
            raise AssertionError("derived feature contains non-finite non-missing values")
    del tail_inputs, tail_roll, lag_ret, crowd, daily
    gc.collect()
    return groups, universe_masks


def add_derived_features(
    panel: pd.DataFrame,
) -> tuple[dict[str, tuple[str, ...]], dict[str, pd.Series]]:
    """Attach v0.8 features without depending on caller row order or index labels.

    Historical rolling operations are evaluated on a stable code/date order.
    If the caller supplied another row order, every derived column and mask is
    restored by integer position—not by pandas index label—before returning.
    """

    work, order = _stable_sorted_work(panel)
    groups, sorted_masks = _add_derived_features_sorted(work)
    if work is panel:
        return groups, sorted_masks

    inverse = np.empty(len(order), dtype="int64")
    inverse[order] = np.arange(len(order), dtype="int64")
    derived_columns = list(
        dict.fromkeys(column for columns in groups.values() for column in columns)
    )
    for column in derived_columns:
        panel[column] = work[column].to_numpy()[inverse]
    restored_masks = {
        mask_id: pd.Series(
            mask.to_numpy(dtype=bool)[inverse],
            index=panel.index,
            dtype=bool,
        )
        for mask_id, mask in sorted_masks.items()
    }
    return groups, restored_masks
