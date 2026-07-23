#!/usr/bin/env python3
"""Walk-forward validation of literature-derived feature blocks.

This runner is deliberately self-contained. It reads the frozen panel supplied
under ``/tmp`` and refreshes only the registered Model v10 research result plus
an untracked picks file. It does not modify any production inference artifact.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RESEARCH_DIR = Path(__file__).resolve().parent
PANEL_PATH = Path("/tmp/model_v07_corrected_panel.pkl")
PROTOCOL_PATH = RESEARCH_DIR / "model_v10_literature_protocol.json"
RESULT_PATH = RESEARCH_DIR / "model_v10_literature_result.json"
PICKS_PATH = Path("/tmp/model_v10_literature_picks.csv.gz")
HYPOTHESES_PATH = RESEARCH_DIR / "model_v10_literature_hypotheses.json"
LITERATURE_REPORT_PATH = RESEARCH_DIR / "model_v10_literature_review.md"

TRAIN_START = pd.Timestamp("2024-01-04")
SCORE_START = pd.Timestamp("2024-07-01")
SCORE_END = pd.Timestamp("2025-07-31")
PERIODS = {
    "discovery": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-31")),
    "confirmation_a": (pd.Timestamp("2024-11-01"), pd.Timestamp("2025-03-31")),
    "confirmation_b": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
    "overall": (SCORE_START, SCORE_END),
}
COSTS = (20.0, 40.0, 60.0)
CONTROL = (
    "oc_last", "oc_mean_5", "oc_mean_20", "oc_mean_60", "oc_win_20",
    "oc_std_20", "overnight_last", "overnight_mean_20", "overnight_mean_60",
    "night_day_corr_60", "xrank_atr14_pct", "xrank_close_momentum_5",
    "xrank_close_momentum_20", "xrank_close_momentum_60",
    "xrank_prior_close_location_20", "flat_oc_rate_20",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


def rolling_sum_by_code(
    values: pd.Series, codes: pd.Series, window: int, min_periods: int
) -> pd.Series:
    rolled = (
        values.groupby(codes, sort=False)
        .rolling(window=window, min_periods=min_periods)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return rolled.reindex(values.index)


def add_price_blocks(panel: pd.DataFrame) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    cc = pd.to_numeric(panel["cc_reversal_1_atr"], errors="coerce")
    neg = (-cc).clip(lower=0.0, upper=5.0)
    pos = cc.clip(lower=0.0, upper=5.0)
    panel["h01_neg_cc_shock"] = neg.astype("float32")
    panel["h01_pos_cc_shock"] = pos.astype("float32")
    panel["h01_neg_cc_shock_sq"] = (neg * neg).astype("float32")
    panel["h01_pos_cc_shock_sq"] = (pos * pos).astype("float32")
    panel["h01_neg_x_range_shock"] = (
        neg * pd.to_numeric(panel["range_shock_1_20"], errors="coerce")
    ).astype("float32")
    market = pd.to_numeric(panel["prior_market_cc_return_pct"], errors="coerce")
    market_nonnegative = pd.Series(
        np.where(market.notna(), market.ge(0).astype(float), np.nan),
        index=panel.index,
    )
    panel["h01_neg_x_nonnegative_market"] = (
        neg * market_nonnegative
    ).astype("float32")

    friction_inputs = panel.loc[:, [
        "no_trade_rate_20", "flat_oc_rate_20", "zero_range_rate_20"
    ]].apply(pd.to_numeric, errors="coerce")
    friction = friction_inputs.mean(axis=1, skipna=False)
    reversal = -pd.to_numeric(panel["oc_mean_5"], errors="coerce")
    panel["h02_reversal_5"] = reversal.astype("float32")
    panel["h02_activity_friction"] = friction.astype("float32")
    panel["h02_reversal_x_friction"] = (reversal * friction).astype("float32")
    panel["h02_reversal_x_market_vol"] = (
        reversal * pd.to_numeric(panel["market_cc_vol_ratio_5_20"], errors="coerce")
    ).astype("float32")

    codes = panel["code"]
    lag_oc = panel.groupby("code", sort=False)["oc_return_pct"].shift(1)
    lag_observed = (
        panel["outcome_observed"]
        .groupby(codes, sort=False)
        .shift(1)
        .fillna(False)
        .astype(bool)
    )
    overnight = pd.to_numeric(panel["overnight_last"], errors="coerce")
    joint_valid = lag_observed & lag_oc.notna() & overnight.notna()
    posneg = pd.Series(
        np.where(joint_valid, (overnight.gt(0) & lag_oc.lt(0)).astype(float), np.nan),
        index=panel.index,
        dtype=float,
    )
    negpos = pd.Series(
        np.where(joint_valid, (overnight.lt(0) & lag_oc.gt(0)).astype(float), np.nan),
        index=panel.index,
        dtype=float,
    )
    valid_numeric = joint_valid.astype(float)
    valid_numeric.loc[~joint_valid] = np.nan

    h03_features: list[str] = []
    valid_counts: dict[int, pd.Series] = {}
    for window, minimum in ((20, 8), (60, 20)):
        denominator = rolling_sum_by_code(
            joint_valid.astype(float), codes, window, minimum
        )
        # A window can meet min_periods through invalid rows; enforce the minimum
        # number of valid joint observations explicitly.
        denominator = denominator.where(denominator.ge(minimum))
        pn_num = rolling_sum_by_code(posneg.fillna(0.0), codes, window, minimum)
        np_num = rolling_sum_by_code(negpos.fillna(0.0), codes, window, minimum)
        pn_rate = pn_num.div(denominator)
        np_rate = np_num.div(denominator)
        valid_counts[window] = denominator
        if window == 20:
            panel["h03_posnight_negday_rate20"] = pn_rate.astype("float32")
            h03_features.append("h03_posnight_negday_rate20")
        balance_name = f"h03_negpos_balance{window}"
        panel[balance_name] = (np_rate - pn_rate).astype("float32")
        h03_features.append(balance_name)

    positive_overnight = pd.Series(
        np.where(overnight.notna(), overnight.gt(0).astype(float), np.nan),
        index=panel.index,
        dtype=float,
    )
    overnight_valid = overnight.notna()
    den20 = rolling_sum_by_code(
        overnight_valid.astype(float), codes, window=20, min_periods=8
    ).where(lambda x: x.ge(8))
    pos20 = rolling_sum_by_code(
        positive_overnight.fillna(0.0), codes, window=20, min_periods=8
    )
    positive_rate20 = pos20.div(den20)
    positive_bool = overnight.gt(0) & overnight.notna()
    run_break = (~positive_bool).groupby(codes, sort=False).cumsum()
    streak = (
        positive_bool.astype("int16")
        .groupby([codes, run_break], sort=False)
        .cumsum()
        .clip(upper=10)
    )
    streak = streak.where(overnight.notna())
    atr_rank_positive = pd.to_numeric(
        panel["xrank_atr14_pct"], errors="coerce"
    ).clip(lower=0.0, upper=1.0)
    attention = positive_rate20 * atr_rank_positive * friction
    positive_last = pd.Series(
        np.where(overnight.notna(), overnight.gt(0).astype(float), np.nan),
        index=panel.index,
    )
    panel["h04_positive_overnight_rate20"] = positive_rate20.astype("float32")
    panel["h04_positive_overnight_streak"] = streak.astype("float32")
    panel["h04_attention"] = attention.astype("float32")
    panel["h04_attention_x_positive_last"] = (
        attention * positive_last
    ).astype("float32")

    compare = lag_observed & lag_oc.notna() & panel["oc_last"].notna()
    max_lag_diff = float(
        np.abs(
            lag_oc.loc[compare].to_numpy(dtype=float)
            - panel.loc[compare, "oc_last"].to_numpy(dtype=float)
        ).max(initial=0.0)
    )
    code_date_monotonic = bool(
        panel[["code", "date"]]
        .sort_values(["code", "date"], kind="stable")
        .index.equals(panel.index)
    )
    qa = {
        "code_date_monotonic_before_shift": code_date_monotonic,
        "lag_oc_rows_compared_to_frozen_oc_last": int(compare.sum()),
        "lag_oc_max_abs_diff_vs_frozen_oc_last": max_lag_diff,
        "joint_valid_rows": int(joint_valid.sum()),
        "lag_outcome_missing_rows_excluded": int((~lag_observed | lag_oc.isna()).sum()),
        "lag_overnight_missing_rows_excluded": int(overnight.isna().sum()),
        "zero_lag_oc_valid_denominator_rows": int((joint_valid & lag_oc.eq(0)).sum()),
        "zero_lag_overnight_valid_denominator_rows": int(
            (joint_valid & overnight.eq(0)).sum()
        ),
        "h03_rate20_nonmissing_rows": int(
            panel["h03_posnight_negday_rate20"].notna().sum()
        ),
        "h03_rate60_nonmissing_rows": int(
            panel["h03_negpos_balance60"].notna().sum()
        ),
        "shift_strictly_one_exchange_session": max_lag_diff == 0.0,
        "no_trade_excluded_from_denominator": True,
        "zeros_retained_in_denominator": True,
    }
    blocks = {
        "H01": (
            "h01_neg_cc_shock", "h01_pos_cc_shock",
            "h01_neg_cc_shock_sq", "h01_pos_cc_shock_sq",
            "h01_neg_x_range_shock", "h01_neg_x_nonnegative_market",
        ),
        "H02": (
            "h02_reversal_5", "h02_activity_friction",
            "h02_reversal_x_friction", "h02_reversal_x_market_vol",
        ),
        "H03": tuple(h03_features),
        "H04": (
            "h04_positive_overnight_rate20",
            "h04_positive_overnight_streak",
            "h04_attention",
            "h04_attention_x_positive_last",
        ),
    }
    return blocks, qa


def _direction_flags(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    support_columns = [
        "tdnet_clean_revision_up_title",
        "tdnet_clean_dividend_up_title",
        "tdnet_clean_has_buyback_decision",
    ]
    adverse_columns = [
        "tdnet_clean_revision_down_title",
        "tdnet_clean_dividend_down_title",
        "tdnet_clean_has_external_equity_financing",
        "tdnet_clean_has_impairment_loss",
        "tdnet_clean_has_audit_problem",
    ]
    support_frame = panel.loc[:, support_columns].fillna(False).astype(bool)
    adverse_frame = panel.loc[:, adverse_columns].fillna(False).astype(bool)
    return (
        support_frame.any(axis=1),
        adverse_frame.any(axis=1),
        pd.concat([
            support_frame.astype("int8").add_prefix("support__"),
            adverse_frame.astype("int8").add_prefix("adverse__"),
        ], axis=1),
    )


def _earnings_wave_daily(panel: pd.DataFrame) -> pd.DataFrame:
    daily = (
        panel.groupby("date", sort=True)
        .agg(
            complete=("tdnet_source_complete", "first"),
            earnings_count=("tdnet_clean_has_earnings", "sum"),
        )
        .reset_index()
    )
    counts = daily["earnings_count"].to_numpy(dtype=float)
    complete = daily["complete"].to_numpy(dtype=bool)
    n = len(daily)
    starts = np.zeros(n, dtype=bool)
    active = np.zeros(n, dtype=bool)
    cumulative = np.zeros(n, dtype=float)
    days_since_start = np.full(n, np.nan)
    prior_median = np.full(n, np.nan)
    wave_id = np.zeros(n, dtype=int)
    completed_sizes: list[float] = []
    current_id = 0
    current_total = 0.0
    current_start = -1
    last_event = -10_000
    for i in range(n):
        if not complete[i]:
            # Incomplete market-level TDnet days cannot be used to advance or
            # terminate a point-in-time wave.
            continue
        if counts[i] > 0 and i - last_event > 10:
            if current_id > 0:
                completed_sizes.append(current_total)
            current_id += 1
            current_total = 0.0
            current_start = i
            starts[i] = True
        if current_id > 0:
            current_total += counts[i]
            wave_id[i] = current_id
            cumulative[i] = current_total
            days_since_start[i] = i - current_start
            active[i] = i - last_event <= 10 or counts[i] > 0
            if completed_sizes:
                prior_median[i] = float(np.median(completed_sizes))
        if counts[i] > 0:
            last_event = i
            active[i] = True
    ratio = np.divide(
        cumulative,
        prior_median,
        out=np.full(n, np.nan),
        where=np.isfinite(prior_median) & (prior_median > 0),
    )
    daily["h09_market_earnings_count"] = counts
    daily["h09_wave_start"] = starts
    daily["h09_wave_id"] = wave_id
    daily["h09_wave_active"] = active
    daily["h09_wave_cumulative_count"] = cumulative
    daily["h09_days_since_wave_start"] = days_since_start
    daily["h09_prior_completed_wave_median_size"] = prior_median
    daily["h09_wave_progress_ratio"] = ratio
    return daily


def add_tdnet_blocks(panel: pd.DataFrame) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    support, adverse, direction_frame = _direction_flags(panel)
    any_event = panel["tdnet_clean_any"].fillna(False).astype(bool)
    docs = np.expm1(
        pd.to_numeric(panel["tdnet_clean_document_count_log1p"], errors="coerce")
    ).round().clip(lower=0)
    daily_counts = (
        pd.DataFrame({
            "date": panel["date"],
            "event": any_event.astype(int),
            "docs": docs.fillna(0.0),
            "earnings": panel["tdnet_clean_has_earnings"].fillna(False).astype(int),
        })
        .groupby("date", sort=True)
        .sum()
        .rename(columns={
            "event": "market_event_codes",
            "docs": "market_document_count",
            "earnings": "market_earnings_codes",
        })
    )
    for col in daily_counts.columns:
        mapped = panel["date"].map(np.log1p(daily_counts[col]))
        panel[f"h06_log1p_{col}"] = mapped.astype("float32")
    panel["h06_supportive"] = support.astype("float32")
    panel["h06_adverse"] = adverse.astype("float32")
    panel["h06_support_x_market_events"] = (
        support.astype(float) * panel["h06_log1p_market_event_codes"]
    ).astype("float32")
    panel["h06_adverse_x_market_events"] = (
        adverse.astype(float) * panel["h06_log1p_market_event_codes"]
    ).astype("float32")
    panel["h06_support_x_market_earnings"] = (
        support.astype(float) * panel["h06_log1p_market_earnings_codes"]
    ).astype("float32")
    panel["h06_adverse_x_market_earnings"] = (
        adverse.astype(float) * panel["h06_log1p_market_earnings_codes"]
    ).astype("float32")

    timestamp = pd.to_datetime(
        panel["tdnet_clean_feature_source_max_timestamp"], errors="coerce"
    )
    source_friday = (timestamp.dt.dayofweek == 4) & any_event
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    gaps = pd.Series(sessions, index=sessions).diff().dt.days
    post_long_holiday_by_date = gaps.ge(4)
    post_long_holiday = panel["date"].map(post_long_holiday_by_date).fillna(False)
    weekend = panel["tdnet_clean_weekend_age"].fillna(False).astype(bool) & any_event
    age = pd.to_numeric(
        panel["tdnet_clean_latest_age_hours_log1p"], errors="coerce"
    )
    panel["h07_source_friday"] = source_friday.astype("float32")
    panel["h07_weekend_age"] = weekend.astype("float32")
    panel["h07_post_long_holiday_event"] = (
        post_long_holiday.astype(bool) & any_event
    ).astype("float32")
    panel["h07_log_age_x_support"] = (age * support.astype(float)).astype("float32")
    panel["h07_log_age_x_adverse"] = (age * adverse.astype(float)).astype("float32")
    panel["h07_friday_x_support"] = (
        source_friday.astype(float) * support.astype(float)
    ).astype("float32")
    panel["h07_friday_x_adverse"] = (
        source_friday.astype(float) * adverse.astype(float)
    ).astype("float32")

    supportive_count = direction_frame.filter(like="support__").sum(axis=1)
    adverse_count = direction_frame.filter(like="adverse__").sum(axis=1)
    panel["h08_supportive_count"] = supportive_count.astype("float32")
    panel["h08_adverse_count"] = adverse_count.astype("float32")
    panel["h08_supportive_corroboration"] = supportive_count.ge(2).astype("float32")
    panel["h08_adverse_corroboration"] = adverse_count.ge(2).astype("float32")
    panel["h08_directional_conflict"] = (
        support & adverse
    ).astype("float32")
    panel["h08_directional_net"] = (
        supportive_count - adverse_count
    ).astype("float32")
    panel["h08_directional_width"] = (
        supportive_count + adverse_count
    ).astype("float32")

    wave = _earnings_wave_daily(panel)
    wave_index = wave.set_index("date")
    own_earnings = panel["tdnet_clean_has_earnings"].fillna(False).astype(bool)
    ratio = panel["date"].map(wave_index["h09_wave_progress_ratio"])
    days_since = panel["date"].map(wave_index["h09_days_since_wave_start"])
    current_count = panel["date"].map(wave_index["h09_market_earnings_count"])
    prior_median = panel["date"].map(
        wave_index["h09_prior_completed_wave_median_size"]
    )
    panel["h09_earnings_event"] = own_earnings.astype("float32")
    panel["h09_earnings_x_market_count"] = (
        own_earnings.astype(float) * np.log1p(current_count)
    ).astype("float32")
    panel["h09_earnings_x_wave_progress"] = (
        own_earnings.astype(float) * ratio
    ).astype("float32")
    panel["h09_earnings_x_days_since_start"] = (
        own_earnings.astype(float) * days_since
    ).astype("float32")
    panel["h09_earnings_early"] = (
        own_earnings & ratio.notna() & ratio.le(0.5)
    ).astype("float32")
    panel["h09_earnings_late"] = (
        own_earnings & ratio.notna() & ratio.gt(0.5)
    ).astype("float32")
    panel["h09_earnings_x_prior_wave_size"] = (
        own_earnings.astype(float) * np.log1p(prior_median)
    ).astype("float32")

    complete_by_date = panel.groupby("date", sort=True)["tdnet_source_complete"].first()
    complete_dates = complete_by_date.index[complete_by_date]
    qa = {
        "complete_dates": int(len(complete_dates)),
        "first_complete_date": str(complete_dates.min().date()),
        "last_complete_date": str(complete_dates.max().date()),
        "incomplete_dates": int((~complete_by_date).sum()),
        "source_complete_constant_within_date": bool(
            panel.groupby("date")["tdnet_source_complete"].nunique().max() == 1
        ),
        "market_count_features_constant_within_date": bool(
            panel.groupby("date")["h06_log1p_market_event_codes"].nunique().max() == 1
        ),
        "waves_seen": int(wave["h09_wave_id"].max()),
        "completed_wave_denominator_available_dates": int(
            wave["h09_prior_completed_wave_median_size"].notna().sum()
        ),
        "h09_definition": {
            "new_wave": "earnings_count>0 after more than 10 exchange sessions since the prior earnings event",
            "progress": "point-in-time cumulative current-wave earnings-code count divided by median final size of previously completed waves",
            "incomplete_dates": "do not advance, start, or terminate a wave",
        },
    }
    blocks = {
        "H06": (
            "h06_log1p_market_event_codes", "h06_log1p_market_document_count",
            "h06_log1p_market_earnings_codes", "h06_supportive", "h06_adverse",
            "h06_support_x_market_events", "h06_adverse_x_market_events",
            "h06_support_x_market_earnings", "h06_adverse_x_market_earnings",
        ),
        "H07": (
            "h07_source_friday", "h07_weekend_age",
            "h07_post_long_holiday_event", "h07_log_age_x_support",
            "h07_log_age_x_adverse", "h07_friday_x_support",
            "h07_friday_x_adverse",
        ),
        "H08": (
            "h08_supportive_count", "h08_adverse_count",
            "h08_supportive_corroboration", "h08_adverse_corroboration",
            "h08_directional_conflict", "h08_directional_net",
            "h08_directional_width",
        ),
        "H09": (
            "h09_earnings_event", "h09_earnings_x_market_count",
            "h09_earnings_x_wave_progress", "h09_earnings_x_days_since_start",
            "h09_earnings_early", "h09_earnings_late",
            "h09_earnings_x_prior_wave_size",
        ),
    }
    return blocks, qa


def monthly_windows() -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    result = []
    periods = pd.period_range(
        SCORE_START.to_period("M"), SCORE_END.to_period("M"), freq="M"
    )
    for period in periods:
        result.append((
            max(SCORE_START, period.start_time.normalize()),
            min(SCORE_END, period.end_time.normalize()),
            str(period),
        ))
    return result


def date_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("date", sort=False)["date"].transform("size")
    values = 1.0 / counts.to_numpy(dtype=float)
    return values


def fit_model(training: pd.DataFrame, columns: tuple[str, ...]) -> Pipeline:
    y = (
        training["oc_return_pct"]
        .groupby(training["date"], sort=False)
        .rank(method="average", pct=True)
        .mul(2.0)
        .sub(1.0)
        .to_numpy(dtype=float)
    )
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    model.fit(
        training.loc[:, list(columns)],
        y,
        ridge__sample_weight=date_equal_weights(training),
    )
    return model


def walk_forward(
    panel: pd.DataFrame,
    blocks: dict[str, tuple[str, ...]],
    *,
    family: str,
    tdnet_complete_only: bool,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    recipes = {"control": CONTROL}
    recipes.update({
        block_id: CONTROL + block_features
        for block_id, block_features in blocks.items()
    })
    picks_parts: dict[str, list[pd.DataFrame]] = {k: [] for k in recipes}
    fold_qa: list[dict[str, Any]] = []
    union_columns = sorted(set().union(*recipes.values()))
    keep = [
        "date", "code", "name", "oc_return_pct",
        "price_eligible", "price_training_eligible", "tdnet_source_complete",
        *union_columns,
    ]
    for start, end, fold in monthly_windows():
        train_mask = (
            panel["date"].between(TRAIN_START, start - pd.Timedelta(days=1))
            & panel["price_training_eligible"].eq(True)
            & panel["oc_return_pct"].notna()
        )
        score_mask = (
            panel["date"].between(start, end)
            & panel["price_eligible"].eq(True)
        )
        if tdnet_complete_only:
            train_mask &= panel["tdnet_source_complete"].eq(True)
            score_mask &= panel["tdnet_source_complete"].eq(True)
        training = panel.loc[train_mask, keep].copy()
        scoring = panel.loc[score_mask, keep].copy()
        train_sessions = int(training["date"].nunique())
        if train_sessions < 60:
            raise RuntimeError(f"{family}/{fold}: only {train_sessions} training dates")
        score_sessions = int(scoring["date"].nunique())
        fold_entry = {
            "fold": fold,
            "train_start": str(training["date"].min().date()),
            "train_end": str(training["date"].max().date()),
            "train_sessions": train_sessions,
            "train_rows": int(len(training)),
            "score_sessions": score_sessions,
            "score_rows": int(len(scoring)),
        }
        if score_sessions == 0:
            fold_entry["status"] = "no_source_complete_score_sessions"
            fold_qa.append(fold_entry)
            print(f"{family}: skipped {fold} (no score sessions)", flush=True)
            del training, scoring
            gc.collect()
            continue
        for recipe, columns in recipes.items():
            estimator = fit_model(training, columns)
            values = np.asarray(
                estimator.predict(scoring.loc[:, list(columns)]), dtype=float
            )
            if not np.isfinite(values).all():
                raise RuntimeError(f"{family}/{fold}/{recipe}: non-finite score")
            ranked = scoring.loc[:, [
                "date", "code", "name", "oc_return_pct"
            ]].copy()
            ranked["model_score"] = values
            ranked = (
                ranked.sort_values(
                    ["date", "model_score", "code"],
                    ascending=[True, False, True],
                    kind="stable",
                )
                .groupby("date", sort=True, as_index=False)
                .head(2)
                .copy()
            )
            ranked["model_rank"] = (
                ranked.groupby("date", sort=False).cumcount() + 1
            )
            ranked["family"] = family
            ranked["recipe"] = recipe
            picks_parts[recipe].append(ranked)
            del estimator, values, ranked
            gc.collect()
        fold_entry["status"] = "completed"
        fold_qa.append(fold_entry)
        print(
            f"{family}: completed {fold} ({score_sessions} score sessions)",
            flush=True,
        )
        del training, scoring
        gc.collect()
    picks = {
        recipe: pd.concat(parts, ignore_index=True)
        for recipe, parts in picks_parts.items()
    }
    return picks, fold_qa


def selected_daily(
    picks: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    top_k: int,
    cost_bps: float,
    cash_codes: set[str] | None = None,
) -> pd.Series:
    chosen = picks.loc[picks["model_rank"].le(top_k)].copy()
    executed = chosen["oc_return_pct"].notna() & chosen["code"].notna()
    if cash_codes:
        executed &= ~chosen["code"].astype(str).isin(cash_codes)
    contribution = np.where(
        executed,
        (
            chosen["oc_return_pct"].to_numpy(dtype=float)
            - float(cost_bps) / 100.0
        ) / top_k,
        0.0,
    )
    daily = pd.Series(
        contribution, index=pd.to_datetime(chosen["date"]), dtype=float
    ).groupby(level=0).sum()
    return daily.reindex(sessions, fill_value=0.0).sort_index()


def top_profit_codes(
    picks: pd.DataFrame, *, top_k: int, cost_bps: float, count: int = 10
) -> list[str]:
    chosen = picks.loc[
        picks["model_rank"].le(top_k)
        & picks["oc_return_pct"].notna()
        & picks["code"].notna()
    ].copy()
    chosen["contribution"] = (
        chosen["oc_return_pct"].astype(float) - cost_bps / 100.0
    ) / top_k
    by_code = chosen.groupby(chosen["code"].astype(str))["contribution"].sum()
    return [str(x) for x in by_code.nlargest(min(count, len(by_code))).index]


def portfolio_summary(
    picks: pd.DataFrame, sessions: pd.DatetimeIndex
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sessions": int(len(sessions)),
        "displayed_rank1": int(picks["model_rank"].eq(1).sum()),
        "displayed_rank2": int(picks["model_rank"].eq(2).sum()),
        "portfolios": {},
    }
    for top_k in (1, 2):
        portfolio: dict[str, Any] = {"costs": {}}
        gross = selected_daily(
            picks, sessions, top_k=top_k, cost_bps=0.0
        )
        portfolio["gross_mean_pct"] = scalar(gross.mean())
        portfolio["gross_median_pct"] = scalar(gross.median())
        for cost in COSTS:
            daily = selected_daily(
                picks, sessions, top_k=top_k, cost_bps=cost
            )
            portfolio["costs"][str(int(cost))] = {
                "mean_pct": scalar(daily.mean()),
                "median_pct": scalar(daily.median()),
                "win_rate": scalar(daily.gt(0).mean()),
                "std_pct": scalar(daily.std(ddof=1)),
            }
        daily20 = selected_daily(
            picks, sessions, top_k=top_k, cost_bps=20.0
        )
        monthly = daily20.groupby(daily20.index.to_period("M")).mean()
        top_codes = top_profit_codes(
            picks, top_k=top_k, cost_bps=20.0, count=10
        )
        cash_daily = selected_daily(
            picks,
            sessions,
            top_k=top_k,
            cost_bps=20.0,
            cash_codes=set(top_codes),
        )
        keep_after_best20 = daily20.drop(
            daily20.nlargest(min(20, len(daily20))).index
        )
        portfolio.update({
            "monthly_net20_pct": {
                str(k): scalar(v) for k, v in monthly.items()
            },
            "positive_months": int(monthly.gt(0).sum()),
            "months": int(len(monthly)),
            "best20_days_removed_net20_mean_pct": scalar(
                keep_after_best20.mean()
            ),
            "top10_profit_codes": top_codes,
            "top10_profit_codes_set_to_cash_net20_mean_pct": scalar(
                cash_daily.mean()
            ),
        })
        result["portfolios"][f"top{top_k}"] = portfolio
    return result


def slice_period(
    picks: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    return picks.loc[picks["date"].between(start, end)].copy()


def period_sessions(
    all_sessions: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DatetimeIndex:
    return all_sessions[(all_sessions >= start) & (all_sessions <= end)]


def candidate_report(
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    all_sessions: pd.DatetimeIndex,
) -> dict[str, Any]:
    report: dict[str, Any] = {"periods": {}}
    for period, (start, end) in PERIODS.items():
        sessions = period_sessions(all_sessions, start, end)
        base_period = slice_period(base, start, end)
        candidate_period = slice_period(candidate, start, end)
        base_daily = selected_daily(
            base_period, sessions, top_k=2, cost_bps=20.0
        )
        candidate_daily = selected_daily(
            candidate_period, sessions, top_k=2, cost_bps=20.0
        )
        delta = candidate_daily - base_daily
        report["periods"][period] = {
            "metrics": portfolio_summary(candidate_period, sessions),
            "top2_net20_uplift_vs_control_pct": scalar(delta.mean()),
            "top2_net20_uplift_median_pct": scalar(delta.median()),
            "top2_net20_uplift_positive_days": scalar(delta.gt(0).mean()),
        }
    subperiod_positive = []
    for period in ("discovery", "confirmation_a", "confirmation_b"):
        value = report["periods"][period]["metrics"]["portfolios"]["top2"][
            "costs"
        ]["20"]["mean_pct"]
        subperiod_positive.append(value is not None and value > 0)
    overall = report["periods"]["overall"]["metrics"]["portfolios"]["top2"]
    report["retrospective_decision_checks"] = {
        "top2_net20_mean_exceeds_control": (
            report["periods"]["overall"]["top2_net20_uplift_vs_control_pct"] > 0
        ),
        "all_three_subperiod_top2_net20_positive": all(subperiod_positive),
        "best20_removed_top2_net20_positive": (
            overall["best20_days_removed_net20_mean_pct"] is not None
            and overall["best20_days_removed_net20_mean_pct"] > 0
        ),
        "top10_profit_codes_cash_top2_net20_positive": (
            overall["top10_profit_codes_set_to_cash_net20_mean_pct"] is not None
            and overall["top10_profit_codes_set_to_cash_net20_mean_pct"] > 0
        ),
    }
    return report


def circular_indices(
    n: int, block: int, samples: int, rng: np.random.Generator
) -> np.ndarray:
    blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n, size=(samples, blocks), endpoint=False)
    offsets = np.arange(block, dtype=int)
    return ((starts[:, :, None] + offsets) % n).reshape(samples, -1)[:, :n]


def familywise_bootstrap(
    base: pd.DataFrame,
    candidates: dict[str, pd.DataFrame],
    all_sessions: pd.DatetimeIndex,
    *,
    top_k: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for period_number, (period, (start, end)) in enumerate(PERIODS.items()):
        sessions = period_sessions(all_sessions, start, end)
        base_period = slice_period(base, start, end)
        base_daily = selected_daily(
            base_period, sessions, top_k=top_k, cost_bps=20.0
        )
        ids = list(candidates)
        matrix = np.column_stack([
            (
                selected_daily(
                    slice_period(candidates[candidate_id], start, end),
                    sessions,
                    top_k=top_k,
                    cost_bps=20.0,
                )
                - base_daily
            ).to_numpy(dtype=float)
            for candidate_id in ids
        ])
        n = len(sessions)
        if n == 0:
            report[period] = {"sessions": 0, "status": "no_complete_sessions"}
            continue
        block = min(5, n)
        rng = np.random.default_rng(20260723 + period_number)
        indices = circular_indices(n, block, 10000, rng)
        observed = matrix.mean(axis=0)
        boot_means = matrix[indices].mean(axis=1)
        se = boot_means.std(axis=0, ddof=1)
        safe_se = np.where(se > 0, se, np.nan)
        centered_t = (boot_means - observed[None, :]) / safe_se[None, :]
        max_t = np.nanmax(centered_t, axis=1)
        critical = float(np.nanquantile(max_t, 0.90))
        entries: dict[str, Any] = {}
        for j, candidate_id in enumerate(ids):
            lower = observed[j] - critical * se[j] if se[j] > 0 else observed[j]
            unadjusted_lower = float(np.quantile(boot_means[:, j], 0.10))
            observed_t = observed[j] / se[j] if se[j] > 0 else np.nan
            adjusted_p = (
                float(np.mean(max_t >= observed_t))
                if np.isfinite(observed_t) else None
            )
            entries[candidate_id] = {
                "mean_uplift_pct": scalar(observed[j]),
                "bootstrap_se_pct": scalar(se[j]),
                "unadjusted_one_sided_90pct_lower_pct": scalar(unadjusted_lower),
                "familywise_one_sided_90pct_lower_pct": scalar(lower),
                "max_t_adjusted_one_sided_p": scalar(adjusted_p),
                "familywise_lower_nonnegative": bool(lower >= 0),
            }
        report[period] = {
            "sessions": int(n),
            "block_length": int(block),
            "samples": 10000,
            "confidence": 0.90,
            "max_t_critical": critical,
            "low_power_warning": bool(n < 20),
            "candidates": entries,
        }
    return report


def analyze_family(
    picks: dict[str, pd.DataFrame],
    folds: list[dict[str, Any]],
    *,
    family: str,
) -> dict[str, Any]:
    base = picks["control"]
    sessions = pd.DatetimeIndex(sorted(base["date"].unique())).normalize()
    result = {
        "family": family,
        "score_sessions": int(len(sessions)),
        "first_score_session": str(sessions.min().date()),
        "last_score_session": str(sessions.max().date()),
        "folds": folds,
        "control": {"periods": {}},
        "candidates": {},
    }
    for period, (start, end) in PERIODS.items():
        subset_sessions = period_sessions(sessions, start, end)
        result["control"]["periods"][period] = portfolio_summary(
            slice_period(base, start, end), subset_sessions
        )
    candidates = {k: v for k, v in picks.items() if k != "control"}
    for candidate_id, candidate_picks in candidates.items():
        result["candidates"][candidate_id] = candidate_report(
            base, candidate_picks, sessions
        )
    result["familywise_block_bootstrap"] = {
        "top1_net20": familywise_bootstrap(
            base, candidates, sessions, top_k=1
        ),
        "top2_net20": familywise_bootstrap(
            base, candidates, sessions, top_k=2
        ),
    }
    for candidate_id in candidates:
        familywise_lower = result["familywise_block_bootstrap"][
            "top2_net20"
        ]["overall"]["candidates"][candidate_id][
            "familywise_one_sided_90pct_lower_pct"
        ]
        result["candidates"][candidate_id]["retrospective_decision_checks"][
            "familywise_overall_lower_nonnegative"
        ] = familywise_lower >= 0
        checks = result["candidates"][candidate_id][
            "retrospective_decision_checks"
        ]
        checks["all_research_and_confirmatory_checks"] = all(checks.values())
    return result


def tdnet_timestamp_cutoff_qa(panel: pd.DataFrame) -> dict[str, Any]:
    timestamp = pd.to_datetime(
        panel["tdnet_clean_feature_source_max_timestamp"], errors="coerce"
    )
    target_cutoff = (
        pd.to_datetime(panel["date"]).dt.tz_localize("Asia/Tokyo")
        + pd.Timedelta(hours=8, minutes=59)
    )
    observed = timestamp.notna()
    seconds_before = (
        target_cutoff.loc[observed] - timestamp.loc[observed]
    ).dt.total_seconds()
    return {
        "timestamp_rows": int(observed.sum()),
        "all_feature_timestamps_strictly_before_085900_jst": bool(
            (seconds_before > 0).all()
        ),
        "minimum_seconds_before_085900_jst": scalar(seconds_before.min()),
        "violations": int((seconds_before <= 0).sum()),
    }


def reanalyze_from_picks() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    picks_frame = pd.read_csv(
        PICKS_PATH, parse_dates=["date"], dtype={"code": str}
    )
    for result_key, family_name in (
        ("price_family", "price_H01_H04"),
        ("tdnet_family", "tdnet_H06_H09_complete_dates"),
    ):
        family_frame = picks_frame.loc[picks_frame["family"].eq(family_name)]
        picks = {
            recipe: subset.copy()
            for recipe, subset in family_frame.groupby("recipe", sort=False)
        }
        folds = result[result_key]["folds"]
        result[result_key] = analyze_family(
            picks, folds, family=family_name
        )
    panel = joblib.load(PANEL_PATH, mmap_mode="r")
    result["source_qa"]["tdnet_timestamp_cutoff"] = tdnet_timestamp_cutoff_qa(
        panel
    )
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=scalar) + "\n",
        encoding="utf-8",
    )
    print(f"reanalyzed {RESULT_PATH} from {PICKS_PATH}", flush=True)


def main() -> None:
    protocol_sha = sha256(PROTOCOL_PATH)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    panel_sha = sha256(PANEL_PATH)
    if panel_sha != protocol["panel"]["expected_sha256"]:
        raise RuntimeError(
            f"panel SHA mismatch: expected {protocol['panel']['expected_sha256']}, got {panel_sha}"
        )
    panel = joblib.load(PANEL_PATH, mmap_mode="r")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype(str)
    if not panel[["code", "date"]].sort_values(
        ["code", "date"], kind="stable"
    ).index.equals(panel.index):
        raise RuntimeError("frozen panel is not sorted code/date")

    price_blocks, h03_qa = add_price_blocks(panel)
    tdnet_blocks, tdnet_qa = add_tdnet_blocks(panel)
    expected_price = set(protocol["price_blocks"])
    if set(price_blocks) != expected_price:
        raise RuntimeError("implemented price blocks differ from protocol")
    expected_tdnet = {"H06", "H07", "H08", "H09"}
    if set(tdnet_blocks) != expected_tdnet:
        raise RuntimeError("implemented TDnet blocks differ from protocol")

    price_picks, price_folds = walk_forward(
        panel,
        price_blocks,
        family="price_H01_H04",
        tdnet_complete_only=False,
    )
    price_result = analyze_family(
        price_picks, price_folds, family="price_H01_H04"
    )
    tdnet_picks, tdnet_folds = walk_forward(
        panel,
        tdnet_blocks,
        family="tdnet_H06_H09_complete_dates",
        tdnet_complete_only=True,
    )
    tdnet_result = analyze_family(
        tdnet_picks, tdnet_folds, family="tdnet_H06_H09_complete_dates"
    )

    all_picks = []
    for registry in (price_picks, tdnet_picks):
        all_picks.extend(registry.values())
    pd.concat(all_picks, ignore_index=True).to_csv(
        PICKS_PATH, index=False, compression="gzip"
    )

    result = {
        "schema_version": 1,
        "repository_artifact": "research/model_v10_literature_result.json",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "panel_sha256": panel_sha,
        "authority": "retrospective_research_only",
        "production_promotion_allowed": False,
        "generated_at_jst": "2026-07-23",
        "source_hypotheses_sha256": sha256(HYPOTHESES_PATH),
        "source_literature_report_sha256": sha256(LITERATURE_REPORT_PATH),
        "source_qa": {
            "rows": int(len(panel)),
            "dates": int(panel["date"].nunique()),
            "first_date": str(panel["date"].min().date()),
            "last_date": str(panel["date"].max().date()),
            "candidate_price_source_strictly_prior": bool(
                (
                    pd.to_datetime(panel["candidate_price_source_max_date"]).isna()
                    | pd.to_datetime(panel["candidate_price_source_max_date"]).lt(
                        panel["date"]
                    )
                ).all()
            ),
            "h03": h03_qa,
            "tdnet": tdnet_qa,
            "tdnet_timestamp_cutoff": tdnet_timestamp_cutoff_qa(panel),
        },
        "feature_blocks": {
            "price": {k: list(v) for k, v in price_blocks.items()},
            "tdnet": {k: list(v) for k, v in tdnet_blocks.items()},
        },
        "price_family": price_result,
        "tdnet_family": tdnet_result,
        "artifacts": {
            "picks_csv_gz": str(PICKS_PATH),
        },
    }
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=scalar) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {RESULT_PATH}", flush=True)
    print(f"wrote {PICKS_PATH}", flush=True)


if __name__ == "__main__":
    if "--reanalyze-only" in sys.argv:
        reanalyze_from_picks()
    else:
        main()
