#!/usr/bin/env python3
"""Evaluate a strictly-prior-beta market-residual ranking target."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/model_v10_market_residual_protocol.json"
PANEL = Path("/tmp/model_v07_corrected_panel.pkl")
OUTPUT = ROOT / "research/model_v10_market_residual_result.json"
PICKS_OUTPUT = ROOT / "research/model_v10_market_residual_picks.csv"
PANEL_SHA256 = "6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb"
TRAIN_START = pd.Timestamp("2024-01-04")
SCORE_START = pd.Timestamp("2024-07-01")
SCORE_END = pd.Timestamp("2025-07-31")
FEATURES = (
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
    "discovery": ("2024-07-01", "2024-10-31"),
    "confirmation_a": ("2024-11-01", "2025-03-31"),
    "confirmation_b": ("2025-04-01", "2025-07-31"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def monthly_periods() -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    periods = []
    for period in pd.period_range(
        SCORE_START.to_period("M"), SCORE_END.to_period("M"), freq="M"
    ):
        periods.append((
            max(SCORE_START, period.start_time.normalize()),
            min(SCORE_END, period.end_time.normalize()),
            str(period),
        ))
    return periods


def add_residual_target(panel: pd.DataFrame) -> dict[str, Any]:
    observed = (
        panel["price_training_eligible"].eq(True)
        & panel["oc_return_pct"].notna()
    )
    eligible_return = panel["oc_return_pct"].where(observed)
    market = (
        eligible_return.groupby(panel["date"], sort=True)
        .median()
    )
    market_row = panel["date"].map(market).astype(float)
    paired_market = market_row.where(observed)
    paired_stock = eligible_return

    codes = panel["code"].astype(str)
    lagged = pd.DataFrame({
        "x": paired_market.groupby(codes, sort=False).shift(1),
        "y": paired_stock.groupby(codes, sort=False).shift(1),
    })
    lagged["xy"] = lagged["x"] * lagged["y"]
    lagged["xx"] = lagged["x"] * lagged["x"]
    moments = (
        lagged.groupby(codes, sort=False)
        .rolling(60, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    covariance = moments["xy"] - moments["x"] * moments["y"]
    variance = moments["xx"] - moments["x"].pow(2)
    beta = (covariance / variance.where(variance.gt(1e-10))).clip(-3.0, 3.0)
    panel["v10_prior_market_beta_60"] = beta.astype("float32")
    beta_for_target = beta.fillna(1.0)
    panel["v10_market_oc_return_pct"] = market_row.astype("float32")
    panel["v10_residual_oc_return_pct"] = (
        panel["oc_return_pct"] - beta_for_target * market_row
    ).astype("float32")
    audit = {
        "market_sessions": int(market.notna().sum()),
        "market_return_min_pct": float(market.min()),
        "market_return_max_pct": float(market.max()),
        "beta_observed_rows": int(beta.notna().sum()),
        "beta_min": float(beta.min()),
        "beta_median": float(beta.median()),
        "beta_max": float(beta.max()),
        "beta_window": 60,
        "beta_minimum": 20,
        "beta_shifted_code_sessions": 1,
    }
    del lagged, moments, covariance, variance
    gc.collect()
    return audit


def target_by_date(training: pd.DataFrame, source: str) -> pd.Series:
    return (
        training[source]
        .groupby(training["date"], sort=False)
        .rank(method="average", pct=True)
        .mul(2.0)
        .sub(1.0)
    )


def date_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("date", sort=False)["date"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    dates = frame["date"].reset_index(drop=True)
    totals = pd.Series(weights).groupby(dates, sort=False).sum()
    if not np.allclose(totals.to_numpy(), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("date weights do not sum to one")
    return weights


def fit_model(training: pd.DataFrame, target_column: str) -> Pipeline:
    estimator = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    estimator.fit(
        training.loc[:, list(FEATURES)],
        target_by_date(training, target_column),
        model__sample_weight=date_equal_weights(training),
    )
    return estimator


def score_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    keep = [
        "date", "code", "name", "label", "oc_return_pct",
        "price_eligible", "price_training_eligible",
        "v10_residual_oc_return_pct", *FEATURES,
    ]
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    recipes = {
        "R00_daily_raw_return_rank_control": "oc_return_pct",
        "R01_prior_beta_market_residual_rank": "v10_residual_oc_return_pct",
    }
    for start, end, fold_id in monthly_periods():
        train_mask = (
            panel["date"].between(TRAIN_START, start - pd.Timedelta(days=1))
            & panel["price_training_eligible"].eq(True)
            & panel["oc_return_pct"].notna()
        )
        score_mask = (
            panel["date"].between(start, end)
            & panel["price_eligible"].eq(True)
        )
        training = panel.loc[train_mask, keep].copy()
        scoring = panel.loc[score_mask, keep].copy()
        if training["date"].max() >= scoring["date"].min():
            raise AssertionError("training dates overlap score dates")
        fold = {
            "fold": fold_id,
            "train_start": str(training["date"].min().date()),
            "train_end": str(training["date"].max().date()),
            "score_start": str(scoring["date"].min().date()),
            "score_end": str(scoring["date"].max().date()),
            "train_rows": int(len(training)),
            "score_rows": int(len(scoring)),
        }
        for recipe_id, target_column in recipes.items():
            estimator = fit_model(training, target_column)
            scores = estimator.predict(scoring.loc[:, list(FEATURES)])
            if not np.isfinite(scores).all():
                raise AssertionError(f"{recipe_id} emitted non-finite scores")
            part = scoring.loc[
                :, ["date", "code", "name", "label", "oc_return_pct"]
            ].copy()
            part["model_score"] = scores
            part["recipe_id"] = recipe_id
            ranked = part.sort_values(
                ["date", "model_score", "code"],
                ascending=[True, False, True],
                kind="stable",
            )
            ranked = ranked.groupby("date", sort=True).head(2).copy()
            ranked["model_rank"] = (
                ranked.groupby("date", sort=False).cumcount() + 1
            )
            parts.append(ranked)
            del estimator, part, ranked
            gc.collect()
        folds.append(fold)
        del training, scoring
        gc.collect()
    picks = pd.concat(parts, ignore_index=True)
    if picks.duplicated(["recipe_id", "date", "model_rank"]).any():
        raise AssertionError("duplicate recipe/date/rank rows")
    return picks, folds


def daily_returns(frame: pd.DataFrame, slots: int, cost_bps: float) -> pd.Series:
    selected = frame.loc[frame["model_rank"].le(slots)].copy()
    observed = selected["oc_return_pct"].notna().astype(float)
    slot = (
        selected["oc_return_pct"].fillna(0.0)
        - observed * cost_bps / 100.0
    )
    return slot.groupby(selected["date"], sort=True).sum().div(float(slots))


def summarize(
    frame: pd.DataFrame,
    control: pd.DataFrame,
    slots: int,
) -> dict[str, Any]:
    daily = {
        cost: daily_returns(frame, slots, cost)
        for cost in (20, 40, 60)
    }
    best20 = daily[20].nlargest(min(20, len(daily[20]))).index
    monthly = daily[40].groupby(daily[40].index.to_period("M")).mean()
    slices = {}
    for name, (start, end) in PERIODS.items():
        values = daily[20].loc[daily[20].index.to_series().between(start, end)]
        slices[name] = float(values.mean())

    selected = frame.loc[frame["model_rank"].le(slots)].copy()
    slot_profit = (
        selected["oc_return_pct"].fillna(0.0)
        - selected["oc_return_pct"].notna().astype(float) * 0.20
    ) / float(slots)
    code_profit = slot_profit.groupby(selected["code"].fillna("__cash__")).sum()
    top_codes = (
        code_profit.drop(labels="__cash__", errors="ignore")
        .sort_values(ascending=False)
        .head(10)
        .index
    )
    retained = selected.loc[~selected["code"].isin(top_codes)]
    retained_daily = (
        daily_returns(retained, slots, 20)
        .reindex(daily[20].index, fill_value=0.0)
    )

    control_selected = control.loc[control["model_rank"].le(slots)]
    own_sets = selected.groupby("date")["code"].apply(
        lambda values: frozenset(values.dropna().astype(str))
    )
    control_sets = control_selected.groupby("date")["code"].apply(
        lambda values: frozenset(values.dropna().astype(str))
    )
    overlap = np.mean([
        len(own_sets.loc[date] & control_sets.loc[date]) / float(slots)
        for date in own_sets.index
    ])
    return {
        "slots": slots,
        "mean_pct": {str(cost): float(values.mean()) for cost, values in daily.items()},
        "median_net40_pct": float(daily[40].median()),
        "positive_months_net40": int(monthly.gt(0).sum()),
        "monthly_net40_pct": {
            str(period): float(value) for period, value in monthly.items()
        },
        "temporal_slices_net20_pct": slices,
        "best_20_days_removed_net20_pct": float(daily[20].drop(best20).mean()),
        "top_10_profit_codes_to_cash_net20_pct": float(retained_daily.mean()),
        "mean_slot_overlap_with_control": float(overlap),
        "unique_codes": int(selected["code"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--picks-output", type=Path, default=PICKS_OUTPUT)
    args = parser.parse_args()
    if sha256_file(args.panel) != PANEL_SHA256:
        raise ValueError("frozen panel SHA-256 mismatch")
    panel = joblib.load(args.panel, mmap_mode="r").copy()
    panel["date"] = pd.to_datetime(panel["date"])
    if not panel[["code", "date"]].equals(
        panel[["code", "date"]]
        .sort_values(["code", "date"], kind="stable")
        .reset_index(drop=True)
    ):
        raise ValueError("panel must be sorted by code/date")
    beta_audit = add_residual_target(panel)
    picks, folds = score_models(panel)
    score_sessions = int(picks["date"].nunique())
    if score_sessions != 266:
        raise AssertionError(f"expected 266 score sessions, got {score_sessions}")

    candidates: dict[str, Any] = {}
    for slots in (1, 2):
        control = picks.loc[
            picks["recipe_id"].eq("R00_daily_raw_return_rank_control")
        ]
        for recipe_id, frame in picks.groupby("recipe_id", sort=True):
            candidates[f"{recipe_id}__top{slots}"] = summarize(
                frame, control, slots
            )
    args.picks_output.parent.mkdir(parents=True, exist_ok=True)
    picks.to_csv(args.picks_output, index=False)
    result = {
        "schema_version": 1,
        "protocol_id": "model_v10_market_residual_target_20260723",
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(Path(__file__)),
        "panel_sha256": sha256_file(args.panel),
        "picks_sha256": sha256_file(args.picks_output),
        "score_sessions": score_sessions,
        "folds": folds,
        "beta_audit": beta_audit,
        "candidates": candidates,
        "production_model_changed": False,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        candidate_id: metrics["mean_pct"]
        for candidate_id, metrics in candidates.items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
