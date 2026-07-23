#!/usr/bin/env python3
"""Run the preregistered v1.1 distributional/selective-trading screen.

The frozen panel is retrospective and already known to the wider project.
Consequently this runner can nominate, but can never promote, one exact
forward-shadow specification.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PROTOCOL_SHA256 = "9e7b98550f3066ee6b6eb8ec0ebdaf8120d211020b7c24be29618786e124e480"
PANEL_SHA256 = "6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb"
MANIFEST_SHA256 = "25e08c564ef6400b7a29168db9fd7e8220bd0e2386c7a3c7e71b05c2ff950b02"
SEED = 20260723
CONTROL = "C00_DAILY_RANK_RIDGE"
CANDIDATES = (
    "D01_CODE_EB_RESIDUAL",
    "D02_SURVIVAL_INTEGRAL",
    "D03_DOWNSIDE_FEASIBLE_MEAN",
    "D04_NORMALISED_CONFORMAL_LCB",
    "D05_FOREST_TREE_LCB",
    "D06_EVT_RESIDUAL_ES",
    "D07_REGIME_ABSTAIN",
    "D08_SLOTWISE_CASH_STOP",
)
BASE_MODELS = (CONTROL, *CANDIDATES)
CAPACITIES = (1, 2)
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
CONTEXT = ("prior_market_dispersion", "prior_market_tail_balance")
PERIODS = {
    "discovery": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-31")),
    "confirmation_a": (pd.Timestamp("2024-11-01"), pd.Timestamp("2025-03-31")),
    "confirmation_b": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
}


@dataclass(frozen=True)
class Prediction:
    """A score plus a point-in-time decision mask aligned to a scoring frame."""

    score: np.ndarray
    allowed: np.ndarray
    native_rank: np.ndarray | None = None


Scorer = Callable[[pd.DataFrame, np.ndarray], Prediction]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def exact_file(path: str | Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def policy_id(base_id: str, capacity: int) -> str:
    return f"{base_id}_K{capacity}"


def validate_input(
    panel: pd.DataFrame,
    manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    if protocol.get("protocol_id") != "model_v11_distributional_zero_base_20260723":
        raise ValueError("unexpected protocol id")
    if (
        protocol["authority"]["production_promotion_allowed_from_this_panel"]
        is not False
    ):
        raise ValueError("retrospective protocol cannot promote production")
    if int(protocol["candidate_family_size"]) != len(CANDIDATES) * len(CAPACITIES):
        raise ValueError("candidate family size differs from implementation")
    frozen = protocol["frozen_input"]
    if len(panel) != int(frozen["rows"]) or len(panel) != int(manifest["rows"]):
        raise ValueError("panel row count differs from protocol or manifest")
    if int(panel["code"].nunique()) != int(frozen["codes"]):
        raise ValueError("panel code count differs from protocol")
    required = {
        "date",
        "code",
        "name",
        "label",
        "oc_return_pct",
        "price_eligible",
        "price_training_eligible",
        "candidate_price_source_max_date",
        *FEATURES,
        *CONTEXT,
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel lacks required columns: {missing}")
    if panel.duplicated(["date", "code"]).any():
        raise ValueError("panel contains duplicate date/code rows")
    dates = pd.to_datetime(panel["date"], errors="coerce")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("panel dates are invalid")
    sessions = pd.DatetimeIndex(dates.drop_duplicates().sort_values())
    manifest_sessions = pd.DatetimeIndex(pd.to_datetime(manifest["sessions"]))
    if not sessions.equals(manifest_sessions):
        raise ValueError("panel and manifest session sets differ")
    returns = pd.to_numeric(panel["oc_return_pct"], errors="coerce")
    labels = pd.to_numeric(panel["label"], errors="coerce")
    if returns.isna().ne(labels.isna()).any():
        raise ValueError("return/label missingness differs")
    observed = returns.notna()
    if not np.array_equal(
        labels.loc[observed].to_numpy(float),
        returns.loc[observed].gt(0).to_numpy(float),
    ):
        raise ValueError("label is not exactly the return sign")
    source = pd.to_datetime(panel["candidate_price_source_max_date"], errors="coerce")
    if (source.notna() & source.ge(dates)).any():
        raise ValueError("candidate feature source is not strictly prior")
    score_start, score_end = map(pd.Timestamp, frozen["score_period"])
    scheduled = sessions[(sessions >= score_start) & (sessions <= score_end)]
    if len(scheduled) != int(frozen["score_sessions"]):
        raise ValueError("score-session count differs from protocol")
    return sessions, scheduled


def date_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("date", sort=False)["date"].transform("size").to_numpy(float)
    weights = 1.0 / counts
    totals = pd.Series(weights).groupby(
        frame["date"].reset_index(drop=True), sort=False
    ).sum()
    if not np.allclose(totals.to_numpy(), 1.0, atol=1e-12, rtol=0):
        raise AssertionError("date weights do not sum to one")
    return weights


def rank_target(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["oc_return_pct"]
        .groupby(frame["date"], sort=False)
        .rank(method="average", pct=True)
        .mul(2.0)
        .sub(1.0)
        .to_numpy(float)
    )


def fit_ridge(
    x: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(x, target, sample_weight=weights)
    return model


def fit_probability_ridge(
    x: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> Ridge:
    return fit_ridge(x, np.asarray(target, dtype=float), weights, alpha=10.0)


def predict_probability(model: Ridge, x: np.ndarray) -> np.ndarray:
    return np.clip(model.predict(x), 0.0, 1.0)


def chronological_mask(frame: pd.DataFrame, fraction: float) -> np.ndarray:
    dates = pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())
    cut = min(max(int(math.floor(len(dates) * fraction)), 1), len(dates) - 1)
    early_dates = set(dates[:cut])
    return frame["date"].isin(early_dates).to_numpy(bool)


def finite_sample_quantile(values: np.ndarray, coverage: float) -> float:
    clean = np.sort(np.asarray(values, dtype=float))
    if not len(clean) or not np.isfinite(clean).all():
        raise ValueError("invalid conformal calibration values")
    position = min(int(math.ceil((len(clean) + 1) * coverage)), len(clean)) - 1
    return float(clean[position])


def score_rank_metadata(
    frame: pd.DataFrame,
    score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = frame[["date", "code"]].copy()
    work["_position"] = np.arange(len(work))
    work["_score"] = np.asarray(score, dtype=float)
    work = work.sort_values(
        ["date", "_score", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    work["_rank"] = work.groupby("date", sort=False).cumcount() + 1
    work["_count"] = work.groupby("date", sort=False)["_score"].transform("size")
    work["_pct"] = 1.0 - (work["_rank"] - 1.0) / work["_count"].clip(lower=1.0)
    work["_std"] = work.groupby("date", sort=False)["_score"].transform("std")
    work["_next"] = work.groupby("date", sort=False)["_score"].shift(-1)
    work["_gap"] = (
        (work["_score"] - work["_next"].fillna(work["_score"]))
        / work["_std"].replace(0.0, np.nan)
    ).fillna(0.0)
    work = work.sort_values("_position", kind="stable")
    return (
        work["_rank"].to_numpy(int),
        work["_pct"].to_numpy(float),
        work["_gap"].to_numpy(float),
    )


def date_regime(
    frame: pd.DataFrame,
    dispersion_cut: float,
) -> np.ndarray:
    dispersion = pd.to_numeric(frame["prior_market_dispersion"], errors="coerce")
    tail = pd.to_numeric(frame["prior_market_tail_balance"], errors="coerce")
    dispersion = dispersion.fillna(dispersion_cut)
    tail = tail.fillna(0.0)
    return (
        dispersion.ge(dispersion_cut).astype(int).mul(2)
        + tail.ge(0.0).astype(int)
    ).to_numpy(int)


def fit_fold(
    training: pd.DataFrame,
    scoring: pd.DataFrame,
    x_train: np.ndarray,
    x_score: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, Prediction], dict[str, Scorer], dict[str, Any]]:
    clipped = np.clip(training["oc_return_pct"].to_numpy(float), -5.0, 5.0)
    rank_y = rank_target(training)
    scores: dict[str, Prediction] = {}
    scorers: dict[str, Scorer] = {}
    details: dict[str, Any] = {}

    control_model = fit_ridge(x_train, rank_y, weights, alpha=1.0)

    def control_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
        value = control_model.predict(x)
        return Prediction(value, np.ones(len(frame), dtype=bool))

    scorers[CONTROL] = control_scorer

    mean_model = fit_ridge(x_train, clipped, weights, alpha=10.0)
    mean_fitted = mean_model.predict(x_train)
    residual = clipped - mean_fitted
    residual_frame = pd.DataFrame(
        {
            "code": training["code"].to_numpy(),
            "residual": residual,
        }
    )
    code_stats = residual_frame.groupby("code", sort=False).agg(
        residual_sum=("residual", "sum"),
        observations=("residual", "size"),
    )
    code_effect = (
        code_stats["residual_sum"] / (code_stats["observations"] + 20.0)
    ).to_dict()

    def d01_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
        effect = frame["code"].map(code_effect).fillna(0.0).to_numpy(float)
        value = mean_model.predict(x) + effect
        return Prediction(value, np.ones(len(frame), dtype=bool))

    scorers["D01_CODE_EB_RESIDUAL"] = d01_scorer
    details["D01_CODE_EB_RESIDUAL"] = {
        "codes_with_posterior_effect": int(len(code_effect)),
        "prior_strength": 20.0,
    }

    up_040 = fit_probability_ridge(x_train, clipped > 0.40, weights)
    up_150 = fit_probability_ridge(x_train, clipped > 1.50, weights)
    down_040 = fit_probability_ridge(x_train, clipped < -0.40, weights)
    down_150 = fit_probability_ridge(x_train, clipped < -1.50, weights)

    def d02_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
        upside = (
            0.40 * predict_probability(up_040, x)
            + 1.10 * predict_probability(up_150, x)
        )
        downside = (
            0.40 * predict_probability(down_040, x)
            + 1.10 * predict_probability(down_150, x)
        )
        value = upside - 1.25 * downside
        return Prediction(value, np.ones(len(frame), dtype=bool))

    scorers["D02_SURVIVAL_INTEGRAL"] = d02_scorer
    details["D02_SURVIVAL_INTEGRAL"] = {
        "positive_thresholds_pct": [0.40, 1.50],
        "negative_thresholds_pct": [-0.40, -1.50],
        "downside_multiplier": 1.25,
    }

    severe_model = fit_probability_ridge(x_train, clipped < -1.0, weights)
    severe_base_rate = float(np.average(clipped < -1.0, weights=weights))

    def d03_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
        value = mean_model.predict(x)
        probability = predict_probability(severe_model, x)
        return Prediction(value, probability <= severe_base_rate)

    scorers["D03_DOWNSIDE_FEASIBLE_MEAN"] = d03_scorer
    details["D03_DOWNSIDE_FEASIBLE_MEAN"] = {
        "severe_loss_threshold_pct": -1.0,
        "training_base_rate": severe_base_rate,
    }

    early75 = chronological_mask(training, 0.75)
    late75 = ~early75
    mean75 = fit_ridge(
        x_train[early75], clipped[early75], weights[early75], alpha=10.0
    )
    early_residual = np.abs(clipped[early75] - mean75.predict(x_train[early75]))
    scale75 = fit_ridge(
        x_train[early75], early_residual, weights[early75], alpha=10.0
    )
    calibration_scale = np.maximum(scale75.predict(x_train[late75]), 0.05)
    calibration_error = (
        np.abs(clipped[late75] - mean75.predict(x_train[late75]))
        / calibration_scale
    )
    conformal_q = finite_sample_quantile(calibration_error, 0.80)

    def d04_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
        centre = mean75.predict(x)
        scale = np.maximum(scale75.predict(x), 0.05)
        lower = centre - conformal_q * scale
        return Prediction(lower, lower > 0.40)

    scorers["D04_NORMALISED_CONFORMAL_LCB"] = d04_scorer
    details["D04_NORMALISED_CONFORMAL_LCB"] = {
        "fit_rows": int(early75.sum()),
        "calibration_rows": int(late75.sum()),
        "normalised_error_q80": conformal_q,
    }

    # sklearn may interpret a fractional max_samples against the effective
    # sample-weight mass. Date-equal weights sum to the number of sessions, not
    # rows, so convert the preregistered 15% row fraction to an explicit count.
    forest_bootstrap_rows = max(1, int(math.ceil(len(x_train) * 0.15)))
    forest = RandomForestRegressor(
        n_estimators=24,
        max_depth=7,
        min_samples_leaf=500,
        max_features=0.75,
        max_samples=forest_bootstrap_rows,
        bootstrap=True,
        random_state=SEED,
        n_jobs=2,
    )
    forest.fit(x_train, clipped, sample_weight=weights)

    def d05_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
        tree_values = np.vstack([tree.predict(x) for tree in forest.estimators_])
        lower = np.quantile(tree_values, 0.20, axis=0)
        return Prediction(lower, lower > 0.40)

    scorers["D05_FOREST_TREE_LCB"] = d05_scorer
    details["D05_FOREST_TREE_LCB"] = {
        "trees": int(len(forest.estimators_)),
        "tree_prediction_quantile": 0.20,
        "bootstrap_row_fraction": 0.15,
        "bootstrap_rows": forest_bootstrap_rows,
    }

    early70 = chronological_mask(training, 0.70)
    late70 = ~early70
    mean70 = fit_ridge(
        x_train[early70], clipped[early70], weights[early70], alpha=10.0
    )
    downside_residual = mean70.predict(x_train[late70]) - clipped[late70]
    evt_threshold = float(np.quantile(downside_residual, 0.90))
    exceed = downside_residual > evt_threshold
    excess = downside_residual[exceed] - evt_threshold
    if len(excess) < 10:
        raise ValueError("too few EVT excess observations")
    excess_mean = float(np.mean(excess))
    excess_variance = float(np.var(excess, ddof=1))
    if excess_variance <= 0:
        evt_shape = -0.25
    else:
        evt_shape = 0.5 * (1.0 - excess_mean**2 / excess_variance)
    evt_shape = float(np.clip(evt_shape, -0.25, 0.45))
    evt_scale = float(max(excess_mean * (1.0 - evt_shape), 1e-6))
    evt_mean_excess = float(evt_scale / (1.0 - evt_shape))
    exceedance_model = fit_probability_ridge(
        x_train[late70], exceed, weights[late70]
    )

    def d06_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
        centre = mean_model.predict(x)
        exceedance_probability = predict_probability(exceedance_model, x)
        expected_shortfall_penalty = exceedance_probability * (
            evt_threshold + evt_mean_excess
        )
        value = centre - expected_shortfall_penalty
        return Prediction(value, value > 0.40)

    scorers["D06_EVT_RESIDUAL_ES"] = d06_scorer
    details["D06_EVT_RESIDUAL_ES"] = {
        "fit_rows": int(early70.sum()),
        "calibration_rows": int(late70.sum()),
        "threshold_pct": evt_threshold,
        "exceedances": int(exceed.sum()),
        "gpd_shape": evt_shape,
        "gpd_scale": evt_scale,
        "gpd_mean_excess": evt_mean_excess,
    }

    early_control75 = fit_ridge(
        x_train[early75], rank_y[early75], weights[early75], alpha=1.0
    )
    calibration = training.loc[late75].copy().reset_index(drop=True)
    calibration_score = early_control75.predict(x_train[late75])
    calibration["_score"] = calibration_score
    calibration = calibration.sort_values(
        ["date", "_score", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    calibration["_rank"] = calibration.groupby("date", sort=False).cumcount() + 1
    calibration_top2 = calibration.loc[calibration["_rank"] <= 2].copy()
    early_context = training.loc[early75, ["date", *CONTEXT]].drop_duplicates("date")
    dispersion_cut = float(
        pd.to_numeric(
            early_context["prior_market_dispersion"], errors="coerce"
        ).median()
    )
    calibration_day = (
        calibration_top2.groupby("date", sort=True)
        .agg(
            gross=("oc_return_pct", "mean"),
            prior_market_dispersion=("prior_market_dispersion", "first"),
            prior_market_tail_balance=("prior_market_tail_balance", "first"),
        )
        .reset_index()
    )
    calibration_day["_regime"] = date_regime(calibration_day, dispersion_cut)
    calibration_day["_win"] = calibration_day["gross"].sub(0.40).gt(0.0)
    regime_posterior: dict[int, dict[str, float | int | bool]] = {}
    for regime in range(4):
        subset = calibration_day.loc[calibration_day["_regime"] == regime]
        count = int(len(subset))
        wins = int(subset["_win"].sum())
        posterior = float((wins + 4.0) / (count + 8.0))
        regime_posterior[regime] = {
            "calibration_days": count,
            "profitable_days": wins,
            "posterior_probability": posterior,
            "trade": bool(count >= 20 and posterior > 0.55),
        }

    def d07_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
        value = control_model.predict(x)
        regime = date_regime(frame, dispersion_cut)
        allowed = np.array(
            [bool(regime_posterior[int(item)]["trade"]) for item in regime],
            dtype=bool,
        )
        return Prediction(value, allowed)

    scorers["D07_REGIME_ABSTAIN"] = d07_scorer
    details["D07_REGIME_ABSTAIN"] = {
        "dispersion_cut": dispersion_cut,
        "calibration_days": int(len(calibration_day)),
        "regime_posterior": {
            str(key): value for key, value in regime_posterior.items()
        },
    }

    cal_rank, cal_pct, cal_gap = score_rank_metadata(
        training.loc[late75].reset_index(drop=True), calibration_score
    )
    cal_slot_mask = cal_rank <= 2
    cal_meta_x = np.column_stack(
        [
            cal_pct[cal_slot_mask],
            cal_gap[cal_slot_mask],
            (cal_rank[cal_slot_mask] == 2).astype(float),
        ]
    )
    cal_meta_y = clipped[late75][cal_slot_mask]
    cal_meta_dates = training.loc[late75, "date"].reset_index(drop=True).loc[
        cal_slot_mask
    ]
    cal_meta_counts = cal_meta_dates.groupby(cal_meta_dates, sort=False).transform(
        "size"
    )
    cal_meta_weights = 1.0 / cal_meta_counts.to_numpy(float)
    slot_model = fit_ridge(
        cal_meta_x, cal_meta_y, cal_meta_weights, alpha=1.0
    )

    def d08_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
        value = control_model.predict(x)
        native_rank, percentile, gap = score_rank_metadata(frame, value)
        meta_x = np.column_stack(
            [
                percentile,
                gap,
                (native_rank == 2).astype(float),
            ]
        )
        expected_gross = slot_model.predict(meta_x)
        allowed = (native_rank <= 2) & (expected_gross > 0.40)
        return Prediction(value, allowed, native_rank)

    scorers["D08_SLOTWISE_CASH_STOP"] = d08_scorer
    details["D08_SLOTWISE_CASH_STOP"] = {
        "calibration_slot_rows": int(cal_slot_mask.sum()),
        "meta_coefficients": [float(value) for value in slot_model.coef_],
        "meta_intercept": float(slot_model.intercept_),
    }

    for candidate_id, scorer in scorers.items():
        prediction = scorer(scoring, x_score)
        if prediction.score.shape != (len(scoring),):
            raise ValueError(f"{candidate_id} produced a score with invalid shape")
        if prediction.allowed.shape != (len(scoring),):
            raise ValueError(f"{candidate_id} produced a mask with invalid shape")
        if not np.isfinite(prediction.score).all():
            raise ValueError(f"{candidate_id} produced a non-finite score")
        scores[candidate_id] = prediction
    if set(scores) != set(BASE_MODELS):
        raise AssertionError("candidate set differs from protocol")
    return scores, scorers, details


def select_policy_rows(
    scoring: pd.DataFrame,
    prediction: Prediction,
    base_id: str,
    capacity: int,
) -> pd.DataFrame:
    ranked = scoring.assign(
        model_score=np.asarray(prediction.score, dtype=float),
        trade_allowed=np.asarray(prediction.allowed, dtype=bool),
    )
    if prediction.native_rank is not None:
        ranked["native_rank"] = np.asarray(prediction.native_rank, dtype=int)
    ranked = ranked.loc[ranked["trade_allowed"]].copy()
    if base_id == "D08_SLOTWISE_CASH_STOP":
        ranked = ranked.loc[ranked["native_rank"] <= capacity].copy()
        ranked["model_rank"] = ranked["native_rank"].astype(int)
        ranked = ranked.sort_values(["date", "model_rank"], kind="stable")
    else:
        ranked = ranked.sort_values(
            ["date", "model_score", "code"],
            ascending=[True, False, True],
            kind="stable",
        )
        ranked = ranked.groupby("date", sort=True, as_index=False).head(capacity)
        ranked["model_rank"] = ranked.groupby("date", sort=False).cumcount() + 1
    ranked["hypothesis_id"] = base_id
    ranked["policy_id"] = policy_id(base_id, capacity)
    ranked["capacity"] = capacity
    return ranked


def desired_slots(sessions: pd.DatetimeIndex, capacity: int) -> pd.DataFrame:
    return pd.MultiIndex.from_product(
        [sessions, range(1, capacity + 1)], names=["date", "model_rank"]
    ).to_frame(index=False)


def daily_returns(
    picks: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    cost_bps: float,
) -> pd.DataFrame:
    capacity = int(picks["capacity"].iloc[0])
    weight = 1.0 / capacity
    executed = picks["oc_return_pct"].notna() & picks["code"].notna()
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(picks["date"]),
            "gross": weight * picks["oc_return_pct"].fillna(0.0),
            "cost": weight * executed.astype(float) * cost_bps / 100.0,
            "exposure": weight * executed.astype(float),
        }
    )
    grouped = frame.groupby("date", sort=True).sum()
    grouped["net"] = grouped["gross"] - grouped["cost"]
    return grouped.reindex(sessions, fill_value=0.0)[
        ["gross", "net", "exposure"]
    ]


def profit_factor(values: pd.Series) -> float | None:
    gains = float(values.clip(lower=0.0).sum())
    losses = float(-values.clip(upper=0.0).sum())
    return float(gains / losses) if losses > 0 else None


def policy_metrics(
    picks: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> tuple[dict[str, Any], pd.DataFrame]:
    capacity = int(picks["capacity"].iloc[0])
    weight = 1.0 / capacity
    daily = {
        cost: daily_returns(picks, sessions, float(cost))
        for cost in (20, 40, 60)
    }
    executed = picks["oc_return_pct"].notna() & picks["code"].notna()
    net20 = daily[20]["net"]
    net40 = daily[40]["net"]
    monthly40 = net40.groupby(net40.index.to_period("M")).mean()
    q05 = float(net40.quantile(0.05))
    es05 = float(net40.loc[net40.le(q05)].mean())
    slots = picks.loc[executed, ["date", "code", "oc_return_pct"]].copy()
    slots["weight"] = weight
    slots["net20_contribution"] = weight * (slots["oc_return_pct"] - 0.20)
    if slots.empty:
        code_concentration = {
            "unique_codes": 0,
            "largest_weight_share": None,
            "top10_weight_share": None,
            "largest_positive_net20_pnl_share": None,
            "top10_positive_pnl_codes_to_cash_net20_pct": 0.0,
        }
    else:
        by_weight = slots.groupby("code", sort=False)["weight"].sum().sort_values(
            ascending=False
        )
        by_pnl = slots.groupby("code", sort=False)["net20_contribution"].sum()
        positive = by_pnl.clip(lower=0.0)
        top_positive_codes = set(positive.nlargest(10).index)
        cash_mask = picks["code"].isin(top_positive_codes)
        cash_version = picks.copy()
        cash_version.loc[cash_mask, ["code", "name", "label", "oc_return_pct"]] = np.nan
        code_concentration = {
            "unique_codes": int(slots["code"].nunique()),
            "largest_weight_share": float(by_weight.iloc[0] / by_weight.sum()),
            "top10_weight_share": float(by_weight.head(10).sum() / by_weight.sum()),
            "largest_positive_net20_pnl_share": (
                float(positive.max() / positive.sum())
                if positive.sum() > 0
                else None
            ),
            "top10_positive_pnl_codes_to_cash_net20_pct": float(
                daily_returns(cash_version, sessions, 20.0)["net"].mean()
            ),
        }
    metrics: dict[str, Any] = {
        "capacity": capacity,
        "scheduled_days": int(len(sessions)),
        "scheduled_slots": int(len(sessions) * capacity),
        "executed_slots": int(executed.sum()),
        "executed_slot_fraction": float(executed.sum() / (len(sessions) * capacity)),
        "traded_days": int(
            picks.assign(_executed=executed)
            .groupby("date", sort=True)["_executed"]
            .any()
            .reindex(sessions, fill_value=False)
            .sum()
        ),
        "cash_days": int(
            len(sessions)
            - picks.assign(_executed=executed)
            .groupby("date", sort=True)["_executed"]
            .any()
            .reindex(sessions, fill_value=False)
            .sum()
        ),
        "hit_rate_executed": (
            float(picks.loc[executed, "label"].mean()) if executed.any() else None
        ),
        "gross_mean_pct": float(daily[20]["gross"].mean()),
        "net_mean_pct": {
            str(cost): float(frame["net"].mean())
            for cost, frame in daily.items()
        },
        "net40_profit_factor": profit_factor(net40),
        "monthly_net40_pct": {
            str(period): float(value) for period, value in monthly40.items()
        },
        "positive_months_net40": int(monthly40.gt(0.0).sum()),
        "months": int(len(monthly40)),
        "temporal_slices_net40_pct": {
            name: float(net40.loc[start:end].mean())
            for name, (start, end) in PERIODS.items()
        },
        "winning_days_removed_net20_pct": {
            str(count): float(net20.drop(net20.nlargest(count).index).mean())
            for count in (5, 10, 20)
        },
        "losing_days_removed_net20_pct": {
            str(count): float(net20.drop(net20.nsmallest(count).index).mean())
            for count in (5, 10, 20)
        },
        "expected_shortfall05_net40_pct": es05,
        "p05_net40_pct": q05,
        "worst_day_net40_pct": float(net40.min()),
        "code_concentration": code_concentration,
    }
    daily_export = pd.DataFrame(
        {
            "date": sessions,
            "net20": daily[20]["net"].to_numpy(float),
            "net40": net40.to_numpy(float),
            "net60": daily[60]["net"].to_numpy(float),
        }
    )
    return metrics, daily_export


def hac_mean_se(values: np.ndarray, lag: int = 5) -> float:
    x = np.asarray(values, dtype=float)
    n = len(x)
    centred = x - x.mean()
    long_run = float(np.dot(centred, centred) / n)
    for offset in range(1, min(lag, n - 1) + 1):
        covariance = float(np.dot(centred[offset:], centred[:-offset]) / n)
        long_run += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    return float(math.sqrt(max(long_run, 1e-18) / n))


def familywise_multiplicity(
    daily: dict[str, pd.DataFrame],
    *,
    repetitions: int = 5000,
    block_length: int = 10,
) -> dict[str, Any]:
    candidate_policies = [
        policy_id(candidate, capacity)
        for candidate in CANDIDATES
        for capacity in CAPACITIES
    ]
    deltas = np.vstack(
        [
            daily[key]["net40"].to_numpy(float)
            - daily[policy_id(CONTROL, int(key.rsplit("_K", 1)[1]))][
                "net40"
            ].to_numpy(float)
            for key in candidate_policies
        ]
    )
    if not np.isfinite(deltas).all():
        raise ValueError("daily uplift matrix is incomplete")
    point = deltas.mean(axis=1)
    standard_error = np.array([hac_mean_se(row) for row in deltas])
    n = deltas.shape[1]
    blocks_needed = int(math.ceil(n / block_length))
    offsets = np.arange(block_length)
    rng = np.random.default_rng(SEED)
    boot_means = np.empty((repetitions, len(candidate_policies)), dtype=float)
    for repetition in range(repetitions):
        starts = rng.integers(0, n, size=blocks_needed)
        positions = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        boot_means[repetition] = deltas[:, positions].mean(axis=1)
    centred_t = (boot_means - point) / standard_error
    max_t = centred_t.max(axis=1)
    critical = float(np.quantile(max_t, 0.95))
    hypotheses: dict[str, Any] = {}
    for index, key in enumerate(candidate_policies):
        observed_t = point[index] / standard_error[index]
        hypotheses[key] = {
            "net40_uplift_vs_matched_control_pct": float(point[index]),
            "hac5_standard_error_pct": float(standard_error[index]),
            "ordinary_one_sided95_lower_pct": float(
                np.quantile(boot_means[:, index], 0.05)
            ),
            "familywise_max_t_one_sided95_lower_pct": float(
                point[index] - critical * standard_error[index]
            ),
            "familywise_adjusted_p_approx": float(np.mean(max_t >= observed_t)),
        }
    return {
        "method": "studentised circular moving-block familywise max-t",
        "seed": SEED,
        "repetitions": repetitions,
        "block_length_sessions": block_length,
        "family_size": len(candidate_policies),
        "familywise_one_sided_level": 0.95,
        "critical_max_t": critical,
        "hypotheses": hypotheses,
    }


def retrospective_checks(
    metrics: dict[str, dict[str, Any]],
    multiplicity: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for candidate in CANDIDATES:
        for capacity in CAPACITIES:
            key = policy_id(candidate, capacity)
            control_key = policy_id(CONTROL, capacity)
            value = metrics[key]
            control = metrics[control_key]
            concentration = value["code_concentration"]
            family = multiplicity["hypotheses"][key]
            largest_weight = concentration["largest_weight_share"]
            top10_weight = concentration["top10_weight_share"]
            largest_pnl = concentration["largest_positive_net20_pnl_share"]
            checks = {
                "net40_mean_positive": value["net_mean_pct"]["40"] > 0.0,
                "net60_mean_positive": value["net_mean_pct"]["60"] > 0.0,
                "net40_greater_than_capacity_matched_control": (
                    value["net_mean_pct"]["40"] > control["net_mean_pct"]["40"]
                ),
                "familywise_95_lower_net40_uplift_positive": (
                    family["familywise_max_t_one_sided95_lower_pct"] > 0.0
                ),
                "all_three_temporal_slices_net40_positive": all(
                    item > 0.0
                    for item in value["temporal_slices_net40_pct"].values()
                ),
                "positive_months_net40_at_least_10": (
                    value["positive_months_net40"] >= 10
                ),
                "best_20_days_removed_net20_positive": (
                    value["winning_days_removed_net20_pct"]["20"] > 0.0
                ),
                "top_10_positive_pnl_codes_to_cash_net20_positive": (
                    concentration[
                        "top10_positive_pnl_codes_to_cash_net20_pct"
                    ]
                    > 0.0
                ),
                "expected_shortfall05_net40_not_worse_than_control": (
                    value["expected_shortfall05_net40_pct"]
                    >= control["expected_shortfall05_net40_pct"]
                ),
                "executed_slot_fraction_at_least_0_3": (
                    value["executed_slot_fraction"] >= 0.30
                ),
                "traded_days_at_least_150": value["traded_days"] >= 150,
                "unique_codes_at_least_100": concentration["unique_codes"] >= 100,
                "largest_code_weight_share_at_most_0_05": (
                    largest_weight is not None and largest_weight <= 0.05
                ),
                "top10_code_weight_share_at_most_0_25": (
                    top10_weight is not None and top10_weight <= 0.25
                ),
                "largest_positive_code_pnl_share_at_most_0_25": (
                    largest_pnl is not None and largest_pnl <= 0.25
                ),
                "target_mutation_exact": True,
            }
            output[key] = {
                "checks_before_independent_pnl_audit": checks,
                "passes_before_independent_pnl_audit": all(checks.values()),
                "independent_pnl_audit_exact": None,
                "passes_all_retrospective_gates": None,
            }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="/tmp/model_v07_corrected_panel.pkl")
    parser.add_argument(
        "--protocol", default="research/model_v11_distributional_protocol.json"
    )
    parser.add_argument(
        "--output", default="research/model_v11_distributional_result.json"
    )
    parser.add_argument(
        "--picks-output", default="research/model_v11_distributional_picks.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    panel_path = Path(args.panel).resolve()
    manifest_path = Path(str(panel_path) + ".manifest.json")
    protocol_path = Path(args.protocol).resolve()
    output_path = Path(args.output).resolve()
    picks_path = Path(args.picks_output).resolve()
    exact_file(protocol_path, PROTOCOL_SHA256, "protocol")
    exact_file(panel_path, PANEL_SHA256, "panel")
    exact_file(manifest_path, MANIFEST_SHA256, "panel manifest")
    protocol = read_json(protocol_path)
    manifest = read_json(manifest_path)
    panel = joblib.load(panel_path, mmap_mode="r")
    all_sessions, scheduled = validate_input(panel, manifest, protocol)
    projection = [
        "date",
        "code",
        "name",
        "label",
        "oc_return_pct",
        "price_eligible",
        "price_training_eligible",
        *CONTEXT,
        *FEATURES,
    ]
    score_months = pd.period_range("2024-07", "2025-07", freq="M")
    parts: dict[str, list[pd.DataFrame]] = {
        policy_id(base, capacity): []
        for base in BASE_MODELS
        for capacity in CAPACITIES
    }
    fold_audit: list[dict[str, Any]] = []
    mutation_audit: dict[str, Any] = {}
    for fold_number, month in enumerate(score_months, start=1):
        fold_started = time.time()
        score_start = month.start_time.normalize()
        score_end = month.end_time.normalize()
        training = panel.loc[
            panel["date"].between(
                pd.Timestamp("2024-01-04"), score_start - pd.Timedelta(days=1)
            )
            & panel["price_training_eligible"].eq(True)
            & panel["oc_return_pct"].notna(),
            projection,
        ].copy().reset_index(drop=True)
        scoring = panel.loc[
            panel["date"].between(score_start, score_end)
            & panel["price_eligible"].eq(True),
            projection,
        ].copy().reset_index(drop=True)
        if training.empty or training["date"].max() >= score_start:
            raise ValueError(f"{month} training is not strictly prior")
        expected_dates = scheduled[scheduled.to_period("M") == month]
        score_dates = pd.DatetimeIndex(scoring["date"].drop_duplicates().sort_values())
        if not score_dates.equals(expected_dates):
            raise ValueError(f"{month} score-date set differs from schedule")
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        scaler = StandardScaler()
        x_train = imputer.fit_transform(training.loc[:, FEATURES])
        x_train = scaler.fit_transform(x_train).astype("float32", copy=False)
        x_score = imputer.transform(scoring.loc[:, FEATURES])
        x_score = scaler.transform(x_score).astype("float32", copy=False)
        weights = date_equal_weights(training)
        predictions, scorers, details = fit_fold(
            training, scoring, x_train, x_score, weights
        )
        for base_id, prediction in predictions.items():
            for capacity in CAPACITIES:
                key = policy_id(base_id, capacity)
                parts[key].append(
                    select_policy_rows(
                        scoring, prediction, base_id, capacity
                    )
                )
        if str(month) == "2025-07":
            changed = scoring.copy()
            changed["oc_return_pct"] = np.where(
                np.arange(len(changed)) % 2 == 0, 99.0, -99.0
            )
            changed["label"] = changed["oc_return_pct"].gt(0.0).astype(float)
            changed_x = scaler.transform(
                imputer.transform(changed.loc[:, FEATURES])
            ).astype("float32", copy=False)
            if not np.array_equal(x_score, changed_x):
                raise ValueError("target mutation changed the feature matrix")
            for base_id, scorer in scorers.items():
                original = predictions[base_id]
                mutated = scorer(changed, changed_x)
                score_difference = float(
                    np.max(np.abs(original.score - mutated.score))
                )
                allowed_exact = bool(np.array_equal(original.allowed, mutated.allowed))
                native_exact = bool(
                    (original.native_rank is None and mutated.native_rank is None)
                    or (
                        original.native_rank is not None
                        and mutated.native_rank is not None
                        and np.array_equal(
                            original.native_rank, mutated.native_rank
                        )
                    )
                )
                capacity_results: dict[str, bool] = {}
                for capacity in CAPACITIES:
                    original_keys = select_policy_rows(
                        scoring, original, base_id, capacity
                    )[["date", "model_rank", "code"]].reset_index(drop=True)
                    mutated_keys = select_policy_rows(
                        changed, mutated, base_id, capacity
                    )[["date", "model_rank", "code"]].reset_index(drop=True)
                    capacity_results[str(capacity)] = bool(
                        original_keys.equals(mutated_keys)
                    )
                mutation_audit[base_id] = {
                    "max_abs_score_difference": score_difference,
                    "trade_allowed_exact": allowed_exact,
                    "native_rank_exact": native_exact,
                    "selected_keys_exact_by_capacity": capacity_results,
                }
        fold_audit.append(
            {
                "month": str(month),
                "fold_number": fold_number,
                "train_start": str(training["date"].min().date()),
                "train_end": str(training["date"].max().date()),
                "training_rows": int(len(training)),
                "training_sessions": int(training["date"].nunique()),
                "score_rows": int(len(scoring)),
                "score_sessions": int(scoring["date"].nunique()),
                "strictly_prior_training": True,
                "model_details": details,
                "runtime_seconds": float(time.time() - fold_started),
            }
        )
        print(
            f"completed {month}: train={len(training):,} "
            f"score={len(scoring):,} seconds={time.time() - fold_started:.1f}",
            flush=True,
        )
        del (
            training,
            scoring,
            x_train,
            x_score,
            weights,
            predictions,
            scorers,
            imputer,
            scaler,
        )
        gc.collect()
    mutation_ok = (
        set(mutation_audit) == set(BASE_MODELS)
        and all(
            item["max_abs_score_difference"] == 0.0
            and item["trade_allowed_exact"]
            and item["native_rank_exact"]
            and all(item["selected_keys_exact_by_capacity"].values())
            for item in mutation_audit.values()
        )
    )
    if not mutation_ok:
        raise ValueError("target-day outcome mutation audit failed")

    picks_by_policy: dict[str, pd.DataFrame] = {}
    all_picks: list[pd.DataFrame] = []
    keep = [
        "date",
        "model_rank",
        "code",
        "name",
        "model_score",
        "trade_allowed",
        "label",
        "oc_return_pct",
    ]
    for base_id in BASE_MODELS:
        for capacity in CAPACITIES:
            key = policy_id(base_id, capacity)
            actual = pd.concat(parts[key], ignore_index=True)
            desired = desired_slots(scheduled, capacity)
            picks = desired.merge(
                actual[keep],
                on=["date", "model_rank"],
                how="left",
                validate="one_to_one",
                sort=True,
            )
            picks["hypothesis_id"] = base_id
            picks["policy_id"] = key
            picks["capacity"] = capacity
            picks["executed"] = (
                picks["code"].notna() & picks["oc_return_pct"].notna()
            )
            if len(picks) != len(scheduled) * capacity:
                raise ValueError(f"{key} scheduled slot count is wrong")
            picks_by_policy[key] = picks
            all_picks.append(picks)
    pick_frame = pd.concat(all_picks, ignore_index=True)
    pick_frame.to_csv(picks_path, index=False)

    metrics: dict[str, dict[str, Any]] = {}
    daily: dict[str, pd.DataFrame] = {}
    for key, picks in picks_by_policy.items():
        metrics[key], daily[key] = policy_metrics(picks, scheduled)
    multiplicity = familywise_multiplicity(daily)
    gates = retrospective_checks(metrics, multiplicity)
    preaudit_passers = [
        key
        for key, value in gates.items()
        if value["passes_before_independent_pnl_audit"]
    ]
    point_winner = max(
        (
            policy_id(candidate, capacity)
            for candidate in CANDIDATES
            for capacity in CAPACITIES
        ),
        key=lambda key: metrics[key]["net_mean_pct"]["40"],
    )
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "authority": protocol["authority"],
        "production_model_changed": False,
        "input": {
            "panel_path": str(panel_path),
            "panel_sha256": PANEL_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "protocol_path": str(protocol_path),
            "protocol_sha256": PROTOCOL_SHA256,
            "runner_sha256": sha256_file(Path(__file__)),
            "rows": int(len(panel)),
            "codes": int(panel["code"].nunique()),
            "sessions": int(len(all_sessions)),
            "score_sessions": int(len(scheduled)),
        },
        "integrity": {
            "monthly_expanding_folds": int(len(fold_audit)),
            "all_training_strictly_prior": all(
                value["strictly_prior_training"] for value in fold_audit
            ),
            "candidate_feature_source_violations": 0,
            "target_day_outcome_mutation_exact": mutation_ok,
            "target_day_outcome_mutation": mutation_audit,
            "cash_slots_not_renormalised": True,
            "candidate_policy_count": len(CANDIDATES) * len(CAPACITIES),
            "control_policy_count": len(CAPACITIES),
        },
        "decision_before_independent_audit": {
            "highest_unadjusted_net40_policy": point_winner,
            "highest_unadjusted_net40_pct": metrics[point_winner]["net_mean_pct"][
                "40"
            ],
            "preaudit_gate_passers": preaudit_passers,
            "forward_shadow_finalist": None,
            "production_action": "none",
            "status": "independent_pnl_audit_pending",
        },
        "metrics": metrics,
        "multiplicity": multiplicity,
        "retrospective_gates": gates,
        "folds": fold_audit,
        "artifacts": {
            "picks_path": str(picks_path),
            "picks_sha256": sha256_file(picks_path),
        },
        "runtime_seconds": float(time.time() - started),
    }
    write_json(output_path, result)
    print(
        f"done: point_winner={point_winner} "
        f"preaudit_passers={preaudit_passers} "
        f"seconds={time.time() - started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
