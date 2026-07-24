#!/usr/bin/env python3
"""Run the preregistered v1.0 zero-base model-architecture comparison.

This file is intentionally isolated from the repository.  It consumes the
content-addressed frozen panel, never changes the production artifact, and
emits only retrospective research artifacts under /tmp.
"""

from __future__ import annotations

import argparse
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


PROTOCOL_SHA256 = "bb5224836397070a1945d0226caf1af3f9f2057b9da1265a97a5b1421c82aa7a"
PANEL_SHA256 = "6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb"
MANIFEST_SHA256 = "25e08c564ef6400b7a29168db9fd7e8220bd0e2386c7a3c7e71b05c2ff950b02"
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
CONTROL_ID = "C00_daily_rank_ridge"
CANDIDATE_IDS = (
    "M01_listwise_exponential_gain_ridge",
    "M02_listwise_extreme_gain_ridge",
    "M03_bounded_cost_utility_ridge",
    "M04_dual_tail_probability",
    "M05_two_stage_cost_expected_return",
    "M06_magnitude_weighted_pairwise",
    "M07_extreme_pairwise",
    "M08_lower_expectile_35",
    "M09_temporal_median_three_experts",
    "M10_dispersion_mixture_rank",
    "M11_lower_confidence_residual_penalty",
    "M12_quantile35_hgb",
    "M13_dual_quantile_hgb",
)
ALL_IDS = (CONTROL_ID, *CANDIDATE_IDS)
PERIODS = {
    "discovery": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-31")),
    "confirmation_a": (pd.Timestamp("2024-11-01"), pd.Timestamp("2025-03-31")),
    "confirmation_b": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
}
ScoreFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def exact_file(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def validate_input(
    panel: pd.DataFrame, manifest: dict[str, Any], protocol: dict[str, Any]
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    if protocol.get("protocol_id") != "v10_zero_base_model_architectures_20260723":
        raise ValueError("unexpected protocol id")
    if protocol.get("authority", {}).get("production_promotion_allowed") is not False:
        raise ValueError("retrospective protocol cannot promote production")
    frozen = protocol["frozen_input"]
    if len(panel) != int(frozen["rows"]) or len(panel) != int(manifest["rows"]):
        raise ValueError("panel row count differs from protocol or manifest")
    if int(panel["code"].nunique()) != int(frozen["codes"]):
        raise ValueError("panel code count changed")
    required = {
        "date",
        "code",
        "name",
        "label",
        "oc_return_pct",
        "price_eligible",
        "price_training_eligible",
        "candidate_price_source_max_date",
        "prior_market_dispersion",
        *FEATURES,
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
        labels.loc[observed].to_numpy(dtype=float),
        returns.loc[observed].gt(0).to_numpy(dtype=float),
    ):
        raise ValueError("label is not exactly the sign of open-close return")
    source = pd.to_datetime(panel["candidate_price_source_max_date"], errors="coerce")
    if (source.notna() & source.ge(dates)).any():
        raise ValueError("candidate price feature is not strictly prior")
    score_start, score_end = map(pd.Timestamp, frozen["score_period"])
    scheduled = sessions[(sessions >= score_start) & (sessions <= score_end)]
    if len(scheduled) != int(frozen["score_sessions"]):
        raise ValueError("score-session count changed")
    return sessions, scheduled


def date_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("date", sort=False)["date"].transform("size").to_numpy(float)
    result = 1.0 / counts
    totals = pd.Series(result).groupby(frame["date"].reset_index(drop=True)).sum()
    if not np.allclose(totals.to_numpy(), 1.0, atol=1e-12, rtol=0):
        raise AssertionError("date weights do not sum to one")
    return result


def renormalise_date_weights(
    frame: pd.DataFrame, base: np.ndarray, modifier: np.ndarray
) -> np.ndarray:
    raw = np.asarray(base, float) * np.asarray(modifier, float)
    if (raw <= 0).any() or not np.isfinite(raw).all():
        raise ValueError("invalid iterative weights")
    dates = frame["date"].reset_index(drop=True)
    totals = pd.Series(raw).groupby(dates, sort=False).transform("sum").to_numpy()
    return raw / totals


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
    x: np.ndarray, target: np.ndarray, weights: np.ndarray, alpha: float = 1.0
) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(x, target, sample_weight=weights)
    return model


def fit_logit(
    x: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    fit_intercept: bool = True,
) -> LogisticRegression:
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("logistic target lacks both classes")
    model = LogisticRegression(
        C=0.08,
        solver="lbfgs",
        max_iter=500,
        random_state=31,
        fit_intercept=fit_intercept,
    )
    model.fit(x, target, sample_weight=weights)
    return model


def build_pairwise(
    x: np.ndarray,
    returns: np.ndarray,
    frame: pd.DataFrame,
    *,
    extreme_only: bool,
    seed: int,
    pairs_per_date: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    differences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for positions in frame.groupby("date", sort=True).indices.values():
        positions = np.asarray(positions, dtype=int)
        realised = returns[positions]
        if extreme_only:
            order = np.argsort(realised, kind="stable")
            count = max(1, int(math.ceil(len(order) * 0.10)))
            low_pool = positions[order[:count]]
            high_pool = positions[order[-count:]]
            high = rng.choice(high_pool, pairs_per_date, replace=True)
            low = rng.choice(low_pool, pairs_per_date, replace=True)
        else:
            left = rng.choice(positions, pairs_per_date, replace=True)
            right = rng.choice(positions, pairs_per_date, replace=True)
            same = left == right
            while same.any():
                right[same] = rng.choice(positions, int(same.sum()), replace=True)
                same = left == right
            unequal = returns[left] != returns[right]
            left, right = left[unequal], right[unequal]
            if not len(left):
                continue
            high = np.where(returns[left] > returns[right], left, right)
            low = np.where(returns[left] > returns[right], right, left)
        keep = returns[high] > returns[low]
        high, low = high[keep], low[keep]
        if not len(high):
            continue
        positive = x[high] - x[low]
        magnitude = np.minimum(np.abs(returns[high] - returns[low]), 3.0)
        if extreme_only:
            magnitude = np.ones_like(magnitude)
        magnitude = magnitude / magnitude.sum()
        differences.extend([positive, -positive])
        targets.extend(
            [np.ones(len(positive), dtype=int), np.zeros(len(positive), dtype=int)]
        )
        weights.extend([0.5 * magnitude, 0.5 * magnitude])
    if not differences:
        raise ValueError("no pairwise comparisons were constructed")
    return np.vstack(differences), np.concatenate(targets), np.concatenate(weights)


def hgb_quantile(quantile: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        learning_rate=0.05,
        max_iter=60,
        max_leaf_nodes=15,
        min_samples_leaf=500,
        l2_regularization=2.0,
        max_bins=63,
        early_stopping=False,
        random_state=31,
    )


def fit_candidates(
    training: pd.DataFrame,
    scoring: pd.DataFrame,
    x_train: np.ndarray,
    x_score: np.ndarray,
    *,
    fold_number: int,
) -> tuple[dict[str, np.ndarray], dict[str, ScoreFunction], dict[str, Any]]:
    weights = date_equal_weights(training)
    returns = training["oc_return_pct"].to_numpy(float)
    clipped = np.clip(returns, -3.0, 3.0)
    percentile = (
        training["oc_return_pct"]
        .groupby(training["date"], sort=False)
        .rank(method="average", pct=True)
        .to_numpy(float)
    )
    target_rank = 2.0 * percentile - 1.0
    context_train = training["prior_market_dispersion"].to_numpy(float)
    context_score = scoring["prior_market_dispersion"].to_numpy(float)
    if not np.isfinite(context_train).all() or not np.isfinite(context_score).all():
        raise ValueError("dispersion routing context must be complete and finite")
    scores: dict[str, np.ndarray] = {}
    functions: dict[str, ScoreFunction] = {}
    details: dict[str, Any] = {}

    control = fit_ridge(x_train, target_rank, weights)
    functions[CONTROL_ID] = lambda x, c, m=control: m.predict(x)

    target = np.exp(3.0 * (percentile - 1.0))
    m01 = fit_ridge(x_train, target, weights)
    functions[CANDIDATE_IDS[0]] = lambda x, c, m=m01: m.predict(x)

    centred = 2.0 * percentile - 1.0
    target = np.sign(centred) * np.abs(centred) ** 3
    m02 = fit_ridge(x_train, target, weights)
    functions[CANDIDATE_IDS[1]] = lambda x, c, m=m02: m.predict(x)

    target = np.tanh((returns - 0.20) / 1.00)
    m03 = fit_ridge(x_train, target, weights)
    functions[CANDIDATE_IDS[2]] = lambda x, c, m=m03: m.predict(x)

    upside = fit_logit(x_train, (returns > 1.0).astype(int), weights)
    downside = fit_logit(x_train, (returns < -1.0).astype(int), weights)
    functions[CANDIDATE_IDS[3]] = (
        lambda x, c, up=upside, down=downside:
        up.predict_proba(x)[:, 1] - 1.5 * down.predict_proba(x)[:, 1]
    )

    clears_cost = returns > 0.20
    hurdle = fit_logit(x_train, clears_cost.astype(int), weights)
    positive = fit_ridge(
        x_train[clears_cost],
        clipped[clears_cost],
        weights[clears_cost],
    )
    nonpositive = fit_ridge(
        x_train[~clears_cost],
        clipped[~clears_cost],
        weights[~clears_cost],
    )
    functions[CANDIDATE_IDS[4]] = (
        lambda x, c, gate=hurdle, pos=positive, neg=nonpositive: (
            gate.predict_proba(x)[:, 1] * np.clip(pos.predict(x), -3.0, 3.0)
            + (1.0 - gate.predict_proba(x)[:, 1])
            * np.clip(neg.predict(x), -3.0, 3.0)
        )
    )

    pair_x, pair_y, pair_w = build_pairwise(
        x_train,
        returns,
        training,
        extreme_only=False,
        seed=3100 + fold_number,
    )
    pair_model = fit_logit(
        pair_x, pair_y, pair_w, fit_intercept=False
    )
    functions[CANDIDATE_IDS[5]] = (
        lambda x, c, m=pair_model: m.decision_function(x)
    )
    details["m06_pairs"] = int(len(pair_y) // 2)

    extreme_x, extreme_y, extreme_w = build_pairwise(
        x_train,
        returns,
        training,
        extreme_only=True,
        seed=7100 + fold_number,
    )
    extreme_model = fit_logit(
        extreme_x, extreme_y, extreme_w, fit_intercept=False
    )
    functions[CANDIDATE_IDS[6]] = (
        lambda x, c, m=extreme_model: m.decision_function(x)
    )
    details["m07_pairs"] = int(len(extreme_y) // 2)
    del pair_x, pair_y, pair_w, extreme_x, extreme_y, extreme_w

    expectile: Ridge | None = None
    iterative_weights = weights
    for _ in range(4):
        expectile = fit_ridge(x_train, clipped, iterative_weights)
        residual = clipped - expectile.predict(x_train)
        modifier = np.where(residual >= 0.0, 0.35, 0.65)
        iterative_weights = renormalise_date_weights(training, weights, modifier)
    assert expectile is not None
    functions[CANDIDATE_IDS[7]] = lambda x, c, m=expectile: m.predict(x)

    unique_dates = np.array(sorted(training["date"].unique()))
    temporal_models: list[Ridge] = []
    temporal_bounds: list[list[str]] = []
    for block_dates in np.array_split(unique_dates, 3):
        mask = training["date"].isin(block_dates).to_numpy()
        model = fit_ridge(x_train[mask], target_rank[mask], weights[mask])
        temporal_models.append(model)
        temporal_bounds.append(
            [str(pd.Timestamp(block_dates[0]).date()), str(pd.Timestamp(block_dates[-1]).date())]
        )
    functions[CANDIDATE_IDS[8]] = (
        lambda x, c, models=tuple(temporal_models):
        np.median(np.vstack([m.predict(x) for m in models]), axis=0)
    )
    details["m09_temporal_bounds"] = temporal_bounds

    daily_context = training[["date", "prior_market_dispersion"]].drop_duplicates("date")
    dispersion_threshold = float(daily_context["prior_market_dispersion"].median())
    low_mask = context_train <= dispersion_threshold
    if low_mask.sum() < 1000 or (~low_mask).sum() < 1000:
        raise ValueError("dispersion expert has insufficient training rows")
    low_expert = fit_ridge(
        x_train[low_mask], target_rank[low_mask], weights[low_mask]
    )
    high_expert = fit_ridge(
        x_train[~low_mask], target_rank[~low_mask], weights[~low_mask]
    )
    functions[CANDIDATE_IDS[9]] = (
        lambda x, c, low=low_expert, high=high_expert, threshold=dispersion_threshold:
        np.where(c <= threshold, low.predict(x), high.predict(x))
    )
    details["m10_dispersion_threshold"] = dispersion_threshold

    split_position = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * 0.70)))
    early_dates = unique_dates[:split_position]
    early_mask = training["date"].isin(early_dates).to_numpy()
    late_mask = ~early_mask
    early_mean = fit_ridge(
        x_train[early_mask], clipped[early_mask], weights[early_mask]
    )
    late_residual = np.abs(clipped[late_mask] - early_mean.predict(x_train[late_mask]))
    residual_model = fit_ridge(
        x_train[late_mask],
        np.clip(late_residual, 0.0, 3.0),
        weights[late_mask],
    )
    full_mean = fit_ridge(x_train, clipped, weights)
    functions[CANDIDATE_IDS[10]] = (
        lambda x, c, mean=full_mean, error=residual_model:
        mean.predict(x) - 0.5 * np.maximum(error.predict(x), 0.0)
    )
    details["m11_residual_split"] = {
        "early_end": str(pd.Timestamp(early_dates[-1]).date()),
        "late_start": str(pd.Timestamp(unique_dates[split_position]).date()),
    }

    q35 = hgb_quantile(0.35)
    q35.fit(x_train, clipped, sample_weight=weights)
    functions[CANDIDATE_IDS[11]] = lambda x, c, m=q35: m.predict(x)

    q20 = hgb_quantile(0.20)
    q50 = hgb_quantile(0.50)
    q20.fit(x_train, clipped, sample_weight=weights)
    q50.fit(x_train, clipped, sample_weight=weights)
    functions[CANDIDATE_IDS[12]] = (
        lambda x, c, low=q20, med=q50:
        med.predict(x) - 0.5 * (med.predict(x) - low.predict(x))
    )

    for candidate_id, scorer in functions.items():
        value = np.asarray(scorer(x_score, context_score), dtype=float)
        if value.shape != (len(scoring),) or not np.isfinite(value).all():
            raise ValueError(f"{candidate_id} produced invalid scores")
        scores[candidate_id] = value
    if set(scores) != set(ALL_IDS):
        raise AssertionError("candidate score set differs from protocol")
    return scores, functions, details


def rank_top_two(
    scoring: pd.DataFrame, scores: np.ndarray, candidate_id: str
) -> pd.DataFrame:
    ranked = scoring.assign(model_score=scores).sort_values(
        ["date", "model_score", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    selected = ranked.groupby("date", sort=True, as_index=False).head(2).copy()
    selected["model_rank"] = selected.groupby("date", sort=False).cumcount() + 1
    selected["candidate_id"] = candidate_id
    return selected


def desired_slots(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.MultiIndex.from_product(
        [sessions, (1, 2)], names=["date", "model_rank"]
    ).to_frame(index=False)


def daily_returns(picks: pd.DataFrame, cost_bps: float) -> pd.Series:
    executed = picks["label"].notna()
    slot = picks["oc_return_pct"].fillna(0.0) - executed.astype(float) * cost_bps / 100.0
    return slot.groupby(picks["date"], sort=True).sum().div(2.0)


def profit_factor(values: pd.Series) -> float | None:
    gains = float(values.clip(lower=0).sum())
    losses = float(-values.clip(upper=0).sum())
    return gains / losses if losses > 0 else None


def candidate_metrics(picks: pd.DataFrame) -> dict[str, Any]:
    executed = picks["label"].notna()
    output: dict[str, Any] = {
        "displayed_slots": int(len(picks)),
        "executed_slots": int(executed.sum()),
        "executed_hit_rate": float(picks.loc[executed, "label"].mean()),
        "unique_codes": int(picks["code"].nunique()),
        "cost": {},
        "period_net20": {},
        "monthly_net20": {},
        "tail_exclusion": {},
    }
    for cost in (20.0, 40.0, 60.0):
        daily = daily_returns(picks, cost)
        equity = (1.0 + daily / 100.0).cumprod()
        output["cost"][str(int(cost))] = {
            "mean_pct": float(daily.mean()),
            "median_pct": float(daily.median()),
            "compounded_pct": float((equity.iloc[-1] - 1.0) * 100.0),
            "profit_factor": profit_factor(daily),
            "positive_days": int(daily.gt(0).sum()),
            "days": int(len(daily)),
        }
        output["tail_exclusion"][str(int(cost))] = {
            str(count): float(daily.drop(daily.nlargest(count).index).mean())
            for count in (5, 10, 20)
        }
    daily20 = daily_returns(picks, 20.0)
    for period_id, (start, end) in PERIODS.items():
        output["period_net20"][period_id] = float(
            daily20.loc[(daily20.index >= start) & (daily20.index <= end)].mean()
        )
    monthly = daily20.groupby(daily20.index.to_period("M")).mean()
    output["monthly_net20"] = {str(key): float(value) for key, value in monthly.items()}
    output["positive_months_net20"] = int(monthly.gt(0).sum())
    output["months"] = int(len(monthly))
    slots = picks.assign(
        net20_slot=picks["oc_return_pct"].fillna(0.0)
        - executed.astype(float) * 0.20
    )
    by_code = slots.groupby("code", sort=False)["net20_slot"].sum()
    total = float(by_code.sum())
    output["largest_code_share_total_net20_slot"] = (
        float(by_code.max() / total) if total > 0 else None
    )
    return output


def hac_mean_se(values: np.ndarray, lag: int = 5) -> float:
    x = np.asarray(values, dtype=float)
    n = len(x)
    centred = x - x.mean()
    long_run = float(np.dot(centred, centred) / n)
    for offset in range(1, min(lag, n - 1) + 1):
        covariance = float(np.dot(centred[offset:], centred[:-offset]) / n)
        long_run += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    return math.sqrt(max(long_run, 1e-18) / n)


def multiplicity(
    picks_by_candidate: dict[str, pd.DataFrame],
    sessions: pd.DatetimeIndex,
) -> dict[str, dict[str, float]]:
    control = daily_returns(picks_by_candidate[CONTROL_ID], 40.0).reindex(sessions)
    matrix = np.column_stack(
        [
            (
                daily_returns(picks_by_candidate[candidate_id], 40.0).reindex(sessions)
                - control
            ).to_numpy(float)
            for candidate_id in CANDIDATE_IDS
        ]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("daily uplift matrix is incomplete")
    point = matrix.mean(axis=0)
    standard_error = np.array([hac_mean_se(matrix[:, i]) for i in range(matrix.shape[1])])
    rng = np.random.default_rng(20260723)
    n = len(sessions)
    block = 10
    draws = 2000
    bootstrap = np.empty((draws, len(CANDIDATE_IDS)), dtype=float)
    blocks_needed = int(math.ceil(n / block))
    offsets = np.arange(block)
    for draw in range(draws):
        starts = rng.integers(0, n, size=blocks_needed)
        positions = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        bootstrap[draw] = matrix[positions].mean(axis=0)
    centred_t = (bootstrap - point) / standard_error
    max_t = centred_t.max(axis=1)
    q80 = float(np.quantile(max_t, 0.80))
    q90 = float(np.quantile(max_t, 0.90))
    output: dict[str, dict[str, float]] = {}
    for index, candidate_id in enumerate(CANDIDATE_IDS):
        output[candidate_id] = {
            "net40_uplift_vs_control_pct": float(point[index]),
            "hac5_standard_error_pct": float(standard_error[index]),
            "ordinary_moving_block_80_lower_pct": float(
                np.quantile(bootstrap[:, index], 0.20)
            ),
            "ordinary_moving_block_90_lower_pct": float(
                np.quantile(bootstrap[:, index], 0.10)
            ),
            "familywise_max_t_80_lower_pct": float(
                point[index] - q80 * standard_error[index]
            ),
            "familywise_max_t_90_lower_pct": float(
                point[index] - q90 * standard_error[index]
            ),
        }
    return output


def selection_overlap(
    candidate: pd.DataFrame, control: pd.DataFrame
) -> dict[str, float | int]:
    left = candidate.groupby("date", sort=True)["code"].apply(lambda x: set(x.dropna()))
    right = control.groupby("date", sort=True)["code"].apply(lambda x: set(x.dropna()))
    common = [len(a & b) for a, b in zip(left, right, strict=True)]
    return {
        "shared_slots": int(sum(common)),
        "total_candidate_slots": int(candidate["code"].notna().sum()),
        "mean_shared_names_per_day": float(np.mean(common)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="/tmp/model_v07_corrected_panel.pkl")
    parser.add_argument(
        "--protocol", default="/tmp/v10_model_architectures/protocol.json"
    )
    parser.add_argument(
        "--output", default="/tmp/v10_model_architectures/result.json"
    )
    parser.add_argument(
        "--picks-output", default="/tmp/v10_model_architectures/picks.csv"
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
    sessions, scheduled = validate_input(panel, manifest, protocol)
    score_periods = pd.period_range("2024-07", "2025-07", freq="M")
    projection = [
        "date",
        "code",
        "name",
        "label",
        "oc_return_pct",
        "price_eligible",
        "price_training_eligible",
        "prior_market_dispersion",
        *FEATURES,
    ]
    parts: dict[str, list[pd.DataFrame]] = {candidate_id: [] for candidate_id in ALL_IDS}
    folds: list[dict[str, Any]] = []
    mutation_audit: dict[str, Any] = {}
    for fold_number, period in enumerate(score_periods, start=1):
        fold_started = time.time()
        score_start = period.start_time.normalize()
        score_end = period.end_time.normalize()
        training = panel.loc[
            panel["date"].between(pd.Timestamp("2024-01-04"), score_start - pd.Timedelta(days=1))
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
            raise ValueError(f"fold {period} training is not strictly prior")
        expected_dates = scheduled[scheduled.to_period("M") == period]
        if not pd.DatetimeIndex(scoring["date"].unique()).isin(expected_dates).all():
            raise ValueError(f"fold {period} contains an unexpected score date")
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        scaler = StandardScaler()
        x_train = imputer.fit_transform(training.loc[:, FEATURES])
        x_train = scaler.fit_transform(x_train).astype("float32", copy=False)
        x_score = imputer.transform(scoring.loc[:, FEATURES])
        x_score = scaler.transform(x_score).astype("float32", copy=False)
        candidate_scores, score_functions, details = fit_candidates(
            training,
            scoring,
            x_train,
            x_score,
            fold_number=fold_number,
        )
        for candidate_id, values in candidate_scores.items():
            parts[candidate_id].append(rank_top_two(scoring, values, candidate_id))
        if str(period) == "2025-07":
            changed = scoring.copy()
            changed["oc_return_pct"] = np.where(
                np.arange(len(changed)) % 2 == 0, 99.0, -99.0
            )
            changed["label"] = changed["oc_return_pct"].gt(0).astype(float)
            changed_x = scaler.transform(
                imputer.transform(changed.loc[:, FEATURES])
            ).astype("float32", copy=False)
            changed_context = changed["prior_market_dispersion"].to_numpy(float)
            if not np.array_equal(x_score, changed_x):
                raise ValueError("target-day outcome mutation changed model features")
            for candidate_id, scorer in score_functions.items():
                changed_scores = np.asarray(
                    scorer(changed_x, changed_context), dtype=float
                )
                maximum = float(
                    np.max(np.abs(changed_scores - candidate_scores[candidate_id]))
                )
                original_keys = rank_top_two(
                    scoring, candidate_scores[candidate_id], candidate_id
                )[["date", "model_rank", "code"]]
                changed_keys = rank_top_two(
                    changed, changed_scores, candidate_id
                )[["date", "model_rank", "code"]]
                mutation_audit[candidate_id] = {
                    "max_abs_score_difference": maximum,
                    "top2_keys_exact": bool(original_keys.equals(changed_keys)),
                }
        folds.append(
            {
                "period": str(period),
                "train_start": str(training["date"].min().date()),
                "train_end": str(training["date"].max().date()),
                "training_rows": int(len(training)),
                "score_rows": int(len(scoring)),
                "score_sessions": int(scoring["date"].nunique()),
                "strictly_prior_training": True,
                "details": details,
                "runtime_seconds": float(time.time() - fold_started),
            }
        )
        print(
            f"completed {period}: train={len(training):,} score={len(scoring):,} "
            f"seconds={time.time() - fold_started:.1f}",
            flush=True,
        )
        del (
            training,
            scoring,
            x_train,
            x_score,
            candidate_scores,
            score_functions,
            imputer,
            scaler,
        )
        gc.collect()

    desired = desired_slots(scheduled)
    picks_by_candidate: dict[str, pd.DataFrame] = {}
    all_picks: list[pd.DataFrame] = []
    keep = [
        "date",
        "model_rank",
        "code",
        "name",
        "model_score",
        "label",
        "oc_return_pct",
    ]
    for candidate_id in ALL_IDS:
        actual = pd.concat(parts[candidate_id], ignore_index=True)
        picks = desired.merge(
            actual[keep],
            on=["date", "model_rank"],
            how="left",
            sort=True,
            validate="one_to_one",
        )
        picks["candidate_id"] = candidate_id
        if len(picks) != 2 * len(scheduled):
            raise ValueError(f"{candidate_id} does not have two scheduled slots")
        picks_by_candidate[candidate_id] = picks
        all_picks.append(picks)
    all_pick_frame = pd.concat(all_picks, ignore_index=True)
    all_pick_frame.to_csv(picks_path, index=False)

    metrics = {
        candidate_id: candidate_metrics(picks)
        for candidate_id, picks in picks_by_candidate.items()
    }
    multiplicity_result = multiplicity(picks_by_candidate, scheduled)
    gates: dict[str, dict[str, Any]] = {}
    for candidate_id in CANDIDATE_IDS:
        metric = metrics[candidate_id]
        checks = {
            "net40_positive": metric["cost"]["40"]["mean_pct"] > 0,
            "top20_removed_net20_positive": (
                metric["tail_exclusion"]["20"]["20"] > 0
            ),
            "at_least_9_positive_months_net20": metric["positive_months_net20"] >= 9,
            "all_three_periods_net20_positive": all(
                value > 0 for value in metric["period_net20"].values()
            ),
            "familywise_80_lower_uplift_positive": (
                multiplicity_result[candidate_id][
                    "familywise_max_t_80_lower_pct"
                ]
                > 0
            ),
        }
        gates[candidate_id] = {
            "checks": checks,
            "passes_all": all(checks.values()),
        }
    passing = [
        candidate_id
        for candidate_id in CANDIDATE_IDS
        if gates[candidate_id]["passes_all"]
    ]
    selected = (
        max(passing, key=lambda key: metrics[key]["cost"]["40"]["mean_pct"])
        if passing
        else None
    )
    point_best = max(
        CANDIDATE_IDS, key=lambda key: metrics[key]["cost"]["40"]["mean_pct"]
    )
    for candidate_id in CANDIDATE_IDS:
        metrics[candidate_id]["overlap_vs_control"] = selection_overlap(
            picks_by_candidate[candidate_id], picks_by_candidate[CONTROL_ID]
        )
    if (
        set(mutation_audit) != set(ALL_IDS)
        or any(
            item["max_abs_score_difference"] != 0.0
            or item["top2_keys_exact"] is not True
            for item in mutation_audit.values()
        )
    ):
        raise ValueError("target-day outcome mutation audit failed")
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "authority": "retrospective_hypothesis_generation_only",
        "production_model_changed": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "panel_sha256": PANEL_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "integrity": {
            "rows": int(len(panel)),
            "codes": int(panel["code"].nunique()),
            "sessions": int(len(sessions)),
            "score_sessions": int(len(scheduled)),
            "monthly_folds": int(len(folds)),
            "candidate_count_excluding_control": int(len(CANDIDATE_IDS)),
            "fixed_top2_equal_weight": True,
            "strictly_prior_training_all_folds": all(
                fold["strictly_prior_training"] for fold in folds
            ),
            "strict_prior_candidate_feature_source_violations": 0,
            "target_day_outcome_mutation": mutation_audit,
        },
        "decision": {
            "robustness_gate_winner": selected,
            "passing_candidate_ids": passing,
            "highest_unadjusted_net40_candidate": point_best,
            "highest_unadjusted_net40_pct": metrics[point_best]["cost"]["40"][
                "mean_pct"
            ],
            "production_action": "none",
        },
        "metrics": metrics,
        "multiplicity": multiplicity_result,
        "robustness_gates": gates,
        "folds": folds,
        "artifacts": {
            "picks_path": str(picks_path),
            "picks_sha256": sha256_file(picks_path),
        },
        "runtime_seconds": float(time.time() - started),
    }
    write_json(output_path, result)
    print(
        f"done: point_best={point_best} robust_winner={selected} "
        f"seconds={time.time() - started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
