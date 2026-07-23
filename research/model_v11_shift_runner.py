#!/usr/bin/env python3
"""Run the preregistered v1.1 structural/covariate-shift screen."""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import time
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.preprocessing import StandardScaler

from research.model_v11_distributional_runner import (
    Prediction,
    daily_returns,
    date_equal_weights,
    desired_slots,
    exact_file,
    finite_sample_quantile,
    fit_ridge,
    hac_mean_se,
    policy_id,
    policy_metrics,
    rank_target,
    read_json,
    select_policy_rows,
    sha256_file,
    write_json,
)


PROTOCOL_SHA256 = "76d8c4129f57526c6d0a4d826779414c3ee4dcba48e818c72697ba8d3627a0cc"
PANEL_SHA256 = "6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb"
MANIFEST_SHA256 = "25e08c564ef6400b7a29168db9fd7e8220bd0e2386c7a3c7e71b05c2ff950b02"
SEED = 20260723
CONTROL = "C00_DAILY_RANK_RIDGE"
CANDIDATES = (
    "S01_DENSITY_RATIO_RECENT",
    "S02_COVARIATE_CHANGEPOINT_RESET",
    "S03_WORST_MONTH_GROUP_DRO",
    "S04_ERA_INVARIANT_SIGN",
    "S05_LEAVE_ONE_STATE_OUT_MEDIAN_COEF",
    "S06_UNSUPERVISED_LATENT_EXPERTS",
    "S07_RECENT_SUPPORT_CONFORMAL",
    "S08_ENVIRONMENT_RESIDUALISED",
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
STATE_FEATURES = (
    "prior_market_breadth",
    "prior_market_dispersion",
    "prior_market_tail_balance",
    "market_cc_vol_ratio_5_20",
    "market_cc_momentum_5",
    "no_trade_rate_20",
)
Scorer = Callable[[pd.DataFrame, np.ndarray], Prediction]


def validate_input(
    panel: pd.DataFrame,
    manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    if protocol.get("protocol_id") != "model_v11_shift_zero_base_20260723":
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
        raise ValueError("panel row count differs from lock")
    if int(panel["code"].nunique()) != int(frozen["codes"]):
        raise ValueError("panel code count differs from lock")
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
        *STATE_FEATURES,
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel lacks required columns: {missing}")
    if panel.duplicated(["date", "code"]).any():
        raise ValueError("panel contains duplicate date/code")
    dates = pd.to_datetime(panel["date"], errors="coerce")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("panel dates are invalid")
    sessions = pd.DatetimeIndex(dates.drop_duplicates().sort_values())
    manifest_sessions = pd.DatetimeIndex(pd.to_datetime(manifest["sessions"]))
    if not sessions.equals(manifest_sessions):
        raise ValueError("manifest sessions differ from panel")
    returns = pd.to_numeric(panel["oc_return_pct"], errors="coerce")
    labels = pd.to_numeric(panel["label"], errors="coerce")
    if returns.isna().ne(labels.isna()).any():
        raise ValueError("label/return missingness differs")
    observed = returns.notna()
    if not np.array_equal(
        labels.loc[observed].to_numpy(float),
        returns.loc[observed].gt(0.0).to_numpy(float),
    ):
        raise ValueError("label is not the return sign")
    source = pd.to_datetime(panel["candidate_price_source_max_date"], errors="coerce")
    if (source.notna() & source.ge(dates)).any():
        raise ValueError("candidate feature source is not strictly prior")
    start, end = map(pd.Timestamp, frozen["score_period"])
    scheduled = sessions[(sessions >= start) & (sessions <= end)]
    if len(scheduled) != int(frozen["score_sessions"]):
        raise ValueError("score-session count differs from protocol")
    return sessions, scheduled


def chronological_date_mask(frame: pd.DataFrame, fraction: float) -> np.ndarray:
    dates = pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())
    cut = min(max(int(math.floor(len(dates) * fraction)), 1), len(dates) - 1)
    early = set(dates[:cut])
    return frame["date"].isin(early).to_numpy(bool)


def normalise_within_date(frame: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=float)
    if (values <= 0.0).any() or not np.isfinite(values).all():
        raise ValueError("invalid positive weights")
    dates = frame["date"].reset_index(drop=True)
    totals = pd.Series(values).groupby(dates, sort=False).transform("sum").to_numpy()
    result = values / totals
    check = pd.Series(result).groupby(dates, sort=False).sum().to_numpy()
    if not np.allclose(check, 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("within-date weights do not sum to one")
    return result


def environment_classifier(
    frame: pd.DataFrame,
    x: np.ndarray,
) -> tuple[SGDClassifier, np.ndarray, dict[str, Any]]:
    old = chronological_date_mask(frame, 0.70)
    environment = (~old).astype(int)
    base = date_equal_weights(frame)
    weights = np.empty(len(frame), dtype=float)
    for label in (0, 1):
        mask = environment == label
        weights[mask] = 0.5 * len(frame) * base[mask] / base[mask].sum()
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=0.0001,
        max_iter=100,
        tol=0.0001,
        random_state=SEED,
        fit_intercept=True,
        average=True,
    )
    classifier.fit(x, environment, sample_weight=weights)
    details = {
        "old_rows": int(old.sum()),
        "recent_rows": int((~old).sum()),
        "old_dates": int(frame.loc[old, "date"].nunique()),
        "recent_dates": int(frame.loc[~old, "date"].nunique()),
    }
    return classifier, environment, details


def daily_state_transform(
    training: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    SimpleImputer,
    StandardScaler,
]:
    daily = (
        training[["date", *STATE_FEATURES]]
        .sort_values("date", kind="stable")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    transformed = imputer.fit_transform(daily.loc[:, STATE_FEATURES])
    transformed = scaler.fit_transform(transformed).astype("float32", copy=False)
    if not np.isfinite(transformed).all():
        raise ValueError("daily market-state transform is non-finite")
    return daily, transformed, imputer, scaler


def fit_fold(
    training: pd.DataFrame,
    scoring: pd.DataFrame,
    x_train: np.ndarray,
    x_score: np.ndarray,
    base_weights: np.ndarray,
) -> tuple[dict[str, Prediction], dict[str, Scorer], dict[str, Any]]:
    target = rank_target(training)
    predictions: dict[str, Prediction] = {}
    scorers: dict[str, Scorer] = {}
    details: dict[str, Any] = {}
    control_model = fit_ridge(x_train, target, base_weights, alpha=1.0)

    def control_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
        return Prediction(
            control_model.predict(x), np.ones(len(frame), dtype=bool)
        )

    scorers[CONTROL] = control_scorer

    env_model, environment, env_details = environment_classifier(training, x_train)
    recent_probability = np.clip(
        env_model.predict_proba(x_train)[:, 1], 1e-6, 1.0 - 1e-6
    )
    density_ratio = np.clip(
        recent_probability / (1.0 - recent_probability), 0.25, 4.0
    )
    density_weights = normalise_within_date(
        training, base_weights * density_ratio
    )
    density_model = fit_ridge(x_train, target, density_weights, alpha=1.0)

    def s01_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
        return Prediction(
            density_model.predict(x), np.ones(len(frame), dtype=bool)
        )

    scorers["S01_DENSITY_RATIO_RECENT"] = s01_scorer
    details["S01_DENSITY_RATIO_RECENT"] = {
        **env_details,
        "density_ratio_min": float(density_ratio.min()),
        "density_ratio_median": float(np.median(density_ratio)),
        "density_ratio_max": float(density_ratio.max()),
    }

    daily, daily_x, state_imputer, state_scaler = daily_state_transform(training)
    n_dates = len(daily)
    reset_allowed = n_dates >= 80
    reset_position: int | None = None
    reset_statistic: float | None = None
    reset_date: pd.Timestamp | None = None
    if reset_allowed:
        cumulative = np.vstack(
            [np.zeros((1, daily_x.shape[1])), np.cumsum(daily_x, axis=0)]
        )
        positions = np.arange(40, n_dates - 39)
        statistics = []
        for position in positions:
            pre = cumulative[position] / position
            post = (cumulative[n_dates] - cumulative[position]) / (
                n_dates - position
            )
            scale = math.sqrt(position * (n_dates - position) / n_dates)
            statistics.append(scale * float(np.linalg.norm(post - pre)))
        best = int(np.argmax(statistics))
        reset_position = int(positions[best])
        reset_statistic = float(statistics[best])
        reset_date = pd.Timestamp(daily.loc[reset_position, "date"])
        reset_mask = training["date"].ge(reset_date).to_numpy(bool)
        reset_allowed = bool(
            int(training.loc[reset_mask, "date"].nunique()) >= 20
            and bool(reset_mask.any())
        )
    if reset_allowed and reset_date is not None:
        reset_weights = date_equal_weights(training.loc[reset_mask].reset_index(drop=True))
        reset_model = fit_ridge(
            x_train[reset_mask], target[reset_mask], reset_weights, alpha=1.0
        )

        def s02_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
            return Prediction(
                reset_model.predict(x), np.ones(len(frame), dtype=bool)
            )

    else:

        def s02_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
            return Prediction(np.zeros(len(frame)), np.zeros(len(frame), dtype=bool))

    scorers["S02_COVARIATE_CHANGEPOINT_RESET"] = s02_scorer
    details["S02_COVARIATE_CHANGEPOINT_RESET"] = {
        "training_dates": n_dates,
        "reset_position": reset_position,
        "reset_date": str(reset_date.date()) if reset_date is not None else None,
        "contrast_statistic": reset_statistic,
        "fit_allowed": reset_allowed,
    }

    month = training["date"].dt.to_period("M")
    month_values = pd.Index(month.drop_duplicates().sort_values())
    group_weight = pd.Series(
        1.0 / len(month_values), index=month_values, dtype=float
    )
    group_model: Ridge | None = None
    loss_history: list[dict[str, float]] = []
    month_date_counts = training[["date"]].drop_duplicates().assign(
        month=lambda value: value["date"].dt.to_period("M")
    )["month"].value_counts()
    for _ in range(6):
        multiplier = month.map(
            group_weight / month_date_counts.reindex(month_values)
        ).to_numpy(float)
        dro_weights = base_weights * multiplier
        group_model = fit_ridge(x_train, target, dro_weights, alpha=1.0)
        squared = np.square(target - group_model.predict(x_train))
        losses: dict[pd.Period, float] = {}
        for value in month_values:
            mask = month.eq(value).to_numpy(bool)
            losses[value] = float(
                np.average(squared[mask], weights=base_weights[mask])
            )
        loss_series = pd.Series(losses)
        loss_history.append(
            {str(key): float(value) for key, value in loss_series.items()}
        )
        scaled = loss_series / max(float(loss_series.mean()), 1e-12) - 1.0
        update = np.exp(np.clip(2.0 * scaled, -20.0, 20.0))
        group_weight = group_weight * update
        group_weight = group_weight / group_weight.sum()
    multiplier = month.map(
        group_weight / month_date_counts.reindex(month_values)
    ).to_numpy(float)
    group_model = fit_ridge(
        x_train, target, base_weights * multiplier, alpha=1.0
    )

    def s03_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
        return Prediction(
            group_model.predict(x), np.ones(len(frame), dtype=bool)
        )

    scorers["S03_WORST_MONTH_GROUP_DRO"] = s03_scorer
    details["S03_WORST_MONTH_GROUP_DRO"] = {
        "groups": int(len(month_values)),
        "final_group_weights": {
            str(key): float(value) for key, value in group_weight.items()
        },
        "loss_history": loss_history,
    }

    training_dates = pd.DatetimeIndex(
        training["date"].drop_duplicates().sort_values()
    )
    eras = np.array_split(training_dates, 3)
    era_models: list[Ridge] = []
    era_details: list[dict[str, Any]] = []
    for era in eras:
        mask = training["date"].isin(set(era)).to_numpy(bool)
        era_weights = date_equal_weights(training.loc[mask].reset_index(drop=True))
        era_models.append(
            fit_ridge(x_train[mask], target[mask], era_weights, alpha=1.0)
        )
        era_details.append(
            {
                "start": str(pd.Timestamp(era[0]).date()),
                "end": str(pd.Timestamp(era[-1]).date()),
                "dates": int(len(era)),
                "rows": int(mask.sum()),
            }
        )
    coefficient_sign = np.vstack([np.sign(model.coef_) for model in era_models])
    invariant = np.all(coefficient_sign > 0.0, axis=0) | np.all(
        coefficient_sign < 0.0, axis=0
    )
    invariant_allowed = bool(invariant.any())
    if invariant_allowed:
        invariant_model = fit_ridge(
            x_train[:, invariant], target, base_weights, alpha=1.0
        )

        def s04_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
            return Prediction(
                invariant_model.predict(x[:, invariant]),
                np.ones(len(frame), dtype=bool),
            )

    else:

        def s04_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
            return Prediction(np.zeros(len(frame)), np.zeros(len(frame), dtype=bool))

    scorers["S04_ERA_INVARIANT_SIGN"] = s04_scorer
    details["S04_ERA_INVARIANT_SIGN"] = {
        "eras": era_details,
        "transformed_features": int(x_train.shape[1]),
        "retained_features": int(invariant.sum()),
        "fit_allowed": invariant_allowed,
    }

    kmeans = KMeans(n_clusters=3, n_init=20, random_state=SEED)
    daily_state = kmeans.fit_predict(daily_x)
    date_to_state = dict(zip(daily["date"], daily_state, strict=True))
    training_state = training["date"].map(date_to_state).to_numpy(int)

    def predict_state(frame: pd.DataFrame) -> np.ndarray:
        score_daily = (
            frame[["date", *STATE_FEATURES]]
            .sort_values("date", kind="stable")
            .drop_duplicates("date")
            .reset_index(drop=True)
        )
        value = state_imputer.transform(score_daily.loc[:, STATE_FEATURES])
        value = state_scaler.transform(value)
        states = kmeans.predict(value)
        mapping = dict(zip(score_daily["date"], states, strict=True))
        return frame["date"].map(mapping).to_numpy(int)

    omission_models: list[Ridge] = []
    for state in range(3):
        keep = training_state != state
        omission_models.append(
            fit_ridge(
                x_train[keep], target[keep], base_weights[keep], alpha=1.0
            )
        )
    robust_coefficient = np.median(
        np.vstack([model.coef_ for model in omission_models]), axis=0
    )
    robust_intercept = float(
        np.median([float(model.intercept_) for model in omission_models])
    )

    def s05_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
        score = x @ robust_coefficient + robust_intercept
        return Prediction(score, np.ones(len(frame), dtype=bool))

    scorers["S05_LEAVE_ONE_STATE_OUT_MEDIAN_COEF"] = s05_scorer
    details["S05_LEAVE_ONE_STATE_OUT_MEDIAN_COEF"] = {
        "state_dates": {
            str(state): int((daily_state == state).sum()) for state in range(3)
        },
        "coefficient_aggregation": "coordinate_median",
    }

    state_date_counts = {
        state: int((daily_state == state).sum()) for state in range(3)
    }
    experts_allowed = all(value >= 20 for value in state_date_counts.values())
    experts: dict[int, Ridge] = {}
    if experts_allowed:
        for state in range(3):
            mask = training_state == state
            experts[state] = fit_ridge(
                x_train[mask], target[mask], base_weights[mask], alpha=1.0
            )

        def s06_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
            state = predict_state(frame)
            score = np.empty(len(frame), dtype=float)
            for value in range(3):
                mask = state == value
                score[mask] = experts[value].predict(x[mask])
            return Prediction(score, np.ones(len(frame), dtype=bool))

    else:

        def s06_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
            return Prediction(np.zeros(len(frame)), np.zeros(len(frame), dtype=bool))

    scorers["S06_UNSUPERVISED_LATENT_EXPERTS"] = s06_scorer
    details["S06_UNSUPERVISED_LATENT_EXPERTS"] = {
        "state_dates": {str(key): value for key, value in state_date_counts.items()},
        "fit_allowed": experts_allowed,
    }

    recent = ~chronological_date_mask(training, 0.75)
    recent_centre = x_train[recent].mean(axis=0)
    recent_scale = np.maximum(x_train[recent].std(axis=0), 0.10)
    recent_distance = np.sqrt(
        np.mean(
            np.square((x_train[recent] - recent_centre) / recent_scale), axis=1
        )
    )
    support_quantile = finite_sample_quantile(recent_distance, 0.90)

    def s07_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
        distance = np.sqrt(
            np.mean(np.square((x - recent_centre) / recent_scale), axis=1)
        )
        return Prediction(control_model.predict(x), distance <= support_quantile)

    scorers["S07_RECENT_SUPPORT_CONFORMAL"] = s07_scorer
    details["S07_RECENT_SUPPORT_CONFORMAL"] = {
        "reference_rows": int(recent.sum()),
        "support_q90": support_quantile,
    }

    environment_direction = np.asarray(env_model.coef_[0], dtype=float)
    direction_norm = float(np.dot(environment_direction, environment_direction))
    residual_allowed = direction_norm > 1e-18
    if residual_allowed:

        def residualise(x: np.ndarray) -> np.ndarray:
            projection = (x @ environment_direction) / direction_norm
            return x - projection[:, None] * environment_direction[None, :]

        residual_train = residualise(x_train)
        residual_model = fit_ridge(
            residual_train, target, base_weights, alpha=1.0
        )

        def s08_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
            return Prediction(
                residual_model.predict(residualise(x)),
                np.ones(len(frame), dtype=bool),
            )

    else:

        def s08_scorer(frame: pd.DataFrame, x: np.ndarray) -> Prediction:
            return Prediction(np.zeros(len(frame)), np.zeros(len(frame), dtype=bool))

    scorers["S08_ENVIRONMENT_RESIDUALISED"] = s08_scorer
    details["S08_ENVIRONMENT_RESIDUALISED"] = {
        **env_details,
        "environment_direction_squared_norm": direction_norm,
        "fit_allowed": residual_allowed,
    }

    for base_id, scorer in scorers.items():
        prediction = scorer(scoring, x_score)
        if prediction.score.shape != (len(scoring),):
            raise ValueError(f"{base_id} score has invalid shape")
        if prediction.allowed.shape != (len(scoring),):
            raise ValueError(f"{base_id} mask has invalid shape")
        if not np.isfinite(prediction.score).all():
            raise ValueError(f"{base_id} score is non-finite")
        predictions[base_id] = prediction
    if set(predictions) != set(BASE_MODELS):
        raise AssertionError("candidate set differs from protocol")
    return predictions, scorers, details


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
    point = deltas.mean(axis=1)
    standard_error = np.array([hac_mean_se(row) for row in deltas])
    n = deltas.shape[1]
    rng = np.random.default_rng(SEED)
    blocks = int(math.ceil(n / block_length))
    offsets = np.arange(block_length)
    boot = np.empty((repetitions, len(candidate_policies)), dtype=float)
    for repetition in range(repetitions):
        starts = rng.integers(0, n, size=blocks)
        positions = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        boot[repetition] = deltas[:, positions].mean(axis=1)
    centred_t = (boot - point) / standard_error
    max_t = centred_t.max(axis=1)
    critical = float(np.quantile(max_t, 0.95))
    hypotheses: dict[str, Any] = {}
    for index, key in enumerate(candidate_policies):
        observed_t = point[index] / standard_error[index]
        hypotheses[key] = {
            "net40_uplift_vs_matched_control_pct": float(point[index]),
            "hac5_standard_error_pct": float(standard_error[index]),
            "ordinary_one_sided95_lower_pct": float(
                np.quantile(boot[:, index], 0.05)
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
            largest_weight = concentration["largest_weight_share"]
            top10_weight = concentration["top10_weight_share"]
            largest_pnl = concentration["largest_positive_net20_pnl_share"]
            lower = multiplicity["hypotheses"][key][
                "familywise_max_t_one_sided95_lower_pct"
            ]
            checks = {
                "net40_mean_positive": value["net_mean_pct"]["40"] > 0.0,
                "net60_mean_positive": value["net_mean_pct"]["60"] > 0.0,
                "net40_greater_than_capacity_matched_control": (
                    value["net_mean_pct"]["40"] > control["net_mean_pct"]["40"]
                ),
                "familywise_95_lower_net40_uplift_positive": lower > 0.0,
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
    parser.add_argument("--protocol", default="research/model_v11_shift_protocol.json")
    parser.add_argument("--output", default="research/model_v11_shift_result.json")
    parser.add_argument(
        "--picks-output", default="research/model_v11_shift_picks.csv"
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
    exact_file(manifest_path, MANIFEST_SHA256, "manifest")
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
        *FEATURES,
        *STATE_FEATURES,
    ]
    months = pd.period_range("2024-07", "2025-07", freq="M")
    parts: dict[str, list[pd.DataFrame]] = {
        policy_id(base, capacity): []
        for base in BASE_MODELS
        for capacity in CAPACITIES
    }
    folds: list[dict[str, Any]] = []
    mutation_audit: dict[str, Any] = {}
    for fold_number, month in enumerate(months, start=1):
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
        expected = scheduled[scheduled.to_period("M") == month]
        actual_dates = pd.DatetimeIndex(
            scoring["date"].drop_duplicates().sort_values()
        )
        if not actual_dates.equals(expected):
            raise ValueError(f"{month} score-date set differs from schedule")
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(
            imputer.fit_transform(training.loc[:, FEATURES])
        ).astype("float32", copy=False)
        x_score = scaler.transform(
            imputer.transform(scoring.loc[:, FEATURES])
        ).astype("float32", copy=False)
        base_weights = date_equal_weights(training)
        predictions, scorers, details = fit_fold(
            training, scoring, x_train, x_score, base_weights
        )
        for base_id, prediction in predictions.items():
            for capacity in CAPACITIES:
                parts[policy_id(base_id, capacity)].append(
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
                raise ValueError("target mutation changed feature matrix")
            for base_id, scorer in scorers.items():
                original = predictions[base_id]
                mutated = scorer(changed, changed_x)
                capacity_exact: dict[str, bool] = {}
                for capacity in CAPACITIES:
                    original_keys = select_policy_rows(
                        scoring, original, base_id, capacity
                    )[["date", "model_rank", "code"]].reset_index(drop=True)
                    mutated_keys = select_policy_rows(
                        changed, mutated, base_id, capacity
                    )[["date", "model_rank", "code"]].reset_index(drop=True)
                    capacity_exact[str(capacity)] = bool(
                        original_keys.equals(mutated_keys)
                    )
                mutation_audit[base_id] = {
                    "max_abs_score_difference": float(
                        np.max(np.abs(original.score - mutated.score))
                    ),
                    "trade_allowed_exact": bool(
                        np.array_equal(original.allowed, mutated.allowed)
                    ),
                    "selected_keys_exact_by_capacity": capacity_exact,
                }
        folds.append(
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
            base_weights,
            predictions,
            scorers,
            imputer,
            scaler,
        )
        gc.collect()
    mutation_exact = (
        set(mutation_audit) == set(BASE_MODELS)
        and all(
            value["max_abs_score_difference"] == 0.0
            and value["trade_allowed_exact"]
            and all(value["selected_keys_exact_by_capacity"].values())
            for value in mutation_audit.values()
        )
    )
    if not mutation_exact:
        raise ValueError("target mutation audit failed")

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
            picks = desired_slots(scheduled, capacity).merge(
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
                raise ValueError(f"{key} scheduled slots are incomplete")
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
    candidate_policies = [
        policy_id(candidate, capacity)
        for candidate in CANDIDATES
        for capacity in CAPACITIES
    ]
    point_winner = max(
        candidate_policies,
        key=lambda key: metrics[key]["net_mean_pct"]["40"],
    )
    common_runner = Path(
        __file__
    ).with_name("model_v11_distributional_runner.py")
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
            "evaluation_dependency_path": str(common_runner.resolve()),
            "evaluation_dependency_sha256": sha256_file(common_runner),
            "rows": int(len(panel)),
            "codes": int(panel["code"].nunique()),
            "sessions": int(len(all_sessions)),
            "score_sessions": int(len(scheduled)),
        },
        "integrity": {
            "monthly_expanding_folds": int(len(folds)),
            "all_training_strictly_prior": all(
                value["strictly_prior_training"] for value in folds
            ),
            "candidate_feature_source_violations": 0,
            "target_day_outcome_mutation_exact": mutation_exact,
            "target_day_outcome_mutation": mutation_audit,
            "cash_slots_not_renormalised": True,
            "candidate_policy_count": len(candidate_policies),
            "control_policy_count": len(CAPACITIES),
        },
        "decision_before_independent_audit": {
            "highest_unadjusted_net40_policy": point_winner,
            "highest_unadjusted_net40_pct": metrics[point_winner][
                "net_mean_pct"
            ]["40"],
            "preaudit_gate_passers": preaudit_passers,
            "forward_shadow_finalist": None,
            "production_action": "none",
            "status": "independent_pnl_audit_pending",
        },
        "metrics": metrics,
        "multiplicity": multiplicity,
        "retrospective_gates": gates,
        "folds": folds,
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
