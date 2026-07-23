#!/usr/bin/env python3
"""Run the frozen model-v10 Z17 peer-residual hypothesis once.

The protocol hash below binds this runner to the preregistration written before
any Z17 score was generated.  The runner is intentionally a single fixed
specification: there are no model, cluster, feature, or portfolio search knobs.
"""

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
import sklearn
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "model_v10_peer_residual_protocol.json"
DEFAULT_PANEL = Path("/tmp/model_v07_corrected_panel.pkl")
DEFAULT_MANIFEST = Path("/tmp/model_v07_corrected_panel.pkl.manifest.json")
DEFAULT_RESULT = Path("/tmp/model_v10_peer_residual_result.json")
DEFAULT_AUDIT = Path("/tmp/model_v10_peer_residual_audit.json")

PROTOCOL_ID = "model_v10_Z17_peer_residual_20260723"
PROTOCOL_SHA256 = "c45d854cf35a6af3920e6f763d726b385b33559d18e981174503609cdfa36dbb"
PANEL_SHA256 = "6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb"
MANIFEST_SHA256 = "25e08c564ef6400b7a29168db9fd7e8220bd0e2386c7a3c7e71b05c2ff950b02"

TRAIN_START = pd.Timestamp("2024-01-04")
SCORE_START = pd.Timestamp("2024-07-01")
SCORE_END = pd.Timestamp("2025-07-31")
EXPECTED_ROWS = 1_524_104
EXPECTED_CODES = 4_124
EXPECTED_SCORE_SESSIONS = 266
N_PEER_GROUPS = 8
MIN_PROFILE_ROWS = 20
RANDOM_STATE = 20_260_723
COSTS_BPS = (20, 40, 60)

PROFILE_FEATURES = (
    "oc_mean_20",
    "oc_std_20",
    "overnight_mean_20",
    "xrank_atr14_pct",
    "xrank_close_momentum_20",
    "no_trade_rate_20",
)
G0_FEATURES = (
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
REQUIRED_COLUMNS = {
    "date",
    "code",
    "name",
    "label",
    "oc_return_pct",
    "outcome_observed",
    "price_eligible",
    "price_training_eligible",
    "feature_source_max_date",
    *PROFILE_FEATURES,
    *G0_FEATURES,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    raise TypeError(f"cannot JSON-encode {type(value)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=json_scalar,
        )
        + "\n",
        encoding="utf-8",
    )


def monthly_folds() -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    folds: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for period in pd.period_range(
        SCORE_START.to_period("M"), SCORE_END.to_period("M"), freq="M"
    ):
        folds.append(
            (
                max(SCORE_START, period.start_time.normalize()),
                min(SCORE_END, period.end_time.normalize()),
                str(period),
            )
        )
    if len(folds) != 13:
        raise AssertionError(f"expected 13 monthly folds, got {len(folds)}")
    return folds


def validate_inputs(
    panel_path: Path,
    manifest_path: Path,
    protocol_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    hashes = {
        "panel_sha256": sha256_file(panel_path),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": sha256_file(protocol_path),
    }
    expected = {
        "panel_sha256": PANEL_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
    }
    if hashes != expected:
        raise ValueError(f"frozen input hash mismatch: got={hashes}, expected={expected}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("protocol id mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("panel_file_sha256") != PANEL_SHA256
        or manifest.get("rows") != EXPECTED_ROWS
        or manifest.get("codes") != EXPECTED_CODES
    ):
        raise ValueError("panel manifest content mismatch")

    panel = joblib.load(panel_path, mmap_mode="r")
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("frozen panel is not a pandas DataFrame")
    missing = sorted(REQUIRED_COLUMNS - set(panel.columns))
    if missing:
        raise ValueError(f"frozen panel lacks required columns: {missing}")
    if len(panel) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} panel rows, got {len(panel)}")
    if panel["code"].astype(str).nunique() != EXPECTED_CODES:
        raise ValueError("panel code count mismatch")
    if panel[["date", "code"]].isna().any().any():
        raise ValueError("panel has null date/code keys")
    if panel[["date", "code"]].duplicated().any():
        raise ValueError("panel has duplicate date/code keys")
    if not pd.api.types.is_datetime64_any_dtype(panel["date"]):
        raise TypeError("panel date column is not datetime64")
    return panel, manifest, hashes


def finite_numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.loc[:, list(columns)].astype(float)
    return result.replace([np.inf, -np.inf], np.nan)


def build_peer_assignments(
    panel: pd.DataFrame,
    fold_start: pd.Timestamp,
    model_train_mask: pd.Series,
    score_mask: pd.Series,
) -> tuple[pd.Series, dict[str, Any]]:
    profile_mask = (
        panel["date"].between(TRAIN_START, fold_start - pd.Timedelta(days=1))
        & panel["price_training_eligible"].eq(True)
    )
    if not profile_mask.equals(
        panel["date"].between(TRAIN_START, fold_start - pd.Timedelta(days=1))
        & panel["price_training_eligible"].eq(True)
    ):
        raise AssertionError("profile mask is not deterministic")
    history = panel.loc[profile_mask, ["date", "code", *PROFILE_FEATURES]].copy()
    history["code"] = history["code"].astype(str)
    numeric = finite_numeric(history, PROFILE_FEATURES)
    finite = numeric.notna()
    complete_mask = finite.all(axis=1)
    complete = pd.concat(
        [
            history.loc[complete_mask, ["code"]].reset_index(drop=True),
            numeric.loc[complete_mask].reset_index(drop=True),
        ],
        axis=1,
    )
    complete_counts = complete.groupby("code", sort=True).size()
    qualified_codes = complete_counts.loc[
        complete_counts.ge(MIN_PROFILE_ROWS)
    ].index
    qualified_profiles = (
        complete.loc[complete["code"].isin(qualified_codes)]
        .groupby("code", sort=True)[list(PROFILE_FEATURES)]
        .mean()
        .sort_index()
    )
    if len(qualified_profiles) < N_PEER_GROUPS:
        raise ValueError(
            f"fold {fold_start:%Y-%m}: only {len(qualified_profiles)} "
            f"qualified profiles for {N_PEER_GROUPS} groups"
        )
    if not np.isfinite(qualified_profiles.to_numpy(dtype=float)).all():
        raise ValueError("qualified profiles contain nonfinite values")
    if len(np.unique(qualified_profiles.to_numpy(dtype=float), axis=0)) < N_PEER_GROUPS:
        raise ValueError("fewer than eight distinct qualified profiles")

    scaler = StandardScaler()
    qualified_scaled = scaler.fit_transform(qualified_profiles)
    clusterer = KMeans(
        n_clusters=N_PEER_GROUPS,
        random_state=RANDOM_STATE,
        n_init=10,
        algorithm="lloyd",
    )
    qualified_labels = clusterer.fit_predict(qualified_scaled)
    occupied = sorted(np.unique(qualified_labels).astype(int).tolist())
    if occupied != list(range(N_PEER_GROUPS)):
        raise ValueError(f"fold {fold_start:%Y-%m} did not occupy all peer groups")

    train_codes = panel.loc[model_train_mask, "code"].astype(str)
    score_codes = panel.loc[score_mask, "code"].astype(str)
    scope_codes = pd.Index(
        np.unique(np.concatenate([train_codes.to_numpy(), score_codes.to_numpy()])),
        dtype=object,
    ).sort_values()
    partial_source = pd.concat(
        [history.loc[:, ["code"]].reset_index(drop=True), numeric.reset_index(drop=True)],
        axis=1,
    )
    partial_profiles = (
        partial_source.groupby("code", sort=True)[list(PROFILE_FEATURES)]
        .mean()
        .reindex(scope_codes)
    )
    global_profile = qualified_profiles.mean(axis=0)
    profile_for_assignment = partial_profiles.fillna(global_profile)
    profile_for_assignment.loc[qualified_profiles.index, :] = qualified_profiles
    if not np.isfinite(profile_for_assignment.to_numpy(dtype=float)).all():
        raise ValueError("profile fallback left nonfinite assignment values")
    assignment_labels = clusterer.predict(
        scaler.transform(profile_for_assignment.loc[:, list(PROFILE_FEATURES)])
    ).astype("int8")
    assignments = pd.Series(
        assignment_labels,
        index=scope_codes.astype(str),
        name="peer_group",
    )
    if assignments.isna().any() or not assignments.between(0, 7).all():
        raise AssertionError("peer assignment is incomplete")
    if not np.array_equal(
        assignments.loc[qualified_profiles.index].to_numpy(dtype=int),
        qualified_labels,
    ):
        raise AssertionError("qualified-code labels changed during assignment")

    group_counts = (
        assignments.value_counts(sort=False)
        .reindex(range(N_PEER_GROUPS), fill_value=0)
        .sort_index()
    )
    train_assignment_counts = (
        train_codes.map(assignments)
        .value_counts(sort=False)
        .reindex(range(N_PEER_GROUPS), fill_value=0)
        .sort_index()
    )
    score_assignment_counts = (
        score_codes.map(assignments)
        .value_counts(sort=False)
        .reindex(range(N_PEER_GROUPS), fill_value=0)
        .sort_index()
    )
    if (group_counts <= 0).any():
        raise AssertionError("an occupied KMeans group has no scoped codes")
    if train_codes.map(assignments).isna().any() or score_codes.map(assignments).isna().any():
        raise AssertionError("model train/score code lacks a peer assignment")
    no_finite_profile = int(partial_profiles.notna().sum(axis=1).eq(0).sum())
    incomplete_profile = int(
        partial_profiles.notna().sum(axis=1).between(1, len(PROFILE_FEATURES) - 1).sum()
    )
    unqualified = scope_codes.difference(qualified_profiles.index)

    audit = {
        "fold": str(fold_start.to_period("M")),
        "profile_start": str(history["date"].min().date()),
        "profile_end": str(history["date"].max().date()),
        "profile_rows": int(len(history)),
        "complete_profile_rows": int(complete_mask.sum()),
        "qualified_codes": int(len(qualified_profiles)),
        "unqualified_scoped_codes_nearest_fallback": int(len(unqualified)),
        "incomplete_partial_profile_codes": incomplete_profile,
        "all_global_profile_codes": no_finite_profile,
        "scoped_codes": int(len(scope_codes)),
        "occupied_peer_groups": occupied,
        "scoped_code_counts_by_group": {
            str(key): int(value) for key, value in group_counts.items()
        },
        "training_row_counts_by_group": {
            str(key): int(value) for key, value in train_assignment_counts.items()
        },
        "scoring_row_counts_by_group": {
            str(key): int(value) for key, value in score_assignment_counts.items()
        },
        "standard_scaler_mean": {
            feature: float(value)
            for feature, value in zip(PROFILE_FEATURES, scaler.mean_, strict=True)
        },
        "standard_scaler_scale": {
            feature: float(value)
            for feature, value in zip(PROFILE_FEATURES, scaler.scale_, strict=True)
        },
        "kmeans_inertia": float(clusterer.inertia_),
        "kmeans_iterations": int(clusterer.n_iter_),
        "profile_dates_strictly_before_score": bool(
            history["date"].max() < fold_start
        ),
    }
    if not audit["profile_dates_strictly_before_score"]:
        raise AssertionError("profile history reaches the score fold")
    del (
        history,
        numeric,
        finite,
        complete,
        partial_source,
        partial_profiles,
        profile_for_assignment,
        qualified_scaled,
        clusterer,
        scaler,
    )
    gc.collect()
    return assignments, audit


def build_fold_training_target(
    panel: pd.DataFrame,
    model_train_mask: pd.Series,
    assignments: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, dict[str, Any]]:
    columns = [
        "date",
        "code",
        "oc_return_pct",
        "feature_source_max_date",
        *G0_FEATURES,
    ]
    training = panel.loc[model_train_mask, columns].copy()
    training["code"] = training["code"].astype(str)
    training["peer_group"] = training["code"].map(assignments)
    if training["peer_group"].isna().any():
        raise AssertionError("training row lacks peer group")
    returns = pd.to_numeric(training["oc_return_pct"], errors="coerce")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("model training return is nonfinite")

    grouped = training.assign(_return=returns).groupby(
        ["date", "peer_group"], sort=False
    )["_return"]
    peer_sum = grouped.transform("sum")
    peer_count = grouped.transform("count")
    global_group = training.assign(_return=returns).groupby("date", sort=False)[
        "_return"
    ]
    global_sum = global_group.transform("sum")
    global_count = global_group.transform("count")
    peer_mean = (peer_sum - returns) / (peer_count - 1).where(peer_count.gt(1))
    fallback = peer_count.le(1)
    global_leave_one_out = (
        (global_sum - returns) / (global_count - 1).where(global_count.gt(1))
    )
    peer_mean = peer_mean.where(~fallback, global_leave_one_out)
    if not np.isfinite(peer_mean.to_numpy(dtype=float)).all():
        bad_dates = training.loc[~np.isfinite(peer_mean), "date"].drop_duplicates()
        raise ValueError(
            "global leave-one-out peer fallback unavailable on dates "
            + ", ".join(str(value.date()) for value in bad_dates.head(5))
        )
    raw_target = returns - peer_mean
    rank_target = (
        raw_target.groupby(training["date"], sort=False)
        .rank(method="average", pct=True)
        .mul(2.0)
        .sub(1.0)
    )
    if (
        not np.isfinite(rank_target.to_numpy(dtype=float)).all()
        or rank_target.lt(-1.0 - 1e-12).any()
        or rank_target.gt(1.0 + 1e-12).any()
    ):
        raise ValueError("within-date peer-residual rank target is invalid")

    counts = training.groupby("date", sort=False)["date"].transform("size")
    weights = 1.0 / counts.to_numpy(dtype=float)
    weight_sums = pd.Series(weights).groupby(
        training["date"].reset_index(drop=True), sort=False
    ).sum()
    max_weight_error = float(np.max(np.abs(weight_sums.to_numpy() - 1.0)))
    if max_weight_error > 1e-12:
        raise AssertionError(f"date-equal weights differ by {max_weight_error}")
    source_ok = training["feature_source_max_date"].notna() & (
        training["feature_source_max_date"] < training["date"]
    )
    if not source_ok.all():
        raise ValueError("training feature source is not strictly prior")
    audit = {
        "training_rows": int(len(training)),
        "training_dates": int(training["date"].nunique()),
        "training_start": str(training["date"].min().date()),
        "training_end": str(training["date"].max().date()),
        "peer_mean_group_rows": int((~fallback).sum()),
        "peer_mean_global_fallback_rows": int(fallback.sum()),
        "peer_mean_global_fallback_rate": float(fallback.mean()),
        "raw_target_mean": float(raw_target.mean()),
        "raw_target_std": float(raw_target.std(ddof=1)),
        "rank_target_min": float(rank_target.min()),
        "rank_target_max": float(rank_target.max()),
        "date_equal_weight_max_abs_error": max_weight_error,
        "feature_source_strictly_prior": True,
        "maximum_feature_source_date": str(
            training["feature_source_max_date"].max().date()
        ),
    }
    return training, rank_target, weights, audit


def fit_rank_ridge(
    training: pd.DataFrame,
    target: pd.Series,
    weights: np.ndarray,
) -> Pipeline:
    estimator = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    estimator.fit(
        training.loc[:, list(G0_FEATURES)],
        target.to_numpy(dtype=float),
        model__sample_weight=weights,
    )
    return estimator


def ranked_top_two(scoring: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    if not np.isfinite(scores).all():
        raise ValueError("model emitted nonfinite scores")
    ranked = scoring.loc[
        :, ["date", "code", "name", "oc_return_pct", "peer_group"]
    ].copy()
    ranked["model_score"] = scores
    ranked["code"] = ranked["code"].astype(str)
    ranked = ranked.sort_values(
        ["date", "model_score", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranked = ranked.groupby("date", sort=True).head(2).copy()
    ranked["model_rank"] = ranked.groupby("date", sort=False).cumcount() + 1
    return ranked.sort_values(["date", "model_rank"], kind="stable").reset_index(
        drop=True
    )


def score_fold(
    panel: pd.DataFrame,
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
    fold_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_train_mask = (
        panel["date"].between(TRAIN_START, fold_start - pd.Timedelta(days=1))
        & panel["price_training_eligible"].eq(True)
        & panel["outcome_observed"].eq(True)
        & panel["oc_return_pct"].notna()
        & np.isfinite(pd.to_numeric(panel["oc_return_pct"], errors="coerce"))
    )
    score_mask = (
        panel["date"].between(fold_start, fold_end)
        & panel["price_eligible"].eq(True)
    )
    if not model_train_mask.any() or not score_mask.any():
        raise ValueError(f"fold {fold_id} has empty train or score rows")
    train_max = panel.loc[model_train_mask, "date"].max()
    score_min = panel.loc[score_mask, "date"].min()
    if not train_max < score_min:
        raise AssertionError(f"fold {fold_id} training overlaps scoring")

    assignments, peer_audit = build_peer_assignments(
        panel, fold_start, model_train_mask, score_mask
    )
    training, target, weights, target_audit = build_fold_training_target(
        panel, model_train_mask, assignments
    )
    estimator = fit_rank_ridge(training, target, weights)

    score_columns = [
        "date",
        "code",
        "name",
        "label",
        "oc_return_pct",
        "feature_source_max_date",
        *G0_FEATURES,
    ]
    scoring = panel.loc[score_mask, score_columns].copy()
    scoring["code"] = scoring["code"].astype(str)
    scoring["peer_group"] = scoring["code"].map(assignments)
    if scoring["peer_group"].isna().any():
        raise AssertionError("score row lacks peer group")
    source_ok = scoring["feature_source_max_date"].notna() & (
        scoring["feature_source_max_date"] < scoring["date"]
    )
    if not source_ok.all():
        raise ValueError(f"fold {fold_id} score feature source is not strictly prior")
    scores = estimator.predict(scoring.loc[:, list(G0_FEATURES)])
    picks = ranked_top_two(scoring, scores)

    # Mutate every scoring outcome and label while keeping the already-present
    # point-in-time feature matrix unchanged.  Re-predict and re-rank exactly.
    mutated = scoring.copy()
    mutation_values = np.linspace(
        1_000_000.0,
        -1_000_000.0,
        num=len(mutated),
        dtype=float,
    )
    mutated["oc_return_pct"] = mutation_values
    mutated["label"] = np.mod(np.arange(len(mutated), dtype=int), 2).astype(float)
    mutated_scores = estimator.predict(mutated.loc[:, list(G0_FEATURES)])
    mutated_picks = ranked_top_two(mutated, mutated_scores)
    score_exact = bool(np.array_equal(scores, mutated_scores))
    pick_exact = bool(
        picks.loc[:, ["date", "model_rank", "code"]].equals(
            mutated_picks.loc[:, ["date", "model_rank", "code"]]
        )
    )
    if not score_exact or not pick_exact:
        raise ValueError(f"fold {fold_id} target-day outcome mutation changed ranking")

    fold_sessions = pd.DatetimeIndex(
        panel.loc[
            panel["date"].between(fold_start, fold_end), "date"
        ].drop_duplicates().sort_values()
    )
    pick_sessions = pd.DatetimeIndex(picks["date"].drop_duplicates().sort_values())
    if not fold_sessions.equals(pick_sessions):
        raise ValueError(f"fold {fold_id} lacks a scored panel session")
    rank_counts = picks.groupby("date", sort=False)["model_rank"].nunique()
    if not rank_counts.eq(2).all():
        raise ValueError(f"fold {fold_id} cannot fill two displayed ranks")
    audit = {
        "fold": fold_id,
        "train_start": str(training["date"].min().date()),
        "train_end": str(training["date"].max().date()),
        "score_start": str(scoring["date"].min().date()),
        "score_end": str(scoring["date"].max().date()),
        "score_rows": int(len(scoring)),
        "score_sessions": int(len(fold_sessions)),
        "training_strictly_before_scoring": bool(train_max < score_min),
        "score_feature_source_strictly_prior": True,
        "maximum_score_feature_source_date": str(
            scoring["feature_source_max_date"].max().date()
        ),
        "peer_model": peer_audit,
        "training_target": target_audit,
        "target_day_outcome_mutation": {
            "mutated_rows": int(len(mutated)),
            "model_scores_bitwise_exact": score_exact,
            "top_two_code_rank_exact": pick_exact,
            "maximum_absolute_score_difference": float(
                np.max(np.abs(scores - mutated_scores))
            ),
        },
    }
    del (
        assignments,
        training,
        target,
        weights,
        estimator,
        scoring,
        mutated,
        mutated_scores,
        mutated_picks,
        scores,
    )
    gc.collect()
    return picks, audit


def primary_daily_returns(
    picks: pd.DataFrame,
    score_sessions: pd.DatetimeIndex,
    slots: int,
    cost_bps: int,
    cash_codes: set[str] | None = None,
) -> pd.Series:
    selected = picks.loc[picks["model_rank"].le(slots)].copy()
    observed = selected["oc_return_pct"].notna() & np.isfinite(
        pd.to_numeric(selected["oc_return_pct"], errors="coerce")
    )
    if cash_codes:
        observed &= ~selected["code"].astype(str).isin(cash_codes)
    contribution = (
        selected["oc_return_pct"].where(observed, 0.0)
        - observed.astype(float) * (float(cost_bps) / 100.0)
    ) / float(slots)
    daily = contribution.groupby(selected["date"], sort=True).sum()
    return daily.reindex(score_sessions, fill_value=0.0).astype(float)


def independent_daily_returns(
    picks: pd.DataFrame,
    score_sessions: pd.DatetimeIndex,
    slots: int,
    cost_bps: int,
    cash_codes: set[str] | None = None,
) -> pd.Series:
    lookup: dict[tuple[pd.Timestamp, int], tuple[str, float | None]] = {}
    for row in picks.itertuples(index=False):
        value = float(row.oc_return_pct) if pd.notna(row.oc_return_pct) else None
        lookup[(pd.Timestamp(row.date), int(row.model_rank))] = (str(row.code), value)
    values: list[float] = []
    for date in score_sessions:
        total = 0.0
        for model_rank in range(1, slots + 1):
            item = lookup.get((pd.Timestamp(date), model_rank))
            if item is None:
                continue
            code, outcome = item
            if cash_codes and code in cash_codes:
                continue
            if outcome is None or not np.isfinite(outcome):
                continue
            total += (outcome - float(cost_bps) / 100.0) / float(slots)
        values.append(total)
    return pd.Series(values, index=score_sessions, dtype=float)


def top_profit_codes(picks: pd.DataFrame, slots: int) -> tuple[list[str], pd.Series]:
    selected = picks.loc[picks["model_rank"].le(slots)].copy()
    observed = selected["oc_return_pct"].notna() & np.isfinite(
        pd.to_numeric(selected["oc_return_pct"], errors="coerce")
    )
    selected["_net20_contribution"] = (
        selected["oc_return_pct"].where(observed, 0.0)
        - observed.astype(float) * 0.20
    ) / float(slots)
    code_profit = (
        selected.groupby(selected["code"].astype(str), sort=True)[
            "_net20_contribution"
        ]
        .sum()
        .sort_values(ascending=False, kind="stable")
    )
    return code_profit.head(10).index.astype(str).tolist(), code_profit


def portfolio_summary(
    picks: pd.DataFrame,
    score_sessions: pd.DatetimeIndex,
    slots: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    daily: dict[int, pd.Series] = {}
    recomputed: dict[int, pd.Series] = {}
    max_differences: dict[str, float] = {}
    for cost in COSTS_BPS:
        daily[cost] = primary_daily_returns(picks, score_sessions, slots, cost)
        recomputed[cost] = independent_daily_returns(
            picks, score_sessions, slots, cost
        )
        difference = float(
            np.max(
                np.abs(
                    daily[cost].to_numpy(dtype=float)
                    - recomputed[cost].to_numpy(dtype=float)
                )
            )
        )
        max_differences[str(cost)] = difference
        if difference > 1e-12:
            raise ValueError(
                f"top{slots} cost{cost} independent P&L differs by {difference}"
            )

    monthly: dict[str, dict[str, float]] = {}
    for period in pd.period_range(
        SCORE_START.to_period("M"), SCORE_END.to_period("M"), freq="M"
    ):
        mask = score_sessions.to_period("M") == period
        monthly[str(period)] = {
            str(cost): float(daily[cost].loc[mask].mean())
            for cost in COSTS_BPS
        }
    if len(monthly) != 13:
        raise AssertionError("monthly summary does not contain 13 months")
    slices: dict[str, dict[str, float]] = {}
    for name, (start, end) in PERIODS.items():
        mask = score_sessions.to_series(index=score_sessions).between(start, end)
        slices[name] = {
            str(cost): float(daily[cost].loc[mask.to_numpy()].mean())
            for cost in COSTS_BPS
        }

    best20_order = (
        pd.DataFrame({"date": score_sessions, "value": daily[20].to_numpy()})
        .sort_values(["value", "date"], ascending=[False, True], kind="stable")
        .head(20)
    )
    best20_dates = pd.DatetimeIndex(best20_order["date"])
    remaining = daily[20].drop(best20_dates)
    if len(remaining) != len(score_sessions) - 20:
        raise AssertionError("best-20 removal did not remove exactly 20 dates")

    profit_codes, code_profit = top_profit_codes(picks, slots)
    cash_set = set(profit_codes)
    cash_daily = primary_daily_returns(
        picks, score_sessions, slots, 20, cash_codes=cash_set
    )
    cash_recomputed = independent_daily_returns(
        picks, score_sessions, slots, 20, cash_codes=cash_set
    )
    cash_difference = float(
        np.max(np.abs(cash_daily.to_numpy() - cash_recomputed.to_numpy()))
    )
    if cash_difference > 1e-12:
        raise ValueError(f"top-profit-code cash P&L differs by {cash_difference}")

    selected = picks.loc[picks["model_rank"].le(slots)].copy()
    selection_counts = selected["code"].astype(str).value_counts()
    top10_selection_share = float(
        selection_counts.head(10).sum() / (len(score_sessions) * slots)
    )
    summary = {
        "slots": slots,
        "score_sessions": int(len(score_sessions)),
        "displayed_slots": int(len(selected)),
        "observed_executed_slots": int(
            np.isfinite(
                pd.to_numeric(selected["oc_return_pct"], errors="coerce")
            ).sum()
        ),
        "mean_daily_net_pct": {
            str(cost): float(daily[cost].mean()) for cost in COSTS_BPS
        },
        "monthly_mean_daily_net_pct": monthly,
        "positive_months": {
            str(cost): int(
                sum(month_values[str(cost)] > 0 for month_values in monthly.values())
            )
            for cost in COSTS_BPS
        },
        "temporal_slices_mean_daily_net_pct": slices,
        "best_20_days_removed_net20_pct": float(remaining.mean()),
        "best_20_days_net20": [
            {
                "date": str(row.date.date()),
                "net20_pct": float(row.value),
            }
            for row in best20_order.itertuples(index=False)
        ],
        "top_10_profit_codes": profit_codes,
        "top_10_profit_code_net20_contribution_pct": {
            str(code): float(code_profit.loc[code]) for code in profit_codes
        },
        "top_10_profit_codes_to_cash_net20_pct": float(cash_daily.mean()),
        "unique_selected_codes": int(selected["code"].astype(str).nunique()),
        "top_10_selection_share": top10_selection_share,
    }
    audit = {
        "slots": slots,
        "primary_vs_independent_max_abs_daily_difference_pct": max_differences,
        "top_10_profit_codes_cash_max_abs_daily_difference_pct": cash_difference,
        "all_cost_paths_within_1e_12": bool(
            max(max_differences.values()) <= 1e-12 and cash_difference <= 1e-12
        ),
    }
    return summary, audit


def selection_sha256(picks: pd.DataFrame) -> str:
    canonical = picks.loc[
        :,
        [
            "date",
            "model_rank",
            "code",
            "peer_group",
            "model_score",
            "oc_return_pct",
        ],
    ].copy()
    canonical["date"] = canonical["date"].dt.strftime("%Y-%m-%d")
    payload = canonical.to_csv(index=False, float_format="%.17g").encode("utf-8")
    return sha256_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()

    panel, manifest, input_hashes = validate_inputs(
        args.panel, args.manifest, args.protocol
    )
    score_sessions = pd.DatetimeIndex(
        panel.loc[
            panel["date"].between(SCORE_START, SCORE_END), "date"
        ].drop_duplicates().sort_values()
    )
    if len(score_sessions) != EXPECTED_SCORE_SESSIONS:
        raise ValueError(
            f"expected {EXPECTED_SCORE_SESSIONS} score sessions, "
            f"got {len(score_sessions)}"
        )

    fold_picks: list[pd.DataFrame] = []
    fold_audits: list[dict[str, Any]] = []
    for fold_start, fold_end, fold_id in monthly_folds():
        picks, fold_audit = score_fold(panel, fold_start, fold_end, fold_id)
        fold_picks.append(picks)
        fold_audits.append(fold_audit)
        print(
            f"completed {fold_id}: "
            f"train={fold_audit['training_target']['training_rows']}, "
            f"score={fold_audit['score_rows']}, "
            f"qualified_profiles={fold_audit['peer_model']['qualified_codes']}",
            flush=True,
        )

    picks = pd.concat(fold_picks, ignore_index=True)
    picks = picks.sort_values(["date", "model_rank"], kind="stable").reset_index(
        drop=True
    )
    if picks.duplicated(["date", "model_rank"]).any():
        raise AssertionError("duplicate date/rank selection")
    if picks["date"].nunique() != EXPECTED_SCORE_SESSIONS:
        raise AssertionError("combined picks do not cover 266 sessions")
    expected_picks = EXPECTED_SCORE_SESSIONS * 2
    if len(picks) != expected_picks:
        raise AssertionError(f"expected {expected_picks} top-two rows, got {len(picks)}")

    portfolios: dict[str, Any] = {}
    pnl_audits: dict[str, Any] = {}
    for slots in (1, 2):
        portfolios[f"top{slots}"], pnl_audits[f"top{slots}"] = portfolio_summary(
            picks, score_sessions, slots
        )

    mutation_all_exact = all(
        item["target_day_outcome_mutation"]["model_scores_bitwise_exact"]
        and item["target_day_outcome_mutation"]["top_two_code_rank_exact"]
        and item["target_day_outcome_mutation"][
            "maximum_absolute_score_difference"
        ]
        == 0.0
        for item in fold_audits
    )
    pit_all_passed = all(
        item["training_strictly_before_scoring"]
        and item["score_feature_source_strictly_prior"]
        and item["peer_model"]["profile_dates_strictly_before_score"]
        and item["training_target"]["feature_source_strictly_prior"]
        and item["training_target"]["date_equal_weight_max_abs_error"] <= 1e-12
        and item["peer_model"]["occupied_peer_groups"] == list(range(8))
        for item in fold_audits
    )
    pnl_all_exact = all(
        item["all_cost_paths_within_1e_12"] for item in pnl_audits.values()
    )
    if not mutation_all_exact or not pit_all_passed or not pnl_all_exact:
        raise ValueError("a required integrity audit failed")

    runner_hash = sha256_file(Path(__file__))
    audit = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": runner_hash,
        **input_hashes,
        "environment": {
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
        },
        "input_validation": {
            "panel_rows": int(len(panel)),
            "panel_codes": int(panel["code"].astype(str).nunique()),
            "manifest_schema_version": manifest.get("schema_version"),
            "score_sessions": int(len(score_sessions)),
            "monthly_folds": len(fold_audits),
        },
        "folds": fold_audits,
        "integrity_summary": {
            "target_day_outcome_mutation_all_13_folds_exact": mutation_all_exact,
            "point_in_time_all_13_folds_passed": pit_all_passed,
            "independent_pnl_all_paths_within_1e_12": pnl_all_exact,
            "production_model_changed": false_value(),
        },
        "independent_pnl": pnl_audits,
    }
    write_json(args.audit_output, audit)
    audit_hash = sha256_file(args.audit_output)

    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "hypothesis_id": "Z17_peer_residual_model",
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": runner_hash,
        "panel_sha256": input_hashes["panel_sha256"],
        "manifest_sha256": input_hashes["manifest_sha256"],
        "audit_sha256": audit_hash,
        "score_period": [str(SCORE_START.date()), str(SCORE_END.date())],
        "score_sessions": int(len(score_sessions)),
        "monthly_folds": len(fold_audits),
        "selection_rows_top2": int(len(picks)),
        "selection_sha256": selection_sha256(picks),
        "model_specification": {
            "peer_groups": 8,
            "qualified_profile_minimum_complete_rows": 20,
            "profile_features": list(PROFILE_FEATURES),
            "kmeans_random_state": RANDOM_STATE,
            "kmeans_n_init": 10,
            "ridge_alpha": 1.0,
            "g0_features": list(G0_FEATURES),
            "date_equal_weights": True,
        },
        "portfolios": portfolios,
        "integrity": audit["integrity_summary"],
        "production_model_changed": False,
        "production_promotion": "none",
        "interpretation": (
            "Retrospective preregistered mechanism-test output only; "
            "not an untouched holdout and not eligible for production promotion."
        ),
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                key: value["mean_daily_net_pct"]
                for key, value in portfolios.items()
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def false_value() -> bool:
    """Return a literal false for an explicit no-production audit field."""

    return False


if __name__ == "__main__":
    main()
