#!/usr/bin/env python3
"""Reproducible profit-first comparison on the production feature panel.

The script deliberately keeps the candidate set, monthly folds, transaction
cost convention, and daily ranking logic identical for every model.  Model
hyperparameters are fixed below; the exploration period is reported but is not
used for an additional grid search.  The selection-period winner alone is run
on the already-viewed benchmark period.

Example
-------
PYTHONPATH=src python research/compare_profit_models.py \
    --daily /tmp/package_collected.pkl \
    --output research/profit_model_comparison.json
"""

from __future__ import annotations

import argparse
import hashlib
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tse_session_ranker.config import RankerConfig
from tse_session_ranker.data.common import (
    normalize_daily_prices,
    normalize_expected_sessions,
    prepare_modeling_prices,
    session_calendar_hash,
)
from tse_session_ranker.features import PRICE_FEATURE_COLUMNS, build_feature_panel
from tse_session_ranker.inference import rank_candidates
from tse_session_ranker.io import read_frame, write_frame, write_json
from tse_session_ranker.profit import profit_metrics
from tse_session_ranker.training import date_equal_weights, fit_estimator


DEFAULT_SEED = 31
FEATURE_COLUMNS = PRICE_FEATURE_COLUMNS
EXPLORATION_START = pd.Timestamp("2024-09-02")
EXPLORATION_END = pd.Timestamp("2024-10-31")
EXPLORATION_TRAIN_START = pd.Timestamp("2024-04-01")
SELECTION_START = pd.Timestamp("2025-01-06")
SELECTION_END = pd.Timestamp("2025-03-31")
BENCHMARK_START = pd.Timestamp("2025-04-01")
BENCHMARK_END = pd.Timestamp("2025-07-31")


class FittedScorer(Protocol):
    def score(self, frame: pd.DataFrame) -> np.ndarray: ...


class ModelFamily(Protocol):
    name: str

    def fit(self, frame: pd.DataFrame) -> FittedScorer: ...

    def specification(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PeriodDefinition:
    start: str
    end: str
    train_start: str


class PipelineScorer:
    def __init__(self, estimator: Any, score_method: str) -> None:
        self.estimator = estimator
        self.score_method = score_method

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        features = frame[list(FEATURE_COLUMNS)]
        if self.score_method == "positive_probability":
            return self.estimator.predict_proba(features)[:, 1]
        if self.score_method == "prediction":
            return np.asarray(self.estimator.predict(features), dtype=float)
        raise RuntimeError(f"unknown score method: {self.score_method}")


class CurrentLogitFamily:
    name = "current_logit"

    def __init__(self, config: RankerConfig) -> None:
        self.config = config

    def fit(self, frame: pd.DataFrame) -> FittedScorer:
        # Call the production fitter so this candidate is an exact control.
        return PipelineScorer(fit_estimator(frame, self.config), "positive_probability")

    def specification(self) -> dict[str, Any]:
        return {
            "family": "binary_logistic_regression",
            "target": "close_gt_open",
            "features": list(FEATURE_COLUMNS),
            "imputer": "median_with_missing_indicators",
            "scaler": "standard",
            "C": self.config.model.c,
            "class_weight": self.config.model.class_weight,
            "date_weight": "1 / executed_training_rows_on_date",
            "score": "P(close > open); used only as a same-day rank score",
            "random_state": self.config.model.random_state,
        }


def _regression_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


class RidgeReturnFamily:
    name = "direct_ridge_return"

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha

    def fit(self, frame: pd.DataFrame) -> FittedScorer:
        estimator = _regression_pipeline(Ridge(alpha=self.alpha))
        estimator.fit(
            frame[list(FEATURE_COLUMNS)],
            frame["oc_return_pct"],
            model__sample_weight=date_equal_weights(frame),
        )
        return PipelineScorer(estimator, "prediction")

    def specification(self) -> dict[str, Any]:
        return {
            "family": "ridge_regression",
            "target": "raw open_to_close return percent",
            "features": list(FEATURE_COLUMNS),
            "imputer": "median_with_missing_indicators",
            "scaler": "standard",
            "alpha": self.alpha,
            "date_weight": "1 / executed_training_rows_on_date",
            "score": "predicted raw open_to_close return percent",
        }


class RobustReturnFamily:
    name = "robust_huber_sgd_return"

    def __init__(self, seed: int, alpha: float = 0.0001, epsilon: float = 1.0) -> None:
        self.seed = seed
        self.alpha = alpha
        self.epsilon = epsilon

    def fit(self, frame: pd.DataFrame) -> FittedScorer:
        estimator = _regression_pipeline(
            SGDRegressor(
                loss="huber",
                penalty="l2",
                alpha=self.alpha,
                epsilon=self.epsilon,
                max_iter=2_000,
                tol=1e-4,
                shuffle=True,
                random_state=self.seed,
                average=True,
            )
        )
        estimator.fit(
            frame[list(FEATURE_COLUMNS)],
            frame["oc_return_pct"],
            model__sample_weight=date_equal_weights(frame),
        )
        return PipelineScorer(estimator, "prediction")

    def specification(self) -> dict[str, Any]:
        return {
            "family": "SGDRegressor(loss='huber')",
            "target": "raw open_to_close return percent",
            "features": list(FEATURE_COLUMNS),
            "imputer": "median_with_missing_indicators",
            "scaler": "standard",
            "alpha": self.alpha,
            "epsilon_return_pct": self.epsilon,
            "max_iter": 2_000,
            "tol": 1e-4,
            "average": True,
            "date_weight": "1 / executed_training_rows_on_date",
            "score": "robust predicted open_to_close return percent",
            "random_state": self.seed,
        }


class ReturnWeightedLogitFamily:
    name = "return_weighted_logit"

    def __init__(
        self,
        seed: int,
        c: float = 0.08,
        magnitude_floor_pct: float = 0.25,
        magnitude_cap_pct: float = 3.0,
    ) -> None:
        self.seed = seed
        self.c = c
        self.magnitude_floor_pct = magnitude_floor_pct
        self.magnitude_cap_pct = magnitude_cap_pct

    def fit(self, frame: pd.DataFrame) -> FittedScorer:
        magnitude = frame["oc_return_pct"].abs().clip(
            lower=self.magnitude_floor_pct,
            upper=self.magnitude_cap_pct,
        )
        # Normalising within each date preserves equal total date influence.
        weights = magnitude / magnitude.groupby(frame["date"], sort=False).transform("sum")
        estimator = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=self.c,
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        estimator.fit(
            frame[list(FEATURE_COLUMNS)],
            frame["label"].astype(int),
            model__sample_weight=weights,
        )
        return PipelineScorer(estimator, "positive_probability")

    def specification(self) -> dict[str, Any]:
        return {
            "family": "return_magnitude_weighted_logistic_regression",
            "target": "close_gt_open",
            "features": list(FEATURE_COLUMNS),
            "C": self.c,
            "class_weight": "balanced",
            "magnitude_floor_pct": self.magnitude_floor_pct,
            "magnitude_cap_pct": self.magnitude_cap_pct,
            "date_weight": (
                "clip(abs(open_to_close_return_pct), floor, cap), then normalise "
                "weights to sum to one within each training date"
            ),
            "score": "weighted P(close > open); same-day rank only",
            "random_state": self.seed,
        }


class PairwiseScorer:
    def __init__(self, preprocessor: Pipeline, estimator: LogisticRegression) -> None:
        self.preprocessor = preprocessor
        self.estimator = estimator

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(frame[list(FEATURE_COLUMNS)])
        return np.asarray(self.estimator.decision_function(transformed), dtype=float)


class PairwiseRankingFamily:
    name = "pairwise_linear_rank"

    def __init__(self, seed: int, c: float = 0.08, pairs_per_date: int = 96) -> None:
        self.seed = seed
        self.c = c
        self.pairs_per_date = pairs_per_date

    def fit(self, frame: pd.DataFrame) -> FittedScorer:
        preprocessor = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]
        )
        transformed = np.asarray(
            preprocessor.fit_transform(frame[list(FEATURE_COLUMNS)]), dtype=float
        )
        returns = frame["oc_return_pct"].to_numpy(dtype=float)
        differences: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        # RNG is reset once per fit; stable row sorting makes every fold reproducible.
        rng = np.random.default_rng(self.seed)
        for _, positions in frame.groupby("date", sort=True).indices.items():
            positions = np.asarray(positions, dtype=int)
            if len(positions) < 2:
                continue
            pair_count = min(self.pairs_per_date, len(positions) * (len(positions) - 1) // 2)
            left = rng.choice(positions, size=pair_count, replace=True)
            right = rng.choice(positions, size=pair_count, replace=True)
            equal_position = left == right
            while equal_position.any():
                right[equal_position] = rng.choice(
                    positions, size=int(equal_position.sum()), replace=True
                )
                equal_position = left == right
            unequal_return = returns[left] != returns[right]
            left = left[unequal_return]
            right = right[unequal_return]
            if not len(left):
                continue
            high = np.where(returns[left] > returns[right], left, right)
            low = np.where(returns[left] > returns[right], right, left)
            positive = transformed[high] - transformed[low]
            # Add each comparison in both orientations for exact class balance.
            differences.extend([positive, -positive])
            labels.extend(
                [np.ones(len(positive), dtype=int), np.zeros(len(positive), dtype=int)]
            )
            per_pair_weight = 0.5 / len(positive)
            weights.extend(
                [
                    np.full(len(positive), per_pair_weight),
                    np.full(len(positive), per_pair_weight),
                ]
            )
        if not differences:
            raise ValueError("pairwise model could not construct any unequal-return pairs")
        pair_features = np.vstack(differences)
        pair_labels = np.concatenate(labels)
        pair_weights = np.concatenate(weights)
        estimator = LogisticRegression(
            C=self.c,
            class_weight=None,
            fit_intercept=False,
            max_iter=1_000,
            random_state=self.seed,
        )
        estimator.fit(pair_features, pair_labels, sample_weight=pair_weights)
        return PairwiseScorer(preprocessor, estimator)

    def specification(self) -> dict[str, Any]:
        return {
            "family": "within_date_pairwise_linear_logistic_ranker",
            "pair_target": "higher realised open_to_close return within training date",
            "features": list(FEATURE_COLUMNS),
            "imputer": "median_with_missing_indicators fitted on original training rows",
            "scaler": "standard fitted on original training rows",
            "C": self.c,
            "pairs_per_date": self.pairs_per_date,
            "pair_sampling": "seeded with replacement; equal-return pairs discarded",
            "orientation": "each high-minus-low pair plus its negative",
            "date_weight": "all oriented pair weights sum to one per training date",
            "score": "linear decision value w dot transformed_features",
            "random_state": self.seed,
        }


class MultiThresholdScorer:
    def __init__(
        self,
        models: list[Pipeline],
        widths: tuple[float, ...],
        lower_bound: float,
    ) -> None:
        self.models = models
        self.widths = widths
        self.lower_bound = lower_bound

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        features = frame[list(FEATURE_COLUMNS)]
        score = np.full(len(frame), self.lower_bound, dtype=float)
        for width, model in zip(self.widths, self.models, strict=True):
            score += width * model.predict_proba(features)[:, 1]
        return score


class MultiThresholdExpectedReturnFamily:
    name = "multi_threshold_expected_return"

    # Midpoint integration of E[clip(R,-3,3)] = -3 + integral P(R > t) dt.
    thresholds = (-2.0, -0.5, 0.5, 2.0)
    widths = (2.0, 1.0, 1.0, 2.0)

    def __init__(self, seed: int, c: float = 0.08) -> None:
        self.seed = seed
        self.c = c

    def fit(self, frame: pd.DataFrame) -> FittedScorer:
        weights = date_equal_weights(frame)
        models: list[Pipeline] = []
        for threshold in self.thresholds:
            target = frame["oc_return_pct"].gt(threshold).astype(int)
            if target.nunique() != 2:
                raise ValueError(
                    f"multi-threshold target at {threshold} has only one class"
                )
            estimator = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=self.c,
                            # Balanced class weights would turn predict_proba
                            # into a reweighted-class score rather than the
                            # tail probability required by the integral below.
                            class_weight=None,
                            max_iter=1_000,
                            random_state=self.seed,
                        ),
                    ),
                ]
            )
            estimator.fit(
                frame[list(FEATURE_COLUMNS)],
                target,
                model__sample_weight=weights,
            )
            models.append(estimator)
        return MultiThresholdScorer(models, self.widths, lower_bound=-3.0)

    def specification(self) -> dict[str, Any]:
        return {
            "family": "multi_threshold_logistic_expected_return",
            "target": "four cumulative indicators open_to_close_return > threshold",
            "features": list(FEATURE_COLUMNS),
            "threshold_midpoints_pct": list(self.thresholds),
            "integration_widths_pct": list(self.widths),
            "clip_range_pct": [-3.0, 3.0],
            "C": self.c,
            "class_weight": None,
            "date_weight": "1 / executed_training_rows_on_date",
            "score": (
                "-3 + 2*P(R>-2) + 1*P(R>-0.5) + "
                "1*P(R>0.5) + 2*P(R>2)"
            ),
            "random_state": self.seed,
        }


def _families(config: RankerConfig, seed: int) -> dict[str, ModelFamily]:
    families: list[ModelFamily] = [
        CurrentLogitFamily(config),
        RidgeReturnFamily(alpha=10.0),
        RobustReturnFamily(seed=seed, alpha=0.0001, epsilon=1.0),
        PairwiseRankingFamily(seed=seed, c=0.08, pairs_per_date=96),
        ReturnWeightedLogitFamily(
            seed=seed,
            c=0.08,
            magnitude_floor_pct=0.25,
            magnitude_cap_pct=3.0,
        ),
        MultiThresholdExpectedReturnFamily(seed=seed, c=0.08),
    ]
    return {family.name: family for family in families}


def _daily_picks(scored: pd.DataFrame, top_k: int) -> pd.DataFrame:
    pieces = [
        rank_candidates(group, top_k)
        for _, group in scored.groupby("date", sort=True)
    ]
    return pd.concat(pieces, ignore_index=True) if pieces else scored.iloc[:0].copy()


def _evaluate_models(
    panel: pd.DataFrame,
    families: dict[str, ModelFamily],
    model_names: list[str],
    *,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    train_start: pd.Timestamp,
    cost_bps: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    scored_by_model: dict[str, list[pd.DataFrame]] = {
        name: [] for name in model_names
    }
    fold_records: list[dict[str, Any]] = []
    for period in pd.period_range(
        evaluation_start.to_period("M"), evaluation_end.to_period("M"), freq="M"
    ):
        score_start = max(evaluation_start, period.start_time.normalize())
        score_end = min(evaluation_end, period.end_time.normalize())
        training = panel[
            panel["date"].between(train_start, score_start - pd.Timedelta(days=1))
            & panel["training_eligible"]
            & panel["label"].notna()
        ].copy()
        # Do not require a target label or target OHLC here.  Unfilled rows must
        # be ranked before execution is known and remain zero-return slots.
        scoring = panel[
            panel["date"].between(score_start, score_end) & panel["eligible"]
        ].copy()
        if training.empty or scoring.empty:
            raise ValueError(
                f"empty train/score fold for {period}: "
                f"train={len(training)}, score={len(scoring)}"
            )
        if not training["date"].max() < scoring["date"].min():
            raise AssertionError(f"overlapping training/scoring dates for {period}")
        if (scoring["feature_source_max_date"] >= scoring["date"]).fillna(False).any():
            raise AssertionError(f"non-prior feature source detected for {period}")
        training = training.sort_values(["date", "code"], kind="stable")
        scoring = scoring.sort_values(["date", "code"], kind="stable")
        for model_name in model_names:
            fitted = families[model_name].fit(training)
            model_scored = scoring.copy()
            model_scored["model_score"] = fitted.score(scoring)
            model_scored["model_name"] = model_name
            scored_by_model[model_name].append(model_scored)
            fold_records.append(
                {
                    "period": str(period),
                    "model_name": model_name,
                    "train_start": str(training["date"].min().date()),
                    "train_end": str(training["date"].max().date()),
                    "score_start": str(scoring["date"].min().date()),
                    "score_end": str(scoring["date"].max().date()),
                    "train_rows": int(len(training)),
                    "score_rows": int(len(scoring)),
                    "score_days": int(scoring["date"].nunique()),
                }
            )

    results: dict[str, Any] = {}
    pick_frames: list[pd.DataFrame] = []
    for model_name in model_names:
        scored = pd.concat(scored_by_model[model_name], ignore_index=True)
        top1 = _daily_picks(scored, 1)
        top2 = _daily_picks(scored, 2)
        results[model_name] = {
            "top1": profit_metrics(top1, top_k=1, cost_bps=cost_bps),
            "top2": profit_metrics(top2, top_k=2, cost_bps=cost_bps),
            "folds": [
                row for row in fold_records if row["model_name"] == model_name
            ],
        }
        for top_k, picks in ((1, top1), (2, top2)):
            keep = [
                column
                for column in (
                    "date",
                    "code",
                    "name",
                    "model_rank",
                    "model_score",
                    "label",
                    "oc_return_pct",
                    "outcome_observed",
                    "source_complete",
                    "universe_source_complete",
                    "universe_source_date",
                )
                if column in picks
            ]
            output = picks[keep].copy()
            output.insert(0, "portfolio_top_k", top_k)
            output.insert(0, "model_name", model_name)
            pick_frames.append(output)
    return results, pd.concat(pick_frames, ignore_index=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_content_hash(frame: pd.DataFrame) -> str:
    fields = [
        column
        for column in ("date", "code", "open", "high", "low", "close", "traded")
        if column in frame
    ]
    hashed = pd.util.hash_pandas_object(frame[fields], index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fixed profit-first models on the production panel"
    )
    parser.add_argument("--daily", required=True, help="canonical daily .pkl/.csv/.parquet")
    parser.add_argument(
        "--output",
        default="research/profit_model_comparison.json",
        help="comparison JSON output",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    daily_path = Path(args.daily).resolve()
    output_path = Path(args.output).resolve()
    config = RankerConfig()
    canonical = normalize_daily_prices(read_frame(daily_path))
    canonical = canonical[canonical["date"].le(BENCHMARK_END)].copy()
    # Make the calendar contract explicit even for this historical comparison.
    # This calendar is the canonical input's observed-session list, not an
    # independent exchange calendar; that limitation is recorded in the output.
    expected_sessions = normalize_expected_sessions(
        canonical["date"].drop_duplicates(),
        through=BENCHMARK_END,
    )
    expected_sessions_sha256 = session_calendar_hash(
        expected_sessions,
        through=BENCHMARK_END,
    )
    modeling, coverage = prepare_modeling_prices(
        canonical,
        coverage_lookback=config.source_coverage_lookback,
        minimum_source_coverage=config.minimum_source_coverage,
        expected_sessions=expected_sessions,
    )
    panel = build_feature_panel(modeling, config)
    families = _families(config, args.seed)
    model_names = list(families)

    exploration_results, exploration_picks = _evaluate_models(
        panel,
        families,
        model_names,
        evaluation_start=EXPLORATION_START,
        evaluation_end=EXPLORATION_END,
        train_start=EXPLORATION_TRAIN_START,
        cost_bps=config.cost_bps,
    )
    selection_results, selection_picks = _evaluate_models(
        panel,
        families,
        model_names,
        evaluation_start=SELECTION_START,
        evaluation_end=SELECTION_END,
        train_start=pd.Timestamp(config.regime_start),
        cost_bps=config.cost_bps,
    )
    winner = sorted(
        model_names,
        key=lambda name: (
            -selection_results[name]["top1"]["net_mean_pct_at_cost"],
            name,
        ),
    )[0]
    benchmark_results, benchmark_picks = _evaluate_models(
        panel,
        families,
        [winner],
        evaluation_start=BENCHMARK_START,
        evaluation_end=BENCHMARK_END,
        train_start=pd.Timestamp(config.regime_start),
        cost_bps=config.cost_bps,
    )

    incomplete = coverage.loc[~coverage["source_complete"], "date"]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "objective": "top1 scheduled-day mean net open_to_close return after cost",
        "selection_rule": (
            "choose the fixed candidate with maximum 2025-01..03 "
            "top1.net_mean_pct_at_cost; benchmark winner only"
        ),
        "hyperparameter_policy": (
            "one predeclared specification per family; exploration is diagnostic "
            "and does not tune a grid"
        ),
        "leakage_guards": {
            "candidate_universe": "prior-session universe from production panel",
            "training_cutoff": "strictly before each scoring month",
            "scoring_filter": "eligible only; no label/OHLC/execution filter",
            "unfilled": "rank first, then zero return and zero cost",
            "feature_time_assertion": "feature_source_max_date < target date",
            "benchmark_access": "only the selection-period winner is evaluated",
            "session_calendar": (
                "explicit observed-session list from canonical input; detects "
                "row-level incomplete sessions but cannot prove that an entirely "
                "absent exchange session was not omitted upstream"
            ),
        },
        "seed": int(args.seed),
        "cost_bps": config.cost_bps,
        "config": config.to_dict(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "file_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "data": {
            "path": str(daily_path),
            "file_sha256": _sha256_file(daily_path),
            "canonical_content_sha256": _data_content_hash(canonical),
            "raw_rows_through_benchmark_end": int(len(canonical)),
            "modeling_rows": int(len(modeling)),
            "panel_rows": int(len(panel)),
            "codes": int(canonical["code"].nunique()),
            "sessions": int(canonical["date"].nunique()),
            "session_calendar_mode": "explicit_observed_sessions_from_canonical_input",
            "session_calendar_sha256": expected_sessions_sha256,
            "session_calendar_sessions": int(len(expected_sessions)),
            "session_calendar_through": str(BENCHMARK_END.date()),
            "min_date": str(canonical["date"].min().date()),
            "max_date": str(canonical["date"].max().date()),
            "partial_session_rows": int(canonical["partial_session"].sum()),
            "no_trade_rows": int((~canonical["traded"]).sum()),
            "source_incomplete_dates": [
                str(pd.Timestamp(value).date()) for value in incomplete
            ],
        },
        "model_specifications": {
            name: families[name].specification() for name in model_names
        },
        "periods": {
            "exploration": asdict(
                PeriodDefinition(
                    str(EXPLORATION_START.date()),
                    str(EXPLORATION_END.date()),
                    str(EXPLORATION_TRAIN_START.date()),
                )
            ),
            "selection": asdict(
                PeriodDefinition(
                    str(SELECTION_START.date()),
                    str(SELECTION_END.date()),
                    config.regime_start,
                )
            ),
            "known_benchmark": asdict(
                PeriodDefinition(
                    str(BENCHMARK_START.date()),
                    str(BENCHMARK_END.date()),
                    config.regime_start,
                )
            ),
        },
        "exploration": exploration_results,
        "selection": selection_results,
        "selected_winner": winner,
        "known_benchmark_winner_only": benchmark_results[winner],
        "benchmark_is_true_holdout": False,
    }
    write_json(payload, output_path)
    stem = output_path.with_suffix("")
    write_frame(exploration_picks, stem.with_name(stem.name + "_exploration_picks.csv"))
    write_frame(selection_picks, stem.with_name(stem.name + "_selection_picks.csv"))
    write_frame(benchmark_picks, stem.with_name(stem.name + "_benchmark_picks.csv"))
    print(
        pd.DataFrame(
            [
                {
                    "model": name,
                    "selection_top1_net_pct": selection_results[name]["top1"][
                        "net_mean_pct_at_cost"
                    ],
                    "selection_top2_net_pct": selection_results[name]["top2"][
                        "net_mean_pct_at_cost"
                    ],
                }
                for name in model_names
            ]
        )
        .sort_values("selection_top1_net_pct", ascending=False)
        .to_string(index=False)
    )
    print(f"winner={winner}")
    print(
        "known_benchmark_top1_net_pct="
        f"{benchmark_results[winner]['top1']['net_mean_pct_at_cost']:.10f}"
    )
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
