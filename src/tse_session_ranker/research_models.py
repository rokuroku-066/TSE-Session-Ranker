"""Auditable model families for profit-first ranking research.

This module is deliberately separate from the production artifact path.  The
production model remains the schema-5 logistic-regression artifact until a
future, genuinely forward validation promotes a research specification.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .exceptions import DataValidationError


SUPPORTED_RESEARCH_FAMILIES = frozenset(
    {
        "legacy_weight_logit",
        "raked_logit",
        "return_weighted_logit",
        "ridge_return",
        "ridge_daily_rank",
        "elastic_net_sgd_return",
        "huber_sgd_return",
        "hist_gradient_boosting_return",
        "hist_gradient_boosting_classifier",
        "hist_gradient_boosting_rank",
        "extra_trees_return",
        "extra_trees_classifier",
        "random_forest_return",
        "pairwise_linear_rank",
        "multi_threshold_expected_return",
    }
)

SUPPORTED_BINARY_OBJECTIVES = frozenset(
    {
        "positive_session",
        "net_positive_20bp",
        "same_day_top_quintile",
    }
)


_ALLOWED_PARAMETERS: dict[str, frozenset[str]] = {
    "legacy_weight_logit": frozenset(
        {"C", "class_balance", "cost_bps", "max_iter"}
    ),
    "raked_logit": frozenset({"C", "class_balance", "cost_bps", "max_iter"}),
    "return_weighted_logit": frozenset(
        {
            "C",
            "class_balance",
            "cost_bps",
            "magnitude_floor_pct",
            "magnitude_cap_pct",
            "max_iter",
        }
    ),
    "ridge_return": frozenset({"alpha", "return_clip_pct"}),
    "ridge_daily_rank": frozenset({"alpha"}),
    "elastic_net_sgd_return": frozenset(
        {
            "alpha",
            "l1_ratio",
            "epsilon",
            "max_iter",
            "tol",
            "average",
            "return_clip_pct",
        }
    ),
    "huber_sgd_return": frozenset(
        {
            "alpha",
            "l1_ratio",
            "epsilon",
            "max_iter",
            "tol",
            "average",
            "return_clip_pct",
        }
    ),
    "hist_gradient_boosting_return": frozenset(
        {
            "loss",
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "max_depth",
            "min_samples_leaf",
            "l2_regularization",
            "max_bins",
            "return_clip_pct",
        }
    ),
    "hist_gradient_boosting_rank": frozenset(
        {
            "loss",
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "max_depth",
            "min_samples_leaf",
            "l2_regularization",
            "max_bins",
            "return_clip_pct",
        }
    ),
    "hist_gradient_boosting_classifier": frozenset(
        {
            "class_balance",
            "cost_bps",
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "max_depth",
            "min_samples_leaf",
            "l2_regularization",
            "max_bins",
        }
    ),
    "extra_trees_return": frozenset(
        {
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "max_samples",
            "n_jobs",
            "return_clip_pct",
        }
    ),
    "extra_trees_classifier": frozenset(
        {
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "max_samples",
            "n_jobs",
            "class_balance",
            "cost_bps",
        }
    ),
    "random_forest_return": frozenset(
        {
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "max_samples",
            "n_jobs",
            "return_clip_pct",
        }
    ),
    "pairwise_linear_rank": frozenset({"C", "pairs_per_date", "max_iter"}),
    "multi_threshold_expected_return": frozenset(
        {"C", "thresholds_pct", "return_clip_pct", "max_iter"}
    ),
}


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _finite_positive(value: Any, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _validate_parameters(family: str, objective: str, parameters: Mapping[str, Any]) -> None:
    unknown = sorted(set(parameters) - _ALLOWED_PARAMETERS[family])
    if unknown:
        raise ValueError(f"unknown parameters for {family}: {unknown}")
    binary_families = {
        "legacy_weight_logit",
        "raked_logit",
        "return_weighted_logit",
        "hist_gradient_boosting_classifier",
        "extra_trees_classifier",
    }
    if family in binary_families and objective not in SUPPORTED_BINARY_OBJECTIVES:
        raise ValueError(f"{family} requires a supported binary objective")
    exact_objectives = {
        "ridge_return": "raw_return",
        "ridge_daily_rank": "same_day_return_percentile",
        "elastic_net_sgd_return": "raw_return",
        "huber_sgd_return": "raw_return",
        "hist_gradient_boosting_return": "raw_return",
        "hist_gradient_boosting_rank": "same_day_return_percentile",
        "extra_trees_return": "raw_return",
        "random_forest_return": "raw_return",
        "pairwise_linear_rank": "same_day_pairwise_return",
        "multi_threshold_expected_return": "threshold_integrated_return",
    }
    if family in exact_objectives and objective != exact_objectives[family]:
        raise ValueError(
            f"{family} requires objective {exact_objectives[family]!r}"
        )
    if "C" in parameters:
        _finite_positive(parameters["C"], "C")
    if "alpha" in parameters:
        _finite_positive(parameters["alpha"], "alpha")
    if "learning_rate" in parameters:
        _finite_positive(parameters["learning_rate"], "learning_rate")
    if "return_clip_pct" in parameters:
        _finite_positive(parameters["return_clip_pct"], "return_clip_pct")
    for name in ("max_iter", "max_leaf_nodes", "min_samples_leaf", "max_bins", "n_estimators", "pairs_per_date"):
        if name in parameters and int(parameters[name]) < 1:
            raise ValueError(f"{name} must be at least one")
    if "l1_ratio" in parameters:
        ratio = float(parameters["l1_ratio"])
        if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("l1_ratio must be between zero and one")
    if "max_samples" in parameters:
        sample_fraction = float(parameters["max_samples"])
        if not np.isfinite(sample_fraction) or not 0.0 < sample_fraction <= 1.0:
            raise ValueError("max_samples must be in (0, 1]")
        if not bool(parameters.get("bootstrap", False)):
            raise ValueError("max_samples requires bootstrap=true")
    if family == "multi_threshold_expected_return":
        cap = _finite_positive(parameters.get("return_clip_pct", 3.0), "return_clip_pct")
        thresholds = tuple(
            float(value)
            for value in parameters.get("thresholds_pct", (-2.0, -0.5, 0.5, 2.0))
        )
        if (
            not thresholds
            or not np.isfinite(np.asarray(thresholds, dtype=float)).all()
            or any(
                left >= right
                for left, right in zip(thresholds[:-1], thresholds[1:])
            )
            or thresholds[0] <= -cap
            or thresholds[-1] >= cap
        ):
            raise ValueError(
                "multi-threshold values must be finite, strictly ordered, and inside the return clip"
            )


@dataclass(frozen=True)
class ResearchModelSpec:
    """Immutable, content-addressed research model specification."""

    name: str
    family: str
    parameters: Mapping[str, Any]
    objective: str = "raw_return"
    random_state: int = 31

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("research model name must not be empty")
        if self.family not in SUPPORTED_RESEARCH_FAMILIES:
            raise ValueError(f"unsupported research model family: {self.family}")
        if self.random_state < 0:
            raise ValueError("random_state must be non-negative")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("research model parameters must be a mapping")
        frozen = _freeze_json(self.parameters)
        object.__setattr__(self, "parameters", frozen)
        _validate_parameters(self.family, self.objective, frozen)
        try:
            json.dumps(self.canonical_dict(), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("research model parameters must be finite JSON") from exc

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "objective": self.objective,
            "parameters": _thaw_json(self.parameters),
            "random_state": self.random_state,
        }

    @property
    def spec_id(self) -> str:
        encoded = json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return f"{self.name}__{hashlib.sha256(encoded).hexdigest()[:16]}"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ResearchModelSpec":
        return cls(
            name=str(raw["name"]),
            family=str(raw["family"]),
            objective=str(raw.get("objective", "raw_return")),
            parameters=dict(raw.get("parameters", {})),
            random_state=int(raw.get("random_state", 31)),
        )


class FittedResearchScorer(Protocol):
    def score(self, frame: pd.DataFrame) -> np.ndarray: ...


class _EstimatorScorer:
    def __init__(
        self,
        estimator: Any,
        feature_columns: Sequence[str],
        score_method: str,
    ) -> None:
        self.estimator = estimator
        self.feature_columns = tuple(feature_columns)
        self.score_method = score_method

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        features = frame.loc[:, list(self.feature_columns)]
        if self.score_method == "prediction":
            values = self.estimator.predict(features)
        elif self.score_method == "positive_probability":
            values = self.estimator.predict_proba(features)[:, 1]
        else:  # pragma: no cover - construction is internal
            raise RuntimeError(f"unknown score method: {self.score_method}")
        result = np.asarray(values, dtype=float)
        if result.shape != (len(frame),) or not np.isfinite(result).all():
            raise DataValidationError("research scorer produced invalid scores")
        return result


class _PairwiseScorer:
    def __init__(
        self,
        preprocessor: Pipeline,
        estimator: LogisticRegression,
        feature_columns: Sequence[str],
    ) -> None:
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.feature_columns = tuple(feature_columns)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(
            frame.loc[:, list(self.feature_columns)]
        )
        result = np.asarray(self.estimator.decision_function(transformed), dtype=float)
        if result.shape != (len(frame),) or not np.isfinite(result).all():
            raise DataValidationError("pairwise scorer produced invalid scores")
        return result


class _MultiThresholdScorer:
    def __init__(
        self,
        models: Sequence[Pipeline],
        thresholds: Sequence[float],
        lower: float,
        upper: float,
        feature_columns: Sequence[str],
    ) -> None:
        self.models = tuple(models)
        self.thresholds = tuple(float(value) for value in thresholds)
        self.lower = float(lower)
        self.upper = float(upper)
        self.feature_columns = tuple(feature_columns)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        features = frame.loc[:, list(self.feature_columns)]
        midpoints = tuple(
            (left + right) / 2.0
            for left, right in zip(self.thresholds[:-1], self.thresholds[1:])
        )
        boundaries = (self.lower, *midpoints, self.upper)
        widths = np.diff(np.asarray(boundaries, dtype=float))
        result = np.full(len(frame), self.lower, dtype=float)
        for width, model in zip(widths, self.models, strict=True):
            result += width * model.predict_proba(features)[:, 1]
        if not np.isfinite(result).all():
            raise DataValidationError("multi-threshold scorer produced invalid scores")
        return result


def _required_training(frame: pd.DataFrame, feature_columns: Sequence[str]) -> None:
    required = {"date", "code", "oc_return_pct", *feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"research training frame lacks columns: {missing}")
    if frame.empty:
        raise DataValidationError("research training frame is empty")
    returns = pd.to_numeric(frame["oc_return_pct"], errors="coerce")
    if returns.isna().any() or not np.isfinite(returns.to_numpy()).all():
        raise DataValidationError("research training returns are invalid")
    if frame[["date", "code"]].duplicated().any():
        raise DataValidationError("research training frame has duplicate date/code rows")


def date_equal_weights(frame: pd.DataFrame, magnitude: pd.Series | None = None) -> pd.Series:
    """Return weights whose total is exactly one for every training date."""

    base = (
        pd.Series(1.0, index=frame.index, dtype=float)
        if magnitude is None
        else pd.to_numeric(magnitude, errors="coerce").astype(float)
    )
    if base.isna().any() or (~np.isfinite(base.to_numpy())).any() or base.le(0).any():
        raise ValueError("training magnitudes must be finite and positive")
    totals = base.groupby(frame["date"], sort=False).transform("sum")
    weights = base / totals
    per_date = weights.groupby(frame["date"], sort=False).sum()
    if not np.allclose(per_date.to_numpy(), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("training weights are not date-equal")
    return weights


def date_and_class_equal_weights(
    frame: pd.DataFrame,
    target: pd.Series,
    magnitude: pd.Series | None = None,
) -> pd.Series:
    """Rake positive/negative class and date margins without double weighting."""

    y = pd.Series(target, index=frame.index).astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("binary target must contain both classes")
    base = (
        pd.Series(1.0, index=frame.index, dtype=float)
        if magnitude is None
        else pd.to_numeric(magnitude, errors="coerce").astype(float)
    )
    if base.isna().any() or (~np.isfinite(base.to_numpy())).any() or base.le(0).any():
        raise ValueError("training magnitudes must be finite and positive")
    sufficient = pd.DataFrame(
        {"date": frame["date"].to_numpy(), "target": y.to_numpy(), "weight": base}
    ).pivot_table(
        index="date", columns="target", values="weight", aggfunc="sum", fill_value=0.0
    )
    negative = sufficient.get(0, pd.Series(0.0, index=sufficient.index)).to_numpy()
    positive = sufficient.get(1, pd.Series(0.0, index=sufficient.index)).to_numpy()
    desired = frame["date"].nunique() / 2.0

    def positive_total(log_multiplier: float) -> float:
        multiplier = np.exp(log_multiplier)
        denominator = negative + multiplier * positive
        return float(
            np.divide(
                multiplier * positive,
                denominator,
                out=np.zeros_like(denominator, dtype=float),
                where=denominator > 0,
            ).sum()
        )

    lower, upper = -50.0, 50.0
    if not positive_total(lower) <= desired <= positive_total(upper):
        raise ValueError("date/class weight margins are infeasible")
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if positive_total(midpoint) < desired:
            lower = midpoint
        else:
            upper = midpoint
    multiplier = float(np.exp((lower + upper) / 2.0))
    weighted = base * np.where(y.eq(1), multiplier, 1.0)
    weights = date_equal_weights(frame, pd.Series(weighted, index=frame.index))
    expected = frame["date"].nunique() / 2.0
    per_class = weights.groupby(y, sort=False).sum().reindex([0, 1])
    if not np.allclose(per_class.to_numpy(), expected, atol=1e-8, rtol=0.0):
        raise AssertionError("training weights are not class-balanced")
    return weights


def binary_objective(
    frame: pd.DataFrame,
    objective: str,
    *,
    cost_bps: float = 20.0,
) -> pd.Series:
    returns = pd.to_numeric(frame["oc_return_pct"], errors="coerce")
    if objective == "positive_session":
        target = returns.gt(0.0)
    elif objective == "net_positive_20bp":
        target = returns.gt(cost_bps / 100.0)
    elif objective == "same_day_top_quintile":
        ranks = returns.groupby(frame["date"], sort=False).rank(
            method="average", pct=True
        )
        target = ranks.gt(0.80)
    else:
        raise ValueError(f"unsupported binary objective: {objective}")
    result = target.astype(int)
    if set(result.unique()) != {0, 1}:
        raise ValueError("binary objective has fewer than two classes")
    return result


def same_day_return_percentile_target(frame: pd.DataFrame) -> pd.Series:
    """Map each date's realised return ranks to the interval ``[-1, 1]``.

    This is a research target, not a pre-open feature.  It may therefore use
    realised returns from the *training* dates, but callers must keep every
    scoring date strictly outside the fitted frame.  Ranking within each date
    makes the objective insensitive to market-wide shifts in the absolute
    return level and aligns training with the daily cross-sectional selection
    decision.  The frozen formula is ``2 * rank(pct=True) - 1``.  For a finite
    tie-free cross-section of size ``n`` its minimum is ``2 / n - 1`` and its
    daily mean is ``1 / n``; it is percentile-scaled but not exactly centred.
    """

    required = {"date", "oc_return_pct"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(
            f"daily-rank target frame lacks columns: {missing}"
        )
    returns = pd.to_numeric(frame["oc_return_pct"], errors="coerce")
    if returns.isna().any() or not np.isfinite(returns.to_numpy()).all():
        raise DataValidationError("daily-rank target returns are invalid")
    target = (
        returns.groupby(frame["date"], sort=False)
        .rank(method="average", pct=True)
        .mul(2.0)
        .sub(1.0)
    )
    if target.shape != (len(frame),) or not np.isfinite(target.to_numpy()).all():
        raise DataValidationError("daily-rank target is invalid")
    return target.astype(float)


def _linear_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def _tree_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", model),
        ]
    )


def _clipped_return(frame: pd.DataFrame, cap: float) -> pd.Series:
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("return clip must be finite and positive")
    return pd.to_numeric(frame["oc_return_pct"], errors="coerce").clip(-cap, cap)


def _fit_pairwise(
    spec: ResearchModelSpec,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> FittedResearchScorer:
    params = dict(spec.parameters)
    preprocessor = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    transformed = np.asarray(
        preprocessor.fit_transform(frame.loc[:, list(feature_columns)]), dtype=float
    )
    realised = frame["oc_return_pct"].to_numpy(dtype=float)
    pairs_per_date = int(params.get("pairs_per_date", 64))
    rng = np.random.default_rng(spec.random_state)
    differences: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for positions in frame.groupby("date", sort=True).indices.values():
        positions = np.asarray(positions, dtype=int)
        if len(positions) < 2:
            continue
        pair_count = min(pairs_per_date, len(positions) * (len(positions) - 1) // 2)
        left = rng.choice(positions, pair_count, replace=True)
        right = rng.choice(positions, pair_count, replace=True)
        equal_position = left == right
        while equal_position.any():
            right[equal_position] = rng.choice(
                positions, int(equal_position.sum()), replace=True
            )
            equal_position = left == right
        unequal = realised[left] != realised[right]
        left, right = left[unequal], right[unequal]
        if not len(left):
            continue
        high = np.where(realised[left] > realised[right], left, right)
        low = np.where(realised[left] > realised[right], right, left)
        positive = transformed[high] - transformed[low]
        differences.extend([positive, -positive])
        labels.extend(
            [np.ones(len(positive), dtype=int), np.zeros(len(positive), dtype=int)]
        )
        per_orientation = 0.5 / len(positive)
        weights.extend(
            [
                np.full(len(positive), per_orientation),
                np.full(len(positive), per_orientation),
            ]
        )
    if not differences:
        raise DataValidationError("pairwise model could not construct training pairs")
    estimator = LogisticRegression(
        C=float(params.get("C", 0.08)),
        class_weight=None,
        fit_intercept=False,
        max_iter=int(params.get("max_iter", 1_000)),
        random_state=spec.random_state,
    )
    estimator.fit(
        np.vstack(differences), np.concatenate(labels), sample_weight=np.concatenate(weights)
    )
    return _PairwiseScorer(preprocessor, estimator, feature_columns)


def fit_research_model(
    spec: ResearchModelSpec,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> FittedResearchScorer:
    """Fit one predeclared research model on a point-in-time training frame."""

    columns = tuple(feature_columns)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("feature columns must be non-empty and unique")
    _required_training(frame, columns)
    params = dict(spec.parameters)
    family = spec.family
    features = frame.loc[:, list(columns)]

    if family == "pairwise_linear_rank":
        return _fit_pairwise(spec, frame, columns)

    if family in {"legacy_weight_logit", "raked_logit", "return_weighted_logit"}:
        target = binary_objective(
            frame,
            spec.objective,
            cost_bps=float(params.get("cost_bps", 20.0)),
        )
        magnitude = None
        if family == "return_weighted_logit":
            threshold = float(params.get("cost_bps", 20.0)) / 100.0
            magnitude = (
                (frame["oc_return_pct"] - threshold)
                .abs()
                .clip(
                    float(params.get("magnitude_floor_pct", 0.25)),
                    float(params.get("magnitude_cap_pct", 3.0)),
                )
            )
        if family == "legacy_weight_logit":
            weights = date_equal_weights(frame, magnitude)
            estimator_class_weight: str | None = (
                "balanced" if bool(params.get("class_balance", True)) else None
            )
        else:
            weights = (
                date_and_class_equal_weights(frame, target, magnitude)
                if bool(params.get("class_balance", True))
                else date_equal_weights(frame, magnitude)
            )
            estimator_class_weight = None
        estimator = _linear_pipeline(
            LogisticRegression(
                C=float(params.get("C", 0.08)),
                class_weight=estimator_class_weight,
                max_iter=int(params.get("max_iter", 1_000)),
                random_state=spec.random_state,
                solver="lbfgs",
            )
        )
        estimator.fit(features, target, model__sample_weight=weights)
        return _EstimatorScorer(estimator, columns, "positive_probability")

    if family == "multi_threshold_expected_return":
        cap = float(params.get("return_clip_pct", 3.0))
        thresholds = tuple(
            float(value)
            for value in params.get("thresholds_pct", (-2, -0.5, 0.5, 2))
        )
        if (
            not np.isfinite(cap)
            or cap <= 0
            or not thresholds
            or not np.isfinite(np.asarray(thresholds, dtype=float)).all()
            or any(left >= right for left, right in zip(thresholds[:-1], thresholds[1:]))
            or thresholds[0] <= -cap
            or thresholds[-1] >= cap
        ):
            raise ValueError(
                "multi-threshold values must be finite, strictly ordered, and inside the return clip"
            )
        weights = date_equal_weights(frame)
        models: list[Pipeline] = []
        for threshold in thresholds:
            target = frame["oc_return_pct"].gt(threshold).astype(int)
            if set(target.unique()) != {0, 1}:
                raise DataValidationError("multi-threshold target lacks both classes")
            estimator = _linear_pipeline(
                LogisticRegression(
                    C=float(params.get("C", 0.08)),
                    class_weight=None,
                    max_iter=int(params.get("max_iter", 1_000)),
                    random_state=spec.random_state,
                )
            )
            estimator.fit(features, target, model__sample_weight=weights)
            models.append(estimator)
        return _MultiThresholdScorer(models, thresholds, -cap, cap, columns)

    weights = date_equal_weights(frame)
    clip = float(params.get("return_clip_pct", 5.0))
    return_target = _clipped_return(frame, clip)

    if family == "ridge_return":
        estimator = _linear_pipeline(Ridge(alpha=float(params.get("alpha", 10.0))))
        estimator.fit(features, return_target, model__sample_weight=weights)
        return _EstimatorScorer(estimator, columns, "prediction")

    if family == "ridge_daily_rank":
        target = same_day_return_percentile_target(frame)
        estimator = _linear_pipeline(Ridge(alpha=float(params.get("alpha", 10.0))))
        estimator.fit(features, target, model__sample_weight=weights)
        return _EstimatorScorer(estimator, columns, "prediction")

    if family in {"elastic_net_sgd_return", "huber_sgd_return"}:
        loss = "squared_error" if family == "elastic_net_sgd_return" else "huber"
        penalty = "elasticnet" if family == "elastic_net_sgd_return" else "l2"
        estimator = _linear_pipeline(
            SGDRegressor(
                loss=loss,
                penalty=penalty,
                alpha=float(params.get("alpha", 0.0001)),
                l1_ratio=float(params.get("l1_ratio", 0.15)),
                epsilon=float(params.get("epsilon", 1.0)),
                max_iter=int(params.get("max_iter", 2_000)),
                tol=float(params.get("tol", 1e-4)),
                average=bool(params.get("average", True)),
                shuffle=True,
                random_state=spec.random_state,
            )
        )
        estimator.fit(features, return_target, model__sample_weight=weights)
        return _EstimatorScorer(estimator, columns, "prediction")

    if family in {
        "hist_gradient_boosting_return",
        "hist_gradient_boosting_rank",
    }:
        target = return_target
        if family == "hist_gradient_boosting_rank":
            target = (
                frame["oc_return_pct"]
                .groupby(frame["date"], sort=False)
                .rank(method="average", pct=True)
                .mul(2.0)
                .sub(1.0)
            )
        estimator = HistGradientBoostingRegressor(
            loss=str(params.get("loss", "squared_error")),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_iter=int(params.get("max_iter", 120)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 200)),
            l2_regularization=float(params.get("l2_regularization", 1.0)),
            max_bins=int(params.get("max_bins", 63)),
            early_stopping=False,
            random_state=spec.random_state,
        )
        estimator.fit(features, target, sample_weight=weights)
        return _EstimatorScorer(estimator, columns, "prediction")

    if family == "hist_gradient_boosting_classifier":
        target = binary_objective(
            frame,
            spec.objective,
            cost_bps=float(params.get("cost_bps", 20.0)),
        )
        class_weights = (
            date_and_class_equal_weights(frame, target)
            if bool(params.get("class_balance", True))
            else weights
        )
        estimator = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_iter=int(params.get("max_iter", 120)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 200)),
            l2_regularization=float(params.get("l2_regularization", 1.0)),
            max_bins=int(params.get("max_bins", 63)),
            early_stopping=False,
            random_state=spec.random_state,
        )
        estimator.fit(features, target, sample_weight=class_weights)
        return _EstimatorScorer(estimator, columns, "positive_probability")

    if family in {
        "extra_trees_return",
        "extra_trees_classifier",
        "random_forest_return",
    }:
        bootstrap = bool(params.get("bootstrap", False))
        common = {
            "n_estimators": int(params.get("n_estimators", 64)),
            "max_depth": params.get("max_depth", 8),
            "min_samples_leaf": int(params.get("min_samples_leaf", 200)),
            "max_features": params.get("max_features", 0.7),
            "bootstrap": bootstrap,
            "n_jobs": int(params.get("n_jobs", 1)),
            "random_state": spec.random_state,
        }
        if bootstrap:
            common["max_samples"] = float(params.get("max_samples", 0.65))
        if family == "extra_trees_classifier":
            target = binary_objective(
                frame,
                spec.objective,
                cost_bps=float(params.get("cost_bps", 20.0)),
            )
            model = ExtraTreesClassifier(class_weight=None, **common)
            fit_weights = (
                date_and_class_equal_weights(frame, target)
                if bool(params.get("class_balance", True))
                else weights
            )
            method = "positive_probability"
        elif family == "random_forest_return":
            target = return_target
            model = RandomForestRegressor(**common)
            fit_weights = weights
            method = "prediction"
        else:
            target = return_target
            model = ExtraTreesRegressor(**common)
            fit_weights = weights
            method = "prediction"
        estimator = _tree_pipeline(model)
        estimator.fit(features, target, model__sample_weight=fit_weights)
        return _EstimatorScorer(estimator, columns, method)

    raise ValueError(f"unimplemented research model family: {family}")
