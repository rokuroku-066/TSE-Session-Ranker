#!/usr/bin/env python3
"""Preregistered profit-first selection for the session-v4 logit ranker.

This program has two deliberately separate phases:

``select``
    Uses only the development data and writes an immutable selection lock.

``holdout``
    Reads the winner from that lock and evaluates it once on the sealed
    2025-08..2026-03 interval.  The holdout phase has no candidate-selection
    code path.

The estimator family is always an L2-regularized logistic regression.  Model
design means the point-in-time feature block, binary target, sample weighting,
regularization strength, training horizon, and class weighting.  The primary
selection metric is daily top-1 open-to-close return after 20 bp round-trip
cost, not classification accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import warnings
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tse_session_ranker.config import RankerConfig
from tse_session_ranker.data.common import (
    SESSION_OHLC,
    merge_daily_prices,
    normalize_daily_prices,
    normalize_expected_sessions,
    prepare_modeling_prices,
    session_calendar_hash,
)
from tse_session_ranker.data.jpx import (
    PARSER_VERSION as JPX_PARSER_VERSION,
    collect_jpx,
)
from tse_session_ranker.data.tdnet import (
    TDNET_FEATURE_COLUMNS,
    TDNET_MODEL_FEATURE_COLUMNS,
    build_tdnet_features,
    normalize_tdnet_disclosures,
)
from tse_session_ranker.features import PRICE_FEATURE_COLUMNS, build_feature_panel
from tse_session_ranker.io import json_dumps, read_frame, write_frame, write_json
from tse_session_ranker.profit import daily_portfolio_returns, profit_metrics
from tse_session_ranker.research_features import (
    MARKET_FEATURE_COLUMNS,
    SESSION_FEATURE_COLUMNS,
    add_session_market_features,
)
from tse_session_ranker.validation import (
    moving_block_bootstrap_suite,
    paired_moving_block_bootstrap_suite,
    return_diagnostics,
)


PROTOCOL_PATH = Path(__file__).with_name("model_v04_protocol.json")
TRAIN_START = pd.Timestamp("2021-01-04")
FEATURE_SCREEN_START = pd.Timestamp("2022-03-01")
FEATURE_SCREEN_END = pd.Timestamp("2022-12-30")
MODEL_DESIGN_START = pd.Timestamp("2023-01-04")
MODEL_DESIGN_END = pd.Timestamp("2023-12-29")
CONFIRMATION_START = pd.Timestamp("2024-01-04")
CONFIRMATION_END = pd.Timestamp("2025-07-31")
LEGACY_COMPARISON_START = pd.Timestamp("2025-01-06")
HOLDOUT_START = pd.Timestamp("2025-08-04")
HOLDOUT_END = pd.Timestamp("2026-03-31")
LEGACY_REGIME_START = pd.Timestamp("2024-11-06")
DEFAULT_SEED = 31
DEFAULT_BOOTSTRAP_SAMPLES = 20_000
PRIMARY_COST_BPS = 20.0
STRESS_COST_BPS = 40.0
SEALED_JPX_CANONICAL_FILES_SHA256 = (
    "c10d68a08f872cf47f1eccb6f36a4c06568b8f72be7fe987105b6c3960689492"
)
SEALED_JPX_MANIFEST_SHA256 = (
    "3c4e5dac4d1f2557f26b1f3c79e29d559e8fd6ec18e4266cff7600e7d4fea532"
)
HOLDOUT_WARMUP_JPX_NAME = "stq_20250801.pdf"
HOLDOUT_WARMUP_JPX_BYTES = 5_106_805
HOLDOUT_WARMUP_JPX_SHA256 = (
    "6cf19993734428bef231defba216277b5683bd0e5f5a892f21e9495f57774d18"
)


DAILY_RANK_SOURCES: tuple[str, ...] = (
    "oc_last",
    "oc_mean_5",
    "oc_mean_20",
    "oc_std_20",
    "overnight_last",
    "overnight_mean_20",
    "night_day_corr_60",
    "atr14_pct",
    "close_momentum_5",
    "close_momentum_20",
    "close_momentum_60",
    "prior_close_location_20",
)

DAILY_RANK_COLUMNS: tuple[str, ...] = tuple(
    f"xrank_{column}" for column in DAILY_RANK_SOURCES
)

DAILY_NONLINEAR_COLUMNS: tuple[str, ...] = (
    "prior_oc_atr_clipped",
    "prior_oc_atr_abs",
    "prior_overnight_atr_clipped",
    "prior_overnight_atr_abs",
)

BOUNDED_DAILY_COLUMNS: tuple[str, ...] = (
    *DAILY_RANK_COLUMNS,
    *DAILY_NONLINEAR_COLUMNS,
)

TDNET_RESEARCH_COLUMNS: tuple[str, ...] = (
    "tdnet_any",
    "tdnet_count_log1p",
    "tdnet_postclose_count_log1p",
    "tdnet_latest_age_hours_log1p",
    "tdnet_has_earnings",
    "tdnet_has_revision",
    "tdnet_has_revision_up",
    "tdnet_has_revision_down",
    "tdnet_has_dividend",
    "tdnet_has_dividend_up",
    "tdnet_has_dividend_down",
    "tdnet_has_buyback_decision",
    "tdnet_has_buyback_tostnet",
    "tdnet_has_equity_financing",
    "tdnet_has_benefit",
    "tdnet_has_split",
    "tdnet_has_ma_alliance",
    "tdnet_has_control_transaction",
    "tdnet_has_impairment_loss",
    "tdnet_has_audit_problem",
)

TDNET_RUNUP_INTERACTION_COLUMNS: tuple[str, ...] = (
    "tdnet_revision_x_xrank_oc_mean_20",
    "tdnet_equity_financing_x_xrank_oc_mean_20",
)

TDNET_MARKET_INTERACTION_COLUMNS: tuple[str, ...] = (
    "tdnet_any_x_market_beta_x_prior_market_return",
)


def _ordered_union(*groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(column for group in groups for column in group))


def feature_blocks() -> dict[str, tuple[str, ...]]:
    """Return the six blocks frozen in ``model_v04_protocol.json``."""

    session_shape = _ordered_union(BOUNDED_DAILY_COLUMNS, SESSION_FEATURE_COLUMNS)
    session_market = _ordered_union(session_shape, MARKET_FEATURE_COLUMNS)
    return {
        "legacy_v03": _ordered_union(
            PRICE_FEATURE_COLUMNS, TDNET_MODEL_FEATURE_COLUMNS
        ),
        "bounded_daily": BOUNDED_DAILY_COLUMNS,
        "session_shape": session_shape,
        "session_market": session_market,
        "session_tdnet": _ordered_union(
            session_shape,
            TDNET_RESEARCH_COLUMNS,
            TDNET_RUNUP_INTERACTION_COLUMNS,
        ),
        "session_market_tdnet": _ordered_union(
            session_market,
            TDNET_RESEARCH_COLUMNS,
            TDNET_RUNUP_INTERACTION_COLUMNS,
            TDNET_MARKET_INTERACTION_COLUMNS,
        ),
    }


OBJECTIVES: tuple[str, ...] = (
    "positive_session",
    "net_positive_20bp",
    "net_positive_magnitude_weighted",
    "same_day_top_quintile",
)


@dataclass(frozen=True)
class ModelSpec:
    feature_block: str
    objective: str
    c: float = 0.08
    training_horizon_sessions: int | None = None
    class_weight: str | None = "balanced"
    train_start: str = str(TRAIN_START.date())
    min_train_sessions: int = 252
    legacy_estimator_class_weight: bool = False

    def __post_init__(self) -> None:
        if self.feature_block not in feature_blocks():
            raise ValueError(f"unknown feature block: {self.feature_block}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"unknown objective: {self.objective}")
        if self.c <= 0:
            raise ValueError("C must be positive")
        if self.training_horizon_sessions is not None and (
            self.training_horizon_sessions < self.min_train_sessions
        ):
            raise ValueError("training horizon is below min_train_sessions")
        if self.class_weight not in {None, "balanced"}:
            raise ValueError("class_weight must be null or balanced")
        if self.min_train_sessions < 20:
            raise ValueError("min_train_sessions must be at least 20")
        try:
            parsed_train_start = pd.Timestamp(self.train_start)
        except (TypeError, ValueError) as exc:
            raise ValueError("train_start must be a valid date") from exc
        if pd.isna(parsed_train_start):
            raise ValueError("train_start must be a valid date")
        if self.legacy_estimator_class_weight and (
            self.feature_block != "legacy_v03"
            or self.objective != "positive_session"
            or self.class_weight != "balanced"
        ):
            raise ValueError(
                "legacy estimator weighting is reserved for the exact v0.3 control"
            )

    @property
    def id(self) -> str:
        horizon = self.training_horizon_sessions or "expanding"
        class_token = self.class_weight or "none"
        weighting = "legacycw" if self.legacy_estimator_class_weight else "datecw"
        train_token = pd.Timestamp(self.train_start).strftime("%Y%m%d")
        return (
            f"{self.feature_block}__{self.objective}__C{self.c:g}__"
            f"h{horizon}__cw{class_token}__{weighting}__"
            f"ts{train_token}__min{self.min_train_sessions}"
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelSpec":
        return cls(**raw)


@dataclass
class EvaluationResult:
    spec: ModelSpec
    summary: dict[str, Any]
    picks: pd.DataFrame
    daily_20bp: pd.Series
    daily_40bp: pd.Series


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"content hash is missing required columns: {missing}")
    values = pd.util.hash_pandas_object(frame[list(columns)], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _runtime_versions() -> dict[str, str]:
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        raise RuntimeError("pdftotext is required for the sealed JPX parser")
    try:
        completed = subprocess.run(
            [pdftotext, "-v"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot identify the pdftotext runtime") from exc
    version_output = (completed.stderr or completed.stdout).splitlines()
    if not version_output:
        raise RuntimeError("pdftotext did not report a version")
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "pdftotext": version_output[0].strip(),
        "pdftotext_sha256": _sha256_file(pdftotext),
    }


def _sorted_file_hashes(paths: Sequence[str | Path]) -> list[str]:
    return sorted(_sha256_file(path) for path in paths)


def _cross_section_rank(
    frame: pd.DataFrame, values: pd.Series, universe: pd.Series
) -> pd.Series:
    source = values.where(universe & np.isfinite(values))
    ranks = source.groupby(frame["date"], sort=False).rank(method="average")
    counts = source.groupby(frame["date"], sort=False).transform("count")
    output = 2.0 * (ranks - 1.0) / (counts - 1.0) - 1.0
    output = output.where(counts.gt(1), 0.0)
    return output.where(source.notna())


def add_bounded_daily_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    discontinuity = pd.to_numeric(
        frame["overnight"], errors="coerce"
    ).abs().gt(30.0)
    prior_discontinuity = discontinuity.groupby(
        frame["code"], sort=False
    ).shift(1)
    recent_discontinuity = (
        prior_discontinuity.groupby(frame["code"], sort=False)
        .rolling(60, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .sort_index()
        .fillna(0.0)
        .astype(bool)
    )
    frame["price_history_continuous_60"] = ~recent_discontinuity
    frame["eligible"] &= frame["price_history_continuous_60"]
    frame["training_eligible"] &= frame["price_history_continuous_60"]

    group = frame.groupby("code", sort=False)
    effective_close = group["close"].ffill()
    prior_effective_close = effective_close.groupby(
        frame["code"], sort=False
    ).shift(1)
    for window in (5, 20, 60):
        older = prior_effective_close.groupby(
            frame["code"], sort=False
        ).shift(window)
        frame[f"close_momentum_{window}"] = 100.0 * (
            prior_effective_close / older - 1.0
        )
    prior_high_20 = (
        frame["high"]
        .groupby(frame["code"], sort=False)
        .shift(1)
        .groupby(frame["code"], sort=False)
        .rolling(20, min_periods=10)
        .max()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    prior_low_20 = (
        frame["low"]
        .groupby(frame["code"], sort=False)
        .shift(1)
        .groupby(frame["code"], sort=False)
        .rolling(20, min_periods=10)
        .min()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    span = (prior_high_20 - prior_low_20).where(prior_high_20.gt(prior_low_20))
    frame["prior_close_location_20"] = (
        (prior_effective_close - prior_low_20) / span
    ).clip(0.0, 1.0)

    universe = frame["training_eligible"].fillna(False).astype(bool)
    for source in DAILY_RANK_SOURCES:
        frame[f"xrank_{source}"] = _cross_section_rank(
            frame, pd.to_numeric(frame[source], errors="coerce"), universe
        )
    safe_atr = frame["atr14_pct"].where(frame["atr14_pct"].gt(0))
    ratios = {
        "prior_oc": frame["oc_last"] / safe_atr,
        "prior_overnight": frame["overnight_last"] / safe_atr,
    }
    for prefix, raw in ratios.items():
        clipped = raw.clip(-5.0, 5.0)
        frame[f"{prefix}_atr_clipped"] = clipped
        frame[f"{prefix}_atr_abs"] = clipped.abs()
    frame[list(BOUNDED_DAILY_COLUMNS)] = frame[
        list(BOUNDED_DAILY_COLUMNS)
    ].astype("float32")
    return frame


def attach_tdnet_features(
    panel: pd.DataFrame,
    disclosures: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    complete_calendar_dates: pd.DatetimeIndex,
    *,
    decision_time: str = "08:58:59",
) -> pd.DataFrame:
    aggregate = build_tdnet_features(
        disclosures,
        sessions,
        decision_time=decision_time,
    )
    frame = panel.merge(
        aggregate,
        on=["date", "code"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    frame[list(TDNET_FEATURE_COLUMNS)] = frame[
        list(TDNET_FEATURE_COLUMNS)
    ].astype("float32")
    complete = set(pd.DatetimeIndex(complete_calendar_dates).normalize())
    target_complete: dict[pd.Timestamp, bool] = {sessions[0]: False}
    for previous, target in zip(sessions[:-1], sessions[1:], strict=True):
        required = pd.date_range(previous.normalize(), target.normalize(), freq="D")
        target_complete[target] = all(value in complete for value in required)
    frame["tdnet_source_complete"] = frame["date"].map(target_complete).eq(True)
    complete_rows = frame["tdnet_source_complete"]
    frame.loc[complete_rows, list(TDNET_FEATURE_COLUMNS)] = frame.loc[
        complete_rows, list(TDNET_FEATURE_COLUMNS)
    ].fillna(0.0)
    frame.loc[~complete_rows, list(TDNET_FEATURE_COLUMNS)] = np.nan
    frame["eligible"] &= complete_rows
    frame["training_eligible"] &= complete_rows
    frame["tdnet_revision_x_xrank_oc_mean_20"] = (
        frame["tdnet_has_revision"] * frame["xrank_oc_mean_20"]
    )
    frame["tdnet_equity_financing_x_xrank_oc_mean_20"] = (
        frame["tdnet_has_equity_financing"] * frame["xrank_oc_mean_20"]
    )
    frame["tdnet_any_x_market_beta_x_prior_market_return"] = (
        frame["tdnet_any"] * frame["market_beta_x_prior_market_return"]
    )
    interaction_columns = (
        *TDNET_RUNUP_INTERACTION_COLUMNS,
        *TDNET_MARKET_INTERACTION_COLUMNS,
    )
    frame[list(interaction_columns)] = frame[list(interaction_columns)].astype(
        "float32"
    )
    return frame


def build_research_panel(
    prices: pd.DataFrame,
    disclosures: pd.DataFrame,
    sessions: Iterable[object],
    tdnet_complete_calendar_dates: Iterable[object],
    config: RankerConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = config or RankerConfig()
    calendar = normalize_expected_sessions(sessions)
    canonical = normalize_daily_prices(prices)
    modeling, coverage = prepare_modeling_prices(
        canonical,
        coverage_lookback=settings.source_coverage_lookback,
        minimum_source_coverage=settings.minimum_source_coverage,
        expected_sessions=calendar,
    )
    panel = build_feature_panel(modeling, settings)
    # Apply the prior-session discontinuity hygiene before any cross-sectional
    # ranks are formed so an already-known contaminated history cannot alter
    # the rank of otherwise eligible securities.
    panel = add_bounded_daily_features(panel)
    panel = add_session_market_features(panel)
    panel = attach_tdnet_features(
        panel,
        disclosures,
        calendar,
        normalize_expected_sessions(tdnet_complete_calendar_dates),
        decision_time=settings.preopen.decision_time,
    )
    coverage["tdnet_source_complete"] = coverage["date"].map(
        panel.groupby("date", sort=False)["tdnet_source_complete"].first()
    ).eq(True)
    coverage["model_source_complete"] = (
        coverage["source_complete"] & coverage["tdnet_source_complete"]
    )
    required = sorted(
        set(column for columns in feature_blocks().values() for column in columns)
        - set(panel.columns)
    )
    if required:
        raise RuntimeError(f"research panel is missing frozen features: {required}")
    max_features = max(len(columns) for columns in feature_blocks().values())
    if max_features > 64:
        raise RuntimeError(f"feature cap exceeded: {max_features} > 64")
    return panel, coverage


def objective_and_weights(
    training: pd.DataFrame,
    objective: str,
    class_weight: str | None,
) -> tuple[pd.Series, pd.Series]:
    """Build a frozen binary target and equal-total-per-date weights."""

    returns = pd.to_numeric(training["oc_return_pct"], errors="coerce")
    if returns.isna().any():
        raise ValueError("training outcomes contain missing returns")
    if objective == "positive_session":
        target = returns.gt(0.0).astype(int)
        magnitude = pd.Series(1.0, index=training.index)
    elif objective == "net_positive_20bp":
        target = returns.gt(PRIMARY_COST_BPS / 100.0).astype(int)
        magnitude = pd.Series(1.0, index=training.index)
    elif objective == "net_positive_magnitude_weighted":
        threshold = PRIMARY_COST_BPS / 100.0
        target = returns.gt(threshold).astype(int)
        magnitude = (returns - threshold).abs().clip(0.25, 3.0)
    elif objective == "same_day_top_quintile":
        ranks = returns.groupby(training["date"], sort=False).rank(
            method="average", pct=True
        )
        target = ranks.gt(0.80).astype(int)
        magnitude = pd.Series(1.0, index=training.index)
    else:
        raise ValueError(f"unknown objective: {objective}")
    if target.nunique() != 2:
        raise ValueError("training target must contain both classes")
    weights = magnitude.astype(float)
    if class_weight == "balanced":
        # Solve the two margins on date-level sufficient statistics, then map
        # the single positive-class multiplier back to rows. This is equivalent
        # to two-margin raking but avoids repeated group-bys over millions of
        # security rows.
        desired_class_total = training["date"].nunique() / 2.0
        sufficient = pd.DataFrame(
            {
                "date": training["date"].to_numpy(),
                "target": target.to_numpy(),
                "weight": weights.to_numpy(),
            }
        ).pivot_table(
            index="date",
            columns="target",
            values="weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        negative = sufficient.get(0, pd.Series(0.0, index=sufficient.index)).to_numpy()
        positive = sufficient.get(1, pd.Series(0.0, index=sufficient.index)).to_numpy()

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
        if not (
            positive_total(lower) <= desired_class_total
            <= positive_total(upper)
        ):
            raise ValueError(
                "date-equal/class-balanced sample-weight margins are infeasible"
            )
        for _ in range(100):
            midpoint = (lower + upper) / 2.0
            if positive_total(midpoint) < desired_class_total:
                lower = midpoint
            else:
                upper = midpoint
        multiplier = float(np.exp((lower + upper) / 2.0))
        weights *= np.where(target.eq(1), multiplier, 1.0)
    elif class_weight is not None:
        raise ValueError("class_weight must be null or balanced")
    date_totals = weights.groupby(training["date"], sort=False).transform("sum")
    weights = weights / date_totals
    per_date = weights.groupby(training["date"], sort=False).sum()
    if not np.allclose(per_date.to_numpy(), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("sample weights are not normalized by date")
    if class_weight == "balanced":
        expected = training["date"].nunique() / 2.0
        per_class = weights.groupby(target, sort=False).sum().reindex([0, 1])
        if not np.allclose(
            per_class.to_numpy(), expected, atol=1e-8, rtol=0.0
        ):
            raise AssertionError("sample weights are not class-balanced")
    return target, weights


def make_estimator(spec: ModelSpec) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=spec.c,
                    class_weight=(
                        spec.class_weight
                        if spec.legacy_estimator_class_weight
                        else None
                    ),
                    max_iter=1_000,
                    random_state=DEFAULT_SEED,
                    solver="lbfgs",
                    l1_ratio=0.0,
                ),
            ),
        ]
    )


def _periods(start: pd.Timestamp, end: pd.Timestamp, frequency: str) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    values: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for period in pd.period_range(start.to_period(frequency), end.to_period(frequency), freq=frequency):
        left = max(start, period.start_time.normalize())
        right = min(end, period.end_time.normalize())
        values.append((left, right, str(period)))
    return values


def evaluate_spec(
    panel: pd.DataFrame,
    spec: ModelSpec,
    *,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    evaluation_sessions: pd.DatetimeIndex,
    retrain_frequency: str,
    bootstrap_samples: int,
) -> EvaluationResult:
    columns = list(feature_blocks()[spec.feature_block])
    train_start = pd.Timestamp(spec.train_start).normalize()
    pick_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    scheduled_days = evaluation_sessions[
        (evaluation_sessions >= evaluation_start)
        & (evaluation_sessions <= evaluation_end)
    ]
    if scheduled_days.empty:
        raise ValueError("evaluation interval has no scheduled exchange sessions")
    training_columns = list(
        dict.fromkeys(["date", "code", "oc_return_pct", *columns])
    )
    scoring_columns = [
        column
        for column in dict.fromkeys(
            [
                "date",
                "code",
                "name",
                "eligible",
                "label",
                "oc_return_pct",
                "outcome_observed",
                "source_complete",
                "universe_source_complete",
                *columns,
            ]
        )
        if column in panel
    ]
    for score_start, score_end, period_name in _periods(
        evaluation_start, evaluation_end, retrain_frequency
    ):
        training_mask = (
            panel["date"].between(
                train_start, score_start - pd.Timedelta(days=1)
            )
            & panel["training_eligible"]
            & panel["label"].notna()
        )
        training = panel.loc[training_mask, training_columns]
        training_dates = pd.DatetimeIndex(sorted(training["date"].unique()))
        if len(training_dates) < spec.min_train_sessions:
            raise ValueError(
                f"{spec.id} has only {len(training_dates)} training sessions "
                f"before {score_start.date()}"
            )
        if spec.training_horizon_sessions is not None:
            keep_dates = training_dates[-spec.training_horizon_sessions :]
            training = training[training["date"].isin(keep_dates)]
        scoring_mask = (
            panel["date"].between(score_start, score_end)
            & panel["eligible"]
        )
        scoring = panel.loc[scoring_mask, scoring_columns]
        if not scoring.empty and not training["date"].max() < scoring["date"].min():
            raise AssertionError("training and scoring periods overlap")
        target, weights = objective_and_weights(
            training,
            spec.objective,
            None if spec.legacy_estimator_class_weight else spec.class_weight,
        )
        estimator = make_estimator(spec)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            estimator.fit(
                training[columns],
                target,
                model__sample_weight=weights,
            )
        if not scoring.empty:
            scores = estimator.predict_proba(scoring[columns])[:, 1]
            ranked = scoring.assign(model_score=scores).sort_values(
                ["date", "model_score", "code"],
                ascending=[True, False, True],
                kind="stable",
            )
            ranked = ranked.groupby("date", sort=True, as_index=False).head(2).copy()
            ranked["model_rank"] = ranked.groupby("date", sort=False).cumcount() + 1
            ranked["spec_id"] = spec.id
            pick_parts.append(ranked)
        fold_rows.append(
            {
                "period": period_name,
                "train_start": str(training["date"].min().date()),
                "train_end": str(training["date"].max().date()),
                "train_days": int(training["date"].nunique()),
                "train_rows": int(len(training)),
                "score_start": str(score_start.date()),
                "score_end": str(score_end.date()),
                "scheduled_score_days": int(
                    ((scheduled_days >= score_start) & (scheduled_days <= score_end)).sum()
                ),
                "score_days": int(scoring["date"].nunique()),
                "score_rows": int(len(scoring)),
            }
        )
    actual = (
        pd.concat(pick_parts, ignore_index=True)
        if pick_parts
        else pd.DataFrame(
            columns=[
                *scoring_columns,
                "model_score",
                "model_rank",
                "spec_id",
            ]
        )
    )
    desired = pd.MultiIndex.from_product(
        [scheduled_days, (1, 2)], names=["date", "model_rank"]
    ).to_frame(index=False)
    picks = desired.merge(
        actual,
        on=["date", "model_rank"],
        how="left",
        validate="one_to_one",
        sort=True,
    )
    picks["spec_id"] = spec.id
    top1 = picks[picks["model_rank"].eq(1)].copy()
    daily20 = daily_portfolio_returns(
        top1, top_k=1, cost_bps=PRIMARY_COST_BPS
    ).set_index("date")["net_return_pct"].sort_index()
    daily40 = daily_portfolio_returns(
        top1, top_k=1, cost_bps=STRESS_COST_BPS
    ).set_index("date")["net_return_pct"].sort_index()
    diagnostics = return_diagnostics(
        daily20,
        top_k=5,
        period_frequency="Q",
    )
    bootstrap = moving_block_bootstrap_suite(
        daily20,
        block_lengths=(5, 10, 20),
        samples=bootstrap_samples,
        confidence=0.90,
        random_state=DEFAULT_SEED,
    )
    summary = {
        "spec": asdict(spec),
        "spec_id": spec.id,
        "feature_count": len(columns),
        "features": columns,
        "evaluation_start": str(evaluation_start.date()),
        "evaluation_end": str(evaluation_end.date()),
        "retrain_frequency": retrain_frequency,
        "folds": fold_rows,
        "valid_days": int(len(scheduled_days)),
        "display_days": int(top1["code"].notna().sum()),
        "candidate_display_rate": float(
            top1["code"].notna().sum() / len(scheduled_days)
        ),
        "top1_20bp": profit_metrics(
            picks, top_k=1, cost_bps=PRIMARY_COST_BPS
        ),
        "top2_20bp": profit_metrics(
            picks, top_k=2, cost_bps=PRIMARY_COST_BPS
        ),
        "top1_diagnostics_quarterly": asdict(diagnostics),
        "top1_bootstrap": {
            str(length): asdict(interval)
            for length, interval in bootstrap.items()
        },
        "top1_40bp_mean_pct": float(daily40.mean()),
    }
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
            "spec_id",
        )
        if column in picks
    ]
    return EvaluationResult(spec, summary, picks[keep], daily20, daily40)


def _selection_key(result: EvaluationResult) -> tuple[float, float, int, str]:
    lcb = result.summary["top1_bootstrap"]["5"]["one_sided_lower_pct"]
    mean = result.summary["top1_20bp"]["net_mean_pct_at_cost"]
    return (-float(lcb), -float(mean), int(result.summary["feature_count"]), result.spec.id)


def _evaluate_many(
    panel: pd.DataFrame,
    specs: Sequence[ModelSpec],
    *,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    evaluation_sessions: pd.DatetimeIndex,
    retrain_frequency: str,
    bootstrap_samples: int,
    cache: dict[tuple[str, str, str, str], EvaluationResult],
) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    for position, spec in enumerate(specs, start=1):
        key = (
            spec.id,
            str(evaluation_start.date()),
            str(evaluation_end.date()),
            retrain_frequency,
        )
        if key not in cache:
            print(
                f"evaluate {position}/{len(specs)} {spec.id} "
                f"{evaluation_start.date()}..{evaluation_end.date()} {retrain_frequency}",
                flush=True,
            )
            cache[key] = evaluate_spec(
                panel,
                spec,
                evaluation_start=evaluation_start,
                evaluation_end=evaluation_end,
                evaluation_sessions=evaluation_sessions,
                retrain_frequency=retrain_frequency,
                bootstrap_samples=bootstrap_samples,
            )
        results.append(cache[key])
    return results


def _ranking_changed_days(
    candidate: EvaluationResult, baseline: EvaluationResult
) -> int:
    left = candidate.picks[candidate.picks["model_rank"].eq(1)][
        ["date", "code"]
    ].rename(columns={"code": "candidate"})
    right = baseline.picks[baseline.picks["model_rank"].eq(1)][
        ["date", "code"]
    ].rename(columns={"code": "baseline"})
    if not pd.Index(left["date"]).equals(pd.Index(right["date"])):
        raise ValueError("ranking comparison requires identical scheduled dates")
    paired = left.merge(right, on="date", validate="one_to_one")
    return int(paired["candidate"].ne(paired["baseline"]).sum())


def _development_gate(
    result: EvaluationResult,
    comparison: EvaluationResult,
    baseline: EvaluationResult,
) -> dict[str, Any]:
    diagnostic = result.summary["top1_diagnostics_quarterly"]
    changed = _ranking_changed_days(comparison, baseline)
    checks = {
        "net_20bp_mean_gt_0": diagnostic["mean_pct"] > 0.0,
        "top5_removed_mean_gt_0": diagnostic["top_k_removed_mean_pct"] > 0.0,
        "positive_quarter_ratio_gte_0_6": diagnostic[
            "positive_period_ratio"
        ]
        >= 0.60,
        "profit_factor_gt_1": diagnostic["profit_factor"] > 1.0,
        "max_positive_contribution_lte_0_25": diagnostic[
            "max_positive_contribution"
        ]
        <= 0.25,
        "ranking_changed_days_vs_v03_gte_30": changed >= 30,
        "observations_gte_252": diagnostic["observations"] >= 252,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "ranking_changed_days_vs_v03": changed,
    }


def _paired_report(
    candidate: EvaluationResult,
    baseline: EvaluationResult,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if not candidate.daily_20bp.index.equals(baseline.daily_20bp.index):
        raise ValueError("paired comparison requires identical scheduled dates")
    left = candidate.daily_20bp
    right = baseline.daily_20bp
    intervals = paired_moving_block_bootstrap_suite(
        left,
        right,
        block_lengths=(5, 10, 20),
        samples=bootstrap_samples,
        confidence=0.90,
        random_state=DEFAULT_SEED,
    )
    return {
        "days": int(len(left)),
        "mean_delta_pct": float((left - right).mean()),
        "bootstrap": {
            str(length): asdict(interval)
            for length, interval in intervals.items()
        },
        "ranking_changed_days": _ranking_changed_days(candidate, baseline),
    }


def _white_reality_check(
    results: Sequence[EvaluationResult],
    *,
    samples: int,
    block_length: int = 5,
    random_state: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Shared-block max-mean diagnostic across the frozen finalists.

    Each candidate series is centred under its zero-mean null.  Using the same
    sampled blocks for every column preserves contemporaneous dependence and
    estimates the distribution of the best result after searching this finite
    registry.  The sealed holdout remains the actual confirmation test.
    """

    if len(results) < 2:
        raise ValueError("reality check requires at least two candidates")
    common = results[0].daily_20bp.index
    for result in results[1:]:
        if not common.equals(result.daily_20bp.index):
            raise ValueError(
                "reality check requires identical scheduled dates"
            )
    matrix = np.column_stack(
        [result.daily_20bp.reindex(common).to_numpy() for result in results]
    )
    if len(matrix) < block_length:
        raise ValueError("reality-check block exceeds observations")
    observed_means = matrix.mean(axis=0)
    observed_max = float(observed_means.max())
    centered = matrix - observed_means
    block_count = int(np.ceil(len(matrix) / block_length))
    offsets = np.arange(block_length)
    max_start = len(matrix) - block_length + 1
    rng = np.random.default_rng(random_state)
    maxima = np.empty(samples, dtype=float)
    for start in range(0, samples, 1_000):
        size = min(1_000, samples - start)
        starts = rng.integers(0, max_start, size=(size, block_count))
        indices = (starts[..., None] + offsets).reshape(size, -1)[:, : len(matrix)]
        boot_means = centered[indices].mean(axis=1)
        maxima[start : start + size] = boot_means.max(axis=1)
    p_value = float((1 + np.count_nonzero(maxima >= observed_max)) / (samples + 1))
    return {
        "candidate_ids": [result.spec.id for result in results],
        "days": int(len(common)),
        "block_length": block_length,
        "samples": samples,
        "random_state": random_state,
        "candidate_mean_pct": {
            result.spec.id: float(mean)
            for result, mean in zip(results, observed_means, strict=True)
        },
        "observed_max_mean_pct": observed_max,
        "p_value": p_value,
        "interpretation": (
            "development multiple-testing diagnostic only; the one-time sealed "
            "holdout determines confirmation"
        ),
    }


def _summary_rows(results: Sequence[EvaluationResult]) -> list[dict[str, Any]]:
    return [result.summary for result in sorted(results, key=_selection_key)]


def _manifest_complete_dates(manifest: dict[str, Any]) -> pd.DatetimeIndex:
    if "complete_dates" in manifest:
        dates = normalize_expected_sessions(manifest["complete_dates"])
    else:
        start = manifest.get("complete_dates_from", manifest.get("min_index_date"))
        end = manifest.get("complete_dates_through", manifest.get("max_index_date"))
        if start is None or end is None:
            raise ValueError("TDnet manifest has no complete calendar interval")
        dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
        claimed = manifest.get(
            "complete_calendar_dates", manifest.get("source_files")
        )
        if claimed is None or int(claimed) != len(dates):
            raise ValueError("TDnet manifest calendar coverage is inconsistent")
    if manifest.get("all_used_requests_count_match") is False:
        raise ValueError("TDnet manifest contains a count-mismatched request")
    if manifest.get("all_used_requests_below_limit") is False:
        raise ValueError("TDnet manifest contains a truncated request")
    return dates


def _verify_tdnet_api_manifest(
    manifest: dict[str, Any],
    export_hashes: set[str],
) -> None:
    if manifest.get("source") != "yanoshin_tdnet_list_api":
        raise ValueError("formal research requires the complete TDnet list API source")
    if manifest.get("output_sha256") not in export_hashes:
        raise ValueError("TDnet manifest does not bind an input export")
    if manifest.get("all_used_requests_count_match") is not True:
        raise ValueError("TDnet manifest has an unverified item count")
    if manifest.get("all_used_requests_below_limit") is not True:
        raise ValueError("TDnet manifest has a truncated used request")
    if int(manifest.get("rows", -1)) != int(manifest.get("ids", -2)):
        raise ValueError("TDnet canonical IDs are not unique")
    raw_manifest_path = Path(str(manifest.get("raw_manifest_path", "")))
    if (
        not raw_manifest_path.is_file()
        or _sha256_file(raw_manifest_path) != manifest.get("raw_manifest_sha256")
    ):
        raise ValueError("TDnet raw manifest hash changed")
    queries = manifest.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("TDnet manifest has no raw query provenance")
    for query in queries:
        source = Path(str(query.get("path", "")))
        if not source.is_file():
            raise ValueError("TDnet raw response is missing")
        if source.stat().st_size != int(query.get("bytes", -1)):
            raise ValueError("TDnet raw response byte count changed")
        if _sha256_file(source) != query.get("sha256"):
            raise ValueError("TDnet raw response hash changed")
        if query.get("used_in_canonical") and (
            query.get("count_matches") is not True
            or query.get("limit_reached") is True
        ):
            raise ValueError("TDnet canonical input is incomplete")


def _load_tdnet(
    paths: Sequence[str | Path],
    manifest_paths: Sequence[str | Path],
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    frames = [read_frame(path) for path in paths]
    disclosures = normalize_tdnet_disclosures(
        pd.concat(frames, ignore_index=True, sort=False)
    )
    complete_dates = _tdnet_manifest_coverage(paths, manifest_paths)
    return disclosures, complete_dates


def _tdnet_manifest_coverage(
    paths: Sequence[str | Path],
    manifest_paths: Sequence[str | Path],
) -> pd.DatetimeIndex:
    file_hashes = {_sha256_file(path) for path in paths}
    complete_parts: list[pd.DatetimeIndex] = []
    for path in manifest_paths:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        _verify_tdnet_api_manifest(manifest, file_hashes)
        complete_parts.append(_manifest_complete_dates(manifest))
    if not complete_parts:
        raise ValueError("at least one TDnet completeness manifest is required")
    complete_dates = pd.DatetimeIndex(
        sorted(set().union(*(set(values) for values in complete_parts)))
    )
    return complete_dates


def _load_calendar(path: str | Path, through: pd.Timestamp) -> pd.DatetimeIndex:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        values = pd.read_csv(source)["date"]
    else:
        values = [
            line.strip()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return normalize_expected_sessions(values, through=through)


def _verify_full_calendar_seal(
    path: str | Path,
    development_sessions: pd.DatetimeIndex,
    sealed_manifest: dict[str, Any],
) -> pd.DatetimeIndex:
    full = _load_calendar(path, HOLDOUT_END)
    prefix = full[full <= CONFIRMATION_END]
    if not prefix.equals(development_sessions):
        raise ValueError("full calendar development prefix changed")
    expected_holdout = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-08-01"),
            *(
                pd.Timestamp(Path(item["path"]).stem.removeprefix("stq_"))
                for item in sealed_manifest["files"]
            ),
        ]
    )
    suffix = full[(full > CONFIRMATION_END) & (full <= HOLDOUT_END)]
    if not suffix.equals(expected_holdout):
        raise ValueError("full calendar holdout suffix does not match sealed JPX")
    return full


def _verify_development_jpx_manifest(
    manifest_path: str | Path,
    daily_path: str | Path,
    prices: pd.DataFrame,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != 2:
        raise ValueError("development JPX manifest has the wrong schema")
    if manifest.get("parser_version") != JPX_PARSER_VERSION:
        raise ValueError("development JPX parser version changed")
    if manifest.get("export_sha256") != _sha256_file(daily_path):
        raise ValueError("development JPX manifest does not bind its export")
    if int(manifest.get("rows", -1)) != len(prices):
        raise ValueError("development JPX manifest row count changed")
    if manifest.get("min_date") != str(prices["date"].min().date()) or manifest.get(
        "max_date"
    ) != str(prices["date"].max().date()):
        raise ValueError("development JPX manifest date range changed")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("development JPX manifest has no source inputs")
    for item in inputs:
        source = Path(item["path"])
        if not source.is_file() or _sha256_file(source) != item.get("sha256"):
            raise ValueError("development JPX source hash changed")
        if int(item.get("rejected_rows", -1)) != 0:
            raise ValueError("development JPX parser rejected source rows")
    return manifest


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        PROTOCOL_PATH,
        root / "src/tse_session_ranker/api.py",
        root / "src/tse_session_ranker/config.py",
        root / "src/tse_session_ranker/data/common.py",
        root / "src/tse_session_ranker/data/jpx.py",
        root / "src/tse_session_ranker/data/market_context.py",
        root / "src/tse_session_ranker/data/tdnet.py",
        root / "src/tse_session_ranker/features.py",
        root / "src/tse_session_ranker/io.py",
        root / "src/tse_session_ranker/profit.py",
        root / "src/tse_session_ranker/research_features.py",
        root / "src/tse_session_ranker/validation.py",
    )
    return {str(path.relative_to(root)): _sha256_file(path) for path in paths}


def select_phase(args: argparse.Namespace) -> None:
    daily_path = Path(args.development_daily).resolve()
    calendar_path = Path(args.development_calendar).resolve()
    tdnet_paths = [Path(value).resolve() for value in args.development_tdnet]
    tdnet_manifest_paths = [
        Path(value).resolve() for value in args.development_tdnet_manifest
    ]
    holdout_tdnet_paths = [
        Path(value).resolve() for value in args.holdout_tdnet_seal
    ]
    holdout_tdnet_manifest_paths = [
        Path(value).resolve() for value in args.holdout_tdnet_manifest_seal
    ]
    output_path = Path(args.output).resolve()
    output_companions = (
        output_path,
        output_path.with_name(output_path.stem + "_winner_picks.csv"),
        output_path.with_name(output_path.stem + "_v03_control_picks.csv"),
    )
    if any(path.exists() for path in output_companions):
        raise FileExistsError("selection lock or companion output already exists")
    prices = normalize_daily_prices(read_frame(daily_path))
    development_jpx_manifest = _verify_development_jpx_manifest(
        args.development_daily_manifest, daily_path, prices
    )
    if prices["date"].max() > CONFIRMATION_END:
        raise ValueError("selection input contains dates after development cutoff")
    sessions = _load_calendar(calendar_path, CONFIRMATION_END)
    sealed_manifest = _verify_sealed_manifest(args.sealed_jpx_manifest)
    full_sessions = _verify_full_calendar_seal(
        args.full_calendar_seal, sessions, sealed_manifest
    )
    disclosures, tdnet_complete_dates = _load_tdnet(
        tdnet_paths, tdnet_manifest_paths
    )
    holdout_tdnet_complete_dates = _tdnet_manifest_coverage(
        holdout_tdnet_paths, holdout_tdnet_manifest_paths
    )
    all_tdnet_complete = set(tdnet_complete_dates) | set(
        holdout_tdnet_complete_dates
    )
    required_tdnet_calendar = pd.date_range(
        CONFIRMATION_END, HOLDOUT_END, freq="D"
    )
    if not all(value in all_tdnet_complete for value in required_tdnet_calendar):
        raise ValueError("sealed TDnet inputs do not cover the full holdout interval")
    if disclosures["published_at"].dt.tz_localize(None).max().normalize() > CONFIRMATION_END:
        disclosures = disclosures[
            disclosures["published_at"].dt.tz_localize(None).dt.normalize().le(
                CONFIRMATION_END
            )
        ].copy()
    print("build development panel", flush=True)
    panel, coverage = build_research_panel(
        prices, disclosures, sessions, tdnet_complete_dates
    )
    cache: dict[tuple[str, str, str, str], EvaluationResult] = {}

    stage1_specs = [
        ModelSpec(block, objective)
        for block in feature_blocks()
        for objective in OBJECTIVES
    ]
    stage1 = _evaluate_many(
        panel,
        stage1_specs,
        evaluation_start=FEATURE_SCREEN_START,
        evaluation_end=FEATURE_SCREEN_END,
        evaluation_sessions=sessions,
        retrain_frequency="Q",
        bootstrap_samples=args.bootstrap_samples,
        cache=cache,
    )
    retained_stage1_pairs = []
    for block in feature_blocks():
        block_results = [
            result for result in stage1 if result.spec.feature_block == block
        ]
        best = sorted(block_results, key=_selection_key)[0]
        retained_stage1_pairs.append(
            (best.spec.feature_block, best.spec.objective)
        )

    stage2_specs = [
        ModelSpec(block, objective)
        for block, objective in retained_stage1_pairs
    ]
    stage2 = _evaluate_many(
        panel,
        stage2_specs,
        evaluation_start=MODEL_DESIGN_START,
        evaluation_end=MODEL_DESIGN_END,
        evaluation_sessions=sessions,
        retrain_frequency="Q",
        bootstrap_samples=args.bootstrap_samples,
        cache=cache,
    )
    retained_pairs = [
        (result.spec.feature_block, result.spec.objective)
        for result in sorted(stage2, key=_selection_key)[:3]
    ]

    stage3_specs = [
        ModelSpec(
            block,
            objective,
            c=c,
            training_horizon_sessions=horizon,
            class_weight=class_weight,
        )
        for block, objective in retained_pairs
        for c in (0.03, 0.08, 0.20)
        for horizon in (504, None)
        for class_weight in (None, "balanced")
    ]
    stage3 = _evaluate_many(
        panel,
        stage3_specs,
        evaluation_start=MODEL_DESIGN_START,
        evaluation_end=MODEL_DESIGN_END,
        evaluation_sessions=sessions,
        retrain_frequency="Q",
        bootstrap_samples=args.bootstrap_samples,
        cache=cache,
    )
    finalists = [result.spec for result in sorted(stage3, key=_selection_key)[:3]]

    baseline_spec = ModelSpec(
        "legacy_v03",
        "positive_session",
        c=0.08,
        training_horizon_sessions=None,
        class_weight="balanced",
        train_start=str(LEGACY_REGIME_START.date()),
        min_train_sessions=20,
        legacy_estimator_class_weight=True,
    )
    confirmation = _evaluate_many(
        panel,
        finalists,
        evaluation_start=CONFIRMATION_START,
        evaluation_end=CONFIRMATION_END,
        evaluation_sessions=sessions,
        retrain_frequency="M",
        bootstrap_samples=args.bootstrap_samples,
        cache=cache,
    )
    comparison = _evaluate_many(
        panel,
        [*finalists, baseline_spec],
        evaluation_start=LEGACY_COMPARISON_START,
        evaluation_end=CONFIRMATION_END,
        evaluation_sessions=sessions,
        retrain_frequency="M",
        bootstrap_samples=args.bootstrap_samples,
        cache=cache,
    )
    by_id = {result.spec.id: result for result in confirmation}
    comparison_by_id = {result.spec.id: result for result in comparison}
    baseline = comparison_by_id[baseline_spec.id]
    finalist_results = [by_id[spec.id] for spec in finalists]
    gates = {
        result.spec.id: _development_gate(
            result, comparison_by_id[result.spec.id], baseline
        )
        for result in finalist_results
    }
    passers = [result for result in finalist_results if gates[result.spec.id]["passed"]]
    winner = sorted(passers or finalist_results, key=_selection_key)[0]
    development_status = (
        "development_qualified" if passers else "development_best_no_edge"
    )

    write_frame(
        winner.picks,
        output_path.with_name(output_path.stem + "_winner_picks.csv"),
    )
    write_frame(
        baseline.picks,
        output_path.with_name(output_path.stem + "_v03_control_picks.csv"),
    )
    payload = {
        "schema_version": 1,
        "phase": "selection_lock",
        "protocol_id": "session_v4_profit_logit_preregistered_20260721",
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "sealed_holdout_access_count": 0,
        "sealed_jpx_manifest_sha256": SEALED_JPX_MANIFEST_SHA256,
        "sealed_jpx_canonical_files_sha256": SEALED_JPX_CANONICAL_FILES_SHA256,
        "development_status": development_status,
        "winner": asdict(winner.spec),
        "winner_spec_id": winner.spec.id,
        "winner_development_gate": gates[winner.spec.id],
        "winner_vs_v03": _paired_report(
            comparison_by_id[winner.spec.id], baseline, args.bootstrap_samples
        ),
        "feature_blocks": {
            name: list(columns) for name, columns in feature_blocks().items()
        },
        "stage_1_feature_screen": _summary_rows(stage1),
        "stage_1_retained_pairs": [
            list(value) for value in retained_stage1_pairs
        ],
        "stage_2_objective_screen": _summary_rows(stage2),
        "stage_2_retained_pairs": [list(value) for value in retained_pairs],
        "stage_3_hyperparameter_screen": _summary_rows(stage3),
        "stage_3_finalists": [asdict(spec) for spec in finalists],
        "stage_4_confirmation": _summary_rows(confirmation),
        "stage_4_v03_comparison_start": str(
            LEGACY_COMPARISON_START.date()
        ),
        "stage_4_v03_comparison": _summary_rows(comparison),
        "stage_4_development_gates": gates,
        "stage_4_white_reality_check": _white_reality_check(
            finalist_results,
            samples=args.bootstrap_samples,
        ),
        "data": {
            "daily_path": str(daily_path),
            "daily_file_sha256": _sha256_file(daily_path),
            "daily_manifest_path": str(
                Path(args.development_daily_manifest).resolve()
            ),
            "daily_manifest_sha256": _sha256_file(
                args.development_daily_manifest
            ),
            "daily_parser_version": development_jpx_manifest.get(
                "parser_version"
            ),
            "daily_content_sha256": _content_hash(
                prices,
                (
                    "date",
                    "code",
                    "open",
                    "high",
                    "low",
                    "close",
                    *SESSION_OHLC,
                    "volume",
                    "turnover",
                    "traded",
                    "partial_session",
                ),
            ),
            "daily_rows": int(len(prices)),
            "daily_codes": int(prices["code"].nunique()),
            "daily_sessions": int(prices["date"].nunique()),
            "calendar_path": str(calendar_path),
            "calendar_file_sha256": _sha256_file(calendar_path),
            "calendar_sha256": session_calendar_hash(
                sessions, through=CONFIRMATION_END
            ),
            "full_calendar_seal_path": str(
                Path(args.full_calendar_seal).resolve()
            ),
            "full_calendar_seal_file_sha256": _sha256_file(
                args.full_calendar_seal
            ),
            "full_calendar_sha256": session_calendar_hash(
                full_sessions, through=HOLDOUT_END
            ),
            "tdnet_paths": [str(path) for path in tdnet_paths],
            "tdnet_file_sha256": {
                str(path): _sha256_file(path) for path in tdnet_paths
            },
            "tdnet_manifest_paths": [
                str(path) for path in tdnet_manifest_paths
            ],
            "tdnet_manifest_sha256": {
                str(path): _sha256_file(path) for path in tdnet_manifest_paths
            },
            "development_tdnet_file_hashes_sorted": _sorted_file_hashes(
                tdnet_paths
            ),
            "development_tdnet_manifest_hashes_sorted": _sorted_file_hashes(
                tdnet_manifest_paths
            ),
            "holdout_tdnet_paths_sealed": [
                str(path) for path in holdout_tdnet_paths
            ],
            "holdout_tdnet_manifest_paths_sealed": [
                str(path) for path in holdout_tdnet_manifest_paths
            ],
            "holdout_tdnet_file_hashes_sorted": _sorted_file_hashes(
                holdout_tdnet_paths
            ),
            "holdout_tdnet_manifest_hashes_sorted": _sorted_file_hashes(
                holdout_tdnet_manifest_paths
            ),
            "tdnet_content_sha256": _content_hash(
                disclosures,
                ("published_at", "code", "title", "url"),
            ),
            "tdnet_rows": int(len(disclosures)),
            "source_incomplete_dates": [
                str(value.date())
                for value in coverage.loc[
                    ~coverage["source_complete"], "date"
                ]
            ],
            "tdnet_source_incomplete_dates": [
                str(value.date())
                for value in coverage.loc[
                    ~coverage["tdnet_source_complete"], "date"
                ]
            ],
        },
        "implementation_sha256": _implementation_hashes(),
        "bootstrap_samples": args.bootstrap_samples,
        "runtime": _runtime_versions(),
    }
    write_json(payload, output_path)
    print(f"selection_lock={output_path}", flush=True)
    print(f"winner={winner.spec.id}", flush=True)
    print(f"development_status={development_status}", flush=True)


def _verify_sealed_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    if _sha256_file(manifest_path) != SEALED_JPX_MANIFEST_SHA256:
        raise ValueError("sealed JPX manifest bytes do not match the protocol")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != 2:
        raise ValueError("sealed JPX manifest has the wrong schema")
    if manifest.get("labels_opened") is not False:
        raise ValueError("sealed JPX manifest is not marked unopened")
    if int(manifest.get("file_count", -1)) != 159:
        raise ValueError("sealed JPX manifest must contain exactly 159 scored files")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 159:
        raise ValueError("sealed JPX file list is incomplete")
    names = [Path(item["path"]).name for item in files]
    if len(names) != len(set(names)):
        raise ValueError("sealed JPX manifest contains duplicate basenames")
    if names[0] != "stq_20250804.pdf" or names[-1] != "stq_20260331.pdf":
        raise ValueError("sealed JPX manifest has the wrong date boundaries")
    canonical: list[dict[str, Any]] = []
    for item, name in zip(files, names, strict=True):
        source = Path(item["path"])
        if not source.is_file():
            raise ValueError(f"sealed JPX source is missing: {name}")
        if Path(item["url"]).name != name or not str(item["url"]).startswith(
            "https://www.jpx.co.jp/"
        ):
            raise ValueError(f"sealed JPX URL is invalid: {name}")
        if source.stat().st_size != int(item["bytes"]):
            raise ValueError(f"sealed JPX byte count changed: {name}")
        if _sha256_file(source) != item["sha256"]:
            raise ValueError(f"sealed JPX file hash changed: {name}")
        canonical.append(
            {
                "name": name,
                "bytes": int(item["bytes"]),
                "sha256": item["sha256"],
            }
        )
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != SEALED_JPX_CANONICAL_FILES_SHA256 or manifest.get(
        "canonical_files_sha256"
    ) != digest:
        raise ValueError("sealed JPX canonical file digest changed")
    warmup = manifest_path.parent / HOLDOUT_WARMUP_JPX_NAME
    if (
        not warmup.is_file()
        or warmup.stat().st_size != HOLDOUT_WARMUP_JPX_BYTES
        or _sha256_file(warmup) != HOLDOUT_WARMUP_JPX_SHA256
    ):
        raise ValueError("holdout feature-warmup JPX file changed")
    return manifest


def _sealed_jpx_paths(
    sealed_manifest_path: str | Path,
    manifest: dict[str, Any],
) -> list[Path]:
    warmup = Path(sealed_manifest_path).resolve().parent / HOLDOUT_WARMUP_JPX_NAME
    return [warmup, *(Path(item["path"]) for item in manifest["files"])]


def _verify_parsed_jpx_manifest(
    parsed_manifest: dict[str, Any],
    sealed_paths: Sequence[Path],
    blind: pd.DataFrame,
) -> None:
    if parsed_manifest.get("parser_version") != JPX_PARSER_VERSION:
        raise ValueError("holdout JPX parser version changed")
    inputs = parsed_manifest.get("inputs", [])
    if not isinstance(inputs, list) or len(inputs) != len(sealed_paths):
        raise ValueError("parsed JPX input report count is incomplete")
    parsed = {Path(item["path"]).name: item for item in inputs}
    if len(parsed) != len(inputs):
        raise ValueError("parsed JPX input reports contain duplicate basenames")
    expected = {path.name: _sha256_file(path) for path in sealed_paths}
    if {name: item.get("sha256") for name, item in parsed.items()} != expected:
        raise ValueError("parsed JPX inputs do not match the sealed files")
    if "source_file" not in blind:
        raise ValueError("parsed JPX rows are missing source provenance")
    actual_rows = blind["source_file"].map(
        lambda value: Path(str(value)).stem
    ).value_counts()
    for name, item in parsed.items():
        try:
            ordinary = int(item["ordinary_rows"])
            parsed_rows = int(item["parsed_rows"])
            full = int(item["full_session_rows"])
            partial = int(item["partial_session_rows"])
            no_trade = int(item["no_trade_rows"])
            rejected = int(item["rejected_rows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("parsed JPX row accounting is incomplete") from exc
        if item.get("parser_version") != JPX_PARSER_VERSION:
            raise ValueError("parsed JPX per-input parser version changed")
        if item.get("source_format") != (
            "jpx_stock_quotations_auction_regular_way_domestic_ordinary"
        ):
            raise ValueError("parsed JPX holdout source format changed")
        if rejected != 0 or parsed_rows <= 0 or ordinary != parsed_rows:
            raise ValueError("parsed JPX holdout contains rejected or missing rows")
        if full + partial + no_trade != parsed_rows:
            raise ValueError("parsed JPX session row accounting is inconsistent")
        if int(actual_rows.get(Path(name).stem, 0)) != parsed_rows:
            raise ValueError("parsed JPX manifest does not match exported rows")


def _verify_blind_dates(
    blind: pd.DataFrame,
    sealed_manifest: dict[str, Any],
) -> None:
    expected = pd.DatetimeIndex(
        sorted(
            [
                pd.Timestamp(HOLDOUT_WARMUP_JPX_NAME.removeprefix("stq_").removesuffix(".pdf")),
                *(
                    pd.Timestamp(Path(item["path"]).stem.removeprefix("stq_"))
                    for item in sealed_manifest["files"]
                ),
            ]
        )
    )
    actual = pd.DatetimeIndex(sorted(blind["date"].unique()))
    if not actual.equals(expected):
        raise ValueError("holdout daily dates do not match the sealed JPX files")
    if "source_file" not in blind:
        raise ValueError("holdout daily export is missing source_file provenance")
    source_dates = pd.to_datetime(
        blind["source_file"].map(
            lambda value: Path(str(value)).stem.removeprefix("stq_")
        ),
        errors="coerce",
    )
    if source_dates.isna().any() or not source_dates.dt.normalize().equals(
        blind["date"].reset_index(drop=True)
    ):
        raise ValueError("holdout daily source_file provenance is inconsistent")


def _holdout_gate_report(
    candidate: EvaluationResult,
    baseline: EvaluationResult,
    bootstrap_samples: int,
) -> dict[str, Any]:
    diagnostic = return_diagnostics(
        candidate.daily_20bp,
        top_k=5,
        period_frequency="M",
    )
    bootstrap = moving_block_bootstrap_suite(
        candidate.daily_20bp,
        block_lengths=(5, 10, 20),
        samples=bootstrap_samples,
        confidence=0.90,
        random_state=DEFAULT_SEED,
    )
    paired = _paired_report(candidate, baseline, bootstrap_samples)
    checks = {
        "observations_gte_120": diagnostic.observations >= 120,
        "net_20bp_mean_gt_0": diagnostic.mean_pct > 0.0,
        "net_20bp_median_gt_0": diagnostic.median_pct > 0.0,
        "bootstrap_5_lcb_gt_0": bootstrap[5].one_sided_lower_pct > 0.0,
        "bootstrap_10_lcb_gt_0": bootstrap[10].one_sided_lower_pct > 0.0,
        "bootstrap_20_lcb_gt_0": bootstrap[20].one_sided_lower_pct > 0.0,
        "positive_month_ratio_gte_0_625": diagnostic.positive_period_ratio
        >= 0.625,
        "top5_removed_mean_gt_0": diagnostic.top_k_removed_mean_pct > 0.0,
        "profit_factor_gt_1": diagnostic.profit_factor > 1.0,
        "net_40bp_mean_gte_0": float(candidate.daily_40bp.mean()) >= 0.0,
        "max_positive_contribution_lte_0_25": diagnostic.max_positive_contribution
        <= 0.25,
        "paired_mean_delta_vs_v03_gt_0": paired["mean_delta_pct"] > 0.0,
        "candidate_display_rate_gte_0_95": candidate.summary[
            "candidate_display_rate"
        ]
        >= 0.95,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "diagnostics_monthly": asdict(diagnostic),
        "bootstrap": {
            str(length): asdict(interval)
            for length, interval in bootstrap.items()
        },
        "paired_vs_v03": paired,
    }


def holdout_phase(args: argparse.Namespace) -> None:
    selection_path = Path(args.selection_lock).resolve()
    selection_sha256 = _sha256_file(selection_path)
    if selection_sha256 != args.expected_selection_sha256:
        raise ValueError("selection lock does not match the externally frozen hash")
    output_path = Path(args.output).resolve()
    jpx_output_path = Path(args.holdout_jpx_output).resolve()
    jpx_manifest_path = jpx_output_path.with_suffix(
        jpx_output_path.suffix + ".manifest.json"
    )
    receipt_path = selection_path.with_name(
        selection_path.stem + "_holdout_consumed.json"
    )
    output_companions = (
        output_path,
        output_path.with_name(output_path.stem + "_winner_picks.csv"),
        output_path.with_name(output_path.stem + "_v03_control_picks.csv"),
        jpx_output_path,
        jpx_manifest_path,
        receipt_path,
    )
    if any(path.exists() for path in output_companions):
        raise FileExistsError("holdout output or consumed receipt already exists")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("phase") != "selection_lock":
        raise ValueError("selection lock has the wrong phase")
    if selection.get("sealed_holdout_access_count") != 0:
        raise ValueError("selection lock indicates prior holdout access")
    if selection.get("protocol_sha256") != _sha256_file(PROTOCOL_PATH):
        raise ValueError("protocol changed after the selection lock")
    if selection.get("implementation_sha256") != _implementation_hashes():
        raise ValueError("selection implementation changed before holdout")
    if selection.get("runtime") != _runtime_versions():
        raise ValueError("runtime changed after the selection lock")
    if selection.get("bootstrap_samples") != args.bootstrap_samples:
        raise ValueError("bootstrap sample count changed after selection")
    if _sha256_file(args.development_daily) != selection["data"][
        "daily_file_sha256"
    ]:
        raise ValueError("development daily input changed after selection")
    if _sorted_file_hashes(args.development_tdnet) != selection["data"][
        "development_tdnet_file_hashes_sorted"
    ]:
        raise ValueError("development TDnet input changed after selection")
    if _sorted_file_hashes(args.development_tdnet_manifest) != selection[
        "data"
    ]["development_tdnet_manifest_hashes_sorted"]:
        raise ValueError("development TDnet manifest changed after selection")
    if _sorted_file_hashes(args.holdout_tdnet) != selection["data"][
        "holdout_tdnet_file_hashes_sorted"
    ]:
        raise ValueError("holdout TDnet input changed after selection")
    if _sorted_file_hashes(args.holdout_tdnet_manifest) != selection["data"][
        "holdout_tdnet_manifest_hashes_sorted"
    ]:
        raise ValueError("holdout TDnet manifest changed after selection")
    if _sha256_file(args.full_calendar) != selection["data"][
        "full_calendar_seal_file_sha256"
    ]:
        raise ValueError("full calendar changed after selection")
    sealed_manifest = _verify_sealed_manifest(args.sealed_jpx_manifest)
    development_sessions = _load_calendar(
        selection["data"]["calendar_path"], CONFIRMATION_END
    )
    sessions = _verify_full_calendar_seal(
        args.full_calendar, development_sessions, sealed_manifest
    )
    if session_calendar_hash(sessions, through=HOLDOUT_END) != selection[
        "data"
    ]["full_calendar_sha256"]:
        raise ValueError("full session calendar digest changed after selection")
    _tdnet_manifest_coverage(
        [*args.development_tdnet, *args.holdout_tdnet],
        [
            *args.development_tdnet_manifest,
            *args.holdout_tdnet_manifest,
        ],
    )
    winner_spec = ModelSpec.from_dict(selection["winner"])
    baseline_spec = ModelSpec(
        "legacy_v03",
        "positive_session",
        c=0.08,
        class_weight="balanced",
        train_start=str(LEGACY_REGIME_START.date()),
        min_train_sessions=20,
        legacy_estimator_class_weight=True,
    )

    receipt = {
        "schema_version": 1,
        "phase": "sealed_holdout_consumed",
        "started_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "selection_lock_sha256": selection_sha256,
        "sealed_jpx_manifest_sha256": _sha256_file(args.sealed_jpx_manifest),
        "holdout_tdnet_hashes": _sorted_file_hashes(args.holdout_tdnet),
        "holdout_tdnet_manifest_hashes": _sorted_file_hashes(
            args.holdout_tdnet_manifest
        ),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as stream:
        stream.write(json_dumps(receipt) + "\n")

    sealed_paths = _sealed_jpx_paths(args.sealed_jpx_manifest, sealed_manifest)
    blind, parse_manifest = collect_jpx(sealed_paths)
    _verify_parsed_jpx_manifest(parse_manifest, sealed_paths, blind)
    _verify_blind_dates(blind, sealed_manifest)
    written_jpx = write_frame(blind, jpx_output_path)
    parse_manifest = {
        **parse_manifest,
        "schema_version": 2,
        "export_path": str(written_jpx),
        "export_sha256": _sha256_file(written_jpx),
    }
    write_json(parse_manifest, jpx_manifest_path)
    development = normalize_daily_prices(read_frame(args.development_daily))
    prices = merge_daily_prices([development, blind])
    disclosures, tdnet_complete_dates = _load_tdnet(
        [*args.development_tdnet, *args.holdout_tdnet],
        [
            *args.development_tdnet_manifest,
            *args.holdout_tdnet_manifest,
        ],
    )
    disclosures = disclosures[
        disclosures["published_at"].dt.tz_localize(None).dt.normalize().le(
            HOLDOUT_END
        )
    ].copy()
    print("build full panel after winner lock", flush=True)
    panel, coverage = build_research_panel(
        prices, disclosures, sessions, tdnet_complete_dates
    )
    warmup_only = panel["date"].eq(pd.Timestamp("2025-08-01"))
    panel.loc[warmup_only, "training_eligible"] = False
    if panel.loc[warmup_only, "training_eligible"].any():
        raise AssertionError("contaminated holdout warmup entered model training")
    winner = evaluate_spec(
        panel,
        winner_spec,
        evaluation_start=HOLDOUT_START,
        evaluation_end=HOLDOUT_END,
        evaluation_sessions=sessions,
        retrain_frequency="M",
        bootstrap_samples=args.bootstrap_samples,
    )
    baseline = evaluate_spec(
        panel,
        baseline_spec,
        evaluation_start=HOLDOUT_START,
        evaluation_end=HOLDOUT_END,
        evaluation_sessions=sessions,
        retrain_frequency="M",
        bootstrap_samples=args.bootstrap_samples,
    )
    gate = _holdout_gate_report(winner, baseline, args.bootstrap_samples)
    if gate["passed"]:
        status = "validated_shadow"
    elif gate["diagnostics_monthly"]["mean_pct"] > 0.0:
        status = "provisional_shadow"
    else:
        status = "no_demonstrated_tradable_edge"
    write_frame(
        winner.picks,
        output_path.with_name(output_path.stem + "_winner_picks.csv"),
    )
    write_frame(
        baseline.picks,
        output_path.with_name(output_path.stem + "_v03_control_picks.csv"),
    )
    payload = {
        "schema_version": 1,
        "phase": "sealed_holdout_result",
        "holdout_access_count": 1,
        "selection_lock_path": str(selection_path),
        "selection_lock_sha256": selection_sha256,
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "sealed_jpx_manifest_path": str(Path(args.sealed_jpx_manifest).resolve()),
        "sealed_jpx_manifest_sha256": _sha256_file(args.sealed_jpx_manifest),
        "sealed_jpx_canonical_files_sha256": SEALED_JPX_CANONICAL_FILES_SHA256,
        "holdout_consumed_receipt_path": str(receipt_path),
        "holdout_consumed_receipt_sha256": _sha256_file(receipt_path),
        "holdout_jpx_parse_manifest_path": str(jpx_manifest_path),
        "holdout_jpx_parse_manifest_sha256": _sha256_file(jpx_manifest_path),
        "holdout_jpx_parser_version": parse_manifest.get("parser_version"),
        "winner": asdict(winner_spec),
        "winner_spec_id": winner_spec.id,
        "status": status,
        "holdout_gate": gate,
        "winner_result": winner.summary,
        "v03_control_result": baseline.summary,
        "data": {
            "development_daily_sha256": _sha256_file(args.development_daily),
            "holdout_daily_sha256": _sha256_file(jpx_output_path),
            "full_calendar_sha256": _sha256_file(args.full_calendar),
            "full_calendar_prefix_sha256": session_calendar_hash(
                sessions, through=HOLDOUT_END
            ),
            "development_tdnet_sha256": {
                str(path): _sha256_file(path) for path in args.development_tdnet
            },
            "holdout_tdnet_sha256": {
                str(path): _sha256_file(path) for path in args.holdout_tdnet
            },
            "development_tdnet_manifest_sha256": {
                str(path): _sha256_file(path)
                for path in args.development_tdnet_manifest
            },
            "holdout_tdnet_manifest_sha256": {
                str(path): _sha256_file(path)
                for path in args.holdout_tdnet_manifest
            },
            "source_incomplete_dates": [
                str(value.date())
                for value in coverage.loc[
                    ~coverage["source_complete"], "date"
                ]
            ],
        },
        "implementation_sha256": _implementation_hashes(),
        "holdout_retuning_performed": False,
        "next_action": (
            "freeze this specification and collect 60-100 truly forward sessions; "
            "the prospective futures challenger remains separate"
        ),
    }
    write_json(payload, output_path)
    print(f"holdout_result={output_path}", flush=True)
    print(f"status={status}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--development-daily", required=True)
    select.add_argument("--development-daily-manifest", required=True)
    select.add_argument("--development-calendar", required=True)
    select.add_argument("--development-tdnet", required=True, nargs="+")
    select.add_argument(
        "--development-tdnet-manifest", required=True, nargs="+"
    )
    select.add_argument("--full-calendar-seal", required=True)
    select.add_argument("--holdout-tdnet-seal", required=True, nargs="+")
    select.add_argument(
        "--holdout-tdnet-manifest-seal", required=True, nargs="+"
    )
    select.add_argument("--sealed-jpx-manifest", required=True)
    select.add_argument(
        "--output", default="research/model_v04_selection_lock.json"
    )
    select.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )

    holdout = subparsers.add_parser("holdout")
    holdout.add_argument("--selection-lock", required=True)
    holdout.add_argument("--expected-selection-sha256", required=True)
    holdout.add_argument("--development-daily", required=True)
    holdout.add_argument(
        "--holdout-jpx-output", default="/tmp/jpx_holdout_session.pkl"
    )
    holdout.add_argument("--full-calendar", required=True)
    holdout.add_argument("--development-tdnet", required=True, nargs="+")
    holdout.add_argument(
        "--development-tdnet-manifest", required=True, nargs="+"
    )
    holdout.add_argument("--holdout-tdnet", required=True, nargs="+")
    holdout.add_argument(
        "--holdout-tdnet-manifest", required=True, nargs="+"
    )
    holdout.add_argument("--sealed-jpx-manifest", required=True)
    holdout.add_argument("--output", default="research/model_v04_holdout.json")
    holdout.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.bootstrap_samples != DEFAULT_BOOTSTRAP_SAMPLES:
        raise ValueError(
            f"formal selection and holdout require exactly "
            f"{DEFAULT_BOOTSTRAP_SAMPLES} bootstrap samples"
        )
    if args.phase == "select":
        select_phase(args)
    elif args.phase == "holdout":
        holdout_phase(args)
    else:  # pragma: no cover
        raise RuntimeError(args.phase)


if __name__ == "__main__":
    main()
