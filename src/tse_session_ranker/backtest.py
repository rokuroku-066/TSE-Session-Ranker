from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .config import RankerConfig
from .data.common import (
    normalize_daily_prices,
    normalize_expected_sessions,
    prepare_modeling_prices,
    session_calendar_hash,
)
from .exceptions import DataValidationError
from .features import FEATURE_COLUMNS, build_feature_panel
from .inference import rank_candidates
from .profit import profit_metrics
from .training import fit_estimator


@dataclass
class WalkForwardResult:
    summary: dict[str, Any]
    folds: pd.DataFrame
    scores: pd.DataFrame
    picks_top1: pd.DataFrame
    picks_top2: pd.DataFrame


def _daily_picks(scores: pd.DataFrame, count: int) -> pd.DataFrame:
    parts = [rank_candidates(group, count) for _, group in scores.groupby("date", sort=True)]
    return pd.concat(parts, ignore_index=True) if parts else scores.iloc[0:0].copy()


def monthly_walk_forward(
    prices: pd.DataFrame,
    evaluation_start: object,
    evaluation_end: object,
    config: RankerConfig | None = None,
    train_start: object | None = None,
    expected_sessions: object | None = None,
) -> WalkForwardResult:
    """Fixed-spec expanding monthly walk-forward evaluation."""

    settings = config or RankerConfig()
    start = pd.Timestamp(evaluation_start).normalize()
    end = pd.Timestamp(evaluation_end).normalize()
    regime_start = pd.Timestamp(train_start or settings.regime_start).normalize()
    if not regime_start < start <= end:
        raise DataValidationError(
            "walk-forward requires train_start < evaluation_start <= evaluation_end"
        )
    canonical = normalize_daily_prices(prices)
    source = canonical.loc[canonical["date"].le(end)].copy()
    calendar = None
    if expected_sessions is not None:
        calendar_all = normalize_expected_sessions(expected_sessions)
        if calendar_all.max() < end:
            raise DataValidationError(
                f"session calendar ends at {calendar_all.max().date()} before "
                f"evaluation_end {end.date()}"
            )
        if end not in calendar_all:
            raise DataValidationError(
                "evaluation_end is not in the exchange session calendar"
            )
        calendar = calendar_all[calendar_all <= end]
        expected_latest = calendar[calendar >= source["date"].min()].max()
        actual_latest = source["date"].max()
        if expected_latest > actual_latest:
            raise DataValidationError(
                f"daily history ends at {actual_latest.date()} but the session "
                f"calendar expects {expected_latest.date()}"
            )
    modeling, coverage = prepare_modeling_prices(
        source,
        coverage_lookback=settings.source_coverage_lookback,
        minimum_source_coverage=settings.minimum_source_coverage,
        expected_sessions=calendar,
    )
    panel = build_feature_panel(modeling, settings)
    scored_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for period in pd.period_range(start.to_period("M"), end.to_period("M"), freq="M"):
        score_start = max(start, period.start_time.normalize())
        score_end = min(end, period.end_time.normalize())
        training = panel[
            panel["date"].between(regime_start, score_start - pd.Timedelta(days=1))
            & panel["training_eligible"]
            & panel["label"].notna()
        ].copy()
        scoring = panel[
            panel["date"].between(score_start, score_end)
            & panel["eligible"]
        ].copy()
        if scoring.empty:
            continue
        estimator = fit_estimator(training, settings)
        scoring["model_score"] = estimator.predict_proba(
            scoring[list(FEATURE_COLUMNS)]
        )[:, 1]
        scored_parts.append(scoring)
        train_max = training["date"].max()
        if not train_max < scoring["date"].min():
            raise DataValidationError("walk-forward fold has overlapping train/test dates")
        fold_rows.append(
            {
                "period": str(period),
                "train_start": str(training["date"].min().date()),
                "train_end": str(train_max.date()),
                "score_start": str(scoring["date"].min().date()),
                "score_end": str(scoring["date"].max().date()),
                "train_rows": int(len(training)),
                "score_rows": int(len(scoring)),
                "score_days": int(scoring["date"].nunique()),
            }
        )
    if not scored_parts:
        raise DataValidationError("walk-forward produced no scoring rows")
    scores = pd.concat(scored_parts, ignore_index=True)
    top1 = _daily_picks(scores, 1)
    top2 = _daily_picks(scores, 2)
    evaluated = scores[scores["label"].notna()].copy()
    labels = evaluated["label"].astype(int)
    top1_metrics = profit_metrics(top1, top_k=1, cost_bps=settings.cost_bps)
    top2_metrics = profit_metrics(top2, top_k=2, cost_bps=settings.cost_bps)
    excluded_dates = coverage.loc[~coverage["source_complete"], "date"]
    embargoed_dates = panel.loc[
        (~panel["source_complete"] | ~panel["universe_source_complete"])
        & panel["date"].between(start, end),
        "date",
    ].drop_duplicates()
    summary = {
        "evaluation_start": str(start.date()),
        "evaluation_end": str(end.date()),
        "selection_objective": settings.selection_objective,
        "data_semantics": settings.data_semantics,
        "assumed_round_trip_cost_bps": settings.cost_bps,
        "primary_objective": "top1.net_mean_pct_at_cost",
        "primary_objective_value": top1_metrics["net_mean_pct_at_cost"],
        "excluded_source_incomplete_dates": [
            str(pd.Timestamp(value).date()) for value in excluded_dates
        ],
        "embargoed_evaluation_dates": [
            str(pd.Timestamp(value).date()) for value in embargoed_dates
        ],
        "source_coverage_policy": "sticky_prior_accepted_q90_v1",
        "session_calendar_mode": (
            "explicit_exchange_sessions" if calendar is not None else "observed_sessions_only"
        ),
        "session_calendar_sha256": (
            session_calendar_hash(calendar, through=end)
            if calendar is not None
            else None
        ),
        "session_calendar_through": (
            str(calendar.max().date()) if calendar is not None else None
        ),
        "baseline_rows": int(len(scores)),
        "baseline_executed_rows": int(len(evaluated)),
        "baseline_execution_rate": float(len(evaluated) / len(scores)),
        "baseline_days": int(scores["date"].nunique()),
        "baseline_hit_rate": float(labels.mean()),
        "auc": float(roc_auc_score(labels, evaluated["model_score"])),
        "brier": float(brier_score_loss(labels, evaluated["model_score"])),
        "top1": top1_metrics,
        "top2": top2_metrics,
        "calibration_status": "uncalibrated",
        "selection_spec_changed_during_evaluation": False,
    }
    return WalkForwardResult(
        summary=summary,
        folds=pd.DataFrame(fold_rows),
        scores=scores,
        picks_top1=top1,
        picks_top2=top2,
    )
