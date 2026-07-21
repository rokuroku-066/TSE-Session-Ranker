from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .config import RankerConfig
from .data.common import normalize_daily_prices
from .exceptions import DataValidationError
from .features import FEATURE_COLUMNS, build_feature_panel
from .inference import rank_candidates
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


def _pick_metrics(frame: pd.DataFrame, cost_bps: float) -> dict[str, Any]:
    if frame.empty:
        return {
            "n": 0,
            "days": 0,
            "executed": 0,
            "execution_rate": np.nan,
            "hit_rate": np.nan,
            "signal_hit_rate_including_unfilled": np.nan,
            "gross_mean_pct": np.nan,
        }
    executed = frame["label"].notna()
    gross = frame["oc_return_pct"].fillna(0.0)
    costs = executed.astype(float) * (cost_bps / 100.0)
    return {
        "n": int(len(frame)),
        "days": int(frame["date"].nunique()),
        "executed": int(executed.sum()),
        "execution_rate": float(executed.mean()),
        "hit_rate": float(frame.loc[executed, "label"].mean()),
        "signal_hit_rate_including_unfilled": float(frame["label"].fillna(0).mean()),
        "gross_mean_pct": float(gross.mean()),
        "gross_median_pct": float(gross.median()),
        "net_mean_pct_at_cost": float((gross - costs).mean()),
    }


def monthly_walk_forward(
    prices: pd.DataFrame,
    evaluation_start: object,
    evaluation_end: object,
    config: RankerConfig | None = None,
    train_start: object | None = None,
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
    panel = build_feature_panel(canonical.loc[canonical["date"].le(end)].copy(), settings)
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
    summary = {
        "evaluation_start": str(start.date()),
        "evaluation_end": str(end.date()),
        "baseline_rows": int(len(scores)),
        "baseline_executed_rows": int(len(evaluated)),
        "baseline_execution_rate": float(len(evaluated) / len(scores)),
        "baseline_days": int(scores["date"].nunique()),
        "baseline_hit_rate": float(labels.mean()),
        "auc": float(roc_auc_score(labels, evaluated["model_score"])),
        "brier": float(brier_score_loss(labels, evaluated["model_score"])),
        "top1": _pick_metrics(top1, settings.cost_bps),
        "top2": _pick_metrics(top2, settings.cost_bps),
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
