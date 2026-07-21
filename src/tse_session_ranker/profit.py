from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .exceptions import DataValidationError


def daily_portfolio_returns(
    picks: pd.DataFrame,
    *,
    top_k: int,
    cost_bps: float,
) -> pd.DataFrame:
    """Return fixed-allocation daily P&L for the first ``top_k`` ranks.

    Each slot receives ``1 / top_k`` of capital before execution is known.
    An unfilled slot earns zero and incurs no cost; capital is not reallocated
    retrospectively to the filled slot.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least one")
    if not math.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and non-negative")
    required = {"date", "model_rank", "label", "oc_return_pct"}
    missing = sorted(required - set(picks.columns))
    if missing:
        raise DataValidationError(f"picks are missing profit columns: {missing}")
    selected = picks[picks["model_rank"].le(top_k)].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=["date", "gross_return_pct", "net_return_pct", "executed_slots"]
        )
    executed = selected["label"].notna()
    selected["gross_slot_pct"] = selected["oc_return_pct"].fillna(0.0)
    selected["net_slot_pct"] = selected["gross_slot_pct"] - (
        executed.astype(float) * (cost_bps / 100.0)
    )
    selected["executed_slot"] = executed.astype(int)
    daily = selected.groupby("date", sort=True).agg(
        gross_slot_sum=("gross_slot_pct", "sum"),
        net_slot_sum=("net_slot_pct", "sum"),
        executed_slots=("executed_slot", "sum"),
        signal_slots=("model_rank", "size"),
    )
    daily["gross_return_pct"] = daily["gross_slot_sum"] / top_k
    daily["net_return_pct"] = daily["net_slot_sum"] / top_k
    return daily.reset_index()[
        [
            "date",
            "gross_return_pct",
            "net_return_pct",
            "executed_slots",
            "signal_slots",
        ]
    ]


def _profit_factor(values: pd.Series) -> float:
    gains = float(values.clip(lower=0).sum())
    losses = float(-values.clip(upper=0).sum())
    return gains / losses if losses else float("inf")


def profit_metrics(
    picks: pd.DataFrame,
    *,
    top_k: int,
    cost_bps: float,
    cost_sensitivity_bps: Iterable[float] = (0.0, 10.0, 20.0, 40.0, 60.0),
) -> dict[str, Any]:
    """Summarise the scheduled-day return objective and robustness diagnostics."""

    if not math.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and non-negative")
    sensitivity_costs = tuple(float(value) for value in cost_sensitivity_bps)
    if any(not math.isfinite(value) or value < 0 for value in sensitivity_costs):
        raise ValueError("cost sensitivity values must be finite and non-negative")
    selected = picks[picks["model_rank"].le(top_k)].copy()
    if selected.empty:
        return {
            "n": 0,
            "days": 0,
            "executed": 0,
            "execution_rate": np.nan,
            "hit_rate": np.nan,
            "net_mean_pct_at_cost": np.nan,
        }
    daily = daily_portfolio_returns(selected, top_k=top_k, cost_bps=cost_bps)
    executed = selected["label"].notna()
    net = daily.set_index("date")["net_return_pct"].sort_index()
    gross = daily.set_index("date")["gross_return_pct"].sort_index()
    monthly = net.groupby(net.index.to_period("M")).mean()
    equity = (1.0 + net / 100.0).cumprod()
    equity_with_initial_nav = np.concatenate(([1.0], equity.to_numpy()))
    running_peak = np.maximum.accumulate(equity_with_initial_nav)
    drawdown = equity_with_initial_nav / running_peak - 1.0
    top_days = net.nlargest(min(5, len(net))).index
    without_top5 = net.drop(top_days)
    positive_net = float(net.clip(lower=0).sum())
    largest_share = (
        float(net.max() / positive_net) if positive_net > 0 else np.nan
    )
    sensitivity: dict[str, float] = {}
    for scenario_cost in sensitivity_costs:
        scenario = daily_portfolio_returns(
            selected, top_k=top_k, cost_bps=float(scenario_cost)
        )
        sensitivity[str(float(scenario_cost))] = float(
            scenario["net_return_pct"].mean()
        )
    return {
        "n": int(len(selected)),
        "days": int(len(daily)),
        "executed": int(executed.sum()),
        "execution_rate": float(executed.mean()),
        "hit_rate": float(selected.loc[executed, "label"].mean()),
        "signal_hit_rate_including_unfilled": float(
            selected["label"].fillna(0).mean()
        ),
        "gross_mean_pct": float(gross.mean()),
        "gross_median_pct": float(gross.median()),
        "net_mean_pct_at_cost": float(net.mean()),
        "net_median_pct_at_cost": float(net.median()),
        "compounded_net_return_pct": float(100.0 * (equity.iloc[-1] - 1.0)),
        "max_drawdown_pct": float(100.0 * drawdown.min()),
        "profit_factor": _profit_factor(net),
        "positive_months": int(monthly.gt(0).sum()),
        "months": int(len(monthly)),
        "worst_month_pct": float(monthly.min()),
        "top5_removed_net_mean_pct": (
            float(without_top5.mean()) if len(without_top5) else np.nan
        ),
        "largest_day_net_pct": float(net.max()),
        "largest_day_share_of_positive_net": largest_share,
        "monthly_net_mean_pct": {str(key): float(value) for key, value in monthly.items()},
        "cost_sensitivity_net_mean_pct": sensitivity,
    }
