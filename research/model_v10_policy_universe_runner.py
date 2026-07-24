#!/usr/bin/env python3
"""Run the preregistered v1.0 zero-base policy/universe screen.

The runner deliberately lives outside the repository.  It consumes the
content-addressed v0.8/v0.9 frozen panel, refits the three already-defined
monthly walk-forward scorers, and tests only the policy/universe mechanisms
registered in ``protocol.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd


ROOT = Path("/workspace/scratch/8678b1d14f37")
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tse_session_ranker.research_models import (  # noqa: E402
    ResearchModelSpec,
    fit_research_model,
)


BASE_FEATURES = (
    "oc_last", "oc_mean_5", "oc_mean_20", "oc_mean_60", "oc_win_20",
    "oc_std_20", "overnight_last", "overnight_mean_20", "overnight_mean_60",
    "night_day_corr_60", "xrank_atr14_pct", "xrank_close_momentum_5",
    "xrank_close_momentum_20", "xrank_close_momentum_60",
    "xrank_prior_close_location_20",
)
G5_FEATURES = (
    "no_trade_rate_20", "no_trade_rate_60", "flat_oc_rate_20",
    "zero_range_rate_20",
)
SCORER_FEATURES = {
    "G0": BASE_FEATURES,
    "L4": (*BASE_FEATURES, "flat_oc_rate_20"),
    "L6": (*BASE_FEATURES, *G5_FEATURES),
}
ADVERSE_FLAGS = (
    "tdnet_clean_revision_down_title",
    "tdnet_clean_dividend_down_title",
    "tdnet_clean_has_external_equity_financing",
    "tdnet_clean_has_impairment_loss",
    "tdnet_clean_has_audit_problem",
    "tdnet_clean_has_correction",
    "tdnet_clean_support_adverse_conflict",
)
SUPPORTIVE_FLAGS = (
    "tdnet_clean_revision_up_title",
    "tdnet_clean_dividend_up_title",
    "tdnet_clean_has_buyback_decision",
)
POLICY_IDS = (
    "C00_L4_TOP2_EQUAL",
    "H01_ENSEMBLE_MEAN",
    "H02_ENSEMBLE_WORSTCASE",
    "H03_DISAGREEMENT_PENALTY",
    "H04_STABLE_TOPSET",
    "H05_INVERSE_UNCERTAINTY_WEIGHT",
    "H06_GAP_CONFIDENCE_COUNT",
    "H07_LIQUIDITY_RELIABILITY_UNIVERSE",
    "H08_ADVERSE_EVENT_EXCLUSION",
    "H09_EVENT_BARBBELL",
    "H10_STYLE_RESIDUAL",
    "H11_RISK_UTILITY",
    "H12_MOMENTUM_BUCKET_DIVERSIFICATION",
    "H13_LOW_CORRELATION_PAIR",
    "H14_POSTERIOR_NET_ABSTAIN",
    "H15_SPREAD_CASH_OVERLAY",
    "H16_EVENT_AWARE_UTILITY",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_protocol(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    ids = [item["id"] for item in raw["hypotheses"]]
    expected = list(POLICY_IDS[1:])
    if ids != expected or len(ids) < 10:
        raise RuntimeError("protocol hypotheses do not match the locked runner")
    if raw["authority"]["production_promotion_allowed"] is not False:
        raise RuntimeError("retrospective runner cannot promote production")
    return raw


def _fit(training: pd.DataFrame, columns: tuple[str, ...], name: str):
    return fit_research_model(
        ResearchModelSpec(
            name=f"v10_{name}_rank_ridge",
            family="ridge_daily_rank",
            objective="same_day_return_percentile",
            parameters={"alpha": 1.0},
        ),
        training,
        columns,
    )


def build_walk_forward_scores(
    panel: pd.DataFrame,
    *,
    training_start: pd.Timestamp,
    score_start: pd.Timestamp,
    score_end: pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    projection = list(dict.fromkeys([
        "date", "code", "name", "label", "oc_return_pct",
        "price_eligible", "price_training_eligible",
        "candidate_price_source_max_date",
        *BASE_FEATURES, *G5_FEATURES,
        "tdnet_clean_any", *ADVERSE_FLAGS, *SUPPORTIVE_FLAGS,
    ]))
    periods = pd.period_range(score_start.to_period("M"), score_end.to_period("M"))
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for period in periods:
        fold_start = max(score_start, period.start_time.normalize())
        fold_end = min(score_end, period.end_time.normalize())
        training = panel.loc[
            panel["date"].between(training_start, fold_start - pd.Timedelta(days=1))
            & panel["price_training_eligible"].eq(True)
            & panel["oc_return_pct"].notna(),
            projection,
        ].copy()
        scoring = panel.loc[
            panel["date"].between(fold_start, fold_end)
            & panel["price_eligible"].eq(True),
            projection,
        ].copy()
        if training.empty or scoring.empty or training["date"].max() >= fold_start:
            raise RuntimeError(f"invalid walk-forward fold {period}")
        fold_record: dict[str, Any] = {
            "period": str(period),
            "train_start": str(training["date"].min().date()),
            "train_end": str(training["date"].max().date()),
            "train_rows": int(len(training)),
            "score_rows": int(len(scoring)),
            "strictly_prior": True,
        }
        for model_name, features in SCORER_FEATURES.items():
            scorer = _fit(training, features, model_name)
            scoring[f"score_{model_name}"] = scorer.score(scoring)
        parts.append(scoring)
        folds.append(fold_record)
    scores = pd.concat(parts, ignore_index=True)
    scores = scores.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
    for model_name in SCORER_FEATURES:
        col = f"score_{model_name}"
        scores[f"pct_{model_name}"] = scores.groupby("date", sort=False)[col].rank(
            method="average", pct=True
        )
        scores[f"rank_{model_name}"] = scores.groupby("date", sort=False)[col].rank(
            method="first", ascending=False
        ).astype(int)
    pcols = [f"pct_{name}" for name in SCORER_FEATURES]
    scores["ensemble_mean"] = scores[pcols].mean(axis=1)
    scores["ensemble_worst"] = scores[pcols].min(axis=1)
    scores["ensemble_disagreement"] = scores[pcols].std(axis=1, ddof=0)
    scores["adverse_event"] = scores[list(ADVERSE_FLAGS)].fillna(0).gt(0).any(axis=1)
    scores["supportive_event"] = scores[list(SUPPORTIVE_FLAGS)].fillna(0).gt(0).any(axis=1)
    scores["has_event"] = scores["tdnet_clean_any"].fillna(0).gt(0)
    return scores, folds


def _ordered(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    return frame.sort_values(
        [key, "code"], ascending=[False, True], kind="stable"
    )


def _pick_rows(
    frame: pd.DataFrame,
    indices: list[int],
    weights: list[float],
    policy_id: str,
) -> pd.DataFrame:
    if len(indices) != len(weights):
        raise RuntimeError("selection/weight length mismatch")
    if not indices:
        return pd.DataFrame(columns=[
            "date", "code", "name", "label", "oc_return_pct",
            "weight", "policy_id",
        ])
    selected = frame.loc[indices, [
        "date", "code", "name", "label", "oc_return_pct",
    ]].copy()
    selected["weight"] = np.asarray(weights, dtype=float)
    selected["policy_id"] = policy_id
    return selected


def _two_equal(frame: pd.DataFrame, key: str, policy_id: str) -> pd.DataFrame:
    ordered = _ordered(frame, key).head(2)
    if ordered.empty:
        return _pick_rows(frame, [], [], policy_id)
    weights = [1.0 / len(ordered)] * len(ordered)
    return _pick_rows(frame, ordered.index.tolist(), weights, policy_id)


def _style_residual(frame: pd.DataFrame) -> pd.Series:
    raw = frame[[
        "xrank_close_momentum_20", "xrank_atr14_pct",
        "xrank_prior_close_location_20",
    ]].astype(float)
    values = raw.fillna(raw.median()).fillna(0.0).to_numpy()
    design = np.column_stack([np.ones(len(frame)), values])
    target = frame["ensemble_mean"].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return pd.Series(target - design @ coef, index=frame.index)


def _pair_correlation(
    returns: pd.DataFrame,
    prior_dates: pd.DatetimeIndex,
    first_code: Any,
    second_code: Any,
) -> float:
    if first_code not in returns.columns or second_code not in returns.columns:
        return 0.0
    pair = returns.loc[prior_dates, [first_code, second_code]].dropna()
    if len(pair) < 20:
        return 0.0
    value = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    return value if np.isfinite(value) else 0.0


def build_policy_picks(
    scores: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    sessions = pd.DatetimeIndex(scores["date"].drop_duplicates().sort_values())
    all_sessions = pd.DatetimeIndex(panel["date"].drop_duplicates().sort_values())
    return_matrix = panel.pivot(
        index="date", columns="code", values="oc_return_pct"
    ).sort_index()
    picks: dict[str, list[pd.DataFrame]] = {key: [] for key in POLICY_IDS}
    gap_history: list[float] = []
    posterior_sum = 0.0
    posterior_count = 0
    daily_audit: list[dict[str, Any]] = []

    for date in sessions:
        frame = scores.loc[scores["date"].eq(date)].copy()
        if len(frame) < 20:
            raise RuntimeError(f"too few eligible candidates on {date}")
        frame["topset_count"] = sum(
            frame[f"rank_{name}"].le(20).astype(int) for name in SCORER_FEATURES
        )
        frame["disagreement_utility"] = (
            frame["ensemble_mean"] - 0.50 * frame["ensemble_disagreement"]
        )
        for source, target in (
            ("xrank_atr14_pct", "atr_pct"),
            ("oc_std_20", "vol_pct"),
            ("oc_mean_20", "ocmean_pct"),
        ):
            frame[target] = frame[source].rank(method="average", pct=True).fillna(0.5)
        frame["risk_utility"] = (
            frame["ensemble_mean"]
            - 0.15 * frame["atr_pct"]
            - 0.10 * frame["vol_pct"]
            + 0.05 * frame["ocmean_pct"]
        )
        frame["style_residual"] = _style_residual(frame)
        frame["event_utility"] = (
            frame["ensemble_mean"]
            + 0.05 * frame["supportive_event"].astype(float)
            - 0.10 * frame["adverse_event"].astype(float)
        )

        # Control: retained only for comparison, never eligible as a new idea.
        picks["C00_L4_TOP2_EQUAL"].append(
            _two_equal(frame, "score_L4", "C00_L4_TOP2_EQUAL")
        )
        picks["H01_ENSEMBLE_MEAN"].append(
            _two_equal(frame, "ensemble_mean", "H01_ENSEMBLE_MEAN")
        )
        picks["H02_ENSEMBLE_WORSTCASE"].append(
            _two_equal(frame, "ensemble_worst", "H02_ENSEMBLE_WORSTCASE")
        )
        picks["H03_DISAGREEMENT_PENALTY"].append(
            _two_equal(frame, "disagreement_utility", "H03_DISAGREEMENT_PENALTY")
        )

        stable = frame.sort_values(
            ["topset_count", "ensemble_mean", "code"],
            ascending=[False, False, True], kind="stable"
        ).head(2)
        picks["H04_STABLE_TOPSET"].append(
            _pick_rows(
                frame, stable.index.tolist(), [0.5] * len(stable),
                "H04_STABLE_TOPSET",
            )
        )

        uncertain = _ordered(frame, "ensemble_mean").head(2)
        inv = 1.0 / (0.05 + uncertain["ensemble_disagreement"].to_numpy(float))
        inv /= inv.sum()
        picks["H05_INVERSE_UNCERTAINTY_WEIGHT"].append(
            _pick_rows(
                frame, uncertain.index.tolist(), inv.tolist(),
                "H05_INVERSE_UNCERTAINTY_WEIGHT",
            )
        )

        ensemble_order = _ordered(frame, "ensemble_mean")
        current_gap = float(
            ensemble_order["ensemble_mean"].iloc[0]
            - ensemble_order["ensemble_mean"].iloc[2]
        )
        prior_gap = (
            float(np.median(gap_history[-60:])) if len(gap_history) >= 20 else None
        )
        count = 1 if prior_gap is not None and current_gap > prior_gap else 2
        chosen = ensemble_order.head(count)
        picks["H06_GAP_CONFIDENCE_COUNT"].append(
            _pick_rows(
                frame, chosen.index.tolist(), [1.0 / count] * count,
                "H06_GAP_CONFIDENCE_COUNT",
            )
        )

        no_trade_median = float(frame["no_trade_rate_20"].median())
        flat_q75 = float(frame["flat_oc_rate_20"].quantile(0.75))
        reliable = frame.loc[
            frame["no_trade_rate_20"].le(no_trade_median)
            & frame["flat_oc_rate_20"].le(flat_q75)
        ]
        picks["H07_LIQUIDITY_RELIABILITY_UNIVERSE"].append(
            _two_equal(
                reliable, "ensemble_mean", "H07_LIQUIDITY_RELIABILITY_UNIVERSE"
            )
        )

        nonadverse = frame.loc[~frame["adverse_event"]]
        picks["H08_ADVERSE_EVENT_EXCLUSION"].append(
            _two_equal(nonadverse, "ensemble_mean", "H08_ADVERSE_EVENT_EXCLUSION")
        )

        event_cutoff = float(frame["ensemble_mean"].quantile(0.90))
        events = _ordered(
            frame.loc[frame["has_event"] & frame["ensemble_mean"].ge(event_cutoff)],
            "ensemble_mean",
        )
        nonevents = _ordered(frame.loc[~frame["has_event"]], "ensemble_mean")
        if not events.empty and not nonevents.empty:
            indices = [int(events.index[0]), int(nonevents.index[0])]
            weights = [0.25, 0.75]
        else:
            chosen_nonevent = nonevents.head(2)
            indices = chosen_nonevent.index.tolist()
            weights = [1.0 / len(indices)] * len(indices)
        picks["H09_EVENT_BARBBELL"].append(
            _pick_rows(frame, indices, weights, "H09_EVENT_BARBBELL")
        )

        picks["H10_STYLE_RESIDUAL"].append(
            _two_equal(frame, "style_residual", "H10_STYLE_RESIDUAL")
        )
        picks["H11_RISK_UTILITY"].append(
            _two_equal(frame, "risk_utility", "H11_RISK_UTILITY")
        )

        leader = ensemble_order.iloc[0]
        momentum_median = float(frame["xrank_close_momentum_20"].median())
        leader_value = float(leader["xrank_close_momentum_20"])
        if not np.isfinite(leader_value):
            opposite = frame.drop(index=leader.name)
        elif leader_value >= momentum_median:
            opposite = frame.loc[frame["xrank_close_momentum_20"].lt(momentum_median)]
        else:
            opposite = frame.loc[frame["xrank_close_momentum_20"].ge(momentum_median)]
        opposite = _ordered(opposite.drop(index=leader.name, errors="ignore"), "ensemble_mean")
        if opposite.empty:
            opposite = ensemble_order.drop(index=leader.name).head(1)
        bucket_indices = [int(leader.name), int(opposite.index[0])]
        picks["H12_MOMENTUM_BUCKET_DIVERSIFICATION"].append(
            _pick_rows(
                frame, bucket_indices, [0.5, 0.5],
                "H12_MOMENTUM_BUCKET_DIVERSIFICATION",
            )
        )

        prior_sessions = all_sessions[all_sessions < date][-60:]
        pair_candidates = ensemble_order.iloc[1:10].copy()
        correlations = [
            _pair_correlation(
                return_matrix, prior_sessions, leader["code"], candidate["code"]
            )
            for _, candidate in pair_candidates.iterrows()
        ]
        pair_candidates["pair_utility"] = (
            pair_candidates["ensemble_mean"].to_numpy(float)
            - 0.15 * ((np.asarray(correlations) + 1.0) / 2.0)
        )
        second = _ordered(pair_candidates, "pair_utility").iloc[0]
        picks["H13_LOW_CORRELATION_PAIR"].append(
            _pick_rows(
                frame, [int(leader.name), int(second.name)], [0.5, 0.5],
                "H13_LOW_CORRELATION_PAIR",
            )
        )

        posterior_mean = posterior_sum / (posterior_count + 50)
        if posterior_mean > 0.0:
            abstain_selection = ensemble_order.head(2)
            abstain_indices = abstain_selection.index.tolist()
            abstain_weights = [0.5] * len(abstain_indices)
        else:
            abstain_indices, abstain_weights = [], []
        picks["H14_POSTERIOR_NET_ABSTAIN"].append(
            _pick_rows(
                frame, abstain_indices, abstain_weights,
                "H14_POSTERIOR_NET_ABSTAIN",
            )
        )

        if len(gap_history) >= 20:
            gap_ref = float(np.quantile(gap_history[-60:], 0.75))
            exposure = float(np.clip(current_gap / max(gap_ref, 1e-12), 0.25, 1.0))
        else:
            exposure = 0.50
        spread_selection = ensemble_order.head(2)
        picks["H15_SPREAD_CASH_OVERLAY"].append(
            _pick_rows(
                frame, spread_selection.index.tolist(),
                [exposure / 2.0] * len(spread_selection),
                "H15_SPREAD_CASH_OVERLAY",
            )
        )
        picks["H16_EVENT_AWARE_UTILITY"].append(
            _two_equal(frame, "event_utility", "H16_EVENT_AWARE_UTILITY")
        )

        # Update state only after all selections for this session are frozen.
        top_decile = frame.loc[frame["ensemble_mean"].ge(event_cutoff)]
        observed = top_decile["oc_return_pct"].dropna()
        posterior_sum += float((observed - 0.20).sum())
        posterior_count += int(len(observed))
        gap_history.append(current_gap)
        daily_audit.append({
            "date": str(date.date()),
            "eligible": int(len(frame)),
            "gap_top1_top3": current_gap,
            "prior_gap_median60": prior_gap,
            "posterior_net20_before_selection": posterior_mean,
            "posterior_observations_before_selection": posterior_count - len(observed),
            "event_top_decile_count": int(len(events)),
        })

    combined: dict[str, pd.DataFrame] = {}
    for policy_id, chunks in picks.items():
        nonempty = [chunk for chunk in chunks if not chunk.empty]
        if nonempty:
            value = pd.concat(nonempty, ignore_index=True)
            value["date"] = pd.to_datetime(value["date"])
        else:
            value = pd.DataFrame(columns=[
                "date", "code", "name", "label", "oc_return_pct",
                "weight", "policy_id",
            ])
        combined[policy_id] = value
    return combined, pd.DataFrame(daily_audit)


def daily_returns(
    picks: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    cost_bps: float,
) -> pd.DataFrame:
    if picks.empty:
        result = pd.DataFrame(index=sessions)
        result["gross"] = 0.0
        result["net"] = 0.0
        result["executed_weight"] = 0.0
        result["signal_weight"] = 0.0
        result["candidate_count"] = 0
        return result
    frame = picks.copy()
    executed = frame["oc_return_pct"].notna()
    frame["gross_contribution"] = (
        frame["weight"] * frame["oc_return_pct"].fillna(0.0)
    )
    frame["cost_contribution"] = (
        frame["weight"] * executed.astype(float) * cost_bps / 100.0
    )
    frame["executed_weight"] = frame["weight"] * executed.astype(float)
    grouped = frame.groupby("date", sort=True).agg(
        gross=("gross_contribution", "sum"),
        cost=("cost_contribution", "sum"),
        executed_weight=("executed_weight", "sum"),
        signal_weight=("weight", "sum"),
        candidate_count=("code", "size"),
    )
    grouped["net"] = grouped["gross"] - grouped["cost"]
    return grouped.reindex(sessions, fill_value=0.0)[
        ["gross", "net", "executed_weight", "signal_weight", "candidate_count"]
    ]


def _profit_factor(values: pd.Series) -> float | None:
    gains = float(values.clip(lower=0).sum())
    losses = float(-values.clip(upper=0).sum())
    return gains / losses if losses > 0 else None


def policy_metrics(
    picks: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> tuple[dict[str, Any], pd.DataFrame]:
    daily_by_cost = {
        cost: daily_returns(picks, sessions, float(cost))
        for cost in (20, 40, 60)
    }
    daily20 = daily_by_cost[20]["net"]
    monthly = daily20.groupby(daily20.index.to_period("M")).mean()
    blocks = np.array_split(np.arange(len(sessions)), 4)
    block_values = {
        f"B{idx + 1}": float(daily20.iloc[positions].mean())
        for idx, positions in enumerate(blocks)
    }
    q05 = float(daily20.quantile(0.05))
    es = float(daily20.loc[daily20.le(q05)].mean())
    equity = (1.0 + daily20 / 100.0).cumprod()
    initial = np.concatenate(([1.0], equity.to_numpy()))
    drawdown = initial / np.maximum.accumulate(initial) - 1.0
    executed = picks["oc_return_pct"].notna() if not picks.empty else pd.Series(dtype=bool)
    weighted_hit = (
        float(
            (picks.loc[executed, "weight"] * picks.loc[executed, "label"]).sum()
            / picks.loc[executed, "weight"].sum()
        )
        if executed.any() else None
    )
    code_concentration: dict[str, Any]
    if picks.empty:
        code_concentration = {
            "unique_codes": 0,
            "largest_weight_share": None,
            "top10_weight_share": None,
            "largest_positive_net20_code_share": None,
        }
    else:
        by_weight = picks.groupby("code")["weight"].sum().sort_values(ascending=False)
        net_contrib = picks.assign(
            net20_contribution=picks["weight"]
            * (
                picks["oc_return_pct"].fillna(0.0)
                - picks["oc_return_pct"].notna().astype(float) * 0.20
            )
        ).groupby("code")["net20_contribution"].sum()
        positive = net_contrib.clip(lower=0)
        code_concentration = {
            "unique_codes": int(picks["code"].nunique()),
            "largest_weight_share": float(by_weight.iloc[0] / by_weight.sum()),
            "top10_weight_share": float(by_weight.head(10).sum() / by_weight.sum()),
            "largest_positive_net20_code_share": (
                float(positive.max() / positive.sum()) if positive.sum() > 0 else None
            ),
        }
    positive_days = daily20.clip(lower=0)
    metrics: dict[str, Any] = {
        "days": int(len(sessions)),
        "rows_selected": int(len(picks)),
        "mean_candidates_per_day": float(
            daily_by_cost[20]["candidate_count"].mean()
        ),
        "zero_candidate_days": int(
            daily_by_cost[20]["candidate_count"].eq(0).sum()
        ),
        "one_candidate_days": int(
            daily_by_cost[20]["candidate_count"].eq(1).sum()
        ),
        "two_candidate_days": int(
            daily_by_cost[20]["candidate_count"].eq(2).sum()
        ),
        "mean_signal_exposure": float(daily_by_cost[20]["signal_weight"].mean()),
        "mean_executed_exposure": float(daily_by_cost[20]["executed_weight"].mean()),
        "hit_rate_executed": (
            float(picks.loc[executed, "label"].mean()) if executed.any() else None
        ),
        "weighted_hit_rate_executed": weighted_hit,
        "gross_mean_pct": float(daily_by_cost[20]["gross"].mean()),
        "net_mean_pct": {
            str(cost): float(frame["net"].mean())
            for cost, frame in daily_by_cost.items()
        },
        "net_median20_pct": float(daily20.median()),
        "compounded_net20_pct": float((equity.iloc[-1] - 1.0) * 100.0),
        "max_drawdown20_pct": float(drawdown.min() * 100.0),
        "profit_factor20": _profit_factor(daily20),
        "worst_day20_pct": float(daily20.min()),
        "p05_day20_pct": q05,
        "expected_shortfall05_day20_pct": es,
        "positive_months": int(monthly.gt(0).sum()),
        "months": int(len(monthly)),
        "monthly_net20_pct": {
            str(period): float(value) for period, value in monthly.items()
        },
        "four_blocks_net20_pct": block_values,
        "nonnegative_blocks": int(sum(value >= 0 for value in block_values.values())),
        "winning_days_removed_net20_pct": {
            str(count): float(daily20.drop(daily20.nlargest(count).index).mean())
            for count in (5, 10, 20)
        },
        "largest_day_share_positive_net20": (
            float(positive_days.max() / positive_days.sum())
            if positive_days.sum() > 0 else None
        ),
        "code_concentration": code_concentration,
    }
    daily_export = pd.DataFrame({
        "date": sessions,
        "net20": daily_by_cost[20]["net"].to_numpy(),
        "net40": daily_by_cost[40]["net"].to_numpy(),
        "net60": daily_by_cost[60]["net"].to_numpy(),
    })
    return metrics, daily_export


def bootstrap_multiplicity(
    daily: dict[str, pd.DataFrame],
    *,
    seed: int = 20260723,
    repetitions: int = 5000,
    block_length: int = 10,
) -> dict[str, Any]:
    control = daily["C00_L4_TOP2_EQUAL"]["net20"].to_numpy(float)
    hypothesis_ids = list(POLICY_IDS[1:])
    deltas = np.vstack([
        daily[key]["net20"].to_numpy(float) - control for key in hypothesis_ids
    ])
    n_h, n = deltas.shape
    rng = np.random.default_rng(seed)
    boot_means = np.empty((repetitions, n_h), dtype=float)
    starts = np.arange(n)
    blocks_needed = math.ceil(n / block_length)
    for rep in range(repetitions):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        indices = np.concatenate([
            (start + np.arange(block_length)) % n for start in sampled_starts
        ])[:n]
        boot_means[rep] = deltas[:, indices].mean(axis=1)
    point = deltas.mean(axis=1)
    ordinary_lower = np.quantile(boot_means, 0.10, axis=0)
    centered = boot_means - point
    max_centered = centered.max(axis=1)
    critical = float(np.quantile(max_centered, 0.90))
    output: dict[str, Any] = {
        "method": "circular moving-block bootstrap; unstudentized family max-mean",
        "seed": seed,
        "repetitions": repetitions,
        "block_length": block_length,
        "family_size": n_h,
        "family_critical90_pct": critical,
        "hypotheses": {},
    }
    for idx, key in enumerate(hypothesis_ids):
        adjusted_p = float(
            np.mean(max_centered >= point[idx])
        )
        output["hypotheses"][key] = {
            "point_uplift_net20_pct": float(point[idx]),
            "ordinary_one_sided90_lower_pct": float(ordinary_lower[idx]),
            "familywise_one_sided90_lower_pct": float(point[idx] - critical),
            "familywise_adjusted_p_approx": adjusted_p,
        }
    return output


def write_report(
    path: Path,
    result: dict[str, Any],
) -> None:
    metrics = result["policies"]
    bootstrap = result["multiplicity"]
    order = sorted(
        POLICY_IDS,
        key=lambda key: metrics[key]["net_mean_pct"]["20"],
        reverse=True,
    )
    lines = [
        "# v1.0 zero-base policy/universe screen",
        "",
        "## Status",
        "",
        "This is a retrospective mechanism screen on an already-viewed frozen panel. "
        "It cannot authorize production use.",
        "",
        "The previous rank-2-only and market-breadth switch mechanisms were prohibited. "
        "All sixteen preregistered new mechanisms are retained below.",
        "",
        "## Ranked results",
        "",
        "| Policy | Net20 | Net40 | Net60 | Months + | Blocks >=0 | Top20 removed | Mean names | Zero days | Family 90% lower uplift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in order:
        m = metrics[key]
        adjusted = bootstrap["hypotheses"].get(key, {})
        family_lower = adjusted.get("familywise_one_sided90_lower_pct")
        lines.append(
            f"| {key} | {m['net_mean_pct']['20']:+.4f}% "
            f"| {m['net_mean_pct']['40']:+.4f}% "
            f"| {m['net_mean_pct']['60']:+.4f}% "
            f"| {m['positive_months']}/{m['months']} "
            f"| {m['nonnegative_blocks']}/4 "
            f"| {m['winning_days_removed_net20_pct']['20']:+.4f}% "
            f"| {m['mean_candidates_per_day']:.2f} "
            f"| {m['zero_candidate_days']} "
            f"| {family_lower:+.4f}% |"
            if family_lower is not None
            else
            f"| {key} | {m['net_mean_pct']['20']:+.4f}% "
            f"| {m['net_mean_pct']['40']:+.4f}% "
            f"| {m['net_mean_pct']['60']:+.4f}% "
            f"| {m['positive_months']}/{m['months']} "
            f"| {m['nonnegative_blocks']}/4 "
            f"| {m['winning_days_removed_net20_pct']['20']:+.4f}% "
            f"| {m['mean_candidates_per_day']:.2f} "
            f"| {m['zero_candidate_days']} | n/a |"
        )
    lines.extend([
        "",
        "## Preregistered shortlist checks",
        "",
        "| Policy | Beat control | Net40 + | Top20 removed + | >=8 positive months | 4 blocks >=0 | Family lower >0 | All |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key in POLICY_IDS[1:]:
        checks = result["shortlist_checks"][key]
        values = [
            checks["net20_greater_than_control"],
            checks["net40_positive"],
            checks["top20_removed_net20_positive"],
            checks["at_least_8_positive_months"],
            checks["four_nonnegative_blocks"],
            checks["familywise90_lower_uplift_positive"],
        ]
        lines.append(
            f"| {key} | "
            + " | ".join("yes" if value else "no" for value in values)
            + f" | {'yes' if checks['all'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        result["decision"]["summary"],
        "",
        "The point winner is descriptive only. Any new mechanism requires a new, "
        "content-addressed forward shadow beginning after registration.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONPATH=src:. python /tmp/v10_policy_universe/runner.py",
        "```",
        "",
        f"- Panel SHA-256: `{result['input']['panel_sha256']}`",
        f"- Protocol SHA-256: `{result['input']['protocol_sha256']}`",
        f"- Runner SHA-256: `{result['input']['runner_sha256']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="/tmp/model_v07_corrected_panel.pkl")
    parser.add_argument("--protocol", default="/tmp/v10_policy_universe/protocol.json")
    parser.add_argument("--output", default="/tmp/v10_policy_universe/result.json")
    parser.add_argument("--report", default="/tmp/v10_policy_universe/report.md")
    parser.add_argument(
        "--scores-cache",
        default="/tmp/v10_policy_universe/scored_candidates.pkl",
    )
    args = parser.parse_args()
    panel_path = Path(args.panel).resolve()
    protocol_path = Path(args.protocol).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    protocol = read_protocol(protocol_path)
    actual_sha = sha256_file(panel_path)
    expected_sha = protocol["frozen_input"]["panel_sha256"]
    if actual_sha != expected_sha:
        raise RuntimeError(f"panel SHA mismatch: {actual_sha}")
    panel = joblib.load(panel_path, mmap_mode="r")
    panel["date"] = pd.to_datetime(panel["date"])
    if panel.duplicated(["date", "code"]).any():
        raise RuntimeError("panel contains duplicate date/code")
    source_dates = pd.to_datetime(
        panel["candidate_price_source_max_date"], errors="coerce"
    )
    violations = (
        source_dates.notna() & source_dates.ge(panel["date"])
    )
    if violations.any():
        raise RuntimeError("candidate features are not strictly prior")
    score_start, score_end = map(pd.Timestamp, protocol["frozen_input"]["score_period"])
    scores, folds = build_walk_forward_scores(
        panel,
        training_start=pd.Timestamp(protocol["frozen_input"]["training_start"]),
        score_start=score_start,
        score_end=score_end,
    )
    compact_score_columns = list(dict.fromkeys([
        "date", "code", "name", "label", "oc_return_pct",
        "score_G0", "score_L4", "score_L6",
        "pct_G0", "pct_L4", "pct_L6",
        "rank_G0", "rank_L4", "rank_L6",
        "ensemble_mean", "ensemble_worst", "ensemble_disagreement",
        "xrank_close_momentum_5", "xrank_close_momentum_20",
        "xrank_close_momentum_60", "xrank_atr14_pct",
        "xrank_prior_close_location_20", "overnight_mean_20",
        "oc_mean_20", "oc_std_20",
        "no_trade_rate_20", "flat_oc_rate_20",
        "has_event", "supportive_event", "adverse_event",
    ]))
    joblib.dump(
        scores.loc[:, compact_score_columns],
        Path(args.scores_cache).resolve(),
        compress=3,
    )
    picks_by_policy, state_audit = build_policy_picks(scores, panel)
    sessions = pd.DatetimeIndex(scores["date"].drop_duplicates().sort_values())
    metrics: dict[str, Any] = {}
    daily: dict[str, pd.DataFrame] = {}
    for policy_id in POLICY_IDS:
        metrics[policy_id], daily[policy_id] = policy_metrics(
            picks_by_policy[policy_id], sessions
        )
    multiplicity = bootstrap_multiplicity(daily)
    control_net20 = metrics["C00_L4_TOP2_EQUAL"]["net_mean_pct"]["20"]
    shortlist: dict[str, Any] = {}
    for policy_id in POLICY_IDS[1:]:
        m = metrics[policy_id]
        family_lower = multiplicity["hypotheses"][policy_id][
            "familywise_one_sided90_lower_pct"
        ]
        checks = {
            "net20_greater_than_control": m["net_mean_pct"]["20"] > control_net20,
            "net40_positive": m["net_mean_pct"]["40"] > 0,
            "top20_removed_net20_positive": (
                m["winning_days_removed_net20_pct"]["20"] > 0
            ),
            "at_least_8_positive_months": m["positive_months"] >= 8,
            "four_nonnegative_blocks": m["nonnegative_blocks"] == 4,
            "familywise90_lower_uplift_positive": family_lower > 0,
        }
        checks["all"] = all(checks.values())
        shortlist[policy_id] = checks
    point_winner = max(
        POLICY_IDS[1:], key=lambda key: metrics[key]["net_mean_pct"]["20"]
    )
    passers = [key for key, checks in shortlist.items() if checks["all"]]
    if passers:
        summary = (
            f"{len(passers)} mechanism(s) passed every retrospective screen: "
            f"{', '.join(passers)}. They remain shadow-only because there is no "
            "untouched historical holdout."
        )
    else:
        summary = (
            f"The point winner was {point_winner}, but no new mechanism passed "
            "all preregistered return, period, tail and multiplicity checks. "
            "The zero-base screen therefore does not justify replacing the control."
        )
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "authority": protocol["authority"],
        "input": {
            "panel": str(panel_path),
            "panel_sha256": actual_sha,
            "protocol": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "runner_sha256": sha256_file(Path(__file__)),
            "scores_cache": str(Path(args.scores_cache).resolve()),
            "scores_cache_sha256": sha256_file(Path(args.scores_cache).resolve()),
            "rows": int(len(panel)),
            "score_rows": int(len(scores)),
            "score_days": int(len(sessions)),
            "strictly_prior_candidate_feature_violations": 0,
        },
        "walk_forward": {
            "folds": folds,
            "fold_count": len(folds),
            "all_training_strictly_prior": all(item["strictly_prior"] for item in folds),
        },
        "policy_count": len(POLICY_IDS) - 1,
        "control": "C00_L4_TOP2_EQUAL",
        "policies": metrics,
        "multiplicity": multiplicity,
        "shortlist_checks": shortlist,
        "decision": {
            "point_winner": point_winner,
            "passing_all_retrospective_checks": passers,
            "production_change": False,
            "summary": summary,
        },
        "state_audit": {
            "rows": int(len(state_audit)),
            "gap_reference_is_strictly_prior": True,
            "posterior_reference_is_strictly_prior": True,
            "pair_correlation_is_strictly_prior_60_sessions": True,
        },
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(report_path, result)


if __name__ == "__main__":
    main()
