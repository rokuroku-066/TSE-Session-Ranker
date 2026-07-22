#!/usr/bin/env python3
"""Reproduce the canonical model-v0.8 feature and universe diagnostics.

The input panel and daily-price cache are caller-supplied so the registered
research can be replayed without any dependency on the original scratch paths.
No path in this runner touches the production artifact or inference policy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from research.screen_feature_candidates_v06 import max_t_adjusted_uplifts  # noqa: E402
from research.model_v08_feature_engine import (  # noqa: E402
    G0,
    add_derived_features,
    add_prior_close,
)
from tse_session_ranker.profit import (  # noqa: E402
    daily_portfolio_returns,
    profit_metrics,
)
from tse_session_ranker.research_models import date_equal_weights  # noqa: E402
from tse_session_ranker.validation import paired_moving_block_bootstrap  # noqa: E402


PROTOCOL_PATH = ROOT / "research/model_v08_feature_protocol.json"
EXPECTED_PROTOCOL_SHA256 = (
    "0620d86113bfe6672104e6bfb4d68bc1eaeaf9de0f3e8b6408156fcd83a70a7f"
)
DEFAULT_PANEL_PATH = Path("/tmp/model_v07_corrected_panel.pkl")
DEFAULT_DAILY_PRICES_PATH = ROOT / "jpx_daily_2024_2025.pkl"
DEFAULT_OUTPUT_PATH = ROOT / "research/model_v08_feature_result.json"

SCORE_START = pd.Timestamp("2024-07-01")
SCORE_END = pd.Timestamp("2025-07-31")
TRAIN_START = pd.Timestamp("2024-01-04")
PERIODS = {
    "discovery": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-31")),
    "confirmation_a": (pd.Timestamp("2024-11-01"), pd.Timestamp("2025-03-31")),
    "confirmation_b": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
    "overall": (SCORE_START, SCORE_END),
}
BLOCKS = {
    "A": PERIODS["discovery"],
    "B": (pd.Timestamp("2024-11-01"), pd.Timestamp("2024-12-30")),
    "C": (pd.Timestamp("2025-01-06"), pd.Timestamp("2025-03-31")),
    "D": PERIODS["confirmation_b"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def monthly_periods() -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    result = []
    for period in pd.period_range(SCORE_START.to_period("M"), SCORE_END.to_period("M"), freq="M"):
        result.append((
            max(SCORE_START, period.start_time.normalize()),
            min(SCORE_END, period.end_time.normalize()),
            str(period),
        ))
    return result


def desired_slots(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.MultiIndex.from_product(
        [sessions, (1, 2)], names=["date", "model_rank"]
    ).to_frame(index=False)


def fit_rank_ridge(
    training: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    alpha: float = 10.0,
) -> Pipeline:
    target = (
        training["oc_return_pct"]
        .groupby(training["date"], sort=False)
        .rank(method="average", pct=True)
        .mul(2.0)
        .sub(1.0)
    )
    estimator = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])
    estimator.fit(
        training.loc[:, list(feature_columns)],
        target,
        model__sample_weight=date_equal_weights(training),
    )
    return estimator


def score_recipe(
    panel: pd.DataFrame,
    feature_columns: tuple[str, ...],
    recipe_id: str,
    *,
    alpha: float = 10.0,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    keep = [
        "date", "code", "name", "label", "oc_return_pct",
        "price_eligible", "price_training_eligible", *feature_columns,
    ]
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for score_start, score_end, fold_id in monthly_periods():
        train_mask = (
            panel["date"].between(TRAIN_START, score_start - pd.Timedelta(days=1))
            & panel["price_training_eligible"].eq(True)
            & panel["oc_return_pct"].notna()
        )
        score_mask = (
            panel["date"].between(score_start, score_end)
            & panel["price_eligible"].eq(True)
        )
        training = panel.loc[train_mask, keep].copy()
        scoring = panel.loc[score_mask, keep].copy()
        if training["date"].nunique() < 60:
            raise ValueError(f"{recipe_id} lacks 60 prior training sessions in {fold_id}")
        estimator = fit_rank_ridge(training, feature_columns, alpha=alpha)
        score_values = np.asarray(
            estimator.predict(scoring.loc[:, list(feature_columns)]), dtype=float
        )
        if not np.isfinite(score_values).all():
            raise AssertionError(f"{recipe_id} produced non-finite scores")
        part = scoring.loc[:, ["date", "code", "name", "label", "oc_return_pct"]].copy()
        part["row_index"] = scoring.index.to_numpy(dtype="int64")
        part["model_score"] = score_values
        parts.append(part)
        folds.append({
            "fold": fold_id,
            "train_start": str(training["date"].min().date()),
            "train_end": str(training["date"].max().date()),
            "train_sessions": int(training["date"].nunique()),
            "train_rows": int(len(training)),
            "score_start": str(score_start.date()),
            "score_end": str(score_end.date()),
            "score_sessions": int(scoring["date"].nunique()),
            "score_rows": int(len(scoring)),
        })
        del training, scoring, estimator
        gc.collect()
    return pd.concat(parts, ignore_index=True), folds


def rank_predictions(
    predictions: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    recipe_id: str,
    candidate_mask: pd.Series | None = None,
) -> pd.DataFrame:
    frame = predictions
    if candidate_mask is not None:
        allowed = candidate_mask.reindex(frame["row_index"].to_numpy()).fillna(False).to_numpy(dtype=bool)
        frame = frame.loc[allowed].copy()
    ranked = frame.sort_values(
        ["date", "model_score", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranked = ranked.groupby("date", sort=True, as_index=False).head(2).copy()
    ranked["model_rank"] = ranked.groupby("date", sort=False).cumcount() + 1
    actual = ranked.loc[:, [
        "date", "model_rank", "code", "name", "model_score", "label", "oc_return_pct"
    ]]
    result = desired_slots(sessions).merge(
        actual,
        on=["date", "model_rank"],
        how="left",
        validate="one_to_one",
        sort=True,
    )
    result["recipe_id"] = recipe_id
    return result


def slice_picks(picks: pd.DataFrame, period: str) -> pd.DataFrame:
    start, end = PERIODS[period]
    return picks.loc[picks["date"].between(start, end)].copy()


def net_daily(picks: pd.DataFrame, cost_bps: float = 20.0) -> pd.Series:
    return (
        daily_portfolio_returns(picks, top_k=2, cost_bps=cost_bps)
        .set_index("date")["net_return_pct"]
        .sort_index()
    )


def summarize(picks: pd.DataFrame) -> dict[str, Any]:
    top2_20 = profit_metrics(picks, top_k=2, cost_bps=20.0)
    top2_40 = profit_metrics(picks, top_k=2, cost_bps=40.0)
    top1 = profit_metrics(picks, top_k=1, cost_bps=20.0)
    rank2 = picks.loc[picks["model_rank"].eq(2)].copy()
    rank2["model_rank"] = 1
    rank2_metrics = profit_metrics(rank2, top_k=1, cost_bps=20.0)
    daily = net_daily(picks)
    monthly = daily.groupby(daily.index.to_period("M")).mean()
    return {
        "days": int(len(daily)),
        "display_rate_rank1": float(picks.loc[picks["model_rank"].eq(1), "code"].notna().mean()),
        "display_rate_rank2": float(picks.loc[picks["model_rank"].eq(2), "code"].notna().mean()),
        "top2_net20": top2_20,
        "top2_net40": top2_40,
        "top1_net20": top1,
        "rank2_net20": rank2_metrics,
        "monthly_net20": {str(k): float(v) for k, v in monthly.items()},
        "positive_months": int(monthly.gt(0).sum()),
        "months": int(len(monthly)),
    }


def uplift(base: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    paired = pd.concat(
        [net_daily(base).rename("base"), net_daily(candidate).rename("candidate")],
        axis=1,
    )
    if paired.isna().any(axis=None):
        raise AssertionError("candidate and baseline days differ")
    delta = paired["candidate"] - paired["base"]
    monthly = delta.groupby(delta.index.to_period("M")).mean()
    base5 = set(paired["base"].nlargest(min(5, len(paired))).index)
    candidate5 = set(paired["candidate"].nlargest(min(5, len(paired))).index)
    interval = paired_moving_block_bootstrap(
        paired["candidate"],
        paired["base"],
        block_length=5,
        samples=5000,
        confidence=0.80,
        random_state=31,
    )
    return {
        "mean_pct": float(delta.mean()),
        "median_pct": float(delta.median()),
        "positive_month_fraction": float(monthly.gt(0).mean()),
        "monthly_pct": {str(k): float(v) for k, v in monthly.items()},
        "top5_removed_pct": float(
            paired.loc[~paired.index.isin(candidate5), "candidate"].mean()
            - paired.loc[~paired.index.isin(base5), "base"].mean()
        ),
        "paired_block5_80pct": asdict(interval),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=DEFAULT_PANEL_PATH,
        help="Immutable point-in-time feature panel (joblib pickle).",
    )
    parser.add_argument(
        "--daily-prices",
        type=Path,
        default=DEFAULT_DAILY_PRICES_PATH,
        help="Daily JPX OHLC cache used only to join the strictly-prior close.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Canonical JSON result path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("canonical v0.8 protocol changed after registration")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "canonical_ledger_of_three_pre_registered_retrospective_generations":
        raise RuntimeError("canonical v0.8 protocol has an unexpected status")
    for registration_id, registration in protocol["original_registrations"].items():
        registration_path = (ROOT / registration["path"]).resolve()
        if not registration_path.exists():
            raise FileNotFoundError(registration_path)
        if sha256_file(registration_path) != registration["sha256"]:
            raise RuntimeError(
                f"registered bytes changed for {registration_id}"
            )
    panel_path = args.panel.resolve()
    daily_prices_path = args.daily_prices.resolve()
    output_path = args.output.resolve()
    manifest_path = Path(str(panel_path) + ".manifest.json")
    for required in (panel_path, manifest_path, daily_prices_path):
        if not required.exists():
            raise FileNotFoundError(required)
    panel_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if panel_manifest.get("panel_file_sha256") != sha256_file(panel_path):
        raise RuntimeError("panel bytes differ from the caller-supplied manifest")

    panel = joblib.load(panel_path, mmap_mode="r")
    panel["code"] = panel["code"].astype(str)
    source_dates = pd.to_datetime(panel["candidate_price_source_max_date"])
    if not (source_dates.isna() | source_dates.lt(pd.to_datetime(panel["date"]))).all():
        raise AssertionError("candidate price source is not strictly prior")
    add_prior_close(panel, daily_prices_path)
    groups, universe_masks = add_derived_features(panel)
    if list(groups) != list(protocol["feature_groups"]):
        raise AssertionError("feature group order differs from protocol")
    if list(universe_masks) != list(protocol["universe_groups"]):
        raise AssertionError("universe group order differs from protocol")
    for group_id, columns in groups.items():
        if list(columns) != protocol["feature_groups"][group_id]["features"]:
            raise AssertionError(f"feature columns differ for {group_id}")

    score_sessions = pd.DatetimeIndex(sorted(
        pd.to_datetime(panel.loc[panel["date"].between(SCORE_START, SCORE_END), "date"]).unique()
    )).normalize()
    baseline_predictions, baseline_folds = score_recipe(panel, G0, "rank_ridge_G0")
    baseline_picks = rank_predictions(
        baseline_predictions, score_sessions, "rank_ridge_G0"
    )
    print("completed rank-Ridge G0", flush=True)

    feature_picks: dict[str, pd.DataFrame] = {}
    feature_folds: dict[str, Any] = {}
    for group_id, columns in groups.items():
        predictions, folds = score_recipe(
            panel, G0 + columns, f"rank_ridge_G0+{group_id}"
        )
        feature_picks[group_id] = rank_predictions(
            predictions, score_sessions, f"rank_ridge_G0+{group_id}"
        )
        feature_folds[group_id] = folds
        del predictions
        gc.collect()
        print(f"completed rank-Ridge {group_id}", flush=True)

    universe_picks = {
        universe_id: rank_predictions(
            baseline_predictions,
            score_sessions,
            f"rank_ridge_G0|{universe_id}",
            mask,
        )
        for universe_id, mask in universe_masks.items()
    }
    print("completed five rank-Ridge universe masks", flush=True)

    discovery_base = slice_picks(baseline_picks, "discovery")
    feature_max_t = max_t_adjusted_uplifts(
        discovery_base,
        {k: slice_picks(v, "discovery") for k, v in feature_picks.items()},
    )
    universe_max_t = max_t_adjusted_uplifts(
        discovery_base,
        {k: slice_picks(v, "discovery") for k, v in universe_picks.items()},
    )

    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "authority": protocol["authority"],
        "production_promotion_allowed": False,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "original_registration_sha256": {
            key: value["sha256"]
            for key, value in protocol["original_registrations"].items()
        },
        "registered_original_result_sha256": {
            "generation_1_sign_logit": "d26446a2506103f6eedb487bb185e66fecfac416973dc44243888169a868a6ae",
            "generation_2_rank_ridge_alpha10": "1068139f64fc9f441f4d2d3561ffd72026c3898526e563312b9fc40d43f751c4",
            "generation_3_rank_ridge_alpha1_h07": "8ac08b5e536c4ac1d6bffdef8ba7e07e9c1bc4bc60bfaa7ff299384a47ebeacf"
        },
        "panel_manifest_sha256": sha256_file(manifest_path),
        "panel_file_sha256": sha256_file(panel_path),
        "daily_prices_sha256": sha256_file(daily_prices_path),
        "source_qa": {
            "rows": int(len(panel)),
            "score_sessions": int(len(score_sessions)),
            "candidate_price_source_strictly_prior": True,
            "strict_prior_close_coverage_price_eligible": float(
                panel.loc[panel["price_eligible"].eq(True), "strict_prior_close"].notna().mean()
            ),
        },
        "generation_1_sign_logit_summary": {
            "registered_candidates": 17,
            "discovery_formal_survivors": [],
            "stable_observations_only": [
                "H01_cross_sectional_state",
                "H05_market_regime_switch",
                "H06_calendar_stock_interaction",
                "U02_middle_atr",
                "U05_tail_clean"
            ],
            "production_change": False
        },
        "generation_2_rank_ridge_alpha10": {
            "baseline": {"folds": baseline_folds, "periods": {}},
            "feature_candidates": {},
            "universe_candidates": {}
        },
    }
    generation2 = result["generation_2_rank_ridge_alpha10"]
    for period in PERIODS:
        generation2["baseline"]["periods"][period] = summarize(
            slice_picks(baseline_picks, period)
        )

    for family, picks_registry, max_t_registry in (
        ("feature", feature_picks, feature_max_t),
        ("universe", universe_picks, universe_max_t),
    ):
        target_registry = generation2[f"{family}_candidates"]
        for candidate_id, picks in picks_registry.items():
            period_report: dict[str, Any] = {}
            for period in PERIODS:
                candidate_period = slice_picks(picks, period)
                baseline_period = slice_picks(baseline_picks, period)
                period_report[period] = {
                    "metrics": summarize(candidate_period),
                    "uplift": uplift(baseline_period, candidate_period),
                }
            discovery = period_report["discovery"]["uplift"]
            discovery_checks = {
                "uplift_positive": discovery["mean_pct"] > 0,
                "top5_removed_nonnegative": discovery["top5_removed_pct"] >= 0,
                "positive_month_fraction_at_least_half": discovery["positive_month_fraction"] >= 0.5,
                "max_t_adjusted_lower_nonnegative": max_t_registry[candidate_id]["adjusted_one_sided_80pct_lower_pct"] >= 0,
            }
            overall = period_report["overall"]["metrics"]
            three_period_uplifts = [
                period_report[p]["uplift"]["mean_pct"]
                for p in ("discovery", "confirmation_a", "confirmation_b")
            ]
            block_net = {
                block: summarize(picks.loc[picks["date"].between(start, end)].copy())["top2_net20"]["net_mean_pct_at_cost"]
                for block, (start, end) in BLOCKS.items()
            }
            shadow_checks = {
                "overall_net20_positive": overall["top2_net20"]["net_mean_pct_at_cost"] > 0,
                "overall_net40_nonnegative": overall["top2_net40"]["net_mean_pct_at_cost"] >= 0,
                "overall_top5_removed_positive": overall["top2_net20"]["top5_removed_net_mean_pct"] > 0,
                "positive_months_at_least_9_of_13": overall["positive_months"] >= 9,
                "positive_blocks_at_least_3_of_4": sum(value > 0 for value in block_net.values()) >= 3,
                "rank2_net20_nonnegative": overall["rank2_net20"]["net_mean_pct_at_cost"] >= 0,
                "all_three_period_uplifts_positive": all(value > 0 for value in three_period_uplifts),
            }
            target_registry[candidate_id] = {
                "periods": period_report,
                "discovery_max_t": max_t_registry[candidate_id],
                "discovery_checks": discovery_checks,
                "discovery_qualified": all(discovery_checks.values()),
                "fixed_block_net20": block_net,
                "shadow_checks": shadow_checks,
                "shadow_survivor": all(shadow_checks.values()),
            }
            if family == "feature":
                target_registry[candidate_id]["folds"] = feature_folds[candidate_id]

    generation2["discovery_feature_survivors"] = [
        key for key, value in generation2["feature_candidates"].items()
        if value["discovery_qualified"]
    ]
    generation2["discovery_universe_survivors"] = [
        key for key, value in generation2["universe_candidates"].items()
        if value["discovery_qualified"]
    ]
    generation2["shadow_feature_survivors"] = [
        key for key, value in generation2["feature_candidates"].items()
        if value["shadow_survivor"]
    ]
    generation2["shadow_universe_survivors"] = [
        key for key, value in generation2["universe_candidates"].items()
        if value["shadow_survivor"]
    ]

    # Direct label-blind ranking audit: mutating scoring outcomes in the cached
    # prediction rows cannot alter ranks or selected codes.
    mutated = baseline_predictions.copy()
    mutated["label"] = 1.0 - mutated["label"].fillna(0.0)
    mutated["oc_return_pct"] = np.linspace(-99.0, 99.0, len(mutated))
    mutated_picks = rank_predictions(mutated, score_sessions, "mutation")
    result["outcome_mutation_rank_invariant"] = bool(
        baseline_picks[["date", "model_rank", "code"]].reset_index(drop=True).equals(
            mutated_picks[["date", "model_rank", "code"]].reset_index(drop=True)
        )
    )
    if not result["outcome_mutation_rank_invariant"]:
        raise AssertionError("scoring outcomes changed fixed predictions/ranks")

    # Generation 3: the only alpha-10 whole-period shadow survivor (H07) is
    # transferred without alteration to the independently registered alpha-1
    # anchor. No second feature or combination is tried.
    h07_columns = groups["H07_tradability_quality"]
    alpha1_base_predictions, alpha1_base_folds = score_recipe(
        panel, G0, "rank_ridge_alpha1_G0", alpha=1.0
    )
    alpha1_h07_predictions, alpha1_h07_folds = score_recipe(
        panel, G0 + h07_columns, "rank_ridge_alpha1_G0+H07", alpha=1.0
    )
    alpha1_base = rank_predictions(
        alpha1_base_predictions, score_sessions, "rank_ridge_alpha1_G0"
    )
    alpha1_h07 = rank_predictions(
        alpha1_h07_predictions, score_sessions, "rank_ridge_alpha1_G0+H07"
    )
    alpha1_periods: dict[str, Any] = {}
    for period in PERIODS:
        base_period = slice_picks(alpha1_base, period)
        candidate_period = slice_picks(alpha1_h07, period)
        alpha1_periods[period] = {
            "baseline": summarize(base_period),
            "candidate": summarize(candidate_period),
            "uplift": uplift(base_period, candidate_period),
        }
    alpha1_blocks = {
        block: summarize(
            alpha1_h07.loc[alpha1_h07["date"].between(start, end)].copy()
        )["top2_net20"]["net_mean_pct_at_cost"]
        for block, (start, end) in BLOCKS.items()
    }
    alpha1_overall = alpha1_periods["overall"]
    alpha1_metrics = alpha1_overall["candidate"]
    alpha1_checks = {
        "all_three_period_uplifts_positive": all(
            alpha1_periods[period]["uplift"]["mean_pct"] > 0
            for period in ("discovery", "confirmation_a", "confirmation_b")
        ),
        "paired_block5_80pct_lower_nonnegative": (
            alpha1_overall["uplift"]["paired_block5_80pct"]
            ["one_sided_lower_delta_pct"] >= 0
        ),
        "overall_net20_positive": (
            alpha1_metrics["top2_net20"]["net_mean_pct_at_cost"] > 0
        ),
        "overall_net40_nonnegative": (
            alpha1_metrics["top2_net40"]["net_mean_pct_at_cost"] >= 0
        ),
        "overall_top5_removed_positive": (
            alpha1_metrics["top2_net20"]["top5_removed_net_mean_pct"] > 0
        ),
        "positive_months_at_least_9_of_13": alpha1_metrics["positive_months"] >= 9,
        "positive_blocks_at_least_3_of_4": (
            sum(value > 0 for value in alpha1_blocks.values()) >= 3
        ),
        "rank2_net20_nonnegative": (
            alpha1_metrics["rank2_net20"]["net_mean_pct_at_cost"] >= 0
        ),
    }
    result["generation_3_rank_ridge_alpha1_h07"] = {
        "baseline_folds": alpha1_base_folds,
        "candidate_folds": alpha1_h07_folds,
        "periods": alpha1_periods,
        "candidate_fixed_block_net20": alpha1_blocks,
        "acceptance_checks": alpha1_checks,
        "accepted_as_prospective_shadow_only": all(alpha1_checks.values()),
        "combination_tested": False,
        "combination_reason": (
            "H07 was the only alpha-10 group with positive uplift in all three "
            "periods and every whole-period shadow check."
        ),
    }
    result["decision"] = {
        "feature_or_universe_modifier_selected": None,
        "generation_2_discovery_qualified_count": (
            len(generation2["discovery_feature_survivors"])
            + len(generation2["discovery_universe_survivors"])
        ),
        "generation_2_whole_period_shadow_observation": "H07_tradability_quality",
        "generation_3_h07_transfer_passed": all(alpha1_checks.values()),
        "production_changed": False,
        "stop_reason": (
            "No registered feature or candidate-universe change passed the "
            "multiple-testing-adjusted discovery gate. H07 improved alpha-10 "
            "but reduced alpha-1 G0 in discovery, confirmation B and overall, "
            "so its apparent gain is regularisation-specific rather than a "
            "portable feature effect."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "result": str(output_path),
        "discovery_feature_survivors": generation2["discovery_feature_survivors"],
        "discovery_universe_survivors": generation2["discovery_universe_survivors"],
        "alpha10_shadow_feature_survivors": generation2["shadow_feature_survivors"],
        "alpha1_h07_transfer_passed": all(alpha1_checks.values()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
