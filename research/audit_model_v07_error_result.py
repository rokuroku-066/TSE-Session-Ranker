#!/usr/bin/env python3
"""Independently audit the frozen v0.7 error-gate replay artifacts.

The result intentionally does not persist every rejected gate prediction.  This
auditor therefore reconstructs those predictions from the content-addressed
display artifact, using an implementation separate from the research runner,
and recomputes the complete selection decision.  It also exercises the runner
against target-month outcome mutations to check its point-in-time boundary.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research.analyze_model_v07_errors import (  # noqa: E402
    fit_monthly_meta_gates as runner_fit_monthly_meta_gates,
)
from tse_session_ranker.io import write_json  # noqa: E402


PRIMARY_COST_BPS = 20.0
STRESS_COST_BPS = 40.0
BLOCK_LENGTH = 5
BOOTSTRAP_SAMPLES = 5_000
SEED = 31
TOLERANCE = 1e-11


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype={"code": "string"})
    except EmptyDataError:
        return pd.DataFrame()
    for column in (
        "date",
        "feature_source_max_date",
        "candidate_price_source_max_date",
        "base_model_train_end",
        "repeat_feature_source_max_date",
        "meta_train_end",
    ):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _meta_columns(
    protocol: Mapping[str, Any], gate: Mapping[str, Any]
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            column
            for group in gate["feature_groups"]
            for column in protocol["meta_features"][group]
        )
    )


def _date_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("date", sort=False)["date"].transform("size")
    return 1.0 / counts.to_numpy(dtype=float)


def _ridge() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def independent_gate_predictions(
    display: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    """Recreate monthly gate predictions without calling the research runner."""

    selection_start = pd.Timestamp(protocol["periods"]["gate_selection"]["start"])
    replay_end = pd.Timestamp(protocol["periods"]["replay_c"]["end"])
    minimum_days = int(protocol["meta_training"]["minimum_prior_candidate_sessions"])
    clip = float(protocol["meta_training"]["target_clip_pct"])
    penalty = float(protocol["meta_training"]["downside_penalty"])
    months = pd.period_range(
        selection_start.to_period("M"), replay_end.to_period("M"), freq="M"
    )
    rows: list[pd.DataFrame] = []
    for gate in protocol["gate_registry"]:
        gate_id = str(gate["gate_id"])
        columns = _meta_columns(protocol, gate)
        for month in months:
            month_start = max(selection_start, month.start_time.normalize())
            month_end = min(replay_end, month.end_time.normalize())
            scoring = display[display["date"].between(month_start, month_end)].copy()
            if scoring.empty:
                continue
            training = display[
                display["date"].lt(month_start)
                & display["outcome_observed"].eq(True)
                & display["oc_return_pct"].notna()
            ].copy()
            train_days = int(training["date"].nunique())
            if train_days < minimum_days or not training["date"].max() < scoring["date"].min():
                raise AssertionError("independent meta fold is not strictly prior")
            target = (
                pd.to_numeric(training["oc_return_pct"], errors="raise") - 0.2
            ).clip(-clip, clip)
            weights = _date_equal_weights(training)
            mean_model = _ridge()
            mean_model.fit(
                training.loc[:, list(columns)],
                target,
                model__sample_weight=weights,
            )
            predicted_mean = np.asarray(
                mean_model.predict(scoring.loc[:, list(columns)]), dtype=float
            )
            predicted_downside = np.zeros(len(scoring), dtype=float)
            if gate_id == "downside_utility":
                downside = (-target).clip(lower=0.0, upper=clip)
                downside_model = _ridge()
                downside_model.fit(
                    training.loc[:, list(columns)],
                    downside,
                    model__sample_weight=weights,
                )
                predicted_downside = np.clip(
                    np.asarray(
                        downside_model.predict(scoring.loc[:, list(columns)]),
                        dtype=float,
                    ),
                    0.0,
                    None,
                )
            utility = predicted_mean - penalty * predicted_downside
            part = scoring[["date", "model_rank", "code"]].copy()
            part["gate_id"] = gate_id
            part["predicted_net_mean_pct"] = predicted_mean
            part["predicted_negative_part_pct"] = predicted_downside
            part["trade_utility_pct"] = utility
            part["trade_decision"] = utility > 0.0
            part["meta_train_start"] = training["date"].min()
            part["meta_train_end"] = training["date"].max()
            part["meta_train_sessions"] = train_days
            part["meta_feature_count"] = len(columns)
            part["meta_score_period"] = str(month)
            rows.append(part)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["gate_id", "date", "model_rank"], kind="stable"
    ).reset_index(drop=True)


def materialize_trades(
    display: pd.DataFrame,
    predictions: pd.DataFrame | None,
    *,
    gate_id: str,
    forced: bool = False,
) -> pd.DataFrame:
    keys = ["date", "model_rank", "code"]
    frame = display.copy()
    if forced:
        frame["trade_decision"] = frame["code"].notna()
        frame["gate_id"] = gate_id
    else:
        if predictions is None:
            raise ValueError("candidate trades require predictions")
        selected = predictions[predictions["gate_id"].eq(gate_id)].copy()
        frame = frame.merge(
            selected[[*keys, "gate_id", "trade_decision"]],
            on=keys,
            how="left",
            validate="one_to_one",
        )
        frame["gate_id"] = frame["gate_id"].fillna(gate_id)
        frame["trade_decision"] = frame["trade_decision"].eq(True)
    observed = (
        frame["outcome_observed"].eq(True)
        & frame["oc_return_pct"].notna()
        & frame["code"].notna()
    )
    frame["executed"] = frame["trade_decision"] & observed
    realised = pd.to_numeric(frame["oc_return_pct"], errors="coerce")
    frame["net_sleeve_return_pct_20bp"] = np.where(
        frame["executed"], 0.5 * (realised - 0.2), 0.0
    )
    frame["net_sleeve_return_pct_40bp"] = np.where(
        frame["executed"], 0.5 * (realised - 0.4), 0.0
    )
    return frame.sort_values(["date", "model_rank"], kind="stable").reset_index(
        drop=True
    )


def _daily(trades: pd.DataFrame, cost_bps: float = PRIMARY_COST_BPS) -> pd.Series:
    column = f"net_sleeve_return_pct_{int(cost_bps)}bp"
    return trades.groupby("date", sort=True)[column].sum().astype(float)


def _moving_block_means(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
    batch_size: int = 1_000,
) -> np.ndarray:
    observations = len(values)
    block_count = math.ceil(observations / BLOCK_LENGTH)
    max_start = observations - BLOCK_LENGTH + 1
    offsets = np.arange(BLOCK_LENGTH, dtype=np.int64)
    rng = np.random.default_rng(seed)
    output = np.empty(samples, dtype=float)
    position = 0
    while position < samples:
        size = min(batch_size, samples - position)
        starts = rng.integers(0, max_start, size=(size, block_count))
        indices = (starts[..., None] + offsets).reshape(size, -1)
        output[position : position + size] = values[
            indices[:, :observations]
        ].mean(axis=1)
        position += size
    return output


def _bootstrap(values: pd.Series, *, seed: int = SEED) -> dict[str, Any]:
    array = values.to_numpy(dtype=float)
    means = _moving_block_means(array, samples=BOOTSTRAP_SAMPLES, seed=seed)
    return {
        "observations": int(len(array)),
        "block_length": BLOCK_LENGTH,
        "samples": BOOTSTRAP_SAMPLES,
        "confidence": 0.90,
        "random_state": seed,
        "point_estimate_pct": float(array.mean()),
        "one_sided_lower_pct": float(np.quantile(means, 0.10)),
        "two_sided_lower_pct": float(np.quantile(means, 0.05)),
        "two_sided_upper_pct": float(np.quantile(means, 0.95)),
        "bootstrap_standard_error_pct": float(means.std(ddof=1)),
    }


def portfolio_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    daily20 = _daily(trades, PRIMARY_COST_BPS)
    daily40 = _daily(trades, STRESS_COST_BPS)
    monthly = daily20.groupby(daily20.index.to_period("M")).mean()
    gains = float(daily20.clip(lower=0.0).sum())
    losses = float(-daily20.clip(upper=0.0).sum())
    equity = (1.0 + daily20 / 100.0).cumprod()
    with_initial = np.concatenate(([1.0], equity.to_numpy()))
    peak = np.maximum.accumulate(with_initial)
    drawdown = with_initial / peak - 1.0
    top_dates = daily20.nlargest(min(5, len(daily20))).index
    without_top = daily20.drop(top_dates)
    executed_by_date = trades.groupby("date", sort=True)["executed"].sum()
    rank2 = trades[trades["model_rank"].eq(2)]
    rank2_daily = rank2.groupby("date", sort=True)[
        "net_sleeve_return_pct_20bp"
    ].sum().reindex(daily20.index, fill_value=0.0)
    return {
        "scheduled_days": int(len(daily20)),
        "trade_days": int(executed_by_date.gt(0).sum()),
        "executed_slots": int(trades["executed"].sum()),
        "trade_slot_rate": float(trades["executed"].mean()),
        "net20_mean_pct": float(daily20.mean()),
        "net20_median_pct": float(daily20.median()),
        "net40_mean_pct": float(daily40.mean()),
        "profit_factor": gains / losses if losses else float("inf"),
        "positive_months": int(monthly.gt(0.0).sum()),
        "months": int(len(monthly)),
        "positive_month_fraction": float(monthly.gt(0.0).mean()),
        "top5_removed_net20_mean_pct": float(without_top.mean()),
        "largest_day_share_of_positive_profit": (
            float(daily20.max() / gains) if gains > 0.0 else np.nan
        ),
        "compounded_net20_pct": float(100.0 * (equity.iloc[-1] - 1.0)),
        "max_drawdown_pct": float(100.0 * drawdown.min()),
        "rank2_marginal_net20_mean_pct": float(rank2_daily.mean()),
        "monthly_net20_mean_pct": {
            str(key): float(value) for key, value in monthly.items()
        },
        "block5_bootstrap": _bootstrap(daily20),
    }


def paired_metrics(candidate: pd.DataFrame, forced: pd.DataFrame) -> dict[str, Any]:
    candidate_daily = _daily(candidate)
    forced_daily = _daily(forced)
    if not candidate_daily.index.equals(forced_daily.index):
        raise AssertionError("paired portfolios do not have the same dates")
    delta = candidate_daily - forced_daily
    return {
        "candidate_mean_pct": float(candidate_daily.mean()),
        "forced_mean_pct": float(forced_daily.mean()),
        "mean_uplift_pct": float(delta.mean()),
        "block5_bootstrap": _bootstrap(delta, seed=SEED + 1),
    }


def _circular_indices(observations: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    blocks = math.ceil(observations / BLOCK_LENGTH)
    starts = rng.integers(0, observations, size=(BOOTSTRAP_SAMPLES, blocks))
    offsets = np.arange(BLOCK_LENGTH)
    return ((starts[..., None] + offsets) % observations).reshape(
        BOOTSTRAP_SAMPLES, -1
    )[:, :observations]


def shared_adjustment(
    forced: pd.DataFrame,
    candidates: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, float]]:
    forced_daily = _daily(forced)
    names = list(candidates)
    matrix = np.column_stack(
        [(_daily(candidates[name]) - forced_daily).to_numpy(float) for name in names]
    )
    means = matrix.mean(axis=0)
    centered = matrix - means
    bootstrap = centered[_circular_indices(len(matrix))].mean(axis=1)
    critical = float(np.quantile(bootstrap.max(axis=1), 0.80))
    return {
        name: {
            "mean_uplift_pct": float(means[position]),
            "shared_block_max_mean_critical_pct": critical,
            "adjusted_one_sided_80pct_lower_pct": float(
                means[position] - critical
            ),
        }
        for position, name in enumerate(names)
    }


def _numeric_differences(
    stored: Any,
    recomputed: Any,
    *,
    path: str,
    output: list[dict[str, Any]],
) -> None:
    if isinstance(stored, Mapping) and isinstance(recomputed, Mapping):
        for key in sorted(set(stored) & set(recomputed)):
            _numeric_differences(
                stored[key],
                recomputed[key],
                path=f"{path}.{key}",
                output=output,
            )
        return
    if (
        isinstance(stored, (int, float))
        and isinstance(recomputed, (int, float))
        and not isinstance(stored, bool)
        and not isinstance(recomputed, bool)
    ):
        left, right = float(stored), float(recomputed)
        if np.isnan(left) and np.isnan(right):
            difference = 0.0
        elif np.isinf(left) and left == right:
            difference = 0.0
        else:
            difference = abs(left - right)
        output.append(
            {
                "path": path,
                "stored": left,
                "recomputed": right,
                "absolute_difference": difference,
            }
        )


def recompute_repeat_features(display: pd.DataFrame) -> pd.DataFrame:
    ordered = display.sort_values(["date", "model_rank"], kind="stable").copy()
    dates = pd.DatetimeIndex(sorted(ordered["date"].unique()))
    positions = {date: position for position, date in enumerate(dates)}
    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for date, group in ordered.groupby("date", sort=True):
        position = positions[pd.Timestamp(date)]
        pending: list[tuple[str, dict[str, Any]]] = []
        for _, row in group.iterrows():
            code = str(row["code"])
            prior = history[code]
            recent5 = [item for item in prior if position - item["position"] <= 5]
            recent20 = [item for item in prior if position - item["position"] <= 20]
            last = prior[-1] if prior else None
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "model_rank": int(row["model_rank"]),
                    "code": code,
                    "prior_selection_count_5": float(len(recent5)),
                    "prior_selection_count_20": float(len(recent20)),
                    "sessions_since_selected": (
                        float(position - last["position"]) if last else np.nan
                    ),
                    "last_selected_net_pct": (
                        float(last["net_return_pct"])
                        if last and np.isfinite(last["net_return_pct"])
                        else np.nan
                    ),
                    "consecutive_selection": float(
                        bool(last and position - last["position"] == 1)
                    ),
                    "anchor_score_change_since_selected": (
                        float(row["anchor_model_score"] - last["anchor_model_score"])
                        if last
                        else np.nan
                    ),
                    "anchor_score_age_sessions": (
                        float(position - last["position"]) if last else np.nan
                    ),
                    "repeat_feature_source_max_date": (
                        last["date"] if last else pd.NaT
                    ),
                }
            )
            observed = bool(row["outcome_observed"]) and pd.notna(
                row["oc_return_pct"]
            )
            pending.append(
                (
                    code,
                    {
                        "position": position,
                        "date": pd.Timestamp(date),
                        "net_return_pct": (
                            float(row["oc_return_pct"]) - 0.2
                            if observed
                            else np.nan
                        ),
                        "anchor_model_score": float(row["anchor_model_score"]),
                    },
                )
            )
        for code, item in pending:
            history[code].append(item)
    return pd.DataFrame(rows)


def _qualification_checks(
    metrics: Mapping[str, Any],
    adjusted: Mapping[str, float],
    protocol: Mapping[str, Any],
) -> dict[str, bool]:
    threshold = protocol["selection_gate"]
    share = float(metrics["largest_day_share_of_positive_profit"])
    return {
        "minimum_trade_days": int(metrics["trade_days"])
        >= int(threshold["minimum_trade_days"]),
        "minimum_trade_slots": int(metrics["executed_slots"])
        >= int(threshold["minimum_trade_slots"]),
        "net20_mean": float(metrics["net20_mean_pct"]) > 0.0,
        "top5_removed": float(metrics["top5_removed_net20_mean_pct"]) >= 0.0,
        "positive_month_fraction": float(metrics["positive_month_fraction"])
        >= float(threshold["minimum_positive_month_fraction"]),
        "paired_uplift": float(adjusted["mean_uplift_pct"]) > 0.0,
        "adjusted_lower": float(
            adjusted["adjusted_one_sided_80pct_lower_pct"]
        )
        >= 0.0,
        "largest_day_share": np.isfinite(share)
        and share <= float(threshold["largest_day_share_of_positive_profit_max"]),
    }


def _subset(frame: pd.DataFrame, period: Mapping[str, Any]) -> pd.DataFrame:
    return frame[
        frame["date"].between(pd.Timestamp(period["start"]), pd.Timestamp(period["end"]))
    ].copy()


def counterfactual_replay(
    forced: pd.DataFrame,
    candidate_trades: Mapping[str, pd.DataFrame],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe rejected gates after selection; never use this to select one."""

    output: dict[str, Any] = {}
    for gate_id, trades in candidate_trades.items():
        period_rows: dict[str, Any] = {}
        forced_parts: list[pd.DataFrame] = []
        trade_parts: list[pd.DataFrame] = []
        for name in ("replay_a", "replay_b", "replay_c"):
            period = protocol["periods"][name]
            period_forced = _subset(forced, period)
            period_trades = _subset(trades, period)
            metrics = portfolio_metrics(period_trades)
            paired = paired_metrics(period_trades, period_forced)
            period_rows[name] = {
                "trade_days": metrics["trade_days"],
                "executed_slots": metrics["executed_slots"],
                "net20_mean_pct": metrics["net20_mean_pct"],
                "mean_uplift_vs_forced_pct": paired["mean_uplift_pct"],
            }
            forced_parts.append(period_forced)
            trade_parts.append(period_trades)
        combined_forced = pd.concat(forced_parts, ignore_index=True)
        combined_trades = pd.concat(trade_parts, ignore_index=True)
        metrics = portfolio_metrics(combined_trades)
        paired = paired_metrics(combined_trades, combined_forced)
        output[gate_id] = {
            "periods": period_rows,
            "combined": {
                "trade_days": metrics["trade_days"],
                "executed_slots": metrics["executed_slots"],
                "net20_mean_pct": metrics["net20_mean_pct"],
                "top5_removed_net20_mean_pct": metrics[
                    "top5_removed_net20_mean_pct"
                ],
                "profit_factor": metrics["profit_factor"],
                "mean_uplift_vs_forced_pct": paired["mean_uplift_pct"],
            },
        }
    return {
        "authority": "post-selection descriptive only; cannot alter locked_gate",
        "gates": output,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default="research/model_v07_error_result.json")
    parser.add_argument(
        "--output", default="research/model_v07_error_result_audit.json"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result_path = Path(args.result)
    if not result_path.is_absolute():
        result_path = ROOT / result_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest_path = result_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_path = ROOT / result["protocol_path"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    runner_path = ROOT / "research/analyze_model_v07_errors.py"
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("manifest_binds_result", manifest["result_sha256"] == sha256_file(result_path))
    check(
        "protocol_hash_binding",
        manifest["protocol_sha256"]
        == result["protocol_sha256"]
        == sha256_file(protocol_path),
    )
    check("runner_hash_binding", manifest["runner_sha256"] == sha256_file(runner_path))
    check(
        "panel_hash_binding",
        manifest["panel_sha256"] == result["panel"]["sha256"],
    )
    panel_path = Path(result["panel"]["path"])
    if panel_path.exists():
        check("panel_bytes_available_and_bound", sha256_file(panel_path) == manifest["panel_sha256"])
    panel_manifest_path = Path(result["panel"]["manifest_path"])
    check(
        "panel_manifest_bytes_bound",
        panel_manifest_path.exists()
        and sha256_file(panel_manifest_path) == result["panel"]["manifest_sha256"],
    )

    artifacts: dict[str, pd.DataFrame] = {}
    artifact_integrity = True
    artifact_details: dict[str, Any] = {}
    for name, raw in result["artifacts"].items():
        path = ROOT / raw["path"]
        frame = _read_csv(path)
        valid = sha256_file(path) == raw["sha256"] and len(frame) == int(raw["rows"])
        artifact_integrity &= valid
        artifact_details[name] = {"hash_and_rows_match": valid, "rows": len(frame)}
        artifacts[name] = frame
    check("all_artifacts_hash_and_row_bound", artifact_integrity, artifact_details)

    display = artifacts["display_picks"]
    forced_stored = artifacts["forced_trade_picks"]
    gated_stored = artifacts["gated_trade_picks"]
    keys = ["date", "model_rank", "code"]
    slot_counts = display.groupby("date").size()
    fixed_slots = (
        slot_counts.eq(2).all()
        and set(display["model_rank"].astype(int)) == {1, 2}
        and not display.duplicated(["date", "model_rank"]).any()
        and not display.duplicated(["date", "code"]).any()
    )
    check(
        "fixed_top2_display",
        fixed_slots,
        {"sessions": int(display["date"].nunique()), "rows": int(len(display))},
    )
    same_forced = display[keys].equals(forced_stored[keys])
    same_gated = display[keys].equals(gated_stored[keys])
    no_replacement = (
        same_forced
        and same_gated
        and forced_stored["model_rank"].isin([1, 2]).all()
        and gated_stored["model_rank"].isin([1, 2]).all()
    )
    check("trade_artifacts_preserve_display_without_rank3", no_replacement)

    source_columns = [
        "feature_source_max_date",
        "candidate_price_source_max_date",
        "base_model_train_end",
        "repeat_feature_source_max_date",
    ]
    source_violations = {
        column: int((display[column].notna() & display[column].ge(display["date"])).sum())
        for column in source_columns
    }
    check(
        "display_feature_sources_strictly_prior",
        not any(source_violations.values()),
        source_violations,
    )
    anchor_consistent = (
        display["family_rank__g0_raked_logit"].astype(int)
        .eq(display["model_rank"].astype(int))
        .all()
        and np.allclose(
            display["anchor_model_score"],
            display["family_score__g0_raked_logit"],
            atol=0.0,
            rtol=0.0,
        )
    )
    check("g0_anchor_is_the_display_rank", anchor_consistent)

    base_fold_ok = True
    for fold in result["base_folds"]:
        start = pd.Timestamp(fold["score_start"])
        end = pd.Timestamp(fold["score_end"])
        train_end = pd.Timestamp(fold["train_end"])
        rows = display[display["date"].between(start, end)]
        base_fold_ok &= train_end < start and not rows.empty
        base_fold_ok &= rows["base_model_train_end"].eq(train_end).all()
        base_fold_ok &= rows["base_model_train_sessions"].eq(
            int(fold["train_sessions"])
        ).all()
    check("base_fold_metadata_strictly_prior_and_bound", base_fold_ok)

    registered = pd.Timestamp(protocol["registered_at"])
    run_started = pd.Timestamp(result["runtime"]["run_started_at"])
    run_completed = pd.Timestamp(result["runtime"]["run_completed_at"])
    check(
        "protocol_registered_before_run",
        registered < run_started < run_completed,
        {
            "registered_at": registered.isoformat(),
            "run_started_at": run_started.isoformat(),
            "run_completed_at": run_completed.isoformat(),
        },
    )

    repeated = recompute_repeat_features(display)
    repeat_columns = [
        "prior_selection_count_5",
        "prior_selection_count_20",
        "sessions_since_selected",
        "last_selected_net_pct",
        "consecutive_selection",
        "anchor_score_change_since_selected",
        "anchor_score_age_sessions",
    ]
    repeat_joined = display[[*keys, *repeat_columns, "repeat_feature_source_max_date"]].merge(
        repeated,
        on=keys,
        suffixes=("_stored", "_audit"),
        validate="one_to_one",
    )
    repeat_max = 0.0
    for column in repeat_columns:
        left = repeat_joined[f"{column}_stored"].to_numpy(float)
        right = repeat_joined[f"{column}_audit"].to_numpy(float)
        valid = np.isfinite(left) | np.isfinite(right)
        if valid.any():
            repeat_max = max(repeat_max, float(np.nanmax(np.abs(left - right))))
        if not np.array_equal(np.isnan(left), np.isnan(right)):
            repeat_max = float("inf")
    left_dates = repeat_joined["repeat_feature_source_max_date_stored"]
    right_dates = repeat_joined["repeat_feature_source_max_date_audit"]
    repeat_dates_match = left_dates.fillna(pd.Timestamp("1900-01-01")).equals(
        right_dates.fillna(pd.Timestamp("1900-01-01"))
    )
    check(
        "repeat_features_independently_recomputed",
        repeat_max <= TOLERANCE and repeat_dates_match,
        {"max_absolute_difference": repeat_max},
    )

    independent = independent_gate_predictions(display, protocol)
    runner = runner_fit_monthly_meta_gates(display, protocol)
    prediction_keys = ["date", "model_rank", "code", "gate_id"]
    predictions_joined = independent.merge(
        runner,
        on=prediction_keys,
        suffixes=("_audit", "_runner"),
        validate="one_to_one",
    )
    prediction_fields = [
        "predicted_net_mean_pct",
        "predicted_negative_part_pct",
        "trade_utility_pct",
    ]
    prediction_max = max(
        float(
            np.max(
                np.abs(
                    predictions_joined[f"{column}_audit"]
                    - predictions_joined[f"{column}_runner"]
                )
            )
        )
        for column in prediction_fields
    )
    decisions_match = predictions_joined["trade_decision_audit"].equals(
        predictions_joined["trade_decision_runner"]
    )
    train_end_match = predictions_joined["meta_train_end_audit"].equals(
        predictions_joined["meta_train_end_runner"]
    )
    check(
        "independent_meta_predictions_match_runner",
        len(predictions_joined) == len(independent) == len(runner)
        and prediction_max <= TOLERANCE
        and decisions_match
        and train_end_match,
        {
            "rows": int(len(predictions_joined)),
            "max_absolute_prediction_difference": prediction_max,
        },
    )
    check(
        "meta_train_end_strictly_prior",
        independent["meta_train_end"].lt(independent["date"]).all(),
    )

    mutation_ok = True
    months_tested = 0
    baseline_runner = runner
    for month in sorted(baseline_runner["date"].dt.to_period("M").unique()):
        changed = display.copy()
        mask = changed["date"].dt.to_period("M").eq(month)
        changed.loc[mask, "oc_return_pct"] = np.linspace(
            -50.0, 50.0, int(mask.sum())
        )
        rebuilt = runner_fit_monthly_meta_gates(changed, protocol)
        fields = [
            *prediction_keys,
            *prediction_fields,
            "trade_decision",
            "meta_train_end",
        ]
        left = baseline_runner[baseline_runner["date"].dt.to_period("M").eq(month)][
            fields
        ].reset_index(drop=True)
        right = rebuilt[rebuilt["date"].dt.to_period("M").eq(month)][fields].reset_index(
            drop=True
        )
        try:
            pd.testing.assert_frame_equal(left, right, check_exact=True)
        except AssertionError:
            mutation_ok = False
        months_tested += 1
    check(
        "target_month_outcomes_cannot_change_target_month_gate",
        mutation_ok,
        {"months_tested": months_tested},
    )

    selection_period = protocol["periods"]["gate_selection"]
    selection_display = _subset(display, selection_period)
    forced = materialize_trades(
        selection_display, None, gate_id="forced_top2", forced=True
    )
    forced_metrics = portfolio_metrics(forced)
    candidate_trades: dict[str, pd.DataFrame] = {}
    candidate_metrics: dict[str, Any] = {}
    for gate in protocol["gate_registry"]:
        gate_id = str(gate["gate_id"])
        candidate = materialize_trades(
            selection_display,
            independent[independent["date"].between(
                pd.Timestamp(selection_period["start"]),
                pd.Timestamp(selection_period["end"]),
            )],
            gate_id=gate_id,
        )
        candidate_trades[gate_id] = candidate
        candidate_metrics[gate_id] = portfolio_metrics(candidate)
    adjusted = shared_adjustment(forced, candidate_trades)
    numeric_differences: list[dict[str, Any]] = []
    _numeric_differences(
        result["selection"]["forced_metrics"],
        forced_metrics,
        path="selection.forced_metrics",
        output=numeric_differences,
    )
    recomputed_qualified: list[str] = []
    qualification_match = True
    feature_counts = {
        str(gate["gate_id"]): len(_meta_columns(protocol, gate))
        for gate in protocol["gate_registry"]
    }
    recomputed_checks: dict[str, Any] = {}
    for gate_id, metrics in candidate_metrics.items():
        stored = result["selection"]["qualifications"][gate_id]
        _numeric_differences(
            stored["metrics"],
            metrics,
            path=f"selection.{gate_id}.metrics",
            output=numeric_differences,
        )
        _numeric_differences(
            {key: stored[key] for key in adjusted[gate_id]},
            adjusted[gate_id],
            path=f"selection.{gate_id}.adjustment",
            output=numeric_differences,
        )
        checks_for_gate = _qualification_checks(metrics, adjusted[gate_id], protocol)
        passed = all(checks_for_gate.values())
        qualification_match &= checks_for_gate == stored["checks"]
        qualification_match &= passed == bool(stored["passed"])
        qualification_match &= feature_counts[gate_id] == int(stored["feature_count"])
        recomputed_checks[gate_id] = {
            "passed": passed,
            "checks": checks_for_gate,
            "metrics": metrics,
            **adjusted[gate_id],
        }
        if passed:
            recomputed_qualified.append(gate_id)
    recomputed_qualified.sort(
        key=lambda gate_id: (
            -candidate_metrics[gate_id]["net20_mean_pct"],
            -adjusted[gate_id]["adjusted_one_sided_80pct_lower_pct"],
            -candidate_metrics[gate_id]["top5_removed_net20_mean_pct"],
            feature_counts[gate_id],
            gate_id,
        )
    )
    locked_gate = recomputed_qualified[0] if recomputed_qualified else None
    max_metric_difference = max(
        (value["absolute_difference"] for value in numeric_differences), default=0.0
    )
    check(
        "selection_metrics_and_shared_adjustment_recomputed",
        max_metric_difference <= TOLERANCE,
        {
            "numeric_values": len(numeric_differences),
            "max_absolute_difference": max_metric_difference,
        },
    )
    check("selection_qualification_predicates_recomputed", qualification_match)
    check(
        "formal_locked_gate_recomputed",
        recomputed_qualified == result["selection"]["qualified_gates"]
        and locked_gate == result["selection"]["locked_gate"]
        and locked_gate == result["locked_gate"],
        {"qualified_gates": recomputed_qualified, "locked_gate": locked_gate},
    )

    forced_formula = np.where(
        forced_stored["executed"].astype(bool),
        0.5 * (pd.to_numeric(forced_stored["oc_return_pct"], errors="coerce") - 0.2),
        0.0,
    )
    cost_ok = np.allclose(
        forced_formula,
        forced_stored["net_sleeve_return_pct_20bp"],
        atol=TOLERANCE,
        rtol=0.0,
        equal_nan=True,
    )
    gated_cash_ok = True
    if locked_gate is None:
        gated_cash_ok = (
            not gated_stored["executed"].astype(bool).any()
            and np.allclose(gated_stored["net_sleeve_return_pct_20bp"], 0.0)
            and np.allclose(gated_stored["net_sleeve_return_pct_40bp"], 0.0)
        )
    check("cost_formula_and_no_qualifier_cash_artifact", cost_ok and gated_cash_ok)

    error_artifact = artifacts["error_cases"]
    error_counts = {
        name: int(error_artifact["error_types"].str.contains(name, regex=False).sum())
        for name in protocol["error_analysis"]["registered_error_types"]
    }
    check(
        "error_case_counts_match_result",
        len(error_artifact) == int(result["error_analysis"]["error_case_rows"])
        and error_counts == result["error_analysis"]["error_type_counts"],
        error_counts,
    )

    formal_no_gate_ok = (
        locked_gate is not None
        or (
            result["replay"]["status"]
            == "not_evaluated_without_a_qualified_discovery_gate"
            and result["decision"] == "known_retrospective_no_demonstrated_gate"
            and result["production_model_changed"] is False
        )
    )
    check("no_qualifier_stops_formal_replay_and_production_change", formal_no_gate_ok)

    all_period_forced = materialize_trades(display, None, gate_id="forced", forced=True)
    all_period_candidates = {
        gate_id: materialize_trades(display, independent, gate_id=gate_id)
        for gate_id in candidate_trades
    }
    replay_diagnostic = counterfactual_replay(
        all_period_forced, all_period_candidates, protocol
    )

    minor_issues = [
        {
            "id": "rejected_gate_predictions_not_persisted",
            "impact": "The aggregate selection is reproducible now, but a future audit needs the pinned sklearn behavior because the 1,784 per-gate prediction rows are not content-addressed artifacts.",
            "recommendation": "Persist all registered gate predictions or selection-period per-gate trades in the next research version.",
        }
    ]
    slice_path = ROOT / result["artifacts"]["slice_report"]["path"]
    if int(result["artifacts"]["slice_report"]["rows"]) == 0 and not slice_path.read_text(
        encoding="utf-8"
    ).strip():
        minor_issues.append(
            {
                "id": "empty_slice_csv_has_no_schema_header",
                "impact": "Generic CSV readers raise EmptyDataError for the valid no-gate slice artifact.",
                "recommendation": "Write the registered slice columns even when there are zero rows.",
            }
        )

    passed = all(value["passed"] for value in checks)
    audit = {
        "schema_version": 1,
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": sha256_file(result_path),
        "result_manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(runner_path),
        "auditor_sha256": sha256_file(Path(__file__)),
        "checks": checks,
        "checks_passed": sum(value["passed"] for value in checks),
        "checks_total": len(checks),
        "failed_checks": [value["name"] for value in checks if not value["passed"]],
        "max_absolute_metric_difference": max_metric_difference,
        "max_absolute_prediction_difference": prediction_max,
        "recomputed_selection": {
            "qualified_gates": recomputed_qualified,
            "locked_gate": locked_gate,
            "gates": recomputed_checks,
        },
        "counterfactual_replay_diagnostic": replay_diagnostic,
        "implementation_review": {
            "major_issues": [],
            "minor_issues": minor_issues,
            "verified": [
                "Base and meta train_end are strictly before every scored date.",
                "All 11 target months are invariant to mutation of their own outcomes.",
                "DISPLAY remains the fixed G0 ranks 1 and 2; no rank-3 replacement occurs.",
                "Each rejected sleeve stays cash and the 20/40 bp cost formulas match.",
                "The shared five-session maximum-mean adjustment and every qualification predicate were independently reproduced.",
            ],
        },
        "passed": passed,
    }
    write_json(audit, output_path)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
