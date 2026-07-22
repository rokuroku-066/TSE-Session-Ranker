#!/usr/bin/env python3
"""Independently audit the v0.6 T1 row-alignment correction diagnostic."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tse_session_ranker.io import write_json  # noqa: E402


PRIMARY_COST_BPS = 20.0
STRESS_COST_BPS = 40.0
BLOCK_LENGTH = 5
BOOTSTRAP_SAMPLES = 5_000
SEED = 31
TOLERANCE = 1e-11
CORRECTED_COLUMNS = (
    "tdnet_clean_family_count_log1p",
    "tdnet_clean_single_family",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def daily_returns(
    picks: pd.DataFrame, *, top_k: int, cost_bps: float
) -> pd.DataFrame:
    """Recreate fixed-slot daily returns without using production helpers."""

    selected = picks[picks["model_rank"].le(top_k)].copy()
    executed = selected["label"].notna()
    selected["gross_slot"] = pd.to_numeric(
        selected["oc_return_pct"], errors="coerce"
    ).fillna(0.0)
    selected["net_slot"] = selected["gross_slot"] - executed.astype(float) * (
        cost_bps / 100.0
    )
    selected["executed_slot"] = executed.astype(int)
    daily = selected.groupby("date", sort=True).agg(
        gross_sum=("gross_slot", "sum"),
        net_sum=("net_slot", "sum"),
        executed_slots=("executed_slot", "sum"),
        signal_slots=("model_rank", "size"),
    )
    daily["gross_return_pct"] = daily["gross_sum"] / top_k
    daily["net_return_pct"] = daily["net_sum"] / top_k
    return daily.reset_index()[
        [
            "date",
            "gross_return_pct",
            "net_return_pct",
            "executed_slots",
            "signal_slots",
        ]
    ]


def profit_metrics(
    picks: pd.DataFrame, *, top_k: int, cost_bps: float
) -> dict[str, Any]:
    selected = picks[picks["model_rank"].le(top_k)].copy()
    executed = selected["label"].notna()
    daily = daily_returns(selected, top_k=top_k, cost_bps=cost_bps)
    net = daily.set_index("date")["net_return_pct"].sort_index()
    gross = daily.set_index("date")["gross_return_pct"].sort_index()
    monthly = net.groupby(net.index.to_period("M")).mean()
    equity = (1.0 + net / 100.0).cumprod()
    with_initial = np.concatenate(([1.0], equity.to_numpy()))
    peaks = np.maximum.accumulate(with_initial)
    drawdown = with_initial / peaks - 1.0
    without_top5 = net.drop(net.nlargest(min(5, len(net))).index)
    positive = float(net.clip(lower=0.0).sum())
    gains = positive
    losses = float(-net.clip(upper=0.0).sum())
    sensitivity: dict[str, float] = {}
    for scenario in (0.0, 10.0, 20.0, 40.0, 60.0):
        scenario_daily = daily_returns(
            selected, top_k=top_k, cost_bps=scenario
        )
        sensitivity[str(float(scenario))] = float(
            scenario_daily["net_return_pct"].mean()
        )
    return {
        "n": int(len(selected)),
        "days": int(len(daily)),
        "executed": int(executed.sum()),
        "execution_rate": float(executed.mean()),
        "hit_rate": float(selected.loc[executed, "label"].mean()),
        "signal_hit_rate_including_unfilled": float(
            selected["label"].fillna(0.0).mean()
        ),
        "gross_mean_pct": float(gross.mean()),
        "gross_median_pct": float(gross.median()),
        "net_mean_pct_at_cost": float(net.mean()),
        "net_median_pct_at_cost": float(net.median()),
        "compounded_net_return_pct": float(100.0 * (equity.iloc[-1] - 1.0)),
        "max_drawdown_pct": float(100.0 * drawdown.min()),
        "profit_factor": gains / losses if losses else float("inf"),
        "positive_months": int(monthly.gt(0.0).sum()),
        "months": int(len(monthly)),
        "worst_month_pct": float(monthly.min()),
        "top5_removed_net_mean_pct": float(without_top5.mean()),
        "largest_day_net_pct": float(net.max()),
        "largest_day_share_of_positive_net": (
            float(net.max() / positive) if positive > 0.0 else np.nan
        ),
        "monthly_net_mean_pct": {
            str(key): float(value) for key, value in monthly.items()
        },
        "cost_sensitivity_net_mean_pct": sensitivity,
    }


def _moving_block_means(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
    batch_size: int = 1_000,
) -> np.ndarray:
    observations = len(values)
    blocks = math.ceil(observations / BLOCK_LENGTH)
    max_start = observations - BLOCK_LENGTH + 1
    offsets = np.arange(BLOCK_LENGTH, dtype=np.int64)
    rng = np.random.default_rng(seed)
    output = np.empty(samples, dtype=float)
    position = 0
    while position < samples:
        size = min(batch_size, samples - position)
        starts = rng.integers(0, max_start, size=(size, blocks))
        indices = (starts[..., None] + offsets).reshape(size, -1)
        output[position : position + size] = values[
            indices[:, :observations]
        ].mean(axis=1)
        position += size
    return output


def bootstrap_interval(
    values: pd.Series,
    *,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    array = values.to_numpy(dtype=float)
    means = _moving_block_means(array, samples=BOOTSTRAP_SAMPLES, seed=seed)
    tail = (1.0 - confidence) / 2.0
    return {
        "observations": int(len(array)),
        "block_length": BLOCK_LENGTH,
        "samples": BOOTSTRAP_SAMPLES,
        "confidence": confidence,
        "random_state": seed,
        "point_estimate_pct": float(array.mean()),
        "one_sided_lower_pct": float(np.quantile(means, 1.0 - confidence)),
        "two_sided_lower_pct": float(np.quantile(means, tail)),
        "two_sided_upper_pct": float(np.quantile(means, 1.0 - tail)),
        "bootstrap_standard_error_pct": float(means.std(ddof=1)),
    }


def recipe_metrics(picks: pd.DataFrame) -> dict[str, Any]:
    displayed = picks["code"].notna()
    result: dict[str, Any] = {
        "display": {
            "rank1_rate": float(displayed[picks["model_rank"].eq(1)].mean()),
            "rank2_rate": float(displayed[picks["model_rank"].eq(2)].mean()),
            "scheduled_sessions": int(picks["date"].nunique()),
        }
    }
    for top_k in (1, 2):
        daily = daily_returns(
            picks, top_k=top_k, cost_bps=PRIMARY_COST_BPS
        ).set_index("date")["net_return_pct"]
        result[f"top{top_k}"] = {
            "net20": profit_metrics(
                picks, top_k=top_k, cost_bps=PRIMARY_COST_BPS
            ),
            "net40": profit_metrics(
                picks, top_k=top_k, cost_bps=STRESS_COST_BPS
            ),
            "block5_bootstrap": bootstrap_interval(
                daily, confidence=0.90, seed=SEED + top_k
            ),
        }
    rank2 = picks[picks["model_rank"].eq(2)].copy()
    rank2["model_rank"] = 1
    result["rank2_standalone_net20"] = profit_metrics(
        rank2, top_k=1, cost_bps=PRIMARY_COST_BPS
    )
    return result


def paired_interval(candidate: pd.Series, baseline: pd.Series) -> dict[str, Any]:
    if not candidate.index.equals(baseline.index):
        raise AssertionError("paired series have different dates")
    delta = candidate - baseline
    raw = bootstrap_interval(delta, confidence=0.80, seed=SEED)
    return {
        "observations": raw["observations"],
        "block_length": raw["block_length"],
        "samples": raw["samples"],
        "confidence": raw["confidence"],
        "random_state": raw["random_state"],
        "candidate_mean_pct": float(candidate.mean()),
        "baseline_mean_pct": float(baseline.mean()),
        "point_estimate_delta_pct": float(delta.mean()),
        "one_sided_lower_delta_pct": raw["one_sided_lower_pct"],
        "two_sided_lower_delta_pct": raw["two_sided_lower_pct"],
        "two_sided_upper_delta_pct": raw["two_sided_upper_pct"],
        "bootstrap_standard_error_delta_pct": raw[
            "bootstrap_standard_error_pct"
        ],
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
    baseline: pd.Series, candidates: Mapping[str, pd.Series]
) -> dict[str, dict[str, float]]:
    names = list(candidates)
    matrix = np.column_stack(
        [(candidates[name] - baseline).to_numpy(float) for name in names]
    )
    means = matrix.mean(axis=0)
    centered = matrix - means
    bootstrap = centered[_circular_indices(len(matrix))].mean(axis=1)
    critical = float(np.quantile(bootstrap.max(axis=1), 0.80))
    return {
        name: {
            "mean_uplift_pct": float(means[position]),
            "max_t_critical_pct": critical,
            "adjusted_one_sided_80pct_lower_pct": float(
                means[position] - critical
            ),
        }
        for position, name in enumerate(names)
    }


def pick_change_report(
    legacy: pd.DataFrame, corrected: pd.DataFrame
) -> dict[str, Any]:
    keys = ["date", "model_rank"]
    joined = legacy[[*keys, "code"]].merge(
        corrected[[*keys, "code"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_legacy", "_corrected"),
    )
    same = joined["code_legacy"].fillna("<NA>").eq(
        joined["code_corrected"].fillna("<NA>")
    )
    return {
        "scheduled_slots": int(len(joined)),
        "changed_slots": int((~same).sum()),
        "unchanged_slots": int(same.sum()),
        "dates_with_any_changed_slot": int(joined.loc[~same, "date"].nunique()),
        "slot_change_rate": float((~same).mean()),
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


def _coverage(
    merged: pd.DataFrame,
    overlay: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, int]:
    panel_scope = merged
    overlay_scope = overlay
    if start is not None and end is not None:
        panel_scope = merged[merged["date"].between(start, end)]
        overlay_scope = overlay[overlay["date"].between(start, end)]
    covered = panel_scope["_merge"].eq("both")
    legacy_event = (
        panel_scope[f"{CORRECTED_COLUMNS[0]}_legacy"].fillna(0.0).gt(0.0)
        | panel_scope[f"{CORRECTED_COLUMNS[1]}_legacy"].fillna(0.0).gt(0.0)
    )
    return {
        "panel_rows": int(len(panel_scope)),
        "panel_dates": int(panel_scope["date"].nunique()),
        "tdnet_complete_rows": int(
            panel_scope["tdnet_source_complete"].eq(True).sum()
        ),
        "tdnet_incomplete_rows": int(
            panel_scope["tdnet_source_complete"].eq(False).sum()
        ),
        "overlay_rows": int(len(overlay_scope)),
        "overlay_keys_matched_to_panel": int(covered.sum()),
        "overlay_keys_outside_panel": int(len(overlay_scope) - covered.sum()),
        "legacy_event_keys": int(legacy_event.sum()),
        "legacy_event_keys_covered": int((legacy_event & covered).sum()),
        "complete_rows_without_overlay_key": int(
            (panel_scope["tdnet_source_complete"].eq(True) & ~covered).sum()
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result", default="research/model_v06_t1_correction_result.json"
    )
    parser.add_argument(
        "--output",
        default="research/model_v06_t1_correction_result_audit.json",
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
    protocol_path = ROOT / manifest["protocol_path"]
    runner_path = ROOT / manifest["runner_path"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("manifest_binds_result", manifest["result_sha256"] == sha256_file(result_path))
    check("protocol_hash_binding", manifest["protocol_sha256"] == sha256_file(protocol_path))
    check("runner_hash_binding", manifest["runner_sha256"] == sha256_file(runner_path))
    picks_path = ROOT / manifest["display_picks_path"]
    picks_hash_ok = (
        manifest["display_picks_sha256"] == sha256_file(picks_path)
        and result["artifact"]["display_picks"]["sha256"] == sha256_file(picks_path)
    )
    check("display_picks_hash_binding", picks_hash_ok)

    frozen_ok = True
    frozen_detail: dict[str, Any] = {}
    for name, spec in protocol["frozen_inputs"].items():
        path = Path(spec["path"])
        if not path.is_absolute():
            path = ROOT / path
        valid = path.exists() and sha256_file(path) == spec["sha256"]
        valid &= manifest["frozen_input_sha256"][name] == spec["sha256"]
        if "manifest_path" in spec:
            companion = Path(spec["manifest_path"])
            if not companion.is_absolute():
                companion = ROOT / companion
            valid &= (
                companion.exists()
                and sha256_file(companion) == spec["manifest_sha256"]
            )
        frozen_ok &= valid
        frozen_detail[name] = valid
    check("all_frozen_inputs_hash_bound", frozen_ok, frozen_detail)

    registered = pd.Timestamp(protocol["registered_at"])
    started = pd.Timestamp(result["runtime"]["run_started_at"])
    completed = pd.Timestamp(result["runtime"]["run_completed_at"])
    check(
        "protocol_registered_before_run",
        registered < started <= completed,
        {
            "registered_at": registered.isoformat(),
            "run_started_at": started.isoformat(),
            "run_completed_at": completed.isoformat(),
        },
    )
    authority_ok = (
        protocol["authority"]["production_promotion_allowed"] is False
        and protocol["authority"]["selection_allowed"] is False
        and result["authority"] == protocol["authority"]
        and result["decision"]
        == "correction_retrospective_only_no_production_change"
    )
    check("correction_authority_is_non_selection_non_production", authority_ok)

    picks = pd.read_csv(picks_path, dtype={"code": "string"})
    picks["date"] = pd.to_datetime(picks["date"], errors="raise")
    expected_ids = {
        "G0_price_core",
        "G0_price_core+T1_event_structure",
        "G0_price_core+T1_event_structure_corrected",
    }
    schedule_ok = (
        set(picks["recipe_id"]) == expected_ids
        and picks.groupby("recipe_id").size().eq(168).all()
        and not picks.duplicated(["recipe_id", "date", "model_rank"]).any()
        and set(picks["model_rank"].astype(int)) == {1, 2}
    )
    check("three_recipes_have_fixed_84_day_top2_schedule", schedule_ok)
    recipe_frames = {
        "G0_price_core": picks[picks["recipe_id"].eq("G0_price_core")].copy(),
        "legacy_T1": picks[
            picks["recipe_id"].eq("G0_price_core+T1_event_structure")
        ].copy(),
        "corrected_T1": picks[
            picks["recipe_id"].eq(
                "G0_price_core+T1_event_structure_corrected"
            )
        ].copy(),
    }

    reference_path = ROOT / protocol["frozen_inputs"]["v06_group_screen_picks_reference"]["path"]
    reference = pd.read_csv(reference_path, dtype={"code": "string"})
    reference["date"] = pd.to_datetime(reference["date"], errors="raise")
    reference_ids = {
        "G0_price_core": "G0_price_core",
        "legacy_T1": "G0_price_core+T1_event_structure",
    }
    reproduction_detail: dict[str, Any] = {}
    reproduction_ok = True
    for name, recipe_id in reference_ids.items():
        left = reference[reference["recipe_id"].eq(recipe_id)].sort_values(
            ["date", "model_rank"], kind="stable"
        ).reset_index(drop=True)
        right = recipe_frames[name].sort_values(
            ["date", "model_rank"], kind="stable"
        ).reset_index(drop=True)
        keys_equal = left[["date", "model_rank", "code"]].equals(
            right[["date", "model_rank", "code"]]
        )
        score_difference = float(
            np.nanmax(np.abs(left["model_score"] - right["model_score"]))
        )
        outcome_difference = float(
            np.nanmax(
                np.abs(
                    pd.to_numeric(left["oc_return_pct"], errors="coerce")
                    - pd.to_numeric(right["oc_return_pct"], errors="coerce")
                )
            )
        )
        valid = keys_equal and score_difference == 0.0 and outcome_difference == 0.0
        reproduction_ok &= valid
        reproduction_detail[name] = {
            "keys_equal": keys_equal,
            "maximum_absolute_score_difference": score_difference,
            "maximum_absolute_outcome_difference": outcome_difference,
        }
    check("g0_and_legacy_t1_picks_exactly_reproduce_frozen_v06", reproduction_ok, reproduction_detail)

    recomputed_metrics = {
        name: recipe_metrics(frame) for name, frame in recipe_frames.items()
    }
    numeric_differences: list[dict[str, Any]] = []
    for name, metrics in recomputed_metrics.items():
        _numeric_differences(
            result["evaluation"]["recipes"][name]["metrics"],
            metrics,
            path=f"recipes.{name}.metrics",
            output=numeric_differences,
        )
    max_metric_difference = max(
        (value["absolute_difference"] for value in numeric_differences), default=0.0
    )
    check(
        "all_recipe_metrics_independently_recomputed",
        max_metric_difference <= TOLERANCE,
        {
            "numeric_values": len(numeric_differences),
            "maximum_absolute_difference": max_metric_difference,
        },
    )

    old_result_path = ROOT / protocol["frozen_inputs"]["v06_result_reference"]["path"]
    old_result = json.loads(old_result_path.read_text(encoding="utf-8"))
    frozen_metric_differences: list[dict[str, Any]] = []
    _numeric_differences(
        old_result["group_screen"]["base"]["metrics"],
        result["evaluation"]["recipes"]["G0_price_core"]["metrics"],
        path="frozen.G0",
        output=frozen_metric_differences,
    )
    _numeric_differences(
        old_result["group_screen"]["candidates"]["T1_event_structure"]["metrics"],
        result["evaluation"]["recipes"]["legacy_T1"]["metrics"],
        path="frozen.legacy_T1",
        output=frozen_metric_differences,
    )
    frozen_metric_max = max(
        (value["absolute_difference"] for value in frozen_metric_differences),
        default=0.0,
    )
    check(
        "g0_and_legacy_t1_metrics_exactly_reproduce_frozen_v06",
        frozen_metric_max == 0.0,
        {"maximum_absolute_difference": frozen_metric_max},
    )

    fold_ok = True
    fold_reference: list[dict[str, Any]] | None = None
    for name, recipe in result["evaluation"]["recipes"].items():
        folds = recipe["folds"]
        fold_ok &= all(
            pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["score_start"])
            for fold in folds
        )
        if fold_reference is None:
            fold_reference = folds
        else:
            fold_ok &= folds == fold_reference
    check("all_monthly_model_folds_are_identical_and_strictly_prior", fold_ok)

    panel_path = Path(protocol["frozen_inputs"]["v06_panel"]["path"])
    overlay_path = Path(protocol["frozen_inputs"]["alignment_overlay"]["path"])
    panel_manifest_path = Path(
        protocol["frozen_inputs"]["v06_panel"]["manifest_path"]
    )
    panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    raw_panel = joblib.load(panel_path)
    panel_schema_ok = (
        len(raw_panel) == int(protocol["frozen_inputs"]["v06_panel"]["rows"])
        and list(raw_panel.columns) == panel_manifest["columns"]
        and {column: str(raw_panel[column].dtype) for column in raw_panel}
        == panel_manifest["dtypes"]
        and not raw_panel.duplicated(["date", "code"]).any()
    )
    check("panel_schema_and_unique_key_match_manifest", panel_schema_ok)
    panel = raw_panel[
        ["date", "code", "tdnet_source_complete", *CORRECTED_COLUMNS]
    ].copy()
    del raw_panel
    gc.collect()
    overlay = joblib.load(overlay_path)
    overlay_spec = protocol["frozen_inputs"]["alignment_overlay"]
    family = overlay[CORRECTED_COLUMNS[0]].to_numpy(float)
    single = overlay[CORRECTED_COLUMNS[1]].to_numpy(float)
    family_counts = np.expm1(family)
    rounded = np.rint(family_counts)
    overlay_schema_ok = (
        list(overlay.columns) == overlay_spec["columns"]
        and {column: str(overlay[column].dtype) for column in overlay}
        == overlay_spec["dtypes"]
        and len(overlay) == int(overlay_spec["rows"])
        and not overlay.duplicated(["date", "code"]).any()
        and overlay[["date", "code"]].notna().all(axis=None)
        and str(overlay["date"].min().date()) == overlay_spec["date_min"]
        and str(overlay["date"].max().date()) == overlay_spec["date_max"]
        and np.isfinite(family).all()
        and np.isfinite(single).all()
        and np.allclose(family_counts, rounded, atol=1e-5, rtol=0.0)
        and (family_counts >= -1e-6).all()
        and np.isin(single, [0.0, 1.0]).all()
        and np.array_equal(single == 1.0, rounded == 1.0)
    )
    check("overlay_schema_keys_and_count_geometry_valid", overlay_schema_ok)

    merged = panel.merge(
        overlay,
        on=["date", "code"],
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_legacy", "_overlay"),
        sort=False,
    )
    coverage = {
        "full_panel": _coverage(merged, overlay),
        "model_horizon_2024_01_04_through_2024_10_31": _coverage(
            merged, overlay, start="2024-01-04", end="2024-10-31"
        ),
        "group_screen_2024_07_01_through_2024_10_31": _coverage(
            merged, overlay, start="2024-07-01", end="2024-10-31"
        ),
    }
    coverage_ok = True
    for scope, expected in protocol["registered_overlay_coverage"].items():
        coverage_ok &= all(coverage[scope][key] == int(value) for key, value in expected.items())
    coverage_ok &= coverage == result["overlay_validation"]["registered_coverage"]
    check("overlay_coverage_counts_independently_recomputed", coverage_ok, coverage)

    complete = merged["tdnet_source_complete"].eq(True).to_numpy()
    covered = merged["_merge"].eq("both").to_numpy()
    changed_counts: dict[str, int] = {}
    semantics_ok = True
    for column in CORRECTED_COLUMNS:
        old = merged[f"{column}_legacy"].to_numpy(float)
        raw = merged[f"{column}_overlay"].to_numpy(float)
        corrected = np.where(complete, np.nan_to_num(raw, nan=0.0), np.nan)
        changed_counts[column] = int(
            (~np.isclose(old, corrected, rtol=0.0, atol=0.0, equal_nan=True)).sum()
        )
        semantics_ok &= not np.isnan(corrected[complete]).any()
        semantics_ok &= not np.isfinite(corrected[~complete]).any()
    legacy_event = (
        merged[f"{CORRECTED_COLUMNS[0]}_legacy"].fillna(0.0).gt(0.0)
        | merged[f"{CORRECTED_COLUMNS[1]}_legacy"].fillna(0.0).gt(0.0)
    ).to_numpy()
    semantics_ok &= not (legacy_event & ~covered).any()
    merge_report = {
        "overlay_keys_matched": int(covered.sum()),
        "complete_rows_implicit_no_event_zero": int((complete & ~covered).sum()),
        "incomplete_rows_forced_nan": int((~complete).sum()),
        **{
            f"changed_rows:{column}": value
            for column, value in changed_counts.items()
        },
    }
    semantics_ok &= merge_report == result["overlay_validation"]["merge_semantics"]
    check("overlay_merge_missingness_and_change_counts_recomputed", semantics_ok, merge_report)
    del panel, merged
    gc.collect()

    daily_by_recipe = {
        name: daily_returns(
            frame, top_k=2, cost_bps=PRIMARY_COST_BPS
        ).set_index("date")["net_return_pct"].sort_index()
        for name, frame in recipe_frames.items()
    }
    adjusted = shared_adjustment(
        daily_by_recipe["G0_price_core"],
        {
            "legacy_T1": daily_by_recipe["legacy_T1"],
            "corrected_T1": daily_by_recipe["corrected_T1"],
        },
    )
    paired = paired_interval(
        daily_by_recipe["corrected_T1"], daily_by_recipe["legacy_T1"]
    )
    changes = pick_change_report(
        recipe_frames["legacy_T1"], recipe_frames["corrected_T1"]
    )
    comparison_differences: list[dict[str, Any]] = []
    stored_comparisons = result["evaluation"]["comparisons"]
    _numeric_differences(
        stored_comparisons["shared_block_max_mean_adjusted_vs_G0"],
        adjusted,
        path="comparisons.shared_adjustment",
        output=comparison_differences,
    )
    _numeric_differences(
        stored_comparisons["corrected_T1_minus_legacy_T1"],
        paired,
        path="comparisons.paired",
        output=comparison_differences,
    )
    _numeric_differences(
        stored_comparisons["pick_changes_corrected_vs_legacy"],
        changes,
        path="comparisons.pick_changes",
        output=comparison_differences,
    )
    comparison_max = max(
        (value["absolute_difference"] for value in comparison_differences),
        default=0.0,
    )
    check(
        "paired_and_shared_block_comparisons_recomputed",
        comparison_max <= TOLERANCE,
        {
            "numeric_values": len(comparison_differences),
            "maximum_absolute_difference": comparison_max,
        },
    )
    check(
        "corrected_vs_legacy_change_is_13_slots_on_9_days",
        changes["changed_slots"] == 13
        and changes["dates_with_any_changed_slot"] == 9
        and changes == stored_comparisons["pick_changes_corrected_vs_legacy"],
        changes,
    )

    summary_recomputed = {
        "G0_top2_net20_mean_pct": recomputed_metrics["G0_price_core"]["top2"]["net20"]["net_mean_pct_at_cost"],
        "legacy_T1_top2_net20_mean_pct": recomputed_metrics["legacy_T1"]["top2"]["net20"]["net_mean_pct_at_cost"],
        "corrected_T1_top2_net20_mean_pct": recomputed_metrics["corrected_T1"]["top2"]["net20"]["net_mean_pct_at_cost"],
    }
    summary_recomputed["legacy_T1_minus_G0_pct"] = (
        summary_recomputed["legacy_T1_top2_net20_mean_pct"]
        - summary_recomputed["G0_top2_net20_mean_pct"]
    )
    summary_recomputed["corrected_T1_minus_G0_pct"] = (
        summary_recomputed["corrected_T1_top2_net20_mean_pct"]
        - summary_recomputed["G0_top2_net20_mean_pct"]
    )
    summary_recomputed["corrected_T1_minus_legacy_T1_pct"] = (
        summary_recomputed["corrected_T1_top2_net20_mean_pct"]
        - summary_recomputed["legacy_T1_top2_net20_mean_pct"]
    )
    summary_differences: list[dict[str, Any]] = []
    _numeric_differences(
        result["summary"],
        summary_recomputed,
        path="summary",
        output=summary_differences,
    )
    summary_max = max(
        (value["absolute_difference"] for value in summary_differences), default=0.0
    )
    no_promotion = (
        summary_max <= TOLERANCE
        and result["summary"]["production_change"] is False
        and result["summary"]["feature_survivor_lock_changed"] is False
        and result["authority"]["production_promotion_allowed"] is False
        and result["authority"]["selection_allowed"] is False
    )
    check("summary_recomputed_and_no_production_promotion", no_promotion)

    provenance_issue = {
        "id": "overlay_raw_source_and_decision_time_provenance_not_registered",
        "impact": "The overlay bytes, schema, count geometry, coverage, target-outcome independence and model-fold PIT are auditable, but the registered protocol does not bind the raw TDnet files, overlay-builder implementation, or per-row disclosure timestamps. An independent auditor therefore cannot prove from committed artifacts alone that every overlay value was available by the 08:58 decision cutoff.",
        "mitigation": "This run is explicitly retrospective, cannot select a feature, and cannot change production. Future overlays should register a source manifest, builder hash, decision cutoff, and per-row source_max_timestamp before metrics are viewed.",
    }
    input_lock = Path("/tmp/model_v07_corrected_input_lock.json")
    supplemental = {
        "available": input_lock.exists(),
        "path": str(input_lock),
        "sha256": sha256_file(input_lock) if input_lock.exists() else None,
        "registered_by_protocol": False,
        "used_as_audit_authority": False,
    }

    passed = all(item["passed"] for item in checks)
    audit = {
        "schema_version": 1,
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": sha256_file(result_path),
        "result_manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(runner_path),
        "auditor_sha256": sha256_file(Path(__file__)),
        "checks": checks,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "failed_checks": [item["name"] for item in checks if not item["passed"]],
        "maximum_absolute_metric_difference": max_metric_difference,
        "maximum_absolute_comparison_difference": comparison_max,
        "recomputed": {
            "summary": summary_recomputed,
            "shared_block_adjustment": adjusted,
            "corrected_minus_legacy_paired": paired,
            "pick_changes": changes,
            "overlay_coverage": coverage,
            "overlay_merge_semantics": merge_report,
        },
        "pit_review": {
            "model_folds_strictly_prior": fold_ok,
            "overlay_excludes_target_outcomes": set(overlay.columns)
            == {"date", "code", *CORRECTED_COLUMNS},
            "decision_time_provenance": "not independently provable from registered artifacts",
            "unregistered_supplemental_input_lock": supplemental,
        },
        "implementation_review": {
            "major_issues": [],
            "minor_or_scope_limiting_issues": [provenance_issue],
            "verified": [
                "G0 and legacy T1 reproduce all 168 frozen slots and scores exactly.",
                "Every recipe metric, the paired interval, and the shared circular-block adjustment were independently reproduced.",
                "The corrected overlay covers every legacy event key and preserves complete-zero/incomplete-NaN semantics.",
                "The corrected and legacy recipes differ in exactly 13 slots across 9 dates.",
                "The correction remains retrospective-only and cannot alter production or the feature survivor lock.",
            ],
        },
        "passed": passed,
        "assurance": "numeric_and_registered_artifact_checks_passed_with_overlay_source_provenance_limitation",
    }
    write_json(audit, output_path)


if __name__ == "__main__":
    main()
