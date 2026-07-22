#!/usr/bin/env python3
"""Independent, CSV-first audit of the frozen v0.7 tail-risk diagnostic.

This module deliberately does not import the v0.7 research runner.  It rebuilds
the fixed two-slot portfolios, costs, period statistics, moving-block
intervals, gate decisions, condition slices, and the unrestricted E0 comparison
from the content-addressed inputs.  The original result and manifest are never
modified; the audit is written to a separate JSON file.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "research/model_v07_tail_risk_result.json"
DEFAULT_OUTPUT = ROOT / "research/model_v07_tail_risk_result.audit.json"
RUNNER = ROOT / "research/analyze_model_v07_tail_risk.py"
TOLERANCE = 1e-11


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": "string"})
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if "code" in frame:
        frame["code"] = frame["code"].astype(str)
    return frame


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA or (not isinstance(value, str) and pd.isna(value)):
        return None
    return value


class AuditRecorder:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: Any = None) -> None:
        self.checks.append(
            {"name": name, "passed": bool(passed), "detail": _json_safe(detail)}
        )

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [item for item in self.checks if not item["passed"]]


def _equal_values(left: Any, right: Any, tolerance: float = TOLERANCE) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, (bool, np.bool_)) and isinstance(
            right, (bool, np.bool_)
        ) and bool(left) == bool(right)
    if isinstance(left, (int, float, np.number)) and isinstance(
        right, (int, float, np.number)
    ):
        a, b = float(left), float(right)
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        if math.isinf(a) or math.isinf(b):
            return a == b
        return abs(a - b) <= tolerance
    return left == right


def _nested_mismatches(
    stored: Any,
    recomputed: Any,
    *,
    path: str = "root",
    limit: int = 100,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    def visit(left: Any, right: Any, current: str) -> None:
        if len(output) >= limit:
            return
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            left_keys, right_keys = set(left), set(right)
            for key in sorted(left_keys - right_keys):
                output.append(
                    {"path": f"{current}.{key}", "stored": _json_safe(left[key]), "recomputed": "<missing>"}
                )
            for key in sorted(right_keys - left_keys):
                output.append(
                    {"path": f"{current}.{key}", "stored": "<missing>", "recomputed": _json_safe(right[key])}
                )
            for key in sorted(left_keys & right_keys):
                visit(left[key], right[key], f"{current}.{key}")
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                output.append(
                    {"path": current, "stored_length": len(left), "recomputed_length": len(right)}
                )
                return
            for position, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f"{current}[{position}]")
            return
        if not _equal_values(left, right):
            output.append(
                {"path": current, "stored": _json_safe(left), "recomputed": _json_safe(right)}
            )

    visit(stored, recomputed, path)
    return output


def _frame_mismatches(
    stored: pd.DataFrame,
    recomputed: pd.DataFrame,
    *,
    keys: Sequence[str],
    columns: Sequence[str],
    limit: int = 50,
    tolerance: float = TOLERANCE,
) -> list[dict[str, Any]]:
    left = stored.loc[:, [*keys, *columns]].sort_values(list(keys), kind="stable")
    right = recomputed.loc[:, [*keys, *columns]].sort_values(list(keys), kind="stable")
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    if len(left) != len(right):
        return [{"row_count": {"stored": len(left), "recomputed": len(right)}}]
    output: list[dict[str, Any]] = []
    for position in range(len(left)):
        for column in [*keys, *columns]:
            a, b = left.at[position, column], right.at[position, column]
            a_missing, b_missing = pd.isna(a), pd.isna(b)
            if a_missing or b_missing:
                equal = bool(a_missing and b_missing)
            else:
                equal = _equal_values(a, b, tolerance=tolerance)
            if not equal:
                output.append(
                    {
                        "row": position,
                        "column": column,
                        "stored": _json_safe(a),
                        "recomputed": _json_safe(b),
                    }
                )
                if len(output) >= limit:
                    return output
    return output


def _period_bounds(
    protocol: Mapping[str, Any], period_id: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if period_id == "replay_combined":
        return (
            pd.Timestamp(protocol["periods"]["replay_a"]["start"]),
            pd.Timestamp(protocol["periods"]["replay_c"]["end"]),
        )
    period = protocol["periods"][period_id]
    return pd.Timestamp(period["start"]), pd.Timestamp(period["end"])


def _period(frame: pd.DataFrame, protocol: Mapping[str, Any], period_id: str) -> pd.DataFrame:
    start, end = _period_bounds(protocol, period_id)
    return frame.loc[frame["date"].between(start, end)].copy()


def _assert_fixed_two(frame: pd.DataFrame) -> bool:
    if frame.empty or frame[["date", "model_rank"]].duplicated().any():
        return False
    rank_lists = frame.groupby("date", sort=False)["model_rank"].agg(list)
    if not rank_lists.map(lambda values: sorted(values) == [1, 2]).all():
        return False
    return not frame[["date", "code"]].duplicated().any()


def load_registered_display(protocol: Mapping[str, Any]) -> pd.DataFrame:
    discovery = _read_csv(_resolve(protocol["inputs"]["discovery_display_picks"]))
    discovery = discovery.loc[
        discovery["recipe_id"].eq(protocol["inputs"]["discovery_recipe_id"])
    ].copy()
    discovery["display_source"] = "v06_error_discovery"
    replay = _read_csv(_resolve(protocol["inputs"]["replay_display_picks"]))
    replay["display_source"] = "v07_g0_replay"
    columns = [
        "date",
        "model_rank",
        "code",
        "name",
        "model_score",
        "label",
        "oc_return_pct",
        "outcome_observed",
        "display_source",
    ]
    return pd.concat([discovery[columns], replay[columns]], ignore_index=True).sort_values(
        ["date", "model_rank"], kind="stable"
    ).reset_index(drop=True)


def recompute_flags(frame: pd.DataFrame, protocol: Mapping[str, Any]) -> pd.DataFrame:
    output = frame.copy()
    flags: list[str] = []
    availability: list[str] = []
    for condition in protocol["tail_conditions"]:
        condition_id = str(condition["condition_id"])
        feature = str(condition["feature"])
        values = pd.to_numeric(output[feature], errors="coerce")
        available = values.notna()
        if condition["operator"] == ">=":
            raw = values.ge(float(condition["threshold"]))
        elif condition["operator"] == "<=":
            raw = values.le(float(condition["threshold"]))
        else:
            raise ValueError(f"unsupported operator {condition['operator']!r}")
        flag_column = f"tail_flag__{condition_id}"
        available_column = f"tail_available__{condition_id}"
        output[flag_column] = raw.astype("boolean").where(available, pd.NA)
        output[available_column] = available
        flags.append(flag_column)
        availability.append(available_column)
    output["tail_all_features_available"] = output[availability].all(axis=1)
    lower_bound = output[flags].fillna(False).sum(axis=1).astype("Int64")
    output["tail_condition_count"] = lower_bound.where(
        output["tail_all_features_available"], pd.NA
    )
    return output


def recipe_acceptance(frame: pd.DataFrame, recipe_id: str) -> pd.Series:
    if recipe_id == "forced_top2":
        return pd.Series(True, index=frame.index, dtype=bool)
    if recipe_id.startswith("veto_any_"):
        threshold = int(recipe_id.split("_")[2])
        return ~frame["tail_condition_count"].ge(threshold).fillna(False).astype(bool)
    condition_id = recipe_id.removeprefix("veto_")
    flag = frame[f"tail_flag__{condition_id}"]
    return ~flag.fillna(False).astype(bool)


def build_recipe_rows(
    candidates: pd.DataFrame, protocol: Mapping[str, Any]
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    slot_weight = float(protocol["decision"]["slot_weight"])
    primary = float(protocol["decision"]["primary_cost_bps_round_trip"])
    stress = float(protocol["decision"]["stress_cost_bps_round_trip"])
    for recipe in protocol["recipe_registry"]:
        recipe_id = str(recipe["recipe_id"])
        part = candidates.copy()
        part["recipe_id"] = recipe_id
        part["marker_accepted"] = recipe_acceptance(part, recipe_id)
        part["executed"] = part["marker_accepted"] & part["oc_return_pct"].notna()
        part["label"] = pd.Series(pd.NA, index=part.index, dtype="Float64")
        observed = part["oc_return_pct"].notna()
        part.loc[observed, "label"] = part.loc[observed, "oc_return_pct"].gt(0).astype(float)
        part["gross_contribution_pct"] = np.where(
            part["executed"], slot_weight * part["oc_return_pct"], 0.0
        )
        part["net_contribution_pct_20bp"] = np.where(
            part["executed"],
            slot_weight * (part["oc_return_pct"] - primary / 100.0),
            0.0,
        )
        part["net_contribution_pct_40bp"] = np.where(
            part["executed"],
            slot_weight * (part["oc_return_pct"] - stress / 100.0),
            0.0,
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["recipe_id", "date", "model_rank"], kind="stable"
    ).reset_index(drop=True)


def daily_returns(recipe_rows: pd.DataFrame) -> pd.DataFrame:
    daily = recipe_rows.groupby(["recipe_id", "date"], sort=True).agg(
        gross_return_pct=("gross_contribution_pct", "sum"),
        net20_return_pct=("net_contribution_pct_20bp", "sum"),
        net40_return_pct=("net_contribution_pct_40bp", "sum"),
        executed_slots=("executed", "sum"),
        signal_slots=("model_rank", "size"),
    )
    daily["executed_day"] = daily["executed_slots"].gt(0)
    return daily.reset_index()


def period_metrics(rows: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    ordered = daily.sort_values("date", kind="stable")
    gross = ordered.set_index("date")["gross_return_pct"]
    net20 = ordered.set_index("date")["net20_return_pct"]
    net40 = ordered.set_index("date")["net40_return_pct"]
    top_dates = net20.nlargest(min(5, len(net20))).index
    without_top = net20.drop(top_dates)
    monthly = net20.groupby(net20.index.to_period("M")).mean()
    gains = float(net20.clip(lower=0.0).sum())
    losses = float(-net20.clip(upper=0.0).sum())
    executed = rows.loc[rows["executed"]]
    return {
        "days": int(len(ordered)),
        "scheduled_slots": int(len(rows)),
        "executed_slots": int(rows["executed"].sum()),
        "executed_days": int(ordered["executed_day"].sum()),
        "executed_day_fraction": float(ordered["executed_day"].mean()),
        "executed_slot_fraction": float(rows["executed"].mean()),
        "executed_hit_rate": float(executed["label"].mean()) if len(executed) else None,
        "gross_mean_pct": float(gross.mean()),
        "net20_mean_pct": float(net20.mean()),
        "net20_median_pct": float(net20.median()),
        "net40_mean_pct": float(net40.mean()),
        "profit_factor_net20": gains / losses if losses > 0.0 else None,
        "top5_removed_net20_mean_pct": float(without_top.mean()) if len(without_top) else None,
        "positive_months": int(monthly.gt(0.0).sum()),
        "months": int(len(monthly)),
        "positive_month_fraction": float(monthly.gt(0.0).mean()),
        "monthly_net20_mean_pct": {str(key): float(value) for key, value in monthly.items()},
        "largest_day_share_of_positive_profit": float(net20.max() / gains) if gains > 0.0 else None,
    }


def moving_block_means(
    values: np.ndarray,
    *,
    block_length: int,
    samples: int,
    seed: int,
    batch_size: int = 1_000,
) -> np.ndarray:
    observations = len(values)
    block_count = math.ceil(observations / block_length)
    max_start = observations - block_length + 1
    offsets = np.arange(block_length, dtype=np.int64)
    rng = np.random.default_rng(seed)
    output = np.empty(samples, dtype=float)
    position = 0
    while position < samples:
        size = min(batch_size, samples - position)
        starts = rng.integers(0, max_start, size=(size, block_count))
        indices = (starts[..., None] + offsets).reshape(size, -1)[:, :observations]
        output[position : position + size] = values[indices].mean(axis=1)
        position += size
    return output


def bootstrap_mean(values: pd.Series, protocol: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    config = protocol["bootstrap"]
    array = values.to_numpy(dtype=float)
    means = moving_block_means(
        array,
        block_length=int(config["block_length"]),
        samples=int(config["samples"]),
        seed=seed,
    )
    confidence = float(config["confidence"])
    tail = (1.0 - confidence) / 2.0
    return {
        "observations": int(len(array)),
        "block_length": int(config["block_length"]),
        "samples": int(config["samples"]),
        "confidence": confidence,
        "random_state": seed,
        "point_estimate_pct": float(array.mean()),
        "one_sided_lower_pct": float(np.quantile(means, 1.0 - confidence)),
        "two_sided_lower_pct": float(np.quantile(means, tail)),
        "two_sided_upper_pct": float(np.quantile(means, 1.0 - tail)),
        "bootstrap_standard_error_pct": float(means.std(ddof=1)),
    }


def bootstrap_pair(
    candidate: pd.Series,
    baseline: pd.Series,
    protocol: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    if not candidate.index.equals(baseline.index):
        raise AssertionError("paired daily indexes differ")
    config = protocol["bootstrap"]
    left = candidate.to_numpy(dtype=float)
    right = baseline.to_numpy(dtype=float)
    delta = left - right
    means = moving_block_means(
        delta,
        block_length=int(config["block_length"]),
        samples=int(config["samples"]),
        seed=seed,
    )
    confidence = float(config["confidence"])
    tail = (1.0 - confidence) / 2.0
    return {
        "observations": int(len(delta)),
        "block_length": int(config["block_length"]),
        "samples": int(config["samples"]),
        "confidence": confidence,
        "random_state": seed,
        "candidate_mean_pct": float(left.mean()),
        "baseline_mean_pct": float(right.mean()),
        "point_estimate_delta_pct": float(delta.mean()),
        "one_sided_lower_delta_pct": float(np.quantile(means, 1.0 - confidence)),
        "two_sided_lower_delta_pct": float(np.quantile(means, tail)),
        "two_sided_upper_delta_pct": float(np.quantile(means, 1.0 - tail)),
        "bootstrap_standard_error_delta_pct": float(means.std(ddof=1)),
    }


PERIOD_IDS = (
    "error_discovery",
    "replay_a",
    "replay_b",
    "replay_c",
    "replay_combined",
    "retrospective_union",
)


def recompute_recipe_results(
    recipe_rows: pd.DataFrame, protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    daily = daily_returns(recipe_rows)
    recipe_ids = [str(item["recipe_id"]) for item in protocol["recipe_registry"]]
    period_results: dict[str, dict[str, Any]] = {}
    slice_rows: list[dict[str, Any]] = []
    for recipe_id in recipe_ids:
        recipe_part = recipe_rows.loc[recipe_rows["recipe_id"].eq(recipe_id)]
        daily_part = daily.loc[daily["recipe_id"].eq(recipe_id)]
        periods: dict[str, Any] = {}
        for period_id in PERIOD_IDS:
            start, end = _period_bounds(protocol, period_id)
            rows = _period(recipe_part, protocol, period_id)
            days = _period(daily_part, protocol, period_id)
            metrics = period_metrics(rows, days)
            forced_days = _period(
                daily.loc[daily["recipe_id"].eq("forced_top2")],
                protocol,
                period_id,
            )
            left = days.set_index("date")["net20_return_pct"].sort_index()
            right = forced_days.set_index("date")["net20_return_pct"].sort_index()
            if not left.index.equals(right.index):
                raise AssertionError("recipe changed scheduled daily index")
            metrics["paired_uplift_vs_forced_pct"] = float((left - right).mean())
            periods[period_id] = metrics
            slice_rows.append(
                {
                    "recipe_id": recipe_id,
                    "period_id": period_id,
                    "start": str(start.date()),
                    "end": str(end.date()),
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key != "monthly_net20_mean_pct"
                    },
                }
            )
        period_results[recipe_id] = periods

    output: dict[str, Any] = {}
    bootstrap_seed = int(protocol["bootstrap"]["random_state"])
    gate = protocol["shadow_marker_gate"]
    for recipe_id in recipe_ids:
        periods = period_results[recipe_id]
        daily_part = daily.loc[daily["recipe_id"].eq(recipe_id)]
        replay = _period(daily_part, protocol, "replay_combined")
        forced = _period(
            daily.loc[daily["recipe_id"].eq("forced_top2")],
            protocol,
            "replay_combined",
        )
        left = replay.set_index("date")["net20_return_pct"].sort_index()
        right = forced.set_index("date")["net20_return_pct"].sort_index()
        absolute = bootstrap_mean(left, protocol, seed=bootstrap_seed)
        paired = bootstrap_pair(left, right, protocol, seed=bootstrap_seed + 1)
        combined = periods["replay_combined"]
        replay_periods = [periods[key] for key in ("replay_a", "replay_b", "replay_c")]
        checks = {
            "minimum_executed_days": combined["executed_days"]
            >= int(gate["minimum_executed_days"]),
            "net20_mean_positive": combined["net20_mean_pct"] > 0.0,
            "net40_mean_nonnegative": combined["net40_mean_pct"] >= 0.0,
            "top5_removed_positive": combined["top5_removed_net20_mean_pct"] > 0.0,
            "profit_factor": combined["profit_factor_net20"] is not None
            and combined["profit_factor_net20"] >= float(gate["profit_factor_min"]),
            "positive_month_fraction": combined["positive_month_fraction"]
            >= float(gate["positive_month_fraction_min"]),
            "largest_day_share": combined["largest_day_share_of_positive_profit"]
            is not None
            and combined["largest_day_share_of_positive_profit"]
            <= float(gate["largest_day_share_of_positive_profit_max"]),
            "absolute_block_lower": absolute["one_sided_lower_pct"] >= 0.0,
            "paired_mean_uplift": paired["point_estimate_delta_pct"] > 0.0,
            "paired_block_lower": paired["one_sided_lower_delta_pct"] >= 0.0,
            "minimum_positive_replay_periods": sum(
                item["net20_mean_pct"] > 0.0 for item in replay_periods
            )
            >= int(gate["minimum_absolute_positive_replay_periods"]),
            "all_replay_periods_beat_forced": all(
                item["paired_uplift_vs_forced_pct"] > 0.0 for item in replay_periods
            ),
        }
        passed = recipe_id != "forced_top2" and all(checks.values())
        if passed:
            classification = "shadow_risk_marker_candidate_only"
        elif recipe_id != "forced_top2" and checks["paired_mean_uplift"]:
            classification = "loss_mitigation_observation_only"
        else:
            classification = "rejected"
        output[recipe_id] = {
            "periods": periods,
            "absolute_block5_interval": absolute,
            "paired_block5_interval_vs_forced": paired,
            "gate_checks": checks,
            "passed_all_shadow_marker_checks": passed,
            "classification": classification,
        }
    return output, pd.DataFrame(slice_rows), daily


def recompute_condition_slice(
    candidates: pd.DataFrame, protocol: Mapping[str, Any]
) -> pd.DataFrame:
    condition_ids = [str(item["condition_id"]) for item in protocol["tail_conditions"]]
    markers = [*condition_ids, "count_ge_1", "count_ge_2", "count_ge_3"]
    rows: list[dict[str, Any]] = []
    for period_id in ("error_discovery", "replay_a", "replay_b", "replay_c"):
        frame = _period(candidates, protocol, period_id)
        for marker in markers:
            if marker.startswith("count_ge_"):
                threshold = int(marker.rsplit("_", 1)[1])
                marked = frame["tail_condition_count"].ge(threshold)
            else:
                marked = frame[f"tail_flag__{marker}"]
            available = marked.notna()
            states = (
                ("marked", marked.fillna(False).astype(bool)),
                ("unmarked", available & ~marked.fillna(False).astype(bool)),
                ("unavailable", ~available),
            )
            for state, mask in states:
                values = frame.loc[mask, "oc_return_pct"].dropna()
                rows.append(
                    {
                        "period_id": period_id,
                        "marker": marker,
                        "state": state,
                        "candidate_rows": int(mask.sum()),
                        "observed_return_rows": int(len(values)),
                        "mean_oc_return_pct": float(values.mean()) if len(values) else None,
                        "median_oc_return_pct": float(values.median()) if len(values) else None,
                        "loss_rate": float(values.lt(0.0).mean()) if len(values) else None,
                        "catastrophic_loss_rate": float(values.le(-5.0).mean()) if len(values) else None,
                    }
                )
    return pd.DataFrame(rows)


def recompute_extremes(candidates: pd.DataFrame) -> pd.DataFrame:
    result = candidates.loc[candidates["oc_return_pct"].abs().ge(5.0)].copy()
    result["tail_side"] = np.where(
        result["oc_return_pct"].lt(0.0), "catastrophic_loss", "large_win"
    )
    columns = [
        "date",
        "model_rank",
        "code",
        "name",
        "model_score",
        "oc_return_pct",
        "tail_side",
        "tail_condition_count",
        *(column for column in result if column.startswith("tail_flag__")),
        "session_range_ratio_5_20",
        "cc_vol_ratio_5_20",
        "xrank_close_momentum_60",
        "flat_oc_rate_20",
        "overnight_last",
        "display_source",
    ]
    return result.loc[:, columns].sort_values(
        ["tail_side", "oc_return_pct", "date"], kind="stable"
    ).reset_index(drop=True)


def missingness_summary(
    candidates: pd.DataFrame, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    periods = [*PERIOD_IDS]
    output: dict[str, Any] = {}
    for period_id in periods:
        frame = _period(candidates, protocol, period_id)
        condition_missing = {
            str(item["condition_id"]): int(frame[str(item["feature"])].isna().sum())
            for item in protocol["tail_conditions"]
        }
        complete = frame["tail_condition_count"].notna()
        output[period_id] = {
            "candidate_slots": int(len(frame)),
            "complete_five_feature_slots": int(complete.sum()),
            "incomplete_five_feature_slots": int((~complete).sum()),
            "condition_missing_slots": condition_missing,
            "count_ge_2_fraction_complete_case": (
                float(frame.loc[complete, "tail_condition_count"].ge(2).mean())
                if complete.any()
                else None
            ),
        }
    return output


def _panel_lookup(
    panel: pd.DataFrame,
    keys: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    panel_index = pd.MultiIndex.from_arrays(
        [panel["date"].array, panel["code"].astype(str).array],
        names=["date", "code"],
    )
    if panel_index.has_duplicates:
        raise AssertionError("feature panel has duplicate date/code keys")
    requested = keys[["date", "code"]].drop_duplicates().copy()
    requested["date"] = pd.to_datetime(requested["date"], errors="raise").dt.normalize()
    requested["code"] = requested["code"].astype(str)
    requested_index = pd.MultiIndex.from_frame(requested[["date", "code"]])
    positions = panel_index.get_indexer(requested_index)
    if (positions < 0).any():
        missing = requested.loc[positions < 0].head().to_dict("records")
        raise AssertionError(f"requested candidates absent from panel: {missing}")
    columns = [
        "date",
        "code",
        "feature_source_max_date",
        "candidate_price_source_max_date",
        *features,
    ]
    result = panel.iloc[positions].loc[:, columns].copy().reset_index(drop=True)
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["code"] = result["code"].astype(str)
    return result


def attach_panel_features(
    display: pd.DataFrame,
    lookup: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    base_columns = set(display)
    feature_columns = [str(item["feature"]) for item in protocol["tail_conditions"]]
    redundant = [column for column in feature_columns if column in base_columns]
    base = display.drop(columns=redundant, errors="ignore")
    merged = base.merge(
        lookup,
        on=["date", "code"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise AssertionError("candidate/panel merge is incomplete")
    return recompute_flags(merged.drop(columns="_merge"), protocol)


def _forced_daily(frame: pd.DataFrame, protocol: Mapping[str, Any]) -> pd.DataFrame:
    primary = float(protocol["decision"]["primary_cost_bps_round_trip"]) / 100.0
    stress = float(protocol["decision"]["stress_cost_bps_round_trip"]) / 100.0
    weight = float(protocol["decision"]["slot_weight"])
    observed = frame["oc_return_pct"].notna()
    working = frame.copy()
    working["executed"] = observed
    working["label"] = pd.Series(pd.NA, index=working.index, dtype="Float64")
    working.loc[observed, "label"] = working.loc[observed, "oc_return_pct"].gt(0).astype(float)
    working["gross_contribution_pct"] = np.where(
        observed, weight * working["oc_return_pct"], 0.0
    )
    working["net_contribution_pct_20bp"] = np.where(
        observed, weight * (working["oc_return_pct"] - primary), 0.0
    )
    working["net_contribution_pct_40bp"] = np.where(
        observed, weight * (working["oc_return_pct"] - stress), 0.0
    )
    daily = working.groupby("date", sort=True).agg(
        gross_return_pct=("gross_contribution_pct", "sum"),
        net20_return_pct=("net_contribution_pct_20bp", "sum"),
        net40_return_pct=("net_contribution_pct_40bp", "sum"),
        executed_slots=("executed", "sum"),
        signal_slots=("model_rank", "size"),
    )
    daily["executed_day"] = daily["executed_slots"].gt(0)
    return working, daily.reset_index()


def recompute_e0(
    e0: pd.DataFrame,
    g0: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    e0 = _period(e0, protocol, "replay_combined")
    g0 = _period(g0, protocol, "replay_combined")
    e0_rows, e0_daily = _forced_daily(e0, protocol)
    g0_rows, g0_daily = _forced_daily(g0, protocol)
    e0_metrics = period_metrics(e0_rows, e0_daily)
    e0_net = e0_daily.set_index("date")["net20_return_pct"].sort_index()
    g0_net = g0_daily.set_index("date")["net20_return_pct"].sort_index()
    paired = bootstrap_pair(
        e0_net,
        g0_net,
        protocol,
        seed=int(protocol["bootstrap"]["random_state"]) + 7,
    )
    g0_keys = set(zip(g0["date"], g0["code"]))
    e0_keys = set(zip(e0["date"], e0["code"]))
    added_mask = [
        (date, code) not in g0_keys for date, code in zip(e0["date"], e0["code"])
    ]
    removed_mask = [
        (date, code) not in e0_keys for date, code in zip(g0["date"], g0["code"])
    ]
    added = e0.loc[added_mask]
    removed = g0.loc[removed_mask]
    g0_complete = g0["tail_condition_count"].notna()
    e0_complete = e0["tail_condition_count"].notna()
    exposure = {
        "g0_mean_tail_condition_count": float(
            g0.loc[g0_complete, "tail_condition_count"].mean()
        ),
        "e0_mean_tail_condition_count": float(
            e0.loc[e0_complete, "tail_condition_count"].mean()
        ),
        "g0_count_ge_2_fraction": float(
            g0.loc[g0_complete, "tail_condition_count"].ge(2).mean()
        ),
        "e0_count_ge_2_fraction": float(
            e0.loc[e0_complete, "tail_condition_count"].ge(2).mean()
        ),
    }
    result = {
        "period": "replay_combined",
        "metrics": e0_metrics,
        "paired_interval_vs_g0": paired,
        "same_security_overlap_fraction": float(len(g0_keys & e0_keys) / len(e0_keys)),
        "added_security_slots": int(len(added)),
        "removed_security_slots": int(len(removed)),
        "added_mean_oc_return_pct": float(added["oc_return_pct"].mean()),
        "removed_mean_oc_return_pct": float(removed["oc_return_pct"].mean()),
        "tail_exposure": exposure,
    }
    denominator = {
        "g0_replay_slots": int(len(g0)),
        "g0_complete_tail_slots": int(g0_complete.sum()),
        "g0_incomplete_tail_slots": int((~g0_complete).sum()),
        "e0_replay_slots": int(len(e0)),
        "e0_complete_tail_slots": int(e0_complete.sum()),
        "e0_incomplete_tail_slots": int((~e0_complete).sum()),
        "exposure_statistics_use_complete_cases": True,
    }
    return result, denominator


def _hash_checks(
    recorder: AuditRecorder,
    result_path: Path,
    result: Mapping[str, Any],
    protocol_path: Path,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    protocol_hash = sha256_file(protocol_path)
    recorder.check(
        "protocol_hash_binding",
        protocol_hash
        == result["inputs"]["protocol_sha256"]
        == manifest["protocol_sha256"],
        protocol_hash,
    )
    recorder.check(
        "runner_hash_binding",
        sha256_file(RUNNER) == manifest["runner_sha256"],
        {"actual": sha256_file(RUNNER), "stored": manifest["runner_sha256"]},
    )
    recorder.check(
        "result_hash_binding",
        sha256_file(result_path) == manifest["result_sha256"],
        {"actual": sha256_file(result_path), "stored": manifest["result_sha256"]},
    )
    artifact_ok = True
    artifact_detail: dict[str, Any] = {}
    for artifact_id, raw in result["artifacts"].items():
        path = _resolve(raw["path"])
        actual = sha256_file(path)
        stored_result = raw["sha256"]
        stored_manifest = manifest["artifact_sha256"].get(artifact_id)
        passed = actual == stored_result == stored_manifest
        artifact_ok &= passed
        artifact_detail[artifact_id] = {
            "path": _portable_path(path),
            "actual": actual,
            "result": stored_result,
            "manifest": stored_manifest,
            "passed": passed,
        }
    recorder.check("artifact_hash_bindings", artifact_ok, artifact_detail)
    expected_inputs = {
        "protocol_sha256": protocol_hash,
        "discovery_picks_sha256": sha256_file(
            _resolve(protocol["inputs"]["discovery_display_picks"])
        ),
        "replay_picks_sha256": sha256_file(
            _resolve(protocol["inputs"]["replay_display_picks"])
        ),
        "e0_picks_sha256": sha256_file(
            _resolve(protocol["inputs"]["unrestricted_feature_model_picks"])
        ),
        "feature_panel_sha256": sha256_file(
            _resolve(protocol["inputs"]["feature_panel"])
        ),
        "feature_panel_manifest_sha256": sha256_file(
            _resolve(protocol["inputs"]["feature_panel_manifest"])
        ),
    }
    mismatches = _nested_mismatches(result["inputs"], expected_inputs, path="inputs")
    manifest_mismatches = _nested_mismatches(
        manifest["input_sha256"], expected_inputs, path="manifest.input_sha256"
    )
    recorder.check(
        "input_hash_bindings",
        not mismatches and not manifest_mismatches,
        {"result_mismatches": mismatches, "manifest_mismatches": manifest_mismatches},
    )


def _audit_protocol(recorder: AuditRecorder, protocol: Mapping[str, Any]) -> None:
    recipe_ids = [str(item["recipe_id"]) for item in protocol["recipe_registry"]]
    recorder.check(
        "posthoc_authority_prohibits_production",
        protocol["status"] == "locked_posthoc_retrospective_diagnostic"
        and protocol["authority"]["production_promotion_allowed"] is False
        and protocol["authority"]["hypothesis_was_selected_after_outcome_inspection"]
        is True,
    )
    recorder.check(
        "registered_recipe_set",
        len(recipe_ids) == 9
        and recipe_ids[0] == "forced_top2"
        and len(set(recipe_ids)) == len(recipe_ids),
        recipe_ids,
    )
    recorder.check(
        "fixed_display_and_cost_contract",
        protocol["decision"]["display_slots"] == 2
        and _equal_values(protocol["decision"]["slot_weight"], 0.5)
        and _equal_values(
            protocol["decision"]["primary_cost_bps_round_trip"], 20.0
        )
        and _equal_values(
            protocol["decision"]["stress_cost_bps_round_trip"], 40.0
        )
        and "do not inspect or promote rank 3"
        in protocol["decision"]["veto_policy"],
    )


def audit(
    *,
    result_path: Path = DEFAULT_RESULT,
    output_path: Path | None = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    result_path = _resolve(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest_path = result_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_path = ROOT / "research/model_v07_tail_risk_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    recorder = AuditRecorder()
    _audit_protocol(recorder, protocol)
    _hash_checks(recorder, result_path, result, protocol_path, protocol, manifest)

    candidates_path = _resolve(result["artifacts"]["candidate_picks"]["path"])
    period_slice_path = _resolve(result["artifacts"]["period_recipe_slice"]["path"])
    condition_slice_path = _resolve(result["artifacts"]["condition_slice"]["path"])
    extremes_path = _resolve(result["artifacts"]["extreme_cases"]["path"])
    candidates_stored = _read_csv(candidates_path)
    period_slice_stored = _read_csv(period_slice_path)
    condition_slice_stored = _read_csv(condition_slice_path)
    extremes_stored = _read_csv(extremes_path)

    registered = load_registered_display(protocol)
    recorder.check(
        "registered_display_is_fixed_top2",
        _assert_fixed_two(registered),
        {"rows": len(registered), "sessions": registered["date"].nunique()},
    )
    recorder.check(
        "registered_labels_match_open_close_returns",
        bool(
            (
                registered.loc[registered["oc_return_pct"].notna(), "label"].astype(int)
                == registered.loc[
                    registered["oc_return_pct"].notna(), "oc_return_pct"
                ].gt(0).astype(int)
            ).all()
        )
        and bool(
            (
                registered["label"].isna()
                == registered["oc_return_pct"].isna()
            ).all()
        ),
    )
    fixed_columns = [
        "model_rank",
        "code",
        "name",
        "model_score",
        "oc_return_pct",
        "display_source",
    ]
    display_mismatches = _frame_mismatches(
        candidates_stored,
        registered,
        keys=["date"],
        columns=fixed_columns,
    )
    recorder.check(
        "candidate_artifact_exactly_preserves_registered_top2_without_replacement",
        not display_mismatches
        and _assert_fixed_two(candidates_stored)
        and result["display"]["reranking_permitted"] is False,
        display_mismatches,
    )
    recorder.check(
        "display_summary",
        result["display"]["candidate_rows"] == len(candidates_stored) == 532
        and result["display"]["sessions"]
        == candidates_stored["date"].nunique()
        == 266
        and result["display"]["fixed_slots_per_day"] == 2,
        result["display"],
    )

    candidates = recompute_flags(candidates_stored, protocol)
    condition_ids = [str(item["condition_id"]) for item in protocol["tail_conditions"]]
    flag_columns = [f"tail_flag__{condition_id}" for condition_id in condition_ids]
    flag_mismatches = _frame_mismatches(
        candidates_stored,
        candidates,
        keys=["date", "model_rank", "code"],
        columns=[*flag_columns, "tail_condition_count"],
    )
    recorder.check(
        "tail_flags_and_complete_case_counts_recomputed_from_raw_features",
        not flag_mismatches,
        flag_mismatches,
    )

    recipes = build_recipe_rows(candidates, protocol)
    missing_count = candidates["tail_condition_count"].isna()
    count_pass_through = all(
        bool(recipe_acceptance(candidates, recipe_id).loc[missing_count].all())
        for recipe_id in ("veto_any_1_of_5", "veto_any_2_of_5", "veto_any_3_of_5")
    )
    single_pass_through = all(
        bool(
            recipe_acceptance(candidates, f"veto_{condition_id}")
            .loc[candidates[f"tail_flag__{condition_id}"].isna()]
            .all()
        )
        for condition_id in condition_ids
    )
    recorder.check(
        "missing_conditions_pass_through_without_invented_veto",
        count_pass_through and single_pass_through,
        {
            "incomplete_union_slots": int(missing_count.sum()),
            "count_recipe_pass_through": count_pass_through,
            "single_recipe_pass_through": single_pass_through,
        },
    )
    recipe_results, period_slice, daily = recompute_recipe_results(recipes, protocol)
    recipe_mismatches = _nested_mismatches(
        result["recipe_results"], recipe_results, path="recipe_results"
    )
    recorder.check(
        "all_recipe_period_metrics_bootstraps_gates_and_classifications_recomputed",
        not recipe_mismatches,
        recipe_mismatches,
    )
    slice_columns = [
        column
        for column in period_slice_stored.columns
        if column not in {"recipe_id", "period_id"}
    ]
    slice_mismatches = _frame_mismatches(
        period_slice_stored,
        period_slice,
        keys=["recipe_id", "period_id"],
        columns=slice_columns,
    )
    recorder.check("period_recipe_slice_recomputed", not slice_mismatches, slice_mismatches)

    condition_slice = recompute_condition_slice(candidates, protocol)
    condition_columns = [
        column
        for column in condition_slice_stored.columns
        if column not in {"period_id", "marker", "state"}
    ]
    condition_mismatches = _frame_mismatches(
        condition_slice_stored,
        condition_slice,
        keys=["period_id", "marker", "state"],
        columns=condition_columns,
    )
    recorder.check("condition_slice_recomputed", not condition_mismatches, condition_mismatches)
    extremes = recompute_extremes(candidates)
    extreme_columns = [column for column in extremes_stored if column not in {"date", "model_rank", "code"}]
    extreme_mismatches = _frame_mismatches(
        extremes_stored,
        extremes,
        keys=["date", "model_rank", "code"],
        columns=extreme_columns,
    )
    recorder.check("extreme_case_artifact_recomputed", not extreme_mismatches, extreme_mismatches)

    feature_names = [str(item["feature"]) for item in protocol["tail_conditions"]]
    e0_source = _read_csv(_resolve(protocol["inputs"]["unrestricted_feature_model_picks"]))
    replay_g0 = _read_csv(_resolve(protocol["inputs"]["replay_display_picks"]))
    recorder.check(
        "e0_and_g0_replay_are_fixed_top2_on_identical_dates",
        _assert_fixed_two(e0_source)
        and _assert_fixed_two(replay_g0)
        and set(e0_source["date"]) == set(replay_g0["date"]),
    )
    panel_path = _resolve(protocol["inputs"]["feature_panel"])
    panel_manifest = json.loads(
        _resolve(protocol["inputs"]["feature_panel_manifest"]).read_text(encoding="utf-8")
    )
    panel = joblib.load(panel_path, mmap_mode="r")
    recorder.check(
        "panel_manifest_and_result_summary",
        panel_manifest["panel_file_sha256"] == sha256_file(panel_path)
        and panel_manifest["rows"] == len(panel) == result["panel"]["rows"]
        and len(panel_manifest["sessions"])
        == panel["date"].nunique()
        == result["panel"]["sessions"],
        {
            "rows": len(panel),
            "sessions": panel["date"].nunique(),
        },
    )
    lookup = _panel_lookup(
        panel,
        pd.concat(
            [registered[["date", "code"]], e0_source[["date", "code"]]],
            ignore_index=True,
        ),
        feature_names,
    )
    del panel
    panel_g0 = attach_panel_features(registered, lookup, protocol)
    panel_e0 = attach_panel_features(e0_source, lookup, protocol)
    pit_ok = True
    for frame in (panel_g0, panel_e0):
        for column in ("feature_source_max_date", "candidate_price_source_max_date"):
            values = pd.to_datetime(frame[column], errors="coerce")
            pit_ok &= bool((values.isna() | values.lt(frame["date"])).all())
    recorder.check("panel_features_are_strictly_prior_to_decision_date", pit_ok)
    raw_feature_mismatches = _frame_mismatches(
        candidates_stored,
        panel_g0,
        keys=["date", "model_rank", "code"],
        columns=[*feature_names, *flag_columns, "tail_condition_count"],
        tolerance=5e-7,
    )
    recorder.check(
        "candidate_features_and_flags_match_immutable_panel",
        not raw_feature_mismatches,
        raw_feature_mismatches,
    )

    e0_recomputed, e0_denominators = recompute_e0(panel_e0, panel_g0, protocol)
    stored_e0_numeric = {
        key: value
        for key, value in result["unrestricted_e0"].items()
        if key not in {"empirical_diagnosis", "causal_limit"}
    }
    e0_mismatches = _nested_mismatches(
        stored_e0_numeric, e0_recomputed, path="unrestricted_e0"
    )
    recorder.check(
        "unrestricted_e0_metrics_turnover_exposure_and_paired_interval_recomputed",
        not e0_mismatches,
        e0_mismatches,
    )

    qualifying = [
        recipe_id
        for recipe_id, value in recipe_results.items()
        if value["passed_all_shadow_marker_checks"]
    ]
    mitigations = [
        recipe_id
        for recipe_id, value in recipe_results.items()
        if value["classification"] == "loss_mitigation_observation_only"
    ]
    best = (
        sorted(
            mitigations,
            key=lambda recipe_id: (
                -recipe_results[recipe_id]["periods"]["replay_combined"]["net20_mean_pct"],
                recipe_id,
            ),
        )[0]
        if mitigations
        else None
    )
    decision_expected = {
        "qualifying_shadow_marker_recipes": qualifying,
        "production_promotion": "prohibited_by_protocol",
        "best_loss_mitigation_observation": best,
    }
    decision_stored = {
        key: result["decision"][key] for key in decision_expected
    }
    decision_mismatches = _nested_mismatches(
        decision_stored, decision_expected, path="decision"
    )
    recorder.check("decision_recomputed", not decision_mismatches, decision_mismatches)

    artifact_registry = set(protocol["artifacts"])
    condition_registered = any("condition_slice" in item for item in artifact_registry)
    issues: dict[str, list[dict[str, Any]]] = {"critical": [], "minor": []}
    if recorder.failed:
        issues["critical"].append(
            {
                "issue_id": "AUDIT_RECOMPUTATION_OR_BINDING_FAILURE",
                "summary": "At least one content binding or independent recomputation check failed.",
                "failed_checks": [item["name"] for item in recorder.failed],
            }
        )
    if not condition_registered:
        issues["minor"].append(
            {
                "issue_id": "PROTOCOL_ARTIFACT_REGISTRY_OMITS_CONDITION_SLICE",
                "summary": (
                    "The protocol artifact list omits condition_slice, although the runner emits it "
                    "and both result and manifest content-address it."
                ),
                "decision_impact": "none; bytes are independently bound and recomputed",
            }
        )
    if e0_denominators["g0_incomplete_tail_slots"] or e0_denominators["e0_incomplete_tail_slots"]:
        issues["minor"].append(
            {
                "issue_id": "E0_TAIL_EXPOSURE_DENOMINATOR_IMPLICIT",
                "summary": (
                    "E0/G0 tail-count means and >=2 fractions use complete cases, but the result "
                    "field names do not state that denominator and do not report unavailable slots."
                ),
                "denominators": e0_denominators,
                "decision_impact": "none; stored complete-case values are numerically correct",
            }
        )

    no_critical = not issues["critical"]
    report = {
        "schema_version": 1,
        "audit_id": "model_v07_tail_risk_result_independent_audit_20260722",
        "audited_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "scope": {
            "result": str(result_path.relative_to(ROOT)),
            "manifest": str(manifest_path.relative_to(ROOT)),
            "protocol": str(protocol_path.relative_to(ROOT)),
            "method": (
                "Independent CSV/input reconstruction; no import from the v0.7 research runner."
            ),
        },
        "checks": {
            "total": len(recorder.checks),
            "passed": sum(item["passed"] for item in recorder.checks),
            "failed": len(recorder.failed),
            "items": recorder.checks,
        },
        "recomputed_summary": {
            "display_rows": int(len(candidates)),
            "sessions": int(candidates["date"].nunique()),
            "recipe_count": int(len(recipe_results)),
            "period_recipe_rows": int(len(period_slice)),
            "condition_slice_rows": int(len(condition_slice)),
            "extreme_case_rows": int(len(extremes)),
            "missingness": missingness_summary(candidates, protocol),
            "replay_forced_net20_mean_pct": recipe_results["forced_top2"]["periods"]["replay_combined"]["net20_mean_pct"],
            "best_loss_mitigation_observation": best,
            "best_loss_mitigation_replay_net20_mean_pct": (
                recipe_results[best]["periods"]["replay_combined"]["net20_mean_pct"]
                if best
                else None
            ),
            "e0": {
                "net20_mean_pct": e0_recomputed["metrics"]["net20_mean_pct"],
                "uplift_vs_g0_pct": e0_recomputed["paired_interval_vs_g0"]["point_estimate_delta_pct"],
                "paired_one_sided_80pct_lower_pct": e0_recomputed["paired_interval_vs_g0"]["one_sided_lower_delta_pct"],
                "tail_exposure_denominators": e0_denominators,
            },
        },
        "issues": issues,
        "verdict": {
            "computational_integrity": "pass" if no_critical else "fail",
            "production_adoption": "reject",
            "shadow_marker_adoption": "none_qualified",
            "loss_mitigation_observation_only": best,
            "unrestricted_e0": "reject",
            "reason": (
                "All stored calculations reproduce, but no recipe passes the registered gate; "
                "the protocol is post-hoc and explicitly prohibits production promotion."
                if no_critical
                else "Audit failures prevent reliance on the stored result."
            ),
        },
    }
    report = _json_safe(report)
    if output_path is not None:
        target = _resolve(output_path)
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = audit(result_path=args.result, output_path=args.output)
    print(
        json.dumps(
            {
                "checks": {
                    key: report["checks"][key] for key in ("total", "passed", "failed")
                },
                "issues": {
                    key: [item["issue_id"] for item in value]
                    for key, value in report["issues"].items()
                },
                "verdict": report["verdict"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["verdict"]["computational_integrity"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
