#!/usr/bin/env python3
"""Reproduce the post-hoc v0.7 G0 tail-risk diagnostic.

The five cut points in this module were discovered after inspecting the same
historical outcomes evaluated here.  Consequently this runner can describe
loss concentration and reject a proposed rule, but it cannot establish an
out-of-sample edge or promote production trading.

Every recipe keeps the G0 display fixed.  It may independently execute either
of the two 50 percent sleeves or leave that sleeve in cash; rank three is never
loaded or used as a replacement.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tse_session_ranker.exceptions import DataValidationError, LeakageError  # noqa: E402
from tse_session_ranker.validation import (  # noqa: E402
    moving_block_bootstrap,
    paired_moving_block_bootstrap,
)


PROTOCOL_PATH = ROOT / "research/model_v07_tail_risk_protocol.json"
EXPECTED_PROTOCOL_SHA256 = (
    "3b76e0ec8824953146e50392f0a406b4b95238e23992bb4d0951b16daf049dc8"
)
RESULT_SCHEMA_VERSION = 1
SLOT_WEIGHT = 0.5


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    source = Path(path)
    if source.resolve() == PROTOCOL_PATH.resolve():
        if sha256_file(source) != EXPECTED_PROTOCOL_SHA256:
            raise ValueError("v0.7 tail-risk protocol changed after registration")
    protocol = json.loads(source.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported v0.7 tail-risk protocol")
    if protocol.get("status") != "locked_posthoc_retrospective_diagnostic":
        raise ValueError("tail-risk protocol is not a locked post-hoc diagnostic")
    if protocol["authority"].get("production_promotion_allowed") is not False:
        raise ValueError("post-hoc tail-risk protocol cannot promote production")
    conditions = protocol.get("tail_conditions", [])
    if len(conditions) != 5:
        raise ValueError("tail-risk protocol must contain exactly five conditions")
    condition_ids = [str(value["condition_id"]) for value in conditions]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("tail-risk condition ids must be unique")
    recipes = [str(value["recipe_id"]) for value in protocol["recipe_registry"]]
    if recipes[0] != "forced_top2" or len(recipes) != 9:
        raise ValueError("tail-risk protocol must freeze forced plus eight veto recipes")
    periods = protocol["periods"]
    previous = pd.Timestamp(periods["error_discovery"]["end"])
    for period_id in ("replay_a", "replay_b", "replay_c"):
        start = pd.Timestamp(periods[period_id]["start"])
        end = pd.Timestamp(periods[period_id]["end"])
        if start <= previous or end < start:
            raise ValueError("tail-risk periods overlap or are reversed")
        previous = end
    return protocol


def assert_protocol_registered_before_run(
    protocol: Mapping[str, Any], run_started_at: datetime
) -> None:
    registered_at = pd.Timestamp(protocol["registered_at"])
    started_at = pd.Timestamp(run_started_at)
    if registered_at.tzinfo is None or started_at.tzinfo is None:
        raise ValueError("tail-risk protocol and run timestamps must be timezone aware")
    if registered_at >= started_at:
        raise ValueError("tail-risk protocol was not registered before run start")


def _read_picks(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": "string"})
    required = {
        "date",
        "model_rank",
        "code",
        "name",
        "model_score",
        "label",
        "oc_return_pct",
        "outcome_observed",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"display picks lack columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if frame["date"].isna().any() or frame["code"].isna().any():
        raise DataValidationError("display picks contain invalid date/code keys")
    frame["code"] = frame["code"].astype(str)
    return frame


def _assert_fixed_top_two(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise DataValidationError("display pick input is empty")
    if frame[["date", "model_rank"]].duplicated().any():
        raise DataValidationError("display picks duplicate a date/rank slot")
    counts = frame.groupby("date", sort=False)["model_rank"].agg(list)
    invalid = counts.map(lambda values: sorted(values) != [1, 2])
    if invalid.any():
        raise DataValidationError("every display date must contain ranks one and two")
    if not frame["outcome_observed"].astype(bool).all():
        raise DataValidationError("tail-risk diagnostic requires observed sessions")
    mismatched = frame["label"].notna() ^ frame["oc_return_pct"].notna()
    if mismatched.any():
        raise DataValidationError("label and return execution state disagree")


def load_fixed_display(
    discovery_path: str | Path,
    replay_path: str | Path,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    """Load only the registered G0 top-two rows and join chronological inputs."""

    discovery = _read_picks(discovery_path)
    if "recipe_id" not in discovery:
        raise DataValidationError("v0.6 discovery picks lack recipe_id")
    discovery = discovery.loc[
        discovery["recipe_id"].eq(protocol["inputs"]["discovery_recipe_id"])
    ].copy()
    discovery["display_source"] = "v06_error_discovery"
    replay = _read_picks(replay_path)
    replay["display_source"] = "v07_g0_replay"
    keep = [
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
    display = pd.concat([discovery[keep], replay[keep]], ignore_index=True)
    display = display.sort_values(["date", "model_rank"], kind="stable").reset_index(
        drop=True
    )
    _assert_fixed_top_two(display)
    if display[["date", "code"]].duplicated().any():
        raise DataValidationError("one security occupies both display slots on a date")
    periods = protocol["periods"]
    union_start = pd.Timestamp(periods["retrospective_union"]["start"])
    union_end = pd.Timestamp(periods["retrospective_union"]["end"])
    if display["date"].min() < union_start or display["date"].max() > union_end:
        raise DataValidationError("display picks fall outside the registered union")
    return display


def _panel_manifest_path(panel_path: Path) -> Path:
    return panel_path.with_suffix(panel_path.suffix + ".manifest.json")


def load_tail_feature_projection(
    panel_path: str | Path,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only the five registered PIT features from the immutable panel."""

    source = Path(panel_path)
    manifest_path = _panel_manifest_path(source)
    if not source.exists() or not manifest_path.exists():
        raise FileNotFoundError("tail-risk diagnostic requires panel and manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("panel_file_sha256") != sha256_file(source):
        raise ValueError("feature panel bytes differ from its manifest")
    features = [str(value["feature"]) for value in protocol["tail_conditions"]]
    required = [
        "date",
        "code",
        "feature_source_max_date",
        "candidate_price_source_max_date",
        *features,
    ]
    panel = joblib.load(source, mmap_mode="r")
    missing = sorted(set(required) - set(panel.columns))
    if missing:
        raise DataValidationError(f"feature panel lacks tail-risk columns: {missing}")
    projection = panel.loc[:, required].copy()
    projection["date"] = pd.to_datetime(
        projection["date"], errors="coerce"
    ).dt.normalize()
    projection["code"] = projection["code"].astype(str)
    if projection[["date", "code"]].duplicated().any():
        raise DataValidationError("feature panel duplicates date/code keys")
    return projection, manifest


def attach_tail_conditions(
    display: pd.DataFrame,
    feature_projection: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    """Attach five flags without looking at the current return or label."""

    base = display.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
    base["code"] = base["code"].astype(str)
    merged = base.merge(
        feature_projection,
        on=["date", "code"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[merged["_merge"].ne("both"), ["date", "code"]]
        raise DataValidationError(
            f"display candidates are absent from feature panel: {missing.head().to_dict('records')}"
        )
    merged = merged.drop(columns="_merge")
    for source_column in (
        "feature_source_max_date",
        "candidate_price_source_max_date",
    ):
        source_date = pd.to_datetime(merged[source_column], errors="coerce")
        leaked = source_date.notna() & source_date.ge(merged["date"])
        if leaked.any():
            raise LeakageError(f"{source_column} is not strictly prior")
    flag_columns: list[str] = []
    for condition in protocol["tail_conditions"]:
        condition_id = str(condition["condition_id"])
        feature = str(condition["feature"])
        available = merged[feature].notna()
        threshold = float(condition["threshold"])
        operator = str(condition["operator"])
        if operator == ">=":
            flag = merged[feature].ge(threshold)
        elif operator == "<=":
            flag = merged[feature].le(threshold)
        else:
            raise ValueError(f"unsupported condition operator: {operator}")
        column = f"tail_flag__{condition_id}"
        merged[column] = flag.astype("boolean").where(available, pd.NA)
        merged[f"tail_available__{condition_id}"] = available.astype(bool)
        flag_columns.append(column)
    merged["tail_all_features_available"] = merged[
        [f"tail_available__{value['condition_id']}" for value in protocol["tail_conditions"]]
    ].all(axis=1)
    lower_bound = merged[flag_columns].fillna(False).sum(axis=1).astype("Int64")
    merged["tail_condition_count"] = lower_bound.where(
        merged["tail_all_features_available"], pd.NA
    )
    return merged.sort_values(["date", "model_rank"], kind="stable").reset_index(
        drop=True
    )


def _recipe_execution(frame: pd.DataFrame, recipe_id: str) -> pd.Series:
    if recipe_id == "forced_top2":
        return pd.Series(True, index=frame.index)
    if recipe_id.startswith("veto_any_"):
        threshold = int(recipe_id.split("_")[2])
        return ~frame["tail_condition_count"].ge(threshold).fillna(False)
    prefix = "veto_"
    if not recipe_id.startswith(prefix):
        raise ValueError(f"unknown tail-risk recipe: {recipe_id}")
    condition_id = recipe_id[len(prefix) :]
    column = f"tail_flag__{condition_id}"
    if column not in frame:
        raise ValueError(f"recipe references absent tail flag: {condition_id}")
    return ~frame[column].fillna(False).astype(bool)


def build_recipe_picks(
    candidates: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    """Create long recipe rows while retaining every original display slot."""

    parts: list[pd.DataFrame] = []
    for recipe in protocol["recipe_registry"]:
        recipe_id = str(recipe["recipe_id"])
        part = candidates.copy()
        part["recipe_id"] = recipe_id
        part["marker_accepted"] = _recipe_execution(part, recipe_id).astype(bool)
        part["executed"] = part["marker_accepted"] & part["label"].notna()
        part["slot_weight"] = SLOT_WEIGHT
        part["gross_contribution_pct"] = np.where(
            part["executed"], SLOT_WEIGHT * part["oc_return_pct"], 0.0
        )
        for cost_bps in (20.0, 40.0):
            column = f"net_contribution_pct_{int(cost_bps)}bp"
            part[column] = np.where(
                part["executed"],
                SLOT_WEIGHT * (part["oc_return_pct"] - cost_bps / 100.0),
                0.0,
            )
        part["trade_reason"] = np.select(
            [part["executed"], part["marker_accepted"]],
            ["display_slot_executed", "accepted_but_unfilled"],
            default="tail_marker_cash_veto",
        )
        parts.append(part)
    output = pd.concat(parts, ignore_index=True)
    counts = output.groupby(["recipe_id", "date"], sort=False).size()
    if not counts.eq(2).all():
        raise AssertionError("a veto recipe changed the fixed two-slot display")
    return output.sort_values(
        ["recipe_id", "date", "model_rank"], kind="stable"
    ).reset_index(drop=True)


def daily_recipe_returns(recipe_picks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fixed 50 percent sleeve contributions, including all-cash days."""

    required = {
        "recipe_id",
        "date",
        "model_rank",
        "executed",
        "gross_contribution_pct",
        "net_contribution_pct_20bp",
        "net_contribution_pct_40bp",
    }
    missing = sorted(required - set(recipe_picks.columns))
    if missing:
        raise DataValidationError(f"recipe picks lack daily-return columns: {missing}")
    daily = recipe_picks.groupby(["recipe_id", "date"], sort=True).agg(
        gross_return_pct=("gross_contribution_pct", "sum"),
        net20_return_pct=("net_contribution_pct_20bp", "sum"),
        net40_return_pct=("net_contribution_pct_40bp", "sum"),
        executed_slots=("executed", "sum"),
        signal_slots=("model_rank", "size"),
    )
    if not daily["signal_slots"].eq(2).all():
        raise AssertionError("daily portfolio does not preserve two scheduled slots")
    daily["executed_day"] = daily["executed_slots"].gt(0)
    return daily.reset_index()


def _profit_factor(values: pd.Series) -> float:
    gains = float(values.clip(lower=0).sum())
    losses = float(-values.clip(upper=0).sum())
    return gains / losses if losses > 0 else float("inf")


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def period_metrics(
    recipe_rows: pd.DataFrame,
    daily: pd.DataFrame,
) -> dict[str, Any]:
    net20 = daily.set_index("date")["net20_return_pct"].sort_index()
    net40 = daily.set_index("date")["net40_return_pct"].sort_index()
    gross = daily.set_index("date")["gross_return_pct"].sort_index()
    top_days = net20.nlargest(min(5, len(net20))).index
    without_top5 = net20.drop(top_days)
    monthly = net20.groupby(net20.index.to_period("M")).mean()
    positive_sum = float(net20.clip(lower=0).sum())
    executed_rows = recipe_rows.loc[recipe_rows["executed"]]
    return {
        "days": int(len(daily)),
        "scheduled_slots": int(len(recipe_rows)),
        "executed_slots": int(recipe_rows["executed"].sum()),
        "executed_days": int(daily["executed_day"].sum()),
        "executed_day_fraction": float(daily["executed_day"].mean()),
        "executed_slot_fraction": float(recipe_rows["executed"].mean()),
        "executed_hit_rate": (
            float(executed_rows["label"].mean()) if len(executed_rows) else None
        ),
        "gross_mean_pct": float(gross.mean()),
        "net20_mean_pct": float(net20.mean()),
        "net20_median_pct": float(net20.median()),
        "net40_mean_pct": float(net40.mean()),
        "profit_factor_net20": _finite_or_none(_profit_factor(net20)),
        "top5_removed_net20_mean_pct": (
            float(without_top5.mean()) if len(without_top5) else None
        ),
        "positive_months": int(monthly.gt(0).sum()),
        "months": int(len(monthly)),
        "positive_month_fraction": float(monthly.gt(0).mean()),
        "monthly_net20_mean_pct": {
            str(key): float(value) for key, value in monthly.items()
        },
        "largest_day_share_of_positive_profit": (
            float(net20.max() / positive_sum) if positive_sum > 0 else None
        ),
    }


def _select_period(
    frame: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    return frame.loc[
        frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    ].copy()


def _period_bounds(
    protocol: Mapping[str, Any], period_id: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if period_id == "replay_combined":
        return (
            pd.Timestamp(protocol["periods"]["replay_a"]["start"]),
            pd.Timestamp(protocol["periods"]["replay_c"]["end"]),
        )
    raw = protocol["periods"][period_id]
    return pd.Timestamp(raw["start"]), pd.Timestamp(raw["end"])


def _bootstrap_dict(value: Any) -> dict[str, Any]:
    return {key: _finite_or_none(item) if isinstance(item, float) else item for key, item in asdict(value).items()}


def evaluate_recipes(
    recipe_picks: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate all frozen recipes and the strict shadow-marker gate."""

    daily = daily_recipe_returns(recipe_picks)
    recipe_ids = [str(value["recipe_id"]) for value in protocol["recipe_registry"]]
    period_ids = [
        "error_discovery",
        "replay_a",
        "replay_b",
        "replay_c",
        "replay_combined",
        "retrospective_union",
    ]
    output: dict[str, Any] = {}
    slices: list[dict[str, Any]] = []
    bootstrap = protocol["bootstrap"]
    for recipe_id in recipe_ids:
        recipe_rows = recipe_picks.loc[recipe_picks["recipe_id"].eq(recipe_id)]
        recipe_daily = daily.loc[daily["recipe_id"].eq(recipe_id)]
        periods: dict[str, Any] = {}
        for period_id in period_ids:
            start, end = _period_bounds(protocol, period_id)
            rows = _select_period(recipe_rows, start, end)
            days = _select_period(recipe_daily, start, end)
            metrics = period_metrics(rows, days)
            forced_days = _select_period(
                daily.loc[daily["recipe_id"].eq("forced_top2")], start, end
            )
            left = days.set_index("date")["net20_return_pct"].sort_index()
            right = forced_days.set_index("date")["net20_return_pct"].sort_index()
            if not left.index.equals(right.index):
                raise AssertionError("recipe and forced daily indexes differ")
            metrics["paired_uplift_vs_forced_pct"] = float((left - right).mean())
            periods[period_id] = metrics
            slices.append(
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

        replay_start, replay_end = _period_bounds(protocol, "replay_combined")
        replay_days = _select_period(recipe_daily, replay_start, replay_end)
        forced_replay = _select_period(
            daily.loc[daily["recipe_id"].eq("forced_top2")],
            replay_start,
            replay_end,
        )
        absolute_interval = moving_block_bootstrap(
            replay_days.set_index("date")["net20_return_pct"],
            block_length=int(bootstrap["block_length"]),
            samples=int(bootstrap["samples"]),
            confidence=float(bootstrap["confidence"]),
            random_state=int(bootstrap["random_state"]),
        )
        paired_interval = paired_moving_block_bootstrap(
            replay_days.set_index("date")["net20_return_pct"],
            forced_replay.set_index("date")["net20_return_pct"],
            block_length=int(bootstrap["block_length"]),
            samples=int(bootstrap["samples"]),
            confidence=float(bootstrap["confidence"]),
            random_state=int(bootstrap["random_state"]) + 1,
        )
        gate_metrics = periods["replay_combined"]
        replay_periods = [periods[key] for key in ("replay_a", "replay_b", "replay_c")]
        gate = protocol["shadow_marker_gate"]
        checks = {
            "minimum_executed_days": gate_metrics["executed_days"]
            >= int(gate["minimum_executed_days"]),
            "net20_mean_positive": gate_metrics["net20_mean_pct"] > 0.0,
            "net40_mean_nonnegative": gate_metrics["net40_mean_pct"] >= 0.0,
            "top5_removed_positive": gate_metrics[
                "top5_removed_net20_mean_pct"
            ]
            > 0.0,
            "profit_factor": (
                gate_metrics["profit_factor_net20"] is not None
                and gate_metrics["profit_factor_net20"]
                >= float(gate["profit_factor_min"])
            ),
            "positive_month_fraction": gate_metrics["positive_month_fraction"]
            >= float(gate["positive_month_fraction_min"]),
            "largest_day_share": (
                gate_metrics["largest_day_share_of_positive_profit"] is not None
                and gate_metrics["largest_day_share_of_positive_profit"]
                <= float(gate["largest_day_share_of_positive_profit_max"])
            ),
            "absolute_block_lower": absolute_interval.one_sided_lower_pct >= 0.0,
            "paired_mean_uplift": paired_interval.point_estimate_delta_pct > 0.0,
            "paired_block_lower": paired_interval.one_sided_lower_delta_pct >= 0.0,
            "minimum_positive_replay_periods": sum(
                value["net20_mean_pct"] > 0.0 for value in replay_periods
            )
            >= int(gate["minimum_absolute_positive_replay_periods"]),
            "all_replay_periods_beat_forced": all(
                value["paired_uplift_vs_forced_pct"] > 0.0
                for value in replay_periods
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
            "absolute_block5_interval": _bootstrap_dict(absolute_interval),
            "paired_block5_interval_vs_forced": _bootstrap_dict(paired_interval),
            "gate_checks": checks,
            "passed_all_shadow_marker_checks": passed,
            "classification": classification,
        }
    return output, pd.DataFrame(slices)


def condition_slices(
    candidates: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Describe marked and unmarked candidate returns without causal language."""

    result: list[dict[str, Any]] = []
    periods = ["error_discovery", "replay_a", "replay_b", "replay_c"]
    condition_ids = [str(value["condition_id"]) for value in protocol["tail_conditions"]]
    markers = [*condition_ids, "count_ge_1", "count_ge_2", "count_ge_3"]
    for period_id in periods:
        start, end = _period_bounds(protocol, period_id)
        frame = _select_period(candidates, start, end)
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
                scheduled_rows = int(mask.sum())
                values = frame.loc[mask, "oc_return_pct"].dropna()
                result.append(
                    {
                        "period_id": period_id,
                        "marker": marker,
                        "state": state,
                        "candidate_rows": scheduled_rows,
                        "observed_return_rows": int(len(values)),
                        "mean_oc_return_pct": (
                            float(values.mean()) if len(values) else None
                        ),
                        "median_oc_return_pct": (
                            float(values.median()) if len(values) else None
                        ),
                        "loss_rate": (
                            float(values.lt(0).mean()) if len(values) else None
                        ),
                        "catastrophic_loss_rate": (
                            float(values.le(-5).mean()) if len(values) else None
                        ),
                    }
                )
    return result


def extreme_cases(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return all observed five-percent tails for auditable case inspection."""

    result = candidates.loc[candidates["oc_return_pct"].abs().ge(5.0)].copy()
    result["tail_side"] = np.where(
        result["oc_return_pct"].lt(0), "catastrophic_loss", "large_win"
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
    )


def evaluate_unrestricted_e0(
    e0_path: str | Path,
    feature_projection: pd.DataFrame,
    g0_candidates: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare E0 display turnover and tail exposure on registered replay dates."""

    e0 = _read_picks(e0_path)
    _assert_fixed_top_two(e0)
    e0 = attach_tail_conditions(e0, feature_projection, protocol)
    replay_start, replay_end = _period_bounds(protocol, "replay_combined")
    e0 = _select_period(e0, replay_start, replay_end)
    g0 = _select_period(g0_candidates, replay_start, replay_end)
    if not pd.DatetimeIndex(e0["date"].unique()).equals(
        pd.DatetimeIndex(g0["date"].unique())
    ):
        raise DataValidationError("E0 and G0 replay dates differ")

    def daily_forced(frame: pd.DataFrame) -> pd.DataFrame:
        executed = frame["label"].notna()
        daily = frame.assign(
            gross_return_pct=np.where(
                executed, SLOT_WEIGHT * frame["oc_return_pct"], 0.0
            ),
            net20_return_pct=np.where(
                executed, SLOT_WEIGHT * (frame["oc_return_pct"] - 0.2), 0.0
            ),
            net40_return_pct=np.where(
                executed, SLOT_WEIGHT * (frame["oc_return_pct"] - 0.4), 0.0
            ),
            executed=executed,
        ).groupby("date", sort=True).agg(
            gross_return_pct=("gross_return_pct", "sum"),
            net20_return_pct=("net20_return_pct", "sum"),
            net40_return_pct=("net40_return_pct", "sum"),
            executed_slots=("executed", "sum"),
        )
        daily["executed_day"] = daily["executed_slots"].gt(0)
        return daily.reset_index()

    e0_daily = daily_forced(e0)
    g0_daily = daily_forced(g0)
    e0_rows = e0.assign(executed=e0["label"].notna())
    metrics = period_metrics(e0_rows, e0_daily)
    e0_net = e0_daily.set_index("date")["net20_return_pct"]
    g0_net = g0_daily.set_index("date")["net20_return_pct"]
    paired = paired_moving_block_bootstrap(
        e0_net,
        g0_net,
        block_length=int(protocol["bootstrap"]["block_length"]),
        samples=int(protocol["bootstrap"]["samples"]),
        confidence=float(protocol["bootstrap"]["confidence"]),
        random_state=int(protocol["bootstrap"]["random_state"]) + 7,
    )
    g0_keys = set(zip(g0["date"], g0["code"]))
    e0_keys = set(zip(e0["date"], e0["code"]))
    added = e0.loc[
        [(date, code) not in g0_keys for date, code in zip(e0["date"], e0["code"])]
    ]
    removed = g0.loc[
        [(date, code) not in e0_keys for date, code in zip(g0["date"], g0["code"])]
    ]
    exposure = {
        "g0_mean_tail_condition_count": float(g0["tail_condition_count"].mean()),
        "e0_mean_tail_condition_count": float(e0["tail_condition_count"].mean()),
        "g0_count_ge_2_fraction": float(g0["tail_condition_count"].ge(2).mean()),
        "e0_count_ge_2_fraction": float(e0["tail_condition_count"].ge(2).mean()),
    }
    result = {
        "period": "replay_combined",
        "metrics": metrics,
        "paired_interval_vs_g0": _bootstrap_dict(paired),
        "same_security_overlap_fraction": float(len(g0_keys & e0_keys) / len(e0_keys)),
        "added_security_slots": int(len(added)),
        "removed_security_slots": int(len(removed)),
        "added_mean_oc_return_pct": float(added["oc_return_pct"].mean()),
        "removed_mean_oc_return_pct": float(removed["oc_return_pct"].mean()),
        "tail_exposure": exposure,
        "empirical_diagnosis": (
            "E0 changed the ranking instead of enforcing monotone avoidance. "
            "Its displayed candidates carried materially more registered tail flags "
            "than G0, so free coefficients did not encode the cash-veto mechanism."
        ),
        "causal_limit": (
            "The comparison is retrospective and does not identify which coefficient "
            "caused the turnover or prove that the flags cause losses."
        ),
    }
    e0_output = e0.copy()
    e0_output["display_source"] = "G0_E0_unrestricted"
    return result, e0_output


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _artifact_paths(prefix: Path) -> dict[str, Path]:
    return {
        "result_json": prefix.with_suffix(".json"),
        "manifest": prefix.with_suffix(".manifest.json"),
        "candidate_picks": prefix.with_name(prefix.name + "_candidate_picks.csv"),
        "period_recipe_slice": prefix.with_name(
            prefix.name + "_period_recipe_slice.csv"
        ),
        "condition_slice": prefix.with_name(prefix.name + "_condition_slice.csv"),
        "extreme_cases": prefix.with_name(prefix.name + "_extreme_cases.csv"),
    }


def run(
    *,
    discovery_path: Path,
    replay_path: Path,
    e0_path: Path,
    panel_path: Path,
    output_prefix: Path,
    protocol_path: Path = PROTOCOL_PATH,
) -> dict[str, Any]:
    run_started_at = datetime.now(ZoneInfo("Asia/Tokyo"))
    protocol = load_protocol(protocol_path)
    assert_protocol_registered_before_run(protocol, run_started_at)
    display = load_fixed_display(discovery_path, replay_path, protocol)
    projection, panel_manifest = load_tail_feature_projection(panel_path, protocol)
    candidates = attach_tail_conditions(display, projection, protocol)
    recipes = build_recipe_picks(candidates, protocol)
    recipe_result, period_slice = evaluate_recipes(recipes, protocol)
    condition_slice = pd.DataFrame(condition_slices(candidates, protocol))
    extremes = extreme_cases(candidates)
    e0_result, _ = evaluate_unrestricted_e0(
        e0_path, projection, candidates, protocol
    )
    qualifying = [
        recipe_id
        for recipe_id, value in recipe_result.items()
        if value["passed_all_shadow_marker_checks"]
    ]
    mitigation = [
        recipe_id
        for recipe_id, value in recipe_result.items()
        if value["classification"] == "loss_mitigation_observation_only"
    ]
    ranked_mitigation = sorted(
        mitigation,
        key=lambda recipe_id: (
            -recipe_result[recipe_id]["periods"]["replay_combined"][
                "net20_mean_pct"
            ],
            recipe_id,
        ),
    )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "authority": protocol["authority"],
        "inputs": {
            "protocol_sha256": sha256_file(protocol_path),
            "discovery_picks_sha256": sha256_file(discovery_path),
            "replay_picks_sha256": sha256_file(replay_path),
            "e0_picks_sha256": sha256_file(e0_path),
            "feature_panel_sha256": sha256_file(panel_path),
            "feature_panel_manifest_sha256": sha256_file(
                _panel_manifest_path(panel_path)
            ),
        },
        "panel": {
            "rows": int(panel_manifest["rows"]),
            "sessions": int(len(panel_manifest["sessions"])),
        },
        "display": {
            "candidate_rows": int(len(candidates)),
            "sessions": int(candidates["date"].nunique()),
            "first_date": str(candidates["date"].min().date()),
            "last_date": str(candidates["date"].max().date()),
            "fixed_slots_per_day": 2,
            "reranking_permitted": False,
        },
        "recipe_results": recipe_result,
        "unrestricted_e0": e0_result,
        "decision": {
            "qualifying_shadow_marker_recipes": qualifying,
            "production_promotion": "prohibited_by_protocol",
            "best_loss_mitigation_observation": (
                ranked_mitigation[0] if ranked_mitigation else None
            ),
            "conclusion": (
                "No recipe may be promoted. A passing recipe would only become a "
                "prospective shadow marker because its thresholds were selected after "
                "these outcomes were known."
            ),
        },
        "limitations": [
            "All five thresholds are post-hoc and every evaluated outcome was known.",
            "Cash veto results measure avoided historical P&L, not causal effects.",
            "No rank-three replacement or portfolio reallocation was tested.",
            "Exact 08:58 futures, board imbalance, PTS and indicative-open fields are absent.",
        ],
        "runtime": {
            "run_started_at": run_started_at.isoformat(),
            "run_completed_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    }

    paths = _artifact_paths(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    candidate_columns = [
        "date",
        "model_rank",
        "code",
        "name",
        "model_score",
        "oc_return_pct",
        "tail_condition_count",
        *(column for column in candidates if column.startswith("tail_flag__")),
        "session_range_ratio_5_20",
        "cc_vol_ratio_5_20",
        "xrank_close_momentum_60",
        "flat_oc_rate_20",
        "overnight_last",
        "display_source",
    ]
    candidates.loc[:, candidate_columns].to_csv(
        paths["candidate_picks"], index=False, lineterminator="\n"
    )
    period_slice.to_csv(paths["period_recipe_slice"], index=False, lineterminator="\n")
    condition_slice.to_csv(paths["condition_slice"], index=False, lineterminator="\n")
    extremes.to_csv(paths["extreme_cases"], index=False, lineterminator="\n")
    result["artifacts"] = {
        key: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
        for key, path in paths.items()
        if key not in {"result_json", "manifest"}
    }
    _write_json(paths["result_json"], result)
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "result_sha256": sha256_file(paths["result_json"]),
        "artifact_sha256": {
            key: sha256_file(path)
            for key, path in paths.items()
            if key not in {"result_json", "manifest"}
        },
        "input_sha256": result["inputs"],
    }
    _write_json(paths["manifest"], manifest)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discovery-picks",
        type=Path,
        default=ROOT
        / "research/model_v06_feature_result_group_screen_display_picks.csv",
    )
    parser.add_argument(
        "--replay-picks", type=Path, default=Path("/tmp/v07_G0_replay.csv")
    )
    parser.add_argument(
        "--e0-picks", type=Path, default=Path("/tmp/v07_G0_E0_replay.csv")
    )
    parser.add_argument(
        "--panel", type=Path, default=Path("/tmp/model_v06_feature_panel.pkl")
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "research/model_v07_tail_risk_result",
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args(argv)
    result = run(
        discovery_path=args.discovery_picks,
        replay_path=args.replay_picks,
        e0_path=args.e0_picks,
        panel_path=args.panel,
        output_prefix=args.output_prefix,
        protocol_path=args.protocol,
    )
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
