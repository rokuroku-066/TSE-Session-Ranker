#!/usr/bin/env python3
"""Independent P&L and multiplicity audit for model_v11_new_data.

This module intentionally does not import the experiment runner.  It rebuilds
portfolio returns from the serialized slot-level picks and compares every
reported P&L statistic at a strict numerical tolerance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "research" / "model_v11_new_data_protocol.json"
RESULT_PATH = ROOT / "research" / "model_v11_new_data_result.json"
PICKS_PATH = ROOT / "research" / "model_v11_new_data_picks.csv"
AUDIT_PATH = ROOT / "research" / "model_v11_new_data_audit.json"
TOLERANCE = 1e-12
COSTS_BPS = (0.0, 20.0, 40.0, 60.0)
TOP_K_VALUES = (1, 2)
CANDIDATES = (
    "ND04_simultaneous_adverse_bundle",
    "ND05_release_clock_and_fiscal_horizon",
    "ND04_plus_ND05",
)
ALL_POLICIES = ("L4_price_control", *CANDIDATES)
SLICES = {
    "previously_unused_early_2024": (
        pd.Timestamp("2024-05-01"),
        pd.Timestamp("2024-06-30"),
    ),
    "later_reference": (
        pd.Timestamp("2024-07-01"),
        pd.Timestamp("2025-07-31"),
    ),
    "2024_reference": (
        pd.Timestamp("2024-07-01"),
        pd.Timestamp("2024-07-31"),
    ),
    "2025_reference": (
        pd.Timestamp("2025-04-01"),
        pd.Timestamp("2025-07-31"),
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: Any, right: Any, label: str) -> None:
    if left is None or right is None:
        if left is not None or right is not None:
            raise AssertionError(f"{label}: null mismatch {left!r} != {right!r}")
        return
    if not np.isclose(float(left), float(right), atol=TOLERANCE, rtol=0.0):
        raise AssertionError(f"{label}: {left!r} != {right!r}")


def daily_returns(frame: pd.DataFrame, top_k: int, cost_bps: float) -> pd.Series:
    executed = frame["oc_return_pct"].notna()
    slot_gross_pct = frame["oc_return_pct"].fillna(0.0)
    slot_cost_pct = executed.astype(float) * cost_bps / 100.0
    slot_net_pct = slot_gross_pct - slot_cost_pct
    return slot_net_pct.groupby(frame["date"], sort=True).sum().div(top_k)


def recompute_metrics(frame: pd.DataFrame, top_k: int) -> dict[str, Any]:
    by_cost: dict[str, Any] = {}
    daily_by_cost: dict[float, pd.Series] = {}
    for cost in COSTS_BPS:
        daily = daily_returns(frame, top_k, cost)
        daily_by_cost[cost] = daily
        by_cost[str(int(cost))] = {
            "days": int(len(daily)),
            "mean_pct": float(daily.mean()),
            "median_pct": float(daily.median()),
            "win_rate": float(daily.gt(0).mean()),
            "worst_day_pct": float(daily.min()),
        }
    daily20 = daily_by_cost[20.0]
    daily40 = daily_by_cost[40.0]
    monthly = daily40.groupby(daily40.index.to_period("M")).mean()
    slices: dict[str, Any] = {}
    for name, (start, end) in SLICES.items():
        values = daily40.loc[(daily40.index >= start) & (daily40.index <= end)]
        slices[name] = {
            "days": int(len(values)),
            "net40_mean_pct": float(values.mean()) if len(values) else None,
        }
    removed = daily20.drop(daily20.nlargest(min(20, len(daily20))).index)
    executed = frame["oc_return_pct"].notna()
    contribution = (
        frame["oc_return_pct"].fillna(0.0) - executed.astype(float) * 0.20
    ).div(top_k)
    by_code = (
        contribution.groupby(frame["code"], dropna=True).sum().sort_values(ascending=False)
    )
    positive_code = by_code.loc[by_code.gt(0)]
    top_codes = [str(value) for value in positive_code.head(10).index]
    replaced = contribution.mask(frame["code"].isin(top_codes), 0.0)
    replaced_daily = replaced.groupby(frame["date"], sort=True).sum()
    top_share = (
        float(positive_code.head(10).sum() / positive_code.sum())
        if float(positive_code.sum()) > 0
        else None
    )
    return {
        "costs_bps": by_cost,
        "monthly_net40_mean_pct": {
            str(period): float(value) for period, value in monthly.items()
        },
        "positive_months": int(monthly.gt(0).sum()),
        "months": int(len(monthly)),
        "slices": slices,
        "top20_winning_days_removed_net20_mean_pct": (
            float(removed.mean()) if len(removed) else None
        ),
        "top10_profit_codes_cash_net20_mean_pct": float(replaced_daily.mean()),
        "top10_profit_codes": top_codes,
        "top10_positive_profit_share": top_share,
        "filled_slots": int(executed.sum()),
        "scheduled_slots": int(len(frame)),
        "slot_fill_rate": float(executed.mean()),
        "days_with_at_least_one_filled_slot": int(
            executed.groupby(frame["date"], sort=True).any().sum()
        ),
        "unique_codes": int(frame["code"].nunique(dropna=True)),
    }


def compare_metrics(
    actual: dict[str, Any],
    expected: dict[str, Any],
    prefix: str,
) -> int:
    checks = 0
    for cost in COSTS_BPS:
        key = str(int(cost))
        for name in ("mean_pct", "median_pct", "win_rate", "worst_day_pct"):
            close(
                actual["costs_bps"][key][name],
                expected["costs_bps"][key][name],
                f"{prefix}.costs_bps.{key}.{name}",
            )
            checks += 1
        if actual["costs_bps"][key]["days"] != expected["costs_bps"][key]["days"]:
            raise AssertionError(f"{prefix}.costs_bps.{key}.days mismatch")
        checks += 1
    if actual["monthly_net40_mean_pct"].keys() != expected[
        "monthly_net40_mean_pct"
    ].keys():
        raise AssertionError(f"{prefix}.monthly keys mismatch")
    for month, value in actual["monthly_net40_mean_pct"].items():
        close(
            value,
            expected["monthly_net40_mean_pct"][month],
            f"{prefix}.monthly.{month}",
        )
        checks += 1
    for name in ("positive_months", "months"):
        if actual[name] != expected[name]:
            raise AssertionError(f"{prefix}.{name} mismatch")
        checks += 1
    for slice_name in SLICES:
        if actual["slices"][slice_name]["days"] != expected["slices"][slice_name]["days"]:
            raise AssertionError(f"{prefix}.slices.{slice_name}.days mismatch")
        close(
            actual["slices"][slice_name]["net40_mean_pct"],
            expected["slices"][slice_name]["net40_mean_pct"],
            f"{prefix}.slices.{slice_name}.net40_mean_pct",
        )
        checks += 2
    for name in (
        "top20_winning_days_removed_net20_mean_pct",
        "top10_profit_codes_cash_net20_mean_pct",
        "top10_positive_profit_share",
        "slot_fill_rate",
    ):
        close(actual[name], expected[name], f"{prefix}.{name}")
        checks += 1
    for name in (
        "filled_slots",
        "scheduled_slots",
        "days_with_at_least_one_filled_slot",
        "unique_codes",
    ):
        if actual[name] != expected[name]:
            raise AssertionError(f"{prefix}.{name} mismatch")
        checks += 1
    if actual["top10_profit_codes"] != expected["top10_profit_codes"]:
        raise AssertionError(f"{prefix}.top10_profit_codes mismatch")
    checks += 1
    return checks


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for index, key in enumerate(ordered):
        value = min(1.0, (family_size - index) * p_values[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    picks = pd.read_csv(
        PICKS_PATH,
        parse_dates=["date"],
        dtype={"code": str, "candidate_id": str},
    )
    if result["protocol_sha256"] != sha256_file(PROTOCOL_PATH):
        raise AssertionError("result protocol hash mismatch")
    if result["artifacts"]["picks_sha256"] != sha256_file(PICKS_PATH):
        raise AssertionError("result picks hash mismatch")
    if result["protocol_id"] != protocol["protocol_id"]:
        raise AssertionError("protocol id mismatch")
    if set(picks["candidate_id"].unique()) != set(ALL_POLICIES):
        raise AssertionError("unexpected candidate policy in picks")
    if set(picks["top_k"].unique()) != set(TOP_K_VALUES):
        raise AssertionError("unexpected top_k in picks")

    structural_checks: list[dict[str, Any]] = []
    recomputed: dict[str, Any] = {}
    numerical_checks = 0
    daily_values: dict[tuple[str, int], pd.Series] = {}
    canonical_sessions: pd.DatetimeIndex | None = None
    for candidate_id in ALL_POLICIES:
        recomputed[candidate_id] = {}
        for top_k in TOP_K_VALUES:
            local = picks.loc[
                picks["candidate_id"].eq(candidate_id)
                & picks["top_k"].eq(top_k)
            ].copy()
            counts = local.groupby("date", sort=True).size()
            if not counts.eq(top_k).all():
                raise AssertionError(f"{candidate_id} top{top_k}: wrong slots per day")
            expected_ranks = set(range(1, top_k + 1))
            rank_sets = local.groupby("date", sort=True)["model_rank"].apply(set)
            if not rank_sets.map(lambda value: value == expected_ranks).all():
                raise AssertionError(f"{candidate_id} top{top_k}: rank set mismatch")
            duplicated_codes = (
                local.dropna(subset=["code"])
                .duplicated(["date", "code"], keep=False)
                .any()
            )
            if duplicated_codes:
                raise AssertionError(f"{candidate_id} top{top_k}: duplicate code")
            if not local["executed"].eq(local["oc_return_pct"].notna()).all():
                raise AssertionError(f"{candidate_id} top{top_k}: execution flag mismatch")
            sessions = pd.DatetimeIndex(counts.index)
            if canonical_sessions is None:
                canonical_sessions = sessions
            elif not sessions.equals(canonical_sessions):
                raise AssertionError(f"{candidate_id} top{top_k}: session set mismatch")
            actual = recompute_metrics(local, top_k)
            expected = result["candidates"][candidate_id][f"top{top_k}"]
            numerical_checks += compare_metrics(
                actual,
                expected,
                f"{candidate_id}.top{top_k}",
            )
            recomputed[candidate_id][f"top{top_k}"] = actual
            daily_values[(candidate_id, top_k)] = daily_returns(local, top_k, 40.0)
            structural_checks.append(
                {
                    "candidate_id": candidate_id,
                    "top_k": top_k,
                    "sessions": int(len(sessions)),
                    "rows": int(len(local)),
                    "slots_per_day_exact": True,
                    "rank_set_exact": True,
                    "duplicate_code_within_day": False,
                    "execution_flag_exact": True,
                }
            )

    raw_p_values: dict[str, float] = {}
    familywise_recomputed: dict[str, Any] = {}
    for top_k in TOP_K_VALUES:
        baseline = daily_values[("L4_price_control", top_k)]
        for candidate_id in CANDIDATES:
            candidate = daily_values[(candidate_id, top_k)]
            aligned = pd.concat(
                [candidate.rename("candidate"), baseline.rename("baseline")],
                axis=1,
                join="inner",
            ).dropna()
            statistic, p_value = ttest_rel(
                aligned["candidate"],
                aligned["baseline"],
                nan_policy="raise",
            )
            key = f"{candidate_id}__top{top_k}"
            raw_p_values[key] = float(p_value)
            familywise_recomputed[key] = {
                "sessions": int(len(aligned)),
                "candidate_net40_mean_pct": float(aligned["candidate"].mean()),
                "L4_net40_mean_pct": float(aligned["baseline"].mean()),
                "delta_net40_mean_pct": float(
                    (aligned["candidate"] - aligned["baseline"]).mean()
                ),
                "paired_t_statistic": float(statistic),
                "raw_two_sided_p_value": float(p_value),
            }
    adjusted = holm_adjust(raw_p_values)
    for key, value in adjusted.items():
        familywise_recomputed[key]["holm_adjusted_p_value"] = float(value)
        familywise_recomputed[key]["reject_familywise_5pct"] = bool(value <= 0.05)
    expected_familywise = result["familywise"]["tests"]
    if familywise_recomputed.keys() != expected_familywise.keys():
        raise AssertionError("familywise test keys mismatch")
    for key, actual in familywise_recomputed.items():
        expected = expected_familywise[key]
        if actual["sessions"] != expected["sessions"]:
            raise AssertionError(f"{key}: paired session count mismatch")
        for name in (
            "candidate_net40_mean_pct",
            "L4_net40_mean_pct",
            "delta_net40_mean_pct",
            "paired_t_statistic",
            "raw_two_sided_p_value",
            "holm_adjusted_p_value",
        ):
            close(actual[name], expected[name], f"familywise.{key}.{name}")
            numerical_checks += 1
        if actual["reject_familywise_5pct"] != expected["reject_familywise_5pct"]:
            raise AssertionError(f"{key}: rejection flag mismatch")
        numerical_checks += 2

    if canonical_sessions is None:
        raise AssertionError("no audited sessions")
    audit = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "audit_status": "pass",
        "independent_of_runner_imports": True,
        "tolerance": TOLERANCE,
        "hashes": {
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "result_sha256": sha256_file(RESULT_PATH),
            "picks_sha256": sha256_file(PICKS_PATH),
            "audit_program_sha256": sha256_file(Path(__file__)),
        },
        "scope": {
            "sessions": int(len(canonical_sessions)),
            "first_session": str(canonical_sessions.min().date()),
            "last_session": str(canonical_sessions.max().date()),
            "policies": list(ALL_POLICIES),
            "top_k": list(TOP_K_VALUES),
            "costs_bps": list(COSTS_BPS),
        },
        "assertions": {
            "slot_level_gross_return_recomputed": True,
            "executed_slot_cost_recomputed": True,
            "equal_weight_divisor_recomputed": True,
            "cash_slots_zero_return_zero_cost": True,
            "all_reported_metrics_match": True,
            "tail_day_removal_matches": True,
            "top_code_cash_replacement_matches": True,
            "monthly_and_slice_metrics_match": True,
            "familywise_tests_match": True,
            "structural_groups_checked": int(len(structural_checks)),
            "numeric_or_scalar_checks": int(numerical_checks),
        },
        "structural_checks": structural_checks,
        "familywise_recomputed": familywise_recomputed,
    }
    AUDIT_PATH.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audit": str(AUDIT_PATH),
                "status": "pass",
                "sessions": int(len(canonical_sessions)),
                "numerical_checks": int(numerical_checks),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
