#!/usr/bin/env python3
"""Independently rejoin and audit v1.1 distributional policy P&L.

This module deliberately does not import the research runner. Selection keys
are joined back to the content-addressed panel, embedded CSV outcomes are
ignored for recomputation, and only then is the retrospective decision
finalised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PANEL_SHA256 = "6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb"
PROTOCOL_SHA256 = "9e7b98550f3066ee6b6eb8ec0ebdaf8120d211020b7c24be29618786e124e480"
CONTROL = "C00_DAILY_RANK_RIDGE"
CANDIDATES = (
    "D01_CODE_EB_RESIDUAL",
    "D02_SURVIVAL_INTEGRAL",
    "D03_DOWNSIDE_FEASIBLE_MEAN",
    "D04_NORMALISED_CONFORMAL_LCB",
    "D05_FOREST_TREE_LCB",
    "D06_EVT_RESIDUAL_ES",
    "D07_REGIME_ABSTAIN",
    "D08_SLOTWISE_CASH_STOP",
)
CAPACITIES = (1, 2)
PERIODS = {
    "discovery": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-31")),
    "confirmation_a": (pd.Timestamp("2024-11-01"), pd.Timestamp("2025-03-31")),
    "confirmation_b": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def policy_id(base_id: str, capacity: int) -> str:
    return f"{base_id}_K{capacity}"


def independently_join_outcomes(
    picks: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    keys = picks[
        [
            "date",
            "model_rank",
            "code",
            "policy_id",
            "hypothesis_id",
            "capacity",
        ]
    ].copy()
    keys["_row"] = np.arange(len(keys))
    panel_outcomes = panel[["date", "code", "oc_return_pct", "label"]].copy()
    panel_outcomes["date"] = pd.to_datetime(panel_outcomes["date"])
    panel_outcomes["code"] = panel_outcomes["code"].astype("string")
    if panel_outcomes.duplicated(["date", "code"]).any():
        raise ValueError("panel outcome keys are not unique")
    noncash = keys.loc[keys["code"].notna()].merge(
        panel_outcomes,
        on=["date", "code"],
        how="left",
        validate="many_to_one",
        sort=False,
        indicator=True,
    )
    if noncash["_merge"].ne("both").any():
        missing = noncash.loc[
            noncash["_merge"].ne("both"), ["date", "code"]
        ].head()
        raise ValueError(f"selected keys do not exist in the panel:\n{missing}")
    audited = keys.copy()
    audited["oc_return_pct"] = np.nan
    audited["label"] = np.nan
    audited.loc[noncash["_row"].to_numpy(int), "oc_return_pct"] = noncash[
        "oc_return_pct"
    ].to_numpy(float)
    audited.loc[noncash["_row"].to_numpy(int), "label"] = noncash["label"].to_numpy(
        float
    )
    original_return = pd.to_numeric(picks["oc_return_pct"], errors="coerce").to_numpy()
    original_label = pd.to_numeric(picks["label"], errors="coerce").to_numpy()
    embedded_exact = bool(
        np.allclose(
            np.nan_to_num(original_return, nan=987654.0),
            np.nan_to_num(audited["oc_return_pct"].to_numpy(float), nan=987654.0),
            atol=1e-12,
            rtol=1e-12,
        )
        and np.allclose(
            np.nan_to_num(original_label, nan=987654.0),
            np.nan_to_num(audited["label"].to_numpy(float), nan=987654.0),
            atol=1e-12,
            rtol=1e-12,
        )
    )
    return audited.drop(columns="_row"), embedded_exact


def independent_daily(
    frame: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    cost_bps: float,
) -> pd.DataFrame:
    capacity_values = frame["capacity"].drop_duplicates()
    if len(capacity_values) != 1:
        raise ValueError("policy mixes capacities")
    capacity = int(capacity_values.iloc[0])
    executed = frame["code"].notna() & frame["oc_return_pct"].notna()
    slot_weight = 1.0 / capacity
    contributions = pd.DataFrame(
        {
            "date": frame["date"],
            "gross": slot_weight * frame["oc_return_pct"].fillna(0.0),
            "cost": slot_weight * executed.astype(float) * cost_bps / 100.0,
            "exposure": slot_weight * executed.astype(float),
        }
    )
    day = contributions.groupby("date", sort=True).sum()
    day["net"] = day["gross"] - day["cost"]
    return day.reindex(sessions, fill_value=0.0)[["gross", "net", "exposure"]]


def independent_profit_factor(values: pd.Series) -> float | None:
    positive = float(values.where(values > 0.0, 0.0).sum())
    negative = float(-values.where(values < 0.0, 0.0).sum())
    return float(positive / negative) if negative > 0.0 else None


def independent_metrics(
    frame: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> tuple[dict[str, Any], pd.DataFrame]:
    capacity = int(frame["capacity"].iloc[0])
    slot_weight = 1.0 / capacity
    by_cost = {
        cost: independent_daily(frame, sessions, float(cost))
        for cost in (20, 40, 60)
    }
    executed = frame["code"].notna() & frame["oc_return_pct"].notna()
    traded_indicator = (
        frame.assign(_executed=executed)
        .groupby("date", sort=True)["_executed"]
        .any()
        .reindex(sessions, fill_value=False)
    )
    net20 = by_cost[20]["net"]
    net40 = by_cost[40]["net"]
    month40 = net40.groupby(net40.index.to_period("M")).mean()
    q05 = float(net40.quantile(0.05))
    selected = frame.loc[executed, ["date", "code", "oc_return_pct"]].copy()
    selected["weight"] = slot_weight
    selected["net20_contribution"] = slot_weight * (
        selected["oc_return_pct"] - 0.20
    )
    if selected.empty:
        concentration = {
            "unique_codes": 0,
            "largest_weight_share": None,
            "top10_weight_share": None,
            "largest_positive_net20_pnl_share": None,
            "top10_positive_pnl_codes_to_cash_net20_pct": 0.0,
        }
    else:
        weights = selected.groupby("code", sort=False)["weight"].sum().sort_values(
            ascending=False
        )
        pnl = selected.groupby("code", sort=False)["net20_contribution"].sum()
        positive_pnl = pnl.where(pnl > 0.0, 0.0)
        removed_codes = set(positive_pnl.nlargest(10).index)
        removed = frame.copy()
        removed.loc[
            removed["code"].isin(removed_codes),
            ["code", "label", "oc_return_pct"],
        ] = np.nan
        concentration = {
            "unique_codes": int(selected["code"].nunique()),
            "largest_weight_share": float(weights.iloc[0] / weights.sum()),
            "top10_weight_share": float(weights.head(10).sum() / weights.sum()),
            "largest_positive_net20_pnl_share": (
                float(positive_pnl.max() / positive_pnl.sum())
                if positive_pnl.sum() > 0.0
                else None
            ),
            "top10_positive_pnl_codes_to_cash_net20_pct": float(
                independent_daily(removed, sessions, 20.0)["net"].mean()
            ),
        }
    metrics = {
        "capacity": capacity,
        "scheduled_days": int(len(sessions)),
        "scheduled_slots": int(len(sessions) * capacity),
        "executed_slots": int(executed.sum()),
        "executed_slot_fraction": float(executed.sum() / (len(sessions) * capacity)),
        "traded_days": int(traded_indicator.sum()),
        "cash_days": int(len(sessions) - traded_indicator.sum()),
        "hit_rate_executed": (
            float(frame.loc[executed, "label"].mean()) if executed.any() else None
        ),
        "gross_mean_pct": float(by_cost[20]["gross"].mean()),
        "net_mean_pct": {
            str(cost): float(values["net"].mean())
            for cost, values in by_cost.items()
        },
        "net40_profit_factor": independent_profit_factor(net40),
        "monthly_net40_pct": {
            str(key): float(value) for key, value in month40.items()
        },
        "positive_months_net40": int((month40 > 0.0).sum()),
        "months": int(len(month40)),
        "temporal_slices_net40_pct": {
            name: float(net40.loc[start:end].mean())
            for name, (start, end) in PERIODS.items()
        },
        "winning_days_removed_net20_pct": {
            str(count): float(net20.drop(net20.nlargest(count).index).mean())
            for count in (5, 10, 20)
        },
        "losing_days_removed_net20_pct": {
            str(count): float(net20.drop(net20.nsmallest(count).index).mean())
            for count in (5, 10, 20)
        },
        "expected_shortfall05_net40_pct": float(
            net40.loc[net40 <= q05].mean()
        ),
        "p05_net40_pct": q05,
        "worst_day_net40_pct": float(net40.min()),
        "code_concentration": concentration,
    }
    export = pd.DataFrame(
        {
            "date": sessions,
            "net20": by_cost[20]["net"].to_numpy(float),
            "net40": net40.to_numpy(float),
            "net60": by_cost[60]["net"].to_numpy(float),
        }
    )
    return metrics, export


def numeric_differences(
    expected: Any,
    actual: Any,
    path: str = "",
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            differences.append(
                {
                    "path": path,
                    "expected_keys": sorted(expected),
                    "actual_keys": sorted(actual),
                }
            )
            return differences
        for key in expected:
            differences.extend(
                numeric_differences(
                    expected[key],
                    actual[key],
                    f"{path}.{key}" if path else str(key),
                )
            )
        return differences
    if expected is None or actual is None:
        if expected is not actual:
            differences.append(
                {"path": path, "expected": expected, "actual": actual}
            )
        return differences
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not np.isclose(float(expected), float(actual), atol=1e-12, rtol=1e-12):
            differences.append(
                {
                    "path": path,
                    "expected": expected,
                    "actual": actual,
                    "absolute_difference": abs(float(expected) - float(actual)),
                }
            )
        return differences
    if expected != actual:
        differences.append({"path": path, "expected": expected, "actual": actual})
    return differences


def write_report(path: Path, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    gates = result["retrospective_gates"]
    multiplicity = result["multiplicity"]["hypotheses"]
    candidate_policies = [
        policy_id(candidate, capacity)
        for candidate in CANDIDATES
        for capacity in CAPACITIES
    ]
    ordered = sorted(
        candidate_policies,
        key=lambda key: metrics[key]["net_mean_pct"]["40"],
        reverse=True,
    )
    decision = result["decision"]
    lines = [
        "# model_v11_distributional zero-base screen",
        "",
        "## Decision",
        "",
        decision["summary"],
        "",
        "This panel is retrospective and previously available to the project. "
        "Passing the retrospective gate can freeze only one exact forward-shadow "
        "candidate; it cannot change production.",
        "",
        "## All preregistered candidate policies",
        "",
        "| Policy | Net20 | Net40 | Net60 | Months + | Slice min | Top20 removed | ES05 net40 | Exec frac | Days | Codes | FW95 lower uplift | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ordered:
        value = metrics[key]
        concentration = value["code_concentration"]
        lower = multiplicity[key][
            "familywise_max_t_one_sided95_lower_pct"
        ]
        passed = gates[key]["passes_all_retrospective_gates"]
        lines.append(
            f"| {key} "
            f"| {value['net_mean_pct']['20']:+.4f}% "
            f"| {value['net_mean_pct']['40']:+.4f}% "
            f"| {value['net_mean_pct']['60']:+.4f}% "
            f"| {value['positive_months_net40']}/{value['months']} "
            f"| {min(value['temporal_slices_net40_pct'].values()):+.4f}% "
            f"| {value['winning_days_removed_net20_pct']['20']:+.4f}% "
            f"| {value['expected_shortfall05_net40_pct']:+.4f}% "
            f"| {value['executed_slot_fraction']:.3f} "
            f"| {value['traded_days']} "
            f"| {concentration['unique_codes']} "
            f"| {lower:+.4f}% "
            f"| {'PASS' if passed else 'reject'} |"
        )
    lines.extend(
        [
            "",
            "## Capacity-matched controls",
            "",
            "| Policy | Net20 | Net40 | Net60 | Months + | Top20 removed | ES05 net40 | Codes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for capacity in CAPACITIES:
        key = policy_id(CONTROL, capacity)
        value = metrics[key]
        lines.append(
            f"| {key} "
            f"| {value['net_mean_pct']['20']:+.4f}% "
            f"| {value['net_mean_pct']['40']:+.4f}% "
            f"| {value['net_mean_pct']['60']:+.4f}% "
            f"| {value['positive_months_net40']}/{value['months']} "
            f"| {value['winning_days_removed_net20_pct']['20']:+.4f}% "
            f"| {value['expected_shortfall05_net40_pct']:+.4f}% "
            f"| {value['code_concentration']['unique_codes']} |"
        )
    lines.extend(
        [
            "",
            "## Gate failures by policy",
            "",
        ]
    )
    for key in ordered:
        checks = gates[key]["checks_before_independent_pnl_audit"]
        failed = [name for name, passed in checks.items() if not passed]
        audit_passed = gates[key]["independent_pnl_audit_exact"]
        if not audit_passed:
            failed.append("independent_pnl_audit_exact")
        lines.append(f"- `{key}`: {', '.join(failed) if failed else 'none'}")
    audit = result["independent_audit"]
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Independent panel rejoin and P&L reproduction: "
            f"{'exact' if audit['exact'] else 'FAILED'}",
            f"- Embedded pick outcomes equal independently joined outcomes: "
            f"{'yes' if audit['embedded_outcomes_exact'] else 'no'}",
            f"- Target-day outcome mutation invariance: "
            f"{'exact' if result['integrity']['target_day_outcome_mutation_exact'] else 'FAILED'}",
            f"- Monthly expanding folds: {result['integrity']['monthly_expanding_folds']}",
            f"- Candidate policies in the familywise test: "
            f"{result['multiplicity']['family_size']}",
            "",
            "## Production gate",
            "",
            "Even a retrospective passer must remain unchanged for at least 60 new "
            "sessions and pass the preregistered forward mean, block-bootstrap, "
            "three-slice, tail-removal, and independent-audit checks. No candidate "
            "in this report is production-authorised.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "PYTHONPATH=src:. .venv/bin/python research/model_v11_distributional_runner.py",
            "PYTHONPATH=src:. .venv/bin/python research/model_v11_distributional_audit.py",
            "```",
            "",
            f"- Panel SHA-256: `{result['input']['panel_sha256']}`",
            f"- Protocol SHA-256: `{result['input']['protocol_sha256']}`",
            f"- Runner SHA-256: `{result['input']['runner_sha256']}`",
            f"- Picks SHA-256: `{result['artifacts']['picks_sha256']}`",
            f"- Audit SHA-256: `{audit['audit_sha256']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="/tmp/model_v07_corrected_panel.pkl")
    parser.add_argument(
        "--protocol", default="research/model_v11_distributional_protocol.json"
    )
    parser.add_argument(
        "--result", default="research/model_v11_distributional_result.json"
    )
    parser.add_argument(
        "--picks", default="research/model_v11_distributional_picks.csv"
    )
    parser.add_argument(
        "--audit-output", default="research/model_v11_distributional_audit.json"
    )
    parser.add_argument(
        "--report", default="research/model_v11_distributional_report.md"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel).resolve()
    protocol_path = Path(args.protocol).resolve()
    result_path = Path(args.result).resolve()
    picks_path = Path(args.picks).resolve()
    audit_path = Path(args.audit_output).resolve()
    report_path = Path(args.report).resolve()
    if sha256_file(panel_path) != PANEL_SHA256:
        raise ValueError("panel SHA-256 differs from audit lock")
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("protocol SHA-256 differs from audit lock")
    result_before_sha = sha256_file(result_path)
    result = read_json(result_path)
    if result["input"]["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("result refers to another protocol")
    if sha256_file(picks_path) != result["artifacts"]["picks_sha256"]:
        raise ValueError("picks SHA-256 differs from runner result")
    picks = pd.read_csv(
        picks_path,
        dtype={
            "code": "string",
            "policy_id": "string",
            "hypothesis_id": "string",
        },
        parse_dates=["date"],
    )
    panel = joblib.load(panel_path, mmap_mode="r")
    audited_picks, embedded_exact = independently_join_outcomes(picks, panel)
    score_start = pd.Timestamp("2024-07-01")
    score_end = pd.Timestamp("2025-07-31")
    sessions = pd.DatetimeIndex(
        pd.to_datetime(panel["date"])
        .loc[
            pd.to_datetime(panel["date"]).between(score_start, score_end)
        ]
        .drop_duplicates()
        .sort_values()
    )
    if len(sessions) != 266:
        raise ValueError("audit score-session count differs from protocol")
    expected_policies = {
        policy_id(base, capacity)
        for base in (CONTROL, *CANDIDATES)
        for capacity in CAPACITIES
    }
    actual_policies = set(audited_picks["policy_id"].dropna().astype(str))
    if actual_policies != expected_policies:
        raise ValueError("pick policy set differs from protocol")
    audited_metrics: dict[str, Any] = {}
    audited_daily: dict[str, pd.DataFrame] = {}
    all_differences: list[dict[str, Any]] = []
    for key in sorted(expected_policies):
        frame = audited_picks.loc[audited_picks["policy_id"] == key].copy()
        capacity = int(key.rsplit("_K", 1)[1])
        if len(frame) != len(sessions) * capacity:
            raise ValueError(f"{key} slot count differs from protocol")
        audited_metrics[key], audited_daily[key] = independent_metrics(frame, sessions)
        differences = numeric_differences(
            result["metrics"][key], audited_metrics[key], path=key
        )
        all_differences.extend(differences)
    exact = len(all_differences) == 0 and embedded_exact
    passers: list[str] = []
    for key, gate in result["retrospective_gates"].items():
        gate["independent_pnl_audit_exact"] = exact
        gate["passes_all_retrospective_gates"] = bool(
            gate["passes_before_independent_pnl_audit"] and exact
        )
        if gate["passes_all_retrospective_gates"]:
            passers.append(key)
    finalist = (
        max(passers, key=lambda key: result["metrics"][key]["net_mean_pct"]["40"])
        if passers
        else None
    )
    point_winner = result["decision_before_independent_audit"][
        "highest_unadjusted_net40_policy"
    ]
    if finalist is None:
        summary = (
            f"The unadjusted point winner was {point_winner}, but no policy passed "
            "every preregistered retrospective return, familywise, stability, "
            "tail, coverage, concentration, mutation, and independent-P&L gate. "
            "No forward-shadow finalist was frozen and production remains unchanged."
        )
    else:
        summary = (
            f"{len(passers)} policy or policies passed every retrospective gate. "
            f"{finalist} had the highest net40 mean among them and is frozen only "
            "as a prospective shadow finalist. Production remains unchanged until "
            "the separate 60-session forward gate passes."
        )
    audit = {
        "schema_version": 1,
        "protocol_id": result["protocol_id"],
        "panel_sha256": PANEL_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "result_sha256_before_audit_finalisation": result_before_sha,
        "picks_sha256": sha256_file(picks_path),
        "audit_script_sha256": sha256_file(Path(__file__)),
        "policy_count": int(len(expected_policies)),
        "candidate_policy_count": int(len(CANDIDATES) * len(CAPACITIES)),
        "score_sessions": int(len(sessions)),
        "embedded_outcomes_exact": embedded_exact,
        "numeric_comparison_tolerance": 1e-12,
        "metric_differences": all_differences,
        "metric_difference_count": int(len(all_differences)),
        "independent_pnl_exact": exact,
        "retrospective_gate_passers": passers,
        "forward_shadow_finalist": finalist,
        "production_model_changed": False,
    }
    write_json(audit_path, audit)
    result["independent_audit"] = {
        "exact": exact,
        "embedded_outcomes_exact": embedded_exact,
        "metric_difference_count": int(len(all_differences)),
        "audit_path": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "audit_script_sha256": sha256_file(Path(__file__)),
    }
    result["decision"] = {
        "highest_unadjusted_net40_policy": point_winner,
        "highest_unadjusted_net40_pct": result["metrics"][point_winner][
            "net_mean_pct"
        ]["40"],
        "retrospective_gate_passers": passers,
        "forward_shadow_finalist": finalist,
        "production_action": "none",
        "production_model_changed": False,
        "summary": summary,
    }
    result["artifacts"]["audit_path"] = str(audit_path)
    result["artifacts"]["report_path"] = str(report_path)
    write_json(result_path, result)
    write_report(report_path, result)
    print(
        f"audit exact={exact} passers={passers} finalist={finalist}",
        flush=True,
    )


if __name__ == "__main__":
    main()
