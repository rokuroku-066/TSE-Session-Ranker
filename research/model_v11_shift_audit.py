#!/usr/bin/env python3
"""Independently rejoin and audit structural-shift policy P&L."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from research.model_v11_distributional_audit import (
    independently_join_outcomes,
    independent_metrics,
    numeric_differences,
)
from research.model_v11_distributional_runner import (
    policy_id,
    read_json,
    sha256_file,
    write_json,
)


PANEL_SHA256 = "6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb"
PROTOCOL_SHA256 = "76d8c4129f57526c6d0a4d826779414c3ee4dcba48e818c72697ba8d3627a0cc"
CONTROL = "C00_DAILY_RANK_RIDGE"
CANDIDATES = (
    "S01_DENSITY_RATIO_RECENT",
    "S02_COVARIATE_CHANGEPOINT_RESET",
    "S03_WORST_MONTH_GROUP_DRO",
    "S04_ERA_INVARIANT_SIGN",
    "S05_LEAVE_ONE_STATE_OUT_MEDIAN_COEF",
    "S06_UNSUPERVISED_LATENT_EXPERTS",
    "S07_RECENT_SUPPORT_CONFORMAL",
    "S08_ENVIRONMENT_RESIDUALISED",
)
CAPACITIES = (1, 2)


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
        "# model_v11_shift zero-base screen",
        "",
        "## Decision",
        "",
        decision["summary"],
        "",
        "This is a retrospective mechanism screen on a panel already available to "
        "the wider project. A retrospective passer could freeze only one exact "
        "forward-shadow specification; it cannot authorise production.",
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
    lines.extend(["", "## Gate failures by policy", ""])
    for key in ordered:
        checks = gates[key]["checks_before_independent_pnl_audit"]
        failed = [name for name, passed in checks.items() if not passed]
        if not gates[key]["independent_pnl_audit_exact"]:
            failed.append("independent_pnl_audit_exact")
        lines.append(f"- `{key}`: {', '.join(failed) if failed else 'none'}")
    lines.extend(
        [
            "",
            "## Mechanism rationale and limits",
            "",
            "- Han, Huang, and Wang study model assessment and selection under "
            "[temporal distribution shift](https://proceedings.mlr.press/v235/han24b.html), "
            "including synthesis of current and historical epochs. Their adaptive "
            "rolling-window method is not reproduced here: window or candidate "
            "selection on these outcomes was explicitly prohibited. The paper "
            "supports treating non-stationarity as a model-selection problem, not "
            "the profitability of any registered policy.",
            "- Fujii and Yasumura report a Japanese-stock prediction design based on "
            "[autoencoder change-point detection](https://www.jstage.jst.go.jp/article/jsaisigtwo/2024/SAI-051/2024_04/_article/-char/ja). "
            "It motivates an outcome-free structural-break hypothesis in this "
            "domain, but does not validate this protocol's multivariate mean-contrast "
            "detector or its trading economics.",
            "- Kim et al. introduce continual causal/uplift tasks under "
            "[temporal and domain shifts](https://proceedings.mlr.press/v208/kim23a.html). "
            "That work motivates evaluating mechanisms across changing environments; "
            "its observational uplift setting is materially different from same-day "
            "return ranking.",
            "- Zhou et al. show that "
            "[Group-DRO can fail when predefined groups do not capture the relevant "
            "spurious correlations](https://arxiv.org/abs/2106.07171). "
            "Calendar months are therefore a falsifiable environment definition, "
            "not an assumed sufficient partition. A positive or negative S03 result "
            "cannot establish causal invariance.",
            "",
            "These sources were fixed as mechanism context and limitations. They were "
            "not used to change candidates, thresholds, gates, or interpretation after "
            "seeing this family's returns.",
            "",
            "## Integrity",
            "",
            f"- Independent panel rejoin and P&L reproduction: "
            f"{'exact' if result['independent_audit']['exact'] else 'FAILED'}",
            f"- Embedded outcomes match independent panel values: "
            f"{'yes' if result['independent_audit']['embedded_outcomes_exact'] else 'no'}",
            f"- Target-day outcome mutation: "
            f"{'exact' if result['integrity']['target_day_outcome_mutation_exact'] else 'FAILED'}",
            f"- Monthly expanding folds: {result['integrity']['monthly_expanding_folds']}",
            f"- Candidate policies in familywise test: "
            f"{result['multiplicity']['family_size']}",
            "",
            "## Production gate",
            "",
            "A retrospective passer must remain unchanged for at least 60 new "
            "sessions and pass the preregistered forward mean, block-bootstrap, "
            "three-slice, tail-removal, and independent-audit checks. No candidate "
            "in this report is production-authorised.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "PYTHONPATH=src:. .venv/bin/python research/model_v11_shift_runner.py",
            "PYTHONPATH=src:. .venv/bin/python research/model_v11_shift_audit.py",
            "```",
            "",
            f"- Panel SHA-256: `{result['input']['panel_sha256']}`",
            f"- Protocol SHA-256: `{result['input']['protocol_sha256']}`",
            f"- Runner SHA-256: `{result['input']['runner_sha256']}`",
            f"- Evaluation dependency SHA-256: "
            f"`{result['input']['evaluation_dependency_sha256']}`",
            f"- Picks SHA-256: `{result['artifacts']['picks_sha256']}`",
            f"- Audit SHA-256: `{result['independent_audit']['audit_sha256']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="/tmp/model_v07_corrected_panel.pkl")
    parser.add_argument("--protocol", default="research/model_v11_shift_protocol.json")
    parser.add_argument("--result", default="research/model_v11_shift_result.json")
    parser.add_argument("--picks", default="research/model_v11_shift_picks.csv")
    parser.add_argument(
        "--audit-output", default="research/model_v11_shift_audit.json"
    )
    parser.add_argument("--report", default="research/model_v11_shift_report.md")
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
        raise ValueError("panel SHA-256 differs from lock")
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("protocol SHA-256 differs from lock")
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
    panel_dates = pd.to_datetime(panel["date"])
    sessions = pd.DatetimeIndex(
        panel_dates.loc[
            panel_dates.between("2024-07-01", "2025-07-31")
        ]
        .drop_duplicates()
        .sort_values()
    )
    if len(sessions) != 266:
        raise ValueError("score-session count differs from protocol")
    expected_policies = {
        policy_id(base, capacity)
        for base in (CONTROL, *CANDIDATES)
        for capacity in CAPACITIES
    }
    if set(audited_picks["policy_id"].dropna().astype(str)) != expected_policies:
        raise ValueError("policy set differs from protocol")
    differences: list[dict[str, Any]] = []
    for key in sorted(expected_policies):
        frame = audited_picks.loc[audited_picks["policy_id"] == key].copy()
        capacity = int(key.rsplit("_K", 1)[1])
        if len(frame) != len(sessions) * capacity:
            raise ValueError(f"{key} has an invalid slot count")
        audited_metric, _ = independent_metrics(frame, sessions)
        differences.extend(
            numeric_differences(result["metrics"][key], audited_metric, path=key)
        )
    exact = embedded_exact and not differences
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
            "every preregistered return, familywise, stability, tail, coverage, "
            "concentration, mutation, and independent-P&L gate. No forward-shadow "
            "finalist was frozen and production remains unchanged."
        )
    else:
        summary = (
            f"{len(passers)} policy or policies passed every retrospective gate. "
            f"{finalist} had the highest net40 mean among them and is frozen only "
            "as a forward-shadow finalist. Production remains unchanged until the "
            "separate prospective gate passes."
        )
    common_audit = Path(__file__).with_name("model_v11_distributional_audit.py")
    audit = {
        "schema_version": 1,
        "protocol_id": result["protocol_id"],
        "panel_sha256": PANEL_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "result_sha256_before_audit_finalisation": result_before_sha,
        "picks_sha256": sha256_file(picks_path),
        "audit_script_sha256": sha256_file(Path(__file__)),
        "audit_dependency_sha256": sha256_file(common_audit),
        "policy_count": len(expected_policies),
        "candidate_policy_count": len(CANDIDATES) * len(CAPACITIES),
        "score_sessions": len(sessions),
        "embedded_outcomes_exact": embedded_exact,
        "numeric_comparison_tolerance": 1e-12,
        "metric_differences": differences,
        "metric_difference_count": len(differences),
        "independent_pnl_exact": exact,
        "retrospective_gate_passers": passers,
        "forward_shadow_finalist": finalist,
        "production_model_changed": False,
    }
    write_json(audit_path, audit)
    result["independent_audit"] = {
        "exact": exact,
        "embedded_outcomes_exact": embedded_exact,
        "metric_difference_count": len(differences),
        "audit_path": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "audit_script_sha256": sha256_file(Path(__file__)),
        "audit_dependency_sha256": sha256_file(common_audit),
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
