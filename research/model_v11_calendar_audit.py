#!/usr/bin/env python3
"""Independent audit for the preregistered calendar/institution family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from research import model_v11_uplift_audit as independent  # noqa: E402


PROTOCOL = ROOT / "research" / "model_v11_calendar_protocol.json"
RESULT = ROOT / "research" / "model_v11_calendar_result.json"
PICKS = ROOT / "research" / "model_v11_calendar_picks.csv"
EXPECTED_PROTOCOL_SHA256 = (
    "23b4642e747e52c13ff62ffac36f438b4f88a622bb8cbb7323c7bd0495f58ddf"
)
INDEPENDENT_SLICES = {
    "discovery": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-31")),
    "confirmation_a": (pd.Timestamp("2024-11-01"), pd.Timestamp("2025-03-31")),
    "confirmation_b": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
}


def deterministic_rerun() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v11_calendar_audit_") as directory:
        target = Path(directory)
        result_path = target / "result.json"
        picks_path = target / "picks.csv"
        completed = subprocess.run(
            [
                "python",
                str(ROOT / "research" / "model_v11_calendar_runner.py"),
                "--result",
                str(result_path),
                "--picks",
                str(picks_path),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            return {
                "passes": False,
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-2000:],
            }
        result_equal = result_path.read_bytes() == RESULT.read_bytes()
        picks_equal = picks_path.read_bytes() == PICKS.read_bytes()
        return {
            "passes": result_equal and picks_equal,
            "returncode": 0,
            "result_byte_identical": result_equal,
            "picks_byte_identical": picks_equal,
            "result_sha256": independent.sha256_file(result_path),
            "picks_sha256": independent.sha256_file(picks_path),
        }


def candidate_gate(
    metric: dict[str, Any],
    bootstrap: dict[str, Any],
    integrity_passes: bool,
) -> dict[str, Any]:
    positive_share = metric["largest_positive_code_pnl_share"]
    checks = {
        "net40_mean_positive": metric["net40_mean_pct"] > 0,
        "net60_mean_positive": metric["net60_mean_pct"] > 0,
        "familywise_95_lower_net40_mean_positive": (
            bootstrap["familywise_max_t_lower95_pct"] > 0
        ),
        "familywise_reality_check_p_at_most_0_10": (
            bootstrap["reality_check_adjusted_p_value"] <= 0.10
        ),
        "all_three_temporal_slices_net40_positive": all(
            value > 0 for value in metric["temporal_slices_net40_pct"].values()
        ),
        "positive_months_at_least_8": metric["positive_months_net40"] >= 8,
        "best_20_days_removed_net20_positive": (
            metric["best_days_removed_net20_pct"]["20"] is not None
            and metric["best_days_removed_net20_pct"]["20"] > 0
        ),
        "top_10_positive_pnl_codes_to_cash_net20_positive": (
            metric["top10_positive_pnl_codes_to_cash_net20_pct"] > 0
        ),
        "traded_days_at_least_80": metric["traded_days"] >= 80,
        "executed_slot_fraction_at_least_0_3": (
            metric["executed_slot_fraction"] >= 0.30
        ),
        "unique_codes_at_least_100": metric["unique_codes"] >= 100,
        "largest_code_weight_share_at_most_0_05": (
            metric["largest_code_weight_share"] <= 0.05
        ),
        "top10_code_weight_share_at_most_0_25": (
            metric["top10_code_weight_share"] <= 0.25
        ),
        "largest_positive_code_pnl_share_at_most_0_25": (
            positive_share is not None and positive_share <= 0.25
        ),
        "all_integrity_audits_pass": integrity_passes,
    }
    return {
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "passes_all": all(checks.values()),
    }


def report_markdown(
    protocol: dict[str, Any],
    result: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    lines = [
        "# v1.1 calendar and institutional-date zero-base experiment",
        "",
        "## Decision",
        "",
        f"- Retrospective decision: **{audit['decision']['retrospective_decision']}**",
        f"- Production ready: **{str(audit['decision']['production_ready']).lower()}**",
        f"- Scheduled sessions: **{result['coverage']['score_sessions']}**",
        f"- Strict TDnet score sessions: **{result['coverage']['strict_tdnet_score_sessions']}**",
        f"- Global reality-check p: **{audit['multiplicity']['global_reality_check_p_value']:.4f}**",
        "",
        "Every weekday, holiday, turn-of-month, quarter, fiscal, SQ, reporting-window, "
        "release-clock and density definition was frozen before execution. No calendar "
        "window or posterior strength was altered after reading returns.",
        "",
        "## Results",
        "",
        "| Policy | net20 | net40 | net60 | days | codes | max-t L95 | reality p | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for policy in sorted(result["metrics"]):
        metric = result["metrics"][policy]
        boot = audit["multiplicity"]["candidates"][policy]
        gate = audit["candidate_gates"][policy]["passes_all"]
        lines.append(
            f"| `{policy}` | {metric['net20_mean_pct']:+.4f}% | "
            f"{metric['net40_mean_pct']:+.4f}% | {metric['net60_mean_pct']:+.4f}% | "
            f"{metric['traded_days']} | {metric['unique_codes']} | "
            f"{boot['familywise_max_t_lower95_pct']:+.4f}% | "
            f"{boot['reality_check_adjusted_p_value']:.4f} | "
            f"{'PASS' if gate else 'FAIL'} |"
        )
    best = audit["decision"]["best_point_estimate_policy"]
    lines.extend(
        [
            "",
            "## Point-estimate leader",
            "",
            f"`{best}` is rejected because it failed: "
            + ", ".join(
                f"`{value}`"
                for value in audit["candidate_gates"][best]["failed_checks"]
            )
            + ".",
            "",
            "## Integrity",
            "",
            f"- Protocol/input/PIT/mutation: **{'PASS' if audit['integrity']['passes'] else 'FAIL'}**",
            f"- Independent outcome join and P&L: **{'PASS' if audit['pnl_reproduction']['passes'] else 'FAIL'}**",
            f"- Byte-identical rerun: **{'PASS' if audit['deterministic_reproduction']['passes'] else 'FAIL'}**",
            "",
            "## Research scope warning",
            "",
            "The cited Japanese calendar studies are old, mixed, and largely evaluate "
            "index or close-to-close returns. One pre-holiday re-examination found that "
            "most holiday effects did not persist outside narrower Golden Week/afternoon "
            "settings, and a later sample reported disappearance of monthly effects. "
            "Accordingly this experiment treated every mechanism as falsifiable and did "
            "not transfer published index effects directly to individual-stock O-C returns.",
            "",
        ]
    )
    lines.extend(
        f"- {url}" for url in protocol["prior_research_scope_warning"]["sources"]
    )
    lines.extend(
        [
            "",
            "No candidate passes every gate, so no forward-shadow finalist is frozen.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=ROOT / "research" / "model_v11_calendar_audit.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "research" / "model_v11_calendar_report.md",
    )
    parser.add_argument("--skip-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = independent.read_json(PROTOCOL)
    result = independent.read_json(RESULT)
    if independent.sha256_file(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("calendar protocol hash mismatch")
    if result["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("calendar result is not bound to protocol")
    if protocol["authority"]["production_promotion_allowed_from_this_run"]:
        raise RuntimeError("calendar retrospective run cannot promote")
    picks = pd.read_csv(PICKS, dtype={"code": "string", "policy_id": "string"})
    panel = joblib.load(protocol["frozen_inputs"]["panel"]["path"])
    joined, embedded_exact = independent.independent_outcome_join(picks, panel)
    sessions = pd.DatetimeIndex(
        sorted(pd.to_datetime(picks["date"]).dt.normalize().unique())
    )
    independent.SLICES = INDEPENDENT_SLICES
    recomputed = {
        str(policy): independent.independent_metrics(group.reset_index(drop=True), sessions)
        for policy, group in joined.groupby("policy_id", sort=True)
    }
    errors = independent.compare_values(result["metrics"], recomputed, "metrics")
    multiplicity = independent.multiplicity(recomputed)
    rerun = (
        {"passes": False, "skipped": True}
        if args.skip_rerun
        else deterministic_rerun()
    )
    integrity = {
        "protocol_hash_exact": (
            independent.sha256_file(PROTOCOL) == EXPECTED_PROTOCOL_SHA256
        ),
        "panel_hash_exact": (
            independent.sha256_file(protocol["frozen_inputs"]["panel"]["path"])
            == protocol["frozen_inputs"]["panel"]["sha256"]
        ),
        "manifest_hash_exact": (
            independent.sha256_file(
                protocol["frozen_inputs"]["panel"]["manifest_path"]
            )
            == protocol["frozen_inputs"]["panel"]["manifest_sha256"]
        ),
        "monthly_expanding_strict_prior": result["integrity"][
            "monthly_expanding_strict_prior"
        ],
        "future_source_violations_zero": (
            result["integrity"]["future_source_violations"] == 0
        ),
        "target_session_outcome_mutation_passes": result["integrity"][
            "target_session_outcome_mutation"
        ]["passes"],
    }
    integrity["passes"] = all(integrity.values())
    pnl = {
        "embedded_outcomes_match_independent_join": embedded_exact,
        "metric_mismatch_count": len(errors),
        "metric_mismatch_examples": errors[:20],
        "passes": embedded_exact and not errors,
    }
    all_integrity = integrity["passes"] and pnl["passes"] and rerun["passes"]
    gates = {
        policy: candidate_gate(
            metric,
            multiplicity["candidates"][policy],
            all_integrity,
        )
        for policy, metric in recomputed.items()
    }
    passing = [policy for policy, value in gates.items() if value["passes_all"]]
    best = max(recomputed, key=lambda name: recomputed[name]["net40_mean_pct"])
    finalist = (
        max(passing, key=lambda name: recomputed[name]["net40_mean_pct"])
        if passing
        else None
    )
    audit = {
        "schema_version": 1,
        "experiment_id": result["experiment_id"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "artifact_hashes_before_audit": {
            "result_sha256": independent.sha256_file(RESULT),
            "picks_sha256": independent.sha256_file(PICKS),
        },
        "integrity": integrity,
        "pnl_reproduction": pnl,
        "deterministic_reproduction": rerun,
        "multiplicity": multiplicity,
        "candidate_gates": gates,
        "decision": {
            "best_point_estimate_policy": best,
            "best_point_estimate_net40_pct": recomputed[best]["net40_mean_pct"],
            "passing_retrospective_candidates": passing,
            "forward_shadow_finalist": finalist,
            "retrospective_decision": (
                "FREEZE_FORWARD_SHADOW_FINALIST"
                if finalist
                else "REJECT_ALL_CALENDAR_HYPOTHESES"
            ),
            "production_ready": False,
            "production_blockers": [
                "Retrospective panel cannot authorize production promotion.",
                (
                    "No calendar policy passes all retrospective gates."
                    if finalist is None
                    else "A finalist still requires at least 60 prospective sessions."
                ),
            ],
        },
    }
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.report_output.write_text(
        report_markdown(protocol, result, audit), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "audit": str(args.audit_output),
                "report": str(args.report_output),
                "decision": audit["decision"]["retrospective_decision"],
                "best": best,
                "best_net40": recomputed[best]["net40_mean_pct"],
                "global_reality_p": multiplicity["global_reality_check_p_value"],
                "pnl_reproduction": pnl["passes"],
                "deterministic_rerun": rerun["passes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
