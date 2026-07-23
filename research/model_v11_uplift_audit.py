#!/usr/bin/env python3
"""Independent P&L, multiplicity, concentration, and rerun audit for v1.1 uplift."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research" / "model_v11_uplift_protocol.json"
RESULT = ROOT / "research" / "model_v11_uplift_result.json"
PICKS = ROOT / "research" / "model_v11_uplift_picks.csv"
EXPECTED_PROTOCOL_SHA256 = (
    "611f9d16467fee02d5a17e415d0a0a6490917310509174ed43379c60d04c69cb"
)
SEED = 20260723
BOOTSTRAPS = 5000
BLOCK_LENGTH = 10
COSTS = (20, 40, 60)
SLICES = {
    "early": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-31")),
    "middle": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-05-31")),
    "late": (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-31")),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def independent_outcome_join(
    picks: pd.DataFrame, panel: pd.DataFrame
) -> tuple[pd.DataFrame, bool]:
    panel = panel[["date", "code", "oc_return_pct", "label"]].copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["code"] = panel["code"].astype(str).str.zfill(4)
    panel = panel.rename(
        columns={
            "oc_return_pct": "joined_oc_return_pct",
            "label": "joined_label",
        }
    )
    if panel.duplicated(["date", "code"]).any():
        raise RuntimeError("audit outcome panel contains duplicate keys")
    result = picks.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["code"] = result["code"].astype("string").str.zfill(4)
    result = result.merge(
        panel,
        on=["date", "code"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    executed = result["code"].notna()
    embedded_y = pd.to_numeric(result["oc_return_pct"], errors="coerce")
    embedded_label = pd.to_numeric(result["label"], errors="coerce")
    joined_y = result["joined_oc_return_pct"]
    joined_label = result["joined_label"]
    missingness_exact = bool(
        np.array_equal(
            embedded_y.loc[executed].isna().to_numpy(),
            joined_y.loc[executed].isna().to_numpy(),
        )
        and np.array_equal(
            embedded_label.loc[executed].isna().to_numpy(),
            joined_label.loc[executed].isna().to_numpy(),
        )
    )
    observed = executed & joined_y.notna()
    embedded_exact = bool(
        missingness_exact
        and
        np.allclose(
            embedded_y.loc[observed],
            joined_y.loc[observed],
            atol=1e-12,
            rtol=0,
        )
        and np.allclose(
            embedded_label.loc[observed],
            joined_label.loc[observed],
            atol=1e-12,
            rtol=0,
        )
    )
    result["oc_return_pct"] = result["joined_oc_return_pct"]
    result["label"] = result["joined_label"]
    return result.drop(
        columns=["joined_oc_return_pct", "joined_label"]
    ), embedded_exact


def daily_return(
    frame: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    cost_bps: int,
) -> pd.DataFrame:
    capacity_values = frame["capacity"].drop_duplicates()
    if len(capacity_values) != 1:
        raise RuntimeError("policy has more than one capacity")
    capacity = int(capacity_values.iloc[0])
    executed = frame["code"].notna().astype(float)
    gross = (
        pd.to_numeric(frame["oc_return_pct"], errors="coerce").fillna(0.0)
        / capacity
    )
    contribution = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"]),
            "gross": gross,
            "net": gross - executed * (cost_bps / 100.0) / capacity,
            "exposure": executed / capacity,
        }
    )
    return contribution.groupby("date", sort=True).sum().reindex(
        sessions, fill_value=0.0
    )


def finite_mean(values: pd.Series | np.ndarray) -> float | None:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    return None if not len(clean) else float(clean.mean())


def independent_metrics(
    frame: pd.DataFrame, sessions: pd.DatetimeIndex
) -> dict[str, Any]:
    capacity = int(frame["capacity"].iloc[0])
    daily = {cost: daily_return(frame, sessions, cost) for cost in COSTS}
    gross = daily[40]["gross"]
    net20 = daily[20]["net"]
    net40 = daily[40]["net"]
    net60 = daily[60]["net"]
    exposure = daily[40]["exposure"]
    executed = frame.loc[frame["code"].notna()].copy()
    monthly = net40.groupby(net40.index.to_period("M")).mean()
    slices = {
        name: float(net40.loc[(net40.index >= start) & (net40.index <= end)].mean())
        for name, (start, end) in SLICES.items()
    }
    best_removed: dict[str, float | None] = {}
    worst_removed: dict[str, float | None] = {}
    for count in (5, 10, 20):
        keep = max(len(net20) - count, 0)
        best_removed[str(count)] = finite_mean(net20.nsmallest(keep))
        worst_removed[str(count)] = finite_mean(net20.nlargest(keep))

    if len(executed):
        code_weight = executed.groupby("code", sort=False).size() / capacity
        shares = code_weight / code_weight.sum()
        largest_code = float(shares.max())
        top10_code = float(shares.nlargest(10).sum())
        slot_net20 = (
            pd.to_numeric(executed["oc_return_pct"], errors="raise") - 0.20
        ) / capacity
        by_code = slot_net20.groupby(executed["code"]).sum()
        positive = by_code[by_code > 0].sort_values(ascending=False)
        positive_total = float(positive.sum())
        largest_positive = (
            None
            if positive_total <= 0
            else float(positive.iloc[0] / positive_total)
        )
        removed_codes = list(positive.head(10).index)
        kept = executed.loc[~executed["code"].isin(removed_codes)]
        reduced = (
            (
                pd.to_numeric(kept["oc_return_pct"], errors="raise") - 0.20
            )
            .div(capacity)
            .groupby(pd.to_datetime(kept["date"]))
            .sum()
            .reindex(sessions, fill_value=0.0)
        )
        top10_cash = float(reduced.mean())
    else:
        largest_code = 0.0
        top10_code = 0.0
        largest_positive = None
        top10_cash = 0.0
    tail_n = max(1, int(math.ceil(len(net40) * 0.05)))
    return {
        "scheduled_days": int(len(sessions)),
        "gross_mean_pct": float(gross.mean()),
        "net20_mean_pct": float(net20.mean()),
        "net40_mean_pct": float(net40.mean()),
        "net60_mean_pct": float(net60.mean()),
        "daily_gross_win_rate": float((gross > 0).mean()),
        "executed_slots": int(len(executed)),
        "executed_slot_fraction": float(
            len(executed) / (len(sessions) * capacity)
        ),
        "traded_days": int((exposure > 0).sum()),
        "cash_days": int((exposure == 0).sum()),
        "unique_codes": int(executed["code"].nunique()),
        "monthly_net40_pct": {
            str(period): float(value) for period, value in monthly.items()
        },
        "positive_months_net40": int((monthly > 0).sum()),
        "temporal_slices_net40_pct": slices,
        "best_days_removed_net20_pct": best_removed,
        "worst_days_removed_net20_pct": worst_removed,
        "expected_shortfall05_net40_pct": float(
            net40.nsmallest(tail_n).mean()
        ),
        "largest_code_weight_share": largest_code,
        "top10_code_weight_share": top10_code,
        "largest_positive_code_pnl_share": largest_positive,
        "top10_positive_pnl_codes_to_cash_net20_pct": top10_cash,
        "daily_net40_pct": [float(value) for value in net40],
    }


def compare_values(
    expected: Any, actual: Any, path: str = ""
) -> list[str]:
    errors: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return [f"{path}: keys differ"]
        for key in expected:
            errors.extend(
                compare_values(expected[key], actual[key], f"{path}.{key}")
            )
        return errors
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: lengths differ"]
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            errors.extend(compare_values(left, right, f"{path}[{index}]"))
        return errors
    if expected is None or actual is None:
        return [] if expected is actual else [f"{path}: {expected!r} != {actual!r}"]
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return (
            []
            if np.isclose(float(expected), float(actual), atol=1e-12, rtol=0)
            else [f"{path}: {expected!r} != {actual!r}"]
        )
    return [] if expected == actual else [f"{path}: {expected!r} != {actual!r}"]


def circular_block_indices(
    rng: np.random.Generator, sessions: int
) -> np.ndarray:
    blocks = int(math.ceil(sessions / BLOCK_LENGTH))
    starts = rng.integers(0, sessions, size=blocks)
    offsets = np.arange(BLOCK_LENGTH)
    return ((starts[:, None] + offsets[None, :]) % sessions).ravel()[:sessions]


def multiplicity(
    metrics: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    policy_ids = sorted(metrics)
    values = np.column_stack(
        [np.asarray(metrics[name]["daily_net40_pct"], dtype=float) for name in policy_ids]
    )
    n = len(values)
    means = values.mean(axis=0)
    ses = values.std(axis=0, ddof=1) / math.sqrt(n)
    valid = ses > 0
    observed_t = np.zeros(len(policy_ids))
    observed_t[valid] = means[valid] / ses[valid]
    centered = values - means
    rng = np.random.default_rng(SEED)
    max_t = np.empty(BOOTSTRAPS)
    max_mean = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        index = circular_block_indices(rng, n)
        boot_mean = centered[index].mean(axis=0)
        boot_t = np.full(len(policy_ids), -np.inf)
        boot_t[valid] = boot_mean[valid] / ses[valid]
        max_t[draw] = float(np.max(boot_t))
        max_mean[draw] = float(np.max(boot_mean))
    critical = float(np.quantile(max_t, 0.95, method="higher"))
    best_observed = float(means.max())
    global_reality_p = float(
        (1 + np.count_nonzero(max_mean >= best_observed)) / (BOOTSTRAPS + 1)
    )
    candidates: dict[str, Any] = {}
    for position, name in enumerate(policy_ids):
        if valid[position]:
            max_t_p = float(
                (1 + np.count_nonzero(max_t >= observed_t[position]))
                / (BOOTSTRAPS + 1)
            )
            lower = float(means[position] - critical * ses[position])
        else:
            max_t_p = 1.0
            lower = float(means[position])
        reality_p = float(
            (1 + np.count_nonzero(max_mean >= means[position]))
            / (BOOTSTRAPS + 1)
        )
        candidates[name] = {
            "mean_net40_pct": float(means[position]),
            "standard_error_pct": float(ses[position]),
            "studentised_statistic": float(observed_t[position]),
            "familywise_max_t_lower95_pct": lower,
            "familywise_max_t_adjusted_p_value": max_t_p,
            "reality_check_adjusted_p_value": reality_p,
        }
    return {
        "resamples": BOOTSTRAPS,
        "block_length_sessions": BLOCK_LENGTH,
        "family_size": len(policy_ids),
        "critical_max_t_95": critical,
        "global_reality_check_p_value": global_reality_p,
        "candidates": candidates,
    }


def deterministic_rerun() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v11_uplift_audit_") as directory:
        target = Path(directory)
        result_path = target / "result.json"
        picks_path = target / "picks.csv"
        completed = subprocess.run(
            [
                "python",
                str(ROOT / "research" / "model_v11_uplift_runner.py"),
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
        if completed.returncode != 0:
            return {
                "passes": False,
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-2000:],
            }
        result_equal = result_path.read_bytes() == RESULT.read_bytes()
        picks_equal = picks_path.read_bytes() == PICKS.read_bytes()
        return {
            "passes": result_equal and picks_equal,
            "returncode": completed.returncode,
            "result_byte_identical": result_equal,
            "picks_byte_identical": picks_equal,
            "result_sha256": sha256_file(result_path),
            "picks_sha256": sha256_file(picks_path),
        }


def gates_for(
    metrics: dict[str, Any],
    bootstrap: dict[str, Any],
    integrity_passes: bool,
) -> dict[str, Any]:
    positive_share = metrics["largest_positive_code_pnl_share"]
    checks = {
        "net40_mean_positive": metrics["net40_mean_pct"] > 0,
        "net60_mean_positive": metrics["net60_mean_pct"] > 0,
        "familywise_95_lower_net40_mean_positive": (
            bootstrap["familywise_max_t_lower95_pct"] > 0
        ),
        "familywise_reality_check_p_at_most_0_10": (
            bootstrap["reality_check_adjusted_p_value"] <= 0.10
        ),
        "all_three_temporal_slices_net40_positive": all(
            value > 0 for value in metrics["temporal_slices_net40_pct"].values()
        ),
        "positive_months_at_least_4": metrics["positive_months_net40"] >= 4,
        "best_20_days_removed_net20_positive": (
            metrics["best_days_removed_net20_pct"]["20"] is not None
            and metrics["best_days_removed_net20_pct"]["20"] > 0
        ),
        "top_10_positive_pnl_codes_to_cash_net20_positive": (
            metrics["top10_positive_pnl_codes_to_cash_net20_pct"] > 0
        ),
        "traded_days_at_least_50": metrics["traded_days"] >= 50,
        "executed_slot_fraction_at_least_0_3": (
            metrics["executed_slot_fraction"] >= 0.30
        ),
        "unique_codes_at_least_40": metrics["unique_codes"] >= 40,
        "largest_code_weight_share_at_most_0_08": (
            metrics["largest_code_weight_share"] <= 0.08
        ),
        "top10_code_weight_share_at_most_0_35": (
            metrics["top10_code_weight_share"] <= 0.35
        ),
        "largest_positive_code_pnl_share_at_most_0_25": (
            positive_share is not None and positive_share <= 0.25
        ),
        "all_integrity_audits_pass": integrity_passes,
    }
    return {
        "checks": checks,
        "failed_checks": [name for name, passes in checks.items() if not passes],
        "passes_all": all(checks.values()),
    }


def report_markdown(
    result: dict[str, Any], audit: dict[str, Any]
) -> str:
    lines = [
        "# v1.1 causal/uplift zero-base experiment",
        "",
        "## Decision",
        "",
        f"- Retrospective decision: **{audit['decision']['retrospective_decision']}**",
        f"- Production ready: **{str(audit['decision']['production_ready']).lower()}**",
        f"- Strict scheduled sessions: **{result['coverage']['strict_score_sessions']}**",
        f"- Global familywise reality-check p: **{audit['multiplicity']['global_reality_check_p_value']:.4f}**",
        "",
        "The eight mechanisms estimate an event-versus-no-event counterfactual. "
        "They are structurally separate from v10 title-value regression and v11 "
        "historical-title analog retrieval. The panel is retrospective, so even a "
        "passing mechanism could only become a forward-shadow candidate.",
        "",
        "## Locked hypotheses",
        "",
        "| Policy | net20 | net40 | net60 | days | slots | max-t L95 | reality p | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for policy_id in sorted(result["metrics"]):
        metric = result["metrics"][policy_id]
        boot = audit["multiplicity"]["candidates"][policy_id]
        gate = audit["candidate_gates"][policy_id]["passes_all"]
        lines.append(
            f"| `{policy_id}` | {metric['net20_mean_pct']:+.4f}% | "
            f"{metric['net40_mean_pct']:+.4f}% | {metric['net60_mean_pct']:+.4f}% | "
            f"{metric['traded_days']} | {metric['executed_slots']} | "
            f"{boot['familywise_max_t_lower95_pct']:+.4f}% | "
            f"{boot['reality_check_adjusted_p_value']:.4f} | "
            f"{'PASS' if gate else 'FAIL'} |"
        )
    best = audit["decision"]["best_point_estimate_policy"]
    lines.extend(
        [
            "",
            "## Why the point-estimate leader is rejected",
            "",
            f"`{best}` had the highest net40 point estimate, but failed: "
            + ", ".join(f"`{value}`" for value in audit["candidate_gates"][best]["failed_checks"])
            + ".",
            "",
            "## Integrity",
            "",
            f"- Input/protocol/PIT/mutation: **{'PASS' if audit['integrity']['passes'] else 'FAIL'}**",
            f"- Independent P&L reconstruction: **{'PASS' if audit['pnl_reproduction']['passes'] else 'FAIL'}**",
            f"- Byte-identical runner rerun: **{'PASS' if audit['deterministic_reproduction']['passes'] else 'FAIL'}**",
            "",
            "No threshold was retuned after observing results. No candidate is frozen "
            "because no policy passes the preregistered retrospective gate.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=ROOT / "research" / "model_v11_uplift_audit.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "research" / "model_v11_uplift_report.md",
    )
    parser.add_argument("--skip-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = read_json(PROTOCOL)
    result = read_json(RESULT)
    if sha256_file(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("audit protocol hash mismatch")
    if result["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("result is not bound to uplift protocol")
    if protocol["authority"]["production_promotion_allowed_from_this_run"]:
        raise RuntimeError("protocol unexpectedly allows production promotion")

    picks = pd.read_csv(
        PICKS,
        dtype={"code": "string", "policy_id": "string", "hypothesis_id": "string"},
    )
    panel = joblib.load(protocol["frozen_inputs"]["panel"]["path"])
    audited_picks, embedded_exact = independent_outcome_join(picks, panel)
    sessions = pd.DatetimeIndex(
        sorted(pd.to_datetime(picks["date"]).dt.normalize().unique())
    )
    recomputed = {
        str(policy_id): independent_metrics(group.reset_index(drop=True), sessions)
        for policy_id, group in audited_picks.groupby("policy_id", sort=True)
    }
    metric_errors = compare_values(result["metrics"], recomputed, "metrics")
    bootstrap = multiplicity(recomputed)
    rerun = (
        {"passes": False, "skipped": True}
        if args.skip_rerun
        else deterministic_rerun()
    )
    integrity = {
        "protocol_hash_exact": sha256_file(PROTOCOL) == EXPECTED_PROTOCOL_SHA256,
        "panel_hash_exact": (
            sha256_file(protocol["frozen_inputs"]["panel"]["path"])
            == protocol["frozen_inputs"]["panel"]["sha256"]
        ),
        "manifest_hash_exact": (
            sha256_file(protocol["frozen_inputs"]["panel"]["manifest_path"])
            == protocol["frozen_inputs"]["panel"]["manifest_sha256"]
        ),
        "target_session_outcome_mutation_passes": result["integrity"][
            "target_session_outcome_mutation"
        ]["passes"],
        "monthly_expanding_strict_prior": result["integrity"][
            "monthly_expanding_strict_prior"
        ],
        "future_source_violations_zero": (
            result["integrity"]["future_source_violations"] == 0
        ),
    }
    integrity["passes"] = all(integrity.values())
    pnl_reproduction = {
        "embedded_outcomes_match_independent_join": embedded_exact,
        "metric_mismatch_count": len(metric_errors),
        "metric_mismatch_examples": metric_errors[:20],
        "passes": embedded_exact and not metric_errors,
    }
    all_integrity = (
        integrity["passes"]
        and pnl_reproduction["passes"]
        and rerun["passes"]
    )
    candidate_gates = {
        policy_id: gates_for(
            recomputed[policy_id],
            bootstrap["candidates"][policy_id],
            all_integrity,
        )
        for policy_id in sorted(recomputed)
    }
    passing = [
        policy_id
        for policy_id, value in candidate_gates.items()
        if value["passes_all"]
    ]
    best = max(
        recomputed,
        key=lambda name: recomputed[name]["net40_mean_pct"],
    )
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
            "result_sha256": sha256_file(RESULT),
            "picks_sha256": sha256_file(PICKS),
        },
        "integrity": integrity,
        "pnl_reproduction": pnl_reproduction,
        "deterministic_reproduction": rerun,
        "multiplicity": bootstrap,
        "candidate_gates": candidate_gates,
        "decision": {
            "best_point_estimate_policy": best,
            "best_point_estimate_net40_pct": recomputed[best]["net40_mean_pct"],
            "passing_retrospective_candidates": passing,
            "forward_shadow_finalist": finalist,
            "retrospective_decision": (
                "FREEZE_FORWARD_SHADOW_FINALIST"
                if finalist is not None
                else "REJECT_ALL_UPLIFT_HYPOTHESES"
            ),
            "production_ready": False,
            "production_blockers": [
                "Retrospective panel cannot authorize production promotion.",
                (
                    "No candidate passes all retrospective gates."
                    if finalist is None
                    else "A frozen finalist still requires at least 60 prospective sessions."
                ),
            ],
        },
    }
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.report_output.write_text(
        report_markdown(result, audit),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audit": str(args.audit_output),
                "report": str(args.report_output),
                "decision": audit["decision"]["retrospective_decision"],
                "best": best,
                "best_net40": recomputed[best]["net40_mean_pct"],
                "global_reality_p": bootstrap["global_reality_check_p_value"],
                "pnl_reproduction": pnl_reproduction["passes"],
                "deterministic_rerun": rerun["passes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
