#!/usr/bin/env python3
"""Audit the immutable v0.4 post-failure diagnostic result.

This script never trains or scores a model.  It verifies the fixed artifacts,
recomputes portfolio metrics from the published picks, and records one
presentation-only correction: dates where both models had no candidate are not
ranking changes.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tse_session_ranker.io import json_dumps  # noqa: E402
from tse_session_ranker.profit import profit_metrics  # noqa: E402


LOCK_PATH = ROOT / "research/model_v04_post_failure_diagnostic_lock_v2.json"
LOCK_SHA256 = "868a496459ab2276dd2e6cd9a68e8e1e110883e7fed8219c687d317c123abd06"
RECEIPT_PATH = ROOT / "research/model_v04_post_failure_diagnostic_consumed.json"
RECEIPT_SHA256 = "01b62b92f87e1d4feb895c66712d209068f760b3b44e693fd3f88e9fcb978b4a"
RESULT_PATH = ROOT / "research/model_v04_post_failure_diagnostic_result.json"
RESULT_SHA256 = "36a1e7ec557d0549399392224d0acbd89f369012c294290724465605447854ed"
WINNER_PICKS_PATH = (
    ROOT / "research/model_v04_post_failure_diagnostic_winner_picks.csv"
)
WINNER_PICKS_SHA256 = (
    "5c06123f78cb5f9d09c26862983ed25498684b01195e29ce393d08db4b0b6a81"
)
CONTROL_PICKS_PATH = (
    ROOT / "research/model_v04_post_failure_diagnostic_v03_control_picks.csv"
)
CONTROL_PICKS_SHA256 = (
    "7a7fe3bd36596d84d0dce49572eb197a0d40b7511838c79b1993e2a0a0c97f39"
)
RUNNER_PATH = ROOT / "research/evaluate_model_v04_post_failure_diagnostic.py"
RUNNER_SHA256 = "38ce95b092f951194452385b41f9a22de0b36957d771204dd0e77ad1253a3613"
OUTPUT_PATH = (
    ROOT / "research/model_v04_post_failure_diagnostic_result_audit.json"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"artifact changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return value


def maximum_numeric_difference(left: Any, right: Any) -> float:
    """Return the maximum finite numeric delta in two identical JSON trees."""

    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError("metric dictionaries have different keys")
        return max(
            (maximum_numeric_difference(left[key], right[key]) for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("metric lists have different lengths")
        return max(
            (maximum_numeric_difference(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        if not isinstance(right, (int, float)) or isinstance(right, bool):
            raise ValueError("metric value types differ")
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            raise ValueError("metric values must be finite")
        return abs(float(left) - float(right))
    if left != right:
        raise ValueError(f"metric values differ: {left!r} != {right!r}")
    return 0.0


def corrected_ranking_changes(
    winner: pd.DataFrame, control: pd.DataFrame
) -> tuple[int, int, int]:
    left = winner.loc[winner["model_rank"].eq(1), ["date", "code"]].rename(
        columns={"code": "winner"}
    )
    right = control.loc[control["model_rank"].eq(1), ["date", "code"]].rename(
        columns={"code": "control"}
    )
    paired = left.merge(right, on="date", validate="one_to_one", sort=True)
    both_missing = paired["winner"].isna() & paired["control"].isna()
    same = paired["winner"].eq(paired["control"]) | both_missing
    winner_object = paired["winner"].astype(object).where(
        paired["winner"].notna(), float("nan")
    )
    control_object = paired["control"].astype(object).where(
        paired["control"].notna(), float("nan")
    )
    raw_nan_unequal_count = int(winner_object.ne(control_object).sum())
    return int((~same).sum()), int(both_missing.sum()), raw_nan_unequal_count


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"audit output already exists: {OUTPUT_PATH}")
    lock = read_json(LOCK_PATH, LOCK_SHA256)
    receipt = read_json(RECEIPT_PATH, RECEIPT_SHA256)
    result = read_json(RESULT_PATH, RESULT_SHA256)
    fixed_hashes = {
        "runner": (RUNNER_PATH, RUNNER_SHA256),
        "winner_picks": (WINNER_PICKS_PATH, WINNER_PICKS_SHA256),
        "control_picks": (CONTROL_PICKS_PATH, CONTROL_PICKS_SHA256),
    }
    checks: dict[str, bool] = {
        f"{name}_sha256": sha256_file(path) == expected
        for name, (path, expected) in fixed_hashes.items()
    }
    checks.update(
        {
            "receipt_binds_lock": receipt.get("diagnostic_lock_sha256")
            == LOCK_SHA256,
            "result_binds_lock": result.get("diagnostic_lock_sha256")
            == LOCK_SHA256,
            "result_binds_receipt": result.get("diagnostic_receipt_sha256")
            == RECEIPT_SHA256,
            "result_binds_winner_picks": result.get("outputs", {}).get(
                "winner_picks_sha256"
            )
            == WINNER_PICKS_SHA256,
            "result_binds_control_picks": result.get("outputs", {}).get(
                "v03_control_picks_sha256"
            )
            == CONTROL_PICKS_SHA256,
            "lock_binds_runner": lock.get("diagnostic_runner_sha256")
            == RUNNER_SHA256,
            "formal_metric_run_zero": result.get("formal_metric_run_count") == 0,
            "diagnostic_metric_run_once": result.get(
                "diagnostic_metric_run_count"
            )
            == 1,
            "winner_evaluated_once": result.get(
                "diagnostic_model_evaluation_counts", {}
            ).get("winner")
            == 1,
            "control_evaluated_once": result.get(
                "diagnostic_model_evaluation_counts", {}
            ).get("v03_control")
            == 1,
            "no_diagnostic_selection": result.get(
                "selection_performed_in_diagnostic"
            )
            is False,
            "no_holdout_retuning": result.get("holdout_retuning_performed")
            is False,
            "formal_validation_unavailable": result.get(
                "formal_validation_available"
            )
            is False,
            "nonformal_diagnostic_recorded": result.get(
                "holdout_nonformal_diagnostic_evaluated"
            )
            is True,
            "status_is_no_edge": result.get("status")
            == "diagnostic_no_demonstrated_edge",
        }
    )

    winner = pd.read_csv(WINNER_PICKS_PATH, dtype={"code": "string"})
    control = pd.read_csv(CONTROL_PICKS_PATH, dtype={"code": "string"})
    winner["date"] = pd.to_datetime(winner["date"], errors="raise")
    control["date"] = pd.to_datetime(control["date"], errors="raise")
    recomputed: dict[str, Any] = {}
    maximum_difference = 0.0
    for label, picks, result_key in (
        ("winner", winner, "winner_result"),
        ("v03_control", control, "v03_control_result"),
    ):
        recomputed[label] = {}
        for top_k in (1, 2):
            observed = profit_metrics(picks, top_k=top_k, cost_bps=20.0)
            expected = result[result_key][f"top{top_k}_20bp"]
            difference = maximum_numeric_difference(observed, expected)
            maximum_difference = max(maximum_difference, difference)
            recomputed[label][f"top{top_k}_20bp"] = observed
            checks[f"{label}_top{top_k}_metrics_match"] = difference <= 1e-12
        stress = profit_metrics(picks, top_k=1, cost_bps=40.0)[
            "net_mean_pct_at_cost"
        ]
        expected_stress = result[result_key]["top1_40bp_mean_pct"]
        stress_difference = abs(float(stress) - float(expected_stress))
        maximum_difference = max(maximum_difference, stress_difference)
        checks[f"{label}_top1_40bp_matches"] = stress_difference <= 1e-12
        recomputed[label]["top1_40bp_mean_pct"] = stress

    corrected, both_missing, raw_count = corrected_ranking_changes(
        winner, control
    )
    recorded = int(
        result["diagnostic_gate"]["paired_vs_v03"]["ranking_changed_days"]
    )
    checks["recorded_ranking_change_issue_reproduced"] = (
        recorded == raw_count == 149
    )
    checks["corrected_ranking_changes_are_143"] = (
        corrected == 143 and both_missing == 6
    )
    checks["ranking_change_does_not_enter_diagnostic_gate"] = (
        "ranking_changed_days" not in result["diagnostic_gate"]["checks"]
    )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise AssertionError("diagnostic result audit failed: " + ", ".join(failed))

    payload = {
        "schema_version": 1,
        "phase": "post_failure_nonformal_diagnostic_result_audit",
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "model_evaluation_performed": False,
        "source_artifacts": {
            "lock_sha256": LOCK_SHA256,
            "receipt_sha256": RECEIPT_SHA256,
            "result_sha256": RESULT_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "winner_picks_sha256": WINNER_PICKS_SHA256,
            "v03_control_picks_sha256": CONTROL_PICKS_SHA256,
        },
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": int(len(checks)),
        "maximum_metric_absolute_difference": maximum_difference,
        "independently_recomputed_metrics": recomputed,
        "presentation_correction": {
            "field": "diagnostic_gate.paired_vs_v03.ranking_changed_days",
            "recorded_value": recorded,
            "corrected_value": corrected,
            "both_models_missing_candidate_days": both_missing,
            "cause": "pandas missing code values compare unequal to each other",
            "impact": (
                "presentation only; not used by diagnostic gates, returns, "
                "bootstrap intervals, or status"
            ),
        },
        "audited_conclusion": "diagnostic_no_demonstrated_edge",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("x", encoding="utf-8") as stream:
        stream.write(json_dumps(payload) + "\n")
    print(f"audit={OUTPUT_PATH}")
    print(f"checks={payload['checks_passed']}/{payload['checks_total']}")
    print(f"corrected_ranking_changed_days={corrected}")


if __name__ == "__main__":
    main()
