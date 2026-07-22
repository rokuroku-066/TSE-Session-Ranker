#!/usr/bin/env python3
"""Independently recompute v0.6 feature-screen decisions from pick artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tse_session_ranker.io import write_json  # noqa: E402
from tse_session_ranker.profit import (  # noqa: E402
    daily_portfolio_returns,
    profit_metrics,
)
from tse_session_ranker.validation import moving_block_bootstrap  # noqa: E402


PRIMARY_COST_BPS = 20.0
STRESS_COST_BPS = 40.0
BLOCK_LENGTH = 5
BOOTSTRAP_SAMPLES = 5_000
MAX_T_CONFIDENCE = 0.80
SEED = 31
TOP_K = 2
TOLERANCE = 1e-11


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pick(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def _recomputed_metrics(picks: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    displayed = picks["code"].notna()
    output["display"] = {
        "rank1_rate": float(displayed[picks["model_rank"].eq(1)].mean()),
        "rank2_rate": float(displayed[picks["model_rank"].eq(2)].mean()),
        "scheduled_sessions": int(picks["date"].nunique()),
    }
    for top_k in (1, 2):
        daily = daily_portfolio_returns(
            picks, top_k=top_k, cost_bps=PRIMARY_COST_BPS
        ).set_index("date")["net_return_pct"]
        interval = moving_block_bootstrap(
            daily,
            block_length=BLOCK_LENGTH,
            samples=BOOTSTRAP_SAMPLES,
            confidence=0.90,
            random_state=SEED + top_k,
        )
        output[f"top{top_k}"] = {
            "net20": profit_metrics(
                picks, top_k=top_k, cost_bps=PRIMARY_COST_BPS
            ),
            "net40": profit_metrics(
                picks, top_k=top_k, cost_bps=STRESS_COST_BPS
            ),
            "block5_bootstrap": asdict(interval),
        }
    rank2 = picks[picks["model_rank"].eq(2)].copy()
    rank2["model_rank"] = 1
    output["rank2_standalone_net20"] = profit_metrics(
        rank2, top_k=1, cost_bps=PRIMARY_COST_BPS
    )
    return output


def _numeric_differences(
    expected: Any,
    actual: Any,
    *,
    path: str,
    output: list[dict[str, Any]],
) -> None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) & set(actual)):
            _numeric_differences(
                expected[key], actual[key], path=f"{path}.{key}", output=output
            )
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, bool) or isinstance(actual, bool):
            return
        left, right = float(expected), float(actual)
        if np.isfinite(left) and np.isfinite(right):
            output.append(
                {
                    "path": path,
                    "stored": left,
                    "recomputed": right,
                    "absolute_difference": abs(left - right),
                }
            )


def _daily_net(picks: pd.DataFrame) -> pd.Series:
    return daily_portfolio_returns(
        picks, top_k=TOP_K, cost_bps=PRIMARY_COST_BPS
    ).set_index("date")["net_return_pct"].sort_index()


def _block_indices(n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    blocks = math.ceil(n / BLOCK_LENGTH)
    starts = rng.integers(0, n, size=(BOOTSTRAP_SAMPLES, blocks))
    offsets = np.arange(BLOCK_LENGTH)
    return ((starts[..., None] + offsets) % n).reshape(
        BOOTSTRAP_SAMPLES, -1
    )[:, :n]


def _max_t(
    baseline: pd.DataFrame, candidates: Mapping[str, pd.DataFrame]
) -> dict[str, dict[str, float]]:
    base = _daily_net(baseline)
    names = list(candidates)
    columns: list[np.ndarray] = []
    for name in names:
        candidate = _daily_net(candidates[name])
        if not candidate.index.equals(base.index):
            raise AssertionError("audit candidates do not share scheduled dates")
        columns.append((candidate - base).to_numpy(dtype=float))
    matrix = np.column_stack(columns)
    means = matrix.mean(axis=0)
    centered = matrix - means
    bootstrap = centered[_block_indices(len(matrix))].mean(axis=1)
    critical = float(np.quantile(bootstrap.max(axis=1), MAX_T_CONFIDENCE))
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result", default="research/model_v06_feature_result.json"
    )
    parser.add_argument(
        "--output", default="research/model_v06_feature_result_audit.json"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result_path = Path(args.result).resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest_path = result_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check(
        "manifest_binds_result",
        manifest["result_sha256"] == sha256_file(result_path),
    )
    protocol_path = ROOT / "research/model_v06_feature_protocol.json"
    catalog_path = ROOT / "research/model_v06_feature_catalog.json"
    check(
        "protocol_hash",
        manifest["protocol_sha256"] == sha256_file(protocol_path),
    )
    check(
        "catalog_hash",
        manifest["catalog_sha256"] == sha256_file(catalog_path),
    )
    input_lock_path = ROOT / result["input_lock"]["path"]
    check(
        "input_lock_hash",
        result["input_lock"]["sha256"] == sha256_file(input_lock_path),
    )
    for name, artifact in result["artifacts"].items():
        path = ROOT / artifact["path"]
        check(
            f"artifact_hash:{name}",
            artifact["sha256"] == sha256_file(path),
        )

    screen_path = ROOT / result["artifacts"]["group_screen_display_picks"]["path"]
    union_path = ROOT / result["artifacts"]["union_prune_display_picks"]["path"]
    stability_path = ROOT / result["artifacts"]["stability_display_picks"]["path"]
    screen = _read_pick(screen_path)
    union = _read_pick(union_path)
    stability = _read_pick(stability_path)

    stored_results: list[tuple[dict[str, Any], pd.DataFrame]] = [
        (result["group_screen"]["base"], screen)
    ]
    stored_results.extend(
        (value, screen) for value in result["group_screen"]["candidates"].values()
    )
    stored_results.append(
        (result["union_prune"]["initial_survivor_union"], union)
    )
    stored_results.extend(
        (value["result"], union)
        for value in result["union_prune"]["leave_one_group_out"].values()
    )
    if result["union_prune"]["interaction_diagnostic"] is not None:
        stored_results.append(
            (result["union_prune"]["interaction_diagnostic"]["result"], union)
        )
    stored_results.extend(
        [
            (result["stability_report"]["locked_recipe"], stability),
            (result["stability_report"]["price_core_control"], stability),
        ]
    )
    numeric_differences: list[dict[str, Any]] = []
    recipes_checked = 0
    for stored, stage_frame in stored_results:
        recipe_id = stored["recipe_id"]
        matching = stage_frame[stage_frame["recipe_id"].eq(recipe_id)].copy()
        if matching.empty:
            raise AssertionError(
                f"audit found no pick artifact for recipe {recipe_id}"
            )
        recomputed = _recomputed_metrics(matching)
        _numeric_differences(
            stored["metrics"],
            recomputed,
            path=f"metrics.{recipe_id}",
            output=numeric_differences,
        )
        recipes_checked += 1
    max_difference = max(
        (item["absolute_difference"] for item in numeric_differences), default=0.0
    )
    check(
        "all_recipe_metrics_recomputed",
        max_difference <= TOLERANCE,
        {"recipes": recipes_checked, "max_absolute_difference": max_difference},
    )

    base_id = result["group_screen"]["base"]["recipe_id"]
    base_picks = screen[screen["recipe_id"].eq(base_id)].copy()
    candidate_ids = {
        name: stored["recipe_id"]
        for name, stored in result["group_screen"]["candidates"].items()
    }
    candidate_picks = {
        name: screen[screen["recipe_id"].eq(recipe_id)].copy()
        for name, recipe_id in candidate_ids.items()
    }
    adjusted = _max_t(base_picks, candidate_picks)
    max_t_difference = 0.0
    for name, values in adjusted.items():
        stored = result["group_screen"]["qualification"][name]
        for field, value in values.items():
            max_t_difference = max(max_t_difference, abs(value - stored[field]))
    check(
        "max_t_uplifts_recomputed",
        max_t_difference <= TOLERANCE,
        {"max_absolute_difference": max_t_difference},
    )

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    threshold = protocol["screen"]["all_required"]
    base_top5 = result["group_screen"]["base"]["metrics"]["top2"]["net20"][
        "top5_removed_net_mean_pct"
    ]
    recomputed_qualified: list[str] = []
    qualification_match = True
    for name, stored_result in result["group_screen"]["candidates"].items():
        stored_q = result["group_screen"]["qualification"][name]
        metrics = stored_result["metrics"]["top2"]["net20"]
        positive_month_fraction = metrics["positive_months"] / metrics["months"]
        sparse = name.startswith("T")
        checks_recomputed = {
            "availability": stored_q["availability_rate"]
            >= threshold["feature_availability_rate"],
            "positive_uplift": adjusted[name]["mean_uplift_pct"] > 0,
            "positive_month_fraction": positive_month_fraction
            >= threshold["positive_month_fraction"],
            "top5_removed_uplift": (
                metrics["top5_removed_net_mean_pct"] - base_top5
            )
            >= 0,
            "max_t_adjusted_lower_bound": adjusted[name][
                "adjusted_one_sided_80pct_lower_pct"
            ]
            >= 0,
            "event_days": (not sparse)
            or stored_q["event_days"]
            >= threshold["minimum_independent_event_days_for_sparse_event_group"],
        }
        passed = all(checks_recomputed.values())
        qualification_match &= (
            checks_recomputed == stored_q["checks"] and passed == stored_q["passed"]
        )
        if passed:
            recomputed_qualified.append(name)
    recomputed_qualified.sort(
        key=lambda name: (
            -result["group_screen"]["qualification"][name]["mean_uplift_pct"],
            -result["group_screen"]["qualification"][name][
                "adjusted_one_sided_80pct_lower_pct"
            ],
            -result["group_screen"]["qualification"][name][
                "top5_removed_uplift_pct"
            ],
            result["group_screen"]["candidates"][name]["feature_count"]
            - result["group_screen"]["base"]["feature_count"],
            name,
        )
    )
    recomputed_survivors = recomputed_qualified[
        : protocol["screen"]["maximum_survivor_groups"]
    ]
    check("qualification_predicates", qualification_match)
    check(
        "formal_survivor_selection",
        recomputed_survivors == result["group_screen"]["formal_survivors"],
        {"recomputed": recomputed_survivors},
    )

    union_mean = result["union_prune"]["initial_survivor_union"]["metrics"][
        "top2"
    ]["net20"]["net_mean_pct_at_cost"]
    retained: list[str] = []
    prune_match = True
    for name, raw in result["union_prune"]["leave_one_group_out"].items():
        without = raw["result"]["metrics"]["top2"]["net20"][
            "net_mean_pct_at_cost"
        ]
        keep = union_mean > without
        prune_match &= keep == raw["retained"]
        if keep:
            retained.append(name)
    check("union_leave_one_out", prune_match)
    check(
        "retained_groups",
        retained == result["union_prune"]["retained_groups_after_one_pass"],
        {"recomputed": retained},
    )
    stability_delta = (
        result["stability_report"]["locked_recipe"]["metrics"]["top2"][
            "net20"
        ]["net_mean_pct_at_cost"]
        - result["stability_report"]["price_core_control"]["metrics"]["top2"][
            "net20"
        ]["net_mean_pct_at_cost"]
    )
    check(
        "stability_delta",
        abs(
            stability_delta
            - result["stability_report"]["top2_net20_delta_vs_price_core_pct"]
        )
        <= TOLERANCE,
    )

    failed = [item["name"] for item in checks if not item["passed"]]
    payload = {
        "schema_version": 1,
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": sha256_file(result_path),
        "checks": checks,
        "checks_passed": len(checks) - len(failed),
        "checks_total": len(checks),
        "failed_checks": failed,
        "recipes_recomputed": recipes_checked,
        "numeric_values_compared": len(numeric_differences),
        "max_absolute_metric_difference": max_difference,
        "max_absolute_max_t_difference": max_t_difference,
        "decision_recomputed": {
            "formal_survivors": recomputed_survivors,
            "retained_after_union_prune": retained,
            "stability_delta_vs_price_core_pct": stability_delta,
        },
        "passed": not failed,
    }
    write_json(payload, Path(args.output).resolve())
    if failed:
        raise SystemExit("v0.6 result audit failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
