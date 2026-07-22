#!/usr/bin/env python3
"""Independently reaggregate and audit the model-v0.5 research result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tse_session_ranker.io import read_frame, write_json  # noqa: E402
from tse_session_ranker.profit import (  # noqa: E402
    daily_portfolio_returns,
    profit_metrics,
)
from tse_session_ranker.validation import moving_block_bootstrap  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_recorded(path: str, relative_to: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root_candidate = ROOT / candidate
    return root_candidate if root_candidate.exists() else relative_to / candidate


def compare_values(expected: Any, observed: Any, path: str, diffs: list[float]) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(expected) != set(observed):
            raise AssertionError(f"mapping mismatch at {path}")
        for key in expected:
            compare_values(expected[key], observed[key], f"{path}.{key}", diffs)
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            raise AssertionError(f"list mismatch at {path}")
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            compare_values(left, right, f"{path}[{index}]", diffs)
        return
    if expected is None:
        if observed is None:
            return
        if isinstance(observed, (float, np.floating)) and not math.isfinite(
            float(observed)
        ):
            return
        raise AssertionError(f"null mismatch at {path}")
    if isinstance(expected, bool) or isinstance(observed, bool):
        if expected is not observed:
            raise AssertionError(f"boolean mismatch at {path}")
        return
    if isinstance(expected, (int, float)) and isinstance(
        observed, (int, float, np.integer, np.floating)
    ):
        left = float(expected)
        right = float(observed)
        if not math.isfinite(left) and not math.isfinite(right):
            return
        difference = abs(left - right)
        diffs.append(difference)
        if difference > 1e-10:
            raise AssertionError(f"numeric mismatch at {path}: {left} != {right}")
        return
    if expected != observed:
        raise AssertionError(f"value mismatch at {path}: {expected!r} != {observed!r}")


def recompute_metrics(
    picks: pd.DataFrame, *, bootstrap_samples: int, seed: int = 31
) -> dict[str, Any]:
    displayed = picks["code"].notna()
    rank1_display = displayed[picks["model_rank"].eq(1)]
    rank2_display = displayed[picks["model_rank"].eq(2)]
    by_date = picks.assign(_displayed=displayed).pivot(
        index="date", columns="model_rank", values="_displayed"
    )
    output: dict[str, Any] = {
        "display": {
            "rank1_rate": float(rank1_display.mean()),
            "rank2_rate": float(rank2_display.mean()),
            "both_slots_rate": float(
                by_date.reindex(columns=[1, 2], fill_value=False).all(axis=1).mean()
            ),
            "rank1_sessions": int(rank1_display.sum()),
            "rank2_sessions": int(rank2_display.sum()),
            "scheduled_sessions": int(picks["date"].nunique()),
        }
    }
    lower_bounds: list[float] = []
    for top_k in (1, 2):
        daily = daily_portfolio_returns(
            picks, top_k=top_k, cost_bps=20.0
        ).set_index("date")["net_return_pct"]
        interval = moving_block_bootstrap(
            daily,
            block_length=5,
            samples=bootstrap_samples,
            confidence=0.90,
            random_state=seed + top_k,
        )
        lower_bounds.append(interval.one_sided_lower_pct)
        output[f"top{top_k}"] = {
            "net20": profit_metrics(picks, top_k=top_k, cost_bps=20.0),
            "net40": profit_metrics(picks, top_k=top_k, cost_bps=40.0),
            "block5_bootstrap": asdict(interval),
        }
    rank2 = picks[picks["model_rank"].eq(2)].copy()
    rank2["model_rank"] = 1
    output["rank2_standalone_net20"] = profit_metrics(
        rank2, top_k=1, cost_bps=20.0
    )
    output["joint_selection_score"] = float(min(lower_bounds))
    return output


def candidate_order(item: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = item["metrics"]
    return (
        -metrics["top2"]["net20"]["net_mean_pct_at_cost"],
        -metrics["top2"]["block5_bootstrap"]["one_sided_lower_pct"],
        -metrics["top2"]["net20"]["top5_removed_net_mean_pct"],
        -metrics["top1"]["net20"]["net_mean_pct_at_cost"],
        int(item["feature_count"]),
        str(item["candidate_id"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        default=str(ROOT / "research/model_v05_broad_family_result.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "research/model_v05_broad_family_result_audit.json"),
    )
    args = parser.parse_args()
    result_path = Path(args.result).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"audit output already exists: {output_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest_path = result_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_path = resolve_recorded(result["protocol_path"], result_path.parent)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    input_lock_path = resolve_recorded(
        result["data"]["input_lock_path"], result_path.parent
    )
    input_lock = json.loads(input_lock_path.read_text(encoding="utf-8"))

    checks: dict[str, bool] = {}
    checks["result_manifest_hash"] = manifest["result_sha256"] == sha256_file(
        result_path
    )
    checks["protocol_hash"] = result["protocol_sha256"] == sha256_file(protocol_path)
    checks["input_lock_hash"] = result["data"]["input_lock_sha256"] == sha256_file(
        input_lock_path
    )
    checks["embedded_input_lock"] = result["data"]["input_lock"] == input_lock
    checks["calendar_hash"] = (
        result["data"]["calendar_sha256"]
        == input_lock["calendar"]["sha256"]
        == "966f4a416d0929487d850b16be87c9c1cd69cd9f8c0447a3e76214a5715c489e"
    )
    checks["known_benchmark_not_gate"] = not result[
        "known_benchmark_affects_selection_or_gate"
    ]
    checks["production_unchanged"] = (
        not result["production_model_changed"]
        and result["production_promotion"]
        == "forbidden_until_fresh_forward_validation"
    )
    run_receipt_path = resolve_recorded(
        result["run_receipt_path"], result_path.parent
    )
    checks["atomic_run_receipt"] = (
        run_receipt_path.exists()
        and result["run_receipt_sha256"] == sha256_file(run_receipt_path)
        and manifest["run_receipt_sha256"] == sha256_file(run_receipt_path)
    )

    stage_pick_frames: dict[str, pd.DataFrame] = {}
    for stage, recorded_path in result["pick_paths"].items():
        path = resolve_recorded(recorded_path, result_path.parent)
        checks[f"pick_hash_{stage}"] = (
            result["pick_sha256"][stage] == sha256_file(path)
        )
        frame = read_frame(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        stage_pick_frames[stage] = frame

    maximum_metric_difference = 0.0
    metric_candidates = 0
    protocol_periods = protocol["periods"]
    for stage, stage_results in result["stages"].items():
        frame = stage_pick_frames[stage]
        period = protocol_periods[stage]
        samples = int(period["bootstrap_samples"])
        for item in stage_results:
            if item["status"] != "completed":
                raise AssertionError(f"non-completed registry item: {item['candidate_id']}")
            candidate = frame[frame["candidate_id"].eq(item["candidate_id"])].copy()
            expected_rows = int(item["period"]["scheduled_sessions"]) * 2
            if len(candidate) != expected_rows:
                raise AssertionError(
                    f"scheduled slot mismatch for {stage}/{item['candidate_id']}"
                )
            if candidate.duplicated(["date", "model_rank"]).any():
                raise AssertionError(
                    f"duplicate slots for {stage}/{item['candidate_id']}"
                )
            if set(candidate["model_rank"].astype(int).unique()) != {1, 2}:
                raise AssertionError(
                    f"rank set mismatch for {stage}/{item['candidate_id']}"
                )
            observed = recompute_metrics(candidate, bootstrap_samples=samples)
            diffs: list[float] = []
            compare_values(item["metrics"], observed, "metrics", diffs)
            maximum_metric_difference = max(
                maximum_metric_difference, max(diffs, default=0.0)
            )
            metric_candidates += 1
    checks["all_metrics_recomputed"] = metric_candidates == sum(
        len(values) for values in result["stages"].values()
    )

    threshold = float(
        protocol["stage_policy"]["minimum_rank1_and_rank2_display_rate"]
    )
    family_results = result["stages"]["family_screen"]
    model_group: dict[str, str] = {
        model_name: group
        for group, model_names in protocol["family_groups"].items()
        for model_name in model_names
    }
    expected_shortlist: list[str] = []
    for group in protocol["family_groups"]:
        options = [
            item
            for item in family_results
            if model_group[item["model"]["name"]] == group
            and item["metrics"]["display"]["rank1_rate"] >= threshold
            and item["metrics"]["display"]["rank2_rate"] >= threshold
        ]
        expected_shortlist.append(min(options, key=candidate_order)["model"]["name"])
    observed_shortlist = [item["name"] for item in result["family_screen_shortlist"]]
    checks["family_group_shortlist"] = expected_shortlist == observed_shortlist

    design = [
        item
        for item in result["stages"]["feature_design"]
        if item["metrics"]["display"]["rank1_rate"] >= threshold
        and item["metrics"]["display"]["rank2_rate"] >= threshold
    ]
    ranked_design = sorted(design, key=candidate_order)
    expected_finalists = [item["candidate_id"] for item in ranked_design[:3]]
    observed_finalists = [
        item["candidate_id"] for item in result["feature_design_finalists"]
    ]
    checks["feature_design_finalists"] = expected_finalists == observed_finalists
    checks["champion_locked_before_confirmation"] = (
        result["selected_winner"]["candidate_id"] == expected_finalists[0]
        and result["selected_winner_stage"] == "feature_design"
        and result["confirmation_did_not_reselect"]
    )
    price_only = [
        item for item in ranked_design if not any(name.startswith("tdnet_") for name in item["features"])
    ]
    checks["price_only_reference"] = (
        result["price_only_reference"]["candidate_id"]
        == price_only[0]["candidate_id"]
    )
    checks["confirmation_contains_locked_champion"] = (
        result["selected_winner_confirmation"]["candidate_id"]
        == result["selected_winner"]["candidate_id"]
    )

    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise AssertionError("model-v05 audit checks failed: " + ", ".join(failed))
    payload = {
        "schema_version": 1,
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": sha256_file(result_path),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": sha256_file(protocol_path),
        "input_lock_sha256": sha256_file(input_lock_path),
        "status": "passed",
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "metric_candidates_recomputed": metric_candidates,
        "maximum_absolute_metric_difference": maximum_metric_difference,
    }
    write_json(payload, output_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
