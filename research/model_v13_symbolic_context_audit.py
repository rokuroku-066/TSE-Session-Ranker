#!/usr/bin/env python3
"""Independently audit the v1.3 symbolic-context result and picks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tse_session_ranker.validation import paired_moving_block_bootstrap  # noqa: E402


BASE_PROTOCOL = ROOT / "research/model_v13_symbolic_context_protocol.json"
BASE_PROTOCOL_SHA256 = (
    "7762221b33781a9d976487e50c1f5483bfc7ec6cea0e2b615d19ed36cf0b9703"
)
INPUT_ERRATUM = ROOT / "research/model_v13_symbolic_context_input_erratum_v2.json"
INPUT_ERRATUM_SHA256 = (
    "991ef4dfd20171d9be371d1b8d69d4074fe6536339bb7ff264f9708758c67853"
)
RUNNER = ROOT / "research/model_v13_symbolic_context_runner.py"
KNOWN_CONTROL_RESULT = ROOT / "research/model_v08_target_result.json"
CONTROL = "C00_DAILY_RANK_RIDGE"
CANDIDATES = (
    "CT01_ABSOLUTE_PATH",
    "CT02_RELATIVE_PATH",
    "CT03_DIRECTION_PATH",
    "CT04_FIXED_CONSENSUS",
)
CAPACITIES = (1, 2)
SUBPERIODS = {
    "discovery": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-31")),
    "confirmation_a": (pd.Timestamp("2024-11-01"), pd.Timestamp("2025-03-31")),
    "confirmation_b": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
}
FAMILY_SIZE = 8
SAMPLES = 10_000
RANDOM_STATE = 20_260_727
TOLERANCE = 1e-12


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


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    return value


def write_json(value: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(
            json_safe(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def assert_close(left: float, right: float, label: str) -> None:
    if not math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=TOLERANCE,
    ):
        raise AssertionError(f"{label}: expected {right}, got {left}")


def semantic_hash(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    canonical["date"] = pd.to_datetime(canonical["date"]).dt.strftime("%Y-%m-%d")
    canonical = canonical.sort_values(
        ["candidate_id", "date", "model_rank"], kind="stable"
    ).reset_index(drop=True)
    return hashlib.sha256(
        canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def daily_returns(
    picks: pd.DataFrame,
    *,
    capacity: int,
    cost_bps: float,
) -> pd.Series:
    selected = picks[picks["model_rank"].le(capacity)].copy()
    executed = selected["label"].notna()
    selected["_net"] = selected["oc_return_pct"].fillna(0.0) - (
        executed.astype(float) * cost_bps / 100.0
    )
    values = selected.groupby("date", sort=True)["_net"].sum().div(capacity)
    if len(values) != 266 or values.isna().any():
        raise AssertionError("audit daily schedule is incomplete")
    return values


def top_codes_cash(
    picks: pd.DataFrame,
    *,
    capacity: int,
    count: int = 10,
) -> tuple[pd.Series, list[str]]:
    selected = picks[picks["model_rank"].le(capacity)].copy()
    executed = selected["label"].notna()
    selected["_net"] = selected["oc_return_pct"].fillna(0.0) - (
        executed.astype(float) * 0.40
    )
    codes = [
        str(code)
        for code in (
            selected.dropna(subset=["code"])
            .groupby("code", sort=False)["_net"]
            .sum()
            .sort_values(ascending=False, kind="stable")
            .head(count)
            .index
        )
    ]
    neutral = picks.copy()
    mask = neutral["model_rank"].le(capacity) & neutral["code"].astype(str).isin(codes)
    neutral.loc[mask, ["label", "oc_return_pct"]] = np.nan
    return daily_returns(neutral, capacity=capacity, cost_bps=40.0), codes


def known_control_net20() -> float:
    document = read_json(KNOWN_CONTROL_RESULT)
    return float(
        document["candidate_results"]["stage_2_rank_target_sensitivity"][
            "results"
        ]["rank_ridge_expand_a1"]["cost_metrics"]["20"][
            "net_mean_pct_at_cost"
        ]
    )


def audit_variant(
    stored: dict[str, Any],
    candidate: pd.DataFrame,
    control: pd.DataFrame,
) -> dict[str, Any]:
    capacity = int(stored["capacity"])
    daily_by_cost = {
        str(cost): daily_returns(candidate, capacity=capacity, cost_bps=float(cost))
        for cost in (20, 40, 60)
    }
    for cost, daily in daily_by_cost.items():
        assert_close(
            float(daily.mean()),
            float(stored["cost_metrics"][cost]["net_mean_pct_at_cost"]),
            f"{stored['variant_id']} net{cost}",
        )
    daily40 = daily_by_cost["40"]
    control40 = daily_returns(control, capacity=capacity, cost_bps=40.0)
    subperiods = {
        name: float(daily40.loc[start:end].mean())
        for name, (start, end) in SUBPERIODS.items()
    }
    for name, value in subperiods.items():
        assert_close(
            value,
            float(stored["subperiod_net40_mean_pct"][name]),
            f"{stored['variant_id']} {name}",
        )
    top10_removed = float(daily40.drop(daily40.nlargest(10).index).mean())
    assert_close(
        top10_removed,
        float(stored["top10_days_removed_net40_mean_pct"]),
        f"{stored['variant_id']} top10-day removal",
    )
    code_cash, codes = top_codes_cash(candidate, capacity=capacity)
    if codes != [str(value) for value in stored["top10_profit_codes"]]:
        raise AssertionError(f"{stored['variant_id']} top-profit code set changed")
    assert_close(
        float(code_cash.mean()),
        float(stored["top10_profit_codes_cash_net40_mean_pct"]),
        f"{stored['variant_id']} top-code removal",
    )
    monthly = daily40.groupby(daily40.index.to_period("M")).mean()
    if int(monthly.gt(0.0).sum()) != int(stored["positive_months_net40"]):
        raise AssertionError(f"{stored['variant_id']} positive-month count changed")
    signaled = candidate[candidate["model_rank"].le(capacity)].dropna(
        subset=["code"]
    )
    counts = signaled["code"].astype(str).value_counts()
    unique_codes = int(len(counts))
    maximum_share = float(counts.iloc[0] / len(signaled))
    if unique_codes != int(stored["unique_selected_codes"]):
        raise AssertionError(f"{stored['variant_id']} unique-code count changed")
    assert_close(
        maximum_share,
        float(stored["maximum_code_selection_share"]),
        f"{stored['variant_id']} code share",
    )
    confidence = 1.0 - (1.0 - 0.90) / FAMILY_SIZE
    paired = paired_moving_block_bootstrap(
        daily40,
        control40,
        block_length=5,
        samples=SAMPLES,
        confidence=confidence,
        random_state=RANDOM_STATE + CAPACITIES.index(capacity),
    )
    stored_paired = stored["paired_vs_control_net40"]
    for key, value in asdict(paired).items():
        if isinstance(value, float):
            assert_close(
                value,
                float(stored_paired[key]),
                f"{stored['variant_id']} paired {key}",
            )
        elif value != stored_paired[key]:
            raise AssertionError(
                f"{stored['variant_id']} paired {key} changed"
            )
    checks = {
        "net40_positive": float(daily40.mean()) > 0.0,
        "all_three_subperiods_positive": all(value > 0.0 for value in subperiods.values()),
        "top10_days_removed_positive": top10_removed > 0.0,
        "top10_profit_codes_cash_positive": float(code_cash.mean()) > 0.0,
        "positive_months_at_least_9_of_13": int(monthly.gt(0.0).sum()) >= 9,
        "familywise_paired_lower_vs_control_nonnegative": (
            paired.one_sided_lower_delta_pct >= 0.0
        ),
        "unique_codes_at_least_50": unique_codes >= 50,
        "maximum_code_share_at_most_10pct": maximum_share <= 0.10,
    }
    if checks != stored["gate_checks"]:
        raise AssertionError(f"{stored['variant_id']} gate checks changed")
    if all(checks.values()) is not bool(stored["gate_passed"]):
        raise AssertionError(f"{stored['variant_id']} gate decision changed")
    return {
        "variant_id": stored["variant_id"],
        "net20_mean_pct": float(daily_by_cost["20"].mean()),
        "net40_mean_pct": float(daily40.mean()),
        "net60_mean_pct": float(daily_by_cost["60"].mean()),
        "paired_lower_delta_pct": paired.one_sided_lower_delta_pct,
        "gate_passed": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        default="research/model_v13_symbolic_context_result.json",
    )
    parser.add_argument(
        "--picks",
        default="research/model_v13_symbolic_context_picks.csv",
    )
    parser.add_argument(
        "--output",
        default="research/model_v13_symbolic_context_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = read_json(args.result)
    if sha256_file(BASE_PROTOCOL) != BASE_PROTOCOL_SHA256:
        raise AssertionError("base protocol hash changed")
    if sha256_file(INPUT_ERRATUM) != INPUT_ERRATUM_SHA256:
        raise AssertionError("input erratum hash changed")
    if result["base_protocol_sha256"] != BASE_PROTOCOL_SHA256:
        raise AssertionError("result does not bind the base protocol")
    if result["input_erratum_sha256"] != INPUT_ERRATUM_SHA256:
        raise AssertionError("result does not bind the input erratum")
    if result["runner_sha256"] != sha256_file(RUNNER):
        raise AssertionError("result runner hash changed")
    if result["integrity"]["orders_allowed"] is not False:
        raise AssertionError("orders must remain disabled")
    if result["integrity"]["production_model_changed"] is not False:
        raise AssertionError("production model must remain unchanged")

    picks = pd.read_csv(
        args.picks,
        dtype={"code": "string"},
        float_precision="round_trip",
    )
    picks["date"] = pd.to_datetime(picks["date"], errors="raise")
    if semantic_hash(picks) != result["integrity"]["picks_semantic_sha256"]:
        raise AssertionError("picks semantic hash changed")
    expected_models = {CONTROL, *CANDIDATES}
    if set(picks["candidate_id"]) != expected_models:
        raise AssertionError("picks model set changed")
    if len(picks) != 5 * 266 * 2:
        raise AssertionError("picks row count changed")
    if picks.duplicated(["candidate_id", "date", "model_rank"]).any():
        raise AssertionError("picks contain duplicate slots")
    if not picks.groupby("candidate_id")["date"].nunique().eq(266).all():
        raise AssertionError("a model does not cover every scheduled date")
    observed = picks["oc_return_pct"].notna()
    expected_label = picks.loc[observed, "oc_return_pct"].gt(0.0).astype(float)
    if not np.array_equal(
        picks.loc[observed, "label"].to_numpy(dtype=float),
        expected_label.to_numpy(dtype=float),
    ):
        raise AssertionError("pick labels differ from return signs")

    by_model = {
        model: picks[picks["candidate_id"].eq(model)].copy()
        for model in expected_models
    }
    known = known_control_net20()
    observed_control = float(
        daily_returns(by_model[CONTROL], capacity=2, cost_bps=20.0).mean()
    )
    assert_close(observed_control, known, "v0.8 G0 control reproduction")

    stored_variants = {
        item["variant_id"]: item for item in result["variants"]
    }
    expected_variants = {
        f"{candidate}__top{capacity}"
        for candidate in CANDIDATES
        for capacity in CAPACITIES
    }
    if set(stored_variants) != expected_variants:
        raise AssertionError("stored variant set changed")
    variants = [
        audit_variant(
            stored_variants[f"{candidate}__top{capacity}"],
            by_model[candidate],
            by_model[CONTROL],
        )
        for candidate in CANDIDATES
        for capacity in CAPACITIES
    ]
    passers = [item["variant_id"] for item in variants if item["gate_passed"]]
    if passers != result["decision"]["gate_passers"]:
        raise AssertionError("result gate-passer list changed")
    if result["decision"]["forward_shadow_candidate"] is not None:
        raise AssertionError("failed family cannot qualify a forward shadow")
    if result["decision"]["production_candidate"] is not None:
        raise AssertionError("retrospective family cannot produce production")
    if result["decision"]["orders_allowed"] is not False:
        raise AssertionError("result decision must keep orders disabled")

    best = max(variants, key=lambda item: item["net40_mean_pct"])
    audit = {
        "schema_version": 1,
        "audit_id": "model_v13_symbolic_context_independent_audit_20260727",
        "result_sha256": sha256_file(args.result),
        "picks_file_sha256": sha256_file(args.picks),
        "base_protocol_sha256": BASE_PROTOCOL_SHA256,
        "input_erratum_sha256": INPUT_ERRATUM_SHA256,
        "runner_sha256": sha256_file(RUNNER),
        "audit_runner_sha256": sha256_file(__file__),
        "checks": {
            "artifact_binding": "PASS",
            "five_models_x_266_sessions_x_two_slots": "PASS",
            "label_return_sign": "PASS",
            "v08_g0_control_exact_reproduction": "PASS",
            "cost_and_robustness_recalculation": "PASS",
            "familywise_bootstrap_recalculation": "PASS",
            "gate_recalculation": "PASS",
            "production_and_order_boundary": "PASS"
        },
        "known_control_top2_net20_pct": known,
        "variants": variants,
        "best_point_estimate_variant": best["variant_id"],
        "best_point_estimate_net40_pct": best["net40_mean_pct"],
        "gate_passers": passers,
        "decision": {
            "audit_passed": True,
            "forward_shadow_candidate": None,
            "production_candidate": None,
            "orders_allowed": False,
            "conclusion": "The artifacts are internally reproducible. All eight preregistered variants fail the qualification gate."
        }
    }
    write_json(audit, args.output)


if __name__ == "__main__":
    main()
