#!/usr/bin/env python3
"""Independent P&L, multiplicity, reproducibility, and promotion-gate audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "research" / "model_v11_analog_protocol.json"
PROTOCOL_SHA_PATH = ROOT / "research" / "model_v11_analog_protocol.sha256"
RUNNER_PATH = ROOT / "research" / "model_v11_analog_runner.py"
V10_RESULT_PATH = ROOT / "research" / "model_v10_tdnet_text_result.json"
EXPECTED_PROTOCOL_SHA256 = (
    "823012afa89fe9ec982a7ca82c78c3f24697b6969d1d3fcaa73f5ee7d3b82e86"
)
ANALOG_IDS = (
    "A01_global_similarity_mean",
    "A02_issuer_excluded_mean",
    "A03_time_decay_mean",
    "A04_cross_code_one_per_issuer",
    "A05_neighbor_lower_quantile",
    "A06_dual_tail_neighbor_utility",
    "A07_event_family_prototype",
    "A08_same_issuer_memory",
    "A09_robust_neighbor_consensus",
    "A10_family_then_text_cross_code",
)
COMPARATOR_IDS = ("L4_price_control", "T02_char_value_event_only")
COSTS = (0, 20, 40, 60)
SEED = 20260723
REPLICATES = 10_000
BLOCK = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def daily_returns(
    picks: pd.DataFrame, top_k: int, cost_bps: float
) -> pd.Series:
    executed = picks["oc_return_pct"].notna()
    slot = (
        picks["oc_return_pct"].fillna(0.0)
        - executed.astype(float) * cost_bps / 100.0
    )
    return slot.groupby(picks["date"], sort=True).sum().div(top_k)


def recompute_metrics(
    picks: pd.DataFrame,
    top_k: int,
    slices: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, Any]:
    daily_by_cost: dict[int, pd.Series] = {}
    costs: dict[str, Any] = {}
    for cost in COSTS:
        values = daily_returns(picks, top_k, cost)
        daily_by_cost[cost] = values
        costs[str(cost)] = {
            "mean_pct": float(values.mean()),
            "median_pct": float(values.median()),
            "win_rate": float(values.gt(0.0).mean()),
            "days": int(len(values)),
        }
    monthly: dict[str, Any] = {}
    periods = daily_by_cost[40].index.to_period("M")
    for period in periods.unique():
        mask = periods == period
        monthly[str(period)] = {
            f"net{cost}_mean_pct" if cost else "gross_mean_pct": float(
                values.loc[mask].mean()
            )
            for cost, values in daily_by_cost.items()
        }
    slice_metrics: dict[str, Any] = {}
    for name, (start, end) in slices.items():
        mask = (
            (daily_by_cost[40].index >= start)
            & (daily_by_cost[40].index <= end)
        )
        row: dict[str, Any] = {"days": int(mask.sum())}
        for cost, values in daily_by_cost.items():
            label = f"net{cost}_mean_pct" if cost else "gross_mean_pct"
            row[label] = float(values.loc[mask].mean()) if mask.any() else None
        slice_metrics[name] = row

    daily20 = daily_by_cost[20]
    removed_dates = list(daily20.nlargest(min(20, len(daily20))).index)
    removed = daily20.drop(removed_dates)
    executed = picks["oc_return_pct"].notna()
    contribution = (
        picks["oc_return_pct"].fillna(0.0) - executed.astype(float) * 0.20
    ).div(top_k)
    by_code = contribution.groupby(picks["code"], dropna=True).sum().sort_values(
        ascending=False
    )
    top_codes = [str(value) for value in by_code.loc[by_code.gt(0.0)].head(10).index]
    cash_contribution = contribution.mask(picks["code"].isin(top_codes), 0.0)
    cash_daily = cash_contribution.groupby(picks["date"], sort=True).sum()
    positive_total = float(by_code.loc[by_code.gt(0.0)].sum())
    largest_positive_share = (
        float(by_code.iloc[0] / positive_total)
        if len(by_code) and by_code.iloc[0] > 0.0 and positive_total > 0.0
        else 0.0
    )
    return {
        "costs": costs,
        "monthly": monthly,
        "positive_months_net40": int(
            sum(row["net40_mean_pct"] > 0.0 for row in monthly.values())
        ),
        "months": int(len(monthly)),
        "slices": slice_metrics,
        "best20_session_removal": {
            "removed_dates": [str(pd.Timestamp(value).date()) for value in removed_dates],
            "remaining_days": int(len(removed)),
            "net20_mean_pct": float(removed.mean()) if len(removed) else None,
        },
        "top10_code_cash": {
            "codes": top_codes,
            "net20_mean_pct": float(cash_daily.mean()),
        },
        "concentration": {
            "largest_positive_code_share": largest_positive_share,
            "unique_codes": int(picks["code"].nunique(dropna=True)),
        },
        "filled_slots": int(executed.sum()),
        "scheduled_slots": int(len(picks)),
        "cash_slots": int((~executed).sum()),
        "slot_fill_rate": float(executed.mean()),
        "days_with_at_least_one_filled_slot": int(
            executed.groupby(picks["date"], sort=True).any().sum()
        ),
    }


def values_equal(left: Any, right: Any, path: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            failures.append(f"{path}: key mismatch")
            return failures
        for key in left:
            failures.extend(values_equal(left[key], right[key], f"{path}/{key}"))
        return failures
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            failures.append(f"{path}: length mismatch")
            return failures
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            failures.extend(values_equal(a, b, f"{path}/{index}"))
        return failures
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        if not np.isclose(float(left), float(right), atol=1e-12, rtol=1e-12):
            failures.append(f"{path}: {left!r} != {right!r}")
        return failures
    if left != right:
        failures.append(f"{path}: {left!r} != {right!r}")
    return failures


def month_block_indices(
    dates: pd.DatetimeIndex, rng: np.random.Generator
) -> np.ndarray:
    output: list[int] = []
    periods = dates.to_period("M")
    for period in periods.unique():
        local = np.flatnonzero(periods == period)
        chosen: list[int] = []
        while len(chosen) < len(local):
            start = int(rng.integers(0, len(local)))
            chosen.extend(
                local[(start + np.arange(BLOCK)) % len(local)].tolist()
            )
        output.extend(chosen[: len(local)])
    return np.asarray(output, dtype=int)


def bootstrap_multiplicity(
    picks: pd.DataFrame,
) -> tuple[dict[str, Any], str, dict[str, pd.Series]]:
    top1 = picks.loc[picks["top_k"].eq(1)]
    daily: dict[str, pd.Series] = {}
    for candidate_id in (*COMPARATOR_IDS, *ANALOG_IDS):
        local = top1.loc[top1["candidate_id"].eq(candidate_id)]
        daily[candidate_id] = daily_returns(local, 1, 40.0)
    stronger = max(
        COMPARATOR_IDS, key=lambda name: float(daily[name].mean())
    )
    dates = pd.DatetimeIndex(daily[stronger].index)
    comparator = daily[stronger].reindex(dates).to_numpy(dtype=float)
    matrix = np.column_stack(
        [
            daily[name].reindex(dates).to_numpy(dtype=float) - comparator
            for name in ANALOG_IDS
        ]
    )
    if not np.isfinite(matrix).all():
        raise RuntimeError("bootstrap daily matrix contains missing values")
    observed = matrix.mean(axis=0)
    centered = matrix - observed
    rng = np.random.default_rng(SEED)
    max_centered = np.empty(REPLICATES, dtype=float)
    for repeat in range(REPLICATES):
        index = month_block_indices(dates, rng)
        max_centered[repeat] = centered[index].mean(axis=0).max()
    critical = float(np.quantile(max_centered, 0.90))
    by_candidate = {}
    for column, candidate_id in enumerate(ANALOG_IDS):
        adjusted_p = float(
            (1 + np.count_nonzero(max_centered >= observed[column]))
            / (REPLICATES + 1)
        )
        by_candidate[candidate_id] = {
            "delta_vs_stronger_comparator_net40_mean_pct": float(
                observed[column]
            ),
            "simultaneous_90pct_lower_bound_pct": float(
                observed[column] - critical
            ),
            "familywise_adjusted_p_value": adjusted_p,
        }
    best_column = int(np.argmax(observed))
    return (
        {
            "seed": SEED,
            "replicates": REPLICATES,
            "block_length_sessions_within_month": BLOCK,
            "stronger_comparator": stronger,
            "simultaneous_90pct_critical_value": critical,
            "best_candidate_by_uplift": ANALOG_IDS[best_column],
            "best_observed_uplift_pct": float(observed[best_column]),
            "familywise_reality_check_p_value": by_candidate[
                ANALOG_IDS[best_column]
            ]["familywise_adjusted_p_value"],
            "candidates": by_candidate,
        },
        stronger,
        daily,
    )


def compare_known_comparators(
    result: dict[str, Any], v10_result: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    for candidate_id in COMPARATOR_IDS:
        for top_k in (1, 2):
            name = f"top{top_k}"
            current = result["metrics"][candidate_id][name]
            known = v10_result["candidates"][candidate_id][name]
            failures.extend(
                values_equal(
                    current["costs"],
                    known["costs"],
                    f"known_comparator/{candidate_id}/{name}/costs",
                )
            )
            scalar_pairs = {
                "positive_months_net40": "positive_months",
                "months": "months",
                "filled_slots": "filled_slots",
                "scheduled_slots": "scheduled_slots",
                "slot_fill_rate": "slot_fill_rate",
                "days_with_at_least_one_filled_slot": (
                    "days_with_at_least_one_filled_slot"
                ),
            }
            for current_key, known_key in scalar_pairs.items():
                failures.extend(
                    values_equal(
                        current[current_key],
                        known[known_key],
                        (
                            f"known_comparator/{candidate_id}/{name}/"
                            f"{current_key}"
                        ),
                    )
                )
            failures.extend(
                values_equal(
                    current["best20_session_removal"]["net20_mean_pct"],
                    known["top20_winning_days_removed_net20_mean_pct"],
                    f"known_comparator/{candidate_id}/{name}/best20",
                )
            )
            failures.extend(
                values_equal(
                    current["top10_code_cash"]["net20_mean_pct"],
                    known["top10_profit_codes_cash_net20_mean_pct"],
                    f"known_comparator/{candidate_id}/{name}/top10_cash",
                )
            )
            failures.extend(
                values_equal(
                    current["top10_code_cash"]["codes"],
                    known["top10_profit_codes"],
                    f"known_comparator/{candidate_id}/{name}/top10_codes",
                )
            )
            current_monthly = {
                month: row["net40_mean_pct"]
                for month, row in current["monthly"].items()
            }
            failures.extend(
                values_equal(
                    current_monthly,
                    known["monthly_net40_mean_pct"],
                    f"known_comparator/{candidate_id}/{name}/monthly_net40",
                )
            )
    return failures


def candidate_gates(
    result: dict[str, Any],
    bootstrap: dict[str, Any],
    integrity_passes: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    l4 = result["metrics"]["L4_price_control"]["top1"]["costs"]["40"][
        "mean_pct"
    ]
    t02 = result["metrics"]["T02_char_value_event_only"]["top1"]["costs"][
        "40"
    ]["mean_pct"]
    for candidate_id in ANALOG_IDS:
        top1 = result["metrics"][candidate_id]["top1"]
        top2 = result["metrics"][candidate_id]["top2"]
        checks = {
            "top1_net40_positive": top1["costs"]["40"]["mean_pct"] > 0.0,
            "top2_net40_positive": top2["costs"]["40"]["mean_pct"] > 0.0,
            "top1_uplift_vs_L4_positive": (
                top1["costs"]["40"]["mean_pct"] - l4 > 0.0
            ),
            "top1_uplift_vs_T02_positive": (
                top1["costs"]["40"]["mean_pct"] - t02 > 0.0
            ),
            "familywise_lower_bound_positive": bootstrap["candidates"][
                candidate_id
            ]["simultaneous_90pct_lower_bound_pct"]
            > 0.0,
            "familywise_p_at_most_0_10": bootstrap["candidates"][
                candidate_id
            ]["familywise_adjusted_p_value"]
            <= 0.10,
            "best20_removed_net20_positive": top1[
                "best20_session_removal"
            ]["net20_mean_pct"]
            > 0.0,
            "top10_codes_cash_net20_positive": top1["top10_code_cash"][
                "net20_mean_pct"
            ]
            > 0.0,
            "all_three_slices_net40_positive": all(
                row["net40_mean_pct"] > 0.0
                for row in top1["slices"].values()
            ),
            "at_least_four_of_five_months_net40_positive": (
                top1["months"] == 5 and top1["positive_months_net40"] >= 4
            ),
            "at_least_80pct_days_filled": (
                top1["days_with_at_least_one_filled_slot"]
                / top1["costs"]["40"]["days"]
                >= 0.80
            ),
            "all_integrity_audits_pass": integrity_passes,
        }
        output[candidate_id] = {
            "checks": checks,
            "failed_checks": [
                name for name, passes in checks.items() if not passes
            ],
            "passes_all": all(checks.values()),
        }
    return output


def report_markdown(
    protocol: dict[str, Any],
    result: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    comparator = audit["multiplicity"]["stronger_comparator"]
    comparator_mean = result["metrics"][comparator]["top1"]["costs"]["40"][
        "mean_pct"
    ]
    lines = [
        "# Model v11 — Historical analog retrieval",
        "",
        "## 結論",
        "",
        f"- 判定: **{audit['decision']['retrospective_decision']}**",
        f"- 本番昇格: **なし**（production gate: {str(audit['decision']['production_gate_passes']).lower()}）",
        f"- familywise比較対照: `{comparator}`（top1 net40 {comparator_mean:.4f}%）",
        f"- PIT対象: {result['coverage']['strict_complete_score_sessions']} sessions / "
        f"{', '.join(result['coverage']['score_months'])}",
        "- TDnet履歴HTMLには観測時刻sidecarがなく、このrun単独では本番採用を認めない。",
        "",
        "## 事前登録",
        "",
        f"- Protocol: `{protocol['protocol_id']}`",
        f"- SHA-256: `{result['protocol_sha256']}`",
        f"- 構造的仮説: {protocol['hypothesis_family_size']}件",
        "- 共通k=31。k/閾値grid、結果確認後の候補変更、analog候補の係数学習はいずれもなし。",
        "",
        "## 結果",
        "",
        "| candidate | top1 net40 | top2 net40 | Δ vs L4 | Δ vs T02 | best20除去 net20 | top10 code cash net20 | +months | fill days | simultaneous LB | adj. p | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    l4 = result["metrics"]["L4_price_control"]["top1"]["costs"]["40"][
        "mean_pct"
    ]
    t02 = result["metrics"]["T02_char_value_event_only"]["top1"]["costs"][
        "40"
    ]["mean_pct"]
    for candidate_id in ANALOG_IDS:
        top1 = result["metrics"][candidate_id]["top1"]
        top2 = result["metrics"][candidate_id]["top2"]
        boot = audit["multiplicity"]["candidates"][candidate_id]
        gate = audit["candidate_gates"][candidate_id]["passes_all"]
        lines.append(
            f"| `{candidate_id}` | {top1['costs']['40']['mean_pct']:.4f}% | "
            f"{top2['costs']['40']['mean_pct']:.4f}% | "
            f"{top1['costs']['40']['mean_pct'] - l4:+.4f}% | "
            f"{top1['costs']['40']['mean_pct'] - t02:+.4f}% | "
            f"{top1['best20_session_removal']['net20_mean_pct']:.4f}% | "
            f"{top1['top10_code_cash']['net20_mean_pct']:.4f}% | "
            f"{top1['positive_months_net40']}/5 | "
            f"{top1['days_with_at_least_one_filled_slot']}/"
            f"{top1['costs']['40']['days']} | "
            f"{boot['simultaneous_90pct_lower_bound_pct']:.4f}% | "
            f"{boot['familywise_adjusted_p_value']:.4f} | "
            f"{'PASS' if gate else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "### 対照",
            "",
            "| comparator | top1 net20 | top1 net40 | top1 net60 | top2 net40 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for candidate_id in COMPARATOR_IDS:
        top1 = result["metrics"][candidate_id]["top1"]
        top2 = result["metrics"][candidate_id]["top2"]
        lines.append(
            f"| `{candidate_id}` | {top1['costs']['20']['mean_pct']:.4f}% | "
            f"{top1['costs']['40']['mean_pct']:.4f}% | "
            f"{top1['costs']['60']['mean_pct']:.4f}% | "
            f"{top2['costs']['40']['mean_pct']:.4f}% |"
        )
    best = audit["decision"]["best_point_estimate_candidate"]
    best_metrics = result["metrics"][best]["top1"]
    failed = audit["candidate_gates"][best]["failed_checks"]
    lines.extend(
        [
            "",
            "## 独立gate",
            "",
            f"点推定最大は `{best}`（top1 net40 "
            f"{best_metrics['costs']['40']['mean_pct']:.4f}%）。",
            "",
            "未達条件: " + ", ".join(f"`{name}`" for name in failed),
            "",
            "全候補について月次・3 slice・best20除去・top10 code cash・top2・"
            "familywise補正を同時に要求した。通過候補がなければwinnerを選ばない。",
            "",
            "## 監査",
            "",
            f"- Input/PIT/mutation: **{'PASS' if audit['integrity']['passes'] else 'FAIL'}**",
            f"- P&L再計算: **{'PASS' if audit['pnl_reproduction']['passes'] else 'FAIL'}**",
            f"- v10 L4/T02同値再現: **{'PASS' if audit['known_comparator_reproduction']['passes'] else 'FAIL'}**",
            f"- byte-identical picks rerun: **{'PASS' if audit['deterministic_reproduction']['picks_byte_identical'] else 'FAIL'}**",
            f"- candidate metrics rerun: **{'PASS' if audit['deterministic_reproduction']['metrics_identical'] else 'FAIL'}**",
            "",
            "## 本番判断",
            "",
            "このretrospective runから本番モデルへの変更は行っていない。"
            "事前登録された全retrospective条件に加え、観測時刻provenanceと登録後120件以上の"
            "新規strict-source-complete sessionでprospective gateを満たし、明示的人手承認を"
            "得るまでは本番採用不可。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT / "research" / "model_v11_analog_result.json",
    )
    parser.add_argument(
        "--picks",
        type=Path,
        default=ROOT / "research" / "model_v11_analog_picks.csv",
    )
    parser.add_argument(
        "--repro-result",
        type=Path,
        default=Path("/tmp/model_v11_analog_repro_result.json"),
    )
    parser.add_argument(
        "--repro-picks",
        type=Path,
        default=Path("/tmp/model_v11_analog_repro_picks.csv"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=ROOT / "research" / "model_v11_analog_audit.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "research" / "model_v11_analog_report.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    repro = json.loads(args.repro_result.read_text(encoding="utf-8"))
    v10_result = json.loads(V10_RESULT_PATH.read_text(encoding="utf-8"))
    picks = pd.read_csv(
        args.picks, parse_dates=["date"], dtype={"code": "object"}
    )
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("protocol changed after registration")
    if result["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("result is not bound to the registered protocol")
    if sha256_file(args.picks) != result["artifacts"]["picks_sha256"]:
        raise RuntimeError("picks hash does not match result")
    if set(picks["candidate_id"].unique()) != set(
        (*COMPARATOR_IDS, *ANALOG_IDS)
    ):
        raise RuntimeError("unexpected candidate ids in picks")

    slices = {
        name: (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
        for name, bounds in protocol["sample_and_folds"][
            "predeclared_slices"
        ].items()
    }
    pnl_failures: list[str] = []
    for candidate_id in (*COMPARATOR_IDS, *ANALOG_IDS):
        for top_k in (1, 2):
            local = picks.loc[
                picks["candidate_id"].eq(candidate_id)
                & picks["top_k"].eq(top_k)
            ].copy()
            expected_rows = result["coverage"][
                "strict_complete_score_sessions"
            ] * top_k
            if len(local) != expected_rows:
                pnl_failures.append(
                    f"{candidate_id}/top{top_k}: {len(local)} rows, "
                    f"expected {expected_rows}"
                )
                continue
            recomputed = recompute_metrics(local, top_k, slices)
            pnl_failures.extend(
                values_equal(
                    recomputed,
                    result["metrics"][candidate_id][f"top{top_k}"],
                    f"{candidate_id}/top{top_k}",
                )
            )

    repro_metric_failures = values_equal(
        result["metrics"], repro["metrics"], "deterministic_metrics"
    )
    repro_fold_failures = values_equal(
        result["folds"], repro["folds"], "deterministic_folds"
    )
    picks_identical = sha256_file(args.picks) == sha256_file(args.repro_picks)
    known_comparator_failures = compare_known_comparators(result, v10_result)
    integrity_checks = {
        "protocol_sidecar_matches": (
            PROTOCOL_SHA_PATH.read_text(encoding="utf-8").strip()
            == f"{EXPECTED_PROTOCOL_SHA256}  model_v11_analog_protocol.json"
        ),
        "runner_hash_matches_result": (
            sha256_file(RUNNER_PATH) == result["runner_sha256"]
        ),
        "strictly_prior_training_all_folds": result["integrity"][
            "strictly_prior_training_all_folds"
        ],
        "future_publication_violations_zero": (
            result["integrity"]["future_publication_violations"] == 0
        ),
        "source_missing_never_encoded_no_event": result["integrity"][
            "source_missing_never_encoded_no_event"
        ],
        "target_session_outcome_mutation_passes": result["integrity"][
            "target_session_outcome_mutation"
        ]["passes"],
        "historical_cache_observation_metadata_available": result["coverage"][
            "historical_cache_observation_metadata_available"
        ],
    }
    technical_integrity_passes = all(
        value
        for name, value in integrity_checks.items()
        if name != "historical_cache_observation_metadata_available"
    )
    deterministic_passes = (
        picks_identical
        and not repro_metric_failures
        and not repro_fold_failures
    )
    pnl_passes = not pnl_failures
    known_comparators_pass = not known_comparator_failures
    all_retrospective_audits_pass = (
        technical_integrity_passes
        and deterministic_passes
        and pnl_passes
        and known_comparators_pass
    )

    multiplicity, stronger, _ = bootstrap_multiplicity(picks)
    gates = candidate_gates(
        result, multiplicity, all_retrospective_audits_pass
    )
    passing = [
        candidate_id
        for candidate_id in ANALOG_IDS
        if gates[candidate_id]["passes_all"]
    ]
    selected = (
        max(
            passing,
            key=lambda name: result["metrics"][name]["top1"]["costs"]["40"][
                "mean_pct"
            ],
        )
        if passing
        else None
    )
    best_point = max(
        ANALOG_IDS,
        key=lambda name: result["metrics"][name]["top1"]["costs"]["40"][
            "mean_pct"
        ],
    )
    retrospective_decision = (
        "retain_single_candidate_for_new_forward_shadow"
        if selected is not None
        else "reject_family"
    )
    production_checks = {
        "retrospective_gate_has_winner": selected is not None,
        "historical_observation_provenance_proven": bool(
            result["coverage"][
                "historical_cache_observation_metadata_available"
            ]
        ),
        "new_120_session_prospective_gate_passed": False,
        "human_production_approval_recorded": False,
    }
    audit: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": result["runner_sha256"],
        "result_sha256": sha256_file(args.result),
        "picks_sha256": sha256_file(args.picks),
        "repro_result_sha256": sha256_file(args.repro_result),
        "repro_picks_sha256": sha256_file(args.repro_picks),
        "integrity": {
            "checks": integrity_checks,
            "passes": technical_integrity_passes,
            "note": "Historical observation provenance is a production gate, not a retrospective implementation-integrity check.",
        },
        "pnl_reproduction": {
            "passes": pnl_passes,
            "failure_count": len(pnl_failures),
            "failures": pnl_failures[:100],
        },
        "known_comparator_reproduction": {
            "v10_result_sha256": sha256_file(V10_RESULT_PATH),
            "passes": known_comparators_pass,
            "failure_count": len(known_comparator_failures),
            "failures": known_comparator_failures[:100],
        },
        "deterministic_reproduction": {
            "picks_byte_identical": picks_identical,
            "metrics_identical": not repro_metric_failures,
            "folds_identical": not repro_fold_failures,
            "metric_failures": repro_metric_failures[:100],
            "fold_failures": repro_fold_failures[:100],
        },
        "multiplicity": multiplicity,
        "candidate_gates": gates,
        "decision": {
            "stronger_comparator": stronger,
            "best_point_estimate_candidate": best_point,
            "passing_candidate_ids": passing,
            "selected_forward_shadow_candidate": selected,
            "retrospective_decision": retrospective_decision,
            "production_checks": production_checks,
            "production_gate_passes": all(production_checks.values()),
            "production_model_changed": False,
            "orders_allowed": False,
        },
    }
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = report_markdown(protocol, result, audit)
    args.report_output.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "audit": str(args.audit_output),
                "report": str(args.report_output),
                "retrospective_decision": retrospective_decision,
                "passing_candidate_ids": passing,
                "production_gate_passes": False,
                "best_point_estimate_candidate": best_point,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
