#!/usr/bin/env python3
"""Independent audit for the model-v11 cross-stock graph experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "research" / "model_v11_graph_protocol.json"
PROTOCOL_SHA_PATH = ROOT / "research" / "model_v11_graph_protocol.sha256"
RUNNER_PATH = ROOT / "research" / "model_v11_graph_runner.py"
V10_ARCHITECTURE_RESULT = (
    ROOT / "research" / "model_v10_architectures_result.json"
)
V10_Z17_RESULT = ROOT / "research" / "model_v10_peer_residual_result.json"
EXPECTED_PROTOCOL_SHA256 = (
    "2f2382716ba9f7b151988470334b4da93bee5f68e30f7264beedcf16bf790244"
)
COMPARATOR_IDS = ("C00_daily_rank_ridge", "C01_L4_price_control")
GRAPH_IDS = (
    "G01_positive_oc_corr_message",
    "G02_signed_oc_corr_message",
    "G03_directed_lead_lag_message",
    "G04_overnight_corr_message",
    "G05_cojump_neighbor_pressure",
    "G06_tdnet_event_neighbor_spillover",
    "G07_neighbor_conditioned_empirical_uplift",
    "G08_dynamic_community_archetype",
    "G09_two_hop_oc_diffusion",
    "G10_graph_disagreement_cash",
)
COSTS = (0, 20, 40, 60)
PERIODS = {
    "discovery": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-31")),
    "confirmation_a": (pd.Timestamp("2024-11-01"), pd.Timestamp("2025-03-31")),
    "confirmation_b": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
}
SEED = 20260723
REPLICATES = 10_000
BLOCK = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def daily_returns(
    picks: pd.DataFrame, top_k: int, cost_bps: int
) -> pd.Series:
    executed = picks["oc_return_pct"].notna()
    slot = (
        picks["oc_return_pct"].fillna(0.0)
        - executed.astype(float) * cost_bps / 100.0
    )
    return slot.groupby(picks["date"], sort=True).sum().div(top_k)


def recompute_metrics(picks: pd.DataFrame, top_k: int) -> dict[str, Any]:
    daily_by_cost = {
        cost: daily_returns(picks, top_k, cost) for cost in COSTS
    }
    costs = {
        str(cost): {
            "mean_pct": float(values.mean()),
            "median_pct": float(values.median()),
            "win_rate": float(values.gt(0.0).mean()),
            "days": int(len(values)),
        }
        for cost, values in daily_by_cost.items()
    }
    periods = daily_by_cost[40].index.to_period("M")
    monthly = {}
    for period in periods.unique():
        mask = periods == period
        monthly[str(period)] = {
            f"net{cost}_mean_pct" if cost else "gross_mean_pct": float(
                values.loc[mask].mean()
            )
            for cost, values in daily_by_cost.items()
        }
    slices = {}
    for name, (start, end) in PERIODS.items():
        mask = (
            (daily_by_cost[40].index >= start)
            & (daily_by_cost[40].index <= end)
        )
        row: dict[str, Any] = {"days": int(mask.sum())}
        for cost, values in daily_by_cost.items():
            label = f"net{cost}_mean_pct" if cost else "gross_mean_pct"
            row[label] = float(values.loc[mask].mean())
        slices[name] = row

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
    top_profit_codes = [
        str(value) for value in by_code.loc[by_code.gt(0.0)].head(10).index
    ]
    cash_daily = (
        contribution.mask(picks["code"].isin(top_profit_codes), 0.0)
        .groupby(picks["date"], sort=True)
        .sum()
    )
    positive_total = float(by_code.loc[by_code.gt(0.0)].sum())
    largest_share = (
        float(by_code.iloc[0] / positive_total)
        if len(by_code) and by_code.iloc[0] > 0.0 and positive_total > 0.0
        else 0.0
    )
    selection_counts = picks["code"].dropna().astype(str).value_counts()
    top10_selection_share = (
        float(selection_counts.head(10).sum() / len(picks))
        if len(picks)
        else 0.0
    )
    return {
        "costs": costs,
        "monthly": monthly,
        "positive_months_net40": int(
            sum(row["net40_mean_pct"] > 0.0 for row in monthly.values())
        ),
        "months": int(len(monthly)),
        "slices": slices,
        "best20_session_removal": {
            "removed_dates": [
                str(pd.Timestamp(value).date()) for value in removed_dates
            ],
            "remaining_days": int(len(removed)),
            "net20_mean_pct": float(removed.mean()),
        },
        "top10_code_cash": {
            "codes": top_profit_codes,
            "net20_mean_pct": float(cash_daily.mean()),
        },
        "concentration": {
            "largest_positive_code_share": largest_share,
            "top10_selection_share": top10_selection_share,
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
            return [f"{path}: key mismatch"]
        for key in left:
            failures.extend(values_equal(left[key], right[key], f"{path}/{key}"))
        return failures
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}: length mismatch"]
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
) -> tuple[dict[str, Any], str]:
    top1 = picks.loc[picks["top_k"].eq(1)]
    daily: dict[str, pd.Series] = {}
    for candidate_id in (*COMPARATOR_IDS, *GRAPH_IDS):
        daily[candidate_id] = daily_returns(
            top1.loc[top1["candidate_id"].eq(candidate_id)], 1, 40
        )
    stronger = max(
        COMPARATOR_IDS, key=lambda name: float(daily[name].mean())
    )
    dates = pd.DatetimeIndex(daily[stronger].index)
    comparator = daily[stronger].reindex(dates).to_numpy(dtype=float)
    matrix = np.column_stack(
        [
            daily[name].reindex(dates).to_numpy(dtype=float) - comparator
            for name in GRAPH_IDS
        ]
    )
    if not np.isfinite(matrix).all():
        raise RuntimeError("paired graph bootstrap matrix is nonfinite")
    observed = matrix.mean(axis=0)
    centered = matrix - observed
    rng = np.random.default_rng(SEED)
    maximum = np.empty(REPLICATES, dtype=float)
    for repeat in range(REPLICATES):
        index = month_block_indices(dates, rng)
        maximum[repeat] = centered[index].mean(axis=0).max()
    critical = float(np.quantile(maximum, 0.90))
    candidates = {}
    for column, candidate_id in enumerate(GRAPH_IDS):
        candidates[candidate_id] = {
            "delta_vs_stronger_comparator_net40_mean_pct": float(
                observed[column]
            ),
            "simultaneous_90pct_lower_bound_pct": float(
                observed[column] - critical
            ),
            "familywise_adjusted_p_value": float(
                (1 + np.count_nonzero(maximum >= observed[column]))
                / (REPLICATES + 1)
            ),
        }
    best = int(np.argmax(observed))
    return {
        "seed": SEED,
        "replicates": REPLICATES,
        "block_length_sessions_within_month": BLOCK,
        "stronger_comparator": stronger,
        "simultaneous_90pct_critical_value": critical,
        "best_candidate_by_uplift": GRAPH_IDS[best],
        "best_observed_uplift_pct": float(observed[best]),
        "familywise_reality_check_p_value": candidates[
            GRAPH_IDS[best]
        ]["familywise_adjusted_p_value"],
        "candidates": candidates,
    }, stronger


def compare_c00_v10(
    result: dict[str, Any], known: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    current = result["metrics"]["C00_daily_rank_ridge"]["top2"]
    reference = known["metrics"]["C00_daily_rank_ridge"]
    for cost in (20, 40, 60):
        failures.extend(
            values_equal(
                current["costs"][str(cost)]["mean_pct"],
                reference["cost"][str(cost)]["mean_pct"],
                f"C00/top2/net{cost}",
            )
        )
    current_monthly = {
        month: row["net20_mean_pct"]
        for month, row in current["monthly"].items()
    }
    failures.extend(
        values_equal(
            current_monthly,
            reference["monthly_net20"],
            "C00/top2/monthly_net20",
        )
    )
    current_slices = {
        name: row["net20_mean_pct"]
        for name, row in current["slices"].items()
    }
    failures.extend(
        values_equal(
            current_slices,
            reference["period_net20"],
            "C00/top2/slices_net20",
        )
    )
    failures.extend(
        values_equal(
            current["best20_session_removal"]["net20_mean_pct"],
            reference["tail_exclusion"]["20"]["20"],
            "C00/top2/best20_net20",
        )
    )
    failures.extend(
        values_equal(
            current["concentration"]["unique_codes"],
            reference["unique_codes"],
            "C00/top2/unique_codes",
        )
    )
    return failures


def candidate_gates(
    result: dict[str, Any],
    multiplicity: dict[str, Any],
    all_audits_pass: bool,
) -> dict[str, Any]:
    c00 = result["metrics"]["C00_daily_rank_ridge"]["top1"]["costs"]["40"][
        "mean_pct"
    ]
    c01 = result["metrics"]["C01_L4_price_control"]["top1"]["costs"]["40"][
        "mean_pct"
    ]
    output = {}
    for candidate_id in GRAPH_IDS:
        top1 = result["metrics"][candidate_id]["top1"]
        top2 = result["metrics"][candidate_id]["top2"]
        checks = {
            "top1_net40_positive": top1["costs"]["40"]["mean_pct"] > 0.0,
            "top2_net40_positive": top2["costs"]["40"]["mean_pct"] > 0.0,
            "top1_uplift_vs_C00_positive": (
                top1["costs"]["40"]["mean_pct"] - c00 > 0.0
            ),
            "top1_uplift_vs_C01_positive": (
                top1["costs"]["40"]["mean_pct"] - c01 > 0.0
            ),
            "familywise_lower_bound_positive": multiplicity["candidates"][
                candidate_id
            ]["simultaneous_90pct_lower_bound_pct"]
            > 0.0,
            "familywise_p_at_most_0_10": multiplicity["candidates"][
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
            "at_least_nine_of_thirteen_months_net40_positive": (
                top1["months"] == 13 and top1["positive_months_net40"] >= 9
            ),
            "at_least_80pct_days_filled": (
                top1["days_with_at_least_one_filled_slot"]
                / top1["costs"]["40"]["days"]
                >= 0.80
            ),
            "top10_selection_share_below_0_50": (
                top1["concentration"]["top10_selection_share"] < 0.50
            ),
            "all_integrity_audits_pass": all_audits_pass,
        }
        output[candidate_id] = {
            "checks": checks,
            "failed_checks": [
                name for name, passes in checks.items() if not passes
            ],
            "passes_all": all(checks.values()),
        }
    return output


def build_report(
    protocol: dict[str, Any],
    result: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    stronger = audit["multiplicity"]["stronger_comparator"]
    stronger_mean = result["metrics"][stronger]["top1"]["costs"]["40"][
        "mean_pct"
    ]
    lines = [
        "# Model v11 — Cross-stock graph and information propagation",
        "",
        "## 結論",
        "",
        f"- 判定: **{audit['decision']['retrospective_decision']}**",
        f"- 本番昇格: **なし**（production gate: {str(audit['decision']['production_gate_passes']).lower()}）",
        f"- familywise比較対照: `{stronger}`（top1 net40 {stronger_mean:.4f}%）",
        f"- PIT対象: {result['coverage']['score_sessions']} sessions / 13 calendar months",
        "",
        "## 事前登録と分離",
        "",
        f"- Protocol SHA-256: `{result['protocol_sha256']}`",
        "- 10 graph hypotheses、共通k=8、trailing 120 sessions、node capacity 1536。",
        "- 閾値・alpha gridなし。Z17の同日peer平均残差targetは再実装していない。",
        "- 外部economic causal-chain、日米sector、news/Twitter sentimentは利用不能のためfail-closed。価格相関を代理の因果関係とは解釈しない。",
        "",
        "## 結果",
        "",
        "| candidate | top1 net40 | top2 net40 | Δ vs C00 | Δ vs C01 | best20除去 net20 | top10 code cash net20 | +months | fill days | simultaneous LB | adj. p | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    c00 = result["metrics"]["C00_daily_rank_ridge"]["top1"]["costs"]["40"][
        "mean_pct"
    ]
    c01 = result["metrics"]["C01_L4_price_control"]["top1"]["costs"]["40"][
        "mean_pct"
    ]
    for candidate_id in GRAPH_IDS:
        top1 = result["metrics"][candidate_id]["top1"]
        top2 = result["metrics"][candidate_id]["top2"]
        boot = audit["multiplicity"]["candidates"][candidate_id]
        gate = audit["candidate_gates"][candidate_id]["passes_all"]
        lines.append(
            f"| `{candidate_id}` | {top1['costs']['40']['mean_pct']:.4f}% | "
            f"{top2['costs']['40']['mean_pct']:.4f}% | "
            f"{top1['costs']['40']['mean_pct'] - c00:+.4f}% | "
            f"{top1['costs']['40']['mean_pct'] - c01:+.4f}% | "
            f"{top1['best20_session_removal']['net20_mean_pct']:.4f}% | "
            f"{top1['top10_code_cash']['net20_mean_pct']:.4f}% | "
            f"{top1['positive_months_net40']}/13 | "
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
    lines.append(
        f"| `Z17_peer_residual_reference` | — | "
        f"{result['known_z17_reference']['top1_net40_mean_pct']:.4f}% | — | "
        f"{result['known_z17_reference']['top2_net40_mean_pct']:.4f}% |"
    )
    best = audit["decision"]["best_point_estimate_candidate"]
    failed = audit["candidate_gates"][best]["failed_checks"]
    lines.extend(
        [
            "",
            "## 独立gate",
            "",
            f"点推定最大は `{best}`（top1 net40 "
            f"{result['metrics'][best]['top1']['costs']['40']['mean_pct']:.4f}%）。",
            "",
            "未達条件: " + ", ".join(f"`{name}`" for name in failed),
            "",
            "## 監査",
            "",
            f"- PIT/source/mutation: **{'PASS' if audit['integrity']['passes'] else 'FAIL'}**",
            f"- 独立P&L再計算: **{'PASS' if audit['pnl_reproduction']['passes'] else 'FAIL'}**",
            f"- v10 C00 top2同値再現: **{'PASS' if audit['known_control_reproduction']['passes'] else 'FAIL'}**",
            f"- byte-identical picks rerun: **{'PASS' if audit['deterministic_reproduction']['picks_byte_identical'] else 'FAIL'}**",
            f"- metrics/folds rerun: **{'PASS' if audit['deterministic_reproduction']['metrics_and_folds_identical'] else 'FAIL'}**",
            "",
            "## 一次資料と限界",
            "",
        ]
    )
    for source in protocol["prior_research"]:
        lines.append(
            f"- [{source['title']}]({source['url']}) — {source['protocol_use']}"
        )
    lines.extend(
        [
            "",
            "本検証のedgeは内生的な価格・overnight・TDnet関係であり、供給網、業種、"
            "sentiment、因果関係の観測ではない。retrospective panelからの判断は"
            "forward-shadow候補までで、本番採用には登録後120件以上のprospective gateと"
            "明示的人手承認が必要。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT / "research" / "model_v11_graph_result.json",
    )
    parser.add_argument(
        "--picks",
        type=Path,
        default=ROOT / "research" / "model_v11_graph_picks.csv",
    )
    parser.add_argument(
        "--repro-result",
        type=Path,
        default=Path("/tmp/model_v11_graph_repro_result.json"),
    )
    parser.add_argument(
        "--repro-picks",
        type=Path,
        default=Path("/tmp/model_v11_graph_repro_picks.csv"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=ROOT / "research" / "model_v11_graph_audit.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "research" / "model_v11_graph_report.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    repro = json.loads(args.repro_result.read_text(encoding="utf-8"))
    v10_architecture = json.loads(
        V10_ARCHITECTURE_RESULT.read_text(encoding="utf-8")
    )
    v10_z17 = json.loads(V10_Z17_RESULT.read_text(encoding="utf-8"))
    picks = pd.read_csv(
        args.picks, parse_dates=["date"], dtype={"code": "object"}
    )
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("protocol changed after registration")
    if result["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("result protocol binding mismatch")
    if sha256_file(args.picks) != result["artifacts"]["picks_sha256"]:
        raise RuntimeError("picks hash mismatch")
    if set(picks["candidate_id"].unique()) != set(
        (*COMPARATOR_IDS, *GRAPH_IDS)
    ):
        raise RuntimeError("unexpected pick candidate ids")

    pnl_failures: list[str] = []
    for candidate_id in (*COMPARATOR_IDS, *GRAPH_IDS):
        for top_k in (1, 2):
            local = picks.loc[
                picks["candidate_id"].eq(candidate_id)
                & picks["top_k"].eq(top_k)
            ].copy()
            expected_rows = result["coverage"]["score_sessions"] * top_k
            if len(local) != expected_rows:
                pnl_failures.append(
                    f"{candidate_id}/top{top_k}: {len(local)} != {expected_rows}"
                )
                continue
            actual = recompute_metrics(local, top_k)
            pnl_failures.extend(
                values_equal(
                    actual,
                    result["metrics"][candidate_id][f"top{top_k}"],
                    f"{candidate_id}/top{top_k}",
                )
            )

    repro_metric_failures = values_equal(
        result["metrics"], repro["metrics"], "repro/metrics"
    )
    repro_fold_failures = values_equal(
        result["folds"], repro["folds"], "repro/folds"
    )
    picks_identical = sha256_file(args.picks) == sha256_file(args.repro_picks)
    control_failures = compare_c00_v10(result, v10_architecture)
    z17_reference_checks = {
        "protocol_id_matches": (
            v10_z17["protocol_id"] == "model_v10_Z17_peer_residual_20260723"
        ),
        "top1_net40_matches": bool(
            np.isclose(
                result["known_z17_reference"]["top1_net40_mean_pct"],
                v10_z17["portfolios"]["top1"]["mean_daily_net_pct"]["40"],
                atol=1e-15,
                rtol=0.0,
            )
        ),
        "top2_net40_matches": bool(
            np.isclose(
                result["known_z17_reference"]["top2_net40_mean_pct"],
                v10_z17["portfolios"]["top2"]["mean_daily_net_pct"]["40"],
                atol=1e-15,
                rtol=0.0,
            )
        ),
        "z17_not_paired_in_familywise": not result["known_z17_reference"][
            "paired_in_familywise_inference"
        ],
    }
    integrity_checks = {
        "protocol_sidecar_matches": (
            PROTOCOL_SHA_PATH.read_text(encoding="utf-8").strip()
            == f"{EXPECTED_PROTOCOL_SHA256}  model_v11_graph_protocol.json"
        ),
        "runner_hash_matches_result": (
            sha256_file(RUNNER_PATH) == result["runner_sha256"]
        ),
        "strictly_prior_all_folds": result["integrity"][
            "strictly_prior_all_folds"
        ],
        "price_source_violations_zero": (
            result["integrity"]["price_source_violations"] == 0
        ),
        "target_mutation_passes": result["integrity"][
            "target_session_outcome_mutation"
        ]["passes"],
        "z17_target_not_reimplemented": not result["integrity"][
            "z17_same_date_peer_residual_target_reimplemented"
        ],
        "external_sources_not_proxied": not result["integrity"][
            "external_unavailable_sources_proxied"
        ],
        "all_four_communities_occupied_each_fold": all(
            set(fold["graph"]["community_counts"]) == {"0", "1", "2", "3"}
            and all(
                count > 0
                for count in fold["graph"]["community_counts"].values()
            )
            for fold in result["folds"]
        ),
    }
    integrity_passes = all(integrity_checks.values())
    pnl_passes = not pnl_failures
    deterministic_passes = (
        picks_identical
        and not repro_metric_failures
        and not repro_fold_failures
    )
    control_passes = not control_failures and all(z17_reference_checks.values())
    all_audits_pass = (
        integrity_passes
        and pnl_passes
        and deterministic_passes
        and control_passes
    )

    multiplicity, stronger = bootstrap_multiplicity(picks)
    gates = candidate_gates(result, multiplicity, all_audits_pass)
    passing = [
        candidate_id
        for candidate_id in GRAPH_IDS
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
    best = max(
        GRAPH_IDS,
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
        "new_120_session_prospective_gate_passed": False,
        "prospective_one_sided_95pct_lower_bound_positive": False,
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
            "passes": integrity_passes,
        },
        "pnl_reproduction": {
            "passes": pnl_passes,
            "failure_count": len(pnl_failures),
            "failures": pnl_failures[:100],
        },
        "known_control_reproduction": {
            "v10_c00_top2_passes": not control_failures,
            "v10_c00_failures": control_failures[:100],
            "z17_reference_checks": z17_reference_checks,
            "passes": control_passes,
        },
        "deterministic_reproduction": {
            "picks_byte_identical": picks_identical,
            "metrics_identical": not repro_metric_failures,
            "folds_identical": not repro_fold_failures,
            "metrics_and_folds_identical": (
                not repro_metric_failures and not repro_fold_failures
            ),
            "metric_failures": repro_metric_failures[:100],
            "fold_failures": repro_fold_failures[:100],
        },
        "multiplicity": multiplicity,
        "candidate_gates": gates,
        "decision": {
            "stronger_comparator": stronger,
            "best_point_estimate_candidate": best,
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
    args.report_output.write_text(
        build_report(protocol, result, audit), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "audit": str(args.audit_output),
                "report": str(args.report_output),
                "retrospective_decision": retrospective_decision,
                "passing_candidate_ids": passing,
                "production_gate_passes": False,
                "best_point_estimate_candidate": best,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
