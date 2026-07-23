#!/usr/bin/env python3
"""Audit whether H12's incremental value over L4 is robust."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd


HERE = Path("/tmp/v10_policy_universe")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import runner as common  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_pick(
    row: pd.Series,
    date: pd.Timestamp,
    role: str,
    policy: str,
) -> dict[str, Any]:
    return {
        "date": date,
        "code": row["code"],
        "name": row["name"],
        "label": row["label"],
        "oc_return_pct": row["oc_return_pct"],
        "weight": 0.5,
        "role": role,
        "policy_id": policy,
        "ensemble_rank": int(row["ensemble_rank"]),
        "momentum20": row["xrank_close_momentum_20"],
    }


def build_picks(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    h12: list[dict[str, Any]] = []
    control: list[dict[str, Any]] = []
    for date, frame in scores.groupby("date", sort=True):
        order = frame.sort_values(
            ["ensemble_mean", "code"],
            ascending=[False, True], kind="stable",
        ).copy()
        order["ensemble_rank"] = np.arange(1, len(order) + 1)
        leader = order.iloc[0]
        median = float(order["xrank_close_momentum_20"].median())
        if float(leader["xrank_close_momentum_20"]) >= median:
            opposite = order.loc[order["xrank_close_momentum_20"].lt(median)]
        else:
            opposite = order.loc[order["xrank_close_momentum_20"].ge(median)]
        second = opposite.iloc[0]
        h12.extend([
            _row_pick(leader, date, "ensemble_leader", "H12"),
            _row_pick(second, date, "opposite_momentum_half", "H12"),
        ])
        l4 = order.sort_values(
            ["score_L4", "code"], ascending=[False, True], kind="stable"
        ).head(2)
        for rank, (_, row) in enumerate(l4.iterrows(), start=1):
            control.append(_row_pick(row, date, f"L4_rank{rank}", "L4"))
    return pd.DataFrame(h12), pd.DataFrame(control)


def moving_block(
    values: np.ndarray,
    *,
    repetitions: int = 5000,
    block_length: int = 10,
    seed: int = 20260723,
) -> dict[str, Any]:
    n = len(values)
    blocks = math.ceil(n / block_length)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for rep in range(repetitions):
        starts = rng.integers(0, n, size=blocks)
        idx = np.concatenate([
            (start + np.arange(block_length)) % n for start in starts
        ])[:n]
        means[rep] = values[idx].mean()
    return {
        "repetitions": repetitions,
        "block_length": block_length,
        "point": float(values.mean()),
        "one_sided90_lower": float(np.quantile(means, 0.10)),
        "two_sided80_interval": [
            float(np.quantile(means, 0.10)),
            float(np.quantile(means, 0.90)),
        ],
    }


def random_split_null(
    scores: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    dates: list[pd.Timestamp] = []
    leader_net: list[float] = []
    candidate_net: list[np.ndarray] = []
    for date, frame in scores.groupby("date", sort=True):
        order = frame.sort_values(
            ["ensemble_mean", "code"], ascending=[False, True], kind="stable"
        )
        value = (
            order["oc_return_pct"].fillna(0.0).to_numpy(float)
            - order["oc_return_pct"].notna().to_numpy(float) * 0.20
        )
        dates.append(date)
        leader_net.append(float(value[0]))
        candidate_net.append(value[1:])
    rng = np.random.default_rng(seed)
    geometric = rng.geometric(0.5, size=(repetitions, len(dates)))
    simulated = np.empty((repetitions, len(dates)), dtype=float)
    for date_idx, candidates in enumerate(candidate_net):
        selected = np.minimum(geometric[:, date_idx] - 1, len(candidates) - 1)
        simulated[:, date_idx] = (
            0.5 * leader_net[date_idx] + 0.5 * candidates[selected]
        )
    means = simulated.mean(axis=1)
    return {
        "repetitions": repetitions,
        "seed": seed,
        "null_mean_net20_pct": float(means.mean()),
        "null_p05_net20_pct": float(np.quantile(means, 0.05)),
        "null_p95_net20_pct": float(np.quantile(means, 0.95)),
        "simulated_means": means,
    }


def main() -> None:
    protocol_path = HERE / "protocol_round3.json"
    scores_path = HERE / "scored_candidates.pkl"
    round1_path = HERE / "result.json"
    output_path = HERE / "result_round3.json"
    report_path = HERE / "report_round3.md"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    scores = joblib.load(scores_path)
    scores["date"] = pd.to_datetime(scores["date"])
    sessions = pd.DatetimeIndex(scores["date"].drop_duplicates().sort_values())
    h12, control = build_picks(scores)

    h_metrics, h_daily_export = common.policy_metrics(h12, sessions)
    c_metrics, c_daily_export = common.policy_metrics(control, sessions)
    h_daily = h_daily_export.set_index("date")
    c_daily = c_daily_export.set_index("date")
    deltas = {
        str(cost): h_daily[f"net{cost}"] - c_daily[f"net{cost}"]
        for cost in (20, 40, 60)
    }
    delta20 = deltas["20"]
    monthly = delta20.groupby(delta20.index.to_period("M")).mean()
    block_positions = np.array_split(np.arange(len(delta20)), 4)
    blocks = {
        f"B{idx + 1}": float(delta20.iloc[position].mean())
        for idx, position in enumerate(block_positions)
    }
    removal = {
        str(count): float(delta20.drop(delta20.nlargest(count).index).mean())
        for count in (5, 10, 20)
    }
    positive = delta20.clip(lower=0)
    concentration = {
        str(count): float(delta20.nlargest(count).sum() / positive.sum())
        for count in (5, 10, 20)
    }
    winsorized = {}
    for lower, upper, key in ((0.01, 0.99, "1_99"), (0.05, 0.95, "5_95")):
        lo, hi = delta20.quantile([lower, upper])
        winsorized[key] = float(delta20.clip(lo, hi).mean())
    block_bootstrap = moving_block(delta20.to_numpy(float))

    # Incremental code contribution. Removing a code leaves its fixed sleeve in cash.
    def net_contribution(frame: pd.DataFrame) -> pd.Series:
        observed = frame["oc_return_pct"].notna().astype(float)
        return (
            frame["weight"]
            * (frame["oc_return_pct"].fillna(0.0) - observed * 0.20)
        )
    h_code = h12.assign(contribution=net_contribution(h12)).groupby(
        "code"
    )["contribution"].sum()
    c_code = control.assign(contribution=net_contribution(control)).groupby(
        "code"
    )["contribution"].sum()
    code_increment = h_code.sub(c_code, fill_value=0.0).sort_values(ascending=False)
    base_uplift = float(delta20.mean())
    leave_code_out = (base_uplift - code_increment / len(sessions)).sort_values()

    h_sets = h12.groupby("date")["code"].agg(set)
    c_sets = control.groupby("date")["code"].agg(set)
    overlap = pd.Series({
        date: len(h_sets[date] & c_sets[date]) for date in sessions
    })
    leg_metrics = {}
    for role, frame in h12.groupby("role"):
        observed = frame["oc_return_pct"].notna()
        leg_metrics[role] = {
            "rows": int(len(frame)),
            "executed": int(observed.sum()),
            "gross_mean_executed_pct": float(frame.loc[observed, "oc_return_pct"].mean()),
            "gross_median_executed_pct": float(frame.loc[observed, "oc_return_pct"].median()),
            "hit_rate_executed": float(frame.loc[observed, "label"].mean()),
            "mean_ensemble_rank": float(frame["ensemble_rank"].mean()),
            "median_ensemble_rank": float(frame["ensemble_rank"].median()),
        }

    null_config = protocol["random_null"]
    null = random_split_null(
        scores,
        repetitions=int(null_config["repetitions"]),
        seed=int(null_config["seed"]),
    )
    observed_h12 = float(h_metrics["net_mean_pct"]["20"])
    null_means = null.pop("simulated_means")
    null["observed_H12_net20_pct"] = observed_h12
    null["one_sided_p_null_at_least_observed"] = float(
        (np.sum(null_means >= observed_h12) + 1) / (len(null_means) + 1)
    )

    checks = {
        "top20_positive_uplift_days_removed_mean_positive": removal["20"] > 0,
        "at_least_8_of_13_monthly_uplifts_positive": int(monthly.gt(0).sum()) >= 8,
        "all_four_blocks_nonnegative": all(value >= 0 for value in blocks.values()),
        "winsor_1_99_uplift_positive": winsorized["1_99"] > 0,
        "ordinary_block_one_sided90_lower_positive": (
            block_bootstrap["one_sided90_lower"] > 0
        ),
        "random_split_one_sided_p_below_0_10": (
            null["one_sided_p_null_at_least_observed"] < 0.10
        ),
        "expected_shortfall_not_worse_than_control": (
            h_metrics["expected_shortfall05_day20_pct"]
            >= c_metrics["expected_shortfall05_day20_pct"]
        ),
    }
    checks["all"] = all(checks.values())
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "authority": protocol["authority"],
        "input": {
            "protocol_sha256": sha256_file(protocol_path),
            "runner_sha256": sha256_file(Path(__file__)),
            "scores_sha256": sha256_file(scores_path),
            "round1_result_sha256": sha256_file(round1_path),
            "days": int(len(sessions)),
        },
        "strategy_metrics": {
            "H12": h_metrics,
            "L4_control": c_metrics,
        },
        "matched_incremental": {
            "mean_uplift_pct": {
                cost: float(values.mean()) for cost, values in deltas.items()
            },
            "median_uplift_net20_pct": float(delta20.median()),
            "positive_uplift_day_rate": float(delta20.gt(0).mean()),
            "zero_uplift_day_rate": float(delta20.eq(0).mean()),
            "monthly_uplift_pct": {
                str(period): float(value) for period, value in monthly.items()
            },
            "positive_uplift_months": int(monthly.gt(0).sum()),
            "four_blocks_uplift_pct": blocks,
            "winning_uplift_days_removed_mean_pct": removal,
            "top_positive_uplift_share": concentration,
            "winsorized_uplift_pct": winsorized,
            "moving_block": block_bootstrap,
        },
        "selection_diagnostics": {
            "overlap_days": {
                str(count): int(overlap.eq(count).sum()) for count in (0, 1, 2)
            },
            "leg_metrics": leg_metrics,
            "largest_positive_increment_codes": [
                {"code": str(code), "total_increment": float(value)}
                for code, value in code_increment.head(10).items()
            ],
            "lowest_leave_one_code_out_mean_uplift": [
                {"code": str(code), "mean_uplift": float(value)}
                for code, value in leave_code_out.head(10).items()
            ],
        },
        "random_binary_split_null": null,
        "support_checks": checks,
        "decision": {
            "robust_replacement_supported": checks["all"],
            "production_change": False,
            "forward_status": (
                "primary_shadow_eligible" if checks["all"]
                else "research_arm_only"
            ),
        },
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Round 3: H12 incremental concentration audit",
        "",
        "The relevant question is whether H12 adds robust value over L4, not merely "
        "whether H12 is positive in isolation.",
        "",
        "## Matched uplift",
        "",
        f"- Mean uplift net20: {delta20.mean():+.4f}%.",
        f"- Median uplift net20: {delta20.median():+.4f}%.",
        f"- Positive/zero uplift days: {delta20.gt(0).mean():.1%} / "
        f"{delta20.eq(0).mean():.1%}.",
        f"- Positive uplift months: {monthly.gt(0).sum()}/{len(monthly)}.",
        f"- Four blocks: {json.dumps(blocks)}.",
        f"- After top 5/10/20 positive uplift days: "
        f"{removal['5']:+.4f}% / {removal['10']:+.4f}% / {removal['20']:+.4f}%.",
        f"- Top 5/10/20 share of all positive uplift: "
        f"{concentration['5']:.1%} / {concentration['10']:.1%} / "
        f"{concentration['20']:.1%}.",
        f"- Moving-block one-sided 90% lower: "
        f"{block_bootstrap['one_sided90_lower']:+.4f}%.",
        "",
        "## Random-split falsification",
        "",
        f"- Random binary-split null mean: {null['null_mean_net20_pct']:+.4f}%.",
        f"- Observed H12: {observed_h12:+.4f}%.",
        f"- One-sided null p-value: "
        f"{null['one_sided_p_null_at_least_observed']:.4f}.",
        "",
        "## Tail comparison",
        "",
        f"- H12 ES5 / worst day: "
        f"{h_metrics['expected_shortfall05_day20_pct']:+.4f}% / "
        f"{h_metrics['worst_day20_pct']:+.4f}%.",
        f"- L4 ES5 / worst day: "
        f"{c_metrics['expected_shortfall05_day20_pct']:+.4f}% / "
        f"{c_metrics['worst_day20_pct']:+.4f}%.",
        "",
        "## Fixed support checks",
        "",
    ]
    for key, value in checks.items():
        if key != "all":
            lines.append(f"- {key}: {'pass' if value else 'fail'}")
    lines.extend([
        "",
        f"Overall: {'pass' if checks['all'] else 'fail'}.",
        "",
        "Failure does not prove that momentum diversification has no effect. It "
        "means the already-viewed panel does not support replacing L4. The idea "
        "can remain only as a separately labelled forward research arm.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
