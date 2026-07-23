#!/usr/bin/env python3
"""Falsify the posthoc momentum-diversification point winner."""

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
import runner as round1  # noqa: E402


VARIANT_IDS = (
    "F00_H12_EXACT",
    "F01_MOMENTUM5_HALF",
    "F02_MOMENTUM60_HALF",
    "F03_OVERNIGHT20_HALF",
    "F04_OCMEAN20_HALF",
    "F05_ATR_HALF",
    "F06_CLOSE_LOCATION_HALF",
    "F07_MOMENTUM20_LEADER75",
    "F08_MOMENTUM20_OPPOSITE75",
    "F09_MOMENTUM20_EXTREME_TERCILE",
    "F10_MOMENTUM20_SIGN",
    "F11_MOMENTUM20_TOP10_OPPOSITE",
    "F12_MOMENTUM20_TOP20_NEUTRAL",
    "F13_MOMENTUM20_EXTREMES_ONLY",
    "F14_MOMENTUM20_RESIDUAL_ONLY",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["ensemble_mean", "code"], ascending=[False, True], kind="stable"
    )


def rows(
    frame: pd.DataFrame,
    indices: list[int],
    weights: list[float],
    variant: str,
) -> pd.DataFrame:
    chosen = frame.loc[indices, [
        "date", "code", "name", "label", "oc_return_pct",
    ]].copy()
    chosen["weight"] = weights
    chosen["policy_id"] = variant
    return chosen


def opposite_half(
    frame: pd.DataFrame,
    feature: str,
    variant: str,
    weights: tuple[float, float] = (0.5, 0.5),
) -> pd.DataFrame:
    order = ordered(frame)
    leader = order.iloc[0]
    median = float(frame[feature].median())
    value = float(leader[feature])
    if not np.isfinite(value):
        candidates = order.iloc[1:]
    elif value >= median:
        candidates = frame.loc[frame[feature].lt(median)]
    else:
        candidates = frame.loc[frame[feature].ge(median)]
    candidates = ordered(candidates.drop(index=leader.name, errors="ignore"))
    if candidates.empty:
        candidates = order.iloc[1:2]
    return rows(
        frame,
        [int(leader.name), int(candidates.index[0])],
        list(weights),
        variant,
    )


def build_variants(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    chunks: dict[str, list[pd.DataFrame]] = {key: [] for key in VARIANT_IDS}
    for _, frame in scores.groupby("date", sort=True):
        frame = frame.copy()
        order = ordered(frame)
        leader = order.iloc[0]
        chunks["F00_H12_EXACT"].append(opposite_half(
            frame, "xrank_close_momentum_20", "F00_H12_EXACT"
        ))
        chunks["F01_MOMENTUM5_HALF"].append(opposite_half(
            frame, "xrank_close_momentum_5", "F01_MOMENTUM5_HALF"
        ))
        chunks["F02_MOMENTUM60_HALF"].append(opposite_half(
            frame, "xrank_close_momentum_60", "F02_MOMENTUM60_HALF"
        ))
        chunks["F03_OVERNIGHT20_HALF"].append(opposite_half(
            frame, "overnight_mean_20", "F03_OVERNIGHT20_HALF"
        ))
        chunks["F04_OCMEAN20_HALF"].append(opposite_half(
            frame, "oc_mean_20", "F04_OCMEAN20_HALF"
        ))
        chunks["F05_ATR_HALF"].append(opposite_half(
            frame, "xrank_atr14_pct", "F05_ATR_HALF"
        ))
        chunks["F06_CLOSE_LOCATION_HALF"].append(opposite_half(
            frame, "xrank_prior_close_location_20", "F06_CLOSE_LOCATION_HALF"
        ))
        chunks["F07_MOMENTUM20_LEADER75"].append(opposite_half(
            frame, "xrank_close_momentum_20",
            "F07_MOMENTUM20_LEADER75", (0.75, 0.25)
        ))
        chunks["F08_MOMENTUM20_OPPOSITE75"].append(opposite_half(
            frame, "xrank_close_momentum_20",
            "F08_MOMENTUM20_OPPOSITE75", (0.25, 0.75)
        ))

        momentum = "xrank_close_momentum_20"
        q33, q67 = frame[momentum].quantile([1 / 3, 2 / 3]).tolist()
        leader_value = float(leader[momentum])
        if leader_value >= float(frame[momentum].median()):
            far = frame.loc[frame[momentum].le(q33)]
        else:
            far = frame.loc[frame[momentum].ge(q67)]
        far = ordered(far.drop(index=leader.name, errors="ignore"))
        if far.empty:
            far = order.iloc[1:2]
        chunks["F09_MOMENTUM20_EXTREME_TERCILE"].append(rows(
            frame, [int(leader.name), int(far.index[0])], [0.5, 0.5],
            "F09_MOMENTUM20_EXTREME_TERCILE",
        ))

        if leader_value >= 0:
            sign_opposite = frame.loc[frame[momentum].lt(0)]
        else:
            sign_opposite = frame.loc[frame[momentum].ge(0)]
        sign_opposite = ordered(sign_opposite.drop(index=leader.name, errors="ignore"))
        if sign_opposite.empty:
            sign_pick = opposite_half(
                frame, momentum, "F10_MOMENTUM20_SIGN"
            )
        else:
            sign_pick = rows(
                frame, [int(leader.name), int(sign_opposite.index[0])],
                [0.5, 0.5], "F10_MOMENTUM20_SIGN",
            )
        chunks["F10_MOMENTUM20_SIGN"].append(sign_pick)

        median = float(frame[momentum].median())
        top10 = order.head(10)
        if leader_value >= median:
            top10_opposite = top10.loc[top10[momentum].lt(median)]
        else:
            top10_opposite = top10.loc[top10[momentum].ge(median)]
        top10_opposite = ordered(
            top10_opposite.drop(index=leader.name, errors="ignore")
        )
        second = (
            top10_opposite.iloc[0] if not top10_opposite.empty else order.iloc[1]
        )
        chunks["F11_MOMENTUM20_TOP10_OPPOSITE"].append(rows(
            frame, [int(leader.name), int(second.name)], [0.5, 0.5],
            "F11_MOMENTUM20_TOP10_OPPOSITE",
        ))

        neutral = order.iloc[1:20].copy()
        neutral["pair_utility"] = (
            neutral["ensemble_mean"]
            - 0.10 * ((leader_value + neutral[momentum]) / 2.0).abs()
        )
        neutral = neutral.sort_values(
            ["pair_utility", "code"], ascending=[False, True], kind="stable"
        )
        chunks["F12_MOMENTUM20_TOP20_NEUTRAL"].append(rows(
            frame, [int(leader.name), int(neutral.index[0])], [0.5, 0.5],
            "F12_MOMENTUM20_TOP20_NEUTRAL",
        ))

        bottom = ordered(frame.loc[frame[momentum].le(q33)])
        top = ordered(frame.loc[frame[momentum].ge(q67)])
        chunks["F13_MOMENTUM20_EXTREMES_ONLY"].append(rows(
            frame, [int(bottom.index[0]), int(top.index[0])], [0.5, 0.5],
            "F13_MOMENTUM20_EXTREMES_ONLY",
        ))

        raw_momentum = frame[momentum].fillna(frame[momentum].median()).fillna(0.0)
        design = np.column_stack([np.ones(len(frame)), raw_momentum.to_numpy(float)])
        target = frame["ensemble_mean"].to_numpy(float)
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        frame["momentum_residual"] = target - design @ coef
        residual = frame.sort_values(
            ["momentum_residual", "code"],
            ascending=[False, True], kind="stable",
        ).head(2)
        chunks["F14_MOMENTUM20_RESIDUAL_ONLY"].append(rows(
            frame, residual.index.tolist(), [0.5, 0.5],
            "F14_MOMENTUM20_RESIDUAL_ONLY",
        ))
    return {
        key: pd.concat(value, ignore_index=True) for key, value in chunks.items()
    }


def family_bootstrap(
    daily: dict[str, pd.DataFrame],
    repetitions: int = 5000,
    seed: int = 20260723,
    block_length: int = 10,
) -> dict[str, Any]:
    reference = daily["F00_H12_EXACT"]["net20"].to_numpy(float)
    variants = list(VARIANT_IDS[1:])
    deltas = np.vstack([
        daily[key]["net20"].to_numpy(float) - reference for key in variants
    ])
    points = deltas.mean(axis=1)
    rng = np.random.default_rng(seed)
    n = len(reference)
    n_blocks = math.ceil(n / block_length)
    bootstrap = np.empty((repetitions, len(variants)))
    for rep in range(repetitions):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([
            (start + np.arange(block_length)) % n for start in starts
        ])[:n]
        bootstrap[rep] = deltas[:, idx].mean(axis=1)
    centered = bootstrap - points
    max_abs_critical = float(np.quantile(np.max(np.abs(centered), axis=1), 0.90))
    return {
        "method": "circular moving-block unstudentized max-absolute-mean",
        "repetitions": repetitions,
        "block_length": block_length,
        "family_size": len(variants),
        "critical90": max_abs_critical,
        "variants": {
            key: {
                "uplift_vs_F00_net20_pct": float(points[idx]),
                "ordinary_two_sided80_interval_pct": [
                    float(np.quantile(bootstrap[:, idx], 0.10)),
                    float(np.quantile(bootstrap[:, idx], 0.90)),
                ],
                "familywise_two_sided80_interval_pct": [
                    float(points[idx] - max_abs_critical),
                    float(points[idx] + max_abs_critical),
                ],
            }
            for idx, key in enumerate(variants)
        },
    }


def main() -> None:
    protocol_path = HERE / "protocol_round2.json"
    scores_path = HERE / "scored_candidates.pkl"
    round1_path = HERE / "result.json"
    output_path = HERE / "result_round2.json"
    report_path = HERE / "report_round2.md"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = [item["id"] for item in protocol["variants"]]
    if expected != list(VARIANT_IDS):
        raise RuntimeError("variant lock mismatch")
    scores = joblib.load(scores_path)
    scores["date"] = pd.to_datetime(scores["date"])
    picks = build_variants(scores)
    sessions = pd.DatetimeIndex(scores["date"].drop_duplicates().sort_values())
    metrics: dict[str, Any] = {}
    daily: dict[str, pd.DataFrame] = {}
    for key in VARIANT_IDS:
        metrics[key], daily[key] = round1.policy_metrics(picks[key], sessions)
    round1_result = json.loads(round1_path.read_text(encoding="utf-8"))
    exact = round1_result["policies"]["H12_MOMENTUM_BUCKET_DIVERSIFICATION"]
    reproduction_diff = max(
        abs(metrics["F00_H12_EXACT"]["net_mean_pct"][str(cost)]
            - exact["net_mean_pct"][str(cost)])
        for cost in (20, 40, 60)
    )
    if reproduction_diff > 1e-12:
        raise RuntimeError("F00 does not reproduce round-one H12")
    bootstrap = family_bootstrap(daily)
    l4 = round1_result["policies"]["C00_L4_TOP2_EQUAL"]
    unrelated = (
        "F03_OVERNIGHT20_HALF", "F04_OCMEAN20_HALF", "F05_ATR_HALF",
        "F06_CLOSE_LOCATION_HALF",
    )
    specificity_wins = sum(
        metrics["F00_H12_EXACT"]["net_mean_pct"]["20"]
        > metrics[key]["net_mean_pct"]["20"]
        for key in unrelated
    )
    alternative_specs = (
        "F07_MOMENTUM20_LEADER75", "F08_MOMENTUM20_OPPOSITE75",
        "F09_MOMENTUM20_EXTREME_TERCILE", "F10_MOMENTUM20_SIGN",
        "F11_MOMENTUM20_TOP10_OPPOSITE", "F12_MOMENTUM20_TOP20_NEUTRAL",
        "F13_MOMENTUM20_EXTREMES_ONLY", "F14_MOMENTUM20_RESIDUAL_ONLY",
    )
    smooth_alternatives = [
        key for key in alternative_specs
        if (
            metrics[key]["net_mean_pct"]["20"] > l4["net_mean_pct"]["20"]
            and metrics[key]["winning_days_removed_net20_pct"]["20"] > 0
        )
    ]
    adjacent = {
        key: metrics[key]["net_mean_pct"]["20"]
        for key in ("F01_MOMENTUM5_HALF", "F02_MOMENTUM60_HALF")
    }
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "authority": protocol["authority"],
        "input": {
            "protocol_sha256": sha256_file(protocol_path),
            "runner_sha256": sha256_file(Path(__file__)),
            "scores_sha256": sha256_file(scores_path),
            "round1_result_sha256": sha256_file(round1_path),
            "score_rows": int(len(scores)),
            "days": int(len(sessions)),
        },
        "exact_reproduction_max_abs_net_cost_diff": reproduction_diff,
        "metrics": metrics,
        "family_bootstrap": bootstrap,
        "mechanism_checks": {
            "specificity_wins_vs_unrelated_splits": specificity_wins,
            "specificity_total_unrelated_splits": len(unrelated),
            "adjacent_horizon_net20_pct": adjacent,
            "smooth_alternatives_above_L4_and_top20_removed_positive": smooth_alternatives,
            "smooth_alternative_count": len(smooth_alternatives),
            "specificity_supported": specificity_wins >= 4,
            "smoothness_supported": len(smooth_alternatives) >= 2,
        },
    }
    result["decision"] = {
        "retain_for_forward_shadow": bool(
            result["mechanism_checks"]["specificity_supported"]
            and result["mechanism_checks"]["smoothness_supported"]
        ),
        "production_change": False,
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    order = sorted(
        VARIANT_IDS,
        key=lambda key: metrics[key]["net_mean_pct"]["20"],
        reverse=True,
    )
    lines = [
        "# Round 2: momentum-diversification falsification",
        "",
        "This round was registered after the H12 point result was visible. "
        "It is a posthoc falsification study, not confirmation.",
        "",
        "| Variant | Net20 | Net40 | Net60 | Months + | Top20 removed | ES5 | Worst day |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in order:
        m = metrics[key]
        lines.append(
            f"| {key} | {m['net_mean_pct']['20']:+.4f}% "
            f"| {m['net_mean_pct']['40']:+.4f}% "
            f"| {m['net_mean_pct']['60']:+.4f}% "
            f"| {m['positive_months']}/{m['months']} "
            f"| {m['winning_days_removed_net20_pct']['20']:+.4f}% "
            f"| {m['expected_shortfall05_day20_pct']:+.4f}% "
            f"| {m['worst_day20_pct']:+.4f}% |"
        )
    checks = result["mechanism_checks"]
    lines.extend([
        "",
        "## Mechanism checks",
        "",
        f"- Specificity wins versus unrelated splits: "
        f"{checks['specificity_wins_vs_unrelated_splits']}/"
        f"{checks['specificity_total_unrelated_splits']}.",
        f"- Adjacent horizons net20: {json.dumps(adjacent, ensure_ascii=False)}.",
        f"- Smooth alternatives beating L4 with positive top-20-removed return: "
        f"{', '.join(smooth_alternatives) if smooth_alternatives else 'none'}.",
        f"- Retain for forward shadow: "
        f"{'yes' if result['decision']['retain_for_forward_shadow'] else 'no'}.",
        "",
        "The reference can be retained only when both specificity and smoothness "
        "checks pass. No retrospective result changes production.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
