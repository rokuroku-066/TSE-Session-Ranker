#!/usr/bin/env python3
"""Replay the preregistered v0.10 online expert ensemble.

The runner consumes only already-created point-in-time expert picks.  For each
score date it constructs the ensemble candidates before using that date's
realised returns to update expert state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/model_v10_online_ensemble_protocol.json"
INPUT = Path("/tmp/v10_model_architectures/picks.csv")
OUTPUT = ROOT / "research/model_v10_online_ensemble_result.json"
PICKS_OUTPUT = ROOT / "research/model_v10_online_ensemble_picks.csv"
HALF_LIFE = 20.0
TEMPERATURE_PCT = 0.50
ALPHA = 1.0 - np.exp(np.log(0.5) / HALF_LIFE)
PERIODS = {
    "discovery": ("2024-07-01", "2024-10-31"),
    "confirmation_a": ("2024-11-01", "2025-03-31"),
    "confirmation_b": ("2025-04-01", "2025-07-31"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expert_day_return(frame: pd.DataFrame, cost_bps: float = 40.0) -> float:
    observed = frame["oc_return_pct"].notna().to_numpy(dtype=float)
    gross = frame["oc_return_pct"].fillna(0.0).to_numpy(dtype=float)
    return float(np.sum(gross - observed * cost_bps / 100.0) / 2.0)


def outcome_lookup(day: pd.DataFrame) -> pd.DataFrame:
    observed = day.loc[
        day["code"].notna(),
        ["code", "name", "label", "oc_return_pct"],
    ].copy()
    observed["code"] = observed["code"].astype(str)
    conflicts = (
        observed.groupby("code", sort=False)["oc_return_pct"]
        .nunique(dropna=False)
        .gt(1)
    )
    if conflicts.any():
        raise ValueError(f"conflicting outcomes for codes: {conflicts[conflicts].index.tolist()}")
    return observed.drop_duplicates("code", keep="first").set_index("code")


def emit_rows(
    *,
    candidate_id: str,
    date: pd.Timestamp,
    codes: list[str],
    lookup: pd.DataFrame,
    scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in (1, 2):
        code = codes[rank - 1] if rank <= len(codes) else None
        if code is not None and code in lookup.index:
            item = lookup.loc[code]
            rows.append({
                "candidate_id": candidate_id,
                "date": date,
                "model_rank": rank,
                "code": code,
                "name": item["name"],
                "label": item["label"],
                "oc_return_pct": item["oc_return_pct"],
                "ensemble_score": None if scores is None else scores.get(code),
            })
        else:
            rows.append({
                "candidate_id": candidate_id,
                "date": date,
                "model_rank": rank,
                "code": None,
                "name": None,
                "label": None,
                "oc_return_pct": None,
                "ensemble_score": None,
            })
    return rows


def vote_candidates(
    day: pd.DataFrame,
    experts: list[str],
    weights: np.ndarray,
) -> tuple[list[str], dict[str, float]]:
    votes: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for expert_index, expert in enumerate(experts):
        frame = day.loc[day["candidate_id"].eq(expert)].sort_values("model_rank")
        for row in frame.itertuples(index=False):
            if pd.isna(row.code):
                continue
            code = str(row.code)
            rank_weight = 1.0 if int(row.model_rank) == 1 else 0.5
            votes[code] = votes.get(code, 0.0) + float(weights[expert_index]) * rank_weight
            best_rank[code] = min(best_rank.get(code, 99), int(row.model_rank))
    ordered = sorted(votes, key=lambda code: (-votes[code], best_rank[code], code))
    return ordered[:2], votes


def simulate(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    experts = sorted(raw["candidate_id"].unique().tolist())
    if len(experts) != 14 or "C00_daily_rank_ridge" not in experts:
        raise ValueError(f"unexpected expert library: {experts}")
    state = np.zeros(len(experts), dtype=float)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for day_index, (date, day) in enumerate(raw.groupby("date", sort=True)):
        lookup = outcome_lookup(day)

        logits = (state - np.max(state)) / TEMPERATURE_PCT
        softmax = np.exp(logits)
        softmax /= np.sum(softmax)
        soft_codes, soft_votes = vote_candidates(day, experts, softmax)
        rows.extend(emit_rows(
            candidate_id="O01_softmax_ewma_vote_hl20",
            date=date,
            codes=soft_codes,
            lookup=lookup,
            scores=soft_votes,
        ))

        equal = np.full(len(experts), 1.0 / len(experts))
        equal_codes, equal_votes = vote_candidates(day, experts, equal)
        rows.extend(emit_rows(
            candidate_id="O02_equal_vote_control",
            date=date,
            codes=equal_codes,
            lookup=lookup,
            scores=equal_votes,
        ))

        if day_index == 0:
            leader = "C00_daily_rank_ridge"
        else:
            leader = min(
                experts,
                key=lambda expert: (-state[experts.index(expert)], expert),
            )
        leader_rows = (
            day.loc[day["candidate_id"].eq(leader)]
            .sort_values("model_rank")
        )
        leader_codes = [
            str(code) for code in leader_rows["code"].tolist() if pd.notna(code)
        ][:2]
        rows.extend(emit_rows(
            candidate_id="O03_follow_leader_hl20",
            date=date,
            codes=leader_codes,
            lookup=lookup,
        ))

        control_rows = (
            day.loc[day["candidate_id"].eq("C00_daily_rank_ridge")]
            .sort_values("model_rank")
        )
        control_codes = [
            str(code) for code in control_rows["code"].tolist() if pd.notna(code)
        ][:2]
        rows.extend(emit_rows(
            candidate_id="C00_daily_rank_ridge",
            date=date,
            codes=control_codes,
            lookup=lookup,
        ))

        # Only after all target-date selections are fixed may today's outcomes
        # enter the state used for the next date.
        today_returns = np.asarray([
            expert_day_return(day.loc[day["candidate_id"].eq(expert)])
            for expert in experts
        ])
        clipped = np.clip(today_returns, -5.0, 5.0)
        state = (1.0 - ALPHA) * state + ALPHA * clipped
        diagnostics.append({
            "date": str(pd.Timestamp(date).date()),
            "leader": leader,
            "largest_softmax_weight": float(np.max(softmax)),
            "effective_experts": float(1.0 / np.sum(np.square(softmax))),
        })

    return pd.DataFrame(rows), {
        "experts": experts,
        "state_update_alpha": float(ALPHA),
        "mean_largest_softmax_weight": float(np.mean([
            item["largest_softmax_weight"] for item in diagnostics
        ])),
        "mean_effective_experts": float(np.mean([
            item["effective_experts"] for item in diagnostics
        ])),
        "leader_counts": pd.Series([
            item["leader"] for item in diagnostics
        ]).value_counts().sort_index().astype(int).to_dict(),
    }


def daily_returns(picks: pd.DataFrame, cost_bps: float) -> pd.Series:
    observed = picks["oc_return_pct"].notna().astype(float)
    slot = picks["oc_return_pct"].fillna(0.0) - observed * cost_bps / 100.0
    return slot.groupby(picks["date"], sort=True).sum().div(2.0)


def summarize(frame: pd.DataFrame, control: pd.DataFrame) -> dict[str, Any]:
    costs = {
        str(cost): float(daily_returns(frame, cost).mean())
        for cost in (20, 40, 60)
    }
    daily20 = daily_returns(frame, 20)
    daily40 = daily_returns(frame, 40)
    best20 = daily20.nlargest(min(20, len(daily20))).index
    tail_removed = float(daily20.drop(best20).mean())
    monthly = (
        daily40.groupby(daily40.index.to_period("M"))
        .mean()
        .rename(index=str)
    )
    slices: dict[str, float] = {}
    for name, (start, end) in PERIODS.items():
        mask = daily20.index.to_series().between(start, end).to_numpy()
        slices[name] = float(daily20.iloc[np.flatnonzero(mask)].mean())

    slot_profit = (
        frame["oc_return_pct"].fillna(0.0)
        - frame["oc_return_pct"].notna().astype(float) * 0.20
    ) / 2.0
    by_code = slot_profit.groupby(frame["code"].fillna("__cash__")).sum()
    top_codes = (
        by_code.drop(labels="__cash__", errors="ignore")
        .sort_values(ascending=False)
        .head(10)
        .index
    )
    retained = frame.loc[~frame["code"].isin(top_codes)].copy()
    removed_code_net20 = float(daily_returns(retained, 20).reindex(daily20.index, fill_value=0.0).mean())

    control_sets = control.groupby("date")["code"].apply(
        lambda values: frozenset(values.dropna().astype(str))
    )
    own_sets = frame.groupby("date")["code"].apply(
        lambda values: frozenset(values.dropna().astype(str))
    )
    overlap = [
        len(own_sets.loc[date] & control_sets.loc[date]) / 2.0
        for date in own_sets.index
    ]
    return {
        "mean_pct": costs,
        "monthly_net40_pct": {str(key): float(value) for key, value in monthly.items()},
        "positive_months_net40": int(monthly.gt(0).sum()),
        "temporal_slices_net20_pct": slices,
        "best_20_days_removed_net20_pct": tail_removed,
        "top_10_profit_codes_to_cash_net20_pct": removed_code_net20,
        "mean_slot_overlap_with_C00": float(np.mean(overlap)),
        "unique_codes": int(frame["code"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--picks-output", type=Path, default=PICKS_OUTPUT)
    args = parser.parse_args()

    raw = pd.read_csv(args.input, parse_dates=["date"], dtype={"code": "string"})
    required = {
        "date", "candidate_id", "model_rank", "code", "name",
        "label", "oc_return_pct",
    }
    if not required.issubset(raw.columns):
        raise ValueError(f"missing columns: {sorted(required - set(raw.columns))}")
    if raw.duplicated(["candidate_id", "date", "model_rank"]).any():
        raise ValueError("duplicate expert/date/rank rows")
    if raw["date"].nunique() != 266:
        raise ValueError("expected 266 score sessions")

    picks, online = simulate(raw)
    candidates = {}
    control = picks.loc[picks["candidate_id"].eq("C00_daily_rank_ridge")]
    for candidate_id, frame in picks.groupby("candidate_id", sort=True):
        candidates[candidate_id] = summarize(frame, control)

    # The implementation's selection order never accesses target-date returns.
    # This structural audit records the exact source lines through the runner
    # hash and verifies that stripping outcomes before selection leaves the
    # equal-vote candidate keys unchanged.
    no_outcome = raw.copy()
    no_outcome[["label", "oc_return_pct"]] = np.nan
    equal_original = picks.loc[
        picks["candidate_id"].eq("O02_equal_vote_control"),
        ["date", "model_rank", "code"],
    ].reset_index(drop=True)
    equal_mutated, _ = simulate(no_outcome)
    equal_mutated = equal_mutated.loc[
        equal_mutated["candidate_id"].eq("O02_equal_vote_control"),
        ["date", "model_rank", "code"],
    ].reset_index(drop=True)
    equal_exact = equal_original.equals(equal_mutated)
    if not equal_exact:
        raise AssertionError("outcome-free equal-vote candidate keys changed")

    args.picks_output.parent.mkdir(parents=True, exist_ok=True)
    picks.to_csv(args.picks_output, index=False)
    result = {
        "schema_version": 1,
        "protocol_id": "model_v10_online_expert_ensemble_20260723",
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(Path(__file__)),
        "input_sha256": sha256_file(args.input),
        "picks_sha256": sha256_file(args.picks_output),
        "score_sessions": int(raw["date"].nunique()),
        "online_diagnostics": online,
        "integrity": {
            "equal_vote_keys_exact_without_outcomes": equal_exact,
            "selection_precedes_same_day_state_update": True,
        },
        "candidates": candidates,
        "production_model_changed": False,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        candidate_id: metrics["mean_pct"]
        for candidate_id, metrics in candidates.items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
