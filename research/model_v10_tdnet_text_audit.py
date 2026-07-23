#!/usr/bin/env python3
"""Deterministic paired moving-block audit for the TDnet title experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path("/tmp/v10_tdnet_text")
SEED = 20260723
REPLICATES = 10_000
BLOCK = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def daily(frame: pd.DataFrame, top_k: int, cost_bps: float = 40.0) -> pd.Series:
    executed = frame["oc_return_pct"].notna()
    slot = (
        frame["oc_return_pct"].fillna(0.0)
        - executed.astype(float) * cost_bps / 100.0
    )
    return slot.groupby(frame["date"], sort=True).sum().div(top_k)


def month_block_indices(
    dates: pd.DatetimeIndex, rng: np.random.Generator
) -> np.ndarray:
    output: list[int] = []
    periods = dates.to_period("M")
    for period in periods.unique():
        local = np.flatnonzero(periods == period)
        n = len(local)
        chosen: list[int] = []
        while len(chosen) < n:
            start = int(rng.integers(0, n))
            chosen.extend(local[(start + np.arange(BLOCK)) % n].tolist())
        output.extend(chosen[:n])
    return np.asarray(output, dtype=int)


def main() -> None:
    picks_path = HERE / "picks.csv"
    picks = pd.read_csv(picks_path, parse_dates=["date"], dtype={"code": str})
    rng = np.random.default_rng(SEED)
    candidate_ids = sorted(
        value
        for value in picks["candidate_id"].unique()
        if value.startswith("T")
    )
    output: dict[str, object] = {
        "schema_version": 1,
        "seed": SEED,
        "replicates": REPLICATES,
        "block_length_sessions_within_month": BLOCK,
        "picks_sha256": sha256_file(picks_path),
        "top_k": {},
    }
    for top_k in (1, 2):
        local = picks.loc[picks["top_k"].eq(top_k)]
        values: dict[str, pd.Series] = {}
        for candidate_id in sorted(local["candidate_id"].unique()):
            values[candidate_id] = daily(
                local.loc[local["candidate_id"].eq(candidate_id)], top_k
            )
        dates = pd.DatetimeIndex(values["L4_price_control"].index)
        base = values["L4_price_control"].reindex(dates).to_numpy()
        matrix = np.column_stack(
            [values[name].reindex(dates).to_numpy() - base for name in candidate_ids]
        )
        if not np.isfinite(matrix).all():
            raise RuntimeError("paired daily matrix contains missing values")
        observed_delta = matrix.mean(axis=0)
        centered = matrix - observed_delta
        candidate_returns = np.column_stack(
            [values[name].reindex(dates).to_numpy() for name in candidate_ids]
        )
        boot_candidate_mean = np.empty((REPLICATES, len(candidate_ids)))
        boot_delta_mean = np.empty_like(boot_candidate_mean)
        max_centered = np.empty(REPLICATES)
        for repeat in range(REPLICATES):
            index = month_block_indices(dates, rng)
            boot_candidate_mean[repeat] = candidate_returns[index].mean(axis=0)
            boot_delta_mean[repeat] = matrix[index].mean(axis=0)
            max_centered[repeat] = centered[index].mean(axis=0).max()
        per_candidate = {}
        for column, candidate_id in enumerate(candidate_ids):
            per_candidate[candidate_id] = {
                "net40_mean_pct": float(candidate_returns[:, column].mean()),
                "net40_mean_90pct_block_interval": [
                    float(np.quantile(boot_candidate_mean[:, column], 0.05)),
                    float(np.quantile(boot_candidate_mean[:, column], 0.95)),
                ],
                "delta_vs_L4_net40_mean_pct": float(observed_delta[column]),
                "delta_vs_L4_90pct_block_interval": [
                    float(np.quantile(boot_delta_mean[:, column], 0.05)),
                    float(np.quantile(boot_delta_mean[:, column], 0.95)),
                ],
            }
        best_column = int(np.argmax(observed_delta))
        best_observed = float(observed_delta[best_column])
        output["top_k"][f"top{top_k}"] = {
            "sessions": int(len(dates)),
            "candidate_family_size": int(len(candidate_ids)),
            "best_candidate_by_delta": candidate_ids[best_column],
            "best_delta_vs_L4_net40_mean_pct": best_observed,
            "familywise_reality_check_p_value": float(
                (1 + np.count_nonzero(max_centered >= best_observed))
                / (REPLICATES + 1)
            ),
            "candidates": per_candidate,
        }
    target = HERE / "audit.json"
    target.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(target)


if __name__ == "__main__":
    main()
