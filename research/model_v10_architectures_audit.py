#!/usr/bin/env python3
"""Independent artifact and P&L audit for the v1.0 architecture experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED = {
    "panel": "6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb",
    "manifest": "25e08c564ef6400b7a29168db9fd7e8220bd0e2386c7a3c7e71b05c2ff950b02",
    "protocol": "bb5224836397070a1945d0226caf1af3f9f2057b9da1265a97a5b1421c82aa7a",
    "runner": "7f51bdd67ba4caafeb2d228676249598c34d7ca5afbab0e55397ac8b1db35c78",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="/tmp/v10_model_architectures/result.json")
    parser.add_argument("--picks", default="/tmp/v10_model_architectures/picks.csv")
    parser.add_argument("--output", default="/tmp/v10_model_architectures/audit.json")
    args = parser.parse_args()
    paths = {
        "panel": Path("/tmp/model_v07_corrected_panel.pkl"),
        "manifest": Path("/tmp/model_v07_corrected_panel.pkl.manifest.json"),
        "protocol": Path("/tmp/v10_model_architectures/protocol.json"),
        "runner": Path("/tmp/v10_model_architectures/run.py"),
        "result": Path(args.result),
        "picks": Path(args.picks),
        "output": Path(args.output),
    }
    actual = {key: sha256_file(paths[key]) for key in EXPECTED}
    if actual != EXPECTED:
        raise ValueError(f"locked input/code hashes changed: {actual}")
    result = load_object(paths["result"])
    picks = pd.read_csv(paths["picks"], parse_dates=["date"])
    if sha256_file(paths["picks"]) != result["artifacts"]["picks_sha256"]:
        raise ValueError("picks hash differs from result")
    if result["protocol_sha256"] != EXPECTED["protocol"]:
        raise ValueError("result protocol hash changed")
    if result["runner_sha256"] != EXPECTED["runner"]:
        raise ValueError("result runner hash changed")
    if result["production_model_changed"] is not False:
        raise ValueError("retrospective result changed production")
    candidate_ids = set(result["metrics"])
    if set(picks["candidate_id"]) != candidate_ids or len(candidate_ids) != 14:
        raise ValueError("candidate set or count changed")
    if picks.duplicated(["candidate_id", "date", "model_rank"]).any():
        raise ValueError("duplicate candidate/date/rank slot")
    slot_counts = picks.groupby(["candidate_id", "date"], sort=False).size()
    if not slot_counts.eq(2).all() or picks["date"].nunique() != 266:
        raise ValueError("scheduled top-two grid is incomplete")
    maximum_difference = 0.0
    recomputed: dict[str, dict[str, float]] = {}
    for candidate_id, frame in picks.groupby("candidate_id", sort=True):
        executed = frame["label"].notna().astype(float)
        recomputed[candidate_id] = {}
        for cost in (20, 40, 60):
            slot = frame["oc_return_pct"].fillna(0.0) - executed * cost / 100.0
            daily = slot.groupby(frame["date"], sort=True).sum().div(2.0)
            value = float(daily.mean())
            expected = float(result["metrics"][candidate_id]["cost"][str(cost)]["mean_pct"])
            maximum_difference = max(maximum_difference, abs(value - expected))
            recomputed[candidate_id][str(cost)] = value
    if maximum_difference > 1e-12:
        raise ValueError("independent P&L recomputation differs from result")
    mutation = result["integrity"]["target_day_outcome_mutation"]
    if len(mutation) != 14 or any(
        value["max_abs_score_difference"] != 0.0
        or value["top2_keys_exact"] is not True
        for value in mutation.values()
    ):
        raise ValueError("target-day mutation audit is incomplete")
    output = {
        "schema_version": 1,
        "ok": True,
        "locked_hashes_exact": True,
        "result_sha256": sha256_file(paths["result"]),
        "picks_sha256": sha256_file(paths["picks"]),
        "candidate_count": len(candidate_ids),
        "score_sessions": int(picks["date"].nunique()),
        "slots_per_candidate": int(len(picks) / len(candidate_ids)),
        "independent_pnl_max_abs_difference": maximum_difference,
        "target_day_outcome_mutation_all_exact": True,
        "production_model_changed": False,
        "recomputed_cost_means": recomputed,
    }
    paths["output"].write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: output[key] for key in (
        "ok",
        "candidate_count",
        "score_sessions",
        "independent_pnl_max_abs_difference",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
