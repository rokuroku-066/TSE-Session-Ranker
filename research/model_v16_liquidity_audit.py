#!/usr/bin/env python3
"""Independently audit the saved v1.6 liquidity selection artifacts.

This script deliberately does not import ``model_v16_liquidity_runner`` or
the project's P&L/bootstrap helpers.  It reconstructs fixed-slot daily
returns, cost haircuts, concentration/tail diagnostics, the paired moving
block bootstrap, every selection gate, and the final no-winner decision from
the saved picks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "research/model_v16_liquidity_protocol.json"
DEFAULT_RESULT = ROOT / "research/model_v16_liquidity_result.json"
DEFAULT_PICKS = ROOT / "research/model_v16_liquidity_picks.csv"
DEFAULT_RUNNER = ROOT / "research/model_v16_liquidity_runner.py"
DEFAULT_OUTPUT = ROOT / "research/model_v16_liquidity_audit.json"
PARSER_SOURCE = ROOT / "src/tse_session_ranker/data/jpx.py"
PRICE_MANIFEST = ROOT / "research/model_v05_input_lock.json"

CONTROL = "C00_PRICE_RIDGE"
SCORE_LEDGER_COLUMNS = (
    "candidate_id",
    "date",
    "model_rank",
    "code",
    "name",
    "model_score",
    "pre_veto_code",
    "vetoed",
)
OUTCOME_COLUMNS = ("label", "oc_return_pct")
NUMERIC_TOLERANCE = 1e-12


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


def warmup_source_bindings_match(
    protocol: dict[str, Any],
    result: dict[str, Any],
    price_manifest: dict[str, Any],
) -> bool:
    """Require the exact registered warm-up source set and byte bindings."""

    expected_names = {
        f"{month}.pdf"
        for month in protocol["source_contract"]["price_warmup"]["months"]
    }
    observed = result["input"]["price_warmup_sources"]
    observed_names = [str(item.get("name", "")) for item in observed]
    if (
        len(observed) != len(expected_names)
        or len(observed_names) != len(set(observed_names))
        or set(observed_names) != expected_names
    ):
        return False
    manifest_by_name = {
        str(item["filename"]): item
        for item in price_manifest["jpx"]["sources"]
        if str(item.get("filename", "")) in expected_names
    }
    if set(manifest_by_name) != expected_names:
        return False
    return all(
        set(item) == {"bytes", "name", "sha256"}
        and int(item["bytes"]) == int(manifest_by_name[item["name"]]["bytes"])
        and str(item["sha256"])
        == str(manifest_by_name[item["name"]]["sha256"])
        for item in observed
    )


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
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    return value


def write_json(value: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
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


class AuditRecorder:
    """Collect auditable predicates and precise discrepancies."""

    def __init__(self) -> None:
        self.checks: dict[str, bool] = {}
        self.discrepancies: list[str] = []
        self.max_abs_numeric_difference = 0.0

    def check(
        self,
        name: str,
        passed: bool,
        *,
        observed: Any | None = None,
        expected: Any | None = None,
    ) -> None:
        self.checks[name] = bool(passed)
        if not passed:
            self.discrepancies.append(
                f"{name}: observed={observed!r}, expected={expected!r}"
            )

    def compare_nested(
        self,
        name: str,
        observed: Any,
        expected: Any,
        *,
        tolerance: float = NUMERIC_TOLERANCE,
    ) -> None:
        before = len(self.discrepancies)
        self._compare_value(name, observed, expected, tolerance=tolerance)
        self.checks[name] = len(self.discrepancies) == before

    def _compare_value(
        self,
        path: str,
        observed: Any,
        expected: Any,
        *,
        tolerance: float,
    ) -> None:
        if isinstance(expected, dict):
            if not isinstance(observed, dict):
                self.discrepancies.append(
                    f"{path}: observed type {type(observed).__name__}, "
                    "expected dict"
                )
                return
            observed_keys = set(observed)
            expected_keys = set(expected)
            if observed_keys != expected_keys:
                self.discrepancies.append(
                    f"{path}: keys observed={sorted(observed_keys)}, "
                    f"expected={sorted(expected_keys)}"
                )
            for key in sorted(observed_keys & expected_keys):
                self._compare_value(
                    f"{path}.{key}",
                    observed[key],
                    expected[key],
                    tolerance=tolerance,
                )
            return
        if isinstance(expected, list):
            if not isinstance(observed, list) or len(observed) != len(expected):
                self.discrepancies.append(
                    f"{path}: observed={observed!r}, expected={expected!r}"
                )
                return
            for index, (left, right) in enumerate(
                zip(observed, expected, strict=True)
            ):
                self._compare_value(
                    f"{path}[{index}]",
                    left,
                    right,
                    tolerance=tolerance,
                )
            return
        if expected is None:
            observed_missing = observed is None or (
                isinstance(observed, (float, np.floating))
                and math.isnan(float(observed))
            )
            if not observed_missing:
                self.discrepancies.append(
                    f"{path}: observed={observed!r}, expected=None"
                )
            return
        numeric_types = (int, float, np.integer, np.floating)
        if (
            isinstance(expected, numeric_types)
            and not isinstance(expected, bool)
            and isinstance(observed, numeric_types)
            and not isinstance(observed, bool)
        ):
            left = float(observed)
            right = float(expected)
            if math.isnan(left) and math.isnan(right):
                return
            difference = abs(left - right)
            if math.isfinite(difference):
                self.max_abs_numeric_difference = max(
                    self.max_abs_numeric_difference, difference
                )
            if not math.isclose(
                left,
                right,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                self.discrepancies.append(
                    f"{path}: observed={left!r}, expected={right!r}, "
                    f"abs_diff={difference!r}"
                )
            return
        if observed != expected:
            self.discrepancies.append(
                f"{path}: observed={observed!r}, expected={expected!r}"
            )


def load_picks(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={
            "candidate_id": str,
            "code": str,
            "name": str,
            "pre_veto_code": str,
        },
        float_precision="round_trip",
    )
    expected = {*SCORE_LEDGER_COLUMNS, *OUTCOME_COLUMNS}
    if set(frame.columns) != expected:
        raise ValueError(
            "v1.6 picks schema differs from the score-ledger plus outcomes: "
            f"{sorted(frame.columns)}"
        )
    frame["date"] = pd.to_datetime(
        frame["date"], errors="coerce"
    ).dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("v1.6 picks contain an invalid date")
    frame["model_rank"] = pd.to_numeric(
        frame["model_rank"], errors="raise"
    ).astype(int)
    for column in ("model_score", "label", "oc_return_pct"):
        raw = frame[column].copy()
        parsed = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & parsed.isna()
        if invalid.any():
            raise ValueError(
                f"v1.6 picks contain a non-numeric {column} value"
            )
        frame[column] = parsed
    if frame["vetoed"].dtype != bool:
        mapped = frame["vetoed"].map(
            {True: True, False: False, "True": True, "False": False}
        )
        if mapped.isna().any():
            raise ValueError("v1.6 picks contain an invalid veto flag")
        frame["vetoed"] = mapped.astype(bool)
    numeric = frame.loc[:, ["model_score", "label", "oc_return_pct"]]
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("v1.6 picks contain an infinite numeric value")
    if not frame["label"].dropna().isin((0.0, 1.0)).all():
        raise ValueError("v1.6 picks contain a non-binary label")
    if not frame["label"].notna().eq(frame["oc_return_pct"].notna()).all():
        raise ValueError("v1.6 label and return presence differ")
    if not frame["code"].notna().eq(frame["name"].notna()).all():
        raise ValueError("v1.6 code and name presence differ")
    if not frame["code"].notna().eq(frame["model_score"].notna()).all():
        raise ValueError("v1.6 selected code and model-score presence differ")
    return frame.sort_values(
        ["candidate_id", "date", "model_rank"], kind="stable"
    ).reset_index(drop=True)


def outcome_contract_matches(picks: pd.DataFrame) -> bool:
    observed = picks["label"].notna()
    returns = picks["oc_return_pct"].notna()
    return bool(
        observed.eq(returns).all()
        and picks.loc[observed, "label"].isin((0.0, 1.0)).all()
        and np.isfinite(
            picks.loc[observed, "oc_return_pct"].to_numpy(dtype=float)
        ).all()
        and picks.loc[observed, "label"]
        .eq(
            picks.loc[observed, "oc_return_pct"]
            .gt(0.0)
            .astype(float)
        )
        .all()
    )


def fixed_slot_contract_matches(picks: pd.DataFrame) -> bool:
    vetoed = picks["vetoed"]
    vetoed_rows = picks.loc[vetoed]
    non_vetoed = picks.loc[~vetoed]
    non_veto_same = (
        (
            non_vetoed["code"].isna()
            & non_vetoed["pre_veto_code"].isna()
        )
        | non_vetoed["code"].eq(non_vetoed["pre_veto_code"])
    )
    return bool(
        vetoed_rows["pre_veto_code"].notna().all()
        and vetoed_rows[
            ["code", "name", "model_score", "label", "oc_return_pct"]
        ]
        .isna()
        .all(axis=None)
        and non_veto_same.all()
    )


def semantic_score_hash(picks: pd.DataFrame) -> str:
    canonical = picks.loc[:, list(SCORE_LEDGER_COLUMNS)].copy()
    canonical["date"] = canonical["date"].dt.strftime("%Y-%m-%d")
    canonical = canonical.sort_values(
        ["candidate_id", "date", "model_rank"], kind="stable"
    ).reset_index(drop=True)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def daily_returns(
    picks: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    capacity: int,
    cost_bps: float,
) -> pd.DataFrame:
    selected = picks.loc[picks["model_rank"].le(capacity)].copy()
    executed = selected["label"].notna()
    selected["_gross_slot"] = selected["oc_return_pct"].fillna(0.0)
    selected["_net_slot"] = selected["_gross_slot"] - (
        executed.astype(float) * (float(cost_bps) / 100.0)
    )
    selected["_executed"] = executed.astype(int)
    daily = selected.groupby("date", sort=True).agg(
        gross_slot_sum=("_gross_slot", "sum"),
        net_slot_sum=("_net_slot", "sum"),
        executed_slots=("_executed", "sum"),
        signal_slots=("model_rank", "size"),
    )
    daily = daily.reindex(scheduled)
    if daily.isna().any(axis=None):
        raise ValueError("independent daily return series is incomplete")
    daily["gross_return_pct"] = daily["gross_slot_sum"] / capacity
    daily["net_return_pct"] = daily["net_slot_sum"] / capacity
    return daily


def profit_factor(values: pd.Series) -> float:
    gains = float(values.clip(lower=0.0).sum())
    losses = float(-values.clip(upper=0.0).sum())
    return gains / losses if losses else float("inf")


def profit_metrics(
    picks: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    capacity: int,
    cost_bps: float,
    sensitivity_costs: Sequence[float],
) -> dict[str, Any]:
    selected = picks.loc[picks["model_rank"].le(capacity)].copy()
    daily = daily_returns(
        picks,
        scheduled,
        capacity=capacity,
        cost_bps=cost_bps,
    )
    executed = selected["label"].notna()
    net = daily["net_return_pct"]
    gross = daily["gross_return_pct"]
    monthly = net.groupby(net.index.to_period("M")).mean()
    equity = (1.0 + net / 100.0).cumprod()
    equity_with_initial = np.concatenate(([1.0], equity.to_numpy()))
    peaks = np.maximum.accumulate(equity_with_initial)
    drawdown = equity_with_initial / peaks - 1.0
    top_days = net.nlargest(min(5, len(net))).index
    without_top5 = net.drop(top_days)
    positive_net = float(net.clip(lower=0.0).sum())
    largest_share = (
        float(net.max() / positive_net) if positive_net > 0.0 else np.nan
    )
    sensitivity = {
        str(float(cost)): float(
            daily_returns(
                picks,
                scheduled,
                capacity=capacity,
                cost_bps=float(cost),
            )["net_return_pct"].mean()
        )
        for cost in sensitivity_costs
    }
    return {
        "n": int(len(selected)),
        "days": int(len(daily)),
        "executed": int(executed.sum()),
        "execution_rate": float(executed.mean()),
        "hit_rate": float(selected.loc[executed, "label"].mean()),
        "signal_hit_rate_including_unfilled": float(
            selected["label"].fillna(0.0).mean()
        ),
        "gross_mean_pct": float(gross.mean()),
        "gross_median_pct": float(gross.median()),
        "net_mean_pct_at_cost": float(net.mean()),
        "net_median_pct_at_cost": float(net.median()),
        "compounded_net_return_pct": float(
            100.0 * (equity.iloc[-1] - 1.0)
        ),
        "max_drawdown_pct": float(100.0 * drawdown.min()),
        "profit_factor": profit_factor(net),
        "positive_months": int(monthly.gt(0.0).sum()),
        "months": int(len(monthly)),
        "worst_month_pct": float(monthly.min()),
        "top5_removed_net_mean_pct": float(without_top5.mean()),
        "largest_day_net_pct": float(net.max()),
        "largest_day_share_of_positive_net": largest_share,
        "monthly_net_mean_pct": {
            str(period): float(value) for period, value in monthly.items()
        },
        "cost_sensitivity_net_mean_pct": sensitivity,
    }


def moving_block_means(
    values: np.ndarray,
    *,
    block_length: int,
    samples: int,
    random_state: int,
    batch_size: int = 1_000,
) -> np.ndarray:
    observations = len(values)
    if block_length < 2 or block_length > observations:
        raise ValueError("invalid moving-block length")
    block_count = math.ceil(observations / block_length)
    max_start = observations - block_length + 1
    offsets = np.arange(block_length, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    output = np.empty(samples, dtype=float)
    position = 0
    while position < samples:
        size = min(batch_size, samples - position)
        starts = rng.integers(0, max_start, size=(size, block_count))
        indices = (starts[..., None] + offsets).reshape(size, -1)
        indices = indices[:, :observations]
        output[position : position + size] = values[indices].mean(axis=1)
        position += size
    return output


def paired_bootstrap(
    candidate: pd.Series,
    baseline: pd.Series,
    *,
    block_length: int,
    samples: int,
    confidence: float,
    random_state: int,
) -> dict[str, Any]:
    if not candidate.index.equals(baseline.index):
        raise ValueError("paired bootstrap indexes differ")
    delta = candidate.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)
    means = moving_block_means(
        delta,
        block_length=block_length,
        samples=samples,
        random_state=random_state,
    )
    tail = (1.0 - confidence) / 2.0
    return {
        "observations": int(len(delta)),
        "block_length": int(block_length),
        "samples": int(samples),
        "confidence": float(confidence),
        "random_state": int(random_state),
        "candidate_mean_pct": float(candidate.mean()),
        "baseline_mean_pct": float(baseline.mean()),
        "point_estimate_delta_pct": float(delta.mean()),
        "one_sided_lower_delta_pct": float(
            np.quantile(means, 1.0 - confidence)
        ),
        "two_sided_lower_delta_pct": float(np.quantile(means, tail)),
        "two_sided_upper_delta_pct": float(
            np.quantile(means, 1.0 - tail)
        ),
        "bootstrap_standard_error_delta_pct": float(means.std(ddof=1)),
    }


def top_codes_cash(
    picks: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    capacity: int,
    cost_bps: float,
    count: int,
) -> tuple[pd.Series, list[str]]:
    selected = picks.loc[picks["model_rank"].le(capacity)].copy()
    executed = selected["label"].notna()
    selected["_net_slot"] = selected["oc_return_pct"].fillna(0.0) - (
        executed.astype(float) * float(cost_bps) / 100.0
    )
    by_code = (
        selected.dropna(subset=["code"])
        .groupby("code", sort=False)["_net_slot"]
        .sum()
        .sort_values(ascending=False, kind="stable")
    )
    codes = [str(code) for code in by_code.head(count).index]
    neutral = picks.copy()
    mask = neutral["model_rank"].le(capacity) & neutral["code"].astype(
        str
    ).isin(codes)
    neutral.loc[mask, ["label", "oc_return_pct"]] = np.nan
    daily = daily_returns(
        neutral,
        scheduled,
        capacity=capacity,
        cost_bps=cost_bps,
    )
    return daily["net_return_pct"], codes


def variant_metrics(
    picks: pd.DataFrame,
    control: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    candidate_id: str,
    capacity: int,
    costs: Sequence[float],
    primary_cost: float,
    slices: dict[str, Sequence[str]],
    bootstrap: dict[str, Any],
    random_state: int,
) -> dict[str, Any]:
    cost_metrics = {
        str(int(cost)): profit_metrics(
            picks,
            scheduled,
            capacity=capacity,
            cost_bps=float(cost),
            sensitivity_costs=(0.0, 10.0, 20.0, 40.0, 60.0),
        )
        for cost in costs
    }
    daily40 = daily_returns(
        picks,
        scheduled,
        capacity=capacity,
        cost_bps=primary_cost,
    )["net_return_pct"]
    daily60 = daily_returns(
        picks,
        scheduled,
        capacity=capacity,
        cost_bps=60.0,
    )["net_return_pct"]
    control40 = daily_returns(
        control,
        scheduled,
        capacity=capacity,
        cost_bps=primary_cost,
    )["net_return_pct"]
    paired = paired_bootstrap(
        daily40,
        control40,
        block_length=int(bootstrap["block_length"]),
        samples=int(bootstrap["samples"]),
        confidence=float(bootstrap["bonferroni_individual_confidence"]),
        random_state=random_state,
    )
    slice_values = {
        name: float(
            daily40.loc[
                pd.Timestamp(bounds[0]) : pd.Timestamp(bounds[1])
            ].mean()
        )
        for name, bounds in slices.items()
    }
    top4_removed = float(
        daily40.drop(daily40.nlargest(4).index).mean()
    )
    code_cash, top_codes = top_codes_cash(
        picks,
        scheduled,
        capacity=capacity,
        cost_bps=primary_cost,
        count=5,
    )
    monthly = daily40.groupby(daily40.index.to_period("M")).mean()
    selected = picks.loc[picks["model_rank"].le(capacity)].copy()
    signaled = selected.dropna(subset=["code"])
    counts = signaled["code"].astype(str).value_counts()
    total_signals = int(len(signaled))
    unique_codes = int(len(counts))
    maximum_share = (
        float(counts.iloc[0] / total_signals) if total_signals else 1.0
    )
    top10_share = (
        float(counts.head(10).sum() / total_signals)
        if total_signals
        else 1.0
    )
    executed = selected["label"].notna()
    executed_days = int(
        selected.assign(_executed=executed.astype(int))
        .groupby("date", sort=True)["_executed"]
        .sum()
        .gt(0)
        .sum()
    )
    executed_fraction = float(
        executed.sum() / (len(scheduled) * capacity)
    )
    checks = {
        "net40_mean_positive": float(daily40.mean()) > 0.0,
        "net40_median_positive": float(daily40.median()) > 0.0,
        "net60_mean_positive": float(daily60.mean()) > 0.0,
        "both_fixed_slices_net40_positive": all(
            value > 0.0 for value in slice_values.values()
        ),
        "positive_months_net40_at_least_3": (
            int(monthly.gt(0.0).sum()) >= 3
        ),
        "top4_days_removed_net40_positive": top4_removed > 0.0,
        "top5_profit_codes_cash_net40_positive": (
            float(code_cash.mean()) > 0.0
        ),
        "familywise_paired_lower_vs_control_nonnegative": (
            float(paired["one_sided_lower_delta_pct"]) >= 0.0
        ),
        "unique_codes_at_least_40": unique_codes >= 40,
        "maximum_code_share_at_most_0_05": maximum_share <= 0.05,
        "top10_code_share_at_most_0_25": top10_share <= 0.25,
        "executed_days_at_least_72": executed_days >= 72,
        "executed_slot_fraction_at_least_0_8": (
            executed_fraction >= 0.80
        ),
    }
    return {
        "variant_id": f"{candidate_id}__top{capacity}",
        "candidate_id": candidate_id,
        "capacity": capacity,
        "cost_metrics": cost_metrics,
        "net40_mean_pct": float(daily40.mean()),
        "net40_median_pct": float(daily40.median()),
        "net60_mean_pct": float(daily60.mean()),
        "slice_net40_mean_pct": slice_values,
        "monthly_net40_mean_pct": {
            str(period): float(value) for period, value in monthly.items()
        },
        "positive_months_net40": int(monthly.gt(0.0).sum()),
        "top4_days_removed_net40_mean_pct": top4_removed,
        "top5_profit_codes_cash_net40_mean_pct": float(code_cash.mean()),
        "top5_profit_codes": top_codes,
        "paired_vs_control_net40": paired,
        "bonferroni_individual_confidence": float(
            bootstrap["bonferroni_individual_confidence"]
        ),
        "unique_selected_codes": unique_codes,
        "maximum_code_selection_share": maximum_share,
        "top10_code_selection_share": top10_share,
        "executed_days": executed_days,
        "executed_slot_fraction": executed_fraction,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def choose_winner(
    variants: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    passers = [item for item in variants if item["gate_passed"]]
    if not passers:
        return None
    winner = sorted(
        passers,
        key=lambda item: (
            -float(
                item["paired_vs_control_net40"][
                    "one_sided_lower_delta_pct"
                ]
            ),
            -float(item["top4_days_removed_net40_mean_pct"]),
            str(item["variant_id"]),
        ),
    )[0]
    return {
        "variant_id": winner["variant_id"],
        "candidate_id": winner["candidate_id"],
        "capacity": winner["capacity"],
        "selection_gate_passed": True,
        "selection_rule_rank": 1,
    }


def audit(
    *,
    protocol_path: Path,
    result_path: Path,
    picks_path: Path,
    runner_path: Path,
) -> dict[str, Any]:
    recorder = AuditRecorder()
    protocol = read_json(protocol_path)
    result = read_json(result_path)
    picks = load_picks(picks_path)

    artifact_hashes = {
        "protocol": sha256_file(protocol_path),
        "result": sha256_file(result_path),
        "picks": sha256_file(picks_path),
        "runner": sha256_file(runner_path),
        "audit_runner": sha256_file(__file__),
        "parser_source": sha256_file(PARSER_SOURCE),
        "price_manifest": sha256_file(PRICE_MANIFEST),
    }
    parser_audit_path = ROOT / protocol["source_contract"][
        "selection_daily"
    ]["parser_audit"]
    artifact_hashes["parser_audit"] = sha256_file(parser_audit_path)

    recorder.check(
        "protocol_hash_matches_result",
        artifact_hashes["protocol"] == result.get("protocol_sha256"),
        observed=artifact_hashes["protocol"],
        expected=result.get("protocol_sha256"),
    )
    recorder.check(
        "runner_hash_matches_result",
        artifact_hashes["runner"] == result.get("runner_sha256"),
        observed=artifact_hashes["runner"],
        expected=result.get("runner_sha256"),
    )
    recorder.check(
        "picks_hash_matches_result",
        artifact_hashes["picks"] == result.get("picks_sha256"),
        observed=artifact_hashes["picks"],
        expected=result.get("picks_sha256"),
    )
    recorder.check(
        "protocol_id_matches",
        protocol.get("protocol_id") == result.get("protocol_id"),
        observed=result.get("protocol_id"),
        expected=protocol.get("protocol_id"),
    )
    source_contract = protocol["source_contract"]["selection_daily"]
    recorder.check(
        "parser_audit_hash_matches_protocol",
        artifact_hashes["parser_audit"]
        == source_contract["parser_audit_sha256"],
        observed=artifact_hashes["parser_audit"],
        expected=source_contract["parser_audit_sha256"],
    )
    recorder.check(
        "parser_source_hash_matches_protocol",
        artifact_hashes["parser_source"]
        == source_contract["required_parser_sha256"],
        observed=artifact_hashes["parser_source"],
        expected=source_contract["required_parser_sha256"],
    )

    price_manifest = read_json(PRICE_MANIFEST)
    recorder.check(
        "warmup_source_hashes_match_price_manifest",
        warmup_source_bindings_match(protocol, result, price_manifest),
    )
    recorder.check(
        "daily_source_contract_matches_result",
        (
            int(result["input"]["daily_source_count"])
            == int(source_contract["files"])
            and int(result["input"]["daily_sessions"])
            == int(source_contract["sessions"])
            and int(result["input"]["daily_parsed_rows"])
            == int(source_contract["required_rows"])
            and int(result["input"]["daily_rejected_rows"])
            == int(source_contract["required_rejected_rows"])
            and result["input"]["parser_version"]
            == source_contract["required_parser_version"]
        ),
        observed={
            "files": result["input"]["daily_source_count"],
            "sessions": result["input"]["daily_sessions"],
            "rows": result["input"]["daily_parsed_rows"],
            "rejected": result["input"]["daily_rejected_rows"],
            "parser": result["input"]["parser_version"],
        },
        expected={
            "files": source_contract["files"],
            "sessions": source_contract["sessions"],
            "rows": source_contract["required_rows"],
            "rejected": source_contract["required_rejected_rows"],
            "parser": source_contract["required_parser_version"],
        },
    )

    candidates = tuple(item["id"] for item in protocol["candidates"])
    all_models = (CONTROL, *candidates)
    capacities = tuple(int(item) for item in protocol["evaluation"]["capacities"])
    costs = tuple(float(item) for item in protocol["evaluation"]["costs_bps"])
    primary_cost = float(protocol["evaluation"]["primary_cost_bps"])
    expected_variants = len(candidates) * len(capacities)
    scheduled = pd.DatetimeIndex(
        sorted(picks["date"].drop_duplicates())
    )
    expected_selection_bounds = protocol["periods"]["locked_selection"]
    recorder.check(
        "selection_calendar_complete",
        (
            len(scheduled)
            == int(protocol["periods"]["locked_selection_sessions"])
            and scheduled.min()
            == pd.Timestamp(expected_selection_bounds[0])
            and scheduled.max()
            == pd.Timestamp(expected_selection_bounds[1])
        ),
        observed={
            "sessions": len(scheduled),
            "bounds": [
                str(scheduled.min().date()),
                str(scheduled.max().date()),
            ],
        },
        expected={
            "sessions": protocol["periods"]["locked_selection_sessions"],
            "bounds": expected_selection_bounds,
        },
    )
    observed_models = tuple(picks["candidate_id"].drop_duplicates())
    recorder.check(
        "picks_model_registry_exact",
        observed_models == all_models,
        observed=observed_models,
        expected=all_models,
    )
    expected_rows = len(all_models) * len(scheduled) * 2
    slot_counts = (
        picks.groupby(["candidate_id", "date"], sort=True)["model_rank"]
        .agg(lambda values: tuple(sorted(values)))
    )
    recorder.check(
        "picks_fixed_slots_complete",
        (
            len(picks) == expected_rows
            and not picks.duplicated(
                ["candidate_id", "date", "model_rank"]
            ).any()
            and slot_counts.map(lambda value: value == (1, 2)).all()
        ),
        observed=len(picks),
        expected=expected_rows,
    )
    recorder.check(
        "outcome_label_matches_return_sign",
        outcome_contract_matches(picks),
    )
    recorder.check(
        "vetoed_slots_are_cash_without_replacement",
        fixed_slot_contract_matches(picks),
    )

    score_hash = semantic_score_hash(picks)
    recorder.check(
        "score_ledger_semantic_hash_matches",
        score_hash
        == result["selection"]["score_ledger_semantic_sha256"],
        observed=score_hash,
        expected=result["selection"]["score_ledger_semantic_sha256"],
    )
    recorder.check(
        "score_ledger_declares_no_outcomes",
        (
            result["selection"]["score_ledger_outcome_columns"] == []
            and result["selection"][
                "score_ledger_hashed_before_outcome_join"
            ]
            is True
        ),
    )

    folds = result["selection"]["folds"]
    fitted_models = (
        CONTROL,
        *(
            item["id"]
            for item in protocol["candidates"]
            if "model" in item
        ),
    )
    selection_months = tuple(
        str(period)
        for period in pd.period_range(
            protocol["periods"]["locked_selection"][0],
            protocol["periods"]["locked_selection"][1],
            freq="M",
        )
    )
    expected_folds = {
        (model, period)
        for model in fitted_models
        for period in selection_months
    }
    observed_folds = [
        (str(item["candidate_id"]), str(item["period"]))
        for item in folds
    ]
    folds_prior = (
        len(observed_folds) == len(expected_folds)
        and len(observed_folds) == len(set(observed_folds))
        and set(observed_folds) == expected_folds
        and all(
            pd.Timestamp(item["train_end"])
            < pd.Period(item["period"], freq="M").start_time
            and item["strictly_prior_training"] is True
            and item["mid_month_refit"] is False
            for item in folds
        )
    )
    recorder.check("all_folds_use_strictly_prior_training", folds_prior)

    result_authority = result["authority"]
    protocol_authority = protocol["authority"]
    no_replay = (
        result_authority["locked_replay_input_opened"] is False
        and result["selection"]["locked_replay_allowed"] is False
        and result["selection"]["winner"] is None
        and result["input"]["daily_date_bounds"][1] <= "2026-03-31"
        and str(picks["date"].max().date()) <= "2026-03-31"
    )
    recorder.check("locked_replay_not_opened", no_replay)
    recorder.check(
        "authority_remains_research_only",
        (
            protocol_authority["project_level_untouched"] is False
            and protocol_authority["production_promotion_allowed"] is False
            and protocol_authority["production_model_changed"] is False
            and protocol_authority["orders_allowed"] is False
            and result_authority["project_level_untouched"] is False
            and result_authority["production_promotion_allowed"] is False
            and result_authority["production_model_changed"] is False
            and result_authority["orders_allowed"] is False
        ),
    )

    picks_by_model = {
        model: picks.loc[picks["candidate_id"].eq(model)].copy()
        for model in all_models
    }
    independent_control = {
        f"top{capacity}": {
            str(int(cost)): profit_metrics(
                picks_by_model[CONTROL],
                scheduled,
                capacity=capacity,
                cost_bps=cost,
                sensitivity_costs=(0.0, 10.0, 20.0, 40.0, 60.0),
            )
            for cost in costs
        }
        for capacity in capacities
    }
    recorder.compare_nested(
        "control_metrics_recomputed",
        independent_control,
        result["selection"]["control_metrics"],
    )

    bootstrap = protocol["evaluation"]["bootstrap"]
    independent_variants: list[dict[str, Any]] = []
    for candidate_index, candidate_id in enumerate(candidates):
        for capacity_index, capacity in enumerate(capacities):
            independent_variants.append(
                variant_metrics(
                    picks_by_model[candidate_id],
                    picks_by_model[CONTROL],
                    scheduled,
                    candidate_id=candidate_id,
                    capacity=capacity,
                    costs=costs,
                    primary_cost=primary_cost,
                    slices=protocol["evaluation"]["selection_slices"],
                    bootstrap=bootstrap,
                    random_state=(
                        int(bootstrap["random_state"])
                        + candidate_index * len(capacities)
                        + capacity_index
                    ),
                )
            )
    recorder.check(
        "candidate_family_size_exact",
        (
            expected_variants
            == int(protocol["evaluation"]["candidate_variants"])
            == int(result["selection"]["candidate_variants"])
            == len(independent_variants)
        ),
        observed=len(independent_variants),
        expected=protocol["evaluation"]["candidate_variants"],
    )
    recorder.compare_nested(
        "variant_metrics_and_gates_recomputed",
        independent_variants,
        result["selection"]["variants"],
    )

    independent_winner = choose_winner(independent_variants)
    independent_gate_passers = sum(
        item["gate_passed"] for item in independent_variants
    )
    independent_status = (
        "selection_passed_one_locked_replay_nominee"
        if independent_winner is not None
        else "selection_rejected_all_candidates"
    )
    independent_decision = {
        "candidate_family_rejected": independent_winner is None,
        "forward_shadow_nominee": (
            None
            if independent_winner is None
            else independent_winner["variant_id"]
        ),
        "production_model_changed": False,
        "orders_allowed": False,
    }
    recorder.compare_nested(
        "winner_rule_recomputed",
        independent_winner,
        result["selection"]["winner"],
    )
    recorder.check(
        "gate_passer_count_recomputed",
        independent_gate_passers == result["selection"]["gate_passers"],
        observed=independent_gate_passers,
        expected=result["selection"]["gate_passers"],
    )
    recorder.check(
        "status_recomputed",
        independent_status == result["status"],
        observed=independent_status,
        expected=result["status"],
    )
    recorder.compare_nested(
        "decision_recomputed",
        independent_decision,
        result["decision"],
    )

    summary_variants = [
        {
            "variant_id": item["variant_id"],
            "net20_mean_pct": item["cost_metrics"]["20"][
                "net_mean_pct_at_cost"
            ],
            "net40_mean_pct": item["net40_mean_pct"],
            "net60_mean_pct": item["net60_mean_pct"],
            "paired_lower_vs_control_net40": item[
                "paired_vs_control_net40"
            ]["one_sided_lower_delta_pct"],
            "gate_passed": item["gate_passed"],
            "failed_gate_count": sum(
                not passed for passed in item["gate_checks"].values()
            ),
        }
        for item in independent_variants
    ]
    passed = not recorder.discrepancies and all(recorder.checks.values())
    return {
        "schema_version": 1,
        "audit_id": "model_v16_liquidity_independent_audit_20260728",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "independence": {
            "selection_runner_imported": False,
            "project_profit_helpers_imported": False,
            "project_bootstrap_helpers_imported": False,
            "recomputed_from": [
                "research/model_v16_liquidity_protocol.json",
                "research/model_v16_liquidity_result.json",
                "research/model_v16_liquidity_picks.csv",
            ],
        },
        "artifact_hashes": artifact_hashes,
        "scope": {
            "daily_fixed_slot_returns": True,
            "costs_bps": list(costs),
            "control_metrics": True,
            "candidate_metrics": True,
            "tail_day_stress": True,
            "profit_code_cash_stress": True,
            "concentration_and_execution": True,
            "paired_moving_block_bootstrap": True,
            "all_selection_gate_checks": True,
            "winner_and_decision": True,
            "authority_and_no_locked_replay": True,
            "feature_reconstruction": False,
            "raw_pdf_reparse": False,
            "same_day_mutation_tests_reexecuted": False,
        },
        "integrity": {
            "models": list(all_models),
            "scheduled_sessions": len(scheduled),
            "picks_rows": len(picks),
            "score_ledger_semantic_sha256": score_hash,
            "observed_outcome_slots": int(picks["label"].notna().sum()),
            "vetoed_slots": int(picks["vetoed"].sum()),
        },
        "recomputed": {
            "variant_summaries": summary_variants,
            "gate_passers": independent_gate_passers,
            "winner": independent_winner,
            "status": independent_status,
            "decision": independent_decision,
            "maximum_absolute_numeric_difference": (
                recorder.max_abs_numeric_difference
            ),
            "numeric_tolerance": NUMERIC_TOLERANCE,
        },
        "authority": {
            "project_level_untouched": False,
            "locked_replay_input_opened": False,
            "production_model_changed": False,
            "production_promotion_allowed": False,
            "orders_allowed": False,
        },
        "checks": recorder.checks,
        "discrepancies": recorder.discrepancies,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--picks", type=Path, default=DEFAULT_PICKS)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(
        protocol_path=args.protocol,
        result_path=args.result,
        picks_path=args.picks,
        runner_path=args.runner,
    )
    write_json(report, args.output)
    if report["status"] != "pass":
        raise SystemExit(
            "v1.6 independent audit failed; see discrepancies in "
            f"{args.output}"
        )
    print(
        "v1.6 independent audit passed: "
        f"{len(report['checks'])} checks, "
        f"{report['recomputed']['gate_passers']} gate passers"
    )


if __name__ == "__main__":
    main()
