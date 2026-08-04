#!/usr/bin/env python3
"""Independently audit the saved v1.7 liquidity-reliability artifacts.

This module intentionally does not import the v1.7 runner or the project's
profit/bootstrap helpers.  It validates the frozen source and artifact
bindings, reconstructs fixed-slot daily returns and cost/tail diagnostics,
repeats the paired moving-block bootstrap, evaluates every selection gate,
and derives the winner and research-only decision from persisted artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "research/model_v17_liquidity_reliability_protocol.json"
DEFAULT_RESULT = ROOT / "research/model_v17_liquidity_reliability_result.json"
DEFAULT_SCORES = ROOT / "research/model_v17_liquidity_reliability_scores.csv"
DEFAULT_PICKS = ROOT / "research/model_v17_liquidity_reliability_picks.csv"
DEFAULT_RUNNER = ROOT / "research/model_v17_liquidity_reliability_runner.py"
DEFAULT_OUTPUT = ROOT / "research/model_v17_liquidity_reliability_audit.json"
INPUT_LOCK = ROOT / "research/model_v17_replay_input_lock.json"
PARSER_SOURCE = ROOT / "src/tse_session_ranker/data/jpx.py"
PRICE_MANIFEST = ROOT / "research/model_v05_input_lock.json"
LEGACY_PARSER_AUDIT = ROOT / "research/model_v04_parser_recovery_audit.json"

PROTOCOL_SHA256 = "f7d2efa30c5f6ca03a95e1f6e84e0fb6de3f68e8f0d520183877e2f0ab4a416f"
INPUT_LOCK_SHA256 = "1d9a391c8e6b8d2003498c18ba09dac904e672e1f17c05516998ffb71e11c475"
INPUT_SOURCE_SET_SHA256 = "a0fdea19f9e7c771d2e0450720eb551917d9c8b9cfade75260d5cc43adb3a43f"
PARSER_SHA256 = "1bd2e74acced608eb36c3606b593ea407d2d1e5f54b3283790ef8fd0fb1041f7"
PROTOCOL_ID = "model_v17_liquidity_reliability_selection_20260804"
PARSER_VERSION = "jpx_daily_text_v6_special_quote_marker"

C00 = "C00_PRICE_RIDGE_TOP1"
PAIR00 = "PAIR00_PRICE_ONLY_TOP2_RERANK"
C02 = "C02_C00_RANK2"
CANDIDATES = (
    "PAIR01_TOP2_EXACT_LIQ_RERANK",
    "TW02_TURNOVER_WEIGHTED_PRICE_MEMORY",
    "VW03_AGGREGATE_COST_BASIS",
    "RPY04_RETURN_PER_TURNOVER_REVERSAL",
    "AR05_SECURITY_ACTIVITY_REGIME_EXPERTS",
    "PS06_PERSISTENT_VS_ISOLATED_ACTIVITY",
    "VP07_VWAP_RANGE_PRESSURE_MEMORY",
    "RW08_EXACT_LOT_RELIABILITY_WEIGHT",
    "MR09_TURNOVER_WEIGHTED_MARKET_REGIME",
    "AT10_ATTENTION_MIGRATION",
)
ALL_MODELS = (C00, PAIR00, C02, *CANDIDATES)
MATCHED_CONTROL = {
    candidate: PAIR00 if candidate.startswith("PAIR01_") else C00
    for candidate in CANDIDATES
}
SCORE_LEDGER_COLUMNS = (
    "candidate_id",
    "date",
    "model_rank",
    "code",
    "name",
    "model_score",
    "pre_veto_code",
    "vetoed",
    "source_rank",
)
OUTCOME_COLUMNS = ("label", "oc_return_pct")
SELECTION_START = pd.Timestamp("2026-04-01")
SELECTION_END = pd.Timestamp("2026-07-27")
SELECTION_SESSIONS = 79
SELECTION_SLICES = {
    "replay_a": (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-29")),
    "replay_b": (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-27")),
}
COSTS_BPS = (20.0, 40.0, 60.0)
PRIMARY_COST_BPS = 40.0
BOOTSTRAP_BLOCK_LENGTH = 5
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_RANDOM_STATE = 20_260_804
BOOTSTRAP_CONFIDENCE = 0.99
NUMERIC_TOLERANCE = 1e-12
_DAILY_NAME = re.compile(r"^stq_(\d{8})\.pdf$")


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
    """Collect named predicates and exact nested discrepancies."""

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

    def compare_nested(self, name: str, observed: Any, expected: Any) -> None:
        before = len(self.discrepancies)
        self._compare(name, json_safe(observed), json_safe(expected))
        self.checks[name] = len(self.discrepancies) == before

    def _compare(self, path: str, observed: Any, expected: Any) -> None:
        if isinstance(expected, dict):
            if not isinstance(observed, dict):
                self.discrepancies.append(
                    f"{path}: observed type={type(observed).__name__}, expected=dict"
                )
                return
            if set(observed) != set(expected):
                self.discrepancies.append(
                    f"{path}: keys observed={sorted(observed)}, expected={sorted(expected)}"
                )
            for key in sorted(set(observed) & set(expected)):
                self._compare(f"{path}.{key}", observed[key], expected[key])
            return
        if isinstance(expected, list):
            if not isinstance(observed, list) or len(observed) != len(expected):
                self.discrepancies.append(
                    f"{path}: observed={observed!r}, expected={expected!r}"
                )
                return
            for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
                self._compare(f"{path}[{index}]", left, right)
            return
        if expected is None:
            if observed is not None:
                self.discrepancies.append(
                    f"{path}: observed={observed!r}, expected=None"
                )
            return
        numeric = (int, float, np.integer, np.floating)
        if (
            isinstance(expected, numeric)
            and not isinstance(expected, bool)
            and isinstance(observed, numeric)
            and not isinstance(observed, bool)
        ):
            left = float(observed)
            right = float(expected)
            difference = abs(left - right)
            if math.isfinite(difference):
                self.max_abs_numeric_difference = max(
                    self.max_abs_numeric_difference, difference
                )
            if not math.isclose(
                left, right, rel_tol=0.0, abs_tol=NUMERIC_TOLERANCE
            ):
                self.discrepancies.append(
                    f"{path}: observed={left!r}, expected={right!r}, abs_diff={difference!r}"
                )
            return
        if observed != expected:
            self.discrepancies.append(
                f"{path}: observed={observed!r}, expected={expected!r}"
            )


def _load_ledger(path: str | Path, *, outcomes: bool) -> pd.DataFrame:
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
    expected = set(SCORE_LEDGER_COLUMNS) | (set(OUTCOME_COLUMNS) if outcomes else set())
    if set(frame.columns) != expected:
        raise ValueError(
            f"v1.7 {'picks' if outcomes else 'scores'} schema changed: "
            f"{sorted(frame.columns)}"
        )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("v1.7 ledger contains an invalid date")
    frame["model_rank"] = pd.to_numeric(frame["model_rank"], errors="raise").astype(int)
    numeric_columns = ["model_score", "source_rank"]
    if outcomes:
        numeric_columns.extend(OUTCOME_COLUMNS)
    for column in numeric_columns:
        raw = frame[column].copy()
        parsed = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & parsed.isna()).any():
            raise ValueError(f"v1.7 ledger contains non-numeric {column}")
        frame[column] = parsed
    mapped = frame["vetoed"].map(
        {True: True, False: False, "True": True, "False": False}
    )
    if mapped.isna().any():
        raise ValueError("v1.7 ledger contains an invalid vetoed value")
    frame["vetoed"] = mapped.astype(bool)
    if np.isinf(frame[numeric_columns].to_numpy(dtype=float)).any():
        raise ValueError("v1.7 ledger contains an infinite numeric value")
    if outcomes:
        if not frame["label"].dropna().isin((0.0, 1.0)).all():
            raise ValueError("v1.7 picks contain a non-binary label")
        if not frame["label"].notna().eq(frame["oc_return_pct"].notna()).all():
            raise ValueError("v1.7 outcome presence differs")
    if not frame["code"].notna().eq(frame["name"].notna()).all():
        raise ValueError("v1.7 code/name presence differs")
    if not frame["code"].notna().eq(frame["model_score"].notna()).all():
        raise ValueError("v1.7 code/score presence differs")
    return frame.sort_values(["candidate_id", "date"], kind="stable").reset_index(drop=True)


def load_scores(path: str | Path) -> pd.DataFrame:
    return _load_ledger(path, outcomes=False)


def load_picks(path: str | Path) -> pd.DataFrame:
    return _load_ledger(path, outcomes=True)


def semantic_score_hash(frame: pd.DataFrame) -> str:
    canonical = frame.loc[:, list(SCORE_LEDGER_COLUMNS)].copy()
    canonical["date"] = pd.to_datetime(canonical["date"]).dt.strftime("%Y-%m-%d")
    canonical = canonical.sort_values(["candidate_id", "date"], kind="stable")
    return hashlib.sha256(
        canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _ledger_values_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    left = left.loc[:, list(SCORE_LEDGER_COLUMNS)].reset_index(drop=True)
    right = right.loc[:, list(SCORE_LEDGER_COLUMNS)].reset_index(drop=True)
    if left.shape != right.shape:
        return False
    for column in SCORE_LEDGER_COLUMNS:
        if column in {"model_score", "source_rank"}:
            a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(a, b, atol=0.0, rtol=0.0, equal_nan=True):
                return False
        elif column == "date":
            if not pd.to_datetime(left[column]).equals(pd.to_datetime(right[column])):
                return False
        else:
            a = left[column].astype("object").where(left[column].notna(), "<NA>")
            b = right[column].astype("object").where(right[column].notna(), "<NA>")
            if not a.equals(b):
                return False
    return True


def outcome_contract_matches(picks: pd.DataFrame) -> bool:
    observed = picks["label"].notna()
    return bool(
        observed.eq(picks["oc_return_pct"].notna()).all()
        and np.isfinite(picks.loc[observed, "oc_return_pct"].to_numpy(dtype=float)).all()
        and picks.loc[observed, "label"]
        .eq(picks.loc[observed, "oc_return_pct"].gt(0.0).astype(float))
        .all()
    )


def fixed_slot_contract_matches(picks: pd.DataFrame) -> bool:
    same_pre_veto = (
        (picks["code"].isna() & picks["pre_veto_code"].isna())
        | picks["code"].eq(picks["pre_veto_code"])
    )
    return bool(
        not picks["vetoed"].any()
        and same_pre_veto.all()
        and picks["model_rank"].eq(1).all()
    )


def daily_returns(
    picks: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    cost_bps: float,
) -> pd.DataFrame:
    if picks["date"].duplicated().any() or not picks["model_rank"].eq(1).all():
        raise ValueError("v1.7 independent return input is not one slot per date")
    selected = picks.set_index("date").reindex(scheduled)
    if selected["model_rank"].isna().any():
        raise ValueError("v1.7 independent return schedule is incomplete")
    executed = selected["label"].notna()
    gross = selected["oc_return_pct"].fillna(0.0).astype(float)
    net = gross - executed.astype(float) * float(cost_bps) / 100.0
    return pd.DataFrame(
        {
            "gross_return_pct": gross,
            "net_return_pct": net,
            "executed_slots": executed.astype(int),
            "signal_slots": 1,
        },
        index=scheduled,
    )


def _profit_factor(values: pd.Series) -> float:
    gains = float(values.clip(lower=0.0).sum())
    losses = float(-values.clip(upper=0.0).sum())
    return gains / losses if losses else float("inf")


def profit_metrics(
    picks: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    cost_bps: float,
    sensitivity_costs: Sequence[float] = (0.0, 10.0, 20.0, 40.0, 60.0),
) -> dict[str, Any]:
    daily = daily_returns(picks, scheduled, cost_bps=cost_bps)
    executed = picks["label"].notna()
    net = daily["net_return_pct"]
    gross = daily["gross_return_pct"]
    monthly = net.groupby(net.index.to_period("M")).mean()
    equity = (1.0 + net / 100.0).cumprod()
    equity_initial = np.concatenate(([1.0], equity.to_numpy()))
    peaks = np.maximum.accumulate(equity_initial)
    drawdown = equity_initial / peaks - 1.0
    without_top5 = net.drop(net.nlargest(min(5, len(net))).index)
    positive_net = float(net.clip(lower=0.0).sum())
    largest_share = float(net.max() / positive_net) if positive_net > 0.0 else np.nan
    sensitivity = {
        str(float(cost)): float(
            daily_returns(picks, scheduled, cost_bps=cost)["net_return_pct"].mean()
        )
        for cost in sensitivity_costs
    }
    return {
        "n": int(len(picks)),
        "days": int(len(daily)),
        "executed": int(executed.sum()),
        "execution_rate": float(executed.mean()),
        "hit_rate": float(picks.loc[executed, "label"].mean()),
        "signal_hit_rate_including_unfilled": float(picks["label"].fillna(0.0).mean()),
        "gross_mean_pct": float(gross.mean()),
        "gross_median_pct": float(gross.median()),
        "net_mean_pct_at_cost": float(net.mean()),
        "net_median_pct_at_cost": float(net.median()),
        "compounded_net_return_pct": float(100.0 * (equity.iloc[-1] - 1.0)),
        "max_drawdown_pct": float(100.0 * drawdown.min()),
        "profit_factor": _profit_factor(net),
        "positive_months": int(monthly.gt(0.0).sum()),
        "months": int(len(monthly)),
        "worst_month_pct": float(monthly.min()),
        "top5_removed_net_mean_pct": (
            float(without_top5.mean()) if len(without_top5) else np.nan
        ),
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
    if block_length < 2 or block_length > observations or samples < 100:
        raise ValueError("invalid moving-block bootstrap parameters")
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
        output[position : position + size] = values[indices[:, :observations]].mean(axis=1)
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
        "one_sided_lower_delta_pct": float(np.quantile(means, 1.0 - confidence)),
        "two_sided_lower_delta_pct": float(np.quantile(means, tail)),
        "two_sided_upper_delta_pct": float(np.quantile(means, 1.0 - tail)),
        "bootstrap_standard_error_delta_pct": float(means.std(ddof=1)),
    }


def top_codes_cash(
    picks: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    cost_bps: float,
    count: int,
) -> tuple[pd.Series, list[str]]:
    work = picks.copy()
    executed = work["label"].notna()
    work["_net"] = work["oc_return_pct"].fillna(0.0) - (
        executed.astype(float) * float(cost_bps) / 100.0
    )
    by_code = (
        work.dropna(subset=["code"])
        .groupby("code", sort=False)["_net"]
        .sum()
        .sort_values(ascending=False, kind="stable")
    )
    codes = [str(code) for code in by_code.head(count).index]
    neutral = work.copy()
    removed = neutral["code"].astype(str).isin(codes)
    neutral.loc[removed, ["label", "oc_return_pct"]] = np.nan
    return (
        daily_returns(neutral, scheduled, cost_bps=cost_bps)["net_return_pct"],
        codes,
    )


def variant_metrics(
    picks: pd.DataFrame,
    control: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    candidate_id: str,
    control_id: str,
    random_state: int,
) -> dict[str, Any]:
    cost_metrics = {
        str(int(cost)): profit_metrics(picks, scheduled, cost_bps=cost)
        for cost in COSTS_BPS
    }
    daily40 = daily_returns(picks, scheduled, cost_bps=PRIMARY_COST_BPS)[
        "net_return_pct"
    ]
    daily60 = daily_returns(picks, scheduled, cost_bps=60.0)["net_return_pct"]
    control40 = daily_returns(control, scheduled, cost_bps=PRIMARY_COST_BPS)[
        "net_return_pct"
    ]
    paired = paired_bootstrap(
        daily40,
        control40,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        samples=BOOTSTRAP_SAMPLES,
        confidence=BOOTSTRAP_CONFIDENCE,
        random_state=random_state,
    )
    slices = {
        name: float(daily40.loc[start:end].mean())
        for name, (start, end) in SELECTION_SLICES.items()
    }
    slice_counts = {
        name: int(len(daily40.loc[start:end]))
        for name, (start, end) in SELECTION_SLICES.items()
    }
    top4_removed = float(daily40.drop(daily40.nlargest(4).index).mean())
    code_cash, top_codes = top_codes_cash(
        picks, scheduled, cost_bps=PRIMARY_COST_BPS, count=5
    )
    monthly = daily40.groupby(daily40.index.to_period("M")).mean()
    signaled = picks.dropna(subset=["code"])
    counts = signaled["code"].astype(str).value_counts()
    total = int(len(signaled))
    unique_codes = int(len(counts))
    maximum_share = float(counts.iloc[0] / total) if total else 1.0
    top10_share = float(counts.head(10).sum() / total) if total else 1.0
    executed = picks["label"].notna()
    executed_days = int(executed.sum())
    executed_fraction = float(executed.mean())
    checks = {
        "net40_mean_positive": float(daily40.mean()) > 0.0,
        "net40_median_positive": float(daily40.median()) > 0.0,
        "net60_mean_positive": float(daily60.mean()) > 0.0,
        "both_fixed_slices_net40_positive": all(value > 0.0 for value in slices.values()),
        "positive_months_net40_at_least_3": int(monthly.gt(0.0).sum()) >= 3,
        "top4_days_removed_net40_positive": top4_removed > 0.0,
        "top5_profit_codes_cash_net40_positive": float(code_cash.mean()) > 0.0,
        "familywise_paired_lower_vs_control_nonnegative": (
            paired["one_sided_lower_delta_pct"] >= 0.0
        ),
        "unique_codes_at_least_40": unique_codes >= 40,
        "maximum_code_share_at_most_0_05": maximum_share <= 0.05,
        "top10_code_share_at_most_0_25": top10_share <= 0.25,
        "executed_days_at_least_72": executed_days >= 72,
        "executed_slot_fraction_at_least_0_8": executed_fraction >= 0.80,
    }
    return {
        "variant_id": candidate_id,
        "candidate_id": candidate_id,
        "matched_control_id": control_id,
        "capacity": 1,
        "cost_metrics": cost_metrics,
        "net40_mean_pct": float(daily40.mean()),
        "net40_median_pct": float(daily40.median()),
        "net60_mean_pct": float(daily60.mean()),
        "slice_net40_mean_pct": slices,
        "slice_session_counts": slice_counts,
        "monthly_net40_mean_pct": {
            str(period): float(value) for period, value in monthly.items()
        },
        "positive_months_net40": int(monthly.gt(0.0).sum()),
        "top4_days_removed_net40_mean_pct": top4_removed,
        "top5_profit_codes_cash_net40_mean_pct": float(code_cash.mean()),
        "top5_profit_codes": top_codes,
        "paired_vs_matched_control_net40": paired,
        "bonferroni_individual_confidence": BOOTSTRAP_CONFIDENCE,
        "unique_selected_codes": unique_codes,
        "maximum_code_selection_share": maximum_share,
        "top10_code_selection_share": top10_share,
        "executed_days": executed_days,
        "executed_slot_fraction": executed_fraction,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def choose_winner(variants: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    passers = [item for item in variants if item["gate_passed"]]
    if not passers:
        return None
    winner = sorted(
        passers,
        key=lambda item: (
            -float(
                item["paired_vs_matched_control_net40"][
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
        "matched_control_id": winner["matched_control_id"],
        "selection_gate_passed": True,
        "selection_rule_rank": 1,
    }


def _source_set_digest(records: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(records, key=lambda item: str(item["name"]))
    return hashlib.sha256(
        "\n".join(str(item["sha256"]) for item in ordered).encode("ascii")
    ).hexdigest()


def _locked_source_set_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Recompute the input-lock's name/size/hash/URL manifest digest."""

    ordered = sorted(records, key=lambda item: str(item["name"]))
    payload = "\n".join(
        "\t".join(
            (
                str(item["name"]),
                str(item["bytes"]),
                str(item["sha256"]),
                str(item["source_url"]),
            )
        )
        for item in ordered
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _warmup_bindings_match(
    protocol: dict[str, Any], result: dict[str, Any], manifest: dict[str, Any]
) -> bool:
    names = {f"{month}.pdf" for month in protocol["source_contract"]["price_warmup"]["months"]}
    observed = result["input"]["price_warmup_sources"]
    registered = {
        str(item["filename"]): item
        for item in manifest["jpx"]["sources"]
        if str(item.get("filename")) in names
    }
    if set(registered) != names or {str(item.get("name")) for item in observed} != names:
        return False
    if len(observed) != len(names):
        return False
    return all(
        set(item) == {"bytes", "name", "sha256"}
        and int(item["bytes"]) == int(registered[str(item["name"])]["bytes"])
        and str(item["sha256"]) == str(registered[str(item["name"])]["sha256"])
        for item in observed
    )


def _protocol_contract_matches(protocol: dict[str, Any]) -> bool:
    candidates = tuple(item.get("id") for item in protocol.get("candidates", []))
    controls = tuple(item.get("id") for item in protocol.get("controls", []))
    matched = {
        str(item.get("id")): item.get("matched_control")
        for item in protocol.get("candidates", [])
    }
    evaluation = protocol.get("evaluation", {})
    bootstrap = evaluation.get("bootstrap", {})
    gates = evaluation.get("selection_gate_all_required", {})
    expected_gates = {
        "net40_mean_positive": True,
        "net40_median_positive": True,
        "net60_mean_positive": True,
        "both_fixed_slices_net40_positive": True,
        "positive_months_net40_at_least": 3,
        "top4_days_removed_net40_positive": True,
        "top5_profit_codes_cash_net40_positive": True,
        "familywise_paired_lower_vs_control_nonnegative": True,
        "unique_codes_at_least": 40,
        "maximum_code_share_at_most": 0.05,
        "top10_code_share_at_most": 0.25,
        "executed_days_at_least": 72,
        "executed_slot_fraction_at_least": 0.8,
    }
    at10 = protocol.get("candidates", [{}])[-1]
    return bool(
        protocol.get("schema_version") == 1
        and protocol.get("protocol_id") == PROTOCOL_ID
        and candidates == CANDIDATES
        and controls == (C00, PAIR00, C02)
        and matched == MATCHED_CONTROL
        and all(item.get("slot") == 1 for item in protocol["candidates"])
        and evaluation.get("candidate_variants") == 10
        and evaluation.get("fixed_slots_per_model_date") == 1
        and evaluation.get("no_capacity_grid") is True
        and tuple(float(value) for value in evaluation.get("costs_bps", [])) == COSTS_BPS
        and float(evaluation.get("primary_cost_bps", -1)) == PRIMARY_COST_BPS
        and evaluation.get("selection_slices")
        == {key: [str(bounds[0].date()), str(bounds[1].date())] for key, bounds in SELECTION_SLICES.items()}
        and bootstrap.get("method") == "paired moving block"
        and bootstrap.get("block_length") == BOOTSTRAP_BLOCK_LENGTH
        and bootstrap.get("samples") == BOOTSTRAP_SAMPLES
        and bootstrap.get("random_state") == BOOTSTRAP_RANDOM_STATE
        and bootstrap.get("candidate_seed_rule")
        == (
            "random_state + zero-based candidate order index, yielding "
            "20260804 through 20260813"
        )
        and bootstrap.get("familywise_one_sided_confidence") == 0.9
        and bootstrap.get("bonferroni_individual_confidence")
        == BOOTSTRAP_CONFIDENCE
        and gates == expected_gates
        and "four issuer turnover ranks" in str(at10.get("availability_rule", ""))
        and "never imputed or proxied" in str(at10.get("availability_rule", ""))
    )


def _fold_contract_matches(result: dict[str, Any]) -> bool:
    folds = result["selection"]["folds"]
    identities = [(str(item.get("candidate_id")), str(item.get("period"))) for item in folds]
    expected: set[tuple[str, str]] = {
        (C00, period)
        for period in (
            "2025-12", "2026-01", "2026-02", "2026-03",
            "2026-04", "2026-05", "2026-06", "2026-07",
        )
    }
    fitted = (
        PAIR00,
        "PAIR01_TOP2_EXACT_LIQ_RERANK",
        "TW02_TURNOVER_WEIGHTED_PRICE_MEMORY",
        "VW03_AGGREGATE_COST_BASIS",
        "AR05_SECURITY_ACTIVITY_REGIME_EXPERTS",
        "PS06_PERSISTENT_VS_ISOLATED_ACTIVITY",
        "VP07_VWAP_RANGE_PRESSURE_MEMORY",
        "RW08_EXACT_LOT_RELIABILITY_WEIGHT",
        "MR09_TURNOVER_WEIGHTED_MARKET_REGIME",
        "AT10_ATTENTION_MIGRATION",
    )
    months = ("2026-04", "2026-05", "2026-06", "2026-07")
    expected.update((model, period) for model in fitted for period in months)
    if len(identities) != len(set(identities)) or set(identities) != expected:
        return False
    for item in folds:
        period_start = pd.Period(str(item["period"]), freq="M").start_time
        if (
            pd.Timestamp(item["train_end"]) >= period_start
            or item.get("strictly_prior_training") is not True
            or item.get("replacement_model_used", False) is not False
        ):
            return False
    rw = [item for item in folds if item.get("candidate_id") == "RW08_EXACT_LOT_RELIABILITY_WEIGHT"]
    return bool(
        len(rw) == 4
        and all(
            int(item.get("score_rows", -1)) == int(item.get("common_score_rows", -2))
            and int(item.get("feature_unavailable_rows", -1)) == 0
            for item in rw
        )
    )


def _pair_top2_contract_matches(picks: pd.DataFrame) -> bool:
    by_model = {
        model: picks.loc[picks["candidate_id"].eq(model)].set_index("date")
        for model in (C00, C02, PAIR00, "PAIR01_TOP2_EXACT_LIQ_RERANK")
    }
    for date in by_model[C00].index:
        c00_code = by_model[C00].at[date, "code"]
        c02_code = by_model[C02].at[date, "code"]
        if pd.isna(c02_code):
            # A frozen C00 set of size zero or one is an incomplete pair.  C00
            # may still display its sole row, but both pairwise policies and
            # the rank-two diagnostic must remain cash without replacement.
            if pd.isna(c00_code):
                if not pd.isna(by_model[C00].at[date, "source_rank"]):
                    return False
            elif int(by_model[C00].at[date, "source_rank"]) != 1:
                return False
            if not pd.isna(by_model[C02].at[date, "source_rank"]):
                return False
            if not all(
                pd.isna(by_model[model].at[date, "code"])
                and pd.isna(by_model[model].at[date, "source_rank"])
                for model in (PAIR00, "PAIR01_TOP2_EXACT_LIQ_RERANK")
            ):
                return False
            continue
        if pd.isna(c00_code):
            return False
        top = {
            str(c00_code),
            str(c02_code),
        }
        if len(top) != 2:
            return False
        if int(by_model[C00].at[date, "source_rank"]) != 1:
            return False
        if int(by_model[C02].at[date, "source_rank"]) != 2:
            return False
        for model in (PAIR00, "PAIR01_TOP2_EXACT_LIQ_RERANK"):
            code = by_model[model].at[date, "code"]
            rank = by_model[model].at[date, "source_rank"]
            if pd.isna(code) or str(code) not in top or int(rank) not in (1, 2):
                return False
            expected = str(by_model[C00 if int(rank) == 1 else C02].at[date, "code"])
            if str(code) != expected:
                return False
    return True


def audit(
    *,
    protocol_path: Path,
    result_path: Path,
    scores_path: Path,
    picks_path: Path,
    runner_path: Path,
    input_lock_path: Path = INPUT_LOCK,
) -> dict[str, Any]:
    recorder = AuditRecorder()
    protocol = read_json(protocol_path)
    result = read_json(result_path)
    input_lock = read_json(input_lock_path)
    price_manifest = read_json(PRICE_MANIFEST)
    legacy_audit = read_json(LEGACY_PARSER_AUDIT)
    scores = load_scores(scores_path)
    picks = load_picks(picks_path)

    artifact_hashes = {
        "protocol": sha256_file(protocol_path),
        "result": sha256_file(result_path),
        "scores": sha256_file(scores_path),
        "picks": sha256_file(picks_path),
        "runner": sha256_file(runner_path),
        "audit_runner": sha256_file(__file__),
        "input_lock": sha256_file(input_lock_path),
        "parser_source": sha256_file(PARSER_SOURCE),
        "price_manifest": sha256_file(PRICE_MANIFEST),
        "legacy_parser_audit": sha256_file(LEGACY_PARSER_AUDIT),
    }
    recorder.check(
        "protocol_hash_frozen",
        artifact_hashes["protocol"] == PROTOCOL_SHA256,
        observed=artifact_hashes["protocol"],
        expected=PROTOCOL_SHA256,
    )
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
        "score_file_hash_matches_result",
        artifact_hashes["scores"]
        == result.get("selection", {}).get("score_ledger_file_sha256"),
        observed=artifact_hashes["scores"],
        expected=result.get("selection", {}).get("score_ledger_file_sha256"),
    )
    recorder.check(
        "picks_hash_matches_result",
        artifact_hashes["picks"] == result.get("picks_sha256"),
        observed=artifact_hashes["picks"],
        expected=result.get("picks_sha256"),
    )
    recorder.check(
        "input_lock_hash_frozen",
        artifact_hashes["input_lock"] == INPUT_LOCK_SHA256,
        observed=artifact_hashes["input_lock"],
        expected=INPUT_LOCK_SHA256,
    )
    recorder.check(
        "parser_source_hash_frozen",
        artifact_hashes["parser_source"] == PARSER_SHA256,
        observed=artifact_hashes["parser_source"],
        expected=PARSER_SHA256,
    )
    recorder.check("protocol_contract_exact", _protocol_contract_matches(protocol))
    recorder.check(
        "protocol_id_matches_result",
        result.get("protocol_id") == PROTOCOL_ID == protocol.get("protocol_id"),
    )

    replay_contract = protocol["source_contract"]["replay_daily"]
    lock_records = input_lock.get("files", [])
    lock_names = [str(item.get("name", "")) for item in lock_records]
    lock_dates = [str(item.get("date", "")) for item in lock_records]
    lock_valid = bool(
        input_lock.get("schema_version") == 1
        and input_lock.get("lock_id") == "model_v17_replay_inputs_20260804"
        and input_lock.get("status") == "source_locked_before_outcome_parse"
        and input_lock.get("source_set_sha256") == INPUT_SOURCE_SET_SHA256
        and len(lock_records) == 79
        and len(lock_names) == len(set(lock_names))
        and all(_DAILY_NAME.fullmatch(name) for name in lock_names)
        and lock_dates[0] == "2026-04-01"
        and lock_dates[-1] == "2026-07-27"
        and _locked_source_set_digest(lock_records) == INPUT_SOURCE_SET_SHA256
        and replay_contract["input_lock_sha256"] == INPUT_LOCK_SHA256
        and replay_contract["source_set_sha256"] == INPUT_SOURCE_SET_SHA256
        and input_lock.get("parser_contract", {}).get("source_sha256") == PARSER_SHA256
        and input_lock.get("parser_contract", {}).get("version") == PARSER_VERSION
    )
    recorder.check("input_lock_contract_and_source_set_exact", lock_valid)

    legacy_records = legacy_audit.get("per_file", [])
    combined_records = [*legacy_records, *lock_records]
    source_input = result.get("input", {})
    source_result_valid = bool(
        len(legacy_records) == 160
        and len(combined_records) == 239
        and source_input.get("legacy_daily_source_count") == 160
        and source_input.get("extension_daily_source_count") == 79
        and source_input.get("daily_source_count") == 239
        and source_input.get("daily_sessions") == 239
        and source_input.get("daily_date_bounds") == ["2025-08-01", "2026-07-27"]
        and source_input.get("daily_rejected_rows") == 0
        and source_input.get("parser_version") == PARSER_VERSION
        and source_input.get("replay_input_lock_sha256") == INPUT_LOCK_SHA256
        and source_input.get("replay_input_lock_source_set_sha256") == INPUT_SOURCE_SET_SHA256
        and source_input.get("extension_source_set_sha256") == _source_set_digest(lock_records)
        and source_input.get("daily_source_set_sha256") == _source_set_digest(combined_records)
    )
    recorder.check("result_source_bindings_recomputed", source_result_valid)
    recorder.check(
        "warmup_sources_match_manifest",
        _warmup_bindings_match(protocol, result, price_manifest),
    )

    recorder.check(
        "score_and_picks_ledgers_identical_before_outcomes",
        _ledger_values_equal(scores, picks),
    )
    score_hash = semantic_score_hash(scores)
    picks_score_hash = semantic_score_hash(picks)
    expected_score_hash = result["selection"]["score_ledger_semantic_sha256"]
    recorder.check(
        "score_semantic_hash_matches_result",
        score_hash == expected_score_hash == picks_score_hash,
        observed={"scores": score_hash, "picks": picks_score_hash},
        expected=expected_score_hash,
    )
    recorder.check(
        "score_ledger_outcome_free_before_join",
        result["selection"].get("score_ledger_outcome_columns") == []
        and result["selection"].get("score_ledger_hashed_before_outcome_join") is True,
    )
    recorder.check("outcome_label_matches_return_sign", outcome_contract_matches(picks))
    recorder.check("fixed_slot_cash_and_no_veto_contract", fixed_slot_contract_matches(picks))

    scheduled = pd.DatetimeIndex(sorted(picks["date"].drop_duplicates()))
    observed_models = tuple(picks["candidate_id"].drop_duplicates())
    slots = picks.groupby(["candidate_id", "date"])["model_rank"].agg(tuple)
    recorder.check(
        "selection_calendar_and_fixed_slots_complete",
        len(scheduled) == SELECTION_SESSIONS
        and scheduled.min() == SELECTION_START
        and scheduled.max() == SELECTION_END
        and len(picks) == len(ALL_MODELS) * SELECTION_SESSIONS
        and set(observed_models) == set(ALL_MODELS)
        and result["selection"].get("models") == list(ALL_MODELS)
        and result["selection"].get("candidates") == list(CANDIDATES)
        and result["selection"].get("matched_controls") == MATCHED_CONTROL
        and not picks.duplicated(["candidate_id", "date", "model_rank"]).any()
        and slots.map(lambda value: value == (1,)).all(),
    )
    recorder.check("pairwise_models_restricted_to_frozen_top2", _pair_top2_contract_matches(picks))
    recorder.check("all_folds_strictly_prior_and_registered", _fold_contract_matches(result))

    result_authority = result.get("authority", {})
    protocol_authority = protocol.get("authority", {})
    authority_valid = all(
        mapping.get(key) is False
        for mapping in (protocol_authority, result_authority)
        for key in (
            "project_level_untouched",
            "production_promotion_allowed",
            "production_model_changed",
            "orders_allowed",
        )
    ) and result.get("decision", {}).get("production_model_changed") is False and result.get("decision", {}).get("orders_allowed") is False
    recorder.check("authority_remains_research_only", authority_valid)

    picks_by_model = {
        model: picks.loc[picks["candidate_id"].eq(model)].copy()
        for model in ALL_MODELS
    }
    independent_controls = {
        control: {
            str(int(cost)): profit_metrics(
                picks_by_model[control], scheduled, cost_bps=cost
            )
            for cost in COSTS_BPS
        }
        for control in (C00, PAIR00)
    }
    independent_diagnostic = {
        C02: {
            str(int(cost)): profit_metrics(picks_by_model[C02], scheduled, cost_bps=cost)
            for cost in COSTS_BPS
        }
    }
    recorder.compare_nested(
        "control_metrics_recomputed",
        result["selection"]["control_metrics"],
        independent_controls,
    )
    recorder.compare_nested(
        "diagnostic_metrics_recomputed",
        result["selection"]["diagnostic_metrics"],
        independent_diagnostic,
    )
    independent_variants = [
        variant_metrics(
            picks_by_model[candidate],
            picks_by_model[MATCHED_CONTROL[candidate]],
            scheduled,
            candidate_id=candidate,
            control_id=MATCHED_CONTROL[candidate],
            random_state=BOOTSTRAP_RANDOM_STATE + index,
        )
        for index, candidate in enumerate(CANDIDATES)
    ]
    recorder.check(
        "candidate_family_size_exact",
        len(independent_variants)
        == protocol["evaluation"]["candidate_variants"]
        == result["selection"]["candidate_variants"]
        == 10,
    )
    recorder.compare_nested(
        "variant_metrics_bootstrap_and_gates_recomputed",
        result["selection"]["variants"],
        independent_variants,
    )
    winner = choose_winner(independent_variants)
    gate_passers = sum(item["gate_passed"] for item in independent_variants)
    status = (
        "selection_passed_one_research_nominee"
        if winner is not None
        else "selection_rejected_all_candidates"
    )
    decision = {
        "candidate_family_rejected": winner is None,
        "research_nominee": None if winner is None else winner["variant_id"],
        "production_model_changed": False,
        "orders_allowed": False,
    }
    recorder.compare_nested("winner_rule_recomputed", result["selection"]["winner"], winner)
    recorder.check(
        "gate_passer_count_recomputed",
        result["selection"]["gate_passers"] == gate_passers,
        observed=result["selection"]["gate_passers"],
        expected=gate_passers,
    )
    recorder.check(
        "status_recomputed", result.get("status") == status, observed=result.get("status"), expected=status
    )
    recorder.compare_nested("decision_recomputed", result.get("decision"), decision)

    summaries = [
        {
            "variant_id": item["variant_id"],
            "net20_mean_pct": item["cost_metrics"]["20"]["net_mean_pct_at_cost"],
            "net40_mean_pct": item["net40_mean_pct"],
            "net60_mean_pct": item["net60_mean_pct"],
            "paired_lower_vs_control_net40": item[
                "paired_vs_matched_control_net40"
            ]["one_sided_lower_delta_pct"],
            "gate_passed": item["gate_passed"],
            "failed_gate_count": sum(not value for value in item["gate_checks"].values()),
        }
        for item in independent_variants
    ]
    passed = not recorder.discrepancies and all(recorder.checks.values())
    return {
        "schema_version": 1,
        "audit_id": "model_v17_liquidity_reliability_independent_audit_20260804",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "independence": {
            "selection_runner_imported": False,
            "project_profit_helpers_imported": False,
            "project_bootstrap_helpers_imported": False,
            "recomputed_from": [
                "research/model_v17_liquidity_reliability_protocol.json",
                "research/model_v17_liquidity_reliability_result.json",
                "research/model_v17_liquidity_reliability_scores.csv",
                "research/model_v17_liquidity_reliability_picks.csv",
            ],
        },
        "artifact_hashes": artifact_hashes,
        "scope": {
            "protocol_input_and_artifact_bindings": True,
            "score_ledger_before_outcome_join": True,
            "daily_fixed_slot_returns": True,
            "cost_metrics": True,
            "tail_and_code_cash_stress": True,
            "paired_moving_block_bootstrap": True,
            "all_selection_gates": True,
            "winner_status_and_decision": True,
            "fold_and_pairwise_contracts": True,
            "raw_pdf_reparse": False,
            "feature_reconstruction": False,
        },
        "integrity": {
            "models": list(ALL_MODELS),
            "scheduled_sessions": len(scheduled),
            "score_rows": len(scores),
            "picks_rows": len(picks),
            "score_ledger_semantic_sha256": score_hash,
            "observed_outcome_slots": int(picks["label"].notna().sum()),
            "vetoed_slots": int(picks["vetoed"].sum()),
        },
        "recomputed": {
            "variant_summaries": summaries,
            "gate_passers": gate_passers,
            "winner": winner,
            "status": status,
            "decision": decision,
            "maximum_absolute_numeric_difference": recorder.max_abs_numeric_difference,
            "numeric_tolerance": NUMERIC_TOLERANCE,
        },
        "authority": {
            "project_level_untouched": False,
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
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--picks", type=Path, default=DEFAULT_PICKS)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--input-lock", type=Path, default=INPUT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(
        protocol_path=args.protocol,
        result_path=args.result,
        scores_path=args.scores,
        picks_path=args.picks,
        runner_path=args.runner,
        input_lock_path=args.input_lock,
    )
    write_json(report, args.output)
    if report["status"] != "pass":
        raise SystemExit(
            "v1.7 independent audit failed; see discrepancies in " f"{args.output}"
        )
    print(
        "v1.7 independent audit passed: "
        f"{len(report['checks'])} checks, "
        f"{report['recomputed']['gate_passers']} gate passers"
    )


if __name__ == "__main__":
    main()
