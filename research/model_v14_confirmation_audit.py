#!/usr/bin/env python3
"""Independently audit the v1.4 confirmation result from its frozen picks.

The audit deliberately does not import the v1.4 confirmation runner.  It
reconstructs the scheduled cash-slot portfolios, robustness removals,
concentration/execution statistics, multiplicity-adjusted paired bootstrap,
qualification gates, and final decision directly from the emitted CSV.
"""

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


PROTOCOL = ROOT / "research/model_v14_confirmation_protocol.json"
PROTOCOL_SHA256 = (
    "ddc635c986dcb56072beda4889bfbf319019db31dfe3f74eedec24e51d8276af"
)
RUNNER = ROOT / "research/model_v14_confirmation_runner.py"
STAGE_A_PROTOCOL = ROOT / "research/model_v14_feature_contrast_protocol.json"
STAGE_A_PROTOCOL_SHA256 = (
    "8b2fcca037d36776a7803e066396e9cc711a47f1f67a57df5e117b653fd45f28"
)
STAGE_A_RESULT = ROOT / "research/model_v14_feature_contrast_result.json"
STAGE_A_RESULT_SHA256 = (
    "6cd878eca0392f87a781a3a4b6c3d38dd9948ba071ebc9f8f92f58557705eb31"
)
INPUT_ERRATUM = ROOT / "research/model_v14_feature_contrast_input_erratum.json"
INPUT_ERRATUM_SHA256 = (
    "656be1249bc8d4f9fa8ce2f22bb6ab3c14450a16cf15c7c86f3e6492cf77c90f"
)
V13_INPUT_ERRATUM_SHA256 = (
    "991ef4dfd20171d9be371d1b8d69d4074fe6536339bb7ff264f9708758c67853"
)
INPUT_MANIFEST = ROOT / "research/model_v05_input_lock.json"
INPUT_MANIFEST_SHA256 = (
    "02370bda9c5fe73b166c557bcdc837d363f5deafbe91baa2d450b33dfcd45272"
)
V13_RESULT = ROOT / "research/model_v13_symbolic_context_result.json"
V13_RESULT_SHA256 = (
    "1f1a47ce04c4f72e5c581195120741c45d3a9cf013801627ecd2c1a356fa33de"
)
V13_PICKS = ROOT / "research/model_v13_symbolic_context_picks.csv"
PANEL_CACHE_SHA256 = (
    "abcc6de28217f721358278c17039a60e4542361a60e2b1ac929325ab1a97516f"
)

PROTOCOL_ID = "model_v14_feature_contrast_confirmation_20260727"
CONTROL = "C00_DAILY_RANK_RIDGE"
CANDIDATES = ("DMD01", "DMD02", "SP01", "GN01")
ALL_MODELS = (CONTROL, *CANDIDATES)
CAPACITIES = (1, 2)
COSTS_BPS = (20.0, 40.0, 60.0)
FAMILY_SIZE = 8
SCHEDULED_SESSIONS = 182
CONFIRMATION_START = pd.Timestamp("2024-11-01")
CONFIRMATION_END = pd.Timestamp("2025-07-31")
CONFIRMATION_SLICES = {
    "confirmation_a": (
        pd.Timestamp("2024-11-01"),
        pd.Timestamp("2025-01-31"),
    ),
    "confirmation_b": (
        pd.Timestamp("2025-02-03"),
        pd.Timestamp("2025-04-30"),
    ),
    "confirmation_c": (
        pd.Timestamp("2025-05-01"),
        pd.Timestamp("2025-07-31"),
    ),
}
BOOTSTRAP_BLOCK_LENGTH = 5
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_RANDOM_STATE = 20_260_727
BONFERRONI_CONFIDENCE = 1.0 - (1.0 - 0.90) / FAMILY_SIZE
TOLERANCE = 1e-12
PICK_COLUMNS = (
    "candidate_id",
    "date",
    "model_rank",
    "code",
    "name",
    "model_score",
    "label",
    "oc_return_pct",
)
CHECK_NAMES = (
    "artifact_bindings",
    "slot_schedule",
    "label_return_sign",
    "v13_control_exact_reproduction",
    "cost_and_robustness_recalculation",
    "concentration_and_execution_recalculation",
    "familywise_bootstrap_recalculation",
    "gate_recalculation",
    "decision_recalculation",
    "strictly_prior_fold_metadata",
    "production_and_order_boundary",
)


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


class Audit:
    """Collect strict audit mismatches without stopping after the first one."""

    def __init__(self) -> None:
        self.mismatches: list[dict[str, Any]] = []

    def mismatch(
        self,
        check: str,
        path: str,
        *,
        expected: Any,
        observed: Any,
        detail: str | None = None,
    ) -> None:
        item = {
            "check": check,
            "path": path,
            "expected": json_safe(expected),
            "observed": json_safe(observed),
        }
        if detail is not None:
            item["detail"] = detail
        self.mismatches.append(item)

    def equal(
        self,
        check: str,
        path: str,
        expected: Any,
        observed: Any,
    ) -> None:
        if expected != observed:
            self.mismatch(
                check,
                path,
                expected=expected,
                observed=observed,
            )

    def close(
        self,
        check: str,
        path: str,
        expected: float,
        observed: float,
    ) -> None:
        try:
            matches = math.isclose(
                float(expected),
                float(observed),
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            self.mismatch(
                check,
                path,
                expected=expected,
                observed=observed,
                detail=f"absolute tolerance {TOLERANCE}",
            )

    def tree(
        self,
        check: str,
        path: str,
        expected: Any,
        observed: Any,
    ) -> None:
        """Compare nested JSON-like values, with strict keys and float tolerance."""

        if isinstance(expected, dict):
            if not isinstance(observed, dict):
                self.mismatch(
                    check, path, expected=expected, observed=observed
                )
                return
            self.equal(
                check,
                f"{path}.__keys__",
                sorted(expected),
                sorted(observed),
            )
            for key in sorted(set(expected) & set(observed)):
                self.tree(
                    check,
                    f"{path}.{key}",
                    expected[key],
                    observed[key],
                )
            return
        if isinstance(expected, list):
            if not isinstance(observed, list):
                self.mismatch(
                    check, path, expected=expected, observed=observed
                )
                return
            self.equal(check, f"{path}.__length__", len(expected), len(observed))
            for index, (left, right) in enumerate(zip(expected, observed)):
                self.tree(check, f"{path}[{index}]", left, right)
            return
        if (
            isinstance(expected, (float, np.floating))
            and not isinstance(expected, bool)
        ):
            self.close(check, path, float(expected), observed)
            return
        self.equal(check, path, expected, observed)

    def status(self, check: str) -> str:
        return (
            "FAIL"
            if any(item["check"] == check for item in self.mismatches)
            else "PASS"
        )


def semantic_hash(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    canonical["date"] = pd.to_datetime(canonical["date"]).dt.strftime(
        "%Y-%m-%d"
    )
    canonical = canonical.sort_values(
        ["candidate_id", "date", "model_rank"], kind="stable"
    ).reset_index(drop=True)
    return hashlib.sha256(
        canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def load_picks(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"code": "string"},
        float_precision="round_trip",
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def daily_returns(
    picks: pd.DataFrame,
    schedule: pd.DatetimeIndex,
    *,
    capacity: int,
    cost_bps: float,
) -> pd.Series:
    selected = picks.loc[picks["model_rank"].le(capacity)].copy()
    executed = selected["label"].notna()
    selected["_net_slot"] = selected["oc_return_pct"].fillna(0.0) - (
        executed.astype(float) * cost_bps / 100.0
    )
    daily = (
        selected.groupby("date", sort=True)["_net_slot"]
        .sum()
        .div(capacity)
        .reindex(schedule)
    )
    if len(daily) != len(schedule) or daily.isna().any():
        raise AssertionError("daily cash-slot schedule is incomplete")
    return daily


def top_profitable_codes_cash(
    picks: pd.DataFrame,
    schedule: pd.DatetimeIndex,
    *,
    capacity: int,
    count: int = 10,
) -> tuple[pd.Series, list[str]]:
    selected = picks.loc[picks["model_rank"].le(capacity)].copy()
    executed = selected["label"].notna()
    selected["_net_slot"] = selected["oc_return_pct"].fillna(0.0) - (
        executed.astype(float) * 0.40
    )
    totals = (
        selected.dropna(subset=["code"])
        .groupby("code", sort=False)["_net_slot"]
        .sum()
        .sort_values(ascending=False, kind="stable")
    )
    codes = [str(code) for code in totals.head(count).index]
    neutral = picks.copy()
    mask = (
        neutral["model_rank"].le(capacity)
        & neutral["code"].astype(str).isin(codes)
    )
    neutral.loc[mask, ["label", "oc_return_pct"]] = np.nan
    return (
        daily_returns(
            neutral,
            schedule,
            capacity=capacity,
            cost_bps=40.0,
        ),
        codes,
    )


def audit_bindings(
    audit: Audit,
    result: dict[str, Any],
    protocol: dict[str, Any],
    stage_a_result: dict[str, Any],
    v13_result: dict[str, Any],
    picks: pd.DataFrame,
) -> None:
    check = "artifact_bindings"
    bindings = (
        ("protocol file", PROTOCOL_SHA256, sha256_file(PROTOCOL)),
        (
            "stage A protocol file",
            STAGE_A_PROTOCOL_SHA256,
            sha256_file(STAGE_A_PROTOCOL),
        ),
        (
            "stage A result file",
            STAGE_A_RESULT_SHA256,
            sha256_file(STAGE_A_RESULT),
        ),
        (
            "input erratum file",
            INPUT_ERRATUM_SHA256,
            sha256_file(INPUT_ERRATUM),
        ),
        (
            "input manifest file",
            INPUT_MANIFEST_SHA256,
            sha256_file(INPUT_MANIFEST),
        ),
        ("v1.3 result file", V13_RESULT_SHA256, sha256_file(V13_RESULT)),
    )
    for label, expected, observed in bindings:
        audit.equal(check, label, expected, observed)

    audit.equal(check, "result.protocol_id", PROTOCOL_ID, result["protocol_id"])
    audit.equal(
        check,
        "result.protocol_sha256",
        PROTOCOL_SHA256,
        result["protocol_sha256"],
    )
    audit.equal(
        check,
        "result.stage_a_protocol_sha256",
        STAGE_A_PROTOCOL_SHA256,
        result["stage_a_protocol_sha256"],
    )
    audit.equal(
        check,
        "result.stage_a_result_sha256",
        STAGE_A_RESULT_SHA256,
        result["stage_a_result_sha256"],
    )
    audit.equal(
        check,
        "result.input_erratum_sha256",
        INPUT_ERRATUM_SHA256,
        result["input_erratum_sha256"],
    )
    audit.equal(
        check,
        "result.runner_sha256",
        sha256_file(RUNNER),
        result["runner_sha256"],
    )
    audit.equal(
        check,
        "result.integrity.picks_semantic_sha256",
        semantic_hash(picks),
        result["integrity"]["picks_semantic_sha256"],
    )
    audit.equal(
        check,
        "result.input.panel_cache_sha256",
        PANEL_CACHE_SHA256,
        result["input"]["panel_cache_sha256"],
    )
    audit.equal(
        check,
        "result.input.input_manifest_sha256",
        INPUT_MANIFEST_SHA256,
        result["input"]["input_manifest_sha256"],
    )
    audit.equal(
        check,
        "result.input.input_erratum_sha256",
        V13_INPUT_ERRATUM_SHA256,
        result["input"]["input_erratum_sha256"],
    )
    audit.equal(
        check,
        "protocol.stage_a_binding.result.sha256",
        STAGE_A_RESULT_SHA256,
        protocol["stage_a_binding"]["result"]["sha256"],
    )
    audit.equal(
        check,
        "protocol.frozen_input.panel_cache_sha256",
        PANEL_CACHE_SHA256,
        protocol["frozen_input"]["panel_cache_sha256"],
    )
    input_dimensions = {
        "rows": protocol["frozen_input"]["canonical_rows"],
        "modeling_rows": protocol["frozen_input"]["modeling_rows"],
        "panel_rows": protocol["frozen_input"]["modeling_rows"],
        "codes": protocol["frozen_input"]["codes"],
        "modeling_codes": protocol["frozen_input"]["codes"],
        "panel_codes": protocol["frozen_input"]["codes"],
        "sessions": protocol["frozen_input"]["sessions"],
        "date_bounds": protocol["frozen_input"]["date_bounds"],
        "frozen_panel_dimensions_verified": True,
        "rejected_rows": 0,
        "source_incomplete_sessions": 0,
        "strictly_prior_context_source_violations": 0,
    }
    for field, expected in input_dimensions.items():
        audit.tree(
            check,
            f"result.input.{field}",
            expected,
            result["input"][field],
        )
    audit.equal(
        check,
        "result.input official source file count",
        19,
        len(result["input"]["source_files"]),
    )
    audit.equal(
        check,
        "stage_a_result.protocol_sha256",
        STAGE_A_PROTOCOL_SHA256,
        stage_a_result["protocol_sha256"],
    )
    audit.equal(
        check,
        "v13 picks semantic binding",
        v13_result["integrity"]["picks_semantic_sha256"],
        semantic_hash(load_picks(V13_PICKS)),
    )


def audit_schedule_and_labels(
    audit: Audit,
    result: dict[str, Any],
    picks: pd.DataFrame,
    v13_picks: pd.DataFrame,
) -> pd.DatetimeIndex:
    schedule_check = "slot_schedule"
    audit.equal(
        schedule_check,
        "picks columns",
        list(PICK_COLUMNS),
        list(picks.columns),
    )
    reference_control = v13_picks.loc[
        v13_picks["candidate_id"].eq(CONTROL)
        & v13_picks["date"].between(
            CONFIRMATION_START, CONFIRMATION_END, inclusive="both"
        )
    ].copy()
    schedule = pd.DatetimeIndex(
        reference_control["date"].drop_duplicates().sort_values()
    )
    audit.equal(
        schedule_check,
        "reference scheduled session count",
        SCHEDULED_SESSIONS,
        len(schedule),
    )
    audit.equal(
        schedule_check,
        "reference first session",
        CONFIRMATION_START,
        schedule.min(),
    )
    audit.equal(
        schedule_check,
        "reference last session",
        CONFIRMATION_END,
        schedule.max(),
    )
    audit.equal(
        schedule_check,
        "candidate/model set",
        sorted(ALL_MODELS),
        sorted(picks["candidate_id"].unique().tolist()),
    )
    audit.equal(
        schedule_check,
        "picks row count",
        len(ALL_MODELS) * SCHEDULED_SESSIONS * 2,
        len(picks),
    )
    audit.equal(
        schedule_check,
        "duplicate candidate/date/rank slots",
        0,
        int(picks.duplicated(["candidate_id", "date", "model_rank"]).sum()),
    )
    expected_slots = pd.MultiIndex.from_product(
        [schedule, (1, 2)], names=["date", "model_rank"]
    )
    for model in ALL_MODELS:
        model_slots = pd.MultiIndex.from_frame(
            picks.loc[
                picks["candidate_id"].eq(model), ["date", "model_rank"]
            ].sort_values(["date", "model_rank"], kind="stable")
        )
        audit.equal(
            schedule_check,
            f"{model} exact 182-date x two-rank schedule",
            expected_slots.tolist(),
            model_slots.tolist(),
        )
    integrity_expected = {
        "candidate_count": len(CANDIDATES),
        "capacity_variants": list(CAPACITIES),
        "family_size": FAMILY_SIZE,
        "scheduled_sessions": SCHEDULED_SESSIONS,
        "confirmation_date_bounds": [
            str(CONFIRMATION_START.date()),
            str(CONFIRMATION_END.date()),
        ],
        "slots_per_model": SCHEDULED_SESSIONS * 2,
        "picks_rows": len(picks),
        "picks_semantic_sha256": semantic_hash(picks),
        "strictly_prior_features": True,
        "monthly_scoring_before_same_month_outcomes": True,
        "orders_allowed": False,
        "production_model_changed": False,
    }
    audit.tree(
        schedule_check,
        "result.integrity",
        integrity_expected,
        result["integrity"],
    )

    label_check = "label_return_sign"
    observed = picks["oc_return_pct"].notna()
    audit.equal(
        label_check,
        "return and label missingness",
        picks["oc_return_pct"].isna().tolist(),
        picks["label"].isna().tolist(),
    )
    expected_labels = picks.loc[observed, "oc_return_pct"].gt(0.0).astype(float)
    audit.equal(
        label_check,
        "close>open label including zero-return class 0",
        expected_labels.tolist(),
        picks.loc[observed, "label"].astype(float).tolist(),
    )
    audit.equal(
        label_check,
        "finite executed returns",
        True,
        bool(np.isfinite(picks.loc[observed, "oc_return_pct"]).all()),
    )
    return schedule


def audit_control_reproduction(
    audit: Audit,
    result: dict[str, Any],
    picks: pd.DataFrame,
    v13_picks: pd.DataFrame,
    schedule: pd.DatetimeIndex,
) -> dict[str, Any]:
    check = "v13_control_exact_reproduction"
    columns = list(PICK_COLUMNS)
    current = (
        picks.loc[picks["candidate_id"].eq(CONTROL), columns]
        .sort_values(["date", "model_rank"], kind="stable")
        .reset_index(drop=True)
    )
    reference = (
        v13_picks.loc[
            v13_picks["candidate_id"].eq(CONTROL)
            & v13_picks["date"].between(
                CONFIRMATION_START, CONFIRMATION_END, inclusive="both"
            ),
            columns,
        ]
        .sort_values(["date", "model_rank"], kind="stable")
        .reset_index(drop=True)
    )
    if not current.equals(reference):
        unequal = current.ne(reference) & ~(current.isna() & reference.isna())
        locations = [
            f"row={row}, column={column}"
            for row, column in zip(*np.where(unequal.to_numpy()))
        ][:20]
        audit.mismatch(
            check,
            "C00 confirmation rows versus v1.3",
            expected=True,
            observed=False,
            detail=", ".join(locations),
        )
    control_net_means: dict[str, dict[str, float]] = {}
    for capacity in CAPACITIES:
        capacity_key = f"top{capacity}"
        control_net_means[capacity_key] = {}
        for cost in COSTS_BPS:
            cost_key = str(int(cost))
            net_mean = float(
                daily_returns(
                    current,
                    schedule,
                    capacity=capacity,
                    cost_bps=cost,
                ).mean()
            )
            control_net_means[capacity_key][cost_key] = net_mean
            audit.close(
                check,
                (
                    f"result.control.metrics.{capacity_key}.{cost_key}"
                    ".net_mean_pct_at_cost"
                ),
                net_mean,
                result["control"]["metrics"][capacity_key][cost_key][
                    "net_mean_pct_at_cost"
                ],
            )
    return {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "reference_semantic_sha256": semantic_hash(reference),
        "current_semantic_sha256": semantic_hash(current),
        "all_columns_exact": current.equals(reference),
        "net_mean_pct_by_capacity_and_cost": control_net_means,
    }


def audit_variant(
    audit: Audit,
    stored: dict[str, Any],
    candidate: pd.DataFrame,
    control: pd.DataFrame,
    schedule: pd.DatetimeIndex,
) -> dict[str, Any]:
    metric_check = "cost_and_robustness_recalculation"
    concentration_check = "concentration_and_execution_recalculation"
    bootstrap_check = "familywise_bootstrap_recalculation"
    gate_check = "gate_recalculation"
    candidate_id = str(stored["candidate_id"])
    capacity = int(stored["capacity"])
    variant_id = f"{candidate_id}__top{capacity}"
    audit.equal(gate_check, f"{variant_id}.variant_id", variant_id, stored["variant_id"])

    daily_by_cost = {
        str(int(cost)): daily_returns(
            candidate,
            schedule,
            capacity=capacity,
            cost_bps=cost,
        )
        for cost in COSTS_BPS
    }
    for cost, daily in daily_by_cost.items():
        audit.close(
            metric_check,
            f"{variant_id}.cost_metrics.{cost}.net_mean_pct_at_cost",
            float(daily.mean()),
            stored["cost_metrics"][cost]["net_mean_pct_at_cost"],
        )
    daily40 = daily_by_cost["40"]
    control40 = daily_returns(
        control, schedule, capacity=capacity, cost_bps=40.0
    )
    slice_values = {
        name: float(daily40.loc[start:end].mean())
        for name, (start, end) in CONFIRMATION_SLICES.items()
    }
    audit.tree(
        metric_check,
        f"{variant_id}.confirmation_slice_net40_mean_pct",
        slice_values,
        stored["confirmation_slice_net40_mean_pct"],
    )
    top10_removed = float(daily40.drop(daily40.nlargest(10).index).mean())
    audit.close(
        metric_check,
        f"{variant_id}.top10_days_removed_net40_mean_pct",
        top10_removed,
        stored["top10_days_removed_net40_mean_pct"],
    )
    code_cash, top_profit_codes = top_profitable_codes_cash(
        candidate, schedule, capacity=capacity
    )
    code_cash_mean = float(code_cash.mean())
    audit.equal(
        metric_check,
        f"{variant_id}.top10_profit_codes",
        top_profit_codes,
        [str(code) for code in stored["top10_profit_codes"]],
    )
    audit.close(
        metric_check,
        f"{variant_id}.top10_profit_codes_cash_net40_mean_pct",
        code_cash_mean,
        stored["top10_profit_codes_cash_net40_mean_pct"],
    )
    monthly = daily40.groupby(daily40.index.to_period("M")).mean()
    monthly_values = {
        str(period): float(value) for period, value in monthly.items()
    }
    positive_months = int(monthly.gt(0.0).sum())
    audit.tree(
        metric_check,
        f"{variant_id}.monthly_net40_mean_pct",
        monthly_values,
        stored["monthly_net40_mean_pct"],
    )
    audit.equal(
        metric_check,
        f"{variant_id}.months",
        9,
        stored["months"],
    )
    audit.equal(
        metric_check,
        f"{variant_id}.positive_months_net40",
        positive_months,
        stored["positive_months_net40"],
    )

    selected = candidate.loc[candidate["model_rank"].le(capacity)].copy()
    signaled = selected.dropna(subset=["code"])
    counts = signaled["code"].astype(str).value_counts()
    signaled_count = int(len(signaled))
    unique_codes = int(len(counts))
    maximum_share = (
        float(counts.iloc[0] / signaled_count) if signaled_count else 1.0
    )
    top10_share = (
        float(counts.head(10).sum() / signaled_count)
        if signaled_count
        else 1.0
    )
    executed = selected["label"].notna()
    traded_days = int(
        selected.assign(_executed=executed)
        .groupby("date", sort=True)["_executed"]
        .any()
        .sum()
    )
    expected_slots = SCHEDULED_SESSIONS * capacity
    executed_slots = int(executed.sum())
    executed_fraction = float(executed_slots / expected_slots)
    concentration_values = {
        "unique_selected_codes": unique_codes,
        "maximum_code_selection_share": maximum_share,
        "top10_code_selection_share": top10_share,
        "traded_days": traded_days,
        "executed_slots": executed_slots,
        "expected_capacity_slots": expected_slots,
        "executed_slot_fraction": executed_fraction,
    }
    for field, value in concentration_values.items():
        if isinstance(value, float):
            audit.close(
                concentration_check,
                f"{variant_id}.{field}",
                value,
                stored[field],
            )
        else:
            audit.equal(
                concentration_check,
                f"{variant_id}.{field}",
                value,
                stored[field],
            )

    paired = paired_moving_block_bootstrap(
        daily40,
        control40,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        samples=BOOTSTRAP_SAMPLES,
        confidence=BONFERRONI_CONFIDENCE,
        random_state=BOOTSTRAP_RANDOM_STATE + CAPACITIES.index(capacity),
    )
    audit.tree(
        bootstrap_check,
        f"{variant_id}.paired_vs_control_net40",
        asdict(paired),
        stored["paired_vs_control_net40"],
    )
    audit.close(
        bootstrap_check,
        f"{variant_id}.bonferroni_individual_confidence",
        BONFERRONI_CONFIDENCE,
        stored["bonferroni_individual_confidence"],
    )

    checks = {
        "net40_positive": float(daily40.mean()) > 0.0,
        "net60_positive": float(daily_by_cost["60"].mean()) > 0.0,
        "all_three_confirmation_slices_net40_positive": all(
            value > 0.0 for value in slice_values.values()
        ),
        "top10_days_removed_net40_positive": top10_removed > 0.0,
        "top10_profit_codes_cash_net40_positive": code_cash_mean > 0.0,
        "positive_months_at_least_6_of_9": positive_months >= 6,
        "familywise_paired_lower_vs_control_nonnegative": (
            paired.one_sided_lower_delta_pct >= 0.0
        ),
        "unique_codes_at_least_100": unique_codes >= 100,
        "maximum_code_share_at_most_5pct": maximum_share <= 0.05,
        "top10_code_share_at_most_25pct": top10_share <= 0.25,
        "traded_days_at_least_150": traded_days >= 150,
        "executed_slot_fraction_at_least_80pct": executed_fraction >= 0.80,
    }
    audit.tree(
        gate_check,
        f"{variant_id}.gate_checks",
        checks,
        stored["gate_checks"],
    )
    gate_passed = all(checks.values())
    audit.equal(
        gate_check,
        f"{variant_id}.gate_passed",
        gate_passed,
        stored["gate_passed"],
    )
    return {
        "variant_id": variant_id,
        "net20_mean_pct": float(daily_by_cost["20"].mean()),
        "net40_mean_pct": float(daily40.mean()),
        "net60_mean_pct": float(daily_by_cost["60"].mean()),
        "confirmation_slice_net40_mean_pct": slice_values,
        "top10_days_removed_net40_mean_pct": top10_removed,
        "top10_profit_codes": top_profit_codes,
        "top10_profit_codes_cash_net40_mean_pct": code_cash_mean,
        "positive_months_net40": positive_months,
        "unique_selected_codes": unique_codes,
        "maximum_code_selection_share": maximum_share,
        "top10_code_selection_share": top10_share,
        "traded_days": traded_days,
        "executed_slots": executed_slots,
        "expected_capacity_slots": expected_slots,
        "executed_slot_fraction": executed_fraction,
        "paired_lower_delta_pct": paired.one_sided_lower_delta_pct,
        "gate_checks": checks,
        "gate_passed": gate_passed,
        "failed_checks": [
            name for name, passed in checks.items() if not passed
        ],
    }


def audit_fold_metadata(audit: Audit, result: dict[str, Any]) -> None:
    check = "strictly_prior_fold_metadata"
    expected_periods = [
        "2024-11",
        "2024-12",
        "2025-01",
        "2025-02",
        "2025-03",
        "2025-04",
        "2025-05",
        "2025-06",
        "2025-07",
    ]
    fold_groups = {
        "control": result["control"]["folds"],
        "SP01": result["models"]["SP01"]["folds"],
        "GN01": result["models"]["GN01"]["folds"],
    }
    for name, folds in fold_groups.items():
        audit.equal(
            check,
            f"{name}.periods",
            expected_periods,
            [str(fold["period"]) for fold in folds],
        )
        for fold in folds:
            period = pd.Period(str(fold["period"]), freq="M")
            audit.equal(
                check,
                f"{name}.{period}.strictly_prior_training",
                True,
                fold["strictly_prior_training"],
            )
            audit.equal(
                check,
                f"{name}.{period}.train_end before score month",
                True,
                pd.Timestamp(fold["train_end"]) < period.start_time,
            )
    for model in ("DMD01", "DMD02"):
        details = result["models"]["DMD"][model]
        fits = details["folds"]
        audit.equal(
            check,
            f"{model}.periods",
            expected_periods,
            [str(item["period"]) for item in fits],
        )
        for item in fits:
            period = pd.Period(str(item["period"]), freq="M")
            audit.equal(
                check,
                f"{model}.{period}.fit_status",
                "PASS",
                item["fit_status"],
            )
            audit.equal(
                check,
                f"{model}.{period}.strictly_prior_fit",
                True,
                item["strictly_prior_fit"],
            )
            audit.equal(
                check,
                f"{model}.{period}.operator_frozen_within_month",
                True,
                item["operator_frozen_within_month"],
            )
            audit.equal(
                check,
                f"{model}.{period}.fit_end before score month",
                True,
                pd.Timestamp(item["state_end"]) < period.start_time,
            )


def audit_decision_and_boundaries(
    audit: Audit,
    result: dict[str, Any],
    protocol: dict[str, Any],
    variants: list[dict[str, Any]],
) -> dict[str, Any]:
    decision_check = "decision_recalculation"
    passers = [
        item["variant_id"] for item in variants if item["gate_passed"]
    ]
    expected_decision = {
        "gate_passers": passers,
        "forward_shadow_candidate": (
            passers[0] if len(passers) == 1 else None
        ),
        "ambiguous_multiple_passers": len(passers) > 1,
        "production_candidate": None,
        "production_model_changed": False,
        "orders_allowed": False,
        "conclusion": (
            "One preregistered variant cleared every retrospective "
            "confirmation gate and may only be frozen for a new forward "
            "shadow."
            if len(passers) == 1
            else "No unique preregistered variant qualified for a "
            "forward shadow."
        ),
    }
    audit.tree(
        decision_check,
        "result.decision",
        expected_decision,
        result["decision"],
    )

    boundary_check = "production_and_order_boundary"
    boundary_values = {
        "protocol.authority.production_promotion_allowed": protocol[
            "authority"
        ]["production_promotion_allowed"],
        "protocol.authority.orders_allowed": protocol["authority"][
            "orders_allowed"
        ],
        "result.integrity.production_model_changed": result["integrity"][
            "production_model_changed"
        ],
        "result.integrity.orders_allowed": result["integrity"][
            "orders_allowed"
        ],
        "result.decision.production_candidate": result["decision"][
            "production_candidate"
        ],
        "result.decision.production_model_changed": result["decision"][
            "production_model_changed"
        ],
        "result.decision.orders_allowed": result["decision"]["orders_allowed"],
    }
    expected_boundaries = {
        "protocol.authority.production_promotion_allowed": False,
        "protocol.authority.orders_allowed": False,
        "result.integrity.production_model_changed": False,
        "result.integrity.orders_allowed": False,
        "result.decision.production_candidate": None,
        "result.decision.production_model_changed": False,
        "result.decision.orders_allowed": False,
    }
    audit.tree(
        boundary_check,
        "production_and_order_values",
        expected_boundaries,
        boundary_values,
    )
    return expected_decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        default="research/model_v14_confirmation_result.json",
    )
    parser.add_argument(
        "--picks",
        default="research/model_v14_confirmation_picks.csv",
    )
    parser.add_argument(
        "--output",
        default="research/model_v14_confirmation_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = Audit()
    variants: list[dict[str, Any]] = []
    control_reproduction: dict[str, Any] = {}
    expected_decision: dict[str, Any] = {}
    result_path = Path(args.result)
    picks_path = Path(args.picks)
    try:
        result = read_json(result_path)
        protocol = read_json(PROTOCOL)
        stage_a_result = read_json(STAGE_A_RESULT)
        v13_result = read_json(V13_RESULT)
        picks = load_picks(picks_path)
        v13_picks = load_picks(V13_PICKS)
        audit_bindings(
            audit,
            result,
            protocol,
            stage_a_result,
            v13_result,
            picks,
        )
        schedule = audit_schedule_and_labels(
            audit, result, picks, v13_picks
        )
        control_reproduction = audit_control_reproduction(
            audit, result, picks, v13_picks, schedule
        )
        stored_variants = {
            str(item["variant_id"]): item for item in result["variants"]
        }
        expected_variant_ids = [
            f"{candidate}__top{capacity}"
            for candidate in CANDIDATES
            for capacity in CAPACITIES
        ]
        audit.equal(
            "gate_recalculation",
            "result variant order and set",
            expected_variant_ids,
            [str(item["variant_id"]) for item in result["variants"]],
        )
        by_model = {
            model: picks.loc[picks["candidate_id"].eq(model)].copy()
            for model in ALL_MODELS
        }
        for candidate in CANDIDATES:
            for capacity in CAPACITIES:
                variant_id = f"{candidate}__top{capacity}"
                if variant_id not in stored_variants:
                    audit.mismatch(
                        "gate_recalculation",
                        variant_id,
                        expected="stored variant",
                        observed="missing",
                    )
                    continue
                variants.append(
                    audit_variant(
                        audit,
                        stored_variants[variant_id],
                        by_model[candidate],
                        by_model[CONTROL],
                        schedule,
                    )
                )
        audit_fold_metadata(audit, result)
        expected_decision = audit_decision_and_boundaries(
            audit, result, protocol, variants
        )
    except Exception as error:  # Keep a machine-readable FAIL artifact.
        audit.mismatch(
            "artifact_bindings",
            "audit execution",
            expected="completed without exception",
            observed=type(error).__name__,
            detail=str(error),
        )

    passed = not audit.mismatches
    output = {
        "schema_version": 1,
        "audit_id": "model_v14_confirmation_independent_audit_20260727",
        "status": "PASS" if passed else "FAIL",
        "audit_passed": passed,
        "result_sha256": (
            sha256_file(result_path) if result_path.is_file() else None
        ),
        "picks_file_sha256": (
            sha256_file(picks_path) if picks_path.is_file() else None
        ),
        "protocol_sha256": (
            sha256_file(PROTOCOL) if PROTOCOL.is_file() else None
        ),
        "runner_sha256": sha256_file(RUNNER) if RUNNER.is_file() else None,
        "audit_runner_sha256": sha256_file(__file__),
        "checks": {
            name: audit.status(name) for name in CHECK_NAMES
        },
        "mismatch_count": len(audit.mismatches),
        "mismatches": audit.mismatches,
        "schedule": {
            "expected_sessions": SCHEDULED_SESSIONS,
            "expected_slots_per_model": SCHEDULED_SESSIONS * 2,
            "expected_models": list(ALL_MODELS),
        },
        "control_reproduction": control_reproduction,
        "variants": variants,
        "recomputed_decision": expected_decision,
        "conclusion": (
            "All frozen v1.4 confirmation artifacts and decisions reproduce "
            "independently from the picks; production remains unchanged and "
            "orders remain disabled."
            if passed
            else "The independent v1.4 confirmation audit found one or more "
            "mismatches; the result must not be used."
        ),
    }
    write_json(output, args.output)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
