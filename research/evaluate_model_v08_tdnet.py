#!/usr/bin/env python3
"""Reproduce the retrospective v0.8 TDnet title-only feature screen.

This runner evaluates all twelve registered title/bundle groups with a
daily-rank Ridge(alpha=10) anchor and then runs the one-shot alpha=1 H08/H10
check.  It deliberately fails closed after the historical TDnet cache becomes
incomplete.  The output has no production-promotion authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research.screen_feature_candidates_v06 import (  # noqa: E402
    max_t_adjusted_uplifts,
)
from tse_session_ranker.data.common import (  # noqa: E402
    normalize_expected_sessions,
)
from tse_session_ranker.data.tdnet import (  # noqa: E402
    collect_tdnet_dataset,
)
from tse_session_ranker.io import read_frame, write_json  # noqa: E402
from tse_session_ranker.profit import (  # noqa: E402
    daily_portfolio_returns,
    profit_metrics,
)
from tse_session_ranker.research_candidates import (  # noqa: E402
    TDNET_CANDIDATE_COLUMNS,
    build_clean_tdnet_candidate_features,
)


PROTOCOL_PATH = ROOT / "research/model_v08_tdnet_protocol.json"
EXPECTED_PROTOCOL_SHA256 = (
    "6cca576b054de7abd2684137e11b2c9b01d489e1b62cb7ce780375868d0ce4a8"
)
DEFAULT_PANEL_MANIFEST = ROOT / "research/model_v05_panel_manifest.json"
DEFAULT_PANEL_CACHE = Path("/tmp/model_v05_broad_family_panel_r2.pkl")
DEFAULT_TDNET_CACHE = ROOT / "research/.cache/model_v05_tdnet"
DEFAULT_OUTPUT = ROOT / "research/model_v08_tdnet_result.json"
DEFAULT_MANIFEST_OUTPUT = ROOT / "research/model_v08_tdnet_result.manifest.json"
FULL_START = pd.Timestamp("2024-07-01")
FULL_END = pd.Timestamp("2025-04-07")
TRAIN_START = pd.Timestamp("2024-01-04")
PRIMARY_COST_BPS = 20.0
STRESS_COST_BPS = 40.0
TOP_K = 2


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _load_protocol(path: Path) -> dict[str, Any]:
    if path.resolve() == PROTOCOL_PATH.resolve():
        actual = sha256_file(path)
        if actual != EXPECTED_PROTOCOL_SHA256:
            raise ValueError("canonical v0.8 TDnet protocol changed")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "model_v08_tdnet_title_screen_20260722":
        raise ValueError("unexpected v0.8 TDnet protocol id")
    if protocol.get("authority", {}).get("production_promotion_allowed") is not False:
        raise ValueError("v0.8 protocol must remain retrospective-only")
    _validate_historical_bindings(protocol)
    return protocol


def _validate_historical_bindings(protocol: Mapping[str, Any]) -> None:
    bindings = protocol.get("historical_precommit_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError("historical precommit bindings are missing")
    for generation, raw_binding in bindings.items():
        if not isinstance(raw_binding, Mapping):
            raise ValueError(f"invalid historical binding: {generation}")
        stable_path = raw_binding.get("stable_path")
        expected_sha256 = raw_binding.get("sha256")
        if not isinstance(stable_path, str) or not stable_path:
            raise ValueError(f"stable path is missing: {generation}")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(f"invalid bound hash: {generation}")
        resolved = (ROOT / stable_path).resolve()
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError as error:
            raise ValueError(
                f"historical stable path escapes the repository: {generation}"
            ) from error
        if not resolved.is_file():
            raise FileNotFoundError(
                f"historical stable artifact is missing: {generation}"
            )
        if sha256_file(resolved) != expected_sha256:
            raise ValueError(
                f"historical stable artifact changed: {generation}"
            )


def _monthly_periods(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    output: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for period in pd.period_range(start.to_period("M"), end.to_period("M"), freq="M"):
        output.append(
            (
                max(start, period.start_time.normalize()),
                min(end, period.end_time.normalize()),
                str(period),
            )
        )
    return output


def _add_v08_derived_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    fresh = frame["tdnet_v07_fresh_classified_economic_any"]
    follow = frame["tdnet_v07_followup_family_count"].gt(0).astype(float)
    economic_count = frame["tdnet_v07_economic_family_count"]
    frame["tdnet_v08_economic_count_ge_2"] = economic_count.ge(2).astype(float)
    frame["tdnet_v08_economic_count_ge_3"] = economic_count.ge(3).astype(float)
    frame["tdnet_v08_fresh_only"] = fresh * (1.0 - follow)
    frame["tdnet_v08_followup_only"] = follow * (1.0 - fresh)
    frame["tdnet_v08_fresh_and_followup"] = fresh * follow
    width = frame["tdnet_clean_bundle_width_minutes_log1p"]
    documents = frame["tdnet_clean_document_count_log1p"]
    frame["tdnet_v08_bundle_intensity"] = documents / (1.0 + width)
    frame["tdnet_v08_repeat_pressure"] = (
        frame["tdnet_clean_prior_bundle_count_60_log1p"]
        - np.log1p(60.0 / 252.0)
        - frame["tdnet_clean_prior_bundle_count_252_log1p"]
    )
    momentum20 = frame["xrank_close_momentum_20"]
    momentum60 = frame["xrank_close_momentum_60"]
    frame["tdnet_v08_fresh_x_momentum20"] = fresh * momentum20
    frame["tdnet_v08_fresh_x_momentum60"] = fresh * momentum60
    frame["tdnet_v08_followup_x_momentum20"] = follow * momentum20
    frame["tdnet_v08_followup_x_momentum60"] = follow * momentum60
    frame["tdnet_v08_fresh_x_negative_momentum20"] = fresh * (
        -momentum20
    ).clip(lower=0)
    frame["tdnet_v08_fresh_x_positive_momentum20"] = fresh * momentum20.clip(
        lower=0
    )
    frame["tdnet_v08_fresh_x_premarket"] = (
        fresh * frame["tdnet_clean_premarket_count_log1p"]
    )
    frame["tdnet_v08_fresh_x_postclose"] = (
        fresh * frame["tdnet_clean_postclose_count_log1p"]
    )
    frame["tdnet_v08_fresh_x_age"] = (
        fresh * frame["tdnet_clean_latest_age_hours_log1p"]
    )
    frame["tdnet_v08_followup_x_age"] = (
        follow * frame["tdnet_clean_latest_age_hours_log1p"]
    )
    return frame


def _prepare_panel(
    *,
    panel_cache: Path,
    panel_manifest: Mapping[str, Any],
    tdnet_cache: Path,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, Any]]:
    if sha256_file(panel_cache) != panel_manifest["panel_file_sha256"]:
        raise ValueError("broad-family panel cache differs from its manifest")
    panel = read_frame(panel_cache)
    sessions = normalize_expected_sessions(panel_manifest["sessions"])
    base_features = list(protocol["shared_model"]["base_features"])
    required = {
        "date",
        "code",
        "name",
        "label",
        "oc_return_pct",
        "outcome_observed",
        "price_eligible",
        "price_training_eligible",
        "tdnet_source_complete",
        *base_features,
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel lacks v0.8 required columns: {missing}")
    dataset = collect_tdnet_dataset(
        tdnet_cache, allow_historical_provenance=True
    )
    expected_source = panel_manifest["provenance"]["tdnet_source_sha256"]
    if dataset.source_sha256 != expected_source:
        raise ValueError("TDnet logical source differs from panel manifest")
    features = build_clean_tdnet_candidate_features(
        dataset.disclosures,
        sessions,
        decision_time=protocol["decision"]["time_jst"],
    )
    frame = panel[
        list(
            dict.fromkeys(
                [
                    "date",
                    "code",
                    "name",
                    "label",
                    "oc_return_pct",
                    "outcome_observed",
                    "price_eligible",
                    "price_training_eligible",
                    "tdnet_source_complete",
                    *base_features,
                ]
            )
        )
    ].copy()
    frame = frame.merge(
        features,
        on=["date", "code"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    complete = frame["tdnet_source_complete"].eq(True)
    frame.loc[complete, list(TDNET_CANDIDATE_COLUMNS)] = frame.loc[
        complete, list(TDNET_CANDIDATE_COLUMNS)
    ].fillna(0.0)
    frame.loc[~complete, list(TDNET_CANDIDATE_COLUMNS)] = np.nan
    frame = _add_v08_derived_features(frame)
    requested = sessions[
        (sessions >= pd.Timestamp(protocol["data"]["requested_score_window"][0]))
        & (sessions <= pd.Timestamp(protocol["data"]["requested_score_window"][1]))
    ]
    date_complete = (
        frame[["date", "tdnet_source_complete"]]
        .drop_duplicates("date")
        .set_index("date")["tdnet_source_complete"]
        .eq(True)
    )
    common = requested[requested.isin(date_complete[date_complete].index)]
    if len(requested) != protocol["data"]["requested_scheduled_sessions"]:
        raise ValueError("requested session count changed")
    if len(common) != protocol["data"]["tdnet_source_complete_sessions"]:
        raise ValueError("TDnet complete-session count changed")
    if common.max() != pd.Timestamp(
        protocol["data"]["last_tdnet_source_complete_session"]
    ):
        raise ValueError("last TDnet-complete session changed")
    source = {
        "documents": int(len(dataset.disclosures)),
        "source_files": int(dataset.source_files),
        "logical_source_sha256": dataset.source_sha256,
        "requested_sessions": int(len(requested)),
        "complete_sessions": int(len(common)),
        "complete_fraction": float(len(common) / len(requested)),
        "last_complete_session": str(common.max().date()),
    }
    return frame, common, source


def _fit_daily_rank_ridge(
    training: pd.DataFrame,
    columns: Sequence[str],
    *,
    alpha: float,
) -> Pipeline:
    target = (
        training["oc_return_pct"]
        .groupby(training["date"], sort=False)
        .rank(method="average", pct=True)
        .mul(2.0)
        .sub(1.0)
    )
    weights = 1.0 / training.groupby("date", sort=False)["date"].transform(
        "size"
    )
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )
    model.fit(
        training.loc[:, list(columns)],
        target,
        model__sample_weight=weights,
    )
    return model


def _desired_slots(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.MultiIndex.from_product(
        [sessions, range(1, TOP_K + 1)], names=["date", "model_rank"]
    ).to_frame(index=False)


def _evaluate_recipe(
    panel: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    recipe_id: str,
    columns: Sequence[str],
    alpha: float,
    base_feature_count: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for score_start, score_end, period_name in _monthly_periods(
        FULL_START, FULL_END
    ):
        training = panel[
            panel["date"].between(
                TRAIN_START, score_start - pd.Timedelta(days=1)
            )
            & panel["price_training_eligible"].eq(True)
            & panel["tdnet_source_complete"].eq(True)
            & panel["label"].notna()
        ].copy()
        scoring = panel[
            panel["date"].between(score_start, score_end)
            & panel["price_eligible"].eq(True)
            & panel["tdnet_source_complete"].eq(True)
        ].copy()
        if not training["date"].max() < scoring["date"].min():
            raise AssertionError("training and scoring dates overlap")
        fitted = _fit_daily_rank_ridge(training, columns, alpha=alpha)
        scoring["model_score"] = fitted.predict(scoring.loc[:, list(columns)])
        ranked = scoring.sort_values(
            ["date", "model_score", "code"],
            ascending=[True, False, True],
            kind="stable",
        )
        ranked = ranked.groupby("date", sort=True, as_index=False).head(TOP_K)
        ranked["model_rank"] = ranked.groupby("date", sort=False).cumcount() + 1
        parts.append(ranked)
        coefficients = np.asarray(fitted.named_steps["model"].coef_, dtype=float)
        folds.append(
            {
                "period": period_name,
                "train_start": str(training["date"].min().date()),
                "train_end": str(training["date"].max().date()),
                "train_rows": int(len(training)),
                "train_sessions": int(training["date"].nunique()),
                "score_rows": int(len(scoring)),
                "score_sessions": int(scoring["date"].nunique()),
                "added_standardized_coefficients": {
                    column: float(coefficients[position])
                    for position, column in enumerate(columns)
                    if position >= base_feature_count
                },
            }
        )
    actual = pd.concat(parts, ignore_index=True)
    keep = [
        "date",
        "model_rank",
        "code",
        "name",
        "model_score",
        "label",
        "oc_return_pct",
    ]
    picks = _desired_slots(sessions).merge(
        actual[keep],
        on=["date", "model_rank"],
        how="left",
        validate="one_to_one",
        sort=True,
    )
    picks["recipe_id"] = recipe_id
    return picks, folds


def _daily_net(picks: pd.DataFrame, cost_bps: float) -> pd.Series:
    return (
        daily_portfolio_returns(picks, top_k=TOP_K, cost_bps=cost_bps)
        .set_index("date")["net_return_pct"]
        .sort_index()
    )


def _effective_changes(
    candidate: pd.DataFrame, baseline: pd.DataFrame
) -> dict[str, Any]:
    changed_dates: list[str] = []
    added_slots = 0
    for date, base_day in baseline.groupby("date", sort=True):
        candidate_day = candidate[candidate["date"].eq(date)]
        base_codes = set(base_day["code"].dropna().astype(str))
        candidate_codes = set(candidate_day["code"].dropna().astype(str))
        added = candidate_codes - base_codes
        if added:
            changed_dates.append(str(pd.Timestamp(date).date()))
            added_slots += len(added)
    return {
        "days": len(changed_dates),
        "added_slots": added_slots,
        "dates": changed_dates,
    }


def _period_metrics(
    picks: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, Any]:
    delta = _daily_net(picks, PRIMARY_COST_BPS) - _daily_net(
        baseline, PRIMARY_COST_BPS
    )
    return {
        "net20": profit_metrics(
            picks, top_k=TOP_K, cost_bps=PRIMARY_COST_BPS
        ),
        "net40": profit_metrics(
            picks, top_k=TOP_K, cost_bps=STRESS_COST_BPS
        ),
        "uplift_net20_pct": float(delta.mean()),
        "effective_changes": _effective_changes(picks, baseline),
    }


def _active_exposure(
    panel: pd.DataFrame,
    columns: Sequence[str],
    *,
    start: str,
    end: str,
) -> dict[str, Any]:
    rows = panel[
        panel["date"].between(start, end)
        & panel["price_eligible"].eq(True)
        & panel["tdnet_source_complete"].eq(True)
    ][["date", *columns]]
    active = (
        rows.loc[:, list(columns)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .abs()
        .gt(1e-12)
        .any(axis=1)
    )
    return {
        "active_rows": int(active.sum()),
        "eligible_rows": int(len(rows)),
        "active_row_fraction": float(active.mean()),
        "active_days": int(rows.loc[active, "date"].nunique()),
    }


def _coefficient_stability(
    folds: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for column in columns:
        values = np.asarray(
            [
                float(fold["added_standardized_coefficients"][column])
                for fold in folds
            ],
            dtype=float,
        )
        signs = np.sign(values)
        output[column] = {
            "positive_fraction": float((values > 0).mean()),
            "adjacent_sign_flips": int((signs[1:] != signs[:-1]).sum()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
            "standard_deviation": float(values.std()),
        }
    return {
        "feature_count": len(output),
        "features_with_any_sign_flip": sum(
            detail["adjacent_sign_flips"] > 0 for detail in output.values()
        ),
        "features": output,
    }


def _bundle_ambiguity(panel: pd.DataFrame) -> dict[str, Any]:
    scope = panel[
        panel["date"].between(FULL_START, FULL_END)
        & panel["price_eligible"].eq(True)
        & panel["tdnet_source_complete"].eq(True)
    ]
    events = scope[scope["tdnet_v07_observed_any"].eq(1.0)]
    multi_document = events["tdnet_clean_document_count_log1p"].ge(
        math.log(3.0) - 1e-7
    )
    multi_family = events["tdnet_v07_economic_family_count"].ge(2.0)
    stage_conflict = events["tdnet_v07_fresh_classified_economic_any"].eq(
        1.0
    ) & events["tdnet_v07_followup_family_count"].gt(0.0)
    capital_conflict = (
        events["tdnet_v07_has_fresh_buyback"].eq(1.0)
        | events["tdnet_v07_has_fresh_share_cancellation"].eq(1.0)
    ) & events["tdnet_v07_has_fresh_equity"].eq(1.0)
    return {
        "eligible_rows": int(len(scope)),
        "observed_event_rows": int(len(events)),
        "event_row_fraction": float(len(events) / len(scope)),
        "fresh_economic_rows": int(
            events["tdnet_v07_fresh_classified_economic_any"].eq(1.0).sum()
        ),
        "multi_document_rows": int(multi_document.sum()),
        "multi_document_fraction": float(multi_document.mean()),
        "multi_economic_family_rows": int(multi_family.sum()),
        "multi_economic_family_fraction": float(multi_family.mean()),
        "fresh_and_followup_rows": int(stage_conflict.sum()),
        "fresh_and_followup_fraction": float(stage_conflict.mean()),
        "support_adverse_conflict_rows": int(
            events["tdnet_clean_support_adverse_conflict"].eq(1.0).sum()
        ),
        "fresh_capital_positive_and_equity_rows": int(capital_conflict.sum()),
    }


def _evaluate_generation(
    panel: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    base_features: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    alpha: float,
    generation_id: str,
    source_fraction: float,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    picks: dict[str, pd.DataFrame] = {}
    folds: dict[str, list[dict[str, Any]]] = {}
    baseline_name = f"{generation_id}_G0"
    picks[baseline_name], folds[baseline_name] = _evaluate_recipe(
        panel,
        sessions,
        recipe_id=baseline_name,
        columns=base_features,
        alpha=alpha,
        base_feature_count=len(base_features),
    )
    for group_name, columns in groups.items():
        recipe_name = f"{generation_id}_{group_name}"
        feature_columns = list(dict.fromkeys([*base_features, *columns]))
        picks[group_name], folds[group_name] = _evaluate_recipe(
            panel,
            sessions,
            recipe_id=recipe_name,
            columns=feature_columns,
            alpha=alpha,
            base_feature_count=len(base_features),
        )
    periods = protocol["periods"]
    baseline = picks[baseline_name]
    result: dict[str, Any] = {
        "alpha": alpha,
        "baseline": {"periods": {}, "folds": folds[baseline_name]},
        "candidates": {},
    }
    for period_name, bounds in periods.items():
        start, end = bounds
        period_base = baseline[baseline["date"].between(start, end)]
        result["baseline"]["periods"][period_name] = _period_metrics(
            period_base, period_base
        )
    adjusted_periods = ("full", "exploration", "confirmation")
    adjusted: dict[str, dict[str, dict[str, float]]] = {}
    for period_name in adjusted_periods:
        start, end = periods[period_name]
        period_base = baseline[baseline["date"].between(start, end)]
        period_candidates = {
            name: frame[frame["date"].between(start, end)]
            for name, frame in picks.items()
            if name != baseline_name
        }
        adjusted[period_name] = max_t_adjusted_uplifts(
            period_base, period_candidates
        )
    for group_name, columns in groups.items():
        candidate = picks[group_name]
        candidate_result: dict[str, Any] = {
            "features": list(columns),
            "periods": {},
            "folds": folds[group_name],
            "coefficient_stability": _coefficient_stability(
                folds[group_name], columns
            ),
        }
        for period_name, bounds in periods.items():
            start, end = bounds
            period_candidate = candidate[candidate["date"].between(start, end)]
            period_base = baseline[baseline["date"].between(start, end)]
            candidate_result["periods"][period_name] = _period_metrics(
                period_candidate, period_base
            )
            if period_name in adjusted:
                candidate_result["periods"][period_name][
                    "max_t_adjusted"
                ] = adjusted[period_name][group_name]
        if generation_id == "generation_3":
            candidate_result["exposure"] = {
                "exploration": _active_exposure(
                    panel,
                    columns,
                    start=periods["exploration"][0],
                    end=periods["exploration"][1],
                ),
                "confirmation": _active_exposure(
                    panel,
                    columns,
                    start=periods["confirmation"][0],
                    end=periods["confirmation"][1],
                ),
                "full": _active_exposure(
                    panel,
                    columns,
                    start=periods["full"][0],
                    end=periods["full"][1],
                ),
            }
        full = candidate_result["periods"]["full"]
        exploration = candidate_result["periods"]["exploration"]
        confirmation = candidate_result["periods"]["confirmation"]
        full_net20 = full["net20"]
        checks = {
            "exploration_uplift_positive": exploration[
                "uplift_net20_pct"
            ]
            > 0,
            "confirmation_uplift_positive": confirmation[
                "uplift_net20_pct"
            ]
            > 0,
            "full_net20_positive": full_net20["net_mean_pct_at_cost"] > 0,
            "full_net40_nonnegative": full["net40"][
                "net_mean_pct_at_cost"
            ]
            >= 0,
            "full_top5_removed_positive": full_net20[
                "top5_removed_net_mean_pct"
            ]
            > 0,
            "positive_month_fraction_at_least_0_6": (
                full_net20["positive_months"] / max(1, full_net20["months"])
            )
            >= 0.6,
            "max_t_adjusted_lower_nonnegative": full["max_t_adjusted"][
                "adjusted_one_sided_80pct_lower_pct"
            ]
            >= 0,
            "effective_change_days_at_least_30": full[
                "effective_changes"
            ]["days"]
            >= 30,
            "requested_source_coverage_at_least_0_98": source_fraction
            >= 0.98,
        }
        if generation_id == "generation_3":
            checks.update(
                {
                    "exploration_active_days_at_least_30": candidate_result[
                        "exposure"
                    ]["exploration"]["active_days"]
                    >= 30,
                    "confirmation_active_days_at_least_30": candidate_result[
                        "exposure"
                    ]["confirmation"]["active_days"]
                    >= 30,
                }
            )
        candidate_result["decision"] = {
            "checks": checks,
            "qualified": all(checks.values()),
        }
        result["candidates"][group_name] = candidate_result
    result["qualified"] = [
        name
        for name, detail in result["candidates"].items()
        if detail["decision"]["qualified"]
    ]
    return result, picks


def _implementation_binding() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src/tse_session_ranker/research_candidates.py",
        ROOT / "src/tse_session_ranker/data/tdnet.py",
        ROOT / "src/tse_session_ranker/profit.py",
    )
    records = {
        _portable_path(path): sha256_file(path)
        for path in paths
    }
    digest = hashlib.sha256()
    for name, value in sorted(records.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return {"files": records, "aggregate_sha256": digest.hexdigest()}


def _write_manifest(
    *,
    output_path: Path,
    manifest_path: Path,
    protocol_path: Path,
    panel_manifest_path: Path,
    panel_cache: Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "production_promotion_allowed": False,
        "production_changed": False,
        "result_path": _portable_path(output_path),
        "result_sha256": sha256_file(output_path),
        "protocol_path": _portable_path(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "runner_path": _portable_path(Path(__file__)),
        "runner_sha256": sha256_file(Path(__file__)),
        "panel_manifest_path": _portable_path(panel_manifest_path),
        "panel_manifest_sha256": sha256_file(panel_manifest_path),
        "panel_cache_path": _portable_path(panel_cache),
        "panel_cache_sha256": sha256_file(panel_cache),
        "tdnet_logical_source_sha256": source["logical_source_sha256"],
        "implementation": _implementation_binding(),
    }
    write_json(manifest, manifest_path)
    return manifest


def evaluate(
    *,
    panel_cache: Path,
    tdnet_cache: Path,
    panel_manifest_path: Path,
    protocol_path: Path,
    output_path: Path,
    manifest_output_path: Path,
) -> dict[str, Any]:
    protocol = _load_protocol(protocol_path)
    panel_manifest = json.loads(
        panel_manifest_path.read_text(encoding="utf-8")
    )
    panel, sessions, source = _prepare_panel(
        panel_cache=panel_cache,
        panel_manifest=panel_manifest,
        tdnet_cache=tdnet_cache,
        protocol=protocol,
    )
    base_features = list(protocol["shared_model"]["base_features"])
    groups = {
        str(name): [str(column) for column in columns]
        for name, columns in protocol["feature_groups"].items()
    }
    generation_3, _ = _evaluate_generation(
        panel,
        sessions,
        base_features=base_features,
        groups=groups,
        alpha=10.0,
        generation_id="generation_3",
        source_fraction=source["complete_fraction"],
        protocol=protocol,
    )
    generation_4_groups = {
        "H08_arrival_timing": groups["H08_arrival_timing"],
        "H10_stage_linear_runup": groups["H10_stage_linear_runup"],
        "H08_H10": [
            *groups["H08_arrival_timing"],
            *groups["H10_stage_linear_runup"],
        ],
    }
    generation_4, _ = _evaluate_generation(
        panel,
        sessions,
        base_features=base_features,
        groups=generation_4_groups,
        alpha=1.0,
        generation_id="generation_4",
        source_fraction=source["complete_fraction"],
        protocol=protocol,
    )
    diagnostics = {
        "coverage": source,
        "sparsity": {
            name: generation_3["candidates"][name]["exposure"]["full"]
            for name in groups
        },
        "bundle_ambiguity": _bundle_ambiguity(panel),
        "coefficient_instability": {
            name: generation_3["candidates"][name][
                "coefficient_stability"
            ]
            for name in groups
        },
    }
    stop_triggered = not generation_3["qualified"] and not generation_4[
        "qualified"
    ]
    result = {
        "schema_version": 1,
        "research_id": "model_v08_tdnet_title_screen_20260722",
        "scope": {
            "authority": "retrospective_diagnostic_only",
            "production_changed": False,
            "decision_time_jst": protocol["decision"]["time_jst"],
            "target": protocol["decision"]["target"],
            "title_only": True,
        },
        "bindings": {
            "protocol_path": _portable_path(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "panel_manifest_path": _portable_path(panel_manifest_path),
            "panel_manifest_sha256": sha256_file(panel_manifest_path),
            "panel_cache_path": _portable_path(panel_cache),
            "panel_cache_sha256": sha256_file(panel_cache),
            "tdnet_logical_source_sha256": source[
                "logical_source_sha256"
            ],
            "historical_precommit_bindings": protocol[
                "historical_precommit_bindings"
            ],
            "implementation": _implementation_binding(),
        },
        "data_coverage": source,
        "generation_3_alpha10": generation_3,
        "generation_4_alpha1": generation_4,
        "diagnostics": diagnostics,
        "iteration_history": {
            "generation_1": {
                "summary": "All 12 outcome-blind title hypotheses were screened with the common raked-logit G0 anchor; none qualified.",
                "best_confirmation_observations": "H10/H11 improvement came from only a few changed slots, including IK Holdings and Shannon, while direct event cohorts remained negative after 20 bp.",
            },
            "generation_2": {
                "summary": "Event-specialist reranking improved G0 by 0.128668 percentage points per day but remained -0.183220 percent per day after 20 bp; every fixed overlay/veto recipe failed.",
                "execution_incidents": [
                    "The first attempt stopped before metric generation because cash-veto feature columns had been projected out.",
                    "The second attempt stopped before metric generation because one candidate-row concatenation coerced the date column to object dtype.",
                    "Both implementation defects were corrected without changing the bound generation-2 protocol; the third run completed.",
                ],
            },
            "generation_3": {
                "summary": "All 12 groups were rescreened with daily-rank Ridge(alpha=10); no group qualified.",
            },
            "generation_4": {
                "summary": "The one-shot Ridge(alpha=1) H08, H10 and H08+H10 recipes all underperformed their alpha=1 G0 baseline; no recipe qualified.",
            },
        },
        "limitations": [
            "Every outcome is historically known; time ordering prevents direct leakage but does not create a sealed holdout.",
            "Only 187 of 266 requested sessions have point-in-time-complete TDnet input, below the registered 98 percent gate.",
            "The cache contains titles and timestamps, not disclosure PDF-body quantities, transaction magnitudes or forecast deltas.",
            "Exact historical 08:58 board, indicative open, PTS, futures, volume and turnover snapshots are unavailable.",
            "Generation-1 and generation-2 findings were used only to formulate explicitly posthoc diagnostic generations; they have no promotion authority.",
        ],
        "verdict": {
            "generation_3_qualified": generation_3["qualified"],
            "generation_4_qualified": generation_4["qualified"],
            "stop_rule_triggered": stop_triggered,
            "title_only_incremental_value": "not_demonstrated",
            "production_change": "none",
            "next_required_evidence": "prospective complete TDnet capture with PDF-body economic magnitudes and complete 08:58 market snapshots",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    write_json(result, output_path)
    _write_manifest(
        output_path=output_path,
        manifest_path=manifest_output_path,
        protocol_path=protocol_path,
        panel_manifest_path=panel_manifest_path,
        panel_cache=panel_cache,
        source=source,
    )
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-cache", default=str(DEFAULT_PANEL_CACHE))
    parser.add_argument("--tdnet-cache", default=str(DEFAULT_TDNET_CACHE))
    parser.add_argument(
        "--panel-manifest", default=str(DEFAULT_PANEL_MANIFEST)
    )
    parser.add_argument("--protocol", default=str(PROTOCOL_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--manifest-output", default=str(DEFAULT_MANIFEST_OUTPUT)
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = evaluate(
        panel_cache=Path(args.panel_cache).resolve(),
        tdnet_cache=Path(args.tdnet_cache).resolve(),
        panel_manifest_path=Path(args.panel_manifest).resolve(),
        protocol_path=Path(args.protocol).resolve(),
        output_path=Path(args.output).resolve(),
        manifest_output_path=Path(args.manifest_output).resolve(),
    )
    print(
        json.dumps(
            {
                "generation_3_qualified": result["verdict"][
                    "generation_3_qualified"
                ],
                "generation_4_qualified": result["verdict"][
                    "generation_4_qualified"
                ],
                "stop_rule_triggered": result["verdict"][
                    "stop_rule_triggered"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
