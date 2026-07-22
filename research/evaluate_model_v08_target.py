#!/usr/bin/env python3
"""Reproduce the locked v0.8 shadow finalists on the frozen research panel.

This runner intentionally does not reselect among the exploratory candidates.
Those already-viewed outcomes are retained in ``model_v08_target_result.json``.
The command below independently rebuilds only the fixed G0 control, L4
mean-return shadow role and L6 robustness shadow role.

Example
-------
PYTHONPATH=src:. python research/evaluate_model_v08_target.py \
  --panel /tmp/model_v07_corrected_panel.pkl \
  --output /tmp/model_v08_target_reproduction.json \
  --picks-output /tmp/model_v08_target_final_picks.csv \
  --error-output /tmp/model_v08_target_error_cases.csv
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tse_session_ranker.data.common import (  # noqa: E402
    normalize_expected_sessions,
    session_calendar_hash,
)
from tse_session_ranker.exceptions import DataValidationError  # noqa: E402
from tse_session_ranker.io import write_frame, write_json  # noqa: E402
from tse_session_ranker.profit import (  # noqa: E402
    daily_portfolio_returns,
    profit_metrics,
)
from tse_session_ranker.research_models import (  # noqa: E402
    FittedResearchScorer,
    ResearchModelSpec,
    fit_research_model,
)


DEFAULT_PROTOCOL = ROOT / "research/model_v08_target_protocol.json"
LOCKED_PROTOCOL_SHA256 = (
    "14f660e59d43818a2a0150df6614c6665d017c5fc82ba4b9b0f037d08f47b12d"
)
EXPECTED_PROTOCOL_ID = "model_v08_zero_base_rank_target_20260722"
BASE_FEATURES = (
    "oc_last", "oc_mean_5", "oc_mean_20", "oc_mean_60", "oc_win_20",
    "oc_std_20", "overnight_last", "overnight_mean_20", "overnight_mean_60",
    "night_day_corr_60", "xrank_atr14_pct", "xrank_close_momentum_5",
    "xrank_close_momentum_20", "xrank_close_momentum_60",
    "xrank_prior_close_location_20",
)
G5_FEATURES = (
    "no_trade_rate_20", "no_trade_rate_60", "flat_oc_rate_20",
    "zero_range_rate_20",
)
TOP_K = 2


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit_rank_ridge(
    training: pd.DataFrame,
    columns: Sequence[str],
    *,
    alpha: float = 1.0,
) -> FittedResearchScorer:
    """Fit the core daily-percentile Ridge implementation without local drift."""

    return fit_research_model(
        ResearchModelSpec(
            name=f"model_v08_rank_ridge_alpha_{alpha:g}",
            family="ridge_daily_rank",
            objective="same_day_return_percentile",
            parameters={"alpha": float(alpha)},
        ),
        training,
        tuple(columns),
    )


def _rank_two(
    scoring: pd.DataFrame, scores: np.ndarray, candidate_id: str
) -> pd.DataFrame:
    ranked = scoring.assign(model_score=np.asarray(scores, dtype=float)).sort_values(
        ["date", "model_score", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranked = ranked.groupby("date", sort=True, as_index=False).head(TOP_K).copy()
    ranked["model_rank"] = ranked.groupby("date", sort=False).cumcount() + 1
    ranked["candidate_id"] = candidate_id
    return ranked


def _desired_slots(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.MultiIndex.from_product(
        [sessions, range(1, TOP_K + 1)], names=["date", "model_rank"]
    ).to_frame(index=False)


def _manual_daily(picks: pd.DataFrame, cost_bps: float) -> pd.Series:
    executed = picks["label"].notna()
    slot = picks["oc_return_pct"].fillna(0.0) - (
        executed.astype(float) * cost_bps / 100.0
    )
    return slot.groupby(picks["date"], sort=True).sum().div(TOP_K)


def _candidate_metrics(picks: pd.DataFrame) -> dict[str, Any]:
    daily20 = _manual_daily(picks, 20.0)
    monthly = daily20.groupby(daily20.index.to_period("M")).mean()
    output: dict[str, Any] = {
        "cost": {
            str(int(cost)): profit_metrics(picks, top_k=TOP_K, cost_bps=cost)
            for cost in (20.0, 40.0, 60.0)
        },
        "winning_days_removed_net20": {
            str(count): float(daily20.drop(daily20.nlargest(count).index).mean())
            for count in (5, 10, 20)
        },
        "monthly_net20": {
            str(period): float(value) for period, value in monthly.items()
        },
    }
    executed = picks["label"].notna()
    slots = picks.assign(
        net20_slot=picks["oc_return_pct"].fillna(0.0)
        - executed.astype(float) * 0.20
    )
    by_code = slots.groupby(["code", "name"], dropna=False).agg(
        selections=("date", "size"),
        net20_slot_sum=("net20_slot", "sum"),
    ).sort_values("net20_slot_sum", ascending=False)
    total = float(by_code["net20_slot_sum"].sum())
    output["code_concentration"] = {
        "unique_codes": int(picks["code"].nunique()),
        "max_selections_one_code": int(by_code["selections"].max()),
        "largest_code_share_of_total_net_slot_pnl": (
            float(by_code["net20_slot_sum"].iloc[0] / total) if total > 0 else None
        ),
    }
    return output


def _locked_sha256(path: Path, expected: str, artifact: str) -> str:
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise DataValidationError(f"v0.8 {artifact} cannot be read") from exc
    if actual != expected:
        raise DataValidationError(
            f"v0.8 {artifact} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _read_json_object(path: Path, artifact: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"v0.8 {artifact} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DataValidationError(f"v0.8 {artifact} must be a JSON object")
    return value


def _load_locked_protocol(path: Path) -> dict[str, Any]:
    _locked_sha256(path, LOCKED_PROTOCOL_SHA256, "protocol")
    protocol = _read_json_object(path, "protocol")
    if protocol.get("protocol_id") != EXPECTED_PROTOCOL_ID:
        raise DataValidationError("unexpected v0.8 protocol")
    if protocol.get("authority", {}).get("production_promotion_allowed") is not False:
        raise DataValidationError("v0.8 retrospective runner cannot promote production")
    return protocol


def _load_locked_manifest(
    path: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    expected = str(protocol.get("frozen_input", {}).get("panel_manifest_sha256", ""))
    if len(expected) != 64:
        raise DataValidationError("v0.8 protocol lacks a panel-manifest SHA-256")
    _locked_sha256(path, expected, "panel manifest")
    return _read_json_object(path, "panel manifest")


def _assert_panel_lock(
    path: Path,
    manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> str:
    expected_protocol = str(
        protocol.get("frozen_input", {}).get("panel_sha256", "")
    )
    expected_manifest = str(manifest.get("panel_file_sha256", ""))
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise DataValidationError("v0.8 panel cannot be read") from exc
    if not (
        len(expected_protocol) == 64
        and actual == expected_protocol
        and actual == expected_manifest
    ):
        raise DataValidationError(
            "v0.8 panel SHA-256 must match the actual file, protocol, and manifest"
        )
    return actual


def _validate_manifest_calendar(
    manifest: dict[str, Any], protocol: dict[str, Any]
) -> pd.DatetimeIndex:
    frozen = protocol.get("frozen_input", {})
    raw_sessions = manifest.get("sessions")
    if not isinstance(raw_sessions, list):
        raise DataValidationError("v0.8 manifest session calendar must be a list")
    expected_count = int(frozen.get("sessions", -1))
    if len(raw_sessions) != expected_count:
        raise DataValidationError("v0.8 manifest session count changed")
    sessions = normalize_expected_sessions(raw_sessions)
    canonical = [session.strftime("%Y-%m-%d") for session in sessions]
    if len(sessions) != expected_count or raw_sessions != canonical:
        raise DataValidationError(
            "v0.8 manifest sessions must be unique, sorted canonical dates"
        )
    actual_calendar_sha = session_calendar_hash(sessions)
    expected_calendar_sha = str(frozen.get("calendar_sha256", ""))
    if not (
        actual_calendar_sha == expected_calendar_sha
        and manifest.get("calendar_sha256") == expected_calendar_sha
    ):
        raise DataValidationError(
            "v0.8 calendar SHA-256 must match the sessions, protocol, and manifest"
        )
    expected_period = frozen.get("calendar_period")
    actual_period = [
        sessions.min().strftime("%Y-%m-%d"),
        sessions.max().strftime("%Y-%m-%d"),
    ]
    if expected_period != actual_period:
        raise DataValidationError("v0.8 calendar date bounds changed")
    return sessions


def _validate_panel(
    panel: pd.DataFrame,
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    sessions: pd.DatetimeIndex,
) -> dict[str, Any]:
    frozen = protocol["frozen_input"]
    if len(panel) != int(frozen["rows"]):
        raise DataValidationError("v0.8 panel row count changed")
    if int(manifest.get("rows", -1)) != len(panel):
        raise DataValidationError("v0.8 panel manifest row count mismatch")
    actual_codes = int(panel["code"].nunique()) if "code" in panel else -1
    if not (
        actual_codes == int(frozen["codes"])
        and actual_codes == int(manifest.get("codes", -1))
    ):
        raise DataValidationError("v0.8 panel code count changed")
    required = {
        "date", "code", "name", "label", "oc_return_pct",
        "price_eligible", "price_training_eligible",
        "candidate_price_source_max_date", *BASE_FEATURES, *G5_FEATURES,
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise DataValidationError(f"v0.8 panel lacks columns: {missing}")
    if list(panel.columns) != manifest.get("columns"):
        raise DataValidationError("v0.8 panel columns differ from its manifest")
    if panel.duplicated(["date", "code"]).any():
        raise DataValidationError("v0.8 panel has duplicate date/code rows")

    dates = pd.to_datetime(panel["date"], errors="coerce")
    if dates.isna().any() or getattr(dates.dt, "tz", None) is not None:
        raise DataValidationError("v0.8 panel dates must be valid timezone-naive dates")
    if not dates.eq(dates.dt.normalize()).all():
        raise DataValidationError("v0.8 panel dates must be normalized")
    panel_sessions = pd.DatetimeIndex(dates.drop_duplicates().sort_values())
    if not panel_sessions.equals(sessions):
        raise DataValidationError(
            "v0.8 panel date set must exactly match the locked session calendar"
        )

    label_missing = panel["label"].isna()
    return_missing = panel["oc_return_pct"].isna()
    labels = pd.to_numeric(panel["label"], errors="coerce")
    returns = pd.to_numeric(panel["oc_return_pct"], errors="coerce")
    if labels.isna().ne(label_missing).any():
        raise DataValidationError("v0.8 panel contains a nonnumeric label")
    if returns.isna().ne(return_missing).any():
        raise DataValidationError("v0.8 panel contains a nonnumeric return")
    if label_missing.ne(return_missing).any():
        raise DataValidationError("v0.8 label/return missingness is inconsistent")
    observed = ~return_missing
    observed_returns = returns.loc[observed].to_numpy(dtype=float)
    observed_labels = labels.loc[observed].to_numpy(dtype=float)
    if not np.isfinite(observed_returns).all():
        raise DataValidationError("v0.8 observed returns must be finite")
    if not np.isin(observed_labels, (0.0, 1.0)).all():
        raise DataValidationError("v0.8 observed labels must be binary")
    expected_labels = (observed_returns > 0.0).astype(float)
    if not np.array_equal(observed_labels, expected_labels):
        raise DataValidationError("v0.8 labels must equal (open-close return > 0)")
    for flag in ("price_eligible", "price_training_eligible"):
        if panel[flag].isna().any() or not panel[flag].isin((True, False)).all():
            raise DataValidationError(f"v0.8 {flag} must be complete and binary")

    source_dates = pd.to_datetime(
        panel["candidate_price_source_max_date"], errors="coerce"
    )
    invalid_source = (
        source_dates.isna()
        & panel["candidate_price_source_max_date"].notna()
    )
    if invalid_source.any():
        raise DataValidationError("v0.8 panel has invalid source dates")
    source_bad = source_dates.notna() & (source_dates >= dates)
    if source_bad.any():
        raise DataValidationError("v0.8 panel contains non-prior price features")
    return {
        "rows": int(len(panel)),
        "codes": actual_codes,
        "calendar_sessions": int(len(panel_sessions)),
        "calendar_date_set_exact": True,
        "observed_outcomes": int(observed.sum()),
        "missing_outcomes": int(return_missing.sum()),
        "label_return_missingness_exact": True,
        "finite_observed_returns": True,
        "binary_labels": True,
        "label_return_sign_exact": True,
        "strictly_prior_source_violations": 0,
    }


def _validate_score_schedule(
    sessions: pd.DatetimeIndex, protocol: dict[str, Any]
) -> tuple[pd.DatetimeIndex, pd.PeriodIndex]:
    frozen = protocol["frozen_input"]
    raw_score_period = frozen.get("score_period")
    if not isinstance(raw_score_period, list) or len(raw_score_period) != 2:
        raise DataValidationError("v0.8 score period is invalid")
    score_start, score_end = map(pd.Timestamp, raw_score_period)
    scheduled = sessions[(sessions >= score_start) & (sessions <= score_end)]
    if len(scheduled) != int(frozen.get("score_sessions", -1)):
        raise DataValidationError("v0.8 score-session count changed")
    periods = pd.period_range(score_start.to_period("M"), score_end.to_period("M"))
    if len(periods) != int(frozen.get("fold_count", -1)):
        raise DataValidationError("v0.8 monthly fold count changed")
    folded_dates = pd.DatetimeIndex(
        np.concatenate(
            [
                scheduled[scheduled.to_period("M") == period].to_numpy()
                for period in periods
            ]
        )
    ).sort_values()
    if not folded_dates.equals(scheduled):
        raise DataValidationError("v0.8 score dates are not covered exactly once")
    return scheduled, periods


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="/tmp/model_v07_corrected_panel.pkl")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output", default="/tmp/model_v08_target_reproduction.json")
    parser.add_argument("--picks-output", default="/tmp/model_v08_target_final_picks.csv")
    parser.add_argument("--error-output", default="/tmp/model_v08_target_error_cases.csv")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    panel_path = Path(args.panel).resolve()
    manifest_path = Path(str(panel_path) + ".manifest.json")
    protocol_path = Path(args.protocol).resolve()
    output_path = Path(args.output).resolve()
    picks_output = Path(args.picks_output).resolve()
    error_output = Path(args.error_output).resolve()
    protocol = _load_locked_protocol(protocol_path)
    manifest = _load_locked_manifest(manifest_path, protocol)
    actual_panel_sha256 = _assert_panel_lock(panel_path, manifest, protocol)
    sessions = _validate_manifest_calendar(manifest, protocol)
    panel = joblib.load(panel_path, mmap_mode="r")
    panel_checks = _validate_panel(panel, manifest, protocol, sessions)
    scheduled, periods = _validate_score_schedule(sessions, protocol)
    train_start = pd.Timestamp(protocol["frozen_input"]["training_start"])
    candidates = {
        "G0_control": BASE_FEATURES,
        "L4_mean_shadow": (*BASE_FEATURES, "flat_oc_rate_20"),
        "L6_robustness_shadow": (*BASE_FEATURES, *G5_FEATURES),
    }
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in candidates}
    folds: list[dict[str, Any]] = []
    mutation_capture: dict[
        str, tuple[FittedResearchScorer, pd.DataFrame, np.ndarray]
    ] = {}
    projection = list(
        dict.fromkeys(
            [
                "date", "code", "name", "label", "oc_return_pct",
                "price_eligible", "price_training_eligible",
                *BASE_FEATURES, *G5_FEATURES,
            ]
        )
    )
    for period in periods:
        score_start = period.start_time.normalize()
        score_end = period.end_time.normalize()
        training = panel.loc[
            panel["date"].between(train_start, score_start - pd.Timedelta(days=1))
            & panel["price_training_eligible"].eq(True)
            & panel["oc_return_pct"].notna(),
            projection,
        ].copy()
        scoring = panel.loc[
            panel["date"].between(score_start, score_end)
            & panel["price_eligible"].eq(True),
            projection,
        ].copy()
        if training.empty or training["date"].max() >= score_start:
            raise DataValidationError(
                f"v0.8 fold {period} does not have strictly prior training rows"
            )
        expected_fold_dates = scheduled[scheduled.to_period("M") == period]
        scoring_dates = pd.DatetimeIndex(
            scoring["date"].drop_duplicates().sort_values()
        )
        if not scoring_dates.isin(expected_fold_dates).all():
            raise DataValidationError(f"v0.8 fold {period} has foreign scoring dates")
        for candidate_id, columns in candidates.items():
            scorer = _fit_rank_ridge(training, columns, alpha=1.0)
            values = scorer.score(scoring)
            parts[candidate_id].append(_rank_two(scoring, values, candidate_id))
            folds.append(
                {
                    "period": str(period),
                    "candidate_id": candidate_id,
                    "train_start": str(training["date"].min().date()),
                    "train_end": str(training["date"].max().date()),
                    "train_rows": int(len(training)),
                    "score_rows": int(len(scoring)),
                    "strictly_prior_training": True,
                }
            )
            if str(period) == "2025-07" and candidate_id != "G0_control":
                mutation_capture[candidate_id] = (scorer, scoring.copy(), values.copy())
        del training, scoring
        gc.collect()

    for candidate_id in candidates:
        candidate_folds = [
            fold for fold in folds if fold["candidate_id"] == candidate_id
        ]
        if len(candidate_folds) != len(periods):
            raise DataValidationError(
                f"v0.8 candidate {candidate_id} does not have exactly 13 folds"
            )
        if [fold["period"] for fold in candidate_folds] != [
            str(period) for period in periods
        ]:
            raise DataValidationError(
                f"v0.8 candidate {candidate_id} fold date set changed"
            )

    picks_by_candidate: dict[str, pd.DataFrame] = {}
    pick_parts: list[pd.DataFrame] = []
    for candidate_id in candidates:
        actual = pd.concat(parts[candidate_id], ignore_index=True)
        keep = [
            "date", "model_rank", "code", "name", "model_score", "label",
            "oc_return_pct", "no_trade_rate_20", "flat_oc_rate_20",
        ]
        picks = _desired_slots(scheduled).merge(
            actual[keep],
            on=["date", "model_rank"],
            how="left",
            validate="one_to_one",
            sort=True,
        )
        picks["candidate_id"] = candidate_id
        if len(picks) != len(scheduled) * TOP_K:
            raise DataValidationError(
                f"v0.8 candidate {candidate_id} does not have two scheduled slots"
            )
        pick_dates = pd.DatetimeIndex(picks["date"].drop_duplicates().sort_values())
        if not pick_dates.equals(scheduled):
            raise DataValidationError(
                f"v0.8 candidate {candidate_id} score-date set changed"
            )
        if set(picks["model_rank"].unique()) != {1, 2}:
            raise DataValidationError(
                f"v0.8 candidate {candidate_id} rank slots changed"
            )
        picks_by_candidate[candidate_id] = picks
        pick_parts.append(picks)

    independent_pnl_max_diff = 0.0
    for picks in picks_by_candidate.values():
        for cost in (20.0, 40.0, 60.0):
            package = daily_portfolio_returns(
                picks, top_k=TOP_K, cost_bps=cost
            ).set_index("date")["net_return_pct"]
            independent_pnl_max_diff = max(
                independent_pnl_max_diff,
                float((package - _manual_daily(picks, cost)).abs().max()),
            )
    if independent_pnl_max_diff > 1e-12:
        raise DataValidationError("v0.8 independent PnL recomputation changed")
    mutation: dict[str, Any] = {}
    for candidate_id, (scorer, scoring, original_scores) in mutation_capture.items():
        changed = scoring.copy()
        extreme = np.where(np.arange(len(changed)) % 2 == 0, 99.0, -99.0)
        changed["oc_return_pct"] = extreme
        changed["label"] = (extreme > 0).astype(float)
        changed_scores = scorer.score(changed)
        original_keys = _rank_two(scoring, original_scores, "original")[[
            "date", "model_rank", "code"
        ]]
        changed_keys = _rank_two(changed, changed_scores, "changed")[[
            "date", "model_rank", "code"
        ]]
        mutation[candidate_id] = {
            "rows": int(len(scoring)),
            "max_abs_score_difference": float(
                np.max(np.abs(original_scores - changed_scores))
            ),
            "top2_keys_exact": bool(original_keys.equals(changed_keys)),
        }
    if set(mutation) != {"L4_mean_shadow", "L6_robustness_shadow"} or any(
        audit["max_abs_score_difference"] != 0.0
        or audit["top2_keys_exact"] is not True
        for audit in mutation.values()
    ):
        raise DataValidationError("v0.8 scorer depends on target-day outcomes")

    error_parts: list[pd.DataFrame] = []
    for candidate_id, picks in picks_by_candidate.items():
        daily = _manual_daily(picks, 20.0)
        for category, dates in (
            ("best_day", daily.nlargest(10).index),
            ("worst_day", daily.nsmallest(10).index),
        ):
            rows = picks.loc[picks["date"].isin(dates)].copy()
            rows["category"] = category
            rows["daily_net20"] = rows["date"].map(daily)
            error_parts.append(rows)
        rows = picks.loc[picks["oc_return_pct"].notna()].nsmallest(
            20, "oc_return_pct"
        ).copy()
        rows["category"] = "high_score_selected_loss"
        rows["daily_net20"] = rows["date"].map(daily)
        error_parts.append(rows)
    errors = pd.concat(error_parts, ignore_index=True).sort_values(
        ["candidate_id", "category", "date", "model_rank"], kind="stable"
    )
    all_picks = pd.concat(pick_parts, ignore_index=True)
    write_frame(all_picks, picks_output)
    write_frame(errors, error_output)
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "authority": "retrospective_finalist_reproduction_only",
        "production_model_changed": False,
        "protocol_sha256": LOCKED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "panel_sha256": actual_panel_sha256,
        "panel_manifest_sha256": protocol["frozen_input"][
            "panel_manifest_sha256"
        ],
        "integrity_checks": {
            "protocol_sha256_locked": True,
            "panel_sha256_actual_protocol_manifest_exact": True,
            "panel_manifest_sha256_protocol_exact": True,
            "calendar_sha256_sessions_protocol_manifest_exact": True,
            **panel_checks,
            "score_sessions": int(len(scheduled)),
            "score_date_set_exact": True,
            "monthly_folds": int(len(periods)),
            "candidate_fold_fits": int(len(folds)),
            "strictly_prior_training_all_folds": True,
            "slots_per_candidate": int(len(scheduled) * TOP_K),
            "core_model_family": "ridge_daily_rank",
            "target_formula": "2 * rank(method='average', pct=True) - 1",
            "target_exactly_zero_centered": False,
        },
        "candidates": {
            candidate_id: _candidate_metrics(picks)
            for candidate_id, picks in picks_by_candidate.items()
        },
        "outcome_mutation": mutation,
        "independent_pnl_max_abs_difference": independent_pnl_max_diff,
        "folds": folds,
        "picks_path": str(picks_output),
        "picks_sha256": sha256_file(picks_output),
        "error_path": str(error_output),
        "error_sha256": sha256_file(error_output),
    }
    write_json(result, output_path)


if __name__ == "__main__":
    main()
