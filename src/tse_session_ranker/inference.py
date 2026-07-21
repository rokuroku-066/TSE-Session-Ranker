from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .artifact import ModelArtifact
from .config import PreopenPolicy
from .data.preopen import latest_preopen_snapshots
from .data.common import normalize_expected_sessions, session_calendar_hash
from .exceptions import DataValidationError, LeakageError
from .features import FEATURE_COLUMNS, build_inference_frame


@dataclass
class PredictionResult:
    target_date: pd.Timestamp
    run_id: str
    candidates: pd.DataFrame
    eligible_universe_size: int
    total_universe_size: int
    calibration_status: str = "uncalibrated"
    score_semantics: str = "uncalibrated_positive_session_rank_score"


def rank_candidates(scored: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    required = {"code", "eligible", "model_score"}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise DataValidationError(f"scored frame is missing columns: {missing}")
    selected = scored[scored["eligible"]].copy()
    selected = selected.sort_values(
        ["model_score", "code"], ascending=[False, True], kind="stable"
    ).head(top_k)
    selected["model_rank"] = np.arange(1, len(selected) + 1)
    return selected.reset_index(drop=True)


def _expected_open_timestamp(
    value: object, target_date: pd.Timestamp, timezone: str
) -> pd.Timestamp | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        timestamp = pd.Timestamp(text)
    except Exception as exc:
        raise DataValidationError(f"invalid expected_open_at: {value}") from exc
    if timestamp.tzinfo is not None:
        return timestamp.tz_convert(timezone)
    if timestamp.date() == pd.Timestamp("1970-01-01").date() or len(text) <= 8:
        time_value = pd.to_datetime(text, format="%H:%M:%S", errors="coerce")
        if pd.isna(time_value):
            time_value = pd.to_datetime(text, format="%H:%M", errors="coerce")
        if pd.isna(time_value):
            raise DataValidationError(f"invalid expected_open_at: {value}")
        timestamp = target_date + pd.Timedelta(
            hours=time_value.hour, minutes=time_value.minute, seconds=time_value.second
        )
    return timestamp.tz_localize(timezone)


def apply_preopen_overlay(
    candidates: pd.DataFrame,
    snapshots: pd.DataFrame | None,
    target_date: object,
    as_of: datetime | pd.Timestamp | str | None,
    policy: PreopenPolicy | None = None,
) -> pd.DataFrame:
    """Attach provisional execution status without changing scores or ranks."""

    rules = policy or PreopenPolicy()
    output = candidates.copy()
    original_scores = output[["code", "model_score", "model_rank"]].copy()
    if snapshots is None or snapshots.empty:
        output["order_status"] = "MODEL_ONLY"
        output["veto_reasons"] = output.get(
            "selection_role", pd.Series("CORE", index=output.index)
        ).map(
            lambda role: (
                "preopen_snapshot_not_supplied,reserve_rank"
                if role == "RESERVE"
                else "preopen_snapshot_not_supplied"
            )
        )
        output["indicative_gap_pct"] = np.nan
        output["indicative_gap_atr"] = np.nan
        return output
    if as_of is None:
        raise LeakageError("as_of is required when pre-open snapshots are supplied")
    target = pd.Timestamp(target_date).normalize()
    latest = latest_preopen_snapshots(
        snapshots, target, as_of, timezone=rules.timezone
    )
    snapshot_columns = [
        column
        for column in latest.columns
        if column not in {"target_date"}
    ]
    output = output.merge(
        latest[snapshot_columns], on="code", how="left", suffixes=("", "_snapshot")
    )
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise LeakageError("as_of must include a timezone")
    cutoff = cutoff.tz_convert(rules.timezone)
    decision_cutoff = pd.Timestamp(
        f"{target.date()} {rules.decision_time}", tz=rules.timezone
    )
    if cutoff > decision_cutoff:
        raise LeakageError(
            f"as_of {cutoff.isoformat()} is after decision cutoff "
            f"{decision_cutoff.isoformat()}"
        )
    output["indicative_gap_pct"] = 100.0 * (
        output["indicative_price"] / output["prior_close"] - 1.0
    )
    output["indicative_gap_atr"] = (
        output["indicative_gap_pct"] / output["atr14_pct"]
    )
    latest_allowed_open = pd.Timestamp(
        f"{target.date()} {rules.latest_expected_open_time}", tz=rules.timezone
    )
    statuses: list[str] = []
    reasons_text: list[str] = []
    for _, row in output.iterrows():
        reasons: list[str] = []
        observed = row.get("observed_at")
        if pd.isna(observed):
            reasons.append("missing_snapshot")
        else:
            age = (cutoff - pd.Timestamp(observed)).total_seconds()
            if age > rules.max_snapshot_age_seconds:
                reasons.append("stale_snapshot")
        if pd.isna(row.get("indicative_price")) or pd.isna(row.get("prior_close")):
            reasons.append("missing_indicative_price")
        if pd.isna(row.get("atr14_pct")):
            reasons.append("missing_atr")
        gap_atr = row.get("indicative_gap_atr")
        if pd.notna(gap_atr) and float(gap_atr) >= rules.positive_gap_atr_veto:
            reasons.append("positive_gap_atr")
        buy_special = row.get("buy_special", False)
        if pd.notna(buy_special) and bool(buy_special):
            reasons.append("buy_special_quote")
        expected = _expected_open_timestamp(
            row.get("expected_open_at"), target, rules.timezone
        )
        if expected is not None and expected > latest_allowed_open:
            reasons.append("expected_open_after_0905")
        if row.get("selection_role", "CORE") == "RESERVE":
            reasons.append("reserve_rank")
        statuses.append("DISPLAY_ONLY" if reasons else "ORDER_ELIGIBLE")
        reasons_text.append(",".join(dict.fromkeys(reasons)) or "none")
    output["order_status"] = statuses
    output["veto_reasons"] = reasons_text

    after = output[["code", "model_score", "model_rank"]]
    if not original_scores.reset_index(drop=True).equals(after.reset_index(drop=True)):
        raise LeakageError("pre-open overlay changed the statistical ranking")
    return output


def predict_candidates(
    artifact: ModelArtifact,
    prices: pd.DataFrame,
    target_date: object,
    top_k: int | None = None,
    snapshots: pd.DataFrame | None = None,
    as_of: datetime | pd.Timestamp | str | None = None,
    expected_history_date: object | None = None,
    expected_sessions: object | None = None,
) -> PredictionResult:
    artifact.validate()
    target = pd.Timestamp(target_date).normalize()
    training_end = pd.Timestamp(artifact.manifest["training_end"]).normalize()
    if target <= training_end:
        raise LeakageError(
            "live inference target_date must be after the artifact training_end"
        )
    calendar_mode = artifact.manifest.get(
        "session_calendar_mode", "observed_sessions_only"
    )
    trained_calendar_hash = artifact.manifest.get("session_calendar_sha256")
    calendar = (
        normalize_expected_sessions(expected_sessions, through=target)
        if expected_sessions is not None
        else None
    )
    if calendar_mode == "explicit_exchange_sessions":
        if calendar is None:
            raise DataValidationError(
                "the artifact requires the explicit exchange session calendar"
            )
        if target not in calendar:
            raise DataValidationError("target_date is not in the exchange session calendar")
        prior_sessions = calendar[calendar < target]
        if prior_sessions.empty:
            raise DataValidationError(
                "exchange session calendar has no session before target_date"
            )
        calendar_previous = prior_sessions.max()
        if expected_history_date is None:
            raise DataValidationError(
                "expected_history_date is required with an explicit session calendar"
            )
        expected = pd.Timestamp(expected_history_date).normalize()
        if expected != calendar_previous:
            raise DataValidationError(
                f"calendar previous session is {calendar_previous.date()}, "
                f"expected_history_date is {expected.date()}"
            )
        actual_calendar_hash = session_calendar_hash(
            calendar,
            through=artifact.manifest["training_end"],
        )
        if actual_calendar_hash != trained_calendar_hash:
            raise DataValidationError(
                "session calendar does not match the artifact training calendar"
            )
    elif calendar is not None:
        raise DataValidationError(
            "artifact was trained without an explicit session calendar"
        )
    feature_frame = build_inference_frame(
        prices,
        target,
        artifact.config,
        expected_history_date=expected_history_date,
        expected_sessions=calendar,
    )
    missing = sorted(set(artifact.feature_columns) - set(feature_frame.columns))
    if missing:
        raise DataValidationError(f"inference features are missing: {missing}")
    scored = feature_frame.copy()
    scored["model_score"] = artifact.estimator.predict_proba(
        scored[list(artifact.feature_columns)]
    )[:, 1]
    requested_top_k = artifact.config.display_top_k if top_k is None else top_k
    selected = rank_candidates(scored, top_k=requested_top_k)
    selected["selection_role"] = np.where(
        selected["model_rank"].le(artifact.config.trade_top_k),
        "CORE",
        "RESERVE",
    )
    selected = apply_preopen_overlay(
        selected,
        snapshots=snapshots,
        target_date=target,
        as_of=as_of,
        policy=artifact.config.preopen,
    )
    selected["run_id"] = str(artifact.manifest["run_id"])
    selected["score_semantics"] = str(
        artifact.manifest.get(
            "score_semantics", "uncalibrated_positive_session_rank_score"
        )
    )
    selected["selection_objective"] = artifact.config.selection_objective
    selected["data_semantics"] = artifact.config.data_semantics
    selected["session_calendar_mode"] = calendar_mode
    selected["session_calendar_sha256"] = trained_calendar_hash
    return PredictionResult(
        target_date=target,
        run_id=str(artifact.manifest["run_id"]),
        candidates=selected,
        eligible_universe_size=int(scored["eligible"].sum()),
        total_universe_size=int(len(scored)),
        calibration_status=str(
            artifact.manifest.get("calibration_status", "uncalibrated")
        ),
        score_semantics=str(
            artifact.manifest.get(
                "score_semantics", "uncalibrated_positive_session_rank_score"
            )
        ),
    )
