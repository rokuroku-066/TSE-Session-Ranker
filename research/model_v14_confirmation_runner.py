#!/usr/bin/env python3
"""Run the remotely preregistered v1.4 confirmation experiment.

The four candidates are deliberately unlike the project's earlier row-wise
rankers: two exact low-rank state-dynamics models, a small spectral MLP, and a
complexity-feature Gaussian generative classifier.  Every score is formed
from completed prior sessions.  This script never changes production or
enables orders.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research.model_v13_symbolic_context_runner import (  # noqa: E402
    G0_FEATURES,
    load_and_build_panel,
    _validate_protocol as validate_v13_protocol,
)
from research.model_v14_feature_contrast_runner import (  # noqa: E402
    add_v14_features,
    _load_panel_cache,
)
from tse_session_ranker.data.common import normalize_expected_sessions  # noqa: E402
from tse_session_ranker.profit import (  # noqa: E402
    daily_portfolio_returns,
    profit_metrics,
)
from tse_session_ranker.research_models import (  # noqa: E402
    ResearchModelSpec,
    fit_research_model,
)
from tse_session_ranker.validation import (  # noqa: E402
    paired_moving_block_bootstrap,
)


PROTOCOL = ROOT / "research/model_v14_confirmation_protocol.json"
PROTOCOL_SHA256 = "ddc635c986dcb56072beda4889bfbf319019db31dfe3f74eedec24e51d8276af"
PROTOCOL_ID = "model_v14_feature_contrast_confirmation_20260727"
STAGE_A_PROTOCOL = ROOT / "research/model_v14_feature_contrast_protocol.json"
STAGE_A_PROTOCOL_SHA256 = (
    "8b2fcca037d36776a7803e066396e9cc711a47f1f67a57df5e117b653fd45f28"
)
STAGE_A_RESULT = ROOT / "research/model_v14_feature_contrast_result.json"
STAGE_A_RESULT_SHA256 = (
    "6cd878eca0392f87a781a3a4b6c3d38dd9948ba071ebc9f8f92f58557705eb31"
)
INPUT_ERRATUM = (
    ROOT / "research/model_v14_feature_contrast_input_erratum.json"
)
INPUT_ERRATUM_SHA256 = (
    "656be1249bc8d4f9fa8ce2f22bb6ab3c14450a16cf15c7c86f3e6492cf77c90f"
)
FROZEN_PANEL_CACHE_SHA256 = (
    "abcc6de28217f721358278c17039a60e4542361a60e2b1ac929325ab1a97516f"
)

TRAIN_START = pd.Timestamp("2024-01-04")
CONFIRMATION_START = pd.Timestamp("2024-11-01")
CONFIRMATION_END = pd.Timestamp("2025-07-31")
SCHEDULED_CONFIRMATION_SESSIONS = 182
CONFIRMATION_MONTHS = 9
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

CONTROL = "C00_DAILY_RANK_RIDGE"
CANDIDATES = ("DMD01", "DMD02", "SP01", "GN01")
ALL_MODELS = (CONTROL, *CANDIDATES)
CAPACITIES = (1, 2)
FAMILY_SIZE = 8
COSTS_BPS = (20.0, 40.0, 60.0)
PRIMARY_COST_BPS = 40.0
BOOTSTRAP_BLOCK_LENGTH = 5
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_RANDOM_STATE = 20_260_727

STABLE_FEATURES = (
    ("oc_mean_60", 1),
    ("oc_win_20", 1),
    ("overnight_mean_20", -1),
    ("component_sign_entropy_20", -1),
    ("overnight_mean_60", -1),
    ("overnight_last", -1),
    ("lunch_mean_5", -1),
    ("oc_mean_20", 1),
    ("xrank_close_momentum_5", -1),
    ("xrank_atr14_pct", 1),
    ("prior_range_pct", 1),
    ("range_abs_oc_ratio_20", 1),
    ("lunch_spectral_low_energy_ratio_32", 1),
    ("am_pm_return_corr_20", -1),
    ("prior_pm_return_pct", -1),
    ("am_mean_5", 1),
    ("overnight_spectral_high_energy_ratio_32", 1),
    ("am_std_20", 1),
    ("prior_lunch_gap_pct", -1),
    ("oc_sign_transition_rate_20", -1),
    ("overnight_spectral_dominant_frequency_32", 1),
)
SP_FEATURES = (
    "lunch_spectral_low_energy_ratio_32",
    "overnight_spectral_high_energy_ratio_32",
    "overnight_spectral_dominant_frequency_32",
)
GN_FEATURE_DIRECTIONS = (
    ("component_sign_entropy_20", -1),
    ("range_abs_oc_ratio_20", 1),
    ("am_pm_return_corr_20", -1),
    ("oc_sign_transition_rate_20", -1),
)


@dataclass(frozen=True)
class DMDSpec:
    candidate_id: str
    channels: tuple[str, ...]
    lookback: int
    rank: int


DMD_SPECS = {
    "DMD01": DMDSpec("DMD01", ("oc",), 60, 8),
    "DMD02": DMDSpec("DMD02", ("am", "lunch", "pm"), 32, 12),
}


class DMDStateError(ValueError):
    """Expected fail-closed condition for an inadequate DMD state."""


@dataclass(frozen=True)
class ExactDMDOperator:
    """Low-rank factors of ``Y V_r S_r^-1 U_r^T``."""

    left_factor: np.ndarray
    right_factor: np.ndarray
    rank: int

    def predict(self, latest_state: np.ndarray) -> np.ndarray:
        latest = np.asarray(latest_state, dtype=float)
        if latest.ndim != 1 or latest.shape[0] != self.right_factor.shape[1]:
            raise DMDStateError("latest DMD state has the wrong dimension")
        if not np.isfinite(latest).all():
            raise DMDStateError("latest DMD state is non-finite")
        predicted = self.left_factor @ (self.right_factor @ latest)
        if not np.isfinite(predicted).all():
            raise DMDStateError("DMD prediction is non-finite")
        return predicted


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


def _validate_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate every discovery and input binding before any scoring."""

    observed_protocol = sha256_file(PROTOCOL)
    if observed_protocol != PROTOCOL_SHA256:
        raise ValueError(
            "v1.4 confirmation protocol SHA-256 mismatch: "
            f"expected {PROTOCOL_SHA256}, got {observed_protocol}"
        )
    protocol = read_json(PROTOCOL)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected v1.4 confirmation protocol id")
    authority = protocol["authority"]
    if authority["production_promotion_allowed"] is not False:
        raise ValueError("v1.4 confirmation cannot promote production")
    if authority["orders_allowed"] is not False:
        raise ValueError("v1.4 confirmation must keep orders disabled")
    if int(protocol["confirmation"]["family_size"]) != FAMILY_SIZE:
        raise ValueError("v1.4 confirmation family size changed")
    if int(protocol["confirmation"]["scheduled_sessions"]) != (
        SCHEDULED_CONFIRMATION_SESSIONS
    ):
        raise ValueError("v1.4 confirmation session count changed")
    if tuple(protocol["confirmation"]["slices"]) != tuple(
        CONFIRMATION_SLICES
    ):
        raise ValueError("v1.4 confirmation slices changed")
    if tuple(item["id"] for item in protocol["registered_candidates"]) != (
        CANDIDATES
    ):
        raise ValueError("v1.4 registered candidates changed")

    for path, expected, label in (
        (STAGE_A_PROTOCOL, STAGE_A_PROTOCOL_SHA256, "Stage A protocol"),
        (STAGE_A_RESULT, STAGE_A_RESULT_SHA256, "Stage A result"),
    ):
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"v1.4 {label} SHA-256 mismatch: expected {expected}, got {observed}"
            )
    stage_a = read_json(STAGE_A_RESULT)
    if stage_a["authority"]["confirmation_outcomes_scored"] is not False:
        raise ValueError("Stage A result reports confirmation outcomes")
    observed_stable = tuple(
        (str(item["feature"]), int(np.sign(item["daily_rank_gap"]["mean"])))
        for item in stage_a["contrast"]["features"]
        if item["stable_signal"]
    )
    if observed_stable != STABLE_FEATURES:
        raise ValueError("v1.4 discovery-stable feature binding changed")
    protocol_stable = tuple(
        (str(item["feature"]), int(item["direction"]))
        for item in protocol["stage_a_binding"][
            "all_stable_features_in_discovery_order"
        ]
    )
    if protocol_stable != STABLE_FEATURES:
        raise ValueError("v1.4 protocol stable-feature selection changed")
    selections = protocol["stage_a_binding"]["candidate_feature_selections"]
    if tuple(item["feature"] for item in selections["SP01"]) != SP_FEATURES:
        raise ValueError("v1.4 SP01 feature selection changed")
    if tuple(
        (item["feature"], int(item["direction"]))
        for item in selections["GN01"]
    ) != GN_FEATURE_DIRECTIONS:
        raise ValueError("v1.4 GN01 feature selection changed")

    if INPUT_ERRATUM_SHA256.startswith("__"):
        raise ValueError(
            "v1.4 input erratum SHA-256 placeholder remains; confirmation "
            "scoring is forbidden"
        )
    protocol_erratum_sha = protocol["frozen_input"]["v14_input_erratum"][
        "sha256"
    ]
    if protocol_erratum_sha != INPUT_ERRATUM_SHA256:
        raise ValueError("v1.4 input erratum binding differs from runner")
    if not INPUT_ERRATUM.is_file():
        raise ValueError("v1.4 input erratum is missing")
    observed_erratum = sha256_file(INPUT_ERRATUM)
    if observed_erratum != INPUT_ERRATUM_SHA256:
        raise ValueError(
            "v1.4 input erratum SHA-256 mismatch: "
            f"expected {INPUT_ERRATUM_SHA256}, got {observed_erratum}"
        )
    erratum = read_json(INPUT_ERRATUM)
    erratum_authority = erratum.get("authority", {})
    if erratum_authority.get("production_promotion_allowed") is not False:
        raise ValueError("v1.4 input erratum cannot promote production")
    if erratum_authority.get("orders_allowed") is not False:
        raise ValueError("v1.4 input erratum must keep orders disabled")
    return protocol, erratum


def make_binary_label(
    open_price: pd.Series,
    close_price: pd.Series,
) -> pd.Series:
    """Return 1(close>open), 0(close<=open), and NaN for unobserved rows."""

    opened = pd.to_numeric(open_price, errors="coerce")
    closed = pd.to_numeric(close_price, errors="coerce")
    observed = opened.gt(0.0) & closed.gt(0.0)
    label = pd.Series(np.nan, index=opened.index, dtype=float)
    label.loc[observed] = closed.loc[observed].gt(opened.loc[observed]).astype(
        float
    )
    return label


def _rank_to_unit(
    values: pd.Series,
    dates: pd.Series,
    eligible: pd.Series | None = None,
) -> pd.Series:
    """Map average within-date ranks to [-1, 1], preserving missing rows."""

    numeric = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    if eligible is not None:
        numeric = numeric.where(eligible.fillna(False).astype(bool))
    date_values = pd.to_datetime(dates, errors="coerce").dt.normalize()
    if date_values.isna().any():
        raise ValueError("rank transform received an invalid date")
    ranks = numeric.groupby(date_values, sort=False).rank(method="average")
    counts = numeric.notna().groupby(date_values, sort=False).transform("sum")
    denominator = counts - 1
    output = 2.0 * (ranks - 1.0) / denominator.where(denominator.gt(0.0)) - 1.0
    output = output.where(counts.ne(1), 0.0).where(numeric.notna())
    return output.astype(float)


def _safe_return(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    top = pd.to_numeric(numerator, errors="coerce")
    bottom = pd.to_numeric(denominator, errors="coerce").where(
        lambda values: values.gt(0.0)
    )
    return 100.0 * (top / bottom - 1.0)


def add_confirmation_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the exact label and ranked SP01/GN01 inputs."""

    required = {
        "date",
        "code",
        "open",
        "close",
        "price_eligible",
        "price_training_eligible",
        *SP_FEATURES,
        *(feature for feature, _ in GN_FEATURE_DIRECTIONS),
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"v1.4 confirmation panel lacks columns: {missing}")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str)
    if frame["date"].isna().any() or frame.duplicated(["date", "code"]).any():
        raise ValueError("v1.4 confirmation panel has invalid date/code keys")
    frame["label"] = make_binary_label(frame["open"], frame["close"])
    eligible = frame["price_eligible"].fillna(False).astype(bool)
    for feature in dict.fromkeys(
        (*SP_FEATURES, *(item[0] for item in GN_FEATURE_DIRECTIONS))
    ):
        frame[f"_v14_rank_{feature}"] = _rank_to_unit(
            frame[feature], frame["date"], eligible
        )
    return frame


def make_completed_rank_states(
    panel: pd.DataFrame,
    sessions: Iterable[object],
) -> dict[str, pd.DataFrame]:
    """Create completed-session cross-stock rank states for the DMD models."""

    required = {
        "date",
        "code",
        "open",
        "close",
        "am_open",
        "am_close",
        "pm_open",
        "pm_close",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"v1.4 DMD panel lacks columns: {missing}")
    calendar = normalize_expected_sessions(sessions)
    frame = panel.loc[
        :,
        [
            "date",
            "code",
            "open",
            "close",
            "am_open",
            "am_close",
            "pm_open",
            "pm_close",
            *(["traded"] if "traded" in panel else []),
        ],
    ].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str)
    if frame["date"].isna().any() or frame.duplicated(["date", "code"]).any():
        raise ValueError("v1.4 DMD panel has invalid date/code keys")
    completed = (
        frame["traded"].fillna(False).astype(bool)
        if "traded" in frame
        else pd.Series(True, index=frame.index)
    )
    values = {
        "oc": _safe_return(frame["close"], frame["open"]).where(completed),
        "am": _safe_return(frame["am_close"], frame["am_open"]).where(
            completed
        ),
        "lunch": _safe_return(frame["pm_open"], frame["am_close"]).where(
            completed
        ),
        "pm": _safe_return(frame["pm_close"], frame["pm_open"]).where(
            completed
        ),
    }
    states: dict[str, pd.DataFrame] = {}
    for channel, channel_values in values.items():
        ranked = _rank_to_unit(channel_values, frame["date"])
        state = (
            frame.assign(_rank=ranked)
            .pivot(index="date", columns="code", values="_rank")
            .reindex(calendar)
            .reindex(columns=sorted(frame["code"].unique()))
        )
        state.index.name = "date"
        state.columns.name = "code"
        states[channel] = state
    return states


def _fit_exact_dmd(states: np.ndarray, rank: int) -> ExactDMDOperator:
    """Fit an exact rank-r truncated DMD operator to fixed prior states."""

    matrix = np.asarray(states, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise DMDStateError("DMD requires a two-dimensional state history")
    if not np.isfinite(matrix).all():
        raise DMDStateError("DMD state history contains a non-finite value")
    if rank < 1:
        raise ValueError("DMD rank must be positive")
    left = matrix[:, :-1]
    right = matrix[:, 1:]
    u, singular, vt = np.linalg.svd(left, full_matrices=False)
    if len(singular) < rank or singular[0] <= 0.0:
        raise DMDStateError("DMD state has fewer singular values than fixed rank")
    tolerance = (
        np.finfo(float).eps * max(left.shape) * float(singular[0])
    )
    if int(np.count_nonzero(singular > tolerance)) < rank:
        raise DMDStateError("DMD state numerical rank is below fixed rank")
    u_rank = u[:, :rank]
    singular_rank = singular[:rank]
    v_rank = vt[:rank, :].T
    return ExactDMDOperator(
        left_factor=right @ (v_rank / singular_rank),
        right_factor=u_rank.T,
        rank=rank,
    )


def _exact_dmd_predict(states: np.ndarray, rank: int) -> np.ndarray:
    """Fit exact DMD and apply it to the latest state (unit-test helper)."""

    matrix = np.asarray(states, dtype=float)
    return _fit_exact_dmd(matrix, rank).predict(matrix[:, -1])


def _fit_dmd_month(
    month_first: pd.Timestamp,
    sessions: pd.DatetimeIndex,
    states: dict[str, pd.DataFrame],
    spec: DMDSpec,
) -> tuple[ExactDMDOperator, tuple[str, ...], pd.DatetimeIndex]:
    """Fit one fixed operator from states strictly before a score month."""

    first = pd.Timestamp(month_first).normalize()
    try:
        location = int(sessions.get_loc(first))
    except KeyError as error:
        raise DMDStateError("DMD month start is outside the calendar") from error
    if location < spec.lookback:
        raise DMDStateError("DMD has fewer pre-month states than fixed lookback")
    history = sessions[location - spec.lookback : location]
    if len(history) != spec.lookback or history[-1] >= first:
        raise AssertionError("DMD month-fit history is not strictly prior")
    column_sets = [
        tuple(states[channel].columns.astype(str)) for channel in spec.channels
    ]
    if not column_sets or any(columns != column_sets[0] for columns in column_sets):
        raise DMDStateError("DMD channel code mappings differ")
    all_codes = column_sets[0]
    active = np.ones(len(all_codes), dtype=bool)
    for channel in spec.channels:
        active &= (
            states[channel]
            .loc[history, list(all_codes)]
            .notna()
            .any(axis=0)
            .to_numpy()
        )
    codes = tuple(
        code
        for code, is_active in zip(all_codes, active, strict=True)
        if is_active
    )
    if not codes:
        raise DMDStateError("DMD state mapping is empty")
    blocks = [
        states[channel]
        .loc[history, list(codes)]
        .fillna(0.0)
        .to_numpy(dtype=float)
        .T
        for channel in spec.channels
    ]
    operator = _fit_exact_dmd(np.vstack(blocks), spec.rank)
    return operator, codes, history


def _score_dmd_date(
    scoring: pd.DataFrame,
    score_date: pd.Timestamp,
    sessions: pd.DatetimeIndex,
    states: dict[str, pd.DataFrame],
    spec: DMDSpec,
    operator: ExactDMDOperator,
    codes: Sequence[str],
) -> pd.DataFrame:
    """Apply a frozen monthly operator to the latest completed actual state."""

    date = pd.Timestamp(score_date).normalize()
    try:
        location = int(sessions.get_loc(date))
    except KeyError as error:
        raise DMDStateError("DMD score date is outside the calendar") from error
    if location < 1:
        raise DMDStateError("DMD score date has no preceding state")
    latest_date = sessions[location - 1]
    if latest_date >= date:
        raise AssertionError("DMD latest input state is not strictly prior")
    candidate_codes = sorted(
        scoring.loc[
            scoring["price_eligible"].fillna(False).astype(bool), "code"
        ]
        .astype(str)
        .unique()
    )
    if not candidate_codes:
        raise DMDStateError("DMD score date has no eligible codes")
    known = set(str(code) for code in codes)
    candidate_codes = [code for code in candidate_codes if code in known]
    if not candidate_codes:
        raise DMDStateError("DMD has no eligible code in the state mapping")
    state_blocks = [
        states[channel]
        .loc[latest_date, list(codes)]
        .fillna(0.0)
        .to_numpy(dtype=float)
        for channel in spec.channels
    ]
    predicted = operator.predict(np.concatenate(state_blocks))
    by_channel = predicted.reshape(len(spec.channels), len(codes))
    score = by_channel.mean(axis=0)
    mapped = pd.Series(score, index=pd.Index(codes, name="code"))
    current = scoring.loc[
        scoring["price_eligible"].fillna(False).astype(bool)
        & scoring["code"].astype(str).isin(candidate_codes)
    ].copy()
    values = current["code"].astype(str).map(mapped).to_numpy(dtype=float)
    return _top_two(current, values, spec.candidate_id)


def _top_two(
    scoring: pd.DataFrame,
    values: Sequence[float] | np.ndarray,
    candidate_id: str,
) -> pd.DataFrame:
    if len(scoring) != len(values):
        raise ValueError("score length differs from scoring rows")
    ranked = scoring.assign(model_score=np.asarray(values, dtype=float))
    ranked = ranked.loc[np.isfinite(ranked["model_score"])].copy()
    ranked["code"] = ranked["code"].astype(str)
    ranked = ranked.sort_values(
        ["date", "model_score", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranked = ranked.groupby("date", sort=True, as_index=False).head(2).copy()
    ranked["model_rank"] = ranked.groupby("date", sort=False).cumcount() + 1
    ranked["candidate_id"] = candidate_id
    return ranked


def _desired_slots(
    scheduled: pd.DatetimeIndex,
    candidate_id: str,
) -> pd.DataFrame:
    slots = pd.MultiIndex.from_product(
        [scheduled, (1, 2)], names=["date", "model_rank"]
    ).to_frame(index=False)
    slots["candidate_id"] = candidate_id
    return slots


def _complete_slots(
    actual: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    candidate_id: str,
) -> pd.DataFrame:
    keep = [
        "date",
        "model_rank",
        "candidate_id",
        "code",
        "name",
        "model_score",
        "label",
        "oc_return_pct",
    ]
    desired = _desired_slots(scheduled, candidate_id)
    if actual.empty:
        result = desired.copy()
        for column in keep:
            if column not in result:
                result[column] = np.nan
        result["candidate_id"] = candidate_id
    else:
        result = desired.merge(
            actual.loc[:, keep],
            on=["date", "model_rank", "candidate_id"],
            how="left",
            validate="one_to_one",
            sort=True,
        )
    if len(result) != len(scheduled) * 2:
        raise AssertionError("v1.4 slot completion changed the schedule")
    return result.loc[:, keep]


def run_dmd_models(
    panel: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    sessions: pd.DatetimeIndex,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    states = make_completed_rank_states(panel, sessions)
    projection = [
        "date",
        "code",
        "name",
        "label",
        "oc_return_pct",
        "price_eligible",
    ]
    output: dict[str, pd.DataFrame] = {}
    details: dict[str, Any] = {}
    for candidate_id, spec in DMD_SPECS.items():
        parts: list[pd.DataFrame] = []
        failures: list[dict[str, str]] = []
        folds: list[dict[str, Any]] = []
        periods = pd.period_range(
            CONFIRMATION_START.to_period("M"),
            CONFIRMATION_END.to_period("M"),
        )
        for period in periods:
            month_dates = scheduled[
                (scheduled >= period.start_time.normalize())
                & (scheduled <= period.end_time.normalize())
            ]
            if month_dates.empty:
                raise ValueError(f"v1.4 DMD month {period} has no score date")
            first = month_dates[0]
            try:
                operator, codes, history = _fit_dmd_month(
                    first, sessions, states, spec
                )
            except DMDStateError as error:
                failures.extend(
                    {"date": str(date.date()), "reason": str(error)}
                    for date in month_dates
                )
                folds.append(
                    {
                        "period": str(period),
                        "fit_status": "FAIL_CLOSED",
                        "reason": str(error),
                        "score_dates": len(month_dates),
                    }
                )
                continue
            for date in month_dates:
                scoring = panel.loc[panel["date"].eq(date), projection].copy()
                try:
                    parts.append(
                        _score_dmd_date(
                            scoring,
                            date,
                            sessions,
                            states,
                            spec,
                            operator,
                            codes,
                        )
                    )
                except DMDStateError as error:
                    failures.append(
                        {"date": str(date.date()), "reason": str(error)}
                    )
            folds.append(
                {
                    "period": str(period),
                    "fit_status": "PASS",
                    "state_start": str(history.min().date()),
                    "state_end": str(history.max().date()),
                    "state_sessions": len(history),
                    "score_start": str(month_dates.min().date()),
                    "score_end": str(month_dates.max().date()),
                    "score_dates": len(month_dates),
                    "operator_frozen_within_month": True,
                    "strictly_prior_fit": bool(history.max() < first),
                    "state_coordinates": int(
                        len(codes) * len(spec.channels)
                    ),
                }
            )
        actual = (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame()
        )
        output[candidate_id] = _complete_slots(
            actual, scheduled, candidate_id
        )
        details[candidate_id] = {
            "channels": list(spec.channels),
            "state_lookback_sessions": spec.lookback,
            "fixed_rank": spec.rank,
            "score_dates": len(scheduled),
            "fit_schedule": "month_fixed",
            "folds": folds,
            "fail_closed_dates": failures,
            "strictly_prior_states": True,
        }
    return output, details


def _date_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("date", sort=False)["code"].transform("size")
    weights = 1.0 / counts.to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("invalid date-equal sample weights")
    return weights


def _positive_probability(model: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == 1.0)
    if len(matches) != 1:
        raise ValueError("classifier does not expose the positive class")
    output = probabilities[:, int(matches[0])]
    if not np.isfinite(output).all():
        raise ValueError("classifier produced a non-finite probability")
    return output


def _monthly_classifier_scores(
    panel: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    candidate_id: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if candidate_id == "SP01":
        feature_columns = [f"_v14_rank_{feature}" for feature in SP_FEATURES]
        directions = np.ones(len(feature_columns), dtype=float)
    elif candidate_id == "GN01":
        feature_columns = [
            f"_v14_rank_{feature}" for feature, _ in GN_FEATURE_DIRECTIONS
        ]
        directions = np.asarray(
            [direction for _, direction in GN_FEATURE_DIRECTIONS],
            dtype=float,
        )
    else:
        raise ValueError(f"unknown monthly classifier: {candidate_id}")
    projection = [
        "date",
        "code",
        "name",
        "label",
        "oc_return_pct",
        "price_eligible",
        "price_training_eligible",
        *feature_columns,
    ]
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    periods = pd.period_range(
        CONFIRMATION_START.to_period("M"),
        CONFIRMATION_END.to_period("M"),
    )
    for period in periods:
        first = period.start_time.normalize()
        last = period.end_time.normalize()
        training_mask = (
            panel["date"].between(
                TRAIN_START, first - pd.Timedelta(days=1)
            )
            & panel["price_training_eligible"].fillna(False).astype(bool)
            & panel["label"].notna()
            & panel[feature_columns].notna().all(axis=1)
        )
        scoring_mask = (
            panel["date"].between(first, last)
            & panel["date"].isin(scheduled)
            & panel["price_eligible"].fillna(False).astype(bool)
            & panel[feature_columns].notna().all(axis=1)
        )
        training = panel.loc[training_mask, projection].copy()
        scoring = panel.loc[scoring_mask, projection].copy()
        if training.empty or training["date"].max() >= first:
            raise ValueError(
                f"v1.4 {candidate_id} fold {period} has invalid training"
            )
        if training["label"].nunique() != 2:
            raise ValueError(
                f"v1.4 {candidate_id} fold {period} lacks both classes"
            )
        train_x = training[feature_columns].to_numpy(dtype=float) * directions
        score_x = scoring[feature_columns].to_numpy(dtype=float) * directions
        weights = _date_equal_weights(training)
        if candidate_id == "SP01":
            model: Any = MLPClassifier(
                hidden_layer_sizes=(6,),
                activation="tanh",
                solver="lbfgs",
                alpha=0.05,
                max_iter=500,
                tol=1e-7,
                random_state=20_260_727,
            )
        else:
            model = GaussianNB(
                priors=np.asarray((0.5, 0.5), dtype=float),
                var_smoothing=1e-6,
            )
        model.fit(
            train_x,
            training["label"].to_numpy(dtype=float),
            sample_weight=weights,
        )
        if not scoring.empty:
            values = _positive_probability(model, score_x)
            parts.append(_top_two(scoring, values, candidate_id))
        folds.append(
            {
                "period": str(period),
                "train_start": str(training["date"].min().date()),
                "train_end": str(training["date"].max().date()),
                "train_rows": int(len(training)),
                "train_dates": int(training["date"].nunique()),
                "score_rows": int(len(scoring)),
                "strictly_prior_training": bool(
                    training["date"].max() < first
                ),
                "date_equal_training_weight": bool(
                    np.allclose(
                        pd.Series(weights, index=training.index)
                        .groupby(training["date"])
                        .sum()
                        .to_numpy(),
                        1.0,
                        rtol=0.0,
                        atol=1e-12,
                    )
                ),
            }
        )
    actual = (
        pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    )
    return _complete_slots(actual, scheduled, candidate_id), folds


def run_control(
    panel: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    periods = pd.period_range(
        CONFIRMATION_START.to_period("M"),
        CONFIRMATION_END.to_period("M"),
    )
    projection = [
        "date",
        "code",
        "name",
        "label",
        "oc_return_pct",
        "price_eligible",
        "price_training_eligible",
        *G0_FEATURES,
    ]
    specification = ResearchModelSpec(
        name="model_v14_control_daily_rank_ridge",
        family="ridge_daily_rank",
        objective="same_day_return_percentile",
        parameters={"alpha": 1.0},
    )
    for period in periods:
        first = period.start_time.normalize()
        last = period.end_time.normalize()
        training = panel.loc[
            panel["date"].between(
                TRAIN_START, first - pd.Timedelta(days=1)
            )
            & panel["price_training_eligible"].fillna(False).astype(bool)
            & panel["oc_return_pct"].notna(),
            projection,
        ].copy()
        scoring = panel.loc[
            panel["date"].between(first, last)
            & panel["date"].isin(scheduled)
            & panel["price_eligible"].fillna(False).astype(bool),
            projection,
        ].copy()
        if training.empty or training["date"].max() >= first:
            raise ValueError(f"v1.4 control fold {period} has invalid training")
        scorer = fit_research_model(specification, training, G0_FEATURES)
        if not scoring.empty:
            parts.append(
                _top_two(scoring, scorer.score(scoring), CONTROL)
            )
        folds.append(
            {
                "period": str(period),
                "train_start": str(training["date"].min().date()),
                "train_end": str(training["date"].max().date()),
                "train_rows": int(len(training)),
                "score_rows": int(len(scoring)),
                "strictly_prior_training": True,
            }
        )
    actual = (
        pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    )
    return _complete_slots(actual, scheduled, CONTROL), folds


def _daily(
    picks: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    capacity: int,
    cost_bps: float,
) -> pd.Series:
    frame = daily_portfolio_returns(
        picks, top_k=capacity, cost_bps=cost_bps
    )
    values = frame.set_index("date")["net_return_pct"].sort_index()
    expected = pd.DatetimeIndex(scheduled)
    if (
        len(values) != len(expected)
        or values.isna().any()
        or not values.index.equals(expected)
    ):
        raise AssertionError("v1.4 daily return schedule is incomplete")
    return values


def _top_codes_cash(
    picks: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    capacity: int,
    cost_bps: float,
    count: int,
) -> tuple[pd.Series, list[str]]:
    selected = picks[picks["model_rank"].le(capacity)].copy()
    executed = selected["label"].notna()
    selected["_net_slot"] = selected["oc_return_pct"].fillna(0.0) - (
        executed.astype(float) * cost_bps / 100.0
    )
    by_code = (
        selected.dropna(subset=["code"])
        .groupby("code", sort=False)["_net_slot"]
        .sum()
        .sort_values(ascending=False, kind="stable")
    )
    codes = [str(value) for value in by_code.head(count).index]
    neutral = picks.copy()
    mask = (
        neutral["model_rank"].le(capacity)
        & neutral["code"].astype(str).isin(codes)
    )
    neutral.loc[mask, ["label", "oc_return_pct"]] = np.nan
    return (
        _daily(
            neutral,
            scheduled,
            capacity=capacity,
            cost_bps=cost_bps,
        ),
        codes,
    )


def _variant_metrics(
    picks: pd.DataFrame,
    control: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    candidate_id: str,
    capacity: int,
) -> dict[str, Any]:
    costs = {
        str(int(cost)): profit_metrics(
            picks, top_k=capacity, cost_bps=cost
        )
        for cost in COSTS_BPS
    }
    daily40 = _daily(
        picks, scheduled, capacity=capacity, cost_bps=PRIMARY_COST_BPS
    )
    daily60 = _daily(picks, scheduled, capacity=capacity, cost_bps=60.0)
    control40 = _daily(
        control, scheduled, capacity=capacity, cost_bps=PRIMARY_COST_BPS
    )
    bonferroni_confidence = 1.0 - (1.0 - 0.90) / FAMILY_SIZE
    paired = paired_moving_block_bootstrap(
        daily40,
        control40,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        samples=BOOTSTRAP_SAMPLES,
        confidence=bonferroni_confidence,
        random_state=BOOTSTRAP_RANDOM_STATE + CAPACITIES.index(capacity),
    )
    slice_values = {
        name: float(daily40.loc[start:end].mean())
        for name, (start, end) in CONFIRMATION_SLICES.items()
    }
    top10_removed = float(
        daily40.drop(daily40.nlargest(10).index).mean()
    )
    top_codes_cash, profitable_codes = _top_codes_cash(
        picks,
        scheduled,
        capacity=capacity,
        cost_bps=PRIMARY_COST_BPS,
        count=10,
    )
    monthly = daily40.groupby(daily40.index.to_period("M")).mean()
    selected = picks.loc[picks["model_rank"].le(capacity)].copy()
    signaled = selected.dropna(subset=["code"])
    counts = signaled["code"].astype(str).value_counts()
    unique_codes = int(len(counts))
    signaled_count = int(len(signaled))
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
    expected_slots = len(scheduled) * capacity
    executed_fraction = float(executed.sum() / expected_slots)
    if len(monthly) != CONFIRMATION_MONTHS:
        raise AssertionError("v1.4 confirmation month count changed")
    checks = {
        "net40_positive": float(daily40.mean()) > 0.0,
        "net60_positive": float(daily60.mean()) > 0.0,
        "all_three_confirmation_slices_net40_positive": all(
            value > 0.0 for value in slice_values.values()
        ),
        "top10_days_removed_net40_positive": top10_removed > 0.0,
        "top10_profit_codes_cash_net40_positive": (
            float(top_codes_cash.mean()) > 0.0
        ),
        "positive_months_at_least_6_of_9": (
            int(monthly.gt(0.0).sum()) >= 6
        ),
        "familywise_paired_lower_vs_control_nonnegative": (
            paired.one_sided_lower_delta_pct >= 0.0
        ),
        "unique_codes_at_least_100": unique_codes >= 100,
        "maximum_code_share_at_most_5pct": maximum_share <= 0.05,
        "top10_code_share_at_most_25pct": top10_share <= 0.25,
        "traded_days_at_least_150": traded_days >= 150,
        "executed_slot_fraction_at_least_80pct": (
            executed_fraction >= 0.80
        ),
    }
    return {
        "variant_id": f"{candidate_id}__top{capacity}",
        "candidate_id": candidate_id,
        "capacity": capacity,
        "cost_metrics": costs,
        "confirmation_slice_net40_mean_pct": slice_values,
        "top10_days_removed_net40_mean_pct": top10_removed,
        "top10_profit_codes_cash_net40_mean_pct": float(
            top_codes_cash.mean()
        ),
        "top10_profit_codes": profitable_codes,
        "positive_months_net40": int(monthly.gt(0.0).sum()),
        "months": int(len(monthly)),
        "monthly_net40_mean_pct": {
            str(period): float(value) for period, value in monthly.items()
        },
        "paired_vs_control_net40": asdict(paired),
        "bonferroni_individual_confidence": bonferroni_confidence,
        "unique_selected_codes": unique_codes,
        "maximum_code_selection_share": maximum_share,
        "top10_code_selection_share": top10_share,
        "traded_days": traded_days,
        "executed_slots": int(executed.sum()),
        "expected_capacity_slots": expected_slots,
        "executed_slot_fraction": executed_fraction,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def _semantic_hash(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    canonical["date"] = pd.to_datetime(canonical["date"]).dt.strftime(
        "%Y-%m-%d"
    )
    canonical = canonical.sort_values(
        ["candidate_id", "date", "model_rank"], kind="stable"
    ).reset_index(drop=True)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def load_confirmation_panel(
    *,
    panel_cache: str | Path | None,
    jpx_directory: str | Path | None,
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, Any]]:
    if (panel_cache is None) == (jpx_directory is None):
        raise ValueError("provide exactly one of panel_cache or jpx_directory")
    if panel_cache is not None:
        observed_cache_sha = sha256_file(panel_cache)
        if observed_cache_sha != FROZEN_PANEL_CACHE_SHA256:
            raise ValueError(
                "v1.4 panel-cache SHA-256 mismatch: "
                f"expected {FROZEN_PANEL_CACHE_SHA256}, "
                f"got {observed_cache_sha}"
            )
        panel, sessions, checks = _load_panel_cache(panel_cache)
    else:
        v13_protocol, v13_erratum = validate_v13_protocol()
        panel, sessions, checks = load_and_build_panel(
            jpx_directory, v13_protocol, v13_erratum
        )
        checks = {"source": "official_jpx_pdf", **checks}
    panel_dates = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    if panel_dates.isna().any():
        raise ValueError("v1.4 confirmation input has an invalid date")
    observed = {
        "panel_rows": int(len(panel)),
        "codes": int(panel["code"].astype(str).nunique()),
        "sessions": int(len(sessions)),
        "date_bounds": [
            str(panel_dates.min().date()),
            str(panel_dates.max().date()),
        ],
    }
    expected = {
        "panel_rows": int(protocol["frozen_input"]["modeling_rows"]),
        "codes": int(protocol["frozen_input"]["codes"]),
        "sessions": int(protocol["frozen_input"]["sessions"]),
        "date_bounds": list(protocol["frozen_input"]["date_bounds"]),
    }
    if observed != expected:
        raise ValueError(
            f"v1.4 frozen panel dimensions changed: expected {expected}, "
            f"got {observed}"
        )
    checks["frozen_panel_dimensions_verified"] = True
    return panel.copy(), normalize_expected_sessions(sessions), checks


def run(
    *,
    protocol: dict[str, Any],
    panel_cache: str | Path | None,
    jpx_directory: str | Path | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    started = time.perf_counter()
    panel, sessions, input_checks = load_confirmation_panel(
        panel_cache=panel_cache,
        jpx_directory=jpx_directory,
        protocol=protocol,
    )
    scheduled = sessions[
        (sessions >= CONFIRMATION_START) & (sessions <= CONFIRMATION_END)
    ]
    if len(scheduled) != SCHEDULED_CONFIRMATION_SESSIONS:
        raise ValueError(
            "v1.4 scheduled confirmation sessions changed: "
            f"expected {SCHEDULED_CONFIRMATION_SESSIONS}, got {len(scheduled)}"
        )
    featured = add_v14_features(panel, sessions)
    featured = add_confirmation_columns(featured)
    dmd_picks, dmd_details = run_dmd_models(featured, scheduled, sessions)
    sp_picks, sp_folds = _monthly_classifier_scores(
        featured, scheduled, candidate_id="SP01"
    )
    gn_picks, gn_folds = _monthly_classifier_scores(
        featured, scheduled, candidate_id="GN01"
    )
    control, control_folds = run_control(featured, scheduled)
    picks_by_model = {
        CONTROL: control,
        **dmd_picks,
        "SP01": sp_picks,
        "GN01": gn_picks,
    }
    all_picks = pd.concat(
        [picks_by_model[model] for model in ALL_MODELS],
        ignore_index=True,
    )
    all_picks = all_picks[
        [
            "candidate_id",
            "date",
            "model_rank",
            "code",
            "name",
            "model_score",
            "label",
            "oc_return_pct",
        ]
    ].sort_values(
        ["candidate_id", "date", "model_rank"], kind="stable"
    ).reset_index(drop=True)
    control_metrics = {
        f"top{capacity}": {
            str(int(cost)): profit_metrics(
                control, top_k=capacity, cost_bps=cost
            )
            for cost in COSTS_BPS
        }
        for capacity in CAPACITIES
    }
    variants = [
        _variant_metrics(
            picks_by_model[candidate],
            control,
            scheduled,
            candidate_id=candidate,
            capacity=capacity,
        )
        for candidate in CANDIDATES
        for capacity in CAPACITIES
    ]
    passers = [
        item["variant_id"] for item in variants if item["gate_passed"]
    ]
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "stage_a_protocol_sha256": STAGE_A_PROTOCOL_SHA256,
        "stage_a_result_sha256": STAGE_A_RESULT_SHA256,
        "input_erratum_sha256": INPUT_ERRATUM_SHA256,
        "runner_sha256": sha256_file(__file__),
        "input": input_checks,
        "runtime": {
            "seconds": float(time.perf_counter() - started),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "integrity": {
            "candidate_count": len(CANDIDATES),
            "capacity_variants": list(CAPACITIES),
            "family_size": len(variants),
            "scheduled_sessions": len(scheduled),
            "confirmation_date_bounds": [
                str(scheduled.min().date()),
                str(scheduled.max().date()),
            ],
            "slots_per_model": len(scheduled) * 2,
            "picks_rows": int(len(all_picks)),
            "picks_semantic_sha256": _semantic_hash(all_picks),
            "strictly_prior_features": True,
            "monthly_scoring_before_same_month_outcomes": True,
            "orders_allowed": False,
            "production_model_changed": False,
        },
        "models": {
            "DMD": dmd_details,
            "SP01": {"folds": sp_folds},
            "GN01": {"folds": gn_folds},
        },
        "control": {
            "candidate_id": CONTROL,
            "folds": control_folds,
            "metrics": control_metrics,
        },
        "variants": variants,
        "decision": {
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
        },
    }
    return result, all_picks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--panel-cache", type=Path)
    source.add_argument("--jpx-directory", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research/model_v14_confirmation_result.json",
    )
    parser.add_argument(
        "--picks-output",
        type=Path,
        default=ROOT / "research/model_v14_confirmation_picks.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol, _ = _validate_protocol()
    result, picks = run(
        protocol=protocol,
        panel_cache=args.panel_cache,
        jpx_directory=args.jpx_directory,
    )
    args.picks_output.parent.mkdir(parents=True, exist_ok=True)
    picks.to_csv(args.picks_output, index=False, lineterminator="\n")
    write_json(result, args.output)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
