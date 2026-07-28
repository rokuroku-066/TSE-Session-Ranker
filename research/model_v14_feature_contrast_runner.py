#!/usr/bin/env python3
"""Run the preregistered v1.4 discovery-only feature contrast.

This stage describes strictly lagged features on 2024-07-01 through
2024-10-31.  It neither fits a trading model nor reads confirmation outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research.model_v13_symbolic_context_runner import (  # noqa: E402
    G0_FEATURES,
    _validate_protocol as validate_v13_protocol,
    load_and_build_panel,
)
from tse_session_ranker.data.common import normalize_expected_sessions  # noqa: E402


PROTOCOL = ROOT / "research/model_v14_feature_contrast_protocol.json"
PROTOCOL_SHA256 = "8b2fcca037d36776a7803e066396e9cc711a47f1f67a57df5e117b653fd45f28"
PROTOCOL_ID = "model_v14_feature_contrast_discovery_20260727"
DISCOVERY_START = pd.Timestamp("2024-07-01")
DISCOVERY_END = pd.Timestamp("2024-10-31")
FORBIDDEN_CONFIRMATION_START = pd.Timestamp("2024-11-01")
SCHEDULED_DISCOVERY_SESSIONS = 84
SLICES = {
    "discovery_a": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-08-30")),
    "discovery_b": (pd.Timestamp("2024-09-02"), pd.Timestamp("2024-09-30")),
    "discovery_c": (pd.Timestamp("2024-10-01"), pd.Timestamp("2024-10-31")),
}

EXISTING_FEATURES = tuple(G0_FEATURES)
SESSION_FEATURES = (
    "prior_am_return_pct",
    "prior_lunch_gap_pct",
    "prior_pm_return_pct",
    "prior_range_pct",
    "prior_close_location",
    "am_mean_5",
    "lunch_mean_5",
    "pm_mean_5",
    "am_std_20",
    "pm_std_20",
)
SPECTRAL_CHANNELS = ("oc", "overnight", "am", "lunch", "pm")
SPECTRAL_MEASURES = (
    "spectral_low_energy_ratio_32",
    "spectral_high_energy_ratio_32",
    "spectral_entropy_32",
    "spectral_dominant_frequency_32",
)
SPECTRAL_FEATURES = tuple(
    f"{channel}_{measure}"
    for channel in SPECTRAL_CHANNELS
    for measure in SPECTRAL_MEASURES
)
COMPLEXITY_FEATURES = (
    "oc_sign_transition_rate_20",
    "oc_turning_rate_20",
    "oc_permutation_entropy_20",
    "oc_abs_autocorr_20",
    "am_pm_sign_agreement_20",
    "am_pm_return_corr_20",
    "component_sign_entropy_20",
    "range_abs_oc_ratio_20",
)
FEATURE_GROUPS = {
    "existing_g0": EXISTING_FEATURES,
    "lagged_session_decomposition": SESSION_FEATURES,
    "lagged_spectrum": SPECTRAL_FEATURES,
    "lagged_path_complexity": COMPLEXITY_FEATURES,
}
ALL_FEATURES = tuple(
    feature for features in FEATURE_GROUPS.values() for feature in features
)
FEATURE_TO_GROUP = {
    feature: group for group, features in FEATURE_GROUPS.items() for feature in features
}


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


def _validate_protocol() -> dict[str, Any]:
    observed = sha256_file(PROTOCOL)
    if observed != PROTOCOL_SHA256:
        raise ValueError(
            f"v1.4 protocol SHA-256 mismatch: expected {PROTOCOL_SHA256}, "
            f"got {observed}"
        )
    protocol = read_json(PROTOCOL)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected v1.4 protocol id")
    authority = protocol["authority"]
    if authority["confirmation_scoring_allowed_by_this_protocol"] is not False:
        raise ValueError("v1.4 Stage A cannot score confirmation outcomes")
    if authority["production_promotion_allowed"] is not False:
        raise ValueError("v1.4 Stage A cannot promote production")
    if authority["orders_allowed"] is not False:
        raise ValueError("v1.4 Stage A must keep orders disabled")
    discovery = protocol["discovery"]
    if int(discovery["scheduled_sessions"]) != SCHEDULED_DISCOVERY_SESSIONS:
        raise ValueError("v1.4 scheduled discovery count changed")
    if tuple(discovery["slices"]) != tuple(SLICES):
        raise ValueError("v1.4 discovery slices changed")
    catalog = protocol["feature_catalog"]
    if tuple(catalog["existing_g0"]) != EXISTING_FEATURES:
        raise ValueError("v1.4 existing feature catalog changed")
    if tuple(catalog["lagged_session_decomposition"]) != SESSION_FEATURES:
        raise ValueError("v1.4 session feature catalog changed")
    if tuple(catalog["lagged_spectrum"]["channels"]) != SPECTRAL_CHANNELS:
        raise ValueError("v1.4 spectrum channels changed")
    if tuple(catalog["lagged_path_complexity"]["features"]) != COMPLEXITY_FEATURES:
        raise ValueError("v1.4 complexity feature catalog changed")
    if int(catalog["total_features"]) != len(ALL_FEATURES):
        raise ValueError("v1.4 feature count changed")
    return protocol


def _safe_percent(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    top = pd.to_numeric(numerator, errors="coerce")
    bottom = pd.to_numeric(denominator, errors="coerce").where(
        lambda values: values.gt(0.0)
    )
    return 100.0 * (top / bottom - 1.0)


def _safe_range_percent(frame: pd.DataFrame) -> pd.Series:
    span = (
        pd.to_numeric(frame["high"], errors="coerce")
        - pd.to_numeric(frame["low"], errors="coerce")
    )
    prior_close = pd.to_numeric(frame["prior_close"], errors="coerce").where(
        lambda values: values.gt(0.0)
    )
    return 100.0 * span / prior_close


def _rolling_prior(
    values: pd.Series,
    codes: pd.Series,
    session_positions: pd.Series,
    window: int,
    operation: str,
) -> pd.Series:
    filled = pd.to_numeric(values, errors="coerce").fillna(0.0)
    lagged = filled.groupby(codes, sort=False).shift(1)
    rolling = lagged.groupby(codes, sort=False).rolling(
        window, min_periods=window
    )
    if operation == "mean":
        output = rolling.mean()
    elif operation == "std":
        output = rolling.std(ddof=0)
    else:
        raise ValueError(f"unsupported rolling operation: {operation}")
    output = output.reset_index(level=0, drop=True).reindex(values.index)
    oldest = session_positions.groupby(codes, sort=False).shift(window)
    consecutive = (session_positions - oldest).eq(window)
    return output.where(consecutive)


def _correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered * left_centered, axis=1)
        * np.sum(right_centered * right_centered, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    )


def _normalised_entropy(counts: np.ndarray, states: int) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        counts,
        totals,
        out=np.zeros_like(counts, dtype=float),
        where=totals > 0.0,
    )
    terms = np.zeros_like(probabilities, dtype=float)
    positive = probabilities > 0.0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    return -terms.sum(axis=1) / math.log(states)


def _spectral_values(windows: np.ndarray) -> tuple[np.ndarray, ...]:
    median = np.median(windows, axis=2, keepdims=True)
    centered = windows - median
    scale = 1.4826 * np.median(np.abs(centered), axis=2, keepdims=True)
    normalised = np.divide(
        centered,
        scale,
        out=np.zeros_like(centered, dtype=float),
        where=scale > 0.0,
    )
    energy = np.abs(np.fft.rfft(normalised, axis=2)) ** 2
    non_dc = energy[:, :, 1:]
    total = non_dc.sum(axis=2)
    low = np.divide(
        non_dc[:, :, :4].sum(axis=2),
        total,
        out=np.zeros_like(total),
        where=total > 0.0,
    )
    high = np.divide(
        non_dc[:, :, 8:16].sum(axis=2),
        total,
        out=np.zeros_like(total),
        where=total > 0.0,
    )
    probabilities = np.divide(
        non_dc,
        total[:, :, None],
        out=np.zeros_like(non_dc),
        where=total[:, :, None] > 0.0,
    )
    entropy_terms = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    entropy_terms[positive] = probabilities[positive] * np.log(
        probabilities[positive]
    )
    entropy = -entropy_terms.sum(axis=2) / math.log(16.0)
    dominant = (np.argmax(non_dc, axis=2) + 1.0) / 16.0
    dominant[total <= 0.0] = 0.0
    return low, high, entropy, dominant


def _complexity_values(
    channel_windows: np.ndarray,
    range_windows: np.ndarray,
) -> tuple[np.ndarray, ...]:
    oc = channel_windows[:, 0, :]
    am = channel_windows[:, 2, :]
    lunch = channel_windows[:, 3, :]
    pm = channel_windows[:, 4, :]

    oc_sign = np.sign(oc)
    transition = (oc_sign[:, 1:] != oc_sign[:, :-1]).mean(axis=1)
    differences = np.diff(oc, axis=1)
    turning = (differences[:, 1:] * differences[:, :-1] < 0.0).mean(axis=1)

    triples = np.lib.stride_tricks.sliding_window_view(oc, 3, axis=1)
    orders = np.argsort(triples, axis=2, kind="stable")
    pattern_codes = orders[:, :, 0] * 9 + orders[:, :, 1] * 3 + orders[:, :, 2]
    possible_codes = np.asarray((5, 7, 11, 15, 19, 21), dtype=int)
    pattern_counts = np.stack(
        [(pattern_codes == code).sum(axis=1) for code in possible_codes],
        axis=1,
    )
    permutation_entropy = _normalised_entropy(pattern_counts, 6)

    abs_oc = np.abs(oc)
    abs_autocorr = _correlation(abs_oc[:, :-1], abs_oc[:, 1:])
    sign_agreement = (np.sign(am) == np.sign(pm)).mean(axis=1)
    am_pm_corr = _correlation(am, pm)

    component_code = (
        (np.sign(am).astype(int) + 1) * 9
        + (np.sign(lunch).astype(int) + 1) * 3
        + (np.sign(pm).astype(int) + 1)
    )
    component_counts = np.stack(
        [(component_code == code).sum(axis=1) for code in range(27)],
        axis=1,
    )
    component_entropy = _normalised_entropy(component_counts, 27)
    abs_oc_sum = abs_oc.sum(axis=1)
    range_ratio = np.divide(
        range_windows.sum(axis=1),
        abs_oc_sum,
        out=np.zeros_like(abs_oc_sum, dtype=float),
        where=abs_oc_sum > 0.0,
    )
    return (
        transition,
        turning,
        permutation_entropy,
        abs_autocorr,
        sign_agreement,
        am_pm_corr,
        component_entropy,
        range_ratio,
    )


def add_v14_features(
    panel: pd.DataFrame,
    sessions: Iterable[object],
) -> pd.DataFrame:
    """Add the locked 53-feature catalog using only completed prior sessions."""

    required = {
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "prior_close",
        "overnight",
        "am_open",
        "am_close",
        "pm_open",
        "pm_close",
        *EXISTING_FEATURES,
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"v1.4 panel lacks columns: {missing}")
    frame = panel.copy().reset_index(drop=True)
    frame["_v14_original_order"] = np.arange(len(frame))
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("v1.4 panel contains an invalid date")
    frame["code"] = frame["code"].astype(str)
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("v1.4 panel contains duplicate date/code rows")
    frame = frame.sort_values(["code", "date"], kind="stable").reset_index(drop=True)

    calendar = normalize_expected_sessions(sessions)
    session_number = {date: index for index, date in enumerate(calendar)}
    frame["_v14_session_position"] = frame["date"].map(session_number)
    if frame["_v14_session_position"].isna().any():
        raise ValueError("v1.4 panel date is outside the supplied session calendar")
    frame["_v14_session_position"] = frame["_v14_session_position"].astype(int)

    channels = pd.DataFrame(
        {
            "oc": _safe_percent(frame["close"], frame["open"]),
            "overnight": pd.to_numeric(frame["overnight"], errors="coerce"),
            "am": _safe_percent(frame["am_close"], frame["am_open"]),
            "lunch": _safe_percent(frame["pm_open"], frame["am_close"]),
            "pm": _safe_percent(frame["pm_close"], frame["pm_open"]),
        },
        index=frame.index,
    )
    range_pct = _safe_range_percent(frame)
    span = (
        pd.to_numeric(frame["high"], errors="coerce")
        - pd.to_numeric(frame["low"], errors="coerce")
    )
    close_location = (
        (
            pd.to_numeric(frame["close"], errors="coerce")
            - pd.to_numeric(frame["low"], errors="coerce")
        )
        / span.where(span.gt(0.0))
    ).clip(0.0, 1.0)
    code_group = frame["code"]
    frame["prior_am_return_pct"] = channels["am"].groupby(
        code_group, sort=False
    ).shift(1)
    frame["prior_lunch_gap_pct"] = channels["lunch"].groupby(
        code_group, sort=False
    ).shift(1)
    frame["prior_pm_return_pct"] = channels["pm"].groupby(
        code_group, sort=False
    ).shift(1)
    frame["prior_range_pct"] = range_pct.groupby(code_group, sort=False).shift(1)
    frame["prior_close_location"] = close_location.groupby(
        code_group, sort=False
    ).shift(1)
    positions = frame["_v14_session_position"]
    immediately_prior = (
        positions - positions.groupby(code_group, sort=False).shift(1)
    ).eq(1)
    frame.loc[
        ~immediately_prior,
        [
            "prior_am_return_pct",
            "prior_lunch_gap_pct",
            "prior_pm_return_pct",
            "prior_range_pct",
            "prior_close_location",
        ],
    ] = np.nan
    for output_name, channel_name in (
        ("am_mean_5", "am"),
        ("lunch_mean_5", "lunch"),
        ("pm_mean_5", "pm"),
    ):
        frame[output_name] = _rolling_prior(
            channels[channel_name], code_group, positions, 5, "mean"
        )
    for output_name, channel_name in (
        ("am_std_20", "am"),
        ("pm_std_20", "pm"),
    ):
        frame[output_name] = _rolling_prior(
            channels[channel_name], code_group, positions, 20, "std"
        )

    generated = {
        feature: np.full(len(frame), np.nan, dtype=float)
        for feature in (*SPECTRAL_FEATURES, *COMPLEXITY_FEATURES)
    }
    channel_matrix = channels.loc[:, SPECTRAL_CHANNELS].fillna(0.0).to_numpy()
    range_values = pd.to_numeric(range_pct, errors="coerce").fillna(0.0).to_numpy()
    grouped_indices = frame.groupby("code", sort=False).indices
    for indices in grouped_indices.values():
        index = np.asarray(indices, dtype=int)
        count = len(index)
        code_channels = channel_matrix[index]
        code_range = range_values[index]
        code_sessions = positions.iloc[index].to_numpy(dtype=int)

        if count > 32:
            windows32 = np.lib.stride_tricks.sliding_window_view(
                code_channels, 32, axis=0
            )[:-1]
            target_index = index[32:]
            consecutive = (code_sessions[32:] - code_sessions[:-32]) == 32
            low, high, entropy, dominant = _spectral_values(windows32)
            measures = (low, high, entropy, dominant)
            for channel_index, channel in enumerate(SPECTRAL_CHANNELS):
                for measure_index, measure in enumerate(SPECTRAL_MEASURES):
                    values = measures[measure_index][:, channel_index]
                    generated[f"{channel}_{measure}"][
                        target_index[consecutive]
                    ] = values[consecutive]

        if count > 20:
            windows20 = np.lib.stride_tricks.sliding_window_view(
                code_channels, 20, axis=0
            )[:-1]
            range20 = np.lib.stride_tricks.sliding_window_view(
                code_range, 20
            )[:-1]
            target_index = index[20:]
            consecutive = (code_sessions[20:] - code_sessions[:-20]) == 20
            values = _complexity_values(windows20, range20)
            for feature, feature_values in zip(
                COMPLEXITY_FEATURES, values, strict=True
            ):
                generated[feature][target_index[consecutive]] = feature_values[
                    consecutive
                ]

    frame = pd.concat(
        [frame, pd.DataFrame(generated, index=frame.index)],
        axis=1,
    )
    all_feature_frame = frame.loc[:, list(ALL_FEATURES)].replace(
        [np.inf, -np.inf], np.nan
    )
    if all_feature_frame.isna().all().any():
        empty = frame.loc[:, list(ALL_FEATURES)].columns[
            all_feature_frame.isna().all()
        ]
        raise ValueError(f"v1.4 features are entirely missing: {list(empty)}")
    frame.loc[:, list(ALL_FEATURES)] = all_feature_frame
    return (
        frame.sort_values("_v14_original_order", kind="stable")
        .drop(columns=["_v14_original_order"])
        .reset_index(drop=True)
    )


def _cohen_d(positive: np.ndarray, nonpositive: np.ndarray) -> float | None:
    if len(positive) < 2 or len(nonpositive) < 2:
        return None
    numerator = (len(positive) - 1) * np.var(
        positive, ddof=1
    ) + (len(nonpositive) - 1) * np.var(nonpositive, ddof=1)
    denominator = len(positive) + len(nonpositive) - 2
    pooled = math.sqrt(max(float(numerator / denominator), 0.0))
    if pooled == 0.0:
        return 0.0
    return float((np.mean(positive) - np.mean(nonpositive)) / pooled)


def _bh_q_values(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float), kind="stable")
    adjusted = np.ones(count, dtype=float)
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, float(p_values[index]) * count / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def _rank_gap_rows(
    frame: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    available = frame.loc[frame[feature].notna(), ["date", "target_positive", feature]].copy()
    available["feature_rank"] = available.groupby("date", sort=True)[feature].rank(
        method="average", pct=True
    )
    grouped = (
        available.groupby(["date", "target_positive"], sort=True)["feature_rank"]
        .mean()
        .unstack()
        .rename(columns={False: "nonpositive", True: "positive"})
    )
    if "positive" not in grouped or "nonpositive" not in grouped:
        return pd.DataFrame(columns=["date", "rank_gap"])
    grouped = grouped.dropna(subset=["positive", "nonpositive"])
    return grouped.assign(
        rank_gap=grouped["positive"] - grouped["nonpositive"]
    ).reset_index()[["date", "rank_gap"]]


def _feature_result(
    frame: pd.DataFrame,
    feature: str,
) -> dict[str, Any]:
    missing_rate = float(frame[feature].isna().mean())
    available = frame.loc[frame[feature].notna()]
    positive = available.loc[available["target_positive"], feature].to_numpy(
        dtype=float
    )
    nonpositive = available.loc[
        ~available["target_positive"], feature
    ].to_numpy(dtype=float)
    gaps = _rank_gap_rows(frame, feature)
    daily = gaps["rank_gap"].to_numpy(dtype=float)
    mean_gap = float(np.mean(daily)) if len(daily) else 0.0
    if len(daily) >= 2:
        standard_error = float(np.std(daily, ddof=1) / math.sqrt(len(daily)))
    else:
        standard_error = 0.0
    if standard_error > 0.0:
        t_statistic = mean_gap / standard_error
        normal_p = math.erfc(abs(t_statistic) / math.sqrt(2.0))
    elif mean_gap == 0.0:
        t_statistic = 0.0
        normal_p = 1.0
    else:
        t_statistic = math.copysign(float("inf"), mean_gap)
        normal_p = 0.0

    slice_gaps: dict[str, float | None] = {}
    for name, (start, end) in SLICES.items():
        values = gaps.loc[
            gaps["date"].between(start, end, inclusive="both"), "rank_gap"
        ]
        slice_gaps[name] = float(values.mean()) if len(values) else None
    return {
        "feature": feature,
        "group": FEATURE_TO_GROUP[feature],
        "rows": int(len(frame)),
        "available_rows": int(len(available)),
        "missing_rate": missing_rate,
        "positive_rows": int(len(positive)),
        "nonpositive_rows": int(len(nonpositive)),
        "raw": {
            "positive_mean": float(np.mean(positive)) if len(positive) else None,
            "nonpositive_mean": (
                float(np.mean(nonpositive)) if len(nonpositive) else None
            ),
            "positive_median": (
                float(np.median(positive)) if len(positive) else None
            ),
            "nonpositive_median": (
                float(np.median(nonpositive)) if len(nonpositive) else None
            ),
            "cohen_d": _cohen_d(positive, nonpositive),
        },
        "daily_rank_gap": {
            "dates": int(len(daily)),
            "mean": mean_gap,
            "median": float(np.median(daily)) if len(daily) else None,
            "standard_error": standard_error,
            "t_statistic": t_statistic,
            "normal_p_two_sided": float(normal_p),
            "slices": slice_gaps,
        },
    }


def analyze_feature_contrasts(panel: pd.DataFrame) -> dict[str, Any]:
    """Compute the locked discovery contrast without reading confirmation rows."""

    dates = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("v1.4 contrast panel contains an invalid date")
    if dates.ge(FORBIDDEN_CONFIRMATION_START).any():
        raise ValueError("v1.4 Stage A received a forbidden confirmation row")
    eligible = panel.get("price_eligible", pd.Series(False, index=panel.index))
    eligible = eligible.fillna(False).astype(bool)
    open_price = pd.to_numeric(panel["open"], errors="coerce")
    close_price = pd.to_numeric(panel["close"], errors="coerce")
    observed = open_price.gt(0.0) & close_price.gt(0.0)
    discovery = dates.between(DISCOVERY_START, DISCOVERY_END, inclusive="both")
    frame = panel.loc[
        eligible & observed & discovery,
        ["date", "open", "close", *ALL_FEATURES],
    ].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["target_positive"] = (
        pd.to_numeric(frame["close"], errors="coerce")
        > pd.to_numeric(frame["open"], errors="coerce")
    )
    scheduled = pd.DatetimeIndex(sorted(frame["date"].unique()))
    if len(scheduled) != SCHEDULED_DISCOVERY_SESSIONS:
        raise ValueError(
            "v1.4 scheduled discovery sessions changed: "
            f"expected {SCHEDULED_DISCOVERY_SESSIONS}, got {len(scheduled)}"
        )

    features = [_feature_result(frame, feature) for feature in ALL_FEATURES]
    q_values = _bh_q_values(
        [
            float(item["daily_rank_gap"]["normal_p_two_sided"])
            for item in features
        ]
    )
    for item, q_value in zip(features, q_values, strict=True):
        item["bh_q"] = q_value
        aggregate = float(item["daily_rank_gap"]["mean"])
        slice_values = list(item["daily_rank_gap"]["slices"].values())
        sign = int(np.sign(aggregate))
        same_slice_sign = sign != 0 and all(
            value is not None and int(np.sign(float(value))) == sign
            for value in slice_values
        )
        item["stable_signal"] = bool(
            q_value <= 0.05
            and abs(aggregate) >= 0.005
            and float(item["missing_rate"]) <= 0.10
            and same_slice_sign
        )
    features.sort(
        key=lambda item: (
            -abs(float(item["daily_rank_gap"]["mean"])),
            str(item["feature"]),
        )
    )
    return {
        "rows": int(len(frame)),
        "scheduled_sessions": int(len(scheduled)),
        "date_bounds": [
            scheduled.min().strftime("%Y-%m-%d"),
            scheduled.max().strftime("%Y-%m-%d"),
        ],
        "positive_rows": int(frame["target_positive"].sum()),
        "nonpositive_rows": int((~frame["target_positive"]).sum()),
        "feature_count": len(features),
        "stable_signals": [
            item["feature"] for item in features if item["stable_signal"]
        ],
        "features": features,
    }


def _load_panel_cache(
    path: str | Path,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, Any]]:
    value = joblib.load(path)
    checks: dict[str, Any] = {"source": "joblib_panel_cache"}
    if isinstance(value, pd.DataFrame):
        panel = value
        sessions = normalize_expected_sessions(panel["date"])
    elif isinstance(value, dict) and isinstance(value.get("panel"), pd.DataFrame):
        panel = value["panel"]
        sessions = normalize_expected_sessions(value.get("sessions", panel["date"]))
        supplied = value.get("input_checks", value.get("checks"))
        if isinstance(supplied, dict):
            checks.update(supplied)
    elif (
        isinstance(value, tuple)
        and len(value) == 3
        and isinstance(value[0], pd.DataFrame)
    ):
        panel, raw_sessions, supplied = value
        sessions = normalize_expected_sessions(raw_sessions)
        if isinstance(supplied, dict):
            checks.update(supplied)
    else:
        raise ValueError(
            "panel cache must contain a DataFrame, a panel dictionary, "
            "or the v13 (panel, sessions, checks) tuple"
        )
    checks["panel_cache_sha256"] = sha256_file(path)
    checks["panel_rows"] = int(len(panel))
    checks["panel_codes"] = int(panel["code"].astype(str).nunique())
    return panel.copy(), sessions, checks


def load_stage_a_panel(
    *,
    panel_cache: str | Path | None,
    jpx_directory: str | Path | None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, Any]]:
    if (panel_cache is None) == (jpx_directory is None):
        raise ValueError("provide exactly one of panel_cache or jpx_directory")
    if panel_cache is not None:
        panel, sessions, checks = _load_panel_cache(panel_cache)
    else:
        v13_protocol, v13_erratum = validate_v13_protocol()
        panel, sessions, checks = load_and_build_panel(
            jpx_directory, v13_protocol, v13_erratum
        )
        checks = {"source": "official_jpx_pdf", **checks}
    panel_dates = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    if panel_dates.isna().any():
        raise ValueError("v1.4 input panel contains an invalid date")
    expected_input = {
        "panel_rows": 1_524_104,
        "panel_codes": 4_124,
        "sessions": 386,
        "date_start": pd.Timestamp("2024-01-04"),
        "date_end": pd.Timestamp("2025-07-31"),
    }
    observed_input = {
        "panel_rows": int(len(panel)),
        "panel_codes": int(panel["code"].astype(str).nunique()),
        "sessions": int(len(sessions)),
        "date_start": panel_dates.min(),
        "date_end": panel_dates.max(),
    }
    if observed_input != expected_input:
        raise ValueError(
            "v1.4 frozen panel dimensions changed: "
            f"expected {expected_input}, got {observed_input}"
        )
    checks["frozen_panel_dimensions_verified"] = True
    # The feature engine never receives confirmation outcomes.
    stage_a = panel.loc[panel_dates.lt(FORBIDDEN_CONFIRMATION_START)].copy()
    checks["stage_a_rows"] = int(len(stage_a))
    checks["confirmation_rows_used"] = 0
    checks["stage_a_max_date"] = str(
        pd.to_datetime(stage_a["date"]).max().date()
    )
    return stage_a, sessions, checks


def run(
    *,
    protocol: dict[str, Any],
    panel_cache: str | Path | None,
    jpx_directory: str | Path | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    panel, sessions, input_checks = load_stage_a_panel(
        panel_cache=panel_cache,
        jpx_directory=jpx_directory,
    )
    featured = add_v14_features(panel, sessions)
    contrast = analyze_feature_contrasts(featured)
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "authority": {
            "analysis_type": "retrospective_discovery_only",
            "confirmation_outcomes_scored": False,
            "production_model_changed": False,
            "orders_allowed": False,
        },
        "input": input_checks,
        "feature_catalog": {
            "groups": {
                group: list(features) for group, features in FEATURE_GROUPS.items()
            },
            "total_features": len(ALL_FEATURES),
        },
        "contrast": contrast,
        "runtime": {
            "seconds": float(time.perf_counter() - started),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--panel-cache", type=Path)
    source.add_argument("--jpx-directory", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research/model_v14_feature_contrast_result.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = _validate_protocol()
    result = run(
        protocol=protocol,
        panel_cache=args.panel_cache,
        jpx_directory=args.jpx_directory,
    )
    write_json(result, args.output)
    print(json.dumps(result["contrast"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
