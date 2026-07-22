#!/usr/bin/env python3
"""Run the frozen v0.7 ranking-error and zero-to-two-slot gate replay.

The runner deliberately separates three objects which older diagnostics mixed:

* ``DISPLAY`` is always the G0 anchor's first two ranks;
* ``FORCED`` is the counterfactual which trades both 50 percent sleeves; and
* ``GATED`` independently leaves either displayed sleeve in cash.

Only 2024-07 through 2024-10 may select a gate.  The selected gate is then
replayed, without replacement or re-selection, over the registered A/B/C
periods.  Every historical outcome is already known, so even a passing replay
is research evidence only and can never promote production.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import gc
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tse_session_ranker.exceptions import DataValidationError, LeakageError  # noqa: E402
from tse_session_ranker.io import write_frame, write_json  # noqa: E402
from tse_session_ranker.research_models import (  # noqa: E402
    ResearchModelSpec,
    fit_research_model,
)
from tse_session_ranker.validation import moving_block_bootstrap  # noqa: E402


PROTOCOL_PATH = ROOT / "research/model_v07_error_protocol.json"
EXPECTED_PROTOCOL_SHA256 = (
    "31cb5512b197ce59f418bc4021af164d362a5be6c6ad06454f4508f2eec48ebd"
)
RESULT_SCHEMA_VERSION = 1
BLOCK_LENGTH = 5
BOOTSTRAP_SAMPLES = 5_000
DEFAULT_SEED = 31
TOP_K = 2


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    source = Path(path)
    if source.resolve() == PROTOCOL_PATH.resolve():
        actual = sha256_file(source)
        if actual != EXPECTED_PROTOCOL_SHA256:
            raise ValueError("v0.7 error protocol changed after registration")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported v0.7 protocol schema")
    if raw.get("status") != "frozen_before_model_v07_metrics":
        raise ValueError("v0.7 error protocol is not frozen")
    gates = raw.get("gate_registry", [])
    if not 1 <= len(gates) <= 4:
        raise ValueError("v0.7 must register between one and four gates")
    gate_ids = [str(value["gate_id"]) for value in gates]
    if len(gate_ids) != len(set(gate_ids)):
        raise ValueError("v0.7 gate ids must be unique")
    feature_groups = raw["meta_features"]
    for gate in gates:
        unknown = sorted(set(gate["feature_groups"]) - set(feature_groups))
        if unknown:
            raise ValueError(f"gate references unknown meta groups: {unknown}")
    periods = {
        key: (pd.Timestamp(value["start"]), pd.Timestamp(value["end"]))
        for key, value in raw["periods"].items()
    }
    for key, (start, end) in periods.items():
        if start > end:
            raise ValueError(f"v0.7 period {key} is reversed")
    discovery_end = periods["error_discovery"][1]
    replay_order = ("replay_a", "replay_b", "replay_c")
    previous = discovery_end
    for key in replay_order:
        start, end = periods[key]
        if start <= previous:
            raise ValueError("v0.7 discovery and replay periods overlap")
        previous = end
    specs = [ResearchModelSpec.from_dict(value) for value in raw["family_registry"]]
    names = [value.name for value in specs]
    if len(names) != len(set(names)):
        raise ValueError("v0.7 family registry names must be unique")
    if raw["decision"]["display_anchor"] not in names:
        raise ValueError("v0.7 display anchor is absent from the family registry")
    return raw


def _periods(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    output: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for value in pd.period_range(start.to_period("M"), end.to_period("M"), freq="M"):
        output.append(
            (
                max(start, value.start_time.normalize()),
                min(end, value.end_time.normalize()),
                str(value),
            )
        )
    return output


def _manifest_path(panel_path: Path) -> Path:
    return panel_path.with_suffix(panel_path.suffix + ".manifest.json")


def load_v06_panel(
    panel_path: str | Path,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, Any]]:
    """Load the immutable v0.6 panel and enforce the v0.7 PIT projection."""

    source = Path(panel_path)
    manifest_path = _manifest_path(source)
    if not source.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "v0.7 requires the previously built v0.6 panel and manifest"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("panel_file_sha256") != sha256_file(source):
        raise ValueError("v0.6 panel bytes differ from its manifest")
    panel = joblib.load(source)
    required = {
        "date",
        "code",
        "name",
        "label",
        "oc_return_pct",
        "outcome_observed",
        "source_complete",
        "universe_source_complete",
        "price_eligible",
        "price_training_eligible",
        "feature_source_max_date",
        "candidate_price_source_max_date",
        *protocol["g0_features"],
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise DataValidationError(f"v0.6 panel lacks v0.7 columns: {missing}")
    frame = panel.loc[:, list(dict.fromkeys(required))].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if frame["date"].isna().any():
        raise DataValidationError("v0.6 panel contains an invalid date")
    if frame[["date", "code"]].duplicated().any():
        raise DataValidationError("v0.6 panel contains duplicate date/code rows")
    for source_column in (
        "feature_source_max_date",
        "candidate_price_source_max_date",
    ):
        source_date = pd.to_datetime(frame[source_column], errors="coerce")
        leaked = source_date.notna() & source_date.ge(frame["date"])
        if leaked.any():
            raise LeakageError(f"{source_column} is not strictly prior")
    sessions = pd.DatetimeIndex(pd.to_datetime(manifest["sessions"])).normalize()
    if not sessions.is_monotonic_increasing or sessions.has_duplicates:
        raise DataValidationError("v0.6 manifest sessions are invalid")
    return frame, sessions, manifest


def _rank_positions(frame: pd.DataFrame, score_column: str) -> pd.Series:
    ordered = frame.sort_values(
        ["date", score_column, "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    positions = ordered.groupby("date", sort=False).cumcount().add(1)
    result = pd.Series(index=ordered.index, data=positions.to_numpy(), dtype="int64")
    return result.reindex(frame.index)


def _date_iqr(values: pd.Series, dates: pd.Series) -> tuple[pd.Series, pd.Series]:
    grouped = values.groupby(dates, sort=False)
    median = grouped.transform("median")
    q25 = grouped.transform(lambda value: value.quantile(0.25))
    q75 = grouped.transform(lambda value: value.quantile(0.75))
    return median, (q75 - q25)


def build_month_meta_features(
    scoring: pd.DataFrame,
    family_scores: Mapping[str, np.ndarray],
    *,
    anchor_name: str,
) -> pd.DataFrame:
    """Create top-two meta rows from scores that are already PIT-safe.

    No outcome column participates in ranking, normalization, margins, votes,
    or agreement.  Keeping this function pure makes the invariance directly
    testable by mutating current outcomes.
    """

    required = {"date", "code", "oc_return_pct", "outcome_observed"}
    missing = sorted(required - set(scoring.columns))
    if missing:
        raise DataValidationError(f"scoring frame lacks columns: {missing}")
    if anchor_name not in family_scores:
        raise ValueError("anchor score is missing")
    names = list(family_scores)
    if not names:
        raise ValueError("family score registry is empty")
    frame = scoring.copy().reset_index(drop=True)
    for name, values in family_scores.items():
        array = np.asarray(values, dtype=float)
        if array.shape != (len(frame),) or not np.isfinite(array).all():
            raise DataValidationError(f"invalid scores for {name}")
        score_column = f"family_score__{name}"
        frame[score_column] = array
        rank_column = f"family_rank__{name}"
        frame[rank_column] = _rank_positions(frame, score_column)
        counts = frame.groupby("date", sort=False)["code"].transform("size")
        denominator = (counts - 1).where(counts.gt(1), 1)
        frame[f"family_rank_pct__{name}"] = 1.0 - (
            frame[rank_column] - 1.0
        ) / denominator
        median, iqr = _date_iqr(frame[score_column], frame["date"])
        safe_iqr = iqr.where(iqr.abs().gt(1e-12))
        frame[f"family_score_z__{name}"] = (
            (frame[score_column] - median) / safe_iqr
        ).fillna(0.0)
        frame[f"family_score_iqr__{name}"] = iqr.fillna(0.0)

    anchor_score = f"family_score__{anchor_name}"
    anchor_rank = f"family_rank__{anchor_name}"
    anchor_pct = f"family_rank_pct__{anchor_name}"
    anchor_z = f"family_score_z__{anchor_name}"
    anchor_iqr = f"family_score_iqr__{anchor_name}"
    display = frame.loc[frame[anchor_rank].le(TOP_K)].copy()
    display["model_rank"] = display[anchor_rank].astype(int)
    display = display.sort_values(["date", "model_rank"], kind="stable")

    ordered_anchor = frame.loc[
        frame[anchor_rank].isin([1, 2, 3, 10]),
        ["date", anchor_rank, anchor_score],
    ].copy()
    score_at_rank = ordered_anchor.pivot(
        index="date", columns=anchor_rank, values=anchor_score
    ).rename(columns=lambda value: f"_anchor_score_rank_{int(value)}")
    display = display.merge(
        score_at_rank.reset_index(), on="date", how="left", validate="many_to_one"
    )
    local_raw = np.where(
        display["model_rank"].eq(1),
        display["_anchor_score_rank_1"] - display["_anchor_score_rank_2"],
        display["_anchor_score_rank_2"] - display["_anchor_score_rank_3"],
    )
    safe_iqr = display[anchor_iqr].where(display[anchor_iqr].abs().gt(1e-12))
    display["local_margin_iqr"] = pd.Series(local_raw, index=display.index) / safe_iqr
    display["top_tail_slope_iqr"] = (
        display["_anchor_score_rank_2"] - display["_anchor_score_rank_10"]
    ) / safe_iqr
    display["anchor_score_tail_z"] = display[anchor_z]
    display["anchor_rank_percentile"] = display[anchor_pct]
    display["anchor_score_iqr"] = display[anchor_iqr]
    display["slot_is_rank2"] = display["model_rank"].eq(2).astype(float)

    rank_pct_columns = [f"family_rank_pct__{name}" for name in names]
    rank_columns = [f"family_rank__{name}" for name in names]
    display["family_rank_percentile_mean"] = display[rank_pct_columns].mean(axis=1)
    display["family_rank_percentile_std"] = display[rank_pct_columns].std(
        axis=1, ddof=0
    )
    display["family_rank_percentile_min"] = display[rank_pct_columns].min(axis=1)
    display["family_top2_vote_rate"] = display[rank_columns].le(2).mean(axis=1)
    display["family_top5_vote_rate"] = display[rank_columns].le(5).mean(axis=1)
    correlation_by_date: dict[pd.Timestamp, float] = {}
    for date, group in frame.groupby("date", sort=True):
        correlation = group[rank_pct_columns].corr(method="spearman").to_numpy()
        upper = correlation[np.triu_indices(len(names), k=1)]
        finite = upper[np.isfinite(upper)]
        correlation_by_date[pd.Timestamp(date)] = (
            float(finite.mean()) if len(finite) else 0.0
        )
    display["family_rank_correlation_mean"] = display["date"].map(
        correlation_by_date
    )
    display["anchor_model_score"] = display[anchor_score]
    display["family_model_count"] = len(names)
    drop_internal = [
        column
        for column in display.columns
        if column.startswith("_anchor_score_rank_")
    ]
    return display.drop(columns=drop_internal).reset_index(drop=True)


def add_strictly_prior_repeat_features(display: pd.DataFrame) -> pd.DataFrame:
    """Attach selection-history features, updating state only after each date."""

    frame = display.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if frame[["date", "model_rank"]].duplicated().any():
        raise DataValidationError("display has duplicate date/rank slots")
    ordered_dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    date_position = {date: position for position, date in enumerate(ordered_dates)}
    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    values: dict[str, list[Any]] = defaultdict(list)
    ordered = frame.sort_values(["date", "model_rank"], kind="stable")
    for date, group in ordered.groupby("date", sort=True):
        current_position = date_position[pd.Timestamp(date)]
        pending: list[tuple[str, dict[str, Any]]] = []
        for _, row in group.iterrows():
            code = str(row["code"])
            prior = history[code]
            recent5 = [item for item in prior if current_position - item["position"] <= 5]
            recent20 = [item for item in prior if current_position - item["position"] <= 20]
            last = prior[-1] if prior else None
            values["prior_selection_count_5"].append(float(len(recent5)))
            values["prior_selection_count_20"].append(float(len(recent20)))
            values["sessions_since_selected"].append(
                float(current_position - last["position"]) if last else np.nan
            )
            values["last_selected_net_pct"].append(
                float(last["net_return_pct"]) if last and np.isfinite(last["net_return_pct"]) else np.nan
            )
            values["consecutive_selection"].append(
                float(bool(last and current_position - last["position"] == 1))
            )
            values["anchor_score_change_since_selected"].append(
                float(row["anchor_model_score"] - last["anchor_model_score"])
                if last
                else np.nan
            )
            values["anchor_score_age_sessions"].append(
                float(current_position - last["position"]) if last else np.nan
            )
            values["repeat_feature_source_max_date"].append(
                last["date"] if last else pd.NaT
            )
            observed = bool(row.get("outcome_observed", False)) and pd.notna(
                row.get("oc_return_pct")
            )
            net_return = (
                float(row["oc_return_pct"]) - 0.2 if observed else np.nan
            )
            pending.append(
                (
                    code,
                    {
                        "position": current_position,
                        "date": pd.Timestamp(date),
                        "net_return_pct": net_return,
                        "anchor_model_score": float(row["anchor_model_score"]),
                    },
                )
            )
        for code, item in pending:
            history[code].append(item)
    for column, column_values in values.items():
        ordered[column] = column_values
    leaked = ordered["repeat_feature_source_max_date"].notna() & ordered[
        "repeat_feature_source_max_date"
    ].ge(ordered["date"])
    if leaked.any():
        raise LeakageError("repeat meta features use a non-prior selection")
    return ordered.sort_index().reset_index(drop=True)


def _family_specs(protocol: Mapping[str, Any]) -> list[ResearchModelSpec]:
    return [ResearchModelSpec.from_dict(value) for value in protocol["family_registry"]]


def generate_display_meta(
    panel: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Generate monthly out-of-fold family scores and the fixed G0 top two."""

    discovery_start = pd.Timestamp(protocol["periods"]["error_discovery"]["start"])
    replay_end = pd.Timestamp(protocol["periods"]["replay_c"]["end"])
    train_start = pd.Timestamp(protocol["base_training"]["start"])
    minimum_sessions = int(protocol["base_training"]["minimum_sessions"])
    features = tuple(protocol["g0_features"])
    anchor_name = str(protocol["decision"]["display_anchor"])
    specs = _family_specs(protocol)
    projection = list(
        dict.fromkeys(
            [
                "date",
                "code",
                "name",
                "label",
                "oc_return_pct",
                "outcome_observed",
                "source_complete",
                "universe_source_complete",
                "feature_source_max_date",
                "candidate_price_source_max_date",
                *features,
            ]
        )
    )
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for score_start, score_end, period_name in _periods(discovery_start, replay_end):
        training = panel.loc[
            panel["date"].between(train_start, score_start - pd.Timedelta(days=1))
            & panel["price_training_eligible"].eq(True)
            & panel["label"].notna(),
            projection,
        ].copy()
        scoring = panel.loc[
            panel["date"].between(score_start, score_end)
            & panel["price_eligible"].eq(True),
            projection,
        ].copy()
        train_days = int(training["date"].nunique())
        if train_days < minimum_sessions:
            raise ValueError(
                f"v0.7 has {train_days} training sessions before {score_start.date()}"
            )
        if scoring.empty:
            continue
        if not training["date"].max() < scoring["date"].min():
            raise LeakageError("base training overlaps a scoring month")
        family_scores: dict[str, np.ndarray] = {}
        for spec in specs:
            fitted = fit_research_model(spec, training, features)
            family_scores[spec.name] = fitted.score(scoring)
            del fitted
            gc.collect()
        month_meta = build_month_meta_features(
            scoring,
            family_scores,
            anchor_name=anchor_name,
        )
        month_meta["base_model_train_end"] = training["date"].max()
        month_meta["base_model_train_sessions"] = train_days
        parts.append(month_meta)
        folds.append(
            {
                "period": period_name,
                "train_start": str(training["date"].min().date()),
                "train_end": str(training["date"].max().date()),
                "train_sessions": train_days,
                "train_rows": int(len(training)),
                "score_start": str(scoring["date"].min().date()),
                "score_end": str(scoring["date"].max().date()),
                "score_sessions": int(scoring["date"].nunique()),
                "score_rows": int(len(scoring)),
                "family_models": [spec.name for spec in specs],
            }
        )
        del training, scoring, family_scores, month_meta
        gc.collect()
    if not parts:
        raise ValueError("v0.7 produced no display rows")
    display = add_strictly_prior_repeat_features(
        pd.concat(parts, ignore_index=True)
    )
    desired_sessions = sessions[
        (sessions >= discovery_start) & (sessions <= replay_end)
    ]
    expected = pd.MultiIndex.from_product(
        [desired_sessions, [1, 2]], names=["date", "model_rank"]
    )
    actual = pd.MultiIndex.from_frame(display[["date", "model_rank"]])
    missing_slots = expected.difference(actual)
    if len(missing_slots):
        raise DataValidationError(
            f"v0.7 display is missing {len(missing_slots)} registered slots"
        )
    if display["base_model_train_end"].ge(display["date"]).any():
        raise LeakageError("a display row was scored by a non-prior base model")
    return display.sort_values(["date", "model_rank"]).reset_index(drop=True), folds


def _meta_feature_columns(
    protocol: Mapping[str, Any], gate: Mapping[str, Any]
) -> tuple[str, ...]:
    groups = protocol["meta_features"]
    return tuple(
        dict.fromkeys(
            column
            for group_name in gate["feature_groups"]
            for column in groups[group_name]
        )
    )


def _candidate_date_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("date", sort=False)["date"].transform("size")
    weights = 1.0 / counts.to_numpy(dtype=float)
    totals = pd.Series(weights).groupby(frame["date"].reset_index(drop=True)).sum()
    if not np.allclose(totals.to_numpy(), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("meta weights are not date-equal")
    return weights


def _ridge_pipeline(alpha: float = 10.0) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def fit_monthly_meta_gates(
    display: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    """Fit each registered gate on strictly prior OOF displayed candidates."""

    selection_start = pd.Timestamp(protocol["periods"]["gate_selection"]["start"])
    replay_end = pd.Timestamp(protocol["periods"]["replay_c"]["end"])
    minimum_days = int(protocol["meta_training"]["minimum_prior_candidate_sessions"])
    clip = float(protocol["meta_training"]["target_clip_pct"])
    penalty = float(protocol["meta_training"]["downside_penalty"])
    rows: list[pd.DataFrame] = []
    for gate in protocol["gate_registry"]:
        gate_id = str(gate["gate_id"])
        columns = _meta_feature_columns(protocol, gate)
        missing = sorted(set(columns) - set(display.columns))
        if missing:
            raise DataValidationError(f"meta features are missing: {missing}")
        for score_start, score_end, period_name in _periods(selection_start, replay_end):
            training = display.loc[
                display["date"].lt(score_start)
                & display["outcome_observed"].eq(True)
                & display["oc_return_pct"].notna()
            ].copy()
            scoring = display.loc[display["date"].between(score_start, score_end)].copy()
            if scoring.empty:
                continue
            train_days = int(training["date"].nunique())
            if train_days < minimum_days:
                raise ValueError(
                    f"gate {gate_id} has only {train_days} prior candidate sessions"
                )
            if not training["date"].max() < scoring["date"].min():
                raise LeakageError("meta training overlaps its scoring month")
            target = (
                pd.to_numeric(training["oc_return_pct"], errors="coerce") - 0.2
            ).clip(-clip, clip)
            weights = _candidate_date_weights(training)
            mean_model = _ridge_pipeline()
            mean_model.fit(
                training.loc[:, list(columns)],
                target,
                model__sample_weight=weights,
            )
            predicted_mean = np.asarray(
                mean_model.predict(scoring.loc[:, list(columns)]), dtype=float
            )
            predicted_downside = np.zeros(len(scoring), dtype=float)
            if gate_id == "downside_utility":
                downside_target = (-target).clip(lower=0.0, upper=clip)
                downside_model = _ridge_pipeline()
                downside_model.fit(
                    training.loc[:, list(columns)],
                    downside_target,
                    model__sample_weight=weights,
                )
                predicted_downside = np.clip(
                    np.asarray(
                        downside_model.predict(scoring.loc[:, list(columns)]),
                        dtype=float,
                    ),
                    0.0,
                    None,
                )
            utility = predicted_mean - penalty * predicted_downside
            if not (
                np.isfinite(predicted_mean).all()
                and np.isfinite(predicted_downside).all()
                and np.isfinite(utility).all()
            ):
                raise DataValidationError("meta gate produced invalid predictions")
            output = scoring[["date", "model_rank", "code"]].copy()
            output["gate_id"] = gate_id
            output["predicted_net_mean_pct"] = predicted_mean
            output["predicted_negative_part_pct"] = predicted_downside
            output["trade_utility_pct"] = utility
            output["trade_decision"] = utility > 0.0
            output["meta_train_start"] = training["date"].min()
            output["meta_train_end"] = training["date"].max()
            output["meta_train_sessions"] = train_days
            output["meta_feature_count"] = len(columns)
            output["meta_score_period"] = period_name
            rows.append(output)
    if not rows:
        raise ValueError("v0.7 produced no gate predictions")
    result = pd.concat(rows, ignore_index=True)
    if result["meta_train_end"].ge(result["date"]).any():
        raise LeakageError("meta gate used a non-prior candidate outcome")
    expected_gates = {str(value["gate_id"]) for value in protocol["gate_registry"]}
    if set(result["gate_id"]) != expected_gates:
        raise DataValidationError("one or more registered meta gates are missing")
    return result.sort_values(["gate_id", "date", "model_rank"]).reset_index(drop=True)


def make_trade_picks(
    display: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    *,
    gate_id: str,
    forced: bool = False,
) -> pd.DataFrame:
    """Materialize fixed 50/50 sleeves without rank replacement."""

    keys = ["date", "model_rank", "code"]
    frame = display.copy()
    if frame[["date", "model_rank"]].duplicated().any():
        raise DataValidationError("display slots are not unique")
    if forced:
        frame["gate_id"] = gate_id
        frame["predicted_net_mean_pct"] = np.nan
        frame["predicted_negative_part_pct"] = np.nan
        frame["trade_utility_pct"] = np.nan
        frame["trade_decision"] = frame["code"].notna()
        frame["meta_train_end"] = pd.NaT
        frame["meta_train_sessions"] = np.nan
    else:
        if predictions is None:
            raise ValueError("gated trade picks require predictions")
        selected = predictions[predictions["gate_id"].eq(gate_id)].copy()
        if selected.duplicated(keys).any():
            raise DataValidationError("gate predictions contain duplicate slots")
        before_codes = frame[keys].copy()
        frame = frame.merge(
            selected[
                [
                    *keys,
                    "gate_id",
                    "predicted_net_mean_pct",
                    "predicted_negative_part_pct",
                    "trade_utility_pct",
                    "trade_decision",
                    "meta_train_end",
                    "meta_train_sessions",
                ]
            ],
            on=keys,
            how="left",
            validate="one_to_one",
        )
        if not before_codes.reset_index(drop=True).equals(frame[keys].reset_index(drop=True)):
            raise LeakageError("trade gate replaced a displayed code")
        frame["gate_id"] = frame["gate_id"].fillna(gate_id)
        frame["trade_decision"] = frame["trade_decision"].fillna(False).astype(bool)
    observed = (
        frame["outcome_observed"].eq(True)
        & frame["oc_return_pct"].notna()
        & frame["code"].notna()
    )
    frame["executed"] = frame["trade_decision"] & observed
    frame["gross_sleeve_return_pct"] = np.where(
        frame["executed"], 0.5 * frame["oc_return_pct"].astype(float), 0.0
    )
    for cost in (20.0, 40.0):
        frame[f"net_sleeve_return_pct_{int(cost)}bp"] = np.where(
            frame["executed"],
            0.5 * (frame["oc_return_pct"].astype(float) - cost / 100.0),
            0.0,
        )
    frame["trade_reason"] = np.select(
        [
            ~frame["trade_decision"],
            frame["trade_decision"] & ~observed,
            frame["executed"],
        ],
        ["utility_nonpositive_or_warmup", "unobserved_outcome", "utility_positive"],
        default="unknown",
    )
    if not set(frame["model_rank"].dropna().astype(int).unique()).issubset({1, 2}):
        raise LeakageError("trade picks contain a replacement rank")
    return frame.sort_values(["date", "model_rank"]).reset_index(drop=True)


def _profit_factor(values: pd.Series) -> float:
    gain = float(values.clip(lower=0).sum())
    loss = float(-values.clip(upper=0).sum())
    return gain / loss if loss else float("inf")


def _daily_returns(trades: pd.DataFrame, cost_bps: float = 20.0) -> pd.Series:
    column = f"net_sleeve_return_pct_{int(cost_bps)}bp"
    if column not in trades:
        raise DataValidationError(f"trade picks lack {column}")
    return trades.groupby("date", sort=True)[column].sum().astype(float)


def trade_portfolio_metrics(
    trades: pd.DataFrame,
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    daily20 = _daily_returns(trades, 20.0)
    daily40 = _daily_returns(trades, 40.0)
    if daily20.empty:
        raise ValueError("trade portfolio has no scheduled days")
    monthly = daily20.groupby(daily20.index.to_period("M")).mean()
    equity = (1.0 + daily20 / 100.0).cumprod()
    with_initial = np.concatenate(([1.0], equity.to_numpy()))
    peak = np.maximum.accumulate(with_initial)
    drawdown = with_initial / peak - 1.0
    top_days = daily20.nlargest(min(5, len(daily20))).index
    without_top = daily20.drop(top_days)
    positive_total = float(daily20.clip(lower=0).sum())
    interval = moving_block_bootstrap(
        daily20,
        block_length=BLOCK_LENGTH,
        samples=bootstrap_samples,
        confidence=0.90,
        random_state=DEFAULT_SEED,
    )
    executed_by_day = trades.groupby("date", sort=True)["executed"].sum()
    rank2 = trades[trades["model_rank"].eq(2)]
    rank2_daily = rank2.groupby("date", sort=True)[
        "net_sleeve_return_pct_20bp"
    ].sum().reindex(daily20.index, fill_value=0.0)
    return {
        "scheduled_days": int(len(daily20)),
        "trade_days": int(executed_by_day.gt(0).sum()),
        "executed_slots": int(trades["executed"].sum()),
        "trade_slot_rate": float(trades["executed"].mean()),
        "net20_mean_pct": float(daily20.mean()),
        "net20_median_pct": float(daily20.median()),
        "net40_mean_pct": float(daily40.mean()),
        "profit_factor": _profit_factor(daily20),
        "positive_months": int(monthly.gt(0).sum()),
        "months": int(len(monthly)),
        "positive_month_fraction": float(monthly.gt(0).mean()),
        "top5_removed_net20_mean_pct": float(without_top.mean()) if len(without_top) else np.nan,
        "largest_day_share_of_positive_profit": (
            float(daily20.max() / positive_total) if positive_total > 0 else np.nan
        ),
        "compounded_net20_pct": float(100.0 * (equity.iloc[-1] - 1.0)),
        "max_drawdown_pct": float(100.0 * drawdown.min()),
        "rank2_marginal_net20_mean_pct": float(rank2_daily.mean()),
        "monthly_net20_mean_pct": {
            str(key): float(value) for key, value in monthly.items()
        },
        "block5_bootstrap": asdict(interval),
    }


def paired_uplift_metrics(
    candidate: pd.DataFrame,
    forced: pd.DataFrame,
    *,
    confidence: float = 0.90,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    candidate_daily = _daily_returns(candidate)
    forced_daily = _daily_returns(forced)
    paired = pd.concat(
        [candidate_daily.rename("candidate"), forced_daily.rename("forced")],
        axis=1,
    )
    if paired.isna().any(axis=None):
        raise DataValidationError("paired portfolios have different scheduled days")
    delta = paired["candidate"] - paired["forced"]
    interval = moving_block_bootstrap(
        delta,
        block_length=BLOCK_LENGTH,
        samples=samples,
        confidence=confidence,
        random_state=DEFAULT_SEED + 1,
    )
    return {
        "candidate_mean_pct": float(candidate_daily.mean()),
        "forced_mean_pct": float(forced_daily.mean()),
        "mean_uplift_pct": float(delta.mean()),
        "block5_bootstrap": asdict(interval),
    }


def _circular_block_indices(
    observations: int,
    *,
    samples: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    if observations < 1:
        raise ValueError("bootstrap observations must be positive")
    rng = np.random.default_rng(seed)
    blocks = math.ceil(observations / block_length)
    starts = rng.integers(0, observations, size=(samples, blocks))
    offsets = np.arange(block_length)
    return ((starts[..., None] + offsets) % observations).reshape(samples, -1)[
        :, :observations
    ]


def shared_block_max_mean_adjustment(
    forced: pd.DataFrame,
    candidates: Mapping[str, pd.DataFrame],
    *,
    confidence: float = 0.80,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, dict[str, float]]:
    """Return a correctly named shared-block maximum-mean adjustment."""

    base = _daily_returns(forced)
    names = list(candidates)
    deltas: list[np.ndarray] = []
    for name in names:
        candidate = _daily_returns(candidates[name])
        paired = pd.concat([base.rename("base"), candidate.rename("candidate")], axis=1)
        if paired.isna().any(axis=None) or len(paired) != len(base):
            raise DataValidationError("selection portfolios have different days")
        deltas.append((paired["candidate"] - paired["base"]).to_numpy(float))
    matrix = np.column_stack(deltas)
    means = matrix.mean(axis=0)
    centered = matrix - means
    indices = _circular_block_indices(
        len(matrix),
        samples=samples,
        block_length=BLOCK_LENGTH,
        seed=DEFAULT_SEED,
    )
    bootstrap = centered[indices].mean(axis=1)
    critical = float(np.quantile(bootstrap.max(axis=1), confidence))
    return {
        name: {
            "mean_uplift_pct": float(means[position]),
            "shared_block_max_mean_critical_pct": critical,
            "adjusted_one_sided_80pct_lower_pct": float(means[position] - critical),
        }
        for position, name in enumerate(names)
    }


def select_gate(
    display: pd.DataFrame,
    predictions: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any], dict[str, pd.DataFrame]]:
    period = protocol["periods"]["gate_selection"]
    start, end = pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
    selection_display = display[display["date"].between(start, end)].copy()
    forced = make_trade_picks(
        selection_display,
        gate_id="forced_top2",
        forced=True,
    )
    candidate_trades: dict[str, pd.DataFrame] = {}
    candidate_metrics: dict[str, Any] = {}
    for gate in protocol["gate_registry"]:
        gate_id = str(gate["gate_id"])
        trades = make_trade_picks(
            selection_display,
            predictions[predictions["date"].between(start, end)],
            gate_id=gate_id,
        )
        candidate_trades[gate_id] = trades
        candidate_metrics[gate_id] = trade_portfolio_metrics(trades)
    adjusted = shared_block_max_mean_adjustment(forced, candidate_trades)
    threshold = protocol["selection_gate"]
    qualifications: dict[str, Any] = {}
    qualified: list[str] = []
    feature_count = {
        str(gate["gate_id"]): len(_meta_feature_columns(protocol, gate))
        for gate in protocol["gate_registry"]
    }
    for gate_id, metrics in candidate_metrics.items():
        checks = {
            "minimum_trade_days": metrics["trade_days"]
            >= int(threshold["minimum_trade_days"]),
            "minimum_trade_slots": metrics["executed_slots"]
            >= int(threshold["minimum_trade_slots"]),
            "net20_mean": metrics["net20_mean_pct"] > 0.0,
            "top5_removed": metrics["top5_removed_net20_mean_pct"] >= 0.0,
            "positive_month_fraction": metrics["positive_month_fraction"]
            >= float(threshold["minimum_positive_month_fraction"]),
            "paired_uplift": adjusted[gate_id]["mean_uplift_pct"] > 0.0,
            "adjusted_lower": adjusted[gate_id][
                "adjusted_one_sided_80pct_lower_pct"
            ]
            >= 0.0,
            "largest_day_share": np.isfinite(
                metrics["largest_day_share_of_positive_profit"]
            )
            and metrics["largest_day_share_of_positive_profit"]
            <= float(threshold["largest_day_share_of_positive_profit_max"]),
        }
        passed = all(checks.values())
        if passed:
            qualified.append(gate_id)
        qualifications[gate_id] = {
            "passed": passed,
            "checks": checks,
            "metrics": metrics,
            "feature_count": feature_count[gate_id],
            **adjusted[gate_id],
        }
    qualified.sort(
        key=lambda gate_id: (
            -candidate_metrics[gate_id]["net20_mean_pct"],
            -adjusted[gate_id]["adjusted_one_sided_80pct_lower_pct"],
            -candidate_metrics[gate_id]["top5_removed_net20_mean_pct"],
            feature_count[gate_id],
            gate_id,
        )
    )
    selected = qualified[0] if qualified else None
    report = {
        "period": {"start": str(start.date()), "end": str(end.date())},
        "forced_metrics": trade_portfolio_metrics(forced),
        "qualifications": qualifications,
        "qualified_gates": qualified,
        "locked_gate": selected,
        "multiple_testing": "shared five-session block maximum-mean adjustment",
    }
    return selected, report, candidate_trades


def classify_error_cases(
    display: pd.DataFrame,
    protocol: Mapping[str, Any],
    selected_trades: pd.DataFrame | None = None,
) -> pd.DataFrame:
    period = protocol["periods"]["error_discovery"]
    frame = display[
        display["date"].between(pd.Timestamp(period["start"]), pd.Timestamp(period["end"]))
    ].copy()
    frame["forced_slot_net_pct"] = frame["oc_return_pct"] - 0.2
    threshold = frame["anchor_score_tail_z"].quantile(
        float(protocol["error_analysis"]["high_confidence_quantile"])
    )
    frame["error_types"] = ""

    def add_type(mask: pd.Series, name: str) -> None:
        current = frame.loc[mask, "error_types"]
        frame.loc[mask, "error_types"] = np.where(
            current.eq(""), name, current + "," + name
        )

    add_type(
        frame["oc_return_pct"].le(
            float(protocol["error_analysis"]["catastrophic_loss_pct"])
        ),
        "catastrophic_loss",
    )
    add_type(
        frame["oc_return_pct"].ge(
            float(protocol["error_analysis"]["large_win_pct"])
        ),
        "large_win",
    )
    add_type(
        frame["anchor_score_tail_z"].ge(threshold)
        & frame["forced_slot_net_pct"].lt(0),
        "high_confidence_loss",
    )
    add_type(
        frame["prior_selection_count_5"].gt(0)
        & frame["last_selected_net_pct"].lt(0)
        & frame["forced_slot_net_pct"].lt(0),
        "repeat_loss",
    )
    if selected_trades is not None and not selected_trades.empty:
        selected = selected_trades[["date", "model_rank", "code", "trade_decision"]]
        frame = frame.merge(
            selected,
            on=["date", "model_rank", "code"],
            how="left",
            validate="one_to_one",
        )
        decision_known = frame["trade_decision"].notna()
        off = decision_known & ~frame["trade_decision"].fillna(False)
        add_type(off & frame["forced_slot_net_pct"].lt(0), "avoided_loss")
        add_type(off & frame["forced_slot_net_pct"].gt(0), "missed_win")
    else:
        frame["trade_decision"] = pd.NA
    result = frame[frame["error_types"].ne("")].copy()
    result["high_confidence_tail_z_threshold"] = float(threshold)
    keep = [
        "date",
        "model_rank",
        "code",
        "name",
        "oc_return_pct",
        "forced_slot_net_pct",
        "anchor_model_score",
        "anchor_score_tail_z",
        "local_margin_iqr",
        "family_top2_vote_rate",
        "family_rank_percentile_std",
        "prior_selection_count_5",
        "last_selected_net_pct",
        "trade_decision",
        "error_types",
        "high_confidence_tail_z_threshold",
    ]
    return result.loc[:, keep].sort_values(
        ["date", "model_rank"], kind="stable"
    ).reset_index(drop=True)


def build_slice_report(
    display: pd.DataFrame,
    forced: pd.DataFrame,
    gated: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    gate_id: str,
) -> pd.DataFrame:
    discovery = protocol["periods"]["error_discovery"]
    discovery_rows = display[
        display["date"].between(
            pd.Timestamp(discovery["start"]), pd.Timestamp(discovery["end"])
        )
    ]
    margin_cut = float(discovery_rows["local_margin_iqr"].median())
    agreement_cut = float(discovery_rows["family_top2_vote_rate"].median())
    meta = display[[
        "date",
        "model_rank",
        "code",
        "local_margin_iqr",
        "family_top2_vote_rate",
        "prior_selection_count_5",
    ]].copy()
    joined = forced[[
        "date", "model_rank", "code", "net_sleeve_return_pct_20bp"
    ]].rename(columns={"net_sleeve_return_pct_20bp": "forced_sleeve_net"})
    joined = joined.merge(
        gated[[
            "date",
            "model_rank",
            "code",
            "net_sleeve_return_pct_20bp",
            "executed",
        ]].rename(columns={"net_sleeve_return_pct_20bp": "gated_sleeve_net"}),
        on=["date", "model_rank", "code"],
        how="left",
        validate="one_to_one",
    ).merge(
        meta,
        on=["date", "model_rank", "code"],
        how="left",
        validate="one_to_one",
    )
    joined["uplift"] = joined["gated_sleeve_net"] - joined["forced_sleeve_net"]
    joined["margin_bucket"] = np.where(
        joined["local_margin_iqr"].ge(margin_cut), "high", "low"
    )
    joined["agreement_bucket"] = np.where(
        joined["family_top2_vote_rate"].ge(agreement_cut), "high", "low"
    )
    joined["repeat_bucket"] = np.where(
        joined["prior_selection_count_5"].gt(0), "repeat", "new"
    )
    period_map = {
        key: (pd.Timestamp(value["start"]), pd.Timestamp(value["end"]))
        for key, value in protocol["periods"].items()
        if key in {"gate_selection", "replay_a", "replay_b", "replay_c"}
    }
    rows: list[dict[str, Any]] = []
    for period_name, (start, end) in period_map.items():
        period_rows = joined[joined["date"].between(start, end)]
        for column in ("model_rank", "margin_bucket", "agreement_bucket", "repeat_bucket"):
            for value, group in period_rows.groupby(column, dropna=False, sort=True):
                rows.append(
                    {
                        "gate_id": gate_id,
                        "period": period_name,
                        "slice": column,
                        "value": value,
                        "candidate_rows": int(len(group)),
                        "executed_rows": int(group["executed"].sum()),
                        "trade_rate": float(group["executed"].mean()),
                        "forced_sleeve_net_mean_pct": float(group["forced_sleeve_net"].mean()),
                        "gated_sleeve_net_mean_pct": float(group["gated_sleeve_net"].mean()),
                        "uplift_mean_pct": float(group["uplift"].mean()),
                        "margin_cut_from_discovery": margin_cut,
                        "agreement_cut_from_discovery": agreement_cut,
                    }
                )
    return pd.DataFrame(rows)


def _subset_trades(trades: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return trades[trades["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()


def replay_report(
    forced: pd.DataFrame,
    gated: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    names = ("replay_a", "replay_b", "replay_c")
    period_results: dict[str, Any] = {}
    positive_periods = 0
    every_period_beats_forced = True
    replay_parts_forced: list[pd.DataFrame] = []
    replay_parts_gated: list[pd.DataFrame] = []
    for name in names:
        period = protocol["periods"][name]
        period_forced = _subset_trades(forced, period["start"], period["end"])
        period_gated = _subset_trades(gated, period["start"], period["end"])
        metrics = trade_portfolio_metrics(period_gated)
        paired = paired_uplift_metrics(period_gated, period_forced)
        positive_periods += int(metrics["net20_mean_pct"] > 0.0)
        every_period_beats_forced &= paired["mean_uplift_pct"] > 0.0
        minimum_fraction = float(
            protocol["replay_gate"]["individual_period"][
                "minimum_trade_day_fraction"
            ]
        )
        period_results[name] = {
            "period": period,
            "forced_metrics": trade_portfolio_metrics(period_forced),
            "gated_metrics": metrics,
            "paired_uplift": paired,
            "checks": {
                "paired_uplift_positive": paired["mean_uplift_pct"] > 0.0,
                "minimum_trade_day_fraction": metrics["trade_days"]
                >= math.ceil(minimum_fraction * metrics["scheduled_days"]),
            },
        }
        replay_parts_forced.append(period_forced)
        replay_parts_gated.append(period_gated)
    combined_forced = pd.concat(replay_parts_forced, ignore_index=True)
    combined_gated = pd.concat(replay_parts_gated, ignore_index=True)
    combined_metrics = trade_portfolio_metrics(combined_gated)
    combined_paired = paired_uplift_metrics(combined_gated, combined_forced)
    gate = protocol["replay_gate"]["combined"]
    checks = {
        "net20_mean": combined_metrics["net20_mean_pct"] > 0.0,
        "block5_lower": combined_metrics["block5_bootstrap"][
            "one_sided_lower_pct"
        ]
        >= 0.0,
        "paired_lower": combined_paired["block5_bootstrap"][
            "one_sided_lower_pct"
        ]
        >= 0.0,
        "top5_removed": combined_metrics["top5_removed_net20_mean_pct"] > 0.0,
        "net40": combined_metrics["net40_mean_pct"] >= 0.0,
        "profit_factor": combined_metrics["profit_factor"]
        >= float(gate["profit_factor_min"]),
        "positive_month_fraction": combined_metrics["positive_month_fraction"]
        >= float(gate["positive_month_fraction_min"]),
        "minimum_trade_days": combined_metrics["trade_days"]
        >= int(gate["minimum_trade_days"]),
        "largest_day_share": np.isfinite(
            combined_metrics["largest_day_share_of_positive_profit"]
        )
        and combined_metrics["largest_day_share_of_positive_profit"]
        <= float(gate["largest_day_share_of_positive_profit_max"]),
        "rank2_marginal": combined_metrics["rank2_marginal_net20_mean_pct"] >= 0.0,
        "minimum_positive_replay_periods": positive_periods
        >= int(gate["minimum_positive_replay_periods"]),
        "all_replay_periods_beat_forced": every_period_beats_forced,
    }
    return {
        "periods": period_results,
        "combined": {
            "forced_metrics": trade_portfolio_metrics(combined_forced),
            "gated_metrics": combined_metrics,
            "paired_uplift": combined_paired,
            "positive_replay_periods": positive_periods,
            "checks": checks,
            "passed": all(checks.values()),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(PROTOCOL_PATH.relative_to(ROOT)))
    parser.add_argument("--panel-cache", default="/tmp/model_v06_feature_panel.pkl")
    parser.add_argument("--output", default="research/model_v07_error_result.json")
    return parser.parse_args()


def main() -> None:
    run_started_at = pd.Timestamp.now(tz="Asia/Tokyo")
    args = _parse_args()
    protocol_path = Path(args.protocol)
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    protocol = load_protocol(protocol_path)
    registered_at = pd.Timestamp(protocol["registered_at"])
    if registered_at.tzinfo is None or registered_at >= run_started_at:
        raise ValueError("v0.7 protocol must be registered before the run starts")
    panel, sessions, panel_manifest = load_v06_panel(args.panel_cache, protocol)
    display, base_folds = generate_display_meta(panel, sessions, protocol)
    predictions = fit_monthly_meta_gates(display, protocol)
    locked_gate, selection, discovery_trade_candidates = select_gate(
        display, predictions, protocol
    )

    forced = make_trade_picks(display, gate_id="forced_top2", forced=True)
    if locked_gate is None:
        locked_predictions = predictions.iloc[0:0].copy()
        gated = make_trade_picks(
            display,
            locked_predictions,
            gate_id="no_qualified_gate",
        )
        replay: dict[str, Any] = {
            "status": "not_evaluated_without_a_qualified_discovery_gate"
        }
    else:
        gated = make_trade_picks(
            display,
            predictions,
            gate_id=locked_gate,
        )
        replay = replay_report(forced, gated, protocol)

    discovery_selected = (
        discovery_trade_candidates.get(locked_gate)
        if locked_gate is not None
        else None
    )
    errors = classify_error_cases(display, protocol, discovery_selected)
    slices = (
        build_slice_report(
            display,
            forced,
            gated,
            protocol,
            gate_id=locked_gate or "no_qualified_gate",
        )
        if locked_gate is not None
        else pd.DataFrame()
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.with_suffix("")
    display_path = Path(f"{stem}_display_picks.csv")
    forced_path = Path(f"{stem}_forced_trade_picks.csv")
    gated_path = Path(f"{stem}_gated_trade_picks.csv")
    errors_path = Path(f"{stem}_error_cases.csv")
    slices_path = Path(f"{stem}_slice_report.csv")
    write_frame(display, display_path)
    write_frame(forced, forced_path)
    write_frame(gated, gated_path)
    write_frame(errors, errors_path)
    write_frame(slices, slices_path)

    artifacts = {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
        }
        for name, path, frame in (
            ("display_picks", display_path, display),
            ("forced_trade_picks", forced_path, forced),
            ("gated_trade_picks", gated_path, gated),
            ("error_cases", errors_path, errors),
            ("slice_report", slices_path, slices),
        )
    }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha256_file(protocol_path),
        "authority": protocol["authority"],
        "decision": (
            "known_retrospective_gate_passed_no_production_promotion"
            if locked_gate is not None
            and replay.get("combined", {}).get("passed", False)
            else "known_retrospective_no_demonstrated_gate"
        ),
        "production_model_changed": False,
        "display_anchor": protocol["decision"]["display_anchor"],
        "family_registry": [value.canonical_dict() for value in _family_specs(protocol)],
        "base_folds": base_folds,
        "selection": selection,
        "locked_gate": locked_gate,
        "replay": replay,
        "error_analysis": {
            "period": protocol["periods"]["error_discovery"],
            "error_case_rows": int(len(errors)),
            "error_type_counts": {
                name: int(errors["error_types"].str.contains(name, regex=False).sum())
                for name in protocol["error_analysis"]["registered_error_types"]
            },
        },
        "panel": {
            "path": str(Path(args.panel_cache)),
            "sha256": panel_manifest["panel_file_sha256"],
            "manifest_path": str(_manifest_path(Path(args.panel_cache))),
            "manifest_sha256": sha256_file(_manifest_path(Path(args.panel_cache))),
            "rows": panel_manifest["rows"],
            "codes": panel_manifest["codes"],
            "calendar_sha256": panel_manifest["calendar_sha256"],
        },
        "artifacts": artifacts,
        "runtime": {
            "run_started_at": run_started_at.isoformat(),
            "run_completed_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "thread_limit": 1,
        },
        "limitations": [
            "Every replay outcome is already known and is not a sealed holdout.",
            "Exact historical 08:58 futures, board, PTS and indicative-open values are absent and were not proxied.",
            "The gate uses fixed 20 bp cost and the v0.6 price-only source limitations.",
        ],
    }
    write_json(result, output_path)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "schema_version": 1,
        "result_path": str(output_path.relative_to(ROOT)),
        "result_sha256": sha256_file(output_path),
        "protocol_sha256": sha256_file(protocol_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "panel_sha256": panel_manifest["panel_file_sha256"],
        "artifacts": artifacts,
    }
    write_json(manifest, manifest_path)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
