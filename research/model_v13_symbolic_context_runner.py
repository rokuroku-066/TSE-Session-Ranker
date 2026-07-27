#!/usr/bin/env python3
"""Run the preregistered v1.3 symbolic-context zero-base experiment.

The candidate family is a deterministic prequential variable-order context
tree.  It treats completed OHLC paths as a symbolic sequence and stores
date-equal reward means by suffix.  It is deliberately separate from the
package's production model and can never enable orders.
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


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research.finalize_logit_v04 import add_bounded_daily_features  # noqa: E402
from tse_session_ranker.config import RankerConfig  # noqa: E402
from tse_session_ranker.data.common import (  # noqa: E402
    normalize_expected_sessions,
    prepare_modeling_prices,
    session_calendar_hash,
)
from tse_session_ranker.features import build_feature_panel  # noqa: E402
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


PROTOCOL = ROOT / "research/model_v13_symbolic_context_protocol.json"
PROTOCOL_SHA256 = "7762221b33781a9d976487e50c1f5483bfc7ec6cea0e2b615d19ed36cf0b9703"
INPUT_SHA256 = "18878ab5597bea531b06e94eef2395a52f94aa64c7265a736bd137a8d48b602d"
PROTOCOL_ID = "model_v13_symbolic_context_zero_base_20260727"
SCORE_START = pd.Timestamp("2024-07-01")
SCORE_END = pd.Timestamp("2025-07-31")
TARGET_CLIP = 10.0
ROW_PRIOR_STRENGTH = 200.0
DATE_PRIOR_STRENGTH = 20.0
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_RANDOM_STATE = 20_260_727
FAMILY_SIZE = 8
CAPACITIES = (1, 2)
CONTROL = "C00_DAILY_RANK_RIDGE"
CANDIDATES = (
    "CT01_ABSOLUTE_PATH",
    "CT02_RELATIVE_PATH",
    "CT03_DIRECTION_PATH",
    "CT04_FIXED_CONSENSUS",
)
ALL_MODELS = (CONTROL, *CANDIDATES)
G0_FEATURES = (
    "oc_last",
    "oc_mean_5",
    "oc_mean_20",
    "oc_mean_60",
    "oc_win_20",
    "oc_std_20",
    "overnight_last",
    "overnight_mean_20",
    "overnight_mean_60",
    "night_day_corr_60",
    "xrank_atr14_pct",
    "xrank_close_momentum_5",
    "xrank_close_momentum_20",
    "xrank_close_momentum_60",
    "xrank_prior_close_location_20",
)
SUBPERIODS = {
    "discovery": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-31")),
    "confirmation_a": (pd.Timestamp("2024-11-01"), pd.Timestamp("2025-03-31")),
    "confirmation_b": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
}
TOKEN_SPECS = {
    "absolute": (82, 3),
    "relative": (82, 3),
    "direction": (10, 6),
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
    payload = json.dumps(
        json_safe(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    target.write_text(payload + "\n", encoding="utf-8")


def _digitize(values: pd.Series, boundaries: Sequence[float]) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.digitize(numeric, np.asarray(boundaries, dtype=float), right=False)
    return result.astype(np.int16, copy=False)


def _rank_tertile(
    frame: pd.DataFrame,
    values: pd.Series,
    traded: pd.Series,
) -> np.ndarray:
    ranked = values.where(traded).groupby(frame["date"], sort=False).rank(
        method="average", pct=True
    )
    numeric = ranked.to_numpy(dtype=float)
    output = np.floor(np.clip(numeric, 0.0, 1.0) * 3.0).astype(
        np.float64, copy=False
    )
    output = np.minimum(output, 2.0)
    output[~np.isfinite(numeric)] = -1.0
    return output.astype(np.int16, copy=False)


def make_symbol_tokens(panel: pd.DataFrame) -> pd.DataFrame:
    """Create current-session tokens.

    These tokens may use the current completed session because target-row
    contexts are formed only after a per-security shift.
    """

    required = {
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "prior_close",
        "overnight",
        "oc_return_pct",
        "traded",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"symbol panel lacks columns: {missing}")
    frame = panel.loc[
        :,
        [
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "prior_close",
            "overnight",
            "oc_return_pct",
            "traded",
        ],
    ].copy()
    traded = (
        frame["traded"].fillna(False).astype(bool)
        & frame[["open", "high", "low", "close"]].notna().all(axis=1)
    )
    safe_prior_close = pd.to_numeric(frame["prior_close"], errors="coerce").where(
        lambda values: values.gt(0.0)
    )
    range_pct = (
        100.0
        * (
            pd.to_numeric(frame["high"], errors="coerce")
            - pd.to_numeric(frame["low"], errors="coerce")
        )
        / safe_prior_close
    )
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
    ).where(span.gt(0.0), 0.5)
    oc = pd.to_numeric(frame["oc_return_pct"], errors="coerce")
    gap = pd.to_numeric(frame["overnight"], errors="coerce")

    abs_oc = _digitize(oc, (-0.5, 0.5))
    abs_gap = _digitize(gap, (-0.5, 0.5))
    abs_range = _digitize(range_pct, (1.0, 3.0))
    abs_location = _digitize(close_location, (1.0 / 3.0, 2.0 / 3.0))
    absolute = (((abs_oc * 3 + abs_gap) * 3 + abs_range) * 3 + abs_location)
    absolute = absolute.astype(np.int16, copy=False)
    absolute[~traded.to_numpy()] = 81

    rel_oc = _rank_tertile(frame, oc, traded)
    rel_gap = _rank_tertile(frame, gap, traded)
    rel_range = _rank_tertile(frame, range_pct, traded)
    rel_location = _rank_tertile(frame, close_location, traded)
    relative = (((rel_oc * 3 + rel_gap) * 3 + rel_range) * 3 + rel_location)
    relative = relative.astype(np.int16, copy=False)
    relative[~traded.to_numpy()] = 81
    invalid_relative = traded.to_numpy() & (
        (rel_oc < 0) | (rel_gap < 0) | (rel_range < 0) | (rel_location < 0)
    )
    relative[invalid_relative] = -1

    direction_oc = _digitize(oc, (-0.25, 0.25))
    direction_gap = _digitize(gap, (-0.25, 0.25))
    direction = (direction_oc * 3 + direction_gap).astype(np.int16, copy=False)
    direction[~traded.to_numpy()] = 9
    invalid_direction = traded.to_numpy() & (
        ~np.isfinite(oc.to_numpy(dtype=float))
        | ~np.isfinite(gap.to_numpy(dtype=float))
    )
    direction[invalid_direction] = -1

    output = frame[["date", "code"]].copy()
    output["absolute_token"] = absolute
    output["relative_token"] = relative
    output["direction_token"] = direction
    return output


def _context_columns(
    frame: pd.DataFrame,
    token_column: str,
    *,
    base: int,
    maximum_depth: int,
    session_index: pd.Series,
) -> dict[str, np.ndarray]:
    token = frame[token_column]
    code = frame["code"]
    current_position = session_index.to_numpy(dtype=np.int32)
    context = np.zeros(len(frame), dtype=np.int64)
    valid = np.ones(len(frame), dtype=bool)
    output: dict[str, np.ndarray] = {}
    for lag in range(1, maximum_depth + 1):
        lagged = token.groupby(code, sort=False).shift(lag)
        lagged_position = session_index.groupby(code, sort=False).shift(lag)
        lag_values = lagged.fillna(-1).to_numpy(dtype=np.int64)
        consecutive = (
            lagged_position.notna().to_numpy()
            & (
                current_position
                - lagged_position.fillna(-10_000).to_numpy(dtype=np.int32)
                == lag
            )
        )
        valid &= consecutive & (lag_values >= 0) & (lag_values < base)
        context = context * base + np.where(valid, lag_values, 0)
        values = context.copy()
        values[~valid] = -1
        output[f"{token_column.removesuffix('_token')}_context_{lag}"] = values
    return output


def add_symbolic_contexts(
    panel: pd.DataFrame,
    sessions: Iterable[object],
) -> pd.DataFrame:
    """Attach strictly prior symbolic suffixes to the canonical panel."""

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["code"] = frame["code"].astype(str)
    frame = frame.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    calendar = normalize_expected_sessions(sessions)
    positions = pd.Series(
        frame["date"].map({date: i for i, date in enumerate(calendar)}),
        index=frame.index,
    )
    if positions.isna().any():
        raise ValueError("symbol panel contains a date outside the frozen calendar")
    tokens = make_symbol_tokens(frame)
    for column in ("absolute_token", "relative_token", "direction_token"):
        frame[column] = tokens[column].to_numpy()
    for family, (base, depth) in TOKEN_SPECS.items():
        columns = _context_columns(
            frame,
            f"{family}_token",
            base=base,
            maximum_depth=depth,
            session_index=positions,
        )
        for name, values in columns.items():
            frame[name] = values
    frame["_session_index"] = positions.to_numpy(dtype=np.int16)
    for family, (_, depth) in TOKEN_SPECS.items():
        source = (
            frame["date"]
            .groupby(frame["code"], sort=False)
            .shift(1)
            .where(frame[f"{family}_context_{depth}"].ge(0))
        )
        if (source.notna() & source.ge(frame["date"])).any():
            raise AssertionError(f"{family} context contains a non-prior source")
    return frame


@dataclass
class ContextLevel:
    sum_weight: np.ndarray
    sum_weighted_target: np.ndarray
    rows: np.ndarray
    dates: np.ndarray


class DenseContextTree:
    """Dense deterministic sufficient statistics for bounded integer words."""

    def __init__(
        self,
        family: str,
        *,
        base: int,
        maximum_depth: int,
        row_prior_strength: float = ROW_PRIOR_STRENGTH,
        date_prior_strength: float = DATE_PRIOR_STRENGTH,
    ) -> None:
        self.family = family
        self.base = int(base)
        self.maximum_depth = int(maximum_depth)
        self.row_prior_strength = float(row_prior_strength)
        self.date_prior_strength = float(date_prior_strength)
        self.levels: list[ContextLevel] = []
        for depth in range(1, maximum_depth + 1):
            size = self.base**depth
            self.levels.append(
                ContextLevel(
                    sum_weight=np.zeros(size, dtype=np.float64),
                    sum_weighted_target=np.zeros(size, dtype=np.float64),
                    rows=np.zeros(size, dtype=np.int32),
                    dates=np.zeros(size, dtype=np.int16),
                )
            )
        self.global_sum_weight = 0.0
        self.global_sum_weighted_target = 0.0
        self.global_rows = 0
        self.global_dates: set[int] = set()

    def add(self, rows: pd.DataFrame) -> None:
        if rows.empty:
            return
        target = pd.to_numeric(rows["oc_return_pct"], errors="coerce")
        valid_target = np.isfinite(target.to_numpy(dtype=float))
        if not valid_target.all():
            raise ValueError("context update contains a missing outcome")
        target_values = np.clip(
            target.to_numpy(dtype=float), -TARGET_CLIP, TARGET_CLIP
        )
        counts = rows.groupby("date", sort=False)["code"].transform("size")
        weights = 1.0 / counts.to_numpy(dtype=float)
        if not np.isfinite(weights).all() or (weights <= 0.0).any():
            raise ValueError("invalid date-equal context weights")
        date_ids = rows["_session_index"].to_numpy(dtype=np.int64)
        self.global_sum_weight += float(weights.sum())
        self.global_sum_weighted_target += float(np.dot(weights, target_values))
        self.global_rows += len(rows)
        self.global_dates.update(int(value) for value in np.unique(date_ids))
        date_base = 1 + int(date_ids.max())
        for depth, level in enumerate(self.levels, start=1):
            keys = rows[f"{self.family}_context_{depth}"].to_numpy(dtype=np.int64)
            valid = keys >= 0
            if not valid.any():
                continue
            local_keys = keys[valid]
            local_weights = weights[valid]
            local_target = target_values[valid]
            np.add.at(level.sum_weight, local_keys, local_weights)
            np.add.at(
                level.sum_weighted_target,
                local_keys,
                local_weights * local_target,
            )
            np.add.at(level.rows, local_keys, 1)
            pairs = local_keys * date_base + date_ids[valid]
            unique_keys = np.unique(pairs) // date_base
            np.add.at(level.dates, unique_keys, 1)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if self.global_sum_weight <= 0.0:
            raise ValueError("context tree has no training outcomes")
        prediction = np.full(
            len(rows),
            self.global_sum_weighted_target / self.global_sum_weight,
            dtype=float,
        )
        for depth, level in enumerate(self.levels, start=1):
            keys = rows[f"{self.family}_context_{depth}"].to_numpy(dtype=np.int64)
            valid = keys >= 0
            if not valid.any():
                continue
            local_keys = keys[valid]
            sum_weight = level.sum_weight[local_keys]
            observed = sum_weight > 0.0
            if not observed.any():
                continue
            row_support = level.rows[local_keys].astype(float)
            date_support = level.dates[local_keys].astype(float)
            reliability = np.minimum(
                row_support / (row_support + self.row_prior_strength),
                date_support / (date_support + self.date_prior_strength),
            )
            context_mean = np.divide(
                level.sum_weighted_target[local_keys],
                sum_weight,
                out=np.zeros_like(sum_weight),
                where=observed,
            )
            local_prediction = prediction[valid]
            local_prediction[observed] = (
                reliability[observed] * context_mean[observed]
                + (1.0 - reliability[observed])
                * local_prediction[observed]
            )
            prediction[valid] = local_prediction
        if not np.isfinite(prediction).all():
            raise AssertionError("context prediction is non-finite")
        return prediction

    def summary(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "base": self.base,
            "maximum_depth": self.maximum_depth,
            "global_rows": int(self.global_rows),
            "global_dates": int(len(self.global_dates)),
            "global_date_equal_mean_pct": (
                float(self.global_sum_weighted_target / self.global_sum_weight)
                if self.global_sum_weight > 0.0
                else None
            ),
            "populated_contexts": [
                int(np.count_nonzero(level.rows)) for level in self.levels
            ],
        }


def _validate_protocol() -> dict[str, Any]:
    actual = sha256_file(PROTOCOL)
    if actual != PROTOCOL_SHA256:
        raise ValueError(
            f"v1.3 protocol SHA-256 mismatch: expected {PROTOCOL_SHA256}, got {actual}"
        )
    protocol = read_json(PROTOCOL)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected v1.3 protocol id")
    if protocol["authority"]["production_promotion_allowed"] is not False:
        raise ValueError("retrospective v1.3 protocol cannot promote production")
    if protocol["authority"]["orders_allowed"] is not False:
        raise ValueError("v1.3 protocol must keep orders disabled")
    if int(protocol["family_size"]) != len(CANDIDATES) * len(CAPACITIES):
        raise ValueError("v1.3 family size differs from implementation")
    return protocol


def load_and_build_panel(
    prices_path: str | Path,
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, Any]]:
    prices_path = Path(prices_path)
    actual_sha = sha256_file(prices_path)
    if actual_sha != INPUT_SHA256:
        raise ValueError(
            f"v1.3 input SHA-256 mismatch: expected {INPUT_SHA256}, got {actual_sha}"
        )
    prices = pd.read_pickle(prices_path)
    frozen = protocol["frozen_input"]
    if list(prices.columns) != frozen["columns"]:
        raise ValueError("v1.3 input columns differ from the protocol")
    dates = pd.to_datetime(prices["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("v1.3 input contains an invalid date")
    sessions = normalize_expected_sessions(dates.drop_duplicates())
    checks = {
        "sha256": actual_sha,
        "rows": int(len(prices)),
        "codes": int(prices["code"].astype(str).nunique()),
        "sessions": int(len(sessions)),
        "date_bounds": [
            sessions.min().strftime("%Y-%m-%d"),
            sessions.max().strftime("%Y-%m-%d"),
        ],
        "calendar_sha256": session_calendar_hash(sessions),
    }
    expected = {
        "rows": int(frozen["rows"]),
        "codes": int(frozen["codes"]),
        "sessions": int(frozen["sessions"]),
        "date_bounds": list(frozen["date_bounds"]),
    }
    for name, value in expected.items():
        if checks[name] != value:
            raise ValueError(
                f"v1.3 input {name} changed: expected {value}, got {checks[name]}"
            )
    settings = RankerConfig()
    modeling, coverage = prepare_modeling_prices(
        prices,
        coverage_lookback=settings.source_coverage_lookback,
        minimum_source_coverage=settings.minimum_source_coverage,
        expected_sessions=sessions,
    )
    incomplete = coverage.loc[~coverage["source_complete"], "date"]
    if len(incomplete):
        raise ValueError(
            "v1.3 official-price source is incomplete on: "
            + ", ".join(str(pd.Timestamp(value).date()) for value in incomplete[:5])
        )
    panel = build_feature_panel(modeling, settings)
    panel = add_bounded_daily_features(panel)
    panel["price_eligible"] = panel["eligible"].astype(bool)
    panel["price_training_eligible"] = panel["training_eligible"].astype(bool)
    panel = add_symbolic_contexts(panel, sessions)
    scheduled = sessions[(sessions >= SCORE_START) & (sessions <= SCORE_END)]
    if len(scheduled) != int(protocol["evaluation"]["scheduled_score_sessions"]):
        raise ValueError("v1.3 scheduled score-session count changed")
    checks.update(
        {
            "modeling_rows": int(len(panel)),
            "modeling_codes": int(panel["code"].nunique()),
            "source_incomplete_sessions": int(len(incomplete)),
            "score_sessions": int(len(scheduled)),
            "strictly_prior_context_source_violations": 0,
        }
    )
    return panel, sessions, checks


def _top_two(
    scoring: pd.DataFrame,
    values: np.ndarray,
    candidate_id: str,
) -> pd.DataFrame:
    if len(scoring) != len(values):
        raise ValueError("score length differs from scoring rows")
    ranked = scoring.assign(model_score=np.asarray(values, dtype=float))
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
    result = desired.merge(
        actual.loc[:, keep],
        on=["date", "model_rank", "candidate_id"],
        how="left",
        validate="one_to_one",
        sort=True,
    )
    if len(result) != len(scheduled) * 2:
        raise AssertionError("v1.3 slot completion changed the schedule")
    return result


def run_symbolic(
    panel: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    trees = {
        family: DenseContextTree(family, base=base, maximum_depth=depth)
        for family, (base, depth) in TOKEN_SPECS.items()
    }
    update_projection = [
        "date",
        "code",
        "oc_return_pct",
        "_session_index",
        *[
            f"{family}_context_{depth}"
            for family, (_, maximum_depth) in TOKEN_SPECS.items()
            for depth in range(1, maximum_depth + 1)
        ],
    ]
    training_mask = (
        panel["date"].lt(SCORE_START)
        & panel["price_training_eligible"]
        & panel["oc_return_pct"].notna()
    )
    initial = panel.loc[training_mask, update_projection]
    if initial.empty or initial["date"].max() >= SCORE_START:
        raise ValueError("v1.3 initial context training is invalid")
    for tree in trees.values():
        tree.add(initial)

    parts: dict[str, list[pd.DataFrame]] = {
        candidate: [] for candidate in CANDIDATES
    }
    training_max_by_score_date: dict[str, str] = {}
    for date in scheduled:
        scoring = panel.loc[
            panel["date"].eq(date) & panel["price_eligible"],
            [
                "date",
                "code",
                "name",
                "label",
                "oc_return_pct",
                *[
                    f"{family}_context_{depth}"
                    for family, (_, maximum_depth) in TOKEN_SPECS.items()
                    for depth in range(1, maximum_depth + 1)
                ],
            ],
        ].copy()
        score_eligible = np.ones(len(scoring), dtype=bool)
        for family in TOKEN_SPECS:
            score_eligible &= scoring[f"{family}_context_1"].ge(0).to_numpy()
        scoring = scoring.loc[score_eligible].copy()
        absolute = trees["absolute"].predict(scoring)
        relative = trees["relative"].predict(scoring)
        direction = trees["direction"].predict(scoring)
        values_by_candidate = {
            "CT01_ABSOLUTE_PATH": absolute,
            "CT02_RELATIVE_PATH": relative,
            "CT03_DIRECTION_PATH": direction,
            "CT04_FIXED_CONSENSUS": (absolute + relative + direction) / 3.0,
        }
        for candidate_id, values in values_by_candidate.items():
            parts[candidate_id].append(_top_two(scoring, values, candidate_id))

        update = panel.loc[
            panel["date"].eq(date)
            & panel["price_training_eligible"]
            & panel["oc_return_pct"].notna(),
            update_projection,
        ]
        if not update.empty:
            if not update["date"].lt(date + pd.Timedelta(days=1)).all():
                raise AssertionError("v1.3 update date is invalid")
            for tree in trees.values():
                tree.add(update)
        training_max_by_score_date[str(date.date())] = str(date.date())

    picks = {
        candidate: _complete_slots(
            pd.concat(parts[candidate], ignore_index=True),
            scheduled,
            candidate,
        )
        for candidate in CANDIDATES
    }
    details = {
        "initial_train_rows": int(len(initial)),
        "initial_train_start": str(initial["date"].min().date()),
        "initial_train_end": str(initial["date"].max().date()),
        "scoring_before_same_date_update": True,
        "same_date_outcome_first_usable_on_following_session": True,
        "final_trees": {family: tree.summary() for family, tree in trees.items()},
        "score_dates": len(training_max_by_score_date),
    }
    return picks, details


def run_control(
    panel: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    periods = pd.period_range(SCORE_START.to_period("M"), SCORE_END.to_period("M"))
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
        name="model_v13_control_daily_rank_ridge",
        family="ridge_daily_rank",
        objective="same_day_return_percentile",
        parameters={"alpha": 1.0},
    )
    for period in periods:
        first = period.start_time.normalize()
        last = period.end_time.normalize()
        training = panel.loc[
            panel["date"].between(pd.Timestamp("2024-01-04"), first - pd.Timedelta(days=1))
            & panel["price_training_eligible"]
            & panel["oc_return_pct"].notna(),
            projection,
        ].copy()
        scoring = panel.loc[
            panel["date"].between(first, last) & panel["price_eligible"],
            projection,
        ].copy()
        if training.empty or training["date"].max() >= first:
            raise ValueError(f"v1.3 control fold {period} has invalid training")
        scorer = fit_research_model(specification, training, G0_FEATURES)
        values = scorer.score(scoring)
        parts.append(_top_two(scoring, values, CONTROL))
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
    actual = pd.concat(parts, ignore_index=True)
    return _complete_slots(actual, scheduled, CONTROL), folds


def _daily(
    picks: pd.DataFrame,
    *,
    capacity: int,
    cost_bps: float,
) -> pd.Series:
    frame = daily_portfolio_returns(
        picks,
        top_k=capacity,
        cost_bps=cost_bps,
    )
    values = frame.set_index("date")["net_return_pct"].sort_index()
    if len(values) != 266 or values.isna().any():
        raise AssertionError("v1.3 daily return schedule is incomplete")
    return values


def _top_codes_cash(
    picks: pd.DataFrame,
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
    mask = neutral["model_rank"].le(capacity) & neutral["code"].astype(str).isin(codes)
    neutral.loc[mask, ["label", "oc_return_pct"]] = np.nan
    return _daily(neutral, capacity=capacity, cost_bps=cost_bps), codes


def _variant_metrics(
    picks: pd.DataFrame,
    control: pd.DataFrame,
    *,
    candidate_id: str,
    capacity: int,
) -> dict[str, Any]:
    costs = {
        str(int(cost)): profit_metrics(
            picks,
            top_k=capacity,
            cost_bps=cost,
        )
        for cost in (20.0, 40.0, 60.0)
    }
    daily40 = _daily(picks, capacity=capacity, cost_bps=40.0)
    control40 = _daily(control, capacity=capacity, cost_bps=40.0)
    bonferroni_confidence = 1.0 - (1.0 - 0.90) / FAMILY_SIZE
    paired = paired_moving_block_bootstrap(
        daily40,
        control40,
        block_length=5,
        samples=BOOTSTRAP_SAMPLES,
        confidence=bonferroni_confidence,
        random_state=BOOTSTRAP_RANDOM_STATE + CAPACITIES.index(capacity),
    )
    subperiods = {
        name: float(values.loc[start:end].mean())
        for name, (start, end) in SUBPERIODS.items()
        for values in (daily40,)
    }
    top10_removed = float(daily40.drop(daily40.nlargest(10).index).mean())
    top_codes_cash, top_codes = _top_codes_cash(
        picks,
        capacity=capacity,
        cost_bps=40.0,
        count=10,
    )
    monthly = daily40.groupby(daily40.index.to_period("M")).mean()
    selected = picks[picks["model_rank"].le(capacity)]
    signaled = selected.dropna(subset=["code"])
    counts = signaled["code"].astype(str).value_counts()
    unique_codes = int(len(counts))
    maximum_code_share = float(counts.iloc[0] / len(signaled)) if len(signaled) else 1.0
    checks = {
        "net40_positive": float(daily40.mean()) > 0.0,
        "all_three_subperiods_positive": all(value > 0.0 for value in subperiods.values()),
        "top10_days_removed_positive": top10_removed > 0.0,
        "top10_profit_codes_cash_positive": float(top_codes_cash.mean()) > 0.0,
        "positive_months_at_least_9_of_13": int(monthly.gt(0.0).sum()) >= 9,
        "familywise_paired_lower_vs_control_nonnegative": (
            paired.one_sided_lower_delta_pct >= 0.0
        ),
        "unique_codes_at_least_50": unique_codes >= 50,
        "maximum_code_share_at_most_10pct": maximum_code_share <= 0.10,
    }
    return {
        "variant_id": f"{candidate_id}__top{capacity}",
        "candidate_id": candidate_id,
        "capacity": capacity,
        "cost_metrics": costs,
        "subperiod_net40_mean_pct": subperiods,
        "top10_days_removed_net40_mean_pct": top10_removed,
        "top10_profit_codes_cash_net40_mean_pct": float(top_codes_cash.mean()),
        "top10_profit_codes": top_codes,
        "positive_months_net40": int(monthly.gt(0.0).sum()),
        "months": int(len(monthly)),
        "monthly_net40_mean_pct": {
            str(period): float(value) for period, value in monthly.items()
        },
        "paired_vs_control_net40": asdict(paired),
        "bonferroni_individual_confidence": bonferroni_confidence,
        "unique_selected_codes": unique_codes,
        "maximum_code_selection_share": maximum_code_share,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def _semantic_hash(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    canonical["date"] = pd.to_datetime(canonical["date"]).dt.strftime("%Y-%m-%d")
    canonical = canonical.sort_values(
        ["candidate_id", "date", "model_rank"], kind="stable"
    ).reset_index(drop=True)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", default="jpx_daily_2024_2025.pkl")
    parser.add_argument(
        "--output",
        default="research/model_v13_symbolic_context_result.json",
    )
    parser.add_argument(
        "--picks-output",
        default="research/model_v13_symbolic_context_picks.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    protocol = _validate_protocol()
    panel, sessions, input_checks = load_and_build_panel(args.prices, protocol)
    scheduled = sessions[(sessions >= SCORE_START) & (sessions <= SCORE_END)]
    symbolic, symbolic_details = run_symbolic(panel, scheduled)
    control, control_folds = run_control(panel, scheduled)
    picks_by_model = {CONTROL: control, **symbolic}
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
    picks_path = Path(args.picks_output)
    picks_path.parent.mkdir(parents=True, exist_ok=True)
    all_picks.to_csv(picks_path, index=False, lineterminator="\n")

    control_metrics = {
        f"top{capacity}": {
            str(int(cost)): profit_metrics(
                control,
                top_k=capacity,
                cost_bps=cost,
            )
            for cost in (20.0, 40.0, 60.0)
        }
        for capacity in CAPACITIES
    }
    variants = [
        _variant_metrics(
            symbolic[candidate],
            control,
            candidate_id=candidate,
            capacity=capacity,
        )
        for candidate in CANDIDATES
        for capacity in CAPACITIES
    ]
    passers = [item["variant_id"] for item in variants if item["gate_passed"]]
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "input": input_checks,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "seconds": float(time.monotonic() - started),
        },
        "integrity": {
            "candidate_count": len(CANDIDATES),
            "capacity_variants": list(CAPACITIES),
            "family_size": len(variants),
            "scheduled_sessions": len(scheduled),
            "slots_per_model": len(scheduled) * 2,
            "picks_rows": int(len(all_picks)),
            "picks_semantic_sha256": _semantic_hash(all_picks),
            "scoring_before_same_date_update": True,
            "strictly_prior_training": True,
            "orders_allowed": False,
            "production_model_changed": False,
        },
        "symbolic_model": symbolic_details,
        "control": {
            "candidate_id": CONTROL,
            "folds": control_folds,
            "metrics": control_metrics,
        },
        "variants": variants,
        "decision": {
            "gate_passers": passers,
            "forward_shadow_candidate": passers[0] if len(passers) == 1 else None,
            "ambiguous_multiple_passers": len(passers) > 1,
            "production_candidate": None,
            "production_model_changed": False,
            "orders_allowed": False,
            "conclusion": (
                "One preregistered variant cleared every retrospective gate and may only be frozen for a new forward shadow."
                if len(passers) == 1
                else "No unique preregistered variant qualified for a forward shadow."
            ),
        },
    }
    write_json(result, args.output)


if __name__ == "__main__":
    main()
