"""Point-in-time market regimes and shadow policies introduced in v0.9.

This module is intentionally separate from :mod:`research_candidates`.
Earlier research artifacts bind that module byte-for-byte, so adding a new
feature there would make the v0.7/v0.8 audit chain unreproducible.

The v0.9 breadth feature measures the fraction of valid, traded securities
whose completed open-to-close return was positive.  It is shifted by one
exchange session before being attached to a scoring row.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import DataValidationError, LeakageError


PRIOR_MARKET_OC_BREADTH = "prior_market_oc_breadth"
PRIOR_MARKET_OC_BREADTH_SOURCE_DATE = (
    "prior_market_oc_breadth_source_date"
)
V09_VALID_OC_BREADTH_RANK2_POLICY = "v09_valid_oc_breadth_rank2"


def _require_columns(
    frame: pd.DataFrame, columns: set[str], source: str
) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise DataValidationError(f"{source} is missing columns: {missing}")


def _normalise_dates(values: pd.Series, source: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if parsed.isna().any():
        raise DataValidationError(f"{source} contains an invalid date")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return parsed.dt.normalize()


def _strict_boolean(values: pd.Series, column: str) -> pd.Series:
    def convert(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            return bool(value)
        raise DataValidationError(
            f"{column} must contain only non-missing booleans"
        )

    if values.isna().any():
        raise DataValidationError(
            f"{column} must contain only non-missing booleans"
        )
    return values.map(convert).astype(bool)


def add_prior_market_oc_breadth(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the strictly-prior valid open-to-close market breadth.

    The source denominator is:

    ``traded & outcome_observed & source_complete & oc_return_pct.notna()``.

    Flat but valid trades remain in the denominator as non-positive.  Explicit
    no-trade, unobserved, incomplete, and missing-return rows are excluded.
    The first available session has no prior value.
    """

    required = {
        "date",
        "code",
        "oc_return_pct",
        "traded",
        "outcome_observed",
        "source_complete",
    }
    _require_columns(panel, required, "v0.9 regime panel")
    frame = panel.copy()
    dates = _normalise_dates(frame["date"], "v0.9 regime panel")
    codes = frame["code"].astype(str).str.strip()
    if codes.eq("").any():
        raise DataValidationError("v0.9 regime panel contains an empty code")
    keys = pd.DataFrame({"date": dates, "code": codes})
    if keys.duplicated(["date", "code"]).any():
        raise DataValidationError(
            "v0.9 regime panel contains duplicate date/code rows"
        )

    traded = _strict_boolean(frame["traded"], "traded")
    observed = _strict_boolean(
        frame["outcome_observed"], "outcome_observed"
    )
    complete = _strict_boolean(frame["source_complete"], "source_complete")
    returns = pd.to_numeric(frame["oc_return_pct"], errors="coerce")
    valid = traded & observed & complete & returns.notna()
    if not np.isfinite(returns.loc[valid].to_numpy(dtype=float)).all():
        raise DataValidationError(
            "valid v0.9 open-to-close returns must be finite"
        )

    sessions = pd.DatetimeIndex(sorted(dates.unique()))
    positive = returns.gt(0).astype(float).where(valid)
    source_breadth = (
        positive.groupby(dates, sort=True).mean().reindex(sessions)
    )
    prior_breadth = source_breadth.shift(1)
    source_dates = pd.Series(sessions, index=sessions).shift(1)
    breadth = dates.map(prior_breadth)
    feature_source = dates.map(source_dates)

    leaked = feature_source.notna() & feature_source.ge(dates)
    if leaked.any():
        raise LeakageError(
            "v0.9 market OC breadth contains a non-prior source date"
        )
    finite = breadth.dropna()
    if not finite.between(0.0, 1.0, inclusive="both").all():
        raise DataValidationError("v0.9 market OC breadth must be in [0, 1]")

    frame[PRIOR_MARKET_OC_BREADTH] = breadth.astype("float32")
    frame[PRIOR_MARKET_OC_BREADTH_SOURCE_DATE] = feature_source
    return frame


def select_valid_oc_breadth_rank2(
    picks: pd.DataFrame,
    context: pd.DataFrame,
    *,
    low_breadth_candidate: str,
    high_breadth_candidate: str,
    threshold: float = 0.5,
    model_rank: int = 2,
) -> pd.DataFrame:
    """Select one rank-2 shadow candidate using prior valid OC breadth.

    Breadth below ``threshold`` selects ``low_breadth_candidate``; breadth at
    or above it selects ``high_breadth_candidate``.  Missing breadth falls
    back to the low-breadth candidate, matching the frozen v0.9 protocol.
    """

    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    if model_rank < 1:
        raise ValueError("model_rank must be positive")
    if not low_breadth_candidate or not high_breadth_candidate:
        raise ValueError("candidate identifiers must be non-empty")

    _require_columns(
        picks,
        {"date", "candidate", "model_rank", "code"},
        "v0.9 source picks",
    )
    _require_columns(
        context,
        {
            "date",
            PRIOR_MARKET_OC_BREADTH,
            PRIOR_MARKET_OC_BREADTH_SOURCE_DATE,
        },
        "v0.9 breadth context",
    )
    ranked = picks.copy()
    ranked["_v09_date"] = _normalise_dates(
        ranked["date"], "v0.9 source picks"
    )
    ranked["candidate"] = ranked["candidate"].astype(str)
    ranked["code"] = ranked["code"].astype(str)
    ranked["model_rank"] = pd.to_numeric(
        ranked["model_rank"], errors="coerce"
    )
    if ranked["model_rank"].isna().any():
        raise DataValidationError("v0.9 source picks contain an invalid rank")
    if ranked.duplicated(["_v09_date", "candidate", "model_rank"]).any():
        raise DataValidationError(
            "v0.9 source picks contain duplicate date/candidate/rank"
        )

    state = context[
        [
            "date",
            PRIOR_MARKET_OC_BREADTH,
            PRIOR_MARKET_OC_BREADTH_SOURCE_DATE,
        ]
    ].copy()
    state["_v09_date"] = _normalise_dates(
        state["date"], "v0.9 breadth context"
    )
    if state.duplicated("_v09_date").any():
        raise DataValidationError(
            "v0.9 breadth context must contain one row per date"
        )
    state[PRIOR_MARKET_OC_BREADTH] = pd.to_numeric(
        state[PRIOR_MARKET_OC_BREADTH], errors="coerce"
    )
    finite = state[PRIOR_MARKET_OC_BREADTH].dropna()
    if (
        not np.isfinite(finite.to_numpy(dtype=float)).all()
        or not finite.between(0.0, 1.0, inclusive="both").all()
    ):
        raise DataValidationError("v0.9 breadth context must be in [0, 1]")
    source = pd.to_datetime(
        state[PRIOR_MARKET_OC_BREADTH_SOURCE_DATE],
        errors="coerce",
        format="mixed",
    )
    nonmissing_breadth = state[PRIOR_MARKET_OC_BREADTH].notna()
    if source.loc[nonmissing_breadth].isna().any():
        raise DataValidationError(
            "non-missing v0.9 breadth requires a source date"
        )
    if (
        source.loc[nonmissing_breadth]
        .dt.normalize()
        .ge(state.loc[nonmissing_breadth, "_v09_date"])
        .any()
    ):
        raise LeakageError(
            "v0.9 breadth context contains a non-prior source date"
        )

    source_dates = pd.DatetimeIndex(sorted(ranked["_v09_date"].unique()))
    state_dates = pd.DatetimeIndex(sorted(state["_v09_date"].unique()))
    if not source_dates.equals(state_dates):
        raise DataValidationError(
            "v0.9 picks and breadth context must cover the same dates"
        )
    state["chosen_candidate"] = np.where(
        state[PRIOR_MARKET_OC_BREADTH].notna()
        & state[PRIOR_MARKET_OC_BREADTH].ge(threshold),
        high_breadth_candidate,
        low_breadth_candidate,
    )
    merged = ranked.merge(
        state[
            [
                "_v09_date",
                PRIOR_MARKET_OC_BREADTH,
                PRIOR_MARKET_OC_BREADTH_SOURCE_DATE,
                "chosen_candidate",
            ]
        ],
        on="_v09_date",
        how="inner",
        validate="many_to_one",
    )
    selected = merged[
        merged["candidate"].eq(merged["chosen_candidate"])
        & merged["model_rank"].eq(model_rank)
    ].copy()
    counts = selected.groupby("_v09_date", sort=True).size()
    if len(counts) != len(source_dates) or not counts.eq(1).all():
        raise DataValidationError(
            "v0.9 breadth policy requires exactly one selected row per date"
        )

    selected["shadow_policy_id"] = V09_VALID_OC_BREADTH_RANK2_POLICY
    selected["portfolio_weight"] = 1.0
    selected["display_rank"] = 1
    selected["breadth_threshold"] = float(threshold)
    return (
        selected.sort_values("_v09_date", kind="stable")
        .drop(columns="_v09_date")
        .reset_index(drop=True)
    )
