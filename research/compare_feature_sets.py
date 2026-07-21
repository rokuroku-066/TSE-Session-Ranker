#!/usr/bin/env python3
"""Compare predeclared feature blocks with one fixed regularized logit.

The estimator, hyperparameters, monthly folds, eligibility policy, candidate
universe, ranking, and transaction-cost convention are identical for every
candidate.  Only the columns supplied to the estimator change.

The 2025-04..07 benchmark is already known from earlier research and has an
access budget of zero.  The script discards those rows before constructing any
candidate feature and selects one winner using development periods only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tse_session_ranker.config import RankerConfig
from tse_session_ranker.data.common import (
    normalize_daily_prices,
    normalize_expected_sessions,
    prepare_modeling_prices,
    session_calendar_hash,
)
from tse_session_ranker.data.tdnet import (
    TDNET_FEATURE_COLUMNS,
    TDNET_MODEL_FEATURE_COLUMNS,
    build_tdnet_features,
    collect_tdnet_dataset,
    tdnet_target_completeness,
)
from tse_session_ranker.features import (
    PRICE_FEATURE_COLUMNS,
    _prior_rolling,
    build_feature_panel,
)
from tse_session_ranker.inference import rank_candidates
from tse_session_ranker.io import read_frame, write_frame, write_json
from tse_session_ranker.profit import daily_portfolio_returns, profit_metrics
from tse_session_ranker.training import date_equal_weights


EXPLORATION_START = pd.Timestamp("2024-09-02")
EXPLORATION_END = pd.Timestamp("2024-10-31")
EXPLORATION_TRAIN_START = pd.Timestamp("2024-04-01")
EARLY_DEVELOPMENT_START = pd.Timestamp("2024-06-03")
EARLY_DEVELOPMENT_END = pd.Timestamp("2024-08-30")
EARLY_DEVELOPMENT_TRAIN_START = pd.Timestamp("2024-01-04")
SELECTION_START = pd.Timestamp("2025-01-06")
SELECTION_END = pd.Timestamp("2025-03-31")
BENCHMARK_START = pd.Timestamp("2025-04-01")
BENCHMARK_END = pd.Timestamp("2025-07-31")

COMPACT_BASE: tuple[str, ...] = (
    "oc_last",
    "oc_mean_20",
    "oc_win_20",
    "oc_std_20",
    "overnight_last",
    "overnight_mean_20",
    "night_day_corr_60",
)

RANK_FEATURES: tuple[str, ...] = (
    "xrank_oc_mean_20",
    "xrank_overnight_mean_20",
    "xrank_oc_std_20",
    "xrank_atr14_pct",
)

TREND_SHAPE_FEATURES: tuple[str, ...] = (
    "cc_momentum_5_atr",
    "cc_momentum_20_atr",
    "distance_high_20_atr",
    "close_location_last",
    "log_atr14_pct",
    "overnight_std_20",
    "gap_fill_rate_20",
)

TDNET_INTERACTION_FEATURES: tuple[str, ...] = (
    "tdnet_any_x_xrank_cc_momentum_20_atr",
    "tdnet_positive_x_xrank_cc_momentum_20_atr",
    "tdnet_negative_x_xrank_cc_momentum_20_atr",
    "tdnet_earnings_x_xrank_oc_mean_20",
    "tdnet_revision_x_xrank_cc_momentum_20_atr",
)

MINIMAL_PRICE_SHAPE: tuple[str, ...] = (
    "close_location_last",
    "log_atr14_pct",
    "overnight_std_20",
    "gap_fill_rate_20",
)

MOMENTUM_FEATURES: tuple[str, ...] = (
    "cc_momentum_5_atr",
    "cc_momentum_20_atr",
    "distance_high_20_atr",
)

TDNET_TIMING_FEATURES: tuple[str, ...] = (
    "tdnet_any",
    "tdnet_count_log1p",
    "tdnet_premarket_count_log1p",
    "tdnet_intraday_count_log1p",
    "tdnet_postclose_count_log1p",
    "tdnet_latest_age_hours_log1p",
)

TDNET_CLEAR_FLAGS: tuple[str, ...] = TDNET_MODEL_FEATURE_COLUMNS

TDNET_BUYBACK_DIVIDEND_FLAGS: tuple[str, ...] = (
    "tdnet_has_buyback_decision",
    "tdnet_has_dividend_up",
    "tdnet_has_dividend_down",
)

TDNET_SIGNED_FEATURES: tuple[str, ...] = (
    "tdnet_support_score",
    "tdnet_adverse_score",
    "tdnet_signed_score",
    "tdnet_clear_signal",
)


@dataclass(frozen=True)
class PeriodDefinition:
    start: str
    end: str
    train_start: str


def _ordered_union(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(column for group in groups for column in group))


def feature_sets() -> dict[str, tuple[str, ...]]:
    base = tuple(PRICE_FEATURE_COLUMNS)
    return {
        "baseline_12": base,
        "baseline_price_shape": _ordered_union(base, MINIMAL_PRICE_SHAPE),
        "baseline_momentum": _ordered_union(base, MOMENTUM_FEATURES),
        "baseline_buyback": _ordered_union(
            base, ("tdnet_has_buyback_decision",)
        ),
        "baseline_dividend_sign": _ordered_union(
            base, ("tdnet_has_dividend_up", "tdnet_has_dividend_down")
        ),
        "baseline_buyback_dividend": _ordered_union(
            base, TDNET_BUYBACK_DIVIDEND_FLAGS
        ),
        "baseline_tdnet_clear_flags": _ordered_union(base, TDNET_CLEAR_FLAGS),
        "baseline_tdnet_clear_scores": _ordered_union(base, TDNET_SIGNED_FEATURES),
        "baseline_tdnet_timing": _ordered_union(base, TDNET_TIMING_FEATURES),
        "baseline_tdnet_timing_clear": _ordered_union(
            base, TDNET_TIMING_FEATURES, TDNET_CLEAR_FLAGS
        ),
        "baseline_price_shape_tdnet_clear": _ordered_union(
            base, MINIMAL_PRICE_SHAPE, TDNET_CLEAR_FLAGS
        ),
        "baseline_momentum_tdnet_clear": _ordered_union(
            base, MOMENTUM_FEATURES, TDNET_CLEAR_FLAGS
        ),
    }


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = denominator.replace([np.inf, -np.inf], np.nan).where(denominator > 0)
    return numerator / valid


def add_price_candidate_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add candidate columns using rows strictly before each target date."""

    frame = panel.sort_values(["code", "date"], kind="stable").copy()
    group = frame.groupby("code", sort=False)
    valid_cc = frame["traded"] & frame["prior_close"].notna()
    cc = pd.Series(np.nan, index=frame.index, dtype=float)
    cc.loc[valid_cc] = 100.0 * (
        frame.loc[valid_cc, "close"] / frame.loc[valid_cc, "prior_close"] - 1.0
    )
    # Daily moves beyond 30% are treated as possible corporate actions for the
    # candidate trend block.  The baseline features remain unchanged.
    cc = cc.where(cc.abs().le(30.0))
    frame["_cc_log_pct"] = 100.0 * np.log1p(cc / 100.0)
    cc_log_5 = _prior_rolling(
        frame, "_cc_log_pct", 5, "sum", min_periods=5
    )
    cc_log_20 = _prior_rolling(
        frame, "_cc_log_pct", 20, "sum", min_periods=15
    )
    frame["cc_momentum_5_atr"] = _safe_ratio(cc_log_5, frame["atr14_pct"])
    frame["cc_momentum_20_atr"] = _safe_ratio(cc_log_20, frame["atr14_pct"])

    prior_open = group["open"].shift(1)
    prior_high = group["high"].shift(1)
    prior_low = group["low"].shift(1)
    prior_range = prior_high - prior_low
    frame["close_location_last"] = (
        (frame["prior_close"] - prior_low) / prior_range.where(prior_range > 0)
    ).clip(0.0, 1.0)

    high_20 = _prior_rolling(frame, "high", 20, "max", min_periods=15)
    low_20 = _prior_rolling(frame, "low", 20, "min", min_periods=15)
    plausible_range = _safe_ratio(high_20, low_20).le(2.0)
    distance_high_pct = 100.0 * (frame["prior_close"] / high_20 - 1.0)
    frame["distance_high_20_atr"] = _safe_ratio(
        distance_high_pct.where(plausible_range), frame["atr14_pct"]
    )
    frame["log_atr14_pct"] = np.log1p(frame["atr14_pct"].clip(lower=0.0))

    overnight_clipped = frame["overnight"].where(frame["overnight"].abs().le(30.0))
    frame["_overnight_clipped"] = overnight_clipped
    frame["overnight_std_20"] = _prior_rolling(
        frame, "_overnight_clipped", 20, "std", min_periods=10
    )
    gap_response_known = (
        frame["overnight"].notna()
        & frame["oc_return_pct"].notna()
        & frame["overnight"].abs().ge(0.25)
        & frame["overnight"].abs().le(30.0)
    )
    frame["_gap_filled"] = np.nan
    frame.loc[gap_response_known, "_gap_filled"] = (
        frame.loc[gap_response_known, "overnight"]
        * frame.loc[gap_response_known, "oc_return_pct"]
    ).lt(0.0).astype(float)
    frame["gap_fill_rate_20"] = _prior_rolling(
        frame, "_gap_filled", 20, "mean", min_periods=8
    )

    rank_sources = {
        "oc_mean_20": "oc_mean_20",
        "overnight_mean_20": "overnight_mean_20",
        "oc_std_20": "oc_std_20",
        "atr14_pct": "atr14_pct",
        "cc_momentum_20_atr": "cc_momentum_20_atr",
    }
    # The broader training-eligible universe is defined entirely by prior data
    # and is used for both historical and live cross-sectional transforms.
    rank_universe = frame["training_eligible"].fillna(False)
    for output_name, source_name in rank_sources.items():
        source = frame[source_name].where(rank_universe)
        percentile = source.groupby(frame["date"], sort=False).rank(
            method="average", pct=True
        )
        frame[f"xrank_{output_name}"] = 2.0 * percentile - 1.0

    drop = [
        "_cc_log_pct",
        "_overnight_clipped",
        "_gap_filled",
    ]
    frame = frame.drop(columns=drop)
    candidate_columns = set(RANK_FEATURES) | set(TREND_SHAPE_FEATURES) | {
        "xrank_cc_momentum_20_atr"
    }
    frame[list(candidate_columns)] = frame[list(candidate_columns)].astype("float32")
    return frame


def attach_tdnet_candidate_features(
    panel: pd.DataFrame,
    disclosures: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    decision_time: str,
) -> pd.DataFrame:
    tdnet = build_tdnet_features(
        disclosures, sessions, decision_time=decision_time
    )
    frame = panel.merge(
        tdnet, on=["date", "code"], how="left", validate="many_to_one", sort=False
    )
    frame[list(TDNET_FEATURE_COLUMNS)] = frame[
        list(TDNET_FEATURE_COLUMNS)
    ].fillna(0.0).astype("float32")
    supportive = frame[
        [
            "tdnet_has_revision_up",
            "tdnet_has_dividend_up",
            "tdnet_has_buyback_decision",
        ]
    ].sum(axis=1)
    adverse = frame[
        [
            "tdnet_has_revision_down",
            "tdnet_has_dividend_down",
            "tdnet_has_equity_financing",
        ]
    ].sum(axis=1)
    frame["tdnet_support_score"] = np.log1p(supportive)
    frame["tdnet_adverse_score"] = np.log1p(adverse)
    frame["tdnet_signed_score"] = supportive - adverse
    frame["tdnet_clear_signal"] = (supportive.add(adverse).gt(0)).astype(float)
    frame["tdnet_any_x_xrank_cc_momentum_20_atr"] = (
        frame["tdnet_any"] * frame["xrank_cc_momentum_20_atr"]
    )
    frame["tdnet_positive_x_xrank_cc_momentum_20_atr"] = (
        frame["tdnet_positive"] * frame["xrank_cc_momentum_20_atr"]
    )
    frame["tdnet_negative_x_xrank_cc_momentum_20_atr"] = (
        frame["tdnet_negative"] * frame["xrank_cc_momentum_20_atr"]
    )
    frame["tdnet_earnings_x_xrank_oc_mean_20"] = (
        frame["tdnet_has_earnings"] * frame["xrank_oc_mean_20"]
    )
    frame["tdnet_revision_x_xrank_cc_momentum_20_atr"] = (
        frame["tdnet_has_revision"] * frame["xrank_cc_momentum_20_atr"]
    )
    frame[list(TDNET_INTERACTION_FEATURES)] = frame[
        list(TDNET_INTERACTION_FEATURES)
    ].astype("float32")
    frame[list(TDNET_SIGNED_FEATURES)] = frame[
        list(TDNET_SIGNED_FEATURES)
    ].astype("float32")
    return frame


def fixed_estimator(config: RankerConfig) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=config.model.c,
                    class_weight=config.model.class_weight,
                    max_iter=config.model.max_iter,
                    random_state=config.model.random_state,
                ),
            ),
        ]
    )


def _daily_picks(scored: pd.DataFrame, top_k: int) -> pd.DataFrame:
    pieces = [
        rank_candidates(group, top_k)
        for _, group in scored.groupby("date", sort=True)
    ]
    return pd.concat(pieces, ignore_index=True) if pieces else scored.iloc[:0].copy()


def evaluate_feature_sets(
    panel: pd.DataFrame,
    specifications: dict[str, tuple[str, ...]],
    names: list[str],
    *,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    train_start: pd.Timestamp,
    config: RankerConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    scored_by_set: dict[str, list[pd.DataFrame]] = {name: [] for name in names}
    folds: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for period in pd.period_range(
        evaluation_start.to_period("M"), evaluation_end.to_period("M"), freq="M"
    ):
        score_start = max(evaluation_start, period.start_time.normalize())
        score_end = min(evaluation_end, period.end_time.normalize())
        training = panel[
            panel["date"].between(train_start, score_start - pd.Timedelta(days=1))
            & panel["training_eligible"]
            & panel["label"].notna()
        ].sort_values(["date", "code"], kind="stable")
        scoring = panel[
            panel["date"].between(score_start, score_end) & panel["eligible"]
        ].sort_values(["date", "code"], kind="stable")
        if training.empty or scoring.empty:
            raise ValueError(
                f"empty train/score fold for {period}: "
                f"train={len(training)}, score={len(scoring)}"
            )
        if not training["date"].max() < scoring["date"].min():
            raise AssertionError(f"overlapping train/score dates for {period}")
        if (scoring["feature_source_max_date"] >= scoring["date"]).fillna(False).any():
            raise AssertionError(f"non-prior price source detected for {period}")
        for name in names:
            columns = list(specifications[name])
            estimator = fixed_estimator(config)
            estimator.fit(
                training[columns],
                training["label"].astype(int),
                model__sample_weight=date_equal_weights(training),
            )
            scored = scoring.copy()
            scored["model_score"] = estimator.predict_proba(scoring[columns])[:, 1]
            scored["feature_set"] = name
            scored_by_set[name].append(scored)
            folds[name].append(
                {
                    "period": str(period),
                    "train_start": str(training["date"].min().date()),
                    "train_end": str(training["date"].max().date()),
                    "score_start": str(scoring["date"].min().date()),
                    "score_end": str(scoring["date"].max().date()),
                    "train_rows": int(len(training)),
                    "score_rows": int(len(scoring)),
                }
            )

    results: dict[str, Any] = {}
    outputs: list[pd.DataFrame] = []
    for name in names:
        scored = pd.concat(scored_by_set[name], ignore_index=True)
        top1 = _daily_picks(scored, 1)
        top2 = _daily_picks(scored, 2)
        results[name] = {
            "feature_count": len(specifications[name]),
            "features": list(specifications[name]),
            "top1": profit_metrics(top1, top_k=1, cost_bps=config.cost_bps),
            "top2": profit_metrics(top2, top_k=2, cost_bps=config.cost_bps),
            "folds": folds[name],
        }
        for top_k, picks in ((1, top1), (2, top2)):
            keep = [
                column
                for column in (
                    "date",
                    "code",
                    "name",
                    "model_rank",
                    "model_score",
                    "label",
                    "oc_return_pct",
                    "outcome_observed",
                    "source_complete",
                    "universe_source_complete",
                    "universe_source_date",
                    "tdnet_any",
                    "tdnet_positive",
                    "tdnet_negative",
                )
                if column in picks
            ]
            output = picks[keep].copy()
            output.insert(0, "portfolio_top_k", top_k)
            output.insert(0, "feature_set", name)
            outputs.append(output)
    return results, pd.concat(outputs, ignore_index=True)


def combine_development_metrics(
    pick_frames: list[pd.DataFrame],
    specifications: dict[str, tuple[str, ...]],
    names: list[str],
    *,
    cost_bps: float,
) -> dict[str, Any]:
    picks = pd.concat(pick_frames, ignore_index=True)
    if picks.duplicated(["feature_set", "portfolio_top_k", "date", "model_rank"]).any():
        raise AssertionError("development periods overlap")
    results: dict[str, Any] = {}
    for name in names:
        top1 = picks[
            picks["feature_set"].eq(name) & picks["portfolio_top_k"].eq(1)
        ]
        top2 = picks[
            picks["feature_set"].eq(name) & picks["portfolio_top_k"].eq(2)
        ]
        results[name] = {
            "feature_count": len(specifications[name]),
            "features": list(specifications[name]),
            "top1": profit_metrics(top1, top_k=1, cost_bps=cost_bps),
            "top2": profit_metrics(top2, top_k=2, cost_bps=cost_bps),
        }
    return results


def paired_top1_comparison(
    pick_frames: list[pd.DataFrame],
    *,
    baseline: str,
    candidate: str,
    cost_bps: float,
    random_state: int,
    bootstrap_samples: int = 20_000,
) -> dict[str, Any]:
    """Compare candidate and baseline on identical scheduled days."""

    picks = pd.concat(pick_frames, ignore_index=True)

    def daily(name: str) -> pd.DataFrame:
        selected = picks[
            picks["feature_set"].eq(name)
            & picks["portfolio_top_k"].eq(1)
        ].copy()
        returns = daily_portfolio_returns(
            selected, top_k=1, cost_bps=cost_bps
        )[["date", "net_return_pct"]]
        codes = selected[["date", "code"]].drop_duplicates("date")
        return returns.merge(codes, on="date", validate="one_to_one")

    left = daily(baseline).rename(
        columns={"net_return_pct": "baseline_net_pct", "code": "baseline_code"}
    )
    right = daily(candidate).rename(
        columns={"net_return_pct": "candidate_net_pct", "code": "candidate_code"}
    )
    paired = left.merge(right, on="date", validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise AssertionError("feature sets do not cover identical scheduled days")
    delta = paired["candidate_net_pct"] - paired["baseline_net_pct"]
    rng = np.random.default_rng(random_state)
    sample_indices = rng.integers(
        0, len(delta), size=(bootstrap_samples, len(delta))
    )
    bootstrap_means = delta.to_numpy()[sample_indices].mean(axis=1)
    changed = paired["candidate_code"].ne(paired["baseline_code"])
    return {
        "baseline": baseline,
        "candidate": candidate,
        "days": int(len(paired)),
        "ranking_changed_days": int(changed.sum()),
        "ranking_unchanged_days": int((~changed).sum()),
        "mean_delta_pct_points_per_day": float(delta.mean()),
        "positive_delta_days": int(delta.gt(0).sum()),
        "negative_delta_days": int(delta.lt(0).sum()),
        "zero_delta_days": int(delta.eq(0).sum()),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": random_state,
        "bootstrap_90pct_ci": [
            float(np.quantile(bootstrap_means, 0.05)),
            float(np.quantile(bootstrap_means, 0.95)),
        ],
        "interpretation": (
            "diagnostic only; the feature sets and winner were selected on these "
            "development periods, so this is not a holdout confidence interval"
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_hash(paths: list[Path]) -> tuple[str, int]:
    digest = hashlib.sha256()
    for file in sorted(paths):
        digest.update(file.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(file)))
    return digest.hexdigest(), len(paths)


def _tdnet_provenance(paths: list[Path]) -> dict[str, Any]:
    """Summarize cache observation provenance without hiding legacy inputs."""

    counts: dict[str, int] = {}
    legacy_dates: list[str] = []
    for path in sorted(paths):
        metadata_path = path.with_suffix(path.suffix + ".meta.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        provenance = str(
            metadata.get("provenance", "schema_v1_untagged_request_start")
        )
        counts[provenance] = counts.get(provenance, 0) + 1
        if provenance == "legacy_historical_file_mtime_assumption":
            legacy_dates.append(path.stem)
    return {
        "counts": dict(sorted(counts.items())),
        "legacy_historical_file_mtime_dates": legacy_dates,
        "legacy_limit": (
            "Historical research only: these pages predate recorded HTTP request "
            "starts, so file mtime is used only to prove they were observed long "
            "after the indexed date. Live inference rejects missing sidecars."
            if legacy_dates
            else None
        ),
    }


def _content_hash(frame: pd.DataFrame) -> str:
    columns = ["date", "code", "open", "high", "low", "close", "traded"]
    hashed = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _implementation_hashes(repository_root: Path) -> dict[str, str]:
    relative_paths = (
        "research/compare_feature_sets.py",
        "src/tse_session_ranker/config.py",
        "src/tse_session_ranker/data/common.py",
        "src/tse_session_ranker/data/tdnet.py",
        "src/tse_session_ranker/features.py",
        "src/tse_session_ranker/inference.py",
        "src/tse_session_ranker/profit.py",
        "src/tse_session_ranker/training.py",
    )
    return {
        relative: _sha256_file(repository_root / relative)
        for relative in relative_paths
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fixed-logit feature blocks on the production panel"
    )
    parser.add_argument("--daily", required=True)
    parser.add_argument(
        "--calendar",
        required=True,
        help="explicit JPX exchange-session calendar (CSV date column or one date/line)",
    )
    parser.add_argument("--tdnet-cache", required=True)
    parser.add_argument(
        "--output", default="research/feature_set_comparison.json"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    daily_path = Path(args.daily).resolve()
    calendar_path = Path(args.calendar).resolve()
    tdnet_path = Path(args.tdnet_cache).resolve()
    output_path = Path(args.output).resolve()
    repository_root = Path(__file__).resolve().parents[1]
    config = RankerConfig()
    canonical = normalize_daily_prices(read_frame(daily_path))
    # All dates after the selection cutoff are discarded before any feature
    # candidate is built.  The known 2025-04..07 benchmark is access-budget 0.
    canonical = canonical[canonical["date"].le(SELECTION_END)].copy()
    if calendar_path.suffix.lower() == ".csv":
        calendar_frame = pd.read_csv(calendar_path)
        if "date" not in calendar_frame:
            raise ValueError("calendar CSV requires a date column")
        calendar_values = calendar_frame["date"]
    else:
        calendar_values = [
            line.strip()
            for line in calendar_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    sessions = normalize_expected_sessions(
        calendar_values, through=SELECTION_END
    )
    sessions = sessions[sessions >= canonical["date"].min()]
    unexpected_price_dates = pd.DatetimeIndex(
        canonical["date"].drop_duplicates()
    ).difference(sessions)
    if len(unexpected_price_dates):
        examples = ", ".join(str(value.date()) for value in unexpected_price_dates[:5])
        raise ValueError(f"daily data contains dates outside the calendar: {examples}")
    required_calendar_dates = pd.date_range(
        canonical["date"].min(), SELECTION_END, freq="D"
    )
    all_tdnet_files = sorted(tdnet_path.glob("*.html"))
    parsed_file_dates = pd.to_datetime(
        [path.stem for path in all_tdnet_files], format="%Y%m%d", errors="coerce"
    )
    tdnet_files_for_period = [
        path
        for path, date in zip(all_tdnet_files, parsed_file_dates, strict=True)
        if pd.notna(date) and canonical["date"].min() <= date <= SELECTION_END
    ]
    cached_dates = pd.DatetimeIndex(
        [pd.Timestamp(path.stem) for path in tdnet_files_for_period]
    )
    missing_cache = required_calendar_dates.difference(cached_dates)
    if len(missing_cache):
        examples = ", ".join(str(value.date()) for value in missing_cache[:5])
        raise ValueError(f"TDnet cache is incomplete; missing {examples}")

    modeling, coverage = prepare_modeling_prices(
        canonical,
        coverage_lookback=config.source_coverage_lookback,
        minimum_source_coverage=config.minimum_source_coverage,
        expected_sessions=sessions,
    )
    panel = add_price_candidate_features(build_feature_panel(modeling, config))
    tdnet_dataset = collect_tdnet_dataset(
        tdnet_files_for_period,
        allow_historical_provenance=True,
    )
    tdnet_coverage = tdnet_target_completeness(
        sessions,
        tdnet_dataset.complete_dates,
        tdnet_dataset.observed_at_by_date,
        decision_time=config.preopen.decision_time,
    )
    incomplete_tdnet_sessions = sessions[1:][
        ~tdnet_coverage.reindex(sessions[1:], fill_value=False).to_numpy()
    ]
    if len(incomplete_tdnet_sessions):
        examples = ", ".join(
            str(value.date()) for value in incomplete_tdnet_sessions[:5]
        )
        raise ValueError(f"TDnet cutoff coverage is incomplete: {examples}")
    disclosures = tdnet_dataset.disclosures
    panel = attach_tdnet_candidate_features(
        panel,
        disclosures,
        sessions,
        decision_time=config.preopen.decision_time,
    )
    specifications = feature_sets()
    all_names = list(specifications)
    early_development, early_development_picks = evaluate_feature_sets(
        panel,
        specifications,
        all_names,
        evaluation_start=EARLY_DEVELOPMENT_START,
        evaluation_end=EARLY_DEVELOPMENT_END,
        train_start=EARLY_DEVELOPMENT_TRAIN_START,
        config=config,
    )
    exploration, exploration_picks = evaluate_feature_sets(
        panel,
        specifications,
        all_names,
        evaluation_start=EXPLORATION_START,
        evaluation_end=EXPLORATION_END,
        train_start=EXPLORATION_TRAIN_START,
        config=config,
    )
    selection, selection_picks = evaluate_feature_sets(
        panel,
        specifications,
        all_names,
        evaluation_start=SELECTION_START,
        evaluation_end=SELECTION_END,
        train_start=pd.Timestamp(config.regime_start),
        config=config,
    )
    combined_development = combine_development_metrics(
        [early_development_picks, exploration_picks, selection_picks],
        specifications,
        all_names,
        cost_bps=config.cost_bps,
    )
    winner = sorted(
        all_names,
        key=lambda name: (
            -combined_development[name]["top1"]["net_mean_pct_at_cost"],
            -combined_development[name]["top2"]["net_mean_pct_at_cost"],
            len(specifications[name]),
            name,
        ),
    )[0]
    paired_winner_vs_baseline = paired_top1_comparison(
        [early_development_picks, exploration_picks, selection_picks],
        baseline="baseline_12",
        candidate=winner,
        cost_bps=config.cost_bps,
        random_state=config.model.random_state,
    )

    tdnet_hash, tdnet_files = _directory_hash(tdnet_files_for_period)
    tdnet_metadata_hash, tdnet_metadata_files = _directory_hash(
        [path.with_suffix(path.suffix + ".meta.json") for path in tdnet_files_for_period]
    )
    tdnet_provenance = _tdnet_provenance(tdnet_files_for_period)
    incomplete = coverage.loc[~coverage["source_complete"], "date"]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "objective": "top1 scheduled-day mean net open_to_close return after cost",
        "fixed_model": {
            "algorithm": "LogisticRegression",
            "C": config.model.c,
            "class_weight": config.model.class_weight,
            "random_state": config.model.random_state,
            "date_weight": "1 / executed training rows on date",
            "imputer": "median_with_missing_indicators",
            "scaler": "standard",
        },
        "selection_rule": (
            "maximum combined-development top1 net mean; then top2 net mean; "
            "then fewer features; then feature-set id. Benchmark access zero."
        ),
        "candidate_policy": (
            "12 predeclared economic feature blocks; no per-column greedy search"
        ),
        "benchmark_is_true_holdout": False,
        "benchmark_status": "BURNED; reporting only; access budget zero",
        "leakage_guards": {
            "price_features": "strictly prior rows",
            "cross_section": "prior-data training-eligible universe only",
            "tdnet_mapping": (
                "first exchange session whose 08:58:59 JST cutoff is not "
                "before the public mirror's listed published_at"
            ),
            "tdnet_source_limit": (
                "third-party public date index; listing delay or later correction "
                "cannot be ruled out"
            ),
            "tdnet_cache_completeness": "every calendar date in data range required",
            "unfilled": "rank first, then zero return and zero cost",
            "training_cutoff": "strictly before each scoring month",
        },
        "cost_bps": config.cost_bps,
        "periods": {
            "early_development": asdict(
                PeriodDefinition(
                    str(EARLY_DEVELOPMENT_START.date()),
                    str(EARLY_DEVELOPMENT_END.date()),
                    str(EARLY_DEVELOPMENT_TRAIN_START.date()),
                )
            ),
            "exploration": asdict(
                PeriodDefinition(
                    str(EXPLORATION_START.date()),
                    str(EXPLORATION_END.date()),
                    str(EXPLORATION_TRAIN_START.date()),
                )
            ),
            "selection": asdict(
                PeriodDefinition(
                    str(SELECTION_START.date()),
                    str(SELECTION_END.date()),
                    config.regime_start,
                )
            ),
            "known_benchmark": asdict(
                PeriodDefinition(
                    str(BENCHMARK_START.date()),
                    str(BENCHMARK_END.date()),
                    config.regime_start,
                )
            ),
        },
        "feature_sets": {name: list(columns) for name, columns in specifications.items()},
        "early_development": early_development,
        "exploration": exploration,
        "selection": selection,
        "combined_development": combined_development,
        "selected_winner": winner,
        "paired_winner_vs_baseline": paired_winner_vs_baseline,
        "known_benchmark": "not accessed by this feature-selection round",
        "rejected_round1": {
            "feature_set": "compact_trend_shape_tdnet",
            "selection_top1_net_mean_pct": 0.42735994728749466,
            "known_benchmark_top1_net_mean_pct": -0.0216173639,
            "decision": "rejected as period-specific",
        },
        "data": {
            "daily_path": str(daily_path),
            "daily_file_sha256": _sha256_file(daily_path),
            "daily_content_sha256": _content_hash(canonical),
            "rows": int(len(canonical)),
            "codes": int(canonical["code"].nunique()),
            "sessions": int(canonical["date"].nunique()),
            "calendar_sha256": session_calendar_hash(
                sessions, through=SELECTION_END
            ),
            "calendar_path": str(calendar_path),
            "calendar_file_sha256": _sha256_file(calendar_path),
            "calendar_mode": "explicit_exchange_sessions",
            "calendar_sessions_without_price_rows": [
                str(value.date())
                for value in sessions.difference(
                    pd.DatetimeIndex(canonical["date"].drop_duplicates())
                )
            ],
            "source_incomplete_dates": [
                str(pd.Timestamp(value).date()) for value in incomplete
            ],
            "tdnet_cache_path": str(tdnet_path),
            "tdnet_cache_sha256": tdnet_hash,
            "tdnet_cache_metadata_sha256": tdnet_metadata_hash,
            "tdnet_dataset_source_sha256": tdnet_dataset.source_sha256,
            "tdnet_cache_files": tdnet_files,
            "tdnet_cache_metadata_files": tdnet_metadata_files,
            "tdnet_cache_observation_provenance": tdnet_provenance,
            "tdnet_disclosures": int(len(disclosures)),
            "tdnet_codes": int(disclosures["code"].nunique()),
            "tdnet_min_published_at": disclosures["published_at"].min(),
            "tdnet_max_published_at": disclosures["published_at"].max(),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "implementation_sha256": _implementation_hashes(repository_root),
    }
    write_json(payload, output_path)
    stem = output_path.with_suffix("")
    write_frame(
        early_development_picks,
        stem.with_name(stem.name + "_early_development_picks.csv"),
    )
    write_frame(
        exploration_picks,
        stem.with_name(stem.name + "_exploration_picks.csv"),
    )
    write_frame(
        selection_picks,
        stem.with_name(stem.name + "_selection_picks.csv"),
    )
    table = pd.DataFrame(
        [
            {
                "feature_set": name,
                "features": len(specifications[name]),
                "early_top1": early_development[name]["top1"][
                    "net_mean_pct_at_cost"
                ],
                "exploration_top1": exploration[name]["top1"][
                    "net_mean_pct_at_cost"
                ],
                "selection_top1": selection[name]["top1"][
                    "net_mean_pct_at_cost"
                ],
                "combined_top1": combined_development[name]["top1"][
                    "net_mean_pct_at_cost"
                ],
                "combined_top2": combined_development[name]["top2"][
                    "net_mean_pct_at_cost"
                ],
            }
            for name in all_names
        ]
    ).sort_values(["combined_top1", "combined_top2"], ascending=False)
    print(table.to_string(index=False))
    print(f"winner={winner}")
    print("known_benchmark=not_accessed")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
