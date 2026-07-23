#!/usr/bin/env python3
"""PIT walk-forward validation of TDnet title and event-structure signals.

This is an exploratory runner bound to ``protocol.json``.  It deliberately
keeps missing TDnet source days distinct from observed no-event days.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/workspace/scratch/8678b1d14f37")
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tse_session_ranker.data.tdnet import (  # noqa: E402
    _title_flags,
    parse_tdnet_index,
)
from tse_session_ranker.research_models import (  # noqa: E402
    ResearchModelSpec,
    fit_research_model,
)


HERE = Path("/tmp/v10_tdnet_text")
PROTOCOL_PATH = HERE / "protocol.json"
PANEL_PATH = Path("/tmp/model_v07_corrected_panel.pkl")
CACHE_PATHS = [ROOT / "tdnet_date_cache_2024", ROOT / "tdnet_date_cache"]
SCORE_START = pd.Timestamp("2024-07-01")
SCORE_END = pd.Timestamp("2025-07-31")
COSTS = (0.0, 20.0, 40.0, 60.0)

BASE_FEATURES = (
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
L4_FEATURES = (*BASE_FEATURES, "flat_oc_rate_20")
NOVELTY_FEATURES = (
    "novelty_global",
    "novelty_same_code",
    "title_mean_idf",
)
INTENSITY_FEATURES = (
    "document_count_log1p",
    "bundle_width_minutes_log1p",
    "latest_age_hours_log1p",
    "market_bundle_count_log1p",
    "market_document_count_log1p",
    "prior_code_event_count_60_complete_log1p",
)
SIGNAL_FEATURES = (
    "positive_family_count",
    "negative_family_count",
    "positive_corroboration",
    "negative_corroboration",
    "mixed_sign",
    "unclear_direction",
)
STRUCTURED_FEATURES = (*NOVELTY_FEATURES, *INTENSITY_FEATURES, *SIGNAL_FEATURES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    files = sorted(path.glob("*.html"))
    for item in files:
        digest.update(item.name.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
        digest.update(b"\0")
    return len(files), digest.hexdigest()


def load_protocol() -> dict[str, Any]:
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if value["protocol_id"] != "v10_tdnet_text_zero_base_20260723":
        raise RuntimeError("unexpected protocol")
    if value["authority"]["production_promotion_allowed"] is not False:
        raise RuntimeError("exploratory protocol must not permit promotion")
    return value


def validate_inputs(protocol: dict[str, Any]) -> dict[str, Any]:
    expected_panel = protocol["frozen_inputs"]["panel"]
    actual_panel = sha256_file(PANEL_PATH)
    actual_manifest = sha256_file(Path(str(PANEL_PATH) + ".manifest.json"))
    if actual_panel != expected_panel["sha256"]:
        raise RuntimeError("panel SHA-256 changed")
    if actual_manifest != expected_panel["manifest_sha256"]:
        raise RuntimeError("panel manifest SHA-256 changed")
    cache_checks = []
    for path, expected in zip(
        CACHE_PATHS, protocol["frozen_inputs"]["tdnet_caches"], strict=True
    ):
        count, digest = cache_digest(path)
        if count != expected["html_files"] or digest != expected["aggregate_sha256"]:
            raise RuntimeError(f"cache changed: {path}")
        cache_checks.append(
            {"path": str(path), "html_files": count, "aggregate_sha256": digest}
        )
    return {
        "panel_sha256": actual_panel,
        "panel_manifest_sha256": actual_manifest,
        "cache_checks": cache_checks,
    }


def parse_caches() -> tuple[pd.DataFrame, set[pd.Timestamp], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    source_dates: set[pd.Timestamp] = set()
    parse_counts: dict[str, int] = {}
    for directory in CACHE_PATHS:
        for path in sorted(directory.glob("*.html")):
            match = re.search(r"(20\d{6})", path.stem)
            if match is None:
                raise RuntimeError(f"cache filename has no date: {path}")
            date = pd.to_datetime(match.group(1), format="%Y%m%d").normalize()
            if date in source_dates:
                raise RuntimeError(f"duplicate cache date: {date.date()}")
            frame = parse_tdnet_index(path.read_bytes(), date)
            frames.append(frame)
            source_dates.add(date)
            parse_counts[str(date.date())] = int(len(frame))
    disclosures = pd.concat(frames, ignore_index=True).sort_values(
        ["published_at", "code", "title"], kind="stable"
    )
    return disclosures, source_dates, {
        "pages": len(source_dates),
        "disclosures": int(len(disclosures)),
        "empty_pages": int(sum(value == 0 for value in parse_counts.values())),
        "first_page": str(min(source_dates).date()),
        "last_page": str(max(source_dates).date()),
    }


def strict_complete_sessions(
    sessions: pd.DatetimeIndex, source_dates: set[pd.Timestamp]
) -> tuple[pd.Series, dict[str, list[str]]]:
    complete: dict[pd.Timestamp, bool] = {}
    missing: dict[str, list[str]] = {}
    for position, date in enumerate(sessions):
        if position == 0:
            complete[date] = False
            missing[str(date.date())] = ["prior panel session unavailable"]
            continue
        prior = sessions[position - 1]
        required = pd.date_range(prior, date, freq="D")
        absent = [str(item.date()) for item in required if item not in source_dates]
        complete[date] = not absent
        if absent:
            missing[str(date.date())] = absent
    return pd.Series(complete, name="strict_tdnet_source_complete"), missing


def map_disclosures(
    disclosures: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    strict_complete: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cutoffs = pd.DatetimeIndex(
        [
            pd.Timestamp.combine(
                date.date(), pd.Timestamp("08:58:59").time()
            ).tz_localize("Asia/Tokyo")
            for date in sessions
        ]
    )
    published = pd.DatetimeIndex(disclosures["published_at"])
    target_positions = cutoffs.searchsorted(published, side="left")
    in_range = target_positions < len(cutoffs)
    events = disclosures.loc[in_range].copy()
    target_positions = target_positions[in_range]
    events["date"] = sessions[target_positions].to_numpy()
    same_day = events["index_date"].eq(events["date"])
    index_is_session = events["index_date"].isin(sessions)
    minute = events["published_at"].dt.hour * 60 + events["published_at"].dt.minute
    close_minute = np.where(
        events["index_date"].lt(pd.Timestamp("2024-11-05")), 15 * 60, 15 * 60 + 30
    )
    intraday_prior_session = (
        index_is_session & ~same_day & minute.lt(close_minute)
    )
    events["qualifying_overnight"] = ~intraday_prior_session
    events["target_source_complete"] = events["date"].map(strict_complete).fillna(False)
    events["target_cutoff"] = pd.DatetimeIndex(cutoffs[target_positions])
    events["age_hours"] = (
        events["target_cutoff"] - events["published_at"]
    ).dt.total_seconds().div(3600.0)
    if events["age_hours"].lt(0).any():
        raise RuntimeError("event mapped before publication")
    qualifying = events.loc[
        events["qualifying_overnight"] & events["target_source_complete"]
    ].copy()
    return qualifying, {
        "mapped_disclosures": int(len(events)),
        "excluded_intraday_prior_session": int(intraday_prior_session.sum()),
        "excluded_target_source_missing": int(
            (events["qualifying_overnight"] & ~events["target_source_complete"]).sum()
        ),
        "qualifying_disclosures": int(len(qualifying)),
        "future_publication_violations": int(events["age_hours"].lt(0).sum()),
    }


def build_bundles(
    events: pd.DataFrame,
    complete_sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    flags = _title_flags(events["title"])
    events = pd.concat([events.reset_index(drop=True), flags.reset_index(drop=True)], axis=1)
    keys = ["date", "code"]
    grouped = events.groupby(keys, sort=True)
    bundles = grouped.agg(
        name=("name", "last"),
        bundle_text=("title", lambda values: " [SEP] ".join(values.astype(str))),
        document_count=("title", "size"),
        first_published=("published_at", "min"),
        last_published=("published_at", "max"),
        latest_age_hours=("age_hours", "min"),
    ).reset_index()
    bundles["bundle_width_minutes"] = (
        bundles["last_published"] - bundles["first_published"]
    ).dt.total_seconds().div(60.0)
    bundles["document_count_log1p"] = np.log1p(bundles["document_count"])
    bundles["bundle_width_minutes_log1p"] = np.log1p(bundles["bundle_width_minutes"])
    bundles["latest_age_hours_log1p"] = np.log1p(bundles["latest_age_hours"])

    family_names = (
        "revision",
        "revision_up",
        "revision_down",
        "dividend",
        "dividend_up",
        "dividend_down",
        "buyback_decision",
        "benefit",
        "split",
        "equity_financing",
        "equity_financing_status",
        "impairment_loss",
        "audit_problem",
    )
    family = grouped[list(family_names)].max().reset_index()
    bundles = bundles.merge(family, on=keys, how="left", validate="one_to_one")
    # Status-only financing titles are not new adverse financing decisions.
    external_financing = (
        bundles["equity_financing"] * (1.0 - bundles["equity_financing_status"])
    )
    positive = pd.DataFrame(
        {
            "revision_up": bundles["revision_up"],
            "dividend_up": bundles["dividend_up"],
            "buyback": bundles["buyback_decision"],
            "benefit": bundles["benefit"],
            "split": bundles["split"],
        }
    )
    negative = pd.DataFrame(
        {
            "revision_down": bundles["revision_down"],
            "dividend_down": bundles["dividend_down"],
            "external_financing": external_financing,
            "impairment": bundles["impairment_loss"],
            "audit": bundles["audit_problem"],
        }
    )
    bundles["positive_family_count"] = positive.sum(axis=1)
    bundles["negative_family_count"] = negative.sum(axis=1)
    bundles["positive_corroboration"] = (
        bundles["positive_family_count"] >= 2
    ).astype(float)
    bundles["negative_corroboration"] = (
        bundles["negative_family_count"] >= 2
    ).astype(float)
    bundles["mixed_sign"] = (
        bundles["positive_family_count"].gt(0)
        & bundles["negative_family_count"].gt(0)
    ).astype(float)
    unknown_revision = (
        bundles["revision"].gt(0)
        & bundles["revision_up"].eq(0)
        & bundles["revision_down"].eq(0)
    )
    unknown_dividend = (
        bundles["dividend"].gt(0)
        & bundles["dividend_up"].eq(0)
        & bundles["dividend_down"].eq(0)
    )
    bundles["unclear_direction"] = (unknown_revision | unknown_dividend).astype(float)

    market = bundles.groupby("date", sort=True).agg(
        market_bundle_count=("code", "size"),
        market_document_count=("document_count", "sum"),
    )
    bundles = bundles.merge(market, on="date", how="left", validate="many_to_one")
    bundles["market_bundle_count_log1p"] = np.log1p(
        bundles["market_bundle_count"]
    )
    bundles["market_document_count_log1p"] = np.log1p(
        bundles["market_document_count"]
    )

    session_position = {date: position for position, date in enumerate(complete_sessions)}
    bundles["_complete_position"] = bundles["date"].map(session_position)
    if bundles["_complete_position"].isna().any():
        raise RuntimeError("bundle exists on non-complete session")
    prior_counts = np.zeros(len(bundles), dtype=float)
    for _, index in bundles.groupby("code", sort=False).groups.items():
        positions = bundles.loc[index, "_complete_position"].to_numpy(dtype=int)
        order = np.argsort(positions, kind="stable")
        sorted_index = np.asarray(index)[order]
        sorted_positions = positions[order]
        left = 0
        for right, (row_index, position) in enumerate(
            zip(sorted_index, sorted_positions, strict=True)
        ):
            while left < right and sorted_positions[left] < position - 60:
                left += 1
            prior_counts[row_index] = right - left
    bundles["prior_code_event_count_60_complete_log1p"] = np.log1p(prior_counts)
    return bundles.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def daily_midrank(values: pd.Series, dates: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"value": values.to_numpy(dtype=float), "date": dates.to_numpy()})
    rank = frame.groupby("date", sort=False)["value"].rank(method="average")
    count = frame.groupby("date", sort=False)["value"].transform("count")
    return (rank - 0.5).div(count).astype(float)


def fit_price_scorer(training: pd.DataFrame, columns: Sequence[str], name: str):
    return fit_research_model(
        ResearchModelSpec(
            name=name,
            family="ridge_daily_rank",
            objective="same_day_return_percentile",
            parameters={"alpha": 1.0},
        ),
        training,
        tuple(columns),
    )


def chronological_novelty(
    matrix: sparse.csr_matrix,
    dates: np.ndarray,
    codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    global_novelty = np.ones(matrix.shape[0], dtype=float)
    code_novelty = np.ones(matrix.shape[0], dtype=float)
    code_history: dict[str, list[int]] = {}
    unique_dates = pd.Index(dates).unique()
    prior_end = 0
    for date in unique_dates:
        rows = np.flatnonzero(dates == date)
        if prior_end:
            similarities = matrix[rows] @ matrix[:prior_end].T
            maximum = np.asarray(similarities.max(axis=1).toarray()).ravel()
            global_novelty[rows] = 1.0 - maximum
        for row in rows:
            prior = code_history.get(str(codes[row]), [])
            if prior:
                similarities = matrix[row] @ matrix[prior].T
                code_novelty[row] = 1.0 - float(similarities.max())
        for row in rows:
            code_history.setdefault(str(codes[row]), []).append(int(row))
        prior_end = int(rows[-1]) + 1
    return np.clip(global_novelty, 0, 1), np.clip(code_novelty, 0, 1)


def scoring_novelty(
    training_matrix: sparse.csr_matrix,
    training_codes: np.ndarray,
    scoring_matrix: sparse.csr_matrix,
    scoring_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not training_matrix.shape[0]:
        return np.ones(scoring_matrix.shape[0]), np.ones(scoring_matrix.shape[0])
    similarities = scoring_matrix @ training_matrix.T
    global_max = np.asarray(similarities.max(axis=1).toarray()).ravel()
    global_novelty = 1.0 - global_max
    by_code: dict[str, np.ndarray] = {}
    for code in np.unique(training_codes):
        by_code[str(code)] = np.flatnonzero(training_codes == code)
    code_novelty = np.ones(scoring_matrix.shape[0], dtype=float)
    for code in np.unique(scoring_codes):
        score_rows = np.flatnonzero(scoring_codes == code)
        train_rows = by_code.get(str(code))
        if train_rows is None:
            continue
        local = scoring_matrix[score_rows] @ training_matrix[train_rows].T
        code_novelty[score_rows] = 1.0 - np.asarray(
            local.max(axis=1).toarray()
        ).ravel()
    return np.clip(global_novelty, 0, 1), np.clip(code_novelty, 0, 1)


def mean_idf(matrix: sparse.csr_matrix, idf: np.ndarray) -> np.ndarray:
    result = np.zeros(matrix.shape[0], dtype=float)
    for row in range(matrix.shape[0]):
        indices = matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]]
        result[row] = float(idf[indices].mean()) if len(indices) else 0.0
    return result


def add_novelty_features(
    train: pd.DataFrame,
    score: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    x_train: sparse.csr_matrix,
    x_score: sparse.csr_matrix,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    score = score.copy()
    train_global, train_code = chronological_novelty(
        x_train,
        train["date"].to_numpy(),
        train["code"].astype(str).to_numpy(),
    )
    score_global, score_code = scoring_novelty(
        x_train,
        train["code"].astype(str).to_numpy(),
        x_score,
        score["code"].astype(str).to_numpy(),
    )
    train["novelty_global"] = train_global
    train["novelty_same_code"] = train_code
    train["title_mean_idf"] = mean_idf(x_train, vectorizer.idf_)
    score["novelty_global"] = score_global
    score["novelty_same_code"] = score_code
    score["title_mean_idf"] = mean_idf(x_score, vectorizer.idf_)
    return train, score


def dense_rank_prediction(
    train: pd.DataFrame, score: pd.DataFrame, columns: Sequence[str]
) -> np.ndarray:
    estimator = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    estimator.fit(train[list(columns)], train["rank_target"])
    return np.asarray(estimator.predict(score[list(columns)]), dtype=float)


def make_ranked_frame(
    scoring: pd.DataFrame, score_values: Iterable[float], candidate_id: str
) -> pd.DataFrame:
    keep = scoring[
        ["date", "code", "name", "label", "oc_return_pct"]
    ].copy()
    keep["model_score"] = np.asarray(list(score_values), dtype=float)
    keep["candidate_id"] = candidate_id
    return keep


def add_overlay(
    base: pd.DataFrame,
    event_score: pd.DataFrame,
    candidate_id: str,
) -> pd.DataFrame:
    result = base.copy()
    result["base_pct"] = daily_midrank(result["model_score"], result["date"])
    event = event_score[["date", "code", "model_score"]].copy()
    event["event_pct"] = daily_midrank(event["model_score"], event["date"])
    event = event.rename(columns={"model_score": "event_raw"})
    result = result.merge(
        event, on=["date", "code"], how="left", validate="one_to_one"
    )
    result["model_score"] = result["base_pct"] + np.where(
        result["event_pct"].notna(), 0.5 * (result["event_pct"] - 0.5), 0.0
    )
    result["candidate_id"] = candidate_id
    return result.drop(columns=["base_pct", "event_pct", "event_raw"])


def build_barbell(
    base: pd.DataFrame, event_value: pd.DataFrame, candidate_id: str
) -> pd.DataFrame:
    result = base.copy()
    result["model_score"] = daily_midrank(result["model_score"], result["date"])
    event = event_value.loc[event_value["model_score"].gt(0)].sort_values(
        ["date", "model_score", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    best = event.groupby("date", sort=True).head(1)[["date", "code"]]
    priority = pd.MultiIndex.from_frame(best)
    keys = pd.MultiIndex.from_frame(result[["date", "code"]])
    result.loc[keys.isin(priority), "model_score"] = 2.0
    result["candidate_id"] = candidate_id
    return result


def select_slots(
    scores: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    top_k: int,
) -> pd.DataFrame:
    ranked = scores.sort_values(
        ["date", "model_score", "code"],
        ascending=[True, False, True],
        kind="stable",
    ).groupby("date", sort=True).head(top_k).copy()
    ranked["model_rank"] = ranked.groupby("date", sort=False).cumcount() + 1
    desired = pd.MultiIndex.from_product(
        [sessions, range(1, top_k + 1)], names=["date", "model_rank"]
    ).to_frame(index=False)
    selected = desired.merge(
        ranked[
            [
                "date",
                "model_rank",
                "code",
                "name",
                "model_score",
                "label",
                "oc_return_pct",
                "candidate_id",
            ]
        ],
        on=["date", "model_rank"],
        how="left",
        validate="one_to_one",
    )
    selected["candidate_id"] = selected["candidate_id"].fillna(
        str(scores["candidate_id"].iloc[0])
    )
    return selected


def daily_returns(picks: pd.DataFrame, top_k: int, cost_bps: float) -> pd.Series:
    executed = picks["oc_return_pct"].notna()
    slot = picks["oc_return_pct"].fillna(0.0) - executed.astype(float) * cost_bps / 100.0
    return slot.groupby(picks["date"], sort=True).sum().div(top_k)


def metrics_for_picks(
    picks: pd.DataFrame, top_k: int, slices: dict[str, tuple[pd.Timestamp, pd.Timestamp]]
) -> dict[str, Any]:
    costs: dict[str, Any] = {}
    for cost in COSTS:
        daily = daily_returns(picks, top_k, cost)
        costs[str(int(cost))] = {
            "mean_pct": float(daily.mean()),
            "median_pct": float(daily.median()),
            "win_rate": float(daily.gt(0).mean()),
            "days": int(len(daily)),
        }
    daily20 = daily_returns(picks, top_k, 20.0)
    daily40 = daily_returns(picks, top_k, 40.0)
    monthly = daily40.groupby(daily40.index.to_period("M")).mean()
    slice_metrics = {}
    for name, (start, end) in slices.items():
        values = daily40.loc[(daily40.index >= start) & (daily40.index <= end)]
        slice_metrics[name] = {
            "days": int(len(values)),
            "net40_mean_pct": float(values.mean()) if len(values) else None,
        }
    removed = daily20.drop(daily20.nlargest(min(20, len(daily20))).index)

    executed = picks["oc_return_pct"].notna()
    contribution = (
        picks["oc_return_pct"].fillna(0.0) - executed.astype(float) * 0.20
    ).div(top_k)
    by_code = contribution.groupby(picks["code"], dropna=True).sum().sort_values(
        ascending=False
    )
    top_profit_codes = list(by_code.loc[by_code.gt(0)].head(10).index)
    cash_contribution = contribution.mask(picks["code"].isin(top_profit_codes), 0.0)
    cash_daily = cash_contribution.groupby(picks["date"], sort=True).sum()
    return {
        "costs": costs,
        "monthly_net40_mean_pct": {
            str(period): float(value) for period, value in monthly.items()
        },
        "positive_months": int(monthly.gt(0).sum()),
        "months": int(len(monthly)),
        "slices": slice_metrics,
        "top20_winning_days_removed_net20_mean_pct": (
            float(removed.mean()) if len(removed) else None
        ),
        "top10_profit_codes_cash_net20_mean_pct": float(cash_daily.mean()),
        "top10_profit_codes": [str(value) for value in top_profit_codes],
        "filled_slots": int(executed.sum()),
        "scheduled_slots": int(len(picks)),
        "slot_fill_rate": float(executed.mean()),
        "days_with_at_least_one_filled_slot": int(
            executed.groupby(picks["date"]).any().sum()
        ),
        "unique_codes": int(picks["code"].nunique(dropna=True)),
    }


def main() -> None:
    protocol = load_protocol()
    integrity = validate_inputs(protocol)
    disclosures, source_dates, parse_report = parse_caches()

    projection = list(
        dict.fromkeys(
            [
                "date",
                "code",
                "name",
                "label",
                "oc_return_pct",
                "price_eligible",
                "price_training_eligible",
                *L4_FEATURES,
            ]
        )
    )
    raw_panel = joblib.load(PANEL_PATH, mmap_mode="r")
    panel = raw_panel[projection].copy()
    del raw_panel
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["code"] = panel["code"].astype(str)
    sessions = pd.DatetimeIndex(panel["date"].drop_duplicates().sort_values())
    strict_complete, missing_by_session = strict_complete_sessions(
        sessions, source_dates
    )
    complete_sessions = pd.DatetimeIndex(
        strict_complete.loc[strict_complete].index
    )
    events, mapping_report = map_disclosures(
        disclosures, sessions, strict_complete
    )
    bundles = build_bundles(events, complete_sessions)
    del disclosures, events
    gc.collect()

    score_sessions = complete_sessions[
        (complete_sessions >= SCORE_START) & (complete_sessions <= SCORE_END)
    ]
    score_months = pd.PeriodIndex(score_sessions.to_period("M").unique()).sort_values()
    if len(score_months) != 5:
        raise RuntimeError(f"expected five source-covered score months, got {score_months}")

    score_parts: dict[str, list[pd.DataFrame]] = {
        name: []
        for name in (
            "G0_price_core",
            "L4_price_control",
            "N0_no_event_L4",
            "T01_char_rank_event_only",
            "T02_char_value_event_only",
            "T03_char_rank_overlay_L4",
            "T04_char_value_overlay_L4",
            "T05_novelty_event_only",
            "T06_intensity_congestion_event_only",
            "T07_corroboration_contradiction_event_only",
            "T08_structured_all_event_only",
            "T09_structured_overlay_L4",
            "T10_event_barbell",
        )
    }
    fold_reports: list[dict[str, Any]] = []

    for period in score_months:
        month_dates = score_sessions[score_sessions.to_period("M") == period]
        month_start = period.start_time.normalize()
        panel_train = panel.loc[
            panel["date"].lt(month_start)
            & panel["price_training_eligible"].eq(True)
            & panel["oc_return_pct"].notna()
        ].copy()
        panel_score = panel.loc[
            panel["date"].isin(month_dates)
            & panel["price_eligible"].eq(True)
        ].copy()
        if panel_train.empty or panel_train["date"].max() >= month_start:
            raise RuntimeError(f"non-PIT panel fold {period}")
        g0_model = fit_price_scorer(panel_train, BASE_FEATURES, f"G0_{period}")
        l4_model = fit_price_scorer(panel_train, L4_FEATURES, f"L4_{period}")
        g0 = make_ranked_frame(
            panel_score, g0_model.score(panel_score), "G0_price_core"
        )
        l4 = make_ranked_frame(
            panel_score, l4_model.score(panel_score), "L4_price_control"
        )
        score_parts["G0_price_core"].append(g0)
        score_parts["L4_price_control"].append(l4)

        panel_rank = panel_train[["date", "code", "oc_return_pct"]].copy()
        panel_rank["rank_target"] = (
            panel_rank.groupby("date", sort=False)["oc_return_pct"]
            .rank(method="average", pct=True)
            .mul(2.0)
            .sub(1.0)
        )
        event_train = bundles.loc[bundles["date"].lt(month_start)].merge(
            panel_train[
                ["date", "code", "name", "label", "oc_return_pct"]
            ],
            on=["date", "code"],
            how="inner",
            suffixes=("", "_panel"),
            validate="one_to_one",
        )
        event_train = event_train.merge(
            panel_rank[["date", "code", "rank_target"]],
            on=["date", "code"],
            how="left",
            validate="one_to_one",
        ).sort_values(["date", "code"], kind="stable").reset_index(drop=True)
        event_score = bundles.loc[bundles["date"].isin(month_dates)].merge(
            panel_score[["date", "code", "name", "label", "oc_return_pct"]],
            on=["date", "code"],
            how="inner",
            suffixes=("", "_panel"),
            validate="one_to_one",
        ).sort_values(["date", "code"], kind="stable").reset_index(drop=True)
        if len(event_train) < 100:
            raise RuntimeError(f"too few event train bundles in {period}: {len(event_train)}")
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=3,
            max_features=30_000,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        x_train = vectorizer.fit_transform(event_train["bundle_text"]).tocsr()
        x_score = vectorizer.transform(event_score["bundle_text"]).tocsr()
        rank_model = Ridge(alpha=10.0, solver="lsqr")
        rank_model.fit(x_train, event_train["rank_target"])
        rank_prediction = np.asarray(rank_model.predict(x_score), dtype=float)
        lower = max(float(event_train["oc_return_pct"].quantile(0.01)), -10.0)
        upper = min(float(event_train["oc_return_pct"].quantile(0.99)), 10.0)
        value_target = event_train["oc_return_pct"].clip(lower, upper)
        value_model = Ridge(alpha=20.0, solver="lsqr")
        value_model.fit(x_train, value_target)
        value_prediction = np.asarray(value_model.predict(x_score), dtype=float)

        event_train, event_score = add_novelty_features(
            event_train, event_score, vectorizer, x_train, x_score
        )
        novelty_prediction = dense_rank_prediction(
            event_train, event_score, NOVELTY_FEATURES
        )
        intensity_prediction = dense_rank_prediction(
            event_train, event_score, INTENSITY_FEATURES
        )
        signal_prediction = dense_rank_prediction(
            event_train, event_score, SIGNAL_FEATURES
        )
        structured_prediction = dense_rank_prediction(
            event_train, event_score, STRUCTURED_FEATURES
        )
        event_models = {
            "T01_char_rank_event_only": rank_prediction,
            "T02_char_value_event_only": value_prediction,
            "T05_novelty_event_only": novelty_prediction,
            "T06_intensity_congestion_event_only": intensity_prediction,
            "T07_corroboration_contradiction_event_only": signal_prediction,
            "T08_structured_all_event_only": structured_prediction,
        }
        event_frames: dict[str, pd.DataFrame] = {}
        for candidate_id, prediction in event_models.items():
            frame = make_ranked_frame(event_score, prediction, candidate_id)
            event_frames[candidate_id] = frame
            score_parts[candidate_id].append(frame)

        event_keys = pd.MultiIndex.from_frame(event_score[["date", "code"]])
        l4_keys = pd.MultiIndex.from_frame(l4[["date", "code"]])
        no_event = l4.loc[~l4_keys.isin(event_keys)].copy()
        no_event["candidate_id"] = "N0_no_event_L4"
        score_parts["N0_no_event_L4"].append(no_event)

        overlays = {
            "T03_char_rank_overlay_L4": event_frames[
                "T01_char_rank_event_only"
            ],
            "T04_char_value_overlay_L4": event_frames[
                "T02_char_value_event_only"
            ],
            "T09_structured_overlay_L4": event_frames[
                "T08_structured_all_event_only"
            ],
        }
        for candidate_id, event_frame in overlays.items():
            score_parts[candidate_id].append(
                add_overlay(l4, event_frame, candidate_id)
            )
        score_parts["T10_event_barbell"].append(
            build_barbell(
                l4, event_frames["T02_char_value_event_only"], "T10_event_barbell"
            )
        )
        fold_reports.append(
            {
                "period": str(period),
                "score_sessions": int(len(month_dates)),
                "score_session_first": str(month_dates.min().date()),
                "score_session_last": str(month_dates.max().date()),
                "panel_train_rows": int(len(panel_train)),
                "panel_train_end": str(panel_train["date"].max().date()),
                "panel_score_rows": int(len(panel_score)),
                "event_train_bundles": int(len(event_train)),
                "event_train_end": str(event_train["date"].max().date()),
                "event_score_bundles": int(len(event_score)),
                "tfidf_features": int(x_train.shape[1]),
                "strictly_prior_training": bool(
                    event_train["date"].max() < month_start
                    and panel_train["date"].max() < month_start
                ),
                "value_target_clip": [lower, upper],
            }
        )
        del (
            panel_train,
            panel_score,
            panel_rank,
            event_train,
            event_score,
            x_train,
            x_score,
            vectorizer,
            rank_model,
            value_model,
            g0_model,
            l4_model,
        )
        gc.collect()

    scores = {
        name: pd.concat(parts, ignore_index=True)
        for name, parts in score_parts.items()
    }
    slices = {
        "discovery": (pd.Timestamp("2024-07-01"), pd.Timestamp("2025-05-31")),
        "confirmation_a": (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-30")),
        "confirmation_b": (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-07-31")),
    }
    all_picks: list[pd.DataFrame] = []
    candidate_results: dict[str, Any] = {}
    for candidate_id, frame in scores.items():
        candidate_results[candidate_id] = {}
        for top_k in (1, 2):
            picks = select_slots(frame, score_sessions, top_k)
            picks["top_k"] = top_k
            all_picks.append(picks)
            candidate_results[candidate_id][f"top{top_k}"] = metrics_for_picks(
                picks, top_k, slices
            )

    picks_frame = pd.concat(all_picks, ignore_index=True)
    picks_path = HERE / "picks.csv"
    picks_frame.to_csv(picks_path, index=False)
    bundles_path = HERE / "bundle_coverage.csv"
    bundles[
        [
            "date",
            "code",
            "document_count",
            "positive_family_count",
            "negative_family_count",
            "mixed_sign",
        ]
    ].to_csv(bundles_path, index=False)
    complete_score_set = set(score_sessions)
    score_missing = {
        date: value
        for date, value in missing_by_session.items()
        if SCORE_START <= pd.Timestamp(date) <= SCORE_END
    }
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "authority": "exploratory_only_no_production_promotion",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "integrity": {
            **integrity,
            "strictly_prior_training_all_folds": all(
                fold["strictly_prior_training"] for fold in fold_reports
            ),
            "future_publication_violations": mapping_report[
                "future_publication_violations"
            ],
            "source_missing_never_encoded_no_event": True,
            "score_sessions_exactly_once": bool(
                len(score_sessions) == len(complete_score_set)
            ),
        },
        "coverage": {
            **parse_report,
            **mapping_report,
            "panel_sessions": int(len(sessions)),
            "strict_complete_sessions_all_panel": int(len(complete_sessions)),
            "strict_complete_score_sessions": int(len(score_sessions)),
            "score_months": [str(value) for value in score_months],
            "source_missing_score_sessions": int(
                sum(
                    SCORE_START <= date <= SCORE_END and not bool(value)
                    for date, value in strict_complete.items()
                )
            ),
            "source_missing_examples": dict(list(score_missing.items())[:20]),
            "qualifying_bundles": int(len(bundles)),
            "qualifying_score_bundles": int(
                bundles["date"].isin(score_sessions).sum()
            ),
            "score_days_with_event_bundle": int(
                bundles.loc[bundles["date"].isin(score_sessions), "date"].nunique()
            ),
            "historical_cache_observation_metadata_available": False,
        },
        "folds": fold_reports,
        "candidates": candidate_results,
        "artifacts": {
            "picks_csv": str(picks_path),
            "picks_sha256": sha256_file(picks_path),
            "bundle_coverage_csv": str(bundles_path),
            "bundle_coverage_sha256": sha256_file(bundles_path),
        },
    }
    result_path = HERE / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "result": str(result_path),
                "score_sessions": len(score_sessions),
                "score_months": [str(value) for value in score_months],
                "candidates": len(candidate_results),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
