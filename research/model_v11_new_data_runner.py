#!/usr/bin/env python3
"""Zero-base validation of information channels absent from the price panel.

The protocol was frozen before this lane inspected any candidate outcomes.
Only two locally available channels are evaluated: disclosure-bundle conflict
metadata and release-clock/fiscal-horizon metadata.  URL-only TDnet records
are never treated as document bodies or structured numeric disclosures.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tse_session_ranker.data.tdnet import _title_flags, parse_tdnet_index  # noqa: E402
from tse_session_ranker.research_models import (  # noqa: E402
    ResearchModelSpec,
    fit_research_model,
)


PROTOCOL_PATH = ROOT / "research" / "model_v11_new_data_protocol.json"
PANEL_PATH = Path("/tmp/model_v07_corrected_panel.pkl")
PANEL_MANIFEST_PATH = Path("/tmp/model_v07_corrected_panel.pkl.manifest.json")
CACHE_PATHS = (ROOT / "tdnet_date_cache_2024", ROOT / "tdnet_date_cache")
RESULT_PATH = ROOT / "research" / "model_v11_new_data_result.json"
PICKS_PATH = ROOT / "research" / "model_v11_new_data_picks.csv"
INVENTORY_PATH = ROOT / "research" / "model_v11_new_data_inventory.json"

COSTS_BPS = (0.0, 20.0, 40.0, 60.0)
TOP_K_VALUES = (1, 2)
SCORE_START = pd.Timestamp("2024-05-01")
SCORE_END = pd.Timestamp("2025-07-31")

L4_FEATURES = (
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
    "flat_oc_rate_20",
)

ND04_FEATURES = (
    "document_count_log1p",
    "bundle_width_minutes_log1p",
    "supportive_family_count",
    "adverse_family_count",
    "supportive_corroboration",
    "adverse_corroboration",
    "supportive_adverse_conflict",
    "same_minute_adverse_companion",
    "unclear_direction",
)

ND05_FEATURES = (
    "premarket_fraction",
    "postclose_fraction",
    "latest_age_hours_log1p",
    "minutes_since_prior_close_log1p",
    "latest_release_clock_sin",
    "latest_release_clock_cos",
    "fiscal_q1",
    "fiscal_q2_or_interim",
    "fiscal_q3",
    "fiscal_full_year",
    "fiscal_month_sin",
    "fiscal_month_cos",
    "fiscal_month_missing",
    "earnings_and_revision",
)

CANDIDATE_FEATURES: dict[str, tuple[str, ...]] = {
    "ND04_simultaneous_adverse_bundle": ND04_FEATURES,
    "ND05_release_clock_and_fiscal_horizon": ND05_FEATURES,
    "ND04_plus_ND05": (*ND04_FEATURES, *ND05_FEATURES),
}

SUPPORTIVE_FLAG_NAMES = (
    "revision_up",
    "dividend_up",
    "buyback_decision",
    "benefit",
    "split",
)
ADVERSE_FLAG_NAMES = (
    "revision_down",
    "dividend_down",
    "external_financing",
    "impairment_loss",
    "audit_problem",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
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
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != "model_v11_new_data_zero_base_20260723":
        raise RuntimeError("unexpected protocol id")
    if protocol["authority"]["production_promotion_allowed"] is not False:
        raise RuntimeError("this retrospective protocol cannot promote production")
    if protocol["retrospective_implementation"]["estimator"]["hyperparameter_search"] != "none":
        raise RuntimeError("hyperparameter search is forbidden")
    expected = set(CANDIDATE_FEATURES)
    registered = set(protocol["retrospective_implementation"]["candidate_family"])
    if expected != registered:
        raise RuntimeError("runner candidate family differs from frozen protocol")
    return protocol


def validate_inputs(protocol: dict[str, Any]) -> dict[str, Any]:
    expected_panel = protocol["frozen_inputs"]["corrected_panel"]
    actual_panel = sha256_file(PANEL_PATH)
    actual_manifest = sha256_file(PANEL_MANIFEST_PATH)
    if actual_panel != expected_panel["sha256"]:
        raise RuntimeError("corrected panel SHA-256 changed")
    if actual_manifest != expected_panel["manifest_sha256"]:
        raise RuntimeError("corrected panel manifest SHA-256 changed")
    cache_checks: list[dict[str, Any]] = []
    expected_caches = protocol["frozen_inputs"]["tdnet_index_caches"]
    for path, expected in zip(CACHE_PATHS, expected_caches, strict=True):
        count, digest = cache_digest(path)
        if count != int(expected["html_files"]) or digest != expected["aggregate_sha256"]:
            raise RuntimeError(f"TDnet cache changed: {path}")
        cache_checks.append(
            {
                "path": str(path.relative_to(ROOT)),
                "html_files": count,
                "aggregate_sha256": digest,
            }
        )
    return {
        "panel_sha256": actual_panel,
        "panel_manifest_sha256": actual_manifest,
        "cache_checks": cache_checks,
    }


def parse_caches() -> tuple[pd.DataFrame, set[pd.Timestamp], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    source_dates: set[pd.Timestamp] = set()
    page_counts: dict[str, int] = {}
    for directory in CACHE_PATHS:
        for path in sorted(directory.glob("*.html")):
            match = re.search(r"(20\d{6})", path.stem)
            if match is None:
                raise RuntimeError(f"cache file lacks a date: {path}")
            date = pd.to_datetime(match.group(1), format="%Y%m%d").normalize()
            if date in source_dates:
                raise RuntimeError(f"duplicate cached date: {date.date()}")
            parsed = parse_tdnet_index(path.read_bytes(), date)
            frames.append(parsed)
            source_dates.add(date)
            page_counts[str(date.date())] = int(len(parsed))
    disclosures = pd.concat(frames, ignore_index=True).sort_values(
        ["published_at", "code", "url"], kind="stable"
    )
    urls = disclosures["url"].fillna("").astype(str)
    release_url = urls.str.contains("release.tdnet.info", case=False, regex=False)
    document_id = urls.str.extract(r"/([0-9]{18,})\.pdf(?:$|[?&])", expand=False)
    return disclosures, source_dates, {
        "cached_index_pages": int(len(source_dates)),
        "cached_index_rows": int(len(disclosures)),
        "empty_index_pages": int(sum(value == 0 for value in page_counts.values())),
        "first_cached_date": str(min(source_dates).date()),
        "last_cached_date": str(max(source_dates).date()),
        "nonempty_document_urls": int(urls.str.len().gt(0).sum()),
        "release_tdnet_urls": int(release_url.sum()),
        "unique_document_urls": int(urls.nunique()),
        "extractable_document_ids": int(document_id.notna().sum()),
        "local_tdnet_document_bodies": 0,
        "local_tdnet_pdf_files": 0,
        "local_tdnet_xbrl_files": 0,
    }


def strict_complete_sessions(
    sessions: pd.DatetimeIndex,
    source_dates: set[pd.Timestamp],
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
        absent = [str(value.date()) for value in required if value not in source_dates]
        complete[date] = not absent
        if absent:
            missing[str(date.date())] = absent
    return pd.Series(complete, dtype=bool), missing


def close_timestamp(session: pd.Timestamp) -> pd.Timestamp:
    hour, minute = (15, 0) if session < pd.Timestamp("2024-11-05") else (15, 30)
    return pd.Timestamp(
        year=session.year,
        month=session.month,
        day=session.day,
        hour=hour,
        minute=minute,
        tz="Asia/Tokyo",
    )


def map_disclosures(
    disclosures: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    strict_complete: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cutoffs = pd.DatetimeIndex(
        [
            pd.Timestamp(
                year=date.year,
                month=date.month,
                day=date.day,
                hour=8,
                minute=58,
                second=59,
                tz="Asia/Tokyo",
            )
            for date in sessions
        ]
    )
    published = pd.DatetimeIndex(disclosures["published_at"])
    target_positions = cutoffs.searchsorted(published, side="left")
    in_range = (target_positions > 0) & (target_positions < len(sessions))
    events = disclosures.loc[in_range].copy()
    target_positions = target_positions[in_range]
    events["date"] = sessions[target_positions].to_numpy()
    events["prior_session"] = sessions[target_positions - 1].to_numpy()
    events["target_cutoff"] = pd.DatetimeIndex(cutoffs[target_positions])
    events["prior_close_at"] = pd.DatetimeIndex(
        [close_timestamp(value) for value in events["prior_session"]]
    )
    same_day = events["index_date"].eq(events["date"])
    index_is_session = events["index_date"].isin(sessions)
    published_minute = events["published_at"].dt.hour * 60 + events["published_at"].dt.minute
    source_close_minute = np.where(
        events["index_date"].lt(pd.Timestamp("2024-11-05")),
        15 * 60,
        15 * 60 + 30,
    )
    intraday_prior_session = (
        index_is_session & ~same_day & published_minute.lt(source_close_minute)
    )
    events["qualifying_overnight"] = ~intraday_prior_session
    events["target_source_complete"] = events["date"].map(strict_complete).fillna(False)
    events["age_hours"] = (
        events["target_cutoff"] - events["published_at"]
    ).dt.total_seconds().div(3600.0)
    events["minutes_since_prior_close"] = (
        events["published_at"] - events["prior_close_at"]
    ).dt.total_seconds().div(60.0)
    events["same_day_premarket"] = same_day.astype(float)
    events["prior_or_intervening_postclose"] = (~same_day).astype(float)
    if events["age_hours"].lt(0).any():
        raise RuntimeError("a disclosure was mapped before its publication")
    if events.loc[events["qualifying_overnight"], "minutes_since_prior_close"].lt(0).any():
        raise RuntimeError("a qualifying disclosure predates the prior close")
    qualified = events.loc[
        events["qualifying_overnight"] & events["target_source_complete"]
    ].copy()
    return qualified, {
        "mapped_disclosures": int(len(events)),
        "excluded_intraday_prior_session": int(intraday_prior_session.sum()),
        "excluded_source_incomplete": int(
            (events["qualifying_overnight"] & ~events["target_source_complete"]).sum()
        ),
        "qualifying_disclosures": int(len(qualified)),
        "future_publication_violations": int(events["age_hours"].lt(0).sum()),
        "pre_prior_close_violations": int(
            events.loc[events["qualifying_overnight"], "minutes_since_prior_close"].lt(0).sum()
        ),
    }


def normalized_title(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value)).casefold()


def fiscal_features(text: str) -> dict[str, float]:
    normalized = normalized_title(text)
    months = [
        int(value)
        for value in re.findall(r"(?<!\d)(1[0-2]|[1-9])月期", normalized)
    ]
    if months:
        radians = [2.0 * math.pi * (value - 1.0) / 12.0 for value in months]
        month_sin = float(np.mean(np.sin(radians)))
        month_cos = float(np.mean(np.cos(radians)))
        missing = 0.0
    else:
        month_sin = 0.0
        month_cos = 0.0
        missing = 1.0
    return {
        "fiscal_q1": float(bool(re.search(r"第?1四半期", normalized))),
        "fiscal_q2_or_interim": float(
            bool(re.search(r"第?2四半期|中間期|中間決算", normalized))
        ),
        "fiscal_q3": float(bool(re.search(r"第?3四半期", normalized))),
        "fiscal_full_year": float(
            bool(re.search(r"通期|年度|年[度期]決算|本決算", normalized))
        ),
        "fiscal_month_sin": month_sin,
        "fiscal_month_cos": month_cos,
        "fiscal_month_missing": missing,
    }


def build_bundles(
    events: pd.DataFrame,
    complete_sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    flags = _title_flags(events["title"]).reset_index(drop=True)
    events = pd.concat([events.reset_index(drop=True), flags], axis=1)
    events["external_financing_initial"] = (
        events["equity_financing"].gt(0)
        & events["equity_financing_status"].eq(0)
    ).astype(float)
    events["supportive_row"] = events[list(SUPPORTIVE_FLAG_NAMES)].max(axis=1)
    adverse_columns = [
        "revision_down",
        "dividend_down",
        "external_financing_initial",
        "impairment_loss",
        "audit_problem",
    ]
    events["adverse_row"] = events[adverse_columns].max(axis=1)
    events["publication_minute"] = events["published_at"].dt.floor("min")
    minute = (
        events.groupby(["date", "code", "publication_minute"], sort=True)
        .agg(
            minute_supportive=("supportive_row", "max"),
            minute_adverse=("adverse_row", "max"),
        )
        .reset_index()
    )
    minute["same_minute_adverse_companion"] = (
        minute["minute_supportive"].gt(0) & minute["minute_adverse"].gt(0)
    ).astype(float)
    minute_conflict = (
        minute.groupby(["date", "code"], sort=True)["same_minute_adverse_companion"]
        .max()
        .rename("same_minute_adverse_companion")
        .reset_index()
    )

    keys = ["date", "code"]
    grouped = events.groupby(keys, sort=True)
    bundles = grouped.agg(
        name=("name", "last"),
        bundle_text=("title", lambda values: " [DOC] ".join(values.astype(str))),
        document_count=("title", "size"),
        first_published=("published_at", "min"),
        last_published=("published_at", "max"),
        latest_age_hours=("age_hours", "min"),
        maximum_minutes_since_prior_close=("minutes_since_prior_close", "max"),
        premarket_fraction=("same_day_premarket", "mean"),
        postclose_fraction=("prior_or_intervening_postclose", "mean"),
    ).reset_index()
    bundles = bundles.merge(minute_conflict, on=keys, how="left", validate="one_to_one")

    family_names = (
        "earnings",
        "revision",
        "revision_up",
        "revision_down",
        "dividend",
        "dividend_up",
        "dividend_down",
        "buyback_decision",
        "benefit",
        "split",
        "external_financing_initial",
        "impairment_loss",
        "audit_problem",
    )
    family = grouped[list(family_names)].max().reset_index()
    bundles = bundles.merge(family, on=keys, how="left", validate="one_to_one")
    bundles["supportive_family_count"] = bundles[
        list(SUPPORTIVE_FLAG_NAMES)
    ].sum(axis=1)
    bundle_adverse_columns = [
        "revision_down",
        "dividend_down",
        "external_financing_initial",
        "impairment_loss",
        "audit_problem",
    ]
    bundles["adverse_family_count"] = bundles[bundle_adverse_columns].sum(axis=1)
    bundles["supportive_corroboration"] = (
        bundles["supportive_family_count"].ge(2)
    ).astype(float)
    bundles["adverse_corroboration"] = (
        bundles["adverse_family_count"].ge(2)
    ).astype(float)
    bundles["supportive_adverse_conflict"] = (
        bundles["supportive_family_count"].gt(0)
        & bundles["adverse_family_count"].gt(0)
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
    bundles["earnings_and_revision"] = (
        bundles["earnings"].gt(0) & bundles["revision"].gt(0)
    ).astype(float)
    bundles["bundle_width_minutes"] = (
        bundles["last_published"] - bundles["first_published"]
    ).dt.total_seconds().div(60.0)
    bundles["document_count_log1p"] = np.log1p(bundles["document_count"])
    bundles["bundle_width_minutes_log1p"] = np.log1p(
        bundles["bundle_width_minutes"].clip(lower=0)
    )
    bundles["latest_age_hours_log1p"] = np.log1p(
        bundles["latest_age_hours"].clip(lower=0)
    )
    bundles["minutes_since_prior_close_log1p"] = np.log1p(
        bundles["maximum_minutes_since_prior_close"].clip(lower=0)
    )
    latest_minutes = (
        bundles["last_published"].dt.hour * 60
        + bundles["last_published"].dt.minute
        + bundles["last_published"].dt.second / 60.0
    )
    angle = 2.0 * np.pi * latest_minutes / (24.0 * 60.0)
    bundles["latest_release_clock_sin"] = np.sin(angle)
    bundles["latest_release_clock_cos"] = np.cos(angle)
    fiscal = pd.DataFrame(
        [fiscal_features(value) for value in bundles["bundle_text"]],
        index=bundles.index,
    )
    bundles = pd.concat([bundles, fiscal], axis=1)

    complete_set = set(complete_sessions)
    if not set(bundles["date"]).issubset(complete_set):
        raise RuntimeError("bundle assigned to a source-incomplete session")
    if bundles[list(dict.fromkeys((*ND04_FEATURES, *ND05_FEATURES)))].isna().any().any():
        raise RuntimeError("registered structured features contain missing values")
    return bundles.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def ranked_frame(
    scoring: pd.DataFrame,
    scores: Iterable[float],
    candidate_id: str,
    fold_period: str,
) -> pd.DataFrame:
    keep = scoring[["date", "code", "name", "oc_return_pct"]].copy()
    keep["model_score"] = np.asarray(list(scores), dtype=float)
    keep["candidate_id"] = candidate_id
    keep["fold_period"] = fold_period
    return keep


def select_slots(
    scores: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    top_k: int,
) -> pd.DataFrame:
    ranked = (
        scores.sort_values(
            ["date", "model_score", "code"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("date", sort=True)
        .head(top_k)
        .copy()
    )
    ranked["model_rank"] = ranked.groupby("date", sort=False).cumcount() + 1
    desired = pd.MultiIndex.from_product(
        [sessions, range(1, top_k + 1)],
        names=["date", "model_rank"],
    ).to_frame(index=False)
    selected = desired.merge(
        ranked[
            [
                "date",
                "model_rank",
                "code",
                "name",
                "model_score",
                "oc_return_pct",
                "candidate_id",
                "fold_period",
            ]
        ],
        on=["date", "model_rank"],
        how="left",
        validate="one_to_one",
    )
    selected["candidate_id"] = selected["candidate_id"].fillna(
        str(scores["candidate_id"].iloc[0])
    )
    selected["top_k"] = int(top_k)
    selected["executed"] = selected["oc_return_pct"].notna()
    return selected


def daily_returns(picks: pd.DataFrame, top_k: int, cost_bps: float) -> pd.Series:
    executed = picks["oc_return_pct"].notna()
    slot = (
        picks["oc_return_pct"].fillna(0.0)
        - executed.astype(float) * cost_bps / 100.0
    )
    return slot.groupby(picks["date"], sort=True).sum().div(top_k)


def metrics_for_picks(
    picks: pd.DataFrame,
    top_k: int,
    slices: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, Any]:
    by_cost: dict[str, Any] = {}
    daily_by_cost: dict[float, pd.Series] = {}
    for cost in COSTS_BPS:
        daily = daily_returns(picks, top_k, cost)
        daily_by_cost[cost] = daily
        by_cost[str(int(cost))] = {
            "days": int(len(daily)),
            "mean_pct": float(daily.mean()),
            "median_pct": float(daily.median()),
            "win_rate": float(daily.gt(0).mean()),
            "worst_day_pct": float(daily.min()),
        }
    daily20 = daily_by_cost[20.0]
    daily40 = daily_by_cost[40.0]
    monthly = daily40.groupby(daily40.index.to_period("M")).mean()
    slice_metrics: dict[str, Any] = {}
    for name, (start, end) in slices.items():
        values = daily40.loc[(daily40.index >= start) & (daily40.index <= end)]
        slice_metrics[name] = {
            "days": int(len(values)),
            "net40_mean_pct": float(values.mean()) if len(values) else None,
        }
    remove_count = min(20, len(daily20))
    removed = daily20.drop(daily20.nlargest(remove_count).index)

    executed = picks["oc_return_pct"].notna()
    contribution = (
        picks["oc_return_pct"].fillna(0.0) - executed.astype(float) * 0.20
    ).div(top_k)
    by_code = (
        contribution.groupby(picks["code"], dropna=True).sum().sort_values(ascending=False)
    )
    positive_code = by_code.loc[by_code.gt(0)]
    top_codes = [str(value) for value in positive_code.head(10).index]
    replaced = contribution.mask(picks["code"].isin(top_codes), 0.0)
    replaced_daily = replaced.groupby(picks["date"], sort=True).sum()
    top_share = (
        float(positive_code.head(10).sum() / positive_code.sum())
        if float(positive_code.sum()) > 0
        else None
    )
    return {
        "costs_bps": by_cost,
        "monthly_net40_mean_pct": {
            str(period): float(value) for period, value in monthly.items()
        },
        "positive_months": int(monthly.gt(0).sum()),
        "months": int(len(monthly)),
        "slices": slice_metrics,
        "top20_winning_days_removed_net20_mean_pct": (
            float(removed.mean()) if len(removed) else None
        ),
        "top10_profit_codes_cash_net20_mean_pct": float(replaced_daily.mean()),
        "top10_profit_codes": top_codes,
        "top10_positive_profit_share": top_share,
        "filled_slots": int(executed.sum()),
        "scheduled_slots": int(len(picks)),
        "slot_fill_rate": float(executed.mean()),
        "days_with_at_least_one_filled_slot": int(
            executed.groupby(picks["date"], sort=True).any().sum()
        ),
        "unique_codes": int(picks["code"].nunique(dropna=True)),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for index, key in enumerate(ordered):
        value = min(1.0, (family_size - index) * p_values[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def paired_familywise(picks: pd.DataFrame) -> dict[str, Any]:
    raw: dict[str, float] = {}
    details: dict[str, Any] = {}
    for top_k in TOP_K_VALUES:
        local = picks.loc[picks["top_k"].eq(top_k)]
        daily: dict[str, pd.Series] = {}
        for candidate_id in sorted(local["candidate_id"].unique()):
            daily[candidate_id] = daily_returns(
                local.loc[local["candidate_id"].eq(candidate_id)],
                top_k,
                40.0,
            )
        baseline = daily["L4_price_control"]
        for candidate_id in sorted(CANDIDATE_FEATURES):
            aligned = pd.concat(
                [daily[candidate_id].rename("candidate"), baseline.rename("baseline")],
                axis=1,
                join="inner",
            ).dropna()
            key = f"{candidate_id}__top{top_k}"
            statistic, p_value = ttest_rel(
                aligned["candidate"],
                aligned["baseline"],
                nan_policy="raise",
            )
            raw[key] = float(p_value)
            details[key] = {
                "sessions": int(len(aligned)),
                "candidate_net40_mean_pct": float(aligned["candidate"].mean()),
                "L4_net40_mean_pct": float(aligned["baseline"].mean()),
                "delta_net40_mean_pct": float(
                    (aligned["candidate"] - aligned["baseline"]).mean()
                ),
                "paired_t_statistic": float(statistic),
                "raw_two_sided_p_value": float(p_value),
            }
    adjusted = holm_adjust(raw)
    for key, value in adjusted.items():
        details[key]["holm_adjusted_p_value"] = float(value)
        details[key]["reject_familywise_5pct"] = bool(value <= 0.05)
    return {
        "method": "Holm correction over paired two-sided daily t-tests",
        "family_size": int(len(details)),
        "alpha": 0.05,
        "tests": details,
    }


def build_inventory(
    protocol: dict[str, Any],
    parse_report: dict[str, Any],
    strict_complete: pd.Series,
    bundles: pd.DataFrame,
    missing_by_session: dict[str, list[str]],
) -> dict[str, Any]:
    all_files = [path for path in ROOT.rglob("*") if path.is_file()]
    tdnet_pdfs = [
        path
        for path in all_files
        if path.suffix.casefold() == ".pdf" and "tdnet" in path.as_posix().casefold()
    ]
    structured = [
        path
        for path in all_files
        if path.suffix.casefold() in {".xbrl", ".parquet", ".feather"}
    ]
    complete_dates = pd.DatetimeIndex(strict_complete.loc[strict_complete].index)
    early = complete_dates[
        (complete_dates >= pd.Timestamp("2024-05-01"))
        & (complete_dates <= pd.Timestamp("2024-06-30"))
    ]
    gap = strict_complete.loc[
        (strict_complete.index >= pd.Timestamp("2024-08-01"))
        & (strict_complete.index <= pd.Timestamp("2025-03-30"))
    ]
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "inspection_scope": [str(ROOT), "/tmp"],
        "tdnet_index": {
            **parse_report,
            "contains_publication_timestamp": True,
            "contains_code_name_title_and_document_url": True,
            "contains_local_receipt_timestamp": False,
            "contains_document_body": False,
            "contains_structured_numeric_fields": False,
            "url_is_not_body": True,
        },
        "local_body_and_structured_files": {
            "tdnet_pdf_count": int(len(tdnet_pdfs)),
            "tdnet_pdf_examples": [
                str(value.relative_to(ROOT)) for value in tdnet_pdfs[:10]
            ],
            "xbrl_parquet_feather_count": int(len(structured)),
            "note": "Monthly JPX OHLC PDFs are market-price archives, not TDnet document bodies.",
        },
        "periods": {
            "panel_bounds": ["2024-01-04", "2025-07-31"],
            "cached_tdnet_bounds": [
                parse_report["first_cached_date"],
                parse_report["last_cached_date"],
            ],
            "strict_complete_early_2024_sessions": int(len(early)),
            "early_2024_session_bounds": (
                [str(early.min().date()), str(early.max().date())] if len(early) else []
            ),
            "cache_gap_2024_08_to_2025_03_panel_sessions": int(len(gap)),
            "cache_gap_strict_complete_sessions": int(gap.sum()),
            "missing_source_examples": dict(list(missing_by_session.items())[:20]),
            "qualifying_bundles_early_2024": int(
                bundles["date"].isin(early).sum()
            ),
        },
        "availability_matrix": {
            "ND01_forecast_old_to_new": {
                "local": False,
                "reason": "Index URLs and titles exist, but PDF/structured cells and receipt sidecars do not.",
                "needed": "J-Quants TDnet add-on or Pro statements feed, or immutable PDF archive plus a versioned table parser."
            },
            "ND02_profit_delta_to_market_cap": {
                "local": False,
                "reason": "ND01 inputs plus effective-dated shares outstanding and PIT market cap are absent.",
                "needed": "Structured forecast feed, shares/treasury-share security master and D-1 close."
            },
            "ND03_buyback_pressure": {
                "local": False,
                "reason": "Buyback title flags exist but maximum shares/amount/method/dates and PIT capital structure do not.",
                "needed": "J-Quants Pro share-buyback TDnet fields or versioned PDF extraction plus security master."
            },
            "ND04_simultaneous_adverse_bundle": {
                "local": True,
                "provenance": "historical-only; publication time is present, receipt time is absent",
                "rows": int(len(bundles))
            },
            "ND05_release_clock_and_fiscal_horizon": {
                "local": True,
                "provenance": "historical-only; publication time is present, receipt time is absent",
                "rows": int(len(bundles))
            },
            "ND06_pdf_table_shape_and_numeric_density": {
                "local": False,
                "reason": "No TDnet PDF bytes are cached.",
                "needed": "Immutable document bytes, receipt hash/time, parser version and confidence."
            },
            "ND07_body_negation_and_cancellation": {
                "local": False,
                "reason": "No TDnet body text or PDF bytes are cached.",
                "needed": "Official body/PDF archive plus section-aware Japanese parser."
            },
            "ND08_ose_futures_0858": {
                "local": False,
                "reason": "No exact historical 08:58 futures snapshots are present.",
                "needed": "Broker market-data API for prospective capture or a licensed historical OSE feed."
            },
            "ND09_tse_auction_0858": {
                "local": False,
                "reason": "No pre-open order-book snapshot is present; actual open is a forbidden proxy.",
                "needed": "Prospective broker API capture or licensed JPX 10-level reconstruction data."
            },
            "ND10_pts_night_and_day": {
                "local": False,
                "reason": "No venue/session-specific PTS trade or quote history is present.",
                "needed": "Japannext ITCH/GLIMPSE or an authorized vendor contract."
            },
            "ND11_pit_liquidity_and_unit_cost": {
                "local": False,
                "reason": "The frozen JPX daily table and monthly PDFs contain OHLC but no usable volume, turnover, lot, spread or market cap.",
                "needed": "J-Quants/DataCube daily volume/turnover plus effective-dated security master and execution ledger."
            }
        },
        "fail_closed": {
            "blocked_hypotheses_scored": 0,
            "missing_source_encoded_as_no_event": False,
            "URL_treated_as_document_body": False,
            "daily_OHLC_treated_as_0858_snapshot": False
        }
    }


def main() -> None:
    protocol = load_protocol()
    integrity = validate_inputs(protocol)
    disclosures, source_dates, parse_report = parse_caches()

    projection = [
        "date",
        "code",
        "name",
        "oc_return_pct",
        "price_eligible",
        "price_training_eligible",
        *L4_FEATURES,
    ]
    raw_panel = joblib.load(PANEL_PATH, mmap_mode="r")
    panel = raw_panel[projection].copy()
    del raw_panel
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["code"] = panel["code"].astype(str)
    sessions = pd.DatetimeIndex(panel["date"].drop_duplicates().sort_values())

    strict_complete, missing_by_session = strict_complete_sessions(
        sessions, source_dates
    )
    complete_sessions = pd.DatetimeIndex(strict_complete.loc[strict_complete].index)
    events, mapping_report = map_disclosures(
        disclosures, sessions, strict_complete
    )
    bundles = build_bundles(events, complete_sessions)
    del disclosures, events
    gc.collect()

    inventory = build_inventory(
        protocol,
        parse_report,
        strict_complete,
        bundles,
        missing_by_session,
    )
    INVENTORY_PATH.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    candidate_parts: dict[str, list[pd.DataFrame]] = {
        "L4_price_control": [],
        **{name: [] for name in CANDIDATE_FEATURES},
    }
    fold_reports: list[dict[str, Any]] = []
    score_periods = pd.PeriodIndex(
        complete_sessions[
            (complete_sessions >= SCORE_START) & (complete_sessions <= SCORE_END)
        ].to_period("M").unique()
    ).sort_values()
    accepted_score_sessions: list[pd.Timestamp] = []

    for period in score_periods:
        month_start = period.start_time.normalize()
        month_dates = complete_sessions[
            (complete_sessions.to_period("M") == period)
            & (complete_sessions >= SCORE_START)
            & (complete_sessions <= SCORE_END)
        ]
        prior_complete = complete_sessions[complete_sessions < month_start]
        bundle_train_count = int(
            bundles.loc[bundles["date"].lt(month_start), ["date", "code"]]
            .drop_duplicates()
            .shape[0]
        )
        if len(prior_complete) < 20 or bundle_train_count < 100:
            fold_reports.append(
                {
                    "period": str(period),
                    "status": "skipped_preregistered_minimum_training_not_met",
                    "score_sessions": int(len(month_dates)),
                    "prior_complete_sessions": int(len(prior_complete)),
                    "prior_event_bundles": bundle_train_count,
                }
            )
            continue

        panel_train = panel.loc[
            panel["date"].lt(month_start)
            & panel["price_training_eligible"].eq(True)
            & panel["oc_return_pct"].notna()
        ].copy()
        panel_score = panel.loc[
            panel["date"].isin(month_dates)
            & panel["price_eligible"].eq(True)
        ].copy()
        if panel_train.empty or panel_score.empty:
            raise RuntimeError(f"empty price fold: {period}")
        if panel_train["date"].max() >= month_start:
            raise RuntimeError(f"non-PIT price training fold: {period}")
        price_model = fit_research_model(
            ResearchModelSpec(
                name=f"model_v11_new_data_L4_{period}",
                family="ridge_daily_rank",
                objective="same_day_return_percentile",
                parameters={"alpha": 1.0},
            ),
            panel_train,
            L4_FEATURES,
        )
        candidate_parts["L4_price_control"].append(
            ranked_frame(
                panel_score,
                price_model.score(panel_score),
                "L4_price_control",
                str(period),
            )
        )

        event_train = bundles.loc[bundles["date"].lt(month_start)].merge(
            panel_train[["date", "code", "name", "oc_return_pct"]],
            on=["date", "code"],
            how="inner",
            suffixes=("", "_panel"),
            validate="one_to_one",
        )
        event_score = bundles.loc[bundles["date"].isin(month_dates)].merge(
            panel_score[["date", "code", "name", "oc_return_pct"]],
            on=["date", "code"],
            how="inner",
            suffixes=("", "_panel"),
            validate="one_to_one",
        )
        if len(event_train) < 100:
            raise RuntimeError(f"joined event train falls below minimum: {period}")
        if event_train["date"].max() >= month_start:
            raise RuntimeError(f"non-PIT event training fold: {period}")

        for candidate_id, columns in CANDIDATE_FEATURES.items():
            model = fit_research_model(
                ResearchModelSpec(
                    name=f"model_v11_new_data_{candidate_id}_{period}",
                    family="ridge_daily_rank",
                    objective="same_day_return_percentile",
                    parameters={"alpha": 10.0},
                ),
                event_train,
                columns,
            )
            candidate_parts[candidate_id].append(
                ranked_frame(
                    event_score,
                    model.score(event_score),
                    candidate_id,
                    str(period),
                )
            )

        accepted_score_sessions.extend(pd.Timestamp(value) for value in month_dates)
        fold_reports.append(
            {
                "period": str(period),
                "status": "scored",
                "score_sessions": int(len(month_dates)),
                "score_start": str(month_dates.min().date()),
                "score_end": str(month_dates.max().date()),
                "prior_complete_sessions": int(len(prior_complete)),
                "price_train_rows": int(len(panel_train)),
                "price_train_end": str(panel_train["date"].max().date()),
                "price_score_rows": int(len(panel_score)),
                "event_train_bundles": int(len(event_train)),
                "event_train_end": str(event_train["date"].max().date()),
                "event_score_bundles": int(len(event_score)),
                "strictly_prior_training": bool(
                    panel_train["date"].max() < month_start
                    and event_train["date"].max() < month_start
                ),
                "title_vectorizer_used": False,
                "hyperparameter_search_used": False,
            }
        )
        del panel_train, panel_score, event_train, event_score, price_model
        gc.collect()

    if not accepted_score_sessions:
        raise RuntimeError("no fold met the frozen training minimum")
    score_sessions = pd.DatetimeIndex(sorted(set(accepted_score_sessions)))
    scores = {
        name: pd.concat(parts, ignore_index=True)
        for name, parts in candidate_parts.items()
    }
    slices = {
        "previously_unused_early_2024": (
            pd.Timestamp("2024-05-01"),
            pd.Timestamp("2024-06-30"),
        ),
        "later_reference": (
            pd.Timestamp("2024-07-01"),
            pd.Timestamp("2025-07-31"),
        ),
        "2024_reference": (
            pd.Timestamp("2024-07-01"),
            pd.Timestamp("2024-07-31"),
        ),
        "2025_reference": (
            pd.Timestamp("2025-04-01"),
            pd.Timestamp("2025-07-31"),
        ),
    }

    all_picks: list[pd.DataFrame] = []
    candidate_results: dict[str, Any] = {}
    for candidate_id, frame in scores.items():
        candidate_results[candidate_id] = {}
        for top_k in TOP_K_VALUES:
            picks = select_slots(frame, score_sessions, top_k)
            all_picks.append(picks)
            candidate_results[candidate_id][f"top{top_k}"] = metrics_for_picks(
                picks,
                top_k,
                slices,
            )
    picks_frame = pd.concat(all_picks, ignore_index=True)
    picks_frame.to_csv(PICKS_PATH, index=False)

    familywise = paired_familywise(picks_frame)
    scored_folds = [value for value in fold_reports if value["status"] == "scored"]
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "authority": "exploratory_only_no_production_promotion",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "integrity": {
            **integrity,
            "strictly_prior_training_all_scored_folds": bool(
                all(value["strictly_prior_training"] for value in scored_folds)
            ),
            "future_publication_violations": mapping_report[
                "future_publication_violations"
            ],
            "pre_prior_close_violations": mapping_report[
                "pre_prior_close_violations"
            ],
            "source_missing_never_encoded_as_no_event": True,
            "blocked_hypotheses_scored": 0,
            "URL_treated_as_document_body": False,
            "title_vectorizer_used": False,
            "hyperparameter_search_used": False,
            "score_sessions_unique": bool(len(score_sessions) == len(set(score_sessions))),
        },
        "availability": {
            "registered_hypotheses": int(
                len(protocol["registered_data_generation_hypotheses"])
            ),
            "implementable_hypotheses": [
                "ND04_simultaneous_adverse_bundle",
                "ND05_release_clock_and_fiscal_horizon",
            ],
            "fixed_combination": "ND04_plus_ND05",
            "blocked_hypotheses": [
                value["id"]
                for value in protocol["registered_data_generation_hypotheses"]
                if str(value["status_before_outcome"]).startswith("blocked")
            ],
        },
        "coverage": {
            **parse_report,
            **mapping_report,
            "panel_sessions": int(len(sessions)),
            "strict_complete_sessions": int(len(complete_sessions)),
            "accepted_score_sessions": int(len(score_sessions)),
            "accepted_score_first": str(score_sessions.min().date()),
            "accepted_score_last": str(score_sessions.max().date()),
            "accepted_score_months": [
                str(value) for value in score_sessions.to_period("M").unique()
            ],
            "qualifying_bundles": int(len(bundles)),
            "qualifying_score_bundles": int(
                bundles["date"].isin(score_sessions).sum()
            ),
            "historical_cache_receipt_metadata_available": False,
            "historical_document_bodies_available": False,
        },
        "folds": fold_reports,
        "candidates": candidate_results,
        "familywise": familywise,
        "artifacts": {
            "inventory": str(INVENTORY_PATH.relative_to(ROOT)),
            "inventory_sha256": sha256_file(INVENTORY_PATH),
            "picks": str(PICKS_PATH.relative_to(ROOT)),
            "picks_sha256": sha256_file(PICKS_PATH),
        },
        "conclusion_policy": {
            "production_ready": False,
            "promotion_gate_enabled": False,
            "reason": protocol["promotion_gate"]["reason"],
        },
    }
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": str(RESULT_PATH),
                "inventory": str(INVENTORY_PATH),
                "picks": str(PICKS_PATH),
                "scored_sessions": int(len(score_sessions)),
                "scored_months": [
                    str(value) for value in score_sessions.to_period("M").unique()
                ],
                "candidate_policies": list(CANDIDATE_FEATURES),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
