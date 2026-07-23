#!/usr/bin/env python3
"""Preregistered nonparametric historical-event reaction experiment.

The candidate models in this runner have no fitted outcome coefficients.
Their only learned object is a fold-local TF-IDF vocabulary; predictions are
fixed summaries of strictly prior event outcomes selected by title or event
family similarity.
"""

from __future__ import annotations

import argparse
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
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
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


PROTOCOL_PATH = ROOT / "research" / "model_v11_analog_protocol.json"
PROTOCOL_SHA_PATH = ROOT / "research" / "model_v11_analog_protocol.sha256"
EXPECTED_PROTOCOL_SHA256 = (
    "823012afa89fe9ec982a7ca82c78c3f24697b6969d1d3fcaa73f5ee7d3b82e86"
)
CACHE_PATHS = (ROOT / "tdnet_date_cache_2024", ROOT / "tdnet_date_cache")
SCORE_START = pd.Timestamp("2024-07-01")
SCORE_END = pd.Timestamp("2025-07-31")
COSTS = (0.0, 20.0, 40.0, 60.0)
NEIGHBORS = 31
TIME_HALF_LIFE_SESSIONS = 126.0
PRIMARY_COST_PCT = 0.40
BATCH_SIZE = 128

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
FAMILY_FEATURES = (
    "earnings",
    "revision_up",
    "revision_down",
    "dividend_up",
    "dividend_down",
    "buyback_decision",
    "benefit",
    "split",
    "external_equity_financing",
    "impairment_loss",
    "audit_problem",
    "ma_alliance",
    "control_transaction",
    "correction",
)
ANALOG_IDS = (
    "A01_global_similarity_mean",
    "A02_issuer_excluded_mean",
    "A03_time_decay_mean",
    "A04_cross_code_one_per_issuer",
    "A05_neighbor_lower_quantile",
    "A06_dual_tail_neighbor_utility",
    "A07_event_family_prototype",
    "A08_same_issuer_memory",
    "A09_robust_neighbor_consensus",
    "A10_family_then_text_cross_code",
)
COMPARATOR_IDS = ("L4_price_control", "T02_char_value_event_only")
ALL_IDS = (*COMPARATOR_IDS, *ANALOG_IDS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_arrays(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.ascontiguousarray(values[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def cache_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    files = sorted(path.glob("*.html"))
    for item in files:
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
        digest.update(b"\0")
    return len(files), digest.hexdigest()


def load_and_validate_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    actual_protocol_sha = sha256_file(PROTOCOL_PATH)
    if actual_protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("v11 analog protocol changed after registration")
    recorded = PROTOCOL_SHA_PATH.read_text(encoding="utf-8").strip().split()
    if recorded != [EXPECTED_PROTOCOL_SHA256, "model_v11_analog_protocol.json"]:
        raise RuntimeError("protocol SHA sidecar is invalid")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != "v11_historical_analog_zero_base_20260723":
        raise RuntimeError("unexpected protocol id")
    if protocol["hypothesis_family_size"] != len(ANALOG_IDS):
        raise RuntimeError("candidate family size does not match runner")
    registered = tuple(item["id"] for item in protocol["registered_hypotheses"])
    if registered != ANALOG_IDS:
        raise RuntimeError("candidate order or identity changed")
    if protocol["authority"]["production_promotion_allowed_from_this_run"]:
        raise RuntimeError("retrospective protocol cannot authorize promotion")

    panel_spec = protocol["frozen_inputs"]["panel"]
    panel_path = Path(panel_spec["path"])
    manifest_path = Path(panel_spec["manifest_path"])
    if sha256_file(panel_path) != panel_spec["sha256"]:
        raise RuntimeError("frozen panel hash mismatch")
    if sha256_file(manifest_path) != panel_spec["manifest_sha256"]:
        raise RuntimeError("frozen panel manifest hash mismatch")

    cache_checks = []
    for path, expected in zip(
        CACHE_PATHS, protocol["frozen_inputs"]["tdnet_caches"], strict=True
    ):
        count, digest = cache_digest(path)
        if count != expected["html_files"] or digest != expected["aggregate_sha256"]:
            raise RuntimeError(f"TDnet cache hash mismatch: {path}")
        cache_checks.append(
            {
                "path": str(path.relative_to(ROOT)),
                "html_files": count,
                "aggregate_sha256": digest,
            }
        )
    comparator_checks = {}
    for name, expected in protocol["frozen_inputs"][
        "known_comparator_artifacts"
    ].items():
        path = ROOT / expected["path"]
        actual = sha256_file(path)
        if actual != expected["sha256"]:
            raise RuntimeError(f"known comparator artifact changed: {path}")
        comparator_checks[name] = actual
    return protocol, {
        "protocol_sha256": actual_protocol_sha,
        "panel_sha256": panel_spec["sha256"],
        "panel_manifest_sha256": panel_spec["manifest_sha256"],
        "cache_checks": cache_checks,
        "known_comparator_checks": comparator_checks,
    }


def parse_caches() -> tuple[pd.DataFrame, set[pd.Timestamp], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    source_dates: set[pd.Timestamp] = set()
    empty_pages = 0
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
            empty_pages += int(frame.empty)
    disclosures = pd.concat(frames, ignore_index=True).sort_values(
        ["published_at", "code", "title"], kind="stable"
    )
    return disclosures, source_dates, {
        "pages": len(source_dates),
        "disclosures": int(len(disclosures)),
        "empty_pages": empty_pages,
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
        events["index_date"].lt(pd.Timestamp("2024-11-05")),
        15 * 60,
        15 * 60 + 30,
    )
    intraday_prior = index_is_session & ~same_day & minute.lt(close_minute)
    events["qualifying_overnight"] = ~intraday_prior
    events["target_source_complete"] = (
        events["date"].map(strict_complete).fillna(False)
    )
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
        "excluded_intraday_prior_session": int(intraday_prior.sum()),
        "excluded_target_source_missing": int(
            (
                events["qualifying_overnight"]
                & ~events["target_source_complete"]
            ).sum()
        ),
        "qualifying_disclosures": int(len(qualifying)),
        "future_publication_violations": int(events["age_hours"].lt(0).sum()),
    }


def build_bundles(events: pd.DataFrame) -> pd.DataFrame:
    flags = _title_flags(events["title"])
    events = pd.concat(
        [events.reset_index(drop=True), flags.reset_index(drop=True)], axis=1
    )
    grouped = events.groupby(["date", "code"], sort=True)
    bundles = grouped.agg(
        name=("name", "last"),
        bundle_text=("title", lambda x: " [SEP] ".join(x.astype(str))),
        document_count=("title", "size"),
        first_published=("published_at", "min"),
        last_published=("published_at", "max"),
    ).reset_index()
    raw_family = (
        grouped[
            [
                "earnings",
                "revision_up",
                "revision_down",
                "dividend_up",
                "dividend_down",
                "buyback_decision",
                "benefit",
                "split",
                "equity_financing",
                "equity_financing_status",
                "impairment_loss",
                "audit_problem",
                "ma_alliance",
                "control_transaction",
                "correction",
            ]
        ]
        .max()
        .reset_index()
    )
    raw_family["external_equity_financing"] = (
        raw_family["equity_financing"]
        * (1.0 - raw_family["equity_financing_status"])
    )
    bundles = bundles.merge(
        raw_family[["date", "code", *FAMILY_FEATURES]],
        on=["date", "code"],
        how="left",
        validate="one_to_one",
    )
    market_count = bundles.groupby("date", sort=True)["code"].transform("size")
    bundles["market_bundle_count"] = market_count.astype(int)
    return bundles.sort_values(["date", "code"], kind="stable").reset_index(
        drop=True
    )


def fit_l4(training: pd.DataFrame, period: pd.Period):
    return fit_research_model(
        ResearchModelSpec(
            name=f"v11_L4_{period}",
            family="ridge_daily_rank",
            objective="same_day_return_percentile",
            parameters={"alpha": 1.0},
        ),
        training,
        L4_FEATURES,
    )


def make_score_frame(
    scoring: pd.DataFrame,
    scores: Iterable[float],
    eligible: Iterable[bool],
    candidate_id: str,
) -> pd.DataFrame:
    frame = scoring[["date", "code", "name", "label", "oc_return_pct"]].copy()
    frame["model_score"] = np.asarray(scores, dtype=float)
    frame["score_eligible"] = np.asarray(eligible, dtype=bool)
    frame["candidate_id"] = candidate_id
    return frame


def deterministic_top_indices(
    similarity: np.ndarray,
    allowed: np.ndarray,
    k: int,
    dates_ns: np.ndarray,
    codes: np.ndarray,
) -> np.ndarray:
    valid = np.flatnonzero(allowed & np.isfinite(similarity))
    if not len(valid):
        return np.empty(0, dtype=int)
    take = min(k, len(valid))
    values = similarity[valid]
    if len(valid) > take:
        boundary = np.partition(values, len(values) - take)[len(values) - take]
        pool = valid[values >= boundary]
    else:
        pool = valid
    order = np.lexsort(
        (
            pool,
            codes[pool],
            dates_ns[pool],
            -similarity[pool],
        )
    )
    return pool[order[:take]]


def diversified_cross_code_indices(
    similarity: np.ndarray,
    train_codes: np.ndarray,
    score_code: str,
    dates_ns: np.ndarray,
) -> np.ndarray:
    allowed = train_codes != score_code
    requested = min(max(NEIGHBORS * 8, 512), int(allowed.sum()))
    candidates = deterministic_top_indices(
        similarity, allowed, requested, dates_ns, train_codes
    )
    chosen: list[int] = []
    seen: set[str] = set()
    for index in candidates:
        code = str(train_codes[index])
        if code in seen:
            continue
        chosen.append(int(index))
        seen.add(code)
        if len(chosen) == NEIGHBORS:
            break
    if len(chosen) < min(NEIGHBORS, len(np.unique(train_codes[allowed]))):
        candidates = deterministic_top_indices(
            similarity, allowed, int(allowed.sum()), dates_ns, train_codes
        )
        chosen = []
        seen = set()
        for index in candidates:
            code = str(train_codes[index])
            if code in seen:
                continue
            chosen.append(int(index))
            seen.add(code)
            if len(chosen) == NEIGHBORS:
                break
    return np.asarray(chosen, dtype=int)


def normalized_weights(similarity: np.ndarray) -> np.ndarray:
    weights = np.square(np.clip(similarity.astype(float), 0.0, None))
    if not len(weights):
        return weights
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(len(weights), 1.0 / len(weights), dtype=float)
    return weights / total


def weighted_mean(
    target: np.ndarray,
    similarity: np.ndarray,
    extra_weight: np.ndarray | None = None,
) -> float:
    weights = normalized_weights(similarity)
    if extra_weight is not None:
        weights = weights * np.asarray(extra_weight, dtype=float)
        total = float(weights.sum())
        weights = (
            weights / total
            if total > 0.0
            else np.full(len(weights), 1.0 / len(weights), dtype=float)
        )
    return float(np.dot(weights, target))


def weighted_quantile(
    target: np.ndarray, similarity: np.ndarray, quantile: float
) -> float:
    weights = normalized_weights(similarity)
    order = np.argsort(target, kind="stable")
    cumulative = np.cumsum(weights[order])
    position = min(int(np.searchsorted(cumulative, quantile, side="left")), len(order) - 1)
    return float(target[order[position]])


def family_jaccard(
    score_flags: np.ndarray, train_flags: np.ndarray
) -> np.ndarray:
    intersection = train_flags @ score_flags
    union = train_flags.sum(axis=1) + score_flags.sum() - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros(len(train_flags), dtype=float),
        where=union > 0,
    )


def score_analog_family(
    x_train: sparse.csr_matrix,
    x_score: sparse.csr_matrix,
    train: pd.DataFrame,
    score: pd.DataFrame,
    clipped_target: np.ndarray,
    session_positions: dict[pd.Timestamp, int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    train_codes = train["code"].astype(str).to_numpy()
    score_codes = score["code"].astype(str).to_numpy()
    train_dates = pd.to_datetime(train["date"]).to_numpy()
    dates_ns = train_dates.astype("datetime64[ns]").astype(np.int64)
    train_positions = np.asarray(
        [session_positions[pd.Timestamp(value)] for value in train["date"]],
        dtype=int,
    )
    score_positions = np.asarray(
        [session_positions[pd.Timestamp(value)] for value in score["date"]],
        dtype=int,
    )
    train_flags = train[list(FAMILY_FEATURES)].to_numpy(dtype=np.float32)
    score_flags = score[list(FAMILY_FEATURES)].to_numpy(dtype=np.float32)
    predictions = {
        candidate_id: np.full(len(score), np.nan, dtype=float)
        for candidate_id in ANALOG_IDS
    }
    eligibility = {
        candidate_id: np.ones(len(score), dtype=bool)
        for candidate_id in ANALOG_IDS
    }
    support = {
        candidate_id: np.zeros(len(score), dtype=np.int32)
        for candidate_id in ANALOG_IDS
    }
    global_mean = float(np.mean(clipped_target))
    all_allowed = np.ones(len(train), dtype=bool)

    for batch_start in range(0, len(score), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(score))
        similarities = (
            x_score[batch_start:batch_end] @ x_train.T
        ).toarray()
        for local_row, row in enumerate(range(batch_start, batch_end)):
            similarity = np.asarray(similarities[local_row], dtype=float)
            score_code = score_codes[row]
            base = deterministic_top_indices(
                similarity, all_allowed, NEIGHBORS, dates_ns, train_codes
            )
            issuer_excluded = deterministic_top_indices(
                similarity,
                train_codes != score_code,
                NEIGHBORS,
                dates_ns,
                train_codes,
            )
            same_issuer = deterministic_top_indices(
                similarity,
                train_codes == score_code,
                NEIGHBORS,
                dates_ns,
                train_codes,
            )
            diversified = diversified_cross_code_indices(
                similarity, train_codes, score_code, dates_ns
            )
            if not len(base):
                raise RuntimeError("empty global analog support")

            base_target = clipped_target[base]
            base_similarity = similarity[base]
            predictions["A01_global_similarity_mean"][row] = weighted_mean(
                base_target, base_similarity
            )
            support["A01_global_similarity_mean"][row] = len(base)

            predictions["A02_issuer_excluded_mean"][row] = weighted_mean(
                clipped_target[issuer_excluded], similarity[issuer_excluded]
            )
            support["A02_issuer_excluded_mean"][row] = len(issuer_excluded)

            age = score_positions[row] - train_positions[base]
            decay = np.exp(-np.log(2.0) * age / TIME_HALF_LIFE_SESSIONS)
            predictions["A03_time_decay_mean"][row] = weighted_mean(
                base_target, base_similarity, decay
            )
            support["A03_time_decay_mean"][row] = len(base)

            predictions["A04_cross_code_one_per_issuer"][row] = weighted_mean(
                clipped_target[diversified], similarity[diversified]
            )
            support["A04_cross_code_one_per_issuer"][row] = len(diversified)

            lower = weighted_quantile(base_target, base_similarity, 0.35)
            predictions["A05_neighbor_lower_quantile"][row] = lower
            eligibility["A05_neighbor_lower_quantile"][row] = (
                lower > PRIMARY_COST_PCT
            )
            support["A05_neighbor_lower_quantile"][row] = len(base)

            weights = normalized_weights(base_similarity)
            upside = float(np.dot(weights, base_target > 1.0))
            downside = float(np.dot(weights, base_target < -1.0))
            tail_utility = upside - 1.5 * downside
            predictions["A06_dual_tail_neighbor_utility"][row] = tail_utility
            eligibility["A06_dual_tail_neighbor_utility"][row] = (
                tail_utility > 0.0
            )
            support["A06_dual_tail_neighbor_utility"][row] = len(base)

            jaccard = family_jaccard(score_flags[row], train_flags)
            positive_family = jaccard > 0.0
            if positive_family.any():
                family_weights = jaccard[positive_family]
                family_target = clipped_target[positive_family]
                predictions["A07_event_family_prototype"][row] = float(
                    np.average(family_target, weights=family_weights)
                )
                support["A07_event_family_prototype"][row] = int(
                    positive_family.sum()
                )
            else:
                predictions["A07_event_family_prototype"][row] = global_mean
                support["A07_event_family_prototype"][row] = len(train)

            if len(same_issuer):
                predictions["A08_same_issuer_memory"][row] = weighted_mean(
                    clipped_target[same_issuer], similarity[same_issuer]
                )
                support["A08_same_issuer_memory"][row] = len(same_issuer)
            else:
                predictions["A08_same_issuer_memory"][row] = 0.0
                eligibility["A08_same_issuer_memory"][row] = False

            median = weighted_quantile(base_target, base_similarity, 0.50)
            positive_share = float(np.dot(weights, base_target > 0.0))
            negative_share = float(np.dot(weights, base_target < 0.0))
            predictions["A09_robust_neighbor_consensus"][row] = median * abs(
                positive_share - negative_share
            )
            support["A09_robust_neighbor_consensus"][row] = len(base)

            other = train_codes != score_code
            maximum_family = float(jaccard[other].max()) if other.any() else 0.0
            hierarchical_allowed = other & np.isclose(
                jaccard, maximum_family, atol=0.0, rtol=0.0
            )
            hierarchical = deterministic_top_indices(
                similarity,
                hierarchical_allowed,
                NEIGHBORS,
                dates_ns,
                train_codes,
            )
            predictions["A10_family_then_text_cross_code"][row] = weighted_mean(
                clipped_target[hierarchical], similarity[hierarchical]
            )
            support["A10_family_then_text_cross_code"][row] = len(hierarchical)
        del similarities

    for candidate_id in ANALOG_IDS:
        if not np.isfinite(predictions[candidate_id]).all():
            raise RuntimeError(f"non-finite prediction: {candidate_id}")
    support_report = {
        candidate_id: {
            "minimum": int(values.min()) if len(values) else 0,
            "median": float(np.median(values)) if len(values) else 0.0,
            "maximum": int(values.max()) if len(values) else 0,
            "abstained_bundles": int((~eligibility[candidate_id]).sum()),
        }
        for candidate_id, values in support.items()
    }
    return predictions, eligibility, support_report


def select_slots(
    scores: pd.DataFrame, sessions: pd.DatetimeIndex, top_k: int
) -> pd.DataFrame:
    eligible = scores.loc[scores["score_eligible"]].copy()
    ranked = (
        eligible.sort_values(
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
                "score_eligible",
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
    selected["score_eligible"] = selected["score_eligible"].fillna(False)
    return selected


def daily_returns(
    picks: pd.DataFrame, top_k: int, cost_bps: float
) -> pd.Series:
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
    costs: dict[str, Any] = {}
    daily_by_cost: dict[int, pd.Series] = {}
    for cost in COSTS:
        daily = daily_returns(picks, top_k, cost)
        daily_by_cost[int(cost)] = daily
        costs[str(int(cost))] = {
            "mean_pct": float(daily.mean()),
            "median_pct": float(daily.median()),
            "win_rate": float(daily.gt(0.0).mean()),
            "days": int(len(daily)),
        }
    monthly: dict[str, Any] = {}
    for period in daily_by_cost[40].index.to_period("M").unique():
        key = str(period)
        mask = daily_by_cost[40].index.to_period("M") == period
        monthly[key] = {
            f"net{cost}_mean_pct" if cost else "gross_mean_pct": float(
                values.loc[mask].mean()
            )
            for cost, values in daily_by_cost.items()
        }
    slice_metrics: dict[str, Any] = {}
    for name, (start, end) in slices.items():
        row = {}
        mask = (
            (daily_by_cost[40].index >= start)
            & (daily_by_cost[40].index <= end)
        )
        row["days"] = int(mask.sum())
        for cost, values in daily_by_cost.items():
            label = f"net{cost}_mean_pct" if cost else "gross_mean_pct"
            row[label] = float(values.loc[mask].mean()) if mask.any() else None
        slice_metrics[name] = row

    daily20 = daily_by_cost[20]
    removed_dates = list(daily20.nlargest(min(20, len(daily20))).index)
    removed = daily20.drop(removed_dates)
    executed = picks["oc_return_pct"].notna()
    contribution = (
        picks["oc_return_pct"].fillna(0.0) - executed.astype(float) * 0.20
    ).div(top_k)
    by_code = contribution.groupby(picks["code"], dropna=True).sum().sort_values(
        ascending=False
    )
    top_codes = [str(value) for value in by_code.loc[by_code.gt(0.0)].head(10).index]
    cash_contribution = contribution.mask(picks["code"].isin(top_codes), 0.0)
    cash_daily = cash_contribution.groupby(picks["date"], sort=True).sum()
    positive_total = float(by_code.loc[by_code.gt(0.0)].sum())
    largest_positive_share = (
        float(by_code.iloc[0] / positive_total)
        if len(by_code) and by_code.iloc[0] > 0.0 and positive_total > 0.0
        else 0.0
    )
    return {
        "costs": costs,
        "monthly": monthly,
        "positive_months_net40": int(
            sum(row["net40_mean_pct"] > 0.0 for row in monthly.values())
        ),
        "months": int(len(monthly)),
        "slices": slice_metrics,
        "best20_session_removal": {
            "removed_dates": [str(pd.Timestamp(value).date()) for value in removed_dates],
            "remaining_days": int(len(removed)),
            "net20_mean_pct": float(removed.mean()) if len(removed) else None,
        },
        "top10_code_cash": {
            "codes": top_codes,
            "net20_mean_pct": float(cash_daily.mean()),
        },
        "concentration": {
            "largest_positive_code_share": largest_positive_share,
            "unique_codes": int(picks["code"].nunique(dropna=True)),
        },
        "filled_slots": int(executed.sum()),
        "scheduled_slots": int(len(picks)),
        "cash_slots": int((~executed).sum()),
        "slot_fill_rate": float(executed.mean()),
        "days_with_at_least_one_filled_slot": int(
            executed.groupby(picks["date"], sort=True).any().sum()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "research" / "model_v11_analog",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    protocol, integrity = load_and_validate_protocol()
    panel_path = Path(protocol["frozen_inputs"]["panel"]["path"])
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
    raw_panel = joblib.load(panel_path, mmap_mode="r")
    panel = raw_panel[projection].copy()
    del raw_panel
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["code"] = panel["code"].astype(str)
    sessions = pd.DatetimeIndex(panel["date"].drop_duplicates().sort_values())
    session_positions = {date: position for position, date in enumerate(sessions)}
    strict_complete, missing_by_session = strict_complete_sessions(
        sessions, source_dates
    )
    complete_sessions = pd.DatetimeIndex(strict_complete.loc[strict_complete].index)
    events, mapping_report = map_disclosures(
        disclosures, sessions, strict_complete
    )
    bundles = build_bundles(events)
    del disclosures, events
    gc.collect()

    score_sessions = complete_sessions[
        (complete_sessions >= SCORE_START) & (complete_sessions <= SCORE_END)
    ]
    score_months = pd.PeriodIndex(
        score_sessions.to_period("M").unique()
    ).sort_values()
    expected_months = pd.PeriodIndex(
        ["2024-07", "2025-04", "2025-05", "2025-06", "2025-07"],
        freq="M",
    )
    if not score_months.equals(expected_months):
        raise RuntimeError(f"unexpected scored months: {list(score_months)}")

    score_parts: dict[str, list[pd.DataFrame]] = {
        candidate_id: [] for candidate_id in ALL_IDS
    }
    folds: list[dict[str, Any]] = []
    mutation_fold_digests: list[dict[str, Any]] = []

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
            raise RuntimeError(f"non-PIT panel fold: {period}")

        l4_model = fit_l4(panel_train, period)
        l4_prediction = np.asarray(l4_model.score(panel_score), dtype=float)
        score_parts["L4_price_control"].append(
            make_score_frame(
                panel_score,
                l4_prediction,
                np.ones(len(panel_score), dtype=bool),
                "L4_price_control",
            )
        )

        event_train = bundles.loc[bundles["date"].lt(month_start)].merge(
            panel_train[["date", "code", "name", "label", "oc_return_pct"]],
            on=["date", "code"],
            how="inner",
            suffixes=("", "_panel"),
            validate="one_to_one",
        )
        event_score = bundles.loc[bundles["date"].isin(month_dates)].merge(
            panel_score[["date", "code", "name", "label", "oc_return_pct"]],
            on=["date", "code"],
            how="inner",
            suffixes=("", "_panel"),
            validate="one_to_one",
        )
        event_train = event_train.sort_values(
            ["date", "code"], kind="stable"
        ).reset_index(drop=True)
        event_score = event_score.sort_values(
            ["date", "code"], kind="stable"
        ).reset_index(drop=True)
        if len(event_train) < protocol["sample_and_folds"][
            "minimum_training_bundles"
        ]:
            raise RuntimeError(f"too few event training bundles: {period}")
        if event_train["date"].max() >= month_start:
            raise RuntimeError(f"non-PIT event fold: {period}")

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
        lower = max(float(event_train["oc_return_pct"].quantile(0.01)), -10.0)
        upper = min(float(event_train["oc_return_pct"].quantile(0.99)), 10.0)
        clipped_target = event_train["oc_return_pct"].clip(
            lower, upper
        ).to_numpy(dtype=float)

        t02_model = Ridge(alpha=20.0, solver="lsqr")
        t02_model.fit(x_train, clipped_target)
        t02_prediction = np.asarray(t02_model.predict(x_score), dtype=float)
        score_parts["T02_char_value_event_only"].append(
            make_score_frame(
                event_score,
                t02_prediction,
                np.ones(len(event_score), dtype=bool),
                "T02_char_value_event_only",
            )
        )

        analog_prediction, analog_eligibility, support_report = (
            score_analog_family(
                x_train,
                x_score,
                event_train,
                event_score,
                clipped_target,
                session_positions,
            )
        )
        for candidate_id in ANALOG_IDS:
            score_parts[candidate_id].append(
                make_score_frame(
                    event_score,
                    analog_prediction[candidate_id],
                    analog_eligibility[candidate_id],
                    candidate_id,
                )
            )

        original_values = {
            "L4_price_control": l4_prediction,
            "T02_char_value_event_only": t02_prediction,
            **analog_prediction,
            **{
                f"{name}__eligible": values.astype(np.uint8)
                for name, values in analog_eligibility.items()
            },
        }
        original_digest = digest_arrays(original_values)
        mutated_panel_score = panel_score.copy()
        mutated_event_score = event_score.copy()
        mutated_panel_score["oc_return_pct"] = (
            np.arange(len(mutated_panel_score), dtype=float) * 1000.0 + 123.0
        )
        mutated_event_score["oc_return_pct"] = (
            np.arange(len(mutated_event_score), dtype=float) * -1000.0 - 321.0
        )
        mutation_l4 = np.asarray(l4_model.score(mutated_panel_score), dtype=float)
        mutation_t02 = np.asarray(t02_model.predict(x_score), dtype=float)
        mutation_analog, mutation_eligibility, _ = score_analog_family(
            x_train,
            x_score,
            event_train,
            mutated_event_score,
            clipped_target,
            session_positions,
        )
        mutated_values = {
            "L4_price_control": mutation_l4,
            "T02_char_value_event_only": mutation_t02,
            **mutation_analog,
            **{
                f"{name}__eligible": values.astype(np.uint8)
                for name, values in mutation_eligibility.items()
            },
        }
        mutated_digest = digest_arrays(mutated_values)
        if original_digest != mutated_digest:
            raise RuntimeError(f"target-session outcome mutation changed scores: {period}")
        mutation_fold_digests.append(
            {
                "period": str(period),
                "original_score_and_fill_digest": original_digest,
                "mutated_score_and_fill_digest": mutated_digest,
                "passes": True,
            }
        )

        folds.append(
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
                "value_target_clip": [lower, upper],
                "strictly_prior_training": bool(
                    panel_train["date"].max() < month_start
                    and event_train["date"].max() < month_start
                ),
                "analog_support": support_report,
            }
        )
        del (
            panel_train,
            panel_score,
            event_train,
            event_score,
            vectorizer,
            x_train,
            x_score,
            t02_model,
            l4_model,
            analog_prediction,
            analog_eligibility,
            mutation_analog,
            mutation_eligibility,
        )
        gc.collect()

    scores = {
        candidate_id: pd.concat(parts, ignore_index=True)
        for candidate_id, parts in score_parts.items()
    }
    slices = {
        name: (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
        for name, bounds in protocol["sample_and_folds"][
            "predeclared_slices"
        ].items()
    }
    all_picks: list[pd.DataFrame] = []
    metrics: dict[str, Any] = {}
    for candidate_id in ALL_IDS:
        metrics[candidate_id] = {}
        for top_k in (1, 2):
            picks = select_slots(scores[candidate_id], score_sessions, top_k)
            picks["top_k"] = top_k
            all_picks.append(picks)
            metrics[candidate_id][f"top{top_k}"] = metrics_for_picks(
                picks, top_k, slices
            )
    picks_frame = pd.concat(all_picks, ignore_index=True)
    picks_path = Path(f"{prefix}_picks.csv")
    picks_frame.to_csv(
        picks_path,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )

    score_missing = {
        date: value
        for date, value in missing_by_session.items()
        if SCORE_START <= pd.Timestamp(date) <= SCORE_END
    }
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "authority": {
            "classification": "retrospective_exploratory_no_production_promotion",
            "production_model_changed": False,
            "orders_allowed": False,
        },
        "protocol_sha256": integrity["protocol_sha256"],
        "runner_sha256": sha256_file(Path(__file__)),
        "integrity": {
            **integrity,
            "strictly_prior_training_all_folds": all(
                fold["strictly_prior_training"] for fold in folds
            ),
            "future_publication_violations": mapping_report[
                "future_publication_violations"
            ],
            "source_missing_never_encoded_no_event": True,
            "target_session_outcome_mutation": {
                "passes": all(row["passes"] for row in mutation_fold_digests),
                "folds": mutation_fold_digests,
            },
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
                bundles.loc[
                    bundles["date"].isin(score_sessions), "date"
                ].nunique()
            ),
            "historical_cache_observation_metadata_available": False,
        },
        "implementation": {
            "candidate_ids": list(ANALOG_IDS),
            "comparator_ids": list(COMPARATOR_IDS),
            "neighbor_count": NEIGHBORS,
            "time_half_life_sessions": TIME_HALF_LIFE_SESSIONS,
            "primary_cost_hurdle_pct": PRIMARY_COST_PCT,
            "family_features": list(FAMILY_FEATURES),
            "outcome_coefficients_fitted_for_analog_candidates": False,
            "hyperparameter_grid_searched": False,
        },
        "folds": folds,
        "metrics": metrics,
        "decision": {
            "status": "pending_independent_audit",
            "production_model_changed": False,
        },
        "artifacts": {
            "picks_csv": str(picks_path),
            "picks_sha256": sha256_file(picks_path),
        },
    }
    result_path = Path(f"{prefix}_result.json")
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": str(result_path),
                "picks": str(picks_path),
                "score_sessions": len(score_sessions),
                "score_months": [str(value) for value in score_months],
                "candidate_family_size": len(ANALOG_IDS),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
