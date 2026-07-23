#!/usr/bin/env python3
"""Run the preregistered exchange-calendar and disclosure-seasonality screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tse_session_ranker.data.tdnet import _title_flags, parse_tdnet_index  # noqa: E402


PROTOCOL_PATH = ROOT / "research" / "model_v11_calendar_protocol.json"
PROTOCOL_SHA_PATH = ROOT / "research" / "model_v11_calendar_protocol.sha256"
EXPECTED_PROTOCOL_SHA256 = (
    "23b4642e747e52c13ff62ffac36f438b4f88a622bb8cbb7323c7bd0495f58ddf"
)
PANEL_SHA256 = "6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb"
MANIFEST_SHA256 = "25e08c564ef6400b7a29168db9fd7e8220bd0e2386c7a3c7e71b05c2ff950b02"
SCORE_START = pd.Timestamp("2024-07-01")
SCORE_END = pd.Timestamp("2025-07-31")
BREAK_EVEN = 0.40
CAPACITIES = (1, 2)
COSTS = (20, 40, 60)
CACHE_PATHS = (ROOT / "tdnet_date_cache_2024", ROOT / "tdnet_date_cache")
HYPOTHESES = (
    "C01_WEEKDAY_ISSUER_EB",
    "C02_POST_HOLIDAY_ISSUER_EB",
    "C03_PRE_HOLIDAY_ISSUER_EB",
    "C04_MONTH_START_ISSUER_EB",
    "C05_MONTH_END_ISSUER_EB",
    "C06_TURN_MONTH_DIFFERENTIAL_EB",
    "C07_QUARTER_END_ISSUER_EB",
    "C08_FISCAL_HALF_YEAR_ISSUER_EB",
    "C09_SQ_PROXY_ISSUER_EB",
    "C10_EARNINGS_SEASON_TDNET_FAMILY",
    "C11_RELEASE_CLOCK_FAMILY_EB",
    "C12_HIGH_DENSITY_FAMILY_EB",
)
PERIODS = {
    "discovery": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-31")),
    "confirmation_a": (pd.Timestamp("2024-11-01"), pd.Timestamp("2025-03-31")),
    "confirmation_b": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-31")),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_digest(path: Path) -> tuple[int, str, set[pd.Timestamp]]:
    digest = hashlib.sha256()
    dates: set[pd.Timestamp] = set()
    files = sorted(path.glob("*.html"))
    for item in files:
        dates.add(pd.to_datetime(item.stem, format="%Y%m%d").normalize())
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
        digest.update(b"\0")
    return len(files), digest.hexdigest(), dates


def load_inputs() -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], set[pd.Timestamp]]:
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("calendar protocol changed after registration")
    if PROTOCOL_SHA_PATH.read_text(encoding="utf-8").strip().split() != [
        EXPECTED_PROTOCOL_SHA256,
        "model_v11_calendar_protocol.json",
    ]:
        raise RuntimeError("calendar protocol SHA sidecar is invalid")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != "model_v11_calendar_institution_zero_base_20260723":
        raise RuntimeError("unexpected calendar protocol")
    if tuple(item["id"] for item in protocol["registered_hypotheses"]) != HYPOTHESES:
        raise RuntimeError("registered calendar hypotheses changed")
    if protocol["candidate_family_size"] != 24:
        raise RuntimeError("calendar family size changed")
    if protocol["authority"]["production_promotion_allowed_from_this_run"]:
        raise RuntimeError("retrospective protocol cannot promote production")
    panel_spec = protocol["frozen_inputs"]["panel"]
    if sha256_file(panel_spec["path"]) != PANEL_SHA256:
        raise RuntimeError("frozen panel hash mismatch")
    if sha256_file(panel_spec["manifest_path"]) != MANIFEST_SHA256:
        raise RuntimeError("frozen manifest hash mismatch")
    panel = joblib.load(panel_spec["path"])
    manifest = json.loads(Path(panel_spec["manifest_path"]).read_text(encoding="utf-8"))
    cache_dates: set[pd.Timestamp] = set()
    for path, expected in zip(
        CACHE_PATHS, protocol["frozen_inputs"]["tdnet_caches"], strict=True
    ):
        count, digest, dates = cache_digest(path)
        if count != expected["html_files"] or digest != expected["aggregate_sha256"]:
            raise RuntimeError(f"TDnet cache hash mismatch: {path}")
        if cache_dates & dates:
            raise RuntimeError("duplicate cache dates")
        cache_dates |= dates
    return protocol, panel, manifest, cache_dates


def validate_panel(
    panel: pd.DataFrame, manifest: dict[str, Any], protocol: dict[str, Any]
) -> pd.DatetimeIndex:
    frozen = protocol["frozen_inputs"]["panel"]
    if len(panel) != frozen["rows"] or len(panel) != manifest["rows"]:
        raise RuntimeError("panel row count mismatch")
    if panel["code"].nunique() != frozen["codes"]:
        raise RuntimeError("panel code count mismatch")
    required = {
        "date",
        "code",
        "name",
        "oc_return_pct",
        "label",
        "price_eligible",
        "price_training_eligible",
        "candidate_price_source_max_date",
    }
    if required - set(panel.columns):
        raise RuntimeError("panel lacks required calendar columns")
    if panel.duplicated(["date", "code"]).any():
        raise RuntimeError("duplicate date/code rows")
    panel["date"] = pd.to_datetime(panel["date"], errors="raise").dt.normalize()
    panel["code"] = panel["code"].astype(str).str.zfill(4)
    sessions = pd.DatetimeIndex(panel["date"].drop_duplicates().sort_values())
    expected = pd.DatetimeIndex(pd.to_datetime(manifest["sessions"]))
    if not sessions.equals(expected):
        raise RuntimeError("session set differs from manifest")
    source = pd.to_datetime(panel["candidate_price_source_max_date"], errors="coerce")
    if (source.notna() & source.ge(panel["date"])).any():
        raise RuntimeError("calendar score has non-prior price source")
    observed = panel["oc_return_pct"].notna()
    if not np.array_equal(
        panel.loc[observed, "label"].to_numpy(float),
        panel.loc[observed, "oc_return_pct"].gt(0).to_numpy(float),
    ):
        raise RuntimeError("label is not exact return sign")
    return sessions


def strict_sessions(
    sessions: pd.DatetimeIndex, cache_dates: set[pd.Timestamp]
) -> tuple[pd.DatetimeIndex, dict[str, list[str]]]:
    accepted: list[pd.Timestamp] = []
    missing: dict[str, list[str]] = {}
    for position, date in enumerate(sessions):
        if position == 0:
            missing[str(date.date())] = ["prior panel session unavailable"]
            continue
        prior = sessions[position - 1]
        absent = [
            str(value.date())
            for value in pd.date_range(prior, date, freq="D")
            if value not in cache_dates
        ]
        if absent:
            missing[str(date.date())] = absent
        else:
            accepted.append(date)
    return pd.DatetimeIndex(accepted), missing


def annotate_calendar(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame({"date": sessions})
    frame["weekday"] = frame["date"].dt.weekday
    frame["previous_gap_days"] = frame["date"].diff().dt.days
    frame["next_gap_days"] = frame["date"].shift(-1).sub(frame["date"]).dt.days
    frame["post_holiday"] = frame["previous_gap_days"].ge(3)
    frame["pre_holiday"] = frame["next_gap_days"].ge(3)
    month = frame["date"].dt.to_period("M")
    frame["month_position"] = frame.groupby(month, sort=False).cumcount() + 1
    frame["month_size"] = frame.groupby(month, sort=False)["date"].transform("size")
    frame["month_reverse_position"] = frame["month_size"] - frame["month_position"] + 1
    frame["month_start"] = frame["month_position"].le(3)
    frame["month_end"] = frame["month_reverse_position"].le(3)
    frame["turn_month"] = frame["month_position"].le(3) | frame[
        "month_reverse_position"
    ].le(2)
    frame["quarter_end"] = (
        frame["date"].dt.month.isin([3, 6, 9, 12])
        & frame["month_reverse_position"].le(5)
    )
    frame["fiscal_half_year_end"] = (
        frame["date"].dt.month.isin([3, 9])
        & frame["month_reverse_position"].le(5)
    )
    frame["earnings_season"] = (
        frame["date"].dt.month.isin([1, 4, 7, 10])
        & frame["month_reverse_position"].le(5)
    ) | (
        frame["date"].dt.month.isin([2, 5, 8, 11])
        & frame["month_position"].le(10)
    )
    sq_dates: set[pd.Timestamp] = set()
    for period, group in frame.groupby(month, sort=False):
        first = pd.Timestamp(period.start_time)
        offset = (4 - first.weekday()) % 7
        second_friday = first + pd.Timedelta(days=offset + 7)
        monday = second_friday - pd.Timedelta(days=4)
        candidates = group.loc[
            group["date"].between(monday, second_friday), "date"
        ]
        if len(candidates):
            sq_dates.add(candidates.max())
    frame["sq_proxy"] = frame["date"].isin(sq_dates)
    if frame["date"].duplicated().any():
        raise RuntimeError("calendar annotation duplicated a session")
    return frame


def parse_event_bundles(
    sessions: pd.DatetimeIndex, strict: pd.DatetimeIndex
) -> tuple[pd.DataFrame, dict[str, Any]]:
    disclosures = pd.concat(
        [
            parse_tdnet_index(path.read_bytes(), pd.to_datetime(path.stem, format="%Y%m%d"))
            for directory in CACHE_PATHS
            for path in sorted(directory.glob("*.html"))
        ],
        ignore_index=True,
    ).sort_values(["published_at", "code", "title"], kind="stable")
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
    target_position = cutoffs.searchsorted(published, side="left")
    in_range = target_position < len(sessions)
    events = disclosures.loc[in_range].copy().reset_index(drop=True)
    target_position = target_position[in_range]
    events["date"] = sessions[target_position].to_numpy()
    same_day = events["index_date"].eq(events["date"])
    source_is_session = events["index_date"].isin(sessions)
    minute = events["published_at"].dt.hour * 60 + events["published_at"].dt.minute
    close_minute = np.where(
        events["index_date"].lt(pd.Timestamp("2024-11-05")),
        15 * 60,
        15 * 60 + 30,
    )
    intraday = source_is_session & ~same_day & minute.lt(close_minute)
    events = events.loc[~intraday & events["date"].isin(strict)].reset_index(drop=True)
    flags = _title_flags(events["title"]).reset_index(drop=True)
    events = pd.concat([events, flags], axis=1)
    grouped = events.groupby(["date", "code"], sort=True)
    bundles = grouped.agg(
        name=("name", "last"),
        first_published=("published_at", "min"),
        last_published=("published_at", "max"),
        last_index_date=("index_date", "max"),
        document_count=("title", "size"),
    ).reset_index()
    flag_names = (
        "earnings",
        "revision",
        "revision_up",
        "revision_down",
        "dividend",
        "dividend_up",
        "dividend_down",
        "buyback_decision",
        "equity_financing",
        "benefit",
        "split",
        "ma_alliance",
        "control_transaction",
        "impairment_loss",
        "audit_problem",
        "correction",
    )
    bundles = bundles.merge(
        grouped[list(flag_names)].max().reset_index(),
        on=["date", "code"],
        how="left",
        validate="one_to_one",
    )

    family = np.full(len(bundles), "other", dtype=object)
    assignments = [
        (
            "ma_or_alliance",
            bundles[["ma_alliance", "control_transaction"]].max(axis=1).gt(0),
        ),
        ("benefit_or_split", bundles[["benefit", "split"]].max(axis=1).gt(0)),
        ("dividend", bundles["dividend"].gt(0)),
        ("earnings", bundles["earnings"].gt(0)),
        ("revision", bundles["revision"].gt(0)),
        ("buyback", bundles["buyback_decision"].gt(0)),
        (
            "adverse",
            bundles[
                [
                    "revision_down",
                    "dividend_down",
                    "equity_financing",
                    "impairment_loss",
                    "audit_problem",
                    "correction",
                ]
            ]
            .max(axis=1)
            .gt(0),
        ),
    ]
    for name, mask in assignments:
        family[mask.to_numpy()] = name
    bundles["event_family"] = family

    session_position = {date: position for position, date in enumerate(sessions)}
    prior_session = {
        date: (sessions[position - 1] if position > 0 else pd.NaT)
        for date, position in session_position.items()
    }
    release: list[str] = []
    for row in bundles.itertuples(index=False):
        last = pd.Timestamp(row.last_published)
        if last.tz_convert("Asia/Tokyo").date() == row.date.date():
            release.append("same_day_premarket")
            continue
        prior = prior_session[row.date]
        if pd.isna(prior) or last.tz_localize(None).normalize() != prior:
            release.append("late_weekend_or_older")
            continue
        release_minute = last.hour * 60 + last.minute
        if release_minute < 16 * 60 + 30:
            release.append("close_to_1629")
        elif release_minute < 20 * 60:
            release.append("1630_to_1959")
        else:
            release.append("late_weekend_or_older")
    bundles["release_clock"] = release
    density = bundles.groupby("date", sort=True)["code"].size()
    bundles["bundle_density"] = bundles["date"].map(density).astype(int)
    return bundles, {
        "parsed_disclosures": int(len(disclosures)),
        "qualifying_disclosures": int(len(events)),
        "bundles": int(len(bundles)),
        "strict_event_sessions": int(bundles["date"].nunique()),
    }


def add_density_state(
    calendar: pd.DataFrame, bundles: pd.DataFrame, strict: pd.DatetimeIndex
) -> pd.DataFrame:
    density = bundles.groupby("date", sort=True)["code"].size()
    output = calendar.copy()
    output["bundle_density"] = output["date"].map(density)
    output.loc[output["date"].isin(strict), "bundle_density"] = output.loc[
        output["date"].isin(strict), "bundle_density"
    ].fillna(0)
    thresholds: list[float] = []
    history: list[float] = []
    for row in output.itertuples(index=False):
        if row.date not in strict:
            thresholds.append(np.nan)
            continue
        thresholds.append(
            np.nan
            if not history
            else float(np.quantile(history, 0.75, method="higher"))
        )
        history.append(float(row.bundle_density))
    output["density_threshold_prior"] = thresholds
    output["high_density"] = (
        output["bundle_density"].notna()
        & output["density_threshold_prior"].notna()
        & output["bundle_density"].ge(output["density_threshold_prior"])
    )
    return output


def posterior_scores(
    training: pd.DataFrame,
    state: str,
    *,
    prior_strength: float = 20.0,
) -> tuple[dict[str, float], float, int]:
    active = training.loc[training[state].eq(True)].copy()
    y = np.clip(active["oc_return_pct"].to_numpy(float), -5.0, 5.0)
    global_mean = 0.0 if not len(active) else float(y.mean())
    if not len(active):
        return {}, global_mean, 0
    active["_y"] = y
    grouped = active.groupby("code", sort=False)["_y"].agg(["sum", "count"])
    value = (
        grouped["sum"] + prior_strength * global_mean
    ) / (grouped["count"] + prior_strength)
    return value.to_dict(), global_mean, int(len(active))


def weekday_scores(
    training: pd.DataFrame,
) -> tuple[dict[int, dict[str, float]], dict[int, float]]:
    scores: dict[int, dict[str, float]] = {}
    priors: dict[int, float] = {}
    for weekday in range(5):
        subset = training.loc[training["weekday"].eq(weekday)].copy()
        y = np.clip(subset["oc_return_pct"].to_numpy(float), -5.0, 5.0)
        prior = 0.0 if not len(subset) else float(y.mean())
        subset["_y"] = y
        grouped = subset.groupby("code", sort=False)["_y"].agg(["sum", "count"])
        value = (grouped["sum"] + 20.0 * prior) / (grouped["count"] + 20.0)
        scores[weekday] = value.to_dict()
        priors[weekday] = prior
    return scores, priors


def turn_differential(training: pd.DataFrame) -> tuple[dict[str, float], float]:
    work = training.copy()
    work["_y"] = np.clip(work["oc_return_pct"].to_numpy(float), -5.0, 5.0)
    turn = work.loc[work["turn_month"]].groupby("code")["_y"].agg(["sum", "count"])
    other = work.loc[~work["turn_month"]].groupby("code")["_y"].agg(["sum", "count"])
    joined = turn.join(other, how="outer", lsuffix="_turn", rsuffix="_other").fillna(0.0)
    mean_turn = joined["sum_turn"] / joined["count_turn"].replace(0, np.nan)
    mean_other = joined["sum_other"] / joined["count_other"].replace(0, np.nan)
    differential = (mean_turn - mean_other).fillna(0.0)
    effective_n = np.minimum(joined["count_turn"], joined["count_other"])
    shrunk = differential * effective_n / (effective_n + 30.0)
    global_turn = (
        0.0
        if not work["turn_month"].any()
        else float(work.loc[work["turn_month"], "_y"].mean())
    )
    return (shrunk + global_turn).to_dict(), global_turn


def family_hierarchy(
    training_events: pd.DataFrame,
    state: str | None,
    *,
    issuer: bool,
    cell: str = "event_family",
) -> tuple[dict[Any, float], dict[str, float], dict[Any, int]]:
    active = (
        training_events
        if state is None
        else training_events.loc[training_events[state].eq(True)]
    ).copy()
    active["_y"] = np.clip(active["oc_return_pct"].to_numpy(float), -5.0, 5.0)
    family_stats = active.groupby("event_family", sort=False)["_y"].agg(
        ["sum", "count"]
    )
    family_mean = (
        family_stats["sum"] / family_stats["count"]
    ).to_dict()
    if issuer:
        stats = active.groupby(["code", "event_family"], sort=False)["_y"].agg(
            ["sum", "count"]
        )
        values: dict[Any, float] = {}
        counts: dict[Any, int] = {}
        for key, row in stats.iterrows():
            prior = family_mean.get(key[1], 0.0)
            values[key] = float((row["sum"] + 20.0 * prior) / (row["count"] + 20.0))
            counts[key] = int(row["count"])
        return values, family_mean, counts
    stats = active.groupby(["event_family", cell], sort=False)["_y"].agg(
        ["sum", "count"]
    )
    values = {}
    counts = {}
    for key, row in stats.iterrows():
        prior = family_mean.get(key[0], 0.0)
        values[key] = float((row["sum"] + 50.0 * prior) / (row["count"] + 50.0))
        counts[key] = int(row["count"])
    return values, family_mean, counts


PredictionMap = dict[str, tuple[np.ndarray, np.ndarray]]
Predictor = Callable[[pd.DataFrame], PredictionMap]


def fit_fold(
    training: pd.DataFrame, training_events: pd.DataFrame
) -> tuple[Predictor, dict[str, Any]]:
    weekday_map, weekday_prior = weekday_scores(training)
    posterior_definitions = {
        HYPOTHESES[1]: "post_holiday",
        HYPOTHESES[2]: "pre_holiday",
        HYPOTHESES[3]: "month_start",
        HYPOTHESES[4]: "month_end",
        HYPOTHESES[6]: "quarter_end",
        HYPOTHESES[7]: "fiscal_half_year_end",
        HYPOTHESES[8]: "sq_proxy",
    }
    state_maps: dict[str, tuple[dict[str, float], float, int]] = {
        name: posterior_scores(training, state)
        for name, state in posterior_definitions.items()
    }
    turn_map, turn_prior = turn_differential(training)
    season_map, season_family, season_counts = family_hierarchy(
        training_events, "earnings_season", issuer=True
    )
    release_map, release_family, release_counts = family_hierarchy(
        training_events, None, issuer=False, cell="release_clock"
    )
    density_map, density_family, density_counts = family_hierarchy(
        training_events, "high_density", issuer=False, cell="high_density"
    )

    def predict(scoring: pd.DataFrame) -> PredictionMap:
        result: PredictionMap = {}
        weekday_value = np.asarray(
            [
                weekday_map.get(int(row.weekday), {}).get(
                    row.code, weekday_prior.get(int(row.weekday), 0.0)
                )
                for row in scoring.itertuples(index=False)
            ],
            dtype=float,
        )
        result[HYPOTHESES[0]] = (
            weekday_value,
            np.ones(len(scoring), dtype=bool),
        )
        for name, state in posterior_definitions.items():
            value_map, prior, _ = state_maps[name]
            value = scoring["code"].map(value_map).fillna(prior).to_numpy(float)
            result[name] = (value, scoring[state].eq(True).to_numpy())
        value = scoring["code"].map(turn_map).fillna(turn_prior).to_numpy(float)
        result[HYPOTHESES[5]] = (
            value,
            scoring["turn_month"].eq(True).to_numpy(),
        )

        event = scoring["event_any"].eq(1)
        season_allowed = (
            event
            & scoring["tdnet_strict"].eq(True)
            & scoring["earnings_season"].eq(True)
            & scoring["event_family"].map(
                lambda value: season_family.get(value) is not None
            )
        )
        season_value = np.asarray(
            [
                season_map.get(
                    (row.code, row.event_family),
                    season_family.get(row.event_family, 0.0),
                )
                if season_counts.get((row.code, row.event_family), 0) >= 0
                else 0.0
                for row in scoring.itertuples(index=False)
            ],
            dtype=float,
        )
        family_total = training_events.loc[
            training_events["earnings_season"]
        ].groupby("event_family").size()
        season_allowed &= scoring["event_family"].map(family_total).fillna(0).ge(30)
        result[HYPOTHESES[9]] = (season_value, season_allowed.to_numpy())

        release_keys = list(
            zip(scoring["event_family"], scoring["release_clock"], strict=True)
        )
        release_value = np.asarray(
            [
                release_map.get(key, release_family.get(key[0], 0.0))
                for key in release_keys
            ],
            dtype=float,
        )
        release_allowed = (
            event
            & scoring["tdnet_strict"].eq(True)
            & pd.Series(release_keys, index=scoring.index)
            .map(release_counts)
            .fillna(0)
            .ge(30)
        )
        result[HYPOTHESES[10]] = (
            release_value,
            release_allowed.to_numpy(),
        )

        density_keys = list(
            zip(scoring["event_family"], scoring["high_density"], strict=True)
        )
        density_value = np.asarray(
            [
                density_map.get(key, density_family.get(key[0], 0.0))
                for key in density_keys
            ],
            dtype=float,
        )
        density_allowed = (
            event
            & scoring["tdnet_strict"].eq(True)
            & scoring["high_density"].eq(True)
            & pd.Series(density_keys, index=scoring.index)
            .map(density_counts)
            .fillna(0)
            .ge(30)
        )
        result[HYPOTHESES[11]] = (
            density_value,
            density_allowed.to_numpy(),
        )
        return result

    return predict, {
        "training_rows": int(len(training)),
        "training_sessions": int(training["date"].nunique()),
        "training_event_rows": int(len(training_events)),
        "state_rows": {
            name: value[2] for name, value in state_maps.items()
        },
        "season_family_rows": {
            str(key): int(value)
            for key, value in training_events.loc[
                training_events["earnings_season"]
            ].groupby("event_family").size().items()
        },
        "release_cells": len(release_map),
        "density_cells": len(density_map),
    }


def prediction_digest(predictions: PredictionMap) -> str:
    digest = hashlib.sha256()
    for name in sorted(predictions):
        score, allowed = predictions[name]
        digest.update(name.encode("ascii"))
        digest.update(np.ascontiguousarray(score).tobytes())
        digest.update(np.ascontiguousarray(allowed).tobytes())
    return digest.hexdigest()


def selected_keys(
    scoring: pd.DataFrame,
    score: np.ndarray,
    allowed: np.ndarray,
    capacity: int,
) -> tuple[tuple[str, str], ...]:
    eligible = allowed & np.isfinite(score) & (score > BREAK_EVEN)
    work = scoring.loc[eligible, ["date", "code"]].copy()
    work["score"] = score[eligible]
    work = work.sort_values(
        ["date", "score", "code"],
        ascending=[True, False, True],
        kind="stable",
    ).groupby("date", sort=False).head(capacity)
    return tuple(
        (str(date.date()), code)
        for date, code in work[["date", "code"]].itertuples(index=False)
    )


def make_picks(
    scoring: pd.DataFrame,
    predictions: PredictionMap,
    sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for hypothesis in HYPOTHESES:
        score, allowed = predictions[hypothesis]
        eligible = allowed & np.isfinite(score) & (score > BREAK_EVEN)
        base = scoring.loc[
            eligible, ["date", "code", "name", "oc_return_pct", "label"]
        ].copy()
        base["posterior_score_pct"] = score[eligible]
        base = base.sort_values(
            ["date", "posterior_score_pct", "code"],
            ascending=[True, False, True],
            kind="stable",
        )
        for capacity in CAPACITIES:
            selected = base.groupby("date", sort=False).head(capacity).copy()
            selected["slot"] = selected.groupby("date", sort=False).cumcount() + 1
            lookup = {
                (row.date, int(row.slot)): row
                for row in selected.itertuples(index=False)
            }
            for date in sessions:
                for slot in range(1, capacity + 1):
                    row = lookup.get((date, slot))
                    rows.append(
                        {
                            "date": str(date.date()),
                            "policy_id": f"{hypothesis}_K{capacity}",
                            "hypothesis_id": hypothesis,
                            "capacity": capacity,
                            "slot": slot,
                            "code": None if row is None else row.code,
                            "name": None if row is None else row.name,
                            "posterior_score_pct": (
                                None if row is None else float(row.posterior_score_pct)
                            ),
                            "oc_return_pct": (
                                None if row is None else float(row.oc_return_pct)
                            ),
                            "label": None if row is None else float(row.label),
                        }
                    )
    return pd.DataFrame(rows)


def daily_returns(
    picks: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    cost_bps: int,
) -> pd.DataFrame:
    capacity = int(picks["capacity"].iloc[0])
    executed = picks["code"].notna().astype(float)
    gross = pd.to_numeric(picks["oc_return_pct"], errors="coerce").fillna(0.0) / capacity
    value = pd.DataFrame(
        {
            "date": pd.to_datetime(picks["date"]),
            "gross": gross,
            "net": gross - executed * (cost_bps / 100.0) / capacity,
            "exposure": executed / capacity,
        }
    ).groupby("date", sort=True).sum()
    return value.reindex(sessions, fill_value=0.0)


def finite_mean(values: pd.Series | np.ndarray) -> float | None:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    return None if not len(clean) else float(clean.mean())


def policy_metrics(
    picks: pd.DataFrame, sessions: pd.DatetimeIndex
) -> dict[str, Any]:
    capacity = int(picks["capacity"].iloc[0])
    daily = {cost: daily_returns(picks, sessions, cost) for cost in COSTS}
    gross = daily[40]["gross"]
    net20, net40, net60 = (daily[cost]["net"] for cost in COSTS)
    exposure = daily[40]["exposure"]
    executed = picks.loc[picks["code"].notna()].copy()
    monthly = net40.groupby(net40.index.to_period("M")).mean()
    slices = {
        name: float(net40.loc[(net40.index >= start) & (net40.index <= end)].mean())
        for name, (start, end) in PERIODS.items()
    }
    best_removed: dict[str, float | None] = {}
    worst_removed: dict[str, float | None] = {}
    for count in (5, 10, 20):
        keep = max(len(net20) - count, 0)
        best_removed[str(count)] = finite_mean(net20.nsmallest(keep))
        worst_removed[str(count)] = finite_mean(net20.nlargest(keep))
    if len(executed):
        code_weight = executed.groupby("code").size() / capacity
        shares = code_weight / code_weight.sum()
        largest_code = float(shares.max())
        top10_code = float(shares.nlargest(10).sum())
        slot_net20 = (
            pd.to_numeric(executed["oc_return_pct"], errors="raise") - 0.20
        ) / capacity
        positive = slot_net20.groupby(executed["code"]).sum()
        positive = positive[positive > 0].sort_values(ascending=False)
        positive_total = float(positive.sum())
        largest_positive = (
            None if positive_total <= 0 else float(positive.iloc[0] / positive_total)
        )
        kept = executed.loc[~executed["code"].isin(positive.head(10).index)]
        cash = (
            (pd.to_numeric(kept["oc_return_pct"], errors="raise") - 0.20)
            .div(capacity)
            .groupby(pd.to_datetime(kept["date"]))
            .sum()
            .reindex(sessions, fill_value=0.0)
        )
        top10_cash = float(cash.mean())
    else:
        largest_code = top10_code = 0.0
        largest_positive = None
        top10_cash = 0.0
    tail_n = max(1, int(math.ceil(len(net40) * 0.05)))
    return {
        "scheduled_days": int(len(sessions)),
        "gross_mean_pct": float(gross.mean()),
        "net20_mean_pct": float(net20.mean()),
        "net40_mean_pct": float(net40.mean()),
        "net60_mean_pct": float(net60.mean()),
        "daily_gross_win_rate": float((gross > 0).mean()),
        "executed_slots": int(len(executed)),
        "executed_slot_fraction": float(len(executed) / (len(sessions) * capacity)),
        "traded_days": int((exposure > 0).sum()),
        "cash_days": int((exposure == 0).sum()),
        "unique_codes": int(executed["code"].nunique()),
        "monthly_net40_pct": {
            str(period): float(value) for period, value in monthly.items()
        },
        "positive_months_net40": int((monthly > 0).sum()),
        "temporal_slices_net40_pct": slices,
        "best_days_removed_net20_pct": best_removed,
        "worst_days_removed_net20_pct": worst_removed,
        "expected_shortfall05_net40_pct": float(net40.nsmallest(tail_n).mean()),
        "largest_code_weight_share": largest_code,
        "top10_code_weight_share": top10_code,
        "largest_positive_code_pnl_share": largest_positive,
        "top10_positive_pnl_codes_to_cash_net20_pct": top10_cash,
        "daily_net40_pct": [float(value) for value in net40],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT / "research" / "model_v11_calendar_result.json",
    )
    parser.add_argument(
        "--picks",
        type=Path,
        default=ROOT / "research" / "model_v11_calendar_picks.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol, panel, manifest, cache_dates = load_inputs()
    sessions = validate_panel(panel, manifest, protocol)
    strict, rejected = strict_sessions(sessions, cache_dates)
    bundles, event_coverage = parse_event_bundles(sessions, strict)
    calendar = add_density_state(annotate_calendar(sessions), bundles, strict)
    panel = panel.merge(calendar, on="date", how="left", validate="many_to_one")
    panel["tdnet_strict"] = panel["date"].isin(strict)
    bundle_columns = [
        "date",
        "code",
        "event_family",
        "release_clock",
        "bundle_density",
    ]
    panel = panel.merge(
        bundles[bundle_columns],
        on=["date", "code"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_event"),
    )
    panel["event_any"] = np.where(
        panel["tdnet_strict"], panel["event_family"].notna().astype(float), np.nan
    )
    panel.loc[
        panel["tdnet_strict"] & panel["event_family"].isna(), "event_family"
    ] = "none"
    panel.loc[
        panel["tdnet_strict"] & panel["release_clock"].isna(), "release_clock"
    ] = "none"
    panel["bundle_density"] = panel["bundle_density"].fillna(
        panel["bundle_density_event"]
    )
    panel = panel.drop(columns=["bundle_density_event"])
    score_sessions = sessions[(sessions >= SCORE_START) & (sessions <= SCORE_END)]
    if len(score_sessions) != 266:
        raise RuntimeError("calendar score-session count differs from protocol")

    fold_picks: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    mutations: list[dict[str, Any]] = []
    score_periods = score_sessions.to_period("M").drop_duplicates()
    for period in score_periods:
        period_sessions = score_sessions[score_sessions.to_period("M") == period]
        fold_start = period_sessions.min()
        training = panel.loc[
            panel["date"].lt(fold_start)
            & panel["price_training_eligible"].fillna(False)
            & panel["oc_return_pct"].notna()
        ].sort_values(["date", "code"], kind="stable").reset_index(drop=True)
        scoring = panel.loc[
            panel["date"].isin(period_sessions)
            & panel["price_eligible"].fillna(False)
        ].sort_values(["date", "code"], kind="stable").reset_index(drop=True)
        training_events = training.loc[
            training["tdnet_strict"]
            & training["event_any"].eq(1)
        ].reset_index(drop=True)
        if training["date"].max() >= fold_start:
            raise RuntimeError("calendar fold is not strictly prior")
        predictor, detail = fit_fold(training, training_events)
        predictions = predictor(scoring)
        mutated = scoring.copy()
        mutated["oc_return_pct"] = np.linspace(-99.0, 99.0, len(mutated))
        mutated["label"] = 1.0 - pd.to_numeric(
            mutated["label"], errors="coerce"
        ).fillna(0.0)
        mutated_predictions = predictor(mutated)
        digest = prediction_digest(predictions)
        mutated_digest = prediction_digest(mutated_predictions)
        keys_equal = all(
            selected_keys(scoring, *predictions[name], capacity)
            == selected_keys(mutated, *mutated_predictions[name], capacity)
            for name in HYPOTHESES
            for capacity in CAPACITIES
        )
        if digest != mutated_digest or not keys_equal:
            raise RuntimeError(f"calendar mutation changed score: {period}")
        fold_picks.append(make_picks(scoring, predictions, period_sessions))
        folds.append(
            {
                "period": str(period),
                "score_days": int(len(period_sessions)),
                "score_rows": int(len(scoring)),
                **detail,
            }
        )
        mutations.append(
            {
                "period": str(period),
                "prediction_digest": digest,
                "mutated_prediction_digest": mutated_digest,
                "selected_keys_equal": keys_equal,
                "passes": digest == mutated_digest and keys_equal,
            }
        )
    picks = pd.concat(fold_picks, ignore_index=True).sort_values(
        ["policy_id", "date", "slot"], kind="stable"
    ).reset_index(drop=True)
    picks.to_csv(args.picks, index=False, lineterminator="\n")
    metrics = {
        str(policy): policy_metrics(group.reset_index(drop=True), score_sessions)
        for policy, group in picks.groupby("policy_id", sort=True)
    }
    best = max(metrics, key=lambda name: metrics[name]["net40_mean_pct"])
    result = {
        "schema_version": 1,
        "experiment_id": "model_v11_calendar_institution_zero_base_20260723",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "input_hashes": {
            "panel_sha256": PANEL_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "cache_2024_sha256": protocol["frozen_inputs"]["tdnet_caches"][0][
                "aggregate_sha256"
            ],
            "cache_2025_sha256": protocol["frozen_inputs"]["tdnet_caches"][1][
                "aggregate_sha256"
            ],
        },
        "coverage": {
            "panel_sessions": int(len(sessions)),
            "score_sessions": int(len(score_sessions)),
            "strict_tdnet_sessions": int(len(strict)),
            "strict_tdnet_score_sessions": int(
                score_sessions.isin(strict).sum()
            ),
            "tdnet_missing_score_sessions": int(
                (~score_sessions.isin(strict)).sum()
            ),
            "tdnet_missing_examples": {
                key: value
                for key, value in rejected.items()
                if SCORE_START <= pd.Timestamp(key) <= SCORE_END
            },
            **event_coverage,
        },
        "folds": folds,
        "integrity": {
            "monthly_expanding_strict_prior": True,
            "future_source_violations": 0,
            "target_session_outcome_mutation": {
                "passes": all(value["passes"] for value in mutations),
                "folds": mutations,
            },
        },
        "metrics": metrics,
        "point_estimate_best_net40_policy": best,
        "point_estimate_best_net40_pct": metrics[best]["net40_mean_pct"],
        "decision": {
            "status": "pending_independent_audit",
            "retrospective_candidate": None,
            "production_ready": False,
        },
    }
    args.result.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": str(args.result),
                "picks": str(args.picks),
                "best": best,
                "best_net40": metrics[best]["net40_mean_pct"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
