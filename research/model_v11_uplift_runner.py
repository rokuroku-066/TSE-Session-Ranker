#!/usr/bin/env python3
"""Run the preregistered v1.1 causal/uplift mechanism screen.

The treatment is the presence of a pre-open TDnet disclosure bundle.  The
panel has already been viewed by the wider project, so this experiment may
falsify mechanisms or freeze a forward-shadow candidate, but can never
promote a production model.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tse_session_ranker.data.tdnet import _title_flags, parse_tdnet_index  # noqa: E402


PROTOCOL_PATH = ROOT / "research" / "model_v11_uplift_protocol.json"
PROTOCOL_SHA_PATH = ROOT / "research" / "model_v11_uplift_protocol.sha256"
EXPECTED_PROTOCOL_SHA256 = (
    "611f9d16467fee02d5a17e415d0a0a6490917310509174ed43379c60d04c69cb"
)
PANEL_SHA256 = "6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb"
MANIFEST_SHA256 = "25e08c564ef6400b7a29168db9fd7e8220bd0e2386c7a3c7e71b05c2ff950b02"
SEED = 20260723
SCORE_START = pd.Timestamp("2024-07-01")
SCORE_END = pd.Timestamp("2025-07-31")
BREAK_EVEN_PCT = 0.40
COSTS_BPS = (20, 40, 60)
CAPACITIES = (1, 2)

HYPOTHESES = (
    "U01_MATCHED_NO_EVENT",
    "U02_FAMILY_T_LEARNER",
    "U03_AIPW_CATE",
    "U04_DATE_RESIDUAL_UPLIFT",
    "U05_ISSUER_SELF_CONTROL",
    "U06_PROPENSITY_OVERLAP_T",
    "U07_POSITIVE_TAIL_UPLIFT",
    "U08_HONEST_CROSSFIT_POLICY_TREE",
)

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
EVENT_FEATURES = (
    "tdnet_clean_has_earnings",
    "tdnet_clean_has_revision",
    "tdnet_clean_revision_up_title",
    "tdnet_clean_revision_down_title",
    "tdnet_clean_has_dividend",
    "tdnet_clean_dividend_up_title",
    "tdnet_clean_dividend_down_title",
    "tdnet_clean_has_buyback_decision",
    "tdnet_clean_has_buyback_tostnet",
    "tdnet_clean_has_external_equity_financing",
    "tdnet_clean_has_benefit",
    "tdnet_clean_has_split",
    "tdnet_clean_has_ma_transaction",
    "tdnet_clean_has_business_alliance",
    "tdnet_clean_has_control_transaction",
    "tdnet_clean_has_impairment_loss",
    "tdnet_clean_has_audit_problem",
    "tdnet_clean_has_correction",
    "tdnet_clean_document_count_log1p",
    "tdnet_clean_family_count_log1p",
    "tdnet_clean_latest_age_hours_log1p",
    "tdnet_clean_support_adverse_conflict",
)
FULL_FEATURES = (*BASE_FEATURES, *EVENT_FEATURES)
MATCH_FEATURES = (
    "overnight_last",
    "oc_mean_20",
    "oc_std_20",
    "xrank_close_momentum_20",
    "xrank_prior_close_location_20",
)
CACHE_PATHS = (ROOT / "tdnet_date_cache_2024", ROOT / "tdnet_date_cache")
PERIODS = {
    "early": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-07-31")),
    "middle": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-05-31")),
    "late": (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-31")),
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
        date = pd.to_datetime(item.stem, format="%Y%m%d").normalize()
        dates.add(date)
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
        digest.update(b"\0")
    return len(files), digest.hexdigest(), dates


def load_inputs() -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], set[pd.Timestamp]]:
    protocol_sha = sha256_file(PROTOCOL_PATH)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("uplift protocol changed after registration")
    sidecar = PROTOCOL_SHA_PATH.read_text(encoding="utf-8").strip().split()
    if sidecar != [EXPECTED_PROTOCOL_SHA256, "model_v11_uplift_protocol.json"]:
        raise RuntimeError("uplift protocol SHA sidecar is invalid")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != "model_v11_uplift_zero_base_20260723":
        raise RuntimeError("unexpected uplift protocol id")
    registered = tuple(item["id"] for item in protocol["registered_hypotheses"])
    if registered != HYPOTHESES or protocol["candidate_family_size"] != 16:
        raise RuntimeError("registered uplift candidate family changed")
    if protocol["authority"]["production_promotion_allowed_from_this_run"]:
        raise RuntimeError("retrospective run cannot promote production")

    panel_path = Path(protocol["frozen_inputs"]["panel"]["path"])
    manifest_path = Path(protocol["frozen_inputs"]["panel"]["manifest_path"])
    if sha256_file(panel_path) != PANEL_SHA256:
        raise RuntimeError("frozen panel hash mismatch")
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("frozen panel manifest hash mismatch")
    panel = joblib.load(panel_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    cache_dates: set[pd.Timestamp] = set()
    cache_checks: list[dict[str, Any]] = []
    for path, frozen in zip(
        CACHE_PATHS, protocol["frozen_inputs"]["tdnet_caches"], strict=True
    ):
        count, digest, dates = cache_digest(path)
        if count != frozen["html_files"] or digest != frozen["aggregate_sha256"]:
            raise RuntimeError(f"TDnet cache changed: {path}")
        if cache_dates & dates:
            raise RuntimeError("TDnet caches contain duplicate dates")
        cache_dates |= dates
        cache_checks.append(
            {
                "path": str(path.relative_to(ROOT)),
                "html_files": count,
                "aggregate_sha256": digest,
            }
        )
    return protocol, panel, manifest, cache_dates


def validate_panel(
    panel: pd.DataFrame,
    manifest: dict[str, Any],
    protocol: dict[str, Any],
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
        "label",
        "oc_return_pct",
        "price_eligible",
        "price_training_eligible",
        "candidate_price_source_max_date",
        "tdnet_clean_feature_source_max_timestamp",
        "tdnet_clean_any",
        *FULL_FEATURES,
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise RuntimeError(f"panel lacks uplift columns: {missing}")
    if panel.duplicated(["date", "code"]).any():
        raise RuntimeError("duplicate panel date/code")
    panel["date"] = pd.to_datetime(panel["date"], errors="raise").dt.normalize()
    panel["code"] = panel["code"].astype(str).str.zfill(4)
    sessions = pd.DatetimeIndex(panel["date"].drop_duplicates().sort_values())
    expected_sessions = pd.DatetimeIndex(pd.to_datetime(manifest["sessions"]))
    if not sessions.equals(expected_sessions):
        raise RuntimeError("panel session set differs from manifest")
    source = pd.to_datetime(panel["candidate_price_source_max_date"], errors="coerce")
    if (source.notna() & source.ge(panel["date"])).any():
        raise RuntimeError("price feature source is not strictly prior")
    tdnet_ts = pd.to_datetime(
        panel["tdnet_clean_feature_source_max_timestamp"], errors="coerce"
    )
    cutoff = (
        panel["date"]
        .dt.tz_localize("Asia/Tokyo")
        .add(pd.Timedelta(hours=8, minutes=58, seconds=59))
    )
    if (tdnet_ts.notna() & tdnet_ts.gt(cutoff)).any():
        raise RuntimeError("TDnet feature source exceeds decision cutoff")
    y = pd.to_numeric(panel["oc_return_pct"], errors="coerce")
    label = pd.to_numeric(panel["label"], errors="coerce")
    observed = y.notna()
    if not np.array_equal(
        label.loc[observed].to_numpy(float),
        y.loc[observed].gt(0).to_numpy(float),
    ):
        raise RuntimeError("label is not exact return sign")
    return sessions


def strict_sessions(
    sessions: pd.DatetimeIndex, cache_dates: set[pd.Timestamp]
) -> tuple[pd.DatetimeIndex, dict[str, list[str]]]:
    accepted: list[pd.Timestamp] = []
    rejected: dict[str, list[str]] = {}
    for position, date in enumerate(sessions):
        if position == 0:
            rejected[str(date.date())] = ["prior panel session unavailable"]
            continue
        prior = sessions[position - 1]
        required = pd.date_range(prior, date, freq="D")
        missing = [str(item.date()) for item in required if item not in cache_dates]
        if missing:
            rejected[str(date.date())] = missing
        else:
            accepted.append(date)
    return pd.DatetimeIndex(accepted), rejected


def rebuild_tdnet_covariates(
    panel: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    accepted: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct the locked clean-title covariates from the frozen indexes.

    The corrected price panel predates some of the cached 2025 title features.
    Treating those missing values as no-event would be invalid, so all complete
    sessions are reconstructed uniformly from the hash-bound raw indexes.
    """

    frames: list[pd.DataFrame] = []
    for directory in CACHE_PATHS:
        for path in sorted(directory.glob("*.html")):
            date = pd.to_datetime(path.stem, format="%Y%m%d").normalize()
            frames.append(parse_tdnet_index(path.read_bytes(), date))
    disclosures = pd.concat(frames, ignore_index=True).sort_values(
        ["published_at", "code", "title"], kind="stable"
    )
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
    index_is_session = events["index_date"].isin(sessions)
    minute = events["published_at"].dt.hour * 60 + events["published_at"].dt.minute
    close_minute = np.where(
        events["index_date"].lt(pd.Timestamp("2024-11-05")),
        15 * 60,
        15 * 60 + 30,
    )
    intraday_prior = index_is_session & ~same_day & minute.lt(close_minute)
    events = events.loc[
        ~intraday_prior & events["date"].isin(accepted)
    ].reset_index(drop=True)
    target_cutoff = pd.Series(
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
            for date in events["date"]
        ]
    )
    events["age_hours"] = (
        target_cutoff.reset_index(drop=True) - events["published_at"].reset_index(drop=True)
    ).dt.total_seconds() / 3600.0
    if events["age_hours"].lt(0).any():
        raise RuntimeError("reconstructed event mapped before publication")
    flags = _title_flags(events["title"]).reset_index(drop=True)
    events = pd.concat([events, flags], axis=1)
    keys = ["date", "code"]
    grouped = events.groupby(keys, sort=True)
    latest = (
        grouped["published_at"]
        .max()
        .rename("tdnet_clean_feature_source_max_timestamp")
        .reset_index()
    )
    bundles = grouped.agg(
        tdnet_clean_document_count=("title", "size"),
        tdnet_clean_latest_age_hours=("age_hours", "min"),
    ).reset_index()
    bundles = bundles.merge(latest, on=keys, how="left", validate="one_to_one")
    source_names = (
        "earnings",
        "revision",
        "revision_up",
        "revision_down",
        "dividend",
        "dividend_up",
        "dividend_down",
        "buyback_decision",
        "buyback_tostnet",
        "equity_financing",
        "benefit",
        "split",
        "ma_alliance",
        "control_transaction",
        "impairment_loss",
        "audit_problem",
        "correction",
    )
    maxima = grouped[list(source_names)].max().reset_index()
    bundles = bundles.merge(maxima, on=keys, how="left", validate="one_to_one")
    mapping = {
        "earnings": "tdnet_clean_has_earnings",
        "revision": "tdnet_clean_has_revision",
        "revision_up": "tdnet_clean_revision_up_title",
        "revision_down": "tdnet_clean_revision_down_title",
        "dividend": "tdnet_clean_has_dividend",
        "dividend_up": "tdnet_clean_dividend_up_title",
        "dividend_down": "tdnet_clean_dividend_down_title",
        "buyback_decision": "tdnet_clean_has_buyback_decision",
        "buyback_tostnet": "tdnet_clean_has_buyback_tostnet",
        "equity_financing": "tdnet_clean_has_external_equity_financing",
        "benefit": "tdnet_clean_has_benefit",
        "split": "tdnet_clean_has_split",
        "ma_alliance": "tdnet_clean_has_ma_transaction",
        "control_transaction": "tdnet_clean_has_control_transaction",
        "impairment_loss": "tdnet_clean_has_impairment_loss",
        "audit_problem": "tdnet_clean_has_audit_problem",
        "correction": "tdnet_clean_has_correction",
    }
    for source, target in mapping.items():
        bundles[target] = bundles[source].astype(float)
    bundles["tdnet_clean_has_business_alliance"] = bundles["ma_alliance"].astype(float)
    family_columns = list(mapping.values()) + ["tdnet_clean_has_business_alliance"]
    bundles["tdnet_clean_family_count_log1p"] = np.log1p(
        bundles[family_columns].sum(axis=1)
    )
    bundles["tdnet_clean_document_count_log1p"] = np.log1p(
        bundles["tdnet_clean_document_count"]
    )
    bundles["tdnet_clean_latest_age_hours_log1p"] = np.log1p(
        bundles["tdnet_clean_latest_age_hours"]
    )
    supportive = bundles[
        [
            "tdnet_clean_revision_up_title",
            "tdnet_clean_dividend_up_title",
            "tdnet_clean_has_buyback_decision",
            "tdnet_clean_has_benefit",
            "tdnet_clean_has_split",
        ]
    ].max(axis=1)
    adverse = bundles[
        [
            "tdnet_clean_revision_down_title",
            "tdnet_clean_dividend_down_title",
            "tdnet_clean_has_external_equity_financing",
            "tdnet_clean_has_impairment_loss",
            "tdnet_clean_has_audit_problem",
        ]
    ].max(axis=1)
    bundles["tdnet_clean_support_adverse_conflict"] = (
        supportive.gt(0) & adverse.gt(0)
    ).astype(float)
    bundles["tdnet_clean_any"] = 1.0

    rebuilt_columns = [
        "tdnet_clean_any",
        *EVENT_FEATURES,
        "tdnet_clean_feature_source_max_timestamp",
    ]
    rebuilt = panel.drop(columns=rebuilt_columns, errors="ignore").merge(
        bundles[["date", "code", *rebuilt_columns]],
        on=["date", "code"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    complete_rows = rebuilt["date"].isin(accepted)
    numeric_columns = ["tdnet_clean_any", *EVENT_FEATURES]
    rebuilt.loc[complete_rows, numeric_columns] = rebuilt.loc[
        complete_rows, numeric_columns
    ].fillna(0.0)
    rebuilt.loc[~complete_rows, numeric_columns] = np.nan
    source = pd.to_datetime(
        rebuilt["tdnet_clean_feature_source_max_timestamp"], errors="coerce"
    )
    cutoff = (
        rebuilt["date"]
        .dt.tz_localize("Asia/Tokyo")
        .add(pd.Timedelta(hours=8, minutes=58, seconds=59))
    )
    if (source.notna() & source.gt(cutoff)).any():
        raise RuntimeError("reconstructed TDnet source exceeds decision cutoff")
    return rebuilt, {
        "parsed_disclosures": int(len(disclosures)),
        "qualifying_disclosures": int(len(events)),
        "reconstructed_bundles": int(len(bundles)),
        "reconstructed_complete_rows": int(complete_rows.sum()),
    }


def date_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("date", sort=False)["date"].transform("size").to_numpy(float)
    value = 1.0 / counts
    totals = pd.Series(value).groupby(
        frame["date"].reset_index(drop=True), sort=False
    ).sum()
    if not np.allclose(totals, 1.0, atol=1e-12, rtol=0):
        raise AssertionError("date weights do not sum to one")
    return value


class FoldTransform:
    """Fold-local deterministic imputation and standardisation."""

    def __init__(self, features: tuple[str, ...], frame: pd.DataFrame):
        self.features = features
        raw = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy()
        self.imputer = SimpleImputer(strategy="median", add_indicator=True)
        values = self.imputer.fit_transform(raw)
        self.scaler = StandardScaler()
        self.scaler.fit(values)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame.loc[:, self.features].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy()
        return self.scaler.transform(self.imputer.transform(raw))


def primary_family(frame: pd.DataFrame) -> np.ndarray:
    def flag(name: str) -> np.ndarray:
        return pd.to_numeric(frame[name], errors="coerce").fillna(0).gt(0).to_numpy()

    result = np.full(len(frame), "other", dtype=object)
    masks = [
        (
            "ma_or_alliance",
            flag("tdnet_clean_has_ma_transaction")
            | flag("tdnet_clean_has_business_alliance")
            | flag("tdnet_clean_has_control_transaction"),
        ),
        (
            "benefit_or_split",
            flag("tdnet_clean_has_benefit") | flag("tdnet_clean_has_split"),
        ),
        ("dividend", flag("tdnet_clean_has_dividend")),
        ("earnings", flag("tdnet_clean_has_earnings")),
        ("revision", flag("tdnet_clean_has_revision")),
        ("buyback", flag("tdnet_clean_has_buyback_decision")),
        (
            "adverse",
            flag("tdnet_clean_has_external_equity_financing")
            | flag("tdnet_clean_revision_down_title")
            | flag("tdnet_clean_dividend_down_title")
            | flag("tdnet_clean_has_impairment_loss")
            | flag("tdnet_clean_has_audit_problem")
            | flag("tdnet_clean_has_correction"),
        ),
    ]
    for name, mask in masks:
        result[mask] = name
    return result


def fit_ridge(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray, alpha: float = 10.0
) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(x, y, sample_weight=weight)
    return model


def fit_propensity(
    x: np.ndarray, treatment: np.ndarray, weight: np.ndarray
) -> LogisticRegression:
    if np.unique(treatment).size != 2:
        raise RuntimeError("propensity fit requires treated and control rows")
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=500,
        random_state=SEED,
    )
    model.fit(x, treatment, sample_weight=weight)
    return model


def probability(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.clip(model.predict_proba(x)[:, 1], 0.01, 0.99)
    return np.clip(model.predict(x), 0.0, 1.0)


def matched_pseudo_outcome(training: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    treated = training["tdnet_clean_any"].eq(1).to_numpy()
    transform = FoldTransform(MATCH_FEATURES, training.loc[~treated])
    x = transform.transform(training)
    y = np.clip(training["oc_return_pct"].to_numpy(float), -5.0, 5.0)
    pseudo = np.full(len(training), np.nan)
    for _, positions in training.groupby("date", sort=False).indices.items():
        positions = np.asarray(positions, dtype=int)
        event_pos = positions[treated[positions]]
        control_pos = positions[~treated[positions]]
        if not len(event_pos) or not len(control_pos):
            continue
        neighbors = min(10, len(control_pos))
        model = NearestNeighbors(n_neighbors=neighbors, metric="euclidean")
        model.fit(x[control_pos])
        indices = model.kneighbors(x[event_pos], return_distance=False)
        pseudo[event_pos] = y[event_pos] - y[control_pos][indices].mean(axis=1)
    valid = treated & np.isfinite(pseudo)
    return pseudo, valid


def crossfit_aipw(
    training: pd.DataFrame, x_base: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    treatment = training["tdnet_clean_any"].eq(1).to_numpy(int)
    y = np.clip(training["oc_return_pct"].to_numpy(float), -5.0, 5.0)
    unique_dates = pd.DatetimeIndex(training["date"].drop_duplicates().sort_values())
    date_fold = {date: position % 3 for position, date in enumerate(unique_dates)}
    assignment = training["date"].map(date_fold).to_numpy(int)
    pseudo = np.full(len(training), np.nan)
    fold_details: list[dict[str, Any]] = []
    for fold in range(3):
        holdout = assignment == fold
        fit = ~holdout
        fit_t = treatment[fit]
        if fit_t.sum() < 50 or (fit_t == 0).sum() < 500:
            raise RuntimeError("insufficient cross-fit treatment support")
        fit_frame = training.loc[fit].reset_index(drop=True)
        weights = date_weights(fit_frame)
        propensity = fit_propensity(x_base[fit], fit_t, weights)
        e = probability(propensity, x_base[holdout])
        treated_fit = fit_t == 1
        control_fit = ~treated_fit
        m1 = fit_ridge(
            x_base[fit][treated_fit],
            y[fit][treated_fit],
            date_weights(fit_frame.loc[treated_fit].reset_index(drop=True)),
        )
        m0 = fit_ridge(
            x_base[fit][control_fit],
            y[fit][control_fit],
            date_weights(fit_frame.loc[control_fit].reset_index(drop=True)),
        )
        m1_value = m1.predict(x_base[holdout])
        m0_value = m0.predict(x_base[holdout])
        t_hold = treatment[holdout]
        y_hold = y[holdout]
        dr = (
            m1_value
            - m0_value
            + t_hold * (y_hold - m1_value) / e
            - (1 - t_hold) * (y_hold - m0_value) / (1.0 - e)
        )
        pseudo[holdout] = np.clip(dr, -10.0, 10.0)
        fold_details.append(
            {
                "fold": fold,
                "fit_rows": int(fit.sum()),
                "holdout_rows": int(holdout.sum()),
                "fit_treated": int(fit_t.sum()),
                "holdout_treated": int(t_hold.sum()),
            }
        )
    if not np.isfinite(pseudo).all():
        raise RuntimeError("cross-fitted AIPW pseudo-outcome is incomplete")
    return pseudo, {"folds": fold_details}


Predictor = Callable[[pd.DataFrame], dict[str, tuple[np.ndarray, np.ndarray]]]


def fit_fold(training: pd.DataFrame) -> tuple[Predictor, dict[str, Any]]:
    training = training.reset_index(drop=True)
    treatment = training["tdnet_clean_any"].eq(1).to_numpy()
    if treatment.sum() < 100 or (~treatment).sum() < 1000:
        raise RuntimeError("insufficient treatment/control support")
    y = np.clip(training["oc_return_pct"].to_numpy(float), -5.0, 5.0)
    base_transform = FoldTransform(BASE_FEATURES, training)
    full_transform = FoldTransform(FULL_FEATURES, training)
    x_base = base_transform.transform(training)
    x_full = full_transform.transform(training)

    matched_y, matched_valid = matched_pseudo_outcome(training)
    matched_frame = training.loc[matched_valid].reset_index(drop=True)
    matched_model = fit_ridge(
        x_full[matched_valid],
        matched_y[matched_valid],
        date_weights(matched_frame),
    )

    train_family = primary_family(training)
    family_models: dict[str, tuple[Ridge, Ridge]] = {}
    family_counts: dict[str, int] = {}
    controls = ~treatment
    for family in (
        "adverse",
        "buyback",
        "revision",
        "earnings",
        "dividend",
        "benefit_or_split",
        "ma_or_alliance",
        "other",
    ):
        treated_family = treatment & (train_family == family)
        family_counts[family] = int(treated_family.sum())
        if treated_family.sum() < 50:
            continue
        family_models[family] = (
            fit_ridge(
                x_base[treated_family],
                y[treated_family],
                date_weights(training.loc[treated_family].reset_index(drop=True)),
            ),
            fit_ridge(
                x_base[controls],
                y[controls],
                date_weights(training.loc[controls].reset_index(drop=True)),
            ),
        )

    aipw_y, aipw_detail = crossfit_aipw(training, x_base)
    aipw_model = fit_ridge(x_full, aipw_y, date_weights(training))

    control_date_mean = (
        training.loc[controls]
        .groupby("date", sort=False)["oc_return_pct"]
        .mean()
    )
    event_date_residual = (
        training.loc[treatment, "oc_return_pct"].to_numpy(float)
        - training.loc[treatment, "date"].map(control_date_mean).to_numpy(float)
    )
    date_valid = np.isfinite(event_date_residual)
    event_positions = np.flatnonzero(treatment)
    date_positions = event_positions[date_valid]
    date_model = fit_ridge(
        x_full[date_positions],
        np.clip(event_date_residual[date_valid], -10.0, 10.0),
        date_weights(training.loc[date_positions].reset_index(drop=True)),
    )

    issuer = training[["code", "tdnet_clean_any", "oc_return_pct"]].copy()
    issuer["treated_sum"] = issuer["oc_return_pct"].where(
        issuer["tdnet_clean_any"].eq(1), 0.0
    )
    issuer["control_sum"] = issuer["oc_return_pct"].where(
        issuer["tdnet_clean_any"].eq(0), 0.0
    )
    issuer["treated_n"] = issuer["tdnet_clean_any"].eq(1).astype(int)
    issuer["control_n"] = issuer["tdnet_clean_any"].eq(0).astype(int)
    issuer_stats = issuer.groupby("code", sort=False)[
        ["treated_sum", "control_sum", "treated_n", "control_n"]
    ].sum()
    issuer_score = (
        issuer_stats["treated_sum"] / (issuer_stats["treated_n"] + 20.0)
        - issuer_stats["control_sum"] / (issuer_stats["control_n"] + 20.0)
    ).to_dict()

    all_weights = date_weights(training)
    propensity_model = fit_propensity(x_base, treatment.astype(int), all_weights)
    train_propensity = probability(propensity_model, x_base)
    lower = max(
        float(np.quantile(train_propensity[treatment], 0.05)),
        float(np.quantile(train_propensity[controls], 0.05)),
    )
    upper = min(
        float(np.quantile(train_propensity[treatment], 0.95)),
        float(np.quantile(train_propensity[controls], 0.95)),
    )
    overlap_valid = lower <= upper
    pooled_treated = fit_ridge(
        x_base[treatment],
        y[treatment],
        date_weights(training.loc[treatment].reset_index(drop=True)),
    )
    pooled_control = fit_ridge(
        x_base[controls],
        y[controls],
        date_weights(training.loc[controls].reset_index(drop=True)),
    )

    positive = y > 1.0
    positive_treated = fit_ridge(
        x_base[treatment],
        positive[treatment].astype(float),
        date_weights(training.loc[treatment].reset_index(drop=True)),
    )
    positive_control = fit_ridge(
        x_base[controls],
        positive[controls].astype(float),
        date_weights(training.loc[controls].reset_index(drop=True)),
    )
    tail_scale = float(y[treatment & positive].mean())

    unique_dates = pd.DatetimeIndex(training["date"].drop_duplicates().sort_values())
    cut = max(1, min(len(unique_dates) - 1, int(math.floor(len(unique_dates) * 0.5))))
    structure_dates = set(unique_dates[:cut])
    structure = training["date"].isin(structure_dates).to_numpy()
    estimation = ~structure
    tree = DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=500,
        random_state=SEED,
    )
    tree.fit(
        x_full[structure],
        aipw_y[structure],
        sample_weight=date_weights(training.loc[structure].reset_index(drop=True)),
    )
    leaves = tree.apply(x_full)
    leaf_values: dict[int, float] = {}
    for leaf in np.unique(leaves[estimation]):
        mask = estimation & (leaves == leaf)
        values = aipw_y[mask]
        leaf_values[int(leaf)] = float(values.sum() / (len(values) + 100.0))

    def predict(scoring: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        scoring = scoring.reset_index(drop=True)
        treated = scoring["tdnet_clean_any"].eq(1).to_numpy()
        score_base = base_transform.transform(scoring)
        score_full = full_transform.transform(scoring)
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        value = matched_model.predict(score_full)
        result[HYPOTHESES[0]] = (value, treated & np.isfinite(value))

        family = primary_family(scoring)
        value = np.full(len(scoring), -np.inf)
        allowed = np.zeros(len(scoring), dtype=bool)
        for name, (model_t, model_c) in family_models.items():
            mask = treated & (family == name)
            value[mask] = model_t.predict(score_base[mask]) - model_c.predict(
                score_base[mask]
            )
            allowed[mask] = True
        result[HYPOTHESES[1]] = (value, allowed)

        value = aipw_model.predict(score_full)
        result[HYPOTHESES[2]] = (value, treated & np.isfinite(value))

        value = date_model.predict(score_full)
        result[HYPOTHESES[3]] = (value, treated & np.isfinite(value))

        value = scoring["code"].map(issuer_score).fillna(0.0).to_numpy(float)
        result[HYPOTHESES[4]] = (value, treated)

        score_propensity = probability(propensity_model, score_base)
        value = pooled_treated.predict(score_base) - pooled_control.predict(score_base)
        allowed = (
            treated
            & overlap_valid
            & (score_propensity >= lower)
            & (score_propensity <= upper)
        )
        result[HYPOTHESES[5]] = (value, allowed)

        p_treated = np.clip(positive_treated.predict(score_base), 0.0, 1.0)
        p_control = np.clip(positive_control.predict(score_base), 0.0, 1.0)
        value = (p_treated - p_control) * tail_scale
        result[HYPOTHESES[6]] = (value, treated & np.isfinite(value))

        score_leaves = tree.apply(score_full)
        value = np.asarray(
            [leaf_values.get(int(leaf), 0.0) for leaf in score_leaves],
            dtype=float,
        )
        allowed = treated & np.asarray(
            [int(leaf) in leaf_values for leaf in score_leaves], dtype=bool
        )
        result[HYPOTHESES[7]] = (value, allowed)
        return result

    detail = {
        "training_rows": int(len(training)),
        "training_sessions": int(training["date"].nunique()),
        "treated_rows": int(treatment.sum()),
        "control_rows": int(controls.sum()),
        "matched_pseudo_rows": int(matched_valid.sum()),
        "family_treated_counts": family_counts,
        "aipw": aipw_detail,
        "overlap_interval": [lower, upper] if overlap_valid else None,
        "positive_tail_scale_pct": tail_scale,
        "policy_tree_leaves": int(tree.get_n_leaves()),
        "policy_tree_estimated_leaves": int(len(leaf_values)),
    }
    return predict, detail


def digest_predictions(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]]
) -> str:
    digest = hashlib.sha256()
    for name in sorted(predictions):
        score, allowed = predictions[name]
        digest.update(name.encode("ascii"))
        digest.update(np.ascontiguousarray(score).tobytes())
        digest.update(np.ascontiguousarray(allowed).tobytes())
    return digest.hexdigest()


def selected_keys(
    frame: pd.DataFrame,
    score: np.ndarray,
    allowed: np.ndarray,
    capacity: int,
) -> tuple[tuple[str, str], ...]:
    work = frame.loc[
        allowed & np.isfinite(score) & (score > BREAK_EVEN_PCT),
        ["date", "code"],
    ].copy()
    work["score"] = score[
        allowed & np.isfinite(score) & (score > BREAK_EVEN_PCT)
    ]
    work = work.sort_values(
        ["date", "score", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    work = work.groupby("date", sort=False).head(capacity)
    return tuple(
        (str(date.date()), str(code))
        for date, code in work[["date", "code"]].itertuples(index=False)
    )


def make_picks(
    scoring: pd.DataFrame,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for hypothesis in HYPOTHESES:
        score, allowed = predictions[hypothesis]
        eligible = allowed & np.isfinite(score) & (score > BREAK_EVEN_PCT)
        base = scoring.loc[
            eligible, ["date", "code", "name", "oc_return_pct", "label"]
        ].copy()
        base["uplift_score_pct"] = score[eligible]
        base = base.sort_values(
            ["date", "uplift_score_pct", "code"],
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
                            "uplift_score_pct": (
                                None if row is None else float(row.uplift_score_pct)
                            ),
                            "oc_return_pct": (
                                None if row is None else float(row.oc_return_pct)
                            ),
                            "label": None if row is None else float(row.label),
                        }
                    )
    return pd.DataFrame(rows)


def daily_returns(
    policy_picks: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    cost_bps: int,
) -> pd.DataFrame:
    capacity = int(policy_picks["capacity"].iloc[0])
    executed = policy_picks["code"].notna().astype(float)
    gross_contribution = (
        pd.to_numeric(policy_picks["oc_return_pct"], errors="coerce").fillna(0.0)
        / capacity
    )
    cost_contribution = executed * (cost_bps / 100.0) / capacity
    value = pd.DataFrame(
        {
            "date": pd.to_datetime(policy_picks["date"]),
            "gross": gross_contribution,
            "net": gross_contribution - cost_contribution,
            "exposure": executed / capacity,
        }
    ).groupby("date", sort=True).sum()
    return value.reindex(sessions, fill_value=0.0)


def safe_mean(values: pd.Series | np.ndarray) -> float | None:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    return None if not len(clean) else float(clean.mean())


def policy_metrics(
    policy_picks: pd.DataFrame, sessions: pd.DatetimeIndex
) -> dict[str, Any]:
    capacity = int(policy_picks["capacity"].iloc[0])
    daily = {cost: daily_returns(policy_picks, sessions, cost) for cost in COSTS_BPS}
    gross = daily[40]["gross"]
    net20 = daily[20]["net"]
    net40 = daily[40]["net"]
    net60 = daily[60]["net"]
    executed = policy_picks.loc[policy_picks["code"].notna()].copy()
    exposure = daily[40]["exposure"]
    months = net40.groupby(net40.index.to_period("M")).mean()
    slices = {
        name: float(net40.loc[(net40.index >= start) & (net40.index <= end)].mean())
        for name, (start, end) in PERIODS.items()
    }

    best_removed: dict[str, float | None] = {}
    worst_removed: dict[str, float | None] = {}
    for count in (5, 10, 20):
        best_removed[str(count)] = safe_mean(net20.nsmallest(max(len(net20) - count, 0)))
        worst_removed[str(count)] = safe_mean(
            net20.nlargest(max(len(net20) - count, 0))
        )

    if len(executed):
        code_weight = executed.groupby("code", sort=False).size() / capacity
        total_weight = float(code_weight.sum())
        code_share = code_weight / total_weight
        largest_code = float(code_share.max())
        top10_code = float(code_share.nlargest(10).sum())
        slot_net20 = (
            pd.to_numeric(executed["oc_return_pct"], errors="raise") - 0.20
        ) / capacity
        pnl_by_code = slot_net20.groupby(executed["code"]).sum()
        positive = pnl_by_code[pnl_by_code > 0].sort_values(ascending=False)
        positive_total = float(positive.sum())
        largest_positive = (
            None
            if positive_total <= 0
            else float(positive.iloc[0] / positive_total)
        )
        top_positive_codes = list(positive.head(10).index)
        removed = executed.loc[~executed["code"].isin(top_positive_codes)].copy()
        reduced_daily = (
            (
                pd.to_numeric(removed["oc_return_pct"], errors="raise") - 0.20
            )
            .div(capacity)
            .groupby(pd.to_datetime(removed["date"]))
            .sum()
            .reindex(sessions, fill_value=0.0)
        )
        top10_cash = float(reduced_daily.mean())
    else:
        largest_code = top10_code = 0.0
        largest_positive = None
        top10_cash = 0.0

    tail_count = max(1, int(math.ceil(len(net40) * 0.05)))
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
            str(period): float(value) for period, value in months.items()
        },
        "positive_months_net40": int((months > 0).sum()),
        "temporal_slices_net40_pct": slices,
        "best_days_removed_net20_pct": best_removed,
        "worst_days_removed_net20_pct": worst_removed,
        "expected_shortfall05_net40_pct": float(net40.nsmallest(tail_count).mean()),
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
        default=ROOT / "research" / "model_v11_uplift_result.json",
    )
    parser.add_argument(
        "--picks",
        type=Path,
        default=ROOT / "research" / "model_v11_uplift_picks.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol, panel, manifest, cache_dates = load_inputs()
    sessions = validate_panel(panel, manifest, protocol)
    accepted, rejected = strict_sessions(sessions, cache_dates)
    panel, reconstruction = rebuild_tdnet_covariates(panel, sessions, accepted)
    score_sessions = accepted[(accepted >= SCORE_START) & (accepted <= SCORE_END)]
    if not len(score_sessions):
        raise RuntimeError("no strict score sessions")

    panel["_strict"] = panel["date"].isin(accepted)
    fold_rows: list[pd.DataFrame] = []
    fold_details: list[dict[str, Any]] = []
    mutation_checks: list[dict[str, Any]] = []
    periods = score_sessions.to_period("M").drop_duplicates()
    for period in periods:
        period_sessions = score_sessions[score_sessions.to_period("M") == period]
        fold_start = period_sessions.min()
        training = panel.loc[
            panel["_strict"]
            & panel["date"].lt(fold_start)
            & panel["price_training_eligible"].fillna(False)
            & panel["oc_return_pct"].notna()
            & panel["tdnet_clean_any"].notna()
        ].copy()
        scoring = panel.loc[
            panel["date"].isin(period_sessions)
            & panel["price_eligible"].fillna(False)
            & panel["tdnet_clean_any"].notna()
        ].copy()
        training = training.sort_values(["date", "code"], kind="stable").reset_index(
            drop=True
        )
        scoring = scoring.sort_values(["date", "code"], kind="stable").reset_index(
            drop=True
        )
        if training["date"].max() >= fold_start:
            raise RuntimeError("fold training is not strictly prior")
        predictor, detail = fit_fold(training)
        predictions = predictor(scoring)

        mutated = scoring.copy()
        mutated["oc_return_pct"] = np.linspace(-99.0, 99.0, len(mutated))
        mutated["label"] = 1.0 - pd.to_numeric(
            mutated["label"], errors="coerce"
        ).fillna(0.0)
        mutation_predictions = predictor(mutated)
        original_digest = digest_predictions(predictions)
        mutation_digest = digest_predictions(mutation_predictions)
        key_equal = all(
            selected_keys(scoring, *predictions[hypothesis], capacity)
            == selected_keys(mutated, *mutation_predictions[hypothesis], capacity)
            for hypothesis in HYPOTHESES
            for capacity in CAPACITIES
        )
        if original_digest != mutation_digest or not key_equal:
            raise RuntimeError(f"score outcome mutation changed policy: {period}")

        fold_rows.append(make_picks(scoring, predictions, period_sessions))
        fold_details.append(
            {
                "period": str(period),
                "score_days": int(len(period_sessions)),
                "score_rows": int(len(scoring)),
                "score_treated_rows": int(scoring["tdnet_clean_any"].eq(1).sum()),
                **detail,
            }
        )
        mutation_checks.append(
            {
                "period": str(period),
                "prediction_digest": original_digest,
                "mutated_prediction_digest": mutation_digest,
                "selected_keys_equal": key_equal,
                "passes": original_digest == mutation_digest and key_equal,
            }
        )

    picks = pd.concat(fold_rows, ignore_index=True)
    picks = picks.sort_values(
        ["policy_id", "date", "slot"], kind="stable"
    ).reset_index(drop=True)
    metrics = {
        policy_id: policy_metrics(group.reset_index(drop=True), score_sessions)
        for policy_id, group in picks.groupby("policy_id", sort=True)
    }
    picks.to_csv(args.picks, index=False, lineterminator="\n")

    result = {
        "schema_version": 1,
        "experiment_id": "model_v11_uplift_zero_base_20260723",
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
            "strict_complete_sessions": int(len(accepted)),
            "strict_score_sessions": int(len(score_sessions)),
            "strict_score_first": str(score_sessions.min().date()),
            "strict_score_last": str(score_sessions.max().date()),
            "score_months": [str(value) for value in periods],
            "rejected_score_sessions": int(
                len(
                    sessions[
                        (sessions >= SCORE_START)
                        & (sessions <= SCORE_END)
                        & ~sessions.isin(score_sessions)
                    ]
                )
            ),
            "rejected_examples": {
                key: value
                for key, value in rejected.items()
                if SCORE_START <= pd.Timestamp(key) <= SCORE_END
            },
            "tdnet_reconstruction": reconstruction,
        },
        "folds": fold_details,
        "integrity": {
            "monthly_expanding_strict_prior": True,
            "future_source_violations": 0,
            "target_session_outcome_mutation": {
                "passes": all(item["passes"] for item in mutation_checks),
                "folds": mutation_checks,
            },
        },
        "metrics": metrics,
        "point_estimate_best_net40_policy": max(
            metrics,
            key=lambda name: metrics[name]["net40_mean_pct"],
        ),
        "point_estimate_best_net40_pct": max(
            value["net40_mean_pct"] for value in metrics.values()
        ),
        "decision": {
            "status": "pending_independent_audit",
            "retrospective_candidate": None,
            "production_ready": False,
            "reason": "The outcome panel is retrospective and an independent P&L, multiplicity, concentration, and reproducibility audit is required.",
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
                "score_days": len(score_sessions),
                "best": result["point_estimate_best_net40_policy"],
                "best_net40": result["point_estimate_best_net40_pct"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
