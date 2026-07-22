#!/usr/bin/env python3
"""Run the frozen v0.6 feature-group retrospective diagnostic.

The historical outcomes used here are already known from earlier research.
This runner can identify broken, redundant or period-specific hypotheses, but
it cannot promote a production model.  The catalog and protocol are checked by
SHA before any metric is computed, and every comparison uses one fixed raked
logistic-regression anchor.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits


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
from tse_session_ranker.data.jpx import collect_jpx  # noqa: E402
from tse_session_ranker.data.tdnet import (  # noqa: E402
    collect_tdnet_dataset,
    tdnet_target_completeness,
)
from tse_session_ranker.features import build_feature_panel  # noqa: E402
from tse_session_ranker.io import write_frame, write_json  # noqa: E402
from tse_session_ranker.profit import (  # noqa: E402
    daily_portfolio_returns,
    profit_metrics,
)
from tse_session_ranker.research_candidates import (  # noqa: E402
    HISTORICAL_PRICE_CANDIDATE_COLUMNS,
    T0_CLEAN_EVENT_COLUMNS,
    T1_EVENT_STRUCTURE_COLUMNS,
    X0_EVENT_CONTEXT_COLUMNS,
    add_event_context_interactions,
    add_historical_candidate_features,
    attach_clean_tdnet_candidate_features,
)
from tse_session_ranker.research_features import add_session_market_features  # noqa: E402
from tse_session_ranker.research_models import (  # noqa: E402
    ResearchModelSpec,
    fit_research_model,
)
from tse_session_ranker.validation import moving_block_bootstrap  # noqa: E402


CATALOG_PATH = ROOT / "research/model_v06_feature_catalog.json"
PROTOCOL_PATH = ROOT / "research/model_v06_feature_protocol.json"
EXPECTED_CATALOG_SHA256 = (
    "bbf87983979d48afbe1eaa51f3368cca713c306f2b7d4cdc8cda54e5fe43414a"
)
EXPECTED_PROTOCOL_SHA256 = (
    "20038eb33d87a1fbcd187548f7f14c65dca136d00850566311a6888bd0083021"
)
RESULT_SCHEMA_VERSION = 1
PANEL_SCHEMA_VERSION = 1
PRIMARY_COST_BPS = 20.0
STRESS_COST_BPS = 40.0
TOP_K = 2
BOOTSTRAP_SAMPLES = 5_000
MAX_T_CONFIDENCE = 0.80
BLOCK_LENGTH = 5
DEFAULT_SEED = 31


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_hashes(directory: Path, pattern: str) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(directory.glob(pattern))
        if path.is_file()
    }


def _implementation_hashes() -> dict[str, str]:
    relatives = (
        "research/screen_feature_candidates_v06.py",
        "research/finalize_logit_v04.py",
        "src/tse_session_ranker/research_candidates.py",
        "src/tse_session_ranker/research_features.py",
        "src/tse_session_ranker/research_models.py",
        "src/tse_session_ranker/features.py",
        "src/tse_session_ranker/data/common.py",
        "src/tse_session_ranker/data/jpx.py",
        "src/tse_session_ranker/data/tdnet.py",
        "src/tse_session_ranker/profit.py",
        "src/tse_session_ranker/validation.py",
    )
    return {relative: sha256_file(ROOT / relative) for relative in relatives}


def _load_frozen_definitions() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(CATALOG_PATH) != EXPECTED_CATALOG_SHA256:
        raise ValueError("v0.6 feature catalog changed after registration")
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("v0.6 feature protocol changed after registration")
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != 1:
        raise ValueError("unsupported v0.6 feature catalog")
    if protocol.get("status") != "frozen_before_model_v06_feature_metrics":
        raise ValueError("v0.6 feature protocol is not frozen")
    if protocol["catalog"]["sha256"] != EXPECTED_CATALOG_SHA256:
        raise ValueError("protocol references a different feature catalog")
    return catalog, protocol


def _input_lock(
    *,
    jpx_directory: Path,
    tdnet_directory: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "catalog_sha256": EXPECTED_CATALOG_SHA256,
        "implementation_sha256": _implementation_hashes(),
        "jpx_pdf_sha256": _directory_hashes(jpx_directory, "*.pdf"),
        "tdnet_html_sha256": _directory_hashes(tdnet_directory, "*.html"),
        "tdnet_metadata_sha256": _directory_hashes(
            tdnet_directory, "*.html.meta.json"
        ),
    }


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_or_validate_lock(path: Path, lock: dict[str, Any]) -> str:
    payload = _canonical_json(lock)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("existing v0.6 input lock differs from current inputs")
    if not path.exists():
        path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _catalog_groups(catalog: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    groups = {
        str(name): tuple(str(value) for value in raw["features"])
        for name, raw in catalog["historical_screen"].items()
    }
    implementation = {
        "G1_short_reversal": tuple(HISTORICAL_PRICE_CANDIDATE_COLUMNS[:4]),
        "T0_clean_event": tuple(T0_CLEAN_EVENT_COLUMNS),
        "T1_event_structure": tuple(T1_EVENT_STRUCTURE_COLUMNS),
        "X0_event_context": tuple(X0_EVENT_CONTEXT_COLUMNS),
    }
    for name, columns in implementation.items():
        if groups[name] != columns:
            raise ValueError(f"catalog/implementation mismatch for {name}")
    return groups


def _panel_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _project_panel(
    panel: pd.DataFrame, groups: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    columns = list(
        dict.fromkeys(
            [
                "date",
                "code",
                "name",
                "eligible",
                "training_eligible",
                "price_eligible",
                "price_training_eligible",
                "label",
                "oc_return_pct",
                "outcome_observed",
                "source_complete",
                "universe_source_complete",
                "feature_source_max_date",
                "candidate_price_source_max_date",
                "tdnet_source_complete",
                "tdnet_clean_feature_source_max_timestamp",
                *(
                    column
                    for values in groups.values()
                    for column in values
                ),
            ]
        )
    )
    missing = sorted(set(columns) - set(panel.columns))
    if missing:
        raise ValueError(f"v0.6 panel lacks registered columns: {missing}")
    return panel.loc[:, columns].copy()


def build_candidate_panel(
    *,
    jpx_directory: Path,
    tdnet_directory: Path,
    catalog_groups: Mapping[str, Sequence[str]],
    panel_cache: Path,
    input_lock_sha256: str,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, Any]]:
    manifest_path = _panel_manifest_path(panel_cache)
    if panel_cache.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != PANEL_SCHEMA_VERSION
            or manifest.get("input_lock_sha256") != input_lock_sha256
            or manifest.get("panel_file_sha256") != sha256_file(panel_cache)
        ):
            raise ValueError("cached v0.6 panel manifest does not match the frozen run")
        panel = joblib.load(panel_cache)
        if list(panel.columns) != manifest["columns"]:
            raise ValueError("cached v0.6 panel columns changed")
        sessions = normalize_expected_sessions(manifest["sessions"])
        return panel, sessions, manifest

    config = RankerConfig()
    prices, jpx_report = collect_jpx(jpx_directory)
    sessions = normalize_expected_sessions(sorted(prices["date"].unique()))
    modeling, coverage = prepare_modeling_prices(
        prices,
        coverage_lookback=config.source_coverage_lookback,
        minimum_source_coverage=config.minimum_source_coverage,
        expected_sessions=sessions,
    )
    panel = build_feature_panel(modeling, config)
    panel = add_bounded_daily_features(panel)
    panel = add_session_market_features(panel)
    panel["price_eligible"] = panel["eligible"].astype(bool)
    panel["price_training_eligible"] = panel["training_eligible"].astype(bool)
    panel = add_historical_candidate_features(panel)

    tdnet_dataset = collect_tdnet_dataset(
        tdnet_directory, allow_historical_provenance=True
    )
    tdnet_complete = tdnet_target_completeness(
        sessions,
        tdnet_dataset.complete_dates,
        tdnet_dataset.observed_at_by_date,
        decision_time=config.preopen.decision_time,
    )
    panel["tdnet_source_complete"] = panel["date"].map(tdnet_complete).eq(True)
    panel = attach_clean_tdnet_candidate_features(
        panel,
        tdnet_dataset.disclosures,
        sessions,
        decision_time=config.preopen.decision_time,
    )
    panel = add_event_context_interactions(panel)
    panel = _project_panel(panel, catalog_groups)
    panel_cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(panel, panel_cache, compress=0)
    manifest = {
        "schema_version": PANEL_SCHEMA_VERSION,
        "input_lock_sha256": input_lock_sha256,
        "panel_file_sha256": sha256_file(panel_cache),
        "rows": int(len(panel)),
        "codes": int(panel["code"].nunique()),
        "columns": list(panel.columns),
        "dtypes": {column: str(panel[column].dtype) for column in panel.columns},
        "sessions": [str(value.date()) for value in sessions],
        "calendar_sha256": session_calendar_hash(sessions),
        "jpx_report": jpx_report,
        "tdnet": {
            "source_sha256": tdnet_dataset.source_sha256,
            "source_files": tdnet_dataset.source_files,
            "disclosures": int(len(tdnet_dataset.disclosures)),
            "complete_through": str(tdnet_dataset.complete_dates.max().date()),
        },
        "price_coverage_incomplete_dates": [
            str(pd.Timestamp(value).date())
            for value in coverage.loc[~coverage["source_complete"], "date"]
        ],
    }
    write_json(manifest, manifest_path)
    return panel, sessions, manifest


def _anchor_spec(protocol: Mapping[str, Any]) -> ResearchModelSpec:
    raw = protocol["anchor_model"]
    return ResearchModelSpec(
        name="v06_feature_anchor",
        family=str(raw["family"]),
        objective=str(raw["objective"]),
        parameters={
            "C": float(raw["C"]),
            "class_balance": bool(raw["class_balance"]),
            "max_iter": 1_000,
        },
        random_state=int(raw["random_state"]),
    )


def _periods(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    output: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for period in pd.period_range(start.to_period("M"), end.to_period("M"), freq="M"):
        output.append(
            (
                max(start, period.start_time.normalize()),
                min(end, period.end_time.normalize()),
                str(period),
            )
        )
    return output


def _desired_slots(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.MultiIndex.from_product(
        [sessions, range(1, TOP_K + 1)], names=["date", "model_rank"]
    ).to_frame(index=False)


def _rank_top_two(
    scoring: pd.DataFrame, scores: np.ndarray, recipe_id: str
) -> pd.DataFrame:
    ranked = scoring.assign(model_score=np.asarray(scores, dtype=float)).sort_values(
        ["date", "model_score", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranked = ranked.groupby("date", sort=True, as_index=False).head(TOP_K).copy()
    ranked["model_rank"] = ranked.groupby("date", sort=False).cumcount() + 1
    ranked["recipe_id"] = recipe_id
    return ranked


def _uses_tdnet(columns: Sequence[str]) -> bool:
    return any(column.startswith("tdnet_clean_") for column in columns)


def _source_mask(
    panel: pd.DataFrame, columns: Sequence[str], *, training: bool
) -> pd.Series:
    base = panel[
        "price_training_eligible" if training else "price_eligible"
    ].eq(True)
    if _uses_tdnet(columns):
        base &= panel["tdnet_source_complete"].eq(True)
    return base


def _metrics(picks: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    displayed = picks["code"].notna()
    output["display"] = {
        "rank1_rate": float(displayed[picks["model_rank"].eq(1)].mean()),
        "rank2_rate": float(displayed[picks["model_rank"].eq(2)].mean()),
        "scheduled_sessions": int(picks["date"].nunique()),
    }
    for top_k in (1, 2):
        daily = daily_portfolio_returns(
            picks, top_k=top_k, cost_bps=PRIMARY_COST_BPS
        ).set_index("date")["net_return_pct"]
        interval = moving_block_bootstrap(
            daily,
            block_length=BLOCK_LENGTH,
            samples=BOOTSTRAP_SAMPLES,
            confidence=0.90,
            random_state=seed + top_k,
        )
        output[f"top{top_k}"] = {
            "net20": profit_metrics(
                picks, top_k=top_k, cost_bps=PRIMARY_COST_BPS
            ),
            "net40": profit_metrics(
                picks, top_k=top_k, cost_bps=STRESS_COST_BPS
            ),
            "block5_bootstrap": asdict(interval),
        }
    rank2 = picks[picks["model_rank"].eq(2)].copy()
    rank2["model_rank"] = 1
    output["rank2_standalone_net20"] = profit_metrics(
        rank2, top_k=1, cost_bps=PRIMARY_COST_BPS
    )
    return output


def evaluate_recipe(
    panel: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    spec: ResearchModelSpec,
    *,
    recipe_id: str,
    columns: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_start: pd.Timestamp,
    minimum_training_sessions: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    feature_columns = tuple(columns)
    scheduled = sessions[(sessions >= start) & (sessions <= end)]
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
                "price_eligible",
                "price_training_eligible",
                "tdnet_source_complete",
                *feature_columns,
            ]
        )
    )
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for score_start, score_end, period_name in _periods(start, end):
        training = panel.loc[
            panel["date"].between(
                train_start, score_start - pd.Timedelta(days=1)
            )
            & _source_mask(panel, feature_columns, training=True)
            & panel["label"].notna(),
            projection,
        ].copy()
        train_days = int(training["date"].nunique())
        if train_days < minimum_training_sessions:
            raise ValueError(
                f"{recipe_id} has only {train_days} training sessions before {score_start.date()}"
            )
        scoring = panel.loc[
            panel["date"].between(score_start, score_end)
            & _source_mask(panel, feature_columns, training=False),
            projection,
        ].copy()
        if not scoring.empty and not training["date"].max() < scoring["date"].min():
            raise AssertionError("training and scoring dates overlap")
        fitted = fit_research_model(spec, training, feature_columns)
        if not scoring.empty:
            parts.append(_rank_top_two(scoring, fitted.score(scoring), recipe_id))
        folds.append(
            {
                "period": period_name,
                "train_start": str(training["date"].min().date()),
                "train_end": str(training["date"].max().date()),
                "train_sessions": train_days,
                "train_rows": int(len(training)),
                "score_start": str(score_start.date()),
                "score_end": str(score_end.date()),
                "scheduled_sessions": int(
                    ((scheduled >= score_start) & (scheduled <= score_end)).sum()
                ),
                "scored_sessions": int(scoring["date"].nunique()),
                "score_rows": int(len(scoring)),
            }
        )
        del fitted, training, scoring
        gc.collect()
    actual = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if actual.empty:
        actual = pd.DataFrame(
            columns=[
                "date",
                "model_rank",
                "code",
                "name",
                "model_score",
                "label",
                "oc_return_pct",
            ]
        )
    keep = [
        value
        for value in (
            "date",
            "model_rank",
            "code",
            "name",
            "model_score",
            "label",
            "oc_return_pct",
            "outcome_observed",
            "source_complete",
            "universe_source_complete",
            "tdnet_source_complete",
        )
        if value in actual
    ]
    picks = _desired_slots(scheduled).merge(
        actual[keep],
        on=["date", "model_rank"],
        how="left",
        validate="one_to_one",
        sort=True,
    )
    picks["recipe_id"] = recipe_id
    result = {
        "recipe_id": recipe_id,
        "feature_count": len(feature_columns),
        "features": list(feature_columns),
        "period": {
            "start": str(start.date()),
            "end": str(end.date()),
            "scheduled_sessions": int(len(scheduled)),
        },
        "metrics": _metrics(picks, seed=DEFAULT_SEED),
        "folds": folds,
    }
    return result, picks


def _daily_net(picks: pd.DataFrame) -> pd.Series:
    return daily_portfolio_returns(
        picks, top_k=TOP_K, cost_bps=PRIMARY_COST_BPS
    ).set_index("date")["net_return_pct"].sort_index()


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
    indices = (starts[..., None] + offsets) % observations
    return indices.reshape(samples, -1)[:, :observations]


def max_t_adjusted_uplifts(
    baseline_picks: pd.DataFrame,
    candidate_picks: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, float]]:
    baseline = _daily_net(baseline_picks)
    names = list(candidate_picks)
    deltas: list[np.ndarray] = []
    for name in names:
        candidate = _daily_net(candidate_picks[name])
        paired = pd.concat([baseline.rename("base"), candidate.rename("candidate")], axis=1)
        if paired.isna().any(axis=None) or len(paired) != len(baseline):
            raise AssertionError("candidate and baseline scheduled days differ")
        deltas.append((paired["candidate"] - paired["base"]).to_numpy(dtype=float))
    matrix = np.column_stack(deltas)
    means = matrix.mean(axis=0)
    centered = matrix - means
    indices = _circular_block_indices(
        len(matrix),
        samples=BOOTSTRAP_SAMPLES,
        block_length=BLOCK_LENGTH,
        seed=DEFAULT_SEED,
    )
    bootstrap_centered = centered[indices].mean(axis=1)
    max_stat = bootstrap_centered.max(axis=1)
    critical = float(np.quantile(max_stat, MAX_T_CONFIDENCE))
    return {
        name: {
            "mean_uplift_pct": float(means[position]),
            "max_t_critical_pct": critical,
            "adjusted_one_sided_80pct_lower_pct": float(
                means[position] - critical
            ),
        }
        for position, name in enumerate(names)
    }


def _availability(
    panel: pd.DataFrame,
    columns: Sequence[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    rows = panel.loc[
        panel["date"].between(start, end)
        & _source_mask(panel, columns, training=False),
        list(columns),
    ]
    if rows.empty:
        return 0.0
    numeric = rows.apply(pd.to_numeric, errors="coerce")
    return float(np.isfinite(numeric.to_numpy()).all(axis=1).mean())


def _event_days(
    panel: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp
) -> int:
    rows = panel[
        panel["date"].between(start, end)
        & panel["tdnet_source_complete"].eq(True)
        & panel["tdnet_clean_any"].eq(1.0)
    ]
    return int(rows["date"].nunique())


def _label_blind_qa(
    panel: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    availability = {
        name: _availability(panel, columns, start=start, end=end)
        for name, columns in groups.items()
    }
    all_columns = list(
        dict.fromkeys(column for values in groups.values() for column in values)
    )
    source_rows = panel[
        panel["date"].between(start, end)
        & panel["price_training_eligible"].eq(True)
    ][all_columns]
    sampled = source_rows.sample(
        n=min(100_000, len(source_rows)), random_state=DEFAULT_SEED
    )
    correlations = sampled.corr(numeric_only=True).abs()
    redundant: list[dict[str, Any]] = []
    for left_index, left in enumerate(correlations.columns):
        for right in correlations.columns[left_index + 1 :]:
            value = correlations.loc[left, right]
            if pd.notna(value) and value >= 0.95:
                redundant.append(
                    {"left": left, "right": right, "abs_correlation": float(value)}
                )
    redundant.sort(key=lambda item: -item["abs_correlation"])
    return {
        "label_columns_accessed": False,
        "availability_by_group": availability,
        "sample_rows_for_redundancy": int(len(sampled)),
        "pairs_with_abs_correlation_at_least_0_95": redundant,
        "candidate_price_source_strictly_prior": bool(
            (
                panel["candidate_price_source_max_date"].isna()
                | (
                    panel["candidate_price_source_max_date"]
                    < panel["date"]
                )
            ).all()
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jpx-cache", default="research/.cache/model_v05_jpx"
    )
    parser.add_argument(
        "--tdnet-cache", default="research/.cache/model_v05_tdnet"
    )
    parser.add_argument(
        "--panel-cache", default="/tmp/model_v06_feature_panel.pkl"
    )
    parser.add_argument(
        "--output", default="research/model_v06_feature_result.json"
    )
    parser.add_argument(
        "--input-lock", default="research/model_v06_feature_input_lock.json"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    catalog, protocol = _load_frozen_definitions()
    groups = _catalog_groups(catalog)
    jpx_directory = Path(args.jpx_cache).resolve()
    tdnet_directory = Path(args.tdnet_cache).resolve()
    output_path = Path(args.output).resolve()
    input_lock_path = Path(args.input_lock).resolve()
    panel_cache = Path(args.panel_cache).resolve()
    lock = _input_lock(
        jpx_directory=jpx_directory, tdnet_directory=tdnet_directory
    )
    lock_sha = _write_or_validate_lock(input_lock_path, lock)

    panel, sessions, panel_manifest = build_candidate_panel(
        jpx_directory=jpx_directory,
        tdnet_directory=tdnet_directory,
        catalog_groups=groups,
        panel_cache=panel_cache,
        input_lock_sha256=lock_sha,
    )
    periods = {
        name: {
            "start": pd.Timestamp(raw["start"]),
            "end": pd.Timestamp(raw["end"]),
        }
        for name, raw in protocol["retrospective_periods"].items()
    }
    qa = _label_blind_qa(
        panel,
        groups,
        start=periods["group_screen"]["start"],
        end=periods["group_screen"]["end"],
    )
    if not qa["candidate_price_source_strictly_prior"]:
        raise LeakageError("v0.6 QA found a non-prior price source")

    spec = _anchor_spec(protocol)
    train_start = pd.Timestamp(protocol["anchor_model"]["training_window"].split()[-1])
    minimum_training_sessions = int(
        protocol["anchor_model"]["minimum_training_sessions"]
    )
    base_name = str(protocol["screen"]["base_group"])
    base_columns = groups[base_name]
    screen_period = periods["group_screen"]
    screen_results: dict[str, dict[str, Any]] = {}
    screen_picks: dict[str, pd.DataFrame] = {}
    base_result, base_picks = evaluate_recipe(
        panel,
        sessions,
        spec,
        recipe_id=base_name,
        columns=base_columns,
        start=screen_period["start"],
        end=screen_period["end"],
        train_start=train_start,
        minimum_training_sessions=minimum_training_sessions,
    )
    screen_results[base_name] = base_result
    screen_picks[base_name] = base_picks
    for group_name in protocol["screen"]["single_addition_groups"]:
        recipe_id = f"{base_name}+{group_name}"
        columns = tuple(dict.fromkeys((*base_columns, *groups[group_name])))
        result, picks = evaluate_recipe(
            panel,
            sessions,
            spec,
            recipe_id=recipe_id,
            columns=columns,
            start=screen_period["start"],
            end=screen_period["end"],
            train_start=train_start,
            minimum_training_sessions=minimum_training_sessions,
        )
        screen_results[group_name] = result
        screen_picks[group_name] = picks

    candidate_picks = {
        name: screen_picks[name]
        for name in protocol["screen"]["single_addition_groups"]
    }
    adjusted = max_t_adjusted_uplifts(base_picks, candidate_picks)
    base_top5 = base_result["metrics"]["top2"]["net20"][
        "top5_removed_net_mean_pct"
    ]
    thresholds = protocol["screen"]["all_required"]
    event_days = _event_days(
        panel, start=screen_period["start"], end=screen_period["end"]
    )
    qualification: dict[str, Any] = {}
    qualified: list[str] = []
    for group_name in protocol["screen"]["single_addition_groups"]:
        result = screen_results[group_name]
        metrics = result["metrics"]["top2"]["net20"]
        positive_month_fraction = metrics["positive_months"] / metrics["months"]
        availability = qa["availability_by_group"][group_name]
        sparse_event = group_name.startswith("T")
        checks = {
            "availability": availability >= float(
                thresholds["feature_availability_rate"]
            ),
            "positive_uplift": adjusted[group_name]["mean_uplift_pct"] > 0,
            "positive_month_fraction": positive_month_fraction
            >= float(thresholds["positive_month_fraction"]),
            "top5_removed_uplift": (
                metrics["top5_removed_net_mean_pct"] - base_top5
            )
            >= 0,
            "max_t_adjusted_lower_bound": adjusted[group_name][
                "adjusted_one_sided_80pct_lower_pct"
            ]
            >= 0,
            "event_days": (not sparse_event)
            or event_days
            >= int(thresholds["minimum_independent_event_days_for_sparse_event_group"]),
        }
        passed = all(checks.values())
        if passed:
            qualified.append(group_name)
        qualification[group_name] = {
            "passed": passed,
            "checks": checks,
            "availability_rate": availability,
            "positive_month_fraction": positive_month_fraction,
            "event_days": event_days if sparse_event else None,
            "top5_removed_uplift_pct": float(
                metrics["top5_removed_net_mean_pct"] - base_top5
            ),
            **adjusted[group_name],
        }
    qualified.sort(
        key=lambda name: (
            -qualification[name]["mean_uplift_pct"],
            -qualification[name]["adjusted_one_sided_80pct_lower_pct"],
            -qualification[name]["top5_removed_uplift_pct"],
            len(groups[name]),
            name,
        )
    )
    survivors = qualified[: int(protocol["screen"]["maximum_survivor_groups"])]
    point_leader = max(
        protocol["screen"]["single_addition_groups"],
        key=lambda name: screen_results[name]["metrics"]["top2"]["net20"][
            "net_mean_pct_at_cost"
        ],
    )

    union_period = periods["union_prune"]
    union_columns = tuple(
        dict.fromkeys(
            (
                *base_columns,
                *(column for group in survivors for column in groups[group]),
            )
        )
    )
    union_id = (
        f"{base_name}+" + "+".join(survivors) if survivors else base_name
    )
    union_result, union_picks = evaluate_recipe(
        panel,
        sessions,
        spec,
        recipe_id=union_id,
        columns=union_columns,
        start=union_period["start"],
        end=union_period["end"],
        train_start=train_start,
        minimum_training_sessions=minimum_training_sessions,
    )
    union_mean = union_result["metrics"]["top2"]["net20"]["net_mean_pct_at_cost"]
    leave_out_results: dict[str, Any] = {}
    retained: list[str] = []
    leave_out_picks: list[pd.DataFrame] = []
    for removed in survivors:
        remaining = [name for name in survivors if name != removed]
        columns = tuple(
            dict.fromkeys(
                (
                    *base_columns,
                    *(column for group in remaining for column in groups[group]),
                )
            )
        )
        recipe_id = f"{union_id}-without-{removed}"
        result, picks = evaluate_recipe(
            panel,
            sessions,
            spec,
            recipe_id=recipe_id,
            columns=columns,
            start=union_period["start"],
            end=union_period["end"],
            train_start=train_start,
            minimum_training_sessions=minimum_training_sessions,
        )
        without_mean = result["metrics"]["top2"]["net20"][
            "net_mean_pct_at_cost"
        ]
        keep = union_mean > without_mean
        if keep:
            retained.append(removed)
        leave_out_results[removed] = {
            "retained": keep,
            "union_minus_group_uplift_pct": float(union_mean - without_mean),
            "result": result,
        }
        leave_out_picks.append(picks)

    locked_groups = list(retained)
    locked_columns = tuple(
        dict.fromkeys(
            (
                *base_columns,
                *(column for group in locked_groups for column in groups[group]),
            )
        )
    )
    interaction_diagnostic: dict[str, Any] | None = None
    interaction_picks: pd.DataFrame | None = None
    if "T0_clean_event" in survivors:
        interaction_columns = tuple(
            dict.fromkeys((*locked_columns, *groups["X0_event_context"]))
        )
        result, picks = evaluate_recipe(
            panel,
            sessions,
            spec,
            recipe_id="locked_union+X0_event_context",
            columns=interaction_columns,
            start=union_period["start"],
            end=union_period["end"],
            train_start=train_start,
            minimum_training_sessions=minimum_training_sessions,
        )
        locked_without_x_result, _ = evaluate_recipe(
            panel,
            sessions,
            spec,
            recipe_id="locked_union_without_X0",
            columns=locked_columns,
            start=union_period["start"],
            end=union_period["end"],
            train_start=train_start,
            minimum_training_sessions=minimum_training_sessions,
        )
        with_x = result["metrics"]["top2"]["net20"]["net_mean_pct_at_cost"]
        without_x = locked_without_x_result["metrics"]["top2"]["net20"][
            "net_mean_pct_at_cost"
        ]
        interaction_kept = with_x > without_x
        if interaction_kept:
            locked_groups.append("X0_event_context")
            locked_columns = interaction_columns
        interaction_diagnostic = {
            "retained": interaction_kept,
            "uplift_pct": float(with_x - without_x),
            "result": result,
        }
        interaction_picks = picks

    locked_id = base_name + (
        "+" + "+".join(locked_groups) if locked_groups else ""
    )
    stability = periods["stability_report"]
    stability_result, stability_picks = evaluate_recipe(
        panel,
        sessions,
        spec,
        recipe_id=locked_id,
        columns=locked_columns,
        start=stability["start"],
        end=stability["end"],
        train_start=train_start,
        minimum_training_sessions=minimum_training_sessions,
    )
    stability_base_result, stability_base_picks = evaluate_recipe(
        panel,
        sessions,
        spec,
        recipe_id=f"{base_name}_stability_control",
        columns=base_columns,
        start=stability["start"],
        end=stability["end"],
        train_start=train_start,
        minimum_training_sessions=minimum_training_sessions,
    )
    stability_delta = float(
        stability_result["metrics"]["top2"]["net20"]["net_mean_pct_at_cost"]
        - stability_base_result["metrics"]["top2"]["net20"][
            "net_mean_pct_at_cost"
        ]
    )

    output_stem = output_path.with_suffix("")
    screen_pick_frame = pd.concat(
        [screen_picks[name] for name in screen_picks], ignore_index=True
    )
    union_frames = [union_picks, *leave_out_picks]
    if interaction_picks is not None:
        union_frames.append(interaction_picks)
    union_pick_frame = pd.concat(union_frames, ignore_index=True)
    stability_pick_frame = pd.concat(
        [stability_picks, stability_base_picks], ignore_index=True
    )
    screen_picks_path = Path(f"{output_stem}_group_screen_display_picks.csv")
    union_picks_path = Path(f"{output_stem}_union_prune_display_picks.csv")
    stability_picks_path = Path(f"{output_stem}_stability_display_picks.csv")
    trade_picks_path = Path(f"{output_stem}_trade_picks.csv")
    write_frame(screen_pick_frame, screen_picks_path)
    write_frame(union_pick_frame, union_picks_path)
    write_frame(stability_pick_frame, stability_picks_path)
    # Trade-gate design is intentionally deferred.  Keep a distinct, explicit
    # empty artifact rather than silently relabelling forced-display picks.
    write_frame(
        pd.DataFrame(
            columns=[
                "date",
                "model_rank",
                "code",
                "trade_eligible",
                "reason",
            ]
        ),
        trade_picks_path,
    )

    result_payload: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "authority": protocol["authority"],
        "decision": "retrospective_feature_diagnostic_only",
        "input_lock": {
            "path": str(input_lock_path.relative_to(ROOT)),
            "sha256": lock_sha,
        },
        "panel": {
            "cache_path": str(panel_cache),
            "manifest_path": str(_panel_manifest_path(panel_cache)),
            "manifest_sha256": sha256_file(_panel_manifest_path(panel_cache)),
            "file_sha256": panel_manifest["panel_file_sha256"],
            "rows": panel_manifest["rows"],
            "codes": panel_manifest["codes"],
            "calendar_sha256": panel_manifest["calendar_sha256"],
        },
        "anchor_model": spec.canonical_dict(),
        "label_blind_qa": qa,
        "group_screen": {
            "base": base_result,
            "candidates": {
                name: screen_results[name]
                for name in protocol["screen"]["single_addition_groups"]
            },
            "qualification": qualification,
            "formal_survivors": survivors,
            "point_estimate_leader_hypothesis_only": point_leader,
        },
        "union_prune": {
            "initial_survivor_union": union_result,
            "leave_one_group_out": leave_out_results,
            "retained_groups_after_one_pass": retained,
            "interaction_diagnostic": interaction_diagnostic,
            "locked_retrospective_groups": locked_groups,
            "locked_retrospective_recipe_id": locked_id,
            "locked_features": list(locked_columns),
        },
        "stability_report": {
            "locked_recipe": stability_result,
            "price_core_control": stability_base_result,
            "top2_net20_delta_vs_price_core_pct": stability_delta,
            "is_sealed_holdout": False,
        },
        "artifacts": {
            "group_screen_display_picks": {
                "path": str(screen_picks_path.relative_to(ROOT)),
                "sha256": sha256_file(screen_picks_path),
            },
            "union_prune_display_picks": {
                "path": str(union_picks_path.relative_to(ROOT)),
                "sha256": sha256_file(union_picks_path),
            },
            "stability_display_picks": {
                "path": str(stability_picks_path.relative_to(ROOT)),
                "sha256": sha256_file(stability_picks_path),
            },
            "trade_picks": {
                "path": str(trade_picks_path.relative_to(ROOT)),
                "sha256": sha256_file(trade_picks_path),
                "status": "empty_by_protocol_until_out_of_fold_trade_gate_is_designed",
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": protocol["data"]["known_limits"],
        "prospective_plan": protocol["prospective_plan"],
    }
    write_json(result_payload, output_path)
    manifest = {
        "schema_version": 1,
        "result_path": str(output_path.relative_to(ROOT)),
        "result_sha256": sha256_file(output_path),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "catalog_sha256": EXPECTED_CATALOG_SHA256,
        "input_lock_sha256": lock_sha,
        "implementation_sha256": _implementation_hashes(),
        "artifacts": result_payload["artifacts"],
    }
    write_json(manifest, output_path.with_suffix(".manifest.json"))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
