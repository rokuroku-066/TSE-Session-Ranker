#!/usr/bin/env python3
"""Build a memory-bounded v0.4 panel after the formal runtime failure.

This program has no estimator or return-metric code.  It reuses the exact
frozen feature functions, but releases each large intermediate before starting
the next stage and removes provenance/name columns after their source hashes
have been verified.  The output is eligible only for a clearly labelled
post-failure diagnostic, never for a formal sealed-holdout claim.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research import finalize_logit_v04 as base  # noqa: E402
from research import recover_logit_v04_holdout as formal  # noqa: E402


FORMAL_LOCK_PATH = ROOT / "research/model_v04_parser_recovery_lock.json"
FORMAL_LOCK_SHA256 = (
    "1f87a81e21175dc52c82f8a42762ab52cfd8ab39a3b18bcdb6061d908ab95554"
)
FORMAL_RECEIPT_PATH = ROOT / "research/model_v04_parser_recovery_consumed.json"
FORMAL_RECEIPT_SHA256 = (
    "dfaaa3cf94614e76a5bc826386d63380cbcd0473523744d2571dce98125587e2"
)
RUNTIME_FAILURE_PATH = (
    ROOT / "research/model_v04_holdout_recovery_runtime_failure.json"
)
RUNTIME_FAILURE_SHA256 = (
    "f34912dbc9c9c787d10caffc9766e132f6527f3ccc1ab3acb66a9145286bb5c3"
)
PANEL_PATH = Path("/tmp/model_v04_post_failure_diagnostic_panel.pkl")
MANIFEST_PATH = PANEL_PATH.with_suffix(PANEL_PATH.suffix + ".manifest.json")

PRICE_INPUT_COLUMNS = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    *base.SESSION_OHLC,
    "volume",
    "turnover",
    "traded",
    "partial_session",
)
EVALUATION_COLUMNS = (
    "date",
    "code",
    "eligible",
    "training_eligible",
    "label",
    "oc_return_pct",
    "outcome_observed",
    "source_complete",
    "universe_source_complete",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    values = pd.util.hash_pandas_object(frame[list(columns)], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def load_minimal_prices(path: str | Path) -> pd.DataFrame:
    full = base.read_frame(path)
    missing = sorted(set(PRICE_INPUT_COLUMNS) - set(full.columns))
    if missing:
        raise ValueError(f"price input lacks frozen columns: {missing}")
    minimal = full.loc[:, list(PRICE_INPUT_COLUMNS)].copy()
    del full
    gc.collect()
    return minimal


def build_panel_memory_bounded(
    prices: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    disclosures: pd.DataFrame,
    tdnet_complete_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = base.RankerConfig()
    calendar = base.normalize_expected_sessions(sessions)
    canonical = base.normalize_daily_prices(prices)
    del prices
    gc.collect()

    modeling, coverage = base.prepare_modeling_prices(
        canonical,
        coverage_lookback=settings.source_coverage_lookback,
        minimum_source_coverage=settings.minimum_source_coverage,
        expected_sessions=calendar,
    )
    del canonical
    gc.collect()

    panel = base.build_feature_panel(modeling, settings)
    del modeling
    gc.collect()

    next_panel = base.add_bounded_daily_features(panel)
    del panel
    gc.collect()
    panel = next_panel
    del next_panel
    gc.collect()

    next_panel = base.add_session_market_features(panel)
    del panel
    gc.collect()
    panel = next_panel
    del next_panel
    gc.collect()

    winner = base.ModelSpec.from_dict(
        json.loads(FORMAL_LOCK_PATH.read_text(encoding="utf-8"))["winner"]
    )
    control = formal.baseline_spec()
    frozen_features = tuple(
        dict.fromkeys(
            [
                *base.feature_blocks()[winner.feature_block],
                *base.feature_blocks()[control.feature_block],
            ]
        )
    )
    pre_tdnet_columns = [
        column
        for column in (*EVALUATION_COLUMNS, *frozen_features)
        if column in panel.columns
    ]
    attach_dependencies = {
        "date",
        "code",
        "eligible",
        "training_eligible",
        "xrank_oc_mean_20",
        "market_beta_x_prior_market_return",
    }
    missing_dependencies = sorted(attach_dependencies - set(pre_tdnet_columns))
    if missing_dependencies:
        raise RuntimeError(
            f"pre-TDnet projection lacks dependencies: {missing_dependencies}"
        )
    panel = panel.loc[:, list(dict.fromkeys(pre_tdnet_columns))].copy()
    gc.collect()

    next_panel = base.attach_tdnet_features(
        panel,
        disclosures,
        calendar,
        base.normalize_expected_sessions(tdnet_complete_dates),
        decision_time=settings.preopen.decision_time,
    )
    del panel, disclosures
    gc.collect()
    panel = next_panel
    del next_panel
    gc.collect()

    coverage["tdnet_source_complete"] = coverage["date"].map(
        panel.groupby("date", sort=False)["tdnet_source_complete"].first()
    ).eq(True)
    coverage["model_source_complete"] = (
        coverage["source_complete"] & coverage["tdnet_source_complete"]
    )
    required = sorted(set(frozen_features) - set(panel.columns))
    if required:
        raise RuntimeError(f"diagnostic panel lacks frozen features: {required}")
    keep = list(dict.fromkeys([*EVALUATION_COLUMNS, *frozen_features]))
    panel = panel.loc[panel["date"].ge(base.TRAIN_START), keep].copy()
    warmup_only = panel["date"].eq(pd.Timestamp("2025-08-01"))
    panel.loc[warmup_only, "training_eligible"] = False
    if panel.loc[warmup_only, "training_eligible"].any():
        raise AssertionError("holdout warmup entered diagnostic model training")
    return panel, coverage


def main() -> None:
    if PANEL_PATH.exists() or MANIFEST_PATH.exists():
        raise FileExistsError("diagnostic panel output already exists")
    lock, selection, _, sessions = formal.load_and_verify_lock(
        FORMAL_LOCK_PATH,
        FORMAL_LOCK_SHA256,
        require_outputs_absent=False,
    )
    if sha256_file(FORMAL_RECEIPT_PATH) != FORMAL_RECEIPT_SHA256:
        raise ValueError("formal recovery receipt changed")
    if sha256_file(RUNTIME_FAILURE_PATH) != RUNTIME_FAILURE_SHA256:
        raise ValueError("formal runtime failure record changed")
    failure = json.loads(RUNTIME_FAILURE_PATH.read_text(encoding="utf-8"))
    if (
        failure.get("formal_metric_run_count") != 0
        or failure.get("model_evaluation_started") is not False
        or failure.get("result_output_exists") is not False
    ):
        raise ValueError("runtime failure no longer proves zero model evaluations")
    for path in (
        lock["outputs"]["result"],
        lock["outputs"]["winner_picks"],
        lock["outputs"]["v03_control_picks"],
    ):
        if Path(path).exists():
            raise ValueError("formal result or picks unexpectedly exist")
    holdout_path = Path(failure["holdout_daily_export_path"])
    holdout_manifest_path = Path(failure["holdout_daily_manifest_path"])
    if sha256_file(holdout_path) != failure["holdout_daily_export_sha256"]:
        raise ValueError("formal parsed holdout export changed")
    if sha256_file(holdout_manifest_path) != failure[
        "holdout_daily_manifest_sha256"
    ]:
        raise ValueError("formal parsed holdout manifest changed")

    pd.options.mode.copy_on_write = True
    development = load_minimal_prices(selection["data"]["daily_path"])
    blind = load_minimal_prices(holdout_path)
    keys = ["date", "code"]
    if development.duplicated(keys).any() or blind.duplicated(keys).any():
        raise ValueError("metric-only price projection has duplicate date/code keys")
    if not pd.Timestamp(development["date"].max()) < pd.Timestamp(
        blind["date"].min()
    ):
        raise ValueError("development and holdout date ranges overlap")
    prices = base.merge_daily_prices([development, blind])
    del development, blind
    gc.collect()

    data = selection["data"]
    disclosures, tdnet_complete_dates = base._load_tdnet(
        [*data["tdnet_paths"], *data["holdout_tdnet_paths_sealed"]],
        [
            *data["tdnet_manifest_paths"],
            *data["holdout_tdnet_manifest_paths_sealed"],
        ],
    )
    disclosures = disclosures[
        disclosures["published_at"].dt.tz_localize(None).dt.normalize().le(
            base.HOLDOUT_END
        )
    ].copy()
    print("build memory-bounded diagnostic panel", flush=True)
    price_holder = [prices]
    disclosure_holder = [disclosures]
    del prices, disclosures
    panel, coverage = build_panel_memory_bounded(
        price_holder.pop(),
        sessions,
        disclosure_holder.pop(),
        tdnet_complete_dates,
    )
    if panel.duplicated(["date", "code"]).any():
        raise AssertionError("diagnostic panel has duplicate date/code keys")
    expected_order = panel[["code", "date"]].sort_values(
        ["code", "date"], kind="stable"
    ).reset_index(drop=True)
    if not panel[["code", "date"]].reset_index(drop=True).equals(expected_order):
        raise AssertionError("diagnostic panel changed canonical code/date order")
    del expected_order
    gc.collect()
    written = base.write_frame(panel, PANEL_PATH)
    formal_lock = json.loads(FORMAL_LOCK_PATH.read_text(encoding="utf-8"))
    winner = base.ModelSpec.from_dict(formal_lock["winner"])
    control = formal.baseline_spec()
    winner_features = list(base.feature_blocks()[winner.feature_block])
    control_features = list(base.feature_blocks()[control.feature_block])
    eligibility = (
        panel.groupby("date", sort=True)[["eligible", "training_eligible"]]
        .sum()
        .astype(int)
        .reset_index()
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "post_failure_diagnostic_panel",
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "formal_status": "not_a_formal_holdout_result",
        "formal_lock_sha256": FORMAL_LOCK_SHA256,
        "formal_receipt_sha256": FORMAL_RECEIPT_SHA256,
        "formal_runtime_failure_sha256": RUNTIME_FAILURE_SHA256,
        "builder_path": str(Path(__file__).resolve()),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "source_development_daily_sha256": sha256_file(
            selection["data"]["daily_path"]
        ),
        "source_holdout_daily_sha256": sha256_file(holdout_path),
        "source_holdout_manifest_sha256": sha256_file(holdout_manifest_path),
        "panel_path": str(written),
        "panel_sha256": sha256_file(written),
        "panel_content_sha256": frame_hash(panel, panel.columns),
        "ordered_date_code_sha256": frame_hash(panel, ("date", "code")),
        "rows": int(len(panel)),
        "columns": list(panel.columns),
        "dtypes": {column: str(dtype) for column, dtype in panel.dtypes.items()},
        "min_date": str(panel["date"].min().date()),
        "max_date": str(panel["date"].max().date()),
        "coverage_source_incomplete_dates": [
            str(value.date())
            for value in coverage.loc[~coverage["source_complete"], "date"]
        ],
        "coverage_tdnet_incomplete_dates": [
            str(value.date())
            for value in coverage.loc[
                ~coverage["tdnet_source_complete"], "date"
            ]
        ],
        "coverage_model_incomplete_dates": [
            str(value.date())
            for value in coverage.loc[
                ~coverage["model_source_complete"], "date"
            ]
        ],
        "coverage_content_sha256": frame_hash(coverage, coverage.columns),
        "per_date_eligibility": [
            {
                "date": str(row.date.date()),
                "eligible": int(row.eligible),
                "training_eligible": int(row.training_eligible),
            }
            for row in eligibility.itertuples(index=False)
        ],
        "winner_spec_id": winner.id,
        "winner_features": winner_features,
        "v03_control_spec_id": control.id,
        "v03_control_features": control_features,
        "base_implementation_sha256": base._implementation_hashes(),
        "pandas_copy_on_write": True,
        "date_code_unique": True,
        "canonical_code_date_order": True,
        "warmup_training_ineligible": True,
        "memory_only_changes": [
            "drop source provenance and company name after hash verification; metric-only equivalence, not full-panel identity",
            "release large intermediate frames between frozen feature functions",
            "enable pandas copy-on-write without changing formulas",
            "project to the union of frozen winner and v0.3 control columns",
        ],
        "feature_functions_changed": False,
        "model_evaluation_code_present": False,
        "holdout_labels_materialized": True,
        "per_row_returns_persisted": True,
        "aggregate_return_metrics_computed": False,
        "model_evaluation_count": 0,
        "individual_returns_printed_or_manually_inspected": False,
    }
    base.write_json(manifest, MANIFEST_PATH)
    print(f"panel={written}", flush=True)
    print(f"manifest={MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
