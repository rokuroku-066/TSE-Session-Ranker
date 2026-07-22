#!/usr/bin/env python3
"""Run the frozen v0.6 T1 row-alignment correction diagnostic.

This is a correction retrospective, not a feature-selection or production
promotion run.  It reuses the exact v0.6 raked-logit evaluator and monthly
walk-forward folds, first reproduces the frozen G0 and legacy-T1 references,
then replaces only the two registered T1 columns with the sparse correction
overlay.  All inputs and coverage counts fail closed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research.screen_feature_candidates_v06 import (  # noqa: E402
    _anchor_spec,
    _catalog_groups,
    evaluate_recipe,
    max_t_adjusted_uplifts,
    sha256_file,
)
from tse_session_ranker.data.common import normalize_expected_sessions  # noqa: E402
from tse_session_ranker.exceptions import DataValidationError  # noqa: E402
from tse_session_ranker.io import write_frame, write_json  # noqa: E402
from tse_session_ranker.profit import daily_portfolio_returns  # noqa: E402
from tse_session_ranker.validation import (  # noqa: E402
    paired_moving_block_bootstrap,
)


PROTOCOL_PATH = ROOT / "research/model_v06_t1_correction_protocol.json"
EXPECTED_PROTOCOL_SHA256 = (
    "939fe5a330cb973f3a43bbf5d2302ed9495093075acbb8ed98624e85a0e2e4d6"
)
RESULT_SCHEMA_VERSION = 1
PRIMARY_COST_BPS = 20.0
TOP_K = 2
BLOCK_LENGTH = 5
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_CONFIDENCE = 0.80
DEFAULT_SEED = 31


def _resolve_registered_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_PROTOCOL_SHA256:
        raise DataValidationError("v0.6 T1 correction protocol hash changed")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise DataValidationError("unsupported v0.6 T1 correction protocol")
    if protocol.get("status") != "frozen_before_corrected_t1_metrics":
        raise DataValidationError("v0.6 T1 correction protocol is not frozen")
    if protocol["authority"]["production_promotion_allowed"] is not False:
        raise DataValidationError("correction diagnostic cannot promote production")
    return protocol


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise DataValidationError(f"missing frozen {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise DataValidationError(
            f"frozen {label} hash mismatch: expected {expected}, got {actual}"
        )


def _validate_frozen_file_hashes(protocol: Mapping[str, Any]) -> None:
    for label, spec in protocol["frozen_inputs"].items():
        path = _resolve_registered_path(str(spec["path"]))
        _require_hash(path, str(spec["sha256"]), label)
        if "manifest_path" in spec:
            manifest_path = _resolve_registered_path(str(spec["manifest_path"]))
            _require_hash(
                manifest_path,
                str(spec["manifest_sha256"]),
                f"{label} manifest",
            )


def _validate_panel(
    panel: pd.DataFrame,
    manifest: Mapping[str, Any],
    panel_spec: Mapping[str, Any],
    *,
    required_columns: Sequence[str],
) -> None:
    if not isinstance(panel, pd.DataFrame):
        raise DataValidationError("v0.6 panel is not a DataFrame")
    if len(panel) != int(panel_spec["rows"]):
        raise DataValidationError("v0.6 panel row count changed")
    if int(manifest.get("rows", -1)) != len(panel):
        raise DataValidationError("v0.6 panel manifest row count mismatch")
    if manifest.get("panel_file_sha256") != panel_spec["sha256"]:
        raise DataValidationError("v0.6 panel manifest binds another panel")
    if list(panel.columns) != list(manifest.get("columns", [])):
        raise DataValidationError("v0.6 panel columns differ from its manifest")
    dtypes = {column: str(panel[column].dtype) for column in panel.columns}
    if dtypes != manifest.get("dtypes"):
        raise DataValidationError("v0.6 panel dtypes differ from its manifest")
    missing = sorted(set(required_columns) - set(panel.columns))
    if missing:
        raise DataValidationError(f"v0.6 panel lacks columns: {missing}")
    keys = list(panel_spec["unique_key"])
    if panel[keys].isna().any(axis=None):
        raise DataValidationError("v0.6 panel has null date/code keys")
    if panel.duplicated(keys).any():
        raise DataValidationError("v0.6 panel date/code keys are not unique")


def validate_overlay_schema(
    overlay: pd.DataFrame,
    spec: Mapping[str, Any],
) -> None:
    """Validate the overlay itself, without consulting target outcomes."""

    if not isinstance(overlay, pd.DataFrame):
        raise DataValidationError("T1 correction overlay is not a DataFrame")
    if list(overlay.columns) != list(spec["columns"]):
        raise DataValidationError("T1 correction overlay schema changed")
    actual_dtypes = {column: str(overlay[column].dtype) for column in overlay}
    if actual_dtypes != dict(spec["dtypes"]):
        raise DataValidationError("T1 correction overlay dtypes changed")
    if len(overlay) != int(spec["rows"]):
        raise DataValidationError("T1 correction overlay row count changed")
    keys = list(spec["unique_key"])
    if overlay[keys].isna().any(axis=None):
        raise DataValidationError("T1 correction overlay has null keys")
    if overlay.duplicated(keys).any():
        raise DataValidationError("T1 correction overlay has duplicate keys")
    if not overlay["date"].dt.normalize().equals(overlay["date"]):
        raise DataValidationError("T1 correction overlay dates are not normalized")
    if str(overlay["date"].min().date()) != str(spec["date_min"]):
        raise DataValidationError("T1 correction overlay minimum date changed")
    if str(overlay["date"].max().date()) != str(spec["date_max"]):
        raise DataValidationError("T1 correction overlay maximum date changed")
    if not overlay["code"].map(lambda value: isinstance(value, str)).all():
        raise DataValidationError("T1 correction overlay codes must be strings")

    family = overlay["tdnet_clean_family_count_log1p"].to_numpy(dtype=float)
    single = overlay["tdnet_clean_single_family"].to_numpy(dtype=float)
    if not np.isfinite(family).all() or not np.isfinite(single).all():
        raise DataValidationError("T1 correction overlay contains non-finite values")
    counts = np.expm1(family)
    rounded = np.rint(counts)
    if (counts < -1e-6).any() or not np.allclose(
        counts, rounded, rtol=0.0, atol=1e-5
    ):
        raise DataValidationError("family count is not log1p(nonnegative integer)")
    expected_single = rounded == 1.0
    if not np.isin(single, [0.0, 1.0]).all() or not np.array_equal(
        single == 1.0, expected_single
    ):
        raise DataValidationError("single-family flag disagrees with family count")


def _scope_coverage(
    panel: pd.DataFrame,
    overlay: pd.DataFrame,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> dict[str, int]:
    keys = ["date", "code"]
    corrected = [
        "tdnet_clean_family_count_log1p",
        "tdnet_clean_single_family",
    ]
    panel_scope = panel
    overlay_scope = overlay
    if start is not None and end is not None:
        panel_scope = panel.loc[panel["date"].between(start, end)]
        overlay_scope = overlay.loc[overlay["date"].between(start, end)]
    merged = panel_scope[
        [*keys, "tdnet_source_complete", *corrected]
    ].merge(
        overlay_scope,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_legacy", "_overlay"),
        sort=False,
    )
    covered = merged["_merge"].eq("both")
    legacy_event = (
        merged[f"{corrected[0]}_legacy"].fillna(0.0).gt(0.0)
        | merged[f"{corrected[1]}_legacy"].fillna(0.0).gt(0.0)
    )
    return {
        "panel_rows": int(len(panel_scope)),
        "panel_dates": int(panel_scope["date"].nunique()),
        "tdnet_complete_rows": int(panel_scope["tdnet_source_complete"].eq(True).sum()),
        "tdnet_incomplete_rows": int(panel_scope["tdnet_source_complete"].eq(False).sum()),
        "overlay_rows": int(len(overlay_scope)),
        "overlay_keys_matched_to_panel": int(covered.sum()),
        "overlay_keys_outside_panel": int(len(overlay_scope) - covered.sum()),
        "legacy_event_keys": int(legacy_event.sum()),
        "legacy_event_keys_covered": int((legacy_event & covered).sum()),
        "complete_rows_without_overlay_key": int(
            (merged["tdnet_source_complete"].eq(True) & ~covered).sum()
        ),
    }


def validate_registered_coverage(
    panel: pd.DataFrame,
    overlay: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    scopes = {
        "full_panel": _scope_coverage(panel, overlay),
        "model_horizon_2024_01_04_through_2024_10_31": _scope_coverage(
            panel,
            overlay,
            start=pd.Timestamp("2024-01-04"),
            end=pd.Timestamp("2024-10-31"),
        ),
        "group_screen_2024_07_01_through_2024_10_31": _scope_coverage(
            panel,
            overlay,
            start=pd.Timestamp("2024-07-01"),
            end=pd.Timestamp("2024-10-31"),
        ),
    }
    registered = protocol["registered_overlay_coverage"]
    for scope_name, expected in registered.items():
        actual = scopes[scope_name]
        for key, value in expected.items():
            if actual.get(key) != int(value):
                raise DataValidationError(
                    f"overlay coverage changed at {scope_name}.{key}: "
                    f"expected {value}, got {actual.get(key)}"
                )
        if actual["legacy_event_keys"] != actual["legacy_event_keys_covered"]:
            raise DataValidationError(
                f"overlay misses a legacy event key in {scope_name}"
            )
    return scopes


def apply_corrected_overlay(
    panel: pd.DataFrame,
    overlay: pd.DataFrame,
    *,
    corrected_columns: Sequence[str] = (
        "tdnet_clean_family_count_log1p",
        "tdnet_clean_single_family",
    ),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace only registered columns and preserve TDnet missingness semantics."""

    keys = ["date", "code"]
    if panel.duplicated(keys).any() or overlay.duplicated(keys).any():
        raise DataValidationError("panel/overlay keys must be unique before merge")
    required = {*keys, "tdnet_source_complete", *corrected_columns}
    if missing := sorted(required - set(panel.columns)):
        raise DataValidationError(f"panel lacks correction columns: {missing}")
    if missing := sorted(
        (set(keys) | set(corrected_columns)) - set(overlay.columns)
    ):
        raise DataValidationError(f"overlay lacks correction columns: {missing}")

    aligned = panel[keys].merge(
        overlay[[*keys, *corrected_columns]],
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    covered = aligned["_merge"].eq("both")
    legacy_event = panel[list(corrected_columns)].fillna(0.0).gt(0.0).any(axis=1)
    if (legacy_event.to_numpy() & ~covered.to_numpy()).any():
        raise DataValidationError("overlay does not cover every legacy event key")

    complete = panel["tdnet_source_complete"].eq(True).to_numpy()
    result = panel.copy(deep=False)
    changed_counts: dict[str, int] = {}
    for column in corrected_columns:
        values = pd.to_numeric(aligned[column], errors="coerce").to_numpy(dtype=float)
        values = np.where(complete, np.nan_to_num(values, nan=0.0), np.nan)
        old = pd.to_numeric(panel[column], errors="coerce").to_numpy(dtype=float)
        changed_counts[column] = int(
            (~np.isclose(old, values, rtol=0.0, atol=0.0, equal_nan=True)).sum()
        )
        result[column] = values.astype("float32")

    corrected = result[list(corrected_columns)]
    if corrected.loc[result["tdnet_source_complete"].eq(True)].isna().any(axis=None):
        raise DataValidationError("complete TDnet rows became missing after correction")
    if corrected.loc[result["tdnet_source_complete"].eq(False)].notna().any(axis=None):
        raise DataValidationError("incomplete TDnet rows became finite after correction")
    report = {
        "overlay_keys_matched": int(covered.sum()),
        "complete_rows_implicit_no_event_zero": int((complete & ~covered.to_numpy()).sum()),
        "incomplete_rows_forced_nan": int((~complete).sum()),
        **{f"changed_rows:{key}": value for key, value in changed_counts.items()},
    }
    return result, report


def assert_feature_isolation(
    legacy: pd.DataFrame,
    corrected: pd.DataFrame,
    *,
    unchanged_columns: Sequence[str],
) -> None:
    if not legacy.index.equals(corrected.index):
        raise DataValidationError("correction changed the panel row index")
    for column in unchanged_columns:
        left = legacy[column].to_numpy()
        right = corrected[column].to_numpy()
        if not np.array_equal(left, right, equal_nan=True):
            raise DataValidationError(f"correction changed unregistered column {column}")


def _numeric_differences(expected: Any, actual: Any) -> list[float]:
    values: list[float] = []
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            raise DataValidationError("reference metric structure changed")
        for key in expected:
            values.extend(_numeric_differences(expected[key], actual[key]))
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not isinstance(expected, bool) and not isinstance(actual, bool):
            left, right = float(expected), float(actual)
            if np.isfinite(left) and np.isfinite(right):
                values.append(abs(left - right))
    return values


def compare_reference_recipe(
    result: Mapping[str, Any],
    picks: pd.DataFrame,
    reference_result: Mapping[str, Any],
    reference_picks: pd.DataFrame,
    *,
    tolerance: float,
) -> dict[str, Any]:
    numeric = _numeric_differences(reference_result["metrics"], result["metrics"])
    max_metric_difference = max(numeric, default=0.0)
    if max_metric_difference > tolerance:
        raise DataValidationError("rerun metrics do not reproduce frozen v0.6")

    keys = ["date", "model_rank"]
    compare_columns = [
        "code",
        "name",
        "label",
        "oc_return_pct",
        "outcome_observed",
        "source_complete",
        "universe_source_complete",
        "tdnet_source_complete",
    ]
    left = reference_picks.sort_values(keys, kind="stable").reset_index(drop=True)
    right = picks.sort_values(keys, kind="stable").reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise DataValidationError("rerun scheduled slots differ from frozen v0.6")
    for column in compare_columns:
        left_values = left[column]
        right_values = right[column]
        if column in {"label", "oc_return_pct"}:
            equal = np.allclose(
                pd.to_numeric(left_values, errors="coerce"),
                pd.to_numeric(right_values, errors="coerce"),
                rtol=0.0,
                atol=tolerance,
                equal_nan=True,
            )
        else:
            equal = left_values.fillna("<NA>").astype(str).equals(
                right_values.fillna("<NA>").astype(str)
            )
        if not equal:
            raise DataValidationError(f"rerun pick column differs: {column}")
    score_difference = np.nanmax(
        np.abs(
            pd.to_numeric(left["model_score"], errors="coerce").to_numpy()
            - pd.to_numeric(right["model_score"], errors="coerce").to_numpy()
        )
    )
    if float(score_difference) > tolerance:
        raise DataValidationError("rerun model scores differ from frozen v0.6")
    return {
        "recipe_id": str(result["recipe_id"]),
        "scheduled_slots": int(len(right)),
        "maximum_absolute_metric_difference": float(max_metric_difference),
        "maximum_absolute_score_difference": float(score_difference),
        "reproduced": True,
    }


def _daily_net(picks: pd.DataFrame) -> pd.Series:
    return daily_portfolio_returns(
        picks, top_k=TOP_K, cost_bps=PRIMARY_COST_BPS
    ).set_index("date")["net_return_pct"].sort_index()


def _pick_change_report(
    legacy: pd.DataFrame, corrected: pd.DataFrame
) -> dict[str, Any]:
    keys = ["date", "model_rank"]
    merged = legacy[keys + ["code"]].merge(
        corrected[keys + ["code"]],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_legacy", "_corrected"),
    )
    same = merged["code_legacy"].fillna("<NA>").eq(
        merged["code_corrected"].fillna("<NA>")
    )
    changed_dates = merged.loc[~same, "date"].nunique()
    return {
        "scheduled_slots": int(len(merged)),
        "changed_slots": int((~same).sum()),
        "unchanged_slots": int(same.sum()),
        "dates_with_any_changed_slot": int(changed_dates),
        "slot_change_rate": float((~same).mean()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="/tmp/model_v06_feature_panel.pkl")
    parser.add_argument(
        "--overlay", default="/tmp/model_v07_tdnet_alignment_overlay.pkl"
    )
    parser.add_argument(
        "--output", default="research/model_v06_t1_correction_result.json"
    )
    return parser.parse_args()


def main() -> None:
    run_started_at = datetime.now(ZoneInfo("Asia/Tokyo"))
    args = _parse_args()
    protocol = load_protocol()
    registered_at = pd.Timestamp(protocol["registered_at"])
    if registered_at.tzinfo is None:
        raise DataValidationError("protocol registered_at must be timezone-aware")
    if registered_at >= pd.Timestamp(run_started_at):
        raise DataValidationError("protocol must be registered before the run starts")
    _validate_frozen_file_hashes(protocol)
    frozen = protocol["frozen_inputs"]
    panel_path = Path(args.panel).resolve()
    overlay_path = Path(args.overlay).resolve()
    if panel_path != _resolve_registered_path(frozen["v06_panel"]["path"]):
        raise DataValidationError("runner panel path differs from registered path")
    if overlay_path != _resolve_registered_path(frozen["alignment_overlay"]["path"]):
        raise DataValidationError("runner overlay path differs from registered path")

    catalog = json.loads(
        _resolve_registered_path(frozen["v06_catalog"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    v06_protocol = json.loads(
        _resolve_registered_path(frozen["v06_protocol"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    groups = _catalog_groups(catalog)
    base_columns = tuple(groups["G0_price_core"])
    t1_columns = tuple(groups["T1_event_structure"])
    registered_correction = protocol["correction"]
    if tuple(registered_correction["replace_columns"]) != (
        "tdnet_clean_family_count_log1p",
        "tdnet_clean_single_family",
    ):
        raise DataValidationError("registered correction columns changed")
    if set(registered_correction["unchanged_t1_columns"]) != (
        set(t1_columns) - set(registered_correction["replace_columns"])
    ):
        raise DataValidationError("registered unchanged T1 columns are incomplete")

    panel_manifest_path = _resolve_registered_path(
        frozen["v06_panel"]["manifest_path"]
    )
    panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    raw_panel = joblib.load(panel_path)
    required_metadata = (
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
        "tdnet_clean_any",
    )
    _validate_panel(
        raw_panel,
        panel_manifest,
        frozen["v06_panel"],
        required_columns=(*required_metadata, *base_columns, *t1_columns),
    )
    projection = list(
        dict.fromkeys((*required_metadata, *base_columns, *t1_columns))
    )
    panel = raw_panel.loc[:, projection].copy()
    del raw_panel
    gc.collect()

    overlay = joblib.load(overlay_path)
    validate_overlay_schema(overlay, frozen["alignment_overlay"])
    coverage = validate_registered_coverage(panel, overlay, protocol)
    corrected_panel, correction_report = apply_corrected_overlay(panel, overlay)
    assert_feature_isolation(
        panel,
        corrected_panel,
        unchanged_columns=(
            *base_columns,
            *registered_correction["unchanged_t1_columns"],
        ),
    )

    sessions = normalize_expected_sessions(panel_manifest["sessions"])
    evaluation = protocol["evaluation"]
    start = pd.Timestamp(evaluation["period"]["start"])
    end = pd.Timestamp(evaluation["period"]["end"])
    train_start = pd.Timestamp(evaluation["training_start"])
    minimum_training_sessions = int(evaluation["minimum_training_sessions"])
    spec = _anchor_spec(v06_protocol)

    base_result, base_picks = evaluate_recipe(
        panel,
        sessions,
        spec,
        recipe_id="G0_price_core",
        columns=base_columns,
        start=start,
        end=end,
        train_start=train_start,
        minimum_training_sessions=minimum_training_sessions,
    )
    legacy_result, legacy_picks = evaluate_recipe(
        panel,
        sessions,
        spec,
        recipe_id="G0_price_core+T1_event_structure",
        columns=(*base_columns, *t1_columns),
        start=start,
        end=end,
        train_start=train_start,
        minimum_training_sessions=minimum_training_sessions,
    )
    corrected_result, corrected_picks = evaluate_recipe(
        corrected_panel,
        sessions,
        spec,
        recipe_id="G0_price_core+T1_event_structure_corrected",
        columns=(*base_columns, *t1_columns),
        start=start,
        end=end,
        train_start=train_start,
        minimum_training_sessions=minimum_training_sessions,
    )
    del corrected_panel
    gc.collect()

    frozen_result = json.loads(
        _resolve_registered_path(frozen["v06_result_reference"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    frozen_pick_frame = pd.read_csv(
        _resolve_registered_path(frozen["v06_group_screen_picks_reference"]["path"]),
        dtype={"code": "string"},
    )
    frozen_pick_frame["date"] = pd.to_datetime(
        frozen_pick_frame["date"], errors="raise"
    )
    tolerance = float(evaluation["reference_reproduction_tolerance"])
    reproduction = {
        "G0_price_core": compare_reference_recipe(
            base_result,
            base_picks,
            frozen_result["group_screen"]["base"],
            frozen_pick_frame[
                frozen_pick_frame["recipe_id"].eq("G0_price_core")
            ],
            tolerance=tolerance,
        ),
        "legacy_T1": compare_reference_recipe(
            legacy_result,
            legacy_picks,
            frozen_result["group_screen"]["candidates"]["T1_event_structure"],
            frozen_pick_frame[
                frozen_pick_frame["recipe_id"].eq(
                    "G0_price_core+T1_event_structure"
                )
            ],
            tolerance=tolerance,
        ),
    }

    adjusted = max_t_adjusted_uplifts(
        base_picks,
        {"legacy_T1": legacy_picks, "corrected_T1": corrected_picks},
    )
    corrected_vs_legacy = paired_moving_block_bootstrap(
        _daily_net(corrected_picks),
        _daily_net(legacy_picks),
        block_length=BLOCK_LENGTH,
        samples=BOOTSTRAP_SAMPLES,
        confidence=BOOTSTRAP_CONFIDENCE,
        random_state=DEFAULT_SEED,
    )
    comparisons = {
        "shared_block_max_mean_adjusted_vs_G0": adjusted,
        "corrected_T1_minus_legacy_T1": asdict(corrected_vs_legacy),
        "pick_changes_corrected_vs_legacy": _pick_change_report(
            legacy_picks, corrected_picks
        ),
    }

    output_path = Path(args.output).resolve()
    if not output_path.is_relative_to(ROOT):
        raise DataValidationError("result output must stay inside repository")
    picks_path = output_path.with_name(f"{output_path.stem}_display_picks.csv")
    pick_frame = pd.concat(
        [base_picks, legacy_picks, corrected_picks], ignore_index=True
    )
    write_frame(pick_frame, picks_path)

    top2_key = "net_mean_pct_at_cost"
    base_mean = base_result["metrics"]["top2"]["net20"][top2_key]
    legacy_mean = legacy_result["metrics"]["top2"]["net20"][top2_key]
    corrected_mean = corrected_result["metrics"]["top2"]["net20"][top2_key]
    run_completed_at = datetime.now(ZoneInfo("Asia/Tokyo"))
    result_payload: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "authority": protocol["authority"],
        "decision": "correction_retrospective_only_no_production_change",
        "frozen_inputs": {
            name: {
                key: value
                for key, value in item.items()
                if key in {"path", "sha256", "manifest_path", "manifest_sha256"}
            }
            for name, item in frozen.items()
        },
        "overlay_validation": {
            "schema_valid": True,
            "unique_keys": True,
            "registered_coverage": coverage,
            "merge_semantics": correction_report,
            "legacy_event_key_coverage_complete": True,
            "complete_no_event_zero_preserved": True,
            "incomplete_source_nan_preserved": True,
        },
        "feature_isolation": {
            "replaced_columns": list(registered_correction["replace_columns"]),
            "unchanged_t1_columns": list(
                registered_correction["unchanged_t1_columns"]
            ),
            "g0_columns_unchanged": True,
            "other_t1_columns_unchanged": True,
        },
        "evaluation": {
            "period": evaluation["period"],
            "anchor_model": spec.canonical_dict(),
            "monthly_walk_forward": True,
            "reference_reproduction": reproduction,
            "recipes": {
                "G0_price_core": base_result,
                "legacy_T1": legacy_result,
                "corrected_T1": corrected_result,
            },
            "comparisons": comparisons,
        },
        "summary": {
            "G0_top2_net20_mean_pct": float(base_mean),
            "legacy_T1_top2_net20_mean_pct": float(legacy_mean),
            "corrected_T1_top2_net20_mean_pct": float(corrected_mean),
            "legacy_T1_minus_G0_pct": float(legacy_mean - base_mean),
            "corrected_T1_minus_G0_pct": float(corrected_mean - base_mean),
            "corrected_T1_minus_legacy_T1_pct": float(
                corrected_mean - legacy_mean
            ),
            "production_change": False,
            "feature_survivor_lock_changed": False,
        },
        "artifact": {
            "display_picks": {
                "path": str(picks_path.relative_to(ROOT)),
                "sha256": sha256_file(picks_path),
                "rows": int(len(pick_frame)),
            }
        },
        "runtime": {
            "protocol_registered_at": str(protocol["registered_at"]),
            "run_started_at": run_started_at.isoformat(),
            "run_completed_at": run_completed_at.isoformat(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "All outcomes in this interval were already viewed before this correction diagnostic.",
            "The two-variant correction family is not the original seven-group multiplicity family.",
            "No result from this run may promote a feature or production model.",
        ],
    }
    write_json(result_payload, output_path)
    manifest = {
        "schema_version": 1,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "result_path": str(output_path.relative_to(ROOT)),
        "result_sha256": sha256_file(output_path),
        "display_picks_path": str(picks_path.relative_to(ROOT)),
        "display_picks_sha256": sha256_file(picks_path),
        "frozen_input_sha256": {
            name: str(item["sha256"]) for name, item in frozen.items()
        },
    }
    write_json(manifest, output_path.with_suffix(".manifest.json"))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
