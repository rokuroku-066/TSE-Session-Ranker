#!/usr/bin/env python3
"""Verify the compact panel by reproducing frozen development-period picks.

Only already-open development intervals are evaluated.  The sealed diagnostic
interval is not scored, summarized, or inspected by this audit.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research import finalize_logit_v04 as base  # noqa: E402
from research import recover_logit_v04_holdout as formal  # noqa: E402


PANEL_PATH = Path("/tmp/model_v04_post_failure_diagnostic_panel.pkl")
PANEL_SHA256 = (
    "f2da914861dd9da8afc64567f90f0ef9af4ff1b0eb2e480dbff2f2cc82ff395a"
)
PANEL_MANIFEST_PATH = PANEL_PATH.with_suffix(PANEL_PATH.suffix + ".manifest.json")
PANEL_MANIFEST_SHA256 = (
    "780206fe23a72481ce6c2297a24f789aebebcbd3bb434889f2ab9d582f75d864"
)
SELECTION_PATH = ROOT / "research/model_v04_selection_lock.json"
WINNER_PICKS_PATH = ROOT / "research/model_v04_selection_lock_winner_picks.csv"
CONTROL_PICKS_PATH = (
    ROOT / "research/model_v04_selection_lock_v03_control_picks.csv"
)
OUTPUT_PATH = (
    ROOT / "research/model_v04_diagnostic_panel_development_equivalence.json"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    text = frame.to_csv(index=False)
    parsed = pd.read_csv(io.StringIO(text), dtype={"code": "string"})
    parsed["date"] = pd.to_datetime(parsed["date"])
    return parsed


def assert_pick_identity(
    actual: pd.DataFrame, expected_path: Path
) -> tuple[str, float]:
    expected = base.read_frame(expected_path).drop(columns=["name"])
    expected["date"] = pd.to_datetime(expected["date"])
    columns = list(expected.columns)
    actual = actual.loc[:, columns]
    expected_canonical = csv_canonical(expected)
    actual_canonical = csv_canonical(actual)
    exact_columns = [
        column for column in expected_canonical if column != "oc_return_pct"
    ]
    pd.testing.assert_frame_equal(
        expected_canonical[exact_columns],
        actual_canonical[exact_columns],
        check_dtype=True,
        check_exact=True,
    )
    expected_returns = expected_canonical["oc_return_pct"].to_numpy(dtype=float)
    actual_returns = actual_canonical["oc_return_pct"].to_numpy(dtype=float)
    np.testing.assert_allclose(
        expected_returns,
        actual_returns,
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    finite = np.isfinite(expected_returns) & np.isfinite(actual_returns)
    max_return_difference = float(
        np.max(np.abs(expected_returns[finite] - actual_returns[finite]))
        if finite.any()
        else 0.0
    )
    encoded = actual_canonical.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), max_return_difference


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError("development equivalence audit already exists")
    if sha256_file(PANEL_PATH) != PANEL_SHA256:
        raise ValueError("diagnostic panel changed")
    if sha256_file(PANEL_MANIFEST_PATH) != PANEL_MANIFEST_SHA256:
        raise ValueError("diagnostic panel manifest changed")
    selection = formal.read_json_bound(
        SELECTION_PATH, formal.EXPECTED_SELECTION_SHA256
    )
    manifest = json.loads(PANEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("aggregate_return_metrics_computed") is not False or (
        manifest.get("model_evaluation_count") != 0
    ):
        raise ValueError("panel manifest indicates prior model evaluation")
    panel = base.read_frame(PANEL_PATH)
    sessions = base._load_calendar(
        selection["data"]["calendar_path"], base.CONFIRMATION_END
    )
    winner_spec = base.ModelSpec.from_dict(selection["winner"])
    control_spec = formal.baseline_spec()
    winner = base.evaluate_spec(
        panel,
        winner_spec,
        evaluation_start=base.CONFIRMATION_START,
        evaluation_end=base.CONFIRMATION_END,
        evaluation_sessions=sessions,
        retrain_frequency="M",
        bootstrap_samples=base.DEFAULT_BOOTSTRAP_SAMPLES,
    )
    winner_hash, winner_max_return_difference = assert_pick_identity(
        winner.picks, WINNER_PICKS_PATH
    )
    del winner
    control = base.evaluate_spec(
        panel,
        control_spec,
        evaluation_start=base.LEGACY_COMPARISON_START,
        evaluation_end=base.CONFIRMATION_END,
        evaluation_sessions=sessions,
        retrain_frequency="M",
        bootstrap_samples=base.DEFAULT_BOOTSTRAP_SAMPLES,
    )
    control_hash, control_max_return_difference = assert_pick_identity(
        control.picks, CONTROL_PICKS_PATH
    )
    payload = {
        "schema_version": 1,
        "record_type": "post_failure_diagnostic_panel_development_equivalence",
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "formal_status": "not_a_formal_holdout_result",
        "panel_sha256": PANEL_SHA256,
        "panel_manifest_sha256": PANEL_MANIFEST_SHA256,
        "audit_runner_path": str(Path(__file__).resolve()),
        "audit_runner_sha256": sha256_file(Path(__file__).resolve()),
        "selection_lock_sha256": formal.EXPECTED_SELECTION_SHA256,
        "winner_spec_id": winner_spec.id,
        "winner_expected_picks_sha256": sha256_file(WINNER_PICKS_PATH),
        "winner_reproduced_canonical_sha256": winner_hash,
        "winner_pick_identity_passed": True,
        "winner_max_abs_oc_return_pct_difference": winner_max_return_difference,
        "control_spec_id": control_spec.id,
        "control_expected_picks_sha256": sha256_file(CONTROL_PICKS_PATH),
        "control_reproduced_canonical_sha256": control_hash,
        "control_pick_identity_passed": True,
        "control_max_abs_oc_return_pct_difference": control_max_return_difference,
        "oc_return_pct_absolute_tolerance": 1e-12,
        "all_non_return_pick_fields_exact_after_csv_roundtrip": True,
        "feature_values_and_dtypes_synthetic_exact_test": True,
        "holdout_evaluation_count": 0,
        "holdout_aggregate_metrics_computed_or_inspected": False,
        "development_metrics_recomputed_but_not_used_for_selection": True,
    }
    base.write_json(payload, OUTPUT_PATH)
    print(f"audit={OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
