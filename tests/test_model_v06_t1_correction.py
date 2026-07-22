from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import unittest
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.evaluate_model_v06_t1_correction import (
    apply_corrected_overlay,
    assert_feature_isolation,
    load_protocol,
    validate_overlay_schema,
)
from tse_session_ranker.exceptions import DataValidationError


CORRECTED = (
    "tdnet_clean_family_count_log1p",
    "tdnet_clean_single_family",
)


def _overlay(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime([row[0] for row in rows]),
            "code": pd.Series([row[1] for row in rows], dtype=object),
            CORRECTED[0]: pd.Series([row[2] for row in rows], dtype="float32"),
            CORRECTED[1]: pd.Series([row[3] for row in rows], dtype="float32"),
        }
    )


def _overlay_spec(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": {column: str(frame[column].dtype) for column in frame},
        "date_min": str(frame["date"].min().date()),
        "date_max": str(frame["date"].max().date()),
        "unique_key": ["date", "code"],
    }


class ProtocolTests(unittest.TestCase):
    def test_protocol_is_prior_and_correction_only(self) -> None:
        protocol = load_protocol()
        registered = pd.Timestamp(protocol["registered_at"])
        self.assertIsNotNone(registered.tzinfo)
        self.assertLess(registered, pd.Timestamp(datetime.now(ZoneInfo("Asia/Tokyo"))))
        self.assertFalse(protocol["authority"]["production_promotion_allowed"])
        self.assertFalse(protocol["authority"]["selection_allowed"])
        self.assertEqual(
            protocol["correction"]["replace_columns"], list(CORRECTED)
        )
        self.assertEqual(len(protocol["correction"]["unchanged_t1_columns"]), 11)


class OverlayValidationTests(unittest.TestCase):
    def test_valid_count_geometry_passes(self) -> None:
        frame = _overlay(
            [
                ("2024-07-01", "A", 0.0, 0.0),
                ("2024-07-02", "B", float(np.log1p(1)), 1.0),
                ("2024-07-03", "C", float(np.log1p(3)), 0.0),
            ]
        )
        validate_overlay_schema(frame, _overlay_spec(frame))

    def test_duplicate_keys_and_inconsistent_single_flag_fail(self) -> None:
        duplicate = _overlay(
            [
                ("2024-07-01", "A", 0.0, 0.0),
                ("2024-07-01", "A", 0.0, 0.0),
            ]
        )
        with self.assertRaisesRegex(DataValidationError, "duplicate"):
            validate_overlay_schema(duplicate, _overlay_spec(duplicate))

        inconsistent = _overlay(
            [("2024-07-01", "A", float(np.log1p(2)), 1.0)]
        )
        with self.assertRaisesRegex(DataValidationError, "single-family"):
            validate_overlay_schema(inconsistent, _overlay_spec(inconsistent))


class MergeSemanticsTests(unittest.TestCase):
    @staticmethod
    def _panel() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-07-01"] * 4),
                "code": ["A", "B", "C", "D"],
                "tdnet_source_complete": [True, True, False, False],
                CORRECTED[0]: pd.Series(
                    [float(np.log1p(1)), 0.0, np.nan, np.nan], dtype="float32"
                ),
                CORRECTED[1]: pd.Series([1.0, 0.0, np.nan, np.nan], dtype="float32"),
                "other_t1": pd.Series([1.0, 2.0, 3.0, 4.0], dtype="float32"),
                "g0": [10.0, 20.0, 30.0, 40.0],
                "label": [0.0, 1.0, 0.0, 1.0],
                "oc_return_pct": [-1.0, 1.0, -2.0, 2.0],
            }
        )

    def test_complete_no_event_zero_and_incomplete_nan_are_preserved(self) -> None:
        panel = self._panel()
        overlay = _overlay(
            [
                ("2024-07-01", "A", float(np.log1p(2)), 0.0),
                ("2024-07-01", "C", float(np.log1p(1)), 1.0),
            ]
        )
        corrected, report = apply_corrected_overlay(panel, overlay)
        self.assertAlmostEqual(corrected.loc[0, CORRECTED[0]], np.log1p(2))
        self.assertEqual(corrected.loc[0, CORRECTED[1]], 0.0)
        self.assertEqual(corrected.loc[1, CORRECTED[0]], 0.0)
        self.assertEqual(corrected.loc[1, CORRECTED[1]], 0.0)
        self.assertTrue(corrected.loc[[2, 3], list(CORRECTED)].isna().all(axis=None))
        self.assertEqual(report["complete_rows_implicit_no_event_zero"], 1)
        assert_feature_isolation(
            panel, corrected, unchanged_columns=("other_t1", "g0")
        )

    def test_missing_legacy_event_key_fails_closed(self) -> None:
        panel = self._panel()
        overlay = _overlay([("2024-07-01", "B", 0.0, 0.0)])
        with self.assertRaisesRegex(DataValidationError, "legacy event"):
            apply_corrected_overlay(panel, overlay)

    def test_target_outcomes_cannot_change_corrected_features(self) -> None:
        panel = self._panel()
        overlay = _overlay(
            [("2024-07-01", "A", float(np.log1p(2)), 0.0)]
        )
        baseline, _ = apply_corrected_overlay(panel, overlay)
        mutated = panel.copy()
        mutated["label"] = 1.0 - mutated["label"]
        mutated["oc_return_pct"] = np.linspace(-99.0, 99.0, len(mutated))
        rebuilt, _ = apply_corrected_overlay(mutated, overlay)
        pd.testing.assert_frame_equal(
            baseline[list(CORRECTED)], rebuilt[list(CORRECTED)]
        )


class ResultArtifactTests(unittest.TestCase):
    def test_formal_result_is_hash_bound_and_non_production(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result_path = root / "research/model_v06_t1_correction_result.json"
        manifest_path = result_path.with_suffix(".manifest.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        def sha256(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        self.assertEqual(manifest["result_sha256"], sha256(result_path))
        self.assertEqual(
            manifest["protocol_sha256"],
            sha256(root / manifest["protocol_path"]),
        )
        self.assertEqual(
            manifest["runner_sha256"], sha256(root / manifest["runner_path"])
        )
        picks_path = root / manifest["display_picks_path"]
        self.assertEqual(manifest["display_picks_sha256"], sha256(picks_path))
        self.assertFalse(result["summary"]["production_change"])
        self.assertFalse(result["summary"]["feature_survivor_lock_changed"])
        self.assertTrue(
            result["evaluation"]["reference_reproduction"]["G0_price_core"][
                "reproduced"
            ]
        )
        self.assertTrue(
            result["evaluation"]["reference_reproduction"]["legacy_T1"][
                "reproduced"
            ]
        )
        registered = pd.Timestamp(result["runtime"]["protocol_registered_at"])
        started = pd.Timestamp(result["runtime"]["run_started_at"])
        completed = pd.Timestamp(result["runtime"]["run_completed_at"])
        self.assertLess(registered, started)
        self.assertLessEqual(started, completed)
        picks = pd.read_csv(picks_path)
        self.assertEqual(picks.groupby("recipe_id").size().tolist(), [168, 168, 168])
        self.assertFalse(picks.duplicated(["recipe_id", "date", "model_rank"]).any())


if __name__ == "__main__":
    unittest.main()
