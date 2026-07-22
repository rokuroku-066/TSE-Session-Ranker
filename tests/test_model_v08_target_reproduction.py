from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research.evaluate_model_v08_target import (
    BASE_FEATURES,
    DEFAULT_PROTOCOL,
    G5_FEATURES,
    _assert_panel_lock,
    _fit_rank_ridge,
    _load_locked_manifest,
    _load_locked_protocol,
    _validate_manifest_calendar,
    _validate_panel,
    _validate_score_schedule,
    sha256_file,
)
from tse_session_ranker.exceptions import DataValidationError
from tse_session_ranker.research_models import same_day_return_percentile_target


class ModelV08ArtifactLockTests(unittest.TestCase):
    def test_stable_stage_registrations_match_locked_hashes(self) -> None:
        protocol = _load_locked_protocol(DEFAULT_PROTOCOL)
        root = DEFAULT_PROTOCOL.parents[1]
        records = protocol["traceability"]["stage_registrations"]
        self.assertEqual(set(records), {"stage_1", "stage_2", "stage_3", "stage_4"})
        for stage, record in records.items():
            with self.subTest(stage=stage):
                path = root / record["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(sha256_file(path), record["sha256"])

    def test_protocol_tamper_is_rejected(self) -> None:
        original = DEFAULT_PROTOCOL.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "protocol.json"
            tampered.write_bytes(original + b"\n")
            with self.assertRaisesRegex(DataValidationError, "protocol SHA-256 mismatch"):
                _load_locked_protocol(tampered)

    def test_manifest_tamper_is_rejected(self) -> None:
        protocol = deepcopy(_load_locked_protocol(DEFAULT_PROTOCOL))
        clean = b'{"panel_file_sha256":"' + (b"0" * 64) + b'"}\n'
        expected = hashlib.sha256(clean).hexdigest()
        protocol["frozen_input"]["panel_manifest_sha256"] = expected
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.manifest.json"
            path.write_bytes(clean)
            self.assertEqual(
                _load_locked_manifest(path, protocol)["panel_file_sha256"],
                "0" * 64,
            )
            path.write_bytes(clean + b" ")
            with self.assertRaisesRegex(
                DataValidationError, "panel manifest SHA-256 mismatch"
            ):
                _load_locked_manifest(path, protocol)

    def test_panel_tamper_is_rejected(self) -> None:
        protocol = deepcopy(_load_locked_protocol(DEFAULT_PROTOCOL))
        manifest: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.pkl"
            path.write_bytes(b"locked-panel")
            expected = sha256_file(path)
            protocol["frozen_input"]["panel_sha256"] = expected
            manifest["panel_file_sha256"] = expected
            self.assertEqual(_assert_panel_lock(path, manifest, protocol), expected)
            path.write_bytes(b"tampered-panel")
            with self.assertRaisesRegex(
                DataValidationError,
                "actual file, protocol, and manifest",
            ):
                _assert_panel_lock(path, manifest, protocol)


class ModelV08FailClosedPanelTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> tuple[
        pd.DataFrame, dict[str, object], dict[str, object], pd.DatetimeIndex
    ]:
        sessions = pd.DatetimeIndex(["2024-01-04", "2024-01-05"])
        rows: list[dict[str, object]] = []
        for date, value in zip(sessions, (1.0, -1.0), strict=True):
            row: dict[str, object] = {
                "date": date,
                "code": "1001",
                "name": "fixture",
                "label": float(value > 0.0),
                "oc_return_pct": value,
                "price_eligible": True,
                "price_training_eligible": True,
                "candidate_price_source_max_date": date - pd.Timedelta(days=1),
            }
            row.update({column: 0.0 for column in (*BASE_FEATURES, *G5_FEATURES)})
            rows.append(row)
        panel = pd.DataFrame(rows)
        manifest: dict[str, object] = {
            "rows": len(panel),
            "codes": 1,
            "columns": list(panel.columns),
            "sessions": [session.strftime("%Y-%m-%d") for session in sessions],
        }
        protocol: dict[str, object] = {
            "frozen_input": {
                "rows": len(panel),
                "codes": 1,
                "sessions": len(sessions),
            }
        }
        return panel, manifest, protocol, sessions

    def test_valid_fixture_passes_all_outcome_checks(self) -> None:
        panel, manifest, protocol, sessions = self._fixture()
        checks = _validate_panel(panel, manifest, protocol, sessions)
        self.assertTrue(checks["label_return_missingness_exact"])
        self.assertTrue(checks["finite_observed_returns"])
        self.assertTrue(checks["binary_labels"])
        self.assertTrue(checks["label_return_sign_exact"])

    def test_label_return_missingness_mismatch_is_rejected(self) -> None:
        panel, manifest, protocol, sessions = self._fixture()
        panel.loc[0, "label"] = np.nan
        with self.assertRaisesRegex(DataValidationError, "missingness"):
            _validate_panel(panel, manifest, protocol, sessions)

    def test_nonfinite_return_binary_label_and_sign_tampering_are_rejected(self) -> None:
        panel, manifest, protocol, sessions = self._fixture()
        for column, value, message in (
            ("oc_return_pct", np.inf, "finite"),
            ("label", 2.0, "binary"),
            ("label", 0.0, "return > 0"),
        ):
            changed = panel.copy()
            changed.loc[0, column] = value
            with self.subTest(column=column, value=value):
                with self.assertRaisesRegex(DataValidationError, message):
                    _validate_panel(changed, manifest, protocol, sessions)

    def test_panel_date_set_tampering_is_rejected(self) -> None:
        panel, manifest, protocol, sessions = self._fixture()
        panel.loc[1, "date"] = pd.Timestamp("2024-01-08")
        with self.assertRaisesRegex(DataValidationError, "date set"):
            _validate_panel(panel, manifest, protocol, sessions)

    def test_calendar_and_score_count_tampering_are_rejected(self) -> None:
        protocol = deepcopy(_load_locked_protocol(DEFAULT_PROTOCOL))
        raw = ["2024-01-04", "2024-01-05"]
        manifest = {
            "sessions": raw,
            "calendar_sha256": "not-the-locked-calendar",
        }
        with self.assertRaisesRegex(DataValidationError, "session count"):
            _validate_manifest_calendar(manifest, protocol)

        sessions = pd.date_range("2024-07-01", periods=266, freq="D")
        protocol["frozen_input"]["score_period"] = [
            str(sessions.min().date()),
            str(sessions.max().date()),
        ]
        protocol["frozen_input"]["score_sessions"] = 265
        protocol["frozen_input"]["fold_count"] = 9
        with self.assertRaisesRegex(DataValidationError, "score-session count"):
            _validate_score_schedule(pd.DatetimeIndex(sessions), protocol)


class ModelV08CoreRankTargetTests(unittest.TestCase):
    def test_historical_percentile_formula_is_not_zero_centered(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-04"] * 4),
                "oc_return_pct": [1.0, 2.0, 3.0, 4.0],
            }
        )
        target = same_day_return_percentile_target(frame)
        np.testing.assert_allclose(target, [-0.5, 0.0, 0.5, 1.0])
        self.assertAlmostEqual(float(target.mean()), 0.25)

    def test_runner_delegates_to_core_ridge_daily_rank(self) -> None:
        sentinel = object()
        with patch(
            "research.evaluate_model_v08_target.fit_research_model",
            return_value=sentinel,
        ) as core_fit:
            returned = _fit_rank_ridge(
                pd.DataFrame(), ("feature_one",), alpha=1.0
            )
        self.assertIs(returned, sentinel)
        spec = core_fit.call_args.args[0]
        self.assertEqual(spec.family, "ridge_daily_rank")
        self.assertEqual(spec.objective, "same_day_return_percentile")
        self.assertEqual(dict(spec.parameters), {"alpha": 1.0})


if __name__ == "__main__":
    unittest.main()
