from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from tse_session_ranker import RankerConfig, SessionRanker
from tse_session_ranker.artifact import load_artifact
from tse_session_ranker.data.tdnet import TDnetDataset, normalize_tdnet_disclosures
from tse_session_ranker.exceptions import ArtifactError, DataValidationError
from tse_session_ranker.inference import predict_candidates
from tse_session_ranker.training import train_model

from .helpers import synthetic_prices, synthetic_tdnet


class TrainingArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prices = synthetic_prices(periods=230, codes=4)
        cls.dates = sorted(cls.prices["date"].unique())
        cls.calendar = pd.DatetimeIndex(cls.dates)
        cls.tdnet = synthetic_tdnet(cls.prices)
        cls.config = RankerConfig(regime_start=str(pd.Timestamp(cls.dates[70]).date()))

    def _train(
        self, prices: pd.DataFrame, cutoff: pd.Timestamp
    ):
        return train_model(
            prices,
            cutoff,
            tdnet_dataset=self.tdnet,
            config=self.config,
            expected_sessions=self.calendar,
        )

    def test_train_end_is_hard_cutoff(self) -> None:
        cutoff = pd.Timestamp(self.dates[180])
        first = self._train(self.prices, cutoff).artifact
        changed = self.prices.copy()
        future = changed["date"] > cutoff
        changed.loc[future, ["open", "high", "low", "close"]] *= 7.0
        second = self._train(changed, cutoff).artifact
        first_model = first.estimator.named_steps["model"]
        second_model = second.estimator.named_steps["model"]
        np.testing.assert_allclose(first_model.coef_, second_model.coef_, atol=0, rtol=0)
        np.testing.assert_allclose(first_model.intercept_, second_model.intercept_, atol=0, rtol=0)
        self.assertEqual(
            first.manifest["training_data_sha256"],
            second.manifest["training_data_sha256"],
        )
        self.assertEqual(first.manifest["input_max_date"], str(cutoff.date()))

    def test_tdnet_after_train_cutoff_cannot_change_model_or_hash(self) -> None:
        cutoff = pd.Timestamp(self.dates[180])
        first = self._train(self.prices, cutoff).artifact
        future_disclosure = normalize_tdnet_disclosures(
            pd.DataFrame(
                {
                    "published_at": [
                        pd.Timestamp(
                            f"{cutoff.date()} 09:00:00", tz="Asia/Tokyo"
                        )
                    ],
                    "code": [str(self.prices.iloc[0]["code"])],
                    "name": [str(self.prices.iloc[0]["name"])],
                    "title": ["業績予想の上方修正に関するお知らせ"],
                    "url": ["https://example.invalid/future.pdf"],
                }
            )
        )
        changed_tdnet = TDnetDataset(
            disclosures=normalize_tdnet_disclosures(
                pd.concat(
                    [self.tdnet.disclosures, future_disclosure], ignore_index=True
                )
            ),
            complete_dates=self.tdnet.complete_dates,
            observed_at_by_date=self.tdnet.observed_at_by_date,
            provenance_by_date=self.tdnet.provenance_by_date,
            source_sha256="changed-after-cutoff",
            source_files=self.tdnet.source_files,
        )
        second = train_model(
            self.prices,
            cutoff,
            tdnet_dataset=changed_tdnet,
            config=self.config,
            expected_sessions=self.calendar,
        ).artifact
        first_model = first.estimator.named_steps["model"]
        second_model = second.estimator.named_steps["model"]
        np.testing.assert_allclose(first_model.coef_, second_model.coef_, atol=0, rtol=0)
        np.testing.assert_allclose(
            first_model.intercept_, second_model.intercept_, atol=0, rtol=0
        )
        self.assertEqual(
            first.manifest["tdnet_source_sha256"],
            second.manifest["tdnet_source_sha256"],
        )
        self.assertEqual(
            first.manifest["tdnet_disclosures_through_cutoff"],
            second.manifest["tdnet_disclosures_through_cutoff"],
        )

    def test_missing_tdnet_calendar_date_fails_training_closed(self) -> None:
        cutoff = pd.Timestamp(self.dates[180])
        missing_date = pd.Timestamp(self.dates[120]) + pd.Timedelta(days=1)
        incomplete = TDnetDataset(
            disclosures=self.tdnet.disclosures,
            complete_dates=self.tdnet.complete_dates.difference(
                pd.DatetimeIndex([missing_date])
            ),
            observed_at_by_date=self.tdnet.observed_at_by_date.drop(
                missing_date, errors="ignore"
            ),
            provenance_by_date=self.tdnet.provenance_by_date.drop(
                missing_date, errors="ignore"
            ),
            source_sha256="incomplete-tdnet",
            source_files=self.tdnet.source_files - 1,
        )
        with self.assertRaisesRegex(DataValidationError, "TDnet index coverage"):
            train_model(
                self.prices,
                cutoff,
                tdnet_dataset=incomplete,
                config=self.config,
                expected_sessions=self.calendar,
            )

    def test_historical_tdnet_provenance_cannot_enter_production_training(self) -> None:
        historical = TDnetDataset(
            disclosures=self.tdnet.disclosures,
            complete_dates=self.tdnet.complete_dates,
            observed_at_by_date=self.tdnet.observed_at_by_date,
            source_sha256="historical-only-tdnet",
            source_files=self.tdnet.source_files,
            provenance_by_date=pd.Series(
                "legacy_historical_file_mtime_assumption",
                index=self.tdnet.complete_dates,
            ),
        )
        with self.assertRaisesRegex(DataValidationError, "production TDnet"):
            train_model(
                self.prices,
                pd.Timestamp(self.dates[180]),
                tdnet_dataset=historical,
                config=self.config,
                expected_sessions=self.calendar,
            )

    def test_artifact_roundtrip_and_prediction(self) -> None:
        cutoff = pd.Timestamp(self.dates[180])
        artifact = self._train(self.prices, cutoff).artifact
        self.assertEqual(
            artifact.manifest["selection_objective"],
            "top1_net_mean_pct_at_cost",
        )
        self.assertEqual(
            artifact.manifest["data_semantics"],
            "prior_session_universe_source_mask_tdnet_v3",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            artifact.save(path)
            loaded = load_artifact(path)
            before = predict_candidates(
                artifact,
                self.prices,
                pd.Timestamp(self.dates[181]),
                self.tdnet,
                top_k=2,
                expected_history_date=pd.Timestamp(self.dates[180]),
                expected_sessions=self.calendar,
            )
            after = predict_candidates(
                loaded,
                self.prices,
                pd.Timestamp(self.dates[181]),
                self.tdnet,
                top_k=2,
                expected_history_date=pd.Timestamp(self.dates[180]),
                expected_sessions=self.calendar,
            )
        self.assertEqual(artifact.feature_columns, loaded.feature_columns)
        self.assertEqual(artifact.config, loaded.config)
        np.testing.assert_allclose(
            before.candidates["model_score"],
            after.candidates["model_score"],
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            before.candidates["code"].tolist(), after.candidates["code"].tolist()
        )
        for column in (
            "run_id",
            "score_semantics",
            "selection_objective",
            "data_semantics",
            "session_calendar_mode",
            "session_calendar_sha256",
        ):
            self.assertIn(column, before.candidates)

    def test_artifact_requires_the_fixed_logistic_pipeline(self) -> None:
        artifact = self._train(
            self.prices, pd.Timestamp(self.dates[180])
        ).artifact
        artifact.estimator.steps[-1] = ("model", DummyClassifier())
        with self.assertRaisesRegex(ArtifactError, "fixed regularized logistic"):
            artifact.validate()
        drifted = self._train(
            self.prices, pd.Timestamp(self.dates[180])
        ).artifact
        drifted.estimator.named_steps["model"].set_params(penalty=None)
        with self.assertRaisesRegex(ArtifactError, "logistic parameters"):
            drifted.validate()

    def test_sidecar_metadata_must_match_embedded_artifact(self) -> None:
        cutoff = pd.Timestamp(self.dates[180])
        artifact = self._train(self.prices, cutoff).artifact
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            artifact.save(path)
            sidecar = path.with_suffix(path.suffix + ".manifest.json")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            metadata["run_id"] = "tampered"
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "disagrees"):
                load_artifact(path)
            sidecar.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "could not read"):
                load_artifact(path)

    def test_ranker_rejects_missing_or_conflicting_artifact(self) -> None:
        with self.assertRaises(ArtifactError):
            SessionRanker().predict(
                self.prices,
                target_date=pd.Timestamp(self.dates[181]),
                tdnet_indexes=self.tdnet,
            )
        cutoff = pd.Timestamp(self.dates[180])
        artifact = self._train(self.prices, cutoff).artifact
        with self.assertRaisesRegex(ArtifactError, "does not match"):
            SessionRanker(config=RankerConfig(cost_bps=21.0), artifact=artifact)

    def test_profit_cost_config_is_fail_closed(self) -> None:
        for invalid in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                RankerConfig(cost_bps=invalid)

    def test_explicit_training_calendar_is_required_again_at_inference(self) -> None:
        cutoff = pd.Timestamp(self.dates[180])
        calendar = pd.DatetimeIndex(self.dates[:182])
        with self.assertRaisesRegex(DataValidationError, "requires an explicit"):
            train_model(
                self.prices,
                cutoff,
                tdnet_dataset=self.tdnet,
                config=self.config,
            )
        artifact = train_model(
            self.prices,
            cutoff,
            tdnet_dataset=self.tdnet,
            config=self.config,
            expected_sessions=calendar,
        ).artifact
        self.assertEqual(
            artifact.manifest["session_calendar_mode"],
            "explicit_exchange_sessions",
        )
        with self.assertRaisesRegex(DataValidationError, "requires"):
            predict_candidates(
                artifact,
                self.prices,
                pd.Timestamp(self.dates[181]),
                self.tdnet,
                expected_history_date=cutoff,
            )
        result = predict_candidates(
            artifact,
            self.prices,
            pd.Timestamp(self.dates[181]),
            self.tdnet,
            expected_history_date=cutoff,
            expected_sessions=(value for value in calendar),
        )
        self.assertGreater(len(result.candidates), 0)
        with self.assertRaisesRegex(DataValidationError, "calendar previous"):
            predict_candidates(
                artifact,
                self.prices,
                pd.Timestamp(self.dates[181]),
                self.tdnet,
                expected_history_date=pd.Timestamp(self.dates[179]),
                expected_sessions=calendar,
            )
        with self.assertRaisesRegex(DataValidationError, "not in"):
            predict_candidates(
                artifact,
                self.prices,
                pd.Timestamp(self.dates[181]) + pd.Timedelta(hours=24),
                self.tdnet,
                expected_history_date=pd.Timestamp(self.dates[181]),
                expected_sessions=calendar,
            )

    def test_truncated_calendar_cannot_certify_a_later_cutoff(self) -> None:
        truncated_end = pd.Timestamp(self.dates[180])
        requested_end = pd.Timestamp(self.dates[190])
        truncated_prices = self.prices[self.prices["date"].le(truncated_end)]
        truncated_calendar = pd.DatetimeIndex(self.dates[:181])
        with self.assertRaisesRegex(DataValidationError, "calendar ends"):
            train_model(
                truncated_prices,
                requested_end,
                tdnet_dataset=self.tdnet,
                config=self.config,
                expected_sessions=truncated_calendar,
            )

        calendar_with_cutoff_gap = pd.DatetimeIndex(
            [value for value in self.dates[:196] if value != requested_end]
        )
        prices_with_cutoff_gap = self.prices[
            ~self.prices["date"].eq(requested_end)
        ]
        with self.assertRaisesRegex(DataValidationError, "train_end is not"):
            train_model(
                prices_with_cutoff_gap,
                requested_end,
                tdnet_dataset=self.tdnet,
                config=self.config,
                expected_sessions=calendar_with_cutoff_gap,
            )
        with self.assertRaisesRegex(DataValidationError, "evaluation_end is not"):
            SessionRanker(self.config).backtest(
                prices_with_cutoff_gap,
                evaluation_start=pd.Timestamp(self.dates[170]),
                evaluation_end=requested_end,
                tdnet_indexes=self.tdnet,
                expected_sessions=calendar_with_cutoff_gap,
            )
        with self.assertRaisesRegex(DataValidationError, "calendar ends"):
            SessionRanker(self.config).backtest(
                truncated_prices,
                evaluation_start=pd.Timestamp(self.dates[170]),
                evaluation_end=requested_end,
                tdnet_indexes=self.tdnet,
                expected_sessions=truncated_calendar,
            )

    def test_invalid_date_is_not_silently_dropped(self) -> None:
        bad = self.prices.copy()
        bad["date"] = bad["date"].dt.strftime("%Y-%m-%d")
        bad.loc[bad.index[0], "date"] = "not-a-date"
        with self.assertRaisesRegex(DataValidationError, "invalid dates"):
            train_model(
                bad,
                pd.Timestamp(self.dates[180]),
                tdnet_dataset=self.tdnet,
                config=self.config,
                expected_sessions=self.calendar,
            )


if __name__ == "__main__":
    unittest.main()
