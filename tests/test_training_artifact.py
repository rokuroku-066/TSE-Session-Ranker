from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tse_session_ranker import RankerConfig
from tse_session_ranker.artifact import load_artifact
from tse_session_ranker.exceptions import DataValidationError
from tse_session_ranker.inference import predict_candidates
from tse_session_ranker.training import train_model

from .helpers import synthetic_prices


class TrainingArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prices = synthetic_prices(periods=230, codes=4)
        cls.dates = sorted(cls.prices["date"].unique())
        cls.config = RankerConfig(regime_start=str(pd.Timestamp(cls.dates[70]).date()))

    def test_train_end_is_hard_cutoff(self) -> None:
        cutoff = pd.Timestamp(self.dates[180])
        first = train_model(self.prices, cutoff, config=self.config).artifact
        changed = self.prices.copy()
        future = changed["date"] > cutoff
        changed.loc[future, ["open", "high", "low", "close"]] *= 7.0
        second = train_model(changed, cutoff, config=self.config).artifact
        first_model = first.estimator.named_steps["model"]
        second_model = second.estimator.named_steps["model"]
        np.testing.assert_allclose(first_model.coef_, second_model.coef_, atol=0, rtol=0)
        np.testing.assert_allclose(first_model.intercept_, second_model.intercept_, atol=0, rtol=0)
        self.assertEqual(
            first.manifest["training_data_sha256"],
            second.manifest["training_data_sha256"],
        )
        self.assertEqual(first.manifest["input_max_date"], str(cutoff.date()))

    def test_artifact_roundtrip_and_prediction(self) -> None:
        cutoff = pd.Timestamp(self.dates[180])
        artifact = train_model(self.prices, cutoff, config=self.config).artifact
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            artifact.save(path)
            loaded = load_artifact(path)
            before = predict_candidates(
                artifact,
                self.prices,
                pd.Timestamp(self.dates[181]),
                top_k=2,
                expected_history_date=pd.Timestamp(self.dates[180]),
            )
            after = predict_candidates(
                loaded,
                self.prices,
                pd.Timestamp(self.dates[181]),
                top_k=2,
                expected_history_date=pd.Timestamp(self.dates[180]),
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

    def test_invalid_date_is_not_silently_dropped(self) -> None:
        bad = self.prices.copy()
        bad["date"] = bad["date"].dt.strftime("%Y-%m-%d")
        bad.loc[bad.index[0], "date"] = "not-a-date"
        with self.assertRaisesRegex(DataValidationError, "invalid dates"):
            train_model(
                bad,
                pd.Timestamp(self.dates[180]),
                config=self.config,
            )


if __name__ == "__main__":
    unittest.main()
