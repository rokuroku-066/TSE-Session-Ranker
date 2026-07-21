from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .artifact import ModelArtifact
from .config import RankerConfig
from .data.common import normalize_daily_prices
from .exceptions import DataValidationError
from .features import FEATURE_COLUMNS, build_feature_panel, validate_feature_columns


@dataclass
class TrainingResult:
    artifact: ModelArtifact
    metrics: dict[str, Any]


def make_estimator(config: RankerConfig) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=config.model.c,
                    class_weight=config.model.class_weight,
                    max_iter=config.model.max_iter,
                    random_state=config.model.random_state,
                ),
            ),
        ]
    )


def date_equal_weights(frame: pd.DataFrame) -> pd.Series:
    counts = frame.groupby("date", sort=False)["code"].transform("size")
    return 1.0 / counts


def fit_estimator(frame: pd.DataFrame, config: RankerConfig) -> Pipeline:
    validate_feature_columns(FEATURE_COLUMNS)
    if frame.empty:
        raise DataValidationError("no eligible labelled rows are available for training")
    labels = frame["label"].dropna().astype(int)
    if labels.nunique() != 2:
        raise DataValidationError("training labels must contain both winning and losing sessions")
    estimator = make_estimator(config)
    weights = date_equal_weights(frame) if config.model.date_equal_weight else None
    fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
    estimator.fit(frame[list(FEATURE_COLUMNS)], frame["label"].astype(int), **fit_kwargs)
    return estimator


def _data_hash(frame: pd.DataFrame) -> str:
    fields = [
        column
        for column in ("date", "code", "open", "high", "low", "close", "traded")
        if column in frame
    ]
    values = pd.util.hash_pandas_object(frame[fields], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _config_hash(config: RankerConfig) -> str:
    encoded = json.dumps(
        config.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def train_model(
    prices: pd.DataFrame,
    train_end: object,
    config: RankerConfig | None = None,
    train_start: object | None = None,
) -> TrainingResult:
    settings = config or RankerConfig()
    end = pd.Timestamp(train_end).normalize()
    start = pd.Timestamp(train_start or settings.regime_start).normalize()
    if end < start:
        raise DataValidationError("train_end precedes train_start")

    canonical = normalize_daily_prices(prices)
    history = canonical.loc[canonical["date"].le(end)].copy()
    panel = build_feature_panel(history, settings)
    training = panel[
        panel["date"].between(start, end)
        & panel["training_eligible"]
        & panel["label"].notna()
    ].copy()
    estimator = fit_estimator(training, settings)
    scores = estimator.predict_proba(training[list(FEATURE_COLUMNS)])[:, 1]
    labels = training["label"].astype(int)
    metrics = {
        "scope": "in_sample_diagnostic_only",
        "rows": int(len(training)),
        "days": int(training["date"].nunique()),
        "codes": int(training["code"].nunique()),
        "label_rate": float(labels.mean()),
        "auc": float(roc_auc_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "score_is_calibrated_probability": False,
    }
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    config_hash = _config_hash(settings)
    data_hash = _data_hash(history)
    run_id = f"session-v2-{end:%Y%m%d}-{config_hash[:8]}-{data_hash[:8]}"
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at_utc": created_at,
        "feature_set": settings.feature_set,
        "training_start": str(start.date()),
        "training_end": str(end.date()),
        "training_rows": int(len(training)),
        "training_days": int(training["date"].nunique()),
        "training_codes": int(training["code"].nunique()),
        "input_max_date": str(pd.to_datetime(history["date"]).max().date()),
        "training_data_sha256": data_hash,
        "config_sha256": config_hash,
        "python_runtime": __import__("sys").version.split()[0],
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "metrics": metrics,
        "calibration_status": "uncalibrated",
    }
    artifact = ModelArtifact(
        estimator=estimator,
        config=settings,
        manifest=manifest,
        feature_columns=FEATURE_COLUMNS,
    )
    artifact.validate()
    return TrainingResult(artifact=artifact, metrics=metrics)
