from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RankerConfig
from .exceptions import ArtifactError
from .features import FEATURE_COLUMNS, validate_feature_columns
from .io import write_json


ARTIFACT_SCHEMA_VERSION = 5
PACKAGE_VERSION = "0.3.0"


def _estimator_params_match(
    actual: dict[str, Any], expected: dict[str, Any]
) -> bool:
    if set(actual) != set(expected):
        return False
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(actual_value, float) and isinstance(expected_value, float):
            if math.isnan(actual_value) and math.isnan(expected_value):
                continue
        if actual_value != expected_value:
            return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


@dataclass
class ModelArtifact:
    estimator: Any
    config: RankerConfig
    manifest: dict[str, Any]
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    package_version: str = PACKAGE_VERSION

    def validate(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactError(
                f"unsupported artifact schema {self.schema_version}; "
                f"expected {ARTIFACT_SCHEMA_VERSION}"
            )
        if self.package_version != PACKAGE_VERSION:
            raise ArtifactError(
                f"artifact package version {self.package_version!r} is incompatible "
                f"with {PACKAGE_VERSION!r}"
            )
        try:
            validate_feature_columns(self.feature_columns)
        except Exception as exc:
            raise ArtifactError(str(exc)) from exc
        if self.config.feature_set != "session_v3_tdnet_clear":
            raise ArtifactError(
                "artifact feature_set is "
                f"{self.config.feature_set!r}, expected 'session_v3_tdnet_clear'"
            )
        if not isinstance(self.estimator, Pipeline) or list(
            self.estimator.named_steps
        ) != ["impute", "scale", "model"]:
            raise ArtifactError("artifact is not the fixed session_v3 pipeline")
        imputer = self.estimator.named_steps["impute"]
        scaler = self.estimator.named_steps["scale"]
        model = self.estimator.named_steps["model"]
        if not (
            isinstance(imputer, SimpleImputer)
            and isinstance(scaler, StandardScaler)
            and isinstance(model, LogisticRegression)
        ):
            raise ArtifactError(
                "artifact is not the fixed regularized logistic pipeline"
            )
        expected_model = self.config.model
        expected_imputer_params = SimpleImputer(
            strategy="median", add_indicator=True
        ).get_params(deep=False)
        expected_scaler_params = StandardScaler().get_params(deep=False)
        expected_model_params = LogisticRegression(
            C=expected_model.c,
            class_weight=expected_model.class_weight,
            max_iter=expected_model.max_iter,
            random_state=expected_model.random_state,
        ).get_params(deep=False)
        if not _estimator_params_match(
            imputer.get_params(deep=False), expected_imputer_params
        ):
            raise ArtifactError(
                "artifact imputer parameters do not match the fixed config"
            )
        if not _estimator_params_match(
            scaler.get_params(deep=False), expected_scaler_params
        ):
            raise ArtifactError(
                "artifact scaler parameters do not match the fixed config"
            )
        if not _estimator_params_match(
            model.get_params(deep=False), expected_model_params
        ):
            raise ArtifactError(
                "artifact logistic parameters do not match the fixed config"
            )
        required_manifest = {
            "training_end",
            "run_id",
            "feature_set",
            "selection_objective",
            "data_semantics",
            "score_semantics",
            "sklearn_version",
            "session_calendar_mode",
            "session_calendar_sha256",
            "tdnet_source_sha256",
            "tdnet_complete_through",
            "tdnet_decision_time",
        }
        if not required_manifest.issubset(self.manifest):
            raise ArtifactError("artifact training manifest is incomplete")
        if self.manifest["selection_objective"] != self.config.selection_objective:
            raise ArtifactError("artifact selection objective does not match config")
        if self.manifest["data_semantics"] != self.config.data_semantics:
            raise ArtifactError("artifact data semantics do not match config")
        if self.manifest["feature_set"] != self.config.feature_set:
            raise ArtifactError("artifact feature set does not match config")
        if self.manifest["tdnet_decision_time"] != self.config.preopen.decision_time:
            raise ArtifactError("artifact TDnet decision time does not match config")
        input_count = getattr(self.estimator, "n_features_in_", None)
        if input_count != len(self.feature_columns):
            raise ArtifactError(
                "artifact estimator input dimension does not match feature manifest"
            )

    def save(self, path: str | Path) -> Path:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=target.name + ".",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            joblib.dump(self, temporary)
            digest = _sha256(temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        public_manifest = {
            **self.manifest,
            "artifact_schema_version": self.schema_version,
            "package_version": self.package_version,
            "feature_columns": list(self.feature_columns),
            "config": self.config.to_dict(),
            "artifact_sha256": digest,
        }
        write_json(public_manifest, _sidecar(target))
        return target


def load_artifact(path: str | Path, verify_checksum: bool = True) -> ModelArtifact:
    """Load a trusted local joblib artifact and validate its sidecar checksum."""

    target = Path(path)
    if not target.exists():
        raise ArtifactError(f"model artifact does not exist: {target}")
    sidecar = _sidecar(target)
    metadata: dict[str, Any] | None = None
    if verify_checksum:
        if not sidecar.exists():
            raise ArtifactError(f"artifact manifest does not exist: {sidecar}")
        try:
            loaded_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"could not read artifact manifest: {exc}") from exc
        if not isinstance(loaded_metadata, dict):
            raise ArtifactError("artifact manifest must be a JSON object")
        metadata = loaded_metadata
        expected = metadata.get("artifact_sha256")
        actual = _sha256(target)
        if not expected or expected != actual:
            raise ArtifactError("artifact checksum does not match its manifest")
    try:
        artifact = joblib.load(target)
    except Exception as exc:
        raise ArtifactError(f"could not load artifact: {exc}") from exc
    if not isinstance(artifact, ModelArtifact):
        raise ArtifactError("joblib payload is not a ModelArtifact")
    artifact.validate()
    trained_sklearn = str(artifact.manifest["sklearn_version"])
    trained_minor = tuple(trained_sklearn.split(".")[:2])
    runtime_minor = tuple(sklearn.__version__.split(".")[:2])
    if trained_minor != runtime_minor:
        raise ArtifactError(
            "artifact scikit-learn minor version is incompatible: "
            f"trained={trained_sklearn}, runtime={sklearn.__version__}"
        )
    if metadata is not None:
        expected_metadata = {
            "artifact_schema_version": artifact.schema_version,
            "package_version": artifact.package_version,
            "feature_columns": list(artifact.feature_columns),
            "config": artifact.config.to_dict(),
            "run_id": artifact.manifest["run_id"],
            "training_end": artifact.manifest["training_end"],
            "selection_objective": artifact.manifest["selection_objective"],
            "data_semantics": artifact.manifest["data_semantics"],
            "score_semantics": artifact.manifest["score_semantics"],
            "feature_set": artifact.manifest["feature_set"],
            "config_sha256": artifact.manifest.get("config_sha256"),
            "training_data_sha256": artifact.manifest.get("training_data_sha256"),
            "sklearn_version": artifact.manifest["sklearn_version"],
            "session_calendar_mode": artifact.manifest["session_calendar_mode"],
            "session_calendar_sha256": artifact.manifest[
                "session_calendar_sha256"
            ],
            "session_calendar_through": artifact.manifest.get(
                "session_calendar_through"
            ),
            "tdnet_source_sha256": artifact.manifest["tdnet_source_sha256"],
            "tdnet_complete_through": artifact.manifest[
                "tdnet_complete_through"
            ],
            "tdnet_decision_time": artifact.manifest["tdnet_decision_time"],
        }
        mismatches = [
            key
            for key, expected_value in expected_metadata.items()
            if metadata.get(key) != expected_value
        ]
        if mismatches:
            raise ArtifactError(
                "artifact manifest disagrees with payload: " + ", ".join(mismatches)
            )
    return artifact
