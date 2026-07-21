from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import sklearn

from .config import RankerConfig
from .exceptions import ArtifactError
from .features import FEATURE_COLUMNS, validate_feature_columns
from .io import write_json


ARTIFACT_SCHEMA_VERSION = 4
PACKAGE_VERSION = "0.2.0"


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
        if self.config.feature_set != "session_v2":
            raise ArtifactError(
                f"artifact feature_set is {self.config.feature_set!r}, expected 'session_v2'"
            )
        if not hasattr(self.estimator, "predict_proba"):
            raise ArtifactError("artifact estimator does not support predict_proba")
        required_manifest = {
            "training_end",
            "run_id",
            "selection_objective",
            "data_semantics",
            "score_semantics",
            "sklearn_version",
            "session_calendar_mode",
            "session_calendar_sha256",
        }
        if not required_manifest.issubset(self.manifest):
            raise ArtifactError("artifact training manifest is incomplete")
        if self.manifest["selection_objective"] != self.config.selection_objective:
            raise ArtifactError("artifact selection objective does not match config")
        if self.manifest["data_semantics"] != self.config.data_semantics:
            raise ArtifactError("artifact data semantics do not match config")

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
