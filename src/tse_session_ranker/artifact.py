from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from .config import RankerConfig
from .exceptions import ArtifactError
from .features import FEATURE_COLUMNS, validate_feature_columns
from .io import write_json


ARTIFACT_SCHEMA_VERSION = 2
PACKAGE_VERSION = "0.1.0"


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
        if "training_end" not in self.manifest or "run_id" not in self.manifest:
            raise ArtifactError("artifact training manifest is incomplete")

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
    if verify_checksum:
        if not sidecar.exists():
            raise ArtifactError(f"artifact manifest does not exist: {sidecar}")
        import json

        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
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
    return artifact
