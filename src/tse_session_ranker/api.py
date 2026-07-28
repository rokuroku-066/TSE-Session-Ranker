from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .artifact import ModelArtifact, load_artifact
from .backtest import WalkForwardResult, monthly_walk_forward
from .config import RankerConfig
from .data.jpx import collect_jpx, download_jpx_urls
from .data.market_context import (
    MARKET_CONTEXT_COLUMNS,
    append_market_context,
)
from .data.preopen import normalize_preopen_snapshots
from .data.tdnet import (
    TDnetDataset,
    collect_tdnet_dataset,
    download_tdnet_indexes,
    normalize_tdnet_disclosures,
    require_production_tdnet_provenance,
    tdnet_dataset_digest,
)
from .data.tdnet_material import (
    collect_official_forecast_revisions,
    download_official_tdnet_urls,
)
from .exceptions import ArtifactError, DataValidationError
from .inference import PredictionResult, predict_candidates
from .io import read_frame, write_frame, write_json
from .training import TrainingResult, train_model


def _frame(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else read_frame(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tdnet_manifest(dataset: TDnetDataset) -> dict[str, object]:
    return {
        "schema_version": 2,
        "source_sha256": dataset.source_sha256,
        "dataset_sha256": tdnet_dataset_digest(dataset),
        "source_files": dataset.source_files,
        "complete_dates": [str(value.date()) for value in dataset.complete_dates],
        "observed_at_by_date": {
            str(pd.Timestamp(date).date()): pd.Timestamp(observed_at).isoformat()
            for date, observed_at in dataset.observed_at_by_date.items()
        },
        "provenance_by_date": {
            str(pd.Timestamp(date).date()): str(provenance)
            for date, provenance in dataset.provenance_by_date.items()
        },
        "disclosures": len(dataset.disclosures),
    }


def _load_tdnet_export(path: Path) -> TDnetDataset:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise DataValidationError(
            f"TDnet export manifest does not exist: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError("TDnet export manifest is unreadable") from exc
    required = {
        "schema_version",
        "source_sha256",
        "dataset_sha256",
        "source_files",
        "complete_dates",
        "observed_at_by_date",
        "provenance_by_date",
        "disclosures",
        "export_sha256",
    }
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        raise DataValidationError("TDnet export manifest is incomplete")
    if manifest["schema_version"] != 2:
        raise DataValidationError("unsupported TDnet export manifest schema")
    if manifest["export_sha256"] != _sha256_file(path):
        raise DataValidationError("TDnet export checksum does not match its manifest")
    disclosures = normalize_tdnet_disclosures(read_frame(path))
    if len(disclosures) != int(manifest["disclosures"]):
        raise DataValidationError("TDnet export row count does not match its manifest")
    complete_dates = pd.DatetimeIndex(pd.to_datetime(manifest["complete_dates"]))
    observed_at = pd.Series(
        {
            pd.Timestamp(date).normalize(): pd.Timestamp(value)
            for date, value in manifest["observed_at_by_date"].items()
        },
        name="observed_at",
    ).sort_index()
    if set(observed_at.index) != set(complete_dates):
        raise DataValidationError("TDnet export observation dates are incomplete")
    provenance = pd.Series(
        {
            pd.Timestamp(date).normalize(): str(value)
            for date, value in manifest["provenance_by_date"].items()
        },
        name="provenance",
        dtype="object",
    ).sort_index()
    if set(provenance.index) != set(complete_dates):
        raise DataValidationError("TDnet export provenance dates are incomplete")
    dataset = TDnetDataset(
        disclosures=disclosures,
        complete_dates=complete_dates,
        observed_at_by_date=observed_at,
        source_sha256=str(manifest["source_sha256"]),
        source_files=int(manifest["source_files"]),
        provenance_by_date=provenance,
    )
    if tdnet_dataset_digest(dataset) != str(manifest["dataset_sha256"]):
        raise DataValidationError(
            "TDnet export metadata does not match its dataset checksum"
        )
    require_production_tdnet_provenance(dataset)
    return dataset


def _tdnet(
    value: TDnetDataset | Iterable[str | Path] | str | Path,
) -> TDnetDataset:
    if isinstance(value, TDnetDataset):
        return value
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.is_file() and path.suffix.lower() != ".html":
            return _load_tdnet_export(path)
    return collect_tdnet_dataset(value)


class SessionRanker:
    """Facade covering collection, fixed-spec training, validation, and inference."""

    def __init__(
        self,
        config: RankerConfig | None = None,
        artifact: ModelArtifact | None = None,
    ) -> None:
        if artifact is not None and config is not None and config != artifact.config:
            raise ArtifactError("explicit config does not match the model artifact config")
        self.config = artifact.config if artifact is not None else (config or RankerConfig())
        self.artifact = artifact

    @classmethod
    def from_artifact(
        cls, path: str | Path, verify_checksum: bool = True
    ) -> "SessionRanker":
        artifact = load_artifact(path, verify_checksum=verify_checksum)
        return cls(config=artifact.config, artifact=artifact)

    @classmethod
    def from_config(cls, path: str | Path) -> "SessionRanker":
        return cls(config=RankerConfig.from_json(path))

    def download_jpx(
        self,
        urls: Iterable[str] | str,
        destination: str | Path,
        overwrite: bool = False,
    ) -> list[dict[str, object]]:
        return download_jpx_urls(urls, destination, overwrite=overwrite)

    def collect_jpx(
        self,
        inputs: Iterable[str | Path] | str | Path,
        output: str | Path | None = None,
        existing: pd.DataFrame | str | Path | None = None,
        manifest_path: str | Path | None = None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        existing_frame = None if existing is None else _frame(existing)
        prices, manifest = collect_jpx(inputs, existing=existing_frame)
        if output is not None:
            target = write_frame(prices, output)
            manifest = {
                **manifest,
                "schema_version": 2,
                "export_path": str(target),
                "export_sha256": _sha256_file(target),
            }
            manifest_target = manifest_path or target.with_suffix(
                target.suffix + ".manifest.json"
            )
            write_json(manifest, manifest_target)
        elif manifest_path is not None:
            write_json(manifest, manifest_path)
        return prices, manifest

    def download_tdnet(
        self,
        start_date: object,
        end_date: object,
        destination: str | Path,
        *,
        overwrite: bool = False,
        max_workers: int = 4,
    ) -> list[dict[str, object]]:
        return download_tdnet_indexes(
            start_date,
            end_date,
            destination,
            overwrite=overwrite,
            max_workers=max_workers,
        )

    def download_official_tdnet(
        self,
        urls: Iterable[str] | str,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> list[dict[str, object]]:
        """Download explicit official TDnet index/PDF URLs with receipts."""

        return [
            receipt.to_dict()
            for receipt in download_official_tdnet_urls(
                urls,
                destination,
                overwrite=overwrite,
            )
        ]

    def probe_tdnet_material(
        self,
        index_url: str,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> dict[str, object]:
        """Acquire and strictly parse current official forecast revisions."""

        return collect_official_forecast_revisions(
            index_url,
            destination,
            overwrite=overwrite,
        )

    def collect_tdnet(
        self,
        inputs: Iterable[str | Path] | str | Path,
        output: str | Path | None = None,
        manifest_path: str | Path | None = None,
    ) -> TDnetDataset:
        dataset = collect_tdnet_dataset(inputs)
        manifest = _tdnet_manifest(dataset)
        if output is not None:
            target = write_frame(dataset.disclosures, output)
            canonical_manifest = target.with_suffix(
                target.suffix + ".manifest.json"
            )
            manifest["export_sha256"] = _sha256_file(target)
            write_json(manifest, canonical_manifest)
            if (
                manifest_path is not None
                and Path(manifest_path) != canonical_manifest
            ):
                write_json(manifest, manifest_path)
        elif manifest_path is not None:
            write_json(manifest, manifest_path)
        return dataset

    def ingest_preopen(
        self,
        snapshots: pd.DataFrame | str | Path,
        output: str | Path | None = None,
        existing: pd.DataFrame | str | Path | None = None,
    ) -> pd.DataFrame:
        frames = []
        if existing is not None:
            frames.append(_frame(existing))
        frames.append(_frame(snapshots))
        combined = normalize_preopen_snapshots(
            pd.concat(frames, ignore_index=True, sort=False),
            timezone=self.config.preopen.timezone,
        )
        if output is not None:
            write_frame(combined, output)
        return combined

    def ingest_market_context(
        self,
        snapshots: pd.DataFrame | str | Path,
        output: str | Path | None = None,
        existing: pd.DataFrame | str | Path | None = None,
    ) -> pd.DataFrame:
        """Validate and append exact-date pre-open futures context.

        Existing observations are immutable.  New rows must be observed no
        later than the configured pre-open decision time, and missing dates
        are never backfilled or carried forward.
        """

        current = (
            pd.DataFrame(columns=MARKET_CONTEXT_COLUMNS)
            if existing is None
            else _frame(existing)
        )
        combined = append_market_context(
            current,
            _frame(snapshots),
            decision_time=self.config.preopen.decision_time,
        )
        if output is not None:
            write_frame(combined, output)
        return combined

    def train(
        self,
        daily_prices: pd.DataFrame | str | Path,
        train_end: object,
        tdnet_indexes: TDnetDataset | Iterable[str | Path] | str | Path,
        artifact_path: str | Path | None = None,
        train_start: object | None = None,
        expected_sessions: object | None = None,
    ) -> TrainingResult:
        result = train_model(
            _frame(daily_prices),
            train_end=train_end,
            tdnet_dataset=_tdnet(tdnet_indexes),
            train_start=train_start,
            config=self.config,
            expected_sessions=expected_sessions,
        )
        self.artifact = result.artifact
        if artifact_path is not None:
            result.artifact.save(artifact_path)
        return result

    def predict(
        self,
        daily_prices: pd.DataFrame | str | Path,
        target_date: object,
        tdnet_indexes: TDnetDataset | Iterable[str | Path] | str | Path,
        top_k: int | None = None,
        preopen_snapshots: pd.DataFrame | str | Path | None = None,
        as_of: object | None = None,
        expected_history_date: object | None = None,
        expected_sessions: object | None = None,
    ) -> PredictionResult:
        if self.artifact is None:
            raise ArtifactError("load or train a model artifact before prediction")
        snapshots = (
            None if preopen_snapshots is None else _frame(preopen_snapshots)
        )
        return predict_candidates(
            self.artifact,
            _frame(daily_prices),
            target_date=target_date,
            tdnet_dataset=_tdnet(tdnet_indexes),
            top_k=top_k,
            snapshots=snapshots,
            as_of=as_of,
            expected_history_date=expected_history_date,
            expected_sessions=expected_sessions,
        )

    def backtest(
        self,
        daily_prices: pd.DataFrame | str | Path,
        evaluation_start: object,
        evaluation_end: object,
        tdnet_indexes: TDnetDataset | Iterable[str | Path] | str | Path,
        train_start: object | None = None,
        expected_sessions: object | None = None,
    ) -> WalkForwardResult:
        return monthly_walk_forward(
            _frame(daily_prices),
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            tdnet_dataset=_tdnet(tdnet_indexes),
            train_start=train_start,
            config=self.config,
            expected_sessions=expected_sessions,
        )
