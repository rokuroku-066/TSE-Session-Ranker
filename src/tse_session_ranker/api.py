from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .artifact import ModelArtifact, load_artifact
from .backtest import WalkForwardResult, monthly_walk_forward
from .config import RankerConfig
from .data.jpx import collect_jpx, download_jpx_urls
from .data.preopen import normalize_preopen_snapshots
from .inference import PredictionResult, predict_candidates
from .io import read_frame, write_frame, write_json
from .training import TrainingResult, train_model


def _frame(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else read_frame(value)


class SessionRanker:
    """Facade covering collection, fixed-spec training, validation, and inference."""

    def __init__(
        self,
        config: RankerConfig | None = None,
        artifact: ModelArtifact | None = None,
    ) -> None:
        self.config = config or (artifact.config if artifact else RankerConfig())
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
            manifest_target = manifest_path or target.with_suffix(
                target.suffix + ".manifest.json"
            )
            write_json(manifest, manifest_target)
        elif manifest_path is not None:
            write_json(manifest, manifest_path)
        return prices, manifest

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

    def train(
        self,
        daily_prices: pd.DataFrame | str | Path,
        train_end: object,
        artifact_path: str | Path | None = None,
        train_start: object | None = None,
    ) -> TrainingResult:
        result = train_model(
            _frame(daily_prices),
            train_end=train_end,
            train_start=train_start,
            config=self.config,
        )
        self.artifact = result.artifact
        if artifact_path is not None:
            result.artifact.save(artifact_path)
        return result

    def predict(
        self,
        daily_prices: pd.DataFrame | str | Path,
        target_date: object,
        top_k: int | None = None,
        preopen_snapshots: pd.DataFrame | str | Path | None = None,
        as_of: object | None = None,
        expected_history_date: object | None = None,
    ) -> PredictionResult:
        if self.artifact is None:
            raise RuntimeError("load or train a model artifact before prediction")
        snapshots = (
            None if preopen_snapshots is None else _frame(preopen_snapshots)
        )
        return predict_candidates(
            self.artifact,
            _frame(daily_prices),
            target_date=target_date,
            top_k=top_k,
            snapshots=snapshots,
            as_of=as_of,
            expected_history_date=expected_history_date,
        )

    def backtest(
        self,
        daily_prices: pd.DataFrame | str | Path,
        evaluation_start: object,
        evaluation_end: object,
        train_start: object | None = None,
    ) -> WalkForwardResult:
        return monthly_walk_forward(
            _frame(daily_prices),
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            train_start=train_start,
            config=self.config,
        )
