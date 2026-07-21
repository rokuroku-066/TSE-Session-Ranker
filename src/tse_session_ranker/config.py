from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import time
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UniversePolicy:
    """Eligibility policy kept separate from the statistical model."""

    min_history: int = 60
    price_min: float = 100.0
    price_max: float = 30_000.0
    atr14_min_pct: float = 0.25
    atr14_max_pct: float = 5.0
    model_training_atr_max_pct: float = 15.0
    max_zero_oc_20: float = 0.20

    def __post_init__(self) -> None:
        if self.min_history < 20:
            raise ValueError("min_history must be at least 20")
        if not 0 < self.price_min < self.price_max:
            raise ValueError("price bounds are invalid")
        if not 0 <= self.atr14_min_pct < self.atr14_max_pct:
            raise ValueError("ATR bounds are invalid")
        if self.model_training_atr_max_pct < self.atr14_max_pct:
            raise ValueError(
                "model_training_atr_max_pct cannot be below the selection ATR cap"
            )
        if not 0 <= self.max_zero_oc_20 <= 1:
            raise ValueError("max_zero_oc_20 must be in [0, 1]")


@dataclass(frozen=True)
class ModelConfig:
    algorithm: str = "logistic_regression"
    c: float = 0.08
    class_weight: str | None = "balanced"
    random_state: int = 31
    max_iter: int = 1_000
    date_equal_weight: bool = True

    def __post_init__(self) -> None:
        if self.algorithm != "logistic_regression":
            raise ValueError("only logistic_regression is supported in session_v2")
        if self.c <= 0:
            raise ValueError("c must be positive")


@dataclass(frozen=True)
class PreopenPolicy:
    """Provisional execution overlay; it never changes the model ranking."""

    timezone: str = "Asia/Tokyo"
    decision_time: str = "08:58:59"
    max_snapshot_age_seconds: int = 180
    positive_gap_atr_veto: float = 2.0
    latest_expected_open_time: str = "09:05:00"

    def __post_init__(self) -> None:
        if self.max_snapshot_age_seconds < 0:
            raise ValueError("max_snapshot_age_seconds must not be negative")
        if self.positive_gap_atr_veto <= 0:
            raise ValueError("positive_gap_atr_veto must be positive")
        for field_name, value in (
            ("decision_time", self.decision_time),
            ("latest_expected_open_time", self.latest_expected_open_time),
        ):
            try:
                time.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"{field_name} must be an ISO local time") from exc


@dataclass(frozen=True)
class RankerConfig:
    schema_version: int = 2
    feature_set: str = "session_v2"
    regime_start: str = "2024-11-06"
    display_top_k: int = 2
    trade_top_k: int = 1
    cost_bps: float = 20.0
    selection_objective: str = "top1_net_mean_pct_at_cost"
    data_semantics: str = "prior_session_universe_source_mask_v2"
    source_coverage_lookback: int = 20
    minimum_source_coverage: float = 0.90
    max_history_age_calendar_days: int = 7
    min_latest_session_coverage: float = 0.90
    require_expected_history_date: bool = True
    universe: UniversePolicy = field(default_factory=UniversePolicy)
    model: ModelConfig = field(default_factory=ModelConfig)
    preopen: PreopenPolicy = field(default_factory=PreopenPolicy)

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("unsupported config schema_version")
        if self.feature_set != "session_v2":
            raise ValueError("only session_v2 is currently supported")
        if self.display_top_k < 1 or self.trade_top_k < 0:
            raise ValueError("selection counts are invalid")
        if self.trade_top_k > self.display_top_k:
            raise ValueError("trade_top_k cannot exceed display_top_k")
        if self.trade_top_k != 1:
            raise ValueError("profit-first session_v2 requires trade_top_k=1")
        if not math.isfinite(self.cost_bps) or self.cost_bps < 0:
            raise ValueError("cost_bps must be finite and non-negative")
        if self.max_history_age_calendar_days < 1:
            raise ValueError("max_history_age_calendar_days must be positive")
        if self.selection_objective != "top1_net_mean_pct_at_cost":
            raise ValueError("unsupported selection_objective")
        if self.data_semantics != "prior_session_universe_source_mask_v2":
            raise ValueError("unsupported data_semantics")
        if self.source_coverage_lookback < 10:
            raise ValueError("source_coverage_lookback must be at least 10")
        if not 0 < self.minimum_source_coverage <= 1:
            raise ValueError("minimum_source_coverage must be in (0, 1]")
        if not 0 < self.min_latest_session_coverage <= 1:
            raise ValueError("min_latest_session_coverage must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RankerConfig":
        values = dict(raw)
        values["universe"] = UniversePolicy(**values.get("universe", {}))
        values["model"] = ModelConfig(**values.get("model", {}))
        values["preopen"] = PreopenPolicy(**values.get("preopen", {}))
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "RankerConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
