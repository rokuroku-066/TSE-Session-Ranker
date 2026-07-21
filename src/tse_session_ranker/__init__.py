"""Public API for :mod:`tse_session_ranker`."""

from .api import SessionRanker
from .artifact import ModelArtifact, load_artifact
from .backtest import WalkForwardResult
from .config import ModelConfig, PreopenPolicy, RankerConfig, UniversePolicy
from .exceptions import ArtifactError, DataValidationError, LeakageError, SessionRankerError
from .features import (
    FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
    build_feature_panel,
    build_inference_frame,
)
from .inference import PredictionResult
from .data.market_context import append_market_context, normalize_market_context
from .profit import daily_portfolio_returns, profit_metrics
from .training import TrainingResult

__all__ = [
    "FEATURE_COLUMNS",
    "PRICE_FEATURE_COLUMNS",
    "ModelArtifact",
    "ModelConfig",
    "PredictionResult",
    "PreopenPolicy",
    "RankerConfig",
    "SessionRanker",
    "SessionRankerError",
    "DataValidationError",
    "LeakageError",
    "ArtifactError",
    "TrainingResult",
    "UniversePolicy",
    "WalkForwardResult",
    "build_feature_panel",
    "build_inference_frame",
    "load_artifact",
    "daily_portfolio_returns",
    "profit_metrics",
    "append_market_context",
    "normalize_market_context",
]

__version__ = "0.3.0"
