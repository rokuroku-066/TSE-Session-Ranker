"""Time-series validation primitives for daily strategy returns.

The functions in this module deliberately operate on *daily portfolio return*
series, not on security-level observations.  Trading days are the independent
unit used by the research protocol.  Confidence intervals therefore use a
moving-block bootstrap; an IID fallback is intentionally not provided.

Return values are expressed in percentage points (``1.0`` means one percent),
matching :mod:`tse_session_ranker.profit`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_BLOCK_LENGTHS: tuple[int, ...] = (5, 10, 20)


@dataclass(frozen=True)
class ReturnDiagnostics:
    """Deterministic diagnostics for one ordered daily return series."""

    observations: int
    mean_pct: float
    median_pct: float
    top_k: int
    top_k_removed_mean_pct: float
    profit_factor: float
    max_drawdown_pct: float
    positive_periods: int
    periods: int
    positive_period_ratio: float
    max_positive_contribution: float


@dataclass(frozen=True)
class MeanBootstrapInterval:
    """Moving-block bootstrap interval for a daily mean."""

    observations: int
    block_length: int
    samples: int
    confidence: float
    random_state: int
    point_estimate_pct: float
    one_sided_lower_pct: float
    two_sided_lower_pct: float
    two_sided_upper_pct: float
    bootstrap_standard_error_pct: float


@dataclass(frozen=True)
class PairedBootstrapInterval:
    """Moving-block interval for candidate minus baseline daily returns."""

    observations: int
    block_length: int
    samples: int
    confidence: float
    random_state: int
    candidate_mean_pct: float
    baseline_mean_pct: float
    point_estimate_delta_pct: float
    one_sided_lower_delta_pct: float
    two_sided_lower_delta_pct: float
    two_sided_upper_delta_pct: float
    bootstrap_standard_error_delta_pct: float


@dataclass(frozen=True)
class ValidationGate:
    """Predeclared thresholds for a development or holdout decision.

    ``period_frequency`` follows pandas' period aliases.  ``"M"`` therefore
    means that the positive-period ratio is calculated from monthly mean daily
    returns.  Every configured bootstrap block length must clear the lower
    bound threshold.
    """

    name: str
    min_observations: int
    top_k: int = 5
    period_frequency: str | None = "M"
    min_mean_pct: float = 0.0
    min_top_k_removed_mean_pct: float = 0.0
    min_profit_factor: float = 1.0
    min_positive_period_ratio: float = 0.60
    max_positive_contribution: float = 0.25
    min_one_sided_lower_pct: float = 0.0
    block_lengths: tuple[int, ...] = DEFAULT_BLOCK_LENGTHS
    confidence: float = 0.90
    bootstrap_samples: int = 20_000
    random_state: int = 31
    min_paired_delta_pct: float | None = None
    min_paired_one_sided_lower_delta_pct: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("gate name must not be empty")
        if self.min_observations < 1:
            raise ValueError("min_observations must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 <= self.min_positive_period_ratio <= 1.0:
            raise ValueError("min_positive_period_ratio must be in [0, 1]")
        if not 0.0 <= self.max_positive_contribution <= 1.0:
            raise ValueError("max_positive_contribution must be in [0, 1]")
        _validate_bootstrap_parameters(
            self.block_lengths,
            self.bootstrap_samples,
            self.confidence,
        )


@dataclass(frozen=True)
class GateCheck:
    """One auditable predicate contributing to a gate decision."""

    name: str
    passed: bool
    actual: float
    operator: str
    required: float


@dataclass(frozen=True)
class GateDecision:
    """Complete, reproducible development/holdout gate result."""

    gate: ValidationGate
    passed: bool
    diagnostics: ReturnDiagnostics
    bootstrap: Mapping[int, MeanBootstrapInterval]
    paired_bootstrap: Mapping[int, PairedBootstrapInterval] | None
    checks: tuple[GateCheck, ...]
    failure_reasons: tuple[str, ...]


def development_gate(
    *,
    min_observations: int = 252,
    bootstrap_samples: int = 20_000,
    random_state: int = 31,
) -> ValidationGate:
    """Return the predeclared conservative development gate."""

    return ValidationGate(
        name="development",
        min_observations=min_observations,
        min_positive_period_ratio=0.60,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )


def holdout_gate(
    *,
    min_observations: int = 120,
    bootstrap_samples: int = 20_000,
    random_state: int = 31,
    require_paired_improvement: bool = True,
) -> ValidationGate:
    """Return the predeclared final holdout gate."""

    return ValidationGate(
        name="holdout",
        min_observations=min_observations,
        min_positive_period_ratio=0.625,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
        min_paired_delta_pct=0.0 if require_paired_improvement else None,
    )


def _coerce_daily_returns(
    values: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    return_column: str = "net_return_pct",
    date_column: str = "date",
    nan_policy: str = "raise",
) -> pd.Series:
    if nan_policy not in {"raise", "drop"}:
        raise ValueError("nan_policy must be 'raise' or 'drop'")
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
        if return_column in frame:
            series = frame[return_column].copy()
        else:
            candidates = [column for column in frame.columns if column != date_column]
            if len(candidates) != 1:
                raise ValueError(
                    f"DataFrame must contain {return_column!r} or exactly one value column"
                )
            series = frame[candidates[0]].copy()
        if date_column in frame:
            parsed = pd.to_datetime(frame[date_column], errors="coerce")
            if parsed.isna().any():
                raise ValueError("date column contains invalid values")
            series.index = pd.DatetimeIndex(parsed)
    elif isinstance(values, pd.Series):
        series = values.copy()
    else:
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError("daily returns must be one-dimensional")
        series = pd.Series(array)

    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    non_finite = ~np.isfinite(numeric.to_numpy())
    if non_finite.any():
        if nan_policy == "raise":
            raise ValueError("daily returns contain NaN or infinite values")
        numeric = numeric.loc[~non_finite]
    if numeric.empty:
        raise ValueError("daily returns are empty")
    if (numeric < -100.0).any():
        raise ValueError("a percentage return cannot be below -100")
    if numeric.index.has_duplicates:
        raise ValueError("daily return index contains duplicate observations")
    if isinstance(numeric.index, pd.DatetimeIndex):
        numeric = numeric.sort_index(kind="stable")
    return numeric.astype(float)


def top_k_removed_mean(
    returns: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    k: int = 5,
    return_column: str = "net_return_pct",
    nan_policy: str = "raise",
) -> float:
    """Mean after removing the ``k`` largest daily returns.

    If every observation is removed the result is explicitly ``NaN``; callers
    can then fail a validation gate rather than treating an empty remainder as
    evidence of robustness.
    """

    if k < 0:
        raise ValueError("k must be non-negative")
    series = _coerce_daily_returns(
        returns, return_column=return_column, nan_policy=nan_policy
    )
    if k == 0:
        return float(series.mean())
    if k >= len(series):
        return float("nan")
    return float(series.drop(series.nlargest(k).index).mean())


def profit_factor(
    returns: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    return_column: str = "net_return_pct",
    nan_policy: str = "raise",
) -> float:
    """Gross gains divided by absolute gross losses.

    All-loss or zero-only series return ``0.0``.  A series with gains and no
    losses returns positive infinity.
    """

    series = _coerce_daily_returns(
        returns, return_column=return_column, nan_policy=nan_policy
    )
    gains = float(series.clip(lower=0.0).sum())
    losses = float(-series.clip(upper=0.0).sum())
    if losses == 0.0:
        return float("inf") if gains > 0.0 else 0.0
    return gains / losses


def maximum_drawdown_pct(
    returns: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    return_column: str = "net_return_pct",
    nan_policy: str = "raise",
) -> float:
    """Maximum compounded drawdown, including the initial unit NAV."""

    series = _coerce_daily_returns(
        returns, return_column=return_column, nan_policy=nan_policy
    )
    equity = np.cumprod(1.0 + series.to_numpy(dtype=float) / 100.0)
    equity = np.concatenate(([1.0], equity))
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    return float(100.0 * drawdowns.min())


def positive_period_ratio(
    returns: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    frequency: str | None = "M",
    return_column: str = "net_return_pct",
    nan_policy: str = "raise",
) -> tuple[int, int, float]:
    """Return positive periods, total periods, and their ratio.

    Period returns use the mean daily return, which is the project's scheduled-
    day objective.  Set ``frequency=None`` for the positive-day ratio.
    """

    series = _coerce_daily_returns(
        returns, return_column=return_column, nan_policy=nan_policy
    )
    if frequency is None:
        period_values = series
    else:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("a DatetimeIndex is required for period aggregation")
        try:
            period_values = series.groupby(series.index.to_period(frequency)).mean()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid period frequency: {frequency!r}") from exc
    positives = int(period_values.gt(0.0).sum())
    periods = int(len(period_values))
    return positives, periods, positives / periods


def max_positive_contribution(
    returns: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    return_column: str = "net_return_pct",
    nan_policy: str = "raise",
) -> float:
    """Largest positive day as a share of total positive return.

    A series with no positive day returns ``0.0`` explicitly.
    """

    series = _coerce_daily_returns(
        returns, return_column=return_column, nan_policy=nan_policy
    )
    positive = series.clip(lower=0.0)
    total = float(positive.sum())
    return 0.0 if total == 0.0 else float(positive.max() / total)


def return_diagnostics(
    returns: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    top_k: int = 5,
    period_frequency: str | None = "M",
    return_column: str = "net_return_pct",
    nan_policy: str = "raise",
) -> ReturnDiagnostics:
    """Calculate all deterministic robustness diagnostics in one pass."""

    series = _coerce_daily_returns(
        returns, return_column=return_column, nan_policy=nan_policy
    )
    positives, periods, ratio = positive_period_ratio(
        series, frequency=period_frequency
    )
    return ReturnDiagnostics(
        observations=len(series),
        mean_pct=float(series.mean()),
        median_pct=float(series.median()),
        top_k=top_k,
        top_k_removed_mean_pct=top_k_removed_mean(series, k=top_k),
        profit_factor=profit_factor(series),
        max_drawdown_pct=maximum_drawdown_pct(series),
        positive_periods=positives,
        periods=periods,
        positive_period_ratio=ratio,
        max_positive_contribution=max_positive_contribution(series),
    )


def _validate_bootstrap_parameters(
    block_lengths: Sequence[int], samples: int, confidence: float
) -> None:
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if not block_lengths:
        raise ValueError("at least one block length is required")
    for length in block_lengths:
        if not isinstance(length, (int, np.integer)) or int(length) < 2:
            raise ValueError(
                "moving-block length must be an integer >= 2; IID bootstrap is forbidden"
            )


def _moving_block_bootstrap_means(
    values: np.ndarray,
    *,
    block_length: int,
    samples: int,
    random_state: int,
    batch_size: int = 1_000,
) -> np.ndarray:
    observations = len(values)
    if observations < block_length:
        raise ValueError(
            f"block length {block_length} exceeds {observations} observations"
        )
    block_count = math.ceil(observations / block_length)
    max_start = observations - block_length + 1
    offsets = np.arange(block_length, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    result = np.empty(samples, dtype=float)
    position = 0
    while position < samples:
        size = min(batch_size, samples - position)
        starts = rng.integers(0, max_start, size=(size, block_count))
        indices = (starts[..., None] + offsets).reshape(size, -1)
        indices = indices[:, :observations]
        result[position : position + size] = values[indices].mean(axis=1)
        position += size
    return result


def moving_block_bootstrap(
    returns: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    block_length: int = 5,
    samples: int = 20_000,
    confidence: float = 0.90,
    random_state: int = 31,
    return_column: str = "net_return_pct",
    nan_policy: str = "raise",
) -> MeanBootstrapInterval:
    """Estimate a daily-mean interval with overlapping moving blocks."""

    _validate_bootstrap_parameters((block_length,), samples, confidence)
    series = _coerce_daily_returns(
        returns, return_column=return_column, nan_policy=nan_policy
    )
    means = _moving_block_bootstrap_means(
        series.to_numpy(dtype=float),
        block_length=block_length,
        samples=samples,
        random_state=random_state,
    )
    tail = (1.0 - confidence) / 2.0
    return MeanBootstrapInterval(
        observations=len(series),
        block_length=block_length,
        samples=samples,
        confidence=confidence,
        random_state=random_state,
        point_estimate_pct=float(series.mean()),
        one_sided_lower_pct=float(np.quantile(means, 1.0 - confidence)),
        two_sided_lower_pct=float(np.quantile(means, tail)),
        two_sided_upper_pct=float(np.quantile(means, 1.0 - tail)),
        bootstrap_standard_error_pct=float(means.std(ddof=1)),
    )


def moving_block_bootstrap_suite(
    returns: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    block_lengths: Sequence[int] = DEFAULT_BLOCK_LENGTHS,
    samples: int = 20_000,
    confidence: float = 0.90,
    random_state: int = 31,
    return_column: str = "net_return_pct",
    nan_policy: str = "raise",
) -> dict[int, MeanBootstrapInterval]:
    """Run deterministic sensitivity intervals for several block lengths."""

    lengths = tuple(int(value) for value in block_lengths)
    _validate_bootstrap_parameters(lengths, samples, confidence)
    series = _coerce_daily_returns(
        returns, return_column=return_column, nan_policy=nan_policy
    )
    return {
        length: moving_block_bootstrap(
            series,
            block_length=length,
            samples=samples,
            confidence=confidence,
            random_state=random_state + offset,
        )
        for offset, length in enumerate(lengths)
    }


def _aligned_pair(
    candidate: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    baseline: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    candidate_return_column: str,
    baseline_return_column: str,
    nan_policy: str,
) -> tuple[pd.Series, pd.Series]:
    left = _coerce_daily_returns(
        candidate,
        return_column=candidate_return_column,
        nan_policy=nan_policy,
    )
    right = _coerce_daily_returns(
        baseline,
        return_column=baseline_return_column,
        nan_policy=nan_policy,
    )
    if len(left) != len(right) or not left.index.equals(right.index):
        raise ValueError(
            "paired bootstrap requires identical ordered daily observation indexes"
        )
    return left, right


def paired_moving_block_bootstrap(
    candidate: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    baseline: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    block_length: int = 5,
    samples: int = 20_000,
    confidence: float = 0.90,
    random_state: int = 31,
    candidate_return_column: str = "net_return_pct",
    baseline_return_column: str = "net_return_pct",
    nan_policy: str = "raise",
) -> PairedBootstrapInterval:
    """Bootstrap candidate-minus-baseline daily deltas using shared blocks."""

    _validate_bootstrap_parameters((block_length,), samples, confidence)
    left, right = _aligned_pair(
        candidate,
        baseline,
        candidate_return_column=candidate_return_column,
        baseline_return_column=baseline_return_column,
        nan_policy=nan_policy,
    )
    delta = left.to_numpy(dtype=float) - right.to_numpy(dtype=float)
    means = _moving_block_bootstrap_means(
        delta,
        block_length=block_length,
        samples=samples,
        random_state=random_state,
    )
    tail = (1.0 - confidence) / 2.0
    return PairedBootstrapInterval(
        observations=len(delta),
        block_length=block_length,
        samples=samples,
        confidence=confidence,
        random_state=random_state,
        candidate_mean_pct=float(left.mean()),
        baseline_mean_pct=float(right.mean()),
        point_estimate_delta_pct=float(delta.mean()),
        one_sided_lower_delta_pct=float(np.quantile(means, 1.0 - confidence)),
        two_sided_lower_delta_pct=float(np.quantile(means, tail)),
        two_sided_upper_delta_pct=float(np.quantile(means, 1.0 - tail)),
        bootstrap_standard_error_delta_pct=float(means.std(ddof=1)),
    )


def paired_moving_block_bootstrap_suite(
    candidate: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    baseline: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    block_lengths: Sequence[int] = DEFAULT_BLOCK_LENGTHS,
    samples: int = 20_000,
    confidence: float = 0.90,
    random_state: int = 31,
    candidate_return_column: str = "net_return_pct",
    baseline_return_column: str = "net_return_pct",
    nan_policy: str = "raise",
) -> dict[int, PairedBootstrapInterval]:
    """Paired block-length sensitivity suite."""

    lengths = tuple(int(value) for value in block_lengths)
    _validate_bootstrap_parameters(lengths, samples, confidence)
    left, right = _aligned_pair(
        candidate,
        baseline,
        candidate_return_column=candidate_return_column,
        baseline_return_column=baseline_return_column,
        nan_policy=nan_policy,
    )
    return {
        length: paired_moving_block_bootstrap(
            left,
            right,
            block_length=length,
            samples=samples,
            confidence=confidence,
            random_state=random_state + offset,
        )
        for offset, length in enumerate(lengths)
    }


def _check(
    name: str,
    actual: float,
    operator: str,
    required: float,
) -> GateCheck:
    known = not math.isnan(actual)
    if operator == ">":
        passed = bool(known and actual > required)
    elif operator == ">=":
        passed = bool(known and actual >= required)
    elif operator == "<=":
        passed = bool(known and actual <= required)
    else:  # pragma: no cover - all callers are module constants
        raise RuntimeError(f"unsupported gate operator: {operator}")
    return GateCheck(name, passed, float(actual), operator, float(required))


def evaluate_gate(
    returns: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    gate: ValidationGate,
    *,
    baseline: pd.Series | pd.DataFrame | Sequence[float] | np.ndarray | None = None,
    return_column: str = "net_return_pct",
    baseline_return_column: str = "net_return_pct",
    nan_policy: str = "raise",
) -> GateDecision:
    """Evaluate every predeclared condition without short-circuiting.

    Returning all checks, including failed checks, keeps the validation ledger
    auditable and prevents a caller from reporting only favourable criteria.
    """

    series = _coerce_daily_returns(
        returns, return_column=return_column, nan_policy=nan_policy
    )
    diagnostics = return_diagnostics(
        series,
        top_k=gate.top_k,
        period_frequency=gate.period_frequency,
    )
    intervals = moving_block_bootstrap_suite(
        series,
        block_lengths=gate.block_lengths,
        samples=gate.bootstrap_samples,
        confidence=gate.confidence,
        random_state=gate.random_state,
    )
    checks: list[GateCheck] = [
        _check(
            "observations",
            float(diagnostics.observations),
            ">=",
            float(gate.min_observations),
        ),
        _check("mean_pct", diagnostics.mean_pct, ">", gate.min_mean_pct),
        _check(
            "top_k_removed_mean_pct",
            diagnostics.top_k_removed_mean_pct,
            ">",
            gate.min_top_k_removed_mean_pct,
        ),
        _check(
            "profit_factor",
            diagnostics.profit_factor,
            ">",
            gate.min_profit_factor,
        ),
        _check(
            "positive_period_ratio",
            diagnostics.positive_period_ratio,
            ">=",
            gate.min_positive_period_ratio,
        ),
        _check(
            "max_positive_contribution",
            diagnostics.max_positive_contribution,
            "<=",
            gate.max_positive_contribution,
        ),
    ]
    for length, interval in intervals.items():
        checks.append(
            _check(
                f"bootstrap_{length}_one_sided_lower_pct",
                interval.one_sided_lower_pct,
                ">",
                gate.min_one_sided_lower_pct,
            )
        )

    paired: dict[int, PairedBootstrapInterval] | None = None
    paired_required = (
        gate.min_paired_delta_pct is not None
        or gate.min_paired_one_sided_lower_delta_pct is not None
    )
    if paired_required and baseline is None:
        checks.append(
            GateCheck(
                "paired_baseline_supplied",
                False,
                float("nan"),
                "required",
                1.0,
            )
        )
    elif baseline is not None:
        paired = paired_moving_block_bootstrap_suite(
            series,
            baseline,
            block_lengths=gate.block_lengths,
            samples=gate.bootstrap_samples,
            confidence=gate.confidence,
            random_state=gate.random_state,
            baseline_return_column=baseline_return_column,
            nan_policy=nan_policy,
        )
        if gate.min_paired_delta_pct is not None:
            checks.append(
                _check(
                    "paired_mean_delta_pct",
                    next(iter(paired.values())).point_estimate_delta_pct,
                    ">",
                    gate.min_paired_delta_pct,
                )
            )
        if gate.min_paired_one_sided_lower_delta_pct is not None:
            for length, interval in paired.items():
                checks.append(
                    _check(
                        f"paired_bootstrap_{length}_one_sided_lower_delta_pct",
                        interval.one_sided_lower_delta_pct,
                        ">",
                        gate.min_paired_one_sided_lower_delta_pct,
                    )
                )

    failures = tuple(
        f"{item.name}: actual={item.actual} {item.operator} {item.required}"
        for item in checks
        if not item.passed
    )
    return GateDecision(
        gate=gate,
        passed=not failures,
        diagnostics=diagnostics,
        bootstrap=intervals,
        paired_bootstrap=paired,
        checks=tuple(checks),
        failure_reasons=failures,
    )
