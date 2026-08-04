#!/usr/bin/env python3
"""Run the preregistered v1.6 exact-liquidity selection.

The experiment uses directly observed JPX daily volume, turnover, VWAP, and
trading-unit fields.  Every candidate input is formed from D-1 or earlier.  A
score ledger is hashed before target outcomes are joined, and this runner can
never change the production model or enable orders.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research.finalize_logit_v04 import add_bounded_daily_features  # noqa: E402
from research.model_v13_symbolic_context_runner import G0_FEATURES  # noqa: E402
from tse_session_ranker.config import RankerConfig  # noqa: E402
from tse_session_ranker.data.common import (  # noqa: E402
    normalize_expected_sessions,
    prepare_modeling_prices,
    session_calendar_hash,
)
from tse_session_ranker.data.jpx import (  # noqa: E402
    PARSER_VERSION as JPX_PARSER_VERSION,
    collect_jpx,
)
from tse_session_ranker.features import (  # noqa: E402
    build_feature_panel,
    eligibility_mask,
)
from tse_session_ranker.profit import (  # noqa: E402
    daily_portfolio_returns,
    profit_metrics,
)
from tse_session_ranker.research_models import (  # noqa: E402
    ResearchModelSpec,
    fit_research_model,
)
from tse_session_ranker.validation import paired_moving_block_bootstrap  # noqa: E402


PROTOCOL = ROOT / "research/model_v16_liquidity_protocol.json"
PROTOCOL_SHA256 = (
    "f7b1329959b237323d6aa08d87494c9523bb5bdd4e29ee8c30cd04ece9e4387e"
)
PROTOCOL_ID = "model_v16_exact_liquidity_selection_20260728"
PARSER_AUDIT = ROOT / "research/model_v04_parser_recovery_audit.json"
PARSER_AUDIT_SHA256 = (
    "317d5cde9741429263cc91c11575300660f94f62123725fc8ce2d76cdee02eeb"
)
JPX_PARSER_SOURCE = ROOT / "src/tse_session_ranker/data/jpx.py"
PRICE_MANIFEST = ROOT / "research/model_v05_input_lock.json"
PRICE_MANIFEST_SHA256 = (
    "02370bda9c5fe73b166c557bcdc837d363f5deafbe91baa2d450b33dfcd45272"
)

PRICE_WARMUP_MONTHS = ("202505", "202506", "202507")
DAILY_START = pd.Timestamp("2025-08-01")
DAILY_END = pd.Timestamp("2026-03-31")
INITIAL_FIT_START = pd.Timestamp("2025-09-01")
SELECTION_START = pd.Timestamp("2025-12-01")
SELECTION_END = pd.Timestamp("2026-03-31")
SELECTION_SESSIONS = 80

CONTROL = "C00_PRICE_RIDGE"
CANDIDATES = (
    "LQ01_ACTIVITY_VETO",
    "LQ02_EXACT_LIQUIDITY_RIDGE",
    "LQ03_VWAP_FLOW_REVERSAL",
)
ALL_MODELS = (CONTROL, *CANDIDATES)
CAPACITIES = (1, 2)
COSTS_BPS = (20.0, 40.0, 60.0)
PRIMARY_COST_BPS = 40.0
FAMILY_SIZE = len(CANDIDATES) * len(CAPACITIES)
BOOTSTRAP_BLOCK_LENGTH = 5
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_RANDOM_STATE = 20_260_728
BONFERRONI_CONFIDENCE = 1.0 - 0.10 / FAMILY_SIZE

SELECTION_SLICES = {
    "selection_a": (
        pd.Timestamp("2025-12-01"),
        pd.Timestamp("2026-01-30"),
    ),
    "selection_b": (
        pd.Timestamp("2026-02-02"),
        pd.Timestamp("2026-03-31"),
    ),
}

LIQUIDITY_RAW_FEATURES = (
    "liq_turnover_med20",
    "liq_volume_med20",
    "liq_turnover_shock1",
    "liq_turnover_log_iqr20",
    "liq_lot_fraction20",
    "liq_close_vwap_dev1",
    "liq_close_vwap_abs_med20",
)
LIQUIDITY_FEATURES = tuple(
    f"xrank_{feature}" for feature in LIQUIDITY_RAW_FEATURES
)
PRICE_RANK_SOURCES = (
    "atr14_pct",
    "close_momentum_5",
    "close_momentum_20",
    "close_momentum_60",
    "prior_close_location_20",
)
SCORE_LEDGER_COLUMNS = (
    "candidate_id",
    "date",
    "model_rank",
    "code",
    "name",
    "model_score",
    "pre_veto_code",
    "vetoed",
)

VETO_TURNOVER_MED20_MIN = 100_000_000.0
VETO_VOLUME_MED20_MIN = 50_000.0
VETO_LOT_FRACTION20_MAX = 0.01


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    return value


def write_json(value: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_protocol() -> dict[str, Any]:
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("v1.6 protocol SHA-256 mismatch")
    protocol = read_json(PROTOCOL)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected v1.6 protocol id")
    authority = protocol["authority"]
    if authority["project_level_untouched"] is not False:
        raise ValueError("v1.6 must not claim a project-level untouched period")
    if authority["production_promotion_allowed"] is not False:
        raise ValueError("v1.6 cannot promote production")
    if authority["orders_allowed"] is not False:
        raise ValueError("v1.6 must keep orders disabled")
    if tuple(protocol["common_price_features"]) != tuple(G0_FEATURES):
        raise ValueError("v1.6 common price features changed")
    registered = tuple(item["id"] for item in protocol["candidates"])
    if registered != CANDIDATES:
        raise ValueError("v1.6 candidate order changed")
    evaluation = protocol["evaluation"]
    if tuple(evaluation["capacities"]) != CAPACITIES:
        raise ValueError("v1.6 capacities changed")
    if int(evaluation["candidate_variants"]) != FAMILY_SIZE:
        raise ValueError("v1.6 family size changed")
    if tuple(float(item) for item in evaluation["costs_bps"]) != COSTS_BPS:
        raise ValueError("v1.6 cost grid changed")
    bootstrap = evaluation["bootstrap"]
    if int(bootstrap["samples"]) != BOOTSTRAP_SAMPLES:
        raise ValueError("v1.6 bootstrap samples changed")
    if not math.isclose(
        float(bootstrap["bonferroni_individual_confidence"]),
        BONFERRONI_CONFIDENCE,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("v1.6 familywise confidence changed")
    return protocol


def _expected_price_sources() -> dict[str, dict[str, Any]]:
    if sha256_file(PRICE_MANIFEST) != PRICE_MANIFEST_SHA256:
        raise ValueError("v1.6 price manifest SHA-256 mismatch")
    manifest = read_json(PRICE_MANIFEST)
    sources = {
        str(item["filename"]): item for item in manifest["jpx"]["sources"]
    }
    names = {f"{month}.pdf" for month in PRICE_WARMUP_MONTHS}
    if not names.issubset(sources):
        raise ValueError("v1.6 price warm-up sources are missing from the manifest")
    return {name: sources[name] for name in sorted(names)}


def _pdftotext_runtime() -> dict[str, str]:
    executable = shutil.which("pdftotext")
    if executable is None:
        raise ValueError("v1.6 requires pdftotext")
    completed = subprocess.run(
        [executable, "-v"],
        check=True,
        capture_output=True,
        text=True,
    )
    version = (completed.stderr or completed.stdout).splitlines()[0].strip()
    return {
        "version": version,
        "executable_sha256": sha256_file(executable),
    }


def _expected_daily_sources() -> dict[str, dict[str, Any]]:
    if sha256_file(PARSER_AUDIT) != PARSER_AUDIT_SHA256:
        raise ValueError("v1.6 parser-audit SHA-256 mismatch")
    audit = read_json(PARSER_AUDIT)
    if audit.get("new_parser_version") != JPX_PARSER_VERSION:
        raise ValueError("v1.6 parser version differs from the recovery audit")
    if sha256_file(JPX_PARSER_SOURCE) != str(audit.get("new_parser_sha256")):
        raise ValueError("v1.6 parser source differs from the recovery audit")
    if int(audit.get("pdf_count", -1)) != 160:
        raise ValueError("v1.6 daily source count changed")
    if int(audit.get("new_rejected_rows", -1)) != 0:
        raise ValueError("v1.6 recovery audit contains rejected rows")
    sources = {str(item["name"]): item for item in audit["per_file"]}
    if len(sources) != 160:
        raise ValueError("v1.6 recovery-audit file names are not unique")
    return sources


def _validate_directory(
    directory: str | Path,
    expected: dict[str, dict[str, Any]],
) -> tuple[list[Path], list[dict[str, Any]]]:
    root = Path(directory)
    actual = sorted(root.glob("*.pdf"))
    actual_names = [path.name for path in actual]
    expected_names = sorted(expected)
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(expected_names))
        raise ValueError(
            f"v1.6 source set mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    checks: list[dict[str, Any]] = []
    for path in actual:
        item = expected[path.name]
        observed = {
            "name": path.name,
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        if observed["sha256"] != str(item["sha256"]):
            raise ValueError(f"v1.6 source SHA-256 mismatch: {path.name}")
        if "bytes" in item and observed["bytes"] != int(item["bytes"]):
            raise ValueError(f"v1.6 source byte-size mismatch: {path.name}")
        checks.append(observed)
    return actual, checks


def load_selection_prices(
    *,
    price_warmup_directory: str | Path,
    daily_directory: str | Path,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, Any]]:
    warmup, warmup_checks = _validate_directory(
        price_warmup_directory, _expected_price_sources()
    )
    daily, daily_checks = _validate_directory(
        daily_directory, _expected_daily_sources()
    )
    prices, parse = collect_jpx([*warmup, *daily])
    reports = parse["inputs"]
    daily_reports = [
        item for item in reports if str(Path(str(item["path"])).name).startswith("stq_")
    ]
    if len(daily_reports) != 160:
        raise ValueError("v1.6 did not parse exactly 160 daily PDFs")
    if any(int(item["rejected_rows"]) != 0 for item in daily_reports):
        raise ValueError("v1.6 daily parser rejected an ordinary-stock row")
    if any(
        item["source_format"]
        != "jpx_stock_quotations_auction_regular_way_domestic_ordinary"
        for item in daily_reports
    ):
        raise ValueError("v1.6 daily source format changed")
    if any(item["volume_semantics"] is None for item in daily_reports):
        raise ValueError("v1.6 daily volume semantics are missing")
    if any(item["turnover_semantics"] is None for item in daily_reports):
        raise ValueError("v1.6 daily turnover semantics are missing")
    if any(item["trading_unit_semantics"] is None for item in daily_reports):
        raise ValueError("v1.6 daily trading-unit semantics are missing")
    daily_rows = sum(int(item["parsed_rows"]) for item in daily_reports)
    if daily_rows != 622_724:
        raise ValueError(
            f"v1.6 daily parsed-row count changed: expected 622724, got {daily_rows}"
        )
    dates = pd.to_datetime(prices["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("v1.6 prices contain an invalid date")
    sessions = normalize_expected_sessions(dates.drop_duplicates())
    daily_sessions = sessions[
        (sessions >= DAILY_START) & (sessions <= DAILY_END)
    ]
    if len(daily_sessions) != 160:
        raise ValueError("v1.6 daily session count changed")
    if daily_sessions.min() != DAILY_START or daily_sessions.max() != DAILY_END:
        raise ValueError("v1.6 daily date bounds changed")
    return prices, sessions, {
        "parser_version": JPX_PARSER_VERSION,
        "price_warmup_sources": warmup_checks,
        "daily_source_count": len(daily_checks),
        "daily_source_set_sha256": hashlib.sha256(
            "\n".join(item["sha256"] for item in daily_checks).encode("ascii")
        ).hexdigest(),
        "daily_parsed_rows": daily_rows,
        "daily_rejected_rows": 0,
        "daily_sessions": len(daily_sessions),
        "daily_date_bounds": [
            str(daily_sessions.min().date()),
            str(daily_sessions.max().date()),
        ],
        "calendar_sha256": session_calendar_hash(sessions),
        "actual_volume": bool(prices["volume"].notna().any()),
        "actual_turnover": bool(prices["turnover"].notna().any()),
        "actual_vwap": bool(prices.get("vwap", pd.Series(dtype=float)).notna().any()),
        "actual_trading_unit": bool(
            prices.get("trading_unit", pd.Series(dtype=float)).notna().any()
        ),
        "pdftotext": _pdftotext_runtime(),
    }


def _rolling(
    values: pd.Series,
    codes: pd.Series,
    *,
    window: int,
    operation: str,
) -> pd.Series:
    grouped = pd.to_numeric(values, errors="coerce").groupby(codes, sort=False)
    rolling = grouped.rolling(window, min_periods=window)
    if operation == "median":
        result = rolling.median()
    elif operation == "sum":
        result = rolling.sum()
    elif operation == "min":
        result = rolling.min()
    elif operation == "max":
        result = rolling.max()
    elif operation == "q25":
        result = rolling.quantile(0.25)
    elif operation == "q75":
        result = rolling.quantile(0.75)
    else:
        raise ValueError(f"unsupported rolling operation: {operation}")
    return result.reset_index(level=0, drop=True).reindex(values.index)


def _rank_to_unit(
    values: pd.Series,
    dates: pd.Series,
    universe: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    source = numeric.where(universe & np.isfinite(numeric))
    ranks = source.groupby(dates, sort=False).rank(method="average")
    counts = source.groupby(dates, sort=False).transform("count")
    result = 2.0 * (ranks - 1.0) / (counts - 1.0) - 1.0
    result = result.where(counts.gt(1), 0.0)
    return result.where(source.notna())


def build_exact_liquidity_features(
    prices: pd.DataFrame,
    sessions: Iterable[object],
) -> pd.DataFrame:
    """Build D-1 exact-liquidity features without target-date fields."""

    required = {
        "date",
        "code",
        "name",
        "close",
        "volume",
        "turnover",
        "vwap",
        "trading_unit",
        "source_format",
    }
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"v1.6 exact-liquidity source lacks columns: {missing}")
    calendar = normalize_expected_sessions(sessions)
    daily = prices.loc[
        prices["source_format"].eq(
            "jpx_stock_quotations_auction_regular_way_domestic_ordinary"
        ),
        sorted(required),
    ].copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["code"] = daily["code"].astype(str)
    daily = daily.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    if daily[["date", "code"]].duplicated().any():
        raise ValueError("v1.6 exact-liquidity source has duplicate date/code rows")

    position_by_date = pd.Series(np.arange(len(calendar)), index=calendar)
    daily["_session_position"] = daily["date"].map(position_by_date)
    if daily["_session_position"].isna().any():
        raise ValueError("v1.6 daily liquidity row is outside the session calendar")
    codes = daily["code"]
    numeric_columns = ("close", "volume", "turnover", "vwap", "trading_unit")
    for column in numeric_columns:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    positive = daily[list(numeric_columns)].gt(0.0).all(axis=1)
    daily["_positive_activity_row"] = positive.astype(float)
    oldest = daily["_session_position"].groupby(codes, sort=False).shift(19)
    consecutive20 = (daily["_session_position"] - oldest).eq(19)
    positive20 = _rolling(
        daily["_positive_activity_row"], codes, window=20, operation="sum"
    ).eq(20.0)
    unit_min20 = _rolling(
        daily["trading_unit"], codes, window=20, operation="min"
    )
    unit_max20 = _rolling(
        daily["trading_unit"], codes, window=20, operation="max"
    )
    unit_stable20 = unit_min20.eq(unit_max20) & unit_min20.gt(0.0)
    daily["liq_history_complete20"] = (
        consecutive20 & positive20 & unit_stable20
    )

    daily["liq_turnover_med20"] = _rolling(
        daily["turnover"], codes, window=20, operation="median"
    )
    daily["liq_volume_med20"] = _rolling(
        daily["volume"], codes, window=20, operation="median"
    )
    previous_turnover = daily["turnover"].groupby(codes, sort=False).shift(1)
    previous_turnover_med19 = _rolling(
        previous_turnover, codes, window=19, operation="median"
    )
    ratio = daily["turnover"] / previous_turnover_med19.where(
        previous_turnover_med19.gt(0.0)
    )
    daily["liq_turnover_shock1"] = np.log(ratio).clip(-3.0, 3.0)
    log_turnover = np.log1p(daily["turnover"])
    daily["liq_turnover_log_iqr20"] = _rolling(
        log_turnover, codes, window=20, operation="q75"
    ) - _rolling(log_turnover, codes, window=20, operation="q25")
    daily["liq_lot_fraction20"] = (
        daily["trading_unit"] * daily["close"]
    ) / daily["liq_turnover_med20"].where(
        daily["liq_turnover_med20"].gt(0.0)
    )
    close_vwap = daily["close"] / daily["vwap"].where(daily["vwap"].gt(0.0)) - 1.0
    daily["liq_close_vwap_dev1"] = close_vwap.clip(-0.20, 0.20)
    daily["liq_close_vwap_abs_med20"] = _rolling(
        close_vwap.abs(), codes, window=20, operation="median"
    )

    next_session = dict(zip(calendar[:-1], calendar[1:], strict=True))
    daily["feature_source_max_date"] = daily["date"]
    daily["date"] = daily["date"].map(next_session)
    daily = daily.dropna(subset=["date"]).copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    finite_features = daily[list(LIQUIDITY_RAW_FEATURES)].notna().all(axis=1)
    daily["liq_history_complete20"] &= finite_features
    rank_universe = daily["liq_history_complete20"].fillna(False).astype(bool)
    for feature in LIQUIDITY_RAW_FEATURES:
        daily[f"xrank_{feature}"] = _rank_to_unit(
            daily[feature], daily["date"], rank_universe
        )
    daily["activity_veto_pass"] = (
        rank_universe
        & daily["liq_turnover_med20"].ge(VETO_TURNOVER_MED20_MIN)
        & daily["liq_volume_med20"].ge(VETO_VOLUME_MED20_MIN)
        & daily["liq_lot_fraction20"].le(VETO_LOT_FRACTION20_MAX)
    )
    projection = [
        "date",
        "code",
        "feature_source_max_date",
        "liq_history_complete20",
        "activity_veto_pass",
        *LIQUIDITY_RAW_FEATURES,
        *LIQUIDITY_FEATURES,
    ]
    result = daily.loc[:, projection].copy()
    if result[["date", "code"]].duplicated().any():
        raise ValueError("v1.6 exact-liquidity target rows are duplicated")
    if (
        result["feature_source_max_date"].notna()
        & result["feature_source_max_date"].ge(result["date"])
    ).any():
        raise ValueError("v1.6 exact-liquidity source is not strictly prior")
    return result.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def build_model_panel(
    prices: pd.DataFrame,
    sessions: Iterable[object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    settings = RankerConfig()
    calendar = normalize_expected_sessions(sessions)
    modeling, coverage = prepare_modeling_prices(
        prices,
        coverage_lookback=settings.source_coverage_lookback,
        minimum_source_coverage=settings.minimum_source_coverage,
        expected_sessions=calendar,
    )
    panel = add_bounded_daily_features(build_feature_panel(modeling, settings))
    prior_rank_universe = (
        eligibility_mask(panel, settings.universe, for_training=True)
        & panel["price_history_continuous_60"].fillna(False).astype(bool)
        & panel["prior_universe_member"].fillna(False).astype(bool)
        & panel["universe_source_complete"].fillna(False).astype(bool)
    )
    # add_bounded_daily_features retains the historical v0.x implementation,
    # whose xrank universe included target-date source finality.  Overwrite all
    # G0 rank fields with a universe that is knowable from D-1 alone.
    for source in PRICE_RANK_SOURCES:
        panel[f"xrank_{source}"] = _rank_to_unit(
            panel[source],
            panel["date"],
            prior_rank_universe,
        ).astype("float32")
    exact = build_exact_liquidity_features(prices, calendar)
    panel = panel.merge(
        exact,
        on=["date", "code"],
        how="left",
        validate="one_to_one",
        sort=False,
        suffixes=("", "_liquidity"),
    )
    price_score = (
        eligibility_mask(panel, settings.universe)
        & panel["price_history_continuous_60"].fillna(False).astype(bool)
        & panel["prior_universe_member"].fillna(False).astype(bool)
        & panel["universe_source_complete"].fillna(False).astype(bool)
    )
    price_training = (
        eligibility_mask(panel, settings.universe, for_training=True)
        & panel["price_history_continuous_60"].fillna(False).astype(bool)
        & panel["evaluation_ready"].fillna(False).astype(bool)
        & panel["outcome_observed"].fillna(False).astype(bool)
        & panel["oc_return_pct"].notna()
    )
    liquidity_ready = panel["liq_history_complete20"].eq(True)
    panel["common_score_eligible"] = price_score & liquidity_ready
    panel["common_training_eligible"] = price_training & liquidity_ready
    panel["activity_veto_pass"] = panel["activity_veto_pass"].eq(True)
    if panel.loc[
        panel["common_score_eligible"], list(LIQUIDITY_FEATURES)
    ].isna().any(axis=None):
        raise ValueError("v1.6 eligible liquidity ranks contain missing values")
    source_dates = panel.loc[
        panel["common_score_eligible"],
        ["date", "feature_source_max_date_liquidity"],
    ]
    if (
        source_dates["feature_source_max_date_liquidity"]
        .ge(source_dates["date"])
        .any()
    ):
        raise ValueError("v1.6 eligible liquidity feature is not strictly lagged")
    checks = {
        "panel_rows": int(len(panel)),
        "panel_codes": int(panel["code"].nunique()),
        "source_incomplete_sessions": int((~coverage["source_complete"]).sum()),
        "common_training_rows": int(panel["common_training_eligible"].sum()),
        "common_scoring_rows": int(panel["common_score_eligible"].sum()),
        "liquidity_ready_rows": int(liquidity_ready.sum()),
        "same_day_finality_used_for_scoring": False,
        "maximum_liquidity_source_before_target": True,
    }
    return panel, coverage, checks


def _top_two(
    scoring: pd.DataFrame,
    values: np.ndarray | pd.Series,
    candidate_id: str,
    *,
    apply_veto: bool,
) -> pd.DataFrame:
    ranked = scoring.loc[
        :,
        ["date", "code", "name", "activity_veto_pass"],
    ].copy()
    ranked["model_score"] = np.asarray(values, dtype=float)
    if len(ranked) != len(scoring) or not np.isfinite(ranked["model_score"]).all():
        raise ValueError("v1.6 scorer produced invalid scores")
    ranked = ranked.sort_values(
        ["date", "model_score", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranked = ranked.groupby("date", sort=False, as_index=False).head(2).copy()
    ranked["model_rank"] = ranked.groupby("date", sort=False).cumcount() + 1
    ranked["candidate_id"] = candidate_id
    ranked["pre_veto_code"] = ranked["code"]
    ranked["vetoed"] = False
    if apply_veto:
        veto = ~ranked["activity_veto_pass"].fillna(False).astype(bool)
        ranked.loc[veto, "vetoed"] = True
        ranked.loc[veto, ["code", "name", "model_score"]] = np.nan
    return ranked.loc[:, list(SCORE_LEDGER_COLUMNS)]


def _complete_slots(
    picks: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    candidate_id: str,
) -> pd.DataFrame:
    desired = pd.MultiIndex.from_product(
        [scheduled, (1, 2)], names=["date", "model_rank"]
    ).to_frame(index=False)
    output = desired.merge(
        picks,
        on=["date", "model_rank"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    output["candidate_id"] = candidate_id
    output["vetoed"] = output["vetoed"].fillna(False).astype(bool)
    output = output.loc[:, list(SCORE_LEDGER_COLUMNS)]
    if len(output) != len(scheduled) * 2:
        raise AssertionError("v1.6 slot completion changed the schedule")
    return output


def _model_spec(candidate_id: str) -> tuple[ResearchModelSpec, Sequence[str]]:
    if candidate_id == CONTROL:
        return (
            ResearchModelSpec(
                name="model_v16_control_price_ridge",
                family="ridge_daily_rank",
                objective="same_day_return_percentile",
                parameters={"alpha": 1.0},
            ),
            G0_FEATURES,
        )
    if candidate_id == "LQ02_EXACT_LIQUIDITY_RIDGE":
        return (
            ResearchModelSpec(
                name="model_v16_exact_liquidity_ridge",
                family="ridge_daily_rank",
                objective="same_day_return_percentile",
                parameters={"alpha": 10.0},
            ),
            (*G0_FEATURES, *LIQUIDITY_FEATURES),
        )
    raise ValueError(f"v1.6 has no fitted model spec for {candidate_id}")


def _monthly_scores(
    panel: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    candidate_id: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    specification, features = _model_spec(candidate_id)
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    periods = pd.period_range(
        scheduled.min().to_period("M"), scheduled.max().to_period("M")
    )
    for period in periods:
        first = period.start_time.normalize()
        last = period.end_time.normalize()
        training = panel.loc[
            panel["date"].between(
                INITIAL_FIT_START, first - pd.Timedelta(days=1)
            )
            & panel["common_training_eligible"],
            ["date", "code", "oc_return_pct", *features],
        ].copy()
        scoring = panel.loc[
            panel["date"].between(first, last)
            & panel["date"].isin(scheduled)
            & panel["common_score_eligible"],
            ["date", "code", "name", "activity_veto_pass", *features],
        ].copy()
        if training.empty:
            raise ValueError(f"v1.6 {candidate_id} fold {period} has no training rows")
        if training["date"].max() >= first:
            raise ValueError(f"v1.6 {candidate_id} fold {period} leaked a label")
        if candidate_id == "LQ02_EXACT_LIQUIDITY_RIDGE" and training.loc[
            :, list(LIQUIDITY_FEATURES)
        ].isna().any(axis=None):
            raise ValueError(
                f"v1.6 {candidate_id} exact-liquidity features are missing"
            )
        scorer = fit_research_model(specification, training, features)
        if not scoring.empty:
            parts.append(
                _top_two(
                    scoring,
                    scorer.score(scoring),
                    candidate_id,
                    apply_veto=False,
                )
            )
        folds.append(
            {
                "candidate_id": candidate_id,
                "period": str(period),
                "train_start": str(training["date"].min().date()),
                "train_end": str(training["date"].max().date()),
                "train_dates": int(training["date"].nunique()),
                "train_rows": int(len(training)),
                "score_rows": int(len(scoring)),
                "strictly_prior_training": bool(training["date"].max() < first),
                "mid_month_refit": False,
                "spec_id": specification.spec_id,
            }
        )
    actual = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=SCORE_LEDGER_COLUMNS
    )
    return _complete_slots(actual, scheduled, candidate_id), folds


def build_score_ledger(
    panel: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    control, control_folds = _monthly_scores(panel, scheduled, CONTROL)
    liquidity, liquidity_folds = _monthly_scores(
        panel, scheduled, "LQ02_EXACT_LIQUIDITY_RIDGE"
    )
    activity_veto = control.copy()
    activity_veto["candidate_id"] = "LQ01_ACTIVITY_VETO"
    activity_veto["pre_veto_code"] = activity_veto["code"]
    veto_lookup = panel.set_index(["date", "code"])[
        "activity_veto_pass"
    ].to_dict()
    veto = [
        (
            False
            if pd.isna(code)
            else not bool(veto_lookup.get((pd.Timestamp(date), str(code)), False))
        )
        for date, code in zip(
            activity_veto["date"], activity_veto["code"], strict=True
        )
    ]
    activity_veto["vetoed"] = veto
    activity_veto.loc[veto, ["code", "name", "model_score"]] = np.nan

    deterministic_scoring = panel.loc[
        panel["date"].isin(scheduled) & panel["common_score_eligible"],
        [
            "date",
            "code",
            "name",
            "activity_veto_pass",
            "xrank_liq_close_vwap_dev1",
            "xrank_liq_turnover_shock1",
        ],
    ].copy()
    deterministic_values = (
        -deterministic_scoring["xrank_liq_close_vwap_dev1"]
        * (1.0 + deterministic_scoring["xrank_liq_turnover_shock1"])
        / 2.0
    )
    reversal = _complete_slots(
        _top_two(
            deterministic_scoring,
            deterministic_values,
            "LQ03_VWAP_FLOW_REVERSAL",
            apply_veto=True,
        ),
        scheduled,
        "LQ03_VWAP_FLOW_REVERSAL",
    )
    ledger = pd.concat(
        [control, activity_veto, liquidity, reversal],
        ignore_index=True,
    )
    ledger = ledger.sort_values(
        ["candidate_id", "date", "model_rank"], kind="stable"
    ).reset_index(drop=True)
    if set(ledger.columns) != set(SCORE_LEDGER_COLUMNS):
        raise AssertionError("v1.6 score ledger schema changed")
    forbidden = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "vwap",
        "trading_unit",
        "label",
        "oc_return_pct",
        "overnight",
    }
    if forbidden & set(ledger.columns):
        raise AssertionError("v1.6 score ledger contains an outcome field")
    expected_rows = len(ALL_MODELS) * len(scheduled) * 2
    if len(ledger) != expected_rows:
        raise AssertionError("v1.6 score ledger is incomplete")
    return ledger, [*control_folds, *liquidity_folds]


def semantic_score_hash(ledger: pd.DataFrame) -> str:
    canonical = ledger.loc[:, list(SCORE_LEDGER_COLUMNS)].copy()
    canonical["date"] = pd.to_datetime(canonical["date"]).dt.strftime("%Y-%m-%d")
    canonical = canonical.sort_values(
        ["candidate_id", "date", "model_rank"], kind="stable"
    ).reset_index(drop=True)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def attach_outcomes(
    score_ledger: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    if {"label", "oc_return_pct"} & set(score_ledger.columns):
        raise ValueError("v1.6 score ledger already contains outcomes")
    outcomes = panel.loc[
        :,
        ["date", "code", "label", "oc_return_pct"],
    ].copy()
    outcomes = outcomes.dropna(subset=["code"])
    if outcomes[["date", "code"]].duplicated().any():
        raise ValueError("v1.6 outcome ledger contains duplicate date/code rows")
    picks = score_ledger.merge(
        outcomes,
        on=["date", "code"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    return picks.sort_values(
        ["candidate_id", "date", "model_rank"], kind="stable"
    ).reset_index(drop=True)


def _daily(
    picks: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    capacity: int,
    cost_bps: float,
) -> pd.Series:
    daily = daily_portfolio_returns(
        picks, top_k=capacity, cost_bps=cost_bps
    )
    values = daily.set_index("date")["net_return_pct"].reindex(scheduled)
    if len(values) != len(scheduled) or values.isna().any():
        raise AssertionError("v1.6 scheduled-day return series is incomplete")
    return values.astype(float)


def _top_codes_cash(
    picks: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    capacity: int,
    cost_bps: float,
    count: int,
) -> tuple[pd.Series, list[str]]:
    selected = picks[picks["model_rank"].le(capacity)].copy()
    executed = selected["label"].notna()
    selected["_net_slot"] = selected["oc_return_pct"].fillna(0.0) - (
        executed.astype(float) * cost_bps / 100.0
    )
    by_code = (
        selected.dropna(subset=["code"])
        .groupby("code", sort=False)["_net_slot"]
        .sum()
        .sort_values(ascending=False, kind="stable")
    )
    codes = [str(code) for code in by_code.head(count).index]
    neutral = picks.copy()
    mask = neutral["model_rank"].le(capacity) & neutral["code"].astype(
        str
    ).isin(codes)
    neutral.loc[
        mask,
        ["code", "name", "model_score", "label", "oc_return_pct"],
    ] = np.nan
    return (
        _daily(
            neutral,
            scheduled,
            capacity=capacity,
            cost_bps=cost_bps,
        ),
        codes,
    )


def _variant_metrics(
    picks: pd.DataFrame,
    control: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    candidate_id: str,
    capacity: int,
    random_state: int,
) -> dict[str, Any]:
    costs = {
        str(int(cost)): profit_metrics(
            picks, top_k=capacity, cost_bps=cost
        )
        for cost in COSTS_BPS
    }
    daily40 = _daily(
        picks, scheduled, capacity=capacity, cost_bps=PRIMARY_COST_BPS
    )
    control40 = _daily(
        control, scheduled, capacity=capacity, cost_bps=PRIMARY_COST_BPS
    )
    daily60 = _daily(
        picks, scheduled, capacity=capacity, cost_bps=60.0
    )
    paired = paired_moving_block_bootstrap(
        daily40,
        control40,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        samples=BOOTSTRAP_SAMPLES,
        confidence=BONFERRONI_CONFIDENCE,
        random_state=random_state,
    )
    slices = {
        name: float(daily40.loc[start:end].mean())
        for name, (start, end) in SELECTION_SLICES.items()
    }
    top4_removed = float(daily40.drop(daily40.nlargest(4).index).mean())
    code_cash, top_codes = _top_codes_cash(
        picks,
        scheduled,
        capacity=capacity,
        cost_bps=PRIMARY_COST_BPS,
        count=5,
    )
    monthly = daily40.groupby(daily40.index.to_period("M")).mean()
    selected = picks[picks["model_rank"].le(capacity)].copy()
    signaled = selected.dropna(subset=["code"])
    counts = signaled["code"].astype(str).value_counts()
    total_signals = int(len(signaled))
    unique_codes = int(len(counts))
    maximum_share = (
        float(counts.iloc[0] / total_signals) if total_signals else 1.0
    )
    top10_share = (
        float(counts.head(10).sum() / total_signals) if total_signals else 1.0
    )
    executed = selected["label"].notna()
    executed_days = int(
        selected.assign(_executed=executed.astype(int))
        .groupby("date", sort=True)["_executed"]
        .sum()
        .gt(0)
        .sum()
    )
    executed_fraction = float(executed.sum() / (len(scheduled) * capacity))
    checks = {
        "net40_mean_positive": float(daily40.mean()) > 0.0,
        "net40_median_positive": float(daily40.median()) > 0.0,
        "net60_mean_positive": float(daily60.mean()) > 0.0,
        "both_fixed_slices_net40_positive": all(
            value > 0.0 for value in slices.values()
        ),
        "positive_months_net40_at_least_3": int(monthly.gt(0.0).sum()) >= 3,
        "top4_days_removed_net40_positive": top4_removed > 0.0,
        "top5_profit_codes_cash_net40_positive": float(code_cash.mean()) > 0.0,
        "familywise_paired_lower_vs_control_nonnegative": (
            paired.one_sided_lower_delta_pct >= 0.0
        ),
        "unique_codes_at_least_40": unique_codes >= 40,
        "maximum_code_share_at_most_0_05": maximum_share <= 0.05,
        "top10_code_share_at_most_0_25": top10_share <= 0.25,
        "executed_days_at_least_72": executed_days >= 72,
        "executed_slot_fraction_at_least_0_8": executed_fraction >= 0.80,
    }
    return {
        "variant_id": f"{candidate_id}__top{capacity}",
        "candidate_id": candidate_id,
        "capacity": capacity,
        "cost_metrics": costs,
        "net40_mean_pct": float(daily40.mean()),
        "net40_median_pct": float(daily40.median()),
        "net60_mean_pct": float(daily60.mean()),
        "slice_net40_mean_pct": slices,
        "monthly_net40_mean_pct": {
            str(period): float(value) for period, value in monthly.items()
        },
        "positive_months_net40": int(monthly.gt(0.0).sum()),
        "top4_days_removed_net40_mean_pct": top4_removed,
        "top5_profit_codes_cash_net40_mean_pct": float(code_cash.mean()),
        "top5_profit_codes": top_codes,
        "paired_vs_control_net40": asdict(paired),
        "bonferroni_individual_confidence": BONFERRONI_CONFIDENCE,
        "unique_selected_codes": unique_codes,
        "maximum_code_selection_share": maximum_share,
        "top10_code_selection_share": top10_share,
        "executed_days": executed_days,
        "executed_slot_fraction": executed_fraction,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def _choose_winner(variants: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    passers = [item for item in variants if item["gate_passed"]]
    if not passers:
        return None
    ordered = sorted(
        passers,
        key=lambda item: (
            -float(
                item["paired_vs_control_net40"][
                    "one_sided_lower_delta_pct"
                ]
            ),
            -float(item["top4_days_removed_net40_mean_pct"]),
            str(item["variant_id"]),
        ),
    )
    winner = ordered[0]
    return {
        "variant_id": winner["variant_id"],
        "candidate_id": winner["candidate_id"],
        "capacity": winner["capacity"],
        "selection_gate_passed": True,
        "selection_rule_rank": 1,
    }


def run_selection(
    *,
    protocol: dict[str, Any],
    price_warmup_directory: str | Path,
    daily_directory: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    started = time.perf_counter()
    prices, sessions, source_checks = load_selection_prices(
        price_warmup_directory=price_warmup_directory,
        daily_directory=daily_directory,
    )
    panel, coverage, panel_checks = build_model_panel(prices, sessions)
    scheduled = sessions[
        (sessions >= SELECTION_START) & (sessions <= SELECTION_END)
    ]
    if len(scheduled) != SELECTION_SESSIONS:
        raise ValueError(
            f"v1.6 selection schedule changed: expected 80, got {len(scheduled)}"
        )
    score_ledger, folds = build_score_ledger(panel, scheduled)
    score_hash = semantic_score_hash(score_ledger)
    picks = attach_outcomes(score_ledger, panel)
    picks_by_model = {
        model: picks[picks["candidate_id"].eq(model)].copy()
        for model in ALL_MODELS
    }
    control_metrics = {
        f"top{capacity}": {
            str(int(cost)): profit_metrics(
                picks_by_model[CONTROL],
                top_k=capacity,
                cost_bps=cost,
            )
            for cost in COSTS_BPS
        }
        for capacity in CAPACITIES
    }
    variants: list[dict[str, Any]] = []
    for candidate_index, candidate_id in enumerate(CANDIDATES):
        for capacity_index, capacity in enumerate(CAPACITIES):
            variants.append(
                _variant_metrics(
                    picks_by_model[candidate_id],
                    picks_by_model[CONTROL],
                    scheduled,
                    candidate_id=candidate_id,
                    capacity=capacity,
                    random_state=(
                        BOOTSTRAP_RANDOM_STATE
                        + candidate_index * len(CAPACITIES)
                        + capacity_index
                    ),
                )
            )
    winner = _choose_winner(variants)
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "status": (
            "selection_passed_one_locked_replay_nominee"
            if winner is not None
            else "selection_rejected_all_candidates"
        ),
        "authority": {
            "analysis_type": "retrospective_candidate_specific_selection",
            "project_level_untouched": False,
            "production_model_changed": False,
            "production_promotion_allowed": False,
            "orders_allowed": False,
            "locked_replay_input_opened": False,
        },
        "input": {
            **source_checks,
            **panel_checks,
            "selection_source_complete_sessions": int(
                coverage.loc[
                    coverage["date"].isin(scheduled), "source_complete"
                ].sum()
            ),
        },
        "selection": {
            "date_bounds": [
                str(scheduled.min().date()),
                str(scheduled.max().date()),
            ],
            "scheduled_sessions": len(scheduled),
            "models": list(ALL_MODELS),
            "candidate_variants": FAMILY_SIZE,
            "score_ledger_outcome_columns": [],
            "score_ledger_semantic_sha256": score_hash,
            "score_ledger_hashed_before_outcome_join": True,
            "folds": folds,
            "control_metrics": control_metrics,
            "variants": variants,
            "gate_passers": sum(item["gate_passed"] for item in variants),
            "winner": winner,
            "locked_replay_allowed": winner is not None,
            "locked_replay_requires_separate_remote_winner_lock": True,
        },
        "decision": {
            "candidate_family_rejected": winner is None,
            "forward_shadow_nominee": (
                None if winner is None else winner["variant_id"]
            ),
            "production_model_changed": False,
            "orders_allowed": False,
        },
        "runtime": {
            "seconds": float(time.perf_counter() - started),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    return result, picks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-warmup-directory", type=Path, required=True)
    parser.add_argument("--daily-directory", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research/model_v16_liquidity_result.json",
    )
    parser.add_argument(
        "--picks-output",
        type=Path,
        default=ROOT / "research/model_v16_liquidity_picks.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = _validate_protocol()
    result, picks = run_selection(
        protocol=protocol,
        price_warmup_directory=args.price_warmup_directory,
        daily_directory=args.daily_directory,
    )
    args.picks_output.parent.mkdir(parents=True, exist_ok=True)
    picks.to_csv(args.picks_output, index=False, lineterminator="\n")
    result["picks_sha256"] = sha256_file(args.picks_output)
    write_json(result, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "gate_passers": result["selection"]["gate_passers"],
                "winner": result["selection"]["winner"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
