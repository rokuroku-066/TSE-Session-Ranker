#!/usr/bin/env python3
"""Run the v1.7 exact-liquidity reliability hypothesis family.

The runner deliberately does not tune the rejected v1.6 hard-veto, additive
liquidity, or VWAP-reversal specifications.  It evaluates ten predeclared
mechanisms which use completed JPX D-1 activity as memory, reliability, regime,
or within-pair information.  Every displayed policy has one fixed slot.  The
outcome-free score ledger is canonicalised and hashed before target-session
open-to-close outcomes are joined.

This is retrospective research.  It cannot promote a production model, enable
orders, or claim executable spread/slippage evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
import sys

for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research import model_v16_liquidity_runner as v16  # noqa: E402
from research.model_v13_symbolic_context_runner import G0_FEATURES  # noqa: E402
from tse_session_ranker.data.common import (  # noqa: E402
    normalize_expected_sessions,
    session_calendar_hash,
)
from tse_session_ranker.data.jpx import (  # noqa: E402
    PARSER_VERSION as JPX_PARSER_VERSION,
    collect_jpx,
)
from tse_session_ranker.profit import profit_metrics  # noqa: E402
from tse_session_ranker.research_models import (  # noqa: E402
    date_equal_weights,
    same_day_return_percentile_target,
)
from tse_session_ranker.validation import (  # noqa: E402
    paired_moving_block_bootstrap,
)


PROTOCOL = ROOT / "research/model_v17_liquidity_reliability_protocol.json"
PROTOCOL_SHA256 = (
    "f7d2efa30c5f6ca03a95e1f6e84e0fb6de3f68e8f0d520183877e2f0ab4a416f"
)
PROTOCOL_ID = "model_v17_liquidity_reliability_selection_20260804"
INPUT_LOCK = ROOT / "research/model_v17_replay_input_lock.json"
INPUT_LOCK_SHA256 = (
    "1d9a391c8e6b8d2003498c18ba09dac904e672e1f17c05516998ffb71e11c475"
)
INPUT_LOCK_ID = "model_v17_replay_inputs_20260804"
INPUT_LOCK_SOURCE_SET_SHA256 = (
    "a0fdea19f9e7c771d2e0450720eb551917d9c8b9cfade75260d5cc43adb3a43f"
)
JPX_PARSER_SOURCE = ROOT / "src/tse_session_ranker/data/jpx.py"

DAILY_FORMAT = "jpx_stock_quotations_auction_regular_way_domestic_ordinary"
DAILY_START = pd.Timestamp("2025-08-01")
LEGACY_DAILY_END = pd.Timestamp("2026-03-31")
DAILY_END = pd.Timestamp("2026-07-27")
LEGACY_DAILY_FILES = 160
EXTENSION_DAILY_FILES = 79
DAILY_FILES = LEGACY_DAILY_FILES + EXTENSION_DAILY_FILES

INITIAL_FIT_START = pd.Timestamp("2025-09-01")
PAIR_CROSSFIT_START = pd.Timestamp("2025-12-01")
SELECTION_START = pd.Timestamp("2026-04-01")
SELECTION_END = pd.Timestamp("2026-07-27")
SELECTION_SESSIONS = 79
SELECTION_SLICES = {
    "replay_a": (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-29")),
    "replay_b": (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-27")),
}

C00 = "C00_PRICE_RIDGE_TOP1"
PAIR00 = "PAIR00_PRICE_ONLY_TOP2_RERANK"
C02 = "C02_C00_RANK2"
CANDIDATES = (
    "PAIR01_TOP2_EXACT_LIQ_RERANK",
    "TW02_TURNOVER_WEIGHTED_PRICE_MEMORY",
    "VW03_AGGREGATE_COST_BASIS",
    "RPY04_RETURN_PER_TURNOVER_REVERSAL",
    "AR05_SECURITY_ACTIVITY_REGIME_EXPERTS",
    "PS06_PERSISTENT_VS_ISOLATED_ACTIVITY",
    "VP07_VWAP_RANGE_PRESSURE_MEMORY",
    "RW08_EXACT_LOT_RELIABILITY_WEIGHT",
    "MR09_TURNOVER_WEIGHTED_MARKET_REGIME",
    "AT10_ATTENTION_MIGRATION",
)
ALL_MODELS = (C00, PAIR00, C02, *CANDIDATES)
MATCHED_CONTROL = {
    candidate: (PAIR00 if candidate.startswith("PAIR01_") else C00)
    for candidate in CANDIDATES
}

COSTS_BPS = (20.0, 40.0, 60.0)
PRIMARY_COST_BPS = 40.0
FAMILY_SIZE = len(CANDIDATES)
BOOTSTRAP_BLOCK_LENGTH = 5
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_RANDOM_STATE = 20_260_804
BONFERRONI_CONFIDENCE = 1.0 - 0.10 / FAMILY_SIZE
RIDGE_ALPHA = 1.0
PAIRWISE_C = 1.0

TW02_FEATURES = (
    "tw02_turnover_weighted_oc_mean_5",
    "tw02_turnover_weighted_oc_mean_20",
    "tw02_vs_equal_oc_mean_5",
    "tw02_vs_equal_oc_mean_20",
)
VW03_FEATURES = (
    "vw03_xrank_close_to_avwap20",
    "vw03_xrank_avwap5_to_avwap20",
)
PS06_FEATURES = (
    "ps06_xrank_sustained_activity",
    "ps06_xrank_isolated_activity",
    "ps06_sustained_x_close_vwap",
    "ps06_isolated_x_close_vwap",
)
VP07_FEATURES = (
    "vp07_xrank_pressure_5",
    "vp07_xrank_pressure_20",
)
MR09_FEATURES = (
    "mr09_pressure_breadth_x_momentum_5",
    "mr09_pressure_breadth_x_momentum_20",
    "mr09_turnover_hhi_x_momentum_5",
    "mr09_turnover_hhi_x_momentum_20",
)
AT10_FEATURES = (
    "at10_xrank_attention_migration",
    "at10_attention_x_close_vwap",
)

PAIR00_COLUMNS = ("c00_model_score",)
PAIR01_COLUMNS = (*PAIR00_COLUMNS, *v16.LIQUIDITY_FEATURES)
C00_OOF_SCORE_COLUMNS = (
    "date",
    "code",
    "name",
    "c00_model_score",
    *v16.LIQUIDITY_FEATURES,
)

RAW_RELIABILITY_FEATURES = (
    "tw02_turnover_weighted_oc_mean_5",
    "tw02_turnover_weighted_oc_mean_20",
    "vw03_close_to_avwap20",
    "vw03_avwap5_to_avwap20",
    "rpy04_return_per_turnover_sum5",
    "ps06_sustained_activity",
    "ps06_isolated_activity",
    "vp07_pressure_5",
    "vp07_pressure_20",
    "rw08_traded_lots_med20",
    "mr09_source_turnover",
    "mr09_source_pressure_sign",
    "at10_attention_migration",
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
    "source_rank",
)

_DAILY_NAME = re.compile(r"^stq_(\d{8})\.pdf$")


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


def _atomic_text(path: str | Path, payload: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json(value: dict[str, Any], path: str | Path) -> None:
    _atomic_text(
        path,
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    _atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def validate_protocol(path: str | Path = PROTOCOL) -> tuple[dict[str, Any], str]:
    observed_sha = sha256_file(path)
    if observed_sha != PROTOCOL_SHA256:
        raise ValueError(
            "v1.7 protocol SHA-256 mismatch: "
            f"expected {PROTOCOL_SHA256}, got {observed_sha}"
        )
    protocol = read_json(path)
    if protocol.get("schema_version") != 1:
        raise ValueError("unexpected v1.7 protocol schema")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected v1.7 protocol id")
    authority = protocol.get("authority", {})
    for key in (
        "project_level_untouched",
        "production_promotion_allowed",
        "production_model_changed",
        "orders_allowed",
    ):
        if authority.get(key) is not False:
            raise ValueError(f"v1.7 protocol must set {key}=false")
    if authority.get("analysis_type") != "retrospective_candidate_specific_selection":
        raise ValueError("v1.7 protocol analysis authority changed")

    sources = protocol.get("source_contract", {})
    legacy = sources.get("legacy_daily", {})
    replay = sources.get("replay_daily", {})
    parser_contract = sources.get("parser", {})
    if (
        int(legacy.get("files", -1)) != LEGACY_DAILY_FILES
        or int(legacy.get("sessions", -1)) != LEGACY_DAILY_FILES
        or legacy.get("first_file") != "stq_20250801.pdf"
        or legacy.get("last_file") != "stq_20260331.pdf"
    ):
        raise ValueError("v1.7 legacy daily source contract changed")
    if (
        replay.get("start") != str(SELECTION_START.date())
        or replay.get("end") != str(SELECTION_END.date())
        or int(replay.get("files", -1)) != EXTENSION_DAILY_FILES
        or int(replay.get("sessions", -1)) != SELECTION_SESSIONS
        or replay.get("input_lock") != "research/model_v17_replay_input_lock.json"
        or replay.get("input_lock_sha256") != INPUT_LOCK_SHA256
        or replay.get("source_set_sha256") != INPUT_LOCK_SOURCE_SET_SHA256
    ):
        raise ValueError("v1.7 replay daily source contract changed")
    if parser_contract != {
        "path": "src/tse_session_ranker/data/jpx.py",
        "version": JPX_PARSER_VERSION,
        "sha256": sha256_file(JPX_PARSER_SOURCE),
        "required_rejected_rows": 0,
    }:
        raise ValueError("v1.7 parser contract changed")

    periods = protocol.get("periods", {})
    if (
        periods.get("initial_training_start") != str(INITIAL_FIT_START.date())
        or periods.get("pair_crossfit_start") != str(PAIR_CROSSFIT_START.date())
        or periods.get("selection")
        != [str(SELECTION_START.date()), str(SELECTION_END.date())]
        or int(periods.get("selection_sessions", -1)) != SELECTION_SESSIONS
    ):
        raise ValueError("v1.7 registered periods changed")
    if tuple(protocol.get("common_price_features", ())) != tuple(G0_FEATURES):
        raise ValueError("v1.7 common price features changed")
    if tuple(protocol.get("exact_liquidity_rank_features", ())) != tuple(
        v16.LIQUIDITY_FEATURES
    ):
        raise ValueError("v1.7 exact-liquidity feature registry changed")

    control_ids = tuple(item["id"] for item in protocol.get("controls", []))
    if control_ids != (C00, PAIR00, C02):
        raise ValueError("v1.7 control registry changed")
    pair_control = protocol["controls"][1]
    if (
        float(pair_control.get("C", -1.0)) != PAIRWISE_C
        or pair_control.get("fit_intercept") is not False
        or int(pair_control.get("max_iter", -1)) != 1_000
        or int(pair_control.get("random_state", -1)) != BOOTSTRAP_RANDOM_STATE
    ):
        raise ValueError("v1.7 PAIR00 estimator contract changed")
    registered = tuple(item["id"] for item in protocol.get("candidates", []))
    if registered != CANDIDATES:
        raise ValueError("v1.7 protocol candidate registry changed")
    registered_controls = {
        item["id"]: item.get("matched_control")
        for item in protocol["candidates"]
    }
    if registered_controls != MATCHED_CONTROL:
        raise ValueError("v1.7 matched-control registry changed")
    if any(int(item.get("slot", -1)) != 1 for item in protocol["candidates"]):
        raise ValueError("v1.7 candidate slot count changed")
    pair_candidate = protocol["candidates"][0]
    if (
        float(pair_candidate.get("C", -1.0)) != PAIRWISE_C
        or pair_candidate.get("fit_intercept") is not False
        or int(pair_candidate.get("max_iter", -1)) != 1_000
        or int(pair_candidate.get("random_state", -1))
        != BOOTSTRAP_RANDOM_STATE + 1
    ):
        raise ValueError("v1.7 PAIR01 estimator contract changed")

    evaluation = protocol.get("evaluation", {})
    if (
        int(evaluation.get("fixed_slots_per_model_date", -1)) != 1
        or int(evaluation.get("candidate_variants", -1)) != FAMILY_SIZE
        or evaluation.get("no_capacity_grid") is not True
        or tuple(float(value) for value in evaluation.get("costs_bps", ()))
        != COSTS_BPS
        or float(evaluation.get("primary_cost_bps", -1.0)) != PRIMARY_COST_BPS
    ):
        raise ValueError("v1.7 evaluation family or costs changed")
    expected_slices = {
        key: [str(bounds[0].date()), str(bounds[1].date())]
        for key, bounds in SELECTION_SLICES.items()
    }
    if evaluation.get("selection_slices") != expected_slices:
        raise ValueError("v1.7 fixed selection slices changed")
    bootstrap = evaluation.get("bootstrap", {})
    if (
        bootstrap.get("method") != "paired moving block"
        or int(bootstrap.get("block_length", -1)) != BOOTSTRAP_BLOCK_LENGTH
        or int(bootstrap.get("samples", -1)) != BOOTSTRAP_SAMPLES
        or int(bootstrap.get("random_state", -1)) != BOOTSTRAP_RANDOM_STATE
        or bootstrap.get("candidate_seed_rule")
        != (
            "random_state + zero-based candidate order index, yielding "
            "20260804 through 20260813"
        )
        or not math.isclose(
            float(bootstrap.get("bonferroni_individual_confidence", -1.0)),
            BONFERRONI_CONFIDENCE,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("v1.7 bootstrap contract changed")
    expected_gates = {
        "net40_mean_positive": True,
        "net40_median_positive": True,
        "net60_mean_positive": True,
        "both_fixed_slices_net40_positive": True,
        "positive_months_net40_at_least": 3,
        "top4_days_removed_net40_positive": True,
        "top5_profit_codes_cash_net40_positive": True,
        "familywise_paired_lower_vs_control_nonnegative": True,
        "unique_codes_at_least": 40,
        "maximum_code_share_at_most": 0.05,
        "top10_code_share_at_most": 0.25,
        "executed_days_at_least": 72,
        "executed_slot_fraction_at_least": 0.8,
    }
    if evaluation.get("selection_gate_all_required") != expected_gates:
        raise ValueError("v1.7 selection gate registry changed")
    return protocol, observed_sha


def _daily_path_date(path: Path) -> pd.Timestamp:
    match = _DAILY_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected daily PDF name: {path.name}")
    return pd.to_datetime(match.group(1), format="%Y%m%d", errors="raise").normalize()


def _expected_extension_sources(
    path: str | Path = INPUT_LOCK,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Read the outcome-blind extension lock and validate it fail-closed."""

    lock_path = Path(path)
    observed_lock_sha = sha256_file(lock_path)
    if observed_lock_sha != INPUT_LOCK_SHA256:
        raise ValueError(
            "v1.7 replay input-lock SHA-256 mismatch: "
            f"expected {INPUT_LOCK_SHA256}, got {observed_lock_sha}"
        )
    lock = read_json(lock_path)
    if lock.get("schema_version") != 1:
        raise ValueError("v1.7 replay input-lock schema changed")
    if lock.get("lock_id") != INPUT_LOCK_ID:
        raise ValueError("v1.7 replay input-lock id changed")
    if lock.get("status") != "source_locked_before_outcome_parse":
        raise ValueError("v1.7 replay sources were not locked before outcome parse")
    if lock.get("source_set_sha256") != INPUT_LOCK_SOURCE_SET_SHA256:
        raise ValueError("v1.7 replay input-lock source-set digest changed")
    authority = lock.get("authority", {})
    expected_authority = {
        "project_level_untouched": False,
        "raw_market_outcomes_parsed_before_lock": False,
        "retrospective_acquisition": True,
    }
    if authority != expected_authority:
        raise ValueError("v1.7 replay input-lock authority changed")
    period = lock.get("period", {})
    if period != {
        "start": str(SELECTION_START.date()),
        "end": str(SELECTION_END.date()),
        "sessions": SELECTION_SESSIONS,
    }:
        raise ValueError("v1.7 replay input-lock period changed")
    parser_contract = lock.get("parser_contract", {})
    if parser_contract != {
        "source_path": "src/tse_session_ranker/data/jpx.py",
        "source_sha256": sha256_file(JPX_PARSER_SOURCE),
        "version": JPX_PARSER_VERSION,
    }:
        raise ValueError("v1.7 replay parser contract changed")

    records = lock.get("files")
    if not isinstance(records, list) or len(records) != EXTENSION_DAILY_FILES:
        raise ValueError("v1.7 replay input-lock must contain exactly 79 files")
    sources: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("v1.7 replay input-lock file record is invalid")
        name = str(item.get("name", ""))
        if name in sources:
            raise ValueError(f"v1.7 replay input-lock duplicates {name}")
        date = _daily_path_date(Path(name))
        if str(date.date()) != str(item.get("date")):
            raise ValueError(f"v1.7 replay input-lock date mismatch: {name}")
        if not SELECTION_START <= date <= SELECTION_END:
            raise ValueError(f"v1.7 replay input-lock date outside selection: {name}")
        if not isinstance(item.get("bytes"), int) or int(item["bytes"]) <= 0:
            raise ValueError(f"v1.7 replay input-lock byte size invalid: {name}")
        if re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) is None:
            raise ValueError(f"v1.7 replay input-lock digest invalid: {name}")
        sources[name] = item
    locked_dates = pd.DatetimeIndex(
        [_daily_path_date(Path(name)) for name in sorted(sources)]
    )
    if (
        len(locked_dates) != SELECTION_SESSIONS
        or locked_dates.min() != SELECTION_START
        or locked_dates.max() != SELECTION_END
    ):
        raise ValueError("v1.7 replay input-lock session bounds changed")
    return sources, lock


def _validate_daily_directory(
    directory: str | Path,
) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any]]:
    root = Path(directory)
    paths = sorted(root.glob("*.pdf"))
    if len(paths) != DAILY_FILES:
        raise ValueError(
            f"v1.7 requires exactly {DAILY_FILES} daily PDFs, got {len(paths)}"
        )
    dated = [(_daily_path_date(path), path) for path in paths]
    dates = pd.DatetimeIndex(date for date, _ in dated)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("v1.7 daily PDF dates are duplicated or unsorted")
    if dates.min() != DAILY_START or dates.max() != DAILY_END:
        raise ValueError("v1.7 daily PDF bounds changed")

    legacy_expected = v16._expected_daily_sources()
    extension_expected, input_lock = _expected_extension_sources()
    legacy_paths = [path for date, path in dated if date <= LEGACY_DAILY_END]
    extension_paths = [path for date, path in dated if date > LEGACY_DAILY_END]
    if len(legacy_paths) != LEGACY_DAILY_FILES:
        raise ValueError("v1.7 legacy daily source count changed")
    if len(extension_paths) != EXTENSION_DAILY_FILES:
        raise ValueError("v1.7 extension daily source count changed")
    if [path.name for path in legacy_paths] != sorted(legacy_expected):
        raise ValueError("v1.7 legacy daily source names changed")
    if [path.name for path in extension_paths] != sorted(extension_expected):
        raise ValueError("v1.7 extension daily source names differ from the lock")

    checks: list[dict[str, Any]] = []
    for path in paths:
        observed = {
            "name": path.name,
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        legacy_record = legacy_expected.get(path.name)
        extension_record = extension_expected.get(path.name)
        if legacy_record is not None:
            if observed["sha256"] != str(legacy_record["sha256"]):
                raise ValueError(f"v1.7 legacy source SHA-256 mismatch: {path.name}")
        elif extension_record is not None:
            if observed["bytes"] != int(extension_record["bytes"]):
                raise ValueError(f"v1.7 extension source byte mismatch: {path.name}")
            if observed["sha256"] != str(extension_record["sha256"]):
                raise ValueError(f"v1.7 extension source SHA-256 mismatch: {path.name}")
        else:
            raise AssertionError(f"v1.7 unlocked daily source reached parser: {path.name}")
        checks.append(observed)
    return paths, checks, input_lock


def load_prices(
    *,
    price_warmup_directory: str | Path,
    daily_directory: str | Path,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, Any]]:
    """Load three locked warm-up PDFs and 239 daily PDFs fail-closed."""

    warmup, warmup_checks = v16._validate_directory(
        price_warmup_directory, v16._expected_price_sources()
    )
    daily, daily_checks, input_lock = _validate_daily_directory(daily_directory)
    prices, parse = collect_jpx([*warmup, *daily])
    daily_reports = [
        item
        for item in parse["inputs"]
        if _DAILY_NAME.fullmatch(Path(str(item["path"])).name)
    ]
    if len(daily_reports) != DAILY_FILES:
        raise ValueError("v1.7 did not parse exactly 239 daily PDFs")
    if any(int(item["rejected_rows"]) != 0 for item in daily_reports):
        raise ValueError("v1.7 daily parser rejected an ordinary-stock row")
    if any(item["source_format"] != DAILY_FORMAT for item in daily_reports):
        raise ValueError("v1.7 daily source format changed")
    for semantics in (
        "volume_semantics",
        "turnover_semantics",
        "trading_unit_semantics",
    ):
        if any(item.get(semantics) is None for item in daily_reports):
            raise ValueError(f"v1.7 daily {semantics} is missing")

    parsed_dates = pd.to_datetime(prices["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("v1.7 prices contain an invalid date")
    sessions = normalize_expected_sessions(parsed_dates.drop_duplicates())
    daily_rows = prices.loc[prices["source_format"].eq(DAILY_FORMAT)].copy()
    daily_sessions = normalize_expected_sessions(daily_rows["date"].drop_duplicates())
    if len(daily_sessions) != DAILY_FILES:
        raise ValueError("v1.7 parsed daily session count changed")
    if daily_sessions.min() != DAILY_START or daily_sessions.max() != DAILY_END:
        raise ValueError("v1.7 parsed daily date bounds changed")
    extension_checks = [
        item
        for item in daily_checks
        if _daily_path_date(Path(item["name"])) > LEGACY_DAILY_END
    ]
    return prices, sessions, {
        "parser_version": JPX_PARSER_VERSION,
        "price_warmup_sources": warmup_checks,
        "daily_source_count": len(daily_checks),
        "legacy_daily_source_count": LEGACY_DAILY_FILES,
        "extension_daily_source_count": EXTENSION_DAILY_FILES,
        "replay_input_lock_id": input_lock["lock_id"],
        "replay_input_lock_sha256": INPUT_LOCK_SHA256,
        "replay_input_lock_source_set_sha256": input_lock["source_set_sha256"],
        "daily_source_set_sha256": hashlib.sha256(
            "\n".join(item["sha256"] for item in daily_checks).encode("ascii")
        ).hexdigest(),
        "extension_source_set_sha256": hashlib.sha256(
            "\n".join(item["sha256"] for item in extension_checks).encode("ascii")
        ).hexdigest(),
        "daily_parsed_rows": int(sum(int(item["parsed_rows"]) for item in daily_reports)),
        "daily_rejected_rows": 0,
        "daily_sessions": len(daily_sessions),
        "daily_date_bounds": [
            str(daily_sessions.min().date()),
            str(daily_sessions.max().date()),
        ],
        "calendar_sha256": session_calendar_hash(sessions),
        "pdftotext": v16._pdftotext_runtime(),
    }


def _rolling(
    values: pd.Series,
    codes: pd.Series,
    *,
    window: int,
    operation: str,
) -> pd.Series:
    return v16._rolling(values, codes, window=window, operation=operation)


def _rank_to_unit(
    values: pd.Series,
    dates: pd.Series,
    universe: pd.Series,
) -> pd.Series:
    return v16._rank_to_unit(values, dates, universe)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(denominator, errors="coerce")
    return pd.to_numeric(numerator, errors="coerce") / numeric.where(numeric.gt(0.0))


def build_reliability_features(
    prices: pd.DataFrame,
    sessions: Iterable[object],
) -> pd.DataFrame:
    """Build all v1.7 raw mechanisms from completed D-1 daily rows.

    Every rolling statistic is calculated on the source-date table first and
    only then mapped to the next official session.  A target-date mutation
    therefore cannot change that target's feature row.
    """

    required = {
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "vwap",
        "trading_unit",
        "source_format",
    }
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"v1.7 daily source lacks columns: {missing}")
    calendar = normalize_expected_sessions(sessions)
    daily = prices.loc[prices["source_format"].eq(DAILY_FORMAT), sorted(required)].copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["code"] = daily["code"].astype(str)
    daily = daily.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    if daily[["date", "code"]].duplicated().any():
        raise ValueError("v1.7 daily source has duplicate date/code rows")

    position_by_date = pd.Series(np.arange(len(calendar)), index=calendar)
    daily["_session_position"] = daily["date"].map(position_by_date)
    if daily["_session_position"].isna().any():
        raise ValueError("v1.7 daily row is outside the official calendar")
    codes = daily["code"]
    numeric_columns = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "vwap",
        "trading_unit",
    )
    for column in numeric_columns:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    positive_activity_columns = (
        "close",
        "volume",
        "turnover",
        "vwap",
        "trading_unit",
    )
    positive = daily[list(positive_activity_columns)].gt(0.0).all(axis=1)
    oldest = daily["_session_position"].groupby(codes, sort=False).shift(19)
    consecutive20 = (daily["_session_position"] - oldest).eq(19)
    positive20 = _rolling(
        positive.astype(float), codes, window=20, operation="sum"
    ).eq(20.0)
    unit_min20 = _rolling(daily["trading_unit"], codes, window=20, operation="min")
    unit_max20 = _rolling(daily["trading_unit"], codes, window=20, operation="max")
    daily["_history_complete20"] = (
        consecutive20 & positive20 & unit_min20.eq(unit_max20) & unit_min20.gt(0.0)
    )

    oc_return = 100.0 * (_safe_ratio(daily["close"], daily["open"]) - 1.0)
    turnover_oc = daily["turnover"] * oc_return
    turnover_sum5 = _rolling(daily["turnover"], codes, window=5, operation="sum")
    turnover_sum20 = _rolling(daily["turnover"], codes, window=20, operation="sum")
    volume_sum5 = _rolling(daily["volume"], codes, window=5, operation="sum")
    volume_sum20 = _rolling(daily["volume"], codes, window=20, operation="sum")
    daily["tw02_turnover_weighted_oc_mean_5"] = _safe_ratio(
        _rolling(turnover_oc, codes, window=5, operation="sum"), turnover_sum5
    )
    daily["tw02_turnover_weighted_oc_mean_20"] = _safe_ratio(
        _rolling(turnover_oc, codes, window=20, operation="sum"), turnover_sum20
    )

    avwap5 = _safe_ratio(turnover_sum5, volume_sum5)
    avwap20 = _safe_ratio(turnover_sum20, volume_sum20)
    daily["vw03_close_to_avwap20"] = _safe_ratio(daily["close"], avwap20) - 1.0
    daily["vw03_avwap5_to_avwap20"] = _safe_ratio(avwap5, avwap20) - 1.0

    return_per_turnover = oc_return / daily["turnover"].where(daily["turnover"].gt(0.0))
    daily["rpy04_return_per_turnover_sum5"] = _rolling(
        return_per_turnover, codes, window=5, operation="sum"
    )

    turnover_med3 = _rolling(daily["turnover"], codes, window=3, operation="median")
    prior17 = _rolling(
        daily["turnover"].groupby(codes, sort=False).shift(3),
        codes,
        window=17,
        operation="median",
    )
    prior19 = _rolling(
        daily["turnover"].groupby(codes, sort=False).shift(1),
        codes,
        window=19,
        operation="median",
    )
    sustained = np.log(_safe_ratio(turnover_med3, prior17)).clip(-3.0, 3.0)
    isolated = (
        np.log(_safe_ratio(daily["turnover"], prior19)) - sustained
    ).clip(-3.0, 3.0)
    daily["ps06_sustained_activity"] = sustained
    daily["ps06_isolated_activity"] = isolated

    price_range = daily["high"] - daily["low"]
    pressure = pd.Series(np.nan, index=daily.index, dtype=float)
    nonzero_range = price_range.gt(0.0)
    pressure.loc[nonzero_range] = (
        (daily.loc[nonzero_range, "close"] - daily.loc[nonzero_range, "vwap"])
        / price_range.loc[nonzero_range]
    ).clip(-1.0, 1.0)
    zero_range_valid = price_range.eq(0.0) & np.isclose(
        daily["close"].to_numpy(dtype=float),
        daily["vwap"].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )
    pressure.loc[zero_range_valid] = 0.0
    turnover_pressure = daily["turnover"] * pressure
    daily["vp07_pressure_5"] = _safe_ratio(
        _rolling(turnover_pressure, codes, window=5, operation="sum"), turnover_sum5
    )
    daily["vp07_pressure_20"] = _safe_ratio(
        _rolling(turnover_pressure, codes, window=20, operation="sum"), turnover_sum20
    )

    traded_lots = _safe_ratio(daily["volume"], daily["trading_unit"])
    daily["rw08_traded_lots_med20"] = _rolling(
        traded_lots, codes, window=20, operation="median"
    )

    # Keep the per-security D-1 ingredients here.  The market aggregates are
    # deliberately delayed until after the v1.6 panel is merged, because the
    # registered denominator is that panel's common eligible cross-section.
    daily["mr09_source_turnover"] = daily["turnover"]
    daily["mr09_source_pressure_sign"] = np.sign(
        daily["close"] - daily["vwap"]
    )

    source_turnover_rank = _rank_to_unit(
        daily["turnover"],
        daily["date"],
        daily["_history_complete20"].fillna(False).astype(bool)
        & daily["turnover"].gt(0.0),
    )
    prior_rank_med4 = _rolling(
        source_turnover_rank.groupby(codes, sort=False).shift(1),
        codes,
        window=4,
        operation="median",
    )
    daily["at10_attention_migration"] = source_turnover_rank - prior_rank_med4

    next_session = dict(zip(calendar[:-1], calendar[1:], strict=True))
    daily["feature_source_max_date_v17"] = daily["date"]
    daily["date"] = daily["date"].map(next_session)
    daily = daily.dropna(subset=["date"]).copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    rank_universe = daily["_history_complete20"].fillna(False).astype(bool)
    rank_sources = {
        "vw03_xrank_close_to_avwap20": "vw03_close_to_avwap20",
        "vw03_xrank_avwap5_to_avwap20": "vw03_avwap5_to_avwap20",
        "rpy04_xrank_return_per_turnover_sum5": "rpy04_return_per_turnover_sum5",
        "ps06_xrank_sustained_activity": "ps06_sustained_activity",
        "ps06_xrank_isolated_activity": "ps06_isolated_activity",
        "vp07_xrank_pressure_5": "vp07_pressure_5",
        "vp07_xrank_pressure_20": "vp07_pressure_20",
        "at10_xrank_attention_migration": "at10_attention_migration",
    }
    for output, source in rank_sources.items():
        daily[output] = _rank_to_unit(
            daily[source], daily["date"], rank_universe & daily[source].notna()
        )

    projection = [
        "date",
        "code",
        "feature_source_max_date_v17",
        "_history_complete20",
        *RAW_RELIABILITY_FEATURES,
        *rank_sources,
    ]
    result = daily.loc[:, projection].copy()
    if result[["date", "code"]].duplicated().any():
        raise ValueError("v1.7 target feature rows are duplicated")
    if (
        result["feature_source_max_date_v17"].notna()
        & result["feature_source_max_date_v17"].ge(result["date"])
    ).any():
        raise ValueError("v1.7 reliability feature is not strictly prior")
    return result.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def build_model_panel(
    prices: pd.DataFrame,
    sessions: Iterable[object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    panel, coverage, checks = v16.build_model_panel(prices, sessions)
    reliability = build_reliability_features(prices, sessions)
    panel = panel.merge(
        reliability,
        on=["date", "code"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    panel["tw02_vs_equal_oc_mean_5"] = (
        panel["tw02_turnover_weighted_oc_mean_5"] - panel["oc_mean_5"]
    )
    panel["tw02_vs_equal_oc_mean_20"] = (
        panel["tw02_turnover_weighted_oc_mean_20"] - panel["oc_mean_20"]
    )
    panel["ps06_sustained_x_close_vwap"] = (
        panel["ps06_xrank_sustained_activity"]
        * panel["xrank_liq_close_vwap_dev1"]
    )
    panel["ps06_isolated_x_close_vwap"] = (
        panel["ps06_xrank_isolated_activity"]
        * panel["xrank_liq_close_vwap_dev1"]
    )
    panel["at10_attention_x_close_vwap"] = (
        panel["at10_xrank_attention_migration"]
        * panel["xrank_liq_close_vwap_dev1"]
    )

    # MR09's D-1 market breadth and HHI use exactly the registered common
    # score-eligible cross-section.  The values are attached to every row on
    # the target date only after that denominator has been fixed.
    strict = panel["common_score_eligible"].fillna(False).astype(bool)
    source_turnover = pd.to_numeric(
        panel["mr09_source_turnover"], errors="coerce"
    )
    source_sign = pd.to_numeric(
        panel["mr09_source_pressure_sign"], errors="coerce"
    )
    market_valid = (
        strict
        & source_turnover.gt(0.0)
        & source_turnover.notna()
        & source_sign.notna()
        & np.isfinite(source_sign)
    )
    eligible_turnover = source_turnover.where(market_valid)
    market_total = eligible_turnover.groupby(panel["date"], sort=False).transform(
        "sum"
    )
    market_total = market_total.where(market_total.gt(0.0))
    panel["mr09_market_pressure_breadth"] = (
        (source_turnover * source_sign)
        .where(market_valid)
        .groupby(panel["date"], sort=False)
        .transform("sum")
        / market_total
    )
    market_share = eligible_turnover / market_total
    panel["mr09_market_turnover_hhi"] = market_share.pow(2).groupby(
        panel["date"], sort=False
    ).transform("sum")
    panel["mr09_pressure_breadth_x_momentum_5"] = (
        panel["mr09_market_pressure_breadth"] * panel["xrank_close_momentum_5"]
    )
    panel["mr09_pressure_breadth_x_momentum_20"] = (
        panel["mr09_market_pressure_breadth"] * panel["xrank_close_momentum_20"]
    )
    panel["mr09_turnover_hhi_x_momentum_5"] = (
        panel["mr09_market_turnover_hhi"] * panel["xrank_close_momentum_5"]
    )
    panel["mr09_turnover_hhi_x_momentum_20"] = (
        panel["mr09_market_turnover_hhi"] * panel["xrank_close_momentum_20"]
    )
    source_dates = panel.loc[
        strict & panel["feature_source_max_date_v17"].notna(),
        ["date", "feature_source_max_date_v17"],
    ]
    if source_dates["feature_source_max_date_v17"].ge(source_dates["date"]).any():
        raise ValueError("v1.7 merged reliability feature is not strictly prior")
    checks = {
        **checks,
        "v17_feature_rows": int(len(reliability)),
        "v17_feature_source_before_target": True,
    }
    return panel, coverage, checks


def _linear_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def fit_daily_rank_ridge(
    training: pd.DataFrame,
    features: Sequence[str],
    *,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> Pipeline:
    """Fit the frozen alpha-one daily-rank Ridge with date-equal weights."""

    columns = tuple(features)
    required = {"date", "code", "oc_return_pct", *columns}
    missing = sorted(required - set(training.columns))
    if missing:
        raise ValueError(f"v1.7 Ridge training lacks columns: {missing}")
    if training.empty or training[["date", "code"]].duplicated().any():
        raise ValueError("v1.7 Ridge training is empty or duplicated")
    target = (
        training["_daily_rank_target"].astype(float)
        if "_daily_rank_target" in training
        else same_day_return_percentile_target(training)
    )
    weights = (
        date_equal_weights(training)
        if sample_weight is None
        else pd.Series(np.asarray(sample_weight, dtype=float), index=training.index)
    )
    if (
        weights.isna().any()
        or not np.isfinite(weights.to_numpy()).all()
        or weights.le(0.0).any()
    ):
        raise ValueError("v1.7 Ridge weights must be finite and positive")
    per_date = weights.groupby(training["date"], sort=False).sum()
    if not np.allclose(per_date.to_numpy(), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("v1.7 Ridge weights are not date-equal")
    model = _linear_pipeline(Ridge(alpha=RIDGE_ALPHA))
    model.fit(
        training.loc[:, list(columns)],
        target,
        model__sample_weight=weights,
    )
    return model


def reliability_weights(training: pd.DataFrame) -> pd.Series:
    """Return positive lot-reliability weights summing to one per date."""

    lots = pd.to_numeric(training["rw08_traded_lots_med20"], errors="coerce")
    valid = lots.gt(0.0) & np.isfinite(lots)
    if not valid.all():
        raise ValueError("RW08 requires positive observed traded-lot history")
    percentile = lots.groupby(training["date"], sort=False).rank(
        method="average", pct=True
    )
    weights = percentile / percentile.groupby(training["date"], sort=False).transform("sum")
    if weights.le(0.0).any() or not np.isfinite(weights.to_numpy()).all():
        raise ValueError("RW08 reliability weights are invalid")
    if not np.allclose(
        weights.groupby(training["date"], sort=False).sum().to_numpy(),
        1.0,
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError("RW08 reliability weights are not date-equal")
    return weights.astype(float)


def _finite_feature_mask(frame: pd.DataFrame, features: Sequence[str]) -> pd.Series:
    if not features:
        return pd.Series(True, index=frame.index)
    numeric = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    finite = pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1), index=frame.index
    )
    return numeric.notna().all(axis=1) & finite


def _candidate_training(
    panel: pd.DataFrame,
    *,
    end: pd.Timestamp,
    exact_features: Sequence[str] = (),
) -> pd.DataFrame:
    mask = (
        panel["date"].between(INITIAL_FIT_START, end)
        & panel["common_training_eligible"].fillna(False).astype(bool)
    )
    frame = panel.loc[mask].copy()
    if frame.empty:
        raise ValueError("v1.7 candidate training frame is empty")
    # Freeze the daily-rank label on the registered common universe before a
    # mechanism-specific availability mask is considered (VP07 only).
    frame["_daily_rank_target"] = same_day_return_percentile_target(frame)
    if exact_features:
        frame = frame.loc[_finite_feature_mask(frame, exact_features)].copy()
    if frame.empty:
        raise ValueError("v1.7 candidate training frame is empty")
    return frame


def _candidate_scoring(
    panel: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    exact_features: Sequence[str] = (),
) -> pd.DataFrame:
    mask = (
        panel["date"].isin(scheduled)
        & panel["common_score_eligible"].fillna(False).astype(bool)
    )
    frame = panel.loc[mask].copy()
    if exact_features:
        frame = frame.loc[_finite_feature_mask(frame, exact_features)].copy()
    return frame


def _top_scored_rows(
    scoring: pd.DataFrame,
    scores: Sequence[float] | np.ndarray | pd.Series,
    *,
    count: int,
) -> pd.DataFrame:
    ranked = scoring.copy()
    ranked["c00_model_score"] = np.asarray(scores, dtype=float)
    if len(ranked) != len(scoring) or not np.isfinite(ranked["c00_model_score"]).all():
        raise ValueError("v1.7 scorer produced invalid values")
    ranked = ranked.sort_values(
        ["date", "c00_model_score", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranked = ranked.groupby("date", sort=False, as_index=False).head(count).copy()
    ranked["source_rank"] = ranked.groupby("date", sort=False).cumcount() + 1
    return ranked


def _empty_ledger(candidate_id: str, scheduled: pd.DatetimeIndex) -> pd.DataFrame:
    output = pd.DataFrame({"date": scheduled})
    output["candidate_id"] = candidate_id
    output["model_rank"] = 1
    output["code"] = pd.NA
    output["name"] = pd.NA
    output["model_score"] = np.nan
    output["pre_veto_code"] = pd.NA
    output["vetoed"] = False
    output["source_rank"] = pd.NA
    return output.loc[:, list(SCORE_LEDGER_COLUMNS)]


def _complete_one_slot(
    selected: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    candidate_id: str,
) -> pd.DataFrame:
    desired = pd.DataFrame({"date": scheduled})
    projection = selected.loc[
        :,
        ["date", "code", "name", "model_score", "source_rank"],
    ].copy()
    if projection["date"].duplicated().any():
        raise ValueError(f"v1.7 {candidate_id} selected duplicate dates")
    output = desired.merge(projection, on="date", how="left", validate="one_to_one")
    output["candidate_id"] = candidate_id
    output["model_rank"] = 1
    output["pre_veto_code"] = output["code"]
    output["vetoed"] = False
    return output.loc[:, list(SCORE_LEDGER_COLUMNS)]


def _model_top_one(
    scoring: pd.DataFrame,
    scores: Sequence[float] | np.ndarray | pd.Series,
    scheduled: pd.DatetimeIndex,
    candidate_id: str,
) -> pd.DataFrame:
    if scoring.empty:
        return _empty_ledger(candidate_id, scheduled)
    ranked = scoring.loc[:, ["date", "code", "name"]].copy()
    ranked["model_score"] = np.asarray(scores, dtype=float)
    if not np.isfinite(ranked["model_score"]).all():
        raise ValueError(f"v1.7 {candidate_id} produced a non-finite score")
    ranked = ranked.sort_values(
        ["date", "model_score", "code"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranked = ranked.groupby("date", sort=False, as_index=False).head(1).copy()
    ranked["source_rank"] = 1
    return _complete_one_slot(ranked, scheduled, candidate_id)


def _monthly_c00_scores(
    panel: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    periods = pd.period_range(scheduled.min().to_period("M"), scheduled.max().to_period("M"))
    for period in periods:
        first = period.start_time.normalize()
        month_dates = scheduled[scheduled.to_period("M") == period]
        training = _candidate_training(panel, end=first - pd.Timedelta(days=1))
        scoring = _candidate_scoring(panel, month_dates)
        model = fit_daily_rank_ridge(training, G0_FEATURES)
        if not scoring.empty:
            scored = scoring.copy()
            scored["c00_model_score"] = model.predict(scoring.loc[:, list(G0_FEATURES)])
            # Keep the cross-fit score table outcome-free.  Historical pair
            # labels are joined later and only to rows strictly before the
            # outer pair-model scoring month.
            parts.append(scored.loc[:, list(C00_OOF_SCORE_COLUMNS)].copy())
        folds.append(
            {
                "candidate_id": C00,
                "period": str(period),
                "train_start": str(training["date"].min().date()),
                "train_end": str(training["date"].max().date()),
                "train_dates": int(training["date"].nunique()),
                "train_rows": int(len(training)),
                "score_rows": int(len(scoring)),
                "strictly_prior_training": bool(training["date"].max() < first),
            }
        )
    frame = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=C00_OOF_SCORE_COLUMNS)
    )
    return frame, folds


def build_pair_training_rows(
    c00_top2: pd.DataFrame,
    *,
    before: pd.Timestamp,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Build symmetric, prior cross-fit pair rows for one pairwise model."""

    eligible = c00_top2.loc[c00_top2["date"].lt(before)].copy()
    rows: list[dict[str, Any]] = []
    for date, group in eligible.groupby("date", sort=True):
        group = group.sort_values("code", kind="stable")
        if len(group) != 2 or group["oc_return_pct"].isna().any():
            continue
        left, right = group.iloc[0], group.iloc[1]
        left_return = float(left["oc_return_pct"])
        right_return = float(right["oc_return_pct"])
        if left_return == right_return:
            continue
        difference = {
            column: float(left[column]) - float(right[column])
            if pd.notna(left[column]) and pd.notna(right[column])
            else np.nan
            for column in feature_columns
        }
        target = int(left_return > right_return)
        rows.append({"date": date, "target": target, **difference})
        rows.append(
            {
                "date": date,
                "target": 1 - target,
                **{
                    column: (-value if np.isfinite(value) else np.nan)
                    for column, value in difference.items()
                },
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or set(frame["target"].unique()) != {0, 1}:
        raise ValueError("v1.7 pairwise training has insufficient classes")
    return frame


def fit_pairwise_model(
    rows: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    random_state: int,
) -> Pipeline:
    weights = pd.Series(0.5, index=rows.index, dtype=float)
    if not np.allclose(
        weights.groupby(rows["date"], sort=False).sum().to_numpy(), 1.0
    ):
        raise AssertionError("v1.7 pairwise rows are not date-equal")
    model = _linear_pipeline(
        LogisticRegression(
            C=PAIRWISE_C,
            class_weight=None,
            fit_intercept=False,
            max_iter=1_000,
            random_state=random_state,
            solver="lbfgs",
        )
    )
    model.fit(
        rows.loc[:, list(feature_columns)],
        rows["target"].astype(int),
        model__sample_weight=weights,
    )
    return model


def choose_pairwise_top_one(
    top2: pd.DataFrame,
    model: Pipeline,
    feature_columns: Sequence[str],
    scheduled: pd.DatetimeIndex,
    candidate_id: str,
) -> pd.DataFrame:
    selected: list[dict[str, Any]] = []
    for date, group in top2.groupby("date", sort=True):
        group = group.sort_values("code", kind="stable")
        if len(group) != 2:
            continue
        left, right = group.iloc[0], group.iloc[1]
        difference = pd.DataFrame(
            [
                {
                    column: float(left[column]) - float(right[column])
                    if pd.notna(left[column]) and pd.notna(right[column])
                    else np.nan
                    for column in feature_columns
                }
            ]
        )
        decision = float(model.decision_function(difference)[0])
        # ``group`` is code-ascending, so decision zero follows the registered
        # security-code tie break.
        winner = left if decision >= 0.0 else right
        selected.append(
            {
                "date": date,
                "code": winner["code"],
                "name": winner["name"],
                "model_score": abs(decision),
                "source_rank": int(winner["source_rank"]),
            }
        )
    if not selected:
        return _empty_ledger(candidate_id, scheduled)
    return _complete_one_slot(pd.DataFrame(selected), scheduled, candidate_id)


def _selection_periods(scheduled: pd.DatetimeIndex) -> Iterable[tuple[pd.Period, pd.DatetimeIndex]]:
    for period in pd.period_range(scheduled.min().to_period("M"), scheduled.max().to_period("M")):
        yield period, scheduled[scheduled.to_period("M") == period]


def _ridge_candidate(
    panel: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    candidate_id: str,
    features: Sequence[str],
    exact_features: Sequence[str],
    weighted: bool = False,
    allow_feature_unavailable: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for period, month_dates in _selection_periods(scheduled):
        first = period.start_time.normalize()
        training_all = _candidate_training(
            panel, end=first - pd.Timedelta(days=1)
        )
        scoring_all = _candidate_scoring(panel, month_dates)
        training_valid = _finite_feature_mask(training_all, exact_features)
        scoring_valid = _finite_feature_mask(scoring_all, exact_features)
        if not allow_feature_unavailable:
            if not training_valid.all() or not scoring_valid.all():
                raise ValueError(
                    f"v1.7 {candidate_id} exact features are incomplete on "
                    f"the common universe in {period}"
                )
            training = training_all
            scoring = scoring_all
            unavailable_dates = pd.DatetimeIndex([])
        else:
            # VP07's structurally invalid zero-range pressure rows and AT10's
            # intentionally longer rank-memory warm-up are never imputed or
            # proxied.  As registered, such rows are removed only from that
            # candidate's universe; every other finite row remains, while the
            # target ranks remain those frozen on the common universe above.
            training = training_all.loc[training_valid].copy()
            scoring = scoring_all.loc[scoring_valid].copy()
            available_dates = pd.DatetimeIndex(scoring["date"].drop_duplicates())
            unavailable_dates = month_dates.difference(available_dates)
            if training.empty:
                raise ValueError(f"v1.7 {candidate_id} has no observed training rows")
        weights = reliability_weights(training) if weighted else None
        model = fit_daily_rank_ridge(training, features, sample_weight=weights)
        if not scoring.empty:
            parts.append(
                _model_top_one(
                    scoring,
                    model.predict(scoring.loc[:, list(features)]),
                    month_dates,
                    candidate_id,
                )
            )
        else:
            parts.append(_empty_ledger(candidate_id, month_dates))
        folds.append(
            {
                "candidate_id": candidate_id,
                "period": str(period),
                "train_start": str(training["date"].min().date()),
                "train_end": str(training["date"].max().date()),
                "train_dates": int(training["date"].nunique()),
                "train_rows": int(len(training)),
                "score_rows": int(len(scoring)),
                "common_score_rows": int(len(scoring_all)),
                "feature_unavailable_rows": int((~scoring_valid).sum()),
                "candidate_specific_availability": allow_feature_unavailable,
                "cash_dates_for_feature_unavailability": [
                    str(date.date()) for date in unavailable_dates
                ],
                "strictly_prior_training": bool(training["date"].max() < first),
                "reliability_weighted": weighted,
                "replacement_model_used": False,
            }
        )
    return pd.concat(parts, ignore_index=True), folds


def _activity_regime_candidate(
    panel: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    candidate_id = "AR05_SECURITY_ACTIVITY_REGIME_EXPERTS"
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    state_column = "xrank_liq_turnover_shock1"
    for period, month_dates in _selection_periods(scheduled):
        first = period.start_time.normalize()
        training = _candidate_training(panel, end=first - pd.Timedelta(days=1))
        scoring = _candidate_scoring(panel, month_dates)
        scores = pd.Series(np.nan, index=scoring.index, dtype=float)
        regime_rows: dict[str, int] = {}
        if (
            not _finite_feature_mask(training, (state_column,)).all()
            or not _finite_feature_mask(scoring, (state_column,)).all()
        ):
            raise ValueError(
                f"v1.7 AR05 {period} registered activity state is incomplete"
            )
        for name, state in (
            ("positive", training[state_column].gt(0.0)),
            ("nonpositive", training[state_column].le(0.0)),
        ):
            expert_training = training.loc[state].copy()
            if expert_training.empty:
                raise ValueError(f"AR05 {period} {name} regime has no training rows")
            expert = fit_daily_rank_ridge(expert_training, G0_FEATURES)
            score_state = (
                scoring[state_column].gt(0.0)
                if name == "positive"
                else scoring[state_column].le(0.0)
            )
            if score_state.any():
                scores.loc[score_state] = expert.predict(
                    scoring.loc[score_state, list(G0_FEATURES)]
                )
            regime_rows[name] = int(len(expert_training))
        valid = scores.notna() & np.isfinite(scores)
        if not valid.all():
            raise ValueError(
                f"v1.7 AR05 {period} did not produce every routed score"
            )
        parts.append(
            _model_top_one(
                scoring,
                scores,
                month_dates,
                candidate_id,
            )
        )
        folds.append(
            {
                "candidate_id": candidate_id,
                "period": str(period),
                "train_start": str(training["date"].min().date()),
                "train_end": str(training["date"].max().date()),
                "train_rows_by_regime": regime_rows,
                "score_rows": int(valid.sum()),
                "strictly_prior_training": bool(training["date"].max() < first),
                "fit_succeeded": True,
                "cash_slot_days": 0,
                "replacement_model_used": False,
            }
        )
    return pd.concat(parts, ignore_index=True), folds


def build_score_ledger(
    panel: pd.DataFrame,
    sessions: Iterable[object],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    calendar = normalize_expected_sessions(sessions)
    scheduled = calendar[(calendar >= SELECTION_START) & (calendar <= SELECTION_END)]
    if len(scheduled) != SELECTION_SESSIONS:
        raise ValueError(
            f"v1.7 selection schedule changed: expected 79, got {len(scheduled)}"
        )
    crossfit_schedule = calendar[
        (calendar >= PAIR_CROSSFIT_START) & (calendar <= SELECTION_END)
    ]
    c00_scores, folds = _monthly_c00_scores(panel, crossfit_schedule)
    c00_top2 = _top_scored_rows(c00_scores, c00_scores["c00_model_score"], count=2)
    c00_top2 = c00_top2.loc[
        :, [*C00_OOF_SCORE_COLUMNS, "source_rank"]
    ].copy()
    outcomes = panel.loc[:, ["date", "code", "label", "oc_return_pct"]]
    pair_training_pool = c00_top2.merge(
        outcomes,
        on=["date", "code"],
        how="left",
        validate="many_to_one",
        sort=False,
    )

    selection_top2 = c00_top2[c00_top2["date"].isin(scheduled)].copy()
    c00_rank1 = selection_top2[selection_top2["source_rank"].eq(1)].copy()
    c00_rank1 = c00_rank1.rename(columns={"c00_model_score": "model_score"})
    c00_ledger = _complete_one_slot(c00_rank1, scheduled, C00)
    c00_rank2 = selection_top2[selection_top2["source_rank"].eq(2)].copy()
    c00_rank2 = c00_rank2.rename(columns={"c00_model_score": "model_score"})
    c02_ledger = _complete_one_slot(c00_rank2, scheduled, C02)

    pair00_parts: list[pd.DataFrame] = []
    pair01_parts: list[pd.DataFrame] = []
    for period, month_dates in _selection_periods(scheduled):
        first = period.start_time.normalize()
        current_top2 = c00_top2[c00_top2["date"].isin(month_dates)].copy()
        for candidate_id, feature_columns, seed, destination in (
            (PAIR00, PAIR00_COLUMNS, BOOTSTRAP_RANDOM_STATE, pair00_parts),
            (
                "PAIR01_TOP2_EXACT_LIQ_RERANK",
                PAIR01_COLUMNS,
                BOOTSTRAP_RANDOM_STATE + 1,
                pair01_parts,
            ),
        ):
            pair_counts = current_top2.groupby("date", sort=False).size().reindex(
                month_dates, fill_value=0
            )
            if pair_counts.gt(2).any():
                raise ValueError(
                    f"v1.7 {candidate_id} {period} has more than one frozen pair"
                )
            complete_pair_dates = pd.DatetimeIndex(
                pair_counts[pair_counts.eq(2)].index
            )
            current_complete_top2 = current_top2.loc[
                current_top2["date"].isin(complete_pair_dates)
            ].copy()
            registered_inputs = pd.concat(
                [
                    pair_training_pool.loc[
                        pair_training_pool["date"].lt(first)
                    ],
                    current_complete_top2,
                ],
                ignore_index=True,
            )
            if not _finite_feature_mask(registered_inputs, feature_columns).all():
                raise ValueError(
                    f"v1.7 {candidate_id} registered pair input is incomplete"
                )
            rows = build_pair_training_rows(
                pair_training_pool,
                before=first,
                feature_columns=feature_columns,
            )
            model = fit_pairwise_model(rows, feature_columns, random_state=seed)
            destination.append(
                choose_pairwise_top_one(
                    current_complete_top2,
                    model,
                    feature_columns,
                    month_dates,
                    candidate_id,
                )
            )
            historical_max = rows["date"].max()
            cash_dates = pd.DatetimeIndex(pair_counts[pair_counts.lt(2)].index)
            folds.append(
                {
                    "candidate_id": candidate_id,
                    "period": str(period),
                    "train_end": str(pd.Timestamp(historical_max).date()),
                    "train_pairs": int(len(rows) // 2),
                    "score_pairs": int(pair_counts.eq(2).sum()),
                    "strictly_prior_training": bool(historical_max < first),
                    "fit_succeeded": True,
                    "cash_slot_days": int(len(cash_dates)),
                    "cash_dates_for_missing_frozen_pair": [
                        str(date.date()) for date in cash_dates
                    ],
                    "replacement_model_used": False,
                }
            )

    model_parts: list[pd.DataFrame] = [
        c00_ledger,
        pd.concat(pair00_parts, ignore_index=True),
        c02_ledger,
        pd.concat(pair01_parts, ignore_index=True),
    ]
    candidate_specs = (
        (
            "TW02_TURNOVER_WEIGHTED_PRICE_MEMORY",
            (*G0_FEATURES, *TW02_FEATURES),
            TW02_FEATURES,
            False,
            False,
        ),
        (
            "VW03_AGGREGATE_COST_BASIS",
            (*G0_FEATURES, *VW03_FEATURES),
            VW03_FEATURES,
            False,
            False,
        ),
        (
            "PS06_PERSISTENT_VS_ISOLATED_ACTIVITY",
            (*G0_FEATURES, *PS06_FEATURES),
            PS06_FEATURES,
            False,
            False,
        ),
        (
            "VP07_VWAP_RANGE_PRESSURE_MEMORY",
            (*G0_FEATURES, *VP07_FEATURES),
            VP07_FEATURES,
            False,
            True,
        ),
        (
            "RW08_EXACT_LOT_RELIABILITY_WEIGHT",
            G0_FEATURES,
            ("rw08_traded_lots_med20",),
            True,
            False,
        ),
        (
            "MR09_TURNOVER_WEIGHTED_MARKET_REGIME",
            (*G0_FEATURES, *MR09_FEATURES),
            MR09_FEATURES,
            False,
            False,
        ),
        (
            "AT10_ATTENTION_MIGRATION",
            (*G0_FEATURES, *AT10_FEATURES),
            AT10_FEATURES,
            False,
            True,
        ),
    )
    for (
        candidate_id,
        features,
        exact_features,
        weighted,
        allow_feature_unavailable,
    ) in candidate_specs:
        ledger, candidate_folds = _ridge_candidate(
            panel,
            scheduled,
            candidate_id=candidate_id,
            features=features,
            exact_features=exact_features,
            weighted=weighted,
            allow_feature_unavailable=allow_feature_unavailable,
        )
        model_parts.append(ledger)
        folds.extend(candidate_folds)

    ar_ledger, ar_folds = _activity_regime_candidate(panel, scheduled)
    model_parts.append(ar_ledger)
    folds.extend(ar_folds)

    deterministic = (
        (
            "RPY04_RETURN_PER_TURNOVER_REVERSAL",
            "rpy04_xrank_return_per_turnover_sum5",
            -1.0,
        ),
    )
    for candidate_id, feature, direction in deterministic:
        scoring = _candidate_scoring(panel, scheduled)
        if not _finite_feature_mask(scoring, (feature,)).all():
            raise ValueError(
                f"v1.7 {candidate_id} exact feature is incomplete on the common universe"
            )
        model_parts.append(
            _model_top_one(
                scoring,
                direction * scoring[feature],
                scheduled,
                candidate_id,
            )
        )

    ledger = pd.concat(model_parts, ignore_index=True)
    ledger = ledger.sort_values(["candidate_id", "date"], kind="stable").reset_index(drop=True)
    if tuple(sorted(ledger["candidate_id"].unique())) != tuple(sorted(ALL_MODELS)):
        raise AssertionError("v1.7 score ledger model registry changed")
    if len(ledger) != len(ALL_MODELS) * len(scheduled):
        raise AssertionError("v1.7 score ledger is not one fixed slot per model/date")
    if {"label", "oc_return_pct"} & set(ledger.columns):
        raise AssertionError("v1.7 score ledger contains target outcomes")
    return ledger.loc[:, list(SCORE_LEDGER_COLUMNS)], folds


def semantic_score_hash(ledger: pd.DataFrame) -> str:
    canonical = ledger.loc[:, list(SCORE_LEDGER_COLUMNS)].copy()
    canonical["date"] = pd.to_datetime(canonical["date"]).dt.strftime("%Y-%m-%d")
    canonical = canonical.sort_values(["candidate_id", "date"], kind="stable")
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def attach_outcomes(score_ledger: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    if {"label", "oc_return_pct"} & set(score_ledger.columns):
        raise ValueError("v1.7 score ledger already contains outcomes")
    outcomes = panel.loc[:, ["date", "code", "label", "oc_return_pct"]].dropna(
        subset=["code"]
    )
    if outcomes[["date", "code"]].duplicated().any():
        raise ValueError("v1.7 outcome ledger contains duplicate date/code rows")
    picks = score_ledger.merge(
        outcomes,
        on=["date", "code"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    return picks.sort_values(["candidate_id", "date"], kind="stable").reset_index(drop=True)


def _daily_returns(
    picks: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    cost_bps: float,
) -> pd.Series:
    selected = picks.sort_values("date", kind="stable").set_index("date")
    executed = selected["label"].notna()
    values = selected["oc_return_pct"].fillna(0.0) - executed.astype(float) * cost_bps / 100.0
    values = values.reindex(scheduled)
    if len(values) != len(scheduled) or values.isna().any():
        raise AssertionError("v1.7 scheduled daily return series is incomplete")
    return values.astype(float)


def _top_codes_cash(
    picks: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    cost_bps: float,
    count: int,
) -> tuple[pd.Series, list[str]]:
    work = picks.copy()
    executed = work["label"].notna()
    work["_net"] = work["oc_return_pct"].fillna(0.0) - executed.astype(float) * cost_bps / 100.0
    by_code = (
        work.dropna(subset=["code"])
        .groupby("code", sort=False)["_net"]
        .sum()
        .sort_values(ascending=False, kind="stable")
    )
    codes = [str(code) for code in by_code.head(count).index]
    neutral = work.copy()
    removed = neutral["code"].astype(str).isin(codes)
    neutral["label"] = neutral["label"].astype("object")
    neutral.loc[removed, "label"] = pd.NA
    neutral.loc[removed, "oc_return_pct"] = np.nan
    return _daily_returns(neutral, scheduled, cost_bps=cost_bps), codes


def _variant_metrics(
    picks: pd.DataFrame,
    control: pd.DataFrame,
    scheduled: pd.DatetimeIndex,
    *,
    candidate_id: str,
    control_id: str,
    random_state: int,
) -> dict[str, Any]:
    cost_metrics = {
        str(int(cost)): profit_metrics(picks, top_k=1, cost_bps=cost)
        for cost in COSTS_BPS
    }
    daily40 = _daily_returns(picks, scheduled, cost_bps=PRIMARY_COST_BPS)
    daily60 = _daily_returns(picks, scheduled, cost_bps=60.0)
    control40 = _daily_returns(control, scheduled, cost_bps=PRIMARY_COST_BPS)
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
    slice_counts = {
        name: int(len(daily40.loc[start:end]))
        for name, (start, end) in SELECTION_SLICES.items()
    }
    if slice_counts != {"replay_a": 39, "replay_b": 40}:
        raise AssertionError(f"v1.7 fixed selection slice counts changed: {slice_counts}")
    top4_removed = float(daily40.drop(daily40.nlargest(4).index).mean())
    code_cash, top_codes = _top_codes_cash(
        picks, scheduled, cost_bps=PRIMARY_COST_BPS, count=5
    )
    monthly = daily40.groupby(daily40.index.to_period("M")).mean()
    signaled = picks.dropna(subset=["code"])
    counts = signaled["code"].astype(str).value_counts()
    total = int(len(signaled))
    unique_codes = int(len(counts))
    maximum_share = float(counts.iloc[0] / total) if total else 1.0
    top10_share = float(counts.head(10).sum() / total) if total else 1.0
    executed = picks["label"].notna()
    executed_days = int(executed.sum())
    executed_fraction = float(executed.mean())
    checks = {
        "net40_mean_positive": float(daily40.mean()) > 0.0,
        "net40_median_positive": float(daily40.median()) > 0.0,
        "net60_mean_positive": float(daily60.mean()) > 0.0,
        "both_fixed_slices_net40_positive": all(value > 0.0 for value in slices.values()),
        "positive_months_net40_at_least_3": int(monthly.gt(0.0).sum()) >= 3,
        "top4_days_removed_net40_positive": top4_removed > 0.0,
        "top5_profit_codes_cash_net40_positive": float(code_cash.mean()) > 0.0,
        "familywise_paired_lower_vs_control_nonnegative": paired.one_sided_lower_delta_pct >= 0.0,
        "unique_codes_at_least_40": unique_codes >= 40,
        "maximum_code_share_at_most_0_05": maximum_share <= 0.05,
        "top10_code_share_at_most_0_25": top10_share <= 0.25,
        "executed_days_at_least_72": executed_days >= 72,
        "executed_slot_fraction_at_least_0_8": executed_fraction >= 0.80,
    }
    return {
        "variant_id": candidate_id,
        "candidate_id": candidate_id,
        "matched_control_id": control_id,
        "capacity": 1,
        "cost_metrics": cost_metrics,
        "net40_mean_pct": float(daily40.mean()),
        "net40_median_pct": float(daily40.median()),
        "net60_mean_pct": float(daily60.mean()),
        "slice_net40_mean_pct": slices,
        "slice_session_counts": slice_counts,
        "monthly_net40_mean_pct": {str(key): float(value) for key, value in monthly.items()},
        "positive_months_net40": int(monthly.gt(0.0).sum()),
        "top4_days_removed_net40_mean_pct": top4_removed,
        "top5_profit_codes_cash_net40_mean_pct": float(code_cash.mean()),
        "top5_profit_codes": top_codes,
        "paired_vs_matched_control_net40": asdict(paired),
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
            -float(item["paired_vs_matched_control_net40"]["one_sided_lower_delta_pct"]),
            -float(item["top4_days_removed_net40_mean_pct"]),
            str(item["variant_id"]),
        ),
    )
    winner = ordered[0]
    return {
        "variant_id": winner["variant_id"],
        "candidate_id": winner["candidate_id"],
        "matched_control_id": winner["matched_control_id"],
        "selection_gate_passed": True,
        "selection_rule_rank": 1,
    }


def run_selection(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    price_warmup_directory: str | Path,
    daily_directory: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    prices, sessions, source_checks = load_prices(
        price_warmup_directory=price_warmup_directory,
        daily_directory=daily_directory,
    )
    panel, coverage, panel_checks = build_model_panel(prices, sessions)
    scheduled = sessions[(sessions >= SELECTION_START) & (sessions <= SELECTION_END)]
    score_ledger, folds = build_score_ledger(panel, sessions)
    score_hash = semantic_score_hash(score_ledger)
    picks = attach_outcomes(score_ledger, panel)
    picks_by_model = {
        model: picks[picks["candidate_id"].eq(model)].copy() for model in ALL_MODELS
    }
    control_metrics = {
        control: {
            str(int(cost)): profit_metrics(
                picks_by_model[control], top_k=1, cost_bps=cost
            )
            for cost in COSTS_BPS
        }
        for control in (C00, PAIR00)
    }
    diagnostic_metrics = {
        C02: {
            str(int(cost)): profit_metrics(
                picks_by_model[C02], top_k=1, cost_bps=cost
            )
            for cost in COSTS_BPS
        }
    }
    variants = [
        _variant_metrics(
            picks_by_model[candidate],
            picks_by_model[MATCHED_CONTROL[candidate]],
            scheduled,
            candidate_id=candidate,
            control_id=MATCHED_CONTROL[candidate],
            random_state=BOOTSTRAP_RANDOM_STATE + index,
        )
        for index, candidate in enumerate(CANDIDATES)
    ]
    winner = _choose_winner(variants)
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "runner_sha256": sha256_file(__file__),
        "status": (
            "selection_passed_one_research_nominee"
            if winner is not None
            else "selection_rejected_all_candidates"
        ),
        "authority": {
            "analysis_type": "retrospective_candidate_specific_selection",
            "project_level_untouched": False,
            "production_model_changed": False,
            "production_promotion_allowed": False,
            "orders_allowed": False,
        },
        "input": {
            **source_checks,
            **panel_checks,
            "selection_source_complete_sessions": int(
                coverage.loc[coverage["date"].isin(scheduled), "source_complete"].sum()
            ),
        },
        "selection": {
            "date_bounds": [str(scheduled.min().date()), str(scheduled.max().date())],
            "scheduled_sessions": len(scheduled),
            "models": list(ALL_MODELS),
            "candidates": list(CANDIDATES),
            "candidate_variants": FAMILY_SIZE,
            "matched_controls": MATCHED_CONTROL,
            "score_ledger_outcome_columns": [],
            "score_ledger_semantic_sha256": score_hash,
            "score_ledger_hashed_before_outcome_join": True,
            "fixed_slots_per_model_date": 1,
            "folds": folds,
            "control_metrics": control_metrics,
            "diagnostic_metrics": diagnostic_metrics,
            "variants": variants,
            "gate_passers": sum(item["gate_passed"] for item in variants),
            "winner": winner,
        },
        "decision": {
            "candidate_family_rejected": winner is None,
            "research_nominee": None if winner is None else winner["variant_id"],
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
    return result, score_ledger, picks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--price-warmup-directory", type=Path, required=True)
    parser.add_argument("--daily-directory", type=Path, required=True)
    parser.add_argument(
        "--score-output",
        type=Path,
        default=ROOT / "research/model_v17_liquidity_reliability_scores.csv",
    )
    parser.add_argument(
        "--picks-output",
        type=Path,
        default=ROOT / "research/model_v17_liquidity_reliability_picks.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research/model_v17_liquidity_reliability_result.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace existing score, picks, and result artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = (args.score_output, args.picks_output, args.output)
    resolved = [path.resolve() for path in output_paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("v1.7 output paths must be distinct")
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"v1.7 refuses to overwrite existing artifacts without --overwrite: {names}"
        )
    protocol, protocol_sha256 = validate_protocol(args.protocol)
    result, score_ledger, picks = run_selection(
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        price_warmup_directory=args.price_warmup_directory,
        daily_directory=args.daily_directory,
    )
    write_csv(score_ledger, args.score_output)
    # The semantic digest canonicalises row/date formatting; the byte digest
    # independently binds the persisted CSV representation.
    result["selection"]["score_ledger_file_sha256"] = sha256_file(
        args.score_output
    )
    write_csv(picks, args.picks_output)
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
