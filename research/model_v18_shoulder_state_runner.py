#!/usr/bin/env python3
"""Prospective v1.8 lagged monthly shoulder-state shadow runner.

The sole candidate, SH01, chooses only between the already-frozen C00 top two.
For target month M it computes a rank-1-minus-rank-2 return median separately
for M-3, M-2, and M-1 (at least ten complete pairs in every month), then freezes
the median of those three medians for all of M.  Positive selects rank 1,
negative selects rank 2, and exact zero or unavailable history selects cash.

Decisions and outcomes are separate append-only, one-record-per-session JSONL
chains.  A decision is sealed before its target-session outcome exists.  The
forward result remains sealed/awaiting until the deterministic terminal
month-end after at least 120 scheduled sessions and six represented months.

This module cannot promote production or enable orders.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from enum import Enum
import fcntl
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import ssl
import stat
import struct
import subprocess
import sys
import sysconfig
import tempfile
import time as time_module
from typing import Any
from urllib.parse import quote, urlparse
import urllib.error
import urllib.request
from zoneinfo import TZPATH, ZoneInfo

import numpy as np
import pandas as pd
import sklearn
import threadpoolctl


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research import model_v17_liquidity_reliability_runner as v17  # noqa: E402
from research.model_v13_symbolic_context_runner import G0_FEATURES  # noqa: E402
from tse_session_ranker.data.common import merge_daily_prices  # noqa: E402
from tse_session_ranker.data.jpx import collect_jpx  # noqa: E402
from tse_session_ranker.validation import paired_moving_block_bootstrap  # noqa: E402


PROTOCOL = ROOT / "research/model_v18_shoulder_state_protocol.json"
PROTOCOL_ID = "model_v18_shoulder_state_forward_20260804"
PROTOCOL_SHA256 = "930a82163f347aa7c303dfea1fb8b593ac6bff95ff774a2c6804c437d679cb5f"
CALENDAR = ROOT / "research/model_v18_tse_session_calendar.csv"
CALENDAR_SHA256 = "c5c5908b0e26ebd57eb2e473b9d4ce7f92a7c8b336f6ad70596152de971b77a7"
HYPOTHESIS_SHA256 = "7f750b0613252caa6d87bcff7c5b4a783cc33a09a0c5f7bcd6b3d4ed47630025"
RUNTIME_LOCK = ROOT / "research/model_v18_runtime_lock.json"
RUNTIME_LOCK_SHA256 = "2cd701e0ae5969b3a13236908e260c7f072344f5ca477287b14a3ed22b257a54"

ACTIVATION_PAYLOAD = ROOT / "research/model_v18_shoulder_state_activation_payload.json"
ACTIVATION_RECEIPT = ROOT / "research/model_v18_shoulder_state_activation_receipt.json"
ACTIVATION_CONTEXT = ROOT / "research/model_v18_shoulder_state_activation_context.json"
DECISION_LEDGER = ROOT / "research/model_v18_shoulder_state_decisions.jsonl"
OUTCOME_LEDGER = ROOT / "research/model_v18_shoulder_state_outcomes.jsonl"
COMPLETED_MONTH_LEDGER = ROOT / "research/model_v18_shoulder_state_months.jsonl"
DECISION_RECORD_DIR = ROOT / "research/model_v18_shoulder_state_decision_records"
OUTCOME_RECORD_DIR = ROOT / "research/model_v18_shoulder_state_outcome_records"
COMPLETED_MONTH_RECORD_DIR = (
    ROOT / "research/model_v18_shoulder_state_month_records"
)
SOURCE_MANIFEST_DIR = ROOT / "research/model_v18_shoulder_state_source_manifests"
MONTH_SOURCE_MANIFEST_DIR = (
    ROOT / "research/model_v18_shoulder_state_month_source_manifests"
)
OUTCOME_MANIFEST_DIR = ROOT / "research/model_v18_shoulder_state_outcome_manifests"
CHECKPOINT_PROPOSAL_DIR = ROOT / "research/model_v18_shoulder_state_checkpoint_proposals"
FOLD_MANIFEST_DIR = ROOT / "research/model_v18_shoulder_state_fold_manifests"
FOLD_MODEL_DIR = ROOT / "research/model_v18_shoulder_state_fold_models"
STATE_MANIFEST_DIR = ROOT / "research/model_v18_shoulder_state_state_manifests"
SCORE_SESSION_DIR = ROOT / "research/model_v18_shoulder_state_score_sessions"
SCORE_OUTPUT = ROOT / "research/model_v18_shoulder_state_scores.csv"
PICKS_OUTPUT = ROOT / "research/model_v18_shoulder_state_picks.csv"
RESULT_OUTPUT = ROOT / "research/model_v18_shoulder_state_result.json"

# The base v1.8 test retains the legacy tests_path/tests_sha256 payload fields.
# These additional paths are protocol-fixed in this exact order; their hashes
# are computed only after commit A's protocol and tests are final, then bound by
# payload B.  Keeping hashes out of the protocol avoids a protocol/test cycle.
ADDITIONAL_TEST_ARTIFACT_PATHS = (
    "tests/test_model_v18_a2_atomic_restart.py",
    "tests/test_model_v18_a2_core_integration.py",
    "tests/test_model_v18_a2_cross_role_anchor_abort.py",
    "tests/test_model_v18_a2_local_authority_recovery.py",
    "tests/test_model_v18_a2_nondiscretion_terminal_blind.py",
    "tests/test_model_v18_a2_p0_negative_contracts.py",
    "tests/test_model_v18_a2_preactivation_authority.py",
    "tests/test_model_v18_a2_record_authority.py",
    "tests/test_model_v18_a2_result_publication_retry.py",
    "tests/test_model_v18_a2_resume_authority.py",
    "tests/test_model_v18_a2_real_rehearsal.py",
    "tests/test_model_v18_a2_independent_audit.py",
    "tests/test_model_v18_operations.py",
)
ADDITIONAL_TEST_ARTIFACT_FIELDS = ("path", "sha256")

CANDIDATE_ID = "SH01_LAGGED_MONTHLY_SHOULDER_STATE"
C00_ID = "C00_PRICE_RIDGE_TOP1"
C02_ID = "C02_C00_RANK2"
STATE_MONTHS = 3
MIN_COMPLETE_PAIRS = 10
MIN_FORWARD_SESSIONS = 120
MIN_FORWARD_MONTHS = 6
COSTS_BPS = (20.0, 40.0, 60.0)
PRIMARY_COST_BPS = 40.0
BOOTSTRAP_BLOCK_LENGTH = 20
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_RANDOM_STATE = 20_260_805
BOOTSTRAP_CONFIDENCE = 0.90

TOKYO = ZoneInfo("Asia/Tokyo")
CUTOFF_TIME = time(8, 58, 59)
CANONICAL_JSON_CONTRACT = "project_canonical_json_v1"
RAW_SOURCE_PROVENANCE_MODE = "manual_operator_attested_v1"
RAW_SOURCE_PROVENANCE_CAVEAT_ID = "manual_jpx_origin_not_independently_verified_v1"
RAW_SOURCE_PROVENANCE_CAVEAT = (
    "The operator attests faithful manual acquisition of the labeled official JPX "
    "PDF without omission, substitution, or pre-seal alteration. The protocol "
    "proves post-seal bytes and computation only; it has no independent server "
    "receipt, availability, acquisition-time, origin, or authenticity proof."
)
RAW_SOURCE_PROVENANCE_CAVEAT_SHA256 = hashlib.sha256(
    RAW_SOURCE_PROVENANCE_CAVEAT.encode("utf-8")
).hexdigest()
ZERO_SHA256 = "0" * 64
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CODE_RE = re.compile(r"^[0-9A-Z]{4,5}$")
INTEGRITY_FAILURE_REASON_RE = re.compile(r"^[a-z][a-z0-9_]{2,127}$")
JPX_PARSER_PATH = "src/tse_session_ranker/data/jpx.py"
JPX_PARSER_VERSION = "jpx_daily_text_v6_special_quote_marker"
JPX_PARSER_SHA256 = "1bd2e74acced608eb36c3606b593ea407d2d1e5f54b3283790ef8fd0fb1041f7"
PREDICTOR_OBJECT_PREFIX = "model_v18_shoulder_state/predictor/"
OUTCOME_OBJECT_PREFIX = "model_v18_shoulder_state/outcome/"
PREDICTOR_SHARD_OBJECT_PREFIX = "model_v18_shoulder_state/predictor-shard/"
G0_PANEL_CACHE_OBJECT_PREFIX = "model_v18_shoulder_state/g0-panel/"
CACHE_ANCHOR_OBJECT_PREFIX = "model_v18_shoulder_state/cache-anchor/"
MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX = (
    "model_v18_shoulder_state/model-price-snapshot/"
)
FOLD_STAGE_OBJECT_PREFIX = "model_v18_shoulder_state/fold-stage/"
PREDICTOR_CACHE_CONTRACT_ID = "model_v18_predictor_cache_a2_v1"
PARSED_SHARD_JSONL_CONTRACT = (
    "canonical_json_array_rows_v1:utf8_no_bom_lf_final_lf;"
    "registered_column_order;stable_date_code;strict_types;finite_binary64;"
    "ieee_negative_zero_preserved"
)
G0_CACHE_JSONL_CONTRACT = PARSED_SHARD_JSONL_CONTRACT

CHECKPOINT_CORE_OBJECT_PREFIX = "model_v18_shoulder_state/checkpoint-core/"
CHECKPOINT_CORE_ENVELOPE_BYTES = 16_384
CHECKPOINT_CORE_MAGIC = b"TSEV18CP"
GITHUB_API_BASE = "https://api.github.com"
GITHUB_API_ACCEPT = "application/vnd.github+json"
GITHUB_API_VERSION = "2022-11-28"
GITHUB_API_USER_AGENT = "TSE-Session-Ranker-model-v18-activation"
GITHUB_REPOSITORY = "rokuroku-066/TSE-Session-Ranker"
GITHUB_BRANCH = "agent/v18-a2-shoulder-state-20260805"
GITHUB_REF = f"refs/heads/{GITHUB_BRANCH}"
WORKFLOW_PATH = ".github/workflows/tests.yml"
WORKFLOW_SHA256 = "7c9811768511bacb889e0f7bfa9508c2d2212079e8857db3b152d75868c232c7"
GITHUB_OBSERVATION_FIELDS = (
    "schema_version",
    "observation_kind",
    "endpoint",
    "repository",
    "branch",
    "requested_commit_sha",
    "http_status",
    "content_type",
    "http_date",
    "etag",
    "response_body_sha256",
    "response_headers_sha256",
    "canonical_projection",
    "retrieved_at",
    "canonical_json_contract",
    "observation_sha256",
)
GITHUB_PROJECTION_FIELDS = (
    "commit_sha",
    "html_url",
    "committer_date",
    "parent_shas",
)
GITHUB_WORKFLOW_OBSERVATION_FIELDS = (
    "schema_version",
    "endpoint",
    "repository",
    "run_id",
    "expected_head_sha",
    "http_status",
    "content_type",
    "http_date",
    "etag",
    "response_body_sha256",
    "response_headers_sha256",
    "canonical_projection",
    "retrieved_at",
    "canonical_json_contract",
    "observation_sha256",
)
GITHUB_WORKFLOW_PROJECTION_FIELDS = (
    "run_id",
    "workflow_id",
    "workflow_name",
    "workflow_path",
    "event",
    "head_sha",
    "run_attempt",
    "status",
    "conclusion",
    "created_at",
    "run_started_at",
    "updated_at",
    "html_url",
)
RECEIPT_OBSERVATION_RUNTIME_FIELDS = (
    "activation_receipt_commit_sha",
    "activation_receipt_commit_url",
    "activation_receipt_commit_committed_at",
    "activation_receipt_commit_observed_at",
    "branch_tip_sha_when_receipt_observed",
    "activation_receipt_file_sha256",
    "receipt_commit_observation",
    "receipt_branch_observation",
    "activation_receipt_workflow_run_id",
    "activation_receipt_workflow_run_updated_at",
    "activation_receipt_workflow_run_observed_at",
    "receipt_workflow_run_observation",
)
ABORT_INTEGRITY_STAGES = (
    "preregistration",
    "activation_payload",
    "activation_receipt",
    "receipt_workflow_observation",
    "pre_count_runtime",
    "source_ingestion",
    "monthly_fold",
    "state_construction",
    "checkpoint_seal",
    "decision_seal",
    "outcome_ingestion",
    "completed_month_close",
    "terminal_precondition",
    "terminal_reconstruction",
    "independent_audit",
)
ABORT_FAILURE_REASONS = (
    "protocol_mismatch",
    "runtime_lock_mismatch",
    "activation_evidence_invalid",
    "workflow_evidence_invalid",
    "source_integrity_failure",
    "fold_integrity_failure",
    "state_integrity_failure",
    "checkpoint_integrity_failure",
    "decision_integrity_failure",
    "outcome_integrity_failure",
    "completed_month_integrity_failure",
    "terminal_precondition_failure",
    "terminal_reconstruction_failure",
    "audit_integrity_failure",
    "canonical_path_conflict",
)
DAILY_FAILURE_REASONS = (
    "state_insufficient_prior_months",
    "state_value_exact_zero",
)
CHECKPOINT_ROLES = ("safety_cash", "primary")
CHECKPOINT_RESOLUTIONS = ("primary",)
CHECKPOINT_RESOLUTION_REASONS = ("primary_commitment_timely",)
CHECKPOINT_PROPOSAL_FIELDS = (
    "schema_version",
    "checkpoint_id",
    "checkpoint_batch_id",
    "publication_ordinal",
    "repository",
    "branch",
    "protocol_id",
    "protocol_sha256",
    "runner_sha256",
    "runtime_lock_sha256",
    "activation_payload_sha256",
    "activation_receipt_sha256",
    "activation_receipt_commit_sha",
    "target_session",
    "checkpoint_role",
    "decision_sequence_number",
    "previous_decision_record_sha256",
    "sealed_core_object_key",
    "sealed_core_byte_count",
    "sealed_core_sha256",
    "created_at",
    "canonical_json_contract",
    "proposal_sha256",
)
CHECKPOINT_CORE_FIELDS = (
    "candidate_id",
    "runtime_lock_verified_at",
    "source_manifest_sha256",
    "c00_fold_manifest_sha256",
    "fold_model_bundle_file_sha256",
    "state_manifest_sha256",
    "score_session_file_sha256",
    "score_session_semantic_sha256",
    "score_session_set_sha256",
    "decision_cutoff",
    "computed_at",
    "source_complete",
    "model_complete",
    "state_available",
    "three_prior_calendar_months",
    "three_complete_pair_day_counts",
    "three_month_medians_pct",
    "state_value_pct",
    "selected_source_rank",
    "c00_rank1_code",
    "c00_rank1_score",
    "c02_rank2_code",
    "c02_rank2_score",
    "candidate_selected_code",
    "decision",
    "failure_reason",
)
CHECKPOINT_BINDING_FIELDS = (
    "decision_materialized_at",
    "checkpoint_resolution",
    "checkpoint_resolution_reason",
    "checkpoint_core_sha256",
    "checkpoint_proposal_path",
    "checkpoint_proposal_file_sha256",
    "checkpoint_proposal_sha256",
    "checkpoint_commit_sha",
    "checkpoint_commit_url",
    "checkpoint_commit_committed_at",
    "checkpoint_commit_observed_at",
    "checkpoint_branch_tip_sha_when_observed",
    "checkpoint_commit_observation",
    "checkpoint_branch_observation",
    "checkpoint_workflow_run_id",
    "checkpoint_workflow_run_updated_at",
    "checkpoint_workflow_run_observed_at",
    "checkpoint_workflow_run_observation",
)
GIT_DATA_OPERATION_IDS = (
    "get_ref",
    "get_parent_commit",
    "get_parent_tree",
    "create_blob",
    "create_tree",
    "create_commit",
    "update_ref",
    "verify_ref",
    "verify_commit",
    "verify_tree",
    "verify_blob",
    "verify_compare",
)
GIT_DATA_OPERATION_METHODS = (
    "GET",
    "GET",
    "GET",
    "POST",
    "POST",
    "POST",
    "PATCH",
    "GET",
    "GET",
    "GET",
    "GET",
    "GET",
)
GIT_DATA_OPERATION_STATUSES = (200, 200, 200, 201, 201, 201, 200, 200, 200, 200, 200, 200)

# Operational CLI entry points set this only after an exact runtime-lock
# preflight.  Pure state/unit APIs remain usable in an isolated test runtime,
# while every real numeric operation also checks the registered native backend
# set inside its one-thread execution context.
_STRICT_RUNTIME_ACTIVE = False
_STRICT_RUNTIME_VERIFIED_AT: datetime | None = None
_LOCKED_GIT_EXECUTABLE: Path | None = None
_LOCKED_PDFTOTEXT_EXECUTABLE: Path | None = None
_LOCKED_TLS_CAFILE: Path | None = None
_LIVE_MODULE_CLOSURE_STATE: dict[str, Any] | None = None


A2_REHEARSAL_INPUT_KINDS = (
    "full31_reference",
    "month_boundary_compact",
    "intramonth_fold_reuse_upper_bound_proxy",
)


@dataclass(frozen=True)
class _A2RehearsalContract:
    """Opaque, process-local capability for the nonauthority timing seam."""

    nonce: str
    protocol_sha256: str
    runner_sha256: str
    runtime_lock_sha256: str
    calendar_sha256: str
    c00_contract_json: str
    fold_manifest_required_fields: tuple[str, ...]
    fold_model_bundle_required_fields: tuple[str, ...]
    runtime_python_version: str
    numpy_version: str
    pandas_version: str
    scikit_learn_version: str
    numeric_threadpool_limit: int
    target_session: str
    latest_required_source_session: str
    preflight_sha256: str


@dataclass(frozen=True)
class _A2RehearsalFoldToken:
    """In-memory-only fold reuse token; never a serialised authority."""

    fold_manifest: dict[str, Any]
    model_bundle: dict[str, Any]
    month_source_manifest: dict[str, Any]
    exact_prefix_proof_json: str
    exact_prefix_proof_sha256: str


_ACTIVE_A2_REHEARSAL_CONTRACT: _A2RehearsalContract | None = None

# collect_jpx sorts the registered external paths before concatenation; under
# the fixed predictor/{daily|price_warmup} layout the daily projection is the
# bound stable order below.  Hashing a caller-selected subset (or an extra
# convenience column) would no longer be a clean-room parser binding.
PARSED_PRICE_COLUMNS = (
    "date",
    "code",
    "name",
    "raw_name",
    "trading_unit",
    "final_special_quote",
    "net_change",
    "vwap",
    "volume",
    "turnover",
    "volume_unit",
    "turnover_unit",
    "source_volume_unit",
    "source_turnover_unit",
    "source_file",
    "source_line",
    "source_format",
    "open",
    "high",
    "low",
    "close",
    "am_open",
    "am_high",
    "am_low",
    "am_close",
    "pm_open",
    "pm_high",
    "pm_low",
    "pm_close",
    "traded",
    "partial_session",
)

PARSED_SHARD_BINDING_FIELDS = (
    "raw_object_key",
    "raw_file",
    "raw_url",
    "raw_byte_count",
    "raw_sha256",
    "shard_manifest_object_key",
    "shard_manifest_byte_count",
    "shard_manifest_file_sha256",
    "shard_manifest_sha256",
    "shard_object_key",
    "shard_byte_count",
    "shard_sha256",
    "parsed_row_count",
    "parsed_semantic_sha256",
    "pdftotext_text_byte_count",
    "pdftotext_text_sha256",
    "parser_report_sha256",
)

PARSED_SHARD_MANIFEST_FIELDS = (
    "schema_version",
    "cache_contract_id",
    "official_source_file_name",
    "official_source_url",
    "raw_byte_count",
    "raw_sha256",
    "chronology_class",
    "raw_received_at",
    "parser_path",
    "parser_version",
    "parser_sha256",
    "pdftotext_file_sha256",
    "pdftotext_elf_closure_sha256",
    "pdftotext_argv_environment_contract_sha256",
    "runtime_lock_sha256",
    "runtime_lock_verified_at",
    "created_at",
    "sealed_at",
    "data_object_key",
    "data_byte_count",
    "data_sha256",
    "columns",
    "columns_sha256",
    "row_count",
    "rejected_row_count",
    "duplicate_date_code_count",
    "unique_date_count",
    "min_date",
    "max_date",
    "pdftotext_text_byte_count",
    "pdftotext_text_sha256",
    "parser_report_sha256",
    "parsed_semantic_sha256",
    "canonical_jsonl_contract",
    "manifest_sha256",
)

G0_PANEL_COLUMNS = (
    "date",
    "code",
    "name",
    "oc_return_pct",
    "common_training_eligible",
    "common_score_eligible",
    "feature_source_max_date",
    *G0_FEATURES,
)

# Exact minimal projection consumed by v17.build_model_panel and its v16/G0
# callees.  normalize_daily_prices deterministically reconstructs the unused
# session/trade state from OHLC.  Activation additionally requires exact
# real-corpus equality against all 31 parsed columns.
MODEL_PRICE_COLUMNS = (
    "date",
    "code",
    "name",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "vwap",
    "trading_unit",
    "source_format",
)
MODEL_PRICE_REQUIRED_BY_CONSUMERS = {
    "build_forward_c00_panel": {"date", "code", "name"},
    "normalize_daily_prices": {
        "date", "code", "open", "high", "low", "close",
    },
    "v16.build_exact_liquidity_features": {
        "date", "code", "name", "close", "volume", "turnover", "vwap",
        "trading_unit", "source_format",
    },
    "v17.build_reliability_features": {
        "date", "code", "open", "high", "low", "close", "volume",
        "turnover", "vwap", "trading_unit", "source_format",
    },
}
if set().union(*MODEL_PRICE_REQUIRED_BY_CONSUMERS.values()) != set(
    MODEL_PRICE_COLUMNS
):  # pragma: no cover - import-time invariant
    raise RuntimeError("model-price projection no longer equals its consumer union")

MODEL_PRICE_CSV_CONTRACT = {
    "format": "RFC4180-compatible canonical CSV",
    "encoding": "UTF-8 without BOM",
    "line_ending": "LF",
    "final_lf": True,
    "columns": list(MODEL_PRICE_COLUMNS),
    "row_order": ["date", "code"],
    "null": "empty field only in numeric columns",
    "float": "finite IEEE-754 round-trip text preserving negative zero",
    "reader": "pandas.read_csv(float_precision=round_trip)",
}

MODEL_PRICE_SNAPSHOT_FIELDS = (
    "schema_version",
    "cache_contract_id",
    "target_month",
    "latest_source_session",
    "raw_source_set_sha256",
    "parsed_shard_set_sha256",
    "raw_source_count",
    "parsed_row_count",
    "columns",
    "columns_sha256",
    "data_object_key",
    "data_byte_count",
    "data_sha256",
    "row_count",
    "unique_date_count",
    "duplicate_date_code_count",
    "model_price_semantic_sha256",
    "previous_snapshot_manifest_sha256",
    "previous_snapshot_target_month",
    "previous_snapshot_latest_source_session",
    "previous_snapshot_manifest_object_key",
    "previous_snapshot_manifest_byte_count",
    "previous_snapshot_manifest_file_sha256",
    "previous_snapshot_raw_source_count",
    "previous_snapshot_raw_source_set_sha256",
    "previous_snapshot_parsed_shard_set_sha256",
    "runtime_lock_sha256",
    "runtime_lock_verified_at",
    "protocol_sha256",
    "runner_sha256",
    "parser_sha256",
    "created_at",
    "sealed_at",
    "canonical_csv_contract",
    "snapshot_manifest_sha256",
)

G0_CACHE_MANIFEST_FIELDS = (
    "schema_version",
    "cache_contract_id",
    "scope",
    "target_session",
    "latest_required_source_session",
    "source_set_sha256",
    "parsed_shard_set_sha256",
    "parsed_row_count",
    "columns",
    "columns_sha256",
    "data_object_key",
    "data_byte_count",
    "data_sha256",
    "row_count",
    "unique_date_count",
    "duplicate_date_code_count",
    "data_semantic_sha256",
    "target_row_count",
    "target_date_scoring_input_semantic_sha256",
    "target_slice_semantic_sha256",
    "target_outcome_nonnull_count",
    "max_feature_source_date",
    "v17_protocol_sha256",
    "v17_runner_sha256",
    "runtime_lock_sha256",
    "runtime_lock_verified_at",
    "protocol_sha256",
    "runner_sha256",
    "parser_sha256",
    "created_at",
    "sealed_at",
    "canonical_jsonl_contract",
    "cache_manifest_sha256",
)

CACHE_ANCHOR_FIELDS = (
    "schema_version",
    "cache_contract_id",
    "latest_source_session",
    "raw_source_set_sha256",
    "raw_source_count",
    "raw_sources",
    "ordered_shard_set_sha256",
    "ordered_shard_count",
    "parsed_shards",
    "columns",
    "columns_sha256",
    "cumulative_snapshot_object_key",
    "cumulative_snapshot_byte_count",
    "cumulative_snapshot_file_sha256",
    "cumulative_snapshot_semantic_sha256",
    "model_price_snapshot_target_month",
    "model_price_snapshot_object_key",
    "model_price_snapshot_byte_count",
    "model_price_snapshot_file_sha256",
    "model_price_snapshot_semantic_sha256",
    "model_price_snapshot_manifest_object_key",
    "model_price_snapshot_manifest_byte_count",
    "model_price_snapshot_manifest_file_sha256",
    "model_price_snapshot_manifest_sha256",
    "snapshot_manifest_object_key",
    "snapshot_manifest_sha256",
    "direct_reparse_started_at",
    "direct_reparse_completed_at",
    "direct_clean_room_verification_receipt_sha256",
    "compact_consumer_equivalence_receipt",
    "runtime_lock_sha256",
    "runtime_lock_verified_at",
    "created_at",
    "sealed_at",
    "protocol_sha256",
    "runner_sha256",
    "parser_sha256",
    "canonical_jsonl_contract",
    "verified_at",
    "anchor_manifest_sha256",
)

CACHE_ANCHOR_SUMMARY_FIELDS = (
    "latest_source_session",
    "raw_source_set_sha256",
    "raw_source_count",
    "ordered_shard_set_sha256",
    "ordered_shard_count",
    "cumulative_snapshot_object_key",
    "cumulative_snapshot_byte_count",
    "cumulative_snapshot_file_sha256",
    "cumulative_snapshot_semantic_sha256",
    "model_price_snapshot_target_month",
    "model_price_snapshot_object_key",
    "model_price_snapshot_byte_count",
    "model_price_snapshot_file_sha256",
    "model_price_snapshot_semantic_sha256",
    "model_price_snapshot_manifest_object_key",
    "model_price_snapshot_manifest_byte_count",
    "model_price_snapshot_manifest_file_sha256",
    "model_price_snapshot_manifest_sha256",
    "snapshot_manifest_object_key",
    "snapshot_manifest_file_sha256",
    "snapshot_manifest_sha256",
    "direct_clean_room_verification_receipt_sha256",
    "compact_consumer_equivalence_receipt_sha256",
    "sealed_at",
    "verified_at",
)

COMPACT_CONSUMER_EQUIVALENCE_RECEIPT_FIELDS = (
    "schema_version",
    "consumer_projection_columns",
    "consumer_projection_columns_sha256",
    "consumer_union_contract_sha256",
    "raw_date_code_identity_sha256",
    "historical_session_registry_sha256",
    "synthetic_target_session",
    "latest_feature_source_session",
    "full_g0_exact_digest",
    "compact_g0_exact_digest",
    "synthetic_target_exact_digest_sha256",
    "synthetic_target_row_count",
    "synthetic_target_outcome_nonnull_count",
    "synthetic_target_max_feature_source_date",
    "exact_columns_order_dtypes_nulls_ieee_strings_bools_equal",
    "canonical_json_contract",
    "receipt_sha256",
)

V16_BINDINGS = {
    "protocol": (
        "research/model_v16_liquidity_protocol.json",
        "f7b1329959b237323d6aa08d87494c9523bb5bdd4e29ee8c30cd04ece9e4387e",
    ),
    "runner": (
        "research/model_v16_liquidity_runner.py",
        "de5b0a6382f53ce4d513a187b4d2243149e60b8b9a7a69043e9b74d12171a537",
    ),
    "result": (
        "research/model_v16_liquidity_result.json",
        "7f8aff8b1e86240c00de3b124e9564ff5b3a0277f91a2cb9b27ef859207f4008",
    ),
    "picks": (
        "research/model_v16_liquidity_picks.csv",
        "65fb583501e30658273cb3d98725e2e4d142fa5681d3ed82023da597ddaf4b71",
    ),
}
V17_BINDINGS = {
    "protocol": (
        "research/model_v17_liquidity_reliability_protocol.json",
        "f7d2efa30c5f6ca03a95e1f6e84e0fb6de3f68e8f0d520183877e2f0ab4a416f",
    ),
    "runner": (
        "research/model_v17_liquidity_reliability_runner.py",
        "6394161d70d8ca76862ee34bdaa3a0aeb95adf57c3dd3e6685fc65d456980807",
    ),
    "result": (
        "research/model_v17_liquidity_reliability_result.json",
        "1463ae399c5de8ca33e762303d1ba3322ce21a383d2ad81f202a063bb8c148dd",
    ),
    "scores": (
        "research/model_v17_liquidity_reliability_scores.csv",
        "8e2e8d0fc4fcdbb716ec1de0cafba2b6300015b93f93020d6309987018e18244",
    ),
    "picks": (
        "research/model_v17_liquidity_reliability_picks.csv",
        "32de442e2012d901963399f9fd91fc69fb3082e93f0cfa7981c279b2b7467273",
    ),
    "audit": (
        "research/model_v17_liquidity_reliability_audit.json",
        "41a626f96dff850aec581838cd698695d541e6bb4e9882b5f1ce27fc68218dec",
    ),
}

FORBIDDEN_DECISION_COLUMNS = frozenset(
    {
        "label",
        "oc_return_pct",
        "open_to_close_return_pct",
        "open",
        "high",
        "low",
        "close",
        "outcome_observed",
        "rank1_oc_return_pct",
        "rank2_oc_return_pct",
        "candidate_outcome_observed",
        "candidate_gross_return_pct",
        "candidate_net20_return_pct",
        "candidate_net40_return_pct",
        "candidate_net60_return_pct",
        "gross_return_pct",
        "net_return_pct",
    }
)

DECISION_FIELDS = (
    "schema_version",
    "sequence_number",
    "session_date",
    "protocol_id",
    "protocol_sha256",
    "runner_sha256",
    "runtime_lock_sha256",
    "runtime_lock_verified_at",
    "candidate_id",
    "activation_payload_sha256",
    "activation_receipt_sha256",
    "activation_receipt_commit_sha",
    "activation_receipt_commit_url",
    "activation_receipt_commit_committed_at",
    "activation_receipt_commit_observed_at",
    "branch_tip_sha_when_receipt_observed",
    "activation_receipt_file_sha256",
    "receipt_commit_observation",
    "receipt_branch_observation",
    "activation_receipt_workflow_run_id",
    "activation_receipt_workflow_run_updated_at",
    "activation_receipt_workflow_run_observed_at",
    "receipt_workflow_run_observation",
    *CHECKPOINT_BINDING_FIELDS,
    "source_manifest_sha256",
    "c00_fold_manifest_sha256",
    "fold_model_bundle_file_sha256",
    "state_manifest_sha256",
    "score_session_file_sha256",
    "score_session_semantic_sha256",
    "score_session_set_sha256",
    "decision_cutoff",
    "computed_at",
    "source_complete",
    "model_complete",
    "state_available",
    "three_prior_calendar_months",
    "three_complete_pair_day_counts",
    "three_month_medians_pct",
    "state_value_pct",
    "selected_source_rank",
    "c00_rank1_code",
    "c00_rank1_score",
    "c02_rank2_code",
    "c02_rank2_score",
    "candidate_selected_code",
    "decision",
    "failure_reason",
    "previous_record_sha256",
    "record_sha256",
)

OUTCOME_FIELDS = (
    "schema_version",
    "sequence_number",
    "session_date",
    "decision_record_sha256",
    "outcome_source_sha256",
    "outcome_manifest_sha256",
    "outcome_received_at",
    "computed_at",
    "rank1_outcome_observed",
    "rank1_oc_return_pct",
    "rank2_outcome_observed",
    "rank2_oc_return_pct",
    "candidate_outcome_observed",
    "candidate_gross_return_pct",
    "candidate_net20_return_pct",
    "candidate_net40_return_pct",
    "candidate_net60_return_pct",
    "c00_top1_gross_return_pct",
    "c00_top1_net20_return_pct",
    "c00_top1_net40_return_pct",
    "c00_top1_net60_return_pct",
    "c02_rank2_gross_return_pct",
    "c02_rank2_net20_return_pct",
    "c02_rank2_net40_return_pct",
    "c02_rank2_net60_return_pct",
    "protocol_sha256",
    "activation_payload_sha256",
    "activation_receipt_sha256",
    "previous_record_sha256",
    "record_sha256",
)


class V18Error(ValueError):
    """Fail-closed v1.8 contract error."""


class ShoulderDecision(str, Enum):
    RANK1 = "rank1"
    RANK2 = "rank2"
    CASH = "cash"


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if _is_registered_local_authority_path(candidate):
        return hashlib.sha256(
            _read_local_authority_bytes(
                candidate, label=f"SHA authority {candidate}"
            )
        ).hexdigest()
    try:
        require_single_link = candidate.resolve(strict=True).is_relative_to(ROOT.resolve())
    except (FileNotFoundError, OSError):
        require_single_link = False
    return hashlib.sha256(
        _plain_file_bytes(
            path,
            label=f"SHA authority {candidate}",
            require_single_link=require_single_link,
        )
    ).hexdigest()


def _registered_local_authority_directories() -> tuple[Path, ...]:
    """Return local create-once authority parents using live patched constants."""

    return (
        SOURCE_MANIFEST_DIR,
        MONTH_SOURCE_MANIFEST_DIR,
        OUTCOME_MANIFEST_DIR,
        FOLD_MANIFEST_DIR,
        FOLD_MODEL_DIR,
        STATE_MANIFEST_DIR,
        SCORE_SESSION_DIR,
        DECISION_RECORD_DIR,
        OUTCOME_RECORD_DIR,
        COMPLETED_MONTH_RECORD_DIR,
    )


def _registered_local_authority_files() -> tuple[Path, ...]:
    """Return create-once authorities whose parent also holds nonauthority files."""

    return (
        ACTIVATION_PAYLOAD,
        ACTIVATION_RECEIPT,
        ACTIVATION_CONTEXT,
        RESULT_OUTPUT,
    )


def _is_registered_local_authority_directory(path: str | Path) -> bool:
    candidate = Path(os.path.abspath(os.fspath(path)))
    return any(
        candidate == Path(os.path.abspath(os.fspath(directory)))
        for directory in _registered_local_authority_directories()
    )


def _is_registered_local_authority_path(path: str | Path) -> bool:
    exact = Path(os.path.abspath(os.fspath(Path(path))))
    candidate = exact.parent
    return exact in {
        Path(os.path.abspath(os.fspath(authority)))
        for authority in _registered_local_authority_files()
    } or _is_registered_local_authority_directory(candidate)


def _read_local_authority_bytes(path: str | Path, *, label: str) -> bytes:
    """Read one private local authority and heal only its proven link crash.

    The only mutation is removal of the canonical staging name when final and
    stage are both private names for the same pinned two-link inode.  A
    distinct/unpublished stage is never interpreted as authority.
    """

    target = Path(path)
    parent_path = Path(os.path.abspath(os.fspath(target.parent)))
    try:
        resolved_parent = parent_path.resolve(strict=True)
    except OSError as exc:
        raise V18Error(f"{label} parent cannot be resolved") from exc
    if resolved_parent != parent_path or target.parent.is_symlink():
        raise V18Error(f"{label} parent must not use a symlink")
    file_name = target.name
    if file_name in {"", ".", ".."} or "/" in file_name or "\x00" in file_name:
        raise V18Error(f"{label} name is not a plain component")
    stage_name = f".{file_name}.staging"
    exact_target = Path(os.path.abspath(os.fspath(target)))
    git_activation_files = {
        Path(os.path.abspath(os.fspath(ACTIVATION_PAYLOAD))),
        Path(os.path.abspath(os.fspath(ACTIVATION_RECEIPT))),
    }
    completed_final_modes = (
        {0o600, 0o644} if exact_target in git_activation_files else {0o600}
    )
    parent_fd = os.open(
        resolved_parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
    )
    fcntl.flock(parent_fd, fcntl.LOCK_EX)
    try:
        parent_stat = os.fstat(parent_fd)
        registered_private_parent = _is_registered_local_authority_directory(
            resolved_parent
        )
        if (
            not stat.S_ISDIR(parent_stat.st_mode)
            or parent_stat.st_uid != os.geteuid()
            or (
                stat.S_IMODE(parent_stat.st_mode) != 0o700
                if registered_private_parent
                else bool(stat.S_IMODE(parent_stat.st_mode) & 0o022)
            )
        ):
            raise V18Error(f"{label} parent owner/permissions are not private")

        def observed(name: str) -> os.stat_result | None:
            try:
                return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return None

        def private_regular(
            value: os.stat_result | None, *, nlink: int, modes: set[int]
        ) -> bool:
            return bool(
                value is not None
                and stat.S_ISREG(value.st_mode)
                and value.st_uid == os.geteuid()
                and stat.S_IMODE(value.st_mode) in modes
                and value.st_nlink == nlink
            )

        final_stat = observed(file_name)
        if final_stat is None:
            raise FileNotFoundError(os.fspath(target))
        stage_stat = observed(stage_name)
        if stage_stat is not None:
            if (
                not private_regular(final_stat, nlink=2, modes={0o600})
                or not private_regular(stage_stat, nlink=2, modes={0o600})
                or (stage_stat.st_dev, stage_stat.st_ino)
                != (final_stat.st_dev, final_stat.st_ino)
            ):
                raise V18Error(f"{label} has a conflicting crash stage")
        elif not private_regular(
            final_stat, nlink=1, modes=completed_final_modes
        ):
            raise V18Error(f"{label} owner/mode/link changed")

        descriptor = os.open(
            stage_name if stage_stat is not None else file_name,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            before = os.fstat(descriptor)
            expected_links = 2 if stage_stat is not None else 1
            expected_modes = {0o600} if stage_stat is not None else completed_final_modes
            if not private_regular(
                before, nlink=expected_links, modes=expected_modes
            ):
                raise V18Error(f"{label} pinned inode is not private")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if (
                not private_regular(after, nlink=expected_links, modes=expected_modes)
                or (
                    after.st_dev,
                    after.st_ino,
                    after.st_nlink,
                    after.st_size,
                    after.st_uid,
                    stat.S_IMODE(after.st_mode),
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
                != (
                    before.st_dev,
                    before.st_ino,
                    before.st_nlink,
                    before.st_size,
                    before.st_uid,
                    stat.S_IMODE(before.st_mode),
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
            ):
                raise V18Error(f"{label} changed while being read")
            payload = b"".join(chunks)
            if len(payload) != before.st_size:
                raise V18Error(f"{label} changed length while being read")
            current_final = observed(file_name)
            if (
                not private_regular(
                    current_final, nlink=expected_links, modes=expected_modes
                )
                or (
                    current_final.st_dev,
                    current_final.st_ino,
                    current_final.st_size,
                    current_final.st_mtime_ns,
                    current_final.st_ctime_ns,
                )
                != (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
            ):
                raise V18Error(f"{label} final changed while pinned")
            if stage_stat is not None:
                current_stage = observed(stage_name)
                if (
                    not private_regular(current_stage, nlink=2, modes={0o600})
                    or (
                        current_stage.st_dev,
                        current_stage.st_ino,
                        current_stage.st_size,
                        current_stage.st_mtime_ns,
                        current_stage.st_ctime_ns,
                    )
                    != (
                        before.st_dev,
                        before.st_ino,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    )
                ):
                    raise V18Error(f"{label} stage changed before recovery")
                os.unlink(stage_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
                recovered = observed(file_name)
                if (
                    not private_regular(recovered, nlink=1, modes={0o600})
                    or (recovered.st_dev, recovered.st_ino)
                    != (before.st_dev, before.st_ino)
                ):
                    raise V18Error(f"{label} final changed after recovery")
            return payload
        finally:
            os.close(descriptor)
    finally:
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(parent_fd)


def _local_authority_presence_state(path: str | Path, *, label: str) -> str:
    """Classify a final authority without opening or healing its content."""

    target = Path(path)
    exact_target = Path(os.path.abspath(os.fspath(target)))
    git_activation_files = {
        Path(os.path.abspath(os.fspath(ACTIVATION_PAYLOAD))),
        Path(os.path.abspath(os.fspath(ACTIVATION_RECEIPT))),
    }
    completed_final_modes = (
        {0o600, 0o644} if exact_target in git_activation_files else {0o600}
    )
    try:
        final = os.stat(target, follow_symlinks=False)
    except FileNotFoundError:
        return "absent"
    parent = Path(os.path.abspath(os.fspath(target.parent)))
    if parent.resolve(strict=True) != parent or target.parent.is_symlink():
        raise V18Error(f"{label} parent must not use a symlink")
    parent_stat = os.stat(parent, follow_symlinks=False)
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != os.geteuid()
        or stat.S_IMODE(parent_stat.st_mode) & 0o022
    ):
        raise V18Error(f"{label} parent owner/permissions are not private")
    if (
        not stat.S_ISREG(final.st_mode)
        or final.st_uid != os.geteuid()
        or stat.S_IMODE(final.st_mode) not in completed_final_modes
    ):
        raise V18Error(f"{label} final owner/type/mode changed")
    stage = target.parent / f".{target.name}.staging"
    try:
        staged = os.stat(stage, follow_symlinks=False)
    except FileNotFoundError:
        staged = None
    if staged is None:
        if final.st_nlink != 1:
            raise V18Error(f"{label} final has an unexplained hard-link alias")
        return "completed"
    if (
        not stat.S_ISREG(staged.st_mode)
        or staged.st_uid != os.geteuid()
        or stat.S_IMODE(staged.st_mode) != 0o600
        or stat.S_IMODE(final.st_mode) != 0o600
        or (staged.st_dev, staged.st_ino, staged.st_nlink)
        != (final.st_dev, final.st_ino, 2)
        or final.st_nlink != 2
    ):
        raise V18Error(f"{label} final/stage crash marker is inconsistent")
    return "crash_linked"


def _result_status_token_without_performance_read(
    path: str | Path, *, presence: str
) -> str:
    """Read only the final sorted-JSON status line from a retained result.

    ``write_json`` sorts top-level keys, so ``status`` is the final member of
    every canonical result.  This classifier deliberately walks backward only
    across that one line and the closing brace.  It therefore distinguishes an
    irreversible integrity-abort result before any performance-bearing result
    member, ledger, cache, checkpoint, or network authority is opened.
    """

    if presence not in {"completed", "crash_linked"}:
        raise V18Error("retained result status requires an existing authority")
    target = Path(path)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise V18Error("canonical result could not be pinned for status") from exc
    try:
        before = os.fstat(descriptor)
        expected_links = 1 if presence == "completed" else 2
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != expected_links
            or before.st_size < 5
        ):
            raise V18Error("canonical result status authority metadata changed")
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        if os.pread(descriptor, 3, before.st_size - 3) != b"\n}\n":
            raise V18Error("canonical result status suffix is not canonical")
        cursor = before.st_size - 4
        reversed_line = bytearray()
        while cursor >= 0:
            observed = os.pread(descriptor, 1, cursor)
            if len(observed) != 1:
                raise V18Error("canonical result status read was truncated")
            if observed == b"\n":
                break
            reversed_line.extend(observed)
            if len(reversed_line) > 96:
                raise V18Error("canonical result status line is too long")
            cursor -= 1
        if cursor < 0:
            raise V18Error("canonical result status line is not delimited")
        line = bytes(reversed(reversed_line))
        match = re.fullmatch(rb'  "status": "([a-z0-9_]+)"', line)
        if match is None:
            raise V18Error("canonical result status line is malformed")
        after = os.fstat(descriptor)
        if any(
            getattr(before, field) != getattr(after, field)
            for field in (
                "st_dev",
                "st_ino",
                "st_uid",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        ):
            raise V18Error("canonical result changed during status classification")
        status_token = match.group(1).decode("ascii")
        if status_token not in {
            "aborted_integrity_failure",
            "forward_passed_one_v19_research_nominee",
            "forward_rejected_candidate",
        }:
            raise V18Error("canonical result status is not registered")
        return status_token
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        label = f"JSON artifact {Path(path)}"
        payload = (
            _read_local_authority_bytes(path, label=label)
            if _is_registered_local_authority_path(path)
            else _plain_file_bytes(path, label=label)
        )
        value = json.loads(payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise V18Error(f"{path} is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise V18Error(f"{path} must contain a JSON object")
    return value


def _safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise V18Error("canonical datetime must be timezone-aware")
        return value.isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, np.generic):
        return _safe(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise V18Error("canonical JSON rejects non-finite floats")
        return value
    raise V18Error(f"unsupported canonical type: {type(value).__name__}")


def canonical_json_bytes(
    value: Any,
    *,
    exclude_fields: Iterable[str] = (),
) -> bytes:
    excluded = frozenset(str(field) for field in exclude_fields)

    def strip(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): strip(child)
                for key, child in item.items()
                if str(key) not in excluded
            }
        if isinstance(item, (list, tuple)):
            return [strip(child) for child in item]
        return item

    return json.dumps(
        _safe(strip(value)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(
    value: Any,
    *,
    exclude_fields: Iterable[str] = (),
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(value, exclude_fields=exclude_fields)
    ).hexdigest()


def _require_sha(value: Any, name: str) -> str:
    token = str(value)
    if SHA256_RE.fullmatch(token) is None:
        raise V18Error(f"{name} must be a lowercase SHA-256")
    return token


def _date(value: Any, name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise V18Error(f"{name} must be a date") from exc
    if pd.isna(result):
        raise V18Error(f"{name} must be a date")
    if result.tzinfo is not None:
        result = result.tz_convert(TOKYO).tz_localize(None)
    return result.normalize()


def _month(value: Any, name: str = "month") -> pd.Period:
    try:
        result = pd.Period(str(value), freq="M")
    except (TypeError, ValueError) as exc:
        raise V18Error(f"{name} must be YYYY-MM") from exc
    if str(result) != str(value):
        raise V18Error(f"{name} must be canonical YYYY-MM")
    return result


def _timestamp(value: Any, name: str) -> datetime:
    try:
        result = datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise V18Error(f"{name} must be an ISO timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise V18Error(f"{name} must be timezone-aware")
    return result.astimezone(TOKYO)


def _cutoff(session: Any) -> datetime:
    return datetime.combine(_date(session, "session_date").date(), CUTOFF_TIME, TOKYO)


def _first_registered_session_in_month(target_month: pd.Period) -> pd.Timestamp:
    calendar = load_registered_calendar()
    matches = calendar[calendar.to_period("M") == target_month]
    if len(matches) == 0:
        raise V18Error(f"registered calendar does not cover {target_month}")
    return pd.Timestamp(matches[0])


def _latest_registered_source_before_month(target_month: pd.Period) -> pd.Timestamp:
    calendar = load_registered_calendar()
    previous = calendar[calendar < target_month.start_time.normalize()]
    if len(previous):
        return pd.Timestamp(previous[-1])
    if target_month == pd.Period("2026-08", freq="M"):
        # The forward target calendar starts on 2026-08-05.  The August
        # monthly training prefix nevertheless ends at the final registered
        # JPX source session of July; 2026-08-04 is D-1 for the first daily
        # target and must never leak into the M-1 snapshot/fold.
        return pd.Timestamp("2026-07-31")
    raise V18Error(f"registered calendar has no prior source session for {target_month}")


def _latest_required_predictor_source_session(
    target_session: Any,
    calendar: pd.DatetimeIndex | None = None,
) -> pd.Timestamp:
    """Return the protocol-registered D-1 predictor source for one target."""

    target = _date(target_session, "predictor target session")
    scheduled = load_registered_calendar() if calendar is None else calendar
    positions = np.flatnonzero(scheduled == target)
    if len(positions) != 1:
        raise V18Error("predictor target is outside the registered calendar")
    position = int(positions[0])
    if position == 0:
        if target != pd.Timestamp("2026-08-05"):
            raise V18Error("registered predictor calendar first-session contract changed")
        return pd.Timestamp("2026-08-04")
    return pd.Timestamp(scheduled[position - 1])


def _strict_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.lower().strip() in {"true", "false", "1", "0"}:
        return value.lower().strip() in {"true", "1"}
    raise V18Error(f"invalid boolean: {value!r}")


def _ieee_float_equal(left: Any, right: Any) -> bool:
    """Compare finite authority values by their exact binary64 encoding."""

    return struct.pack(">d", float(left)) == struct.pack(">d", float(right))


def _atomic_local_bytes_once(path: str | Path, payload: bytes) -> None:
    """Create one local authority file with deterministic crash recovery.

    A deterministic same-directory staging inode is completed only when its
    retained bytes are an exact prefix of ``payload``.  ``link(2)`` is the
    no-replace publication point.  A crash after link and before staging-name
    cleanup is healed only when both names still identify the pinned inode.
    The directory descriptor is the single cooperative writer lock; no
    unregistered lock-file namespace is created.
    """

    if not isinstance(payload, bytes) or not payload:
        raise V18Error("canonical local artifact payload must be non-empty bytes")
    target = Path(path)
    parent_path = Path(os.path.abspath(os.fspath(target.parent)))
    try:
        resolved_parent = parent_path.resolve(strict=True)
    except OSError as exc:
        raise V18Error("canonical artifact parent cannot be resolved") from exc
    if resolved_parent != parent_path or target.parent.is_symlink():
        raise V18Error("canonical artifact parent must not use a symlink")
    parent_fd = os.open(
        resolved_parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
    )
    file_name = target.name
    if file_name in {"", ".", ".."} or "/" in file_name or "\x00" in file_name:
        os.close(parent_fd)
        raise V18Error("canonical artifact file name is not a plain component")
    stage_name = f".{file_name}.staging"
    fcntl.flock(parent_fd, fcntl.LOCK_EX)
    try:
        parent_stat = os.fstat(parent_fd)
        registered_private_parent = _is_registered_local_authority_directory(
            resolved_parent
        )
        if (
            not stat.S_ISDIR(parent_stat.st_mode)
            or parent_stat.st_uid != os.geteuid()
            or (
                stat.S_IMODE(parent_stat.st_mode) != 0o700
                if registered_private_parent
                else bool(stat.S_IMODE(parent_stat.st_mode) & 0o022)
            )
        ):
            raise V18Error("canonical artifact parent owner/permissions are not private")

        def observed(name: str) -> os.stat_result | None:
            try:
                return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return None

        def private_regular(value: os.stat_result | None, *, nlink: int) -> bool:
            return bool(
                value is not None
                and stat.S_ISREG(value.st_mode)
                and value.st_uid == os.geteuid()
                and stat.S_IMODE(value.st_mode) == 0o600
                and value.st_nlink == nlink
            )

        def exact_pinned_bytes(descriptor: int, *, label: str) -> os.stat_result:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600
            ):
                raise V18Error(f"{label} is not a private regular file")
            os.lseek(descriptor, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if (
                (
                    after.st_dev,
                    after.st_ino,
                    after.st_nlink,
                    after.st_size,
                    after.st_uid,
                    stat.S_IMODE(after.st_mode),
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
                != (
                    before.st_dev,
                    before.st_ino,
                    before.st_nlink,
                    before.st_size,
                    before.st_uid,
                    stat.S_IMODE(before.st_mode),
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                or b"".join(chunks) != payload
            ):
                raise V18Error(f"{label} differs from exact retry bytes")
            return after

        final_stat = observed(file_name)
        stage_stat = observed(stage_name)
        if final_stat is not None:
            expected_final_links = 2 if stage_stat is not None else 1
            if not private_regular(final_stat, nlink=expected_final_links):
                raise V18Error("canonical artifact final is not private")
            if stage_stat is not None:
                if (
                    not private_regular(stage_stat, nlink=2)
                    or (stage_stat.st_dev, stage_stat.st_ino, stage_stat.st_nlink)
                    != (final_stat.st_dev, final_stat.st_ino, 2)
                ):
                    raise V18Error("canonical artifact has a conflicting stranded stage")
                stage_fd = os.open(
                    stage_name,
                    os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                try:
                    pinned = exact_pinned_bytes(
                        stage_fd, label="canonical artifact crash stage"
                    )
                    expected_inode = (pinned.st_dev, pinned.st_ino, 2)
                    current_stage = observed(stage_name)
                    current_final = observed(file_name)
                    if (
                        not private_regular(current_stage, nlink=2)
                        or not private_regular(current_final, nlink=2)
                        or (
                            current_stage.st_dev,
                            current_stage.st_ino,
                            current_stage.st_nlink,
                        )
                        != expected_inode
                        or (
                            current_final.st_dev,
                            current_final.st_ino,
                            current_final.st_nlink,
                        )
                        != expected_inode
                    ):
                        raise V18Error("canonical crash publication changed before cleanup")
                    os.unlink(stage_name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                    recovered = observed(file_name)
                    if not private_regular(recovered, nlink=1) or (
                        recovered.st_dev,
                        recovered.st_ino,
                        recovered.st_nlink,
                    ) != (pinned.st_dev, pinned.st_ino, 1):
                        raise V18Error("canonical recovered final changed after cleanup")
                finally:
                    os.close(stage_fd)
            final_fd = os.open(
                file_name,
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            try:
                retained = exact_pinned_bytes(
                    final_fd, label="canonical completed retry"
                )
                path_stat = observed(file_name)
                if not private_regular(path_stat, nlink=1) or (
                    path_stat.st_dev,
                    path_stat.st_ino,
                    path_stat.st_nlink,
                ) != (retained.st_dev, retained.st_ino, 1):
                    raise V18Error("canonical completed retry changed during validation")
            finally:
                os.close(final_fd)
            return

        if stage_stat is None:
            stage_fd = os.open(
                stage_name,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
        else:
            if not private_regular(stage_stat, nlink=1):
                raise V18Error("canonical stranded stage is not recoverable")
            stage_fd = os.open(
                stage_name,
                os.O_RDWR | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        try:
            pinned_before = os.fstat(stage_fd)
            current_before = observed(stage_name)
            if (
                not private_regular(pinned_before, nlink=1)
                or not private_regular(current_before, nlink=1)
                or (
                    current_before.st_dev,
                    current_before.st_ino,
                    current_before.st_nlink,
                )
                != (pinned_before.st_dev, pinned_before.st_ino, 1)
            ):
                raise V18Error("canonical stage changed during open")
            os.lseek(stage_fd, 0, os.SEEK_SET)
            retained_chunks: list[bytes] = []
            while True:
                chunk = os.read(stage_fd, 1024 * 1024)
                if not chunk:
                    break
                retained_chunks.append(chunk)
            retained_prefix = b"".join(retained_chunks)
            if not payload.startswith(retained_prefix):
                # A staging name is explicitly nonauthority until the
                # no-replace final link exists.  Under the private parent and
                # retained directory lock, abandon/rebuild a conflicting
                # unpublished dynamic timestamp/nonce payload.
                os.ftruncate(stage_fd, 0)
                os.fsync(stage_fd)
                retained_prefix = b""
            os.lseek(stage_fd, len(retained_prefix), os.SEEK_SET)
            remaining = memoryview(payload)[len(retained_prefix) :]
            while remaining:
                written = os.write(stage_fd, remaining)
                if written <= 0:  # pragma: no cover - kernel invariant
                    raise OSError("short write while publishing canonical artifact")
                remaining = remaining[written:]
            os.ftruncate(stage_fd, len(payload))
            os.fchmod(stage_fd, 0o600)
            os.fsync(stage_fd)
            pinned = exact_pinned_bytes(stage_fd, label="canonical completed stage")
            if not private_regular(pinned, nlink=1):
                raise V18Error("canonical completed stage acquired a hard-link alias")
            current_stage = observed(stage_name)
            if not private_regular(current_stage, nlink=1) or (
                current_stage.st_dev,
                current_stage.st_ino,
                current_stage.st_nlink,
            ) != (pinned.st_dev, pinned.st_ino, 1):
                raise V18Error("canonical completed stage changed before publication")
            try:
                os.link(
                    stage_name,
                    file_name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise V18Error("canonical artifact appeared during publication") from exc
            os.fsync(parent_fd)
            expected_inode = (pinned.st_dev, pinned.st_ino, 2)
            current_stage = observed(stage_name)
            current_final = observed(file_name)
            if (
                not private_regular(current_stage, nlink=2)
                or not private_regular(current_final, nlink=2)
                or (
                    current_stage.st_dev,
                    current_stage.st_ino,
                    current_stage.st_nlink,
                )
                != expected_inode
                or (
                    current_final.st_dev,
                    current_final.st_ino,
                    current_final.st_nlink,
                )
                != expected_inode
            ):
                raise V18Error("canonical publication changed before stage cleanup")
            os.unlink(stage_name, dir_fd=parent_fd)
            os.fsync(parent_fd)
            sealed = observed(file_name)
            if not private_regular(sealed, nlink=1) or (
                sealed.st_dev,
                sealed.st_ino,
                sealed.st_nlink,
            ) != (pinned.st_dev, pinned.st_ino, 1):
                raise V18Error("canonical final changed after publication")
        finally:
            os.close(stage_fd)
    finally:
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(parent_fd)


def _atomic_local_derived_bytes(path: str | Path, payload: bytes) -> None:
    """Replace a derived local view only from an exact retained prefix.

    Immutable per-session artifacts remain authority.  This helper is only
    for a deterministic concatenated view (currently the score CSV).  A torn
    prior view or staging file is recoverable iff its bytes are an exact
    prefix of the freshly reconstructed payload; any divergent byte aborts.
    """

    if not isinstance(payload, bytes) or not payload:
        raise V18Error("derived local artifact payload must be non-empty bytes")
    target = Path(path)
    parent_path = Path(os.path.abspath(os.fspath(target.parent)))
    if parent_path.resolve(strict=True) != parent_path or target.parent.is_symlink():
        raise V18Error("derived artifact parent must not use a symlink")
    parent_fd = os.open(
        parent_path,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
    )
    file_name = target.name
    stage_name = f".{file_name}.derived-staging"
    fcntl.flock(parent_fd, fcntl.LOCK_EX)
    try:
        parent_stat = os.fstat(parent_fd)
        if (
            parent_stat.st_uid != os.geteuid()
            or stat.S_IMODE(parent_stat.st_mode) & 0o022
        ):
            raise V18Error("derived artifact parent owner/permissions are not private")

        def read_name(name: str) -> bytes | None:
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
            except FileNotFoundError:
                return None
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    raise V18Error("derived artifact/stage is not single-link regular")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                after = os.fstat(descriptor)
                if (
                    after.st_dev,
                    after.st_ino,
                    after.st_nlink,
                    after.st_size,
                ) != (
                    before.st_dev,
                    before.st_ino,
                    before.st_nlink,
                    before.st_size,
                ):
                    raise V18Error("derived artifact changed while being read")
                return b"".join(chunks)
            finally:
                os.close(descriptor)

        retained = read_name(file_name)
        if retained == payload:
            stranded = read_name(stage_name)
            if stranded is not None:
                if not payload.startswith(stranded):
                    raise V18Error("derived completed view has a conflicting stage")
                os.unlink(stage_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            return
        if retained is not None and not payload.startswith(retained):
            raise V18Error("derived retained view is not an exact authority prefix")
        staged = read_name(stage_name)
        if staged is not None and not payload.startswith(staged):
            raise V18Error("derived stranded stage is not an exact authority prefix")
        descriptor = os.open(
            stage_name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        try:
            pinned = os.fstat(descriptor)
            if not stat.S_ISREG(pinned.st_mode) or pinned.st_nlink != 1:
                raise V18Error("derived stage changed during open")
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.ftruncate(descriptor, 0)
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:  # pragma: no cover - kernel invariant
                    raise OSError("short derived artifact write")
                remaining = remaining[written:]
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            sealed = os.fstat(descriptor)
            if (
                sealed.st_dev,
                sealed.st_ino,
                sealed.st_nlink,
                sealed.st_size,
            ) != (pinned.st_dev, pinned.st_ino, 1, len(payload)):
                raise V18Error("derived stage metadata changed")
        finally:
            os.close(descriptor)
        os.fsync(parent_fd)
        os.replace(
            stage_name,
            file_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
        if read_name(file_name) != payload:
            raise V18Error("derived final differs after atomic replacement")
    finally:
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(parent_fd)


def _atomic_text(path: str | Path, payload: str, *, exclusive: bool = False) -> None:
    target = Path(path)
    encoded = payload.encode("utf-8")
    if exclusive:
        _atomic_local_bytes_once(target, encoded)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
        directory_fd = os.open(
            target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def write_json(value: Mapping[str, Any], path: str | Path, *, exclusive: bool = False) -> None:
    _atomic_text(
        path,
        json.dumps(_safe(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        exclusive=exclusive,
    )


def _write_json_once_exact(
    value: Mapping[str, Any], path: str | Path, *, label: str
) -> bytes:
    """Seal canonical JSON once, accepting only an exact completed retry."""

    target = Path(path)
    expected = _json_file_bytes(value)
    write_json(value, target, exclusive=True)
    retained = (
        _read_local_authority_bytes(target, label=label)
        if _is_registered_local_authority_path(target)
        else _plain_file_bytes(target, label=label)
    )
    if retained != expected:
        raise V18Error(f"{label} changed immediately after seal")
    return retained


def _runtime_relative_path(value: Any, name: str) -> str:
    token = str(value)
    path = Path(token)
    if (
        not token
        or path.is_absolute()
        or path.as_posix() != token
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\x00" in token
    ):
        raise V18Error(f"{name} is not a canonical relative POSIX path")
    return token


def _runtime_project_files(value: Mapping[str, Any]) -> list[dict[str, str]]:
    entries = value.get("project_files")
    if not isinstance(entries, list) or not entries:
        raise V18Error("runtime lock project_files must be a non-empty array")
    observed: list[dict[str, str]] = []
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise V18Error(f"runtime lock project_files[{index}] schema changed")
        relative = _runtime_relative_path(item["path"], f"project_files[{index}].path")
        digest = _require_sha(item["sha256"], f"project_files[{index}].sha256")
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise V18Error(f"runtime-locked project file is missing or symlinked: {relative}")
        if sha256_file(path) != digest:
            raise V18Error(f"runtime-locked project file bytes changed: {relative}")
        observed.append({"path": relative, "sha256": digest})
    paths = [item["path"] for item in observed]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise V18Error("runtime lock project_files are not uniquely path-sorted")
    expected_python = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src/tse_session_ranker").rglob("*.py")
        if path.is_file() and not path.is_symlink()
    )
    expected_research = sorted(
        [
            "research/finalize_logit_v04.py",
            "research/model_v13_symbolic_context_runner.py",
            "research/model_v16_liquidity_runner.py",
            "research/model_v17_liquidity_reliability_runner.py",
        ]
    )
    if paths != sorted([*expected_research, *expected_python]):
        raise V18Error("runtime lock does not cover the exact transitive project closure")
    if canonical_json_sha256(observed) != value.get("project_file_set_sha256"):
        raise V18Error("runtime lock project-file set hash changed")
    return observed


def _distribution_tree(distribution: importlib.metadata.Distribution) -> list[dict[str, str]]:
    files = distribution.files
    if files is None:
        raise V18Error(f"distribution has no installed file registry: {distribution.metadata['Name']}")
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for package_path in files:
        relative = Path(str(package_path)).as_posix()
        parts = Path(relative).parts
        if Path(relative).suffix in {".pyc", ".pyo"} or "__pycache__" in parts:
            continue
        located = Path(distribution.locate_file(package_path))
        if not located.exists():
            continue
        if located.is_symlink():
            raise V18Error(
                f"runtime distribution contains a symlinked registered file: {relative}"
            )
        if not located.is_file():
            continue
        if relative in seen:
            raise V18Error("runtime distribution file registry contains a duplicate")
        seen.add(relative)
        records.append({"path": relative, "sha256": sha256_file(located)})
    return sorted(records, key=lambda item: item["path"])


_STARTUP_CONTROL_ENVIRONMENT = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "PYTHONSAFEPATH",
    "PYTHONUSERBASE",
    "SETUPTOOLS_USE_DISTUTILS",
)


def _runtime_authority_path(path: Path, *, name: str) -> Path:
    """Return one absolute, resolved, non-symlink regular-file authority."""

    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except OSError as exc:
        raise V18Error(f"{name} is unavailable") from exc
    if (
        not path.is_absolute()
        or resolved != path
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise V18Error(f"{name} is not an absolute resolved plain file")
    return resolved


def _normalise_live_module_path(
    raw_path: Any,
    *,
    module_name: str,
    allow_relative_direct: bool = False,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise V18Error(f"live module {module_name} has an invalid origin")
    path = Path(raw_path)
    if path.suffix in {".pyc", ".pyo"}:
        try:
            path = Path(importlib.util.source_from_cache(str(path)))
        except ValueError as exc:
            raise V18Error(
                f"live module {module_name} has an unregistered bytecode origin"
            ) from exc
    if not path.is_absolute():
        if not allow_relative_direct:
            raise V18Error(f"live module {module_name} origin is relative")
        path = Path.cwd() / path
    return _runtime_authority_path(path, name=f"live module {module_name} origin")


def _build_live_module_closure_state(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Validate startup authorities and cache exact file ownership for live imports."""

    runtime = lock["runtime"]
    startup = runtime.get("startup_and_module_closure")
    required_startup_fields = {
        "schema_version",
        "site_packages_roots",
        "startup_file_required_fields",
        "startup_files",
        "startup_file_set_sha256",
        "startup_file_set_hash_rule",
        "startup_module_required_fields",
        "startup_modules",
        "startup_module_set_sha256",
        "customize_modules",
        "user_site",
        "direct_activation_project_module_paths",
        "originless_runtime_module_names",
        "generated_alias_module_required_fields",
        "generated_alias_modules",
        "generated_alias_rule",
        "loaded_module_origin_rule",
        "existing_cli_rule",
    }
    if not isinstance(startup, Mapping) or set(startup) != required_startup_fields:
        raise V18Error("startup/module closure schema changed")
    startup_file_fields = [
        "path",
        "basename",
        "distribution_name",
        "size_bytes",
        "sha256",
    ]
    startup_module_fields = [
        "module_name",
        "origin_path",
        "distribution_name",
        "origin_sha256",
    ]
    generated_alias_fields = [
        "module_name",
        "spec_origin",
        "module_file",
        "search_locations",
        "loader_class_module",
        "loader_class_qualname",
        "provider_distribution_name",
        "provider_path",
        "provider_sha256",
    ]
    originless = startup["originless_runtime_module_names"]
    generated_aliases = startup["generated_alias_modules"]
    direct_relatives = startup["direct_activation_project_module_paths"]
    site_root_values = startup["site_packages_roots"]
    if (
        startup["schema_version"] != 1
        or startup["startup_file_required_fields"] != startup_file_fields
        or startup["startup_module_required_fields"] != startup_module_fields
        or startup["generated_alias_module_required_fields"] != generated_alias_fields
        or not isinstance(startup["startup_files"], list)
        or not isinstance(startup["startup_modules"], list)
        or startup["startup_file_set_sha256"]
        != canonical_json_sha256(startup["startup_files"])
        or startup["startup_module_set_sha256"]
        != canonical_json_sha256(startup["startup_modules"])
        or not isinstance(startup["startup_file_set_hash_rule"], str)
        or not startup["startup_file_set_hash_rule"]
        or not isinstance(startup["loaded_module_origin_rule"], str)
        or not startup["loaded_module_origin_rule"]
        or not isinstance(startup["generated_alias_rule"], str)
        or not startup["generated_alias_rule"]
        or not isinstance(startup["existing_cli_rule"], str)
        or not startup["existing_cli_rule"]
        or not isinstance(originless, list)
        or originless != sorted(set(str(item) for item in originless))
        or not isinstance(generated_aliases, list)
        or [str(item.get("module_name")) for item in generated_aliases]
        != sorted(set(str(item.get("module_name")) for item in generated_aliases))
        or not isinstance(site_root_values, list)
        or not site_root_values
        or site_root_values != sorted(set(str(item) for item in site_root_values))
        or direct_relatives
        != [
            "research/model_v18_a2_rehearsal.py",
            "research/model_v18_shoulder_state_audit.py",
            "research/model_v18_shoulder_state_runner.py",
        ]
    ):
        raise V18Error("startup/module closure registry changed")
    if any(name in os.environ for name in _STARTUP_CONTROL_ENVIRONMENT):
        raise V18Error("Python startup-control environment is not clean")

    authorized_files: dict[str, str] = {}
    path_groups: dict[str, set[str]] = {}
    group_files: dict[str, set[str]] = {}

    def register_file(path: Path, digest: str, group: str, name: str) -> Path:
        authority = _runtime_authority_path(path, name=name)
        expected = _require_sha(digest, f"{name} SHA")
        if sha256_file(authority) != expected:
            raise V18Error(f"{name} bytes changed")
        key = str(authority)
        prior = authorized_files.setdefault(key, expected)
        if prior != expected:
            raise V18Error("runtime authorities disagree about one file")
        path_groups.setdefault(key, set()).add(group)
        group_files.setdefault(group, set()).add(key)
        return authority

    stdlib_root = Path(sysconfig.get_paths()["stdlib"]).resolve(strict=True)
    stdlib_records: list[dict[str, str]] = []
    for observed_path in stdlib_root.rglob("*"):
        relative = observed_path.relative_to(stdlib_root)
        if (
            "site-packages" in relative.parts
            or "dist-packages" in relative.parts
            or "__pycache__" in relative.parts
            or observed_path.suffix in {".pyc", ".pyo"}
        ):
            continue
        if observed_path.is_symlink():
            raise V18Error("live-module stdlib authority contains a symlink")
        if observed_path.is_file():
            digest = sha256_file(observed_path)
            stdlib_records.append({"path": relative.as_posix(), "sha256": digest})
            register_file(
                observed_path,
                digest,
                "stdlib",
                "live-module stdlib authority",
            )
    stdlib_records.sort(key=lambda item: item["path"])
    python_lock = runtime["python"]
    if (
        len(stdlib_records) != int(python_lock["stdlib_file_count"])
        or canonical_json_sha256(stdlib_records)
        != python_lock["stdlib_tree_sha256"]
    ):
        raise V18Error("live-module stdlib authority changed")

    for item in _runtime_project_files(lock):
        register_file(
            ROOT / item["path"],
            item["sha256"],
            "project:locked",
            f"live-module project authority {item['path']}",
        )

    direct_paths: set[str] = set()
    for relative in direct_relatives:
        canonical_relative = _runtime_relative_path(relative, "direct activation path")
        path = _runtime_authority_path(
            ROOT / canonical_relative,
            name=f"direct activation module {canonical_relative}",
        )
        # These direct entry points cannot be self-bound by the runtime lock. Capture
        # their exact bytes at strict startup; activation subsequently binds
        # the same bytes to commit A, and every phase re-hashes them here.
        register_file(
            path,
            sha256_file(path),
            "project:direct",
            f"direct activation module {canonical_relative}",
        )
        direct_paths.add(str(path))

    distribution_file_groups: dict[str, set[str]] = {}
    for locked in runtime["distributions"]:
        name = str(locked["name"])
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise V18Error(f"live-module distribution is unavailable: {name}") from exc
        tree = _distribution_tree(distribution)
        if (
            len(tree) != int(locked["file_count"])
            or canonical_json_sha256(tree) != locked["tree_sha256"]
        ):
            raise V18Error(f"live-module distribution authority changed: {name}")
        group = f"distribution:{name}"
        for item in tree:
            located = Path(distribution.locate_file(item["path"])).resolve(strict=True)
            register_file(
                located,
                item["sha256"],
                group,
                f"live-module distribution authority {name}",
            )
        distribution_file_groups[name] = set(group_files.get(group, set()))

    site_roots: list[Path] = []
    sys_path_roots = {
        str((Path.cwd() if item == "" else Path(item)).resolve(strict=False))
        for item in sys.path
    }
    for raw_root in site_root_values:
        root = Path(str(raw_root))
        try:
            root_stat = root.lstat()
            resolved_root = root.resolve(strict=True)
        except OSError as exc:
            raise V18Error("registered site-packages root is unavailable") from exc
        if (
            not root.is_absolute()
            or root != resolved_root
            or stat.S_ISLNK(root_stat.st_mode)
            or not stat.S_ISDIR(root_stat.st_mode)
            or str(root) not in sys_path_roots
        ):
            raise V18Error("registered site-packages root identity changed")
        site_roots.append(root)

    startup_files = startup["startup_files"]
    registered_startup_paths: list[str] = []
    for index, item in enumerate(startup_files):
        if not isinstance(item, Mapping) or set(item) != set(startup_file_fields):
            raise V18Error(f"startup file registry item {index} changed")
        path = _runtime_authority_path(
            Path(str(item["path"])), name=f"startup file {index}"
        )
        owner = str(item["distribution_name"])
        if (
            path.name != item["basename"]
            or path.stat().st_size != int(item["size_bytes"])
            or str(path) not in distribution_file_groups.get(owner, set())
            or sha256_file(path) != _require_sha(item["sha256"], "startup file SHA")
        ):
            raise V18Error(f"startup file registry item {index} differs")
        registered_startup_paths.append(str(path))
    if registered_startup_paths != sorted(set(registered_startup_paths)):
        raise V18Error("startup file registry is not uniquely path-sorted")

    observed_startup_files: list[dict[str, Any]] = []
    for root in site_roots:
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            if path.suffix != ".pth":
                continue
            authority = _runtime_authority_path(path, name="observed startup .pth")
            owners = sorted(
                name
                for name, files in distribution_file_groups.items()
                if str(authority) in files
            )
            if len(owners) != 1:
                raise V18Error("startup .pth lacks one registered distribution owner")
            observed_startup_files.append(
                {
                    "path": str(authority),
                    "basename": authority.name,
                    "distribution_name": owners[0],
                    "size_bytes": authority.stat().st_size,
                    "sha256": sha256_file(authority),
                }
            )
    observed_startup_files.sort(key=lambda item: item["path"])
    if observed_startup_files != startup_files:
        raise V18Error("interpreter startup .pth set differs from the runtime lock")

    startup_modules = startup["startup_modules"]
    observed_startup_modules: list[dict[str, Any]] = []
    for index, item in enumerate(startup_modules):
        if not isinstance(item, Mapping) or set(item) != set(startup_module_fields):
            raise V18Error(f"startup module registry item {index} changed")
        module_name = str(item["module_name"])
        module = sys.modules.get(module_name)
        spec = None if module is None else getattr(module, "__spec__", None)
        raw_origin = None if spec is None else getattr(spec, "origin", None)
        if module is None or raw_origin in {None, "built-in", "frozen"}:
            raise V18Error(f"registered startup module is not live: {module_name}")
        origin = _normalise_live_module_path(raw_origin, module_name=module_name)
        owner = str(item["distribution_name"])
        observed = {
            "module_name": module_name,
            "origin_path": str(origin),
            "distribution_name": owner,
            "origin_sha256": sha256_file(origin),
        }
        if (
            str(origin) not in distribution_file_groups.get(owner, set())
            or observed != dict(item)
        ):
            raise V18Error(f"registered startup module differs: {module_name}")
        observed_startup_modules.append(observed)
    if [item["module_name"] for item in observed_startup_modules] != sorted(
        set(item["module_name"] for item in observed_startup_modules)
    ):
        raise V18Error("startup module registry is not uniquely name-sorted")

    validated_aliases: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(generated_aliases):
        if not isinstance(item, Mapping) or set(item) != set(generated_alias_fields):
            raise V18Error(f"generated alias registry item {index} changed")
        module_name = str(item["module_name"])
        provider_name = str(item["provider_distribution_name"])
        provider_path = _runtime_authority_path(
            Path(str(item["provider_path"])),
            name=f"generated alias provider {module_name}",
        )
        if (
            item["spec_origin"] is not None
            or item["module_file"] is not None
            or item["search_locations"] != []
            or not str(item["loader_class_module"])
            or not str(item["loader_class_qualname"])
            or str(provider_path) not in distribution_file_groups.get(provider_name, set())
            or sha256_file(provider_path)
            != _require_sha(item["provider_sha256"], "generated alias provider SHA")
        ):
            raise V18Error(f"generated alias registry item {index} differs")
        validated_aliases[module_name] = dict(item)

    customize = startup["customize_modules"]
    if not isinstance(customize, Mapping) or customize != {
        "sitecustomize": None,
        "usercustomize": None,
    }:
        raise V18Error("Python customize-module registry changed")
    for module_name in customize:
        if module_name in sys.modules:
            raise V18Error(f"forbidden startup customization module is live: {module_name}")
        try:
            discovered = importlib.util.find_spec(module_name)
        except (ImportError, ValueError) as exc:
            raise V18Error(f"startup customization discovery failed: {module_name}") from exc
        if discovered is not None:
            raise V18Error(f"forbidden startup customization is discoverable: {module_name}")

    user_site = startup["user_site"]
    if not isinstance(user_site, Mapping) or set(user_site) != {
        "path",
        "must_exist",
        "must_be_on_sys_path",
    }:
        raise V18Error("Python user-site registry changed")
    user_site_path = Path(str(user_site["path"]))
    if (
        not user_site_path.is_absolute()
        or user_site["must_exist"] is not False
        or user_site["must_be_on_sys_path"] is not False
        or os.path.lexists(user_site_path)
        or str(user_site_path.resolve(strict=False)) in sys_path_roots
    ):
        raise V18Error("Python user-site boundary changed")

    return {
        "startup": dict(startup),
        "authorized_files": authorized_files,
        "path_groups": {key: frozenset(value) for key, value in path_groups.items()},
        "group_files": {key: frozenset(value) for key, value in group_files.items()},
        "distribution_file_groups": {
            key: frozenset(value) for key, value in distribution_file_groups.items()
        },
        "direct_paths": frozenset(direct_paths),
        "allowed_originless": frozenset(str(item) for item in originless),
        "generated_aliases": validated_aliases,
        "site_roots": tuple(site_roots),
        "user_site_path": user_site_path,
    }


def _validate_startup_surface(state: Mapping[str, Any], *, phase: str) -> None:
    """Re-check mutable startup controls before/after each lazy-import phase."""

    if any(name in os.environ for name in _STARTUP_CONTROL_ENVIRONMENT):
        raise V18Error(f"{phase} Python startup-control environment changed")
    startup = state["startup"]
    observed: list[dict[str, Any]] = []
    distribution_groups = state["distribution_file_groups"]
    for root in state["site_roots"]:
        try:
            metadata = root.lstat()
        except OSError as exc:
            raise V18Error(f"{phase} site-packages root is unavailable") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or root.resolve(strict=True) != root
        ):
            raise V18Error(f"{phase} site-packages root identity changed")
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            if path.suffix != ".pth":
                continue
            authority = _runtime_authority_path(path, name=f"{phase} startup .pth")
            owners = sorted(
                name
                for name, files in distribution_groups.items()
                if str(authority) in files
            )
            if len(owners) != 1:
                raise V18Error(f"{phase} startup .pth owner changed")
            observed.append(
                {
                    "path": str(authority),
                    "basename": authority.name,
                    "distribution_name": owners[0],
                    "size_bytes": authority.stat().st_size,
                    "sha256": sha256_file(authority),
                }
            )
    observed.sort(key=lambda item: item["path"])
    if observed != startup["startup_files"]:
        raise V18Error(f"{phase} interpreter startup .pth set changed")
    for name in startup["customize_modules"]:
        if name in sys.modules:
            raise V18Error(f"{phase} loaded forbidden customization module {name}")
        try:
            discovered = importlib.util.find_spec(name)
        except (ImportError, ValueError) as exc:
            raise V18Error(f"{phase} customization-module discovery failed") from exc
        if discovered is not None:
            raise V18Error(f"{phase} discovered forbidden customization module {name}")
    user_site = state["user_site_path"]
    sys_path_roots = {
        str((Path.cwd() if item == "" else Path(item)).resolve(strict=False))
        for item in sys.path
    }
    if os.path.lexists(user_site) or str(user_site.resolve(strict=False)) in sys_path_roots:
        raise V18Error(f"{phase} Python user-site boundary changed")


def _validate_live_modules(state: Mapping[str, Any], *, phase: str) -> dict[str, str]:
    """Re-hash every live module against its exact registered file authority."""

    authorized: Mapping[str, str] = state["authorized_files"]
    path_groups: Mapping[str, frozenset[str]] = state["path_groups"]
    group_files: Mapping[str, frozenset[str]] = state["group_files"]
    direct_paths: frozenset[str] = state["direct_paths"]
    allowed_originless: frozenset[str] = state["allowed_originless"]
    generated_aliases: Mapping[str, Mapping[str, Any]] = state["generated_aliases"]
    allowed_file_loaders = {
        ("_frozen_importlib_external", "SourceFileLoader"),
        ("_frozen_importlib_external", "ExtensionFileLoader"),
        ("_frozen_importlib_external", "SourcelessFileLoader"),
    }

    def validate_file(module_name: str, raw_path: Any) -> tuple[str, frozenset[str]]:
        path = _normalise_live_module_path(
            raw_path,
            module_name=module_name,
            allow_relative_direct=module_name in {"__main__", "__mp_main__"},
        )
        key = str(path)
        expected = authorized.get(key)
        if expected is None or sha256_file(path) != expected:
            raise V18Error(f"{phase} live module {module_name} origin is not locked")
        if module_name in {"__main__", "__mp_main__"} and key not in direct_paths:
            raise V18Error(f"{phase} live {module_name} is not a registered v1.8 CLI")
        return key, path_groups[key]

    def validate_namespace(module_name: str, locations: Sequence[Any]) -> None:
        if not locations:
            raise V18Error(f"{phase} namespace module {module_name} has no path")
        for raw_location in locations:
            location = Path(str(raw_location))
            try:
                metadata = location.lstat()
                resolved = location.resolve(strict=True)
            except OSError as exc:
                raise V18Error(
                    f"{phase} namespace module {module_name} path is unavailable"
                ) from exc
            if (
                not location.is_absolute()
                or location != resolved
                or stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(metadata.st_mode)
            ):
                raise V18Error(f"{phase} namespace module {module_name} path is unsafe")
            matching_groups = [
                group
                for group, files in group_files.items()
                if any(Path(path).is_relative_to(location) for path in files)
            ]
            if not matching_groups:
                raise V18Error(f"{phase} namespace module {module_name} is not locked")

    observed: dict[str, str] = {}
    for module_name, module in sorted(tuple(sys.modules.items())):
        if module is None:
            continue
        spec = getattr(module, "__spec__", None)
        origin = None if spec is None else getattr(spec, "origin", None)
        module_file = getattr(module, "__file__", None)
        loader = None if spec is None else getattr(spec, "loader", None)
        loader_identity = (
            None
            if loader is None
            else (type(loader).__module__, type(loader).__qualname__)
        )
        if origin in {"built-in", "frozen"}:
            if loader is not None and loader_identity != ("builtins", "type"):
                raise V18Error(f"{phase} built-in/frozen module {module_name} loader changed")
            if module_file not in {None, origin}:
                validate_file(module_name, module_file)
            observed[module_name] = str(origin)
            continue

        candidate = origin if origin not in {None, "namespace"} else module_file
        if candidate is not None:
            key, _ = validate_file(module_name, candidate)
            if module_file is not None and module_file != candidate:
                validate_file(module_name, module_file)
            if spec is not None and loader_identity not in allowed_file_loaders:
                raise V18Error(f"{phase} live module {module_name} uses an unregistered loader")
            observed[module_name] = key
            continue

        search_locations = (
            None if spec is None else getattr(spec, "submodule_search_locations", None)
        )
        if search_locations is not None:
            locations = list(search_locations)
            alias = generated_aliases.get(module_name)
            if alias is not None:
                provider_module = sys.modules.get(str(alias["loader_class_module"]))
                provider_file = (
                    None
                    if provider_module is None
                    else getattr(provider_module, "__file__", None)
                )
                expected_loader = (
                    str(alias["loader_class_module"]),
                    str(alias["loader_class_qualname"]),
                )
                if (
                    origin is not alias["spec_origin"]
                    or module_file is not alias["module_file"]
                    or locations != alias["search_locations"]
                    or loader_identity != expected_loader
                    or provider_file is None
                ):
                    raise V18Error(f"{phase} generated alias {module_name} changed")
                provider_path, provider_groups = validate_file(
                    str(alias["loader_class_module"]), provider_file
                )
                if (
                    provider_path != alias["provider_path"]
                    or sha256_file(Path(provider_path)) != alias["provider_sha256"]
                    or f"distribution:{alias['provider_distribution_name']}"
                    not in provider_groups
                ):
                    raise V18Error(f"{phase} generated alias provider is not locked")
                observed[module_name] = "registered-generated-alias"
                continue
            if loader_identity != ("_frozen_importlib_external", "NamespaceLoader"):
                raise V18Error(f"{phase} namespace module {module_name} loader changed")
            validate_namespace(module_name, locations)
            observed[module_name] = "namespace"
            continue

        if module_name not in allowed_originless or spec is not None:
            raise V18Error(f"{phase} live module {module_name} has no registered origin")
        observed[module_name] = "originless"
    return observed


def _validate_startup_and_module_closure(
    lock: Mapping[str, Any] | None = None,
    *,
    phase: str = "runtime startup",
    initialise: bool = False,
) -> dict[str, str]:
    """Validate startup hooks and the exhaustive live-module origin closure."""

    global _LIVE_MODULE_CLOSURE_STATE
    if initialise:
        if lock is None:
            raise V18Error("module-closure initialisation lacks the runtime lock")
        state = _build_live_module_closure_state(lock)
        _LIVE_MODULE_CLOSURE_STATE = state
    else:
        state = _LIVE_MODULE_CLOSURE_STATE
        if state is None:
            if _STRICT_RUNTIME_ACTIVE:
                # A few pure unit fixtures emulate the old strict flag without
                # entering the operational CLI.  They cannot write canonical
                # artifacts; real strict activation always initialises state.
                return {}
            return {}
    _validate_startup_surface(state, phase=phase)
    return _validate_live_modules(state, phase=phase)


def _normalised_native_threadpools(*, hash_libraries: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in threadpoolctl.threadpool_info():
        library_path = Path(str(item.get("filepath", "")))
        if not library_path.is_file():
            raise V18Error("registered native threadpool library is missing")
        record: dict[str, Any] = {
            "user_api": item.get("user_api"),
            "internal_api": item.get("internal_api"),
            "prefix": item.get("prefix"),
            "version": item.get("version"),
            "threading_layer": item.get("threading_layer"),
            "architecture": item.get("architecture"),
            "num_threads": int(item.get("num_threads", -1)),
            "library_file_name": library_path.resolve().name,
        }
        if hash_libraries:
            record["library_sha256"] = sha256_file(library_path.resolve())
        records.append(record)
    return sorted(
        records,
        key=lambda item: (
            str(item["user_api"]),
            str(item["internal_api"]),
            str(item["prefix"]),
            str(item["library_file_name"]),
        ),
    )


def _external_process_environment(kind: str) -> dict[str, str]:
    """Construct the exact non-inherited child environment from the lock."""

    closure = read_json(RUNTIME_LOCK)["elf_closure"]
    contract = closure["external_process_environment"]
    if contract.get("inherit_parent_environment") is not False:
        raise V18Error("external child environment inheritance is not disabled")
    if kind not in {"git", "pdftotext"}:
        raise V18Error("unregistered external process kind")
    common = contract.get("common_exact")
    extra = contract.get(f"{kind}_exact_extra")
    if not isinstance(common, Mapping) or not isinstance(extra, Mapping):
        raise V18Error("external child environment contract is invalid")
    overlap = set(common) & set(extra)
    if overlap:
        raise V18Error("external child environment has overlapping keys")
    result = {str(key): str(value) for key, value in {**common, **extra}.items()}
    if any(not key or "\x00" in key or "\x00" in value for key, value in result.items()):
        raise V18Error("external child environment contains an invalid value")
    return result


def _subprocess_environment() -> dict[str, str]:
    return _external_process_environment("pdftotext")


def _git_environment() -> dict[str, str]:
    return _external_process_environment("git")


def _validate_tls_ca_trust(ca_trust: Any) -> Path:
    """Validate the sole registered HTTPS trust anchor without default paths."""

    ca_fields = (
        "schema_version",
        "cafile_path",
        "cafile_basename",
        "cafile_size_bytes",
        "cafile_sha256",
        "environment",
        "context_rule",
    )
    if not isinstance(ca_trust, Mapping) or tuple(ca_trust) != ca_fields:
        raise V18Error("TLS CA trust-store schema changed")
    if ca_trust["schema_version"] != 1 or ca_trust["environment"] != {
        "SSL_CERT_FILE": None,
        "SSL_CERT_DIR": None,
    }:
        raise V18Error("TLS CA trust-store identity changed")
    if any(name in os.environ for name in ca_trust["environment"]):
        raise V18Error("TLS CA environment overrides are forbidden")
    ca_path = Path(str(ca_trust["cafile_path"]))
    try:
        ca_stat = ca_path.stat()
    except OSError as exc:
        raise V18Error("registered TLS CA file is unavailable") from exc
    if (
        not ca_path.is_absolute()
        or ca_path.is_symlink()
        or not stat.S_ISREG(ca_stat.st_mode)
        or ca_path.resolve() != ca_path
        or ca_path.name != ca_trust["cafile_basename"]
        or ca_stat.st_size != int(ca_trust["cafile_size_bytes"])
        or sha256_file(ca_path) != ca_trust["cafile_sha256"]
    ):
        raise V18Error("registered TLS CA trust-store bytes changed")
    return ca_path


def _elf_role_path(lock: Mapping[str, Any], role: str) -> Path:
    matches = [
        Path(str(item["path"]))
        for item in lock["elf_closure"]["root_objects"]
        if role in item["roles"]
    ]
    if len(matches) != 1:
        raise V18Error(f"ELF closure does not bind exactly one {role}")
    return matches[0]


def _validate_git_executable(lock: Mapping[str, Any]) -> Path:
    global _LOCKED_GIT_EXECUTABLE
    git_lock = lock["git"]
    if set(git_lock) != {
        "implementation",
        "executable_path",
        "executable_basename",
        "version",
        "version_output_sha256",
        "executable_sha256",
        "verification_rule",
    } or git_lock["implementation"] != "Git":
        raise V18Error("Git runtime-lock schema changed")
    path = _elf_role_path(lock, "git_executable")
    if path.is_symlink() or not path.is_file():
        raise V18Error("Git executable is missing or symlinked")
    if (
        str(path) != git_lock["executable_path"]
        or path.name != git_lock["executable_basename"]
        or path.resolve() != path
        or sha256_file(path) != git_lock["executable_sha256"]
    ):
        raise V18Error("Git executable bytes differ from the runtime lock")
    completed = subprocess.run(
        [str(path), "--version"],
        check=False,
        capture_output=True,
        env=_git_environment(),
        shell=False,
    )
    if (
        completed.returncode != 0
        or completed.stderr
        or hashlib.sha256(completed.stdout).hexdigest()
        != git_lock["version_output_sha256"]
        or completed.stdout != f"git version {git_lock['version']}\n".encode("ascii")
    ):
        raise V18Error("Git version output differs from the runtime lock")
    _LOCKED_GIT_EXECUTABLE = path
    return path


def _validate_elf_closure(lock: Mapping[str, Any]) -> None:
    """Verify preregistered ELF roots/dependencies without a linkage subprocess."""

    closure = lock["elf_closure"]
    required_top = {
        "schema_version",
        "root_object_required_fields",
        "shared_object_required_fields",
        "root_objects",
        "root_object_count",
        "root_object_set_sha256",
        "shared_objects",
        "shared_object_count",
        "shared_object_set_sha256",
        "dynamic_loader",
        "loader_cache",
        "virtual_object_exception",
        "loader_environment",
        "external_process_environment",
        "object_path_rule",
        "acquisition_rule",
        "runtime_validation_rule",
        "external_process_rule",
    }
    if not isinstance(closure, Mapping) or set(closure) != required_top:
        raise V18Error("ELF closure top-level schema changed")
    if int(closure["schema_version"]) != 1:
        raise V18Error("ELF closure schema version changed")
    root_fields = tuple(closure["root_object_required_fields"])
    shared_fields = tuple(closure["shared_object_required_fields"])
    if root_fields != ("path", "basename", "roles", "sha256") or shared_fields != (
        "path",
        "basename",
        "sha256",
    ):
        raise V18Error("ELF closure object schemas changed")

    def validate_objects(
        records: Any,
        *,
        required_fields: tuple[str, ...],
        name: str,
    ) -> list[dict[str, Any]]:
        if not isinstance(records, list):
            raise V18Error(f"ELF {name} registry is not an array")
        paths = [str(item.get("path")) for item in records if isinstance(item, Mapping)]
        if len(paths) != len(records) or paths != sorted(paths) or len(paths) != len(set(paths)):
            raise V18Error(f"ELF {name} paths are not strictly sorted and unique")
        values: list[dict[str, Any]] = []
        for index, item in enumerate(records):
            value = dict(item)
            if tuple(value) != required_fields:
                raise V18Error(f"ELF {name}[{index}] fields changed")
            path = Path(str(value["path"]))
            if (
                not path.is_absolute()
                or path.is_symlink()
                or not path.is_file()
                or path.resolve() != path
                or value["basename"] != path.name
            ):
                raise V18Error(f"ELF {name}[{index}] path identity changed")
            _require_sha(value["sha256"], f"ELF {name}[{index}] SHA")
            if sha256_file(path) != value["sha256"]:
                raise V18Error(f"ELF {name}[{index}] bytes changed")
            if "roles" in value:
                roles = value["roles"]
                if (
                    not isinstance(roles, list)
                    or not roles
                    or roles != sorted(set(str(role) for role in roles))
                ):
                    raise V18Error(f"ELF {name}[{index}] roles changed")
            values.append(value)
        return values

    roots = validate_objects(
        closure["root_objects"], required_fields=root_fields, name="root object"
    )
    shared = validate_objects(
        closure["shared_objects"],
        required_fields=shared_fields,
        name="shared object",
    )
    if (
        len(roots) != int(closure["root_object_count"])
        or canonical_json_sha256(roots) != closure["root_object_set_sha256"]
        or len(shared) != int(closure["shared_object_count"])
        or canonical_json_sha256(shared) != closure["shared_object_set_sha256"]
    ):
        raise V18Error("ELF closure count or set digest changed")
    for field in ("dynamic_loader", "loader_cache"):
        record = closure[field]
        validated = validate_objects(
            [record], required_fields=shared_fields, name=field.replace("_", " ")
        )[0]
        if field == "dynamic_loader" and validated not in shared:
            raise V18Error(f"ELF {field} is outside the shared-object closure")
    if closure["virtual_object_exception"] != ["linux-vdso.so.1"]:
        raise V18Error("ELF virtual-object exception changed")
    loader_environment = closure["loader_environment"]
    if not isinstance(loader_environment, Mapping) or any(
        expected is not None or name in os.environ
        for name, expected in loader_environment.items()
    ):
        raise V18Error("ELF loader environment is not clean")
    external = closure["external_process_environment"]
    if (
        not isinstance(external, Mapping)
        or set(external)
        != {
            "inherit_parent_environment",
            "common_exact",
            "git_exact_extra",
            "pdftotext_exact_extra",
        }
        or external["inherit_parent_environment"] is not False
        or external["common_exact"] != {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
        or external["git_exact_extra"]
        != {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        }
        or external["pdftotext_exact_extra"] != {}
    ):
        raise V18Error("ELF external-process environment contract changed")


def _validate_runtime_environment(lock: Mapping[str, Any]) -> None:
    global _LOCKED_PDFTOTEXT_EXECUTABLE, _LOCKED_TLS_CAFILE
    _validate_elf_closure(lock)
    runtime = lock["runtime"]
    platform_lock = runtime["platform"]
    libc_implementation, libc_version = platform.libc_ver()
    observed_platform = {
        "system": platform.system(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "libc_implementation": libc_implementation,
        "libc_version": libc_version,
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "python_platform_tag": sysconfig.get_platform(),
        "timezone_name": TOKYO.key,
    }
    if any(platform_lock.get(field) != observed for field, observed in observed_platform.items()):
        raise V18Error("runtime platform differs from the preregistered lock")
    zone_candidates = [Path(root) / "Asia/Tokyo" for root in TZPATH]
    zone_hashes = {
        sha256_file(path)
        for path in zone_candidates
        if path.is_file() and not path.is_symlink()
    }
    if platform_lock.get("timezone_zoneinfo_sha256") not in zone_hashes:
        raise V18Error("Asia/Tokyo zoneinfo bytes differ from the runtime lock")

    python_lock = runtime["python"]
    if set(python_lock) != {
        "implementation",
        "version",
        "cache_tag",
        "executable_sha256",
        "stdlib_file_count",
        "stdlib_tree_sha256",
        "stdlib_tree_hash_rule",
        "ssl",
    }:
        raise V18Error("Python runtime-lock schema changed")
    observed_python = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "cache_tag": sys.implementation.cache_tag,
        "executable_sha256": sha256_file(Path(sys.executable).resolve()),
    }
    if any(python_lock.get(key) != observed for key, observed in observed_python.items()):
        raise V18Error("Python runtime differs from the preregistered lock")
    stdlib_root = Path(sysconfig.get_paths()["stdlib"]).resolve(strict=True)
    stdlib_records: list[dict[str, str]] = []
    for observed_path in stdlib_root.rglob("*"):
        relative = observed_path.relative_to(stdlib_root)
        if (
            "site-packages" in relative.parts
            or "dist-packages" in relative.parts
            or "__pycache__" in relative.parts
            or observed_path.suffix in {".pyc", ".pyo"}
        ):
            continue
        if observed_path.is_symlink():
            raise V18Error("Python stdlib contains a symlink")
        if observed_path.is_file():
            stdlib_records.append(
                {"path": relative.as_posix(), "sha256": sha256_file(observed_path)}
            )
    stdlib_records.sort(key=lambda item: item["path"])
    if (
        len(stdlib_records) != int(python_lock["stdlib_file_count"])
        or canonical_json_sha256(stdlib_records)
        != python_lock["stdlib_tree_sha256"]
    ):
        raise V18Error("Python stdlib tree differs from the preregistered lock")

    ssl_lock = python_lock["ssl"]
    if set(ssl_lock) != {
        "openssl_version",
        "openssl_version_info",
        "ssl_module_origin",
        "ssl_extension_file_sha256",
        "dynamic_libssl_file_sha256",
        "dynamic_libcrypto_file_sha256",
        "binding_rule",
        "ca_trust",
    }:
        raise V18Error("Python SSL runtime-lock schema changed")
    ca_trust = ssl_lock["ca_trust"]
    ca_path = _validate_tls_ca_trust(ca_trust)
    ssl_spec = importlib.util.find_spec("_ssl")
    ssl_origin = None if ssl_spec is None else ssl_spec.origin
    ssl_extension_hash = None
    if ssl_origin not in {None, "built-in", "frozen"}:
        ssl_path = Path(str(ssl_origin))
        if ssl_path.is_symlink() or not ssl_path.is_file():
            raise V18Error("registered _ssl extension is missing or symlinked")
        ssl_extension_hash = sha256_file(ssl_path)
    observed_ssl = {
        "openssl_version": ssl.OPENSSL_VERSION,
        "openssl_version_info": list(ssl.OPENSSL_VERSION_INFO),
        "ssl_module_origin": ssl_origin,
        "ssl_extension_file_sha256": ssl_extension_hash,
        "dynamic_libssl_file_sha256": None,
        "dynamic_libcrypto_file_sha256": None,
        "binding_rule": ssl_lock["binding_rule"],
        "ca_trust": dict(ca_trust),
    }
    if observed_ssl != ssl_lock:
        raise V18Error("Python/OpenSSL binding differs from the runtime lock")

    distributions = runtime["distributions"]
    if not isinstance(distributions, list) or [item.get("name") for item in distributions] != sorted(
        item.get("name") for item in distributions
    ):
        raise V18Error("runtime distributions are not name-sorted")
    requirements: list[str] = []
    for locked in distributions:
        if set(locked) != {
            "name",
            "import_name",
            "version",
            "file_count",
            "tree_sha256",
            "module_relative_path",
            "module_file_sha256",
        }:
            raise V18Error("runtime distribution schema changed")
        name = str(locked["name"])
        try:
            distribution = importlib.metadata.distribution(name)
            module = importlib.import_module(str(locked["import_name"]))
        except (importlib.metadata.PackageNotFoundError, ImportError) as exc:
            raise V18Error(f"runtime distribution is unavailable: {name}") from exc
        if distribution.version != locked["version"]:
            raise V18Error(f"runtime distribution version changed: {name}")
        tree = _distribution_tree(distribution)
        if len(tree) != int(locked["file_count"]) or canonical_json_sha256(tree) != locked[
            "tree_sha256"
        ]:
            raise V18Error(f"runtime distribution tree changed: {name}")
        module_path = Path(str(getattr(module, "__file__", "")))
        module_relative = _runtime_relative_path(
            locked["module_relative_path"], f"{name}.module_relative_path"
        )
        if (
            not module_path.is_file()
            or module_path.is_symlink()
            or not module_path.as_posix().endswith(f"/{module_relative}")
            or sha256_file(module_path) != locked["module_file_sha256"]
        ):
            raise V18Error(f"runtime import root changed: {name}")
        requirements.append(f"{name}=={locked['version']}")
    if requirements != runtime["locked_requirements"]:
        raise V18Error("runtime requirements differ from distribution bindings")
    requirements_bytes = ("\n".join(requirements) + "\n").encode("utf-8")
    if hashlib.sha256(requirements_bytes).hexdigest() != runtime[
        "locked_requirements_sha256"
    ]:
        raise V18Error("runtime requirement-set hash changed")

    thread_environment = runtime["thread_environment"]
    if any(os.environ.get(name) != expected for name, expected in thread_environment.items()):
        raise V18Error("numeric thread environment differs from the runtime lock")
    observed_native = _normalised_native_threadpools(hash_libraries=True)
    # Startup capacity is acquisition provenance, not a portable identity
    # gate.  Exact libraries/bytes and their one-thread state inside registered
    # numeric contexts are the deterministic boundary.
    without_capacity = lambda rows: [
        {key: child for key, child in item.items() if key != "num_threads"}
        for item in rows
    ]
    if without_capacity(observed_native) != without_capacity(runtime["native_threadpools"]):
        raise V18Error("native numeric libraries differ from the runtime lock")

    pdftotext_lock = lock["pdftotext"]
    if set(pdftotext_lock) != {
        "implementation",
        "executable_path",
        "executable_basename",
        "version",
        "version_output_sha256",
        "executable_sha256",
        "verification_rule",
    } or pdftotext_lock["implementation"] != "Poppler pdftotext":
        raise V18Error("pdftotext runtime-lock schema changed")
    executable = _elf_role_path(lock, "pdftotext_executable")
    if executable.is_symlink() or not executable.is_file():
        raise V18Error("pdftotext executable is missing or symlinked")
    if (
        str(executable) != pdftotext_lock["executable_path"]
        or executable.name != pdftotext_lock["executable_basename"]
        or executable.resolve() != executable
        or sha256_file(executable) != pdftotext_lock["executable_sha256"]
    ):
        raise V18Error("pdftotext executable bytes differ from the runtime lock")
    completed = subprocess.run(
        [str(executable), "-v"],
        check=False,
        capture_output=True,
        env=_subprocess_environment(),
        shell=False,
    )
    if completed.returncode != 0 or completed.stdout:
        raise V18Error("pdftotext version probe changed behavior")
    if hashlib.sha256(completed.stderr).hexdigest() != pdftotext_lock[
        "version_output_sha256"
    ]:
        raise V18Error("pdftotext version output differs from the runtime lock")
    match = re.search(rb"^pdftotext version ([^\n]+)$", completed.stderr, re.MULTILINE)
    if match is None or match.group(1).decode("ascii") != pdftotext_lock["version"]:
        raise V18Error("pdftotext version differs from the runtime lock")
    _LOCKED_PDFTOTEXT_EXECUTABLE = executable
    _validate_git_executable(lock)
    _LOCKED_TLS_CAFILE = ca_path
    _validate_startup_and_module_closure(
        lock,
        phase="runtime startup",
        initialise=True,
    )


def validate_runtime_lock(
    path: str | Path = RUNTIME_LOCK,
    *,
    strict_environment: bool = False,
) -> tuple[dict[str, Any], str]:
    """Validate immutable dependencies; optionally validate the executing host."""

    global _STRICT_RUNTIME_ACTIVE, _STRICT_RUNTIME_VERIFIED_AT
    global _LOCKED_GIT_EXECUTABLE, _LOCKED_PDFTOTEXT_EXECUTABLE, _LOCKED_TLS_CAFILE
    global _LIVE_MODULE_CLOSURE_STATE
    if strict_environment:
        _STRICT_RUNTIME_ACTIVE = False
        _STRICT_RUNTIME_VERIFIED_AT = None
        _LOCKED_GIT_EXECUTABLE = None
        _LOCKED_PDFTOTEXT_EXECUTABLE = None
        _LOCKED_TLS_CAFILE = None
        _LIVE_MODULE_CLOSURE_STATE = None
    runtime_path = Path(path)
    if not runtime_path.is_file() or runtime_path.is_symlink():
        raise V18Error("v1.8 runtime lock is missing or symlinked")
    observed_file_hash = sha256_file(runtime_path)
    if observed_file_hash != RUNTIME_LOCK_SHA256:
        raise V18Error(
            "v1.8 runtime-lock file hash mismatch: "
            f"expected {RUNTIME_LOCK_SHA256}, got {observed_file_hash}"
        )
    value = read_json(runtime_path)
    if set(value) != {
        "schema_version",
        "lock_id",
        "registered_on",
        "scope",
        "project_files",
        "project_file_set_sha256",
        "project_file_set_hash_rule",
        "runtime",
        "elf_closure",
        "pdftotext",
        "git",
        "canonical_json_contract",
        "runtime_lock_self_hash_rule",
        "runtime_lock_self_sha256",
    }:
        raise V18Error("runtime-lock top-level schema changed")
    if (
        value["schema_version"] != 1
        or value["lock_id"] != "model_v18_runtime_lock_20260805_a2"
        or value["registered_on"] != "2026-08-05"
        or value["canonical_json_contract"] != CANONICAL_JSON_CONTRACT
    ):
        raise V18Error("runtime-lock identity changed")
    _runtime_project_files(value)
    expected_self_hash = canonical_json_sha256(
        value, exclude_fields={"runtime_lock_self_sha256"}
    )
    if value["runtime_lock_self_sha256"] != expected_self_hash:
        raise V18Error("runtime-lock self-hash mismatch")
    if strict_environment:
        _validate_runtime_environment(value)
        _STRICT_RUNTIME_ACTIVE = True
        _STRICT_RUNTIME_VERIFIED_AT = datetime.now(TOKYO)
    return value, observed_file_hash


def _runtime_verified_timestamp(
    value: Any | None = None,
    *,
    allow_historical_replay: bool = False,
) -> datetime:
    """Return operational evidence time or an explicitly injected test time."""

    if _STRICT_RUNTIME_ACTIVE:
        if _STRICT_RUNTIME_VERIFIED_AT is None:  # pragma: no cover - invariant
            raise V18Error("strict runtime verification timestamp is unavailable")
        if allow_historical_replay and value is not None:
            return _timestamp(value, "runtime_lock_verified_at")
        if value is not None and _timestamp(value, "runtime_lock_verified_at") != (
            _STRICT_RUNTIME_VERIFIED_AT
        ):
            raise V18Error("canonical operation cannot inject a runtime verification time")
        return _STRICT_RUNTIME_VERIFIED_AT
    if value is None:
        # Pure deterministic helpers may run in a test environment.  They bind
        # the immutable lock/dependency closure, but may not write canonical
        # operational artifacts because the CLI always activates strict mode.
        validate_runtime_lock(strict_environment=False)
        return datetime.now(TOKYO)
    return _timestamp(value, "runtime_lock_verified_at")


def _operation_timestamp(value: Any | None, name: str) -> datetime:
    if _STRICT_RUNTIME_ACTIVE:
        if value is not None:
            raise V18Error(f"canonical operation cannot inject {name}")
        return datetime.now(TOKYO)
    return datetime.now(TOKYO) if value is None else _timestamp(value, name)


def _require_active_a2_rehearsal_contract(
    value: _A2RehearsalContract | None,
) -> _A2RehearsalContract:
    if (
        value is None
        or not isinstance(value, _A2RehearsalContract)
        or value is not _ACTIVE_A2_REHEARSAL_CONTRACT
    ):
        raise V18Error("A2 rehearsal capability is absent, forged, or expired")
    return value


@contextmanager
def _numeric_execution(
    *, _a2_rehearsal_contract: _A2RehearsalContract | None = None
) -> Iterable[None]:
    """Run registered numerical work with one native thread and verify it."""

    if _a2_rehearsal_contract is not None:
        rehearsal = _require_active_a2_rehearsal_contract(
            _a2_rehearsal_contract
        )
        if rehearsal.numeric_threadpool_limit != 1:
            raise V18Error("A2 rehearsal numeric thread limit changed")
        # The locked runtime and loaded native backends are validated by the
        # untimed capability preflight and postflight.  The timed seam performs
        # only the same one-thread numerical dispatch; it must not reopen the
        # runtime lock or any project/module file.
        with threadpoolctl.threadpool_limits(
            limits=rehearsal.numeric_threadpool_limit
        ):
            yield
        return

    lock = read_json(RUNTIME_LOCK)
    expected_limit = int(lock["runtime"]["numeric_execution_contract"]["threadpoolctl_limit"])
    if expected_limit != 1:
        raise V18Error("runtime numeric thread limit changed")

    def verify() -> None:
        observed = _normalised_native_threadpools(hash_libraries=False)
        if any(item["num_threads"] != expected_limit for item in observed):
            raise V18Error("numeric backend escaped the one-thread execution boundary")
        if _STRICT_RUNTIME_ACTIVE:
            expected = [
                {
                    key: (expected_limit if key == "num_threads" else item[key])
                    for key in item
                    if key != "library_sha256"
                }
                for item in lock["runtime"]["native_threadpools"]
            ]
            if observed != expected:
                raise V18Error("numeric execution loaded an unregistered native backend")

    with threadpoolctl.threadpool_limits(limits=expected_limit):
        verify()
        _validate_startup_and_module_closure(phase="numeric phase entry")
        try:
            yield
        finally:
            verify()
            _validate_startup_and_module_closure(phase="numeric phase exit")


def _artifact_hashes(bindings: Mapping[str, tuple[str, str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, (relative, expected) in bindings.items():
        observed = sha256_file(ROOT / relative)
        if observed != expected:
            raise V18Error(
                f"historical {name} hash mismatch: expected {expected}, got {observed}"
            )
        result[name] = observed
    return result


def validate_historical_bindings() -> dict[str, Any]:
    result = {"v16": _artifact_hashes(V16_BINDINGS), "v17": _artifact_hashes(V17_BINDINGS)}
    for version, bindings in (("v16", V16_BINDINGS), ("v17", V17_BINDINGS)):
        frozen = read_json(ROOT / bindings["result"][0])
        if frozen.get("status") != "selection_rejected_all_candidates":
            raise V18Error(f"{version} frozen result status changed")
        if frozen.get("authority", {}).get("orders_allowed") is not False:
            raise V18Error(f"{version} frozen result authorizes orders")
    result["binding_set_sha256"] = canonical_json_sha256(result)
    return result


def _validate_required_array_uniqueness(value: Any, *, path: str = "protocol") -> None:
    """Reject duplicate members before any required-array set comparison."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if isinstance(child, list) and (
                key == "required_fields"
                or key.startswith("required_")
                or key.endswith("_required_fields")
                or key.endswith("_required_paths")
            ):
                identities = [canonical_json_bytes(item) for item in child]
                if len(identities) != len(set(identities)):
                    raise V18Error(
                        f"duplicate member in ordered required array: {child_path}"
                    )
            _validate_required_array_uniqueness(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_required_array_uniqueness(child, path=f"{path}[{index}]")


def _validate_additional_test_artifact_paths(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise V18Error("additional test artifact paths must be an ordered array")
    paths = tuple(value)
    if paths != ADDITIONAL_TEST_ARTIFACT_PATHS:
        raise V18Error("additional test artifact path registry changed")
    if len(paths) != len(set(paths)) or "tests/test_model_v18_shoulder_state.py" in paths:
        raise V18Error("additional test artifact paths overlap or duplicate")
    return paths


def _protocol_additional_test_artifact_paths(
    protocol: Mapping[str, Any],
) -> tuple[str, ...]:
    activation = protocol.get("activation", {})
    paths = _validate_additional_test_artifact_paths(
        activation.get("preregistration_commit", {}).get(
            "additional_test_artifact_paths"
        )
    )
    payload_contract = activation.get("payload", {})
    if "additional_test_artifacts" not in payload_contract.get(
        "required_fields", ()
    ):
        raise V18Error("activation payload omits additional test artifacts")
    if "additional_test_artifacts" in payload_contract.get("fixed_values", {}):
        raise V18Error("protocol must not bind cyclic additional test hashes")
    return paths


def _validate_additional_test_artifacts(
    value: Any,
    *,
    expected_paths: Sequence[str] = ADDITIONAL_TEST_ARTIFACT_PATHS,
    verify_worktree: bool,
) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or len(value) != len(expected_paths):
        raise V18Error("additional test artifact records changed")
    records: list[dict[str, str]] = []
    for expected_path, item in zip(expected_paths, value, strict=True):
        if not isinstance(item, Mapping) or tuple(item) != ADDITIONAL_TEST_ARTIFACT_FIELDS:
            raise V18Error("additional test artifact record schema changed")
        path = item.get("path")
        digest = item.get("sha256")
        if path != expected_path or not isinstance(digest, str) or SHA256_RE.fullmatch(
            digest
        ) is None:
            raise V18Error("additional test artifact path/hash changed")
        if verify_worktree:
            test_path = ROOT / expected_path
            if not test_path.is_file() or sha256_file(test_path) != digest:
                raise V18Error(
                    f"additional test artifact working-tree bytes changed: {expected_path}"
                )
        records.append({"path": expected_path, "sha256": digest})
    if len({item["path"] for item in records}) != len(records):
        raise V18Error("additional test artifact records contain duplicate paths")
    return tuple(records)


def _additional_test_artifacts_from_worktree(
    paths: Sequence[str] = ADDITIONAL_TEST_ARTIFACT_PATHS,
) -> list[dict[str, str]]:
    if tuple(paths) != ADDITIONAL_TEST_ARTIFACT_PATHS:
        raise V18Error("additional test artifact construction path registry changed")
    records: list[dict[str, str]] = []
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise V18Error(f"additional test artifact is missing: {relative}")
        records.append({"path": relative, "sha256": sha256_file(path)})
    return list(
        _validate_additional_test_artifacts(
            records, expected_paths=paths, verify_worktree=True
        )
    )


def _validate_a2_protocol_contract(protocol: Mapping[str, Any]) -> None:
    """Bind the finite A2 correction, evidence schemas, and operational surface."""

    correction = protocol.get("a2_correction", {})
    if set(correction) != {
        "correction_id",
        "superseded_unactivated_preregistration_commit_sha",
        "superseded_protocol_sha256",
        "superseded_runtime_lock_sha256",
        "superseded_state",
        "reasons",
        "unchanged",
        "activation_rule",
    } or {
        "correction_id": correction.get("correction_id"),
        "superseded_unactivated_preregistration_commit_sha": correction.get(
            "superseded_unactivated_preregistration_commit_sha"
        ),
        "superseded_protocol_sha256": correction.get(
            "superseded_protocol_sha256"
        ),
        "superseded_runtime_lock_sha256": correction.get(
            "superseded_runtime_lock_sha256"
        ),
    } != {
        "correction_id": "model_v18_shoulder_state_a2_20260805",
        "superseded_unactivated_preregistration_commit_sha": (
            "0c2f1633ca7f8584f6550b1d858402e20fc55d6b"
        ),
        "superseded_protocol_sha256": (
            "c623fabfa8e94381bce27d359cefc6e51a9a80f1c18f62f6098cfdfc8e9f6112"
        ),
        "superseded_runtime_lock_sha256": (
            "95a867e2f8f187528a7ba3d24f4db4f1bf0964531b6e0e2b85d88d0c758f1e32"
        ),
    }:
        raise V18Error("A2 correction identity changed")
    if "never activated" not in str(correction["superseded_state"]) or tuple(
        correction["reasons"]
    ) != (
        "zstandard 0.25.0/backend_c transitive runtime closure was missing",
        "one full raw parse exceeded the 11-minute pre-open publication window",
    ):
        raise V18Error("A2 correction history changed")

    activation = protocol.get("activation", {})
    payload = activation.get("payload", {})
    receipt = activation.get("receipt", {})
    payload_fixed = payload.get("fixed_values", {})
    receipt_fixed = receipt.get("fixed_values", {})
    required_prereg_paths = tuple(
        activation.get("preregistration_commit", {}).get("required_paths", ())
    )
    if (
        activation.get("status")
        != "pending_three_commit_activation_a2_correction"
        or activation.get("not_before_session") != "2026-08-06"
        or activation.get(
            "expected_earliest_session_if_receipt_workflow_is_observed_before_cutoff"
        )
        != "2026-08-06"
        or payload_fixed.get("activation_id")
        != "model_v18_shoulder_state_activation_a2_20260805"
        or payload_fixed.get("not_before_session") != "2026-08-06"
        or receipt_fixed.get("activation_id")
        != "model_v18_shoulder_state_activation_a2_20260805"
        or receipt_fixed.get("not_before_session") != "2026-08-06"
        or payload_fixed.get("iteration_report_path")
        != "research/model_v18_postmerge_hypothesis_iteration_report.md"
        or payload_fixed.get("validation_report_path") != "VALIDATION.md"
        or payload_fixed.get("rehearsal_path")
        != "research/model_v18_a2_rehearsal.py"
        or "research/model_v18_postmerge_hypothesis_iteration_report.md"
        not in required_prereg_paths
        or "VALIDATION.md" not in required_prereg_paths
        or "research/model_v18_a2_rehearsal.py" not in required_prereg_paths
        or "iteration_report_path" not in payload.get("required_fields", ())
        or "iteration_report_sha256" not in payload.get("required_fields", ())
        or "validation_report_path" not in payload.get("required_fields", ())
        or "validation_report_sha256" not in payload.get("required_fields", ())
        or "rehearsal_path" not in payload.get("required_fields", ())
        or "rehearsal_sha256" not in payload.get("required_fields", ())
        or "predictor_cache_anchor" not in payload.get("required_fields", ())
    ):
        raise V18Error("A2 activation/preregistration contract changed")
    periods = protocol.get("periods", {})
    expected_period = periods.get(
        "expected_if_first_counted_session_is_2026_08_06", {}
    )
    if periods.get("not_before_session") != "2026-08-06" or expected_period != {
        "session_120": "2027-02-03",
        "sessions_through_2027_01_29": 117,
        "terminal_session": "2027-02-26",
        "terminal_scheduled_sessions": 135,
        "represented_calendar_months": 7,
        "early_slice_sessions": 67,
        "late_slice_sessions": 68,
        "required_positive_months_net40": 6,
        "required_executed_days": 122,
    }:
        raise V18Error("A2 not-before/terminal example changed")

    a2 = protocol.get("a2_operational_repair_contract", {})
    expected_schema_arrays = {
        "parsed_shard_manifest_required_fields": PARSED_SHARD_MANIFEST_FIELDS,
        "parsed_shard_binding_required_fields": PARSED_SHARD_BINDING_FIELDS,
        "model_price_snapshot_manifest_required_fields": MODEL_PRICE_SNAPSHOT_FIELDS,
        "g0_target_cache_manifest_required_fields": G0_CACHE_MANIFEST_FIELDS,
        "cache_anchor_manifest_required_fields": CACHE_ANCHOR_FIELDS,
        "cache_anchor_summary_required_fields": CACHE_ANCHOR_SUMMARY_FIELDS,
        "compact_consumer_equivalence_receipt_required_fields": (
            COMPACT_CONSUMER_EQUIVALENCE_RECEIPT_FIELDS
        ),
        "month_source_manifest_required_fields": MONTH_SOURCE_MANIFEST_FIELDS,
    }
    if a2.get("schema_version") != 1 or a2.get("cache_contract_id") != (
        PREDICTOR_CACHE_CONTRACT_ID
    ):
        raise V18Error("A2 predictor-cache contract identity changed")
    for name, expected in expected_schema_arrays.items():
        if tuple(a2.get(name, ())) != expected:
            raise V18Error(f"A2 protocol schema differs from runner: {name}")
    if (
        tuple(a2.get("compact_model_price_columns", ())) != MODEL_PRICE_COLUMNS
        or tuple(a2.get("canonical_cli_order", ()))
        != (
            "prepare-day",
            "publish-checkpoint",
            "decide",
            "finalize-terminal",
            "evaluate",
        )
        or tuple(a2.get("forbidden_split_mutation_commands", ()))
        != (
            "prepare-source-manifest",
            "prepare-month",
            "prepare-top2",
            "prepare-state",
            "prepare-checkpoint",
            "prepare-outcome-manifest",
            "attach-outcomes",
            "close-month",
        )
        or tuple(a2.get("root_arguments", ()))
        != (
            "--predictor-raw-store-root",
            "--predictor-derived-store-root",
            "--outcome-raw-store-root",
            "--checkpoint-core-store-root",
        )
    ):
        raise V18Error("A2 operational surface changed")
    timing = a2.get("timing_activation_gate", {})
    if timing != {
        "cold_runs": 3,
        "intramonth_local_e2e_seconds_max": 120,
        "month_boundary_local_e2e_seconds_max": 300,
        "measured_compact_e2e_seconds": 69.1,
        "measured_peak_gib": 3.815,
        "fixed_synthetic_target_session": "2026-08-05",
        "untimed_reference_input_kind": "full31_reference",
        "month_boundary_input_kind": "month_boundary_compact",
        "intramonth_input_kind": "intramonth_fold_reuse_upper_bound_proxy",
        "preflight_api": "prepare_a2_nonauthority_rehearsal_contract",
        "runner_api": "build_a2_nonauthority_rehearsal_day",
        "postflight_api": "validate_a2_nonauthority_rehearsal_postflight",
        "authority": False,
        "comparison_rule": "The input-kind/snapshot/suffix/timing envelope is documentary. Only the nested comparison object is exact-compared: source-manifest/source-set/parsed-shard-set identities, model-price CSV/semantics, one full G0 digest, exact pre-month training semantic/count, target-cache bytes/semantics, real fold/bundle bytes, real score bytes/semantics, IEEE top-two scores, and panel call count. Reference and boundary calls freshly compute every proof field and bind them into a process-local, nonserialisable fold token together with the exact canonical full-prefix CSV byte count/SHA, row count, target/latest sessions, and source/shard identities. The intramonth proxy must freshly strict-decode its compact snapshot, project its suffix, canonical-encode the merged full prefix once, and require every exact identity to equal that token before reusing only the sealed model-semantic/full-G0/training-semantic proof fields. It must still freshly build exactly one panel, encode/round-trip the target cache, recompute current fold row/target/feature hashes, validate the reused fold/bundle, and score; it may not fit inside its timed call. Any prefix or token mismatch aborts before proof reuse.",
        "rule": "A slow full-raw reference build is outside the daily timing gate and fixes exact comparison hashes. The untimed preflight creates one process-local opaque capability after strict runtime/project/canonical-authority-absence validation; timed calls accept only that exact capability and perform no canonical repository read/write, network call, calendar lookup, or module-closure scan; untimed postflight revalidates the same closure and revokes the capability. Production intramonth preparation uses the same causal optimization: after exact retained month-source, snapshot data-byte/canonical-decode/self/source/shard bindings are validated, the already sealed model/training semantic fields may be reused without caller-supplied frames or hashes, while current fold row/target/feature hashes and target score remain fresh; boundary creation and terminal independently compute all semantics. Three fresh compact nonauthority runs for each registered boundary and intramonth case must exact-match that reference; any compact run over its budget forbids activation.",
    }:
        raise V18Error("A2 cold timing gate changed")

    source_contract = protocol.get("source_contract", {})
    provenance = source_contract.get("raw_source_provenance", {})
    provenance_policy = source_contract.get("raw_source_provenance_policy", {})
    if provenance != _raw_source_provenance_envelope() or (
        provenance_policy.get("caveat") != RAW_SOURCE_PROVENANCE_CAVEAT
        or len(provenance_policy.get("trusted_claims", ())) != 2
        or len(provenance_policy.get("nonclaims", ())) != 4
        or not isinstance(provenance_policy.get("upgrade_rule"), str)
        or not provenance_policy["upgrade_rule"]
    ):
        raise V18Error("A2 manual source-provenance contract changed")
    if tuple(a2.get("result_input_fields", ())) != tuple(
        protocol.get("result_contract", {}).get("required_input_fields", ())
    ):
        raise V18Error("A2 result-input registry changed")
    c00_contract = protocol.get("c00_contract", {})
    if tuple(
        c00_contract.get("fold_model_bundle_contract", {}).get(
            "required_fields", ()
        )
    ) != (
        "schema_version",
        "target_month",
        "created_at",
        "runtime_lock_sha256",
        "runtime_lock_verified_at",
        "input_feature_order",
        "input_feature_dtype",
        "input_feature_shape",
        "transformed_feature_order",
        "transformed_feature_dtype",
        "transformed_feature_shape",
        "imputer_strategy",
        "imputer_add_indicator",
        "imputer_keep_empty_features",
        "imputer_statistics",
        "imputer_indicator_features",
        "scaler_with_mean",
        "scaler_with_std",
        "scaler_mean",
        "scaler_scale",
        "ridge_alpha",
        "ridge_fit_intercept",
        "ridge_coef",
        "ridge_intercept",
        "protocol_sha256",
        "runner_sha256",
        "python_version",
        "numpy_version",
        "scikit_learn_version",
        "canonical_json_contract",
        "fold_model_bundle_sha256",
    ) or tuple(c00_contract.get("fold_manifest_required_fields", ())) != (
        "schema_version",
        "target_month",
        "fit_started_at",
        "fit_completed_at",
        "sealed_at",
        "runtime_lock_sha256",
        "runtime_lock_verified_at",
        "training_first_session",
        "training_last_session",
        "training_session_count",
        "training_row_identity_sha256",
        "training_target_sha256",
        "feature_matrix_sha256",
        "feature_names",
        "month_source_manifest_path",
        "month_source_manifest_sha256",
        "training_source_set_sha256",
        "training_parsed_shard_set_sha256",
        "training_g0_panel_semantic_sha256",
        "ridge_alpha",
        "fold_model_bundle_path",
        "fold_model_bundle_schema_version",
        "fold_model_bundle_file_sha256",
        "fold_model_bundle_sha256",
        "input_feature_order_sha256",
        "transformed_feature_order_sha256",
        "imputer_statistics_sha256",
        "imputer_indicator_features_sha256",
        "scaler_mean_sha256",
        "scaler_scale_sha256",
        "ridge_coef_sha256",
        "ridge_intercept",
        "universe_contract_sha256",
        "protocol_sha256",
        "runner_sha256",
        "python_version",
        "numpy_version",
        "pandas_version",
        "scikit_learn_version",
        "canonical_json_contract",
        "fold_manifest_sha256",
    ):
        raise V18Error("A2 fold/bundle protocol schema changed")
    state_contract = protocol.get("state_contract", {})
    if tuple(
        state_contract.get("target_month_state_manifest_required_fields", ())
    ) != (
        "schema_version",
        "target_month",
        "created_at",
        "three_prior_calendar_months",
        "three_complete_pair_day_counts",
        "three_month_medians_pct",
        "state_available",
        "state_value_pct",
        "selected_source_rank",
        "c00_fold_manifest_sha256",
        "fold_model_bundle_file_sha256",
        "protocol_sha256",
        "activation_payload_sha256",
        "activation_receipt_sha256",
        "state_manifest_sha256",
    ) or tuple(
        state_contract.get("completed_month_record_required_fields", ())
    ) != COMPLETED_MONTH_FIELDS:
        raise V18Error("A2 state/completed-month protocol schema changed")


def validate_protocol(path: str | Path = PROTOCOL) -> tuple[dict[str, Any], str]:
    observed = sha256_file(path)
    if PROTOCOL_SHA256 == "__PENDING_PROTOCOL_SHA256__":
        raise V18Error("v1.8 protocol SHA has not been pinned in the runner")
    if observed != PROTOCOL_SHA256:
        raise V18Error(f"v1.8 protocol hash mismatch: expected {PROTOCOL_SHA256}, got {observed}")
    protocol = read_json(path)
    _validate_required_array_uniqueness(protocol)
    if protocol.get("schema_version") != 1 or protocol.get("protocol_id") != PROTOCOL_ID:
        raise V18Error("v1.8 protocol identity changed")
    if (
        protocol.get("repository") != "rokuroku-066/TSE-Session-Ranker"
        or protocol.get("branch") != GITHUB_BRANCH
    ):
        raise V18Error("v1.8 repository/branch changed")
    _validate_a2_protocol_contract(protocol)
    activation_contract = protocol.get("activation", {})
    _protocol_additional_test_artifact_paths(protocol)
    observation_contract = activation_contract.get("github_observation_contract", {})
    workflow_observation_contract = activation_contract.get(
        "github_workflow_run_observation_contract", {}
    )
    transport = observation_contract.get("transport", {})
    if (
        tuple(observation_contract.get("required_fields", ()))
        != GITHUB_OBSERVATION_FIELDS
        or tuple(
            observation_contract.get("canonical_projection_required_fields", ())
        )
        != GITHUB_PROJECTION_FIELDS
        or observation_contract.get("fixed_values")
        != {
            "schema_version": 1,
            "repository": "rokuroku-066/TSE-Session-Ranker",
            "http_status": 200,
            "canonical_json_contract": CANONICAL_JSON_CONTRACT,
        }
        or tuple(workflow_observation_contract.get("required_fields", ()))
        != GITHUB_WORKFLOW_OBSERVATION_FIELDS
        or tuple(
            workflow_observation_contract.get(
                "canonical_projection_required_fields", ()
            )
        )
        != GITHUB_WORKFLOW_PROJECTION_FIELDS
        or workflow_observation_contract.get("fixed_values")
        != {
            "schema_version": 1,
            "repository": "rokuroku-066/TSE-Session-Ranker",
            "http_status": 200,
            "canonical_json_contract": CANONICAL_JSON_CONTRACT,
        }
        or set(transport)
        != {
            "implementation",
            "scheme",
            "host",
            "fixed_request_headers",
            "token_rule",
            "tls_trust_rule",
            "response_rule",
        }
        or transport.get("scheme") != "https"
        or transport.get("host") != "api.github.com"
        or not isinstance(transport.get("tls_trust_rule"), str)
        or not transport["tls_trust_rule"]
        or transport.get("fixed_request_headers")
        != {
            "Accept": GITHUB_API_ACCEPT,
            "Accept-Encoding": "identity",
            "User-Agent": GITHUB_API_USER_AGENT,
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }
        or tuple(
            activation_contract.get("receipt_commit_observation", {}).get(
                "required_runtime_fields", ()
            )
        )
        != RECEIPT_OBSERVATION_RUNTIME_FIELDS
    ):
        raise V18Error("v1.8 GitHub observation contract changed")
    authority = protocol.get("authority", {})
    if authority.get("analysis_type") != "genuinely_later_forward_shadow_development":
        raise V18Error("v1.8 authority analysis_type changed")
    for field in ("production_promotion_allowed", "production_model_changed", "orders_allowed"):
        if authority.get(field) is not False:
            raise V18Error(f"v1.8 authority.{field} must be false")
    if authority.get("candidate_variants") != 1:
        raise V18Error("v1.8 must have one candidate")
    controls = tuple(item.get("id") for item in protocol.get("controls", []))
    if controls != (C00_ID, C02_ID):
        raise V18Error("v1.8 control registry changed")
    candidate = protocol.get("candidate", {})
    if candidate.get("id") != CANDIDATE_ID or candidate.get("family_size") != 1:
        raise V18Error("v1.8 candidate registry changed")
    state = protocol.get("state_contract", {})
    if state.get("minimum_complete_pairs_per_month") != MIN_COMPLETE_PAIRS:
        raise V18Error("v1.8 monthly pair minimum changed")
    if state.get("lookback_calendar_months") != STATE_MONTHS:
        raise V18Error("v1.8 state lookback changed")
    if state.get("decision_rule") != {
        "state_value_pct_greater_than_zero": 1,
        "state_value_pct_less_than_zero": 2,
        "state_value_pct_exactly_zero": None,
        "any_required_month_unavailable": None,
    }:
        raise V18Error("v1.8 state direction rule changed")
    checkpoint = protocol.get("daily_preopen_checkpoint_contract", {})
    sealed_core = checkpoint.get("sealed_core_store", {})
    checkpoint_workflow = checkpoint.get(
        "checkpoint_workflow_run_observation_contract", {}
    )
    git_data = checkpoint.get("git_data_api_publication_contract", {})
    git_operations = git_data.get("ordered_operations_per_role", [])
    daily_failures = protocol.get("daily_decision_failure_reason_contract", {})
    decision_values = (
        "selected_rank1",
        "selected_rank2",
        "cash_state_zero",
        "cash_state_unavailable",
    )
    if (
        checkpoint.get("proposal_directory")
        != "research/model_v18_shoulder_state_checkpoint_proposals"
        or tuple(checkpoint.get("proposal_roles", ())) != CHECKPOINT_ROLES
        or tuple(checkpoint.get("proposal_required_fields", ()))
        != CHECKPOINT_PROPOSAL_FIELDS
        or tuple(checkpoint.get("decision_binding_fields", ()))
        != CHECKPOINT_BINDING_FIELDS
        or tuple(checkpoint.get("checkpoint_resolution_values", ()))
        != CHECKPOINT_RESOLUTIONS
        or tuple(checkpoint.get("checkpoint_resolution_reason_values", ()))
        != CHECKPOINT_RESOLUTION_REASONS
        or sealed_core.get("cli_argument") != "--checkpoint-core-store-root"
        or sealed_core.get("logical_prefix") != CHECKPOINT_CORE_OBJECT_PREFIX
        or tuple(sealed_core.get("required_fields", ()))
        != (
            "schema_version",
            "target_session",
            "checkpoint_role",
            "decision_sequence_number",
            "previous_decision_record_sha256",
            "nonce_hex",
            "decision_core",
            "decision_core_sha256",
        )
        or tuple(sealed_core.get("decision_core_required_fields", ()))
        != CHECKPOINT_CORE_FIELDS
        or "exact 16384-byte binary envelope"
        not in str(sealed_core.get("serialization_and_commitment_rule", ""))
        or tuple(daily_failures.get("allowed_nonnull_values", ()))
        != DAILY_FAILURE_REASONS
        or tuple(protocol.get("append_only_artifacts", {}).get("decision_values", ()))
        != decision_values
        or tuple(daily_failures.get("decision_reason_map", {})) != decision_values
        or tuple(checkpoint_workflow.get("required_fields", ()))
        != GITHUB_WORKFLOW_OBSERVATION_FIELDS
        or tuple(
            checkpoint_workflow.get("canonical_projection_required_fields", ())
        )
        != GITHUB_WORKFLOW_PROJECTION_FIELDS
        or checkpoint_workflow.get("terminal_conclusion_values")
        != [
            "success",
            "failure",
            "neutral",
            "cancelled",
            "skipped",
            "timed_out",
            "action_required",
            "stale",
            "startup_failure",
        ]
        or set(checkpoint.get("cli_contract", {}))
        != {"prepare", "publish", "decide", "terminal"}
        or git_data.get("fixed_repository") != GITHUB_REPOSITORY
        or git_data.get("fixed_branch") != GITHUB_BRANCH
        or git_data.get("fixed_ref") != GITHUB_REF
        or tuple(git_data.get("operation_required_fields", ()))
        != (
            "operation_id",
            "method",
            "endpoint",
            "expected_status",
            "request_body_fields",
        )
        or not isinstance(git_operations, list)
        or tuple(item.get("operation_id") for item in git_operations)
        != GIT_DATA_OPERATION_IDS
        or tuple(item.get("method") for item in git_operations)
        != GIT_DATA_OPERATION_METHODS
        or tuple(item.get("expected_status") for item in git_operations)
        != GIT_DATA_OPERATION_STATUSES
        or any(
            set(item)
            != {
                "operation_id",
                "method",
                "endpoint",
                "expected_status",
                "request_body_fields",
            }
            or not isinstance(item.get("request_body_fields"), list)
            for item in git_operations
        )
    ):
        raise V18Error("v1.8 daily checkpoint contract changed")
    periods = protocol.get("periods", {})
    if periods.get("minimum_scheduled_sessions") != MIN_FORWARD_SESSIONS:
        raise V18Error("v1.8 forward session minimum changed")
    if periods.get("minimum_distinct_calendar_months") != MIN_FORWARD_MONTHS:
        raise V18Error("v1.8 forward month minimum changed")
    if protocol.get("common_price_features") != list(G0_FEATURES):
        raise V18Error("v1.8 G0 feature registry changed")
    evaluation = protocol.get("evaluation", {})
    if evaluation.get("candidate_variants") != 1:
        raise V18Error("v1.8 evaluation family changed")
    if tuple(float(item) for item in evaluation.get("costs_bps", [])) != COSTS_BPS:
        raise V18Error("v1.8 costs changed")
    expected_gates = {
        "net40_mean_positive": True,
        "net40_median_positive": True,
        "net60_mean_positive": True,
        "both_fixed_slices_net40_positive": True,
        "positive_months_net40_at_least_ceil_75pct": True,
        "top4_days_removed_net40_positive": True,
        "top5_profit_codes_cash_net40_positive": True,
        "paired_point_delta_vs_C00_net40_positive": True,
        "paired_one_sided_90_lower_vs_C00_nonnegative": True,
        "unique_codes_at_least": 40,
        "maximum_code_share_at_most": 0.05,
        "top10_code_share_at_most": 0.25,
        "executed_days_at_least_ceil_90pct_scheduled": True,
        "executed_slot_fraction_at_least": 0.8,
    }
    if evaluation.get("selection_gate_all_required") != expected_gates:
        raise V18Error("v1.8 gate registry changed")
    bootstrap = evaluation.get("bootstrap", {})
    if (
        bootstrap.get("block_length_sessions") != BOOTSTRAP_BLOCK_LENGTH
        or bootstrap.get("samples") != BOOTSTRAP_SAMPLES
        or bootstrap.get("random_state") != BOOTSTRAP_RANDOM_STATE
        or not math.isclose(
            float(bootstrap.get("individual_one_sided_confidence", -1)),
            BOOTSTRAP_CONFIDENCE,
            abs_tol=1e-15,
            rel_tol=0,
        )
    ):
        raise V18Error("v1.8 bootstrap contract changed")
    required_decision = protocol.get("append_only_artifacts", {}).get(
        "decision_record_required_fields"
    )
    if tuple(required_decision or ()) != DECISION_FIELDS:
        raise V18Error("runner decision schema differs from protocol")
    required_outcome = protocol.get("append_only_artifacts", {}).get(
        "outcome_record_required_fields"
    )
    if tuple(required_outcome or ()) != OUTCOME_FIELDS:
        raise V18Error("runner outcome schema differs from protocol")
    source_fields = protocol.get("source_contract", {}).get("forward_daily", {}).get(
        "source_manifest_required_fields"
    )
    if tuple(source_fields or ()) != SOURCE_MANIFEST_FIELDS:
        raise V18Error("runner source-manifest schema differs from protocol")
    outcome_fields = protocol.get("source_contract", {}).get("outcome_daily", {}).get(
        "manifest_required_fields"
    )
    if tuple(outcome_fields or ()) != OUTCOME_MANIFEST_FIELDS:
        raise V18Error("runner outcome-manifest schema differs from protocol")
    if tuple(
        protocol.get("c00_contract", {}).get("score_record_required_fields", ())
    ) != SCORE_FIELDS:
        raise V18Error("runner score schema differs from protocol")
    runtime_contract = protocol.get("runtime_lock_contract", {})
    runtime_value, runtime_digest = validate_runtime_lock(strict_environment=False)
    if (
        runtime_contract.get("path") != "research/model_v18_runtime_lock.json"
        or runtime_contract.get("file_sha256") != RUNTIME_LOCK_SHA256
        or runtime_contract.get("file_sha256") != runtime_digest
        or runtime_contract.get("self_sha256")
        != runtime_value["runtime_lock_self_sha256"]
        or tuple(runtime_contract.get("required_top_level_fields", ()))
        != tuple(runtime_value)
    ):
        raise V18Error("v1.8 runtime-lock protocol binding changed")
    forward_source = protocol.get("source_contract", {}).get("forward_daily", {})
    forward_store = forward_source.get("external_raw_evidence_store", {})
    if (
        forward_source.get("parser_path") != JPX_PARSER_PATH
        or forward_source.get("parser_version") != JPX_PARSER_VERSION
        or forward_source.get("parser_sha256") != JPX_PARSER_SHA256
        or forward_source.get("same_day_source_forbidden") is not True
        or forward_source.get("first_registered_target_session") != "2026-08-05"
        or forward_source.get("first_registered_target_latest_required_source_session")
        != "2026-08-04"
        or forward_store.get("object_key_prefix") != PREDICTOR_OBJECT_PREFIX
        or forward_store.get("terminal_audit_cli_argument")
        != "--predictor-raw-store-root"
    ):
        raise V18Error("v1.8 predictor source contract changed")
    outcome_source = protocol.get("source_contract", {}).get("outcome_daily", {})
    outcome_store = outcome_source.get("external_raw_evidence_store", {})
    if (
        outcome_source.get("parser_path") != JPX_PARSER_PATH
        or outcome_source.get("parser_version") != JPX_PARSER_VERSION
        or outcome_source.get("parser_sha256") != JPX_PARSER_SHA256
        or outcome_store.get("object_key_pattern")
        != "model_v18_shoulder_state/outcome/YYYY-MM-DD.pdf"
        or outcome_store.get("terminal_audit_cli_argument")
        != "--outcome-raw-store-root"
    ):
        raise V18Error("v1.8 outcome source contract changed")
    required_result_inputs = (
        "runtime_lock_sha256",
        "checkpoint_proposal_set_sha256",
        "checkpoint_core_object_set_sha256",
        "checkpoint_evidence_set_sha256",
        "predictor_source_manifest_set_sha256",
        "predictor_raw_source_set_sha256",
        "predictor_unique_raw_object_count",
        "predictor_parser_sha256",
        "predictor_parsed_shard_binding_set_sha256",
        "predictor_target_slice_semantic_set_sha256",
        "predictor_target_date_scoring_input_semantic_set_sha256",
        "c00_fold_manifest_set_sha256",
        "c00_fold_model_bundle_file_set_sha256",
        "v17_c00_protocol_sha256",
        "v17_c00_runner_sha256",
    )
    if tuple(protocol.get("result_contract", {}).get("required_input_fields", ())) != (
        required_result_inputs
    ):
        raise V18Error("v1.8 terminal result input contract changed")
    result_contract = protocol.get("result_contract", {})
    abort_stage_map = result_contract.get("abort_stage_reason_values", {})
    if (
        tuple(result_contract.get("abort_integrity_stage_values", ()))
        != ABORT_INTEGRITY_STAGES
        or tuple(result_contract.get("abort_failure_reason_values", ()))
        != ABORT_FAILURE_REASONS
        or set(abort_stage_map) != set(ABORT_INTEGRITY_STAGES)
        or any(
            not isinstance(abort_stage_map.get(stage), list)
            or not abort_stage_map[stage]
            or any(reason not in ABORT_FAILURE_REASONS for reason in abort_stage_map[stage])
            for stage in ABORT_INTEGRITY_STAGES
        )
    ):
        raise V18Error("v1.8 integrity-abort registry changed")
    calendar = protocol.get("source_contract", {}).get("calendar", {})
    if (
        calendar.get("path") != "research/model_v18_tse_session_calendar.csv"
        or calendar.get("sha256") != CALENDAR_SHA256
        or calendar.get("rows") != 343
    ):
        raise V18Error("v1.8 registered calendar changed")
    load_registered_calendar()
    validate_historical_bindings()
    return protocol, observed


def _normalise_history(history: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame):
        raise V18Error("pair history must be a DataFrame")
    frame = history.copy()
    if "date" not in frame and "session_date" in frame:
        frame = frame.rename(columns={"session_date": "date"})
    if {"rank1_oc_return_pct", "rank2_oc_return_pct"}.issubset(frame):
        frame = frame[["date", "rank1_oc_return_pct", "rank2_oc_return_pct"]].melt(
            id_vars="date", var_name="rank_name", value_name="oc_return_pct"
        )
        frame["source_rank"] = frame["rank_name"].map(
            {"rank1_oc_return_pct": 1, "rank2_oc_return_pct": 2}
        )
    missing = sorted({"date", "source_rank", "oc_return_pct"} - set(frame))
    if missing:
        raise V18Error(f"pair history is missing columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    frame["source_rank"] = pd.to_numeric(frame["source_rank"], errors="coerce")
    frame["oc_return_pct"] = pd.to_numeric(frame["oc_return_pct"], errors="coerce")
    if frame["date"].isna().any() or not frame["source_rank"].isin([1, 2]).all():
        raise V18Error("pair history date/rank is invalid")
    if getattr(frame["date"].dt, "tz", None) is not None:
        frame["date"] = frame["date"].dt.tz_convert(TOKYO).dt.tz_localize(None)
    frame["date"] = frame["date"].dt.normalize()
    frame["source_rank"] = frame["source_rank"].astype(int)
    if frame[["date", "source_rank"]].duplicated().any():
        raise V18Error("pair history contains duplicate date/rank rows")
    finite = frame["oc_return_pct"].notna() & ~np.isfinite(
        frame["oc_return_pct"].fillna(0).to_numpy(dtype=float)
    )
    if finite.any():
        raise V18Error("pair history contains a non-finite return")
    return frame[["date", "source_rank", "oc_return_pct"]].sort_values(
        ["date", "source_rank"], kind="stable"
    ).reset_index(drop=True)


def pair_history_semantic_hash(history: pd.DataFrame) -> str:
    frame = _normalise_history(history)
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def derive_month_state(history: pd.DataFrame, target_month: str) -> dict[str, Any]:
    """Pure SH01 state function used by runtime and independent unit tests."""

    target = _month(target_month, "target_month")
    frame = _normalise_history(history)
    frame = frame.loc[frame["date"].lt(target.start_time)].copy()
    wide = frame.pivot(index="date", columns="source_rank", values="oc_return_pct")
    for rank in (1, 2):
        if rank not in wide:
            wide[rank] = np.nan
    wide["difference"] = wide[1] - wide[2]
    required = list(pd.period_range(target - 3, target - 1, freq="M"))
    month_records: list[dict[str, Any]] = []
    medians: list[float] = []
    for source_month in required:
        mask = wide.index.to_period("M") == source_month
        values = wide.loc[mask, "difference"].dropna()
        count = int(len(values))
        median = float(values.median()) if count >= MIN_COMPLETE_PAIRS else None
        month_records.append(
            {
                "month": str(source_month),
                "complete_pairs": count,
                "qualified": median is not None,
                "median_rank1_minus_rank2_pct": median,
            }
        )
        if median is not None:
            medians.append(median)
    if len(medians) != 3:
        state_value = None
        selected_rank = None
        decision = ShoulderDecision.CASH
        reason = "insufficient_consecutive_month_history"
    else:
        state_value = float(np.median(np.asarray(medians, dtype=float)))
        selected_rank = 1 if state_value > 0 else 2 if state_value < 0 else None
        decision = (
            ShoulderDecision.RANK1
            if selected_rank == 1
            else ShoulderDecision.RANK2
            if selected_rank == 2
            else ShoulderDecision.CASH
        )
        reason = (
            "positive_three_month_median"
            if selected_rank == 1
            else "negative_three_month_median"
            if selected_rank == 2
            else "exact_zero_three_month_median"
        )
    used = frame.loc[frame["date"].dt.to_period("M").isin(required)]
    return {
        "target_month": str(target),
        "required_source_months": [str(item) for item in required],
        "source_months": month_records,
        "state_value_pct": state_value,
        "decision": decision.value,
        "selected_source_rank": selected_rank,
        "reason": reason,
        "history_cutoff_date": str((target.start_time - pd.Timedelta(days=1)).date()),
        "state_history_sha256": pair_history_semantic_hash(used),
        "production_model_changed": False,
        "orders_allowed": False,
    }


def load_initial_pair_history() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only the protocol-authorised v1.7 May-July initial seed."""

    bindings = validate_historical_bindings()
    picks = _read_csv_plain(
        ROOT / V17_BINDINGS["picks"][0],
        label="bound v1.7 picks",
        dtype={"code": "string"},
    )
    top1 = picks.loc[
        picks["candidate_id"].eq(C00_ID), ["date", "oc_return_pct"]
    ].assign(source_rank=1)
    top2 = picks.loc[
        picks["candidate_id"].eq(C02_ID), ["date", "oc_return_pct"]
    ].assign(source_rank=2)
    history = _normalise_history(pd.concat([top1, top2], ignore_index=True))
    history = history.loc[
        history["date"].dt.to_period("M").isin(
            [pd.Period("2026-05", "M"), pd.Period("2026-06", "M"), pd.Period("2026-07", "M")]
        )
    ].reset_index(drop=True)
    july_dates = history.loc[
        history["date"].dt.to_period("M").eq(pd.Period("2026-07", freq="M")),
        "date",
    ]
    if july_dates.empty or july_dates.max() != pd.Timestamp("2026-07-27"):
        raise V18Error("v1.7 partial-July seed exception changed")
    state = derive_month_state(history, "2026-08")
    expected_counts = [17, 19, 18]
    expected_medians = [0.4532617412224217, 0.5353494177210093, 0.09214571919513584]
    if [item["complete_pairs"] for item in state["source_months"]] != expected_counts or not np.allclose(
        [item["median_rank1_minus_rank2_pct"] for item in state["source_months"]],
        expected_medians,
        atol=1e-15,
        rtol=0,
    ):
        raise V18Error("v1.7 initial state does not reproduce the protocol")
    metadata = {
        "authority_path": V17_BINDINGS["picks"][0],
        "authority_sha256": V17_BINDINGS["picks"][1],
        "history_semantic_sha256": pair_history_semantic_hash(history),
        "source_bindings_sha256": bindings["binding_set_sha256"],
        "initial_state": state,
    }
    return history, metadata


SOURCE_MANIFEST_FIELDS = (
    "schema_version",
    "target_session",
    "latest_required_source_session",
    "previous_counted_target_session",
    "previous_counted_source_manifest_sha256",
    "created_at",
    "sealed_at",
    "runtime_lock_sha256",
    "runtime_lock_verified_at",
    "source_files",
    "source_urls",
    "source_object_keys",
    "source_byte_counts",
    "source_sha256",
    "source_set_sha256",
    "parsed_shards",
    "parsed_shard_set_sha256",
    "month_source_manifest_sha256",
    "model_price_snapshot_target_month",
    "model_price_snapshot_latest_source_session",
    "model_price_snapshot_object_key",
    "model_price_snapshot_byte_count",
    "model_price_snapshot_file_sha256",
    "model_price_snapshot_semantic_sha256",
    "model_price_snapshot_manifest_object_key",
    "model_price_snapshot_manifest_byte_count",
    "model_price_snapshot_manifest_file_sha256",
    "model_price_snapshot_manifest_sha256",
    "source_received_at",
    "parser_path",
    "parser_version",
    "parser_sha256",
    "parsed_row_count",
    "rejected_row_count",
    "duplicate_date_code_count",
    "target_date_scoring_input_semantic_sha256",
    "target_slice_semantic_sha256",
    "g0_panel_cache_object_key",
    "g0_panel_cache_byte_count",
    "g0_panel_cache_sha256",
    "g0_panel_cache_manifest_object_key",
    "g0_panel_cache_manifest_byte_count",
    "g0_panel_cache_manifest_file_sha256",
    "g0_panel_cache_manifest_sha256",
    "source_complete",
    "failure_reason",
    "python_version",
    "canonical_json_contract",
    "source_manifest_sha256",
)
SOURCE_MANIFEST_REQUIRED = frozenset(SOURCE_MANIFEST_FIELDS)

MONTH_SOURCE_MANIFEST_FIELDS = (
    "schema_version",
    "target_month",
    "first_counted_session",
    "seal_session",
    "activation_observed_at",
    "latest_required_source_session",
    "created_at",
    "sealed_at",
    "runtime_lock_sha256",
    "runtime_lock_verified_at",
    "source_files",
    "source_urls",
    "source_object_keys",
    "source_byte_counts",
    "source_sha256",
    "source_set_sha256",
    "source_received_at",
    "parsed_shards",
    "parsed_shard_set_sha256",
    "model_price_snapshot_origin",
    "previous_model_price_snapshot_target_month",
    "previous_model_price_snapshot_latest_source_session",
    "previous_model_price_snapshot_manifest_object_key",
    "previous_model_price_snapshot_manifest_file_sha256",
    "previous_model_price_snapshot_manifest_sha256",
    "model_price_suffix_shard_count",
    "model_price_suffix_shard_set_sha256",
    "model_price_snapshot_target_month",
    "model_price_snapshot_latest_source_session",
    "model_price_snapshot_object_key",
    "model_price_snapshot_byte_count",
    "model_price_snapshot_file_sha256",
    "model_price_snapshot_semantic_sha256",
    "model_price_snapshot_manifest_object_key",
    "model_price_snapshot_manifest_byte_count",
    "model_price_snapshot_manifest_file_sha256",
    "model_price_snapshot_manifest_sha256",
    "parsed_row_count",
    "model_price_full_prefix_semantic_sha256",
    "g0_training_panel_semantic_sha256",
    "g0_training_row_count",
    "activation_payload_sha256",
    "activation_receipt_sha256",
    "protocol_sha256",
    "runner_sha256",
    "parser_sha256",
    "canonical_json_contract",
    "month_source_manifest_sha256",
)

OUTCOME_MANIFEST_FIELDS = (
    "schema_version",
    "target_session",
    "created_at",
    "sealed_at",
    "runtime_lock_sha256",
    "runtime_lock_verified_at",
    "source_file_name",
    "source_url",
    "raw_source_object_key",
    "source_byte_count",
    "source_sha256",
    "source_received_at",
    "parser_path",
    "parser_version",
    "parser_sha256",
    "parsed_row_count",
    "rejected_row_count",
    "duplicate_date_code_count",
    "parsed_unique_date_count",
    "target_session_row_count",
    "rank1_code",
    "rank1_open",
    "rank1_close",
    "rank1_recomputed_oc_return_pct",
    "rank2_code",
    "rank2_open",
    "rank2_close",
    "rank2_recomputed_oc_return_pct",
    "decision_record_sha256",
    "protocol_sha256",
    "activation_payload_sha256",
    "activation_receipt_sha256",
    "python_version",
    "canonical_json_contract",
    "outcome_manifest_sha256",
)


def _official_jpx_url(value: Any, name: str) -> str:
    token = str(value)
    parsed = urlparse(token)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "www.jpx.co.jp"
        or parsed.netloc != "www.jpx.co.jp"
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
        or parsed.params
        or "%" in parsed.path
        or "//" in parsed.path
        or any(part in {".", ".."} for part in parsed.path.split("/"))
    ):
        raise V18Error(f"{name} is not an absolute official JPX HTTPS URL")
    return token


def _official_jpx_daily_url(
    value: Any,
    name: str,
    *,
    file_name: Any,
    source_session: Any,
) -> str:
    """Validate an operator-attested daily JPX URL label (not origin proof)."""

    token = _official_jpx_url(value, name)
    file_token = str(file_name)
    session = _date(source_session, f"{name} source session")
    if file_token != f"stq_{session.strftime('%Y%m%d')}.pdf":
        raise V18Error(f"{name} filename differs from its source session")
    parsed = urlparse(token)
    pattern = re.compile(
        r"^/markets/statistics-equities/daily/"
        r"[a-z0-9]+-att/"
        + re.escape(file_token)
        + r"$"
    )
    if pattern.fullmatch(parsed.path) is None:
        raise V18Error(f"{name} is not the canonical daily JPX URL label")
    return token


def _raw_source_provenance_envelope() -> dict[str, Any]:
    return {
        "mode": RAW_SOURCE_PROVENANCE_MODE,
        "official_source_verified": False,
        "caveat_id": RAW_SOURCE_PROVENANCE_CAVEAT_ID,
        "caveat_text_sha256": RAW_SOURCE_PROVENANCE_CAVEAT_SHA256,
    }


def _source_file_date(name: str, kind: str) -> pd.Timestamp:
    pattern = r"^(\d{6})\.pdf$" if kind == "price_warmup" else r"^stq_(\d{8})\.pdf$"
    match = re.fullmatch(pattern, name)
    if match is None:
        raise V18Error(f"predictor {kind} source filename is invalid: {name}")
    raw = match.group(1) + ("01" if kind == "price_warmup" else "")
    return pd.to_datetime(raw, format="%Y%m%d", errors="raise").normalize()


def _expected_predictor_files(latest: pd.Timestamp) -> tuple[list[str], dict[str, str]]:
    """Return the exact cumulative v1.7-plus-forward raw source registry."""

    try:
        warmup = v17.v16._expected_price_sources()
        legacy = v17.v16._expected_daily_sources()
        extension, _ = v17._expected_extension_sources()
    except Exception as exc:
        raise V18Error(f"bound v1.7 predictor registry failed validation: {exc}") from exc
    warmup_names = sorted(warmup, key=lambda item: (_source_file_date(item, "price_warmup"), item))
    daily_names = set(legacy) | set(extension)
    fixed_forward = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2026-07-28",
                "2026-07-29",
                "2026-07-30",
                "2026-07-31",
                "2026-08-03",
                "2026-08-04",
            ]
        )
    )
    forward_dates = fixed_forward[fixed_forward <= latest]
    if latest >= pd.Timestamp("2026-08-05"):
        calendar = load_registered_calendar()
        forward_dates = forward_dates.union(calendar[calendar <= latest])
    daily_names.update(f"stq_{item:%Y%m%d}.pdf" for item in forward_dates)
    ordered_daily = sorted(
        daily_names, key=lambda item: (_source_file_date(item, "daily"), item)
    )
    ordered = [*warmup_names, *ordered_daily]
    kinds = {
        **{name: "price_warmup" for name in warmup_names},
        **{name: "daily" for name in ordered_daily},
    }
    return ordered, kinds


def _historical_predictor_metadata() -> dict[str, dict[str, Any]]:
    try:
        warmup = v17.v16._expected_price_sources()
        legacy = v17.v16._expected_daily_sources()
        extension, _ = v17._expected_extension_sources()
    except Exception as exc:
        raise V18Error(f"bound v1.7 predictor metadata failed validation: {exc}") from exc
    return {**warmup, **legacy, **extension}


def _external_store_root(store_root: str | Path, label: str) -> Path:
    supplied = Path(store_root)
    if not supplied.exists() or not supplied.is_dir() or supplied.is_symlink():
        raise V18Error(f"{label} raw store root must be an existing non-symlink directory")
    root = supplied.resolve(strict=True)
    repository = ROOT.resolve()
    if root == repository or repository in root.parents:
        raise V18Error(f"{label} raw store must be outside the repository")
    return root


def _external_key_components(
    object_key: Any,
    *,
    prefix: str,
    label: str,
) -> tuple[str, ...]:
    token = str(object_key)
    components = token.split("/")
    if (
        "\x00" in token
        or "\\" in token
        or not token.startswith(prefix)
        or any(component in {"", ".", ".."} for component in components)
        or Path(*components).is_absolute()
        or Path(*components).as_posix() != token
    ):
        raise V18Error(f"{label} raw object key is unsafe")
    return tuple(components)


@contextmanager
def _external_parent_fd(
    store_root: str | Path,
    object_key: Any,
    *,
    prefix: str,
    label: str,
    create_parents: bool = False,
) -> Iterable[tuple[int, str]]:
    """Pin every store component with openat/O_NOFOLLOW before file I/O."""

    root = _external_store_root(store_root, label)
    components = _external_key_components(object_key, prefix=prefix, label=label)
    try:
        expected_root = os.stat(root, follow_symlinks=False)
        current_fd = os.open(
            root,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise V18Error(f"{label} raw store root cannot be pinned") from exc
    try:
        observed_root = os.fstat(current_fd)
        if (observed_root.st_dev, observed_root.st_ino) != (
            expected_root.st_dev,
            expected_root.st_ino,
        ):
            raise V18Error(f"{label} raw store root changed during open")
        if (
            observed_root.st_uid != os.geteuid()
            or stat.S_IMODE(observed_root.st_mode) & 0o022
        ):
            raise V18Error(
                f"{label} raw store root owner/permissions are not private"
            )
        for component in components[:-1]:
            try:
                before = os.stat(
                    component,
                    dir_fd=current_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if not create_parents:
                    raise V18Error(f"{label} raw object parent is missing")
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
                before = os.stat(
                    component,
                    dir_fd=current_fd,
                    follow_symlinks=False,
                )
            if not stat.S_ISDIR(before.st_mode):
                raise V18Error(f"{label} raw object parent is not a plain directory")
            try:
                next_fd = os.open(
                    component,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_CLOEXEC
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current_fd,
                )
            except OSError as exc:
                raise V18Error(f"{label} raw object parent cannot be pinned") from exc
            after = os.fstat(next_fd)
            if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
                os.close(next_fd)
                raise V18Error(f"{label} raw object parent changed during open")
            if after.st_uid != os.geteuid() or stat.S_IMODE(after.st_mode) & 0o022:
                os.close(next_fd)
                raise V18Error(
                    f"{label} raw object parent owner/permissions are not private"
                )
            os.close(current_fd)
            current_fd = next_fd
        yield current_fd, components[-1]
    finally:
        os.close(current_fd)


@contextmanager
def _external_object_fd(
    store_root: str | Path,
    object_key: Any,
    *,
    prefix: str,
    label: str,
) -> Iterable[int]:
    with _external_parent_fd(
        store_root,
        object_key,
        prefix=prefix,
        label=label,
    ) as (parent_fd, file_name):
        try:
            before = os.stat(file_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise V18Error(f"{label} raw evidence object is missing") from exc
        if not stat.S_ISREG(before.st_mode):
            raise V18Error(f"{label} raw evidence object is not a plain file")
        if before.st_nlink != 1:
            raise V18Error(f"{label} raw evidence object has a hard-link alias")
        if (
            before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise V18Error(
                f"{label} raw evidence object owner/mode/link contract changed"
            )
        try:
            descriptor = os.open(
                file_name,
                os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise V18Error(f"{label} raw evidence object cannot be pinned") from exc
        try:
            after = os.fstat(descriptor)
            if (
                (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
                or after.st_nlink != 1
                or after.st_uid != os.geteuid()
                or stat.S_IMODE(after.st_mode) != 0o600
            ):
                raise V18Error(f"{label} raw evidence object changed during open")
            yield descriptor
        finally:
            os.close(descriptor)


def _external_stat_is_private_regular(
    observed: os.stat_result | None,
    *,
    nlink: int,
) -> bool:
    """Return whether a retained external file has the exact private contract."""

    return observed is not None and (
        stat.S_ISREG(observed.st_mode)
        and observed.st_uid == os.geteuid()
        and stat.S_IMODE(observed.st_mode) == 0o600
        and observed.st_nlink == nlink
    )


def _stream_stable_file(
    descriptor: int,
    *,
    label: str,
    destination_fd: int | None = None,
    required_signature: bytes | None = b"%PDF",
) -> tuple[int, str]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise V18Error(f"{label} source is not a regular file")
    if before.st_nlink != 1:
        raise V18Error(f"{label} source has a forbidden hard-link alias")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    byte_count = 0
    signature = b""
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        if len(signature) < 4:
            signature += chunk[: 4 - len(signature)]
        digest.update(chunk)
        byte_count += len(chunk)
        if destination_fd is not None:
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(destination_fd, remaining)
                if written <= 0:  # pragma: no cover - kernel invariant
                    raise OSError("short write while snapshotting sealed evidence")
                remaining = remaining[written:]
    after = os.fstat(descriptor)
    immutable_fields = (
        "st_dev",
        "st_ino",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(getattr(before, field) != getattr(after, field) for field in immutable_fields)
        or byte_count != before.st_size
        or after.st_nlink != 1
    ):
        raise V18Error(f"{label} source changed while being read")
    if required_signature is not None and signature != required_signature:
        raise V18Error(f"{label} source signature changed")
    return byte_count, digest.hexdigest()


def _external_object_metadata(
    store_root: str | Path,
    object_key: Any,
    *,
    prefix: str,
    label: str,
    required_signature: bytes | None = b"%PDF",
) -> tuple[int, str]:
    with _external_object_fd(
        store_root, object_key, prefix=prefix, label=label
    ) as descriptor:
        return _stream_stable_file(
            descriptor,
            label=f"{label} raw object",
            required_signature=required_signature,
        )


def _external_object_bytes(
    store_root: str | Path,
    object_key: Any,
    *,
    prefix: str,
    label: str,
) -> bytes:
    """Read exact create-once external bytes through a pinned descriptor."""

    with _external_object_fd(
        store_root, object_key, prefix=prefix, label=label
    ) as descriptor:
        before = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if any(
            getattr(before, field) != getattr(after, field)
            for field in (
                "st_dev",
                "st_ino",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        ):
            raise V18Error(f"{label} object changed while being read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size or after.st_nlink != 1:
            raise V18Error(f"{label} object changed length or link count")
        return payload


def _external_object_identity(
    store_root: str | Path,
    object_key: Any,
    *,
    prefix: str,
    label: str,
) -> tuple[int, int]:
    with _external_object_fd(
        store_root, object_key, prefix=prefix, label=label
    ) as descriptor:
        observed = os.fstat(descriptor)
        return int(observed.st_dev), int(observed.st_ino)


def _external_object_exists(
    store_root: str | Path,
    object_key: Any,
    *,
    prefix: str,
    label: str,
) -> bool:
    """Return existence, healing only a proven link-before-unlink crash state."""

    try:
        with _external_parent_fd(
            store_root,
            object_key,
            prefix=prefix,
            label=label,
        ) as (parent_fd, file_name):
            fcntl.flock(parent_fd, fcntl.LOCK_EX)
            try:
                try:
                    observed = os.stat(
                        file_name, dir_fd=parent_fd, follow_symlinks=False
                    )
                except FileNotFoundError:
                    return False
                if not stat.S_ISREG(observed.st_mode):
                    raise V18Error(f"{label} existing object is not regular")
                if (
                    observed.st_uid != os.geteuid()
                    or stat.S_IMODE(observed.st_mode) != 0o600
                ):
                    raise V18Error(
                        f"{label} existing object owner/mode contract changed"
                    )
                if observed.st_nlink == 2:
                    stage_name = f".{file_name}.staging"
                    try:
                        stage = os.stat(
                            stage_name, dir_fd=parent_fd, follow_symlinks=False
                        )
                    except FileNotFoundError as exc:
                        raise V18Error(
                            f"{label} final has an unexplained hard-link alias"
                        ) from exc
                    if (
                        not _external_stat_is_private_regular(stage, nlink=2)
                        or (stage.st_dev, stage.st_ino, stage.st_nlink)
                        != (observed.st_dev, observed.st_ino, 2)
                    ):
                        raise V18Error(
                            f"{label} final/stage crash state is inconsistent"
                        )
                    stage_fd = os.open(
                        stage_name,
                        os.O_RDONLY
                        | os.O_CLOEXEC
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=parent_fd,
                    )
                    try:
                        pinned = os.fstat(stage_fd)
                        current_stage = os.stat(
                            stage_name, dir_fd=parent_fd, follow_symlinks=False
                        )
                        current_final = os.stat(
                            file_name, dir_fd=parent_fd, follow_symlinks=False
                        )
                        expected_inode = (pinned.st_dev, pinned.st_ino, 2)
                        if (
                            not _external_stat_is_private_regular(pinned, nlink=2)
                            or not _external_stat_is_private_regular(
                                current_stage, nlink=2
                            )
                            or not _external_stat_is_private_regular(
                                current_final, nlink=2
                            )
                            or
                            (
                                current_stage.st_dev,
                                current_stage.st_ino,
                                current_stage.st_nlink,
                            )
                            != expected_inode
                            or (
                                current_final.st_dev,
                                current_final.st_ino,
                                current_final.st_nlink,
                            )
                            != expected_inode
                        ):
                            raise V18Error(
                                f"{label} crash publication changed before cleanup"
                            )
                        os.unlink(stage_name, dir_fd=parent_fd)
                        os.fsync(parent_fd)
                    finally:
                        os.close(stage_fd)
                    observed = os.stat(
                        file_name, dir_fd=parent_fd, follow_symlinks=False
                    )
                    if (
                        not _external_stat_is_private_regular(observed, nlink=1)
                        or (
                            observed.st_dev,
                            observed.st_ino,
                            observed.st_nlink,
                        )
                        != (pinned.st_dev, pinned.st_ino, 1)
                    ):
                        raise V18Error(
                            f"{label} recovered final changed after stage cleanup"
                        )
                if observed.st_nlink != 1:
                    raise V18Error(
                        f"{label} existing object is not single-link regular"
                    )
                return True
            finally:
                try:
                    fcntl.flock(parent_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
    except V18Error as exc:
        if "object parent is missing" in str(exc):
            return False
        raise



def _write_external_bytes_once(
    payload: bytes,
    *,
    store_root: str | Path,
    object_key: str,
    prefix: str,
    label: str,
    replace_unpublished_stage: bool = False,
) -> tuple[int, str]:
    """Atomically publish immutable bytes with deterministic crash recovery."""

    if not isinstance(payload, bytes) or not payload:
        raise V18Error(f"{label} payload must be non-empty exact bytes")
    expected = len(payload), hashlib.sha256(payload).hexdigest()
    with _external_parent_fd(
        store_root,
        object_key,
        prefix=prefix,
        label=label,
        create_parents=True,
    ) as (parent_fd, file_name):
        stage_name = f".{file_name}.staging"
        fcntl.flock(parent_fd, fcntl.LOCK_EX)
        try:

            def observed(name: str) -> os.stat_result | None:
                try:
                    return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    return None

            final_stat = observed(file_name)
            stage_stat = observed(stage_name)
            if final_stat is not None:
                if not stat.S_ISREG(final_stat.st_mode):
                    raise V18Error(f"{label} final object is not regular")
                if (
                    final_stat.st_uid != os.geteuid()
                    or stat.S_IMODE(final_stat.st_mode) != 0o600
                ):
                    raise V18Error(
                        f"{label} final object owner/mode contract changed"
                    )
                if stage_stat is not None:
                    if (
                        not _external_stat_is_private_regular(stage_stat, nlink=2)
                        or (stage_stat.st_dev, stage_stat.st_ino)
                        != (final_stat.st_dev, final_stat.st_ino)
                        or stage_stat.st_nlink != 2
                        or final_stat.st_nlink != 2
                    ):
                        raise V18Error(
                            f"{label} publication has a conflicting stranded stage"
                        )
                    stage_fd = os.open(
                        stage_name,
                        os.O_RDONLY
                        | os.O_CLOEXEC
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=parent_fd,
                    )
                    try:
                        pinned = os.fstat(stage_fd)
                        current = observed(stage_name)
                        current_final = observed(file_name)
                        expected_inode = (pinned.st_dev, pinned.st_ino, 2)
                        if (
                            not _external_stat_is_private_regular(pinned, nlink=2)
                            or not _external_stat_is_private_regular(
                                current, nlink=2
                            )
                            or not _external_stat_is_private_regular(
                                current_final, nlink=2
                            )
                            or
                            current is None
                            or current_final is None
                            or (
                                current.st_dev,
                                current.st_ino,
                                current.st_nlink,
                            )
                            != expected_inode
                            or (
                                current_final.st_dev,
                                current_final.st_ino,
                                current_final.st_nlink,
                            )
                            != expected_inode
                        ):
                            raise V18Error(
                                f"{label} stranded stage changed before cleanup"
                            )
                        os.unlink(stage_name, dir_fd=parent_fd)
                        os.fsync(parent_fd)
                        recovered = observed(file_name)
                        if (
                            recovered is None
                            or not _external_stat_is_private_regular(
                                recovered, nlink=1
                            )
                            or (
                                recovered.st_dev,
                                recovered.st_ino,
                                recovered.st_nlink,
                            )
                            != (pinned.st_dev, pinned.st_ino, 1)
                        ):
                            raise V18Error(
                                f"{label} recovered final changed after cleanup"
                            )
                    finally:
                        os.close(stage_fd)
                    final_stat = observed(file_name)
                if final_stat is None or final_stat.st_nlink != 1:
                    raise V18Error(f"{label} final object has a hard-link alias")
            else:
                if stage_stat is None:
                    try:
                        stage_fd = os.open(
                            stage_name,
                            os.O_RDWR
                            | os.O_CREAT
                            | os.O_EXCL
                            | os.O_CLOEXEC
                            | getattr(os, "O_NOFOLLOW", 0),
                            0o600,
                            dir_fd=parent_fd,
                        )
                    except OSError as exc:
                        raise V18Error(
                            f"{label} staging object cannot be created"
                        ) from exc
                else:
                    if not _external_stat_is_private_regular(stage_stat, nlink=1):
                        raise V18Error(
                            f"{label} stranded staging object is not recoverable"
                        )
                    stage_fd = os.open(
                        stage_name,
                        os.O_RDWR
                        | os.O_CLOEXEC
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=parent_fd,
                    )
                try:
                    pinned = os.fstat(stage_fd)
                    current = observed(stage_name)
                    if (
                        current is None
                        or not _external_stat_is_private_regular(pinned, nlink=1)
                        or not _external_stat_is_private_regular(current, nlink=1)
                        or (current.st_dev, current.st_ino, current.st_nlink)
                        != (pinned.st_dev, pinned.st_ino, 1)
                    ):
                        raise V18Error(f"{label} staging object changed during recovery")
                    os.lseek(stage_fd, 0, os.SEEK_SET)
                    existing_chunks: list[bytes] = []
                    while True:
                        chunk = os.read(stage_fd, 1024 * 1024)
                        if not chunk:
                            break
                        existing_chunks.append(chunk)
                    existing = b"".join(existing_chunks)
                    if len(existing) > len(payload) or existing != payload[: len(existing)]:
                        if not replace_unpublished_stage:
                            raise V18Error(
                                f"{label} stranded staging bytes conflict with payload"
                            )
                        # No registered final name exists, so this inode has
                        # never become authority.  Timestamp/nonce-bearing
                        # manifests may restart their exact unpublished bytes
                        # while holding the same retained-parent lock.
                        os.ftruncate(stage_fd, 0)
                        os.fsync(stage_fd)
                        existing = b""
                    os.lseek(stage_fd, len(existing), os.SEEK_SET)
                    remaining = memoryview(payload)[len(existing) :]
                    while remaining:
                        written = os.write(stage_fd, remaining)
                        if written <= 0:  # pragma: no cover - kernel invariant
                            raise OSError("short write while sealing staged evidence")
                        remaining = remaining[written:]
                    os.fchmod(stage_fd, 0o600)
                    os.fsync(stage_fd)
                    sealed = os.fstat(stage_fd)
                    if (
                        not _external_stat_is_private_regular(sealed, nlink=1)
                        or sealed.st_size != len(payload)
                    ):
                        raise V18Error(f"{label} staged bytes changed before publish")
                    os.lseek(stage_fd, 0, os.SEEK_SET)
                    digest = hashlib.sha256()
                    count = 0
                    while True:
                        chunk = os.read(stage_fd, 1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        count += len(chunk)
                    if (count, digest.hexdigest()) != expected:
                        raise V18Error(f"{label} staged exact bytes changed")
                    os.fsync(parent_fd)
                    try:
                        os.link(
                            stage_name,
                            file_name,
                            src_dir_fd=parent_fd,
                            dst_dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                    except FileExistsError as exc:
                        raise V18Error(
                            f"{label} final object appeared during no-replace publish"
                        ) from exc
                    published = observed(file_name)
                    staged = observed(stage_name)
                    if (
                        published is None
                        or staged is None
                        or not _external_stat_is_private_regular(
                            published, nlink=2
                        )
                        or not _external_stat_is_private_regular(staged, nlink=2)
                        or (published.st_dev, published.st_ino)
                        != (sealed.st_dev, sealed.st_ino)
                        or (staged.st_dev, staged.st_ino)
                        != (sealed.st_dev, sealed.st_ino)
                        or published.st_nlink != 2
                        or staged.st_nlink != 2
                    ):
                        raise V18Error(f"{label} no-replace publication changed inode")
                    os.fsync(parent_fd)
                    current = observed(stage_name)
                    current_final = observed(file_name)
                    expected_inode = (sealed.st_dev, sealed.st_ino, 2)
                    if (
                        current is None
                        or current_final is None
                        or not _external_stat_is_private_regular(current, nlink=2)
                        or not _external_stat_is_private_regular(
                            current_final, nlink=2
                        )
                        or (
                            current.st_dev,
                            current.st_ino,
                            current.st_nlink,
                        )
                        != expected_inode
                        or (
                            current_final.st_dev,
                            current_final.st_ino,
                            current_final.st_nlink,
                        )
                        != expected_inode
                    ):
                        raise V18Error(f"{label} stage changed before final unlink")
                    os.unlink(stage_name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                    published = observed(file_name)
                    if (
                        published is None
                        or not _external_stat_is_private_regular(
                            published, nlink=1
                        )
                        or (published.st_dev, published.st_ino)
                        != (sealed.st_dev, sealed.st_ino)
                        or published.st_nlink != 1
                    ):
                        raise V18Error(f"{label} final publication is not single-link")
                finally:
                    os.close(stage_fd)
        finally:
            try:
                fcntl.flock(parent_fd, fcntl.LOCK_UN)
            except OSError:
                pass
    retained = _external_object_metadata(
        store_root,
        object_key,
        prefix=prefix,
        label=label,
        required_signature=None,
    )
    if retained != expected:
        raise V18Error(f"{label} create-once object conflicts with exact bytes")
    observed_payload = _external_object_bytes(
        store_root, object_key, prefix=prefix, label=label
    )
    if observed_payload != payload:
        raise V18Error(f"{label} create-once object has a byte collision")
    return retained


def _validate_external_root_pair_disjoint(
    left_root: str | Path,
    left_label: str,
    right_root: str | Path,
    right_label: str,
) -> None:
    left = _external_store_root(left_root, left_label)
    right = _external_store_root(right_root, right_label)
    if left == right or left in right.parents or right in left.parents:
        raise V18Error(f"{left_label} and {right_label} store roots overlap")
    left_stat = os.stat(left, follow_symlinks=False)
    right_stat = os.stat(right, follow_symlinks=False)
    if (left_stat.st_dev, left_stat.st_ino) == (
        right_stat.st_dev,
        right_stat.st_ino,
    ):
        raise V18Error(f"{left_label} and {right_label} roots alias one inode")


def _validate_external_store_disjointness(
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
) -> None:
    _validate_external_root_pair_disjoint(
        predictor_raw_store_root,
        "predictor",
        predictor_derived_store_root,
        "predictor derived",
    )


def _assert_external_objects_nonalias(
    left_root: str | Path,
    left_key: str,
    left_prefix: str,
    right_root: str | Path,
    right_key: str,
    right_prefix: str,
) -> None:
    """Prove two separately sealed authorities are distinct retained inodes."""

    with _external_object_fd(
        left_root, left_key, prefix=left_prefix, label="predictor rollover raw"
    ) as left_fd, _external_object_fd(
        right_root, right_key, prefix=right_prefix, label="outcome rollover raw"
    ) as right_fd:
        left = os.fstat(left_fd)
        right = os.fstat(right_fd)
        if (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino):
            raise V18Error("predictor and outcome raw authorities alias one inode")


@contextmanager
def _external_snapshot_paths(
    store_root: str | Path,
    objects: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    label: str,
    identity_registry: dict[tuple[int, int], str] | None = None,
    required_signature: bytes | None = b"%PDF",
) -> Iterable[tuple[list[Path], list[tuple[int, str]]]]:
    """Copy pinned immutable bytes to parser-only temporary files."""

    registry = {} if identity_registry is None else identity_registry
    with tempfile.TemporaryDirectory(prefix=f"v18-{label}-snapshot-") as temporary:
        directory = Path(temporary)
        paths: list[Path] = []
        metadata: list[tuple[int, str]] = []
        seen_names: set[str] = set()
        for index, item in enumerate(objects):
            key = str(item["object_key"])
            file_name = str(item.get("file", Path(key).name))
            if Path(file_name).name != file_name or file_name in seen_names:
                raise V18Error(f"{label} snapshot filename is unsafe or duplicated")
            seen_names.add(file_name)
            snapshot = directory / file_name
            destination = os.open(
                snapshot,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
            )
            try:
                with _external_object_fd(
                    store_root, key, prefix=prefix, label=label
                ) as source:
                    source_stat = os.fstat(source)
                    identity = (int(source_stat.st_dev), int(source_stat.st_ino))
                    logical_object = f"{label}:{key}"
                    registered_object = registry.setdefault(identity, logical_object)
                    if registered_object != logical_object:
                        raise V18Error(
                            f"{label} raw object aliases another sealed object inode"
                        )
                    observed = _stream_stable_file(
                        source,
                        label=f"{label} raw object",
                        destination_fd=destination,
                        required_signature=required_signature,
                    )
                os.fsync(destination)
            finally:
                os.close(destination)
            expected_count = item.get("byte_count")
            expected_hash = item.get("sha256")
            if expected_count is not None and int(expected_count) != observed[0]:
                raise V18Error(f"{label} snapshot byte count changed at index {index}")
            if expected_hash is not None and str(expected_hash) != observed[1]:
                raise V18Error(f"{label} snapshot SHA changed at index {index}")
            paths.append(snapshot)
            metadata.append(observed)
        yield paths, metadata


def _collect_jpx_registered(
    inputs: Sequence[str | Path] | str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse PDF snapshots through the absolute locked converter in production."""

    if not _STRICT_RUNTIME_ACTIVE:
        # Pure helpers/tests may inject a parser double.  Canonical operations
        # always pass the strict branch below after runtime validation.
        return collect_jpx(inputs)
    if _LOCKED_PDFTOTEXT_EXECUTABLE is None:
        raise V18Error("strict JPX parse lacks the locked pdftotext executable")
    _validate_startup_and_module_closure(phase="JPX parser phase entry")
    try:
        paths = (
            [Path(inputs)]
            if isinstance(inputs, (str, Path))
            else [Path(item) for item in inputs]
        )
        if not paths:
            raise V18Error("strict JPX parse requires at least one PDF")
        text_paths: list[Path] = []
        for source in paths:
            if source.suffix.lower() != ".pdf" or source.is_symlink() or not source.is_file():
                raise V18Error("strict JPX parse input is not a plain PDF snapshot")
            target = source.with_suffix(".txt")
            if os.path.lexists(target):
                raise V18Error("strict JPX parse text target already exists")
            completed = subprocess.run(
                [
                    str(_LOCKED_PDFTOTEXT_EXECUTABLE),
                    "-layout",
                    str(source),
                    str(target),
                ],
                check=False,
                capture_output=True,
                env=_subprocess_environment(),
                shell=False,
                stdin=subprocess.DEVNULL,
            )
            if completed.returncode != 0 or completed.stdout:
                raise V18Error("locked pdftotext conversion failed")
            if target.is_symlink() or not target.is_file():
                raise V18Error("locked pdftotext did not create a plain text snapshot")
            text_paths.append(target)
        return collect_jpx(text_paths)
    finally:
        _validate_startup_and_module_closure(phase="JPX parser phase exit")


def _source_file_metadata(source: str | Path, *, label: str) -> tuple[int, str]:
    path = Path(source)
    if path.is_symlink() or not path.is_file():
        raise V18Error(f"{label} source PDF does not exist as a plain file")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        return _stream_stable_file(descriptor, label=label)
    finally:
        os.close(descriptor)


def _plain_file_bytes(
    path: str | Path,
    *,
    label: str,
    require_single_link: bool = True,
) -> bytes:
    """Read one pinned regular file under a cooperative shared lock."""

    try:
        descriptor = os.open(
            Path(path),
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise V18Error(f"{label} cannot be opened as a plain file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise V18Error(f"{label} is not a regular file")
        if require_single_link and before.st_nlink != 1:
            raise V18Error(f"{label} has a forbidden hard-link alias")
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if any(
            getattr(before, field) != getattr(after, field)
            for field in (
                "st_dev",
                "st_ino",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        ):
            raise V18Error(f"{label} changed while being read")
        if require_single_link and after.st_nlink != 1:
            raise V18Error(f"{label} acquired a hard-link alias while being read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise V18Error(f"{label} changed length while being read")
        return payload
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def _read_csv_plain(path: str | Path, *, label: str, **kwargs: Any) -> pd.DataFrame:
    payload = (
        _read_local_authority_bytes(path, label=label)
        if _is_registered_local_authority_path(path)
        else _plain_file_bytes(path, label=label)
    )
    return pd.read_csv(io.BytesIO(payload), **kwargs)


def _seal_external_object(
    source: str | Path,
    *,
    store_root: str | Path,
    object_key: str,
    prefix: str,
    label: str,
) -> tuple[int, str]:
    payload = _plain_file_bytes(source, label=f"{label} source")
    if payload[:4] != b"%PDF":
        raise V18Error(f"{label} source signature changed")
    expected = len(payload), hashlib.sha256(payload).hexdigest()
    retained = _write_external_bytes_once(
        payload,
        store_root=store_root,
        object_key=object_key,
        prefix=prefix,
        label=label,
    )
    if retained != expected:
        raise V18Error(f"{label} create-once object conflicts with supplied bytes")
    verified = _external_object_metadata(
        store_root,
        object_key,
        prefix=prefix,
        label=label,
    )
    if verified != expected:
        raise V18Error(f"{label} retained PDF bytes changed after publication")
    return verified


def _predictor_object_key(file_name: str, kind: str) -> str:
    return f"{PREDICTOR_OBJECT_PREFIX}{kind}/{file_name}"


def _source_set_records(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "object_key": object_key,
            "file": file_name,
            "url": source_url,
            "byte_count": int(byte_count),
            "sha256": digest,
        }
        for object_key, file_name, source_url, byte_count, digest in zip(
            value["source_object_keys"],
            value["source_files"],
            value["source_urls"],
            value["source_byte_counts"],
            value["source_sha256"],
            strict=True,
        )
    ]


def _model_price_snapshot_binding_from_anchor(
    anchor_summary: Mapping[str, Any],
) -> dict[str, Any]:
    summary = validate_predictor_cache_anchor_summary(anchor_summary)
    target_month = _month(
        summary["model_price_snapshot_target_month"],
        "anchor model-price target month",
    )
    return {
        "model_price_snapshot_target_month": str(target_month),
        "model_price_snapshot_latest_source_session": str(
            _latest_registered_source_before_month(target_month).date()
        ),
        "model_price_snapshot_object_key": summary[
            "model_price_snapshot_object_key"
        ],
        "model_price_snapshot_byte_count": int(
            summary["model_price_snapshot_byte_count"]
        ),
        "model_price_snapshot_file_sha256": summary[
            "model_price_snapshot_file_sha256"
        ],
        "model_price_snapshot_semantic_sha256": summary[
            "model_price_snapshot_semantic_sha256"
        ],
        "model_price_snapshot_manifest_object_key": summary[
            "model_price_snapshot_manifest_object_key"
        ],
        "model_price_snapshot_manifest_byte_count": int(
            summary["model_price_snapshot_manifest_byte_count"]
        ),
        "model_price_snapshot_manifest_file_sha256": summary[
            "model_price_snapshot_manifest_file_sha256"
        ],
        "model_price_snapshot_manifest_sha256": summary[
            "model_price_snapshot_manifest_sha256"
        ],
    }


def _model_price_snapshot_binding_from_month_source(
    month_source: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(month_source)
    if set(value) != set(MONTH_SOURCE_MANIFEST_FIELDS):
        raise V18Error("month-source snapshot binding schema changed")
    return {
        field: value[field]
        for field in SOURCE_MANIFEST_FIELDS
        if field.startswith("model_price_snapshot_")
    }


def _validate_model_price_snapshot_binding(
    binding: Mapping[str, Any],
    *,
    predictor_derived_store_root: str | Path,
    raw_records: Sequence[Mapping[str, Any]],
    shard_bindings: Sequence[Mapping[str, Any]],
    _reuse_sealed_semantic: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, bytes]:
    expected_fields = {
        field
        for field in SOURCE_MANIFEST_FIELDS
        if field.startswith("model_price_snapshot_")
    }
    value = dict(binding)
    if set(value) != expected_fields:
        raise V18Error("model-price snapshot binding fields changed")
    month = _month(value["model_price_snapshot_target_month"], "snapshot month")
    latest = _date(
        value["model_price_snapshot_latest_source_session"], "snapshot latest"
    )
    if latest != _latest_registered_source_before_month(month):
        raise V18Error("model-price snapshot binding latest is not exact M-1")
    manifest, manifest_payload = _read_external_canonical_json(
        predictor_derived_store_root,
        value["model_price_snapshot_manifest_object_key"],
        prefix=MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX,
        label="bound model-price snapshot manifest",
    )
    prefix_count = int(manifest.get("raw_source_count", 0))
    if prefix_count <= 0 or prefix_count > len(raw_records):
        raise V18Error("bound model-price snapshot prefix count is invalid")
    snapshot, frame, manifest_key, _ = _validate_model_price_snapshot_manifest(
        manifest,
        predictor_derived_store_root=predictor_derived_store_root,
        expected_target_month=month,
        expected_latest_source_session=latest,
        expected_raw_source_set_sha256=canonical_json_sha256(
            [dict(item) for item in raw_records[:prefix_count]]
        ),
        expected_parsed_shard_set_sha256=_parsed_shard_set_sha256(
            shard_bindings[:prefix_count]
        ),
        _reuse_sealed_semantic=_reuse_sealed_semantic,
    )
    exact = {
        "model_price_snapshot_target_month": str(month),
        "model_price_snapshot_latest_source_session": str(latest.date()),
        "model_price_snapshot_object_key": snapshot["data_object_key"],
        "model_price_snapshot_byte_count": int(snapshot["data_byte_count"]),
        "model_price_snapshot_file_sha256": snapshot["data_sha256"],
        "model_price_snapshot_semantic_sha256": snapshot[
            "model_price_semantic_sha256"
        ],
        "model_price_snapshot_manifest_object_key": manifest_key,
        "model_price_snapshot_manifest_byte_count": len(manifest_payload),
        "model_price_snapshot_manifest_file_sha256": hashlib.sha256(
            manifest_payload
        ).hexdigest(),
        "model_price_snapshot_manifest_sha256": snapshot[
            "snapshot_manifest_sha256"
        ],
    }
    if value != exact:
        raise V18Error("model-price snapshot exact binding changed")
    return snapshot, frame, manifest_payload


def _daily_source_predecessor(
    target: pd.Timestamp,
    *,
    first_counted_session_value: Any,
) -> dict[str, Any] | None:
    first = _date(first_counted_session_value, "cache-chain first counted session")
    if target < first:
        raise V18Error("cache-chain target predates first counted session")
    calendar = load_registered_calendar()
    positions = np.flatnonzero(calendar == target)
    first_positions = np.flatnonzero(calendar == first)
    if len(positions) != 1 or len(first_positions) != 1:
        raise V18Error("cache-chain target/first is outside registered calendar")
    for prior in reversed(
        calendar[int(first_positions[0]) : int(positions[0])]
    ):
        path = SOURCE_MANIFEST_DIR / f"{pd.Timestamp(prior).date()}.json"
        if not path.is_file():
            raise V18Error("cache-chain has a missing counted source manifest")
        candidate = read_json(path)
        validate_source_manifest(
            candidate,
            session_date=prior,
            first_counted_session_value=first,
            require_predecessor_decision=False,
        )
        if _strict_bool(candidate["source_complete"]):
            return candidate
    return None


def _source_manifest_predecessor_binding(
    target_session: Any,
    *,
    first_counted_session_value: Any,
    require_decision_binding: bool,
) -> tuple[str | None, str | None]:
    """Bind the immediate counted source to its already sealed decision."""

    target = _date(target_session, "source-chain target")
    first = _date(first_counted_session_value, "source-chain first counted session")
    calendar = load_registered_calendar()
    target_positions = np.flatnonzero(calendar == target)
    first_positions = np.flatnonzero(calendar == first)
    if len(target_positions) != 1 or len(first_positions) != 1 or target < first:
        raise V18Error("source-chain target/first is outside its counted calendar")
    if target == first:
        return None, None
    position = int(target_positions[0])
    previous = pd.Timestamp(calendar[position - 1])
    if previous < first:
        raise V18Error("source-chain predecessor predates the counted denominator")
    path = SOURCE_MANIFEST_DIR / f"{previous.date()}.json"
    if not path.is_file() or path.is_symlink():
        raise V18Error("source-chain immediate predecessor manifest is missing")
    predecessor = read_json(path)
    if (
        set(predecessor) != SOURCE_MANIFEST_REQUIRED
        or predecessor.get("target_session") != str(previous.date())
        or predecessor.get("source_manifest_sha256")
        != canonical_json_sha256(
            predecessor, exclude_fields={"source_manifest_sha256"}
        )
    ):
        raise V18Error("source-chain immediate predecessor manifest changed")
    predecessor_hash = _require_sha(
        predecessor["source_manifest_sha256"],
        "source-chain predecessor manifest SHA",
    )
    if require_decision_binding:
        if not DECISION_LEDGER.is_file():
            raise V18Error("source-chain predecessor decision ledger is missing")
        decisions = _load_decision_record_authority(heal_derived=True)
        matches = [
            item for item in decisions if item["session_date"] == str(previous.date())
        ]
        if len(matches) != 1 or matches[0]["source_manifest_sha256"] != (
            predecessor_hash
        ):
            raise V18Error("source-chain predecessor differs from sealed decision")
    return str(previous.date()), predecessor_hash


def validate_source_manifest(
    manifest: Mapping[str, Any],
    *,
    session_date: Any,
    predictor_raw_store_root: str | Path | None = None,
    predictor_derived_store_root: str | Path | None = None,
    parsed_prices: pd.DataFrame | None = None,
    panel: pd.DataFrame | None = None,
    first_counted_session_value: Any | None = None,
    require_predecessor_decision: bool = False,
) -> tuple[dict[str, Any], str]:
    if not isinstance(manifest, Mapping):
        raise V18Error("source manifest must be an object")
    missing = sorted(SOURCE_MANIFEST_REQUIRED - set(manifest))
    if missing:
        raise V18Error(f"source manifest is missing fields: {missing}")
    value = dict(manifest)
    if set(value) != SOURCE_MANIFEST_REQUIRED:
        raise V18Error("source manifest fields differ from protocol")
    if value["schema_version"] != 1:
        raise V18Error("source manifest schema changed")
    target = _date(session_date, "session_date")
    if _date(value["target_session"], "target_session") != target:
        raise V18Error("source manifest target differs")
    created = _timestamp(value["created_at"], "source manifest created_at")
    sealed = _timestamp(value["sealed_at"], "source manifest sealed_at")
    verified = _timestamp(
        value["runtime_lock_verified_at"], "source runtime_lock_verified_at"
    )
    if value["runtime_lock_sha256"] != RUNTIME_LOCK_SHA256:
        raise V18Error("source manifest runtime-lock binding changed")
    if verified > created or created > sealed or sealed > _cutoff(target):
        raise V18Error("source manifest timestamp DAG is invalid")
    latest = _date(value["latest_required_source_session"], "latest source")
    expected_latest = _latest_required_predictor_source_session(target)
    if latest != expected_latest:
        raise V18Error("source manifest latest session is not exact registered D-1")
    predecessor_target = value["previous_counted_target_session"]
    predecessor_hash = value["previous_counted_source_manifest_sha256"]
    if (predecessor_target is None) != (predecessor_hash is None):
        raise V18Error("source predecessor target/hash nullability differs")
    if first_counted_session_value is not None:
        expected_predecessor = _source_manifest_predecessor_binding(
            target,
            first_counted_session_value=first_counted_session_value,
            require_decision_binding=require_predecessor_decision,
        )
        if (predecessor_target, predecessor_hash) != expected_predecessor:
            raise V18Error("source manifest immediate predecessor binding changed")
    elif predecessor_target is not None:
        predecessor = _date(predecessor_target, "source predecessor target")
        calendar = load_registered_calendar()
        positions = np.flatnonzero(calendar == target)
        if len(positions) != 1 or int(positions[0]) == 0 or predecessor != (
            pd.Timestamp(calendar[int(positions[0]) - 1])
        ):
            raise V18Error("source predecessor is not the immediate calendar session")
        _require_sha(predecessor_hash, "source predecessor manifest SHA")
    files, urls, object_keys, byte_counts, hashes = (
        value["source_files"],
        value["source_urls"],
        value["source_object_keys"],
        value["source_byte_counts"],
        value["source_sha256"],
    )
    if not all(
        isinstance(item, list)
        for item in (files, urls, object_keys, byte_counts, hashes)
    ):
        raise V18Error("source manifest source vectors must be arrays")
    if len({len(files), len(urls), len(object_keys), len(byte_counts), len(hashes)}) != 1:
        raise V18Error("source manifest vectors have different lengths")
    if len(set(str(item) for item in files)) != len(files) or len(
        set(str(item) for item in object_keys)
    ) != len(object_keys):
        raise V18Error("source manifest duplicates a file or object key")
    if any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, (int, np.integer))
        or int(item) <= 0
        for item in byte_counts
    ):
        raise V18Error("source byte counts must be positive")
    for index, digest in enumerate(hashes):
        _require_sha(digest, f"source_sha256[{index}]")
    expected_files, source_kinds = _expected_predictor_files(latest)
    historical_metadata = _historical_predictor_metadata()
    expected_members = set(expected_files)
    if any(str(item) not in expected_members for item in files):
        raise V18Error("source manifest contains a source outside the registered history")
    expected_subset = [item for item in expected_files if item in set(str(v) for v in files)]
    if [str(item) for item in files] != expected_subset:
        raise V18Error("source manifest is not in canonical warmup/date/file order")
    for index, (file_name, source_url, object_key) in enumerate(
        zip(files, urls, object_keys, strict=True)
    ):
        name = str(file_name)
        if Path(name).name != name:
            raise V18Error(f"source_files[{index}] is not a plain official filename")
        if name.startswith("stq_"):
            _official_jpx_daily_url(
                source_url,
                f"source_urls[{index}]",
                file_name=name,
                source_session=_source_file_date(name, source_kinds[name]),
            )
        else:
            _official_jpx_url(source_url, f"source_urls[{index}]")
        bound = historical_metadata.get(name)
        if bound is not None:
            if hashes[index] != str(bound["sha256"]):
                raise V18Error(f"historical predictor SHA changed: {name}")
            if "bytes" in bound and int(byte_counts[index]) != int(bound["bytes"]):
                raise V18Error(f"historical predictor byte count changed: {name}")
            bound_url = bound.get("source_url")
            if bound_url is not None and str(source_url) != str(bound_url):
                raise V18Error(f"historical predictor source URL changed: {name}")
        expected_key = _predictor_object_key(name, source_kinds[name])
        if object_key != expected_key:
            raise V18Error(f"source_object_keys[{index}] differs from registered key")
        if predictor_raw_store_root is not None:
            observed_count, observed_hash = _external_object_metadata(
                predictor_raw_store_root,
                object_key,
                prefix=PREDICTOR_OBJECT_PREFIX,
                label="predictor",
            )
            if Path(str(object_key)).name != name:
                raise V18Error("predictor object basename differs from source file")
            if observed_count != int(byte_counts[index]):
                raise V18Error("predictor source byte count changed")
            if observed_hash != hashes[index]:
                raise V18Error("predictor source SHA changed")
    source_set = _source_set_records(value)
    expected_source_set_hash = canonical_json_sha256(source_set)
    _require_sha(value["source_set_sha256"], "source_set_sha256")
    if value["source_set_sha256"] != expected_source_set_hash:
        raise V18Error("source manifest source-set self-hash mismatch")
    shard_bindings = value["parsed_shards"]
    if not isinstance(shard_bindings, list) or len(shard_bindings) != len(source_set):
        raise V18Error("source manifest raw/shard binding vectors are misaligned")
    for index, (raw_record, binding) in enumerate(
        zip(source_set, shard_bindings, strict=True)
    ):
        if not isinstance(binding, Mapping) or set(binding) != set(
            PARSED_SHARD_BINDING_FIELDS
        ):
            raise V18Error(f"source parsed_shards[{index}] schema changed")
        for field, raw_field in (
            ("raw_object_key", "object_key"),
            ("raw_file", "file"),
            ("raw_url", "url"),
            ("raw_byte_count", "byte_count"),
            ("raw_sha256", "sha256"),
        ):
            expected = (
                int(raw_record[raw_field])
                if raw_field == "byte_count"
                else str(raw_record[raw_field])
            )
            if binding[field] != expected:
                raise V18Error(
                    f"source parsed_shards[{index}] raw binding changed: {field}"
                )
        _, expected_data_key, expected_manifest_key = _predictor_shard_identity(
            raw_record
        )
        if (
            binding["shard_object_key"] != expected_data_key
            or binding["shard_manifest_object_key"] != expected_manifest_key
        ):
            raise V18Error(f"source parsed_shards[{index}] key is not derived")
        for field in (
            "raw_sha256",
            "shard_manifest_file_sha256",
            "shard_manifest_sha256",
            "shard_sha256",
            "parsed_semantic_sha256",
            "pdftotext_text_sha256",
            "parser_report_sha256",
        ):
            _require_sha(binding[field], f"source parsed_shards[{index}].{field}")
        for field in (
            "raw_byte_count",
            "shard_manifest_byte_count",
            "shard_byte_count",
            "parsed_row_count",
            "pdftotext_text_byte_count",
        ):
            if isinstance(binding[field], bool) or int(binding[field]) <= 0:
                raise V18Error(
                    f"source parsed_shards[{index}].{field} must be positive"
                )
    expected_shard_set_hash = _parsed_shard_set_sha256(shard_bindings)
    if value["parsed_shard_set_sha256"] != expected_shard_set_hash:
        raise V18Error("source manifest parsed-shard-set self-hash mismatch")
    if value["source_received_at"] is None:
        raise V18Error("complete source manifest requires its actual receipt")
    received = _timestamp(value["source_received_at"], "source_received_at")
    if received > _cutoff(target):
        raise V18Error("source manifest arrived after cutoff")
    if received > created:
        raise V18Error("source manifest creation predates source receipt")
    shard_manifests: list[dict[str, Any]] = []
    if predictor_derived_store_root is not None and source_set:
        if predictor_raw_store_root is None:
            raise V18Error("parsed-shard validation also requires the raw store")
        shard_manifests = _validate_bound_predictor_shard_metadata(
            source_set,
            shard_bindings,
            predictor_raw_store_root=predictor_raw_store_root,
            predictor_derived_store_root=predictor_derived_store_root,
        )
        forward_receipts: list[datetime] = []
        for index, shard in enumerate(shard_manifests):
            shard_created = _timestamp(
                shard["created_at"], f"source parsed shard[{index}] created_at"
            )
            shard_sealed = _timestamp(
                shard["sealed_at"], f"source parsed shard[{index}] sealed_at"
            )
            if shard_sealed > created:
                raise V18Error("source manifest predates a bound parsed-shard seal")
            if shard["chronology_class"] == "forward":
                forward_receipts.append(
                    _timestamp(
                        shard["raw_received_at"],
                        f"source parsed shard[{index}] raw_received_at",
                    )
                )
                if forward_receipts[-1] > shard_created:
                    raise V18Error("source shard creation predates its raw receipt")
        if not forward_receipts or max(forward_receipts) != received:
            raise V18Error(
                "source receipt differs from the latest bound forward shard"
            )
    if value["parser_path"] != JPX_PARSER_PATH:
        raise V18Error("source parser path changed")
    if value["parser_version"] != JPX_PARSER_VERSION:
        raise V18Error("source parser version changed")
    if value["parser_sha256"] != JPX_PARSER_SHA256:
        raise V18Error("source parser SHA changed")
    if int(value["parsed_row_count"]) < 0:
        raise V18Error("source parsed row count is invalid")
    if int(value["rejected_row_count"]) != 0:
        raise V18Error("source parser rejection aborts v1.8")
    if int(value["duplicate_date_code_count"]) != 0:
        raise V18Error("source duplicate date/code aborts v1.8")
    source_complete = _strict_bool(value["source_complete"])
    if not source_complete:
        raise V18Error(
            "incomplete/missing source evidence is an experiment integrity abort"
        )
    if not files:
        raise V18Error("complete source manifest has no source files")
    if [str(item) for item in files] != expected_files:
        raise V18Error("complete source manifest lacks the cumulative registered history")
    if value["failure_reason"] is not None:
        raise V18Error("complete source manifest receipt/failure state is invalid")
    semantic_fields = (
        "target_date_scoring_input_semantic_sha256",
        "target_slice_semantic_sha256",
    )
    snapshot_fields = tuple(
        field
        for field in SOURCE_MANIFEST_FIELDS
        if field.startswith("model_price_snapshot_")
    )
    normal_month_hash = value["month_source_manifest_sha256"]
    _require_sha(normal_month_hash, "source month-source manifest SHA")
    for field in semantic_fields:
        _require_sha(value[field], field)
        if value[field] == ZERO_SHA256:
            raise V18Error(f"complete source manifest uses zero sentinel for {field}")
    if int(value["parsed_row_count"]) <= 0:
        raise V18Error("complete source manifest has no parsed rows")
    if any(value[field] is None for field in snapshot_fields):
        raise V18Error("complete source manifest lacks its model-price snapshot")
    if _month(
        value["model_price_snapshot_target_month"],
        "source model-price target month",
    ) != target.to_period("M"):
        raise V18Error("source model-price snapshot targets another month")
    for field in snapshot_fields:
        if field.endswith("sha256"):
            _require_sha(value[field], f"source {field}")
    cache_fields = (
        "g0_panel_cache_object_key",
        "g0_panel_cache_byte_count",
        "g0_panel_cache_sha256",
        "g0_panel_cache_manifest_object_key",
        "g0_panel_cache_manifest_byte_count",
        "g0_panel_cache_manifest_file_sha256",
        "g0_panel_cache_manifest_sha256",
    )
    if any(value[field] is None for field in cache_fields):
        raise V18Error("complete source manifest must bind its G0 cache")
    for field in (
        "g0_panel_cache_sha256",
        "g0_panel_cache_manifest_file_sha256",
        "g0_panel_cache_manifest_sha256",
    ):
        _require_sha(value[field], field)
    for field in (
        "g0_panel_cache_byte_count",
        "g0_panel_cache_manifest_byte_count",
    ):
        if isinstance(value[field], bool) or int(value[field]) <= 0:
            raise V18Error(f"complete source manifest {field} must be positive")
    if (parsed_prices is None) != (panel is None):
        raise V18Error("source semantic validation requires parsed prices and panel together")
    if parsed_prices is not None and panel is not None:
        if not source_complete:
            raise V18Error("an incomplete source manifest cannot bind a parsed panel")
        panel_dates = pd.to_datetime(panel["date"], errors="coerce", format="mixed")
        if panel_dates.isna().any():
            raise V18Error("source supplied panel contains invalid dates")
        if not panel_dates.eq(target).all():
            raise V18Error("daily source panel must be the exact target slice")
        scoring_columns = (
            "date",
            "code",
            "name",
            "common_score_eligible",
            "feature_source_max_date",
            *G0_FEATURES,
        )
        hashes_observed = {
            "target_date_scoring_input_semantic_sha256": semantic_frame_sha256(
                panel, scoring_columns
            ),
            "target_slice_semantic_sha256": semantic_frame_sha256(
                panel, G0_PANEL_COLUMNS
            ),
        }
        for field, observed_hash in hashes_observed.items():
            if value[field] != observed_hash:
                raise V18Error(f"source manifest semantic mismatch: {field}")
    if predictor_derived_store_root is not None:
        if predictor_raw_store_root is None:
            raise V18Error("derived source validation also requires the raw store")
        month_path = MONTH_SOURCE_MANIFEST_DIR / f"{target.to_period('M')}.json"
        if month_path.is_symlink() or not month_path.is_file():
            raise V18Error("source canonical month-source context is missing")
        month_value, _, observed_month_hash = validate_month_source_manifest(
            read_json(month_path),
            predictor_raw_store_root=predictor_raw_store_root,
            predictor_derived_store_root=predictor_derived_store_root,
        )
        if observed_month_hash != normal_month_hash:
            raise V18Error("source month-source context hash changed")
        context_snapshot_binding = _model_price_snapshot_binding_from_month_source(
            month_value
        )
        context_sealed = _timestamp(
            month_value["sealed_at"], "source month context sealed"
        )
        bound_snapshot = {field: value[field] for field in snapshot_fields}
        if bound_snapshot != context_snapshot_binding:
            raise V18Error("source swapped its monthly context snapshot")
        _validate_model_price_snapshot_binding(
            bound_snapshot,
            predictor_derived_store_root=predictor_derived_store_root,
            raw_records=source_set,
            shard_bindings=shard_bindings,
        )
        cache_manifest, cache_manifest_payload = _read_external_canonical_json(
            predictor_derived_store_root,
            str(value["g0_panel_cache_manifest_object_key"]),
            prefix=G0_PANEL_CACHE_OBJECT_PREFIX,
            label="G0 cache manifest",
        )
        cache_value, cache_panel, expected_cache_manifest_key, _ = (
            _validate_g0_cache_manifest(
                cache_manifest,
                predictor_derived_store_root=predictor_derived_store_root,
                expected_target_session=target,
                expected_latest_source_session=latest,
                expected_source_set_sha256=value["source_set_sha256"],
                expected_parsed_shard_set_sha256=value[
                    "parsed_shard_set_sha256"
                ],
            )
        )
        cache_created = _timestamp(cache_value["created_at"], "G0 cache created_at")
        if context_sealed > cache_created:
            raise V18Error("G0 cache predates its monthly predictor context")
        if any(
            _timestamp(shard["sealed_at"], "bound parsed shard sealed_at")
            > cache_created
            for shard in shard_manifests
        ):
            raise V18Error("G0 cache predates a bound parsed-shard seal")
        if _timestamp(cache_value["sealed_at"], "G0 cache sealed_at") > created:
            raise V18Error("source manifest predates its G0 cache seal")
        exact_cache_fields = {
            "g0_panel_cache_object_key": cache_value["data_object_key"],
            "g0_panel_cache_byte_count": int(cache_value["data_byte_count"]),
            "g0_panel_cache_sha256": cache_value["data_sha256"],
            "g0_panel_cache_manifest_object_key": expected_cache_manifest_key,
            "g0_panel_cache_manifest_byte_count": len(cache_manifest_payload),
            "g0_panel_cache_manifest_file_sha256": hashlib.sha256(
                cache_manifest_payload
            ).hexdigest(),
            "g0_panel_cache_manifest_sha256": cache_value[
                "cache_manifest_sha256"
            ],
        }
        for field, expected in exact_cache_fields.items():
            if value[field] != expected:
                raise V18Error(f"source G0 cache binding changed: {field}")
        if int(cache_value["parsed_row_count"]) != int(value["parsed_row_count"]):
            raise V18Error("source/G0 parsed row count differs")
        for field in (
            "target_date_scoring_input_semantic_sha256",
            "target_slice_semantic_sha256",
        ):
            if value[field] != cache_value[field]:
                raise V18Error(f"source/G0 semantic differs: {field}")
        cache_target = cache_panel.loc[
            pd.to_datetime(cache_panel["date"]).eq(target)
        ].reset_index(drop=True)
        if parsed_prices is not None and not _coerce_jsonl_frame(
            panel, G0_PANEL_COLUMNS, label="source supplied G0 panel"
        ).equals(cache_target):
            raise V18Error("source supplied panel differs exactly from G0 cache")
    locked_python = read_json(RUNTIME_LOCK)["runtime"]["python"]["version"]
    if value["python_version"] != locked_python:
        raise V18Error("source manifest Python version differs from runtime lock")
    if value["canonical_json_contract"] != CANONICAL_JSON_CONTRACT:
        raise V18Error("source manifest canonical contract changed")
    expected = canonical_json_sha256(value, exclude_fields={"source_manifest_sha256"})
    if value["source_manifest_sha256"] != expected:
        raise V18Error("source manifest self-hash mismatch")
    return value, expected


def _semantic_cell(value: Any, *, column: str) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if column in {"date", "feature_source_max_date"}:
        try:
            parsed = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise V18Error(f"semantic {column} is not a date") from exc
        if pd.isna(parsed):
            return None
        if parsed.tzinfo is not None:
            parsed = parsed.tz_convert(TOKYO).tz_localize(None)
        if parsed != parsed.normalize():
            raise V18Error(f"semantic {column} contains a time component")
        return str(parsed.date())
    if column in {
        "traded",
        "partial_session",
        "common_training_eligible",
        "common_score_eligible",
    }:
        return _strict_bool(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        parsed = float(value)
        if math.isnan(parsed):
            return None
        if not math.isfinite(parsed):
            raise V18Error(f"semantic {column} contains a non-finite value")
        return parsed
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if column in {
        "code",
        "name",
        "raw_name",
        "source_file",
        "source_format",
        "volume_unit",
        "turnover_unit",
        "source_volume_unit",
        "source_turnover_unit",
    }:
        return str(value)
    if isinstance(value, str):
        return value
    raise V18Error(f"unsupported semantic value in {column}: {type(value).__name__}")


_JSONL_DATE_COLUMNS = frozenset({"date", "feature_source_max_date"})
_JSONL_BOOL_COLUMNS = frozenset(
    {
        "traded",
        "partial_session",
        "common_training_eligible",
        "common_score_eligible",
    }
)
_JSONL_INTEGER_COLUMNS = frozenset({"source_line"})
_JSONL_NULLABLE_INTEGER_COLUMNS = frozenset({"trading_unit"})
_JSONL_STRING_COLUMNS = frozenset(
    {
        "code",
        "name",
        "raw_name",
        "volume_unit",
        "turnover_unit",
        "source_volume_unit",
        "source_turnover_unit",
        "source_file",
        "source_format",
    }
)
_JSONL_NULLABLE_STRING_COLUMNS = frozenset(
    {
        "volume_unit",
        "turnover_unit",
        "source_volume_unit",
        "source_turnover_unit",
    }
)
_PARSED_OPTIONAL_COLUMNS = frozenset(
    {
        "trading_unit",
        "final_special_quote",
        "net_change",
        "vwap",
        "volume_unit",
        "turnover_unit",
        "source_volume_unit",
        "source_turnover_unit",
    }
)


def _coerce_jsonl_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> pd.DataFrame:
    """Reconstruct the exact registered in-memory dtypes for cache rows."""

    if not isinstance(frame, pd.DataFrame):
        raise V18Error(f"{label} must be a DataFrame")
    registered = tuple(str(column) for column in columns)
    if len(registered) != len(set(registered)) or not {"date", "code"}.issubset(
        registered
    ):
        raise V18Error(f"{label} registered columns are invalid")
    source = frame.copy()
    for column in registered:
        if column not in source and column in _PARSED_OPTIONAL_COLUMNS:
            source[column] = None
    missing = [column for column in registered if column not in source]
    if missing:
        raise V18Error(f"{label} lacks registered columns: {missing}")
    value = source.loc[:, list(registered)].copy()
    for column in registered:
        if column in _JSONL_DATE_COLUMNS:
            original_nonnull = value[column].notna()
            parsed = pd.to_datetime(value[column], errors="coerce", format="mixed")
            if parsed.loc[original_nonnull].isna().any():
                raise V18Error(f"{label} date {column} loses a non-null value")
            if column == "date" and parsed.isna().any():
                raise V18Error(f"{label} contains a null/invalid primary date")
            if getattr(parsed.dt, "tz", None) is not None:
                parsed = parsed.dt.tz_convert(TOKYO).dt.tz_localize(None)
            nonnull = parsed.dropna()
            if not nonnull.eq(nonnull.dt.normalize()).all():
                raise V18Error(f"{label} date contains a time component")
            value[column] = parsed.dt.normalize()
        elif column in _JSONL_BOOL_COLUMNS:
            if value[column].isna().any():
                raise V18Error(f"{label} boolean {column} contains null")
            value[column] = value[column].map(_strict_bool).astype(bool)
        elif column in _JSONL_INTEGER_COLUMNS:
            numeric = pd.to_numeric(value[column], errors="coerce")
            if numeric.isna().any() or not np.equal(
                numeric.to_numpy(dtype=float),
                np.floor(numeric.to_numpy(dtype=float)),
            ).all():
                raise V18Error(f"{label} integer {column} is null or fractional")
            value[column] = numeric.astype("int64")
        elif column in _JSONL_NULLABLE_INTEGER_COLUMNS:
            original_nonnull = value[column].notna()
            numeric = pd.to_numeric(value[column], errors="coerce")
            if numeric.loc[original_nonnull].isna().any():
                raise V18Error(
                    f"{label} nullable integer {column} loses a non-null value"
                )
            finite = numeric.dropna().to_numpy(dtype=float)
            if not np.isfinite(finite).all() or not np.equal(
                finite, np.floor(finite)
            ).all():
                raise V18Error(f"{label} nullable integer {column} is invalid")
            # The cumulative direct parser promotes this column to float64
            # because monthly warmup rows legitimately contain null.
            value[column] = numeric.astype("float64")
        elif column in _JSONL_STRING_COLUMNS:
            if (
                column not in _JSONL_NULLABLE_STRING_COLUMNS
                and value[column].isna().any()
            ):
                raise V18Error(f"{label} string {column} contains null")
            if column in _JSONL_NULLABLE_STRING_COLUMNS:
                value[column] = value[column].map(
                    lambda child: None if pd.isna(child) else str(child)
                ).astype(object)
            else:
                value[column] = value[column].map(str).astype(object)
            if value[column].dropna().eq("").any():
                raise V18Error(f"{label} string {column} is empty")
        else:
            original_nonnull = value[column].notna()
            numeric = pd.to_numeric(value[column], errors="coerce")
            if numeric.loc[original_nonnull].isna().any():
                raise V18Error(f"{label} numeric {column} loses a non-null value")
            finite = numeric.dropna().to_numpy(dtype=float)
            if not np.isfinite(finite).all():
                raise V18Error(f"{label} numeric {column} is non-finite")
            value[column] = numeric.astype("float64")
    value["code"] = value["code"].astype(object)
    if value["code"].eq("").any():
        raise V18Error(f"{label} contains an empty code")
    if value[["date", "code"]].duplicated().any():
        raise V18Error(f"{label} contains duplicate date/code rows")
    return value.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def canonical_frame_jsonl_bytes(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> bytes:
    """Encode strict UTF-8/LF canonical JSON arrays in registered column order."""

    value = _coerce_jsonl_frame(frame, columns, label=label)
    rows: list[bytes] = []
    for record in value.to_dict(orient="records"):
        row = []
        for column in columns:
            child = _semantic_cell(record[column], column=column)
            if column in _JSONL_NULLABLE_INTEGER_COLUMNS and child is not None:
                child = int(child)
            row.append(child)
        rows.append(canonical_json_bytes(row) + b"\n")
    if not rows:
        raise V18Error(f"{label} cannot encode an empty JSONL artifact")
    payload = b"".join(rows)
    if payload.startswith(b"\xef\xbb\xbf") or b"\r" in payload or not payload.endswith(b"\n"):
        raise V18Error(f"{label} JSONL encoder violated its byte contract")
    return payload


def decode_canonical_frame_jsonl(
    payload: bytes,
    columns: Sequence[str],
    *,
    label: str,
) -> pd.DataFrame:
    """Strictly decode and byte-reproduce a registered canonical JSONL frame."""

    if (
        not isinstance(payload, bytes)
        or not payload
        or payload.startswith(b"\xef\xbb\xbf")
        or b"\r" in payload
        or not payload.endswith(b"\n")
        or b"\n\n" in payload
    ):
        raise V18Error(f"{label} is not canonical UTF-8 final-LF JSONL")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise V18Error(f"{label} is not strict UTF-8") from exc
    registered = tuple(str(column) for column in columns)
    records: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise V18Error(f"{label} contains a blank row")
        try:
            row = json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON token {token}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise V18Error(f"{label} row {index} is not strict JSON") from exc
        if not isinstance(row, list) or len(row) != len(registered):
            raise V18Error(f"{label} row {index} has the wrong schema width")
        record: dict[str, Any] = {}
        canonical_row: list[Any] = []
        for column, cell in zip(registered, row, strict=True):
            if cell is None:
                if column not in _JSONL_DATE_COLUMNS and column in (
                    _JSONL_BOOL_COLUMNS
                    | _JSONL_INTEGER_COLUMNS
                    | (_JSONL_STRING_COLUMNS - _JSONL_NULLABLE_STRING_COLUMNS)
                ):
                    raise V18Error(f"{label} row {index} nulls {column}")
                if column == "date" or column == "code":
                    raise V18Error(f"{label} row {index} nulls its identity")
                parsed = None
            elif column in _JSONL_BOOL_COLUMNS:
                if not isinstance(cell, bool):
                    raise V18Error(f"{label} row {index} boolean type changed")
                parsed = cell
            elif column in _JSONL_INTEGER_COLUMNS:
                if isinstance(cell, bool) or not isinstance(cell, int):
                    raise V18Error(f"{label} row {index} integer type changed")
                parsed = cell
            elif column in _JSONL_NULLABLE_INTEGER_COLUMNS:
                if isinstance(cell, bool) or not isinstance(cell, int):
                    raise V18Error(
                        f"{label} row {index} nullable integer type changed"
                    )
                parsed = cell
            elif column in _JSONL_STRING_COLUMNS or column in _JSONL_DATE_COLUMNS:
                if not isinstance(cell, str):
                    raise V18Error(f"{label} row {index} string/date type changed")
                parsed = cell
            else:
                if isinstance(cell, bool) or not isinstance(cell, (int, float)):
                    raise V18Error(f"{label} row {index} numeric type changed")
                parsed = float(cell)
                if not math.isfinite(parsed):
                    raise V18Error(f"{label} row {index} numeric value is non-finite")
            canonical = _semantic_cell(parsed, column=column)
            if column in _JSONL_NULLABLE_INTEGER_COLUMNS and canonical is not None:
                canonical = int(canonical)
            canonical_row.append(canonical)
            record[column] = parsed
        if canonical_json_bytes(canonical_row) != line.encode("utf-8"):
            raise V18Error(f"{label} row {index} is not canonical exact JSON")
        records.append(record)
    frame = _coerce_jsonl_frame(pd.DataFrame(records), registered, label=label)
    if canonical_frame_jsonl_bytes(frame, registered, label=label) != payload:
        raise V18Error(f"{label} row order or dtype reconstruction changed bytes")
    return frame


_MODEL_PRICE_STRING_COLUMNS = frozenset({"code", "name", "source_format"})
_MODEL_PRICE_NUMERIC_COLUMNS = frozenset(MODEL_PRICE_COLUMNS) - {
    "date",
    *_MODEL_PRICE_STRING_COLUMNS,
}


def _coerce_model_price_frame(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Reconstruct the exact compact v17 input projection and dtypes."""

    if not isinstance(frame, pd.DataFrame):
        raise V18Error(f"{label} must be a DataFrame")
    missing = [column for column in MODEL_PRICE_COLUMNS if column not in frame]
    if missing:
        raise V18Error(f"{label} lacks model-price columns: {missing}")
    value = frame.loc[:, list(MODEL_PRICE_COLUMNS)].copy()
    parsed_dates = pd.to_datetime(value["date"], errors="coerce", format="mixed")
    if parsed_dates.isna().any():
        raise V18Error(f"{label} contains a null/invalid date")
    if getattr(parsed_dates.dt, "tz", None) is not None:
        parsed_dates = parsed_dates.dt.tz_convert(TOKYO).dt.tz_localize(None)
    if not parsed_dates.eq(parsed_dates.dt.normalize()).all():
        raise V18Error(f"{label} date contains a time component")
    value["date"] = parsed_dates.dt.normalize()
    for column in _MODEL_PRICE_STRING_COLUMNS:
        if value[column].isna().any():
            raise V18Error(f"{label} string {column} contains null")
        value[column] = value[column].map(str).astype(object)
        if value[column].eq("").any():
            raise V18Error(f"{label} string {column} is empty")
        if value[column].map(lambda child: any(mark in child for mark in ("\r", "\n", "\x00"))).any():
            raise V18Error(f"{label} string {column} contains a control separator")
    if value["code"].eq("").any():
        raise V18Error(f"{label} contains an empty code")
    for column in _MODEL_PRICE_NUMERIC_COLUMNS:
        original_nonnull = value[column].notna()
        numeric = pd.to_numeric(value[column], errors="coerce").astype("float64")
        if numeric.loc[original_nonnull].isna().any():
            raise V18Error(f"{label} numeric {column} loses a non-null value")
        if not np.isfinite(numeric.dropna().to_numpy(dtype=float)).all():
            raise V18Error(f"{label} numeric {column} is non-finite")
        value[column] = numeric
    if value[["date", "code"]].duplicated().any():
        raise V18Error(f"{label} contains duplicate date/code rows")
    return value.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def canonical_model_price_csv_bytes(frame: pd.DataFrame, *, label: str) -> bytes:
    """Encode the compact full-prefix model input as strict round-trip CSV."""

    value = _coerce_model_price_frame(frame, label=label)
    payload = value.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="",
        date_format="%Y-%m-%d",
    ).encode("utf-8")
    if (
        not payload
        or payload.startswith(b"\xef\xbb\xbf")
        or b"\r" in payload
        or not payload.endswith(b"\n")
    ):
        raise V18Error(f"{label} encoder violated canonical CSV bytes")
    return payload


def decode_canonical_model_price_csv(payload: bytes, *, label: str) -> pd.DataFrame:
    """Decode, dtype-reconstruct, and exactly re-encode a compact snapshot."""

    if (
        not isinstance(payload, bytes)
        or not payload
        or payload.startswith(b"\xef\xbb\xbf")
        or b"\r" in payload
        or not payload.endswith(b"\n")
    ):
        raise V18Error(f"{label} is not canonical UTF-8 final-LF CSV")
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise V18Error(f"{label} is not strict UTF-8") from exc
    dtype: dict[str, Any] = {
        "code": "string",
        "name": "string",
        "source_format": "string",
        **{column: "float64" for column in _MODEL_PRICE_NUMERIC_COLUMNS},
    }
    try:
        decoded = pd.read_csv(
            io.BytesIO(payload),
            dtype=dtype,
            keep_default_na=False,
            na_values=[""],
            float_precision="round_trip",
        )
    except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
        raise V18Error(f"{label} cannot be decoded under the CSV contract") from exc
    if decoded.columns.tolist() != list(MODEL_PRICE_COLUMNS):
        raise V18Error(f"{label} CSV column order changed")
    value = _coerce_model_price_frame(decoded, label=label)
    if canonical_model_price_csv_bytes(value, label=label) != payload:
        raise V18Error(f"{label} CSV bytes/dtypes/order are not canonical")
    return value


def model_price_semantic_sha256(frame: pd.DataFrame) -> str:
    return semantic_frame_sha256(
        _coerce_model_price_frame(frame, label="model-price semantic frame"),
        MODEL_PRICE_COLUMNS,
    )


def _pdftotext_cache_contract() -> dict[str, Any]:
    lock = read_json(RUNTIME_LOCK)
    pdftotext = lock["pdftotext"]
    argv_environment = {
        "argv": [str(pdftotext["executable_path"]), "-layout", "{source}", "{target}"],
        "environment": _external_process_environment("pdftotext"),
        "shell": False,
        "stdin": "DEVNULL",
    }
    return {
        "file_sha256": str(pdftotext["executable_sha256"]),
        "elf_closure_sha256": canonical_json_sha256(lock["elf_closure"]),
        "argv_environment_contract_sha256": canonical_json_sha256(argv_environment),
    }


def _predictor_shard_identity(
    raw_record: Mapping[str, Any],
) -> tuple[str, str, str]:
    converter = _pdftotext_cache_contract()
    identity = {
        "cache_contract_id": PREDICTOR_CACHE_CONTRACT_ID,
        "official_source_file_name": str(raw_record["file"]),
        "official_source_url": str(raw_record["url"]),
        "raw_byte_count": int(raw_record["byte_count"]),
        "raw_sha256": str(raw_record["sha256"]),
        "parser_path": JPX_PARSER_PATH,
        "parser_version": JPX_PARSER_VERSION,
        "parser_sha256": JPX_PARSER_SHA256,
        "pdftotext_file_sha256": converter["file_sha256"],
        "pdftotext_elf_closure_sha256": converter["elf_closure_sha256"],
        "pdftotext_argv_environment_contract_sha256": converter[
            "argv_environment_contract_sha256"
        ],
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "columns_sha256": canonical_json_sha256(list(PARSED_PRICE_COLUMNS)),
        "canonical_jsonl_contract": PARSED_SHARD_JSONL_CONTRACT,
    }
    token = canonical_json_sha256(identity)
    return (
        token,
        f"{PREDICTOR_SHARD_OBJECT_PREFIX}{token}.jsonl",
        f"{PREDICTOR_SHARD_OBJECT_PREFIX}{token}.manifest.json",
    )


def _read_external_canonical_json(
    store_root: str | Path,
    object_key: str,
    *,
    prefix: str,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    payload = _external_object_bytes(
        store_root, object_key, prefix=prefix, label=label
    )
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V18Error(f"{label} manifest is not strict JSON") from exc
    if not isinstance(value, dict) or _json_file_bytes(value) != payload:
        raise V18Error(f"{label} manifest bytes are not canonical")
    return value, payload


def _validate_parsed_shard_manifest(
    manifest: Mapping[str, Any],
    *,
    raw_record: Mapping[str, Any],
    predictor_derived_store_root: str | Path,
    expected_chronology_class: str | None = None,
    expected_raw_received_at: Any | None = None,
    decode_data: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame | None, dict[str, Any]]:
    value = dict(manifest)
    if set(value) != set(PARSED_SHARD_MANIFEST_FIELDS):
        raise V18Error("parsed shard manifest fields differ from A2 contract")
    if value["schema_version"] != 1 or value["cache_contract_id"] != (
        PREDICTOR_CACHE_CONTRACT_ID
    ):
        raise V18Error("parsed shard manifest schema/contract changed")
    _, expected_data_key, expected_manifest_key = _predictor_shard_identity(raw_record)
    source_file_name = str(raw_record["file"])
    if source_file_name.startswith("stq_"):
        source_kind = "daily"
        source_session = _source_file_date(source_file_name, source_kind)
        official_source_url = _official_jpx_daily_url(
            raw_record["url"],
            "parsed shard official URL",
            file_name=source_file_name,
            source_session=source_session,
        )
    else:
        source_kind = "price_warmup"
        source_session = _source_file_date(source_file_name, source_kind)
        official_source_url = _official_jpx_url(
            raw_record["url"], "parsed shard official URL"
        )
    fixed = {
        "official_source_file_name": str(raw_record["file"]),
        "official_source_url": official_source_url,
        "raw_byte_count": int(raw_record["byte_count"]),
        "raw_sha256": str(raw_record["sha256"]),
        "parser_path": JPX_PARSER_PATH,
        "parser_version": JPX_PARSER_VERSION,
        "parser_sha256": JPX_PARSER_SHA256,
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "data_object_key": expected_data_key,
        "columns": list(PARSED_PRICE_COLUMNS),
        "columns_sha256": canonical_json_sha256(list(PARSED_PRICE_COLUMNS)),
        "canonical_jsonl_contract": PARSED_SHARD_JSONL_CONTRACT,
    }
    converter = _pdftotext_cache_contract()
    fixed.update(
        {
            "pdftotext_file_sha256": converter["file_sha256"],
            "pdftotext_elf_closure_sha256": converter["elf_closure_sha256"],
            "pdftotext_argv_environment_contract_sha256": converter[
                "argv_environment_contract_sha256"
            ],
        }
    )
    for field, expected in fixed.items():
        if value[field] != expected:
            raise V18Error(f"parsed shard manifest binding changed: {field}")
    _require_sha(value["manifest_sha256"], "parsed shard manifest_sha256")
    if value["manifest_sha256"] != canonical_json_sha256(
        value, exclude_fields={"manifest_sha256"}
    ):
        raise V18Error("parsed shard manifest self-hash mismatch")
    chronology = str(value["chronology_class"])
    if chronology not in {"anchor", "forward"}:
        raise V18Error("parsed shard chronology class is invalid")
    if expected_chronology_class is not None and chronology != expected_chronology_class:
        raise V18Error("parsed shard chronology differs from caller contract")
    received = (
        None
        if value["raw_received_at"] is None
        else _timestamp(value["raw_received_at"], "parsed shard raw_received_at")
    )
    if chronology == "anchor":
        if received is not None:
            raise V18Error("historical anchor shard must null raw receipt time")
    elif received is None:
        raise V18Error("forward shard requires raw receipt time")
    if expected_raw_received_at is not None:
        expected_received = _timestamp(
            expected_raw_received_at, "expected parsed shard raw_received_at"
        )
        if received != expected_received:
            raise V18Error("parsed shard raw receipt differs from sealed source")
    verified = _timestamp(
        value["runtime_lock_verified_at"], "parsed shard runtime_lock_verified_at"
    )
    created = _timestamp(value["created_at"], "parsed shard created_at")
    sealed = _timestamp(value["sealed_at"], "parsed shard sealed_at")
    if verified > created or created > sealed or (
        chronology == "forward" and received is not None and received > created
    ):
        raise V18Error("parsed shard timestamp DAG is invalid")
    for field in (
        "data_sha256",
        "pdftotext_text_sha256",
        "parser_report_sha256",
        "parsed_semantic_sha256",
    ):
        _require_sha(value[field], f"parsed shard {field}")
    for field in (
        "data_byte_count",
        "row_count",
        "pdftotext_text_byte_count",
        "unique_date_count",
    ):
        if isinstance(value[field], bool) or int(value[field]) <= 0:
            raise V18Error(f"parsed shard {field} must be positive")
    if int(value["rejected_row_count"]) != 0 or int(
        value["duplicate_date_code_count"]
    ) != 0:
        raise V18Error("parsed shard rejection/duplicate count is nonzero")
    manifest_min = _date(value["min_date"], "parsed shard min_date")
    manifest_max = _date(value["max_date"], "parsed shard max_date")
    if source_kind == "daily":
        if (
            manifest_min != source_session
            or manifest_max != source_session
            or int(value["unique_date_count"]) != 1
        ):
            raise V18Error("daily parsed shard dates differ from its filename")
    elif (
        manifest_min.to_period("M") != source_session.to_period("M")
        or manifest_max.to_period("M") != source_session.to_period("M")
    ):
        raise V18Error("monthly parsed shard dates differ from its filename")
    observed_data = _external_object_metadata(
        predictor_derived_store_root,
        expected_data_key,
        prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
        label="predictor shard data",
        required_signature=None,
    )
    if observed_data != (int(value["data_byte_count"]), value["data_sha256"]):
        raise V18Error("parsed shard data bytes differ from manifest")
    frame: pd.DataFrame | None = None
    if decode_data:
        payload = _external_object_bytes(
            predictor_derived_store_root,
            expected_data_key,
            prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
            label="predictor shard data",
        )
        frame = decode_canonical_frame_jsonl(
            payload, PARSED_PRICE_COLUMNS, label="predictor parsed shard"
        )
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if source_kind == "daily":
            if not dates.eq(source_session).all():
                raise V18Error("daily parsed shard rows differ from its filename date")
        elif not dates.dt.to_period("M").eq(source_session.to_period("M")).all():
            raise V18Error("monthly parsed shard rows differ from its filename month")
        expected_frame_fields = {
            "row_count": int(len(frame)),
            "unique_date_count": int(dates.nunique()),
            "duplicate_date_code_count": int(
                frame[["date", "code"]].duplicated(keep=False).sum()
            ),
            "min_date": str(dates.min().date()),
            "max_date": str(dates.max().date()),
            "parsed_semantic_sha256": parsed_panel_semantic_sha256(frame),
        }
        for field, expected in expected_frame_fields.items():
            if value[field] != expected:
                raise V18Error(f"parsed shard decoded value changed: {field}")
    binding = {
        "raw_object_key": str(raw_record["object_key"]),
        "raw_file": str(raw_record["file"]),
        "raw_url": str(raw_record["url"]),
        "raw_byte_count": int(raw_record["byte_count"]),
        "raw_sha256": str(raw_record["sha256"]),
        "shard_manifest_object_key": expected_manifest_key,
        "shard_manifest_byte_count": len(_json_file_bytes(value)),
        "shard_manifest_file_sha256": hashlib.sha256(
            _json_file_bytes(value)
        ).hexdigest(),
        "shard_manifest_sha256": value["manifest_sha256"],
        "shard_object_key": expected_data_key,
        "shard_byte_count": int(value["data_byte_count"]),
        "shard_sha256": value["data_sha256"],
        "parsed_row_count": int(value["row_count"]),
        "parsed_semantic_sha256": value["parsed_semantic_sha256"],
        "pdftotext_text_byte_count": int(value["pdftotext_text_byte_count"]),
        "pdftotext_text_sha256": value["pdftotext_text_sha256"],
        "parser_report_sha256": value["parser_report_sha256"],
    }
    if set(binding) != set(PARSED_SHARD_BINDING_FIELDS):  # pragma: no cover
        raise V18Error("internal parsed shard binding schema changed")
    return value, frame, binding


def ensure_predictor_parsed_shard(
    raw_record: Mapping[str, Any],
    *,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
    chronology_class: str,
    raw_received_at: Any | None,
    runtime_lock_verified_at: Any | None = None,
    created_at: Any | None = None,
    sealed_at: Any | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Validate or create one content-addressed, time-free quotation shard."""

    _validate_external_store_disjointness(
        predictor_raw_store_root, predictor_derived_store_root
    )
    chronology = str(chronology_class)
    if chronology not in {"anchor", "forward"}:
        raise V18Error("parsed shard chronology class must be anchor or forward")
    if chronology == "anchor" and raw_received_at is not None:
        raise V18Error("historical anchor shard cannot claim a raw receipt time")
    if chronology == "forward" and raw_received_at is None:
        raise V18Error("forward parsed shard requires raw receipt time")
    source_file_name = str(raw_record["file"])
    if source_file_name.startswith("stq_"):
        source_kind = "daily"
        source_session = _source_file_date(source_file_name, source_kind)
        official_source_url = _official_jpx_daily_url(
            raw_record["url"],
            "parsed shard official URL",
            file_name=source_file_name,
            source_session=source_session,
        )
    else:
        source_kind = "price_warmup"
        source_session = _source_file_date(source_file_name, source_kind)
        official_source_url = _official_jpx_url(
            raw_record["url"], "parsed shard official URL"
        )
    raw_count, raw_hash = _external_object_metadata(
        predictor_raw_store_root,
        raw_record["object_key"],
        prefix=PREDICTOR_OBJECT_PREFIX,
        label="predictor",
    )
    if raw_count != int(raw_record["byte_count"]) or raw_hash != str(
        raw_record["sha256"]
    ):
        raise V18Error("predictor raw bytes differ before shard creation")
    _, data_key, manifest_key = _predictor_shard_identity(raw_record)
    if _external_object_exists(
        predictor_derived_store_root,
        manifest_key,
        prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
        label="predictor shard manifest",
    ):
        existing, exact_manifest_bytes = _read_external_canonical_json(
            predictor_derived_store_root,
            manifest_key,
            prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
            label="predictor shard manifest",
        )
        value, frame, binding = _validate_parsed_shard_manifest(
            existing,
            raw_record=raw_record,
            predictor_derived_store_root=predictor_derived_store_root,
            expected_chronology_class=chronology,
            expected_raw_received_at=raw_received_at,
        )
        if binding["shard_manifest_byte_count"] != len(exact_manifest_bytes) or binding[
            "shard_manifest_file_sha256"
        ] != hashlib.sha256(exact_manifest_bytes).hexdigest():
            raise V18Error("parsed shard manifest exact-byte binding changed")
        return value, frame, binding

    verified = _runtime_verified_timestamp(runtime_lock_verified_at)
    snapshot_record = {
        "object_key": str(raw_record["object_key"]),
        "file": str(raw_record["file"]),
        "byte_count": int(raw_record["byte_count"]),
        "sha256": str(raw_record["sha256"]),
    }
    with _external_snapshot_paths(
        predictor_raw_store_root,
        [snapshot_record],
        prefix=PREDICTOR_OBJECT_PREFIX,
        label="predictor",
    ) as (raw_paths, _):
        try:
            prices, report = _collect_jpx_registered(raw_paths)
        except Exception as exc:
            raise V18Error(f"predictor per-file shard parse failed: {exc}") from exc
        text_path = raw_paths[0].with_suffix(".txt")
        if text_path.is_file() and not text_path.is_symlink():
            text_payload = _plain_file_bytes(
                text_path, label="locked pdftotext shard output"
            )
            text_count = len(text_payload)
            text_hash = hashlib.sha256(text_payload).hexdigest()
        else:
            inputs = report.get("inputs")
            if (
                not isinstance(inputs, list)
                or len(inputs) != 1
                or not isinstance(inputs[0].get("text_byte_count"), int)
            ):
                raise V18Error("parsed shard report lacks exact converted text bytes")
            text_count = int(inputs[0]["text_byte_count"])
            text_hash = _require_sha(
                inputs[0].get("text_sha256"), "parsed shard text SHA"
            )
    inputs = report.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise V18Error("parsed shard report does not bind exactly one source")
    parser_input = dict(inputs[0])
    if int(parser_input.get("rejected_rows", -1)) != 0:
        raise V18Error("parsed shard parser rejected a source row")
    reported_text_hash = str(parser_input.get("sha256", ""))
    if reported_text_hash != text_hash:
        raise V18Error("parsed shard report differs from converted text hash")
    parser_input["path"] = Path(str(raw_record["file"])).with_suffix(".txt").name
    report_projection = {**dict(report), "inputs": [parser_input]}
    rejected = int(parser_input["rejected_rows"])
    duplicates = int(prices[["date", "code"]].duplicated(keep=False).sum())
    if rejected != 0 or duplicates != 0:
        raise V18Error("parsed shard rejection/duplicate aborts A2")
    canonical_prices = _coerce_jsonl_frame(
        prices, PARSED_PRICE_COLUMNS, label="predictor parsed shard"
    )
    payload = canonical_frame_jsonl_bytes(
        canonical_prices, PARSED_PRICE_COLUMNS, label="predictor parsed shard"
    )
    decoded_preflight = decode_canonical_frame_jsonl(
        payload, PARSED_PRICE_COLUMNS, label="predictor parsed shard preflight"
    )
    if not decoded_preflight.equals(canonical_prices):
        raise V18Error("predictor parsed shard codec is not exactly reversible")
    dates = pd.to_datetime(canonical_prices["date"], errors="coerce")
    if dates.isna().any():
        raise V18Error("predictor parsed shard contains an invalid date")
    if source_kind == "daily":
        if not dates.eq(source_session).all() or int(dates.nunique()) != 1:
            raise V18Error("daily parsed shard rows differ from its filename date")
    elif not dates.dt.to_period("M").eq(source_session.to_period("M")).all():
        raise V18Error("monthly parsed shard rows differ from its filename month")
    data_count, data_hash = _write_external_bytes_once(
        payload,
        store_root=predictor_derived_store_root,
        object_key=data_key,
        prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
        label="predictor shard data",
    )
    created = _operation_timestamp(created_at, "parsed shard created_at")
    sealed = _operation_timestamp(sealed_at, "parsed shard sealed_at")
    received = (
        None
        if raw_received_at is None
        else _timestamp(raw_received_at, "parsed shard raw_received_at")
    )
    converter = _pdftotext_cache_contract()
    value: dict[str, Any] = {
        "schema_version": 1,
        "cache_contract_id": PREDICTOR_CACHE_CONTRACT_ID,
        "official_source_file_name": source_file_name,
        "official_source_url": official_source_url,
        "raw_byte_count": int(raw_record["byte_count"]),
        "raw_sha256": str(raw_record["sha256"]),
        "chronology_class": chronology,
        "raw_received_at": received,
        "parser_path": JPX_PARSER_PATH,
        "parser_version": JPX_PARSER_VERSION,
        "parser_sha256": JPX_PARSER_SHA256,
        "pdftotext_file_sha256": converter["file_sha256"],
        "pdftotext_elf_closure_sha256": converter["elf_closure_sha256"],
        "pdftotext_argv_environment_contract_sha256": converter[
            "argv_environment_contract_sha256"
        ],
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "runtime_lock_verified_at": verified,
        "created_at": created,
        "sealed_at": sealed,
        "data_object_key": data_key,
        "data_byte_count": data_count,
        "data_sha256": data_hash,
        "columns": list(PARSED_PRICE_COLUMNS),
        "columns_sha256": canonical_json_sha256(list(PARSED_PRICE_COLUMNS)),
        "row_count": int(len(canonical_prices)),
        "rejected_row_count": rejected,
        "duplicate_date_code_count": duplicates,
        "unique_date_count": int(dates.nunique()),
        "min_date": str(dates.min().date()),
        "max_date": str(dates.max().date()),
        "pdftotext_text_byte_count": text_count,
        "pdftotext_text_sha256": text_hash,
        "parser_report_sha256": canonical_json_sha256(report_projection),
        "parsed_semantic_sha256": parsed_panel_semantic_sha256(canonical_prices),
        "canonical_jsonl_contract": PARSED_SHARD_JSONL_CONTRACT,
    }
    value["manifest_sha256"] = canonical_json_sha256(
        value, exclude_fields={"manifest_sha256"}
    )
    manifest_payload = _json_file_bytes(value)
    _write_external_bytes_once(
        manifest_payload,
        store_root=predictor_derived_store_root,
        object_key=manifest_key,
        prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
        label="predictor shard manifest",
        replace_unpublished_stage=True,
    )
    validated, frame, binding = _validate_parsed_shard_manifest(
        value,
        raw_record=raw_record,
        predictor_derived_store_root=predictor_derived_store_root,
        expected_chronology_class=chronology,
        expected_raw_received_at=raw_received_at,
    )
    if binding["shard_manifest_byte_count"] != len(manifest_payload) or binding[
        "shard_manifest_file_sha256"
    ] != hashlib.sha256(manifest_payload).hexdigest():
        raise V18Error("new parsed shard manifest exact-byte binding changed")
    raw_identity = _external_object_identity(
        predictor_raw_store_root,
        raw_record["object_key"],
        prefix=PREDICTOR_OBJECT_PREFIX,
        label="predictor",
    )
    for key in (data_key, manifest_key):
        if raw_identity == _external_object_identity(
            predictor_derived_store_root,
            key,
            prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
            label="predictor shard",
        ):
            raise V18Error("predictor raw and shard objects alias one inode")
    return validated, frame, binding


def _parsed_shard_set_sha256(bindings: Sequence[Mapping[str, Any]]) -> str:
    records = [dict(item) for item in bindings]
    if any(set(item) != set(PARSED_SHARD_BINDING_FIELDS) for item in records):
        raise V18Error("parsed shard binding set has an invalid schema")
    return canonical_json_sha256(records)


def _load_bound_predictor_shards(
    raw_records: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
    *,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
) -> pd.DataFrame:
    """Validate the aligned raw->manifest->shard set, then merge exact rows."""

    _validate_external_store_disjointness(
        predictor_raw_store_root, predictor_derived_store_root
    )
    if len(raw_records) != len(bindings) or not raw_records:
        raise V18Error("raw and parsed-shard binding sets are empty or misaligned")
    frames: list[pd.DataFrame] = []
    raw_identities: set[tuple[int, int]] = set()
    derived_identities: set[tuple[int, int]] = set()
    for index, (raw_record, binding) in enumerate(
        zip(raw_records, bindings, strict=True)
    ):
        if set(binding) != set(PARSED_SHARD_BINDING_FIELDS):
            raise V18Error(f"parsed shard binding {index} has an invalid schema")
        for field, raw_field in (
            ("raw_object_key", "object_key"),
            ("raw_file", "file"),
            ("raw_url", "url"),
            ("raw_byte_count", "byte_count"),
            ("raw_sha256", "sha256"),
        ):
            expected = (
                int(raw_record[raw_field])
                if raw_field == "byte_count"
                else str(raw_record[raw_field])
            )
            if binding[field] != expected:
                raise V18Error(f"parsed shard binding {index} swaps raw field {field}")
        observed_count, observed_hash = _external_object_metadata(
            predictor_raw_store_root,
            raw_record["object_key"],
            prefix=PREDICTOR_OBJECT_PREFIX,
            label="predictor",
        )
        if (observed_count, observed_hash) != (
            int(raw_record["byte_count"]),
            str(raw_record["sha256"]),
        ):
            raise V18Error("bound predictor raw bytes changed")
        _, expected_data_key, expected_manifest_key = _predictor_shard_identity(
            raw_record
        )
        if (
            binding["shard_object_key"] != expected_data_key
            or binding["shard_manifest_object_key"] != expected_manifest_key
        ):
            raise V18Error("parsed shard binding uses a caller-selected key")
        manifest, manifest_payload = _read_external_canonical_json(
            predictor_derived_store_root,
            expected_manifest_key,
            prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
            label="predictor shard manifest",
        )
        if (
            len(manifest_payload) != int(binding["shard_manifest_byte_count"])
            or hashlib.sha256(manifest_payload).hexdigest()
            != binding["shard_manifest_file_sha256"]
            or manifest["manifest_sha256"] != binding["shard_manifest_sha256"]
        ):
            raise V18Error("parsed shard manifest exact bytes differ from binding")
        _, frame, observed_binding = _validate_parsed_shard_manifest(
            manifest,
            raw_record=raw_record,
            predictor_derived_store_root=predictor_derived_store_root,
        )
        if observed_binding != dict(binding):
            raise V18Error("parsed shard binding differs from validated artifact")
        raw_identity = _external_object_identity(
            predictor_raw_store_root,
            raw_record["object_key"],
            prefix=PREDICTOR_OBJECT_PREFIX,
            label="predictor",
        )
        if raw_identity in raw_identities:
            raise V18Error("predictor raw set aliases one inode")
        raw_identities.add(raw_identity)
        for key in (expected_data_key, expected_manifest_key):
            derived_identity = _external_object_identity(
                predictor_derived_store_root,
                key,
                prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
                label="predictor shard",
            )
            if derived_identity in raw_identities or derived_identity in derived_identities:
                raise V18Error("predictor shard/raw evidence aliases an inode")
            derived_identities.add(derived_identity)
        frames.append(frame)
    try:
        merged = merge_daily_prices(frames)
    except Exception as exc:
        raise V18Error(f"parsed shard merge failed: {exc}") from exc
    return _coerce_jsonl_frame(
        merged, PARSED_PRICE_COLUMNS, label="merged predictor parsed shards"
    )


def _model_price_snapshot_identity(
    *,
    target_month: pd.Period,
    latest_source_session: pd.Timestamp,
    raw_source_set_sha256: str,
    parsed_shard_set_sha256: str,
    previous_snapshot_manifest_sha256: str | None,
) -> tuple[str, str, str]:
    identity = {
        "cache_contract_id": PREDICTOR_CACHE_CONTRACT_ID,
        "target_month": str(target_month),
        "latest_source_session": str(latest_source_session.date()),
        "raw_source_set_sha256": raw_source_set_sha256,
        "parsed_shard_set_sha256": parsed_shard_set_sha256,
        "previous_snapshot_manifest_sha256": previous_snapshot_manifest_sha256,
        "columns_sha256": canonical_json_sha256(list(MODEL_PRICE_COLUMNS)),
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "parser_sha256": JPX_PARSER_SHA256,
        "canonical_csv_contract": MODEL_PRICE_CSV_CONTRACT,
    }
    token = canonical_json_sha256(identity)
    return (
        token,
        f"{MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX}{target_month}/{token}.csv",
        f"{MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX}{target_month}/{token}.manifest.json",
    )


def _validate_model_price_snapshot_manifest(
    manifest: Mapping[str, Any],
    *,
    predictor_derived_store_root: str | Path,
    expected_target_month: Any,
    expected_latest_source_session: Any,
    expected_raw_source_set_sha256: str,
    expected_parsed_shard_set_sha256: str,
    decode_data: bool = True,
    _reuse_sealed_semantic: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, str, bytes]:
    value = dict(manifest)
    if set(value) != set(MODEL_PRICE_SNAPSHOT_FIELDS):
        raise V18Error("model-price snapshot manifest fields differ from A2")
    if value["schema_version"] != 1 or value["cache_contract_id"] != (
        PREDICTOR_CACHE_CONTRACT_ID
    ):
        raise V18Error("model-price snapshot schema/contract changed")
    month = _month(expected_target_month, "model-price snapshot target month")
    latest = _date(
        expected_latest_source_session, "model-price snapshot latest source"
    )
    fixed = {
        "target_month": str(month),
        "latest_source_session": str(latest.date()),
        "raw_source_set_sha256": _require_sha(
            expected_raw_source_set_sha256, "model-price raw set SHA"
        ),
        "parsed_shard_set_sha256": _require_sha(
            expected_parsed_shard_set_sha256, "model-price shard set SHA"
        ),
        "columns": list(MODEL_PRICE_COLUMNS),
        "columns_sha256": canonical_json_sha256(list(MODEL_PRICE_COLUMNS)),
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "parser_sha256": JPX_PARSER_SHA256,
        "canonical_csv_contract": MODEL_PRICE_CSV_CONTRACT,
    }
    for field, expected in fixed.items():
        if value[field] != expected:
            raise V18Error(f"model-price snapshot binding changed: {field}")
    previous = value["previous_snapshot_manifest_sha256"]
    previous_detail_fields = (
        "previous_snapshot_target_month",
        "previous_snapshot_latest_source_session",
        "previous_snapshot_manifest_object_key",
        "previous_snapshot_manifest_byte_count",
        "previous_snapshot_manifest_file_sha256",
        "previous_snapshot_raw_source_count",
        "previous_snapshot_raw_source_set_sha256",
        "previous_snapshot_parsed_shard_set_sha256",
    )
    if previous is None:
        if any(value[field] is not None for field in previous_detail_fields):
            raise V18Error("initial model-price snapshot has predecessor details")
    else:
        _require_sha(previous, "previous model-price snapshot SHA")
        if any(value[field] is None for field in previous_detail_fields):
            raise V18Error("model-price snapshot predecessor details are incomplete")
        previous_month = _month(
            value["previous_snapshot_target_month"],
            "previous model-price snapshot target month",
        )
        previous_latest = _date(
            value["previous_snapshot_latest_source_session"],
            "previous model-price snapshot latest source",
        )
        if previous_month != month - 1 or previous_latest != (
            _latest_registered_source_before_month(previous_month)
        ):
            raise V18Error("model-price snapshot predecessor is not immediate")
        previous_manifest, previous_payload = _read_external_canonical_json(
            predictor_derived_store_root,
            value["previous_snapshot_manifest_object_key"],
            prefix=MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX,
            label="previous model-price snapshot manifest",
        )
        if (
            len(previous_payload)
            != int(value["previous_snapshot_manifest_byte_count"])
            or hashlib.sha256(previous_payload).hexdigest()
            != value["previous_snapshot_manifest_file_sha256"]
            or previous_manifest.get("snapshot_manifest_sha256") != previous
            or int(previous_manifest.get("raw_source_count", 0))
            != int(value["previous_snapshot_raw_source_count"])
            or previous_manifest.get("raw_source_set_sha256")
            != value["previous_snapshot_raw_source_set_sha256"]
            or previous_manifest.get("parsed_shard_set_sha256")
            != value["previous_snapshot_parsed_shard_set_sha256"]
        ):
            raise V18Error("model-price snapshot predecessor exact binding changed")
        previous_value, _, previous_key, _ = _validate_model_price_snapshot_manifest(
            previous_manifest,
            predictor_derived_store_root=predictor_derived_store_root,
            expected_target_month=previous_month,
            expected_latest_source_session=previous_latest,
            expected_raw_source_set_sha256=value[
                "previous_snapshot_raw_source_set_sha256"
            ],
            expected_parsed_shard_set_sha256=value[
                "previous_snapshot_parsed_shard_set_sha256"
            ],
            decode_data=decode_data,
            _reuse_sealed_semantic=_reuse_sealed_semantic,
        )
        if previous_key != value["previous_snapshot_manifest_object_key"]:
            raise V18Error("model-price snapshot predecessor key changed")
    _, expected_data_key, expected_manifest_key = _model_price_snapshot_identity(
        target_month=month,
        latest_source_session=latest,
        raw_source_set_sha256=expected_raw_source_set_sha256,
        parsed_shard_set_sha256=expected_parsed_shard_set_sha256,
        previous_snapshot_manifest_sha256=previous,
    )
    if value["data_object_key"] != expected_data_key:
        raise V18Error("model-price snapshot data key is caller-selectable")
    if value["snapshot_manifest_sha256"] != canonical_json_sha256(
        value, exclude_fields={"snapshot_manifest_sha256"}
    ):
        raise V18Error("model-price snapshot self-hash mismatch")
    verified = _timestamp(
        value["runtime_lock_verified_at"], "model-price runtime verified"
    )
    created = _timestamp(value["created_at"], "model-price created_at")
    sealed = _timestamp(value["sealed_at"], "model-price sealed_at")
    if verified > created or created > sealed:
        raise V18Error("model-price snapshot timestamp DAG is invalid")
    if previous is not None and _timestamp(
        previous_value["sealed_at"], "previous model-price snapshot sealed"
    ) > created:
        raise V18Error("model-price snapshot predates its predecessor seal")
    for field in ("raw_source_count", "parsed_row_count", "row_count"):
        if isinstance(value[field], bool) or int(value[field]) <= 0:
            raise V18Error(f"model-price snapshot {field} must be positive")
    if int(value["duplicate_date_code_count"]) != 0 or int(
        value["unique_date_count"]
    ) <= 0:
        raise V18Error("model-price snapshot date/code counts are invalid")
    retained_metadata = _external_object_metadata(
        predictor_derived_store_root,
        expected_data_key,
        prefix=MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX,
        label="model-price snapshot data",
        required_signature=None,
    )
    if retained_metadata != (
        int(value["data_byte_count"]),
        str(value["data_sha256"]),
    ):
        raise V18Error("model-price snapshot exact bytes changed")
    manifest_payload = _json_file_bytes(value)
    if not decode_data:
        return value, pd.DataFrame(), expected_manifest_key, manifest_payload
    payload = _external_object_bytes(
        predictor_derived_store_root,
        expected_data_key,
        prefix=MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX,
        label="model-price snapshot data",
    )
    frame = decode_canonical_model_price_csv(
        payload, label="model-price snapshot"
    )
    dates = pd.to_datetime(frame["date"])
    observed: dict[str, Any] = {
        "row_count": int(len(frame)),
        "parsed_row_count": int(len(frame)),
        "unique_date_count": int(dates.nunique()),
        "duplicate_date_code_count": int(
            frame[["date", "code"]].duplicated(keep=False).sum()
        ),
    }
    if _reuse_sealed_semantic:
        _require_sha(
            value["model_price_semantic_sha256"],
            "sealed model-price semantic SHA",
        )
    else:
        observed["model_price_semantic_sha256"] = model_price_semantic_sha256(
            frame
        )
    for field, expected in observed.items():
        if value[field] != expected:
            raise V18Error(f"model-price snapshot decoded value changed: {field}")
    if dates.max().normalize() != latest or dates.ge(month.start_time).any():
        raise V18Error("model-price snapshot is not the exact prior-month prefix")
    return value, frame, expected_manifest_key, manifest_payload


def materialize_model_price_snapshot(
    model_prices: pd.DataFrame,
    *,
    predictor_derived_store_root: str | Path,
    target_month: Any,
    latest_source_session: Any,
    raw_records: Sequence[Mapping[str, Any]],
    shard_bindings: Sequence[Mapping[str, Any]],
    previous_snapshot_manifest_sha256: str | None,
    previous_snapshot_binding: Mapping[str, Any] | None = None,
    runtime_lock_verified_at: Any | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, str, bytes]:
    """Seal one cumulative compact full-prefix snapshot for a target month."""

    month = _month(target_month, "model-price snapshot target month")
    latest = _date(latest_source_session, "model-price snapshot latest source")
    if latest != _latest_registered_source_before_month(month):
        raise V18Error("model-price snapshot latest is not exact M-1")
    if len(raw_records) != len(shard_bindings) or not raw_records:
        raise V18Error("model-price snapshot raw/shard sets are misaligned")
    raw_hash = canonical_json_sha256([dict(item) for item in raw_records])
    shard_hash = _parsed_shard_set_sha256(shard_bindings)
    previous_fields: dict[str, Any]
    if previous_snapshot_manifest_sha256 is None:
        if previous_snapshot_binding is not None:
            raise V18Error("initial model-price snapshot cannot bind a predecessor")
        previous_fields = {
            "previous_snapshot_target_month": None,
            "previous_snapshot_latest_source_session": None,
            "previous_snapshot_manifest_object_key": None,
            "previous_snapshot_manifest_byte_count": None,
            "previous_snapshot_manifest_file_sha256": None,
            "previous_snapshot_raw_source_count": None,
            "previous_snapshot_raw_source_set_sha256": None,
            "previous_snapshot_parsed_shard_set_sha256": None,
        }
    else:
        if previous_snapshot_binding is None:
            raise V18Error("forward model-price snapshot lacks predecessor binding")
        previous, _, previous_payload = _validate_model_price_snapshot_binding(
            previous_snapshot_binding,
            predictor_derived_store_root=predictor_derived_store_root,
            raw_records=raw_records,
            shard_bindings=shard_bindings,
        )
        if previous["snapshot_manifest_sha256"] != (
            previous_snapshot_manifest_sha256
        ):
            raise V18Error("model-price predecessor self-hash changed")
        previous_fields = {
            "previous_snapshot_target_month": previous["target_month"],
            "previous_snapshot_latest_source_session": previous[
                "latest_source_session"
            ],
            "previous_snapshot_manifest_object_key": previous_snapshot_binding[
                "model_price_snapshot_manifest_object_key"
            ],
            "previous_snapshot_manifest_byte_count": len(previous_payload),
            "previous_snapshot_manifest_file_sha256": hashlib.sha256(
                previous_payload
            ).hexdigest(),
            "previous_snapshot_raw_source_count": int(previous["raw_source_count"]),
            "previous_snapshot_raw_source_set_sha256": previous[
                "raw_source_set_sha256"
            ],
            "previous_snapshot_parsed_shard_set_sha256": previous[
                "parsed_shard_set_sha256"
            ],
        }
    canonical = _coerce_model_price_frame(
        model_prices, label="model-price snapshot source"
    )
    if pd.to_datetime(canonical["date"]).max().normalize() != latest:
        raise V18Error("model-price snapshot source does not end at exact M-1")
    payload = canonical_model_price_csv_bytes(
        canonical, label="model-price snapshot source"
    )
    decoded_preflight = decode_canonical_model_price_csv(
        payload, label="model-price snapshot preflight"
    )
    if not decoded_preflight.equals(canonical):
        raise V18Error("model-price snapshot codec is not exactly reversible")
    _, data_key, manifest_key = _model_price_snapshot_identity(
        target_month=month,
        latest_source_session=latest,
        raw_source_set_sha256=raw_hash,
        parsed_shard_set_sha256=shard_hash,
        previous_snapshot_manifest_sha256=previous_snapshot_manifest_sha256,
    )
    if _external_object_exists(
        predictor_derived_store_root,
        manifest_key,
        prefix=MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX,
        label="model-price snapshot manifest",
    ):
        existing, _ = _read_external_canonical_json(
            predictor_derived_store_root,
            manifest_key,
            prefix=MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX,
            label="model-price snapshot manifest",
        )
        validated, retained_frame, retained_key, retained_payload = (
            _validate_model_price_snapshot_manifest(
                existing,
                predictor_derived_store_root=predictor_derived_store_root,
                expected_target_month=month,
                expected_latest_source_session=latest,
                expected_raw_source_set_sha256=raw_hash,
                expected_parsed_shard_set_sha256=shard_hash,
            )
        )
        if retained_key != manifest_key:
            raise V18Error("model-price snapshot retry resolved another identity")
        if not retained_frame.equals(canonical):
            raise V18Error("model-price snapshot retry changes exact rows/dtypes")
        if int(validated["raw_source_count"]) != len(raw_records):
            raise V18Error("model-price snapshot retry changes raw-source count")
        return validated, retained_frame, retained_key, retained_payload
    data_count, data_hash = _write_external_bytes_once(
        payload,
        store_root=predictor_derived_store_root,
        object_key=data_key,
        prefix=MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX,
        label="model-price snapshot data",
    )
    verified = _runtime_verified_timestamp(runtime_lock_verified_at)
    created = datetime.now(TOKYO)
    sealed = datetime.now(TOKYO)
    dates = pd.to_datetime(canonical["date"])
    value: dict[str, Any] = {
        "schema_version": 1,
        "cache_contract_id": PREDICTOR_CACHE_CONTRACT_ID,
        "target_month": str(month),
        "latest_source_session": str(latest.date()),
        "raw_source_set_sha256": raw_hash,
        "parsed_shard_set_sha256": shard_hash,
        "raw_source_count": len(raw_records),
        "parsed_row_count": int(len(canonical)),
        "columns": list(MODEL_PRICE_COLUMNS),
        "columns_sha256": canonical_json_sha256(list(MODEL_PRICE_COLUMNS)),
        "data_object_key": data_key,
        "data_byte_count": data_count,
        "data_sha256": data_hash,
        "row_count": int(len(canonical)),
        "unique_date_count": int(dates.nunique()),
        "duplicate_date_code_count": 0,
        "model_price_semantic_sha256": model_price_semantic_sha256(canonical),
        "previous_snapshot_manifest_sha256": previous_snapshot_manifest_sha256,
        **previous_fields,
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "runtime_lock_verified_at": verified,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "parser_sha256": JPX_PARSER_SHA256,
        "created_at": created,
        "sealed_at": sealed,
        "canonical_csv_contract": MODEL_PRICE_CSV_CONTRACT,
    }
    value["snapshot_manifest_sha256"] = canonical_json_sha256(
        value, exclude_fields={"snapshot_manifest_sha256"}
    )
    manifest_payload = _json_file_bytes(value)
    _write_external_bytes_once(
        manifest_payload,
        store_root=predictor_derived_store_root,
        object_key=manifest_key,
        prefix=MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX,
        label="model-price snapshot manifest",
        replace_unpublished_stage=True,
    )
    return _validate_model_price_snapshot_manifest(
        value,
        predictor_derived_store_root=predictor_derived_store_root,
        expected_target_month=month,
        expected_latest_source_session=latest,
        expected_raw_source_set_sha256=raw_hash,
        expected_parsed_shard_set_sha256=shard_hash,
    )


def _validate_bound_predictor_shard_metadata(
    raw_records: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
    *,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
) -> list[dict[str, Any]]:
    """Hash and inode-pin a complete aligned set without decoding shard rows."""

    if len(raw_records) != len(bindings) or not raw_records:
        raise V18Error("predictor shard metadata set is empty or misaligned")
    manifests: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    for index, (raw_record, binding) in enumerate(
        zip(raw_records, bindings, strict=True)
    ):
        raw_metadata = _external_object_metadata(
            predictor_raw_store_root,
            raw_record["object_key"],
            prefix=PREDICTOR_OBJECT_PREFIX,
            label="predictor",
        )
        if raw_metadata != (
            int(raw_record["byte_count"]),
            str(raw_record["sha256"]),
        ):
            raise V18Error(f"predictor shard raw binding changed at index {index}")
        manifest, manifest_payload = _read_external_canonical_json(
            predictor_derived_store_root,
            binding["shard_manifest_object_key"],
            prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
            label="predictor shard manifest",
        )
        if (
            len(manifest_payload) != int(binding["shard_manifest_byte_count"])
            or hashlib.sha256(manifest_payload).hexdigest()
            != binding["shard_manifest_file_sha256"]
        ):
            raise V18Error("predictor shard manifest exact bytes changed")
        _, _, observed_binding = _validate_parsed_shard_manifest(
            manifest,
            raw_record=raw_record,
            predictor_derived_store_root=predictor_derived_store_root,
            decode_data=False,
        )
        if observed_binding != dict(binding):
            raise V18Error("predictor shard binding differs from its manifest")
        for root, key, prefix, label in (
            (
                predictor_raw_store_root,
                raw_record["object_key"],
                PREDICTOR_OBJECT_PREFIX,
                "predictor",
            ),
            (
                predictor_derived_store_root,
                binding["shard_object_key"],
                PREDICTOR_SHARD_OBJECT_PREFIX,
                "predictor shard",
            ),
            (
                predictor_derived_store_root,
                binding["shard_manifest_object_key"],
                PREDICTOR_SHARD_OBJECT_PREFIX,
                "predictor shard manifest",
            ),
        ):
            identity = _external_object_identity(
                root, key, prefix=prefix, label=label
            )
            if identity in identities:
                raise V18Error("predictor raw/shard set aliases an inode")
            identities.add(identity)
        manifests.append(manifest)
    return manifests


def _load_model_price_prefix(
    raw_records: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
    *,
    snapshot_target_month: Any,
    latest_required_source_session: Any,
    snapshot_manifest_object_key: str,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], bytes]:
    """Decode one M-1 compact snapshot plus only its ordered forward suffix."""

    month = _month(snapshot_target_month, "model-price prefix snapshot month")
    latest = _date(latest_required_source_session, "model-price prefix latest")
    manifests = _validate_bound_predictor_shard_metadata(
        raw_records,
        bindings,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
    )
    snapshot_manifest, snapshot_manifest_payload = _read_external_canonical_json(
        predictor_derived_store_root,
        snapshot_manifest_object_key,
        prefix=MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX,
        label="model-price snapshot manifest",
    )
    prefix_count = int(snapshot_manifest.get("raw_source_count", 0))
    if prefix_count <= 0 or prefix_count > len(raw_records):
        raise V18Error("model-price snapshot prefix count is invalid")
    prefix_raw = [dict(item) for item in raw_records[:prefix_count]]
    prefix_bindings = [dict(item) for item in bindings[:prefix_count]]
    snapshot, base, expected_key, _ = _validate_model_price_snapshot_manifest(
        snapshot_manifest,
        predictor_derived_store_root=predictor_derived_store_root,
        expected_target_month=month,
        expected_latest_source_session=snapshot_manifest["latest_source_session"],
        expected_raw_source_set_sha256=canonical_json_sha256(prefix_raw),
        expected_parsed_shard_set_sha256=_parsed_shard_set_sha256(prefix_bindings),
    )
    if snapshot_manifest_object_key != expected_key:
        raise V18Error("model-price snapshot manifest key changed")
    base_latest = _date(snapshot["latest_source_session"], "snapshot base latest")
    if base_latest > latest:
        raise V18Error("model-price snapshot is ahead of the daily source")
    suffix_frames: list[pd.DataFrame] = []
    for index in range(prefix_count, len(raw_records)):
        manifest = manifests[index]
        min_date = _date(manifest["min_date"], "model-price suffix min date")
        if min_date <= base_latest:
            raise V18Error("model-price snapshot suffix overlaps/rewrites its prefix")
        _, shard_frame, _ = _validate_parsed_shard_manifest(
            manifest,
            raw_record=raw_records[index],
            predictor_derived_store_root=predictor_derived_store_root,
            decode_data=True,
        )
        assert shard_frame is not None
        suffix_frames.append(
            _coerce_model_price_frame(
                shard_frame, label="model-price ordered forward shard"
            )
        )
    combined = _coerce_model_price_frame(
        pd.concat([base, *suffix_frames], ignore_index=True),
        label="model-price full prefix",
    )
    if pd.to_datetime(combined["date"]).max().normalize() != latest:
        raise V18Error("model-price full prefix does not end at exact D-1")
    return combined, snapshot, snapshot_manifest_payload


def _g0_cache_identity(
    *,
    target_session: pd.Timestamp,
    latest_required_source_session: pd.Timestamp,
    source_set_sha256: str,
    parsed_shard_set_sha256: str,
) -> tuple[str, str, str]:
    identity = {
        "cache_contract_id": PREDICTOR_CACHE_CONTRACT_ID,
        "scope": "target_slice",
        "target_session": str(target_session.date()),
        "latest_required_source_session": str(latest_required_source_session.date()),
        "source_set_sha256": source_set_sha256,
        "parsed_shard_set_sha256": parsed_shard_set_sha256,
        "columns_sha256": canonical_json_sha256(list(G0_PANEL_COLUMNS)),
        "v17_protocol_sha256": V17_BINDINGS["protocol"][1],
        "v17_runner_sha256": V17_BINDINGS["runner"][1],
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "parser_sha256": JPX_PARSER_SHA256,
        "canonical_jsonl_contract": G0_CACHE_JSONL_CONTRACT,
    }
    token = canonical_json_sha256(identity)
    return (
        token,
        f"{G0_PANEL_CACHE_OBJECT_PREFIX}target-slice/{token}.jsonl",
        f"{G0_PANEL_CACHE_OBJECT_PREFIX}target-slice/{token}.manifest.json",
    )


def _validate_g0_cache_manifest(
    manifest: Mapping[str, Any],
    *,
    predictor_derived_store_root: str | Path,
    expected_target_session: Any,
    expected_latest_source_session: Any,
    expected_source_set_sha256: str,
    expected_parsed_shard_set_sha256: str,
) -> tuple[dict[str, Any], pd.DataFrame, str, bytes]:
    """Validate the only persisted G0 accelerator: one outcome-null target slice."""

    value = dict(manifest)
    if set(value) != set(G0_CACHE_MANIFEST_FIELDS):
        raise V18Error("G0 target-slice manifest fields differ from A2 contract")
    target = _date(expected_target_session, "expected G0 target session")
    latest = _date(expected_latest_source_session, "expected G0 latest source")
    if latest >= target:
        raise V18Error("G0 target-slice source is not strictly prior")
    fixed = {
        "schema_version": 1,
        "cache_contract_id": PREDICTOR_CACHE_CONTRACT_ID,
        "scope": "target_slice",
        "target_session": str(target.date()),
        "latest_required_source_session": str(latest.date()),
        "source_set_sha256": _require_sha(
            expected_source_set_sha256, "G0 source-set SHA"
        ),
        "parsed_shard_set_sha256": _require_sha(
            expected_parsed_shard_set_sha256, "G0 shard-set SHA"
        ),
        "columns": list(G0_PANEL_COLUMNS),
        "columns_sha256": canonical_json_sha256(list(G0_PANEL_COLUMNS)),
        "v17_protocol_sha256": V17_BINDINGS["protocol"][1],
        "v17_runner_sha256": V17_BINDINGS["runner"][1],
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "parser_sha256": JPX_PARSER_SHA256,
        "canonical_jsonl_contract": G0_CACHE_JSONL_CONTRACT,
    }
    for field, expected in fixed.items():
        if value[field] != expected:
            raise V18Error(f"G0 target-slice binding changed: {field}")
    _, expected_data_key, expected_manifest_key = _g0_cache_identity(
        target_session=target,
        latest_required_source_session=latest,
        source_set_sha256=expected_source_set_sha256,
        parsed_shard_set_sha256=expected_parsed_shard_set_sha256,
    )
    if value["data_object_key"] != expected_data_key:
        raise V18Error("G0 target-slice key is caller-selectable")
    if value["cache_manifest_sha256"] != canonical_json_sha256(
        value, exclude_fields={"cache_manifest_sha256"}
    ):
        raise V18Error("G0 target-slice self-hash mismatch")
    verified = _timestamp(
        value["runtime_lock_verified_at"], "G0 target-slice runtime verified"
    )
    created = _timestamp(value["created_at"], "G0 target-slice created")
    sealed = _timestamp(value["sealed_at"], "G0 target-slice sealed")
    if verified > created or created > sealed or sealed > _cutoff(target):
        raise V18Error("G0 target-slice timestamp DAG is invalid")
    payload = _external_object_bytes(
        predictor_derived_store_root,
        expected_data_key,
        prefix=G0_PANEL_CACHE_OBJECT_PREFIX,
        label="G0 target-slice data",
    )
    if len(payload) != int(value["data_byte_count"]) or hashlib.sha256(
        payload
    ).hexdigest() != value["data_sha256"]:
        raise V18Error("G0 target-slice exact bytes changed")
    frame = decode_canonical_frame_jsonl(
        payload, G0_PANEL_COLUMNS, label="G0 target-slice"
    )
    dates = pd.to_datetime(frame["date"], errors="coerce")
    feature_dates = pd.to_datetime(
        frame["feature_source_max_date"], errors="coerce"
    )
    if (
        frame.empty
        or not dates.eq(target).all()
        or frame["oc_return_pct"].notna().any()
        or feature_dates.isna().any()
        or feature_dates.ge(target).any()
    ):
        raise V18Error("G0 target-slice contains outcome/future/non-target rows")
    scoring_columns = (
        "date",
        "code",
        "name",
        "common_score_eligible",
        "feature_source_max_date",
        *G0_FEATURES,
    )
    observed = {
        "row_count": int(len(frame)),
        "unique_date_count": 1,
        "duplicate_date_code_count": int(
            frame[["date", "code"]].duplicated(keep=False).sum()
        ),
        "data_semantic_sha256": semantic_frame_sha256(frame, G0_PANEL_COLUMNS),
        "target_row_count": int(len(frame)),
        "target_date_scoring_input_semantic_sha256": semantic_frame_sha256(
            frame, scoring_columns
        ),
        "target_slice_semantic_sha256": semantic_frame_sha256(
            frame, G0_PANEL_COLUMNS
        ),
        "target_outcome_nonnull_count": 0,
        "max_feature_source_date": str(feature_dates.max().date()),
    }
    for field, expected in observed.items():
        if value[field] != expected:
            raise V18Error(f"G0 target-slice decoded value changed: {field}")
    for field in (
        "data_sha256",
        "data_semantic_sha256",
        "target_date_scoring_input_semantic_sha256",
        "target_slice_semantic_sha256",
        "cache_manifest_sha256",
    ):
        _require_sha(value[field], f"G0 target-slice {field}")
    return value, frame, expected_manifest_key, _json_file_bytes(value)


def materialize_g0_panel_cache(
    parsed_prices: pd.DataFrame,
    *,
    predictor_derived_store_root: str | Path,
    target_session: Any,
    latest_required_source_session: Any,
    source_set_sha256: str,
    parsed_shard_set_sha256: str,
    total_parsed_row_count: int,
    prepared_full_panel: pd.DataFrame | None = None,
    runtime_lock_verified_at: Any | None = None,
    created_at: Any | None = None,
    sealed_at: Any | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, str, bytes]:
    """Persist a tiny target slice; the cumulative panel remains in memory only."""

    target = _date(target_session, "G0 target session")
    latest = _date(latest_required_source_session, "G0 latest source")
    if latest >= target or isinstance(total_parsed_row_count, bool) or int(
        total_parsed_row_count
    ) <= 0:
        raise V18Error("G0 target-slice input identity is invalid")
    panel = (
        build_forward_c00_panel(parsed_prices, target)
        if prepared_full_panel is None
        else prepared_full_panel
    )
    target_rows = _coerce_jsonl_frame(
        panel.loc[pd.to_datetime(panel["date"], errors="coerce").eq(target)],
        G0_PANEL_COLUMNS,
        label="G0 target-slice source",
    )
    payload = canonical_frame_jsonl_bytes(
        target_rows, G0_PANEL_COLUMNS, label="G0 target-slice source"
    )
    decoded_preflight = decode_canonical_frame_jsonl(
        payload, G0_PANEL_COLUMNS, label="G0 target-slice preflight"
    )
    if not decoded_preflight.equals(target_rows):
        raise V18Error("G0 target-slice codec is not exactly reversible")
    _, data_key, manifest_key = _g0_cache_identity(
        target_session=target,
        latest_required_source_session=latest,
        source_set_sha256=source_set_sha256,
        parsed_shard_set_sha256=parsed_shard_set_sha256,
    )
    if _external_object_exists(
        predictor_derived_store_root,
        manifest_key,
        prefix=G0_PANEL_CACHE_OBJECT_PREFIX,
        label="G0 target-slice manifest",
    ):
        existing, _ = _read_external_canonical_json(
            predictor_derived_store_root,
            manifest_key,
            prefix=G0_PANEL_CACHE_OBJECT_PREFIX,
            label="G0 target-slice manifest",
        )
        validated, retained_frame, retained_key, retained_payload = (
            _validate_g0_cache_manifest(
                existing,
                predictor_derived_store_root=predictor_derived_store_root,
                expected_target_session=target,
                expected_latest_source_session=latest,
                expected_source_set_sha256=source_set_sha256,
                expected_parsed_shard_set_sha256=parsed_shard_set_sha256,
            )
        )
        if retained_key != manifest_key:
            raise V18Error("G0 target-slice retry resolved another identity")
        if not retained_frame.equals(target_rows):
            raise V18Error("G0 target-slice retry changes exact rows/dtypes")
        if int(validated["parsed_row_count"]) != int(total_parsed_row_count):
            raise V18Error("G0 target-slice retry changes parsed-row count")
        return validated, retained_frame, retained_key, retained_payload
    data_count, data_hash = _write_external_bytes_once(
        payload,
        store_root=predictor_derived_store_root,
        object_key=data_key,
        prefix=G0_PANEL_CACHE_OBJECT_PREFIX,
        label="G0 target-slice data",
    )
    verified = _runtime_verified_timestamp(runtime_lock_verified_at)
    created = _operation_timestamp(created_at, "G0 target-slice created_at")
    sealed = _operation_timestamp(sealed_at, "G0 target-slice sealed_at")
    feature_dates = pd.to_datetime(
        target_rows["feature_source_max_date"], errors="coerce"
    )
    scoring_columns = (
        "date",
        "code",
        "name",
        "common_score_eligible",
        "feature_source_max_date",
        *G0_FEATURES,
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "cache_contract_id": PREDICTOR_CACHE_CONTRACT_ID,
        "scope": "target_slice",
        "target_session": str(target.date()),
        "latest_required_source_session": str(latest.date()),
        "source_set_sha256": _require_sha(source_set_sha256, "G0 source-set SHA"),
        "parsed_shard_set_sha256": _require_sha(
            parsed_shard_set_sha256, "G0 shard-set SHA"
        ),
        "parsed_row_count": int(total_parsed_row_count),
        "columns": list(G0_PANEL_COLUMNS),
        "columns_sha256": canonical_json_sha256(list(G0_PANEL_COLUMNS)),
        "data_object_key": data_key,
        "data_byte_count": data_count,
        "data_sha256": data_hash,
        "row_count": int(len(target_rows)),
        "unique_date_count": 1,
        "duplicate_date_code_count": 0,
        "data_semantic_sha256": semantic_frame_sha256(
            target_rows, G0_PANEL_COLUMNS
        ),
        "target_row_count": int(len(target_rows)),
        "target_date_scoring_input_semantic_sha256": semantic_frame_sha256(
            target_rows, scoring_columns
        ),
        "target_slice_semantic_sha256": semantic_frame_sha256(
            target_rows, G0_PANEL_COLUMNS
        ),
        "target_outcome_nonnull_count": int(
            target_rows["oc_return_pct"].notna().sum()
        ),
        "max_feature_source_date": (
            None if feature_dates.isna().all() else str(feature_dates.max().date())
        ),
        "v17_protocol_sha256": V17_BINDINGS["protocol"][1],
        "v17_runner_sha256": V17_BINDINGS["runner"][1],
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "runtime_lock_verified_at": verified,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "parser_sha256": JPX_PARSER_SHA256,
        "created_at": created,
        "sealed_at": sealed,
        "canonical_jsonl_contract": G0_CACHE_JSONL_CONTRACT,
    }
    value["cache_manifest_sha256"] = canonical_json_sha256(
        value, exclude_fields={"cache_manifest_sha256"}
    )
    manifest_payload = _json_file_bytes(value)
    _write_external_bytes_once(
        manifest_payload,
        store_root=predictor_derived_store_root,
        object_key=manifest_key,
        prefix=G0_PANEL_CACHE_OBJECT_PREFIX,
        label="G0 target-slice manifest",
        replace_unpublished_stage=True,
    )
    return _validate_g0_cache_manifest(
        value,
        predictor_derived_store_root=predictor_derived_store_root,
        expected_target_session=target,
        expected_latest_source_session=latest,
        expected_source_set_sha256=source_set_sha256,
        expected_parsed_shard_set_sha256=parsed_shard_set_sha256,
    )


def _direct_parse_predictor_raw_records(
    raw_records: Sequence[Mapping[str, Any]],
    *,
    predictor_raw_store_root: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    with _external_snapshot_paths(
        predictor_raw_store_root,
        raw_records,
        prefix=PREDICTOR_OBJECT_PREFIX,
        label="predictor",
    ) as (raw_paths, _):
        try:
            prices, report = _collect_jpx_registered(raw_paths)
        except Exception as exc:
            raise V18Error(f"predictor raw clean-room parse failed: {exc}") from exc
    inputs = report.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != len(raw_records):
        raise V18Error("predictor clean-room parser did not bind every raw object")
    if any(int(item.get("rejected_rows", -1)) != 0 for item in inputs):
        raise V18Error("predictor clean-room parser rejected a row")
    canonical = _coerce_jsonl_frame(
        prices, PARSED_PRICE_COLUMNS, label="predictor clean-room raw parse"
    )
    if canonical[["date", "code"]].duplicated().any():
        raise V18Error("predictor clean-room parse produced duplicates")
    return canonical, dict(report)


def _terminal_reparse_predictor_shards_once(
    raw_records: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
    *,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
    identity_registry: dict[tuple[int, int], str] | None = None,
) -> pd.DataFrame:
    """Reparse each unique raw once and verify every standalone shard claim."""

    if len(raw_records) != len(bindings) or not raw_records:
        raise V18Error("terminal raw/shard union is empty or misaligned")
    frames: list[pd.DataFrame] = []
    for raw_record, binding in zip(raw_records, bindings, strict=True):
        manifest, _ = _read_external_canonical_json(
            predictor_derived_store_root,
            binding["shard_manifest_object_key"],
            prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
            label="terminal parsed-shard manifest",
        )
        validated, cached_frame, observed_binding = _validate_parsed_shard_manifest(
            manifest,
            raw_record=raw_record,
            predictor_derived_store_root=predictor_derived_store_root,
            decode_data=True,
        )
        if observed_binding != dict(binding) or cached_frame is None:
            raise V18Error("terminal parsed-shard binding changed")
        with _external_snapshot_paths(
            predictor_raw_store_root,
            [raw_record],
            prefix=PREDICTOR_OBJECT_PREFIX,
            label="terminal predictor shard",
            identity_registry=identity_registry,
        ) as (raw_paths, _):
            try:
                direct_prices, report = _collect_jpx_registered(raw_paths)
            except Exception as exc:
                raise V18Error(
                    f"terminal standalone predictor parse failed: {exc}"
                ) from exc
            text_path = raw_paths[0].with_suffix(".txt")
            if text_path.is_file() and not text_path.is_symlink():
                text_payload = _plain_file_bytes(
                    text_path, label="terminal locked pdftotext output"
                )
                text_count = len(text_payload)
                text_hash = hashlib.sha256(text_payload).hexdigest()
            else:
                report_inputs = report.get("inputs")
                if not isinstance(report_inputs, list) or len(report_inputs) != 1:
                    raise V18Error("terminal standalone parser report is incomplete")
                text_count = int(report_inputs[0].get("text_byte_count", -1))
                text_hash = _require_sha(
                    report_inputs[0].get("text_sha256"),
                    "terminal standalone text SHA",
                )
        report_inputs = report.get("inputs")
        if not isinstance(report_inputs, list) or len(report_inputs) != 1:
            raise V18Error("terminal standalone parser report count changed")
        parser_input = dict(report_inputs[0])
        if int(parser_input.get("rejected_rows", -1)) != 0:
            raise V18Error("terminal standalone parser rejected a row")
        if str(parser_input.get("sha256", "")) != text_hash:
            raise V18Error("terminal standalone text/report SHA changed")
        parser_input["path"] = Path(str(raw_record["file"])).with_suffix(
            ".txt"
        ).name
        report_projection = {**dict(report), "inputs": [parser_input]}
        direct = _coerce_jsonl_frame(
            direct_prices,
            PARSED_PRICE_COLUMNS,
            label="terminal standalone parsed shard",
        )
        direct_payload = canonical_frame_jsonl_bytes(
            direct,
            PARSED_PRICE_COLUMNS,
            label="terminal standalone parsed shard",
        )
        exact = {
            "row_count": int(len(direct)),
            "rejected_row_count": 0,
            "duplicate_date_code_count": int(
                direct[["date", "code"]].duplicated(keep=False).sum()
            ),
            "pdftotext_text_byte_count": text_count,
            "pdftotext_text_sha256": text_hash,
            "parser_report_sha256": canonical_json_sha256(report_projection),
            "data_byte_count": len(direct_payload),
            "data_sha256": hashlib.sha256(direct_payload).hexdigest(),
            "parsed_semantic_sha256": parsed_panel_semantic_sha256(direct),
        }
        for field, expected in exact.items():
            if validated[field] != expected:
                raise V18Error(f"terminal standalone shard changed: {field}")
        if not direct.equals(cached_frame):
            raise V18Error("terminal standalone raw differs exactly from shard")
        frames.append(direct)
    try:
        merged = merge_daily_prices(frames)
    except Exception as exc:
        raise V18Error(f"terminal standalone shard merge failed: {exc}") from exc
    return _coerce_jsonl_frame(
        merged, PARSED_PRICE_COLUMNS, label="terminal predictor raw union"
    )


def _cache_anchor_identity(
    *,
    latest_source_session: pd.Timestamp,
    raw_source_set_sha256: str,
    ordered_shard_set_sha256: str,
) -> tuple[str, str, str]:
    identity = {
        "cache_contract_id": PREDICTOR_CACHE_CONTRACT_ID,
        "latest_source_session": str(latest_source_session.date()),
        "raw_source_set_sha256": raw_source_set_sha256,
        "ordered_shard_set_sha256": ordered_shard_set_sha256,
        "columns_sha256": canonical_json_sha256(list(PARSED_PRICE_COLUMNS)),
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "parser_sha256": JPX_PARSER_SHA256,
        "canonical_jsonl_contract": PARSED_SHARD_JSONL_CONTRACT,
    }
    token = canonical_json_sha256(identity)
    return (
        token,
        f"{CACHE_ANCHOR_OBJECT_PREFIX}{token}.jsonl",
        f"{CACHE_ANCHOR_OBJECT_PREFIX}{token}.manifest.json",
    )


def validate_predictor_cache_anchor(
    anchor: Mapping[str, Any],
    *,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
    require_direct_clean_room_reparse: bool,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    """Validate the immutable historical anchor; optionally recompute raw first."""

    value = dict(anchor)
    if set(value) != set(CACHE_ANCHOR_FIELDS):
        raise V18Error("predictor cache anchor fields differ from A2 contract")
    if value["schema_version"] != 1 or value["cache_contract_id"] != (
        PREDICTOR_CACHE_CONTRACT_ID
    ):
        raise V18Error("predictor cache anchor schema/contract changed")
    latest = _date(value["latest_source_session"], "cache anchor latest session")
    raw_records = value["raw_sources"]
    bindings = value["parsed_shards"]
    if not isinstance(raw_records, list) or not isinstance(bindings, list):
        raise V18Error("cache anchor raw/shard sets must be arrays")
    expected_files, _ = _expected_predictor_files(latest)
    if [str(item.get("file")) for item in raw_records] != expected_files:
        raise V18Error("cache anchor does not cover exact cumulative registry")
    if len(raw_records) != int(value["raw_source_count"]) or len(bindings) != int(
        value["ordered_shard_count"]
    ):
        raise V18Error("cache anchor raw/shard counts differ")
    raw_set_hash = canonical_json_sha256(raw_records)
    shard_set_hash = _parsed_shard_set_sha256(bindings)
    if (
        value["raw_source_set_sha256"] != raw_set_hash
        or value["ordered_shard_set_sha256"] != shard_set_hash
    ):
        raise V18Error("cache anchor ordered set hash changed")
    _, expected_data_key, expected_manifest_key = _cache_anchor_identity(
        latest_source_session=latest,
        raw_source_set_sha256=raw_set_hash,
        ordered_shard_set_sha256=shard_set_hash,
    )
    fixed = {
        "cumulative_snapshot_object_key": expected_data_key,
        "snapshot_manifest_object_key": expected_manifest_key,
        "columns": list(PARSED_PRICE_COLUMNS),
        "columns_sha256": canonical_json_sha256(list(PARSED_PRICE_COLUMNS)),
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "parser_sha256": JPX_PARSER_SHA256,
        "canonical_jsonl_contract": PARSED_SHARD_JSONL_CONTRACT,
    }
    for field, expected in fixed.items():
        if value[field] != expected:
            raise V18Error(f"cache anchor binding changed: {field}")
    for field in (
        "raw_source_set_sha256",
        "ordered_shard_set_sha256",
        "cumulative_snapshot_file_sha256",
        "cumulative_snapshot_semantic_sha256",
        "model_price_snapshot_file_sha256",
        "model_price_snapshot_semantic_sha256",
        "model_price_snapshot_manifest_file_sha256",
        "model_price_snapshot_manifest_sha256",
        "snapshot_manifest_sha256",
        "direct_clean_room_verification_receipt_sha256",
        "anchor_manifest_sha256",
    ):
        _require_sha(value[field], f"cache anchor {field}")
    equivalence_receipt = validate_compact_consumer_equivalence_receipt(
        value["compact_consumer_equivalence_receipt"]
    )
    if value["snapshot_manifest_sha256"] != value["anchor_manifest_sha256"]:
        raise V18Error("cache anchor nested/self manifest hashes differ")
    expected_self = canonical_json_sha256(
        value, exclude_fields={"anchor_manifest_sha256", "snapshot_manifest_sha256"}
    )
    if value["anchor_manifest_sha256"] != expected_self:
        raise V18Error("cache anchor self-hash mismatch")
    verified = _timestamp(
        value["runtime_lock_verified_at"], "cache anchor runtime verified"
    )
    direct_started = _timestamp(
        value["direct_reparse_started_at"], "cache anchor direct reparse started"
    )
    direct_completed = _timestamp(
        value["direct_reparse_completed_at"], "cache anchor direct reparse completed"
    )
    created = _timestamp(value["created_at"], "cache anchor created_at")
    sealed = _timestamp(value["sealed_at"], "cache anchor sealed_at")
    observed_verified = _timestamp(value["verified_at"], "cache anchor verified_at")
    if not (
        verified <= direct_started <= direct_completed <= created <= sealed
        and observed_verified == direct_completed
    ):
        raise V18Error("cache anchor timestamp DAG is invalid")
    shard_prices: pd.DataFrame | None
    if require_direct_clean_room_reparse:
        shard_prices = _load_bound_predictor_shards(
            raw_records,
            bindings,
            predictor_raw_store_root=predictor_raw_store_root,
            predictor_derived_store_root=predictor_derived_store_root,
        )
    else:
        _validate_bound_predictor_shard_metadata(
            raw_records,
            bindings,
            predictor_raw_store_root=predictor_raw_store_root,
            predictor_derived_store_root=predictor_derived_store_root,
        )
        shard_prices = None
    anchor_shard_sealed_at: list[datetime] = []
    for raw_record, binding in zip(raw_records, bindings, strict=True):
        shard_manifest, _ = _read_external_canonical_json(
            predictor_derived_store_root,
            str(binding["shard_manifest_object_key"]),
            prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
            label="anchor parsed shard manifest",
        )
        if shard_manifest["chronology_class"] != "anchor" or shard_manifest[
            "raw_received_at"
        ] is not None:
            raise V18Error("cache anchor contains a forward-chronology shard")
        shard_sealed = _timestamp(
            shard_manifest["sealed_at"], "anchor shard sealed_at"
        )
        anchor_shard_sealed_at.append(shard_sealed)
        if shard_sealed > direct_started:
            raise V18Error("cache anchor direct verification predates shard seal")
    snapshot_metadata = _external_object_metadata(
        predictor_derived_store_root,
        expected_data_key,
        prefix=CACHE_ANCHOR_OBJECT_PREFIX,
        label="predictor cache anchor snapshot",
        required_signature=None,
    )
    if snapshot_metadata != (
        int(value["cumulative_snapshot_byte_count"]),
        str(value["cumulative_snapshot_file_sha256"]),
    ):
        raise V18Error("cache anchor cumulative snapshot bytes changed")
    snapshot: pd.DataFrame | None = None
    if require_direct_clean_room_reparse:
        snapshot_payload = _external_object_bytes(
            predictor_derived_store_root,
            expected_data_key,
            prefix=CACHE_ANCHOR_OBJECT_PREFIX,
            label="predictor cache anchor snapshot",
        )
        snapshot = decode_canonical_frame_jsonl(
            snapshot_payload, PARSED_PRICE_COLUMNS, label="cache anchor snapshot"
        )
        if shard_prices is None or not snapshot.equals(shard_prices):
            raise V18Error("cache anchor snapshot differs exactly from parsed shards")
        if value["cumulative_snapshot_semantic_sha256"] != (
            parsed_panel_semantic_sha256(snapshot)
        ):
            raise V18Error("cache anchor snapshot semantic hash changed")
    model_month = _month(
        value["model_price_snapshot_target_month"],
        "cache anchor model-price target month",
    )
    model_latest = _latest_registered_source_before_month(model_month)
    model_files, _ = _expected_predictor_files(model_latest)
    model_prefix_count = len(model_files)
    if [item["file"] for item in raw_records[:model_prefix_count]] != model_files:
        raise V18Error("cache anchor compact snapshot raw prefix changed")
    model_manifest, model_manifest_payload = _read_external_canonical_json(
        predictor_derived_store_root,
        value["model_price_snapshot_manifest_object_key"],
        prefix=MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX,
        label="cache anchor model-price snapshot manifest",
    )
    model_snapshot, model_prices, model_manifest_key, _ = (
        _validate_model_price_snapshot_manifest(
            model_manifest,
            predictor_derived_store_root=predictor_derived_store_root,
            expected_target_month=model_month,
            expected_latest_source_session=model_latest,
            expected_raw_source_set_sha256=canonical_json_sha256(
                raw_records[:model_prefix_count]
            ),
            expected_parsed_shard_set_sha256=_parsed_shard_set_sha256(
                bindings[:model_prefix_count]
            ),
            decode_data=require_direct_clean_room_reparse,
        )
    )
    model_snapshot_created = _timestamp(
        model_snapshot["created_at"], "anchor compact snapshot created_at"
    )
    model_snapshot_sealed = _timestamp(
        model_snapshot["sealed_at"], "anchor compact snapshot sealed_at"
    )
    if (
        max(anchor_shard_sealed_at[:model_prefix_count]) > model_snapshot_created
        or model_snapshot_created > model_snapshot_sealed
        or model_snapshot_sealed > created
    ):
        raise V18Error("cache anchor compact snapshot timestamp DAG is invalid")
    exact_model_bindings = {
        "model_price_snapshot_object_key": model_snapshot["data_object_key"],
        "model_price_snapshot_byte_count": int(model_snapshot["data_byte_count"]),
        "model_price_snapshot_file_sha256": model_snapshot["data_sha256"],
        "model_price_snapshot_semantic_sha256": model_snapshot[
            "model_price_semantic_sha256"
        ],
        "model_price_snapshot_manifest_object_key": model_manifest_key,
        "model_price_snapshot_manifest_byte_count": len(model_manifest_payload),
        "model_price_snapshot_manifest_file_sha256": hashlib.sha256(
            model_manifest_payload
        ).hexdigest(),
        "model_price_snapshot_manifest_sha256": model_snapshot[
            "snapshot_manifest_sha256"
        ],
    }
    for field, expected in exact_model_bindings.items():
        if value[field] != expected:
            raise V18Error(f"cache anchor compact snapshot changed: {field}")
    if require_direct_clean_room_reparse:
        if snapshot is None:
            raise V18Error("cache anchor direct validation lacks its snapshot")
        direct_model_prices = _coerce_model_price_frame(
            snapshot.loc[pd.to_datetime(snapshot["date"]).le(model_latest)],
            label="cache anchor direct compact model-price projection",
        )
        if not direct_model_prices.equals(model_prices):
            raise V18Error(
                "cache anchor compact snapshot differs from exact raw projection"
            )
    receipt = {
        "cache_contract_id": PREDICTOR_CACHE_CONTRACT_ID,
        "raw_source_set_sha256": raw_set_hash,
        "ordered_shard_set_sha256": shard_set_hash,
        "direct_reparse_started_at": value["direct_reparse_started_at"],
        "direct_reparse_completed_at": value["direct_reparse_completed_at"],
        "direct_parsed_semantic_sha256": value[
            "cumulative_snapshot_semantic_sha256"
        ],
        "snapshot_semantic_sha256": value["cumulative_snapshot_semantic_sha256"],
        "model_price_snapshot_semantic_sha256": value[
            "model_price_snapshot_semantic_sha256"
        ],
        "exact_frame_and_dtype_equal": True,
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
    }
    if canonical_json_sha256(receipt) != value[
        "direct_clean_room_verification_receipt_sha256"
    ]:
        raise V18Error("cache anchor clean-room receipt hash changed")
    if require_direct_clean_room_reparse:
        fresh, _ = _direct_parse_predictor_raw_records(
            raw_records, predictor_raw_store_root=predictor_raw_store_root
        )
        if snapshot is None or not fresh.equals(snapshot):
            raise V18Error(
                "activation clean-room raw parse differs exactly from cache anchor"
            )
        if parsed_panel_semantic_sha256(fresh) != value[
            "cumulative_snapshot_semantic_sha256"
        ]:
            raise V18Error("activation clean-room raw semantic differs from anchor")
        fresh_model_prices = _coerce_model_price_frame(
            fresh.loc[pd.to_datetime(fresh["date"]).le(model_latest)],
            label="activation clean-room compact model-price projection",
        )
        if not fresh_model_prices.equals(model_prices):
            raise V18Error(
                "activation clean-room raw differs from compact model-price snapshot"
            )
        fresh_equivalence = build_compact_consumer_equivalence_receipt(
            fresh,
            _coerce_model_price_frame(
                fresh, label="activation compact-consumer full prefix"
            ),
            synthetic_target_session=equivalence_receipt[
                "synthetic_target_session"
            ],
        )
        if fresh_equivalence != equivalence_receipt:
            raise V18Error(
                "activation compact-consumer proof differs from anchor receipt"
            )
    manifest_payload = _json_file_bytes(value)
    summary = {
        "latest_source_session": value["latest_source_session"],
        "raw_source_set_sha256": raw_set_hash,
        "raw_source_count": len(raw_records),
        "ordered_shard_set_sha256": shard_set_hash,
        "ordered_shard_count": len(bindings),
        "cumulative_snapshot_object_key": expected_data_key,
        "cumulative_snapshot_byte_count": int(value["cumulative_snapshot_byte_count"]),
        "cumulative_snapshot_file_sha256": value[
            "cumulative_snapshot_file_sha256"
        ],
        "cumulative_snapshot_semantic_sha256": value[
            "cumulative_snapshot_semantic_sha256"
        ],
        "model_price_snapshot_target_month": value[
            "model_price_snapshot_target_month"
        ],
        **exact_model_bindings,
        "snapshot_manifest_object_key": expected_manifest_key,
        "snapshot_manifest_file_sha256": hashlib.sha256(
            manifest_payload
        ).hexdigest(),
        "snapshot_manifest_sha256": value["snapshot_manifest_sha256"],
        "direct_clean_room_verification_receipt_sha256": value[
            "direct_clean_room_verification_receipt_sha256"
        ],
        "compact_consumer_equivalence_receipt_sha256": equivalence_receipt[
            "receipt_sha256"
        ],
        "sealed_at": value["sealed_at"],
        "verified_at": value["verified_at"],
    }
    return value, summary, manifest_payload


def build_predictor_cache_anchor(
    source_pdfs: Sequence[str | Path],
    source_file_names: Sequence[str],
    source_urls: Sequence[str],
    *,
    through_session: Any,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
    runtime_lock_verified_at: Any | None = None,
    output_object_key: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build and twice-check the historical cache anchor before payload B."""

    latest = _date(through_session, "cache anchor through session")
    expected_files, kinds = _expected_predictor_files(latest)
    names = [str(item) for item in source_file_names]
    if (
        names != expected_files
        or len(source_pdfs) != len(names)
        or len(source_urls) != len(names)
    ):
        raise V18Error("cache anchor inputs differ from exact cumulative registry")
    _validate_external_store_disjointness(
        predictor_raw_store_root, predictor_derived_store_root
    )
    verified = _runtime_verified_timestamp(runtime_lock_verified_at)
    historical_metadata = _historical_predictor_metadata()
    raw_records: list[dict[str, Any]] = []
    for index, (source_pdf, name, source_url) in enumerate(
        zip(source_pdfs, names, source_urls, strict=True)
    ):
        if kinds[name] == "daily":
            official = _official_jpx_daily_url(
                source_url,
                f"cache anchor source_urls[{index}]",
                file_name=name,
                source_session=_source_file_date(name, "daily"),
            )
        else:
            official = _official_jpx_url(
                source_url, f"cache anchor source_urls[{index}]"
            )
        count, digest = _source_file_metadata(source_pdf, label="cache anchor predictor")
        bound = historical_metadata.get(name)
        if bound is not None:
            if digest != str(bound["sha256"]) or (
                "bytes" in bound and count != int(bound["bytes"])
            ):
                raise V18Error(f"cache anchor historical raw changed: {name}")
            if bound.get("source_url") is not None and official != str(
                bound["source_url"]
            ):
                raise V18Error(f"cache anchor historical URL changed: {name}")
        key = _predictor_object_key(name, kinds[name])
        retained_count, retained_hash = _seal_predictor_object(
            source_pdf,
            predictor_raw_store_root=predictor_raw_store_root,
            object_key=key,
        )
        raw_records.append(
            {
                "object_key": key,
                "file": name,
                "url": official,
                "byte_count": retained_count,
                "sha256": retained_hash,
            }
        )
    bindings: list[dict[str, Any]] = []
    for raw_record in raw_records:
        shard, _, binding = ensure_predictor_parsed_shard(
            raw_record,
            predictor_raw_store_root=predictor_raw_store_root,
            predictor_derived_store_root=predictor_derived_store_root,
            chronology_class="anchor",
            raw_received_at=None,
            runtime_lock_verified_at=verified,
        )
        if shard["chronology_class"] != "anchor":  # pragma: no cover
            raise V18Error("cache anchor unexpectedly reused a forward shard")
        bindings.append(binding)
    shard_prices = _load_bound_predictor_shards(
        raw_records,
        bindings,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
    )
    raw_set_hash = canonical_json_sha256(raw_records)
    shard_set_hash = _parsed_shard_set_sha256(bindings)
    _, snapshot_key, manifest_key = _cache_anchor_identity(
        latest_source_session=latest,
        raw_source_set_sha256=raw_set_hash,
        ordered_shard_set_sha256=shard_set_hash,
    )
    if output_object_key is not None and str(output_object_key) != manifest_key:
        raise V18Error("cache anchor output key is caller-selected")
    snapshot_payload = canonical_frame_jsonl_bytes(
        shard_prices, PARSED_PRICE_COLUMNS, label="cache anchor snapshot"
    )
    snapshot_count, snapshot_hash = _write_external_bytes_once(
        snapshot_payload,
        store_root=predictor_derived_store_root,
        object_key=snapshot_key,
        prefix=CACHE_ANCHOR_OBJECT_PREFIX,
        label="predictor cache anchor snapshot",
    )
    direct_started = datetime.now(TOKYO)
    direct_prices, _ = _direct_parse_predictor_raw_records(
        raw_records, predictor_raw_store_root=predictor_raw_store_root
    )
    direct_completed = datetime.now(TOKYO)
    if not direct_prices.equals(shard_prices):
        raise V18Error("cache anchor direct raw parse differs from parsed shards")
    semantic_hash = parsed_panel_semantic_sha256(direct_prices)
    calendar = load_registered_calendar()
    future_sessions = calendar[calendar > latest]
    if len(future_sessions) == 0:
        raise V18Error("cache anchor has no registered target month after its prefix")
    synthetic_target = pd.Timestamp(future_sessions[0])
    initial_target_month = synthetic_target.to_period("M")
    equivalence_receipt = build_compact_consumer_equivalence_receipt(
        direct_prices,
        _coerce_model_price_frame(
            direct_prices, label="cache anchor compact-consumer full prefix"
        ),
        synthetic_target_session=synthetic_target,
    )
    model_latest = _latest_registered_source_before_month(initial_target_month)
    model_prefix_files, _ = _expected_predictor_files(model_latest)
    model_prefix_count = len(model_prefix_files)
    if names[:model_prefix_count] != model_prefix_files:
        raise V18Error("cache anchor model-price prefix is not an exact raw prefix")
    model_prices = _coerce_model_price_frame(
        direct_prices.loc[
            pd.to_datetime(direct_prices["date"]).le(model_latest)
        ],
        label="cache anchor model-price prefix",
    )
    (
        model_snapshot,
        validated_model_prices,
        model_snapshot_manifest_key,
        model_snapshot_manifest_payload,
    ) = materialize_model_price_snapshot(
        model_prices,
        predictor_derived_store_root=predictor_derived_store_root,
        target_month=initial_target_month,
        latest_source_session=model_latest,
        raw_records=raw_records[:model_prefix_count],
        shard_bindings=bindings[:model_prefix_count],
        previous_snapshot_manifest_sha256=None,
        runtime_lock_verified_at=verified,
    )
    if not validated_model_prices.equals(model_prices):
        raise V18Error("cache anchor compact model-price snapshot changed rows/dtypes")
    receipt = {
        "cache_contract_id": PREDICTOR_CACHE_CONTRACT_ID,
        "raw_source_set_sha256": raw_set_hash,
        "ordered_shard_set_sha256": shard_set_hash,
        "direct_reparse_started_at": direct_started,
        "direct_reparse_completed_at": direct_completed,
        "direct_parsed_semantic_sha256": semantic_hash,
        "snapshot_semantic_sha256": semantic_hash,
        "model_price_snapshot_semantic_sha256": model_snapshot[
            "model_price_semantic_sha256"
        ],
        "exact_frame_and_dtype_equal": True,
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
    }
    created = datetime.now(TOKYO)
    sealed = datetime.now(TOKYO)
    value: dict[str, Any] = {
        "schema_version": 1,
        "cache_contract_id": PREDICTOR_CACHE_CONTRACT_ID,
        "latest_source_session": str(latest.date()),
        "raw_source_set_sha256": raw_set_hash,
        "raw_source_count": len(raw_records),
        "raw_sources": raw_records,
        "ordered_shard_set_sha256": shard_set_hash,
        "ordered_shard_count": len(bindings),
        "parsed_shards": bindings,
        "columns": list(PARSED_PRICE_COLUMNS),
        "columns_sha256": canonical_json_sha256(list(PARSED_PRICE_COLUMNS)),
        "cumulative_snapshot_object_key": snapshot_key,
        "cumulative_snapshot_byte_count": snapshot_count,
        "cumulative_snapshot_file_sha256": snapshot_hash,
        "cumulative_snapshot_semantic_sha256": semantic_hash,
        "model_price_snapshot_target_month": str(initial_target_month),
        "model_price_snapshot_object_key": model_snapshot["data_object_key"],
        "model_price_snapshot_byte_count": model_snapshot["data_byte_count"],
        "model_price_snapshot_file_sha256": model_snapshot["data_sha256"],
        "model_price_snapshot_semantic_sha256": model_snapshot[
            "model_price_semantic_sha256"
        ],
        "model_price_snapshot_manifest_object_key": model_snapshot_manifest_key,
        "model_price_snapshot_manifest_byte_count": len(
            model_snapshot_manifest_payload
        ),
        "model_price_snapshot_manifest_file_sha256": hashlib.sha256(
            model_snapshot_manifest_payload
        ).hexdigest(),
        "model_price_snapshot_manifest_sha256": model_snapshot[
            "snapshot_manifest_sha256"
        ],
        "snapshot_manifest_object_key": manifest_key,
        "snapshot_manifest_sha256": ZERO_SHA256,
        "direct_reparse_started_at": direct_started,
        "direct_reparse_completed_at": direct_completed,
        "direct_clean_room_verification_receipt_sha256": canonical_json_sha256(receipt),
        "compact_consumer_equivalence_receipt": equivalence_receipt,
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "runtime_lock_verified_at": verified,
        "created_at": created,
        "sealed_at": sealed,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "parser_sha256": JPX_PARSER_SHA256,
        "canonical_jsonl_contract": PARSED_SHARD_JSONL_CONTRACT,
        "verified_at": direct_completed,
    }
    self_hash = canonical_json_sha256(
        value,
        exclude_fields={"anchor_manifest_sha256", "snapshot_manifest_sha256"},
    )
    value["snapshot_manifest_sha256"] = self_hash
    value["anchor_manifest_sha256"] = self_hash
    manifest_payload = _json_file_bytes(value)
    _write_external_bytes_once(
        manifest_payload,
        store_root=predictor_derived_store_root,
        object_key=manifest_key,
        prefix=CACHE_ANCHOR_OBJECT_PREFIX,
        label="predictor cache anchor manifest",
        replace_unpublished_stage=True,
    )
    persisted, persisted_payload = _read_external_canonical_json(
        predictor_derived_store_root,
        manifest_key,
        prefix=CACHE_ANCHOR_OBJECT_PREFIX,
        label="predictor cache anchor manifest",
    )
    if persisted_payload != manifest_payload:
        raise V18Error("persisted cache anchor manifest bytes changed")
    validated, summary, _ = validate_predictor_cache_anchor(
        persisted,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
        require_direct_clean_room_reparse=True,
    )
    return validated, summary


def semantic_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Hash a registered date/code view as project-canonical JSON rows."""

    if not isinstance(frame, pd.DataFrame):
        raise V18Error("semantic view must be a DataFrame")
    missing = [column for column in columns if column not in frame]
    if missing:
        raise V18Error(f"semantic view lacks registered columns: {missing}")
    if not {"date", "code"}.issubset(columns):
        raise V18Error("semantic view requires date and code")
    view = frame.loc[:, list(columns)].copy()
    view["date"] = pd.to_datetime(view["date"], errors="coerce", format="mixed")
    if view["date"].isna().any():
        raise V18Error("semantic view contains an invalid date")
    view["code"] = view["code"].astype("string")
    if view["code"].isna().any() or view["code"].eq("").any():
        raise V18Error("semantic view contains an invalid code")
    if view[["date", "code"]].duplicated().any():
        raise V18Error("semantic view contains duplicate date/code rows")
    view = view.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
    records = [
        {
            column: _semantic_cell(row[column], column=column)
            for column in columns
        }
        for row in view.to_dict(orient="records")
    ]
    return canonical_json_sha256(records)


def parsed_panel_semantic_sha256(prices: pd.DataFrame) -> str:
    return semantic_frame_sha256(prices, PARSED_PRICE_COLUMNS)


def build_forward_c00_panel(
    parsed_prices: pd.DataFrame,
    session_date: Any,
) -> pd.DataFrame:
    """Create the v1.7 PIT panel through target using only exact D-1 prices."""

    target = _date(session_date, "session_date")
    prices = parsed_prices.copy()
    if "date" not in prices or "code" not in prices or "name" not in prices:
        raise V18Error("parsed predictor prices lack date/code/name")
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce", format="mixed")
    if prices["date"].isna().any() or prices[["date", "code"]].duplicated().any():
        raise V18Error("parsed predictor date/code integrity failed")
    if getattr(prices["date"].dt, "tz", None) is not None:
        prices["date"] = prices["date"].dt.tz_convert(TOKYO).dt.tz_localize(None)
    prices["date"] = prices["date"].dt.normalize()
    latest = prices["date"].max()
    if latest >= target:
        raise V18Error("predictor raw prices contain target or future outcomes")
    latest_rows = prices.loc[prices["date"].eq(latest), ["code", "name"]].copy()
    if latest_rows.empty or latest_rows["code"].duplicated().any():
        raise V18Error("exact D-1 predictor universe is empty or duplicated")
    placeholder = latest_rows.assign(date=target)
    for column in ("open", "high", "low", "close", "volume", "turnover"):
        placeholder[column] = np.nan
    for column in (
        "am_open",
        "am_high",
        "am_low",
        "am_close",
        "pm_open",
        "pm_high",
        "pm_low",
        "pm_close",
    ):
        placeholder[column] = np.nan
    placeholder["traded"] = False
    placeholder["partial_session"] = False
    augmented = pd.concat([prices, placeholder], ignore_index=True, sort=False)
    sessions = pd.DatetimeIndex(sorted(prices["date"].unique())).append(
        pd.DatetimeIndex([target])
    )
    try:
        panel, _, _ = v17.build_model_panel(augmented, sessions)
    except Exception as exc:
        raise V18Error(f"bound v1.7 C00 panel construction failed: {exc}") from exc
    target_rows = panel.loc[pd.to_datetime(panel["date"]).eq(target)]
    if target_rows.empty or target_rows["oc_return_pct"].notna().any():
        raise V18Error("forward C00 target rows are missing or outcome-populated")
    return panel


def _date_code_identity_sha256(frame: pd.DataFrame, *, label: str) -> str:
    view = frame.loc[:, ["date", "code"]].copy()
    view["date"] = pd.to_datetime(view["date"], errors="coerce", format="mixed")
    view["code"] = view["code"].astype("string")
    if (
        view["date"].isna().any()
        or view["code"].isna().any()
        or view["code"].eq("").any()
        or view.duplicated().any()
    ):
        raise V18Error(f"{label} date/code identity is invalid")
    view = view.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
    digest = hashlib.sha256()
    dates = view["date"].astype("datetime64[ns]").astype("<i8").to_numpy()
    for date_bits, code in zip(dates, view["code"].astype(str), strict=True):
        encoded = code.encode("utf-8", errors="strict")
        digest.update(struct.pack("<qI", int(date_bits), len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _exact_g0_frame_digest(frame: pd.DataFrame) -> dict[str, Any]:
    """Digest an exact stable G0 frame without materializing JSON row objects."""

    value = frame.loc[:, list(G0_PANEL_COLUMNS)].copy()
    value["date"] = pd.to_datetime(value["date"], errors="coerce", format="mixed")
    value["code"] = value["code"].astype("string")
    if (
        value["date"].isna().any()
        or value["code"].isna().any()
        or value["code"].eq("").any()
        or value[["date", "code"]].duplicated().any()
    ):
        raise V18Error("compact-consumer G0 frame identity is invalid")
    value = value.sort_values(["date", "code"], kind="stable").reset_index(
        drop=True
    )
    identity_hash = _date_code_identity_sha256(value, label="exact G0 digest")
    column_digests: list[dict[str, Any]] = []
    for column in G0_PANEL_COLUMNS:
        series = value[column]
        nulls = series.isna().to_numpy(dtype=np.uint8)
        null_sha = hashlib.sha256(nulls.tobytes(order="C")).hexdigest()
        dtype_token = str(series.dtype)
        metadata: dict[str, Any] | None = None
        payload_hash = hashlib.sha256()
        if isinstance(series.dtype, pd.CategoricalDtype):
            metadata = {
                "ordered": bool(series.dtype.ordered),
                "categories": [
                    _semantic_cell(item, column=column)
                    for item in series.dtype.categories.tolist()
                ],
            }
            codes_array = series.cat.codes.to_numpy(dtype="<i8", copy=True)
            payload_hash.update(codes_array.tobytes(order="C"))
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            array = pd.to_datetime(series).astype("datetime64[ns]").astype("<i8")
            payload_hash.update(array.to_numpy().tobytes(order="C"))
        elif pd.api.types.is_bool_dtype(series.dtype):
            array = np.zeros(len(series), dtype=np.uint8)
            present = nulls == 0
            array[present] = series.loc[present].astype(bool).to_numpy(
                dtype=np.uint8
            )
            payload_hash.update(array.tobytes(order="C"))
        elif pd.api.types.is_float_dtype(series.dtype):
            array = series.to_numpy(dtype="<f8", na_value=np.nan, copy=True)
            # Null payload bits are not semantic; their exact positions are
            # already committed above.  Every present float retains its
            # little-endian IEEE bytes, including negative zero.
            array[nulls.astype(bool)] = 0.0
            payload_hash.update(array.tobytes(order="C"))
        elif pd.api.types.is_integer_dtype(series.dtype):
            array = np.zeros(len(series), dtype="<i8")
            present = nulls == 0
            array[present] = series.loc[present].astype("int64").to_numpy()
            payload_hash.update(array.tobytes(order="C"))
        else:
            for item, is_null in zip(series.tolist(), nulls, strict=True):
                if is_null:
                    payload_hash.update(b"N")
                    continue
                encoded = canonical_json_bytes(
                    {"v": _semantic_cell(item, column=column)}
                )
                payload_hash.update(b"V" + struct.pack("<I", len(encoded)))
                payload_hash.update(encoded)
        column_digests.append(
            {
                "column": column,
                "dtype": dtype_token,
                "null_bitmap_sha256": null_sha,
                "value_bytes_sha256": payload_hash.hexdigest(),
                "dtype_metadata": metadata,
            }
        )
    digest: dict[str, Any] = {
        "columns": list(G0_PANEL_COLUMNS),
        "row_count": int(len(value)),
        "row_identity_sha256": identity_hash,
        "column_digests": column_digests,
    }
    digest["exact_digest_sha256"] = canonical_json_sha256(
        digest, exclude_fields={"exact_digest_sha256"}
    )
    return digest


def validate_compact_consumer_equivalence_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(receipt)
    if set(value) != set(COMPACT_CONSUMER_EQUIVALENCE_RECEIPT_FIELDS):
        raise V18Error("compact-consumer equivalence receipt fields changed")
    fixed = {
        "schema_version": 1,
        "consumer_projection_columns": list(MODEL_PRICE_COLUMNS),
        "consumer_projection_columns_sha256": canonical_json_sha256(
            list(MODEL_PRICE_COLUMNS)
        ),
        "consumer_union_contract_sha256": canonical_json_sha256(
            {
                key: sorted(columns)
                for key, columns in sorted(MODEL_PRICE_REQUIRED_BY_CONSUMERS.items())
            }
        ),
        "exact_columns_order_dtypes_nulls_ieee_strings_bools_equal": True,
        "canonical_json_contract": CANONICAL_JSON_CONTRACT,
    }
    for field, expected in fixed.items():
        if value[field] != expected:
            raise V18Error(f"compact-consumer receipt changed: {field}")
    for field in (
        "raw_date_code_identity_sha256",
        "historical_session_registry_sha256",
        "synthetic_target_exact_digest_sha256",
    ):
        _require_sha(value[field], f"compact-consumer {field}")
    full_digest = value["full_g0_exact_digest"]
    compact_digest = value["compact_g0_exact_digest"]
    if not isinstance(full_digest, Mapping) or dict(full_digest) != dict(
        compact_digest
    ):
        raise V18Error("full/compact G0 exact digests differ")
    if full_digest.get("columns") != list(G0_PANEL_COLUMNS) or int(
        full_digest.get("row_count", -1)
    ) <= 0:
        raise V18Error("compact-consumer G0 digest shape changed")
    if full_digest.get("exact_digest_sha256") != canonical_json_sha256(
        full_digest, exclude_fields={"exact_digest_sha256"}
    ):
        raise V18Error("compact-consumer G0 digest self-hash changed")
    target = _date(
        value["synthetic_target_session"], "compact-consumer synthetic target"
    )
    latest = _date(
        value["latest_feature_source_session"],
        "compact-consumer latest feature source",
    )
    if latest >= target or int(value["synthetic_target_row_count"]) <= 0:
        raise V18Error("compact-consumer synthetic target chronology changed")
    if int(value["synthetic_target_outcome_nonnull_count"]) != 0:
        raise V18Error("compact-consumer synthetic target contains an outcome")
    max_feature = _date(
        value["synthetic_target_max_feature_source_date"],
        "compact-consumer synthetic feature max",
    )
    if max_feature > latest:
        raise V18Error("compact-consumer feature source exceeds D-1")
    if value["receipt_sha256"] != canonical_json_sha256(
        value, exclude_fields={"receipt_sha256"}
    ):
        raise V18Error("compact-consumer equivalence receipt self-hash changed")
    return value


def build_compact_consumer_equivalence_receipt(
    full_prices: pd.DataFrame,
    compact_prices: pd.DataFrame,
    *,
    synthetic_target_session: Any,
) -> dict[str, Any]:
    """Prove the registered 12-column projection preserves every G0 consumer."""

    target = _date(
        synthetic_target_session, "compact-consumer synthetic target session"
    )
    full_input = _coerce_jsonl_frame(
        full_prices, PARSED_PRICE_COLUMNS, label="compact-consumer full31 input"
    )
    compact_input = _coerce_model_price_frame(
        compact_prices, label="compact-consumer compact12 input"
    )
    if not _coerce_model_price_frame(
        full_input, label="compact-consumer full31 projection"
    ).equals(compact_input):
        raise V18Error("compact-consumer input is not exact full31 projection")
    latest = pd.to_datetime(full_input["date"]).max().normalize()
    if latest >= target:
        raise V18Error("compact-consumer synthetic target is not after its prefix")
    full_panel = build_forward_c00_panel(full_input, target)
    full_digest = _exact_g0_frame_digest(full_panel)
    target_rows = full_panel.loc[pd.to_datetime(full_panel["date"]).eq(target)]
    target_digest = _exact_g0_frame_digest(target_rows)["exact_digest_sha256"]
    target_count = int(len(target_rows))
    target_outcomes = int(target_rows["oc_return_pct"].notna().sum())
    feature_dates = pd.to_datetime(
        target_rows["feature_source_max_date"], errors="coerce"
    )
    if feature_dates.isna().any():
        raise V18Error("compact-consumer target has a null feature source date")
    feature_max = feature_dates.max().normalize()
    del full_panel, target_rows
    gc.collect()
    compact_panel = build_forward_c00_panel(compact_input, target)
    compact_digest = _exact_g0_frame_digest(compact_panel)
    del compact_panel
    gc.collect()
    if full_digest != compact_digest:
        raise V18Error("full31 and compact12 G0 consumers are not bit-exact")
    calendar = load_registered_calendar()
    registry = [str(pd.Timestamp(item).date()) for item in calendar if item <= target]
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "consumer_projection_columns": list(MODEL_PRICE_COLUMNS),
        "consumer_projection_columns_sha256": canonical_json_sha256(
            list(MODEL_PRICE_COLUMNS)
        ),
        "consumer_union_contract_sha256": canonical_json_sha256(
            {
                key: sorted(columns)
                for key, columns in sorted(MODEL_PRICE_REQUIRED_BY_CONSUMERS.items())
            }
        ),
        "raw_date_code_identity_sha256": _date_code_identity_sha256(
            full_input, label="compact-consumer raw input"
        ),
        "historical_session_registry_sha256": canonical_json_sha256(registry),
        "synthetic_target_session": str(target.date()),
        "latest_feature_source_session": str(latest.date()),
        "full_g0_exact_digest": full_digest,
        "compact_g0_exact_digest": compact_digest,
        "synthetic_target_exact_digest_sha256": target_digest,
        "synthetic_target_row_count": target_count,
        "synthetic_target_outcome_nonnull_count": target_outcomes,
        "synthetic_target_max_feature_source_date": str(feature_max.date()),
        "exact_columns_order_dtypes_nulls_ieee_strings_bools_equal": True,
        "canonical_json_contract": CANONICAL_JSON_CONTRACT,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(
        receipt, exclude_fields={"receipt_sha256"}
    )
    return validate_compact_consumer_equivalence_receipt(receipt)


def build_forward_c00_target_slice(
    parsed_prices: pd.DataFrame,
    session_date: Any,
) -> pd.DataFrame:
    """Build the target slice from the exact cumulative compact/full prefix."""

    target = _date(session_date, "target-slice session")
    full_prefix = _coerce_model_price_frame(
        parsed_prices, label="target-slice cumulative model-price prefix"
    )
    panel = build_forward_c00_panel(full_prefix, target)
    target_rows = panel.loc[pd.to_datetime(panel["date"]).eq(target)].copy()
    if target_rows.empty:
        raise V18Error("target-slice panel is empty")
    return _coerce_jsonl_frame(
        target_rows, G0_PANEL_COLUMNS, label="C00 target-slice panel"
    )


def predictor_semantic_hashes(
    parsed_prices: pd.DataFrame,
    panel: pd.DataFrame,
    session_date: Any,
) -> dict[str, str]:
    target = _date(session_date, "session_date")
    dates = pd.to_datetime(panel["date"], errors="coerce", format="mixed")
    if dates.isna().any():
        raise V18Error("C00 panel contains invalid dates")
    through = panel.loc[dates.le(target)].copy()
    target_rows = through.loc[pd.to_datetime(through["date"]).eq(target)].copy()
    g0_columns = (
        "date",
        "code",
        "name",
        "oc_return_pct",
        "common_training_eligible",
        "common_score_eligible",
        "feature_source_max_date",
        *G0_FEATURES,
    )
    universe_columns = (
        "date",
        "code",
        "common_training_eligible",
        "common_score_eligible",
        "feature_source_max_date",
    )
    scoring_columns = (
        "date",
        "code",
        "name",
        "common_score_eligible",
        "feature_source_max_date",
        *G0_FEATURES,
    )
    if target_rows.empty or target_rows["oc_return_pct"].notna().any():
        raise V18Error("target scoring semantic view contains an outcome")
    return {
        "parsed_panel_semantic_sha256": parsed_panel_semantic_sha256(parsed_prices),
        "g0_panel_semantic_sha256": semantic_frame_sha256(through, g0_columns),
        "common_universe_semantic_sha256": semantic_frame_sha256(
            through, universe_columns
        ),
        "target_date_scoring_input_semantic_sha256": semantic_frame_sha256(
            target_rows, scoring_columns
        ),
    }


def _load_source_cache_artifacts(
    source: Mapping[str, Any],
    *,
    session_date: Any,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_records = _source_set_records(source)
    bindings = source["parsed_shards"]
    snapshot_binding = {
        field: source[field]
        for field in SOURCE_MANIFEST_FIELDS
        if field.startswith("model_price_snapshot_")
    }
    _validate_model_price_snapshot_binding(
        snapshot_binding,
        predictor_derived_store_root=predictor_derived_store_root,
        raw_records=raw_records,
        shard_bindings=bindings,
    )
    prices, _, _ = _load_model_price_prefix(
        raw_records,
        bindings,
        snapshot_target_month=_date(
            session_date, "source cache target"
        ).to_period("M"),
        latest_required_source_session=source["latest_required_source_session"],
        snapshot_manifest_object_key=source[
            "model_price_snapshot_manifest_object_key"
        ],
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
    )
    cache_manifest, manifest_payload = _read_external_canonical_json(
        predictor_derived_store_root,
        str(source["g0_panel_cache_manifest_object_key"]),
        prefix=G0_PANEL_CACHE_OBJECT_PREFIX,
        label="G0 cache manifest",
    )
    cache, cached_rows, expected_manifest_key, _ = _validate_g0_cache_manifest(
        cache_manifest,
        predictor_derived_store_root=predictor_derived_store_root,
        expected_target_session=session_date,
        expected_latest_source_session=source["latest_required_source_session"],
        expected_source_set_sha256=source["source_set_sha256"],
        expected_parsed_shard_set_sha256=source["parsed_shard_set_sha256"],
    )
    target = _date(session_date, "source cache target")
    panel = cached_rows.loc[
        pd.to_datetime(cached_rows["date"]).eq(target)
    ].reset_index(drop=True)
    cache_created = _timestamp(cache["created_at"], "G0 cache created_at")
    cache_sealed = _timestamp(cache["sealed_at"], "G0 cache sealed_at")
    source_created = _timestamp(source["created_at"], "source manifest created_at")
    if cache_sealed > source_created:
        raise V18Error("source manifest predates its G0 cache seal")
    for raw_record, binding in zip(raw_records, bindings, strict=True):
        shard_manifest, _ = _read_external_canonical_json(
            predictor_derived_store_root,
            str(binding["shard_manifest_object_key"]),
            prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
            label="predictor shard manifest",
        )
        if _timestamp(shard_manifest["sealed_at"], "parsed shard sealed_at") > (
            cache_created
        ):
            raise V18Error("G0 cache predates a parsed shard seal")
    exact = {
        "g0_panel_cache_object_key": cache["data_object_key"],
        "g0_panel_cache_byte_count": int(cache["data_byte_count"]),
        "g0_panel_cache_sha256": cache["data_sha256"],
        "g0_panel_cache_manifest_object_key": expected_manifest_key,
        "g0_panel_cache_manifest_byte_count": len(manifest_payload),
        "g0_panel_cache_manifest_file_sha256": hashlib.sha256(
            manifest_payload
        ).hexdigest(),
        "g0_panel_cache_manifest_sha256": cache["cache_manifest_sha256"],
    }
    for field, expected in exact.items():
        if source[field] != expected:
            raise V18Error(f"source cache exact binding changed: {field}")
    if int(cache["parsed_row_count"]) != int(source["parsed_row_count"]):
        raise V18Error("source/cache parsed row count differs")
    for field in (
        "target_date_scoring_input_semantic_sha256",
        "target_slice_semantic_sha256",
    ):
        if source[field] != cache[field]:
            raise V18Error(f"source/cache semantic differs: {field}")
    scoring_columns = (
        "date",
        "code",
        "name",
        "common_score_eligible",
        "feature_source_max_date",
        *G0_FEATURES,
    )
    observed = {
        "target_date_scoring_input_semantic_sha256": semantic_frame_sha256(
            panel, scoring_columns
        ),
        "target_slice_semantic_sha256": semantic_frame_sha256(
            panel, G0_PANEL_COLUMNS
        ),
    }
    for field, digest in observed.items():
        if source[field] != digest:
            raise V18Error(f"cache replay semantic differs: {field}")
    return prices, panel, cache


def replay_predictor_source_manifest(
    manifest: Mapping[str, Any],
    *,
    session_date: Any,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path | None = None,
    terminal_clean_room: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], str]:
    """Use bound caches preterminal; terminal mode computes from raw first."""

    value, digest = validate_source_manifest(
        manifest,
        session_date=session_date,
        predictor_raw_store_root=predictor_raw_store_root,
    )
    if not _strict_bool(value["source_complete"]):
        raise V18Error("cannot replay an incomplete predictor source manifest")
    if predictor_derived_store_root is not None and not terminal_clean_room:
        prices, panel, _ = _load_source_cache_artifacts(
            value,
            session_date=session_date,
            predictor_raw_store_root=predictor_raw_store_root,
            predictor_derived_store_root=predictor_derived_store_root,
        )
        validate_source_manifest(
            value,
            session_date=session_date,
            predictor_raw_store_root=predictor_raw_store_root,
            parsed_prices=prices,
            panel=panel,
        )
        return prices, panel, value, digest
    snapshot_objects = _source_set_records(value)
    with _external_snapshot_paths(
        predictor_raw_store_root,
        snapshot_objects,
        prefix=PREDICTOR_OBJECT_PREFIX,
        label="predictor",
    ) as (raw_paths, _):
        try:
            prices, report = _collect_jpx_registered(raw_paths)
        except Exception as exc:
            raise V18Error(f"predictor raw clean-room parse failed: {exc}") from exc
    inputs = report.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != len(raw_paths):
        raise V18Error("predictor parser report does not bind every raw object")
    if any(int(item.get("rejected_rows", -1)) != 0 for item in inputs):
        raise V18Error("predictor clean-room parse rejected a row")
    duplicate_count = int(prices[["date", "code"]].duplicated(keep=False).sum())
    if duplicate_count != 0:
        raise V18Error("predictor clean-room parse produced duplicate date/code rows")
    if int(value["parsed_row_count"]) != len(prices):
        raise V18Error("predictor clean-room parsed row count differs")
    if int(value["rejected_row_count"]) != 0 or int(
        value["duplicate_date_code_count"]
    ) != duplicate_count:
        raise V18Error("predictor clean-room parse counts differ")
    latest = pd.to_datetime(prices["date"], errors="coerce").max().normalize()
    if latest != _date(value["latest_required_source_session"], "latest source"):
        raise V18Error("predictor clean-room parse does not end at exact D-1")
    panel = build_forward_c00_panel(prices, session_date)
    target = _date(session_date, "terminal target session")
    direct_target_slice = _coerce_jsonl_frame(
        panel.loc[pd.to_datetime(panel["date"]).eq(target)],
        G0_PANEL_COLUMNS,
        label="terminal direct G0 target slice",
    )
    validate_source_manifest(
        value,
        session_date=session_date,
        predictor_raw_store_root=predictor_raw_store_root,
        parsed_prices=prices,
        panel=direct_target_slice,
    )
    if predictor_derived_store_root is not None:
        cached_model_prices, cached_panel, _ = _load_source_cache_artifacts(
            value,
            session_date=session_date,
            predictor_raw_store_root=predictor_raw_store_root,
            predictor_derived_store_root=predictor_derived_store_root,
        )
        direct_prices = _coerce_jsonl_frame(
            prices, PARSED_PRICE_COLUMNS, label="terminal direct predictor parse"
        )
        cached_prices = _load_bound_predictor_shards(
            _source_set_records(value),
            value["parsed_shards"],
            predictor_raw_store_root=predictor_raw_store_root,
            predictor_derived_store_root=predictor_derived_store_root,
        )
        if not direct_prices.equals(cached_prices):
            raise V18Error("terminal raw parse differs exactly from parsed shards")
        if not _coerce_model_price_frame(
            direct_prices, label="terminal direct compact model-price projection"
        ).equals(cached_model_prices):
            raise V18Error(
                "terminal raw parse differs from compact model-price prefix"
            )
        if not direct_target_slice.equals(cached_panel):
            raise V18Error("terminal raw G0 target slice differs exactly from cache")
    return prices, panel, value, digest


def _seal_predictor_object(
    source_pdf: str | Path,
    *,
    predictor_raw_store_root: str | Path,
    object_key: str,
) -> tuple[int, str]:
    return _seal_external_object(
        source_pdf,
        store_root=predictor_raw_store_root,
        object_key=object_key,
        prefix=PREDICTOR_OBJECT_PREFIX,
        label="predictor",
    )


def build_predictor_source_manifest(
    source_pdfs: Sequence[str | Path],
    source_file_names: Sequence[str],
    source_urls: Sequence[str],
    *,
    session_date: Any,
    source_received_at: Any | Sequence[Any],
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
    cache_chronology_class: str = "forward",
    predictor_cache_anchor: Mapping[str, Any] | None = None,
    month_source_manifest: Mapping[str, Any] | None = None,
    _month_source_factory: Any | None = None,
    model_price_snapshot_binding: Mapping[str, Any] | None = None,
    first_counted_session_value: Any | None = None,
    runtime_lock_verified_at: Any | None = None,
    created_at: Any | None = None,
    sealed_at: Any | None = None,
    output: str | Path | None = None,
    _resume_exact_timestamps: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Seal raw bytes, extend parsed shards, and build the C00 panel exactly once."""

    target = _date(session_date, "session_date")
    if _resume_exact_timestamps and output is not None:
        raise V18Error("source exact-timestamp resume cannot publish a new manifest")
    verified = _runtime_verified_timestamp(
        runtime_lock_verified_at,
        allow_historical_replay=_resume_exact_timestamps,
    )
    calendar = load_registered_calendar()
    positions = np.flatnonzero(calendar == target)
    if len(positions) != 1:
        raise V18Error("predictor manifest target is outside registered calendar")
    position = int(positions[0])
    latest = _latest_required_predictor_source_session(target, calendar)
    expected_files, kinds = _expected_predictor_files(latest)
    if not (len(source_pdfs) == len(source_file_names) == len(source_urls)):
        raise V18Error("new predictor source vectors are misaligned")
    names = [str(item) for item in source_file_names]
    if len(names) != len(set(names)) or any(name not in kinds for name in names):
        raise V18Error("new predictor inputs contain duplicates/unregistered files")
    if names != [name for name in expected_files if name in set(names)]:
        raise V18Error("new predictor inputs are not in canonical registry order")
    if isinstance(source_received_at, Sequence) and not isinstance(
        source_received_at, (str, bytes, bytearray)
    ):
        received_values = list(source_received_at)
        if len(received_values) != len(names):
            raise V18Error("new predictor receipt vector is misaligned")
    else:
        received_values = [source_received_at] * len(names)
    received_times = [
        _timestamp(item, f"source_received_at[{index}]")
        for index, item in enumerate(received_values)
    ]
    if not received_times:
        raise V18Error("canonical predictor preparation requires a post-anchor source")
    received: datetime | None = max(received_times)
    if received > _cutoff(target):
        raise V18Error("predictor source arrived after target cutoff")
    if _STRICT_RUNTIME_ACTIVE and received is not None and received > datetime.now(TOKYO):
        raise V18Error("predictor source receipt is in the future")
    _validate_external_store_disjointness(
        predictor_raw_store_root, predictor_derived_store_root
    )
    chronology_mode = str(cache_chronology_class)
    if chronology_mode not in {"anchor", "forward"}:
        raise V18Error("predictor cache chronology must be anchor or forward")
    anchor_latest: pd.Timestamp | None = None
    daily_snapshot_binding: dict[str, Any] | None = None
    month_source_hash: str | None = None
    predecessor_source: dict[str, Any] | None = None
    predecessor_target_session: str | None = None
    predecessor_manifest_sha256: str | None = None
    retained_raw_by_file: dict[str, dict[str, Any]] = {}
    retained_binding_by_file: dict[str, dict[str, Any]] = {}
    if chronology_mode == "anchor":
        if (
            predictor_cache_anchor is not None
            or month_source_manifest is not None
            or _month_source_factory is not None
            or model_price_snapshot_binding is not None
            or first_counted_session_value is not None
        ):
            raise V18Error("historical anchor cache cannot extend a forward chain")
    else:
        if (
            predictor_cache_anchor is None
            or first_counted_session_value is None
            or (month_source_manifest is None and _month_source_factory is None)
        ):
            raise V18Error(
                "forward predictor cache requires activated anchor/month context"
            )
        anchor_summary = validate_predictor_cache_anchor_summary(
            predictor_cache_anchor
        )
        (
            predecessor_target_session,
            predecessor_manifest_sha256,
        ) = _source_manifest_predecessor_binding(
            target,
            first_counted_session_value=first_counted_session_value,
            require_decision_binding=False,
        )
        anchor_manifest, anchor_payload = _read_external_canonical_json(
            predictor_derived_store_root,
            anchor_summary["snapshot_manifest_object_key"],
            prefix=CACHE_ANCHOR_OBJECT_PREFIX,
            label="forward predictor cache anchor manifest",
        )
        _, observed_anchor_summary, _ = validate_predictor_cache_anchor(
            anchor_manifest,
            predictor_raw_store_root=predictor_raw_store_root,
            predictor_derived_store_root=predictor_derived_store_root,
            require_direct_clean_room_reparse=False,
        )
        if observed_anchor_summary != anchor_summary or hashlib.sha256(
            anchor_payload
        ).hexdigest() != anchor_summary["snapshot_manifest_file_sha256"]:
            raise V18Error("forward predictor anchor differs from payload B")
        anchor_latest = _date(
            anchor_summary["latest_source_session"], "anchor latest source session"
        )
        if latest <= anchor_latest:
            raise V18Error(
                "first counted predecessor must strictly follow the activated anchor"
            )
        if month_source_manifest is None:
            target_month = target.to_period("M")
            first_month = _date(
                first_counted_session_value, "first counted session"
            ).to_period("M")
            if target_month == first_month:
                canonical_snapshot_binding = _model_price_snapshot_binding_from_anchor(
                    anchor_summary
                )
            else:
                previous_path = MONTH_SOURCE_MANIFEST_DIR / f"{target_month - 1}.json"
                if not previous_path.is_file():
                    raise V18Error("prepare-day compact snapshot predecessor is missing")
                canonical_snapshot_binding = _model_price_snapshot_binding_from_month_source(
                    read_json(previous_path)
                )
        else:
            month_source_value = dict(month_source_manifest)
            if set(month_source_value) != set(MONTH_SOURCE_MANIFEST_FIELDS):
                raise V18Error("daily source month-source schema changed")
            if month_source_value["target_month"] != str(target.to_period("M")):
                raise V18Error("daily source month-source targets another month")
            month_source_hash = canonical_json_sha256(
                month_source_value, exclude_fields={"month_source_manifest_sha256"}
            )
            if month_source_value["month_source_manifest_sha256"] != month_source_hash:
                raise V18Error("daily source month-source self-hash changed")
            canonical_snapshot_binding = _model_price_snapshot_binding_from_month_source(
                month_source_value
            )
        if model_price_snapshot_binding is not None and dict(
            model_price_snapshot_binding
        ) != canonical_snapshot_binding:
            raise V18Error("daily source caller swapped its monthly snapshot")
        daily_snapshot_binding = canonical_snapshot_binding
        snapshot_month = _month(
            daily_snapshot_binding["model_price_snapshot_target_month"],
            "daily model-price snapshot target month",
        )
        if month_source_manifest is not None and snapshot_month != target.to_period("M"):
            raise V18Error(
                "daily source requires its target-month model-price snapshot"
            )
        predecessor_source = _daily_source_predecessor(
            target,
            first_counted_session_value=first_counted_session_value,
        )
        for raw_record, binding in zip(
            anchor_manifest["raw_sources"],
            anchor_manifest["parsed_shards"],
            strict=True,
        ):
            retained_raw_by_file[str(raw_record["file"])] = dict(raw_record)
            retained_binding_by_file[str(raw_record["file"])] = dict(binding)
        first = _date(first_counted_session_value, "first counted session")
        first_position = int(np.flatnonzero(calendar == first)[0])
        for prior in calendar[first_position:position]:
            prior_path = SOURCE_MANIFEST_DIR / f"{pd.Timestamp(prior).date()}.json"
            if not prior_path.is_file():
                raise V18Error("incremental predictor chain lacks a prior source manifest")
            prior_source, _ = validate_source_manifest(
                read_json(prior_path),
                session_date=prior,
                first_counted_session_value=first_counted_session_value,
                require_predecessor_decision=False,
            )
            for raw_record, binding in zip(
                _source_set_records(prior_source),
                prior_source["parsed_shards"],
                strict=True,
            ):
                name = str(raw_record["file"])
                if name in retained_raw_by_file and (
                    retained_raw_by_file[name] != raw_record
                    or retained_binding_by_file[name] != dict(binding)
                ):
                    raise V18Error("incremental predictor retained prefix conflicts")
                retained_raw_by_file[name] = dict(raw_record)
                retained_binding_by_file[name] = dict(binding)
        if any(name in retained_raw_by_file for name in names):
            raise V18Error("incremental predictor input repeats a retained source")
    if received is None:
        raise V18Error("predictor source lacks a causal receipt timestamp")
    object_keys: list[str] = []
    byte_counts: list[int] = []
    digests: list[str] = []
    official_urls: list[str] = []
    historical_metadata = _historical_predictor_metadata()
    new_received_by_file = dict(zip(names, received_times, strict=True))
    # Validate the entire cumulative batch before the first irreversible
    # create-once write.  A late bad historical object must not strand a
    # partially sealed prefix.
    for index, (source_pdf, file_name, source_url) in enumerate(
        zip(source_pdfs, names, source_urls, strict=True)
    ):
        if file_name.startswith("stq_"):
            official = _official_jpx_daily_url(
                source_url,
                f"source_urls[{index}]",
                file_name=file_name,
                source_session=_source_file_date(file_name, kinds[file_name]),
            )
        else:
            official = _official_jpx_url(source_url, f"source_urls[{index}]")
        supplied_count, supplied_hash = _source_file_metadata(
            source_pdf, label="predictor"
        )
        bound = historical_metadata.get(file_name)
        if bound is not None:
            if supplied_hash != str(bound["sha256"]):
                raise V18Error(f"historical predictor bytes changed before seal: {file_name}")
            if "bytes" in bound and supplied_count != int(bound["bytes"]):
                raise V18Error(f"historical predictor byte count changed before seal: {file_name}")
            if bound.get("source_url") is not None and official != str(
                bound["source_url"]
            ):
                raise V18Error(f"historical predictor URL changed before seal: {file_name}")
        official_urls.append(official)
    for source_pdf, file_name in zip(source_pdfs, names, strict=True):
        key = _predictor_object_key(file_name, kinds[file_name])
        retained_count, retained_hash = _seal_predictor_object(
            source_pdf,
            predictor_raw_store_root=predictor_raw_store_root,
            object_key=key,
        )
        if Path(key).name != file_name:
            raise V18Error("predictor retained object filename changed")
        object_keys.append(key)
        byte_counts.append(retained_count)
        digests.append(retained_hash)
    new_source_records = [
        {
            "object_key": key,
            "file": name,
            "url": url,
            "byte_count": count,
            "sha256": digest,
        }
        for key, name, url, count, digest in zip(
            object_keys, names, official_urls, byte_counts, digests, strict=True
        )
    ]
    new_by_file = {str(item["file"]): item for item in new_source_records}
    source_set = [
        dict(retained_raw_by_file.get(name, new_by_file.get(name)))
        for name in expected_files
        if name in retained_raw_by_file or name in new_by_file
    ]
    if [str(item["file"]) for item in source_set] != expected_files:
        missing = [name for name in expected_files if name not in {
            str(item["file"]) for item in source_set
        }]
        raise V18Error(f"complete predictor preparation still lacks sources: {missing}")
    if chronology_mode == "forward":
        assert predictor_cache_anchor is not None
        if predecessor_source is not None:
            predecessor_raw = _source_set_records(predecessor_source)
            predecessor_shards = predecessor_source["parsed_shards"]
        else:
            predecessor_raw = anchor_manifest["raw_sources"]
            predecessor_shards = anchor_manifest["parsed_shards"]
        if source_set[: len(predecessor_raw)] != predecessor_raw:
            raise V18Error("forward predictor raw set forks/revises its predecessor")
    shard_bindings: list[dict[str, Any]] = []
    for raw_record in source_set:
        _, _, shard_manifest_key = _predictor_shard_identity(raw_record)
        if _external_object_exists(
            predictor_derived_store_root,
            shard_manifest_key,
            prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
            label="predictor shard manifest",
        ):
            existing, _ = _read_external_canonical_json(
                predictor_derived_store_root,
                shard_manifest_key,
                prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
                label="predictor shard manifest",
            )
            shard_value, _, binding = _validate_parsed_shard_manifest(
                existing,
                raw_record=raw_record,
                predictor_derived_store_root=predictor_derived_store_root,
                expected_raw_received_at=new_received_by_file.get(
                    str(raw_record["file"])
                ),
                decode_data=False,
            )
        else:
            source_date = _source_file_date(
                str(raw_record["file"]), kinds[str(raw_record["file"])]
            )
            shard_chronology = (
                "anchor"
                if chronology_mode == "anchor"
                or (anchor_latest is not None and source_date <= anchor_latest)
                else "forward"
            )
            if chronology_mode == "forward" and shard_chronology == "anchor":
                raise V18Error(
                    "activated historical shard is missing; post-B repair is forbidden"
                )
            raw_received = new_received_by_file.get(str(raw_record["file"]))
            if chronology_mode == "forward" and raw_received is None:
                raise V18Error("retained forward source is missing its sealed shard")
            shard_value, _, binding = ensure_predictor_parsed_shard(
                raw_record,
                predictor_raw_store_root=predictor_raw_store_root,
                predictor_derived_store_root=predictor_derived_store_root,
                chronology_class=shard_chronology,
                raw_received_at=None if shard_chronology == "anchor" else raw_received,
                runtime_lock_verified_at=verified,
            )
        if chronology_mode == "forward" and anchor_latest is not None:
            source_date = _source_file_date(
                str(raw_record["file"]), kinds[str(raw_record["file"])]
            )
            expected_chronology = "anchor" if source_date <= anchor_latest else "forward"
            if shard_value["chronology_class"] != expected_chronology:
                raise V18Error("parsed shard chronology crosses activated anchor boundary")
        shard_bindings.append(binding)
    if chronology_mode == "forward" and shard_bindings[
        : len(predecessor_shards)
    ] != predecessor_shards:
        raise V18Error("forward parsed-shard set forks/reorders its predecessor")
    if daily_snapshot_binding is None:
        raise V18Error("predictor source lacks a compact model-price snapshot")
    _validate_model_price_snapshot_binding(
        daily_snapshot_binding,
        predictor_derived_store_root=predictor_derived_store_root,
        raw_records=source_set,
        shard_bindings=shard_bindings,
    )
    source_set_hash = canonical_json_sha256(source_set)
    shard_set_hash = _parsed_shard_set_sha256(shard_bindings)
    prices, _, _ = _load_model_price_prefix(
        source_set,
        shard_bindings,
        snapshot_target_month=daily_snapshot_binding[
            "model_price_snapshot_target_month"
        ],
        latest_required_source_session=latest,
        snapshot_manifest_object_key=daily_snapshot_binding[
            "model_price_snapshot_manifest_object_key"
        ],
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
    )
    prepared_full_panel = None
    rejected = 0
    duplicates = int(prices[["date", "code"]].duplicated(keep=False).sum())
    if duplicates != 0:
        raise V18Error("predictor parsed-shard merge produced duplicates")
    parsed_latest = pd.to_datetime(prices["date"], errors="coerce").max().normalize()
    if parsed_latest != latest:
        raise V18Error("predictor parsed data does not end at exact D-1")
    if _month_source_factory is not None:
        prepared_full_panel = build_forward_c00_panel(prices, target)
        produced = _month_source_factory(
            source_set,
            shard_bindings,
            prices,
            prepared_full_panel,
            daily_snapshot_binding,
        )
        if (
            not isinstance(produced, tuple)
            or len(produced) != 2
            or not isinstance(produced[0], Mapping)
            or not isinstance(produced[1], Mapping)
        ):
            raise V18Error("prepare-day month factory returned an invalid binding")
        month_source_value = dict(produced[0])
        daily_snapshot_binding = dict(produced[1])
        month_source_hash = canonical_json_sha256(
            month_source_value, exclude_fields={"month_source_manifest_sha256"}
        )
        if (
            set(month_source_value) != set(MONTH_SOURCE_MANIFEST_FIELDS)
            or month_source_value["month_source_manifest_sha256"]
            != month_source_hash
            or month_source_value["target_month"] != str(target.to_period("M"))
        ):
            raise V18Error("prepare-day month factory produced invalid authority")
        if month_source_manifest is not None and month_source_value != dict(
            month_source_manifest
        ):
            raise V18Error("prepare-day month factory changed existing authority")
        if daily_snapshot_binding != _model_price_snapshot_binding_from_month_source(
            month_source_value
        ):
            raise V18Error("prepare-day month factory swapped its snapshot")
        _validate_model_price_snapshot_binding(
            daily_snapshot_binding,
            predictor_derived_store_root=predictor_derived_store_root,
            raw_records=source_set,
            shard_bindings=shard_bindings,
        )
    if month_source_hash is None:
        raise V18Error("complete daily source lacks a sealed monthly authority")
    total_parsed_rows = int(
        sum(int(binding["parsed_row_count"]) for binding in shard_bindings)
    )
    cache_value, panel, cache_manifest_key, cache_manifest_payload = (
        materialize_g0_panel_cache(
            prices,
            predictor_derived_store_root=predictor_derived_store_root,
            target_session=target,
            latest_required_source_session=latest,
            source_set_sha256=source_set_hash,
            parsed_shard_set_sha256=shard_set_hash,
            total_parsed_row_count=total_parsed_rows,
            prepared_full_panel=prepared_full_panel,
            runtime_lock_verified_at=verified,
        )
    )
    semantics = {
        field: cache_value[field]
        for field in (
            "target_date_scoring_input_semantic_sha256",
            "target_slice_semantic_sha256",
        )
    }
    created = (
        _timestamp(created_at, "source manifest created_at")
        if _resume_exact_timestamps
        else _operation_timestamp(created_at, "source manifest created_at")
    )
    sealed = (
        _timestamp(sealed_at, "source manifest sealed_at")
        if _resume_exact_timestamps
        else _operation_timestamp(sealed_at, "source manifest sealed_at")
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "target_session": str(target.date()),
        "latest_required_source_session": str(latest.date()),
        "previous_counted_target_session": predecessor_target_session,
        "previous_counted_source_manifest_sha256": predecessor_manifest_sha256,
        "created_at": created,
        "sealed_at": sealed,
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "runtime_lock_verified_at": verified,
        "source_files": [item["file"] for item in source_set],
        "source_urls": [item["url"] for item in source_set],
        "source_object_keys": [item["object_key"] for item in source_set],
        "source_byte_counts": [item["byte_count"] for item in source_set],
        "source_sha256": [item["sha256"] for item in source_set],
        "source_set_sha256": source_set_hash,
        "parsed_shards": shard_bindings,
        "parsed_shard_set_sha256": shard_set_hash,
        "month_source_manifest_sha256": month_source_hash,
        **daily_snapshot_binding,
        "source_received_at": received,
        "parser_path": JPX_PARSER_PATH,
        "parser_version": JPX_PARSER_VERSION,
        "parser_sha256": JPX_PARSER_SHA256,
        "parsed_row_count": total_parsed_rows,
        "rejected_row_count": rejected,
        "duplicate_date_code_count": duplicates,
        **semantics,
        "g0_panel_cache_object_key": cache_value["data_object_key"],
        "g0_panel_cache_byte_count": int(cache_value["data_byte_count"]),
        "g0_panel_cache_sha256": cache_value["data_sha256"],
        "g0_panel_cache_manifest_object_key": cache_manifest_key,
        "g0_panel_cache_manifest_byte_count": len(cache_manifest_payload),
        "g0_panel_cache_manifest_file_sha256": hashlib.sha256(
            cache_manifest_payload
        ).hexdigest(),
        "g0_panel_cache_manifest_sha256": cache_value["cache_manifest_sha256"],
        "source_complete": True,
        "failure_reason": None,
        "python_version": platform.python_version(),
        "canonical_json_contract": CANONICAL_JSON_CONTRACT,
    }
    value["source_manifest_sha256"] = canonical_json_sha256(
        value, exclude_fields={"source_manifest_sha256"}
    )
    validate_source_manifest(
        value,
        session_date=target,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
        parsed_prices=prices,
        panel=panel,
        first_counted_session_value=(
            None if chronology_mode == "anchor" else first_counted_session_value
        ),
        require_predecessor_decision=False,
    )
    if output is not None:
        write_json(value, output, exclusive=True)
    return value, panel


def _validate_month_source_manifest_impl(
    manifest: Mapping[str, Any],
    *,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
    training_panel: pd.DataFrame | None = None,
    model_prices: pd.DataFrame | None = None,
    precomputed_model_semantic_sha256: str | None = None,
    precomputed_training_semantic_sha256: str | None = None,
    reuse_sealed_snapshot_semantic: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, str]:
    """Validate one independent M-1 source and compact snapshot extension."""

    if reuse_sealed_snapshot_semantic and any(
        item is not None
        for item in (
            training_panel,
            model_prices,
            precomputed_model_semantic_sha256,
            precomputed_training_semantic_sha256,
        )
    ):
        raise V18Error(
            "retained month-source semantic reuse cannot accept caller frames/hashes"
        )
    value = dict(manifest)
    if set(value) != set(MONTH_SOURCE_MANIFEST_FIELDS):
        raise V18Error("month-source manifest fields differ from A2 contract")
    if value["schema_version"] != 1:
        raise V18Error("month-source manifest schema changed")
    month = _month(value["target_month"], "month-source target_month")
    first_counted = _date(
        value["first_counted_session"], "month-source first counted session"
    )
    seal_session = _date(value["seal_session"], "month-source seal session")
    if seal_session != _month_seal_session(
        month, first_counted_session_value=first_counted
    ):
        raise V18Error("month-source seal session differs from activation calendar")
    latest = _latest_registered_source_before_month(month)
    if _date(value["latest_required_source_session"], "month-source latest") != latest:
        raise V18Error("month-source latest session is not exact M-1 session")
    created = _timestamp(value["created_at"], "month-source created_at")
    sealed = _timestamp(value["sealed_at"], "month-source sealed_at")
    verified = _timestamp(
        value["runtime_lock_verified_at"], "month-source runtime verified"
    )
    received = _timestamp(value["source_received_at"], "month-source receipt")
    activation_observed = _timestamp(
        value["activation_observed_at"], "month-source activation observed"
    )
    if max(verified, received, activation_observed) > created or created > sealed:
        raise V18Error("month-source timestamp DAG is invalid")
    if sealed > _cutoff(seal_session):
        raise V18Error("month-source was not sealed before its monthly cutoff")
    fixed = {
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "parser_sha256": JPX_PARSER_SHA256,
        "canonical_json_contract": CANONICAL_JSON_CONTRACT,
    }
    for field, expected in fixed.items():
        if value[field] != expected:
            raise V18Error(f"month-source binding changed: {field}")
    _require_sha(value["activation_payload_sha256"], "month-source payload SHA")
    _require_sha(value["activation_receipt_sha256"], "month-source receipt SHA")
    vectors = (
        value["source_files"],
        value["source_urls"],
        value["source_object_keys"],
        value["source_byte_counts"],
        value["source_sha256"],
    )
    if not all(isinstance(item, list) for item in vectors) or len(
        {len(item) for item in vectors}
    ) != 1:
        raise V18Error("month-source vectors are not aligned arrays")
    expected_files, kinds = _expected_predictor_files(latest)
    if [str(item) for item in value["source_files"]] != expected_files:
        raise V18Error("month-source does not cover exact registered prefix")
    raw_records = _source_set_records(value)
    if value["source_set_sha256"] != canonical_json_sha256(raw_records):
        raise V18Error("month-source raw set hash changed")
    bindings = value["parsed_shards"]
    if not isinstance(bindings, list) or len(bindings) != len(raw_records):
        raise V18Error("month-source raw/shard arrays are misaligned")
    if value["parsed_shard_set_sha256"] != _parsed_shard_set_sha256(bindings):
        raise V18Error("month-source shard set hash changed")
    for index, (raw_record, kind) in enumerate(
        zip(raw_records, (kinds[name] for name in expected_files), strict=True)
    ):
        if raw_record["object_key"] != _predictor_object_key(
            str(raw_record["file"]), kind
        ):
            raise V18Error(f"month-source raw key changed at index {index}")
    shard_manifests = _validate_bound_predictor_shard_metadata(
        raw_records,
        bindings,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
    )
    snapshot_binding = _model_price_snapshot_binding_from_month_source(value)
    snapshot, snapshot_prices, _ = _validate_model_price_snapshot_binding(
        snapshot_binding,
        predictor_derived_store_root=predictor_derived_store_root,
        raw_records=raw_records,
        shard_bindings=bindings,
        _reuse_sealed_semantic=reuse_sealed_snapshot_semantic,
    )
    if snapshot["target_month"] != str(month) or snapshot[
        "latest_source_session"
    ] != str(latest.date()):
        raise V18Error("month-source compact snapshot does not end at exact M-1")
    prices = (
        snapshot_prices
        if model_prices is None
        else _coerce_model_price_frame(
            model_prices, label="month-source supplied model-price prefix"
        )
    )
    if not prices.equals(snapshot_prices):
        raise V18Error("month-source supplied prefix differs from compact snapshot")
    if pd.to_datetime(prices["date"]).max().normalize() != latest:
        raise V18Error("month-source model-price prefix does not end at exact M-1")
    origin = str(value["model_price_snapshot_origin"])
    if origin not in {"activation_anchor", "forward_extension"}:
        raise V18Error("month-source compact snapshot origin changed")
    previous_fields = (
        "previous_model_price_snapshot_target_month",
        "previous_model_price_snapshot_latest_source_session",
        "previous_model_price_snapshot_manifest_object_key",
        "previous_model_price_snapshot_manifest_file_sha256",
        "previous_model_price_snapshot_manifest_sha256",
    )
    suffix_count = int(value["model_price_suffix_shard_count"])
    suffix = bindings[len(bindings) - suffix_count :] if suffix_count else []
    if value["model_price_suffix_shard_set_sha256"] != canonical_json_sha256(
        [dict(item) for item in suffix]
    ):
        raise V18Error("month-source compact suffix set hash changed")
    if origin == "activation_anchor":
        if any(value[field] is not None for field in previous_fields) or suffix_count != 0:
            raise V18Error("initial month snapshot must be the activated anchor")
        if month != first_counted.to_period("M"):
            raise V18Error("only the first counted month may use the activated snapshot")
        if snapshot["previous_snapshot_manifest_sha256"] is not None:
            raise V18Error("activated compact snapshot unexpectedly has a predecessor")
        expected_received = max(
            _timestamp(snapshot["sealed_at"], "activated snapshot sealed"),
            activation_observed,
        )
        if received != expected_received:
            raise V18Error(
                "initial month-source availability is not derived from anchor/C"
            )
    else:
        if any(value[field] is None for field in previous_fields) or suffix_count <= 0:
            raise V18Error("forward month snapshot lacks predecessor/suffix")
        previous_manifest, previous_payload = _read_external_canonical_json(
            predictor_derived_store_root,
            value["previous_model_price_snapshot_manifest_object_key"],
            prefix=MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX,
            label="previous model-price snapshot manifest",
        )
        previous_count = int(previous_manifest.get("raw_source_count", 0))
        if previous_count != len(raw_records) - suffix_count:
            raise V18Error("month-source compact suffix is not an exact append")
        previous, _, previous_key, _ = _validate_model_price_snapshot_manifest(
            previous_manifest,
            predictor_derived_store_root=predictor_derived_store_root,
            expected_target_month=value[
                "previous_model_price_snapshot_target_month"
            ],
            expected_latest_source_session=value[
                "previous_model_price_snapshot_latest_source_session"
            ],
            expected_raw_source_set_sha256=canonical_json_sha256(
                raw_records[:previous_count]
            ),
            expected_parsed_shard_set_sha256=_parsed_shard_set_sha256(
                bindings[:previous_count]
            ),
            _reuse_sealed_semantic=reuse_sealed_snapshot_semantic,
        )
        if (
            previous_key
            != value["previous_model_price_snapshot_manifest_object_key"]
            or hashlib.sha256(previous_payload).hexdigest()
            != value["previous_model_price_snapshot_manifest_file_sha256"]
            or previous["snapshot_manifest_sha256"]
            != value["previous_model_price_snapshot_manifest_sha256"]
            or snapshot["previous_snapshot_manifest_sha256"]
            != previous["snapshot_manifest_sha256"]
        ):
            raise V18Error("month-source compact predecessor binding changed")
        previous_month = _month(
            value["previous_model_price_snapshot_target_month"],
            "month-source previous snapshot month",
        )
        previous_latest = _date(
            value["previous_model_price_snapshot_latest_source_session"],
            "month-source previous snapshot latest",
        )
        if (
            previous_month != month - 1
            or previous_latest != _latest_registered_source_before_month(previous_month)
            or suffix_count != len(raw_records) - previous_count
        ):
            raise V18Error("month-source predecessor/suffix is not immediate")
        previous_frame = _validate_model_price_snapshot_manifest(
            previous_manifest,
            predictor_derived_store_root=predictor_derived_store_root,
            expected_target_month=previous_month,
            expected_latest_source_session=previous_latest,
            expected_raw_source_set_sha256=canonical_json_sha256(
                raw_records[:previous_count]
            ),
            expected_parsed_shard_set_sha256=_parsed_shard_set_sha256(
                bindings[:previous_count]
            ),
            _reuse_sealed_semantic=reuse_sealed_snapshot_semantic,
        )[1]
        suffix_frames: list[pd.DataFrame] = []
        snapshot_created = _timestamp(
            snapshot["created_at"], "month-source snapshot created"
        )
        if _timestamp(
            previous["sealed_at"], "month-source previous snapshot sealed"
        ) > snapshot_created:
            raise V18Error("month-source snapshot predates its predecessor seal")
        for index in range(previous_count, len(raw_records)):
            shard_manifest = shard_manifests[index]
            if (
                shard_manifest["chronology_class"] != "forward"
                or _date(shard_manifest["min_date"], "month suffix min")
                <= previous_latest
                or _date(shard_manifest["max_date"], "month suffix max") > latest
                or _timestamp(shard_manifest["sealed_at"], "month suffix sealed")
                > snapshot_created
            ):
                raise V18Error("month-source suffix chronology/date DAG changed")
            _, shard_frame, _ = _validate_parsed_shard_manifest(
                shard_manifest,
                raw_record=raw_records[index],
                predictor_derived_store_root=predictor_derived_store_root,
                expected_chronology_class="forward",
                decode_data=True,
            )
            assert shard_frame is not None
            suffix_frames.append(
                _coerce_model_price_frame(
                    shard_frame, label="validated month-source suffix"
                )
            )
        reconstructed = _coerce_model_price_frame(
            pd.concat([previous_frame, *suffix_frames], ignore_index=True),
            label="validated month-source predecessor plus suffix",
        )
        if not reconstructed.equals(snapshot_prices):
            raise V18Error(
                "month-source snapshot differs from predecessor plus exact suffix"
            )
        suffix_receipts = [
            _timestamp(item["raw_received_at"], "month-source suffix receipt")
            for item in shard_manifests[-suffix_count:]
        ]
        if not suffix_receipts or received != max(suffix_receipts):
            raise V18Error(
                "forward month-source availability is not exact suffix receipt max"
            )
    if _timestamp(snapshot["sealed_at"], "month-source snapshot sealed") > created:
        raise V18Error("month-source manifest predates its snapshot seal")
    if reuse_sealed_snapshot_semantic:
        retained_model_semantic = _require_sha(
            snapshot["model_price_semantic_sha256"],
            "retained month-source model-price semantic SHA",
        )
    else:
        retained_model_semantic = (
            model_price_semantic_sha256(prices)
            if precomputed_model_semantic_sha256 is None
            else _require_sha(
                precomputed_model_semantic_sha256,
                "precomputed month-source model-price semantic SHA",
            )
        )
    exact = {
        "parsed_row_count": int(len(prices)),
        "model_price_full_prefix_semantic_sha256": retained_model_semantic,
    }
    for field, expected in exact.items():
        if value[field] != expected:
            raise V18Error(f"month-source cache binding changed: {field}")
    _require_sha(
        value["g0_training_panel_semantic_sha256"],
        "month-source G0 training semantic SHA",
    )
    if isinstance(value["g0_training_row_count"], bool) or int(
        value["g0_training_row_count"]
    ) <= 0:
        raise V18Error("month-source G0 training row count must be positive")
    if training_panel is not None:
        canonical_panel = _coerce_jsonl_frame(
            training_panel, G0_PANEL_COLUMNS, label="month-source training panel"
        )
        dates = pd.to_datetime(canonical_panel["date"])
        if dates.ge(month.start_time.normalize()).any() or dates.max() > latest:
            raise V18Error("month-source training panel contains target-month rows")
        if len(canonical_panel) != int(value["g0_training_row_count"]):
            raise V18Error("month-source G0 training row count changed")
        observed_training_semantic = (
            semantic_frame_sha256(canonical_panel, G0_PANEL_COLUMNS)
            if precomputed_training_semantic_sha256 is None
            else _require_sha(
                precomputed_training_semantic_sha256,
                "precomputed month-source training semantic SHA",
            )
        )
        if observed_training_semantic != value[
            "g0_training_panel_semantic_sha256"
        ]:
            raise V18Error("month-source G0 training semantic changed")
    expected_hash = canonical_json_sha256(
        value, exclude_fields={"month_source_manifest_sha256"}
    )
    if value["month_source_manifest_sha256"] != expected_hash:
        raise V18Error("month-source self-hash mismatch")
    return value, (
        training_panel if training_panel is not None else pd.DataFrame()
    ), expected_hash


def validate_month_source_manifest(
    manifest: Mapping[str, Any],
    *,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
    training_panel: pd.DataFrame | None = None,
    model_prices: pd.DataFrame | None = None,
    precomputed_model_semantic_sha256: str | None = None,
    precomputed_training_semantic_sha256: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, str]:
    """Public/full validator; every supplied semantic is freshly proven."""

    return _validate_month_source_manifest_impl(
        manifest,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
        training_panel=training_panel,
        model_prices=model_prices,
        precomputed_model_semantic_sha256=precomputed_model_semantic_sha256,
        precomputed_training_semantic_sha256=(
            precomputed_training_semantic_sha256
        ),
        reuse_sealed_snapshot_semantic=False,
    )


def _validate_retained_intramonth_month_source(
    manifest: Mapping[str, Any],
    *,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame, str]:
    """Validate canonical retained context without a redundant semantic rehash.

    The private call has no frame/hash input.  It still pins every raw/shard,
    snapshot/predecessor manifest, exact data byte SHA, canonical CSV decode,
    timestamp DAG, and self-hash.  Only the semantic digest already sealed by
    those exact bytes at the month boundary is reused; terminal validation uses
    the public/full path and recomputes it independently.
    """

    if not _STRICT_RUNTIME_ACTIVE:
        raise V18Error("retained month-source reuse requires strict runtime")
    return _validate_month_source_manifest_impl(
        manifest,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
        reuse_sealed_snapshot_semantic=True,
    )


def build_month_source_manifest(
    source_pdfs: Sequence[str | Path],
    source_file_names: Sequence[str],
    source_urls: Sequence[str],
    *,
    target_month: Any,
    source_received_at: Any,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
    activation_payload_sha256: str,
    activation_receipt_sha256: str,
    anchor_latest_source_session: Any,
    previous_model_price_snapshot_binding: Mapping[str, Any],
    previous_prefix_raw_records: Sequence[Mapping[str, Any]],
    previous_prefix_shard_bindings: Sequence[Mapping[str, Any]],
    first_counted_session_value: Any,
    activation_observed_at: Any,
    runtime_lock_verified_at: Any | None = None,
    output: str | Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Extend one compact M-1 prefix and build the monthly panel once.

    The predecessor snapshot is the activated snapshot for the first counted
    month and the immediately preceding month-source snapshot thereafter.
    Only the registered suffix is decoded.  The resulting full-prefix panel is
    constructed once and may be handed to the canonical daily orchestrator.
    """

    month = _month(target_month, "month-source target_month")
    latest = _latest_registered_source_before_month(month)
    first_counted = _date(
        first_counted_session_value, "month-source first counted session"
    )
    seal_session = _month_seal_session(
        month, first_counted_session_value=first_counted
    )
    activation_observed = _timestamp(
        activation_observed_at, "month-source activation observed"
    )
    expected_files, kinds = _expected_predictor_files(latest)
    names = [str(item) for item in source_file_names]
    if len(source_pdfs) != len(names) or len(source_urls) != len(names):
        raise V18Error("new month-source input vectors are misaligned")
    base_raw = [dict(item) for item in previous_prefix_raw_records]
    base_bindings = [dict(item) for item in previous_prefix_shard_bindings]
    if len(base_raw) != len(base_bindings):
        raise V18Error("month-source predecessor raw/shard vectors are misaligned")
    base_files = [str(item.get("file")) for item in base_raw]
    if base_files != expected_files[: len(base_files)]:
        raise V18Error("month-source predecessor is not an exact registry prefix")
    missing_files = expected_files[len(base_files) :]
    if names != missing_files:
        raise V18Error("new month-source inputs are not the exact registered suffix")
    _validate_external_store_disjointness(
        predictor_raw_store_root, predictor_derived_store_root
    )
    verified = _runtime_verified_timestamp(runtime_lock_verified_at)
    received = _timestamp(source_received_at, "month-source source_received_at")
    monthly_cutoff = _cutoff(seal_session)
    if received > monthly_cutoff or (
        _STRICT_RUNTIME_ACTIVE and received > datetime.now(TOKYO)
    ):
        raise V18Error("month-source receipt is late or in the future")
    anchor_latest = _date(
        anchor_latest_source_session, "month-source activated anchor latest"
    )
    raw_records: list[dict[str, Any]] = list(base_raw)
    historical_metadata = _historical_predictor_metadata()
    for index, (source_pdf, name, source_url) in enumerate(
        zip(source_pdfs, names, source_urls, strict=True)
    ):
        if kinds[name] == "daily":
            official = _official_jpx_daily_url(
                source_url,
                f"month-source URLs[{index}]",
                file_name=name,
                source_session=_source_file_date(name, "daily"),
            )
        else:
            official = _official_jpx_url(
                source_url, f"month-source URLs[{index}]"
            )
        supplied_count, supplied_hash = _source_file_metadata(
            source_pdf, label="month-source predictor"
        )
        bound = historical_metadata.get(name)
        if bound is not None and (
            supplied_hash != str(bound["sha256"])
            or ("bytes" in bound and supplied_count != int(bound["bytes"]))
            or (
                bound.get("source_url") is not None
                and official != str(bound["source_url"])
            )
        ):
            raise V18Error(f"month-source historical binding changed: {name}")
        key = _predictor_object_key(name, kinds[name])
        count, digest = _seal_predictor_object(
            source_pdf,
            predictor_raw_store_root=predictor_raw_store_root,
            object_key=key,
        )
        raw_records.append(
            {
                "object_key": key,
                "file": name,
                "url": official,
                "byte_count": count,
                "sha256": digest,
            }
        )
    bindings: list[dict[str, Any]] = list(base_bindings)
    for raw_record in raw_records[len(base_raw) :]:
        source_date = _source_file_date(
            str(raw_record["file"]), kinds[str(raw_record["file"])]
        )
        expected_chronology = "anchor" if source_date <= anchor_latest else "forward"
        _, _, shard_manifest_key = _predictor_shard_identity(raw_record)
        if _external_object_exists(
            predictor_derived_store_root,
            shard_manifest_key,
            prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
            label="month-source shard manifest",
        ):
            shard_manifest, _ = _read_external_canonical_json(
                predictor_derived_store_root,
                shard_manifest_key,
                prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
                label="month-source shard manifest",
            )
            shard, _, binding = _validate_parsed_shard_manifest(
                shard_manifest,
                raw_record=raw_record,
                predictor_derived_store_root=predictor_derived_store_root,
            )
            if shard["chronology_class"] != expected_chronology:
                raise V18Error("month-source shard chronology crosses anchor")
        else:
            if expected_chronology == "anchor":
                raise V18Error("month-source activated anchor shard is missing")
            _, _, binding = ensure_predictor_parsed_shard(
                raw_record,
                predictor_raw_store_root=predictor_raw_store_root,
                predictor_derived_store_root=predictor_derived_store_root,
                chronology_class="forward",
                raw_received_at=received,
                runtime_lock_verified_at=verified,
            )
        bindings.append(binding)
    shard_manifests = _validate_bound_predictor_shard_metadata(
        raw_records,
        bindings,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
    )
    previous_binding = dict(previous_model_price_snapshot_binding)
    previous_snapshot, previous_prices, previous_manifest_payload = (
        _validate_model_price_snapshot_binding(
            previous_binding,
            predictor_derived_store_root=predictor_derived_store_root,
            raw_records=raw_records,
            shard_bindings=bindings,
        )
    )
    previous_month = _month(
        previous_binding["model_price_snapshot_target_month"],
        "month-source predecessor snapshot month",
    )
    previous_latest = _date(
        previous_binding["model_price_snapshot_latest_source_session"],
        "month-source predecessor snapshot latest",
    )
    prefix_count = int(previous_snapshot["raw_source_count"])
    if prefix_count > len(raw_records):
        raise V18Error("month-source predecessor exceeds the registered prefix")
    initial_snapshot = previous_month == month
    if initial_snapshot:
        if previous_latest != latest or prefix_count != len(raw_records):
            raise V18Error("initial month must consume the exact activated M-1 snapshot")
        suffix_bindings: list[dict[str, Any]] = []
        prices = previous_prices
    else:
        if previous_month != month - 1 or previous_latest >= latest:
            raise V18Error("month-source predecessor is not the immediately prior month")
        suffix_frames: list[pd.DataFrame] = []
        for index in range(prefix_count, len(raw_records)):
            shard_manifest = shard_manifests[index]
            if _date(
                shard_manifest["min_date"], "month-source suffix min date"
            ) <= previous_latest:
                raise V18Error("month-source suffix overlaps its immutable predecessor")
            _, shard_frame, _ = _validate_parsed_shard_manifest(
                shard_manifest,
                raw_record=raw_records[index],
                predictor_derived_store_root=predictor_derived_store_root,
                expected_chronology_class="forward",
                decode_data=True,
            )
            assert shard_frame is not None
            suffix_frames.append(
                _coerce_model_price_frame(
                    shard_frame, label="month-source compact forward suffix"
                )
            )
        if not suffix_frames:
            raise V18Error("forward month-source extension has an empty suffix")
        suffix_bindings = [dict(item) for item in bindings[prefix_count:]]
        prices = _coerce_model_price_frame(
            pd.concat([previous_prices, *suffix_frames], ignore_index=True),
            label="month-source compact full prefix",
        )
    if pd.to_datetime(prices["date"]).max().normalize() != latest or pd.to_datetime(
        prices["date"]
    ).ge(month.start_time.normalize()).any():
        raise V18Error("month-source compact data is not the exact M-1 prefix")
    source_set_hash = canonical_json_sha256(raw_records)
    shard_set_hash = _parsed_shard_set_sha256(bindings)
    # Panel construction is intentionally before the create-once snapshot
    # write.  A model/codec failure cannot strand a newly reserved artifact.
    full_panel = build_forward_c00_panel(prices, seal_session)
    panel = _coerce_jsonl_frame(
        full_panel.loc[
            pd.to_datetime(full_panel["date"], errors="coerce").lt(
                month.start_time.normalize()
            )
        ],
        G0_PANEL_COLUMNS,
        label="month-source training panel",
    )
    if initial_snapshot:
        current_snapshot = previous_snapshot
        current_binding = previous_binding
        snapshot_origin = "activation_anchor"
    else:
        (
            current_snapshot,
            _,
            current_manifest_key,
            current_manifest_payload,
        ) = materialize_model_price_snapshot(
            prices,
            predictor_derived_store_root=predictor_derived_store_root,
            target_month=month,
            latest_source_session=latest,
            raw_records=raw_records,
            shard_bindings=bindings,
            previous_snapshot_manifest_sha256=previous_snapshot[
                "snapshot_manifest_sha256"
            ],
            previous_snapshot_binding=previous_binding,
            runtime_lock_verified_at=verified,
        )
        current_binding = {
            "model_price_snapshot_target_month": str(month),
            "model_price_snapshot_latest_source_session": str(latest.date()),
            "model_price_snapshot_object_key": current_snapshot["data_object_key"],
            "model_price_snapshot_byte_count": int(
                current_snapshot["data_byte_count"]
            ),
            "model_price_snapshot_file_sha256": current_snapshot["data_sha256"],
            "model_price_snapshot_semantic_sha256": current_snapshot[
                "model_price_semantic_sha256"
            ],
            "model_price_snapshot_manifest_object_key": current_manifest_key,
            "model_price_snapshot_manifest_byte_count": len(
                current_manifest_payload
            ),
            "model_price_snapshot_manifest_file_sha256": hashlib.sha256(
                current_manifest_payload
            ).hexdigest(),
            "model_price_snapshot_manifest_sha256": current_snapshot[
                "snapshot_manifest_sha256"
            ],
        }
        snapshot_origin = "forward_extension"
    if initial_snapshot:
        expected_received = max(
            _timestamp(
                current_snapshot["sealed_at"],
                "activated month snapshot sealed_at",
            ),
            activation_observed,
        )
    else:
        suffix_receipts = [
            _timestamp(
                item["raw_received_at"], "forward month-source suffix receipt"
            )
            for item in shard_manifests[prefix_count:]
        ]
        if not suffix_receipts:
            raise V18Error("forward month-source has no receipt-bearing suffix")
        expected_received = max(suffix_receipts)
    if received != expected_received:
        raise V18Error(
            "month-source availability must be derived from its exact anchor/suffix"
        )
    created = datetime.now(TOKYO)
    sealed = datetime.now(TOKYO)
    if max(verified, received, activation_observed) > created or sealed > monthly_cutoff:
        raise V18Error("month-source construction missed its causal cutoff")
    for shard_manifest in shard_manifests:
        if _timestamp(shard_manifest["sealed_at"], "month-source shard sealed") > created:
            raise V18Error("month-source predates a contributing shard seal")
    if _timestamp(
        current_snapshot["sealed_at"], "month-source snapshot sealed"
    ) > created:
        raise V18Error("month-source predates its compact snapshot seal")
    model_semantic_hash = model_price_semantic_sha256(prices)
    training_semantic_hash = semantic_frame_sha256(panel, G0_PANEL_COLUMNS)
    value: dict[str, Any] = {
        "schema_version": 1,
        "target_month": str(month),
        "first_counted_session": str(first_counted.date()),
        "seal_session": str(seal_session.date()),
        "activation_observed_at": activation_observed,
        "latest_required_source_session": str(latest.date()),
        "created_at": created,
        "sealed_at": sealed,
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "runtime_lock_verified_at": verified,
        "source_files": [item["file"] for item in raw_records],
        "source_urls": [item["url"] for item in raw_records],
        "source_object_keys": [item["object_key"] for item in raw_records],
        "source_byte_counts": [item["byte_count"] for item in raw_records],
        "source_sha256": [item["sha256"] for item in raw_records],
        "source_set_sha256": source_set_hash,
        "source_received_at": received,
        "parsed_shards": bindings,
        "parsed_shard_set_sha256": shard_set_hash,
        "model_price_snapshot_origin": snapshot_origin,
        "previous_model_price_snapshot_target_month": (
            None if initial_snapshot else str(previous_month)
        ),
        "previous_model_price_snapshot_latest_source_session": (
            None if initial_snapshot else str(previous_latest.date())
        ),
        "previous_model_price_snapshot_manifest_object_key": (
            None
            if initial_snapshot
            else previous_binding["model_price_snapshot_manifest_object_key"]
        ),
        "previous_model_price_snapshot_manifest_file_sha256": (
            None
            if initial_snapshot
            else hashlib.sha256(previous_manifest_payload).hexdigest()
        ),
        "previous_model_price_snapshot_manifest_sha256": (
            None
            if initial_snapshot
            else previous_snapshot["snapshot_manifest_sha256"]
        ),
        "model_price_suffix_shard_count": len(suffix_bindings),
        "model_price_suffix_shard_set_sha256": canonical_json_sha256(
            suffix_bindings
        ),
        **current_binding,
        "parsed_row_count": int(len(prices)),
        "model_price_full_prefix_semantic_sha256": model_semantic_hash,
        "g0_training_panel_semantic_sha256": training_semantic_hash,
        "g0_training_row_count": int(len(panel)),
        "activation_payload_sha256": _require_sha(
            activation_payload_sha256, "month-source payload SHA"
        ),
        "activation_receipt_sha256": _require_sha(
            activation_receipt_sha256, "month-source receipt SHA"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "parser_sha256": JPX_PARSER_SHA256,
        "canonical_json_contract": CANONICAL_JSON_CONTRACT,
    }
    value["month_source_manifest_sha256"] = canonical_json_sha256(
        value, exclude_fields={"month_source_manifest_sha256"}
    )
    if output is not None:
        write_json(value, output, exclusive=True)
    return value, panel


def _build_month_source_from_prepared_day(
    raw_records: Sequence[Mapping[str, Any]],
    shard_bindings: Sequence[Mapping[str, Any]],
    model_prices: pd.DataFrame,
    full_panel: pd.DataFrame,
    base_snapshot_binding: Mapping[str, Any],
    *,
    target_month: pd.Period,
    first_counted_session_value: Any,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
    activation_payload_sha256: str,
    activation_receipt_sha256: str,
    activation_observed_at: Any,
    output: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Seal monthly authority from the canonical day's already-built panel."""

    month = _month(target_month, "prepared month-source target")
    first = _date(first_counted_session_value, "prepared first counted session")
    seal_session = _month_seal_session(month, first_counted_session_value=first)
    latest = _latest_registered_source_before_month(month)
    expected_files, _ = _expected_predictor_files(latest)
    all_raw = [dict(item) for item in raw_records]
    all_bindings = [dict(item) for item in shard_bindings]
    prefix_count = len(expected_files)
    month_raw = all_raw[:prefix_count]
    month_bindings = all_bindings[:prefix_count]
    if [str(item["file"]) for item in month_raw] != expected_files:
        raise V18Error("prepared month-source is not the exact M-1 raw prefix")
    month_prices = _coerce_model_price_frame(
        model_prices.loc[pd.to_datetime(model_prices["date"]).le(latest)],
        label="prepared month-source compact M-1 prefix",
    )
    if pd.to_datetime(month_prices["date"]).max().normalize() != latest:
        raise V18Error("prepared month-source compact prefix does not end at M-1")
    base_binding = dict(base_snapshot_binding)
    base_snapshot, _, base_payload = _validate_model_price_snapshot_binding(
        base_binding,
        predictor_derived_store_root=predictor_derived_store_root,
        raw_records=month_raw,
        shard_bindings=month_bindings,
    )
    base_month = _month(
        base_binding["model_price_snapshot_target_month"],
        "prepared base snapshot month",
    )
    base_count = int(base_snapshot["raw_source_count"])
    initial = base_month == month
    if initial:
        if base_count != prefix_count or base_snapshot["latest_source_session"] != str(
            latest.date()
        ):
            raise V18Error("prepared initial month snapshot is not exact")
        snapshot = base_snapshot
        current_binding = base_binding
        snapshot_origin = "activation_anchor"
        suffix_bindings: list[dict[str, Any]] = []
    else:
        if base_month != month - 1 or base_count >= prefix_count:
            raise V18Error("prepared month snapshot predecessor is not immediate")
        suffix_bindings = month_bindings[base_count:]
        snapshot, _, manifest_key, manifest_payload = materialize_model_price_snapshot(
            month_prices,
            predictor_derived_store_root=predictor_derived_store_root,
            target_month=month,
            latest_source_session=latest,
            raw_records=month_raw,
            shard_bindings=month_bindings,
            previous_snapshot_manifest_sha256=base_snapshot[
                "snapshot_manifest_sha256"
            ],
            previous_snapshot_binding=base_binding,
        )
        current_binding = {
            "model_price_snapshot_target_month": str(month),
            "model_price_snapshot_latest_source_session": str(latest.date()),
            "model_price_snapshot_object_key": snapshot["data_object_key"],
            "model_price_snapshot_byte_count": int(snapshot["data_byte_count"]),
            "model_price_snapshot_file_sha256": snapshot["data_sha256"],
            "model_price_snapshot_semantic_sha256": snapshot[
                "model_price_semantic_sha256"
            ],
            "model_price_snapshot_manifest_object_key": manifest_key,
            "model_price_snapshot_manifest_byte_count": len(manifest_payload),
            "model_price_snapshot_manifest_file_sha256": hashlib.sha256(
                manifest_payload
            ).hexdigest(),
            "model_price_snapshot_manifest_sha256": snapshot[
                "snapshot_manifest_sha256"
            ],
        }
        snapshot_origin = "forward_extension"
    training_panel = _coerce_jsonl_frame(
        full_panel.loc[
            pd.to_datetime(full_panel["date"], errors="coerce").lt(
                month.start_time.normalize()
            )
        ],
        G0_PANEL_COLUMNS,
        label="prepared month-source training panel",
    )
    observed = _timestamp(activation_observed_at, "prepared activation observed")
    if initial:
        received = max(
            _timestamp(snapshot["sealed_at"], "prepared anchor snapshot sealed"),
            observed,
        )
    else:
        shard_manifests = _validate_bound_predictor_shard_metadata(
            month_raw,
            month_bindings,
            predictor_raw_store_root=predictor_raw_store_root,
            predictor_derived_store_root=predictor_derived_store_root,
        )
        suffix_receipts = [
            _timestamp(
                item["raw_received_at"], "prepared month-source suffix receipt"
            )
            for item in shard_manifests[base_count:]
        ]
        if len(suffix_receipts) != len(suffix_bindings) or not suffix_receipts:
            raise V18Error("prepared month-source suffix receipt set is incomplete")
        received = max(suffix_receipts)
    verified = _runtime_verified_timestamp()
    created = datetime.now(TOKYO)
    sealed = datetime.now(TOKYO)
    if max(received, observed, verified) > created or sealed > _cutoff(seal_session):
        raise V18Error("prepared month-source missed its causal cutoff")
    source_set_hash = canonical_json_sha256(month_raw)
    shard_set_hash = _parsed_shard_set_sha256(month_bindings)
    value: dict[str, Any] = {
        "schema_version": 1,
        "target_month": str(month),
        "first_counted_session": str(first.date()),
        "seal_session": str(seal_session.date()),
        "activation_observed_at": observed,
        "latest_required_source_session": str(latest.date()),
        "created_at": created,
        "sealed_at": sealed,
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "runtime_lock_verified_at": verified,
        "source_files": [item["file"] for item in month_raw],
        "source_urls": [item["url"] for item in month_raw],
        "source_object_keys": [item["object_key"] for item in month_raw],
        "source_byte_counts": [item["byte_count"] for item in month_raw],
        "source_sha256": [item["sha256"] for item in month_raw],
        "source_set_sha256": source_set_hash,
        "source_received_at": received,
        "parsed_shards": month_bindings,
        "parsed_shard_set_sha256": shard_set_hash,
        "model_price_snapshot_origin": snapshot_origin,
        "previous_model_price_snapshot_target_month": (
            None if initial else base_snapshot["target_month"]
        ),
        "previous_model_price_snapshot_latest_source_session": (
            None if initial else base_snapshot["latest_source_session"]
        ),
        "previous_model_price_snapshot_manifest_object_key": (
            None
            if initial
            else base_binding["model_price_snapshot_manifest_object_key"]
        ),
        "previous_model_price_snapshot_manifest_file_sha256": (
            None if initial else hashlib.sha256(base_payload).hexdigest()
        ),
        "previous_model_price_snapshot_manifest_sha256": (
            None if initial else base_snapshot["snapshot_manifest_sha256"]
        ),
        "model_price_suffix_shard_count": len(suffix_bindings),
        "model_price_suffix_shard_set_sha256": canonical_json_sha256(
            suffix_bindings
        ),
        **current_binding,
        "parsed_row_count": int(len(month_prices)),
        "model_price_full_prefix_semantic_sha256": model_price_semantic_sha256(
            month_prices
        ),
        "g0_training_panel_semantic_sha256": semantic_frame_sha256(
            training_panel, G0_PANEL_COLUMNS
        ),
        "g0_training_row_count": int(len(training_panel)),
        "activation_payload_sha256": _require_sha(
            activation_payload_sha256, "prepared month payload SHA"
        ),
        "activation_receipt_sha256": _require_sha(
            activation_receipt_sha256, "prepared month receipt SHA"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "parser_sha256": JPX_PARSER_SHA256,
        "canonical_json_contract": CANONICAL_JSON_CONTRACT,
    }
    value["month_source_manifest_sha256"] = canonical_json_sha256(
        value, exclude_fields={"month_source_manifest_sha256"}
    )
    validate_month_source_manifest(
        value,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
        training_panel=training_panel,
        model_prices=month_prices,
        precomputed_model_semantic_sha256=value[
            "model_price_full_prefix_semantic_sha256"
        ],
        precomputed_training_semantic_sha256=value[
            "g0_training_panel_semantic_sha256"
        ],
    )
    write_json(value, output, exclusive=True)
    return value, training_panel, current_binding


def prepare_day(
    *,
    session_date: Any,
    new_predictor_pdfs: Sequence[str | Path],
    new_predictor_file_names: Sequence[str],
    new_predictor_urls: Sequence[str],
    new_predictor_received_at: Sequence[Any],
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
    outcome_raw_store_root: str | Path,
    checkpoint_core_store_root: str | Path,
    activation_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonical one-process prepare path; publication remains a separate command."""

    if not _STRICT_RUNTIME_ACTIVE:
        raise V18Error("prepare-day requires the strict operational runtime")
    target = _date(session_date, "prepare-day session")
    context = validate_activation_context(activation_context)
    payload, payload_hash = validate_activation_payload(
        ACTIVATION_PAYLOAD,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
    )
    if context["activation_payload_sha256"] != payload_hash:
        raise V18Error("prepare-day activation context differs from payload B")
    _validate_external_store_disjointness(
        predictor_raw_store_root, predictor_derived_store_root
    )
    _validate_external_root_pair_disjoint(
        predictor_raw_store_root,
        "predictor",
        outcome_raw_store_root,
        "outcome",
    )
    _validate_external_root_pair_disjoint(
        predictor_derived_store_root,
        "predictor derived",
        outcome_raw_store_root,
        "outcome",
    )
    month = target.to_period("M")
    month_source_path = MONTH_SOURCE_MANIFEST_DIR / f"{month}.json"
    month_source: dict[str, Any] | None = (
        read_json(month_source_path) if month_source_path.is_file() else None
    )
    fold_holder: dict[str, Any] = {}
    bundle_holder: dict[str, Any] = {}

    receipts = [
        _timestamp(item, f"prepare-day predictor receipt[{index}]")
        for index, item in enumerate(new_predictor_received_at)
    ]
    if len(receipts) != len(new_predictor_file_names):
        raise V18Error("prepare-day predictor receipt vector is misaligned")
    if not receipts:
        raise V18Error("prepare-day requires at least one post-anchor source")

    def month_factory(
        raw_records: Sequence[Mapping[str, Any]],
        shard_bindings: Sequence[Mapping[str, Any]],
        model_prices: pd.DataFrame,
        full_panel: pd.DataFrame,
        base_snapshot_binding: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal month_source
        if month_source is None:
            if target != _month_seal_session(
                month,
                first_counted_session_value=context["first_counted_session"],
            ):
                raise V18Error(
                    "monthly authority cannot first appear after its seal session"
                )
            month_source, training_panel, current_binding = (
                _build_month_source_from_prepared_day(
                    raw_records,
                    shard_bindings,
                    model_prices,
                    full_panel,
                    base_snapshot_binding,
                    target_month=month,
                    first_counted_session_value=context["first_counted_session"],
                    predictor_raw_store_root=predictor_raw_store_root,
                    predictor_derived_store_root=predictor_derived_store_root,
                    activation_payload_sha256=context["activation_payload_sha256"],
                    activation_receipt_sha256=context["activation_receipt_sha256"],
                    activation_observed_at=context[
                        "activation_receipt_commit_observed_at"
                    ],
                    output=month_source_path,
                )
            )
        else:
            month_latest = _latest_registered_source_before_month(month)
            month_prices = _coerce_model_price_frame(
                model_prices.loc[
                    pd.to_datetime(model_prices["date"]).le(month_latest)
                ],
                label="prepare-day recovered month model-price prefix",
            )
            training_panel = _coerce_jsonl_frame(
                full_panel.loc[
                    pd.to_datetime(full_panel["date"]).lt(month.start_time)
                ],
                G0_PANEL_COLUMNS,
                label="prepare-day recovered month training panel",
            )
            validated_month_source, _, _ = (
                _validate_retained_intramonth_month_source(
                    month_source,
                    predictor_raw_store_root=predictor_raw_store_root,
                    predictor_derived_store_root=predictor_derived_store_root,
                )
            )
            if validated_month_source != month_source:
                raise V18Error("prepare-day retained month-source bytes changed")
            if _model_price_snapshot_binding_from_month_source(
                validated_month_source
            ) != dict(base_snapshot_binding):
                raise V18Error(
                    "prepare-day retained month-source snapshot binding changed"
                )
            if (
                len(month_prices)
                != int(validated_month_source["parsed_row_count"])
                or len(training_panel)
                != int(validated_month_source["g0_training_row_count"])
            ):
                raise V18Error("prepare-day retained month-source row counts changed")
            # Boundary creation sealed the exact compact snapshot and full G0
            # training semantic.  Intramonth validation above reopens that
            # content-addressed snapshot, checks its exact bytes/canonical
            # decode/source+shard/self bindings, and the fold loader below
            # freshly recomputes row/target/feature-matrix identities.  Rehashing
            # the unchanged 22-column training prefix here is redundant and was
            # the measured daily timing blocker.
            current_binding = _model_price_snapshot_binding_from_month_source(
                validated_month_source
            )
        fold, bundle = _load_or_create_month_fold(
            training_panel,
            target_month=month,
            month_source_manifest=month_source,
            predictor_derived_store_root=predictor_derived_store_root,
            first_counted_session_value=context["first_counted_session"],
            activation_observed_at=context[
                "activation_receipt_commit_observed_at"
            ],
        )
        fold_holder.update(fold)
        bundle_holder.update(bundle)
        return month_source, current_binding

    source_path = SOURCE_MANIFEST_DIR / f"{target.date()}.json"
    retained_source: dict[str, Any] | None = None
    if os.path.lexists(source_path):
        if source_path.is_symlink() or not source_path.is_file():
            raise V18Error("prepare-day retained source is not a plain file")
        retained_source = read_json(source_path)
        validate_source_manifest(
            retained_source,
            session_date=target,
            predictor_raw_store_root=predictor_raw_store_root,
            predictor_derived_store_root=predictor_derived_store_root,
            first_counted_session_value=context["first_counted_session"],
            require_predecessor_decision=False,
        )
    source, target_panel = build_predictor_source_manifest(
        new_predictor_pdfs,
        new_predictor_file_names,
        new_predictor_urls,
        session_date=target,
        source_received_at=new_predictor_received_at,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
        predictor_cache_anchor=payload["predictor_cache_anchor"],
        month_source_manifest=month_source,
        _month_source_factory=month_factory,
        first_counted_session_value=context["first_counted_session"],
        runtime_lock_verified_at=(
            None
            if retained_source is None
            else retained_source["runtime_lock_verified_at"]
        ),
        created_at=(None if retained_source is None else retained_source["created_at"]),
        sealed_at=(None if retained_source is None else retained_source["sealed_at"]),
        output=(source_path if retained_source is None else None),
        _resume_exact_timestamps=retained_source is not None,
    )
    if retained_source is not None:
        if canonical_json_bytes(source) != canonical_json_bytes(retained_source):
            raise V18Error(
                "prepare-day completed source retry changes exact caller/artifact identity"
            )
        source = retained_source

    # The canonical rollover is intentionally outcome-blind until the current
    # target source/cache above is sealed.  Only then may the exact D-1 PDF be
    # copied to a physically distinct outcome authority, attached to the
    # already published D-1 decision, and (at a boundary) close the prior
    # month.  Failure at any stage aborts; it cannot create a counted cash day.
    decision_rows = _load_decision_record_authority(heal_derived=True)
    outcome_rows = _load_outcome_record_authority(
        decision_rows, heal_derived=True
    )
    first_counted = _date(
        context["first_counted_session"], "prepare-day first counted session"
    )
    calendar = load_registered_calendar()
    target_positions = np.flatnonzero(calendar == target)
    first_positions = np.flatnonzero(calendar == first_counted)
    if (
        len(target_positions) != 1
        or len(first_positions) != 1
        or target < first_counted
    ):
        raise V18Error("prepare-day target is outside its counted calendar")
    prior_counted = calendar[int(first_positions[0]) : int(target_positions[0])]
    if [str(item.date()) for item in prior_counted] != [
        str(item["session_date"]) for item in decision_rows
    ]:
        raise V18Error("prepare-day prior decision prefix is incomplete")
    if target == first_counted:
        if (
            source["previous_counted_target_session"] is not None
            or source["previous_counted_source_manifest_sha256"] is not None
        ):
            raise V18Error("first prepare-day source has a predecessor binding")
    else:
        if not decision_rows or (
            source["previous_counted_target_session"]
            != decision_rows[-1]["session_date"]
            or source["previous_counted_source_manifest_sha256"]
            != decision_rows[-1]["source_manifest_sha256"]
        ):
            raise V18Error(
                "prepare-day source predecessor differs from the sealed prior decision"
            )
    if target != first_counted:
        previous_target = pd.Timestamp(calendar[int(target_positions[0]) - 1])
        if not decision_rows or decision_rows[-1]["session_date"] != str(
            previous_target.date()
        ):
            raise V18Error("prepare-day lacks the exact prior counted decision")
        latest = _latest_required_predictor_source_session(target, calendar)
        if latest != previous_target:
            raise V18Error("prepare-day D-1 predictor/outcome session diverged")
        expected_files, kinds = _expected_predictor_files(latest)
        rollover_matches: list[tuple[str | Path, str, str, datetime]] = []
        for pdf, file_name, url, receipt in zip(
            new_predictor_pdfs,
            new_predictor_file_names,
            new_predictor_urls,
            receipts,
            strict=True,
        ):
            if file_name not in kinds:
                raise V18Error("prepare-day rollover source is unregistered")
            if _source_file_date(file_name, kinds[file_name]) == previous_target:
                rollover_matches.append((pdf, file_name, url, receipt))
        if len(rollover_matches) != 1:
            raise V18Error("prepare-day requires exactly one supplied D-1 outcome PDF")
        outcome_pdf, outcome_file, outcome_url, outcome_received = rollover_matches[0]
        if outcome_file not in expected_files:
            raise V18Error("prepare-day D-1 outcome file is outside the source registry")
        predictor_record = next(
            (
                item
                for item in _source_set_records(source)
                if item["file"] == outcome_file
            ),
            None,
        )
        if predictor_record is None:
            raise V18Error("prepare-day D-1 predictor raw binding is missing")
        canonical_outcome_url = _official_jpx_daily_url(
            outcome_url,
            "prepare-day D-1 outcome URL",
            file_name=outcome_file,
            source_session=previous_target,
        )
        supplied_count, supplied_sha = _source_file_metadata(
            outcome_pdf, label="prepare-day D-1 supplied input"
        )
        if (
            predictor_record["file"] != outcome_file
            or predictor_record["url"] != canonical_outcome_url
            or int(predictor_record["byte_count"]) != supplied_count
            or predictor_record["sha256"] != supplied_sha
        ):
            raise V18Error(
                "prepare-day D-1 supplied input differs from sealed predictor bytes"
            )
        predictor_index = source["source_files"].index(outcome_file)
        predictor_binding = source["parsed_shards"][predictor_index]
        predictor_shard_manifest, _ = _read_external_canonical_json(
            predictor_derived_store_root,
            predictor_binding["shard_manifest_object_key"],
            prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
            label="prepare-day D-1 predictor shard manifest",
        )
        _validate_parsed_shard_manifest(
            predictor_shard_manifest,
            raw_record=predictor_record,
            predictor_derived_store_root=predictor_derived_store_root,
            expected_chronology_class="forward",
            expected_raw_received_at=outcome_received,
            decode_data=False,
        )
        outcome_path = OUTCOME_MANIFEST_DIR / f"{previous_target.date()}.json"
        if os.path.lexists(outcome_path):
            if outcome_path.is_symlink() or not outcome_path.is_file():
                raise V18Error("prepare-day retained outcome manifest is not plain")
            outcome_manifest, _ = validate_outcome_manifest(
                outcome_path,
                decision_rows[-1],
                outcome_raw_store_root=outcome_raw_store_root,
            )
        else:
            # Copy only from the already sealed predictor object.  A caller
            # mutation after the precheck therefore cannot strand a different
            # outcome authority before the cross-role check.
            with _external_snapshot_paths(
                predictor_raw_store_root,
                [predictor_record],
                prefix=PREDICTOR_OBJECT_PREFIX,
                label="prepare-day D-1 predictor rollover",
            ) as (rollover_paths, _):
                outcome_manifest = build_outcome_manifest(
                    decision_rows[-1],
                    rollover_paths[0],
                    outcome_raw_store_root=outcome_raw_store_root,
                    source_file_name=outcome_file,
                    source_url=canonical_outcome_url,
                    source_received_at=outcome_received,
                    output=outcome_path,
                )
        expected_cross_role = {
            "file": outcome_manifest["source_file_name"],
            "url": outcome_manifest["source_url"],
            "byte_count": int(outcome_manifest["source_byte_count"]),
            "sha256": outcome_manifest["source_sha256"],
        }
        if any(
            predictor_record[field] != expected
            for field, expected in expected_cross_role.items()
        ):
            raise V18Error(
                "prepare-day predictor/outcome roles do not bind identical D-1 bytes"
            )
        if _timestamp(
            outcome_manifest["source_received_at"], "D-1 outcome received_at"
        ) != outcome_received:
            raise V18Error("prepare-day D-1 outcome receipt changed")
        _assert_external_objects_nonalias(
            predictor_raw_store_root,
            str(predictor_record["object_key"]),
            PREDICTOR_OBJECT_PREFIX,
            outcome_raw_store_root,
            str(outcome_manifest["raw_source_object_key"]),
            OUTCOME_OBJECT_PREFIX,
        )
        if len(outcome_rows) == len(decision_rows) - 1:
            combined_outcomes = attach_outcomes(
                decision_rows,
                outcome_path,
                outcome_raw_store_root=outcome_raw_store_root,
                existing_outcomes=outcome_rows,
            )
            append_jsonl_record(
                OUTCOME_LEDGER,
                combined_outcomes[-1],
                required_fields=OUTCOME_FIELDS,
                validator=lambda rows: validate_outcome_records(
                    rows, decision_rows
                ),
            )
            outcome_rows = _load_outcome_record_authority(
                decision_rows, heal_derived=True
            )
        elif len(outcome_rows) == len(decision_rows):
            if outcome_rows[-1]["outcome_manifest_sha256"] != outcome_manifest[
                "outcome_manifest_sha256"
            ]:
                raise V18Error("prepare-day retained outcome ledger binding changed")
        else:
            raise V18Error("prepare-day outcome ledger is not the exact prior prefix")
    elif outcome_rows:
        raise V18Error("first counted prepare-day cannot prebind an outcome")

    completed_rows = _load_completed_month_record_authority(heal_derived=True)
    if target != first_counted:
        previous_month = previous_target.to_period("M")
        if previous_month != target.to_period("M"):
            if completed_rows and completed_rows[-1]["completed_month"] == str(
                previous_month
            ):
                rebuilt = build_completed_month_record(
                    decision_rows,
                    outcome_rows,
                    completed_month=str(previous_month),
                    created_at=completed_rows[-1]["created_at"],
                    existing_records=completed_rows[:-1],
                )
                if canonical_json_bytes(rebuilt[-1]) != canonical_json_bytes(
                    completed_rows[-1]
                ):
                    raise V18Error("prepare-day retained month close changed")
            else:
                combined_months = build_completed_month_record(
                    decision_rows,
                    outcome_rows,
                    completed_month=str(previous_month),
                    capture_operational_timestamp=True,
                    existing_records=completed_rows,
                )
                append_jsonl_record(
                    COMPLETED_MONTH_LEDGER,
                    combined_months[-1],
                    required_fields=COMPLETED_MONTH_FIELDS,
                    validator=validate_completed_month_records,
                )
                completed_rows = _load_completed_month_record_authority(
                    heal_derived=True
                )
    expected_closed_months = [
        str(item)
        for item in pd.period_range(
            first_counted.to_period("M"),
            target.to_period("M") - 1,
            freq="M",
        )
    ]
    if [item["completed_month"] for item in completed_rows] != expected_closed_months:
        raise V18Error("prepare-day prior completed-month prefix is incomplete")

    fold = fold_holder or read_json(FOLD_MANIFEST_DIR / f"{month}.json")
    bundle = bundle_holder or read_json(FOLD_MODEL_DIR / f"{month}.json")
    state_path = STATE_MANIFEST_DIR / f"{month}.json"
    retained_state: dict[str, Any] | None = None
    if os.path.lexists(state_path):
        if state_path.is_symlink() or not state_path.is_file():
            raise V18Error("prepare-day retained state is not a plain file")
        retained_state = read_json(state_path)
        validate_state_manifest(retained_state)
    # Outcome history is opened only after the target cache and source
    # manifest above have been fully sealed and independently validated.
    history = pair_history_from_ledgers(decision_rows, outcome_rows)
    state_value = derive_month_state(history, month)
    state_value_artifact = build_state_manifest(
        state_value,
        created_at=(
            datetime.now(TOKYO)
            if retained_state is None
            else retained_state["created_at"]
        ),
        c00_fold_manifest_sha256=fold["fold_manifest_sha256"],
        fold_model_bundle_file_sha256=fold[
            "fold_model_bundle_file_sha256"
        ],
        activation_payload_sha256=context["activation_payload_sha256"],
        activation_receipt_sha256=context["activation_receipt_sha256"],
        first_counted_session_value=context["first_counted_session"],
        activation_observed_at=context[
            "activation_receipt_commit_observed_at"
        ],
    )
    if retained_state is None:
        _write_json_once_exact(
            state_value_artifact,
            state_path,
            label="canonical monthly state manifest",
        )
    elif canonical_json_bytes(state_value_artifact) != canonical_json_bytes(
        retained_state
    ):
        raise V18Error("prepare-day retained state differs from exact history replay")
    state = read_json(state_path)
    validate_state_manifest(state)
    expected_state = {
        "target_month": str(month),
        "activation_payload_sha256": context["activation_payload_sha256"],
        "activation_receipt_sha256": context["activation_receipt_sha256"],
        "c00_fold_manifest_sha256": fold["fold_manifest_sha256"],
        "fold_model_bundle_file_sha256": fold[
            "fold_model_bundle_file_sha256"
        ],
    }
    if any(state[field] != expected for field, expected in expected_state.items()):
        raise V18Error("prepare-day state does not bind current month authority")
    score_shard_path = SCORE_SESSION_DIR / f"{target.date()}.csv"
    retained_pair: pd.DataFrame | None = None
    if os.path.lexists(score_shard_path):
        if score_shard_path.is_symlink() or not score_shard_path.is_file():
            raise V18Error("prepare-day retained score shard is not a plain file")
        retained_pair = validate_score_rows(
            _read_csv_plain(
                score_shard_path,
                label="canonical score-session shard",
                dtype={"code": "string"},
                float_precision="round_trip",
            )
        )
        if retained_pair["session_date"].iloc[0] != str(target.date()):
            raise V18Error("prepare-day retained score shard targets another session")
    if retained_pair is None:
        scores, _, _ = freeze_c00_top2(
            target_panel,
            target,
            capture_operational_timestamp=True,
            source_manifest_sha256=source["source_manifest_sha256"],
            first_counted_session_value=context["first_counted_session"],
            activation_observed_at=context["activation_receipt_commit_observed_at"],
            return_bundle=True,
            model_bundle=bundle,
            fold_manifest=fold,
        )
        append_score_rows(scores, SCORE_OUTPUT)
    else:
        scores, _, _ = freeze_c00_top2(
            target_panel,
            target,
            score_generated_at=retained_pair["score_generated_at"].iloc[0],
            runtime_lock_verified_at=retained_pair[
                "runtime_lock_verified_at"
            ].iloc[0],
            terminal_replay=True,
            source_manifest_sha256=source["source_manifest_sha256"],
            first_counted_session_value=context["first_counted_session"],
            activation_observed_at=context["activation_receipt_commit_observed_at"],
            return_bundle=True,
            model_bundle=bundle,
            fold_manifest=fold,
        )
        if not validate_score_rows(scores).equals(retained_pair):
            raise V18Error("prepare-day retained score pair differs from exact replay")
        # Rebuild/heal the cumulative CSV view from immutable session shards.
        append_score_rows(retained_pair, SCORE_OUTPUT)
    checkpoint = prepare_checkpoint(
        session_date=target,
        checkpoint_core_store_root=checkpoint_core_store_root,
    )
    return {
        "target_session": str(target.date()),
        "month_source_manifest_sha256": source["month_source_manifest_sha256"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "target_slice_semantic_sha256": source["target_slice_semantic_sha256"],
        "fold_manifest_sha256": fold["fold_manifest_sha256"],
        "state_manifest_sha256": state["state_manifest_sha256"],
        "score_semantic_sha256": semantic_score_hash(scores),
        "checkpoint_batch_id": checkpoint["checkpoint_batch_id"],
        "production_model_changed": False,
        "orders_allowed": False,
    }


def finalize_terminal(
    *,
    source_pdf: str | Path,
    source_file_name: str,
    source_url: str,
    source_received_at: Any,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
    outcome_raw_store_root: str | Path,
    checkpoint_core_store_root: str | Path,
    activation_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach the last outcome and close the terminal month in one operation.

    This is the only operational terminal-outcome mutation.  It is deliberately
    separate from ``evaluate`` so no performance gate can be opened before the
    complete terminal outcome/month authority has been sealed.
    """

    if not _STRICT_RUNTIME_ACTIVE:
        raise V18Error("finalize-terminal requires the strict operational runtime")
    context = validate_activation_context(activation_context)
    decisions = _load_decision_record_authority(heal_derived=True)
    if not decisions:
        raise V18Error("finalize-terminal requires a nonempty decision ledger")
    first = _date(context["first_counted_session"], "first counted session")
    if decisions[0]["session_date"] != str(first.date()):
        raise V18Error("finalize-terminal decision prefix starts on another session")
    calendar = load_registered_calendar()
    terminal = deterministic_terminal_session(first, calendar)
    denominator = calendar[(calendar >= first) & (calendar <= terminal)]
    decision_index = pd.DatetimeIndex(
        [_date(item["session_date"], "terminal decision session") for item in decisions]
    )
    if not decision_index.equals(denominator) or decision_index[-1] != terminal:
        raise V18Error("finalize-terminal requires the exact terminal denominator")
    if any(
        decisions[-1][field] != context[field]
        for field in (
            "activation_payload_sha256",
            "activation_receipt_sha256",
        )
    ):
        raise V18Error("finalize-terminal activation authority changed")

    # Final performance evidence remains sealed until the entire terminal
    # predictor/cache and checkpoint authority has been reconstructed.
    identity_registry: dict[tuple[int, int], str] = {}
    predictor_bindings = validate_predictor_evidence(
        decisions,
        None,
        source_manifest_directory=SOURCE_MANIFEST_DIR,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
        external_identity_registry=identity_registry,
    )
    validate_checkpoint_evidence(
        decisions,
        checkpoint_core_store_root=checkpoint_core_store_root,
        external_identity_registry=identity_registry,
    )

    outcomes = _load_outcome_record_authority(decisions, heal_derived=True)
    if len(outcomes) not in {len(decisions) - 1, len(decisions)}:
        raise V18Error("finalize-terminal outcome ledger is not T-1 or exact terminal")
    prior_manifests = validate_outcome_evidence(
        decisions,
        outcomes,
        outcome_manifest_directory=OUTCOME_MANIFEST_DIR,
        outcome_raw_store_root=outcome_raw_store_root,
        external_identity_registry=identity_registry,
    )
    _validate_terminal_predictor_outcome_cross_role(
        predictor_bindings["_terminal_predictor_raw_records"],
        predictor_bindings["_terminal_predictor_shard_bindings"],
        prior_manifests,
        terminal_session=terminal,
        allow_terminal_outcome_only=len(outcomes) == len(decisions),
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
        outcome_raw_store_root=outcome_raw_store_root,
    )
    received = _timestamp(source_received_at, "terminal outcome source_received_at")
    manifest_path = OUTCOME_MANIFEST_DIR / f"{terminal.date()}.json"
    if os.path.lexists(manifest_path):
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise V18Error("terminal outcome manifest is not a plain file")
        manifest, _ = validate_outcome_manifest(
            manifest_path,
            decisions[-1],
            outcome_raw_store_root=outcome_raw_store_root,
        )
        supplied_count, supplied_sha = _source_file_metadata(
            source_pdf, label="retained terminal outcome input"
        )
        expected_input = {
            "source_file_name": source_file_name,
            "source_url": _official_jpx_daily_url(
                source_url,
                "terminal outcome source URL",
                file_name=source_file_name,
                source_session=terminal,
            ),
            "source_byte_count": supplied_count,
            "source_sha256": supplied_sha,
        }
        if any(manifest[field] != expected for field, expected in expected_input.items()) or (
            _timestamp(manifest["source_received_at"], "retained terminal receipt")
            != received
        ):
            raise V18Error("finalize-terminal retry changes its exact source input")
    else:
        manifest = build_outcome_manifest(
            decisions[-1],
            source_pdf,
            outcome_raw_store_root=outcome_raw_store_root,
            source_file_name=source_file_name,
            source_url=source_url,
            source_received_at=received,
            output=manifest_path,
        )

    if len(outcomes) == len(decisions) - 1:
        combined_outcomes = attach_outcomes(
            decisions,
            manifest_path,
            outcome_raw_store_root=outcome_raw_store_root,
            existing_outcomes=outcomes,
        )
        if len(combined_outcomes) != len(decisions):
            raise V18Error("finalize-terminal did not derive exactly one final outcome")
        append_jsonl_record(
            OUTCOME_LEDGER,
            combined_outcomes[-1],
            required_fields=OUTCOME_FIELDS,
            validator=lambda rows: validate_outcome_records(rows, decisions),
        )
        combined_outcomes = _load_outcome_record_authority(
            decisions, heal_derived=True
        )
    else:
        combined_outcomes = outcomes
        if combined_outcomes[-1]["outcome_manifest_sha256"] != manifest[
            "outcome_manifest_sha256"
        ]:
            raise V18Error("retained terminal outcome ledger binding changed")

    terminal_month = terminal.to_period("M")
    completed = _load_completed_month_record_authority(heal_derived=True)
    if completed and completed[-1]["completed_month"] == str(terminal_month):
        replayed = build_completed_month_record(
            decisions,
            combined_outcomes,
            completed_month=str(terminal_month),
            created_at=completed[-1]["created_at"],
            existing_records=completed[:-1],
        )
        if canonical_json_bytes(replayed[-1]) != canonical_json_bytes(completed[-1]):
            raise V18Error("retained terminal month close differs from exact replay")
    else:
        closed = build_completed_month_record(
            decisions,
            combined_outcomes,
            completed_month=str(terminal_month),
            capture_operational_timestamp=True,
            existing_records=completed,
        )
        append_jsonl_record(
            COMPLETED_MONTH_LEDGER,
            closed[-1],
            required_fields=COMPLETED_MONTH_FIELDS,
            validator=validate_completed_month_records,
        )
        completed = _load_completed_month_record_authority(heal_derived=True)
    expected_months = [
        str(item)
        for item in pd.period_range(
            first.to_period("M"), terminal_month, freq="M"
        )
    ]
    if [item["completed_month"] for item in completed] != expected_months:
        raise V18Error("finalize-terminal completed-month prefix is incomplete")
    return {
        "terminal_session": str(terminal.date()),
        "outcome_manifest_sha256": manifest["outcome_manifest_sha256"],
        "outcome_record_sha256": combined_outcomes[-1]["record_sha256"],
        "completed_month": str(terminal_month),
        "completed_month_record_sha256": completed[-1]["record_sha256"],
        "raw_source_provenance": _raw_source_provenance_envelope(),
        "production_model_changed": False,
        "orders_allowed": False,
    }


def _frame_sha(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    canonical = frame.loc[:, list(columns)].copy()
    for column in canonical:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = canonical[column].dt.strftime("%Y-%m-%d")
    return hashlib.sha256(
        canonical.to_csv(index=False, lineterminator="\n", na_rep="<NA>").encode()
    ).hexdigest()


def _numeric_sha(values: Any) -> str:
    array = np.asarray(values, dtype="<f8")
    return canonical_json_sha256(
        {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "bytes_sha256": hashlib.sha256(
                np.ascontiguousarray(array).tobytes(order="C")
            ).hexdigest(),
        }
    )


def _numeric_array_object(values: Any, *, dtype: str) -> dict[str, Any]:
    numpy_dtype = "<f8" if dtype == "float64" else "<i8"
    array = np.asarray(values, dtype=numpy_dtype)
    value: dict[str, Any] = {
        "dtype": dtype,
        "shape": list(array.shape),
        "values": array.reshape(-1).tolist(),
    }
    if dtype == "float64" and not np.isfinite(array).all():
        raise V18Error("numeric bundle arrays must be finite")
    value["sha256"] = canonical_json_sha256(value, exclude_fields={"sha256"})
    return value


def _validate_numeric_array(
    value: Mapping[str, Any],
    *,
    name: str,
    dtype: str,
    shape: Sequence[int],
    positive: bool = False,
) -> np.ndarray:
    if set(value) != {"dtype", "shape", "values", "sha256"}:
        raise V18Error(f"{name} array fields changed")
    if value["dtype"] != dtype or value["shape"] != list(shape):
        raise V18Error(f"{name} dtype/shape changed")
    numpy_dtype = "<f8" if dtype == "float64" else "<i8"
    array = np.asarray(value["values"], dtype=numpy_dtype)
    if array.size != int(np.prod(shape, dtype=int)):
        raise V18Error(f"{name} values do not match shape")
    array = array.reshape(tuple(shape))
    if dtype == "float64" and not np.isfinite(array).all():
        raise V18Error(f"{name} contains non-finite state")
    if positive and not (array > 0.0).all():
        raise V18Error(f"{name} must be strictly positive")
    expected = canonical_json_sha256(value, exclude_fields={"sha256"})
    if value["sha256"] != expected:
        raise V18Error(f"{name} component hash changed")
    return array


def _json_file_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _safe(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def export_c00_model_bundle(
    model: Any,
    *,
    target_month: str | pd.Period,
    created_at: Any,
    runtime_lock_verified_at: Any | None = None,
    historical_runtime_replay: bool = False,
    first_counted_session_value: Any | None = None,
    _a2_rehearsal_contract: _A2RehearsalContract | None = None,
) -> dict[str, Any]:
    """Export the full fitted C00 pipeline as non-executable canonical numbers."""

    if list(getattr(model, "named_steps", {})) != ["impute", "scale", "model"]:
        raise V18Error("C00 model is not the registered three-step pipeline")
    target = _month(target_month, "bundle target_month")
    created = _timestamp(created_at, "bundle created_at")
    rehearsal = (
        None
        if _a2_rehearsal_contract is None
        else _require_active_a2_rehearsal_contract(_a2_rehearsal_contract)
    )
    verified = (
        _timestamp(runtime_lock_verified_at, "runtime_lock_verified_at")
        if rehearsal is not None
        else _runtime_verified_timestamp(
            runtime_lock_verified_at,
            allow_historical_replay=historical_runtime_replay,
        )
    )
    if verified > created:
        raise V18Error("C00 bundle creation predates runtime verification")
    if created > _cutoff(
        _month_seal_session(
            target, first_counted_session_value=first_counted_session_value
        )
    ):
        raise V18Error("C00 model bundle was not sealed before first month cutoff")
    imputer = model.named_steps["impute"]
    scaler = model.named_steps["scale"]
    ridge = model.named_steps["model"]
    indicator = np.asarray(imputer.indicator_.features_, dtype="<i8")
    transformed_order = [*G0_FEATURES, *(
        f"missingindicator::{G0_FEATURES[int(index)]}" for index in indicator
    )]
    transformed_width = len(transformed_order)
    value: dict[str, Any] = {
        "schema_version": 1,
        "target_month": str(target),
        "created_at": created,
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "runtime_lock_verified_at": verified,
        "input_feature_order": list(G0_FEATURES),
        "input_feature_dtype": "float64",
        "input_feature_shape": [len(G0_FEATURES)],
        "transformed_feature_order": transformed_order,
        "transformed_feature_dtype": "float64",
        "transformed_feature_shape": [transformed_width],
        "imputer_strategy": "median",
        "imputer_add_indicator": True,
        "imputer_keep_empty_features": False,
        "imputer_statistics": _numeric_array_object(
            imputer.statistics_, dtype="float64"
        ),
        "imputer_indicator_features": _numeric_array_object(
            indicator, dtype="int64"
        ),
        "scaler_with_mean": True,
        "scaler_with_std": True,
        "scaler_mean": _numeric_array_object(scaler.mean_, dtype="float64"),
        "scaler_scale": _numeric_array_object(scaler.scale_, dtype="float64"),
        "ridge_alpha": float(ridge.alpha),
        "ridge_fit_intercept": bool(ridge.fit_intercept),
        "ridge_coef": _numeric_array_object(ridge.coef_, dtype="float64"),
        "ridge_intercept": float(np.asarray(ridge.intercept_).reshape(-1)[0]),
        "protocol_sha256": (
            PROTOCOL_SHA256 if rehearsal is None else rehearsal.protocol_sha256
        ),
        "runner_sha256": (
            sha256_file(__file__) if rehearsal is None else rehearsal.runner_sha256
        ),
        "python_version": (
            platform.python_version()
            if rehearsal is None
            else rehearsal.runtime_python_version
        ),
        "numpy_version": (
            np.__version__ if rehearsal is None else rehearsal.numpy_version
        ),
        "scikit_learn_version": (
            sklearn.__version__
            if rehearsal is None
            else rehearsal.scikit_learn_version
        ),
        "canonical_json_contract": CANONICAL_JSON_CONTRACT,
    }
    value["fold_model_bundle_sha256"] = canonical_json_sha256(
        value, exclude_fields={"fold_model_bundle_sha256"}
    )
    validate_c00_model_bundle(
        value, _a2_rehearsal_contract=rehearsal
    )
    return value


def validate_c00_model_bundle(
    bundle: Mapping[str, Any],
    *,
    _a2_rehearsal_contract: _A2RehearsalContract | None = None,
) -> str:
    rehearsal = (
        None
        if _a2_rehearsal_contract is None
        else _require_active_a2_rehearsal_contract(_a2_rehearsal_contract)
    )
    if rehearsal is None:
        protocol = read_json(PROTOCOL)
        required = set(
            protocol["c00_contract"]["fold_model_bundle_contract"][
                "required_fields"
            ]
        )
        runtime_python_version = read_json(RUNTIME_LOCK)["runtime"]["python"][
            "version"
        ]
        runner_sha256 = sha256_file(__file__)
        expected_protocol_sha256 = PROTOCOL_SHA256
        expected_numpy_version = "2.3.5"
        expected_sklearn_version = "1.8.0"
    else:
        required = set(rehearsal.fold_model_bundle_required_fields)
        runtime_python_version = rehearsal.runtime_python_version
        runner_sha256 = rehearsal.runner_sha256
        expected_protocol_sha256 = rehearsal.protocol_sha256
        expected_numpy_version = rehearsal.numpy_version
        expected_sklearn_version = rehearsal.scikit_learn_version
    if set(bundle) != required:
        raise V18Error("C00 model bundle fields differ from protocol")
    value = dict(bundle)
    target = _month(value["target_month"], "bundle target_month")
    created = _timestamp(value["created_at"], "bundle created_at")
    verified = _timestamp(
        value["runtime_lock_verified_at"], "bundle runtime_lock_verified_at"
    )
    if (
        value["schema_version"] != 1
        or value["input_feature_order"] != list(G0_FEATURES)
        or value["input_feature_dtype"] != "float64"
        or value["input_feature_shape"] != [len(G0_FEATURES)]
        or value["transformed_feature_dtype"] != "float64"
        or value["imputer_strategy"] != "median"
        or value["imputer_add_indicator"] is not True
        or value["imputer_keep_empty_features"] is not False
        or value["scaler_with_mean"] is not True
        or value["scaler_with_std"] is not True
        or float(value["ridge_alpha"]) != 1.0
        or value["ridge_fit_intercept"] is not True
        or value["runtime_lock_sha256"] != RUNTIME_LOCK_SHA256
        or verified > created
        or value["protocol_sha256"] != expected_protocol_sha256
        or value["runner_sha256"] != runner_sha256
        or value["python_version"] != runtime_python_version
        or value["numpy_version"] != expected_numpy_version
        or value["scikit_learn_version"] != expected_sklearn_version
        or value["canonical_json_contract"] != CANONICAL_JSON_CONTRACT
    ):
        raise V18Error("C00 model bundle fixed contract changed")
    statistics = _validate_numeric_array(
        value["imputer_statistics"],
        name="imputer_statistics",
        dtype="float64",
        shape=(len(G0_FEATURES),),
    )
    indicator_shape = value["imputer_indicator_features"].get("shape")
    if not isinstance(indicator_shape, list) or len(indicator_shape) != 1:
        raise V18Error("imputer indicator shape is invalid")
    indicator = _validate_numeric_array(
        value["imputer_indicator_features"],
        name="imputer_indicator_features",
        dtype="int64",
        shape=tuple(indicator_shape),
    )
    if (
        np.any(indicator < 0)
        or np.any(indicator >= len(G0_FEATURES))
        or (len(indicator) > 1 and np.any(np.diff(indicator) <= 0))
    ):
        raise V18Error("imputer indicator indexes are invalid")
    width = len(G0_FEATURES) + len(indicator)
    expected_order = [*G0_FEATURES, *(
        f"missingindicator::{G0_FEATURES[int(index)]}" for index in indicator
    )]
    if (
        value["transformed_feature_order"] != expected_order
        or value["transformed_feature_shape"] != [width]
    ):
        raise V18Error("transformed feature order/shape changed")
    _validate_numeric_array(
        value["scaler_mean"], name="scaler_mean", dtype="float64", shape=(width,)
    )
    _validate_numeric_array(
        value["scaler_scale"],
        name="scaler_scale",
        dtype="float64",
        shape=(width,),
        positive=True,
    )
    _validate_numeric_array(
        value["ridge_coef"], name="ridge_coef", dtype="float64", shape=(width,)
    )
    if not np.isfinite(statistics).all() or not math.isfinite(
        float(value["ridge_intercept"])
    ):
        raise V18Error("C00 bundle contains non-finite fitted state")
    expected = canonical_json_sha256(
        value, exclude_fields={"fold_model_bundle_sha256"}
    )
    if value["fold_model_bundle_sha256"] != expected:
        raise V18Error("C00 model bundle self-hash mismatch")
    return expected


def predict_c00_model_bundle(
    bundle: Mapping[str, Any],
    features: Any,
    *,
    _a2_rehearsal_contract: _A2RehearsalContract | None = None,
) -> np.ndarray:
    """Clean-room score reconstruction from the registered numeric bundle."""

    validate_c00_model_bundle(
        bundle, _a2_rehearsal_contract=_a2_rehearsal_contract
    )
    with _numeric_execution(
        _a2_rehearsal_contract=_a2_rehearsal_contract
    ):
        raw = np.asarray(features, dtype="<f8")
        if raw.ndim != 2 or raw.shape[1] != len(G0_FEATURES):
            raise V18Error("C00 bundle scoring matrix has wrong shape")
        statistics = np.asarray(bundle["imputer_statistics"]["values"], dtype="<f8")
        imputed = raw.copy()
        missing = np.isnan(imputed)
        if missing.any():
            imputed[missing] = np.broadcast_to(statistics, imputed.shape)[missing]
        indicator_indexes = np.asarray(
            bundle["imputer_indicator_features"]["values"], dtype="<i8"
        )
        indicators = np.isnan(raw[:, indicator_indexes]).astype("<f8")
        transformed = (
            np.column_stack([imputed, indicators]) if indicators.shape[1] else imputed
        )
        mean = np.asarray(bundle["scaler_mean"]["values"], dtype="<f8")
        scale = np.asarray(bundle["scaler_scale"]["values"], dtype="<f8")
        coefficients = np.asarray(bundle["ridge_coef"]["values"], dtype="<f8")
        return ((transformed - mean) / scale) @ coefficients + float(
            bundle["ridge_intercept"]
        )


def _build_fold(
    panel: pd.DataFrame,
    target_month: pd.Period,
    *,
    fit_started_at: Any | None = None,
    fit_completed_at: Any | None = None,
    bundle_created_at: Any | None = None,
    sealed_at: Any | None = None,
    runtime_lock_verified_at: Any | None = None,
    capture_operational_timestamps: bool = False,
    terminal_replay: bool = False,
    first_counted_session_value: Any | None = None,
    activation_observed_at: Any | None = None,
    month_source_manifest: Mapping[str, Any] | None = None,
    _a2_rehearsal_contract: _A2RehearsalContract | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    rehearsal = (
        None
        if _a2_rehearsal_contract is None
        else _require_active_a2_rehearsal_contract(_a2_rehearsal_contract)
    )
    month_start = target_month.start_time.normalize()
    if month_source_manifest is None:
        raise V18Error("C00 fold requires the sealed independent month-source manifest")
    month_source = dict(month_source_manifest)
    if set(month_source) != set(MONTH_SOURCE_MANIFEST_FIELDS):
        raise V18Error("C00 fold month-source schema changed")
    if month_source["target_month"] != str(target_month):
        raise V18Error("C00 fold month-source target differs")
    expected_month_source_hash = canonical_json_sha256(
        month_source, exclude_fields={"month_source_manifest_sha256"}
    )
    if month_source["month_source_manifest_sha256"] != expected_month_source_hash:
        raise V18Error("C00 fold month-source self-hash changed")
    verified = (
        _timestamp(runtime_lock_verified_at, "runtime_lock_verified_at")
        if rehearsal is not None
        else _runtime_verified_timestamp(
            runtime_lock_verified_at, allow_historical_replay=terminal_replay
        )
    )
    if capture_operational_timestamps:
        if any(
            item is not None
            for item in (fit_started_at, fit_completed_at, bundle_created_at, sealed_at)
        ):
            raise V18Error("canonical fold timestamps are captured internally")
    else:
        if fit_started_at is None or fit_completed_at is None:
            raise V18Error("pure fold replay requires explicit fit timestamps")
        fit_started_at = _timestamp(fit_started_at, "fold fit_started_at")
        fit_completed_at = _timestamp(fit_completed_at, "fold fit_completed_at")
        bundle_created_at = _timestamp(
            fit_completed_at if bundle_created_at is None else bundle_created_at,
            "bundle created_at",
        )
        sealed_at = _timestamp(
            bundle_created_at if sealed_at is None else sealed_at, "fold sealed_at"
        )
    first_cutoff = _cutoff(
        _month_seal_session(
            target_month, first_counted_session_value=first_counted_session_value
        )
    )
    training = v17._candidate_training(
        panel, end=month_start - pd.Timedelta(days=1)
    )
    if training["date"].max() >= month_start:
        raise V18Error("C00 monthly training leaked target-month labels")
    # Ridge ultimately dispatches to native BLAS/LAPACK.  Constraining that
    # dispatch to one thread is part of the preregistered reproducibility
    # boundary: a machine's ambient thread-pool settings may not change the
    # persisted numeric bundle.
    if capture_operational_timestamps:
        fit_started_at = datetime.now(TOKYO)
    with _numeric_execution(_a2_rehearsal_contract=rehearsal):
        model = v17.fit_daily_rank_ridge(training, G0_FEATURES)
    if capture_operational_timestamps:
        fit_completed_at = datetime.now(TOKYO)
        bundle_created_at = datetime.now(TOKYO)
    assert fit_started_at is not None and fit_completed_at is not None
    assert bundle_created_at is not None
    if verified > fit_started_at or fit_started_at > fit_completed_at:
        raise V18Error("fold runtime/fit timestamps are not causal")
    if _timestamp(
        month_source["sealed_at"], "fold month-source sealed_at"
    ) > fit_started_at:
        raise V18Error("C00 fold fit predates its sealed month-source authority")
    if activation_observed_at is not None and fit_started_at < _timestamp(
        activation_observed_at, "activation_receipt_commit_observed_at"
    ):
        raise V18Error("C00 fold fit timestamp predates activation observation")
    bundle = export_c00_model_bundle(
        model,
        target_month=target_month,
        created_at=bundle_created_at,
        runtime_lock_verified_at=verified,
        historical_runtime_replay=terminal_replay,
        first_counted_session_value=first_counted_session_value,
        _a2_rehearsal_contract=rehearsal,
    )
    target = training["_daily_rank_target"].to_numpy(dtype=float)
    identity = training.loc[:, ["date", "code"]].sort_values(
        ["date", "code"], kind="stable"
    )
    coefficients = model.named_steps["model"].coef_
    intercept = model.named_steps["model"].intercept_
    if capture_operational_timestamps:
        sealed_at = datetime.now(TOKYO)
    assert sealed_at is not None
    if fit_completed_at > bundle_created_at or bundle_created_at > sealed_at:
        raise V18Error("fold fit/bundle/seal timestamps are not causal")
    if sealed_at > first_cutoff:
        raise V18Error("monthly C00 fold was not sealed before first cutoff")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "target_month": str(target_month),
        "fit_started_at": fit_started_at,
        "fit_completed_at": fit_completed_at,
        "sealed_at": sealed_at,
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "runtime_lock_verified_at": verified,
        "training_first_session": str(training["date"].min().date()),
        "training_last_session": str(training["date"].max().date()),
        "training_session_count": int(training["date"].nunique()),
        "training_row_identity_sha256": _frame_sha(identity, ("date", "code")),
        "training_target_sha256": _numeric_sha(target),
        "feature_matrix_sha256": _numeric_sha(
            training.loc[:, list(G0_FEATURES)].to_numpy(dtype=float)
        ),
        "feature_names": list(G0_FEATURES),
        "month_source_manifest_path": (
            "research/model_v18_shoulder_state_month_source_manifests/"
            f"{target_month}.json"
        ),
        "month_source_manifest_sha256": expected_month_source_hash,
        "training_source_set_sha256": month_source["source_set_sha256"],
        "training_parsed_shard_set_sha256": month_source[
            "parsed_shard_set_sha256"
        ],
        "training_g0_panel_semantic_sha256": month_source[
            "g0_training_panel_semantic_sha256"
        ],
        "ridge_alpha": 1.0,
        "fold_model_bundle_path": (
            "research/model_v18_shoulder_state_fold_models/"
            f"{target_month}.json"
        ),
        "fold_model_bundle_schema_version": 1,
        "fold_model_bundle_file_sha256": hashlib.sha256(
            _json_file_bytes(bundle)
        ).hexdigest(),
        "fold_model_bundle_sha256": bundle["fold_model_bundle_sha256"],
        "input_feature_order_sha256": canonical_json_sha256(list(G0_FEATURES)),
        "transformed_feature_order_sha256": canonical_json_sha256(
            bundle["transformed_feature_order"]
        ),
        "imputer_statistics_sha256": bundle["imputer_statistics"]["sha256"],
        "imputer_indicator_features_sha256": bundle[
            "imputer_indicator_features"
        ]["sha256"],
        "scaler_mean_sha256": bundle["scaler_mean"]["sha256"],
        "scaler_scale_sha256": bundle["scaler_scale"]["sha256"],
        "ridge_coef_sha256": bundle["ridge_coef"]["sha256"],
        "ridge_intercept": float(np.asarray(intercept).reshape(-1)[0]),
        "universe_contract_sha256": (
            canonical_json_sha256(read_json(PROTOCOL)["c00_contract"])
            if rehearsal is None
            else canonical_json_sha256(json.loads(rehearsal.c00_contract_json))
        ),
        "protocol_sha256": (
            PROTOCOL_SHA256 if rehearsal is None else rehearsal.protocol_sha256
        ),
        "runner_sha256": (
            sha256_file(__file__) if rehearsal is None else rehearsal.runner_sha256
        ),
        "python_version": (
            platform.python_version()
            if rehearsal is None
            else rehearsal.runtime_python_version
        ),
        "numpy_version": (
            np.__version__ if rehearsal is None else rehearsal.numpy_version
        ),
        "pandas_version": (
            pd.__version__ if rehearsal is None else rehearsal.pandas_version
        ),
        "scikit_learn_version": (
            sklearn.__version__
            if rehearsal is None
            else rehearsal.scikit_learn_version
        ),
        "canonical_json_contract": CANONICAL_JSON_CONTRACT,
    }
    manifest["fold_manifest_sha256"] = canonical_json_sha256(
        manifest, exclude_fields={"fold_manifest_sha256"}
    )
    return model, manifest, bundle


def validate_fold_manifest(
    manifest: Mapping[str, Any],
    bundle: Mapping[str, Any] | None = None,
    *,
    _a2_rehearsal_contract: _A2RehearsalContract | None = None,
    _a2_rehearsal_month_source: Mapping[str, Any] | None = None,
) -> str:
    rehearsal = (
        None
        if _a2_rehearsal_contract is None
        else _require_active_a2_rehearsal_contract(_a2_rehearsal_contract)
    )
    if rehearsal is None:
        protocol = read_json(PROTOCOL)
        required = set(protocol["c00_contract"]["fold_manifest_required_fields"])
        expected_protocol_sha256 = PROTOCOL_SHA256
        runner_sha256 = sha256_file(__file__)
        runtime_python_version = read_json(RUNTIME_LOCK)["runtime"]["python"][
            "version"
        ]
        expected_numpy_version = "2.3.5"
        expected_pandas_version = "2.2.3"
        expected_sklearn_version = "1.8.0"
    else:
        if _a2_rehearsal_month_source is None:
            raise V18Error("A2 rehearsal fold lacks its in-memory month source")
        required = set(rehearsal.fold_manifest_required_fields)
        expected_protocol_sha256 = rehearsal.protocol_sha256
        runner_sha256 = rehearsal.runner_sha256
        runtime_python_version = rehearsal.runtime_python_version
        expected_numpy_version = rehearsal.numpy_version
        expected_pandas_version = rehearsal.pandas_version
        expected_sklearn_version = rehearsal.scikit_learn_version
    missing = sorted(required - set(manifest))
    if missing:
        raise V18Error(f"fold manifest is missing fields: {missing}")
    if set(manifest) != required:
        raise V18Error("fold manifest fields differ from protocol")
    if manifest["protocol_sha256"] != expected_protocol_sha256:
        raise V18Error("fold protocol hash changed")
    if manifest["runner_sha256"] != runner_sha256:
        raise V18Error("fold runner hash changed")
    if manifest["runtime_lock_sha256"] != RUNTIME_LOCK_SHA256:
        raise V18Error("fold runtime-lock binding changed")
    if manifest["feature_names"] != list(G0_FEATURES) or float(manifest["ridge_alpha"]) != 1.0:
        raise V18Error("fold C00 feature/alpha contract changed")
    target = _month(manifest["target_month"], "fold target_month")
    expected_month_source_path = (
        "research/model_v18_shoulder_state_month_source_manifests/"
        f"{target}.json"
    )
    if manifest["month_source_manifest_path"] != expected_month_source_path:
        raise V18Error("fold month-source manifest path changed")
    for field in (
        "month_source_manifest_sha256",
        "training_source_set_sha256",
        "training_parsed_shard_set_sha256",
        "training_g0_panel_semantic_sha256",
    ):
        _require_sha(manifest[field], f"fold {field}")
    month_source_path = ROOT / expected_month_source_path
    month_source_sealed: datetime | None = None
    month_source_cutoff: datetime | None = None
    if rehearsal is not None:
        month_source = dict(_a2_rehearsal_month_source or {})
        has_month_source = True
    else:
        has_month_source = month_source_path.is_file()
        month_source = read_json(month_source_path) if has_month_source else {}
    if has_month_source:
        if set(month_source) != set(MONTH_SOURCE_MANIFEST_FIELDS):
            raise V18Error("fold canonical month-source schema changed")
        expected_month_source_hash = canonical_json_sha256(
            month_source, exclude_fields={"month_source_manifest_sha256"}
        )
        if month_source["month_source_manifest_sha256"] != expected_month_source_hash:
            raise V18Error("fold canonical month-source self-hash changed")
        month_source_sealed = _timestamp(
            month_source["sealed_at"], "fold month-source sealed_at"
        )
        month_source_cutoff = _cutoff(
            _date(month_source["seal_session"], "fold month-source seal session")
        )
        bindings = {
            "month_source_manifest_sha256": month_source[
                "month_source_manifest_sha256"
            ],
            "training_source_set_sha256": month_source["source_set_sha256"],
            "training_parsed_shard_set_sha256": month_source[
                "parsed_shard_set_sha256"
            ],
            "training_g0_panel_semantic_sha256": month_source[
                "g0_training_panel_semantic_sha256"
            ],
        }
        for field, expected in bindings.items():
            if manifest[field] != expected:
                raise V18Error(f"fold month-source binding changed: {field}")
    expected_path = f"research/model_v18_shoulder_state_fold_models/{target}.json"
    if manifest["fold_model_bundle_path"] != expected_path:
        raise V18Error("fold model bundle path changed")
    if rehearsal is not None and bundle is None:
        raise V18Error("A2 rehearsal fold lacks its in-memory model bundle")
    bundle_value = read_json(ROOT / expected_path) if bundle is None else dict(bundle)
    bundle_hash = validate_c00_model_bundle(
        bundle_value, _a2_rehearsal_contract=rehearsal
    )
    if bundle_value["target_month"] != str(target):
        raise V18Error("fold model bundle target month differs")
    fit_started = _timestamp(manifest["fit_started_at"], "fold fit_started_at")
    fit_completed = _timestamp(manifest["fit_completed_at"], "fold fit_completed_at")
    sealed = _timestamp(manifest["sealed_at"], "fold sealed_at")
    verified = _timestamp(
        manifest["runtime_lock_verified_at"], "fold runtime_lock_verified_at"
    )
    bundle_created = _timestamp(bundle_value["created_at"], "bundle created_at")
    if (
        verified > fit_started
        or fit_started > fit_completed
        or fit_completed > bundle_created
        or bundle_created > sealed
        or bundle_value["runtime_lock_sha256"] != RUNTIME_LOCK_SHA256
        or _timestamp(
            bundle_value["runtime_lock_verified_at"],
            "bundle runtime_lock_verified_at",
        )
        != verified
    ):
        raise V18Error("fold/bundle observed timestamps are not causal")
    if month_source_sealed is not None and month_source_sealed > fit_started:
        raise V18Error("fold fit predates its canonical month-source seal")
    if month_source_cutoff is not None and sealed > month_source_cutoff:
        raise V18Error("fold seal is later than its canonical month cutoff")
    file_hash = hashlib.sha256(_json_file_bytes(bundle_value)).hexdigest()
    expected_bundle_fields = {
        "fold_model_bundle_schema_version": 1,
        "fold_model_bundle_file_sha256": file_hash,
        "fold_model_bundle_sha256": bundle_hash,
        "input_feature_order_sha256": canonical_json_sha256(list(G0_FEATURES)),
        "transformed_feature_order_sha256": canonical_json_sha256(
            bundle_value["transformed_feature_order"]
        ),
        "imputer_statistics_sha256": bundle_value["imputer_statistics"]["sha256"],
        "imputer_indicator_features_sha256": bundle_value[
            "imputer_indicator_features"
        ]["sha256"],
        "scaler_mean_sha256": bundle_value["scaler_mean"]["sha256"],
        "scaler_scale_sha256": bundle_value["scaler_scale"]["sha256"],
        "ridge_coef_sha256": bundle_value["ridge_coef"]["sha256"],
    }
    for field, expected_value in expected_bundle_fields.items():
        if manifest[field] != expected_value:
            raise V18Error(f"fold manifest bundle binding changed: {field}")
    if not math.isclose(
        float(manifest["ridge_intercept"]),
        float(bundle_value["ridge_intercept"]),
        abs_tol=0,
        rel_tol=0,
    ):
        raise V18Error("fold manifest ridge intercept changed")
    if (
        manifest["python_version"] != runtime_python_version
        or manifest["numpy_version"] != expected_numpy_version
        or manifest["pandas_version"] != expected_pandas_version
        or manifest["scikit_learn_version"] != expected_sklearn_version
    ):
        raise V18Error("fold manifest runtime versions differ from lock")
    expected = canonical_json_sha256(manifest, exclude_fields={"fold_manifest_sha256"})
    if manifest["fold_manifest_sha256"] != expected:
        raise V18Error("fold manifest self-hash mismatch")
    return expected


def _fold_stage_object_key(
    target_month: pd.Period, month_source_manifest_sha256: str
) -> str:
    identity = {
        "cache_contract_id": PREDICTOR_CACHE_CONTRACT_ID,
        "target_month": str(target_month),
        "month_source_manifest_sha256": _require_sha(
            month_source_manifest_sha256, "fold-stage month-source SHA"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
    }
    return f"{FOLD_STAGE_OBJECT_PREFIX}{canonical_json_sha256(identity)}.json"


def _load_or_create_month_fold(
    training_panel: pd.DataFrame,
    *,
    target_month: pd.Period,
    month_source_manifest: Mapping[str, Any],
    predictor_derived_store_root: str | Path,
    first_counted_session_value: Any,
    activation_observed_at: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Persist fold+bundle bytes before installing either canonical file."""

    month_source = dict(month_source_manifest)
    month_source_hash = _require_sha(
        month_source["month_source_manifest_sha256"],
        "fold-stage month-source manifest SHA",
    )
    object_key = _fold_stage_object_key(target_month, month_source_hash)
    if _external_object_exists(
        predictor_derived_store_root,
        object_key,
        prefix=FOLD_STAGE_OBJECT_PREFIX,
        label="fold-stage artifact",
    ):
        stage, _ = _read_external_canonical_json(
            predictor_derived_store_root,
            object_key,
            prefix=FOLD_STAGE_OBJECT_PREFIX,
            label="fold-stage artifact",
        )
        required = {
            "schema_version",
            "cache_contract_id",
            "target_month",
            "month_source_manifest_sha256",
            "fold_manifest",
            "fold_model_bundle",
            "protocol_sha256",
            "runner_sha256",
            "runtime_lock_sha256",
            "stage_sha256",
        }
        if set(stage) != required:
            raise V18Error("fold-stage fields differ from A2 contract")
        expected_fixed = {
            "schema_version": 1,
            "cache_contract_id": PREDICTOR_CACHE_CONTRACT_ID,
            "target_month": str(target_month),
            "month_source_manifest_sha256": month_source_hash,
            "protocol_sha256": PROTOCOL_SHA256,
            "runner_sha256": sha256_file(__file__),
            "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        }
        if any(stage[field] != expected for field, expected in expected_fixed.items()):
            raise V18Error("fold-stage fixed identity changed")
        if stage["stage_sha256"] != canonical_json_sha256(
            stage, exclude_fields={"stage_sha256"}
        ):
            raise V18Error("fold-stage self-hash changed")
        fold = dict(stage["fold_manifest"])
        bundle = dict(stage["fold_model_bundle"])
        validate_fold_manifest(fold, bundle)
    else:
        _, fold, bundle = _build_fold(
            training_panel,
            target_month,
            capture_operational_timestamps=True,
            first_counted_session_value=first_counted_session_value,
            activation_observed_at=activation_observed_at,
            month_source_manifest=month_source,
        )
        stage = {
            "schema_version": 1,
            "cache_contract_id": PREDICTOR_CACHE_CONTRACT_ID,
            "target_month": str(target_month),
            "month_source_manifest_sha256": month_source_hash,
            "fold_manifest": fold,
            "fold_model_bundle": bundle,
            "protocol_sha256": PROTOCOL_SHA256,
            "runner_sha256": sha256_file(__file__),
            "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        }
        stage["stage_sha256"] = canonical_json_sha256(
            stage, exclude_fields={"stage_sha256"}
        )
        _write_external_bytes_once(
            _json_file_bytes(stage),
            store_root=predictor_derived_store_root,
            object_key=object_key,
            prefix=FOLD_STAGE_OBJECT_PREFIX,
            label="fold-stage artifact",
            replace_unpublished_stage=True,
        )
    training = v17._candidate_training(
        training_panel, end=target_month.start_time.normalize() - pd.Timedelta(days=1)
    )
    identity = training.loc[:, ["date", "code"]].sort_values(
        ["date", "code"], kind="stable"
    )
    current_hashes = {
        "training_row_identity_sha256": _frame_sha(identity, ("date", "code")),
        "training_target_sha256": _numeric_sha(
            training["_daily_rank_target"].to_numpy(dtype=float)
        ),
        "feature_matrix_sha256": _numeric_sha(
            training.loc[:, list(G0_FEATURES)].to_numpy(dtype=float)
        ),
    }
    if any(fold[field] != expected for field, expected in current_hashes.items()):
        raise V18Error("fold-stage training input differs from current month source")
    bundle_path = FOLD_MODEL_DIR / f"{target_month}.json"
    fold_path = FOLD_MANIFEST_DIR / f"{target_month}.json"
    bundle_payload = _write_json_once_exact(
        bundle, bundle_path, label="canonical fold model bundle"
    )
    if hashlib.sha256(bundle_payload).hexdigest() != fold[
        "fold_model_bundle_file_sha256"
    ]:
        raise V18Error("fold-stage canonical bundle file hash changed")
    _write_json_once_exact(fold, fold_path, label="canonical fold manifest")
    validate_fold_manifest(fold, bundle)
    return fold, bundle


SCORE_FIELDS = (
    "session_date",
    "source_rank",
    "code",
    "name",
    "model_score",
    "feature_source_max_date",
    "score_generated_at",
    "runtime_lock_sha256",
    "runtime_lock_verified_at",
    "source_manifest_sha256",
    "c00_fold_manifest_sha256",
)


def freeze_c00_top2(
    panel: pd.DataFrame,
    session_date: Any,
    *,
    score_generated_at: Any | None = None,
    runtime_lock_verified_at: Any | None = None,
    capture_operational_timestamp: bool = False,
    terminal_replay: bool = False,
    source_manifest_sha256: str,
    fit_started_at: Any | None = None,
    fit_completed_at: Any | None = None,
    first_counted_session_value: Any | None = None,
    activation_observed_at: Any | None = None,
    return_manifest: bool = False,
    return_bundle: bool = False,
    model_bundle: Mapping[str, Any] | None = None,
    fold_manifest: Mapping[str, Any] | None = None,
    month_source_manifest: Mapping[str, Any] | None = None,
    _a2_rehearsal_contract: _A2RehearsalContract | None = None,
) -> (
    pd.DataFrame
    | tuple[pd.DataFrame, dict[str, Any]]
    | tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]
):
    """Reproduce v1.7 C00 monthly Ridge and freeze one outcome-free top pair."""

    rehearsal = (
        None
        if _a2_rehearsal_contract is None
        else _require_active_a2_rehearsal_contract(_a2_rehearsal_contract)
    )
    target = _date(session_date, "session_date")
    verified = (
        _timestamp(runtime_lock_verified_at, "runtime_lock_verified_at")
        if rehearsal is not None
        else _runtime_verified_timestamp(
            runtime_lock_verified_at, allow_historical_replay=terminal_replay
        )
    )
    if capture_operational_timestamp:
        if score_generated_at is not None:
            raise V18Error("canonical score timestamp is captured internally")
        generated: datetime | None = None
    else:
        if score_generated_at is None:
            raise V18Error("pure score replay requires score_generated_at")
        generated = _timestamp(score_generated_at, "score_generated_at")
    _require_sha(source_manifest_sha256, "source_manifest_sha256")
    required = {
        "date",
        "code",
        "name",
        "oc_return_pct",
        "common_training_eligible",
        "common_score_eligible",
        "feature_source_max_date",
        *G0_FEATURES,
    }
    missing = sorted(required - set(panel))
    if missing:
        raise V18Error(f"parsed panel lacks C00 columns: {missing}")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    frame["feature_source_max_date"] = pd.to_datetime(
        frame["feature_source_max_date"], errors="coerce", format="mixed"
    )
    if frame["date"].isna().any() or frame[["date", "code"]].duplicated().any():
        raise V18Error("parsed panel date/code integrity failed")
    for column in ("date", "feature_source_max_date"):
        if getattr(frame[column].dt, "tz", None) is not None:
            frame[column] = frame[column].dt.tz_convert(TOKYO).dt.tz_localize(None)
        frame[column] = frame[column].dt.normalize()
    target_rows = frame.loc[frame["date"].eq(target)]
    if target_rows.empty:
        raise V18Error("parsed panel has no target rows")
    if target_rows["oc_return_pct"].notna().any():
        raise V18Error("target-session outcome is present in pre-open C00 panel")
    if target_rows["feature_source_max_date"].isna().any() or target_rows[
        "feature_source_max_date"
    ].ge(target).any():
        raise V18Error("target C00 feature source is not strictly prior")
    month = target.to_period("M")
    if (model_bundle is None) != (fold_manifest is None):
        raise V18Error("reused C00 fold requires both bundle and manifest")
    if model_bundle is None:
        if fit_started_at is None or fit_completed_at is None:
            raise V18Error("new C00 fold requires explicit observed fit timestamps")
        started = _timestamp(fit_started_at, "fit_started_at")
        completed = _timestamp(fit_completed_at, "fit_completed_at")
        model, fold, bundle = _build_fold(
            frame,
            month,
            fit_started_at=started,
            fit_completed_at=completed,
            runtime_lock_verified_at=verified,
            first_counted_session_value=first_counted_session_value,
            activation_observed_at=activation_observed_at,
            month_source_manifest=month_source_manifest,
            _a2_rehearsal_contract=rehearsal,
        )
    else:
        bundle = dict(model_bundle)
        bundle_hash = validate_c00_model_bundle(
            bundle, _a2_rehearsal_contract=rehearsal
        )
        fold = dict(fold_manifest or {})
        validate_fold_manifest(
            fold,
            bundle,
            _a2_rehearsal_contract=rehearsal,
            _a2_rehearsal_month_source=(
                month_source_manifest if rehearsal is not None else None
            ),
        )
        if fold["target_month"] != str(month):
            raise V18Error("reused C00 fold target month differs")
        if fold.get("fold_model_bundle_sha256") != bundle_hash:
            raise V18Error("C00 fold does not bind model bundle")
        started = _timestamp(fold["fit_started_at"], "fit_started_at")
        completed = _timestamp(fold["fit_completed_at"], "fit_completed_at")
        if started > completed or completed > _cutoff(
            _month_seal_session(
                month, first_counted_session_value=first_counted_session_value
            )
        ):
            raise V18Error("reused C00 fold timestamps violate the monthly seal")
        if activation_observed_at is not None and started < _timestamp(
            activation_observed_at, "activation_receipt_commit_observed_at"
        ):
            raise V18Error("reused C00 fold predates activation observation")
        model = None
    scoring = v17._candidate_scoring(frame, pd.DatetimeIndex([target]))
    if scoring.empty:
        raise V18Error("C00 target score universe is empty")
    scoring_matrix = scoring.loc[:, list(G0_FEATURES)]
    with _numeric_execution(_a2_rehearsal_contract=rehearsal):
        predicted = (
            model.predict(scoring_matrix)
            if model is not None
            else predict_c00_model_bundle(
                bundle,
                scoring_matrix,
                _a2_rehearsal_contract=rehearsal,
            )
        )
        ranked = v17._top_scored_rows(scoring, predicted, count=2)
    if capture_operational_timestamp:
        generated = datetime.now(TOKYO)
    assert generated is not None
    if verified > generated or generated > _cutoff(target):
        raise V18Error("score runtime/generation timestamp is outside its PIT window")
    if generated < completed:
        raise V18Error("score generation predates monthly fold completion")
    if activation_observed_at is not None and generated < _timestamp(
        activation_observed_at, "activation_receipt_commit_observed_at"
    ):
        raise V18Error("score generation predates activation observation")
    if len(ranked) != 2 or sorted(ranked["source_rank"].astype(int)) != [1, 2]:
        raise V18Error("C00 could not seal two distinct top codes")
    if ranked["code"].astype(str).nunique() != 2:
        raise V18Error("C00 frozen pair codes are not distinct")
    output = pd.DataFrame(
        {
            "session_date": [str(target.date())] * 2,
            "source_rank": ranked["source_rank"].astype(int).to_list(),
            "code": ranked["code"].astype(str).to_list(),
            "name": ranked["name"].astype(str).to_list(),
            "model_score": ranked["c00_model_score"].astype(float).to_list(),
            "feature_source_max_date": ranked["feature_source_max_date"].dt.strftime(
                "%Y-%m-%d"
            ).to_list(),
            "score_generated_at": [generated.isoformat(timespec="microseconds")] * 2,
            "runtime_lock_sha256": [RUNTIME_LOCK_SHA256] * 2,
            "runtime_lock_verified_at": [
                verified.isoformat(timespec="microseconds")
            ]
            * 2,
            "source_manifest_sha256": [source_manifest_sha256] * 2,
            "c00_fold_manifest_sha256": [fold["fold_manifest_sha256"]] * 2,
        }
    )
    output = validate_score_rows(output)
    if return_bundle:
        return output, fold, bundle
    return (output, fold) if return_manifest else output


def validate_score_rows(scores: pd.DataFrame) -> pd.DataFrame:
    forbidden = sorted(FORBIDDEN_DECISION_COLUMNS & set(scores))
    if forbidden:
        raise V18Error(f"score rows contain target outcomes: {forbidden}")
    missing = sorted(set(SCORE_FIELDS) - set(scores))
    if missing:
        raise V18Error(f"score rows are missing fields: {missing}")
    frame = scores.loc[:, list(SCORE_FIELDS)].copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"], errors="coerce")
    frame["feature_source_max_date"] = pd.to_datetime(
        frame["feature_source_max_date"], errors="coerce"
    )
    frame["source_rank"] = pd.to_numeric(frame["source_rank"], errors="coerce")
    frame["model_score"] = pd.to_numeric(frame["model_score"], errors="coerce")
    if frame[["session_date", "feature_source_max_date"]].isna().any(axis=None):
        raise V18Error("score date is invalid")
    if len(frame) != 2 or sorted(frame["source_rank"].astype(int)) != [1, 2]:
        raise V18Error("score output must contain exactly ranks one and two")
    if frame["session_date"].nunique() != 1 or frame["code"].astype(str).nunique() != 2:
        raise V18Error("score output is not one distinct pair")
    if frame["feature_source_max_date"].ge(frame["session_date"]).any():
        raise V18Error("score feature source is not strictly prior")
    if not np.isfinite(frame["model_score"].to_numpy(dtype=float)).all():
        raise V18Error("score output contains non-finite scores")
    for value in frame["score_generated_at"]:
        if _timestamp(value, "score_generated_at") > _cutoff(frame["session_date"].iloc[0]):
            raise V18Error("score output was generated after cutoff")
    if frame["runtime_lock_sha256"].nunique() != 1 or frame[
        "runtime_lock_sha256"
    ].iloc[0] != RUNTIME_LOCK_SHA256:
        raise V18Error("score runtime-lock binding changed")
    if frame["runtime_lock_verified_at"].nunique() != 1:
        raise V18Error("score pair has different runtime verification timestamps")
    verified = _timestamp(
        frame["runtime_lock_verified_at"].iloc[0], "score runtime_lock_verified_at"
    )
    generated = _timestamp(frame["score_generated_at"].iloc[0], "score_generated_at")
    if verified > generated:
        raise V18Error("score generation predates runtime verification")
    for field in ("source_manifest_sha256", "c00_fold_manifest_sha256"):
        if frame[field].nunique() != 1:
            raise V18Error(f"score pair has different {field}")
        _require_sha(frame[field].iloc[0], field)
    frame["session_date"] = frame["session_date"].dt.strftime("%Y-%m-%d")
    frame["feature_source_max_date"] = frame["feature_source_max_date"].dt.strftime(
        "%Y-%m-%d"
    )
    return frame.sort_values("source_rank", kind="stable").reset_index(drop=True)


def semantic_score_hash(scores: pd.DataFrame) -> str:
    return hashlib.sha256(
        validate_score_rows(scores).to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()


def _a2_rehearsal_canonical_authority_paths() -> tuple[Path, ...]:
    return (
        *_registered_local_authority_files(),
        DECISION_LEDGER,
        OUTCOME_LEDGER,
        COMPLETED_MONTH_LEDGER,
        SCORE_OUTPUT,
        PICKS_OUTPUT,
        CHECKPOINT_PROPOSAL_DIR,
        *_registered_local_authority_directories(),
    )


def _a2_rehearsal_preflight_payload() -> dict[str, Any]:
    if not _STRICT_RUNTIME_ACTIVE or _STRICT_RUNTIME_VERIFIED_AT is None:
        raise V18Error("A2 nonauthority rehearsal requires the strict runtime")
    if any(os.path.lexists(path) for path in _a2_rehearsal_canonical_authority_paths()):
        raise V18Error("A2 nonauthority rehearsal refuses canonical authority")
    _validate_startup_and_module_closure(phase="A2 rehearsal untimed preflight")
    protocol, protocol_sha256 = validate_protocol()
    runtime_lock, runtime_lock_sha256 = validate_runtime_lock(
        strict_environment=False
    )
    calendar = load_registered_calendar()
    target = pd.Timestamp("2026-08-05")
    latest = _latest_required_predictor_source_session(target, calendar)
    if latest != pd.Timestamp("2026-08-04"):
        raise V18Error("A2 rehearsal fixed target/D-1 contract changed")
    runtime = runtime_lock["runtime"]
    expected_limit = int(runtime["numeric_execution_contract"]["threadpoolctl_limit"])
    if expected_limit != 1:
        raise V18Error("A2 rehearsal numeric thread limit changed")
    with threadpoolctl.threadpool_limits(limits=expected_limit):
        observed = _normalised_native_threadpools(hash_libraries=False)
    expected_native = [
        {
            key: (expected_limit if key == "num_threads" else child)
            for key, child in item.items()
            if key != "library_sha256"
        }
        for item in runtime["native_threadpools"]
    ]
    if observed != expected_native:
        raise V18Error("A2 rehearsal native numeric backend changed")
    versions = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }
    if versions != {
        "python_version": runtime["python"]["version"],
        "numpy_version": "2.3.5",
        "pandas_version": "2.2.3",
        "scikit_learn_version": "1.8.0",
    }:
        raise V18Error("A2 rehearsal runtime versions changed")
    c00_contract = protocol["c00_contract"]
    c00_contract_json = json.dumps(
        c00_contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        "protocol_sha256": protocol_sha256,
        "runner_sha256": sha256_file(__file__),
        "runtime_lock_sha256": runtime_lock_sha256,
        "calendar_sha256": CALENDAR_SHA256,
        "calendar_session_count": int(len(calendar)),
        "calendar_first_session": str(pd.Timestamp(calendar[0]).date()),
        "calendar_last_session": str(pd.Timestamp(calendar[-1]).date()),
        "c00_contract_sha256": canonical_json_sha256(c00_contract),
        "c00_contract_json": c00_contract_json,
        "fold_manifest_required_fields": list(
            c00_contract["fold_manifest_required_fields"]
        ),
        "fold_model_bundle_required_fields": list(
            c00_contract["fold_model_bundle_contract"]["required_fields"]
        ),
        **versions,
        "numeric_threadpool_limit": expected_limit,
        "target_session": str(target.date()),
        "latest_required_source_session": str(latest.date()),
    }


def prepare_a2_nonauthority_rehearsal_contract() -> _A2RehearsalContract:
    """Perform all project/runtime I/O before the separately timed pure seam."""

    global _ACTIVE_A2_REHEARSAL_CONTRACT
    if _ACTIVE_A2_REHEARSAL_CONTRACT is not None:
        raise V18Error("A2 rehearsal capability is already active")
    payload = _a2_rehearsal_preflight_payload()
    capability = _A2RehearsalContract(
        nonce=secrets.token_hex(32),
        protocol_sha256=str(payload["protocol_sha256"]),
        runner_sha256=str(payload["runner_sha256"]),
        runtime_lock_sha256=str(payload["runtime_lock_sha256"]),
        calendar_sha256=str(payload["calendar_sha256"]),
        c00_contract_json=str(payload["c00_contract_json"]),
        fold_manifest_required_fields=tuple(
            payload["fold_manifest_required_fields"]
        ),
        fold_model_bundle_required_fields=tuple(
            payload["fold_model_bundle_required_fields"]
        ),
        runtime_python_version=str(payload["python_version"]),
        numpy_version=str(payload["numpy_version"]),
        pandas_version=str(payload["pandas_version"]),
        scikit_learn_version=str(payload["scikit_learn_version"]),
        numeric_threadpool_limit=int(payload["numeric_threadpool_limit"]),
        target_session=str(payload["target_session"]),
        latest_required_source_session=str(
            payload["latest_required_source_session"]
        ),
        preflight_sha256=canonical_json_sha256(payload),
    )
    _ACTIVE_A2_REHEARSAL_CONTRACT = capability
    return capability


def validate_a2_nonauthority_rehearsal_postflight(
    rehearsal_contract: _A2RehearsalContract,
) -> str:
    """Revalidate the untimed closure/absence snapshot and revoke capability."""

    global _ACTIVE_A2_REHEARSAL_CONTRACT
    capability = _require_active_a2_rehearsal_contract(rehearsal_contract)
    try:
        observed = canonical_json_sha256(_a2_rehearsal_preflight_payload())
        if observed != capability.preflight_sha256:
            raise V18Error("A2 rehearsal preflight/postflight contract changed")
        return observed
    finally:
        _ACTIVE_A2_REHEARSAL_CONTRACT = None


def build_a2_nonauthority_rehearsal_day(
    snapshot_prices: bytes | pd.DataFrame,
    suffix_prices: Sequence[pd.DataFrame],
    *,
    rehearsal_contract: _A2RehearsalContract,
    input_kind: str,
    target_session: Any,
    runtime_lock_verified_at: Any,
    month_source_sealed_at: Any,
    fit_started_at: Any,
    fit_completed_at: Any,
    score_generated_at: Any,
    source_manifest_sha256: str,
    source_set_sha256: str,
    parsed_shard_set_sha256: str,
    reuse_fold_token: _A2RehearsalFoldToken | None = None,
) -> tuple[dict[str, Any], _A2RehearsalFoldToken]:
    """Run the bound preactivation timing seam without creating authority.

    The full31 reference and the two compact cases share the production panel,
    numeric-fold, bundle-validation, and score paths.  Compact cases additionally
    exercise the exact compact CSV decoder plus suffix projection/merge.  No path
    or output argument exists, and the returned envelope deliberately differs
    from every production manifest schema.  The fold token is an in-memory-only
    Python object so the intramonth proxy can validate and reuse the boundary fit
    without serialising a second authority channel.
    """

    capability = _require_active_a2_rehearsal_contract(rehearsal_contract)
    if input_kind not in A2_REHEARSAL_INPUT_KINDS:
        raise V18Error("A2 nonauthority rehearsal input kind changed")
    target = _date(target_session, "A2 rehearsal target session")
    if str(target.date()) != capability.target_session:
        raise V18Error("A2 nonauthority rehearsal target is not the fixed synthetic day")
    verified = _timestamp(
        runtime_lock_verified_at, "A2 rehearsal runtime_lock_verified_at"
    )
    month_sealed = _timestamp(
        month_source_sealed_at, "A2 rehearsal month_source_sealed_at"
    )
    fit_started = _timestamp(fit_started_at, "A2 rehearsal fit_started_at")
    fit_completed = _timestamp(fit_completed_at, "A2 rehearsal fit_completed_at")
    score_generated = _timestamp(
        score_generated_at, "A2 rehearsal score_generated_at"
    )
    cutoff = datetime.combine(target.date(), CUTOFF_TIME, TOKYO)
    if not (
        verified <= month_sealed <= fit_started <= fit_completed <= score_generated
        < cutoff
    ):
        raise V18Error("A2 rehearsal replay timestamps are not causal/precutoff")
    source_manifest_hash = _require_sha(
        source_manifest_sha256, "A2 rehearsal source manifest SHA"
    )
    source_set_hash = _require_sha(source_set_sha256, "A2 rehearsal source set SHA")
    shard_set_hash = _require_sha(
        parsed_shard_set_sha256, "A2 rehearsal parsed shard set SHA"
    )

    suffix_rows = 0
    if input_kind == "full31_reference":
        if not isinstance(snapshot_prices, pd.DataFrame):
            raise V18Error("A2 full31 rehearsal snapshot must be a DataFrame")
        frames = [
            _coerce_jsonl_frame(
                snapshot_prices,
                PARSED_PRICE_COLUMNS,
                label="A2 rehearsal full31 reference prefix",
            )
        ]
        for index, frame in enumerate(suffix_prices):
            frames.append(
                _coerce_jsonl_frame(
                    frame,
                    PARSED_PRICE_COLUMNS,
                    label=f"A2 rehearsal full31 suffix {index}",
                )
            )
            suffix_rows += len(frames[-1])
        try:
            prices = merge_daily_prices(frames)
        except Exception as exc:
            raise V18Error(f"A2 rehearsal full31 merge failed: {exc}") from exc
        prices = _coerce_jsonl_frame(
            prices, PARSED_PRICE_COLUMNS, label="A2 rehearsal full31 merged prefix"
        )
        snapshot_rows = len(frames[0])
    else:
        if not isinstance(snapshot_prices, bytes):
            raise V18Error("A2 compact rehearsal snapshot must be canonical bytes")
        snapshot_frame = decode_canonical_model_price_csv(
            snapshot_prices, label="A2 rehearsal compact snapshot"
        )
        expected_snapshot_latest = (
            pd.Timestamp("2026-07-31")
            if input_kind == "month_boundary_compact"
            else pd.Timestamp("2026-06-30")
        )
        if pd.to_datetime(snapshot_frame["date"]).max().normalize() != (
            expected_snapshot_latest
        ):
            raise V18Error("A2 compact rehearsal snapshot cutoff changed")
        projected_suffix: list[pd.DataFrame] = []
        for index, frame in enumerate(suffix_prices):
            projected = _coerce_model_price_frame(
                frame, label=f"A2 rehearsal compact suffix {index}"
            )
            projected_suffix.append(projected)
            suffix_rows += len(projected)
        if not projected_suffix:
            raise V18Error("A2 compact rehearsal requires a nonempty suffix")
        prices = _coerce_model_price_frame(
            pd.concat([snapshot_frame, *projected_suffix], ignore_index=True),
            label="A2 rehearsal compact snapshot plus suffix",
        )
        snapshot_rows = len(snapshot_frame)

    latest = pd.to_datetime(prices["date"], errors="coerce").max().normalize()
    if str(latest.date()) != capability.latest_required_source_session:
        raise V18Error("A2 rehearsal merged prefix is not exact D-1")
    compact_projection = _coerce_model_price_frame(
        prices, label="A2 rehearsal model-price projection"
    )
    compact_payload = canonical_model_price_csv_bytes(
        compact_projection, label="A2 rehearsal model-price projection"
    )
    compact_payload_sha256 = hashlib.sha256(compact_payload).hexdigest()
    exact_prefix_identity = {
        "schema_version": 1,
        "target_session": str(target.date()),
        "latest_required_source_session": str(latest.date()),
        "model_price_row_count": int(len(compact_projection)),
        "model_price_csv_byte_count": len(compact_payload),
        "model_price_csv_sha256": compact_payload_sha256,
        "source_manifest_sha256": source_manifest_hash,
        "source_set_sha256": source_set_hash,
        "parsed_shard_set_sha256": shard_set_hash,
    }
    retained_proof: dict[str, Any] | None = None
    if reuse_fold_token is None:
        if not decode_canonical_model_price_csv(
            compact_payload, label="A2 rehearsal model-price round trip"
        ).equals(compact_projection):
            raise V18Error("A2 rehearsal compact projection does not round-trip")
    else:
        if not isinstance(reuse_fold_token, _A2RehearsalFoldToken):
            raise V18Error("A2 rehearsal fold reuse token type changed")
        try:
            retained_proof = json.loads(reuse_fold_token.exact_prefix_proof_json)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise V18Error("A2 rehearsal exact-prefix proof JSON is invalid") from exc
        expected_proof_fields = {
            *exact_prefix_identity,
            "model_price_semantic_sha256",
            "g0_panel_exact_digest",
            "g0_training_row_count",
            "g0_training_panel_semantic_sha256",
        }
        if (
            not isinstance(retained_proof, dict)
            or set(retained_proof) != expected_proof_fields
            or canonical_json_bytes(retained_proof).decode("utf-8")
            != reuse_fold_token.exact_prefix_proof_json
            or canonical_json_sha256(retained_proof)
            != reuse_fold_token.exact_prefix_proof_sha256
        ):
            raise V18Error("A2 rehearsal exact-prefix proof binding changed")
        if any(
            retained_proof[field] != expected
            for field, expected in exact_prefix_identity.items()
        ):
            raise V18Error(
                "A2 rehearsal reused proof differs from canonical full-prefix bytes"
            )

    # Exactly one production panel build occurs in every timed/reference call.
    panel = build_forward_c00_panel(prices, target)
    target_rows = _coerce_jsonl_frame(
        panel.loc[pd.to_datetime(panel["date"]).eq(target)],
        G0_PANEL_COLUMNS,
        label="A2 rehearsal target slice",
    )
    target_payload = canonical_frame_jsonl_bytes(
        target_rows, G0_PANEL_COLUMNS, label="A2 rehearsal target slice"
    )
    if not decode_canonical_frame_jsonl(
        target_payload, G0_PANEL_COLUMNS, label="A2 rehearsal target round trip"
    ).equals(target_rows):
        raise V18Error("A2 rehearsal target cache does not round-trip")
    if target_rows["oc_return_pct"].notna().any():
        raise V18Error("A2 rehearsal target cache contains an outcome")

    month = target.to_period("M")
    training_panel = panel.loc[
        pd.to_datetime(panel["date"]).lt(month.start_time.normalize()),
        list(G0_PANEL_COLUMNS),
    ]
    if reuse_fold_token is None:
        model_price_semantic = model_price_semantic_sha256(compact_projection)
        panel_digest = _exact_g0_frame_digest(panel)
        rehearsal_training_binding = semantic_frame_sha256(
            training_panel, G0_PANEL_COLUMNS
        )
        rehearsal_training_row_count = int(len(training_panel))
        exact_prefix_proof = {
            **exact_prefix_identity,
            "model_price_semantic_sha256": model_price_semantic,
            "g0_panel_exact_digest": panel_digest,
            "g0_training_row_count": rehearsal_training_row_count,
            "g0_training_panel_semantic_sha256": rehearsal_training_binding,
        }
        exact_prefix_proof_json = canonical_json_bytes(exact_prefix_proof).decode(
            "utf-8"
        )
        exact_prefix_proof_sha256 = canonical_json_sha256(exact_prefix_proof)
        month_source: dict[str, Any] = {
            field: None for field in MONTH_SOURCE_MANIFEST_FIELDS
        }
        month_source.update(
            {
                "schema_version": 1,
                "target_month": str(month),
                "first_counted_session": str(target.date()),
                "seal_session": str(target.date()),
                "activation_observed_at": verified,
                "latest_required_source_session": str(latest.date()),
                "created_at": month_sealed,
                "sealed_at": month_sealed,
                "runtime_lock_sha256": capability.runtime_lock_sha256,
                "runtime_lock_verified_at": verified,
                "source_set_sha256": source_set_hash,
                "parsed_shard_set_sha256": shard_set_hash,
                "parsed_row_count": int(len(prices)),
                "model_price_full_prefix_semantic_sha256": model_price_semantic,
                "g0_training_panel_semantic_sha256": rehearsal_training_binding,
                "g0_training_row_count": rehearsal_training_row_count,
                "protocol_sha256": capability.protocol_sha256,
                "runner_sha256": capability.runner_sha256,
                "parser_sha256": JPX_PARSER_SHA256,
                "canonical_json_contract": CANONICAL_JSON_CONTRACT,
            }
        )
        month_source["month_source_manifest_sha256"] = canonical_json_sha256(
            month_source, exclude_fields={"month_source_manifest_sha256"}
        )
        _, fold, bundle = _build_fold(
            panel,
            month,
            fit_started_at=fit_started,
            fit_completed_at=fit_completed,
            bundle_created_at=fit_completed,
            sealed_at=fit_completed,
            runtime_lock_verified_at=verified,
            terminal_replay=True,
            first_counted_session_value=target,
            activation_observed_at=verified,
            month_source_manifest=month_source,
            _a2_rehearsal_contract=capability,
        )
        token = _A2RehearsalFoldToken(
            dict(fold),
            dict(bundle),
            dict(month_source),
            exact_prefix_proof_json,
            exact_prefix_proof_sha256,
        )
    else:
        assert retained_proof is not None
        model_price_semantic = _require_sha(
            retained_proof["model_price_semantic_sha256"],
            "A2 rehearsal retained model-price semantic SHA",
        )
        panel_digest = retained_proof["g0_panel_exact_digest"]
        if not isinstance(panel_digest, dict):
            raise V18Error("A2 rehearsal retained G0 digest is invalid")
        rehearsal_training_binding = _require_sha(
            retained_proof["g0_training_panel_semantic_sha256"],
            "A2 rehearsal retained training semantic SHA",
        )
        rehearsal_training_row_count = int(
            retained_proof["g0_training_row_count"]
        )
        if rehearsal_training_row_count <= 0 or len(training_panel) != (
            rehearsal_training_row_count
        ):
            raise V18Error("A2 rehearsal retained training row count changed")
        fold = dict(reuse_fold_token.fold_manifest)
        bundle = dict(reuse_fold_token.model_bundle)
        token_month_source = dict(reuse_fold_token.month_source_manifest)
        exact_month_bindings = {
            "target_month": str(month),
            "latest_required_source_session": str(latest.date()),
            "source_set_sha256": source_set_hash,
            "parsed_shard_set_sha256": shard_set_hash,
            "parsed_row_count": int(len(prices)),
            "model_price_full_prefix_semantic_sha256": model_price_semantic,
            "g0_training_panel_semantic_sha256": rehearsal_training_binding,
            "g0_training_row_count": rehearsal_training_row_count,
        }
        if (
            set(token_month_source) != set(MONTH_SOURCE_MANIFEST_FIELDS)
            or token_month_source["month_source_manifest_sha256"]
            != canonical_json_sha256(
                token_month_source,
                exclude_fields={"month_source_manifest_sha256"},
            )
            or any(
                token_month_source[field] != expected
                for field, expected in exact_month_bindings.items()
            )
        ):
            raise V18Error("A2 rehearsal reused fold month-source binding changed")
        validate_fold_manifest(
            fold,
            bundle,
            _a2_rehearsal_contract=capability,
            _a2_rehearsal_month_source=token_month_source,
        )
        training = v17._candidate_training(
            panel, end=month.start_time.normalize() - pd.Timedelta(days=1)
        )
        identity = training.loc[:, ["date", "code"]].sort_values(
            ["date", "code"], kind="stable"
        )
        current_hashes = {
            "training_row_identity_sha256": _frame_sha(identity, ("date", "code")),
            "training_target_sha256": _numeric_sha(
                training["_daily_rank_target"].to_numpy(dtype=float)
            ),
            "feature_matrix_sha256": _numeric_sha(
                training.loc[:, list(G0_FEATURES)].to_numpy(dtype=float)
            ),
            "training_source_set_sha256": source_set_hash,
            "training_parsed_shard_set_sha256": shard_set_hash,
            "training_g0_panel_semantic_sha256": rehearsal_training_binding,
        }
        if fold.get("target_month") != str(month) or any(
            fold.get(field) != expected for field, expected in current_hashes.items()
        ):
            raise V18Error("A2 rehearsal reused fold differs from current prefix")
        token = reuse_fold_token
        month_source = token_month_source

    scores, replayed_fold, replayed_bundle = freeze_c00_top2(
        panel,
        target,
        score_generated_at=score_generated,
        runtime_lock_verified_at=verified,
        terminal_replay=True,
        source_manifest_sha256=source_manifest_hash,
        first_counted_session_value=target,
        activation_observed_at=verified,
        return_bundle=True,
        model_bundle=bundle,
        fold_manifest=fold,
        month_source_manifest=month_source,
        _a2_rehearsal_contract=capability,
    )
    if replayed_fold != fold or replayed_bundle != bundle:
        raise V18Error("A2 rehearsal score path changed the reused fold")
    score_payload = validate_score_rows(scores).to_csv(
        index=False, lineterminator="\n"
    ).encode("utf-8")
    comparison: dict[str, Any] = {
        "target_session": str(target.date()),
        "latest_required_source_session": str(latest.date()),
        "model_price_row_count": int(len(compact_projection)),
        "model_price_semantic_sha256": model_price_semantic,
        "model_price_csv_sha256": compact_payload_sha256,
        "source_manifest_sha256": source_manifest_hash,
        "source_set_sha256": source_set_hash,
        "parsed_shard_set_sha256": shard_set_hash,
        "g0_panel_exact_digest": panel_digest,
        "g0_training_row_count": rehearsal_training_row_count,
        "g0_training_panel_semantic_sha256": rehearsal_training_binding,
        "target_cache_byte_count": len(target_payload),
        "target_cache_sha256": hashlib.sha256(target_payload).hexdigest(),
        "target_cache_semantic_sha256": semantic_frame_sha256(
            target_rows, G0_PANEL_COLUMNS
        ),
        "fold_manifest_file_sha256": hashlib.sha256(
            _json_file_bytes(fold)
        ).hexdigest(),
        "fold_manifest_sha256": fold["fold_manifest_sha256"],
        "fold_model_bundle_file_sha256": hashlib.sha256(
            _json_file_bytes(bundle)
        ).hexdigest(),
        "fold_model_bundle_sha256": bundle["fold_model_bundle_sha256"],
        "score_file_sha256": hashlib.sha256(score_payload).hexdigest(),
        "score_semantic_sha256": semantic_score_hash(scores),
        "top2_code_score_ieee_sha256": canonical_json_sha256(
            [
                {
                    "source_rank": int(row.source_rank),
                    "code": str(row.code),
                    "model_score_ieee_le_sha256": hashlib.sha256(
                        struct.pack("<d", float(row.model_score))
                    ).hexdigest(),
                }
                for row in scores.itertuples(index=False)
            ]
        ),
        "build_forward_c00_panel_call_count": 1,
    }
    envelope: dict[str, Any] = {
        "schema_version": 1,
        "scope": "nonauthority_rehearsal_only",
        "input_kind": input_kind,
        "snapshot_row_count": int(snapshot_rows),
        "suffix_row_count": int(suffix_rows),
        "comparison": comparison,
        "production_authority": False,
        "canonical_artifact_written": False,
    }
    envelope["envelope_sha256"] = canonical_json_sha256(
        envelope, exclude_fields={"envelope_sha256"}
    )
    return envelope, token


def validate_score_ledger(
    scores: pd.DataFrame,
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    if not isinstance(scores, pd.DataFrame):
        raise V18Error("score ledger must be a DataFrame")
    if scores.empty:
        if not allow_empty:
            raise V18Error("score ledger must be nonempty")
        if scores.columns.tolist() != list(SCORE_FIELDS):
            raise V18Error("empty score ledger must retain the exact registered header")
        return scores.copy()
    groups: list[pd.DataFrame] = []
    parsed_sessions = pd.to_datetime(scores["session_date"], errors="coerce")
    if parsed_sessions.isna().any():
        raise V18Error("score ledger contains an invalid session")
    for _, group in scores.assign(_session=parsed_sessions).groupby(
        "_session", sort=True
    ):
        groups.append(validate_score_rows(group.drop(columns="_session")))
    combined = pd.concat(groups, ignore_index=True)
    sessions = pd.to_datetime(combined["session_date"])
    if not sessions.is_monotonic_increasing:
        raise V18Error("score ledger sessions are not chronological")
    return combined


def semantic_score_ledger_hash(
    scores: pd.DataFrame,
    *,
    allow_empty: bool = False,
) -> str:
    return hashlib.sha256(
        validate_score_ledger(scores, allow_empty=allow_empty)
        .to_csv(index=False, lineterminator="\n")
        .encode()
    ).hexdigest()


def validate_score_session_authority(scores: pd.DataFrame) -> str:
    """Exact-compare the derived ledger with immutable per-session shards."""

    ledger = validate_score_ledger(scores, allow_empty=True)
    expected_groups = {
        str(session.date()): validate_score_rows(group.drop(columns="_session"))
        for session, group in ledger.assign(
            _session=pd.to_datetime(ledger["session_date"], errors="coerce")
        ).groupby("_session", sort=True)
    }
    if not SCORE_SESSION_DIR.is_dir() or SCORE_SESSION_DIR.is_symlink():
        raise V18Error("score-session authority directory is missing")
    directory_stat = os.stat(SCORE_SESSION_DIR, follow_symlinks=False)
    if (
        directory_stat.st_uid != os.geteuid()
        or stat.S_IMODE(directory_stat.st_mode) != 0o700
    ):
        raise V18Error("score-session authority directory is not private")
    entries = sorted(SCORE_SESSION_DIR.iterdir(), key=lambda item: item.name)
    expected_names = [f"{session}.csv" for session in expected_groups]
    for entry in entries:
        if re.fullmatch(r"(?:\d{4}-\d{2}-\d{2}\.csv|\.\d{4}-\d{2}-\d{2}\.csv\.staging)", entry.name) is None:
            raise V18Error("score-session authority has an unregistered entry")
    observed_final_names = [
        item.name for item in entries if not item.name.startswith(".")
    ]
    if observed_final_names != expected_names:
        raise V18Error("score-session authority has a missing/extra/non-plain file")
    bindings: list[dict[str, Any]] = []
    for file_name in expected_names:
        entry = SCORE_SESSION_DIR / file_name
        session = entry.stem
        payload = _read_local_authority_bytes(
            entry, label="terminal score-session shard"
        )
        expected = expected_groups[session].to_csv(
            index=False, lineterminator="\n"
        ).encode()
        if payload != expected:
            raise V18Error("score-session shard differs from deferred ledger")
        bindings.append(
            {
                "session_date": session,
                "byte_count": len(payload),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "semantic_sha256": semantic_score_hash(expected_groups[session]),
            }
        )
    derived = ledger.to_csv(index=False, lineterminator="\n").encode()
    if SCORE_OUTPUT.is_symlink() or _plain_file_bytes(
        SCORE_OUTPUT, label="derived canonical score ledger"
    ) != derived:
        raise V18Error("derived score ledger differs from session authority")
    return canonical_json_sha256(bindings)


def append_score_rows(scores: pd.DataFrame, path: str | Path = SCORE_OUTPUT) -> None:
    """Seal one immutable score shard, then rebuild the derived CSV view."""

    pair = validate_score_rows(scores)
    target = Path(path)
    canonical_target = Path(os.path.abspath(os.fspath(target)))
    canonical_score = Path(os.path.abspath(os.fspath(SCORE_OUTPUT)))
    shard_directory = (
        SCORE_SESSION_DIR
        if canonical_target == canonical_score
        else target.parent / f"{target.name}.sessions"
    )
    if canonical_target == canonical_score:
        if shard_directory.is_symlink() or not shard_directory.is_dir():
            raise V18Error("score-session authority readiness directory is missing")
        directory_stat = os.stat(shard_directory, follow_symlinks=False)
        if (
            directory_stat.st_uid != os.geteuid()
            or stat.S_IMODE(directory_stat.st_mode) != 0o700
        ):
            raise V18Error("score-session authority readiness mode changed")
    else:
        shard_directory.mkdir(parents=True, exist_ok=True)
    session = str(pair["session_date"].iloc[0])
    shard_path = shard_directory / f"{session}.csv"
    shard_payload = pair.to_csv(index=False, lineterminator="\n").encode()
    _atomic_local_bytes_once(shard_path, shard_payload)

    entries = sorted(shard_directory.iterdir(), key=lambda item: item.name)
    for entry in entries:
        if re.fullmatch(r"(?:\d{4}-\d{2}-\d{2}\.csv|\.\d{4}-\d{2}-\d{2}\.csv\.staging)", entry.name) is None:
            raise V18Error("score-session authority contains an unregistered name")
    final_entries = [item for item in entries if not item.name.startswith(".")]
    frames: list[pd.DataFrame] = []
    observed_sessions: list[str] = []
    for entry in final_entries:
        match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\.csv", entry.name)
        if match is None:
            raise V18Error("score-session authority contains an unregistered name")
        retained = (
            _read_local_authority_bytes(entry, label="score-session shard")
            if canonical_target == canonical_score
            else _plain_file_bytes(entry, label="score-session shard")
        )
        try:
            frame = pd.read_csv(
                io.BytesIO(retained),
                dtype={"code": "string"},
                float_precision="round_trip",
            )
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise V18Error("score-session shard is not canonical CSV") from exc
        validated = validate_score_rows(frame)
        if validated["session_date"].iloc[0] != match.group(1):
            raise V18Error("score-session filename/date binding changed")
        if retained != validated.to_csv(index=False, lineterminator="\n").encode():
            raise V18Error("score-session shard exact bytes changed")
        observed_sessions.append(match.group(1))
        frames.append(validated)
    if observed_sessions != sorted(observed_sessions) or len(observed_sessions) != len(
        set(observed_sessions)
    ):
        raise V18Error("score-session authority chronology changed")
    ledger = validate_score_ledger(pd.concat(frames, ignore_index=True))
    expected_ledger = ledger.to_csv(index=False, lineterminator="\n").encode()
    _atomic_local_derived_bytes(target, expected_ledger)


def load_registered_calendar(path: str | Path = CALENDAR) -> pd.DatetimeIndex:
    """Load and byte-validate the preregistered TSE session denominator."""

    payload = _plain_file_bytes(path, label="registered TSE calendar")
    if hashlib.sha256(payload).hexdigest() != CALENDAR_SHA256:
        raise V18Error("registered TSE calendar SHA-256 mismatch")
    frame = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
    expected_columns = ["session_date", "market", "source_url", "source_retrieved_at"]
    if frame.columns.tolist() != expected_columns or len(frame) != 343:
        raise V18Error("registered TSE calendar schema/count changed")
    sessions = pd.to_datetime(frame["session_date"], errors="coerce")
    if (
        sessions.isna().any()
        or not sessions.is_monotonic_increasing
        or sessions.duplicated().any()
    ):
        raise V18Error("registered TSE calendar dates are invalid")
    if (
        frame["market"].ne("TSE").any()
        or frame["source_url"]
        .ne("https://www.jpx.co.jp/english/corporate/about-jpx/calendar/index.html")
        .any()
        or frame["source_retrieved_at"].ne("2026-08-04").any()
    ):
        raise V18Error("registered TSE calendar source fields changed")
    result = pd.DatetimeIndex(sessions)
    if result[0] != pd.Timestamp("2026-08-05") or result[-1] != pd.Timestamp(
        "2027-12-30"
    ):
        raise V18Error("registered TSE calendar bounds changed")
    return result


def first_counted_session(
    *,
    workflow_run_updated_at: Any,
    workflow_run_observed_at: Any,
    calendar: pd.DatetimeIndex | None = None,
    not_before_session: str = "2026-08-06",
) -> pd.Timestamp:
    """Derive the fixed start from C's successful workflow server timestamp."""

    scheduled = load_registered_calendar() if calendar is None else calendar
    updated = _timestamp(
        workflow_run_updated_at, "activation_receipt_workflow_run_updated_at"
    )
    observed = _timestamp(
        workflow_run_observed_at, "activation_receipt_workflow_run_observed_at"
    )
    if observed < updated:
        raise V18Error("receipt workflow observation predates successful completion")
    not_before = _date(not_before_session, "not_before_session")
    for session in scheduled[scheduled >= not_before]:
        cutoff = _cutoff(session)
        if cutoff > updated:
            if observed >= cutoff:
                raise V18Error(
                    "receipt workflow observation missed the fixed first cutoff"
                )
            return pd.Timestamp(session)
    raise V18Error("registered calendar has no eligible first counted session")


def deterministic_terminal_session(
    first: Any,
    calendar: pd.DatetimeIndex | None = None,
) -> pd.Timestamp:
    """Return the first qualifying calendar month-end, never session 120 itself."""

    scheduled = load_registered_calendar() if calendar is None else calendar
    first_session = _date(first, "first_counted_session")
    eligible = scheduled[scheduled >= first_session]
    if len(eligible) == 0 or eligible[0] != first_session:
        raise V18Error("first counted session is outside the registered calendar")
    periods = eligible.to_period("M")
    for month in periods.unique():
        through = eligible[periods <= month]
        if (
            len(through) >= MIN_FORWARD_SESSIONS
            and through.to_period("M").nunique() >= MIN_FORWARD_MONTHS
        ):
            return pd.Timestamp(through[-1])
    raise V18Error("registered calendar does not cover the deterministic terminal")


def _validate_git_commit(sha: Any, url: Any, name: str) -> str:
    token = str(sha)
    if GIT_SHA_RE.fullmatch(token) is None:
        raise V18Error(f"{name} must be a Git commit SHA")
    expected = f"https://github.com/rokuroku-066/TSE-Session-Ranker/commit/{token}"
    if str(url) != expected:
        raise V18Error(f"{name} URL does not bind the commit")
    return token


def _git(*arguments: str, binary: bool = False) -> str | bytes:
    executable = _LOCKED_GIT_EXECUTABLE or _validate_git_executable(
        read_json(RUNTIME_LOCK)
    )
    try:
        completed = subprocess.run(
            [str(executable), *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=not binary,
            env=_git_environment(),
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise V18Error(f"local Git evidence check failed: {' '.join(arguments)}") from exc
    return completed.stdout


def _git_require_commit(sha: Any, name: str) -> str:
    token = str(sha)
    if GIT_SHA_RE.fullmatch(token) is None:
        raise V18Error(f"{name} is not a canonical Git commit SHA")
    _git("cat-file", "-e", f"{token}^{{commit}}")
    observed = str(_git("rev-parse", f"{token}^{{commit}}")).strip()
    if observed != token:
        raise V18Error(f"{name} does not resolve to its claimed Git commit")
    return observed


def _git_file_bytes(commit: str, path: str) -> bytes:
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise V18Error("Git evidence path is unsafe")
    return bytes(_git("show", f"{commit}:{path}", binary=True))


def _git_is_ancestor(ancestor: str, descendant: str, name: str) -> None:
    if ancestor == descendant:
        raise V18Error(f"{name} commits must be strictly ordered")
    try:
        executable = _LOCKED_GIT_EXECUTABLE or _validate_git_executable(
            read_json(RUNTIME_LOCK)
        )
        subprocess.run(
            [str(executable), "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=ROOT,
            check=True,
            capture_output=True,
            env=_git_environment(),
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise V18Error(f"{name} Git ancestry is not satisfied") from exc


def _git_commit_timestamp(commit: str) -> datetime:
    raw = str(_git("show", "-s", "--format=%cI", commit)).strip()
    return _timestamp(raw, "Git commit committed_at")


def _require_sole_git_parent(commit: str, parent: str, name: str) -> None:
    parents = str(_git("show", "-s", "--format=%P", commit)).strip().split()
    if parents != [parent]:
        raise V18Error(f"{name} must have its registered predecessor as sole parent")


def _validate_local_git_projection(
    commit: str,
    observation: Mapping[str, Any],
    *,
    name: str,
) -> None:
    value = validate_github_observation(observation)
    projection = value["canonical_projection"]
    parents = str(_git("show", "-s", "--format=%P", commit)).strip().split()
    if (
        projection["commit_sha"] != commit
        or sorted(parents) != projection["parent_shas"]
        or _git_commit_timestamp(commit)
        != _timestamp(projection["committer_date"], f"{name} committer_date")
    ):
        raise V18Error(f"local Git object differs from GitHub projection: {name}")


def _git_changed_paths(older: str, newer: str) -> set[str]:
    output = str(_git("diff", "--name-only", "--no-renames", older, newer))
    return {line for line in output.splitlines() if line}


def _git_tree_paths(commit: str) -> set[str]:
    output = str(_git("ls-tree", "-r", "--name-only", commit))
    paths = {line for line in output.splitlines() if line}
    if any(Path(path).is_absolute() or ".." in Path(path).parts for path in paths):
        raise V18Error("Git tree contains an unsafe path")
    return paths


def _github_api_transport(
    endpoint: str,
    *,
    return_observation: bool = False,
    method: str = "GET",
    request_body: Mapping[str, Any] | None = None,
    expected_status: int = 200,
    require_evidence_headers: bool = True,
    require_object: bool = True,
) -> Any:
    """Request GitHub JSON with the sole locked stdlib/CA transport."""

    if not _STRICT_RUNTIME_ACTIVE or _LOCKED_TLS_CAFILE is None:
        raise V18Error("GitHub API transport requires strict runtime/CA validation")
    runtime_lock, _ = validate_runtime_lock(strict_environment=False)
    request_ca = _validate_tls_ca_trust(
        runtime_lock["runtime"]["python"]["ssl"]["ca_trust"]
    )
    if request_ca != _LOCKED_TLS_CAFILE:
        raise V18Error("GitHub API CA authority changed after runtime validation")
    if not endpoint.startswith("/repos/") or "://" in endpoint or ".." in endpoint.split("/"):
        raise V18Error("GitHub API endpoint is unsafe")
    if method not in {"GET", "POST", "PATCH"}:
        raise V18Error("GitHub API method is not registered")
    if (method == "GET") != (request_body is None):
        raise V18Error("GitHub API request body/method combination is invalid")
    if isinstance(expected_status, bool) or expected_status not in {200, 201}:
        raise V18Error("GitHub API expected status is not registered")
    url = f"{GITHUB_API_BASE}{endpoint}"
    headers = {
        "Accept": GITHUB_API_ACCEPT,
        "Accept-Encoding": "identity",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "User-Agent": GITHUB_API_USER_AGENT,
    }
    body_bytes = None if request_body is None else canonical_json_bytes(request_body)
    if body_bytes is not None:
        headers["Content-Type"] = "application/json"
    token = os.environ.get("GITHUB_TOKEN")
    if method != "GET" and not token:
        raise V18Error("GitHub mutation requires the ephemeral GITHUB_TOKEN")
    if token:
        # The optional token affects only the request's authorization.  It is
        # never returned, hashed, written, interpolated into an error, or
        # included in an observation artifact.
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url,
        data=body_bytes,
        headers=headers,
        method=method,
    )
    token = None

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req: Any, fp: Any, code: Any, msg: Any, hdrs: Any, newurl: Any) -> None:
            return None

    try:
        tls_context = ssl.create_default_context(cafile=str(request_ca))
    except (OSError, ssl.SSLError) as exc:
        raise V18Error("registered GitHub TLS context creation failed") from exc
    if not tls_context.check_hostname or tls_context.verify_mode != ssl.CERT_REQUIRED:
        raise V18Error("registered GitHub TLS verification policy changed")
    if _validate_tls_ca_trust(
        runtime_lock["runtime"]["python"]["ssl"]["ca_trust"]
    ) != request_ca:
        raise V18Error("GitHub API CA trust-store changed during context creation")
    opener = urllib.request.build_opener(
        NoRedirect(),
        urllib.request.HTTPSHandler(context=tls_context),
    )
    try:
        with opener.open(request, timeout=30.0) as response:
            status = int(response.status)
            if urlparse(response.geturl()).scheme != "https" or urlparse(
                response.geturl()
            ).hostname != "api.github.com":
                raise V18Error("GitHub API response escaped the registered host")
            raw = response.read()
            content_type = str(response.headers.get("Content-Type", ""))
            content_encoding = response.headers.get("Content-Encoding")
            response_date = response.headers.get("Date")
            etag = response.headers.get("ETag")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise V18Error(f"GitHub commit/API observation failed: {endpoint}") from exc
    observed_at = datetime.now(TOKYO)
    if status != expected_status:
        raise V18Error("GitHub API response status is invalid")
    if not content_type.lower().split(";", 1)[0].strip().endswith("json"):
        raise V18Error("GitHub API response content type is not JSON")
    if content_encoding not in (None, "", "identity"):
        raise V18Error("GitHub API response is not identity encoded")
    if require_evidence_headers and (not response_date or not etag):
        raise V18Error("GitHub API response lacks registered evidence headers")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V18Error("GitHub commit/API observation returned invalid JSON") from exc
    if require_object and not isinstance(value, dict):
        raise V18Error("GitHub commit/API observation is not an object")
    response_headers = {
        "status": status,
        "content_type": content_type,
        "date": response_date,
        "etag": etag,
    }
    observation = {
        "http_status": status,
        "content_type": content_type,
        "http_date": response_date,
        "etag": etag,
        "response_body_sha256": hashlib.sha256(raw).hexdigest(),
        "response_headers_sha256": canonical_json_sha256(response_headers),
        "retrieved_at": observed_at,
    }
    return (value, observation) if return_observation else value


def _github_api(
    endpoint: str,
    *,
    return_observation: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Run one locked network phase bracketed by exhaustive module checks."""

    _validate_startup_and_module_closure(phase="GitHub network phase entry")
    try:
        return _github_api_transport(
            endpoint,
            return_observation=return_observation,
        )
    finally:
        _validate_startup_and_module_closure(phase="GitHub network phase exit")


def _github_git_data_api(
    endpoint: str,
    *,
    method: str = "GET",
    request_body: Mapping[str, Any] | None = None,
    expected_status: int = 200,
) -> dict[str, Any]:
    """Run one registered Git Data API operation without transport evidence."""

    _validate_startup_and_module_closure(phase="GitHub Git Data phase entry")
    try:
        value = _github_api_transport(
            endpoint,
            method=method,
            request_body=request_body,
            expected_status=expected_status,
            require_evidence_headers=False,
        )
        if not isinstance(value, dict):  # pragma: no cover - transport guards
            raise V18Error("GitHub Git Data response is not an object")
        return value
    finally:
        _validate_startup_and_module_closure(phase="GitHub Git Data phase exit")


def _github_json_list(endpoint: str) -> list[dict[str, Any]]:
    """Read one registered terminal-history list page with the locked transport."""

    _validate_startup_and_module_closure(phase="GitHub list phase entry")
    try:
        value = _github_api_transport(
            endpoint,
            require_evidence_headers=False,
            require_object=False,
        )
        if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
            raise V18Error("GitHub terminal history page is not an object array")
        return [dict(item) for item in value]
    finally:
        _validate_startup_and_module_closure(phase="GitHub list phase exit")


def _github_observation(
    *,
    commit_sha: str | None = None,
    branch: str | None = None,
) -> dict[str, Any]:
    if (commit_sha is None) == (branch is None):
        raise V18Error("GitHub observation requires exactly one subject")
    repository = "rokuroku-066/TSE-Session-Ranker"
    if commit_sha is not None:
        if re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None:
            raise V18Error("GitHub observation commit SHA is not full SHA-1")
        kind = "commit"
        endpoint = f"/repos/{repository}/commits/{commit_sha}"
    else:
        kind = "branch_tip"
        endpoint = f"/repos/{repository}/commits/{quote(str(branch), safe='')}"
    body, transport = _github_api(endpoint, return_observation=True)
    projection = {
        "commit_sha": body.get("sha"),
        "html_url": body.get("html_url"),
        "committer_date": body.get("commit", {}).get("committer", {}).get("date"),
        "parent_shas": sorted(
            {str(item.get("sha")) for item in body.get("parents", [])}
        ),
    }
    value = {
        "schema_version": 1,
        "observation_kind": kind,
        "endpoint": endpoint,
        "repository": repository,
        "branch": None if kind == "commit" else str(branch),
        "requested_commit_sha": commit_sha,
        "http_status": transport["http_status"],
        "content_type": transport["content_type"],
        "http_date": transport["http_date"],
        "etag": transport["etag"],
        "response_body_sha256": transport["response_body_sha256"],
        "response_headers_sha256": transport["response_headers_sha256"],
        "canonical_projection": projection,
        "retrieved_at": transport["retrieved_at"],
        "canonical_json_contract": CANONICAL_JSON_CONTRACT,
    }
    value["observation_sha256"] = canonical_json_sha256(
        value, exclude_fields={"observation_sha256"}
    )
    validate_github_observation(value)
    return value


def _select_github_workflow_run_id(
    expected_head_sha: str,
    *,
    require_terminal: bool = True,
    require_success: bool = True,
) -> int:
    """Select the immutable lowest attempt-one PR workflow run across all pages."""

    if GIT_SHA_RE.fullmatch(str(expected_head_sha)) is None:
        raise V18Error("workflow selection head SHA is invalid")
    repository = "rokuroku-066/TSE-Session-Ranker"
    base = (
        f"/repos/{repository}/actions/runs?event=pull_request"
        f"&head_sha={expected_head_sha}&per_page=100"
    )
    collected: list[dict[str, Any]] = []
    total_count: int | None = None
    page = 1
    while True:
        endpoint = base if page == 1 else f"{base}&page={page}"
        body = _github_api(endpoint)
        observed_total = body.get("total_count")
        runs = body.get("workflow_runs")
        if (
            isinstance(observed_total, bool)
            or not isinstance(observed_total, int)
            or observed_total < 0
            or not isinstance(runs, list)
            or any(not isinstance(item, Mapping) for item in runs)
        ):
            raise V18Error("GitHub workflow pagination response is invalid")
        if total_count is None:
            total_count = observed_total
        elif observed_total != total_count:
            raise V18Error("GitHub workflow total_count changed during pagination")
        collected.extend(dict(item) for item in runs)
        if len(collected) >= total_count:
            break
        if len(runs) != 100 or page >= 10_000:
            raise V18Error("GitHub workflow pagination ended inconsistently")
        page += 1
    assert total_count is not None
    if len(collected) != total_count:
        raise V18Error("GitHub workflow pagination count differs")
    run_ids = [item.get("id") for item in collected]
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in run_ids
    ) or len(run_ids) != len(set(run_ids)):
        raise V18Error("GitHub workflow pagination has invalid/duplicate run IDs")
    matching = [
        item
        for item in collected
        if item.get("name") == "tests"
        and item.get("path") == WORKFLOW_PATH
        and item.get("event") == "pull_request"
        and item.get("head_sha") == expected_head_sha
        and item.get("run_attempt") == 1
    ]
    if not matching:
        raise V18Error("GitHub workflow attempt-one run is not yet available")
    selected = min(matching, key=lambda item: int(item["id"]))
    if require_terminal and selected.get("status") != "completed":
        raise V18Error("first GitHub workflow run has not completed")
    if require_success and selected.get("conclusion") != "success":
        raise V18Error("first GitHub workflow run did not succeed")
    return int(selected["id"])


def _github_workflow_run_observation(
    expected_head_sha: str,
    *,
    checkpoint: bool = False,
) -> dict[str, Any]:
    repository = "rokuroku-066/TSE-Session-Ranker"
    run_id = _select_github_workflow_run_id(
        expected_head_sha,
        require_success=not checkpoint,
    )
    endpoint = f"/repos/{repository}/actions/runs/{run_id}"
    body, transport = _github_api(endpoint, return_observation=True)
    projection = {
        "run_id": body.get("id"),
        "workflow_id": body.get("workflow_id"),
        "workflow_name": body.get("name"),
        "workflow_path": body.get("path"),
        "event": body.get("event"),
        "head_sha": body.get("head_sha"),
        "run_attempt": body.get("run_attempt"),
        "status": body.get("status"),
        "conclusion": body.get("conclusion"),
        "created_at": body.get("created_at"),
        "run_started_at": body.get("run_started_at"),
        "updated_at": body.get("updated_at"),
        "html_url": body.get("html_url"),
    }
    value = {
        "schema_version": 1,
        "endpoint": endpoint,
        "repository": repository,
        "run_id": run_id,
        "expected_head_sha": expected_head_sha,
        "http_status": transport["http_status"],
        "content_type": transport["content_type"],
        "http_date": transport["http_date"],
        "etag": transport["etag"],
        "response_body_sha256": transport["response_body_sha256"],
        "response_headers_sha256": transport["response_headers_sha256"],
        "canonical_projection": projection,
        "retrieved_at": transport["retrieved_at"],
        "canonical_json_contract": CANONICAL_JSON_CONTRACT,
    }
    value["observation_sha256"] = canonical_json_sha256(
        value, exclude_fields={"observation_sha256"}
    )
    validate_github_workflow_observation(
        value,
        expected_head_sha=expected_head_sha,
        checkpoint=checkpoint,
    )
    return value


def validate_github_workflow_observation(
    observation: Mapping[str, Any],
    *,
    expected_head_sha: str,
    checkpoint: bool = False,
) -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    contract = (
        protocol["daily_preopen_checkpoint_contract"][
            "checkpoint_workflow_run_observation_contract"
        ]
        if checkpoint
        else protocol["activation"]["github_workflow_run_observation_contract"]
    )
    value = dict(observation)
    if set(value) != set(GITHUB_WORKFLOW_OBSERVATION_FIELDS) or set(value) != set(
        contract["required_fields"]
    ):
        raise V18Error("GitHub workflow observation fields differ from protocol")
    for field, expected in contract["fixed_values"].items():
        if value.get(field) != expected:
            raise V18Error(f"GitHub workflow fixed field changed: {field}")
    if GIT_SHA_RE.fullmatch(str(expected_head_sha)) is None or value[
        "expected_head_sha"
    ] != expected_head_sha:
        raise V18Error("GitHub workflow expected head changed")
    run_id = value["run_id"]
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
        raise V18Error("GitHub workflow run ID is invalid")
    repository = str(value["repository"])
    if value["endpoint"] != f"/repos/{repository}/actions/runs/{run_id}":
        raise V18Error("GitHub workflow observation endpoint changed")
    if int(value["http_status"]) != 200:
        raise V18Error("GitHub workflow observation HTTP status changed")
    for field in ("content_type", "http_date", "etag"):
        if not isinstance(value[field], str) or not value[field]:
            raise V18Error(f"GitHub workflow observation {field} is empty")
    if not value["content_type"].lower().split(";", 1)[0].strip().endswith("json"):
        raise V18Error("GitHub workflow content type is not JSON")
    try:
        http_date = parsedate_to_datetime(value["http_date"])
    except (TypeError, ValueError) as exc:
        raise V18Error("GitHub workflow Date is not RFC-7231") from exc
    retrieved = _timestamp(value["retrieved_at"], "workflow retrieved_at")
    if http_date.tzinfo is None or http_date.astimezone(TOKYO) > retrieved:
        raise V18Error("GitHub workflow Date/retrieval chronology changed")
    for field in (
        "response_body_sha256",
        "response_headers_sha256",
        "observation_sha256",
    ):
        _require_sha(value[field], f"GitHub workflow {field}")
    header_projection = {
        "status": int(value["http_status"]),
        "content_type": value["content_type"],
        "date": value["http_date"],
        "etag": value["etag"],
    }
    if canonical_json_sha256(header_projection) != value["response_headers_sha256"]:
        raise V18Error("GitHub workflow response-header hash changed")
    projection = value["canonical_projection"]
    if not isinstance(projection, Mapping) or set(projection) != set(
        GITHUB_WORKFLOW_PROJECTION_FIELDS
    ) or set(projection) != set(contract["canonical_projection_required_fields"]):
        raise V18Error("GitHub workflow projection fields changed")
    workflow_id = projection["workflow_id"]
    projected_run_id = projection["run_id"]
    if (
        isinstance(workflow_id, bool)
        or not isinstance(workflow_id, int)
        or workflow_id <= 0
        or projected_run_id != run_id
        or projection["workflow_name"] != "tests"
        or projection["workflow_path"] != WORKFLOW_PATH
        or projection["event"] != "pull_request"
        or projection["head_sha"] != expected_head_sha
        or projection["run_attempt"] != 1
        or projection["status"] != "completed"
        or (
            projection["conclusion"]
            not in contract.get("terminal_conclusion_values", ["success"])
        )
        or projection["html_url"]
        != f"https://github.com/{repository}/actions/runs/{run_id}"
    ):
        raise V18Error("GitHub workflow canonical projection changed")
    created = _timestamp(projection["created_at"], "workflow created_at")
    started = _timestamp(projection["run_started_at"], "workflow run_started_at")
    updated = _timestamp(projection["updated_at"], "workflow updated_at")
    if created > started or started > updated or updated > retrieved:
        raise V18Error("GitHub workflow timestamp chronology changed")
    expected_hash = canonical_json_sha256(
        value, exclude_fields={"observation_sha256"}
    )
    if value["observation_sha256"] != expected_hash:
        raise V18Error("GitHub workflow observation self-hash changed")
    return value


def _refetch_github_workflow_observation(
    observation: Mapping[str, Any],
    *,
    checkpoint: bool = False,
) -> None:
    value = validate_github_workflow_observation(
        observation,
        expected_head_sha=str(observation.get("expected_head_sha")),
        checkpoint=checkpoint,
    )
    run_id = int(value["run_id"])
    repository = str(value["repository"])
    body = _github_api(f"/repos/{repository}/actions/runs/{run_id}")
    projection = {
        "run_id": body.get("id"),
        "workflow_id": body.get("workflow_id"),
        "workflow_name": body.get("name"),
        "workflow_path": body.get("path"),
        "event": body.get("event"),
        "head_sha": body.get("head_sha"),
        "run_attempt": body.get("run_attempt"),
        "status": body.get("status"),
        "conclusion": body.get("conclusion"),
        "created_at": body.get("created_at"),
        "run_started_at": body.get("run_started_at"),
        "updated_at": body.get("updated_at"),
        "html_url": body.get("html_url"),
    }
    if projection != value["canonical_projection"]:
        raise V18Error("GitHub workflow immutable projection changed on refetch")
    if _select_github_workflow_run_id(
        str(value["expected_head_sha"]),
        require_success=not checkpoint,
    ) != run_id:
        raise V18Error("GitHub workflow is no longer the lowest attempt-one run")


def validate_github_observation(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    contract = protocol["activation"]["github_observation_contract"]
    value = dict(observation)
    if set(value) != set(contract["required_fields"]):
        raise V18Error("GitHub observation fields differ from protocol")
    for field, expected in contract["fixed_values"].items():
        if value.get(field) != expected:
            raise V18Error(f"GitHub observation fixed field changed: {field}")
    kind = value["observation_kind"]
    repository = value["repository"]
    if kind == "commit":
        requested = str(value["requested_commit_sha"])
        if value["branch"] is not None or re.fullmatch(r"[0-9a-f]{40}", requested) is None:
            raise V18Error("GitHub commit observation subject is invalid")
        expected_endpoint = f"/repos/{repository}/commits/{requested}"
    elif kind == "branch_tip":
        if (
            value["requested_commit_sha"] is not None
            or value["branch"] != read_json(PROTOCOL)["branch"]
        ):
            raise V18Error("GitHub branch observation subject is invalid")
        expected_endpoint = (
            f"/repos/{repository}/commits/{quote(str(value['branch']), safe='')}"
        )
    else:
        raise V18Error("GitHub observation kind is invalid")
    if value["endpoint"] != expected_endpoint:
        raise V18Error("GitHub observation endpoint changed")
    if int(value["http_status"]) != 200:
        raise V18Error("GitHub observation HTTP status changed")
    for field in ("content_type", "http_date", "etag"):
        if not isinstance(value[field], str) or not value[field]:
            raise V18Error(f"GitHub observation {field} is empty")
    if not value["content_type"].lower().split(";", 1)[0].strip().endswith(
        "json"
    ):
        raise V18Error("GitHub observation content type is not JSON")
    try:
        http_date = parsedate_to_datetime(value["http_date"])
    except (TypeError, ValueError) as exc:
        raise V18Error("GitHub observation Date is not RFC-7231") from exc
    retrieved = _timestamp(value["retrieved_at"], "GitHub observation retrieved_at")
    if http_date.tzinfo is None or http_date.astimezone(TOKYO) > retrieved:
        raise V18Error("GitHub observation Date/retrieval chronology changed")
    for field in (
        "response_body_sha256",
        "response_headers_sha256",
        "observation_sha256",
    ):
        _require_sha(value[field], f"GitHub observation {field}")
    header_projection = {
        "status": int(value["http_status"]),
        "content_type": value["content_type"],
        "date": value["http_date"],
        "etag": value["etag"],
    }
    if canonical_json_sha256(header_projection) != value["response_headers_sha256"]:
        raise V18Error("GitHub observation response-header hash changed")
    projection = value["canonical_projection"]
    if not isinstance(projection, Mapping) or set(projection) != set(
        contract["canonical_projection_required_fields"]
    ):
        raise V18Error("GitHub observation projection fields changed")
    commit = str(projection["commit_sha"])
    parents = projection["parent_shas"]
    if (
        re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or not isinstance(parents, list)
        or parents != sorted(set(parents))
        or any(re.fullmatch(r"[0-9a-f]{40}", str(item)) is None for item in parents)
    ):
        raise V18Error("GitHub observation commit/parent projection is invalid")
    expected_url = f"https://github.com/{repository}/commit/{commit}"
    if projection["html_url"] != expected_url:
        raise V18Error("GitHub observation commit URL changed")
    committer = _timestamp(
        projection["committer_date"], "GitHub observation committer_date"
    )
    if committer > retrieved:
        raise V18Error("GitHub observation predates its projected commit")
    if kind == "commit" and commit != value["requested_commit_sha"]:
        raise V18Error("GitHub commit observation returned another commit")
    expected_hash = canonical_json_sha256(
        value, exclude_fields={"observation_sha256"}
    )
    if value["observation_sha256"] != expected_hash:
        raise V18Error("GitHub observation self-hash changed")
    return value


def _validate_github_observation_pair(
    commit_observation: Mapping[str, Any],
    branch_observation: Mapping[str, Any],
    *,
    commit_sha: str,
    branch: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    commit_value = validate_github_observation(commit_observation)
    branch_value = validate_github_observation(branch_observation)
    if (
        commit_value["observation_kind"] != "commit"
        or commit_value["canonical_projection"]["commit_sha"] != commit_sha
        or branch_value["observation_kind"] != "branch_tip"
        or branch_value["branch"] != branch
    ):
        raise V18Error("GitHub commit/branch observation pair changed subjects")
    return commit_value, branch_value


def _refetch_github_observation_pair(
    commit_observation: Mapping[str, Any],
    branch_observation: Mapping[str, Any],
) -> None:
    commit_value = validate_github_observation(commit_observation)
    branch_value = validate_github_observation(branch_observation)
    commit_sha = str(commit_value["canonical_projection"]["commit_sha"])
    current_commit = _github_observation(commit_sha=commit_sha)
    if current_commit["canonical_projection"] != commit_value["canonical_projection"]:
        raise V18Error("GitHub immutable commit identity changed on refetch")
    current_branch = _github_observation(branch=str(branch_value["branch"]))
    current_tip = str(current_branch["canonical_projection"]["commit_sha"])
    historical_tip = str(branch_value["canonical_projection"]["commit_sha"])
    _github_require_ancestor(commit_sha, current_tip, "commit-to-current-branch")
    _github_require_ancestor(
        historical_tip, current_tip, "historical-tip-to-current-branch"
    )


def _github_commit_evidence(
    commit_sha: str,
    *,
    commit_url: str,
    committed_at: Any,
) -> dict[str, Any]:
    observation = _github_observation(commit_sha=commit_sha)
    evidence = observation["canonical_projection"]
    if evidence.get("commit_sha") != commit_sha or evidence.get("html_url") != commit_url:
        raise V18Error("GitHub commit observation identity differs")
    github_committed_at = evidence.get("committer_date")
    if _timestamp(github_committed_at, "GitHub committed_at") != _timestamp(
        committed_at, "observed committed_at"
    ):
        raise V18Error("GitHub commit timestamp differs from observation")
    return observation


def _github_branch_tip(branch: str) -> str:
    evidence = _github_observation(branch=branch)
    sha = str(evidence["canonical_projection"]["commit_sha"])
    if GIT_SHA_RE.fullmatch(sha) is None:
        raise V18Error("GitHub registered branch tip is invalid")
    return sha


def _github_require_ancestor(ancestor: str, descendant: str, name: str) -> None:
    if ancestor == descendant:
        return
    evidence = _github_api(
        "/repos/rokuroku-066/TSE-Session-Ranker/compare/"
        f"{ancestor}...{descendant}"
    )
    if evidence.get("status") != "ahead" or evidence.get("base_commit", {}).get(
        "sha"
    ) != ancestor or evidence.get("merge_base_commit", {}).get("sha") != ancestor:
        raise V18Error(f"GitHub {name} ancestry is not satisfied")


def _payload_additional_test_artifacts(
    payload: Mapping[str, Any], *, verify_worktree: bool
) -> tuple[dict[str, str], ...]:
    records = _validate_additional_test_artifacts(
        payload.get("additional_test_artifacts"),
        verify_worktree=verify_worktree,
    )
    if any(item["path"] == payload.get("tests_path") for item in records):
        raise V18Error("additional test artifacts overlap the legacy base test")
    return records


def _validate_additional_test_git_blobs(
    records: Sequence[Mapping[str, str]], *, commit_sha: str
) -> None:
    validated = _validate_additional_test_artifacts(
        list(records), verify_worktree=False
    )
    for item in validated:
        observed = hashlib.sha256(
            _git_file_bytes(commit_sha, item["path"])
        ).hexdigest()
        if observed != item["sha256"]:
            raise V18Error(
                f"preregistration commit additional test changed: {item['path']}"
            )


def _preregistered_artifact_paths(payload: Mapping[str, Any]) -> tuple[str, ...]:
    legacy = tuple(
        str(payload[field])
        for field in (
            "protocol_path",
            "hypothesis_path",
            "runtime_lock_path",
            "runner_path",
            "audit_path",
            "rehearsal_path",
            "tests_path",
            "iteration_report_path",
            "validation_report_path",
            "session_calendar_path",
            "workflow_path",
        )
    )
    additional = tuple(
        item["path"]
        for item in _payload_additional_test_artifacts(
            payload, verify_worktree=False
        )
    )
    combined = (*legacy, *additional)
    if len(combined) != len(set(combined)):
        raise V18Error("preregistered artifact paths overlap or duplicate")
    return combined


def _validate_preregistration_git_binding(payload: Mapping[str, Any]) -> str:
    prereg = _git_require_commit(
        payload["preregistration_commit_sha"], "preregistration_commit_sha"
    )
    artifact_pairs = (
        ("protocol_path", "protocol_sha256"),
        ("hypothesis_path", "hypothesis_sha256"),
        ("runtime_lock_path", "runtime_lock_sha256"),
        ("runner_path", "runner_sha256"),
        ("audit_path", "audit_sha256"),
        ("rehearsal_path", "rehearsal_sha256"),
        ("tests_path", "tests_sha256"),
        ("iteration_report_path", "iteration_report_sha256"),
        ("validation_report_path", "validation_report_sha256"),
        ("session_calendar_path", "session_calendar_sha256"),
        ("workflow_path", "workflow_sha256"),
    )
    for path_field, hash_field in artifact_pairs:
        observed = hashlib.sha256(
            _git_file_bytes(prereg, str(payload[path_field]))
        ).hexdigest()
        if observed != payload[hash_field]:
            raise V18Error(f"preregistration commit artifact changed: {path_field}")
    additional_tests = _payload_additional_test_artifacts(
        payload, verify_worktree=True
    )
    _validate_additional_test_git_blobs(additional_tests, commit_sha=prereg)
    preregistered_tree = _git_tree_paths(prereg)
    forbidden_exact = {
        "research/model_v18_shoulder_state_activation_payload.json",
        "research/model_v18_shoulder_state_activation_receipt.json",
        "research/model_v18_shoulder_state_decisions.jsonl",
        "research/model_v18_shoulder_state_outcomes.jsonl",
        "research/model_v18_shoulder_state_months.jsonl",
        "research/model_v18_shoulder_state_scores.csv",
        "research/model_v18_shoulder_state_picks.csv",
        "research/model_v18_shoulder_state_result.json",
    }
    forbidden_prefixes = (
        "research/model_v18_shoulder_state_decision_records/",
        "research/model_v18_shoulder_state_outcome_records/",
        "research/model_v18_shoulder_state_month_records/",
        "research/model_v18_shoulder_state_source_manifests/",
        "research/model_v18_shoulder_state_outcome_manifests/",
        "research/model_v18_shoulder_state_fold_manifests/",
        "research/model_v18_shoulder_state_fold_models/",
        "research/model_v18_shoulder_state_state_manifests/",
    )
    if preregistered_tree & forbidden_exact or any(
        path.startswith(forbidden_prefixes) for path in preregistered_tree
    ):
        raise V18Error("preregistration commit already contains a forward artifact")
    committed = _timestamp(
        payload["preregistration_commit_committed_at"],
        "preregistration_commit_committed_at",
    )
    observed = _timestamp(
        payload["preregistration_commit_observed_at"],
        "preregistration_commit_observed_at",
    )
    if observed < committed or _git_commit_timestamp(prereg) != committed:
        raise V18Error("preregistration commit observation is not causal or exact")
    observed_tip = _git_require_commit(
        payload["preregistration_branch_tip_sha_when_observed"],
        "preregistration_branch_tip_sha_when_observed",
    )
    if observed_tip != prereg:
        _git_is_ancestor(prereg, observed_tip, "preregistration-to-observed-branch-tip")
    commit_observation, branch_observation = _validate_github_observation_pair(
        payload["preregistration_commit_observation"],
        payload["preregistration_branch_observation"],
        commit_sha=prereg,
        branch=str(payload["branch"]),
    )
    workflow_observation = validate_github_workflow_observation(
        payload["preregistration_workflow_run_observation"],
        expected_head_sha=prereg,
    )
    workflow_projection = workflow_observation["canonical_projection"]
    _validate_local_git_projection(
        prereg,
        commit_observation,
        name="preregistration_commit_sha",
    )
    if (
        commit_observation["canonical_projection"]["html_url"]
        != payload["preregistration_commit_url"]
        or _timestamp(
            commit_observation["canonical_projection"]["committer_date"],
            "preregistration observation committer_date",
        )
        != committed
        or branch_observation["canonical_projection"]["commit_sha"] != observed_tip
        or max(
            _timestamp(commit_observation["retrieved_at"], "commit retrieved_at"),
            _timestamp(branch_observation["retrieved_at"], "branch retrieved_at"),
        )
        != observed
        or payload["preregistration_workflow_run_id"]
        != workflow_projection["run_id"]
        or _timestamp(
            payload["preregistration_workflow_run_updated_at"],
            "preregistration workflow updated_at",
        )
        != _timestamp(workflow_projection["updated_at"], "workflow updated_at")
        or _timestamp(
            payload["preregistration_workflow_run_observed_at"],
            "preregistration workflow observed_at",
        )
        != _timestamp(workflow_observation["retrieved_at"], "workflow retrieved_at")
        or committed
        > _timestamp(workflow_projection["created_at"], "workflow created_at")
    ):
        raise V18Error("preregistration flat fields differ from GitHub observations")
    return prereg


def _validate_payload_commit_git_binding(
    payload: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> tuple[str, str]:
    prereg = _validate_preregistration_git_binding(payload)
    payload_commit = _git_require_commit(
        receipt["payload_commit_sha"], "payload_commit_sha"
    )
    _require_sole_git_parent(
        payload_commit,
        prereg,
        "activation payload commit",
    )
    _git_is_ancestor(prereg, payload_commit, "preregistration-to-payload")
    payload_bytes = _git_file_bytes(payload_commit, str(receipt["payload_path"]))
    if hashlib.sha256(payload_bytes).hexdigest() != receipt["payload_file_sha256"]:
        raise V18Error("payload commit exact file bytes differ from receipt")
    try:
        committed_payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V18Error("payload commit file is not canonical JSON") from exc
    if committed_payload != dict(payload):
        raise V18Error("payload commit parsed object differs from activation payload")
    if _git_commit_timestamp(payload_commit) != _timestamp(
        receipt["payload_commit_committed_at"], "payload_commit_committed_at"
    ):
        raise V18Error("payload Git commit timestamp differs from GitHub observation")
    commit_observation, branch_observation = _validate_github_observation_pair(
        receipt["payload_commit_observation"],
        receipt["payload_branch_observation"],
        commit_sha=payload_commit,
        branch=str(payload["branch"]),
    )
    workflow_observation = validate_github_workflow_observation(
        receipt["payload_workflow_run_observation"],
        expected_head_sha=payload_commit,
    )
    workflow_projection = workflow_observation["canonical_projection"]
    _validate_local_git_projection(
        payload_commit,
        commit_observation,
        name="payload_commit_sha",
    )
    if commit_observation["canonical_projection"]["parent_shas"] != [prereg]:
        raise V18Error("payload GitHub observation does not prove sole-parent A-to-B")
    if (
        commit_observation["canonical_projection"]["html_url"]
        != receipt["payload_commit_url"]
        or _timestamp(
            commit_observation["canonical_projection"]["committer_date"],
            "payload observation committer_date",
        )
        != _timestamp(receipt["payload_commit_committed_at"], "payload committed_at")
        or branch_observation["canonical_projection"]["commit_sha"]
        != receipt["branch_tip_sha_when_payload_observed"]
        or max(
            _timestamp(commit_observation["retrieved_at"], "commit retrieved_at"),
            _timestamp(branch_observation["retrieved_at"], "branch retrieved_at"),
        )
        != _timestamp(receipt["payload_observed_at"], "payload observed_at")
        or receipt["payload_workflow_run_id"] != workflow_projection["run_id"]
        or _timestamp(
            receipt["payload_workflow_run_updated_at"],
            "payload workflow updated_at",
        )
        != _timestamp(workflow_projection["updated_at"], "workflow updated_at")
        or _timestamp(
            receipt["payload_workflow_run_observed_at"],
            "payload workflow observed_at",
        )
        != _timestamp(workflow_observation["retrieved_at"], "workflow retrieved_at")
        or _timestamp(
            receipt["payload_commit_committed_at"], "payload committed_at"
        )
        > _timestamp(workflow_projection["created_at"], "workflow created_at")
        or _timestamp(
            receipt["payload_commit_committed_at"], "payload committed_at"
        )
        < max(
            _timestamp(
                payload["preregistration_commit_observed_at"],
                "preregistration observed_at",
            ),
            _timestamp(
                payload["preregistration_workflow_run_observed_at"],
                "preregistration workflow observed_at",
            ),
        )
    ):
        raise V18Error("payload flat fields differ from GitHub observations")
    branch_tip = _git_require_commit(
        receipt["branch_tip_sha_when_payload_observed"],
        "branch_tip_sha_when_payload_observed",
    )
    if branch_tip != payload_commit:
        _git_is_ancestor(payload_commit, branch_tip, "payload-to-observed-branch-tip")
    for path in _preregistered_artifact_paths(payload):
        if _git_file_bytes(prereg, path) != _git_file_bytes(payload_commit, path):
            raise V18Error(f"payload commit modified preregistered path: {path}")
    if _git_changed_paths(prereg, payload_commit) != {str(receipt["payload_path"])}:
        raise V18Error("payload commit changed a path other than the activation payload")
    return prereg, payload_commit


def _validate_receipt_commit_git_binding(
    payload: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    receipt_commit_sha: Any,
    receipt_commit_committed_at: Any,
) -> str:
    prereg, payload_commit = _validate_payload_commit_git_binding(payload, receipt)
    receipt_commit = _git_require_commit(
        receipt_commit_sha, "activation_receipt_commit_sha"
    )
    _require_sole_git_parent(
        receipt_commit,
        payload_commit,
        "activation receipt commit",
    )
    _git_is_ancestor(payload_commit, receipt_commit, "payload-to-receipt")
    observed_payload_tip = _git_require_commit(
        receipt["branch_tip_sha_when_payload_observed"],
        "branch_tip_sha_when_payload_observed",
    )
    if observed_payload_tip != receipt_commit:
        _git_is_ancestor(
            observed_payload_tip,
            receipt_commit,
            "observed-payload-tip-to-receipt",
        )
    if _git_commit_timestamp(receipt_commit) != _timestamp(
        receipt_commit_committed_at, "activation_receipt_commit_committed_at"
    ):
        raise V18Error("receipt Git commit timestamp differs from GitHub observation")
    receipt_bytes = _git_file_bytes(
        receipt_commit,
        "research/model_v18_shoulder_state_activation_receipt.json",
    )
    if receipt_bytes != _plain_file_bytes(
        ACTIVATION_RECEIPT,
        label="canonical activation receipt",
    ):
        raise V18Error("receipt commit exact bytes differ from canonical receipt file")
    for path in _preregistered_artifact_paths(payload):
        prereg_bytes = _git_file_bytes(prereg, path)
        if prereg_bytes != _git_file_bytes(receipt_commit, path):
            raise V18Error(f"receipt commit modified preregistered path: {path}")
    payload_path = str(receipt["payload_path"])
    if _git_file_bytes(payload_commit, payload_path) != _git_file_bytes(
        receipt_commit, payload_path
    ):
        raise V18Error("receipt commit modified activation payload bytes")
    receipt_path = "research/model_v18_shoulder_state_activation_receipt.json"
    if _git_changed_paths(payload_commit, receipt_commit) != {receipt_path}:
        raise V18Error("receipt commit changed a path other than the activation receipt")
    branch = str(payload["branch"])
    try:
        branch_tip = str(_git("rev-parse", f"refs/heads/{branch}")).strip()
    except V18Error:
        branch_tip = str(_git("rev-parse", "HEAD")).strip()
    if branch_tip != receipt_commit:
        _git_is_ancestor(receipt_commit, branch_tip, "receipt-to-registered-branch-tip")
    return receipt_commit


def _protocol_activation() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol, _ = validate_protocol()
    return protocol, protocol["activation"]


def validate_predictor_cache_anchor_summary(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(summary)
    if set(value) != set(CACHE_ANCHOR_SUMMARY_FIELDS):
        raise V18Error("payload predictor-cache anchor summary fields changed")
    _date(value["latest_source_session"], "payload cache anchor latest session")
    _timestamp(value["verified_at"], "payload cache anchor verified_at")
    sealed = _timestamp(value["sealed_at"], "payload cache anchor sealed_at")
    if _timestamp(value["verified_at"], "payload cache anchor verified_at") > sealed:
        raise V18Error("payload cache anchor verification follows its seal")
    model_month = _month(
        value["model_price_snapshot_target_month"],
        "payload model-price snapshot target month",
    )
    if _latest_registered_source_before_month(model_month) > _date(
        value["latest_source_session"], "payload cache anchor latest session"
    ):
        raise V18Error("payload model-price snapshot exceeds its raw anchor")
    for field in (
        "raw_source_set_sha256",
        "ordered_shard_set_sha256",
        "cumulative_snapshot_file_sha256",
        "cumulative_snapshot_semantic_sha256",
        "model_price_snapshot_file_sha256",
        "model_price_snapshot_semantic_sha256",
        "model_price_snapshot_manifest_file_sha256",
        "model_price_snapshot_manifest_sha256",
        "snapshot_manifest_file_sha256",
        "snapshot_manifest_sha256",
        "direct_clean_room_verification_receipt_sha256",
        "compact_consumer_equivalence_receipt_sha256",
    ):
        _require_sha(value[field], f"payload cache anchor {field}")
    for field in (
        "raw_source_count",
        "ordered_shard_count",
        "cumulative_snapshot_byte_count",
        "model_price_snapshot_byte_count",
        "model_price_snapshot_manifest_byte_count",
    ):
        if isinstance(value[field], bool) or int(value[field]) <= 0:
            raise V18Error(f"payload cache anchor {field} must be positive")
    if int(value["raw_source_count"]) != int(value["ordered_shard_count"]):
        raise V18Error("payload cache anchor raw/shard counts differ")
    _external_key_components(
        value["cumulative_snapshot_object_key"],
        prefix=CACHE_ANCHOR_OBJECT_PREFIX,
        label="payload cache anchor snapshot",
    )
    _external_key_components(
        value["model_price_snapshot_object_key"],
        prefix=MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX,
        label="payload model-price snapshot",
    )
    _external_key_components(
        value["model_price_snapshot_manifest_object_key"],
        prefix=MODEL_PRICE_SNAPSHOT_OBJECT_PREFIX,
        label="payload model-price snapshot manifest",
    )
    _external_key_components(
        value["snapshot_manifest_object_key"],
        prefix=CACHE_ANCHOR_OBJECT_PREFIX,
        label="payload cache anchor manifest",
    )
    return value


def validate_activation_payload(
    payload: Mapping[str, Any] | str | Path,
    protocol: Mapping[str, Any] | None = None,
    predictor_raw_store_root: str | Path | None = None,
    predictor_derived_store_root: str | Path | None = None,
    reparse_predictor_cache_anchor: bool = False,
) -> tuple[dict[str, Any], str]:
    value = read_json(payload) if isinstance(payload, (str, Path)) else dict(payload)
    active_protocol = dict(protocol) if protocol is not None else validate_protocol()[0]
    contract = active_protocol["activation"]["payload"]
    required = set(contract["required_fields"])
    if set(value) != required:
        raise V18Error("activation payload fields differ from protocol")
    for field, expected in contract["fixed_values"].items():
        if value.get(field) != expected:
            raise V18Error(f"activation payload fixed field changed: {field}")
    additional_paths = _protocol_additional_test_artifact_paths(active_protocol)
    additional_tests = _validate_additional_test_artifacts(
        value["additional_test_artifacts"],
        expected_paths=additional_paths,
        verify_worktree=True,
    )
    if any(item["path"] == value.get("tests_path") for item in additional_tests):
        raise V18Error("activation payload test artifact registries overlap")
    anchor_summary = validate_predictor_cache_anchor_summary(
        value["predictor_cache_anchor"]
    )
    not_before_predecessor = _latest_required_predictor_source_session(
        _date(value["not_before_session"], "activation not-before session")
    )
    if not_before_predecessor <= _date(
        anchor_summary["latest_source_session"], "activation cache anchor H"
    ):
        raise V18Error(
            "activation not-before predecessor must strictly follow cache anchor H"
        )
    if (predictor_raw_store_root is None) != (
        predictor_derived_store_root is None
    ):
        raise V18Error("activation cache anchor validation requires both stores")
    if predictor_derived_store_root is not None:
        anchor_manifest, anchor_payload = _read_external_canonical_json(
            predictor_derived_store_root,
            anchor_summary["snapshot_manifest_object_key"],
            prefix=CACHE_ANCHOR_OBJECT_PREFIX,
            label="activation predictor cache anchor manifest",
        )
        _, observed_summary, _ = validate_predictor_cache_anchor(
            anchor_manifest,
            predictor_raw_store_root=predictor_raw_store_root,
            predictor_derived_store_root=predictor_derived_store_root,
            require_direct_clean_room_reparse=reparse_predictor_cache_anchor,
        )
        if hashlib.sha256(anchor_payload).hexdigest() != anchor_summary[
            "snapshot_manifest_file_sha256"
        ]:
            raise V18Error("activation cache anchor exact manifest bytes changed")
        if observed_summary != anchor_summary:
            raise V18Error("activation cache anchor summary differs from external anchor")
    _validate_git_commit(
        value["preregistration_commit_sha"],
        value["preregistration_commit_url"],
        "preregistration_commit_sha",
    )
    artifact_pairs = (
        ("protocol_path", "protocol_sha256"),
        ("hypothesis_path", "hypothesis_sha256"),
        ("runtime_lock_path", "runtime_lock_sha256"),
        ("runner_path", "runner_sha256"),
        ("audit_path", "audit_sha256"),
        ("rehearsal_path", "rehearsal_sha256"),
        ("tests_path", "tests_sha256"),
        ("iteration_report_path", "iteration_report_sha256"),
        ("validation_report_path", "validation_report_sha256"),
        ("session_calendar_path", "session_calendar_sha256"),
        ("workflow_path", "workflow_sha256"),
    )
    for path_field, hash_field in artifact_pairs:
        path = ROOT / str(value[path_field])
        if not path.is_file() or sha256_file(path) != value[hash_field]:
            raise V18Error(f"activation payload artifact mismatch: {path_field}")
    if value["protocol_sha256"] != PROTOCOL_SHA256:
        raise V18Error("activation payload protocol SHA changed")
    if value["hypothesis_sha256"] != HYPOTHESIS_SHA256:
        raise V18Error("activation payload hypothesis SHA changed")
    if value["runtime_lock_sha256"] != RUNTIME_LOCK_SHA256:
        raise V18Error("activation payload runtime-lock SHA changed")
    if value["session_calendar_sha256"] != CALENDAR_SHA256:
        raise V18Error("activation payload calendar SHA changed")
    if value["workflow_sha256"] != WORKFLOW_SHA256:
        raise V18Error("activation payload workflow SHA changed")
    _validate_preregistration_git_binding(value)
    expected_hash = canonical_json_sha256(value, exclude_fields={"payload_sha256"})
    if value["payload_sha256"] != expected_hash:
        raise V18Error("activation payload self-hash mismatch")
    return value, expected_hash


def create_activation_payload(
    *,
    preregistration_commit_sha: str,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
    predictor_cache_anchor_manifest_object_key: str,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Create commit-B payload; the caller must commit it after prereg commit A."""

    protocol, activation = _protocol_activation()
    fixed = dict(activation["payload"]["fixed_values"])
    preregistration_commit_sha = _git_require_commit(
        preregistration_commit_sha, "preregistration_commit_sha"
    )
    if output is not None and _local_authority_presence_state(
        output, label="activation payload"
    ) != "absent":
        retained = read_json(output)
        validated, _ = validate_activation_payload(
            retained,
            protocol,
            predictor_raw_store_root=predictor_raw_store_root,
            predictor_derived_store_root=predictor_derived_store_root,
            reparse_predictor_cache_anchor=False,
        )
        if (
            validated["preregistration_commit_sha"]
            != preregistration_commit_sha
            or validated["predictor_cache_anchor"][
                "snapshot_manifest_object_key"
            ]
            != predictor_cache_anchor_manifest_object_key
        ):
            raise V18Error("activation payload retry changes its exact authority")
        return validated
    commit_observation = _github_observation(
        commit_sha=preregistration_commit_sha
    )
    branch_observation = _github_observation(branch=str(fixed["branch"]))
    workflow_observation = _github_workflow_run_observation(
        preregistration_commit_sha
    )
    anchor_manifest, anchor_manifest_payload = _read_external_canonical_json(
        predictor_derived_store_root,
        predictor_cache_anchor_manifest_object_key,
        prefix=CACHE_ANCHOR_OBJECT_PREFIX,
        label="activation predictor cache anchor manifest",
    )
    _, predictor_cache_anchor, _ = validate_predictor_cache_anchor(
        anchor_manifest,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
        require_direct_clean_room_reparse=True,
    )
    if hashlib.sha256(anchor_manifest_payload).hexdigest() != (
        predictor_cache_anchor["snapshot_manifest_file_sha256"]
    ):
        raise V18Error("activation cache anchor exact manifest file hash changed")
    projection = commit_observation["canonical_projection"]
    preregistration_commit_url = str(projection["html_url"])
    committed = _timestamp(projection["committer_date"], "commit committer_date")
    observed = max(
        _timestamp(commit_observation["retrieved_at"], "commit retrieved_at"),
        _timestamp(branch_observation["retrieved_at"], "branch retrieved_at"),
    )
    observed_tip = str(branch_observation["canonical_projection"]["commit_sha"])
    _github_require_ancestor(
        preregistration_commit_sha,
        observed_tip,
        "preregistration-to-observed-branch-tip",
    )
    value: dict[str, Any] = {
        **fixed,
        "protocol_sha256": PROTOCOL_SHA256,
        "hypothesis_sha256": sha256_file(ROOT / fixed["hypothesis_path"]),
        "runner_sha256": sha256_file(ROOT / fixed["runner_path"]),
        "audit_sha256": sha256_file(ROOT / fixed["audit_path"]),
        "rehearsal_sha256": sha256_file(ROOT / fixed["rehearsal_path"]),
        "tests_sha256": sha256_file(ROOT / fixed["tests_path"]),
        "iteration_report_sha256": sha256_file(
            ROOT / fixed["iteration_report_path"]
        ),
        "validation_report_sha256": sha256_file(
            ROOT / fixed["validation_report_path"]
        ),
        "additional_test_artifacts": _additional_test_artifacts_from_worktree(
            _protocol_additional_test_artifact_paths(protocol)
        ),
        "predictor_cache_anchor": predictor_cache_anchor,
        "preregistration_commit_sha": preregistration_commit_sha,
        "preregistration_commit_url": preregistration_commit_url,
        "preregistration_commit_committed_at": committed,
        "preregistration_commit_observed_at": observed,
        "preregistration_branch_tip_sha_when_observed": observed_tip,
        "preregistration_commit_observation": commit_observation,
        "preregistration_branch_observation": branch_observation,
        "preregistration_workflow_run_id": workflow_observation[
            "canonical_projection"
        ]["run_id"],
        "preregistration_workflow_run_updated_at": _timestamp(
            workflow_observation["canonical_projection"]["updated_at"],
            "preregistration workflow updated_at",
        ),
        "preregistration_workflow_run_observed_at": _timestamp(
            workflow_observation["retrieved_at"],
            "preregistration workflow observed_at",
        ),
        "preregistration_workflow_run_observation": workflow_observation,
    }
    if value["hypothesis_sha256"] != HYPOTHESIS_SHA256:
        raise V18Error("hypothesis bytes differ from final preregistration authority")
    value["payload_sha256"] = canonical_json_sha256(
        value, exclude_fields={"payload_sha256"}
    )
    validate_activation_payload(
        value,
        protocol,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
        reparse_predictor_cache_anchor=False,
    )
    if output is not None:
        write_json(value, output, exclusive=True)
    return value


def validate_activation_receipt(
    receipt: Mapping[str, Any] | str | Path,
    payload: Mapping[str, Any] | str | Path,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    value = read_json(receipt) if isinstance(receipt, (str, Path)) else dict(receipt)
    active_protocol = dict(protocol) if protocol is not None else validate_protocol()[0]
    payload_value, payload_hash = validate_activation_payload(payload, active_protocol)
    payload_file_hash = (
        sha256_file(payload)
        if isinstance(payload, (str, Path))
        else hashlib.sha256(_json_file_bytes(payload_value)).hexdigest()
    )
    contract = active_protocol["activation"]["receipt"]
    if set(value) != set(contract["required_fields"]):
        raise V18Error("activation receipt fields differ from protocol")
    for field, expected in contract["fixed_values"].items():
        if value.get(field) != expected:
            raise V18Error(f"activation receipt fixed field changed: {field}")
    if value["activation_id"] != payload_value["activation_id"]:
        raise V18Error("activation receipt id differs from payload")
    if value["payload_sha256"] != payload_hash:
        raise V18Error("activation receipt does not bind payload")
    if value["payload_file_sha256"] != payload_file_hash:
        raise V18Error("activation receipt does not bind exact payload file bytes")
    _validate_git_commit(
        value["payload_commit_sha"], value["payload_commit_url"], "payload_commit_sha"
    )
    committed = _timestamp(value["payload_commit_committed_at"], "payload committed")
    observed = _timestamp(value["payload_observed_at"], "payload observed")
    issued = _timestamp(value["receipt_issued_at"], "receipt issued")
    workflow_observation = validate_github_workflow_observation(
        value["payload_workflow_run_observation"],
        expected_head_sha=str(value["payload_commit_sha"]),
    )
    workflow_projection = workflow_observation["canonical_projection"]
    if (
        observed < committed
        or issued < observed
        or value["payload_workflow_run_id"] != workflow_projection["run_id"]
        or _timestamp(
            value["payload_workflow_run_updated_at"],
            "payload workflow updated_at",
        )
        != _timestamp(workflow_projection["updated_at"], "workflow updated_at")
        or _timestamp(
            value["payload_workflow_run_observed_at"],
            "payload workflow observed_at",
        )
        != _timestamp(workflow_observation["retrieved_at"], "workflow retrieved_at")
        or issued
        < _timestamp(
            value["payload_workflow_run_observed_at"],
            "payload workflow observed_at",
        )
    ):
        raise V18Error("activation receipt timestamps are not causal")
    branch_tip = str(value["branch_tip_sha_when_payload_observed"])
    if GIT_SHA_RE.fullmatch(branch_tip) is None:
        raise V18Error("activation receipt branch tip is invalid")
    _validate_payload_commit_git_binding(payload_value, value)
    expected_hash = canonical_json_sha256(value, exclude_fields={"receipt_sha256"})
    if value["receipt_sha256"] != expected_hash:
        raise V18Error("activation receipt self-hash mismatch")
    return value, expected_hash


def create_activation_receipt(
    payload: Mapping[str, Any] | str | Path,
    *,
    payload_commit_sha: str,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Create commit-C receipt from GitHub-observed commit-B metadata."""

    protocol, activation = _protocol_activation()
    payload_value, payload_hash = validate_activation_payload(payload, protocol)
    fixed = dict(activation["receipt"]["fixed_values"])
    payload_commit_sha = _git_require_commit(
        payload_commit_sha, "payload_commit_sha"
    )
    if output is not None and _local_authority_presence_state(
        output, label="activation receipt"
    ) != "absent":
        retained = read_json(output)
        validated, _ = validate_activation_receipt(
            retained, payload_value, protocol
        )
        if validated["payload_commit_sha"] != payload_commit_sha:
            raise V18Error("activation receipt retry changes payload commit B")
        return validated
    commit_observation = _github_observation(commit_sha=payload_commit_sha)
    branch_observation = _github_observation(branch=str(fixed["branch"]))
    workflow_observation = _github_workflow_run_observation(payload_commit_sha)
    projection = commit_observation["canonical_projection"]
    payload_observed_at = max(
        _timestamp(commit_observation["retrieved_at"], "commit retrieved_at"),
        _timestamp(branch_observation["retrieved_at"], "branch retrieved_at"),
    )
    _github_require_ancestor(
        payload_commit_sha,
        str(branch_observation["canonical_projection"]["commit_sha"]),
        "payload-to-observed-branch-tip",
    )
    receipt_issued_at = datetime.now(TOKYO)
    value: dict[str, Any] = {
        **fixed,
        "payload_sha256": payload_hash,
        "payload_file_sha256": (
            sha256_file(payload)
            if isinstance(payload, (str, Path))
            else hashlib.sha256(_json_file_bytes(payload_value)).hexdigest()
        ),
        "payload_commit_sha": payload_commit_sha,
        "payload_commit_url": projection["html_url"],
        "payload_commit_committed_at": _timestamp(
            projection["committer_date"], "payload_commit_committed_at"
        ),
        "branch_tip_sha_when_payload_observed": branch_observation[
            "canonical_projection"
        ]["commit_sha"],
        "payload_observed_at": payload_observed_at,
        "payload_commit_observation": commit_observation,
        "payload_branch_observation": branch_observation,
        "payload_workflow_run_id": workflow_observation["canonical_projection"][
            "run_id"
        ],
        "payload_workflow_run_updated_at": _timestamp(
            workflow_observation["canonical_projection"]["updated_at"],
            "payload workflow updated_at",
        ),
        "payload_workflow_run_observed_at": _timestamp(
            workflow_observation["retrieved_at"],
            "payload workflow observed_at",
        ),
        "payload_workflow_run_observation": workflow_observation,
        "receipt_issued_at": receipt_issued_at,
    }
    if value["activation_id"] != payload_value["activation_id"]:
        raise V18Error("activation id differs")
    value["receipt_sha256"] = canonical_json_sha256(
        value, exclude_fields={"receipt_sha256"}
    )
    validate_activation_receipt(value, payload_value, protocol)
    if output is not None:
        write_json(value, output, exclusive=True)
    return value


def activate(
    payload: Mapping[str, Any] | str | Path,
    receipt: Mapping[str, Any] | str | Path,
    *,
    activation_receipt_commit_sha: str,
    calendar: pd.DatetimeIndex | None = None,
) -> dict[str, Any]:
    """Observe commit C and freeze first/terminal sessions without backfill."""

    protocol, _ = validate_protocol()
    payload_value, payload_hash = validate_activation_payload(payload, protocol)
    receipt_value, receipt_hash = validate_activation_receipt(
        receipt, payload_value, protocol
    )
    receipt_commit = _git_require_commit(
        activation_receipt_commit_sha, "activation_receipt_commit_sha"
    )
    commit_observation = _github_observation(commit_sha=receipt_commit)
    branch_observation = _github_observation(branch=str(payload_value["branch"]))
    workflow_observation = _github_workflow_run_observation(receipt_commit)
    commit_observation, branch_observation = _validate_github_observation_pair(
        commit_observation,
        branch_observation,
        commit_sha=receipt_commit,
        branch=str(payload_value["branch"]),
    )
    _validate_local_git_projection(
        receipt_commit,
        commit_observation,
        name="activation_receipt_commit_sha",
    )
    commit_projection = commit_observation["canonical_projection"]
    branch_projection = branch_observation["canonical_projection"]
    workflow_projection = workflow_observation["canonical_projection"]
    if commit_projection["parent_shas"] != [receipt_value["payload_commit_sha"]]:
        raise V18Error("receipt GitHub observation does not prove sole-parent B-to-C")
    activation_receipt_commit_url = str(commit_projection["html_url"])
    committed = _timestamp(
        commit_projection["committer_date"],
        "activation_receipt_commit_committed_at",
    )
    observed = max(
        _timestamp(commit_observation["retrieved_at"], "receipt commit retrieved_at"),
        _timestamp(branch_observation["retrieved_at"], "receipt branch retrieved_at"),
    )
    if observed < committed:
        raise V18Error("receipt commit observation predates commit")
    if committed < _timestamp(receipt_value["receipt_issued_at"], "receipt_issued_at"):
        raise V18Error("receipt containing commit predates receipt issue")
    if committed > _timestamp(workflow_projection["created_at"], "workflow created_at"):
        raise V18Error("receipt workflow predates its containing commit")
    observed_tip = str(branch_projection["commit_sha"])
    _github_require_ancestor(
        receipt_commit, observed_tip, "receipt-to-observed-branch-tip"
    )
    try:
        local_observed_tip = _git_require_commit(
            observed_tip, "branch_tip_sha_when_receipt_observed"
        )
    except V18Error:
        local_observed_tip = None
    if local_observed_tip is not None and local_observed_tip != receipt_commit:
        _git_is_ancestor(
            receipt_commit,
            local_observed_tip,
            "receipt-to-locally-observed-branch-tip",
        )
    receipt_blob_hash = hashlib.sha256(
        _git_file_bytes(
            receipt_commit,
            "research/model_v18_shoulder_state_activation_receipt.json",
        )
    ).hexdigest()
    _validate_receipt_commit_git_binding(
        payload_value,
        receipt_value,
        receipt_commit_sha=receipt_commit,
        receipt_commit_committed_at=committed,
    )
    scheduled = load_registered_calendar() if calendar is None else calendar
    first = first_counted_session(
        workflow_run_updated_at=workflow_projection["updated_at"],
        workflow_run_observed_at=workflow_observation["retrieved_at"],
        calendar=scheduled,
        not_before_session=payload_value["not_before_session"],
    )
    if _cutoff(first) <= max(
        observed,
        _timestamp(workflow_observation["retrieved_at"], "workflow retrieved_at"),
    ):
        raise V18Error("activation observations missed the nondiscretionary first cutoff")
    first_predecessor = _latest_required_predictor_source_session(first)
    anchor_latest = _date(
        validate_predictor_cache_anchor_summary(
            payload_value["predictor_cache_anchor"]
        )["latest_source_session"],
        "activation predictor anchor latest session",
    )
    if first_predecessor <= anchor_latest:
        raise V18Error(
            "activation first counted predecessor must strictly follow anchor H"
        )
    terminal = deterministic_terminal_session(first, scheduled)
    denominator = scheduled[(scheduled >= first) & (scheduled <= terminal)]
    return {
        "activation_payload_sha256": payload_hash,
        "activation_receipt_sha256": receipt_hash,
        "activation_receipt_commit_sha": receipt_commit,
        "activation_receipt_commit_url": activation_receipt_commit_url,
        "activation_receipt_commit_committed_at": committed.isoformat(
            timespec="microseconds"
        ),
        "activation_receipt_commit_observed_at": observed.isoformat(
            timespec="microseconds"
        ),
        "branch_tip_sha_when_receipt_observed": observed_tip,
        "activation_receipt_file_sha256": receipt_blob_hash,
        "receipt_commit_observation": commit_observation,
        "receipt_branch_observation": branch_observation,
        "activation_receipt_workflow_run_id": workflow_projection["run_id"],
        "activation_receipt_workflow_run_updated_at": _timestamp(
            workflow_projection["updated_at"], "receipt workflow updated_at"
        ).isoformat(timespec="microseconds"),
        "activation_receipt_workflow_run_observed_at": _timestamp(
            workflow_observation["retrieved_at"], "receipt workflow observed_at"
        ).isoformat(timespec="microseconds"),
        "receipt_workflow_run_observation": workflow_observation,
        "first_counted_session": str(first.date()),
        "first_counted_predecessor_session": str(first_predecessor.date()),
        "terminal_session": str(terminal.date()),
        "terminal_scheduled_sessions": int(len(denominator)),
        "represented_calendar_months": int(denominator.to_period("M").nunique()),
        "calendar_sha256": CALENDAR_SHA256,
        "production_model_changed": False,
        "orders_allowed": False,
    }


def validate_canonical_activation_artifacts() -> tuple[dict[str, Any], str, dict[str, Any], str]:
    if not ACTIVATION_PAYLOAD.is_file() or not ACTIVATION_RECEIPT.is_file():
        raise V18Error("canonical activation payload/receipt artifacts are missing")
    payload, payload_hash = validate_activation_payload(ACTIVATION_PAYLOAD)
    receipt, receipt_hash = validate_activation_receipt(
        ACTIVATION_RECEIPT, ACTIVATION_PAYLOAD
    )
    return payload, payload_hash, receipt, receipt_hash


def validate_activation_context(context: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "activation_payload_sha256",
        "activation_receipt_sha256",
        "activation_receipt_commit_sha",
        "activation_receipt_commit_url",
        "activation_receipt_commit_committed_at",
        "activation_receipt_commit_observed_at",
        "branch_tip_sha_when_receipt_observed",
        "activation_receipt_file_sha256",
        "receipt_commit_observation",
        "receipt_branch_observation",
        "activation_receipt_workflow_run_id",
        "activation_receipt_workflow_run_updated_at",
        "activation_receipt_workflow_run_observed_at",
        "receipt_workflow_run_observation",
        "first_counted_session",
        "first_counted_predecessor_session",
        "terminal_session",
        "terminal_scheduled_sessions",
        "represented_calendar_months",
        "calendar_sha256",
        "production_model_changed",
        "orders_allowed",
    }
    value = dict(context)
    if set(value) != required:
        raise V18Error("activation preflight context fields changed")
    payload, payload_hash, receipt, receipt_hash = validate_canonical_activation_artifacts()
    if (
        value["activation_payload_sha256"] != payload_hash
        or value["activation_receipt_sha256"] != receipt_hash
    ):
        raise V18Error("activation context differs from canonical payload/receipt")
    _validate_git_commit(
        value["activation_receipt_commit_sha"],
        value["activation_receipt_commit_url"],
        "activation_receipt_commit_sha",
    )
    committed = _timestamp(
        value["activation_receipt_commit_committed_at"],
        "activation_receipt_commit_committed_at",
    )
    observed = _timestamp(
        value["activation_receipt_commit_observed_at"],
        "activation_receipt_commit_observed_at",
    )
    if observed < committed:
        raise V18Error("activation context observation predates receipt commit")
    commit_observation, branch_observation = _validate_github_observation_pair(
        value["receipt_commit_observation"],
        value["receipt_branch_observation"],
        commit_sha=str(value["activation_receipt_commit_sha"]),
        branch=str(payload["branch"]),
    )
    workflow_observation = validate_github_workflow_observation(
        value["receipt_workflow_run_observation"],
        expected_head_sha=str(value["activation_receipt_commit_sha"]),
    )
    workflow_projection = workflow_observation["canonical_projection"]
    if commit_observation["canonical_projection"]["parent_shas"] != [
        receipt["payload_commit_sha"]
    ]:
        raise V18Error("receipt GitHub observation does not prove sole-parent B-to-C")
    _validate_local_git_projection(
        str(value["activation_receipt_commit_sha"]),
        commit_observation,
        name="activation_receipt_commit_sha",
    )
    if (
        commit_observation["canonical_projection"]["html_url"]
        != value["activation_receipt_commit_url"]
        or _timestamp(
            commit_observation["canonical_projection"]["committer_date"],
            "receipt observation committer_date",
        )
        != committed
        or branch_observation["canonical_projection"]["commit_sha"]
        != value["branch_tip_sha_when_receipt_observed"]
        or max(
            _timestamp(commit_observation["retrieved_at"], "commit retrieved_at"),
            _timestamp(branch_observation["retrieved_at"], "branch retrieved_at"),
        )
        != observed
        or value["activation_receipt_workflow_run_id"]
        != workflow_projection["run_id"]
        or _timestamp(
            value["activation_receipt_workflow_run_updated_at"],
            "receipt workflow updated_at",
        )
        != _timestamp(workflow_projection["updated_at"], "workflow updated_at")
        or _timestamp(
            value["activation_receipt_workflow_run_observed_at"],
            "receipt workflow observed_at",
        )
        != _timestamp(workflow_observation["retrieved_at"], "workflow retrieved_at")
        or committed
        > _timestamp(workflow_projection["created_at"], "workflow created_at")
    ):
        raise V18Error("activation context flat fields differ from observations")
    observed_tip = _git_require_commit(
        value["branch_tip_sha_when_receipt_observed"],
        "branch_tip_sha_when_receipt_observed",
    )
    receipt_commit = _git_require_commit(
        value["activation_receipt_commit_sha"], "activation_receipt_commit_sha"
    )
    if observed_tip != receipt_commit:
        _git_is_ancestor(receipt_commit, observed_tip, "receipt-to-observed-branch-tip")
    receipt_blob_hash = hashlib.sha256(
        _git_file_bytes(
            receipt_commit,
            "research/model_v18_shoulder_state_activation_receipt.json",
        )
    ).hexdigest()
    if value["activation_receipt_file_sha256"] != receipt_blob_hash:
        raise V18Error("activation context receipt file hash changed")
    _validate_receipt_commit_git_binding(
        payload,
        receipt,
        receipt_commit_sha=value["activation_receipt_commit_sha"],
        receipt_commit_committed_at=committed,
    )
    first = first_counted_session(
        workflow_run_updated_at=value["activation_receipt_workflow_run_updated_at"],
        workflow_run_observed_at=value[
            "activation_receipt_workflow_run_observed_at"
        ],
        not_before_session=payload["not_before_session"],
    )
    if _cutoff(first) <= max(
        observed,
        _timestamp(
            value["activation_receipt_workflow_run_observed_at"],
            "receipt workflow observed_at",
        ),
    ):
        raise V18Error("activation observations missed the first counted cutoff")
    first_predecessor = _latest_required_predictor_source_session(first)
    anchor_latest = _date(
        validate_predictor_cache_anchor_summary(
            payload["predictor_cache_anchor"]
        )["latest_source_session"],
        "activation predictor anchor latest session",
    )
    if first_predecessor <= anchor_latest:
        raise V18Error(
            "activation first counted predecessor must strictly follow anchor H"
        )
    terminal = deterministic_terminal_session(first)
    calendar = load_registered_calendar()
    denominator = calendar[(calendar >= first) & (calendar <= terminal)]
    if (
        value["first_counted_session"] != str(first.date())
        or value["first_counted_predecessor_session"]
        != str(first_predecessor.date())
        or value["terminal_session"] != str(terminal.date())
        or int(value["terminal_scheduled_sessions"]) != len(denominator)
        or int(value["represented_calendar_months"])
        != denominator.to_period("M").nunique()
        or value["calendar_sha256"] != CALENDAR_SHA256
        or value["production_model_changed"] is not False
        or value["orders_allowed"] is not False
    ):
        raise V18Error("activation context derived denominator changed")
    return value


def validate_terminal_activation_observations(
    decisions: Sequence[Mapping[str, Any]],
) -> None:
    """Refetch immutable A/B/C identities and current branch ancestry at terminal."""

    if not _STRICT_RUNTIME_ACTIVE:
        raise V18Error("terminal GitHub refetch requires strict operational runtime")
    rows = validate_decision_records(decisions)
    if not rows:
        raise V18Error("terminal GitHub refetch requires a decision ledger")
    payload, _, receipt, _ = validate_canonical_activation_artifacts()
    _refetch_github_observation_pair(
        payload["preregistration_commit_observation"],
        payload["preregistration_branch_observation"],
    )
    _refetch_github_workflow_observation(
        payload["preregistration_workflow_run_observation"]
    )
    _refetch_github_observation_pair(
        receipt["payload_commit_observation"],
        receipt["payload_branch_observation"],
    )
    _refetch_github_workflow_observation(
        receipt["payload_workflow_run_observation"]
    )
    _refetch_github_observation_pair(
        rows[0]["receipt_commit_observation"],
        rows[0]["receipt_branch_observation"],
    )
    _refetch_github_workflow_observation(
        rows[0]["receipt_workflow_run_observation"]
    )


def _checkpoint_proposal_path(session: Any, role: str) -> Path:
    target = _date(session, "checkpoint session")
    if role not in CHECKPOINT_ROLES:
        raise V18Error("checkpoint role is not registered")
    return CHECKPOINT_PROPOSAL_DIR / str(target.date()) / f"{role}.json"


def _checkpoint_core_object_key(session: Any, role: str) -> str:
    target = _date(session, "checkpoint session")
    if role not in CHECKPOINT_ROLES:
        raise V18Error("checkpoint role is not registered")
    return f"{CHECKPOINT_CORE_OBJECT_PREFIX}{target.date()}/{role}.bin"


def _decision_core_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in CHECKPOINT_CORE_FIELDS}


def validate_checkpoint_decision_core(
    core: Mapping[str, Any],
    *,
    session_date: Any,
    resolution: str,
) -> dict[str, Any]:
    value = dict(core)
    if set(value) != set(CHECKPOINT_CORE_FIELDS):
        raise V18Error("checkpoint decision core fields differ from protocol")
    target = _date(session_date, "checkpoint core session")
    if value["candidate_id"] != CANDIDATE_ID:
        raise V18Error("checkpoint core candidate changed")
    if _timestamp(value["decision_cutoff"], "checkpoint decision cutoff") != _cutoff(
        target
    ):
        raise V18Error("checkpoint core cutoff changed")
    if resolution not in CHECKPOINT_RESOLUTIONS:
        raise V18Error("checkpoint core resolution is not registered")
    runtime_verified = _timestamp(
        value["runtime_lock_verified_at"], "checkpoint runtime_lock_verified_at"
    )
    computed = _timestamp(value["computed_at"], "checkpoint core computed_at")
    if computed < runtime_verified or computed > _cutoff(target):
        raise V18Error("checkpoint core computation timestamp is invalid")
    for field in ("source_manifest_sha256", "state_manifest_sha256"):
        _require_sha(value[field], f"checkpoint core {field}")
    score_pair = (
        value["score_session_file_sha256"],
        value["score_session_semantic_sha256"],
    )
    if any(item is None for item in score_pair):
        raise V18Error("checkpoint requires the exact two-row score-session shard")
    _require_sha(score_pair[0], "checkpoint score-session file SHA")
    _require_sha(score_pair[1], "checkpoint score-session semantic SHA")
    _require_sha(value["score_session_set_sha256"], "checkpoint score-session set SHA")
    fold_pair = (
        value["c00_fold_manifest_sha256"],
        value["fold_model_bundle_file_sha256"],
    )
    if any(item is None for item in fold_pair):
        raise V18Error("checkpoint requires the exact monthly fold/bundle pair")
    _require_sha(fold_pair[0], "checkpoint fold manifest")
    _require_sha(fold_pair[1], "checkpoint fold bundle")
    if not _strict_bool(value["source_complete"]) or not _strict_bool(
        value["model_complete"]
    ):
        raise V18Error("checkpoint cannot encode incomplete source/model authority")
    if value["failure_reason"] is not None and value["failure_reason"] not in DAILY_FAILURE_REASONS:
        raise V18Error("checkpoint core failure reason is not registered")
    return value


def _validate_checkpoint_core_object(
    value: Mapping[str, Any],
    proposal: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    contract = read_json(PROTOCOL)["daily_preopen_checkpoint_contract"][
        "sealed_core_store"
    ]
    required = set(contract["required_fields"])
    current = dict(value)
    if set(current) != required:
        raise V18Error("checkpoint sealed-core fields differ from protocol")
    if (
        current["schema_version"] != 1
        or current["target_session"] != proposal["target_session"]
        or current["checkpoint_role"] != proposal["checkpoint_role"]
        or int(current["decision_sequence_number"])
        != int(proposal["decision_sequence_number"])
        or current["previous_decision_record_sha256"]
        != proposal["previous_decision_record_sha256"]
        or re.fullmatch(r"[0-9a-f]{64}", str(current["nonce_hex"])) is None
    ):
        raise V18Error("checkpoint sealed-core identity changed")
    resolution = "primary"
    core = validate_checkpoint_decision_core(
        current["decision_core"],
        session_date=proposal["target_session"],
        resolution=resolution,
    )
    if _timestamp(core["computed_at"], "checkpoint core computed_at") > _timestamp(
        proposal["created_at"], "checkpoint proposal created_at"
    ):
        raise V18Error("checkpoint proposal predates its sealed decision core")
    core_hash = canonical_json_sha256(core)
    if current["decision_core_sha256"] != core_hash:
        raise V18Error("checkpoint decision-core hash changed")
    return core, core_hash


def encode_checkpoint_core_envelope(value: Mapping[str, Any]) -> bytes:
    body = _json_file_bytes(value)
    if len(body) > CHECKPOINT_CORE_ENVELOPE_BYTES - 13:
        raise V18Error("checkpoint core JSON exceeds its fixed envelope")
    header = CHECKPOINT_CORE_MAGIC + b"\x01" + struct.pack(">I", len(body))
    return header + body + bytes(CHECKPOINT_CORE_ENVELOPE_BYTES - len(header) - len(body))


def parse_checkpoint_core_envelope(payload: bytes) -> dict[str, Any]:
    if len(payload) != CHECKPOINT_CORE_ENVELOPE_BYTES:
        raise V18Error("checkpoint core envelope length changed")
    if payload[:8] != CHECKPOINT_CORE_MAGIC or payload[8] != 1:
        raise V18Error("checkpoint core envelope magic/version changed")
    length = struct.unpack(">I", payload[9:13])[0]
    if length <= 0 or 13 + length > len(payload):
        raise V18Error("checkpoint core envelope JSON length is invalid")
    body = payload[13 : 13 + length]
    if any(payload[13 + length :]):
        raise V18Error("checkpoint core envelope padding is not all zero")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V18Error("checkpoint core envelope JSON is invalid") from exc
    if not isinstance(value, dict) or body != _json_file_bytes(value):
        raise V18Error("checkpoint core envelope JSON is not canonical pretty JSON")
    return value


def validate_checkpoint_proposal(
    proposal: Mapping[str, Any] | str | Path,
    *,
    session_date: Any,
    role: str,
    activation: Mapping[str, Any],
    sequence_number: int,
    previous_record_sha256: str,
) -> tuple[dict[str, Any], str, str]:
    value = read_json(proposal) if isinstance(proposal, (str, Path)) else dict(proposal)
    if set(value) != set(CHECKPOINT_PROPOSAL_FIELDS):
        raise V18Error("checkpoint proposal fields differ from protocol")
    target = _date(session_date, "checkpoint proposal session")
    relative_path = _checkpoint_proposal_path(target, role).relative_to(ROOT).as_posix()
    expected_id = f"model_v18_shoulder_state_checkpoint_{target:%Y%m%d}_{role}"
    expected_batch_id = f"model_v18_shoulder_state_checkpoint_batch_{target:%Y%m%d}"
    fixed = {
        "schema_version": 1,
        "checkpoint_id": expected_id,
        "checkpoint_batch_id": expected_batch_id,
        "publication_ordinal": 0 if role == "safety_cash" else 1,
        "repository": "rokuroku-066/TSE-Session-Ranker",
        "branch": read_json(ACTIVATION_PAYLOAD)["branch"],
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "activation_payload_sha256": activation["activation_payload_sha256"],
        "activation_receipt_sha256": activation["activation_receipt_sha256"],
        "activation_receipt_commit_sha": activation["activation_receipt_commit_sha"],
        "target_session": str(target.date()),
        "checkpoint_role": role,
        "decision_sequence_number": int(sequence_number),
        "previous_decision_record_sha256": previous_record_sha256,
        "sealed_core_object_key": _checkpoint_core_object_key(target, role),
        "canonical_json_contract": CANONICAL_JSON_CONTRACT,
    }
    if any(value[field] != expected for field, expected in fixed.items()):
        raise V18Error("checkpoint proposal fixed authority changed")
    _timestamp(value["created_at"], "checkpoint proposal created_at")
    if isinstance(value["sealed_core_byte_count"], bool) or int(
        value["sealed_core_byte_count"]
    ) != CHECKPOINT_CORE_ENVELOPE_BYTES:
        raise V18Error("checkpoint sealed-core byte count is invalid")
    _require_sha(value["sealed_core_sha256"], "checkpoint sealed-core SHA")
    expected_hash = canonical_json_sha256(value, exclude_fields={"proposal_sha256"})
    if value["proposal_sha256"] != expected_hash:
        raise V18Error("checkpoint proposal self-hash changed")
    return value, expected_hash, relative_path


def _seal_checkpoint_core_bytes(
    payload: bytes,
    *,
    checkpoint_core_store_root: str | Path,
    object_key: str,
) -> tuple[int, str]:
    with _external_parent_fd(
        checkpoint_core_store_root,
        object_key,
        prefix=CHECKPOINT_CORE_OBJECT_PREFIX,
        label="checkpoint core",
        create_parents=True,
    ) as (parent_fd, file_name):
        try:
            descriptor = os.open(
                file_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError as exc:
            raise V18Error("checkpoint core object already exists and cannot be reused") from exc
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("short checkpoint-core write")
                remaining = remaining[written:]
            os.fsync(descriptor)
            sealed = os.fstat(descriptor)
            if not stat.S_ISREG(sealed.st_mode) or sealed.st_nlink != 1:
                raise V18Error("checkpoint core acquired a hard-link alias while sealing")
            os.fsync(parent_fd)
        except BaseException:
            os.close(descriptor)
            try:
                os.unlink(file_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            raise
        else:
            os.close(descriptor)
    observed = _external_object_metadata(
        checkpoint_core_store_root,
        object_key,
        prefix=CHECKPOINT_CORE_OBJECT_PREFIX,
        label="checkpoint core",
        required_signature=CHECKPOINT_CORE_MAGIC[:4],
    )
    expected = (len(payload), hashlib.sha256(payload).hexdigest())
    if observed != expected:
        raise V18Error("checkpoint core bytes changed immediately after seal")
    return observed


def _read_installed_directory_file(
    directory_fd: int,
    file_name: str,
    *,
    label: str,
) -> bytes:
    try:
        descriptor = os.open(
            file_name,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise V18Error(f"{label} cannot be pinned") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise V18Error(f"{label} is not a single-link regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if any(
            getattr(before, field) != getattr(after, field)
            for field in (
                "st_dev",
                "st_ino",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        ) or after.st_nlink != 1:
            raise V18Error(f"{label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _existing_checkpoint_session_payloads(
    parent_fd: int,
    *,
    session_name: str,
    file_names: Sequence[str],
    file_mode: int,
    directory_mode: int,
    label: str,
) -> dict[str, bytes] | None:
    """Secure-read a fully published pair, or return None when absent."""

    try:
        directory_fd = os.open(
            session_name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise V18Error(f"{label} final session cannot be pinned") from exc
    try:
        metadata = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != directory_mode
            or set(os.listdir(directory_fd)) != set(file_names)
        ):
            raise V18Error(f"{label} final session is not the exact pair")
        payloads: dict[str, bytes] = {}
        for file_name in file_names:
            child = os.stat(file_name, dir_fd=directory_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(child.st_mode)
                or child.st_uid != os.geteuid()
                or child.st_nlink != 1
                or stat.S_IMODE(child.st_mode) != file_mode
            ):
                raise V18Error(f"{label} final child metadata changed")
            payloads[file_name] = _read_installed_directory_file(
                directory_fd, file_name, label=f"{label} final {file_name}"
            )
        return payloads
    finally:
        os.close(directory_fd)


def _install_checkpoint_session_directory(
    parent_fd: int,
    *,
    session_name: str,
    payloads: Mapping[str, bytes],
    file_mode: int,
    directory_mode: int,
    label: str,
) -> dict[str, tuple[int, str]]:
    """Atomically publish one fixed pair directory, with pre-link recovery."""

    names = tuple(payloads)
    suffixes = {Path(name).suffix for name in names}
    if suffixes == {".bin"}:
        expected_names = tuple(f"{role}.bin" for role in CHECKPOINT_ROLES)
    elif suffixes == {".json"}:
        expected_names = tuple(f"{role}.json" for role in CHECKPOINT_ROLES)
    else:
        expected_names = ()
    if names != expected_names or len(set(names)) != 2:
        raise V18Error(f"{label} pair must contain exactly two distinct files")
    stage_name = f".{session_name}.staging"
    fcntl.flock(parent_fd, fcntl.LOCK_EX)
    try:
        parent_stat = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(parent_stat.st_mode)
            or parent_stat.st_uid != os.geteuid()
            or stat.S_IMODE(parent_stat.st_mode) & 0o022
        ):
            raise V18Error(f"{label} parent owner/permissions are not private")
        try:
            existing = os.stat(
                session_name, dir_fd=parent_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if not stat.S_ISDIR(existing.st_mode):
                raise V18Error(f"{label} final session is not a directory")
            _revalidate_checkpoint_session_directory(
                parent_fd,
                session_name=session_name,
                payloads=payloads,
                file_mode=file_mode,
                directory_mode=directory_mode,
                label=label,
            )
            # Heal the only durable post-rename/pre-parent-fsync crash window.
            os.fsync(parent_fd)
            return {
                file_name: (len(payload), hashlib.sha256(payload).hexdigest())
                for file_name, payload in payloads.items()
            }

        # A staged directory has no registered final name and is explicitly
        # nonauthority.  Under the private parent lock, securely discard only
        # its known single-link regular children before rebuilding fresh
        # timestamp/nonce payloads.
        try:
            stage_fd = os.open(
                stage_name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            stage_fd = -1
        except OSError as exc:
            raise V18Error(f"{label} staged directory cannot be pinned") from exc
        if stage_fd >= 0:
            try:
                staged_stat = os.fstat(stage_fd)
                staged_names = set(os.listdir(stage_fd))
                if (
                    not stat.S_ISDIR(staged_stat.st_mode)
                    or staged_stat.st_uid != os.geteuid()
                    or stat.S_IMODE(staged_stat.st_mode) != directory_mode
                    or not staged_names.issubset(set(names))
                ):
                    raise V18Error(f"{label} staged directory is not recoverable")
                for file_name in sorted(staged_names):
                    metadata = os.stat(
                        file_name, dir_fd=stage_fd, follow_symlinks=False
                    )
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or metadata.st_uid != os.geteuid()
                        or metadata.st_nlink != 1
                        or stat.S_IMODE(metadata.st_mode) != file_mode
                    ):
                        raise V18Error(f"{label} staged child is not recoverable")
                    os.unlink(file_name, dir_fd=stage_fd)
                os.fsync(stage_fd)
            finally:
                os.close(stage_fd)
            entry = os.stat(stage_name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(entry.st_mode):
                raise V18Error(f"{label} staged entry changed before cleanup")
            os.rmdir(stage_name, dir_fd=parent_fd)
            os.fsync(parent_fd)

        os.mkdir(stage_name, mode=directory_mode, dir_fd=parent_fd)
        stage_fd = os.open(
            stage_name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            os.fchmod(stage_fd, directory_mode)
            for file_name in names:
                payload = payloads[file_name]
                descriptor = os.open(
                    file_name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_CLOEXEC
                    | getattr(os, "O_NOFOLLOW", 0),
                    file_mode,
                    dir_fd=stage_fd,
                )
                try:
                    os.fchmod(descriptor, file_mode)
                    remaining = memoryview(payload)
                    while remaining:
                        written = os.write(descriptor, remaining)
                        if written <= 0:  # pragma: no cover - kernel invariant
                            raise OSError("short checkpoint staged-pair write")
                        remaining = remaining[written:]
                    os.fsync(descriptor)
                    metadata = os.fstat(descriptor)
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or metadata.st_uid != os.geteuid()
                        or metadata.st_nlink != 1
                        or stat.S_IMODE(metadata.st_mode) != file_mode
                        or metadata.st_size != len(payload)
                    ):
                        raise V18Error(f"{label} staged file metadata changed")
                finally:
                    os.close(descriptor)
            os.fsync(stage_fd)
            os.fsync(parent_fd)
            if set(os.listdir(stage_fd)) != set(names):
                raise V18Error(f"{label} staged directory is not the exact pair")
            for file_name, expected in payloads.items():
                if _read_installed_directory_file(
                    stage_fd,
                    file_name,
                    label=f"{label} staged {file_name}",
                ) != expected:
                    raise V18Error(f"{label} staged bytes changed")
        finally:
            os.close(stage_fd)
        try:
            os.stat(session_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise V18Error(f"{label} final session appeared before publish")
        # The retained parent is owner-private and held under its advisory
        # lock.  Re-pin the staged directory immediately before rename.  A
        # same-UID process deliberately ignoring the cooperative lock remains
        # outside the registered threat model; every consumer still re-pins
        # the exact installed bytes.
        stage_before_publish = os.stat(
            stage_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(stage_before_publish.st_mode)
            or stage_before_publish.st_uid != os.geteuid()
            or stat.S_IMODE(stage_before_publish.st_mode) != directory_mode
        ):
            raise V18Error(f"{label} staged directory changed before publish")
        os.rename(
            stage_name,
            session_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
        _revalidate_checkpoint_session_directory(
            parent_fd,
            session_name=session_name,
            payloads=payloads,
            file_mode=file_mode,
            directory_mode=directory_mode,
            label=label,
        )
        return {
            file_name: (len(payload), hashlib.sha256(payload).hexdigest())
            for file_name, payload in payloads.items()
        }
    finally:
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_UN)
        except OSError:
            pass


def _revalidate_checkpoint_session_directory(
    parent_fd: int,
    *,
    session_name: str,
    payloads: Mapping[str, bytes],
    file_mode: int,
    directory_mode: int,
    label: str,
) -> None:
    """Reopen one installed pair through its retained parent after both seals."""

    try:
        directory_fd = os.open(
            session_name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise V18Error(f"{label} final pair cannot be pinned") from exc
    try:
        directory_stat = os.fstat(directory_fd)
        observed_names = set(os.listdir(directory_fd))
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or directory_stat.st_uid != os.geteuid()
            or stat.S_IMODE(directory_stat.st_mode) != directory_mode
            or observed_names != set(payloads)
        ):
            raise V18Error(f"{label} final directory metadata changed")
        for file_name, expected in payloads.items():
            metadata = os.stat(file_name, dir_fd=directory_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != file_mode
                or metadata.st_size != len(expected)
            ):
                raise V18Error(f"{label} final file metadata changed")
            if _read_installed_directory_file(
                directory_fd,
                file_name,
                label=f"{label} final {file_name}",
            ) != expected:
                raise V18Error(f"{label} final bytes changed")
        entry_after = os.stat(
            session_name, dir_fd=parent_fd, follow_symlinks=False
        )
        directory_after = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(entry_after.st_mode)
            or (entry_after.st_dev, entry_after.st_ino)
            != (directory_after.st_dev, directory_after.st_ino)
        ):
            raise V18Error(f"{label} final directory entry changed")
    finally:
        os.close(directory_fd)


def _validate_checkpoint_proposal_pair(
    proposals: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(proposals) != set(CHECKPOINT_ROLES):
        raise V18Error("checkpoint proposal pair roles changed")
    safety = proposals["safety_cash"]
    primary = proposals["primary"]
    varying = {
        "checkpoint_id",
        "publication_ordinal",
        "checkpoint_role",
        "sealed_core_object_key",
        "sealed_core_sha256",
        "proposal_sha256",
    }
    common = set(CHECKPOINT_PROPOSAL_FIELDS) - varying
    if any(safety[field] != primary[field] for field in common):
        raise V18Error("checkpoint proposal pair common fields differ")
    if (
        safety["publication_ordinal"] != 0
        or primary["publication_ordinal"] != 1
        or safety["checkpoint_batch_id"] != primary["checkpoint_batch_id"]
        or safety["sealed_core_sha256"] == primary["sealed_core_sha256"]
    ):
        raise V18Error("checkpoint proposal pair identity changed")


def _month_seal_session(
    target_month: pd.Period,
    *,
    first_counted_session_value: Any | None = None,
) -> pd.Timestamp:
    if first_counted_session_value is None:
        return _first_registered_session_in_month(target_month)
    first = _date(first_counted_session_value, "first_counted_session")
    if target_month == first.to_period("M"):
        return first
    return _first_registered_session_in_month(target_month)


def build_state_manifest(
    state: Mapping[str, Any],
    *,
    created_at: Any,
    c00_fold_manifest_sha256: str | None,
    fold_model_bundle_file_sha256: str | None,
    activation_payload_sha256: str,
    activation_receipt_sha256: str,
    first_counted_session_value: Any | None = None,
    activation_observed_at: Any | None = None,
) -> dict[str, Any]:
    created = _timestamp(created_at, "state created_at")
    target = _month(state.get("target_month"), "state target_month")
    seal_session = _month_seal_session(
        target, first_counted_session_value=first_counted_session_value
    )
    if created > _cutoff(seal_session):
        raise V18Error("state manifest was not sealed before the first month cutoff")
    if activation_observed_at is not None and created < _timestamp(
        activation_observed_at, "activation_receipt_commit_observed_at"
    ):
        raise V18Error("state manifest creation timestamp predates activation observation")
    if first_counted_session_value is not None:
        _, actual_payload_hash, _, actual_receipt_hash = (
            validate_canonical_activation_artifacts()
        )
        if (
            activation_payload_sha256 != actual_payload_hash
            or activation_receipt_sha256 != actual_receipt_hash
        ):
            raise V18Error("state manifest activation hashes differ from canonical artifacts")
    source_months = state.get("source_months")
    if not isinstance(source_months, list) or len(source_months) != 3:
        raise V18Error("state must contain exactly three source months")
    if c00_fold_manifest_sha256 is None or fold_model_bundle_file_sha256 is None:
        raise V18Error("counted monthly state requires its exact fold/bundle pair")
    counts = [int(item["complete_pairs"]) for item in source_months]
    medians = [item["median_rank1_minus_rank2_pct"] for item in source_months]
    value = {
        "schema_version": 1,
        "target_month": str(target),
        "created_at": created,
        "three_prior_calendar_months": list(state["required_source_months"]),
        "three_complete_pair_day_counts": counts,
        "three_month_medians_pct": medians,
        "state_available": state.get("state_value_pct") is not None,
        "state_value_pct": state.get("state_value_pct"),
        "selected_source_rank": state.get("selected_source_rank"),
        "c00_fold_manifest_sha256": _require_sha(
            c00_fold_manifest_sha256, "c00_fold_manifest_sha256"
        ),
        "fold_model_bundle_file_sha256": _require_sha(
            fold_model_bundle_file_sha256, "fold_model_bundle_file_sha256"
        ),
        "protocol_sha256": PROTOCOL_SHA256,
        "activation_payload_sha256": _require_sha(
            activation_payload_sha256, "activation_payload_sha256"
        ),
        "activation_receipt_sha256": _require_sha(
            activation_receipt_sha256, "activation_receipt_sha256"
        ),
    }
    value["state_manifest_sha256"] = canonical_json_sha256(
        value, exclude_fields={"state_manifest_sha256"}
    )
    return validate_state_manifest(value)


def validate_state_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    required = set(
        protocol["state_contract"]["target_month_state_manifest_required_fields"]
    )
    if set(manifest) != required:
        raise V18Error("state manifest fields differ from protocol")
    value = dict(manifest)
    target = _month(value["target_month"], "state target_month")
    _timestamp(value["created_at"], "state created_at")
    expected_months = [str(target - offset) for offset in (3, 2, 1)]
    if value["three_prior_calendar_months"] != expected_months:
        raise V18Error("state manifest months are not the immediate prior three")
    counts = value["three_complete_pair_day_counts"]
    medians = value["three_month_medians_pct"]
    if not isinstance(counts, list) or len(counts) != 3:
        raise V18Error("state manifest counts are invalid")
    if not isinstance(medians, list) or len(medians) != 3:
        raise V18Error("state manifest medians are invalid")
    counts = [int(item) for item in counts]
    finite_medians: list[float | None] = []
    for item in medians:
        if item is None:
            finite_medians.append(None)
        else:
            number = float(item)
            if not math.isfinite(number):
                raise V18Error("state manifest median is non-finite")
            finite_medians.append(number)
    available = all(item >= MIN_COMPLETE_PAIRS for item in counts) and all(
        item is not None for item in finite_medians
    )
    if _strict_bool(value["state_available"]) != available:
        raise V18Error("state availability does not recompute")
    state_value = (
        float(np.median(np.asarray(finite_medians, dtype=float))) if available else None
    )
    if value["state_value_pct"] != state_value:
        raise V18Error("state value does not recompute")
    expected_rank = (
        1 if state_value is not None and state_value > 0 else 2
        if state_value is not None and state_value < 0 else None
    )
    if value["selected_source_rank"] != expected_rank:
        raise V18Error("state selected source rank does not recompute")
    fold_hash = value["c00_fold_manifest_sha256"]
    bundle_file_hash = value["fold_model_bundle_file_sha256"]
    if fold_hash is None or bundle_file_hash is None:
        raise V18Error("counted monthly state lacks its fold/bundle pair")
    _require_sha(fold_hash, "c00_fold_manifest_sha256")
    _require_sha(bundle_file_hash, "fold_model_bundle_file_sha256")
    if fold_hash == ZERO_SHA256 or bundle_file_hash == ZERO_SHA256:
        raise V18Error("all-zero SHA is not a missing fold sentinel")
    for field in (
        "protocol_sha256",
        "activation_payload_sha256",
        "activation_receipt_sha256",
    ):
        _require_sha(value[field], field)
    if value["protocol_sha256"] != PROTOCOL_SHA256:
        raise V18Error("state protocol hash changed")
    expected_hash = canonical_json_sha256(
        value, exclude_fields={"state_manifest_sha256"}
    )
    if value["state_manifest_sha256"] != expected_hash:
        raise V18Error("state manifest self-hash mismatch")
    return value


def _parse_jsonl_text(payload: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.splitlines(), start=1):
        if not raw.strip():
            raise V18Error(f"blank JSONL line {line_number}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise V18Error(f"invalid JSONL line {line_number}") from exc
        if not isinstance(value, dict):
            raise V18Error(f"JSONL line {line_number} must be an object")
        records.append(value)
    return records


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    try:
        payload = _plain_file_bytes(target, label="JSONL ledger").decode("utf-8")
    except FileNotFoundError:
        return []
    except UnicodeDecodeError as exc:
        raise V18Error("JSONL ledger is not UTF-8") from exc
    return _parse_jsonl_text(payload)


def _records(value: Sequence[Mapping[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return [dict(item) for item in value.to_dict(orient="records")]
    return [dict(item) for item in value]


def validate_hash_chain(
    records: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    required_fields: Sequence[str],
    exact_fields: bool = True,
) -> list[dict[str, Any]]:
    rows = _records(records)
    expected_fields = set(required_fields)
    previous = ZERO_SHA256
    normalised: list[dict[str, Any]] = []
    for sequence, row in enumerate(rows):
        if exact_fields and set(row) != expected_fields:
            raise V18Error(f"ledger record {sequence} fields differ from protocol")
        missing = sorted(expected_fields - set(row))
        if missing:
            raise V18Error(f"ledger record {sequence} missing fields: {missing}")
        if int(row["sequence_number"]) != sequence:
            raise V18Error("ledger sequence is not zero-based and contiguous")
        if row["previous_record_sha256"] != previous:
            raise V18Error("ledger predecessor chain changed")
        expected = canonical_json_sha256(row, exclude_fields={"record_sha256"})
        if row["record_sha256"] != expected:
            raise V18Error("ledger record self-hash mismatch")
        previous = expected
        normalised.append(row)
    return normalised


def _sealed_record(
    payload: Mapping[str, Any],
    *,
    sequence_number: int,
    previous_record_sha256: str,
) -> dict[str, Any]:
    value = dict(payload)
    value["sequence_number"] = int(sequence_number)
    value["previous_record_sha256"] = previous_record_sha256
    value["record_sha256"] = canonical_json_sha256(
        value, exclude_fields={"record_sha256"}
    )
    return value


def _jsonl_record_authority_spec(
    path: str | Path,
    required_fields: Sequence[str],
) -> tuple[Path, str]:
    target = Path(os.path.abspath(os.fspath(path)))
    canonical_specs = {
        Path(os.path.abspath(os.fspath(DECISION_LEDGER))): (
            DECISION_RECORD_DIR,
            "session_date",
        ),
        Path(os.path.abspath(os.fspath(OUTCOME_LEDGER))): (
            OUTCOME_RECORD_DIR,
            "session_date",
        ),
        Path(os.path.abspath(os.fspath(COMPLETED_MONTH_LEDGER))): (
            COMPLETED_MONTH_RECORD_DIR,
            "completed_month",
        ),
    }
    if target in canonical_specs:
        return canonical_specs[target]
    field_set = set(required_fields)
    if "completed_month" in field_set:
        key_field = "completed_month"
    elif "session_date" in field_set:
        key_field = "session_date"
    else:
        raise V18Error("JSONL record authority lacks a registered key field")
    return target.parent / f"{target.name}.records", key_field


def _jsonl_record_file_name(record: Mapping[str, Any], *, key_field: str) -> str:
    token = str(record.get(key_field, ""))
    pattern = r"\d{4}-\d{2}" if key_field == "completed_month" else r"\d{4}-\d{2}-\d{2}"
    if re.fullmatch(pattern, token) is None:
        raise V18Error("JSONL authority record key is not canonical")
    return f"{token}.json"


def load_jsonl_record_authority(
    path: str | Path,
    *,
    required_fields: Sequence[str],
    validator: Any,
    heal_derived: bool,
) -> list[dict[str, Any]]:
    """Load immutable record shards and exact-compare/heal their JSONL view."""

    target = Path(path)
    authority_directory, key_field = _jsonl_record_authority_spec(
        target, required_fields
    )
    records: list[dict[str, Any]] = []
    if authority_directory.exists():
        if authority_directory.is_symlink() or not authority_directory.is_dir():
            raise V18Error("JSONL record authority directory is not plain")
        directory_stat = os.stat(authority_directory, follow_symlinks=False)
        if (
            directory_stat.st_uid != os.geteuid()
            or stat.S_IMODE(directory_stat.st_mode) != 0o700
        ):
            raise V18Error("JSONL record authority directory is not private")
        entries = sorted(authority_directory.iterdir(), key=lambda item: item.name)
        if heal_derived:
            for stage in entries:
                match = re.fullmatch(r"\.(.+\.json)\.staging", stage.name)
                if match is None:
                    continue
                final = authority_directory / match.group(1)
                if not os.path.lexists(final):
                    continue
                _read_local_authority_bytes(
                    final, label="JSONL record authority crash recovery"
                )
            entries = sorted(authority_directory.iterdir(), key=lambda item: item.name)
        finals: list[Path] = []
        for entry in entries:
            if re.fullmatch(r"\..+\.json\.staging", entry.name):
                # An unpublished local staging name is explicitly nonauthority.
                # The exact candidate writer either resumes/rebuilds it or heals
                # its proven final hard-link on the next append.
                continue
            if re.fullmatch(r"(?:\d{4}-\d{2}|\d{4}-\d{2}-\d{2})\.json", entry.name) is None:
                raise V18Error("JSONL record authority has an unregistered entry")
            if entry.is_symlink() or not entry.is_file():
                raise V18Error("JSONL record authority contains a non-plain file")
            metadata = os.stat(entry, follow_symlinks=False)
            if (
                metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
            ):
                raise V18Error("JSONL record authority owner/mode/link changed")
            finals.append(entry)
        for entry in finals:
            payload = _plain_file_bytes(entry, label="JSONL record authority shard")
            try:
                value = json.loads(payload.decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise V18Error("JSONL record authority shard is not strict JSON") from exc
            if (
                not isinstance(value, dict)
                or payload != canonical_json_bytes(value) + b"\n"
                or entry.name != _jsonl_record_file_name(value, key_field=key_field)
            ):
                raise V18Error("JSONL record authority shard bytes/key are noncanonical")
            records.append(value)
    records.sort(key=lambda item: int(item.get("sequence_number", -1)))
    validated = validator(records)
    if canonical_json_bytes(validated) != canonical_json_bytes(records):
        raise V18Error("JSONL record authority validator changed retained records")
    expected_payload = b"".join(
        canonical_json_bytes(item) + b"\n" for item in validated
    )
    if heal_derived:
        if expected_payload:
            _atomic_local_derived_bytes(target, expected_payload)
        elif os.path.lexists(target):
            raise V18Error("JSONL derived ledger exists without record authority")
    elif expected_payload:
        if target.is_symlink() or _plain_file_bytes(
            target, label="derived canonical JSONL ledger"
        ) != expected_payload:
            raise V18Error("derived JSONL ledger differs from record authority")
    elif os.path.lexists(target):
        raise V18Error("derived JSONL ledger exists without record authority")
    return records


def _load_decision_record_authority(*, heal_derived: bool) -> list[dict[str, Any]]:
    return load_jsonl_record_authority(
        DECISION_LEDGER,
        required_fields=DECISION_FIELDS,
        validator=validate_decision_records,
        heal_derived=heal_derived,
    )


def _load_outcome_record_authority(
    decisions: Sequence[Mapping[str, Any]],
    *,
    heal_derived: bool,
) -> list[dict[str, Any]]:
    return load_jsonl_record_authority(
        OUTCOME_LEDGER,
        required_fields=OUTCOME_FIELDS,
        validator=lambda rows: validate_outcome_records(rows, decisions),
        heal_derived=heal_derived,
    )


def _load_completed_month_record_authority(
    *, heal_derived: bool
) -> list[dict[str, Any]]:
    return load_jsonl_record_authority(
        COMPLETED_MONTH_LEDGER,
        required_fields=COMPLETED_MONTH_FIELDS,
        validator=validate_completed_month_records,
        heal_derived=heal_derived,
    )


def append_jsonl_record(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    required_fields: Sequence[str],
    validator: Any | None = None,
) -> dict[str, Any]:
    """Seal one immutable record shard, then rebuild the recoverable JSONL view."""

    target = Path(path)
    active_validator = (
        (lambda rows: validate_hash_chain(rows, required_fields=required_fields))
        if validator is None
        else validator
    )
    existing = load_jsonl_record_authority(
        target,
        required_fields=required_fields,
        validator=active_validator,
        heal_derived=True,
    )
    candidate = dict(record)
    authority_directory, key_field = _jsonl_record_authority_spec(
        target, required_fields
    )
    candidate_name = _jsonl_record_file_name(candidate, key_field=key_field)
    if existing and candidate_name == _jsonl_record_file_name(
        existing[-1], key_field=key_field
    ):
        if canonical_json_bytes(candidate) != canonical_json_bytes(existing[-1]):
            raise V18Error("append record key conflicts with retained authority")
        return existing[-1]
    expected_sequence = len(existing)
    if int(candidate.get("sequence_number", -1)) != expected_sequence:
        raise V18Error("append record sequence does not follow record authority")
    expected_previous = existing[-1]["record_sha256"] if existing else ZERO_SHA256
    if candidate.get("previous_record_sha256") != expected_previous:
        raise V18Error("append record predecessor does not follow record authority")
    combined = active_validator([*existing, candidate])
    canonical_authorities = {
        Path(os.path.abspath(os.fspath(item)))
        for item in (
            DECISION_RECORD_DIR,
            OUTCOME_RECORD_DIR,
            COMPLETED_MONTH_RECORD_DIR,
        )
    }
    canonical_authority = Path(os.path.abspath(os.fspath(authority_directory)))
    if canonical_authority in canonical_authorities:
        if authority_directory.is_symlink() or not authority_directory.is_dir():
            raise V18Error("JSONL record authority readiness directory is missing")
    else:
        authority_directory.mkdir(parents=True, exist_ok=True)
    shard_path = authority_directory / candidate_name
    _atomic_local_bytes_once(shard_path, canonical_json_bytes(candidate) + b"\n")
    retained = load_jsonl_record_authority(
        target,
        required_fields=required_fields,
        validator=active_validator,
        heal_derived=True,
    )
    if canonical_json_bytes(retained) != canonical_json_bytes(combined):
        raise V18Error("JSONL record authority changed during append")
    return retained[-1]


def semantic_decision_hash(
    records: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> str:
    """Hash ordered outcome-free decisions; never silently drop an outcome."""

    rows = _records(records)
    for record in rows:
        forbidden = FORBIDDEN_DECISION_COLUMNS & set(record)
        if forbidden:
            raise V18Error(f"decision ledger contains outcome fields: {sorted(forbidden)}")
    ordered = sorted(
        rows,
        key=lambda item: (int(item["sequence_number"]), str(item["session_date"])),
    )
    return hashlib.sha256(
        b"".join(canonical_json_bytes(item) + b"\n" for item in ordered)
    ).hexdigest()


def _finite_or_none(value: Any, name: str) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise V18Error(f"{name} must be finite or null")
    return result


def validate_decision_records(
    records: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = validate_hash_chain(records, required_fields=DECISION_FIELDS)
    previous_date: pd.Timestamp | None = None
    month_state: dict[str, tuple[Any, ...]] = {}
    activation_identity: bytes | None = None
    score_session_bindings: list[dict[str, Any]] = []
    for row in rows:
        forbidden = FORBIDDEN_DECISION_COLUMNS & set(row)
        if forbidden:
            raise V18Error(f"decision ledger contains outcome fields: {sorted(forbidden)}")
        if (
            row["schema_version"] != 1
            or row["protocol_id"] != PROTOCOL_ID
            or row["protocol_sha256"] != PROTOCOL_SHA256
            or row["candidate_id"] != CANDIDATE_ID
        ):
            raise V18Error("decision identity changed")
        if row["runtime_lock_sha256"] != RUNTIME_LOCK_SHA256:
            raise V18Error("decision runtime-lock binding changed")
        for field in (
            "runner_sha256",
            "activation_payload_sha256",
            "activation_receipt_sha256",
        ):
            _require_sha(row[field], f"decision {field}")
            if row[field] == ZERO_SHA256:
                raise V18Error(f"decision {field} cannot use all-zero sentinel")
        resolution = str(row["checkpoint_resolution"])
        if resolution not in CHECKPOINT_RESOLUTIONS:
            raise V18Error("decision checkpoint resolution is not registered")
        if row["checkpoint_resolution_reason"] not in CHECKPOINT_RESOLUTION_REASONS:
            raise V18Error("decision checkpoint resolution reason is not registered")
        if row["checkpoint_resolution_reason"] != "primary_commitment_timely":
            raise V18Error("primary checkpoint resolution reason changed")
        for field in ("source_manifest_sha256", "state_manifest_sha256"):
            _require_sha(row[field], f"decision {field}")
            if row[field] == ZERO_SHA256:
                raise V18Error(f"decision {field} cannot use all-zero sentinel")
        fold_hash = row["c00_fold_manifest_sha256"]
        bundle_file_hash = row["fold_model_bundle_file_sha256"]
        if fold_hash is None or bundle_file_hash is None:
            raise V18Error("counted decision lacks its monthly fold/bundle pair")
        _require_sha(fold_hash, "decision c00_fold_manifest_sha256")
        _require_sha(bundle_file_hash, "decision fold_model_bundle_file_sha256")
        if fold_hash == ZERO_SHA256 or bundle_file_hash == ZERO_SHA256:
            raise V18Error("decision fold hashes cannot use all-zero sentinel")
        _validate_git_commit(
            row["activation_receipt_commit_sha"],
            row["activation_receipt_commit_url"],
            "activation_receipt_commit_sha",
        )
        committed = _timestamp(
            row["activation_receipt_commit_committed_at"],
            "activation_receipt_commit_committed_at",
        )
        observed = _timestamp(
            row["activation_receipt_commit_observed_at"],
            "activation_receipt_commit_observed_at",
        )
        if observed < committed:
            raise V18Error("decision activation observation predates commit")
        commit_observation, branch_observation = _validate_github_observation_pair(
            row["receipt_commit_observation"],
            row["receipt_branch_observation"],
            commit_sha=str(row["activation_receipt_commit_sha"]),
            branch=str(read_json(PROTOCOL)["branch"]),
        )
        workflow_observation = validate_github_workflow_observation(
            row["receipt_workflow_run_observation"],
            expected_head_sha=str(row["activation_receipt_commit_sha"]),
        )
        workflow_projection = workflow_observation["canonical_projection"]
        workflow_updated = _timestamp(
            row["activation_receipt_workflow_run_updated_at"],
            "activation_receipt_workflow_run_updated_at",
        )
        workflow_observed = _timestamp(
            row["activation_receipt_workflow_run_observed_at"],
            "activation_receipt_workflow_run_observed_at",
        )
        if (
            commit_observation["canonical_projection"]["html_url"]
            != row["activation_receipt_commit_url"]
            or _timestamp(
                commit_observation["canonical_projection"]["committer_date"],
                "decision receipt committer_date",
            )
            != committed
            or branch_observation["canonical_projection"]["commit_sha"]
            != row["branch_tip_sha_when_receipt_observed"]
            or max(
                _timestamp(commit_observation["retrieved_at"], "commit retrieved_at"),
                _timestamp(branch_observation["retrieved_at"], "branch retrieved_at"),
            )
            != observed
            or row["activation_receipt_workflow_run_id"]
            != workflow_projection["run_id"]
            or workflow_updated
            != _timestamp(workflow_projection["updated_at"], "workflow updated_at")
            or workflow_observed
            != _timestamp(workflow_observation["retrieved_at"], "workflow retrieved_at")
            or committed
            > _timestamp(workflow_projection["created_at"], "workflow created_at")
        ):
            raise V18Error("decision receipt flat fields differ from observations")
        current_activation_identity = canonical_json_bytes(
            {
                "activation_payload_sha256": row["activation_payload_sha256"],
                "activation_receipt_sha256": row["activation_receipt_sha256"],
                "activation_receipt_file_sha256": row[
                    "activation_receipt_file_sha256"
                ],
                "receipt_commit_observation": commit_observation,
                "receipt_branch_observation": branch_observation,
                "receipt_workflow_run_observation": workflow_observation,
            }
        )
        if activation_identity is None:
            activation_identity = current_activation_identity
        elif current_activation_identity != activation_identity:
            raise V18Error("activation observations changed within decision ledger")
        _require_sha(
            row["activation_receipt_file_sha256"],
            "activation_receipt_file_sha256",
        )
        session = _date(row["session_date"], "decision session_date")
        if previous_date is not None and session <= previous_date:
            raise V18Error("decision sessions are not strictly chronological")
        previous_date = session
        core = validate_checkpoint_decision_core(
            _decision_core_from_row(row),
            session_date=session,
            resolution=resolution,
        )
        score_hashes = (
            row["score_session_file_sha256"],
            row["score_session_semantic_sha256"],
        )
        if _strict_bool(row["model_complete"]):
            if score_hashes[0] is None or score_hashes[1] is None:
                raise V18Error("model-complete decision lacks score-session hashes")
            score_session_bindings.append(
                {
                    "session_date": str(session.date()),
                    "file_sha256": score_hashes[0],
                    "semantic_sha256": score_hashes[1],
                }
            )
        elif score_hashes != (None, None):
            raise V18Error("model-incomplete decision binds a score-session shard")
        if row["score_session_set_sha256"] != canonical_json_sha256(
            score_session_bindings
        ):
            raise V18Error("decision score-session set chain changed")
        expected_checkpoint_core_hash = canonical_json_sha256(core)
        if row["checkpoint_core_sha256"] != expected_checkpoint_core_hash:
            raise V18Error("decision checkpoint core hash changed")
        materialized = _timestamp(
            row["decision_materialized_at"], "decision_materialized_at"
        )
        if materialized < max(observed, workflow_observed):
            raise V18Error("decision materialization predates activation evidence")
        role = "primary" if resolution == "primary" else "safety_cash"
        expected_proposal_path = _checkpoint_proposal_path(
            session, role
        ).relative_to(ROOT).as_posix()
        if row["checkpoint_proposal_path"] != expected_proposal_path:
            raise V18Error("decision checkpoint proposal path changed")
        for field in (
            "checkpoint_core_sha256",
            "checkpoint_proposal_file_sha256",
            "checkpoint_proposal_sha256",
        ):
            _require_sha(row[field], f"decision {field}")
        _validate_git_commit(
            row["checkpoint_commit_sha"],
            row["checkpoint_commit_url"],
            "checkpoint_commit_sha",
        )
        checkpoint_commit, checkpoint_branch = _validate_github_observation_pair(
            row["checkpoint_commit_observation"],
            row["checkpoint_branch_observation"],
            commit_sha=str(row["checkpoint_commit_sha"]),
            branch=str(read_json(PROTOCOL)["branch"]),
        )
        checkpoint_workflow = validate_github_workflow_observation(
            row["checkpoint_workflow_run_observation"],
            expected_head_sha=str(row["checkpoint_commit_sha"]),
            checkpoint=True,
        )
        checkpoint_projection = checkpoint_commit["canonical_projection"]
        checkpoint_workflow_projection = checkpoint_workflow["canonical_projection"]
        checkpoint_observed = max(
            _timestamp(checkpoint_commit["retrieved_at"], "checkpoint commit retrieved"),
            _timestamp(checkpoint_branch["retrieved_at"], "checkpoint branch retrieved"),
        )
        if (
            checkpoint_projection["html_url"] != row["checkpoint_commit_url"]
            or len(checkpoint_projection["parent_shas"]) != 1
            or _timestamp(
                checkpoint_projection["committer_date"],
                "checkpoint commit committed_at",
            )
            != _timestamp(
                row["checkpoint_commit_committed_at"],
                "checkpoint_commit_committed_at",
            )
            or checkpoint_observed
            != _timestamp(
                row["checkpoint_commit_observed_at"],
                "checkpoint_commit_observed_at",
            )
            or checkpoint_branch["canonical_projection"]["commit_sha"]
            != row["checkpoint_branch_tip_sha_when_observed"]
            or checkpoint_branch["canonical_projection"]["commit_sha"]
            != row["checkpoint_commit_sha"]
            or checkpoint_workflow_projection["run_id"]
            != row["checkpoint_workflow_run_id"]
            or _timestamp(
                checkpoint_workflow_projection["updated_at"],
                "checkpoint workflow updated_at",
            )
            != _timestamp(
                row["checkpoint_workflow_run_updated_at"],
                "checkpoint_workflow_run_updated_at",
            )
            or _timestamp(
                row["checkpoint_commit_committed_at"],
                "checkpoint_commit_committed_at",
            )
            > _timestamp(
                checkpoint_workflow_projection["created_at"],
                "checkpoint workflow created_at",
            )
            or _timestamp(
                checkpoint_workflow_projection["created_at"],
                "checkpoint workflow created_at",
            )
            >= _cutoff(session)
            or _timestamp(
                checkpoint_workflow["retrieved_at"],
                "checkpoint workflow retrieved_at",
            )
            != _timestamp(
                row["checkpoint_workflow_run_observed_at"],
                "checkpoint_workflow_run_observed_at",
            )
            or materialized
            < max(
                checkpoint_observed,
                _timestamp(
                    checkpoint_workflow["retrieved_at"],
                    "checkpoint workflow retrieved_at",
                ),
            )
            or materialized < _timestamp(row["computed_at"], "computed_at")
        ):
            raise V18Error("decision checkpoint evidence does not recompute")
        cutoff = _timestamp(row["decision_cutoff"], "decision_cutoff")
        computed = _timestamp(row["computed_at"], "computed_at")
        runtime_verified = _timestamp(
            row["runtime_lock_verified_at"], "decision runtime_lock_verified_at"
        )
        if (
            cutoff != _cutoff(session)
            or computed > cutoff
            or computed < observed
            or computed < workflow_observed
            or computed < runtime_verified
        ):
            raise V18Error("decision was not sealed by the registered cutoff")
        source_complete = _strict_bool(row["source_complete"])
        model_complete = _strict_bool(row["model_complete"])
        if not source_complete or not model_complete:
            raise V18Error(
                "counted decision contains incomplete source/model authority"
            )
        state_available = _strict_bool(row["state_available"])
        months = row["three_prior_calendar_months"]
        counts = row["three_complete_pair_day_counts"]
        medians = row["three_month_medians_pct"]
        target_month = session.to_period("M")
        if months != [str(target_month - offset) for offset in (3, 2, 1)]:
            raise V18Error("decision state months are not the immediate prior three")
        if not isinstance(counts, list) or len(counts) != 3:
            raise V18Error("decision state counts are invalid")
        if not isinstance(medians, list) or len(medians) != 3:
            raise V18Error("decision state medians are invalid")
        state_value = _finite_or_none(row["state_value_pct"], "state_value_pct")
        selected_rank = row["selected_source_rank"]
        selected_rank = None if selected_rank is None else int(selected_rank)
        expected_available = all(int(item) >= MIN_COMPLETE_PAIRS for item in counts) and all(
            _finite_or_none(item, "state monthly median") is not None for item in medians
        )
        if state_available != expected_available:
            raise V18Error("decision state availability differs from monthly inputs")
        expected_state = (
            float(np.median(np.asarray(medians, dtype=float)))
            if expected_available
            else None
        )
        if state_value != expected_state:
            raise V18Error("decision state value differs from monthly inputs")
        state_rank = (
            1 if expected_state is not None and expected_state > 0 else 2
            if expected_state is not None and expected_state < 0 else None
        )
        expected_record_rank = (
            state_rank
            if resolution == "primary" and source_complete and model_complete
            else None
        )
        if selected_rank != expected_record_rank:
            raise V18Error("decision selected rank differs from frozen state")
        rank1_code = row["c00_rank1_code"]
        rank2_code = row["c02_rank2_code"]
        candidate_code = row["candidate_selected_code"]
        if source_complete and model_complete:
            if rank1_code is None or rank2_code is None or str(rank1_code) == str(rank2_code):
                raise V18Error("complete C00 pair must contain two distinct codes")
            score1 = _finite_or_none(row["c00_rank1_score"], "c00_rank1_score")
            score2 = _finite_or_none(row["c02_rank2_score"], "c02_rank2_score")
            if score1 is None or score2 is None:
                raise V18Error("complete C00 pair scores must be finite")
        else:
            if any(
                item is not None
                for item in (rank1_code, rank2_code, row["c00_rank1_score"], row["c02_rank2_score"])
            ):
                raise V18Error("incomplete source/model decision exposed a frozen pair")
        decision = str(row["decision"])
        failure_reason = row["failure_reason"]
        if not state_available:
            expected_decision = "cash_state_unavailable"
            expected_code = None
        elif state_value == 0.0:
            expected_decision = "cash_state_zero"
            expected_code = None
        elif state_rank == 1:
            expected_decision = "selected_rank1"
            expected_code = rank1_code
        else:
            expected_decision = "selected_rank2"
            expected_code = rank2_code
        if decision != expected_decision or candidate_code != expected_code:
            raise V18Error("decision does not recompute from frozen inputs")
        expected_reason: str | tuple[str, ...] | None
        if decision in {"selected_rank1", "selected_rank2"}:
            expected_reason = None
        elif decision == "cash_state_zero":
            expected_reason = "state_value_exact_zero"
        elif decision == "cash_state_unavailable":
            expected_reason = "state_insufficient_prior_months"
        else:  # pragma: no cover - exhaustive decision registry above
            raise V18Error("decision is outside the registered reason map")
        if isinstance(expected_reason, tuple):
            reason_valid = failure_reason in expected_reason
        else:
            reason_valid = failure_reason == expected_reason
        if not reason_valid:
            raise V18Error("decision failure reason does not match artifact state")
        if decision == "cash_state_zero" and failure_reason != "state_value_exact_zero":
            raise V18Error("cash_state_zero failure reason changed")
        if decision in {
            "cash_state_unavailable",
        }:
            if not isinstance(failure_reason, str) or not failure_reason.strip():
                raise V18Error("cash/fail-closed decision requires a failure reason")
        elif decision.startswith("selected_") and failure_reason is not None:
            raise V18Error("selected decision must not have a failure reason")
        identity = (
            row["state_manifest_sha256"],
            tuple(months),
            tuple(counts),
            tuple(medians),
            expected_available,
            state_value,
            state_rank,
            fold_hash,
            bundle_file_hash,
        )
        prior = month_state.setdefault(str(target_month), identity)
        if identity != prior:
            raise V18Error("target-month state/fold changed intramonth")
    return rows


def build_decision_ledger(
    scores: pd.DataFrame | None,
    *,
    session_date: Any,
    source_manifest: Mapping[str, Any],
    state_manifest: Mapping[str, Any],
    activation: Mapping[str, Any],
    computed_at: Any | None = None,
    capture_operational_timestamp: bool = False,
    runtime_lock_verified_at: Any | None = None,
    existing_records: Sequence[Mapping[str, Any]] | pd.DataFrame = (),
    fold_manifest: Mapping[str, Any] | None = None,
    fold_model_bundle: Mapping[str, Any] | None = None,
    failure_reason: str | None = None,
    predictor_raw_store_root: str | Path | None = None,
    return_decision_core: bool = False,
    checkpoint_binding: Mapping[str, Any] | None = None,
    _resume_exact_runtime: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Build a validated decision-chain prefix with one new outcome-free record."""

    current_runtime_verified = _runtime_verified_timestamp(
        runtime_lock_verified_at,
        allow_historical_replay=_resume_exact_runtime,
    )
    existing = validate_decision_records(existing_records)
    session = _date(session_date, "session_date")
    activation = validate_activation_context(activation)
    actual_payload_hash = activation["activation_payload_sha256"]
    actual_receipt_hash = activation["activation_receipt_sha256"]
    activation_fields = (
        "activation_payload_sha256",
        "activation_receipt_sha256",
        *RECEIPT_OBSERVATION_RUNTIME_FIELDS,
    )
    for prior in existing:
        if any(prior[field] != activation[field] for field in activation_fields):
            raise V18Error("existing decision activation evidence differs from artifacts")
    source, source_hash = validate_source_manifest(
        source_manifest,
        session_date=session,
        predictor_raw_store_root=predictor_raw_store_root,
        first_counted_session_value=activation["first_counted_session"],
        require_predecessor_decision=False,
    )
    first_counted = _date(
        activation["first_counted_session"], "decision first counted session"
    )
    if session == first_counted:
        if (
            source["previous_counted_target_session"] is not None
            or source["previous_counted_source_manifest_sha256"] is not None
        ):
            raise V18Error("first decision source has a predecessor binding")
    elif not existing or (
        source["previous_counted_target_session"] != existing[-1]["session_date"]
        or source["previous_counted_source_manifest_sha256"]
        != existing[-1]["source_manifest_sha256"]
    ):
        raise V18Error("decision source predecessor differs from prior decision")
    state = validate_state_manifest(state_manifest)
    if (
        state["activation_payload_sha256"] != actual_payload_hash
        or state["activation_receipt_sha256"] != actual_receipt_hash
    ):
        raise V18Error("decision state activation hashes differ from artifacts")
    if state["target_month"] != str(session.to_period("M")):
        raise V18Error("state manifest target month differs from session")
    for field in (
        "activation_payload_sha256",
        "activation_receipt_sha256",
        "activation_receipt_commit_sha",
        "activation_receipt_commit_url",
        "activation_receipt_commit_committed_at",
        "activation_receipt_commit_observed_at",
        "branch_tip_sha_when_receipt_observed",
        "activation_receipt_file_sha256",
        "receipt_commit_observation",
        "receipt_branch_observation",
        "activation_receipt_workflow_run_id",
        "activation_receipt_workflow_run_updated_at",
        "activation_receipt_workflow_run_observed_at",
        "receipt_workflow_run_observation",
        "first_counted_session",
        "terminal_session",
    ):
        if field not in activation:
            raise V18Error(f"activation preflight is missing {field}")
    if session < _date(activation["first_counted_session"], "first_counted_session"):
        raise V18Error("decision backfills before activation")
    terminal = _date(activation["terminal_session"], "terminal_session")
    if session > terminal:
        raise V18Error("decision extends beyond deterministic terminal")
    scheduled = load_registered_calendar()
    denominator = scheduled[
        (scheduled >= _date(activation["first_counted_session"], "first_counted_session"))
        & (scheduled <= terminal)
    ]
    expected_session = denominator[len(existing)] if len(existing) < len(denominator) else None
    if expected_session is None or session != expected_session:
        raise V18Error("decision session is not the next registered denominator day")
    if capture_operational_timestamp and computed_at is not None:
        raise V18Error("canonical decision timestamp is captured internally")
    computed = (
        None
        if capture_operational_timestamp
        else _timestamp(computed_at, "computed_at")
        if computed_at is not None
        else None
    )
    if computed is None and not capture_operational_timestamp:
        raise V18Error("pure decision construction requires computed_at")
    source_complete = _strict_bool(source["source_complete"])
    if not source_complete:
        raise V18Error(
            "incomplete/missing predictor source is an experiment integrity abort"
        )
    model_complete = False
    rank1_code = rank2_code = None
    score1 = score2 = None
    score_session_file_sha: str | None = None
    score_session_semantic_sha: str | None = None
    fold_hash: str | None = None
    bundle_file_hash: str | None = None
    if (fold_manifest is None) != (fold_model_bundle is None):
        raise V18Error("decision fold manifest/bundle presence differs")
    if fold_manifest is None or fold_model_bundle is None:
        raise V18Error("missing monthly fold/bundle is an experiment integrity abort")
    fold_hash = validate_fold_manifest(fold_manifest, fold_model_bundle)
    bundle_file_hash = str(fold_manifest["fold_model_bundle_file_sha256"])
    if computed is not None and computed < _timestamp(
        fold_manifest["fit_completed_at"], "fold fit_completed_at"
    ):
        raise V18Error("decision computed before monthly fold completion")
    if state["c00_fold_manifest_sha256"] != fold_hash or state[
        "fold_model_bundle_file_sha256"
    ] != bundle_file_hash:
        raise V18Error("decision fold pair differs from sealed state manifest")
    if scores is not None:
        score_rows = validate_score_rows(scores)
        if computed is not None and computed < _timestamp(
            score_rows["score_generated_at"].iloc[0], "score_generated_at"
        ):
            raise V18Error("decision computed before C00 score generation")
        if score_rows["session_date"].iloc[0] != str(session.date()):
            raise V18Error("score pair target differs from decision")
        if score_rows["source_manifest_sha256"].iloc[0] != source_hash:
            raise V18Error("score pair does not bind source manifest")
        if score_rows["c00_fold_manifest_sha256"].iloc[0] != fold_hash:
            raise V18Error("score pair does not bind fold manifest")
        if state["c00_fold_manifest_sha256"] != fold_hash:
            raise V18Error("state manifest does not bind monthly fold")
        ranked = score_rows.set_index("source_rank")
        rank1_code = str(ranked.loc[1, "code"])
        rank2_code = str(ranked.loc[2, "code"])
        score1 = float(ranked.loc[1, "model_score"])
        score2 = float(ranked.loc[2, "model_score"])
        score_payload = score_rows.to_csv(index=False, lineterminator="\n").encode()
        score_session_file_sha = hashlib.sha256(score_payload).hexdigest()
        score_session_semantic_sha = semantic_score_hash(score_rows)
        model_complete = True
    else:
        raise V18Error("missing exact target score pair is an experiment integrity abort")
    if failure_reason is not None:
        raise V18Error("caller cannot inject a daily failure reason")
    state_available = _strict_bool(state["state_available"])
    state_value = state["state_value_pct"]
    state_rank = state["selected_source_rank"]
    if not state_available:
        decision, candidate_code = "cash_state_unavailable", None
        failure_reason = "state_insufficient_prior_months"
    elif float(state_value) == 0.0:
        decision, candidate_code = "cash_state_zero", None
        failure_reason = "state_value_exact_zero"
    elif state_rank == 1:
        decision, candidate_code = "selected_rank1", rank1_code
    else:
        decision, candidate_code = "selected_rank2", rank2_code
    if decision.startswith("selected_"):
        failure_reason = None
    selected_rank = state_rank
    decision_runtime_verified = _timestamp(
        score_rows["runtime_lock_verified_at"].iloc[0],
        "score runtime_lock_verified_at",
    )
    if capture_operational_timestamp:
        computed = datetime.now(TOKYO)
    assert computed is not None
    prerequisites = [
        decision_runtime_verified,
        _timestamp(
            activation["activation_receipt_commit_observed_at"],
            "activation_receipt_commit_observed_at",
        ),
        _timestamp(state["created_at"], "state created_at"),
        _timestamp(source["sealed_at"], "source manifest sealed_at"),
    ]
    prerequisites.append(_timestamp(fold_manifest["sealed_at"], "fold sealed_at"))
    prerequisites.append(
        _timestamp(score_rows["score_generated_at"].iloc[0], "score_generated_at")
    )
    if computed < max(prerequisites) or computed > _cutoff(session):
        raise V18Error("decision computation violates its timestamp DAG")
    score_session_bindings = [
        {
            "session_date": str(item["session_date"]),
            "file_sha256": item["score_session_file_sha256"],
            "semantic_sha256": item["score_session_semantic_sha256"],
        }
        for item in existing
        if item.get("score_session_file_sha256") is not None
    ]
    score_session_bindings.append(
        {
            "session_date": str(session.date()),
            "file_sha256": score_session_file_sha,
            "semantic_sha256": score_session_semantic_sha,
        }
    )
    score_session_set_sha = canonical_json_sha256(score_session_bindings)
    decision_core = {
        "candidate_id": CANDIDATE_ID,
        "runtime_lock_verified_at": decision_runtime_verified,
        "source_manifest_sha256": source_hash,
        "c00_fold_manifest_sha256": fold_hash,
        "fold_model_bundle_file_sha256": bundle_file_hash,
        "state_manifest_sha256": state["state_manifest_sha256"],
        "score_session_file_sha256": score_session_file_sha,
        "score_session_semantic_sha256": score_session_semantic_sha,
        "score_session_set_sha256": score_session_set_sha,
        "decision_cutoff": _cutoff(session),
        "computed_at": computed,
        "source_complete": source_complete,
        "model_complete": model_complete,
        "state_available": state_available,
        "three_prior_calendar_months": state["three_prior_calendar_months"],
        "three_complete_pair_day_counts": state["three_complete_pair_day_counts"],
        "three_month_medians_pct": state["three_month_medians_pct"],
        "state_value_pct": state_value,
        "selected_source_rank": selected_rank,
        "c00_rank1_code": rank1_code,
        "c00_rank1_score": score1,
        "c02_rank2_code": rank2_code,
        "c02_rank2_score": score2,
        "candidate_selected_code": candidate_code,
        "decision": decision,
        "failure_reason": failure_reason,
    }
    if set(decision_core) != set(CHECKPOINT_CORE_FIELDS):
        raise V18Error("derived checkpoint decision core schema changed")
    if return_decision_core:
        if checkpoint_binding is not None:
            raise V18Error("core-only derivation cannot accept checkpoint evidence")
        return decision_core
    if checkpoint_binding is None or set(checkpoint_binding) != set(
        CHECKPOINT_BINDING_FIELDS
    ):
        raise V18Error("decision requires exact resolved checkpoint evidence")
    payload = {
        "schema_version": 1,
        "session_date": str(session.date()),
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "activation_payload_sha256": activation["activation_payload_sha256"],
        "activation_receipt_sha256": activation["activation_receipt_sha256"],
        "activation_receipt_commit_sha": activation["activation_receipt_commit_sha"],
        "activation_receipt_commit_url": activation["activation_receipt_commit_url"],
        "activation_receipt_commit_committed_at": activation[
            "activation_receipt_commit_committed_at"
        ],
        "activation_receipt_commit_observed_at": activation[
            "activation_receipt_commit_observed_at"
        ],
        "branch_tip_sha_when_receipt_observed": activation[
            "branch_tip_sha_when_receipt_observed"
        ],
        "activation_receipt_file_sha256": activation[
            "activation_receipt_file_sha256"
        ],
        "receipt_commit_observation": activation["receipt_commit_observation"],
        "receipt_branch_observation": activation["receipt_branch_observation"],
        "activation_receipt_workflow_run_id": activation[
            "activation_receipt_workflow_run_id"
        ],
        "activation_receipt_workflow_run_updated_at": activation[
            "activation_receipt_workflow_run_updated_at"
        ],
        "activation_receipt_workflow_run_observed_at": activation[
            "activation_receipt_workflow_run_observed_at"
        ],
        "receipt_workflow_run_observation": activation[
            "receipt_workflow_run_observation"
        ],
        **dict(checkpoint_binding),
        **decision_core,
    }
    previous = existing[-1]["record_sha256"] if existing else ZERO_SHA256
    record = _sealed_record(
        payload,
        sequence_number=len(existing),
        previous_record_sha256=previous,
    )
    return validate_decision_records([*existing, record])


def _activation_context_from_decision(row: Mapping[str, Any]) -> dict[str, Any]:
    first = _date(row["session_date"], "first decision session")
    terminal = deterministic_terminal_session(first)
    calendar = load_registered_calendar()
    denominator = calendar[(calendar >= first) & (calendar <= terminal)]
    return validate_activation_context(
        {
            "activation_payload_sha256": row["activation_payload_sha256"],
            "activation_receipt_sha256": row["activation_receipt_sha256"],
            **{field: row[field] for field in RECEIPT_OBSERVATION_RUNTIME_FIELDS},
            "first_counted_session": str(first.date()),
            "first_counted_predecessor_session": str(
                _latest_required_predictor_source_session(first).date()
            ),
            "terminal_session": str(terminal.date()),
            "terminal_scheduled_sessions": len(denominator),
            "represented_calendar_months": denominator.to_period("M").nunique(),
            "calendar_sha256": CALENDAR_SHA256,
            "production_model_changed": False,
            "orders_allowed": False,
        }
    )


def _discover_activation_context(
    existing_decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not os.path.lexists(ACTIVATION_CONTEXT):
        raise V18Error("canonical activation context is missing")
    canonical = validate_activation_context(read_json(ACTIVATION_CONTEXT))
    if existing_decisions:
        derived = _activation_context_from_decision(existing_decisions[0])
        if canonical_json_bytes(derived) != canonical_json_bytes(canonical):
            raise V18Error("decision prefix differs from canonical activation context")
    return canonical


def _canonical_target_score_pair(session: pd.Timestamp) -> pd.DataFrame | None:
    shard_path = SCORE_SESSION_DIR / f"{session.date()}.csv"
    if not shard_path.is_file() or shard_path.is_symlink():
        return None
    pair = validate_score_rows(
        _read_csv_plain(
            shard_path,
            label="canonical checkpoint score-session shard",
            dtype={"code": "string"},
            float_precision="round_trip",
        )
    )
    scores = _read_csv_plain(
        SCORE_OUTPUT,
        label="canonical checkpoint derived score ledger",
        dtype={"code": "string"},
        float_precision="round_trip",
    )
    ledger = validate_score_ledger(scores, allow_empty=True)
    target = str(session.date())
    selected = ledger.loc[ledger["session_date"].eq(target)].reset_index(drop=True)
    if selected.empty or not validate_score_rows(selected).equals(pair):
        raise V18Error("checkpoint derived score ledger differs from session shard")
    return pair


def _derive_canonical_checkpoint_core(
    session: pd.Timestamp,
    existing: Sequence[Mapping[str, Any]],
    *,
    computed_at: Any | None = None,
    runtime_lock_verified_at: Any | None = None,
    capture_operational_timestamp: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    activation = _discover_activation_context(existing)
    source_path = SOURCE_MANIFEST_DIR / f"{session.date()}.json"
    state_path = STATE_MANIFEST_DIR / f"{session.to_period('M')}.json"
    source = read_json(source_path)
    state = read_json(state_path)
    month = str(session.to_period("M"))
    fold_path = FOLD_MANIFEST_DIR / f"{month}.json"
    bundle_path = FOLD_MODEL_DIR / f"{month}.json"
    if fold_path.exists() != bundle_path.exists():
        raise V18Error("canonical checkpoint fold pair presence differs")
    fold = read_json(fold_path) if fold_path.exists() else None
    bundle = read_json(bundle_path) if bundle_path.exists() else None
    derived = build_decision_ledger(
        _canonical_target_score_pair(session),
        session_date=session,
        source_manifest=source,
        state_manifest=state,
        activation=activation,
        computed_at=computed_at,
        runtime_lock_verified_at=runtime_lock_verified_at,
        capture_operational_timestamp=capture_operational_timestamp,
        existing_records=existing,
        fold_manifest=fold,
        fold_model_bundle=bundle,
        return_decision_core=True,
        _resume_exact_runtime=not capture_operational_timestamp,
    )
    if not isinstance(derived, dict):  # pragma: no cover - explicit API mode
        raise V18Error("checkpoint core derivation returned a decision ledger")
    return derived, activation


def _prepare_private_local_authority_directory(path: Path, *, label: str) -> int:
    """Create/verify one durable empty local authority parent preactivation."""

    target = Path(os.path.abspath(os.fspath(path)))
    parent = target.parent
    if parent.resolve(strict=True) != parent or parent.is_symlink():
        raise V18Error(f"{label} parent must be an existing plain directory")
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
    )
    fcntl.flock(parent_fd, fcntl.LOCK_EX)
    try:
        parent_stat = os.fstat(parent_fd)
        if (
            parent_stat.st_uid != os.geteuid()
            or stat.S_IMODE(parent_stat.st_mode) & 0o022
        ):
            raise V18Error(f"{label} parent owner/permissions are unsafe")
        try:
            observed = os.stat(
                target.name, dir_fd=parent_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            os.mkdir(target.name, mode=0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
            observed = os.stat(
                target.name, dir_fd=parent_fd, follow_symlinks=False
            )
        if (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) != 0o700
        ):
            raise V18Error(f"{label} owner/mode changed")
        directory_fd = os.open(
            target.name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            pinned = os.fstat(directory_fd)
            if (
                (pinned.st_dev, pinned.st_ino)
                != (observed.st_dev, observed.st_ino)
                or pinned.st_uid != os.geteuid()
                or stat.S_IMODE(pinned.st_mode) != 0o700
            ):
                raise V18Error(f"{label} changed while pinned")
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.fsync(parent_fd)
        return stat.S_IMODE(observed.st_mode)
    finally:
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(parent_fd)


def prepare_operational_stores(
    *,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
    outcome_raw_store_root: str | Path,
    checkpoint_core_store_root: str | Path,
) -> dict[str, Any]:
    """Precreate every private daily authority parent before activation."""

    roots = (
        (predictor_raw_store_root, "predictor raw"),
        (predictor_derived_store_root, "predictor derived"),
        (outcome_raw_store_root, "outcome raw"),
        (checkpoint_core_store_root, "checkpoint core"),
    )
    for index, (left, left_label) in enumerate(roots):
        _external_store_root(left, left_label)
        for right, right_label in roots[index + 1 :]:
            _validate_external_root_pair_disjoint(
                left, left_label, right, right_label
            )
    local_directories = (
        CHECKPOINT_PROPOSAL_DIR,
        *_registered_local_authority_directories(),
    )
    # Preflight every existing path and every derived view before creating a
    # missing directory, so stale-run contamination cannot cause a partial
    # readiness mutation.
    for directory in local_directories:
        if os.path.lexists(directory):
            observed = os.stat(directory, follow_symlinks=False)
            if (
                directory.is_symlink()
                or not stat.S_ISDIR(observed.st_mode)
                or observed.st_uid != os.geteuid()
                or stat.S_IMODE(observed.st_mode) != 0o700
                or any(directory.iterdir())
            ):
                raise V18Error(
                    f"local authority readiness path is not private and empty: {directory}"
                )
    for derived in (
        DECISION_LEDGER,
        OUTCOME_LEDGER,
        COMPLETED_MONTH_LEDGER,
        SCORE_OUTPUT,
        PICKS_OUTPUT,
        ACTIVATION_CONTEXT,
        RESULT_OUTPUT,
    ):
        if os.path.lexists(derived):
            raise V18Error(
                f"preactivation derived/result artifact already exists: {derived}"
            )
    local_modes = {
        str(directory.relative_to(ROOT)): _prepare_private_local_authority_directory(
            directory, label=f"local authority readiness {directory.name}"
        )
        for directory in local_directories
    }
    anchor_key = f"{CHECKPOINT_CORE_OBJECT_PREFIX}.pair-install-anchor"
    with _external_parent_fd(
        checkpoint_core_store_root,
        anchor_key,
        prefix=CHECKPOINT_CORE_OBJECT_PREFIX,
        label="checkpoint core readiness",
        create_parents=True,
    ) as (parent_fd, _):
        os.fsync(parent_fd)
        core_parent = os.fstat(parent_fd)
    return {
        "checkpoint_proposal_directory": str(
            CHECKPOINT_PROPOSAL_DIR.relative_to(ROOT)
        ),
        "checkpoint_proposal_directory_mode": local_modes[
            str(CHECKPOINT_PROPOSAL_DIR.relative_to(ROOT))
        ],
        "checkpoint_core_parent_mode": stat.S_IMODE(core_parent.st_mode),
        "local_authority_directories": sorted(local_modes),
        "local_authority_directory_modes": {
            key: local_modes[key] for key in sorted(local_modes)
        },
        "external_roots_pairwise_disjoint": True,
        "production_model_changed": False,
        "orders_allowed": False,
    }


def prepare_checkpoint(
    *,
    session_date: Any,
    checkpoint_core_store_root: str | Path,
) -> dict[str, Any]:
    """Atomically commit the fixed safety/primary pair for one session."""

    if not _STRICT_RUNTIME_ACTIVE:
        raise V18Error("checkpoint preparation requires strict operational runtime")
    _validate_startup_and_module_closure(phase="checkpoint preparation entry")
    target = _date(session_date, "checkpoint session")
    existing = _load_decision_record_authority(heal_derived=True)
    sequence = len(existing)
    previous = existing[-1]["record_sha256"] if existing else ZERO_SHA256
    batch_id = f"model_v18_shoulder_state_checkpoint_batch_{target:%Y%m%d}"

    # Resolve and retain both destination parents before exposing either pair.
    # The external pair is durably installed first; only then is proposal
    # created_at captured and the public pair built.  This makes the registered
    # seal-before-proposal edge true without relying on filesystem mtimes.
    proposal_root = CHECKPOINT_PROPOSAL_DIR.resolve(strict=True)
    if (
        CHECKPOINT_PROPOSAL_DIR.is_symlink()
        or proposal_root != CHECKPOINT_PROPOSAL_DIR
        or not CHECKPOINT_PROPOSAL_DIR.is_dir()
    ):
        raise V18Error("checkpoint proposal root is not a resolved plain directory")
    expected_proposal_root = os.stat(proposal_root, follow_symlinks=False)
    anchor_key = f"{CHECKPOINT_CORE_OBJECT_PREFIX}.pair-install-anchor"
    with _external_parent_fd(
        checkpoint_core_store_root,
        anchor_key,
        prefix=CHECKPOINT_CORE_OBJECT_PREFIX,
        label="checkpoint core",
        create_parents=False,
    ) as (core_parent_fd, _):
        proposal_parent_fd = os.open(
            proposal_root,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            observed_proposal_root = os.fstat(proposal_parent_fd)
            if (
                not stat.S_ISDIR(observed_proposal_root.st_mode)
                or observed_proposal_root.st_uid != os.geteuid()
                or stat.S_IMODE(observed_proposal_root.st_mode) & 0o022
                or (observed_proposal_root.st_dev, observed_proposal_root.st_ino)
                != (expected_proposal_root.st_dev, expected_proposal_root.st_ino)
            ):
                raise V18Error("checkpoint proposal root changed during open")
            # Establish both sides' exact publication state before mutating
            # either side.  A public proposal without its sealed external core
            # can never be repaired safely; a sealed core without a proposal
            # is the one registered resumable crash state.
            retained_core_payloads = _existing_checkpoint_session_payloads(
                core_parent_fd,
                session_name=str(target.date()),
                file_names=tuple(f"{role}.bin" for role in CHECKPOINT_ROLES),
                file_mode=0o600,
                directory_mode=0o700,
                label="checkpoint core",
            )
            retained_proposal_payloads = _existing_checkpoint_session_payloads(
                proposal_parent_fd,
                session_name=str(target.date()),
                file_names=tuple(f"{role}.json" for role in CHECKPOINT_ROLES),
                file_mode=0o644,
                directory_mode=0o755,
                label="checkpoint proposal",
            )
            if (
                retained_proposal_payloads is not None
                and retained_core_payloads is None
            ):
                raise V18Error(
                    "checkpoint proposal exists without its sealed core pair"
                )

            retained_sealed_values: dict[str, dict[str, Any]] = {}
            if retained_core_payloads is not None:
                for role in CHECKPOINT_ROLES:
                    payload_name = f"{role}.bin"
                    sealed_value = parse_checkpoint_core_envelope(
                        retained_core_payloads[payload_name]
                    )
                    retained_static = {
                        "schema_version": 1,
                        "target_session": str(target.date()),
                        "checkpoint_role": role,
                        "decision_sequence_number": sequence,
                        "previous_decision_record_sha256": previous,
                    }
                    if any(
                        sealed_value.get(field) != expected
                        for field, expected in retained_static.items()
                    ) or re.fullmatch(
                        r"[0-9a-f]{64}", str(sealed_value.get("nonce_hex", ""))
                    ) is None:
                        raise V18Error(
                            "checkpoint retained core differs from current exact inputs"
                        )
                    if encode_checkpoint_core_envelope(sealed_value) != (
                        retained_core_payloads[payload_name]
                    ):
                        raise V18Error("checkpoint retained core envelope is noncanonical")
                    retained_sealed_values[role] = sealed_value
                retained_primary = retained_sealed_values["primary"]["decision_core"]
                if retained_sealed_values["safety_cash"]["decision_core"] != (
                    retained_primary
                ):
                    raise V18Error("checkpoint retained role cores differ")
                primary_core, activation = _derive_canonical_checkpoint_core(
                    target,
                    existing,
                    computed_at=retained_primary.get("computed_at"),
                    runtime_lock_verified_at=retained_primary.get(
                        "runtime_lock_verified_at"
                    ),
                    capture_operational_timestamp=False,
                )
            else:
                primary_core, activation = _derive_canonical_checkpoint_core(
                    target,
                    existing,
                )

            # The historical safety_cash role name is an opaque publication
            # predecessor only.  It has no fallback authority and commits the
            # exact same decision core as primary under an independent nonce.
            cores = {
                "safety_cash": dict(primary_core),
                "primary": primary_core,
            }
            for core in cores.values():
                validate_checkpoint_decision_core(
                    core,
                    session_date=target,
                    resolution="primary",
                )
            core_payloads: dict[str, bytes] = {}
            if retained_core_payloads is not None:
                for role in CHECKPOINT_ROLES:
                    sealed_value = retained_sealed_values[role]
                    expected_fields = {
                        "decision_core": cores[role],
                        "decision_core_sha256": canonical_json_sha256(cores[role]),
                    }
                    if any(
                        sealed_value.get(field) != expected
                        for field, expected in expected_fields.items()
                    ):
                        raise V18Error(
                            "checkpoint retained core differs from current exact inputs"
                        )
                core_payloads = retained_core_payloads
            else:
                for role in CHECKPOINT_ROLES:
                    core = cores[role]
                    sealed_value = {
                        "schema_version": 1,
                        "target_session": str(target.date()),
                        "checkpoint_role": role,
                        "decision_sequence_number": sequence,
                        "previous_decision_record_sha256": previous,
                        "nonce_hex": secrets.token_bytes(32).hex(),
                        "decision_core": core,
                        "decision_core_sha256": canonical_json_sha256(core),
                    }
                    core_payloads[f"{role}.bin"] = encode_checkpoint_core_envelope(
                        sealed_value
                    )
            core_metadata = _install_checkpoint_session_directory(
                core_parent_fd,
                session_name=str(target.date()),
                payloads=core_payloads,
                file_mode=0o600,
                directory_mode=0o700,
                label="checkpoint core",
            )
            proposals: dict[str, dict[str, Any]] = {}
            if retained_proposal_payloads is not None:
                proposal_payloads = retained_proposal_payloads
                for role in CHECKPOINT_ROLES:
                    payload_name = f"{role}.json"
                    try:
                        proposal = json.loads(
                            proposal_payloads[payload_name].decode("utf-8")
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise V18Error(
                            "checkpoint retained proposal is not canonical JSON"
                        ) from exc
                    if _json_file_bytes(proposal) != proposal_payloads[payload_name]:
                        raise V18Error("checkpoint retained proposal bytes are noncanonical")
                    validated, _, _ = validate_checkpoint_proposal(
                        proposal,
                        session_date=target,
                        role=role,
                        activation=activation,
                        sequence_number=sequence,
                        previous_record_sha256=previous,
                    )
                    observed_core = core_metadata[f"{role}.bin"]
                    if (
                        int(validated["sealed_core_byte_count"])
                        != observed_core[0]
                        or validated["sealed_core_sha256"] != observed_core[1]
                    ):
                        raise V18Error(
                            "checkpoint retained proposal differs from retained core"
                        )
                    proposals[role] = validated
            else:
                created_at = datetime.now(TOKYO)
                if any(
                    _timestamp(core["computed_at"], "checkpoint core computed_at")
                    > created_at
                    for core in cores.values()
                ):
                    raise V18Error(
                        "checkpoint proposal timestamp predates its sealed core"
                    )
                common_proposal = {
                    "schema_version": 1,
                    "checkpoint_batch_id": batch_id,
                    "repository": "rokuroku-066/TSE-Session-Ranker",
                    "branch": read_json(ACTIVATION_PAYLOAD)["branch"],
                    "protocol_id": PROTOCOL_ID,
                    "protocol_sha256": PROTOCOL_SHA256,
                    "runner_sha256": sha256_file(__file__),
                    "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
                    "activation_payload_sha256": activation[
                        "activation_payload_sha256"
                    ],
                    "activation_receipt_sha256": activation[
                        "activation_receipt_sha256"
                    ],
                    "activation_receipt_commit_sha": activation[
                        "activation_receipt_commit_sha"
                    ],
                    "target_session": str(target.date()),
                    "decision_sequence_number": sequence,
                    "previous_decision_record_sha256": previous,
                    "sealed_core_byte_count": CHECKPOINT_CORE_ENVELOPE_BYTES,
                    "created_at": created_at,
                    "canonical_json_contract": CANONICAL_JSON_CONTRACT,
                }
                proposal_payloads = {}
                for ordinal, role in enumerate(CHECKPOINT_ROLES):
                    object_key = _checkpoint_core_object_key(target, role)
                    observed_core = core_metadata[f"{role}.bin"]
                    if observed_core[0] != CHECKPOINT_CORE_ENVELOPE_BYTES:
                        raise V18Error("checkpoint core envelope byte count changed")
                    proposal = {
                        **common_proposal,
                        "checkpoint_id": (
                            f"model_v18_shoulder_state_checkpoint_{target:%Y%m%d}_{role}"
                        ),
                        "publication_ordinal": ordinal,
                        "checkpoint_role": role,
                        "sealed_core_object_key": object_key,
                        "sealed_core_sha256": observed_core[1],
                    }
                    proposal["proposal_sha256"] = canonical_json_sha256(
                        proposal, exclude_fields={"proposal_sha256"}
                    )
                    validated, _, _ = validate_checkpoint_proposal(
                        proposal,
                        session_date=target,
                        role=role,
                        activation=activation,
                        sequence_number=sequence,
                        previous_record_sha256=previous,
                    )
                    proposals[role] = validated
                    proposal_payloads[f"{role}.json"] = _json_file_bytes(validated)
            _validate_checkpoint_proposal_pair(proposals)
            proposal_metadata = _install_checkpoint_session_directory(
                proposal_parent_fd,
                session_name=str(target.date()),
                payloads=proposal_payloads,
                file_mode=0o644,
                directory_mode=0o755,
                label="checkpoint proposal",
            )
            _revalidate_checkpoint_session_directory(
                core_parent_fd,
                session_name=str(target.date()),
                payloads=core_payloads,
                file_mode=0o600,
                directory_mode=0o700,
                label="checkpoint core",
            )
            _revalidate_checkpoint_session_directory(
                proposal_parent_fd,
                session_name=str(target.date()),
                payloads=proposal_payloads,
                file_mode=0o644,
                directory_mode=0o755,
                label="checkpoint proposal",
            )
        finally:
            os.close(proposal_parent_fd)

    role_metadata: list[dict[str, Any]] = []
    for role in CHECKPOINT_ROLES:
        output = _checkpoint_proposal_path(target, role)
        persisted, _, relative = validate_checkpoint_proposal(
            output,
            session_date=target,
            role=role,
            activation=activation,
            sequence_number=sequence,
            previous_record_sha256=previous,
        )
        if canonical_json_bytes(persisted) != canonical_json_bytes(proposals[role]):
            raise V18Error("installed checkpoint proposal changed after pair seal")
        role_metadata.append(
            {
                "checkpoint_role": role,
                "checkpoint_proposal_path": relative,
                "checkpoint_proposal_file_sha256": proposal_metadata[f"{role}.json"][1],
                "checkpoint_proposal_sha256": persisted["proposal_sha256"],
                "sealed_core_object_key": persisted["sealed_core_object_key"],
                "sealed_core_byte_count": core_metadata[f"{role}.bin"][0],
                "sealed_core_sha256": core_metadata[f"{role}.bin"][1],
            }
        )
    _validate_startup_and_module_closure(phase="checkpoint preparation exit")
    return {
        "checkpoint_batch_id": batch_id,
        "target_session": str(target.date()),
        "role_artifacts": role_metadata,
        "production_model_changed": False,
        "orders_allowed": False,
    }


def _full_git_sha(value: Any, name: str) -> str:
    token = str(value)
    if GIT_SHA_RE.fullmatch(token) is None:
        raise V18Error(f"{name} is not a full lower-case Git SHA")
    return token


def _git_blob_sha1(payload: bytes) -> str:
    header = b"blob " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(header + payload).hexdigest()


def _git_data_ref_projection(body: Mapping[str, Any], *, expected_sha: str) -> dict[str, str]:
    obj = body.get("object")
    if not isinstance(obj, Mapping):
        raise V18Error("Git Data ref response lacks an object")
    projection = {
        "ref": body.get("ref"),
        "object_type": obj.get("type"),
        "object_sha": obj.get("sha"),
        "object_url": obj.get("url"),
    }
    if (
        projection["ref"] != GITHUB_REF
        or projection["object_type"] != "commit"
        or _full_git_sha(projection["object_sha"], "Git Data ref SHA") != expected_sha
        or projection["object_url"]
        != (
            f"https://api.github.com/repos/{GITHUB_REPOSITORY}/git/commits/"
            f"{expected_sha}"
        )
    ):
        raise V18Error("Git Data ref projection differs from registered authority")
    return projection


def _git_data_commit_projection(body: Mapping[str, Any]) -> dict[str, Any]:
    tree = body.get("tree")
    parents = body.get("parents")
    committer = body.get("committer")
    if (
        not isinstance(tree, Mapping)
        or not isinstance(parents, list)
        or any(not isinstance(item, Mapping) for item in parents)
        or not isinstance(committer, Mapping)
    ):
        raise V18Error("Git Data commit response schema is invalid")
    projection = {
        "sha": _full_git_sha(body.get("sha"), "Git Data commit SHA"),
        "message": body.get("message"),
        "tree_sha": _full_git_sha(tree.get("sha"), "Git Data tree SHA"),
        "parent_shas": [
            _full_git_sha(item.get("sha"), "Git Data parent SHA") for item in parents
        ],
        "html_url": body.get("html_url"),
        "committer_date": committer.get("date"),
    }
    if (
        not isinstance(projection["message"], str)
        or not isinstance(projection["html_url"], str)
        or projection["html_url"]
        != f"https://github.com/{GITHUB_REPOSITORY}/commit/{projection['sha']}"
    ):
        raise V18Error("Git Data commit stable projection is invalid")
    _timestamp(projection["committer_date"], "Git Data commit committer_date")
    return projection


def _git_data_tree_projection(body: Mapping[str, Any]) -> dict[str, Any]:
    entries = body.get("tree")
    if body.get("truncated") is not False or not isinstance(entries, list):
        raise V18Error("Git Data recursive tree is missing or truncated")
    projected: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, Mapping):
            raise V18Error("Git Data tree entry is not an object")
        path = str(item.get("path"))
        if (
            not path
            or path.startswith("/")
            or ".." in Path(path).parts
            or item.get("mode") not in {"100644", "100755", "120000", "040000", "160000"}
            or item.get("type") not in {"blob", "tree", "commit"}
        ):
            raise V18Error("Git Data tree entry is unsafe or unregistered")
        size = item.get("size")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise V18Error("Git Data tree entry size is invalid")
        projected.append(
            {
                "path": path,
                "mode": item.get("mode"),
                "type": item.get("type"),
                "sha": _full_git_sha(item.get("sha"), "Git Data tree object SHA"),
                "size": size,
            }
        )
    if len({item["path"] for item in projected}) != len(projected):
        raise V18Error("Git Data tree entries are not unique")
    projected.sort(key=lambda item: item["path"])
    return {
        "sha": _full_git_sha(body.get("sha"), "Git Data recursive tree SHA"),
        "truncated": False,
        "entries": projected,
    }


def _git_data_blob_bytes(body: Mapping[str, Any], *, expected_sha: str) -> bytes:
    sha = _full_git_sha(body.get("sha"), "Git Data blob SHA")
    encoding = body.get("encoding")
    size = body.get("size")
    content = body.get("content")
    if (
        sha != expected_sha
        or encoding != "base64"
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(content, str)
    ):
        raise V18Error("Git Data blob projection is invalid")
    canonical_content = content.replace("\n", "")
    try:
        payload = base64.b64decode(canonical_content, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise V18Error("Git Data blob content is not canonical base64") from exc
    if (
        len(payload) != size
        or base64.b64encode(payload).decode("ascii") != canonical_content
        or _git_blob_sha1(payload) != sha
    ):
        raise V18Error("Git Data blob bytes or object SHA differ")
    return payload


def _git_data_compare_projection(
    body: Mapping[str, Any],
    *,
    expected_commit: str | None = None,
    expected_path: str | None = None,
) -> dict[str, Any]:
    commits = body.get("commits")
    files = body.get("files")
    if not isinstance(commits, list) or not isinstance(files, list):
        raise V18Error("GitHub compare projection is invalid")
    commit_shas = []
    for item in commits:
        if not isinstance(item, Mapping):
            raise V18Error("GitHub compare commit is invalid")
        commit_shas.append(_full_git_sha(item.get("sha"), "compare commit SHA"))
    file_rows: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, Mapping):
            raise V18Error("GitHub compare file is invalid")
        file_rows.append(
            {
                "filename": item.get("filename"),
                "status": item.get("status"),
                "sha": _full_git_sha(item.get("sha"), "compare file SHA"),
                "previous_filename": item.get("previous_filename"),
            }
        )
    projection = {
        "status": body.get("status"),
        "ahead_by": body.get("ahead_by"),
        "behind_by": body.get("behind_by"),
        "total_commits": body.get("total_commits"),
        "commit_shas": commit_shas,
        "files": file_rows,
    }
    for field in ("ahead_by", "behind_by", "total_commits"):
        if isinstance(projection[field], bool) or not isinstance(projection[field], int):
            raise V18Error("GitHub compare count is invalid")
    if expected_commit is not None and (
        projection["status"] != "ahead"
        or projection["ahead_by"] != 1
        or projection["behind_by"] != 0
        or projection["total_commits"] != 1
        or projection["commit_shas"] != [expected_commit]
        or len(file_rows) != 1
        or file_rows[0]
        != {
            "filename": expected_path,
            "status": "added",
            "sha": file_rows[0]["sha"],
            "previous_filename": None,
        }
    ):
        raise V18Error("checkpoint compare is not one exact path introduction")
    return projection


def _git_data_ref(*, expected_sha: str | None = None) -> dict[str, str]:
    endpoint = f"/repos/{GITHUB_REPOSITORY}/git/ref/heads/{GITHUB_BRANCH}"
    body = _github_git_data_api(endpoint)
    observed = _full_git_sha(
        body.get("object", {}).get("sha") if isinstance(body.get("object"), Mapping) else None,
        "Git Data branch ref SHA",
    )
    return _git_data_ref_projection(body, expected_sha=observed if expected_sha is None else expected_sha)


def _git_data_commit(sha: str) -> dict[str, Any]:
    token = _full_git_sha(sha, "Git Data requested commit")
    projection = _git_data_commit_projection(
        _github_git_data_api(
            f"/repos/{GITHUB_REPOSITORY}/git/commits/{token}"
        )
    )
    if projection["sha"] != token:
        raise V18Error("Git Data commit response SHA differs from request")
    return projection


def _git_data_tree(sha: str) -> dict[str, Any]:
    token = _full_git_sha(sha, "Git Data requested tree")
    projection = _git_data_tree_projection(
        _github_git_data_api(
            f"/repos/{GITHUB_REPOSITORY}/git/trees/{token}?recursive=1"
        )
    )
    if projection["sha"] != token:
        raise V18Error("Git Data recursive tree SHA differs from request")
    return projection


def _git_data_blob(sha: str) -> bytes:
    token = _full_git_sha(sha, "Git Data requested blob")
    return _git_data_blob_bytes(
        _github_git_data_api(f"/repos/{GITHUB_REPOSITORY}/git/blobs/{token}"),
        expected_sha=token,
    )


def _tree_map(tree: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["path"]): dict(item) for item in tree["entries"]}


def _checkpoint_protected_paths() -> tuple[str, ...]:
    protocol = read_json(PROTOCOL)
    paths = protocol["activation"]["preregistration_commit"]["required_paths"]
    if not isinstance(paths, list) or any(not isinstance(item, str) for item in paths):
        raise V18Error("checkpoint protected-path registry is invalid")
    additional_test_paths = _protocol_additional_test_artifact_paths(protocol)
    proposal_root = CHECKPOINT_PROPOSAL_DIR.relative_to(ROOT).as_posix()
    combined = [*paths, *additional_test_paths, proposal_root]
    if len(combined) != len(set(combined)):
        raise V18Error("checkpoint protected-path registry overlaps or duplicates")
    return tuple(combined)


def _checkpoint_protected_blob_authorities() -> dict[str, str]:
    """Derive exact Git blob identities from activation-bound local bytes."""

    payload = read_json(ACTIVATION_PAYLOAD)
    bindings = {
        str(payload["protocol_path"]): str(payload["protocol_sha256"]),
        str(payload["hypothesis_path"]): str(payload["hypothesis_sha256"]),
        str(payload["runtime_lock_path"]): str(payload["runtime_lock_sha256"]),
        str(payload["runner_path"]): str(payload["runner_sha256"]),
        str(payload["audit_path"]): str(payload["audit_sha256"]),
        str(payload["rehearsal_path"]): str(payload["rehearsal_sha256"]),
        str(payload["tests_path"]): str(payload["tests_sha256"]),
        str(payload["iteration_report_path"]): str(
            payload["iteration_report_sha256"]
        ),
        str(payload["validation_report_path"]): str(
            payload["validation_report_sha256"]
        ),
        str(payload["session_calendar_path"]): str(payload["session_calendar_sha256"]),
        str(payload["workflow_path"]): str(payload["workflow_sha256"]),
    }
    for item in _payload_additional_test_artifacts(
        payload, verify_worktree=True
    ):
        if item["path"] in bindings:
            raise V18Error("checkpoint additional test binding overlaps")
        bindings[item["path"]] = item["sha256"]
    result: dict[str, str] = {}
    for relative, expected_sha256 in bindings.items():
        path = ROOT / relative
        payload_bytes = _plain_file_bytes(path, label=f"checkpoint protected {relative}")
        if hashlib.sha256(payload_bytes).hexdigest() != expected_sha256:
            raise V18Error(f"checkpoint protected local bytes changed: {relative}")
        result[relative] = _git_blob_sha1(payload_bytes)
    return result


def _checkpoint_historical_proposal_blob_authorities(
    *,
    session: pd.Timestamp,
    role: str,
    activation: Mapping[str, Any],
    sequence_number: int,
    current_safety_bytes: bytes | None = None,
) -> dict[str, str]:
    """Return every proposal blob that must already exist in the parent tree."""

    calendar = load_registered_calendar()
    first = _date(activation["first_counted_session"], "checkpoint first session")
    denominator = calendar[calendar >= first]
    if sequence_number >= len(denominator) or denominator[sequence_number] != session:
        raise V18Error("checkpoint protected history sequence differs from calendar")
    result: dict[str, str] = {}
    for prior_session in denominator[:sequence_number]:
        for prior_role in CHECKPOINT_ROLES:
            prior_path = _checkpoint_proposal_path(prior_session, prior_role)
            relative = prior_path.relative_to(ROOT).as_posix()
            result[relative] = _git_blob_sha1(
                _plain_file_bytes(prior_path, label="historical checkpoint proposal")
            )
    if role == "primary":
        if current_safety_bytes is None:
            current_safety_bytes = _plain_file_bytes(
                _checkpoint_proposal_path(session, "safety_cash"),
                label="current checkpoint safety proposal",
            )
        relative = _checkpoint_proposal_path(session, "safety_cash").relative_to(
            ROOT
        ).as_posix()
        result[relative] = _git_blob_sha1(current_safety_bytes)
    elif role != "safety_cash":
        raise V18Error("checkpoint protected history role changed")
    return result


def _validate_exact_checkpoint_parent_blobs(
    tree_map: Mapping[str, Mapping[str, Any]],
    *,
    expected_proposals: Mapping[str, str],
) -> None:
    expected = {
        **_checkpoint_protected_blob_authorities(),
        **dict(expected_proposals),
    }
    for path, sha in expected.items():
        entry = tree_map.get(path)
        if entry != {
            "path": path,
            "mode": "100644",
            "type": "blob",
            "sha": sha,
            "size": None if entry is None else entry.get("size"),
        }:
            raise V18Error(f"checkpoint protected blob differs: {path}")
    proposal_root = CHECKPOINT_PROPOSAL_DIR.relative_to(ROOT).as_posix()
    expected_proposal_paths = set(expected_proposals)
    expected_directories = {proposal_root} if expected_proposal_paths else set()
    for proposal_path in expected_proposal_paths:
        parts = Path(proposal_path).parts
        root_parts = Path(proposal_root).parts
        if parts[: len(root_parts)] != root_parts:
            raise V18Error("checkpoint expected proposal path escaped its root")
        expected_directories.update(
            "/".join(parts[:index])
            for index in range(len(root_parts), len(parts))
        )
    observed_under_root = {
        path: dict(entry)
        for path, entry in tree_map.items()
        if path == proposal_root or path.startswith(f"{proposal_root}/")
    }
    expected_under_root = expected_proposal_paths | expected_directories
    if set(observed_under_root) != expected_under_root:
        raise V18Error("checkpoint parent proposal subtree differs")
    for directory in expected_directories:
        entry = observed_under_root[directory]
        if entry.get("mode") != "040000" or entry.get("type") != "tree":
            raise V18Error("checkpoint proposal directory entry differs")


def _validate_tree_transition(
    parent_tree: Mapping[str, Any],
    child_tree: Mapping[str, Any],
    *,
    target_path: str,
    target_blob_sha: str,
    expected_parent_proposals: Mapping[str, str],
) -> None:
    parent = _tree_map(parent_tree)
    child = _tree_map(child_tree)
    _validate_exact_checkpoint_parent_blobs(
        parent,
        expected_proposals=expected_parent_proposals,
    )
    if target_path in parent:
        raise V18Error("checkpoint target path already exists before mutation")
    expected_entry = {
        "path": target_path,
        "mode": "100644",
        "type": "blob",
        "sha": target_blob_sha,
        "size": child.get(target_path, {}).get("size"),
    }
    if child.get(target_path) != expected_entry:
        raise V18Error("checkpoint proposal tree entry differs")
    changed_directories = {
        "/".join(Path(target_path).parts[:index])
        for index in range(1, len(Path(target_path).parts))
    }
    for directory in changed_directories:
        if (
            child.get(directory, {}).get("type") != "tree"
            or (
                directory in parent
                and parent.get(directory, {}).get("type") != "tree"
            )
        ):
            raise V18Error("checkpoint target ancestor directory differs")
    reduced_child = {
        path: value
        for path, value in child.items()
        if path != target_path and path not in changed_directories
    }
    reduced_parent = {
        path: value for path, value in parent.items() if path not in changed_directories
    }
    if reduced_child != reduced_parent:
        raise V18Error("checkpoint commit changes another tree entry")
    _validate_exact_checkpoint_parent_blobs(
        child,
        expected_proposals={
            **dict(expected_parent_proposals),
            target_path: target_blob_sha,
        },
    )


def _checkpoint_local_proposal(
    session: pd.Timestamp,
    role: str,
    *,
    activation: Mapping[str, Any],
    sequence_number: int,
    previous_record_sha256: str,
) -> tuple[dict[str, Any], bytes, str]:
    path = _checkpoint_proposal_path(session, role)
    proposal, _, relative = validate_checkpoint_proposal(
        path,
        session_date=session,
        role=role,
        activation=activation,
        sequence_number=sequence_number,
        previous_record_sha256=previous_record_sha256,
    )
    payload = _plain_file_bytes(path, label=f"checkpoint {role} proposal")
    if payload != _json_file_bytes(proposal):
        raise V18Error("checkpoint proposal file is not canonical JSON bytes")
    return proposal, payload, relative


def _checkpoint_previous_remote_parent(
    *,
    activation: Mapping[str, Any],
    existing_records: Sequence[Mapping[str, Any]],
    observed_tip: str,
) -> str:
    if not existing_records:
        expected = _full_git_sha(
            activation["activation_receipt_commit_sha"],
            "activation receipt C",
        )
        if observed_tip != expected:
            raise V18Error("first checkpoint branch parent is not activation receipt C")
        return expected
    previous_session = _date(
        existing_records[-1]["session_date"], "previous checkpoint session"
    )
    commit = _git_data_commit(observed_tip)
    expected_message = f"model-v18 checkpoint {previous_session.date()} primary"
    if commit["message"] != expected_message or len(commit["parent_shas"]) != 1:
        raise V18Error("remote tip is not the previous counted primary checkpoint")
    previous = _checkpoint_local_proposal(
        previous_session,
        "primary",
        activation=activation,
        sequence_number=len(existing_records) - 1,
        previous_record_sha256=(
            existing_records[-2]["record_sha256"]
            if len(existing_records) > 1
            else ZERO_SHA256
        ),
    )
    tree = _git_data_tree(commit["tree_sha"])
    entry = _tree_map(tree).get(previous[2])
    if (
        entry is None
        or entry.get("mode") != "100644"
        or entry.get("type") != "blob"
        or _git_data_blob(str(entry["sha"])) != previous[1]
    ):
        raise V18Error("previous primary proposal is not the exact remote tip blob")
    return observed_tip


def _publish_checkpoint_role(
    *,
    session: pd.Timestamp,
    role: str,
    exact_parent_sha: str,
    proposal: Mapping[str, Any],
    proposal_bytes: bytes,
    proposal_path: str,
    expected_parent_proposals: Mapping[str, str],
) -> dict[str, Any]:
    """Execute the exact twelve Git Data operations for one role, once."""

    if role not in CHECKPOINT_ROLES:
        raise V18Error("checkpoint publication role is invalid")
    exact_parent_sha = _full_git_sha(exact_parent_sha, "checkpoint exact parent")
    ref_endpoint = f"/repos/{GITHUB_REPOSITORY}/git/ref/heads/{GITHUB_BRANCH}"
    refs_endpoint = f"/repos/{GITHUB_REPOSITORY}/git/refs/heads/{GITHUB_BRANCH}"

    # 0 get_ref, 1 get_parent_commit, 2 get_parent_tree.  In particular, the
    # immutable target-path absence check happens before any POST/PATCH.
    _git_data_ref_projection(
        _github_git_data_api(ref_endpoint), expected_sha=exact_parent_sha
    )
    parent_commit = _git_data_commit(exact_parent_sha)
    parent_tree = _git_data_tree(parent_commit["tree_sha"])
    parent_map = _tree_map(parent_tree)
    if proposal_path in parent_map:
        raise V18Error("checkpoint proposal target already exists before publication")
    _validate_exact_checkpoint_parent_blobs(
        parent_map,
        expected_proposals=expected_parent_proposals,
    )

    # 3 create_blob.
    blob_body = {
        "content": base64.b64encode(proposal_bytes).decode("ascii"),
        "encoding": "base64",
    }
    blob_response = _github_git_data_api(
        f"/repos/{GITHUB_REPOSITORY}/git/blobs",
        method="POST",
        request_body=blob_body,
        expected_status=201,
    )
    blob_sha = _full_git_sha(blob_response.get("sha"), "created proposal blob SHA")
    if (
        blob_sha != _git_blob_sha1(proposal_bytes)
        or blob_response.get("url")
        != f"https://api.github.com/repos/{GITHUB_REPOSITORY}/git/blobs/{blob_sha}"
    ):
        raise V18Error("created proposal blob response differs")

    # 4 create_tree.
    tree_body = {
        "base_tree": parent_commit["tree_sha"],
        "tree": [
            {
                "mode": "100644",
                "path": proposal_path,
                "sha": blob_sha,
                "type": "blob",
            }
        ],
    }
    tree_response = _github_git_data_api(
        f"/repos/{GITHUB_REPOSITORY}/git/trees",
        method="POST",
        request_body=tree_body,
        expected_status=201,
    )
    new_tree_sha = _full_git_sha(tree_response.get("sha"), "created tree SHA")
    if (
        tree_response.get("truncated") is not False
        or tree_response.get("url")
        != f"https://api.github.com/repos/{GITHUB_REPOSITORY}/git/trees/{new_tree_sha}"
    ):
        raise V18Error("created checkpoint tree response differs")

    # 5 create_commit.
    commit_body = {
        "message": f"model-v18 checkpoint {session.date()} {role}",
        "parents": [exact_parent_sha],
        "tree": new_tree_sha,
    }
    create_commit = _git_data_commit_projection(
        _github_git_data_api(
            f"/repos/{GITHUB_REPOSITORY}/git/commits",
            method="POST",
            request_body=commit_body,
            expected_status=201,
        )
    )
    new_commit_sha = create_commit["sha"]
    if (
        create_commit["message"] != commit_body["message"]
        or create_commit["parent_shas"] != [exact_parent_sha]
        or create_commit["tree_sha"] != new_tree_sha
    ):
        raise V18Error("created checkpoint commit projection differs")

    # 6 update_ref is the sole irreversible compare-and-swap boundary.
    updated_ref = _github_git_data_api(
        refs_endpoint,
        method="PATCH",
        request_body={"force": False, "sha": new_commit_sha},
        expected_status=200,
    )
    _git_data_ref_projection(updated_ref, expected_sha=new_commit_sha)

    # 7..11 verify_ref/commit/tree/blob/compare.
    _git_data_ref_projection(
        _github_git_data_api(ref_endpoint), expected_sha=new_commit_sha
    )
    verified_commit = _git_data_commit(new_commit_sha)
    if verified_commit != create_commit:
        raise V18Error("refetched checkpoint commit differs from create response")
    verified_tree = _git_data_tree(new_tree_sha)
    _validate_tree_transition(
        parent_tree,
        verified_tree,
        target_path=proposal_path,
        target_blob_sha=blob_sha,
        expected_parent_proposals=expected_parent_proposals,
    )
    if _git_data_blob(blob_sha) != proposal_bytes:
        raise V18Error("refetched checkpoint proposal blob bytes differ")
    compare = _git_data_compare_projection(
        _github_git_data_api(
            f"/repos/{GITHUB_REPOSITORY}/compare/{exact_parent_sha}...{new_commit_sha}"
        ),
        expected_commit=new_commit_sha,
        expected_path=proposal_path,
    )
    if compare["files"][0]["sha"] != blob_sha:
        raise V18Error("checkpoint compare proposal blob SHA differs")
    return {
        "checkpoint_role": role,
        "checkpoint_proposal_path": proposal_path,
        "checkpoint_proposal_file_sha256": hashlib.sha256(proposal_bytes).hexdigest(),
        "checkpoint_proposal_sha256": proposal["proposal_sha256"],
        "checkpoint_commit_sha": new_commit_sha,
        "parent_shas": [exact_parent_sha],
        "commit_html_url": verified_commit["html_url"],
        "committer_date": verified_commit["committer_date"],
    }


def _wait_for_checkpoint_workflow_exposure(commit_sha: str) -> int:
    deadline = time_module.monotonic() + 60.0
    while True:
        try:
            return _select_github_workflow_run_id(
                commit_sha,
                require_terminal=False,
                require_success=False,
            )
        except V18Error as exc:
            if "not yet available" not in str(exc) or time_module.monotonic() >= deadline:
                raise V18Error("checkpoint safety workflow exposure failed") from exc
            time_module.sleep(1.0)


def publish_checkpoint(*, session_date: Any) -> dict[str, Any]:
    """Publish one prepared pair safety-first using only GitHub Git Data API."""

    if not _STRICT_RUNTIME_ACTIVE:
        raise V18Error("checkpoint publication requires strict runtime validation")
    validate_canonical_activation_artifacts()
    session = _date(session_date, "checkpoint publication session")
    existing = _load_decision_record_authority(heal_derived=True)
    activation = _discover_activation_context(existing)
    expected_sessions = load_registered_calendar()
    expected_sessions = expected_sessions[
        expected_sessions
        >= _date(activation["first_counted_session"], "activation first session")
    ]
    sequence = len(existing)
    if sequence >= len(expected_sessions) or session != expected_sessions[sequence]:
        raise V18Error("checkpoint publication is not the next registered session")
    previous_record = existing[-1]["record_sha256"] if existing else ZERO_SHA256
    pair: dict[str, tuple[dict[str, Any], bytes, str]] = {}
    for role in CHECKPOINT_ROLES:
        pair[role] = _checkpoint_local_proposal(
            session,
            role,
            activation=activation,
            sequence_number=sequence,
            previous_record_sha256=previous_record,
        )
    _validate_checkpoint_proposal_pair(
        {role: value[0] for role, value in pair.items()}
    )
    exact_parent = _full_git_sha(
        (
            existing[-1]["checkpoint_branch_tip_sha_when_observed"]
            if existing
            else activation["activation_receipt_commit_sha"]
        ),
        "checkpoint publication exact previous primary",
    )
    safety_parent_proposals = _checkpoint_historical_proposal_blob_authorities(
        session=session,
        role="safety_cash",
        activation=activation,
        sequence_number=sequence,
    )
    safety = _publish_checkpoint_role(
        session=session,
        role="safety_cash",
        exact_parent_sha=exact_parent,
        proposal=pair["safety_cash"][0],
        proposal_bytes=pair["safety_cash"][1],
        proposal_path=pair["safety_cash"][2],
        expected_parent_proposals=safety_parent_proposals,
    )
    _wait_for_checkpoint_workflow_exposure(safety["checkpoint_commit_sha"])
    primary = _publish_checkpoint_role(
        session=session,
        role="primary",
        exact_parent_sha=safety["checkpoint_commit_sha"],
        proposal=pair["primary"][0],
        proposal_bytes=pair["primary"][1],
        proposal_path=pair["primary"][2],
        expected_parent_proposals={
            **safety_parent_proposals,
            pair["safety_cash"][2]: _git_blob_sha1(pair["safety_cash"][1]),
        },
    )
    _validate_startup_and_module_closure(phase="checkpoint publication exit")
    return {
        "target_session": str(session.date()),
        "checkpoint_batch_id": pair["primary"][0]["checkpoint_batch_id"],
        "safety_commit_sha": safety["checkpoint_commit_sha"],
        "primary_commit_sha": primary["checkpoint_commit_sha"],
        "fixed_ref": GITHUB_REF,
        "production_model_changed": False,
        "orders_allowed": False,
    }


def _remote_checkpoint_role_evidence(
    *,
    session: pd.Timestamp,
    role: str,
    commit_sha: str,
    expected_parent_sha: str,
    activation: Mapping[str, Any],
    sequence_number: int,
    previous_record_sha256: str,
    branch_observation: Mapping[str, Any],
) -> dict[str, Any]:
    proposal, proposal_bytes, relative = _checkpoint_local_proposal(
        session,
        role,
        activation=activation,
        sequence_number=sequence_number,
        previous_record_sha256=previous_record_sha256,
    )
    commit_sha = _full_git_sha(commit_sha, "checkpoint proposal commit")
    expected_parent_sha = _full_git_sha(
        expected_parent_sha, "checkpoint proposal sole parent"
    )
    commit = _git_data_commit(commit_sha)
    if (
        commit["message"] != f"model-v18 checkpoint {session.date()} {role}"
        or commit["parent_shas"] != [expected_parent_sha]
    ):
        raise V18Error("checkpoint remote commit message or sole parent differs")
    if _timestamp(proposal["created_at"], "checkpoint proposal created_at") > _timestamp(
        commit["committer_date"], "checkpoint remote committed_at"
    ):
        raise V18Error("checkpoint proposal creation timestamp follows its commit")
    parent_commit = _git_data_commit(expected_parent_sha)
    parent_tree = _git_data_tree(parent_commit["tree_sha"])
    tree = _git_data_tree(commit["tree_sha"])
    entry = _tree_map(tree).get(relative)
    if entry is None or entry.get("mode") != "100644" or entry.get("type") != "blob":
        raise V18Error("checkpoint proposal remote tree entry differs")
    remote_bytes = _git_data_blob(str(entry["sha"]))
    if remote_bytes != proposal_bytes:
        raise V18Error("checkpoint remote proposal bytes differ from sealed file")
    _validate_tree_transition(
        parent_tree,
        tree,
        target_path=relative,
        target_blob_sha=str(entry["sha"]),
        expected_parent_proposals=(
            _checkpoint_historical_proposal_blob_authorities(
                session=session,
                role=role,
                activation=activation,
                sequence_number=sequence_number,
            )
        ),
    )
    compare = _git_data_compare_projection(
        _github_git_data_api(
            f"/repos/{GITHUB_REPOSITORY}/compare/{expected_parent_sha}...{commit_sha}"
        ),
        expected_commit=commit_sha,
        expected_path=relative,
    )
    if compare["files"][0]["sha"] != entry["sha"]:
        raise V18Error("checkpoint remote compare blob differs")
    workflow_entry = _tree_map(tree).get(WORKFLOW_PATH)
    if (
        workflow_entry is None
        or workflow_entry.get("mode") != "100644"
        or workflow_entry.get("type") != "blob"
        or hashlib.sha256(_git_data_blob(str(workflow_entry["sha"]))).hexdigest()
        != WORKFLOW_SHA256
    ):
        raise V18Error("checkpoint remote workflow bytes differ")
    commit_observation = _github_observation(commit_sha=commit_sha)
    commit_observation, branch_value = _validate_github_observation_pair(
        commit_observation,
        branch_observation,
        commit_sha=commit_sha,
        branch=GITHUB_BRANCH,
    )
    if commit_observation["canonical_projection"]["parent_shas"] != [expected_parent_sha]:
        raise V18Error("checkpoint REST commit parent differs from Git Data commit")
    if (
        commit_observation["canonical_projection"]["html_url"] != commit["html_url"]
        or _timestamp(
            commit_observation["canonical_projection"]["committer_date"],
            "checkpoint REST committed_at",
        )
        != _timestamp(commit["committer_date"], "checkpoint Git Data committed_at")
    ):
        raise V18Error("checkpoint REST/Git Data commit projections differ")
    workflow = _github_workflow_run_observation(commit_sha, checkpoint=True)
    if _timestamp(
        commit["committer_date"], "checkpoint commit committer_date"
    ) > _timestamp(
        workflow["canonical_projection"]["created_at"],
        "checkpoint workflow created_at",
    ):
        raise V18Error("checkpoint workflow predates its proposal commit")
    return {
        "role": role,
        "proposal": proposal,
        "proposal_sha256": proposal["proposal_sha256"],
        "proposal_path": relative,
        "proposal_file_sha256": hashlib.sha256(remote_bytes).hexdigest(),
        "commit_sha": commit_sha,
        "git_data_commit": commit,
        "commit_observation": commit_observation,
        "branch_observation": branch_value,
        "workflow_observation": workflow,
    }


def _remote_checkpoint_pair_evidence(
    *,
    session: pd.Timestamp,
    activation: Mapping[str, Any],
    existing_records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    sequence = len(existing_records)
    previous = existing_records[-1]["record_sha256"] if existing_records else ZERO_SHA256
    ref = _git_data_ref()
    primary_sha = ref["object_sha"]
    primary_commit = _git_data_commit(primary_sha)
    if len(primary_commit["parent_shas"]) != 1:
        raise V18Error("checkpoint primary commit is not first-parent linear")
    safety_sha = primary_commit["parent_shas"][0]
    safety_commit = _git_data_commit(safety_sha)
    if len(safety_commit["parent_shas"]) != 1:
        raise V18Error("checkpoint safety commit is not first-parent linear")
    expected_previous = _full_git_sha(
        (
            existing_records[-1]["checkpoint_branch_tip_sha_when_observed"]
            if existing_records
            else activation["activation_receipt_commit_sha"]
        ),
        "checkpoint expected previous primary",
    )
    if safety_commit["parent_shas"] != [expected_previous]:
        raise V18Error("checkpoint safety parent is not the previous primary/C")
    branch_observation = _github_observation(branch=GITHUB_BRANCH)
    if branch_observation["canonical_projection"]["commit_sha"] != primary_sha:
        raise V18Error("Git Data and REST branch tips differ during checkpoint resolution")
    safety = _remote_checkpoint_role_evidence(
        session=session,
        role="safety_cash",
        commit_sha=safety_sha,
        expected_parent_sha=expected_previous,
        activation=activation,
        sequence_number=sequence,
        previous_record_sha256=previous,
        branch_observation=branch_observation,
    )
    primary = _remote_checkpoint_role_evidence(
        session=session,
        role="primary",
        commit_sha=primary_sha,
        expected_parent_sha=safety_sha,
        activation=activation,
        sequence_number=sequence,
        previous_record_sha256=previous,
        branch_observation=branch_observation,
    )
    _validate_checkpoint_proposal_pair(
        {"safety_cash": safety["proposal"], "primary": primary["proposal"]}
    )
    _git_data_ref(expected_sha=primary_sha)
    return {"safety_cash": safety, "primary": primary}


def _read_checkpoint_core(
    evidence: Mapping[str, Any],
    *,
    checkpoint_core_store_root: str | Path,
    external_identity_registry: dict[tuple[int, int], str] | None = None,
) -> tuple[dict[str, Any], str]:
    proposal = evidence["proposal"]
    item = {
        "object_key": proposal["sealed_core_object_key"],
        "file": Path(str(proposal["sealed_core_object_key"])).name,
        "byte_count": int(proposal["sealed_core_byte_count"]),
        "sha256": proposal["sealed_core_sha256"],
    }
    with _external_snapshot_paths(
        checkpoint_core_store_root,
        [item],
        prefix=CHECKPOINT_CORE_OBJECT_PREFIX,
        label="checkpoint core",
        identity_registry=external_identity_registry,
        required_signature=CHECKPOINT_CORE_MAGIC[:4],
    ) as (paths, _):
        sealed = parse_checkpoint_core_envelope(
            _plain_file_bytes(paths[0], label="checkpoint core snapshot")
        )
    return _validate_checkpoint_core_object(sealed, proposal)


def _checkpoint_pair_resolution(
    pair: Mapping[str, Mapping[str, Any]],
    *,
    session: pd.Timestamp,
) -> tuple[str, str, Mapping[str, Any]]:
    primary = pair["primary"]
    safety = pair["safety_cash"]
    primary_run = primary["workflow_observation"]["canonical_projection"]
    safety_run = safety["workflow_observation"]["canonical_projection"]
    cutoff = _cutoff(session)
    primary_created = _timestamp(primary_run["created_at"], "primary workflow created_at")
    safety_created = _timestamp(safety_run["created_at"], "safety workflow created_at")
    if primary_created >= cutoff:
        raise V18Error("mandatory primary checkpoint was not created before cutoff")
    if safety_created > primary_created:
        raise V18Error("checkpoint safety/primary workflow creation order changed")
    return "primary", "primary_commitment_timely", primary


def resolve_checkpoint(
    *,
    session_date: Any,
    checkpoint_core_store_root: str | Path,
    activation: Mapping[str, Any],
    existing_records: Sequence[Mapping[str, Any]],
    external_identity_registry: dict[tuple[int, int], str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Nondiscretionarily resolve the mandatory primary/safety evidence pair."""

    session = _date(session_date, "checkpoint resolution session")
    pair = _remote_checkpoint_pair_evidence(
        session=session,
        activation=activation,
        existing_records=existing_records,
    )
    primary = pair["primary"]
    safety = pair["safety_cash"]
    resolution, reason, selected = _checkpoint_pair_resolution(pair, session=session)
    core, core_hash = _read_checkpoint_core(
        selected,
        checkpoint_core_store_root=checkpoint_core_store_root,
        external_identity_registry=external_identity_registry,
    )
    validate_checkpoint_decision_core(core, session_date=session, resolution=resolution)
    _git_data_ref(expected_sha=str(selected["commit_sha"]))
    materialized = datetime.now(TOKYO)
    if materialized < max(
        _timestamp(
            pair[role]["workflow_observation"]["retrieved_at"],
            f"checkpoint {role} workflow retrieved_at",
        )
        for role in CHECKPOINT_ROLES
    ):
        raise V18Error("checkpoint decision materialization predates pair evidence")
    commit_observation = selected["commit_observation"]
    branch_observation = selected["branch_observation"]
    workflow = selected["workflow_observation"]
    binding = {
        "decision_materialized_at": materialized,
        "checkpoint_resolution": resolution,
        "checkpoint_resolution_reason": reason,
        "checkpoint_core_sha256": core_hash,
        "checkpoint_proposal_path": selected["proposal_path"],
        "checkpoint_proposal_file_sha256": selected["proposal_file_sha256"],
        "checkpoint_proposal_sha256": selected["proposal_sha256"],
        "checkpoint_commit_sha": selected["commit_sha"],
        "checkpoint_commit_url": commit_observation["canonical_projection"]["html_url"],
        "checkpoint_commit_committed_at": commit_observation["canonical_projection"][
            "committer_date"
        ],
        "checkpoint_commit_observed_at": max(
            _timestamp(commit_observation["retrieved_at"], "checkpoint commit retrieved"),
            _timestamp(branch_observation["retrieved_at"], "checkpoint branch retrieved"),
        ),
        "checkpoint_branch_tip_sha_when_observed": branch_observation[
            "canonical_projection"
        ]["commit_sha"],
        "checkpoint_commit_observation": commit_observation,
        "checkpoint_branch_observation": branch_observation,
        "checkpoint_workflow_run_id": workflow["canonical_projection"]["run_id"],
        "checkpoint_workflow_run_updated_at": workflow["canonical_projection"][
            "updated_at"
        ],
        "checkpoint_workflow_run_observed_at": workflow["retrieved_at"],
        "checkpoint_workflow_run_observation": workflow,
    }
    return core, binding


def materialize_checkpoint_decision(
    core: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    session_date: Any,
    activation: Mapping[str, Any],
    existing_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    existing = validate_decision_records(existing_records)
    session = _date(session_date, "decision session")
    activation_value = validate_activation_context(activation)
    if set(binding) != set(CHECKPOINT_BINDING_FIELDS):
        raise V18Error("checkpoint binding fields differ from protocol")
    resolution = str(binding["checkpoint_resolution"])
    validate_checkpoint_decision_core(
        core,
        session_date=session,
        resolution=resolution,
    )
    expected_core_hash = canonical_json_sha256(core)
    if expected_core_hash != binding["checkpoint_core_sha256"]:
        raise V18Error("resolved checkpoint core hash differs from evidence")
    payload = {
        "schema_version": 1,
        "session_date": str(session.date()),
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "activation_payload_sha256": activation_value["activation_payload_sha256"],
        "activation_receipt_sha256": activation_value["activation_receipt_sha256"],
        **{
            field: activation_value[field]
            for field in RECEIPT_OBSERVATION_RUNTIME_FIELDS
        },
        **dict(binding),
        **dict(core),
    }
    record = _sealed_record(
        payload,
        sequence_number=len(existing),
        previous_record_sha256=(
            existing[-1]["record_sha256"] if existing else ZERO_SHA256
        ),
    )
    return validate_decision_records([*existing, record])


def _github_commits_for_path(path: str, *, pinned_tip: str) -> list[str]:
    if not path or path.startswith("/") or ".." in Path(path).parts:
        raise V18Error("GitHub path-history subject is unsafe")
    base = (
        f"/repos/{GITHUB_REPOSITORY}/commits?sha={quote(GITHUB_BRANCH, safe='')}"
        f"&path={quote(path, safe='')}&per_page=100"
    )
    result: list[str] = []
    page = 1
    pinned_tip = _full_git_sha(pinned_tip, "path-history pinned tip")
    saw_short_page = False
    while True:
        # The path endpoint is named by the registered mutable branch, so
        # bracket every page with the exact Git Data ref authority.  This
        # prevents pages from different branch snapshots being combined.
        _git_data_ref(expected_sha=pinned_tip)
        values = _github_json_list(f"{base}&page={page}")
        _git_data_ref(expected_sha=pinned_tip)
        if len(values) > 100:
            raise V18Error("GitHub path-history page exceeds registered size")
        if not values:
            break
        if saw_short_page:
            raise V18Error("GitHub path-history continued after a short page")
        for value in values:
            result.append(_full_git_sha(value.get("sha"), "path-history commit SHA"))
        if len(values) < 100:
            saw_short_page = True
        if page >= 10_000:
            raise V18Error("GitHub path-history pagination did not terminate")
        page += 1
    if len(result) != len(set(result)):
        raise V18Error("GitHub path-history contains a duplicate commit")
    return result


def _github_fully_paginated_descendant(
    ancestor: str,
    descendant: str,
    *,
    pinned_tip: str,
) -> list[str]:
    """Prove remote ancestry with every compare page under one pinned ref."""

    ancestor = _full_git_sha(ancestor, "compare ancestor")
    descendant = _full_git_sha(descendant, "compare descendant")
    pinned_tip = _full_git_sha(pinned_tip, "compare pinned tip")
    if ancestor == descendant:
        _git_data_ref(expected_sha=pinned_tip)
        return []
    commits: list[str] = []
    expected_total: int | None = None
    page = 1
    saw_short_page = False
    while True:
        endpoint = (
            f"/repos/{GITHUB_REPOSITORY}/compare/{ancestor}...{descendant}"
            f"?per_page=100&page={page}"
        )
        _git_data_ref(expected_sha=pinned_tip)
        body = _github_git_data_api(endpoint)
        if (
            body.get("status") != "ahead"
            or body.get("behind_by") != 0
            or not isinstance(body.get("ahead_by"), int)
            or isinstance(body.get("ahead_by"), bool)
            or not isinstance(body.get("total_commits"), int)
            or isinstance(body.get("total_commits"), bool)
            or body.get("base_commit", {}).get("sha") != ancestor
            or body.get("merge_base_commit", {}).get("sha") != ancestor
            or not isinstance(body.get("commits"), list)
        ):
            raise V18Error("terminal compare page projection is invalid")
        total = int(body["total_commits"])
        if int(body["ahead_by"]) != total or total <= 0:
            raise V18Error("terminal compare ancestry counts differ")
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            raise V18Error("terminal compare total changed during pagination")
        page_shas = [
            _full_git_sha(item.get("sha"), "terminal compare commit")
            for item in body["commits"]
            if isinstance(item, Mapping)
        ]
        if len(page_shas) != len(body["commits"]):
            raise V18Error("terminal compare commit page is invalid")
        if len(page_shas) > 100:
            raise V18Error("terminal compare page exceeds registered size")
        _git_data_ref(expected_sha=pinned_tip)
        if not page_shas:
            break
        if saw_short_page:
            raise V18Error("terminal compare continued after a short page")
        commits.extend(page_shas)
        if len(page_shas) < 100:
            saw_short_page = True
        if page >= 10_000:
            raise V18Error("terminal compare pagination did not terminate")
        page += 1
    if (
        expected_total is None
        or len(commits) != expected_total
        or len(commits) != len(set(commits))
        or commits[-1] != descendant
    ):
        raise V18Error("terminal compare pages do not end at the pinned tip")
    return commits


def _validate_checkpoint_local_session_directories(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if (
        not CHECKPOINT_PROPOSAL_DIR.is_dir()
        or CHECKPOINT_PROPOSAL_DIR.is_symlink()
        or CHECKPOINT_PROPOSAL_DIR.resolve(strict=True) != CHECKPOINT_PROPOSAL_DIR
    ):
        raise V18Error("checkpoint proposal root is not a resolved plain directory")
    expected_sessions = {str(row["session_date"]) for row in rows}
    actual = {entry.name for entry in CHECKPOINT_PROPOSAL_DIR.iterdir()}
    if actual != expected_sessions:
        raise V18Error("checkpoint proposal session-directory set differs")
    for session_name in sorted(expected_sessions):
        directory = CHECKPOINT_PROPOSAL_DIR / session_name
        info = os.stat(directory, follow_symlinks=False)
        if (
            not stat.S_ISDIR(info.st_mode)
            or directory.is_symlink()
            or stat.S_IMODE(info.st_mode) != 0o755
        ):
            raise V18Error("checkpoint proposal session directory metadata differs")
        children = sorted(directory.iterdir(), key=lambda item: item.name)
        if [item.name for item in children] != ["primary.json", "safety_cash.json"]:
            raise V18Error("checkpoint proposal session directory is not an exact pair")
        for child in children:
            child_info = os.stat(child, follow_symlinks=False)
            if (
                not stat.S_ISREG(child_info.st_mode)
                or child_info.st_nlink != 1
                or child.is_symlink()
                or stat.S_IMODE(child_info.st_mode) != 0o644
            ):
                raise V18Error("checkpoint proposal file metadata differs")


def _validate_checkpoint_external_session_directories(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_core_store_root: str | Path,
) -> None:
    expected_sessions = {str(row["session_date"]) for row in rows}
    anchor = f"{CHECKPOINT_CORE_OBJECT_PREFIX}.terminal-enumeration-anchor"
    with _external_parent_fd(
        checkpoint_core_store_root,
        anchor,
        prefix=CHECKPOINT_CORE_OBJECT_PREFIX,
        label="checkpoint core",
    ) as (base_fd, _):
        if set(os.listdir(base_fd)) != expected_sessions:
            raise V18Error("checkpoint external session-directory set differs")
        for session_name in sorted(expected_sessions):
            info = os.stat(session_name, dir_fd=base_fd, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
                raise V18Error("checkpoint external session directory metadata differs")
            session_fd = os.open(
                session_name,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=base_fd,
            )
            try:
                if set(os.listdir(session_fd)) != {"primary.bin", "safety_cash.bin"}:
                    raise V18Error("checkpoint external session directory is not an exact pair")
                for child in ("primary.bin", "safety_cash.bin"):
                    child_info = os.stat(child, dir_fd=session_fd, follow_symlinks=False)
                    if (
                        not stat.S_ISREG(child_info.st_mode)
                        or child_info.st_nlink != 1
                        or stat.S_IMODE(child_info.st_mode) != 0o600
                        or child_info.st_size != CHECKPOINT_CORE_ENVELOPE_BYTES
                    ):
                        raise V18Error("checkpoint external core metadata differs")
            finally:
                os.close(session_fd)


def _checkpoint_stable_evidence_role(evidence: Mapping[str, Any]) -> dict[str, Any]:
    commit = evidence["git_data_commit"]
    workflow = evidence["workflow_observation"]["canonical_projection"]
    return {
        "checkpoint_role": evidence["role"],
        "checkpoint_proposal_path": evidence["proposal_path"],
        "checkpoint_proposal_file_sha256": evidence["proposal_file_sha256"],
        "checkpoint_proposal_sha256": evidence["proposal_sha256"],
        "checkpoint_commit_sha": evidence["commit_sha"],
        "parent_shas": commit["parent_shas"],
        "commit_html_url": commit["html_url"],
        "committer_date": commit["committer_date"],
        "workflow_run_id": workflow["run_id"],
        "workflow_id": workflow["workflow_id"],
        "workflow_name": workflow["workflow_name"],
        "workflow_path": workflow["workflow_path"],
        "event": workflow["event"],
        "head_sha": workflow["head_sha"],
        "run_attempt": workflow["run_attempt"],
        "status": workflow["status"],
        "conclusion": workflow["conclusion"],
        "created_at": workflow["created_at"],
        "run_started_at": workflow["run_started_at"],
        "updated_at": workflow["updated_at"],
        "workflow_html_url": workflow["html_url"],
    }


def validate_checkpoint_evidence(
    decisions: Sequence[Mapping[str, Any]],
    *,
    checkpoint_core_store_root: str | Path,
    external_identity_registry: dict[tuple[int, int], str] | None = None,
) -> dict[str, str]:
    """Terminally replay the complete remote safety/primary first-parent chain."""

    rows = validate_decision_records(decisions)
    if not rows:
        raise V18Error("terminal checkpoint validation requires decisions")
    _validate_checkpoint_local_session_directories(rows)
    _validate_checkpoint_external_session_directories(
        rows,
        checkpoint_core_store_root=checkpoint_core_store_root,
    )
    activation = _activation_context_from_decision(rows[0])
    activation_commit = _full_git_sha(
        activation["activation_receipt_commit_sha"], "activation receipt C"
    )
    proposal_root = CHECKPOINT_PROPOSAL_DIR.relative_to(ROOT).as_posix()
    pinned_ref = _git_data_ref()
    current_tip = pinned_ref["object_sha"]

    # Full pagination is the authority for unique path introductions and proves
    # that no later commit changed, deleted, reverted, or reintroduced one.
    history_shas = _github_commits_for_path(
        proposal_root,
        pinned_tip=current_tip,
    )
    if len(history_shas) != 2 * len(rows):
        raise V18Error("remote checkpoint proposal history cardinality differs")
    remote_commits = {sha: _git_data_commit(sha) for sha in history_shas}
    if len(remote_commits) != len(history_shas):
        raise V18Error("remote checkpoint proposal history is ambiguous")

    expected_chain: list[tuple[pd.Timestamp, str, str, str]] = []
    remaining = set(history_shas)
    parent = activation_commit
    for row in rows:
        session = _date(row["session_date"], "terminal checkpoint session")
        for role in CHECKPOINT_ROLES:
            message = f"model-v18 checkpoint {session.date()} {role}"
            matches = [
                sha
                for sha in remaining
                if remote_commits[sha]["message"] == message
                and remote_commits[sha]["parent_shas"] == [parent]
            ]
            if len(matches) != 1:
                raise V18Error("remote checkpoint history is not exact safety-primary order")
            commit_sha = matches[0]
            expected_chain.append((session, role, parent, commit_sha))
            remaining.remove(commit_sha)
            parent = commit_sha
    if remaining:
        raise V18Error("remote checkpoint history contains an extra or side commit")
    terminal_checkpoint_tip = parent

    _github_fully_paginated_descendant(
        terminal_checkpoint_tip,
        current_tip,
        pinned_tip=current_tip,
    )
    current_branch_observation = _github_observation(branch=GITHUB_BRANCH)
    if current_branch_observation["canonical_projection"]["commit_sha"] != current_tip:
        raise V18Error("terminal Git Data and REST branch tips differ")
    current_tree = _git_data_tree(_git_data_commit(current_tip)["tree_sha"])
    current_tree_map = _tree_map(current_tree)
    expected_proposal_blobs: dict[str, str] = {}
    for row in rows:
        for role in CHECKPOINT_ROLES:
            relative = (
                _checkpoint_proposal_path(
                    _date(row["session_date"], "checkpoint session"), role
                )
                .relative_to(ROOT)
                .as_posix()
            )
            local = _plain_file_bytes(ROOT / relative, label="terminal proposal file")
            expected_proposal_blobs[relative] = _git_blob_sha1(local)
    _validate_exact_checkpoint_parent_blobs(
        current_tree_map,
        expected_proposals=expected_proposal_blobs,
    )
    for relative in sorted(expected_proposal_blobs):
        local = _plain_file_bytes(ROOT / relative, label="terminal proposal file")
        if _git_data_blob(str(current_tree_map[relative]["sha"])) != local:
            raise V18Error("current remote proposal blob differs from sealed file")

    # A later change-and-revert of any preregistered blob is still invalid.
    for protected_path in _checkpoint_protected_blob_authorities():
        path_history = _github_commits_for_path(
            protected_path,
            pinned_tip=current_tip,
        )
        if not path_history:
            raise V18Error(f"protected path has no remote history: {protected_path}")
        latest_change = path_history[0]
        if latest_change != activation_commit:
            _github_fully_paginated_descendant(
                latest_change,
                activation_commit,
                pinned_tip=current_tip,
            )

    proposal_set: list[dict[str, Any]] = []
    core_set: list[dict[str, Any]] = []
    evidence_set: list[dict[str, Any]] = []
    by_session_role: dict[tuple[str, str], dict[str, Any]] = {}
    for session, role, expected_parent, commit_sha in expected_chain:
        sequence = next(
            index
            for index, row in enumerate(rows)
            if row["session_date"] == str(session.date())
        )
        previous_record = rows[sequence - 1]["record_sha256"] if sequence else ZERO_SHA256
        evidence = _remote_checkpoint_role_evidence(
            session=session,
            role=role,
            commit_sha=commit_sha,
            expected_parent_sha=expected_parent,
            activation=activation,
            sequence_number=sequence,
            previous_record_sha256=previous_record,
            branch_observation=current_branch_observation,
        )
        by_session_role[(str(session.date()), role)] = evidence

    for sequence, row in enumerate(rows):
        target = str(row["session_date"])
        session = _date(target, "terminal checkpoint session")
        pair = {
            role: by_session_role[(target, role)] for role in CHECKPOINT_ROLES
        }
        _validate_checkpoint_proposal_pair(
            {role: evidence["proposal"] for role, evidence in pair.items()}
        )
        resolution, reason, selected = _checkpoint_pair_resolution(
            pair, session=session
        )
        if (
            row["checkpoint_resolution"] != resolution
            or row["checkpoint_resolution_reason"] != reason
        ):
            raise V18Error("terminal checkpoint selection no longer recomputes")
        decision_materialized = _timestamp(
            row["decision_materialized_at"], "terminal decision materialized_at"
        )
        for role in CHECKPOINT_ROLES:
            workflow_updated = _timestamp(
                pair[role]["workflow_observation"]["canonical_projection"]["updated_at"],
                f"terminal {role} workflow updated_at",
            )
            if workflow_updated > decision_materialized:
                raise V18Error(
                    "checkpoint role was not terminally materialized before decision"
                )

        selected_workflow = selected["workflow_observation"]["canonical_projection"]
        if (
            row["checkpoint_proposal_path"] != selected["proposal_path"]
            or row["checkpoint_proposal_file_sha256"]
            != selected["proposal_file_sha256"]
            or row["checkpoint_proposal_sha256"] != selected["proposal_sha256"]
            or row["checkpoint_commit_sha"] != selected["commit_sha"]
            or row["checkpoint_commit_url"]
            != selected["git_data_commit"]["html_url"]
            or _timestamp(
                row["checkpoint_commit_committed_at"], "stored checkpoint committed_at"
            )
            != _timestamp(
                selected["git_data_commit"]["committer_date"],
                "terminal checkpoint committed_at",
            )
            or row["checkpoint_workflow_run_id"] != selected_workflow["run_id"]
            or _timestamp(
                row["checkpoint_workflow_run_updated_at"],
                "stored checkpoint workflow updated_at",
            )
            != _timestamp(
                selected_workflow["updated_at"],
                "terminal checkpoint workflow updated_at",
            )
        ):
            raise V18Error("terminal checkpoint selected public evidence changed")
        stored_commit_receipt = validate_github_observation(
            row["checkpoint_commit_observation"]
        )
        if (
            stored_commit_receipt["canonical_projection"]
            != selected["commit_observation"]["canonical_projection"]
        ):
            raise V18Error("stored checkpoint commit receipt projection changed")
        branch_receipt = validate_github_observation(
            row["checkpoint_branch_observation"]
        )
        if branch_receipt["canonical_projection"]["commit_sha"] != pair["primary"]["commit_sha"]:
            raise V18Error("stored checkpoint branch receipt is not session primary")
        _refetch_github_workflow_observation(
            row["checkpoint_workflow_run_observation"], checkpoint=True
        )

        role_cores: dict[str, tuple[dict[str, Any], str]] = {}
        for role in CHECKPOINT_ROLES:
            role_cores[role] = _read_checkpoint_core(
                pair[role],
                checkpoint_core_store_root=checkpoint_core_store_root,
                external_identity_registry=external_identity_registry,
            )
            validate_checkpoint_decision_core(
                role_cores[role][0],
                session_date=session,
                resolution="primary",
            )
        if (
            canonical_json_bytes(role_cores["safety_cash"][0])
            != canonical_json_bytes(role_cores["primary"][0])
            or role_cores["safety_cash"][1] != role_cores["primary"][1]
        ):
            raise V18Error("checkpoint safety/primary decision cores differ")
        core, core_hash = role_cores[resolution]
        if (
            core_hash != row["checkpoint_core_sha256"]
            or canonical_json_bytes(core)
            != canonical_json_bytes(_decision_core_from_row(row))
        ):
            raise V18Error("terminal selected checkpoint core differs from decision")

        proposal_set.append(
            {
                "target_session": target,
                "checkpoint_batch_id": pair["primary"]["proposal"]["checkpoint_batch_id"],
                "checkpoint_resolution": resolution,
                "roles": [
                    {
                        "checkpoint_role": role,
                        "publication_ordinal": pair[role]["proposal"]["publication_ordinal"],
                        "checkpoint_proposal_path": pair[role]["proposal_path"],
                        "checkpoint_proposal_file_sha256": pair[role]["proposal_file_sha256"],
                        "checkpoint_proposal_sha256": pair[role]["proposal_sha256"],
                    }
                    for role in CHECKPOINT_ROLES
                ],
            }
        )
        core_set.append(
            {
                "target_session": target,
                "roles": [
                    {
                        "checkpoint_role": role,
                        "sealed_core_object_key": pair[role]["proposal"]["sealed_core_object_key"],
                        "sealed_core_byte_count": pair[role]["proposal"]["sealed_core_byte_count"],
                        "sealed_core_sha256": pair[role]["proposal"]["sealed_core_sha256"],
                        "decision_core_sha256": role_cores[role][1],
                    }
                    for role in CHECKPOINT_ROLES
                ],
            }
        )
        evidence_set.append(
            {
                "target_session": target,
                "checkpoint_batch_id": pair["primary"]["proposal"]["checkpoint_batch_id"],
                "checkpoint_resolution": resolution,
                "checkpoint_resolution_reason": reason,
                "roles": [
                    _checkpoint_stable_evidence_role(pair[role])
                    for role in CHECKPOINT_ROLES
                ],
            }
        )
    _git_data_ref(expected_sha=current_tip)
    _validate_startup_and_module_closure(phase="terminal checkpoint replay exit")
    return {
        "checkpoint_proposal_set_sha256": canonical_json_sha256(proposal_set),
        "checkpoint_core_object_set_sha256": canonical_json_sha256(core_set),
        "checkpoint_evidence_set_sha256": canonical_json_sha256(evidence_set),
    }


def _parsed_outcome_frame(parsed_panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(parsed_panel, pd.DataFrame):
        raise V18Error("parsed outcome panel must be a DataFrame")
    frame = parsed_panel.copy()
    if "session_date" in frame and "date" not in frame:
        frame = frame.rename(columns={"session_date": "date"})
    required = {"date", "code", "open", "close"}
    missing = sorted(required - set(frame))
    if missing:
        raise V18Error(f"parsed outcome panel is missing fields: {missing}")
    frame = frame.loc[:, ["date", "code", "open", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    if frame["date"].isna().any():
        raise V18Error("parsed outcome panel contains an invalid date")
    if getattr(frame["date"].dt, "tz", None) is not None:
        frame["date"] = frame["date"].dt.tz_convert(TOKYO).dt.tz_localize(None)
    frame["date"] = frame["date"].dt.normalize()
    frame["code"] = frame["code"].astype("string")
    if frame["code"].isna().any() or frame["code"].str.strip().eq("").any():
        raise V18Error("parsed outcome panel contains an invalid code")
    frame["code"] = frame["code"].astype(str)
    for field in ("open", "close"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
        invalid = frame[field].notna() & ~np.isfinite(frame[field].fillna(0).to_numpy())
        if invalid.any():
            raise V18Error(f"parsed outcome {field} is non-finite")
    return frame.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def _manifest_rank_values(
    target_rows: pd.DataFrame,
    code: Any,
) -> tuple[str | None, float | None, float | None, float | None]:
    if code is None:
        return None, None, None, None
    token = str(code)
    matches = target_rows.loc[target_rows["code"].eq(token)]
    if len(matches) > 1:
        raise V18Error("outcome rank code has duplicate target-session rows")
    if matches.empty:
        return token, None, None, None
    opened = _finite_or_none(matches.iloc[0]["open"], "outcome open")
    closed = _finite_or_none(matches.iloc[0]["close"], "outcome close")
    if opened is None or closed is None or opened <= 0.0 or closed <= 0.0:
        return token, None, None, None
    returned = float((closed / opened - 1.0) * 100.0)
    return token, opened, closed, returned


def _outcome_object_key(target: pd.Timestamp) -> str:
    return f"{OUTCOME_OBJECT_PREFIX}{target.date()}.pdf"


def _parse_outcome_object(raw_path: Path, target: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, int]]:
    try:
        parsed, report = _collect_jpx_registered(raw_path)
    except Exception as exc:
        raise V18Error(f"official outcome PDF parse failed: {exc}") from exc
    frame = _parsed_outcome_frame(parsed)
    inputs = report.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise V18Error("outcome parser report does not bind one source")
    parser_report = inputs[0]
    duplicate_count = int(frame[["date", "code"]].duplicated(keep=False).sum())
    rejected = int(parser_report.get("rejected_rows", -1))
    if rejected != 0 or duplicate_count != 0:
        raise V18Error("outcome parser rejection/duplicate aborts v1.8")
    dates = pd.DatetimeIndex(frame["date"].unique())
    if len(dates) != 1 or dates[0] != target:
        raise V18Error("official outcome PDF date differs from target session")
    return frame, {
        "parsed_row_count": int(len(frame)),
        "rejected_row_count": rejected,
        "duplicate_date_code_count": duplicate_count,
        "parsed_unique_date_count": int(len(dates)),
        "target_session_row_count": int(frame["date"].eq(target).sum()),
    }


def build_outcome_manifest(
    decision: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    source_pdf_path: str | Path,
    *,
    outcome_raw_store_root: str | Path,
    source_file_name: str,
    source_url: str,
    source_received_at: Any,
    runtime_lock_verified_at: Any | None = None,
    created_at: Any | None = None,
    sealed_at: Any | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Exclusively seal, parse, and bind one official target-day JPX PDF."""

    verified = _runtime_verified_timestamp(runtime_lock_verified_at)
    current = (
        validate_decision_records([decision])[0]
        if isinstance(decision, Mapping)
        else validate_decision_records(decision)[-1]
    )
    target = _date(current["session_date"], "outcome target_session")
    object_key = _outcome_object_key(target)
    if Path(source_file_name).name != source_file_name or not source_file_name.lower().endswith(
        ".pdf"
    ):
        raise V18Error("official outcome source_file_name is invalid")
    if target.strftime("%Y%m%d") not in source_file_name:
        raise V18Error("official outcome filename does not bind target date")
    official_source_url = _official_jpx_daily_url(
        source_url,
        "outcome source URL",
        file_name=source_file_name,
        source_session=target,
    )
    received = _timestamp(source_received_at, "outcome source_received_at")
    if received <= _timestamp(
        current["decision_materialized_at"], "decision materialized_at"
    ):
        raise V18Error("outcome bytes were received before decision sealing")
    if _STRICT_RUNTIME_ACTIVE and received > datetime.now(TOKYO):
        raise V18Error("outcome source receipt is in the future")
    source_metadata = _seal_external_object(
        source_pdf_path,
        store_root=outcome_raw_store_root,
        object_key=object_key,
        prefix=OUTCOME_OBJECT_PREFIX,
        label="outcome",
    )
    outcome_object = {
        "object_key": object_key,
        "file": Path(object_key).name,
        "byte_count": source_metadata[0],
        "sha256": source_metadata[1],
    }
    with _external_snapshot_paths(
        outcome_raw_store_root,
        [outcome_object],
        prefix=OUTCOME_OBJECT_PREFIX,
        label="outcome",
    ) as (snapshot_paths, _):
        frame, counts = _parse_outcome_object(snapshot_paths[0], target)
    target_rows = frame.loc[frame["date"].eq(target)]
    rank1 = _manifest_rank_values(target_rows, current["c00_rank1_code"])
    rank2 = _manifest_rank_values(target_rows, current["c02_rank2_code"])
    created = _operation_timestamp(created_at, "outcome manifest created_at")
    sealed = _operation_timestamp(sealed_at, "outcome manifest sealed_at")
    value: dict[str, Any] = {
        "schema_version": 1,
        "target_session": str(target.date()),
        "created_at": created,
        "sealed_at": sealed,
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "runtime_lock_verified_at": verified,
        "source_file_name": source_file_name,
        "source_url": official_source_url,
        "raw_source_object_key": object_key,
        "source_byte_count": source_metadata[0],
        "source_sha256": source_metadata[1],
        "source_received_at": received,
        "parser_path": JPX_PARSER_PATH,
        "parser_version": JPX_PARSER_VERSION,
        "parser_sha256": JPX_PARSER_SHA256,
        **counts,
        "rank1_code": rank1[0],
        "rank1_open": rank1[1],
        "rank1_close": rank1[2],
        "rank1_recomputed_oc_return_pct": rank1[3],
        "rank2_code": rank2[0],
        "rank2_open": rank2[1],
        "rank2_close": rank2[2],
        "rank2_recomputed_oc_return_pct": rank2[3],
        "decision_record_sha256": current["record_sha256"],
        "protocol_sha256": PROTOCOL_SHA256,
        "activation_payload_sha256": current["activation_payload_sha256"],
        "activation_receipt_sha256": current["activation_receipt_sha256"],
        "python_version": platform.python_version(),
        "canonical_json_contract": CANONICAL_JSON_CONTRACT,
    }
    value["outcome_manifest_sha256"] = canonical_json_sha256(
        value, exclude_fields={"outcome_manifest_sha256"}
    )
    validate_outcome_manifest(
        value, current, outcome_raw_store_root=outcome_raw_store_root
    )
    if output is not None:
        canonical_manifest = OUTCOME_MANIFEST_DIR / f"{target.date()}.json"
        if Path(output).resolve() != canonical_manifest.resolve():
            raise V18Error("outcome manifest output is not its canonical path")
        write_json(value, output, exclusive=True)
    return value


def validate_outcome_manifest(
    manifest: Mapping[str, Any] | str | Path,
    decision: Mapping[str, Any],
    *,
    outcome_raw_store_root: str | Path,
    external_identity_registry: dict[tuple[int, int], str] | None = None,
) -> tuple[dict[str, Any], str]:
    value = read_json(manifest) if isinstance(manifest, (str, Path)) else dict(manifest)
    if set(value) != set(OUTCOME_MANIFEST_FIELDS):
        raise V18Error("outcome manifest fields differ from protocol")
    current = dict(decision)
    target = _date(current["session_date"], "decision session")
    if value["schema_version"] != 1 or value["target_session"] != str(target.date()):
        raise V18Error("outcome manifest target identity changed")
    created = _timestamp(value["created_at"], "outcome manifest created_at")
    sealed = _timestamp(value["sealed_at"], "outcome manifest sealed_at")
    verified = _timestamp(
        value["runtime_lock_verified_at"], "outcome runtime_lock_verified_at"
    )
    if value["runtime_lock_sha256"] != RUNTIME_LOCK_SHA256:
        raise V18Error("outcome manifest runtime-lock binding changed")
    if verified > created or created > sealed:
        raise V18Error("outcome manifest timestamp DAG is invalid")
    expected_key = _outcome_object_key(target)
    if value["raw_source_object_key"] != expected_key:
        raise V18Error("outcome raw object key changed")
    _official_jpx_daily_url(
        value["source_url"],
        "outcome manifest source URL",
        file_name=value["source_file_name"],
        source_session=target,
    )
    _require_sha(value["source_sha256"], "outcome source_sha256")
    received = _timestamp(value["source_received_at"], "outcome source_received_at")
    if received <= _timestamp(
        current["decision_materialized_at"], "decision materialized_at"
    ):
        raise V18Error("outcome source receipt predates decision")
    if received > created:
        raise V18Error("outcome manifest creation predates source receipt")
    if (
        value["parser_path"] != JPX_PARSER_PATH
        or value["parser_version"] != JPX_PARSER_VERSION
        or value["parser_sha256"] != JPX_PARSER_SHA256
    ):
        raise V18Error("outcome parser identity changed")
    outcome_object = {
        "object_key": expected_key,
        "file": Path(expected_key).name,
        "byte_count": int(value["source_byte_count"]),
        "sha256": value["source_sha256"],
    }
    with _external_snapshot_paths(
        outcome_raw_store_root,
        [outcome_object],
        prefix=OUTCOME_OBJECT_PREFIX,
        label="outcome",
        identity_registry=external_identity_registry,
    ) as (snapshot_paths, _):
        frame, counts = _parse_outcome_object(snapshot_paths[0], target)
    for field, expected_count in counts.items():
        if int(value[field]) != expected_count:
            raise V18Error(f"outcome manifest {field} does not reparse")
    if value["decision_record_sha256"] != current["record_sha256"]:
        raise V18Error("outcome manifest does not bind decision")
    if (
        value["protocol_sha256"] != current["protocol_sha256"]
        or value["activation_payload_sha256"] != current["activation_payload_sha256"]
        or value["activation_receipt_sha256"] != current["activation_receipt_sha256"]
    ):
        raise V18Error("outcome manifest authority changed")
    target_rows = frame.loc[frame["date"].eq(target)]
    recomputed = (
        _manifest_rank_values(target_rows, current["c00_rank1_code"]),
        _manifest_rank_values(target_rows, current["c02_rank2_code"]),
    )
    for rank, expected in zip((1, 2), recomputed, strict=True):
        observed = (
            value[f"rank{rank}_code"],
            _finite_or_none(value[f"rank{rank}_open"], f"rank{rank} open"),
            _finite_or_none(value[f"rank{rank}_close"], f"rank{rank} close"),
            _finite_or_none(
                value[f"rank{rank}_recomputed_oc_return_pct"], f"rank{rank} return"
            ),
        )
        if observed[0] != expected[0]:
            raise V18Error(f"outcome manifest rank{rank} code changed")
        for observed_value, expected_value in zip(observed[1:], expected[1:], strict=True):
            if observed_value is None or expected_value is None:
                if observed_value is not expected_value:
                    raise V18Error(f"outcome manifest rank{rank} nullability changed")
            elif not _ieee_float_equal(observed_value, expected_value):
                raise V18Error(f"outcome manifest rank{rank} value does not reparse")
    if value["python_version"] != read_json(RUNTIME_LOCK)["runtime"]["python"]["version"]:
        raise V18Error("outcome manifest Python version differs from runtime lock")
    if value["canonical_json_contract"] != CANONICAL_JSON_CONTRACT:
        raise V18Error("outcome manifest canonical contract changed")
    expected_hash = canonical_json_sha256(
        value, exclude_fields={"outcome_manifest_sha256"}
    )
    if value["outcome_manifest_sha256"] != expected_hash:
        raise V18Error("outcome manifest self-hash mismatch")
    return value, expected_hash


def validate_outcome_records(
    records: Sequence[Mapping[str, Any]] | pd.DataFrame,
    decisions: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> list[dict[str, Any]]:
    decision_rows = validate_decision_records(decisions)
    rows = validate_hash_chain(records, required_fields=OUTCOME_FIELDS)
    if len(rows) > len(decision_rows):
        raise V18Error("outcome ledger is longer than decision ledger")
    for outcome, decision in zip(rows, decision_rows, strict=False):
        if outcome["schema_version"] != 1:
            raise V18Error("outcome schema changed")
        if outcome["session_date"] != decision["session_date"]:
            raise V18Error("outcome session differs from its decision")
        if outcome["decision_record_sha256"] != decision["record_sha256"]:
            raise V18Error("outcome does not bind decision record")
        if (
            outcome["protocol_sha256"] != decision["protocol_sha256"]
            or outcome["activation_payload_sha256"]
            != decision["activation_payload_sha256"]
            or outcome["activation_receipt_sha256"]
            != decision["activation_receipt_sha256"]
        ):
            raise V18Error("outcome authority binding changed")
        for field in (
            "decision_record_sha256",
            "outcome_source_sha256",
            "outcome_manifest_sha256",
            "protocol_sha256",
            "activation_payload_sha256",
            "activation_receipt_sha256",
        ):
            _require_sha(outcome[field], f"outcome {field}")
        received = _timestamp(outcome["outcome_received_at"], "outcome_received_at")
        computed = _timestamp(outcome["computed_at"], "outcome computed_at")
        if received <= _timestamp(
            decision["decision_materialized_at"], "decision materialized_at"
        ):
            raise V18Error("outcome was attached before decision sealing")
        if computed < received:
            raise V18Error("outcome record computation predates outcome receipt")
        observed1 = _strict_bool(outcome["rank1_outcome_observed"])
        observed2 = _strict_bool(outcome["rank2_outcome_observed"])
        return1 = _finite_or_none(outcome["rank1_oc_return_pct"], "rank1 outcome")
        return2 = _finite_or_none(outcome["rank2_oc_return_pct"], "rank2 outcome")
        if observed1 != (return1 is not None) or observed2 != (return2 is not None):
            raise V18Error("rank outcome flag/value presence differs")
        if decision["decision"] == "selected_rank1" and observed1:
            candidate_observed, gross = True, float(return1)
        elif decision["decision"] == "selected_rank2" and observed2:
            candidate_observed, gross = True, float(return2)
        else:
            candidate_observed, gross = False, 0.0
        if _strict_bool(outcome["candidate_outcome_observed"]) != candidate_observed:
            raise V18Error("candidate outcome flag does not recompute")
        expected = {
            "candidate_gross_return_pct": gross,
            "candidate_net20_return_pct": gross - (0.2 if candidate_observed else 0.0),
            "candidate_net40_return_pct": gross - (0.4 if candidate_observed else 0.0),
            "candidate_net60_return_pct": gross - (0.6 if candidate_observed else 0.0),
            "c00_top1_gross_return_pct": 0.0 if return1 is None else return1,
            "c00_top1_net20_return_pct": 0.0 if return1 is None else return1 - 0.2,
            "c00_top1_net40_return_pct": 0.0 if return1 is None else return1 - 0.4,
            "c00_top1_net60_return_pct": 0.0 if return1 is None else return1 - 0.6,
            "c02_rank2_gross_return_pct": 0.0 if return2 is None else return2,
            "c02_rank2_net20_return_pct": 0.0 if return2 is None else return2 - 0.2,
            "c02_rank2_net40_return_pct": 0.0 if return2 is None else return2 - 0.4,
            "c02_rank2_net60_return_pct": 0.0 if return2 is None else return2 - 0.6,
        }
        for field, expected_value in expected.items():
            observed = _finite_or_none(outcome[field], field)
            if observed is None or not _ieee_float_equal(observed, expected_value):
                raise V18Error(f"outcome {field} does not recompute")
    return rows


def attach_outcomes(
    decisions: Sequence[Mapping[str, Any]] | pd.DataFrame,
    outcome_manifests: (
        Mapping[str, Any]
        | str
        | Path
        | Sequence[Mapping[str, Any] | str | Path]
    ),
    *,
    outcome_raw_store_root: str | Path,
    existing_outcomes: Sequence[Mapping[str, Any]] | pd.DataFrame = (),
    computed_at: Any | None = None,
) -> list[dict[str, Any]]:
    """Attach only returns recomputed from bound official-PDF manifests."""

    decision_rows = validate_decision_records(decisions)
    existing = validate_outcome_records(existing_outcomes, decision_rows)
    if isinstance(outcome_manifests, (Mapping, str, Path)):
        manifest_inputs: list[Mapping[str, Any] | str | Path] = [outcome_manifests]
    else:
        manifest_inputs = list(outcome_manifests)
    if not manifest_inputs:
        raise V18Error("outcome manifest input is empty")
    if len(existing) + len(manifest_inputs) > len(decision_rows):
        raise V18Error("outcome manifests extend beyond decisions")
    expected_decisions = decision_rows[
        len(existing) : len(existing) + len(manifest_inputs)
    ]
    records = list(existing)
    previous = records[-1]["record_sha256"] if records else ZERO_SHA256
    for manifest_input, decision in zip(
        manifest_inputs, expected_decisions, strict=True
    ):
        manifest, manifest_hash = validate_outcome_manifest(
            manifest_input,
            decision,
            outcome_raw_store_root=outcome_raw_store_root,
        )
        if manifest["target_session"] != decision["session_date"]:
            raise V18Error("outcome manifests must follow the decision prefix")
        if isinstance(manifest_input, (str, Path)) and Path(manifest_input).resolve() != (
            OUTCOME_MANIFEST_DIR / f"{decision['session_date']}.json"
        ).resolve():
            raise V18Error("outcome manifest is not at its canonical repo path")
        rank1 = _finite_or_none(
            manifest["rank1_recomputed_oc_return_pct"], "rank1 outcome"
        )
        rank2 = _finite_or_none(
            manifest["rank2_recomputed_oc_return_pct"], "rank2 outcome"
        )
        observed1, observed2 = rank1 is not None, rank2 is not None
        if decision["decision"] == "selected_rank1" and observed1:
            candidate_observed, gross = True, float(rank1)
        elif decision["decision"] == "selected_rank2" and observed2:
            candidate_observed, gross = True, float(rank2)
        else:
            candidate_observed, gross = False, 0.0

        def net(value: float, observed: bool, bps: float) -> float:
            return value - bps / 100.0 if observed else 0.0

        computed = _operation_timestamp(computed_at, "outcome record computed_at")
        if computed < _timestamp(manifest["sealed_at"], "outcome manifest sealed_at"):
            raise V18Error("outcome record computation predates manifest seal")
        payload = {
            "schema_version": 1,
            "session_date": decision["session_date"],
            "decision_record_sha256": decision["record_sha256"],
            "outcome_source_sha256": manifest["source_sha256"],
            "outcome_manifest_sha256": manifest_hash,
            "outcome_received_at": manifest["source_received_at"],
            "computed_at": computed,
            "rank1_outcome_observed": observed1,
            "rank1_oc_return_pct": rank1,
            "rank2_outcome_observed": observed2,
            "rank2_oc_return_pct": rank2,
            "candidate_outcome_observed": candidate_observed,
            "candidate_gross_return_pct": gross,
            "candidate_net20_return_pct": net(gross, candidate_observed, 20.0),
            "candidate_net40_return_pct": net(gross, candidate_observed, 40.0),
            "candidate_net60_return_pct": net(gross, candidate_observed, 60.0),
            "c00_top1_gross_return_pct": 0.0 if rank1 is None else rank1,
            "c00_top1_net20_return_pct": net(0.0 if rank1 is None else rank1, observed1, 20.0),
            "c00_top1_net40_return_pct": net(0.0 if rank1 is None else rank1, observed1, 40.0),
            "c00_top1_net60_return_pct": net(0.0 if rank1 is None else rank1, observed1, 60.0),
            "c02_rank2_gross_return_pct": 0.0 if rank2 is None else rank2,
            "c02_rank2_net20_return_pct": net(0.0 if rank2 is None else rank2, observed2, 20.0),
            "c02_rank2_net40_return_pct": net(0.0 if rank2 is None else rank2, observed2, 40.0),
            "c02_rank2_net60_return_pct": net(0.0 if rank2 is None else rank2, observed2, 60.0),
            "protocol_sha256": decision["protocol_sha256"],
            "activation_payload_sha256": decision["activation_payload_sha256"],
            "activation_receipt_sha256": decision["activation_receipt_sha256"],
        }
        record = _sealed_record(
            payload,
            sequence_number=len(records),
            previous_record_sha256=previous,
        )
        records.append(record)
        previous = record["record_sha256"]
    return validate_outcome_records(records, decision_rows)


COMPLETED_MONTH_FIELDS = (
    "schema_version",
    "sequence_number",
    "completed_month",
    "created_at",
    "counted_scheduled_sessions",
    "complete_pair_days",
    "ordered_complete_pair_session_sha256",
    "ordered_difference_values_sha256",
    "monthly_median_rank1_minus_rank2_pct",
    "available",
    "protocol_sha256",
    "activation_payload_sha256",
    "activation_receipt_sha256",
    "previous_record_sha256",
    "record_sha256",
)


def validate_completed_month_records(
    records: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = validate_hash_chain(records, required_fields=COMPLETED_MONTH_FIELDS)
    previous_month: pd.Period | None = None
    for row in rows:
        month = _month(row["completed_month"], "completed_month")
        if previous_month is not None and month != previous_month + 1:
            raise V18Error("completed-month ledger is not consecutive")
        previous_month = month
        _timestamp(row["created_at"], "completed month created_at")
        count = int(row["complete_pair_days"])
        available = count >= MIN_COMPLETE_PAIRS
        if _strict_bool(row["available"]) != available:
            raise V18Error("completed-month availability does not recompute")
        median = _finite_or_none(
            row["monthly_median_rank1_minus_rank2_pct"], "monthly median"
        )
        if available != (median is not None):
            raise V18Error("completed-month median presence is invalid")
        for field in (
            "ordered_complete_pair_session_sha256",
            "ordered_difference_values_sha256",
            "protocol_sha256",
            "activation_payload_sha256",
            "activation_receipt_sha256",
        ):
            _require_sha(row[field], f"completed month {field}")
        if row["protocol_sha256"] != PROTOCOL_SHA256:
            raise V18Error("completed-month protocol hash changed")
    return rows


def _preflight_terminal_completed_month_ledger(
    records: Sequence[Mapping[str, Any]] | pd.DataFrame,
    decisions: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    calendar: pd.DatetimeIndex | None = None,
) -> bool:
    """Seal terminal month structure before any outcome ledger is opened.

    This intentionally validates only chain identity, represented-month
    coverage, counts, activation bindings, and chronology.  Outcome-derived
    medians and availability are recomputed only after outcome evidence opens.
    """

    decision_rows = validate_decision_records(decisions)
    if not decision_rows:
        return False
    scheduled = load_registered_calendar() if calendar is None else calendar
    first = _date(decision_rows[0]["session_date"], "first decision")
    terminal = deterministic_terminal_session(first, scheduled)
    expected = scheduled[(scheduled >= first) & (scheduled <= terminal)]
    observed = pd.DatetimeIndex(
        [_date(item["session_date"], "decision session") for item in decision_rows]
    )
    if len(observed) != len(expected) or not observed.equals(expected):
        return False
    rows = validate_hash_chain(records, required_fields=COMPLETED_MONTH_FIELDS)
    expected_months = [
        str(item)
        for item in pd.period_range(first.to_period("M"), terminal.to_period("M"), freq="M")
    ]
    if [item["completed_month"] for item in rows] != expected_months:
        raise V18Error("terminal completed-month ledger lacks exact represented months")
    activation_payload_sha = decision_rows[0]["activation_payload_sha256"]
    activation_receipt_sha = decision_rows[0]["activation_receipt_sha256"]
    for row, month_text in zip(rows, expected_months, strict=True):
        month = _month(row["completed_month"], "terminal completed month")
        if str(month) != month_text:
            raise V18Error("terminal completed-month chronology differs")
        represented = expected[expected.to_period("M") == month]
        if not len(represented):
            raise V18Error("terminal completed month has no represented sessions")
        count = row["counted_scheduled_sessions"]
        if (
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            or int(count) != len(represented)
        ):
            raise V18Error("terminal completed-month scheduled count differs")
        created = _timestamp(row["created_at"], "terminal completed month created_at")
        if created <= _cutoff(represented[-1]):
            raise V18Error("terminal completed-month record predates month close")
        if (
            row["protocol_sha256"] != PROTOCOL_SHA256
            or row["activation_payload_sha256"] != activation_payload_sha
            or row["activation_receipt_sha256"] != activation_receipt_sha
        ):
            raise V18Error("terminal completed-month authority binding differs")
        for field in (
            "ordered_complete_pair_session_sha256",
            "ordered_difference_values_sha256",
            "protocol_sha256",
            "activation_payload_sha256",
            "activation_receipt_sha256",
        ):
            _require_sha(row[field], f"terminal completed month {field}")
    return True


def build_completed_month_record(
    decisions: Sequence[Mapping[str, Any]] | pd.DataFrame,
    outcomes: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    completed_month: str,
    created_at: Any | None = None,
    capture_operational_timestamp: bool = False,
    existing_records: Sequence[Mapping[str, Any]] | pd.DataFrame = (),
) -> list[dict[str, Any]]:
    """Close one consecutive counted month for use only by later target months."""

    decision_rows = validate_decision_records(decisions)
    outcome_rows = validate_outcome_records(outcomes, decision_rows)
    existing = validate_completed_month_records(existing_records)
    month = _month(completed_month, "completed_month")
    if existing and month != _month(existing[-1]["completed_month"]) + 1:
        raise V18Error("completed month does not follow ledger")
    if capture_operational_timestamp and created_at is not None:
        raise V18Error("canonical completed-month timestamp is captured internally")
    created = (
        None
        if capture_operational_timestamp
        else _timestamp(created_at, "completed month created_at")
        if created_at is not None
        else None
    )
    if created is None and not capture_operational_timestamp:
        raise V18Error("pure completed-month construction requires created_at")
    pairs: list[tuple[str, float]] = []
    counted = 0
    outcome_by_date = {item["session_date"]: item for item in outcome_rows}
    selected_decisions = [
        item
        for item in decision_rows
        if _date(item["session_date"], "session").to_period("M") == month
    ]
    if not selected_decisions:
        raise V18Error("completed month has no counted sessions")
    calendar = load_registered_calendar()
    first_counted = _date(decision_rows[0]["session_date"], "first counted session")
    expected_sessions = calendar[
        (calendar.to_period("M") == month) & (calendar >= first_counted)
    ]
    observed_sessions = pd.DatetimeIndex(
        [_date(item["session_date"], "completed month session") for item in selected_decisions]
    )
    if not observed_sessions.equals(expected_sessions):
        raise V18Error("forward completed month does not cover every counted session")
    month_outcomes = [outcome_by_date.get(item["session_date"]) for item in selected_decisions]
    if any(item is None for item in month_outcomes):
        raise V18Error("cannot close month before every counted outcome is attached")
    latest_outcome_computed = max(
        _timestamp(item["computed_at"], "outcome computed_at")
        for item in month_outcomes
        if item is not None
    )
    for decision in selected_decisions:
        counted += 1
        outcome = outcome_by_date.get(decision["session_date"])
        if outcome is None:
            raise V18Error("cannot close month before every counted outcome is attached")
        complete = (
            bool(decision["source_complete"])
            and bool(decision["model_complete"])
            and decision["c00_rank1_code"] != decision["c02_rank2_code"]
            and bool(outcome["rank1_outcome_observed"])
            and bool(outcome["rank2_outcome_observed"])
        )
        if complete:
            difference = float(outcome["rank1_oc_return_pct"]) - float(
                outcome["rank2_oc_return_pct"]
            )
            pairs.append((decision["session_date"], difference))
    values = np.asarray([item[1] for item in pairs], dtype=float)
    complete_count = int(len(values))
    median = float(np.median(values)) if complete_count >= MIN_COMPLETE_PAIRS else None
    if capture_operational_timestamp:
        created = datetime.now(TOKYO)
    assert created is not None
    if created <= latest_outcome_computed:
        raise V18Error("completed month record does not follow its final outcome record")
    first = decision_rows[0] if decision_rows else None
    if first is None:
        raise V18Error("cannot close a month without decisions")
    payload = {
        "schema_version": 1,
        "completed_month": str(month),
        "created_at": created,
        "counted_scheduled_sessions": counted,
        "complete_pair_days": complete_count,
        "ordered_complete_pair_session_sha256": canonical_json_sha256(
            [item[0] for item in pairs]
        ),
        "ordered_difference_values_sha256": _numeric_sha(values),
        "monthly_median_rank1_minus_rank2_pct": median,
        "available": median is not None,
        "protocol_sha256": PROTOCOL_SHA256,
        "activation_payload_sha256": first["activation_payload_sha256"],
        "activation_receipt_sha256": first["activation_receipt_sha256"],
    }
    previous = existing[-1]["record_sha256"] if existing else ZERO_SHA256
    record = _sealed_record(
        payload,
        sequence_number=len(existing),
        previous_record_sha256=previous,
    )
    return validate_completed_month_records([*existing, record])


def validate_completed_month_coverage(
    records: Sequence[Mapping[str, Any]] | pd.DataFrame,
    decisions: Sequence[Mapping[str, Any]] | pd.DataFrame,
    outcomes: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> list[dict[str, Any]]:
    """Require and recompute every represented month, including terminal."""

    rows = validate_completed_month_records(records)
    decision_rows = validate_decision_records(decisions)
    outcome_rows = validate_outcome_records(outcomes, decision_rows)
    if not decision_rows:
        if rows:
            raise V18Error("completed-month ledger exists without decisions")
        return rows
    first_month = _date(
        decision_rows[0]["session_date"], "first decision"
    ).to_period("M")
    last_month = _date(
        decision_rows[-1]["session_date"], "last decision"
    ).to_period("M")
    expected_months = [
        str(item) for item in pd.period_range(first_month, last_month, freq="M")
    ]
    if [item["completed_month"] for item in rows] != expected_months:
        raise V18Error("completed-month ledger must include every represented month")
    reconstructed: list[dict[str, Any]] = []
    for observed in rows:
        rebuilt = build_completed_month_record(
            decision_rows,
            outcome_rows,
            completed_month=observed["completed_month"],
            created_at=observed["created_at"],
            existing_records=reconstructed,
        )
        if canonical_json_bytes(rebuilt[-1]) != canonical_json_bytes(observed):
            raise V18Error("completed-month record does not cleanly recompute")
        reconstructed = rebuilt
    return rows


def pair_history_from_ledgers(
    decisions: Sequence[Mapping[str, Any]] | pd.DataFrame,
    outcomes: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    include_initial_seed: bool = True,
) -> pd.DataFrame:
    """Reconstruct forward rank-pair history without any unlogged backfill."""

    decision_rows = validate_decision_records(decisions)
    outcome_rows = validate_outcome_records(outcomes, decision_rows)
    records: list[dict[str, Any]] = []
    for decision, outcome in zip(decision_rows, outcome_rows, strict=False):
        if not (decision["source_complete"] and decision["model_complete"]):
            continue
        records.extend(
            [
                {
                    "date": decision["session_date"],
                    "source_rank": 1,
                    "oc_return_pct": outcome["rank1_oc_return_pct"],
                },
                {
                    "date": decision["session_date"],
                    "source_rank": 2,
                    "oc_return_pct": outcome["rank2_oc_return_pct"],
                },
            ]
        )
    forward = _normalise_history(pd.DataFrame(records)) if records else pd.DataFrame(
        columns=["date", "source_rank", "oc_return_pct"]
    )
    if not include_initial_seed:
        return forward
    initial, _ = load_initial_pair_history()
    return _normalise_history(pd.concat([initial, forward], ignore_index=True))


def _evaluation_frame(
    decisions: Sequence[Mapping[str, Any]] | pd.DataFrame,
    outcomes: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    decision_rows = validate_decision_records(decisions)
    outcome_rows = validate_outcome_records(outcomes, decision_rows)
    records: list[dict[str, Any]] = []
    for decision, outcome in zip(decision_rows, outcome_rows, strict=False):
        records.append(
            {
                "session_date": pd.Timestamp(decision["session_date"]),
                "code": decision["candidate_selected_code"],
                "executed": bool(outcome["candidate_outcome_observed"]),
                "candidate_gross": float(outcome["candidate_gross_return_pct"]),
                "candidate_net20": float(outcome["candidate_net20_return_pct"]),
                "candidate_net40": float(outcome["candidate_net40_return_pct"]),
                "candidate_net60": float(outcome["candidate_net60_return_pct"]),
                "c00_gross": float(outcome["c00_top1_gross_return_pct"]),
                "c00_net20": float(outcome["c00_top1_net20_return_pct"]),
                "c00_net40": float(outcome["c00_top1_net40_return_pct"]),
                "c00_net60": float(outcome["c00_top1_net60_return_pct"]),
                "c02_gross": float(outcome["c02_rank2_gross_return_pct"]),
                "c02_net20": float(outcome["c02_rank2_net20_return_pct"]),
                "c02_net40": float(outcome["c02_rank2_net40_return_pct"]),
                "c02_net60": float(outcome["c02_rank2_net60_return_pct"]),
            }
        )
    if records:
        frame = pd.DataFrame(records).set_index("session_date")
    else:
        frame = pd.DataFrame(
            columns=[
                "code",
                "executed",
                "candidate_gross",
                "candidate_net20",
                "candidate_net40",
                "candidate_net60",
                "c00_gross",
                "c00_net20",
                "c00_net40",
                "c00_net60",
                "c02_gross",
                "c02_net20",
                "c02_net40",
                "c02_net60",
            ],
            index=pd.DatetimeIndex([], name="session_date"),
        )
    return frame, decision_rows, outcome_rows


def validate_outcome_evidence(
    decisions: Sequence[Mapping[str, Any]] | pd.DataFrame,
    outcomes: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    outcome_manifest_directory: str | Path,
    outcome_raw_store_root: str | Path,
    external_identity_registry: dict[tuple[int, int], str] | None = None,
) -> list[dict[str, Any]]:
    """Reparse every externally sealed target-day PDF and bind its ledger row."""

    identity_registry = (
        {} if external_identity_registry is None else external_identity_registry
    )
    decision_rows = validate_decision_records(decisions)
    outcome_rows = validate_outcome_records(outcomes, decision_rows)
    directory = Path(outcome_manifest_directory)
    if directory.resolve() != OUTCOME_MANIFEST_DIR.resolve():
        raise V18Error("outcome manifest directory is not the canonical repo path")
    # Outcomes are an append-only prefix of decisions.  Before terminal, a
    # freshly sealed decision legitimately has no target-day PDF yet; require
    # exactly the manifests already bound by the outcome prefix, not future
    # evidence that cannot exist.
    expected_names = [f"{item['session_date']}.json" for item in outcome_rows]
    if not expected_names and not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise V18Error("outcome manifest directory is missing")
    actual_entries = sorted(directory.iterdir(), key=lambda item: item.name)
    if any(not item.is_file() or item.is_symlink() for item in actual_entries):
        raise V18Error("outcome manifest directory contains a non-plain file")
    if sorted(item.name for item in actual_entries) != sorted(expected_names):
        raise V18Error("outcome manifest directory has missing or extra files")
    manifests: list[dict[str, Any]] = []
    for decision, outcome in zip(decision_rows, outcome_rows, strict=False):
        path = directory / f"{decision['session_date']}.json"
        manifest, digest = validate_outcome_manifest(
            path,
            decision,
            outcome_raw_store_root=outcome_raw_store_root,
            external_identity_registry=identity_registry,
        )
        if (
            outcome["outcome_manifest_sha256"] != digest
            or outcome["outcome_source_sha256"] != manifest["source_sha256"]
            or _timestamp(outcome["outcome_received_at"], "outcome_received_at")
            != _timestamp(manifest["source_received_at"], "source_received_at")
        ):
            raise V18Error("outcome ledger does not bind its official manifest")
        if _timestamp(outcome["computed_at"], "outcome computed_at") < _timestamp(
            manifest["sealed_at"], "outcome manifest sealed_at"
        ):
            raise V18Error("outcome record predates its manifest seal")
        manifests.append(manifest)
    return manifests


def _validate_terminal_predictor_outcome_cross_role(
    predictor_raw_records: Sequence[Mapping[str, Any]],
    predictor_shard_bindings: Sequence[Mapping[str, Any]],
    outcome_manifests: Sequence[Mapping[str, Any]],
    *,
    terminal_session: Any,
    allow_terminal_outcome_only: bool,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
    outcome_raw_store_root: str | Path,
) -> None:
    """Require every overlapping daily PDF role to bind identical bytes.

    Manual acquisition remains an explicit trusted boundary.  This check
    proves only that the separately retained predictor and outcome roles did
    not diverge after acquisition; it does not authenticate the JPX origin.
    """

    raw_records = [dict(item) for item in predictor_raw_records]
    bindings = [dict(item) for item in predictor_shard_bindings]
    if len(raw_records) != len(bindings):
        raise V18Error("terminal cross-role predictor arrays are misaligned")
    predictor_by_file: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for raw_record, binding in zip(raw_records, bindings, strict=True):
        file_name = str(raw_record["file"])
        if file_name in predictor_by_file:
            raise V18Error("terminal cross-role predictor filename is duplicated")
        predictor_by_file[file_name] = (raw_record, binding)
    terminal = _date(terminal_session, "terminal cross-role session")
    manifests = [dict(item) for item in outcome_manifests]
    unmatched: list[pd.Timestamp] = []
    seen_targets: set[pd.Timestamp] = set()
    for outcome in manifests:
        outcome_target = _date(
            outcome["target_session"], "terminal cross-role outcome session"
        )
        if outcome_target in seen_targets:
            raise V18Error("terminal cross-role outcome session is duplicated")
        seen_targets.add(outcome_target)
        file_name = str(outcome["source_file_name"])
        matched = predictor_by_file.get(file_name)
        # The terminal target has no following predictor session and is the
        # sole legitimate outcome-only daily PDF.
        if matched is None:
            if not allow_terminal_outcome_only or outcome_target != terminal:
                raise V18Error(
                    "terminal cross-role predictor PDF is missing before final session"
                )
            unmatched.append(outcome_target)
            continue
        raw_record, binding = matched
        exact = {
            "file": file_name,
            "url": outcome["source_url"],
            "byte_count": int(outcome["source_byte_count"]),
            "sha256": outcome["source_sha256"],
        }
        if any(raw_record[field] != expected for field, expected in exact.items()):
            raise V18Error(
                "terminal predictor/outcome roles bind different daily bytes"
            )
        shard_manifest, _ = _read_external_canonical_json(
            predictor_derived_store_root,
            binding["shard_manifest_object_key"],
            prefix=PREDICTOR_SHARD_OBJECT_PREFIX,
            label="terminal cross-role predictor shard manifest",
        )
        _validate_parsed_shard_manifest(
            shard_manifest,
            raw_record=raw_record,
            predictor_derived_store_root=predictor_derived_store_root,
            expected_chronology_class="forward",
            expected_raw_received_at=outcome["source_received_at"],
            decode_data=False,
        )
        _assert_external_objects_nonalias(
            predictor_raw_store_root,
            str(raw_record["object_key"]),
            PREDICTOR_OBJECT_PREFIX,
            outcome_raw_store_root,
            str(outcome["raw_source_object_key"]),
            OUTCOME_OBJECT_PREFIX,
        )
    expected_unmatched = [terminal] if allow_terminal_outcome_only else []
    if unmatched != expected_unmatched:
        raise V18Error(
            "terminal cross-role outcome-only session set differs from protocol"
        )


def _validate_deferred_state_evidence(
    decisions: Sequence[Mapping[str, Any]],
    *,
    expected_payload_sha256: str,
    expected_receipt_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Open outcome-derived monthly state only after outcome-blind preflight."""

    rows = [dict(item) for item in decisions]
    represented_months = sorted(
        {str(pd.Timestamp(item["session_date"]).to_period("M")) for item in rows}
    )
    if STATE_MANIFEST_DIR.is_symlink() or not STATE_MANIFEST_DIR.is_dir():
        raise V18Error("state manifest directory is missing")
    entries = sorted(STATE_MANIFEST_DIR.iterdir(), key=lambda item: item.name)
    expected_names = [f"{month}.json" for month in represented_months]
    if (
        any(not item.is_file() or item.is_symlink() for item in entries)
        or sorted(item.name for item in entries) != expected_names
    ):
        raise V18Error("state manifest directory has missing or extra files")
    result: dict[str, dict[str, Any]] = {}
    for month in represented_months:
        state = validate_state_manifest(
            read_json(STATE_MANIFEST_DIR / f"{month}.json")
        )
        if (
            state["activation_payload_sha256"] != expected_payload_sha256
            or state["activation_receipt_sha256"] != expected_receipt_sha256
        ):
            raise V18Error(f"state activation hashes differ for {month}")
        month_rows = [
            item
            for item in rows
            if str(pd.Timestamp(item["session_date"]).to_period("M")) == month
        ]
        selection_fields = (
            "three_prior_calendar_months",
            "three_complete_pair_day_counts",
            "three_month_medians_pct",
            "state_value_pct",
            "selected_source_rank",
        )
        for decision in month_rows:
            if decision["state_manifest_sha256"] != state[
                "state_manifest_sha256"
            ]:
                raise V18Error("decision does not bind its canonical monthly state")
            if any(decision[field] != state[field] for field in selection_fields):
                raise V18Error("decision selection differs from monthly state")
            if (
                decision["c00_fold_manifest_sha256"]
                != state["c00_fold_manifest_sha256"]
                or decision["fold_model_bundle_file_sha256"]
                != state["fold_model_bundle_file_sha256"]
            ):
                raise V18Error("monthly state/decision fold binding differs")
            if _timestamp(decision["computed_at"], "decision computed_at") < (
                _timestamp(state["created_at"], "state created_at")
            ):
                raise V18Error("decision timestamp predates canonical monthly state")
        result[month] = state
    return result


def _validate_exact_month_authority_directories(
    represented_months: Sequence[str],
) -> None:
    """Pin the exact local month-source/fold/bundle authority name sets."""

    months = [
        str(_month(value, "represented authority month"))
        for value in represented_months
    ]
    if months != sorted(set(months)):
        raise V18Error("represented authority month set is not exact/sorted")
    expected = {f"{month}.json" for month in months}
    expected_with_stages = expected | {f".{name}.staging" for name in expected}
    for directory, label in (
        (MONTH_SOURCE_MANIFEST_DIR, "month-source manifest"),
        (FOLD_MANIFEST_DIR, "fold manifest"),
        (FOLD_MODEL_DIR, "fold model bundle"),
    ):
        try:
            directory_stat = os.stat(directory, follow_symlinks=False)
        except OSError as exc:
            raise V18Error(f"terminal {label} directory is missing") from exc
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or directory_stat.st_uid != os.geteuid()
            or stat.S_IMODE(directory_stat.st_mode) != 0o700
        ):
            raise V18Error(f"terminal {label} directory metadata changed")
        observed_before = set(os.listdir(directory))
        if (
            not expected.issubset(observed_before)
            or not observed_before.issubset(expected_with_stages)
        ):
            raise V18Error(f"terminal {label} directory name set changed")
        # Heal only the registered final+canonical-stage same-inode crash case,
        # then re-pin the exact final-only set through the directory descriptor.
        for name in sorted(expected):
            _read_local_authority_bytes(
                directory / name, label=f"terminal {label} {name}"
            )
        try:
            directory_fd = os.open(
                directory,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise V18Error(f"terminal {label} directory cannot be pinned") from exc
        try:
            pinned = os.fstat(directory_fd)
            if (
                (pinned.st_dev, pinned.st_ino)
                != (directory_stat.st_dev, directory_stat.st_ino)
                or pinned.st_uid != os.geteuid()
                or stat.S_IMODE(pinned.st_mode) != 0o700
                or set(os.listdir(directory_fd)) != expected
            ):
                raise V18Error(f"terminal {label} directory changed while pinned")
            for name in sorted(expected):
                metadata = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                    or metadata.st_nlink != 1
                ):
                    raise V18Error(f"terminal {label} member metadata changed")
            entry_after = os.stat(directory, follow_symlinks=False)
            if (
                (entry_after.st_dev, entry_after.st_ino)
                != (pinned.st_dev, pinned.st_ino)
                or set(os.listdir(directory_fd)) != expected
            ):
                raise V18Error(f"terminal {label} directory changed after scan")
        finally:
            os.close(directory_fd)


def validate_predictor_evidence(
    decisions: Sequence[Mapping[str, Any]] | pd.DataFrame,
    scores: pd.DataFrame | None,
    *,
    source_manifest_directory: str | Path,
    predictor_raw_store_root: str | Path,
    predictor_derived_store_root: str | Path,
    external_identity_registry: dict[tuple[int, int], str] | None = None,
) -> dict[str, Any]:
    """Clean-replay every counted C00 input, monthly fold, score, and top two."""

    _validate_external_store_disjointness(
        predictor_raw_store_root, predictor_derived_store_root
    )
    if sha256_file(ROOT / JPX_PARSER_PATH) != JPX_PARSER_SHA256:
        raise V18Error("terminal predictor parser source SHA changed")
    decision_rows = validate_decision_records(decisions)
    payload_value, payload_hash, _, receipt_hash = validate_canonical_activation_artifacts()
    validate_activation_payload(
        payload_value,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
    )
    if any(
        item["activation_payload_sha256"] != payload_hash
        or item["activation_receipt_sha256"] != receipt_hash
        for item in decision_rows
    ):
        raise V18Error("terminal decision activation hashes differ from artifacts")
    first_decision = decision_rows[0]
    terminal_calendar = load_registered_calendar()
    first_session = _date(first_decision["session_date"], "first decision session")
    terminal_session = deterministic_terminal_session(first_session, terminal_calendar)
    terminal_denominator = terminal_calendar[
        (terminal_calendar >= first_session) & (terminal_calendar <= terminal_session)
    ]
    validate_activation_context(
        {
            "activation_payload_sha256": first_decision["activation_payload_sha256"],
            "activation_receipt_sha256": first_decision["activation_receipt_sha256"],
            "activation_receipt_commit_sha": first_decision[
                "activation_receipt_commit_sha"
            ],
            "activation_receipt_commit_url": first_decision[
                "activation_receipt_commit_url"
            ],
            "activation_receipt_commit_committed_at": first_decision[
                "activation_receipt_commit_committed_at"
            ],
            "activation_receipt_commit_observed_at": first_decision[
                "activation_receipt_commit_observed_at"
            ],
            "branch_tip_sha_when_receipt_observed": first_decision[
                "branch_tip_sha_when_receipt_observed"
            ],
            "activation_receipt_file_sha256": first_decision[
                "activation_receipt_file_sha256"
            ],
            "receipt_commit_observation": first_decision[
                "receipt_commit_observation"
            ],
            "receipt_branch_observation": first_decision[
                "receipt_branch_observation"
            ],
            "activation_receipt_workflow_run_id": first_decision[
                "activation_receipt_workflow_run_id"
            ],
            "activation_receipt_workflow_run_updated_at": first_decision[
                "activation_receipt_workflow_run_updated_at"
            ],
            "activation_receipt_workflow_run_observed_at": first_decision[
                "activation_receipt_workflow_run_observed_at"
            ],
            "receipt_workflow_run_observation": first_decision[
                "receipt_workflow_run_observation"
            ],
            "first_counted_session": str(first_session.date()),
            "first_counted_predecessor_session": str(
                _latest_required_predictor_source_session(first_session).date()
            ),
            "terminal_session": str(terminal_session.date()),
            "terminal_scheduled_sessions": len(terminal_denominator),
            "represented_calendar_months": terminal_denominator.to_period("M").nunique(),
            "calendar_sha256": CALENDAR_SHA256,
            "production_model_changed": False,
            "orders_allowed": False,
        }
    )
    directory = Path(source_manifest_directory)
    if directory.resolve() != SOURCE_MANIFEST_DIR.resolve():
        raise V18Error("predictor source manifest directory is not canonical")
    expected_manifest_names = [f"{item['session_date']}.json" for item in decision_rows]
    if directory.is_symlink() or not directory.is_dir():
        raise V18Error("predictor source manifest directory is missing")
    actual_entries = sorted(directory.iterdir(), key=lambda item: item.name)
    if any(not item.is_file() or item.is_symlink() for item in actual_entries):
        raise V18Error("predictor source manifest directory contains a non-plain file")
    if sorted(item.name for item in actual_entries) != sorted(expected_manifest_names):
        raise V18Error("predictor source manifest directory has missing or extra files")
    represented_months = sorted(
        {
            str(pd.Timestamp(item["session_date"]).to_period("M"))
            for item in decision_rows
        }
    )
    _validate_exact_month_authority_directories(represented_months)
    if scores is None:
        # State is forward-performance-bearing.  The outcome-blind terminal
        # phase must finish raw/shard/snapshot/cache and checkpoint validation
        # before even enumerating the state directory.
        state_by_month: dict[str, dict[str, Any]] = {}
        score_groups: dict[str, pd.DataFrame] = {}
    else:
        state_by_month = _validate_deferred_state_evidence(
            decision_rows,
            expected_payload_sha256=payload_hash,
            expected_receipt_sha256=receipt_hash,
        )
    if scores is not None and scores.empty:
        if scores.columns.tolist() != list(SCORE_FIELDS):
            raise V18Error("empty score ledger must retain the registered header")
        validate_score_ledger(scores, allow_empty=True)
        score_groups = {}
    elif scores is not None:
        score_rows = validate_score_ledger(scores)
        score_sessions = pd.to_datetime(score_rows["session_date"], errors="coerce")
        score_groups = {
            str(session.date()): group.drop(columns="_session").reset_index(drop=True)
            for session, group in score_rows.assign(_session=score_sessions).groupby(
                "_session", sort=True
            )
        }
    source_manifest_pairs: list[dict[str, Any]] = []
    semantic_sets: dict[str, list[dict[str, Any]]] = {
        "target_date_scoring_input_semantic_sha256": [],
        "target_slice_semantic_sha256": [],
    }
    manifests: list[dict[str, Any]] = []
    anchor_summary = payload_value["predictor_cache_anchor"]
    anchor_manifest, _ = _read_external_canonical_json(
        predictor_derived_store_root,
        anchor_summary["snapshot_manifest_object_key"],
        prefix=CACHE_ANCHOR_OBJECT_PREFIX,
        label="terminal predictor cache anchor manifest",
    )
    _, observed_anchor_summary, _ = validate_predictor_cache_anchor(
        anchor_manifest,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
        require_direct_clean_room_reparse=False,
    )
    if observed_anchor_summary != anchor_summary:
        raise V18Error("terminal predictor anchor differs from payload B")
    anchor_raw = [dict(item) for item in anchor_manifest["raw_sources"]]
    anchor_shards = [dict(item) for item in anchor_manifest["parsed_shards"]]
    if len(anchor_raw) != len(anchor_shards):
        raise V18Error("terminal anchor raw/shard binding count differs")
    raw_union: dict[str, dict[str, Any]] = {}
    shard_union: dict[str, dict[str, Any]] = {}
    for item, binding in zip(anchor_raw, anchor_shards, strict=True):
        object_key = item["object_key"]
        if object_key in raw_union:
            raise V18Error("terminal anchor repeats a raw object key")
        raw_union[object_key] = item
        shard_union[object_key] = binding
    predecessor_raw = anchor_manifest["raw_sources"]
    predecessor_shards = anchor_manifest["parsed_shards"]
    for decision in decision_rows:
        target = decision["session_date"]
        path = directory / f"{target}.json"
        if not path.is_file():
            raise V18Error(f"predictor source manifest is missing: {target}")
        value, digest = validate_source_manifest(
            read_json(path),
            session_date=target,
            predictor_raw_store_root=predictor_raw_store_root,
            first_counted_session_value=first_session,
            require_predecessor_decision=False,
        )
        if decision["source_manifest_sha256"] != digest:
            raise V18Error("decision does not bind its predictor source manifest")
        if _strict_bool(value["source_complete"]):
            if _source_set_records(value)[: len(predecessor_raw)] != predecessor_raw:
                raise V18Error("terminal predictor raw chain forks its predecessor")
            if value["parsed_shards"][: len(predecessor_shards)] != predecessor_shards:
                raise V18Error("terminal parsed-shard chain forks its predecessor")
            predecessor_raw = _source_set_records(value)
            predecessor_shards = value["parsed_shards"]
        source_manifest_pairs.append(
            {"target_session": target, "source_manifest_sha256": digest}
        )
        for field in semantic_sets:
            semantic_sets[field].append(
                {"target_session": target, field: value[field]}
            )
        for item, binding in zip(
            _source_set_records(value), value["parsed_shards"], strict=True
        ):
            prior = raw_union.setdefault(item["object_key"], item)
            if prior != item:
                raise V18Error("predictor raw object metadata conflicts across manifests")
            prior_binding = shard_union.setdefault(item["object_key"], dict(binding))
            if prior_binding != dict(binding):
                raise V18Error("predictor shard binding conflicts across manifests")
        manifests.append(value)
    ordered_union = [raw_union[key] for key in sorted(raw_union)]
    if ordered_union:
        union_prices = _terminal_reparse_predictor_shards_once(
            ordered_union,
            [shard_union[item["object_key"]] for item in ordered_union],
            predictor_raw_store_root=predictor_raw_store_root,
            predictor_derived_store_root=predictor_derived_store_root,
            identity_registry=external_identity_registry,
        )
    else:
        union_prices = pd.DataFrame(columns=PARSED_PRICE_COLUMNS)
    if ordered_union:
        anchor_names = {
            Path(item["file"]).with_suffix(".txt").name
            for item in anchor_manifest["raw_sources"]
        }
        anchor_direct = union_prices.loc[
            union_prices["source_file"].astype(str).isin(anchor_names)
        ].copy()
        anchor_snapshot_payload = _external_object_bytes(
            predictor_derived_store_root,
            anchor_summary["cumulative_snapshot_object_key"],
            prefix=CACHE_ANCHOR_OBJECT_PREFIX,
            label="terminal cache anchor snapshot",
        )
        anchor_snapshot = decode_canonical_frame_jsonl(
            anchor_snapshot_payload,
            PARSED_PRICE_COLUMNS,
            label="terminal cache anchor snapshot",
        )
        if not _coerce_jsonl_frame(
            anchor_direct,
            PARSED_PRICE_COLUMNS,
            label="terminal direct anchor subset",
        ).equals(anchor_snapshot):
            raise V18Error("terminal raw union differs from payload-B anchor snapshot")
        anchored_equivalence = validate_compact_consumer_equivalence_receipt(
            anchor_manifest["compact_consumer_equivalence_receipt"]
        )
        terminal_equivalence = build_compact_consumer_equivalence_receipt(
            anchor_direct,
            _coerce_model_price_frame(
                anchor_direct, label="terminal anchor compact-consumer prefix"
            ),
            synthetic_target_session=anchored_equivalence[
                "synthetic_target_session"
            ],
        )
        if terminal_equivalence != anchored_equivalence:
            raise V18Error(
                "terminal raw31/compact12 consumer proof differs from payload B"
            )
    for manifest in manifests:
        if not _strict_bool(manifest["source_complete"]):
            continue
        target = manifest["target_session"]
        cache_manifest, cache_manifest_payload = _read_external_canonical_json(
            predictor_derived_store_root,
            manifest["g0_panel_cache_manifest_object_key"],
            prefix=G0_PANEL_CACHE_OBJECT_PREFIX,
            label="terminal target-slice cache manifest",
        )
        cache, _, expected_cache_key, _ = _validate_g0_cache_manifest(
            cache_manifest,
            predictor_derived_store_root=predictor_derived_store_root,
            expected_target_session=target,
            expected_latest_source_session=manifest[
                "latest_required_source_session"
            ],
            expected_source_set_sha256=manifest["source_set_sha256"],
            expected_parsed_shard_set_sha256=manifest[
                "parsed_shard_set_sha256"
            ],
        )
        if (
            expected_cache_key != manifest["g0_panel_cache_manifest_object_key"]
            or len(cache_manifest_payload)
            != int(manifest["g0_panel_cache_manifest_byte_count"])
            or hashlib.sha256(cache_manifest_payload).hexdigest()
            != manifest["g0_panel_cache_manifest_file_sha256"]
            or cache["cache_manifest_sha256"]
            != manifest["g0_panel_cache_manifest_sha256"]
        ):
            raise V18Error("terminal target-slice cache exact binding changed")
    fold_by_month: dict[str, tuple[str | None, str | None]] = {}
    replayed_score_sessions: set[str] = set()
    outcome_blind_score_expectations: list[dict[str, Any]] = []
    replayed_months: set[str] = set()
    for decision, manifest in zip(decision_rows, manifests, strict=True):
        target = decision["session_date"]
        month = str(pd.Timestamp(target).to_period("M"))
        resolution = decision["checkpoint_resolution"]
        state = state_by_month.get(month)
        fold_pair = (
            decision["c00_fold_manifest_sha256"],
            decision["fold_model_bundle_file_sha256"],
        )
        prior_pair = fold_by_month.setdefault(month, fold_pair)
        if prior_pair != fold_pair:
            raise V18Error("decision fold pair changed within target month")
        if state is not None and decision["state_manifest_sha256"] != state[
            "state_manifest_sha256"
        ]:
            raise V18Error("decision does not bind its canonical monthly state")
        computed = _timestamp(decision["computed_at"], "decision computed_at")
        if state is not None and computed < _timestamp(
            state["created_at"], "state created_at"
        ):
            raise V18Error("decision timestamp predates canonical monthly state")
        if (
            manifest["source_received_at"] is not None
            and computed < _timestamp(
            manifest["source_received_at"], "source_received_at"
            )
        ):
            raise V18Error("decision timestamp predates predictor receipt")
        if not _strict_bool(manifest["source_complete"]):
            raise V18Error("counted terminal source is incomplete")
        parsed_prices = union_prices.loc[
            union_prices["source_file"].astype(str).isin(
                {Path(name).with_suffix(".txt").name for name in manifest["source_files"]}
            )
        ].copy()
        if len(parsed_prices) != int(manifest["parsed_row_count"]):
            raise V18Error("terminal predictor subset parsed row count differs")
        direct_full_panel = build_forward_c00_panel(parsed_prices, target)
        panel = _coerce_jsonl_frame(
            direct_full_panel.loc[
                pd.to_datetime(direct_full_panel["date"]).eq(_date(target, "target"))
            ],
            G0_PANEL_COLUMNS,
            label="terminal direct G0 target slice",
        )
        del direct_full_panel
        validate_source_manifest(
            manifest,
            session_date=target,
            predictor_raw_store_root=predictor_raw_store_root,
            parsed_prices=parsed_prices,
            panel=panel,
            first_counted_session_value=first_session,
            require_predecessor_decision=False,
        )
        cache_manifest, _ = _read_external_canonical_json(
            predictor_derived_store_root,
            manifest["g0_panel_cache_manifest_object_key"],
            prefix=G0_PANEL_CACHE_OBJECT_PREFIX,
            label="terminal target-slice cache manifest",
        )
        _, cached_target_slice, _, _ = _validate_g0_cache_manifest(
            cache_manifest,
            predictor_derived_store_root=predictor_derived_store_root,
            expected_target_session=target,
            expected_latest_source_session=manifest[
                "latest_required_source_session"
            ],
            expected_source_set_sha256=manifest["source_set_sha256"],
            expected_parsed_shard_set_sha256=manifest[
                "parsed_shard_set_sha256"
            ],
        )
        if not panel.equals(cached_target_slice):
            raise V18Error("terminal target-slice cache differs from raw computation")
        if fold_pair == (None, None):
            raise V18Error("counted terminal decision lacks its monthly fold")
        fold_path = FOLD_MANIFEST_DIR / f"{month}.json"
        bundle_path = FOLD_MODEL_DIR / f"{month}.json"
        if not fold_path.is_file() or not bundle_path.is_file():
            raise V18Error("clean-room C00 fold artifact is missing")
        fold = read_json(fold_path)
        bundle = read_json(bundle_path)
        validate_fold_manifest(fold, bundle)
        if (
            fold["fold_manifest_sha256"] != fold_pair[0]
            or sha256_file(bundle_path) != fold_pair[1]
        ):
            raise V18Error("decision fold pair differs from canonical files")
        if computed is not None and computed < _timestamp(
            fold["fit_completed_at"], "fold fit_completed_at"
        ):
            raise V18Error("decision timestamp predates monthly fold completion")
        if month not in replayed_months:
            month_source_path = MONTH_SOURCE_MANIFEST_DIR / f"{month}.json"
            if not month_source_path.is_file():
                raise V18Error("clean-room monthly source manifest is missing")
            month_source = read_json(month_source_path)
            if (
                fold["month_source_manifest_sha256"]
                != month_source.get("month_source_manifest_sha256")
            ):
                raise V18Error("fold differs from canonical month-source manifest")
            month_source_names = {
                Path(name).with_suffix(".txt").name
                for name in month_source["source_files"]
            }
            month_prices = union_prices.loc[
                union_prices["source_file"].astype(str).isin(month_source_names)
            ].copy()
            month_model_prices = _coerce_model_price_frame(
                month_prices, label="terminal month-source compact projection"
            )
            month_seal = _date(month_source["seal_session"], "month seal session")
            month_raw_panel = build_forward_c00_panel(month_prices, month_seal)
            month_raw_digest = _exact_g0_frame_digest(month_raw_panel)
            month_training_panel = _coerce_jsonl_frame(
                month_raw_panel.loc[
                    pd.to_datetime(month_raw_panel["date"]).lt(
                        pd.Period(month, freq="M").start_time
                    )
                ],
                G0_PANEL_COLUMNS,
                label="terminal raw31 month-source training panel",
            )
            del month_raw_panel
            gc.collect()
            month_compact_panel = build_forward_c00_panel(
                month_model_prices, month_seal
            )
            month_compact_digest = _exact_g0_frame_digest(month_compact_panel)
            if month_compact_digest != month_raw_digest:
                raise V18Error(
                    "terminal month full31/compact12 consumer panels differ"
                )
            compact_training_panel = _coerce_jsonl_frame(
                month_compact_panel.loc[
                    pd.to_datetime(month_compact_panel["date"]).lt(
                        pd.Period(month, freq="M").start_time
                    )
                ],
                G0_PANEL_COLUMNS,
                label="terminal compact12 month-source training panel",
            )
            del month_compact_panel
            gc.collect()
            if not month_training_panel.equals(compact_training_panel):
                raise V18Error(
                    "terminal month training rows differ across raw/compact inputs"
                )
            model_month_semantic = model_price_semantic_sha256(month_model_prices)
            training_month_semantic = semantic_frame_sha256(
                month_training_panel, G0_PANEL_COLUMNS
            )
            validate_month_source_manifest(
                month_source,
                predictor_raw_store_root=predictor_raw_store_root,
                predictor_derived_store_root=predictor_derived_store_root,
                training_panel=month_training_panel,
                model_prices=month_model_prices,
                precomputed_model_semantic_sha256=model_month_semantic,
                precomputed_training_semantic_sha256=training_month_semantic,
            )
            started = _timestamp(fold["fit_started_at"], "fold fit_started_at")
            completed = _timestamp(fold["fit_completed_at"], "fold fit_completed_at")
            _, reconstructed_fold, reconstructed_bundle = _build_fold(
                month_training_panel,
                pd.Period(month, freq="M"),
                fit_started_at=started,
                fit_completed_at=completed,
                bundle_created_at=bundle["created_at"],
                sealed_at=fold["sealed_at"],
                runtime_lock_verified_at=fold["runtime_lock_verified_at"],
                terminal_replay=True,
                first_counted_session_value=decision_rows[0]["session_date"],
                activation_observed_at=decision_rows[0][
                    "activation_receipt_commit_observed_at"
                ],
                month_source_manifest=month_source,
            )
            if canonical_json_bytes(reconstructed_bundle) != canonical_json_bytes(bundle):
                raise V18Error("clean-room numeric C00 fold bundle differs")
            if canonical_json_bytes(reconstructed_fold) != canonical_json_bytes(fold):
                raise V18Error("clean-room C00 fold manifest differs")
            replayed_months.add(month)
        observed_scores = score_groups.get(target)
        replayed, _, _ = freeze_c00_top2(
            panel,
            target,
            score_generated_at=(
                decision["computed_at"]
                if observed_scores is None
                else observed_scores["score_generated_at"].iloc[0]
            ),
            runtime_lock_verified_at=(
                fold["runtime_lock_verified_at"]
                if observed_scores is None
                else observed_scores["runtime_lock_verified_at"].iloc[0]
            ),
            terminal_replay=True,
            source_manifest_sha256=manifest["source_manifest_sha256"],
            first_counted_session_value=decision_rows[0]["session_date"],
            activation_observed_at=decision_rows[0][
                "activation_receipt_commit_observed_at"
            ],
            return_bundle=True,
            model_bundle=bundle,
            fold_manifest=fold,
        )
        expectation_columns = (
            "session_date",
            "source_rank",
            "code",
            "name",
            "model_score",
            "feature_source_max_date",
            "source_manifest_sha256",
            "c00_fold_manifest_sha256",
        )
        outcome_blind_score_expectations.extend(
            replayed.loc[:, list(expectation_columns)].to_dict(orient="records")
        )
        if observed_scores is not None:
            if state is None:
                raise V18Error("score validation lacks deferred monthly state")
            if semantic_score_hash(replayed) != semantic_score_hash(observed_scores):
                raise V18Error("clean-room C00 score/top-two replay differs")
            score_generated = _timestamp(
                observed_scores["score_generated_at"].iloc[0], "score_generated_at"
            )
            score_prerequisites = (
                _timestamp(manifest["sealed_at"], "source manifest sealed_at"),
                _timestamp(fold["sealed_at"], "fold sealed_at"),
                _timestamp(state["created_at"], "state created_at"),
                _timestamp(
                    decision["activation_receipt_commit_observed_at"],
                    "activation_receipt_commit_observed_at",
                ),
            )
            score_verified = _timestamp(
                observed_scores["runtime_lock_verified_at"].iloc[0],
                "score runtime_lock_verified_at",
            )
            if (
                score_verified < max(score_prerequisites)
                or score_generated < score_verified
                or (computed is not None and computed < score_generated)
            ):
                raise V18Error("score/decision timestamps violate PIT causality")
            replayed_score_sessions.add(target)
        if resolution == "primary" and _strict_bool(decision["model_complete"]):
            ranked = replayed.set_index("source_rank")
            if (
                decision["c00_rank1_code"] != str(ranked.loc[1, "code"])
                or decision["c02_rank2_code"] != str(ranked.loc[2, "code"])
                or float(decision["c00_rank1_score"])
                != float(ranked.loc[1, "model_score"])
                or float(decision["c02_rank2_score"])
                != float(ranked.loc[2, "model_score"])
            ):
                raise V18Error("decision pair differs from clean-room C00 replay")
            if scores is not None and observed_scores is None:
                raise V18Error("model-complete decision lacks its score pair")
    for month, fold_pair in fold_by_month.items():
        if fold_pair == (None, None):
            raise V18Error("counted fold-set contains a null monthly authority")
        if month in replayed_months:
            continue
        fold_path = FOLD_MANIFEST_DIR / f"{month}.json"
        bundle_path = FOLD_MODEL_DIR / f"{month}.json"
        if not fold_path.is_file() or not bundle_path.is_file():
            raise V18Error("sealed counted month fold artifact is missing")
        fold = read_json(fold_path)
        bundle = read_json(bundle_path)
        validate_fold_manifest(fold, bundle)
        if (
            fold["fold_manifest_sha256"] != fold_pair[0]
            or sha256_file(bundle_path) != fold_pair[1]
        ):
            raise V18Error("sealed counted month fold pair changed")
    if scores is not None and set(score_groups) != replayed_score_sessions:
        raise V18Error("score ledger contains an uncounted or unreplayed session")
    fold_months = sorted(
        {
            str(pd.Timestamp(item["session_date"]).to_period("M"))
            for item in decision_rows
        }
    )
    fold_manifest_set = [
        {
            "target_month": month,
            "fold_manifest_sha256": fold_by_month.get(month, (None, None))[0],
        }
        for month in fold_months
    ]
    fold_bundle_set = [
        {
            "target_month": month,
            "fold_model_bundle_file_sha256": fold_by_month.get(month, (None, None))[1],
        }
        for month in fold_months
    ]
    return {
        "_outcome_blind_score_expectations": outcome_blind_score_expectations,
        "_terminal_predictor_raw_records": ordered_union,
        "_terminal_predictor_shard_bindings": [
            shard_union[item["object_key"]] for item in ordered_union
        ],
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "predictor_source_manifest_set_sha256": canonical_json_sha256(
            source_manifest_pairs
        ),
        "predictor_raw_source_set_sha256": canonical_json_sha256(ordered_union),
        "predictor_unique_raw_object_count": len(ordered_union),
        "predictor_parser_sha256": JPX_PARSER_SHA256,
        "predictor_parsed_shard_binding_set_sha256": canonical_json_sha256(
            [
                {
                    "target_session": item["target_session"],
                    "parsed_shard_set_sha256": manifest[
                        "parsed_shard_set_sha256"
                    ],
                }
                for item, manifest in zip(
                    source_manifest_pairs, manifests, strict=True
                )
            ]
        ),
        "predictor_target_slice_semantic_set_sha256": canonical_json_sha256(
            semantic_sets["target_slice_semantic_sha256"]
        ),
        "predictor_target_date_scoring_input_semantic_set_sha256": canonical_json_sha256(
            semantic_sets["target_date_scoring_input_semantic_sha256"]
        ),
        "c00_fold_manifest_set_sha256": canonical_json_sha256(fold_manifest_set),
        "c00_fold_model_bundle_file_set_sha256": canonical_json_sha256(
            fold_bundle_set
        ),
        "v17_c00_protocol_sha256": V17_BINDINGS["protocol"][1],
        "v17_c00_runner_sha256": V17_BINDINGS["runner"][1],
    }


def validate_deferred_score_evidence(
    scores: pd.DataFrame,
    expectations: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Open and compare the score ledger only after outcome-blind raw replay."""

    observed = validate_score_ledger(scores, allow_empty=not expectations)
    validate_score_session_authority(observed)
    expected_rows = [dict(item) for item in expectations]
    identity_fields = (
        "session_date",
        "source_rank",
        "code",
        "name",
        "model_score",
        "feature_source_max_date",
        "source_manifest_sha256",
        "c00_fold_manifest_sha256",
    )
    observed_projection = observed.loc[:, list(identity_fields)].copy()
    expected_projection = pd.DataFrame(expected_rows, columns=identity_fields)
    for frame in (observed_projection, expected_projection):
        if not frame.empty:
            frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.strftime(
                "%Y-%m-%d"
            )
            frame["feature_source_max_date"] = pd.to_datetime(
                frame["feature_source_max_date"]
            ).dt.strftime("%Y-%m-%d")
            frame["source_rank"] = frame["source_rank"].astype(int)
            frame["code"] = frame["code"].astype(str)
    observed_projection = observed_projection.sort_values(
        ["session_date", "source_rank"], kind="stable"
    ).reset_index(drop=True)
    expected_projection = expected_projection.sort_values(
        ["session_date", "source_rank"], kind="stable"
    ).reset_index(drop=True)
    if len(observed_projection) != len(expected_projection):
        raise V18Error("deferred score ledger has a missing/extra pair")
    for index in range(len(expected_projection)):
        for field in identity_fields:
            left = observed_projection.at[index, field]
            right = expected_projection.at[index, field]
            if field == "model_score":
                if not _ieee_float_equal(left, right):
                    raise V18Error("deferred score model bits differ from raw replay")
            elif left != right:
                raise V18Error(f"deferred score binding differs: {field}")
    decision_by_session = {
        str(item["session_date"]): dict(item) for item in validate_decision_records(decisions)
    }
    for session, group in observed.groupby(
        pd.to_datetime(observed["session_date"]).dt.strftime("%Y-%m-%d"),
        sort=True,
    ):
        decision = decision_by_session[session]
        source = read_json(SOURCE_MANIFEST_DIR / f"{session}.json")
        month = str(pd.Timestamp(session).to_period("M"))
        fold = read_json(FOLD_MANIFEST_DIR / f"{month}.json")
        state = read_json(STATE_MANIFEST_DIR / f"{month}.json")
        generated = _timestamp(group["score_generated_at"].iloc[0], "score generated")
        verified = _timestamp(
            group["runtime_lock_verified_at"].iloc[0], "score runtime verified"
        )
        prerequisites = (
            _timestamp(source["sealed_at"], "source sealed"),
            _timestamp(fold["sealed_at"], "fold sealed"),
            _timestamp(state["created_at"], "state created"),
            _timestamp(
                decision["activation_receipt_commit_observed_at"],
                "activation observed",
            ),
        )
        if (
            verified < max(prerequisites)
            or generated < verified
            or _timestamp(decision["computed_at"], "decision computed") < generated
        ):
            raise V18Error("deferred score timestamp DAG changed")
    return observed


def evaluate(
    decisions: Sequence[Mapping[str, Any]] | pd.DataFrame,
    outcomes: Sequence[Mapping[str, Any]] | pd.DataFrame,
    completed_months: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    calendar: pd.DatetimeIndex | None = None,
    source_manifest_directory: str | Path | None = None,
    predictor_raw_store_root: str | Path | None = None,
    predictor_derived_store_root: str | Path | None = None,
    scores: pd.DataFrame | str | Path | None = None,
    outcome_manifest_directory: str | Path | None = None,
    outcome_raw_store_root: str | Path | None = None,
    checkpoint_core_store_root: str | Path | None = None,
    prevalidated_predictor_bindings: Mapping[str, Any] | None = None,
    prevalidated_checkpoint_bindings: Mapping[str, Any] | None = None,
    external_identity_registry: dict[tuple[int, int], str] | None = None,
) -> dict[str, Any]:
    """Evaluate only at deterministic terminal month-end; otherwise stay sealed."""

    # The readiness branch is deliberately structural and outcome-aggregate
    # blind.  No performance frame, evidence replay, metric, bootstrap, picks,
    # or gate helper is reached until the exact terminal denominator and its
    # terminal-inclusive completed-month ledger have been established.
    decision_rows = validate_decision_records(decisions)
    scheduled = load_registered_calendar() if calendar is None else calendar
    terminal_candidate = False
    terminal: pd.Timestamp | None = None
    expected = pd.DatetimeIndex([])
    decision_index = pd.DatetimeIndex([])
    if decision_rows:
        first = _date(decision_rows[0]["session_date"], "first decision")
        terminal = deterministic_terminal_session(first, scheduled)
        expected = scheduled[(scheduled >= first) & (scheduled <= terminal)]
        decision_index = pd.DatetimeIndex(
            [_date(item["session_date"], "decision session") for item in decision_rows]
        )
        if len(decision_index) > len(expected) or not decision_index.equals(
            expected[: len(decision_index)]
        ):
            raise V18Error("decision denominator differs from registered calendar")
        terminal_candidate = (
            len(decision_index) == len(expected)
            and decision_index[-1] == terminal
        )
    if not decision_rows:
        return {
            "status": "awaiting_terminal_evaluation",
            "scheduled_sessions": 0,
            "outcome_records": 0,
            "represented_calendar_months": 0,
            "deterministic_terminal_session": None,
            "gate_evaluated": False,
            "production_model_changed": False,
            "orders_allowed": False,
        }
    assert terminal is not None
    sessions = len(decision_rows)
    months = int(decision_index.to_period("M").nunique())
    if not terminal_candidate:
        # The direct API mirrors the CLI's sealed waiting branch.  Do not
        # iterate caller-supplied performance containers until the decision
        # ledger alone proves the exact terminal denominator.
        return {
            "status": "awaiting_terminal_evaluation",
            "scheduled_sessions": sessions,
            "outcome_records": 0,
            "represented_calendar_months": months,
            "minimum_scheduled_sessions": MIN_FORWARD_SESSIONS,
            "minimum_calendar_months": MIN_FORWARD_MONTHS,
            "deterministic_terminal_session": str(terminal.date()),
            "gate_evaluated": False,
            "decision_ledger_sha256": semantic_decision_hash(decision_rows),
            "production_model_changed": False,
            "orders_allowed": False,
        }
    _validate_startup_and_module_closure(phase="terminal reconstruction entry")
    if checkpoint_core_store_root is None:
        raise V18Error("terminal checkpoint core store is required")
    identity_registry = (
        {} if external_identity_registry is None else external_identity_registry
    )
    if outcome_manifest_directory is None or outcome_raw_store_root is None:
        raise V18Error("terminal outcome manifest directory/raw store is required")
    predictor_arguments = (
        source_manifest_directory,
        predictor_raw_store_root,
        predictor_derived_store_root,
        scores,
    )
    if not all(item is not None for item in predictor_arguments):
        raise V18Error("terminal predictor manifest/raw store/scores are required")
    assert source_manifest_directory is not None
    assert predictor_raw_store_root is not None
    assert predictor_derived_store_root is not None
    assert scores is not None
    predictor_bindings = (
        dict(prevalidated_predictor_bindings)
        if prevalidated_predictor_bindings is not None
        else validate_predictor_evidence(
            decision_rows,
            None,
            source_manifest_directory=source_manifest_directory,
            predictor_raw_store_root=predictor_raw_store_root,
            predictor_derived_store_root=predictor_derived_store_root,
            external_identity_registry=identity_registry,
        )
    )
    score_expectations = predictor_bindings.pop(
        "_outcome_blind_score_expectations", None
    )
    if not isinstance(score_expectations, list):
        raise V18Error("outcome-blind predictor replay lacks score expectations")
    terminal_predictor_raw = predictor_bindings.pop(
        "_terminal_predictor_raw_records", None
    )
    terminal_predictor_shards = predictor_bindings.pop(
        "_terminal_predictor_shard_bindings", None
    )
    if not isinstance(terminal_predictor_raw, list) or not isinstance(
        terminal_predictor_shards, list
    ):
        raise V18Error("outcome-blind predictor replay lacks cross-role bindings")
    checkpoint_bindings = (
        dict(prevalidated_checkpoint_bindings)
        if prevalidated_checkpoint_bindings is not None
        else validate_checkpoint_evidence(
            decision_rows,
            checkpoint_core_store_root=checkpoint_core_store_root,
            external_identity_registry=identity_registry,
        )
    )
    _, payload_hash, _, receipt_hash = validate_canonical_activation_artifacts()
    _validate_deferred_state_evidence(
        decision_rows,
        expected_payload_sha256=payload_hash,
        expected_receipt_sha256=receipt_hash,
    )
    if not _preflight_terminal_completed_month_ledger(
        completed_months, decision_rows, calendar=scheduled
    ):
        raise V18Error("terminal completed-month preflight was not reached")
    month_rows = validate_completed_month_records(completed_months)
    outcome_rows = validate_outcome_records(outcomes, decision_rows)
    if len(outcome_rows) != sessions:
        return {
            "status": "awaiting_terminal_evaluation",
            "scheduled_sessions": sessions,
            "outcome_records": len(outcome_rows),
            "represented_calendar_months": months,
            "minimum_scheduled_sessions": MIN_FORWARD_SESSIONS,
            "minimum_calendar_months": MIN_FORWARD_MONTHS,
            "deterministic_terminal_session": str(terminal.date()),
            "gate_evaluated": False,
            "decision_ledger_sha256": semantic_decision_hash(decision_rows),
            "production_model_changed": False,
            "orders_allowed": False,
        }
    validate_completed_month_coverage(month_rows, decision_rows, outcome_rows)
    if isinstance(scores, (str, Path)):
        score_path = Path(scores)
        if score_path.is_symlink() or not score_path.is_file():
            raise V18Error("terminal score ledger is not a plain file")
        score_frame = _read_csv_plain(
            score_path,
            label="terminal score ledger",
            dtype={"code": "string"},
            float_precision="round_trip",
        )
    elif isinstance(scores, pd.DataFrame):
        score_frame = scores
    else:  # pragma: no cover - guarded by the annotated public boundary
        raise V18Error("terminal scores must be a DataFrame or canonical path")
    validate_deferred_score_evidence(
        score_frame, score_expectations, decision_rows
    )
    outcome_manifests = validate_outcome_evidence(
        decision_rows,
        outcome_rows,
        outcome_manifest_directory=outcome_manifest_directory,
        outcome_raw_store_root=outcome_raw_store_root,
        external_identity_registry=identity_registry,
    )
    _validate_terminal_predictor_outcome_cross_role(
        terminal_predictor_raw,
        terminal_predictor_shards,
        outcome_manifests,
        terminal_session=terminal,
        allow_terminal_outcome_only=True,
        predictor_raw_store_root=predictor_raw_store_root,
        predictor_derived_store_root=predictor_derived_store_root,
        outcome_raw_store_root=outcome_raw_store_root,
    )
    _validate_startup_and_module_closure(phase="terminal reconstruction imports")
    if _STRICT_RUNTIME_ACTIVE:
        # All immutable activation identity/workflow evidence must still
        # refetch before the first performance-bearing helper is reached.
        validate_terminal_activation_observations(decision_rows)
    # Checkpoint evidence is the last remote/external reconstruction before
    # unblinding.  Its internally pinned current ref therefore cannot go stale
    # across other terminal network phases before the first performance helper.
    frame, evaluated_decisions, evaluated_outcomes = _evaluation_frame(
        decision_rows, outcome_rows
    )
    if (
        canonical_json_bytes(evaluated_decisions) != canonical_json_bytes(decision_rows)
        or canonical_json_bytes(evaluated_outcomes) != canonical_json_bytes(outcome_rows)
    ):
        raise V18Error("terminal performance frame changed its validated ledgers")
    if len(frame) != sessions:
        raise V18Error("terminal outcome chain is incomplete")
    net40 = frame["candidate_net40"].astype(float)
    midpoint = sessions // 2
    slices = {
        "early": float(net40.iloc[:midpoint].mean()),
        "late": float(net40.iloc[midpoint:].mean()),
    }
    monthly = net40.groupby(net40.index.to_period("M")).mean()
    required_positive_months = math.ceil(0.75 * len(monthly))
    top4_removed = float(net40.drop(net40.nlargest(4).index).mean())
    selected = frame.loc[frame["code"].notna()].copy()
    by_code = (
        selected.assign(code=selected["code"].astype(str))
        .groupby("code", sort=False)["candidate_net40"]
        .sum()
        .reset_index()
        .sort_values(
            ["candidate_net40", "code"],
            ascending=[False, True],
            kind="stable",
        )
    )
    top_codes = by_code.head(5)["code"].tolist()
    code_cash = net40.copy()
    code_cash.loc[frame["code"].astype("string").isin(top_codes)] = 0.0
    counts = selected["code"].astype(str).value_counts()
    selected_count = int(len(selected))
    unique_codes = int(len(counts))
    maximum_share = float(counts.iloc[0] / selected_count) if selected_count else 1.0
    top10_share = (
        float(counts.head(10).sum() / selected_count) if selected_count else 1.0
    )
    executed = int(frame["executed"].sum())
    executed_fraction = float(executed / sessions)
    paired_interval = paired_moving_block_bootstrap(
        net40,
        frame["c00_net40"].astype(float),
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        samples=BOOTSTRAP_SAMPLES,
        confidence=BOOTSTRAP_CONFIDENCE,
        random_state=BOOTSTRAP_RANDOM_STATE,
        nan_policy="raise",
    )
    paired_raw = asdict(paired_interval)
    paired = {
        "observations": paired_raw["observations"],
        "block_length_sessions": paired_raw["block_length"],
        "samples": paired_raw["samples"],
        "random_state": paired_raw["random_state"],
        "confidence": paired_raw["confidence"],
        "candidate_mean_pct": paired_raw["candidate_mean_pct"],
        "control_mean_pct": paired_raw["baseline_mean_pct"],
        "point_estimate_delta_pct": paired_raw["point_estimate_delta_pct"],
        "one_sided_lower_delta_pct": paired_raw["one_sided_lower_delta_pct"],
        "bootstrap_standard_error_delta_pct": paired_raw[
            "bootstrap_standard_error_delta_pct"
        ],
    }
    protocol = read_json(PROTOCOL)
    registered_gates = protocol["evaluation"]["selection_gate_all_required"]
    gate_checks = {
        "net40_mean_positive": float(net40.mean()) > 0.0,
        "net40_median_positive": float(net40.median()) > 0.0,
        "net60_mean_positive": float(frame["candidate_net60"].mean()) > 0.0,
        "both_fixed_slices_net40_positive": all(value > 0 for value in slices.values()),
        "positive_months_net40_at_least_ceil_75pct": int(monthly.gt(0).sum())
        >= required_positive_months,
        "top4_days_removed_net40_positive": top4_removed > 0.0,
        "top5_profit_codes_cash_net40_positive": float(code_cash.mean()) > 0.0,
        "paired_point_delta_vs_C00_net40_positive": paired[
            "point_estimate_delta_pct"
        ]
        > 0.0,
        "paired_one_sided_90_lower_vs_C00_nonnegative": paired[
            "one_sided_lower_delta_pct"
        ]
        >= 0.0,
        "unique_codes_at_least": unique_codes
        >= int(registered_gates["unique_codes_at_least"]),
        "maximum_code_share_at_most": maximum_share
        <= float(registered_gates["maximum_code_share_at_most"]),
        "top10_code_share_at_most": top10_share
        <= float(registered_gates["top10_code_share_at_most"]),
        "executed_days_at_least_ceil_90pct_scheduled": executed
        >= math.ceil(0.90 * sessions),
        "executed_slot_fraction_at_least": executed_fraction
        >= float(registered_gates["executed_slot_fraction_at_least"]),
    }
    if set(gate_checks) != set(registered_gates):
        raise V18Error("implemented gates differ from protocol")
    passed = all(gate_checks.values())
    winner = CANDIDATE_ID if passed else None
    result = {
        "status": (
            "forward_passed_one_v19_research_nominee"
            if passed
            else "forward_rejected_candidate"
        ),
        "scheduled_sessions": sessions,
        "represented_calendar_months": months,
        "deterministic_terminal_session": str(terminal.date()),
        "gate_evaluated": True,
        "cost_metrics": {
            "20": float(frame["candidate_net20"].mean()),
            "40": float(net40.mean()),
            "60": float(frame["candidate_net60"].mean()),
        },
        "control_cost_metrics": {
            C00_ID: {
                "20": float(frame["c00_net20"].mean()),
                "40": float(frame["c00_net40"].mean()),
                "60": float(frame["c00_net60"].mean()),
            },
            C02_ID: {
                "20": float(frame["c02_net20"].mean()),
                "40": float(frame["c02_net40"].mean()),
                "60": float(frame["c02_net60"].mean()),
            },
        },
        "net40_median_pct": float(net40.median()),
        "fixed_slice_net40_mean_pct": slices,
        "monthly_net40_mean_pct": {
            str(month): float(value) for month, value in monthly.items()
        },
        "positive_months_net40": int(monthly.gt(0.0).sum()),
        "required_positive_months_net40": required_positive_months,
        "top4_days_removed_net40_mean_pct": top4_removed,
        "top5_profit_codes": top_codes,
        "top5_profit_codes_cash_net40_mean_pct": float(code_cash.mean()),
        "paired_vs_C00_net40": paired,
        "unique_selected_codes": unique_codes,
        "maximum_code_selection_share": maximum_share,
        "top10_code_selection_share": top10_share,
        "executed_days": executed,
        "executed_slot_fraction": executed_fraction,
        "gate_checks": gate_checks,
        "gate_passed": passed,
        "winner": winner,
        "decision": {
            "research_nominee": winner,
            "production_model_changed": False,
            "production_promotion_allowed": False,
            "orders_allowed": False,
        },
    }
    result["input_bindings"] = {
        **checkpoint_bindings,
        **predictor_bindings,
    }
    _validate_startup_and_module_closure(phase="terminal evaluation exit")
    return result


PICKS_FIELDS = (
    "session_date",
    "candidate_id",
    "source_rank",
    "code",
    "outcome_observed",
    "gross_return_pct",
    "net20_return_pct",
    "net40_return_pct",
    "net60_return_pct",
    "checkpoint_core_sha256",
    "decision_record_sha256",
    "outcome_record_sha256",
)


def materialize_picks(
    decisions: Sequence[Mapping[str, Any]] | pd.DataFrame,
    outcomes: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    decision_rows = validate_decision_records(decisions)
    outcome_rows = validate_outcome_records(outcomes, decision_rows)
    if len(decision_rows) != len(outcome_rows):
        raise V18Error("cannot materialize picks from an incomplete outcome chain")
    records: list[dict[str, Any]] = []
    for decision, outcome in zip(decision_rows, outcome_rows, strict=True):
        specifications = (
            (
                C00_ID,
                1,
                decision["c00_rank1_code"],
                outcome["rank1_outcome_observed"],
                "c00_top1",
            ),
            (
                C02_ID,
                2,
                decision["c02_rank2_code"],
                outcome["rank2_outcome_observed"],
                "c02_rank2",
            ),
            (
                CANDIDATE_ID,
                decision["selected_source_rank"],
                decision["candidate_selected_code"],
                outcome["candidate_outcome_observed"],
                "candidate",
            ),
        )
        for candidate_id, rank, code, observed, prefix in specifications:
            records.append(
                {
                    "session_date": decision["session_date"],
                    "candidate_id": candidate_id,
                    "source_rank": rank,
                    "code": code,
                    "outcome_observed": bool(observed),
                    "gross_return_pct": outcome[f"{prefix}_gross_return_pct"],
                    "net20_return_pct": outcome[f"{prefix}_net20_return_pct"],
                    "net40_return_pct": outcome[f"{prefix}_net40_return_pct"],
                    "net60_return_pct": outcome[f"{prefix}_net60_return_pct"],
                    "checkpoint_core_sha256": decision["checkpoint_core_sha256"],
                    "decision_record_sha256": decision["record_sha256"],
                    "outcome_record_sha256": outcome["record_sha256"],
                }
            )
    return pd.DataFrame(records, columns=PICKS_FIELDS)


def build_result(
    evaluation: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    completed_months: Sequence[Mapping[str, Any]],
    *,
    decision_ledger_path: str | Path,
    outcome_ledger_path: str | Path,
    completed_month_ledger_path: str | Path,
    score_output: str | Path = SCORE_OUTPUT,
    picks_output: str | Path = PICKS_OUTPUT,
    runtime_lock_verified_at: Any | None = None,
) -> dict[str, Any]:
    """Build the protocol result envelope after terminal evidence validation."""

    if evaluation.get("status") == "awaiting_terminal_evaluation":
        raise V18Error("canonical result remains sealed until terminal")
    decision_rows = validate_decision_records(decisions)
    outcome_rows = validate_outcome_records(outcomes, decision_rows)
    month_rows = validate_completed_month_coverage(
        completed_months, decision_rows, outcome_rows
    )
    if not decision_rows or len(decision_rows) != len(outcome_rows):
        raise V18Error("terminal result inputs are incomplete")
    scores_path = Path(score_output)
    picks_path = Path(picks_output)
    if not scores_path.is_file() or not picks_path.is_file():
        raise V18Error("terminal score/picks artifacts are missing")
    if scores_path.is_symlink() or picks_path.is_symlink():
        raise V18Error("terminal score/picks artifacts must be plain files")
    score_bytes = _plain_file_bytes(scores_path, label="terminal score artifact")
    scores = validate_score_ledger(
        pd.read_csv(
            io.BytesIO(score_bytes),
            dtype={"code": "string"},
            float_precision="round_trip",
        ),
        allow_empty=True,
    )
    if score_bytes != scores.to_csv(index=False, lineterminator="\n").encode():
        raise V18Error("terminal score artifact bytes are not canonical")
    expected_picks = materialize_picks(decision_rows, outcome_rows)
    expected_pick_bytes = expected_picks.to_csv(
        index=False, lineterminator="\n"
    ).encode()
    pick_bytes = _plain_file_bytes(picks_path, label="terminal picks artifact")
    if pick_bytes != expected_pick_bytes:
        raise V18Error("terminal picks differ from recomputed decision/outcome picks")

    ledger_bindings = (
        (Path(decision_ledger_path), decision_rows, "decision"),
        (Path(outcome_ledger_path), outcome_rows, "outcome"),
        (Path(completed_month_ledger_path), month_rows, "completed-month"),
    )
    ledger_bytes: dict[str, bytes] = {}
    for path, rows, label in ledger_bindings:
        if path.is_symlink() or not path.is_file():
            raise V18Error(f"terminal {label} ledger must be a plain file")
        expected_bytes = b"".join(
            canonical_json_bytes(item) + b"\n" for item in rows
        )
        if _plain_file_bytes(path, label=f"terminal {label} ledger") != expected_bytes:
            raise V18Error(
                f"terminal {label} ledger bytes differ from validated records"
            )
        ledger_bytes[label] = expected_bytes
    first, last = decision_rows[0], decision_rows[-1]
    artifacts = {
        "decision_ledger_sha256": hashlib.sha256(ledger_bytes["decision"]).hexdigest(),
        "outcome_ledger_sha256": hashlib.sha256(ledger_bytes["outcome"]).hexdigest(),
        "completed_month_ledger_sha256": hashlib.sha256(
            ledger_bytes["completed-month"]
        ).hexdigest(),
        "score_output_sha256": hashlib.sha256(score_bytes).hexdigest(),
        "score_semantic_sha256": semantic_score_ledger_hash(
            scores, allow_empty=True
        ),
        "picks_output_sha256": hashlib.sha256(pick_bytes).hexdigest(),
    }
    input_bindings = dict(evaluation.get("input_bindings", {}))
    required_inputs = set(
        read_json(PROTOCOL)["result_contract"]["required_input_fields"]
    )
    if set(input_bindings) != required_inputs:
        raise V18Error("terminal result lacks exact clean-room predictor input bindings")
    result_verified_at = (
        _runtime_verified_timestamp()
        if runtime_lock_verified_at is None
        else _timestamp(
            runtime_lock_verified_at, "retained result runtime_lock_verified_at"
        )
    )
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": sha256_file(__file__),
        "activation_payload_sha256": first["activation_payload_sha256"],
        "activation_receipt_sha256": first["activation_receipt_sha256"],
        "activation_receipt_commit_sha": first["activation_receipt_commit_sha"],
        "status": evaluation["status"],
        "failure_reason": None,
        "integrity_stage": None,
        "authority": {
            "analysis_type": "genuinely_later_forward_shadow_development",
            "production_model_changed": False,
            "production_promotion_allowed": False,
            "orders_allowed": False,
        },
        "raw_source_provenance": _raw_source_provenance_envelope(),
        "input": input_bindings,
        "forward_period": {
            "first_counted_session": first["session_date"],
            "terminal_session": last["session_date"],
            "scheduled_sessions": len(decision_rows),
            "represented_calendar_months": len(
                {str(pd.Timestamp(item["session_date"]).to_period("M")) for item in decision_rows}
            ),
        },
        "state_months": month_rows,
        "models": {
            CANDIDATE_ID: dict(evaluation),
            C00_ID: evaluation.get("control_cost_metrics", {}).get(C00_ID),
            C02_ID: evaluation.get("control_cost_metrics", {}).get(C02_ID),
        },
        "candidate_gate": {
            "candidate_id": CANDIDATE_ID,
            "checks": evaluation.get("gate_checks"),
            "passed": evaluation.get("gate_passed"),
        },
        "decision": evaluation["decision"],
        "artifact_sha256": artifacts,
        "runtime": {
            "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
            "runtime_lock_self_sha256": read_json(RUNTIME_LOCK)[
                "runtime_lock_self_sha256"
            ],
            "runtime_lock_verified_at": result_verified_at,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
    }


def _stream_plain_file_sha256(path: str | Path, *, label: str) -> str | None:
    """Hash an optional pinned artifact without parsing or retaining its bytes."""

    target = Path(path)
    try:
        before_path = os.lstat(target)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(before_path.st_mode):
        raise V18Error(f"{label} exists but is not a regular non-symlink file")
    if before_path.st_nlink != 1:
        raise V18Error(f"{label} has a forbidden hard-link alias")
    try:
        descriptor = os.open(
            target,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise V18Error(f"{label} cannot be pinned for integrity hashing") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino)
            != (before_path.st_dev, before_path.st_ino)
            or before.st_nlink != 1
        ):
            raise V18Error(f"{label} changed before integrity hashing")
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        if (
            byte_count != before.st_size
            or any(
                getattr(before, field) != getattr(after, field)
                for field in (
                    "st_dev",
                    "st_ino",
                    "st_nlink",
                    "st_size",
                    "st_mtime_ns",
                    "st_ctime_ns",
                )
            )
            or after.st_nlink != 1
        ):
            raise V18Error(f"{label} changed during integrity hashing")
        return digest.hexdigest()
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def build_integrity_abort_result(
    *,
    failure_reason: str,
    integrity_stage: str,
    activation_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an outcome-blind create-once integrity-abort envelope."""

    if not _STRICT_RUNTIME_ACTIVE or _STRICT_RUNTIME_VERIFIED_AT is None:
        raise V18Error("integrity abort requires strict runtime validation")
    reason = str(failure_reason)
    stage = str(integrity_stage)
    if (
        INTEGRITY_FAILURE_REASON_RE.fullmatch(reason) is None
        or reason not in ABORT_FAILURE_REASONS
    ):
        raise V18Error(
            "integrity abort failure reason is not a registered machine token"
        )
    if stage not in ABORT_INTEGRITY_STAGES:
        raise V18Error("integrity abort stage is not registered")
    result_presence = _local_authority_presence_state(
        RESULT_OUTPUT, label="canonical result"
    )
    protocol, observed_protocol_hash = validate_protocol()
    allowed_reasons = protocol["result_contract"]["abort_stage_reason_values"][stage]
    if reason not in allowed_reasons:
        raise V18Error("integrity abort reason is not allowed for its stage")
    runtime_lock, observed_runtime_hash = validate_runtime_lock(
        strict_environment=False
    )
    if observed_runtime_hash != RUNTIME_LOCK_SHA256:
        raise V18Error("integrity abort runtime-lock authority changed")

    context = validate_activation_context(activation_context)
    _, payload_hash = validate_activation_payload(ACTIVATION_PAYLOAD, protocol)
    _, receipt_hash = validate_activation_receipt(
        ACTIVATION_RECEIPT,
        ACTIVATION_PAYLOAD,
        protocol,
    )
    if (
        context["activation_payload_sha256"] != payload_hash
        or context["activation_receipt_sha256"] != receipt_hash
    ):
        raise V18Error("integrity abort activation context changed")

    required_inputs = tuple(protocol["result_contract"]["required_input_fields"])
    input_bindings: dict[str, Any] = dict.fromkeys(required_inputs)
    input_bindings.update(
        {
            "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
            "predictor_parser_sha256": JPX_PARSER_SHA256,
            "v17_c00_protocol_sha256": V17_BINDINGS["protocol"][1],
            "v17_c00_runner_sha256": V17_BINDINGS["runner"][1],
        }
    )
    artifact_hashes = {
        "decision_ledger_sha256": _stream_plain_file_sha256(
            DECISION_LEDGER, label="canonical decision ledger"
        ),
        "outcome_ledger_sha256": _stream_plain_file_sha256(
            OUTCOME_LEDGER, label="canonical outcome ledger"
        ),
        "completed_month_ledger_sha256": _stream_plain_file_sha256(
            COMPLETED_MONTH_LEDGER, label="canonical completed-month ledger"
        ),
        "score_output_sha256": _stream_plain_file_sha256(
            SCORE_OUTPUT, label="canonical score output"
        ),
        "score_semantic_sha256": None,
        "picks_output_sha256": _stream_plain_file_sha256(
            PICKS_OUTPUT, label="canonical picks output"
        ),
    }
    if set(artifact_hashes) != set(
        protocol["result_contract"]["required_artifact_hashes"]
    ):
        raise V18Error("integrity abort artifact schema differs from protocol")
    retained_result: dict[str, Any] | None = None
    result_verified_at = _STRICT_RUNTIME_VERIFIED_AT
    if result_presence != "absent":
        retained_payload = _read_local_authority_bytes(
            RESULT_OUTPUT, label="retained abort result"
        )
        status_tokens = re.findall(
            rb'^  "status": "([a-z0-9_]+)",?$', retained_payload, re.MULTILINE
        )
        if status_tokens != [b"aborted_integrity_failure"]:
            raise V18Error(
                "canonical result is not an abort envelope; abort retry is forbidden"
            )
        try:
            parsed_retained = json.loads(retained_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise V18Error("retained abort result is not strict JSON") from exc
        if not isinstance(parsed_retained, dict):
            raise V18Error("retained abort result is not a JSON object")
        retained_result = parsed_retained
        try:
            retained_verified_at = retained_result["runtime"][
                "runtime_lock_verified_at"
            ]
            _timestamp(
                retained_verified_at,
                "retained abort runtime_lock_verified_at",
            )
        except (KeyError, TypeError) as exc:
            raise V18Error("retained abort runtime timestamp is missing") from exc
        # Validate chronology without normalizing the retained JSON scalar.
        # Exact retry must reproduce the published bytes, including whether a
        # valid timestamp used an explicit fractional-second component.
        result_verified_at = retained_verified_at
    result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": observed_protocol_hash,
        "runner_sha256": sha256_file(__file__),
        "activation_payload_sha256": payload_hash,
        "activation_receipt_sha256": receipt_hash,
        "activation_receipt_commit_sha": context[
            "activation_receipt_commit_sha"
        ],
        "status": "aborted_integrity_failure",
        "failure_reason": reason,
        "integrity_stage": stage,
        "authority": dict(protocol["result_contract"]["authority_values"]),
        "raw_source_provenance": _raw_source_provenance_envelope(),
        "input": input_bindings,
        "forward_period": None,
        "state_months": None,
        "models": None,
        "candidate_gate": None,
        "decision": {
            "research_nominee": None,
            "failure_reason": reason,
            "integrity_stage": stage,
        },
        "artifact_sha256": artifact_hashes,
        "runtime": {
            "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
            "runtime_lock_self_sha256": runtime_lock[
                "runtime_lock_self_sha256"
            ],
            "runtime_lock_verified_at": result_verified_at,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
    }
    if retained_result is not None:
        if canonical_json_bytes(retained_result) != canonical_json_bytes(result):
            raise V18Error("retained canonical result differs from exact abort retry")
        return retained_result
    return result


def _read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    payload = _plain_file_bytes(source, label="parsed panel/input")
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(
            io.BytesIO(payload),
            dtype={"code": "string"},
            float_precision="round_trip",
        )
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(io.BytesIO(payload))
    raise V18Error("parsed panel/input must be CSV or Parquet")


def _write_csv_exclusive(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    _atomic_text(
        target,
        frame.to_csv(index=False, lineterminator="\n"),
        exclusive=True,
    )


def _write_jsonl_exclusive(
    records: Sequence[Mapping[str, Any]], path: str | Path
) -> None:
    payload = b"".join(canonical_json_bytes(item) + b"\n" for item in records)
    _atomic_text(path, payload.decode("utf-8"), exclusive=True)


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", default=str(PROTOCOL))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate frozen protocol/history")
    _add_common_paths(validate)

    subparsers.add_parser(
        "validate-runtime",
        help="strictly validate the preregistered operational runtime and binaries",
    )

    cache_anchor = subparsers.add_parser(
        "prepare-predictor-cache",
        help="seal the exact pre-B historical raw/shard/snapshot cache anchor",
    )
    cache_anchor.add_argument("--source-pdf", action="append", required=True)
    cache_anchor.add_argument("--source-file-name", action="append", required=True)
    cache_anchor.add_argument("--source-url", action="append", required=True)
    cache_anchor.add_argument("--through", required=True)
    cache_anchor.add_argument("--predictor-raw-store-root", required=True)
    cache_anchor.add_argument("--predictor-derived-store-root", required=True)

    store_readiness = subparsers.add_parser(
        "prepare-operational-stores",
        help="precreate and validate private disjoint checkpoint/store parents",
    )
    store_readiness.add_argument("--predictor-raw-store-root", required=True)
    store_readiness.add_argument("--predictor-derived-store-root", required=True)
    store_readiness.add_argument("--outcome-raw-store-root", required=True)
    store_readiness.add_argument("--checkpoint-core-store-root", required=True)

    prepare_day_parser = subparsers.add_parser(
        "prepare-day",
        help=(
            "canonical one-process source/month/fold/state/score/checkpoint preparation"
        ),
    )
    prepare_day_parser.add_argument("--session", required=True)
    prepare_day_parser.add_argument("--new-predictor-pdf", action="append", default=[])
    prepare_day_parser.add_argument(
        "--new-predictor-file-name", action="append", default=[]
    )
    prepare_day_parser.add_argument("--new-predictor-url", action="append", default=[])
    prepare_day_parser.add_argument(
        "--new-predictor-received-at", action="append", default=[]
    )
    prepare_day_parser.add_argument("--predictor-raw-store-root", required=True)
    prepare_day_parser.add_argument("--predictor-derived-store-root", required=True)
    prepare_day_parser.add_argument("--outcome-raw-store-root", required=True)
    prepare_day_parser.add_argument("--checkpoint-core-store-root", required=True)
    prepare_day_parser.add_argument(
        "--activation-context", default=str(ACTIVATION_CONTEXT)
    )

    payload = subparsers.add_parser(
        "create-activation-payload", help="create commit-B activation payload"
    )
    payload.add_argument("--preregistration-commit-sha", required=True)
    payload.add_argument("--predictor-raw-store-root", required=True)
    payload.add_argument("--predictor-derived-store-root", required=True)
    payload.add_argument(
        "--predictor-cache-anchor-manifest-object-key", required=True
    )
    payload.add_argument("--output", default=str(ACTIVATION_PAYLOAD))

    receipt = subparsers.add_parser(
        "create-activation-receipt", help="create commit-C GitHub observation receipt"
    )
    receipt.add_argument("--payload", default=str(ACTIVATION_PAYLOAD))
    receipt.add_argument("--payload-commit-sha", required=True)
    receipt.add_argument("--output", default=str(ACTIVATION_RECEIPT))

    preflight = subparsers.add_parser(
        "preflight", help="validate commit-C and derive first/terminal sessions"
    )
    preflight.add_argument("--payload", default=str(ACTIVATION_PAYLOAD))
    preflight.add_argument("--receipt", default=str(ACTIVATION_RECEIPT))
    preflight.add_argument("--receipt-commit-sha", required=True)
    preflight.add_argument("--output", default=str(ACTIVATION_CONTEXT))

    publish = subparsers.add_parser(
        "publish-checkpoint",
        help="publish the prepared pair once through the fixed Git Data API chain",
    )
    publish.add_argument("--session", required=True)

    decide = subparsers.add_parser(
        "decide", help="resolve daily checkpoint evidence and append one decision"
    )
    decide.add_argument("--session", required=True)
    decide.add_argument("--checkpoint-core-store-root", required=True)

    terminal_finalize = subparsers.add_parser(
        "finalize-terminal",
        help="seal the final outcome and close the terminal month after blind gates",
    )
    terminal_finalize.add_argument("--source-pdf", required=True)
    terminal_finalize.add_argument("--source-file-name", required=True)
    terminal_finalize.add_argument("--source-url", required=True)
    terminal_finalize.add_argument("--source-received-at", required=True)
    terminal_finalize.add_argument("--predictor-raw-store-root", required=True)
    terminal_finalize.add_argument("--predictor-derived-store-root", required=True)
    terminal_finalize.add_argument("--outcome-raw-store-root", required=True)
    terminal_finalize.add_argument("--checkpoint-core-store-root", required=True)
    terminal_finalize.add_argument(
        "--activation-context", default=str(ACTIVATION_CONTEXT)
    )

    evaluation = subparsers.add_parser(
        "evaluate",
        help=(
            "return sealed awaiting status or exclusively create the fixed terminal "
            "result"
        ),
    )
    evaluation.add_argument("--decisions", default=str(DECISION_LEDGER))
    evaluation.add_argument("--outcomes", default=str(OUTCOME_LEDGER))
    evaluation.add_argument("--source-manifest-directory", default=str(SOURCE_MANIFEST_DIR))
    evaluation.add_argument("--predictor-raw-store-root", required=True)
    evaluation.add_argument("--predictor-derived-store-root", required=True)
    evaluation.add_argument("--outcome-manifest-directory", default=str(OUTCOME_MANIFEST_DIR))
    evaluation.add_argument("--outcome-raw-store-root", required=True)
    evaluation.add_argument("--checkpoint-core-store-root", required=True)
    evaluation.add_argument("--completed-months", default=str(COMPLETED_MONTH_LEDGER))
    evaluation.add_argument("--scores", default=str(SCORE_OUTPUT))
    evaluation.add_argument("--picks", default=str(PICKS_OUTPUT))

    abort = subparsers.add_parser(
        "abort",
        help="record an outcome-blind create-once integrity failure after activation",
    )
    abort.add_argument(
        "--failure-reason", required=True, choices=ABORT_FAILURE_REASONS
    )
    abort.add_argument(
        "--integrity-stage", required=True, choices=ABORT_INTEGRITY_STAGES
    )
    abort.add_argument("--activation-context", default=str(ACTIVATION_CONTEXT))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    postactivation_result_guarded = {
        "prepare-day",
        "publish-checkpoint",
        "decide",
        "finalize-terminal",
        "evaluate",
        "abort",
    }
    result_presence = (
        _local_authority_presence_state(RESULT_OUTPUT, label="canonical result")
        if args.command in postactivation_result_guarded
        else "absent"
    )
    if args.command in {
        "prepare-day",
        "publish-checkpoint",
        "decide",
        "finalize-terminal",
    } and result_presence != "absent":
        raise V18Error(
            "experiment already has a terminal result; post-result work is forbidden"
        )
    if args.command == "evaluate" and result_presence != "absent":
        retained_status = _result_status_token_without_performance_read(
            RESULT_OUTPUT, presence=result_presence
        )
        if retained_status == "aborted_integrity_failure":
            raise V18Error(
                "experiment was irreversibly terminated by an integrity abort"
            )
    runtime_value: dict[str, Any] | None = None
    runtime_digest: str | None = None
    if args.command != "validate":
        runtime_value, runtime_digest = validate_runtime_lock(strict_environment=True)
    if args.command not in {"validate", "validate-runtime"}:
        validate_protocol()
    if args.command in {
        "prepare-day",
        "publish-checkpoint",
        "decide",
        "finalize-terminal",
        "evaluate",
    }:
        # Direct v1.8 artifacts are intentionally outside the transitive lock
        # to avoid a hash cycle.  Commit A's payload is therefore the runtime
        # authority for the exact runner/audit/tests/protocol bytes, and every
        # post-activation operation must re-establish it before doing work.
        validate_canonical_activation_artifacts()
    if args.command == "validate":
        protocol, digest = validate_protocol(args.protocol)
        result = {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": digest,
            "calendar_sha256": sha256_file(CALENDAR),
            "historical_bindings": validate_historical_bindings(),
            "production_model_changed": False,
            "orders_allowed": False,
        }
    elif args.command == "validate-runtime":
        assert runtime_value is not None and runtime_digest is not None
        result = {
            "runtime_lock_id": runtime_value["lock_id"],
            "runtime_lock_sha256": runtime_digest,
            "runtime_lock_self_sha256": runtime_value["runtime_lock_self_sha256"],
            "project_file_set_sha256": runtime_value["project_file_set_sha256"],
            "strict_environment_validated": True,
            "production_model_changed": False,
            "orders_allowed": False,
        }
    elif args.command == "abort":
        if Path(args.activation_context).resolve() != ACTIVATION_CONTEXT.resolve():
            raise V18Error("integrity abort requires canonical activation context")
        result = build_integrity_abort_result(
            failure_reason=args.failure_reason,
            integrity_stage=args.integrity_stage,
            activation_context=read_json(args.activation_context),
        )
        write_json(result, RESULT_OUTPUT, exclusive=True)
    elif args.command == "prepare-predictor-cache":
        anchor, summary = build_predictor_cache_anchor(
            args.source_pdf,
            args.source_file_name,
            args.source_url,
            through_session=args.through,
            predictor_raw_store_root=args.predictor_raw_store_root,
            predictor_derived_store_root=args.predictor_derived_store_root,
        )
        result = {
            "cache_contract_id": anchor["cache_contract_id"],
            **summary,
            "production_model_changed": False,
            "orders_allowed": False,
        }
    elif args.command == "prepare-operational-stores":
        result = prepare_operational_stores(
            predictor_raw_store_root=args.predictor_raw_store_root,
            predictor_derived_store_root=args.predictor_derived_store_root,
            outcome_raw_store_root=args.outcome_raw_store_root,
            checkpoint_core_store_root=args.checkpoint_core_store_root,
        )
    elif args.command == "prepare-day":
        if Path(args.activation_context).resolve() != ACTIVATION_CONTEXT.resolve():
            raise V18Error("prepare-day requires canonical activation context")
        result = prepare_day(
            session_date=args.session,
            new_predictor_pdfs=args.new_predictor_pdf,
            new_predictor_file_names=args.new_predictor_file_name,
            new_predictor_urls=args.new_predictor_url,
            new_predictor_received_at=args.new_predictor_received_at,
            predictor_raw_store_root=args.predictor_raw_store_root,
            predictor_derived_store_root=args.predictor_derived_store_root,
            outcome_raw_store_root=args.outcome_raw_store_root,
            checkpoint_core_store_root=args.checkpoint_core_store_root,
            activation_context=read_json(args.activation_context),
        )
    elif args.command == "create-activation-payload":
        if Path(args.output).resolve() != ACTIVATION_PAYLOAD.resolve():
            raise V18Error("canonical activation payload must use its registered path")
        result = create_activation_payload(
            preregistration_commit_sha=args.preregistration_commit_sha,
            predictor_raw_store_root=args.predictor_raw_store_root,
            predictor_derived_store_root=args.predictor_derived_store_root,
            predictor_cache_anchor_manifest_object_key=(
                args.predictor_cache_anchor_manifest_object_key
            ),
            output=args.output,
        )
    elif args.command == "create-activation-receipt":
        if (
            Path(args.payload).resolve() != ACTIVATION_PAYLOAD.resolve()
            or Path(args.output).resolve() != ACTIVATION_RECEIPT.resolve()
        ):
            raise V18Error("canonical activation receipt paths changed")
        result = create_activation_receipt(
            args.payload,
            payload_commit_sha=args.payload_commit_sha,
            output=args.output,
        )
    elif args.command == "preflight":
        if (
            Path(args.payload).resolve() != ACTIVATION_PAYLOAD.resolve()
            or Path(args.receipt).resolve() != ACTIVATION_RECEIPT.resolve()
            or Path(args.output).resolve() != ACTIVATION_CONTEXT.resolve()
        ):
            raise V18Error("activation preflight requires canonical artifacts")
        context_presence = _local_authority_presence_state(
            ACTIVATION_CONTEXT, label="activation context"
        )
        if context_presence != "absent":
            result = validate_activation_context(read_json(ACTIVATION_CONTEXT))
            if result["activation_receipt_commit_sha"] != args.receipt_commit_sha:
                raise V18Error("activation context retry changes receipt commit C")
        else:
            result = activate(
                args.payload,
                args.receipt,
                activation_receipt_commit_sha=args.receipt_commit_sha,
            )
            write_json(result, args.output, exclusive=True)
    elif args.command == "publish-checkpoint":
        result = publish_checkpoint(session_date=args.session)
    elif args.command == "decide":
        existing = _load_decision_record_authority(heal_derived=True)
        target_session = str(_date(args.session, "decide session").date())
        if existing and existing[-1]["session_date"] == target_session:
            # A crash after the immutable record shard but before the derived
            # JSONL view/health response is an exact validation-only retry.
            combined = existing
        else:
            activation_context = _discover_activation_context(existing)
            core, checkpoint_binding = resolve_checkpoint(
                session_date=args.session,
                existing_records=existing,
                activation=activation_context,
                checkpoint_core_store_root=args.checkpoint_core_store_root,
            )
            combined = materialize_checkpoint_decision(
                core,
                checkpoint_binding,
                session_date=args.session,
                activation=activation_context,
                existing_records=existing,
            )
            append_jsonl_record(
                DECISION_LEDGER,
                combined[-1],
                required_fields=DECISION_FIELDS,
                validator=validate_decision_records,
            )
            combined = _load_decision_record_authority(heal_derived=True)
        result = {
            "sequence_number": combined[-1]["sequence_number"],
            "session_date": combined[-1]["session_date"],
            "record_sha256": combined[-1]["record_sha256"],
            "decision_ledger_semantic_sha256": semantic_decision_hash(combined),
        }
    elif args.command == "finalize-terminal":
        if Path(args.activation_context).resolve() != ACTIVATION_CONTEXT.resolve():
            raise V18Error("finalize-terminal requires canonical activation context")
        result = finalize_terminal(
            source_pdf=args.source_pdf,
            source_file_name=args.source_file_name,
            source_url=args.source_url,
            source_received_at=args.source_received_at,
            predictor_raw_store_root=args.predictor_raw_store_root,
            predictor_derived_store_root=args.predictor_derived_store_root,
            outcome_raw_store_root=args.outcome_raw_store_root,
            checkpoint_core_store_root=args.checkpoint_core_store_root,
            activation_context=read_json(args.activation_context),
        )
    elif args.command == "evaluate":
        canonical_inputs = {
            "decision ledger": (args.decisions, DECISION_LEDGER),
            "outcome ledger": (args.outcomes, OUTCOME_LEDGER),
            "completed-month ledger": (
                args.completed_months,
                COMPLETED_MONTH_LEDGER,
            ),
            "score ledger": (args.scores, SCORE_OUTPUT),
            "picks output": (args.picks, PICKS_OUTPUT),
            "source manifest directory": (
                args.source_manifest_directory,
                SOURCE_MANIFEST_DIR,
            ),
            "outcome manifest directory": (
                args.outcome_manifest_directory,
                OUTCOME_MANIFEST_DIR,
            ),
        }
        for label, (supplied, expected) in canonical_inputs.items():
            supplied_path = Path(os.path.abspath(os.fspath(supplied)))
            expected_path = Path(os.path.abspath(os.fspath(expected)))
            if supplied_path != expected_path:
                raise V18Error(f"canonical result {label} path changed")
        decisions = load_jsonl_record_authority(
            args.decisions,
            required_fields=DECISION_FIELDS,
            validator=validate_decision_records,
            heal_derived=False,
        )
        scheduled = load_registered_calendar()
        terminal_candidate = False
        terminal: pd.Timestamp | None = None
        decision_index = pd.DatetimeIndex([])
        if decisions:
            first = _date(decisions[0]["session_date"], "first decision")
            terminal = deterministic_terminal_session(first, scheduled)
            denominator = scheduled[(scheduled >= first) & (scheduled <= terminal)]
            decision_index = pd.DatetimeIndex(
                [_date(item["session_date"], "decision session") for item in decisions]
            )
            if len(decision_index) > len(denominator) or not decision_index.equals(
                denominator[: len(decision_index)]
            ):
                raise V18Error("decision denominator differs from registered calendar")
            terminal_candidate = (
                len(decision_index) == len(denominator)
                and len(decision_index) > 0
                and decision_index[-1] == terminal
            )
        if not terminal_candidate:
            # Decision/calendar structure is the only information opened on
            # a nonterminal invocation.  In particular, do not enumerate or
            # read predictor caches, state, completed-month, outcome, score,
            # picks, checkpoint, or result evidence merely to report waiting.
            if result_presence != "absent":
                raise V18Error(
                    "canonical result exists before the terminal denominator"
                )
            result = {
                "status": "awaiting_terminal_evaluation",
                "scheduled_sessions": len(decisions),
                "outcome_records": 0,
                "represented_calendar_months": int(
                    decision_index.to_period("M").nunique()
                ),
                "minimum_scheduled_sessions": MIN_FORWARD_SESSIONS,
                "minimum_calendar_months": MIN_FORWARD_MONTHS,
                "deterministic_terminal_session": (
                    None if terminal is None else str(terminal.date())
                ),
                "gate_evaluated": False,
                "performance_evidence_opened": False,
                "production_model_changed": False,
                "orders_allowed": False,
            }
            print(
                json.dumps(
                    _safe(result), ensure_ascii=False, sort_keys=True, allow_nan=False
                )
            )
            return 0
        terminal_identity_registry: dict[tuple[int, int], str] = {}
        predictor_bindings = validate_predictor_evidence(
            decisions,
            None,
            source_manifest_directory=args.source_manifest_directory,
            predictor_raw_store_root=args.predictor_raw_store_root,
            predictor_derived_store_root=args.predictor_derived_store_root,
            external_identity_registry=terminal_identity_registry,
        )
        checkpoint_bindings = validate_checkpoint_evidence(
            decisions,
            checkpoint_core_store_root=args.checkpoint_core_store_root,
            external_identity_registry=terminal_identity_registry,
        )
        # Outcome, score, picks, and result paths remain unopened until all
        # outcome-blind raw/shard/snapshot/cache/checkpoint reconstruction
        # above has succeeded.
        payload_value, payload_hash, _, receipt_hash = (
            validate_canonical_activation_artifacts()
        )
        del payload_value
        _validate_deferred_state_evidence(
            decisions,
            expected_payload_sha256=payload_hash,
            expected_receipt_sha256=receipt_hash,
        )
        completed_months = load_jsonl_record_authority(
            args.completed_months,
            required_fields=COMPLETED_MONTH_FIELDS,
            validator=validate_completed_month_records,
            heal_derived=False,
        )
        _preflight_terminal_completed_month_ledger(
            completed_months,
            decisions,
        )
        outcomes = load_jsonl_record_authority(
            args.outcomes,
            required_fields=OUTCOME_FIELDS,
            validator=lambda rows: validate_outcome_records(rows, decisions),
            heal_derived=False,
        )
        result = evaluate(
            decisions,
            outcomes,
            completed_months,
            source_manifest_directory=args.source_manifest_directory,
            predictor_raw_store_root=args.predictor_raw_store_root,
            predictor_derived_store_root=args.predictor_derived_store_root,
            scores=Path(args.scores),
            outcome_manifest_directory=args.outcome_manifest_directory,
            outcome_raw_store_root=args.outcome_raw_store_root,
            checkpoint_core_store_root=args.checkpoint_core_store_root,
            prevalidated_predictor_bindings=predictor_bindings,
            prevalidated_checkpoint_bindings=checkpoint_bindings,
            external_identity_registry=terminal_identity_registry,
        )
        if result["status"] != "awaiting_terminal_evaluation":
            expected_picks = materialize_picks(decisions, outcomes)
            expected_pick_bytes = expected_picks.to_csv(
                index=False, lineterminator="\n"
            ).encode()
            picks_path = Path(args.picks)
            # The exact writer is also the only safe healer for a crash after
            # final link and before staging-name cleanup.
            _write_csv_exclusive(expected_picks, picks_path)
            if (
                _plain_file_bytes(picks_path, label="canonical picks output")
                != expected_pick_bytes
            ):
                raise V18Error("canonical picks differ from recomputed picks")
            retained_result: dict[str, Any] | None = None
            retained_runtime_verified_at: Any | None = None
            if result_presence != "absent":
                # This is deliberately after blind predictor/checkpoint gates
                # and the authorized performance reconstruction above.
                retained_result = read_json(RESULT_OUTPUT)
                try:
                    retained_runtime_verified_at = retained_result["runtime"][
                        "runtime_lock_verified_at"
                    ]
                except (KeyError, TypeError) as exc:
                    raise V18Error(
                        "retained terminal result runtime timestamp is missing"
                    ) from exc
            rebuilt_result = build_result(
                result,
                decisions,
                outcomes,
                completed_months,
                decision_ledger_path=args.decisions,
                outcome_ledger_path=args.outcomes,
                completed_month_ledger_path=args.completed_months,
                score_output=args.scores,
                picks_output=args.picks,
                runtime_lock_verified_at=retained_runtime_verified_at,
            )
            if retained_result is not None:
                if canonical_json_bytes(retained_result) != canonical_json_bytes(
                    rebuilt_result
                ):
                    raise V18Error(
                        "retained canonical result differs from exact terminal retry"
                    )
                result = retained_result
            else:
                result = rebuilt_result
                write_json(result, RESULT_OUTPUT, exclusive=True)
    else:  # pragma: no cover - argparse enforces this
        raise V18Error(f"unknown command: {args.command}")
    print(json.dumps(_safe(result), ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except V18Error as exc:
        print(f"v1.8 fail-closed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
