#!/usr/bin/env python3
"""Independent audit for the prospective v1.8 shoulder-state experiment.

This module intentionally does not import the v1.8 runner or any project
profit/bootstrap helper.  It reconstructs the monthly state, validates the
outcome-free hash-chained decision ledger, joins the separately keyed outcome
ledger, and independently recomputes the stopping rule, metrics, gates,
winner, and research-only authority decision.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import io
import math
import os
from pathlib import Path
import platform
import re
import ssl
import stat
import subprocess
import sys
import sysconfig
import tempfile
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import HTTPSHandler, HTTPRedirectHandler, Request, build_opener

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for _import_root in (ROOT / "src", ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))
DEFAULT_PROTOCOL = ROOT / "research/model_v18_shoulder_state_protocol.json"
DEFAULT_ACTIVATION_PAYLOAD = (
    ROOT / "research/model_v18_shoulder_state_activation_payload.json"
)
DEFAULT_ACTIVATION_RECEIPT = (
    ROOT / "research/model_v18_shoulder_state_activation_receipt.json"
)
DEFAULT_DECISIONS = ROOT / "research/model_v18_shoulder_state_decisions.jsonl"
DEFAULT_OUTCOMES = ROOT / "research/model_v18_shoulder_state_outcomes.jsonl"
DEFAULT_MONTHS = ROOT / "research/model_v18_shoulder_state_months.jsonl"
DEFAULT_STATE_MANIFESTS = ROOT / "research/model_v18_shoulder_state_state_manifests"
DEFAULT_FOLD_MANIFESTS = ROOT / "research/model_v18_shoulder_state_fold_manifests"
DEFAULT_FOLD_MODELS = ROOT / "research/model_v18_shoulder_state_fold_models"
DEFAULT_SOURCE_MANIFESTS = ROOT / "research/model_v18_shoulder_state_source_manifests"
DEFAULT_OUTCOME_MANIFESTS = (
    ROOT / "research/model_v18_shoulder_state_outcome_manifests"
)
DEFAULT_CHECKPOINT_PROPOSALS = (
    ROOT / "research/model_v18_shoulder_state_checkpoint_proposals"
)
DEFAULT_SCORES = ROOT / "research/model_v18_shoulder_state_scores.csv"
DEFAULT_PICKS = ROOT / "research/model_v18_shoulder_state_picks.csv"
DEFAULT_CALENDAR = ROOT / "research/model_v18_tse_session_calendar.csv"
DEFAULT_RESULT = ROOT / "research/model_v18_shoulder_state_result.json"
DEFAULT_RUNNER = ROOT / "research/model_v18_shoulder_state_runner.py"
DEFAULT_OUTPUT = ROOT / "research/model_v18_shoulder_state_audit.json"
DEFAULT_RUNTIME_LOCK = ROOT / "research/model_v18_runtime_lock.json"
V05_PRICE_LOCK = ROOT / "research/model_v05_input_lock.json"
V04_PARSER_AUDIT = ROOT / "research/model_v04_parser_recovery_audit.json"
V17_REPLAY_INPUT_LOCK = ROOT / "research/model_v17_replay_input_lock.json"

PROTOCOL_ID = "model_v18_shoulder_state_forward_20260804"
PROTOCOL_SHA256 = "c623fabfa8e94381bce27d359cefc6e51a9a80f1c18f62f6098cfdfc8e9f6112"
RUNTIME_LOCK_SHA256 = "95a867e2f8f187528a7ba3d24f4db4f1bf0964531b6e0e2b85d88d0c758f1e32"
RUNTIME_LOCK_SELF_SHA256 = (
    "fcb453b3532625e2eefad679972389bd80cb7ee7685a40610d5aa858d7a32c2b"
)
LOCKED_PYTHON_VERSION = "3.12.13"
LOCKED_NUMPY_VERSION = "2.3.5"
LOCKED_PANDAS_VERSION = "2.2.3"
LOCKED_SCIKIT_LEARN_VERSION = "1.8.0"
CALENDAR_SHA256 = "c5c5908b0e26ebd57eb2e473b9d4ce7f92a7c8b336f6ad70596152de971b77a7"
V05_PRICE_LOCK_SHA256 = (
    "02370bda9c5fe73b166c557bcdc837d363f5deafbe91baa2d450b33dfcd45272"
)
V04_PARSER_AUDIT_SHA256 = (
    "317d5cde9741429263cc91c11575300660f94f62123725fc8ce2d76cdee02eeb"
)
V17_REPLAY_INPUT_LOCK_SHA256 = (
    "1d9a391c8e6b8d2003498c18ba09dac904e672e1f17c05516998ffb71e11c475"
)
SH01 = "SH01_LAGGED_MONTHLY_SHOULDER_STATE"
C00_TOP1 = "C00_PRICE_RIDGE_TOP1"
C02_TOP2 = "C02_C00_RANK2"
STATE_MONTHS = 3
MIN_COMPLETE_PAIRS_PER_MONTH = 10
MIN_FORWARD_SESSIONS = 120
MIN_FORWARD_MONTHS = 6
COSTS_BPS = (20.0, 40.0, 60.0)
PRIMARY_COST_BPS = 40.0
BOOTSTRAP_BLOCK_LENGTH = 20
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_RANDOM_STATE = 20_260_805
BOOTSTRAP_CONFIDENCE = 0.90
ZERO_SHA256 = "0" * 64
NUMERIC_TOLERANCE = 1e-12
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

C00_FEATURES = (
    "oc_last",
    "oc_mean_5",
    "oc_mean_20",
    "oc_mean_60",
    "oc_win_20",
    "oc_std_20",
    "overnight_last",
    "overnight_mean_20",
    "overnight_mean_60",
    "night_day_corr_60",
    "xrank_atr14_pct",
    "xrank_close_momentum_5",
    "xrank_close_momentum_20",
    "xrank_close_momentum_60",
    "xrank_prior_close_location_20",
)
PRICE_RANK_SOURCES = (
    "atr14_pct",
    "close_momentum_5",
    "close_momentum_20",
    "close_momentum_60",
    "prior_close_location_20",
)
LIQUIDITY_RAW_FEATURES = (
    "liq_turnover_med20",
    "liq_volume_med20",
    "liq_turnover_shock1",
    "liq_turnover_log_iqr20",
    "liq_lot_fraction20",
    "liq_close_vwap_dev1",
    "liq_close_vwap_abs_med20",
)
PARSED_PANEL_COLUMNS = (
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

OUTCOME_FIELDS = frozenset(
    {
        "label",
        "oc_return_pct",
        "open_to_close_return_pct",
        "rank1_oc_return_pct",
        "rank2_oc_return_pct",
        "open",
        "high",
        "low",
        "close",
        "outcome_observed",
        "gross_return_pct",
        "net_return_pct",
    }
)

DECISION_REQUIRED_FIELDS = (
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
    "source_manifest_sha256",
    "c00_fold_manifest_sha256",
    "fold_model_bundle_file_sha256",
    "state_manifest_sha256",
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

OUTCOME_REQUIRED_FIELDS = (
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

DECISION_VALUES = frozenset(
    {
        "selected_rank1",
        "selected_rank2",
        "cash_state_zero",
        "cash_state_unavailable",
        "fail_closed_source_empty",
        "fail_closed_source_partial",
        "fail_closed_model_fold",
        "fail_closed_model_pair",
    }
)
CHAIN_COLUMNS = ("sequence_number", "previous_record_sha256", "record_sha256")
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


class AuditError(ValueError):
    """Fail-closed independent-audit error."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_json_object_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{label} must contain a JSON object")
    return value


def read_json(path: str | Path) -> dict[str, Any]:
    raw = _stable_plain_file_bytes(path, label=f"JSON artifact {path}")
    return _parse_json_object_bytes(raw, label=str(path))


def _require_lexical_canonical_path(
    provided: str | Path,
    expected: str | Path,
    *,
    label: str,
) -> Path:
    """Require the registered absolute spelling; never accept a resolved alias."""

    observed = Path(provided)
    canonical = Path(expected)
    if (
        not observed.is_absolute()
        or not canonical.is_absolute()
        or str(observed) != str(canonical)
    ):
        raise AuditError(f"{label} is not the lexical canonical repository path")
    return observed


def _require_plain_directory(path: str | Path, *, label: str) -> Path:
    """Reject a missing/non-directory/symlinked repository artifact root."""

    root = Path(path)
    try:
        metadata = os.lstat(root)
    except OSError as exc:
        raise AuditError(f"{label} is missing: {root}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise AuditError(f"{label} is not a plain directory: {root}")
    return root


def validate_required_array_integrity(value: Mapping[str, Any]) -> int:
    """Reject duplicate entries in every protocol-mandated required array."""

    checked = 0

    def visit(item: Any, path: str) -> None:
        nonlocal checked
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                key = str(raw_key)
                child_path = f"{path}.{key}" if path else key
                required_array_key = (
                    key == "required_fields"
                    or key.startswith("required_")
                    or key.endswith("_required_fields")
                    or key.endswith("_required_paths")
                )
                if required_array_key and isinstance(child, list):
                    identities = [canonical_json_bytes(entry) for entry in child]
                    if len(identities) != len(set(identities)):
                        raise AuditError(
                            f"protocol required array contains duplicates: {child_path}"
                        )
                    checked += 1
                visit(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "")
    return checked


def validate_protocol_contract(
    path: str | Path = DEFAULT_PROTOCOL,
) -> tuple[dict[str, Any], str]:
    protocol_bytes = _stable_plain_file_bytes(path, label="v1.8 protocol")
    protocol = _parse_json_object_bytes(protocol_bytes, label="v1.8 protocol")
    validate_required_array_integrity(protocol)
    observed_sha = hashlib.sha256(protocol_bytes).hexdigest()
    if (
        PROTOCOL_SHA256 != "__PENDING_PROTOCOL_SHA256__"
        and observed_sha != PROTOCOL_SHA256
    ):
        raise AuditError("v1.8 protocol SHA-256 mismatch")
    if protocol.get("schema_version") != 1 or protocol.get("protocol_id") != PROTOCOL_ID:
        raise AuditError("v1.8 protocol identity changed")
    if protocol.get("repository") != "rokuroku-066/TSE-Session-Ranker" or protocol.get(
        "branch"
    ) != "agent/v16-real-data-model-eval-20260728":
        raise AuditError("v1.8 repository/branch changed")
    authority = protocol.get("authority", {})
    if any(
        authority.get(field) is not False
        for field in (
            "production_promotion_allowed",
            "production_model_changed",
            "orders_allowed",
        )
    ) or authority.get("candidate_variants") != 1:
        raise AuditError("v1.8 authority changed")
    candidate = protocol.get("candidate", {})
    if (
        candidate.get("id") != SH01
        or candidate.get("family_size") != 1
        or candidate.get("matched_control") != C00_TOP1
        or candidate.get("outside_frozen_pair_allowed") is not False
        or candidate.get("fallback") is not None
    ):
        raise AuditError("v1.8 candidate contract changed")
    if [item.get("id") for item in protocol.get("controls", [])] != [
        C00_TOP1,
        C02_TOP2,
    ]:
        raise AuditError("v1.8 controls changed")

    state = protocol.get("state_contract", {})
    if (
        state.get("minimum_complete_pairs_per_month")
        != MIN_COMPLETE_PAIRS_PER_MONTH
        or state.get("lookback_calendar_months") != STATE_MONTHS
        or state.get("decision_rule")
        != {
            "state_value_pct_greater_than_zero": 1,
            "state_value_pct_less_than_zero": 2,
            "state_value_pct_exactly_zero": None,
            "any_required_month_unavailable": None,
        }
    ):
        raise AuditError("v1.8 state contract changed")
    evaluation = protocol.get("evaluation", {})
    bootstrap = evaluation.get("bootstrap", {})
    if (
        evaluation.get("candidate_variants") != 1
        or evaluation.get("primary_paired_control") != C00_TOP1
        or evaluation.get("diagnostic_control") != C02_TOP2
        or tuple(float(item) for item in evaluation.get("costs_bps", []))
        != COSTS_BPS
        or float(evaluation.get("primary_cost_bps", -1)) != PRIMARY_COST_BPS
        or bootstrap.get("block_length_sessions") != BOOTSTRAP_BLOCK_LENGTH
        or bootstrap.get("samples") != BOOTSTRAP_SAMPLES
        or bootstrap.get("random_state") != BOOTSTRAP_RANDOM_STATE
        or bootstrap.get("familywise_one_sided_confidence")
        != BOOTSTRAP_CONFIDENCE
        or bootstrap.get("individual_one_sided_confidence")
        != BOOTSTRAP_CONFIDENCE
    ):
        raise AuditError("v1.8 evaluation/bootstrap contract changed")
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
        raise AuditError("v1.8 gate registry changed")
    periods = protocol.get("periods", {})
    if (
        periods.get("not_before_session") != "2026-08-05"
        or periods.get("minimum_scheduled_sessions") != MIN_FORWARD_SESSIONS
        or periods.get("minimum_distinct_calendar_months") != MIN_FORWARD_MONTHS
    ):
        raise AuditError("v1.8 forward stop contract changed")
    artifacts = protocol.get("append_only_artifacts", {})
    if tuple(artifacts.get("decision_record_required_fields", [])) != (
        DECISION_REQUIRED_FIELDS
    ) or tuple(artifacts.get("outcome_record_required_fields", [])) != (
        OUTCOME_REQUIRED_FIELDS
    ):
        raise AuditError("v1.8 ledger schema changed")
    if artifacts.get("decision_values") != sorted(DECISION_VALUES):
        # The protocol order is meaningful documentary output, but the set is
        # the executable contract.
        if set(artifacts.get("decision_values", [])) != DECISION_VALUES:
            raise AuditError("v1.8 decision registry changed")
    calendar = protocol.get("source_contract", {}).get("calendar", {})
    if (
        calendar.get("path") != str(DEFAULT_CALENDAR.relative_to(ROOT))
        or calendar.get("sha256") != CALENDAR_SHA256
        or calendar.get("rows") != 343
    ):
        raise AuditError("v1.8 calendar contract changed")
    load_registered_calendar(DEFAULT_CALENDAR)

    runtime_contract = protocol.get("runtime_lock_contract", {})
    if (
        runtime_contract.get("path")
        != str(DEFAULT_RUNTIME_LOCK.relative_to(ROOT))
        or runtime_contract.get("file_sha256") != RUNTIME_LOCK_SHA256
        or runtime_contract.get("self_sha256") != RUNTIME_LOCK_SELF_SHA256
        or set(runtime_contract.get("required_top_level_fields", ()))
        != {
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
        }
    ):
        raise AuditError("v1.8 runtime-lock protocol binding changed")
    runtime_lock, runtime_file_sha = validate_runtime_lock(
        DEFAULT_RUNTIME_LOCK,
        strict_environment=False,
    )
    if (
        runtime_file_sha != RUNTIME_LOCK_SHA256
        or runtime_lock["runtime_lock_self_sha256"] != RUNTIME_LOCK_SELF_SHA256
    ):
        raise AuditError("v1.8 runtime-lock bytes or self-hash changed")
    github_transport = (
        protocol.get("activation", {})
        .get("github_observation_contract", {})
        .get("transport", {})
    )
    if set(github_transport) != {
        "implementation",
        "scheme",
        "host",
        "fixed_request_headers",
        "token_rule",
        "tls_trust_rule",
        "response_rule",
    } or not all(
        fragment in str(github_transport.get("tls_trust_rule", ""))
        for fragment in (
            "ssl.create_default_context(cafile=the registered cafile_path)",
            "no capath, cadata, default trust-path discovery",
            "SSL_CERT_FILE and SSL_CERT_DIR absent",
        )
    ):
        raise AuditError("v1.8 GitHub TLS trust contract changed")
    first_counted_rule = str(
        protocol.get("activation", {}).get("first_counted_session_rule", "")
    )
    if not all(
        fragment in first_counted_rule
        for fragment in (
            "solely from the immutable workflow server time",
            "must be strictly before that already-selected cutoff",
            "requires a new preregistration rather than shifting",
        )
    ):
        raise AuditError("v1.8 first-counted workflow-time authority changed")
    if tuple(protocol.get("c00_contract", {}).get("score_record_required_fields", ())) != (
        SCORE_FIELDS
    ):
        raise AuditError("v1.8 score schema changed")
    workflow_contract = protocol.get("activation", {}).get(
        "github_workflow_run_observation_contract", {}
    )
    if workflow_contract.get("required_fields") != [
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
    ] or workflow_contract.get("canonical_projection_required_fields") != [
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
    ]:
        raise AuditError("v1.8 GitHub workflow observation schema changed")
    result_contract = protocol.get("result_contract", {})
    if (
        result_contract.get("required_top_level_fields")
        != [
            "schema_version",
            "protocol_id",
            "protocol_sha256",
            "runner_sha256",
            "activation_payload_sha256",
            "activation_receipt_sha256",
            "activation_receipt_commit_sha",
            "status",
            "failure_reason",
            "integrity_stage",
            "authority",
            "input",
            "forward_period",
            "state_months",
            "models",
            "candidate_gate",
            "decision",
            "artifact_sha256",
            "runtime",
        ]
        or result_contract.get("required_status_values")
        != [
            "forward_rejected_candidate",
            "forward_passed_one_v19_research_nominee",
            "aborted_integrity_failure",
        ]
        or result_contract.get("required_runtime_fields")
        != [
            "runtime_lock_sha256",
            "runtime_lock_self_sha256",
            "runtime_lock_verified_at",
            "python_version",
            "numpy_version",
            "pandas_version",
            "scikit_learn_version",
        ]
        or result_contract.get("terminal_decision_required_fields")
        != [
            "research_nominee",
            "production_model_changed",
            "production_promotion_allowed",
            "orders_allowed",
        ]
        or result_contract.get("abort_result_contract", {}).get(
            "abort_decision_required_fields"
        )
        != ["research_nominee", "failure_reason", "integrity_stage"]
    ):
        raise AuditError("v1.8 result/abort schema changed")
    abort_stages = result_contract.get("abort_integrity_stage_values")
    abort_reasons = result_contract.get("abort_failure_reason_values")
    abort_stage_map = result_contract.get("abort_stage_reason_values")
    if (
        not isinstance(abort_stages, list)
        or not isinstance(abort_reasons, list)
        or not isinstance(abort_stage_map, Mapping)
        or set(abort_stage_map) != set(abort_stages)
        or any(
            not isinstance(values, list)
            or not values
            or any(reason not in abort_reasons for reason in values)
            for values in abort_stage_map.values()
        )
        or set().union(*(set(values) for values in abort_stage_map.values()))
        != set(abort_reasons)
    ):
        raise AuditError("v1.8 finite abort reason/stage registry changed")

    for version in ("v16", "v17"):
        binding = protocol.get("prior_result_binding", {}).get(version, {})
        for key, value in binding.items():
            if not key.endswith("_path"):
                continue
            stem = key[: -len("_path")]
            hash_key = f"{stem}_sha256"
            if hash_key in binding and sha256_file(ROOT / str(value)) != binding[hash_key]:
                raise AuditError(f"historical {version} {stem} hash mismatch")
    recompute_registered_seed(protocol)
    return protocol, observed_sha


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise AuditError("canonical datetime must be timezone-aware")
        return value.isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AuditError("canonical JSON rejects non-finite floats")
        return value
    raise AuditError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json_bytes(
    value: Any,
    *,
    exclude_fields: Iterable[str] = (),
) -> bytes:
    excluded = frozenset(str(field) for field in exclude_fields)

    def remove(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): remove(child)
                for key, child in item.items()
                if str(key) not in excluded
            }
        if isinstance(item, (list, tuple)):
            return [remove(child) for child in item]
        return item

    return json.dumps(
        _json_safe(remove(value)),
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


def canonical_json_file_bytes(value: Mapping[str, Any]) -> bytes:
    """Exact pretty JSON bytes used for create-once project JSON artifacts."""

    return (
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def validate_elf_closure(
    value: Mapping[str, Any],
    *,
    strict_environment: bool = False,
) -> dict[str, Any]:
    """Validate the preregistered ELF byte closure without linkage subprocesses."""

    required = {
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
    if not isinstance(value, Mapping) or set(value) != required:
        raise AuditError("ELF closure top-level schema changed")
    closure = dict(value)
    root_fields = ["path", "basename", "roles", "sha256"]
    shared_fields = ["path", "basename", "sha256"]
    if (
        closure["schema_version"] != 1
        or closure["root_object_required_fields"] != root_fields
        or closure["shared_object_required_fields"] != shared_fields
        or closure["virtual_object_exception"] != ["linux-vdso.so.1"]
    ):
        raise AuditError("ELF closure fixed registry changed")

    def validate_objects(
        items: Any,
        *,
        fields: list[str],
        label: str,
    ) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            raise AuditError(f"ELF {label} registry is not an array")
        rows: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping) or list(item) != fields:
                raise AuditError(f"ELF {label} record fields/order changed")
            row = dict(item)
            path = Path(str(row["path"]))
            if not path.is_absolute() or path.name != row["basename"]:
                raise AuditError(f"ELF {label} path/basename changed")
            _require_nonzero_sha(row["sha256"], f"ELF {label} SHA")
            if "roles" in row:
                roles = row["roles"]
                if (
                    not isinstance(roles, list)
                    or not roles
                    or roles != sorted(set(roles))
                    or any(not isinstance(role, str) or not role for role in roles)
                ):
                    raise AuditError("ELF root roles changed")
            rows.append(row)
        paths = [str(row["path"]) for row in rows]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise AuditError(f"ELF {label} paths are not sorted unique")
        return rows

    roots = validate_objects(
        closure["root_objects"], fields=root_fields, label="root object"
    )
    shared = validate_objects(
        closure["shared_objects"], fields=shared_fields, label="shared object"
    )
    if (
        closure["root_object_count"] != len(roots)
        or closure["root_object_set_sha256"] != canonical_json_sha256(roots)
        or closure["shared_object_count"] != len(shared)
        or closure["shared_object_set_sha256"] != canonical_json_sha256(shared)
    ):
        raise AuditError("ELF closure count or set hash changed")
    for label in ("dynamic_loader", "loader_cache"):
        record = closure[label]
        if not isinstance(record, Mapping) or list(record) != shared_fields:
            raise AuditError(f"ELF {label} record changed")
        path = Path(str(record["path"]))
        if not path.is_absolute() or path.name != record["basename"]:
            raise AuditError(f"ELF {label} path/basename changed")
        _require_nonzero_sha(record["sha256"], f"ELF {label} SHA")
    loader_environment = closure["loader_environment"]
    expected_loader_names = {
        "GLIBC_TUNABLES",
        "LD_ASSUME_KERNEL",
        "LD_AUDIT",
        "LD_BIND_NOW",
        "LD_DEBUG",
        "LD_DEBUG_OUTPUT",
        "LD_DYNAMIC_WEAK",
        "LD_HWCAP_MASK",
        "LD_LIBRARY_PATH",
        "LD_ORIGIN_PATH",
        "LD_POINTER_GUARD",
        "LD_PRELOAD",
        "LD_PROFILE",
        "LD_PROFILE_OUTPUT",
        "LD_SHOW_AUXV",
        "LD_TRACE_LOADED_OBJECTS",
        "LD_USE_LOAD_BIAS",
            "PYTHONHOME",
            "PYTHONNOUSERSITE",
            "PYTHONPATH",
            "PYTHONSAFEPATH",
            "PYTHONUSERBASE",
            "SETUPTOOLS_USE_DISTUTILS",
            "SSL_CERT_DIR",
        "SSL_CERT_FILE",
    }
    if (
        not isinstance(loader_environment, Mapping)
        or set(loader_environment) != expected_loader_names
        or any(item is not None for item in loader_environment.values())
    ):
        raise AuditError("ELF loader environment registry changed")
    child_environment = closure["external_process_environment"]
    if child_environment != {
        "inherit_parent_environment": False,
        "common_exact": {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"},
        "git_exact_extra": {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        },
        "pdftotext_exact_extra": {},
    }:
        raise AuditError("ELF external-process environment changed")
    if strict_environment:
        contaminated = sorted(
            name for name in loader_environment if name in os.environ
        )
        if contaminated:
            raise AuditError(
                "operational ELF loader environment is contaminated: "
                + ",".join(contaminated)
            )
        for item in [*roots, *shared, closure["dynamic_loader"], closure["loader_cache"]]:
            path = Path(str(item["path"]))
            if path.is_symlink() or not path.is_file():
                raise AuditError("registered ELF object is missing or symlinked")
            if sha256_file(path) != item["sha256"]:
                raise AuditError(f"registered ELF object bytes changed: {path}")
    return closure


def validate_tls_ca_trust(
    value: Mapping[str, Any],
    *,
    strict_environment: bool = False,
) -> Path:
    """Validate the single explicit HTTPS CA file without default trust paths."""

    fields = {
        "schema_version",
        "cafile_path",
        "cafile_basename",
        "cafile_size_bytes",
        "cafile_sha256",
        "environment",
        "context_rule",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise AuditError("TLS CA trust-store schema changed")
    if value["schema_version"] != 1 or value["environment"] != {
        "SSL_CERT_FILE": None,
        "SSL_CERT_DIR": None,
    }:
        raise AuditError("TLS CA trust-store identity changed")
    if strict_environment and any(name in os.environ for name in value["environment"]):
        raise AuditError("TLS CA environment overrides are forbidden")
    path = Path(str(value["cafile_path"]))
    try:
        observed_stat = path.stat()
    except OSError as exc:
        raise AuditError("registered TLS CA file is unavailable") from exc
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not stat.S_ISREG(observed_stat.st_mode)
        or path.resolve() != path
        or path.name != value["cafile_basename"]
        or isinstance(value["cafile_size_bytes"], bool)
        or not isinstance(value["cafile_size_bytes"], int)
        or observed_stat.st_size != value["cafile_size_bytes"]
        or re.fullmatch(r"[0-9a-f]{64}", str(value["cafile_sha256"])) is None
        or sha256_file(path) != value["cafile_sha256"]
    ):
        raise AuditError("registered TLS CA trust-store bytes changed")
    return path


def _locked_external_process(
    role: str,
    *,
    extra_environment_key: str,
    lock: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, str]]:
    runtime_lock = read_json(DEFAULT_RUNTIME_LOCK) if lock is None else lock
    closure = validate_elf_closure(
        runtime_lock["elf_closure"], strict_environment=False
    )
    matches = [
        item
        for item in closure["root_objects"]
        if role in item["roles"]
    ]
    if len(matches) != 1:
        raise AuditError(f"registered external executable role changed: {role}")
    record = matches[0]
    executable = Path(record["path"])
    if (
        executable.is_symlink()
        or not executable.is_file()
        or sha256_file(executable) != record["sha256"]
    ):
        raise AuditError(f"registered external executable bytes changed: {role}")
    process_environment = closure["external_process_environment"]
    environment = {
        **process_environment["common_exact"],
        **process_environment[extra_environment_key],
    }
    return executable, environment


def _locked_pdf_to_text(pdf_path: str | Path, text_path: str | Path) -> None:
    executable, environment = _locked_external_process(
        "pdftotext_executable",
        extra_environment_key="pdftotext_exact_extra",
    )
    completed = subprocess.run(
        [str(executable), "-layout", str(pdf_path), str(text_path)],
        check=False,
        capture_output=True,
        shell=False,
        env=environment,
    )
    if completed.returncode != 0:
        raise AuditError("locked pdftotext failed to parse registered evidence")


def validate_live_module_origin_closure(
    runtime_lock: Mapping[str, Any],
    *,
    project_root: str | Path = ROOT,
    phase: str = "runtime",
) -> dict[str, str]:
    """Re-establish the exact origin authority for every live Python module."""

    root = Path(project_root).resolve(strict=True)
    runtime = runtime_lock["runtime"]
    startup = runtime["startup_and_module_closure"]
    authorized_files: dict[str, str] = {}

    stdlib_root = Path(sysconfig.get_paths()["stdlib"]).resolve(strict=True)
    stdlib_records: list[dict[str, str]] = []
    for observed_path in stdlib_root.rglob("*"):
        relative = observed_path.relative_to(stdlib_root)
        if (
            "site-packages" in relative.parts
            or "dist-packages" in relative.parts
            or "__pycache__" in relative.parts
            or observed_path.suffix == ".pyc"
        ):
            continue
        if observed_path.is_symlink():
            raise AuditError(f"{phase} live-module stdlib contains a symlink")
        if observed_path.is_file():
            digest = sha256_file(observed_path)
            resolved = str(observed_path.resolve(strict=True))
            authorized_files[resolved] = digest
            stdlib_records.append({"path": relative.as_posix(), "sha256": digest})
    stdlib_records.sort(key=lambda item: item["path"])
    python_lock = runtime["python"]
    if (
        len(stdlib_records) != int(python_lock["stdlib_file_count"])
        or canonical_json_sha256(stdlib_records)
        != python_lock["stdlib_tree_sha256"]
    ):
        raise AuditError(f"{phase} live-module stdlib authority changed")

    for item in runtime_lock["project_files"]:
        path = root / str(item["path"])
        resolved = str(path.resolve(strict=True))
        if path.is_symlink() or sha256_file(path) != item["sha256"]:
            raise AuditError(f"{phase} live-module project authority changed")
        authorized_files[resolved] = str(item["sha256"])

    direct_paths: set[str] = set()
    for relative in startup["direct_activation_project_module_paths"]:
        path = root / str(relative)
        if path.is_symlink() or not path.is_file():
            raise AuditError(f"{phase} direct activation module is unavailable")
        direct_paths.add(str(path.resolve(strict=True)))

    distribution_roots: set[Path] = set()
    for registered in runtime["distributions"]:
        try:
            distribution = importlib.metadata.distribution(registered["name"])
        except importlib.metadata.PackageNotFoundError as exc:
            raise AuditError(
                f"{phase} live-module distribution is missing: {registered['name']}"
            ) from exc
        tree_records: list[dict[str, str]] = []
        for relative in distribution.files or ():
            if relative.as_posix().endswith(".pyc") or "__pycache__" in relative.parts:
                continue
            path = Path(distribution.locate_file(relative))
            if not path.exists() or not path.is_file():
                continue
            if path.is_symlink():
                raise AuditError(f"{phase} live-module distribution contains a symlink")
            digest = sha256_file(path)
            resolved = path.resolve(strict=True)
            authorized_files[str(resolved)] = digest
            distribution_roots.add(resolved.parent)
            tree_records.append({"path": relative.as_posix(), "sha256": digest})
        tree_records.sort(key=lambda item: item["path"])
        if (
            len(tree_records) != int(registered["file_count"])
            or canonical_json_sha256(tree_records) != registered["tree_sha256"]
        ):
            raise AuditError(
                f"{phase} live-module distribution authority changed: "
                f"{registered['name']}"
            )

    allowed_originless = set(startup["originless_runtime_module_names"])
    generated_aliases = {
        str(item["module_name"]): dict(item)
        for item in startup["generated_alias_modules"]
    }
    namespace_roots = {
        root,
        *(
            Path(path).resolve(strict=True)
            for path in startup["site_packages_roots"]
        ),
    }

    def validate_file_origin(module_name: str, raw_path: Any) -> None:
        if not isinstance(raw_path, str) or not raw_path:
            raise AuditError(f"{phase} module {module_name} has an invalid origin")
        path = Path(raw_path)
        if path.suffix == ".pyc" and path.name.endswith(".pyc"):
            try:
                path = Path(importlib.util.source_from_cache(str(path)))
            except ValueError as exc:
                raise AuditError(
                    f"{phase} module {module_name} has an unregistered bytecode origin"
                ) from exc
        if path.is_symlink() or not path.is_file():
            raise AuditError(f"{phase} module {module_name} origin is unavailable")
        resolved = str(path.resolve(strict=True))
        if resolved in direct_paths:
            return
        expected = authorized_files.get(resolved)
        if expected is None or sha256_file(path) != expected:
            raise AuditError(f"{phase} module {module_name} origin is not locked")

    observed: dict[str, str] = {}
    for module_name, module in sorted(sys.modules.items()):
        if module is None:
            continue
        spec = getattr(module, "__spec__", None)
        origin = None if spec is None else getattr(spec, "origin", None)
        module_file = getattr(module, "__file__", None)
        if origin in {"built-in", "frozen"}:
            if module_file not in {None, origin}:
                validate_file_origin(module_name, module_file)
            observed[module_name] = str(origin)
            continue
        candidate = origin if origin not in {None, "namespace"} else module_file
        if candidate is not None:
            validate_file_origin(module_name, candidate)
            if module_file is not None and module_file != candidate:
                validate_file_origin(module_name, module_file)
            observed[module_name] = str(Path(str(candidate)).resolve(strict=True))
            continue
        search_locations = (
            None if spec is None else getattr(spec, "submodule_search_locations", None)
        )
        locations = None if search_locations is None else list(search_locations)
        if locations:
            for location_text in locations:
                location = Path(str(location_text))
                if (
                    not location.is_absolute()
                    or location.is_symlink()
                    or not location.is_dir()
                ):
                    raise AuditError(
                        f"{phase} namespace module {module_name} path is unsafe"
                    )
                resolved_location = location.resolve(strict=True)
                if not any(
                    resolved_location == authority
                    or resolved_location.is_relative_to(authority)
                    for authority in namespace_roots
                ) or not any(
                    Path(path).is_relative_to(resolved_location)
                    for path in authorized_files
                ):
                    raise AuditError(
                        f"{phase} namespace module {module_name} is not locked"
                    )
            observed[module_name] = "namespace"
            continue
        alias = generated_aliases.get(module_name)
        if alias is not None:
            loader = None if spec is None else getattr(spec, "loader", None)
            provider = Path(str(alias["provider_path"]))
            if (
                origin != alias["spec_origin"]
                or module_file != alias["module_file"]
                or (locations if locations is not None else None)
                != alias["search_locations"]
                or loader is None
                or type(loader).__module__ != alias["loader_class_module"]
                or type(loader).__qualname__ != alias["loader_class_qualname"]
                or provider.is_symlink()
                or not provider.is_file()
                or sha256_file(provider) != alias["provider_sha256"]
                or authorized_files.get(str(provider.resolve(strict=True)))
                != alias["provider_sha256"]
            ):
                raise AuditError(
                    f"{phase} generated alias module {module_name} changed"
                )
            observed[module_name] = "generated-alias"
            continue
        if module_name not in allowed_originless:
            raise AuditError(f"{phase} module {module_name} has no registered origin")
        observed[module_name] = "originless"
    return observed


def validate_runtime_lock(
    path: str | Path = DEFAULT_RUNTIME_LOCK,
    *,
    strict_environment: bool = False,
    project_root: str | Path = ROOT,
) -> tuple[dict[str, Any], str]:
    """Validate immutable code closure and, operationally, the full runtime."""

    lock_path = Path(path)
    if not lock_path.is_file() or lock_path.is_symlink():
        raise AuditError("runtime lock is missing or symlinked")
    lock_bytes = _stable_plain_file_bytes(lock_path, label="v1.8 runtime lock")
    observed_file_hash = hashlib.sha256(lock_bytes).hexdigest()
    if observed_file_hash != RUNTIME_LOCK_SHA256:
        raise AuditError(
            "runtime lock exact file SHA-256 mismatch: "
            f"expected {RUNTIME_LOCK_SHA256}, got {observed_file_hash}"
        )
    lock = _parse_json_object_bytes(lock_bytes, label="v1.8 runtime lock")
    required_top = {
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
    }
    if set(lock) != required_top:
        raise AuditError("runtime lock top-level schema changed")
    if (
        lock["schema_version"] != 1
        or lock["lock_id"] != "model_v18_runtime_lock_20260804"
        or lock["registered_on"] != "2026-08-04"
        or lock["canonical_json_contract"] != "project_canonical_json_v1"
    ):
        raise AuditError("runtime lock fixed identity changed")
    expected_self_hash = canonical_json_sha256(
        lock, exclude_fields={"runtime_lock_self_sha256"}
    )
    if lock["runtime_lock_self_sha256"] != expected_self_hash:
        raise AuditError("runtime lock canonical self-hash mismatch")
    if lock["runtime_lock_self_sha256"] != RUNTIME_LOCK_SELF_SHA256:
        raise AuditError("runtime lock registered self-hash changed")

    records = lock["project_files"]
    if not isinstance(records, list):
        raise AuditError("runtime project_files is not an array")
    paths = [str(item.get("path")) for item in records if isinstance(item, Mapping)]
    expected_research = {
        "research/finalize_logit_v04.py",
        "research/model_v13_symbolic_context_runner.py",
        "research/model_v16_liquidity_runner.py",
        "research/model_v17_liquidity_reliability_runner.py",
    }
    root = Path(project_root).resolve(strict=True)
    expected_source = {
        item.relative_to(root).as_posix()
        for item in (root / "src/tse_session_ranker").glob("**/*.py")
    }
    if (
        len(records) != len(paths)
        or paths != sorted(paths)
        or len(set(paths)) != len(paths)
        or set(paths) != expected_source | expected_research
    ):
        raise AuditError("runtime project file closure/order changed")
    for item, relative in zip(records, paths, strict=True):
        if set(item) != {"path", "sha256"}:
            raise AuditError("runtime project file record schema changed")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or relative_path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise AuditError("runtime project file path is unsafe")
        observed_path = root / relative_path
        if observed_path.is_symlink() or not observed_path.is_file():
            raise AuditError("runtime project dependency is missing or symlinked")
        if sha256_file(observed_path) != item["sha256"]:
            raise AuditError(f"runtime project dependency bytes changed: {relative}")
    if lock["project_file_set_sha256"] != canonical_json_sha256(records):
        raise AuditError("runtime project file-set hash mismatch")

    runtime = lock["runtime"]
    required_runtime = {
        "platform",
        "python",
        "startup_and_module_closure",
        "locked_requirements",
        "locked_requirements_sha256",
        "locked_requirements_hash_rule",
        "distribution_tree_hash_rule",
        "distributions",
        "thread_environment",
        "native_threadpools",
        "native_threadpool_rule",
        "numeric_execution_contract",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != required_runtime:
        raise AuditError("runtime environment schema changed")
    python_lock = runtime.get("python")
    if not isinstance(python_lock, Mapping) or set(python_lock) != {
        "implementation",
        "version",
        "cache_tag",
        "executable_sha256",
        "stdlib_file_count",
        "stdlib_tree_sha256",
        "stdlib_tree_hash_rule",
        "ssl",
    }:
        raise AuditError("Python runtime lock schema changed")
    ssl_lock = python_lock.get("ssl")
    if not isinstance(ssl_lock, Mapping) or set(ssl_lock) != {
        "openssl_version",
        "openssl_version_info",
        "ssl_module_origin",
        "ssl_extension_file_sha256",
        "dynamic_libssl_file_sha256",
        "dynamic_libcrypto_file_sha256",
        "ca_trust",
        "binding_rule",
    }:
        raise AuditError("Python SSL runtime lock schema changed")
    validate_tls_ca_trust(
        ssl_lock["ca_trust"], strict_environment=strict_environment
    )
    requirements = runtime["locked_requirements"]
    if (
        not isinstance(requirements, list)
        or requirements != sorted(requirements)
        or runtime["locked_requirements_sha256"]
        != hashlib.sha256(("\n".join(requirements) + "\n").encode()).hexdigest()
    ):
        raise AuditError("runtime locked requirements hash/order changed")
    distributions = runtime["distributions"]
    if (
        not isinstance(distributions, list)
        or [item["name"] for item in distributions]
        != [
            "charset-normalizer",
            "joblib",
            "numpy",
            "pandas",
            "python-dateutil",
            "pytz",
            "scikit-learn",
            "scipy",
            "setuptools",
            "six",
            "threadpoolctl",
        ]
    ):
        raise AuditError("runtime distribution registry changed")
    startup = runtime["startup_and_module_closure"]
    startup_required = {
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
    if not isinstance(startup, Mapping) or set(startup) != startup_required:
        raise AuditError("startup/module closure schema changed")
    startup_files = startup["startup_files"]
    startup_modules = startup["startup_modules"]
    if (
        startup["schema_version"] != 1
        or startup["startup_file_required_fields"]
        != ["path", "basename", "distribution_name", "size_bytes", "sha256"]
        or startup["startup_module_required_fields"]
        != ["module_name", "origin_path", "distribution_name", "origin_sha256"]
        or not isinstance(startup_files, list)
        or not isinstance(startup_modules, list)
        or startup["startup_file_set_sha256"]
        != canonical_json_sha256(startup_files)
        or startup["startup_module_set_sha256"]
        != canonical_json_sha256(startup_modules)
        or startup["customize_modules"]
        != {"sitecustomize": None, "usercustomize": None}
        or startup["direct_activation_project_module_paths"]
        != [
            "research/model_v18_shoulder_state_audit.py",
            "research/model_v18_shoulder_state_runner.py",
        ]
        or startup["originless_runtime_module_names"]
        != sorted(set(startup["originless_runtime_module_names"]))
        or startup["generated_alias_module_required_fields"]
        != [
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
    ):
        raise AuditError("startup/module closure registry changed")
    distribution_names = {item["name"] for item in distributions}
    for item in startup_files:
        if (
            list(item)
            != ["path", "basename", "distribution_name", "size_bytes", "sha256"]
            or not Path(item["path"]).is_absolute()
            or Path(item["path"]).name != item["basename"]
            or item["distribution_name"] not in distribution_names
            or _strict_nonnegative_int(item["size_bytes"], "startup file size") <= 0
        ):
            raise AuditError("startup file registry changed")
        _require_nonzero_sha(item["sha256"], "startup file SHA")
    for item in startup_modules:
        if (
            list(item)
            != ["module_name", "origin_path", "distribution_name", "origin_sha256"]
            or not Path(item["origin_path"]).is_absolute()
            or item["distribution_name"] not in distribution_names
        ):
            raise AuditError("startup module registry changed")
        _require_nonzero_sha(item["origin_sha256"], "startup module SHA")
    generated_aliases = startup["generated_alias_modules"]
    if (
        not isinstance(generated_aliases, list)
        or len(generated_aliases) != 1
        or list(generated_aliases[0])
        != startup["generated_alias_module_required_fields"]
        or generated_aliases[0]["module_name"] != "six.moves"
        or generated_aliases[0]["spec_origin"] is not None
        or generated_aliases[0]["module_file"] is not None
        or generated_aliases[0]["search_locations"] != []
        or generated_aliases[0]["loader_class_module"] != "six"
        or generated_aliases[0]["loader_class_qualname"]
        != "_SixMetaPathImporter"
        or generated_aliases[0]["provider_distribution_name"] != "six"
        or not Path(generated_aliases[0]["provider_path"]).is_absolute()
    ):
        raise AuditError("generated runtime alias registry changed")
    _require_nonzero_sha(
        generated_aliases[0]["provider_sha256"], "generated alias provider SHA"
    )
    pdftotext = lock["pdftotext"]
    if set(pdftotext) != {
        "implementation",
        "executable_path",
        "executable_basename",
        "version",
        "version_output_sha256",
        "executable_sha256",
        "verification_rule",
    }:
        raise AuditError("pdftotext lock schema changed")
    git_lock = lock["git"]
    if set(git_lock) != {
        "implementation",
        "executable_path",
        "executable_basename",
        "version",
        "version_output_sha256",
        "executable_sha256",
        "verification_rule",
    }:
        raise AuditError("Git runtime lock schema changed")

    elf_closure = validate_elf_closure(
        lock["elf_closure"], strict_environment=strict_environment
    )
    for executable_lock, role in (
        (pdftotext, "pdftotext_executable"),
        (git_lock, "git_executable"),
    ):
        matches = [
            item
            for item in elf_closure["root_objects"]
            if role in item["roles"]
        ]
        if len(matches) != 1 or (
            executable_lock["executable_path"] != matches[0]["path"]
            or executable_lock["executable_basename"] != matches[0]["basename"]
            or executable_lock["executable_sha256"] != matches[0]["sha256"]
        ):
            raise AuditError("external executable/ELF root binding changed")

    if not strict_environment:
        return lock, observed_file_hash

    startup_environment_names = (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "PYTHONSAFEPATH",
        "PYTHONUSERBASE",
        "SETUPTOOLS_USE_DISTUTILS",
    )
    if any(os.environ.get(name) is not None for name in startup_environment_names):
        raise AuditError("operational Python startup environment changed")
    registered_startup_by_path = {
        str(item["path"]): dict(item) for item in startup_files
    }
    observed_startup_paths: set[str] = set()
    for root_text in startup["site_packages_roots"]:
        site_root = Path(root_text)
        if (
            not site_root.is_absolute()
            or site_root.is_symlink()
            or not site_root.is_dir()
        ):
            raise AuditError("operational site-packages root changed")
        for path in sorted(site_root.glob("*.pth")):
            metadata = os.lstat(path)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise AuditError("operational startup .pth is not a plain file")
            observed_startup_paths.add(str(path))
            registered = registered_startup_by_path.get(str(path))
            if (
                registered is None
                or metadata.st_size != registered["size_bytes"]
                or _stable_plain_file_sha256(path, label="startup .pth")
                != registered["sha256"]
            ):
                raise AuditError("operational startup .pth bytes changed")
    if observed_startup_paths != set(registered_startup_by_path):
        raise AuditError("operational startup .pth set changed")
    for item in startup_modules:
        module_path = Path(item["origin_path"])
        if (
            module_path.is_symlink()
            or _stable_plain_file_sha256(module_path, label="startup module")
            != item["origin_sha256"]
        ):
            raise AuditError("operational startup module bytes changed")
        module = sys.modules.get(item["module_name"])
        if module is not None and Path(str(module.__file__)).resolve(strict=True) != module_path:
            raise AuditError("operational startup module origin changed")
    user_site = startup["user_site"]
    if set(user_site) != {"path", "must_exist", "must_be_on_sys_path"}:
        raise AuditError("runtime user-site registry changed")
    user_site_path = Path(user_site["path"])
    if (
        user_site_path.exists() != bool(user_site["must_exist"])
        or (str(user_site_path) in sys.path) != bool(user_site["must_be_on_sys_path"])
    ):
        raise AuditError("operational Python user-site state changed")
    for module_name in ("sitecustomize", "usercustomize"):
        if module_name in sys.modules:
            raise AuditError("operational customize startup module appeared")

    platform_lock = runtime["platform"]
    zoneinfo_path = Path("/usr/share/zoneinfo") / str(
        platform_lock["timezone_name"]
    )
    observed_platform = {
        "system": platform.system(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "libc_implementation": platform.libc_ver()[0],
        "libc_version": platform.libc_ver()[1],
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "python_platform_tag": sysconfig.get_platform(),
        "timezone_name": platform_lock["timezone_name"],
        "timezone_zoneinfo_sha256": sha256_file(zoneinfo_path),
        "timezone_rule": platform_lock["timezone_rule"],
    }
    if observed_platform != platform_lock:
        raise AuditError("operational platform differs from runtime lock")
    observed_python = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "cache_tag": sys.implementation.cache_tag,
        "executable_sha256": sha256_file(sys.executable),
    }
    if any(python_lock.get(key) != value for key, value in observed_python.items()):
        raise AuditError("operational Python differs from runtime lock")
    stdlib_root = Path(sysconfig.get_paths()["stdlib"]).resolve(strict=True)
    stdlib_records: list[dict[str, str]] = []
    for observed_path in stdlib_root.rglob("*"):
        relative = observed_path.relative_to(stdlib_root)
        if (
            "site-packages" in relative.parts
            or "dist-packages" in relative.parts
            or "__pycache__" in relative.parts
            or observed_path.suffix == ".pyc"
        ):
            continue
        if observed_path.is_symlink():
            raise AuditError("operational Python stdlib contains a symlink")
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
        raise AuditError("operational Python stdlib tree differs from runtime lock")

    ssl_lock = python_lock["ssl"]
    ssl_spec = importlib.util.find_spec("_ssl")
    ssl_origin = None if ssl_spec is None else ssl_spec.origin
    ssl_extension_hash = None
    if ssl_origin not in {None, "built-in", "frozen"}:
        ssl_path = Path(ssl_origin)
        if ssl_path.is_symlink() or not ssl_path.is_file():
            raise AuditError("operational _ssl extension is missing or symlinked")
        ssl_extension_hash = sha256_file(ssl_path)
    observed_ssl = {
        "openssl_version": ssl.OPENSSL_VERSION,
        "openssl_version_info": list(ssl.OPENSSL_VERSION_INFO),
        "ssl_module_origin": ssl_origin,
        "ssl_extension_file_sha256": ssl_extension_hash,
        # The canonical executable statically embeds the registered OpenSSL;
        # no unregistered platform introspection command (for example ldd)
        # participates in the operational trust boundary.
        "dynamic_libssl_file_sha256": None,
        "dynamic_libcrypto_file_sha256": None,
        "ca_trust": dict(ssl_lock["ca_trust"]),
        "binding_rule": ssl_lock["binding_rule"],
    }
    if observed_ssl != ssl_lock:
        raise AuditError("operational Python/OpenSSL binding differs from runtime lock")
    for key, expected in runtime["thread_environment"].items():
        if os.environ.get(key) != expected:
            raise AuditError(f"operational thread environment differs: {key}")

    for registered in distributions:
        if set(registered) != {
            "name",
            "import_name",
            "version",
            "file_count",
            "tree_sha256",
            "module_relative_path",
            "module_file_sha256",
        }:
            raise AuditError("runtime distribution record schema changed")
        try:
            distribution = importlib.metadata.distribution(registered["name"])
        except importlib.metadata.PackageNotFoundError as exc:
            raise AuditError(
                f"operational distribution is missing: {registered['name']}"
            ) from exc
        if distribution.version != registered["version"]:
            raise AuditError(
                f"operational distribution version changed: {registered['name']}"
            )
        tree_records: list[dict[str, str]] = []
        for relative in distribution.files or ():
            relative_text = relative.as_posix()
            if relative_text.endswith(".pyc") or "__pycache__" in relative.parts:
                continue
            observed_path = Path(distribution.locate_file(relative))
            if not observed_path.exists() or not observed_path.is_file():
                continue
            if observed_path.is_symlink():
                raise AuditError("runtime distribution contains a symlinked file")
            tree_records.append(
                {"path": relative_text, "sha256": sha256_file(observed_path)}
            )
        tree_records.sort(key=lambda item: item["path"])
        if (
            len(tree_records) != int(registered["file_count"])
            or canonical_json_sha256(tree_records) != registered["tree_sha256"]
        ):
            raise AuditError(
                f"operational distribution tree changed: {registered['name']}"
            )
        module_path = Path(
            distribution.locate_file(registered["module_relative_path"])
        )
        module = importlib.import_module(registered["import_name"])
        imported_path = Path(str(module.__file__)).resolve(strict=True)
        if (
            module_path.resolve(strict=True) != imported_path
            or module_path.is_symlink()
            or sha256_file(module_path) != registered["module_file_sha256"]
        ):
            raise AuditError(
                f"operational imported module changed: {registered['name']}"
            )

    import scipy  # noqa: F401  # imported before native threadpool inspection
    import sklearn  # noqa: F401
    import threadpoolctl

    native: list[dict[str, Any]] = []
    for item in threadpoolctl.threadpool_info():
        library = Path(item["filepath"])
        if library.is_symlink() or not library.is_file():
            raise AuditError("operational native library is missing or symlinked")
        native.append(
            {
                field: item.get(field)
                for field in (
                    "user_api",
                    "internal_api",
                    "prefix",
                    "version",
                    "threading_layer",
                    "architecture",
                    "num_threads",
                )
            }
            | {
                "library_file_name": library.name,
                "library_sha256": sha256_file(library),
            }
        )
    native.sort(
        key=lambda item: (
            item["user_api"],
            item["internal_api"],
            item["prefix"],
            item["library_file_name"],
        )
    )
    # Startup capacity is provenance only.  Exact backend identity/library
    # bytes and the registered one-thread execution boundary are the stable
    # operational contract across hosts with different CPU counts.
    without_startup_capacity = lambda rows: [
        {key: child for key, child in item.items() if key != "num_threads"}
        for item in rows
    ]
    if without_startup_capacity(native) != without_startup_capacity(
        runtime["native_threadpools"]
    ):
        raise AuditError("operational native threadpool registry changed")
    limit = int(runtime["numeric_execution_contract"]["threadpoolctl_limit"])
    with threadpoolctl.threadpool_limits(limits=limit):
        if any(
            int(item.get("num_threads", -1)) != limit
            for item in threadpoolctl.threadpool_info()
        ):
            raise AuditError("numeric threadpool limit is not enforceable")

    executable_path, pdftotext_environment = _locked_external_process(
        "pdftotext_executable",
        extra_environment_key="pdftotext_exact_extra",
        lock=lock,
    )
    if (
        str(executable_path) != pdftotext["executable_path"]
        or executable_path.name != pdftotext["executable_basename"]
        or sha256_file(executable_path) != pdftotext["executable_sha256"]
    ):
        raise AuditError("operational pdftotext executable changed")
    version = subprocess.run(
        [str(executable_path), "-v"],
        check=True,
        capture_output=True,
        env=pdftotext_environment,
        shell=False,
    )
    version_bytes = version.stderr or version.stdout
    if (
        hashlib.sha256(version_bytes).hexdigest()
        != pdftotext["version_output_sha256"]
        or f"pdftotext version {pdftotext['version']}".encode() not in version_bytes
    ):
        raise AuditError("operational pdftotext version output changed")
    git_path, git_environment = _locked_external_process(
        "git_executable", extra_environment_key="git_exact_extra", lock=lock
    )
    if (
        str(git_path) != git_lock["executable_path"]
        or git_path.name != git_lock["executable_basename"]
        or sha256_file(git_path) != git_lock["executable_sha256"]
    ):
        raise AuditError("operational Git executable changed")
    git_version = subprocess.run(
        [str(git_path), "--version"],
        check=True,
        capture_output=True,
        env=git_environment,
        shell=False,
    )
    if (
        git_version.stderr
        or hashlib.sha256(git_version.stdout).hexdigest()
        != git_lock["version_output_sha256"]
        or git_version.stdout != f"git version {git_lock['version']}\n".encode()
    ):
        raise AuditError("operational Git version output changed")
    validate_live_module_origin_closure(
        lock,
        project_root=project_root,
        phase="initial strict runtime",
    )
    return lock, observed_file_hash


def _aware_timestamp(value: Any, name: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{name} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise AuditError(f"{name} must be timezone-aware")
    return parsed


def _runtime_verification_timestamp(
    value: Mapping[str, Any],
    name: str,
) -> pd.Timestamp:
    if value.get("runtime_lock_sha256") != RUNTIME_LOCK_SHA256:
        raise AuditError(f"{name} runtime-lock binding changed")
    return _aware_timestamp(
        value.get("runtime_lock_verified_at"),
        f"{name} runtime_lock_verified_at",
    )


def validate_github_observation(
    observation: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    observation_kind: str,
    requested_commit_sha: str | None = None,
) -> dict[str, Any]:
    """Validate one immutable normalized historical GitHub observation."""

    contract = protocol["activation"]["github_observation_contract"]
    required = set(contract["required_fields"])
    if not isinstance(observation, Mapping) or set(observation) != required:
        raise AuditError("GitHub observation fields changed")
    value = dict(observation)
    for field, expected in contract["fixed_values"].items():
        if value.get(field) != expected:
            raise AuditError(f"GitHub observation fixed field changed: {field}")
    if value["observation_kind"] != observation_kind:
        raise AuditError("GitHub observation kind changed")
    repository = str(protocol["repository"])
    branch = str(protocol["branch"])
    if observation_kind == "commit":
        if (
            requested_commit_sha is None
            or re.fullmatch(r"[0-9a-f]{40}", requested_commit_sha) is None
            or value["branch"] is not None
            or value["requested_commit_sha"] != requested_commit_sha
            or value["endpoint"]
            != f"/repos/{repository}/commits/{requested_commit_sha}"
        ):
            raise AuditError("GitHub commit observation subject changed")
    elif observation_kind == "branch_tip":
        if (
            requested_commit_sha is not None
            or value["branch"] != branch
            or value["requested_commit_sha"] is not None
            or value["endpoint"]
            != f"/repos/{repository}/commits/{quote(branch, safe='')}"
        ):
            raise AuditError("GitHub branch observation subject changed")
    else:
        raise AuditError("GitHub observation kind is unregistered")

    projection = value["canonical_projection"]
    if not isinstance(projection, Mapping) or set(projection) != set(
        contract["canonical_projection_required_fields"]
    ):
        raise AuditError("GitHub observation projection fields changed")
    commit_sha = str(projection["commit_sha"])
    parent_shas = projection["parent_shas"]
    if not isinstance(parent_shas, list) or any(
        not isinstance(item, str)
        or re.fullmatch(r"[0-9a-f]{40}", item) is None
        for item in parent_shas
    ):
        raise AuditError("GitHub observation parent projection is invalid")
    if (
        re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None
        or parent_shas != sorted(set(parent_shas))
        or projection["html_url"]
        != f"https://github.com/{repository}/commit/{commit_sha}"
    ):
        raise AuditError("GitHub observation canonical projection is invalid")
    _aware_timestamp(projection["committer_date"], "GitHub committer_date")
    if observation_kind == "commit" and commit_sha != requested_commit_sha:
        raise AuditError("GitHub commit observation returned another commit")

    for field in ("content_type", "http_date", "etag"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise AuditError(f"GitHub observation {field} is empty")
    if not value["content_type"].lower().split(";", 1)[0].strip().endswith(
        "json"
    ):
        raise AuditError("GitHub observation content type is not JSON")
    try:
        http_date = pd.Timestamp(parsedate_to_datetime(value["http_date"]))
    except (TypeError, ValueError, OverflowError) as exc:
        raise AuditError("GitHub observation Date header is invalid") from exc
    retrieved = _aware_timestamp(value["retrieved_at"], "GitHub retrieved_at")
    if http_date.tzinfo is None or http_date > retrieved:
        raise AuditError("GitHub observation response/retrieval time is not causal")
    for field in (
        "response_body_sha256",
        "response_headers_sha256",
        "observation_sha256",
    ):
        _require_nonzero_sha(value[field], f"GitHub observation {field}")
    expected_headers_hash = canonical_json_sha256(
        {
            "status": value["http_status"],
            "content_type": value["content_type"],
            "date": value["http_date"],
            "etag": value["etag"],
        }
    )
    if value["response_headers_sha256"] != expected_headers_hash:
        raise AuditError("GitHub observation response-header hash changed")
    expected_hash = canonical_json_sha256(
        value, exclude_fields={"observation_sha256"}
    )
    if value["observation_sha256"] != expected_hash:
        raise AuditError("GitHub observation self-hash mismatch")
    return value


def validate_github_workflow_observation(
    observation: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    expected_head_sha: str,
    checkpoint: bool = False,
) -> dict[str, Any]:
    """Validate one immutable GitHub Actions run observation."""

    contract = (
        protocol["daily_preopen_checkpoint_contract"][
            "checkpoint_workflow_run_observation_contract"
        ]
        if checkpoint
        else protocol["activation"]["github_workflow_run_observation_contract"]
    )
    required = set(contract["required_fields"])
    if not isinstance(observation, Mapping) or set(observation) != required:
        raise AuditError("GitHub workflow observation fields changed")
    value = dict(observation)
    for field, expected in contract["fixed_values"].items():
        if value.get(field) != expected:
            raise AuditError(
                f"GitHub workflow observation fixed field changed: {field}"
            )
    if re.fullmatch(r"[0-9a-f]{40}", expected_head_sha) is None:
        raise AuditError("GitHub workflow expected head SHA is invalid")
    run_id = value["run_id"]
    allowed_conclusions = (
        set(contract["terminal_conclusion_values"])
        if checkpoint
        else {"success"}
    )
    if (
        isinstance(run_id, bool)
        or not isinstance(run_id, int)
        or run_id <= 0
        or value["expected_head_sha"] != expected_head_sha
        or value["endpoint"]
        != f"/repos/{protocol['repository']}/actions/runs/{run_id}"
    ):
        raise AuditError("GitHub workflow observation subject changed")

    projection = value["canonical_projection"]
    if not isinstance(projection, Mapping) or set(projection) != set(
        contract["canonical_projection_required_fields"]
    ):
        raise AuditError("GitHub workflow projection fields changed")
    workflow_id = projection["workflow_id"]
    projected_run_id = projection["run_id"]
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in (workflow_id, projected_run_id)
    ):
        raise AuditError("GitHub workflow identifiers are invalid")
    workflow_url = urlparse(str(projection["html_url"]))
    if (
        projected_run_id != run_id
        or projection["workflow_name"] != "tests"
        or projection["workflow_path"] != ".github/workflows/tests.yml"
        or projection["event"] != "pull_request"
        or projection["head_sha"] != expected_head_sha
        or projection["run_attempt"] != 1
        or projection["status"] != "completed"
        or projection["conclusion"] not in allowed_conclusions
        or workflow_url.scheme != "https"
        or workflow_url.hostname != "github.com"
        or workflow_url.path != (
            f"/{protocol['repository']}/actions/runs/{run_id}"
        )
        or workflow_url.query
        or workflow_url.fragment
    ):
        raise AuditError("GitHub workflow canonical projection is invalid")
    created = _aware_timestamp(
        projection["created_at"], "GitHub workflow created_at"
    )
    started = _aware_timestamp(
        projection["run_started_at"], "GitHub workflow run_started_at"
    )
    updated = _aware_timestamp(
        projection["updated_at"], "GitHub workflow updated_at"
    )
    if not created <= started <= updated:
        raise AuditError("GitHub workflow server timestamps are not causal")

    for field in ("content_type", "http_date", "etag"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise AuditError(f"GitHub workflow observation {field} is empty")
    if not value["content_type"].lower().split(";", 1)[0].strip().endswith(
        "json"
    ):
        raise AuditError("GitHub workflow observation content type is not JSON")
    try:
        http_date = pd.Timestamp(parsedate_to_datetime(value["http_date"]))
    except (TypeError, ValueError, OverflowError) as exc:
        raise AuditError("GitHub workflow Date header is invalid") from exc
    retrieved = _aware_timestamp(
        value["retrieved_at"], "GitHub workflow retrieved_at"
    )
    if http_date.tzinfo is None or http_date > retrieved or retrieved < updated:
        raise AuditError("GitHub workflow response/retrieval time is not causal")
    for field in (
        "response_body_sha256",
        "response_headers_sha256",
        "observation_sha256",
    ):
        _require_nonzero_sha(value[field], f"GitHub workflow observation {field}")
    expected_headers_hash = canonical_json_sha256(
        {
            "status": value["http_status"],
            "content_type": value["content_type"],
            "date": value["http_date"],
            "etag": value["etag"],
        }
    )
    if value["response_headers_sha256"] != expected_headers_hash:
        raise AuditError("GitHub workflow response-header hash changed")
    expected_hash = canonical_json_sha256(
        value, exclude_fields={"observation_sha256"}
    )
    if value["observation_sha256"] != expected_hash:
        raise AuditError("GitHub workflow observation self-hash mismatch")
    return value


def _validate_workflow_observation_derivation(
    *,
    observation: Mapping[str, Any],
    protocol: Mapping[str, Any],
    expected_head_sha: Any,
    run_id: Any,
    updated_at: Any,
    observed_at: Any,
    label: str,
    checkpoint: bool = False,
) -> dict[str, Any]:
    value = validate_github_workflow_observation(
        observation,
        protocol,
        expected_head_sha=str(expected_head_sha),
        checkpoint=checkpoint,
    )
    projection = value["canonical_projection"]
    if (
        isinstance(run_id, bool)
        or not isinstance(run_id, int)
        or run_id != value["run_id"]
        or run_id != projection["run_id"]
        or _aware_timestamp(projection["updated_at"], f"{label} updated_at")
        != _aware_timestamp(updated_at, f"{label} flat updated_at")
        or _aware_timestamp(value["retrieved_at"], f"{label} retrieved_at")
        != _aware_timestamp(observed_at, f"{label} flat observed_at")
    ):
        raise AuditError(f"{label} flat fields do not derive from workflow observation")
    return value


def _validate_observation_derivation(
    *,
    commit_observation: Mapping[str, Any],
    branch_observation: Mapping[str, Any],
    protocol: Mapping[str, Any],
    commit_sha: Any,
    commit_url: Any,
    committed_at: Any,
    branch_tip_sha: Any,
    observed_at: Any,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    commit = validate_github_observation(
        commit_observation,
        protocol,
        observation_kind="commit",
        requested_commit_sha=str(commit_sha),
    )
    branch = validate_github_observation(
        branch_observation,
        protocol,
        observation_kind="branch_tip",
    )
    projection = commit["canonical_projection"]
    branch_projection = branch["canonical_projection"]
    derived_observed = max(
        _aware_timestamp(commit["retrieved_at"], f"{label} commit retrieved_at"),
        _aware_timestamp(branch["retrieved_at"], f"{label} branch retrieved_at"),
    )
    if (
        projection["commit_sha"] != commit_sha
        or projection["html_url"] != commit_url
        or _aware_timestamp(projection["committer_date"], f"{label} committer_date")
        != _aware_timestamp(committed_at, f"{label} committed_at")
        or branch_projection["commit_sha"] != branch_tip_sha
        or derived_observed != _aware_timestamp(observed_at, f"{label} observed_at")
    ):
        raise AuditError(f"{label} flat fields do not derive from observations")
    return commit, branch


def validate_activation_payload(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> str:
    contract = protocol["activation"]["payload"]
    required = set(contract["required_fields"])
    if set(payload) != required:
        raise AuditError("activation payload fields changed")
    for field, expected in contract["fixed_values"].items():
        if payload.get(field) != expected:
            raise AuditError(f"activation payload fixed field changed: {field}")
    preregistration_commit = str(payload["preregistration_commit_sha"])
    if (
        re.fullmatch(r"[0-9a-f]{40}", preregistration_commit) is None
        or payload["preregistration_commit_url"]
        != (
            "https://github.com/rokuroku-066/TSE-Session-Ranker/commit/"
            + preregistration_commit
        )
    ):
        raise AuditError("activation payload preregistration commit is invalid")
    _validate_observation_derivation(
        commit_observation=payload["preregistration_commit_observation"],
        branch_observation=payload["preregistration_branch_observation"],
        protocol=protocol,
        commit_sha=payload["preregistration_commit_sha"],
        commit_url=payload["preregistration_commit_url"],
        committed_at=payload["preregistration_commit_committed_at"],
        branch_tip_sha=payload["preregistration_branch_tip_sha_when_observed"],
        observed_at=payload["preregistration_commit_observed_at"],
        label="preregistration observation",
    )
    _validate_workflow_observation_derivation(
        observation=payload["preregistration_workflow_run_observation"],
        protocol=protocol,
        expected_head_sha=payload["preregistration_commit_sha"],
        run_id=payload["preregistration_workflow_run_id"],
        updated_at=payload["preregistration_workflow_run_updated_at"],
        observed_at=payload["preregistration_workflow_run_observed_at"],
        label="preregistration workflow observation",
    )
    artifact_pairs = (
        ("protocol_path", "protocol_sha256"),
        ("hypothesis_path", "hypothesis_sha256"),
        ("runtime_lock_path", "runtime_lock_sha256"),
        ("runner_path", "runner_sha256"),
        ("audit_path", "audit_sha256"),
        ("tests_path", "tests_sha256"),
        ("session_calendar_path", "session_calendar_sha256"),
        ("workflow_path", "workflow_sha256"),
    )
    for path_field, hash_field in artifact_pairs:
        path = ROOT / str(payload[path_field])
        if sha256_file(path) != payload[hash_field]:
            raise AuditError(f"activation payload artifact mismatch: {path_field}")
    if payload["runtime_lock_sha256"] != RUNTIME_LOCK_SHA256:
        raise AuditError("activation payload runtime-lock binding changed")
    committed = _aware_timestamp(
        payload["preregistration_commit_committed_at"],
        "preregistration_commit_committed_at",
    )
    observed = _aware_timestamp(
        payload["preregistration_commit_observed_at"],
        "preregistration_commit_observed_at",
    )
    observed_tip = str(payload["preregistration_branch_tip_sha_when_observed"])
    if observed < committed or re.fullmatch(r"[0-9a-f]{40}", observed_tip) is None:
        raise AuditError("activation preregistration observation is invalid")
    expected_hash = canonical_json_sha256(
        payload, exclude_fields={"payload_sha256"}
    )
    if payload["payload_sha256"] != expected_hash:
        raise AuditError("activation payload self-hash mismatch")
    return expected_hash


def validate_activation_receipt(
    receipt: Mapping[str, Any],
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    payload_file_path: str | Path | None = None,
) -> str:
    contract = protocol["activation"]["receipt"]
    required = set(contract["required_fields"])
    if set(receipt) != required:
        raise AuditError("activation receipt fields changed")
    for field, expected in contract["fixed_values"].items():
        if receipt.get(field) != expected:
            raise AuditError(f"activation receipt fixed field changed: {field}")
    if receipt["activation_id"] != payload["activation_id"]:
        raise AuditError("activation receipt id differs from payload")
    if receipt["payload_sha256"] != payload["payload_sha256"]:
        raise AuditError("activation receipt does not bind payload")
    expected_payload_file_sha = (
        sha256_file(payload_file_path)
        if payload_file_path is not None
        else hashlib.sha256(canonical_json_file_bytes(payload)).hexdigest()
    )
    if receipt["payload_file_sha256"] != expected_payload_file_sha:
        raise AuditError("activation receipt does not bind exact payload file bytes")
    commit = str(receipt["payload_commit_sha"])
    if (
        re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or receipt["payload_commit_url"]
        != (
            "https://github.com/rokuroku-066/TSE-Session-Ranker/commit/"
            + commit
        )
    ):
        raise AuditError("activation receipt payload commit is invalid")
    _validate_observation_derivation(
        commit_observation=receipt["payload_commit_observation"],
        branch_observation=receipt["payload_branch_observation"],
        protocol=protocol,
        commit_sha=receipt["payload_commit_sha"],
        commit_url=receipt["payload_commit_url"],
        committed_at=receipt["payload_commit_committed_at"],
        branch_tip_sha=receipt["branch_tip_sha_when_payload_observed"],
        observed_at=receipt["payload_observed_at"],
        label="payload observation",
    )
    _validate_workflow_observation_derivation(
        observation=receipt["payload_workflow_run_observation"],
        protocol=protocol,
        expected_head_sha=receipt["payload_commit_sha"],
        run_id=receipt["payload_workflow_run_id"],
        updated_at=receipt["payload_workflow_run_updated_at"],
        observed_at=receipt["payload_workflow_run_observed_at"],
        label="payload workflow observation",
    )
    committed = _aware_timestamp(
        receipt["payload_commit_committed_at"], "payload_commit_committed_at"
    )
    observed = _aware_timestamp(receipt["payload_observed_at"], "payload_observed_at")
    workflow_updated = _aware_timestamp(
        receipt["payload_workflow_run_updated_at"],
        "payload_workflow_run_updated_at",
    )
    workflow_observed = _aware_timestamp(
        receipt["payload_workflow_run_observed_at"],
        "payload_workflow_run_observed_at",
    )
    issued = _aware_timestamp(receipt["receipt_issued_at"], "receipt_issued_at")
    if observed < committed or issued < max(observed, workflow_updated, workflow_observed):
        raise AuditError("activation receipt timestamps are not causal")
    expected_hash = canonical_json_sha256(
        receipt, exclude_fields={"receipt_sha256"}
    )
    if receipt["receipt_sha256"] != expected_hash:
        raise AuditError("activation receipt self-hash mismatch")
    return expected_hash


def _git_command(
    repository: str | Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    executable_path, git_environment = _locked_external_process(
        "git_executable",
        extra_environment_key="git_exact_extra",
    )
    completed = subprocess.run(
        [str(executable_path), "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        env=git_environment,
        shell=False,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AuditError(f"activation git evidence failed: {detail}")
    return completed


def _github_api_fetch(endpoint: str) -> Any:
    """Fetch one authenticated GitHub API object for terminal verification."""

    if (
        not endpoint.lstrip("/").startswith("repos/")
        or "://" in endpoint
        or ".." in endpoint.split("/")
    ):
        raise AuditError("GitHub API evidence endpoint is unsafe")

    runtime_lock, _ = validate_runtime_lock(
        DEFAULT_RUNTIME_LOCK, strict_environment=False
    )
    ca_contract = runtime_lock["runtime"]["python"]["ssl"]["ca_trust"]
    ca_path = validate_tls_ca_trust(ca_contract, strict_environment=True)
    try:
        tls_context = ssl.create_default_context(cafile=str(ca_path))
    except (OSError, ssl.SSLError) as exc:
        raise AuditError("registered GitHub TLS context creation failed") from exc
    if not tls_context.check_hostname or tls_context.verify_mode != ssl.CERT_REQUIRED:
        raise AuditError("registered GitHub TLS verification policy changed")
    if validate_tls_ca_trust(
        ca_contract, strict_environment=True
    ) != ca_path:
        raise AuditError("GitHub API CA trust-store changed during context creation")

    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
            return None

    token = os.environ.get("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "Accept-Encoding": "identity",
        "User-Agent": "TSE-Session-Ranker-model-v18-activation",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        "https://api.github.com/" + endpoint.lstrip("/"),
        headers=headers,
        method="GET",
    )
    try:
        opener = build_opener(
            NoRedirect(),
            HTTPSHandler(context=tls_context),
        )
        with opener.open(request) as response:
            final = urlparse(response.geturl())
            if (
                response.status != 200
                or final.scheme != "https"
                or final.hostname != "api.github.com"
                or response.headers.get("Content-Encoding") not in {None, "identity"}
                or not response.headers.get("Content-Type")
                or not str(response.headers.get("Content-Type"))
                .lower()
                .split(";", 1)[0]
                .strip()
                .endswith("json")
                or not response.headers.get("Date")
                or not response.headers.get("ETag")
            ):
                raise AuditError("GitHub API transport response changed")
            body = response.read()
    except (HTTPError, URLError, OSError) as exc:
        raise AuditError("GitHub API evidence fetch failed") from exc
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError("GitHub API evidence is not valid JSON") from exc
    if not isinstance(value, (Mapping, list)):
        raise AuditError("GitHub API evidence is not an object or array")
    return value


def validate_activation_git_history(
    payload: Mapping[str, Any],
    receipt: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    repository_root: str | Path = ROOT,
    payload_file_path: str | Path = DEFAULT_ACTIVATION_PAYLOAD,
    receipt_file_path: str | Path = DEFAULT_ACTIVATION_RECEIPT,
    github_fetcher: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify activation against real local Git objects, bytes, and ancestry."""

    if not decisions:
        raise AuditError("activation Git verification needs a counted decision")
    receipt_evidence = {
        (
            row["activation_receipt_commit_sha"],
            row["activation_receipt_commit_url"],
            row["activation_receipt_commit_committed_at"],
            row["activation_receipt_commit_observed_at"],
            row["branch_tip_sha_when_receipt_observed"],
            row["activation_receipt_file_sha256"],
            canonical_json_sha256(row["receipt_commit_observation"]),
            canonical_json_sha256(row["receipt_branch_observation"]),
            row["activation_receipt_workflow_run_id"],
            row["activation_receipt_workflow_run_updated_at"],
            row["activation_receipt_workflow_run_observed_at"],
            canonical_json_sha256(row["receipt_workflow_run_observation"]),
        )
        for row in decisions
    }
    if len(receipt_evidence) != 1:
        raise AuditError("activation receipt observation changed across decisions")
    repository = Path(repository_root)
    fetch_github = _github_api_fetch if github_fetcher is None else github_fetcher
    if _git_command(repository, "rev-parse", "--is-inside-work-tree").stdout.strip() != b"true":
        raise AuditError("activation evidence root is not a Git work tree")
    commit_fields = {
        "preregistration": str(payload["preregistration_commit_sha"]),
        "preregistration_observed_tip": str(
            payload["preregistration_branch_tip_sha_when_observed"]
        ),
        "payload": str(receipt["payload_commit_sha"]),
        "receipt": str(decisions[0]["activation_receipt_commit_sha"]),
        "receipt_observed_tip": str(
            decisions[0]["branch_tip_sha_when_receipt_observed"]
        ),
        "payload_observed_tip": str(receipt["branch_tip_sha_when_payload_observed"]),
    }
    for label, commit in commit_fields.items():
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise AuditError(f"activation {label} is not a full Git commit SHA")
        if _git_command(
            repository, "cat-file", "-e", f"{commit}^{{commit}}", check=False
        ).returncode != 0:
            raise AuditError(f"activation {label} commit does not exist locally")

    def commit_timestamp(commit: str) -> pd.Timestamp:
        value = _git_command(repository, "show", "-s", "--format=%cI", commit)
        return _aware_timestamp(value.stdout.decode().strip(), "Git commit timestamp")

    prereg_commit_observation, prereg_branch_observation = (
        _validate_observation_derivation(
            commit_observation=payload["preregistration_commit_observation"],
            branch_observation=payload["preregistration_branch_observation"],
            protocol=protocol,
            commit_sha=payload["preregistration_commit_sha"],
            commit_url=payload["preregistration_commit_url"],
            committed_at=payload["preregistration_commit_committed_at"],
            branch_tip_sha=payload["preregistration_branch_tip_sha_when_observed"],
            observed_at=payload["preregistration_commit_observed_at"],
            label="preregistration observation",
        )
    )
    payload_commit_observation, payload_branch_observation = (
        _validate_observation_derivation(
            commit_observation=receipt["payload_commit_observation"],
            branch_observation=receipt["payload_branch_observation"],
            protocol=protocol,
            commit_sha=receipt["payload_commit_sha"],
            commit_url=receipt["payload_commit_url"],
            committed_at=receipt["payload_commit_committed_at"],
            branch_tip_sha=receipt["branch_tip_sha_when_payload_observed"],
            observed_at=receipt["payload_observed_at"],
            label="payload observation",
        )
    )
    receipt_commit_observation, receipt_branch_observation = (
        _validate_observation_derivation(
            commit_observation=decisions[0]["receipt_commit_observation"],
            branch_observation=decisions[0]["receipt_branch_observation"],
            protocol=protocol,
            commit_sha=decisions[0]["activation_receipt_commit_sha"],
            commit_url=decisions[0]["activation_receipt_commit_url"],
            committed_at=decisions[0]["activation_receipt_commit_committed_at"],
            branch_tip_sha=decisions[0]["branch_tip_sha_when_receipt_observed"],
            observed_at=decisions[0]["activation_receipt_commit_observed_at"],
            label="receipt observation",
        )
    )
    prereg_workflow_observation = _validate_workflow_observation_derivation(
        observation=payload["preregistration_workflow_run_observation"],
        protocol=protocol,
        expected_head_sha=commit_fields["preregistration"],
        run_id=payload["preregistration_workflow_run_id"],
        updated_at=payload["preregistration_workflow_run_updated_at"],
        observed_at=payload["preregistration_workflow_run_observed_at"],
        label="preregistration workflow observation",
    )
    payload_workflow_observation = _validate_workflow_observation_derivation(
        observation=receipt["payload_workflow_run_observation"],
        protocol=protocol,
        expected_head_sha=commit_fields["payload"],
        run_id=receipt["payload_workflow_run_id"],
        updated_at=receipt["payload_workflow_run_updated_at"],
        observed_at=receipt["payload_workflow_run_observed_at"],
        label="payload workflow observation",
    )
    receipt_workflow_observation = _validate_workflow_observation_derivation(
        observation=decisions[0]["receipt_workflow_run_observation"],
        protocol=protocol,
        expected_head_sha=commit_fields["receipt"],
        run_id=decisions[0]["activation_receipt_workflow_run_id"],
        updated_at=decisions[0]["activation_receipt_workflow_run_updated_at"],
        observed_at=decisions[0]["activation_receipt_workflow_run_observed_at"],
        label="receipt workflow observation",
    )
    historical_observations = (
        (commit_fields["preregistration"], prereg_commit_observation),
        (commit_fields["preregistration_observed_tip"], prereg_branch_observation),
        (commit_fields["payload"], payload_commit_observation),
        (commit_fields["payload_observed_tip"], payload_branch_observation),
        (commit_fields["receipt"], receipt_commit_observation),
        (commit_fields["receipt_observed_tip"], receipt_branch_observation),
    )
    for commit, observation in historical_observations:
        projection = observation["canonical_projection"]
        local_parents = sorted(
            _git_command(repository, "show", "-s", "--format=%P", commit)
            .stdout.decode()
            .strip()
            .split()
        )
        if (
            projection["commit_sha"] != commit
            or projection["html_url"]
            != f"https://github.com/{protocol['repository']}/commit/{commit}"
            or _aware_timestamp(
                projection["committer_date"], "historical GitHub committer_date"
            )
            != commit_timestamp(commit)
            or projection["parent_shas"] != local_parents
        ):
            raise AuditError("historical GitHub observation differs from local Git")

    github_workflows: list[dict[str, Any]] = []
    for observation in (
        prereg_workflow_observation,
        payload_workflow_observation,
        receipt_workflow_observation,
    ):
        evidence = fetch_github(str(observation["endpoint"]).lstrip("/"))
        projection = {
            "run_id": evidence.get("id"),
            "workflow_id": evidence.get("workflow_id"),
            "workflow_name": evidence.get("name"),
            "workflow_path": evidence.get("path"),
            "event": evidence.get("event"),
            "head_sha": evidence.get("head_sha"),
            "run_attempt": evidence.get("run_attempt"),
            "status": evidence.get("status"),
            "conclusion": evidence.get("conclusion"),
            "created_at": evidence.get("created_at"),
            "run_started_at": evidence.get("run_started_at"),
            "updated_at": evidence.get("updated_at"),
            "html_url": evidence.get("html_url"),
        }
        if projection != observation["canonical_projection"]:
            raise AuditError("GitHub workflow evidence changed after activation")
        base_endpoint = (
            f"repos/{protocol['repository']}/actions/runs"
            f"?event=pull_request&head_sha={observation['expected_head_sha']}"
            "&per_page=100"
        )
        all_runs: list[Mapping[str, Any]] = []
        total_count: int | None = None
        page = 1
        while True:
            endpoint = base_endpoint if page == 1 else f"{base_endpoint}&page={page}"
            page_value = fetch_github(endpoint)
            observed_total = page_value.get("total_count")
            page_runs = page_value.get("workflow_runs")
            if (
                isinstance(observed_total, bool)
                or not isinstance(observed_total, int)
                or observed_total < 0
                or not isinstance(page_runs, list)
                or any(not isinstance(item, Mapping) for item in page_runs)
            ):
                raise AuditError("GitHub workflow selection response changed")
            if total_count is None:
                total_count = observed_total
            elif observed_total != total_count:
                raise AuditError("GitHub workflow pagination total changed")
            all_runs.extend(page_runs)
            if len(all_runs) >= total_count:
                if len(all_runs) != total_count:
                    raise AuditError("GitHub workflow pagination overran total")
                break
            if len(page_runs) != 100:
                raise AuditError("GitHub workflow pagination ended early")
            page += 1
            if page > 1000:  # pragma: no cover - defensive API bound
                raise AuditError("GitHub workflow pagination is unbounded")
        matching = [
            item
            for item in all_runs
            if item.get("path") == ".github/workflows/tests.yml"
            and item.get("head_sha") == observation["expected_head_sha"]
            and item.get("event") == "pull_request"
            and item.get("run_attempt") == 1
            and isinstance(item.get("id"), int)
            and not isinstance(item.get("id"), bool)
        ]
        matching_ids = [int(item["id"]) for item in matching]
        if (
            not matching_ids
            or len(matching_ids) != len(set(matching_ids))
            or min(matching_ids) != observation["run_id"]
        ):
            raise AuditError("GitHub workflow first-run selection changed")
        github_workflows.append(dict(projection))

    base_url = "https://github.com/rokuroku-066/TSE-Session-Ranker/commit/"
    expected_urls = {
        "preregistration_commit_url": base_url + commit_fields["preregistration"],
        "payload_commit_url": base_url + commit_fields["payload"],
    }
    for field, expected in expected_urls.items():
        owner = payload if field == "preregistration_commit_url" else receipt
        if owner[field] != expected:
            raise AuditError(f"activation {field} is not the canonical GitHub URL")
    if decisions[0]["activation_receipt_commit_url"] != (
        base_url + commit_fields["receipt"]
    ):
        raise AuditError("activation receipt commit URL is not canonical")

    repository_slug = str(protocol["repository"])
    github_commits: dict[str, dict[str, Any]] = {}
    for commit in sorted(set(commit_fields.values())):
        endpoint = f"repos/{repository_slug}/commits/{commit}"
        evidence = fetch_github(endpoint)
        api_parents = sorted(
            str(item.get("sha")) for item in evidence.get("parents", [])
        )
        local_parents = sorted(_git_command(
            repository, "show", "-s", "--format=%P", commit
        ).stdout.decode().strip().split())
        try:
            api_committed_at = evidence["commit"]["committer"]["date"]
        except (KeyError, TypeError) as exc:
            raise AuditError("GitHub commit evidence lacks committed timestamp") from exc
        if (
            evidence.get("sha") != commit
            or evidence.get("html_url") != base_url + commit
            or api_parents != local_parents
            or _aware_timestamp(api_committed_at, "GitHub committed_at")
            != commit_timestamp(commit)
        ):
            raise AuditError("GitHub commit evidence differs from local Git object")
        github_commits[commit] = {
            "sha": commit,
            "html_url": evidence["html_url"],
            "committed_at": str(api_committed_at),
            "parents": api_parents,
        }

    direct_parent_pairs = (
        ("payload", "preregistration"),
        ("receipt", "payload"),
    )
    for child_label, parent_label in direct_parent_pairs:
        child = commit_fields[child_label]
        parent = commit_fields[parent_label]
        local_parents = (
            _git_command(repository, "show", "-s", "--format=%P", child)
            .stdout.decode()
            .strip()
            .split()
        )
        if local_parents != [parent] or github_commits[child]["parents"] != [parent]:
            raise AuditError(
                f"activation {child_label} commit is not the direct child of "
                f"{parent_label}"
            )

    ancestry_pairs = (
        ("preregistration", "preregistration_observed_tip", False),
        ("preregistration_observed_tip", "payload", True),
        ("payload", "payload_observed_tip", False),
        ("payload_observed_tip", "receipt", True),
        ("payload", "receipt", True),
        ("receipt", "receipt_observed_tip", False),
    )
    for ancestor, descendant, strict in ancestry_pairs:
        if (strict and commit_fields[ancestor] == commit_fields[descendant]) or _git_command(
            repository,
            "merge-base",
            "--is-ancestor",
            commit_fields[ancestor],
            commit_fields[descendant],
            check=False,
        ).returncode != 0:
            raise AuditError(
                f"activation Git ancestry is invalid: {ancestor}->{descendant}"
            )

    def github_is_ancestor(ancestor: str, descendant: str) -> dict[str, Any]:
        if ancestor == descendant:
            return {
                "ancestor": ancestor,
                "descendant": descendant,
                "status": "identical",
            }
        evidence = fetch_github(
            f"repos/{repository_slug}/compare/{ancestor}...{descendant}"
        )
        if (
            evidence.get("status") != "ahead"
            or evidence.get("base_commit", {}).get("sha") != ancestor
            or evidence.get("merge_base_commit", {}).get("sha") != ancestor
        ):
            raise AuditError("GitHub API ancestry differs from local Git ancestry")
        return {
            "ancestor": ancestor,
            "descendant": descendant,
            "status": "ahead",
        }

    github_ancestry = [
        github_is_ancestor(commit_fields[ancestor], commit_fields[descendant])
        for ancestor, descendant, _ in ancestry_pairs
    ]

    branch = str(protocol["branch"])
    tips: list[str] = []
    for reference in (f"refs/heads/{branch}", f"refs/remotes/origin/{branch}"):
        resolved = _git_command(
            repository, "rev-parse", "--verify", reference, check=False
        )
        if resolved.returncode == 0:
            tips.append(resolved.stdout.decode().strip())
    current_branch = _git_command(
        repository, "symbolic-ref", "--quiet", "--short", "HEAD", check=False
    )
    if current_branch.returncode == 0 and current_branch.stdout.decode().strip() == branch:
        tips.append(_git_command(repository, "rev-parse", "HEAD").stdout.decode().strip())
    if not tips or not any(
        _git_command(
            repository,
            "merge-base",
            "--is-ancestor",
            commit_fields["receipt"],
            tip,
            check=False,
        ).returncode
        == 0
        for tip in set(tips)
    ):
        raise AuditError("activation receipt commit is not reachable from branch tip")
    if not any(
        _git_command(
            repository,
            "merge-base",
            "--is-ancestor",
            commit_fields["receipt_observed_tip"],
            tip,
            check=False,
        ).returncode
        == 0
        for tip in set(tips)
    ):
        raise AuditError("receipt-observed tip is not on the registered branch")

    api_branch = fetch_github(
        f"repos/{repository_slug}/commits/{quote(branch, safe='')}"
    )
    github_branch_tip = str(api_branch.get("sha", ""))
    if (
        re.fullmatch(r"[0-9a-f]{40}", github_branch_tip) is None
        or github_branch_tip not in set(tips)
    ):
        raise AuditError("GitHub registered branch tip differs from local branch")
    if github_branch_tip not in github_commits:
        evidence = fetch_github(
            f"repos/{repository_slug}/commits/{github_branch_tip}"
        )
        api_parents = sorted(
            str(item.get("sha")) for item in evidence.get("parents", [])
        )
        local_parents = sorted(_git_command(
            repository, "show", "-s", "--format=%P", github_branch_tip
        ).stdout.decode().strip().split())
        try:
            api_committed_at = evidence["commit"]["committer"]["date"]
        except (KeyError, TypeError) as exc:
            raise AuditError("GitHub branch-tip evidence lacks committed timestamp") from exc
        if (
            evidence.get("sha") != github_branch_tip
            or evidence.get("html_url") != base_url + github_branch_tip
            or api_parents != local_parents
            or _aware_timestamp(api_committed_at, "GitHub branch-tip committed_at")
            != commit_timestamp(github_branch_tip)
        ):
            raise AuditError("GitHub branch-tip evidence differs from local Git object")
        github_commits[github_branch_tip] = {
            "sha": github_branch_tip,
            "html_url": evidence["html_url"],
            "committed_at": str(api_committed_at),
            "parents": api_parents,
        }
    for commit in sorted(set(commit_fields.values())):
        github_ancestry.append(github_is_ancestor(commit, github_branch_tip))

    def committed_bytes(commit: str, path: str) -> bytes:
        result = _git_command(repository, "show", f"{commit}:{path}", check=False)
        if result.returncode != 0:
            raise AuditError(f"activation commit lacks required path: {path}")
        return result.stdout

    prereg_paths = protocol["activation"]["preregistration_commit"][
        "required_paths"
    ]
    prereg_hash_by_path = {
        str(payload[path_field]): str(payload[hash_field])
        for path_field, hash_field in (
            ("protocol_path", "protocol_sha256"),
            ("hypothesis_path", "hypothesis_sha256"),
            ("runtime_lock_path", "runtime_lock_sha256"),
            ("runner_path", "runner_sha256"),
            ("audit_path", "audit_sha256"),
            ("tests_path", "tests_sha256"),
            ("session_calendar_path", "session_calendar_sha256"),
            ("workflow_path", "workflow_sha256"),
        )
    }
    if set(map(str, prereg_paths)) != set(prereg_hash_by_path):
        raise AuditError("preregistration path/hash registry is incomplete")
    for path in prereg_paths:
        prereg_bytes = committed_bytes(commit_fields["preregistration"], str(path))
        if (
            hashlib.sha256(prereg_bytes).hexdigest()
            != prereg_hash_by_path[str(path)]
            or committed_bytes(commit_fields["payload"], str(path)) != prereg_bytes
            or committed_bytes(commit_fields["receipt"], str(path)) != prereg_bytes
            or (repository / str(path)).read_bytes() != prereg_bytes
        ):
            raise AuditError(f"preregistered path changed after activation: {path}")

    payload_path = str(protocol["activation"]["payload"]["path"])
    receipt_path = str(protocol["activation"]["receipt"]["path"])
    for commit, path in (
        (commit_fields["preregistration"], payload_path),
        (commit_fields["preregistration"], receipt_path),
        (commit_fields["payload"], receipt_path),
    ):
        if _git_command(repository, "cat-file", "-e", f"{commit}:{path}", check=False).returncode == 0:
            raise AuditError("activation artifact existed before its authorised commit")
    def changed_paths(older: str, newer: str) -> set[str]:
        result = _git_command(
            repository,
            "diff",
            "--name-only",
            "--no-renames",
            older,
            newer,
        )
        return {
            line
            for line in result.stdout.decode("utf-8", errors="strict").splitlines()
            if line
        }

    payload_parents = (
        _git_command(
            repository,
            "show",
            "-s",
            "--format=%P",
            commit_fields["payload"],
        )
        .stdout.decode()
        .strip()
        .split()
    )
    receipt_parents = (
        _git_command(
            repository,
            "show",
            "-s",
            "--format=%P",
            commit_fields["receipt"],
        )
        .stdout.decode()
        .strip()
        .split()
    )
    if payload_parents != [commit_fields["preregistration"]]:
        raise AuditError("payload commit parent is not preregistration commit A")
    if receipt_parents != [commit_fields["payload"]]:
        raise AuditError("receipt commit parent is not payload commit B")

    if changed_paths(commit_fields["preregistration"], commit_fields["payload"]) != {
        payload_path
    }:
        raise AuditError("payload commit changed a path other than the payload")
    if changed_paths(commit_fields["payload"], commit_fields["receipt"]) != {
        receipt_path
    }:
        raise AuditError("receipt commit changed a path other than the receipt")
    payload_bytes = committed_bytes(commit_fields["payload"], payload_path)
    if (
        payload_bytes != Path(payload_file_path).read_bytes()
        or hashlib.sha256(payload_bytes).hexdigest() != receipt["payload_file_sha256"]
        or committed_bytes(commit_fields["receipt"], payload_path) != payload_bytes
    ):
        raise AuditError("payload commit does not contain the exact bound payload bytes")
    receipt_bytes = committed_bytes(commit_fields["receipt"], receipt_path)
    receipt_file_sha = hashlib.sha256(receipt_bytes).hexdigest()
    if (
        receipt_bytes != Path(receipt_file_path).read_bytes()
        or any(
            row["activation_receipt_file_sha256"] != receipt_file_sha
            for row in decisions
        )
    ):
        raise AuditError("receipt commit does not contain the exact receipt bytes")

    preregistration_commit_time = commit_timestamp(commit_fields["preregistration"])
    preregistration_claimed_time = _aware_timestamp(
        payload["preregistration_commit_committed_at"],
        "preregistration_commit_committed_at",
    )
    preregistration_observed_at = _aware_timestamp(
        payload["preregistration_commit_observed_at"],
        "preregistration_commit_observed_at",
    )
    payload_commit_time = commit_timestamp(commit_fields["payload"])
    preregistration_workflow_observed_at = _aware_timestamp(
        payload["preregistration_workflow_run_observed_at"],
        "preregistration_workflow_run_observed_at",
    )
    payload_observed_at = _aware_timestamp(
        receipt["payload_observed_at"], "payload_observed_at"
    )
    receipt_issued_at = _aware_timestamp(
        receipt["receipt_issued_at"], "receipt_issued_at"
    )
    if preregistration_commit_time != preregistration_claimed_time:
        raise AuditError("preregistration Git timestamp differs from payload evidence")
    payload_workflow_observed_at = _aware_timestamp(
        receipt["payload_workflow_run_observed_at"],
        "payload_workflow_run_observed_at",
    )
    if not (
        preregistration_commit_time <= preregistration_observed_at
        and max(preregistration_observed_at, preregistration_workflow_observed_at)
        <= payload_commit_time
        and payload_commit_time <= payload_observed_at
        and max(payload_observed_at, payload_workflow_observed_at)
        <= receipt_issued_at
    ):
        raise AuditError("activation preregistration/payload timestamps are not causal")
    if payload_commit_time != _aware_timestamp(
        receipt["payload_commit_committed_at"], "payload_commit_committed_at"
    ):
        raise AuditError("payload Git commit timestamp differs from receipt evidence")
    receipt_commit_time = commit_timestamp(commit_fields["receipt"])
    if any(
        _aware_timestamp(
            row["activation_receipt_commit_committed_at"],
            "activation_receipt_commit_committed_at",
        )
        != receipt_commit_time
        for row in decisions
    ):
        raise AuditError("receipt Git commit timestamp differs from decision evidence")
    if receipt_issued_at > receipt_commit_time or any(
        _aware_timestamp(
            row["activation_receipt_commit_observed_at"],
            "activation_receipt_commit_observed_at",
        )
        < receipt_commit_time
        for row in decisions
    ):
        raise AuditError("activation receipt timestamps are not causal")
    return {
        "preregistration_commit_sha": commit_fields["preregistration"],
        "preregistration_observed_tip_sha": commit_fields[
            "preregistration_observed_tip"
        ],
        "payload_commit_sha": commit_fields["payload"],
        "receipt_commit_sha": commit_fields["receipt"],
        "receipt_observed_tip_sha": commit_fields["receipt_observed_tip"],
        "branch_tip_commit_sha": sorted(set(tips))[-1],
        "github_branch_tip_sha": github_branch_tip,
        "github_current_evidence_sha256": canonical_json_sha256(
            {
                "commits": [github_commits[key] for key in sorted(github_commits)],
                "workflows": github_workflows,
                "ancestry": github_ancestry,
                "branch_tip": github_branch_tip,
            }
        ),
        "payload_file_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "receipt_file_sha256": receipt_file_sha,
    }


def validate_state_manifest(
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    first_counted_session: Any | None = None,
) -> dict[str, Any]:
    required = set(
        protocol["state_contract"]["target_month_state_manifest_required_fields"]
    )
    if set(manifest) != required:
        raise AuditError("state manifest fields changed")
    value = dict(manifest)
    target = pd.Period(str(value["target_month"]), freq="M")
    if str(target) != str(value["target_month"]):
        raise AuditError("state target month is not canonical")
    expected_months = [str(target - offset) for offset in (3, 2, 1)]
    if value["three_prior_calendar_months"] != expected_months:
        raise AuditError("state manifest did not use the immediate three months")
    counts = value["three_complete_pair_day_counts"]
    medians = value["three_month_medians_pct"]
    if not isinstance(counts, list) or len(counts) != 3:
        raise AuditError("state manifest pair counts changed")
    if not isinstance(medians, list) or len(medians) != 3:
        raise AuditError("state manifest month medians changed")
    counts = [_strict_nonnegative_int(item, "state complete-pair count") for item in counts]
    medians = [
        _finite_or_none(item, "monthly median") for item in medians
    ]
    for count, median in zip(counts, medians, strict=True):
        if (count >= MIN_COMPLETE_PAIRS_PER_MONTH) != (median is not None):
            raise AuditError("state manifest monthly availability is inconsistent")
    available = all(count >= MIN_COMPLETE_PAIRS_PER_MONTH for count in counts) and all(
        item is not None for item in medians
    )
    if _strict_bool(value["state_available"], "state_available") != available:
        raise AuditError("state availability does not recompute")
    expected_state = (
        float(np.median(np.asarray(medians, dtype=float))) if available else None
    )
    observed_state = _finite_or_none(value["state_value_pct"], "state_value_pct")
    if observed_state != expected_state:
        raise AuditError("state median-of-monthly-medians does not recompute")
    expected_rank = (
        1
        if expected_state is not None and expected_state > 0.0
        else 2
        if expected_state is not None and expected_state < 0.0
        else None
    )
    observed_rank = value["selected_source_rank"]
    observed_rank = (
        None
        if observed_rank is None
        else _strict_nonnegative_int(observed_rank, "selected_source_rank")
    )
    if observed_rank != expected_rank:
        raise AuditError("state selected rank does not recompute")
    fold_hash = value["c00_fold_manifest_sha256"]
    bundle_file_hash = value["fold_model_bundle_file_sha256"]
    if (fold_hash is None) != (bundle_file_hash is None):
        raise AuditError("state fold/bundle hashes are not a paired null")
    if fold_hash is not None:
        _require_nonzero_sha(fold_hash, "state c00_fold_manifest_sha256")
        _require_nonzero_sha(
            bundle_file_hash, "state fold_model_bundle_file_sha256"
        )
    for field in (
        "protocol_sha256",
        "activation_payload_sha256",
        "activation_receipt_sha256",
        "state_manifest_sha256",
    ):
        _require_nonzero_sha(value[field], f"state {field}")
    if value["protocol_sha256"] != PROTOCOL_SHA256:
        raise AuditError("state protocol hash changed")
    created = _aware_timestamp(value["created_at"], "state created_at")
    first_session = _month_first_counted_session(target, first_counted_session)
    if created > pd.Timestamp(f"{first_session.date()}T08:58:59+09:00"):
        raise AuditError("state manifest was sealed after the first month cutoff")
    expected_hash = canonical_json_sha256(
        value, exclude_fields={"state_manifest_sha256"}
    )
    if value["state_manifest_sha256"] != expected_hash:
        raise AuditError("state manifest self-hash mismatch")
    return value


def validate_completed_month_records(
    records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    required = tuple(
        protocol["state_contract"]["completed_month_record_required_fields"]
    )
    rows = validate_hash_chain(
        records,
        required_fields=required,
        key_field="completed_month",
    )
    previous_month: pd.Period | None = None
    for row in rows:
        month = pd.Period(str(row["completed_month"]), freq="M")
        if str(month) != str(row["completed_month"]):
            raise AuditError("completed-month id is not canonical")
        if previous_month is not None and month != previous_month + 1:
            raise AuditError("completed-month ledger is not consecutive")
        previous_month = month
        if row["schema_version"] != 1 or row["protocol_sha256"] != PROTOCOL_SHA256:
            raise AuditError("completed-month fixed identity changed")
        created = _aware_timestamp(row["created_at"], "completed month created_at")
        if created <= pd.Timestamp(
            f"{month.end_time.date()}T23:59:59+09:00"
        ):
            raise AuditError("completed month was sealed before calendar month-end")
        counted = _strict_nonnegative_int(
            row["counted_scheduled_sessions"], "counted scheduled sessions"
        )
        count = _strict_nonnegative_int(row["complete_pair_days"], "complete pairs")
        if count > counted:
            raise AuditError("completed-month complete pairs exceed counted sessions")
        available = count >= MIN_COMPLETE_PAIRS_PER_MONTH
        if _strict_bool(row["available"], "available") != available:
            raise AuditError("completed-month availability does not recompute")
        median = _finite_or_none(
            row["monthly_median_rank1_minus_rank2_pct"], "monthly median"
        )
        if available != (median is not None):
            raise AuditError("completed-month median availability differs")
        for field in (
            "ordered_complete_pair_session_sha256",
            "ordered_difference_values_sha256",
            "activation_payload_sha256",
            "activation_receipt_sha256",
        ):
            _require_nonzero_sha(row[field], f"completed month {field}")
    return rows


def _predictor_source_date(file_name: str, kind: str) -> pd.Timestamp:
    pattern = r"^(\d{6})\.pdf$" if kind == "price_warmup" else r"^stq_(\d{8})\.pdf$"
    match = re.fullmatch(pattern, file_name)
    if match is None:
        raise AuditError(f"predictor {kind} source filename is invalid: {file_name}")
    raw = match.group(1) + ("01" if kind == "price_warmup" else "")
    return pd.to_datetime(raw, format="%Y%m%d", errors="raise").normalize()


def expected_predictor_sources(latest_session: Any) -> list[dict[str, Any]]:
    """Independently rebuild the cumulative v1.7-plus-forward source registry."""

    if sha256_file(V05_PRICE_LOCK) != V05_PRICE_LOCK_SHA256:
        raise AuditError("bound v1.5 price input lock changed")
    if sha256_file(V04_PARSER_AUDIT) != V04_PARSER_AUDIT_SHA256:
        raise AuditError("bound legacy daily parser audit changed")
    if sha256_file(V17_REPLAY_INPUT_LOCK) != V17_REPLAY_INPUT_LOCK_SHA256:
        raise AuditError("bound v1.7 replay input lock changed")

    price_lock = read_json(V05_PRICE_LOCK)
    price_by_name = {
        str(item["filename"]): item for item in price_lock["jpx"]["sources"]
    }
    warmup_names = ["202505.pdf", "202506.pdf", "202507.pdf"]
    if not set(warmup_names).issubset(price_by_name):
        raise AuditError("bound price warmup registry is incomplete")
    warmup: list[dict[str, Any]] = []
    for name in warmup_names:
        item = price_by_name[name]
        warmup.append(
            {
                "file": name,
                "kind": "price_warmup",
                "url": str(item["source_url"]),
                "byte_count": int(item["bytes"]),
                "sha256": str(item["sha256"]),
            }
        )

    parser_audit = read_json(V04_PARSER_AUDIT)
    if (
        parser_audit.get("new_parser_version")
        != "jpx_daily_text_v6_special_quote_marker"
        or parser_audit.get("new_parser_sha256")
        != "1bd2e74acced608eb36c3606b593ea407d2d1e5f54b3283790ef8fd0fb1041f7"
        or int(parser_audit.get("pdf_count", -1)) != 160
        or int(parser_audit.get("new_rejected_rows", -1)) != 0
    ):
        raise AuditError("bound legacy parser/source registry changed")
    legacy_by_name = {
        str(item["name"]): item for item in parser_audit["per_file"]
    }
    if len(legacy_by_name) != 160:
        raise AuditError("bound legacy daily registry has duplicate files")

    replay_lock = read_json(V17_REPLAY_INPUT_LOCK)
    replay_by_name = {str(item["name"]): item for item in replay_lock["files"]}
    if (
        replay_lock.get("source_set_sha256")
        != "a0fdea19f9e7c771d2e0450720eb551917d9c8b9cfade75260d5cc43adb3a43f"
        or len(replay_by_name) != 79
    ):
        raise AuditError("bound v1.7 replay source registry changed")

    daily: dict[str, dict[str, Any]] = {}
    for name, item in legacy_by_name.items():
        _predictor_source_date(name, "daily")
        daily[name] = {
            "file": name,
            "kind": "daily",
            "url": None,
            "byte_count": None,
            "sha256": str(item["sha256"]),
        }
    for name, item in replay_by_name.items():
        _predictor_source_date(name, "daily")
        daily[name] = {
            "file": name,
            "kind": "daily",
            "url": str(item["source_url"]),
            "byte_count": int(item["bytes"]),
            "sha256": str(item["sha256"]),
        }

    latest = pd.Timestamp(latest_session).normalize()
    fixed_forward = pd.DatetimeIndex(
        pd.to_datetime(
            (
                "2026-07-28",
                "2026-07-29",
                "2026-07-30",
                "2026-07-31",
                "2026-08-03",
                "2026-08-04",
            )
        )
    )
    forward_dates = fixed_forward[fixed_forward <= latest]
    if latest >= pd.Timestamp("2026-08-05"):
        calendar = load_registered_calendar()
        forward_dates = forward_dates.union(calendar[calendar <= latest])
    for session in forward_dates:
        name = f"stq_{session:%Y%m%d}.pdf"
        daily.setdefault(
            name,
            {
                "file": name,
                "kind": "daily",
                "url": None,
                "byte_count": None,
                "sha256": None,
            },
        )
    ordered_daily = sorted(
        daily.values(),
        key=lambda item: (_predictor_source_date(item["file"], "daily"), item["file"]),
    )
    return [*warmup, *ordered_daily]


def validate_source_manifest(
    manifest: Mapping[str, Any],
    *,
    session_date: Any,
    protocol: Mapping[str, Any],
    predictor_raw_store_root: str | Path | None = None,
    external_identity_registry: dict[tuple[int, int], str] | None = None,
) -> dict[str, Any]:
    required = set(
        protocol["source_contract"]["forward_daily"][
            "source_manifest_required_fields"
        ]
    )
    if set(manifest) != required:
        raise AuditError("source manifest fields changed")
    value = dict(manifest)
    target = pd.Timestamp(session_date).normalize()
    if pd.Timestamp(value["target_session"]).normalize() != target:
        raise AuditError("source manifest target differs")
    latest = pd.Timestamp(value["latest_required_source_session"]).normalize()
    calendar = load_registered_calendar()
    if target == pd.Timestamp("2026-08-05"):
        expected_latest = pd.Timestamp("2026-08-04")
    else:
        positions = np.flatnonzero(calendar == target)
        if len(positions) != 1 or int(positions[0]) == 0:
            raise AuditError("source target is outside the predecessor calendar")
        expected_latest = calendar[int(positions[0]) - 1]
    if latest != expected_latest:
        raise AuditError(
            "source latest-required session is not the exact registered predecessor"
        )
    files = value["source_files"]
    urls = value["source_urls"]
    object_keys = value["source_object_keys"]
    byte_counts = value["source_byte_counts"]
    hashes = value["source_sha256"]
    vectors = (object_keys, files, urls, byte_counts, hashes)
    if not all(isinstance(item, list) for item in vectors):
        raise AuditError("source manifest source vectors are not arrays")
    if len({len(item) for item in vectors}) != 1:
        raise AuditError("source manifest vector lengths differ")
    expected_sources = expected_predictor_sources(latest)
    expected_by_file = {item["file"]: item for item in expected_sources}
    expected_files = [item["file"] for item in expected_sources]
    actual_files = [str(item) for item in files]
    if any(item not in expected_by_file for item in actual_files):
        raise AuditError("source manifest contains an unregistered predictor source")
    canonical_subset = [item for item in expected_files if item in set(actual_files)]
    if actual_files != canonical_subset:
        raise AuditError("source manifest source registry is not canonical")
    source_objects: list[dict[str, Any]] = []
    order_keys: list[tuple[int, str, str]] = []
    seen_files: set[str] = set()
    seen_keys: set[str] = set()
    for index, (object_key, source_file, url, byte_count, digest) in enumerate(
        zip(object_keys, files, urls, byte_counts, hashes, strict=True)
    ):
        if (
            not isinstance(object_key, str)
            or "\x00" in object_key
            or Path(object_key).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(object_key).parts)
            or not object_key.startswith("model_v18_shoulder_state/predictor/")
        ):
            raise AuditError("source manifest predictor object key is unsafe")
        parts = Path(object_key).parts
        expected_source = expected_by_file.get(source_file)
        if (
            len(parts) != 4
            or expected_source is None
            or parts[2] != expected_source["kind"]
            or parts[3] != source_file
        ):
            raise AuditError("source manifest predictor object key pattern changed")
        if (
            not isinstance(source_file, str)
            or Path(source_file).name != source_file
            or not source_file.lower().endswith(".pdf")
            or source_file in seen_files
            or object_key in seen_keys
        ):
            raise AuditError("source manifest source file/key is invalid or duplicated")
        if not isinstance(url, str) or not url.startswith("https://www.jpx.co.jp/"):
            raise AuditError("source manifest URL is not official JPX HTTPS")
        count = _strict_nonnegative_int(byte_count, f"source_byte_counts[{index}]")
        if count <= 0:
            raise AuditError("source manifest byte count is invalid")
        _require_nonzero_sha(digest, f"source_sha256[{index}]")
        if (
            expected_source["sha256"] is not None
            and digest != expected_source["sha256"]
        ):
            raise AuditError("source manifest historical predictor SHA changed")
        if (
            expected_source["byte_count"] is not None
            and count != expected_source["byte_count"]
        ):
            raise AuditError("source manifest historical predictor byte count changed")
        if expected_source["url"] is not None and url != expected_source["url"]:
            raise AuditError("source manifest historical predictor URL changed")
        seen_files.add(source_file)
        seen_keys.add(object_key)
        dates = re.findall(r"(?<!\d)(20\d{4}(?:\d{2})?)(?!\d)", source_file)
        source_date = dates[-1] if dates else ""
        order_keys.append(
            (0 if parts[2] == "price_warmup" else 1, source_date, source_file)
        )
        source_objects.append(
            {
                "object_key": object_key,
                "file": source_file,
                "url": url,
                "byte_count": count,
                "sha256": str(digest),
            }
        )
        if predictor_raw_store_root is not None:
            raw_bytes = _read_external_object_bytes(
                predictor_raw_store_root,
                object_key,
                required_prefix="model_v18_shoulder_state/predictor/",
                identity_registry=external_identity_registry,
            )
            if len(raw_bytes) != count or hashlib.sha256(raw_bytes).hexdigest() != digest:
                raise AuditError("predictor raw object bytes differ from source manifest")
    if order_keys != sorted(order_keys):
        raise AuditError("source manifest objects are not in canonical order")
    expected_set_hash = canonical_json_sha256(source_objects)
    if value["source_set_sha256"] != expected_set_hash:
        raise AuditError("source manifest ordered source-set hash mismatch")
    received = (
        None
        if value["source_received_at"] is None
        else _aware_timestamp(value["source_received_at"], "source_received_at")
    )
    cutoff = pd.Timestamp(f"{target.date()}T08:58:59+09:00")
    runtime_verified = _runtime_verification_timestamp(value, "source manifest")
    created = _aware_timestamp(value["created_at"], "source manifest created_at")
    sealed = _aware_timestamp(value["sealed_at"], "source manifest sealed_at")
    if (
        runtime_verified > created
        or (received is not None and received > created)
        or created > sealed
        or sealed > cutoff
    ):
        raise AuditError("source manifest timestamps violate the pre-open DAG")
    contract = protocol["source_contract"]["forward_daily"]
    if (
        value["schema_version"] != 1
        or _strict_nonnegative_int(value["parsed_row_count"], "parsed_row_count") < 0
        or _strict_nonnegative_int(value["rejected_row_count"], "rejected_row_count")
        != contract["required_rejected_rows"]
        or _strict_nonnegative_int(
            value["duplicate_date_code_count"], "duplicate_date_code_count"
        )
        != 0
        or value["parser_path"] != contract["parser_path"]
        or value["parser_version"] != contract["parser_version"]
        or value["parser_sha256"] != contract["parser_sha256"]
        or value["python_version"] != LOCKED_PYTHON_VERSION
        or value["canonical_json_contract"] != "project_canonical_json_v1"
    ):
        raise AuditError("source/parser manifest contract changed")
    complete = _strict_bool(value["source_complete"], "source_complete")
    failure_reason = value["failure_reason"]
    semantic_fields = (
        "parsed_panel_semantic_sha256",
        "g0_panel_semantic_sha256",
        "common_universe_semantic_sha256",
        "target_date_scoring_input_semantic_sha256",
    )
    if complete:
        if (
            actual_files != expected_files
            or received is None
            or failure_reason is not None
        ):
            raise AuditError("complete source manifest receipt/failure state is invalid")
        for field in semantic_fields:
            _require_nonzero_sha(value[field], f"source manifest {field}")
        if _strict_nonnegative_int(value["parsed_row_count"], "parsed_row_count") <= 0:
            raise AuditError("complete source manifest parsed no rows")
    else:
        if (
            not isinstance(failure_reason, str)
            or not failure_reason.strip()
            or int(value["parsed_row_count"]) != 0
            or any(value[field] is not None for field in semantic_fields)
            or bool(actual_files) != (received is not None)
        ):
            raise AuditError("incomplete source manifest is not valid fail-closed state")
    _require_nonzero_sha(value["source_manifest_sha256"], "source manifest hash")
    expected_hash = canonical_json_sha256(
        value, exclude_fields={"source_manifest_sha256"}
    )
    if value["source_manifest_sha256"] != expected_hash:
        raise AuditError("source manifest self-hash mismatch")
    return value


def _validate_numeric_array(
    value: Mapping[str, Any],
    *,
    name: str,
    dtype: str,
    shape: tuple[int, ...] | None = None,
    positive: bool = False,
) -> np.ndarray:
    """Validate one non-executable numeric component of a fold bundle."""

    if not isinstance(value, Mapping) or set(value) != {
        "dtype",
        "shape",
        "values",
        "sha256",
    }:
        raise AuditError(f"{name} numeric-array schema changed")
    if value["dtype"] != dtype:
        raise AuditError(f"{name} dtype changed")
    raw_shape = value["shape"]
    if (
        not isinstance(raw_shape, list)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in raw_shape
        )
    ):
        raise AuditError(f"{name} shape is invalid")
    observed_shape = tuple(int(item) for item in raw_shape)
    if shape is not None and observed_shape != shape:
        raise AuditError(f"{name} shape changed")
    raw_values = value["values"]
    if not isinstance(raw_values, list):
        raise AuditError(f"{name} values must be a row-major JSON array")
    if dtype == "int64":
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in raw_values
        ):
            raise AuditError(f"{name} contains non-integer values")
        array = np.asarray(raw_values, dtype="<i8")
    elif dtype == "float64":
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in raw_values
        ):
            raise AuditError(f"{name} contains nonnumeric values")
        array = np.asarray(raw_values, dtype="<f8")
        if not np.isfinite(array).all():
            raise AuditError(f"{name} contains non-finite values")
    else:  # pragma: no cover - every registered component is covered above
        raise AuditError(f"{name} has an unregistered dtype")
    if array.size != math.prod(observed_shape):
        raise AuditError(f"{name} values do not match shape")
    array = array.reshape(observed_shape)
    if positive and not (array > 0.0).all():
        raise AuditError(f"{name} must be strictly positive")
    expected_hash = canonical_json_sha256(value, exclude_fields={"sha256"})
    if value["sha256"] != expected_hash:
        raise AuditError(f"{name} component hash mismatch")
    return array


def validate_fold_model_bundle(
    bundle: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    runner_sha256: str | None = None,
    first_counted_session: Any | None = None,
) -> dict[str, Any]:
    """Validate all fitted pipeline state without importing sklearn or runner."""

    required = set(
        protocol["c00_contract"]["fold_model_bundle_contract"]["required_fields"]
    )
    if not isinstance(bundle, Mapping) or set(bundle) != required:
        raise AuditError("fold model bundle fields changed")
    value = dict(bundle)
    target = pd.Period(str(value["target_month"]), freq="M")
    if str(target) != str(value["target_month"]):
        raise AuditError("fold model bundle target month is not canonical")
    created = _aware_timestamp(value["created_at"], "fold model bundle created_at")
    runtime_verified = _runtime_verification_timestamp(value, "fold model bundle")
    first_session = _month_first_counted_session(target, first_counted_session)
    if runtime_verified > created or created >= pd.Timestamp(
        f"{first_session.date()}T08:58:59+09:00"
    ):
        raise AuditError("fold model bundle was sealed after first month cutoff")
    if (
        value["schema_version"] != 1
        or value["input_feature_order"] != list(C00_FEATURES)
        or value["input_feature_dtype"] != "float64"
        or value["input_feature_shape"] != [len(C00_FEATURES)]
        or value["transformed_feature_dtype"] != "float64"
        or value["imputer_strategy"] != "median"
        or value["imputer_add_indicator"] is not True
        or value["imputer_keep_empty_features"] is not False
        or value["scaler_with_mean"] is not True
        or value["scaler_with_std"] is not True
        or value["ridge_fit_intercept"] is not True
        or value["python_version"] != LOCKED_PYTHON_VERSION
        or value["numpy_version"] != LOCKED_NUMPY_VERSION
        or value["scikit_learn_version"] != LOCKED_SCIKIT_LEARN_VERSION
        or value["canonical_json_contract"] != "project_canonical_json_v1"
        or value["protocol_sha256"] != PROTOCOL_SHA256
    ):
        raise AuditError("fold model bundle fixed contract changed")
    if not isinstance(value["ridge_alpha"], (int, float)) or isinstance(
        value["ridge_alpha"], bool
    ) or float(value["ridge_alpha"]) != 1.0:
        raise AuditError("fold model Ridge alpha changed")
    if runner_sha256 is not None and value["runner_sha256"] != runner_sha256:
        raise AuditError("fold model bundle runner hash changed")
    _require_nonzero_sha(value["runner_sha256"], "fold bundle runner_sha256")

    statistics = _validate_numeric_array(
        value["imputer_statistics"],
        name="imputer_statistics",
        dtype="float64",
        shape=(len(C00_FEATURES),),
    )
    indicator_shape = value["imputer_indicator_features"].get("shape")
    if not isinstance(indicator_shape, list) or len(indicator_shape) != 1:
        raise AuditError("imputer indicator shape is invalid")
    indicator = _validate_numeric_array(
        value["imputer_indicator_features"],
        name="imputer_indicator_features",
        dtype="int64",
        shape=(int(indicator_shape[0]),),
    )
    if (
        np.any(indicator < 0)
        or np.any(indicator >= len(C00_FEATURES))
        or (len(indicator) > 1 and np.any(np.diff(indicator) <= 0))
    ):
        raise AuditError("imputer indicator indexes are invalid")
    width = len(C00_FEATURES) + len(indicator)
    expected_order = [
        *C00_FEATURES,
        *(f"missingindicator::{C00_FEATURES[int(index)]}" for index in indicator),
    ]
    if (
        value["transformed_feature_order"] != expected_order
        or value["transformed_feature_shape"] != [width]
    ):
        raise AuditError("fold model transformed feature order/shape changed")
    _validate_numeric_array(
        value["scaler_mean"],
        name="scaler_mean",
        dtype="float64",
        shape=(width,),
    )
    _validate_numeric_array(
        value["scaler_scale"],
        name="scaler_scale",
        dtype="float64",
        shape=(width,),
        positive=True,
    )
    _validate_numeric_array(
        value["ridge_coef"],
        name="ridge_coef",
        dtype="float64",
        shape=(width,),
    )
    if (
        isinstance(value["ridge_intercept"], bool)
        or not isinstance(value["ridge_intercept"], (int, float))
        or not math.isfinite(float(value["ridge_intercept"]))
        or not np.isfinite(statistics).all()
    ):
        raise AuditError("fold model bundle contains non-finite fitted state")
    expected_hash = canonical_json_sha256(
        value, exclude_fields={"fold_model_bundle_sha256"}
    )
    if value["fold_model_bundle_sha256"] != expected_hash:
        raise AuditError("fold model bundle self-hash mismatch")
    return value


def reconstruct_c00_scores(
    bundle: Mapping[str, Any],
    features: pd.DataFrame | np.ndarray,
    protocol: Mapping[str, Any],
) -> np.ndarray:
    """Clean-room SimpleImputer -> StandardScaler -> Ridge reconstruction."""

    value = validate_fold_model_bundle(bundle, protocol)
    if isinstance(features, pd.DataFrame):
        if features.columns.tolist() != list(C00_FEATURES):
            raise AuditError("score reconstruction feature order changed")
        raw = features.to_numpy(dtype="<f8")
    else:
        raw = np.asarray(features, dtype="<f8")
    if raw.ndim != 2 or raw.shape[1] != len(C00_FEATURES):
        raise AuditError("score reconstruction matrix has wrong shape")
    if np.isinf(raw).any():
        raise AuditError("score reconstruction input contains infinity")
    statistics = np.asarray(
        value["imputer_statistics"]["values"], dtype="<f8"
    )
    imputed = raw.copy()
    missing = np.isnan(imputed)
    if missing.any():
        imputed[missing] = np.broadcast_to(statistics, imputed.shape)[missing]
    indicator_indexes = np.asarray(
        value["imputer_indicator_features"]["values"], dtype="<i8"
    )
    indicators = np.isnan(raw[:, indicator_indexes]).astype("<f8")
    transformed = (
        np.column_stack([imputed, indicators]) if indicator_indexes.size else imputed
    )
    mean = np.asarray(value["scaler_mean"]["values"], dtype="<f8")
    scale = np.asarray(value["scaler_scale"]["values"], dtype="<f8")
    coef = np.asarray(value["ridge_coef"]["values"], dtype="<f8")
    try:
        from threadpoolctl import threadpool_limits
    except Exception as exc:  # pragma: no cover - runtime-lock integrity
        raise AuditError("threadpoolctl is unavailable for score replay") from exc
    with threadpool_limits(limits=1):
        return ((transformed - mean) / scale) @ coef + float(
            value["ridge_intercept"]
        )


def _frame_csv_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    value = frame.loc[:, list(columns)].copy()
    for column in value:
        if pd.api.types.is_datetime64_any_dtype(value[column]):
            value[column] = value[column].dt.strftime("%Y-%m-%d")
    return hashlib.sha256(
        value.to_csv(index=False, lineterminator="\n", na_rep="<NA>").encode()
    ).hexdigest()


def validate_clean_room_fold_fit(
    panel: pd.DataFrame,
    target_month: str,
    manifest: Mapping[str, Any],
    bundle: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    """Refit a fold from raw-derived rows and compare every fitted component."""

    month = pd.Period(target_month, freq="M")
    training = panel.loc[
        panel["date"].between(
            pd.Timestamp("2025-09-01"), month.start_time - pd.Timedelta(days=1)
        )
        & panel["common_training_eligible"].fillna(False).astype(bool)
    ].copy()
    if training.empty or training[["date", "code"]].duplicated().any():
        raise AuditError("clean-room monthly training frame is empty/duplicated")
    training["_daily_rank_target"] = (
        training["oc_return_pct"]
        .astype(float)
        .groupby(training["date"], sort=False)
        .rank(method="average", pct=True)
        .mul(2.0)
        .sub(1.0)
    )
    identity = training.loc[:, ["date", "code"]].sort_values(
        ["date", "code"], kind="stable"
    )
    expected_hashes = {
        "training_first_session": str(training["date"].min().date()),
        "training_last_session": str(training["date"].max().date()),
        "training_session_count": int(training["date"].nunique()),
        "training_row_identity_sha256": _frame_csv_sha256(identity, ("date", "code")),
        "training_target_sha256": _numeric_values_sha256(
            training["_daily_rank_target"].to_numpy(dtype=float)
        ),
        "feature_matrix_sha256": _numeric_values_sha256(
            training.loc[:, list(C00_FEATURES)].to_numpy(dtype=float)
        ),
    }
    for field, expected in expected_hashes.items():
        if manifest[field] != expected:
            raise AuditError(f"clean-room fold {field} differs")

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - environment integrity
        raise AuditError("sklearn is unavailable for clean-room refit") from exc
    model = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=False
                ),
            ),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=1.0, fit_intercept=True)),
        ]
    )
    weights = 1.0 / training.groupby("date", sort=False)["date"].transform("size")
    try:
        from threadpoolctl import threadpool_limits
    except Exception as exc:  # pragma: no cover - runtime-lock integrity
        raise AuditError("threadpoolctl is unavailable for clean-room refit") from exc
    with threadpool_limits(limits=1):
        model.fit(
            training.loc[:, list(C00_FEATURES)],
            training["_daily_rank_target"],
            ridge__sample_weight=weights,
        )
    imputer = model.named_steps["impute"]
    scaler = model.named_steps["scale"]
    ridge = model.named_steps["ridge"]
    observed_arrays = {
        "imputer_statistics": np.asarray(imputer.statistics_, dtype=float),
        "imputer_indicator_features": np.asarray(
            imputer.indicator_.features_, dtype=np.int64
        ),
        "scaler_mean": np.asarray(scaler.mean_, dtype=float),
        "scaler_scale": np.asarray(scaler.scale_, dtype=float),
        "ridge_coef": np.asarray(ridge.coef_, dtype=float),
    }
    for field, observed in observed_arrays.items():
        dtype = "int64" if field == "imputer_indicator_features" else "float64"
        sealed = np.asarray(bundle[field]["values"], dtype=dtype)
        if observed.shape != sealed.shape or not np.allclose(
            observed, sealed, rtol=0.0, atol=1e-12
        ):
            raise AuditError(f"clean-room fitted {field} differs from sealed bundle")
    if not math.isclose(
        float(np.asarray(ridge.intercept_).reshape(-1)[0]),
        float(bundle["ridge_intercept"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AuditError("clean-room fitted Ridge intercept differs")


def validate_fold_manifest(
    manifest: Mapping[str, Any],
    bundle: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    bundle_path: str | Path | None = None,
    runner_sha256: str | None = None,
    first_counted_session: Any | None = None,
) -> dict[str, Any]:
    """Independently bind a fold manifest to exact bundle numbers and bytes."""

    required = set(protocol["c00_contract"]["fold_manifest_required_fields"])
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise AuditError("fold manifest fields changed")
    value = dict(manifest)
    validated_bundle = validate_fold_model_bundle(
        bundle,
        protocol,
        runner_sha256=runner_sha256,
        first_counted_session=first_counted_session,
    )
    target = pd.Period(str(value["target_month"]), freq="M")
    if str(target) != str(value["target_month"]):
        raise AuditError("fold target month is not canonical")
    if validated_bundle["target_month"] != str(target):
        raise AuditError("fold manifest/bundle target month differs")
    expected_path = f"research/model_v18_shoulder_state_fold_models/{target}.json"
    if value["fold_model_bundle_path"] != expected_path:
        raise AuditError("fold model bundle path changed")
    started = _aware_timestamp(value["fit_started_at"], "fold fit_started_at")
    completed = _aware_timestamp(value["fit_completed_at"], "fold fit_completed_at")
    sealed = _aware_timestamp(value["sealed_at"], "fold sealed_at")
    runtime_verified = _runtime_verification_timestamp(value, "fold manifest")
    bundle_runtime_verified = _runtime_verification_timestamp(
        validated_bundle, "fold model bundle"
    )
    bundle_created = _aware_timestamp(
        validated_bundle["created_at"], "fold model bundle created_at"
    )
    first_session = _month_first_counted_session(target, first_counted_session)
    first_cutoff = pd.Timestamp(f"{first_session.date()}T08:58:59+09:00")
    if not (
        runtime_verified <= started <= completed <= bundle_created <= sealed < first_cutoff
    ) or bundle_runtime_verified > started:
        raise AuditError("fold fit was not complete before first month cutoff")
    if validated_bundle["runtime_lock_sha256"] != value["runtime_lock_sha256"]:
        raise AuditError("fold manifest/bundle runtime-lock binding differs")
    training_first = pd.Timestamp(value["training_first_session"]).normalize()
    training_last = pd.Timestamp(value["training_last_session"]).normalize()
    if (
        pd.isna(training_first)
        or pd.isna(training_last)
        or training_first > training_last
        or training_last >= target.start_time
        or _strict_nonnegative_int(
            value["training_session_count"], "training_session_count"
        )
        < 1
    ):
        raise AuditError("fold training period is invalid or leaks target month")
    if (
        value["schema_version"] != 1
        or value["feature_names"] != list(C00_FEATURES)
        or float(value["ridge_alpha"]) != 1.0
        or value["protocol_sha256"] != PROTOCOL_SHA256
        or value["python_version"] != LOCKED_PYTHON_VERSION
        or value["numpy_version"] != LOCKED_NUMPY_VERSION
        or value["pandas_version"] != LOCKED_PANDAS_VERSION
        or value["scikit_learn_version"] != LOCKED_SCIKIT_LEARN_VERSION
        or value["canonical_json_contract"] != "project_canonical_json_v1"
    ):
        raise AuditError("fold manifest fixed contract changed")
    if runner_sha256 is not None and value["runner_sha256"] != runner_sha256:
        raise AuditError("fold manifest runner hash changed")
    for field in (
        "training_row_identity_sha256",
        "training_target_sha256",
        "feature_matrix_sha256",
        "universe_contract_sha256",
        "runner_sha256",
        "fold_manifest_sha256",
    ):
        _require_nonzero_sha(value[field], f"fold manifest {field}")
    expected_bundle_fields = {
        "fold_model_bundle_schema_version": 1,
        "fold_model_bundle_sha256": validated_bundle["fold_model_bundle_sha256"],
        "input_feature_order_sha256": canonical_json_sha256(list(C00_FEATURES)),
        "transformed_feature_order_sha256": canonical_json_sha256(
            validated_bundle["transformed_feature_order"]
        ),
        "imputer_statistics_sha256": validated_bundle["imputer_statistics"]["sha256"],
        "imputer_indicator_features_sha256": validated_bundle[
            "imputer_indicator_features"
        ]["sha256"],
        "scaler_mean_sha256": validated_bundle["scaler_mean"]["sha256"],
        "scaler_scale_sha256": validated_bundle["scaler_scale"]["sha256"],
        "ridge_coef_sha256": validated_bundle["ridge_coef"]["sha256"],
    }
    for field, expected in expected_bundle_fields.items():
        if value[field] != expected:
            raise AuditError(f"fold manifest bundle binding changed: {field}")
    if not math.isclose(
        float(value["ridge_intercept"]),
        float(validated_bundle["ridge_intercept"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise AuditError("fold manifest Ridge intercept changed")
    if bundle_path is not None:
        observed_path = Path(bundle_path)
        if _stable_plain_file_sha256(
            observed_path, label="fold model bundle"
        ) != value["fold_model_bundle_file_sha256"]:
            raise AuditError("fold model bundle exact file SHA mismatch")
    else:
        _require_nonzero_sha(
            value["fold_model_bundle_file_sha256"],
            "fold manifest fold_model_bundle_file_sha256",
        )
    expected_hash = canonical_json_sha256(
        value, exclude_fields={"fold_manifest_sha256"}
    )
    if value["fold_manifest_sha256"] != expected_hash:
        raise AuditError("fold manifest self-hash mismatch")
    return value


def _parse_jsonl_bytes(raw_bytes: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditError(f"{label} is not UTF-8") from exc
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            raise AuditError(f"blank JSONL line {line_number}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AuditError(f"invalid JSONL line {line_number} in {label}") from exc
        if not isinstance(value, dict):
            raise AuditError(f"JSONL line {line_number} must be an object")
        records.append(value)
    return records


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    return _parse_jsonl_bytes(
        _stable_plain_file_bytes(target, label=f"JSONL artifact {target}"),
        label=str(target),
    )


def _read_csv_stable(
    path: str | Path,
    *,
    label: str,
    **kwargs: Any,
) -> tuple[pd.DataFrame, bytes]:
    raw = _stable_plain_file_bytes(path, label=label)
    try:
        frame = pd.read_csv(io.BytesIO(raw), **kwargs)
    except Exception as exc:
        raise AuditError(f"{label} is not a valid CSV artifact") from exc
    return frame, raw


def semantic_decision_hash(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash outcome-free decisions independent of JSONL whitespace.

    Chain-envelope fields are retained: changing sequence or predecessor is a
    semantic ledger change.  Any outcome field fails closed rather than being
    silently omitted.
    """

    if any(OUTCOME_FIELDS & set(record) for record in records):
        raise AuditError("decision ledger contains outcome fields")
    ordered = sorted(
        (dict(record) for record in records),
        key=lambda item: (int(item["sequence_number"]), str(item["session_date"])),
    )
    return hashlib.sha256(
        b"".join(canonical_json_bytes(record) + b"\n" for record in ordered)
    ).hexdigest()


def semantic_score_hash(scores: pd.DataFrame) -> str:
    forbidden = OUTCOME_FIELDS & set(scores)
    if forbidden:
        raise AuditError(f"score output contains outcomes: {sorted(forbidden)}")
    if scores.columns.tolist() != list(SCORE_FIELDS):
        raise AuditError(
            "score output fields changed: "
            f"observed={list(scores)}, expected={list(SCORE_FIELDS)}"
        )
    frame = scores.loc[:, list(SCORE_FIELDS)].copy()
    for column in ("session_date", "feature_source_max_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if frame[["session_date", "feature_source_max_date"]].isna().any(axis=None):
        raise AuditError("score output contains invalid dates")
    if frame["feature_source_max_date"].ge(frame["session_date"]).any():
        raise AuditError("score output is not D-1/prior-only")
    generated = [
        _aware_timestamp(item, "score_generated_at")
        for item in frame["score_generated_at"]
    ]
    runtime_verified = [
        _aware_timestamp(item, "score runtime_lock_verified_at")
        for item in frame["runtime_lock_verified_at"]
    ]
    if (
        not frame["runtime_lock_sha256"].eq(RUNTIME_LOCK_SHA256).all()
        or any(verified > created for verified, created in zip(runtime_verified, generated, strict=True))
    ):
        raise AuditError("score runtime binding/timestamps changed")
    for field in ("source_manifest_sha256", "c00_fold_manifest_sha256"):
        for item in frame[field]:
            _require_nonzero_sha(item, f"score {field}")
    frame["source_rank"] = pd.to_numeric(frame["source_rank"], errors="coerce")
    frame["model_score"] = pd.to_numeric(frame["model_score"], errors="coerce")
    if (
        frame[["source_rank", "model_score"]].isna().any(axis=None)
        or not frame["source_rank"].isin((1, 2)).all()
        or not np.isfinite(frame["model_score"].to_numpy(dtype=float)).all()
    ):
        raise AuditError("score output contains invalid rank/score")
    if frame[["session_date", "source_rank"]].duplicated().any():
        raise AuditError("score output contains duplicate date/rank")
    counts = frame.groupby("session_date", sort=False)["source_rank"].agg(list)
    if any(sorted(int(item) for item in values) != [1, 2] for values in counts):
        raise AuditError("score output does not contain exact rank pairs")
    for _, pair in frame.groupby("session_date", sort=False):
        for field in (
            "score_generated_at",
            "runtime_lock_sha256",
            "runtime_lock_verified_at",
            "source_manifest_sha256",
            "c00_fold_manifest_sha256",
            "feature_source_max_date",
        ):
            if pair[field].nunique(dropna=False) != 1:
                raise AuditError(f"score pair differs in {field}")
    frame = frame.sort_values(["session_date", "source_rank"], kind="stable")
    frame["session_date"] = frame["session_date"].dt.strftime("%Y-%m-%d")
    frame["feature_source_max_date"] = frame[
        "feature_source_max_date"
    ].dt.strftime("%Y-%m-%d")
    return hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def validate_pit_causality(
    decisions: Sequence[Mapping[str, Any]],
    scores: pd.DataFrame,
    *,
    source_manifests: Mapping[str, Mapping[str, Any]],
    state_manifests: Mapping[str, Mapping[str, Any]],
    fold_manifests: Mapping[str, Mapping[str, Any]],
    activation_ready_at: Any,
) -> None:
    """Verify every seal timestamp follows all inputs and precedes cutoff."""

    activation_ready = _aware_timestamp(activation_ready_at, "activation_ready_at")
    score_frame = scores.copy()
    score_frame["_session"] = pd.to_datetime(
        score_frame["session_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    if score_frame["_session"].isna().any():
        raise AuditError("score causality has an invalid session")
    score_groups = {
        str(session): group for session, group in score_frame.groupby("_session")
    }
    for decision in decisions:
        session = decision["session_date"]
        cutoff = _aware_timestamp(decision["decision_cutoff"], "decision_cutoff")
        computed = _aware_timestamp(decision["computed_at"], "computed_at")
        decision_runtime = _runtime_verification_timestamp(decision, "decision")
        source = source_manifests.get(decision["source_manifest_sha256"])
        state = state_manifests.get(session[:7])
        if source is None or state is None:
            raise AuditError("decision causality lacks source/state manifest")
        source_runtime = _runtime_verification_timestamp(source, "source manifest")
        source_created = _aware_timestamp(
            source["created_at"], "source manifest created_at"
        )
        source_sealed = _aware_timestamp(
            source["sealed_at"], "source manifest sealed_at"
        )
        state_created = _aware_timestamp(state["created_at"], "state created_at")
        received = source["source_received_at"]
        source_received = (
            None
            if received is None
            else _aware_timestamp(received, "source_received_at")
        )
        if not (
            activation_ready <= source_runtime <= source_created <= source_sealed
            and activation_ready <= state_created
            and source_sealed <= decision_runtime <= computed <= cutoff
        ) or (source_received is not None and source_received > source_created):
            raise AuditError("source/state/decision timestamps violate the PIT DAG")
        dependencies = [activation_ready, source_sealed, state_created, decision_runtime]
        fold_hash = decision["c00_fold_manifest_sha256"]
        fold = None if fold_hash is None else fold_manifests.get(fold_hash)
        if fold_hash is not None and fold is None:
            raise AuditError("decision causality lacks its fold manifest")
        if fold is not None:
            fold_runtime = _runtime_verification_timestamp(fold, "fold manifest")
            fit_started = _aware_timestamp(
                fold["fit_started_at"], "fold fit_started_at"
            )
            fit_completed = _aware_timestamp(
                fold["fit_completed_at"], "fold fit_completed_at"
            )
            fold_sealed = _aware_timestamp(fold["sealed_at"], "fold sealed_at")
            if not (
                activation_ready
                <= fold_runtime
                <= fit_started
                <= fit_completed
                <= fold_sealed
                <= decision_runtime
            ):
                raise AuditError("fold timestamps violate the PIT DAG")
            dependencies.extend((fit_completed, fold_sealed))
        group = score_groups.get(session)
        if decision["model_complete"]:
            if group is None or len(group) != 2:
                raise AuditError("model-complete decision lacks causal score pair")
            if (
                not group["runtime_lock_sha256"].eq(RUNTIME_LOCK_SHA256).all()
                or not group["source_manifest_sha256"]
                .eq(decision["source_manifest_sha256"])
                .all()
                or not group["c00_fold_manifest_sha256"].eq(fold_hash).all()
            ):
                raise AuditError("score pair does not bind decision runtime/source/fold")
            generated_values = {
                _aware_timestamp(value, "score_generated_at")
                for value in group["score_generated_at"]
            }
            score_runtime_values = {
                _aware_timestamp(value, "score runtime_lock_verified_at")
                for value in group["runtime_lock_verified_at"]
            }
            if len(generated_values) != 1 or len(score_runtime_values) != 1:
                raise AuditError("score pair generation/runtime timestamps differ")
            generated = next(iter(generated_values))
            score_runtime = next(iter(score_runtime_values))
            if (
                score_runtime != decision_runtime
                or score_runtime < source_sealed
                or generated < max(*dependencies, score_runtime)
                or generated > cutoff
            ):
                raise AuditError("score was generated before an input or after cutoff")
            dependencies.append(generated)
        elif group is not None:
            raise AuditError("model-incomplete decision has a score pair")
        if computed < max(dependencies) or computed > cutoff:
            raise AuditError("decision predates an input/score or follows cutoff")
    if set(score_groups) != {
        row["session_date"] for row in decisions if row["model_complete"]
    }:
        raise AuditError("score causality includes an uncounted session")


def semantic_rows_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    date_columns: Sequence[str] = ("date", "feature_source_max_date"),
) -> str:
    """Project-canonical semantic row hash used by predictor manifests."""

    if frame.columns.duplicated().any() or not set(columns).issubset(frame):
        raise AuditError("semantic frame does not contain the registered view")
    canonical = frame.loc[:, list(columns)].copy()
    if not {"date", "code"}.issubset(canonical):
        raise AuditError("semantic frame must have date/code identity")
    canonical["date"] = pd.to_datetime(canonical["date"], errors="coerce")
    if canonical["date"].isna().any():
        raise AuditError("semantic frame contains invalid dates")
    canonical["code"] = canonical["code"].astype("string")
    if canonical["code"].isna().any() or canonical[["date", "code"]].duplicated().any():
        raise AuditError("semantic frame contains invalid/duplicate date-code keys")
    canonical = canonical.sort_values(["date", "code"], kind="stable")
    rows: list[dict[str, Any]] = []
    date_fields = set(date_columns)
    for raw in canonical.to_dict(orient="records"):
        row: dict[str, Any] = {}
        for column in columns:
            value = raw[column]
            if value is None or pd.isna(value):
                row[column] = None
            elif column in date_fields:
                parsed = pd.Timestamp(value)
                if parsed.tzinfo is not None:
                    parsed = parsed.tz_convert("Asia/Tokyo").tz_localize(None)
                row[column] = str(parsed.normalize().date())
            elif isinstance(value, (bool, np.bool_)):
                row[column] = bool(value)
            elif isinstance(value, (int, np.integer)):
                row[column] = int(value)
            elif isinstance(value, (float, np.floating)):
                number = float(value)
                if not math.isfinite(number):
                    raise AuditError("semantic frame contains non-finite numeric values")
                row[column] = number
            elif isinstance(value, str):
                row[column] = value
            else:
                raise AuditError(
                    f"semantic frame has unsupported {column} value type"
                )
        rows.append(row)
    return canonical_json_sha256(rows)


def _rolling_by_code(
    values: pd.Series,
    codes: pd.Series,
    *,
    window: int,
    operation: str,
) -> pd.Series:
    roller = values.groupby(codes, sort=False).rolling(window, min_periods=window)
    if operation == "median":
        result = roller.median()
    elif operation == "sum":
        result = roller.sum()
    elif operation == "min":
        result = roller.min()
    elif operation == "max":
        result = roller.max()
    elif operation == "q25":
        result = roller.quantile(0.25)
    elif operation == "q75":
        result = roller.quantile(0.75)
    else:  # pragma: no cover - all registered operations are enumerated
        raise AuditError(f"unregistered rolling operation: {operation}")
    return result.reset_index(level=0, drop=True).reindex(values.index)


def _cross_section_rank(
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


def _add_clean_room_bounded_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Independent copy of the frozen price-continuity/G0 rank transform."""

    frame = panel.copy()
    discontinuity = pd.to_numeric(frame["overnight"], errors="coerce").abs().gt(30.0)
    prior_discontinuity = discontinuity.groupby(frame["code"], sort=False).shift(1)
    recent_discontinuity = (
        prior_discontinuity.groupby(frame["code"], sort=False)
        .rolling(60, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .sort_index()
        .fillna(0.0)
        .astype(bool)
    )
    frame["price_history_continuous_60"] = ~recent_discontinuity
    frame["eligible"] &= frame["price_history_continuous_60"]
    frame["training_eligible"] &= frame["price_history_continuous_60"]
    group = frame.groupby("code", sort=False)
    effective_close = group["close"].ffill()
    prior_effective_close = effective_close.groupby(frame["code"], sort=False).shift(1)
    for window in (5, 20, 60):
        older = prior_effective_close.groupby(frame["code"], sort=False).shift(window)
        frame[f"close_momentum_{window}"] = 100.0 * (
            prior_effective_close / older - 1.0
        )
    prior_high20 = (
        frame["high"]
        .groupby(frame["code"], sort=False)
        .shift(1)
        .groupby(frame["code"], sort=False)
        .rolling(20, min_periods=10)
        .max()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    prior_low20 = (
        frame["low"]
        .groupby(frame["code"], sort=False)
        .shift(1)
        .groupby(frame["code"], sort=False)
        .rolling(20, min_periods=10)
        .min()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    span = (prior_high20 - prior_low20).where(prior_high20.gt(prior_low20))
    frame["prior_close_location_20"] = (
        (prior_effective_close - prior_low20) / span
    ).clip(0.0, 1.0)
    return frame


def _clean_room_liquidity_ready(
    prices: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    required = {
        "date",
        "code",
        "close",
        "volume",
        "turnover",
        "vwap",
        "trading_unit",
        "source_format",
    }
    if not required.issubset(prices):
        raise AuditError("predictor prices lack exact-liquidity inputs")
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
        raise AuditError("predictor liquidity source contains duplicate date/code rows")
    position = pd.Series(np.arange(len(sessions)), index=sessions)
    daily["_session_position"] = daily["date"].map(position)
    if daily["_session_position"].isna().any():
        raise AuditError("predictor liquidity row is outside reconstructed calendar")
    codes = daily["code"]
    numeric_columns = ("close", "volume", "turnover", "vwap", "trading_unit")
    for column in numeric_columns:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    positive = daily[list(numeric_columns)].gt(0.0).all(axis=1)
    oldest = daily["_session_position"].groupby(codes, sort=False).shift(19)
    consecutive20 = (daily["_session_position"] - oldest).eq(19)
    positive20 = _rolling_by_code(
        positive.astype(float), codes, window=20, operation="sum"
    ).eq(20.0)
    unit_min = _rolling_by_code(
        daily["trading_unit"], codes, window=20, operation="min"
    )
    unit_max = _rolling_by_code(
        daily["trading_unit"], codes, window=20, operation="max"
    )
    daily["liq_history_complete20"] = (
        consecutive20 & positive20 & unit_min.eq(unit_max) & unit_min.gt(0.0)
    )
    daily["liq_turnover_med20"] = _rolling_by_code(
        daily["turnover"], codes, window=20, operation="median"
    )
    daily["liq_volume_med20"] = _rolling_by_code(
        daily["volume"], codes, window=20, operation="median"
    )
    previous_turnover = daily["turnover"].groupby(codes, sort=False).shift(1)
    previous_med19 = _rolling_by_code(
        previous_turnover, codes, window=19, operation="median"
    )
    daily["liq_turnover_shock1"] = np.log(
        daily["turnover"] / previous_med19.where(previous_med19.gt(0.0))
    ).clip(-3.0, 3.0)
    log_turnover = np.log1p(daily["turnover"])
    daily["liq_turnover_log_iqr20"] = _rolling_by_code(
        log_turnover, codes, window=20, operation="q75"
    ) - _rolling_by_code(log_turnover, codes, window=20, operation="q25")
    daily["liq_lot_fraction20"] = (
        daily["trading_unit"] * daily["close"]
    ) / daily["liq_turnover_med20"].where(daily["liq_turnover_med20"].gt(0.0))
    close_vwap = daily["close"] / daily["vwap"].where(daily["vwap"].gt(0.0)) - 1.0
    daily["liq_close_vwap_dev1"] = close_vwap.clip(-0.20, 0.20)
    daily["liq_close_vwap_abs_med20"] = _rolling_by_code(
        close_vwap.abs(), codes, window=20, operation="median"
    )
    next_session = dict(zip(sessions[:-1], sessions[1:], strict=True))
    daily["date"] = daily["date"].map(next_session)
    daily = daily.dropna(subset=["date"])
    finite = daily[list(LIQUIDITY_RAW_FEATURES)].notna().all(axis=1)
    daily["liq_history_complete20"] &= finite
    return daily.loc[:, ["date", "code", "liq_history_complete20"]]


def build_clean_room_g0_panel(
    parsed_prices: pd.DataFrame,
    target_session: Any,
) -> pd.DataFrame:
    """Rebuild the bound C00 G0 panel without importing a research runner."""

    try:
        from tse_session_ranker.config import RankerConfig
        from tse_session_ranker.data.common import prepare_modeling_prices
        from tse_session_ranker.features import build_feature_panel, eligibility_mask
    except Exception as exc:  # pragma: no cover - installation integrity
        raise AuditError("frozen low-level feature primitives are unavailable") from exc
    target = pd.Timestamp(target_session).normalize()
    prices = parsed_prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    if prices["date"].isna().any() or prices["date"].ge(target).any():
        raise AuditError("predictor raw prices are not strictly before target")
    latest = prices["date"].max()
    latest_rows = prices.loc[prices["date"].eq(latest), ["code", "name"]].copy()
    if latest_rows.empty or latest_rows["code"].duplicated().any():
        raise AuditError("exact D-1 predictor universe is empty or duplicated")
    placeholder = latest_rows.assign(date=target)
    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
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
    sessions = pd.DatetimeIndex(
        sorted({*prices["date"].tolist(), target})
    )
    settings = RankerConfig()
    modeling, _ = prepare_modeling_prices(
        augmented,
        coverage_lookback=settings.source_coverage_lookback,
        minimum_source_coverage=settings.minimum_source_coverage,
        expected_sessions=sessions,
    )
    panel = _add_clean_room_bounded_features(build_feature_panel(modeling, settings))
    prior_rank_universe = (
        eligibility_mask(panel, settings.universe, for_training=True)
        & panel["price_history_continuous_60"].fillna(False).astype(bool)
        & panel["prior_universe_member"].fillna(False).astype(bool)
        & panel["universe_source_complete"].fillna(False).astype(bool)
    )
    for source in PRICE_RANK_SOURCES:
        panel[f"xrank_{source}"] = _cross_section_rank(
            panel[source], panel["date"], prior_rank_universe
        ).astype("float32")
    liquidity = _clean_room_liquidity_ready(prices, sessions)
    panel = panel.merge(
        liquidity,
        on=["date", "code"],
        how="left",
        validate="one_to_one",
        sort=False,
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
    target_rows = panel.loc[panel["date"].eq(target)]
    if target_rows.empty or target_rows["oc_return_pct"].notna().any():
        raise AuditError("clean-room target rows are missing or expose outcomes")
    return panel.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def _parse_history(history: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame):
        raise AuditError("pair history must be a DataFrame")
    frame = history.copy()
    if "date" not in frame and "session_date" in frame:
        frame = frame.rename(columns={"session_date": "date"})
    if "date" not in frame:
        raise AuditError("pair history is missing date")
    if {"rank1_oc_return_pct", "rank2_oc_return_pct"}.issubset(frame):
        frame = frame.loc[
            :, ["date", "rank1_oc_return_pct", "rank2_oc_return_pct"]
        ].melt(
            id_vars=("date",),
            value_vars=("rank1_oc_return_pct", "rank2_oc_return_pct"),
            var_name="_rank_name",
            value_name="oc_return_pct",
        )
        frame["source_rank"] = frame["_rank_name"].map(
            {"rank1_oc_return_pct": 1, "rank2_oc_return_pct": 2}
        )
        frame = frame.drop(columns="_rank_name")
    required = {"date", "source_rank", "oc_return_pct"}
    if not required.issubset(frame):
        raise AuditError(f"pair history is missing columns: {sorted(required - set(frame))}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    if frame["date"].isna().any():
        raise AuditError("pair history contains an invalid date")
    if getattr(frame["date"].dt, "tz", None) is not None:
        frame["date"] = frame["date"].dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    frame["date"] = frame["date"].dt.normalize()
    frame["source_rank"] = pd.to_numeric(frame["source_rank"], errors="coerce")
    if frame["source_rank"].isna().any() or not frame["source_rank"].isin((1, 2)).all():
        raise AuditError("pair history source_rank must be one or two")
    frame["source_rank"] = frame["source_rank"].astype(int)
    if frame[["date", "source_rank"]].duplicated().any():
        raise AuditError("pair history contains duplicate date/rank rows")
    frame["oc_return_pct"] = pd.to_numeric(frame["oc_return_pct"], errors="coerce")
    nonfinite = frame["oc_return_pct"].notna() & ~np.isfinite(
        frame["oc_return_pct"].to_numpy(dtype=float)
    )
    if nonfinite.any():
        raise AuditError("pair history contains a non-finite return")
    projection = ["date", "source_rank", "oc_return_pct"]
    for column in ("code", "name", "decision_payload_sha256", "origin"):
        if column in frame:
            projection.append(column)
    return frame.loc[:, projection].sort_values(
        ["date", "source_rank"], kind="stable"
    ).reset_index(drop=True)


def pair_history_semantic_hash(history: pd.DataFrame) -> str:
    frame = _parse_history(history)
    canonical = frame.loc[:, ["date", "source_rank", "oc_return_pct"]].copy()
    canonical["date"] = canonical["date"].dt.strftime("%Y-%m-%d")
    return hashlib.sha256(
        canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def derive_month_state(
    history: pd.DataFrame,
    target_month: str | pd.Period,
) -> dict[str, Any]:
    """Independently derive the frozen SH01 state for one target month."""

    frame = _parse_history(history)
    try:
        target = pd.Period(str(target_month), freq="M")
    except (TypeError, ValueError) as exc:
        raise AuditError("target_month must be YYYY-MM") from exc
    if str(target) != str(target_month):
        raise AuditError("target_month must be canonical YYYY-MM")

    frame = frame.loc[frame["date"].lt(target.start_time)].copy()
    wide = frame.pivot(index="date", columns="source_rank", values="oc_return_pct")
    for rank in (1, 2):
        if rank not in wide:
            wide[rank] = np.nan
    wide = wide.loc[:, [1, 2]].sort_index()
    wide["difference_pct"] = (wide[1] - wide[2]).where(
        wide[1].notna() & wide[2].notna()
    )
    source_months = list(pd.period_range(target - STATE_MONTHS, target - 1, freq="M"))
    summaries: list[dict[str, Any]] = []
    medians: list[float] = []
    for month in source_months:
        values = wide.loc[
            wide.index.to_period("M") == month, "difference_pct"
        ].dropna()
        count = int(len(values))
        qualified = count >= MIN_COMPLETE_PAIRS_PER_MONTH
        median = float(values.median()) if qualified else None
        summaries.append(
            {
                "month": str(month),
                "complete_pairs": count,
                "qualified": qualified,
                "median_rank1_minus_rank2_pct": median,
            }
        )
        if qualified:
            assert median is not None
            medians.append(median)

    if len(medians) != STATE_MONTHS:
        state_value: float | None = None
        decision = "cash"
        selected_rank: int | None = None
        reason = "insufficient_consecutive_month_history"
    else:
        state_value = float(np.median(np.asarray(medians, dtype=float)))
        if state_value > 0.0:
            decision, selected_rank, reason = (
                "rank1",
                1,
                "positive_three_month_median",
            )
        elif state_value < 0.0:
            decision, selected_rank, reason = (
                "rank2",
                2,
                "negative_three_month_median",
            )
        else:
            decision, selected_rank, reason = (
                "cash",
                None,
                "exact_zero_three_month_median",
            )
    source_hash_frame = frame.loc[
        frame["date"].dt.to_period("M").isin(source_months)
    ]
    return {
        "target_month": str(target),
        "required_source_months": [str(month) for month in source_months],
        "source_months": summaries,
        "state_value_pct": state_value,
        "decision": decision,
        "selected_source_rank": selected_rank,
        "reason": reason,
        "history_cutoff_date": str((target.start_time - pd.Timedelta(days=1)).date()),
        "state_history_sha256": pair_history_semantic_hash(source_hash_frame),
        "production_model_changed": False,
        "orders_allowed": False,
    }


def recompute_registered_seed(
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source = protocol["source_contract"]["historical_seed"]
    path = ROOT / str(source["only_authority"])
    if sha256_file(path) != source["sha256"]:
        raise AuditError("historical seed SHA-256 mismatch")
    picks = pd.read_csv(path, dtype={"code": str}, float_precision="round_trip")
    required = {"candidate_id", "date", "oc_return_pct", "code"}
    if not required.issubset(picks):
        raise AuditError("historical seed picks schema changed")
    rows: list[pd.DataFrame] = []
    for candidate_id, rank in ((C00_TOP1, 1), (C02_TOP2, 2)):
        current = picks.loc[
            picks["candidate_id"].eq(candidate_id),
            ["date", "code", "oc_return_pct"],
        ].copy()
        current["source_rank"] = rank
        rows.append(current)
    history = _parse_history(pd.concat(rows, ignore_index=True))
    wide = history.pivot(index="date", columns="source_rank", values="oc_return_pct")
    complete = wide[1].notna() & wide[2].notna()
    difference = (wide[1] - wide[2]).where(complete)
    output: list[dict[str, Any]] = []
    for month_text in source["allowed_completed_months"]:
        month = pd.Period(month_text, freq="M")
        values = difference.loc[
            difference.index.to_period("M") == month
        ].dropna()
        output.append(
            {
                "completed_month": month_text,
                "complete_pair_days": int(len(values)),
                "monthly_median_rank1_minus_rank2_pct": float(values.median()),
            }
        )
    expected = protocol["state_contract"]["initial_seed_months"]
    if len(output) != len(expected):
        raise AuditError("registered May-July seed does not recompute")
    for observed, registered in zip(output, expected, strict=True):
        if (
            observed["completed_month"] != registered["completed_month"]
            or observed["complete_pair_days"] != registered["complete_pair_days"]
            or not math.isclose(
                registered["monthly_median_rank1_minus_rank2_pct"],
                observed["monthly_median_rank1_minus_rank2_pct"],
                rel_tol=0.0,
                abs_tol=NUMERIC_TOLERANCE,
            )
        ):
            raise AuditError("registered May-July seed does not recompute")
    state = float(
        np.median(
            [item["monthly_median_rank1_minus_rank2_pct"] for item in output]
        )
    )
    if not math.isclose(
        protocol["state_contract"]["initial_august_2026_state_value_pct"],
        state,
        rel_tol=0.0,
        abs_tol=NUMERIC_TOLERANCE,
    ):
        raise AuditError("registered August state does not recompute")
    expected_rank = 1 if state > 0.0 else 2 if state < 0.0 else None
    if expected_rank != protocol["state_contract"][
        "initial_august_2026_selected_source_rank"
    ]:
        raise AuditError("registered August rank does not recompute")
    return output


def _strict_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise AuditError(f"{name} must be a JSON boolean")


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise AuditError(f"{name} must be a JSON integer")
    parsed = int(value)
    if parsed < 0:
        raise AuditError(f"{name} must be nonnegative")
    return parsed


def _require_nonzero_sha(value: Any, name: str) -> str:
    text = str(value)
    if SHA256_RE.fullmatch(text) is None or text == ZERO_SHA256:
        raise AuditError(f"{name} must be a nonzero SHA-256")
    return text


def _finite_or_none(value: Any, name: str) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{name} must be finite or null") from exc
    if not math.isfinite(parsed):
        raise AuditError(f"{name} must be finite or null")
    return parsed


def _canonical_float_equal(observed: Any, expected: float) -> bool:
    """Compare a sealed JSON number to the exact canonical recomputed float."""

    if (
        isinstance(observed, (bool, np.bool_))
        or not isinstance(observed, (int, float, np.integer, np.floating))
    ):
        return False
    observed_float = float(observed)
    expected_float = float(expected)
    if not math.isfinite(observed_float) or not math.isfinite(expected_float):
        return False
    return canonical_json_bytes(observed) == canonical_json_bytes(expected_float)


def _first_registered_session_in_month(month: pd.Period) -> pd.Timestamp:
    calendar = load_registered_calendar()
    selected = calendar[calendar.to_period("M") == month]
    if not len(selected):
        raise AuditError(f"registered calendar has no session in {month}")
    return selected[0]


def _month_first_counted_session(
    month: pd.Period,
    first_counted_session: Any | None,
) -> pd.Timestamp:
    """Use the activation-derived first count for a partial initial month."""

    if first_counted_session is not None:
        first = pd.Timestamp(first_counted_session).normalize()
        if first.to_period("M") == month:
            return first
    return _first_registered_session_in_month(month)


def validate_hash_chain(
    records: Sequence[Mapping[str, Any]],
    *,
    required_fields: Sequence[str],
    key_field: str = "session_date",
) -> list[dict[str, Any]]:
    """Validate an exact zero-based project-canonical JSONL chain."""

    expected_fields = set(required_fields)
    previous = ZERO_SHA256
    seen_keys: set[str] = set()
    previous_key: str | None = None
    normalised: list[dict[str, Any]] = []
    for expected_sequence, raw in enumerate(records):
        record = dict(raw)
        if set(record) != expected_fields:
            raise AuditError(
                "ledger record fields changed: "
                f"observed={sorted(record)}, expected={sorted(expected_fields)}"
            )
        if int(record["sequence_number"]) != expected_sequence:
            raise AuditError("ledger sequence is not zero-based contiguous")
        if record["previous_record_sha256"] != previous:
            raise AuditError("ledger previous-record hash changed")
        key = str(record[key_field])
        if key_field == "session_date":
            canonical_key = str(pd.Timestamp(key).date())
        elif key_field == "completed_month":
            canonical_key = str(pd.Period(key, freq="M"))
        else:
            canonical_key = key
        if key != canonical_key:
            raise AuditError(f"ledger {key_field} is not canonical")
        if key in seen_keys:
            raise AuditError(f"ledger contains duplicate {key_field}")
        if previous_key is not None and key <= previous_key:
            raise AuditError(f"ledger {key_field} is not strictly ascending")
        seen_keys.add(key)
        previous_key = key
        expected_record_hash = canonical_json_sha256(
            record, exclude_fields={"record_sha256"}
        )
        if record["record_sha256"] != expected_record_hash:
            raise AuditError("ledger record SHA-256 mismatch")
        if SHA256_RE.fullmatch(str(record["record_sha256"])) is None:
            raise AuditError("ledger record SHA-256 is invalid")
        previous = expected_record_hash
        normalised.append(record)
    return normalised


CHECKPOINT_CORE_MAGIC = b"TSEV18CP"
CHECKPOINT_CORE_ENVELOPE_BYTES = 16_384


def _checkpoint_core_from_row(
    row: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    fields = protocol["daily_preopen_checkpoint_contract"]["sealed_core_store"][
        "decision_core_required_fields"
    ]
    return {field: row[field] for field in fields}


def validate_checkpoint_decision_core(
    core: Mapping[str, Any],
    *,
    session_date: Any,
    resolution: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the outcome-free payload committed by one daily checkpoint."""

    contract = protocol["daily_preopen_checkpoint_contract"]
    required = contract["sealed_core_store"]["decision_core_required_fields"]
    value = dict(core)
    if set(value) != set(required):
        raise AuditError("checkpoint decision core fields differ from protocol")
    target = pd.Timestamp(session_date).normalize()
    expected_cutoff = pd.Timestamp(f"{target.date()}T08:58:59+09:00")
    if (
        value["candidate_id"] != SH01
        or _aware_timestamp(value["decision_cutoff"], "checkpoint decision cutoff")
        != expected_cutoff
    ):
        raise AuditError("checkpoint decision core identity/cutoff changed")
    if resolution not in set(contract["checkpoint_resolution_values"]):
        raise AuditError("checkpoint resolution is not registered")
    runtime_verified = _aware_timestamp(
        value["runtime_lock_verified_at"], "checkpoint runtime_lock_verified_at"
    )
    computed = _aware_timestamp(value["computed_at"], "checkpoint computed_at")
    if computed < runtime_verified or computed > expected_cutoff:
        raise AuditError("checkpoint decision core timestamp is invalid")
    for field in ("source_manifest_sha256", "state_manifest_sha256"):
        _require_nonzero_sha(value[field], f"checkpoint {field}")
    fold_pair = (
        value["c00_fold_manifest_sha256"],
        value["fold_model_bundle_file_sha256"],
    )
    if (fold_pair[0] is None) != (fold_pair[1] is None):
        raise AuditError("checkpoint fold/bundle nullability differs")
    if fold_pair[0] is not None:
        _require_nonzero_sha(fold_pair[0], "checkpoint fold manifest")
        _require_nonzero_sha(fold_pair[1], "checkpoint fold bundle")
    allowed_reasons = set(
        protocol["daily_decision_failure_reason_contract"]["allowed_nonnull_values"]
    )
    if value["failure_reason"] is not None and value["failure_reason"] not in allowed_reasons:
        raise AuditError("checkpoint core failure reason is not registered")
    return value


def parse_checkpoint_core_envelope(payload: bytes) -> dict[str, Any]:
    """Parse and canonically re-encode the fixed 16 KiB sealed core envelope."""

    if len(payload) != CHECKPOINT_CORE_ENVELOPE_BYTES:
        raise AuditError("checkpoint core envelope length changed")
    if payload[:8] != CHECKPOINT_CORE_MAGIC or payload[8] != 1:
        raise AuditError("checkpoint core envelope magic/version changed")
    length = int.from_bytes(payload[9:13], byteorder="big", signed=False)
    if length <= 0 or 13 + length > len(payload):
        raise AuditError("checkpoint core envelope JSON length is invalid")
    body = payload[13 : 13 + length]
    if any(payload[13 + length :]):
        raise AuditError("checkpoint core envelope padding is not all zero")
    value = _parse_json_object_bytes(body, label="checkpoint core envelope")
    if body != canonical_json_file_bytes(value):
        raise AuditError("checkpoint core envelope JSON is not canonical")
    return value


def validate_checkpoint_core_object(
    value: Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    contract = protocol["daily_preopen_checkpoint_contract"]["sealed_core_store"]
    current = dict(value)
    if set(current) != set(contract["required_fields"]):
        raise AuditError("checkpoint sealed-core fields differ from protocol")
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
        raise AuditError("checkpoint sealed-core identity changed")
    core = validate_checkpoint_decision_core(
        current["decision_core"],
        session_date=proposal["target_session"],
        # Both role envelopes hide the same primary decision core.  The
        # historical safety_cash role name is evidence metadata, never a
        # selectable resolution or a cash rewrite.
        resolution="primary",
        protocol=protocol,
    )
    if _aware_timestamp(
        core["computed_at"], "checkpoint core computed_at"
    ) > _aware_timestamp(proposal["created_at"], "checkpoint proposal created_at"):
        raise AuditError("checkpoint proposal predates its sealed decision core")
    core_hash = canonical_json_sha256(core)
    if current["decision_core_sha256"] != core_hash:
        raise AuditError("checkpoint decision-core hash changed")
    return core, core_hash


def validate_checkpoint_proposal(
    proposal: Mapping[str, Any],
    *,
    decision: Mapping[str, Any],
    role: str,
    protocol: Mapping[str, Any],
    runner_sha256: str,
) -> dict[str, Any]:
    contract = protocol["daily_preopen_checkpoint_contract"]
    value = dict(proposal)
    if set(value) != set(contract["proposal_required_fields"]):
        raise AuditError("checkpoint proposal fields differ from protocol")
    session = str(decision["session_date"])
    expected_key = f"model_v18_shoulder_state/checkpoint-core/{session}/{role}.bin"
    fixed = {
        "schema_version": 1,
        "checkpoint_id": (
            f"model_v18_shoulder_state_checkpoint_"
            f"{session.replace('-', '')}_{role}"
        ),
        "checkpoint_batch_id": (
            "model_v18_shoulder_state_checkpoint_batch_"
            f"{session.replace('-', '')}"
        ),
        "publication_ordinal": 0 if role == "safety_cash" else 1,
        "repository": protocol["repository"],
        "branch": protocol["branch"],
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "runner_sha256": runner_sha256,
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "activation_payload_sha256": decision["activation_payload_sha256"],
        "activation_receipt_sha256": decision["activation_receipt_sha256"],
        "activation_receipt_commit_sha": decision["activation_receipt_commit_sha"],
        "target_session": session,
        "checkpoint_role": role,
        "decision_sequence_number": int(decision["sequence_number"]),
        "previous_decision_record_sha256": decision["previous_record_sha256"],
        "sealed_core_object_key": expected_key,
        "canonical_json_contract": "project_canonical_json_v1",
    }
    if any(value[field] != expected for field, expected in fixed.items()):
        raise AuditError("checkpoint proposal fixed authority changed")
    _aware_timestamp(value["created_at"], "checkpoint proposal created_at")
    if (
        isinstance(value["sealed_core_byte_count"], bool)
        or value["sealed_core_byte_count"] != CHECKPOINT_CORE_ENVELOPE_BYTES
    ):
        raise AuditError("checkpoint sealed-core byte count changed")
    _require_nonzero_sha(value["sealed_core_sha256"], "checkpoint core file SHA")
    expected_hash = canonical_json_sha256(
        value, exclude_fields={"proposal_sha256"}
    )
    if value["proposal_sha256"] != expected_hash:
        raise AuditError("checkpoint proposal self-hash changed")
    return value


def _checkpoint_api_commit_projection(
    value: Mapping[str, Any],
    *,
    repository: str,
    expected_sha: str | None = None,
) -> dict[str, Any]:
    try:
        sha = str(value["sha"])
        html_url = str(value["html_url"])
        committed_at = str(value["commit"]["committer"]["date"])
        tree_sha = str(value["commit"]["tree"]["sha"])
        message = str(value["commit"]["message"])
        raw_parents = value["parents"]
    except (KeyError, TypeError) as exc:
        raise AuditError("checkpoint GitHub commit response is incomplete") from exc
    if (
        re.fullmatch(r"[0-9a-f]{40}", sha) is None
        or (expected_sha is not None and sha != expected_sha)
        or html_url != f"https://github.com/{repository}/commit/{sha}"
        or re.fullmatch(r"[0-9a-f]{40}", tree_sha) is None
        or not isinstance(raw_parents, list)
        or any(not isinstance(item, Mapping) for item in raw_parents)
    ):
        raise AuditError("checkpoint GitHub commit identity changed")
    parents = [str(item.get("sha")) for item in raw_parents]
    if any(re.fullmatch(r"[0-9a-f]{40}", item) is None for item in parents):
        raise AuditError("checkpoint GitHub commit parents are invalid")
    _aware_timestamp(committed_at, "checkpoint GitHub commit timestamp")
    return {
        "commit_sha": sha,
        "html_url": html_url,
        "committer_date": committed_at,
        "parent_shas": parents,
        "tree_sha": tree_sha,
        "message": message,
    }


def _checkpoint_git_ref_projection(
    value: Any,
    *,
    repository: str,
    branch: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not isinstance(value.get("object"), Mapping):
        raise AuditError("checkpoint Git ref response is incomplete")
    subject = value["object"]
    sha = str(subject.get("sha"))
    expected_ref = f"refs/heads/{branch}"
    expected_object_url = f"https://api.github.com/repos/{repository}/git/commits/{sha}"
    if (
        value.get("ref") != expected_ref
        or subject.get("type") != "commit"
        or re.fullmatch(r"[0-9a-f]{40}", sha) is None
        or subject.get("url") != expected_object_url
    ):
        raise AuditError("checkpoint Git ref projection changed")
    return {
        "ref": expected_ref,
        "object_type": "commit",
        "object_sha": sha,
        "object_url": expected_object_url,
    }


def _checkpoint_git_commit_projection(
    value: Any,
    *,
    repository: str,
    expected_sha: str,
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("tree"), Mapping)
        or not isinstance(value.get("parents"), list)
        or any(not isinstance(item, Mapping) for item in value.get("parents", []))
        or not isinstance(value.get("committer"), Mapping)
    ):
        raise AuditError("checkpoint Git commit response is incomplete")
    sha = str(value.get("sha"))
    tree_sha = str(value["tree"].get("sha"))
    parents = [str(item.get("sha")) for item in value["parents"]]
    html_url = str(value.get("html_url"))
    committed_at = str(value["committer"].get("date"))
    if (
        sha != expected_sha
        or re.fullmatch(r"[0-9a-f]{40}", sha) is None
        or re.fullmatch(r"[0-9a-f]{40}", tree_sha) is None
        or any(re.fullmatch(r"[0-9a-f]{40}", item) is None for item in parents)
        or html_url != f"https://github.com/{repository}/commit/{sha}"
    ):
        raise AuditError("checkpoint Git commit projection changed")
    _aware_timestamp(committed_at, "checkpoint Git commit committer_date")
    return {
        "sha": sha,
        "message": str(value.get("message")),
        "tree_sha": tree_sha,
        "parent_shas": parents,
        "html_url": html_url,
        "committer_date": committed_at,
    }


def _checkpoint_git_tree_projection(
    value: Any,
    *,
    expected_sha: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if (
        not isinstance(value, Mapping)
        or value.get("sha") != expected_sha
        or value.get("truncated") is not False
        or not isinstance(value.get("tree"), list)
        or any(not isinstance(item, Mapping) for item in value.get("tree", []))
    ):
        raise AuditError("checkpoint recursive Git tree response changed")
    entries: list[dict[str, Any]] = []
    by_path: dict[str, dict[str, Any]] = {}
    for raw in value["tree"]:
        path = raw.get("path")
        mode = raw.get("mode")
        object_type = raw.get("type")
        sha = raw.get("sha")
        size = raw.get("size") if "size" in raw else None
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or "\\" in path
            or "\x00" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or not isinstance(mode, str)
            or re.fullmatch(r"[0-7]{6}", mode) is None
            or object_type not in {"blob", "tree", "commit"}
            or not isinstance(sha, str)
            or re.fullmatch(r"[0-9a-f]{40}", sha) is None
            or (
                size is not None
                and (
                    isinstance(size, bool)
                    or not isinstance(size, int)
                    or size < 0
                )
            )
            or path in by_path
        ):
            raise AuditError("checkpoint recursive Git tree entry changed")
        entry = {
            "path": path,
            "mode": mode,
            "type": object_type,
            "sha": sha,
            "size": size,
        }
        entries.append(entry)
        by_path[path] = entry
    if entries != sorted(entries, key=lambda item: item["path"]):
        raise AuditError("checkpoint recursive Git tree is not path-sorted")
    projection = {"sha": expected_sha, "truncated": False, "entries": entries}
    return projection, by_path


def _checkpoint_git_blob_projection(
    value: Any,
    *,
    expected_sha: str,
) -> tuple[dict[str, Any], bytes]:
    if not isinstance(value, Mapping):
        raise AuditError("checkpoint Git blob response is incomplete")
    content = value.get("content")
    size = value.get("size")
    if (
        value.get("sha") != expected_sha
        or value.get("encoding") != "base64"
        or not isinstance(content, str)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or any(character.isspace() and character != "\n" for character in content)
    ):
        raise AuditError("checkpoint Git blob projection changed")
    canonical_content = content.replace("\n", "")
    try:
        payload = base64.b64decode(canonical_content.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise AuditError("checkpoint Git blob base64 changed") from exc
    object_sha = hashlib.sha1(
        b"blob " + str(len(payload)).encode("ascii") + b"\x00" + payload
    ).hexdigest()
    if len(payload) != size or object_sha != expected_sha:
        raise AuditError("checkpoint Git blob bytes changed")
    return {
        "sha": expected_sha,
        "encoding": "base64",
        "size": size,
        "content": canonical_content,
    }, payload


def _checkpoint_compare_projection(
    value: Any,
    *,
    expected_parent: str,
    expected_child: str,
    expected_path: str,
    expected_blob_sha: str,
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("commits"), list)
        or not isinstance(value.get("files"), list)
        or any(not isinstance(item, Mapping) for item in value.get("commits", []))
        or any(not isinstance(item, Mapping) for item in value.get("files", []))
        or not isinstance(value.get("base_commit"), Mapping)
        or not isinstance(value.get("merge_base_commit"), Mapping)
    ):
        raise AuditError("checkpoint compare response is incomplete")
    commit_shas = [str(item.get("sha")) for item in value["commits"]]
    files = [
        {
            "filename": item.get("filename"),
            "status": item.get("status"),
            "sha": item.get("sha"),
            "previous_filename": item.get("previous_filename"),
        }
        for item in value["files"]
    ]
    if (
        value.get("status") != "ahead"
        or value.get("ahead_by") != 1
        or value.get("behind_by") != 0
        or value.get("total_commits") != 1
        or value["base_commit"].get("sha") != expected_parent
        or value["merge_base_commit"].get("sha") != expected_parent
        or commit_shas != [expected_child]
        or files
        != [
            {
                "filename": expected_path,
                "status": "added",
                "sha": expected_blob_sha,
                "previous_filename": None,
            }
        ]
    ):
        raise AuditError("checkpoint compare does not prove one exact addition")
    return {
        "status": "ahead",
        "ahead_by": 1,
        "behind_by": 0,
        "total_commits": 1,
        "commit_shas": commit_shas,
        "files": files,
    }


def _checkpoint_path_history(
    *,
    repository: str,
    branch: str,
    path: str,
    fetch_github: Callable[[str], Any],
) -> list[str]:
    encoded_branch = quote(branch, safe="")
    encoded_path = quote(path, safe="")
    observed: list[str] = []
    page = 1
    saw_short_page = False
    while True:
        response = fetch_github(
            f"repos/{repository}/commits?sha={encoded_branch}&path={encoded_path}"
            f"&per_page=100&page={page}"
        )
        if not isinstance(response, list) or any(
            not isinstance(item, Mapping) for item in response
        ):
            raise AuditError("checkpoint protected-path history page changed")
        if not response:
            break
        if saw_short_page or len(response) > 100:
            raise AuditError("checkpoint protected-path history pagination changed")
        page_shas = [str(item.get("sha")) for item in response]
        if any(re.fullmatch(r"[0-9a-f]{40}", item) is None for item in page_shas):
            raise AuditError("checkpoint protected-path history SHA changed")
        observed.extend(page_shas)
        if len(response) < 100:
            saw_short_page = True
        page += 1
        if page > 1000:  # pragma: no cover - defensive API bound
            raise AuditError("checkpoint protected-path history is unbounded")
    if len(observed) != len(set(observed)):
        raise AuditError("checkpoint protected-path history contains duplicates")
    return observed


def _checkpoint_descendant_commits(
    terminal_tip: str,
    current_tip: str,
    *,
    repository: str,
    fetch_github: Callable[[str], Any],
) -> list[str]:
    if terminal_tip == current_tip:
        return []
    observed: list[str] = []
    expected_total: int | None = None
    page = 1
    while True:
        value = fetch_github(
            f"repos/{repository}/compare/{terminal_tip}...{current_tip}"
            f"?per_page=100&page={page}"
        )
        if (
            not isinstance(value, Mapping)
            or value.get("status") != "ahead"
            or value.get("behind_by") != 0
            or not isinstance(value.get("ahead_by"), int)
            or isinstance(value.get("ahead_by"), bool)
            or not isinstance(value.get("total_commits"), int)
            or isinstance(value.get("total_commits"), bool)
            or not isinstance(value.get("commits"), list)
            or any(not isinstance(item, Mapping) for item in value.get("commits", []))
            or not isinstance(value.get("base_commit"), Mapping)
            or not isinstance(value.get("merge_base_commit"), Mapping)
            or value["base_commit"].get("sha") != terminal_tip
            or value["merge_base_commit"].get("sha") != terminal_tip
        ):
            raise AuditError("checkpoint terminal tip is not a remote ancestor")
        total = int(value["total_commits"])
        if expected_total is None:
            expected_total = total
            if int(value["ahead_by"]) != total or total <= 0:
                raise AuditError("checkpoint post-terminal ancestry count changed")
        elif total != expected_total or int(value["ahead_by"]) != expected_total:
            raise AuditError("checkpoint post-terminal compare pagination changed")
        page_shas = [str(item.get("sha")) for item in value["commits"]]
        if any(re.fullmatch(r"[0-9a-f]{40}", item) is None for item in page_shas):
            raise AuditError("checkpoint post-terminal commit SHA changed")
        if not page_shas:
            break
        observed.extend(page_shas)
        if len(page_shas) > 100 or len(observed) > expected_total:
            raise AuditError("checkpoint post-terminal compare overran total")
        page += 1
        if page > 1000:  # pragma: no cover - defensive API bound
            raise AuditError("checkpoint post-terminal compare is unbounded")
    if len(observed) != expected_total or len(observed) != len(set(observed)):
        raise AuditError("checkpoint post-terminal compare is incomplete")
    if observed[-1] != current_tip:
        raise AuditError("checkpoint post-terminal compare does not end at current ref")
    return observed


def _checkpoint_api_workflow_projection(
    value: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    repository: str,
    expected_head_sha: str,
) -> dict[str, Any]:
    fields = {
        "run_id": value.get("id"),
        "workflow_id": value.get("workflow_id"),
        "workflow_name": value.get("name"),
        "workflow_path": value.get("path"),
        "event": value.get("event"),
        "head_sha": value.get("head_sha"),
        "run_attempt": value.get("run_attempt"),
        "status": value.get("status"),
        "conclusion": value.get("conclusion"),
        "created_at": value.get("created_at"),
        "run_started_at": value.get("run_started_at"),
        "updated_at": value.get("updated_at"),
        "html_url": value.get("html_url"),
    }
    run_id = fields["run_id"]
    if (
        isinstance(run_id, bool)
        or not isinstance(run_id, int)
        or run_id <= 0
        or isinstance(fields["workflow_id"], bool)
        or not isinstance(fields["workflow_id"], int)
        or int(fields["workflow_id"]) <= 0
        or fields["workflow_name"] != "tests"
        or fields["workflow_path"] != ".github/workflows/tests.yml"
        or fields["event"] != "pull_request"
        or fields["head_sha"] != expected_head_sha
        or fields["run_attempt"] != 1
        or fields["html_url"]
        != f"https://github.com/{repository}/actions/runs/{run_id}"
    ):
        raise AuditError("checkpoint GitHub workflow identity changed")
    created = _aware_timestamp(fields["created_at"], "checkpoint workflow created_at")
    started = _aware_timestamp(
        fields["run_started_at"], "checkpoint workflow run_started_at"
    )
    updated = _aware_timestamp(fields["updated_at"], "checkpoint workflow updated_at")
    if not created <= started <= updated:
        raise AuditError("checkpoint GitHub workflow timestamps are not causal")
    allowed_conclusions = set(
        protocol["daily_preopen_checkpoint_contract"][
            "checkpoint_workflow_run_observation_contract"
        ]["terminal_conclusion_values"]
    )
    if fields["status"] != "completed" or fields["conclusion"] not in allowed_conclusions:
        raise AuditError("checkpoint GitHub workflow terminal authority changed")
    return fields


def _checkpoint_first_workflow_run(
    commit_sha: str,
    *,
    protocol: Mapping[str, Any],
    repository: str,
    fetch_github: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    base = (
        f"repos/{repository}/actions/runs?event=pull_request"
        f"&head_sha={commit_sha}&per_page=100"
    )
    runs: list[Mapping[str, Any]] = []
    total: int | None = None
    page = 1
    while True:
        endpoint = base if page == 1 else f"{base}&page={page}"
        response = fetch_github(endpoint)
        observed_total = response.get("total_count")
        page_runs = response.get("workflow_runs")
        if (
            isinstance(observed_total, bool)
            or not isinstance(observed_total, int)
            or observed_total < 0
            or not isinstance(page_runs, list)
            or any(not isinstance(item, Mapping) for item in page_runs)
        ):
            raise AuditError("checkpoint workflow pagination response changed")
        if total is None:
            total = observed_total
        elif total != observed_total:
            raise AuditError("checkpoint workflow pagination total changed")
        runs.extend(page_runs)
        if len(runs) >= total:
            if len(runs) != total:
                raise AuditError("checkpoint workflow pagination overran total")
            break
        if len(page_runs) != 100:
            raise AuditError("checkpoint workflow pagination ended early")
        page += 1
        if page > 1000:  # pragma: no cover - defensive API bound
            raise AuditError("checkpoint workflow pagination is unbounded")
    matching = [
        item
        for item in runs
        if item.get("path") == ".github/workflows/tests.yml"
        and item.get("head_sha") == commit_sha
        and item.get("event") == "pull_request"
        and item.get("run_attempt") == 1
        and isinstance(item.get("id"), int)
        and not isinstance(item.get("id"), bool)
    ]
    all_identifiers = [item.get("id") for item in runs]
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in all_identifiers
    ) or len(all_identifiers) != len(set(all_identifiers)):
        raise AuditError("checkpoint workflow pagination contains invalid run IDs")
    identifiers = [int(item["id"]) for item in matching]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise AuditError("checkpoint first workflow run is missing or duplicated")
    run_id = min(identifiers)
    listed = next(item for item in matching if int(item["id"]) == run_id)
    fetched = fetch_github(f"repos/{repository}/actions/runs/{run_id}")
    listed_projection = _checkpoint_api_workflow_projection(
        listed,
        protocol=protocol,
        repository=repository,
        expected_head_sha=commit_sha,
    )
    fetched_projection = _checkpoint_api_workflow_projection(
        fetched,
        protocol=protocol,
        repository=repository,
        expected_head_sha=commit_sha,
    )
    if listed_projection != fetched_projection:
        raise AuditError("checkpoint first workflow run changed during refetch")
    return fetched_projection


def validate_checkpoint_evidence(
    decisions: Sequence[Mapping[str, Any]],
    *,
    proposal_directory: str | Path,
    checkpoint_core_store_root: str | Path,
    protocol: Mapping[str, Any],
    activation_payload: Mapping[str, Any],
    runner_sha256: str,
    external_identity_registry: dict[tuple[int, int], str] | None = None,
    repository_root: str | Path = ROOT,
    github_fetcher: Callable[[str], Any] | None = None,
) -> dict[str, str]:
    """Reconstruct both daily roles from remote Git Data and Actions evidence."""

    if not decisions:
        raise AuditError("terminal checkpoint validation requires decisions")
    # Daily publication has no local-Git authority.  Keep this compatibility
    # argument non-authoritative so older pure callers cannot accidentally make
    # a working tree or local ref part of terminal validation.
    _ = repository_root
    contract = protocol["daily_preopen_checkpoint_contract"]
    roles = list(contract["proposal_roles"])
    if roles != ["safety_cash", "primary"]:
        raise AuditError("checkpoint role publication order changed")
    proposal_root = _require_plain_directory(
        proposal_directory, label="checkpoint proposal directory"
    )
    proposal_relative_root = str(contract["proposal_directory"])
    expected_sessions = [str(row["session_date"]) for row in decisions]
    if len(expected_sessions) != len(set(expected_sessions)):
        raise AuditError("checkpoint decisions contain duplicate sessions")
    session_entries = sorted(proposal_root.iterdir(), key=lambda item: item.name)
    if [item.name for item in session_entries] != expected_sessions:
        raise AuditError("checkpoint proposal session-directory set changed")
    proposal_paths: dict[tuple[str, str], Path] = {}
    for directory in session_entries:
        _require_plain_directory(directory, label="checkpoint proposal session directory")
        children = sorted(directory.iterdir(), key=lambda item: item.name)
        if [item.name for item in children] != ["primary.json", "safety_cash.json"]:
            raise AuditError("checkpoint proposal pair is incomplete or has extras")
        for path in children:
            _stable_plain_file_bytes(path, label="checkpoint proposal pair member")
            proposal_paths[(directory.name, path.stem)] = path

    fetch_github = _github_api_fetch if github_fetcher is None else github_fetcher
    repository = str(protocol["repository"])
    branch = str(protocol["branch"])
    ref_endpoint = f"repos/{repository}/git/ref/heads/{branch}"
    current_ref = _checkpoint_git_ref_projection(
        fetch_github(ref_endpoint), repository=repository, branch=branch
    )
    current_tip = current_ref["object_sha"]

    proposal_history = _checkpoint_path_history(
        repository=repository,
        branch=branch,
        path=proposal_relative_root,
        fetch_github=fetch_github,
    )
    if len(proposal_history) != 2 * len(decisions):
        raise AuditError("checkpoint proposal history has extra or missing commits")
    ordered_commits = list(reversed(proposal_history))
    terminal_checkpoint_tip = ordered_commits[-1]
    post_terminal_commits = set(
        _checkpoint_descendant_commits(
            terminal_checkpoint_tip,
            current_tip,
            repository=repository,
            fetch_github=fetch_github,
        )
    )
    if post_terminal_commits.intersection(proposal_history):
        raise AuditError("checkpoint proposal path changed after terminal publication")

    protected_paths = list(
        protocol["activation"]["preregistration_commit"]["required_paths"]
    )
    if len(protected_paths) != len(set(protected_paths)):
        raise AuditError("checkpoint protected-path registry changed")
    for protected_path in protected_paths:
        history = _checkpoint_path_history(
            repository=repository,
            branch=branch,
            path=str(protected_path),
            fetch_github=fetch_github,
        )
        if not history:
            raise AuditError("checkpoint protected-path history is incomplete")
        if post_terminal_commits.intersection(history):
            raise AuditError("checkpoint protected path changed after terminal tip")

    activation_commit = str(decisions[0]["activation_receipt_commit_sha"])
    activation_git = _checkpoint_git_commit_projection(
        fetch_github(f"repos/{repository}/git/commits/{activation_commit}"),
        repository=repository,
        expected_sha=activation_commit,
    )
    _, activation_tree = _checkpoint_git_tree_projection(
        fetch_github(
            f"repos/{repository}/git/trees/{activation_git['tree_sha']}?recursive=1"
        ),
        expected_sha=activation_git["tree_sha"],
    )
    activation_files = {
        path: entry for path, entry in activation_tree.items() if entry["type"] != "tree"
    }
    if any(
        path == proposal_relative_root
        or path.startswith(f"{proposal_relative_root}/")
        for path in activation_tree
    ):
        raise AuditError("checkpoint proposal path predates activation receipt C")
    if any(path not in activation_files for path in protected_paths):
        raise AuditError("checkpoint activation tree lacks a protected path")
    artifact_pairs = (
        ("protocol_path", "protocol_sha256"),
        ("hypothesis_path", "hypothesis_sha256"),
        ("runtime_lock_path", "runtime_lock_sha256"),
        ("runner_path", "runner_sha256"),
        ("audit_path", "audit_sha256"),
        ("tests_path", "tests_sha256"),
        ("session_calendar_path", "session_calendar_sha256"),
        ("workflow_path", "workflow_sha256"),
    )
    protected_authority = {
        str(activation_payload[path_field]): str(activation_payload[hash_field])
        for path_field, hash_field in artifact_pairs
    }
    if set(protected_authority) != set(protected_paths):
        raise AuditError("checkpoint activation protected-path authority changed")
    for path, expected_sha256 in protected_authority.items():
        _require_nonzero_sha(expected_sha256, f"checkpoint protected path {path}")
        entry = activation_files[path]
        if entry["type"] != "blob":
            raise AuditError("checkpoint protected authority is not a Git blob")
        _, protected_bytes = _checkpoint_git_blob_projection(
            fetch_github(f"repos/{repository}/git/blobs/{entry['sha']}"),
            expected_sha=entry["sha"],
        )
        if hashlib.sha256(protected_bytes).hexdigest() != expected_sha256:
            raise AuditError("checkpoint remote protected authority changed")

    expected_subjects = [
        (session, role) for session in expected_sessions for role in roles
    ]
    if len(expected_subjects) != len(ordered_commits):
        raise AuditError("checkpoint ordered subject cardinality changed")
    decision_by_session = {str(row["session_date"]): row for row in decisions}
    role_evidence: dict[tuple[str, str], dict[str, Any]] = {}
    previous_commit = activation_commit
    previous_files = activation_files
    for (session, role), commit_sha in zip(
        expected_subjects, ordered_commits, strict=True
    ):
        decision = decision_by_session[session]
        proposal_path = proposal_paths[(session, role)]
        proposal_bytes = _stable_plain_file_bytes(
            proposal_path, label=f"checkpoint {role} proposal"
        )
        proposal_file_sha = hashlib.sha256(proposal_bytes).hexdigest()
        proposal = validate_checkpoint_proposal(
            _parse_json_object_bytes(
                proposal_bytes, label=f"checkpoint {role} proposal"
            ),
            decision=decision,
            role=role,
            protocol=protocol,
            runner_sha256=runner_sha256,
        )
        relative_path = f"{proposal_relative_root}/{session}/{role}.json"
        git_commit = _checkpoint_git_commit_projection(
            fetch_github(f"repos/{repository}/git/commits/{commit_sha}"),
            repository=repository,
            expected_sha=commit_sha,
        )
        expected_message = f"model-v18 checkpoint {session} {role}"
        if (
            git_commit["parent_shas"] != [previous_commit]
            or git_commit["message"] != expected_message
            or _aware_timestamp(
                proposal["created_at"], "checkpoint proposal created_at"
            )
            >= _aware_timestamp(
                git_commit["committer_date"], "checkpoint commit committer_date"
            )
        ):
            raise AuditError("checkpoint remote commit parent/message/time changed")
        rest_commit = _checkpoint_api_commit_projection(
            fetch_github(f"repos/{repository}/commits/{commit_sha}"),
            repository=repository,
            expected_sha=commit_sha,
        )
        if any(
            rest_commit[field] != git_commit[field]
            for field in (
                "message",
                "tree_sha",
                "parent_shas",
                "html_url",
                "committer_date",
            )
        ):
            raise AuditError("checkpoint Git Data and REST commit projections differ")
        _, tree = _checkpoint_git_tree_projection(
            fetch_github(
                f"repos/{repository}/git/trees/{git_commit['tree_sha']}?recursive=1"
            ),
            expected_sha=git_commit["tree_sha"],
        )
        files = {path: entry for path, entry in tree.items() if entry["type"] != "tree"}
        entry = files.get(relative_path)
        if (
            entry is None
            or entry["mode"] != "100644"
            or entry["type"] != "blob"
            or entry["size"] != len(proposal_bytes)
            or relative_path in previous_files
        ):
            raise AuditError("checkpoint tree does not contain one new proposal blob")
        expected_files = dict(previous_files)
        expected_files[relative_path] = entry
        if files != expected_files:
            raise AuditError("checkpoint tree changed more than the proposal path")
        _, remote_proposal_bytes = _checkpoint_git_blob_projection(
            fetch_github(f"repos/{repository}/git/blobs/{entry['sha']}"),
            expected_sha=entry["sha"],
        )
        if remote_proposal_bytes != proposal_bytes:
            raise AuditError("checkpoint remote proposal blob differs from canonical file")
        _checkpoint_compare_projection(
            fetch_github(
                f"repos/{repository}/compare/{previous_commit}...{commit_sha}"
            ),
            expected_parent=previous_commit,
            expected_child=commit_sha,
            expected_path=relative_path,
            expected_blob_sha=entry["sha"],
        )
        workflow = _checkpoint_first_workflow_run(
            commit_sha,
            protocol=protocol,
            repository=repository,
            fetch_github=fetch_github,
        )
        core_bytes = _read_external_object_bytes(
            checkpoint_core_store_root,
            proposal["sealed_core_object_key"],
            required_prefix="model_v18_shoulder_state/checkpoint-core/",
            identity_registry=external_identity_registry,
        )
        if (
            len(core_bytes) != CHECKPOINT_CORE_ENVELOPE_BYTES
            or hashlib.sha256(core_bytes).hexdigest()
            != proposal["sealed_core_sha256"]
        ):
            raise AuditError("checkpoint core exact bytes differ from proposal")
        core, core_hash = validate_checkpoint_core_object(
            parse_checkpoint_core_envelope(core_bytes),
            proposal,
            protocol=protocol,
        )
        role_evidence[(session, role)] = {
            "proposal": proposal,
            "proposal_path": relative_path,
            "proposal_file_sha256": proposal_file_sha,
            "core": core,
            "core_sha256": core_hash,
            "commit": rest_commit,
            "workflow": workflow,
        }
        previous_commit = commit_sha
        previous_files = files
    if previous_commit != terminal_checkpoint_tip:
        raise AuditError("checkpoint pair chain does not end at terminal primary")

    proposal_set: list[dict[str, Any]] = []
    core_set: list[dict[str, Any]] = []
    evidence_set: list[dict[str, Any]] = []
    for session in expected_sessions:
        decision = decision_by_session[session]
        pair = {role: role_evidence[(session, role)] for role in roles}
        safety_proposal = pair["safety_cash"]["proposal"]
        primary_proposal = pair["primary"]["proposal"]
        role_specific = {
            "checkpoint_id",
            "publication_ordinal",
            "checkpoint_role",
            "sealed_core_object_key",
            "sealed_core_sha256",
            "proposal_sha256",
        }
        if {
            key: value
            for key, value in safety_proposal.items()
            if key not in role_specific
        } != {
            key: value
            for key, value in primary_proposal.items()
            if key not in role_specific
        }:
            raise AuditError("checkpoint proposals are not one atomic prepared pair")
        if canonical_json_bytes(pair["safety_cash"]["core"]) != canonical_json_bytes(
            pair["primary"]["core"]
        ):
            raise AuditError("checkpoint safety evidence does not copy the primary core")

        cutoff = _aware_timestamp(decision["decision_cutoff"], "checkpoint cutoff")
        safety_workflow = pair["safety_cash"]["workflow"]
        primary_workflow = pair["primary"]["workflow"]
        safety_created = _aware_timestamp(
            safety_workflow["created_at"], "safety workflow created_at"
        )
        primary_created = _aware_timestamp(
            primary_workflow["created_at"], "primary workflow created_at"
        )
        if not safety_created <= primary_created < cutoff:
            raise AuditError("checkpoint mandatory primary was not timely created")
        materialized = _aware_timestamp(
            decision["decision_materialized_at"],
            "checkpoint decision_materialized_at",
        )
        if any(
            _aware_timestamp(
                pair[role]["workflow"]["updated_at"],
                f"{role} workflow updated_at",
            )
            > materialized
            for role in roles
        ):
            raise AuditError(
                "checkpoint terminal workflow evidence postdates materialization"
            )
        expected_resolution = "primary"
        expected_reason = "primary_commitment_timely"
        if (
            decision["checkpoint_resolution"] != expected_resolution
            or decision["checkpoint_resolution_reason"] != expected_reason
        ):
            raise AuditError("checkpoint role resolution does not independently recompute")
        selected = pair["primary"]
        selected_commit_projection = {
            key: selected["commit"][key]
            for key in ("commit_sha", "html_url", "committer_date", "parent_shas")
        }
        if (
            canonical_json_bytes(_checkpoint_core_from_row(decision, protocol))
            != canonical_json_bytes(selected["core"])
            or decision["checkpoint_core_sha256"] != selected["core_sha256"]
            or decision["checkpoint_proposal_path"] != selected["proposal_path"]
            or decision["checkpoint_proposal_file_sha256"]
            != selected["proposal_file_sha256"]
            or decision["checkpoint_proposal_sha256"]
            != selected["proposal"]["proposal_sha256"]
            or decision["checkpoint_commit_sha"]
            != selected["commit"]["commit_sha"]
            or decision["checkpoint_workflow_run_id"]
            != selected["workflow"]["run_id"]
            or _aware_timestamp(
                decision["checkpoint_workflow_run_updated_at"],
                "stored checkpoint workflow updated_at",
            )
            != _aware_timestamp(
                selected["workflow"]["updated_at"],
                "refetched checkpoint workflow updated_at",
            )
            or decision["checkpoint_commit_observation"]["canonical_projection"]
            != selected_commit_projection
            or decision["checkpoint_branch_observation"]["canonical_projection"][
                "commit_sha"
            ]
            != selected["commit"]["commit_sha"]
            or decision["checkpoint_workflow_run_observation"]["canonical_projection"]
            != selected["workflow"]
        ):
            raise AuditError("checkpoint selected public evidence changed")

        proposal_set.append(
            {
                "target_session": session,
                "checkpoint_batch_id": primary_proposal["checkpoint_batch_id"],
                "checkpoint_resolution": expected_resolution,
                "roles": [
                    {
                        "checkpoint_role": role,
                        "publication_ordinal": pair[role]["proposal"][
                            "publication_ordinal"
                        ],
                        "checkpoint_proposal_path": pair[role]["proposal_path"],
                        "checkpoint_proposal_file_sha256": pair[role][
                            "proposal_file_sha256"
                        ],
                        "checkpoint_proposal_sha256": pair[role]["proposal"][
                            "proposal_sha256"
                        ],
                    }
                    for role in roles
                ],
            }
        )
        core_set.append(
            {
                "target_session": session,
                "roles": [
                    {
                        "checkpoint_role": role,
                        "sealed_core_object_key": pair[role]["proposal"][
                            "sealed_core_object_key"
                        ],
                        "sealed_core_byte_count": pair[role]["proposal"][
                            "sealed_core_byte_count"
                        ],
                        "sealed_core_sha256": pair[role]["proposal"][
                            "sealed_core_sha256"
                        ],
                        "decision_core_sha256": pair[role]["core_sha256"],
                    }
                    for role in roles
                ],
            }
        )
        evidence_set.append(
            {
                "target_session": session,
                "checkpoint_batch_id": primary_proposal["checkpoint_batch_id"],
                "checkpoint_resolution": expected_resolution,
                "checkpoint_resolution_reason": expected_reason,
                "roles": [
                    {
                        "checkpoint_role": role,
                        "checkpoint_proposal_path": pair[role]["proposal_path"],
                        "checkpoint_proposal_file_sha256": pair[role][
                            "proposal_file_sha256"
                        ],
                        "checkpoint_proposal_sha256": pair[role]["proposal"][
                            "proposal_sha256"
                        ],
                        "checkpoint_commit_sha": pair[role]["commit"]["commit_sha"],
                        "parent_shas": pair[role]["commit"]["parent_shas"],
                        "commit_html_url": pair[role]["commit"]["html_url"],
                        "committer_date": pair[role]["commit"]["committer_date"],
                        "workflow_run_id": pair[role]["workflow"]["run_id"],
                        "workflow_id": pair[role]["workflow"]["workflow_id"],
                        "workflow_name": pair[role]["workflow"]["workflow_name"],
                        "workflow_path": pair[role]["workflow"]["workflow_path"],
                        "event": pair[role]["workflow"]["event"],
                        "head_sha": pair[role]["workflow"]["head_sha"],
                        "run_attempt": pair[role]["workflow"]["run_attempt"],
                        "status": pair[role]["workflow"]["status"],
                        "conclusion": pair[role]["workflow"]["conclusion"],
                        "created_at": pair[role]["workflow"]["created_at"],
                        "run_started_at": pair[role]["workflow"]["run_started_at"],
                        "updated_at": pair[role]["workflow"]["updated_at"],
                        "workflow_html_url": pair[role]["workflow"]["html_url"],
                    }
                    for role in roles
                ],
            }
        )
    return {
        "checkpoint_proposal_set_sha256": canonical_json_sha256(proposal_set),
        "checkpoint_core_object_set_sha256": canonical_json_sha256(core_set),
        "checkpoint_evidence_set_sha256": canonical_json_sha256(evidence_set),
    }


def validate_decision_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = validate_hash_chain(records, required_fields=DECISION_REQUIRED_FIELDS)
    protocol = read_json(DEFAULT_PROTOCOL)
    previous_date: pd.Timestamp | None = None
    month_state: dict[str, tuple[Any, ...]] = {}
    receipt_observation_identity: tuple[str, str, str] | None = None
    for row in rows:
        if OUTCOME_FIELDS & set(row):
            raise AuditError("decision ledger contains outcome fields")
        if row["protocol_id"] != PROTOCOL_ID:
            raise AuditError("decision protocol id changed")
        if PROTOCOL_SHA256 != "__PENDING_PROTOCOL_SHA256__" and row[
            "protocol_sha256"
        ] != PROTOCOL_SHA256:
            raise AuditError("decision protocol hash changed")
        if row["candidate_id"] != SH01:
            raise AuditError("decision candidate changed")
        resolution = str(row["checkpoint_resolution"])
        checkpoint_contract = protocol["daily_preopen_checkpoint_contract"]
        if resolution not in set(checkpoint_contract["checkpoint_resolution_values"]):
            raise AuditError("decision checkpoint resolution is not registered")
        reason = row["checkpoint_resolution_reason"]
        if reason not in set(checkpoint_contract["checkpoint_resolution_reason_values"]):
            raise AuditError("decision checkpoint resolution reason is not registered")
        if reason != "primary_commitment_timely":
            raise AuditError("primary checkpoint resolution reason changed")
        for field in (
            "protocol_sha256",
            "runner_sha256",
            "runtime_lock_sha256",
            "activation_payload_sha256",
            "activation_receipt_sha256",
        ):
            _require_nonzero_sha(row[field], f"decision {field}")
        for field in ("source_manifest_sha256", "state_manifest_sha256"):
            _require_nonzero_sha(row[field], f"decision {field}")
        if row["runtime_lock_sha256"] != RUNTIME_LOCK_SHA256:
            raise AuditError("decision runtime-lock binding changed")
        runtime_verified = _runtime_verification_timestamp(row, "decision")
        fold_hash = row["c00_fold_manifest_sha256"]
        bundle_file_hash = row["fold_model_bundle_file_sha256"]
        if (fold_hash is None) != (bundle_file_hash is None):
            raise AuditError("decision fold/bundle hashes are not paired")
        if fold_hash is not None:
            _require_nonzero_sha(fold_hash, "decision c00_fold_manifest_sha256")
            _require_nonzero_sha(
                bundle_file_hash, "decision fold_model_bundle_file_sha256"
            )
        commit_sha = str(row["activation_receipt_commit_sha"])
        expected_commit_url = (
            "https://github.com/rokuroku-066/TSE-Session-Ranker/commit/"
            + commit_sha
        )
        if (
            re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None
            or row["activation_receipt_commit_url"] != expected_commit_url
        ):
            raise AuditError("decision receipt commit binding is invalid")
        receipt_observed_tip = str(row["branch_tip_sha_when_receipt_observed"])
        if re.fullmatch(r"[0-9a-f]{40}", receipt_observed_tip) is None:
            raise AuditError("decision receipt-observed branch tip is invalid")
        _require_nonzero_sha(
            row["activation_receipt_file_sha256"],
            "decision activation_receipt_file_sha256",
        )
        commit_observation, branch_observation = _validate_observation_derivation(
            commit_observation=row["receipt_commit_observation"],
            branch_observation=row["receipt_branch_observation"],
            protocol=protocol,
            commit_sha=row["activation_receipt_commit_sha"],
            commit_url=row["activation_receipt_commit_url"],
            committed_at=row["activation_receipt_commit_committed_at"],
            branch_tip_sha=row["branch_tip_sha_when_receipt_observed"],
            observed_at=row["activation_receipt_commit_observed_at"],
            label="receipt observation",
        )
        workflow_observation = _validate_workflow_observation_derivation(
            observation=row["receipt_workflow_run_observation"],
            protocol=protocol,
            expected_head_sha=row["activation_receipt_commit_sha"],
            run_id=row["activation_receipt_workflow_run_id"],
            updated_at=row["activation_receipt_workflow_run_updated_at"],
            observed_at=row["activation_receipt_workflow_run_observed_at"],
            label="receipt workflow observation",
        )
        observation_identity = (
            canonical_json_sha256(commit_observation),
            canonical_json_sha256(branch_observation),
            canonical_json_sha256(workflow_observation),
        )
        if receipt_observation_identity is None:
            receipt_observation_identity = observation_identity
        elif receipt_observation_identity != observation_identity:
            raise AuditError("receipt observation objects changed across decisions")
        committed = _aware_timestamp(
            row["activation_receipt_commit_committed_at"],
            "activation_receipt_commit_committed_at",
        )
        observed = _aware_timestamp(
            row["activation_receipt_commit_observed_at"],
            "activation_receipt_commit_observed_at",
        )
        workflow_updated = _aware_timestamp(
            row["activation_receipt_workflow_run_updated_at"],
            "activation_receipt_workflow_run_updated_at",
        )
        workflow_observed = _aware_timestamp(
            row["activation_receipt_workflow_run_observed_at"],
            "activation_receipt_workflow_run_observed_at",
        )
        if observed < committed:
            raise AuditError("decision receipt observation predates commit")
        session = pd.Timestamp(row["session_date"])
        if previous_date is not None and session <= previous_date:
            raise AuditError("decision sessions are not strictly chronological")
        previous_date = session
        core = validate_checkpoint_decision_core(
            _checkpoint_core_from_row(row, protocol),
            session_date=session,
            resolution=resolution,
            protocol=protocol,
        )
        expected_core_hash = canonical_json_sha256(core)
        if row["checkpoint_core_sha256"] != expected_core_hash:
            raise AuditError("decision checkpoint core hash changed")
        materialized = _aware_timestamp(
            row["decision_materialized_at"], "decision_materialized_at"
        )
        if materialized < max(observed, workflow_observed):
            raise AuditError("decision materialization predates activation evidence")
        expected_proposal_path = (
            f"research/model_v18_shoulder_state_checkpoint_proposals/"
            f"{session.date()}/primary.json"
        )
        if row["checkpoint_proposal_path"] != expected_proposal_path:
            raise AuditError("decision checkpoint proposal path changed")
        for field in (
            "checkpoint_core_sha256",
            "checkpoint_proposal_file_sha256",
            "checkpoint_proposal_sha256",
        ):
            _require_nonzero_sha(row[field], f"decision {field}")
        checkpoint_commit, checkpoint_branch = _validate_observation_derivation(
            commit_observation=row["checkpoint_commit_observation"],
            branch_observation=row["checkpoint_branch_observation"],
            protocol=protocol,
            commit_sha=row["checkpoint_commit_sha"],
            commit_url=row["checkpoint_commit_url"],
            committed_at=row["checkpoint_commit_committed_at"],
            branch_tip_sha=row["checkpoint_branch_tip_sha_when_observed"],
            observed_at=row["checkpoint_commit_observed_at"],
            label="checkpoint observation",
        )
        checkpoint_workflow = _validate_workflow_observation_derivation(
            observation=row["checkpoint_workflow_run_observation"],
            protocol=protocol,
            expected_head_sha=row["checkpoint_commit_sha"],
            run_id=row["checkpoint_workflow_run_id"],
            updated_at=row["checkpoint_workflow_run_updated_at"],
        observed_at=row["checkpoint_workflow_run_observed_at"],
        label="checkpoint workflow observation",
        checkpoint=True,
    )
        checkpoint_observed = max(
            _aware_timestamp(checkpoint_commit["retrieved_at"], "checkpoint commit retrieved"),
            _aware_timestamp(checkpoint_branch["retrieved_at"], "checkpoint branch retrieved"),
        )
        checkpoint_workflow_observed = _aware_timestamp(
            checkpoint_workflow["retrieved_at"], "checkpoint workflow retrieved"
        )
        checkpoint_projection = checkpoint_commit["canonical_projection"]
        checkpoint_branch_projection = checkpoint_branch["canonical_projection"]
        if (
            checkpoint_branch_projection["commit_sha"]
            != row["checkpoint_commit_sha"]
            or len(checkpoint_projection["parent_shas"]) != 1
        ):
            raise AuditError(
                "decision checkpoint branch/sole-parent authority changed"
            )
        cutoff = _aware_timestamp(row["decision_cutoff"], "decision cutoff")
        computed = _aware_timestamp(row["computed_at"], "decision computed_at")
        if (
            _aware_timestamp(
                checkpoint_workflow["canonical_projection"]["created_at"],
                "checkpoint workflow created_at",
            )
            >= cutoff
            or materialized < max(checkpoint_observed, checkpoint_workflow_observed)
            or materialized < computed
            or computed > cutoff
            or computed < max(observed, workflow_updated, workflow_observed)
            or runtime_verified is None
            or runtime_verified > computed
        ):
            raise AuditError("decision checkpoint evidence/timestamps do not recompute")

        source_complete = _strict_bool(row["source_complete"], "source_complete")
        model_complete = _strict_bool(row["model_complete"], "model_complete")
        if model_complete and not source_complete:
            raise AuditError("model cannot be complete when source is incomplete")
        state_available = _strict_bool(row["state_available"], "state_available")
        decision = str(row["decision"])
        if decision not in DECISION_VALUES:
            raise AuditError("decision value changed")
        month = session.to_period("M")
        expected_prior_months = [str(month - offset) for offset in (3, 2, 1)]
        prior_months = row["three_prior_calendar_months"]
        counts = row["three_complete_pair_day_counts"]
        medians = row["three_month_medians_pct"]
        if prior_months != expected_prior_months:
            raise AuditError("decision state months are not the immediate prior three")
        if not isinstance(counts, list) or len(counts) != 3:
            raise AuditError("decision state counts changed")
        if not isinstance(medians, list) or len(medians) != 3:
            raise AuditError("decision state medians changed")
        counts = [
            _strict_nonnegative_int(item, "decision state complete-pair count")
            for item in counts
        ]
        medians = [_finite_or_none(item, "state monthly median") for item in medians]
        for count, median in zip(counts, medians, strict=True):
            if (count >= MIN_COMPLETE_PAIRS_PER_MONTH) != (median is not None):
                raise AuditError("decision monthly state availability is inconsistent")
        expected_available = all(
            count >= MIN_COMPLETE_PAIRS_PER_MONTH for count in counts
        ) and all(item is not None for item in medians)
        if state_available != expected_available:
            raise AuditError("decision state availability does not recompute")
        expected_state = (
            float(np.median(np.asarray(medians, dtype=float)))
            if expected_available
            else None
        )
        state_value = _finite_or_none(row["state_value_pct"], "state_value_pct")
        if (state_value is None) != (expected_state is None) or (
            state_value is not None
            and not _canonical_float_equal(row["state_value_pct"], expected_state)
        ):
            raise AuditError("decision median-of-monthly-medians does not recompute")
        selected_rank = row["selected_source_rank"]
        selected_rank = (
            None
            if selected_rank is None
            else _strict_nonnegative_int(selected_rank, "selected_source_rank")
        )
        if selected_rank not in {None, 1, 2}:
            raise AuditError("selected_source_rank is invalid")

        rank1_code = row["c00_rank1_code"]
        rank2_code = row["c02_rank2_code"]
        candidate_code = row["candidate_selected_code"]
        if source_complete and model_complete:
            if fold_hash is None:
                raise AuditError("complete model is missing sealed fold/bundle hashes")
            if rank1_code is None or rank2_code is None or rank1_code == rank2_code:
                raise AuditError("complete C00 pair is not two distinct codes")
            if (
                _finite_or_none(row["c00_rank1_score"], "c00_rank1_score") is None
                or _finite_or_none(row["c02_rank2_score"], "c02_rank2_score")
                is None
            ):
                raise AuditError("complete C00 pair has missing scores")
        else:
            if any(
                item is not None
                for item in (
                    rank1_code,
                    rank2_code,
                    row["c00_rank1_score"],
                    row["c02_rank2_score"],
                )
            ):
                raise AuditError("incomplete source/model exposed a frozen pair")
        if not source_complete:
            if selected_rank is not None:
                raise AuditError("fail-closed source exposed selected state rank")
            expected_decision = (
                "fail_closed_source_empty"
                if row["failure_reason"] == "source_missing_before_cutoff"
                else "fail_closed_source_partial"
            )
            expected_code = None
        elif not model_complete:
            if selected_rank is not None:
                raise AuditError("fail-closed model exposed selected state rank")
            expected_decision = (
                "fail_closed_model_fold"
                if fold_hash is None
                else "fail_closed_model_pair"
            )
            expected_code = None
        elif not state_available:
            if selected_rank is not None:
                raise AuditError("unavailable state exposed a selected rank")
            expected_decision, expected_code = "cash_state_unavailable", None
        elif state_value == 0.0:
            if selected_rank is not None:
                raise AuditError("exact-zero state exposed a selected rank")
            expected_decision, expected_code = "cash_state_zero", None
        else:
            expected_rank = 1 if state_value > 0.0 else 2
            if selected_rank != expected_rank:
                raise AuditError("selected rank does not match state sign")
            expected_decision = (
                "selected_rank1" if expected_rank == 1 else "selected_rank2"
            )
            expected_code = rank1_code if expected_rank == 1 else rank2_code
        if decision != expected_decision or candidate_code != expected_code:
            raise AuditError("decision does not recompute from frozen inputs")
        reason_map = protocol["daily_decision_failure_reason_contract"][
            "decision_reason_map"
        ]
        if row["failure_reason"] != reason_map[decision]:
            raise AuditError("decision failure reason is invalid")

        month_text = str(month)
        state_identity = (
            row["state_manifest_sha256"],
            fold_hash,
            bundle_file_hash,
            state_available,
            tuple(prior_months),
            tuple(counts),
            tuple(medians),
            state_value,
            selected_rank,
        )
        prior = month_state.setdefault(month_text, state_identity)
        if state_identity != prior:
            raise AuditError("target-month state changed intramonth")
    return rows


def _read_external_object_bytes(
    raw_store_root: str | Path,
    object_key: str,
    *,
    required_prefix: str,
    identity_registry: dict[tuple[int, int], str] | None = None,
) -> bytes:
    """Open one retained object through pinned directory descriptors.

    Component-wise ``O_NOFOLLOW`` traversal closes the check/open race left by
    a Path.resolve() followed by a later stat/open.  Once the final descriptor
    is open, renaming or relinking any parent cannot redirect the bytes read.
    """

    key_path = Path(object_key)
    if (
        not isinstance(object_key, str)
        or "\x00" in object_key
        or not object_key.startswith(required_prefix)
        or key_path.is_absolute()
        or any(part in {"", ".", ".."} for part in key_path.parts)
    ):
        raise AuditError("external raw object key is unsafe or noncanonical")
    root_input = Path(raw_store_root).absolute()
    if root_input.is_symlink():
        raise AuditError("external raw evidence-store root is a symlink")
    try:
        root_resolved = root_input.resolve(strict=True)
    except OSError as exc:
        raise AuditError("external raw evidence-store root is unavailable") from exc
    repository = ROOT.resolve()
    if root_resolved == repository or repository in root_resolved.parents:
        raise AuditError("external raw evidence store must be outside the repository")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptors: list[int] = []
    try:
        current_fd = os.open(root_input, directory_flags)
        descriptors.append(current_fd)
        for component in key_path.parts[:-1]:
            current_fd = os.open(component, directory_flags, dir_fd=current_fd)
            descriptors.append(current_fd)
        object_fd = os.open(key_path.parts[-1], file_flags, dir_fd=current_fd)
        descriptors.append(object_fd)
        metadata = os.fstat(object_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise AuditError("external raw object is not a single-link regular file")
        if identity_registry is not None:
            identity = (int(metadata.st_dev), int(metadata.st_ino))
            prior_key = identity_registry.setdefault(identity, object_key)
            if prior_key != object_key:
                raise AuditError(
                    "distinct external raw object keys alias the same physical inode"
                )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(object_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(object_fd)
        if (
            after.st_nlink != 1
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
            )
        ):
            raise AuditError("external raw object changed while pinned")
        return b"".join(chunks)
    except AuditError:
        raise
    except OSError as exc:
        raise AuditError(
            "external raw object descriptor traversal raced or encountered a symlink"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
def validate_outcome_manifest(
    manifest: Mapping[str, Any],
    decision: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    raw_store_root: str | Path | None = None,
    external_identity_registry: dict[tuple[int, int], str] | None = None,
) -> dict[str, Any]:
    """Validate one decision-bound outcome manifest and optional raw object."""

    contract = protocol["source_contract"]["outcome_daily"]
    required = set(contract["manifest_required_fields"])
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise AuditError("outcome manifest fields changed")
    value = dict(manifest)
    session = str(pd.Timestamp(value["target_session"]).date())
    if session != value["target_session"] or session != decision["session_date"]:
        raise AuditError("outcome manifest target differs from decision")
    if value["schema_version"] != 1:
        raise AuditError("outcome manifest schema changed")
    expected_key = f"model_v18_shoulder_state/outcome/{session}.pdf"
    if value["raw_source_object_key"] != expected_key:
        raise AuditError("outcome raw object key changed")
    source_name = value["source_file_name"]
    if (
        not isinstance(source_name, str)
        or not source_name
        or Path(source_name).name != source_name
        or not source_name.lower().endswith(".pdf")
        or session.replace("-", "") not in source_name
        or not isinstance(value["source_url"], str)
        or not value["source_url"].startswith("https://www.jpx.co.jp/")
    ):
        raise AuditError("outcome source filename/URL is invalid")
    byte_count = _strict_nonnegative_int(
        value["source_byte_count"], "outcome source byte count"
    )
    if byte_count <= 0:
        raise AuditError("outcome source byte count must be positive")
    _require_nonzero_sha(value["source_sha256"], "outcome source SHA-256")
    received = _aware_timestamp(value["source_received_at"], "source_received_at")
    runtime_verified = _runtime_verification_timestamp(value, "outcome manifest")
    created = _aware_timestamp(value["created_at"], "outcome manifest created_at")
    sealed = _aware_timestamp(value["sealed_at"], "outcome manifest sealed_at")
    decision_computed = _aware_timestamp(
        decision["decision_materialized_at"], "decision materialized_at"
    )
    if not (
        decision_computed < received
        and runtime_verified <= created
        and received <= created <= sealed
    ):
        raise AuditError("outcome manifest timestamps violate the outcome DAG")
    if (
        value["parser_path"] != contract["parser_path"]
        or value["parser_version"] != contract["parser_version"]
        or value["parser_sha256"] != contract["parser_sha256"]
        or value["python_version"] != LOCKED_PYTHON_VERSION
        or value["canonical_json_contract"] != "project_canonical_json_v1"
    ):
        raise AuditError("outcome parser/canonical contract changed")
    counts = {
        field: _strict_nonnegative_int(value[field], f"outcome {field}")
        for field in (
            "parsed_row_count",
            "rejected_row_count",
            "duplicate_date_code_count",
            "parsed_unique_date_count",
            "target_session_row_count",
        )
    }
    if (
        counts["parsed_row_count"] <= 0
        or counts["rejected_row_count"] != 0
        or counts["duplicate_date_code_count"] != 0
        or counts["parsed_unique_date_count"] != 1
        or counts["target_session_row_count"] != counts["parsed_row_count"]
    ):
        raise AuditError("outcome parser/date counts are invalid")
    pair_existed = bool(decision["source_complete"] and decision["model_complete"])
    for rank, decision_code_field in ((1, "c00_rank1_code"), (2, "c02_rank2_code")):
        code = value[f"rank{rank}_code"]
        open_value = _finite_or_none(value[f"rank{rank}_open"], f"rank{rank} open")
        close_value = _finite_or_none(value[f"rank{rank}_close"], f"rank{rank} close")
        return_value = _finite_or_none(
            value[f"rank{rank}_recomputed_oc_return_pct"],
            f"rank{rank} recomputed return",
        )
        if not pair_existed:
            if any(item is not None for item in (code, open_value, close_value, return_value)):
                raise AuditError("fail-closed decision exposed outcome rank values")
            continue
        if code != decision[decision_code_field]:
            raise AuditError("outcome manifest rank code differs from frozen decision")
        values_present = [item is not None for item in (open_value, close_value, return_value)]
        if any(values_present) and not all(values_present):
            raise AuditError("outcome rank open/close/return nullability differs")
        if all(values_present):
            if open_value <= 0.0 or close_value <= 0.0:
                raise AuditError("outcome rank open/close must be positive")
            expected_return = (close_value / open_value - 1.0) * 100.0
            if not _canonical_float_equal(
                value[f"rank{rank}_recomputed_oc_return_pct"], expected_return
            ):
                raise AuditError("outcome return does not recompute from open/close")
    if (
        value["decision_record_sha256"] != decision["record_sha256"]
        or value["protocol_sha256"] != PROTOCOL_SHA256
        or value["activation_payload_sha256"]
        != decision["activation_payload_sha256"]
        or value["activation_receipt_sha256"]
        != decision["activation_receipt_sha256"]
    ):
        raise AuditError("outcome manifest decision/protocol binding changed")
    for field in (
        "decision_record_sha256",
        "protocol_sha256",
        "activation_payload_sha256",
        "activation_receipt_sha256",
        "outcome_manifest_sha256",
    ):
        _require_nonzero_sha(value[field], f"outcome manifest {field}")
    expected_hash = canonical_json_sha256(
        value, exclude_fields={"outcome_manifest_sha256"}
    )
    if value["outcome_manifest_sha256"] != expected_hash:
        raise AuditError("outcome manifest self-hash mismatch")
    if raw_store_root is not None:
        expected_key = f"model_v18_shoulder_state/outcome/{session}.pdf"
        if value["raw_source_object_key"] != expected_key:
            raise AuditError("outcome raw object key is unsafe or noncanonical")
        raw_bytes = _read_external_object_bytes(
            raw_store_root,
            value["raw_source_object_key"],
            required_prefix="model_v18_shoulder_state/outcome/",
            identity_registry=external_identity_registry,
        )
        if (
            len(raw_bytes) != byte_count
            or hashlib.sha256(raw_bytes).hexdigest() != value["source_sha256"]
        ):
            raise AuditError("outcome raw object bytes differ from manifest")
    return value


def reparse_outcome_manifest(
    manifest: Mapping[str, Any],
    decision: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    raw_store_root: str | Path,
    external_identity_registry: dict[tuple[int, int], str] | None = None,
) -> dict[str, Any]:
    """Reparse exact external PDF bytes and repeat the target date/code joins."""

    value = validate_outcome_manifest(manifest, decision, protocol)
    parser_path = ROOT / value["parser_path"]
    if sha256_file(parser_path) != value["parser_sha256"]:
        raise AuditError("outcome parser source SHA-256 changed")
    expected_key = (
        f"model_v18_shoulder_state/outcome/{value['target_session']}.pdf"
    )
    if value["raw_source_object_key"] != expected_key:
        raise AuditError("outcome raw object key is unsafe or noncanonical")
    raw_bytes = _read_external_object_bytes(
        raw_store_root,
        value["raw_source_object_key"],
        required_prefix="model_v18_shoulder_state/outcome/",
        identity_registry=external_identity_registry,
    )
    if (
        len(raw_bytes) != int(value["source_byte_count"])
        or hashlib.sha256(raw_bytes).hexdigest() != value["source_sha256"]
    ):
        raise AuditError("outcome raw object bytes differ from manifest")
    try:
        from tse_session_ranker.data.jpx import parse_jpx_text

        with tempfile.TemporaryDirectory(prefix="v18-outcome-audit-") as temporary:
            temporary_root = Path(temporary)
            pdf_path = temporary_root / f"{value['target_session']}.pdf"
            text_path = temporary_root / f"{value['target_session']}.txt"
            pdf_path.write_bytes(raw_bytes)
            _locked_pdf_to_text(pdf_path, text_path)
            frame, report = parse_jpx_text(text_path)
    except Exception as exc:  # parser/PDF errors are integrity failures
        raise AuditError("outcome raw object could not be independently reparsed") from exc
    parsed = frame.copy()
    parsed["date"] = pd.to_datetime(parsed["date"], errors="coerce").dt.normalize()
    parsed["code"] = parsed["code"].astype("string")
    if parsed[["date", "code"]].isna().any(axis=None):
        raise AuditError("outcome reparse contains invalid date/code keys")
    target = pd.Timestamp(value["target_session"])
    recomputed_counts = {
        "parsed_row_count": len(parsed),
        "rejected_row_count": int(report.rejected_rows),
        "duplicate_date_code_count": int(
            parsed.duplicated(["date", "code"], keep=False).sum()
        ),
        "parsed_unique_date_count": int(parsed["date"].nunique()),
        "target_session_row_count": int(parsed["date"].eq(target).sum()),
    }
    for field, expected in recomputed_counts.items():
        if value[field] != expected:
            raise AuditError(f"outcome reparse {field} differs from manifest")
    target_rows = parsed.loc[parsed["date"].eq(target)]
    for rank in (1, 2):
        code = value[f"rank{rank}_code"]
        if code is None:
            continue
        matched = target_rows.loc[target_rows["code"].eq(str(code))]
        expected_open: float | None = None
        expected_close: float | None = None
        expected_return: float | None = None
        if len(matched) == 1:
            candidate_open = float(matched.iloc[0]["open"])
            candidate_close = float(matched.iloc[0]["close"])
            if (
                math.isfinite(candidate_open)
                and math.isfinite(candidate_close)
                and candidate_open > 0.0
                and candidate_close > 0.0
            ):
                expected_open = candidate_open
                expected_close = candidate_close
                expected_return = (candidate_close / candidate_open - 1.0) * 100.0
        for field, expected in (
            (f"rank{rank}_open", expected_open),
            (f"rank{rank}_close", expected_close),
            (f"rank{rank}_recomputed_oc_return_pct", expected_return),
        ):
            observed = _finite_or_none(value[field], field)
            if observed is None or expected is None:
                if observed is not expected:
                    raise AuditError(f"outcome reparse {field} nullability differs")
            elif not _canonical_float_equal(value[field], expected):
                raise AuditError(f"outcome reparse {field} differs")
    return value


def validate_outcome_records(
    records: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    *,
    outcome_manifests: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = validate_hash_chain(records, required_fields=OUTCOME_REQUIRED_FIELDS)
    if len(rows) != len(decisions):
        raise AuditError("decision/outcome session counts differ")
    for outcome, decision in zip(rows, decisions, strict=True):
        if outcome["session_date"] != decision["session_date"]:
            raise AuditError("outcome session does not match decision")
        if outcome["decision_record_sha256"] != decision["record_sha256"]:
            raise AuditError("outcome does not bind its decision record")
        if outcome["protocol_sha256"] != decision["protocol_sha256"]:
            raise AuditError("outcome protocol hash changed")
        if outcome["activation_payload_sha256"] != decision[
            "activation_payload_sha256"
        ] or outcome["activation_receipt_sha256"] != decision[
            "activation_receipt_sha256"
        ]:
            raise AuditError("outcome activation binding changed")
        for field in (
            "decision_record_sha256",
            "outcome_source_sha256",
            "outcome_manifest_sha256",
            "protocol_sha256",
            "activation_payload_sha256",
            "activation_receipt_sha256",
        ):
            _require_nonzero_sha(outcome[field], f"outcome {field}")
        if outcome_manifests is not None:
            manifest = outcome_manifests.get(str(outcome["outcome_manifest_sha256"]))
            if manifest is None:
                raise AuditError("outcome record does not bind an outcome manifest")
            if (
                manifest["target_session"] != outcome["session_date"]
                or manifest["source_sha256"] != outcome["outcome_source_sha256"]
                or _aware_timestamp(
                    manifest["source_received_at"], "manifest source_received_at"
                )
                != _aware_timestamp(
                    outcome["outcome_received_at"], "outcome_received_at"
                )
                or manifest["decision_record_sha256"]
                != outcome["decision_record_sha256"]
            ):
                raise AuditError("outcome record/manifest binding changed")
            for rank in (1, 2):
                manifest_return = _finite_or_none(
                    manifest[f"rank{rank}_recomputed_oc_return_pct"],
                    f"manifest rank{rank} return",
                )
                ledger_return = _finite_or_none(
                    outcome[f"rank{rank}_oc_return_pct"],
                    f"outcome rank{rank} return",
                )
                if manifest_return is None or ledger_return is None:
                    if manifest_return is not ledger_return:
                        raise AuditError("outcome ledger/manifest return nullability differs")
                elif canonical_json_bytes(
                    manifest[f"rank{rank}_recomputed_oc_return_pct"]
                ) != canonical_json_bytes(outcome[f"rank{rank}_oc_return_pct"]):
                    raise AuditError("outcome ledger return differs from manifest")
        received = _aware_timestamp(
            outcome["outcome_received_at"], "outcome_received_at"
        )
        computed = _aware_timestamp(outcome["computed_at"], "outcome computed_at")
        if received <= _aware_timestamp(
            decision["decision_materialized_at"], "decision materialized_at"
        ) or computed < received:
            raise AuditError("outcome timestamps violate decision/record causality")
        if outcome_manifests is not None:
            manifest = outcome_manifests[str(outcome["outcome_manifest_sha256"])]
            if computed < _aware_timestamp(
                manifest["sealed_at"], "outcome manifest sealed_at"
            ):
                raise AuditError("outcome record predates its sealed manifest")

        observed1 = _strict_bool(
            outcome["rank1_outcome_observed"], "rank1_outcome_observed"
        )
        observed2 = _strict_bool(
            outcome["rank2_outcome_observed"], "rank2_outcome_observed"
        )
        return1 = _finite_or_none(outcome["rank1_oc_return_pct"], "rank1 return")
        return2 = _finite_or_none(outcome["rank2_oc_return_pct"], "rank2 return")
        if observed1 != (return1 is not None) or observed2 != (return2 is not None):
            raise AuditError("rank outcome observation/value presence differs")

        pair_existed = bool(decision["source_complete"] and decision["model_complete"])
        if not pair_existed and (observed1 or observed2):
            raise AuditError("fail-closed source/model exposed control outcomes")

        decision_value = str(decision["decision"])
        if decision_value == "selected_rank1" and observed1:
            candidate_observed, gross = True, float(return1)
        elif decision_value == "selected_rank2" and observed2:
            candidate_observed, gross = True, float(return2)
        else:
            candidate_observed, gross = False, 0.0
        if _strict_bool(
            outcome["candidate_outcome_observed"], "candidate_outcome_observed"
        ) != candidate_observed:
            raise AuditError("candidate outcome-observed flag changed")

        expected_values = {
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
        for field, expected in expected_values.items():
            observed = _finite_or_none(outcome[field], field)
            if observed is None or not _canonical_float_equal(
                outcome[field], expected
            ):
                raise AuditError(f"outcome {field} does not recompute")
    return rows


def recompute_picks_csv_bytes(
    decisions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> bytes:
    """Independently materialize the three registered pick rows per session."""

    records: list[dict[str, Any]] = []
    for decision, outcome in zip(decisions, outcomes, strict=True):
        specifications = (
            (C00_TOP1, 1, "c00_rank1_code", "rank1_outcome_observed", "c00_top1"),
            (C02_TOP2, 2, "c02_rank2_code", "rank2_outcome_observed", "c02_rank2"),
            (
                SH01,
                decision["selected_source_rank"],
                "candidate_selected_code",
                "candidate_outcome_observed",
                "candidate",
            ),
        )
        for candidate_id, rank, code_field, observed_field, prefix in specifications:
            records.append(
                {
                    "session_date": decision["session_date"],
                    "candidate_id": candidate_id,
                    "source_rank": rank,
                    "code": decision[code_field],
                    "outcome_observed": bool(outcome[observed_field]),
                    "gross_return_pct": outcome[f"{prefix}_gross_return_pct"],
                    "net20_return_pct": outcome[f"{prefix}_net20_return_pct"],
                    "net40_return_pct": outcome[f"{prefix}_net40_return_pct"],
                    "net60_return_pct": outcome[f"{prefix}_net60_return_pct"],
                    "checkpoint_core_sha256": decision["checkpoint_core_sha256"],
                    "decision_record_sha256": decision["record_sha256"],
                    "outcome_record_sha256": outcome["record_sha256"],
                }
            )
    frame = pd.DataFrame(records, columns=PICKS_FIELDS)
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _numeric_values_sha256(values: Sequence[float]) -> str:
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


def recompute_completed_month_records(
    records: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Recompute every forward month summary directly from linked ledgers."""

    month_rows = validate_completed_month_records(records, protocol)
    decision_rows = validate_decision_records(decisions)
    outcome_rows = validate_outcome_records(outcomes, decision_rows)
    outcome_by_session = {row["session_date"]: row for row in outcome_rows}
    for month_row in month_rows:
        month = pd.Period(month_row["completed_month"], freq="M")
        selected_decisions = [
            row
            for row in decision_rows
            if pd.Timestamp(row["session_date"]).to_period("M") == month
        ]
        pairs: list[tuple[str, float]] = []
        for decision in selected_decisions:
            outcome = outcome_by_session.get(decision["session_date"])
            if outcome is None:
                raise AuditError("completed month has a counted decision without outcome")
            complete = (
                bool(decision["source_complete"])
                and bool(decision["model_complete"])
                and decision["c00_rank1_code"] != decision["c02_rank2_code"]
                and bool(outcome["rank1_outcome_observed"])
                and bool(outcome["rank2_outcome_observed"])
            )
            if complete:
                pairs.append(
                    (
                        decision["session_date"],
                        float(outcome["rank1_oc_return_pct"])
                        - float(outcome["rank2_oc_return_pct"]),
                    )
                )
        values = [item[1] for item in pairs]
        median = (
            float(np.median(np.asarray(values, dtype=float)))
            if len(values) >= MIN_COMPLETE_PAIRS_PER_MONTH
            else None
        )
        expected = {
            "counted_scheduled_sessions": len(selected_decisions),
            "complete_pair_days": len(values),
            "ordered_complete_pair_session_sha256": canonical_json_sha256(
                [item[0] for item in pairs]
            ),
            "ordered_difference_values_sha256": _numeric_values_sha256(values),
            "monthly_median_rank1_minus_rank2_pct": median,
            "available": median is not None,
        }
        for field, expected_value in expected.items():
            observed = month_row[field]
            if isinstance(expected_value, float):
                if not _canonical_float_equal(observed, expected_value):
                    raise AuditError(f"completed month {field} does not recompute")
            elif observed != expected_value:
                raise AuditError(f"completed month {field} does not recompute")
    return month_rows


def preflight_terminal_completed_month_coverage(
    records: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate the terminal month authority without opening outcome evidence.

    This is the outcome-unsealing boundary.  It deliberately checks only facts
    available from the already-validated decision denominator and the sealed
    completed-month ledger: exact represented months, hash-chain structure,
    activation bindings, counted-session totals, and seal chronology.  Return
    values and outcome timestamps are checked later by
    :func:`validate_terminal_completed_month_coverage`.
    """

    month_rows = validate_completed_month_records(records, protocol)
    decision_rows = validate_decision_records(decisions)
    if not decision_rows:
        if month_rows:
            raise AuditError("completed-month ledger exists without decisions")
        return month_rows

    observed_months = sorted(
        {pd.Timestamp(row["session_date"]).to_period("M") for row in decision_rows}
    )
    represented = list(
        pd.period_range(observed_months[0], observed_months[-1], freq="M")
    )
    if observed_months != represented:
        raise AuditError("decision ledger skips a represented calendar month")
    expected_months = [str(item) for item in represented]
    if [row["completed_month"] for row in month_rows] != expected_months:
        raise AuditError(
            "completed-month ledger must include every represented month through terminal"
        )

    first = decision_rows[0]
    previous_created: pd.Timestamp | None = None
    for month, row in zip(represented, month_rows, strict=True):
        if (
            row["activation_payload_sha256"]
            != first["activation_payload_sha256"]
            or row["activation_receipt_sha256"]
            != first["activation_receipt_sha256"]
        ):
            raise AuditError("completed month activation binding changed")
        month_decisions = [
            decision
            for decision in decision_rows
            if pd.Timestamp(decision["session_date"]).to_period("M") == month
        ]
        if not month_decisions:
            raise AuditError("completed month contains no counted session")
        if row["counted_scheduled_sessions"] != len(month_decisions):
            raise AuditError(
                "completed month counted scheduled sessions do not match decisions"
            )
        created = _aware_timestamp(row["created_at"], "completed month created_at")
        if previous_created is not None and created <= previous_created:
            raise AuditError("completed-month seal chronology is not increasing")
        previous_created = created
        latest_decision = max(
            _aware_timestamp(
                decision["computed_at"], "decision computed_at"
            )
            for decision in month_decisions
        )
        if created <= latest_decision:
            raise AuditError("completed month precedes a counted decision")
        if month != represented[-1]:
            next_month_decisions = [
                decision
                for decision in decision_rows
                if pd.Timestamp(decision["session_date"]).to_period("M")
                == month + 1
            ]
            if not next_month_decisions:
                raise AuditError("completed month lacks its next represented month")
            next_cutoff = pd.Timestamp(
                f"{min(pd.Timestamp(item['session_date']) for item in next_month_decisions).date()}"
                "T08:58:59+09:00"
            )
            if created > next_cutoff:
                raise AuditError("completed month missed the next state cutoff")
    return month_rows


def validate_terminal_completed_month_coverage(
    records: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Require the exact represented-month chain before any gate is exposed."""

    preflight_terminal_completed_month_coverage(records, decisions, protocol)
    month_rows = recompute_completed_month_records(
        records, decisions, outcomes, protocol
    )
    decision_rows = validate_decision_records(decisions)
    outcome_rows = validate_outcome_records(outcomes, decision_rows)
    if not decision_rows:
        return month_rows
    observed_months = sorted(
        {pd.Timestamp(row["session_date"]).to_period("M") for row in decision_rows}
    )
    represented = list(
        pd.period_range(observed_months[0], observed_months[-1], freq="M")
    )
    outcome_by_session = {row["session_date"]: row for row in outcome_rows}
    for month, row in zip(represented, month_rows, strict=True):
        sessions = [
            decision["session_date"]
            for decision in decision_rows
            if pd.Timestamp(decision["session_date"]).to_period("M") == month
        ]
        created = _aware_timestamp(row["created_at"], "completed month created_at")
        latest_outcome = max(
            _aware_timestamp(
                outcome_by_session[session]["computed_at"], "outcome computed_at"
            )
            for session in sessions
        )
        next_month_start = pd.Timestamp((month + 1).start_time).tz_localize(
            "Asia/Tokyo"
        )
        if created <= latest_outcome or created < next_month_start:
            raise AuditError(
                "completed month was not sealed after month-end and all outcomes"
            )
    return month_rows


def recompute_state_schedule(
    *,
    seed_records: Sequence[Mapping[str, Any]],
    completed_months: Sequence[Mapping[str, Any]],
    state_manifests: Mapping[str, Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    first_counted_session_value: Any,
    activation_ready_at: Any,
    activation_payload_sha256: str,
    activation_receipt_sha256: str,
) -> list[dict[str, Any]]:
    """Re-derive every target-month state from seed plus sealed month ledger."""

    if not decisions:
        raise AuditError("state schedule requires counted decisions")
    targets = sorted(
        {pd.Timestamp(row["session_date"]).to_period("M") for row in decisions}
    )
    expected_forward = list(pd.period_range(targets[0], targets[-1], freq="M"))
    observed_forward = [
        pd.Period(str(row["completed_month"]), freq="M")
        for row in completed_months
    ]
    if observed_forward != expected_forward:
        raise AuditError("completed-month ledger does not contain the exact forward set")
    if set(state_manifests) != {str(item) for item in targets}:
        raise AuditError("state manifest set does not equal represented target months")

    activation_ready = _aware_timestamp(activation_ready_at, "activation_ready_at")
    first_counted = pd.Timestamp(first_counted_session_value).normalize()
    outcome_by_session = {row["session_date"]: row for row in outcomes}
    month_values: dict[str, dict[str, Any]] = {
        str(row["completed_month"]): {
            "complete_pair_days": int(row["complete_pair_days"]),
            "monthly_median_rank1_minus_rank2_pct": float(
                row["monthly_median_rank1_minus_rank2_pct"]
            ),
            "created_at": None,
        }
        for row in seed_records
    }
    completed_by_month = {
        str(row["completed_month"]): dict(row) for row in completed_months
    }
    for month, row in completed_by_month.items():
        if (
            row["activation_payload_sha256"] != activation_payload_sha256
            or row["activation_receipt_sha256"] != activation_receipt_sha256
        ):
            raise AuditError("completed month does not bind the active activation")
        created = _aware_timestamp(row["created_at"], "completed month created_at")
        sessions = [
            item["session_date"]
            for item in decisions
            if str(pd.Timestamp(item["session_date"]).to_period("M")) == month
        ]
        outcome_computed = [
            _aware_timestamp(
                outcome_by_session[session]["computed_at"],
                "outcome computed_at",
            )
            for session in sessions
        ]
        if outcome_computed and created < max(outcome_computed):
            raise AuditError("completed month predates a counted outcome record")
        next_month = pd.Period(month, freq="M") + 1
        next_cutoff_session = _month_first_counted_session(next_month, first_counted)
        next_cutoff = pd.Timestamp(
            f"{next_cutoff_session.date()}T08:58:59+09:00"
        )
        if created > next_cutoff:
            raise AuditError("completed month missed the next state cutoff")
        month_values[month] = {
            "complete_pair_days": int(row["complete_pair_days"]),
            "monthly_median_rank1_minus_rank2_pct": row[
                "monthly_median_rank1_minus_rank2_pct"
            ],
            "created_at": created,
        }

    schedule: list[dict[str, Any]] = []
    decisions_by_month: dict[str, list[Mapping[str, Any]]] = {}
    for row in decisions:
        decisions_by_month.setdefault(row["session_date"][:7], []).append(row)
    for target in targets:
        target_text = str(target)
        prior_months = [str(target - offset) for offset in (3, 2, 1)]
        if any(month not in month_values for month in prior_months):
            raise AuditError("state schedule is missing an immediate prior month")
        counts = [int(month_values[month]["complete_pair_days"]) for month in prior_months]
        medians = [
            month_values[month]["monthly_median_rank1_minus_rank2_pct"]
            for month in prior_months
        ]
        available = all(
            count >= MIN_COMPLETE_PAIRS_PER_MONTH and median is not None
            for count, median in zip(counts, medians, strict=True)
        )
        state_value = (
            float(np.median(np.asarray(medians, dtype=float))) if available else None
        )
        selected_rank = (
            1
            if state_value is not None and state_value > 0.0
            else 2
            if state_value is not None and state_value < 0.0
            else None
        )
        manifest = state_manifests[target_text]
        if (
            manifest["activation_payload_sha256"] != activation_payload_sha256
            or manifest["activation_receipt_sha256"] != activation_receipt_sha256
            or manifest["three_prior_calendar_months"] != prior_months
            or manifest["three_complete_pair_day_counts"] != counts
            or bool(manifest["state_available"]) != available
            or manifest["selected_source_rank"] != selected_rank
        ):
            raise AuditError("state manifest differs from seed/month-ledger schedule")
        observed_medians = manifest["three_month_medians_pct"]
        if len(observed_medians) != 3 or any(
            (expected is None) != (observed is None)
            or (
                expected is not None
                and not _canonical_float_equal(observed, float(expected))
            )
            for observed, expected in zip(observed_medians, medians, strict=True)
        ):
            raise AuditError("state manifest monthly medians differ from ledger")
        observed_state = manifest["state_value_pct"]
        if (state_value is None) != (observed_state is None) or (
            state_value is not None
            and not _canonical_float_equal(observed_state, state_value)
        ):
            raise AuditError("state manifest value differs from ledger")
        created = _aware_timestamp(manifest["created_at"], "state created_at")
        latest_prior_creation = max(
            (
                month_values[month]["created_at"]
                for month in prior_months
                if month_values[month]["created_at"] is not None
            ),
            default=activation_ready,
        )
        if created < max(activation_ready, latest_prior_creation):
            raise AuditError("state manifest predates activation/prior-month seal")
        cutoff_session = _month_first_counted_session(target, first_counted)
        if created > pd.Timestamp(f"{cutoff_session.date()}T08:58:59+09:00"):
            raise AuditError("state manifest missed its first counted cutoff")
        expected_decision_state = (
            manifest["state_manifest_sha256"],
            prior_months,
            counts,
            observed_medians,
            available,
            observed_state,
            selected_rank,
        )
        for decision in decisions_by_month[target_text]:
            observed_decision_state = (
                decision["state_manifest_sha256"],
                decision["three_prior_calendar_months"],
                decision["three_complete_pair_day_counts"],
                decision["three_month_medians_pct"],
                decision["state_available"],
                decision["state_value_pct"],
                decision["selected_source_rank"],
            )
            if observed_decision_state != expected_decision_state:
                raise AuditError("decision does not bind the recomputed monthly state")
        schedule.append(
            {
                "target_month": target_text,
                "three_prior_calendar_months": prior_months,
                "three_complete_pair_day_counts": counts,
                "three_month_medians_pct": medians,
                "state_available": available,
                "state_value_pct": state_value,
                "selected_source_rank": selected_rank,
            }
        )
    return schedule


def moving_block_means(
    values: np.ndarray,
    *,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    samples: int = BOOTSTRAP_SAMPLES,
    random_state: int = BOOTSTRAP_RANDOM_STATE,
    batch_size: int = 1_000,
) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise AuditError("bootstrap input must be a finite vector")
    if block_length < 2 or len(data) < block_length or samples < 100:
        raise AuditError("invalid moving-block bootstrap parameters")
    blocks = math.ceil(len(data) / block_length)
    max_start = len(data) - block_length + 1
    offsets = np.arange(block_length, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    output = np.empty(samples, dtype=float)
    position = 0
    while position < samples:
        size = min(batch_size, samples - position)
        starts = rng.integers(0, max_start, size=(size, blocks))
        indices = (starts[..., None] + offsets).reshape(size, -1)
        output[position : position + size] = data[
            indices[:, : len(data)]
        ].mean(axis=1)
        position += size
    return output


def load_registered_calendar(path: str | Path = DEFAULT_CALENDAR) -> pd.DatetimeIndex:
    calendar_bytes = _stable_plain_file_bytes(path, label="registered TSE calendar")
    if hashlib.sha256(calendar_bytes).hexdigest() != CALENDAR_SHA256:
        raise AuditError("registered TSE calendar SHA-256 mismatch")
    frame = pd.read_csv(io.BytesIO(calendar_bytes), dtype=str, keep_default_na=False)
    expected_columns = [
        "session_date",
        "market",
        "source_url",
        "source_retrieved_at",
    ]
    if frame.columns.tolist() != expected_columns or len(frame) != 343:
        raise AuditError("registered TSE calendar schema/count changed")
    sessions = pd.to_datetime(frame["session_date"], errors="coerce")
    if sessions.isna().any() or not sessions.is_monotonic_increasing or sessions.duplicated().any():
        raise AuditError("registered TSE calendar dates are invalid")
    if (
        frame["market"].ne("TSE").any()
        or frame["source_url"]
        .ne("https://www.jpx.co.jp/english/corporate/about-jpx/calendar/index.html")
        .any()
        or frame["source_retrieved_at"].ne("2026-08-04").any()
    ):
        raise AuditError("registered TSE calendar source fields changed")
    index = pd.DatetimeIndex(sessions)
    if index.min() != pd.Timestamp("2026-08-05") or index.max() != pd.Timestamp(
        "2027-12-30"
    ):
        raise AuditError("registered TSE calendar bounds changed")
    return index


def first_counted_session(
    *,
    workflow_run_updated_at: Any,
    workflow_run_observed_at: Any,
    calendar: pd.DatetimeIndex,
    not_before_session: str = "2026-08-05",
) -> pd.Timestamp:
    updated = pd.Timestamp(workflow_run_updated_at)
    observed = pd.Timestamp(workflow_run_observed_at)
    if updated.tzinfo is None or observed.tzinfo is None:
        raise AuditError("receipt-workflow timestamps must be timezone-aware")
    if observed < updated:
        raise AuditError("receipt workflow observation predates run completion")
    not_before = pd.Timestamp(not_before_session)
    for session in calendar[calendar >= not_before]:
        cutoff = pd.Timestamp(
            f"{session.date()}T08:58:59+09:00"
        )
        if cutoff > updated:
            if observed >= cutoff:
                raise AuditError(
                    "receipt workflow was observed after the fixed first-session cutoff"
                )
            return session
    raise AuditError("registered calendar has no eligible first counted session")


def deterministic_terminal_session(
    first: pd.Timestamp,
    calendar: pd.DatetimeIndex,
) -> pd.Timestamp:
    eligible = calendar[calendar >= pd.Timestamp(first)]
    if len(eligible) == 0 or eligible[0] != pd.Timestamp(first):
        raise AuditError("first counted session is outside the registered calendar")
    periods = eligible.to_period("M")
    for month in periods.unique():
        through = eligible[periods <= month]
        if (
            len(through) >= MIN_FORWARD_SESSIONS
            and through.to_period("M").nunique() >= MIN_FORWARD_MONTHS
        ):
            return through[-1]
    raise AuditError("registered calendar does not cover the deterministic terminal")


def paired_bootstrap(candidate: pd.Series, control: pd.Series) -> dict[str, Any]:
    if len(candidate) != len(control) or not candidate.index.equals(control.index):
        raise AuditError("paired bootstrap indexes differ")
    delta = candidate.to_numpy(dtype=float) - control.to_numpy(dtype=float)
    means = moving_block_means(delta)
    return {
        "observations": int(len(delta)),
        "block_length_sessions": BOOTSTRAP_BLOCK_LENGTH,
        "samples": BOOTSTRAP_SAMPLES,
        "random_state": BOOTSTRAP_RANDOM_STATE,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "candidate_mean_pct": float(candidate.mean()),
        "control_mean_pct": float(control.mean()),
        "point_estimate_delta_pct": float(delta.mean()),
        "one_sided_lower_delta_pct": float(
            np.quantile(means, 1.0 - BOOTSTRAP_CONFIDENCE)
        ),
        "bootstrap_standard_error_delta_pct": float(means.std(ddof=1)),
    }


def _evaluation_frame(
    decisions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    decision_rows = validate_decision_records(decisions)
    outcome_rows = validate_outcome_records(outcomes, decision_rows)
    records: list[dict[str, Any]] = []
    for decision, outcome in zip(decision_rows, outcome_rows, strict=True):
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
    frame = pd.DataFrame(records).set_index("session_date")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise AuditError("evaluation sessions are not canonical")
    return frame


def evaluate_candidate(
    decisions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    calendar: pd.DatetimeIndex | None = None,
    completed_months: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    decision_rows = validate_decision_records(decisions)
    outcome_rows = validate_outcome_records(outcomes, decision_rows)
    sessions = len(decision_rows)
    if not decision_rows:
        return {
            "status": "awaiting_terminal_evaluation",
            "scheduled_sessions": 0,
            "outcome_records": 0,
            "represented_calendar_months": 0,
            "minimum_scheduled_sessions": MIN_FORWARD_SESSIONS,
            "minimum_calendar_months": MIN_FORWARD_MONTHS,
            "deterministic_terminal_session": None,
            "gate_evaluated": False,
        }
    observed = pd.DatetimeIndex(
        pd.to_datetime([row["session_date"] for row in decision_rows])
    )
    if not observed.is_monotonic_increasing or observed.has_duplicates:
        raise AuditError("evaluation sessions are not canonical")
    represented_months = int(observed.to_period("M").nunique())
    terminal: pd.Timestamp | None = None
    if calendar is not None:
        first = observed[0]
        terminal = deterministic_terminal_session(first, calendar)
        expected = calendar[(calendar >= first) & (calendar <= terminal)]
        if len(observed) > len(expected) or not observed.equals(expected[: len(observed)]):
            raise AuditError("decision denominator differs from the registered calendar")
        terminal_ready = observed[-1] == terminal
    else:
        terminal_ready = (
            sessions >= MIN_FORWARD_SESSIONS
            and represented_months >= MIN_FORWARD_MONTHS
        )
    if not terminal_ready:
        return {
            "status": "awaiting_terminal_evaluation",
            "scheduled_sessions": sessions,
            "outcome_records": len(outcome_rows),
            "represented_calendar_months": represented_months,
            "minimum_scheduled_sessions": MIN_FORWARD_SESSIONS,
            "minimum_calendar_months": MIN_FORWARD_MONTHS,
            "deterministic_terminal_session": (
                None if terminal is None else str(terminal.date())
            ),
            "gate_evaluated": False,
        }

    if completed_months is None:
        raise AuditError(
            "terminal completed-month ledger is required before gate evaluation"
        )
    validate_terminal_completed_month_coverage(
        completed_months,
        decision_rows,
        outcome_rows,
        read_json(DEFAULT_PROTOCOL),
    )

    # This is the first construction of a performance-bearing frame.
    frame = _evaluation_frame(decision_rows, outcome_rows)

    net40 = frame["candidate_net40"]
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
    total_selected = int(len(selected))
    unique_codes = int(len(counts))
    maximum_share = float(counts.iloc[0] / total_selected) if total_selected else 1.0
    top10_share = (
        float(counts.head(10).sum() / total_selected) if total_selected else 1.0
    )
    executed = int(frame["executed"].sum())
    execution_fraction = float(executed / sessions)
    paired = paired_bootstrap(net40, frame["c00_net40"])
    gate_checks = {
        "net40_mean_positive": float(net40.mean()) > 0.0,
        "net40_median_positive": float(net40.median()) > 0.0,
        "net60_mean_positive": float(frame["candidate_net60"].mean()) > 0.0,
        "both_fixed_slices_net40_positive": all(value > 0.0 for value in slices.values()),
        "positive_months_net40_at_least_ceil_75pct": int(monthly.gt(0.0).sum())
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
        "unique_codes_at_least": unique_codes >= 40,
        "maximum_code_share_at_most": maximum_share <= 0.05,
        "top10_code_share_at_most": top10_share <= 0.25,
        "executed_days_at_least_ceil_90pct_scheduled": executed
        >= math.ceil(0.90 * sessions),
        "executed_slot_fraction_at_least": execution_fraction >= 0.8,
    }
    passed = all(gate_checks.values())
    winner = SH01 if passed else None
    return {
        "status": (
            "forward_passed_one_v19_research_nominee"
            if passed
            else "forward_rejected_candidate"
        ),
        "scheduled_sessions": sessions,
        "represented_calendar_months": represented_months,
        "deterministic_terminal_session": (
            None if terminal is None else str(terminal.date())
        ),
        "gate_evaluated": True,
        "cost_metrics": {
            "20": float(frame["candidate_net20"].mean()),
            "40": float(net40.mean()),
            "60": float(frame["candidate_net60"].mean()),
        },
        "control_cost_metrics": {
            C00_TOP1: {
                "20": float(frame["c00_net20"].mean()),
                "40": float(frame["c00_net40"].mean()),
                "60": float(frame["c00_net60"].mean()),
            },
            C02_TOP2: {
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
        "executed_slot_fraction": execution_fraction,
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


class AuditRecorder:
    def __init__(self) -> None:
        self.checks: dict[str, bool] = {}
        self.discrepancies: list[str] = []
        self.max_abs_numeric_difference = 0.0

    def check(
        self,
        name: str,
        passed: bool,
        *,
        observed: Any | None = None,
        expected: Any | None = None,
    ) -> None:
        self.checks[name] = bool(passed)
        if not passed:
            self.discrepancies.append(
                f"{name}: observed={observed!r}, expected={expected!r}"
            )

    def compare_nested(self, name: str, observed: Any, expected: Any) -> None:
        before = len(self.discrepancies)
        self._compare(name, _json_safe(observed), _json_safe(expected))
        self.checks[name] = len(self.discrepancies) == before

    def _compare(self, path: str, observed: Any, expected: Any) -> None:
        if isinstance(expected, dict):
            if not isinstance(observed, dict):
                self.discrepancies.append(
                    f"{path}: observed type={type(observed).__name__}, expected=dict"
                )
                return
            if set(observed) != set(expected):
                self.discrepancies.append(
                    f"{path}: keys observed={sorted(observed)}, expected={sorted(expected)}"
                )
            for key in sorted(set(observed) & set(expected)):
                self._compare(f"{path}.{key}", observed[key], expected[key])
            return
        if isinstance(expected, list):
            if not isinstance(observed, list) or len(observed) != len(expected):
                self.discrepancies.append(
                    f"{path}: observed={observed!r}, expected={expected!r}"
                )
                return
            for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
                self._compare(f"{path}[{index}]", left, right)
            return
        numeric = (int, float, np.integer, np.floating)
        if (
            isinstance(expected, numeric)
            and not isinstance(expected, bool)
            and isinstance(observed, numeric)
            and not isinstance(observed, bool)
        ):
            difference = abs(float(observed) - float(expected))
            if math.isfinite(difference):
                self.max_abs_numeric_difference = max(
                    self.max_abs_numeric_difference, difference
                )
            if not math.isclose(
                float(observed),
                float(expected),
                rel_tol=0.0,
                abs_tol=NUMERIC_TOLERANCE,
            ):
                self.discrepancies.append(
                    f"{path}: observed={observed!r}, expected={expected!r}"
                )
            return
        if observed != expected:
            self.discrepancies.append(
                f"{path}: observed={observed!r}, expected={expected!r}"
            )


def _manifest_map(
    directory: str | Path,
    protocol: Mapping[str, Any],
    *,
    first_counted_session: Any | None = None,
) -> dict[str, dict[str, Any]]:
    root = _require_plain_directory(directory, label="state manifest directory")
    output: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        raw = read_json(path)
        month = str(raw.get("target_month"))
        if path.name != f"{month}.json":
            raise AuditError("state manifest filename does not equal target month")
        value = validate_state_manifest(
            raw,
            protocol,
            first_counted_session=first_counted_session,
        )
        if month in output:
            raise AuditError("duplicate target-month state manifest")
        output[month] = value
    return output


def _source_manifest_map(
    directory: str | Path,
    protocol: Mapping[str, Any],
    *,
    predictor_raw_store_root: str | Path | None = None,
    external_identity_registry: dict[tuple[int, int], str] | None = None,
) -> dict[str, dict[str, Any]]:
    root = _require_plain_directory(directory, label="source manifest directory")
    output: dict[str, dict[str, Any]] = {}
    observed_targets: set[str] = set()
    for path in sorted(root.glob("*.json")):
        raw = read_json(path)
        target = str(raw.get("target_session"))
        if path.name != f"{target}.json":
            raise AuditError("source manifest filename does not equal target session")
        value = validate_source_manifest(
            raw,
            session_date=target,
            protocol=protocol,
            predictor_raw_store_root=predictor_raw_store_root,
            external_identity_registry=external_identity_registry,
        )
        digest = value["source_manifest_sha256"]
        if digest in output or target in observed_targets:
            raise AuditError("duplicate source manifest hash/target")
        output[digest] = value
        observed_targets.add(target)
    return output


def validate_predictor_evidence(
    source_manifests: Mapping[str, Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    predictor_raw_store_root: str | Path,
    fold_manifests_by_month: Mapping[str, Mapping[str, Any]],
    fold_model_directory: str | Path,
    scores: pd.DataFrame,
    external_identity_registry: dict[tuple[int, int], str] | None = None,
) -> dict[str, Any]:
    """Terminal clean-room replay from external predictor PDFs through C00 top2."""

    if (
        not isinstance(scores, pd.DataFrame)
        or scores.columns.tolist() != list(SCORE_FIELDS)
    ):
        raise AuditError("score ledger must retain the exact registered header")

    parser_path = ROOT / protocol["source_contract"]["forward_daily"]["parser_path"]
    parser_sha = protocol["source_contract"]["forward_daily"]["parser_sha256"]
    if sha256_file(parser_path) != parser_sha:
        raise AuditError("predictor parser source SHA-256 changed")
    object_union: dict[str, dict[str, Any]] = {}
    for manifest in source_manifests.values():
        for key, file_name, url, count, digest in zip(
            manifest["source_object_keys"],
            manifest["source_files"],
            manifest["source_urls"],
            manifest["source_byte_counts"],
            manifest["source_sha256"],
            strict=True,
        ):
            metadata = {
                "object_key": key,
                "file": file_name,
                "url": url,
                "byte_count": int(count),
                "sha256": digest,
            }
            previous = object_union.setdefault(key, metadata)
            if previous != metadata:
                raise AuditError("predictor object key has conflicting metadata")
    ordered_union = [object_union[key] for key in sorted(object_union)]

    try:
        from tse_session_ranker.data.common import merge_daily_prices
        from tse_session_ranker.data.jpx import parse_jpx_text
    except Exception as exc:  # pragma: no cover - installation integrity
        raise AuditError("bound predictor parser is unavailable") from exc
    parsed_by_key: dict[str, tuple[pd.DataFrame, int]] = {}
    with tempfile.TemporaryDirectory(prefix="v18-predictor-audit-") as temporary:
        temporary_root = Path(temporary)
        for metadata in ordered_union:
            raw_bytes = _read_external_object_bytes(
                predictor_raw_store_root,
                metadata["object_key"],
                required_prefix="model_v18_shoulder_state/predictor/",
                identity_registry=external_identity_registry,
            )
            if (
                len(raw_bytes) != metadata["byte_count"]
                or hashlib.sha256(raw_bytes).hexdigest() != metadata["sha256"]
            ):
                raise AuditError("predictor object union byte binding changed")
            pdf_path = temporary_root / Path(metadata["file"])
            text_path = temporary_root / Path(metadata["file"]).with_suffix(".txt")
            if pdf_path.exists() or text_path.exists():
                raise AuditError("predictor source filenames collide during reparse")
            pdf_path.write_bytes(raw_bytes)
            try:
                _locked_pdf_to_text(pdf_path, text_path)
                frame, report = parse_jpx_text(text_path)
            except Exception as exc:
                raise AuditError("predictor raw object could not be reparsed") from exc
            if int(report.rejected_rows) != 0:
                raise AuditError("predictor reparse rejected rows")
            parsed_dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
            if parsed_dates.isna().any() or parsed_dates.empty:
                raise AuditError("predictor reparse has no valid source dates")
            kind = Path(metadata["object_key"]).parts[2]
            expected_source_date = _predictor_source_date(metadata["file"], kind)
            if kind == "daily":
                if set(parsed_dates) != {expected_source_date}:
                    raise AuditError("predictor daily filename/date binding changed")
            elif not parsed_dates.dt.to_period("M").eq(
                expected_source_date.to_period("M")
            ).all():
                raise AuditError("predictor warmup filename/month binding changed")
            parsed_by_key[metadata["object_key"]] = (frame, int(report.rejected_rows))

    scores_copy = scores.copy()
    scores_copy["session_date"] = pd.to_datetime(
        scores_copy["session_date"], errors="coerce"
    ).dt.normalize()
    if scores_copy["session_date"].isna().any():
        raise AuditError("score output contains invalid sessions")
    refitted_months: set[str] = set()
    manifest_pairs: list[dict[str, Any]] = []
    semantic_pairs = {
        field: []
        for field in (
            "parsed_panel_semantic_sha256",
            "g0_panel_semantic_sha256",
            "common_universe_semantic_sha256",
            "target_date_scoring_input_semantic_sha256",
        )
    }
    for decision in decisions:
        digest = decision["source_manifest_sha256"]
        manifest = source_manifests.get(digest)
        if manifest is None or manifest["target_session"] != decision["session_date"]:
            raise AuditError("decision does not bind its predictor source manifest")
        target = pd.Timestamp(decision["session_date"])
        manifest_pairs.append(
            {"target_session": decision["session_date"], "source_manifest_sha256": digest}
        )
        for field, pairs in semantic_pairs.items():
            pairs.append(
                {"target_session": decision["session_date"], field: manifest[field]}
            )
        if not manifest["source_complete"]:
            continue
        frames = [parsed_by_key[key][0] for key in manifest["source_object_keys"]]
        if not frames:
            raise AuditError("complete predictor manifest has no reparsed frames")
        try:
            parsed = merge_daily_prices(frames)
        except Exception as exc:
            raise AuditError("predictor parser panels could not be merged") from exc
        if len(parsed) != int(manifest["parsed_row_count"]):
            raise AuditError("predictor parsed row count does not recompute")
        if parsed[["date", "code"]].duplicated().any():
            raise AuditError("predictor merged panel has duplicate date/code rows")
        if parsed["date"].max() != pd.Timestamp(
            manifest["latest_required_source_session"]
        ):
            raise AuditError("predictor raw union does not end at exact D-1")
        parsed_hash = semantic_rows_sha256(parsed, PARSED_PANEL_COLUMNS)
        if parsed_hash != manifest["parsed_panel_semantic_sha256"]:
            raise AuditError("predictor parsed-panel semantic hash differs")
        panel = build_clean_room_g0_panel(parsed, target)
        through = panel.loc[panel["date"].le(target)].copy()
        g0_columns = (
            "date",
            "code",
            "name",
            "oc_return_pct",
            "common_training_eligible",
            "common_score_eligible",
            "feature_source_max_date",
            *C00_FEATURES,
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
            *C00_FEATURES,
        )
        target_rows = through.loc[through["date"].eq(target)]
        recomputed_semantics = {
            "g0_panel_semantic_sha256": semantic_rows_sha256(through, g0_columns),
            "common_universe_semantic_sha256": semantic_rows_sha256(
                through, universe_columns
            ),
            "target_date_scoring_input_semantic_sha256": semantic_rows_sha256(
                target_rows, scoring_columns
            ),
        }
        for field, expected in recomputed_semantics.items():
            if manifest[field] != expected:
                raise AuditError(f"predictor {field} differs")
        month = str(target.to_period("M"))
        fold = fold_manifests_by_month.get(month)
        if fold is None:
            if decision["c00_fold_manifest_sha256"] is not None:
                raise AuditError("decision names an absent clean-room fold")
            continue
        if decision["c00_fold_manifest_sha256"] != fold["fold_manifest_sha256"]:
            raise AuditError("decision fold hash differs from target-month fold")
        bundle_path = Path(fold_model_directory) / f"{month}.json"
        bundle = validate_fold_model_bundle(
            read_json(bundle_path),
            protocol,
            first_counted_session=decisions[0]["session_date"],
        )
        if month not in refitted_months:
            validate_clean_room_fold_fit(panel, month, fold, bundle, protocol)
            refitted_months.add(month)
        scoring = target_rows.loc[
            target_rows["common_score_eligible"].fillna(False).astype(bool)
        ].copy()
        expected_scores = reconstruct_c00_scores(
            bundle, scoring.loc[:, list(C00_FEATURES)], protocol
        )
        ranked = scoring.loc[:, ["code", "name", "feature_source_max_date"]].copy()
        ranked["model_score"] = expected_scores
        ranked = ranked.sort_values(
            ["model_score", "code"], ascending=[False, True], kind="stable"
        ).head(2)
        observed = scores_copy.loc[scores_copy["session_date"].eq(target)].sort_values(
            "source_rank", kind="stable"
        )
        if len(ranked) < 2:
            if decision["model_complete"]:
                raise AuditError("decision claims a complete model without two scores")
            if len(observed):
                raise AuditError("score output contains a fail-closed model pair")
            continue
        if not decision["model_complete"] or len(observed) != 2:
            raise AuditError("clean-room top2 exists but decision/score output is incomplete")
        ranked = ranked.reset_index(drop=True)
        observed = observed.reset_index(drop=True)
        for rank in (1, 2):
            clean = ranked.iloc[rank - 1]
            sealed = observed.iloc[rank - 1]
            if (
                int(sealed["source_rank"]) != rank
                or str(sealed["code"]) != str(clean["code"])
                or str(sealed["name"]) != str(clean["name"])
                or not math.isclose(
                    float(sealed["model_score"]),
                    float(clean["model_score"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or sealed["source_manifest_sha256"] != digest
                or sealed["c00_fold_manifest_sha256"]
                != fold["fold_manifest_sha256"]
            ):
                raise AuditError("clean-room score-sorted top2 differs from score output")
        if (
            str(decision["c00_rank1_code"]) != str(ranked.iloc[0]["code"])
            or str(decision["c02_rank2_code"]) != str(ranked.iloc[1]["code"])
            or not math.isclose(
                float(decision["c00_rank1_score"]),
                float(ranked.iloc[0]["model_score"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(decision["c02_rank2_score"]),
                float(ranked.iloc[1]["model_score"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise AuditError("clean-room top2 differs from frozen decision")

    represented_months = sorted({row["session_date"][:7] for row in decisions})
    fold_manifest_pairs = [
        {
            "target_month": month,
            "fold_manifest_sha256": (
                None
                if month not in fold_manifests_by_month
                else fold_manifests_by_month[month]["fold_manifest_sha256"]
            ),
        }
        for month in represented_months
    ]
    fold_bundle_pairs = [
        {
            "target_month": month,
            "fold_model_bundle_file_sha256": (
                None
                if month not in fold_manifests_by_month
                else fold_manifests_by_month[month][
                    "fold_model_bundle_file_sha256"
                ]
            ),
        }
        for month in represented_months
    ]
    prior_v17 = protocol["prior_result_binding"]["v17"]
    return {
        "runtime_lock_sha256": RUNTIME_LOCK_SHA256,
        "predictor_source_manifest_set_sha256": canonical_json_sha256(manifest_pairs),
        "predictor_raw_source_set_sha256": canonical_json_sha256(ordered_union),
        "predictor_unique_raw_object_count": len(ordered_union),
        "predictor_parser_sha256": parser_sha,
        "predictor_parsed_panel_semantic_set_sha256": canonical_json_sha256(
            semantic_pairs["parsed_panel_semantic_sha256"]
        ),
        "predictor_g0_panel_semantic_set_sha256": canonical_json_sha256(
            semantic_pairs["g0_panel_semantic_sha256"]
        ),
        "predictor_common_universe_semantic_set_sha256": canonical_json_sha256(
            semantic_pairs["common_universe_semantic_sha256"]
        ),
        "predictor_target_date_scoring_input_semantic_set_sha256": canonical_json_sha256(
            semantic_pairs["target_date_scoring_input_semantic_sha256"]
        ),
        "c00_fold_manifest_set_sha256": canonical_json_sha256(fold_manifest_pairs),
        "c00_fold_model_bundle_file_set_sha256": canonical_json_sha256(
            fold_bundle_pairs
        ),
        "v17_c00_protocol_sha256": prior_v17["protocol_sha256"],
        "v17_c00_runner_sha256": prior_v17["runner_sha256"],
    }


def _outcome_manifest_map(
    directory: str | Path,
    decisions: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    raw_store_root: str | Path,
    external_identity_registry: dict[tuple[int, int], str] | None = None,
) -> dict[str, dict[str, Any]]:
    root = _require_plain_directory(directory, label="outcome manifest directory")
    decisions_by_session = {row["session_date"]: row for row in decisions}
    by_hash: dict[str, dict[str, Any]] = {}
    observed_sessions: set[str] = set()
    for path in sorted(root.glob("*.json")):
        raw = read_json(path)
        session = str(raw.get("target_session"))
        if path.name != f"{session}.json" or session not in decisions_by_session:
            raise AuditError("outcome manifest filename/session is unregistered")
        value = reparse_outcome_manifest(
            raw,
            decisions_by_session[session],
            protocol,
            raw_store_root=raw_store_root,
            external_identity_registry=external_identity_registry,
        )
        digest = value["outcome_manifest_sha256"]
        if digest in by_hash or session in observed_sessions:
            raise AuditError("duplicate outcome manifest hash/session")
        by_hash[digest] = value
        observed_sessions.add(session)
    if observed_sessions != set(decisions_by_session):
        raise AuditError("outcome manifests do not cover every counted session")
    return by_hash


def _fold_artifact_maps(
    manifest_directory: str | Path,
    model_directory: str | Path,
    protocol: Mapping[str, Any],
    *,
    runner_sha256: str,
    first_counted_session: Any | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    manifests_root = _require_plain_directory(
        manifest_directory, label="fold manifest directory"
    )
    models_root = _require_plain_directory(
        model_directory, label="fold model directory"
    )
    by_hash: dict[str, dict[str, Any]] = {}
    by_month: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(manifests_root.glob("*.json")):
        raw_manifest = read_json(manifest_path)
        month = str(raw_manifest.get("target_month"))
        if manifest_path.name != f"{month}.json":
            raise AuditError("fold manifest filename does not equal target month")
        model_path = models_root / f"{month}.json"
        model_bytes = _stable_plain_file_bytes(
            model_path, label=f"fold model bundle {month}"
        )
        model_file_sha256 = hashlib.sha256(model_bytes).hexdigest()
        if model_file_sha256 != raw_manifest.get("fold_model_bundle_file_sha256"):
            raise AuditError("fold model bundle exact file SHA mismatch")
        value = validate_fold_manifest(
            raw_manifest,
            _parse_json_object_bytes(
                model_bytes, label=f"fold model bundle {month}"
            ),
            protocol,
            runner_sha256=runner_sha256,
            first_counted_session=first_counted_session,
        )
        digest = value["fold_manifest_sha256"]
        if digest in by_hash or month in by_month:
            raise AuditError("duplicate fold manifest hash/month")
        by_hash[digest] = value
        by_month[month] = value
    extra_models = {
        path.stem for path in models_root.glob("*.json")
    } - set(by_month)
    if extra_models:
        raise AuditError(f"unbound fold model bundles exist: {sorted(extra_models)}")
    return by_hash, by_month


def _stable_plain_file_bytes(path: str | Path, *, label: str) -> bytes:
    """Read one pinned single-link regular file without following a final link."""

    target = Path(path)
    try:
        before_path = os.lstat(target)
    except FileNotFoundError:
        raise
    if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
        raise AuditError(f"{label} is not a single-link regular file")
    try:
        descriptor = os.open(
            target,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise AuditError(f"{label} cannot be pinned for hashing") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino)
            != (before_path.st_dev, before_path.st_ino)
        ):
            raise AuditError(f"{label} changed before reading")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
        except OSError as exc:
            raise AuditError(f"{label} cannot be pinned for reading") from exc
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        immutable = (
            "st_dev",
            "st_ino",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            any(getattr(before, field) != getattr(after, field) for field in immutable)
            or byte_count != before.st_size
        ):
            raise AuditError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def _stable_plain_file_sha256(path: str | Path, *, label: str) -> str:
    """Hash exact bytes from a pinned single-link regular-file snapshot."""

    return hashlib.sha256(_stable_plain_file_bytes(path, label=label)).hexdigest()


def _result_status_discriminator(payload: bytes) -> str:
    """Read only the canonical top-level status line before unblinding."""

    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AuditError("canonical result status is not UTF-8") from exc
    matches = re.findall(r'^  "status": "([a-z0-9_]+)",?$', text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise AuditError("canonical result has no unique status discriminator")
    return matches[0]


def validate_abort_result(
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    runtime_lock: Mapping[str, Any] | None = None,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    runner_path: str | Path = DEFAULT_RUNNER,
    activation_payload_path: str | Path = DEFAULT_ACTIVATION_PAYLOAD,
    activation_receipt_path: str | Path = DEFAULT_ACTIVATION_RECEIPT,
    artifact_paths: Mapping[str, str | Path] | None = None,
    result_path: str | Path | None = None,
) -> dict[str, Any]:
    """Independently validate the outcome-blind canonical integrity abort."""

    contract = protocol["result_contract"]
    abort_contract = contract["abort_result_contract"]
    required_top = set(contract["required_top_level_fields"])
    if not isinstance(result, Mapping) or set(result) != required_top:
        raise AuditError("abort result top-level fields changed")
    value = dict(result)
    if (
        value["schema_version"] != 1
        or value["protocol_id"] != protocol["protocol_id"]
        or value["protocol_sha256"] != sha256_file(protocol_path)
        or value["runner_sha256"] != sha256_file(runner_path)
        or value["status"] != abort_contract["canonical_status"]
    ):
        raise AuditError("abort result identity changed")
    reason = value["failure_reason"]
    if not isinstance(reason, str) or re.fullmatch(
        r"[a-z][a-z0-9_]{2,127}", reason
    ) is None or reason not in contract["abort_failure_reason_values"]:
        raise AuditError("abort result failure reason is not a registered machine reason")
    stage = value["integrity_stage"]
    if stage not in contract["abort_integrity_stage_values"]:
        raise AuditError("abort result integrity stage changed")
    if reason not in contract["abort_stage_reason_values"].get(stage, []):
        raise AuditError("abort failure reason is not registered for integrity stage")
    if value["authority"] != contract["authority_values"]:
        raise AuditError("abort result authority changed")
    for field in ("forward_period", "state_months", "models", "candidate_gate"):
        if value[field] is not None:
            raise AuditError(f"abort result exposed sealed {field}")
    expected_decision = {
        "research_nominee": None,
        "failure_reason": reason,
        "integrity_stage": stage,
    }
    if value["decision"] != expected_decision:
        raise AuditError("abort result decision shape changed")

    expected_input = {field: None for field in contract["required_input_fields"]}
    expected_input.update(
        {
            "runtime_lock_sha256": protocol["runtime_lock_contract"]["file_sha256"],
            "predictor_parser_sha256": protocol["source_contract"]["forward_daily"][
                "parser_sha256"
            ],
            "v17_c00_protocol_sha256": protocol["prior_result_binding"]["v17"][
                "protocol_sha256"
            ],
            "v17_c00_runner_sha256": protocol["prior_result_binding"]["v17"][
                "runner_sha256"
            ],
        }
    )
    if value["input"] != expected_input:
        raise AuditError("abort result input bindings changed or expose derived content")

    payload_value: Mapping[str, Any] | None = None
    payload_sha: str | None = None
    payload_path = Path(activation_payload_path)
    if payload_path.is_file() and not payload_path.is_symlink():
        try:
            payload_value = read_json(payload_path)
            payload_sha = validate_activation_payload(payload_value, protocol)
        except (AuditError, OSError, ValueError, json.JSONDecodeError):
            payload_value = None
            payload_sha = None
    receipt_sha: str | None = None
    receipt_path = Path(activation_receipt_path)
    if (
        payload_value is not None
        and receipt_path.is_file()
        and not receipt_path.is_symlink()
    ):
        try:
            receipt_sha = validate_activation_receipt(
                read_json(receipt_path),
                payload_value,
                protocol,
                payload_file_path=payload_path,
            )
        except (AuditError, OSError, ValueError, json.JSONDecodeError):
            receipt_sha = None
    if (
        value["activation_payload_sha256"] != payload_sha
        or value["activation_receipt_sha256"] != receipt_sha
        or value["activation_receipt_commit_sha"] is not None
    ):
        raise AuditError("abort result activation nullability or binding changed")

    default_artifacts: dict[str, str | Path] = {
        "decision_ledger_sha256": DEFAULT_DECISIONS,
        "outcome_ledger_sha256": DEFAULT_OUTCOMES,
        "completed_month_ledger_sha256": DEFAULT_MONTHS,
        "score_output_sha256": DEFAULT_SCORES,
        "picks_output_sha256": DEFAULT_PICKS,
    }
    if artifact_paths is not None:
        if set(artifact_paths) != set(default_artifacts):
            raise AuditError("abort audit artifact path registry changed")
        default_artifacts = dict(artifact_paths)
    expected_artifacts: dict[str, str | None] = {
        field: None for field in contract["required_artifact_hashes"]
    }
    for field, path in default_artifacts.items():
        target = Path(path)
        try:
            expected_artifacts[field] = _stable_plain_file_sha256(
                target, label=f"abort {field}"
            )
        except FileNotFoundError:
            expected_artifacts[field] = None
    expected_artifacts["score_semantic_sha256"] = None
    if value["artifact_sha256"] != expected_artifacts:
        raise AuditError("abort result artifact byte hashes changed")

    runtime_value = (
        dict(runtime_lock)
        if runtime_lock is not None
        else read_json(DEFAULT_RUNTIME_LOCK)
    )
    distributions = {
        item["name"]: item["version"]
        for item in runtime_value["runtime"]["distributions"]
    }
    observed_runtime = value["runtime"]
    if not isinstance(observed_runtime, Mapping) or set(observed_runtime) != set(
        contract["required_runtime_fields"]
    ):
        raise AuditError("abort result runtime fields changed")
    _aware_timestamp(
        observed_runtime["runtime_lock_verified_at"],
        "abort runtime_lock_verified_at",
    )
    expected_runtime = {
        "runtime_lock_sha256": protocol["runtime_lock_contract"]["file_sha256"],
        "runtime_lock_self_sha256": protocol["runtime_lock_contract"]["self_sha256"],
        "runtime_lock_verified_at": observed_runtime["runtime_lock_verified_at"],
        "python_version": runtime_value["runtime"]["python"]["version"],
        "numpy_version": distributions["numpy"],
        "pandas_version": distributions["pandas"],
        "scikit_learn_version": distributions["scikit-learn"],
    }
    if dict(observed_runtime) != expected_runtime:
        raise AuditError("abort result runtime does not match the operational lock")

    if result_path is not None:
        result_file = Path(result_path)
        result_bytes = _stable_plain_file_bytes(
            result_file, label="canonical abort result"
        )
        if result_bytes != canonical_json_file_bytes(value):
            raise AuditError("canonical abort result bytes changed")
    return value


def audit(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    activation_payload_path: Path = DEFAULT_ACTIVATION_PAYLOAD,
    activation_receipt_path: Path = DEFAULT_ACTIVATION_RECEIPT,
    decisions_path: Path = DEFAULT_DECISIONS,
    outcomes_path: Path = DEFAULT_OUTCOMES,
    months_path: Path = DEFAULT_MONTHS,
    state_manifest_directory: Path = DEFAULT_STATE_MANIFESTS,
    fold_manifest_directory: Path = DEFAULT_FOLD_MANIFESTS,
    fold_model_directory: Path = DEFAULT_FOLD_MODELS,
    source_manifest_directory: Path = DEFAULT_SOURCE_MANIFESTS,
    outcome_manifest_directory: Path = DEFAULT_OUTCOME_MANIFESTS,
    checkpoint_proposal_directory: Path = DEFAULT_CHECKPOINT_PROPOSALS,
    predictor_raw_store_root: Path | None = None,
    outcome_raw_store_root: Path | None = None,
    checkpoint_core_store_root: Path | None = None,
    scores_path: Path = DEFAULT_SCORES,
    picks_path: Path = DEFAULT_PICKS,
    result_path: Path = DEFAULT_RESULT,
    runner_path: Path = DEFAULT_RUNNER,
    calendar_path: Path = DEFAULT_CALENDAR,
) -> dict[str, Any]:
    recorder = AuditRecorder()
    # Establish every repository-local authority lexically before reading the
    # result discriminator or choosing the outcome-blind abort branch.
    canonical_repo_paths = {
        "protocol path": (protocol_path, DEFAULT_PROTOCOL),
        "activation payload path": (activation_payload_path, DEFAULT_ACTIVATION_PAYLOAD),
        "activation receipt path": (activation_receipt_path, DEFAULT_ACTIVATION_RECEIPT),
        "decision ledger path": (decisions_path, DEFAULT_DECISIONS),
        "outcome ledger path": (outcomes_path, DEFAULT_OUTCOMES),
        "completed-month ledger path": (months_path, DEFAULT_MONTHS),
        "state manifest directory": (state_manifest_directory, DEFAULT_STATE_MANIFESTS),
        "fold manifest directory": (fold_manifest_directory, DEFAULT_FOLD_MANIFESTS),
        "fold model directory": (fold_model_directory, DEFAULT_FOLD_MODELS),
        "source manifest directory": (source_manifest_directory, DEFAULT_SOURCE_MANIFESTS),
        "outcome manifest directory": (outcome_manifest_directory, DEFAULT_OUTCOME_MANIFESTS),
        "checkpoint proposal directory": (
            checkpoint_proposal_directory,
            DEFAULT_CHECKPOINT_PROPOSALS,
        ),
        "score path": (scores_path, DEFAULT_SCORES),
        "picks path": (picks_path, DEFAULT_PICKS),
        "result path": (result_path, DEFAULT_RESULT),
        "runner path": (runner_path, DEFAULT_RUNNER),
        "calendar path": (calendar_path, DEFAULT_CALENDAR),
    }
    for label, (provided, expected) in canonical_repo_paths.items():
        _require_lexical_canonical_path(provided, expected, label=label)
    protocol, protocol_sha = validate_protocol_contract(protocol_path)
    runtime_lock, runtime_lock_sha = validate_runtime_lock(
        DEFAULT_RUNTIME_LOCK,
        strict_environment=True,
    )
    result_bytes = _stable_plain_file_bytes(result_path, label="canonical result")
    result_status = _result_status_discriminator(result_bytes)
    if result_status == "aborted_integrity_failure":
        if any(
            item is not None
            for item in (
                predictor_raw_store_root,
                outcome_raw_store_root,
                checkpoint_core_store_root,
            )
        ):
            raise AuditError("abort audit does not accept external evidence roots")
        result = _parse_json_object_bytes(result_bytes, label="canonical abort result")
        validated_abort = validate_abort_result(
            result,
            protocol,
            runtime_lock=runtime_lock,
            protocol_path=protocol_path,
            runner_path=runner_path,
            activation_payload_path=activation_payload_path,
            activation_receipt_path=activation_receipt_path,
            artifact_paths={
                "decision_ledger_sha256": decisions_path,
                "outcome_ledger_sha256": outcomes_path,
                "completed_month_ledger_sha256": months_path,
                "score_output_sha256": scores_path,
                "picks_output_sha256": picks_path,
            },
            result_path=result_path,
        )
        return {
            "schema_version": 1,
            "audit_id": "model_v18_shoulder_state_independent_audit",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "independence": {
                "runner_imported": False,
                "project_profit_helpers_imported": False,
                "project_bootstrap_helpers_imported": False,
            },
            "artifact_hashes": {
                "protocol": protocol_sha,
                "runtime_lock": runtime_lock_sha,
                "runner": sha256_file(runner_path),
                "audit_runner": sha256_file(__file__),
                "result": sha256_file(result_path),
                **validated_abort["artifact_sha256"],
            },
            "integrity": {
                "abort_result_outcome_blind": True,
                "failure_reason": validated_abort["failure_reason"],
                "integrity_stage": validated_abort["integrity_stage"],
            },
            "recomputed": {
                "status": "aborted_integrity_failure",
                "failure_reason": validated_abort["failure_reason"],
                "integrity_stage": validated_abort["integrity_stage"],
            },
            "authority": dict(protocol["result_contract"]["authority_values"]),
            "checks": {"outcome_blind_abort_result_exact": True},
            "maximum_absolute_numeric_difference": 0.0,
            "numeric_tolerance": NUMERIC_TOLERANCE,
            "discrepancies": [],
        }
    payload_bytes = _stable_plain_file_bytes(
        activation_payload_path, label="canonical activation payload"
    )
    receipt_bytes = _stable_plain_file_bytes(
        activation_receipt_path, label="canonical activation receipt"
    )
    payload = _parse_json_object_bytes(
        payload_bytes, label="canonical activation payload"
    )
    receipt = _parse_json_object_bytes(
        receipt_bytes, label="canonical activation receipt"
    )
    payload_sha = validate_activation_payload(payload, protocol)
    receipt_sha = validate_activation_receipt(
        receipt,
        payload,
        protocol,
        payload_file_path=activation_payload_path,
    )
    decision_bytes = _stable_plain_file_bytes(
        decisions_path, label="canonical decision ledger"
    )
    decisions = validate_decision_records(
        _parse_jsonl_bytes(decision_bytes, label="canonical decision ledger")
    )
    if not decisions:
        raise AuditError("terminal audit requires a non-empty decision ledger")
    calendar_bytes = _stable_plain_file_bytes(
        calendar_path, label="registered TSE calendar"
    )
    calendar = load_registered_calendar(calendar_path)
    first = first_counted_session(
        workflow_run_updated_at=decisions[0][
            "activation_receipt_workflow_run_updated_at"
        ],
        workflow_run_observed_at=decisions[0][
            "activation_receipt_workflow_run_observed_at"
        ],
        calendar=calendar,
        not_before_session=protocol["periods"]["not_before_session"],
    )
    terminal = deterministic_terminal_session(first, calendar)
    expected_sessions = calendar[(calendar >= first) & (calendar <= terminal)]
    observed_sessions = pd.DatetimeIndex(
        pd.to_datetime([row["session_date"] for row in decisions])
    )
    if not observed_sessions.equals(expected_sessions):
        raise AuditError(
            "decision ledger is not the exact registered terminal denominator"
        )
    month_bytes = _stable_plain_file_bytes(
        months_path, label="canonical completed-month ledger"
    )
    month_records = _parse_jsonl_bytes(
        month_bytes, label="canonical completed-month ledger"
    )
    # The exact terminal decision denominator and the completed-month
    # structural/hash/chronology authority must be ready before the outcome
    # ledger is even opened.  This preflight is intentionally outcome-blind.
    preflight_terminal_completed_month_coverage(
        month_records, decisions, protocol
    )
    outcome_bytes = _stable_plain_file_bytes(
        outcomes_path, label="canonical outcome ledger"
    )
    outcome_records = _parse_jsonl_bytes(
        outcome_bytes, label="canonical outcome ledger"
    )
    outcomes = validate_outcome_records(outcome_records, decisions)
    # This terminal-inclusive seal is the performance unblinding boundary.
    # It must succeed before any external outcome reparse, picks materialization,
    # predictor replay, bootstrap, grouping, or evaluation-frame construction.
    months = validate_terminal_completed_month_coverage(
        month_records, decisions, outcomes, protocol
    )
    # Only the exact terminal-inclusive month seal permits parsing a normal
    # pass/reject result.  Before this point only the outcome-blind status line
    # above has been inspected.
    result = _parse_json_object_bytes(result_bytes, label="canonical terminal result")
    if result.get("status") != result_status:
        raise AuditError("canonical result status discriminator changed on parse")
    activation_git = validate_activation_git_history(
        payload,
        receipt,
        decisions,
        protocol,
        repository_root=ROOT,
        payload_file_path=activation_payload_path,
        receipt_file_path=activation_receipt_path,
    )
    validate_live_module_origin_closure(
        runtime_lock, phase="post-activation-network"
    )
    external_identity_registry: dict[tuple[int, int], str] = {}
    if checkpoint_core_store_root is None:
        raise AuditError("terminal audit requires --checkpoint-core-store-root")
    runner_bytes = _stable_plain_file_bytes(runner_path, label="registered v1.8 runner")
    runner_sha = hashlib.sha256(runner_bytes).hexdigest()
    checkpoint_input = validate_checkpoint_evidence(
        decisions,
        proposal_directory=checkpoint_proposal_directory,
        checkpoint_core_store_root=checkpoint_core_store_root,
        protocol=protocol,
        activation_payload=payload,
        runner_sha256=runner_sha,
        external_identity_registry=external_identity_registry,
        repository_root=ROOT,
    )
    validate_live_module_origin_closure(
        runtime_lock, phase="post-checkpoint-network"
    )
    if outcome_raw_store_root is None:
        raise AuditError("terminal audit requires --outcome-raw-store-root")
    outcome_manifests = _outcome_manifest_map(
        outcome_manifest_directory,
        decisions,
        protocol,
        raw_store_root=outcome_raw_store_root,
        external_identity_registry=external_identity_registry,
    )
    outcomes = validate_outcome_records(
        outcome_records,
        decisions,
        outcome_manifests=outcome_manifests,
    )
    validate_live_module_origin_closure(runtime_lock, phase="post-outcome-parser")
    expected_picks_bytes = recompute_picks_csv_bytes(decisions, outcomes)
    picks_bytes = _stable_plain_file_bytes(
        picks_path, label="canonical picks output"
    )
    recorder.check(
        "picks_output_recomputed_from_decision_and_outcome_ledgers",
        picks_bytes == expected_picks_bytes,
    )
    manifests = _manifest_map(
        state_manifest_directory,
        protocol,
        first_counted_session=first,
    )
    if predictor_raw_store_root is None:
        raise AuditError("terminal audit requires --predictor-raw-store-root")
    source_manifests = _source_manifest_map(
        source_manifest_directory,
        protocol,
        predictor_raw_store_root=predictor_raw_store_root,
        external_identity_registry=external_identity_registry,
    )
    fold_manifests, fold_manifests_by_month = _fold_artifact_maps(
        fold_manifest_directory,
        fold_model_directory,
        protocol,
        runner_sha256=runner_sha,
        first_counted_session=first,
    )

    recorder.check(
        "first_counted_session_recomputed",
        decisions[0]["session_date"] == str(first.date()),
        observed=decisions[0]["session_date"],
        expected=str(first.date()),
    )
    recorder.check("activation_git_history_and_committed_bytes_verified", True)
    recorder.check(
        "registered_calendar_denominator_and_terminal_exact",
        observed_sessions.equals(expected_sessions),
        observed=list(observed_sessions.strftime("%Y-%m-%d")),
        expected=list(expected_sessions.strftime("%Y-%m-%d")),
    )
    recorder.check(
        "activation_hashes_bound_to_every_decision",
        all(
            row["activation_payload_sha256"] == payload_sha
            and row["activation_receipt_sha256"] == receipt_sha
            for row in decisions
        ),
    )
    recorder.check(
        "receipt_commit_binding_month_fixed",
        len(
            {
                (
                    row["activation_receipt_commit_sha"],
                    row["activation_receipt_commit_url"],
                    row["activation_receipt_commit_committed_at"],
                    row["activation_receipt_commit_observed_at"],
                    row["branch_tip_sha_when_receipt_observed"],
                    row["activation_receipt_file_sha256"],
                    row["activation_receipt_workflow_run_id"],
                    row["activation_receipt_workflow_run_updated_at"],
                    row["activation_receipt_workflow_run_observed_at"],
                    canonical_json_sha256(row["receipt_workflow_run_observation"]),
                )
                for row in decisions
            }
        )
        == 1,
    )
    activation_ready = max(
        _aware_timestamp(
            decisions[0]["activation_receipt_workflow_run_updated_at"],
            "activation_receipt_workflow_run_updated_at",
        ),
        _aware_timestamp(
            decisions[0]["activation_receipt_workflow_run_observed_at"],
            "activation_receipt_workflow_run_observed_at",
        ),
    )
    recorder.check(
        "activation_is_actual_and_all_seals_are_after_observation",
        all(
            _aware_timestamp(
                row["decision_materialized_at"], "decision_materialized_at"
            )
            >= activation_ready
            for row in decisions
        )
        and all(
            state["activation_payload_sha256"] == payload_sha
            and state["activation_receipt_sha256"] == receipt_sha
            and _aware_timestamp(state["created_at"], "state created_at")
            >= activation_ready
            for state in manifests.values()
        )
        and all(
            _aware_timestamp(fold["fit_started_at"], "fold fit_started_at")
            >= activation_ready
            and _aware_timestamp(fold["fit_completed_at"], "fold fit_completed_at")
            >= activation_ready
            for fold in fold_manifests_by_month.values()
        ),
    )

    represented_months = sorted({row["session_date"][:7] for row in decisions})
    state_schedule = recompute_state_schedule(
        seed_records=recompute_registered_seed(protocol),
        completed_months=months,
        state_manifests=manifests,
        decisions=decisions,
        outcomes=outcomes,
        first_counted_session_value=first,
        activation_ready_at=activation_ready,
        activation_payload_sha256=payload_sha,
        activation_receipt_sha256=receipt_sha,
    )
    recorder.check(
        "state_schedule_recomputed_from_seed_and_exact_completed_month_ledger",
        len(state_schedule) == len(represented_months),
    )
    recorder.check(
        "state_manifest_exists_and_binds_each_target_month",
        set(represented_months) == set(manifests)
        and all(
            row["state_manifest_sha256"]
            == manifests[row["session_date"][:7]]["state_manifest_sha256"]
            and row["three_prior_calendar_months"]
            == manifests[row["session_date"][:7]]["three_prior_calendar_months"]
            and row["three_complete_pair_day_counts"]
            == manifests[row["session_date"][:7]]["three_complete_pair_day_counts"]
            and row["three_month_medians_pct"]
            == manifests[row["session_date"][:7]]["three_month_medians_pct"]
            for row in decisions
        ),
    )
    recorder.check(
        "fold_bundles_are_numeric_sealed_and_bind_every_available_month",
        all(
            (
                state["c00_fold_manifest_sha256"] is None
                and state["fold_model_bundle_file_sha256"] is None
                and month not in fold_manifests_by_month
            )
            or (
                state["c00_fold_manifest_sha256"] in fold_manifests
                and fold_manifests[state["c00_fold_manifest_sha256"]][
                    "target_month"
                ]
                == month
                and fold_manifests[state["c00_fold_manifest_sha256"]][
                    "fold_model_bundle_file_sha256"
                ]
                == state["fold_model_bundle_file_sha256"]
            )
            for month, state in manifests.items()
        )
        and all(
            row["c00_fold_manifest_sha256"]
            == manifests[row["session_date"][:7]]["c00_fold_manifest_sha256"]
            and row["fold_model_bundle_file_sha256"]
            == manifests[row["session_date"][:7]][
                "fold_model_bundle_file_sha256"
            ]
            for row in decisions
        ),
    )
    recorder.check(
        "source_manifests_bind_decisions_and_are_D_minus_one",
        set(source_manifests)
        == {row["source_manifest_sha256"] for row in decisions}
        and all(
            row["source_manifest_sha256"] in source_manifests
            and source_manifests[row["source_manifest_sha256"]]["target_session"]
            == row["session_date"]
            and source_manifests[row["source_manifest_sha256"]]["source_complete"]
            == row["source_complete"]
            for row in decisions
        ),
    )
    recorder.check(
        "every_outcome_manifest_and_external_raw_object_reparsed_including_terminal",
        len(outcome_manifests) == len(decisions)
        and decisions[-1]["session_date"]
        in {item["target_session"] for item in outcome_manifests.values()}
        and all(
            row["outcome_manifest_sha256"] in outcome_manifests
            for row in outcomes
        ),
    )
    recorder.check(
        "completed_month_ledger_covers_every_represented_month_including_terminal",
        [row["completed_month"] for row in months] == represented_months,
    )

    scores, score_bytes = _read_csv_stable(
        scores_path,
        label="canonical score output",
        dtype={"code": str},
        float_precision="round_trip",
    )
    score_semantic_sha = semantic_score_hash(scores)
    if score_bytes != scores.to_csv(index=False, lineterminator="\n").encode("utf-8"):
        raise AuditError("canonical score output bytes are not canonical CSV")
    score_dates = pd.to_datetime(scores["session_date"], errors="coerce")
    score_sessions = {str(item.date()) for item in score_dates}
    expected_score_sessions = {
        row["session_date"] for row in decisions if row["model_complete"]
    }
    recorder.check(
        "score_output_outcome_free_and_two_rows_per_session",
        score_dates.notna().all()
        and len(scores) == 2 * len(expected_score_sessions)
        and score_sessions == expected_score_sessions
        and not (OUTCOME_FIELDS & set(scores)),
    )
    validate_pit_causality(
        decisions,
        scores,
        source_manifests=source_manifests,
        state_manifests=manifests,
        fold_manifests=fold_manifests,
        activation_ready_at=activation_ready,
    )
    recorder.check(
        "score_and_decision_timestamp_causality_recomputed",
        True,
    )
    predictor_input = validate_predictor_evidence(
        source_manifests,
        decisions,
        protocol,
        predictor_raw_store_root=predictor_raw_store_root,
        fold_manifests_by_month=fold_manifests_by_month,
        fold_model_directory=fold_model_directory,
        scores=scores,
        external_identity_registry=external_identity_registry,
    )
    validate_live_module_origin_closure(runtime_lock, phase="post-predictor-fit-score")
    recomputed_input = {**checkpoint_input, **predictor_input}
    recorder.check(
        "predictor_raw_clean_room_replay_and_result_input_exact",
        set(recomputed_input)
        == set(protocol["result_contract"]["required_input_fields"])
        and result.get("input") == recomputed_input,
        observed=result.get("input"),
        expected=recomputed_input,
    )
    evaluation = evaluate_candidate(
        decisions,
        outcomes,
        calendar=calendar,
        completed_months=months,
    )
    validate_live_module_origin_closure(runtime_lock, phase="pre-result-verification")
    recorder.check(
        "terminal_gate_evaluated_once",
        evaluation.get("gate_evaluated") is True
        and evaluation.get("scheduled_sessions") == len(decisions),
    )

    artifact_hashes = {
        "decision_ledger_sha256": hashlib.sha256(decision_bytes).hexdigest(),
        "outcome_ledger_sha256": hashlib.sha256(outcome_bytes).hexdigest(),
        "completed_month_ledger_sha256": hashlib.sha256(month_bytes).hexdigest(),
        "score_output_sha256": hashlib.sha256(score_bytes).hexdigest(),
        "score_semantic_sha256": score_semantic_sha,
        "picks_output_sha256": hashlib.sha256(picks_bytes).hexdigest(),
    }

    required_result = set(protocol["result_contract"]["required_top_level_fields"])
    recorder.check(
        "result_top_level_schema_complete",
        set(result) == required_result,
        observed=sorted(result),
        expected=sorted(required_result),
    )
    recorder.check(
        "result_identity_hashes_exact",
        result.get("protocol_id") == PROTOCOL_ID
        and result.get("schema_version") == 1
        and result.get("protocol_sha256") == protocol_sha
        and result.get("runner_sha256") == runner_sha
        and result.get("activation_payload_sha256") == payload_sha
        and result.get("activation_receipt_sha256") == receipt_sha
        and result.get("activation_receipt_commit_sha")
        == decisions[0]["activation_receipt_commit_sha"],
    )
    runtime_versions = {
        item["name"]: item["version"]
        for item in runtime_lock["runtime"]["distributions"]
    }
    recorder.check(
        "result_runtime_exactly_matches_operational_lock",
        runtime_lock_sha == RUNTIME_LOCK_SHA256
        and isinstance(result.get("runtime"), Mapping)
        and set(result["runtime"])
        == set(protocol["result_contract"]["required_runtime_fields"])
        and result["runtime"].get("runtime_lock_sha256") == runtime_lock_sha
        and result["runtime"].get("runtime_lock_self_sha256")
        == runtime_lock["runtime_lock_self_sha256"]
        and _aware_timestamp(
            result["runtime"].get("runtime_lock_verified_at"),
            "result runtime_lock_verified_at",
        )
        <= datetime.now(timezone.utc)
        and result["runtime"].get("python_version")
        == runtime_lock["runtime"]["python"]["version"]
        and result["runtime"].get("numpy_version") == runtime_versions["numpy"]
        and result["runtime"].get("pandas_version") == runtime_versions["pandas"]
        and result["runtime"].get("scikit_learn_version")
        == runtime_versions["scikit-learn"],
    )
    authority = result.get("authority", {})
    expected_authority = {
        "analysis_type": protocol["authority"]["analysis_type"],
        **protocol["result_contract"]["authority_values"],
    }
    observed_runtime = result.get("runtime")
    if not isinstance(observed_runtime, Mapping) or set(observed_runtime) != set(
        protocol["result_contract"]["required_runtime_fields"]
    ):
        raise AuditError("normal result runtime schema changed")
    _aware_timestamp(
        observed_runtime["runtime_lock_verified_at"],
        "result runtime_lock_verified_at",
    )
    expected_runtime = {
        "runtime_lock_sha256": runtime_lock_sha,
        "runtime_lock_self_sha256": runtime_lock["runtime_lock_self_sha256"],
        "runtime_lock_verified_at": observed_runtime["runtime_lock_verified_at"],
        "python_version": runtime_lock["runtime"]["python"]["version"],
        "numpy_version": runtime_versions["numpy"],
        "pandas_version": runtime_versions["pandas"],
        "scikit_learn_version": runtime_versions["scikit-learn"],
    }
    candidate_evaluation = {**evaluation, "input_bindings": recomputed_input}
    expected_result = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha,
        "runner_sha256": runner_sha,
        "activation_payload_sha256": payload_sha,
        "activation_receipt_sha256": receipt_sha,
        "activation_receipt_commit_sha": decisions[0][
            "activation_receipt_commit_sha"
        ],
        "status": evaluation["status"],
        "failure_reason": None,
        "integrity_stage": None,
        "authority": expected_authority,
        "input": recomputed_input,
        "forward_period": {
            "first_counted_session": decisions[0]["session_date"],
            "terminal_session": decisions[-1]["session_date"],
            "scheduled_sessions": len(decisions),
            "represented_calendar_months": len(represented_months),
        },
        "state_months": months,
        "models": {
            SH01: candidate_evaluation,
            C00_TOP1: evaluation["control_cost_metrics"][C00_TOP1],
            C02_TOP2: evaluation["control_cost_metrics"][C02_TOP2],
        },
        "candidate_gate": {
            "candidate_id": SH01,
            "checks": evaluation["gate_checks"],
            "passed": evaluation["gate_passed"],
        },
        "decision": evaluation["decision"],
        "artifact_sha256": artifact_hashes,
        "runtime": expected_runtime,
    }
    expected_result_bytes = canonical_json_file_bytes(expected_result)
    if result != expected_result or result_bytes != expected_result_bytes:
        raise AuditError(
            "normal result is not the exact independently reconstructed canonical object"
        )
    recorder.check("full_result_object_and_canonical_bytes_exact", True)
    recorder.check(
        "result_authority_research_only",
        authority == expected_authority
        and evaluation["decision"]["production_model_changed"] is False
        and evaluation["decision"]["production_promotion_allowed"] is False
        and evaluation["decision"]["orders_allowed"] is False,
    )
    recorder.check(
        "result_status_and_winner_recomputed",
        result.get("status") == evaluation["status"]
        and result.get("failure_reason") is None
        and result.get("integrity_stage") is None
        and result.get("decision") == evaluation["decision"],
        observed={"status": result.get("status"), "decision": result.get("decision")},
        expected={"status": evaluation["status"], "decision": evaluation["decision"]},
    )
    recorder.compare_nested(
        "candidate_model_metrics_recomputed",
        result.get("models", {}).get(SH01),
        candidate_evaluation,
    )
    recorder.compare_nested(
        "control_cost_metrics_recomputed",
        {
            C00_TOP1: result.get("models", {}).get(C00_TOP1),
            C02_TOP2: result.get("models", {}).get(C02_TOP2),
        },
        evaluation["control_cost_metrics"],
    )
    recorder.compare_nested(
        "completed_month_result_records_recomputed",
        result.get("state_months"),
        months,
    )
    recorder.compare_nested(
        "forward_period_recomputed",
        result.get("forward_period"),
        {
            "first_counted_session": decisions[0]["session_date"],
            "terminal_session": decisions[-1]["session_date"],
            "scheduled_sessions": len(decisions),
            "represented_calendar_months": len(represented_months),
        },
    )
    candidate_gate = result.get("candidate_gate", {})
    recorder.check(
        "candidate_gate_checks_recomputed",
        candidate_gate.get("candidate_id") == SH01
        and candidate_gate.get("checks") == evaluation["gate_checks"]
        and candidate_gate.get("passed") == evaluation["gate_passed"],
    )

    recorder.check(
        "result_artifact_hashes_recomputed",
        result.get("artifact_sha256") == artifact_hashes,
        observed=result.get("artifact_sha256"),
        expected=artifact_hashes,
    )
    recorder.check(
        "result_file_bytes_are_canonical",
        result_bytes == canonical_json_file_bytes(result),
    )
    passed = not recorder.discrepancies and all(recorder.checks.values())
    return {
        "schema_version": 1,
        "audit_id": "model_v18_shoulder_state_independent_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "independence": {
            "runner_imported": False,
            "project_profit_helpers_imported": False,
            "project_bootstrap_helpers_imported": False,
        },
        "artifact_hashes": {
            "protocol": protocol_sha,
            "runtime_lock": runtime_lock_sha,
            "activation_payload_file": hashlib.sha256(payload_bytes).hexdigest(),
            "activation_receipt_file": hashlib.sha256(receipt_bytes).hexdigest(),
            "runner": runner_sha,
            "audit_runner": sha256_file(__file__),
            "calendar": hashlib.sha256(calendar_bytes).hexdigest(),
            **artifact_hashes,
            "result": hashlib.sha256(result_bytes).hexdigest(),
        },
        "integrity": {
            "first_counted_session": str(first.date()),
            "terminal_session": str(terminal.date()),
            "scheduled_sessions": len(decisions),
            "represented_calendar_months": len(represented_months),
            "decision_head_sha256": decisions[-1]["record_sha256"],
            "outcome_head_sha256": outcomes[-1]["record_sha256"],
            "activation_git": activation_git,
        },
        "recomputed": evaluation,
        "recomputed_input": recomputed_input,
        "recomputed_state_schedule": state_schedule,
        "authority": {
            "production_model_changed": False,
            "production_promotion_allowed": False,
            "orders_allowed": False,
        },
        "checks": recorder.checks,
        "maximum_absolute_numeric_difference": recorder.max_abs_numeric_difference,
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "discrepancies": recorder.discrepancies,
    }


def write_json_exclusive(value: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8", newline="") as stream:
            json.dump(
                _json_safe(value),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
    except FileExistsError as exc:
        raise AuditError(f"refusing to overwrite existing artifact: {target}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--activation-payload", type=Path, default=DEFAULT_ACTIVATION_PAYLOAD
    )
    parser.add_argument(
        "--activation-receipt", type=Path, default=DEFAULT_ACTIVATION_RECEIPT
    )
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--months", type=Path, default=DEFAULT_MONTHS)
    parser.add_argument(
        "--state-manifests", type=Path, default=DEFAULT_STATE_MANIFESTS
    )
    parser.add_argument(
        "--fold-manifests", type=Path, default=DEFAULT_FOLD_MANIFESTS
    )
    parser.add_argument("--fold-models", type=Path, default=DEFAULT_FOLD_MODELS)
    parser.add_argument(
        "--source-manifests", type=Path, default=DEFAULT_SOURCE_MANIFESTS
    )
    parser.add_argument(
        "--outcome-manifests", type=Path, default=DEFAULT_OUTCOME_MANIFESTS
    )
    parser.add_argument(
        "--checkpoint-proposals", type=Path, default=DEFAULT_CHECKPOINT_PROPOSALS
    )
    parser.add_argument(
        "--predictor-raw-store-root",
        type=Path,
        help="external append-only sealed predictor JPX PDF evidence-store root",
    )
    parser.add_argument(
        "--outcome-raw-store-root",
        type=Path,
        help="external append-only sealed JPX PDF evidence-store root",
    )
    parser.add_argument(
        "--checkpoint-core-store-root",
        type=Path,
        help="external append-only sealed checkpoint-core evidence-store root",
    )
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--picks", type=Path, default=DEFAULT_PICKS)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_lexical_canonical_path(
        args.output, DEFAULT_OUTPUT, label="audit output path"
    )
    report = audit(
        protocol_path=args.protocol,
        activation_payload_path=args.activation_payload,
        activation_receipt_path=args.activation_receipt,
        decisions_path=args.decisions,
        outcomes_path=args.outcomes,
        months_path=args.months,
        state_manifest_directory=args.state_manifests,
        fold_manifest_directory=args.fold_manifests,
        fold_model_directory=args.fold_models,
        source_manifest_directory=args.source_manifests,
        outcome_manifest_directory=args.outcome_manifests,
        checkpoint_proposal_directory=args.checkpoint_proposals,
        predictor_raw_store_root=args.predictor_raw_store_root,
        outcome_raw_store_root=args.outcome_raw_store_root,
        checkpoint_core_store_root=args.checkpoint_core_store_root,
        scores_path=args.scores,
        picks_path=args.picks,
        result_path=args.result,
        runner_path=args.runner,
        calendar_path=args.calendar,
    )
    write_json_exclusive(report, args.output)
    if report["status"] != "pass":
        raise SystemExit("v1.8 independent audit failed; inspect discrepancies")
    print(
        "v1.8 independent audit passed: "
        f"{len(report['checks'])} checks, "
        f"winner={report['recomputed']['winner']}"
    )


if __name__ == "__main__":
    main()
