from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import pandas as pd
import pytest

from research import model_v18_shoulder_state_audit as audit
from research import model_v18_shoulder_state_runner as runner


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
EXPECTED_PROTOCOL_ID = "model_v18_shoulder_state_forward_20260804"
EXPECTED_CANDIDATE = "SH01_LAGGED_MONTHLY_SHOULDER_STATE"
EXPECTED_PROTOCOL_SHA256 = (
    "c623fabfa8e94381bce27d359cefc6e51a9a80f1c18f62f6098cfdfc8e9f6112"
)


def _install_virtual_registered_ca(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module: object,
    contract: dict[str, object],
) -> Path:
    """Expose only the preregistered CA path to a foreign-host unit test."""

    registered = Path(str(contract["cafile_path"]))
    backing = tmp_path / f"{getattr(module, '__name__').rsplit('.', 1)[-1]}-ca.pem"
    backing.write_bytes(b"x" * int(contract["cafile_size_bytes"]))
    backing_stat = backing.stat()
    real_stat = Path.stat

    def registered_stat(self: Path, *args: object, **kwargs: object):
        if self == registered:
            return backing_stat
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", registered_stat)
    real_sha256_file = getattr(module, "sha256_file")

    def registered_sha256(path: str | Path) -> str:
        if Path(path) == registered:
            return str(contract["cafile_sha256"])
        return real_sha256_file(path)

    monkeypatch.setattr(module, "sha256_file", registered_sha256)
    return registered


def _assert_foreign_runtime_rejection(stderr: str) -> None:
    assert any(
        fragment in stderr
        for fragment in (
            "registered TLS CA file is unavailable",
            "registered TLS CA trust-store bytes changed",
            "registered ELF object is missing or symlinked",
            "registered ELF object bytes changed",
            "operational platform differs from runtime lock",
            "operational Python differs from runtime lock",
        )
    ), stderr


def _wide_history(
    monthly_differences: dict[str, float] | None = None,
    *,
    rows_per_month: int = 10,
) -> pd.DataFrame:
    differences = monthly_differences or {
        "2026-05": 1.0,
        "2026-06": -3.0,
        "2026-07": 2.0,
    }
    rows: list[dict[str, object]] = []
    for month, difference in differences.items():
        start = pd.Period(month, freq="M").start_time
        for session in pd.bdate_range(start, periods=rows_per_month):
            rows.append(
                {
                    "session_date": session,
                    "rank1_oc_return_pct": difference / 2.0,
                    "rank2_oc_return_pct": -difference / 2.0,
                }
            )
    return pd.DataFrame(rows)


def _chain(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    previous = audit.ZERO_SHA256
    output: list[dict[str, object]] = []
    for sequence, raw in enumerate(rows):
        row = {**raw, "sequence_number": sequence, "previous_record_sha256": previous}
        row["record_sha256"] = audit.canonical_json_sha256(
            row, exclude_fields={"record_sha256"}
        )
        previous = str(row["record_sha256"])
        output.append(row)
    return output


def _rehash_checkpoint_core(row: dict[str, object]) -> dict[str, object]:
    fields = audit.read_json(audit.DEFAULT_PROTOCOL)[
        "daily_preopen_checkpoint_contract"
    ]["sealed_core_store"]["decision_core_required_fields"]
    row["checkpoint_core_sha256"] = audit.canonical_json_sha256(
        {field: row[field] for field in fields}
    )
    return row


def _github_observation(
    *,
    observation_kind: str,
    projected_commit_sha: str,
    committed_at: str,
    retrieved_at: str,
    requested_commit_sha: str | None = None,
    parent_shas: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build a protocol-valid immutable transport receipt for pure tests."""

    repository = "rokuroku-066/TSE-Session-Ranker"
    branch = "agent/v16-real-data-model-eval-20260728"
    if observation_kind == "commit":
        assert requested_commit_sha is not None
        endpoint = f"/repos/{repository}/commits/{requested_commit_sha}"
        observation_branch: str | None = None
    else:
        assert observation_kind == "branch_tip"
        assert requested_commit_sha is None
        endpoint = f"/repos/{repository}/commits/{quote(branch, safe='')}"
        observation_branch = branch
    retrieved = pd.Timestamp(retrieved_at).to_pydatetime()
    http_date = format_datetime(
        (retrieved.astimezone(timezone.utc) - timedelta(seconds=1)).replace(
            microsecond=0
        ),
        usegmt=True,
    )
    content_type = "application/json; charset=utf-8"
    etag = f'"test-{observation_kind}-{projected_commit_sha[:12]}"'
    projection = {
        "commit_sha": projected_commit_sha,
        "html_url": f"https://github.com/{repository}/commit/{projected_commit_sha}",
        "committer_date": committed_at,
        "parent_shas": sorted(set(parent_shas)),
    }
    response_body = audit.canonical_json_bytes(projection)
    value: dict[str, object] = {
        "schema_version": 1,
        "observation_kind": observation_kind,
        "endpoint": endpoint,
        "repository": repository,
        "branch": observation_branch,
        "requested_commit_sha": requested_commit_sha,
        "http_status": 200,
        "content_type": content_type,
        "http_date": http_date,
        "etag": etag,
        "response_body_sha256": hashlib.sha256(response_body).hexdigest(),
        "response_headers_sha256": audit.canonical_json_sha256(
            {
                "status": 200,
                "content_type": content_type,
                "date": http_date,
                "etag": etag,
            }
        ),
        "canonical_projection": projection,
        "retrieved_at": retrieved_at,
        "canonical_json_contract": "project_canonical_json_v1",
    }
    value["observation_sha256"] = audit.canonical_json_sha256(value)
    return value


def _github_workflow_observation(
    *,
    head_sha: str,
    run_id: int,
    updated_at: str,
    retrieved_at: str,
    conclusion: str = "success",
) -> dict[str, object]:
    """Build a protocol-shaped terminal Actions-run receipt for pure tests."""

    repository = "rokuroku-066/TSE-Session-Ranker"
    updated = pd.Timestamp(updated_at)
    started = updated - pd.Timedelta(minutes=2)
    created = started - pd.Timedelta(minutes=1)
    retrieved = pd.Timestamp(retrieved_at).to_pydatetime()
    http_date = format_datetime(
        (retrieved.astimezone(timezone.utc) - timedelta(seconds=1)).replace(
            microsecond=0
        ),
        usegmt=True,
    )
    content_type = "application/json; charset=utf-8"
    etag = f'"test-workflow-{run_id}"'
    projection = {
        "run_id": run_id,
        "workflow_id": 18,
        "workflow_name": "tests",
        "workflow_path": ".github/workflows/tests.yml",
        "event": "pull_request",
        "head_sha": head_sha,
        "run_attempt": 1,
        "status": "completed",
        "conclusion": conclusion,
        "created_at": created.isoformat(),
        "run_started_at": started.isoformat(),
        "updated_at": updated_at,
        "html_url": f"https://github.com/{repository}/actions/runs/{run_id}",
    }
    value: dict[str, object] = {
        "schema_version": 1,
        "endpoint": f"/repos/{repository}/actions/runs/{run_id}",
        "repository": repository,
        "run_id": run_id,
        "expected_head_sha": head_sha,
        "http_status": 200,
        "content_type": content_type,
        "http_date": http_date,
        "etag": etag,
        "response_body_sha256": hashlib.sha256(
            audit.canonical_json_bytes(projection)
        ).hexdigest(),
        "response_headers_sha256": audit.canonical_json_sha256(
            {
                "status": 200,
                "content_type": content_type,
                "date": http_date,
                "etag": etag,
            }
        ),
        "canonical_projection": projection,
        "retrieved_at": retrieved_at,
        "canonical_json_contract": "project_canonical_json_v1",
    }
    value["observation_sha256"] = audit.canonical_json_sha256(value)
    return value


def _numeric_component(values: list[float] | list[int], dtype: str) -> dict[str, object]:
    component: dict[str, object] = {
        "dtype": dtype,
        "shape": [len(values)],
        "values": values,
    }
    component["sha256"] = audit.canonical_json_sha256(component)
    return component


def _synthetic_fold_bundle() -> dict[str, object]:
    width = len(audit.C00_FEATURES) + 2
    indicator = [1, 4]
    value: dict[str, object] = {
        "schema_version": 1,
        "target_month": "2026-08",
        "created_at": "2026-08-04T08:00:00+09:00",
        "runtime_lock_sha256": audit.RUNTIME_LOCK_SHA256,
        "runtime_lock_verified_at": "2026-08-04T07:00:00+09:00",
        "input_feature_order": list(audit.C00_FEATURES),
        "input_feature_dtype": "float64",
        "input_feature_shape": [len(audit.C00_FEATURES)],
        "transformed_feature_order": [
            *audit.C00_FEATURES,
            *(f"missingindicator::{audit.C00_FEATURES[index]}" for index in indicator),
        ],
        "transformed_feature_dtype": "float64",
        "transformed_feature_shape": [width],
        "imputer_strategy": "median",
        "imputer_add_indicator": True,
        "imputer_keep_empty_features": False,
        "imputer_statistics": _numeric_component(
            [float(index) / 10.0 for index in range(len(audit.C00_FEATURES))],
            "float64",
        ),
        "imputer_indicator_features": _numeric_component(indicator, "int64"),
        "scaler_with_mean": True,
        "scaler_with_std": True,
        "scaler_mean": _numeric_component([0.0] * width, "float64"),
        "scaler_scale": _numeric_component([1.0] * width, "float64"),
        "ridge_alpha": 1.0,
        "ridge_fit_intercept": True,
        "ridge_coef": _numeric_component(
            [float(index + 1) / 100.0 for index in range(width)], "float64"
        ),
        "ridge_intercept": 0.25,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": audit.sha256_file(
            RESEARCH / "model_v18_shoulder_state_runner.py"
        ),
        "python_version": "3.12.13",
        "numpy_version": "2.3.5",
        "scikit_learn_version": "1.8.0",
        "canonical_json_contract": "project_canonical_json_v1",
    }
    value["fold_model_bundle_sha256"] = audit.canonical_json_sha256(value)
    return value


def _source_manifest(*, complete: bool = True) -> dict[str, object]:
    protocol = json.loads(
        (RESEARCH / "model_v18_shoulder_state_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    contract = protocol["source_contract"]["forward_daily"]
    registered = audit.expected_predictor_sources("2026-08-04") if complete else []
    source_objects = []
    for index, item in enumerate(registered):
        source_file = str(item["file"])
        source_objects.append(
            {
                "object_key": (
                    "model_v18_shoulder_state/predictor/"
                    f"{item['kind']}/{source_file}"
                ),
                "file": source_file,
                "url": item["url"]
                or f"https://www.jpx.co.jp/example/{source_file}",
                "byte_count": item["byte_count"] or 1000 + index,
                "sha256": item["sha256"] or "7" * 64,
            }
        )
    value: dict[str, object] = {
        "schema_version": 1,
        "target_session": "2026-08-05",
        "latest_required_source_session": "2026-08-04",
        "created_at": "2026-08-05T08:10:00+09:00",
        "sealed_at": "2026-08-05T08:15:00+09:00",
        "runtime_lock_sha256": audit.RUNTIME_LOCK_SHA256,
        "runtime_lock_verified_at": "2026-08-05T07:00:00+09:00",
        "source_files": [item["file"] for item in source_objects],
        "source_urls": [item["url"] for item in source_objects],
        "source_object_keys": [item["object_key"] for item in source_objects],
        "source_byte_counts": [item["byte_count"] for item in source_objects],
        "source_sha256": [item["sha256"] for item in source_objects],
        "source_set_sha256": audit.canonical_json_sha256(source_objects),
        "source_received_at": "2026-08-05T08:00:00+09:00" if complete else None,
        "parser_path": contract["parser_path"],
        "parser_version": contract["parser_version"],
        "parser_sha256": contract["parser_sha256"],
        "parsed_row_count": 100 if complete else 0,
        "rejected_row_count": 0,
        "duplicate_date_code_count": 0,
        "parsed_panel_semantic_sha256": "3" * 64 if complete else None,
        "g0_panel_semantic_sha256": "4" * 64 if complete else None,
        "common_universe_semantic_sha256": "5" * 64 if complete else None,
        "target_date_scoring_input_semantic_sha256": "6" * 64 if complete else None,
        "source_complete": complete,
        "failure_reason": None if complete else "source_missing_before_cutoff",
        "python_version": audit.LOCKED_PYTHON_VERSION,
        "canonical_json_contract": "project_canonical_json_v1",
    }
    value["source_manifest_sha256"] = audit.canonical_json_sha256(value)
    return value


def _outcome_manifest(
    decision: dict[str, object], raw_path: Path
) -> dict[str, object]:
    rank1_open, rank1_close = 100.0, 100.2
    rank2_open, rank2_close = 100.0, 101.0
    value: dict[str, object] = {
        "schema_version": 1,
        "target_session": decision["session_date"],
        "created_at": f"{decision['session_date']}T16:05:00+09:00",
        "sealed_at": f"{decision['session_date']}T16:06:00+09:00",
        "runtime_lock_sha256": audit.RUNTIME_LOCK_SHA256,
        "runtime_lock_verified_at": (
            f"{decision['session_date']}T15:55:00+09:00"
        ),
        "source_file_name": f"stq_{str(decision['session_date']).replace('-', '')}.pdf",
        "source_url": "https://www.jpx.co.jp/example.pdf",
        "raw_source_object_key": (
            f"model_v18_shoulder_state/outcome/{decision['session_date']}.pdf"
        ),
        "source_byte_count": raw_path.stat().st_size,
        "source_sha256": audit.sha256_file(raw_path),
        "source_received_at": f"{decision['session_date']}T16:00:00+09:00",
        "parser_path": "src/tse_session_ranker/data/jpx.py",
        "parser_version": "jpx_daily_text_v6_special_quote_marker",
        "parser_sha256": (
            "1bd2e74acced608eb36c3606b593ea407d2d1e5f54b3283790ef8fd0fb1041f7"
        ),
        "parsed_row_count": 2,
        "rejected_row_count": 0,
        "duplicate_date_code_count": 0,
        "parsed_unique_date_count": 1,
        "target_session_row_count": 2,
        "rank1_code": decision["c00_rank1_code"],
        "rank1_open": rank1_open,
        "rank1_close": rank1_close,
        "rank1_recomputed_oc_return_pct": (rank1_close / rank1_open - 1.0) * 100,
        "rank2_code": decision["c02_rank2_code"],
        "rank2_open": rank2_open,
        "rank2_close": rank2_close,
        "rank2_recomputed_oc_return_pct": (rank2_close / rank2_open - 1.0) * 100,
        "decision_record_sha256": decision["record_sha256"],
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "activation_payload_sha256": decision["activation_payload_sha256"],
        "activation_receipt_sha256": decision["activation_receipt_sha256"],
        "python_version": audit.LOCKED_PYTHON_VERSION,
        "canonical_json_contract": "project_canonical_json_v1",
    }
    value["outcome_manifest_sha256"] = audit.canonical_json_sha256(value)
    return value


def _synthetic_predictor_prices(
    *, target: str = "2026-08-05", sessions: int = 100
) -> pd.DataFrame:
    target_date = pd.Timestamp(target)
    dates = pd.bdate_range(end=target_date - pd.offsets.BDay(1), periods=sessions)
    rows: list[dict[str, object]] = []
    for code_index, code in enumerate(("1001", "1002", "1003")):
        for session_index, session in enumerate(dates):
            open_price = 1_000.0 + 25.0 * code_index + 0.5 * session_index
            return_fraction = (
                0.003 if (session_index + code_index) % 2 else -0.003
            )
            close = open_price * (1.0 + return_fraction)
            high = max(open_price, close) * 1.005
            low = min(open_price, close) * 0.995
            volume = 100_000.0 + 1_000.0 * code_index + 100.0 * session_index
            vwap = (open_price + close) / 2.0
            rows.append(
                {
                    "date": session,
                    "code": code,
                    "name": f"name-{code}",
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "am_open": open_price,
                    "am_high": high,
                    "am_low": low,
                    "am_close": vwap,
                    "pm_open": vwap,
                    "pm_high": high,
                    "pm_low": low,
                    "pm_close": close,
                    "trading_unit": 100.0,
                    "volume": volume,
                    "turnover": volume * vwap,
                    "vwap": vwap,
                    "traded": True,
                    "partial_session": False,
                    "source_format": (
                        "jpx_stock_quotations_auction_regular_way_domestic_ordinary"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _forward_ledgers(
    sessions: pd.DatetimeIndex,
    *,
    selected_rank: int = 2,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    receipt_commit = "d" * 40
    receipt_committed_at = "2026-08-04T00:00:00+00:00"
    receipt_commit_observation = _github_observation(
        observation_kind="commit",
        projected_commit_sha=receipt_commit,
        requested_commit_sha=receipt_commit,
        committed_at=receipt_committed_at,
        retrieved_at="2026-08-04T00:04:00+00:00",
    )
    receipt_branch_observation = _github_observation(
        observation_kind="branch_tip",
        projected_commit_sha=receipt_commit,
        committed_at=receipt_committed_at,
        retrieved_at="2026-08-04T00:05:00+00:00",
    )
    receipt_workflow_observation = _github_workflow_observation(
        head_sha=receipt_commit,
        run_id=103,
        updated_at="2026-08-04T00:06:00+00:00",
        retrieved_at="2026-08-04T00:07:00+00:00",
    )
    decision_rows: list[dict[str, object]] = []
    for index, session in enumerate(sessions):
        rank1_code = f"{1000 + index:04d}"
        rank2_code = f"{5000 + index:04d}"
        month = session.to_period("M")
        state_value = 1.0 if selected_rank == 1 else -1.0
        safety_commit = f"{2 * index + 1:040x}"
        checkpoint_commit = f"{2 * index + 2:040x}"
        checkpoint_committed_at = f"{session.date()}T06:00:00+09:00"
        checkpoint_commit_observation = _github_observation(
            observation_kind="commit",
            projected_commit_sha=checkpoint_commit,
            requested_commit_sha=checkpoint_commit,
            committed_at=checkpoint_committed_at,
            retrieved_at=f"{session.date()}T07:20:00+09:00",
            parent_shas=(safety_commit,),
        )
        checkpoint_branch_observation = _github_observation(
            observation_kind="branch_tip",
            projected_commit_sha=checkpoint_commit,
            committed_at=checkpoint_committed_at,
            retrieved_at=f"{session.date()}T07:21:00+09:00",
            parent_shas=(safety_commit,),
        )
        checkpoint_workflow_observation = _github_workflow_observation(
            head_sha=checkpoint_commit,
            run_id=10_000 + index,
            updated_at=f"{session.date()}T07:30:00+09:00",
            retrieved_at=f"{session.date()}T07:31:00+09:00",
        )
        decision_row: dict[str, object] = {
                "schema_version": 1,
                "session_date": str(session.date()),
                "protocol_id": EXPECTED_PROTOCOL_ID,
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                "runner_sha256": "a" * 64,
                "runtime_lock_sha256": audit.RUNTIME_LOCK_SHA256,
                "runtime_lock_verified_at": (
                    f"{session.date()}T07:26:00+09:00"
                ),
                "candidate_id": EXPECTED_CANDIDATE,
                "activation_payload_sha256": "b" * 64,
                "activation_receipt_sha256": "c" * 64,
                "activation_receipt_commit_sha": receipt_commit,
                "activation_receipt_commit_url": (
                    "https://github.com/rokuroku-066/TSE-Session-Ranker/commit/"
                    + receipt_commit
                ),
                "activation_receipt_commit_committed_at": receipt_committed_at,
                "activation_receipt_commit_observed_at": (
                    "2026-08-04T00:05:00+00:00"
                ),
                "branch_tip_sha_when_receipt_observed": receipt_commit,
                "activation_receipt_file_sha256": "7" * 64,
                "receipt_commit_observation": copy.deepcopy(
                    receipt_commit_observation
                ),
                "receipt_branch_observation": copy.deepcopy(
                    receipt_branch_observation
                ),
                "activation_receipt_workflow_run_id": 103,
                "activation_receipt_workflow_run_updated_at": (
                    "2026-08-04T00:06:00+00:00"
                ),
                "activation_receipt_workflow_run_observed_at": (
                    "2026-08-04T00:07:00+00:00"
                ),
                "receipt_workflow_run_observation": copy.deepcopy(
                    receipt_workflow_observation
                ),
                "decision_materialized_at": f"{session.date()}T08:10:00+09:00",
                "checkpoint_resolution": "primary",
                "checkpoint_resolution_reason": "primary_commitment_timely",
                "checkpoint_core_sha256": None,
                "checkpoint_proposal_path": (
                    "research/model_v18_shoulder_state_checkpoint_proposals/"
                    f"{session.date()}/primary.json"
                ),
                "checkpoint_proposal_file_sha256": "4" * 64,
                "checkpoint_proposal_sha256": "5" * 64,
                "checkpoint_commit_sha": checkpoint_commit,
                "checkpoint_commit_url": (
                    "https://github.com/rokuroku-066/TSE-Session-Ranker/commit/"
                    + checkpoint_commit
                ),
                "checkpoint_commit_committed_at": checkpoint_committed_at,
                "checkpoint_commit_observed_at": f"{session.date()}T07:21:00+09:00",
                "checkpoint_branch_tip_sha_when_observed": checkpoint_commit,
                "checkpoint_commit_observation": checkpoint_commit_observation,
                "checkpoint_branch_observation": checkpoint_branch_observation,
                "checkpoint_workflow_run_id": 10_000 + index,
                "checkpoint_workflow_run_updated_at": (
                    f"{session.date()}T07:30:00+09:00"
                ),
                "checkpoint_workflow_run_observed_at": (
                    f"{session.date()}T07:31:00+09:00"
                ),
                "checkpoint_workflow_run_observation": (
                    checkpoint_workflow_observation
                ),
                "source_manifest_sha256": "e" * 64,
                "c00_fold_manifest_sha256": "f" * 64,
                "fold_model_bundle_file_sha256": "9" * 64,
                "state_manifest_sha256": "1" * 64,
                "decision_cutoff": f"{session.date()}T08:58:59+09:00",
                "computed_at": f"{session.date()}T08:00:00+09:00",
                "source_complete": True,
                "model_complete": True,
                "state_available": True,
                "three_prior_calendar_months": [
                    str(month - 3),
                    str(month - 2),
                    str(month - 1),
                ],
                "three_complete_pair_day_counts": [10, 10, 10],
                "three_month_medians_pct": [
                    state_value,
                    state_value,
                    state_value,
                ],
                "state_value_pct": state_value,
                "selected_source_rank": selected_rank,
                "c00_rank1_code": rank1_code,
                "c00_rank1_score": 1.0,
                "c02_rank2_code": rank2_code,
                "c02_rank2_score": 0.5,
                "candidate_selected_code": (
                    rank1_code if selected_rank == 1 else rank2_code
                ),
                "decision": (
                    "selected_rank1" if selected_rank == 1 else "selected_rank2"
                ),
                "failure_reason": None,
            }
        core_fields = audit.read_json(audit.DEFAULT_PROTOCOL)[
            "daily_preopen_checkpoint_contract"
        ]["sealed_core_store"]["decision_core_required_fields"]
        decision_row["checkpoint_core_sha256"] = audit.canonical_json_sha256(
            {field: decision_row[field] for field in core_fields}
        )
        decision_rows.append(decision_row)
    decisions = _chain(decision_rows)

    outcome_rows: list[dict[str, object]] = []
    for decision in decisions:
        rank1_return = 0.2
        rank2_return = 1.0
        candidate_return = rank1_return if selected_rank == 1 else rank2_return
        outcome_rows.append(
            {
                "schema_version": 1,
                "session_date": decision["session_date"],
                "decision_record_sha256": decision["record_sha256"],
                "outcome_source_sha256": "2" * 64,
                "outcome_manifest_sha256": "8" * 64,
                "outcome_received_at": f"{decision['session_date']}T16:00:00+09:00",
                "computed_at": f"{decision['session_date']}T16:10:00+09:00",
                "rank1_outcome_observed": True,
                "rank1_oc_return_pct": rank1_return,
                "rank2_outcome_observed": True,
                "rank2_oc_return_pct": rank2_return,
                "candidate_outcome_observed": True,
                "candidate_gross_return_pct": candidate_return,
                "candidate_net20_return_pct": candidate_return - 0.2,
                "candidate_net40_return_pct": candidate_return - 0.4,
                "candidate_net60_return_pct": candidate_return - 0.6,
                "c00_top1_gross_return_pct": rank1_return,
                "c00_top1_net20_return_pct": rank1_return - 0.2,
                "c00_top1_net40_return_pct": rank1_return - 0.4,
                "c00_top1_net60_return_pct": rank1_return - 0.6,
                "c02_rank2_gross_return_pct": rank2_return,
                "c02_rank2_net20_return_pct": rank2_return - 0.2,
                "c02_rank2_net40_return_pct": rank2_return - 0.4,
                "c02_rank2_net60_return_pct": rank2_return - 0.6,
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                "activation_payload_sha256": "b" * 64,
                "activation_receipt_sha256": "c" * 64,
            }
        )
    return decisions, _chain(outcome_rows)


def _git_blob_sha(payload: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(payload)).encode("ascii") + b"\x00" + payload
    ).hexdigest()


def _remote_checkpoint_fixture(tmp_path: Path) -> dict[str, object]:
    """Build one complete two-role checkpoint and a fake remote GET surface."""

    protocol, _ = audit.validate_protocol_contract()
    decisions, _ = _forward_ledgers(pd.DatetimeIndex([pd.Timestamp("2026-08-05")]))
    payload = {
        key: copy.deepcopy(value)
        for key, value in decisions[0].items()
        if key not in audit.CHAIN_COLUMNS
    }
    receipt_commit = str(payload["activation_receipt_commit_sha"])
    safety_commit = f"{1:040x}"
    primary_commit = f"{2:040x}"
    safety_committed_at = "2026-08-05T08:10:00+09:00"
    primary_committed_at = "2026-08-05T08:15:00+09:00"
    primary_observation = _github_observation(
        observation_kind="commit",
        projected_commit_sha=primary_commit,
        requested_commit_sha=primary_commit,
        committed_at=primary_committed_at,
        retrieved_at="2026-08-05T08:36:00+09:00",
        parent_shas=(safety_commit,),
    )
    primary_branch_observation = _github_observation(
        observation_kind="branch_tip",
        projected_commit_sha=primary_commit,
        committed_at=primary_committed_at,
        retrieved_at="2026-08-05T08:37:00+09:00",
        parent_shas=(safety_commit,),
    )
    primary_workflow_observation = _github_workflow_observation(
        head_sha=primary_commit,
        run_id=70_002,
        updated_at="2026-08-05T08:35:00+09:00",
        retrieved_at="2026-08-05T08:38:00+09:00",
    )
    payload.update(
        {
            "decision_materialized_at": "2026-08-05T08:40:00+09:00",
            "checkpoint_resolution": "primary",
            "checkpoint_resolution_reason": "primary_commitment_timely",
            "checkpoint_commit_sha": primary_commit,
            "checkpoint_commit_url": (
                "https://github.com/rokuroku-066/TSE-Session-Ranker/commit/"
                + primary_commit
            ),
            "checkpoint_commit_committed_at": primary_committed_at,
            "checkpoint_commit_observed_at": "2026-08-05T08:37:00+09:00",
            "checkpoint_branch_tip_sha_when_observed": primary_commit,
            "checkpoint_commit_observation": primary_observation,
            "checkpoint_branch_observation": primary_branch_observation,
            "checkpoint_workflow_run_id": 70_002,
            "checkpoint_workflow_run_updated_at": "2026-08-05T08:35:00+09:00",
            "checkpoint_workflow_run_observed_at": "2026-08-05T08:38:00+09:00",
            "checkpoint_workflow_run_observation": primary_workflow_observation,
        }
    )
    core_fields = protocol["daily_preopen_checkpoint_contract"][
        "sealed_core_store"
    ]["decision_core_required_fields"]
    primary_core = {field: copy.deepcopy(payload[field]) for field in core_fields}
    primary_core_sha = audit.canonical_json_sha256(primary_core)
    payload["checkpoint_core_sha256"] = primary_core_sha
    # safety_cash is now only the opaque predecessor/evidence role.  Its
    # decision core must be byte-for-byte identical to primary and can never
    # become a decision resolution.
    safety_core = copy.deepcopy(primary_core)
    cores = {"safety_cash": safety_core, "primary": primary_core}
    proposal_root = tmp_path / "checkpoint-proposals"
    proposal_session = proposal_root / "2026-08-05"
    proposal_session.mkdir(parents=True)
    core_root = tmp_path / "checkpoint-core-store"
    core_session = (
        core_root
        / "model_v18_shoulder_state"
        / "checkpoint-core"
        / "2026-08-05"
    )
    core_session.mkdir(parents=True)
    proposals: dict[str, dict[str, object]] = {}
    proposal_bytes: dict[str, bytes] = {}
    for ordinal, role in enumerate(("safety_cash", "primary")):
        core_value = cores[role]
        core_object = {
            "schema_version": 1,
            "target_session": "2026-08-05",
            "checkpoint_role": role,
            "decision_sequence_number": 0,
            "previous_decision_record_sha256": audit.ZERO_SHA256,
            "nonce_hex": str(ordinal + 1) * 64,
            "decision_core": core_value,
            "decision_core_sha256": audit.canonical_json_sha256(core_value),
        }
        envelope = runner.encode_checkpoint_core_envelope(core_object)
        (core_session / f"{role}.bin").write_bytes(envelope)
        proposal: dict[str, object] = {
            "schema_version": 1,
            "checkpoint_id": f"model_v18_shoulder_state_checkpoint_20260805_{role}",
            "checkpoint_batch_id": "model_v18_shoulder_state_checkpoint_batch_20260805",
            "publication_ordinal": ordinal,
            "repository": protocol["repository"],
            "branch": protocol["branch"],
            "protocol_id": audit.PROTOCOL_ID,
            "protocol_sha256": audit.PROTOCOL_SHA256,
            "runner_sha256": payload["runner_sha256"],
            "runtime_lock_sha256": audit.RUNTIME_LOCK_SHA256,
            "activation_payload_sha256": payload["activation_payload_sha256"],
            "activation_receipt_sha256": payload["activation_receipt_sha256"],
            "activation_receipt_commit_sha": receipt_commit,
            "target_session": "2026-08-05",
            "checkpoint_role": role,
            "decision_sequence_number": 0,
            "previous_decision_record_sha256": audit.ZERO_SHA256,
            "sealed_core_object_key": (
                f"model_v18_shoulder_state/checkpoint-core/2026-08-05/{role}.bin"
            ),
            "sealed_core_byte_count": 16_384,
            "sealed_core_sha256": hashlib.sha256(envelope).hexdigest(),
            "created_at": "2026-08-05T08:05:00+09:00",
            "canonical_json_contract": "project_canonical_json_v1",
        }
        proposal["proposal_sha256"] = audit.canonical_json_sha256(proposal)
        raw = audit.canonical_json_file_bytes(proposal)
        (proposal_session / f"{role}.json").write_bytes(raw)
        proposals[role] = proposal
        proposal_bytes[role] = raw
    payload.update(
        {
            "checkpoint_proposal_path": (
                "research/model_v18_shoulder_state_checkpoint_proposals/"
                "2026-08-05/primary.json"
            ),
            "checkpoint_proposal_file_sha256": hashlib.sha256(
                proposal_bytes["primary"]
            ).hexdigest(),
            "checkpoint_proposal_sha256": proposals["primary"]["proposal_sha256"],
        }
    )
    decisions = _chain([payload])

    repository = str(protocol["repository"])
    branch = str(protocol["branch"])
    proposal_relative_root = str(
        protocol["daily_preopen_checkpoint_contract"]["proposal_directory"]
    )
    protected_paths = list(
        protocol["activation"]["preregistration_commit"]["required_paths"]
    )
    workflow_path = protocol["activation"]["payload"]["fixed_values"][
        "workflow_path"
    ]
    base_payloads = {
        path: (
            (ROOT / path).read_bytes()
            if path == workflow_path
            else f"protected fixture {path}\n".encode()
        )
        for path in protected_paths
    }
    activation_payload: dict[str, object] = {}
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
        path = str(
            protocol["activation"]["payload"]["fixed_values"][path_field]
        )
        activation_payload[path_field] = path
        activation_payload[hash_field] = hashlib.sha256(base_payloads[path]).hexdigest()
    base_entries = {
        path: {
            "path": path,
            "mode": "100644",
            "type": "blob",
            "sha": _git_blob_sha(raw),
            "size": len(raw),
        }
        for path, raw in base_payloads.items()
    }
    safety_path = f"{proposal_relative_root}/2026-08-05/safety_cash.json"
    primary_path = f"{proposal_relative_root}/2026-08-05/primary.json"
    safety_blob = _git_blob_sha(proposal_bytes["safety_cash"])
    primary_blob = _git_blob_sha(proposal_bytes["primary"])
    safety_entries = {
        **base_entries,
        safety_path: {
            "path": safety_path,
            "mode": "100644",
            "type": "blob",
            "sha": safety_blob,
            "size": len(proposal_bytes["safety_cash"]),
        },
    }
    primary_entries = {
        **safety_entries,
        primary_path: {
            "path": primary_path,
            "mode": "100644",
            "type": "blob",
            "sha": primary_blob,
            "size": len(proposal_bytes["primary"]),
        },
    }
    base_tree, safety_tree, primary_tree = "3" * 40, "4" * 40, "5" * 40

    def git_commit(
        sha: str, parent: str, tree: str, message: str, committed_at: str
    ) -> dict[str, object]:
        return {
            "sha": sha,
            "message": message,
            "tree": {"sha": tree},
            "parents": [] if not parent else [{"sha": parent}],
            "html_url": f"https://github.com/{repository}/commit/{sha}",
            "committer": {"date": committed_at},
        }

    def rest_commit(
        sha: str, parent: str, tree: str, message: str, committed_at: str
    ) -> dict[str, object]:
        return {
            "sha": sha,
            "html_url": f"https://github.com/{repository}/commit/{sha}",
            "commit": {
                "committer": {"date": committed_at},
                "tree": {"sha": tree},
                "message": message,
            },
            "parents": [] if not parent else [{"sha": parent}],
        }

    def tree_response(
        tree_sha: str, entries: dict[str, dict[str, object]]
    ) -> dict[str, object]:
        return {
            "sha": tree_sha,
            "truncated": False,
            "tree": [copy.deepcopy(entries[path]) for path in sorted(entries)],
        }

    def blob_response(blob_sha: str, raw: bytes) -> dict[str, object]:
        import base64

        return {
            "sha": blob_sha,
            "encoding": "base64",
            "size": len(raw),
            "content": base64.b64encode(raw).decode("ascii"),
        }

    safety_workflow = _github_workflow_observation(
        head_sha=safety_commit,
        run_id=70_001,
        updated_at="2026-08-05T08:25:00+09:00",
        retrieved_at="2026-08-05T08:26:00+09:00",
    )["canonical_projection"]
    primary_workflow = primary_workflow_observation["canonical_projection"]
    responses: dict[str, object] = {
        f"repos/{repository}/git/ref/heads/{branch}": {
            "ref": f"refs/heads/{branch}",
            "object": {
                "type": "commit",
                "sha": primary_commit,
                "url": f"https://api.github.com/repos/{repository}/git/commits/{primary_commit}",
            },
        },
        f"repos/{repository}/git/commits/{receipt_commit}": git_commit(
            receipt_commit,
            "",
            base_tree,
            "activation receipt",
            "2026-08-04T12:00:00+09:00",
        ),
        f"repos/{repository}/git/commits/{safety_commit}": git_commit(
            safety_commit,
            receipt_commit,
            safety_tree,
            "model-v18 checkpoint 2026-08-05 safety_cash",
            safety_committed_at,
        ),
        f"repos/{repository}/git/commits/{primary_commit}": git_commit(
            primary_commit,
            safety_commit,
            primary_tree,
            "model-v18 checkpoint 2026-08-05 primary",
            primary_committed_at,
        ),
        f"repos/{repository}/commits/{safety_commit}": rest_commit(
            safety_commit,
            receipt_commit,
            safety_tree,
            "model-v18 checkpoint 2026-08-05 safety_cash",
            safety_committed_at,
        ),
        f"repos/{repository}/commits/{primary_commit}": rest_commit(
            primary_commit,
            safety_commit,
            primary_tree,
            "model-v18 checkpoint 2026-08-05 primary",
            primary_committed_at,
        ),
        f"repos/{repository}/git/trees/{base_tree}?recursive=1": tree_response(
            base_tree, base_entries
        ),
        f"repos/{repository}/git/trees/{safety_tree}?recursive=1": tree_response(
            safety_tree, safety_entries
        ),
        f"repos/{repository}/git/trees/{primary_tree}?recursive=1": tree_response(
            primary_tree, primary_entries
        ),
        f"repos/{repository}/git/blobs/{base_entries[workflow_path]['sha']}": blob_response(
            str(base_entries[workflow_path]["sha"]), base_payloads[workflow_path]
        ),
        f"repos/{repository}/git/blobs/{safety_blob}": blob_response(
            safety_blob, proposal_bytes["safety_cash"]
        ),
        f"repos/{repository}/git/blobs/{primary_blob}": blob_response(
            primary_blob, proposal_bytes["primary"]
        ),
        f"repos/{repository}/compare/{receipt_commit}...{safety_commit}": {
            "status": "ahead",
            "ahead_by": 1,
            "behind_by": 0,
            "total_commits": 1,
            "base_commit": {"sha": receipt_commit},
            "merge_base_commit": {"sha": receipt_commit},
            "commits": [{"sha": safety_commit}],
            "files": [
                {"filename": safety_path, "status": "added", "sha": safety_blob}
            ],
        },
        f"repos/{repository}/compare/{safety_commit}...{primary_commit}": {
            "status": "ahead",
            "ahead_by": 1,
            "behind_by": 0,
            "total_commits": 1,
            "base_commit": {"sha": safety_commit},
            "merge_base_commit": {"sha": safety_commit},
            "commits": [{"sha": primary_commit}],
            "files": [
                {"filename": primary_path, "status": "added", "sha": primary_blob}
            ],
        },
    }
    for path, entry in base_entries.items():
        responses[f"repos/{repository}/git/blobs/{entry['sha']}"] = blob_response(
            str(entry["sha"]), base_payloads[path]
        )
    proposal_query = (
        f"repos/{repository}/commits?sha={quote(branch, safe='')}"
        f"&path={quote(proposal_relative_root, safe='')}&per_page=100"
    )
    responses[f"{proposal_query}&page=1"] = [
        {"sha": primary_commit},
        {"sha": safety_commit},
    ]
    responses[f"{proposal_query}&page=2"] = []
    historical_sha = "e" * 40
    for path in protected_paths:
        query = (
            f"repos/{repository}/commits?sha={quote(branch, safe='')}"
            f"&path={quote(path, safe='')}&per_page=100"
        )
        responses[f"{query}&page=1"] = [{"sha": historical_sha}]
        responses[f"{query}&page=2"] = []
    for commit_sha, workflow in (
        (safety_commit, safety_workflow),
        (primary_commit, primary_workflow),
    ):
        query = (
            f"repos/{repository}/actions/runs?event=pull_request"
            f"&head_sha={commit_sha}&per_page=100"
        )
        raw_workflow = {
            "id": workflow["run_id"],
            "workflow_id": workflow["workflow_id"],
            "name": workflow["workflow_name"],
            "path": workflow["workflow_path"],
            "event": workflow["event"],
            "head_sha": workflow["head_sha"],
            "run_attempt": workflow["run_attempt"],
            "status": workflow["status"],
            "conclusion": workflow["conclusion"],
            "created_at": workflow["created_at"],
            "run_started_at": workflow["run_started_at"],
            "updated_at": workflow["updated_at"],
            "html_url": workflow["html_url"],
        }
        responses[query] = {"total_count": 1, "workflow_runs": [raw_workflow]}
        responses[f"repos/{repository}/actions/runs/{workflow['run_id']}"] = raw_workflow

    def fetch(endpoint: str) -> object:
        if endpoint not in responses:
            raise AssertionError(f"unexpected fake GitHub endpoint: {endpoint}")
        return copy.deepcopy(responses[endpoint])

    return {
        "protocol": protocol,
        "activation_payload": activation_payload,
        "decisions": decisions,
        "proposal_root": proposal_root,
        "core_root": core_root,
        "responses": responses,
        "fetch": fetch,
        "repository": repository,
        "safety_commit": safety_commit,
        "primary_commit": primary_commit,
        "safety_blob": safety_blob,
        "primary_blob": primary_blob,
        "safety_tree": safety_tree,
        "primary_tree": primary_tree,
        "safety_path": safety_path,
        "primary_path": primary_path,
        "proposals": proposals,
    }


def _completed_month_ledger(
    decisions: list[dict[str, object]],
    outcomes: list[dict[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    months = sorted(
        {pd.Timestamp(row["session_date"]).to_period("M") for row in decisions}
    )
    for month in months:
        created = pd.Timestamp((month + 1).start_time).tz_localize(
            "Asia/Tokyo"
        ) + pd.Timedelta(seconds=1)
        records = runner.build_completed_month_record(
            decisions,
            outcomes,
            completed_month=str(month),
            created_at=created.isoformat(),
            existing_records=records,
        )
    return records


def test_independent_audit_imports_no_runner_or_project_metric_helpers() -> None:
    source = (
        RESEARCH / "model_v18_shoulder_state_audit.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any(name.endswith("_runner") for name in imported)
    assert "tse_session_ranker.profit" not in imported
    assert "tse_session_ranker.validation" not in imported


def test_three_consecutive_month_median_of_medians_is_independent() -> None:
    history = _wide_history()
    expected = runner.derive_month_state(history, "2026-08")
    observed = audit.derive_month_state(history, "2026-08")

    assert observed == expected
    assert observed["required_source_months"] == [
        "2026-05",
        "2026-06",
        "2026-07",
    ]
    assert [item["complete_pairs"] for item in observed["source_months"]] == [
        10,
        10,
        10,
    ]
    assert [
        item["median_rank1_minus_rank2_pct"]
        for item in observed["source_months"]
    ] == [1.0, -3.0, 2.0]
    assert observed["state_value_pct"] == 1.0
    assert observed["decision"] == "rank1"
    assert observed["selected_source_rank"] == 1


def test_month_state_is_fixed_and_same_month_or_future_outcomes_are_ignored() -> None:
    history = _wide_history()
    baseline = runner.derive_month_state(history, "2026-08")
    mutated = pd.concat(
        [
            history,
            _wide_history(
                {"2026-08": -1000.0, "2026-09": 1000.0},
                rows_per_month=15,
            ),
        ],
        ignore_index=True,
    )
    assert runner.derive_month_state(mutated, "2026-08") == baseline
    assert audit.derive_month_state(mutated, "2026-08") == baseline
    assert baseline["history_cutoff_date"] == "2026-07-31"


def test_each_prior_month_needs_ten_complete_pairs_without_older_substitution() -> None:
    history = _wide_history(
        {
            "2026-04": 99.0,
            "2026-05": 1.0,
            "2026-06": 2.0,
            "2026-07": 3.0,
        }
    )
    missing_rank2_date = history.loc[
        history["session_date"].dt.to_period("M").eq(pd.Period("2026-06")),
        "session_date",
    ].iloc[0]
    history.loc[
        history["session_date"].eq(missing_rank2_date),
        "rank2_oc_return_pct",
    ] = np.nan

    state = runner.derive_month_state(history, "2026-08")
    independent = audit.derive_month_state(history, "2026-08")
    assert state == independent
    assert state["decision"] == "cash"
    assert state["selected_source_rank"] is None
    assert state["state_value_pct"] is None
    assert state["reason"] == "insufficient_consecutive_month_history"
    assert state["source_months"][1] == {
        "month": "2026-06",
        "complete_pairs": 9,
        "qualified": False,
        "median_rank1_minus_rank2_pct": None,
    }


def test_exact_zero_state_is_cash() -> None:
    history = _wide_history(
        {"2026-05": -1.0, "2026-06": 0.0, "2026-07": 1.0}
    )
    for derive in (runner.derive_month_state, audit.derive_month_state):
        state = derive(history, "2026-08")
        assert state["state_value_pct"] == 0.0
        assert state["decision"] == "cash"
        assert state["selected_source_rank"] is None
        assert state["reason"] == "exact_zero_three_month_median"


def test_rank_swap_symmetry() -> None:
    history = _wide_history()
    swapped = history.rename(
        columns={
            "rank1_oc_return_pct": "rank2_oc_return_pct",
            "rank2_oc_return_pct": "rank1_oc_return_pct",
        }
    )
    original = runner.derive_month_state(history, "2026-08")
    reverse = runner.derive_month_state(swapped, "2026-08")
    assert original["decision"] == "rank1"
    assert reverse["decision"] == "rank2"
    assert reverse["state_value_pct"] == -original["state_value_pct"]
    np.testing.assert_allclose(
        [item["median_rank1_minus_rank2_pct"] for item in reverse["source_months"]],
        [
            -item["median_rank1_minus_rank2_pct"]
            for item in original["source_months"]
        ],
    )
    assert audit.derive_month_state(swapped, "2026-08") == reverse


def test_pair_history_hash_is_row_order_invariant_and_outcome_sensitive() -> None:
    history = _wide_history()
    original = runner.pair_history_semantic_hash(history)
    shuffled = runner.pair_history_semantic_hash(
        history.sample(frac=1.0, random_state=18)
    )
    assert original == shuffled == audit.pair_history_semantic_hash(history)

    changed = history.copy()
    changed.loc[0, "rank1_oc_return_pct"] += 0.01
    assert runner.pair_history_semantic_hash(changed) != original


def test_semantic_decision_hash_rejects_any_outcome_field() -> None:
    record = {
        "sequence_number": 1,
        "session_date": "2026-08-05",
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "candidate_id": EXPECTED_CANDIDATE,
        "decision_state": "cash",
    }
    digest = audit.semantic_decision_hash([record])
    assert digest == audit.semantic_decision_hash([copy.deepcopy(record)])
    with pytest.raises(audit.AuditError, match="outcome"):
        audit.semantic_decision_hash([{**record, "oc_return_pct": 1.0}])


def test_protocol_calendar_and_registered_seed_are_exact() -> None:
    protocol, digest = audit.validate_protocol_contract()
    assert audit.validate_required_array_integrity(protocol) >= 10
    with pytest.raises(audit.AuditError, match="duplicates"):
        audit.validate_required_array_integrity(
            {"nested": {"required_fields": ["same", "same"]}}
        )
    assert digest == EXPECTED_PROTOCOL_SHA256
    assert runner.PROTOCOL_SHA256 == audit.PROTOCOL_SHA256 == digest
    assert runner.BOOTSTRAP_RANDOM_STATE == audit.BOOTSTRAP_RANDOM_STATE == 20260805
    assert runner.BOOTSTRAP_BLOCK_LENGTH == audit.BOOTSTRAP_BLOCK_LENGTH == 20
    assert protocol["candidate"]["id"] == EXPECTED_CANDIDATE
    assert protocol["evaluation"]["candidate_variants"] == 1
    assert tuple(protocol["append_only_artifacts"]["decision_record_required_fields"]) == (
        audit.DECISION_REQUIRED_FIELDS
    )
    assert tuple(protocol["append_only_artifacts"]["outcome_record_required_fields"]) == (
        audit.OUTCOME_REQUIRED_FIELDS
    )
    seed = audit.recompute_registered_seed(protocol)
    assert [item["complete_pair_days"] for item in seed] == [17, 19, 18]
    np.testing.assert_allclose(
        [item["monthly_median_rank1_minus_rank2_pct"] for item in seed],
        [0.4532617412224217, 0.5353494177210093, 0.09214571919513584],
        rtol=0.0,
        atol=1e-12,
    )

    calendar = audit.load_registered_calendar()
    assert len(calendar) == 343
    assert calendar[0] == pd.Timestamp("2026-08-05")
    assert calendar[-1] == pd.Timestamp("2027-12-30")
    terminal = audit.deterministic_terminal_session(calendar[0], calendar)
    assert terminal == pd.Timestamp("2027-02-26")
    assert len(calendar[(calendar >= calendar[0]) & (calendar <= terminal)]) == 136


def test_repo_artifact_paths_are_lexical_and_pinned_reads_reject_hardlinks(
    tmp_path: Path,
) -> None:
    canonical = RESEARCH / "model_v18_shoulder_state_result.json"
    assert audit._require_lexical_canonical_path(
        canonical, canonical, label="result"
    ) == canonical
    with pytest.raises(audit.AuditError, match="lexical canonical"):
        audit._require_lexical_canonical_path(
            Path("research/model_v18_shoulder_state_result.json"),
            canonical,
            label="result",
        )
    with pytest.raises(audit.AuditError, match="lexical canonical"):
        audit._require_lexical_canonical_path(
            RESEARCH / "nested" / ".." / canonical.name,
            canonical,
            label="result",
        )

    fixtures = {
        "result.json": b'{"status":"fixture"}\n',
        "decisions.jsonl": b'{"sequence_number":0}\n',
        "scores.csv": b"session_date,code\n",
    }
    for name, raw in fixtures.items():
        target = tmp_path / name
        alias = tmp_path / f"{name}.hardlink"
        target.write_bytes(raw)
        os.link(target, alias)
        with pytest.raises(audit.AuditError, match="single-link regular file"):
            audit._stable_plain_file_bytes(target, label=name)
        alias.unlink()
        assert audit._stable_plain_file_bytes(target, label=name) == raw


def test_result_status_discriminator_is_outcome_blind_and_unique() -> None:
    passing = b'{\n  "models": {"sealed": true},\n  "status": "forward_rejected_candidate"\n}\n'
    aborted = b'{\n  "failure_reason": "source_integrity_failure",\n  "status": "aborted_integrity_failure"\n}\n'
    assert audit._result_status_discriminator(passing) == (
        "forward_rejected_candidate"
    )
    assert audit._result_status_discriminator(aborted) == (
        "aborted_integrity_failure"
    )
    with pytest.raises(audit.AuditError, match="unique status"):
        audit._result_status_discriminator(
            b'{\n  "status": "forward_rejected_candidate",\n'
            b'  "status": "aborted_integrity_failure"\n}\n'
        )
    with pytest.raises(audit.AuditError, match="unique status"):
        audit._result_status_discriminator(
            b'{"status":"aborted_integrity_failure"}\n'
        )


def test_runtime_lock_closure_strict_host_and_one_byte_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, _ = audit.validate_protocol_contract()
    lock, file_sha = audit.validate_runtime_lock(strict_environment=False)
    assert file_sha == audit.RUNTIME_LOCK_SHA256
    assert lock["runtime_lock_self_sha256"] == audit.RUNTIME_LOCK_SELF_SHA256
    assert protocol["runtime_lock_contract"]["file_sha256"] == file_sha
    assert (
        protocol["runtime_lock_contract"]["self_sha256"]
        == audit.RUNTIME_LOCK_SELF_SHA256
    )

    system_python = shutil.which("python")
    assert system_python is not None
    environment = os.environ.copy()
    for name in lock["elf_closure"]["loader_environment"]:
        environment.pop(name, None)
    strict = subprocess.run(
        [
            system_python,
            "-c",
            (
                "import sys; from research import model_v18_shoulder_state_audit as a; "
                "sys.modules['__main__'].__file__=a.__file__; "
                "a.validate_runtime_lock(strict_environment=True)"
            ),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    strict_host_matches = strict.returncode == 0
    if not strict_host_matches:
        # A generic CI runner is not the preregistered operational host.  It
        # must fail closed, while pure lock validation above remains portable.
        _assert_foreign_runtime_rejection(strict.stderr)

    copied_root = tmp_path / "runtime-project"
    for record in lock["project_files"]:
        relative = Path(record["path"])
        destination = copied_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    audit.validate_runtime_lock(project_root=copied_root, strict_environment=False)
    dependency = copied_root / "src/tse_session_ranker/config.py"
    dependency.write_bytes(dependency.read_bytes() + b"\n# mutation\n")
    with pytest.raises(audit.AuditError, match="dependency bytes changed"):
        audit.validate_runtime_lock(
            project_root=copied_root,
            strict_environment=False,
        )

    if strict_host_matches:
        monkeypatch.setattr(audit.platform, "python_version", lambda: "0.0.0")
        for name in lock["elf_closure"]["loader_environment"]:
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(audit.AuditError, match="Python differs"):
            audit.validate_runtime_lock(strict_environment=True)


def test_runtime_lock_rejects_stdlib_and_git_byte_mutation(tmp_path: Path) -> None:
    lock = json.loads(
        (RESEARCH / "model_v18_runtime_lock.json").read_text(encoding="utf-8")
    )
    system_python = shutil.which("python")
    assert system_python is not None
    environment = os.environ.copy()
    for name in lock["elf_closure"]["loader_environment"]:
        environment.pop(name, None)

    baseline_script = (
        "import sys; from research import model_v18_shoulder_state_audit as a; "
        "sys.modules['__main__'].__file__=a.__file__; "
        "a.validate_runtime_lock(strict_environment=True)"
    )
    baseline = subprocess.run(
        [system_python, "-c", baseline_script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    if baseline.returncode != 0:
        _assert_foreign_runtime_rejection(baseline.stderr)
        return

    mutated_stdlib = tmp_path / "mutated-stdlib"
    mutated_stdlib.mkdir()
    (mutated_stdlib / "one.py").write_text("# changed stdlib\n", encoding="utf-8")
    stdlib_script = (
        "import sys,sysconfig; "
        "from research import model_v18_shoulder_state_audit as a; "
        "sys.modules['__main__'].__file__=a.__file__; "
        "paths=dict(sysconfig.get_paths()); paths['stdlib']=sys.argv[1]; "
        "a.sysconfig.get_paths=lambda: paths; "
        "a.validate_runtime_lock(strict_environment=True)"
    )
    stdlib_result = subprocess.run(
        [system_python, "-c", stdlib_script, str(mutated_stdlib)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert stdlib_result.returncode != 0
    assert "stdlib tree differs" in stdlib_result.stderr

    git_script = (
        "import sys; from pathlib import Path; "
        "from research import model_v18_shoulder_state_audit as a; "
        "sys.modules['__main__'].__file__=a.__file__; "
        "lock=a.read_json(a.DEFAULT_RUNTIME_LOCK); "
        "target=Path(next(x['path'] for x in lock['elf_closure']['root_objects'] "
        "if 'git_executable' in x['roles'])); real=a.sha256_file; "
        "a.sha256_file=lambda p: ('0'*64 if Path(p).resolve()==target else real(p)); "
        "a.validate_runtime_lock(strict_environment=True)"
    )
    git_result = subprocess.run(
        [system_python, "-c", git_script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert git_result.returncode != 0
    assert "registered ELF object bytes changed" in git_result.stderr


def test_live_module_origin_closure_rechecks_lazy_import_phases() -> None:
    lock = json.loads(
        (RESEARCH / "model_v18_runtime_lock.json").read_text(encoding="utf-8")
    )
    system_python = shutil.which("python")
    assert system_python is not None
    environment = os.environ.copy()
    for name in lock["elf_closure"]["loader_environment"]:
        environment.pop(name, None)
    script = (
        "import sys,types; "
        "from research import model_v18_shoulder_state_audit as a; "
        "sys.modules['__main__'].__file__=a.__file__; "
        "lock,_=a.validate_runtime_lock(strict_environment=True); "
        "rogue=types.ModuleType('v18_unregistered_lazy_module'); "
        "sys.modules['v18_unregistered_lazy_module']=rogue; "
        "caught=False; "
        "\ntry: a.validate_live_module_origin_closure(lock,phase='lazy-test')\n"
        "except a.AuditError: caught=True\n"
        "assert caught; del sys.modules['v18_unregistered_lazy_module']; "
        "import six.moves; six.moves.__spec__.loader=None; caught=False; "
        "\ntry: a.validate_live_module_origin_closure(lock,phase='alias-test')\n"
        "except a.AuditError: caught=True\n"
        "assert caught"
    )
    completed = subprocess.run(
        [system_python, "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        _assert_foreign_runtime_rejection(completed.stderr)

    source = (RESEARCH / "model_v18_shoulder_state_audit.py").read_text(
        encoding="utf-8"
    )
    for phase in (
        "initial strict runtime",
        "post-activation-network",
        "post-checkpoint-network",
        "post-outcome-parser",
        "post-predictor-fit-score",
        "pre-result-verification",
    ):
        assert phase in source


def test_elf_closure_and_loader_environment_mutations_fail_without_linkage_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = json.loads(
        (RESEARCH / "model_v18_runtime_lock.json").read_text(encoding="utf-8")
    )
    closure = lock["elf_closure"]
    assert audit.validate_elf_closure(closure)["root_object_count"] >= 1
    assert closure["dynamic_loader"] in closure["shared_objects"]
    assert closure["loader_cache"] not in closure["shared_objects"]

    changed = copy.deepcopy(closure)
    changed["shared_objects"][0]["sha256"] = "1" * 64
    with pytest.raises(audit.AuditError, match="count or set hash"):
        audit.validate_elf_closure(changed)

    # ld.so.cache is an independently registered authority file, not a shared
    # object.  Its exact bytes are nevertheless verified in strict operation.
    for name in closure["loader_environment"]:
        monkeypatch.delenv(name, raising=False)
    changed_cache = copy.deepcopy(closure)
    changed_cache["loader_cache"]["sha256"] = "1" * 64
    registered_rows = [
        *closure["root_objects"],
        *closure["shared_objects"],
        closure["dynamic_loader"],
        closure["loader_cache"],
    ]
    registered_hashes = {
        Path(str(item["path"])): str(item["sha256"]) for item in registered_rows
    }
    real_is_file = Path.is_file
    real_is_symlink = Path.is_symlink
    real_sha256_file = audit.sha256_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: True if path in registered_hashes else real_is_file(path),
    )
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: False if path in registered_hashes else real_is_symlink(path),
    )

    def registered_sha256(path: str | Path) -> str:
        registered_hash = registered_hashes.get(Path(path))
        if registered_hash is not None:
            return registered_hash
        return real_sha256_file(path)

    monkeypatch.setattr(audit, "sha256_file", registered_sha256)
    with pytest.raises(audit.AuditError, match="registered ELF object bytes changed"):
        audit.validate_elf_closure(changed_cache, strict_environment=True)

    monkeypatch.setenv("LD_PRELOAD", "/tmp/forbidden-v18-preload.so")
    with pytest.raises(audit.AuditError, match="loader environment is contaminated"):
        audit.validate_elf_closure(closure, strict_environment=True)

    audit_source = (
        RESEARCH / "model_v18_shoulder_state_audit.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("ldd", "readelf", "objdump"):
        assert f'subprocess.run(["{forbidden}"' not in audit_source
    assert "shutil.which" not in audit_source


def test_tls_ca_registry_rejects_schema_path_size_and_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock = json.loads(
        (RESEARCH / "model_v18_runtime_lock.json").read_text(encoding="utf-8")
    )
    contract = lock["runtime"]["python"]["ssl"]["ca_trust"]
    for name in contract["environment"]:
        monkeypatch.delenv(name, raising=False)
    registered = Path(contract["cafile_path"])
    # Pure validation binds the declaration through the runtime-lock file/self
    # hashes and therefore must not require the preregistered host filesystem.
    assert audit.validate_tls_ca_trust(contract) == registered

    ca_bytes = b"portable strict CA fixture\n"
    portable = copy.deepcopy(contract)
    portable_path = tmp_path / "registered-ca.pem"
    portable_path.write_bytes(ca_bytes)
    portable.update(
        {
            "cafile_path": str(portable_path),
            "cafile_basename": portable_path.name,
            "cafile_size_bytes": len(ca_bytes),
            "cafile_sha256": hashlib.sha256(ca_bytes).hexdigest(),
        }
    )
    assert audit.validate_tls_ca_trust(
        portable, strict_environment=True
    ) == portable_path

    missing = copy.deepcopy(portable)
    missing_path = tmp_path / "missing-ca.pem"
    missing["cafile_path"] = str(missing_path)
    missing["cafile_basename"] = missing_path.name
    assert audit.validate_tls_ca_trust(missing) == missing_path
    with pytest.raises(audit.AuditError, match="CA file is unavailable"):
        audit.validate_tls_ca_trust(missing, strict_environment=True)

    schema_drift = copy.deepcopy(contract)
    schema_drift["unregistered_field"] = None
    with pytest.raises(audit.AuditError, match="schema changed"):
        audit.validate_tls_ca_trust(schema_drift)

    invalid_pins = {
        "cafile_path": "relative/ca.pem",
        "cafile_basename": "different-ca.pem",
        "cafile_size_bytes": 0,
        "cafile_sha256": "not-a-sha256",
    }
    for field, replacement in invalid_pins.items():
        changed = copy.deepcopy(contract)
        changed[field] = replacement
        with pytest.raises(audit.AuditError, match="CA trust-store pin changed"):
            audit.validate_tls_ca_trust(changed)

    byte_mutations = {
        "cafile_size_bytes": len(ca_bytes) + 1,
        "cafile_sha256": "0" * 64,
    }
    for field, replacement in byte_mutations.items():
        changed = copy.deepcopy(portable)
        changed[field] = replacement
        with pytest.raises(audit.AuditError, match="CA trust-store bytes changed"):
            audit.validate_tls_ca_trust(changed, strict_environment=True)

    for variable in portable["environment"]:
        monkeypatch.setenv(variable, str(portable_path))
        with pytest.raises(audit.AuditError, match="environment overrides"):
            audit.validate_tls_ca_trust(portable, strict_environment=True)
        monkeypatch.delenv(variable)


def test_protocol_result_input_registry_and_status_aware_terminal_raw_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, _ = audit.validate_protocol_contract()
    assert protocol["result_contract"]["required_input_fields"] == [
        "runtime_lock_sha256",
        "checkpoint_proposal_set_sha256",
        "checkpoint_core_object_set_sha256",
        "checkpoint_evidence_set_sha256",
        "predictor_source_manifest_set_sha256",
        "predictor_raw_source_set_sha256",
        "predictor_unique_raw_object_count",
        "predictor_parser_sha256",
        "predictor_parsed_panel_semantic_set_sha256",
        "predictor_g0_panel_semantic_set_sha256",
        "predictor_common_universe_semantic_set_sha256",
        "predictor_target_date_scoring_input_semantic_set_sha256",
        "c00_fold_manifest_set_sha256",
        "c00_fold_model_bundle_file_set_sha256",
        "v17_c00_protocol_sha256",
        "v17_c00_runner_sha256",
    ]
    predictor_store = protocol["source_contract"]["forward_daily"][
        "external_raw_evidence_store"
    ]
    outcome_store = protocol["source_contract"]["outcome_daily"][
        "external_raw_evidence_store"
    ]
    assert predictor_store["terminal_audit_cli_argument"] == (
        "--predictor-raw-store-root"
    )
    assert outcome_store["terminal_audit_cli_argument"] == "--outcome-raw-store-root"
    checkpoint_store = protocol["daily_preopen_checkpoint_contract"][
        "sealed_core_store"
    ]
    assert checkpoint_store["cli_argument"] == "--checkpoint-core-store-root"
    picks_fields = protocol["result_contract"]["canonical_materialization_contract"][
        "picks_required_fields"
    ]
    assert picks_fields == list(audit.PICKS_FIELDS)
    assert picks_fields == list(runner.PICKS_FIELDS)
    assert predictor_store["object_key_prefix"].endswith("/predictor/")
    assert outcome_store["object_key_pattern"].startswith(
        "model_v18_shoulder_state/outcome/"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["model_v18_shoulder_state_audit.py"],
    )
    abort_capable = audit.parse_args()
    assert abort_capable.predictor_raw_store_root is None
    assert abort_capable.outcome_raw_store_root is None
    assert abort_capable.checkpoint_core_store_root is None
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_v18_shoulder_state_audit.py",
            "--predictor-raw-store-root",
            str(tmp_path),
            "--outcome-raw-store-root",
            str(tmp_path),
            "--checkpoint-core-store-root",
            str(tmp_path),
        ],
    )
    args = audit.parse_args()
    assert args.predictor_raw_store_root == tmp_path
    assert args.outcome_raw_store_root == tmp_path
    assert args.checkpoint_core_store_root == tmp_path


def test_checkpoint_core_envelope_is_exact_and_rejects_every_structural_drift() -> None:
    value = {
        "schema_version": 1,
        "target_session": "2026-08-05",
        "checkpoint_role": "safety_cash",
        "decision_sequence_number": 0,
        "previous_decision_record_sha256": audit.ZERO_SHA256,
        "nonce_hex": "1" * 64,
        "decision_core": {"opaque_fixture": True},
        "decision_core_sha256": audit.canonical_json_sha256(
            {"opaque_fixture": True}
        ),
    }
    encoded = runner.encode_checkpoint_core_envelope(value)
    assert len(encoded) == 16_384
    assert encoded[:9] == b"TSEV18CP\x01"
    assert audit.parse_checkpoint_core_envelope(encoded) == value
    assert runner.parse_checkpoint_core_envelope(encoded) == value

    mutations: list[bytes] = [encoded[:-1], encoded + b"\x00"]
    changed_magic = bytearray(encoded)
    changed_magic[0] ^= 1
    mutations.append(bytes(changed_magic))
    changed_version = bytearray(encoded)
    changed_version[8] = 2
    mutations.append(bytes(changed_version))
    zero_length = bytearray(encoded)
    zero_length[9:13] = (0).to_bytes(4, "big")
    mutations.append(bytes(zero_length))
    oversized = bytearray(encoded)
    oversized[9:13] = (16_384).to_bytes(4, "big")
    mutations.append(bytes(oversized))
    nonzero_padding = bytearray(encoded)
    nonzero_padding[-1] = 1
    mutations.append(bytes(nonzero_padding))
    body_length = int.from_bytes(encoded[9:13], "big")
    noncanonical_body = b'{"a":1}\n'
    noncanonical = (
        encoded[:9]
        + len(noncanonical_body).to_bytes(4, "big")
        + noncanonical_body
        + bytes(16_384 - 13 - len(noncanonical_body))
    )
    assert body_length > len(noncanonical_body)
    mutations.append(noncanonical)

    for mutated in mutations:
        with pytest.raises(audit.AuditError):
            audit.parse_checkpoint_core_envelope(mutated)


def test_prepare_checkpoint_seals_core_pair_before_proposal_timestamp_and_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, _ = audit.validate_protocol_contract()
    decisions, _ = _forward_ledgers(
        pd.DatetimeIndex([pd.Timestamp("2026-08-05")])
    )
    core_fields = protocol["daily_preopen_checkpoint_contract"][
        "sealed_core_store"
    ]["decision_core_required_fields"]
    primary_core = {field: copy.deepcopy(decisions[0][field]) for field in core_fields}
    activation = {
        "activation_payload_sha256": decisions[0]["activation_payload_sha256"],
        "activation_receipt_sha256": decisions[0]["activation_receipt_sha256"],
        "activation_receipt_commit_sha": decisions[0][
            "activation_receipt_commit_sha"
        ],
    }
    decision_path = tmp_path / "decisions.jsonl"
    decision_path.write_bytes(b"")
    activation_path = tmp_path / "activation.json"
    activation_path.write_text(
        json.dumps({"branch": protocol["branch"]}) + "\n", encoding="utf-8"
    )
    proposal_root = tmp_path / "checkpoint-proposals"
    proposal_root.mkdir()
    external_root = tmp_path / "external-core-store"
    external_root.mkdir()
    monkeypatch.setattr(runner, "_STRICT_RUNTIME_ACTIVE", True)
    monkeypatch.setattr(runner, "DECISION_LEDGER", decision_path)
    monkeypatch.setattr(runner, "ACTIVATION_PAYLOAD", activation_path)
    monkeypatch.setattr(runner, "CHECKPOINT_PROPOSAL_DIR", proposal_root)
    monkeypatch.setattr(
        runner, "_validate_startup_and_module_closure", lambda **kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "_derive_canonical_checkpoint_core",
        lambda target, existing: (copy.deepcopy(primary_core), activation),
    )
    monkeypatch.setattr(
        runner, "validate_checkpoint_decision_core", lambda value, **kwargs: dict(value)
    )

    events: list[str] = []
    installed_payloads: dict[str, dict[str, bytes]] = {}

    def fake_install(parent_fd, *, payloads, label, **kwargs):  # type: ignore[no-untyped-def]
        if label == "checkpoint proposal":
            assert events == ["checkpoint core", "proposal timestamp"]
        events.append(label)
        installed_payloads[label] = dict(payloads)
        return {
            name: (len(payload), hashlib.sha256(payload).hexdigest())
            for name, payload in payloads.items()
        }

    monkeypatch.setattr(runner, "_install_checkpoint_session_directory", fake_install)

    def fake_revalidate(parent_fd, *, label, **kwargs):  # type: ignore[no-untyped-def]
        events.append(f"revalidate {label}")

    monkeypatch.setattr(
        runner, "_revalidate_checkpoint_session_directory", fake_revalidate
    )

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def]
            assert events == ["checkpoint core"]
            events.append("proposal timestamp")
            return cls(2026, 8, 5, 8, 7, 0, tzinfo=tz)

    monkeypatch.setattr(runner, "datetime", FixedDatetime)
    validated_proposals: dict[str, dict[str, object]] = {}

    def fake_validate_proposal(value, *, role, **kwargs):  # type: ignore[no-untyped-def]
        relative = (
            "research/model_v18_shoulder_state_checkpoint_proposals/"
            f"2026-08-05/{role}.json"
        )
        if isinstance(value, (str, Path)):
            current = validated_proposals[role]
        else:
            current = dict(value)
            validated_proposals[role] = current
        return current, audit.canonical_json_sha256(current), relative

    monkeypatch.setattr(runner, "validate_checkpoint_proposal", fake_validate_proposal)
    monkeypatch.setattr(runner, "_validate_checkpoint_proposal_pair", lambda value: None)

    result = runner.prepare_checkpoint(
        session_date="2026-08-05",
        checkpoint_core_store_root=external_root,
    )
    assert events == [
        "checkpoint core",
        "proposal timestamp",
        "checkpoint proposal",
        "revalidate checkpoint core",
        "revalidate checkpoint proposal",
    ]
    assert [item["checkpoint_role"] for item in result["role_artifacts"]] == [
        "safety_cash",
        "primary",
    ]
    proposal_values = [
        json.loads(payload.decode("utf-8"))
        for payload in installed_payloads["checkpoint proposal"].values()
    ]
    assert {item["created_at"] for item in proposal_values} == {
        "2026-08-05T08:07:00.000000+09:00"
    }
    core_values = [
        runner.parse_checkpoint_core_envelope(payload)["decision_core"]
        for payload in installed_payloads["checkpoint core"].values()
    ]
    assert core_values[0] == core_values[1] == primary_core
    assert all(
        pd.Timestamp(item["computed_at"])
        < pd.Timestamp(proposal_values[0]["created_at"])
        for item in core_values
    )


def test_checkpoint_terminal_remote_pair_and_git_data_mutations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = _remote_checkpoint_fixture(tmp_path)
    protocol = fixture["protocol"]
    decisions = fixture["decisions"]
    responses = fixture["responses"]
    repository = fixture["repository"]

    def no_local_git(*args, **kwargs):  # type: ignore[no-untyped-def]
        pytest.fail("daily checkpoint audit consulted local Git")

    monkeypatch.setattr(audit, "_git_command", no_local_git)

    def validate(fetcher=None, decision_rows=None):  # type: ignore[no-untyped-def]
        return audit.validate_checkpoint_evidence(
            decisions if decision_rows is None else decision_rows,
            proposal_directory=fixture["proposal_root"],
            checkpoint_core_store_root=fixture["core_root"],
            protocol=protocol,
            activation_payload=fixture["activation_payload"],
            runner_sha256="a" * 64,
            external_identity_registry={},
            github_fetcher=fixture["fetch"] if fetcher is None else fetcher,
        )

    bindings = validate()
    assert set(bindings) == {
        "checkpoint_proposal_set_sha256",
        "checkpoint_core_object_set_sha256",
        "checkpoint_evidence_set_sha256",
    }
    assert all(audit.SHA256_RE.fullmatch(value) for value in bindings.values())

    primary_core_path = (
        Path(fixture["core_root"])
        / "model_v18_shoulder_state/checkpoint-core/2026-08-05/primary.bin"
    )
    premature_proposal_core = audit.parse_checkpoint_core_envelope(
        primary_core_path.read_bytes()
    )
    premature_proposal_core["decision_core"]["computed_at"] = (
        "2026-08-05T08:06:00+09:00"
    )
    premature_proposal_core["decision_core_sha256"] = audit.canonical_json_sha256(
        premature_proposal_core["decision_core"]
    )
    with pytest.raises(audit.AuditError, match="proposal predates"):
        audit.validate_checkpoint_core_object(
            premature_proposal_core,
            fixture["proposals"]["primary"],
            protocol=protocol,
        )

    # Only the primary first-run server created_at is cutoff authority.  Both
    # role runs may finish unsuccessfully and after the cutoff without changing
    # the already-fixed primary resolution, provided their stable terminal
    # projections are bound before decision materialization/outcome attachment.
    late_decisions = copy.deepcopy(decisions)
    primary_observation = late_decisions[0]["checkpoint_workflow_run_observation"]
    assert isinstance(primary_observation, dict)
    primary_projection = primary_observation["canonical_projection"]
    assert isinstance(primary_projection, dict)
    primary_projection["conclusion"] = "failure"
    primary_projection["updated_at"] = "2026-08-05T10:00:00+09:00"
    primary_observation["retrieved_at"] = "2026-08-05T10:01:00+09:00"
    primary_observation["http_date"] = "Wed, 05 Aug 2026 01:00:59 GMT"
    primary_observation["response_body_sha256"] = hashlib.sha256(
        audit.canonical_json_bytes(primary_projection)
    ).hexdigest()
    primary_observation["response_headers_sha256"] = audit.canonical_json_sha256(
        {
            "status": 200,
            "content_type": primary_observation["content_type"],
            "date": primary_observation["http_date"],
            "etag": primary_observation["etag"],
        }
    )
    primary_observation["observation_sha256"] = audit.canonical_json_sha256(
        primary_observation, exclude_fields={"observation_sha256"}
    )
    late_decisions[0]["checkpoint_workflow_run_updated_at"] = (
        "2026-08-05T10:00:00+09:00"
    )
    late_decisions[0]["checkpoint_workflow_run_observed_at"] = (
        "2026-08-05T10:01:00+09:00"
    )
    late_decisions[0]["decision_materialized_at"] = "2026-08-05T10:02:00+09:00"
    late_decisions = _chain(
        [
            {
                key: value
                for key, value in late_decisions[0].items()
                if key not in audit.CHAIN_COLUMNS
            }
        ]
    )
    assert audit.validate_decision_records(late_decisions)[0][
        "checkpoint_resolution_reason"
    ] == "primary_commitment_timely"

    def failed_late_terminal_runs(endpoint: str):
        value = copy.deepcopy(responses[endpoint])
        if "/actions/runs" in endpoint:
            raw_runs = (
                value.get("workflow_runs", [])
                if isinstance(value, dict)
                else []
            )
            targets = raw_runs if raw_runs else [value]
            for raw in targets:
                if isinstance(raw, dict):
                    raw["conclusion"] = "failure"
                    raw["updated_at"] = "2026-08-05T10:00:00+09:00"
        return value

    late_bindings = validate(
        failed_late_terminal_runs, decision_rows=late_decisions
    )
    assert set(late_bindings) == set(bindings)

    def mutating_fetch(mutator):  # type: ignore[no-untyped-def]
        def fetch(endpoint: str):  # type: ignore[no-untyped-def]
            value = copy.deepcopy(responses[endpoint])
            return mutator(endpoint, value)

        return fetch

    safety_git_endpoint = (
        f"repos/{repository}/git/commits/{fixture['safety_commit']}"
    )
    primary_tree_endpoint = (
        f"repos/{repository}/git/trees/{fixture['primary_tree']}?recursive=1"
    )
    primary_blob_endpoint = (
        f"repos/{repository}/git/blobs/{fixture['primary_blob']}"
    )
    primary_compare_endpoint = (
        f"repos/{repository}/compare/{fixture['safety_commit']}..."
        f"{fixture['primary_commit']}"
    )

    def wrong_parent(endpoint, value):  # type: ignore[no-untyped-def]
        if endpoint == safety_git_endpoint:
            value["parents"] = [{"sha": "0" * 40}]
        return value

    with pytest.raises(audit.AuditError, match="parent/message/time"):
        validate(mutating_fetch(wrong_parent))

    def extra_tree_path(endpoint, value):  # type: ignore[no-untyped-def]
        if endpoint == primary_tree_endpoint:
            value["tree"].append(
                {
                    "path": "zz-unregistered-checkpoint-file",
                    "mode": "100644",
                    "type": "blob",
                    "sha": "9" * 40,
                    "size": 1,
                }
            )
        return value

    with pytest.raises(audit.AuditError, match="changed more than"):
        validate(mutating_fetch(extra_tree_path))

    def changed_blob(endpoint, value):  # type: ignore[no-untyped-def]
        if endpoint == primary_blob_endpoint:
            value["content"] = "QQ=="
            value["size"] = 1
        return value

    with pytest.raises(audit.AuditError, match="blob bytes"):
        validate(mutating_fetch(changed_blob))

    protected_blob_endpoint = next(
        key
        for key in responses
        if "/git/blobs/" in key
        and key not in {primary_blob_endpoint, f"repos/{repository}/git/blobs/{fixture['safety_blob']}"}
    )

    def changed_protected_blob(endpoint, value):  # type: ignore[no-untyped-def]
        if endpoint == protected_blob_endpoint:
            value["content"] = "QQ=="
            value["size"] = 1
        return value

    with pytest.raises(audit.AuditError, match="blob bytes"):
        validate(mutating_fetch(changed_protected_blob))

    def alternate_compare_path(endpoint, value):  # type: ignore[no-untyped-def]
        if endpoint == primary_compare_endpoint:
            value["files"][0]["filename"] = "alternate/primary.json"
        return value

    with pytest.raises(audit.AuditError, match="one exact addition"):
        validate(mutating_fetch(alternate_compare_path))

    primary_actions_query = (
        f"repos/{repository}/actions/runs?event=pull_request"
        f"&head_sha={fixture['primary_commit']}&per_page=100"
    )
    original_run = copy.deepcopy(responses[primary_actions_query]["workflow_runs"][0])
    earlier_run = copy.deepcopy(original_run)
    earlier_run["id"] = 60_000
    earlier_run["html_url"] = f"https://github.com/{repository}/actions/runs/60000"

    def omitted_earlier_run(endpoint: str):
        if endpoint == primary_actions_query:
            return {"total_count": 2, "workflow_runs": [original_run, earlier_run]}
        if endpoint == f"repos/{repository}/actions/runs/60000":
            return copy.deepcopy(earlier_run)
        return copy.deepcopy(responses[endpoint])

    with pytest.raises(audit.AuditError, match="selected public evidence"):
        validate(omitted_earlier_run)

    cutoff_run = copy.deepcopy(original_run)
    cutoff_run.update(
        {
            "created_at": "2026-08-05T08:58:59+09:00",
            "run_started_at": "2026-08-05T08:58:59+09:00",
            "updated_at": "2026-08-05T08:58:59+09:00",
        }
    )

    def cutoff_equal_primary(endpoint: str):
        if endpoint == primary_actions_query:
            return {"total_count": 1, "workflow_runs": [cutoff_run]}
        if endpoint == f"repos/{repository}/actions/runs/70002":
            return copy.deepcopy(cutoff_run)
        return copy.deepcopy(responses[endpoint])

    with pytest.raises(audit.AuditError, match="mandatory primary"):
        validate(cutoff_equal_primary)

    safety_actions_query = (
        f"repos/{repository}/actions/runs?event=pull_request"
        f"&head_sha={fixture['safety_commit']}&per_page=100"
    )
    reordered_safety_run = copy.deepcopy(
        responses[safety_actions_query]["workflow_runs"][0]
    )
    reordered_safety_run.update(
        {
            "created_at": "2026-08-05T08:40:00+09:00",
            "run_started_at": "2026-08-05T08:41:00+09:00",
            "updated_at": "2026-08-05T08:42:00+09:00",
        }
    )

    def reversed_workflow_creation_order(endpoint: str):
        if endpoint == safety_actions_query:
            return {"total_count": 1, "workflow_runs": [reordered_safety_run]}
        if endpoint == f"repos/{repository}/actions/runs/70001":
            return copy.deepcopy(reordered_safety_run)
        return copy.deepcopy(responses[endpoint])

    with pytest.raises(audit.AuditError, match="mandatory primary"):
        validate(reversed_workflow_creation_order)

    post_materialization_safety = copy.deepcopy(
        responses[safety_actions_query]["workflow_runs"][0]
    )
    post_materialization_safety["updated_at"] = "2026-08-05T08:41:00+09:00"

    def safety_completed_after_materialization(endpoint: str):
        if endpoint == safety_actions_query:
            return {
                "total_count": 1,
                "workflow_runs": [post_materialization_safety],
            }
        if endpoint == f"repos/{repository}/actions/runs/70001":
            return copy.deepcopy(post_materialization_safety)
        return copy.deepcopy(responses[endpoint])

    with pytest.raises(audit.AuditError, match="postdates materialization"):
        validate(safety_completed_after_materialization)

    proposal_query_page_one = next(
        key
        for key in responses
        if "commits?sha=" in key
        and "checkpoint_proposals" in key
        and key.endswith("page=1")
    )

    def extra_history(endpoint, value):  # type: ignore[no-untyped-def]
        if endpoint == proposal_query_page_one:
            value.append({"sha": "9" * 40})
        return value

    with pytest.raises(audit.AuditError, match="extra or missing"):
        validate(mutating_fetch(extra_history))

    ref_endpoint = f"repos/{repository}/git/ref/heads/{protocol['branch']}"
    post_commit = "9" * 40
    descendant_prefix = (
        f"repos/{repository}/compare/{fixture['primary_commit']}...{post_commit}"
        "?per_page=100&page="
    )
    workflow_history_page_one = next(
        key
        for key in responses
        if "commits?sha=" in key
        and quote(".github/workflows/tests.yml", safe="") in key
        and key.endswith("page=1")
    )

    def protected_post_terminal_mutation(endpoint: str):
        if endpoint == ref_endpoint:
            value = copy.deepcopy(responses[endpoint])
            value["object"]["sha"] = post_commit
            value["object"]["url"] = (
                f"https://api.github.com/repos/{repository}/git/commits/{post_commit}"
            )
            return value
        if endpoint == descendant_prefix + "1":
            return {
                "status": "ahead",
                "ahead_by": 1,
                "behind_by": 0,
                "total_commits": 1,
                "base_commit": {"sha": fixture["primary_commit"]},
                "merge_base_commit": {"sha": fixture["primary_commit"]},
                "commits": [{"sha": post_commit}],
            }
        if endpoint == descendant_prefix + "2":
            return {
                "status": "ahead",
                "ahead_by": 1,
                "behind_by": 0,
                "total_commits": 1,
                "base_commit": {"sha": fixture["primary_commit"]},
                "merge_base_commit": {"sha": fixture["primary_commit"]},
                "commits": [],
            }
        value = copy.deepcopy(responses[endpoint])
        if endpoint == workflow_history_page_one:
            value.insert(0, {"sha": post_commit})
        return value

    with pytest.raises(audit.AuditError, match="protected path changed"):
        validate(protected_post_terminal_mutation)

    def stale_remote_ref(endpoint: str):
        if endpoint == ref_endpoint:
            value = copy.deepcopy(responses[endpoint])
            value["object"]["sha"] = post_commit
            value["object"]["url"] = (
                f"https://api.github.com/repos/{repository}/git/commits/{post_commit}"
            )
            return value
        if endpoint == descendant_prefix + "1":
            return {
                "status": "diverged",
                "ahead_by": 1,
                "behind_by": 1,
                "total_commits": 1,
                "base_commit": {"sha": fixture["primary_commit"]},
                "merge_base_commit": {"sha": "0" * 40},
                "commits": [{"sha": post_commit}],
            }
        return copy.deepcopy(responses[endpoint])

    with pytest.raises(audit.AuditError, match="not a remote ancestor"):
        validate(stale_remote_ref)

    safety_core = (
        Path(fixture["core_root"])
        / "model_v18_shoulder_state/checkpoint-core/2026-08-05/safety_cash.bin"
    )
    alias = tmp_path / "checkpoint-core-hardlink.bin"
    os.link(safety_core, alias)
    with pytest.raises(audit.AuditError, match="single-link regular file"):
        validate()


def test_numeric_fold_bundle_reconstructs_without_runner_import_or_pickle() -> None:
    protocol, _ = audit.validate_protocol_contract()
    bundle = _synthetic_fold_bundle()
    expected_hash = bundle["fold_model_bundle_sha256"]
    assert audit.validate_fold_model_bundle(bundle, protocol)[
        "fold_model_bundle_sha256"
    ] == expected_hash
    assert runner.validate_c00_model_bundle(bundle) == expected_hash

    features = np.zeros((3, len(audit.C00_FEATURES)), dtype=float)
    features[0, 1] = np.nan
    features[1, 4] = np.nan
    features[2, [1, 4]] = np.nan
    independent = audit.reconstruct_c00_scores(bundle, features, protocol)
    runtime = runner.predict_c00_model_bundle(bundle, features)
    np.testing.assert_allclose(independent, runtime, rtol=0.0, atol=1e-15)

    tampered = copy.deepcopy(bundle)
    tampered["scaler_scale"]["values"][0] = 0.0
    with pytest.raises(audit.AuditError, match="hash|positive"):
        audit.validate_fold_model_bundle(tampered, protocol)
    with pytest.raises(runner.V18Error, match="hash|positive"):
        runner.validate_c00_model_bundle(tampered)


def test_source_manifest_is_exact_d_minus_one_and_fail_closed_hash_is_nonnull() -> None:
    protocol, _ = audit.validate_protocol_contract()
    complete = _source_manifest()
    assert audit.validate_source_manifest(
        complete, session_date="2026-08-05", protocol=protocol
    )["source_complete"] is True
    assert runner.validate_source_manifest(
        complete, session_date="2026-08-05"
    )[0]["source_complete"] is True

    missing = _source_manifest(complete=False)
    validated = audit.validate_source_manifest(
        missing, session_date="2026-08-05", protocol=protocol
    )
    assert validated["source_complete"] is False
    assert validated["source_manifest_sha256"] not in {None, audit.ZERO_SHA256}

    same_day = copy.deepcopy(complete)
    same_day["latest_required_source_session"] = "2026-08-05"
    same_day["source_manifest_sha256"] = audit.canonical_json_sha256(
        same_day, exclude_fields={"source_manifest_sha256"}
    )
    with pytest.raises(audit.AuditError, match="predecessor"):
        audit.validate_source_manifest(
            same_day, session_date="2026-08-05", protocol=protocol
        )

    incomplete_history = copy.deepcopy(complete)
    for field in (
        "source_files",
        "source_urls",
        "source_object_keys",
        "source_byte_counts",
        "source_sha256",
    ):
        incomplete_history[field] = incomplete_history[field][1:]
    incomplete_history["source_set_sha256"] = audit.canonical_json_sha256(
        [
            {
                "object_key": key,
                "file": name,
                "url": url,
                "byte_count": count,
                "sha256": digest,
            }
            for key, name, url, count, digest in zip(
                incomplete_history["source_object_keys"],
                incomplete_history["source_files"],
                incomplete_history["source_urls"],
                incomplete_history["source_byte_counts"],
                incomplete_history["source_sha256"],
                strict=True,
            )
        ]
    )
    incomplete_history["source_manifest_sha256"] = audit.canonical_json_sha256(
        incomplete_history, exclude_fields={"source_manifest_sha256"}
    )
    with pytest.raises(audit.AuditError, match="complete|registry"):
        audit.validate_source_manifest(
            incomplete_history, session_date="2026-08-05", protocol=protocol
        )

    historical_tamper = copy.deepcopy(complete)
    historical_tamper["source_sha256"][0] = "8" * 64
    historical_tamper["source_set_sha256"] = audit.canonical_json_sha256(
        [
            {
                "object_key": key,
                "file": name,
                "url": url,
                "byte_count": count,
                "sha256": digest,
            }
            for key, name, url, count, digest in zip(
                historical_tamper["source_object_keys"],
                historical_tamper["source_files"],
                historical_tamper["source_urls"],
                historical_tamper["source_byte_counts"],
                historical_tamper["source_sha256"],
                strict=True,
            )
        ]
    )
    historical_tamper["source_manifest_sha256"] = audit.canonical_json_sha256(
        historical_tamper, exclude_fields={"source_manifest_sha256"}
    )
    with pytest.raises(audit.AuditError, match="historical predictor SHA"):
        audit.validate_source_manifest(
            historical_tamper, session_date="2026-08-05", protocol=protocol
        )
    with pytest.raises(runner.V18Error, match="historical|SHA"):
        runner.validate_source_manifest(
            historical_tamper, session_date="2026-08-05"
        )


def test_source_failure_builder_accepts_empty_and_canonical_partial_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, _ = audit.validate_protocol_contract()
    timestamps = {
        "runtime_lock_verified_at": "2026-08-05T07:00:00+09:00",
        "created_at": "2026-08-05T08:10:00+09:00",
        "sealed_at": "2026-08-05T08:11:00+09:00",
    }
    empty = runner.build_source_failure_manifest(
        session_date="2026-08-05",
        failure_reason="source_missing_before_cutoff",
        **timestamps,
    )
    assert empty["source_files"] == []
    assert empty["source_received_at"] is None
    assert audit.validate_source_manifest(
        empty,
        session_date="2026-08-05",
        protocol=protocol,
    )["source_complete"] is False

    expected_files = ["stq_20260803.pdf", "stq_20260804.pdf"]
    kinds = {item: "daily" for item in expected_files}
    monkeypatch.setattr(
        runner,
        "_expected_predictor_files",
        lambda latest: (expected_files, kinds),
    )
    monkeypatch.setattr(runner, "_historical_predictor_metadata", lambda: {})
    supplied = tmp_path / expected_files[0]
    supplied.write_bytes(b"%PDF-partial-source-before-cutoff")
    raw_store = tmp_path / "raw-store"
    raw_store.mkdir()
    partial = runner.build_source_failure_manifest(
        session_date="2026-08-05",
        failure_reason="source_partial_or_late_before_cutoff",
        source_pdfs=[supplied],
        source_file_names=[expected_files[0]],
        source_urls=[f"https://www.jpx.co.jp/example/{expected_files[0]}"],
        source_received_at="2026-08-05T08:00:00+09:00",
        predictor_raw_store_root=raw_store,
        **timestamps,
    )
    assert partial["source_files"] == [expected_files[0]]
    assert audit.validate_source_manifest(
        partial,
        session_date="2026-08-05",
        protocol=protocol,
        predictor_raw_store_root=raw_store,
    )["source_complete"] is False

    supplied_second = tmp_path / expected_files[1]
    supplied_second.write_bytes(b"%PDF-latest-source-before-cutoff")
    another_store = tmp_path / "another-store"
    another_store.mkdir()
    with pytest.raises(runner.V18Error, match="canonical order"):
        runner.build_source_failure_manifest(
            session_date="2026-08-05",
            failure_reason="source_partial_or_late_before_cutoff",
            source_pdfs=[supplied_second, supplied],
            source_file_names=[expected_files[1], expected_files[0]],
            source_urls=[
                f"https://www.jpx.co.jp/example/{expected_files[1]}",
                f"https://www.jpx.co.jp/example/{expected_files[0]}",
            ],
            source_received_at="2026-08-05T08:00:00+09:00",
            predictor_raw_store_root=another_store,
            **timestamps,
        )

    missing_receipt = copy.deepcopy(partial)
    missing_receipt["source_received_at"] = None
    missing_receipt["source_manifest_sha256"] = audit.canonical_json_sha256(
        missing_receipt, exclude_fields={"source_manifest_sha256"}
    )
    with pytest.raises(audit.AuditError, match="fail-closed"):
        audit.validate_source_manifest(
            missing_receipt,
            session_date="2026-08-05",
            protocol=protocol,
            predictor_raw_store_root=raw_store,
        )


def test_predictor_raw_object_bytes_mutation_and_symlink_fail_closed(
    tmp_path: Path,
) -> None:
    protocol, _ = audit.validate_protocol_contract()
    object_key = "model_v18_shoulder_state/predictor/daily/stq_20260804.pdf"
    raw_path = tmp_path / object_key
    raw_path.parent.mkdir(parents=True)
    original_bytes = b"%PDF-independent-predictor-evidence"
    raw_path.write_bytes(original_bytes)

    manifest = _source_manifest(complete=False)
    source_object = {
        "object_key": object_key,
        "file": "stq_20260804.pdf",
        "url": "https://www.jpx.co.jp/example/stq_20260804.pdf",
        "byte_count": len(original_bytes),
        "sha256": hashlib.sha256(original_bytes).hexdigest(),
    }
    manifest.update(
        {
            "source_files": [source_object["file"]],
            "source_urls": [source_object["url"]],
            "source_object_keys": [source_object["object_key"]],
            "source_byte_counts": [source_object["byte_count"]],
            "source_sha256": [source_object["sha256"]],
            "source_set_sha256": audit.canonical_json_sha256([source_object]),
            "source_received_at": "2026-08-05T08:00:00+09:00",
        }
    )
    manifest["source_manifest_sha256"] = audit.canonical_json_sha256(
        manifest, exclude_fields={"source_manifest_sha256"}
    )
    assert audit.validate_source_manifest(
        manifest,
        session_date="2026-08-05",
        protocol=protocol,
        predictor_raw_store_root=tmp_path,
    )["source_complete"] is False

    metadata = raw_path.stat()
    aliased_registry = {
        (int(metadata.st_dev), int(metadata.st_ino)):
        "model_v18_shoulder_state/outcome/2026-08-05.pdf"
    }
    with pytest.raises(audit.AuditError, match="alias the same physical inode"):
        audit.validate_source_manifest(
            manifest,
            session_date="2026-08-05",
            protocol=protocol,
            predictor_raw_store_root=tmp_path,
            external_identity_registry=aliased_registry,
        )

    hardlink = tmp_path / "raw-hardlink.pdf"
    os.link(raw_path, hardlink)
    with pytest.raises(audit.AuditError, match="single-link regular file"):
        audit.validate_source_manifest(
            manifest,
            session_date="2026-08-05",
            protocol=protocol,
            predictor_raw_store_root=tmp_path,
        )
    hardlink.unlink()

    raw_path.write_bytes(original_bytes + b"-mutated")
    with pytest.raises(audit.AuditError, match="bytes"):
        audit.validate_source_manifest(
            manifest,
            session_date="2026-08-05",
            protocol=protocol,
            predictor_raw_store_root=tmp_path,
        )

    raw_path.unlink()
    symlink_target = tmp_path / "sealed-other.pdf"
    symlink_target.write_bytes(original_bytes)
    raw_path.symlink_to(symlink_target)
    with pytest.raises(audit.AuditError, match="symlink"):
        audit.validate_source_manifest(
            manifest,
            session_date="2026-08-05",
            protocol=protocol,
            predictor_raw_store_root=tmp_path,
        )


def test_external_store_parent_symlink_swap_cannot_redirect_opened_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "sealed-store"
    predictor = root / "model_v18_shoulder_state" / "predictor"
    daily = predictor / "daily"
    daily.mkdir(parents=True)
    source_name = "stq_20260804.pdf"
    (daily / source_name).write_bytes(b"%PDF-original-sealed-bytes")
    attacker = tmp_path / "attacker"
    (attacker / "daily").mkdir(parents=True)
    (attacker / "daily" / source_name).write_bytes(b"%PDF-substituted-bytes")

    real_open = audit.os.open
    swapped = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):  # type: ignore[no-untyped-def]
        nonlocal swapped
        if path == "predictor" and dir_fd is not None and not swapped:
            swapped = True
            moved = predictor.with_name("predictor-original")
            predictor.rename(moved)
            predictor.symlink_to(attacker, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(audit.os, "open", racing_open)
    with pytest.raises(audit.AuditError, match="raced|symlink"):
        audit._read_external_object_bytes(
            root,
            f"model_v18_shoulder_state/predictor/daily/{source_name}",
            required_prefix="model_v18_shoulder_state/predictor/",
        )
    assert swapped is True


def test_clean_room_g0_semantics_scores_top2_and_same_day_rejection() -> None:
    protocol, _ = audit.validate_protocol_contract()
    target = pd.Timestamp("2026-08-05")
    prices = _synthetic_predictor_prices(target=str(target.date()))
    panel = audit.build_clean_room_g0_panel(prices, target)
    runtime_panel = runner.build_forward_c00_panel(prices, target)
    target_rows = panel.loc[panel["date"].eq(target)].copy()
    assert len(target_rows) == 3
    assert target_rows["common_score_eligible"].all()
    assert target_rows["oc_return_pct"].isna().all()
    assert target_rows["feature_source_max_date"].eq(
        prices["date"].max()
    ).all()

    g0_columns = (
        "date",
        "code",
        "name",
        "oc_return_pct",
        "common_training_eligible",
        "common_score_eligible",
        "feature_source_max_date",
        *audit.C00_FEATURES,
    )
    assert audit.semantic_rows_sha256(panel, g0_columns) == (
        runner.semantic_frame_sha256(runtime_panel, g0_columns)
    )
    shuffled = panel.sample(frac=1.0, random_state=18)
    assert audit.semantic_rows_sha256(shuffled, g0_columns) == (
        audit.semantic_rows_sha256(panel, g0_columns)
    )
    mutated = panel.copy()
    mutated.loc[mutated.index[-1], "oc_last"] += 0.01
    assert audit.semantic_rows_sha256(mutated, g0_columns) != (
        audit.semantic_rows_sha256(panel, g0_columns)
    )

    bundle = _synthetic_fold_bundle()
    scoring = target_rows.loc[:, list(audit.C00_FEATURES)]
    independent_scores = audit.reconstruct_c00_scores(bundle, scoring, protocol)
    runtime_scores = runner.predict_c00_model_bundle(bundle, scoring)
    np.testing.assert_allclose(
        independent_scores, runtime_scores, rtol=0.0, atol=1e-15
    )
    independent_top2 = (
        target_rows.loc[:, ["code"]]
        .assign(model_score=independent_scores)
        .sort_values(
            ["model_score", "code"],
            ascending=[False, True],
            kind="stable",
        )
        .head(2)
    )
    runtime_top2 = (
        target_rows.loc[:, ["code"]]
        .assign(model_score=runtime_scores)
        .sort_values(
            ["model_score", "code"],
            ascending=[False, True],
            kind="stable",
        )
        .head(2)
    )
    pd.testing.assert_frame_equal(independent_top2, runtime_top2)

    same_day = prices.loc[prices["date"].eq(prices["date"].max())].copy()
    same_day["date"] = target
    with pytest.raises(audit.AuditError, match="strictly before"):
        audit.build_clean_room_g0_panel(
            pd.concat([prices, same_day], ignore_index=True), target
        )


def test_decision_fail_closed_hash_nullability_and_cash_costs() -> None:
    session = audit.load_registered_calendar()[:1]
    decisions, outcomes = _forward_ledgers(session)
    fail_model = {
        key: value
        for key, value in decisions[0].items()
        if key not in audit.CHAIN_COLUMNS
    }
    fail_model.update(
        {
            "c00_fold_manifest_sha256": None,
            "fold_model_bundle_file_sha256": None,
            "model_complete": False,
            "selected_source_rank": None,
            "c00_rank1_code": None,
            "c00_rank1_score": None,
            "c02_rank2_code": None,
            "c02_rank2_score": None,
            "candidate_selected_code": None,
            "decision": "fail_closed_model_fold",
            "failure_reason": "fold_unavailable_before_target_month",
        }
    )
    validated = audit.validate_decision_records(
        _chain([_rehash_checkpoint_core(fail_model)])
    )
    assert validated[0]["source_manifest_sha256"] == "e" * 64
    assert validated[0]["state_manifest_sha256"] == "1" * 64
    assert validated[0]["c00_fold_manifest_sha256"] is None

    zero_sentinel = copy.deepcopy(fail_model)
    zero_sentinel["c00_fold_manifest_sha256"] = audit.ZERO_SHA256
    zero_sentinel["fold_model_bundle_file_sha256"] = audit.ZERO_SHA256
    with pytest.raises(audit.AuditError, match="nonzero"):
        audit.validate_decision_records(_chain([zero_sentinel]))

    pre_activation = {
        key: value
        for key, value in decisions[0].items()
        if key not in audit.CHAIN_COLUMNS
    }
    pre_activation["computed_at"] = "2026-08-03T08:00:00+09:00"
    with pytest.raises(audit.AuditError, match="timestamp|cutoff"):
        audit.validate_decision_records(_chain([pre_activation]))

    cash = {
        key: value
        for key, value in decisions[0].items()
        if key not in audit.CHAIN_COLUMNS
    }
    cash.update(
        {
            "three_month_medians_pct": [0.0, 0.0, 0.0],
            "state_value_pct": 0.0,
            "selected_source_rank": None,
            "candidate_selected_code": None,
            "decision": "cash_state_zero",
            "failure_reason": "state_value_exact_zero",
        }
    )
    cash_decisions = _chain([_rehash_checkpoint_core(cash)])
    cash_outcome = {
        key: value
        for key, value in outcomes[0].items()
        if key not in audit.CHAIN_COLUMNS
    }
    cash_outcome.update(
        {
            "decision_record_sha256": cash_decisions[0]["record_sha256"],
            "candidate_outcome_observed": False,
            "candidate_gross_return_pct": 0.0,
            "candidate_net20_return_pct": 0.0,
            "candidate_net40_return_pct": 0.0,
            "candidate_net60_return_pct": 0.0,
        }
    )
    assert audit.validate_outcome_records(
        _chain([cash_outcome]), cash_decisions
    )[0]["candidate_net60_return_pct"] == 0.0


def test_outcome_manifest_binds_external_bytes_decision_and_open_close(
    tmp_path: Path,
) -> None:
    protocol, _ = audit.validate_protocol_contract()
    decisions, outcomes = _forward_ledgers(audit.load_registered_calendar()[:1])
    raw_path = (
        tmp_path
        / "model_v18_shoulder_state"
        / "outcome"
        / f"{decisions[0]['session_date']}.pdf"
    )
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(b"%PDF-synthetic-independent-audit-test")
    manifest = _outcome_manifest(decisions[0], raw_path)
    validated = audit.validate_outcome_manifest(
        manifest,
        decisions[0],
        protocol,
        raw_store_root=tmp_path,
    )
    assert validated["source_sha256"] == audit.sha256_file(raw_path)

    linked_outcome = {
        key: value
        for key, value in outcomes[0].items()
        if key not in audit.CHAIN_COLUMNS
    }
    linked_outcome.update(
        {
            "outcome_source_sha256": manifest["source_sha256"],
            "outcome_manifest_sha256": manifest["outcome_manifest_sha256"],
            "outcome_received_at": manifest["source_received_at"],
            "rank1_oc_return_pct": manifest[
                "rank1_recomputed_oc_return_pct"
            ],
            "rank2_oc_return_pct": manifest[
                "rank2_recomputed_oc_return_pct"
            ],
        }
    )
    rank1_return = float(manifest["rank1_recomputed_oc_return_pct"])
    rank2_return = float(manifest["rank2_recomputed_oc_return_pct"])
    selected_return = (
        rank1_return
        if decisions[0]["decision"] == "selected_rank1"
        else rank2_return
    )
    linked_outcome.update(
        {
            "candidate_gross_return_pct": selected_return,
            "candidate_net20_return_pct": selected_return - 0.2,
            "candidate_net40_return_pct": selected_return - 0.4,
            "candidate_net60_return_pct": selected_return - 0.6,
            "c00_top1_gross_return_pct": rank1_return,
            "c00_top1_net20_return_pct": rank1_return - 0.2,
            "c00_top1_net40_return_pct": rank1_return - 0.4,
            "c00_top1_net60_return_pct": rank1_return - 0.6,
            "c02_rank2_gross_return_pct": rank2_return,
            "c02_rank2_net20_return_pct": rank2_return - 0.2,
            "c02_rank2_net40_return_pct": rank2_return - 0.4,
            "c02_rank2_net60_return_pct": rank2_return - 0.6,
        }
    )
    assert audit.validate_outcome_records(
        _chain([linked_outcome]),
        decisions,
        outcome_manifests={str(manifest["outcome_manifest_sha256"]): manifest},
    )[0]["outcome_manifest_sha256"] == manifest["outcome_manifest_sha256"]

    premature_manifest = copy.deepcopy(manifest)
    premature_manifest["created_at"] = (
        f"{decisions[0]['session_date']}T15:59:00+09:00"
    )
    premature_manifest["outcome_manifest_sha256"] = audit.canonical_json_sha256(
        premature_manifest, exclude_fields={"outcome_manifest_sha256"}
    )
    with pytest.raises(audit.AuditError, match="outcome DAG"):
        audit.validate_outcome_manifest(
            premature_manifest,
            decisions[0],
            protocol,
            raw_store_root=tmp_path,
        )

    premature_record = copy.deepcopy(linked_outcome)
    premature_record["computed_at"] = (
        f"{decisions[0]['session_date']}T15:59:00+09:00"
    )
    with pytest.raises(audit.AuditError, match="causality"):
        audit.validate_outcome_records(_chain([premature_record]), decisions)

    traversal = copy.deepcopy(manifest)
    traversal["raw_source_object_key"] = "../escape.pdf"
    traversal["outcome_manifest_sha256"] = audit.canonical_json_sha256(
        traversal, exclude_fields={"outcome_manifest_sha256"}
    )
    with pytest.raises(audit.AuditError, match="object key"):
        audit.validate_outcome_manifest(
            traversal,
            decisions[0],
            protocol,
            raw_store_root=tmp_path,
        )


def test_first_counted_session_uses_successful_workflow_times_without_delay() -> None:
    calendar = audit.load_registered_calendar()
    assert audit.first_counted_session(
        workflow_run_updated_at="2026-08-04T23:50:00+09:00",
        workflow_run_observed_at="2026-08-05T08:00:00+09:00",
        calendar=calendar,
    ) == pd.Timestamp("2026-08-05")
    assert runner.first_counted_session(
        workflow_run_updated_at="2026-08-04T23:50:00+09:00",
        workflow_run_observed_at="2026-08-05T08:00:00+09:00",
        calendar=calendar,
    ) == pd.Timestamp("2026-08-05")
    assert audit.first_counted_session(
        workflow_run_updated_at="2026-08-05T09:00:00+09:00",
        workflow_run_observed_at="2026-08-05T09:01:00+09:00",
        calendar=calendar,
    ) == pd.Timestamp("2026-08-06")
    with pytest.raises(audit.AuditError, match="after the fixed first-session cutoff"):
        audit.first_counted_session(
            workflow_run_updated_at="2026-08-04T23:50:00+09:00",
            workflow_run_observed_at="2026-08-05T09:00:00+09:00",
            calendar=calendar,
        )
    with pytest.raises(runner.V18Error, match="missed the fixed first cutoff"):
        runner.first_counted_session(
            workflow_run_updated_at="2026-08-04T23:50:00+09:00",
            workflow_run_observed_at="2026-08-05T09:00:00+09:00",
            calendar=calendar,
        )


def test_github_observation_self_hash_projection_and_flat_derivation_mutations() -> None:
    protocol, _ = audit.validate_protocol_contract()
    decisions, _ = _forward_ledgers(audit.load_registered_calendar()[:1])
    assert len(audit.validate_decision_records(decisions)) == 1
    commit = decisions[0]["receipt_commit_observation"]
    audit.validate_github_observation(
        commit,
        protocol,
        observation_kind="commit",
        requested_commit_sha="d" * 40,
    )
    workflow = decisions[0]["receipt_workflow_run_observation"]
    audit.validate_github_workflow_observation(
        workflow,
        protocol,
        expected_head_sha="d" * 40,
    )
    failed_workflow = copy.deepcopy(workflow)
    failed_workflow["canonical_projection"]["conclusion"] = "failure"
    failed_workflow["observation_sha256"] = audit.canonical_json_sha256(
        failed_workflow, exclude_fields={"observation_sha256"}
    )
    with pytest.raises(audit.AuditError, match="workflow canonical projection"):
        audit.validate_github_workflow_observation(
            failed_workflow,
            protocol,
            expected_head_sha="d" * 40,
        )

    changed_retrieval = copy.deepcopy(commit)
    changed_retrieval["retrieved_at"] = "2026-08-04T00:06:00+00:00"
    with pytest.raises(audit.AuditError, match="self-hash"):
        audit.validate_github_observation(
            changed_retrieval,
            protocol,
            observation_kind="commit",
            requested_commit_sha="d" * 40,
        )

    changed_flat_payload = {
        key: value
        for key, value in decisions[0].items()
        if key not in audit.CHAIN_COLUMNS
    }
    changed_flat_payload["receipt_branch_observation"]["retrieved_at"] = (
        "2026-08-04T00:06:00+00:00"
    )
    changed_flat_payload["receipt_branch_observation"]["observation_sha256"] = (
        audit.canonical_json_sha256(
            changed_flat_payload["receipt_branch_observation"],
            exclude_fields={"observation_sha256"},
        )
    )
    with pytest.raises(audit.AuditError, match="flat fields"):
        audit.validate_decision_records(_chain([changed_flat_payload]))

    changed_headers = copy.deepcopy(commit)
    changed_headers["response_headers_sha256"] = "f" * 64
    changed_headers["observation_sha256"] = audit.canonical_json_sha256(
        changed_headers, exclude_fields={"observation_sha256"}
    )
    with pytest.raises(audit.AuditError, match="response-header"):
        audit.validate_github_observation(
            changed_headers,
            protocol,
            observation_kind="commit",
            requested_commit_sha="d" * 40,
        )

    noncanonical_parents = _github_observation(
        observation_kind="commit",
        projected_commit_sha="d" * 40,
        requested_commit_sha="d" * 40,
        committed_at="2026-08-04T00:00:00+00:00",
        retrieved_at="2026-08-04T00:04:00+00:00",
        parent_shas=("a" * 40, "b" * 40),
    )
    noncanonical_parents["canonical_projection"]["parent_shas"] = [
        "b" * 40,
        "a" * 40,
        "a" * 40,
    ]
    noncanonical_parents["observation_sha256"] = audit.canonical_json_sha256(
        noncanonical_parents, exclude_fields={"observation_sha256"}
    )
    with pytest.raises(audit.AuditError, match="canonical projection"):
        audit.validate_github_observation(
            noncanonical_parents,
            protocol,
            observation_kind="commit",
            requested_commit_sha="d" * 40,
        )


def test_checkpoint_workflow_observation_has_finite_terminal_conclusions() -> None:
    protocol, _ = audit.validate_protocol_contract()
    head = "a" * 40
    conclusions = protocol["daily_preopen_checkpoint_contract"][
        "checkpoint_workflow_run_observation_contract"
    ]["terminal_conclusion_values"]
    for index, conclusion in enumerate(conclusions, start=1):
        observation = _github_workflow_observation(
            head_sha=head,
            run_id=50_000 + index,
            updated_at="2026-08-05T08:30:00+09:00",
            retrieved_at="2026-08-05T08:31:00+09:00",
            conclusion=conclusion,
        )
        assert audit.validate_github_workflow_observation(
            observation,
            protocol,
            expected_head_sha=head,
            checkpoint=True,
        )["canonical_projection"]["conclusion"] == conclusion
        if conclusion != "success":
            with pytest.raises(audit.AuditError, match="canonical projection"):
                audit.validate_github_workflow_observation(
                    observation,
                    protocol,
                    expected_head_sha=head,
                )

    unregistered = _github_workflow_observation(
        head_sha=head,
        run_id=60_000,
        updated_at="2026-08-05T08:30:00+09:00",
        retrieved_at="2026-08-05T08:31:00+09:00",
        conclusion="unknown_terminal_state",
    )
    with pytest.raises(audit.AuditError, match="canonical projection"):
        audit.validate_github_workflow_observation(
            unregistered,
            protocol,
            expected_head_sha=head,
            checkpoint=True,
        )


def test_decision_rejects_cutoff_equal_primary_workflow_creation() -> None:
    decisions, _ = _forward_ledgers(
        pd.DatetimeIndex([pd.Timestamp("2026-08-05")])
    )
    late = copy.deepcopy(decisions[0])
    observation = late["checkpoint_workflow_run_observation"]
    assert isinstance(observation, dict)
    projection = observation["canonical_projection"]
    assert isinstance(projection, dict)
    projection.update(
        {
            "created_at": "2026-08-05T08:58:59+09:00",
            "run_started_at": "2026-08-05T08:59:00+09:00",
            "updated_at": "2026-08-05T09:00:00+09:00",
        }
    )
    observation["retrieved_at"] = "2026-08-05T09:01:00+09:00"
    observation["http_date"] = "Wed, 05 Aug 2026 00:00:59 GMT"
    observation["response_body_sha256"] = hashlib.sha256(
        audit.canonical_json_bytes(projection)
    ).hexdigest()
    observation["response_headers_sha256"] = audit.canonical_json_sha256(
        {
            "status": 200,
            "content_type": observation["content_type"],
            "date": observation["http_date"],
            "etag": observation["etag"],
        }
    )
    observation["observation_sha256"] = audit.canonical_json_sha256(
        observation, exclude_fields={"observation_sha256"}
    )
    late["checkpoint_workflow_run_updated_at"] = "2026-08-05T09:00:00+09:00"
    late["checkpoint_workflow_run_observed_at"] = "2026-08-05T09:01:00+09:00"
    late["decision_materialized_at"] = "2026-08-05T09:02:00+09:00"
    late_rows = _chain(
        [
            {
                key: value
                for key, value in late.items()
                if key not in audit.CHAIN_COLUMNS
            }
        ]
    )
    with pytest.raises(audit.AuditError, match="checkpoint evidence/timestamps"):
        audit.validate_decision_records(late_rows)
    with pytest.raises(runner.V18Error, match="checkpoint evidence"):
        runner.validate_decision_records(late_rows)


def test_decision_rejects_nonprimary_branch_tip_and_nonsole_parent() -> None:
    decisions, _ = _forward_ledgers(
        pd.DatetimeIndex([pd.Timestamp("2026-08-05")])
    )

    def chained(row: dict[str, object]) -> list[dict[str, object]]:
        return _chain(
            [
                {
                    key: value
                    for key, value in row.items()
                    if key not in audit.CHAIN_COLUMNS
                }
            ]
        )

    def rehash_observation(observation: dict[str, object]) -> None:
        projection = observation["canonical_projection"]
        assert isinstance(projection, dict)
        observation["response_body_sha256"] = hashlib.sha256(
            audit.canonical_json_bytes(projection)
        ).hexdigest()
        observation["observation_sha256"] = audit.canonical_json_sha256(
            observation, exclude_fields={"observation_sha256"}
        )

    wrong_tip = copy.deepcopy(decisions[0])
    branch_observation = wrong_tip["checkpoint_branch_observation"]
    assert isinstance(branch_observation, dict)
    branch_projection = branch_observation["canonical_projection"]
    assert isinstance(branch_projection, dict)
    alternate_tip = "e" * 40
    branch_projection["commit_sha"] = alternate_tip
    branch_projection["html_url"] = (
        "https://github.com/rokuroku-066/TSE-Session-Ranker/commit/"
        + alternate_tip
    )
    wrong_tip["checkpoint_branch_tip_sha_when_observed"] = alternate_tip
    rehash_observation(branch_observation)
    wrong_tip_rows = chained(wrong_tip)
    with pytest.raises(audit.AuditError, match="branch/sole-parent"):
        audit.validate_decision_records(wrong_tip_rows)
    with pytest.raises(runner.V18Error, match="checkpoint evidence|branch tip"):
        runner.validate_decision_records(wrong_tip_rows)

    no_parent = copy.deepcopy(decisions[0])
    commit_observation = no_parent["checkpoint_commit_observation"]
    assert isinstance(commit_observation, dict)
    commit_projection = commit_observation["canonical_projection"]
    assert isinstance(commit_projection, dict)
    commit_projection["parent_shas"] = []
    rehash_observation(commit_observation)
    no_parent_rows = chained(no_parent)
    with pytest.raises(audit.AuditError, match="branch/sole-parent"):
        audit.validate_decision_records(no_parent_rows)
    with pytest.raises(runner.V18Error, match="checkpoint evidence|sole parent"):
        runner.validate_decision_records(no_parent_rows)


def test_audit_github_refetch_uses_locked_urllib_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock = json.loads(
        (RESEARCH / "model_v18_runtime_lock.json").read_text(encoding="utf-8")
    )
    ca_contract = lock["runtime"]["python"]["ssl"]["ca_trust"]
    ca_path = Path(ca_contract["cafile_path"])
    for name in ca_contract["environment"]:
        monkeypatch.delenv(name, raising=False)
    assert (
        _install_virtual_registered_ca(monkeypatch, tmp_path, audit, ca_contract)
        == ca_path
    )
    response_body = {
        "sha": "d" * 40,
        "html_url": (
            "https://github.com/rokuroku-066/TSE-Session-Ranker/commit/"
            + "d" * 40
        ),
    }
    raw = json.dumps(response_body).encode()
    captured: dict[str, object] = {}

    class Context:
        check_hostname = True
        verify_mode = audit.ssl.CERT_REQUIRED

    context = Context()
    context_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def create_default_context(*args: object, **kwargs: object) -> Context:
        context_calls.append((args, kwargs))
        return context

    class Response:
        status = 200
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Content-Encoding": "identity",
            "Date": "Tue, 04 Aug 2026 00:00:00 GMT",
            "ETag": '"transport-fixture"',
        }

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *args):  # type: ignore[no-untyped-def]
            return False

        def geturl(self) -> str:
            return "https://api.github.com/repos/example/commits/" + "d" * 40

        def read(self) -> bytes:
            return raw

    class Opener:
        def open(self, request):  # type: ignore[no-untyped-def]
            captured["request"] = request
            return Response()

    monkeypatch.setenv("GITHUB_TOKEN", "secret-never-persisted")
    monkeypatch.setattr(audit.ssl, "create_default_context", create_default_context)

    def build_opener(*handlers: object) -> Opener:
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(audit, "build_opener", build_opener)
    observed = audit._github_api_fetch(
        "repos/rokuroku-066/TSE-Session-Ranker/commits/" + "d" * 40
    )
    assert observed == response_body
    request = captured["request"]
    request_headers = {
        key.lower(): value for key, value in request.header_items()
    }
    assert request.full_url.startswith("https://api.github.com/repos/")
    assert request_headers["accept"] == "application/vnd.github+json"
    assert request_headers["accept-encoding"] == "identity"
    assert request_headers["x-github-api-version"] == "2022-11-28"
    assert request_headers["authorization"] == "Bearer secret-never-persisted"
    assert "secret-never-persisted" not in json.dumps(observed)
    assert context_calls == [((), {"cafile": str(ca_path)})]
    assert any(
        isinstance(handler, audit.HTTPSHandler)
        and getattr(handler, "_context", None) is context
        for handler in captured["handlers"]
    )

    for variable in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
        monkeypatch.setenv(variable, str(ca_path))
        captured.pop("request", None)
        with pytest.raises(audit.AuditError, match="environment overrides"):
            audit._github_api_fetch(
                "repos/rokuroku-066/TSE-Session-Ranker/commits/" + "d" * 40
            )
        assert "request" not in captured
        monkeypatch.delenv(variable)

    real_sha256_file = audit.sha256_file

    def changed_ca_sha(path: str | Path) -> str:
        if Path(path) == ca_path:
            return "0" * 64
        return real_sha256_file(path)

    monkeypatch.setattr(audit, "sha256_file", changed_ca_sha)
    captured.pop("request", None)
    with pytest.raises(audit.AuditError, match="CA trust-store bytes changed"):
        audit._github_api_fetch(
            "repos/rokuroku-066/TSE-Session-Ranker/commits/" + "d" * 40
        )
    assert "request" not in captured
    monkeypatch.setattr(audit, "sha256_file", real_sha256_file)

    insecure_context = Context()
    insecure_context.check_hostname = False
    monkeypatch.setattr(
        audit.ssl,
        "create_default_context",
        lambda *args, **kwargs: insecure_context,
    )
    captured.pop("request", None)
    with pytest.raises(audit.AuditError, match="TLS verification policy"):
        audit._github_api_fetch(
            "repos/rokuroku-066/TSE-Session-Ranker/commits/" + "d" * 40
        )
    assert "request" not in captured
    with pytest.raises(audit.AuditError, match="unsafe"):
        audit._github_api_fetch("https://attacker.invalid/repos/x")


def test_runner_github_transport_requires_exact_ca_before_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock = json.loads(
        (RESEARCH / "model_v18_runtime_lock.json").read_text(encoding="utf-8")
    )
    ca_contract = lock["runtime"]["python"]["ssl"]["ca_trust"]
    ca_path = Path(ca_contract["cafile_path"])
    for name in ca_contract["environment"]:
        monkeypatch.delenv(name, raising=False)
    assert (
        _install_virtual_registered_ca(monkeypatch, tmp_path, runner, ca_contract)
        == ca_path
    )
    monkeypatch.setattr(runner, "_STRICT_RUNTIME_ACTIVE", True)
    monkeypatch.setattr(runner, "_LOCKED_TLS_CAFILE", ca_path)

    response_body = {"sha": "d" * 40}
    raw = json.dumps(response_body).encode()
    captured: dict[str, object] = {}

    class Context:
        check_hostname = True
        verify_mode = runner.ssl.CERT_REQUIRED

    context = Context()
    context_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def create_default_context(*args: object, **kwargs: object) -> Context:
        context_calls.append((args, kwargs))
        return context

    class Response:
        status = 200
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Content-Encoding": "identity",
            "Date": "Tue, 04 Aug 2026 00:00:00 GMT",
            "ETag": '"runner-ca-fixture"',
        }

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *args):  # type: ignore[no-untyped-def]
            return False

        def geturl(self) -> str:
            return "https://api.github.com/repos/example/commits/" + "d" * 40

        def read(self) -> bytes:
            return raw

    class Opener:
        def open(self, request, timeout):  # type: ignore[no-untyped-def]
            captured["request"] = request
            captured["timeout"] = timeout
            return Response()

    monkeypatch.setattr(runner.ssl, "create_default_context", create_default_context)

    def build_opener(*handlers: object) -> Opener:
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(runner.urllib.request, "build_opener", build_opener)
    assert runner._github_api("/repos/example/commits/" + "d" * 40) == response_body
    assert context_calls == [((), {"cafile": str(ca_path)})]
    assert captured["timeout"] == 30.0
    assert any(
        isinstance(handler, runner.urllib.request.HTTPSHandler)
        and getattr(handler, "_context", None) is context
        for handler in captured["handlers"]
    )

    for variable in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
        monkeypatch.setenv(variable, str(ca_path))
        captured.pop("request", None)
        with pytest.raises(runner.V18Error, match="environment overrides"):
            runner._github_api("/repos/example/commits/" + "d" * 40)
        assert "request" not in captured
        monkeypatch.delenv(variable)

    real_sha256_file = runner.sha256_file

    def changed_ca_sha(path: str | Path) -> str:
        if Path(path) == ca_path:
            return "0" * 64
        return real_sha256_file(path)

    monkeypatch.setattr(runner, "sha256_file", changed_ca_sha)
    captured.pop("request", None)
    with pytest.raises(runner.V18Error, match="CA trust-store bytes changed"):
        runner._github_api("/repos/example/commits/" + "d" * 40)
    assert "request" not in captured
    monkeypatch.setattr(runner, "sha256_file", real_sha256_file)

    insecure_context = Context()
    insecure_context.verify_mode = runner.ssl.CERT_NONE
    monkeypatch.setattr(
        runner.ssl,
        "create_default_context",
        lambda *args, **kwargs: insecure_context,
    )
    captured.pop("request", None)
    with pytest.raises(runner.V18Error, match="TLS verification policy"):
        runner._github_api("/repos/example/commits/" + "d" * 40)
    assert "request" not in captured


def test_activation_payload_receipt_and_first_terminal_bindings_use_real_git(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, _ = audit.validate_protocol_contract()
    with pytest.raises(
        runner.V18Error,
        match="Git evidence|commit|Git executable is missing or symlinked",
    ):
        runner.create_activation_payload(
            preregistration_commit_sha="1" * 40,
        )

    git_root = tmp_path / "activation-git"
    git_root.mkdir()

    def git(*arguments: str, committed_at: str | None = None) -> str:
        environment = os.environ.copy()
        if committed_at is not None:
            environment["GIT_AUTHOR_DATE"] = committed_at
            environment["GIT_COMMITTER_DATE"] = committed_at
        completed = subprocess.run(
            ["git", "-C", str(git_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return completed.stdout.strip()

    branch = protocol["branch"]
    git("init", "-b", branch)
    git("config", "user.name", "v18-test")
    git("config", "user.email", "v18-test@example.invalid")
    preregistration_paths = set(
        protocol["activation"]["preregistration_commit"]["required_paths"]
    )
    preregistration_paths.update(
        relative
        for bindings in (runner.V16_BINDINGS, runner.V17_BINDINGS)
        for relative, _ in bindings.values()
    )
    runtime_lock = json.loads(
        (RESEARCH / "model_v18_runtime_lock.json").read_text(encoding="utf-8")
    )
    preregistration_paths.update(
        record["path"] for record in runtime_lock["project_files"]
    )
    for relative in preregistration_paths:
        destination = git_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    git("add", ".")
    git(
        "commit",
        "-m",
        "v1.8 preregistration",
        committed_at="2026-08-04T10:00:00+09:00",
    )
    prereg_sha = git("rev-parse", "HEAD")
    base_url = "https://github.com/rokuroku-066/TSE-Session-Ranker/commit/"

    monkeypatch.setattr(runner, "ROOT", git_root)
    monkeypatch.setattr(
        runner,
        "ACTIVATION_PAYLOAD",
        git_root / "research/model_v18_shoulder_state_activation_payload.json",
    )
    monkeypatch.setattr(
        runner,
        "ACTIVATION_RECEIPT",
        git_root / "research/model_v18_shoulder_state_activation_receipt.json",
    )

    clock = {"now": datetime.fromisoformat("2026-08-04T10:10:00+09:00")}

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def]
            fixed = cls.fromisoformat(clock["now"].isoformat())
            return fixed if tz is None else fixed.astimezone(tz)

    monkeypatch.setattr(runner, "datetime", FixedDatetime)

    encoded_branch = quote(str(branch), safe="")
    workflow_run_ids: dict[str, int] = {}

    def workflow_body(commit: str) -> dict[str, object]:
        run_id = workflow_run_ids.setdefault(commit, 101 + len(workflow_run_ids))
        committed = pd.Timestamp(git("show", "-s", "--format=%cI", commit))
        created = committed + pd.Timedelta(minutes=1)
        started = committed + pd.Timedelta(minutes=2)
        updated = committed + pd.Timedelta(minutes=3)
        return {
            "id": run_id,
            "workflow_id": 18,
            "name": "tests",
            "path": ".github/workflows/tests.yml",
            "event": "pull_request",
            "head_sha": commit,
            "run_attempt": 1,
            "status": "completed",
            "conclusion": "success",
            "created_at": created.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_started_at": started.tz_convert("UTC").strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "updated_at": updated.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "html_url": (
                "https://github.com/rokuroku-066/TSE-Session-Ranker/actions/runs/"
                f"{run_id}"
            ),
        }

    def fake_github_api(
        endpoint: str,
        *,
        return_observation: bool = False,
    ) -> object:
        compare_prefix = "/repos/rokuroku-066/TSE-Session-Ranker/compare/"
        commits_prefix = "/repos/rokuroku-066/TSE-Session-Ranker/commits/"
        workflow_list_prefix = (
            "/repos/rokuroku-066/TSE-Session-Ranker/actions/runs?"
        )
        workflow_run_prefix = (
            "/repos/rokuroku-066/TSE-Session-Ranker/actions/runs/"
        )
        if endpoint.startswith(compare_prefix):
            ancestor, descendant = endpoint.removeprefix(compare_prefix).split(
                "...", maxsplit=1
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(git_root),
                    "merge-base",
                    "--is-ancestor",
                    ancestor,
                    descendant,
                ],
                check=True,
                capture_output=True,
            )
            return {
                "status": "ahead",
                "base_commit": {"sha": ancestor},
                "merge_base_commit": {"sha": ancestor},
            }
        if endpoint.startswith(workflow_list_prefix):
            query = parse_qs(urlparse(endpoint).query)
            commit = query["head_sha"][0]
            body: dict[str, object] = {
                "total_count": 1,
                "workflow_runs": [workflow_body(commit)],
            }
        elif endpoint.startswith(workflow_run_prefix):
            run_id = int(endpoint.removeprefix(workflow_run_prefix))
            commit = next(
                sha
                for sha, registered_run_id in workflow_run_ids.items()
                if registered_run_id == run_id
            )
            body = workflow_body(commit)
        else:
            assert endpoint.startswith(commits_prefix), endpoint
            reference = endpoint.removeprefix(commits_prefix)
            commit = git("rev-parse", "HEAD") if reference == encoded_branch else reference
            committed = pd.Timestamp(git("show", "-s", "--format=%cI", commit))
            committed_utc = committed.tz_convert("UTC").strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            body = {
                "sha": commit,
                "html_url": base_url + commit,
                "commit": {"committer": {"date": committed_utc}},
                "parents": [
                    {"sha": parent}
                    for parent in git("show", "-s", "--format=%P", commit).split()
                ],
            }
        if not return_observation:
            return body
        retrieved = FixedDatetime.now(timezone.utc)
        response_date = format_datetime(
            (retrieved.astimezone(timezone.utc) - timedelta(seconds=1)).replace(
                microsecond=0
            ),
            usegmt=True,
        )
        content_type = "application/json; charset=utf-8"
        etag = f'"fixture-{hashlib.sha256(endpoint.encode()).hexdigest()[:16]}"'
        raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        transport = {
            "http_status": 200,
            "content_type": content_type,
            "http_date": response_date,
            "etag": etag,
            "response_body_sha256": hashlib.sha256(raw).hexdigest(),
            "response_headers_sha256": runner.canonical_json_sha256(
                {
                    "status": 200,
                    "content_type": content_type,
                    "date": response_date,
                    "etag": etag,
                }
            ),
            "retrieved_at": retrieved,
        }
        return body, transport

    monkeypatch.setattr(runner, "_github_api", fake_github_api)
    payload = runner.create_activation_payload(
        preregistration_commit_sha=prereg_sha,
        output=runner.ACTIVATION_PAYLOAD,
    )
    runner_payload, runner_payload_sha = runner.validate_activation_payload(payload)
    assert runner_payload == payload
    assert audit.validate_activation_payload(payload, protocol) == runner_payload_sha

    git("add", "research/model_v18_shoulder_state_activation_payload.json")
    git(
        "commit",
        "-m",
        "v1.8 activation payload",
        committed_at="2026-08-04T11:00:00+09:00",
    )
    payload_commit_sha = git("rev-parse", "HEAD")
    clock["now"] = datetime.fromisoformat("2026-08-04T11:10:00+09:00")
    payload = runner.read_json(runner.ACTIVATION_PAYLOAD)
    receipt = runner.create_activation_receipt(
        payload,
        payload_commit_sha=payload_commit_sha,
        output=runner.ACTIVATION_RECEIPT,
    )
    runner_receipt, runner_receipt_sha = runner.validate_activation_receipt(
        receipt, payload
    )
    assert runner_receipt == receipt
    assert audit.validate_activation_receipt(
        receipt, payload, protocol
    ) == runner_receipt_sha
    wrong_payload_bytes = copy.deepcopy(receipt)
    wrong_payload_bytes["payload_file_sha256"] = "9" * 64
    wrong_payload_bytes["receipt_sha256"] = audit.canonical_json_sha256(
        wrong_payload_bytes, exclude_fields={"receipt_sha256"}
    )
    with pytest.raises(audit.AuditError, match="exact payload file bytes"):
        audit.validate_activation_receipt(wrong_payload_bytes, payload, protocol)

    git("add", "research/model_v18_shoulder_state_activation_receipt.json")
    git(
        "commit",
        "-m",
        "v1.8 activation receipt",
        committed_at="2026-08-04T12:00:00+09:00",
    )
    receipt_commit_sha = git("rev-parse", "HEAD")
    clock["now"] = datetime.fromisoformat("2026-08-04T12:10:00+09:00")
    activation = runner.activate(
        payload,
        receipt,
        activation_receipt_commit_sha=receipt_commit_sha,
        calendar=audit.load_registered_calendar(),
    )
    assert activation["first_counted_session"] == "2026-08-05"
    assert activation["terminal_session"] == "2027-02-26"
    assert activation["terminal_scheduled_sessions"] == 136
    assert activation["represented_calendar_months"] == 7
    assert activation["production_model_changed"] is False
    assert activation["orders_allowed"] is False

    decisions, _ = _forward_ledgers(audit.load_registered_calendar()[:1])
    decision_payload = {
        key: value
        for key, value in decisions[0].items()
        if key not in audit.CHAIN_COLUMNS
    }
    decision_payload.update(
        {
            "activation_payload_sha256": runner_payload_sha,
            "activation_receipt_sha256": runner_receipt_sha,
            "activation_receipt_commit_sha": activation[
                "activation_receipt_commit_sha"
            ],
            "activation_receipt_commit_url": activation[
                "activation_receipt_commit_url"
            ],
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
            "receipt_commit_observation": activation[
                "receipt_commit_observation"
            ],
            "receipt_branch_observation": activation[
                "receipt_branch_observation"
            ],
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
        }
    )
    linked_decisions = _chain([decision_payload])

    def github_fetch(endpoint: str) -> dict[str, object]:
        value = fake_github_api("/" + endpoint.lstrip("/"))
        assert isinstance(value, dict)
        return value

    evidence = audit.validate_activation_git_history(
        payload,
        receipt,
        linked_decisions,
        protocol,
        repository_root=git_root,
        payload_file_path=runner.ACTIVATION_PAYLOAD,
        receipt_file_path=runner.ACTIVATION_RECEIPT,
        github_fetcher=github_fetch,
    )
    assert evidence["preregistration_commit_sha"] == prereg_sha
    assert evidence["payload_commit_sha"] == payload_commit_sha
    assert evidence["receipt_commit_sha"] == receipt_commit_sha
    assert evidence["github_branch_tip_sha"] == receipt_commit_sha

    def mutated_github_fetch(endpoint: str) -> dict[str, object]:
        value = copy.deepcopy(github_fetch(endpoint))
        if endpoint.endswith(prereg_sha):
            value["parents"] = [{"sha": "0" * 40}]
        return value

    with pytest.raises(audit.AuditError, match="GitHub commit evidence"):
        audit.validate_activation_git_history(
            payload,
            receipt,
            linked_decisions,
            protocol,
            repository_root=git_root,
            payload_file_path=runner.ACTIVATION_PAYLOAD,
            receipt_file_path=runner.ACTIVATION_RECEIPT,
            github_fetcher=mutated_github_fetch,
        )

    def mutated_workflow_fetch(endpoint: str) -> dict[str, object]:
        value = copy.deepcopy(github_fetch(endpoint))
        if "/actions/runs/" in endpoint and "?" not in endpoint:
            value["conclusion"] = "failure"
        return value

    with pytest.raises(audit.AuditError, match="workflow evidence changed"):
        audit.validate_activation_git_history(
            payload,
            receipt,
            linked_decisions,
            protocol,
            repository_root=git_root,
            payload_file_path=runner.ACTIVATION_PAYLOAD,
            receipt_file_path=runner.ACTIVATION_RECEIPT,
            github_fetcher=mutated_workflow_fetch,
        )

    fake_payload = copy.deepcopy(payload)
    fake_payload["preregistration_commit_sha"] = "1" * 40
    fake_payload["preregistration_commit_url"] = base_url + "1" * 40
    with pytest.raises(audit.AuditError, match="does not exist"):
        audit.validate_activation_git_history(
            fake_payload,
            receipt,
            linked_decisions,
            protocol,
            repository_root=git_root,
            payload_file_path=runner.ACTIVATION_PAYLOAD,
            receipt_file_path=runner.ACTIVATION_RECEIPT,
            github_fetcher=github_fetch,
        )


def test_decision_and_outcome_chains_fail_closed_on_tamper_gap_or_duplicate() -> None:
    calendar = audit.load_registered_calendar()[:3]
    decisions, outcomes = _forward_ledgers(calendar)
    assert len(audit.validate_decision_records(decisions)) == 3
    assert len(audit.validate_outcome_records(outcomes, decisions)) == 3

    tampered = copy.deepcopy(decisions)
    tampered[1]["c02_rank2_score"] = 999.0
    with pytest.raises(audit.AuditError, match="SHA-256"):
        audit.validate_decision_records(tampered)

    gap = copy.deepcopy(decisions)
    gap[1]["sequence_number"] = 7
    with pytest.raises(audit.AuditError, match="sequence"):
        audit.validate_decision_records(gap)

    duplicate = copy.deepcopy(decisions)
    duplicate[1]["session_date"] = duplicate[0]["session_date"]
    duplicate = _chain(
        [
            {key: value for key, value in row.items() if key not in audit.CHAIN_COLUMNS}
            for row in duplicate
        ]
    )
    with pytest.raises(audit.AuditError, match="duplicate"):
        audit.validate_decision_records(duplicate)


def test_completed_month_chain_uses_month_key_and_recomputes_pair_median() -> None:
    calendar = audit.load_registered_calendar()
    sessions = calendar[calendar.to_period("M") == pd.Period("2026-08")]
    decisions, outcomes = _forward_ledgers(sessions)
    records = runner.build_completed_month_record(
        decisions,
        outcomes,
        completed_month="2026-08",
        created_at="2026-09-01T00:00:01+09:00",
    )
    protocol, _ = audit.validate_protocol_contract()
    recomputed = audit.recompute_completed_month_records(
        records, decisions, outcomes, protocol
    )
    assert recomputed[0]["sequence_number"] == 0
    assert recomputed[0]["completed_month"] == "2026-08"
    assert recomputed[0]["counted_scheduled_sessions"] == len(sessions)
    assert recomputed[0]["complete_pair_days"] == len(sessions)
    assert recomputed[0]["monthly_median_rank1_minus_rank2_pct"] == pytest.approx(
        -0.8, abs=1e-15
    )

    tiny_month_drift = copy.deepcopy(records[0])
    tiny_month_drift["monthly_median_rank1_minus_rank2_pct"] += 5e-13
    tiny_month_drift["record_sha256"] = audit.canonical_json_sha256(
        tiny_month_drift, exclude_fields={"record_sha256"}
    )
    with pytest.raises(audit.AuditError, match="monthly_median.*does not recompute"):
        audit.recompute_completed_month_records(
            [tiny_month_drift], decisions, outcomes, protocol
        )

    tiny_zero_drift = copy.deepcopy(outcomes)
    tiny_zero_drift[0]["c00_top1_net20_return_pct"] = 5e-13
    tiny_zero_drift = _chain(
        [
            {key: value for key, value in row.items() if key not in audit.CHAIN_COLUMNS}
            for row in tiny_zero_drift
        ]
    )
    with pytest.raises(audit.AuditError, match="c00_top1_net20.*does not recompute"):
        audit.validate_outcome_records(tiny_zero_drift, decisions)

    gap = copy.deepcopy(records[0])
    gap["sequence_number"] = 1
    gap["record_sha256"] = audit.canonical_json_sha256(
        gap, exclude_fields={"record_sha256"}
    )
    with pytest.raises(audit.AuditError, match="sequence"):
        audit.validate_completed_month_records([gap], protocol)


def test_state_schedule_recomputes_seed_rolloff_and_exact_forward_month_set() -> None:
    seed = [
        {
            "completed_month": "2026-05",
            "complete_pair_days": 17,
            "monthly_median_rank1_minus_rank2_pct": 0.4532617412224217,
        },
        {
            "completed_month": "2026-06",
            "complete_pair_days": 19,
            "monthly_median_rank1_minus_rank2_pct": 0.5353494177210093,
        },
        {
            "completed_month": "2026-07",
            "complete_pair_days": 18,
            "monthly_median_rank1_minus_rank2_pct": 0.09214571919513584,
        },
    ]
    payload_sha, receipt_sha = "b" * 64, "c" * 64
    completed = [
        {
            "completed_month": "2026-08",
            "created_at": "2026-09-01T00:00:01+09:00",
            "complete_pair_days": 12,
            "monthly_median_rank1_minus_rank2_pct": -2.0,
            "activation_payload_sha256": payload_sha,
            "activation_receipt_sha256": receipt_sha,
        },
        {
            "completed_month": "2026-09",
            "created_at": "2026-10-01T00:00:01+09:00",
            "complete_pair_days": 11,
            "monthly_median_rank1_minus_rank2_pct": -3.0,
            "activation_payload_sha256": payload_sha,
            "activation_receipt_sha256": receipt_sha,
        },
        {
            "completed_month": "2026-10",
            "created_at": "2026-11-02T00:00:01+09:00",
            "complete_pair_days": 10,
            "monthly_median_rank1_minus_rank2_pct": -4.0,
            "activation_payload_sha256": payload_sha,
            "activation_receipt_sha256": receipt_sha,
        },
    ]
    specifications = {
        "2026-08": ([17, 19, 18], [0.4532617412224217, 0.5353494177210093, 0.09214571919513584], 0.4532617412224217, 1),
        "2026-09": ([19, 18, 12], [0.5353494177210093, 0.09214571919513584, -2.0], 0.09214571919513584, 1),
        "2026-10": ([18, 12, 11], [0.09214571919513584, -2.0, -3.0], -2.0, 2),
    }
    session_by_month = {
        "2026-08": "2026-08-05",
        "2026-09": "2026-09-01",
        "2026-10": "2026-10-01",
    }
    state_manifests: dict[str, dict[str, object]] = {}
    decisions: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    for month_text, (counts, medians, state_value, rank) in specifications.items():
        month = pd.Period(month_text, freq="M")
        manifest: dict[str, object] = {
            "target_month": month_text,
            "created_at": f"{session_by_month[month_text]}T08:00:00+09:00",
            "three_prior_calendar_months": [
                str(month - 3),
                str(month - 2),
                str(month - 1),
            ],
            "three_complete_pair_day_counts": counts,
            "three_month_medians_pct": medians,
            "state_available": True,
            "state_value_pct": state_value,
            "selected_source_rank": rank,
            "activation_payload_sha256": payload_sha,
            "activation_receipt_sha256": receipt_sha,
            "state_manifest_sha256": (month.month.__str__() * 64)[:64],
        }
        state_manifests[month_text] = manifest
        decisions.append(
            {
                "session_date": session_by_month[month_text],
                "state_manifest_sha256": manifest["state_manifest_sha256"],
                "three_prior_calendar_months": manifest[
                    "three_prior_calendar_months"
                ],
                "three_complete_pair_day_counts": counts,
                "three_month_medians_pct": medians,
                "state_available": True,
                "state_value_pct": state_value,
                "selected_source_rank": rank,
            }
        )
        outcomes.append(
            {
                "session_date": session_by_month[month_text],
                "outcome_received_at": (
                    f"{session_by_month[month_text]}T16:00:00+09:00"
                ),
                "computed_at": (
                    f"{session_by_month[month_text]}T16:10:00+09:00"
                ),
            }
        )
    schedule = audit.recompute_state_schedule(
        seed_records=seed,
        completed_months=completed,
        state_manifests=state_manifests,
        decisions=decisions,
        outcomes=outcomes,
        first_counted_session_value="2026-08-05",
        activation_ready_at="2026-08-04T13:01:00+09:00",
        activation_payload_sha256=payload_sha,
        activation_receipt_sha256=receipt_sha,
    )
    assert [item["selected_source_rank"] for item in schedule] == [1, 1, 2]
    assert schedule[-1]["three_prior_calendar_months"] == [
        "2026-07",
        "2026-08",
        "2026-09",
    ]
    with pytest.raises(audit.AuditError, match="exact forward set"):
        audit.recompute_state_schedule(
            seed_records=seed,
            completed_months=completed[:-1],
            state_manifests=state_manifests,
            decisions=decisions,
            outcomes=outcomes,
            first_counted_session_value="2026-08-05",
            activation_ready_at="2026-08-04T13:01:00+09:00",
            activation_payload_sha256=payload_sha,
            activation_receipt_sha256=receipt_sha,
        )


def test_partial_initial_month_state_cutoff_uses_first_counted_session() -> None:
    protocol, _ = audit.validate_protocol_contract()
    value: dict[str, object] = {
        "schema_version": 1,
        "target_month": "2026-08",
        "created_at": "2026-08-06T08:00:00+09:00",
        "three_prior_calendar_months": ["2026-05", "2026-06", "2026-07"],
        "three_complete_pair_day_counts": [17, 19, 18],
        "three_month_medians_pct": [
            0.4532617412224217,
            0.5353494177210093,
            0.09214571919513584,
        ],
        "state_available": True,
        "state_value_pct": 0.4532617412224217,
        "selected_source_rank": 1,
        "c00_fold_manifest_sha256": None,
        "fold_model_bundle_file_sha256": None,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "activation_payload_sha256": "b" * 64,
        "activation_receipt_sha256": "c" * 64,
    }
    value["state_manifest_sha256"] = audit.canonical_json_sha256(value)
    assert audit.validate_state_manifest(
        value,
        protocol,
        first_counted_session="2026-08-06",
    )["target_month"] == "2026-08"
    with pytest.raises(audit.AuditError, match="cutoff"):
        audit.validate_state_manifest(value, protocol)


def test_terminal_requires_month_end_after_120_sessions_and_six_months(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calendar = audit.load_registered_calendar()
    terminal = audit.deterministic_terminal_session(calendar[0], calendar)
    terminal_sessions = calendar[(calendar >= calendar[0]) & (calendar <= terminal)]
    assert terminal_sessions[117].to_period("M") == pd.Period("2027-01")
    assert terminal_sessions[119] == pd.Timestamp("2027-02-02")

    decisions, outcomes = _forward_ledgers(terminal_sessions[:120])
    awaiting = audit.evaluate_candidate(
        decisions, outcomes, calendar=calendar
    )
    assert awaiting["status"] == "awaiting_terminal_evaluation"
    assert awaiting["gate_evaluated"] is False
    assert awaiting["deterministic_terminal_session"] == "2027-02-26"

    decisions, outcomes = _forward_ledgers(terminal_sessions)
    completed = _completed_month_ledger(decisions, outcomes)
    with pytest.raises(audit.AuditError, match="completed-month"):
        audit.evaluate_candidate(decisions, outcomes, calendar=calendar)
    with pytest.raises(runner.V18Error, match="represented month"):
        runner.evaluate(decisions, outcomes, [], calendar=calendar)
    with pytest.raises(audit.AuditError, match="represented month"):
        audit.evaluate_candidate(
            decisions,
            outcomes,
            calendar=calendar,
            completed_months=completed[:-1],
        )
    premature_payloads = [
        {key: value for key, value in row.items() if key not in audit.CHAIN_COLUMNS}
        for row in completed
    ]
    premature_payloads[-1]["created_at"] = (
        f"{terminal.date()}T15:00:00+09:00"
    )
    premature = _chain(premature_payloads)
    with pytest.raises(audit.AuditError, match="month-end|outcomes"):
        audit.evaluate_candidate(
            decisions,
            outcomes,
            calendar=calendar,
            completed_months=premature,
        )
    with pytest.raises(runner.V18Error, match="final outcome|cleanly recompute"):
        runner.evaluate(
            decisions,
            outcomes,
            premature,
            calendar=calendar,
        )
    result = audit.evaluate_candidate(
        decisions,
        outcomes,
        calendar=calendar,
        completed_months=completed,
    )
    checkpoint_bindings = {
        "checkpoint_proposal_set_sha256": "3" * 64,
        "checkpoint_core_object_set_sha256": "4" * 64,
        "checkpoint_evidence_set_sha256": "5" * 64,
    }
    predictor_bindings = {
        field: (
            1 if field == "predictor_unique_raw_object_count" else "6" * 64
        )
        for field in audit.read_json(audit.DEFAULT_PROTOCOL)["result_contract"][
            "required_input_fields"
        ]
        if field not in checkpoint_bindings
    }
    monkeypatch.setattr(
        runner,
        "validate_checkpoint_evidence",
        lambda *args, **kwargs: checkpoint_bindings,
    )
    monkeypatch.setattr(runner, "validate_outcome_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "validate_predictor_evidence",
        lambda *args, **kwargs: predictor_bindings,
    )
    runtime = runner.evaluate(
        decisions,
        outcomes,
        completed_months=completed,
        calendar=calendar,
        source_manifest_directory=tmp_path,
        predictor_raw_store_root=tmp_path,
        scores=pd.DataFrame(columns=runner.SCORE_FIELDS),
        outcome_manifest_directory=tmp_path,
        outcome_raw_store_root=tmp_path,
        checkpoint_core_store_root=tmp_path,
    )
    assert runtime.pop("input_bindings") == {
        **checkpoint_bindings,
        **predictor_bindings,
    }
    assert result == runtime
    assert result["scheduled_sessions"] == 136
    assert result["represented_calendar_months"] == 7
    assert result["gate_evaluated"] is True
    assert result["gate_passed"] is True
    assert result["winner"] == EXPECTED_CANDIDATE
    assert result["status"] == "forward_passed_one_v19_research_nominee"
    assert result["cost_metrics"] == pytest.approx(
        {"20": 0.8, "40": 0.6, "60": 0.4}, abs=1e-15
    )


def test_independent_audit_checks_terminal_month_before_unblinding_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    files = {
        "DEFAULT_PROTOCOL": tmp_path / "protocol.json",
        "DEFAULT_RUNTIME_LOCK": tmp_path / "runtime.json",
        "DEFAULT_ACTIVATION_PAYLOAD": tmp_path / "payload.json",
        "DEFAULT_ACTIVATION_RECEIPT": tmp_path / "receipt.json",
        "DEFAULT_DECISIONS": tmp_path / "decisions.jsonl",
        "DEFAULT_OUTCOMES": tmp_path / "outcomes.jsonl",
        "DEFAULT_MONTHS": tmp_path / "months.jsonl",
        "DEFAULT_SCORES": tmp_path / "scores.csv",
        "DEFAULT_PICKS": tmp_path / "picks.csv",
        "DEFAULT_CALENDAR": tmp_path / "calendar.csv",
        "DEFAULT_RESULT": tmp_path / "result.json",
        "DEFAULT_RUNNER": tmp_path / "runner.py",
    }
    directories = {
        "DEFAULT_STATE_MANIFESTS": tmp_path / "states",
        "DEFAULT_FOLD_MANIFESTS": tmp_path / "folds",
        "DEFAULT_FOLD_MODELS": tmp_path / "models",
        "DEFAULT_SOURCE_MANIFESTS": tmp_path / "sources",
        "DEFAULT_OUTCOME_MANIFESTS": tmp_path / "outcome-manifests",
    }
    for name, path in {**files, **directories}.items():
        monkeypatch.setattr(audit, name, path)
    for path in directories.values():
        path.mkdir()
    for name in (
        "DEFAULT_PROTOCOL",
        "DEFAULT_RUNTIME_LOCK",
        "DEFAULT_ACTIVATION_PAYLOAD",
        "DEFAULT_ACTIVATION_RECEIPT",
    ):
        files[name].write_text("{}\n", encoding="utf-8")
    files["DEFAULT_RESULT"].write_text(
        '{\n  "status": "forward_rejected_candidate"\n}\n', encoding="utf-8"
    )
    for name in ("DEFAULT_DECISIONS", "DEFAULT_OUTCOMES", "DEFAULT_MONTHS"):
        files[name].write_text("{}\n", encoding="utf-8")
    for name in ("DEFAULT_SCORES", "DEFAULT_PICKS", "DEFAULT_CALENDAR"):
        files[name].write_text("header\n", encoding="utf-8")
    files["DEFAULT_RUNNER"].write_text("# runner fixture\n", encoding="utf-8")

    protocol = {"periods": {"not_before_session": "2026-08-05"}}
    decision = {
        "session_date": "2026-08-05",
        "activation_receipt_workflow_run_updated_at": "2026-08-04T23:00:00+09:00",
        "activation_receipt_workflow_run_observed_at": "2026-08-05T08:00:00+09:00",
    }
    calendar = pd.DatetimeIndex([pd.Timestamp("2026-08-05")])
    monkeypatch.setattr(
        audit, "validate_protocol_contract", lambda path: (protocol, "a" * 64)
    )
    monkeypatch.setattr(
        audit,
        "validate_runtime_lock",
        lambda *args, **kwargs: ({}, "b" * 64),
    )
    monkeypatch.setattr(audit, "validate_activation_payload", lambda *args: "c" * 64)
    monkeypatch.setattr(
        audit, "validate_activation_receipt", lambda *args, **kwargs: "d" * 64
    )
    monkeypatch.setattr(audit, "validate_decision_records", lambda rows: [decision])
    monkeypatch.setattr(audit, "load_registered_calendar", lambda path: calendar)
    monkeypatch.setattr(audit, "first_counted_session", lambda **kwargs: calendar[0])
    monkeypatch.setattr(
        audit, "deterministic_terminal_session", lambda first, calendar: calendar[0]
    )

    calls: list[str] = []
    outcome_calls = {"stable_read": 0, "parse": 0, "validate": 0}
    real_stable_read = audit._stable_plain_file_bytes
    real_parse_jsonl = audit._parse_jsonl_bytes
    real_parse_object = audit._parse_json_object_bytes

    def guarded_stable_read(path, *, label):  # type: ignore[no-untyped-def]
        if Path(path) == files["DEFAULT_OUTCOMES"]:
            outcome_calls["stable_read"] += 1
            pytest.fail("outcome ledger opened before completed-month preflight")
        return real_stable_read(path, label=label)

    def guarded_parse_jsonl(payload, *, label):  # type: ignore[no-untyped-def]
        if label == "canonical outcome ledger":
            outcome_calls["parse"] += 1
            pytest.fail("outcome ledger parsed before completed-month preflight")
        return real_parse_jsonl(payload, label=label)

    def guarded_validate_outcomes(*args, **kwargs):  # type: ignore[no-untyped-def]
        outcome_calls["validate"] += 1
        pytest.fail("outcomes validated before completed-month preflight")

    monkeypatch.setattr(audit, "_stable_plain_file_bytes", guarded_stable_read)
    monkeypatch.setattr(audit, "_parse_jsonl_bytes", guarded_parse_jsonl)
    monkeypatch.setattr(audit, "validate_outcome_records", guarded_validate_outcomes)

    def guarded_parse_object(payload, *, label):  # type: ignore[no-untyped-def]
        if label == "canonical terminal result":
            pytest.fail("normal result parsed before terminal month readiness")
        return real_parse_object(payload, label=label)

    monkeypatch.setattr(audit, "_parse_json_object_bytes", guarded_parse_object)

    def terminal_preflight_failure(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("terminal_month_preflight")
        raise audit.AuditError("terminal month preflight sentinel")

    monkeypatch.setattr(
        audit,
        "preflight_terminal_completed_month_coverage",
        terminal_preflight_failure,
    )
    for helper in (
        "validate_activation_git_history",
        "_outcome_manifest_map",
        "recompute_picks_csv_bytes",
        "_evaluation_frame",
        "evaluate_candidate",
        "validate_predictor_evidence",
    ):
        monkeypatch.setattr(
            audit,
            helper,
            lambda *args, _helper=helper, **kwargs: pytest.fail(
                f"{_helper} ran before terminal month readiness"
            ),
        )

    with pytest.raises(audit.AuditError, match="terminal month preflight sentinel"):
        audit.audit(
            protocol_path=files["DEFAULT_PROTOCOL"],
            activation_payload_path=files["DEFAULT_ACTIVATION_PAYLOAD"],
            activation_receipt_path=files["DEFAULT_ACTIVATION_RECEIPT"],
            decisions_path=files["DEFAULT_DECISIONS"],
            outcomes_path=files["DEFAULT_OUTCOMES"],
            months_path=files["DEFAULT_MONTHS"],
            state_manifest_directory=directories["DEFAULT_STATE_MANIFESTS"],
            fold_manifest_directory=directories["DEFAULT_FOLD_MANIFESTS"],
            fold_model_directory=directories["DEFAULT_FOLD_MODELS"],
            source_manifest_directory=directories["DEFAULT_SOURCE_MANIFESTS"],
            outcome_manifest_directory=directories["DEFAULT_OUTCOME_MANIFESTS"],
            scores_path=files["DEFAULT_SCORES"],
            picks_path=files["DEFAULT_PICKS"],
            result_path=files["DEFAULT_RESULT"],
            runner_path=files["DEFAULT_RUNNER"],
            calendar_path=files["DEFAULT_CALENDAR"],
        )
    assert calls == ["terminal_month_preflight"]
    assert outcome_calls == {"stable_read": 0, "parse": 0, "validate": 0}


def test_evaluate_candidate_checks_terminal_month_before_performance_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calendar = audit.load_registered_calendar()
    terminal = audit.deterministic_terminal_session(calendar[0], calendar)
    sessions = calendar[(calendar >= calendar[0]) & (calendar <= terminal)]
    decisions, outcomes = _forward_ledgers(sessions)
    completed = _completed_month_ledger(decisions, outcomes)
    frame_calls: list[bool] = []

    def forbidden_frame(*args, **kwargs):  # type: ignore[no-untyped-def]
        frame_calls.append(True)
        pytest.fail("performance frame ran before completed-month coverage")

    monkeypatch.setattr(audit, "_evaluation_frame", forbidden_frame)
    with pytest.raises(audit.AuditError, match="represented month"):
        audit.evaluate_candidate(
            decisions,
            outcomes,
            calendar=calendar,
            completed_months=completed[:-1],
        )
    assert frame_calls == []


def test_identical_to_c00_cannot_pass_positive_paired_delta_gate() -> None:
    calendar = audit.load_registered_calendar()
    terminal = audit.deterministic_terminal_session(calendar[0], calendar)
    sessions = calendar[(calendar >= calendar[0]) & (calendar <= terminal)]
    decisions, outcomes = _forward_ledgers(sessions, selected_rank=1)
    result = audit.evaluate_candidate(
        decisions,
        outcomes,
        calendar=calendar,
        completed_months=_completed_month_ledger(decisions, outcomes),
    )
    assert result["paired_vs_C00_net40"]["point_estimate_delta_pct"] == pytest.approx(
        0.0, abs=1e-15
    )
    assert result["gate_checks"][
        "paired_point_delta_vs_C00_net40_positive"
    ] is False
    assert result["gate_passed"] is False
    assert result["winner"] is None


def test_tail_and_code_concentration_gates_are_independently_recomputed() -> None:
    calendar = audit.load_registered_calendar()
    terminal = audit.deterministic_terminal_session(calendar[0], calendar)
    sessions = calendar[(calendar >= calendar[0]) & (calendar <= terminal)]
    decisions, outcomes = _forward_ledgers(sessions)
    concentrated_decision_payloads: list[dict[str, object]] = []
    for row in decisions:
        payload = {
            key: value for key, value in row.items() if key not in audit.CHAIN_COLUMNS
        }
        payload["c02_rank2_code"] = "5999"
        payload["candidate_selected_code"] = "5999"
        concentrated_decision_payloads.append(_rehash_checkpoint_core(payload))
    concentrated_decisions = _chain(concentrated_decision_payloads)
    concentrated_outcome_payloads: list[dict[str, object]] = []
    for row, decision in zip(outcomes, concentrated_decisions, strict=True):
        payload = {
            key: value for key, value in row.items() if key not in audit.CHAIN_COLUMNS
        }
        payload["decision_record_sha256"] = decision["record_sha256"]
        concentrated_outcome_payloads.append(payload)
    result = audit.evaluate_candidate(
        concentrated_decisions,
        _chain(concentrated_outcome_payloads),
        calendar=calendar,
        completed_months=_completed_month_ledger(
            concentrated_decisions,
            _chain(concentrated_outcome_payloads),
        ),
    )
    assert result["unique_selected_codes"] == 1
    assert result["maximum_code_selection_share"] == 1.0
    assert result["top10_code_selection_share"] == 1.0
    assert result["top5_profit_codes_cash_net40_mean_pct"] == 0.0
    assert result["gate_checks"]["unique_codes_at_least"] is False
    assert result["gate_checks"]["maximum_code_share_at_most"] is False
    assert result["gate_checks"]["top10_code_share_at_most"] is False
    assert result["gate_checks"]["top5_profit_codes_cash_net40_positive"] is False
    assert result["gate_passed"] is False


def test_score_hash_is_outcome_free_and_canonical_path_overwrite_is_refused(
    tmp_path: Path,
) -> None:
    session = pd.Timestamp("2026-08-05")
    scores = pd.DataFrame(
        {
            "session_date": [str(session.date()), str(session.date())],
            "source_rank": [1, 2],
            "code": ["1001", "1002"],
            "name": ["a", "b"],
            "model_score": [1.0, 0.5],
            "feature_source_max_date": ["2026-08-04", "2026-08-04"],
            "score_generated_at": [
                "2026-08-05T08:00:00+09:00",
                "2026-08-05T08:00:00+09:00",
            ],
            "runtime_lock_sha256": [audit.RUNTIME_LOCK_SHA256] * 2,
            "runtime_lock_verified_at": [
                "2026-08-05T07:59:00+09:00",
                "2026-08-05T07:59:00+09:00",
            ],
            "source_manifest_sha256": ["a" * 64, "a" * 64],
            "c00_fold_manifest_sha256": ["b" * 64, "b" * 64],
        }
    )
    digest = runner.semantic_score_hash(scores)
    assert audit.semantic_score_hash(scores) == digest
    assert digest == runner.semantic_score_hash(
        scores.sample(frac=1.0, random_state=18)
    )
    with pytest.raises(runner.V18Error, match="outcome"):
        runner.semantic_score_hash(scores.assign(oc_return_pct=[1.0, -1.0]))
    same_day = scores.assign(feature_source_max_date=["2026-08-05"] * 2)
    with pytest.raises(audit.AuditError, match="D-1"):
        audit.semantic_score_hash(same_day)

    header_only = pd.DataFrame(columns=runner.SCORE_FIELDS)
    pd.testing.assert_frame_equal(
        runner.validate_score_ledger(header_only, allow_empty=True),
        header_only,
    )
    expected_empty_hash = hashlib.sha256(
        header_only.to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()
    assert runner.semantic_score_ledger_hash(
        header_only, allow_empty=True
    ) == audit.semantic_score_hash(header_only) == expected_empty_hash
    with pytest.raises(runner.V18Error, match="exact registered header"):
        runner.validate_score_ledger(
            header_only.assign(unregistered=pd.Series(dtype=str)),
            allow_empty=True,
        )
    with pytest.raises(audit.AuditError, match="fields changed"):
        audit.semantic_score_hash(header_only[list(reversed(runner.SCORE_FIELDS))])

    path = tmp_path / "canonical.json"
    runner.write_json({"first": 1}, path, exclusive=True)
    with pytest.raises(runner.V18Error, match="overwrite|exists"):
        runner.write_json({"second": 2}, path, exclusive=True)
    assert json.loads(path.read_text(encoding="utf-8")) == {"first": 1}

    audit_path = tmp_path / "audit.json"
    audit.write_json_exclusive({"first": 1}, audit_path)
    with pytest.raises(audit.AuditError, match="overwrite"):
        audit.write_json_exclusive({"second": 2}, audit_path)


def test_all_fail_closed_terminal_builds_result_from_header_only_score_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calendar = audit.load_registered_calendar()
    terminal = audit.deterministic_terminal_session(calendar[0], calendar)
    sessions = calendar[(calendar >= calendar[0]) & (calendar <= terminal)]
    selected_decisions, selected_outcomes = _forward_ledgers(sessions)
    decision_payloads: list[dict[str, object]] = []
    for row in selected_decisions:
        payload = {
            key: value for key, value in row.items() if key not in audit.CHAIN_COLUMNS
        }
        payload.update(
            {
                "source_complete": False,
                "model_complete": False,
                "c00_fold_manifest_sha256": None,
                "fold_model_bundle_file_sha256": None,
                "selected_source_rank": None,
                "c00_rank1_code": None,
                "c00_rank1_score": None,
                "c02_rank2_code": None,
                "c02_rank2_score": None,
                "candidate_selected_code": None,
                    "decision": "fail_closed_source_empty",
                "failure_reason": "source_missing_before_cutoff",
            }
        )
        decision_payloads.append(_rehash_checkpoint_core(payload))
    decisions = _chain(decision_payloads)
    outcome_payloads: list[dict[str, object]] = []
    zero_fields = (
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
    )
    for source, decision in zip(selected_outcomes, decisions, strict=True):
        payload = {
            key: value for key, value in source.items() if key not in audit.CHAIN_COLUMNS
        }
        payload.update(
            {
                "decision_record_sha256": decision["record_sha256"],
                "rank1_outcome_observed": False,
                "rank1_oc_return_pct": None,
                "rank2_outcome_observed": False,
                "rank2_oc_return_pct": None,
                "candidate_outcome_observed": False,
                **{field: 0.0 for field in zero_fields},
            }
        )
        outcome_payloads.append(payload)
    outcomes = _chain(outcome_payloads)
    completed = _completed_month_ledger(decisions, outcomes)
    monkeypatch.setattr(
        runner,
        "validate_checkpoint_evidence",
        lambda *args, **kwargs: {
            "checkpoint_proposal_set_sha256": "3" * 64,
            "checkpoint_core_object_set_sha256": "4" * 64,
            "checkpoint_evidence_set_sha256": "5" * 64,
        },
    )
    monkeypatch.setattr(
        runner, "validate_outcome_evidence", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "validate_predictor_evidence",
        lambda *args, **kwargs: {
            field: (
                0
                if field == "predictor_unique_raw_object_count"
                else "6" * 64
            )
            for field in audit.read_json(audit.DEFAULT_PROTOCOL)["result_contract"][
                "required_input_fields"
            ]
            if not field.startswith("checkpoint_")
        },
    )
    evaluation = runner.evaluate(
        decisions,
        outcomes,
        completed,
        calendar=calendar,
        source_manifest_directory=tmp_path,
        predictor_raw_store_root=tmp_path,
        scores=pd.DataFrame(columns=runner.SCORE_FIELDS),
        outcome_manifest_directory=tmp_path,
        outcome_raw_store_root=tmp_path,
        checkpoint_core_store_root=tmp_path,
    )
    assert evaluation["gate_evaluated"] is True
    assert evaluation["gate_passed"] is False
    assert evaluation["executed_days"] == 0
    required_inputs = json.loads(
        (RESEARCH / "model_v18_shoulder_state_protocol.json").read_text(
            encoding="utf-8"
        )
    )["result_contract"]["required_input_fields"]
    evaluation["input_bindings"] = {
        field: (0 if field == "predictor_unique_raw_object_count" else "a" * 64)
        for field in required_inputs
    }

    scores_path = tmp_path / "scores.csv"
    header_only = pd.DataFrame(columns=runner.SCORE_FIELDS)
    header_only.to_csv(scores_path, index=False, lineterminator="\n")
    picks_path = tmp_path / "picks.csv"
    runner.materialize_picks(decisions, outcomes).to_csv(
        picks_path, index=False, lineterminator="\n"
    )
    decision_path = tmp_path / "decisions.jsonl"
    outcome_path = tmp_path / "outcomes.jsonl"
    month_path = tmp_path / "months.jsonl"
    for path, rows in (
        (decision_path, decisions),
        (outcome_path, outcomes),
        (month_path, completed),
    ):
        path.write_bytes(
            b"".join(audit.canonical_json_bytes(row) + b"\n" for row in rows)
        )
    decoy_decision_path = tmp_path / "decoy-decisions.jsonl"
    decoy_outcome_path = tmp_path / "decoy-outcomes.jsonl"
    decoy_month_path = tmp_path / "decoy-months.jsonl"
    for path in (decoy_decision_path, decoy_outcome_path, decoy_month_path):
        path.write_text("decoy\n", encoding="utf-8")
    monkeypatch.setattr(runner, "DECISION_LEDGER", decoy_decision_path)
    monkeypatch.setattr(runner, "OUTCOME_LEDGER", decoy_outcome_path)
    monkeypatch.setattr(runner, "COMPLETED_MONTH_LEDGER", decoy_month_path)

    original_picks = picks_path.read_bytes()
    tampered_picks = pd.read_csv(picks_path, dtype={"code": "string"})
    tampered_picks.loc[0, "code"] = "9999"
    tampered_picks.to_csv(picks_path, index=False, lineterminator="\n")
    with pytest.raises(runner.V18Error, match="recomputed"):
        runner.build_result(
            evaluation,
            decisions,
            outcomes,
            completed,
            decision_ledger_path=decision_path,
            outcome_ledger_path=outcome_path,
            completed_month_ledger_path=month_path,
            score_output=scores_path,
            picks_output=picks_path,
        )
    picks_path.write_bytes(original_picks)

    result = runner.build_result(
        evaluation,
        decisions,
        outcomes,
        completed,
        decision_ledger_path=decision_path,
        outcome_ledger_path=outcome_path,
        completed_month_ledger_path=month_path,
        score_output=scores_path,
        picks_output=picks_path,
    )
    assert result["status"] == "forward_rejected_candidate"
    assert result["candidate_gate"]["passed"] is False
    assert result["artifact_sha256"]["score_semantic_sha256"] == (
        audit.semantic_score_hash(header_only)
    )
    assert result["artifact_sha256"]["decision_ledger_sha256"] == (
        runner.sha256_file(decision_path)
    )
    assert result["artifact_sha256"]["outcome_ledger_sha256"] == (
        runner.sha256_file(outcome_path)
    )
    assert result["artifact_sha256"]["completed_month_ledger_sha256"] == (
        runner.sha256_file(month_path)
    )
    assert result["artifact_sha256"]["decision_ledger_sha256"] != (
        runner.sha256_file(decoy_decision_path)
    )


def test_independent_abort_result_is_outcome_blind_exact_and_symlink_safe(
    tmp_path: Path,
) -> None:
    protocol = json.loads(
        (RESEARCH / "model_v18_shoulder_state_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    runtime_lock = json.loads(
        (RESEARCH / "model_v18_runtime_lock.json").read_text(encoding="utf-8")
    )
    artifact_paths = {
        "decision_ledger_sha256": tmp_path / "decisions.jsonl",
        "outcome_ledger_sha256": tmp_path / "outcomes.jsonl",
        "completed_month_ledger_sha256": tmp_path / "months.jsonl",
        "score_output_sha256": tmp_path / "scores.csv",
        "picks_output_sha256": tmp_path / "picks.csv",
    }
    artifact_paths["decision_ledger_sha256"].write_bytes(
        b"sealed-decision-bytes-that-must-not-be-parsed\n"
    )
    artifact_paths["score_output_sha256"].write_bytes(
        b"not-even-a-valid-score-csv\n"
    )
    required_inputs = protocol["result_contract"]["required_input_fields"]
    input_bindings = {field: None for field in required_inputs}
    input_bindings.update(
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
    versions = {
        item["name"]: item["version"]
        for item in runtime_lock["runtime"]["distributions"]
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": audit.sha256_file(
            RESEARCH / "model_v18_shoulder_state_protocol.json"
        ),
        "runner_sha256": audit.sha256_file(
            RESEARCH / "model_v18_shoulder_state_runner.py"
        ),
        "activation_payload_sha256": None,
        "activation_receipt_sha256": None,
        "activation_receipt_commit_sha": None,
        "status": "aborted_integrity_failure",
        "failure_reason": "source_integrity_failure",
        "integrity_stage": "source_ingestion",
        "authority": protocol["result_contract"]["authority_values"],
        "input": input_bindings,
        "forward_period": None,
        "state_months": None,
        "models": None,
        "candidate_gate": None,
        "decision": {
            "research_nominee": None,
            "failure_reason": "source_integrity_failure",
            "integrity_stage": "source_ingestion",
        },
        "artifact_sha256": {
            "decision_ledger_sha256": audit.sha256_file(
                artifact_paths["decision_ledger_sha256"]
            ),
            "outcome_ledger_sha256": None,
            "completed_month_ledger_sha256": None,
            "score_output_sha256": audit.sha256_file(
                artifact_paths["score_output_sha256"]
            ),
            "score_semantic_sha256": None,
            "picks_output_sha256": None,
        },
        "runtime": {
            "runtime_lock_sha256": protocol["runtime_lock_contract"]["file_sha256"],
            "runtime_lock_self_sha256": protocol["runtime_lock_contract"][
                "self_sha256"
            ],
            "runtime_lock_verified_at": "2026-08-05T07:00:00+09:00",
            "python_version": runtime_lock["runtime"]["python"]["version"],
            "numpy_version": versions["numpy"],
            "pandas_version": versions["pandas"],
            "scikit_learn_version": versions["scikit-learn"],
        },
    }
    result_path = tmp_path / "abort-result.json"
    result_path.write_bytes(audit.canonical_json_file_bytes(result))
    missing_payload = tmp_path / "activation-payload.json"
    missing_receipt = tmp_path / "activation-receipt.json"
    assert audit.validate_abort_result(
        result,
        protocol,
        runtime_lock=runtime_lock,
        protocol_path=RESEARCH / "model_v18_shoulder_state_protocol.json",
        runner_path=RESEARCH / "model_v18_shoulder_state_runner.py",
        activation_payload_path=missing_payload,
        activation_receipt_path=missing_receipt,
        artifact_paths=artifact_paths,
        result_path=result_path,
    )["status"] == "aborted_integrity_failure"

    exposed = copy.deepcopy(result)
    exposed["models"] = {"forbidden_metric": 1.0}
    with pytest.raises(audit.AuditError, match="exposed sealed models"):
        audit.validate_abort_result(
            exposed,
            protocol,
            runtime_lock=runtime_lock,
            protocol_path=RESEARCH / "model_v18_shoulder_state_protocol.json",
            runner_path=RESEARCH / "model_v18_shoulder_state_runner.py",
            activation_payload_path=missing_payload,
            activation_receipt_path=missing_receipt,
            artifact_paths=artifact_paths,
        )
    invalid_reason = copy.deepcopy(result)
    invalid_reason["failure_reason"] = "free form / secret"
    with pytest.raises(audit.AuditError, match="machine reason"):
        audit.validate_abort_result(
            invalid_reason,
            protocol,
            runtime_lock=runtime_lock,
            protocol_path=RESEARCH / "model_v18_shoulder_state_protocol.json",
            runner_path=RESEARCH / "model_v18_shoulder_state_runner.py",
            activation_payload_path=missing_payload,
            activation_receipt_path=missing_receipt,
            artifact_paths=artifact_paths,
        )
    unregistered_reason = copy.deepcopy(result)
    unregistered_reason["failure_reason"] = "syntactically_valid_but_unregistered"
    unregistered_reason["decision"]["failure_reason"] = (
        "syntactically_valid_but_unregistered"
    )
    with pytest.raises(audit.AuditError, match="registered machine reason"):
        audit.validate_abort_result(
            unregistered_reason,
            protocol,
            runtime_lock=runtime_lock,
            protocol_path=RESEARCH / "model_v18_shoulder_state_protocol.json",
            runner_path=RESEARCH / "model_v18_shoulder_state_runner.py",
            activation_payload_path=missing_payload,
            activation_receipt_path=missing_receipt,
            artifact_paths=artifact_paths,
        )
    wrong_stage_reason = copy.deepcopy(result)
    wrong_stage_reason["integrity_stage"] = "monthly_fold"
    wrong_stage_reason["decision"]["integrity_stage"] = "monthly_fold"
    with pytest.raises(audit.AuditError, match="not registered for integrity stage"):
        audit.validate_abort_result(
            wrong_stage_reason,
            protocol,
            runtime_lock=runtime_lock,
            protocol_path=RESEARCH / "model_v18_shoulder_state_protocol.json",
            runner_path=RESEARCH / "model_v18_shoulder_state_runner.py",
            activation_payload_path=missing_payload,
            activation_receipt_path=missing_receipt,
            artifact_paths=artifact_paths,
        )

    symlink_target = tmp_path / "attacker-outcomes.jsonl"
    symlink_target.write_bytes(b"attacker\n")
    artifact_paths["outcome_ledger_sha256"].symlink_to(symlink_target)
    with pytest.raises(audit.AuditError, match="single-link regular file"):
        audit.validate_abort_result(
            result,
            protocol,
            runtime_lock=runtime_lock,
            protocol_path=RESEARCH / "model_v18_shoulder_state_protocol.json",
            runner_path=RESEARCH / "model_v18_shoulder_state_runner.py",
            activation_payload_path=missing_payload,
            activation_receipt_path=missing_receipt,
            artifact_paths=artifact_paths,
        )


def test_runner_abort_command_is_canonical_exclusive_and_never_parses_ledgers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol_path = RESEARCH / "model_v18_shoulder_state_protocol.json"
    runtime_path = RESEARCH / "model_v18_runtime_lock.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    runtime_lock = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_sha = audit.sha256_file(runtime_path)
    versions = {
        item["name"]: item["version"]
        for item in runtime_lock["runtime"]["distributions"]
    }
    paths = {
        "RESULT_OUTPUT": tmp_path / "result.json",
        "DECISION_LEDGER": tmp_path / "decisions.jsonl",
        "OUTCOME_LEDGER": tmp_path / "outcomes.jsonl",
        "COMPLETED_MONTH_LEDGER": tmp_path / "months.jsonl",
        "SCORE_OUTPUT": tmp_path / "scores.csv",
        "PICKS_OUTPUT": tmp_path / "picks.csv",
        "ACTIVATION_PAYLOAD": tmp_path / "activation-payload.json",
        "ACTIVATION_RECEIPT": tmp_path / "activation-receipt.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(runner, name, path)
    paths["DECISION_LEDGER"].write_bytes(b"not-json-and-must-not-be-opened-semantically\n")
    paths["SCORE_OUTPUT"].write_bytes(b"not,a,registered,score,header\n")
    monkeypatch.setattr(runner, "RUNTIME_LOCK_SHA256", runtime_sha)
    monkeypatch.setattr(runner, "_STRICT_RUNTIME_ACTIVE", True)
    monkeypatch.setattr(
        runner, "_STRICT_RUNTIME_VERIFIED_AT", "2026-08-05T07:00:00+09:00"
    )
    monkeypatch.setattr(
        runner,
        "validate_protocol",
        lambda *args, **kwargs: (protocol, audit.sha256_file(protocol_path)),
    )
    monkeypatch.setattr(
        runner,
        "validate_runtime_lock",
        lambda *args, **kwargs: (runtime_lock, runtime_sha),
    )
    monkeypatch.setattr(
        runner.platform,
        "python_version",
        lambda: runtime_lock["runtime"]["python"]["version"],
    )
    monkeypatch.setattr(runner.np, "__version__", versions["numpy"])
    monkeypatch.setattr(runner.pd, "__version__", versions["pandas"])
    monkeypatch.setattr(runner.sklearn, "__version__", versions["scikit-learn"])
    monkeypatch.setattr(
        runner.pd,
        "read_csv",
        lambda *args, **kwargs: pytest.fail("abort parsed a sealed score artifact"),
    )
    monkeypatch.setattr(
        runner,
        "evaluate",
        lambda *args, **kwargs: pytest.fail("abort reached performance evaluation"),
    )

    with pytest.raises(runner.V18Error, match="not allowed for its stage"):
        runner.build_integrity_abort_result(
            failure_reason="source_integrity_failure",
            integrity_stage="monthly_fold",
        )

    result = runner.build_integrity_abort_result(
        failure_reason="source_integrity_failure",
        integrity_stage="source_ingestion",
    )
    assert result["models"] is None
    assert result["artifact_sha256"]["decision_ledger_sha256"] == (
        audit.sha256_file(paths["DECISION_LEDGER"])
    )
    assert result["artifact_sha256"]["score_semantic_sha256"] is None
    assert runner.main(
        [
            "abort",
            "--failure-reason",
            "source_integrity_failure",
            "--integrity-stage",
            "source_ingestion",
        ]
    ) == 0
    persisted = json.loads(paths["RESULT_OUTPUT"].read_text(encoding="utf-8"))
    assert persisted == result
    assert paths["RESULT_OUTPUT"].read_bytes() == audit.canonical_json_file_bytes(result)
    audit.validate_abort_result(
        persisted,
        protocol,
        runtime_lock=runtime_lock,
        protocol_path=protocol_path,
        runner_path=RESEARCH / "model_v18_shoulder_state_runner.py",
        activation_payload_path=paths["ACTIVATION_PAYLOAD"],
        activation_receipt_path=paths["ACTIVATION_RECEIPT"],
        artifact_paths={
            "decision_ledger_sha256": paths["DECISION_LEDGER"],
            "outcome_ledger_sha256": paths["OUTCOME_LEDGER"],
            "completed_month_ledger_sha256": paths["COMPLETED_MONTH_LEDGER"],
            "score_output_sha256": paths["SCORE_OUTPUT"],
            "picks_output_sha256": paths["PICKS_OUTPUT"],
        },
        result_path=paths["RESULT_OUTPUT"],
    )
    with pytest.raises(runner.V18Error, match="already exists|overwrite"):
        runner.main(
            [
                "abort",
                "--failure-reason",
                "source_integrity_failure",
                "--integrity-stage",
                "source_ingestion",
            ]
        )
    with pytest.raises(runner.V18Error, match="machine token"):
        runner.build_integrity_abort_result(
            failure_reason="free form / metric=1.2",
            integrity_stage="source_ingestion",
        )
    with pytest.raises(runner.V18Error, match="machine token"):
        runner.build_integrity_abort_result(
            failure_reason="syntactically_valid_but_unregistered",
            integrity_stage="source_ingestion",
        )


def test_score_and_decision_timestamp_causality_rejects_reordering() -> None:
    decisions, _ = _forward_ledgers(audit.load_registered_calendar()[:1])
    decision = copy.deepcopy(decisions[0])
    decision["computed_at"] = "2026-08-05T08:00:00+09:00"
    scores = pd.DataFrame(
        {
            "session_date": ["2026-08-05", "2026-08-05"],
            "source_rank": [1, 2],
            "code": [decision["c00_rank1_code"], decision["c02_rank2_code"]],
            "name": ["rank1", "rank2"],
            "model_score": [1.0, 0.5],
            "feature_source_max_date": ["2026-08-04", "2026-08-04"],
            "score_generated_at": [
                "2026-08-05T07:30:00+09:00",
                "2026-08-05T07:30:00+09:00",
            ],
            "runtime_lock_sha256": [audit.RUNTIME_LOCK_SHA256] * 2,
            "runtime_lock_verified_at": [
                "2026-08-05T07:26:00+09:00",
                "2026-08-05T07:26:00+09:00",
            ],
            "source_manifest_sha256": ["e" * 64, "e" * 64],
            "c00_fold_manifest_sha256": ["f" * 64, "f" * 64],
        }
    )
    arguments = {
        "source_manifests": {
            "e" * 64: {
                "runtime_lock_sha256": audit.RUNTIME_LOCK_SHA256,
                "runtime_lock_verified_at": "2026-08-05T07:00:00+09:00",
                "source_received_at": "2026-08-05T07:15:00+09:00",
                "created_at": "2026-08-05T07:20:00+09:00",
                "sealed_at": "2026-08-05T07:25:00+09:00",
            }
        },
        "state_manifests": {
            "2026-08": {"created_at": "2026-08-04T14:00:00+09:00"}
        },
        "fold_manifests": {
            "f" * 64: {
                "runtime_lock_sha256": audit.RUNTIME_LOCK_SHA256,
                "runtime_lock_verified_at": "2026-08-05T06:00:00+09:00",
                "fit_started_at": "2026-08-05T06:30:00+09:00",
                "fit_completed_at": "2026-08-05T07:00:00+09:00",
                "sealed_at": "2026-08-05T07:10:00+09:00",
            }
        },
        "activation_ready_at": "2026-08-04T13:01:00+09:00",
    }
    audit.validate_pit_causality([decision], scores, **arguments)

    premature_score = scores.copy()
    premature_score["score_generated_at"] = "2026-08-05T07:00:00+09:00"
    with pytest.raises(audit.AuditError, match="score was generated"):
        audit.validate_pit_causality([decision], premature_score, **arguments)

    premature_decision = copy.deepcopy(decision)
    premature_decision["computed_at"] = "2026-08-05T07:20:00+09:00"
    with pytest.raises(audit.AuditError, match="PIT DAG|decision predates"):
        audit.validate_pit_causality([premature_decision], scores, **arguments)


def test_canonical_result_cli_rejects_every_noncanonical_input_and_output_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonical = {
        "RESULT_OUTPUT": tmp_path / "canonical-result.json",
        "DECISION_LEDGER": tmp_path / "canonical-decisions.jsonl",
        "OUTCOME_LEDGER": tmp_path / "canonical-outcomes.jsonl",
        "COMPLETED_MONTH_LEDGER": tmp_path / "canonical-months.jsonl",
        "SCORE_OUTPUT": tmp_path / "canonical-scores.csv",
        "PICKS_OUTPUT": tmp_path / "canonical-picks.csv",
        "SOURCE_MANIFEST_DIR": tmp_path / "canonical-source-manifests",
        "OUTCOME_MANIFEST_DIR": tmp_path / "canonical-outcome-manifests",
    }
    for name, path in canonical.items():
        monkeypatch.setattr(runner, name, path)
    for name in ("DECISION_LEDGER", "OUTCOME_LEDGER", "COMPLETED_MONTH_LEDGER"):
        canonical[name].write_bytes(b"")
    pd.DataFrame(columns=runner.SCORE_FIELDS).to_csv(
        canonical["SCORE_OUTPUT"], index=False, lineterminator="\n"
    )
    canonical["SOURCE_MANIFEST_DIR"].mkdir()
    canonical["OUTCOME_MANIFEST_DIR"].mkdir()

    monkeypatch.setattr(
        runner,
        "validate_runtime_lock",
        lambda *args, **kwargs: ({"lock_id": "test"}, "a" * 64),
    )
    monkeypatch.setattr(
        runner,
        "validate_protocol",
        lambda *args, **kwargs: ({"protocol_id": EXPECTED_PROTOCOL_ID}, "b" * 64),
    )
    monkeypatch.setattr(
        runner,
        "validate_canonical_activation_artifacts",
        lambda: ({}, "c" * 64, {}, "d" * 64),
    )
    monkeypatch.setattr(
        runner, "validate_completed_month_records", lambda records: records
    )
    monkeypatch.setattr(
        runner,
        "evaluate",
        lambda *args, **kwargs: pytest.fail(
            "noncanonical result inputs reached terminal metric evaluation"
        ),
    )

    alternatives = {
        "--output": tmp_path / "noncanonical-result.json",
        "--decisions": tmp_path / "noncanonical-decisions.jsonl",
        "--outcomes": tmp_path / "noncanonical-outcomes.jsonl",
        "--completed-months": tmp_path / "noncanonical-months.jsonl",
        "--scores": tmp_path / "noncanonical-scores.csv",
        "--picks": tmp_path / "noncanonical-picks.csv",
        "--source-manifest-directory": tmp_path / "noncanonical-sources",
        "--outcome-manifest-directory": tmp_path / "noncanonical-outcomes-dir",
    }
    for option, path in alternatives.items():
        if option in {"--decisions", "--outcomes", "--completed-months"}:
            path.write_bytes(b"")
        elif option == "--scores":
            pd.DataFrame(columns=runner.SCORE_FIELDS).to_csv(
                path, index=False, lineterminator="\n"
            )
        arguments = [
            "evaluate",
            "--predictor-raw-store-root",
            str(tmp_path),
            "--outcome-raw-store-root",
            str(tmp_path),
            "--checkpoint-core-store-root",
            str(tmp_path),
            option,
            str(path),
        ]
        if option == "--output":
            # Result output is not a CLI parameter at all; the only writer is
            # hard-bound to RESULT_OUTPUT.
            with pytest.raises(SystemExit):
                runner.main(arguments)
        else:
            with pytest.raises(
                runner.V18Error, match="canonical result .* path changed"
            ):
                runner.main(arguments)
        assert not canonical["RESULT_OUTPUT"].exists()
    assert not alternatives["--output"].exists()


def test_checkpoint_derivation_selects_exact_day2_pair_and_decide_has_no_score_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def pair(session: str, codes: tuple[str, str]) -> pd.DataFrame:
        target = pd.Timestamp(session)
        previous = target - pd.offsets.BDay(1)
        return pd.DataFrame(
            {
                "session_date": [session, session],
                "source_rank": [1, 2],
                "code": list(codes),
                "name": [f"name-{code}" for code in codes],
                "model_score": [1.0, 0.5],
                "feature_source_max_date": [str(previous.date())] * 2,
                "score_generated_at": [f"{session}T08:00:00+09:00"] * 2,
                "runtime_lock_sha256": [audit.RUNTIME_LOCK_SHA256] * 2,
                "runtime_lock_verified_at": [f"{session}T07:59:00+09:00"] * 2,
                "source_manifest_sha256": ["a" * 64] * 2,
                "c00_fold_manifest_sha256": ["b" * 64] * 2,
            },
            columns=runner.SCORE_FIELDS,
        )

    cumulative = pd.concat(
        [
            pair("2026-08-05", ("1001", "1002")),
            pair("2026-08-06", ("2001", "2002")),
        ],
        ignore_index=True,
    )
    score_path = tmp_path / "scores.csv"
    cumulative.to_csv(score_path, index=False, lineterminator="\n")
    monkeypatch.setattr(runner, "SCORE_OUTPUT", score_path)

    selected = runner._canonical_target_score_pair(pd.Timestamp("2026-08-06"))
    assert selected is not None
    assert selected["session_date"].tolist() == ["2026-08-06", "2026-08-06"]
    assert selected["code"].astype(str).tolist() == ["2001", "2002"]

    # The checkpoint builder, rather than the later evidence resolver, is the
    # only component allowed to consume the canonical cumulative score ledger.
    source_directory = tmp_path / "source-manifests"
    state_directory = tmp_path / "state-manifests"
    fold_directory = tmp_path / "fold-manifests"
    bundle_directory = tmp_path / "fold-models"
    source_directory.mkdir()
    state_directory.mkdir()
    fold_directory.mkdir()
    bundle_directory.mkdir()
    source_path = source_directory / "2026-08-06.json"
    state_path = state_directory / "2026-08.json"
    for path in (source_path, state_path):
        path.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(runner, "SOURCE_MANIFEST_DIR", source_directory)
    monkeypatch.setattr(runner, "STATE_MANIFEST_DIR", state_directory)
    monkeypatch.setattr(runner, "FOLD_MANIFEST_DIR", fold_directory)
    monkeypatch.setattr(runner, "FOLD_MODEL_DIR", bundle_directory)
    monkeypatch.setattr(runner, "_discover_activation_context", lambda records: {})

    def fake_build(scores, **kwargs):  # type: ignore[no-untyped-def]
        captured["scores"] = scores.copy()
        assert kwargs["return_decision_core"] is True
        return {"decision": "synthetic-checkpoint-core"}

    monkeypatch.setattr(runner, "build_decision_ledger", fake_build)
    derived, _ = runner._derive_canonical_checkpoint_core(
        pd.Timestamp("2026-08-06"), []
    )
    assert derived == {"decision": "synthetic-checkpoint-core"}
    derived_scores = captured["scores"]
    assert derived_scores["session_date"].tolist() == [
        "2026-08-06",
        "2026-08-06",
    ]
    assert derived_scores["code"].astype(str).tolist() == ["2001", "2002"]

    other_scores = tmp_path / "noncanonical-scores.csv"
    cumulative.to_csv(other_scores, index=False, lineterminator="\n")
    with pytest.raises(SystemExit):
        runner._build_parser().parse_args(
            [
                "decide",
                "--session",
                "2026-08-06",
                "--checkpoint-core-store-root",
                str(tmp_path),
                "--scores",
                str(other_scores),
            ]
        )


def test_exclusive_writer_has_atomic_no_replace_under_concurrency(
    tmp_path: Path,
) -> None:
    target = tmp_path / "raced-audit.json"
    barrier = threading.Barrier(2)

    def attempt(value: int) -> tuple[str, int]:
        barrier.wait(timeout=5)
        try:
            audit.write_json_exclusive({"winner": value}, target)
        except audit.AuditError:
            return "rejected", value
        return "created", value

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, (1, 2)))
    assert sorted(item[0] for item in outcomes) == ["created", "rejected"]
    created_value = next(value for status, value in outcomes if status == "created")
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "winner": created_value
    }
