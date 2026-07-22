#!/usr/bin/env python3
"""Audit the canonical model-v0.8 zero-base research ledger.

The audit is intentionally lightweight: it reads only committed JSON/text
artifacts and never rebuilds the multi-million-row research panels.  It binds
the three protocol/result/manifest chains, proves that every registered
feature hypothesis has a recorded outcome, and locks the retrospective-only
decision that no production promotion is authorised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

TARGET_PROTOCOL = Path("research/model_v08_target_protocol.json")
TARGET_RESULT = Path("research/model_v08_target_result.json")
TARGET_MANIFEST = Path("research/model_v08_target_result.manifest.json")
FEATURE_PROTOCOL = Path("research/model_v08_feature_protocol.json")
FEATURE_RESULT = Path("research/model_v08_feature_result.json")
FEATURE_MANIFEST = Path("research/model_v08_feature_result.manifest.json")
TDNET_PROTOCOL = Path("research/model_v08_tdnet_protocol.json")
TDNET_RESULT = Path("research/model_v08_tdnet_result.json")
TDNET_MANIFEST = Path("research/model_v08_tdnet_result.manifest.json")

EXPECTED_HYPOTHESIS_COUNTS = {
    "zero_base_single_features": 6,
    "liquidity_and_availability": 13,
    "general_feature_groups": 12,
    "tdnet_title_feature_groups": 12,
    "total_new_feature_hypotheses": 43,
    "universe_hypotheses": 5,
}

EXPECTED_SHADOW_VALUES = {
    "L4_mean_shadow": {
        "net20": 0.28319198192908746,
        "net40": 0.08657544057570407,
        "net60": -0.11004110077767944,
        "top20_removed_net20": 0.013510098521216481,
        "max_t_lower": -0.09407765904928733,
    },
    "L6_robustness_shadow": {
        "net20": 0.2593760660947221,
        "net40": 0.06802268263607546,
        "net60": -0.12333070082257115,
        "top20_removed_net20": 0.017912690513375497,
        "max_t_lower": -0.11789357488365276,
    },
}

_MISSING = object()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _json_value(value: Any) -> Any:
    if value is _MISSING:
        return "<missing>"
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _negative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) < 0.0
    )


class _Checks:
    def __init__(self) -> None:
        self.checks: dict[str, dict[str, Any]] = {}

    def add(self, name: str, passed: bool, detail: Any) -> None:
        if name in self.checks:
            raise AssertionError(f"duplicate audit check: {name}")
        self.checks[name] = {
            "passed": bool(passed),
            "detail": _json_value(detail),
        }

    def equal(self, name: str, actual: Any, expected: Any) -> None:
        self.add(
            name,
            actual == expected,
            {"actual": actual, "expected": expected},
        )

    def close(
        self,
        name: str,
        actual: Any,
        expected: float,
        *,
        tolerance: float = 1e-12,
    ) -> None:
        passed = False
        try:
            value = float(actual)
            passed = math.isfinite(value) and math.isclose(
                value, expected, rel_tol=0.0, abs_tol=tolerance
            )
        except (TypeError, ValueError):
            value = actual
        self.add(
            name,
            passed,
            {"actual": value, "expected": expected, "absolute_tolerance": tolerance},
        )

    @property
    def failures(self) -> list[str]:
        return [name for name, value in self.checks.items() if not value["passed"]]


def _manifest_artifact_hash(
    manifest: Mapping[str, Any], artifact: str
) -> Any:
    direct = manifest.get(f"{artifact}_sha256", _MISSING)
    if direct is not _MISSING:
        return direct
    for registry_name in ("canonical_artifacts", "artifacts", "files"):
        registry = manifest.get(registry_name)
        if isinstance(registry, Mapping):
            record = registry.get(artifact)
            if isinstance(record, Mapping) and "sha256" in record:
                return record["sha256"]
    return _MISSING


def _check_hash_chain(
    checks: _Checks,
    *,
    name: str,
    root: Path,
    protocol_path: Path,
    result_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    result_protocol_sha: Any,
) -> dict[str, str]:
    protocol_sha = sha256_file(root / protocol_path)
    result_sha = sha256_file(root / result_path)
    manifest_protocol_sha = _manifest_artifact_hash(manifest, "protocol")
    manifest_result_sha = _manifest_artifact_hash(manifest, "result")
    checks.equal(f"{name}.result_binds_protocol", result_protocol_sha, protocol_sha)
    checks.equal(
        f"{name}.manifest_binds_protocol", manifest_protocol_sha, protocol_sha
    )
    checks.equal(f"{name}.manifest_binds_result", manifest_result_sha, result_sha)
    return {
        "protocol_sha256": protocol_sha,
        "result_sha256": result_sha,
        "manifest_sha256": sha256_file(root / manifest_path),
    }


def _check_local_file_record(
    checks: _Checks,
    *,
    root: Path,
    name: str,
    path_value: Any,
    sha_value: Any,
) -> None:
    if not isinstance(path_value, str) or not isinstance(sha_value, str):
        checks.add(
            name,
            False,
            {"path": path_value, "declared_sha256": sha_value},
        )
        return
    path = Path(path_value)
    if path.is_absolute():
        checks.add(
            name,
            True,
            {"external_path": path_value, "declared_sha256": sha_value},
        )
        return
    resolved = root / path
    actual = sha256_file(resolved) if resolved.is_file() else None
    checks.equal(name, actual, sha_value)


def _check_manifest_local_records(
    checks: _Checks,
    *,
    root: Path,
    prefix: str,
    manifest: Mapping[str, Any],
) -> None:
    """Verify every conventional local path/hash record in a manifest."""

    for key, value in manifest.items():
        if key.endswith("_path"):
            stem = key[: -len("_path")]
            sha_key = f"{stem}_sha256"
            if sha_key in manifest:
                _check_local_file_record(
                    checks,
                    root=root,
                    name=f"{prefix}.{stem}",
                    path_value=value,
                    sha_value=manifest[sha_key],
                )
    for registry_name in ("canonical_artifacts", "artifacts", "files"):
        registry = manifest.get(registry_name)
        if not isinstance(registry, Mapping):
            continue
        for artifact, record in registry.items():
            if (
                isinstance(record, Mapping)
                and "path" in record
                and "sha256" in record
            ):
                _check_local_file_record(
                    checks,
                    root=root,
                    name=f"{prefix}.{registry_name}.{artifact}",
                    path_value=record["path"],
                    sha_value=record["sha256"],
                )
    implementation = _nested(manifest, "implementation", "files")
    if isinstance(implementation, Mapping):
        for path_value, sha_value in implementation.items():
            _check_local_file_record(
                checks,
                root=root,
                name=f"{prefix}.implementation.{path_value}",
                path_value=path_value,
                sha_value=sha_value,
            )


def _candidate_set(
    checks: _Checks,
    name: str,
    registered: Sequence[Any],
    results: Any,
) -> set[str]:
    expected = {str(value) for value in registered}
    actual = set(results) if isinstance(results, Mapping) else set()
    checks.equal(name, sorted(actual), sorted(expected))
    return actual


def _target_checks(
    checks: _Checks,
    *,
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    checks.equal(
        "target.protocol_id",
        result.get("protocol_id"),
        protocol.get("protocol_id"),
    )
    checks.equal(
        "target.protocol_production_promotion_false",
        _nested(protocol, "authority", "production_promotion_allowed"),
        False,
    )
    checks.equal(
        "target.result_production_promotion_false",
        result.get("production_promotion_allowed"),
        False,
    )
    checks.equal(
        "target.result_production_unchanged",
        result.get("production_model_changed"),
        False,
    )
    checks.equal(
        "target.manifest_production_promotion_false",
        manifest.get("production_promotion_allowed"),
        False,
    )
    checks.equal(
        "target.manifest_production_unchanged",
        manifest.get("production_model_changed"),
        False,
    )
    checks.equal(
        "target.result_manifest_runner_hash",
        result.get("runner_sha256"),
        _manifest_artifact_hash(manifest, "runner"),
    )
    checks.equal(
        "target.panel_hash_claims",
        [
            _nested(protocol, "frozen_input", "panel_sha256"),
            result.get("panel_sha256", _MISSING),
            _nested(manifest, "frozen_input", "panel_sha256"),
        ],
        [
            _nested(protocol, "frozen_input", "panel_sha256"),
        ]
        * 3,
    )
    error_record = _nested(result, "reproduction_audits", "error_case_artifact")
    manifest_error = _nested(manifest, "canonical_artifacts", "error_cases")
    if isinstance(error_record, Mapping) and isinstance(manifest_error, Mapping):
        checks.equal(
            "target.error_case_hash_claims",
            error_record.get("sha256"),
            manifest_error.get("sha256"),
        )
    else:
        checks.add(
            "target.error_case_hash_claims",
            False,
            {"result": error_record, "manifest": manifest_error},
        )

    registrations = _nested(protocol, "traceability", "stage_registrations")
    expected_registration_keys = {f"stage_{number}" for number in range(1, 5)}
    checks.equal(
        "target.stable_registration_set",
        sorted(registrations) if isinstance(registrations, Mapping) else [],
        sorted(expected_registration_keys),
    )
    chain_keys = {
        "stage_1": "zero_base_screen",
        "stage_2": "rank_target_sensitivity",
        "stage_3": "liquidity_hypotheses",
        "stage_4": "fixed_role_confirmation",
    }
    registration_payloads: dict[str, dict[str, Any]] = {}
    for registration_id, chain_id in chain_keys.items():
        record = (
            registrations.get(registration_id, {})
            if isinstance(registrations, Mapping)
            else {}
        )
        record = record if isinstance(record, Mapping) else {}
        _check_local_file_record(
            checks,
            root=root,
            name=f"target.registration.{registration_id}.file_hash",
            path_value=record.get("path"),
            sha_value=record.get("sha256"),
        )
        checks.equal(
            f"target.registration.{registration_id}.result_path",
            record.get("path", _MISSING),
            _nested(result, "research_chain", chain_id, "registration_path"),
        )
        checks.equal(
            f"target.registration.{registration_id}.result_registration_hash",
            record.get("sha256", _MISSING),
            _nested(result, "research_chain", chain_id, "registration_sha256"),
        )
        checks.equal(
            f"target.registration.{registration_id}.source_result_hash",
            record.get("source_result_sha256", _MISSING),
            _nested(result, "research_chain", chain_id, "source_result_sha256"),
        )
        checks.equal(
            f"target.registration.{registration_id}.source_runner_hash",
            record.get("source_runner_sha256", _MISSING),
            _nested(result, "research_chain", chain_id, "source_runner_sha256"),
        )
        path_value = record.get("path")
        if isinstance(path_value, str) and not Path(path_value).is_absolute():
            try:
                registration_payloads[registration_id] = _read_json(root / path_value)
                checks.add(
                    f"target.registration.{registration_id}.json_object",
                    True,
                    {"path": path_value},
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                checks.add(
                    f"target.registration.{registration_id}.json_object",
                    False,
                    {"path": path_value, "error": str(exc)},
                )
        else:
            checks.add(
                f"target.registration.{registration_id}.json_object",
                False,
                {"path": path_value},
            )

    stages = protocol.get("stages", {})
    candidate_results = result.get("candidate_results", {})
    stage1_registered = _nested(stages, "zero_base_screen", "candidate_ids")
    stage2_registered = _nested(
        stages, "rank_target_sensitivity", "candidate_ids"
    )
    stage3_registered = _nested(stages, "liquidity_hypotheses", "candidate_ids")
    if not isinstance(stage1_registered, list):
        stage1_registered = []
    if not isinstance(stage2_registered, list):
        stage2_registered = []
    if not isinstance(stage3_registered, list):
        stage3_registered = []

    stage1_registration = registration_payloads.get("stage_1", {})
    stage1_baseline = _nested(stage1_registration, "baseline", "id")
    stage1_feature_records = stage1_registration.get("single_feature_hypotheses", [])
    stage1_model_records = stage1_registration.get("target_model_hypotheses", [])
    stage1_from_registration = (
        [str(stage1_baseline)] if stage1_baseline is not _MISSING else []
    )
    if isinstance(stage1_feature_records, list):
        stage1_from_registration.extend(
            f"feature__{record['id']}"
            for record in stage1_feature_records
            if isinstance(record, Mapping) and "id" in record
        )
    if isinstance(stage1_model_records, list):
        stage1_from_registration.extend(
            str(record["id"] if isinstance(record, Mapping) else record)
            for record in stage1_model_records
            if (isinstance(record, Mapping) and "id" in record)
            or isinstance(record, str)
        )
    stage2_registration = registration_payloads.get("stage_2", {})
    stage2_records = stage2_registration.get("registered_candidates", [])
    stage2_from_registration = [
        str(record["id"])
        for record in stage2_records
        if isinstance(record, Mapping) and "id" in record
    ] if isinstance(stage2_records, list) else []
    stage3_registration = registration_payloads.get("stage_3", {})
    stage3_records = [
        *(
            stage3_registration.get("controls", [])
            if isinstance(stage3_registration.get("controls"), list)
            else []
        ),
        *(
            stage3_registration.get("registered_hypotheses", [])
            if isinstance(stage3_registration.get("registered_hypotheses"), list)
            else []
        ),
    ]
    stage3_from_registration = [
        str(record["id"])
        for record in stage3_records
        if isinstance(record, Mapping) and "id" in record
    ]
    checks.equal(
        "target.stage1_protocol_matches_stable_registration",
        stage1_registered,
        stage1_from_registration,
    )
    checks.equal(
        "target.stage2_protocol_matches_stable_registration",
        stage2_registered,
        stage2_from_registration,
    )
    checks.equal(
        "target.stage3_protocol_matches_stable_registration",
        stage3_registered,
        stage3_from_registration,
    )
    stage1_results = _nested(
        candidate_results, "stage_1_zero_base_screen", "results"
    )
    stage2_results = _nested(
        candidate_results, "stage_2_rank_target_sensitivity", "results"
    )
    stage3_results = _nested(
        candidate_results,
        "stage_3_liquidity_and_availability_hypotheses",
        "results",
    )
    _candidate_set(
        checks, "target.stage1_registered_candidates_reported", stage1_registered, stage1_results
    )
    _candidate_set(
        checks, "target.stage2_registered_candidates_reported", stage2_registered, stage2_results
    )
    _candidate_set(
        checks, "target.stage3_registered_candidates_reported", stage3_registered, stage3_results
    )
    for stage_key, registered in (
        ("stage_1_zero_base_screen", stage1_registered),
        ("stage_2_rank_target_sensitivity", stage2_registered),
        ("stage_3_liquidity_and_availability_hypotheses", stage3_registered),
    ):
        checks.equal(
            f"target.{stage_key}.candidate_count",
            _nested(candidate_results, stage_key, "candidate_count"),
            len(registered),
        )

    # Stage 4 is intentionally registered in a separate immutable artifact
    # because it fixes the roles selected after stage 3.  Bind that artifact
    # directly to the confirmation results so a missing K-candidate cannot be
    # hidden by a protocol/result pair that merely agrees with itself.
    stage4_payload = registration_payloads.get("stage_4", {})
    stage4_candidates = stage4_payload.get("confirmation_candidates", [])
    stage4_registered = [
        str(record["id"])
        for record in stage4_candidates
        if isinstance(record, Mapping) and "id" in record
    ] if isinstance(stage4_candidates, list) else []
    stage4_results = _nested(
        candidate_results,
        "stage_4_fixed_role_confirmation",
        "confirmation_results",
    )
    _candidate_set(
        checks,
        "target.stage4_registered_candidates_reported",
        stage4_registered,
        stage4_results,
    )
    checks.equal(
        "target.stage4_fixed_role_confirmation.candidate_count",
        _nested(
            candidate_results,
            "stage_4_fixed_role_confirmation",
            "candidate_count",
        ),
        len(stage4_registered),
    )

    fz_ids = [
        value
        for value in stage1_registered
        if re.fullmatch(r"feature__FZ[1-6]_.+", str(value))
    ]
    liquidity_ids = []
    for value in stage3_registered:
        match = re.match(r"L(\d+)_", str(value))
        if match and 2 <= int(match.group(1)) <= 14:
            liquidity_ids.append(str(value))
    checks.equal(
        "target.zero_base_new_single_feature_count",
        len(fz_ids),
        EXPECTED_HYPOTHESIS_COUNTS["zero_base_single_features"],
    )
    checks.equal(
        "target.liquidity_new_hypothesis_count",
        len(liquidity_ids),
        EXPECTED_HYPOTHESIS_COUNTS["liquidity_and_availability"],
    )
    checks.equal(
        "target.liquidity_new_hypothesis_numbers",
        sorted(int(re.match(r"L(\d+)_", value).group(1)) for value in liquidity_ids),
        list(range(2, 15)),
    )

    roles = result.get("final_fixed_shadow_roles", {})
    max_t = _nested(
        candidate_results,
        "stage_4_fixed_role_confirmation",
        "iteration3_max_t_adjustment",
        "candidates",
    )
    if not isinstance(max_t, Mapping):
        max_t = {}
    role_mapping = {
        "L4_mean_shadow": ("mean_return", "L4_add_flat20"),
        "L6_robustness_shadow": ("robustness", "L6_add_g5_all"),
    }
    shadow_summary: dict[str, Any] = {}
    for candidate_id, (role_key, research_id) in role_mapping.items():
        expected = EXPECTED_SHADOW_VALUES[candidate_id]
        role = roles.get(role_key, {}) if isinstance(roles, Mapping) else {}
        metrics = role.get("metrics", {}) if isinstance(role, Mapping) else {}
        checks.equal(
            f"target.{candidate_id}.candidate_id",
            role.get("candidate_id") if isinstance(role, Mapping) else None,
            candidate_id,
        )
        for cost in (20, 40, 60):
            value = _nested(metrics, "cost", str(cost), "net_mean_pct_at_cost")
            checks.close(
                f"target.{candidate_id}.net{cost}", value, expected[f"net{cost}"]
            )
        top20 = _nested(metrics, "winning_days_removed_net20", "20")
        checks.close(
            f"target.{candidate_id}.top20_removed_net20",
            top20,
            expected["top20_removed_net20"],
        )
        lower = _nested(
            max_t, research_id, "adjusted_one_sided_80pct_lower"
        )
        checks.close(
            f"target.{candidate_id}.max_t_lower",
            lower,
            expected["max_t_lower"],
        )
        checks.add(
            f"target.{candidate_id}.max_t_lower_negative",
            _negative_number(lower),
            {"actual": lower, "required": "< 0"},
        )
        manifest_role = _nested(manifest, "fixed_shadow_metrics", candidate_id)
        if isinstance(manifest_role, Mapping):
            checks.close(
                f"target.manifest.{candidate_id}.net20",
                manifest_role.get("net20_mean_pct"),
                expected["net20"],
            )
            checks.close(
                f"target.manifest.{candidate_id}.net40",
                manifest_role.get("net40_mean_pct"),
                expected["net40"],
            )
        else:
            checks.add(
                f"target.manifest.{candidate_id}.metrics_present", False, manifest_role
            )
        shadow_summary[candidate_id] = {
            "net20": _nested(metrics, "cost", "20", "net_mean_pct_at_cost"),
            "net40": _nested(metrics, "cost", "40", "net_mean_pct_at_cost"),
            "max_t_lower": lower,
        }

    checks.equal(
        "target.protocol_stop_rule",
        _nested(
            protocol, "stop_rule", "retrospective_target_and_model_tuning_stopped"
        ),
        True,
    )
    checks.equal(
        "target.result_stop_rule",
        _nested(
            result,
            "stop_decision",
            "retrospective_target_model_and_feature_tuning_stopped",
        ),
        True,
    )
    return {
        "fz_ids": fz_ids,
        "liquidity_ids": liquidity_ids,
        "shadow_roles": shadow_summary,
    }


def _feature_checks(
    checks: _Checks,
    *,
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    checks.equal(
        "feature.protocol_id",
        result.get("protocol_id"),
        protocol.get("protocol_id"),
    )
    checks.equal(
        "feature.protocol_production_promotion_false",
        _nested(protocol, "authority", "production_promotion_allowed"),
        False,
    )
    checks.equal(
        "feature.result_production_promotion_false",
        result.get("production_promotion_allowed"),
        False,
    )
    checks.equal(
        "feature.result_production_unchanged",
        _nested(result, "decision", "production_changed"),
        False,
    )
    checks.equal(
        "feature.manifest_production_promotion_false",
        manifest.get("production_promotion_allowed", _MISSING),
        False,
    )
    checks.equal(
        "feature.manifest_production_unchanged",
        manifest.get("production_changed", _MISSING),
        False,
    )
    checks.equal(
        "feature.manifest_decision_matches_result",
        [
            _nested(manifest, "decision", "generation_2_discovery_qualified_count"),
            _nested(manifest, "decision", "generation_3_alpha1_h07_transfer_passed"),
            _nested(manifest, "decision", "selected_feature_or_universe_modifier"),
            _nested(manifest, "decision", "production_changed"),
        ],
        [
            _nested(result, "decision", "generation_2_discovery_qualified_count"),
            _nested(result, "decision", "generation_3_h07_transfer_passed"),
            _nested(result, "decision", "feature_or_universe_modifier_selected"),
            _nested(result, "decision", "production_changed"),
        ],
    )
    checks.equal(
        "feature.registered_input_hash_claims",
        [
            _nested(manifest, "registered_inputs", "panel_manifest_sha256"),
            _nested(manifest, "registered_inputs", "panel_file_sha256"),
            _nested(manifest, "registered_inputs", "daily_prices_sha256"),
        ],
        [
            result.get("panel_manifest_sha256", _MISSING),
            result.get("panel_file_sha256", _MISSING),
            result.get("daily_prices_sha256", _MISSING),
        ],
    )
    checks.equal(
        "feature.safety_fix_metrics_unchanged",
        _nested(
            manifest,
            "reproducibility",
            "metrics_unchanged_after_order_and_alignment_fix",
        ),
        True,
    )
    checks.equal(
        "feature.no_scratch_path_dependency",
        _nested(manifest, "reproducibility", "scratch_path_dependency"),
        False,
    )

    registrations = protocol.get("original_registrations", {})
    result_registrations = result.get("original_registration_sha256", {})
    for registration_id, record in (
        registrations.items() if isinstance(registrations, Mapping) else []
    ):
        record = record if isinstance(record, Mapping) else {}
        _check_local_file_record(
            checks,
            root=root,
            name=f"feature.registration.{registration_id}.file_hash",
            path_value=record.get("path"),
            sha_value=record.get("sha256"),
        )
        checks.equal(
            f"feature.registration.{registration_id}.result_hash",
            result_registrations.get(registration_id)
            if isinstance(result_registrations, Mapping)
            else _MISSING,
            record.get("sha256"),
        )

    feature_ids = list(protocol.get("feature_groups", {}))
    universe_ids = list(protocol.get("universe_groups", {}))
    generation2 = result.get("generation_2_rank_ridge_alpha10", {})
    feature_results = _nested(generation2, "feature_candidates")
    universe_results = _nested(generation2, "universe_candidates")
    _candidate_set(
        checks, "feature.registered_groups_reported", feature_ids, feature_results
    )
    _candidate_set(
        checks, "feature.registered_universes_reported", universe_ids, universe_results
    )
    checks.equal(
        "feature.general_hypothesis_count",
        len(feature_ids),
        EXPECTED_HYPOTHESIS_COUNTS["general_feature_groups"],
    )
    checks.equal(
        "feature.universe_hypothesis_count",
        len(universe_ids),
        EXPECTED_HYPOTHESIS_COUNTS["universe_hypotheses"],
    )
    all_feature_max_t_negative = (
        isinstance(feature_results, Mapping)
        and bool(feature_results)
        and all(
            _negative_number(
                _nested(
                    value,
                    "discovery_max_t",
                    "adjusted_one_sided_80pct_lower_pct",
                )
            )
            for value in feature_results.values()
        )
    )
    checks.add(
        "feature.all_general_max_t_lowers_negative",
        all_feature_max_t_negative,
        {
            key: _nested(
                value, "discovery_max_t", "adjusted_one_sided_80pct_lower_pct"
            )
            for key, value in (
                feature_results.items() if isinstance(feature_results, Mapping) else []
            )
        },
    )
    checks.equal(
        "feature.discovery_feature_survivors_empty",
        generation2.get("discovery_feature_survivors")
        if isinstance(generation2, Mapping)
        else _MISSING,
        [],
    )
    checks.equal(
        "feature.discovery_universe_survivors_empty",
        generation2.get("discovery_universe_survivors")
        if isinstance(generation2, Mapping)
        else _MISSING,
        [],
    )
    checks.equal(
        "feature.no_modifier_selected",
        _nested(result, "decision", "feature_or_universe_modifier_selected"),
        None,
    )
    checks.equal(
        "feature.alpha1_transfer_rejected",
        _nested(result, "decision", "generation_3_h07_transfer_passed"),
        False,
    )
    checks.equal(
        "feature.alpha1_transfer_result_rejected",
        _nested(
            result,
            "generation_3_rank_ridge_alpha1_h07",
            "accepted_as_prospective_shadow_only",
        ),
        False,
    )
    checks.equal(
        "feature.alpha1_transfer_no_unregistered_combination",
        _nested(
            result,
            "generation_3_rank_ridge_alpha1_h07",
            "combination_tested",
        ),
        False,
    )
    checks.equal(
        "feature.outcome_mutation_invariant",
        result.get("outcome_mutation_rank_invariant"),
        True,
    )
    return {"feature_ids": feature_ids, "universe_ids": universe_ids}


def _tdnet_checks(
    checks: _Checks,
    *,
    protocol: Mapping[str, Any],
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    checks.equal(
        "tdnet.protocol_id",
        result.get("research_id"),
        protocol.get("protocol_id"),
    )
    checks.equal(
        "tdnet.protocol_production_promotion_false",
        _nested(protocol, "authority", "production_promotion_allowed"),
        False,
    )
    checks.equal(
        "tdnet.result_production_unchanged",
        _nested(result, "scope", "production_changed"),
        False,
    )
    checks.equal(
        "tdnet.verdict_production_change_none",
        _nested(result, "verdict", "production_change"),
        "none",
    )
    if "production_promotion_allowed" in manifest:
        checks.equal(
            "tdnet.manifest_production_promotion_false",
            manifest.get("production_promotion_allowed"),
            False,
        )

    registrations = protocol.get("historical_precommit_bindings", {})
    expected_registration_keys = {
        "generation_1_executable_contract",
        "generation_2_protocol",
        "generation_3_protocol",
        "generation_4_protocol",
    }
    checks.equal(
        "tdnet.stable_registration_set",
        sorted(registrations) if isinstance(registrations, Mapping) else [],
        sorted(expected_registration_keys),
    )
    result_registrations = _nested(
        result, "bindings", "historical_precommit_bindings"
    )
    for registration_id in sorted(expected_registration_keys):
        record = (
            registrations.get(registration_id, {})
            if isinstance(registrations, Mapping)
            else {}
        )
        result_record = (
            result_registrations.get(registration_id, {})
            if isinstance(result_registrations, Mapping)
            else {}
        )
        record = record if isinstance(record, Mapping) else {}
        result_record = result_record if isinstance(result_record, Mapping) else {}
        _check_local_file_record(
            checks,
            root=root,
            name=f"tdnet.registration.{registration_id}.file_hash",
            path_value=record.get("stable_path"),
            sha_value=record.get("sha256"),
        )
        checks.equal(
            f"tdnet.registration.{registration_id}.result_stable_path",
            result_record.get("stable_path", _MISSING),
            record.get("stable_path", _MISSING),
        )
        checks.equal(
            f"tdnet.registration.{registration_id}.result_hash",
            result_record.get("sha256", _MISSING),
            record.get("sha256", _MISSING),
        )

    feature_ids = list(protocol.get("feature_groups", {}))
    generation3 = result.get("generation_3_alpha10", {})
    generation4 = result.get("generation_4_alpha1", {})
    candidates3 = _nested(generation3, "candidates")
    _candidate_set(
        checks, "tdnet.registered_groups_reported", feature_ids, candidates3
    )
    expected_generation4 = {
        "H08_arrival_timing",
        "H10_stage_linear_runup",
        "H08_H10",
    }
    candidates4 = _nested(generation4, "candidates")
    checks.equal(
        "tdnet.generation4_registered_recipes_reported",
        sorted(candidates4) if isinstance(candidates4, Mapping) else [],
        sorted(expected_generation4),
    )
    checks.equal(
        "tdnet.feature_hypothesis_count",
        len(feature_ids),
        EXPECTED_HYPOTHESIS_COUNTS["tdnet_title_feature_groups"],
    )
    all_generation3_rejected = (
        isinstance(candidates3, Mapping)
        and bool(candidates3)
        and all(
            _nested(value, "decision", "qualified") is False
            for value in candidates3.values()
        )
    )
    all_generation4_rejected = (
        isinstance(candidates4, Mapping)
        and bool(candidates4)
        and all(
            _nested(value, "decision", "qualified") is False
            for value in candidates4.values()
        )
    )
    checks.add(
        "tdnet.all_generation3_candidates_rejected",
        all_generation3_rejected,
        {
            "qualified": generation3.get("qualified")
            if isinstance(generation3, Mapping)
            else None
        },
    )
    checks.add(
        "tdnet.all_generation4_candidates_rejected",
        all_generation4_rejected,
        {
            "qualified": generation4.get("qualified")
            if isinstance(generation4, Mapping)
            else None
        },
    )
    all_max_t_negative = (
        isinstance(candidates3, Mapping)
        and bool(candidates3)
        and all(
            _negative_number(
                _nested(
                    value,
                    "periods",
                    "full",
                    "max_t_adjusted",
                    "adjusted_one_sided_80pct_lower_pct",
                )
            )
            for value in candidates3.values()
        )
    )
    checks.add(
        "tdnet.all_generation3_max_t_lowers_negative",
        all_max_t_negative,
        {
            key: _nested(
                value,
                "periods",
                "full",
                "max_t_adjusted",
                "adjusted_one_sided_80pct_lower_pct",
            )
            for key, value in (
                candidates3.items() if isinstance(candidates3, Mapping) else []
            )
        },
    )
    coverage = result.get("data_coverage", {})
    checks.equal("tdnet.requested_sessions", coverage.get("requested_sessions"), 266)
    checks.equal("tdnet.complete_sessions", coverage.get("complete_sessions"), 187)
    checks.close(
        "tdnet.complete_fraction",
        coverage.get("complete_fraction"),
        187.0 / 266.0,
    )
    checks.add(
        "tdnet.coverage_below_qualification_floor",
        isinstance(coverage.get("complete_fraction"), (int, float))
        and coverage["complete_fraction"] < 0.98,
        {"actual": coverage.get("complete_fraction"), "required": "< 0.98"},
    )
    checks.equal(
        "tdnet.stop_rule_triggered",
        _nested(result, "verdict", "stop_rule_triggered"),
        True,
    )
    checks.equal(
        "tdnet.title_only_value_not_demonstrated",
        _nested(result, "verdict", "title_only_incremental_value"),
        "not_demonstrated",
    )
    return {"feature_ids": feature_ids, "coverage": coverage}


def audit_artifacts(root: str | Path = ROOT) -> dict[str, Any]:
    root_path = Path(root).resolve()
    checks = _Checks()
    paths = {
        "target_protocol": TARGET_PROTOCOL,
        "target_result": TARGET_RESULT,
        "target_manifest": TARGET_MANIFEST,
        "feature_protocol": FEATURE_PROTOCOL,
        "feature_result": FEATURE_RESULT,
        "feature_manifest": FEATURE_MANIFEST,
        "tdnet_protocol": TDNET_PROTOCOL,
        "tdnet_result": TDNET_RESULT,
        "tdnet_manifest": TDNET_MANIFEST,
    }
    payloads: dict[str, dict[str, Any]] = {}
    for name, relative in paths.items():
        path = root_path / relative
        try:
            payloads[name] = _read_json(path)
            checks.add(f"artifact.{name}.json", True, str(relative))
        except Exception as exc:  # audit must report a broken ledger as JSON
            payloads[name] = {}
            checks.add(
                f"artifact.{name}.json",
                False,
                {"path": str(relative), "error": f"{type(exc).__name__}: {exc}"},
            )

    target_protocol = payloads["target_protocol"]
    target_result = payloads["target_result"]
    target_manifest = payloads["target_manifest"]
    feature_protocol = payloads["feature_protocol"]
    feature_result = payloads["feature_result"]
    feature_manifest = payloads["feature_manifest"]
    tdnet_protocol = payloads["tdnet_protocol"]
    tdnet_result = payloads["tdnet_result"]
    tdnet_manifest = payloads["tdnet_manifest"]

    artifact_hashes: dict[str, dict[str, str]] = {}
    if target_protocol and target_result and target_manifest:
        artifact_hashes["target"] = _check_hash_chain(
            checks,
            name="target",
            root=root_path,
            protocol_path=TARGET_PROTOCOL,
            result_path=TARGET_RESULT,
            manifest_path=TARGET_MANIFEST,
            manifest=target_manifest,
            result_protocol_sha=target_result.get("protocol_sha256"),
        )
        _check_manifest_local_records(
            checks,
            root=root_path,
            prefix="target.manifest_file",
            manifest=target_manifest,
        )
    if feature_protocol and feature_result and feature_manifest:
        artifact_hashes["feature"] = _check_hash_chain(
            checks,
            name="feature",
            root=root_path,
            protocol_path=FEATURE_PROTOCOL,
            result_path=FEATURE_RESULT,
            manifest_path=FEATURE_MANIFEST,
            manifest=feature_manifest,
            result_protocol_sha=feature_result.get("protocol_sha256"),
        )
        _check_manifest_local_records(
            checks,
            root=root_path,
            prefix="feature.manifest_file",
            manifest=feature_manifest,
        )
    if tdnet_protocol and tdnet_result and tdnet_manifest:
        artifact_hashes["tdnet"] = _check_hash_chain(
            checks,
            name="tdnet",
            root=root_path,
            protocol_path=TDNET_PROTOCOL,
            result_path=TDNET_RESULT,
            manifest_path=TDNET_MANIFEST,
            manifest=tdnet_manifest,
            result_protocol_sha=_nested(tdnet_result, "bindings", "protocol_sha256"),
        )
        _check_manifest_local_records(
            checks,
            root=root_path,
            prefix="tdnet.manifest_file",
            manifest=tdnet_manifest,
        )

    target_summary = _target_checks(
        checks,
        protocol=target_protocol,
        result=target_result,
        manifest=target_manifest,
        root=root_path,
    )
    feature_summary = _feature_checks(
        checks,
        protocol=feature_protocol,
        result=feature_result,
        manifest=feature_manifest,
        root=root_path,
    )
    tdnet_summary = _tdnet_checks(
        checks,
        protocol=tdnet_protocol,
        result=tdnet_result,
        manifest=tdnet_manifest,
        root=root_path,
    )

    counts = {
        "zero_base_single_features": len(target_summary["fz_ids"]),
        "liquidity_and_availability": len(target_summary["liquidity_ids"]),
        "general_feature_groups": len(feature_summary["feature_ids"]),
        "tdnet_title_feature_groups": len(tdnet_summary["feature_ids"]),
        "universe_hypotheses": len(feature_summary["universe_ids"]),
    }
    counts["total_new_feature_hypotheses"] = sum(
        counts[key]
        for key in (
            "zero_base_single_features",
            "liquidity_and_availability",
            "general_feature_groups",
            "tdnet_title_feature_groups",
        )
    )
    for key, expected in EXPECTED_HYPOTHESIS_COUNTS.items():
        checks.equal(f"aggregate.count.{key}", counts.get(key), expected)

    failures = checks.failures
    report = {
        "schema_version": 1,
        "audit_id": "model_v08_zero_base_artifact_audit_20260722",
        "auditor": {
            "path": "research/audit_model_v08_zero_base.py",
            "sha256": sha256_file(Path(__file__)),
        },
        "passed": not failures,
        "production_promotion_allowed": False,
        "artifact_hashes": artifact_hashes,
        "hypothesis_counts": counts,
        "shadow_roles": target_summary["shadow_roles"],
        "tdnet_coverage": tdnet_summary["coverage"],
        "check_count": len(checks.checks),
        "failure_count": len(failures),
        "failures": failures,
        "checks": checks.checks,
    }
    return _json_value(report)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as stream:
        stream.write(encoded)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Explicit verification mode for documentation/CI compatibility; "
            "the auditor always performs the same complete verification."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research/model_v08_zero_base_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = audit_artifacts(args.root)
    output = args.output.resolve()
    _write_json(report, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": report["passed"],
                "check_count": report["check_count"],
                "failure_count": report["failure_count"],
                "failures": report["failures"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
