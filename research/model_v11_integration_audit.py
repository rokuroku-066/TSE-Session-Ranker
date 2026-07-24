#!/usr/bin/env python3
"""Fail-closed cross-family integration audit for the seven v11 research lanes.

This program does not compare portfolio returns across lanes.  It verifies the
frozen protocols, result bindings, independent audits, research authority, and
production flags, then applies a conservative conjunction rule for production
authorization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
FAMILIES = (
    "new_data",
    "analog",
    "distributional",
    "uplift",
    "graph",
    "shift",
    "calendar",
)
EXPECTED_CONCEPTUAL_COUNTS = {
    "new_data": 11,
    "analog": 10,
    "distributional": 8,
    "uplift": 8,
    "graph": 10,
    "shift": 8,
    "calendar": 12,
}
EXPECTED_CANDIDATE_POLICY_COUNTS = {
    "new_data": 6,
    "analog": 20,
    "distributional": 16,
    "uplift": 16,
    "graph": 20,
    "shift": 16,
    "calendar": 24,
}
EXPECTED_FAMILYWISE_GATE_UNITS = {
    "new_data": 6,
    "analog": 10,
    "distributional": 16,
    "uplift": 16,
    "graph": 10,
    "shift": 16,
    "calendar": 24,
}
CONTROL_BASE_IDS = {
    "new_data": {"L4_price_control"},
    "analog": {"L4_price_control", "T02_char_value_event_only"},
    "distributional": {
        "C00_DAILY_RANK_RIDGE_K1",
        "C00_DAILY_RANK_RIDGE_K2",
    },
    "uplift": set(),
    "graph": {"C00_daily_rank_ridge", "C01_L4_price_control"},
    "shift": {
        "C00_DAILY_RANK_RIDGE_K1",
        "C00_DAILY_RANK_RIDGE_K2",
    },
    "calendar": set(),
}
AUTHORITY_KEYS = {
    "new_data": "production_promotion_allowed",
    "analog": "production_promotion_allowed_from_this_run",
    "distributional": "production_promotion_allowed_from_this_panel",
    "uplift": "production_promotion_allowed_from_this_run",
    "graph": "production_promotion_allowed_from_this_run",
    "shift": "production_promotion_allowed_from_this_panel",
    "calendar": "production_promotion_allowed_from_this_run",
}
EXACT_DATA_BLOCKERS = [
    "frozen T02 OOT raw data missing",
    "08:58 execution data missing",
    "08:58 futures data missing",
    "08:58 orderbook data missing",
    "08:58 PTS data missing",
    "08:58 liquidity data missing",
]
NEW_DATA_ACQUISITION_GROUPS = [
    {
        "group": "structured_forecast_numeric",
        "hypothesis_ids": [
            "ND01_forecast_old_to_new",
            "ND02_profit_delta_to_market_cap",
        ],
    },
    {
        "group": "structured_buyback_and_capital",
        "hypothesis_ids": ["ND03_buyback_pressure"],
    },
    {
        "group": "immutable_PDF_table_bytes",
        "hypothesis_ids": ["ND06_pdf_table_shape_and_numeric_density"],
    },
    {
        "group": "document_body_text",
        "hypothesis_ids": ["ND07_body_negation_and_cancellation"],
    },
    {
        "group": "exact_OSE_futures_snapshots",
        "hypothesis_ids": ["ND08_ose_futures_0858"],
    },
    {
        "group": "preopen_TSE_orderbook",
        "hypothesis_ids": ["ND09_tse_auction_0858"],
    },
    {
        "group": "venue_specific_PTS",
        "hypothesis_ids": ["ND10_pts_night_and_day"],
    },
    {
        "group": "PIT_security_master_and_liquidity",
        "hypothesis_ids": ["ND11_pit_liquidity_and_unit_cost"],
    },
]
PANEL_SHA256 = (
    "6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def family_paths(family: str) -> dict[str, Path]:
    stem = RESEARCH / f"model_v11_{family}"
    return {
        "protocol": Path(f"{stem}_protocol.json"),
        "result": Path(f"{stem}_result.json"),
        "audit": Path(f"{stem}_audit.json"),
    }


def artifact_identity(document: dict[str, Any]) -> str | None:
    return document.get("protocol_id") or document.get("experiment_id")


def audit_protocol_hash(family: str, audit: dict[str, Any]) -> str | None:
    if family == "new_data":
        return audit.get("hashes", {}).get("protocol_sha256")
    return audit.get("protocol_sha256")


def result_protocol_hash(
    family: str, result: dict[str, Any]
) -> str | None:
    if family in {"distributional", "shift"}:
        return result.get("input", {}).get("protocol_sha256")
    return result.get("protocol_sha256")


def hypothesis_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    for key in (
        "registered_data_generation_hypotheses",
        "registered_hypotheses",
        "hypotheses",
    ):
        if key in protocol:
            return protocol[key]
    raise KeyError("protocol contains no registered hypothesis list")


def candidate_policy_count(
    family: str,
    protocol: dict[str, Any],
    result: dict[str, Any],
) -> int:
    if family == "new_data":
        return int(result["familywise"]["family_size"])
    if family == "analog":
        return len(hypothesis_rows(protocol)) * len(
            protocol["outcome_and_portfolio"]["portfolio_sizes"]
        )
    if family == "graph":
        return len(hypothesis_rows(protocol)) * len(
            protocol["outcome_and_portfolio"]["portfolio_sizes"]
        )
    return int(protocol["candidate_family_size"])


def familywise_gate_units(
    family: str,
    result: dict[str, Any],
    audit: dict[str, Any],
) -> int:
    if family == "new_data":
        return len(audit["familywise_recomputed"])
    if family in {"analog", "uplift", "graph", "calendar"}:
        return len(audit["candidate_gates"])
    if family in {"distributional", "shift"}:
        return len(result["retrospective_gates"])
    raise KeyError(f"unhandled family {family}")


def true_value(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "1.0"}


def normalized_capacity(value: str) -> int:
    return int(float(value))


def selection_sequence_inventory(family: str) -> dict[str, Any]:
    """Hash within-family daily economic action paths for every scored series.

    Capacity is represented through sleeve weight.  Empty sleeves disappear,
    so K1 and K2 that both stay fully in cash are economically identical.  We
    deliberately do not deduplicate across families because their eligibility
    and source authority are not interchangeable.
    """

    path = RESEARCH / f"model_v11_{family}_picks.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    variants: set[str] = set()
    candidates: set[str] = set()
    comparators: set[str] = set()
    actions: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            base_id = row.get("policy_id") or row["candidate_id"]
            raw_capacity = row.get("capacity") or row.get("top_k")
            capacity = normalized_capacity(raw_capacity)
            variant = (
                base_id
                if base_id.endswith(("_K1", "_K2"))
                else f"{base_id}__K{capacity}"
            )
            variants.add(variant)
            is_control = (
                base_id in CONTROL_BASE_IDS[family]
                or variant in CONTROL_BASE_IDS[family]
            )
            if is_control:
                comparators.add(variant)
                continue
            candidates.add(variant)
            if "executed" in row:
                executed = true_value(row["executed"])
            elif "trade_allowed" in row:
                executed = true_value(row["trade_allowed"]) and bool(
                    row.get("code")
                )
            elif "score_eligible" in row:
                executed = true_value(row["score_eligible"]) and bool(
                    row.get("code")
                )
            else:
                executed = bool(row.get("code"))
            if executed:
                actions[variant].append(
                    (row["date"], row.get("code", ""), 1.0 / capacity)
                )
    digest_groups: dict[str, list[str]] = defaultdict(list)
    for variant in sorted(candidates):
        serialized = "\n".join(
            f"{date}|{code}|{weight:.12g}"
            for date, code, weight in sorted(actions.get(variant, []))
        )
        digest_groups[
            hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        ].append(variant)
    duplicates = sorted(
        (sorted(group) for group in digest_groups.values() if len(group) > 1),
        key=lambda group: group[0],
    )
    return {
        "candidate_variants": len(candidates),
        "comparator_variants": len(comparators),
        "scored_series": len(variants),
        "unique_candidate_selection_sequences": len(digest_groups),
        "duplicate_economic_sequence_groups": duplicates,
        "deduplication_scope": "within_family_only",
        "sequence_definition": (
            "date/code portfolio weights after the frozen execute/cash rule; "
            "zero-weight cash sleeves omitted"
        ),
        "picks_path": str(path.relative_to(ROOT)),
        "picks_sha256": sha256_file(path),
    }


def panel_hash(family: str, result: dict[str, Any]) -> str:
    if family in {"new_data", "analog", "graph"}:
        return result["integrity"]["panel_sha256"]
    if family in {"distributional", "shift"}:
        return result["input"]["panel_sha256"]
    return result["input_hashes"]["panel_sha256"]


def result_integrity_passes(family: str, result: dict[str, Any]) -> bool:
    integrity = result["integrity"]
    if family == "new_data":
        return all(
            (
                integrity["strictly_prior_training_all_scored_folds"],
                integrity["future_publication_violations"] == 0,
                integrity["pre_prior_close_violations"] == 0,
                integrity["source_missing_never_encoded_as_no_event"],
                integrity["blocked_hypotheses_scored"] == 0,
                not integrity["URL_treated_as_document_body"],
                not integrity["title_vectorizer_used"],
                not integrity["hyperparameter_search_used"],
                integrity["score_sessions_unique"],
            )
        )
    if family == "analog":
        return all(
            (
                integrity["strictly_prior_training_all_folds"],
                integrity["future_publication_violations"] == 0,
                integrity["source_missing_never_encoded_no_event"],
                integrity["target_session_outcome_mutation"]["passes"],
            )
        )
    if family in {"distributional", "shift"}:
        return all(
            (
                integrity["all_training_strictly_prior"],
                integrity["candidate_feature_source_violations"] == 0,
                integrity["cash_slots_not_renormalised"],
                integrity["monthly_expanding_folds"] == 13,
                integrity["target_day_outcome_mutation_exact"],
            )
        )
    if family in {"uplift", "calendar"}:
        return all(
            (
                integrity["monthly_expanding_strict_prior"],
                integrity["future_source_violations"] == 0,
                integrity["target_session_outcome_mutation"]["passes"],
            )
        )
    if family == "graph":
        return all(
            (
                integrity["strictly_prior_all_folds"],
                integrity["price_source_violations"] == 0,
                integrity["target_session_outcome_mutation"]["passes"],
                not integrity[
                    "z17_same_date_peer_residual_target_reimplemented"
                ],
                not integrity["external_unavailable_sources_proxied"],
            )
        )
    raise KeyError(f"unhandled family {family}")


def independent_audit_passes(
    family: str,
    result: dict[str, Any],
    audit: dict[str, Any],
    audit_path: Path,
) -> bool:
    if family == "new_data":
        assertions = audit["assertions"]
        assertion_passes = all(
            assertions[key]
            for key in (
                "slot_level_gross_return_recomputed",
                "executed_slot_cost_recomputed",
                "equal_weight_divisor_recomputed",
                "cash_slots_zero_return_zero_cost",
                "all_reported_metrics_match",
                "tail_day_removal_matches",
                "top_code_cash_replacement_matches",
                "monthly_and_slice_metrics_match",
                "familywise_tests_match",
            )
        )
        structural_passes = all(
            row["slots_per_day_exact"]
            and row["rank_set_exact"]
            and not row["duplicate_code_within_day"]
            and row["execution_flag_exact"]
            for row in audit["structural_checks"]
        )
        return all(
            (
                audit["audit_status"] == "pass",
                audit["independent_of_runner_imports"],
                assertion_passes,
                structural_passes,
            )
        )
    if family == "analog":
        return all(
            (
                audit["integrity"]["passes"],
                audit["pnl_reproduction"]["passes"],
                audit["known_comparator_reproduction"]["passes"],
                audit["deterministic_reproduction"]["picks_byte_identical"],
                audit["deterministic_reproduction"]["metrics_identical"],
                audit["deterministic_reproduction"]["folds_identical"],
            )
        )
    if family in {"distributional", "shift"}:
        embedded = result["independent_audit"]
        return all(
            (
                audit["independent_pnl_exact"],
                audit["embedded_outcomes_exact"],
                audit["metric_difference_count"] == 0,
                embedded["exact"],
                embedded["embedded_outcomes_exact"],
                embedded["metric_difference_count"] == 0,
                embedded["audit_sha256"] == sha256_file(audit_path),
            )
        )
    if family in {"uplift", "calendar"}:
        return all(
            (
                audit["integrity"]["passes"],
                audit["pnl_reproduction"]["passes"],
                audit["deterministic_reproduction"]["passes"],
                audit["deterministic_reproduction"]["result_byte_identical"],
                audit["deterministic_reproduction"]["picks_byte_identical"],
            )
        )
    if family == "graph":
        return all(
            (
                audit["integrity"]["passes"],
                audit["pnl_reproduction"]["passes"],
                audit["known_control_reproduction"]["passes"],
                audit["deterministic_reproduction"][
                    "picks_byte_identical"
                ],
                audit["deterministic_reproduction"][
                    "metrics_and_folds_identical"
                ],
            )
        )
    raise KeyError(f"unhandled family {family}")


def retrospective_passers(
    family: str,
    result: dict[str, Any],
    audit: dict[str, Any],
) -> list[str]:
    if family == "new_data":
        # Its promotion gate is intentionally disabled.  All six candidate
        # policy/capacity tests also have non-positive uplift versus L4.
        positive = [
            name
            for name, row in audit["familywise_recomputed"].items()
            if row["delta_net40_mean_pct"] > 0.0
        ]
        return positive
    if family == "analog":
        return list(audit["decision"]["passing_candidate_ids"])
    if family in {"distributional", "shift"}:
        return list(result["decision"]["retrospective_gate_passers"])
    if family in {"uplift", "calendar"}:
        return list(
            audit["decision"]["passing_retrospective_candidates"]
        )
    if family == "graph":
        return list(audit["decision"]["passing_candidate_ids"])
    raise KeyError(f"unhandled family {family}")


def forward_finalist(
    family: str,
    result: dict[str, Any],
    audit: dict[str, Any],
) -> str | None:
    if family == "new_data":
        return None
    if family in {"analog", "graph"}:
        return audit["decision"]["selected_forward_shadow_candidate"]
    if family in {"distributional", "shift"}:
        return result["decision"]["forward_shadow_finalist"]
    return audit["decision"]["forward_shadow_finalist"]


def production_flags(
    family: str,
    protocol: dict[str, Any],
    result: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, bool]:
    flags = {
        "protocol_promotion_allowed": bool(
            protocol["authority"][AUTHORITY_KEYS[family]]
        )
    }
    if family == "new_data":
        flags.update(
            {
                "result_production_ready": bool(
                    result["conclusion_policy"]["production_ready"]
                ),
                "result_promotion_gate_enabled": bool(
                    result["conclusion_policy"]["promotion_gate_enabled"]
                ),
            }
        )
    elif family in {"analog", "graph"}:
        flags.update(
            {
                "audit_production_gate_passes": bool(
                    audit["decision"]["production_gate_passes"]
                ),
                "audit_production_model_changed": bool(
                    audit["decision"]["production_model_changed"]
                ),
                "audit_orders_allowed": bool(
                    audit["decision"]["orders_allowed"]
                ),
            }
        )
    elif family in {"distributional", "shift"}:
        flags.update(
            {
                "result_production_model_changed": bool(
                    result["production_model_changed"]
                ),
                "audit_production_model_changed": bool(
                    audit["production_model_changed"]
                ),
            }
        )
    elif family in {"uplift", "calendar"}:
        flags["audit_production_ready"] = bool(
            audit["decision"]["production_ready"]
        )
    else:
        raise KeyError(f"unhandled family {family}")
    return flags


def score_coverage(
    family: str,
    protocol: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    if family == "new_data":
        coverage = result["coverage"]
        return {
            "score_start": coverage["accepted_score_first"],
            "score_end": coverage["accepted_score_last"],
            "score_sessions": coverage["accepted_score_sessions"],
            "calendar_months": len(coverage["accepted_score_months"]),
            "denominator": "strict-source-complete accepted sessions",
        }
    if family == "analog":
        folds = result["folds"]
        coverage = result["coverage"]
        return {
            "score_start": folds[0]["score_session_first"],
            "score_end": folds[-1]["score_session_last"],
            "score_sessions": coverage["strict_complete_score_sessions"],
            "calendar_months": len(coverage["score_months"]),
            "denominator": "strict-source-complete event sessions",
        }
    if family in {"distributional", "shift"}:
        frozen = protocol["frozen_input"]
        return {
            "score_start": frozen["score_period"][0],
            "score_end": frozen["score_period"][1],
            "score_sessions": result["input"]["score_sessions"],
            "calendar_months": len(result["folds"]),
            "denominator": "all price-eligible scheduled sessions",
        }
    if family == "uplift":
        coverage = result["coverage"]
        return {
            "score_start": coverage["strict_score_first"],
            "score_end": coverage["strict_score_last"],
            "score_sessions": coverage["strict_score_sessions"],
            "calendar_months": len(coverage["score_months"]),
            "denominator": "strict-TDnet event-policy sessions",
        }
    if family == "graph":
        coverage = result["coverage"]
        score_months = coverage["score_months"]
        return {
            "score_start": coverage["score_start"],
            "score_end": coverage["score_end"],
            "score_sessions": coverage["score_sessions"],
            "calendar_months": (
                len(score_months)
                if isinstance(score_months, list)
                else int(score_months)
            ),
            "denominator": "all price-eligible scheduled sessions",
        }
    if family == "calendar":
        coverage = result["coverage"]
        score_period = protocol["point_in_time"]["score_period"]
        return {
            "score_start": score_period[0],
            "score_end": score_period[1],
            "score_sessions": coverage["score_sessions"],
            "calendar_months": len(result["folds"]),
            "denominator": (
                "all price sessions; TDnet policies separately fail closed"
            ),
        }
    raise KeyError(f"unhandled family {family}")


def decision_label(
    family: str,
    result: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    if family == "new_data":
        return "exploratory_only_promotion_disabled"
    if family in {"analog", "graph"}:
        return audit["decision"]["retrospective_decision"]
    if family in {"distributional", "shift"}:
        return (
            "reject_family"
            if not result["decision"]["retrospective_gate_passers"]
            else "freeze_forward_shadow_finalist"
        )
    return audit["decision"]["retrospective_decision"]


def canonical_decision_metadata(
    family: str,
    result: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    if family == "new_data":
        return {
            "canonical_source": "independent_audit_and_manifest",
            "pre_audit_result_status": None,
            "result_was_pending_independent_audit": False,
            "rule": (
                "The manifest binds the result and independent audit; its "
                "no-promotion conclusion is canonical."
            ),
        }
    if family in {"analog", "graph"}:
        status = result["decision"]["status"]
        return {
            "canonical_source": "independent_audit",
            "pre_audit_result_status": status,
            "result_was_pending_independent_audit": (
                status == "pending_independent_audit"
            ),
            "rule": (
                "The runner result intentionally stops at pending; the later "
                "independent audit is the canonical decision."
            ),
        }
    if family in {"uplift", "calendar"}:
        status = result["decision"]["status"]
        return {
            "canonical_source": "independent_audit",
            "pre_audit_result_status": status,
            "result_was_pending_independent_audit": (
                status == "pending_independent_audit"
            ),
            "rule": (
                "The runner result intentionally stops at pending; the later "
                "independent audit is the canonical decision."
            ),
        }
    return {
        "canonical_source": "finalized_result_and_independent_audit",
        "pre_audit_result_status": None,
        "result_was_pending_independent_audit": False,
        "rule": (
            "The finalized result embeds the independent audit and agrees with "
            "the standalone audit."
        ),
    }


def sidecar_matches(protocol_path: Path) -> bool | None:
    sidecar = protocol_path.with_suffix(".sha256")
    if not sidecar.exists():
        return None
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    return bool(fields) and fields[0] == sha256_file(protocol_path)


def build_family_record(family: str) -> dict[str, Any]:
    paths = family_paths(family)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{family} is missing required artifacts: {', '.join(missing)}"
        )
    protocol = read_json(paths["protocol"])
    result = read_json(paths["result"])
    audit = read_json(paths["audit"])
    protocol_id = protocol["protocol_id"]
    protocol_sha = sha256_file(paths["protocol"])
    result_id = artifact_identity(result)
    audit_id = artifact_identity(audit)
    conceptual = len(hypothesis_rows(protocol))
    policies = candidate_policy_count(family, protocol, result)
    gate_units = familywise_gate_units(family, result, audit)
    passers = retrospective_passers(family, result, audit)
    finalist = forward_finalist(family, result, audit)
    flags = production_flags(family, protocol, result, audit)
    selection = selection_sequence_inventory(family)
    executable_specs = 3 if family == "new_data" else conceptual
    manifest_checks: dict[str, bool] = {}
    extra_artifacts: dict[str, dict[str, Any]] = {}
    if family == "new_data":
        manifest_path = RESEARCH / "model_v11_new_data_manifest.json"
        manifest = read_json(manifest_path)
        manifest_checks = {
            "manifest_protocol_id_exact": (
                manifest["protocol_id"] == protocol_id
            ),
            "manifest_protocol_hash_exact": (
                manifest["files"][
                    "research/model_v11_new_data_protocol.json"
                ]["sha256"]
                == protocol_sha
            ),
            "manifest_result_hash_exact": (
                manifest["files"][
                    "research/model_v11_new_data_result.json"
                ]["sha256"]
                == sha256_file(paths["result"])
            ),
            "manifest_audit_hash_exact": (
                manifest["files"][
                    "research/model_v11_new_data_audit.json"
                ]["sha256"]
                == sha256_file(paths["audit"])
            ),
            "manifest_production_ready_false": not manifest["conclusion"][
                "production_ready"
            ],
            "manifest_candidate_survived_false": not manifest["conclusion"][
                "retrospective_candidate_survived"
            ],
        }
        extra_artifacts["manifest"] = {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": sha256_file(manifest_path),
        }
    checks = {
        "protocol_identity_matches_result_and_audit": (
            protocol_id == result_id == audit_id
        ),
        "result_protocol_hash_exact": (
            result_protocol_hash(family, result) == protocol_sha
        ),
        "audit_protocol_hash_exact": (
            audit_protocol_hash(family, audit) == protocol_sha
        ),
        "protocol_sidecar_matches_if_present": sidecar_matches(
            paths["protocol"]
        )
        is not False,
        "conceptual_hypothesis_count_exact": (
            conceptual == EXPECTED_CONCEPTUAL_COUNTS[family]
        ),
        "candidate_policy_count_exact": (
            policies == EXPECTED_CANDIDATE_POLICY_COUNTS[family]
        ),
        "familywise_gate_units_exact": (
            gate_units == EXPECTED_FAMILYWISE_GATE_UNITS[family]
        ),
        "selection_inventory_candidate_count_exact": (
            selection["candidate_variants"] == policies
        ),
        "selection_inventory_series_partition_exact": (
            selection["scored_series"]
            == selection["candidate_variants"]
            + selection["comparator_variants"]
        ),
        "shared_panel_hash_exact": (
            panel_hash(family, result) == PANEL_SHA256
        ),
        "result_integrity_passes": result_integrity_passes(family, result),
        "independent_audit_passes": independent_audit_passes(
            family, result, audit, paths["audit"]
        ),
        "retrospective_passers_empty": not passers,
        "forward_finalist_absent": finalist is None,
        "all_production_flags_false": not any(flags.values()),
        **manifest_checks,
    }
    if family in {"new_data", "analog"}:
        provenance = {
            "historical_cache_receipt_timestamp_available": False,
            "historical_archive_finality_proven": False,
            "PIT_claim_scope": (
                "publication timestamp plus fail-closed cache-page filter only"
            ),
        }
    elif family in {"uplift", "calendar"}:
        provenance = {
            "historical_cache_receipt_timestamp_available": False,
            "historical_archive_finality_proven": False,
            "PIT_claim_scope": (
                "audit verifies publication-time cutoff and source-completeness "
                "filter, not historical local receipt or archive finality"
            ),
        }
    else:
        provenance = {
            "historical_cache_receipt_timestamp_available": None,
            "historical_archive_finality_proven": None,
            "PIT_claim_scope": (
                "price PIT checks; any TDnet-dependent subpolicy retains the "
                "shared historical-cache caveat"
            ),
        }
    if family == "calendar":
        provenance.update(
            {
                "official_calendar_dataset_provenance_available": False,
                "calendar_feature_source": (
                    "deterministic functions of the frozen panel session index"
                ),
                "SQ_status": "registered proxy, not an official SQ calendar",
            }
        )
    artifacts = {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }
    artifacts.update(extra_artifacts)
    return {
        "family": family,
        "protocol_id": protocol_id,
        "conceptual_hypotheses": conceptual,
        "executable_specs": executable_specs,
        "candidate_policies": policies,
        "familywise_gate_units": gate_units,
        "selection_sequences": selection,
        "coverage": score_coverage(family, protocol, result),
        "shared_panel_sha256": panel_hash(family, result),
        "authority": {
            "retrospective_only": True,
            "production_promotion_allowed": flags[
                "protocol_promotion_allowed"
            ],
            "lane_local_preregistration_does_not_create_fresh_OOT": True,
        },
        "provenance_limitations": provenance,
        "canonical_decision": canonical_decision_metadata(
            family, result, audit
        ),
        "decision": decision_label(family, result, audit),
        "retrospective_passers": passers,
        "forward_shadow_finalist": finalist,
        "production_flags": flags,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "artifacts": artifacts,
    }


def t02_and_execution_gate() -> dict[str, Any]:
    protocol_path = RESEARCH / "model_v11_t02_oot_protocol.json"
    sidecar_path = RESEARCH / "model_v11_t02_oot_protocol.sha256"
    inventory_path = RESEARCH / "model_v11_new_data_inventory.json"
    if not all(
        path.exists()
        for path in (protocol_path, sidecar_path, inventory_path)
    ):
        raise FileNotFoundError("T02 OOT protocol or new-data inventory missing")
    protocol = read_json(protocol_path)
    inventory = read_json(inventory_path)
    protocol_sha = sha256_file(protocol_path)
    sidecar_fields = sidecar_path.read_text(encoding="utf-8").strip().split()
    bound_artifacts = {}
    for name, spec in protocol["bound_candidate_artifacts"].items():
        path = ROOT / spec["path"]
        bound_artifacts[name] = {
            "path": spec["path"],
            "present": path.exists(),
            "sha256_exact": path.exists()
            and sha256_file(path) == spec["sha256"],
        }
    source_end = max(
        inventory["periods"]["panel_bounds"][1],
        inventory["tdnet_index"]["last_cached_date"],
    )
    score_start = protocol["evaluation_window"]["score_start"]
    score_end = protocol["evaluation_window"]["score_end"]
    t02_result = RESEARCH / "model_v11_t02_oot_result.json"
    t02_audit = RESEARCH / "model_v11_t02_oot_audit.json"
    raw_complete = source_end >= score_end
    availability = inventory["availability_matrix"]
    execution_checks = {
        "exact_0858_execution_available": (
            protocol["execution_promotion_gate"]["status_before_data"]
            != "not_evaluable_from_OHLC_and_titles"
        ),
        "exact_0858_futures_available": availability[
            "ND08_ose_futures_0858"
        ]["local"],
        "exact_0858_orderbook_available": availability[
            "ND09_tse_auction_0858"
        ]["local"],
        "exact_0858_PTS_available": availability[
            "ND10_pts_night_and_day"
        ]["local"],
        "exact_0858_liquidity_available": availability[
            "ND11_pit_liquidity_and_unit_cost"
        ]["local"],
    }
    derived_blockers = []
    if not raw_complete:
        derived_blockers.append(EXACT_DATA_BLOCKERS[0])
    for available, blocker in zip(
        execution_checks.values(), EXACT_DATA_BLOCKERS[1:], strict=True
    ):
        if not available:
            derived_blockers.append(blocker)
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "untouched_holdout_claim": protocol["authority"][
            "untouched_holdout_claim"
        ],
        "authority_note": protocol["authority"]["reason_not_untouched"],
        "protocol_sidecar_exact": (
            bool(sidecar_fields) and sidecar_fields[0] == protocol_sha
        ),
        "bound_candidate_artifacts": bound_artifacts,
        "bound_candidate_artifacts_exact": all(
            row["present"] and row["sha256_exact"]
            for row in bound_artifacts.values()
        ),
        "evaluation_window": protocol["evaluation_window"],
        "available_raw_source_end": source_end,
        "historical_cache_receipt_metadata_available": inventory[
            "tdnet_index"
        ]["contains_local_receipt_timestamp"],
        "raw_window_reaches_score_start": source_end >= score_start,
        "t02_result_present": t02_result.exists(),
        "t02_audit_present": t02_audit.exists(),
        "fresh_OOT_raw_data_complete": raw_complete,
        "fresh_OOT_gate_passes": (
            raw_complete and t02_result.exists() and t02_audit.exists()
        ),
        "execution_checks": execution_checks,
        "execution_gate_passes": all(execution_checks.values()),
        "exact_data_blockers": derived_blockers,
        "exact_blocker_summary": (
            "frozen T02 OOT raw data missing; "
            "08:58 execution/futures/orderbook/PTS/liquidity missing"
        ),
    }


def build_report(audit: dict[str, Any]) -> str:
    totals = audit["counts"]
    production = audit["production_decision"]
    lines = [
        "# Model v11 — Cross-family integration audit",
        "",
        "## 結論",
        "",
        f"- 統合監査: **{audit['audit_status'].upper()}**",
        f"- 本番候補: **{production['production_candidate_count']}**",
        f"- 7系統の概念仮説: **{totals['conceptual_hypotheses']}**",
        f"- 実行可能spec: **{totals['executable_specs']}**",
        f"- capacity別候補variant: **{totals['candidate_policies']}**",
        "- family内で経済的に異なるselection sequence: "
        f"**{totals['economically_unique_within_family_selection_sequences']}**",
        f"- comparator variant: **{totals['comparator_variants']}**",
        f"- scored series総数: **{totals['total_scored_series']}**",
        "- family gate通過: "
        f"**{totals['candidate_variants_passing']}/{totals['candidate_policies']}**",
        f"- forward-shadow finalist: **{totals['forward_shadow_finalists']}**",
        "",
        "監査PASSは成果物の結合・PIT・独立再計算が整合したという意味であり、"
        "収益性や本番適格性のPASSではない。7系統はいずれもretrospectiveで、"
        "本番モデルを変更する権限を持たない。",
        "",
        "## 7系統",
        "",
        "| family | conceptual | variants | unique seq. | sessions | months | canonical decision | audit | passers | production |",
        "|---|---:|---:|---:|---:|---:|---|:---:|---:|:---:|",
    ]
    for family in audit["families"]:
        coverage = family["coverage"]
        lines.append(
            f"| `{family['family']}` | {family['conceptual_hypotheses']} | "
            f"{family['candidate_policies']} | "
            f"{family['selection_sequences']['unique_candidate_selection_sequences']} | "
            f"{coverage['score_sessions']} | {coverage['calendar_months']} | "
            f"`{family['canonical_decision']['canonical_source']}` | "
            f"{'PASS' if family['all_checks_pass'] else 'FAIL'} | "
            f"{len(family['retrospective_passers'])} | "
            f"{'false' if not any(family['production_flags'].values()) else 'true'} |"
        )
    lines.extend(
        [
            "",
            "variant数は候補mechanismとcapacityの組を数え、control/comparatorを除く。"
            "new_dataは実装可能な2仮説と固定combinationのtop1/top2を6 policyとし、"
            "利用不能の9仮説も概念仮説総数には含めた。",
            "Analog、graph、uplift、calendarのrunner resultは"
            "`pending_independent_audit`のpre-audit状態であり、最終判断は後発の"
            "独立auditをcanonicalとする。new_dataはauditとmanifest、"
            "distributional/shiftは独立auditを埋め込んだfinalized resultを使う。",
            "",
            "118 variantはすべて経済的に別ではない。family内の実行/cash actionを"
            "日付・銘柄・weightでhashすると112 sequenceとなる。"
            "`D07_*_K1/K2`、`U06_*_K1/K2`、`S06_*_K1/K2`、"
            "calendarの`C11/C12_*_K1/K2`はそれぞれ全cashで重複する。"
            "非互換family間はdeduplicateしない。",
            "",
            "## new_data blocked countの照合",
            "",
            "- 凍結resultのcanonical blocked hypothesisは **9**。"
            "`ND01`, `ND02`, `ND03`, `ND06`, `ND07`, `ND08`, `ND09`, "
            "`ND10`, `ND11`。",
            "- acquisition routeでまとめると **8** group。`ND01`と`ND02`が"
            "structured forecast numeric feedを共通の前提にするためである。",
            "- したがって「blocked hypothesis=8」へ書き換えない。8は取得group数、"
            "9は登録hypothesis数で、数える単位が違う。",
            "",
            "## 横断順位付けの禁止",
            "",
            "- **cross-family ranking permitted: false**",
            "- 107-session、88-session、266-sessionの評価窓が混在する。",
            "- source-completeness、候補universe、cash denominator、capacity、"
            "control、familywise手法が同一ではない。",
            "- したがって各系統のnet40点推定値を並べた「総合1位」は作らない。"
            "共通日だけの事後的再ランキングも未登録の別仮説になるため禁止する。",
            "",
            "## 多重性と証拠権限",
            "",
            f"- 全7系統が同じpanel SHA `{audit['cross_family_authority']['shared_panel_sha256']}` を使う。",
            f"- 観測済み候補policyは{totals['candidate_policies']}、"
            f"family-local gate単位は{totals['familywise_gate_units']}。",
            "- 各familyの補正はfamily内だけであり、7系統を見た後の横断選抜を"
            "補正していない。窓が非互換なので横断p値も算出しない。",
            "- familyごとの事前登録は機構の反証には有効だが、project-levelで"
            "既に見たpanelをfresh OOTへ戻さない。保守的な横断措置は候補を"
            "選ばないことである。",
            "",
            "## PIT・provenanceの限界",
            "",
            "- new_data/analogの履歴TDnet cacheにはlocal receipt timestampがなく、"
            "archive finalityは証明されていない。",
            "- uplift/calendarのPIT PASSはpublication timestamp cutoffと"
            "source-completeness filterの検証であり、履歴時点で同じarchive bytesを"
            "受領済みだったことの証明ではない。",
            "- calendar featureはfrozen panel session indexからの決定論的導出。"
            "official holiday/SQ calendar datasetのprovenanceはなく、SQはproxyである。",
            "",
            "## 未解消データblocker（exact）",
            "",
        ]
    )
    lines.extend(
        f"- `{blocker}`"
        for blocker in audit["fresh_OOT_and_execution"][
            "exact_data_blockers"
        ]
    )
    lines.extend(
        [
            "",
            "凍結T02のscore開始は2025-08-04だが、利用可能なpanel/TDnet rawは"
            f"{audit['fresh_OOT_and_execution']['available_raw_source_end']}まで。"
            "OOT protocol自身も`untouched_holdout_claim=false`で、"
            "T02 result/auditは存在しない。実始値・日次OHLCを08:58気配、先物、"
            "板、PTS、liquidityの代理にすることは禁止する。",
            "",
            "## 本番認可rule",
            "",
            "本番候補は、(1) 該当familyの全事前登録gate、(2) 横断選抜を含めて"
            "凍結したfresh OOT gate、(3) 08:58時点の実行可能性・spread/slippage・"
            "lot/capacity gate、(4) 独立監査、の全てがPASSした場合に限る。",
            "現在はその積集合が空なので、本番候補は0、ordersは不許可、"
            "production modelは変更しない。",
            "",
            "**結論は「現行freeze/dataの下で本番採用を支持する証拠がない」であり、"
            "「edgeが存在しない」ではない。** 未検証dataを取得した後も、既存結果の"
            "微調整ではなく、新しい事前登録とfresh periodが必要である。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=RESEARCH / "model_v11_integration_audit.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=RESEARCH / "model_v11_integration_report.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    family_records = [build_family_record(family) for family in FAMILIES]
    fresh = t02_and_execution_gate()
    conceptual_total = sum(
        family["conceptual_hypotheses"] for family in family_records
    )
    policy_total = sum(
        family["candidate_policies"] for family in family_records
    )
    executable_spec_total = sum(
        family["executable_specs"] for family in family_records
    )
    gate_unit_total = sum(
        family["familywise_gate_units"] for family in family_records
    )
    comparator_total = sum(
        family["selection_sequences"]["comparator_variants"]
        for family in family_records
    )
    scored_series_total = sum(
        family["selection_sequences"]["scored_series"]
        for family in family_records
    )
    unique_sequence_total = sum(
        family["selection_sequences"][
            "unique_candidate_selection_sequences"
        ]
        for family in family_records
    )
    passers_total = sum(
        len(family["retrospective_passers"]) for family in family_records
    )
    finalist_total = sum(
        family["forward_shadow_finalist"] is not None
        for family in family_records
    )
    new_data_result = read_json(
        RESEARCH / "model_v11_new_data_result.json"
    )
    blocked_hypothesis_ids = new_data_result["availability"][
        "blocked_hypotheses"
    ]
    grouped_blocked_ids = sorted(
        {
            hypothesis_id
            for group in NEW_DATA_ACQUISITION_GROUPS
            for hypothesis_id in group["hypothesis_ids"]
        }
    )
    all_family_checks = all(
        family["all_checks_pass"] for family in family_records
    )
    all_family_specific_gates = all(
        bool(family["retrospective_passers"]) for family in family_records
    )
    cross_family_authority = {
        "shared_panel_sha256": PANEL_SHA256,
        "shared_panel_track_count": sum(
            family["shared_panel_sha256"] == PANEL_SHA256
            for family in family_records
        ),
        "all_tracks_use_same_seen_panel": all(
            family["shared_panel_sha256"] == PANEL_SHA256
            for family in family_records
        ),
        "panel_status": "retrospective_project_seen",
        "family_local_preregistration_restores_fresh_holdout": False,
        "family_local_multiplicity_covers_cross_family_selection": False,
        "global_118_variant_multiplicity_correction_performed": False,
        "multiplicity_scope": "track_level_only",
        "cross_family_ranking_permitted": False,
        "cross_family_p_value_reported": False,
        "cross_family_winner_selected": False,
        "reason": (
            "Score windows, source-completeness filters, universes, cash "
            "denominators, controls, and multiplicity methods are incompatible. "
            "All seven tracks reuse one project-seen outcome panel."
        ),
        "authorized_action": (
            "falsify families and preserve zero candidates; any future selection "
            "requires a newly frozen policy and genuinely later data"
        ),
    }
    integration_integrity_checks = {
        "all_seven_family_artifact_checks_pass": all_family_checks,
        "conceptual_hypothesis_total_is_67": conceptual_total == 67,
        "candidate_policy_total_is_118": policy_total == 118,
        "executable_spec_total_is_59": executable_spec_total == 59,
        "unique_selection_sequence_total_is_112": (
            unique_sequence_total == 112
        ),
        "comparator_variant_total_is_14": comparator_total == 14,
        "scored_series_total_is_132": scored_series_total == 132,
        "new_data_canonical_blocked_hypotheses_are_9": (
            len(blocked_hypothesis_ids) == 9
        ),
        "new_data_acquisition_groups_are_8": (
            len(NEW_DATA_ACQUISITION_GROUPS) == 8
        ),
        "blocked_hypothesis_group_union_exact": (
            sorted(blocked_hypothesis_ids) == grouped_blocked_ids
        ),
        "all_tracks_bound_to_shared_panel": cross_family_authority[
            "all_tracks_use_same_seen_panel"
        ],
        "cross_family_ranking_is_prohibited": not cross_family_authority[
            "cross_family_ranking_permitted"
        ],
        "frozen_T02_protocol_sidecar_exact": fresh[
            "protocol_sidecar_exact"
        ],
        "frozen_T02_candidate_artifacts_exact": fresh[
            "bound_candidate_artifacts_exact"
        ],
        "exact_data_blockers_derived": (
            fresh["exact_data_blockers"] == EXACT_DATA_BLOCKERS
        ),
    }
    integration_integrity_passes = all(
        integration_integrity_checks.values()
    )
    production_checks = {
        "all_family_artifact_and_audit_checks_pass": (
            integration_integrity_passes
        ),
        "every_family_specific_retrospective_gate_passes": (
            all_family_specific_gates
        ),
        "cross_family_selection_has_fresh_evidence": False,
        "frozen_T02_fresh_OOT_gate_passes": fresh[
            "fresh_OOT_gate_passes"
        ],
        "execution_gate_passes": fresh["execution_gate_passes"],
        "independent_production_approval_recorded": False,
    }
    production_passes = all(production_checks.values())
    audit = {
        "schema_version": 1,
        "audit_id": "model_v11_cross_family_integration_20260723",
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "audit_status": (
            "pass" if integration_integrity_passes else "fail"
        ),
        "integrity": {
            "checks": integration_integrity_checks,
            "passes": integration_integrity_passes,
        },
        "families": family_records,
        "counts": {
            "tracks": len(family_records),
            "conceptual_hypotheses": conceptual_total,
            "executable_specs": executable_spec_total,
            "candidate_policies": policy_total,
            "economically_unique_within_family_selection_sequences": (
                unique_sequence_total
            ),
            "comparator_variants": comparator_total,
            "total_scored_series": scored_series_total,
            "familywise_gate_units": gate_unit_total,
            "candidate_variants_passing": passers_total,
            "forward_shadow_finalists": finalist_total,
        },
        "new_data_blocked_count_reconciliation": {
            "canonical_blocked_hypothesis_count": len(
                blocked_hypothesis_ids
            ),
            "canonical_blocked_hypothesis_ids": blocked_hypothesis_ids,
            "primary_acquisition_blocker_group_count": len(
                NEW_DATA_ACQUISITION_GROUPS
            ),
            "primary_acquisition_blocker_groups": (
                NEW_DATA_ACQUISITION_GROUPS
            ),
            "explanation": (
                "Nine is the canonical hypothesis count in the frozen result. "
                "Eight is only a source-acquisition grouping: ND01 and ND02 "
                "share the structured forecast-numeric feed as a prerequisite. "
                "The integration audit does not rewrite nine hypotheses as eight."
            ),
        },
        "selection_sequence_accounting": {
            "candidate_variants": policy_total,
            "unique_within_family_economic_sequences": (
                unique_sequence_total
            ),
            "not_all_variants_are_economically_distinct": (
                unique_sequence_total < policy_total
            ),
            "duplicate_groups_by_family": {
                family["family"]: family["selection_sequences"][
                    "duplicate_economic_sequence_groups"
                ]
                for family in family_records
                if family["selection_sequences"][
                    "duplicate_economic_sequence_groups"
                ]
            },
            "cross_family_deduplication_performed": False,
            "reason": (
                "Only within-family economic action paths are comparable; "
                "identical cash paths from incompatible families do not create "
                "a common statistical experiment."
            ),
        },
        "canonical_decision_precedence": {
            family["family"]: family["canonical_decision"]
            for family in family_records
        },
        "evidence_limitations": {
            "historical_TDnet_local_receipt_timestamp_available": False,
            "historical_TDnet_archive_finality_proven": False,
            "affected_families": [
                "new_data",
                "analog",
                "uplift",
                "calendar",
            ],
            "uplift_and_calendar_PIT_scope": (
                "publication timestamp cutoff plus fail-closed "
                "source-completeness filtering only"
            ),
            "calendar_official_calendar_provenance_available": False,
            "calendar_feature_source": (
                "deterministic functions of frozen panel session index"
            ),
            "calendar_SQ_status": (
                "proxy; no official SQ calendar dataset is bound"
            ),
            "OOT_untouched_holdout_claim": fresh[
                "untouched_holdout_claim"
            ],
            "interpretation": (
                "No production evidence under the current freeze and available "
                "data; this is not a claim that no edge exists."
            ),
        },
        "window_compatibility": {
            "cross_family_ranking_permitted": False,
            "point_estimates_directly_comparable": False,
            "common_window_post_hoc_ranking_permitted": False,
            "distinct_session_counts": sorted(
                {
                    family["coverage"]["score_sessions"]
                    for family in family_records
                }
            ),
            "prohibition": (
                "Do not rank or select families by return point estimates across "
                "incompatible windows. Do not retrofit a common window after "
                "viewing results."
            ),
        },
        "cross_family_authority": cross_family_authority,
        "fresh_OOT_and_execution": fresh,
        "production_decision": {
            "conjunctive_rule": (
                "A candidate requires every applicable family-specific gate, a "
                "fresh frozen OOT gate covering cross-family selection, every "
                "08:58 execution gate, and independent approval."
            ),
            "checks": production_checks,
            "all_required": True,
            "passes": production_passes,
            "production_candidate_count": 1 if production_passes else 0,
            "candidate_variants_considered": policy_total,
            "candidate_variants_passing_family_gates": passers_total,
            "production_model_changed": False,
            "orders_allowed": False,
            "decision": "NO_PRODUCTION_CANDIDATE",
            "exact_data_blockers": fresh["exact_data_blockers"],
            "additional_authority_blockers": [
                "no family-specific retrospective gate passer",
                "within-family multiplicity does not cover cross-family selection",
                "shared historical panel is not fresh OOT evidence",
            ],
        },
    }
    if conceptual_total != 67 or policy_total != 118:
        raise RuntimeError(
            "unexpected cross-family hypothesis/policy totals: "
            f"{conceptual_total}/{policy_total}"
        )
    if fresh["exact_data_blockers"] != EXACT_DATA_BLOCKERS:
        raise RuntimeError(
            "the exact fail-closed data blocker list changed unexpectedly"
        )
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.report_output.write_text(
        build_report(audit), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "audit": str(args.audit_output),
                "report": str(args.report_output),
                "audit_status": audit["audit_status"],
                "conceptual_hypotheses": conceptual_total,
                "candidate_policies": policy_total,
                "production_candidate_count": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
