from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
DECISION = RESEARCH / "model_v15_data_boundary.json"
PROBE = RESEARCH / "model_v15_tdnet_live_probe.json"
REPORT = RESEARCH / "model_v15_data_boundary_report.md"
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prior_evidence_hash_bindings_are_exact() -> None:
    decision = _json(DECISION)
    bindings = decision["prior_evidence_bindings"]
    assert set(bindings) == {
        "production_readiness",
        "data_evidence_registry",
        "new_data_inventory",
    }
    for binding in bindings.values():
        path = ROOT / binding["path"]
        assert path.is_file()
        assert binding["sha256"] == _sha256(path)


def test_v15_implementation_and_frozen_runtime_bindings_are_exact() -> None:
    decision = _json(DECISION)
    bindings = decision["bound_artifacts"]
    expected = {
        "live_probe_manifest",
        "collector",
        "api",
        "cli",
        "data_exports",
        "collector_tests",
        "decision_tests",
        "report",
        "production_features",
        "production_inference",
        "production_training",
        "frozen_t02_runtime",
        "pull_request_ci",
    }
    assert set(bindings) == expected
    for binding in bindings.values():
        path = ROOT / binding["path"]
        assert path.is_file()
        assert HASH_PATTERN.fullmatch(binding["sha256"])
        assert binding["sha256"] == _sha256(path)


def test_final_live_probe_is_fresh_stable_and_not_a_candidate_cohort() -> None:
    probe = _json(PROBE)
    assert probe["probe_type"] == "official_tdnet_current_collectibility"
    assert probe["candidate_records"] == []
    integrity = probe["integrity"]
    assert integrity["fresh_network_acquisition"] is True
    assert integrity["clock_sync_evidence_present"] is False
    assert integrity["model_scores_computed"] == 0
    assert integrity["market_outcomes_inspected_by_probe"] is False
    assert integrity["orders_allowed"] is False

    source = probe["source"]
    pages = source["page_receipts"]
    stability = source["stability_receipts"]
    documents = source["document_receipts"]
    assert len(pages) == 2
    assert len(stability) == len(pages)
    assert len(documents) == 9
    assert all(item["downloaded_in_run"] is True for item in pages)
    assert all(item["downloaded_in_run"] is True for item in stability)
    assert all(item["downloaded_in_run"] is True for item in documents)
    assert all(not Path(item["path"]).is_absolute() for item in pages)
    assert all(not Path(item["path"]).is_absolute() for item in stability)
    assert all(not Path(item["path"]).is_absolute() for item in documents)

    stability_by_url = {item["url"]: item for item in stability}
    for page in pages:
        repeated = stability_by_url[page["url"]]
        assert page["sha256"] == repeated["sha256"]
        assert page["bytes"] == repeated["bytes"]
        assert page["last_modified"] == repeated["last_modified"]
        assert (
            datetime.fromisoformat(repeated["requested_at"])
            > datetime.fromisoformat(page["received_at"])
        )
    first = next(item for item in pages if "I_list_001_" in item["url"])
    assert {item["last_modified"] for item in pages} == {
        source["last_modified_header_concordance"]
    }

    coverage = probe["coverage"]
    assert coverage == {
        "index_pages": 2,
        "index_disclosures": 165,
        "forecast_revision_documents": 9,
        "strict_extractions": 1,
        "strict_rejections": 8,
        "historical_validation_sessions": 0,
    }
    readiness = probe["readiness"]
    assert readiness["supported_table_shape_sample_verified"] is True
    assert readiness["source_complete"] is False
    assert readiness["same_day_pit_complete_sessions"] == 0
    assert readiness["historical_validation_ready"] is False
    assert readiness["performance_evaluation_allowed"] is False


def test_full_probe_preserves_all_pdf_receipts_and_fail_closed_reasons() -> None:
    probe = _json(PROBE)
    documents = probe["source"]["document_receipts"]
    assert {item["http_status"] for item in documents} == {200}
    assert {item["content_type"] for item in documents} == {
        "application/pdf"
    }
    assert len({item["url"] for item in documents}) == 9
    assert len({item["sha256"] for item in documents}) == 9
    assert all(HASH_PATTERN.fullmatch(item["sha256"]) for item in documents)

    rejected = probe["rejected_documents"]
    assert len(rejected) == 8
    assert all(item["reason"].startswith("TDnetMaterialError:") for item in rejected)
    rejected_urls = {
        item["document_receipt"]["url"] for item in rejected
    }
    assert len(rejected_urls) == 8
    assert rejected_urls < {item["url"] for item in documents}


def test_verified_extraction_recomputes_but_stays_quarantined() -> None:
    probe = _json(PROBE)
    extractions = probe["quarantined_extractions"]
    assert len(extractions) == 1
    extraction = extractions[0]
    assert extraction["disclosure"]["code"] == "6337"
    assert extraction["disclosure"]["published_at"] == (
        "2026-07-28T15:30:00+09:00"
    )
    assert extraction["document_receipt"]["sha256"] == (
        "b06e03d459f54da7fb84236cc0daa245de208fb53de583149528cfb8598cd772"
    )
    assert extraction["record_timestamp_cutoff_check"] is False
    assert extraction["candidate_eligible"] is False
    assert extraction["quarantine_reason"]

    revision = extraction["revision"]
    fields = (
        ("sales", "reported_sales_delta", "reported_sales_revision_pct"),
        (
            "operating_profit",
            "reported_operating_delta",
            "reported_operating_revision_pct",
        ),
        (
            "ordinary_profit",
            "reported_ordinary_delta",
            "reported_ordinary_revision_pct",
        ),
        (
            "net_profit",
            "reported_net_delta",
            "reported_net_revision_pct",
        ),
    )
    for field, delta_field, rate_field in fields:
        prior = revision[f"prior_{field}"]
        current = revision[f"current_{field}"]
        assert abs((current - prior) - revision[delta_field]) <= 0.01
        recomputed = 100.0 * (current / prior - 1.0)
        assert abs(recomputed - revision[rate_field]) <= 0.11
    assert extraction["material_strength"] == 0.3


def test_unavailable_arms_and_approaches_are_rejected_not_pending() -> None:
    decision = _json(DECISION)
    arms = {item["id"]: item for item in decision["arm_decisions"]}
    assert set(arms) == {
        "C0_FROZEN_T02",
        "A_QUANTITATIVE_MATERIAL",
        "B_0858_PRICE_ABSORPTION",
        "AB_MATERIAL_X_ABSORPTION",
    }
    assert arms["C0_FROZEN_T02"]["decision"] == "CONTROL_ONLY"
    assert arms["A_QUANTITATIVE_MATERIAL"]["decision"] == (
        "COLLECTOR_IMPLEMENTED_NO_MODEL_CANDIDATE"
    )
    for arm_id in (
        "B_0858_PRICE_ABSORPTION",
        "AB_MATERIAL_X_ABSORPTION",
    ):
        assert arms[arm_id]["decision"] == "REJECTED_INPUT_UNAVAILABLE"
    for arm in arms.values():
        assert arm["experiment_status"] in {"NOT_RUN", "NOT_RERUN"}
        assert arm["market_metrics"] is None
        assert arm["gate_status"] == "NOT_RUN"

    approaches = decision["other_approach_decisions"]
    assert len(approaches) == 7
    assert len({item["id"] for item in approaches}) == len(approaches)
    assert {
        item["decision"] for item in approaches
    } == {"REJECTED_INPUT_UNAVAILABLE"}
    assert all(item["missing_real_input"] for item in approaches)
    assert decision["integrity"]["pending_unavailable_candidates"] == 0


def test_public_history_sample_does_not_leave_a_model_candidate_open() -> None:
    decision = _json(DECISION)
    finding = decision["source_access_findings"][
        "listed_company_search_history"
    ]
    assert finding["state"] == (
        "SAMPLE_ACCESS_VERIFIED_NOT_VALIDATION_READY"
    )
    assert finding["sample_code"] == "63370"
    assert finding["public_retention_months"] == 121
    assert finding["pdf_links_observed"] is True
    assert finding["xbrl_links_observed"] is True
    assert finding["bulk_gap_free_export_verified"] is False
    assert finding["contemporaneous_pit_receipts_verified"] is False
    assert decision["family_decision"]["open_model_candidates"] == []


def test_no_performance_or_order_authority_is_implied() -> None:
    decision = _json(DECISION)
    integrity = decision["integrity"]
    assert integrity["synthetic_fixture_counted_as_source_evidence"] is False
    assert integrity["title_used_as_document_body"] is False
    assert integrity["proxy_values_used"] is False
    assert integrity["model_scores_computed"] == 0
    assert integrity["market_outcomes_inspected_by_this_protocol"] is False
    assert integrity["same_day_ohlc_predictors_used"] is False

    family = decision["family_decision"]
    assert family["open_model_candidates"] == []
    assert family["forward_shadow_candidates_added"] == []
    assert family["production_candidates"] == []
    assert family["future_retry_requires_new_protocol"] is True

    authority = decision["authority"]
    assert authority == {
        "performance_claimed": False,
        "production_model_changed": False,
        "production_promotion_allowed": False,
        "orders_allowed": False,
    }


def test_v15_has_no_result_or_pick_artifact() -> None:
    names = {
        path.name for path in RESEARCH.glob("model_v15*") if path.is_file()
    }
    assert names == {
        "model_v15_data_boundary.json",
        "model_v15_data_boundary_report.md",
        "model_v15_tdnet_live_probe.json",
    }
    assert all("result" not in name and "pick" not in name for name in names)


def test_report_states_evidence_limit_and_collect_or_reject_boundary() -> None:
    report = REPORT.read_text(encoding="utf-8")
    assert "is not a model" in report
    assert "`orders_allowed=false`" in report
    assert re.search(r"rejected\s+in\s+full", report)
    assert "not an independent" in report
    assert "121 months" in report
