from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
from datetime import date
import csv

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research" / "model_v11_production_readiness.py"
RESULT = ROOT / "research" / "model_v11_production_readiness.json"
REPORT = ROOT / "research" / "model_v11_production_readiness_report.md"
PROTOCOL = ROOT / "research" / "model_v11_t02_oot_protocol.json"
PARSER_AUDIT = ROOT / "research" / "model_v04_parser_recovery_audit.json"
EVIDENCE_REGISTRY = (
    ROOT / "research" / "model_v11_data_evidence_registry.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def load_verifier():
    specification = importlib.util.spec_from_file_location(
        "model_v11_production_readiness", SCRIPT
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_authoritative_inputs_and_frozen_t02_are_hash_exact() -> None:
    result = load_result()

    assert result["network_accessed"] is False
    assert result["inputs"]["protocol_sha256"] == sha256(PROTOCOL)
    assert result["inputs"]["parser_audit_sha256"] == sha256(PARSER_AUDIT)
    assert result["inputs"]["evidence_registry_sha256"] == sha256(
        EVIDENCE_REGISTRY
    )
    assert result["candidate_freeze"]["all_present_and_exact"] is True
    artifacts = result["candidate_freeze"]["artifacts"]
    assert len(artifacts) == 4
    assert all(value["sha256_exact"] for value in artifacts.values())
    assert all(
        not str(value["path"]).startswith("/")
        for value in artifacts.values()
    )
    assert not str(result["jpx"]["parser"]["canonical_path"]).startswith("/")
    assert not str(result["tdnet"]["parser"]["path"]).startswith("/")


def test_frozen_oot_window_counts_audit_evidence_separately_from_reproducibility() -> None:
    result = load_result()
    jpx = result["jpx"]
    tdnet = result["tdnet"]

    assert result["window"]["warmup_session"] == "2025-08-01"
    assert result["window"]["score_start"] == "2025-08-04"
    assert result["window"]["score_end"] == "2026-03-31"
    assert jpx["expected_files_including_warmup"] == 160
    assert jpx["expected_score_sessions"] == 159
    assert jpx["audit_contract_exact"] is True
    assert jpx["audit_contract_checks"]["parsed_rows_sum"] == 622724
    assert jpx["audit_contract_checks"]["parsed_rows_top_level"] == 622724
    assert jpx["audit_contract_checks"]["rejected_rows_sum"] == 0
    assert jpx["audit_contract_checks"]["recovered_rows_sum"] == 63
    assert jpx["audit_contract_checks"]["all_per_file_sha256_well_formed"] is True
    assert jpx["audit_source_complete_files"] == 160
    assert jpx["audit_source_complete_score_sessions"] == 159
    assert jpx["raw_files_present"] == 0
    assert jpx["raw_files_hash_exact"] == 0
    assert jpx["currently_reproducible_source_complete_files"] == 0
    assert jpx["currently_reproducible_source_complete_score_sessions"] == 0
    assert jpx["provenance_complete_score_sessions"] == 0
    assert len(jpx["missing_raw_files"]) == 160
    assert jpx["sealed_manifest"]["exists"] is False
    assert jpx["sealed_manifest"]["sha256_exact"] is False

    assert tdnet["expected_calendar_pages"] == 243
    assert tdnet["expected_date_start"] == "2025-08-01"
    assert tdnet["expected_date_end"] == "2026-03-31"
    assert tdnet["pages_present"] == 0
    assert tdnet["metadata_sidecars_present"] == 0
    assert tdnet["parser_valid_pages"] == 0
    assert tdnet["source_complete_pages"] == 0
    assert tdnet["provenance_complete_pages"] == 0
    assert tdnet["source_complete_score_sessions"] == 0
    assert tdnet["provenance_complete_score_sessions"] == 0
    assert len(tdnet["missing_page_dates"]) == 243


def test_execution_data_and_both_promotion_gates_fail_closed() -> None:
    result = load_result()
    fields = result["execution_and_context_data"]["fields"]
    readiness = result["readiness"]

    assert set(fields) == {
        "exact_0858_indicative_price",
        "executable_order_simulation",
        "bid_ask_spread",
        "realized_spread",
        "slippage",
        "special_quote_0858",
        "open_time_or_delayed_open",
        "futures_0858",
        "pts_price_and_volume",
        "volume",
        "turnover",
        "minimum_trading_unit",
        "tick_size",
        "predicted_opening_turnover",
    }
    assert not any(fields.values())
    evidence = result["execution_and_context_data"]
    assert evidence["certification_scope"].startswith(
        "registry-bound execution-ledger validation"
    )
    assert evidence["registered_validated_artifact_count"] == 0
    assert evidence["unavailable_optional_research_context_fields"] == [
        "futures_0858",
        "pts_price_and_volume",
        "volume",
    ]
    assert readiness["bound_t02_no_tuning_artifacts_exact"] is True
    assert readiness["joint_provenance_complete_score_sessions"] == 0
    assert readiness["joint_provenance_complete_months"] == []
    assert readiness["full_159_session_window_reproducible"] is False
    assert (
        readiness["t02_exact_no_tuning_oot_statistical_gate_can_run"] is False
    )
    assert readiness["scope"] == "data_readiness_only_not_production_certification"
    assert readiness["can_certify_production"] is False
    assert readiness["execution_external_origin_authenticated"] is False
    assert readiness["genuinely_untouched_holdout_present"] is False
    assert readiness["fresh_paper_live_or_later_holdout_required"] is True
    assert readiness["statistical_input_data_ready"] is False
    assert readiness["execution_fields_ready"] is False
    assert readiness["execution_input_data_ready"] is False
    assert readiness["t02_execution_gate_can_run"] is False
    assert readiness["production_promotion_evaluable"] is False
    assert readiness["production_ready"] is False

    expected_blockers = {
        "JPX_RAW_FILES_MISSING_OR_MISMATCHED",
        "JPX_REGISTERED_MANIFEST_UNAVAILABLE",
        "JPX_PROVENANCE_BELOW_MINIMUM",
        "TDNET_DAILY_PAGES_MISSING",
        "TDNET_REGISTERED_MANIFEST_UNAVAILABLE",
        "TDNET_SOURCE_COMPLETENESS_MISSING",
        "TDNET_PROVENANCE_INCOMPLETE",
        "JOINT_COVERAGE_BELOW_MINIMUM",
        "JOINT_MONTHS_BELOW_MINIMUM",
        "FIELD_EXECUTION_PRICE_OR_FILL_MISSING",
        "FIELD_BID_ASK_SPREAD_MISSING",
        "FIELD_REALIZED_SPREAD_MISSING",
        "FIELD_SLIPPAGE_MISSING",
        "FIELD_SPECIAL_QUOTE_0858_MISSING",
        "FIELD_OPEN_TIME_OR_DELAYED_OPEN_MISSING",
        "FIELD_TURNOVER_MISSING",
        "FIELD_MINIMUM_TRADING_UNIT_MISSING",
        "FIELD_TICK_SIZE_MISSING",
        "FIELD_PREDICTED_OPENING_TURNOVER_MISSING",
        "CAPACITY_AT_INTENDED_CAPITAL_NOT_EVALUABLE",
    }
    actual_blockers = {value["id"] for value in result["blockers"]}
    assert actual_blockers == expected_blockers
    assert result["blocker_count"] == len(expected_blockers) == 20
    assert result["blocker_counts_by_gate"] == {
        "statistical_input": 9,
        "execution_input": 11,
    }
    assert all(
        value["gate"] in {"statistical_input", "execution_input"}
        for value in result["blockers"]
    )


def test_verifier_recomputes_the_core_result_without_network_code() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden_import = re.compile(
        r"^\s*(?:from|import)\s+(?:urllib\.request|requests|httpx|socket)\b",
        flags=re.MULTILINE,
    )
    assert forbidden_import.search(source) is None

    verifier = load_verifier()
    fresh = verifier.build_readiness(PROTOCOL, PARSER_AUDIT)
    stored = load_result()
    for section in ("window", "candidate_freeze", "jpx", "tdnet", "readiness"):
        assert fresh[section] == stored[section]
    assert fresh["blockers"] == stored["blockers"]

    report = REPORT.read_text(encoding="utf-8")
    assert "Raw PDFs currently present and hash-exact: 0/160" in report
    assert "Pages present: 0/243" in report
    assert "Exact no-tuning statistical gate can run: False" in report
    assert "Execution gate can run: False" in report
    assert "Can certify production: False" in report
    assert "No network access is used." in report


def test_header_only_candidate_can_never_satisfy_field_evidence(
    tmp_path: Path,
) -> None:
    fake = tmp_path / "preopen_execution.csv"
    fake.write_text(
        "observed_at,code,indicative_price,order_qty,fill_price\n",
        encoding="utf-8",
    )
    verifier = load_verifier()
    fresh = verifier.build_readiness(
        PROTOCOL,
        PARSER_AUDIT,
        roots=[tmp_path],
    )
    evidence = fresh["execution_and_context_data"]
    candidates = [
        value
        for value in evidence["header_candidates"]
        if value["path"] == str(fake.resolve())
    ]
    assert len(candidates) == 1
    assert candidates[0]["has_at_least_one_data_row"] is False
    assert candidates[0]["usable_for_gate"] is False
    assert not any(evidence["fields"].values())
    assert fresh["readiness"]["execution_input_data_ready"] is False


def test_inconsistent_jpx_audit_blocks_statistical_input(
    tmp_path: Path,
) -> None:
    audit = json.loads(PARSER_AUDIT.read_text(encoding="utf-8"))
    audit["new_parsed_rows"] += 1
    corrupted = tmp_path / "parser_audit_corrupted.json"
    corrupted.write_text(
        json.dumps(audit, ensure_ascii=False),
        encoding="utf-8",
    )
    verifier = load_verifier()
    with pytest.raises(
        ValueError,
        match="data-evidence registry does not bind this parser audit",
    ):
        verifier.build_readiness(
            PROTOCOL,
            corrupted,
            roots=[tmp_path],
        )


def test_modified_evidence_registry_is_rejected_before_scanning(
    tmp_path: Path,
) -> None:
    registry = json.loads(EVIDENCE_REGISTRY.read_text(encoding="utf-8"))
    registry["state"] = "tampered"
    corrupted = tmp_path / "registry_corrupted.json"
    corrupted.write_text(
        json.dumps(registry, ensure_ascii=False),
        encoding="utf-8",
    )
    verifier = load_verifier()
    with pytest.raises(
        ValueError,
        match="data-evidence registry SHA-256 is not frozen",
    ):
        verifier.build_readiness(
            PROTOCOL,
            PARSER_AUDIT,
            evidence_registry_path=corrupted,
            roots=[tmp_path],
        )


def test_tdnet_invalid_content_and_chronology_fail_closed(
    tmp_path: Path,
) -> None:
    verifier = load_verifier()
    page = tmp_path / "20250801.html"
    payload = b"<html>TDnet</html>"
    page.write_bytes(payload)
    parser_path = ROOT / "src" / "tse_session_ranker" / "data" / "tdnet.py"
    parser_sha256 = sha256(parser_path)
    metadata = {
        "schema_version": 1,
        "index_date": "2025-08-01",
        "bytes": len(payload),
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_url": (
            "https://contents.webapi.yanoshin.jp/"
            "contents/tdnet_date/20250801"
        ),
        "observed_at": "2025-08-02T10:00:00+09:00",
        "fetched_at": "2025-08-02T09:00:00+09:00",
        "parse_completed_at": "2025-08-02T08:00:00+09:00",
        "provenance": "network_request_start_recorded",
        "parser_version": "test",
        "parser_sha256": parser_sha256,
        "http_status": 200,
        "content_type": "text/html; charset=utf-8",
        "finalized": True,
    }
    page.with_suffix(".html.meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    status = verifier.validate_tdnet_page(
        page,
        date(2025, 8, 1),
        {"published_at", "code", "name", "title", "document_url", "index_date"},
        "tdnet_index_html_heading_and_card_v1",
        parser_sha256,
        "contents.webapi.yanoshin.jp",
        "/contents/tdnet_date/",
        "www.release.tdnet.info",
        None,
    )
    assert status["parser_valid"] is False
    assert status["metadata_integrity_checks"]["chronology"] is False
    assert status["source_complete"] is False
    assert status["provenance_complete"] is False


def test_structurally_valid_empty_tdnet_page_is_distinguished_from_spoof(
    tmp_path: Path,
) -> None:
    verifier = load_verifier()
    page = tmp_path / "20250801.html"
    payload = b"""
    <html><body>
    <h1>2025\xe5\xb9\xb408\xe6\x9c\x8801\xe6\x97\xa5\xe3\x81\xab\xe6\x8f\x90\xe5\x87\xba\xe3\x81\x95\xe3\x82\x8c\xe3\x81\x9f\xe9\x81\xa9\xe6\x99\x82\xe9\x96\x8b\xe7\xa4\xba\xe6\x83\x85\xe5\xa0\xb1</h1>
    <div class="table">
      <a href="/contents/tdnet_date/20250731">previous</a>
      <div class="row">
        <div><strong>\xe6\x99\x82\xe5\x88\xbb</strong></div>
        <div><strong>\xe9\x8a\x98\xe6\x9f\x84\xe5\x90\x8d</strong></div>
        <div><strong>\xe8\xa1\xa8\xe9\xa1\x8c</strong></div>
      </div>
      <a href="/contents/tdnet_date/20250802">next</a>
    </div>
    </body></html>
    """
    page.write_bytes(payload)
    parser_path = ROOT / "src" / "tse_session_ranker" / "data" / "tdnet.py"
    parser_sha256 = sha256(parser_path)
    metadata = {
        "schema_version": 1,
        "index_date": "2025-08-01",
        "bytes": len(payload),
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_url": (
            "https://contents.webapi.yanoshin.jp/"
            "contents/tdnet_date/20250801"
        ),
        "observed_at": "2025-08-02T08:00:00+09:00",
        "fetched_at": "2025-08-02T08:00:01+09:00",
        "parse_completed_at": "2025-08-02T08:00:02+09:00",
        "provenance": "network_request_start_recorded",
        "parser_version": "tdnet_index_html_heading_and_card_v1",
        "parser_sha256": parser_sha256,
        "http_status": 200,
        "content_type": "text/html; charset=utf-8",
        "finalized": True,
    }
    page.with_suffix(".html.meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    metadata_path = page.with_suffix(".html.meta.json")
    manifest_record = {
        "name": page.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
        "metadata_name": metadata_path.name,
        "metadata_sha256": sha256(metadata_path),
    }
    status = verifier.validate_tdnet_page(
        page,
        date(2025, 8, 1),
        {"published_at", "code", "name", "title", "document_url", "index_date"},
        "tdnet_index_html_heading_and_card_v1",
        parser_sha256,
        "contents.webapi.yanoshin.jp",
        "/contents/tdnet_date/",
        "www.release.tdnet.info",
        manifest_record,
    )
    assert status["row_count"] == 0
    assert status["empty_page_structure_valid"] is True
    assert status["parser_valid"] is True
    assert status["metadata_integrity_valid"] is True
    assert status["source_complete"] is True
    assert status["provenance_complete"] is True


def test_jpx_manifest_provenance_binds_source_size_and_parser() -> None:
    verifier = load_verifier()
    record = {
        "source_url": (
            "https://www.jpx.co.jp/markets/"
            "statistics-equities/daily/data/202508.pdf"
        ),
        "byte_count": 1234,
        "sha256": "a" * 64,
        "request_started_at": "2025-09-01T09:00:00+09:00",
        "receipt_completed_at": "2025-09-01T09:00:01+09:00",
        "parse_completed_at": "2025-09-01T09:00:02+09:00",
        "parser_version": "jpx_daily_text_v6_special_quote_marker",
        "parser_sha256": "b" * 64,
    }
    complete, missing = verifier.jpx_manifest_provenance_complete(
        record,
        "202508.pdf",
        "a" * 64,
        expected_parser_version="jpx_daily_text_v6_special_quote_marker",
        expected_parser_sha256="b" * 64,
        approved_source_host="www.jpx.co.jp",
        approved_source_path_prefix="/markets/statistics-equities/daily/",
        matching_file_sizes={1234},
    )
    assert complete is True
    assert missing == []

    record["source_url"] = "https://example.invalid/202508.pdf"
    record["byte_count"] = 999
    complete, missing = verifier.jpx_manifest_provenance_complete(
        record,
        "202508.pdf",
        "a" * 64,
        expected_parser_version="jpx_daily_text_v6_special_quote_marker",
        expected_parser_sha256="b" * 64,
        approved_source_host="www.jpx.co.jp",
        approved_source_path_prefix="/markets/statistics-equities/daily/",
        matching_file_sizes={1234},
    )
    assert complete is False
    assert "source_url_not_approved" in missing
    assert "byte_count_mismatch" in missing


def test_execution_ledger_has_a_positive_path_and_rejects_pit_mutation(
    tmp_path: Path,
) -> None:
    verifier = load_verifier()
    registry = json.loads(EVIDENCE_REGISTRY.read_text(encoding="utf-8"))
    registry["intended_capital_jpy"] = 1_000_000
    registry["approved_execution_source_ids"] = ["test-broker-export"]
    audit = json.loads(PARSER_AUDIT.read_text(encoding="utf-8"))
    allowed = {
        date.fromisoformat(
            f"{value['name'][4:8]}-{value['name'][8:10]}-{value['name'][10:12]}"
        )
        for value in audit["per_file"][1:41]
    }
    columns = registry["execution_evidence_contract"]["required_columns"]
    data_path = tmp_path / "execution_ledger.csv"
    policy_audit = {
        "valid": True,
        "sha256": "a" * 64,
        "bound_artifact_sha256": {
            "simulator_source": "b" * 64,
            "simulator_config": "c" * 64,
            "mutation_tests": "d" * 64,
        },
    }
    expected_decisions = {
        f"decision-{session.isoformat()}": {
            "session": session,
            "code": "1301",
            "decision_at_parsed": verifier.aware_datetime(
                f"{session.isoformat()}T08:58:30+09:00"
            ),
        }
        for session in sorted(allowed)
    }

    def write_rows(
        *,
        late_first_row: bool = False,
        early_first_row: bool = False,
        wrong_first_code: bool = False,
        first_outcome: str | None = None,
        wrong_first_tick: bool = False,
        wrong_first_spread: bool = False,
        duplicate_first_order: bool = False,
        high_actual_fill_first: bool = False,
        unfilled_special_first: bool = False,
        cancel_policy_after_request: bool = False,
        early_delayed_cancel: bool = False,
    ) -> None:
        with data_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for index, session in enumerate(sorted(allowed)):
                observed = "08:58:00"
                if late_first_row and index == 0:
                    observed = "09:00:00"
                if early_first_row and index == 0:
                    observed = "07:00:00"
                code = "9999" if wrong_first_code and index == 0 else "1301"
                row = {
                        "session_date": session.isoformat(),
                        "decision_id": f"decision-{session.isoformat()}",
                        "order_id": f"order-{session.isoformat()}",
                        "code": code,
                        "side": "buy",
                        "evidence_mode": "simulation",
                        "execution_outcome": "filled",
                        "preopen_observed_at": (
                            f"{session.isoformat()}T{observed}+09:00"
                        ),
                        "decision_at": (
                            f"{session.isoformat()}T08:58:30+09:00"
                        ),
                        "policy_action_at": (
                            f"{session.isoformat()}T08:58:30+09:00"
                        ),
                        "cancel_requested_at": "",
                        "cancel_ack_at": "",
                        "bid_price_0858": "999",
                        "ask_price_0858": "1001",
                        "indicative_price": "1000",
                        "simulated_fill_price": "1000",
                        "actual_fill_price": "1000",
                        "realized_spread_bps": (
                            "1" if wrong_first_spread and index == 0 else "0"
                        ),
                        "slippage_bps": "0",
                        "special_quote_0858": "false",
                        "actual_open_at": (
                            f"{session.isoformat()}T09:00:05+09:00"
                        ),
                        "turnover_jpy": "1000000000",
                        "minimum_trading_unit": "100",
                        "effective_tick_size": (
                            "3" if wrong_first_tick and index == 0 else "1"
                        ),
                        "predicted_opening_turnover_jpy": "100000000",
                        "order_quantity": "100",
                        "planned_order_notional_jpy": "100000",
                        "order_notional_jpy": "100000",
                    }
                if index == 0 and first_outcome is not None:
                    row.update(
                        {
                            "execution_outcome": first_outcome,
                            "simulated_fill_price": "0",
                            "actual_fill_price": "0",
                            "realized_spread_bps": "0",
                            "slippage_bps": "0",
                            "order_notional_jpy": "0",
                            "cancel_requested_at": (
                                f"{session.isoformat()}T08:59:00+09:00"
                            ),
                            "cancel_ack_at": (
                                f"{session.isoformat()}T08:59:01+09:00"
                            ),
                        }
                    )
                    if first_outcome == "cancelled_special_quote":
                        row["special_quote_0858"] = "buy"
                        if cancel_policy_after_request:
                            row["policy_action_at"] = (
                                f"{session.isoformat()}T09:00:00+09:00"
                            )
                    elif first_outcome == "cancelled_delayed_open":
                        row["actual_open_at"] = (
                            f"{session.isoformat()}T09:11:00+09:00"
                        )
                        row["cancel_requested_at"] = (
                            f"{session.isoformat()}T09:10:00+09:00"
                        )
                        row["cancel_ack_at"] = (
                            f"{session.isoformat()}T09:10:01+09:00"
                        )
                        row["policy_action_at"] = (
                            f"{session.isoformat()}T09:10:00+09:00"
                        )
                        if early_delayed_cancel:
                            row["policy_action_at"] = (
                                f"{session.isoformat()}T09:09:00+09:00"
                            )
                            row["cancel_requested_at"] = (
                                f"{session.isoformat()}T09:09:00+09:00"
                            )
                            row["cancel_ack_at"] = (
                                f"{session.isoformat()}T09:09:01+09:00"
                            )
                    elif first_outcome == "cancelled_liquidity":
                        row["turnover_jpy"] = "10000000"
                    elif first_outcome == "unfilled":
                        row["cancel_requested_at"] = ""
                        row["cancel_ack_at"] = ""
                        if unfilled_special_first:
                            row["special_quote_0858"] = "buy"
                if index == 0 and high_actual_fill_first:
                    row.update(
                        {
                            "simulated_fill_price": "2000",
                            "actual_fill_price": "2000",
                            "realized_spread_bps": "20000",
                            "slippage_bps": "10000",
                            "order_notional_jpy": "200000",
                        }
                    )
                writer.writerow(row)
                if index == 0 and duplicate_first_order:
                    extra = {
                        **row,
                        "decision_id": f"decision-extra-{session.isoformat()}",
                        "order_id": f"order-extra-{session.isoformat()}",
                        "code": "1302",
                    }
                    writer.writerow(extra)

    manifest_path = tmp_path / "execution_manifest.json"

    def registration() -> dict[str, str]:
        manifest = {
            "schema_version": 1,
            "validator_id": "execution_ledger_csv_v1",
            "source_id": "test-broker-export",
            "execution_policy_audit_sha256": "a" * 64,
            "simulator_source_sha256": "b" * 64,
            "simulator_config_sha256": "c" * 64,
            "exported_at": "2026-04-01T12:00:00+09:00",
            "request_started_at": "2026-04-01T11:59:00+09:00",
            "receipt_completed_at": "2026-04-01T11:59:01+09:00",
            "parse_completed_at": "2026-04-01T11:59:02+09:00",
            "data_path": str(data_path),
            "data_sha256": sha256(data_path),
            "data_bytes": data_path.stat().st_size,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        }

    write_rows()
    status = verifier.validate_execution_manifest(
        registration(),
        registry=registry,
        repo_root=ROOT,
        registered_files={data_path.resolve(), manifest_path.resolve()},
        allowed_sessions=allowed,
        expected_decisions=expected_decisions,
        policy_audit=policy_audit,
    )
    assert status["internal_integrity_valid"] is True, status["errors"]
    assert status["reference_data_recomputed_from_registered_sources"] is False
    assert status["external_origin_authenticated"] is False
    assert status["valid"] is False
    assert status["rows"] == 40
    assert status["sessions"] == 40
    assert status["capacity_valid"] is True

    write_rows(late_first_row=True)
    status = verifier.validate_execution_manifest(
        registration(),
        registry=registry,
        repo_root=ROOT,
        registered_files={data_path.resolve(), manifest_path.resolve()},
        allowed_sessions=allowed,
        expected_decisions=expected_decisions,
        policy_audit=policy_audit,
    )
    assert status["valid"] is False
    assert any("preopen_pit" in value for value in status["errors"])

    write_rows(early_first_row=True)
    status = verifier.validate_execution_manifest(
        registration(),
        registry=registry,
        repo_root=ROOT,
        registered_files={data_path.resolve(), manifest_path.resolve()},
        allowed_sessions=allowed,
        expected_decisions=expected_decisions,
        policy_audit=policy_audit,
    )
    assert status["valid"] is False
    assert any("preopen_pit" in value for value in status["errors"])

    write_rows(wrong_first_code=True)
    status = verifier.validate_execution_manifest(
        registration(),
        registry=registry,
        repo_root=ROOT,
        registered_files={data_path.resolve(), manifest_path.resolve()},
        allowed_sessions=allowed,
        expected_decisions=expected_decisions,
        policy_audit=policy_audit,
    )
    assert status["valid"] is False
    assert any(
        "decision_artifact_mismatch" in value for value in status["errors"]
    )

    registry["execution_evidence_contract"]["minimum_filled_rows"] = 39
    write_rows(first_outcome="cancelled_special_quote")
    status = verifier.validate_execution_manifest(
        registration(),
        registry=registry,
        repo_root=ROOT,
        registered_files={data_path.resolve(), manifest_path.resolve()},
        allowed_sessions=allowed,
        expected_decisions=expected_decisions,
        policy_audit=policy_audit,
    )
    assert status["internal_integrity_valid"] is True, status["errors"]
    assert status["valid"] is False
    assert status["filled_rows"] == 39

    write_rows(first_outcome="cancelled_delayed_open")
    status = verifier.validate_execution_manifest(
        registration(),
        registry=registry,
        repo_root=ROOT,
        registered_files={data_path.resolve(), manifest_path.resolve()},
        allowed_sessions=allowed,
        expected_decisions=expected_decisions,
        policy_audit=policy_audit,
    )
    assert status["internal_integrity_valid"] is True, status["errors"]
    assert status["valid"] is False

    write_rows(first_outcome="cancelled_liquidity")
    status = verifier.validate_execution_manifest(
        registration(),
        registry=registry,
        repo_root=ROOT,
        registered_files={data_path.resolve(), manifest_path.resolve()},
        allowed_sessions=allowed,
        expected_decisions=expected_decisions,
        policy_audit=policy_audit,
    )
    assert status["internal_integrity_valid"] is True, status["errors"]
    assert status["valid"] is False

    write_rows(
        first_outcome="cancelled_special_quote",
        cancel_policy_after_request=True,
    )
    status = verifier.validate_execution_manifest(
        registration(),
        registry=registry,
        repo_root=ROOT,
        registered_files={data_path.resolve(), manifest_path.resolve()},
        allowed_sessions=allowed,
        expected_decisions=expected_decisions,
        policy_audit=policy_audit,
    )
    assert status["valid"] is False
    assert any("cancel_timestamps" in value for value in status["errors"])

    write_rows(
        first_outcome="cancelled_delayed_open",
        early_delayed_cancel=True,
    )
    status = verifier.validate_execution_manifest(
        registration(),
        registry=registry,
        repo_root=ROOT,
        registered_files={data_path.resolve(), manifest_path.resolve()},
        allowed_sessions=allowed,
        expected_decisions=expected_decisions,
        policy_audit=policy_audit,
    )
    assert status["valid"] is False
    assert any("delayed_open_cancel" in value for value in status["errors"])

    write_rows(first_outcome="unfilled", unfilled_special_first=True)
    status = verifier.validate_execution_manifest(
        registration(),
        registry=registry,
        repo_root=ROOT,
        registered_files={data_path.resolve(), manifest_path.resolve()},
        allowed_sessions=allowed,
        expected_decisions=expected_decisions,
        policy_audit=policy_audit,
    )
    assert status["valid"] is False
    assert any("unfilled_policy" in value for value in status["errors"])

    registry["intended_capital_jpy"] = 150_000
    write_rows(high_actual_fill_first=True)
    status = verifier.validate_execution_manifest(
        registration(),
        registry=registry,
        repo_root=ROOT,
        registered_files={data_path.resolve(), manifest_path.resolve()},
        allowed_sessions=allowed,
        expected_decisions=expected_decisions,
        policy_audit=policy_audit,
    )
    assert status["valid"] is False
    assert status["capacity_valid"] is False

    write_rows(wrong_first_tick=True)
    status = verifier.validate_execution_manifest(
        registration(),
        registry=registry,
        repo_root=ROOT,
        registered_files={data_path.resolve(), manifest_path.resolve()},
        allowed_sessions=allowed,
        expected_decisions=expected_decisions,
        policy_audit=policy_audit,
    )
    assert status["valid"] is False
    assert any("tick_multiple" in value for value in status["errors"])

    write_rows(wrong_first_spread=True)
    status = verifier.validate_execution_manifest(
        registration(),
        registry=registry,
        repo_root=ROOT,
        registered_files={data_path.resolve(), manifest_path.resolve()},
        allowed_sessions=allowed,
        expected_decisions=expected_decisions,
        policy_audit=policy_audit,
    )
    assert status["valid"] is False
    assert any(
        "spread_or_slippage_recalculation" in value
        for value in status["errors"]
    )

    write_rows(duplicate_first_order=True)
    first_session = sorted(allowed)[0]
    aggregate_decisions = {
        **expected_decisions,
        f"decision-extra-{first_session.isoformat()}": {
            "session": first_session,
            "code": "1302",
            "decision_at_parsed": verifier.aware_datetime(
                f"{first_session.isoformat()}T08:58:30+09:00"
            ),
        },
    }
    registry["intended_capital_jpy"] = 150_000
    status = verifier.validate_execution_manifest(
        registration(),
        registry=registry,
        repo_root=ROOT,
        registered_files={data_path.resolve(), manifest_path.resolve()},
        allowed_sessions=allowed,
        expected_decisions=aggregate_decisions,
        policy_audit=policy_audit,
    )
    assert status["valid"] is False
    assert status["capacity_valid"] is False


def test_decision_artifact_rejects_duplicate_cash_id_and_nonfinite_score(
    tmp_path: Path,
) -> None:
    verifier = load_verifier()
    registry = json.loads(EVIDENCE_REGISTRY.read_text(encoding="utf-8"))
    sessions = {date(2025, 8, 4), date(2025, 8, 5)}
    path = tmp_path / "decisions.csv"
    columns = [
        "session_date",
        "decision_id",
        "action",
        "code",
        "decision_at",
        "candidate_id",
        "rank",
        "predicted_value",
    ]

    def write(*, cash_id: str = "cash-2", score: str = "0.25") -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "session_date": "2025-08-04",
                        "decision_id": "order-1",
                        "action": "order",
                        "code": "1301",
                        "decision_at": "2025-08-04T08:58:30+09:00",
                        "candidate_id": "v10_t02_char_value_event_top1",
                        "rank": "1",
                        "predicted_value": score,
                    },
                    {
                        "session_date": "2025-08-05",
                        "decision_id": cash_id,
                        "action": "cash",
                        "code": "",
                        "decision_at": "2025-08-05T08:58:30+09:00",
                        "candidate_id": "v10_t02_char_value_event_top1",
                        "rank": "0",
                        "predicted_value": "",
                    },
                ]
            )

    def load() -> tuple[dict[str, object], dict[str, object]]:
        registry["registered_t02_decision_artifact"] = {
            "path": str(path),
            "sha256": sha256(path),
            "schema_version": 1,
        }
        return verifier.load_registered_t02_decisions(
            registry=registry,
            repo_root=ROOT,
            registered_files={path.resolve()},
            allowed_sessions=sessions,
        )

    write()
    decisions, status = load()
    assert status["valid"] is True
    assert set(decisions) == {"order-1"}
    assert status["decision_id_sha256"]
    assert status["decision_rows_sha256"]

    write(cash_id="order-1")
    decisions, status = load()
    assert decisions == {}
    assert status["valid"] is False
    assert any("duplicate_or_empty_decision_id" in value for value in status["errors"])

    write(score="nan")
    decisions, status = load()
    assert decisions == {}
    assert status["valid"] is False
    assert any("predicted_value_nonfinite" in value for value in status["errors"])


def test_replay_and_policy_audits_are_hash_bound_and_fail_closed(
    tmp_path: Path,
) -> None:
    verifier = load_verifier()
    registry = json.loads(EVIDENCE_REGISTRY.read_text(encoding="utf-8"))
    registry["registered_t02_decision_artifact"]["sha256"] = "1" * 64
    registry["jpx"]["registered_acquisition_parse_manifest"]["sha256"] = "2" * 64
    registry["tdnet"]["registered_page_manifest"]["sha256"] = "3" * 64
    decision_status = {
        "valid": True,
        "sessions": 40,
        "order_decisions": 35,
        "decision_id_sha256": "4" * 64,
        "decision_rows_sha256": "5" * 64,
    }
    generator = tmp_path / "decision_generator.py"
    decision_tests = tmp_path / "test_decision_generator.py"
    generator.write_text("GENERATOR = 1\n", encoding="utf-8")
    decision_tests.write_text("MUTATIONS = 1\n", encoding="utf-8")
    replay_path = tmp_path / "decision_replay_audit.json"
    replay = {
        "schema_version": 1,
        "validator_id": "t02_decision_exact_replay_v1",
        "decision_artifact_sha256": "1" * 64,
        "protocol_sha256": registry["protocol"]["sha256"],
        "jpx_manifest_sha256": "2" * 64,
        "tdnet_manifest_sha256": "3" * 64,
        "decision_id_sha256": "4" * 64,
        "decision_rows_sha256": "5" * 64,
        "session_count": 40,
        "order_count": 35,
        "checks": {
            "source_complete_sessions_exact": True,
            "frozen_candidate_replayed": True,
            "orders_and_cash_exact": True,
            "scores_and_ties_exact": True,
            "mutation_rejects_code_substitution": True,
            "mutation_rejects_cash_substitution": True,
        },
        "bound_artifacts": [
            {
                "role": "decision_generator_source",
                "path": str(generator),
                "sha256": sha256(generator),
            },
            {
                "role": "mutation_tests",
                "path": str(decision_tests),
                "sha256": sha256(decision_tests),
            },
        ],
    }

    def write_replay() -> None:
        replay_path.write_text(json.dumps(replay), encoding="utf-8")
        registry["registered_t02_decision_replay_audit"] = {
            "path": str(replay_path),
            "sha256": sha256(replay_path),
            "schema_version": 1,
        }

    write_replay()
    status = verifier.validate_t02_decision_replay_audit(
        registry=registry,
        repo_root=ROOT,
        registered_files={
            generator.resolve(),
            decision_tests.resolve(),
            replay_path.resolve(),
        },
        decision_artifact=decision_status,
    )
    assert status["internal_integrity_valid"] is True
    assert status["external_origin_authenticated"] is False
    assert status["valid"] is False
    assert "external_attestation_not_verified" in status["errors"]

    replay["checks"]["mutation_rejects_cash_substitution"] = False
    write_replay()
    status = verifier.validate_t02_decision_replay_audit(
        registry=registry,
        repo_root=ROOT,
        registered_files={
            generator.resolve(),
            decision_tests.resolve(),
            replay_path.resolve(),
        },
        decision_artifact=decision_status,
    )
    assert status["valid"] is False
    assert "decision_replay_checks" in status["errors"]

    simulator = tmp_path / "simulator.py"
    config = tmp_path / "simulator.json"
    policy_tests = tmp_path / "test_simulator.py"
    simulator.write_text("SIMULATOR = 1\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    policy_tests.write_text("MUTATIONS = 1\n", encoding="utf-8")
    policy_path = tmp_path / "policy_audit.json"
    scenarios = {
        value: True
        for value in registry["execution_evidence_contract"][
            "required_policy_scenarios"
        ]
    }
    policy = {
        "schema_version": 1,
        "validator_id": "execution_policy_scenarios_v1",
        "scenario_results": scenarios,
        "bound_artifacts": [
            {
                "role": "simulator_source",
                "path": str(simulator),
                "sha256": sha256(simulator),
            },
            {
                "role": "simulator_config",
                "path": str(config),
                "sha256": sha256(config),
            },
            {
                "role": "mutation_tests",
                "path": str(policy_tests),
                "sha256": sha256(policy_tests),
            },
        ],
    }

    def write_policy() -> None:
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        registry["registered_execution_policy_audit"] = {
            "path": str(policy_path),
            "sha256": sha256(policy_path),
            "schema_version": 1,
        }

    registered = {
        simulator.resolve(),
        config.resolve(),
        policy_tests.resolve(),
        policy_path.resolve(),
    }
    write_policy()
    status = verifier.validate_execution_policy_audit(
        registry=registry,
        repo_root=ROOT,
        registered_files=registered,
    )
    assert status["internal_integrity_valid"] is True
    assert status["external_origin_authenticated"] is False
    assert status["valid"] is False
    assert "external_attestation_not_verified" in status["errors"]

    policy["scenario_results"]["liquidity_veto"] = False
    write_policy()
    status = verifier.validate_execution_policy_audit(
        registry=registry,
        repo_root=ROOT,
        registered_files=registered,
    )
    assert status["valid"] is False
    assert "policy_scenarios" in status["errors"]
