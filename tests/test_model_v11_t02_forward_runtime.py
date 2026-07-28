from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tse_session_ranker.exceptions import LeakageError
from tse_session_ranker.t02_forward import (
    CANONICAL_JSON_CONTRACT,
    DecisionHashChainLedger,
    DecisionState,
    FinalTDnetArchive,
    FoldTrainingRow,
    LiveTDnetSnapshot,
    TDnetDisclosure,
    T02ForwardError,
    audit_final_archive_prefix,
    canonical_json_sha256,
    decision_id,
    eligible_monthly_training_rows,
    fit_t02_monthly_fold,
    live_snapshot_sha256,
    make_t02_decision,
    require_jst_timestamp,
    session_cutoff,
    validate_live_snapshot,
)


JST = ZoneInfo("Asia/Tokyo")
UTC = ZoneInfo("UTC")
SESSION = date(2026, 7, 28)
PROTOCOL_SHA = "1" * 64
SOURCE_SHA = "2" * 64
PARSER_SHA = "3" * 64
CODE_SHA = "4" * 64
UNIVERSE_SHA = "5" * 64
ELIGIBILITY_SHA = "6" * 64
ACTIVATION_PAYLOAD_SHA = "7" * 64
ACTIVATION_RECEIPT_SHA = "8" * 64
DEFAULT_BRANCH_TIP_SHA = "a" * 64


def at(
    day: date,
    hour: int,
    minute: int,
    second: int = 0,
) -> datetime:
    return datetime(
        day.year,
        day.month,
        day.day,
        hour,
        minute,
        second,
        tzinfo=JST,
    )


def disclosure(
    code: str = "1301",
    title: str = "業績予想の上方修正に関するお知らせ",
    *,
    published_at: datetime | None = None,
) -> TDnetDisclosure:
    return TDnetDisclosure(
        published_at=published_at or at(SESSION, 8, 0),
        code=code,
        title=title,
        document_url=f"https://example.invalid/{code}.pdf",
    )


def live_snapshot(
    disclosures: tuple[TDnetDisclosure, ...] = (),
    **overrides: object,
) -> LiveTDnetSnapshot:
    values: dict[str, object] = {
        "session_date": SESSION,
        "source_id": "fixture-live-feed",
        "snapshot_id": "snapshot-20260728-085850",
        "request_started_at": at(SESSION, 8, 58, 49),
        "source_received_at": at(SESSION, 8, 58, 50),
        "computed_at": at(SESSION, 8, 58, 51),
        "source_complete": True,
        "source_manifest_sha256": SOURCE_SHA,
        "raw_bytes_sha256": "9" * 64,
        "parser_sha256": PARSER_SHA,
        "source_watermark": "sequence:20260728:085850:42",
        "disclosures": disclosures,
    }
    values.update(overrides)
    return LiveTDnetSnapshot(**values)  # type: ignore[arg-type]


def training_rows(
    *,
    include_target_month: bool = False,
) -> tuple[FoldTrainingRow, ...]:
    rows: list[FoldTrainingRow] = []
    first = date(2026, 6, 2)
    for day_offset in range(20):
        session = first + timedelta(days=day_offset)
        for slot in range(5):
            positive = (day_offset + slot) % 2 == 0
            rows.append(
                FoldTrainingRow(
                    session_date=session,
                    code=f"{2000 + day_offset * 5 + slot:04d}",
                    bundle_text=(
                        "業績予想 上方修正 増益"
                        if positive
                        else "業績予想 下方修正 減益"
                    )
                    + f" 会社{slot}",
                    open_to_close_return_pct=2.0 if positive else -1.5,
                    source_complete=True,
                    price_training_eligible=True,
                    outcome_available_at=at(
                        session + timedelta(days=1),
                        16,
                        0,
                    ),
                )
            )
    if include_target_month:
        rows.append(
            FoldTrainingRow(
                session_date=date(2026, 7, 1),
                code="9999",
                bundle_text="未来の対象月タイトル",
                open_to_close_return_pct=99.0,
                source_complete=True,
                price_training_eligible=True,
                outcome_available_at=at(date(2026, 7, 1), 16, 0),
            )
        )
    return tuple(rows)


def fitted_fold(rows: tuple[FoldTrainingRow, ...] | None = None):
    return fit_t02_monthly_fold(
        rows or training_rows(),
        score_session=SESSION,
        fit_started_at=at(SESSION, 7, 30),
        fit_completed_at=at(SESSION, 7, 31),
        source_manifest_sha256=SOURCE_SHA,
        parser_sha256=PARSER_SHA,
        scoring_code_sha256=CODE_SHA,
        universe_config_sha256=UNIVERSE_SHA,
        eligibility_code_sha256=ELIGIBILITY_SHA,
    )


def make(
    *,
    snapshot: LiveTDnetSnapshot | None,
    fold=None,
    session: date = SESSION,
    price_eligible_codes: tuple[str, ...] = (),
    default_branch_tip_sha: str = DEFAULT_BRANCH_TIP_SHA,
):
    return make_t02_decision(
        protocol_sha256=PROTOCOL_SHA,
        activation_payload_sha256=ACTIVATION_PAYLOAD_SHA,
        activation_receipt_sha256=ACTIVATION_RECEIPT_SHA,
        default_branch_tip_sha_at_decision=default_branch_tip_sha,
        session_date=session,
        candidate_generation_started_at=at(session, 8, 58, 52),
        decision_at=at(session, 8, 58, 55),
        snapshot=snapshot,
        fold=fold,
        price_eligible_codes=price_eligible_codes,
    )


def test_strict_jst_cutoff_rejects_naive_utc_and_post_cutoff() -> None:
    assert session_cutoff(SESSION).isoformat() == "2026-07-28T08:58:59+09:00"
    with pytest.raises(T02ForwardError, match="timezone-aware JST"):
        require_jst_timestamp(
            datetime(2026, 7, 28, 8, 58, 50),
            "observed_at",
        )
    with pytest.raises(T02ForwardError, match="JST UTC"):
        require_jst_timestamp(
            at(SESSION, 8, 58, 50).astimezone(UTC),
            "observed_at",
        )
    with pytest.raises(LeakageError):
        require_jst_timestamp(
            at(SESSION, 8, 59),
            "observed_at",
            session_date=SESSION,
            at_or_before_cutoff=True,
        )


def test_canonical_hash_excludes_only_explicit_self_field() -> None:
    payload = {
        "value": 1,
        "source_manifest_sha256": SOURCE_SHA,
        "fold_manifest_sha256": "old",
    }
    first = canonical_json_sha256(
        payload,
        exclude_fields={"fold_manifest_sha256"},
    )
    payload["fold_manifest_sha256"] = "new"
    assert (
        canonical_json_sha256(
            payload,
            exclude_fields={"fold_manifest_sha256"},
        )
        == first
    )
    payload["source_manifest_sha256"] = "9" * 64
    assert (
        canonical_json_sha256(
            payload,
            exclude_fields={"fold_manifest_sha256"},
        )
        != first
    )


def test_live_snapshot_rejects_future_title_and_post_cutoff_receipt() -> None:
    future = live_snapshot(
        (
            disclosure(
                published_at=at(SESSION, 9, 0),
            ),
        )
    )
    with pytest.raises(LeakageError):
        validate_live_snapshot(future)

    late = live_snapshot(
        source_received_at=at(SESSION, 8, 59),
        computed_at=at(SESSION, 8, 59, 1),
    )
    with pytest.raises(LeakageError):
        validate_live_snapshot(late)
    assert make(snapshot=late, fold=fitted_fold())["decision"] == (
        DecisionState.FAIL_CLOSED_SOURCE.value
    )


def test_live_snapshot_provenance_is_ordered_and_hash_bound() -> None:
    snapshot = live_snapshot((disclosure(),))
    original_sha = live_snapshot_sha256(snapshot)
    assert live_snapshot_sha256(
        replace(snapshot, raw_bytes_sha256="b" * 64)
    ) != original_sha
    assert live_snapshot_sha256(
        replace(snapshot, parser_sha256="c" * 64)
    ) != original_sha
    assert live_snapshot_sha256(
        replace(snapshot, source_watermark="sequence:43")
    ) != original_sha

    invalid_order = replace(
        snapshot,
        request_started_at=at(SESSION, 8, 58, 51),
    )
    with pytest.raises(T02ForwardError, match="request_started_at"):
        validate_live_snapshot(invalid_order)
    with pytest.raises(T02ForwardError, match="source_watermark"):
        validate_live_snapshot(replace(snapshot, source_watermark=""))
    with pytest.raises(T02ForwardError, match="parser_sha256"):
        validate_live_snapshot(replace(snapshot, parser_sha256="not-a-hash"))


def test_final_archive_prefix_matches_and_ignores_post_cutoff_titles() -> None:
    pre = disclosure()
    post = disclosure(
        code="1302",
        title="大引け後の開示",
        published_at=at(SESSION, 15, 30),
    )
    snapshot = live_snapshot((pre,))
    archive = FinalTDnetArchive(
        session_date=SESSION,
        source_id=snapshot.source_id,
        archive_id="final-20260729",
        request_started_at=at(SESSION + timedelta(days=1), 0, 0),
        source_received_at=at(SESSION + timedelta(days=1), 0, 1),
        computed_at=at(SESSION + timedelta(days=1), 0, 2),
        finalized=True,
        raw_bytes_sha256="a" * 64,
        parser_sha256=PARSER_SHA,
        source_watermark="final:20260728",
        disclosures=(pre, post),
    )
    audit = audit_final_archive_prefix(snapshot, archive)
    assert audit.matched is True
    assert audit.live_prefix_sha256 == audit.final_prefix_sha256


def test_final_archive_prefix_detects_missing_and_mutated_title() -> None:
    original = disclosure()
    snapshot = live_snapshot((original,))
    extra = disclosure(
        code="1302",
        title="08時台に存在した別の開示",
        published_at=at(SESSION, 8, 30),
    )
    archive = FinalTDnetArchive(
        session_date=SESSION,
        source_id=snapshot.source_id,
        archive_id="final-missing",
        request_started_at=at(SESSION + timedelta(days=1), 0, 0),
        source_received_at=at(SESSION + timedelta(days=1), 0, 1),
        computed_at=at(SESSION + timedelta(days=1), 0, 2),
        finalized=True,
        raw_bytes_sha256="a" * 64,
        parser_sha256=PARSER_SHA,
        source_watermark="final:20260728",
        disclosures=tuple(
            sorted(
                (original, extra),
                key=lambda item: (
                    item.published_at,
                    item.code,
                    item.title,
                    item.document_url,
                ),
            )
        ),
    )
    audit = audit_final_archive_prefix(snapshot, archive)
    assert audit.matched is False
    assert len(audit.missing_from_live) == 1

    mutated = replace(original, title="後日改変されたタイトル")
    mutated_archive = replace(archive, disclosures=(mutated,))
    audit = audit_final_archive_prefix(snapshot, mutated_archive)
    assert audit.matched is False
    assert len(audit.missing_from_live) == 1
    assert len(audit.unexpected_in_live) == 1


def test_monthly_eligibility_excludes_target_month_and_unknown_outcome() -> None:
    base = training_rows(include_target_month=True)
    unknown = replace(
        base[0],
        code="8888",
        outcome_available_at=at(SESSION, 7, 31),
    )
    price_ineligible = replace(
        base[1],
        code="7777",
        price_training_eligible=False,
    )
    eligible = eligible_monthly_training_rows(
        (*base, unknown, price_ineligible),
        score_session=SESSION,
        fit_started_at=at(SESSION, 7, 30),
    )
    assert len(eligible) == 100
    assert all(row.session_date.month == 6 for row in eligible)
    assert all(
        row.code not in {"7777", "8888", "9999"} for row in eligible
    )


def test_target_month_mutation_cannot_change_fold_manifest() -> None:
    first_rows = training_rows(include_target_month=True)
    second_rows = tuple(
        replace(
            row,
            bundle_text="対象月を書き換え",
            open_to_close_return_pct=-999.0,
        )
        if row.code == "9999"
        else row
        for row in first_rows
    )
    first = fitted_fold(first_rows)
    second = fitted_fold(second_rows)
    assert first.manifest == second.manifest
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.manifest["state"] == "pre_score"
    assert first.manifest["training_bundle_count"] == 100


def test_fold_and_decision_are_deterministic() -> None:
    first = fitted_fold()
    second = fitted_fold()
    assert first.manifest == second.manifest
    snapshot = live_snapshot((disclosure(),))
    assert make(snapshot=snapshot, fold=first, price_eligible_codes=("1301",)) == make(
        snapshot=snapshot,
        fold=second,
        price_eligible_codes=("1301",),
    )


def test_decision_id_hashes_the_project_canonical_json_object() -> None:
    expected = canonical_json_sha256(
        {
            "activation_receipt_sha256": ACTIVATION_RECEIPT_SHA,
            "protocol_sha256": PROTOCOL_SHA,
            "session_date": SESSION.isoformat(),
        }
    )
    assert decision_id(
        PROTOCOL_SHA,
        ACTIVATION_RECEIPT_SHA,
        SESSION,
    ) == expected


def test_protocol_required_manifest_and_decision_fields_are_hash_bound() -> None:
    fold = fitted_fold()
    assert (
        fold.manifest["canonical_json_contract"]
        == CANONICAL_JSON_CONTRACT
    )
    assert str(fold.manifest["python_version"])

    decision = make(
        snapshot=live_snapshot((disclosure(),)),
        fold=fold,
        price_eligible_codes=("1301",),
    )
    required = {
        "session_date",
        "decision_id",
        "python_version",
        "canonical_json_contract",
        "protocol_sha256",
        "activation_payload_sha256",
        "activation_receipt_sha256",
        "default_branch_tip_sha_at_decision",
        "pre_score_fold_manifest_sha256",
        "live_source_snapshot_sha256",
        "candidate_generation_started_at",
        "decision_at",
        "published_at_max",
        "received_at_max",
        "computed_at",
        "source_complete",
        "qualifying_event_count",
        "selected_code",
        "selected_score",
        "decision",
        "decision_payload_sha256",
    }
    assert required <= set(decision)
    assert (
        decision["canonical_json_contract"]
        == CANONICAL_JSON_CONTRACT
    )
    assert (
        decision["default_branch_tip_sha_at_decision"]
        == DEFAULT_BRANCH_TIP_SHA
    )
    assert (
        decision["pre_score_fold_manifest_sha256"]
        == fold.manifest_sha256
    )
    assert "fold_artifact_sha256" not in decision

    changed_branch = make(
        snapshot=live_snapshot((disclosure(),)),
        fold=fold,
        price_eligible_codes=("1301",),
        default_branch_tip_sha="b" * 64,
    )
    assert changed_branch["decision_id"] == decision["decision_id"]
    assert (
        changed_branch["decision_payload_sha256"]
        != decision["decision_payload_sha256"]
    )

    mutated = dict(decision)
    mutated["default_branch_tip_sha_at_decision"] = "b" * 64
    with pytest.raises(T02ForwardError, match="SHA-256 mismatch"):
        DecisionHashChainLedger().append(mutated)

    missing = dict(decision)
    del missing["python_version"]
    with pytest.raises(T02ForwardError, match="missing fields"):
        DecisionHashChainLedger().append(missing)


def test_four_decision_states_and_score_tie_break_by_code() -> None:
    fold = fitted_fold()
    tied = live_snapshot(
        (
            disclosure(code="1301", title="同じ材料"),
            disclosure(code="1302", title="同じ材料"),
        )
    )
    selected = make(
        snapshot=tied,
        fold=fold,
        price_eligible_codes=("1301", "1302"),
    )
    assert selected["decision"] == DecisionState.SELECTED.value
    assert selected["selected_code"] == "1301"

    cash = make(snapshot=live_snapshot(), fold=fold)
    assert cash["decision"] == DecisionState.CASH_NO_EVENT.value
    assert cash["selected_code"] is None

    missing = make(snapshot=None, fold=fold)
    assert missing["decision"] == DecisionState.FAIL_CLOSED_SOURCE.value

    model_missing = make(
        snapshot=live_snapshot((disclosure(),)),
        fold=None,
        price_eligible_codes=("1301",),
    )
    assert model_missing["decision"] == DecisionState.FAIL_CLOSED_MODEL.value


def test_price_eligibility_mask_excludes_higher_scoring_event() -> None:
    fold = fitted_fold()
    snapshot = live_snapshot(
        (
            disclosure(
                code="1301",
                title="業績予想 下方修正 減益",
            ),
            disclosure(
                code="1302",
                title="業績予想 上方修正 増益",
            ),
        )
    )
    unrestricted = make(
        snapshot=snapshot,
        fold=fold,
        price_eligible_codes=("1301", "1302"),
    )
    assert unrestricted["selected_code"] == "1302"

    restricted = make(
        snapshot=snapshot,
        fold=fold,
        price_eligible_codes=("1301",),
    )
    assert restricted["decision"] == DecisionState.SELECTED.value
    assert restricted["selected_code"] == "1301"
    assert restricted["qualifying_event_count"] == 1

    no_eligible_event = make(
        snapshot=snapshot,
        fold=fold,
        price_eligible_codes=(),
    )
    assert no_eligible_event["decision"] == (
        DecisionState.CASH_NO_EVENT.value
    )


def test_missing_or_invalid_price_eligibility_mask_fails_model_closed() -> None:
    fold = fitted_fold()
    snapshot = live_snapshot((disclosure(),))
    missing = make_t02_decision(
        protocol_sha256=PROTOCOL_SHA,
        activation_payload_sha256=ACTIVATION_PAYLOAD_SHA,
        activation_receipt_sha256=ACTIVATION_RECEIPT_SHA,
        default_branch_tip_sha_at_decision=DEFAULT_BRANCH_TIP_SHA,
        session_date=SESSION,
        candidate_generation_started_at=at(SESSION, 8, 58, 52),
        decision_at=at(SESSION, 8, 58, 55),
        snapshot=snapshot,
        fold=fold,
        price_eligible_codes=None,
    )
    assert missing["decision"] == DecisionState.FAIL_CLOSED_MODEL.value
    invalid = make_t02_decision(
        protocol_sha256=PROTOCOL_SHA,
        activation_payload_sha256=ACTIVATION_PAYLOAD_SHA,
        activation_receipt_sha256=ACTIVATION_RECEIPT_SHA,
        default_branch_tip_sha_at_decision=DEFAULT_BRANCH_TIP_SHA,
        session_date=SESSION,
        candidate_generation_started_at=at(SESSION, 8, 58, 52),
        decision_at=at(SESSION, 8, 58, 55),
        snapshot=snapshot,
        fold=fold,
        price_eligible_codes=("bad-code",),
    )
    assert invalid["decision"] == DecisionState.FAIL_CLOSED_MODEL.value


def test_incomplete_source_is_not_encoded_as_no_event() -> None:
    incomplete = live_snapshot(source_complete=False)
    decision = make(snapshot=incomplete, fold=fitted_fold())
    assert decision["decision"] == DecisionState.FAIL_CLOSED_SOURCE.value
    assert decision["decision"] != DecisionState.CASH_NO_EVENT.value


def test_hash_chain_is_idempotent_and_rejects_conflicting_payload() -> None:
    fold = fitted_fold()
    first = make(
        snapshot=live_snapshot((disclosure(),)),
        fold=fold,
        price_eligible_codes=("1301",),
    )
    next_session = date(2026, 7, 29)
    next_snapshot = replace(
        live_snapshot(),
        session_date=next_session,
        snapshot_id="snapshot-20260729",
        request_started_at=at(next_session, 8, 58, 49),
        source_received_at=at(next_session, 8, 58, 50),
        computed_at=at(next_session, 8, 58, 51),
    )
    second = make(
        snapshot=next_snapshot,
        fold=fold,
        session=next_session,
    )

    ledger = DecisionHashChainLedger()
    first_entry = ledger.append(first)
    assert ledger.append(first) == first_entry
    second_entry = ledger.append(second)
    assert first_entry["sequence_number"] == 0
    assert first_entry["previous_record_sha256"] == "0" * 64
    assert second_entry["sequence_number"] == 1
    assert second_entry["previous_record_sha256"] == (
        first_entry["record_sha256"]
    )
    assert len(ledger.entries) == 2

    conflicting = dict(first)
    conflicting["failure_reason"] = "mutated"
    conflicting["decision_payload_sha256"] = canonical_json_sha256(
        conflicting,
        exclude_fields={
            "decision_payload_sha256",
            "sequence_number",
            "previous_record_sha256",
            "record_sha256",
        },
    )
    with pytest.raises(T02ForwardError, match="conflicting payload"):
        ledger.append(conflicting)


def test_activation_hashes_are_bound_into_id_payload_and_ledger() -> None:
    fold = fitted_fold()
    snapshot = live_snapshot((disclosure(),))
    original = make(
        snapshot=snapshot,
        fold=fold,
        price_eligible_codes=("1301",),
    )
    changed = make_t02_decision(
        protocol_sha256=PROTOCOL_SHA,
        activation_payload_sha256="a" * 64,
        activation_receipt_sha256="b" * 64,
        default_branch_tip_sha_at_decision=DEFAULT_BRANCH_TIP_SHA,
        session_date=SESSION,
        candidate_generation_started_at=at(SESSION, 8, 58, 52),
        decision_at=at(SESSION, 8, 58, 55),
        snapshot=snapshot,
        fold=fold,
        price_eligible_codes=("1301",),
    )
    assert original["decision_id"] != changed["decision_id"]
    assert (
        original["decision_payload_sha256"]
        != changed["decision_payload_sha256"]
    )
    ledger = DecisionHashChainLedger()
    ledger.append(original)
    with pytest.raises(T02ForwardError, match="chronologically"):
        ledger.append(changed)


def test_hash_chain_rejects_payload_hash_mutation_and_out_of_order() -> None:
    fold = fitted_fold()
    first = make(
        snapshot=live_snapshot((disclosure(),)),
        fold=fold,
        price_eligible_codes=("1301",),
    )
    ledger = DecisionHashChainLedger()
    ledger.append(first)

    corrupted = dict(first)
    corrupted["selected_code"] = "9999"
    with pytest.raises(T02ForwardError, match="SHA-256 mismatch"):
        ledger.append(corrupted)

    prior_session = date(2026, 7, 27)
    prior_snapshot = replace(
        live_snapshot(),
        session_date=prior_session,
        snapshot_id="snapshot-20260727",
        source_received_at=at(prior_session, 8, 58, 50),
        computed_at=at(prior_session, 8, 58, 51),
    )
    prior = make(
        snapshot=prior_snapshot,
        fold=fold,
        session=prior_session,
    )
    with pytest.raises(T02ForwardError, match="chronologically"):
        ledger.append(prior)
