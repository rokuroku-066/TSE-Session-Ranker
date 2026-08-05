from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import struct
from pathlib import Path

import pandas as pd
import pytest

from research import model_v18_shoulder_state_audit as audit


def _parsed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2025-05-01",
                "code": "0010",
                "name": '銘柄, "十"',
                "raw_name": '銘柄, "十"',
                "volume": 1000.0,
                "turnover": 12345.0,
                "source_file": "202505.txt",
                "source_line": 7,
                "source_format": "monthly_full",
                "open": -0.0,
                "high": 12.5,
                "low": 10.0,
                "close": 11.0,
                "am_open": None,
                "am_high": None,
                "am_low": None,
                "am_close": None,
                "pm_open": None,
                "pm_high": None,
                "pm_low": None,
                "pm_close": None,
                "traded": True,
                "partial_session": False,
            }
        ]
    )


def _model_price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2025-05-01",
                "code": "0010",
                "name": '銘柄, "十"',
                "open": -0.0,
                "high": 12.5,
                "low": 10.0,
                "close": 11.0,
                "volume": 1000.0,
                "turnover": 12345.0,
                "vwap": None,
                "trading_unit": None,
                "source_format": "monthly_full",
            }
        ]
    )


def _private_tree(root: Path, relative: str, payload: bytes) -> Path:
    root.mkdir(mode=0o700)
    current = root
    parts = Path(relative).parts
    for part in parts[:-1]:
        current = current / part
        current.mkdir(mode=0o700)
    target = current / parts[-1]
    target.write_bytes(payload)
    target.chmod(0o600)
    return target


def test_independent_audit_does_not_import_frozen_runner() -> None:
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any("model_v18_shoulder_state_runner" in name for name in imported)


def test_no_legacy_terminal_or_old_result_input_path_remains() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "audit" in top_level_functions
    assert "_legacy_audit" not in top_level_functions
    assert "_legacy_validate_predictor_evidence" not in top_level_functions
    for forbidden in (
        "predictor_parsed_panel_semantic_set_sha256",
        "predictor_g0_panel_semantic_set_sha256",
        "predictor_common_universe_semantic_set_sha256",
    ):
        assert forbidden not in source


def test_rehearsal_runner_surface_has_no_production_caller_or_output_api() -> None:
    payload = audit.DEFAULT_RUNNER.read_bytes()
    observed = audit.validate_a2_nonauthority_rehearsal_runner_surface(payload)
    assert observed["production_call_count"] == 0
    assert observed["production_retained_month_source_call_count"] == 1
    assert observed["production_retained_month_source_caller"] == (
        "prepare_day.month_factory"
    )
    assert observed["runner_rehearsal_api"] == (
        "build_a2_nonauthority_rehearsal_day"
    )

    production_call = payload + b"\ndef _forbidden_production_call():\n    return build_a2_nonauthority_rehearsal_day(None, ())\n"
    with pytest.raises(audit.AuditError, match="production runner calls"):
        audit.validate_a2_nonauthority_rehearsal_runner_surface(production_call)

    needle = (
        b"    capability = _require_active_a2_rehearsal_contract(rehearsal_contract)\n"
        b"    if input_kind not in A2_REHEARSAL_INPUT_KINDS:\n"
    )
    assert payload.count(needle) == 1
    direct_write = payload.replace(
        needle,
        b"    capability = _require_active_a2_rehearsal_contract(rehearsal_contract)\n"
        b'    os.open("/tmp/forbidden", os.O_WRONLY)\n'
        b"    if input_kind not in A2_REHEARSAL_INPUT_KINDS:\n",
    )
    with pytest.raises(audit.AuditError, match="authority/network I/O"):
        audit.validate_a2_nonauthority_rehearsal_runner_surface(direct_write)


def test_runner_fast_path_rejects_caller_semantics_and_token_identity_bypass() -> None:
    payload = audit.DEFAULT_RUNNER.read_bytes()
    header = (
        b"def _validate_retained_intramonth_month_source(\n"
        b"    manifest: Mapping[str, Any],\n"
        b"    *,\n"
        b"    predictor_raw_store_root: str | Path,\n"
    )
    assert payload.count(header) == 1
    caller_frame = payload.replace(
        header,
        (
            b"def _validate_retained_intramonth_month_source(\n"
            b"    manifest: Mapping[str, Any],\n"
            b"    *,\n"
            b"    model_prices: pd.DataFrame | None = None,\n"
            b"    predictor_raw_store_root: str | Path,\n"
        ),
    )
    with pytest.raises(audit.AuditError, match="caller frames/hashes"):
        audit.validate_a2_nonauthority_rehearsal_runner_surface(caller_frame)

    identity_check = b"            retained_proof[field] != expected\n"
    assert payload.count(identity_check) == 1
    token_bypass = payload.replace(identity_check, b"            False\n")
    with pytest.raises(audit.AuditError, match="fresh proof/fold/score"):
        audit.validate_a2_nonauthority_rehearsal_runner_surface(token_bypass)

    current_hash = b"        \"training_target_sha256\": _numeric_sha(\n"
    assert payload.count(current_hash) == 2
    stale_fold_hash = payload.replace(
        current_hash,
        b'            "training_target_sha256": _trusted_numeric_sha(\n',
        1,
    )
    with pytest.raises(audit.AuditError, match="fold-current hashes"):
        audit.validate_a2_nonauthority_rehearsal_runner_surface(stale_fold_hash)

    exact_snapshot_read = b"    payload = _external_object_bytes(\n"
    snapshot_start = payload.index(b"def _validate_model_price_snapshot_manifest(")
    snapshot_read_at = payload.index(exact_snapshot_read, snapshot_start)
    snapshot_bypass = (
        payload[:snapshot_read_at]
        + b"    payload = _trusted_external_object_bytes(\n"
        + payload[snapshot_read_at + len(exact_snapshot_read) :]
    )
    with pytest.raises(audit.AuditError, match="snapshot exact-byte decoder"):
        audit.validate_a2_nonauthority_rehearsal_runner_surface(snapshot_bypass)


def test_rehearsal_driver_times_only_seam_and_always_revokes_capability() -> None:
    payload = audit.DEFAULT_REHEARSAL.read_bytes()
    observed = audit.validate_a2_nonauthority_rehearsal_driver_surface(payload)
    assert observed["runner_seam_call_count"] == 1
    assert observed["reference_and_cold_postflight_finally"] is True

    needle = b"contract_postflight_hash = _postflight_runner_contract("
    assert payload.count(needle) == 2
    missing_postflight = payload.replace(
        needle,
        b"contract_postflight_hash = _skipped_postflight_contract(",
        1,
    )
    with pytest.raises(audit.AuditError, match="not revoked"):
        audit.validate_a2_nonauthority_rehearsal_driver_surface(
            missing_postflight
        )

    network_client = payload + b"\nimport socket\n"
    with pytest.raises(audit.AuditError, match="network/repository client"):
        audit.validate_a2_nonauthority_rehearsal_driver_surface(network_client)

    token_identity = b"        if intramonth_token is not boundary_token:\n"
    assert payload.count(token_identity) == 1
    token_swap_allowed = payload.replace(token_identity, b"        if False:\n")
    with pytest.raises(audit.AuditError, match="token/comparison chain"):
        audit.validate_a2_nonauthority_rehearsal_driver_surface(
            token_swap_allowed
        )


def test_rehearsal_envelope_is_disjoint_from_production_manifest_schema() -> None:
    envelope = {field: None for field in audit.A2_REHEARSAL_ENVELOPE_FIELDS}
    envelope.update(
        {
            "schema_version": 1,
            "scope": "nonauthority_rehearsal_only",
            "production_authority": False,
            "canonical_artifact_written": False,
        }
    )
    with pytest.raises(audit.AuditError, match="source manifest fields"):
        audit.validate_source_manifest(
            envelope,
            session_date="2026-08-06",
            protocol={},
        )


def test_a2_jsonl_real_monthly_nulls_unicode_and_negative_zero_roundtrip() -> None:
    payload = audit._canonical_a2_frame_jsonl_bytes(
        _parsed_frame(), audit.PARSED_PANEL_COLUMNS, label="monthly fixture"
    )
    decoded = audit.decode_a2_canonical_frame_jsonl(
        payload, audit.PARSED_PANEL_COLUMNS, label="monthly fixture"
    )
    assert decoded.columns.tolist() == list(audit.PARSED_PANEL_COLUMNS)
    assert decoded.loc[0, "name"] == '銘柄, "十"'
    assert pd.isna(decoded.loc[0, "trading_unit"])
    assert pd.isna(decoded.loc[0, "final_special_quote"])
    assert struct.pack("<d", float(decoded.loc[0, "open"])) == struct.pack(
        "<d", -0.0
    )
    with pytest.raises(audit.AuditError, match="non-finite|numeric"):
        changed = _parsed_frame()
        changed.loc[0, "close"] = float("inf")
        audit._canonical_a2_frame_jsonl_bytes(
            changed, audit.PARSED_PANEL_COLUMNS, label="bad monthly fixture"
        )


def test_model_price_csv_quotes_nulls_binary_float_and_negative_zero_roundtrip() -> None:
    payload = audit._canonical_a2_model_price_csv_bytes(
        _model_price_frame(), label="compact fixture"
    )
    assert b'"\xe9\x8a\x98\xe6\x9f\x84, ""\xe5\x8d\x81"""' in payload
    decoded = audit.decode_a2_canonical_model_price_csv(
        payload, label="compact fixture"
    )
    assert pd.isna(decoded.loc[0, "vwap"])
    assert struct.pack("<d", float(decoded.loc[0, "open"])) == struct.pack(
        "<d", -0.0
    )
    assert audit._canonical_a2_model_price_csv_bytes(
        decoded, label="compact fixture"
    ) == payload


def test_referenced_external_object_ignores_unreferenced_extras(tmp_path: Path) -> None:
    root = tmp_path / "derived"
    key = "model_v18_shoulder_state/predictor-shard/a.jsonl"
    _private_tree(root, key, b"bound\n")
    extra = root / "totally-unreferenced.extra"
    extra.write_bytes(b"nonauthority")
    extra.chmod(0o600)
    assert audit._read_external_object_bytes(
        root,
        key,
        required_prefix=audit.PREDICTOR_SHARD_OBJECT_PREFIX,
        identity_registry={},
    ) == b"bound\n"


@pytest.mark.parametrize("attack", ["extra", "mode", "symlink", "type"])
def test_local_authority_enumeration_rejects_extra_or_unsafe_entry(
    tmp_path: Path, attack: str
) -> None:
    root = tmp_path / "local-authority"
    root.mkdir(mode=0o700)
    target = root / "2026-08.json"
    target.write_bytes(b"{}\n")
    target.chmod(0o600)
    if attack == "extra":
        extra = root / "unregistered.extra"
        extra.write_bytes(b"x")
        extra.chmod(0o600)
    elif attack == "mode":
        target.chmod(0o644)
    elif attack == "symlink":
        target.unlink()
        target.symlink_to(tmp_path / "victim")
    else:
        target.unlink()
        target.mkdir(mode=0o700)
    with pytest.raises(audit.AuditError, match="extra|filename|mode|link"):
        audit._private_local_authority_entries(
            root,
            label="test local authority",
            expected_names=["2026-08.json"],
            filename_pattern=r"\d{4}-\d{2}\.json",
        )


@pytest.mark.parametrize(
    "authority_kind",
    ["source", "month_source", "state", "outcome", "fold_manifest", "fold_model"],
)
def test_each_canonical_manifest_map_rejects_unregistered_local_entries(
    tmp_path: Path, authority_kind: str
) -> None:
    """An extra local final is never treated like an external derived extra."""

    def authority_dir(name: str, expected_name: str) -> Path:
        root = tmp_path / name
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        expected = root / expected_name
        expected.write_bytes(b"{}\n")
        expected.chmod(0o600)
        return root

    if authority_kind == "source":
        root = authority_dir("source", "2026-08-06.json")
        invoke = lambda: audit._source_manifest_map(  # noqa: E731
            root, {}, expected_sessions=["2026-08-06"]
        )
    elif authority_kind == "month_source":
        root = authority_dir("month-source", "2026-08.json")
        invoke = lambda: audit._month_source_manifest_map(  # noqa: E731
            root, expected_months=["2026-08"]
        )
    elif authority_kind == "state":
        root = authority_dir("state", "2026-08.json")
        invoke = lambda: audit._manifest_map(  # noqa: E731
            root, {}, expected_months=["2026-08"]
        )
    elif authority_kind == "outcome":
        root = authority_dir("outcome", "2026-08-06.json")
        invoke = lambda: audit._outcome_manifest_map(  # noqa: E731
            root,
            [{"session_date": "2026-08-06"}],
            {},
            raw_store_root=tmp_path / "unused-raw",
        )
    else:
        manifests = authority_dir("fold-manifests", "2026-08.json")
        models = authority_dir("fold-models", "2026-08.json")
        root = manifests if authority_kind == "fold_manifest" else models
        invoke = lambda: audit._fold_artifact_maps(  # noqa: E731
            manifests,
            models,
            {},
            runner_sha256="1" * 64,
            expected_months=["2026-08"],
        )

    extra = root / "unregistered.extra"
    extra.write_bytes(b"x")
    extra.chmod(0o600)
    with pytest.raises(audit.AuditError, match="missing or extra"):
        invoke()


@pytest.mark.parametrize("attack", ["symlink", "hardlink"])
def test_referenced_external_object_rejects_link_attacks(
    tmp_path: Path, attack: str
) -> None:
    root = tmp_path / "derived"
    key = "model_v18_shoulder_state/predictor-shard/a.jsonl"
    target = _private_tree(root, key, b"bound\n")
    victim = tmp_path / "victim"
    victim.write_bytes(b"victim")
    victim.chmod(0o600)
    target.unlink()
    if attack == "symlink":
        target.symlink_to(victim)
    else:
        os.link(victim, target)
    with pytest.raises(audit.AuditError, match="symlink|single-link|descriptor"):
        audit._read_external_object_bytes(
            root,
            key,
            required_prefix=audit.PREDICTOR_SHARD_OBJECT_PREFIX,
            identity_registry={},
        )
    assert victim.read_bytes() == b"victim"


def test_model_snapshot_key_binds_immediate_predecessor() -> None:
    protocol = {
        "source_contract": {"forward_daily": {"parser_sha256": "1" * 64}}
    }
    common = {
        "target_month": pd.Period("2026-09"),
        "latest_source_session": pd.Timestamp("2026-08-31"),
        "raw_source_set_sha256": "2" * 64,
        "parsed_shard_set_sha256": "3" * 64,
        "protocol": protocol,
        "runner_sha256": "4" * 64,
    }
    first = audit._a2_model_snapshot_identity(
        **common, previous_snapshot_manifest_sha256="5" * 64
    )
    changed = audit._a2_model_snapshot_identity(
        **common, previous_snapshot_manifest_sha256="6" * 64
    )
    assert first != changed
    assert first[1].startswith(
        "model_v18_shoulder_state/model-price-snapshot/2026-09/"
    )


def _authority_record(session: str = "2026-08-06") -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "sequence_number": 0,
        "session_date": session,
        "previous_record_sha256": audit.ZERO_SHA256,
    }
    value["record_sha256"] = audit.canonical_json_sha256(value)
    return value


def test_record_shard_authority_exact_and_unpublished_stage_ignored(
    tmp_path: Path,
) -> None:
    authority = tmp_path / "records"
    authority.mkdir(mode=0o700)
    row = _authority_record()
    payload = audit.canonical_json_bytes(row) + b"\n"
    final = authority / "2026-08-06.json"
    final.write_bytes(payload)
    final.chmod(0o600)
    stage = authority / ".2026-08-07.json.staging"
    stage.write_bytes(b"unpublished")
    stage.chmod(0o600)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_bytes(payload)
    ledger.chmod(0o600)
    fields = tuple(row)
    observed, observed_bytes = audit.load_record_shard_authority(
        ledger_path=ledger,
        authority_directory=authority,
        key_field="session_date",
        required_fields=fields,
        validator=lambda rows: audit.validate_hash_chain(
            rows, required_fields=fields
        ),
    )
    assert observed == [row]
    assert observed_bytes == payload


def test_record_shard_authority_rejects_extra_or_torn_derived(tmp_path: Path) -> None:
    authority = tmp_path / "records"
    authority.mkdir(mode=0o700)
    row = _authority_record()
    payload = audit.canonical_json_bytes(row) + b"\n"
    final = authority / "2026-08-06.json"
    final.write_bytes(payload)
    final.chmod(0o600)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_bytes(payload[:-1])
    ledger.chmod(0o600)
    fields = tuple(row)
    with pytest.raises(audit.AuditError, match="derived canonical ledger"):
        audit.load_record_shard_authority(
            ledger_path=ledger,
            authority_directory=authority,
            key_field="session_date",
            required_fields=fields,
            validator=lambda rows: audit.validate_hash_chain(
                rows, required_fields=fields
            ),
        )
    ledger.write_bytes(payload)
    extra = authority / "extra"
    extra.write_bytes(b"x")
    with pytest.raises(audit.AuditError, match="unregistered entry"):
        audit.load_record_shard_authority(
            ledger_path=ledger,
            authority_directory=authority,
            key_field="session_date",
            required_fields=fields,
            validator=lambda rows: audit.validate_hash_chain(
                rows, required_fields=fields
            ),
        )


def test_score_session_authority_exact_pair_and_cumulative_binding(
    tmp_path: Path,
) -> None:
    rows = []
    for rank, code in ((1, "0010"), (2, "0020")):
        rows.append(
            {
                "session_date": "2026-08-06",
                "source_rank": rank,
                "code": code,
                "name": f"銘柄{rank}",
                "model_score": 1.0 / rank,
                "feature_source_max_date": "2026-08-05",
                "score_generated_at": "2026-08-06T08:50:00+09:00",
                "runtime_lock_sha256": audit.RUNTIME_LOCK_SHA256,
                "runtime_lock_verified_at": "2026-08-06T08:49:00+09:00",
                "source_manifest_sha256": "a" * 64,
                "c00_fold_manifest_sha256": "b" * 64,
            }
        )
    frame = pd.DataFrame(rows, columns=audit.SCORE_FIELDS)
    payload = frame.to_csv(index=False, lineterminator="\n").encode()
    ledger = tmp_path / "scores.csv"
    ledger.write_bytes(payload)
    ledger.chmod(0o600)
    authority = tmp_path / "score-sessions"
    authority.mkdir(mode=0o700)
    shard = authority / "2026-08-06.csv"
    shard.write_bytes(payload)
    shard.chmod(0o600)
    observed = audit.validate_score_session_authority(
        frame, score_path=ledger, authority_directory=authority
    )
    expected = audit.canonical_json_sha256(
        [
            {
                "session_date": "2026-08-06",
                "byte_count": len(payload),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "semantic_sha256": audit.semantic_score_hash(frame),
            }
        ]
    )
    assert observed == expected
    shard.write_bytes(payload + b"x")
    with pytest.raises(audit.AuditError, match="differs"):
        audit.validate_score_session_authority(
            frame, score_path=ledger, authority_directory=authority
        )


def test_result_status_tail_reader_never_reads_performance_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result.json"
    body = {
        "models": {"sealed_performance": "X" * 200_000},
        "status": "forward_rejected_candidate",
    }
    path.write_bytes(audit.canonical_json_file_bytes(body))
    path.chmod(0o600)
    real_pread = os.pread
    total = 0

    def counted(fd: int, size: int, offset: int) -> bytes:
        nonlocal total
        total += size
        return real_pread(fd, size, offset)

    monkeypatch.setattr(audit.os, "pread", counted)
    assert (
        audit._result_status_token_without_performance_read(path)
        == "forward_rejected_candidate"
    )
    assert total < 100


def test_result_status_tail_reader_rejects_nonprivate_or_noncanonical(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"status":"forward_rejected_candidate"}\n', encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(audit.AuditError, match="metadata"):
        audit._result_status_token_without_performance_read(path)
    path.chmod(0o600)
    with pytest.raises(audit.AuditError, match="suffix|line"):
        audit._result_status_token_without_performance_read(path)


def test_abort_artifact_fingerprint_is_streaming_and_opaque(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "opaque-ledger"
    payload = b"not-json\x00performance-is-never-decoded" * 1024
    path.write_bytes(payload)
    monkeypatch.setattr(
        audit,
        "_stable_plain_file_bytes",
        lambda *a, **k: pytest.fail("opaque abort fingerprint loaded whole bytes"),
    )
    assert audit._stream_plain_file_sha256(path, label="opaque") == hashlib.sha256(
        payload
    ).hexdigest()


def test_cross_role_exact_bytes_receipt_and_terminal_only_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = "2026-08-06T08:47:00+09:00"
    predictor = [
        {
            "object_key": "model_v18_shoulder_state/predictor/daily/stq_20260806.pdf",
            "file": "stq_20260806.pdf",
            "url": "https://www.jpx.co.jp/markets/statistics-equities/daily/x-att/stq_20260806.pdf",
            "byte_count": 10,
            "sha256": "a" * 64,
        }
    ]
    bindings = [{"shard_manifest_object_key": "model_v18_shoulder_state/predictor-shard/a.manifest.json"}]
    outcomes = [
        {
            "target_session": "2026-08-06",
            "source_file_name": "stq_20260806.pdf",
            "source_url": predictor[0]["url"],
            "source_byte_count": 10,
            "source_sha256": "a" * 64,
            "source_received_at": receipt,
        },
        {
            "target_session": "2026-08-07",
            "source_file_name": "stq_20260807.pdf",
            "source_url": "https://www.jpx.co.jp/markets/statistics-equities/daily/x-att/stq_20260807.pdf",
            "source_byte_count": 11,
            "source_sha256": "b" * 64,
            "source_received_at": "2026-08-07T08:47:00+09:00",
        },
    ]
    monkeypatch.setattr(
        audit,
        "_read_external_canonical_json",
        lambda *args, **kwargs: (
            {"chronology_class": "forward", "raw_received_at": receipt},
            b"{}\n",
        ),
    )
    audit.validate_terminal_predictor_outcome_cross_role(
        predictor,
        bindings,
        outcomes,
        terminal_session="2026-08-07",
        predictor_derived_store_root=Path("/unused"),
        external_identity_registry={},
    )
    changed = [dict(item) for item in outcomes]
    changed[0] = {**changed[0], "source_sha256": "c" * 64}
    with pytest.raises(audit.AuditError, match="different daily bytes"):
        audit.validate_terminal_predictor_outcome_cross_role(
            predictor,
            bindings,
            changed,
            terminal_session="2026-08-07",
            predictor_derived_store_root=Path("/unused"),
            external_identity_registry={},
        )


@pytest.mark.parametrize(
    "url,file_name",
    [
        (
            "https://www.jpx.co.jp/markets/statistics-equities/daily/x-att/stq_20260805.pdf?download=1",
            "stq_20260805.pdf",
        ),
        (
            "https://example.com/markets/statistics-equities/daily/x-att/stq_20260805.pdf",
            "stq_20260805.pdf",
        ),
        (
            "https://www.jpx.co.jp/markets/statistics-equities/daily/x-att/stq_20260804.pdf",
            "stq_20260804.pdf",
        ),
    ],
)
def test_daily_url_label_rejects_query_host_or_date_drift(
    url: str, file_name: str
) -> None:
    with pytest.raises(audit.AuditError):
        audit._official_jpx_daily_url_label(
            url,
            file_name=file_name,
            source_session="2026-08-05",
            label="daily URL",
        )


def test_selection_order_is_raw_then_derived_then_checkpoint_then_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    monkeypatch.setattr(audit, "_require_lexical_canonical_path", lambda *a, **k: Path("."))
    monkeypatch.setattr(
        audit, "_require_private_local_file", lambda path, *, label: Path(path)
    )
    monkeypatch.setattr(
        audit,
        "_result_status_token_without_performance_read",
        lambda path: events.append("tail") or "forward_rejected_candidate",
    )
    protocol = {"result_contract": {"required_input_fields": ["checkpoint", "runtime"]}}
    monkeypatch.setattr(
        audit,
        "validate_protocol_contract",
        lambda path: events.append("protocol") or (protocol, "1" * 64),
    )
    monkeypatch.setattr(
        audit,
        "validate_runtime_lock",
        lambda *a, **k: events.append("runtime") or ({}, "2" * 64),
    )
    monkeypatch.setattr(audit, "_validate_external_roots_disjoint", lambda roots: events.append("roots") or {})
    monkeypatch.setattr(
        audit, "_stable_plain_file_bytes", lambda path, **kwargs: b"{}\n"
    )
    monkeypatch.setattr(
        audit,
        "validate_a2_nonauthority_rehearsal_runner_surface",
        lambda payload: {},
    )
    monkeypatch.setattr(
        audit,
        "validate_a2_nonauthority_rehearsal_driver_surface",
        lambda payload: {},
    )
    monkeypatch.setattr(audit, "validate_activation_payload", lambda *a, **k: "a" * 64)
    monkeypatch.setattr(audit, "validate_activation_receipt", lambda *a, **k: "b" * 64)
    context = {field: None for field in audit.ACTIVATION_CONTEXT_FIELDS}
    context.update(
        {
            "first_counted_session": "2026-08-06",
            "terminal_session": "2026-08-06",
            "terminal_scheduled_sessions": 1,
            "activation_receipt_workflow_run_observed_at": "2026-08-05T08:00:00+09:00",
        }
    )
    monkeypatch.setattr(audit, "validate_activation_context", lambda *a, **k: context)
    decision = {field: context[field] for field in audit.ACTIVATION_CONTEXT_FIELDS[:14]}
    decision.update(
        {
            "session_date": "2026-08-06",
            "source_manifest_sha256": "source",
        }
    )
    monkeypatch.setattr(
        audit,
        "load_record_shard_authority",
        lambda **kwargs: ([decision], b"decision\n"),
    )
    calendar = pd.DatetimeIndex([pd.Timestamp("2026-08-06")])
    monkeypatch.setattr(audit, "load_registered_calendar", lambda path=None: calendar)
    monkeypatch.setattr(audit, "validate_activation_git_history", lambda *a, **k: {})
    monkeypatch.setattr(audit, "validate_live_module_origin_closure", lambda *a, **k: None)
    source = {"source_manifest_sha256": "source"}
    monkeypatch.setattr(audit, "_source_manifest_map", lambda *a, **k: {"source": source})
    raw = {"object_key": "model_v18_shoulder_state/predictor/daily/x.pdf"}
    monkeypatch.setattr(audit, "_a2_source_records", lambda value: [raw])
    monkeypatch.setattr(
        audit,
        "_reparse_predictor_raw_objects_once",
        lambda *a, **k: events.append("raw") or {raw["object_key"]: {}},
    )
    monkeypatch.setattr(
        audit,
        "_month_source_manifest_map",
        lambda *a, **k: events.append("derived") or {},
    )
    monkeypatch.setattr(audit, "_fold_artifact_maps", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(
        audit,
        "validate_predictor_evidence",
        lambda *a, **k: events.append("predictor")
        or {
            "_outcome_blind_score_expectations": [],
            "_terminal_predictor_raw_records": [],
            "_terminal_predictor_shard_bindings": [],
            "runtime": "r",
        },
    )
    monkeypatch.setattr(
        audit,
        "validate_checkpoint_evidence",
        lambda *a, **k: events.append("checkpoint") or {"checkpoint": "c"},
    )

    def stop_at_state(*args, **kwargs):  # type: ignore[no-untyped-def]
        events.append("state")
        raise audit.AuditError("state sentinel")

    monkeypatch.setattr(audit, "_manifest_map", stop_at_state)
    roots = [tmp_path / name for name in ("pr", "pd", "out", "cp")]
    with pytest.raises(audit.AuditError, match="state sentinel"):
        audit.audit(
            predictor_raw_store_root=roots[0],
            predictor_derived_store_root=roots[1],
            outcome_raw_store_root=roots[2],
            checkpoint_core_store_root=roots[3],
        )
    assert events.index("raw") < events.index("derived") < events.index("predictor")
    assert events.index("predictor") < events.index("checkpoint") < events.index("state")


def test_abort_branch_rejects_any_store_root_before_full_result_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "_require_lexical_canonical_path", lambda *a, **k: Path("."))
    monkeypatch.setattr(
        audit,
        "_result_status_token_without_performance_read",
        lambda path: "aborted_integrity_failure",
    )
    monkeypatch.setattr(audit, "validate_protocol_contract", lambda path: ({}, "1" * 64))
    monkeypatch.setattr(audit, "validate_runtime_lock", lambda *a, **k: ({}, "2" * 64))
    monkeypatch.setattr(
        audit,
        "_stable_plain_file_bytes",
        lambda *a, **k: pytest.fail("abort root rejection read full result/artifact"),
    )
    with pytest.raises(audit.AuditError, match="does not accept external evidence roots"):
        audit.audit(predictor_derived_store_root=tmp_path / "derived")


def test_abort_envelope_requires_context_c_and_manual_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "protocol.json"
    runner_path = tmp_path / "runner.py"
    payload_path = tmp_path / "payload.json"
    receipt_path = tmp_path / "receipt.json"
    context_path = tmp_path / "context.json"
    for path in (protocol_path, payload_path, receipt_path, context_path):
        path.write_text("{}\n", encoding="utf-8")
    context_path.chmod(0o600)
    runner_path.write_text("# frozen\n", encoding="utf-8")
    required_inputs = [
        "runtime_lock_sha256",
        "predictor_parser_sha256",
        "v17_c00_protocol_sha256",
        "v17_c00_runner_sha256",
        "predictor_parsed_shard_binding_set_sha256",
    ]
    artifact_fields = [
        "decision_ledger_sha256",
        "outcome_ledger_sha256",
        "completed_month_ledger_sha256",
        "score_output_sha256",
        "score_semantic_sha256",
        "picks_output_sha256",
    ]
    runtime_fields = [
        "runtime_lock_sha256",
        "runtime_lock_self_sha256",
        "runtime_lock_verified_at",
        "python_version",
        "numpy_version",
        "pandas_version",
        "scikit_learn_version",
    ]
    authority = {
        "analysis_type": "genuinely_later_forward_shadow_development",
        "production_model_changed": False,
        "production_promotion_allowed": False,
        "orders_allowed": False,
    }
    protocol = {
        "protocol_id": audit.PROTOCOL_ID,
        "result_contract": {
            "required_top_level_fields": [
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
                "raw_source_provenance",
                "input",
                "forward_period",
                "state_months",
                "models",
                "candidate_gate",
                "decision",
                "artifact_sha256",
                "runtime",
            ],
            "abort_result_contract": {"canonical_status": "aborted_integrity_failure"},
            "abort_failure_reason_values": ["audit_integrity_failure"],
            "abort_integrity_stage_values": ["independent_audit"],
            "abort_stage_reason_values": {
                "independent_audit": ["audit_integrity_failure"]
            },
            "authority_values": authority,
            "required_input_fields": required_inputs,
            "required_artifact_hashes": artifact_fields,
            "required_runtime_fields": runtime_fields,
        },
        "runtime_lock_contract": {
            "file_sha256": audit.RUNTIME_LOCK_SHA256,
            "self_sha256": audit.RUNTIME_LOCK_SELF_SHA256,
        },
        "source_contract": {"forward_daily": {"parser_sha256": "1" * 64}},
        "prior_result_binding": {
            "v17": {"protocol_sha256": "2" * 64, "runner_sha256": "3" * 64}
        },
    }
    runtime_lock = {
        "runtime_lock_self_sha256": audit.RUNTIME_LOCK_SELF_SHA256,
        "runtime": {
            "python": {"version": "3.12.13"},
            "distributions": [
                {"name": "numpy", "version": "2.3.5"},
                {"name": "pandas", "version": "2.2.3"},
                {"name": "scikit-learn", "version": "1.8.0"},
            ],
        },
    }
    payload_sha = "4" * 64
    receipt_sha = "5" * 64
    commit = "6" * 40
    monkeypatch.setattr(audit, "validate_activation_payload", lambda *a, **k: payload_sha)
    monkeypatch.setattr(audit, "validate_activation_receipt", lambda *a, **k: receipt_sha)
    monkeypatch.setattr(
        audit,
        "validate_activation_context",
        lambda *a, **k: {"activation_receipt_commit_sha": commit},
    )
    expected_input = dict.fromkeys(required_inputs)
    expected_input.update(
        {
            "runtime_lock_sha256": audit.RUNTIME_LOCK_SHA256,
            "predictor_parser_sha256": "1" * 64,
            "v17_c00_protocol_sha256": "2" * 64,
            "v17_c00_runner_sha256": "3" * 64,
        }
    )
    runtime = {
        "runtime_lock_sha256": audit.RUNTIME_LOCK_SHA256,
        "runtime_lock_self_sha256": audit.RUNTIME_LOCK_SELF_SHA256,
        "runtime_lock_verified_at": "2026-08-05T10:00:00+09:00",
        "python_version": "3.12.13",
        "numpy_version": "2.3.5",
        "pandas_version": "2.2.3",
        "scikit_learn_version": "1.8.0",
    }
    result = {
        "schema_version": 1,
        "protocol_id": audit.PROTOCOL_ID,
        "protocol_sha256": audit.sha256_file(protocol_path),
        "runner_sha256": audit.sha256_file(runner_path),
        "activation_payload_sha256": payload_sha,
        "activation_receipt_sha256": receipt_sha,
        "activation_receipt_commit_sha": commit,
        "status": "aborted_integrity_failure",
        "failure_reason": "audit_integrity_failure",
        "integrity_stage": "independent_audit",
        "authority": authority,
        "raw_source_provenance": audit._raw_source_provenance_envelope(),
        "input": expected_input,
        "forward_period": None,
        "state_months": None,
        "models": None,
        "candidate_gate": None,
        "decision": {
            "research_nominee": None,
            "failure_reason": "audit_integrity_failure",
            "integrity_stage": "independent_audit",
        },
        "artifact_sha256": dict.fromkeys(artifact_fields),
        "runtime": runtime,
    }
    missing_artifacts = {
        "decision_ledger_sha256": tmp_path / "decision-missing",
        "outcome_ledger_sha256": tmp_path / "outcome-missing",
        "completed_month_ledger_sha256": tmp_path / "month-missing",
        "score_output_sha256": tmp_path / "score-missing",
        "picks_output_sha256": tmp_path / "picks-missing",
    }
    assert audit.validate_abort_result(
        result,
        protocol,
        runtime_lock=runtime_lock,
        protocol_path=protocol_path,
        runner_path=runner_path,
        activation_payload_path=payload_path,
        activation_receipt_path=receipt_path,
        activation_context_path=context_path,
        artifact_paths=missing_artifacts,
    )["activation_receipt_commit_sha"] == commit
    changed = {**result, "activation_receipt_commit_sha": None}
    with pytest.raises(audit.AuditError, match="context binding"):
        audit.validate_abort_result(
            changed,
            protocol,
            runtime_lock=runtime_lock,
            protocol_path=protocol_path,
            runner_path=runner_path,
            activation_payload_path=payload_path,
            activation_receipt_path=receipt_path,
            activation_context_path=context_path,
            artifact_paths=missing_artifacts,
        )


def test_protocol_registers_exact_a2_result_and_activation_test_artifacts() -> None:
    protocol, _ = audit.validate_protocol_contract()
    required = protocol["result_contract"]["required_input_fields"]
    assert required[8:11] == [
        "predictor_parsed_shard_binding_set_sha256",
        "predictor_target_slice_semantic_set_sha256",
        "predictor_target_date_scoring_input_semantic_set_sha256",
    ]
    assert not any("parsed_panel_semantic" in item for item in required)
    prereg = protocol["activation"]["preregistration_commit"]
    assert (
        tuple(prereg["additional_test_artifact_paths"])
        == audit.ADDITIONAL_TEST_ARTIFACT_PATHS
    )
    assert "VALIDATION.md" in prereg["required_paths"]
    assert "research/model_v18_a2_rehearsal.py" in prereg["required_paths"]
    payload_fields = protocol["activation"]["payload"]["required_fields"]
    assert "validation_report_sha256" in payload_fields
    assert "rehearsal_sha256" in payload_fields
    timing = protocol["a2_operational_repair_contract"]["timing_activation_gate"]
    assert "exact canonical full-prefix CSV byte count/SHA" in timing[
        "comparison_rule"
    ]
    assert "without caller-supplied frames or hashes" in timing["rule"]
    runtime, _ = audit.validate_runtime_lock(strict_environment=False)
    assert runtime["runtime"]["startup_and_module_closure"][
        "direct_activation_project_module_paths"
    ] == [
        "research/model_v18_a2_rehearsal.py",
        "research/model_v18_shoulder_state_audit.py",
        "research/model_v18_shoulder_state_runner.py",
    ]
