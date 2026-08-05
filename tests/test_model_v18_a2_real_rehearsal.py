from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
from typing import Any

import pandas as pd
import pytest

from research import model_v18_a2_rehearsal as rehearsal


def _fresh_private_root(tmp_path: Path, name: str = "fresh") -> Path:
    root = tmp_path / name
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    return root


def _synthetic_parsed_prices() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dates = pd.bdate_range("2025-09-01", "2026-08-04")
    source_format = "jpx_stock_quotations_auction_regular_way_domestic_ordinary"
    for code_index, code in enumerate(("1001", "1002", "1003")):
        for session_index, session in enumerate(dates):
            open_price = 1_000.0 + 10.0 * code_index + session_index
            return_fraction = (
                ((session_index * 2 + code_index) % 7) - 3
            ) * 0.0015
            close = open_price * (1.0 + return_fraction)
            high = max(open_price, close) * 1.003
            low = min(open_price, close) * 0.997
            volume = 100_000.0 + 1_000.0 * code_index
            vwap = (open_price + close) / 2.0
            rows.append(
                {
                    "date": session,
                    "code": code,
                    "name": f"name-{code}",
                    "raw_name": f"name-{code}",
                    "trading_unit": 100,
                    "final_special_quote": None,
                    "net_change": close - open_price,
                    "vwap": vwap,
                    "volume": volume,
                    "turnover": volume * vwap,
                    "volume_unit": "shares",
                    "turnover_unit": "yen",
                    "source_volume_unit": "shares",
                    "source_turnover_unit": "yen",
                    "source_file": "synthetic_rehearsal.txt",
                    "source_line": code_index + 1,
                    "source_format": source_format,
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
                    "traded": True,
                    "partial_session": False,
                }
            )
    return rehearsal.runner._coerce_jsonl_frame(
        pd.DataFrame(rows),
        rehearsal.runner.PARSED_PRICE_COLUMNS,
        label="varied synthetic rehearsal prices",
    )


def _prepare_synthetic_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Any:
    runner = rehearsal.runner
    monkeypatch.setattr(runner.np, "__version__", "2.3.5")
    monkeypatch.setattr(runner.pd, "__version__", "2.2.3")
    monkeypatch.setattr(runner.sklearn, "__version__", "1.8.0")
    monkeypatch.setattr(runner, "PROTOCOL_SHA256", runner.sha256_file(runner.PROTOCOL))
    protocol_value = runner.read_json(runner.PROTOCOL)
    runtime_value = runner.read_json(runner.RUNTIME_LOCK)
    monkeypatch.setattr(
        runner,
        "validate_protocol",
        lambda: (protocol_value, runner.PROTOCOL_SHA256),
    )
    monkeypatch.setattr(
        runner,
        "validate_runtime_lock",
        lambda *, strict_environment=False: (
            runtime_value,
            runner.RUNTIME_LOCK_SHA256,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_normalised_native_threadpools",
        lambda *, hash_libraries: [
            {
                key: (1 if key == "num_threads" else child)
                for key, child in item.items()
                if key != "library_sha256"
            }
            for item in runtime_value["runtime"]["native_threadpools"]
        ],
    )
    monkeypatch.setattr(runner, "_STRICT_RUNTIME_ACTIVE", True)
    monkeypatch.setattr(
        runner,
        "_STRICT_RUNTIME_VERIFIED_AT",
        runner._timestamp("2026-08-05T00:00:00+09:00", "test runtime verified"),
    )
    monkeypatch.setattr(
        runner, "_validate_startup_and_module_closure", lambda **kwargs: {}
    )
    for name in (
        "ACTIVATION_PAYLOAD",
        "ACTIVATION_RECEIPT",
        "ACTIVATION_CONTEXT",
        "RESULT_OUTPUT",
    ):
        monkeypatch.setattr(runner, name, tmp_path / f"absent-{name}.json")
    monkeypatch.setattr(runner, "MONTH_SOURCE_MANIFEST_DIR", tmp_path / "month")
    monkeypatch.setattr(runner, "FOLD_MANIFEST_DIR", tmp_path / "fold")
    monkeypatch.setattr(runner, "FOLD_MODEL_DIR", tmp_path / "bundle")
    return runner.prepare_a2_nonauthority_rehearsal_contract()


def test_cli_surface_is_fixed_and_has_no_model_or_authority_injection() -> None:
    parser = rehearsal._build_parser()
    prepared = parser.parse_args(
        [
            "prepare-reference",
            "--source-registry",
            "/tmp/registry.json",
            "--output-root",
            "/tmp/reference",
        ]
    )
    assert vars(prepared) == {
        "command": "prepare-reference",
        "source_registry": "/tmp/registry.json",
        "output_root": "/tmp/reference",
    }
    run = parser.parse_args(
        [
            "run",
            "--source-registry",
            "/tmp/registry.json",
            "--reference-root",
            "/tmp/reference",
            "--output-root",
            "/tmp/run-1",
            "--cold-run-index",
            "1",
        ]
    )
    assert vars(run) == {
        "command": "run",
        "source_registry": "/tmp/registry.json",
        "reference_root": "/tmp/reference",
        "output_root": "/tmp/run-1",
        "cold_run_index": 1,
    }
    for forbidden in (
        "--target-session",
        "--panel-builder",
        "--fold-builder",
        "--score-builder",
        "--activation-payload",
        "--checkpoint",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "run",
                    "--source-registry",
                    "/tmp/registry.json",
                    "--reference-root",
                    "/tmp/reference",
                    "--output-root",
                    "/tmp/run-1",
                    "--cold-run-index",
                    "1",
                    forbidden,
                    "injected",
                ]
            )


def test_public_runner_rehearsal_seam_signature_stays_fixed() -> None:
    parameters = inspect.signature(
        rehearsal.runner.build_a2_nonauthority_rehearsal_day
    ).parameters
    assert tuple(parameters) == (
        "snapshot_prices",
        "suffix_prices",
        "rehearsal_contract",
        "input_kind",
        "target_session",
        "runtime_lock_verified_at",
        "month_source_sealed_at",
        "fit_started_at",
        "fit_completed_at",
        "score_generated_at",
        "source_manifest_sha256",
        "source_set_sha256",
        "parsed_shard_set_sha256",
        "reuse_fold_token",
    )
    assert parameters["snapshot_prices"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["suffix_prices"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(
        parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in tuple(parameters)[2:]
    )
    assert parameters["reuse_fold_token"].default is None


def test_real_seam_reuses_only_exact_prefix_proof_and_keeps_core_fresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = rehearsal.runner
    capability = _prepare_synthetic_contract(monkeypatch, tmp_path)
    full = _synthetic_parsed_prices()
    projected = runner._coerce_model_price_frame(
        full, label="synthetic rehearsal compact projection"
    )
    july_snapshot = runner.canonical_model_price_csv_bytes(
        projected.loc[pd.to_datetime(projected["date"]).le("2026-07-31")],
        label="synthetic July snapshot",
    )
    june_snapshot = runner.canonical_model_price_csv_bytes(
        projected.loc[pd.to_datetime(projected["date"]).le("2026-06-30")],
        label="synthetic June snapshot",
    )
    august_suffix = full.loc[pd.to_datetime(full["date"]).gt("2026-07-31")]
    july_august_suffix = full.loc[
        pd.to_datetime(full["date"]).gt("2026-06-30")
    ]
    tampered_suffix = july_august_suffix.copy(deep=True)
    tampered_index = tampered_suffix.index[0]
    tampered_suffix.loc[tampered_index, "volume"] += 100.0
    tampered_suffix.loc[tampered_index, "turnover"] = (
        tampered_suffix.loc[tampered_index, "volume"]
        * tampered_suffix.loc[tampered_index, "vwap"]
    )
    seam_arguments = {
        "target_session": "2026-08-05",
        "runtime_lock_verified_at": "2026-08-04T20:00:00+09:00",
        "month_source_sealed_at": "2026-08-05T06:30:00+09:00",
        "fit_started_at": "2026-08-05T07:00:00+09:00",
        "fit_completed_at": "2026-08-05T07:05:00+09:00",
        "score_generated_at": "2026-08-05T07:10:00+09:00",
        "source_manifest_sha256": "1" * 64,
        "source_set_sha256": "2" * 64,
        "parsed_shard_set_sha256": "3" * 64,
    }
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    active_phase: str | None = None
    real_decode = runner.decode_canonical_model_price_csv
    real_model_semantic = runner.model_price_semantic_sha256
    real_panel_digest = runner._exact_g0_frame_digest
    real_frame_semantic = runner.semantic_frame_sha256
    real_panel = runner.build_forward_c00_panel
    real_fold = runner._build_fold
    real_validate_fold = runner.validate_fold_manifest
    real_frame_sha = runner._frame_sha
    real_numeric_sha = runner._numeric_sha
    real_score = runner.freeze_c00_top2

    def mark(name: str) -> None:
        if active_phase is not None:
            counts[active_phase][name] += 1

    def counted_decode(*args: Any, **kwargs: Any) -> pd.DataFrame:
        if kwargs.get("label") == "A2 rehearsal model-price round trip":
            mark("full_payload_redecode")
        return real_decode(*args, **kwargs)

    def counted_model_semantic(*args: Any, **kwargs: Any) -> str:
        mark("model_semantic")
        return real_model_semantic(*args, **kwargs)

    def counted_panel_digest(*args: Any, **kwargs: Any) -> dict[str, Any]:
        mark("g0_digest")
        return real_panel_digest(*args, **kwargs)

    def counted_frame_semantic(*args: Any, **kwargs: Any) -> str:
        frame = args[0] if args else kwargs["frame"]
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if not frame.empty and dates.max().normalize() < pd.Timestamp("2026-08-01"):
            mark("training_semantic")
        else:
            mark("other_frame_semantic")
        return real_frame_semantic(*args, **kwargs)

    def counted_panel(*args: Any, **kwargs: Any) -> pd.DataFrame:
        mark("panel")
        return real_panel(*args, **kwargs)

    def counted_fold(
        *args: Any, **kwargs: Any
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        mark("fold_fit")
        return real_fold(*args, **kwargs)

    def counted_validate_fold(*args: Any, **kwargs: Any) -> str:
        mark("fold_validate")
        return real_validate_fold(*args, **kwargs)

    def counted_frame_sha(*args: Any, **kwargs: Any) -> str:
        mark("current_frame_hash")
        return real_frame_sha(*args, **kwargs)

    def counted_numeric_sha(*args: Any, **kwargs: Any) -> str:
        mark("current_numeric_hash")
        return real_numeric_sha(*args, **kwargs)

    def counted_score(*args: Any, **kwargs: Any) -> Any:
        mark("score")
        return real_score(*args, **kwargs)

    def invoke(
        phase: str,
        snapshot: bytes | pd.DataFrame,
        suffix: list[pd.DataFrame],
        *,
        input_kind: str,
        reuse_fold_token: Any | None = None,
    ) -> tuple[dict[str, Any], Any]:
        nonlocal active_phase
        active_phase = phase
        try:
            return runner.build_a2_nonauthority_rehearsal_day(
                snapshot,
                suffix,
                rehearsal_contract=capability,
                input_kind=input_kind,
                reuse_fold_token=reuse_fold_token,
                **seam_arguments,
            )
        finally:
            active_phase = None

    def forbidden_io(label: str) -> Any:
        raise AssertionError(f"timed rehearsal performed {label}")

    try:
        with monkeypatch.context() as timed:
            timed.setattr(runner, "decode_canonical_model_price_csv", counted_decode)
            timed.setattr(runner, "model_price_semantic_sha256", counted_model_semantic)
            timed.setattr(runner, "_exact_g0_frame_digest", counted_panel_digest)
            timed.setattr(runner, "semantic_frame_sha256", counted_frame_semantic)
            timed.setattr(runner, "build_forward_c00_panel", counted_panel)
            timed.setattr(runner, "_build_fold", counted_fold)
            timed.setattr(runner, "validate_fold_manifest", counted_validate_fold)
            timed.setattr(runner, "_frame_sha", counted_frame_sha)
            timed.setattr(runner, "_numeric_sha", counted_numeric_sha)
            timed.setattr(runner, "freeze_c00_top2", counted_score)
            timed.setattr(
                runner,
                "read_json",
                lambda *args, **kwargs: forbidden_io("project JSON read"),
            )
            timed.setattr(
                runner,
                "sha256_file",
                lambda *args, **kwargs: forbidden_io("project file hash"),
            )
            timed.setattr(
                runner.os.path,
                "lexists",
                lambda *args, **kwargs: forbidden_io("canonical authority probe"),
            )
            timed.setattr(
                runner,
                "_latest_required_predictor_source_session",
                lambda *args, **kwargs: forbidden_io("calendar reopen"),
            )
            reference, _ = invoke(
                "reference",
                full,
                [],
                input_kind="full31_reference",
            )
            boundary, boundary_token = invoke(
                "boundary",
                july_snapshot,
                [august_suffix],
                input_kind="month_boundary_compact",
            )
            intramonth, returned_token = invoke(
                "proxy",
                june_snapshot,
                [july_august_suffix],
                input_kind="intramonth_fold_reuse_upper_bound_proxy",
                reuse_fold_token=boundary_token,
            )
            with pytest.raises(
                runner.V18Error,
                match="differs from canonical full-prefix bytes",
            ):
                invoke(
                    "tampered_proxy",
                    june_snapshot,
                    [tampered_suffix],
                    input_kind="intramonth_fold_reuse_upper_bound_proxy",
                    reuse_fold_token=boundary_token,
                )
    finally:
        runner.validate_a2_nonauthority_rehearsal_postflight(capability)

    proof = json.loads(boundary_token.exact_prefix_proof_json)
    assert set(proof) == {
        "schema_version",
        "target_session",
        "latest_required_source_session",
        "model_price_row_count",
        "model_price_csv_byte_count",
        "model_price_csv_sha256",
        "source_manifest_sha256",
        "source_set_sha256",
        "parsed_shard_set_sha256",
        "model_price_semantic_sha256",
        "g0_panel_exact_digest",
        "g0_training_row_count",
        "g0_training_panel_semantic_sha256",
    }
    assert runner.canonical_json_bytes(proof).decode("utf-8") == (
        boundary_token.exact_prefix_proof_json
    )
    assert runner.canonical_json_sha256(proof) == (
        boundary_token.exact_prefix_proof_sha256
    )
    assert returned_token is boundary_token
    assert reference["comparison"] == boundary["comparison"] == intramonth[
        "comparison"
    ]
    for phase in ("reference", "boundary"):
        assert counts[phase]["full_payload_redecode"] == 1
        assert counts[phase]["model_semantic"] == 1
        assert counts[phase]["g0_digest"] == 1
        assert counts[phase]["training_semantic"] == 1
        assert counts[phase]["panel"] == 1
        assert counts[phase]["fold_fit"] == 1
        assert counts[phase]["score"] == 1
    assert counts["proxy"]["full_payload_redecode"] == 0
    assert counts["proxy"]["model_semantic"] == 0
    assert counts["proxy"]["g0_digest"] == 0
    assert counts["proxy"]["training_semantic"] == 0
    assert counts["proxy"]["panel"] == 1
    assert counts["proxy"]["fold_fit"] == 0
    assert counts["proxy"]["fold_validate"] == 2
    assert counts["proxy"]["current_frame_hash"] == 1
    assert counts["proxy"]["current_numeric_hash"] == 2
    assert counts["proxy"]["score"] == 1
    assert counts["proxy"]["other_frame_semantic"] == 1
    assert counts["tampered_proxy"]["full_payload_redecode"] == 0
    assert counts["tampered_proxy"]["model_semantic"] == 0
    assert counts["tampered_proxy"]["g0_digest"] == 0
    assert counts["tampered_proxy"]["training_semantic"] == 0
    assert counts["tampered_proxy"]["panel"] == 0
    assert counts["tampered_proxy"]["fold_fit"] == 0
    assert counts["tampered_proxy"]["score"] == 0
    assert list(tmp_path.iterdir()) == []


def test_fresh_output_root_requires_physical_private_empty_tmp(
    tmp_path: Path,
) -> None:
    fresh = _fresh_private_root(tmp_path)
    assert rehearsal._private_tmp_directory(
        fresh, label="test output", require_empty=True
    ) == fresh

    (fresh / "preexisting").write_bytes(b"x")
    with pytest.raises(rehearsal.RehearsalError, match="not fresh"):
        rehearsal._private_tmp_directory(
            fresh, label="test output", require_empty=True
        )

    public = _fresh_private_root(tmp_path, "public")
    public.chmod(0o755)
    with pytest.raises(rehearsal.RehearsalError, match="private"):
        rehearsal._private_tmp_directory(
            public, label="test output", require_empty=True
        )

    physical = _fresh_private_root(tmp_path, "physical")
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)
    with pytest.raises(rehearsal.RehearsalError, match="physical"):
        rehearsal._private_tmp_directory(
            alias, label="test output", require_empty=True
        )


def test_exclusive_writer_is_confined_private_and_create_once(tmp_path: Path) -> None:
    root = _fresh_private_root(tmp_path)
    child = rehearsal._private_subdirectory(root, "reports")
    assert stat.S_IMODE(child.stat().st_mode) == 0o700
    target = rehearsal._write_bytes_exclusive(root, "reports/health.json", b"{}\n")
    assert target.read_bytes() == b"{}\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    with pytest.raises(rehearsal.RehearsalError, match="publication failed"):
        rehearsal._write_bytes_exclusive(root, "reports/health.json", b"changed\n")
    with pytest.raises(rehearsal.RehearsalError, match="unsafe"):
        rehearsal._write_bytes_exclusive(root, "../escaped.json", b"{}\n")


def test_tree_identity_detects_same_length_content_and_structure_changes(
    tmp_path: Path,
) -> None:
    root = _fresh_private_root(tmp_path)
    target = root / "one.txt"
    target.write_bytes(b"aaaa")
    before = rehearsal._tree_state(root, hash_regular_files=True)
    target.write_bytes(b"bbbb")
    assert rehearsal._tree_state(root, hash_regular_files=True) != before
    after_content = rehearsal._tree_state(root, hash_regular_files=True)
    (root / "two.txt").write_bytes(b"x")
    assert rehearsal._tree_state(root, hash_regular_files=True) != after_content


def test_documentary_aug03_aug04_labels_cannot_be_promoted() -> None:
    registry = {
        "independent_acquisition_authority": False,
        "scientific_authority": False,
        "activation_payload_or_receipt_created": False,
        "manual_provenance_caveat": (
            "Aug03 and Aug04 are documentary/manual and not independent authority"
        ),
    }
    sources = [
        {
            "file_name": "stq_20260803.pdf",
            "official_source_independently_verified": False,
            "manual_operator_provenance_caveat_applies": True,
            "url_evidence": {
                "classification": (
                    "documentary_manual_unbound_page_capture_"
                    "not_independent_acquisition_authority"
                ),
                "independent_acquisition_authority": False,
            },
        },
        {
            "file_name": "stq_20260804.pdf",
            "official_source_independently_verified": False,
            "manual_operator_provenance_caveat_applies": True,
            "url_evidence": {
                "classification": (
                    "documentary_manual_legacy_operator_ledger_"
                    "not_independent_acquisition_authority"
                ),
                "independent_acquisition_authority": False,
            },
        },
    ]
    rehearsal._validate_documentary_labels(registry, sources)
    promoted = dict(registry)
    promoted["independent_acquisition_authority"] = True
    with pytest.raises(rehearsal.RehearsalError, match="authority upgrade"):
        rehearsal._validate_documentary_labels(promoted, sources)
    promoted_source = json.loads(json.dumps(sources))
    promoted_source[1]["url_evidence"]["independent_acquisition_authority"] = True
    with pytest.raises(rehearsal.RehearsalError, match="documentary/manual"):
        rehearsal._validate_documentary_labels(registry, promoted_source)


def test_registry_runtime_evidence_rejects_stale_path_size_or_sha() -> None:
    authorities = {
        "protocol": (rehearsal.runner.PROTOCOL, "protocol_sha256"),
        "runner": (Path(rehearsal.runner.__file__), "runner_sha256"),
        "runtime_lock": (rehearsal.runner.RUNTIME_LOCK, "runtime_lock_sha256"),
    }
    runtime_info: dict[str, str] = {}
    evidence: dict[str, dict[str, object]] = {}
    for role, (path, hash_field) in authorities.items():
        resolved = path.resolve(strict=True)
        payload = resolved.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        runtime_info[hash_field] = digest
        evidence[role] = {
            "path": str(resolved),
            "byte_count": len(payload),
            "sha256": digest,
        }
    registry = {"evidence": evidence}
    rehearsal._validate_registry_runtime_evidence(registry, runtime_info)
    for role in authorities:
        for field in ("path", "byte_count", "sha256"):
            stale = json.loads(json.dumps(registry))
            if field == "path":
                stale["evidence"][role][field] += ".stale"
            elif field == "byte_count":
                stale["evidence"][role][field] += 1
            else:
                stale["evidence"][role][field] = "f" * 64
            with pytest.raises(rehearsal.RehearsalError, match="evidence is stale"):
                rehearsal._validate_registry_runtime_evidence(stale, runtime_info)


def test_health_json_rejects_values_but_accepts_aggregate_hashes() -> None:
    safe = {
        "status": "pass",
        "comparison_sha256": "a" * 64,
        "elapsed_seconds": 1.25,
        "scenarios": [{"input_kind": "month_boundary_compact"}],
    }
    assert rehearsal._health_only(safe) is safe
    for key, value in (
        ("code", "1001"),
        ("name", "issuer"),
        ("model_score", 1.0),
        ("return", 2.0),
        ("outcome", "up"),
        ("decision", "rank1"),
    ):
        with pytest.raises(rehearsal.RehearsalError, match="forbidden"):
            rehearsal._health_only({"nested": [{key: value}]})


def test_timed_wrapper_contains_no_file_repo_network_or_preflight_work() -> None:
    source = inspect.getsource(rehearsal._timed_seam)
    tree = ast.parse(source)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert calls == {"perf_counter_ns", "_build_seam", "RehearsalError"}
    assert not {
        "open",
        "read_bytes",
        "write_bytes",
        "validate_runtime_lock",
        "validate_protocol",
        "prepare_a2_nonauthority_rehearsal_contract",
    } & calls


def test_cold_path_outer_times_both_seams_and_reuses_only_opaque_token() -> None:
    source = inspect.getsource(rehearsal._run_cold)
    tree = ast.parse(source)
    timed_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_timed_seam"
    ]
    assert len(timed_calls) == 2
    by_kind = {
        next(
            keyword.value.value
            for keyword in call.keywords
            if keyword.arg == "input_kind"
            and isinstance(keyword.value, ast.Constant)
        ): call
        for call in timed_calls
    }
    assert set(by_kind) == {
        "month_boundary_compact",
        "intramonth_fold_reuse_upper_bound_proxy",
    }
    boundary = by_kind["month_boundary_compact"]
    intramonth = by_kind["intramonth_fold_reuse_upper_bound_proxy"]
    assert not any(keyword.arg == "reuse_fold_token" for keyword in boundary.keywords)
    reuse = next(
        keyword.value
        for keyword in intramonth.keywords
        if keyword.arg == "reuse_fold_token"
    )
    assert isinstance(reuse, ast.Name) and reuse.id == "boundary_token"
    assert "intramonth_token is not boundary_token" in source
    assert "reference[\"exact_comparison\"]" in source
    assert "boundary_comparison" in source
    assert "intramonth_comparison" in source


def test_reference_and_cold_paths_always_revoke_capability_in_finally() -> None:
    for function in (rehearsal._prepare_reference, rehearsal._run_cold):
        tree = ast.parse(inspect.getsource(function))
        final_calls = {
            node.func.id
            for candidate in ast.walk(tree)
            if isinstance(candidate, ast.Try)
            for node in candidate.finalbody
            for node in ast.walk(node)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "_postflight_runner_contract" in final_calls
    timed_source = inspect.getsource(rehearsal._timed_seam)
    assert "postflight" not in timed_source


def test_postflight_hash_must_exact_match_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Capability:
        preflight_sha256 = "a" * 64

    capability = Capability()
    monkeypatch.setattr(
        rehearsal.runner,
        "validate_a2_nonauthority_rehearsal_postflight",
        lambda value: value.preflight_sha256,
    )
    assert rehearsal._postflight_runner_contract(capability) == "a" * 64
    monkeypatch.setattr(
        rehearsal.runner,
        "validate_a2_nonauthority_rehearsal_postflight",
        lambda value: "b" * 64,
    )
    with pytest.raises(rehearsal.RehearsalError, match="preflight/postflight"):
        rehearsal._postflight_runner_contract(capability)


def test_execution_module_has_no_network_or_guard_monkeypatch_surface() -> None:
    path = Path(rehearsal.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    called_attributes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called_attributes.add(node.func.attr)
    assert not {
        "socket",
        "urllib",
        "http",
        "requests",
        "subprocess",
    } & imported
    assert not {
        "setattr",
        "delattr",
        "urlopen",
        "request",
        "create_activation_payload",
        "create_activation_receipt",
        "activate",
        "prepare_checkpoint",
        "publish_checkpoint",
        "materialize_checkpoint_decision",
        "build_outcome_manifest",
        "evaluate",
        "build_result",
    } & called_attributes
    assert "monkeypatch" not in source
    public_execution_parameters = {
        parameter
        for function in (rehearsal._prepare_reference, rehearsal._run_cold)
        for parameter in inspect.signature(function).parameters
    }
    assert not {
        "parser",
        "panel",
        "panel_builder",
        "fold",
        "fold_builder",
        "bundle",
        "scorer",
        "model",
        "callable",
    } & public_execution_parameters


def test_registered_fixed_source_and_gate_identity() -> None:
    assert rehearsal.EXPECTED_SOURCE_COUNT == 248
    assert rehearsal.EXPECTED_SOURCE_BYTE_COUNT == 1_265_218_618
    assert rehearsal.EXPECTED_RAW_SOURCE_SET_SHA256 == (
        "461450f1916440907eddcdac4443d7085954902c51d87b49bff76669c8b0a540"
    )
    assert rehearsal.TARGET_SESSION == "2026-08-05"
    assert rehearsal.JULY_SNAPSHOT_CUTOFF == "2026-07-31"
    assert rehearsal.JUNE_SNAPSHOT_CUTOFF == "2026-06-30"
    assert rehearsal.BOUNDARY_BUDGET_SECONDS == 300.0
    assert rehearsal.INTRAMONTH_BUDGET_SECONDS == 120.0
    assert set(rehearsal.FIXED_REPLAY_TIMESTAMPS) == {
        "runtime_lock_verified_at",
        "month_source_sealed_at",
        "fit_started_at",
        "fit_completed_at",
        "score_generated_at",
    }
