from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from ops import model_v18_operations as operations


def _roots(tmp_path: Path) -> dict[str, Path]:
    roots = {
        "ops": tmp_path / "ops",
        "predictor_raw": tmp_path / "predictor-raw",
        "predictor_derived": tmp_path / "predictor-derived",
        "outcome_raw": tmp_path / "outcome-raw",
        "checkpoint_core": tmp_path / "checkpoint-core",
    }
    operations.initialise_layout(
        ops_root=roots["ops"],
        predictor_raw_store_root=roots["predictor_raw"],
        predictor_derived_store_root=roots["predictor_derived"],
        outcome_raw_store_root=roots["outcome_raw"],
        checkpoint_core_store_root=roots["checkpoint_core"],
    )
    return roots


def _timestamp(minutes_ago: int = 1) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    ).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _as_of() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _write_pdf(path: Path, suffix: bytes = b"manual-test") -> None:
    path.write_bytes(b"%PDF-1.7\n" + suffix + b"\n")


def _daily_url(file_name: str) -> str:
    return (
        "https://www.jpx.co.jp/markets/statistics-equities/daily/"
        f"testtoken-att/{file_name}"
    )


def _register_daily(
    roots: dict[str, Path],
    tmp_path: Path,
    file_name: str,
    *,
    minutes_ago: int = 2,
) -> dict[str, object]:
    source = tmp_path / file_name
    _write_pdf(source, file_name.encode())
    return operations.register_source(
        ops_root=roots["ops"],
        source_file=source,
        source_url=_daily_url(file_name),
        received_at=_timestamp(minutes_ago),
        kind="daily",
    )


def _install_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    first: str = "2026-08-06",
    terminal: str = "2026-08-10",
) -> Path:
    sessions = operations._calendar()
    if operations._parse_date(terminal, "test terminal") not in sessions:
        first_position = sessions.index(
            operations._parse_date(first, "test first")
        )
        terminal = sessions[min(first_position + 2, len(sessions) - 1)].isoformat()
    path = (tmp_path / "canonical-activation-context.json").resolve()
    path.write_text(
        json.dumps(
            {
                "first_counted_session": first,
                "terminal_session": terminal,
                "activation_payload_sha256": "a" * 64,
                "activation_receipt_sha256": "b" * 64,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    monkeypatch.setattr(operations, "ACTIVATION_CONTEXT", path)
    return path


def _commands(plan: dict[str, object]) -> list[str]:
    return [step["argv"][3] for step in plan["steps"]]


def _flag_values(argv: list[str], flag: str) -> list[str]:
    return [argv[index + 1] for index, item in enumerate(argv[:-1]) if item == flag]


def test_init_records_four_pairwise_disjoint_private_roots(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    layout = operations._load_layout(roots["ops"])

    paths = {
        Path(layout["operations_root"]["path"]),
        *(
            Path(record["path"])
            for record in layout["stores"].values()
        ),
    }
    assert len(paths) == 5
    assert set(layout["stores"]) == {
        "predictor_raw",
        "predictor_derived",
        "outcome_raw",
        "checkpoint_core",
    }
    assert layout["raw_source_provenance"] == operations._provenance_envelope()
    assert layout["github_token_persisted"] is False
    assert layout["runner_execution_enabled"] is False
    assert all((path.stat().st_mode & 0o777) == 0o700 for path in paths)
    assert list(roots["predictor_raw"].iterdir()) == []
    assert list(roots["predictor_derived"].iterdir()) == []
    assert list(roots["outcome_raw"].iterdir()) == []
    assert list(roots["checkpoint_core"].iterdir()) == []


def test_init_rejects_nested_or_aliasing_roots(tmp_path: Path) -> None:
    base = tmp_path / "nested"
    with pytest.raises(operations.OpsError, match="disjoint"):
        operations.initialise_layout(
            ops_root=base,
            predictor_raw_store_root=base / "predictor",
            predictor_derived_store_root=tmp_path / "derived",
            outcome_raw_store_root=tmp_path / "outcome",
            checkpoint_core_store_root=tmp_path / "checkpoint",
        )


def test_manual_registration_is_attested_create_once_and_hash_chained(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    source = tmp_path / "stq_20260805.pdf"
    _write_pdf(source)
    received = _timestamp()
    url = _daily_url(source.name)

    first = operations.register_source(
        ops_root=roots["ops"],
        source_file=source,
        source_url=url,
        received_at=received,
        kind="daily",
    )
    repeated = operations.register_source(
        ops_root=roots["ops"],
        source_file=source,
        source_url=url,
        received_at=received,
        kind="daily",
    )

    assert repeated == first
    assert first["provenance_mode"] == "manual_operator_attested_v1"
    assert first["official_source_verified"] is False
    assert first["source_url_label"] == url
    assert "official_url" not in first
    assert first["sequence_number"] == 0
    assert first["previous_record_sha256"] == "0" * 64
    assert len(operations.load_acquisitions(roots["ops"])) == 1
    retained = roots["ops"] / first["staged_relative_path"]
    assert retained.read_bytes() == source.read_bytes()
    assert (retained.stat().st_mode & 0o777) == 0o600


@pytest.mark.parametrize(
    "url",
    (
        "https://jpx.co.jp/markets/statistics-equities/daily/x-att/stq_20260805.pdf",
        "https://www.jpx.co.jp/example/stq_20260805.pdf",
        "https://www.jpx.co.jp/markets/statistics-equities/daily/x-att/stq_20260805.pdf?q=1",
    ),
)
def test_manual_registration_rejects_noncanonical_daily_url_labels(
    tmp_path: Path, url: str
) -> None:
    roots = _roots(tmp_path)
    source = tmp_path / "stq_20260805.pdf"
    _write_pdf(source)

    with pytest.raises(operations.OpsError, match="URL label|noncanonical"):
        operations.register_source(
            ops_root=roots["ops"],
            source_file=source,
            source_url=url,
            received_at=_timestamp(),
            kind="daily",
        )


def test_static_surface_has_no_auto_download_split_or_failure_cash_paths() -> None:
    source = Path(operations.__file__).read_text(encoding="utf-8")
    documentation = (
        operations.ROOT / "docs/model_v18_operations.md"
    ).read_text(encoding="utf-8")
    surfaces = source + "\n" + documentation

    for forbidden in (
        "urllib",
        "urlopen",
        "HTMLParser",
        "discover-source",
        "discover_and_download",
        "prepare-source-manifest",
        "prepare-source-failure-manifest",
        "prepare-fold",
        "prepare-state",
        "prepare-top2",
        "prepare-checkpoint",
        "os.walk",
    ):
        assert forbidden not in surfaces
    assert "manual_operator_attested_v1" in source
    assert '"official_source_verified": False' in source


def test_store_plan_uses_all_runtime_lock_unsets_and_absolute_python(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    plan = operations.build_store_plan(ops_root=roots["ops"])
    runtime_lock = json.loads(operations.RUNTIME_LOCK.read_text(encoding="utf-8"))
    expected_unset = sorted(runtime_lock["elf_closure"]["loader_environment"])

    assert _commands(plan) == ["validate-runtime", "prepare-operational-stores"]
    for step in plan["steps"]:
        assert step["environment_unset"] == expected_unset
        assert "PYTHONPATH" in step["environment_unset"]
        assert "GITHUB_TOKEN" not in step["environment_unset"]
        assert Path(step["argv"][0]).is_absolute()
        assert Path(step["argv"][0]).is_file()
        assert step["github_token_value_persisted"] is False
    payload = json.dumps(plan, sort_keys=True)
    assert "ghp_" not in payload
    store_argv = plan["steps"][1]["argv"]
    assert set(_flag_values(store_argv, "--predictor-raw-store-root")) == {
        str(roots["predictor_raw"])
    }
    assert set(_flag_values(store_argv, "--predictor-derived-store-root")) == {
        str(roots["predictor_derived"])
    }
    assert set(_flag_values(store_argv, "--outcome-raw-store-root")) == {
        str(roots["outcome_raw"])
    }
    assert set(_flag_values(store_argv, "--checkpoint-core-store-root")) == {
        str(roots["checkpoint_core"])
    }


def test_cache_plan_uses_fixed_ordered_anchor_and_only_predictor_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    expected = [
        operations.RegistryItem(
            "stq_20260803.pdf", "daily", operations.date(2026, 8, 3)
        ),
        operations.RegistryItem(
            "stq_20260804.pdf", "daily", operations.date(2026, 8, 4)
        ),
    ]
    records = [
        {
            "file_name": item.file_name,
            "staged_relative_path": f"inbox/daily/{item.file_name}",
            "source_url_label": _daily_url(item.file_name),
        }
        for item in expected
    ]
    monkeypatch.setattr(operations, "expected_registry", lambda latest: expected)
    monkeypatch.setattr(operations, "load_acquisitions", lambda ops_root: records)

    plan = operations.build_predictor_cache_plan(ops_root=roots["ops"])
    argv = plan["steps"][1]["argv"]

    assert _commands(plan) == ["validate-runtime", "prepare-predictor-cache"]
    assert _flag_values(argv, "--source-file-name") == [
        "stq_20260803.pdf",
        "stq_20260804.pdf",
    ]
    assert _flag_values(argv, "--through") == ["2026-08-04"]
    assert "--predictor-raw-store-root" in argv
    assert "--predictor-derived-store-root" in argv
    assert "--outcome-raw-store-root" not in argv
    assert "--checkpoint-core-store-root" not in argv
    assert plan["anchor_source_count"] == 2


def test_activation_plans_use_only_canonical_default_artifact_paths(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    plan = operations.build_activation_plan(
        ops_root=roots["ops"],
        stage="payload",
        commit_sha="1" * 40,
        predictor_cache_anchor_manifest_object_key=(
            "model_v18_shoulder_state/cache-anchor/fixed/manifest.json"
        ),
    )
    argv = plan["steps"][1]["argv"]

    assert _commands(plan) == [
        "validate-runtime",
        "create-activation-payload",
    ]
    assert "--output" not in argv
    assert "--activation-context" not in argv
    assert plan["canonical_activation_context"] == (
        "research/model_v18_shoulder_state_activation_context.json"
    )

    receipt = operations.build_activation_plan(
        ops_root=roots["ops"],
        stage="receipt",
        commit_sha="2" * 40,
    )
    preflight = operations.build_activation_plan(
        ops_root=roots["ops"],
        stage="preflight",
        commit_sha="3" * 40,
    )
    assert "--payload" not in receipt["steps"][1]["argv"]
    assert "--output" not in receipt["steps"][1]["argv"]
    assert "--payload" not in preflight["steps"][1]["argv"]
    assert "--receipt" not in preflight["steps"][1]["argv"]
    assert "--output" not in preflight["steps"][1]["argv"]


def test_first_daily_plan_passes_only_post_anchor_suffix_and_canonical_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    _install_context(monkeypatch, tmp_path, first="2026-08-06")
    _register_daily(roots, tmp_path, "stq_20260805.pdf")

    plan = operations.build_daily_plan(
        ops_root=roots["ops"],
        session="2026-08-06",
        as_of=_as_of(),
    )
    argv = plan["steps"][1]["argv"]

    assert _commands(plan) == [
        "validate-runtime",
        "prepare-day",
        "publish-checkpoint",
        "decide",
    ]
    assert _flag_values(argv, "--new-predictor-file-name") == [
        "stq_20260805.pdf"
    ]
    assert _flag_values(argv, "--new-predictor-received-at")
    assert "--activation-context" not in argv
    assert "--source-pdf" not in argv
    assert plan["source_failure_cash_path_present"] is False
    assert plan["publication_retry_allowed"] is False
    assert plan["steps"][1]["retry_policy"] == "same_inputs_exact_retry"
    assert plan["steps"][2]["retry_policy"] == (
        "no_blind_retry_after_possible_ref_update"
    )


def test_later_daily_plan_passes_exact_d_minus_one_not_cumulative_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    _install_context(monkeypatch, tmp_path, first="2026-08-06")
    _register_daily(roots, tmp_path, "stq_20260805.pdf", minutes_ago=3)
    _register_daily(roots, tmp_path, "stq_20260806.pdf", minutes_ago=2)

    plan = operations.build_daily_plan(
        ops_root=roots["ops"],
        session="2026-08-07",
        as_of=_as_of(),
    )
    argv = plan["steps"][1]["argv"]

    assert _flag_values(argv, "--new-predictor-file-name") == [
        "stq_20260806.pdf"
    ]
    assert plan["incremental_source_files"] == ["stq_20260806.pdf"]
    assert "stq_20260805.pdf" not in argv


def test_incomplete_daily_suffix_writes_no_plan_or_cash_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    _install_context(monkeypatch, tmp_path, first="2026-08-06")

    with pytest.raises(operations.OpsError, match="incomplete"):
        operations.build_daily_plan(
            ops_root=roots["ops"],
            session="2026-08-06",
            as_of=_as_of(),
        )

    assert not (roots["ops"] / "plans/daily-2026-08-06.json").exists()
    assert list((roots["ops"] / "plans").iterdir()) == []


def test_daily_plan_exact_retry_accepts_same_bytes_rejects_changed_as_of(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    _install_context(monkeypatch, tmp_path, first="2026-08-06")
    _register_daily(roots, tmp_path, "stq_20260805.pdf")
    observed = _as_of()

    first = operations.build_daily_plan(
        ops_root=roots["ops"], session="2026-08-06", as_of=observed
    )
    repeated = operations.build_daily_plan(
        ops_root=roots["ops"], session="2026-08-06", as_of=observed
    )
    assert repeated == first

    with pytest.raises(operations.OpsError, match="conflicts"):
        operations.build_daily_plan(
            ops_root=roots["ops"],
            session="2026-08-06",
            as_of=_as_of(),
        )


def test_terminal_plan_is_finalize_then_evaluate_with_four_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    context_path = _install_context(monkeypatch, tmp_path)
    context = json.loads(context_path.read_text(encoding="utf-8"))
    terminal = context["terminal_session"]
    _register_daily(
        roots,
        tmp_path,
        f"stq_{terminal.replace('-', '')}.pdf",
    )

    plan = operations.build_terminal_plan(
        ops_root=roots["ops"],
        session=terminal,
        as_of=_as_of(),
    )

    assert _commands(plan) == [
        "validate-runtime",
        "finalize-terminal",
        "evaluate",
    ]
    finalize = plan["steps"][1]["argv"]
    evaluation = plan["steps"][2]["argv"]
    assert "--activation-context" not in finalize
    assert "--activation-context" not in evaluation
    for _, flag in operations.STORE_ARGUMENTS:
        assert flag in finalize
        assert flag in evaluation
    assert plan["terminal_finalize_precedes_evaluate"] is True


def test_abort_plan_is_one_rootless_outcome_blind_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    _install_context(monkeypatch, tmp_path)

    plan = operations.build_abort_plan(
        ops_root=roots["ops"],
        failure_reason="source_integrity_failure",
        integrity_stage="source_ingestion",
    )
    argv = plan["steps"][0]["argv"]
    payload = json.dumps(plan, sort_keys=True)

    assert _commands(plan) == ["abort"]
    assert "--activation-context" not in argv
    assert all(flag not in argv for _, flag in operations.STORE_ARGUMENTS)
    assert "evaluate" not in argv
    assert "outcome_manifest" not in payload
    assert plan["store_arguments_included"] is False
    assert plan["performance_paths_included"] is False

    with pytest.raises(operations.OpsError, match="not allowed"):
        operations.build_abort_plan(
            ops_root=roots["ops"],
            failure_reason="audit_integrity_failure",
            integrity_stage="source_ingestion",
        )


def test_health_is_referenced_only_and_never_enumerates_external_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    _install_context(monkeypatch, tmp_path, first="2026-08-06")
    _register_daily(roots, tmp_path, "stq_20260805.pdf")

    report = operations.health_report(
        ops_root=roots["ops"], session="2026-08-06"
    )
    payload = json.dumps(report, sort_keys=True)

    assert report["scope"] == (
        "referenced_source_and_exact_path_metadata_only"
    )
    assert report["external_store_children_enumerated"] is False
    assert report["scientific_artifact_bytes_read"] is False
    assert report["performance_semantics_read"] is False
    assert report["outcome_paths_opened"] is False
    assert "outcome_manifest" not in payload
    assert "plain_file_count" not in payload
    assert set(report["external_root_presence"]) == {
        "predictor_raw",
        "predictor_derived",
        "outcome_raw",
        "checkpoint_core",
    }


def test_emitted_commands_and_flags_exist_in_current_runner_cli_help() -> None:
    command_flags = {
        "validate-runtime": set(),
        "prepare-operational-stores": {
            "--predictor-raw-store-root",
            "--predictor-derived-store-root",
            "--outcome-raw-store-root",
            "--checkpoint-core-store-root",
        },
        "prepare-predictor-cache": {
            "--source-pdf",
            "--source-file-name",
            "--source-url",
            "--through",
            "--predictor-raw-store-root",
            "--predictor-derived-store-root",
        },
        "create-activation-payload": {
            "--preregistration-commit-sha",
            "--predictor-raw-store-root",
            "--predictor-derived-store-root",
            "--predictor-cache-anchor-manifest-object-key",
        },
        "create-activation-receipt": {"--payload-commit-sha"},
        "preflight": {"--receipt-commit-sha"},
        "prepare-day": {
            "--session",
            "--new-predictor-pdf",
            "--new-predictor-file-name",
            "--new-predictor-url",
            "--new-predictor-received-at",
            "--predictor-raw-store-root",
            "--predictor-derived-store-root",
            "--outcome-raw-store-root",
            "--checkpoint-core-store-root",
        },
        "publish-checkpoint": {"--session"},
        "decide": {"--session", "--checkpoint-core-store-root"},
        "finalize-terminal": {
            "--source-pdf",
            "--source-file-name",
            "--source-url",
            "--source-received-at",
            "--predictor-raw-store-root",
            "--predictor-derived-store-root",
            "--outcome-raw-store-root",
            "--checkpoint-core-store-root",
        },
        "evaluate": {
            "--predictor-raw-store-root",
            "--predictor-derived-store-root",
            "--outcome-raw-store-root",
            "--checkpoint-core-store-root",
        },
        "abort": {"--failure-reason", "--integrity-stage"},
    }
    environment = dict(os.environ)
    for name in operations._runtime_launch_contract()[1]:
        environment.pop(name, None)

    for command, flags in command_flags.items():
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(operations.RUNNER),
                command,
                "--help",
            ],
            cwd=operations.ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        help_flags = set(re.findall(r"--[a-z0-9-]+", completed.stdout))
        assert flags <= help_flags, (command, flags - help_flags)
