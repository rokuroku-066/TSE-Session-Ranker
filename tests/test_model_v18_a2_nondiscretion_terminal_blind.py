from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import pandas as pd
import pytest

from research import model_v18_shoulder_state_runner as runner


TARGET_SESSION = "2026-08-05"
SOURCE_SHA = "1" * 64
FOLD_SHA = "2" * 64
BUNDLE_SHA = "3" * 64


def _private_directory(path: Path, mode: int = 0o700) -> Path:
    path.mkdir(mode=mode, parents=True, exist_ok=True)
    path.chmod(mode)
    return path


def _tree_snapshot(root: Path) -> tuple[tuple[Any, ...], ...]:
    rows: list[tuple[Any, ...]] = []
    for path in sorted([root, *root.rglob("*")], key=lambda item: item.as_posix()):
        metadata = path.lstat()
        rows.append(
            (
                "." if path == root else path.relative_to(root).as_posix(),
                stat.S_IFMT(metadata.st_mode),
                stat.S_IMODE(metadata.st_mode),
                metadata.st_uid,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                path.read_bytes() if stat.S_ISREG(metadata.st_mode) else None,
                os.readlink(path) if stat.S_ISLNK(metadata.st_mode) else None,
            )
        )
    return tuple(rows)


def _activation() -> dict[str, Any]:
    return {
        "activation_payload_sha256": "4" * 64,
        "activation_receipt_sha256": "5" * 64,
        "activation_receipt_commit_sha": "a" * 40,
        "activation_receipt_commit_url": "https://example.invalid/commit/a",
        "activation_receipt_commit_committed_at": (
            f"{TARGET_SESSION}T06:50:00+09:00"
        ),
        "activation_receipt_commit_observed_at": (
            f"{TARGET_SESSION}T07:00:00+09:00"
        ),
        "branch_tip_sha_when_receipt_observed": "b" * 40,
        "activation_receipt_file_sha256": "6" * 64,
        "receipt_commit_observation": {"kind": "test"},
        "receipt_branch_observation": {"kind": "test"},
        "activation_receipt_workflow_run_id": 18,
        "activation_receipt_workflow_run_updated_at": (
            f"{TARGET_SESSION}T06:55:00+09:00"
        ),
        "activation_receipt_workflow_run_observed_at": (
            f"{TARGET_SESSION}T07:00:00+09:00"
        ),
        "receipt_workflow_run_observation": {"kind": "test"},
        "first_counted_session": TARGET_SESSION,
        "terminal_session": "2026-08-06",
    }


def _source(*, complete: bool = True) -> dict[str, Any]:
    return {
        "source_complete": complete,
        "sealed_at": f"{TARGET_SESSION}T07:20:00+09:00",
        "previous_counted_target_session": None,
        "previous_counted_source_manifest_sha256": None,
    }


def _state(
    *,
    available: bool = True,
    value: float | None = 0.4,
    selected_rank: int | None = 1,
) -> dict[str, Any]:
    return {
        "activation_payload_sha256": "4" * 64,
        "activation_receipt_sha256": "5" * 64,
        "target_month": "2026-08",
        "created_at": f"{TARGET_SESSION}T07:10:00+09:00",
        "state_manifest_sha256": "7" * 64,
        "c00_fold_manifest_sha256": FOLD_SHA,
        "fold_model_bundle_file_sha256": BUNDLE_SHA,
        "state_available": available,
        "state_value_pct": value,
        "selected_source_rank": selected_rank,
        "three_prior_calendar_months": ["2026-05", "2026-06", "2026-07"],
        "three_complete_pair_day_counts": [17, 19, 18],
        "three_month_medians_pct": [0.4, 0.5, 0.1],
    }


def _fold() -> dict[str, Any]:
    return {
        "fold_manifest_sha256": FOLD_SHA,
        "fold_model_bundle_file_sha256": BUNDLE_SHA,
        "fit_completed_at": f"{TARGET_SESSION}T07:30:00+09:00",
        "sealed_at": f"{TARGET_SESSION}T07:31:00+09:00",
    }


def _score_pair() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": [TARGET_SESSION, TARGET_SESSION],
            "source_rank": [1, 2],
            "code": ["1001", "1002"],
            "name": ["rank one", "rank two"],
            "model_score": [1.0, 0.5],
            "feature_source_max_date": ["2026-08-04", "2026-08-04"],
            "score_generated_at": [f"{TARGET_SESSION}T08:00:00+09:00"] * 2,
            "runtime_lock_sha256": [runner.RUNTIME_LOCK_SHA256] * 2,
            "runtime_lock_verified_at": [
                f"{TARGET_SESSION}T07:59:00+09:00"
            ]
            * 2,
            "source_manifest_sha256": [SOURCE_SHA] * 2,
            "c00_fold_manifest_sha256": [FOLD_SHA] * 2,
        },
        columns=runner.SCORE_FIELDS,
    )


def _patch_pure_decision_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    def records(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, pd.DataFrame):
            return [dict(item) for item in value.to_dict(orient="records")]
        return [dict(item) for item in value]

    monkeypatch.setattr(runner, "validate_decision_records", records)
    monkeypatch.setattr(
        runner, "validate_activation_context", lambda value: dict(value)
    )
    monkeypatch.setattr(
        runner,
        "validate_source_manifest",
        lambda value, **kwargs: (dict(value), SOURCE_SHA),
    )
    monkeypatch.setattr(runner, "validate_state_manifest", lambda value: dict(value))
    monkeypatch.setattr(
        runner,
        "validate_fold_manifest",
        lambda fold, bundle: FOLD_SHA,
    )
    monkeypatch.setattr(
        runner,
        "load_registered_calendar",
        lambda: pd.DatetimeIndex([TARGET_SESSION, "2026-08-06"]),
    )
    monkeypatch.setattr(
        runner,
        "_runtime_verified_timestamp",
        lambda value=None, **kwargs: runner._timestamp(
            value or f"{TARGET_SESSION}T07:59:00+09:00",
            "runtime_lock_verified_at",
        ),
    )


def _build_core(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
    scores: pd.DataFrame | None = None,
    fold: Mapping[str, Any] | None = None,
    bundle: Mapping[str, Any] | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    _patch_pure_decision_dependencies(monkeypatch)
    value = runner.build_decision_ledger(
        _score_pair() if scores is None else scores,
        session_date=TARGET_SESSION,
        source_manifest=_source() if source is None else source,
        state_manifest=_state() if state is None else state,
        activation=_activation(),
        computed_at=f"{TARGET_SESSION}T08:10:00+09:00",
        runtime_lock_verified_at=f"{TARGET_SESSION}T07:59:00+09:00",
        fold_manifest=_fold() if fold is None else fold,
        fold_model_bundle={"registered": True} if bundle is None else bundle,
        failure_reason=failure_reason,
        return_decision_core=True,
        _resume_exact_runtime=True,
    )
    assert isinstance(value, dict)
    return value


def test_materialized_decision_record_carries_exact_score_session_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _build_core(monkeypatch)
    binding = {
        "decision_materialized_at": f"{TARGET_SESSION}T08:11:00+09:00",
        "checkpoint_resolution": "primary",
        "checkpoint_resolution_reason": "primary_commitment_timely",
        "checkpoint_core_sha256": runner.canonical_json_sha256(core),
        "checkpoint_proposal_path": (
            "research/model_v18_shoulder_state_checkpoint_proposals/"
            f"{TARGET_SESSION}/primary.json"
        ),
        "checkpoint_proposal_file_sha256": "8" * 64,
        "checkpoint_proposal_sha256": "9" * 64,
        "checkpoint_commit_sha": "c" * 40,
        "checkpoint_commit_url": "https://example.invalid/commit/c",
        "checkpoint_commit_committed_at": f"{TARGET_SESSION}T08:05:00+09:00",
        "checkpoint_commit_observed_at": f"{TARGET_SESSION}T08:06:00+09:00",
        "checkpoint_branch_tip_sha_when_observed": "c" * 40,
        "checkpoint_commit_observation": {"kind": "test"},
        "checkpoint_branch_observation": {"kind": "test"},
        "checkpoint_workflow_run_id": 19,
        "checkpoint_workflow_run_updated_at": f"{TARGET_SESSION}T08:07:00+09:00",
        "checkpoint_workflow_run_observed_at": f"{TARGET_SESSION}T08:08:00+09:00",
        "checkpoint_workflow_run_observation": {"kind": "test"},
    }
    monkeypatch.setattr(
        runner,
        "validate_decision_records",
        lambda rows: runner.validate_hash_chain(
            rows, required_fields=runner.DECISION_FIELDS
        ),
    )
    rows = runner.build_decision_ledger(
        _score_pair(),
        session_date=TARGET_SESSION,
        source_manifest=_source(),
        state_manifest=_state(),
        activation=_activation(),
        computed_at=f"{TARGET_SESSION}T08:10:00+09:00",
        runtime_lock_verified_at=f"{TARGET_SESSION}T07:59:00+09:00",
        fold_manifest=_fold(),
        fold_model_bundle={"registered": True},
        checkpoint_binding=binding,
        _resume_exact_runtime=True,
    )
    assert isinstance(rows, list) and len(rows) == 1
    record = rows[0]
    score_fields = (
        "score_session_file_sha256",
        "score_session_semantic_sha256",
        "score_session_set_sha256",
    )
    assert all(field in runner.DECISION_FIELDS for field in score_fields)
    assert runner._decision_core_from_row(record) == core
    assert record["score_session_file_sha256"] == core[
        "score_session_file_sha256"
    ]
    assert record["score_session_semantic_sha256"] == core[
        "score_session_semantic_sha256"
    ]
    assert record["score_session_set_sha256"] == runner.canonical_json_sha256(
        [
            {
                "session_date": TARGET_SESSION,
                "file_sha256": core["score_session_file_sha256"],
                "semantic_sha256": core["score_session_semantic_sha256"],
            }
        ]
    )
    for field in score_fields:
        tampered = dict(record)
        tampered[field] = "f" * 64
        with pytest.raises(runner.V18Error, match="self-hash mismatch"):
            runner.validate_hash_chain(
                [tampered], required_fields=runner.DECISION_FIELDS
            )
        missing = {key: value for key, value in record.items() if key != field}
        with pytest.raises(runner.V18Error, match="fields differ"):
            runner.validate_hash_chain(
                [missing], required_fields=runner.DECISION_FIELDS
            )


def _invalid_core_derivation(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> dict[str, Any]:
    _patch_pure_decision_dependencies(monkeypatch)
    source = _source(
        complete=failure not in {"incomplete_source", "partial_source"}
    )
    if failure == "partial_source":
        source = {
            **source,
            "partial_session": True,
            "failure_reason": "partial_predictor_source",
        }
    scores: pd.DataFrame | None = _score_pair()
    fold: Mapping[str, Any] | None = _fold()
    bundle: Mapping[str, Any] | None = {"registered": True}
    if failure == "missing_fold":
        fold = None
    elif failure == "missing_bundle":
        bundle = None
    elif failure == "missing_fold_pair":
        fold = None
        bundle = None
    elif failure == "missing_scores":
        scores = None
    elif failure == "one_score_row":
        scores = scores.iloc[[0]].reset_index(drop=True)

    # Unlike _build_core(), None is meaningful here and must not be replaced
    # by a valid default.
    value = runner.build_decision_ledger(
        scores,
        session_date=TARGET_SESSION,
        source_manifest=source,
        state_manifest=_state(),
        activation=_activation(),
        computed_at=f"{TARGET_SESSION}T08:10:00+09:00",
        runtime_lock_verified_at=f"{TARGET_SESSION}T07:59:00+09:00",
        fold_manifest=fold,
        fold_model_bundle=bundle,
        return_decision_core=True,
        _resume_exact_runtime=True,
    )
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("incomplete_source", "incomplete/missing predictor source"),
        ("partial_source", "incomplete/missing predictor source"),
        ("missing_fold", "presence differs"),
        ("missing_bundle", "presence differs"),
        ("missing_fold_pair", "missing monthly fold/bundle"),
        ("missing_scores", "missing exact target score pair"),
        ("one_score_row", "exactly ranks one and two"),
    ],
)
def test_checkpoint_prerequisite_failure_is_zero_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    repository = _private_directory(tmp_path / "isolated-repository")
    proposal_root = _private_directory(repository / "research" / "proposals")
    external_root = _private_directory(tmp_path / "isolated-external")
    _private_directory(
        external_root / "model_v18_shoulder_state" / "checkpoint-core"
    )
    decision_ledger = repository / "decisions.jsonl"
    monkeypatch.setattr(runner, "ROOT", repository)
    monkeypatch.setattr(runner, "CHECKPOINT_PROPOSAL_DIR", proposal_root)
    monkeypatch.setattr(runner, "DECISION_LEDGER", decision_ledger)
    monkeypatch.setattr(runner, "_STRICT_RUNTIME_ACTIVE", True)
    monkeypatch.setattr(
        runner, "_validate_startup_and_module_closure", lambda *, phase: None
    )
    monkeypatch.setattr(
        runner,
        "_derive_canonical_checkpoint_core",
        lambda target, existing: (
            _invalid_core_derivation(monkeypatch, failure),
            _activation(),
        ),
    )
    install_calls: list[str] = []

    def forbidden_install(*args: Any, **kwargs: Any) -> dict[str, Any]:
        install_calls.append(str(kwargs.get("label")))
        pytest.fail("checkpoint publication was reached after prerequisite failure")

    monkeypatch.setattr(
        runner, "_install_checkpoint_session_directory", forbidden_install
    )
    before_repository = _tree_snapshot(repository)
    before_external = _tree_snapshot(external_root)

    with pytest.raises(runner.V18Error, match=message):
        runner.prepare_checkpoint(
            session_date=TARGET_SESSION,
            checkpoint_core_store_root=external_root,
        )

    assert install_calls == []
    assert not decision_ledger.exists()
    assert _tree_snapshot(repository) == before_repository
    assert _tree_snapshot(external_root) == before_external


@pytest.mark.parametrize(
    ("state", "decision", "reason"),
    [
        (
            _state(available=False, value=None, selected_rank=None),
            "cash_state_unavailable",
            "state_insufficient_prior_months",
        ),
        (
            _state(available=True, value=0.0, selected_rank=None),
            "cash_state_zero",
            "state_value_exact_zero",
        ),
        (
            _state(available=True, value=0.4, selected_rank=1),
            "selected_rank1",
            None,
        ),
    ],
)
def test_only_registered_state_conditions_can_derive_cash(
    monkeypatch: pytest.MonkeyPatch,
    state: Mapping[str, Any],
    decision: str,
    reason: str | None,
) -> None:
    core = _build_core(monkeypatch, state=state)

    assert tuple(runner.DAILY_FAILURE_REASONS) == (
        "state_insufficient_prior_months",
        "state_value_exact_zero",
    )
    assert core["decision"] == decision
    assert core["failure_reason"] == reason
    assert core["candidate_selected_code"] == (None if reason else "1001")
    assert runner.validate_checkpoint_decision_core(
        core, session_date=TARGET_SESSION, resolution="primary"
    ) == core


def test_caller_cannot_inject_a_cash_or_failure_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(runner.V18Error, match="cannot inject"):
        _build_core(
            monkeypatch,
            failure_reason="source_integrity_failure",
        )


def test_checkpoint_core_rejects_every_non_state_cash_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _build_core(
        monkeypatch,
        state=_state(available=True, value=0.0, selected_rank=None),
    )
    core["failure_reason"] = "source_integrity_failure"

    with pytest.raises(runner.V18Error, match="not registered"):
        runner.validate_checkpoint_decision_core(
            core,
            session_date=TARGET_SESSION,
            resolution="primary",
        )


class _UnreadablePerformanceInput:
    def __init__(self, label: str) -> None:
        self.label = label
        self.accesses: list[str] = []

    def _fail(self, operation: str) -> None:
        self.accesses.append(operation)
        raise AssertionError(f"nonterminal evaluation read {self.label}: {operation}")

    def __iter__(self) -> Any:
        self._fail("iter")

    def __len__(self) -> int:
        self._fail("len")

    def __getitem__(self, key: Any) -> Any:
        self._fail(f"getitem:{key!r}")

    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        self._fail("array")


def _forbidden_read(label: str) -> Any:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"nonterminal evaluation reached {label}")

    return fail


def _patch_terminal_only_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "validate_predictor_evidence",
        "validate_checkpoint_evidence",
        "_validate_deferred_state_evidence",
        "_preflight_terminal_completed_month_ledger",
        "validate_completed_month_records",
        "validate_outcome_records",
        "validate_deferred_score_evidence",
        "validate_outcome_evidence",
        "_evaluation_frame",
        "materialize_picks",
        "build_result",
    ):
        monkeypatch.setattr(runner, name, _forbidden_read(name))


def test_pure_evaluate_nonterminal_does_not_touch_any_performance_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision_rows = [{"session_date": TARGET_SESSION}]
    outcomes = _UnreadablePerformanceInput("outcomes")
    completed = _UnreadablePerformanceInput("completed months")
    scores = _UnreadablePerformanceInput("scores")
    calendar = pd.DatetimeIndex([TARGET_SESSION, "2026-08-06"])
    monkeypatch.setattr(
        runner, "validate_decision_records", lambda value: decision_rows
    )
    monkeypatch.setattr(
        runner,
        "deterministic_terminal_session",
        lambda first, scheduled: scheduled[-1],
    )
    monkeypatch.setattr(runner, "semantic_decision_hash", lambda value: "a" * 64)
    _patch_terminal_only_helpers(monkeypatch)
    monkeypatch.setattr(runner, "_plain_file_bytes", _forbidden_read("file bytes"))
    monkeypatch.setattr(runner, "_read_csv_plain", _forbidden_read("score CSV"))
    monkeypatch.setattr(runner, "read_json", _forbidden_read("state JSON"))

    result = runner.evaluate(
        decision_rows,
        outcomes,  # type: ignore[arg-type]
        completed,  # type: ignore[arg-type]
        calendar=calendar,
        source_manifest_directory="unreadable-source-manifests",
        predictor_raw_store_root="unreadable-predictor-raw",
        predictor_derived_store_root="unreadable-predictor-derived",
        scores=scores,  # type: ignore[arg-type]
        outcome_manifest_directory="unreadable-outcome-manifests",
        outcome_raw_store_root="unreadable-outcome-raw",
        checkpoint_core_store_root="unreadable-checkpoints",
    )

    assert result["status"].startswith("awaiting_")
    assert result["gate_evaluated"] is False
    assert result["outcome_records"] == 0
    assert outcomes.accesses == completed.accesses == scores.accesses == []


def test_evaluate_cli_nonterminal_reads_only_the_decision_and_calendar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = {
        "DECISION_LEDGER": tmp_path / "decisions.jsonl",
        "OUTCOME_LEDGER": tmp_path / "outcomes.jsonl",
        "COMPLETED_MONTH_LEDGER": tmp_path / "completed-months.jsonl",
        "SCORE_OUTPUT": tmp_path / "scores.csv",
        "PICKS_OUTPUT": tmp_path / "picks.csv",
        "RESULT_OUTPUT": tmp_path / "result.json",
        "SOURCE_MANIFEST_DIR": tmp_path / "source-manifests",
        "OUTCOME_MANIFEST_DIR": tmp_path / "outcome-manifests",
        "STATE_MANIFEST_DIR": tmp_path / "state-manifests",
    }
    for name, path in paths.items():
        monkeypatch.setattr(runner, name, path)
    decision_rows = [{"session_date": TARGET_SESSION}]
    reads: list[Path] = []

    def guarded_authority(
        path: str | Path,
        *,
        required_fields: Any,
        validator: Any,
        heal_derived: bool,
    ) -> list[dict[str, Any]]:
        observed = Path(path)
        reads.append(observed)
        if observed != paths["DECISION_LEDGER"]:
            raise AssertionError(
                f"nonterminal CLI read forbidden authority: {observed}"
            )
        assert tuple(required_fields) == tuple(runner.DECISION_FIELDS)
        assert validator is runner.validate_decision_records
        assert heal_derived is False
        return decision_rows

    monkeypatch.setattr(
        runner, "load_jsonl_record_authority", guarded_authority
    )
    monkeypatch.setattr(
        runner, "validate_decision_records", lambda value: decision_rows
    )
    monkeypatch.setattr(
        runner,
        "load_registered_calendar",
        lambda: pd.DatetimeIndex([TARGET_SESSION, "2026-08-06"]),
    )
    monkeypatch.setattr(
        runner,
        "deterministic_terminal_session",
        lambda first, scheduled: scheduled[-1],
    )
    monkeypatch.setattr(
        runner,
        "validate_runtime_lock",
        lambda **kwargs: ({"lock_id": "test"}, "a" * 64),
    )
    monkeypatch.setattr(
        runner, "validate_protocol", lambda *args, **kwargs: ({}, "b" * 64)
    )
    monkeypatch.setattr(
        runner,
        "validate_canonical_activation_artifacts",
        lambda: ({}, "c" * 64, {}, "d" * 64),
    )
    _patch_terminal_only_helpers(monkeypatch)
    for name in ("_plain_file_bytes", "_read_csv_plain", "read_json"):
        monkeypatch.setattr(runner, name, _forbidden_read(name))
    monkeypatch.setattr(runner, "write_json", _forbidden_read("result write"))
    monkeypatch.setattr(
        runner, "_write_csv_exclusive", _forbidden_read("picks write")
    )
    result_existence_checks: list[Path] = []

    def guarded_result_presence(path: str | Path, *, label: str) -> str:
        observed = Path(path)
        result_existence_checks.append(observed)
        if observed != paths["RESULT_OUTPUT"]:
            raise AssertionError(
                f"nonterminal CLI checked a non-result path: {observed}"
            )
        assert label == "canonical result"
        return "absent"

    with monkeypatch.context() as patch:
        patch.setattr(
            runner, "_local_authority_presence_state", guarded_result_presence
        )
        exit_code = runner.main(
            [
                "evaluate",
                "--predictor-raw-store-root",
                str(tmp_path / "unreadable-predictor-raw"),
                "--predictor-derived-store-root",
                str(tmp_path / "unreadable-predictor-derived"),
                "--outcome-raw-store-root",
                str(tmp_path / "unreadable-outcome-raw"),
                "--checkpoint-core-store-root",
                str(tmp_path / "unreadable-checkpoints"),
            ]
        )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["status"].startswith("awaiting_")
    assert output["performance_evidence_opened"] is False
    assert reads == [paths["DECISION_LEDGER"]]
    assert result_existence_checks == [paths["RESULT_OUTPUT"]]
    for name in (
        "OUTCOME_LEDGER",
        "COMPLETED_MONTH_LEDGER",
        "SCORE_OUTPUT",
        "PICKS_OUTPUT",
        "RESULT_OUTPUT",
        "STATE_MANIFEST_DIR",
    ):
        assert not paths[name].exists()


def _parser_commands(parser: argparse.ArgumentParser) -> set[str]:
    actions = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(actions) == 1
    return set(actions[0].choices)


def test_outcome_blind_integrity_abort_remains_a_distinct_registered_command() -> None:
    parser = runner._build_parser()
    commands = _parser_commands(parser)

    assert "abort" in commands
    parsed = parser.parse_args(
        [
            "abort",
            "--failure-reason",
            "source_integrity_failure",
            "--integrity-stage",
            "source_ingestion",
            "--activation-context",
            "activation-context.json",
        ]
    )
    assert parsed.command == "abort"


@pytest.mark.parametrize(
    "forbidden_command",
    [
        "prepare-source-manifest",
        "prepare-month",
        "prepare-top2",
        "prepare-state",
        "prepare-checkpoint",
        "attach-outcomes",
        "prepare-outcome-manifest",
        "close-month",
    ],
)
def test_old_split_mutation_cli_is_not_parseable(
    forbidden_command: str,
) -> None:
    parser = runner._build_parser()
    commands = _parser_commands(parser)

    assert forbidden_command not in commands
    with pytest.raises(SystemExit):
        parser.parse_args([forbidden_command])
