#!/usr/bin/env python3
"""One-shot parser-only recovery for the frozen v0.4 sealed holdout.

This program cannot select or override a model.  ``build-lock`` binds the
already-selected winner, the v0.3 control, every input, the parser-only audit,
and the exact recovery implementation.  ``evaluate`` accepts only that lock's
externally published SHA-256, creates an exclusive receipt, and evaluates the
two frozen specifications exactly once.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research import finalize_logit_v04 as base  # noqa: E402

if Path(base.__file__).resolve() != (
    ROOT / "research/finalize_logit_v04.py"
).resolve():
    raise RuntimeError("recovery runner imported a non-local selection implementation")
_jpx_module = sys.modules.get(base.collect_jpx.__module__)
if _jpx_module is None or Path(_jpx_module.__file__).resolve() != (
    ROOT / "src/tse_session_ranker/data/jpx.py"
).resolve():
    raise RuntimeError("recovery runner imported a non-local JPX parser")


EXPECTED_SELECTION_SHA256 = (
    "7b1d6922dcb591607730e58a98946511c600bf31bbd8ae2a43ee3be50ada83cb"
)
EXPECTED_PROTOCOL_SHA256 = (
    "b7ccbf62de187cd8f70fdb8fc0acca2fb598ce978d948639b7453d0d60ed30c1"
)
EXPECTED_SELECTION_RUNNER_SHA256 = (
    "55db8008bccfe503977394d8cf3ba64e033e14714059f2862144710e43b1e260"
)
EXPECTED_ORIGINAL_RECEIPT_SHA256 = (
    "b27d59994f191be31b2d591d601c4f7047667833b016065876aafd8863465c85"
)
EXPECTED_FAILURE_SHA256 = (
    "1cc29c11cdee0c7df5fc87de6522b75c136dd11a9265412e93f92d75470f4e0e"
)
EXPECTED_OLD_PARSER_SHA256 = (
    "9b4a38cb4fd02b4248c3cc5edc8e10b54407ba14eb6cf9c89e02184df2e101e7"
)
EXPECTED_NEW_PARSER_SHA256 = (
    "1bd2e74acced608eb36c3606b593ea407d2d1e5f54b3283790ef8fd0fb1041f7"
)
EXPECTED_SEALED_AUDIT_SHA256 = (
    "317d5cde9741429263cc91c11575300660f94f62123725fc8ce2d76cdee02eeb"
)
EXPECTED_DEVELOPMENT_AUDIT_SHA256 = (
    "ee8bc1c30aa8409d18650ca5eaf30e35d3fcc8969bfa2982eba1900ebef9eb9f"
)
OLD_PARSER_VERSION = "jpx_daily_text_v5_domestic_ordinary"
NEW_PARSER_VERSION = "jpx_daily_text_v6_special_quote_marker"
ALLOWED_SELECTION_IMPLEMENTATION_CHANGE = "src/tse_session_ranker/data/jpx.py"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sorted_hashes(paths: Sequence[str | Path]) -> list[str]:
    return sorted(sha256_file(path) for path in paths)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_bound(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"JSON artifact changed: {path}")
    return json.loads(payload.decode("utf-8"))


def baseline_spec() -> base.ModelSpec:
    return base.ModelSpec(
        "legacy_v03",
        "positive_session",
        c=0.08,
        class_weight="balanced",
        train_start=str(base.LEGACY_REGIME_START.date()),
        min_train_sessions=20,
        legacy_estimator_class_weight=True,
    )


def fixed_output_paths() -> dict[str, str]:
    holdout_daily = Path("/tmp/jpx_holdout_v04_recovery.pkl")
    return {
        "result": str(
            ROOT / "research/model_v04_holdout_recovery_result.json"
        ),
        "winner_picks": str(
            ROOT / "research/model_v04_holdout_recovery_winner_picks.csv"
        ),
        "v03_control_picks": str(
            ROOT
            / "research/model_v04_holdout_recovery_v03_control_picks.csv"
        ),
        "recovery_receipt": str(
            ROOT / "research/model_v04_parser_recovery_consumed.json"
        ),
        "holdout_daily": str(holdout_daily),
        "holdout_daily_manifest": str(
            holdout_daily.with_suffix(holdout_daily.suffix + ".manifest.json")
        ),
    }


def parser_patch_sha256(old_parser: Path, new_parser: Path) -> str:
    old_lines = old_parser.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = new_parser.read_text(encoding="utf-8").splitlines(keepends=True)
    patch = "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="src/tse_session_ranker/data/jpx.py@selection-v5",
            tofile="src/tse_session_ranker/data/jpx.py@recovery-v6",
        )
    ).encode("utf-8")
    if not patch:
        raise ValueError("parser recovery patch is empty")
    return hashlib.sha256(patch).hexdigest()


def verify_selection_implementation(selection: dict[str, Any]) -> dict[str, str]:
    selected = selection.get("implementation_sha256")
    current = base._implementation_hashes()
    if not isinstance(selected, dict) or set(selected) != set(current):
        raise ValueError("selection implementation inventory changed")
    changed = [name for name in selected if selected[name] != current[name]]
    if changed != [ALLOWED_SELECTION_IMPLEMENTATION_CHANGE]:
        raise ValueError(f"non-parser selection implementation changed: {changed}")
    if selected[ALLOWED_SELECTION_IMPLEMENTATION_CHANGE] != (
        EXPECTED_OLD_PARSER_SHA256
    ):
        raise ValueError("selection lock has an unexpected old parser")
    if current[ALLOWED_SELECTION_IMPLEMENTATION_CHANGE] != (
        EXPECTED_NEW_PARSER_SHA256
    ):
        raise ValueError("current parser does not match the audited recovery")
    return current


def verify_recovery_evidence(
    *,
    selection: dict[str, Any],
    sealed_manifest_path: Path,
    old_parser_path: Path,
    sealed_audit_path: Path,
    development_audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(old_parser_path) != EXPECTED_OLD_PARSER_SHA256:
        raise ValueError("old parser evidence changed")
    if sha256_file(sealed_audit_path) != EXPECTED_SEALED_AUDIT_SHA256:
        raise ValueError("sealed parser audit changed")
    if sha256_file(development_audit_path) != EXPECTED_DEVELOPMENT_AUDIT_SHA256:
        raise ValueError("development parser audit changed")
    sealed = read_json(sealed_audit_path)
    development = read_json(development_audit_path)
    sealed_checks = {
        "sealed_manifest_sha256": base.SEALED_JPX_MANIFEST_SHA256,
        "pdf_count": 160,
        "old_parser_sha256": EXPECTED_OLD_PARSER_SHA256,
        "old_parser_version": OLD_PARSER_VERSION,
        "new_parser_sha256": EXPECTED_NEW_PARSER_SHA256,
        "new_parser_version": NEW_PARSER_VERSION,
        "old_rejected_rows": 63,
        "new_rejected_rows": 0,
        "recovered_rows": 63,
        "affected_files": 47,
        "old_accepted_rows_canonically_identical": True,
        "return_labels_or_metrics_accessed": False,
    }
    for key, expected in sealed_checks.items():
        if sealed.get(key) != expected:
            raise ValueError(f"sealed parser audit mismatch: {key}")
    if sealed.get("marker_counts") != {"ｶ": 37, "ｳ": 26}:
        raise ValueError("sealed parser marker accounting changed")
    per_file = sealed.get("per_file")
    if not isinstance(per_file, list) or len(per_file) != 160:
        raise ValueError("sealed parser audit has incomplete per-file evidence")
    if any(
        int(item.get("ordinary_rows", -1))
        != int(item.get("new_parsed_rows", -2))
        or int(item.get("new_rejected_rows", -1)) != 0
        or int(item.get("old_parsed_rows", -1))
        + int(item.get("recovered_rows", -1))
        != int(item.get("new_parsed_rows", -2))
        for item in per_file
    ):
        raise ValueError("sealed parser per-file row accounting changed")
    if sealed.get("sealed_manifest_sha256") != sha256_file(sealed_manifest_path):
        raise ValueError("sealed parser audit is not bound to the sealed manifest")

    checks = development.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("development parser compatibility audit failed")
    if development.get("old_daily_sha256") != selection["data"][
        "daily_file_sha256"
    ]:
        raise ValueError("development audit is not bound to the selected daily data")
    if development.get("new_model_feature_content_sha256") != selection["data"][
        "daily_content_sha256"
    ]:
        raise ValueError("development model content changed under parser v6")
    if (
        development.get("old_all_canonical_columns_sha256")
        != development.get("new_all_canonical_columns_sha256")
    ):
        raise ValueError("development canonical output changed under parser v6")
    if development.get("rows") != selection["data"]["daily_rows"]:
        raise ValueError("development audit row count changed")
    if development.get("new_rejected_rows") != 0:
        raise ValueError("parser v6 rejected a development row")
    return sealed, development


def verify_frozen_inputs(
    selection: dict[str, Any], sealed_manifest_path: Path
) -> tuple[dict[str, Any], pd.DatetimeIndex]:
    data = selection["data"]
    if sha256_file(data["daily_path"]) != data["daily_file_sha256"]:
        raise ValueError("development daily input changed")
    if sha256_file(data["daily_manifest_path"]) != data["daily_manifest_sha256"]:
        raise ValueError("development daily manifest changed")
    if sha256_file(data["calendar_path"]) != data["calendar_file_sha256"]:
        raise ValueError("development calendar input changed")
    if sha256_file(data["full_calendar_seal_path"]) != data[
        "full_calendar_seal_file_sha256"
    ]:
        raise ValueError("full calendar input changed")

    development_tdnet = data["tdnet_paths"]
    development_tdnet_manifest = data["tdnet_manifest_paths"]
    holdout_tdnet = data["holdout_tdnet_paths_sealed"]
    holdout_tdnet_manifest = data["holdout_tdnet_manifest_paths_sealed"]
    if sorted_hashes(development_tdnet) != data[
        "development_tdnet_file_hashes_sorted"
    ]:
        raise ValueError("development TDnet input changed")
    if sorted_hashes(development_tdnet_manifest) != data[
        "development_tdnet_manifest_hashes_sorted"
    ]:
        raise ValueError("development TDnet manifest changed")
    if sorted_hashes(holdout_tdnet) != data["holdout_tdnet_file_hashes_sorted"]:
        raise ValueError("holdout TDnet input changed")
    if sorted_hashes(holdout_tdnet_manifest) != data[
        "holdout_tdnet_manifest_hashes_sorted"
    ]:
        raise ValueError("holdout TDnet manifest changed")

    sealed_manifest = base._verify_sealed_manifest(sealed_manifest_path)
    development_sessions = base._load_calendar(
        data["calendar_path"], base.CONFIRMATION_END
    )
    sessions = base._verify_full_calendar_seal(
        data["full_calendar_seal_path"], development_sessions, sealed_manifest
    )
    if base.session_calendar_hash(sessions, through=base.HOLDOUT_END) != data[
        "full_calendar_sha256"
    ]:
        raise ValueError("full calendar semantic digest changed")
    base._tdnet_manifest_coverage(
        [*development_tdnet, *holdout_tdnet],
        [*development_tdnet_manifest, *holdout_tdnet_manifest],
    )
    return sealed_manifest, sessions


def build_lock(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"recovery lock already exists: {output}")
    selection_path = Path(args.selection_lock).resolve()
    receipt_path = Path(args.original_receipt).resolve()
    failure_path = Path(args.failure_record).resolve()
    sealed_manifest_path = Path(args.sealed_jpx_manifest).resolve()
    old_parser_path = Path(args.old_parser).resolve()
    sealed_audit_path = Path(args.sealed_parser_audit).resolve()
    development_audit_path = Path(args.development_parser_audit).resolve()
    sealed_audit_runner = ROOT / "research/audit_jpx_parser_recovery.py"
    development_audit_runner = (
        ROOT / "research/audit_jpx_development_compatibility.py"
    )
    parser_tests = ROOT / "tests/test_jpx.py"
    if sha256_file(selection_path) != EXPECTED_SELECTION_SHA256:
        raise ValueError("selection lock changed")
    if sha256_file(receipt_path) != EXPECTED_ORIGINAL_RECEIPT_SHA256:
        raise ValueError("original consumed receipt changed")
    if sha256_file(failure_path) != EXPECTED_FAILURE_SHA256:
        raise ValueError("parser failure record changed")
    selection = read_json_bound(selection_path, EXPECTED_SELECTION_SHA256)
    failure = read_json_bound(failure_path, EXPECTED_FAILURE_SHA256)
    if selection.get("phase") != "selection_lock":
        raise ValueError("selection lock has the wrong phase")
    if failure.get("metric_evaluation_count") != 0 or failure.get(
        "metrics_started"
    ) is not False:
        raise ValueError("failure record indicates prior metric evaluation")
    current_implementation = verify_selection_implementation(selection)
    sealed_manifest, _ = verify_frozen_inputs(selection, sealed_manifest_path)
    sealed_audit, development_audit = verify_recovery_evidence(
        selection=selection,
        sealed_manifest_path=sealed_manifest_path,
        old_parser_path=old_parser_path,
        sealed_audit_path=sealed_audit_path,
        development_audit_path=development_audit_path,
    )
    if base._runtime_versions() != selection["runtime"]:
        raise ValueError("runtime changed after selection")
    winner = base.ModelSpec.from_dict(selection["winner"])
    control = baseline_spec()
    if winner.id != selection["winner_spec_id"]:
        raise ValueError("winner specification ID changed")

    outputs = fixed_output_paths()
    implementation_diff = {
        key: {
            "selection_sha256": selection["implementation_sha256"][key],
            "recovery_sha256": current_implementation[key],
        }
        for key in current_implementation
        if current_implementation[key] != selection["implementation_sha256"][key]
    }
    payload = {
        "schema_version": 1,
        "phase": "parser_only_recovery_lock",
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "selection_lock_path": str(selection_path),
        "selection_lock_sha256": EXPECTED_SELECTION_SHA256,
        "protocol_sha256": selection["protocol_sha256"],
        "selection_runner_sha256": selection["implementation_sha256"][
            "research/finalize_logit_v04.py"
        ],
        "original_consumed_receipt_path": str(receipt_path),
        "original_consumed_receipt_sha256": EXPECTED_ORIGINAL_RECEIPT_SHA256,
        "failure_record_path": str(failure_path),
        "failure_record_sha256": EXPECTED_FAILURE_SHA256,
        "recovery_runner_path": str(Path(__file__).resolve()),
        "recovery_runner_sha256": sha256_file(Path(__file__).resolve()),
        "old_parser_path": str(old_parser_path),
        "old_parser_sha256": EXPECTED_OLD_PARSER_SHA256,
        "old_parser_version": OLD_PARSER_VERSION,
        "new_parser_path": str(ROOT / ALLOWED_SELECTION_IMPLEMENTATION_CHANGE),
        "new_parser_sha256": EXPECTED_NEW_PARSER_SHA256,
        "new_parser_version": NEW_PARSER_VERSION,
        "parser_patch_sha256": parser_patch_sha256(
            old_parser_path, ROOT / ALLOWED_SELECTION_IMPLEMENTATION_CHANGE
        ),
        "sealed_parser_audit_path": str(sealed_audit_path),
        "sealed_parser_audit_sha256": EXPECTED_SEALED_AUDIT_SHA256,
        "sealed_parser_audit_runner_path": str(sealed_audit_runner),
        "sealed_parser_audit_runner_sha256": sha256_file(sealed_audit_runner),
        "development_parser_audit_path": str(development_audit_path),
        "development_parser_audit_sha256": EXPECTED_DEVELOPMENT_AUDIT_SHA256,
        "development_parser_audit_runner_path": str(development_audit_runner),
        "development_parser_audit_runner_sha256": sha256_file(
            development_audit_runner
        ),
        "parser_tests_path": str(parser_tests),
        "parser_tests_sha256": sha256_file(parser_tests),
        "audit_summary": {
            "sealed_pdf_count": sealed_audit["pdf_count"],
            "sealed_recovered_rows": sealed_audit["recovered_rows"],
            "sealed_affected_files": sealed_audit["affected_files"],
            "sealed_new_rejected_rows": sealed_audit["new_rejected_rows"],
            "old_accepted_rows_canonically_identical": sealed_audit[
                "old_accepted_rows_canonically_identical"
            ],
            "development_rows": development_audit["rows"],
            "development_all_columns_hash_equal": development_audit["checks"][
                "all_canonical_columns_hash_equal"
            ],
            "development_model_content_hash_equal": development_audit["checks"][
                "model_feature_content_hash_equal"
            ],
        },
        "winner": asdict(winner),
        "winner_spec_id": winner.id,
        "v03_control": asdict(control),
        "v03_control_spec_id": control.id,
        "evaluation_contract": {
            "selection_performed": False,
            "candidate_specs_considered": 1,
            "holdout_retuning_performed": False,
            "evaluation_start": str(base.HOLDOUT_START.date()),
            "evaluation_end": str(base.HOLDOUT_END.date()),
            "retrain_frequency": "M",
            "portfolio": "scheduled-day top1; no candidate is zero return",
            "primary_round_trip_cost_bps": base.PRIMARY_COST_BPS,
            "stress_round_trip_cost_bps": base.STRESS_COST_BPS,
            "bootstrap_samples": base.DEFAULT_BOOTSTRAP_SAMPLES,
            "bootstrap_seed": base.DEFAULT_SEED,
            "formal_metric_run_limit": 1,
            "winner_evaluation_limit": 1,
            "control_evaluation_limit": 1,
            "futures_features": (
                "prospective-only; excluded because no sealed historical 08:58 input"
            ),
        },
        "access_accounting_before_recovery": {
            "complete_pdf_extraction_passes": 3,
            "complete_full_corpus_parser_passes": 4,
            "formal_failed_parser_passes": 1,
            "complete_diagnostic_parser_passes": 1,
            "complete_parser_audit_parser_passes": {
                "v5": 1,
                "v6": 1,
            },
            "additional_partial_independent_diagnostic_aborted": True,
            "formal_metric_runs": 0,
        },
        "allowed_implementation_changes": [
            ALLOWED_SELECTION_IMPLEMENTATION_CHANGE
        ],
        "implementation_diff": implementation_diff,
        "selection_implementation_sha256": selection[
            "implementation_sha256"
        ],
        "recovery_implementation_sha256": current_implementation,
        "runtime": selection["runtime"],
        "data": {
            **selection["data"],
            "sealed_jpx_manifest_path": str(sealed_manifest_path),
            "sealed_jpx_manifest_sha256": sha256_file(sealed_manifest_path),
            "sealed_jpx_canonical_files_sha256": sealed_manifest[
                "canonical_files_sha256"
            ],
        },
        "outputs": outputs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(base.json_dumps(payload) + "\n")
    print(f"recovery_lock={output}", flush=True)


def load_and_verify_lock(
    lock_path: Path, expected_sha256: str, *, require_outputs_absent: bool
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], pd.DatetimeIndex]:
    lock = read_json_bound(lock_path, expected_sha256)
    if lock.get("schema_version") != 1 or lock.get(
        "phase"
    ) != "parser_only_recovery_lock":
        raise ValueError("recovery lock has the wrong phase")
    required_bindings = {
        "selection_lock_sha256": EXPECTED_SELECTION_SHA256,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "selection_runner_sha256": EXPECTED_SELECTION_RUNNER_SHA256,
        "original_consumed_receipt_sha256": EXPECTED_ORIGINAL_RECEIPT_SHA256,
        "failure_record_sha256": EXPECTED_FAILURE_SHA256,
        "old_parser_sha256": EXPECTED_OLD_PARSER_SHA256,
        "old_parser_version": OLD_PARSER_VERSION,
        "new_parser_sha256": EXPECTED_NEW_PARSER_SHA256,
        "new_parser_version": NEW_PARSER_VERSION,
        "sealed_parser_audit_sha256": EXPECTED_SEALED_AUDIT_SHA256,
        "development_parser_audit_sha256": EXPECTED_DEVELOPMENT_AUDIT_SHA256,
    }
    for key, expected in required_bindings.items():
        if lock.get(key) != expected:
            raise ValueError(f"recovery lock has an unexpected binding: {key}")
    if lock.get("outputs") != fixed_output_paths():
        raise ValueError("recovery lock output paths changed")
    if Path(lock["recovery_runner_path"]).resolve() != Path(__file__).resolve():
        raise ValueError("recovery lock points to a different runner")
    if Path(lock["new_parser_path"]).resolve() != (
        ROOT / ALLOWED_SELECTION_IMPLEMENTATION_CHANGE
    ).resolve():
        raise ValueError("recovery lock points to a different parser")
    locked_paths = {
        "selection_lock_path": lock["selection_lock_sha256"],
        "original_consumed_receipt_path": lock[
            "original_consumed_receipt_sha256"
        ],
        "failure_record_path": lock["failure_record_sha256"],
        "recovery_runner_path": lock["recovery_runner_sha256"],
        "old_parser_path": lock["old_parser_sha256"],
        "new_parser_path": lock["new_parser_sha256"],
        "sealed_parser_audit_path": lock["sealed_parser_audit_sha256"],
        "sealed_parser_audit_runner_path": lock[
            "sealed_parser_audit_runner_sha256"
        ],
        "development_parser_audit_path": lock[
            "development_parser_audit_sha256"
        ],
        "development_parser_audit_runner_path": lock[
            "development_parser_audit_runner_sha256"
        ],
        "parser_tests_path": lock["parser_tests_sha256"],
    }
    for path_key, expected in locked_paths.items():
        if sha256_file(lock[path_key]) != expected:
            raise ValueError(f"locked artifact changed: {path_key}")
    selection = read_json_bound(
        lock["selection_lock_path"], EXPECTED_SELECTION_SHA256
    )
    if selection.get("schema_version") != 1 or selection.get(
        "phase"
    ) != "selection_lock":
        raise ValueError("selection lock has the wrong phase")
    if selection.get("sealed_holdout_access_count") != 0:
        raise ValueError("selection lock does not predate holdout access")
    if selection.get("winner") != lock.get("winner") or selection.get(
        "winner_spec_id"
    ) != lock.get("winner_spec_id"):
        raise ValueError("locked winner differs from the selection lock")
    if asdict(baseline_spec()) != lock.get("v03_control"):
        raise ValueError("locked v0.3 control changed")
    if selection.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256 or (
        sha256_file(base.PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256
    ):
        raise ValueError("protocol changed after selection")
    original_receipt = read_json_bound(
        lock["original_consumed_receipt_path"],
        EXPECTED_ORIGINAL_RECEIPT_SHA256,
    )
    if original_receipt.get("phase") != "sealed_holdout_consumed" or (
        original_receipt.get("selection_lock_sha256")
        != EXPECTED_SELECTION_SHA256
    ):
        raise ValueError("original consumed receipt is inconsistent")
    current_implementation = verify_selection_implementation(selection)
    if current_implementation != lock.get("recovery_implementation_sha256"):
        raise ValueError("recovery implementation changed after lock")
    if lock.get("runtime") != selection.get("runtime") or (
        base._runtime_versions() != selection.get("runtime")
    ):
        raise ValueError("runtime changed after recovery lock")
    sealed_manifest_path = Path(lock["data"]["sealed_jpx_manifest_path"])
    sealed_manifest, sessions = verify_frozen_inputs(
        selection, sealed_manifest_path
    )
    verify_recovery_evidence(
        selection=selection,
        sealed_manifest_path=sealed_manifest_path,
        old_parser_path=Path(lock["old_parser_path"]),
        sealed_audit_path=Path(lock["sealed_parser_audit_path"]),
        development_audit_path=Path(lock["development_parser_audit_path"]),
    )
    failure = read_json_bound(
        lock["failure_record_path"], EXPECTED_FAILURE_SHA256
    )
    if (
        sha256_file(lock["original_consumed_receipt_path"])
        != EXPECTED_ORIGINAL_RECEIPT_SHA256
        or sha256_file(lock["failure_record_path"]) != EXPECTED_FAILURE_SHA256
    ):
        raise ValueError("incident evidence bytes changed")
    if (
        failure.get("selection_lock_sha256") != EXPECTED_SELECTION_SHA256
        or failure.get("initial_consumed_receipt_sha256")
        != EXPECTED_ORIGINAL_RECEIPT_SHA256
        or failure.get("metrics_started") is not False
        or failure.get("metric_evaluation_count") != 0
        or failure.get("model_or_feature_retuning_performed") is not False
    ):
        raise ValueError("failure record no longer proves zero prior metric runs")
    if require_outputs_absent:
        existing = [
            path for path in map(Path, lock["outputs"].values()) if path.exists()
        ]
        if existing:
            raise FileExistsError(f"recovery output already exists: {existing}")
    return lock, selection, sealed_manifest, sessions


def preflight(args: argparse.Namespace) -> None:
    load_and_verify_lock(
        Path(args.recovery_lock).resolve(),
        args.expected_recovery_lock_sha256,
        require_outputs_absent=True,
    )
    print("preflight=passed", flush=True)


def evaluate(args: argparse.Namespace) -> None:
    lock_path = Path(args.recovery_lock).resolve()
    lock, selection, sealed_manifest, sessions = load_and_verify_lock(
        lock_path,
        args.expected_recovery_lock_sha256,
        require_outputs_absent=True,
    )
    lock_sha256 = args.expected_recovery_lock_sha256
    outputs = {key: Path(value) for key, value in lock["outputs"].items()}
    receipt = {
        "schema_version": 1,
        "phase": "parser_only_recovery_consumed",
        "started_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "recovery_lock_path": str(lock_path),
        "recovery_lock_sha256": lock_sha256,
        "selection_lock_sha256": EXPECTED_SELECTION_SHA256,
        "original_consumed_receipt_sha256": EXPECTED_ORIGINAL_RECEIPT_SHA256,
        "failure_record_sha256": EXPECTED_FAILURE_SHA256,
        "formal_metric_run_limit": 1,
    }
    outputs["recovery_receipt"].parent.mkdir(parents=True, exist_ok=True)
    with outputs["recovery_receipt"].open("x", encoding="utf-8") as stream:
        stream.write(base.json_dumps(receipt) + "\n")

    sealed_manifest_path = Path(lock["data"]["sealed_jpx_manifest_path"])
    sealed_paths = base._sealed_jpx_paths(sealed_manifest_path, sealed_manifest)
    blind, parse_manifest = base.collect_jpx(sealed_paths)
    base._verify_parsed_jpx_manifest(parse_manifest, sealed_paths, blind)
    base._verify_blind_dates(blind, sealed_manifest)
    written_jpx = base.write_frame(blind, outputs["holdout_daily"])
    parse_manifest = {
        **parse_manifest,
        "schema_version": 2,
        "export_path": str(written_jpx),
        "export_sha256": sha256_file(written_jpx),
        "parser_recovery_lock_sha256": lock_sha256,
    }
    base.write_json(parse_manifest, outputs["holdout_daily_manifest"])

    data = selection["data"]
    development = base.normalize_daily_prices(base.read_frame(data["daily_path"]))
    prices = base.merge_daily_prices([development, blind])
    disclosures, tdnet_complete_dates = base._load_tdnet(
        [*data["tdnet_paths"], *data["holdout_tdnet_paths_sealed"]],
        [
            *data["tdnet_manifest_paths"],
            *data["holdout_tdnet_manifest_paths_sealed"],
        ],
    )
    disclosures = disclosures[
        disclosures["published_at"].dt.tz_localize(None).dt.normalize().le(
            base.HOLDOUT_END
        )
    ].copy()
    panel, coverage = base.build_research_panel(
        prices, disclosures, sessions, tdnet_complete_dates
    )
    warmup_only = panel["date"].eq(pd.Timestamp("2025-08-01"))
    panel.loc[warmup_only, "training_eligible"] = False
    if panel.loc[warmup_only, "training_eligible"].any():
        raise AssertionError("holdout warmup entered model training")

    winner_spec = base.ModelSpec.from_dict(lock["winner"])
    control_spec = base.ModelSpec.from_dict(lock["v03_control"])
    winner = base.evaluate_spec(
        panel,
        winner_spec,
        evaluation_start=base.HOLDOUT_START,
        evaluation_end=base.HOLDOUT_END,
        evaluation_sessions=sessions,
        retrain_frequency="M",
        bootstrap_samples=base.DEFAULT_BOOTSTRAP_SAMPLES,
    )
    control = base.evaluate_spec(
        panel,
        control_spec,
        evaluation_start=base.HOLDOUT_START,
        evaluation_end=base.HOLDOUT_END,
        evaluation_sessions=sessions,
        retrain_frequency="M",
        bootstrap_samples=base.DEFAULT_BOOTSTRAP_SAMPLES,
    )
    gate = base._holdout_gate_report(
        winner, control, base.DEFAULT_BOOTSTRAP_SAMPLES
    )
    if gate["passed"]:
        status = "validated_shadow"
    elif gate["diagnostics_monthly"]["mean_pct"] > 0.0:
        status = "provisional_shadow"
    else:
        status = "no_demonstrated_tradable_edge"

    written_winner = base.write_frame(winner.picks, outputs["winner_picks"])
    written_control = base.write_frame(control.picks, outputs["v03_control_picks"])
    sealed_audit = read_json(lock["sealed_parser_audit_path"])
    payload = {
        "schema_version": 1,
        "phase": "sealed_holdout_parser_recovery_result",
        "status": status,
        "recovery_lock_path": str(lock_path),
        "recovery_lock_sha256": lock_sha256,
        "selection_lock_path": lock["selection_lock_path"],
        "selection_lock_sha256": EXPECTED_SELECTION_SHA256,
        "protocol_sha256": selection["protocol_sha256"],
        "selection_runner_sha256": lock["selection_runner_sha256"],
        "original_consumed_receipt_path": lock[
            "original_consumed_receipt_path"
        ],
        "original_consumed_receipt_sha256": EXPECTED_ORIGINAL_RECEIPT_SHA256,
        "failure_record_path": lock["failure_record_path"],
        "failure_record_sha256": EXPECTED_FAILURE_SHA256,
        "recovery_consumed_receipt_path": str(outputs["recovery_receipt"]),
        "recovery_consumed_receipt_sha256": sha256_file(
            outputs["recovery_receipt"]
        ),
        "recovery_runner_sha256": lock["recovery_runner_sha256"],
        "sealed_parser_audit_sha256": lock["sealed_parser_audit_sha256"],
        "development_parser_audit_sha256": lock[
            "development_parser_audit_sha256"
        ],
        "parser_recovery": {
            "performed": True,
            "old_parser_version": OLD_PARSER_VERSION,
            "old_parser_sha256": EXPECTED_OLD_PARSER_SHA256,
            "new_parser_version": NEW_PARSER_VERSION,
            "new_parser_sha256": EXPECTED_NEW_PARSER_SHA256,
            "patch_sha256": lock["parser_patch_sha256"],
            "sealed_pdf_count": sealed_audit["pdf_count"],
            "recovered_rows": sealed_audit["recovered_rows"],
            "affected_files": sealed_audit["affected_files"],
            "marker_counts": sealed_audit["marker_counts"],
            "new_rejected_rows": sealed_audit["new_rejected_rows"],
            "old_accepted_rows_canonically_identical": True,
            "development_canonical_output_unchanged": True,
        },
        "access_accounting": {
            "complete_pdf_extraction_passes_before_recovery": 3,
            "recovery_complete_pdf_extraction_passes": 1,
            "complete_pdf_extraction_passes_total": 4,
            "complete_full_corpus_parser_passes_before_recovery": 4,
            "recovery_complete_full_corpus_parser_passes": 1,
            "complete_full_corpus_parser_passes_total": 5,
            "additional_partial_independent_diagnostic_aborted": True,
            "formal_metric_runs_before_recovery": 0,
            "formal_metric_runs_in_recovery": 1,
            "formal_metric_runs_total": 1,
            "model_spec_evaluation_counts": {
                "winner": 1,
                "v03_control": 1,
            },
        },
        "selection_performed_in_recovery": False,
        "candidate_specs_considered_in_recovery": 1,
        "holdout_retuning_performed": False,
        "model_or_feature_changes_after_selection": False,
        "winner": asdict(winner_spec),
        "winner_spec_id": winner_spec.id,
        "v03_control": asdict(control_spec),
        "v03_control_spec_id": control_spec.id,
        "holdout_gate": gate,
        "winner_result": winner.summary,
        "v03_control_result": control.summary,
        "data": {
            "development_daily_sha256": sha256_file(data["daily_path"]),
            "holdout_daily_path": str(written_jpx),
            "holdout_daily_sha256": sha256_file(written_jpx),
            "holdout_daily_manifest_path": str(
                outputs["holdout_daily_manifest"]
            ),
            "holdout_daily_manifest_sha256": sha256_file(
                outputs["holdout_daily_manifest"]
            ),
            "full_calendar_sha256": sha256_file(
                data["full_calendar_seal_path"]
            ),
            "full_calendar_semantic_sha256": base.session_calendar_hash(
                sessions, through=base.HOLDOUT_END
            ),
            "sealed_jpx_manifest_sha256": sha256_file(sealed_manifest_path),
            "development_tdnet_sha256": {
                str(path): sha256_file(path) for path in data["tdnet_paths"]
            },
            "holdout_tdnet_sha256": {
                str(path): sha256_file(path)
                for path in data["holdout_tdnet_paths_sealed"]
            },
            "source_incomplete_dates": [
                str(value.date())
                for value in coverage.loc[~coverage["source_complete"], "date"]
            ],
        },
        "outputs": {
            "winner_picks_path": str(written_winner),
            "winner_picks_sha256": sha256_file(written_winner),
            "v03_control_picks_path": str(written_control),
            "v03_control_picks_sha256": sha256_file(written_control),
        },
        "implementation_sha256": lock["recovery_implementation_sha256"],
        "runtime": lock["runtime"],
        "futures_feature_status": (
            "prospective-only; not included in this sealed holdout because no "
            "historical exact-date 08:58 futures series was available"
        ),
        "next_action": (
            "do not retune on this holdout; keep the frozen specification in "
            "shadow and test the separately versioned futures challenger forward"
        ),
    }
    base.write_json(payload, outputs["result"])
    print(f"result={outputs['result']}", flush=True)
    print(f"status={status}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    lock = subparsers.add_parser("build-lock")
    lock.add_argument(
        "--selection-lock",
        default="research/model_v04_selection_lock.json",
    )
    lock.add_argument(
        "--original-receipt",
        default="research/model_v04_selection_lock_holdout_consumed.json",
    )
    lock.add_argument(
        "--failure-record",
        default="research/model_v04_holdout_parse_failure.json",
    )
    lock.add_argument("--sealed-jpx-manifest", required=True)
    lock.add_argument("--old-parser", required=True)
    lock.add_argument(
        "--sealed-parser-audit",
        default="research/model_v04_parser_recovery_audit.json",
    )
    lock.add_argument(
        "--development-parser-audit",
        default="research/model_v04_development_parser_compatibility.json",
    )
    lock.add_argument(
        "--output", default="research/model_v04_parser_recovery_lock.json"
    )

    for name in ("preflight", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--recovery-lock", required=True)
        command.add_argument("--expected-recovery-lock-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "build-lock":
        build_lock(args)
    elif args.phase == "preflight":
        preflight(args)
    elif args.phase == "evaluate":
        evaluate(args)
    else:  # pragma: no cover
        raise RuntimeError(args.phase)


if __name__ == "__main__":
    main()
