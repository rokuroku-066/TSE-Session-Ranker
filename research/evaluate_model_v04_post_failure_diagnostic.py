#!/usr/bin/env python3
"""One-shot non-formal diagnostic after the v0.4 formal runtime failure.

The sealed claim remains unevaluated because its receipt-bearing process was
OOM-killed before model evaluation.  This separate diagnostic binds the frozen
winner/control and the exact-equivalent memory-bounded panel, then evaluates
each specification once.  It cannot select, tune, or produce a formal holdout
status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research import finalize_logit_v04 as base  # noqa: E402
from research import recover_logit_v04_holdout as formal  # noqa: E402


FORMAL_LOCK_PATH = ROOT / "research/model_v04_parser_recovery_lock.json"
FORMAL_LOCK_SHA256 = (
    "1f87a81e21175dc52c82f8a42762ab52cfd8ab39a3b18bcdb6061d908ab95554"
)
FORMAL_RECEIPT_PATH = ROOT / "research/model_v04_parser_recovery_consumed.json"
FORMAL_RECEIPT_SHA256 = (
    "dfaaa3cf94614e76a5bc826386d63380cbcd0473523744d2571dce98125587e2"
)
RUNTIME_FAILURE_PATH = (
    ROOT / "research/model_v04_holdout_recovery_runtime_failure.json"
)
RUNTIME_FAILURE_SHA256 = (
    "f34912dbc9c9c787d10caffc9766e132f6527f3ccc1ab3acb66a9145286bb5c3"
)
PANEL_PATH = Path("/tmp/model_v04_post_failure_diagnostic_panel.pkl")
PANEL_SHA256 = (
    "f2da914861dd9da8afc64567f90f0ef9af4ff1b0eb2e480dbff2f2cc82ff395a"
)
PANEL_MANIFEST_PATH = PANEL_PATH.with_suffix(PANEL_PATH.suffix + ".manifest.json")
PANEL_MANIFEST_SHA256 = (
    "780206fe23a72481ce6c2297a24f789aebebcbd3bb434889f2ab9d582f75d864"
)
PANEL_BUILDER_PATH = ROOT / "research/build_model_v04_diagnostic_panel.py"
PANEL_BUILDER_SHA256 = (
    "fe2a38b8e67c3387896c43eb096325f6083d1ecf157a7a033de80e556f450f1f"
)
PANEL_TEST_PATH = ROOT / "tests/test_model_v04_diagnostic_panel.py"
PANEL_TEST_SHA256 = (
    "d260d90dd0a6997f01a92c1e060c64c721a91008fc0c23c27cdc760ac722186e"
)
EQUIVALENCE_PATH = (
    ROOT / "research/model_v04_diagnostic_panel_development_equivalence.json"
)
EQUIVALENCE_SHA256 = (
    "7870171ec64788236003337030a6ee6fc456a1d1567932acb88e07cb7c33cb4c"
)
EQUIVALENCE_RUNNER_PATH = ROOT / "research/audit_model_v04_diagnostic_panel.py"
EQUIVALENCE_RUNNER_SHA256 = (
    "10fa48ea3f6389cc534c0374c086d6f35a047ad559bececf19b55dc4c144eda3"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pickle_bound(path: str | Path, expected_sha256: str) -> tuple[Any, str]:
    """Hash and deserialize one already-open file, then re-hash that handle.

    Keeping one descriptor open prevents a path replacement between provenance
    verification and deserialization.  The second pass also fails closed if the
    underlying file is modified in place while pickle is reading it.
    """

    source = Path(path)
    with source.open("rb") as stream:
        before = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            before.update(block)
        before_sha256 = before.hexdigest()
        if before_sha256 != expected_sha256:
            raise ValueError(f"pickle artifact changed before read: {source}")
        stream.seek(0)
        value = pd.read_pickle(stream)
        stream.seek(0)
        after = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            after.update(block)
        after_sha256 = after.hexdigest()
    if after_sha256 != expected_sha256:
        raise ValueError(f"pickle artifact changed during read: {source}")
    return value, after_sha256


def read_json_bound(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"JSON artifact changed: {path}")
    return json.loads(payload.decode("utf-8"))


def fixed_outputs() -> dict[str, str]:
    return {
        "result": str(
            ROOT / "research/model_v04_post_failure_diagnostic_result.json"
        ),
        "winner_picks": str(
            ROOT / "research/model_v04_post_failure_diagnostic_winner_picks.csv"
        ),
        "v03_control_picks": str(
            ROOT
            / "research/model_v04_post_failure_diagnostic_v03_control_picks.csv"
        ),
        "receipt": str(
            ROOT / "research/model_v04_post_failure_diagnostic_consumed.json"
        ),
    }


def verify_evidence() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], Any
]:
    lock, selection, _, sessions = formal.load_and_verify_lock(
        FORMAL_LOCK_PATH,
        FORMAL_LOCK_SHA256,
        require_outputs_absent=False,
    )
    fixed_hashes = {
        FORMAL_RECEIPT_PATH: FORMAL_RECEIPT_SHA256,
        RUNTIME_FAILURE_PATH: RUNTIME_FAILURE_SHA256,
        PANEL_PATH: PANEL_SHA256,
        PANEL_MANIFEST_PATH: PANEL_MANIFEST_SHA256,
        PANEL_BUILDER_PATH: PANEL_BUILDER_SHA256,
        PANEL_TEST_PATH: PANEL_TEST_SHA256,
        EQUIVALENCE_PATH: EQUIVALENCE_SHA256,
        EQUIVALENCE_RUNNER_PATH: EQUIVALENCE_RUNNER_SHA256,
    }
    for path, expected in fixed_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"diagnostic evidence changed: {path}")
    failure = read_json_bound(RUNTIME_FAILURE_PATH, RUNTIME_FAILURE_SHA256)
    if (
        failure.get("formal_metric_run_count") != 0
        or failure.get("winner_evaluation_count") != 0
        or failure.get("v03_control_evaluation_count") != 0
        or failure.get("result_output_exists") is not False
    ):
        raise ValueError("formal failure no longer proves zero model evaluations")
    manifest = read_json_bound(PANEL_MANIFEST_PATH, PANEL_MANIFEST_SHA256)
    if (
        manifest.get("formal_status") != "not_a_formal_holdout_result"
        or manifest.get("builder_sha256") != PANEL_BUILDER_SHA256
        or manifest.get("panel_sha256") != PANEL_SHA256
        or manifest.get("aggregate_return_metrics_computed") is not False
        or manifest.get("model_evaluation_count") != 0
        or manifest.get("holdout_labels_materialized") is not True
    ):
        raise ValueError("diagnostic panel provenance is inconsistent")
    equivalence = read_json_bound(EQUIVALENCE_PATH, EQUIVALENCE_SHA256)
    if (
        equivalence.get("winner_pick_identity_passed") is not True
        or equivalence.get("control_pick_identity_passed") is not True
        or equivalence.get("holdout_evaluation_count") != 0
        or equivalence.get("holdout_aggregate_metrics_computed_or_inspected")
        is not False
    ):
        raise ValueError("development equivalence audit did not pass")
    for path in (
        lock["outputs"]["result"],
        lock["outputs"]["winner_picks"],
        lock["outputs"]["v03_control_picks"],
    ):
        if Path(path).exists():
            raise ValueError("a formal result unexpectedly exists")
    return selection, manifest, equivalence, sessions


def build_lock(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("diagnostic evaluation lock already exists")
    selection, manifest, equivalence, _ = verify_evidence()
    winner = base.ModelSpec.from_dict(selection["winner"])
    control = formal.baseline_spec()
    payload = {
        "schema_version": 1,
        "phase": "post_failure_nonformal_diagnostic_lock",
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "formal_evaluation_status": (
            "formal_recovery_incomplete_due_runtime_failure"
        ),
        "formal_validation_available": False,
        "selection_lock_sha256": formal.EXPECTED_SELECTION_SHA256,
        "formal_recovery_lock_path": str(FORMAL_LOCK_PATH),
        "formal_recovery_lock_sha256": FORMAL_LOCK_SHA256,
        "formal_recovery_receipt_path": str(FORMAL_RECEIPT_PATH),
        "formal_recovery_receipt_sha256": FORMAL_RECEIPT_SHA256,
        "formal_runtime_failure_path": str(RUNTIME_FAILURE_PATH),
        "formal_runtime_failure_sha256": RUNTIME_FAILURE_SHA256,
        "formal_metric_run_count": 0,
        "panel_path": str(PANEL_PATH),
        "panel_sha256": PANEL_SHA256,
        "panel_manifest_path": str(PANEL_MANIFEST_PATH),
        "panel_manifest_sha256": PANEL_MANIFEST_SHA256,
        "panel_content_sha256": manifest["panel_content_sha256"],
        "panel_builder_path": str(PANEL_BUILDER_PATH),
        "panel_builder_sha256": PANEL_BUILDER_SHA256,
        "panel_test_path": str(PANEL_TEST_PATH),
        "panel_test_sha256": PANEL_TEST_SHA256,
        "development_equivalence_path": str(EQUIVALENCE_PATH),
        "development_equivalence_sha256": EQUIVALENCE_SHA256,
        "development_equivalence_runner_path": str(EQUIVALENCE_RUNNER_PATH),
        "development_equivalence_runner_sha256": EQUIVALENCE_RUNNER_SHA256,
        "development_equivalence_summary": {
            "winner_pick_identity_passed": equivalence[
                "winner_pick_identity_passed"
            ],
            "control_pick_identity_passed": equivalence[
                "control_pick_identity_passed"
            ],
            "winner_max_abs_oc_return_pct_difference": equivalence[
                "winner_max_abs_oc_return_pct_difference"
            ],
            "control_max_abs_oc_return_pct_difference": equivalence[
                "control_max_abs_oc_return_pct_difference"
            ],
        },
        "diagnostic_runner_path": str(Path(__file__).resolve()),
        "diagnostic_runner_sha256": sha256_file(Path(__file__).resolve()),
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
            "diagnostic_metric_run_limit": 1,
            "winner_evaluation_limit": 1,
            "control_evaluation_limit": 1,
            "result_cannot_validate_formal_holdout": True,
            "futures_features": (
                "prospective-only; unavailable in the frozen historical panel"
            ),
        },
        "base_implementation_sha256": base._implementation_hashes(),
        "runtime": base._runtime_versions(),
        "outputs": fixed_outputs(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(base.json_dumps(payload) + "\n")
    print(f"diagnostic_lock={output}", flush=True)


def load_lock(path: Path, expected_sha256: str, *, outputs_absent: bool) -> tuple[
    dict[str, Any], dict[str, Any], Any
]:
    lock = read_json_bound(path, expected_sha256)
    if lock.get("schema_version") != 1 or lock.get(
        "phase"
    ) != "post_failure_nonformal_diagnostic_lock":
        raise ValueError("diagnostic lock has the wrong phase")
    fixed = {
        "selection_lock_sha256": formal.EXPECTED_SELECTION_SHA256,
        "formal_recovery_lock_sha256": FORMAL_LOCK_SHA256,
        "formal_recovery_receipt_sha256": FORMAL_RECEIPT_SHA256,
        "formal_runtime_failure_sha256": RUNTIME_FAILURE_SHA256,
        "panel_sha256": PANEL_SHA256,
        "panel_manifest_sha256": PANEL_MANIFEST_SHA256,
        "panel_builder_sha256": PANEL_BUILDER_SHA256,
        "panel_test_sha256": PANEL_TEST_SHA256,
        "development_equivalence_sha256": EQUIVALENCE_SHA256,
        "development_equivalence_runner_sha256": EQUIVALENCE_RUNNER_SHA256,
    }
    for key, expected in fixed.items():
        if lock.get(key) != expected:
            raise ValueError(f"diagnostic lock binding changed: {key}")
    if lock.get("outputs") != fixed_outputs():
        raise ValueError("diagnostic output paths changed")
    runner_path = Path(lock.get("diagnostic_runner_path", "")).resolve()
    current_runner = Path(__file__).resolve()
    if runner_path != current_runner:
        raise ValueError("diagnostic lock points to a different runner")
    if sha256_file(current_runner) != lock.get("diagnostic_runner_sha256"):
        raise ValueError("diagnostic runner changed after lock")
    contract = lock.get("evaluation_contract", {})
    if (
        lock.get("formal_evaluation_status")
        != "formal_recovery_incomplete_due_runtime_failure"
        or lock.get("formal_validation_available") is not False
        or lock.get("formal_metric_run_count") != 0
        or contract.get("selection_performed") is not False
        or contract.get("holdout_retuning_performed") is not False
        or contract.get("diagnostic_metric_run_limit") != 1
        or contract.get("winner_evaluation_limit") != 1
        or contract.get("control_evaluation_limit") != 1
        or contract.get("result_cannot_validate_formal_holdout") is not True
    ):
        raise ValueError("diagnostic claim or one-shot contract changed")
    selection, _, _, sessions = verify_evidence()
    if lock.get("winner") != selection.get("winner") or lock.get(
        "winner_spec_id"
    ) != selection.get("winner_spec_id"):
        raise ValueError("diagnostic winner differs from selection")
    if lock.get("v03_control") != asdict(formal.baseline_spec()):
        raise ValueError("diagnostic control changed")
    if lock.get("base_implementation_sha256") != base._implementation_hashes():
        raise ValueError("base implementation changed after diagnostic lock")
    if lock.get("runtime") != selection.get("runtime") or (
        base._runtime_versions() != selection.get("runtime")
    ):
        raise ValueError("diagnostic runtime differs from selection")
    if outputs_absent:
        existing = [
            path for path in map(Path, lock["outputs"].values()) if path.exists()
        ]
        if existing:
            raise FileExistsError(f"diagnostic output already exists: {existing}")
    return lock, selection, sessions


def preflight(args: argparse.Namespace) -> None:
    load_lock(
        Path(args.diagnostic_lock).resolve(),
        args.expected_diagnostic_lock_sha256,
        outputs_absent=True,
    )
    print("preflight=passed", flush=True)


def evaluate(args: argparse.Namespace) -> None:
    lock_path = Path(args.diagnostic_lock).resolve()
    lock, _, sessions = load_lock(
        lock_path,
        args.expected_diagnostic_lock_sha256,
        outputs_absent=True,
    )
    outputs = {key: Path(value) for key, value in lock["outputs"].items()}
    receipt = {
        "schema_version": 1,
        "phase": "post_failure_nonformal_diagnostic_consumed",
        "started_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "diagnostic_lock_path": str(lock_path),
        "diagnostic_lock_sha256": args.expected_diagnostic_lock_sha256,
        "formal_metric_run_count": 0,
        "diagnostic_metric_run_limit": 1,
    }
    outputs["receipt"].parent.mkdir(parents=True, exist_ok=True)
    with outputs["receipt"].open("x", encoding="utf-8") as stream:
        stream.write(base.json_dumps(receipt) + "\n")

    panel, panel_loaded_sha256 = read_pickle_bound(PANEL_PATH, PANEL_SHA256)
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
        status = "diagnostic_positive_gate_passed"
    elif gate["diagnostics_monthly"]["mean_pct"] > 0.0:
        status = "diagnostic_positive_mean_only"
    else:
        status = "diagnostic_no_demonstrated_edge"
    winner_path = base.write_frame(winner.picks, outputs["winner_picks"])
    control_path = base.write_frame(control.picks, outputs["v03_control_picks"])
    payload = {
        "schema_version": 1,
        "phase": "post_failure_nonformal_diagnostic_result",
        "status": status,
        "formal_evaluation_status": (
            "formal_recovery_incomplete_due_runtime_failure"
        ),
        "holdout_nonformal_diagnostic_evaluated": True,
        "formal_validation_available": False,
        "can_validate_formal_holdout": False,
        "diagnostic_lock_path": str(lock_path),
        "diagnostic_lock_sha256": args.expected_diagnostic_lock_sha256,
        "diagnostic_receipt_path": str(outputs["receipt"]),
        "diagnostic_receipt_sha256": sha256_file(outputs["receipt"]),
        "selection_lock_sha256": formal.EXPECTED_SELECTION_SHA256,
        "formal_recovery_lock_sha256": FORMAL_LOCK_SHA256,
        "formal_recovery_receipt_sha256": FORMAL_RECEIPT_SHA256,
        "formal_runtime_failure_sha256": RUNTIME_FAILURE_SHA256,
        "panel_sha256": panel_loaded_sha256,
        "panel_manifest_sha256": PANEL_MANIFEST_SHA256,
        "development_equivalence_sha256": EQUIVALENCE_SHA256,
        "formal_metric_run_count": 0,
        "diagnostic_metric_run_count": 1,
        "diagnostic_model_evaluation_counts": {"winner": 1, "v03_control": 1},
        "selection_performed_in_diagnostic": False,
        "candidate_specs_considered_in_diagnostic": 1,
        "holdout_retuning_performed": False,
        "model_or_feature_changes_after_selection": False,
        "holdout_labels_materialized_before_diagnostic": True,
        "winner": asdict(winner_spec),
        "winner_spec_id": winner_spec.id,
        "v03_control": asdict(control_spec),
        "v03_control_spec_id": control_spec.id,
        "diagnostic_gate": gate,
        "winner_result": winner.summary,
        "v03_control_result": control.summary,
        "outputs": {
            "winner_picks_path": str(winner_path),
            "winner_picks_sha256": sha256_file(winner_path),
            "v03_control_picks_path": str(control_path),
            "v03_control_picks_sha256": sha256_file(control_path),
        },
        "runtime": lock["runtime"],
        "interpretation": (
            "Post-failure evidence only. Do not call this a sealed formal "
            "holdout or use it to retune the frozen model."
        ),
        "futures_feature_status": (
            "prospective-only; excluded from this historical diagnostic"
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
        "--output",
        default="research/model_v04_post_failure_diagnostic_lock.json",
    )
    for name in ("preflight", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--diagnostic-lock", required=True)
        command.add_argument("--expected-diagnostic-lock-sha256", required=True)
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
