from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
STAGE_A_PROTOCOL = RESEARCH / "model_v14_feature_contrast_protocol.json"
STAGE_A_RESULT = RESEARCH / "model_v14_feature_contrast_result.json"
INPUT_MANIFEST = RESEARCH / "model_v05_input_lock.json"
V13_INPUT_ERRATUM = (
    RESEARCH / "model_v13_symbolic_context_input_erratum_v2.json"
)
INPUT_ERRATUM = RESEARCH / "model_v14_feature_contrast_input_erratum.json"
PROTOCOL = RESEARCH / "model_v14_confirmation_protocol.json"
RUNNER = RESEARCH / "model_v14_confirmation_runner.py"
RESULT = RESEARCH / "model_v14_confirmation_result.json"
PICKS = RESEARCH / "model_v14_confirmation_picks.csv"
V13_PICKS = RESEARCH / "model_v13_symbolic_context_picks.csv"
AUDIT_RUNNER = RESEARCH / "model_v14_confirmation_audit.py"
AUDIT = RESEARCH / "model_v14_confirmation_audit.json"

CONTROL = "C00_DAILY_RANK_RIDGE"
CANDIDATES = ("DMD01", "DMD02", "SP01", "GN01")
ALL_MODELS = (CONTROL, *CANDIDATES)
CAPACITIES = (1, 2)
SCHEDULED_SESSIONS = 182
PICK_COLUMNS = (
    "candidate_id",
    "date",
    "model_rank",
    "code",
    "name",
    "model_score",
    "label",
    "oc_return_pct",
)
GATE_CHECKS = {
    "all_three_confirmation_slices_net40_positive",
    "executed_slot_fraction_at_least_80pct",
    "familywise_paired_lower_vs_control_nonnegative",
    "maximum_code_share_at_most_5pct",
    "net40_positive",
    "net60_positive",
    "positive_months_at_least_6_of_9",
    "top10_code_share_at_most_25pct",
    "top10_days_removed_net40_positive",
    "top10_profit_codes_cash_net40_positive",
    "traded_days_at_least_150",
    "unique_codes_at_least_100",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _picks(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"code": "string"},
        float_precision="round_trip",
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def _semantic_hash(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    canonical["date"] = canonical["date"].dt.strftime("%Y-%m-%d")
    canonical = canonical.sort_values(
        ["candidate_id", "date", "model_rank"], kind="stable"
    ).reset_index(drop=True)
    payload = canonical.to_csv(index=False, lineterminator="\n").encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def test_result_and_audit_are_bound_to_exact_frozen_artifacts() -> None:
    protocol = _json(PROTOCOL)
    result = _json(RESULT)
    audit = _json(AUDIT)

    assert result["protocol_sha256"] == _sha256(PROTOCOL)
    assert result["stage_a_protocol_sha256"] == _sha256(STAGE_A_PROTOCOL)
    assert result["stage_a_result_sha256"] == _sha256(STAGE_A_RESULT)
    assert result["input_erratum_sha256"] == _sha256(INPUT_ERRATUM)
    assert result["runner_sha256"] == _sha256(RUNNER)

    assert protocol["stage_a_binding"]["protocol"]["sha256"] == _sha256(
        STAGE_A_PROTOCOL
    )
    assert protocol["stage_a_binding"]["result"]["sha256"] == _sha256(
        STAGE_A_RESULT
    )
    assert protocol["frozen_input"]["input_manifest"]["sha256"] == _sha256(
        INPUT_MANIFEST
    )
    assert protocol["frozen_input"]["v13_input_erratum"]["sha256"] == (
        _sha256(V13_INPUT_ERRATUM)
    )
    assert protocol["frozen_input"]["v14_input_erratum"]["sha256"] == (
        _sha256(INPUT_ERRATUM)
    )
    assert result["input"]["input_manifest_sha256"] == _sha256(INPUT_MANIFEST)
    assert result["input"]["input_erratum_sha256"] == _sha256(
        V13_INPUT_ERRATUM
    )

    assert audit["result_sha256"] == _sha256(RESULT)
    assert audit["picks_file_sha256"] == _sha256(PICKS)
    assert audit["protocol_sha256"] == _sha256(PROTOCOL)
    assert audit["runner_sha256"] == _sha256(RUNNER)
    assert audit["audit_runner_sha256"] == _sha256(AUDIT_RUNNER)


def test_picks_form_exact_five_model_date_rank_grid_and_hash() -> None:
    result = _json(RESULT)
    audit = _json(AUDIT)
    picks = _picks(PICKS)

    assert tuple(picks.columns) == PICK_COLUMNS
    assert len(picks) == len(ALL_MODELS) * SCHEDULED_SESSIONS * 2
    assert set(picks["candidate_id"]) == set(ALL_MODELS)
    assert picks["date"].nunique() == SCHEDULED_SESSIONS
    assert picks["date"].min() == pd.Timestamp("2024-11-01")
    assert picks["date"].max() == pd.Timestamp("2025-07-31")
    assert not picks.duplicated(
        ["candidate_id", "date", "model_rank"]
    ).any()

    expected_slots = pd.MultiIndex.from_product(
        [
            pd.DatetimeIndex(picks["date"].drop_duplicates().sort_values()),
            (1, 2),
        ],
        names=["date", "model_rank"],
    )
    for model in ALL_MODELS:
        observed_slots = pd.MultiIndex.from_frame(
            picks.loc[
                picks["candidate_id"].eq(model), ["date", "model_rank"]
            ].sort_values(["date", "model_rank"], kind="stable")
        )
        assert observed_slots.equals(expected_slots)

    semantic_sha256 = _semantic_hash(picks)
    file_sha256 = _sha256(PICKS)
    assert result["integrity"]["picks_rows"] == len(picks)
    assert result["integrity"]["slots_per_model"] == (
        SCHEDULED_SESSIONS * 2
    )
    assert result["integrity"]["picks_semantic_sha256"] == semantic_sha256
    assert audit["picks_file_sha256"] == file_sha256
    # The committed CSV is already in the canonical semantic-hash order.
    assert semantic_sha256 == file_sha256


def test_c00_confirmation_rows_exactly_reproduce_v13() -> None:
    current = _picks(PICKS)
    reference = _picks(V13_PICKS)
    current = (
        current.loc[current["candidate_id"].eq(CONTROL), PICK_COLUMNS]
        .sort_values(["date", "model_rank"], kind="stable")
        .reset_index(drop=True)
    )
    reference = (
        reference.loc[
            reference["candidate_id"].eq(CONTROL)
            & reference["date"].between(
                "2024-11-01", "2025-07-31", inclusive="both"
            ),
            PICK_COLUMNS,
        ]
        .sort_values(["date", "model_rank"], kind="stable")
        .reset_index(drop=True)
    )

    assert len(current) == SCHEDULED_SESSIONS * 2
    pd.testing.assert_frame_equal(
        current,
        reference,
        check_exact=True,
        check_dtype=True,
    )

    audit = _json(AUDIT)
    reproduction = audit["control_reproduction"]
    assert reproduction["reference_rows"] == SCHEDULED_SESSIONS * 2
    assert reproduction["current_rows"] == SCHEDULED_SESSIONS * 2
    assert reproduction["all_columns_exact"] is True
    assert reproduction["reference_semantic_sha256"] == _semantic_hash(
        reference
    )
    assert reproduction["current_semantic_sha256"] == _semantic_hash(current)


def test_all_eight_variants_expose_frozen_gates_and_fail() -> None:
    result = _json(RESULT)
    expected_variants = [
        f"{candidate}__top{capacity}"
        for candidate in CANDIDATES
        for capacity in CAPACITIES
    ]

    assert result["integrity"]["candidate_count"] == len(CANDIDATES)
    assert result["integrity"]["capacity_variants"] == list(CAPACITIES)
    assert result["integrity"]["family_size"] == len(expected_variants)
    assert [item["variant_id"] for item in result["variants"]] == (
        expected_variants
    )

    for variant in result["variants"]:
        assert set(variant["gate_checks"]) == GATE_CHECKS
        assert variant["gate_passed"] is all(
            variant["gate_checks"].values()
        )
        assert variant["gate_passed"] is False
        assert variant["expected_capacity_slots"] == (
            SCHEDULED_SESSIONS * variant["capacity"]
        )
        assert variant["months"] == 9
        assert len(variant["monthly_net40_mean_pct"]) == 9
        assert set(variant["confirmation_slice_net40_mean_pct"]) == {
            "confirmation_a",
            "confirmation_b",
            "confirmation_c",
        }

    assert result["decision"]["gate_passers"] == []
    assert result["decision"]["ambiguous_multiple_passers"] is False
    assert result["decision"]["forward_shadow_candidate"] is None


def test_production_and_order_authority_remain_disabled() -> None:
    protocol = _json(PROTOCOL)
    result = _json(RESULT)
    audit = _json(AUDIT)

    assert protocol["authority"]["production_promotion_allowed"] is False
    assert protocol["authority"]["orders_allowed"] is False
    assert result["integrity"]["production_model_changed"] is False
    assert result["integrity"]["orders_allowed"] is False
    assert result["decision"]["production_candidate"] is None
    assert result["decision"]["production_model_changed"] is False
    assert result["decision"]["orders_allowed"] is False
    assert audit["checks"]["production_and_order_boundary"] == "PASS"
    assert audit["recomputed_decision"]["production_candidate"] is None
    assert audit["recomputed_decision"]["production_model_changed"] is False
    assert audit["recomputed_decision"]["orders_allowed"] is False


def test_independent_audit_reproduces_committed_audit(tmp_path: Path) -> None:
    reproduced = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            str(AUDIT_RUNNER),
            "--result",
            str(RESULT),
            "--picks",
            str(PICKS),
            "--output",
            str(reproduced),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    expected = _json(AUDIT)
    observed = _json(reproduced)
    assert observed == expected
    assert observed["status"] == "PASS"
    assert observed["audit_passed"] is True
    assert observed["mismatch_count"] == 0
    assert observed["mismatches"] == []
    assert observed["recomputed_decision"]["gate_passers"] == []
    assert set(observed["checks"].values()) == {"PASS"}
