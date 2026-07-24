#!/usr/bin/env python3
"""Audit committed model-v0.9 JSON without reading source /tmp results."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator

try:
    from .build_model_v09_manifest import (
        CHAIN_DOMAIN,
        DEFAULT_ARTIFACTS,
        chain_step,
        sha256_file,
    )
except ImportError:  # Standalone: python research/audit_model_v09.py
    from build_model_v09_manifest import (
        CHAIN_DOMAIN,
        DEFAULT_ARTIFACTS,
        chain_step,
        sha256_file,
    )


class AuditError(RuntimeError):
    """Raised when a canonical artifact violates integrity or authority rules."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def read_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read valid JSON {path.name}: {exc}") from exc
    require(isinstance(value, dict), f"{path.name}: top-level value must be object")
    return value


def walk(value: Any, path: str = "$") -> Iterator[tuple[str, Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, f"{path}[{index}]")


def verify_manifest(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = read_object(root / "model_v09_manifest.json")
    require(manifest.get("schema_version") == 1, "manifest schema version")
    require(manifest.get("algorithm") == "sha256", "manifest algorithm")
    require(
        manifest.get("chain_domain") == CHAIN_DOMAIN.decode("ascii"),
        "manifest chain domain",
    )
    seed = hashlib.sha256(CHAIN_DOMAIN).hexdigest()
    require(manifest.get("chain_seed_sha256") == seed, "manifest chain seed")
    entries = manifest.get("artifacts")
    require(isinstance(entries, list), "manifest artifacts must be a list")
    require(
        [entry.get("path") for entry in entries] == list(DEFAULT_ARTIFACTS),
        "manifest artifact order or scope changed",
    )
    previous = seed
    loaded: dict[str, dict[str, Any]] = {}
    for sequence, entry in enumerate(entries, start=1):
        require(entry.get("sequence") == sequence, "manifest sequence mismatch")
        name = entry["path"]
        require("/" not in name and "\\" not in name, "manifest path escapes root")
        path = root / name
        require(path.is_file(), f"manifest artifact missing: {name}")
        actual_sha256 = sha256_file(path)
        require(entry.get("sha256") == actual_sha256, f"{name}: SHA-256 mismatch")
        require(
            entry.get("previous_chain_sha256") == previous,
            f"{name}: previous chain mismatch",
        )
        current = chain_step(previous, name, actual_sha256)
        require(entry.get("chain_sha256") == current, f"{name}: chain mismatch")
        previous = current
        loaded[name] = read_object(path)
    require(
        manifest.get("final_chain_sha256") == previous,
        "manifest final chain mismatch",
    )
    return manifest, loaded


def verify_no_pick_payload(result: dict[str, Any], result_path: Path) -> None:
    require(
        result_path.stat().st_size < 1_000_000,
        "canonical result is unexpectedly large; pick-level payload suspected",
    )
    forbidden_key = re.compile(r"(^|_)(picks?|pick_path|daily_rows?)(_|$)", re.I)
    for path, value in walk(result):
        key = path.rsplit(".", 1)[-1]
        require(
            forbidden_key.search(key) is None,
            f"pick-level key forbidden in compact result: {path}",
        )
        if isinstance(value, list):
            require(
                len(value) <= 100,
                f"oversized array forbidden in compact result: {path}",
            )


def verify_ledger(ledger: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require(ledger.get("schema_version") == 1, "ledger schema version")
    authority = ledger.get("authority", {})
    require(authority.get("retrospective_only") is True, "ledger retrospective")
    require(
        authority.get("untouched_confirmation_available") is False,
        "ledger untouched-confirmation authority",
    )
    require(
        authority.get("production_promotion_allowed") is False,
        "ledger production promotion must be false",
    )
    entries = ledger.get("entries")
    require(isinstance(entries, list), "ledger entries must be list")
    require(len(entries) == 57, "ledger must contain 57 entries")
    expected_counts = {
        "error_meta": 20,
        "target_portfolio": 33,
        "breadth_falsification": 4,
    }
    counts = {
        namespace: sum(entry.get("namespace") == namespace for entry in entries)
        for namespace in expected_counts
    }
    require(counts == expected_counts, "ledger namespace counts mismatch")
    require(
        ledger.get("counts")
        == {**expected_counts, "total": sum(expected_counts.values())},
        "declared ledger counts mismatch",
    )
    id_pattern = re.compile(
        r"^(error_meta|target_portfolio)\.H\d{2}_.+$"
        r"|^breadth_falsification\.(F1|F1b|F2|F3)_.+$"
    )
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        canonical_id = entry.get("canonical_id")
        require(isinstance(canonical_id, str), "ledger canonical id missing")
        require(id_pattern.fullmatch(canonical_id) is not None, f"bad id: {canonical_id}")
        require(canonical_id not in by_id, f"duplicate id: {canonical_id}")
        require(
            canonical_id.startswith(f"{entry.get('namespace')}."),
            f"namespace mismatch: {canonical_id}",
        )
        require(entry.get("retrospective") is True, f"{canonical_id}: retrospective")
        require(
            entry.get("production_promotion_allowed") is False,
            f"{canonical_id}: production promotion must be false",
        )
        local_id = entry.get("local_id")
        require(
            isinstance(local_id, str)
            and (
                canonical_id.split(".", 1)[1].startswith(f"{local_id}_")
                or canonical_id.split(".", 1)[1] == local_id
            ),
            f"{canonical_id}: local id mapping mismatch",
        )
        by_id[canonical_id] = entry

    error_ids = [
        canonical_id
        for canonical_id in by_id
        if canonical_id.startswith("error_meta.")
    ]
    target_ids = [
        canonical_id
        for canonical_id in by_id
        if canonical_id.startswith("target_portfolio.")
    ]
    require(
        sorted(
            int(re.search(r"\.H(\d{2})_", canonical_id).group(1))
            for canonical_id in error_ids
        )
        == list(range(1, 21)),
        "error/meta H01-H20 incomplete",
    )
    require(
        sorted(
            int(re.search(r"\.H(\d{2})_", canonical_id).group(1))
            for canonical_id in target_ids
        )
        == list(range(1, 34)),
        "target/portfolio H01-H33 incomplete",
    )
    error_h20 = "error_meta.H20_breadth_rank2_one"
    target_h20 = "target_portfolio.H20_L4_rank3_only"
    require(error_h20 in by_id, "namespaced error/meta H20 missing")
    require(target_h20 in by_id, "namespaced target/portfolio H20 missing")
    require(error_h20 != target_h20, "H20 namespace collision")
    collision = ledger.get("namespace_policy", {}).get("collision_example", {})
    require(collision.get("error_meta_h20") == error_h20, "collision map error H20")
    require(
        collision.get("target_portfolio_h20") == target_h20,
        "collision map target H20",
    )

    for canonical_id, entry in by_id.items():
        if canonical_id.startswith("error_meta."):
            number = int(re.search(r"\.H(\d{2})_", canonical_id).group(1))
            require(
                entry.get("posthoc") is (number == 20),
                f"{canonical_id}: posthoc classification",
            )
        elif canonical_id.startswith("target_portfolio."):
            number = int(re.search(r"\.H(\d{2})_", canonical_id).group(1))
            require(
                entry.get("posthoc") is (number >= 17),
                f"{canonical_id}: posthoc classification",
            )
        else:
            require(entry.get("posthoc") is True, f"{canonical_id}: must be posthoc")

    require(
        "not an untouched confirmation" in by_id[error_h20]["authority"],
        "error/meta H20 posthoc authority weakened",
    )
    for canonical_id in (
        "target_portfolio.H17_L4_rank2_only",
        "target_portfolio.H33_L4_rolling120_rank2",
    ):
        require(
            "post_hoc_retrospective" in by_id[canonical_id]["authority"],
            f"{canonical_id}: posthoc authority weakened",
        )
    return by_id


def verify_result(
    root: Path,
    result: dict[str, Any],
    ledger: dict[str, Any],
    ledger_by_id: dict[str, dict[str, Any]],
) -> None:
    require(result.get("schema_version") == 1, "result schema version")
    require(
        result.get("protocol_ledger_sha256")
        == sha256_file(root / "model_v09_protocol_ledger.json"),
        "result does not bind exact protocol ledger",
    )
    authority = result.get("authority", {})
    require(authority.get("retrospective_only") is True, "result retrospective")
    require(
        authority.get("untouched_confirmation_available") is False,
        "result untouched-confirmation authority",
    )
    require(
        authority.get("production_promotion_allowed") is False,
        "result production promotion must be false",
    )
    require(
        authority.get("production_model_changed") is False,
        "result must not claim a production-model change",
    )
    require(
        result.get("frozen_input", {}).get("canonical_panel_sha256")
        == "6b86f994a1d15d1da8ed40469d44fc409adadc6cd5b3df3c08aef6717bcdf0eb",
        "canonical panel hash changed",
    )

    namespaces = result.get("namespaces")
    require(isinstance(namespaces, dict), "result namespaces missing")
    require(
        set(namespaces)
        == {"error_meta", "target_portfolio", "breadth_falsification"},
        "result namespace set mismatch",
    )
    result_ids: set[str] = set()
    for namespace, expected_count in (
        ("error_meta", 20),
        ("target_portfolio", 33),
        ("breadth_falsification", 4),
    ):
        block = namespaces[namespace]
        require(
            block.get("production_promotion_allowed") is False,
            f"{namespace}: production promotion must be false",
        )
        hypotheses = block.get("hypotheses")
        require(isinstance(hypotheses, dict), f"{namespace}: hypotheses missing")
        require(
            len(hypotheses) == expected_count,
            f"{namespace}: hypothesis count mismatch",
        )
        for canonical_id in hypotheses:
            require(
                canonical_id.startswith(f"{namespace}."),
                f"{canonical_id}: result namespace mismatch",
            )
            require(canonical_id in ledger_by_id, f"{canonical_id}: absent from ledger")
            result_ids.add(canonical_id)
    require(result_ids == set(ledger_by_id), "ledger/result hypothesis sets differ")

    target = namespaces["target_portfolio"]
    require(target.get("family_size") == 33, "target family size")
    global_max = target.get("global_max_t", {})
    require(global_max.get("family_size") == 33, "global max-T family size")
    require(global_max.get("samples") == 5000, "global max-T samples")
    require(global_max.get("circular_block_length") == 5, "global max-T block")
    require(global_max.get("confidence") == 0.8, "global max-T confidence")
    require(
        global_max.get("critical_pct") == 0.21346890370488542,
        "global max-T critical value",
    )
    h17 = target["hypotheses"]["target_portfolio.H17_L4_rank2_only"]
    require(
        h17["cost"]["20"]["net_mean_pct"] == 0.4252887806623605,
        "target H17 net20 sentinel",
    )
    require(
        h17["cost"]["40"]["net_mean_pct"] == 0.22904817915860112,
        "target H17 net40 sentinel",
    )
    require(
        h17["cost"]["60"]["net_mean_pct"] == 0.032807577654841664,
        "target H17 net60 sentinel",
    )
    require(
        h17["best_days_removed_net20"]["20"] == 0.0066157683056097105,
        "target H17 top20-removed sentinel",
    )
    require(
        h17["global_max_t"]["adjusted_one_sided_80pct_lower_pct"]
        == -0.07137210497161242,
        "target H17 global max-T lower sentinel",
    )
    target_h20 = target["hypotheses"]["target_portfolio.H20_L4_rank3_only"]
    require(
        target_h20["cost"]["20"]["net_mean_pct"] == -0.12857256094989403,
        "target H20 rank3 sentinel",
    )

    error = namespaces["error_meta"]
    error_h20 = error["hypotheses"]["error_meta.H20_breadth_rank2_one"]
    require(
        error_h20.get("production_promotion_allowed") is False,
        "error H20 production promotion",
    )
    require(
        error_h20["confirmation_combined"]["net20"] == 0.46797479490184896,
        "error H20 confirmation net20 sentinel",
    )
    require(
        error_h20["confirmation_combined"]["net40"] == 0.2701725970996513,
        "error H20 confirmation net40 sentinel",
    )
    require(
        error_h20["confirmation_combined"]["net60"] == 0.07237039929745342,
        "error H20 confirmation net60 sentinel",
    )
    require(
        error_h20["confirmation_combined"][
            "top20_winning_days_removed_net20"
        ]
        == 0.004731343224898721,
        "error H20 top20-removed sentinel",
    )
    require(
        error_h20["confirmation_quality"]["hit_rate_executed"] == 0.6,
        "error H20 hit-rate sentinel",
    )
    require(
        error_h20["confirmation_quality"]["unique_codes"] == 102,
        "error H20 code-count sentinel",
    )

    breadth = namespaces["breadth_falsification"]
    require(
        "No result from this protocol is untouched confirmation"
        in breadth.get("authority", ""),
        "breadth posthoc authority weakened",
    )
    status = breadth.get("status")
    require(status in {"pending", "complete"}, "breadth status")
    breadth_rows = list(breadth["hypotheses"].values())
    decision = result.get("integrated_decision", {})
    require(
        decision.get("production_promotion_allowed") is False,
        "integrated decision promotion",
    )
    require(
        decision.get("production_model_changed") is False,
        "integrated decision model changed",
    )
    if status == "pending":
        require(
            all(
                row.get("status") == "pending" and row.get("result") is None
                for row in breadth_rows
            ),
            "pending breadth block must contain null results",
        )
        require(
            decision.get("breadth_falsification_complete") is False,
            "pending breadth marked complete",
        )
        require(
            decision.get("winner_selection_status")
            == "deferred_pending_breadth_falsification",
            "winner must remain deferred while breadth is pending",
        )
        require(
            decision.get("breadth_falsification_outcome") == "pending",
            "pending breadth outcome",
        )
    else:
        require(
            all(
                row.get("status") == "complete" and row.get("result") is not None
                for row in breadth_rows
            ),
            "complete breadth block must contain all results",
        )
        require(
            decision.get("breadth_falsification_complete") is True,
            "complete breadth not reflected in decision",
        )
        require(
            decision.get("breadth_falsification_outcome")
            == (
                "valid_oc_preserves_h20_direction_but_canonical_cc_"
                "materially_degrades_and_refutes_semantic_portability"
            ),
            "breadth outcome interpretation missing",
        )
        f1 = breadth["hypotheses"][
            "breadth_falsification.F1_canonical_breadth_primary"
        ]["result"]
        canonical_confirmation = f1[
            "canonical_cc_breadth_threshold_0_5"
        ]["confirmation_combined"]
        require(
            canonical_confirmation["net20_mean_pct"]
            == 0.29638877960804094,
            "canonical-CC confirmation net20 sentinel",
        )
        require(
            canonical_confirmation["net60_mean_pct"]
            == -0.09921561599635464,
            "canonical-CC confirmation net60 refutation sentinel",
        )
        require(
            canonical_confirmation[
                "top20_winning_days_removed_net20_pct"
            ]
            == -0.01989941985510302,
            "canonical-CC top20-removed refutation sentinel",
        )
        require(
            f1["canonical_minus_custom_paired_deltas"][
                "confirmation_combined"
            ]["net20_mean_delta_pct"]
            == -0.17158601529380815,
            "canonical-CC versus custom delta sentinel",
        )

        f1b = breadth["hypotheses"][
            "breadth_falsification.F1b_traded_row_denominator_control"
        ]["result"]
        valid_oc_confirmation = f1b[
            "valid_oc_breadth_threshold_0_5"
        ]["confirmation_combined"]
        require(
            valid_oc_confirmation["net20_mean_pct"]
            == 0.4748993511993126,
            "valid-OC confirmation net20 sentinel",
        )
        require(
            valid_oc_confirmation["net40_mean_pct"]
            == 0.2770971533971148,
            "valid-OC confirmation net40 sentinel",
        )
        require(
            valid_oc_confirmation["net60_mean_pct"]
            == 0.07929495559491698,
            "valid-OC confirmation net60 sentinel",
        )
        require(
            valid_oc_confirmation[
                "top20_winning_days_removed_net20_pct"
            ]
            == 0.01202876776592851,
            "valid-OC top20-removed sentinel",
        )
        require(
            f1b["valid_oc_minus_custom_paired_deltas"][
                "confirmation_combined"
            ]["net20_mean_delta_pct"]
            == 0.006924556297463558,
            "valid-OC versus custom delta sentinel",
        )
        require(
            f1b["relation_to_custom"]["state_switch_sessions"] == 5,
            "valid-OC state-switch sentinel",
        )

        f2 = breadth["hypotheses"][
            "breadth_falsification.F2_state_semantics_audit"
        ]["result"]["canonical_cc_relation_to_custom"]
        require(f2["state_switch_sessions"] == 63, "canonical-CC switch count")
        require(
            f2["state_switch_rate"] == 0.23684210526315788,
            "canonical-CC switch rate",
        )
        require(
            f2["switched_state_sessions"][
                "canonical_minus_custom_net20_mean_pct"
            ]
            == -0.7262898967746019,
            "canonical-CC switched-session refutation sentinel",
        )


def audit(root: Path) -> dict[str, Any]:
    _, loaded = verify_manifest(root)
    ledger = loaded["model_v09_protocol_ledger.json"]
    result = loaded["model_v09_result.json"]
    verify_no_pick_payload(result, root / "model_v09_result.json")
    ledger_by_id = verify_ledger(ledger)
    verify_result(root, result, ledger, ledger_by_id)
    return {
        "ok": True,
        "artifacts_verified": len(DEFAULT_ARTIFACTS),
        "ledger_entries_verified": len(ledger_by_id),
        "breadth_status": result["namespaces"]["breadth_falsification"]["status"],
        "production_promotion_allowed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args.root)
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
