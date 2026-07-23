#!/usr/bin/env python3
"""Regression tests for canonical v0.9 integrity and authority guards."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from research.audit_model_v09 import AuditError, audit
from research.build_model_v09_manifest import build_manifest


ARTIFACTS = (
    "model_v09_protocol_ledger.json",
    "model_v09_result.json",
    "model_v09_manifest.json",
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ModelV09AuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = Path(__file__).resolve().parents[1] / "research"
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for name in ARTIFACTS:
            source = self.source / name
            if not source.is_file():
                self.fail(f"generate canonical artifacts before tests: missing {source}")
            shutil.copy2(source, self.root / name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def refresh_result_ledger_binding(self) -> None:
        result_path = self.root / "model_v09_result.json"
        result = read_json(result_path)
        result["protocol_ledger_sha256"] = sha256_file(
            self.root / "model_v09_protocol_ledger.json"
        )
        write_json(result_path, result)

    def refresh_manifest(self) -> None:
        build_manifest(
            self.root,
            self.root / "model_v09_manifest.json",
        )

    def test_valid_artifacts_pass(self) -> None:
        report = audit(self.root)
        self.assertTrue(report["ok"])
        self.assertEqual(report["ledger_entries_verified"], 57)
        self.assertEqual(report["breadth_status"], "complete")
        self.assertFalse(report["production_promotion_allowed"])

    def test_content_tamper_fails_hash_before_semantics(self) -> None:
        result_path = self.root / "model_v09_result.json"
        result = read_json(result_path)
        result["namespaces"]["target_portfolio"]["hypotheses"][
            "target_portfolio.H17_L4_rank2_only"
        ]["cost"]["20"]["net_mean_pct"] += 0.01
        write_json(result_path, result)
        with self.assertRaisesRegex(AuditError, "SHA-256 mismatch"):
            audit(self.root)

    def test_production_true_rejected_even_with_fresh_manifest(self) -> None:
        result_path = self.root / "model_v09_result.json"
        result = read_json(result_path)
        result["authority"]["production_promotion_allowed"] = True
        write_json(result_path, result)
        self.refresh_manifest()
        with self.assertRaisesRegex(AuditError, "production promotion"):
            audit(self.root)

    def test_bare_h20_id_rejected_even_with_fresh_hashes(self) -> None:
        ledger_path = self.root / "model_v09_protocol_ledger.json"
        ledger = read_json(ledger_path)
        old_id = "error_meta.H20_breadth_rank2_one"
        new_id = "H20_breadth_rank2_one"
        for entry in ledger["entries"]:
            if entry["canonical_id"] == old_id:
                entry["canonical_id"] = new_id
                break
        ledger["namespace_policy"]["collision_example"]["error_meta_h20"] = new_id
        write_json(ledger_path, ledger)

        result_path = self.root / "model_v09_result.json"
        result = read_json(result_path)
        hypotheses = result["namespaces"]["error_meta"]["hypotheses"]
        hypotheses[new_id] = hypotheses.pop(old_id)
        result["integrated_decision"]["retrospective_leads_only"][0] = new_id
        write_json(result_path, result)
        self.refresh_result_ledger_binding()
        self.refresh_manifest()
        with self.assertRaisesRegex(AuditError, "bad id"):
            audit(self.root)

    def test_complete_breadth_with_null_results_rejected(self) -> None:
        result_path = self.root / "model_v09_result.json"
        result = read_json(result_path)
        result["namespaces"]["breadth_falsification"]["hypotheses"][
            "breadth_falsification.F1_canonical_breadth_primary"
        ]["result"] = None
        write_json(result_path, result)
        self.refresh_manifest()
        with self.assertRaisesRegex(AuditError, "must contain all results"):
            audit(self.root)

    def test_breadth_sentinel_rejected_after_rehash(self) -> None:
        result_path = self.root / "model_v09_result.json"
        result = read_json(result_path)
        result["namespaces"]["breadth_falsification"]["hypotheses"][
            "breadth_falsification.F1b_traded_row_denominator_control"
        ]["result"]["valid_oc_breadth_threshold_0_5"][
            "confirmation_combined"
        ]["net20_mean_pct"] = 0.1
        write_json(result_path, result)
        self.refresh_manifest()
        with self.assertRaisesRegex(AuditError, "valid-OC confirmation net20"):
            audit(self.root)

    def test_pick_payload_rejected_even_with_fresh_manifest(self) -> None:
        result_path = self.root / "model_v09_result.json"
        result = read_json(result_path)
        result["picks"] = []
        write_json(result_path, result)
        self.refresh_manifest()
        with self.assertRaisesRegex(AuditError, "pick-level key"):
            audit(self.root)

    def test_manifest_order_change_rejected(self) -> None:
        manifest_path = self.root / "model_v09_manifest.json"
        manifest = read_json(manifest_path)
        manifest["artifacts"].reverse()
        write_json(manifest_path, manifest)
        with self.assertRaisesRegex(AuditError, "artifact order"):
            audit(self.root)


if __name__ == "__main__":
    unittest.main()
