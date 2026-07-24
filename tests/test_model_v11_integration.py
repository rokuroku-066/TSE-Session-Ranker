from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research" / "model_v11_integration_audit.py"
AUDIT_PATH = ROOT / "research" / "model_v11_integration_audit.json"
REPORT_PATH = ROOT / "research" / "model_v11_integration_report.md"
FAMILIES = (
    "new_data",
    "analog",
    "distributional",
    "uplift",
    "graph",
    "shift",
    "calendar",
)
EXACT_BLOCKERS = [
    "frozen T02 OOT raw data missing",
    "08:58 execution data missing",
    "08:58 futures data missing",
    "08:58 orderbook data missing",
    "08:58 PTS data missing",
    "08:58 liquidity data missing",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ModelV11IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise AssertionError(
                f"integration audit failed:\n{completed.stdout}\n"
                f"{completed.stderr}"
            )
        cls.audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        cls.report = REPORT_PATH.read_text(encoding="utf-8")

    def test_all_seven_bound_family_audits_pass(self) -> None:
        audit = self.audit
        self.assertEqual(audit["audit_status"], "pass")
        self.assertTrue(audit["integrity"]["passes"])
        self.assertEqual(
            [row["family"] for row in audit["families"]],
            list(FAMILIES),
        )
        for row in audit["families"]:
            with self.subTest(family=row["family"]):
                self.assertTrue(row["all_checks_pass"])
                self.assertTrue(all(row["checks"].values()))
                self.assertEqual(row["retrospective_passers"], [])
                self.assertIsNone(row["forward_shadow_finalist"])
                self.assertFalse(any(row["production_flags"].values()))
                for artifact in row["artifacts"].values():
                    path = ROOT / artifact["path"]
                    self.assertTrue(path.is_file())
                    self.assertEqual(
                        sha256_file(path), artifact["sha256"]
                    )

    def test_counts_preserve_conceptual_and_policy_units(self) -> None:
        counts = self.audit["counts"]
        self.assertEqual(
            counts,
            {
                "tracks": 7,
                "conceptual_hypotheses": 67,
                "executable_specs": 59,
                "candidate_policies": 118,
                "economically_unique_within_family_selection_sequences": 112,
                "comparator_variants": 14,
                "total_scored_series": 132,
                "familywise_gate_units": 98,
                "candidate_variants_passing": 0,
                "forward_shadow_finalists": 0,
            },
        )
        self.assertEqual(
            {
                row["family"]: row["conceptual_hypotheses"]
                for row in self.audit["families"]
            },
            {
                "new_data": 11,
                "analog": 10,
                "distributional": 8,
                "uplift": 8,
                "graph": 10,
                "shift": 8,
                "calendar": 12,
            },
        )
        self.assertEqual(
            sum(row["candidate_policies"] for row in self.audit["families"]),
            118,
        )

    def test_incompatible_windows_cannot_be_ranked(self) -> None:
        windows = self.audit["window_compatibility"]
        authority = self.audit["cross_family_authority"]
        self.assertEqual(windows["distinct_session_counts"], [88, 107, 266])
        self.assertFalse(windows["cross_family_ranking_permitted"])
        self.assertFalse(windows["point_estimates_directly_comparable"])
        self.assertFalse(windows["common_window_post_hoc_ranking_permitted"])
        self.assertFalse(authority["cross_family_ranking_permitted"])
        self.assertFalse(authority["cross_family_p_value_reported"])
        self.assertFalse(authority["cross_family_winner_selected"])

    def test_seen_panel_and_cross_family_multiplicity_are_fail_closed(
        self,
    ) -> None:
        authority = self.audit["cross_family_authority"]
        self.assertTrue(authority["all_tracks_use_same_seen_panel"])
        self.assertEqual(authority["shared_panel_track_count"], 7)
        self.assertEqual(
            authority["panel_status"], "retrospective_project_seen"
        )
        self.assertFalse(
            authority["family_local_preregistration_restores_fresh_holdout"]
        )
        self.assertFalse(
            authority[
                "family_local_multiplicity_covers_cross_family_selection"
            ]
        )
        self.assertFalse(
            authority[
                "global_118_variant_multiplicity_correction_performed"
            ]
        )
        self.assertEqual(authority["multiplicity_scope"], "track_level_only")
        self.assertIn("incompatible", authority["reason"])
        self.assertIn("project-seen", authority["reason"])

    def test_exact_oot_and_execution_blockers_are_preserved(self) -> None:
        fresh = self.audit["fresh_OOT_and_execution"]
        self.assertEqual(fresh["exact_data_blockers"], EXACT_BLOCKERS)
        self.assertEqual(
            fresh["exact_blocker_summary"],
            "frozen T02 OOT raw data missing; "
            "08:58 execution/futures/orderbook/PTS/liquidity missing",
        )
        self.assertTrue(fresh["protocol_sidecar_exact"])
        self.assertTrue(fresh["bound_candidate_artifacts_exact"])
        self.assertEqual(fresh["available_raw_source_end"], "2025-07-31")
        self.assertEqual(
            fresh["evaluation_window"]["score_start"], "2025-08-04"
        )
        self.assertFalse(fresh["raw_window_reaches_score_start"])
        self.assertFalse(fresh["t02_result_present"])
        self.assertFalse(fresh["t02_audit_present"])
        self.assertFalse(fresh["untouched_holdout_claim"])
        self.assertFalse(
            fresh["historical_cache_receipt_metadata_available"]
        )
        self.assertFalse(fresh["fresh_OOT_gate_passes"])
        self.assertFalse(fresh["execution_gate_passes"])
        self.assertFalse(any(fresh["execution_checks"].values()))

    def test_production_is_a_strict_conjunction_and_has_zero_candidates(
        self,
    ) -> None:
        decision = self.audit["production_decision"]
        self.assertTrue(decision["all_required"])
        self.assertFalse(decision["passes"])
        self.assertEqual(decision["production_candidate_count"], 0)
        self.assertFalse(decision["production_model_changed"])
        self.assertFalse(decision["orders_allowed"])
        self.assertEqual(decision["decision"], "NO_PRODUCTION_CANDIDATE")
        self.assertEqual(decision["exact_data_blockers"], EXACT_BLOCKERS)
        self.assertTrue(
            decision["checks"][
                "all_family_artifact_and_audit_checks_pass"
            ]
        )
        for key, value in decision["checks"].items():
            if key != "all_family_artifact_and_audit_checks_pass":
                with self.subTest(check=key):
                    self.assertFalse(value)

    def test_audit_and_report_are_deterministic(self) -> None:
        audit_before = AUDIT_PATH.read_bytes()
        report_before = REPORT_PATH.read_bytes()
        subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(AUDIT_PATH.read_bytes(), audit_before)
        self.assertEqual(REPORT_PATH.read_bytes(), report_before)

    def test_report_states_no_ranking_and_no_production(self) -> None:
        self.assertIn("本番候補: **0**", self.report)
        self.assertIn(
            "cross-family ranking permitted: false", self.report
        )
        self.assertIn("概念仮説: **67**", self.report)
        self.assertIn("capacity別候補variant: **118**", self.report)
        self.assertIn("実行可能spec: **59**", self.report)
        self.assertIn("scored series総数: **132**", self.report)
        self.assertIn("blocked hypothesisは **9**", self.report)
        self.assertIn("取得group数", self.report)
        self.assertIn("`untouched_holdout_claim=false`", self.report)
        self.assertIn("「edgeが存在しない」ではない", self.report)
        for blocker in EXACT_BLOCKERS:
            self.assertIn(blocker, self.report)

    def test_selection_sequences_and_canonical_decisions(self) -> None:
        selection = self.audit["selection_sequence_accounting"]
        self.assertEqual(selection["candidate_variants"], 118)
        self.assertEqual(
            selection["unique_within_family_economic_sequences"], 112
        )
        self.assertTrue(
            selection["not_all_variants_are_economically_distinct"]
        )
        self.assertFalse(selection["cross_family_deduplication_performed"])
        pending = {
            family: row["canonical_decision"]
            for family, row in (
                (entry["family"], entry) for entry in self.audit["families"]
            )
            if row["canonical_decision"][
                "result_was_pending_independent_audit"
            ]
        }
        self.assertEqual(
            set(pending), {"analog", "uplift", "graph", "calendar"}
        )
        self.assertTrue(
            all(
                row["canonical_source"] == "independent_audit"
                for row in pending.values()
            )
        )

    def test_provenance_limitations_are_not_overclaimed(self) -> None:
        limits = self.audit["evidence_limitations"]
        self.assertFalse(
            limits["historical_TDnet_local_receipt_timestamp_available"]
        )
        self.assertFalse(limits["historical_TDnet_archive_finality_proven"])
        self.assertFalse(
            limits["calendar_official_calendar_provenance_available"]
        )
        self.assertFalse(limits["OOT_untouched_holdout_claim"])
        self.assertIn(
            "not a claim that no edge exists", limits["interpretation"]
        )
        reconciliation = self.audit[
            "new_data_blocked_count_reconciliation"
        ]
        self.assertEqual(
            reconciliation["canonical_blocked_hypothesis_count"], 9
        )
        self.assertEqual(
            reconciliation["primary_acquisition_blocker_group_count"], 8
        )


if __name__ == "__main__":
    unittest.main()
