#!/usr/bin/env python3
"""Fast integrity checks for the integrated model-v10 research artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
FORWARD_PROTOCOL_NAME = "model_v10_forward_protocol.json"
PRE_ENTRY_015_VALIDATION_SHA256 = (
    "ceb3ea66e2806330e3357c5dbacdca1c2bd6d3a1e90a85cdad384193808ca5e5"
)

RESULT_PROTOCOL_PAIRS = {
    "model_v10_architectures_result.json": "model_v10_architectures_protocol.json",
    "model_v10_architectures_result_repro.json": (
        "model_v10_architectures_protocol.json"
    ),
    "model_v10_external_context_result.json": (
        "model_v10_external_context_protocol.json"
    ),
    "model_v10_literature_result.json": "model_v10_literature_protocol.json",
    "model_v10_market_residual_result.json": (
        "model_v10_market_residual_protocol.json"
    ),
    "model_v10_online_ensemble_result.json": (
        "model_v10_online_ensemble_protocol.json"
    ),
    "model_v10_peer_residual_result.json": (
        "model_v10_peer_residual_protocol.json"
    ),
    "model_v10_policy_universe_result.json": (
        "model_v10_policy_universe_protocol.json"
    ),
    "model_v10_policy_universe_round2_result.json": (
        "model_v10_policy_universe_round2_protocol.json"
    ),
    "model_v10_policy_universe_round3_result.json": (
        "model_v10_policy_universe_round3_protocol.json"
    ),
    "model_v10_tdnet_text_result.json": "model_v10_tdnet_text_protocol.json",
    "model_v10_zero_base_gen1_result.json": "model_v10_zero_base_protocol.json",
}


def read_json(name: str) -> dict:
    with (RESEARCH / name).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AssertionError(f"{name} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authority_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()


class ModelV10ArtifactTests(unittest.TestCase):
    def test_every_model_v10_json_artifact_parses(self) -> None:
        paths = sorted(RESEARCH.glob("model_v10*.json"))
        self.assertTrue(paths, "no model-v10 JSON artifacts found")
        for path in paths:
            with self.subTest(artifact=path.name):
                with path.open("r", encoding="utf-8") as handle:
                    value = json.load(handle)
                self.assertIsInstance(value, dict)

    def test_protocol_result_authority_and_bindings_are_preserved(self) -> None:
        protocols = sorted(RESEARCH.glob("model_v10*protocol.json"))
        self.assertTrue(protocols, "no model-v10 protocols found")
        result_names = {
            path.name for path in RESEARCH.glob("model_v10*result*.json")
        }
        self.assertEqual(
            result_names,
            set(RESULT_PROTOCOL_PAIRS),
            "every integrated result must have an explicit protocol binding",
        )

        for path in protocols:
            with self.subTest(protocol=path.name):
                protocol = read_json(path.name)
                authority = protocol["authority"]
                if path.name == FORWARD_PROTOCOL_NAME:
                    self.assertIn("prospective", authority_text(authority))
                    self.assertIs(
                        authority["production_promotion_allowed_during_run"],
                        False,
                    )
                    self.assertIs(authority["orders_allowed"], False)
                    continue
                self.assertIn("retrospective", authority_text(authority))
                if isinstance(authority, dict):
                    self.assertIs(
                        authority["production_promotion_allowed"],
                        False,
                    )
                else:
                    self.assertIs(
                        protocol["production_promotion_allowed"],
                        False,
                    )

        for result_name, protocol_name in RESULT_PROTOCOL_PAIRS.items():
            with self.subTest(result=result_name):
                result = read_json(result_name)
                protocol = read_json(protocol_name)
                self.assertEqual(result["protocol_id"], protocol["protocol_id"])

                if "protocol_sha256" in result:
                    self.assertEqual(
                        result["protocol_sha256"],
                        sha256_file(RESEARCH / protocol_name),
                    )

                result_authority = result.get("authority")
                if result_authority is not None:
                    marker = authority_text(result_authority)
                    self.assertTrue(
                        "retrospective" in marker or "exploratory" in marker,
                        marker,
                    )

                no_production_markers: list[bool] = []
                if "production_model_changed" in result:
                    self.assertIs(result["production_model_changed"], False)
                    no_production_markers.append(True)
                if "production_promotion_allowed" in result:
                    self.assertIs(
                        result["production_promotion_allowed"],
                        False,
                    )
                    no_production_markers.append(True)

                decision = result.get("decision")
                if isinstance(decision, dict) and "production_change" in decision:
                    self.assertIs(decision["production_change"], False)
                    no_production_markers.append(True)

                if isinstance(result_authority, dict):
                    if "production_promotion_allowed" in result_authority:
                        self.assertIs(
                            result_authority["production_promotion_allowed"],
                            False,
                        )
                        no_production_markers.append(True)
                elif isinstance(result_authority, str):
                    marker = result_authority.lower()
                    if "no_promotion" in marker or "no_production" in marker:
                        no_production_markers.append(True)

                self.assertTrue(
                    no_production_markers,
                    f"{result_name} lacks an explicit no-production marker",
                )

    def test_architecture_family_has_no_passing_candidate(self) -> None:
        protocol = read_json("model_v10_architectures_protocol.json")
        control_id = protocol["control"]["id"]

        for result_name in (
            "model_v10_architectures_result.json",
            "model_v10_architectures_result_repro.json",
        ):
            with self.subTest(result=result_name):
                result = read_json(result_name)
                candidate_ids = set(result["metrics"]) - {control_id}
                gates = result["robustness_gates"]

                self.assertEqual(set(gates), candidate_ids)
                self.assertTrue(candidate_ids)
                self.assertTrue(
                    all(gate["passes_all"] is False for gate in gates.values())
                )
                self.assertEqual(result["decision"]["passing_candidate_ids"], [])
                self.assertIsNone(result["decision"]["robustness_gate_winner"])
                self.assertEqual(result["decision"]["production_action"], "none")
                self.assertIs(result["production_model_changed"], False)

    def test_tdnet_t02_point_metrics_and_coverage_are_consistent(self) -> None:
        protocol = read_json("model_v10_tdnet_text_protocol.json")
        result = read_json("model_v10_tdnet_text_result.json")
        audit = read_json("model_v10_tdnet_text_audit.json")

        score_sessions = result["coverage"]["strict_complete_score_sessions"]
        self.assertEqual(score_sessions, 88)
        self.assertEqual(
            sum(fold["score_sessions"] for fold in result["folds"]),
            score_sessions,
        )
        self.assertEqual(
            score_sessions + result["coverage"]["source_missing_score_sessions"],
            266,
        )
        self.assertEqual(
            result["coverage"]["score_days_with_event_bundle"],
            score_sessions,
        )

        t02 = result["candidates"]["T02_char_value_event_only"]
        top1 = t02["top1"]
        gross_mean = top1["costs"]["0"]["mean_pct"]
        for cost_bps, point_metrics in top1["costs"].items():
            with self.subTest(cost_bps=cost_bps):
                self.assertEqual(point_metrics["days"], score_sessions)
                self.assertAlmostEqual(
                    point_metrics["mean_pct"],
                    gross_mean - int(cost_bps) / 100,
                    places=12,
                )
        self.assertEqual(top1["filled_slots"], score_sessions)
        self.assertEqual(top1["scheduled_slots"], score_sessions)
        self.assertEqual(
            top1["days_with_at_least_one_filled_slot"],
            score_sessions,
        )

        audit_top1 = audit["top_k"]["top1"]
        audit_t02 = audit_top1["candidates"]["T02_char_value_event_only"]
        t02_net40 = top1["costs"]["40"]["mean_pct"]
        l4_net40 = result["candidates"]["L4_price_control"]["top1"]["costs"][
            "40"
        ]["mean_pct"]
        self.assertEqual(audit_top1["sessions"], score_sessions)
        self.assertEqual(
            audit_top1["best_candidate_by_delta"],
            "T02_char_value_event_only",
        )
        self.assertAlmostEqual(
            audit_t02["net40_mean_pct"],
            t02_net40,
            places=15,
        )
        self.assertAlmostEqual(
            audit_t02["delta_vs_L4_net40_mean_pct"],
            t02_net40 - l4_net40,
            places=15,
        )

        self.assertIs(
            protocol["authority"]["production_promotion_allowed"],
            False,
        )
        self.assertIs(protocol["promotion_gate"]["enabled"], False)
        self.assertEqual(
            result["authority"],
            "exploratory_only_no_production_promotion",
        )
        self.assertLess(
            top1["slices"]["confirmation_b"]["net40_mean_pct"],
            0,
        )
        self.assertLess(top1["top20_winning_days_removed_net20_mean_pct"], 0)
        self.assertLess(top1["top10_profit_codes_cash_net20_mean_pct"], 0)
        self.assertLess(
            audit_t02["delta_vs_L4_90pct_block_interval"][0],
            0,
        )
        self.assertGreater(
            audit_top1["familywise_reality_check_p_value"],
            0.10,
        )

    def test_forward_protocol_is_frozen_to_t02_artifacts(self) -> None:
        forward = read_json(FORWARD_PROTOCOL_NAME)
        bound = forward["bound_retrospective_artifacts"]

        for artifact in bound.values():
            with self.subTest(artifact=artifact["path"]):
                path = ROOT / artifact["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(artifact["sha256"], sha256_file(path))

        self.assertEqual(
            forward["candidate"]["id"],
            "v10_t02_char_value_event_top1",
        )
        self.assertEqual(
            forward["prospective_window"]["start_session"],
            "2026-07-24",
        )
        self.assertEqual(
            forward["prospective_window"]["minimum_source_complete_sessions"],
            120,
        )
        self.assertTrue(forward["prospective_window"]["changes_reset_counter"])
        self.assertEqual(
            forward["candidate"]["decision"]["no_qualifying_event"],
            "cash",
        )
        self.assertIsNone(forward["candidate"]["decision"]["fallback"])
        self.assertIs(
            forward["authority"]["production_model_changed"],
            False,
        )

    def test_validation_entry_015_is_append_only(self) -> None:
        text = (ROOT / "VALIDATION.md").read_text(encoding="utf-8")
        marker = "\n---\n\n## Entry 015 "

        self.assertEqual(text.count("## Entry 015 "), 1)
        self.assertIn(marker, text)
        prefix = text[: text.index(marker)]
        self.assertEqual(
            hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
            PRE_ENTRY_015_VALIDATION_SHA256,
        )
        self.assertGreater(
            text.index("## Entry 015 "),
            text.index("## Entry 014 "),
        )

    def test_literature_h09_feasibility_failure_is_represented(self) -> None:
        result = read_json("model_v10_literature_result.json")
        tdnet_qa = result["source_qa"]["tdnet"]
        h09_features = result["feature_blocks"]["tdnet"]["H09"]
        report = (
            RESEARCH / "model_v10_literature_validation_report.md"
        ).read_text(encoding="utf-8")

        self.assertEqual(tdnet_qa["waves_seen"], 1)
        self.assertEqual(
            tdnet_qa["completed_wave_denominator_available_dates"],
            0,
        )
        self.assertIn("h09_earnings_x_wave_progress", h09_features)
        self.assertIn("h09_earnings_x_prior_wave_size", h09_features)
        self.assertRegex(report, r"H09.*feasibility failure")
        self.assertIn("H09 | 波特徴のfeasibility failure。検証不能", report)

    def test_round2_retain_flag_is_invalidated_by_separate_audit(self) -> None:
        protocol = read_json("model_v10_policy_universe_round2_protocol.json")
        result = read_json("model_v10_policy_universe_round2_result.json")
        audit_path = (
            RESEARCH
            / "model_v10_policy_universe_round2_integration_audit.md"
        )
        audit = audit_path.read_text(encoding="utf-8")

        self.assertIn(
            "at least four of five unrelated split variables",
            protocol["interpretation"]["specificity_support"],
        )
        self.assertEqual(
            result["mechanism_checks"][
                "specificity_total_unrelated_splits"
            ],
            4,
        )
        self.assertIs(result["decision"]["retain_for_forward_shadow"], True)
        self.assertIs(result["decision"]["production_change"], False)
        self.assertIn(
            "Status: **`retain_for_forward_shadow` invalidated**",
            audit,
        )
        self.assertIn(
            "`decision.retain_for_forward_shadow = true` is invalidated",
            audit,
        )
        self.assertIn(
            "This executes a 4-of-4 check, not the preregistered 4-of-5 check.",
            audit,
        )

        quoted_hashes = dict(
            re.findall(
                r"^- (Protocol|Runner|Result|Report): `([0-9a-f]{64})`$",
                audit,
                flags=re.MULTILINE,
            )
        )
        locked_paths = {
            "Protocol": RESEARCH
            / "model_v10_policy_universe_round2_protocol.json",
            "Runner": RESEARCH
            / "model_v10_policy_universe_round2_runner.py",
            "Result": RESEARCH
            / "model_v10_policy_universe_round2_result.json",
            "Report": RESEARCH
            / "model_v10_policy_universe_round2_report.md",
        }
        self.assertEqual(set(quoted_hashes), set(locked_paths))
        for label, path in locked_paths.items():
            with self.subTest(locked_artifact=label):
                self.assertEqual(quoted_hashes[label], sha256_file(path))

    def test_external_online_and_residual_leave_production_unchanged(
        self,
    ) -> None:
        external_protocol = read_json(
            "model_v10_external_context_protocol.json"
        )
        external_result = read_json("model_v10_external_context_result.json")
        self.assertIs(
            external_protocol["authority"]["production_promotion_allowed"],
            False,
        )
        self.assertEqual(
            external_result["authority"],
            "retrospective_new_data_source_screen_no_promotion",
        )

        for family in ("online_ensemble", "market_residual", "peer_residual"):
            with self.subTest(family=family):
                protocol = read_json(f"model_v10_{family}_protocol.json")
                result = read_json(f"model_v10_{family}_result.json")
                self.assertIs(
                    protocol["authority"]["production_promotion_allowed"],
                    False,
                )
                self.assertIs(result["production_model_changed"], False)

        peer_result = read_json("model_v10_peer_residual_result.json")
        self.assertEqual(peer_result["production_promotion"], "none")
        self.assertIn("Retrospective", peer_result["interpretation"])

    def test_preopen_schema_records_exact_085859_pit_contract(self) -> None:
        schema = read_json("model_v10_preopen_schema.json")
        record = schema["$defs"]["record"]
        properties = record["properties"]
        required = set(record["required"])
        report = (
            RESEARCH / "model_v10_preopen_data_report.md"
        ).read_text(encoding="utf-8")

        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            "preopen-pit-v1",
        )
        self.assertIn("08:58:59 Asia/Tokyo decision", schema["description"])
        self.assertTrue(
            {
                "target_date",
                "source_event_at",
                "source_received_at",
                "computed_at",
            }.issubset(required)
        )
        self.assertEqual(
            schema["$defs"]["aware_timestamp"]["pattern"],
            r"(Z|[+-][0-9]{2}:[0-9]{2})$",
        )
        self.assertEqual(
            properties["source_received_at"]["$ref"],
            "#/$defs/aware_timestamp",
        )
        self.assertEqual(
            properties["computed_at"]["$ref"],
            "#/$defs/aware_timestamp",
        )
        self.assertIn(
            "available_at=max(source_received_at,computed_at)",
            properties["computed_at"]["description"],
        )
        self.assertIn(
            "This alone never proves model availability",
            properties["source_event_at"]["description"],
        )
        for invariant in (
            "available_at = max(source_received_at, computed_at)",
            "available_at <= 08:58:59 Asia/Tokyo",
            "source_event_at <= source_received_at <= computed_at",
        ):
            with self.subTest(invariant=invariant):
                self.assertIn(invariant, report)


if __name__ == "__main__":
    unittest.main()
