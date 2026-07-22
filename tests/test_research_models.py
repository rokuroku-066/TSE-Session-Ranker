from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from research.compare_model_families_v05 import (
    CandidateSpec,
    EnsembleSpec,
    _acquire_run_lock,
    _block_uses_tdnet,
    _desired_slots,
    _metrics,
    _selectable,
    _source_mask,
    evaluate_ensemble,
    feature_blocks,
)
from tse_session_ranker.research_models import (
    ResearchModelSpec,
    date_and_class_equal_weights,
    date_equal_weights,
    fit_research_model,
    same_day_return_percentile_target,
)


def _synthetic_training_frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260722)
    dates = pd.bdate_range("2024-01-04", periods=14)
    rows: list[dict[str, object]] = []
    for day_number, date in enumerate(dates):
        for stock_number in range(24):
            first = rng.normal()
            second = rng.normal()
            realised = (
                0.55 * first
                - 0.20 * second
                + 0.35 * np.sin(first * second)
                + rng.normal(scale=0.55)
            )
            rows.append(
                {
                    "date": date,
                    "code": f"{stock_number:04d}",
                    "feature_one": first,
                    "feature_two": second,
                    "feature_three": float(day_number % 5),
                    "oc_return_pct": realised,
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[frame.index[::29], "feature_two"] = np.nan
    return frame


class ResearchModelSpecTests(unittest.TestCase):
    def test_spec_id_is_deterministic_and_binds_parameters(self) -> None:
        first = ResearchModelSpec(
            "ridge", "ridge_return", {"alpha": 10.0, "return_clip_pct": 5.0}
        )
        reordered = ResearchModelSpec(
            "ridge", "ridge_return", {"return_clip_pct": 5.0, "alpha": 10.0}
        )
        changed = ResearchModelSpec(
            "ridge", "ridge_return", {"alpha": 100.0, "return_clip_pct": 5.0}
        )
        self.assertEqual(first.spec_id, reordered.spec_id)
        self.assertNotEqual(first.spec_id, changed.spec_id)

    def test_spec_defensively_freezes_nested_parameters_and_rejects_typos(self) -> None:
        thresholds = [-1.0, 0.0, 1.0]
        spec = ResearchModelSpec(
            "ordinal",
            "multi_threshold_expected_return",
            {"thresholds_pct": thresholds, "return_clip_pct": 2.0},
            objective="threshold_integrated_return",
        )
        before = spec.spec_id
        thresholds[0] = -1.5
        self.assertEqual(spec.spec_id, before)
        self.assertEqual(spec.canonical_dict()["parameters"]["thresholds_pct"][0], -1.0)
        with self.assertRaisesRegex(ValueError, "unknown parameters"):
            ResearchModelSpec("bad", "ridge_return", {"alpah": 10.0})

    def test_multi_threshold_spec_requires_strict_in_clip_values(self) -> None:
        for thresholds in ([], [-1.0, -1.0, 1.0], [-3.0, 0.0, 1.0]):
            with self.subTest(thresholds=thresholds):
                with self.assertRaisesRegex(ValueError, "strictly ordered"):
                    ResearchModelSpec(
                        "bad_ordinal",
                        "multi_threshold_expected_return",
                        {"thresholds_pct": thresholds, "return_clip_pct": 2.0},
                        objective="threshold_integrated_return",
                    )

    def test_family_objective_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires objective"):
            ResearchModelSpec("bad_ridge", "ridge_return", {}, objective="positive_session")

    def test_date_equal_weights_have_exact_daily_margin(self) -> None:
        frame = _synthetic_training_frame()
        magnitude = frame["oc_return_pct"].abs().add(0.1)
        weights = date_equal_weights(frame, magnitude)
        totals = weights.groupby(frame["date"]).sum()
        np.testing.assert_allclose(totals, 1.0, atol=1e-12)

    def test_raked_weights_have_daily_and_class_margins(self) -> None:
        frame = _synthetic_training_frame()
        target = frame["oc_return_pct"].gt(0).astype(int)
        weights = date_and_class_equal_weights(frame, target)
        daily = weights.groupby(frame["date"]).sum()
        classes = weights.groupby(target).sum().sort_index()
        np.testing.assert_allclose(daily, 1.0, atol=1e-12)
        np.testing.assert_allclose(
            classes,
            np.repeat(frame["date"].nunique() / 2.0, 2),
            atol=1e-8,
        )

    def test_daily_rank_target_is_date_local_and_monotone_invariant(self) -> None:
        frame = _synthetic_training_frame()
        target = same_day_return_percentile_target(frame)
        changed = frame.copy()
        per_date_shift = changed["date"].factorize()[0] * 100.0
        changed["oc_return_pct"] = np.exp(
            changed["oc_return_pct"] / 10.0
        ) + per_date_shift
        transformed = same_day_return_percentile_target(changed)
        np.testing.assert_allclose(target, transformed, atol=0.0, rtol=0.0)
        self.assertGreaterEqual(float(target.min()), -1.0)
        self.assertLessEqual(float(target.max()), 1.0)
        counts = frame.groupby("date")["date"].transform("size")
        expected_daily_mean = counts.groupby(frame["date"]).first().rdiv(1.0)
        actual_daily_mean = target.groupby(frame["date"]).mean()
        np.testing.assert_allclose(actual_daily_mean, expected_daily_mean, atol=1e-12)


class BroadResearchFamilyTests(unittest.TestCase):
    def test_all_registered_families_produce_finite_deterministic_scores(self) -> None:
        frame = _synthetic_training_frame()
        features = ("feature_one", "feature_two", "feature_three")
        registry = (
            ("legacy_weight_logit", "positive_session", {"C": 0.08}),
            ("raked_logit", "net_positive_20bp", {"C": 0.03}),
            (
                "return_weighted_logit",
                "net_positive_20bp",
                {"C": 0.03, "magnitude_floor_pct": 0.1, "magnitude_cap_pct": 2.0},
            ),
            ("ridge_return", "raw_return", {"alpha": 10.0}),
            (
                "ridge_daily_rank",
                "same_day_return_percentile",
                {"alpha": 10.0},
            ),
            (
                "elastic_net_sgd_return",
                "raw_return",
                {"max_iter": 300, "tol": 1e-3, "average": True},
            ),
            (
                "huber_sgd_return",
                "raw_return",
                {"max_iter": 300, "tol": 1e-3, "average": True},
            ),
            (
                "hist_gradient_boosting_return",
                "raw_return",
                {"max_iter": 8, "min_samples_leaf": 4, "max_leaf_nodes": 7},
            ),
            (
                "hist_gradient_boosting_classifier",
                "same_day_top_quintile",
                {"max_iter": 8, "min_samples_leaf": 4, "max_leaf_nodes": 7},
            ),
            (
                "hist_gradient_boosting_rank",
                "same_day_return_percentile",
                {"max_iter": 8, "min_samples_leaf": 4, "max_leaf_nodes": 7},
            ),
            (
                "extra_trees_return",
                "raw_return",
                {"n_estimators": 8, "min_samples_leaf": 2, "n_jobs": 1},
            ),
            (
                "extra_trees_classifier",
                "same_day_top_quintile",
                {"n_estimators": 8, "min_samples_leaf": 2, "n_jobs": 1},
            ),
            (
                "random_forest_return",
                "raw_return",
                {"n_estimators": 8, "min_samples_leaf": 2, "n_jobs": 1},
            ),
            (
                "pairwise_linear_rank",
                "same_day_pairwise_return",
                {"pairs_per_date": 12, "max_iter": 300},
            ),
            (
                "multi_threshold_expected_return",
                "threshold_integrated_return",
                {
                    "thresholds_pct": [-1.0, -0.2, 0.2, 1.0],
                    "return_clip_pct": 2.0,
                    "max_iter": 300,
                },
            ),
        )
        scoring = frame.iloc[-31:].copy()
        for family, objective, parameters in registry:
            with self.subTest(family=family):
                spec = ResearchModelSpec(
                    name=f"test_{family}",
                    family=family,
                    objective=objective,
                    parameters=parameters,
                    random_state=17,
                )
                first = fit_research_model(spec, frame, features).score(scoring)
                second = fit_research_model(spec, frame, features).score(scoring)
                self.assertEqual(first.shape, (len(scoring),))
                self.assertTrue(np.isfinite(first).all())
                np.testing.assert_allclose(first, second, atol=1e-12)

    def test_tree_research_recipe_does_not_uniformly_bootstrap_rows(self) -> None:
        frame = _synthetic_training_frame()
        spec = ResearchModelSpec(
            "tree",
            "extra_trees_return",
            {"n_estimators": 4, "min_samples_leaf": 2, "bootstrap": False},
        )
        fitted = fit_research_model(
            spec, frame, ("feature_one", "feature_two", "feature_three")
        )
        self.assertFalse(fitted.estimator.named_steps["model"].bootstrap)


class ScheduledSlotTests(unittest.TestCase):
    def test_metric_run_lock_is_exclusive_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.run.lock"
            digest = _acquire_run_lock(path, {"protocol": "abc"})
            self.assertEqual(len(digest), 64)
            with self.assertRaises(FileExistsError):
                _acquire_run_lock(path, {"protocol": "abc"})

    def test_tdnet_completeness_only_gates_the_tdnet_feature_block(self) -> None:
        frame = pd.DataFrame(
            {
                "price_eligible": [True, True],
                "price_training_eligible": [True, False],
                "tdnet_source_complete": [False, True],
            }
        )
        self.assertEqual(
            _source_mask(frame, "session_market", training=False).tolist(),
            [True, True],
        )
        self.assertEqual(
            _source_mask(frame, "session_market_tdnet", training=False).tolist(),
            [False, True],
        )
        self.assertEqual(
            _source_mask(frame, "session_market_tdnet", training=True).tolist(),
            [False, False],
        )
        expected = {
            "legacy_price_12": False,
            "legacy_v03_18": True,
            "session_market": False,
            "session_market_tdnet": True,
        }
        self.assertEqual(
            {block: _block_uses_tdnet(block) for block in expected}, expected
        )
        self.assertEqual(
            _source_mask(frame, "legacy_v03_18", training=False).tolist(),
            [False, True],
        )

    def test_metrics_keep_no_candidate_session_as_cash(self) -> None:
        sessions = pd.bdate_range("2025-01-06", periods=8)
        actual = pd.DataFrame(
            {
                "date": [sessions[0], sessions[0], sessions[2]],
                "model_rank": [1, 2, 1],
                "code": ["1001", "1002", "1003"],
                "label": [1.0, 0.0, 1.0],
                "oc_return_pct": [1.0, -0.5, 0.7],
            }
        )
        picks = _desired_slots(sessions).merge(
            actual,
            on=["date", "model_rank"],
            how="left",
            validate="one_to_one",
        )
        result = _metrics(picks, bootstrap_samples=100, seed=5)
        self.assertEqual(result["top1"]["net20"]["days"], 8)
        self.assertEqual(result["top2"]["net20"]["days"], 8)
        self.assertEqual(result["top1"]["net20"]["executed"], 2)
        self.assertEqual(result["top2"]["net20"]["executed"], 3)
        self.assertAlmostEqual(
            result["top1"]["net20"]["net_mean_pct_at_cost"],
            ((1.0 - 0.2) + (0.7 - 0.2)) / 8,
        )

    def test_all_cash_result_is_not_selectable(self) -> None:
        result = {
            "status": "completed",
            "metrics": {"display": {"rank1_rate": 0.0, "rank2_rate": 0.0}},
        }
        self.assertEqual(_selectable([result]), [])

    def test_ensemble_handles_an_entirely_empty_scoring_period_as_cash(self) -> None:
        rng = np.random.default_rng(44)
        dates = pd.bdate_range("2024-01-04", periods=90)
        training_end = dates[84]
        features = feature_blocks()["legacy_price_12"]
        rows: list[dict[str, object]] = []
        for date in dates:
            for code in ("1001", "1002", "1003"):
                realised = rng.normal()
                row: dict[str, object] = {
                    "date": date,
                    "code": code,
                    "name": code,
                    "eligible": date <= training_end,
                    "training_eligible": date <= training_end,
                    "price_eligible": date <= training_end,
                    "price_training_eligible": date <= training_end,
                    "tdnet_source_complete": True,
                    "label": float(realised > 0),
                    "oc_return_pct": realised,
                    "outcome_observed": True,
                    "source_complete": True,
                    "universe_source_complete": True,
                }
                row.update({column: rng.normal() for column in features})
                rows.append(row)
        panel = pd.DataFrame(rows)
        member = CandidateSpec(
            ResearchModelSpec("ridge", "ridge_return", {"alpha": 10.0}),
            "legacy_price_12",
        )
        result, picks = evaluate_ensemble(
            panel,
            EnsembleSpec((member,)),
            evaluation_start=dates[85],
            evaluation_end=dates[-1],
            sessions=dates,
            bootstrap_samples=100,
        )
        self.assertEqual(len(picks), 10)
        self.assertTrue(picks["code"].isna().all())
        self.assertEqual(result["metrics"]["display"]["rank2_rate"], 0.0)
        self.assertEqual(result["metrics"]["top2"]["net20"]["days"], 5)


if __name__ == "__main__":
    unittest.main()
