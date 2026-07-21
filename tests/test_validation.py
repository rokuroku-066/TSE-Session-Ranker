from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from tse_session_ranker.validation import (
    ValidationGate,
    development_gate,
    evaluate_gate,
    holdout_gate,
    max_positive_contribution,
    maximum_drawdown_pct,
    moving_block_bootstrap,
    moving_block_bootstrap_suite,
    paired_moving_block_bootstrap,
    paired_moving_block_bootstrap_suite,
    positive_period_ratio,
    profit_factor,
    return_diagnostics,
    top_k_removed_mean,
)


class ReturnDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range("2025-01-01", periods=8, freq="D")
        self.returns = pd.Series(
            [1.0, -0.5, 2.0, -1.0, 0.0, 0.5, -0.25, 0.25],
            index=self.index,
        )

    def test_diagnostics_accept_date_and_return_dataframe(self) -> None:
        frame = pd.DataFrame(
            {"date": self.index[::-1], "net_return_pct": self.returns.to_numpy()[::-1]}
        )
        result = return_diagnostics(frame, top_k=2, period_frequency=None)
        self.assertEqual(result.observations, 8)
        self.assertAlmostEqual(result.mean_pct, 0.25)
        self.assertAlmostEqual(result.top_k_removed_mean_pct, -1.0 / 6.0)
        self.assertAlmostEqual(result.profit_factor, 3.75 / 1.75)
        self.assertEqual(result.positive_periods, 4)
        self.assertEqual(result.periods, 8)
        self.assertAlmostEqual(result.positive_period_ratio, 0.5)
        self.assertAlmostEqual(result.max_positive_contribution, 2.0 / 3.75)

    def test_top_k_removed_mean_is_nan_if_no_observation_remains(self) -> None:
        self.assertTrue(math.isnan(top_k_removed_mean([1.0, 2.0], k=2)))
        self.assertAlmostEqual(top_k_removed_mean([1.0, 2.0], k=0), 1.5)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            top_k_removed_mean([1.0], k=-1)

    def test_profit_factor_all_loss_all_gain_and_all_zero(self) -> None:
        self.assertEqual(profit_factor([-1.0, -2.0, 0.0]), 0.0)
        self.assertTrue(math.isinf(profit_factor([1.0, 2.0, 0.0])))
        self.assertEqual(profit_factor([0.0, 0.0]), 0.0)
        self.assertEqual(max_positive_contribution([-1.0, 0.0]), 0.0)
        self.assertAlmostEqual(max_positive_contribution([1.0, 3.0]), 0.75)

    def test_max_drawdown_includes_initial_nav(self) -> None:
        self.assertAlmostEqual(maximum_drawdown_pct([-10.0, 20.0]), -10.0)
        self.assertEqual(maximum_drawdown_pct([1.0, 2.0]), 0.0)
        self.assertEqual(maximum_drawdown_pct([-100.0, 50.0]), -100.0)
        with self.assertRaisesRegex(ValueError, "below -100"):
            maximum_drawdown_pct([-100.01])

    def test_monthly_positive_period_ratio_uses_monthly_mean(self) -> None:
        values = pd.Series(
            [1.0, -0.5, -1.0, -2.0],
            index=pd.to_datetime(
                ["2025-01-02", "2025-01-03", "2025-02-03", "2025-02-04"]
            ),
        )
        positives, periods, ratio = positive_period_ratio(values, frequency="M")
        self.assertEqual((positives, periods), (1, 2))
        self.assertEqual(ratio, 0.5)
        with self.assertRaisesRegex(ValueError, "DatetimeIndex"):
            positive_period_ratio([1.0, -1.0], frequency="M")

    def test_nan_empty_infinite_and_duplicate_handling_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            profit_factor([1.0, np.nan])
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            profit_factor([1.0, np.inf])
        self.assertTrue(math.isinf(profit_factor([1.0, np.nan], nan_policy="drop")))
        with self.assertRaisesRegex(ValueError, "empty"):
            profit_factor([], nan_policy="drop")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            return_diagnostics(
                pd.Series([1.0, 2.0], index=pd.to_datetime(["2025-01-01"] * 2)),
                period_frequency=None,
            )


class MovingBlockBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        index = pd.date_range("2024-01-01", periods=120, freq="B")
        wave = np.tile(np.array([0.6, 0.5, 0.4, 0.3, 0.2, -0.1]), 20)
        self.positive = pd.Series(wave, index=index)

    def test_bootstrap_is_reproducible_and_interval_ordered(self) -> None:
        first = moving_block_bootstrap(
            self.positive, block_length=5, samples=500, random_state=17
        )
        second = moving_block_bootstrap(
            self.positive, block_length=5, samples=500, random_state=17
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.point_estimate_pct, self.positive.mean())
        self.assertLessEqual(first.two_sided_lower_pct, first.one_sided_lower_pct)
        self.assertLessEqual(first.one_sided_lower_pct, first.point_estimate_pct)
        self.assertLessEqual(first.two_sided_lower_pct, first.two_sided_upper_pct)
        self.assertGreater(first.bootstrap_standard_error_pct, 0.0)

    def test_suite_uses_only_requested_non_iid_blocks(self) -> None:
        result = moving_block_bootstrap_suite(
            self.positive, samples=200, block_lengths=(5, 10, 20), random_state=5
        )
        self.assertEqual(set(result), {5, 10, 20})
        self.assertEqual([result[key].random_state for key in result], [5, 6, 7])
        with self.assertRaisesRegex(ValueError, "IID bootstrap is forbidden"):
            moving_block_bootstrap(self.positive, block_length=1, samples=200)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            moving_block_bootstrap([1.0, 2.0], block_length=5, samples=200)
        with self.assertRaisesRegex(ValueError, "at least 100"):
            moving_block_bootstrap(self.positive, block_length=5, samples=99)

    def test_constant_all_gain_and_all_loss_intervals_are_explicit(self) -> None:
        gains = moving_block_bootstrap([1.0] * 30, block_length=5, samples=200)
        losses = moving_block_bootstrap([-1.0] * 30, block_length=5, samples=200)
        self.assertEqual(gains.one_sided_lower_pct, 1.0)
        self.assertEqual(gains.two_sided_upper_pct, 1.0)
        self.assertEqual(gains.bootstrap_standard_error_pct, 0.0)
        self.assertEqual(losses.one_sided_lower_pct, -1.0)

    def test_paired_bootstrap_resamples_daily_delta(self) -> None:
        baseline = self.positive - 0.25
        result = paired_moving_block_bootstrap(
            self.positive,
            baseline,
            block_length=10,
            samples=300,
            random_state=9,
        )
        self.assertAlmostEqual(result.point_estimate_delta_pct, 0.25)
        self.assertAlmostEqual(result.one_sided_lower_delta_pct, 0.25)
        self.assertAlmostEqual(result.two_sided_upper_delta_pct, 0.25)
        self.assertAlmostEqual(result.bootstrap_standard_error_delta_pct, 0.0)

    def test_paired_suite_rejects_mismatched_dates_and_nan(self) -> None:
        shifted = self.positive.copy()
        shifted.index = shifted.index + pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "identical ordered"):
            paired_moving_block_bootstrap(
                self.positive, shifted, block_length=5, samples=200
            )
        bad = self.positive.copy()
        bad.iloc[0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            paired_moving_block_bootstrap(
                bad, self.positive, block_length=5, samples=200
            )
        suite = paired_moving_block_bootstrap_suite(
            self.positive,
            self.positive - 0.1,
            block_lengths=(5, 10, 20),
            samples=200,
        )
        self.assertEqual(set(suite), {5, 10, 20})


class ValidationGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range("2024-01-01", periods=120, freq="B")
        self.strong = pd.Series(np.tile([0.30, 0.25, 0.20, 0.15], 30), index=self.index)
        self.baseline = self.strong - 0.10

    def gate(self, **overrides: object) -> ValidationGate:
        values: dict[str, object] = {
            "name": "test",
            "min_observations": 100,
            "top_k": 5,
            "period_frequency": "M",
            "min_mean_pct": 0.0,
            "min_top_k_removed_mean_pct": 0.0,
            "min_profit_factor": 1.0,
            "min_positive_period_ratio": 0.60,
            "max_positive_contribution": 0.25,
            "min_one_sided_lower_pct": 0.0,
            "block_lengths": (5, 10, 20),
            "confidence": 0.90,
            "bootstrap_samples": 200,
            "random_state": 7,
        }
        values.update(overrides)
        return ValidationGate(**values)  # type: ignore[arg-type]

    def test_passing_gate_records_every_check(self) -> None:
        decision = evaluate_gate(self.strong, self.gate())
        self.assertTrue(decision.passed)
        self.assertFalse(decision.failure_reasons)
        self.assertEqual(set(decision.bootstrap), {5, 10, 20})
        self.assertEqual(len(decision.checks), 9)
        self.assertTrue(all(item.passed for item in decision.checks))

    def test_failed_gate_does_not_short_circuit(self) -> None:
        losses = pd.Series([-0.2] * 120, index=self.index)
        decision = evaluate_gate(losses, self.gate())
        self.assertFalse(decision.passed)
        failed = {item.name for item in decision.checks if not item.passed}
        self.assertIn("mean_pct", failed)
        self.assertIn("top_k_removed_mean_pct", failed)
        self.assertIn("profit_factor", failed)
        self.assertIn("positive_period_ratio", failed)
        self.assertIn("bootstrap_20_one_sided_lower_pct", failed)
        self.assertEqual(decision.diagnostics.max_positive_contribution, 0.0)

    def test_holdout_gate_requires_baseline_and_checks_paired_delta(self) -> None:
        gate = self.gate(min_paired_delta_pct=0.0)
        missing = evaluate_gate(self.strong, gate)
        self.assertFalse(missing.passed)
        self.assertIn("paired_baseline_supplied", missing.failure_reasons[0])

        passed = evaluate_gate(self.strong, gate, baseline=self.baseline)
        self.assertTrue(passed.passed)
        self.assertIsNotNone(passed.paired_bootstrap)
        paired_check = [
            item for item in passed.checks if item.name == "paired_mean_delta_pct"
        ]
        self.assertEqual(len(paired_check), 1)
        self.assertTrue(paired_check[0].passed)

        failed = evaluate_gate(self.baseline, gate, baseline=self.strong)
        self.assertFalse(failed.passed)
        self.assertIn(
            "paired_mean_delta_pct",
            {item.name for item in failed.checks if not item.passed},
        )

    def test_paired_lower_bound_can_be_a_separate_gate(self) -> None:
        gate = self.gate(
            min_paired_delta_pct=0.0,
            min_paired_one_sided_lower_delta_pct=0.0,
        )
        decision = evaluate_gate(self.strong, gate, baseline=self.baseline)
        self.assertTrue(decision.passed)
        names = {item.name for item in decision.checks}
        self.assertIn("paired_bootstrap_5_one_sided_lower_delta_pct", names)
        self.assertIn("paired_bootstrap_20_one_sided_lower_delta_pct", names)

    def test_factory_defaults_are_finite_and_validated(self) -> None:
        self.assertEqual(development_gate(bootstrap_samples=100).name, "development")
        self.assertEqual(holdout_gate(bootstrap_samples=100).name, "holdout")
        self.assertIsNone(
            holdout_gate(
                bootstrap_samples=100, require_paired_improvement=False
            ).min_paired_delta_pct
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            ValidationGate(name="bad", min_observations=0)
        with self.assertRaisesRegex(ValueError, "IID bootstrap is forbidden"):
            ValidationGate(
                name="bad", min_observations=10, block_lengths=(1,)
            )


if __name__ == "__main__":
    unittest.main()
