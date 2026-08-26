from __future__ import annotations

import unittest

import numpy as np

from ashare_stat_arb.admission import (
    PeriodMigrationMetrics,
    decide_migration,
)
from ashare_stat_arb.config import AdmissionConfig
from ashare_stat_arb.signals import cross_sectional_residual_rank_alpha


class ResidualRankAlphaTests(unittest.TestCase):
    def test_uses_only_trailing_information_and_preserves_reversal_order(self) -> None:
        residuals = np.tile(np.linspace(-0.02, 0.02, 30), (12, 1))
        member = np.ones_like(residuals, dtype=bool)
        original = cross_sectional_residual_rank_alpha(
            residuals,
            member,
            horizon=5,
            minimum_cross_section=20,
        )
        changed = residuals.copy()
        changed[9:] *= -10.0
        revised = cross_sectional_residual_rank_alpha(
            changed,
            member,
            horizon=5,
            minimum_cross_section=20,
        )

        np.testing.assert_allclose(original[:9], revised[:9])
        self.assertGreater(original[8, 0], original[8, -1])
        self.assertAlmostEqual(float(original[8].mean()), 0.0, places=12)
        self.assertAlmostEqual(float(original[8].std(ddof=0)), 1.0, places=12)

    def test_missing_or_nonmember_names_receive_zero(self) -> None:
        residuals = np.tile(np.linspace(-0.02, 0.02, 25), (8, 1))
        member = np.ones_like(residuals, dtype=bool)
        member[6, 0] = False
        residuals[5, 1] = np.nan
        alpha = cross_sectional_residual_rank_alpha(
            residuals,
            member,
            horizon=5,
            minimum_cross_section=20,
        )
        self.assertEqual(float(alpha[6, 0]), 0.0)
        self.assertEqual(float(alpha[6, 1]), 0.0)


class MigrationAdmissionTests(unittest.TestCase):
    @staticmethod
    def _metrics(period: str, *, passed: bool) -> PeriodMigrationMetrics:
        return PeriodMigrationMetrics(
            period=period,
            start_date="2020-01-01",
            end_date="2021-12-31",
            observations=504,
            annualized_benchmark_return=0.08,
            annualized_gross_strategy_return=0.11 if passed else 0.07,
            annualized_net_strategy_return=0.10 if passed else 0.06,
            annualized_gross_excess_return=0.03 if passed else -0.01,
            annualized_net_excess_return=0.02 if passed else -0.02,
            gross_information_ratio=1.4 if passed else -0.4,
            net_information_ratio=1.1 if passed else -0.8,
            annualized_cost_drag=0.01,
            cost_share_of_gross_alpha=0.33 if passed else None,
            maximum_active_drawdown=-0.08 if passed else -0.15,
            positive_rolling_12m_share=0.70 if passed else 0.40,
        )

    def test_all_frozen_gates_must_pass(self) -> None:
        passed = self._metrics("development", passed=True)
        result = decide_migration(
            passed,
            self._metrics("validation", passed=False),
            self._metrics("overall", passed=True),
            AdmissionConfig(),
        )
        self.assertFalse(result.migration_supported)
        self.assertIn("sealed", result.holdout_action)


if __name__ == "__main__":
    unittest.main()
