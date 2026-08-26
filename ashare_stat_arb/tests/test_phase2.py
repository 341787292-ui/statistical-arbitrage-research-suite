from __future__ import annotations

import unittest

import numpy as np

from ashare_stat_arb.config import CostConfig, PortfolioConfig
from ashare_stat_arb.phase2 import (
    PHASE2_MAPPING,
    conservative_round_trip_cost,
    validate_development_panel,
)
from ashare_stat_arb.phase3 import PHASE3_MAPPING
from ashare_stat_arb.signals import buffered_residual_rank_alpha


class Phase2ContractTests(unittest.TestCase):
    def test_mapping_matches_registered_contract(self) -> None:
        portfolio = PHASE2_MAPPING.portfolio(PortfolioConfig())
        self.assertEqual(PHASE2_MAPPING.signal_horizon, 5)
        self.assertEqual(PHASE2_MAPPING.decision_interval, 5)
        self.assertEqual(portfolio.maximum_two_way_turnover, 0.05)
        self.assertEqual(portfolio.target_two_way_turnover, 0.05)

    def test_cost_penalty_matches_declared_round_trip_cost(self) -> None:
        costs = CostConfig()
        expected = 2.0 * (0.00025 + 0.00001 + 0.001) + 0.001
        self.assertAlmostEqual(conservative_round_trip_cost(costs), expected)
        self.assertAlmostEqual(expected, 0.00352)

    def test_sealed_holdout_is_rejected(self) -> None:
        validate_development_panel(
            np.array(["2018-01-02", "2022-12-30"], dtype="datetime64[D]")
        )
        with self.assertRaises(ValueError):
            validate_development_panel(
                np.array(["2018-01-02", "2023-01-03"], dtype="datetime64[D]")
            )


class Phase3ContractTests(unittest.TestCase):
    def test_mapping_matches_registered_contract(self) -> None:
        portfolio = PHASE3_MAPPING.portfolio(PortfolioConfig())
        self.assertEqual(PHASE3_MAPPING.signal_horizon, 5)
        self.assertEqual(PHASE3_MAPPING.positive_entry, 0.80)
        self.assertEqual(PHASE3_MAPPING.positive_exit, 0.60)
        self.assertEqual(PHASE3_MAPPING.negative_entry, 0.20)
        self.assertEqual(PHASE3_MAPPING.negative_exit, 0.40)
        self.assertEqual(portfolio.maximum_two_way_turnover, 0.05)

    def test_buffer_retains_then_exits_a_positive_state(self) -> None:
        scores = np.tile(np.arange(20, dtype=float), (3, 1))
        scores[1, -1] = 14.5
        scores[2, -1] = 9.5
        residuals = -scores
        member = np.ones_like(residuals, dtype=bool)
        alpha = buffered_residual_rank_alpha(
            residuals,
            member,
            horizon=1,
            minimum_cross_section=20,
        )

        self.assertGreater(alpha[0, -1], 0.0)
        self.assertGreater(alpha[1, -1], 0.0)
        self.assertLess(abs(alpha[2, -1]), abs(alpha[1, -1]))

    def test_buffer_has_no_future_dependency(self) -> None:
        residuals = np.tile(np.linspace(-0.02, 0.02, 20), (8, 1))
        member = np.ones_like(residuals, dtype=bool)
        original = buffered_residual_rank_alpha(
            residuals,
            member,
            horizon=3,
            minimum_cross_section=20,
        )
        changed = residuals.copy()
        changed[6:] *= -20.0
        revised = buffered_residual_rank_alpha(
            changed,
            member,
            horizon=3,
            minimum_cross_section=20,
        )
        np.testing.assert_allclose(original[:6], revised[:6])


if __name__ == "__main__":
    unittest.main()
