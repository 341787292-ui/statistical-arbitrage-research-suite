from __future__ import annotations

import unittest

import numpy as np

from ashare_stat_arb.config import CostConfig, PortfolioConfig
from ashare_stat_arb.phase2 import (
    PHASE2_MAPPING,
    conservative_round_trip_cost,
    validate_development_panel,
)


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


if __name__ == "__main__":
    unittest.main()
