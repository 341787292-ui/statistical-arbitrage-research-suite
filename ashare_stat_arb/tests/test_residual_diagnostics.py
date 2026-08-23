from __future__ import annotations

import unittest

import numpy as np

from ashare_stat_arb.residual_diagnostics import (
    evaluate_residual_positions,
    ou_residual_positions,
)


class ResidualMechanismTests(unittest.TestCase):
    def test_evaluation_uses_next_residual_return(self) -> None:
        residuals = np.zeros((8, 3), dtype=np.float64)
        positions = np.zeros_like(residuals)
        positions[2, 0] = 1.0
        residuals[3, 0] = 0.01
        result = evaluate_residual_positions(residuals, positions, start=2)
        self.assertGreater(result.annualized_mean, 0.0)
        reversed_result = evaluate_residual_positions(
            residuals,
            positions,
            start=2,
            direction=-1.0,
        )
        self.assertAlmostEqual(result.annualized_mean, -reversed_result.annualized_mean)

    def test_ou_positions_do_not_use_future_returns(self) -> None:
        rng = np.random.default_rng(42)
        residuals = rng.normal(0.0, 0.01, size=(50, 4))
        original = ou_residual_positions(
            residuals,
            lookback=12,
            min_r_squared=-1.0,
        )
        changed = residuals.copy()
        changed[40:] += 100.0
        revised = ou_residual_positions(
            changed,
            lookback=12,
            min_r_squared=-1.0,
        )
        np.testing.assert_allclose(original[:40], revised[:40])


if __name__ == "__main__":
    unittest.main()
