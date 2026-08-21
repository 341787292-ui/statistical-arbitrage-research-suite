from __future__ import annotations

import unittest

import numpy as np

from paper_reproduction.dlsa.analysis import (
    concatenate_blocks,
    performance_metrics,
    rolling_annualized_sharpe,
)


class StabilityAnalysisTest(unittest.TestCase):
    def test_performance_metrics_reconcile_mean_and_sharpe(self) -> None:
        returns = np.array([0.01, -0.005, 0.004, 0.002], dtype=float)
        metrics = performance_metrics(returns)
        expected_mean = returns.mean() * 252
        expected_volatility = returns.std(ddof=0) * np.sqrt(252)
        self.assertAlmostEqual(metrics["annualized_mean"], expected_mean)
        self.assertAlmostEqual(metrics["annualized_volatility"], expected_volatility)
        self.assertAlmostEqual(
            metrics["annualized_sharpe"],
            expected_mean / expected_volatility,
        )

    def test_rolling_sharpe_uses_only_trailing_window(self) -> None:
        returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02], dtype=float)
        baseline = rolling_annualized_sharpe(returns, window=3)
        changed = returns.copy()
        changed[-1] = 0.50
        perturbed = rolling_annualized_sharpe(changed, window=3)
        np.testing.assert_allclose(baseline[:-1], perturbed[:-1], equal_nan=True)
        self.assertFalse(np.isclose(baseline[-1], perturbed[-1]))

    def test_concatenate_blocks_preserves_order(self) -> None:
        combined = concatenate_blocks([np.array([1, 2]), np.array([3])])
        np.testing.assert_array_equal(combined, np.array([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
