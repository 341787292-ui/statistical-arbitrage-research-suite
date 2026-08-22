from __future__ import annotations

import unittest

import numpy as np

from ashare_stat_arb.optimizer import optimize_long_only_index_enhancement
from ashare_stat_arb.research_backtest import _execute_weight_targets, run_long_only_backtest
from ashare_stat_arb.signals import ou_stock_alpha, rolling_monthly_pca_ou_stock_alpha
from ashare_stat_arb.synthetic import make_synthetic_csi500_panel
from paper_reproduction.dlsa.factor_models import rolling_pca_residuals


class OptimizerTests(unittest.TestCase):
    def test_long_only_constraints_hold(self) -> None:
        assets = 80
        benchmark = np.full(assets, 1.0 / assets)
        previous = benchmark * 0.99
        alpha = np.linspace(-0.002, 0.002, assets)
        covariance = np.eye(assets) * 0.0001

        result = optimize_long_only_index_enhancement(
            alpha,
            benchmark,
            previous,
            covariance,
        )

        self.assertAlmostEqual(float(result.weights.sum()), 0.99, places=6)
        self.assertGreaterEqual(float(result.weights.min()), -1e-7)
        self.assertLessEqual(float(result.weights.max()), 0.015 + 1e-7)
        self.assertLessEqual(result.two_way_turnover, 0.20 + 1e-6)
        self.assertLessEqual(result.annualized_tracking_error, 0.08 + 1e-6)
        self.assertGreater(float(np.abs(result.weights - previous).sum()), 1e-5)

    def test_excluded_stock_receives_zero_target(self) -> None:
        assets = 80
        benchmark = np.full(assets, 1.0 / assets)
        previous = benchmark * 0.99
        alpha = np.zeros(assets)
        eligible = np.ones(assets, dtype=bool)
        eligible[0] = False
        result = optimize_long_only_index_enhancement(
            alpha,
            benchmark,
            previous,
            np.eye(assets) * 0.0001,
            eligible=eligible,
        )
        self.assertAlmostEqual(float(result.weights[0]), 0.0, places=7)

    def test_execution_constraints_are_applied_after_target_creation(self) -> None:
        previous = np.array([0.4, 0.4, 0.19])
        target = np.array([0.3, 0.5, 0.19])
        executed = _execute_weight_targets(
            previous,
            target,
            can_buy=np.array([True, False, True]),
            can_sell=np.array([False, True, True]),
        )
        np.testing.assert_allclose(executed, previous)


class SignalTests(unittest.TestCase):
    def test_ou_alpha_has_no_future_dependency(self) -> None:
        rng = np.random.default_rng(11)
        residuals = rng.normal(0.0, 0.01, size=(40, 4))
        compositions = np.repeat(np.eye(4)[None, :, :], 40, axis=0)
        original = ou_stock_alpha(
            residuals,
            compositions,
            lookback=12,
            min_r_squared=-1.0,
        )

        changed = residuals.copy()
        changed[31:] += 10.0
        revised = ou_stock_alpha(
            changed,
            compositions,
            lookback=12,
            min_r_squared=-1.0,
        )

        np.testing.assert_allclose(original[:31], revised[:31])

    def test_monthly_pca_signal_refits_without_future_data(self) -> None:
        panel = make_synthetic_csi500_panel(trading_days=120, assets=70, seed=29)
        returns = panel.adjusted_returns()
        original = rolling_monthly_pca_ou_stock_alpha(
            returns,
            panel.dates,
            panel.member,
            n_factors=3,
            covariance_window=50,
            loading_window=25,
            residual_lookback=12,
            min_r_squared=0.0,
        )
        changed = returns.copy()
        changed[105:] += 0.5
        revised = rolling_monthly_pca_ou_stock_alpha(
            changed,
            panel.dates,
            panel.member,
            n_factors=3,
            covariance_window=50,
            loading_window=25,
            residual_lookback=12,
            min_r_squared=0.0,
        )
        np.testing.assert_allclose(original.stock_alpha[:105], revised.stock_alpha[:105])
        self.assertGreater(len(original.refit_dates), 1)


class EndToEndBaselineTests(unittest.TestCase):
    def test_synthetic_pca_ou_long_only_pipeline(self) -> None:
        panel = make_synthetic_csi500_panel(trading_days=145, assets=70, seed=19)
        pca = rolling_pca_residuals(
            panel.adjusted_returns(),
            n_factors=3,
            covariance_window=60,
            loading_window=30,
            store_composition=True,
        )
        self.assertIsNotNone(pca.composition_matrices)
        alpha = ou_stock_alpha(
            pca.residual_returns,
            pca.composition_matrices,
            member=panel.member,
            lookback=15,
            min_r_squared=0.0,
        )
        result = run_long_only_backtest(
            panel,
            alpha,
            covariance_window=40,
            start=90,
        )

        finite = np.isfinite(result.strategy_returns)
        self.assertGreater(int(finite.sum()), 40)
        self.assertTrue(np.isfinite(result.annualized_return))
        self.assertTrue(np.isfinite(result.information_ratio))
        self.assertLessEqual(float(result.target_weights.max()), 0.015 + 1e-6)
        self.assertLessEqual(float(result.two_way_turnover.max()), 0.20 + 1e-5)
        self.assertGreater(float(result.two_way_turnover.sum()), 1e-5)


if __name__ == "__main__":
    unittest.main()
