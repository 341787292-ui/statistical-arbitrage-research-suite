from __future__ import annotations

import unittest

import numpy as np

from paper_reproduction.dlsa.backtest import (
    backtest_ou_threshold,
    backtest_ou_threshold_streaming,
)
from paper_reproduction.dlsa.factor_models import rolling_pca_residuals
from paper_reproduction.dlsa.ou import fit_ou, ou_threshold_weight
from paper_reproduction.dlsa.preprocessing import (
    cumulative_residual_windows,
    fourier_features,
)
from paper_reproduction.dlsa.synthetic import make_factor_market


class PaperAlignedStageOneTest(unittest.TestCase):
    def test_streaming_ou_matches_materialized_implementation(self) -> None:
        returns = make_factor_market(
            time_count=180,
            asset_count=8,
            factor_count=2,
            seed=19,
        )
        materialized = backtest_ou_threshold(returns, lookback=30)
        streaming = backtest_ou_threshold_streaming(
            returns,
            lookback=30,
            zero_is_missing=False,
        )
        np.testing.assert_allclose(streaming.returns, materialized.returns, atol=1e-12)
        np.testing.assert_allclose(streaming.turnover, materialized.turnover, atol=1e-12)

    def test_streaming_ou_excludes_zero_encoded_missing_windows(self) -> None:
        returns = make_factor_market(
            time_count=100,
            asset_count=4,
            factor_count=1,
            seed=23,
        )
        returns[20, 0] = 0.0
        result = backtest_ou_threshold_streaming(
            returns,
            lookback=30,
            zero_is_missing=True,
        )
        self.assertTrue(np.all(result.active_positions[:21] <= 3))

    def test_pca_residuals_are_strictly_backward_looking(self) -> None:
        returns = make_factor_market(time_count=340, asset_count=10, factor_count=2)
        baseline = rolling_pca_residuals(
            returns,
            n_factors=2,
            covariance_window=100,
            loading_window=40,
        ).residual_returns
        changed = returns.copy()
        changed[250:] += 5.0
        perturbed = rolling_pca_residuals(
            changed,
            n_factors=2,
            covariance_window=100,
            loading_window=40,
        ).residual_returns
        np.testing.assert_allclose(baseline[:250], perturbed[:250], equal_nan=True)

    def test_composition_matrix_reconstructs_residuals(self) -> None:
        returns = make_factor_market(time_count=180, asset_count=8, factor_count=2)
        result = rolling_pca_residuals(
            returns,
            n_factors=2,
            covariance_window=80,
            loading_window=30,
            store_composition=True,
        )
        t = 120
        reconstructed = result.composition_matrices[t] @ returns[t]
        np.testing.assert_allclose(
            reconstructed[result.active_mask[t]],
            result.residual_returns[t, result.active_mask[t]],
            atol=1e-6,
        )

    def test_windows_do_not_include_the_traded_day(self) -> None:
        residuals = np.arange(1, 13, dtype=float).reshape(6, 2)
        windows, valid = cumulative_residual_windows(residuals, lookback=3)
        np.testing.assert_allclose(windows[0, 0], np.cumsum([1.0, 3.0, 5.0]))
        self.assertTrue(valid[0, 0])

    def test_fourier_representation_preserves_dimension(self) -> None:
        windows = np.arange(60, dtype=float).reshape(2, 1, 30)
        features = fourier_features(windows)
        self.assertEqual(features.shape, windows.shape)

    def test_ou_threshold_is_contrarian(self) -> None:
        rng = np.random.default_rng(3)
        x = np.zeros(80)
        for t in range(1, len(x)):
            x[t] = 0.85 * x[t - 1] + rng.normal(0, 0.2)
        x[-1] = 3.0
        fit = fit_ou(x)
        self.assertTrue(fit.valid)
        self.assertEqual(
            ou_threshold_weight(fit, entry_threshold=1.25, min_r_squared=0.0),
            -1.0,
        )

    def test_ou_backtest_uses_stock_space_l1_normalization(self) -> None:
        returns = make_factor_market(time_count=420, asset_count=10, factor_count=2)
        pca = rolling_pca_residuals(
            returns,
            n_factors=2,
            covariance_window=100,
            loading_window=40,
            store_composition=True,
        )
        result = backtest_ou_threshold(
            pca.residual_returns,
            composition_matrices=pca.composition_matrices,
            lookback=30,
            min_r_squared=0.0,
        )
        gross = np.abs(result.stock_weights).sum(axis=1)
        self.assertTrue(np.all((np.isclose(gross, 0.0)) | (np.isclose(gross, 1.0))))
        self.assertTrue(result.used_stock_space_normalization)


if __name__ == "__main__":
    unittest.main()
