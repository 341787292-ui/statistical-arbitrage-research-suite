from __future__ import annotations

import unittest

import numpy as np
import torch

from paper_reproduction.dlsa.models import CNNTransformerAllocation, FeedForwardAllocation
from paper_reproduction.dlsa.objectives import build_portfolio_tensors
from paper_reproduction.dlsa.preprocessing import build_active_feature_batch
from paper_reproduction.dlsa.synthetic import make_factor_market
from paper_reproduction.dlsa.training import (
    evaluate_model_streaming,
    fit_training_window_streaming,
    rolling_train_test,
)


class DeepModelSmokeTest(unittest.TestCase):
    def test_active_batch_uses_only_lagged_nonzero_history(self) -> None:
        returns = make_factor_market(time_count=80, asset_count=5, factor_count=2)
        returns[20, 0] = 0.0
        batch = build_active_feature_batch(
            returns,
            start=30,
            end=31,
            lookback=30,
            zero_is_missing=True,
        )
        self.assertFalse(np.any(batch.valid_mask[:, 0]))
        changed = returns.copy()
        changed[30] += 100.0
        changed_batch = build_active_feature_batch(
            changed,
            start=30,
            end=31,
            lookback=30,
            zero_is_missing=True,
        )
        np.testing.assert_allclose(batch.features, changed_batch.features)

    def test_low_memory_training_and_evaluation_run(self) -> None:
        returns = make_factor_market(time_count=140, asset_count=8, factor_count=2)
        model = FeedForwardAllocation(input_size=30, hidden_units=(8, 4), seed=9)
        fit = fit_training_window_streaming(
            model,
            returns[:100],
            lookback=30,
            epochs=1,
            temporal_batch_size=20,
            model_chunk_size=16,
            feature_type="fourier",
            zero_is_missing=False,
        )
        evaluated = evaluate_model_streaming(
            model,
            returns[70:120],
            lookback=30,
            temporal_batch_size=10,
            model_chunk_size=16,
            feature_type="fourier",
            zero_is_missing=False,
        )
        self.assertEqual(fit.daily_returns.shape, (70,))
        self.assertEqual(evaluated.daily_returns.shape, (20,))
        self.assertTrue(np.all(np.isfinite(evaluated.daily_returns)))

    def test_gradient_checkpointing_preserves_training_result(self) -> None:
        returns = make_factor_market(time_count=75, asset_count=6, factor_count=2, seed=31)
        plain_model = FeedForwardAllocation(
            input_size=30,
            hidden_units=(8, 4),
            dropout=0.25,
            seed=31,
        )
        plain = fit_training_window_streaming(
            plain_model,
            returns,
            lookback=30,
            epochs=1,
            temporal_batch_size=15,
            model_chunk_size=12,
            gradient_checkpointing=False,
            zero_is_missing=False,
        )
        checkpointed_model = FeedForwardAllocation(
            input_size=30,
            hidden_units=(8, 4),
            dropout=0.25,
            seed=31,
        )
        checkpointed = fit_training_window_streaming(
            checkpointed_model,
            returns,
            lookback=30,
            epochs=1,
            temporal_batch_size=15,
            model_chunk_size=12,
            gradient_checkpointing=True,
            zero_is_missing=False,
        )
        np.testing.assert_allclose(
            plain.daily_returns,
            checkpointed.daily_returns,
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(plain.losses, checkpointed.losses, rtol=1e-6)

    def test_model_chunk_size_preserves_training_result(self) -> None:
        returns = make_factor_market(time_count=75, asset_count=6, factor_count=2, seed=37)
        small_chunk_model = FeedForwardAllocation(
            input_size=30,
            hidden_units=(8, 4),
            dropout=0.0,
            seed=37,
        )
        small_chunk = fit_training_window_streaming(
            small_chunk_model,
            returns,
            lookback=30,
            epochs=1,
            temporal_batch_size=15,
            model_chunk_size=7,
            zero_is_missing=False,
        )
        full_batch_model = FeedForwardAllocation(
            input_size=30,
            hidden_units=(8, 4),
            dropout=0.0,
            seed=37,
        )
        full_batch = fit_training_window_streaming(
            full_batch_model,
            returns,
            lookback=30,
            epochs=1,
            temporal_batch_size=15,
            model_chunk_size=10_000,
            zero_is_missing=False,
        )
        np.testing.assert_allclose(
            small_chunk.daily_returns,
            full_batch.daily_returns,
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(small_chunk.losses, full_batch.losses, rtol=1e-6)

    def test_model_outputs_one_score_per_residual_window(self) -> None:
        model = CNNTransformerAllocation(dropout=0.0)
        windows = torch.randn(17, 30)
        scores = model(windows)
        self.assertEqual(tuple(scores.shape), (17,))
        self.assertTrue(torch.isfinite(scores).all())

    def test_stock_space_weights_have_unit_l1_norm(self) -> None:
        scores = torch.tensor([[1.0, -2.0], [0.5, 0.25]])
        residual_returns = torch.tensor([[0.01, -0.02], [0.02, 0.01]])
        composition = torch.tensor(
            [
                [[1.0, -0.5], [0.0, 1.0]],
                [[1.0, -0.5], [0.0, 1.0]],
            ]
        )
        portfolio = build_portfolio_tensors(
            scores,
            residual_returns,
            composition_matrices=composition,
        )
        gross = portfolio.stock_weights.abs().sum(dim=1)
        torch.testing.assert_close(gross, torch.ones_like(gross))

    def test_model_backward_pass_is_finite(self) -> None:
        model = CNNTransformerAllocation(dropout=0.0)
        output = model(torch.randn(24, 30)).mean()
        output.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_rolling_driver_returns_only_oos_days(self) -> None:
        rng = np.random.default_rng(12)
        residuals = rng.normal(0, 0.01, size=(170, 5)).astype(np.float32)
        result = rolling_train_test(
            lambda: FeedForwardAllocation(dropout=0.0, seed=12),
            residuals,
            lookback=30,
            training_length=100,
            retrain_frequency=35,
            temporal_batch_size=35,
            epochs=1,
            feature_type="fourier",
        )
        self.assertEqual(len(result.daily_returns), 70)
        self.assertEqual(result.retrain_origins, [100, 135])
        self.assertTrue(np.all(np.isfinite(result.daily_returns)))


if __name__ == "__main__":
    unittest.main()
