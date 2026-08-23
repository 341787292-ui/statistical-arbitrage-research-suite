from __future__ import annotations

import unittest

import numpy as np

from ashare_stat_arb.residual_comparison import compare_residual_definitions
from ashare_stat_arb.signals import rolling_monthly_pca_ou_stock_alpha


class ResidualDefinitionComparisonTests(unittest.TestCase):
    def test_definitions_match_when_one_pca_map_spans_the_window(self) -> None:
        rng = np.random.default_rng(19)
        rows = 24
        columns = 8
        returns = rng.normal(0.0, 0.01, size=(rows, columns))
        dates = np.arange(
            np.datetime64("2022-01-03"),
            np.datetime64("2022-01-03") + np.timedelta64(rows, "D"),
            dtype="datetime64[D]",
        )
        member = np.ones_like(returns, dtype=bool)
        signal = rolling_monthly_pca_ou_stock_alpha(
            returns,
            dates,
            member,
            n_factors=2,
            covariance_window=10,
            loading_window=5,
            residual_lookback=5,
            entry_threshold=0.0,
            min_r_squared=0.0,
        )

        result = compare_residual_definitions(
            returns,
            dates,
            member,
            signal,
            start=14,
            n_factors=2,
            covariance_window=10,
            loading_window=5,
            residual_lookback=5,
            entry_threshold=0.0,
            min_r_squared=0.0,
        )

        self.assertAlmostEqual(
            result.stitched_asof.annualized_mean,
            result.current_composition.annualized_mean,
            places=12,
        )
        self.assertAlmostEqual(result.annualized_mean_delta, 0.0, places=12)

    def test_start_must_leave_complete_windows(self) -> None:
        returns = np.zeros((20, 5), dtype=np.float64)
        dates = np.arange(
            np.datetime64("2022-01-03"),
            np.datetime64("2022-01-03") + np.timedelta64(20, "D"),
            dtype="datetime64[D]",
        )
        member = np.ones_like(returns, dtype=bool)
        signal = rolling_monthly_pca_ou_stock_alpha(
            returns,
            dates,
            member,
            n_factors=0,
            covariance_window=10,
            loading_window=5,
            residual_lookback=5,
        )

        with self.assertRaisesRegex(ValueError, "start"):
            compare_residual_definitions(
                returns,
                dates,
                member,
                signal,
                start=10,
                n_factors=0,
                covariance_window=10,
                loading_window=5,
                residual_lookback=5,
                entry_threshold=1.25,
                min_r_squared=0.25,
            )


if __name__ == "__main__":
    unittest.main()
