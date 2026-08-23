from __future__ import annotations

import unittest

import numpy as np

from ashare_stat_arb.residual_audit import audit_residual_continuity
from ashare_stat_arb.signals import (
    MonthlyPCASignalResult,
    rolling_monthly_pca_ou_stock_alpha,
)
from ashare_stat_arb.synthetic import make_synthetic_csi500_panel
from paper_reproduction.dlsa.factor_models import rolling_pca_residuals


class ResidualContinuityAuditTests(unittest.TestCase):
    def test_audit_detects_windows_crossing_refits(self) -> None:
        rows = 12
        columns = 3
        dates = np.arange(
            np.datetime64("2022-01-01"),
            np.datetime64("2022-01-01") + np.timedelta64(rows, "D"),
            dtype="datetime64[D]",
        )
        residuals = np.full((rows, columns), 0.01, dtype=np.float64)
        signal = MonthlyPCASignalResult(
            stock_alpha=np.zeros_like(residuals),
            residual_returns=residuals,
            active_count=np.full(rows, columns),
            refit_dates=(str(dates[0]), str(dates[5]), str(dates[10])),
        )
        audit = audit_residual_continuity(
            signal,
            dates,
            np.ones_like(residuals, dtype=bool),
            lookback=6,
        )

        self.assertEqual(audit.single_model_ou_window_rate, 0.0)
        self.assertEqual(audit.cross_model_ou_window_rate, 1.0)
        self.assertGreaterEqual(audit.maximum_models_per_ou_window, 2)
        self.assertEqual(audit.model_day_rate, 1.0)

    def test_monthly_refit_matches_daily_pca_on_refit_date(self) -> None:
        panel = make_synthetic_csi500_panel(trading_days=125, assets=70, seed=31)
        returns = panel.adjusted_returns()
        monthly = rolling_monthly_pca_ou_stock_alpha(
            returns,
            panel.dates,
            panel.member,
            n_factors=3,
            covariance_window=50,
            loading_window=25,
            residual_lookback=12,
            min_r_squared=0.0,
        )
        daily = rolling_pca_residuals(
            returns,
            n_factors=3,
            covariance_window=50,
            loading_window=25,
        )
        lookup = {str(date): index for index, date in enumerate(panel.dates)}
        compared = 0
        for date in monthly.refit_dates:
            row = lookup[date]
            valid = np.isfinite(monthly.residual_returns[row]) & np.isfinite(
                daily.residual_returns[row]
            )
            if np.any(valid):
                np.testing.assert_allclose(
                    monthly.residual_returns[row, valid],
                    daily.residual_returns[row, valid],
                    atol=1e-10,
                )
                compared += 1
        self.assertGreater(compared, 1)


if __name__ == "__main__":
    unittest.main()
