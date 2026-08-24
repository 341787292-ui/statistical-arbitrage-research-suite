from __future__ import annotations

import unittest

import numpy as np

from ashare_stat_arb.residual_predictability import audit_residual_predictability


class ResidualPredictabilityAuditTests(unittest.TestCase):
    def test_detects_stable_synthetic_reversal(self) -> None:
        rng = np.random.default_rng(23)
        rows = 520
        columns = 60
        levels = np.zeros((rows + 1, columns), dtype=np.float64)
        for row in range(rows):
            levels[row + 1] = 0.70 * levels[row] + rng.normal(0.0, 0.01, columns)
        residuals = np.diff(levels, axis=0)
        dates = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-01") + np.timedelta64(rows, "D"),
            dtype="datetime64[D]",
        )
        member = np.ones_like(residuals, dtype=bool)

        result = audit_residual_predictability(
            residuals,
            dates,
            member,
            horizons=(1, 5),
            development_start="2020-01-01",
            development_end="2020-09-30",
            validation_start="2020-10-01",
            validation_end="2021-06-30",
            minimum_rank_ic=0.01,
            minimum_period_days=60,
            required_stable_horizons=1,
        )

        self.assertTrue(result.broad_reversal_evidence_passed)
        self.assertTrue(result.cross_sectional_reversal_evidence_passed)
        self.assertGreaterEqual(len(result.stable_horizons), 1)

    def test_future_change_does_not_change_earlier_rank_ic(self) -> None:
        rng = np.random.default_rng(29)
        rows = 180
        columns = 40
        residuals = rng.normal(0.0, 0.01, size=(rows, columns))
        dates = np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-01-01") + np.timedelta64(rows, "D"),
            dtype="datetime64[D]",
        )
        member = np.ones_like(residuals, dtype=bool)
        changed = residuals.copy()
        changed[120:] += 10.0

        original = audit_residual_predictability(
            residuals,
            dates,
            member,
            horizons=(5,),
            development_start="2020-01-01",
            development_end="2020-04-15",
            validation_start="2020-04-16",
            validation_end="2020-06-28",
            minimum_period_days=20,
            required_stable_horizons=1,
        )
        modified = audit_residual_predictability(
            changed,
            dates,
            member,
            horizons=(5,),
            development_start="2020-01-01",
            development_end="2020-04-15",
            validation_start="2020-04-16",
            validation_end="2020-06-28",
            minimum_period_days=20,
            required_stable_horizons=1,
        )

        self.assertAlmostEqual(
            original.reversal_horizons[0].development.mean_rank_ic,
            modified.reversal_horizons[0].development.mean_rank_ic,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
