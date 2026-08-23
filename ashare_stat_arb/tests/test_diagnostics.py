from __future__ import annotations

import unittest

import numpy as np

from ashare_stat_arb.diagnostics import forward_open_returns, summarize_rank_ic
from ashare_stat_arb.panel import DailyPanel


def _panel() -> DailyPanel:
    days = 6
    assets = 4
    dates = np.arange(
        np.datetime64("2022-01-04"),
        np.datetime64("2022-01-04") + np.timedelta64(days, "D"),
        dtype="datetime64[D]",
    )
    open_price = np.asarray(
        [[10 + day + asset for asset in range(assets)] for day in range(days)],
        dtype=np.float64,
    )
    close_price = open_price + 0.5
    adjusted_close = close_price.copy()
    shape = (days, assets)
    return DailyPanel(
        dates=dates,
        symbols=tuple(f"stock-{index}" for index in range(assets)),
        adjusted_close=adjusted_close,
        open_price=open_price,
        close_price=close_price,
        high_limit=close_price * 1.1,
        low_limit=close_price * 0.9,
        volume=np.full(shape, 1_000_000.0),
        money=np.full(shape, 10_000_000.0),
        paused=np.zeros(shape, dtype=bool),
        is_st=np.zeros(shape, dtype=bool),
        member=np.ones(shape, dtype=bool),
        benchmark_weight=np.full(shape, 1.0 / assets),
    )


class SignalDiagnosticTests(unittest.TestCase):
    def test_forward_return_starts_at_next_open(self) -> None:
        panel = _panel()
        result = forward_open_returns(panel, 1)
        adjusted_open = panel.adjusted_open_prices()
        expected = adjusted_open[2] / adjusted_open[1] - 1.0
        np.testing.assert_allclose(result[0], expected)
        self.assertTrue(np.isnan(result[-2:]).all())

    def test_rank_ic_detects_original_and_reversed_direction(self) -> None:
        alpha = np.tile(np.arange(4, dtype=np.float64), (5, 1))
        future = alpha.copy()
        member = np.ones_like(alpha, dtype=bool)
        original = summarize_rank_ic(
            alpha,
            future,
            member,
            horizon=1,
            min_assets=3,
        )
        reversed_result = summarize_rank_ic(
            -alpha,
            future,
            member,
            horizon=1,
            min_assets=3,
        )
        self.assertAlmostEqual(original.mean_rank_ic, 1.0)
        self.assertAlmostEqual(reversed_result.mean_rank_ic, -1.0)
        self.assertEqual(original.observations, 5)


if __name__ == "__main__":
    unittest.main()
