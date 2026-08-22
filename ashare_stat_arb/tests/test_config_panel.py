from __future__ import annotations

import unittest

import numpy as np

from ashare_stat_arb.config import DEFAULT_CONFIG
from ashare_stat_arb.panel import DailyPanel, audit_panel


def build_panel() -> DailyPanel:
    dates = np.array(["2022-01-04", "2022-01-05", "2022-01-06"], dtype="datetime64[D]")
    symbols = ("000001.XSHE", "600000.XSHG")
    adjusted_close = np.array([[10.0, 20.0], [11.0, 20.0], [11.0, 22.0]])
    open_price = np.array([[9.9, 19.8], [10.5, 20.0], [11.2, 21.0]])
    close_price = adjusted_close.copy()
    high_limit = close_price * 1.1
    low_limit = close_price * 0.9
    volume = np.full((3, 2), 1_000_000.0)
    money = volume * close_price
    paused = np.zeros((3, 2), dtype=bool)
    is_st = np.zeros((3, 2), dtype=bool)
    member = np.ones((3, 2), dtype=bool)
    benchmark_weight = np.full((3, 2), 0.5)
    return DailyPanel(
        dates=dates,
        symbols=symbols,
        adjusted_close=adjusted_close,
        open_price=open_price,
        close_price=close_price,
        high_limit=high_limit,
        low_limit=low_limit,
        volume=volume,
        money=money,
        paused=paused,
        is_st=is_st,
        member=member,
        benchmark_weight=benchmark_weight,
    )


class ResearchConfigTest(unittest.TestCase):
    def test_frozen_baseline_matches_accepted_scope(self) -> None:
        self.assertEqual(DEFAULT_CONFIG.portfolio.benchmark, "000905.XSHG")
        self.assertEqual(DEFAULT_CONFIG.periods.holdout_start, "2023-01-01")
        self.assertEqual(DEFAULT_CONFIG.signal.factor_counts, (0, 1, 3, 5, 8, 10, 15))
        self.assertEqual(DEFAULT_CONFIG.portfolio.maximum_stock_weight, 0.015)
        self.assertEqual(DEFAULT_CONFIG.admission.minimum_net_information_ratio, 1.0)


class DailyPanelTest(unittest.TestCase):
    def test_returns_have_declared_timing(self) -> None:
        panel = build_panel()
        adjusted = panel.adjusted_returns()
        next_open = panel.next_open_returns()
        self.assertTrue(np.isnan(adjusted[0]).all())
        self.assertAlmostEqual(adjusted[1, 0], 0.1)
        self.assertAlmostEqual(adjusted[2, 1], 0.1)
        self.assertAlmostEqual(next_open[0, 0], 0.05)
        self.assertAlmostEqual(next_open[1, 1], 0.05)
        self.assertTrue(np.isnan(next_open[-1]).all())

    def test_open_returns_remove_corporate_action_price_jump(self) -> None:
        panel = build_panel()
        raw_open = panel.open_price.copy()
        raw_close = panel.close_price.copy()
        adjusted_close = panel.adjusted_close.copy()
        raw_open[2, 0] /= 2.0
        raw_close[2, 0] /= 2.0
        adjusted_close[2, 0] = adjusted_close[1, 0]
        split_panel = DailyPanel(
            dates=panel.dates,
            symbols=panel.symbols,
            adjusted_close=adjusted_close,
            open_price=raw_open,
            close_price=raw_close,
            high_limit=panel.high_limit,
            low_limit=panel.low_limit,
            volume=panel.volume,
            money=panel.money,
            paused=panel.paused,
            is_st=panel.is_st,
            member=panel.member,
            benchmark_weight=panel.benchmark_weight,
        )
        returns = split_panel.open_to_open_returns()
        expected = panel.open_price[2, 0] / panel.open_price[1, 0] - 1.0
        self.assertAlmostEqual(returns[1, 0], expected)

    def test_audit_is_deterministic_and_checks_weights(self) -> None:
        panel = build_panel()
        first = audit_panel(panel)
        second = audit_panel(panel)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.invalid_weight_days, 0)
        self.assertEqual(first.member_observations, 6)
        self.assertEqual(first.missing_open_rate, 0.0)

    def test_duplicate_dates_are_rejected(self) -> None:
        panel = build_panel()
        with self.assertRaises(ValueError):
            DailyPanel(
                dates=np.array(["2022-01-04", "2022-01-04", "2022-01-06"]),
                symbols=panel.symbols,
                adjusted_close=panel.adjusted_close,
                open_price=panel.open_price,
                close_price=panel.close_price,
                high_limit=panel.high_limit,
                low_limit=panel.low_limit,
                volume=panel.volume,
                money=panel.money,
                paused=panel.paused,
                is_st=panel.is_st,
                member=panel.member,
                benchmark_weight=panel.benchmark_weight,
            )


if __name__ == "__main__":
    unittest.main()
