from __future__ import annotations

import unittest

from ashare_stat_arb.execution import (
    FeeSchedule,
    PortfolioState,
    TradingStatus,
    begin_trading_day,
    execute_target_portfolio,
)


class AshareExecutionTest(unittest.TestCase):
    def test_t_plus_one_blocks_same_day_sale(self) -> None:
        initial = PortfolioState(cash=10_000.0)
        bought = execute_target_portfolio(
            initial,
            {"000001.SZ": 100},
            {"000001.SZ": 10.0},
        )
        self.assertEqual(bought.state.holdings["000001.SZ"], 100)
        self.assertEqual(bought.state.sellable_shares.get("000001.SZ", 0), 0)

        blocked = execute_target_portfolio(
            bought.state,
            {"000001.SZ": 0},
            {"000001.SZ": 10.0},
        )
        self.assertEqual(blocked.state.holdings["000001.SZ"], 100)
        self.assertEqual(blocked.records[0].reason, "t_plus_one")

        next_day = begin_trading_day(blocked.state)
        sold = execute_target_portfolio(
            next_day,
            {"000001.SZ": 0},
            {"000001.SZ": 10.0},
        )
        self.assertNotIn("000001.SZ", sold.state.holdings)
        self.assertEqual(sold.records[0].filled_shares, 100)

    def test_upper_limit_blocks_buy_and_lower_limit_blocks_sell(self) -> None:
        state = PortfolioState(
            cash=10_000.0,
            holdings={"600000.SH": 100},
            sellable_shares={"600000.SH": 100},
        )
        report = execute_target_portfolio(
            state,
            {"600000.SH": 0, "000001.SZ": 100},
            {"600000.SH": 10.0, "000001.SZ": 10.0},
            statuses={
                "600000.SH": TradingStatus(at_lower_limit=True),
                "000001.SZ": TradingStatus(at_upper_limit=True),
            },
        )
        reasons = {record.symbol: record.reason for record in report.records}
        self.assertEqual(reasons["600000.SH"], "lower_limit")
        self.assertEqual(reasons["000001.SZ"], "upper_limit")
        self.assertEqual(report.state.holdings, {"600000.SH": 100})

    def test_suspension_blocks_both_directions(self) -> None:
        state = PortfolioState(
            cash=10_000.0,
            holdings={"600000.SH": 100},
            sellable_shares={"600000.SH": 100},
        )
        report = execute_target_portfolio(
            state,
            {"600000.SH": 0, "000001.SZ": 100},
            {"600000.SH": 10.0, "000001.SZ": 10.0},
            statuses={
                "600000.SH": TradingStatus(suspended=True),
                "000001.SZ": TradingStatus(suspended=True),
            },
        )
        self.assertTrue(all(record.reason == "suspended" for record in report.records))

    def test_sells_fund_buys_and_buy_quantity_respects_lot_size(self) -> None:
        state = PortfolioState(
            cash=0.0,
            holdings={"600000.SH": 100},
            sellable_shares={"600000.SH": 100},
        )
        report = execute_target_portfolio(
            state,
            {"600000.SH": 0, "000001.SZ": 150},
            {"600000.SH": 10.0, "000001.SZ": 5.0},
        )
        self.assertEqual(report.state.holdings, {"000001.SZ": 100})
        self.assertEqual(report.state.cash, 500.0)
        self.assertEqual(report.records[-1].reason, "lot_rounding")

    def test_directional_fees_are_applied(self) -> None:
        state = PortfolioState(
            cash=0.0,
            holdings={"600000.SH": 100},
            sellable_shares={"600000.SH": 100},
        )
        report = execute_target_portfolio(
            state,
            {"600000.SH": 0},
            {"600000.SH": 10.0},
            fees=FeeSchedule(
                commission_rate=0.001,
                stamp_duty_rate=0.002,
                transfer_fee_rate=0.0001,
            ),
        )
        self.assertAlmostEqual(report.total_fees, 3.1)
        self.assertAlmostEqual(report.state.cash, 996.9)

    def test_negative_target_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            execute_target_portfolio(
                PortfolioState(cash=10_000.0),
                {"000001.SZ": -100},
                {"000001.SZ": 10.0},
            )


if __name__ == "__main__":
    unittest.main()
