"""A-share adaptation of the statistical-arbitrage research pipeline."""

from ashare_stat_arb.execution import (
    ExecutionRecord,
    ExecutionReport,
    FeeSchedule,
    PortfolioState,
    TradingStatus,
    begin_trading_day,
    execute_target_portfolio,
)

__all__ = [
    "ExecutionRecord",
    "ExecutionReport",
    "FeeSchedule",
    "PortfolioState",
    "TradingStatus",
    "begin_trading_day",
    "execute_target_portfolio",
]
