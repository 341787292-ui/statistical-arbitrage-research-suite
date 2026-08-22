"""A-share adaptation of the statistical-arbitrage research pipeline."""

from ashare_stat_arb.config import AshareResearchConfig, DEFAULT_CONFIG
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
    "AshareResearchConfig",
    "DEFAULT_CONFIG",
    "ExecutionRecord",
    "ExecutionReport",
    "FeeSchedule",
    "PortfolioState",
    "TradingStatus",
    "begin_trading_day",
    "execute_target_portfolio",
]
