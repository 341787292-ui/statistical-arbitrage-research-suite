from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ashare_stat_arb.config import CostConfig, PortfolioConfig
from ashare_stat_arb.optimizer import _psd_covariance, optimize_long_only_index_enhancement
from ashare_stat_arb.panel import DailyPanel


@dataclass(frozen=True)
class LongOnlyBacktestResult:
    strategy_returns: np.ndarray
    benchmark_returns: np.ndarray
    active_returns: np.ndarray
    target_weights: np.ndarray
    executed_weights: np.ndarray
    two_way_turnover: np.ndarray
    costs: np.ndarray
    annualized_return: float
    annualized_sharpe: float
    annualized_excess_return: float
    information_ratio: float
    maximum_active_drawdown: float


def _covariance(returns: np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64)
    column_means = np.nanmean(values, axis=0)
    column_means = np.nan_to_num(column_means, nan=0.0)
    filled = np.where(np.isfinite(values), values, column_means[None, :])
    sample = np.cov(filled, rowvar=False, ddof=0)
    if sample.ndim == 0:
        sample = np.array([[float(sample)]])
    diagonal = np.diag(np.diag(sample))
    return 0.5 * sample + 0.5 * diagonal + np.eye(sample.shape[0]) * 1e-8


def _stamp_duty(date: np.datetime64, costs: CostConfig) -> float:
    if date < np.datetime64("2023-08-28"):
        return costs.stamp_duty_before_2023_08_28
    return costs.stamp_duty_from_2023_08_28


def _annualized_sharpe(returns: np.ndarray) -> float:
    finite = np.asarray(returns, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    volatility = float(finite.std(ddof=0))
    return float(finite.mean() / volatility * math.sqrt(252.0)) if volatility > 0 else 0.0


def _maximum_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + np.nan_to_num(returns, nan=0.0))
    peaks = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peaks - 1.0)) if wealth.size else 0.0


def _execute_weight_targets(
    previous: np.ndarray,
    target: np.ndarray,
    *,
    can_buy: np.ndarray,
    can_sell: np.ndarray,
) -> np.ndarray:
    """Apply next-open directional fill constraints without using them in planning."""

    change = target - previous
    sells = np.where((change < 0) & can_sell, change, 0.0)
    buys = np.where((change > 0) & can_buy, change, 0.0)
    cash_before = max(1.0 - float(previous.sum()), 0.0)
    available = cash_before - float(sells.sum())
    desired_buys = float(buys.sum())
    if desired_buys > available and desired_buys > 0:
        buys *= available / desired_buys
    return previous + sells + buys


def run_long_only_backtest(
    panel: DailyPanel,
    stock_alpha: np.ndarray,
    *,
    portfolio: PortfolioConfig | None = None,
    costs: CostConfig | None = None,
    covariance_window: int = 60,
    alpha_scale: float = 0.002,
    start: int | None = None,
) -> LongOnlyBacktestResult:
    """Run a daily close-decision, next-open long-only research approximation."""

    portfolio = portfolio or PortfolioConfig()
    costs = costs or CostConfig()
    alpha = np.asarray(stock_alpha, dtype=np.float64)
    if alpha.shape != panel.shape:
        raise ValueError("stock_alpha must align with the daily panel.")
    if covariance_window < 20:
        raise ValueError("covariance_window must be at least 20 days.")

    time_count, asset_count = panel.shape
    first_decision = covariance_window if start is None else max(start, covariance_window)
    adjusted_returns = panel.adjusted_returns()
    open_returns = np.full(panel.shape, np.nan, dtype=np.float64)
    valid_open = (
        np.isfinite(panel.open_price[:-1])
        & np.isfinite(panel.open_price[1:])
        & (panel.open_price[:-1] > 0)
    )
    open_returns[:-1][valid_open] = (
        panel.open_price[1:][valid_open] / panel.open_price[:-1][valid_open] - 1.0
    )

    target_weights = np.zeros(panel.shape, dtype=np.float64)
    executed_weights = np.zeros(panel.shape, dtype=np.float64)
    strategy_returns = np.full(time_count, np.nan, dtype=np.float64)
    benchmark_returns = np.full(time_count, np.nan, dtype=np.float64)
    turnover = np.zeros(time_count, dtype=np.float64)
    realized_costs = np.zeros(time_count, dtype=np.float64)

    initial_benchmark = np.where(
        panel.member[first_decision], panel.benchmark_weight[first_decision], 0.0
    )
    initial_benchmark = (
        initial_benchmark / initial_benchmark.sum() * portfolio.target_equity_exposure
    )
    previous = initial_benchmark
    risk_month: str | None = None
    working_indices = np.empty(0, dtype=np.int64)
    covariance_psd = np.empty((0, 0), dtype=np.float64)

    for decision in range(first_decision, time_count - 2):
        benchmark = np.where(
            panel.member[decision], panel.benchmark_weight[decision], 0.0
        )
        if benchmark.sum() <= 0:
            continue
        benchmark = benchmark / benchmark.sum()
        eligible = panel.member[decision] & ~panel.is_st[decision]
        month = str(panel.dates[decision].astype("datetime64[M]"))
        if month != risk_month:
            working_indices = np.flatnonzero(eligible | (previous > 1e-12))
            covariance_psd = _psd_covariance(
                _covariance(
                    adjusted_returns[
                        decision - covariance_window + 1 : decision + 1,
                        working_indices,
                    ]
                )
            )
            risk_month = month
        if working_indices.size == 0:
            continue
        result = optimize_long_only_index_enhancement(
            alpha[decision, working_indices] * alpha_scale,
            benchmark[working_indices],
            previous[working_indices],
            covariance_psd,
            equity_exposure=portfolio.target_equity_exposure,
            maximum_stock_weight=portfolio.maximum_stock_weight,
            maximum_two_way_turnover=portfolio.maximum_two_way_turnover,
            annual_tracking_error_limit=portfolio.central_tracking_error,
            maximum_industry_deviation=portfolio.maximum_industry_deviation,
            maximum_style_deviation=portfolio.maximum_style_deviation,
            eligible=eligible[working_indices],
            covariance_is_psd=True,
        )
        target = np.zeros(asset_count, dtype=np.float64)
        target[working_indices] = result.weights
        target_weights[decision] = target

        execution_day = decision + 1
        holding_return_day = decision + 1
        open_price = panel.open_price[execution_day]
        valid_market = (
            ~panel.paused[execution_day]
            & np.isfinite(open_price)
            & (open_price > 0)
            & (panel.volume[execution_day] > 0)
        )
        tolerance = 1e-10
        can_buy = (
            valid_market
            & ~panel.is_st[execution_day]
            & (open_price < panel.high_limit[execution_day] - tolerance)
        )
        can_sell = valid_market & (
            open_price > panel.low_limit[execution_day] + tolerance
        )
        executed = _execute_weight_targets(
            previous,
            target,
            can_buy=can_buy,
            can_sell=can_sell,
        )
        executed_weights[execution_day] = executed
        asset_returns = np.nan_to_num(open_returns[holding_return_day], nan=0.0)
        gross_strategy = float(executed @ asset_returns)
        benchmark_for_return = benchmark * portfolio.target_equity_exposure
        benchmark_return = float(benchmark_for_return @ asset_returns)

        change = executed - previous
        buy_turnover = float(np.maximum(change, 0.0).sum())
        sell_turnover = float(np.maximum(-change, 0.0).sum())
        day_cost = (
            buy_turnover
            * (costs.commission_rate + costs.transfer_fee_rate + costs.base_slippage)
            + sell_turnover
            * (
                costs.commission_rate
                + costs.transfer_fee_rate
                + costs.base_slippage
                + _stamp_duty(panel.dates[execution_day], costs)
            )
        )
        strategy_returns[holding_return_day] = gross_strategy - day_cost
        benchmark_returns[holding_return_day] = benchmark_return
        turnover[execution_day] = 0.5 * float(np.abs(change).sum())
        realized_costs[execution_day] = day_cost
        net_growth = 1.0 + gross_strategy - day_cost
        if net_growth <= 0:
            raise RuntimeError("Portfolio wealth became nonpositive.")
        previous = executed * (1.0 + asset_returns) / net_growth

    active = strategy_returns - benchmark_returns
    finite_strategy = strategy_returns[np.isfinite(strategy_returns)]
    finite_active = active[np.isfinite(active)]
    annualized_return = float(finite_strategy.mean() * 252.0) if finite_strategy.size else 0.0
    annualized_excess = float(finite_active.mean() * 252.0) if finite_active.size else 0.0
    return LongOnlyBacktestResult(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        active_returns=active,
        target_weights=target_weights,
        executed_weights=executed_weights,
        two_way_turnover=turnover,
        costs=realized_costs,
        annualized_return=annualized_return,
        annualized_sharpe=_annualized_sharpe(finite_strategy),
        annualized_excess_return=annualized_excess,
        information_ratio=_annualized_sharpe(finite_active),
        maximum_active_drawdown=_maximum_drawdown(finite_active),
    )
