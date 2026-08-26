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
    rebalance_decisions: np.ndarray
    mandatory_rebalance_decisions: np.ndarray
    effective_turnover_limits: np.ndarray
    annualized_return: float
    annualized_gross_return: float
    annualized_benchmark_return: float
    annualized_sharpe: float
    annualized_excess_return: float
    annualized_gross_excess_return: float
    information_ratio: float
    gross_information_ratio: float
    annualized_cost_drag: float
    maximum_active_drawdown: float


def _covariance(returns: np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    column_means = np.divide(
        np.where(finite, values, 0.0).sum(axis=0),
        counts,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=counts > 0,
    )
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
    decision_interval: int = 1,
    turnover_penalty: float = 0.001,
    force_rebalance_on_universe_change: bool = False,
) -> LongOnlyBacktestResult:
    """Run a daily close-decision, next-open long-only research approximation."""

    portfolio = portfolio or PortfolioConfig()
    costs = costs or CostConfig()
    alpha = np.asarray(stock_alpha, dtype=np.float64)
    if alpha.shape != panel.shape:
        raise ValueError("stock_alpha must align with the daily panel.")
    if covariance_window < 20:
        raise ValueError("covariance_window must be at least 20 days.")
    if decision_interval < 1:
        raise ValueError("decision_interval must be at least one trading day.")
    if turnover_penalty < 0:
        raise ValueError("turnover_penalty must be nonnegative.")

    time_count, asset_count = panel.shape
    first_decision = covariance_window if start is None else max(start, covariance_window)
    adjusted_returns = panel.adjusted_returns()
    open_returns = panel.open_to_open_returns()

    target_weights = np.zeros(panel.shape, dtype=np.float64)
    executed_weights = np.zeros(panel.shape, dtype=np.float64)
    strategy_returns = np.full(time_count, np.nan, dtype=np.float64)
    benchmark_returns = np.full(time_count, np.nan, dtype=np.float64)
    turnover = np.zeros(time_count, dtype=np.float64)
    realized_costs = np.zeros(time_count, dtype=np.float64)
    rebalance_decisions = np.zeros(time_count, dtype=bool)
    mandatory_rebalance_decisions = np.zeros(time_count, dtype=bool)
    effective_turnover_limits = np.zeros(time_count, dtype=np.float64)

    initial_benchmark = np.where(
        panel.member[first_decision], panel.benchmark_weight[first_decision], 0.0
    )
    initial_benchmark = (
        initial_benchmark / initial_benchmark.sum() * portfolio.target_equity_exposure
    )
    previous = initial_benchmark
    pending_target = initial_benchmark.copy()
    pending_mask = np.zeros(asset_count, dtype=bool)
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
        forced_exit_mass = float(previous[~eligible & (previous > 1e-8)].sum())
        universe_changed = not np.array_equal(
            panel.member[decision], panel.member[decision - 1]
        )
        previous_eligible = panel.member[decision - 1] & ~panel.is_st[decision - 1]
        eligibility_changed = not np.array_equal(eligible, previous_eligible)
        mandatory_rebalance = force_rebalance_on_universe_change and (
            universe_changed or eligibility_changed
        )
        should_rebalance = (
            (decision - first_decision) % decision_interval == 0
            or mandatory_rebalance
        )
        if should_rebalance:
            month = str(panel.dates[decision].astype("datetime64[M]"))
            if month != risk_month or universe_changed:
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
            effective_turnover_limit = (
                portfolio.maximum_two_way_turnover
                + (forced_exit_mass if force_rebalance_on_universe_change else 0.0)
            )
            try:
                result = optimize_long_only_index_enhancement(
                    alpha[decision, working_indices] * alpha_scale,
                    benchmark[working_indices],
                    previous[working_indices],
                    covariance_psd,
                    equity_exposure=portfolio.target_equity_exposure,
                    maximum_stock_weight=portfolio.maximum_stock_weight,
                    maximum_two_way_turnover=effective_turnover_limit,
                    annual_tracking_error_limit=portfolio.central_tracking_error,
                    turnover_penalty=turnover_penalty,
                    maximum_industry_deviation=portfolio.maximum_industry_deviation,
                    maximum_style_deviation=portfolio.maximum_style_deviation,
                    eligible=eligible[working_indices],
                    covariance_is_psd=True,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "Portfolio optimization failed on decision date "
                    f"{panel.dates[decision]} with {working_indices.size} working "
                    f"stocks, discretionary turnover cap "
                    f"{portfolio.maximum_two_way_turnover:.4f}, and forced exit "
                    f"mass {forced_exit_mass:.4f}."
                ) from exc
            target = np.zeros(asset_count, dtype=np.float64)
            target[working_indices] = result.weights
            rebalance_decisions[decision] = True
            mandatory_rebalance_decisions[decision] = mandatory_rebalance
            effective_turnover_limits[decision] = effective_turnover_limit
        else:
            target = previous.copy()
            target[pending_mask] = pending_target[pending_mask]
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
        pending_mask = np.abs(executed - target) > 1e-8
        pending_target = target.copy()
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
        previous[np.abs(previous) < 1e-10] = 0.0

    active = strategy_returns - benchmark_returns
    gross_strategy_returns = strategy_returns + realized_costs
    gross_active = gross_strategy_returns - benchmark_returns
    finite_strategy = strategy_returns[np.isfinite(strategy_returns)]
    finite_gross_strategy = gross_strategy_returns[np.isfinite(gross_strategy_returns)]
    finite_benchmark = benchmark_returns[np.isfinite(benchmark_returns)]
    finite_active = active[np.isfinite(active)]
    finite_gross_active = gross_active[np.isfinite(gross_active)]
    finite_costs = realized_costs[np.isfinite(strategy_returns)]
    annualized_return = float(finite_strategy.mean() * 252.0) if finite_strategy.size else 0.0
    annualized_gross_return = (
        float(finite_gross_strategy.mean() * 252.0) if finite_gross_strategy.size else 0.0
    )
    annualized_benchmark_return = (
        float(finite_benchmark.mean() * 252.0) if finite_benchmark.size else 0.0
    )
    annualized_excess = float(finite_active.mean() * 252.0) if finite_active.size else 0.0
    annualized_gross_excess = (
        float(finite_gross_active.mean() * 252.0) if finite_gross_active.size else 0.0
    )
    return LongOnlyBacktestResult(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        active_returns=active,
        target_weights=target_weights,
        executed_weights=executed_weights,
        two_way_turnover=turnover,
        costs=realized_costs,
        rebalance_decisions=rebalance_decisions,
        mandatory_rebalance_decisions=mandatory_rebalance_decisions,
        effective_turnover_limits=effective_turnover_limits,
        annualized_return=annualized_return,
        annualized_gross_return=annualized_gross_return,
        annualized_benchmark_return=annualized_benchmark_return,
        annualized_sharpe=_annualized_sharpe(finite_strategy),
        annualized_excess_return=annualized_excess,
        annualized_gross_excess_return=annualized_gross_excess,
        information_ratio=_annualized_sharpe(finite_active),
        gross_information_ratio=_annualized_sharpe(finite_gross_active),
        annualized_cost_drag=(
            float(finite_costs.mean() * 252.0) if finite_costs.size else 0.0
        ),
        maximum_active_drawdown=_maximum_drawdown(finite_active),
    )
