from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from ashare_stat_arb.config import AdmissionConfig
from ashare_stat_arb.research_backtest import LongOnlyBacktestResult


@dataclass(frozen=True)
class PeriodMigrationMetrics:
    period: str
    start_date: str
    end_date: str
    observations: int
    annualized_benchmark_return: float
    annualized_gross_strategy_return: float
    annualized_net_strategy_return: float
    annualized_gross_excess_return: float
    annualized_net_excess_return: float
    gross_information_ratio: float
    net_information_ratio: float
    annualized_cost_drag: float
    cost_share_of_gross_alpha: float | None
    maximum_active_drawdown: float
    positive_rolling_12m_share: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MigrationAdmissionDecision:
    development_passed: bool
    validation_passed: bool
    overall_constraints_passed: bool
    migration_supported: bool
    holdout_action: str
    failed_gates: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_period(
    result: LongOnlyBacktestResult,
    dates: np.ndarray,
    *,
    period: str,
    start_date: str,
    end_date: str,
    rolling_window: int = 252,
) -> PeriodMigrationMetrics:
    trading_dates = np.asarray(dates, dtype="datetime64[D]")
    mask = (
        (trading_dates >= np.datetime64(start_date))
        & (trading_dates <= np.datetime64(end_date))
        & np.isfinite(result.strategy_returns)
        & np.isfinite(result.benchmark_returns)
    )
    strategy = result.strategy_returns[mask]
    benchmark = result.benchmark_returns[mask]
    costs = result.costs[mask]
    gross_strategy = strategy + costs
    net_active = strategy - benchmark
    gross_active = gross_strategy - benchmark
    gross_excess = _annualized_mean(gross_active)
    cost_drag = _annualized_mean(costs)
    cost_share = cost_drag / gross_excess if gross_excess > 0.0 else None
    return PeriodMigrationMetrics(
        period=period,
        start_date=start_date,
        end_date=end_date,
        observations=int(strategy.size),
        annualized_benchmark_return=_annualized_mean(benchmark),
        annualized_gross_strategy_return=_annualized_mean(gross_strategy),
        annualized_net_strategy_return=_annualized_mean(strategy),
        annualized_gross_excess_return=gross_excess,
        annualized_net_excess_return=_annualized_mean(net_active),
        gross_information_ratio=_annualized_sharpe(gross_active),
        net_information_ratio=_annualized_sharpe(net_active),
        annualized_cost_drag=cost_drag,
        cost_share_of_gross_alpha=cost_share,
        maximum_active_drawdown=_maximum_drawdown(net_active),
        positive_rolling_12m_share=_positive_rolling_excess_share(
            strategy,
            benchmark,
            rolling_window,
        ),
    )


def decide_migration(
    development: PeriodMigrationMetrics,
    validation: PeriodMigrationMetrics,
    overall: PeriodMigrationMetrics,
    admission: AdmissionConfig,
) -> MigrationAdmissionDecision:
    failed: list[str] = []

    def period_passes(metrics: PeriodMigrationMetrics) -> bool:
        prefix = metrics.period
        passed = True
        if not (
            metrics.annualized_gross_excess_return
            > admission.minimum_annualized_gross_excess_return
        ):
            failed.append(f"{prefix}: gross excess is not positive")
            passed = False
        if not (
            metrics.annualized_net_excess_return
            > admission.minimum_annualized_net_excess_return
        ):
            failed.append(f"{prefix}: net excess is not positive")
            passed = False
        if metrics.net_information_ratio < admission.minimum_net_information_ratio:
            failed.append(f"{prefix}: net IR is below 1.00")
            passed = False
        return passed

    development_passed = period_passes(development)
    validation_passed = period_passes(validation)
    overall_constraints_passed = True
    if overall.maximum_active_drawdown < -admission.maximum_active_drawdown:
        failed.append("overall: active drawdown exceeds 12%")
        overall_constraints_passed = False
    if (
        overall.cost_share_of_gross_alpha is None
        or overall.cost_share_of_gross_alpha
        > admission.maximum_cost_share_of_gross_alpha
    ):
        failed.append("overall: costs exceed 50% of gross alpha or gross alpha is nonpositive")
        overall_constraints_passed = False
    if (
        overall.positive_rolling_12m_share is None
        or overall.positive_rolling_12m_share
        < admission.minimum_positive_rolling_12m_share
    ):
        failed.append("overall: positive rolling 12-month excess share is below 60%")
        overall_constraints_passed = False

    supported = development_passed and validation_passed and overall_constraints_passed
    return MigrationAdmissionDecision(
        development_passed=development_passed,
        validation_passed=validation_passed,
        overall_constraints_passed=overall_constraints_passed,
        migration_supported=supported,
        holdout_action=(
            "eligible for human review before one-time holdout access"
            if supported
            else "keep 2023-2025 holdout sealed"
        ),
        failed_gates=tuple(failed),
    )


def _annualized_mean(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    return float(data.mean() * 252.0) if data.size else 0.0


def _annualized_sharpe(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    if not data.size:
        return 0.0
    volatility = float(data.std(ddof=0))
    return float(data.mean() / volatility * math.sqrt(252.0)) if volatility > 0 else 0.0


def _maximum_drawdown(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    if not data.size:
        return 0.0
    wealth = np.cumprod(1.0 + data)
    peaks = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peaks - 1.0))


def _positive_rolling_excess_share(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    window: int,
) -> float | None:
    strategy = np.asarray(strategy_returns, dtype=np.float64)
    benchmark = np.asarray(benchmark_returns, dtype=np.float64)
    if window < 2:
        raise ValueError("rolling_window must be at least two.")
    if strategy.size < window or benchmark.shape != strategy.shape:
        return None
    positive = 0
    total = strategy.size - window + 1
    for end in range(window, strategy.size + 1):
        strategy_growth = float(np.prod(1.0 + strategy[end - window : end]))
        benchmark_growth = float(np.prod(1.0 + benchmark[end - window : end]))
        positive += strategy_growth / benchmark_growth - 1.0 > 0.0
    return float(positive / total)
