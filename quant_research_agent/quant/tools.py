from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from quant_research_agent.quant.backtest import BacktestConfig, run_pair_spread_backtest
from quant_research_agent.quant.synthetic_data import generate_synthetic_pair


def run_stat_arb_experiment(
    *,
    lookback: int = 30,
    entry_z: float = 1.5,
    exit_z: float = 0.1,
    transaction_cost: float = 0.0005,
) -> dict:
    """Agent-facing tool for the first deterministic statistical arbitrage baseline."""
    prices = generate_synthetic_pair()
    config = BacktestConfig(
        lookback=lookback,
        entry_z=entry_z,
        exit_z=exit_z,
        transaction_cost=transaction_cost,
    )
    return run_pair_spread_backtest(prices, config=config)


def run_cost_sensitivity_experiment(
    *,
    transaction_costs: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.002),
    lookback: int = 30,
    entry_z: float = 1.5,
    exit_z: float = 0.1,
) -> dict:
    """Test whether the baseline survives increasingly conservative cost assumptions."""
    scenarios: list[dict] = []
    for cost in transaction_costs:
        result = run_stat_arb_experiment(
            lookback=lookback,
            entry_z=entry_z,
            exit_z=exit_z,
            transaction_cost=cost,
        )
        scenarios.append(
            {
                "transaction_cost": cost,
                "annual_return": result["metrics"]["annual_return"],
                "sharpe": result["metrics"]["sharpe"],
                "max_drawdown": result["metrics"]["max_drawdown"],
            }
        )

    positive_scenarios = sum(item["sharpe"] > 0 for item in scenarios)
    return {
        "experiment": "cost_sensitivity",
        "scenarios": scenarios,
        "summary": {
            "positive_sharpe_scenarios": positive_scenarios,
            "total_scenarios": len(scenarios),
            "survives_highest_cost": scenarios[-1]["sharpe"] > 0,
            "sharpe_change": round(scenarios[-1]["sharpe"] - scenarios[0]["sharpe"], 6),
        },
    }


def run_period_stability_experiment(
    *,
    n_days: int = 504,
    lookback: int = 30,
    entry_z: float = 1.5,
    exit_z: float = 0.1,
    transaction_cost: float = 0.0005,
) -> dict:
    """Split the controlled sample into two periods and compare performance."""
    prices = generate_synthetic_pair(n_days=n_days)
    midpoint = len(prices) // 2
    periods = [("first_half", prices[:midpoint]), ("second_half", prices[midpoint:])]
    results: list[dict] = []
    for label, sample in periods:
        result = run_pair_spread_backtest(
            sample,
            config=BacktestConfig(
                lookback=lookback,
                entry_z=entry_z,
                exit_z=exit_z,
                transaction_cost=transaction_cost,
            ),
        )
        results.append(
            {
                "period": label,
                "start": sample[0].date.isoformat(),
                "end": sample[-1].date.isoformat(),
                "annual_return": result["metrics"]["annual_return"],
                "sharpe": result["metrics"]["sharpe"],
                "max_drawdown": result["metrics"]["max_drawdown"],
            }
        )

    sharpe_gap = abs(results[0]["sharpe"] - results[1]["sharpe"])
    return {
        "experiment": "period_stability",
        "periods": results,
        "summary": {
            "both_periods_positive": all(item["sharpe"] > 0 for item in results),
            "sharpe_gap": round(sharpe_gap, 6),
            "stable_within_threshold": sharpe_gap <= 1.0,
        },
    }


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    function: Callable[..., dict]


class QuantToolRegistry:
    """Small explicit registry that keeps tool choice auditable."""

    def __init__(self) -> None:
        tools = [
            AgentTool(
                name="run_stat_arb_experiment",
                description="Run the deterministic pair-spread baseline backtest.",
                function=run_stat_arb_experiment,
            ),
            AgentTool(
                name="run_cost_sensitivity_experiment",
                description="Test the strategy under multiple transaction cost assumptions.",
                function=run_cost_sensitivity_experiment,
            ),
            AgentTool(
                name="run_period_stability_experiment",
                description="Compare baseline performance across two sample periods.",
                function=run_period_stability_experiment,
            ),
        ]
        self._tools = {tool.name: tool for tool in tools}

    def invoke(self, name: str, **kwargs: Any) -> dict:
        if name not in self._tools:
            available = ", ".join(sorted(self._tools))
            raise KeyError(f"Unknown quant tool '{name}'. Available tools: {available}")
        return self._tools[name].function(**kwargs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)
