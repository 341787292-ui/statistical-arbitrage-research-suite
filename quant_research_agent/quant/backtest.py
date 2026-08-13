from __future__ import annotations

from dataclasses import asdict, dataclass

from quant_research_agent.quant.statistics import (
    annualized_return,
    annualized_volatility,
    fit_linear,
    max_drawdown,
    pct_change,
    rolling_zscore,
    sharpe_ratio,
)
from quant_research_agent.quant.synthetic_data import PricePoint


@dataclass(frozen=True)
class BacktestConfig:
    lookback: int = 30
    entry_z: float = 1.5
    exit_z: float = 0.1
    transaction_cost: float = 0.0005


def run_pair_spread_backtest(
    prices: list[PricePoint],
    config: BacktestConfig | None = None,
) -> dict:
    if len(prices) < 60:
        raise ValueError("Need at least 60 price points for the baseline backtest.")

    cfg = config or BacktestConfig()
    calibration_days = cfg.lookback * 2
    if calibration_days >= len(prices):
        raise ValueError("Calibration period must leave data for out-of-sample trading.")
    asset_a = [point.asset_a for point in prices]
    asset_b = [point.asset_b for point in prices]
    fit = fit_linear(asset_b[:calibration_days], asset_a[:calibration_days])
    spread = [
        value_a - (fit.alpha + fit.beta * value_b)
        for value_a, value_b in zip(asset_a, asset_b)
    ]
    zscores = rolling_zscore(spread, cfg.lookback)
    returns_a = pct_change(asset_a)
    returns_b = pct_change(asset_b)

    positions: list[int] = []
    strategy_returns: list[float] = []
    turnovers: list[float] = []
    previous_position = 0

    for index, zscore in enumerate(zscores):
        position = previous_position
        if index < calibration_days or zscore is None:
            position = 0
        elif zscore > cfg.entry_z:
            position = -1
        elif zscore < -cfg.entry_z:
            position = 1
        elif abs(zscore) < cfg.exit_z:
            position = 0

        pair_return = (returns_a[index] - fit.beta * returns_b[index]) / (1.0 + abs(fit.beta))
        turnover = abs(position - previous_position)
        net_return = previous_position * pair_return - turnover * cfg.transaction_cost

        positions.append(position)
        strategy_returns.append(net_return)
        turnovers.append(float(turnover))
        previous_position = position

    trade_count = sum(1 for item in turnovers if item > 0)
    result = {
        "pair": "SYNTH_A/SYNTH_B",
        "parameters": asdict(cfg),
        "hedge_model": asdict(fit),
        "metrics": {
            "annual_return": annualized_return(strategy_returns),
            "annualized_volatility": annualized_volatility(strategy_returns),
            "sharpe": sharpe_ratio(strategy_returns),
            "max_drawdown": max_drawdown(strategy_returns),
        },
        "diagnostics": {
            "calibration_days": calibration_days,
            "average_turnover": sum(turnovers) / len(turnovers),
            "trade_count": trade_count,
            "non_zero_position_days": sum(1 for item in positions if item != 0),
            "first_position_index": next(
                (index for index, item in enumerate(positions) if item != 0),
                None,
            ),
        },
        "latest_observation": {
            "date": prices[-1].date.isoformat(),
            "spread": spread[-1],
            "zscore": zscores[-1],
            "position": positions[-1],
        },
    }
    return _round_nested(result)


def _round_nested(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_nested(item) for item in value]
    return value
