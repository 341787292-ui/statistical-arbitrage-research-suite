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


def run_pair_spread_backtest_next_open(
    prices: list[PricePoint],
    config: BacktestConfig | None = None,
) -> dict:
    """Run the baseline with close-t signals executed at open t+1.

    This path exists separately from the original engineering baseline so the
    latter remains an unchanged experimental control.
    """

    if len(prices) < 100:
        raise ValueError("Need at least 100 price points for the verified backtest.")

    cfg = config or BacktestConfig()
    calibration_days = cfg.lookback * 2
    if calibration_days >= len(prices) - 2:
        raise ValueError("Calibration period must leave data for delayed execution.")

    asset_a_close = [point.asset_a for point in prices]
    asset_b_close = [point.asset_b for point in prices]
    asset_a_open = [point.asset_a_open for point in prices]
    asset_b_open = [point.asset_b_open for point in prices]
    fit = fit_linear(
        asset_b_close[:calibration_days],
        asset_a_close[:calibration_days],
    )
    spread = [
        value_a - (fit.alpha + fit.beta * value_b)
        for value_a, value_b in zip(asset_a_close, asset_b_close)
    ]
    zscores = rolling_zscore(spread, cfg.lookback)

    held_position = 0
    held_positions: list[int] = []
    desired_positions: list[int] = []
    strategy_returns: list[float] = []
    turnovers: list[float] = []
    execution_events: list[dict] = []

    for signal_session in range(len(prices) - 1):
        desired_position = held_position
        zscore = zscores[signal_session]
        if signal_session < calibration_days or zscore is None:
            desired_position = 0
        elif zscore > cfg.entry_z:
            desired_position = -1
        elif zscore < -cfg.entry_z:
            desired_position = 1
        elif abs(zscore) < cfg.exit_z:
            desired_position = 0

        next_session = signal_session + 1
        return_a = (asset_a_open[next_session] / asset_a_open[signal_session]) - 1.0
        return_b = (asset_b_open[next_session] / asset_b_open[signal_session]) - 1.0
        pair_return = (return_a - fit.beta * return_b) / (1.0 + abs(fit.beta))
        turnover = abs(desired_position - held_position)
        net_return = held_position * pair_return - turnover * cfg.transaction_cost

        if turnover > 0:
            execution_events.append(
                {
                    "signal_session": signal_session,
                    "signal_date": prices[signal_session].date.isoformat(),
                    "execution_session": next_session,
                    "execution_date": prices[next_session].date.isoformat(),
                    "position_before": held_position,
                    "position_after": desired_position,
                }
            )

        held_positions.append(held_position)
        desired_positions.append(desired_position)
        strategy_returns.append(net_return)
        turnovers.append(float(turnover))
        held_position = desired_position

    first_event = execution_events[0] if execution_events else None
    holdout_sessions = min(50, len(prices) // 5)
    holdout_start_session = len(prices) - holdout_sessions
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
            "trade_count": len(execution_events),
            "non_zero_position_periods": sum(
                1 for item in held_positions if item != 0
            ),
            "first_signal_session": (
                first_event["signal_session"] if first_event else None
            ),
            "first_execution_session": (
                first_event["execution_session"] if first_event else None
            ),
        },
        "timing_contract": {
            "session_count": len(prices),
            "training_start_session": 0,
            "training_end_session": calibration_days - 1,
            "model_fit_session": calibration_days,
            "signal_observation_phase": "close",
            "signal_generation_phase": "after_close",
            "execution_phase": "open",
            "return_window_start_phase": "open",
            "development_end_session": holdout_start_session - 1,
            "holdout_start_session": holdout_start_session,
            "holdout_end_session": len(prices) - 1,
            "holdout_accessed_during_selection": False,
        },
        "execution_events": execution_events,
        "latest_observation": {
            "date": prices[-2].date.isoformat(),
            "spread": spread[-2],
            "zscore": zscores[-2],
            "desired_position_for_next_open": desired_positions[-1],
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
