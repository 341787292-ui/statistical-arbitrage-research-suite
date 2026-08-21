from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def performance_metrics(
    daily_returns: np.ndarray,
    *,
    annualization: int = 252,
) -> dict[str, float | int]:
    """Compute the paper metrics plus path diagnostics from daily returns."""

    returns = np.asarray(daily_returns, dtype=np.float64)
    if returns.ndim != 1 or returns.size == 0:
        raise ValueError("Daily returns must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(returns)):
        raise ValueError("Daily returns must all be finite.")
    if np.any(returns <= -1.0):
        raise ValueError("Daily simple returns must be greater than -1.")

    annualized_mean = float(returns.mean() * annualization)
    annualized_volatility = float(returns.std(ddof=0) * np.sqrt(annualization))
    annualized_sharpe = (
        annualized_mean / annualized_volatility
        if annualized_volatility > 0
        else 0.0
    )
    wealth = np.cumprod(1.0 + returns)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    return {
        "days": int(returns.size),
        "annualized_mean": annualized_mean,
        "annualized_volatility": annualized_volatility,
        "annualized_sharpe": annualized_sharpe,
        "cumulative_return": float(wealth[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "positive_day_rate": float(np.mean(returns > 0)),
        "arithmetic_return_sum": float(returns.sum()),
    }


def rolling_annualized_sharpe(
    daily_returns: np.ndarray,
    *,
    window: int = 252,
    annualization: int = 252,
) -> np.ndarray:
    """Return a trailing Sharpe series with NaN before the first full window."""

    returns = np.asarray(daily_returns, dtype=np.float64)
    if returns.ndim != 1 or returns.size == 0:
        raise ValueError("Daily returns must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(returns)):
        raise ValueError("Daily returns must all be finite.")
    if window < 2 or window > returns.size:
        raise ValueError("Rolling window must lie between 2 and the return count.")

    cumulative = np.concatenate(([0.0], np.cumsum(returns)))
    cumulative_squared = np.concatenate(([0.0], np.cumsum(returns**2)))
    window_sum = cumulative[window:] - cumulative[:-window]
    window_squared_sum = cumulative_squared[window:] - cumulative_squared[:-window]
    window_mean = window_sum / window
    window_variance = np.maximum(window_squared_sum / window - window_mean**2, 0.0)
    window_volatility = np.sqrt(window_variance)
    values = np.divide(
        window_mean * np.sqrt(annualization),
        window_volatility,
        out=np.zeros_like(window_mean),
        where=window_volatility > 0,
    )
    result = np.full(returns.size, np.nan, dtype=np.float64)
    result[window - 1 :] = values
    return result


def concatenate_blocks(blocks: Sequence[np.ndarray]) -> np.ndarray:
    """Validate and concatenate one-dimensional result blocks."""

    if not blocks:
        raise ValueError("At least one result block is required.")
    arrays = [np.asarray(block) for block in blocks]
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Every result block must be one-dimensional.")
    return np.concatenate(arrays)
