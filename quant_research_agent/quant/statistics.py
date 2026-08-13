from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LinearFit:
    alpha: float
    beta: float


def pct_change(values: list[float]) -> list[float]:
    returns = [0.0]
    for previous, current in zip(values, values[1:]):
        if previous == 0:
            returns.append(0.0)
        else:
            returns.append((current / previous) - 1.0)
    return returns


def fit_linear(x: list[float], y: list[float]) -> LinearFit:
    if len(x) != len(y) or not x:
        raise ValueError("x and y must be the same non-zero length.")
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    variance_x = sum((value - mean_x) ** 2 for value in x)
    if variance_x == 0:
        return LinearFit(alpha=mean_y, beta=0.0)
    covariance = sum((left - mean_x) * (right - mean_y) for left, right in zip(x, y))
    beta = covariance / variance_x
    alpha = mean_y - beta * mean_x
    return LinearFit(alpha=alpha, beta=beta)


def rolling_zscore(values: list[float], window: int) -> list[float | None]:
    if window < 2:
        raise ValueError("window must be at least 2.")
    zscores: list[float | None] = []
    for index, value in enumerate(values):
        if index < window:
            zscores.append(None)
            continue
        sample = values[index - window : index]
        mean = sum(sample) / len(sample)
        variance = sum((item - mean) ** 2 for item in sample) / (len(sample) - 1)
        std = math.sqrt(variance)
        zscores.append(None if std == 0 else (value - mean) / std)
    return zscores


def annualized_return(returns: list[float], periods_per_year: int = 252) -> float:
    if not returns:
        return 0.0
    value = 1.0
    for item in returns:
        value *= 1.0 + item
    return value ** (periods_per_year / len(returns)) - 1.0


def annualized_volatility(returns: list[float], periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((item - mean) ** 2 for item in returns) / (len(returns) - 1)
    return math.sqrt(variance) * math.sqrt(periods_per_year)


def sharpe_ratio(returns: list[float], periods_per_year: int = 252) -> float:
    vol = annualized_volatility(returns, periods_per_year=periods_per_year)
    if vol == 0:
        return 0.0
    return annualized_return(returns, periods_per_year=periods_per_year) / vol


def max_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for item in returns:
        equity *= 1.0 + item
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        worst = min(worst, drawdown)
    return abs(worst)
