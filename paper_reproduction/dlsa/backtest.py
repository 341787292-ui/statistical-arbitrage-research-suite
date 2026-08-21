from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paper_reproduction.dlsa.ou import fit_ou, ou_threshold_weight
from paper_reproduction.dlsa.preprocessing import cumulative_residual_windows


@dataclass(frozen=True)
class BacktestResult:
    returns: np.ndarray
    turnover: np.ndarray
    short_proportion: np.ndarray
    residual_weights: np.ndarray
    stock_weights: np.ndarray
    annualized_mean: float
    annualized_volatility: float
    annualized_sharpe: float
    used_stock_space_normalization: bool


@dataclass(frozen=True)
class CompactBacktestResult:
    """Metrics and diagnostics for a low-memory, full-universe backtest."""

    returns: np.ndarray
    turnover: np.ndarray
    short_proportion: np.ndarray
    active_positions: np.ndarray
    annualized_mean: float
    annualized_volatility: float
    annualized_sharpe: float
    evaluation_start: int
    used_stock_space_normalization: bool


def backtest_ou_threshold(
    residual_returns: np.ndarray,
    *,
    composition_matrices: np.ndarray | None = None,
    lookback: int = 30,
    entry_threshold: float = 1.25,
    min_r_squared: float = 0.25,
    transaction_cost: float = 0.0,
    short_holding_cost: float = 0.0,
    annualization: int = 252,
    zero_is_missing: bool = False,
) -> BacktestResult:
    """Evaluate the paper's OU+Threshold policy one day out of sample."""

    residuals = np.asarray(residual_returns, dtype=np.float64)
    windows, valid_windows = cumulative_residual_windows(
        residuals,
        lookback,
        zero_is_missing=zero_is_missing,
    )
    time_count, residual_count = residuals.shape
    if composition_matrices is not None:
        compositions = np.asarray(composition_matrices, dtype=np.float64)
        if compositions.shape[:2] != (time_count, residual_count):
            raise ValueError("composition_matrices must have shape (time, residuals, stocks).")
        stock_count = compositions.shape[2]
    else:
        compositions = None
        stock_count = residual_count

    evaluation_count = time_count - lookback
    residual_weights = np.zeros((evaluation_count, residual_count), dtype=np.float64)
    stock_weights = np.zeros((evaluation_count, stock_count), dtype=np.float64)
    strategy_returns = np.zeros(evaluation_count, dtype=np.float64)
    turnover = np.zeros(evaluation_count, dtype=np.float64)
    short_proportion = np.zeros(evaluation_count, dtype=np.float64)

    previous_stock_weights = np.zeros(stock_count, dtype=np.float64)
    for output_index, t in enumerate(range(lookback, time_count)):
        raw_residual_weights = np.zeros(residual_count, dtype=np.float64)
        tradable = valid_windows[output_index] & np.isfinite(residuals[t])
        for asset in np.flatnonzero(tradable):
            raw_residual_weights[asset] = ou_threshold_weight(
                fit_ou(windows[output_index, asset]),
                entry_threshold=entry_threshold,
                min_r_squared=min_r_squared,
            )

        if compositions is not None:
            raw_stock_weights = raw_residual_weights @ compositions[t]
            gross = float(np.abs(raw_stock_weights).sum())
        else:
            raw_stock_weights = raw_residual_weights.copy()
            gross = float(np.abs(raw_residual_weights).sum())

        if gross > 0:
            residual_weights[output_index] = raw_residual_weights / gross
            stock_weights[output_index] = raw_stock_weights / gross

        current_stock_weights = stock_weights[output_index]
        turnover[output_index] = float(
            np.abs(current_stock_weights - previous_stock_weights).sum()
        )
        short_proportion[output_index] = float(
            np.abs(np.minimum(current_stock_weights, 0.0)).sum()
        )
        finite_residuals = np.nan_to_num(residuals[t], nan=0.0)
        gross_return = float(residual_weights[output_index] @ finite_residuals)
        strategy_returns[output_index] = (
            gross_return
            - transaction_cost * turnover[output_index]
            - short_holding_cost * short_proportion[output_index]
        )
        previous_stock_weights = current_stock_weights

    mean = float(strategy_returns.mean()) if evaluation_count else 0.0
    volatility = float(strategy_returns.std(ddof=0)) if evaluation_count else 0.0
    annualized_mean = mean * annualization
    annualized_volatility = volatility * np.sqrt(annualization)
    annualized_sharpe = (
        annualized_mean / annualized_volatility if annualized_volatility > 0 else 0.0
    )
    return BacktestResult(
        returns=strategy_returns,
        turnover=turnover,
        short_proportion=short_proportion,
        residual_weights=residual_weights,
        stock_weights=stock_weights,
        annualized_mean=annualized_mean,
        annualized_volatility=annualized_volatility,
        annualized_sharpe=annualized_sharpe,
        used_stock_space_normalization=compositions is not None,
    )


def backtest_ou_threshold_streaming(
    residual_returns: np.ndarray,
    *,
    composition_matrices: np.ndarray | None = None,
    lookback: int = 30,
    evaluation_start: int | None = None,
    entry_threshold: float = 1.25,
    min_r_squared: float = 0.25,
    transaction_cost: float = 0.0,
    short_holding_cost: float = 0.0,
    annualization: int = 252,
    zero_is_missing: bool = True,
) -> CompactBacktestResult:
    """Run OU+Threshold without materializing all rolling windows.

    The authors encode missing residual observations as zero. This function
    follows their preprocessing rule by excluding a residual whenever any of
    its lagged observations is zero. It retains only daily diagnostics, so the
    published 4,781 x 9,483 arrays can be evaluated on an ordinary machine.
    """

    residuals = np.asarray(residual_returns, dtype=np.float64)
    if residuals.ndim != 2:
        raise ValueError("residual_returns must have shape (time, assets).")
    if lookback < 3:
        raise ValueError("lookback must be at least three observations.")
    start = lookback if evaluation_start is None else int(evaluation_start)
    if start < lookback or start >= residuals.shape[0]:
        raise ValueError("evaluation_start must be within the data after lookback.")

    time_count, residual_count = residuals.shape
    if composition_matrices is not None:
        compositions = np.asarray(composition_matrices, dtype=np.float64)
        if compositions.shape[:2] != (time_count, residual_count):
            raise ValueError("composition_matrices must have shape (time, residuals, stocks).")
        stock_count = compositions.shape[2]
    else:
        compositions = None
        stock_count = residual_count

    evaluation_count = time_count - start
    strategy_returns = np.zeros(evaluation_count, dtype=np.float64)
    turnover = np.zeros(evaluation_count, dtype=np.float64)
    short_proportion = np.zeros(evaluation_count, dtype=np.float64)
    active_positions = np.zeros(evaluation_count, dtype=np.int32)
    previous_stock_weights = np.zeros(stock_count, dtype=np.float64)

    epsilon = 1e-6
    for output_index, t in enumerate(range(start, time_count)):
        history = residuals[t - lookback : t]
        valid = np.all(np.isfinite(history), axis=0)
        if zero_is_missing:
            valid &= ~np.any(history == 0, axis=0)

        raw_residual_weights = np.zeros(residual_count, dtype=np.float64)
        if np.any(valid):
            cumulative = np.cumsum(history[:, valid], axis=0)
            x = cumulative[:-1]
            y = cumulative[1:]
            mean_x = x.mean(axis=0)
            mean_y = y.mean(axis=0)
            variance_x = x.var(axis=0)
            variance_y = y.var(axis=0)
            covariance = np.mean(
                (x - mean_x[None, :]) * (y - mean_y[None, :]),
                axis=0,
            )
            b = np.divide(
                covariance,
                variance_x,
                out=np.full_like(covariance, np.nan),
                where=variance_x > 0,
            )
            intercept = mean_y - b * mean_x
            long_run_mean = intercept / (1.0 - b + epsilon)
            innovations = y - b[None, :] * x - intercept[None, :]
            stationary_std = np.sqrt(
                innovations.var(axis=0) / np.abs(1.0 - b**2 + epsilon)
            )
            r_squared = np.divide(
                covariance**2,
                variance_x * variance_y,
                out=np.zeros_like(covariance),
                where=(variance_x > 0) & (variance_y > 0),
            )
            model_valid = (
                (b > 0)
                & (b < 1)
                & (stationary_std > 0)
                & np.isfinite(stationary_std)
                & (r_squared > min_r_squared)
            )
            ou_signal = np.zeros_like(b)
            ou_signal[model_valid] = (
                long_run_mean[model_valid] - y[-1, model_valid]
            ) / stationary_std[model_valid]
            selected_indices = np.flatnonzero(valid)
            raw_residual_weights[selected_indices[ou_signal > entry_threshold]] = 1.0
            raw_residual_weights[selected_indices[ou_signal < -entry_threshold]] = -1.0

        if compositions is None:
            raw_stock_weights = raw_residual_weights
        else:
            raw_stock_weights = raw_residual_weights @ compositions[t]
        gross = float(np.abs(raw_stock_weights).sum())
        if gross > 0:
            residual_weights = raw_residual_weights / gross
            stock_weights = raw_stock_weights / gross
        else:
            residual_weights = np.zeros_like(raw_residual_weights)
            stock_weights = np.zeros_like(raw_stock_weights)

        turnover[output_index] = float(np.abs(stock_weights - previous_stock_weights).sum())
        short_proportion[output_index] = float(
            np.abs(np.minimum(stock_weights, 0.0)).sum()
        )
        active_positions[output_index] = int(np.count_nonzero(raw_residual_weights))
        current_returns = np.nan_to_num(residuals[t], nan=0.0)
        gross_return = float(residual_weights @ current_returns)
        strategy_returns[output_index] = (
            gross_return
            - transaction_cost * turnover[output_index]
            - short_holding_cost * short_proportion[output_index]
        )
        previous_stock_weights = stock_weights

    mean = float(strategy_returns.mean())
    volatility = float(strategy_returns.std(ddof=0))
    annualized_mean = mean * annualization
    annualized_volatility = volatility * np.sqrt(annualization)
    annualized_sharpe = (
        annualized_mean / annualized_volatility if annualized_volatility > 0 else 0.0
    )
    return CompactBacktestResult(
        returns=strategy_returns,
        turnover=turnover,
        short_proportion=short_proportion,
        active_positions=active_positions,
        annualized_mean=annualized_mean,
        annualized_volatility=annualized_volatility,
        annualized_sharpe=annualized_sharpe,
        evaluation_start=start,
        used_stock_space_normalization=compositions is not None,
    )
