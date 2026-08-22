from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from paper_reproduction.dlsa.ou import fit_ou, ou_threshold_weight


@dataclass(frozen=True)
class MonthlyPCASignalResult:
    stock_alpha: np.ndarray
    residual_returns: np.ndarray
    active_count: np.ndarray
    refit_dates: tuple[str, ...]


def ou_stock_alpha(
    residual_returns: np.ndarray,
    composition_matrices: np.ndarray,
    *,
    member: np.ndarray | None = None,
    lookback: int = 30,
    entry_threshold: float = 1.25,
    min_r_squared: float = 0.25,
) -> np.ndarray:
    """Map lagged OU residual positions into stock-space alpha scores.

    Alpha at day ``t`` uses residual observations through the close of ``t``
    and is intended for execution no earlier than day ``t+1``.
    """

    residuals = np.asarray(residual_returns, dtype=np.float64)
    compositions = np.asarray(composition_matrices, dtype=np.float64)
    if residuals.ndim != 2:
        raise ValueError("residual_returns must have shape (time, residuals).")
    if compositions.ndim != 3 or compositions.shape[:2] != residuals.shape:
        raise ValueError(
            "composition_matrices must have shape (time, residuals, stocks)."
        )
    if lookback < 3:
        raise ValueError("lookback must be at least three observations.")

    time_count, residual_count = residuals.shape
    stock_count = compositions.shape[2]
    if member is None:
        membership = np.ones((time_count, stock_count), dtype=bool)
    else:
        membership = np.asarray(member, dtype=bool)
        if membership.shape != (time_count, stock_count):
            raise ValueError("member must align with stock-space output.")

    stock_alpha = np.zeros((time_count, stock_count), dtype=np.float64)
    for t in range(lookback - 1, time_count):
        history = residuals[t - lookback + 1 : t + 1]
        valid = np.all(np.isfinite(history), axis=0)
        residual_positions = np.zeros(residual_count, dtype=np.float64)
        for residual_index in np.flatnonzero(valid):
            residual_positions[residual_index] = ou_threshold_weight(
                fit_ou(np.cumsum(history[:, residual_index])),
                entry_threshold=entry_threshold,
                min_r_squared=min_r_squared,
            )
        mapped = residual_positions @ compositions[t]
        mapped[~membership[t]] = 0.0
        mapped -= mapped[membership[t]].mean() if np.any(membership[t]) else 0.0
        scale = float(mapped[membership[t]].std(ddof=0)) if np.any(membership[t]) else 0.0
        if scale > 0:
            stock_alpha[t] = mapped / scale
    return stock_alpha


def rolling_monthly_pca_ou_stock_alpha(
    excess_returns: np.ndarray,
    dates: np.ndarray,
    member: np.ndarray,
    *,
    n_factors: int = 5,
    covariance_window: int = 252,
    loading_window: int = 60,
    residual_lookback: int = 30,
    entry_threshold: float = 1.25,
    min_r_squared: float = 0.25,
) -> MonthlyPCASignalResult:
    """Create stock alpha without persisting daily stock-by-stock matrices.

    PCA directions and factor loadings are fitted on the first trading day of
    each month using only prior observations. The fitted residual map is then
    held fixed for that month while residuals and OU positions update daily.
    """

    returns = np.asarray(excess_returns, dtype=np.float64)
    trading_dates = np.asarray(dates, dtype="datetime64[D]")
    membership = np.asarray(member, dtype=bool)
    if returns.ndim != 2 or membership.shape != returns.shape:
        raise ValueError("returns and member must align as (time, assets).")
    if trading_dates.shape != (returns.shape[0],):
        raise ValueError("dates must align with the time dimension.")
    if covariance_window < loading_window:
        raise ValueError("covariance_window must be at least loading_window.")
    if residual_lookback < 3:
        raise ValueError("residual_lookback must be at least three days.")
    if n_factors < 0:
        raise ValueError("n_factors must be nonnegative.")

    time_count, asset_count = returns.shape
    residuals = np.full_like(returns, np.nan)
    stock_alpha = np.zeros_like(returns)
    active_count = np.zeros(time_count, dtype=np.int64)
    refit_dates: list[str] = []
    current_indices = np.empty(0, dtype=np.int64)
    current_phi = np.empty((0, 0), dtype=np.float64)
    previous_month: str | None = None

    for t in range(covariance_window, time_count):
        month = str(trading_dates[t].astype("datetime64[M]"))
        if month != previous_month:
            history = returns[t - covariance_window : t]
            eligible = membership[t] & np.all(np.isfinite(history), axis=0)
            current_indices = np.flatnonzero(eligible)
            current_phi = np.empty((0, 0), dtype=np.float64)
            if current_indices.size > n_factors:
                history_active = history[:, current_indices]
                means = history_active.mean(axis=0)
                vol = np.sqrt(np.mean((history_active - means) ** 2, axis=0))
                nonconstant = vol > np.finfo(np.float64).eps
                current_indices = current_indices[nonconstant]
                history_active = history[:, current_indices]
                means = history_active.mean(axis=0)
                vol = np.sqrt(np.mean((history_active - means) ** 2, axis=0))
                if current_indices.size > n_factors:
                    if n_factors == 0:
                        current_phi = np.eye(current_indices.size)
                    else:
                        standardized = (history_active - means) / vol
                        correlation = standardized.T @ standardized
                        lower = current_indices.size - n_factors
                        eigenvalues, directions = eigh(
                            correlation,
                            subset_by_index=(lower, current_indices.size - 1),
                            check_finite=False,
                        )
                        directions = directions[:, np.argsort(eigenvalues)[::-1]]
                        loading_returns = history_active[-loading_window:]
                        factor_history = (loading_returns / vol) @ directions
                        factor_loadings = np.linalg.lstsq(
                            factor_history,
                            loading_returns,
                            rcond=None,
                        )[0]
                        current_phi = (
                            np.eye(current_indices.size)
                            - factor_loadings.T @ directions.T @ np.diag(1.0 / vol)
                        )
            previous_month = month
            refit_dates.append(str(trading_dates[t]))

        if current_phi.size == 0:
            continue
        current_returns = returns[t, current_indices]
        valid_current = np.isfinite(current_returns)
        if not np.all(valid_current):
            continue
        residuals[t, current_indices] = current_phi @ current_returns
        active_count[t] = current_indices.size
        if t < covariance_window + residual_lookback - 1:
            continue

        history = residuals[t - residual_lookback + 1 : t + 1, current_indices]
        valid_history = np.all(np.isfinite(history), axis=0)
        residual_positions = np.zeros(current_indices.size, dtype=np.float64)
        for local_index in np.flatnonzero(valid_history):
            residual_positions[local_index] = ou_threshold_weight(
                fit_ou(np.cumsum(history[:, local_index])),
                entry_threshold=entry_threshold,
                min_r_squared=min_r_squared,
            )
        mapped = residual_positions @ current_phi
        mapped -= mapped.mean()
        scale = float(mapped.std(ddof=0))
        if scale > 0:
            stock_alpha[t, current_indices] = mapped / scale

    return MonthlyPCASignalResult(
        stock_alpha=stock_alpha,
        residual_returns=residuals,
        active_count=active_count,
        refit_dates=tuple(refit_dates),
    )
