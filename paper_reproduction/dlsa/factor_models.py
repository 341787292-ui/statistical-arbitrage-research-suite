from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAResidualResult:
    residual_returns: np.ndarray
    active_mask: np.ndarray
    explained_variance: np.ndarray
    composition_matrices: np.ndarray | None


def rolling_pca_residuals(
    excess_returns: np.ndarray,
    *,
    n_factors: int = 5,
    covariance_window: int = 252,
    loading_window: int = 60,
    store_composition: bool = False,
    dtype: np.dtype = np.float64,
) -> PCAResidualResult:
    """Create one-day-ahead PCA residual portfolios without look-ahead bias.

    For trading day ``t``, PCA directions and stock loadings are estimated only
    with returns strictly before ``t``. The returned composition matrix ``Phi``
    satisfies ``epsilon_t = Phi_t @ return_t`` in column-vector notation.
    """

    returns = np.asarray(excess_returns, dtype=dtype)
    if returns.ndim != 2:
        raise ValueError("excess_returns must have shape (time, assets).")
    if covariance_window < loading_window:
        raise ValueError("covariance_window must be at least loading_window.")
    if n_factors < 0:
        raise ValueError("n_factors must be non-negative.")

    time_count, asset_count = returns.shape
    if time_count <= covariance_window:
        raise ValueError("Not enough observations for the PCA covariance window.")

    residuals = np.full_like(returns, np.nan, dtype=dtype)
    active_mask = np.zeros((time_count, asset_count), dtype=bool)
    explained = np.full((time_count, n_factors), np.nan, dtype=dtype)
    compositions = (
        np.zeros((time_count, asset_count, asset_count), dtype=np.float32)
        if store_composition
        else None
    )

    for t in range(covariance_window, time_count):
        history = returns[t - covariance_window : t]
        active = np.all(np.isfinite(history), axis=0) & np.isfinite(returns[t])
        active_indices = np.flatnonzero(active)
        if active_indices.size <= n_factors:
            continue

        history_active = history[:, active]
        means = history_active.mean(axis=0)
        vol = np.sqrt(np.mean((history_active - means) ** 2, axis=0))
        nonconstant = vol > np.finfo(dtype).eps
        if not np.all(nonconstant):
            active_indices = active_indices[nonconstant]
            active = np.zeros(asset_count, dtype=bool)
            active[active_indices] = True
            history_active = history[:, active]
            means = history_active.mean(axis=0)
            vol = np.sqrt(np.mean((history_active - means) ** 2, axis=0))
        if active_indices.size <= n_factors:
            continue

        current = returns[t, active]
        if n_factors == 0:
            residuals[t, active] = current
            active_mask[t, active] = True
            if compositions is not None:
                compositions[t][np.ix_(active, active)] = np.eye(active_indices.size)
            continue

        standardized = (history_active - means) / vol
        correlation = standardized.T @ standardized
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        order = np.argsort(eigenvalues)[::-1][:n_factors]
        directions = eigenvectors[:, order]

        total_variance = float(np.maximum(eigenvalues, 0).sum())
        if total_variance > 0:
            explained[t] = np.maximum(eigenvalues[order], 0) / total_variance

        loading_returns = history_active[-loading_window:]
        factor_history = (loading_returns / vol) @ directions
        factor_loadings = np.linalg.lstsq(
            factor_history,
            loading_returns,
            rcond=None,
        )[0]
        current_factors = (current / vol) @ directions
        residuals[t, active] = current - current_factors @ factor_loadings
        active_mask[t, active] = True

        if compositions is not None:
            inverse_vol = np.diag(1.0 / vol)
            phi_active = (
                np.eye(active_indices.size)
                - factor_loadings.T @ directions.T @ inverse_vol
            )
            compositions[t][np.ix_(active, active)] = phi_active.astype(np.float32)

    return PCAResidualResult(
        residual_returns=residuals,
        active_mask=active_mask,
        explained_variance=explained,
        composition_matrices=compositions,
    )

