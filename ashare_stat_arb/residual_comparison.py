from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ashare_stat_arb.residual_diagnostics import (
    ResidualMechanismResult,
    evaluate_residual_positions,
    ou_positions_from_history,
    ou_residual_positions,
)
from ashare_stat_arb.signals import MonthlyPCASignalResult, fit_pca_residual_map


@dataclass(frozen=True)
class ResidualDefinitionComparison:
    start_date: str
    end_date: str
    stitched_asof: ResidualMechanismResult
    current_composition: ResidualMechanismResult
    annualized_mean_delta: float
    sharpe_delta: float
    mechanism_sharpe_gate: float
    current_composition_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return payload


def compare_residual_definitions(
    excess_returns: np.ndarray,
    dates: np.ndarray,
    member: np.ndarray,
    stitched_signal: MonthlyPCASignalResult,
    *,
    start: int,
    n_factors: int,
    covariance_window: int,
    loading_window: int,
    residual_lookback: int,
    entry_threshold: float,
    min_r_squared: float,
    mechanism_sharpe_gate: float = 0.5,
) -> ResidualDefinitionComparison:
    """Compare two pre-registered residual histories with all parameters fixed.

    ``stitched_asof`` uses each day's then-current monthly PCA residual. The
    alternative re-expresses the full trailing OU window and its next return
    under the composition matrix available on the decision day.
    """

    returns = np.asarray(excess_returns, dtype=np.float64)
    trading_dates = np.asarray(dates, dtype="datetime64[D]")
    membership = np.asarray(member, dtype=bool)
    stitched_residuals = np.asarray(stitched_signal.residual_returns, dtype=np.float64)
    if returns.ndim != 2 or membership.shape != returns.shape:
        raise ValueError("excess_returns and member must align as (time, assets).")
    if stitched_residuals.shape != returns.shape:
        raise ValueError("stitched_signal must align with excess_returns.")
    if trading_dates.shape != (returns.shape[0],):
        raise ValueError("dates must align with excess_returns.")
    if start < covariance_window + residual_lookback - 1:
        raise ValueError("start must allow both PCA and residual lookback windows.")
    if start >= returns.shape[0] - 1:
        raise ValueError("start must leave at least one next-day observation.")

    stitched_positions = ou_residual_positions(
        stitched_residuals,
        lookback=residual_lookback,
        entry_threshold=entry_threshold,
        min_r_squared=min_r_squared,
    )
    current_positions = np.zeros_like(returns)
    current_next_residuals = np.full_like(returns, np.nan)
    current_indices = np.empty(0, dtype=np.int64)
    current_phi = np.empty((0, 0), dtype=np.float64)
    previous_month: str | None = None

    for row in range(covariance_window, returns.shape[0] - 1):
        month = str(trading_dates[row].astype("datetime64[M]"))
        if month != previous_month:
            history = returns[row - covariance_window : row]
            current_indices, current_phi = fit_pca_residual_map(
                history,
                membership[row],
                n_factors=n_factors,
                loading_window=loading_window,
            )
            previous_month = month
        if current_phi.size == 0 or row < start:
            continue

        raw_history = returns[
            row - residual_lookback + 1 : row + 1,
            current_indices,
        ]
        if not np.all(np.isfinite(raw_history)):
            continue
        current_history = raw_history @ current_phi.T
        current_positions[row, current_indices] = ou_positions_from_history(
            current_history,
            entry_threshold=entry_threshold,
            min_r_squared=min_r_squared,
        )

        next_returns = returns[row + 1, current_indices]
        if np.all(np.isfinite(next_returns)):
            current_next_residuals[row + 1, current_indices] = (
                current_phi @ next_returns
            )

    stitched_result = evaluate_residual_positions(
        stitched_residuals,
        stitched_positions,
        start=start,
    )
    current_result = evaluate_residual_positions(
        current_next_residuals,
        current_positions,
        start=start,
    )
    return ResidualDefinitionComparison(
        start_date=str(trading_dates[start]),
        end_date=str(trading_dates[-1]),
        stitched_asof=stitched_result,
        current_composition=current_result,
        annualized_mean_delta=(
            current_result.annualized_mean - stitched_result.annualized_mean
        ),
        sharpe_delta=(
            current_result.annualized_sharpe - stitched_result.annualized_sharpe
        ),
        mechanism_sharpe_gate=mechanism_sharpe_gate,
        current_composition_gate_passed=(
            current_result.annualized_sharpe >= mechanism_sharpe_gate
        ),
    )
