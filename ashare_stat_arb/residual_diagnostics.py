from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from paper_reproduction.dlsa.ou import fit_ou, ou_threshold_weight


@dataclass(frozen=True)
class ResidualMechanismResult:
    annualized_mean: float
    annualized_volatility: float
    annualized_sharpe: float
    average_daily_turnover: float
    active_day_rate: float
    average_active_positions: float
    observations: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def ou_residual_positions(
    residual_returns: np.ndarray,
    *,
    lookback: int = 30,
    entry_threshold: float = 1.25,
    min_r_squared: float = 0.25,
) -> np.ndarray:
    """Create close-time OU positions from residual history through that close."""

    residuals = np.asarray(residual_returns, dtype=np.float64)
    if residuals.ndim != 2:
        raise ValueError("residual_returns must have shape (time, residuals).")
    if lookback < 3:
        raise ValueError("lookback must be at least three observations.")
    positions = np.zeros_like(residuals)
    for row in range(lookback - 1, residuals.shape[0] - 1):
        history = residuals[row - lookback + 1 : row + 1]
        positions[row] = ou_positions_from_history(
            history,
            entry_threshold=entry_threshold,
            min_r_squared=min_r_squared,
        )
    return positions


def ou_positions_from_history(
    residual_history: np.ndarray,
    *,
    entry_threshold: float,
    min_r_squared: float,
) -> np.ndarray:
    """Fit one OU position vector from a fixed residual definition."""

    history = np.asarray(residual_history, dtype=np.float64)
    if history.ndim != 2 or history.shape[0] < 3:
        raise ValueError("residual_history must have at least three time rows.")
    positions = np.zeros(history.shape[1], dtype=np.float64)
    valid = np.all(np.isfinite(history), axis=0)
    for column in np.flatnonzero(valid):
        positions[column] = ou_threshold_weight(
            fit_ou(np.cumsum(history[:, column])),
            entry_threshold=entry_threshold,
            min_r_squared=min_r_squared,
        )
    return positions


def evaluate_residual_positions(
    residual_returns: np.ndarray,
    positions: np.ndarray,
    *,
    start: int = 0,
    direction: float = 1.0,
    annualization: int = 252,
) -> ResidualMechanismResult:
    """Evaluate a unit-gross residual portfolio one day out of sample.

    This is a theoretical mechanism test. It ignores A-share execution and
    does not represent an investable cash-equity portfolio.
    """

    residuals = np.asarray(residual_returns, dtype=np.float64)
    raw_positions = np.asarray(positions, dtype=np.float64)
    if residuals.shape != raw_positions.shape or residuals.ndim != 2:
        raise ValueError("residual_returns and positions must align as (time, residuals).")
    if direction not in {-1.0, 1.0}:
        raise ValueError("direction must be either 1.0 or -1.0.")

    normalized = np.zeros_like(raw_positions)
    gross = np.abs(raw_positions).sum(axis=1)
    active = gross > 0
    normalized[active] = direction * raw_positions[active] / gross[active, None]
    daily_returns = np.full(residuals.shape[0], np.nan, dtype=np.float64)
    for row in range(max(start, 0), residuals.shape[0] - 1):
        if not active[row]:
            daily_returns[row + 1] = 0.0
            continue
        next_residual = residuals[row + 1]
        valid = np.isfinite(next_residual)
        daily_returns[row + 1] = float(normalized[row, valid] @ next_residual[valid])

    turnover = np.zeros(residuals.shape[0], dtype=np.float64)
    turnover[1:] = np.abs(normalized[1:] - normalized[:-1]).sum(axis=1)
    evaluation = daily_returns[max(start + 1, 1) :]
    finite = evaluation[np.isfinite(evaluation)]
    if finite.size == 0:
        return ResidualMechanismResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    mean = float(finite.mean())
    volatility = float(finite.std(ddof=0))
    selected_positions = raw_positions[max(start, 0) : -1]
    selected_active = np.count_nonzero(selected_positions, axis=1)
    return ResidualMechanismResult(
        annualized_mean=mean * annualization,
        annualized_volatility=volatility * math.sqrt(annualization),
        annualized_sharpe=(
            mean / volatility * math.sqrt(annualization) if volatility > 0 else 0.0
        ),
        average_daily_turnover=float(turnover[max(start, 0) :].mean()),
        active_day_rate=float((selected_active > 0).mean()) if selected_active.size else 0.0,
        average_active_positions=(
            float(selected_active.mean()) if selected_active.size else 0.0
        ),
        observations=int(finite.size),
    )
