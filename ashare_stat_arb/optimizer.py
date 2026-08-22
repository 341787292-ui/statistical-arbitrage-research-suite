from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class OptimizationResult:
    weights: np.ndarray
    status: str
    objective_value: float
    two_way_turnover: float
    annualized_tracking_error: float


def _psd_covariance(covariance: np.ndarray) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square.")
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    floor = max(float(np.max(eigenvalues)) * 1e-8, 1e-10)
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def optimize_long_only_index_enhancement(
    alpha: np.ndarray,
    benchmark_weights: np.ndarray,
    previous_weights: np.ndarray,
    covariance: np.ndarray,
    *,
    equity_exposure: float = 0.99,
    maximum_stock_weight: float = 0.015,
    maximum_two_way_turnover: float = 0.20,
    annual_tracking_error_limit: float = 0.08,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 0.001,
    industry_matrix: np.ndarray | None = None,
    maximum_industry_deviation: float = 0.03,
    style_matrix: np.ndarray | None = None,
    maximum_style_deviation: float = 0.50,
    eligible: np.ndarray | None = None,
    tradable: np.ndarray | None = None,
    covariance_is_psd: bool = False,
) -> OptimizationResult:
    """Create benchmark-relative long-only weights with hard A-share limits."""

    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError(
            "cvxpy is required. Install ashare_stat_arb/requirements.txt."
        ) from exc

    alpha_values = np.asarray(alpha, dtype=np.float64)
    benchmark = np.asarray(benchmark_weights, dtype=np.float64)
    previous = np.asarray(previous_weights, dtype=np.float64)
    if alpha_values.ndim != 1:
        raise ValueError("alpha must be one-dimensional.")
    if benchmark.shape != alpha_values.shape or previous.shape != alpha_values.shape:
        raise ValueError("alpha, benchmark_weights, and previous_weights must align.")
    if not 0 < equity_exposure <= 1:
        raise ValueError("equity_exposure must be in (0, 1].")
    if maximum_stock_weight * alpha_values.size + 1e-12 < equity_exposure:
        raise ValueError("maximum_stock_weight makes the portfolio infeasible.")

    benchmark = np.nan_to_num(benchmark, nan=0.0, posinf=0.0, neginf=0.0)
    benchmark = np.maximum(benchmark, 0.0)
    if benchmark.sum() <= 0:
        raise ValueError("benchmark_weights must contain positive mass.")
    benchmark = benchmark / benchmark.sum() * equity_exposure
    previous = np.nan_to_num(previous, nan=0.0, posinf=0.0, neginf=0.0)
    alpha_values = np.nan_to_num(alpha_values, nan=0.0, posinf=0.0, neginf=0.0)
    covariance_psd = (
        np.asarray(covariance, dtype=np.float64)
        if covariance_is_psd
        else _psd_covariance(covariance)
    )
    if covariance_psd.shape != (alpha_values.size, alpha_values.size):
        raise ValueError("covariance must align with alpha.")
    covariance_psd = (covariance_psd + covariance_psd.T) / 2.0

    weights = cp.Variable(alpha_values.size)
    active = weights - benchmark
    turnover = 0.5 * cp.norm1(weights - previous)
    risk = cp.quad_form(active, cp.psd_wrap(covariance_psd))
    constraints = [
        weights >= 0,
        weights <= maximum_stock_weight,
        cp.sum(weights) == equity_exposure,
        turnover <= maximum_two_way_turnover,
        risk <= (annual_tracking_error_limit**2) / 252.0,
    ]

    if eligible is not None:
        eligible_mask = np.asarray(eligible, dtype=bool)
        if eligible_mask.shape != alpha_values.shape:
            raise ValueError("eligible must align with alpha.")
        excluded = np.flatnonzero(~eligible_mask)
        if excluded.size:
            constraints.append(weights[excluded] == 0.0)

    if tradable is not None:
        tradable_mask = np.asarray(tradable, dtype=bool)
        if tradable_mask.shape != alpha_values.shape:
            raise ValueError("tradable must align with alpha.")
        blocked = np.flatnonzero(~tradable_mask)
        if blocked.size:
            constraints.append(weights[blocked] == previous[blocked])

    if industry_matrix is not None:
        industries = np.asarray(industry_matrix, dtype=np.float64)
        if industries.ndim != 2 or industries.shape[1] != alpha_values.size:
            raise ValueError("industry_matrix must have shape (industries, assets).")
        deviation = industries @ active
        constraints.extend(
            [
                deviation <= maximum_industry_deviation,
                deviation >= -maximum_industry_deviation,
            ]
        )

    if style_matrix is not None:
        styles = np.asarray(style_matrix, dtype=np.float64)
        if styles.ndim != 2 or styles.shape[1] != alpha_values.size:
            raise ValueError("style_matrix must have shape (styles, assets).")
        deviation = styles @ active
        constraints.extend(
            [
                deviation <= maximum_style_deviation,
                deviation >= -maximum_style_deviation,
            ]
        )

    objective = cp.Maximize(
        alpha_values @ active - risk_aversion * risk - turnover_penalty * turnover
    )
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver="CLARABEL", verbose=False)
    except cp.error.SolverError:
        problem.solve(solver="SCS", verbose=False)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
        raise RuntimeError(f"Portfolio optimization failed with status {problem.status}.")

    result = np.asarray(weights.value, dtype=np.float64)
    result[np.abs(result) < 1e-10] = 0.0
    active_result = result - benchmark
    annual_te = math.sqrt(max(float(active_result @ covariance_psd @ active_result), 0.0) * 252.0)
    return OptimizationResult(
        weights=result,
        status=str(problem.status),
        objective_value=float(problem.value),
        two_way_turnover=0.5 * float(np.abs(result - previous).sum()),
        annualized_tracking_error=annual_te,
    )
