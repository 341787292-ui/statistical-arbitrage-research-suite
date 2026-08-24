from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class LagAutocorrelation:
    lag: int
    asset_count: int
    mean_autocorrelation: float
    median_autocorrelation: float
    negative_asset_share: float


@dataclass(frozen=True)
class ReversalPeriodResult:
    period: str
    start_date: str
    end_date: str
    rank_ic_days: int
    mean_rank_ic: float
    median_rank_ic: float
    positive_rank_ic_share: float
    pooled_correlation: float


@dataclass(frozen=True)
class ReversalHorizonResult:
    horizon: int
    overall: ReversalPeriodResult
    development: ReversalPeriodResult
    validation: ReversalPeriodResult
    cross_sectional_gate_passed: bool
    stable_gate_passed: bool


@dataclass(frozen=True)
class ResidualPredictabilityAudit:
    start_date: str
    end_date: str
    minimum_rank_ic: float
    minimum_positive_share: float
    minimum_period_days: int
    required_stable_horizons: int
    lag_autocorrelation: tuple[LagAutocorrelation, ...]
    reversal_horizons: tuple[ReversalHorizonResult, ...]
    cross_sectional_stable_horizons: tuple[int, ...]
    stable_horizons: tuple[int, ...]
    cross_sectional_reversal_evidence_passed: bool
    broad_reversal_evidence_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_residual_predictability(
    residual_returns: np.ndarray,
    dates: np.ndarray,
    member: np.ndarray,
    *,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    development_start: str,
    development_end: str,
    validation_start: str,
    validation_end: str,
    minimum_rank_ic: float = 0.015,
    minimum_positive_share: float = 0.50,
    minimum_period_days: int = 60,
    required_stable_horizons: int = 2,
    minimum_cross_section: int = 20,
) -> ResidualPredictabilityAudit:
    """Describe residual predictability without a trading policy or fitted threshold."""

    residuals = np.asarray(residual_returns, dtype=np.float64)
    trading_dates = np.asarray(dates, dtype="datetime64[D]")
    membership = np.asarray(member, dtype=bool)
    if residuals.ndim != 2 or membership.shape != residuals.shape:
        raise ValueError("residual_returns and member must align as (time, assets).")
    if trading_dates.shape != (residuals.shape[0],):
        raise ValueError("dates must align with residual_returns.")
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive integers.")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique.")
    if required_stable_horizons < 1:
        raise ValueError("required_stable_horizons must be positive.")

    finite_rows = np.flatnonzero(np.any(np.isfinite(residuals) & membership, axis=1))
    if finite_rows.size == 0:
        raise ValueError("no finite member residuals are available.")

    lag_results = tuple(
        _lag_autocorrelation(residuals, membership, lag)
        for lag in horizons
    )
    horizon_results: list[ReversalHorizonResult] = []
    cross_sectional_stable_horizons: list[int] = []
    stable_horizons: list[int] = []
    period_masks = {
        "overall": np.ones(trading_dates.shape[0], dtype=bool),
        "development": (
            (trading_dates >= np.datetime64(development_start))
            & (trading_dates <= np.datetime64(development_end))
        ),
        "validation": (
            (trading_dates >= np.datetime64(validation_start))
            & (trading_dates <= np.datetime64(validation_end))
        ),
    }

    for horizon in horizons:
        decision_rows, daily_ic, pooled_score, pooled_outcome = _reversal_observations(
            residuals,
            membership,
            horizon,
            minimum_cross_section=minimum_cross_section,
        )
        period_results = {
            name: _period_result(
                name,
                trading_dates,
                decision_rows,
                daily_ic,
                pooled_score,
                pooled_outcome,
                mask,
            )
            for name, mask in period_masks.items()
        }
        development = period_results["development"]
        validation = period_results["validation"]
        cross_sectional_stable = all(
            result.rank_ic_days >= minimum_period_days
            and result.mean_rank_ic >= minimum_rank_ic
            and result.positive_rank_ic_share >= minimum_positive_share
            for result in (development, validation)
        )
        stable = cross_sectional_stable and all(
            result.pooled_correlation > 0.0
            for result in (development, validation)
        )
        if cross_sectional_stable:
            cross_sectional_stable_horizons.append(horizon)
        if stable:
            stable_horizons.append(horizon)
        horizon_results.append(
            ReversalHorizonResult(
                horizon=horizon,
                overall=period_results["overall"],
                development=development,
                validation=validation,
                cross_sectional_gate_passed=cross_sectional_stable,
                stable_gate_passed=stable,
            )
        )

    return ResidualPredictabilityAudit(
        start_date=str(trading_dates[finite_rows[0]]),
        end_date=str(trading_dates[finite_rows[-1]]),
        minimum_rank_ic=minimum_rank_ic,
        minimum_positive_share=minimum_positive_share,
        minimum_period_days=minimum_period_days,
        required_stable_horizons=required_stable_horizons,
        lag_autocorrelation=lag_results,
        reversal_horizons=tuple(horizon_results),
        cross_sectional_stable_horizons=tuple(cross_sectional_stable_horizons),
        stable_horizons=tuple(stable_horizons),
        cross_sectional_reversal_evidence_passed=(
            len(cross_sectional_stable_horizons) >= required_stable_horizons
        ),
        broad_reversal_evidence_passed=(
            len(stable_horizons) >= required_stable_horizons
        ),
    )


def _lag_autocorrelation(
    residuals: np.ndarray,
    membership: np.ndarray,
    lag: int,
) -> LagAutocorrelation:
    correlations: list[float] = []
    for column in range(residuals.shape[1]):
        valid = (
            membership[lag:, column]
            & membership[:-lag, column]
            & np.isfinite(residuals[lag:, column])
            & np.isfinite(residuals[:-lag, column])
        )
        if int(valid.sum()) < 30:
            continue
        current = residuals[lag:, column][valid]
        previous = residuals[:-lag, column][valid]
        correlation = _correlation(previous, current)
        if np.isfinite(correlation):
            correlations.append(correlation)
    values = np.asarray(correlations, dtype=np.float64)
    return LagAutocorrelation(
        lag=lag,
        asset_count=int(values.size),
        mean_autocorrelation=float(values.mean()) if values.size else 0.0,
        median_autocorrelation=float(np.median(values)) if values.size else 0.0,
        negative_asset_share=float((values < 0.0).mean()) if values.size else 0.0,
    )


def _reversal_observations(
    residuals: np.ndarray,
    membership: np.ndarray,
    horizon: int,
    *,
    minimum_cross_section: int,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    decision_rows: list[int] = []
    daily_ic: list[float] = []
    pooled_score: list[np.ndarray] = []
    pooled_outcome: list[np.ndarray] = []
    for row in range(horizon - 1, residuals.shape[0] - horizon):
        past = residuals[row - horizon + 1 : row + 1]
        future = residuals[row + 1 : row + horizon + 1]
        valid = (
            membership[row]
            & np.all(np.isfinite(past), axis=0)
            & np.all(np.isfinite(future), axis=0)
        )
        if int(valid.sum()) < minimum_cross_section:
            continue
        score = -past[:, valid].sum(axis=0)
        outcome = future[:, valid].sum(axis=0)
        rank_ic = _correlation(_rank(score), _rank(outcome))
        if not np.isfinite(rank_ic):
            continue
        decision_rows.append(row)
        daily_ic.append(rank_ic)
        pooled_score.append(score)
        pooled_outcome.append(outcome)
    return (
        np.asarray(decision_rows, dtype=np.int64),
        np.asarray(daily_ic, dtype=np.float64),
        tuple(pooled_score),
        tuple(pooled_outcome),
    )


def _period_result(
    period: str,
    dates: np.ndarray,
    decision_rows: np.ndarray,
    daily_ic: np.ndarray,
    pooled_score: tuple[np.ndarray, ...],
    pooled_outcome: tuple[np.ndarray, ...],
    mask: np.ndarray,
) -> ReversalPeriodResult:
    selected = mask[decision_rows] if decision_rows.size else np.zeros(0, dtype=bool)
    selected_rows = decision_rows[selected]
    selected_ic = daily_ic[selected]
    selected_score = [value for value, keep in zip(pooled_score, selected) if keep]
    selected_outcome = [value for value, keep in zip(pooled_outcome, selected) if keep]
    if selected_score:
        score = np.concatenate(selected_score)
        outcome = np.concatenate(selected_outcome)
        pooled_correlation = _correlation(score, outcome)
    else:
        pooled_correlation = 0.0
    return ReversalPeriodResult(
        period=period,
        start_date=str(dates[selected_rows[0]]) if selected_rows.size else "",
        end_date=str(dates[selected_rows[-1]]) if selected_rows.size else "",
        rank_ic_days=int(selected_ic.size),
        mean_rank_ic=float(selected_ic.mean()) if selected_ic.size else 0.0,
        median_rank_ic=float(np.median(selected_ic)) if selected_ic.size else 0.0,
        positive_rank_ic_share=(
            float((selected_ic > 0.0).mean()) if selected_ic.size else 0.0
        ),
        pooled_correlation=pooled_correlation,
    )


def _rank(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    order = np.argsort(data, kind="mergesort")
    ranks = np.empty(data.size, dtype=np.float64)
    ranks[order] = np.arange(data.size, dtype=np.float64)
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or y.shape != x.shape:
        return 0.0
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(np.sqrt((x_centered @ x_centered) * (y_centered @ y_centered)))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float((x_centered @ y_centered) / denominator)
