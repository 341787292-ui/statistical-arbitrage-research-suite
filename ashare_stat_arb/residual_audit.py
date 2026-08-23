from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ashare_stat_arb.signals import MonthlyPCASignalResult


@dataclass(frozen=True)
class ResidualContinuityAudit:
    start_date: str
    end_date: str
    calendar_days_after_first_refit: int
    model_days: int
    days_without_residual_model: int
    model_day_rate: float
    refit_count: int
    member_residual_coverage: float
    median_daily_residual_coverage: float
    minimum_daily_residual_coverage: float
    median_active_count: float
    minimum_active_count: int
    ou_candidate_windows: int
    single_model_ou_window_rate: float
    cross_model_ou_window_rate: float
    average_models_per_ou_window: float
    maximum_models_per_ou_window: int
    median_refit_universe_jaccard: float
    mean_absolute_residual_on_refit_days: float
    mean_absolute_residual_on_other_days: float
    refit_residual_scale_ratio: float
    mean_alpha_change_on_refit_days: float
    mean_alpha_change_on_other_days: float
    refit_alpha_change_ratio: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def audit_residual_continuity(
    signal: MonthlyPCASignalResult,
    dates: np.ndarray,
    member: np.ndarray,
    *,
    lookback: int,
) -> ResidualContinuityAudit:
    """Measure whether OU histories cross changing monthly PCA definitions."""

    residuals = np.asarray(signal.residual_returns, dtype=np.float64)
    alpha = np.asarray(signal.stock_alpha, dtype=np.float64)
    trading_dates = np.asarray(dates, dtype="datetime64[D]")
    membership = np.asarray(member, dtype=bool)
    if residuals.shape != alpha.shape or residuals.shape != membership.shape:
        raise ValueError("signal arrays and member must align as (time, assets).")
    if trading_dates.shape != (residuals.shape[0],):
        raise ValueError("dates must align with the signal time dimension.")
    if lookback < 3:
        raise ValueError("lookback must be at least three observations.")

    date_to_index = {str(date): index for index, date in enumerate(trading_dates)}
    refit_indices = np.asarray(
        [date_to_index[date] for date in signal.refit_dates if date in date_to_index],
        dtype=np.int64,
    )
    if refit_indices.size == 0:
        raise ValueError("signal contains no refit dates to audit.")

    model_version = np.full(residuals.shape[0], -1, dtype=np.int64)
    for version, start in enumerate(refit_indices):
        end = refit_indices[version + 1] if version + 1 < refit_indices.size else residuals.shape[0]
        model_version[start:end] = version

    model_days = np.asarray(signal.active_count) > 0
    finite_member_residual = membership & np.isfinite(residuals)
    member_count = membership.sum(axis=1)
    residual_count = finite_member_residual.sum(axis=1)
    selected_days = model_days & (member_count > 0)
    daily_coverage = residual_count[selected_days] / member_count[selected_days]
    selected_member_count = int(member_count[selected_days].sum())
    selected_residual_count = int(residual_count[selected_days].sum())

    versions_per_window: list[int] = []
    for row in range(lookback - 1, residuals.shape[0]):
        history = residuals[row - lookback + 1 : row + 1]
        usable = membership[row] & np.all(np.isfinite(history), axis=0)
        if not np.any(usable):
            continue
        versions = np.unique(model_version[row - lookback + 1 : row + 1])
        versions = versions[versions >= 0]
        if versions.size:
            versions_per_window.append(int(versions.size))

    refit_mask = np.zeros(residuals.shape[0], dtype=bool)
    refit_mask[refit_indices] = True
    valid_refit_days = refit_mask & model_days
    valid_other_days = ~refit_mask & model_days

    daily_residual_scale = np.full(residuals.shape[0], np.nan, dtype=np.float64)
    for row in np.flatnonzero(model_days):
        values = residuals[row, finite_member_residual[row]]
        if values.size:
            daily_residual_scale[row] = float(np.mean(np.abs(values)))

    alpha_change = np.full(residuals.shape[0], np.nan, dtype=np.float64)
    for row in range(1, residuals.shape[0]):
        shared = membership[row] & membership[row - 1]
        if np.any(shared):
            alpha_change[row] = float(
                np.mean(np.abs(alpha[row, shared] - alpha[row - 1, shared]))
            )

    universe_jaccard: list[float] = []
    for row in refit_indices[1:]:
        current = finite_member_residual[row]
        previous = finite_member_residual[row - 1]
        union = current | previous
        if np.any(union):
            universe_jaccard.append(float((current & previous).sum() / union.sum()))

    version_counts = np.asarray(versions_per_window, dtype=np.float64)
    refit_residual = _finite_mean(daily_residual_scale[valid_refit_days])
    other_residual = _finite_mean(daily_residual_scale[valid_other_days])
    refit_alpha = _finite_mean(alpha_change[valid_refit_days])
    other_alpha = _finite_mean(alpha_change[valid_other_days])
    first_model_row = int(np.flatnonzero(model_version >= 0)[0])
    calendar_days = residuals.shape[0] - first_model_row
    model_day_count = int(model_days[first_model_row:].sum())
    return ResidualContinuityAudit(
        start_date=str(trading_dates[first_model_row]),
        end_date=str(trading_dates[-1]),
        calendar_days_after_first_refit=calendar_days,
        model_days=model_day_count,
        days_without_residual_model=calendar_days - model_day_count,
        model_day_rate=model_day_count / calendar_days if calendar_days else 0.0,
        refit_count=int(refit_indices.size),
        member_residual_coverage=(
            selected_residual_count / selected_member_count
            if selected_member_count
            else 0.0
        ),
        median_daily_residual_coverage=(
            float(np.median(daily_coverage)) if daily_coverage.size else 0.0
        ),
        minimum_daily_residual_coverage=(
            float(np.min(daily_coverage)) if daily_coverage.size else 0.0
        ),
        median_active_count=(
            float(np.median(np.asarray(signal.active_count)[model_days]))
            if np.any(model_days)
            else 0.0
        ),
        minimum_active_count=(
            int(np.min(np.asarray(signal.active_count)[model_days]))
            if np.any(model_days)
            else 0
        ),
        ou_candidate_windows=int(version_counts.size),
        single_model_ou_window_rate=(
            float((version_counts == 1).mean()) if version_counts.size else 0.0
        ),
        cross_model_ou_window_rate=(
            float((version_counts > 1).mean()) if version_counts.size else 0.0
        ),
        average_models_per_ou_window=(
            float(version_counts.mean()) if version_counts.size else 0.0
        ),
        maximum_models_per_ou_window=(
            int(version_counts.max()) if version_counts.size else 0
        ),
        median_refit_universe_jaccard=(
            float(np.median(universe_jaccard)) if universe_jaccard else 0.0
        ),
        mean_absolute_residual_on_refit_days=refit_residual,
        mean_absolute_residual_on_other_days=other_residual,
        refit_residual_scale_ratio=_safe_ratio(refit_residual, other_residual),
        mean_alpha_change_on_refit_days=refit_alpha,
        mean_alpha_change_on_other_days=other_alpha,
        refit_alpha_change_ratio=_safe_ratio(refit_alpha, other_alpha),
    )


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0
