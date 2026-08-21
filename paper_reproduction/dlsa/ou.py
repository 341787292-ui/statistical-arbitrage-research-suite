from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class OUFit:
    intercept: float
    autoregressive_coefficient: float
    kappa: float
    long_run_mean: float
    stationary_std: float
    r_squared: float
    latest_value: float
    valid: bool

    @property
    def standardized_deviation(self) -> float:
        if not self.valid or self.stationary_std <= 0:
            return math.nan
        return (self.latest_value - self.long_run_mean) / self.stationary_std


def fit_ou(cumulative_residual: np.ndarray) -> OUFit:
    """Fit the paper's discretized OU/AR(1) model to one residual window."""

    values = np.asarray(cumulative_residual, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
        return _invalid_fit(values)

    x = values[:-1]
    y = values[1:]
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_variance = float(np.mean((x - x_mean) ** 2))
    y_variance = float(np.mean((y - y_mean) ** 2))
    if x_variance <= np.finfo(np.float64).eps or y_variance <= 0:
        return _invalid_fit(values)

    covariance = float(np.mean((x - x_mean) * (y - y_mean)))
    b = covariance / x_variance
    intercept = y_mean - b * x_mean
    residual = y - (intercept + b * x)
    innovation_variance = float(np.mean(residual**2))
    r_squared = covariance**2 / (x_variance * y_variance)
    valid = 0 < b < 1 and innovation_variance > 0
    if not valid:
        return OUFit(
            intercept=intercept,
            autoregressive_coefficient=b,
            kappa=math.nan,
            long_run_mean=math.nan,
            stationary_std=math.nan,
            r_squared=r_squared,
            latest_value=float(values[-1]),
            valid=False,
        )

    return OUFit(
        intercept=intercept,
        autoregressive_coefficient=b,
        kappa=-math.log(b),
        long_run_mean=intercept / (1.0 - b),
        stationary_std=math.sqrt(innovation_variance / (1.0 - b**2)),
        r_squared=r_squared,
        latest_value=float(values[-1]),
        valid=True,
    )


def ou_threshold_weight(
    fit: OUFit,
    *,
    entry_threshold: float = 1.25,
    min_r_squared: float = 0.25,
) -> float:
    if not fit.valid or fit.r_squared <= min_r_squared:
        return 0.0
    deviation = fit.standardized_deviation
    if deviation > entry_threshold:
        return -1.0
    if deviation < -entry_threshold:
        return 1.0
    return 0.0


def _invalid_fit(values: np.ndarray) -> OUFit:
    latest = float(values[-1]) if values.size and np.isfinite(values[-1]) else math.nan
    return OUFit(
        intercept=math.nan,
        autoregressive_coefficient=math.nan,
        kappa=math.nan,
        long_run_mean=math.nan,
        stationary_std=math.nan,
        r_squared=math.nan,
        latest_value=latest,
        valid=False,
    )

