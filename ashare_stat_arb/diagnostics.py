from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.stats import rankdata

from ashare_stat_arb.panel import DailyPanel


@dataclass(frozen=True)
class RankICSummary:
    horizon: int
    mean_rank_ic: float
    median_rank_ic: float
    rank_ic_std: float
    annualized_icir: float
    positive_share: float
    observations: int
    average_cross_section: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class SignalCoverage:
    member_observations: int
    finite_signal_rate: float
    nonzero_signal_rate: float
    days_with_nonzero_signal_rate: float
    mean_absolute_signal: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def forward_open_returns(panel: DailyPanel, horizon: int) -> np.ndarray:
    """Return open-to-open total returns available after a close-time signal.

    A signal formed at close ``t`` can first execute at open ``t+1``. A
    horizon of one therefore measures open ``t+1`` to open ``t+2``.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    adjusted_open = panel.adjusted_open_prices()
    time_count = panel.shape[0]
    result = np.full(panel.shape, np.nan, dtype=np.float64)
    usable = time_count - horizon - 1
    if usable <= 0:
        return result
    start_prices = adjusted_open[1 : 1 + usable]
    end_prices = adjusted_open[1 + horizon : 1 + horizon + usable]
    valid = np.isfinite(start_prices) & np.isfinite(end_prices) & (start_prices > 0)
    result[:usable][valid] = end_prices[valid] / start_prices[valid] - 1.0
    return result


def summarize_rank_ic(
    stock_alpha: np.ndarray,
    forward_returns: np.ndarray,
    member: np.ndarray,
    *,
    horizon: int,
    start: int = 0,
    min_assets: int = 20,
) -> RankICSummary:
    alpha = np.asarray(stock_alpha, dtype=np.float64)
    future = np.asarray(forward_returns, dtype=np.float64)
    membership = np.asarray(member, dtype=bool)
    if alpha.shape != future.shape or alpha.shape != membership.shape:
        raise ValueError("stock_alpha, forward_returns, and member must align.")
    if min_assets < 3:
        raise ValueError("min_assets must be at least three.")

    daily_ic: list[float] = []
    cross_section_sizes: list[int] = []
    for row in range(max(start, 0), alpha.shape[0]):
        valid = membership[row] & np.isfinite(alpha[row]) & np.isfinite(future[row])
        count = int(valid.sum())
        if count < min_assets:
            continue
        alpha_values = alpha[row, valid]
        return_values = future[row, valid]
        if np.ptp(alpha_values) <= np.finfo(np.float64).eps:
            continue
        if np.ptp(return_values) <= np.finfo(np.float64).eps:
            continue
        alpha_rank = rankdata(alpha_values, method="average")
        return_rank = rankdata(return_values, method="average")
        correlation = float(np.corrcoef(alpha_rank, return_rank)[0, 1])
        if np.isfinite(correlation):
            daily_ic.append(correlation)
            cross_section_sizes.append(count)

    values = np.asarray(daily_ic, dtype=np.float64)
    if values.size == 0:
        return RankICSummary(horizon, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
    standard_deviation = float(values.std(ddof=0))
    return RankICSummary(
        horizon=horizon,
        mean_rank_ic=float(values.mean()),
        median_rank_ic=float(np.median(values)),
        rank_ic_std=standard_deviation,
        annualized_icir=(
            float(values.mean() / standard_deviation * math.sqrt(252.0))
            if standard_deviation > 0
            else 0.0
        ),
        positive_share=float((values > 0).mean()),
        observations=int(values.size),
        average_cross_section=float(np.mean(cross_section_sizes)),
    )


def forward_rank_ic_by_horizon(
    stock_alpha: np.ndarray,
    panel: DailyPanel,
    *,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    start: int = 0,
    min_assets: int = 20,
) -> dict[str, dict[str, float | int]]:
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must contain positive integers.")
    return {
        str(horizon): summarize_rank_ic(
            stock_alpha,
            forward_open_returns(panel, horizon),
            panel.member,
            horizon=horizon,
            start=start,
            min_assets=min_assets,
        ).to_dict()
        for horizon in horizons
    }


def summarize_signal_coverage(
    stock_alpha: np.ndarray,
    member: np.ndarray,
    *,
    start: int = 0,
    zero_tolerance: float = 1e-12,
) -> SignalCoverage:
    alpha = np.asarray(stock_alpha, dtype=np.float64)
    membership = np.asarray(member, dtype=bool)
    if alpha.shape != membership.shape:
        raise ValueError("stock_alpha and member must align.")
    selected_alpha = alpha[max(start, 0) :]
    selected_member = membership[max(start, 0) :]
    count = int(selected_member.sum())
    finite = selected_member & np.isfinite(selected_alpha)
    nonzero = finite & (np.abs(selected_alpha) > zero_tolerance)
    member_days = selected_member.any(axis=1)
    days_with_signal = nonzero.any(axis=1)
    finite_values = selected_alpha[finite]
    return SignalCoverage(
        member_observations=count,
        finite_signal_rate=float(finite.sum() / count) if count else 0.0,
        nonzero_signal_rate=float(nonzero.sum() / count) if count else 0.0,
        days_with_nonzero_signal_rate=(
            float(days_with_signal[member_days].mean()) if np.any(member_days) else 0.0
        ),
        mean_absolute_signal=(
            float(np.mean(np.abs(finite_values))) if finite_values.size else 0.0
        ),
    )
