from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ashare_stat_arb.config import CostConfig, PortfolioConfig


@dataclass(frozen=True)
class Phase2Mapping:
    signal_horizon: int = 5
    decision_interval: int = 5
    maximum_two_way_turnover: float = 0.05

    def portfolio(self, baseline: PortfolioConfig) -> PortfolioConfig:
        return replace(
            baseline,
            target_two_way_turnover=self.maximum_two_way_turnover,
            maximum_two_way_turnover=self.maximum_two_way_turnover,
            turnover_scenarios=(self.maximum_two_way_turnover,),
        )


def conservative_round_trip_cost(costs: CostConfig) -> float:
    """Return pre-2023 variable cost per unit of two-way turnover."""

    symmetric_side = (
        costs.commission_rate + costs.transfer_fee_rate + costs.base_slippage
    )
    return 2.0 * symmetric_side + costs.stamp_duty_before_2023_08_28


PHASE2_MAPPING = Phase2Mapping()


def validate_development_panel(dates: np.ndarray) -> None:
    trading_dates = np.asarray(dates, dtype="datetime64[D]")
    if trading_dates.ndim != 1 or trading_dates.size < 2:
        raise ValueError("Phase 2 dates must be a one-dimensional trading calendar.")
    if np.max(trading_dates) >= np.datetime64("2023-01-01"):
        raise ValueError("Phase 2 development runs must not contain the sealed holdout.")
