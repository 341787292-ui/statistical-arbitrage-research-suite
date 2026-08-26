from __future__ import annotations

from dataclasses import dataclass, replace

from ashare_stat_arb.config import PortfolioConfig


@dataclass(frozen=True)
class Phase3Mapping:
    signal_horizon: int = 5
    positive_entry: float = 0.80
    positive_exit: float = 0.60
    negative_entry: float = 0.20
    negative_exit: float = 0.40
    maximum_two_way_turnover: float = 0.05

    def portfolio(self, baseline: PortfolioConfig) -> PortfolioConfig:
        return replace(
            baseline,
            target_two_way_turnover=self.maximum_two_way_turnover,
            maximum_two_way_turnover=self.maximum_two_way_turnover,
            turnover_scenarios=(self.maximum_two_way_turnover,),
        )


PHASE3_MAPPING = Phase3Mapping()
