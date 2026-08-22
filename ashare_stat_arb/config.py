from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResearchPeriods:
    raw_start: str = "2010-01-01"
    raw_end: str = "2025-12-31"
    warmup_start: str = "2010-01-01"
    warmup_end: str = "2014-12-31"
    development_start: str = "2015-01-01"
    development_end: str = "2019-12-31"
    validation_start: str = "2020-01-01"
    validation_end: str = "2022-12-31"
    holdout_start: str = "2023-01-01"
    holdout_end: str = "2025-12-31"


@dataclass(frozen=True)
class SignalConfig:
    factor_counts: tuple[int, ...] = (0, 1, 3, 5, 8, 10, 15)
    baseline_factor_count: int = 5
    covariance_window: int = 252
    loading_window: int = 60
    residual_lookback: int = 30
    ou_entry_threshold: float = 1.25
    ou_min_r_squared: float = 0.25
    minimum_history: int = 252
    minimum_history_coverage: float = 0.95


@dataclass(frozen=True)
class PortfolioConfig:
    benchmark: str = "000905.XSHG"
    target_equity_exposure: float = 0.99
    minimum_equity_exposure: float = 0.98
    maximum_equity_exposure: float = 1.00
    maximum_stock_weight: float = 0.015
    maximum_industry_deviation: float = 0.03
    maximum_style_deviation: float = 0.50
    central_tracking_error: float = 0.08
    tracking_error_scenarios: tuple[float, ...] = (0.06, 0.08, 0.10)
    target_two_way_turnover: float = 0.15
    maximum_two_way_turnover: float = 0.20
    turnover_scenarios: tuple[float, ...] = (0.10, 0.15, 0.20)
    adv_participation_limit: float = 0.05
    adv_scenarios: tuple[float, ...] = (0.01, 0.03, 0.05)


@dataclass(frozen=True)
class CostConfig:
    commission_rate: float = 0.00025
    transfer_fee_rate: float = 0.00001
    stamp_duty_before_2023_08_28: float = 0.001
    stamp_duty_from_2023_08_28: float = 0.0005
    base_slippage: float = 0.001
    slippage_scenarios: tuple[float, ...] = (0.0005, 0.001, 0.002)
    minimum_commission: float = 0.0


@dataclass(frozen=True)
class AdmissionConfig:
    minimum_rank_ic: float = 0.015
    minimum_icir: float = 0.50
    minimum_net_information_ratio: float = 1.00
    minimum_ir_uplift: float = 0.15
    strong_information_ratio: float = 1.50
    stretch_information_ratio: float = 2.00
    audit_trigger_information_ratio: float = 3.00
    maximum_active_drawdown: float = 0.12
    maximum_cost_share_of_gross_alpha: float = 0.50
    minimum_positive_rolling_12m_share: float = 0.60


@dataclass(frozen=True)
class AshareResearchConfig:
    product_name: str = "CSI500 residual-alpha index enhancement"
    universe: str = "point-in-time CSI 500 constituents"
    rebalance_frequency: str = "monthly"
    decision_time: str = "trading-day close"
    execution_time: str = "next-trading-day open"
    periods: ResearchPeriods = field(default_factory=ResearchPeriods)
    signal: SignalConfig = field(default_factory=SignalConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    admission: AdmissionConfig = field(default_factory=AdmissionConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = AshareResearchConfig()
