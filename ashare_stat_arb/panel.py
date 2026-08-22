from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DailyPanel:
    """Aligned point-in-time inputs for the A-share research pipeline."""

    dates: np.ndarray
    symbols: tuple[str, ...]
    adjusted_close: np.ndarray
    open_price: np.ndarray
    close_price: np.ndarray
    high_limit: np.ndarray
    low_limit: np.ndarray
    volume: np.ndarray
    money: np.ndarray
    paused: np.ndarray
    is_st: np.ndarray
    member: np.ndarray
    benchmark_weight: np.ndarray

    def __post_init__(self) -> None:
        dates = np.asarray(self.dates, dtype="datetime64[D]")
        object.__setattr__(self, "dates", dates)
        if dates.ndim != 1 or dates.size < 2:
            raise ValueError("dates must contain at least two trading days.")
        if np.any(dates[1:] <= dates[:-1]):
            raise ValueError("dates must be unique and strictly increasing.")
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be nonempty and unique.")

        shape = (dates.size, len(self.symbols))
        float_fields = (
            "adjusted_close",
            "open_price",
            "close_price",
            "high_limit",
            "low_limit",
            "volume",
            "money",
            "benchmark_weight",
        )
        bool_fields = ("paused", "is_st", "member")
        for name in float_fields:
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}.")
            object.__setattr__(self, name, value)
        for name in bool_fields:
            value = np.asarray(getattr(self, name), dtype=bool)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}.")
            object.__setattr__(self, name, value)

    @property
    def shape(self) -> tuple[int, int]:
        return self.adjusted_close.shape

    def adjusted_returns(self) -> np.ndarray:
        returns = np.full(self.shape, np.nan, dtype=np.float64)
        previous = self.adjusted_close[:-1]
        current = self.adjusted_close[1:]
        valid = np.isfinite(previous) & np.isfinite(current) & (previous > 0)
        returns[1:][valid] = current[valid] / previous[valid] - 1.0
        return returns

    def next_open_returns(self) -> np.ndarray:
        """Return close-to-next-open returns aligned to the decision date."""

        returns = np.full(self.shape, np.nan, dtype=np.float64)
        current_close = self.close_price[:-1]
        next_open = self.open_price[1:]
        valid = np.isfinite(current_close) & np.isfinite(next_open) & (current_close > 0)
        returns[:-1][valid] = next_open[valid] / current_close[valid] - 1.0
        return returns

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.dates.astype("datetime64[D]").astype("int64").tobytes())
        digest.update("\n".join(self.symbols).encode("utf-8"))
        for name in (
            "adjusted_close",
            "open_price",
            "close_price",
            "high_limit",
            "low_limit",
            "volume",
            "money",
            "paused",
            "is_st",
            "member",
            "benchmark_weight",
        ):
            digest.update(np.ascontiguousarray(getattr(self, name)).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class PanelAudit:
    start_date: str
    end_date: str
    trading_days: int
    symbols: int
    member_observations: int
    missing_adjusted_close_rate: float
    missing_open_rate: float
    paused_rate: float
    st_rate: float
    missing_limit_rate: float
    invalid_weight_days: int
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, indent=2, sort_keys=True)


def audit_panel(panel: DailyPanel) -> PanelAudit:
    member = panel.member
    member_count = int(member.sum())

    def missing_rate(values: np.ndarray) -> float:
        if member_count == 0:
            return 0.0
        return float((member & ~np.isfinite(values)).sum() / member_count)

    weight_sums = np.where(member, panel.benchmark_weight, 0.0).sum(axis=1)
    rows_with_members = member.any(axis=1)
    invalid_weights = rows_with_members & (
        ~np.isfinite(weight_sums) | (np.abs(weight_sums - 1.0) > 1e-6)
    )
    missing_limits = member & (
        ~np.isfinite(panel.high_limit) | ~np.isfinite(panel.low_limit)
    )
    return PanelAudit(
        start_date=str(panel.dates[0]),
        end_date=str(panel.dates[-1]),
        trading_days=panel.shape[0],
        symbols=panel.shape[1],
        member_observations=member_count,
        missing_adjusted_close_rate=missing_rate(panel.adjusted_close),
        missing_open_rate=missing_rate(panel.open_price),
        paused_rate=float((member & panel.paused).sum() / member_count) if member_count else 0.0,
        st_rate=float((member & panel.is_st).sum() / member_count) if member_count else 0.0,
        missing_limit_rate=float(missing_limits.sum() / member_count) if member_count else 0.0,
        invalid_weight_days=int(invalid_weights.sum()),
        fingerprint=panel.fingerprint(),
    )
