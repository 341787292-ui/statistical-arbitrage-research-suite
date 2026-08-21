from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping


@dataclass(frozen=True)
class FeeSchedule:
    """Effective-dated fee inputs supplied by the experiment manifest."""

    commission_rate: float = 0.0
    minimum_commission: float = 0.0
    stamp_duty_rate: float = 0.0
    transfer_fee_rate: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.commission_rate,
            self.minimum_commission,
            self.stamp_duty_rate,
            self.transfer_fee_rate,
        )
        if any(value < 0 or not math.isfinite(value) for value in values):
            raise ValueError("Fee inputs must be finite and nonnegative.")

    def calculate(self, notional: float, *, side: str) -> float:
        if notional <= 0:
            return 0.0
        commission = max(self.minimum_commission, notional * self.commission_rate)
        transfer_fee = notional * self.transfer_fee_rate
        stamp_duty = notional * self.stamp_duty_rate if side == "sell" else 0.0
        return commission + transfer_fee + stamp_duty


@dataclass(frozen=True)
class TradingStatus:
    suspended: bool = False
    at_upper_limit: bool = False
    at_lower_limit: bool = False


@dataclass(frozen=True)
class PortfolioState:
    cash: float
    holdings: dict[str, int] = field(default_factory=dict)
    sellable_shares: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cash < 0 or not math.isfinite(self.cash):
            raise ValueError("Cash must be finite and nonnegative.")
        for symbol, shares in self.holdings.items():
            if not isinstance(shares, int) or shares < 0:
                raise ValueError(f"Holding for {symbol} must be a nonnegative integer.")
        for symbol, shares in self.sellable_shares.items():
            held = self.holdings.get(symbol, 0)
            if not isinstance(shares, int) or shares < 0 or shares > held:
                raise ValueError(
                    f"Sellable shares for {symbol} must be between zero and holdings."
                )


@dataclass(frozen=True)
class ExecutionRecord:
    symbol: str
    side: str
    requested_shares: int
    filled_shares: int
    price: float
    fees: float
    reason: str | None = None

    @property
    def rejected_shares(self) -> int:
        return self.requested_shares - self.filled_shares


@dataclass(frozen=True)
class ExecutionReport:
    state: PortfolioState
    records: tuple[ExecutionRecord, ...]

    @property
    def total_fees(self) -> float:
        return float(sum(record.fees for record in self.records))


def begin_trading_day(state: PortfolioState) -> PortfolioState:
    """Promote all overnight holdings to sellable inventory."""

    return PortfolioState(
        cash=state.cash,
        holdings=dict(state.holdings),
        sellable_shares=dict(state.holdings),
    )


def _validate_targets(target_shares: Mapping[str, int]) -> dict[str, int]:
    validated: dict[str, int] = {}
    for symbol, shares in target_shares.items():
        if not isinstance(shares, int) or shares < 0:
            raise ValueError(
                f"Target shares for {symbol} must be a nonnegative integer."
            )
        validated[symbol] = shares
    return validated


def _price_for(symbol: str, prices: Mapping[str, float]) -> float:
    if symbol not in prices:
        raise ValueError(f"Missing execution price for {symbol}.")
    price = float(prices[symbol])
    if price <= 0 or not math.isfinite(price):
        raise ValueError(f"Execution price for {symbol} must be finite and positive.")
    return price


def _blocked_reason(status: TradingStatus, *, side: str) -> str | None:
    if status.suspended:
        return "suspended"
    if side == "buy" and status.at_upper_limit:
        return "upper_limit"
    if side == "sell" and status.at_lower_limit:
        return "lower_limit"
    return None


def _round_buy_quantity(shares: int, lot_size: int) -> int:
    if lot_size <= 0:
        raise ValueError("Lot size must be positive.")
    return shares - shares % lot_size


def _affordable_buy_quantity(
    requested: int,
    *,
    lot_size: int,
    price: float,
    cash: float,
    fees: FeeSchedule,
) -> int:
    quantity = _round_buy_quantity(requested, lot_size)
    while quantity > 0:
        notional = quantity * price
        if notional + fees.calculate(notional, side="buy") <= cash + 1e-9:
            return quantity
        quantity -= lot_size
    return 0


def execute_target_portfolio(
    state: PortfolioState,
    target_shares: Mapping[str, int],
    prices: Mapping[str, float],
    *,
    statuses: Mapping[str, TradingStatus] | None = None,
    fees: FeeSchedule | None = None,
    lot_sizes: Mapping[str, int] | None = None,
    default_lot_size: int = 100,
) -> ExecutionReport:
    """Move toward long-only stock targets under daily A-share constraints.

    Sells execute before buys. Shares purchased in this call increase holdings
    but not sellable inventory; ``begin_trading_day`` releases them on the next
    session. Limit-state handling is deliberately conservative for daily data.
    """

    targets = _validate_targets(target_shares)
    statuses = statuses or {}
    fee_schedule = fees or FeeSchedule()
    lot_sizes = lot_sizes or {}
    holdings = dict(state.holdings)
    sellable = dict(state.sellable_shares)
    cash = float(state.cash)
    records: list[ExecutionRecord] = []
    symbols = sorted(set(holdings) | set(targets))

    for symbol in symbols:
        current = holdings.get(symbol, 0)
        target = targets.get(symbol, 0)
        requested = max(current - target, 0)
        if requested == 0:
            continue
        price = _price_for(symbol, prices)
        status = statuses.get(symbol, TradingStatus())
        reason = _blocked_reason(status, side="sell")
        available = sellable.get(symbol, 0)
        filled = 0 if reason else min(requested, available)
        if reason is None and filled < requested:
            reason = "t_plus_one"
        notional = filled * price
        trade_fees = fee_schedule.calculate(notional, side="sell")
        cash += notional - trade_fees
        holdings[symbol] = current - filled
        sellable[symbol] = available - filled
        records.append(
            ExecutionRecord(
                symbol=symbol,
                side="sell",
                requested_shares=requested,
                filled_shares=filled,
                price=price,
                fees=trade_fees,
                reason=reason,
            )
        )

    for symbol in symbols:
        current = holdings.get(symbol, 0)
        target = targets.get(symbol, 0)
        requested = max(target - current, 0)
        if requested == 0:
            continue
        price = _price_for(symbol, prices)
        status = statuses.get(symbol, TradingStatus())
        reason = _blocked_reason(status, side="buy")
        lot_size = lot_sizes.get(symbol, default_lot_size)
        filled = 0
        if reason is None:
            filled = _affordable_buy_quantity(
                requested,
                lot_size=lot_size,
                price=price,
                cash=cash,
                fees=fee_schedule,
            )
            rounded_request = _round_buy_quantity(requested, lot_size)
            if filled < rounded_request:
                reason = "insufficient_cash"
            elif rounded_request < requested:
                reason = "lot_rounding"
        notional = filled * price
        trade_fees = fee_schedule.calculate(notional, side="buy")
        cash -= notional + trade_fees
        holdings[symbol] = current + filled
        sellable.setdefault(symbol, 0)
        records.append(
            ExecutionRecord(
                symbol=symbol,
                side="buy",
                requested_shares=requested,
                filled_shares=filled,
                price=price,
                fees=trade_fees,
                reason=reason,
            )
        )

    cleaned_holdings = {
        symbol: shares for symbol, shares in holdings.items() if shares != 0
    }
    cleaned_sellable = {
        symbol: shares
        for symbol, shares in sellable.items()
        if symbol in cleaned_holdings and shares != 0
    }
    next_state = PortfolioState(
        cash=max(cash, 0.0),
        holdings=cleaned_holdings,
        sellable_shares=cleaned_sellable,
    )
    return ExecutionReport(state=next_state, records=tuple(records))
