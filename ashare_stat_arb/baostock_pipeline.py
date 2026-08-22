from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ashare_stat_arb.baostock import BaoStockProvider
from ashare_stat_arb.panel import DailyPanel


ProgressCallback = Callable[[int, int, str], None]


def _safe_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.replace("", np.nan), errors="coerce")


def _cache_path(
    cache_directory: Path,
    symbol: str,
    start_date: str,
    end_date: str,
    adjustment: str,
) -> Path:
    safe_symbol = symbol.replace(".", "_")
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")
    return cache_directory / adjustment / f"{safe_symbol}_{start}_{end}.parquet"


def _history(
    provider: BaoStockProvider,
    symbol: str,
    start_date: str,
    end_date: str,
    adjustment: str,
    cache_directory: Path,
) -> pd.DataFrame:
    path = _cache_path(cache_directory, symbol, start_date, end_date, adjustment)
    if path.exists():
        return pd.read_parquet(path)
    frame = provider.daily_history(
        symbol,
        start_date,
        end_date,
        adjustment=adjustment,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def _limit_rate(symbol: str, dates: pd.DatetimeIndex, is_st: np.ndarray) -> np.ndarray:
    rate = np.full(dates.size, 0.10, dtype=np.float64)
    code = symbol.split(".")[-1]
    if symbol.startswith("sh.") and code.startswith(("688", "689")):
        rate[:] = 0.20
    elif symbol.startswith("sz.") and code.startswith(("300", "301")):
        rate[dates >= pd.Timestamp("2020-08-24")] = 0.20
    elif symbol.startswith("bj."):
        rate[:] = 0.30
    main_board = rate == 0.10
    rate[is_st & main_board] = 0.05
    return rate


def _round_price(values: np.ndarray) -> np.ndarray:
    return np.floor(values * 100.0 + 0.5) / 100.0


def _prepare_history(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=calendar)
    values = frame.copy()
    values["date"] = pd.to_datetime(values["date"]).dt.normalize()
    values = values.drop_duplicates("date", keep="last").set_index("date")
    for field in (
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "turn",
        "tradestatus",
        "pctChg",
        "isST",
    ):
        if field in values:
            values[field] = _safe_numeric(values[field])
    return values.reindex(calendar)


def build_csi500_panel_baostock(
    provider: BaoStockProvider,
    *,
    start_date: str,
    end_date: str,
    cache_directory: str | Path = "ashare_stat_arb/data/baostock_cache",
    history_buffer_days: int = 400,
    max_symbols: int | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[DailyPanel, dict[str, object]]:
    """Build a free-data CSI 500 panel with approximate float-cap weights."""

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start >= end:
        raise ValueError("start_date must precede end_date.")
    query_start = start - timedelta(days=history_buffer_days)
    complete_calendar = pd.DatetimeIndex(
        pd.to_datetime(
            provider.trading_days(
                query_start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            )
        )
    ).normalize()
    dates = complete_calendar[(complete_calendar >= start) & (complete_calendar <= end)]
    if dates.size < 2:
        raise RuntimeError("BaoStock returned fewer than two trading days.")

    snapshot_members: dict[pd.Period, tuple[str, ...]] = {}
    snapshot_dates: dict[pd.Period, pd.Timestamp] = {}
    panel_months = dates.to_period("M")
    for month in panel_months.unique():
        first_day = dates[panel_months == month][0]
        prior_days = complete_calendar[complete_calendar < first_day]
        if prior_days.size == 0:
            raise RuntimeError(f"No prior trading day is available for {first_day.date()}.")
        as_of = prior_days[-1]
        snapshot_dates[month] = as_of
        snapshot_members[month] = provider.index_members(as_of.strftime("%Y-%m-%d"))

    full_historical_union = {
        symbol for members in snapshot_members.values() for symbol in members
    }
    if max_symbols is not None:
        if max_symbols < 67:
            raise ValueError("max_symbols must be at least 67 for the 1.5% stock cap.")
        snapshot_members = {
            month: tuple(sorted(members)[:max_symbols])
            for month, members in snapshot_members.items()
        }
    all_symbols = {symbol for members in snapshot_members.values() for symbol in members}
    symbols = tuple(sorted(all_symbols))
    time_count = dates.size
    asset_count = len(symbols)
    shape = (time_count, asset_count)
    adjusted_close = np.full(shape, np.nan)
    open_price = np.full(shape, np.nan)
    close_price = np.full(shape, np.nan)
    high_limit = np.full(shape, np.nan)
    low_limit = np.full(shape, np.nan)
    volume = np.zeros(shape)
    money = np.zeros(shape)
    paused = np.ones(shape, dtype=bool)
    is_st = np.zeros(shape, dtype=bool)
    float_market_cap: dict[str, pd.Series] = {}
    cache = Path(cache_directory)

    query_start_text = query_start.strftime("%Y-%m-%d")
    end_text = end.strftime("%Y-%m-%d")
    for column, symbol in enumerate(symbols):
        raw = _prepare_history(
            _history(provider, symbol, query_start_text, end_text, "none", cache),
            complete_calendar,
        )
        raw_panel = raw.reindex(dates)

        status = raw_panel.get("tradestatus", pd.Series(index=dates, dtype=float))
        st_values = raw_panel.get("isST", pd.Series(index=dates, dtype=float)).ffill()
        paused[:, column] = status.fillna(0.0).to_numpy() != 1.0
        is_st[:, column] = st_values.fillna(0.0).to_numpy(dtype=float) == 1.0

        raw_close = raw_panel.get("close", pd.Series(index=dates, dtype=float)).ffill()
        raw_open = raw_panel.get("open", pd.Series(index=dates, dtype=float)).ffill()
        close_price[:, column] = raw_close.to_numpy(dtype=float)
        open_price[:, column] = raw_open.to_numpy(dtype=float)
        observed = raw.get(
            "close", pd.Series(index=complete_calendar, dtype=float)
        ).notna()
        total_returns = raw.get(
            "pctChg", pd.Series(index=complete_calendar, dtype=float)
        ) / 100.0
        adjusted_index = (1.0 + total_returns.fillna(0.0)).cumprod()
        if np.any(observed):
            first_observed = int(np.flatnonzero(observed.to_numpy())[0])
            last_observed = int(np.flatnonzero(observed.to_numpy())[-1])
            adjusted_index.iloc[:first_observed] = np.nan
            adjusted_index.iloc[last_observed + 1 :] = np.nan
        else:
            adjusted_index[:] = np.nan
        adjusted_close[:, column] = adjusted_index.reindex(dates).to_numpy(dtype=float)
        volume[:, column] = raw_panel.get(
            "volume", pd.Series(index=dates, dtype=float)
        ).fillna(0.0).to_numpy(dtype=float)
        money[:, column] = raw_panel.get(
            "amount", pd.Series(index=dates, dtype=float)
        ).fillna(0.0).to_numpy(dtype=float)

        preclose = raw_panel.get("preclose", raw_close.shift(1)).ffill().to_numpy(dtype=float)
        rate = _limit_rate(symbol, dates, is_st[:, column])
        high_limit[:, column] = _round_price(preclose * (1.0 + rate))
        low_limit[:, column] = _round_price(preclose * (1.0 - rate))

        raw_volume = raw.get("volume", pd.Series(index=complete_calendar, dtype=float))
        turnover = raw.get("turn", pd.Series(index=complete_calendar, dtype=float))
        shares = (raw_volume / (turnover / 100.0)).where(turnover > 0).ffill(limit=120)
        valuation_close = raw.get(
            "close", pd.Series(index=complete_calendar, dtype=float)
        ).ffill()
        float_market_cap[symbol] = valuation_close * shares
        if progress is not None:
            progress(column + 1, asset_count, symbol)

    member = np.zeros(shape, dtype=bool)
    benchmark_weight = np.zeros(shape)
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    weight_coverage: list[float] = []
    for row, month in enumerate(panel_months):
        members = tuple(
            symbol for symbol in snapshot_members[month] if symbol in symbol_index
        )
        as_of = snapshot_dates[month]
        columns = np.asarray([symbol_index[symbol] for symbol in members], dtype=int)
        member[row, columns] = True
        caps = np.asarray(
            [float_market_cap[symbol].get(as_of, np.nan) for symbol in members],
            dtype=float,
        )
        valid = np.isfinite(caps) & (caps > 0)
        coverage = float(valid.mean()) if valid.size else 0.0
        weight_coverage.append(coverage)
        if max_symbols is not None:
            weights = np.full(columns.size, 1.0 / columns.size)
        elif np.any(valid):
            caps[~valid] = float(np.median(caps[valid]))
            weights = caps / caps.sum()
        else:
            weights = np.full(columns.size, 1.0 / columns.size)
        benchmark_weight[row, columns] = weights

    panel = DailyPanel(
        dates=dates.to_numpy(dtype="datetime64[D]"),
        symbols=symbols,
        adjusted_close=adjusted_close,
        open_price=open_price,
        close_price=close_price,
        high_limit=high_limit,
        low_limit=low_limit,
        volume=volume,
        money=money,
        paused=paused,
        is_st=is_st,
        member=member,
        benchmark_weight=benchmark_weight,
    )
    metadata: dict[str, object] = {
        "source": "BaoStock 0.9.3 anonymous API",
        "source_cost": "free",
        "universe_method": "monthly historical CSI 500 members as of prior trading day",
        "benchmark_weight_method": (
            "approximate float market cap from volume / turnover"
            if max_symbols is None
            else "equal weight within deterministic point-in-time pilot subset"
        ),
        "official_index_weights": False,
        "signal_return_method": "BaoStock pctChg, verified against post-adjusted close returns",
        "limit_price_method": "computed from prior close, board, date, and ST status",
        "historical_constituent_union": len(full_historical_union),
        "downloaded_symbols": len(symbols),
        "symbol_selection": (
            "full historical constituent union"
            if max_symbols is None
            else (
                f"first {max_symbols} sorted constituents at each point-in-time "
                "snapshot; no future membership used; engineering pilot only"
            )
        ),
        "minimum_weight_input_coverage": min(weight_coverage, default=0.0),
        "mean_weight_input_coverage": float(np.mean(weight_coverage)) if weight_coverage else 0.0,
        "query_start": query_start_text,
        "panel_start": start.strftime("%Y-%m-%d"),
        "panel_end": end_text,
    }
    return panel, metadata
