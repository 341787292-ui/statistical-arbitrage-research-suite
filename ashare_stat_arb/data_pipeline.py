from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ashare_stat_arb.jqdata import JQDataProvider
from ashare_stat_arb.panel import DailyPanel, audit_panel


RAW_FIELDS = (
    "open",
    "close",
    "volume",
    "money",
    "high_limit",
    "low_limit",
    "paused",
)


def _chunks(values: tuple[str, ...], size: int) -> Iterable[tuple[str, ...]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _long_price_frame(frame: object) -> pd.DataFrame:
    values = pd.DataFrame(frame).reset_index()
    rename = {}
    if "datetime" in values.columns and "time" not in values.columns:
        rename["datetime"] = "time"
    if "security" in values.columns and "code" not in values.columns:
        rename["security"] = "code"
    values = values.rename(columns=rename)
    if "time" not in values.columns or "code" not in values.columns:
        raise RuntimeError("Expected JQData panel=False output with time and code columns.")
    values["time"] = pd.to_datetime(values["time"]).dt.normalize()
    values["code"] = values["code"].astype(str)
    return values


def _wide(
    frame: pd.DataFrame,
    field: str,
    dates: pd.DatetimeIndex,
    symbols: tuple[str, ...],
) -> np.ndarray:
    if field not in frame.columns:
        raise RuntimeError(f"JQData price response has no '{field}' field.")
    matrix = frame.pivot_table(index="time", columns="code", values=field, aggfunc="last")
    return matrix.reindex(index=dates, columns=symbols).to_numpy(dtype=np.float64)


def _st_matrix(
    frame: object,
    dates: pd.DatetimeIndex,
    symbols: tuple[str, ...],
) -> np.ndarray:
    values = pd.DataFrame(frame).copy()
    values.index = pd.to_datetime(values.index).normalize()
    values.columns = values.columns.astype(str)
    return values.reindex(index=dates, columns=symbols).fillna(False).to_numpy(dtype=bool)


def build_csi500_panel(
    provider: JQDataProvider,
    *,
    start_date: str,
    end_date: str,
    index_symbol: str = "000905.XSHG",
    symbol_chunk_size: int = 100,
) -> DailyPanel:
    """Download a monthly point-in-time CSI 500 panel from JQData.

    Each calendar month's universe and benchmark weights are frozen using the
    last trading day before that month starts. Raw prices are kept for order
    simulation; post-adjusted closes are fetched separately for signals.
    """

    if symbol_chunk_size <= 0:
        raise ValueError("symbol_chunk_size must be positive.")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start >= end:
        raise ValueError("start_date must precede end_date.")

    calendar_start = (start - timedelta(days=40)).strftime("%Y-%m-%d")
    complete_calendar = pd.DatetimeIndex(
        pd.to_datetime(provider.trading_days(calendar_start, end.strftime("%Y-%m-%d")))
    ).normalize()
    dates = complete_calendar[(complete_calendar >= start) & (complete_calendar <= end)]
    if dates.size < 2:
        raise RuntimeError("JQData returned fewer than two trading days.")

    snapshots: dict[pd.Period, dict[str, float]] = {}
    for month in dates.to_period("M").unique():
        first_day = dates[dates.to_period("M") == month][0]
        prior_days = complete_calendar[complete_calendar < first_day]
        if prior_days.size == 0:
            raise RuntimeError(f"No prior trading day is available for {first_day.date()}.")
        as_of = prior_days[-1].strftime("%Y-%m-%d")
        snapshots[month] = provider.index_weights(index_symbol, as_of)

    symbols = tuple(sorted({symbol for weights in snapshots.values() for symbol in weights}))
    member = np.zeros((dates.size, len(symbols)), dtype=bool)
    benchmark_weight = np.zeros_like(member, dtype=np.float64)
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    months = dates.to_period("M")
    for row, month in enumerate(months):
        weights = snapshots[month]
        for symbol, weight in weights.items():
            column = symbol_index[symbol]
            member[row, column] = True
            benchmark_weight[row, column] = weight

    raw_frames: list[pd.DataFrame] = []
    adjusted_frames: list[pd.DataFrame] = []
    st_frames: list[pd.DataFrame] = []
    for chunk in _chunks(symbols, symbol_chunk_size):
        raw_frames.append(
            _long_price_frame(
                provider.raw_daily_prices(chunk, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            )
        )
        adjusted_frames.append(
            _long_price_frame(
                provider.post_adjusted_close(chunk, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            )
        )
        st = pd.DataFrame(
            provider.st_flags(chunk, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        )
        st_frames.append(st)

    raw = pd.concat(raw_frames, ignore_index=True)
    adjusted = pd.concat(adjusted_frames, ignore_index=True)
    st = pd.concat(st_frames, axis=1)
    return DailyPanel(
        dates=dates.to_numpy(dtype="datetime64[D]"),
        symbols=symbols,
        adjusted_close=_wide(adjusted, "close", dates, symbols),
        open_price=_wide(raw, "open", dates, symbols),
        close_price=_wide(raw, "close", dates, symbols),
        high_limit=_wide(raw, "high_limit", dates, symbols),
        low_limit=_wide(raw, "low_limit", dates, symbols),
        volume=_wide(raw, "volume", dates, symbols),
        money=_wide(raw, "money", dates, symbols),
        paused=_wide(raw, "paused", dates, symbols).astype(bool),
        is_st=_st_matrix(st, dates, symbols),
        member=member,
        benchmark_weight=benchmark_weight,
    )


def save_panel(
    panel: DailyPanel,
    destination: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    path = Path(destination)
    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dates=panel.dates,
        symbols=np.asarray(panel.symbols, dtype="U"),
        adjusted_close=panel.adjusted_close,
        open_price=panel.open_price,
        close_price=panel.close_price,
        high_limit=panel.high_limit,
        low_limit=panel.low_limit,
        volume=panel.volume,
        money=panel.money,
        paused=panel.paused,
        is_st=panel.is_st,
        member=panel.member,
        benchmark_weight=panel.benchmark_weight,
    )
    manifest_path = path.with_suffix(".manifest.json")
    manifest = {
        "schema_version": 1,
        "panel_file": path.name,
        "audit": asdict(audit_panel(panel)),
        "metadata": dict(metadata or {}),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path, manifest_path


def load_panel(source: str | Path) -> DailyPanel:
    with np.load(Path(source), allow_pickle=False) as data:
        return DailyPanel(
            dates=data["dates"],
            symbols=tuple(str(symbol) for symbol in data["symbols"]),
            adjusted_close=data["adjusted_close"],
            open_price=data["open_price"],
            close_price=data["close_price"],
            high_limit=data["high_limit"],
            low_limit=data["low_limit"],
            volume=data["volume"],
            money=data["money"],
            paused=data["paused"],
            is_st=data["is_st"],
            member=data["member"],
            benchmark_weight=data["benchmark_weight"],
        )
