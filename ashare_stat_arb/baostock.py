from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd


DAILY_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,"
    "turn,tradestatus,pctChg,isST"
)


class BaoStockResult(Protocol):
    error_code: str
    error_msg: str
    fields: list[str]

    def next(self) -> bool: ...

    def get_row_data(self) -> list[str]: ...


class BaoStockAPI(Protocol):
    def login(self, user_id: str = "anonymous", password: str = "123456") -> Any: ...

    def logout(self) -> Any: ...

    def query_trade_dates(self, start_date: str, end_date: str) -> BaoStockResult: ...

    def query_zz500_stocks(self, date: str = "") -> BaoStockResult: ...

    def query_history_k_data_plus(
        self,
        code: str,
        fields: str,
        *,
        start_date: str,
        end_date: str,
        frequency: str,
        adjustflag: str,
    ) -> BaoStockResult: ...


def load_baostock_api() -> BaoStockAPI:
    try:
        import baostock
    except ImportError as exc:
        raise RuntimeError(
            "baostock is not installed. Install ashare_stat_arb/requirements.txt."
        ) from exc
    return baostock


def _collect(result: BaoStockResult, operation: str) -> pd.DataFrame:
    if str(result.error_code) != "0":
        raise RuntimeError(f"BaoStock {operation} failed: {result.error_msg}")
    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


@dataclass
class BaoStockProvider:
    """Injectable wrapper around BaoStock's anonymous, free data API."""

    api: BaoStockAPI | None = None
    connected: bool = False

    def __post_init__(self) -> None:
        if self.api is None:
            self.api = load_baostock_api()

    def connect(self) -> None:
        response = self.api.login()
        if str(response.error_code) != "0":
            raise RuntimeError(f"BaoStock login failed: {response.error_msg}")
        self.connected = True

    def close(self) -> None:
        if self.connected:
            self.api.logout()
            self.connected = False

    def __enter__(self) -> "BaoStockProvider":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _require_connection(self) -> None:
        if not self.connected:
            raise RuntimeError("Call BaoStockProvider.connect() before querying data.")

    def trading_days(self, start_date: str, end_date: str) -> tuple[str, ...]:
        self._require_connection()
        frame = _collect(
            self.api.query_trade_dates(start_date=start_date, end_date=end_date),
            "trading calendar query",
        )
        if frame.empty:
            return ()
        open_days = frame.loc[frame["is_trading_day"] == "1", "calendar_date"]
        return tuple(open_days.astype(str))

    def index_members(self, as_of_date: str) -> tuple[str, ...]:
        self._require_connection()
        frame = _collect(
            self.api.query_zz500_stocks(date=as_of_date),
            "CSI 500 constituent query",
        )
        if frame.empty or "code" not in frame:
            raise RuntimeError(f"BaoStock returned no CSI 500 members for {as_of_date}.")
        return tuple(sorted(frame["code"].astype(str)))

    def daily_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        *,
        adjustment: str = "none",
    ) -> pd.DataFrame:
        self._require_connection()
        flags = {"post": "1", "pre": "2", "none": "3"}
        if adjustment not in flags:
            raise ValueError("adjustment must be one of: none, pre, post.")
        return _collect(
            self.api.query_history_k_data_plus(
                symbol,
                DAILY_FIELDS,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag=flags[adjustment],
            ),
            f"daily history query for {symbol}",
        )
