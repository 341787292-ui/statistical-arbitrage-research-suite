from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Protocol, Sequence


class JQDataAPI(Protocol):
    def auth(self, username: str, password: str) -> Any: ...

    def is_auth(self) -> bool: ...

    def get_trade_days(self, start_date: str, end_date: str) -> Any: ...

    def get_index_stocks(self, index_symbol: str, date: str) -> list[str]: ...

    def get_index_weights(self, index_symbol: str, date: str) -> Any: ...

    def get_price(self, security: Sequence[str], **kwargs: Any) -> Any: ...

    def get_extras(self, info: str, security_list: Sequence[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class JQDataCredentials:
    username: str
    password: str

    @classmethod
    def from_environment(cls) -> "JQDataCredentials":
        username = os.getenv("JQDATA_USERNAME", "").strip()
        password = os.getenv("JQDATA_PASSWORD", "").strip()
        if not username or not password:
            raise RuntimeError(
                "Set JQDATA_USERNAME and JQDATA_PASSWORD in the local environment."
            )
        return cls(username=username, password=password)


def load_jqdata_api() -> JQDataAPI:
    try:
        import jqdatasdk
    except ImportError as exc:
        raise RuntimeError(
            "jqdatasdk is not installed. Install ashare_stat_arb/requirements.txt."
        ) from exc
    return jqdatasdk


class JQDataProvider:
    """Thin, injectable wrapper around the official JQData SDK."""

    def __init__(self, api: JQDataAPI | None = None) -> None:
        self.api = api or load_jqdata_api()

    def authenticate(self, credentials: JQDataCredentials | None = None) -> None:
        credentials = credentials or JQDataCredentials.from_environment()
        self.api.auth(credentials.username, credentials.password)
        if not self.api.is_auth():
            raise RuntimeError("JQData authentication failed.")

    def trading_days(self, start_date: str, end_date: str) -> tuple[str, ...]:
        values = self.api.get_trade_days(start_date=start_date, end_date=end_date)
        return tuple(str(value)[:10] for value in values)

    def index_members(self, index_symbol: str, as_of_date: str) -> tuple[str, ...]:
        members = self.api.get_index_stocks(index_symbol, date=as_of_date)
        return tuple(sorted(str(symbol) for symbol in members))

    def index_weights(self, index_symbol: str, as_of_date: str) -> dict[str, float]:
        frame = self.api.get_index_weights(index_symbol, date=as_of_date)
        if frame is None or len(frame) == 0:
            raise RuntimeError(
                f"JQData returned no index weights for {index_symbol} on {as_of_date}."
            )
        if "code" in frame.columns:
            frame = frame.set_index("code")
        if "weight" not in frame.columns:
            raise RuntimeError("JQData index-weight response has no 'weight' column.")
        weights = {
            str(symbol): float(weight)
            for symbol, weight in frame["weight"].items()
            if float(weight) > 0
        }
        total = sum(weights.values())
        if total <= 0:
            raise RuntimeError("JQData index weights contain no positive mass.")
        return {symbol: weight / total for symbol, weight in weights.items()}

    def raw_daily_prices(
        self,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> Any:
        return self.api.get_price(
            list(symbols),
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            fields=(
                "open",
                "close",
                "volume",
                "money",
                "high_limit",
                "low_limit",
                "paused",
            ),
            skip_paused=False,
            fill_paused=False,
            fq=None,
            panel=False,
        )

    def post_adjusted_close(
        self,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> Any:
        return self.api.get_price(
            list(symbols),
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            fields=("close",),
            skip_paused=False,
            fill_paused=False,
            fq="post",
            panel=False,
        )

    def st_flags(
        self,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> Any:
        return self.api.get_extras(
            "is_st",
            list(symbols),
            start_date=start_date,
            end_date=end_date,
            df=True,
        )
