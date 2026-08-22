from __future__ import annotations

import unittest

import pandas as pd

from ashare_stat_arb.jqdata import JQDataCredentials, JQDataProvider


class FakeJQData:
    def __init__(self) -> None:
        self.authenticated = False
        self.calls: list[tuple[str, object]] = []

    def auth(self, username: str, password: str) -> None:
        self.calls.append(("auth", (username, password)))
        self.authenticated = username == "user" and password == "pass"

    def is_auth(self) -> bool:
        return self.authenticated

    def get_trade_days(self, start_date: str, end_date: str) -> list[str]:
        self.calls.append(("calendar", (start_date, end_date)))
        return ["2022-01-04", "2022-01-05"]

    def get_index_stocks(self, index_symbol: str, date: str) -> list[str]:
        self.calls.append(("members", (index_symbol, date)))
        return ["600000.XSHG", "000001.XSHE"]

    def get_index_weights(self, index_symbol: str, date: str) -> pd.DataFrame:
        self.calls.append(("weights", (index_symbol, date)))
        return pd.DataFrame(
            {"weight": [60.0, 40.0]},
            index=["600000.XSHG", "000001.XSHE"],
        )

    def get_price(self, security: list[str], **kwargs: object) -> dict[str, object]:
        self.calls.append(("price", (security, kwargs)))
        return {"security": security, **kwargs}

    def get_extras(self, info: str, security_list: list[str], **kwargs: object) -> dict[str, object]:
        self.calls.append(("extras", (info, security_list, kwargs)))
        return {"info": info, "security": security_list, **kwargs}


class JQDataProviderTest(unittest.TestCase):
    def test_authentication_and_point_in_time_queries(self) -> None:
        api = FakeJQData()
        provider = JQDataProvider(api)
        provider.authenticate(JQDataCredentials("user", "pass"))
        self.assertEqual(provider.trading_days("2022-01-01", "2022-01-31"), ("2022-01-04", "2022-01-05"))
        self.assertEqual(
            provider.index_members("000905.XSHG", "2022-01-04"),
            ("000001.XSHE", "600000.XSHG"),
        )
        self.assertEqual(
            provider.index_weights("000905.XSHG", "2022-01-04"),
            {"600000.XSHG": 0.6, "000001.XSHE": 0.4},
        )

    def test_raw_and_adjusted_prices_are_separate_queries(self) -> None:
        provider = JQDataProvider(FakeJQData())
        raw = provider.raw_daily_prices(["000001.XSHE"], "2022-01-01", "2022-01-31")
        adjusted = provider.post_adjusted_close(["000001.XSHE"], "2022-01-01", "2022-01-31")
        self.assertIsNone(raw["fq"])
        self.assertEqual(adjusted["fq"], "post")
        self.assertFalse(raw["fill_paused"])
        self.assertFalse(adjusted["fill_paused"])

    def test_failed_authentication_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            JQDataProvider(FakeJQData()).authenticate(JQDataCredentials("bad", "creds"))


if __name__ == "__main__":
    unittest.main()
