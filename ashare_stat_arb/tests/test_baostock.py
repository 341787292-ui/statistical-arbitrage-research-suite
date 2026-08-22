from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ashare_stat_arb.baostock import BaoStockProvider
from ashare_stat_arb.baostock_pipeline import build_csi500_panel_baostock


class FakeResult:
    def __init__(self, fields: list[str], rows: list[list[str]], error_code: str = "0") -> None:
        self.fields = fields
        self.rows = rows
        self.error_code = error_code
        self.error_msg = "success" if error_code == "0" else "failed"
        self.position = -1

    def next(self) -> bool:
        self.position += 1
        return self.position < len(self.rows)

    def get_row_data(self) -> list[str]:
        return self.rows[self.position]


class FakeBaoStock:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def login(self, user_id: str = "anonymous", password: str = "123456") -> FakeResult:
        self.calls.append(("login", (user_id, password)))
        return FakeResult([], [])

    def logout(self) -> FakeResult:
        self.calls.append(("logout", None))
        return FakeResult([], [])

    def query_trade_dates(self, start_date: str, end_date: str) -> FakeResult:
        self.calls.append(("calendar", (start_date, end_date)))
        return FakeResult(
            ["calendar_date", "is_trading_day"],
            [["2022-01-03", "0"], ["2022-01-04", "1"]],
        )

    def query_zz500_stocks(self, date: str = "") -> FakeResult:
        self.calls.append(("members", date))
        return FakeResult(
            ["updateDate", "code", "code_name"],
            [["2021-11-15", "sh.600000", "A"], ["2021-11-15", "sz.000001", "B"]],
        )

    def query_history_k_data_plus(self, code: str, fields: str, **kwargs: str) -> FakeResult:
        self.calls.append(("history", (code, fields, kwargs)))
        names = fields.split(",")
        row = ["2022-01-04", code] + ["1"] * (len(names) - 2)
        return FakeResult(names, [row])


class BaoStockProviderTests(unittest.TestCase):
    def test_free_provider_queries_are_structured(self) -> None:
        api = FakeBaoStock()
        with BaoStockProvider(api) as provider:
            self.assertEqual(provider.trading_days("2022-01-01", "2022-01-05"), ("2022-01-04",))
            self.assertEqual(
                provider.index_members("2022-01-04"),
                ("sh.600000", "sz.000001"),
            )
            frame = provider.daily_history(
                "sh.600000",
                "2022-01-01",
                "2022-01-05",
                adjustment="post",
            )
        self.assertEqual(frame.loc[0, "code"], "sh.600000")
        history_call = next(value for name, value in api.calls if name == "history")
        self.assertEqual(history_call[2]["adjustflag"], "1")
        self.assertEqual(api.calls[-1][0], "logout")

    def test_query_requires_connection(self) -> None:
        with self.assertRaises(RuntimeError):
            BaoStockProvider(FakeBaoStock()).index_members("2022-01-04")


class FakePanelProvider:
    symbols = ("sh.600000", "sz.300001")

    def trading_days(self, start_date: str, end_date: str) -> tuple[str, ...]:
        return ("2021-12-31", "2022-01-04", "2022-01-05", "2022-02-07", "2022-02-08")

    def index_members(self, as_of_date: str) -> tuple[str, ...]:
        return self.symbols

    def daily_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        *,
        adjustment: str,
    ) -> pd.DataFrame:
        rows = []
        for index, date in enumerate(self.trading_days(start_date, end_date)):
            close = 10.0 + index + (1.0 if symbol.startswith("sz") else 0.0)
            rows.append(
                {
                    "date": date,
                    "code": symbol,
                    "open": str(close - 0.1),
                    "high": str(close + 0.2),
                    "low": str(close - 0.2),
                    "close": str(close * (1.2 if adjustment == "post" else 1.0)),
                    "preclose": str(close - 0.5),
                    "volume": "1000000",
                    "amount": "10000000",
                    "adjustflag": "1" if adjustment == "post" else "3",
                    "turn": "1.0",
                    "tradestatus": "1",
                    "pctChg": "0.1",
                    "isST": "0",
                }
            )
        return pd.DataFrame(rows)


class FakePilotProvider(FakePanelProvider):
    def index_members(self, as_of_date: str) -> tuple[str, ...]:
        start = 0 if as_of_date == "2021-12-31" else 1
        return tuple(f"sh.{600000 + index:06d}" for index in range(start, start + 68))


class BaoStockPanelTests(unittest.TestCase):
    def test_free_panel_uses_prior_membership_and_approximate_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            panel, metadata = build_csi500_panel_baostock(
                FakePanelProvider(),
                start_date="2022-01-01",
                end_date="2022-02-08",
                cache_directory=Path(directory),
                history_buffer_days=40,
            )
        self.assertEqual(panel.shape, (4, 2))
        np.testing.assert_allclose(panel.benchmark_weight.sum(axis=1), 1.0)
        self.assertFalse(np.array_equal(panel.adjusted_close, panel.close_price))
        self.assertFalse(bool(metadata["official_index_weights"]))
        self.assertIn("volume / turnover", str(metadata["benchmark_weight_method"]))

    def test_bounded_pilot_uses_each_point_in_time_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            panel, metadata = build_csi500_panel_baostock(
                FakePilotProvider(),
                start_date="2022-01-01",
                end_date="2022-02-08",
                cache_directory=Path(directory),
                history_buffer_days=40,
                max_symbols=67,
            )
        first = panel.symbols.index("sh.600000")
        entrant = panel.symbols.index("sh.600067")
        self.assertTrue(panel.member[0, first])
        self.assertFalse(panel.member[-1, first])
        self.assertFalse(panel.member[0, entrant])
        self.assertTrue(panel.member[-1, entrant])
        np.testing.assert_array_equal(panel.member.sum(axis=1), np.full(4, 67))
        np.testing.assert_allclose(panel.benchmark_weight.max(axis=1), 1.0 / 67.0)
        self.assertIn("no future membership", str(metadata["symbol_selection"]))


if __name__ == "__main__":
    unittest.main()
