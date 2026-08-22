from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ashare_stat_arb.data_pipeline import build_csi500_panel, load_panel, save_panel


class FakeProvider:
    symbols = ("000001.XSHE", "600000.XSHG")

    def trading_days(self, start_date: str, end_date: str) -> tuple[str, ...]:
        return ("2021-12-31", "2022-01-04", "2022-01-05", "2022-02-07", "2022-02-08")

    def index_weights(self, index_symbol: str, as_of_date: str) -> dict[str, float]:
        if as_of_date == "2021-12-31":
            return {self.symbols[0]: 0.6, self.symbols[1]: 0.4}
        return {self.symbols[0]: 0.5, self.symbols[1]: 0.5}

    def _long(self, symbols: tuple[str, ...], *, adjusted: bool) -> pd.DataFrame:
        rows = []
        for day_index, date in enumerate(("2022-01-04", "2022-01-05", "2022-02-07", "2022-02-08")):
            for symbol_index, symbol in enumerate(symbols):
                close = 10.0 + day_index + symbol_index
                rows.append(
                    {
                        "time": pd.Timestamp(date),
                        "code": symbol,
                        "open": close - 0.1,
                        "close": close * (1.1 if adjusted else 1.0),
                        "volume": 1_000_000.0,
                        "money": 10_000_000.0,
                        "high_limit": close * 1.1,
                        "low_limit": close * 0.9,
                        "paused": 0.0,
                    }
                )
        frame = pd.DataFrame(rows)
        return frame[["time", "code", "close"]] if adjusted else frame

    def raw_daily_prices(self, symbols: tuple[str, ...], start_date: str, end_date: str) -> pd.DataFrame:
        return self._long(symbols, adjusted=False)

    def post_adjusted_close(self, symbols: tuple[str, ...], start_date: str, end_date: str) -> pd.DataFrame:
        return self._long(symbols, adjusted=True)

    def st_flags(self, symbols: tuple[str, ...], start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.to_datetime(("2022-01-04", "2022-01-05", "2022-02-07", "2022-02-08"))
        return pd.DataFrame(False, index=dates, columns=symbols)


class DataPipelineTests(unittest.TestCase):
    def test_build_and_round_trip_point_in_time_panel(self) -> None:
        panel = build_csi500_panel(
            FakeProvider(),
            start_date="2022-01-01",
            end_date="2022-02-08",
            symbol_chunk_size=1,
        )
        self.assertEqual(panel.shape, (4, 2))
        np.testing.assert_allclose(panel.benchmark_weight[0], [0.6, 0.4])
        np.testing.assert_allclose(panel.benchmark_weight[-1], [0.5, 0.5])
        self.assertFalse(np.array_equal(panel.adjusted_close, panel.close_price))

        with tempfile.TemporaryDirectory() as directory:
            path, manifest = save_panel(panel, Path(directory) / "panel.npz")
            restored = load_panel(path)
            self.assertTrue(manifest.exists())
            self.assertEqual(panel.fingerprint(), restored.fingerprint())


if __name__ == "__main__":
    unittest.main()
