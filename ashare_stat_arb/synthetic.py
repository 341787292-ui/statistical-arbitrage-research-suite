from __future__ import annotations

import numpy as np

from ashare_stat_arb.panel import DailyPanel


def make_synthetic_csi500_panel(
    *,
    trading_days: int = 280,
    assets: int = 80,
    seed: int = 7,
) -> DailyPanel:
    """Create an engineering fixture with mean-reverting idiosyncratic prices."""

    if assets < 67:
        raise ValueError("At least 67 assets are needed for a 1.5% weight cap.")
    rng = np.random.default_rng(seed)
    factor_returns = rng.normal(0.0002, 0.006, size=(trading_days, 3))
    betas = rng.normal(0.0, 0.7, size=(assets, 3))
    state = np.zeros((trading_days, assets), dtype=np.float64)
    innovations = rng.normal(0.0, 0.007, size=(trading_days, assets))
    for t in range(1, trading_days):
        state[t] = 0.86 * state[t - 1] + innovations[t]
    idiosyncratic_returns = np.diff(state, axis=0, prepend=state[[0]])
    returns = factor_returns @ betas.T + idiosyncratic_returns
    returns = np.clip(returns, -0.08, 0.08)
    close = 20.0 * np.exp(np.cumsum(returns, axis=0))
    overnight = rng.normal(0.0, 0.001, size=(trading_days, assets))
    open_price = close * np.exp(overnight)

    start = np.datetime64("2018-01-02")
    calendar_days = np.arange(start, start + np.timedelta64(trading_days * 2, "D"))
    weekdays = calendar_days[
        np.isin((calendar_days.astype("datetime64[D]").astype(int) + 3) % 7, [0, 1, 2, 3, 4])
    ][:trading_days]
    symbols = tuple(f"{600000 + i:06d}.XSHG" for i in range(assets))
    member = np.ones((trading_days, assets), dtype=bool)
    benchmark_weight = np.full((trading_days, assets), 1.0 / assets)
    volume = rng.uniform(2_000_000, 8_000_000, size=(trading_days, assets))
    return DailyPanel(
        dates=weekdays,
        symbols=symbols,
        adjusted_close=close,
        open_price=open_price,
        close_price=close,
        high_limit=close * 1.10,
        low_limit=close * 0.90,
        volume=volume,
        money=volume * close,
        paused=np.zeros((trading_days, assets), dtype=bool),
        is_st=np.zeros((trading_days, assets), dtype=bool),
        member=member,
        benchmark_weight=benchmark_weight,
    )
