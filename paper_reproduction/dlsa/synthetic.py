from __future__ import annotations

import numpy as np


def make_factor_market(
    *,
    time_count: int = 900,
    asset_count: int = 30,
    factor_count: int = 3,
    seed: int = 7,
) -> np.ndarray:
    """Generate returns with common factors and mean-reverting residual prices."""

    if asset_count <= factor_count:
        raise ValueError("asset_count must exceed factor_count.")
    rng = np.random.default_rng(seed)
    factor_returns = rng.normal(0.0002, 0.006, size=(time_count, factor_count))
    loadings = rng.normal(0.5, 0.25, size=(asset_count, factor_count))

    persistence = rng.uniform(0.72, 0.94, size=asset_count)
    residual_price = np.zeros((time_count + 1, asset_count), dtype=np.float64)
    shocks = rng.normal(0.0, 0.004, size=(time_count, asset_count))
    for t in range(time_count):
        residual_price[t + 1] = persistence * residual_price[t] + shocks[t]
    residual_returns = np.diff(residual_price, axis=0)
    return factor_returns @ loadings.T + residual_returns

