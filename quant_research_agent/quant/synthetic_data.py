from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class PricePoint:
    date: date
    asset_a: float
    asset_b: float


def generate_synthetic_pair(n_days: int = 252) -> list[PricePoint]:
    """Create a deterministic pair with a stable relation and temporary deviations."""
    start = date(2023, 1, 2)
    points: list[PricePoint] = []
    business_index = 0
    current = start

    while len(points) < n_days:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        trend = 100.0 + 0.045 * business_index
        common_cycle = 2.0 * math.sin(business_index / 18.0)
        spread_cycle = 1.6 * math.sin(business_index / 7.0)
        shock = 0.9 * math.sin(business_index / 3.5) if 60 <= business_index <= 150 else 0.0
        asset_b = trend + common_cycle
        asset_a = 1.18 * asset_b + spread_cycle + shock
        points.append(PricePoint(date=current, asset_a=asset_a, asset_b=asset_b))

        business_index += 1
        current += timedelta(days=1)

    return points
