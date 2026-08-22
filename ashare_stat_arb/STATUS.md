# A-Share Product Status

## Completed

- Independent product directory created without changing the U.S. paper
  reproduction.
- Reuse and rewrite boundaries documented.
- Phase 1 point-in-time data contract defined.
- T+1 stock inventory implemented.
- Suspensions and conservative price-limit execution implemented.
- Long-only target enforcement, lot rounding, and directional fees
  implemented.
- Unit tests cover the initial execution rules.
- CSI 500 research periods, risk limits, costs, and admission gates are frozen
  in code and documentation.
- A standard daily panel, deterministic fingerprint, and data-audit schema are
  implemented.
- An injectable JQData SDK wrapper is implemented without storing credentials.
- A free anonymous BaoStock adapter, local Parquet cache, and historical CSI
  500 panel builder are implemented.
- Monthly point-in-time constituent and benchmark-weight panel construction is
  implemented and tested with an injectable provider.
- A memory-bounded monthly PCA residual and daily OU stock-alpha pipeline is
  implemented without future-data access.
- A CVXPY long-only index-enhancement optimizer enforces exposure, stock cap,
  turnover, and tracking-error limits.
- The synthetic end-to-end baseline runs successfully; its output is labeled
  as an engineering result rather than investment evidence.
- A 2018-2022 real-data feasibility pilot runs on 100 point-in-time sampled
  constituents per month. It uses no future membership, adjusted open-to-open
  returns, raw prices for execution tests, and an equal-weight pilot benchmark.

## Deterministic engineering run

The current synthetic fixture completes eight monthly PCA refits and produces
finite long-only, after-cost returns with all optimizer and execution tests
passing. Its headline IR is intentionally excluded from research conclusions:
the fixture contains a generated mean-reverting process and is not market data.

## First free-data empirical run

The first BaoStock pilot covers 1,215 trading days and 175 names in the union,
with exactly 100 point-in-time sampled members per month. Its PCA(5)-OU
long-only result is deliberately retained even though it fails admission:

- annualized benchmark return: 11.80%;
- annualized gross strategy return: 9.24%;
- annualized gross excess return: -2.57%;
- annualized cost drag: 7.09%;
- annualized net strategy return: 2.14%;
- annualized net excess return: -9.66%;
- net information ratio: -3.22;
- average daily two-way turnover: 6.13%.

This is a feasibility baseline, not an investable result. It shows that both
the current OU alpha and its turnover require redesign before deeper model
work is justified.

## Next milestone

Diagnose the signal before tuning it: measure forward RankIC by horizon, test
the OU sign and threshold logic in residual space, and separate alpha decay
from portfolio turnover. Only after that gate should the project run the full
CSI 500 universe or add Fourier/deep-learning signal models.

The first data audit must report:

1. date range and symbol count;
2. duplicate and missing observations;
3. historical delisted-stock coverage;
4. risk-warning and suspension coverage;
5. adjustment-factor consistency;
6. upper/lower-limit flag coverage;
7. daily and monthly active-universe sizes.

## Data boundary

No paid data is required for the current feasibility milestone. Exact
historical CSI 500 benchmark-relative claims still require official historical
weights and stronger rule-history coverage. The free pilot is always labeled
`a-share-free-data-feasibility`.
