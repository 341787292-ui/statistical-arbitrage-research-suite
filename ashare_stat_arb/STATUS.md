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
- Monthly point-in-time constituent and benchmark-weight panel construction is
  implemented and tested with an injectable provider.
- A memory-bounded monthly PCA residual and daily OU stock-alpha pipeline is
  implemented without future-data access.
- A CVXPY long-only index-enhancement optimizer enforces exposure, stock cap,
  turnover, and tracking-error limits.
- The synthetic end-to-end baseline runs successfully; its output is labeled
  as an engineering result rather than investment evidence.

## Deterministic engineering run

The current synthetic fixture completes eight monthly PCA refits and produces
finite long-only, after-cost returns with all optimizer and execution tests
passing. Its headline IR is intentionally excluded from research conclusions:
the fixture contains a generated mean-reverting process and is not market data.

## Next milestone

Connect the licensed JQData account, build the historical point-in-time CSI 500
panel, and run the first real-data PCA-OU long-only baseline.

The first data audit must report:

1. date range and symbol count;
2. duplicate and missing observations;
3. historical delisted-stock coverage;
4. risk-warning and suspension coverage;
5. adjustment-factor consistency;
6. upper/lower-limit flag coverage;
7. daily and monthly active-universe sizes.

## External input required

No empirical A-share return result is reported until local JQData credentials
and entitlements are available. Synthetic fixtures may exercise the pipeline,
but they are engineering tests rather than investment results.
