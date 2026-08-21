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

## Next milestone

Connect one point-in-time A-share daily dataset and produce an audited panel
for the Phase 1 main-board universe.

The first data audit must report:

1. date range and symbol count;
2. duplicate and missing observations;
3. historical delisted-stock coverage;
4. risk-warning and suspension coverage;
5. adjustment-factor consistency;
6. upper/lower-limit flag coverage;
7. daily and monthly active-universe sizes.

## Blocked by data choice

No empirical A-share return result should be generated until the source and
license of the point-in-time data are recorded. A close-price-only API is not
enough for the executable track because it cannot reliably reconstruct
historical eligibility and failed executions.
