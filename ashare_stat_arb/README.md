# A-Share Statistical Arbitrage Research

This product adapts the methodology in *Deep Learning Statistical Arbitrage*
to point-in-time A-share data and executable market rules. It is deliberately
separate from `paper_reproduction`, which remains the frozen U.S. paper
reproduction baseline.

## Research question

How much of residual mean reversion remains after replacing the paper's
frictionless long-short assumptions with A-share data, T+1 inventory,
board-specific price limits, suspensions, fees, and restricted shorting?

## Two experiment tracks

1. A theoretical long-short track tests whether the paper's mechanism exists
   in A-share residual returns.
2. An executable track uses long-only stock holdings, with an optional index
   hedge added later, and applies A-share execution constraints.

The two tracks must be reported separately. A theoretical long-short result is
not evidence that the strategy can be traded in the cash equity market.

## Current runnable baseline

The first component is a stock-level execution engine that supports:

- nonnegative cash-equity holdings;
- T+1 sellable inventory;
- suspended-stock rejection;
- conservative upper-limit buy rejection;
- conservative lower-limit sell rejection;
- board-specific lot-size inputs;
- configurable commission, stamp duty, and transfer fees.

Run the demonstration:

```bash
python -m ashare_stat_arb.run_execution_demo
```

Run the tests:

```bash
python -m unittest discover -s ashare_stat_arb/tests -v
```

Fee rates and market rules are experiment inputs rather than permanent code
defaults. Every empirical run must record their effective dates.

## Reuse boundary

The following modules can be adapted from `paper_reproduction`:

- no-lookahead rolling PCA residual construction;
- 30-day cumulative residual preprocessing;
- OU, Fourier+FFN, and CNN+Transformer signal models;
- rolling training and stability analysis;
- annualized performance metrics.

The following components must be A-share-specific:

- point-in-time stock universe and corporate-action data;
- stock-level residual composition matrices;
- signal-to-order conversion;
- long-only or index-hedged allocation;
- T+1 inventory and failed-order carryover;
- suspensions, price limits, lot sizes, and effective-dated fees.

See `RESEARCH_SPEC.md` for the first empirical contract and `STATUS.md` for the
current milestone.
