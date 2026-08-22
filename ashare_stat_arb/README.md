# A-Share Statistical Arbitrage Research

This product adapts the methodology in *Deep Learning Statistical Arbitrage*
to point-in-time A-share data and executable market rules. It is deliberately
separate from `paper_reproduction`, which remains the frozen U.S. paper
reproduction baseline.

## Research question

Can the paper's residual signal add net excess return to a cash-executable
CSI 500 index-enhancement portfolio after point-in-time membership, T+1,
price limits, suspensions, liquidity, costs, and benchmark-relative risk are
modeled?

## Two experiment tracks

1. A theoretical long-short track tests whether the paper's mechanism exists
   in A-share residual returns.
2. An executable track uses long-only stock holdings, with an optional index
   hedge added later, and applies A-share execution constraints.

The two tracks must be reported separately. A theoretical long-short result is
not evidence that the strategy can be traded in the cash equity market.

## Current runnable baseline

The product now contains two connected baselines:

1. a stock-level execution engine; and
2. a no-lookahead PCA residual -> OU signal -> long-only CSI 500 index-
   enhancement research pipeline.

The execution engine supports:

- nonnegative cash-equity holdings;
- T+1 sellable inventory;
- suspended-stock rejection;
- conservative upper-limit buy rejection;
- conservative lower-limit sell rejection;
- board-specific lot-size inputs;
- configurable commission, stamp duty, and transfer fees.

The research pipeline uses monthly point-in-time constituent/weight snapshots,
separate post-adjusted signal prices and raw execution prices, monthly PCA
refits, daily OU updates, CVXPY benchmark-relative portfolio optimization, and
effective-dated A-share costs. `config.py` freezes the accepted periods,
risk constraints, and admission gates.

Run the synthetic engineering demonstration:

```bash
python -m ashare_stat_arb.run_baseline_demo
python -m ashare_stat_arb.run_execution_demo
```

The synthetic result only proves that the pipeline and constraints execute. It
is explicitly not an investment-performance result.

Run the tests:

```bash
python -m unittest discover -s ashare_stat_arb/tests -v
```

Install the A-share research dependencies in a separate Python 3.12
environment:

```bash
pip install -r ashare_stat_arb/requirements.txt
```

JQData credentials are read from local `JQDATA_USERNAME` and
`JQDATA_PASSWORD` environment variables. They and all downloaded market data
remain outside Git.

Once those two environment variables are present, build the licensed local
panel and run the empirical baseline:

```bash
python -m ashare_stat_arb.download_jqdata
python -m ashare_stat_arb.run_empirical_baseline
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
