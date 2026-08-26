# A-Share Statistical Arbitrage Research

This product adapts the methodology in *Deep Learning Statistical Arbitrage*
to point-in-time A-share data and executable market rules. It is deliberately
separate from `paper_reproduction`, which remains the frozen U.S. paper
reproduction baseline.

## Research question

Which parts of the paper's residual methodology retain stable predictive value
and can add net excess return to a cash-executable CSI 500 index-enhancement
portfolio after point-in-time membership, T+1, price limits, suspensions,
liquidity, costs, and benchmark-relative risk are modeled?

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

The research pipeline uses monthly point-in-time constituent snapshots,
separate adjusted signal/return prices and raw execution prices, monthly PCA
refits, daily OU updates, CVXPY benchmark-relative portfolio optimization, and
effective-dated A-share costs. `config.py` freezes the accepted periods, risk
constraints, and admission gates.

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

The primary feasibility path uses BaoStock's free anonymous API. No account or
API key is required. All downloaded market data and caches remain outside Git.

Build the bounded, point-in-time real-data pilot and run the empirical
baseline:

```bash
python -m ashare_stat_arb.download_baostock --start 2018-01-01 --end 2022-12-31 --max-symbols 100 --output ashare_stat_arb/data/baostock_csi500_pilot100_2018_2022.npz
python -m ashare_stat_arb.run_empirical_baseline --panel ashare_stat_arb/data/baostock_csi500_pilot100_2018_2022.npz --output ashare_stat_arb/output/baostock_pilot100_pca_ou.json
python -m ashare_stat_arb.run_signal_diagnostics --panel ashare_stat_arb/data/baostock_csi500_pilot100_2018_2022.npz --output ashare_stat_arb/output/baostock_pilot100_signal_diagnostics.json
python -m ashare_stat_arb.run_residual_audit
python -m ashare_stat_arb.run_residual_comparison
python -m ashare_stat_arb.run_residual_predictability
python -m ashare_stat_arb.run_residual_rank_portfolio
python run_ashare_agent.py
```

The `--max-symbols` run takes a deterministic subset from each historical
membership snapshot without using future membership. It uses an equal-weight
pilot benchmark so the 1.5% stock cap remains feasible. It is an engineering
and signal-feasibility study, not an exact CSI 500 index reproduction.

The optional JQData adapter remains available for licensed users. A
product-grade study still needs exact historical benchmark weights and a more
complete effective-dated market-rule dataset.

The diagnostic command evaluates forward RankIC at 1, 5, 10, and 20 trading
days and compares the original OU direction, its exact reversal, and a zero-
signal benchmark portfolio. It does not search thresholds or select a model.

The residual-audit command measures coverage and determines how many monthly
PCA model versions appear inside each 30-day OU history. The fixed comparison
then contrasts the stitched as-of history with a history recomputed under the
current decision-day composition matrix. The final command lets
`quant_research_agent` consume this evidence through an explicit adapter and
execute the same bounded workflow. The model-free command separately measures
lag autocorrelation and cross-sectional residual reversal at 1, 5, 10, and 20
days without defining a trading strategy. The adapter rejects changed data
fingerprints, panels containing the sealed 2023-2025 holdout, unknown tools,
and parameter-search requests. Agent reports are generated locally under
`reports/` and are not treated as A-share product results until the relevant
evidence is summarized here.

The current evidence closes the OU parameter-tuning branch. It does not close
the broader A-share research question: 1- and 5-day cross-sectional residual
RankIC survives both research periods, while raw pooled magnitude correlations
and longer horizons do not. The one pre-registered five-day rank mapping has
now also failed the executable portfolio gates: overall gross excess is 2.29%,
but 9.48% annualized cost drag produces -7.20% net excess and -1.96 net IR.
The result is a valid negative method-migration finding for this free 100-name
pilot. The holdout remains sealed, and neither rebalance-frequency search nor a
learned time-series model is authorized inside this experiment.

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

See `RESEARCH_SPEC.md` for the empirical contract, `RESULTS.md` for the first
free-data result, `BASELINE_V1_FREEZE.md` for the immutable first migration
decision, `FREE_DATA_AUDIT.md` for the data-source boundary,
`PHASE2_RESEARCH_CONTRACT.md` for the separately registered low-turnover
experiment, `PHASE2_RESULTS.md` for its bounded negative result, and
`PHASE3_RESEARCH_CONTRACT.md` and `PHASE3_RESULTS.md` for the final buffered
test, `MIGRATION_CONCLUSION.md` for the bounded conclusion, and `STATUS.md` for
the current milestone.

Run the frozen Phase 2 bounded check:

```bash
python -m ashare_stat_arb.run_phase2_low_turnover
python -m ashare_stat_arb.run_phase3_buffered
```
