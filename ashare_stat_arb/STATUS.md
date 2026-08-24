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
- Leakage-safe forward RankIC and original/reversed/neutral portfolio controls
  are implemented and run on the same frozen pilot.
- A bounded Quant Research Agent adapter freezes the panel fingerprint,
  PCA/OU parameters, and holdout boundary before invoking A-share tools.
- The Agent's fixed residual-space OU mechanism test is implemented, tested,
  and run on the same pilot without parameter search.
- A residual-continuity audit measures missing model days, cross-sectional
  coverage, PCA model versions within each OU history, universe overlap, and
  refit-day residual/alpha changes.
- A pre-registered comparison tests stitched as-of residual histories against
  histories recomputed under the decision-day PCA composition, with no search.
- A model-free audit reports residual-return autocorrelation and symmetric
  1/5/10/20-day reversal RankIC in development and validation periods.

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

This is a feasibility baseline, not an investable result. The direction
diagnostic confirms that reversing the signal is insufficient: reversed gross
excess is -0.59%, versus +0.15% for the zero-signal control. Mean RankIC is
0.0045 at one day and negative at 5, 10, and 20 days. The signal also creates
about 7% annualized cost drag, compared with 0.11% for the neutral portfolio.

## Agent diagnostic result

The fixed theoretical residual-space experiment produced a 0.45% annualized
paper-direction residual return, Sharpe 0.047, and 97.57% daily unit-gross
turnover. The exact reversal produced -0.45% and Sharpe -0.047. The mechanism
is therefore inconclusive rather than reversed, and parameter optimization is
not authorized.

The continuity audit covers 963 trading days after the first PCA refit. A
residual model is available on 939 days (97.51%); all member residuals are
finite whenever the model runs. Every one of 881 usable 30-day OU histories
crosses a monthly refit and contains 2.42 PCA versions on average. However,
refit-day residual scale is only 1.092 times that of other model days, while
refit-day alpha change is 1.001 times other days. Model mixing is therefore an
unresolved structural confounder, not a demonstrated failure cause.

The fixed residual-definition comparison does not rescue the mechanism. The
current-composition definition produces -2.31% annualized residual return,
Sharpe -0.259, and 106.94% daily residual-position turnover, compared with
+0.45%, 0.047, and 97.57% for the stitched definition. Neither passes the
frozen Sharpe gate of 0.5.

The model-free audit finds a narrower result. One-day development/validation
RankIC is 0.0174/0.0283 and five-day RankIC is 0.0338/0.0210. Both pass the
cross-sectional consistency gate. Ten- and twenty-day validation RankIC falls
to 0.0045 and 0.0068, and raw pooled magnitude correlations are negative at
every horizon. Broad residual mean reversion therefore remains unsupported.

## Next milestone

Keep OU parameter tuning and learned time-series models closed. Pre-register
one continuous cross-sectional residual-rank mapping using the 1/5-day evidence
and pass it through the existing long-only optimizer, T+1 execution, turnover,
and cost model. This is a localization experiment, not a paper reproduction.
The 2023-2025 holdout remains sealed.

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
