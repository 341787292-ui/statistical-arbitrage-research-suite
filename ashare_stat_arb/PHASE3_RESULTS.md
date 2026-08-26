# A-Share Phase 3 Results

## Bounded buffered-mapping result

Phase 3 kept daily signal inspection but introduced fixed 20/40/60/80
entry/exit buffers. It also retained Phase 2's 5% discretionary turnover cap
and cost-based turnover penalty.

| Metric | Baseline v1 daily rank | Phase 2 five-session | Phase 3 daily buffer |
|---|---:|---:|---:|
| Annualized gross excess | 2.29% | 0.30% | 0.59% |
| Annualized cost drag | 9.48% | 1.07% | 3.70% |
| Annualized net excess | -7.20% | -0.77% | -3.11% |
| Net IR | -1.96 | -0.27 | -1.24 |
| Maximum active drawdown | -25.53% | -5.59% | -13.50% |
| Positive rolling 12-month excess | 3.82% | 18.84% | 18.56% |

The buffered state retained only 25.8% of Baseline v1's gross effect while
costs consumed 627.0% of gross alpha. Average daily two-way turnover remained
4.17%. Gross excess fell from 1.72% in 2018-2019 to 0.23% in 2020-2022, so the
weak effect was not stable across the observed subperiods.

## Decision

The buffered mapping fails the frozen gates. The 2018-2022 signal-to-portfolio
mapping branch is now closed:

- do not tune the 20/40/60/80 thresholds;
- do not change alpha scale or turnover limits;
- do not search additional rebalance schedules;
- do not run Fourier or neural allocation models on the same observed pilot;
- do not open the 2023-2025 holdout.

The bounded evidence supports the following conclusion:

> A weak short-horizon cross-sectional residual-ranking effect is visible in
> the free 100-name pilot, but none of the three pre-registered mappings
> converts it into stable positive net excess return. Daily mapping is too
> expensive; five-session mapping is too slow; a simple daily state buffer
> still trades too much and loses most of the gross effect.

This is a valid negative method-migration result for the declared pilot. It is
not proof that all residual models or all CSI 500 index-enhancement methods
fail.

## Reproducibility

- Command: `python -m ashare_stat_arb.run_phase3_buffered`.
- Output: `baostock_pilot100_phase3_buffered.json`.
- Output SHA-256:
  `8031CDEFC7D4C00823845ECCBC538948E96B6DBD5A02E4CA27A9187517D4D2A5`.
- Test audit: 47 A-share tests passed on Python 3.12.
