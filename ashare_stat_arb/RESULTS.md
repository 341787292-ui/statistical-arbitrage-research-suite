# A-Share Feasibility Results

## BaoStock PCA-OU pilot

This run verifies that the A-share research path works on real, freely
re-downloadable market data. It is not an investment-performance claim and it
does not reproduce the official CSI 500 index.

### Data and method

- Period: 2018-01-02 through 2022-12-30.
- Source: BaoStock 0.9.3 anonymous API.
- Universe: first 100 sorted constituents from each prior-trading-day monthly
  CSI 500 snapshot; 175 names in the historical union.
- Benchmark: equal weight within each monthly pilot subset.
- Signal: monthly rolling PCA with five factors, 252-day covariance window,
  60-day loading window, and 30-day OU residual signal.
- Portfolio: cash-equity-only, 99% target exposure, 1.5% stock cap, 20% daily
  two-way turnover cap, and 8% ex-ante tracking-error limit.
- Timing: close decision, next-trading-day open execution, adjusted open-to-open
  holding returns.
- Execution: suspension, ST, upper/lower limits, T+1 timing, directional fees,
  stamp duty, and 10 basis-point slippage.

### Data audit

| Check | Result |
|---|---:|
| Trading days | 1,215 |
| Downloaded symbol union | 175 |
| Member observations | 121,500 |
| Missing adjusted signal price | 0.0272% |
| Suspended member observations | 0.8601% |
| ST member observations | 0.3407% |
| Invalid benchmark-weight days | 0 |

Panel fingerprint:
`006f02da58931fdf0453c9e45525b7dec72c54091827ad4f9c4f1ab796085603`

### Result

| Metric | Result |
|---|---:|
| Annualized benchmark return | 11.80% |
| Annualized gross strategy return | 9.24% |
| Annualized gross excess return | -2.57% |
| Gross information ratio | -0.86 |
| Annualized cost drag | 7.09% |
| Annualized net strategy return | 2.14% |
| Annualized net excess return | -9.66% |
| Net information ratio | -3.22 |
| Average daily two-way turnover | 6.13% |
| Maximum active drawdown | -31.47% |

### Interpretation

The engineering baseline is complete, but the strategy fails every admission
gate. Gross underperformance shows that transaction cost is not the only
problem. The current OU stock-space alpha loses about 2.57% per year before
costs, while high turnover adds about 7.09% annualized cost drag.

The next experiment is diagnostic rather than parameter optimization:

1. calculate forward RankIC at 1, 5, 10, and 20 days;
2. compare the OU signal with its sign reversed;
3. report signal coverage and holding-period decay;
4. test wider OU thresholds and slower rebalance schedules only after the
   predictive direction is confirmed.

## Direction diagnostic

The first diagnostic keeps all model and portfolio parameters frozen. It
compares the original stock-space alpha, its exact sign reversal, and a zero-
signal control portfolio on the same dates and data.

### Forward RankIC

Returns begin at the next trading-day open, matching the declared execution
time.

| Horizon | Mean RankIC | Annualized ICIR | Positive days |
|---|---:|---:|---:|
| 1 day | 0.0045 | 0.53 | 50.85% |
| 5 days | -0.0019 | -0.22 | 48.91% |
| 10 days | -0.0095 | -1.13 | 45.52% |
| 20 days | -0.0023 | -0.28 | 48.37% |

The signal fails the required mean RankIC of 0.015 at every horizon and does
not maintain one predictive direction across horizons.

### Portfolio controls

| Variant | Gross excess | Cost drag | Net excess | Net IR | Daily two-way turnover |
|---|---:|---:|---:|---:|---:|
| Original OU | -2.57% | 7.09% | -9.66% | -3.22 | 6.13% |
| Reversed OU | -0.59% | 7.01% | -7.60% | -2.54 | 6.06% |
| Zero signal | 0.15% | 0.11% | 0.04% | 0.03 | 0.09% |

Reversing the signal improves gross excess by about 1.98 percentage points,
but it still trails the zero-signal control before costs. The failure is
therefore not a simple sign error. Signal-induced daily reallocation explains
almost all of the cost drag, while weak and unstable predictive content
explains the pre-cost shortfall.

The OU stock-space signal is finite for all member observations and nonzero on
94.33% of them. This broad mapped exposure is a likely turnover amplifier and
must be studied after the residual mechanism itself is validated.

### Reproduce

```bash
pip install -r ashare_stat_arb/requirements.txt
python -m ashare_stat_arb.download_baostock --start 2018-01-01 --end 2022-12-31 --max-symbols 100 --output ashare_stat_arb/data/baostock_csi500_pilot100_2018_2022.npz
python -m ashare_stat_arb.run_empirical_baseline --panel ashare_stat_arb/data/baostock_csi500_pilot100_2018_2022.npz --output ashare_stat_arb/output/baostock_pilot100_pca_ou.json
python -m ashare_stat_arb.run_signal_diagnostics --panel ashare_stat_arb/data/baostock_csi500_pilot100_2018_2022.npz --output ashare_stat_arb/output/baostock_pilot100_signal_diagnostics.json
```
