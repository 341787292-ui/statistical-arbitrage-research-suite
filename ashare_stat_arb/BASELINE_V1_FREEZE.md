# A-Share Baseline v1 Freeze Record

## Purpose

This record freezes the first public-data A-share method-migration result. It
prevents later experiments from silently overwriting a negative result.

## Frozen question

Can a daily, five-day PCA-residual reversal rank be mapped into a cash-equity-
only CSI 500 index-enhancement approximation and retain positive net excess
return after the declared A-share execution costs and constraints?

## Frozen data and method

- Data label: `a-share-free-data-feasibility`.
- Source: BaoStock anonymous API.
- Panel period: 2018-01-02 through 2022-12-30.
- Universe: deterministic 100-name point-in-time monthly subset; 175 names in
  the historical union.
- Benchmark: equal weight inside each monthly subset, not official CSI 500
  historical weights.
- Panel fingerprint:
  `006f02da58931fdf0453c9e45525b7dec72c54091827ad4f9c4f1ab796085603`.
- Signal: reverse cross-sectional percentile rank of the trailing five-day
  PCA residual sum.
- Signal and target update: daily.
- Decision and execution: close decision, next-trading-day open execution.
- Holdout: 2023-2025 remains sealed and is not present in the pilot panel.

## Frozen result

| Metric | Development | Validation | Overall |
|---|---:|---:|---:|
| Annualized gross excess | 3.73% | 1.83% | 2.29% |
| Gross IR | 1.19 | 0.48 | 0.62 |
| Annualized cost drag | 9.13% | 9.59% | 9.48% |
| Annualized net excess | -5.40% | -7.76% | -7.20% |
| Net IR | -1.72 | -2.03 | -1.96 |
| Maximum active drawdown | -6.40% | -21.23% | -25.53% |

Migration is rejected for this fixed mapping. The result means that a weak
pre-cost ranking effect does not survive the chosen daily portfolio mapping.
It does not prove that all A-share residual methods fail.

## Reproducibility evidence

- Result file: `baostock_pilot100_residual_rank5_portfolio.json`.
- Result SHA-256:
  `50D4CDBA269485435F37FC6842F6AC51F88E58C732698A6FD90E1A7110A63009`.
- Test command: `python -m unittest discover -s ashare_stat_arb/tests -v`.
- Freeze audit: 38 A-share tests passed on Python 3.12.

Raw panels and output JSON files stay outside Git because they are generated
artifacts. The fingerprint, hash, frozen metrics, and reproduction commands
are retained in tracked documentation.

## Closed actions

The following actions are forbidden inside Baseline v1:

- changing the five-day signal definition;
- slowing the rebalance schedule;
- changing alpha scale, cost assumptions, or risk limits;
- tuning OU thresholds or introducing a learned signal model;
- opening the 2023-2025 holdout.

Any such change belongs to a separately named and pre-registered experiment.
