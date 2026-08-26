# A-Share Phase 2 Results

## Bounded engineering check

Phase 2 retained the frozen five-day residual-rank signal and changed only the
portfolio mapping. It rebalanced every five trading sessions, limited
discretionary two-way turnover to 5% per decision, charged mandatory index and
eligibility turnover, and used the declared round-trip variable cost as the
optimizer's turnover penalty.

The run used the same 2018-2022 100-name panel as Baseline v1. Because this
period had already been observed, the result is development evidence rather
than a new out-of-sample validation.

| Metric | Baseline v1 daily | Phase 2 five-session |
|---|---:|---:|
| Annualized gross excess | 2.29% | 0.30% |
| Annualized cost drag | 9.48% | 1.07% |
| Annualized net excess | -7.20% | -0.77% |
| Net IR | -1.96 | -0.27 |
| Maximum active drawdown | -25.53% | -5.59% |
| Positive rolling 12-month excess | 3.82% | 18.84% |

The slower mapping reduced annualized cost drag by approximately 88.7%, but it
also removed approximately 86.7% of the gross effect. Only 13.3% of Baseline
v1's gross alpha remained. Costs still consumed 353.5% of gross alpha, and net
excess remained negative.

The early 2018-2019 segment had slightly negative gross excess (-0.09%), while
2020-2022 had only 0.43% positive gross excess before 1.11% annualized costs.
The result therefore does not support the hypothesis that weekly execution can
retain the signal while solving the cost problem.

## Decision

The pre-registered five-session branch is closed. No search over two-, three-,
four-, or alternative weekly schedules is allowed after this result. The
2023-2025 holdout remains sealed.

This result adds an important mechanism finding:

> The observed five-day residual-rank information is short-lived. Reducing
> trading by waiting five sessions lowers costs, but the delayed portfolio no
> longer captures most of the pre-cost effect.

This remains a bounded 100-name result with an equal-weight pilot benchmark.
It is not an official CSI 500 performance claim.

## Reproducibility

- Command: `python -m ashare_stat_arb.run_phase2_low_turnover`.
- Output: `baostock_pilot100_phase2_low_turnover.json`.
- Output SHA-256:
  `70ABCD2B58E1CBAEC56B3BE66564CB8B38D409D57574B79DE619E0E911D9AA35`.
- Test audit: 44 A-share tests passed on Python 3.12.
