# A-Share Phase 2 Research Contract

## Research question

Baseline v1 found a weak positive five-day residual-ranking effect before
costs, but daily portfolio updates created 9.48% annualized cost drag. Phase 2
asks one narrower question:

> Can the same residual-ranking information survive a pre-registered,
> turnover-controlled A-share portfolio mapping and produce positive net
> excess return?

Phase 2 does not search for a new signal and does not revise Baseline v1.

## Evidence status

- All 2018-2022 pilot results have already been observed. They are development
  evidence, not a fresh out-of-sample validation set for Phase 2.
- The 2023-2025 holdout remains sealed.
- A holdout run requires a separate human decision after the code, data audit,
  parameters, and admission gates are frozen.

## Frozen signal

The signal remains identical to Baseline v1:

1. estimate five-factor rolling PCA residual returns;
2. sum each stock's residual returns over the trailing five sessions;
3. reverse the sign;
4. convert the cross-section to centered percentile ranks with unit variance.

No OU threshold, Fourier model, neural model, alternative horizon, or alpha-
scale search is permitted in Phase 2.

## Frozen portfolio intervention

Phase 2 changes only the signal-to-portfolio mapping:

- rebalance once every five trading sessions, anchored to the first eligible
  decision date;
- carry existing holdings between decision dates;
- cap discretionary two-way turnover at 5% on each rebalance date;
- execute point-in-time constituent removals and newly ineligible holdings as
  mandatory trades outside the discretionary 5% budget; their turnover and
  costs remain fully included in reported results;
- if an order is blocked, retain the frozen portfolio target and retry toward
  that target on later tradable days without recalculating the alpha target;
- use a turnover penalty of 35.2 basis points per unit of two-way turnover,
  equal to the declared pre-2023 round-trip variable cost:
  two commissions, two transfer fees, two slippage charges, and sell-side
  stamp duty;
- keep the 99% equity target, 1.5% stock cap, 8% tracking-error limit, T+1,
  suspension, ST, price-limit, and next-open rules unchanged.

The five-session schedule matches the signal horizon. The 5% turnover cap has
an economic, not statistical, origin: at the maximum every week, declared
variable costs are approximately 0.92% per year, below half of Baseline v1's
2.29% gross alpha. No alternative cadence or turnover cap will be selected
after results are observed inside this experiment.

## Data stages

### Stage A: bounded implementation check

- Reuse the frozen 2018-2022 100-name pilot panel.
- Purpose: verify timing, turnover, costs, and output structure.
- Result label: `a-share-phase2-bounded-engineering-check`.
- A pass or failure is not a full CSI 500 migration decision.

### Stage B: free full-universe development study

- Use all point-in-time historical CSI 500 constituents returned by BaoStock.
- Approximate benchmark weights from point-in-time float-market-cap proxies.
- Keep the period at 2018-2022 and do not download or inspect 2023-2025.
- Result label: `a-share-phase2-free-full-universe-development`.
- The result remains a feasibility finding, not official index performance.

## Admission gates

The existing portfolio gates remain unchanged:

- positive annualized gross excess return;
- positive annualized net excess return;
- net information ratio at least 1.00;
- active maximum drawdown no worse than -12%;
- costs no more than 50% of gross alpha;
- positive rolling 12-month excess in at least 60% of available windows.

In addition, Phase 2 must materially reduce annualized cost drag from the 9.48%
Baseline v1 result. Passing on the already observed 2018-2022 period only makes
the method eligible for a data-quality and leakage review; it does not
authorize the holdout automatically.

## Stop rules

Stop Phase 2 without model tuning when any of the following occurs:

- weekly mapping removes the gross effect rather than merely reducing costs;
- net excess remains negative;
- cost share remains above 50% of gross alpha;
- the result depends on missing or invalid benchmark weights;
- implementation cannot preserve next-open timing and A-share execution rules.

Opening the holdout, buying data, or changing the signal family requires an
explicit user decision.
