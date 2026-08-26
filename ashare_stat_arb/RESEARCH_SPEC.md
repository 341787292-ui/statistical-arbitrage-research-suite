# A-Share CSI 500 Research Contract

## Product boundary

This product adapts the methodology in *Deep Learning Statistical Arbitrage*
to a cash-executable CSI 500 index-enhancement workflow. It does not overwrite
or extend the empirical claims under `paper_reproduction/`.

Two outputs are always reported separately:

1. a theoretical long-short track tests whether the paper's residual mechanism
   exists in A-share returns;
2. an executable long-only track converts the signal into CSI 500 benchmark
   weights plus an alpha tilt under A-share constraints.

The research target is net excess return and information ratio at an auditable
risk level. A high standalone Sharpe is not an admission criterion by itself.

The frozen internship deliverable is a **method migration report**, not a
production strategy or a promise that the method can be migrated. A well
supported negative result is a valid completion. The first formal benchmark is
CSI 500 and must not be changed after results are observed.

No company factor library, private portfolio data, or internal production
parameters are available or required. Public-data experiments must state which
industry/style controls are absent; they cannot be described as ready for live
integration.

## Point-in-time universe and periods

- Benchmark and universe: historical point-in-time constituents of CSI 500
  (`000905.XSHG`).
- Raw data: 2010-01-01 through 2025-12-31.
- Warmup and initial training: 2010-2014.
- Development: 2015-2019.
- Validation: 2020-2022.
- Sealed holdout: 2023-2025. It must not be inspected during tuning.
- The stock universe is formed monthly using membership known on the prior
  trading day.
- A newly admitted constituent receives zero alpha until it has at least 252
  valid trailing observations.
- Historical departures and delisted securities remain in the research panel;
  a departed constituent is liquidated on the next tradable day.

## Point-in-time data contract

The free feasibility source is BaoStock. Every run records the package version,
query period, field list, panel fingerprint, constituent-selection rule, and
weight approximation. Raw downloaded data is never committed to GitHub.

BaoStock supplies historical CSI 500 members, daily OHLCV/amount, trading
status, ST flags, turnover, and percentage returns. It does not supply official
historical index weights in this pipeline. Full-universe free runs therefore
approximate weights from float-market-cap proxies; bounded pilot runs use equal
weights inside a deterministic point-in-time subset. Exact benchmark-relative
claims require an audited source of official historical weights. JQData remains
an optional licensed adapter.

| Field group | Required fields |
|---|---|
| Calendar | complete Shanghai/Shenzhen trading calendar |
| Identity | date, symbol, exchange, board, listing and delisting dates |
| Signal prices | post-adjusted close or an audited equivalent |
| Execution prices | unadjusted open, close, prior close, high/low limits |
| Liquidity | volume, amount, 20-day ADV, free-float market capitalization |
| Status | suspension, ST/risk warning, limit state |
| Universe | point-in-time CSI 500 membership and benchmark weight |
| Risk | industry and CNE6 style exposure when licensed, otherwise declared proxies |

Signal estimation uses adjusted returns. Execution tests use unadjusted prices.
Suspended observations are marked to the prior close for valuation but remain
untradeable. A non-suspension missing observation is a data-quality error.

An eligible stock needs at least 95% non-suspended observations in the trailing
252 sessions and at least 55 valid observations in a 60-day loading window.
Missingness and winsorization rates are audited; the paper-aligned A1 track does
not winsorize returns by default.

## Signal research

Phase 1 evaluates only rolling PCA residuals. Factor counts are
`K = {0, 1, 3, 5, 8, 10, 15}` and `K=5` is the baseline.

- Covariance window: prior 252 trading days.
- Loading window: prior 60 valid trading days.
- Residual signal window: prior 30 trading days.
- Model order: OU+Threshold, Fourier+FFN, then CNN+Transformer.
- Every signal and composition uses only information available by the decision
  close. Orders execute no earlier than the next trading-day open.
- Residual composition must be retained or reproducibly regenerated so alpha
  can be mapped into stock-space targets.

The U.S. implementation is reused only through independently implemented
paper concepts. The authors' source code is a behavioral reference and is not
a commercial dependency.

## Executable portfolio

- Cash equities only, no individual-stock shorting or futures in the first
  executable version.
- Target weights equal CSI 500 benchmark weights plus an alpha tilt.
- Stock exposure remains between 98% and 100%; central target is 99%.
- Single-stock weight is at most 1.5%.
- Industry deviation from the benchmark is at most 3 percentage points.
- Standardized style deviation is at most 0.5 per controlled style.
- Central ex-ante tracking-error target is 8%, with 6% and 10% sensitivities.
- Daily two-way turnover target is 15% and hard cap is 20%; 10%, 15%, and 20%
  are reported as separate scenarios.
- Position optimization maximizes expected alpha less risk, turnover, and cost
  penalties. CVXPY is the approved optimization dependency.

The public multifactor benchmark is built only after the residual signal passes
its first admission gate. It includes value, quality, 12-1 momentum, five-day
reversal, 60-day low volatility, liquidity, and size-as-risk controls.

### Frozen residual-rank migration experiment

The only authorized experiment after the failed OU branch is a five-day
cross-sectional residual-rank signal:

1. sum each stock's available PCA residual returns over the trailing five
   sessions;
2. reverse the sign so a lower cumulative residual receives a higher score;
3. convert the score to a cross-sectional percentile rank and center/scale the
   ranks to unit variance without using residual magnitude;
4. update that score and the target portfolio daily, then map it through the
   existing long-only optimizer and next-open A-share execution model;
5. keep PCA settings, alpha scale, rebalance timing, benchmark, costs, risk
   limits, and the 2023-2025 holdout unchanged.

Monthly in this contract refers to point-in-time universe snapshots and PCA
refits. The frozen portfolio experiment updates its five-day signal and target
weights daily. A slower rebalance is a different experiment and cannot be
introduced after observing this result.

The free pilot has no licensed industry or style exposure matrix. Its optimizer
therefore enforces long-only, stock-weight, turnover, covariance tracking-error,
cash, and execution constraints, but not explicit industry/style neutrality.
That omission is a stated data limitation, not an implementation success.

## Execution rules

1. Information through day `t` close produces the decision.
2. Orders execute at day `t+1` open plus the declared slippage model.
3. Shares purchased on `t+1` are not sellable until the following session.
4. Suspended stocks cannot trade.
5. Buys at the upper limit and sells at the lower limit are rejected in the
   daily-data baseline.
6. Orders respect board lot sizes and available cash.
7. Participation is capped at 5% of trailing 20-day ADV, with 1%, 3%, and 5%
   sensitivities and partial fills.
8. Failed orders expire; the next trading day recomputes targets from current
   holdings instead of carrying stale orders.
9. ST and delisting-period stocks cannot receive new positions.

Price-cage and queue models are deferred until minute or order-book data are
available.

## Effective-dated costs

- Commission: 2.5 basis points each side, configurable, with no retail minimum.
- Stamp duty: 10 basis points on sells before 2023-08-28 and 5 basis points
  from 2023-08-28 onward.
- Transfer fee: effective-dated and applied on both sides without double
  counting exchange fees.
- Slippage: 10 basis points baseline with 5, 10, and 20 basis-point scenarios.

Every run stores the effective-dated cost table in its manifest.

## Admission gates

Signal-level admission requires:

- out-of-sample mean RankIC at least 0.015;
- annualized ICIR at least 0.50;
- stable efficacy at one horizon between 5 and 20 trading days;
- monotonic grouped returns;
- positive evidence in at least two of three subperiods;
- results not concentrated in a small number of names or dates.

Integrated portfolio admission requires:

- positive annualized gross excess return;
- positive annualized net excess return and net information ratio at least
  1.00 in both development and validation;
- IR 1.50 is strong and 2.00 is a stretch target;
- IR at or above 3.00 triggers a leakage, leverage, and valuation audit rather
  than automatic acceptance;
- active maximum drawdown at most 12%;
- costs consume no more than 50% of gross alpha;
- positive rolling 12-month excess return in at least 60% of windows.

Failure of any frozen gate rejects migration for this public-data pilot and
keeps the 2023-2025 holdout sealed. Gates cannot be relaxed after viewing the
portfolio result.

## Disclosure boundary

- Public repositories and reports use generic descriptions such as
  `CSI 500 index enhancement` and `method migration report`.
- Detailed company product names, internal factor names, private parameters,
  client materials, and non-public performance figures must not appear in
  GitHub or public deliverables.
- Company background may inform the research question, but it is not empirical
  evidence unless a source is explicitly authorized for disclosure.

## Reproducibility labels

- `a-share-method-test`: theoretical long-short mechanism test.
- `a-share-free-data-feasibility`: free point-in-time data with declared
  benchmark-weight and limit-price approximations.
- `a-share-executable-approximation`: daily-data constrained long-only result.
- `a-share-execution-study`: audited point-in-time data and rule history.
- `sealed-holdout-result`: one-time 2023-2025 result after design freeze.
