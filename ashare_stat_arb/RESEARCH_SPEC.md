# A-Share Adaptation Contract

## Product boundary

This is an A-share methodology adaptation, not an extension of the reported
U.S. empirical results. The original public-data outputs remain under
`paper_reproduction` and must not be overwritten.

## Phase 1 universe

- Shanghai and Shenzhen main-board A shares only.
- Exclude historical risk-warning and delisting-period observations.
- Require at least 120 trading days since listing.
- Form the universe monthly using only prior-month information.
- Start with 500 to 800 liquid stocks selected by prior-month free-float
  market capitalization and trading activity.
- Retain delisted securities in historical samples to avoid survivorship bias.

ChiNext and STAR Market stocks are deferred until board-specific rule history
is available and tested.

## Required point-in-time fields

| Field group | Minimum fields |
|---|---|
| Identity | date, symbol, exchange, board |
| Prices | unadjusted open/high/low/close, prior close, adjustment factor |
| Liquidity | volume, amount, free-float market capitalization |
| Lifecycle | listing date, delisting date, listed trading days |
| Trading status | suspended, risk warning, upper limit, lower limit |
| Benchmarks | broad-market index return, optional index-futures return |

Adjusted returns are used for signal and residual estimation. Unadjusted
prices and limit prices are retained for execution tests.

## Signal timing

1. Information through trading day `t` close forms the signal.
2. Orders execute no earlier than trading day `t+1` open or a declared VWAP.
3. Shares bought on `t+1` are not sellable until the next session.
4. Failed orders remain unfilled unless an experiment explicitly enables
   carryover.

## Factor and signal baseline

- First factor model: rolling PCA with five components.
- Covariance window: prior 252 trading days.
- Loading window: prior 60 trading days where applicable.
- Signal lookback: prior 30 trading days of cumulative residual returns.
- First signal model: OU+Threshold for interpretable validation.
- Second signal model: Fourier+FFN after the execution baseline passes.

Every residual must retain its stock-level composition matrix so residual
allocations can be mapped into executable stock orders.

## Portfolio tracks

### Theoretical track

- L1-normalized long-short stock weights.
- Used only to compare the A-share residual mechanism with the paper.
- Clearly labeled non-executable unless short availability is modeled.

### Executable track

- Nonnegative cash-equity positions.
- Initial version converts positive residual scores into long-only targets.
- A later version may add a declared CSI index-futures hedge.
- Position sizing must respect cash, lot size, and failed executions.

## Execution rules

- T+1 sellable inventory is mandatory.
- Suspended stocks cannot trade.
- Daily-data baseline rejects buys at the upper limit and sells at the lower
  limit; a later intraday-data version may model queue-dependent fills.
- Fee rates are supplied by an effective-dated configuration.
- Board-specific price limits and lot sizes are supplied by a versioned rule
  table, never inferred from one permanent constant.

## Phase 1 evaluation

- Gross and net annualized mean, volatility, and Sharpe.
- Maximum drawdown and turnover.
- Unfilled order rate by reason.
- T+1 blocked-sell rate.
- Upper-limit blocked-buy and lower-limit blocked-sell rates.
- Difference between theoretical and executable returns.
- Early/late and market-regime stability.

## Reproducibility labels

- `a-share-method-test`: theoretical long-short result before execution rules.
- `a-share-executable-approximation`: daily-data constrained result.
- `a-share-execution-study`: result using audited rule history and sufficiently
  detailed execution data.
