# A-Share Phase 3 Research Contract

## Motivation

Baseline v1 preserved a small gross effect but traded too often. Phase 2 cut
costs by waiting five sessions, but waiting also removed 86.7% of the gross
effect. Phase 3 tests one final portfolio-mapping hypothesis on the already
observed 2018-2022 pilot:

> Can the portfolio inspect the residual rank every day, react to strong new
> information, and avoid trading on small rank changes by using an entry/exit
> buffer?

This is an A-share execution adaptation, not a reproduction of the paper's
original allocation model.

## Frozen stateful signal

The underlying five-day PCA residual reversal score is unchanged. Only its
cross-sectional mapping becomes stateful:

- enter a positive active-tilt state when percentile rank is at least 80%;
- retain the positive state while rank is at least 60%;
- enter a negative active-tilt state when percentile rank is at most 20%;
- retain the negative state while rank is at most 40%;
- otherwise remain neutral;
- center and scale the `-1/0/+1` states across the eligible cross-section.

The 20/40/60/80 thresholds are symmetric and fixed before the result. No
threshold search is permitted.

## Frozen portfolio mapping

- inspect the signal and solve the portfolio daily;
- cap discretionary two-way turnover at 5% per decision;
- use the same 35.2-basis-point cost-based turnover penalty as Phase 2;
- treat constituent and eligibility exits as mandatory turnover and charge all
  resulting costs;
- keep the 99% equity target, 1.5% stock cap, 8% tracking-error limit, T+1,
  suspension, ST, price-limit, next-open timing, and fee model unchanged.

## Data and inference boundary

- The first run uses the same 2018-2022 100-name equal-weight pilot.
- The period is development evidence because it has already been observed.
- The 2023-2025 holdout remains sealed and is rejected by the runner.
- A free full-universe run is allowed only if the bounded result retains
  positive gross alpha, produces positive net alpha, and materially reduces
  cost drag.

## Gates and final stop rule

The existing net excess, net IR, drawdown, cost-share, and rolling-consistency
gates remain unchanged. If the bounded buffered mapping has negative net
excess or loses the gross effect, the signal-to-portfolio migration branch is
closed on 2018-2022. No additional rank thresholds, alpha scales, rebalance
frequencies, or portfolio buffers may be tested on that period.

The holdout may only be opened after a passing full-universe development run,
a data-quality review, a leakage audit, and explicit user approval.
