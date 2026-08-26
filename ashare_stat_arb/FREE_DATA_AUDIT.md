# Free A-Share Data Decision

## Decision

The next development-stage experiment can remain fully free. BaoStock remains
the primary source because the existing adapter already supports historical
CSI 500 membership, daily prices, turnover, trading status, ST status, and
adjustment-aware returns.

No single verified free source supplies every field needed for an exact
historical CSI 500 index-enhancement reconstruction. In particular, the free
pipeline does not have audited official historical constituent weights or a
licensed point-in-time industry/style risk model.

## Source roles

| Source | Cost | Useful fields | Main limitation | Project role |
|---|---|---|---|---|
| BaoStock | Free, anonymous | Historical CSI 500 membership; daily OHLCV and amount; turnover; suspension/trading status; ST flag; adjusted/unadjusted return inputs | No official historical CSI 500 weights; no institutional risk model | Primary reproducible research source |
| China Securities Index / AKShare wrapper | Free access to published files | Latest published constituent list and weights | The verified wrapper has no historical-date argument and cannot replace a point-in-time weight history | Current-snapshot cross-check only |
| Tushare Pro | Point-gated | Monthly historical index constituents and weights; broad market fields | `index_weight` requires at least 2,000 points and is not treated as a guaranteed free dependency | Optional upgrade, not required for Phase 2 development |
| JQData | Licensed account | Historical index weights, prices, ST and limit fields through the existing adapter | Account and quota required | Optional audited-data upgrade |

## Free-data benchmark rule

For a full-universe free run, benchmark weights are approximated from
point-in-time float-market-cap proxies derived from price, volume, and turnover.
For the bounded 100-name pilot, equal weights remain the declared benchmark.

Results using either approximation must be labeled
`a-share-free-data-feasibility`. They may answer whether a mechanism deserves
further study, but they must not be described as official CSI 500 excess
performance or production-ready evidence.

## Upgrade trigger

Paid or licensed data is justified only after a free full-universe experiment
shows all of the following before the sealed holdout is opened:

1. positive gross excess return;
2. positive net excess return under conservative costs;
3. materially lower turnover than Baseline v1;
4. stability across development subperiods;
5. no obvious concentration in a small set of stocks or dates.

If these conditions fail, paying for historical weights would improve
measurement precision but would be unlikely to rescue the underlying method.
