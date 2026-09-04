# A-Share Diagnostic Case

This case connects the separate `quant_research_agent` research product to the
`ashare_stat_arb` applied product without merging their claims or code ownership.

## Data flow

```text
ashare_stat_arb panel + direction diagnostic
  -> fingerprint and holdout contract
  -> bounded A-share tool registry
  -> evidence review
  -> seven competing hypotheses
  -> one fixed residual-space experiment
  -> residual coverage and PCA-refit continuity audit
  -> one pre-registered residual-definition comparison
  -> model-free residual predictability audit
  -> reflected Agent report and paper-backed protocol audit
```

The Agent does not download data, construct the executable portfolio, choose
parameters, or access the sealed 2023-2025 holdout. Those remain under the
A-share product's research contract.

## Allowed tools

- `inspect_ashare_direction_diagnostic`
- `run_fixed_residual_ou_mechanism_test`
- `audit_ashare_residual_continuity`
- `compare_ashare_residual_definitions`
- `audit_ashare_residual_predictability`

Any other tool name is rejected. The experiment contract also records the
panel fingerprint and the fixed PCA/OU settings.

## First result

On the free 2018-2022 BaoStock pilot, the Agent found:

- simple stock-alpha sign reversal: rejected;
- residual OU paper direction: inconclusive, Sharpe 0.047;
- dense signal-induced turnover: supported;
- all 881 usable 30-day OU histories cross monthly PCA refits, averaging
  2.42 model versions per history;
- refit-day residual scale is 1.092 times other days and alpha change is 1.001
  times other days, so model mixing is not established as the cause;
- recomputing the 30-day history under the current PCA composition worsens
  annualized residual return from 0.45% to -2.31% and Sharpe from 0.047 to
  -0.259;
- 1/5-day cross-sectional residual RankIC survives development and validation,
  while longer horizons and raw pooled correlations do not;
- parameter search authorized: no;
- learned time-series model authorized: no;
- sealed holdout accessed: no.

This was a useful intermediate negative result. The diagnostic Agent completed
an evidence-to-hypothesis-to-experiment-to-reflection loop and stopped the
project from tuning an unvalidated OU mechanism. It distinguished the rejected
OU time-series mechanism from a narrower short-horizon cross-sectional ranking
effect and allowed one pre-registered mapping branch.

That mapping branch is now complete. Daily continuous, five-session, and daily
buffered mappings all produce negative net excess return after the declared
A-share costs. The route is closed. Parameter search, learned time-series
models, and the sealed holdout remain unauthorized.

Every new Agent run also reports its active technical foundations and executes
the research protocol audit described in `TECHNICAL_FOUNDATIONS.md`.

## Run

Recreate the A-share panel and direction diagnostic first, then run:

```bash
pip install -r ashare_stat_arb/requirements.txt
python run_ashare_agent.py
```

The local Markdown and JSON reports are written under `reports/` and excluded
from Git. Audited conclusions are copied into `ashare_stat_arb/RESULTS.md`.
