# A-Share Diagnostic Case

This case connects the separate `quant_research_agent` research product to the
`ashare_stat_arb` applied product without merging their claims or code ownership.

## Data flow

```text
ashare_stat_arb panel + direction diagnostic
  -> fingerprint and holdout contract
  -> bounded A-share tool registry
  -> evidence review
  -> four competing hypotheses
  -> one fixed residual-space experiment
  -> residual coverage and PCA-refit continuity audit
  -> reflected Agent report
```

The Agent does not download data, construct the executable portfolio, choose
parameters, or access the sealed 2023-2025 holdout. Those remain under the
A-share product's research contract.

## Allowed tools

- `inspect_ashare_direction_diagnostic`
- `run_fixed_residual_ou_mechanism_test`
- `audit_ashare_residual_continuity`

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
- parameter search authorized: no;
- sealed holdout accessed: no.

This is a useful negative result. The Agent completed an evidence-to-hypothesis-
to-experiment-to-reflection loop and stopped the project from tuning an
unvalidated mechanism. Its next action is one pre-registered comparison between
the current stitched residual histories and histories recomputed under each
current composition matrix. It will not choose between them using the sealed
holdout.

## Run

Recreate the A-share panel and direction diagnostic first, then run:

```bash
pip install -r ashare_stat_arb/requirements.txt
python run_ashare_agent.py
```

The local Markdown and JSON reports are written under `reports/` and excluded
from Git. Audited conclusions are copied into `ashare_stat_arb/RESULTS.md`.
