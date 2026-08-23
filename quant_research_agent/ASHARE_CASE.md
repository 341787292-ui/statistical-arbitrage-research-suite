# A-Share Diagnostic Case

This case connects the separate `quant_research_agent` research product to the
`ashare_stat_arb` applied product without merging their claims or code ownership.

## Data flow

```text
ashare_stat_arb panel + direction diagnostic
  -> fingerprint and holdout contract
  -> bounded A-share tool registry
  -> evidence review
  -> three competing hypotheses
  -> one fixed residual-space experiment
  -> reflected Agent report
```

The Agent does not download data, construct the executable portfolio, choose
parameters, or access the sealed 2023-2025 holdout. Those remain under the
A-share product's research contract.

## Allowed tools

- `inspect_ashare_direction_diagnostic`
- `run_fixed_residual_ou_mechanism_test`

Any other tool name is rejected. The experiment contract also records the
panel fingerprint and the fixed PCA/OU settings.

## First result

On the free 2018-2022 BaoStock pilot, the Agent found:

- simple stock-alpha sign reversal: rejected;
- residual OU paper direction: inconclusive, Sharpe 0.047;
- dense signal-induced turnover: supported;
- parameter search authorized: no;
- sealed holdout accessed: no.

This is a useful negative result. The Agent completed an evidence-to-hypothesis-
to-experiment-to-reflection loop and stopped the project from tuning an
unvalidated mechanism.

## Run

Recreate the A-share panel and direction diagnostic first, then run:

```bash
pip install -r ashare_stat_arb/requirements.txt
python run_ashare_agent.py
```

The local Markdown and JSON reports are written under `reports/` and excluded
from Git. Audited conclusions are copied into `ashare_stat_arb/RESULTS.md`.
