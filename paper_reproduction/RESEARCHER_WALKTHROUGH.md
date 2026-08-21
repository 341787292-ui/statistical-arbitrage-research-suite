# Researcher Walkthrough

This is the recommended order for reproducing the paper yourself. Do not move
to the Agent layer until you can explain the outputs and assumptions at each
stage.

## 1. Freeze the research contract

Read `REPRODUCTION_SPEC.md` beside the paper. Be able to state:

- the stock universe and sample period;
- how Fama-French, PCA, and IPCA residuals are formed;
- why signals use only the previous 30 trading days;
- why neural models use a 1,000-day training window and retrain every 125 days;
- how residual allocations map back to underlying assets before L1
  normalization.

Do not call a result exact if its input data or normalization differs from this
contract.

## 2. Verify the implementation on controlled data

```bash
python paper_reproduction/run_stage1.py
python paper_reproduction/run_stage2_smoke.py
python paper_reproduction/run_stage3_rolling_smoke.py
python -m unittest discover -s paper_reproduction/tests -v
```

Check that the no-lookahead tests pass, the neural gradients are finite, and
only out-of-sample days are returned. Synthetic Sharpe ratios are plumbing
checks and have no empirical meaning.

## 3. Audit the authors' public data

```bash
python paper_reproduction/download_official_data.py
```

The downloader verifies checksums from `data/official_manifest.json`. Confirm
that each residual array has shape `4781 x 9483` and understand that zero
encodes a missing observation rather than a true zero return.

## 4. Reproduce the parametric benchmark

```bash
python paper_reproduction/run_official_table1_ou.py
```

Compare the three five-factor rows in `RESULTS.md`. Inspect mean and volatility,
not only Sharpe. Explain why residual-space normalization can change both.

## 5. Validate the neural data path

```bash
python paper_reproduction/run_official_neural_smoke.py --factor-model pca --model fourier
python paper_reproduction/run_official_neural_smoke.py --factor-model pca --model cnn
```

These one-epoch runs answer only: can official residuals flow through feature
construction, rolling training, portfolio normalization, and OOS evaluation?
They do not answer whether the model reproduces Table I.

## 6. Run the formal rolling experiments

Prefer CUDA for CNN+Transformer. Start with PCA-5, then repeat for FF-5 and
IPCA-5.

```bash
python paper_reproduction/run_official_neural_full.py --factor-model pca --model fourier --device cuda
python paper_reproduction/run_official_neural_full.py --factor-model pca --model cnn --device cuda
```

Each 125-day period is checkpointed. Inspect training-loss paths and daily OOS
returns before comparing aggregate annualized metrics with Table I.

## 7. Diagnose differences

For every mismatch, test the explanation rather than narrating it. The first
audit list is:

1. source file checksum and missing-value semantics;
2. lookback and OOS alignment;
3. stock-space versus residual-space L1 normalization;
4. optimizer, dropout, seed, and temporal batch size;
5. arithmetic annualization and standard-deviation convention;
6. PyTorch version and hardware nondeterminism.

## 8. Define the completion boundary

The current public-data track is complete when all nine five-factor cells in
Table I have auditable runs: three factor models crossed with OU+Threshold,
Fourier+FFN, and CNN+Transformer. Exact empirical replication remains blocked
until the unpublished residual composition matrices are rebuilt from licensed
or equivalent point-in-time raw data.

Only after this point should the reproduction functions be exposed as tools to
the Quant Research Agent.
