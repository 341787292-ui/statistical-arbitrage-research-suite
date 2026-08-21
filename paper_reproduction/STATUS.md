# Reproduction Status

## Implemented and tested

- Independent rolling PCA residual construction with no-lookahead tests.
- OU parameter extraction and threshold trading policy.
- Fourier representation and feedforward allocation model.
- Causal CNN plus Transformer allocation model.
- Sharpe-objective training and 1,000/125-style rolling OOS driver.
- Stock-space mapping and L1 normalization when composition matrices exist.
- Low-memory execution for the authors' full residual arrays.
- Complete Fama-French-5, PCA-5, and IPCA-5 Fourier+FFN public-data runs: 100
  epochs, 31 rolling origins, and 3,781 out-of-sample days per factor model.
- Reproducible rolling-window stability audit with checkpoint reconciliation.
- Unified five-factor Fourier comparison against the corresponding Table I
  targets.

## Current empirical level

The official Fama-French, PCA, and IPCA five-factor residual-return files are
available locally and have passed shape, missing-value, and checksum audits.
They can be used for residual-space approximations. This is not yet an exact
Table I reproduction because the official repository does not publish the
residual composition matrices required to map residual allocations back to
original stock weights before normalization.

## Remaining work

1. Run all 100 epochs and rolling origins for CNN+Transformer. The complete
   five-factor Fourier+FFN public-data comparison is finished; the CNN model
   now requires a practical CUDA execution environment.
2. Rebuild composition matrices from licensed or equivalent point-in-time raw
   data before labeling any result an exact empirical reproduction.
3. Reproduce the remaining factor-count rows of Table I.
4. Reproduce costs and robustness tables.
