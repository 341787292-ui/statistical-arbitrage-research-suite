# Data Files

Do not commit licensed or large research data to Git.

The author repository publishes compressed residual-return arrays. For the
first empirical smoke test, download the five-factor PCA residual file listed
in `official_manifest.json` into this directory and keep its original name.

The published residual file is enough to test preprocessing and model
training in residual space. It is not enough for the paper's exact stock-space
normalization because the corresponding residual composition matrix is not
published.

Raw-data regeneration additionally requires point-in-time CRSP/Compustat
inputs. Their expected transformations are documented in
`../REPRODUCTION_SPEC.md`.

