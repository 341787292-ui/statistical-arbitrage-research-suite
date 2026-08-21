# Deep Learning Statistical Arbitrage Reproduction

This directory is a clean, paper-first reproduction of Guijarro-Ordonez,
Pelger, and Zanotti (2022). It is intentionally separate from the earlier
AI-agent prototype.

The first runnable stage implements the classical paper-aligned benchmark:

```text
daily excess returns
  -> rolling PCA factor estimation
  -> out-of-sample residual portfolios
  -> 30-day cumulative residual windows
  -> Ornstein-Uhlenbeck signal
  -> threshold allocation
  -> stock-space L1 normalization
  -> out-of-sample performance metrics
```

The implementation uses only information available before each traded day.
It can therefore be used as the reference implementation for later Fourier
and CNN+Transformer stages.

## Run Stage 1

Install the paper-specific dependencies in a Python 3.11 or 3.12 environment:

```bash
pip install -r paper_reproduction/requirements.txt
python paper_reproduction/run_stage1.py
```

Run its tests:

```bash
python -m unittest discover -s paper_reproduction/tests -v
```

Download the authors' public five-factor residual arrays and verify their
checksums:

```bash
python paper_reproduction/download_official_data.py
```

After downloading the authors' PCA-5 residual file into
`paper_reproduction/data`, run the first empirical approximation with:

```bash
python paper_reproduction/run_official_pca5_ou.py
```

The command writes an auditable JSON result under `paper_reproduction/output`.
It is deliberately labeled a residual-space approximation because the public
repository omits the stock-space residual composition matrices.

Once all three official five-factor residual files are present, reproduce the
complete OU+Threshold portion of Table I with:

```bash
python paper_reproduction/run_official_table1_ou.py
```

Before committing GPU time to the full neural experiment, validate one
official-data rolling window with:

```bash
python paper_reproduction/run_official_neural_smoke.py --model fourier
python paper_reproduction/run_official_neural_smoke.py --model cnn
```

These commands default to one epoch and one retraining origin and therefore
must not be reported as Table I results.

Run the formal public-data approximation with the paper's default training
settings on a CUDA machine:

```bash
python paper_reproduction/run_official_neural_full.py --factor-model pca --model cnn --device cuda
python paper_reproduction/run_official_neural_full.py --factor-model pca --model fourier --device cuda --chunk-size 262144
```

Omit `--device cuda` to run on CPU. The formal command defaults to 100 epochs
and all rolling origins and saves both a JSON manifest and daily arrays.
For Fourier+FFN, `--chunk-size 262144` keeps each observed temporal batch in a
single model call for the published arrays. The value is recorded in every
checkpoint because changing chunk boundaries changes dropout RNG ordering.

The complete five-factor Fourier comparison uses the same settings for each
official residual file:

```bash
python paper_reproduction/run_official_neural_full.py --factor-model ff --model fourier --chunk-size 262144
python paper_reproduction/run_official_neural_full.py --factor-model pca --model fourier --chunk-size 262144
python paper_reproduction/run_official_neural_full.py --factor-model ipca --model fourier --chunk-size 262144
```

After a formal run completes, audit performance across rolling windows:

```bash
python paper_reproduction/analyze_official_neural_stability.py \
  --run-name pca_fourier_e100_seed0_chunk262144
```

The audit validates that all checkpoints stitch back to the formal result and
writes per-window metrics, a concise stability summary, and a diagnostic chart
under `paper_reproduction/output/analysis`.

After completing Fourier+FFN for all three five-factor residual datasets,
generate the Table I comparison with:

```bash
python paper_reproduction/compare_official_fourier.py
```

The default run uses simulated returns whose idiosyncratic price components
follow mean-reverting processes. This is a plumbing and correctness test, not
an empirical reproduction claim.

## Exact-Reproduction Boundary

The paper uses licensed CRSP/Compustat data and 46 firm characteristics. The
authors do not publish those raw inputs. Their public repository does publish
some residual-return arrays, but the residual composition matrices required
for the paper's stock-space normalization are not included.

See `REPRODUCTION_SPEC.md` for the exact experiment contract, target results,
and the data needed before a result may be labeled an exact reproduction.
See `CLOUD_RUN.md` for CUDA, Colab, and second-computer commands.
See `RESEARCHER_WALKTHROUGH.md` for the order in which to study and run each
stage yourself.
