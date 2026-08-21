# Reproduction Results

All values below use the authors' published residual-return arrays. Because the
public repository omits the residual composition matrices, the current results
normalize allocations in residual space and are labeled approximations.

## OU+Threshold, five factors

| Factor model | Sharpe observed | Sharpe paper | Mean observed | Mean paper | Vol observed | Vol paper |
|---|---:|---:|---:|---:|---:|---:|
| Fama-French | 0.338 | 0.38 | 1.21% | 0.9% | 3.57% | 2.3% |
| PCA | 0.925 | 0.73 | 2.42% | 4.4% | 2.62% | 6.1% |
| IPCA | 0.640 | 0.97 | 2.61% | 3.8% | 4.07% | 4.0% |

The Fama-French Sharpe is close to the paper. The PCA and IPCA differences are
large enough that they must not be described as exact replication. The missing
stock-space mapping is economically material because it changes portfolio
normalization, leverage distribution, and therefore both mean and volatility.
For Fama-French residuals, the unavailable underlying space also includes the
traded factor portfolios rather than only individual stocks.

## Official-data neural smoke tests

The following runs use one epoch and one rolling origin only. They establish
that the full official-data pipeline executes; they are not performance claims.

| Factor model | Model | Device | OOS days | Sharpe |
|---|---|---|---:|---:|
| PCA | Fourier+FFN | CPU | 125 | -0.488 |
| PCA | CNN+Transformer | CPU | 125 | 1.377 |

The formal neural comparison must use 100 epochs and all rolling origins.

## Five-factor Fourier+FFN, formal public-data runs

The Fama-French, PCA, and IPCA runs use the paper's default 100 training
epochs, a 1,000-day rolling training window, retraining every 125 days, a
30-day lookback, and all 31 rolling origins. Each run covers 3,781
out-of-sample days. All 93 checkpoints completed and all stitched daily
returns are finite.

| Factor model | Sharpe observed | Sharpe paper | Mean observed | Mean paper | Vol observed | Vol paper | Early / late Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fama-French | 1.373 | 1.66 | 4.61% | 3.1% | 3.36% | 1.8% | 2.69 / -0.25 |
| PCA | 3.840 | 1.98 | 6.87% | 12.4% | 1.79% | 6.3% | 5.73 / 0.81 |
| IPCA | 1.818 | 1.90 | 7.11% | 7.7% | 3.91% | 4.1% | 2.90 / 0.50 |

The observed Sharpe ranking, PCA then IPCA then Fama-French, matches Table I.
IPCA is the closest numerical approximation: its relative errors are -4.3%
for Sharpe, -7.7% for annualized mean, and -4.6% for volatility. This closeness
does not make the IPCA result exact. All three runs still normalize allocations
in residual space because the stock-space composition matrices are missing.

The Fama-French and PCA results show why Sharpe alone is not enough to assess
replication quality. Fama-French has a higher mean and volatility than the
paper, while PCA has a much lower mean and less than one third of the paper's
volatility. PCA's inflated Sharpe therefore reflects a materially different
risk scale, not reproduced outperformance. The missing mapping changes
portfolio normalization, leverage distribution, and both mean and volatility.
For Fama-French residuals, the unavailable underlying space also includes the
traded factor portfolios rather than only individual stocks.

The unified comparison artifacts are:

- `output/analysis/fourier_5factor_comparison.csv`
- `output/analysis/fourier_5factor_comparison.json`
- `output/analysis/fourier_5factor_comparison.png`

They are regenerated with:

```bash
python paper_reproduction/compare_official_fourier.py
```

## PCA five-factor Fourier+FFN detail

The complete PCA run differs from Table I as follows:

| Metric | Observed | Paper Table I | Difference |
|---|---:|---:|---:|
| Annualized Sharpe | 3.840 | 1.98 | +1.860 |
| Annualized mean return | 6.87% | 12.4% | -5.53 pp |
| Annualized volatility | 1.79% | 6.3% | -4.51 pp |

Additional diagnostics for this implementation are:

| Diagnostic | Value |
|---|---:|
| Cumulative return | 179.82% |
| Maximum drawdown | -2.26% |
| Positive-day rate | 58.87% |
| Average daily turnover | 0.853 |
| Average short proportion | 21.70% |

The observed Sharpe is higher than the paper's value, but this is **not** an
outperformance result. The annualized mean is only about half of the paper's,
while volatility is less than one third of the paper's. The inflated Sharpe is
therefore driven by a materially different risk scale. This result must be
reported as a public-data, residual-space approximation rather than an exact
replication.

PCA reproducible artifacts:

- `output/pca_fourier_e100_seed0_chunk262144.json`: configuration, paper target,
  observed metrics, and per-window training losses.
- `output/pca_fourier_e100_seed0_chunk262144_daily.npz`: stitched out-of-sample
  returns, turnover, and short-proportion arrays.
- `output/periods/pca_fourier_e100_seed0_chunk262144/`: 31 independently
  resumable rolling-window checkpoints.

### Rolling-window stability

The aggregate Sharpe masks a material decline across the out-of-sample path.
The first 16 retraining windows and final 15 windows produce:

| Segment | OOS days | Annualized mean | Annualized vol | Sharpe |
|---|---:|---:|---:|---:|
| First 16 windows | 2,000 | 12.10% | 2.11% | 5.73 |
| Final 15 windows | 1,781 | 1.01% | 1.25% | 0.81 |

Twenty-seven of the 31 windows have a positive Sharpe, so the aggregate result
is not generated by only one or two isolated winning windows. However, the
first 16 windows contribute 93.1% of the total arithmetic return. The top three
individual windows contribute 28.8%, and the top five contribute 44.5%. The
evidence therefore supports broad early profitability followed by pronounced
late-sample decay, not stable performance at the full-sample Sharpe of 3.84.

The final training-score proxy (the negative of the final Sharpe loss) has a
0.86 correlation with window-level out-of-sample Sharpe. Its average declines
from 0.522 in the first 16 windows to 0.287 in the final 15. The weakening is
therefore visible in both the rolling training samples and subsequent OOS
performance; it is not the simple pattern of stable in-sample fit followed by
isolated OOS failure. The final origin has only 31 OOS days, so its standalone
Sharpe is less precise than those of the 30 complete 125-day windows.

This is descriptive evidence, not a causal diagnosis. Without the unpublished
stock-space composition matrices, the changing residual universe and leverage
normalization may contribute to the pattern. The public arrays also omit an
exact date vector, so periods are identified by rolling origin rather than by
calendar market regime.

The reproducible stability audit is generated with:

```bash
python paper_reproduction/analyze_official_neural_stability.py \
  --run-name pca_fourier_e100_seed0_chunk262144
```

It writes the per-window CSV, validation JSON, and diagnostic PNG under
`output/analysis/`.

### Execution sensitivity

An earlier PCA run used a model chunk size of 2,048 instead of keeping each
observed temporal batch in one model call. Its daily returns have a 0.982
correlation with the full-batch run, and the aggregate Sharpe changes from
3.742 to 3.840. The main late-sample-decay conclusion is therefore robust to
this execution choice. The chunk size remains part of the run identifier and
checkpoint metadata because changing chunk boundaries changes dropout random
number ordering even under a fixed seed.

### Cross-model stability

The early-to-late deterioration is shared rather than PCA-specific:

- Fama-French Sharpe falls from 2.69 to -0.25.
- PCA Sharpe falls from 5.73 to 0.81.
- IPCA Sharpe falls from 2.90 to 0.50.

The corresponding training-score/OOS-Sharpe correlations are 0.74, 0.86, and
0.75. These are descriptive diagnostics, not a causal test. The public arrays
do not include calendar dates, and missing stock-space mappings may alter both
the level and stability of the observed strategies.
