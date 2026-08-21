# Reproduction Contract

## Source Of Truth

- Paper: *Deep Learning Statistical Arbitrage*, September 25, 2022 draft.
- Authors: Jorge Guijarro-Ordonez, Markus Pelger, and Greg Zanotti.
- Local reference: `references/papers/deep_learning_statistical_arbitrage.pdf`.
- Official code: <https://github.com/gregzanotti/dlsa-public>.

This reproduction follows the paper before following implementation quirks in
the public code. Any deviation must be recorded in an experiment manifest.

## Main Empirical Contract

### Data and universe

- Daily US equity adjusted returns from CRSP, January 1978 through December
  2016.
- One-month Treasury bill rate from the Kenneth French Data Library.
- Excess returns are used throughout.
- The eligible universe at month `m` contains stocks whose prior-month market
  capitalization is at least 0.01% of prior-month total market capitalization.
- The resulting universe averages roughly 550 large and liquid stocks.
- IPCA uses 46 cross-sectionally centered and rank-transformed firm
  characteristics lagged by one month.
- Out-of-sample residuals cover January 1998 through December 2016.
- Trading evaluation covers January 2002 through December 2016.

### Residual portfolio construction

- Fama-French: `K in {0, 1, 3, 5, 8}`; estimate stock loadings using the prior
  60 trading days.
- PCA: `K in {0, 1, 3, 5, 8, 10, 15}`; estimate the correlation matrix using
  the prior 252 trading days and loadings using the prior 60 trading days.
- IPCA: `K in {0, 1, 3, 5, 8, 10, 15}`; re-estimate yearly using the prior 240
  months of monthly returns and lagged characteristics.
- Every residual at date `t` must be formed using a composition matrix known at
  `t-1`. No observation at or after `t` may enter factor or loading estimates.

### Signals and trading

- Signal lookback: 30 trading days of cumulative residual returns.
- OU threshold benchmark: `c_thresh = 1.25`, `c_crit = 0.25`.
- CNN+Transformer: two causal convolution layers, 8 filters, filter size 2,
  instance normalization, residual connection, one transformer encoder layer,
  4 attention heads, hidden size 16, and dropout 0.25.
- Neural objective: maximize the daily sample Sharpe ratio.
- Rolling neural training window: 1,000 trading days.
- Retraining frequency: 125 trading days.
- Temporal batch size: 125 trading days.
- Optimizer: Adam, learning rate 0.001, 100 epochs.
- Residual allocations must be mapped into original-stock weights and then
  normalized so the stock-weight L1 norm equals one.
- Report arithmetic annualized mean (`252 * daily mean`), annualized volatility
  (`sqrt(252) * daily standard deviation`), and their ratio.

### First target: Table I, five-factor rows

| Model | Factor model | Sharpe | Mean | Volatility |
|---|---:|---:|---:|---:|
| CNN+Transformer | Fama-French 5 | 3.21 | 4.6% | 1.4% |
| CNN+Transformer | PCA 5 | 3.36 | 14.3% | 4.2% |
| CNN+Transformer | IPCA 5 | 4.16 | 8.7% | 2.1% |
| Fourier+FFN | Fama-French 5 | 1.66 | 3.1% | 1.8% |
| Fourier+FFN | PCA 5 | 1.98 | 12.4% | 6.3% |
| Fourier+FFN | IPCA 5 | 1.90 | 7.7% | 4.1% |
| OU+Threshold | Fama-French 5 | 0.38 | 0.9% | 2.3% |
| OU+Threshold | PCA 5 | 0.73 | 4.4% | 6.1% |
| OU+Threshold | IPCA 5 | 0.97 | 3.8% | 4.0% |

Exact floating-point equality is not expected across hardware and library
versions. Direction, ordering, and economically material magnitudes must agree;
all tolerances must be declared before a comparison is run.

## Data Availability Audit

The public author repository states that original asset returns and
characteristics cannot be released because of licensing restrictions. It
contains compressed residual-return arrays for Fama-French, PCA, and IPCA, but
does not contain the residual composition matrix files referenced by
`run_train_test.py`. Those matrices are required to map residual allocations to
stock allocations before L1 normalization and transaction-cost calculations.
For the observed Fama-French model, the author's unpublished composition array
maps each residual into a joint underlying space of eligible stocks plus the
traded factor portfolios. It therefore cannot be reconstructed from residual
returns alone, and its absence affects both normalization and holdings
interpretation.

Therefore:

- A run on simulated or public-price data is a **method reproduction**.
- A run on the authors' residual arrays without composition matrices is a
  **residual-space approximation**.
- A run may be called an **exact empirical reproduction** only after the raw
  licensed inputs or equivalent point-in-time inputs are available and the
  residual composition matrices are regenerated.

## Milestones

1. Rolling PCA residuals and OU+Threshold benchmark with no-lookahead tests.
2. Load and validate official compressed residual arrays.
3. Implement Fourier signal plus feedforward allocation.
4. Implement the causal CNN+Transformer architecture.
5. Implement 1,000/125 rolling training and out-of-sample evaluation.
6. Add Fama-French and IPCA residual generation.
7. Reproduce Table I and then market-friction and robustness tables.

## Completion Rule

Every reported result must save:

- source-data identity and checksum;
- universe and missing-data rules;
- random seed and software versions;
- factor model and factor count;
- lookback, training, retraining, and optimizer settings;
- whether stock-space residual composition weights were used;
- gross and net returns, turnover, short proportion, and performance metrics;
- the exact code commit that produced the output.
