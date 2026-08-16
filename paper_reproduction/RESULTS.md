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

## Official-data neural smoke tests

The following runs use one epoch and one rolling origin only. They establish
that the full official-data pipeline executes; they are not performance claims.

| Factor model | Model | Device | OOS days | Sharpe |
|---|---|---|---:|---:|
| PCA | Fourier+FFN | CPU | 125 | -0.488 |
| PCA | CNN+Transformer | CPU | 125 | 1.377 |

The formal neural comparison must use 100 epochs and all rolling origins.
