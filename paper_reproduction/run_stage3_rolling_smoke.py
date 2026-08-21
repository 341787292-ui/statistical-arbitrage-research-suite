from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_reproduction.dlsa.factor_models import rolling_pca_residuals
from paper_reproduction.dlsa.models import FeedForwardAllocation
from paper_reproduction.dlsa.synthetic import make_factor_market
from paper_reproduction.dlsa.training import rolling_train_test


def main() -> None:
    raw_returns = make_factor_market(
        time_count=460,
        asset_count=12,
        factor_count=2,
        seed=19,
    )
    pca = rolling_pca_residuals(
        raw_returns,
        n_factors=2,
        covariance_window=100,
        loading_window=40,
        store_composition=True,
    )
    first_valid = np.flatnonzero(np.any(pca.active_mask, axis=1))[0]
    residuals = pca.residual_returns[first_valid:]
    compositions = pca.composition_matrices[first_valid:]
    result = rolling_train_test(
        lambda: FeedForwardAllocation(dropout=0.0, seed=19),
        residuals,
        composition_matrices=compositions,
        lookback=30,
        training_length=180,
        retrain_frequency=60,
        temporal_batch_size=60,
        epochs=2,
        feature_type="fourier",
    )
    summary = {
        "status": "rolling-fourier-ffn-smoke-complete",
        "scope": "simulated rolling OOS test; not an empirical reproduction",
        "oos_days": int(len(result.daily_returns)),
        "retrain_origins": result.retrain_origins,
        "annualized_mean": round(result.annualized_mean, 6),
        "annualized_volatility": round(result.annualized_volatility, 6),
        "annualized_sharpe": round(result.annualized_sharpe, 6),
        "finite": bool(np.all(np.isfinite(result.daily_returns))),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
