from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_reproduction.dlsa.factor_models import rolling_pca_residuals
from paper_reproduction.dlsa.models import CNNTransformerAllocation
from paper_reproduction.dlsa.synthetic import make_factor_market
from paper_reproduction.dlsa.training import fit_single_training_window


def main() -> None:
    raw_returns = make_factor_market(
        time_count=430,
        asset_count=12,
        factor_count=2,
        seed=11,
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
    model = CNNTransformerAllocation(seed=11)
    result = fit_single_training_window(
        model,
        residuals,
        composition_matrices=compositions,
        lookback=30,
        epochs=3,
    )
    summary = {
        "status": "cnn-transformer-smoke-complete",
        "scope": "single simulated training window; not rolling OOS reproduction",
        "epochs": len(result.losses),
        "losses": [round(value, 6) for value in result.losses],
        "in_sample_annualized_sharpe": round(result.annualized_sharpe, 6),
        "finite": bool(np.all(np.isfinite(result.daily_returns))),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

