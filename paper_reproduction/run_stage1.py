from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_reproduction.dlsa.backtest import backtest_ou_threshold
from paper_reproduction.dlsa.factor_models import rolling_pca_residuals
from paper_reproduction.dlsa.synthetic import make_factor_market


def main() -> None:
    raw_returns = make_factor_market()
    pca = rolling_pca_residuals(
        raw_returns,
        n_factors=3,
        covariance_window=252,
        loading_window=60,
        store_composition=True,
    )
    result = backtest_ou_threshold(
        pca.residual_returns,
        composition_matrices=pca.composition_matrices,
        lookback=30,
        entry_threshold=1.25,
        min_r_squared=0.25,
    )
    valid_residuals = pca.residual_returns[np.isfinite(pca.residual_returns)]
    summary = {
        "status": "paper-aligned-stage-1-complete",
        "data": "simulated factor market; not an empirical reproduction",
        "observations": int(raw_returns.shape[0]),
        "assets": int(raw_returns.shape[1]),
        "pca_factors": 3,
        "valid_residual_observations": int(valid_residuals.size),
        "annualized_mean": round(result.annualized_mean, 6),
        "annualized_volatility": round(result.annualized_volatility, 6),
        "annualized_sharpe": round(result.annualized_sharpe, 6),
        "average_turnover": round(float(result.turnover.mean()), 6),
        "average_short_proportion": round(float(result.short_proportion.mean()), 6),
        "stock_space_normalization": result.used_stock_space_normalization,
    }
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage1_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

