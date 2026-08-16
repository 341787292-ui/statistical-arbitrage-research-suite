from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_reproduction.dlsa.backtest import backtest_ou_threshold_streaming
from paper_reproduction.dlsa.data import audit_residual_array, load_numpy_array


DEFAULT_FILENAME = (
    "AvPCA_OOSresiduals_5_factors_1998_initialOOSYear_60_rollingWindow_"
    "252_covWindow_0.01_Cap.npy.gz"
)
PAPER_TARGET = {
    "annualized_sharpe": 0.73,
    "annualized_mean": 0.044,
    "annualized_volatility": 0.061,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper's OU+Threshold policy on official PCA-5 residuals."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("paper_reproduction/data") / DEFAULT_FILENAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_reproduction/output/official_pca5_ou_threshold.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    residuals = load_numpy_array(args.data)
    audit = audit_residual_array(residuals)
    result = backtest_ou_threshold_streaming(
        residuals,
        lookback=30,
        evaluation_start=1000,
        entry_threshold=1.25,
        min_r_squared=0.25,
        zero_is_missing=True,
    )
    observed = {
        "annualized_sharpe": result.annualized_sharpe,
        "annualized_mean": result.annualized_mean,
        "annualized_volatility": result.annualized_volatility,
    }
    report = {
        "status": "residual-space-approximation",
        "reason_not_exact": (
            "The official repository does not publish the residual composition "
            "matrices required for stock-space L1 normalization."
        ),
        "source": {
            "path": str(args.data),
            "sha256": sha256(args.data),
            "audit": audit.__dict__,
        },
        "experiment": {
            "factor_model": "PCA",
            "factor_count": 5,
            "policy": "OU+Threshold",
            "lookback": 30,
            "evaluation_start_row": result.evaluation_start,
            "evaluation_rows": int(result.returns.size),
            "entry_threshold": 1.25,
            "minimum_r_squared": 0.25,
            "zero_encodes_missing": True,
            "stock_space_normalization": result.used_stock_space_normalization,
            "transaction_cost": 0.0,
            "short_holding_cost": 0.0,
        },
        "observed": observed,
        "paper_table_i_target": PAPER_TARGET,
        "difference": {
            key: observed[key] - PAPER_TARGET[key]
            for key in PAPER_TARGET
        },
        "diagnostics": {
            "average_turnover": float(np.mean(result.turnover)),
            "average_short_proportion": float(np.mean(result.short_proportion)),
            "average_active_positions": float(np.mean(result.active_positions)),
            "median_active_positions": float(np.median(result.active_positions)),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
