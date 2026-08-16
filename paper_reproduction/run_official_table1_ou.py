from __future__ import annotations

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


DATASETS = {
    "FamaFrench": {
        "filename": (
            "DailyFamaFrench_OOSresiduals_5_factors_1998_initialOOSYear_"
            "60_rollingWindow_0.01_Cap.npy.gz"
        ),
        "target": {"sharpe": 0.38, "mean": 0.009, "volatility": 0.023},
    },
    "PCA": {
        "filename": (
            "AvPCA_OOSresiduals_5_factors_1998_initialOOSYear_60_rollingWindow_"
            "252_covWindow_0.01_Cap.npy.gz"
        ),
        "target": {"sharpe": 0.73, "mean": 0.044, "volatility": 0.061},
    },
    "IPCA": {
        "filename": (
            "IPCA_DailyOOSresiduals_5_factors_420_initialMonths_240_window_"
            "12_reestimationFreq_0.01_cap.npy.gz"
        ),
        "target": {"sharpe": 0.97, "mean": 0.038, "volatility": 0.040},
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_dataset(name: str, spec: dict[str, object], data_root: Path) -> dict[str, object]:
    path = data_root / str(spec["filename"])
    residuals = load_numpy_array(path)
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
        "sharpe": result.annualized_sharpe,
        "mean": result.annualized_mean,
        "volatility": result.annualized_volatility,
    }
    target = dict(spec["target"])
    return {
        "factor_model": name,
        "factor_count": 5,
        "source": {
            "filename": path.name,
            "sha256": file_sha256(path),
            "audit": audit.__dict__,
        },
        "observed": observed,
        "paper_table_i_target": target,
        "difference": {key: observed[key] - target[key] for key in target},
        "diagnostics": {
            "average_turnover": float(np.mean(result.turnover)),
            "average_short_proportion": float(np.mean(result.short_proportion)),
            "average_active_positions": float(np.mean(result.active_positions)),
            "median_active_positions": float(np.median(result.active_positions)),
        },
    }


def main() -> None:
    data_root = Path("paper_reproduction/data")
    output = Path("paper_reproduction/output/official_table1_ou_threshold.json")
    report = {
        "status": "residual-space-approximation",
        "reason_not_exact": (
            "The official repository omits the residual composition matrices "
            "required for stock-space L1 normalization."
        ),
        "experiment": {
            "policy": "OU+Threshold",
            "lookback": 30,
            "evaluation_start_row": 1000,
            "evaluation_rows": 3781,
            "entry_threshold": 1.25,
            "minimum_r_squared": 0.25,
            "zero_encodes_missing": True,
            "stock_space_normalization": False,
            "transaction_cost": 0.0,
            "short_holding_cost": 0.0,
        },
        "results": [
            run_dataset(name, spec, data_root)
            for name, spec in DATASETS.items()
        ],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
