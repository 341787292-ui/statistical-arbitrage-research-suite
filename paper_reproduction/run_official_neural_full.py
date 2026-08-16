from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_reproduction.dlsa.data import load_numpy_array
from paper_reproduction.dlsa.models import CNNTransformerAllocation, FeedForwardAllocation
from paper_reproduction.dlsa.training import rolling_train_test_streaming


DATASETS = {
    "ff": {
        "filename": "DailyFamaFrench_OOSresiduals_5_factors_1998_initialOOSYear_60_rollingWindow_0.01_Cap.npy.gz",
        "label": "FamaFrench",
    },
    "pca": {
        "filename": "AvPCA_OOSresiduals_5_factors_1998_initialOOSYear_60_rollingWindow_252_covWindow_0.01_Cap.npy.gz",
        "label": "PCA",
    },
    "ipca": {
        "filename": "IPCA_DailyOOSresiduals_5_factors_420_initialMonths_240_window_12_reestimationFreq_0.01_cap.npy.gz",
        "label": "IPCA",
    },
}

TARGETS = {
    "fourier": {
        "ff": {"sharpe": 1.66, "mean": 0.031, "volatility": 0.018},
        "pca": {"sharpe": 1.98, "mean": 0.124, "volatility": 0.063},
        "ipca": {"sharpe": 1.90, "mean": 0.077, "volatility": 0.041},
    },
    "cnn": {
        "ff": {"sharpe": 3.21, "mean": 0.046, "volatility": 0.014},
        "pca": {"sharpe": 3.36, "mean": 0.143, "volatility": 0.042},
        "ipca": {"sharpe": 4.16, "mean": 0.087, "volatility": 0.021},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper's rolling neural experiment.")
    parser.add_argument("--factor-model", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=TARGETS, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-retrains", type=int)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    dataset = DATASETS[args.factor_model]
    source = Path("paper_reproduction/data") / dataset["filename"]
    residuals = load_numpy_array(source)

    if args.model == "fourier":
        model_factory = lambda: FeedForwardAllocation(
            input_size=30,
            hidden_units=(16, 8, 4),
            dropout=0.25,
            seed=args.seed,
        )
        feature_type = "fourier"
    else:
        model_factory = lambda: CNNTransformerAllocation(
            filters=8,
            kernel_size=2,
            attention_heads=4,
            hidden_units=16,
            dropout=0.25,
            seed=args.seed,
        )
        feature_type = "cumsum"

    result = rolling_train_test_streaming(
        model_factory,
        residuals,
        lookback=30,
        training_length=1000,
        retrain_frequency=125,
        temporal_batch_size=125,
        model_chunk_size=args.chunk_size,
        epochs=args.epochs,
        learning_rate=0.001,
        feature_type=feature_type,
        zero_is_missing=True,
        max_retrains=args.max_retrains,
        device=args.device,
    )
    observed = {
        "sharpe": result.annualized_sharpe,
        "mean": result.annualized_mean,
        "volatility": result.annualized_volatility,
    }
    target = TARGETS[args.model][args.factor_model]
    run_name = f"{args.factor_model}_{args.model}_seed{args.seed}"
    output_root = Path("paper_reproduction/output")
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_root / f"{run_name}_daily.npz",
        returns=result.daily_returns,
        turnover=result.turnover,
        short_proportion=result.short_proportion,
    )
    report = {
        "status": (
            "residual-space-approximation"
            if args.max_retrains is None and args.epochs == 100
            else "truncated-official-data-run"
        ),
        "reason_not_exact": (
            "The official repository omits residual composition matrices; "
            "allocations are normalized in residual space."
        ),
        "source": {
            "filename": source.name,
            "sha256": file_sha256(source),
        },
        "experiment": {
            "factor_model": dataset["label"],
            "factor_count": 5,
            "model": args.model,
            "feature_type": feature_type,
            "lookback": 30,
            "training_length": 1000,
            "retrain_frequency": 125,
            "temporal_batch_size": 125,
            "epochs": args.epochs,
            "learning_rate": 0.001,
            "seed": args.seed,
            "device": args.device,
            "stock_space_normalization": False,
            "retrain_origins": result.retrain_origins,
        },
        "observed": observed,
        "paper_table_i_target": target,
        "difference": {key: observed[key] - target[key] for key in target},
        "diagnostics": {
            "oos_days": int(result.daily_returns.size),
            "average_turnover": float(result.turnover.mean()),
            "average_short_proportion": float(result.short_proportion.mean()),
            "training_losses": result.training_losses,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    report_path = output_root / f"{run_name}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
