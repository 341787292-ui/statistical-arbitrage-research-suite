from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_reproduction.dlsa.data import load_numpy_array
from paper_reproduction.dlsa.models import CNNTransformerAllocation, FeedForwardAllocation
from paper_reproduction.dlsa.training import rolling_train_test_streaming


DATA_FILES = {
    "ff": "DailyFamaFrench_OOSresiduals_5_factors_1998_initialOOSYear_60_rollingWindow_0.01_Cap.npy.gz",
    "pca": "AvPCA_OOSresiduals_5_factors_1998_initialOOSYear_60_rollingWindow_252_covWindow_0.01_Cap.npy.gz",
    "ipca": "IPCA_DailyOOSresiduals_5_factors_420_initialMonths_240_window_12_reestimationFreq_0.01_cap.npy.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test neural models on official residuals.")
    parser.add_argument("--factor-model", choices=DATA_FILES, default="pca")
    parser.add_argument("--model", choices=("fourier", "cnn"), default="fourier")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-retrains", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path("paper_reproduction/data") / DATA_FILES[args.factor_model]
    residuals = load_numpy_array(path)
    if args.model == "fourier":
        model_factory = lambda: FeedForwardAllocation(
            input_size=30,
            hidden_units=(16, 8, 4),
            dropout=0.25,
            seed=0,
        )
        feature_type = "fourier"
    else:
        model_factory = lambda: CNNTransformerAllocation(
            filters=8,
            kernel_size=2,
            attention_heads=4,
            hidden_units=16,
            dropout=0.25,
            seed=0,
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
    report = {
        "status": "official-data-neural-smoke-complete",
        "scope": (
            "Residual-space approximation with truncated training. "
            "This is not a Table I result."
        ),
        "factor_model": args.factor_model,
        "model": args.model,
        "device": args.device,
        "epochs": args.epochs,
        "retrain_origins": result.retrain_origins,
        "oos_days": int(result.daily_returns.size),
        "annualized_mean": result.annualized_mean,
        "annualized_volatility": result.annualized_volatility,
        "annualized_sharpe": result.annualized_sharpe,
        "average_turnover": float(np.mean(result.turnover)),
        "average_short_proportion": float(np.mean(result.short_proportion)),
        "training_losses": result.training_losses,
        "finite": bool(np.all(np.isfinite(result.daily_returns))),
        "stock_space_normalization": False,
    }
    output = Path(
        f"paper_reproduction/output/{args.factor_model}_{args.model}_official_smoke.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
