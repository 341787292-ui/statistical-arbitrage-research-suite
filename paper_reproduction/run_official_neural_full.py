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
    parser.add_argument("--start-origin", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
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
    source_hash = file_sha256(source)

    if args.model == "fourier":
        model_factory = lambda: FeedForwardAllocation(
            input_size=30,
            hidden_units=(16, 8, 4),
            dropout=0.25,
            seed=args.seed,
        )
        feature_type = "fourier"
        gradient_checkpointing = False
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
        gradient_checkpointing = True

    expected_origins = list(range(1000, residuals.shape[0], 125))
    selected_origins = [origin for origin in expected_origins if origin >= args.start_origin]
    origins = (
        selected_origins
        if args.max_retrains is None
        else selected_origins[: args.max_retrains]
    )
    if not origins:
        raise ValueError("The experiment must include at least one retraining origin.")

    run_name = (
        f"{args.factor_model}_{args.model}_e{args.epochs}_seed{args.seed}_"
        f"chunk{args.chunk_size}"
    )
    output_root = Path("paper_reproduction/output")
    period_root = output_root / "periods" / run_name
    period_root.mkdir(parents=True, exist_ok=True)
    return_blocks: list[np.ndarray] = []
    turnover_blocks: list[np.ndarray] = []
    short_blocks: list[np.ndarray] = []
    training_losses: list[list[float]] = []

    for period_number, origin in enumerate(origins, start=1):
        period_data = residuals[origin - 1000 : min(origin + 125, residuals.shape[0])]
        array_path = period_root / f"origin_{origin}.npz"
        metadata_path = period_root / f"origin_{origin}.json"
        if array_path.exists() and metadata_path.exists() and not args.force:
            period_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_metadata = {
                "source_sha256": source_hash,
                "epochs": args.epochs,
                "seed": args.seed,
                "model": args.model,
                "factor_model": args.factor_model,
                "model_chunk_size": args.chunk_size,
            }
            mismatched = {
                key: (period_metadata.get(key), value)
                for key, value in expected_metadata.items()
                if period_metadata.get(key) != value
            }
            if mismatched:
                raise ValueError(
                    f"Checkpoint metadata mismatch at origin {origin}: {mismatched}. "
                    "Use --force to recompute it."
                )
            with np.load(array_path, allow_pickle=False) as saved:
                period_returns = saved["returns"]
                period_turnover = saved["turnover"]
                period_short = saved["short_proportion"]
            period_losses = period_metadata["training_losses"]
            print(f"resumed origin {origin} ({period_number}/{len(origins)})", flush=True)
        else:
            result = rolling_train_test_streaming(
                model_factory,
                period_data,
                lookback=30,
                training_length=1000,
                retrain_frequency=125,
                temporal_batch_size=125,
                model_chunk_size=args.chunk_size,
                gradient_checkpointing=gradient_checkpointing,
                epochs=args.epochs,
                learning_rate=0.001,
                feature_type=feature_type,
                zero_is_missing=True,
                max_retrains=1,
                device=args.device,
            )
            period_returns = result.daily_returns
            period_turnover = result.turnover
            period_short = result.short_proportion
            period_losses = result.training_losses[0]
            np.savez_compressed(
                array_path,
                returns=period_returns,
                turnover=period_turnover,
                short_proportion=period_short,
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        "origin": origin,
                        "source_sha256": source_hash,
                        "epochs": args.epochs,
                        "seed": args.seed,
                        "model": args.model,
                        "factor_model": args.factor_model,
                        "model_chunk_size": args.chunk_size,
                        "training_losses": period_losses,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"completed origin {origin} ({period_number}/{len(origins)})", flush=True)
        return_blocks.append(period_returns)
        turnover_blocks.append(period_turnover)
        short_blocks.append(period_short)
        training_losses.append(period_losses)

    daily_returns = np.concatenate(return_blocks)
    turnover = np.concatenate(turnover_blocks)
    short_proportion = np.concatenate(short_blocks)
    annualized_mean = float(daily_returns.mean() * 252)
    annualized_volatility = float(daily_returns.std(ddof=0) * np.sqrt(252))
    annualized_sharpe = (
        annualized_mean / annualized_volatility if annualized_volatility > 0 else 0.0
    )
    observed = {
        "sharpe": annualized_sharpe,
        "mean": annualized_mean,
        "volatility": annualized_volatility,
    }
    target = TARGETS[args.model][args.factor_model]
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_root / f"{run_name}_daily.npz",
        returns=daily_returns,
        turnover=turnover,
        short_proportion=short_proportion,
    )
    report = {
        "status": (
            "residual-space-approximation"
            if origins == expected_origins and args.epochs == 100
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
            "model_chunk_size": args.chunk_size,
            "gradient_checkpointing": gradient_checkpointing,
            "epochs": args.epochs,
            "learning_rate": 0.001,
            "seed": args.seed,
            "device": args.device,
            "stock_space_normalization": False,
            "retrain_origins": origins,
        },
        "observed": observed,
        "paper_table_i_target": target,
        "difference": {key: observed[key] - target[key] for key in target},
        "diagnostics": {
            "oos_days": int(daily_returns.size),
            "average_turnover": float(turnover.mean()),
            "average_short_proportion": float(short_proportion.mean()),
            "training_losses": training_losses,
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
