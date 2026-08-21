from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_reproduction.dlsa.analysis import (
    concatenate_blocks,
    performance_metrics,
    rolling_annualized_sharpe,
)


FACTOR_CODES = {"FamaFrench": "ff", "PCA": "pca", "IPCA": "ipca"}
MODEL_LABELS = {"fourier": "Fourier+FFN", "cnn": "CNN+Transformer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit rolling-window stability for a completed neural run."
    )
    parser.add_argument(
        "--run-name",
        default="pca_fourier_e100_seed0_chunk262144",
    )
    parser.add_argument("--rolling-window", type=int, default=252)
    return parser.parse_args()


def load_periods(
    period_root: Path,
    report: dict,
) -> tuple[list[dict[str, float | int]], dict[str, np.ndarray]]:
    expected_origins = report["experiment"]["retrain_origins"]
    period_files = sorted(
        period_root.glob("origin_*.npz"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    origins = [int(path.stem.split("_")[1]) for path in period_files]
    if origins != expected_origins:
        raise ValueError(
            f"Rolling origins do not match the manifest: {origins} != {expected_origins}"
        )

    return_blocks: list[np.ndarray] = []
    turnover_blocks: list[np.ndarray] = []
    short_blocks: list[np.ndarray] = []
    rows: list[dict[str, float | int]] = []
    source_hash = report["source"]["sha256"]
    for origin, array_path in zip(origins, period_files, strict=True):
        metadata_path = array_path.with_suffix(".json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata checkpoint: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        factor_label = report["experiment"]["factor_model"]
        expected_metadata = {
            "origin": origin,
            "source_sha256": source_hash,
            "epochs": report["experiment"]["epochs"],
            "seed": report["experiment"]["seed"],
            "model": report["experiment"]["model"],
            "factor_model": FACTOR_CODES[factor_label],
            "model_chunk_size": report["experiment"]["model_chunk_size"],
        }
        mismatched = {
            key: (metadata.get(key), value)
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        if mismatched:
            raise ValueError(f"Checkpoint metadata mismatch at {origin}: {mismatched}")

        with np.load(array_path, allow_pickle=False) as saved:
            returns = saved["returns"].astype(np.float64)
            turnover = saved["turnover"].astype(np.float64)
            short_proportion = saved["short_proportion"].astype(np.float64)
        if not (returns.shape == turnover.shape == short_proportion.shape):
            raise ValueError(f"Checkpoint arrays do not align at origin {origin}.")
        metrics = performance_metrics(returns)
        losses = np.asarray(metadata["training_losses"], dtype=np.float64)
        if losses.ndim != 1 or losses.size != int(metadata["epochs"]):
            raise ValueError(f"Training-loss history is incomplete at origin {origin}.")
        rows.append(
            {
                "origin": origin,
                **metrics,
                "average_turnover": float(turnover.mean()),
                "average_short_proportion": float(short_proportion.mean()),
                "initial_training_loss": float(losses[0]),
                "final_training_loss": float(losses[-1]),
                "best_training_loss": float(losses.min()),
                "final_training_score_proxy": float(-losses[-1]),
            }
        )
        return_blocks.append(returns)
        turnover_blocks.append(turnover)
        short_blocks.append(short_proportion)

    return rows, {
        "returns": concatenate_blocks(return_blocks),
        "turnover": concatenate_blocks(turnover_blocks),
        "short_proportion": concatenate_blocks(short_blocks),
    }


def summarize(
    rows: list[dict[str, float | int]],
    arrays: dict[str, np.ndarray],
    report: dict,
    *,
    run_name: str,
    rolling_window: int,
) -> tuple[dict, np.ndarray]:
    returns = arrays["returns"]
    full_metrics = performance_metrics(returns)
    observed = report["observed"]
    metric_pairs = {
        "annualized_mean": "mean",
        "annualized_volatility": "volatility",
        "annualized_sharpe": "sharpe",
    }
    for computed_key, reported_key in metric_pairs.items():
        if not np.isclose(full_metrics[computed_key], observed[reported_key], atol=1e-7):
            raise ValueError(
                f"Recomputed {computed_key} does not match the run manifest."
            )

    full_path = Path("paper_reproduction/output") / f"{run_name}_daily.npz"
    if full_path.exists():
        with np.load(full_path, allow_pickle=False) as saved:
            for key, stitched in arrays.items():
                if not np.array_equal(saved[key].astype(np.float64), stitched):
                    raise ValueError(f"Stitched {key} does not match {full_path}.")

    sharpe_values = np.array([row["annualized_sharpe"] for row in rows], dtype=float)
    training_scores = np.array(
        [row["final_training_score_proxy"] for row in rows],
        dtype=float,
    )
    return_sums = np.array([row["arithmetic_return_sum"] for row in rows], dtype=float)
    split_index = (len(rows) + 1) // 2
    split_day = sum(int(row["days"]) for row in rows[:split_index])
    early_metrics = performance_metrics(returns[:split_day])
    late_metrics = performance_metrics(returns[split_day:])
    rolling_sharpe = rolling_annualized_sharpe(returns, window=rolling_window)
    valid_rolling = rolling_sharpe[np.isfinite(rolling_sharpe)]
    contribution_denominator = float(return_sums.sum())
    contribution = (
        return_sums / contribution_denominator
        if not np.isclose(contribution_denominator, 0.0)
        else np.zeros_like(return_sums)
    )
    ranked = np.argsort(return_sums)[::-1]

    summary = {
        "status": report["status"],
        "run_name": run_name,
        "validation": {
            "window_count": len(rows),
            "full_125_day_windows": int(sum(int(row["days"]) == 125 for row in rows)),
            "partial_windows": int(sum(int(row["days"]) != 125 for row in rows)),
            "all_returns_finite": bool(np.isfinite(returns).all()),
            "stitched_days": int(returns.size),
            "manifest_metrics_recomputed": True,
            "full_arrays_match_period_checkpoints": full_path.exists(),
        },
        "overall": full_metrics,
        "window_stability": {
            "positive_sharpe_windows": int(np.sum(sharpe_values > 0)),
            "nonpositive_sharpe_windows": int(np.sum(sharpe_values <= 0)),
            "median_sharpe": float(np.median(sharpe_values)),
            "sharpe_q1": float(np.quantile(sharpe_values, 0.25)),
            "sharpe_q3": float(np.quantile(sharpe_values, 0.75)),
            "minimum_sharpe": float(sharpe_values.min()),
            "minimum_sharpe_origin": int(rows[int(sharpe_values.argmin())]["origin"]),
            "maximum_sharpe": float(sharpe_values.max()),
            "maximum_sharpe_origin": int(rows[int(sharpe_values.argmax())]["origin"]),
        },
        "half_sample_comparison": {
            "split_after_origin": int(rows[split_index - 1]["origin"]),
            "early_window_count": split_index,
            "late_window_count": len(rows) - split_index,
            "early": early_metrics,
            "late": late_metrics,
            "early_arithmetic_return_contribution": float(contribution[:split_index].sum()),
        },
        "training_diagnostics": {
            "score_definition": "negative final training loss; higher is better",
            "training_score_oos_sharpe_correlation": float(
                np.corrcoef(training_scores, sharpe_values)[0, 1]
            ),
            "early_average_training_score": float(training_scores[:split_index].mean()),
            "late_average_training_score": float(training_scores[split_index:].mean()),
            "windows_with_lower_final_loss": int(
                sum(
                    float(row["final_training_loss"])
                    < float(row["initial_training_loss"])
                    for row in rows
                )
            ),
        },
        "concentration": {
            "top_three_arithmetic_return_share": float(contribution[ranked[:3]].sum()),
            "top_five_arithmetic_return_share": float(contribution[ranked[:5]].sum()),
            "top_windows": [
                {
                    "origin": int(rows[index]["origin"]),
                    "arithmetic_return_share": float(contribution[index]),
                }
                for index in ranked[:5]
            ],
        },
        "rolling_sharpe": {
            "window_days": rolling_window,
            "observation_count": int(valid_rolling.size),
            "median": float(np.median(valid_rolling)),
            "minimum": float(valid_rolling.min()),
            "maximum": float(valid_rolling.max()),
            "positive_fraction": float(np.mean(valid_rolling > 0)),
        },
        "known_limitations": [
            "Residual allocations are normalized in residual space because the "
            "public repository omits stock-space composition matrices.",
            "The published residual arrays do not include a trading-date vector; "
            "the analysis therefore labels periods by rolling origin and OOS day index.",
            "The final rolling origin contains only 31 OOS days, so its standalone "
            "annualized Sharpe is less precise than the 30 complete 125-day windows.",
        ],
    }
    return summary, rolling_sharpe


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_stability(
    path: Path,
    rows: list[dict[str, float | int]],
    returns: np.ndarray,
    rolling_sharpe: np.ndarray,
    *,
    title: str,
) -> None:
    origins = np.array([row["origin"] for row in rows], dtype=int)
    sharpes = np.array([row["annualized_sharpe"] for row in rows], dtype=float)
    positions = np.arange(len(rows))
    wealth = np.cumprod(1.0 + returns)

    figure, axes = plt.subplots(3, 1, figsize=(12, 10))
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.88, hspace=0.48)
    figure.suptitle(
        title,
        y=0.975,
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.945,
        "Residual-space approximation | 100 epochs | 31 rolling origins",
        ha="center",
        color="#4B5563",
        fontsize=10,
    )

    colors = np.where(sharpes >= 0, "#0F766E", "#B45309")
    axes[0].bar(positions, sharpes, color=colors, width=0.78)
    axes[0].axhline(0, color="#111827", linewidth=1)
    axes[0].set_title("Performance decays materially across retraining windows", loc="left")
    axes[0].set_ylabel("Annualized Sharpe")
    tick_positions = positions[::2]
    axes[0].set_xticks(tick_positions, origins[::2], rotation=45, ha="right")
    axes[0].set_xlabel("Rolling origin in the published residual array")

    axes[1].plot(np.arange(returns.size), wealth, color="#1D4ED8", linewidth=1.6)
    axes[1].set_title("Cumulative wealth remains positive but flattens late", loc="left")
    axes[1].set_ylabel("Wealth index (start = 1)")
    axes[1].set_xlabel("Out-of-sample trading-day index")

    axes[2].plot(
        np.arange(rolling_sharpe.size),
        rolling_sharpe,
        color="#C2410C",
        linewidth=1.4,
    )
    axes[2].axhline(0, color="#111827", linewidth=1)
    axes[2].set_title("Trailing 252-day Sharpe confirms late-sample weakening", loc="left")
    axes[2].set_ylabel("Trailing Sharpe")
    axes[2].set_xlabel("Out-of-sample trading-day index")

    for axis in axes:
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_facecolor("#FFFFFF")
    figure.patch.set_facecolor("#FFFFFF")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_root = Path("paper_reproduction/output")
    report_path = output_root / f"{args.run_name}.json"
    period_root = output_root / "periods" / args.run_name
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows, arrays = load_periods(period_root, report)
    summary, rolling_sharpe = summarize(
        rows,
        arrays,
        report,
        run_name=args.run_name,
        rolling_window=args.rolling_window,
    )

    analysis_root = output_root / "analysis"
    csv_path = analysis_root / f"{args.run_name}_window_metrics.csv"
    json_path = analysis_root / f"{args.run_name}_stability.json"
    figure_path = analysis_root / f"{args.run_name}_stability.png"
    write_csv(csv_path, rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    experiment = report["experiment"]
    title = (
        f"{experiment['factor_model']}-{experiment['factor_count']} "
        f"{MODEL_LABELS[experiment['model']]} stability audit"
    )
    plot_stability(
        figure_path,
        rows,
        arrays["returns"],
        rolling_sharpe,
        title=title,
    )
    print(json.dumps(summary, indent=2))
    print(f"window metrics: {csv_path}")
    print(f"stability summary: {json_path}")
    print(f"stability figure: {figure_path}")


if __name__ == "__main__":
    main()
