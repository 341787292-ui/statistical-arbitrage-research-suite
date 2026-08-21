from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUNS = [
    "ff_fourier_e100_seed0_chunk262144",
    "pca_fourier_e100_seed0_chunk262144",
    "ipca_fourier_e100_seed0_chunk262144",
]
FACTOR_ORDER = ["FamaFrench", "PCA", "IPCA"]
MATCHED_CONFIG_KEYS = [
    "factor_count",
    "model",
    "feature_type",
    "lookback",
    "training_length",
    "retrain_frequency",
    "temporal_batch_size",
    "model_chunk_size",
    "gradient_checkpointing",
    "epochs",
    "learning_rate",
    "seed",
    "stock_space_normalization",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare completed five-factor Fourier+FFN approximations."
    )
    parser.add_argument("--runs", nargs=3, default=DEFAULT_RUNS)
    return parser.parse_args()


def load_and_validate(
    output_root: Path,
    run_names: list[str],
) -> tuple[list[dict], list[np.ndarray]]:
    records: list[dict] = []
    returns: list[np.ndarray] = []
    reference_config: dict | None = None
    reference_origins: list[int] | None = None
    for run_name in run_names:
        report_path = output_root / f"{run_name}.json"
        stability_path = output_root / "analysis" / f"{run_name}_stability.json"
        daily_path = output_root / f"{run_name}_daily.npz"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        stability = json.loads(stability_path.read_text(encoding="utf-8"))
        if report["status"] != "residual-space-approximation":
            raise ValueError(f"{run_name} is not a complete residual-space approximation.")
        config = {key: report["experiment"][key] for key in MATCHED_CONFIG_KEYS}
        origins = report["experiment"]["retrain_origins"]
        if reference_config is None:
            reference_config = config
            reference_origins = origins
        elif config != reference_config or origins != reference_origins:
            raise ValueError(f"Experiment configuration mismatch for {run_name}.")
        with np.load(daily_path, allow_pickle=False) as saved:
            daily_returns = saved["returns"].astype(np.float64)
        if daily_returns.size != stability["validation"]["stitched_days"]:
            raise ValueError(f"Daily result length does not match audit for {run_name}.")
        returns.append(daily_returns)
        records.append(
            {
                "run_name": run_name,
                "factor_model": report["experiment"]["factor_model"],
                "observed_sharpe": report["observed"]["sharpe"],
                "paper_sharpe": report["paper_table_i_target"]["sharpe"],
                "observed_mean": report["observed"]["mean"],
                "paper_mean": report["paper_table_i_target"]["mean"],
                "observed_volatility": report["observed"]["volatility"],
                "paper_volatility": report["paper_table_i_target"]["volatility"],
                "early_sharpe": stability["half_sample_comparison"]["early"][
                    "annualized_sharpe"
                ],
                "late_sharpe": stability["half_sample_comparison"]["late"][
                    "annualized_sharpe"
                ],
                "positive_sharpe_windows": stability["window_stability"][
                    "positive_sharpe_windows"
                ],
                "training_oos_correlation": stability["training_diagnostics"][
                    "training_score_oos_sharpe_correlation"
                ],
            }
        )
    records.sort(key=lambda record: FACTOR_ORDER.index(record["factor_model"]))
    ordered_returns = [
        returns[run_names.index(record["run_name"])]
        for record in records
    ]
    return records, ordered_returns


def build_summary(records: list[dict], returns: list[np.ndarray]) -> dict:
    relative_errors: dict[str, dict[str, float]] = {}
    for record in records:
        errors = {
            metric: (
                record[f"observed_{metric}"] - record[f"paper_{metric}"]
            )
            / record[f"paper_{metric}"]
            for metric in ("sharpe", "mean", "volatility")
        }
        errors["mean_absolute_relative_error"] = float(
            np.mean(np.abs(list(errors.values())))
        )
        relative_errors[record["factor_model"]] = errors

    observed_ranking = sorted(
        records,
        key=lambda record: record["observed_sharpe"],
        reverse=True,
    )
    paper_ranking = sorted(
        records,
        key=lambda record: record["paper_sharpe"],
        reverse=True,
    )
    correlations = np.corrcoef(returns)
    return {
        "status": "residual-space-approximation",
        "runs": [record["run_name"] for record in records],
        "comparison": records,
        "relative_errors": relative_errors,
        "closest_factor_model": min(
            relative_errors,
            key=lambda factor: relative_errors[factor]["mean_absolute_relative_error"],
        ),
        "sharpe_ranking": {
            "observed": [record["factor_model"] for record in observed_ranking],
            "paper": [record["factor_model"] for record in paper_ranking],
            "preserved": [record["factor_model"] for record in observed_ranking]
            == [record["factor_model"] for record in paper_ranking],
        },
        "daily_return_correlations": {
            records[i]["factor_model"]: {
                records[j]["factor_model"]: float(correlations[i, j])
                for j in range(len(records))
            }
            for i in range(len(records))
        },
        "shared_finding": (
            "All three factor models have a lower Sharpe in the final 15 "
            "rolling windows than in the first 16."
        ),
        "known_limitation": (
            "The public repository omits residual composition matrices, so all "
            "allocations are normalized in residual space rather than stock space."
        ),
    }


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def add_labels(axis: plt.Axes, bars, *, percentage: bool = False) -> None:
    for bar in bars:
        value = bar.get_height()
        label = f"{value * 100:.1f}%" if percentage else f"{value:.2f}"
        offset = 3 if value >= 0 else -4
        axis.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )


def plot_comparison(path: Path, records: list[dict]) -> None:
    labels = [record["factor_model"] for record in records]
    x = np.arange(len(labels))
    width = 0.34
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.86, hspace=0.36)
    figure.suptitle(
        "Five-factor Fourier+FFN public-data comparison",
        y=0.97,
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.935,
        "100 epochs | 31 rolling origins | residual-space normalization",
        ha="center",
        color="#4B5563",
        fontsize=10,
    )

    chart_specs = [
        (axes[0, 0], "sharpe", "Sharpe ratio", False),
        (axes[0, 1], "mean", "Annualized mean return", True),
        (axes[1, 0], "volatility", "Annualized volatility", True),
    ]
    for axis, metric, title, percentage in chart_specs:
        observed = [record[f"observed_{metric}"] for record in records]
        paper = [record[f"paper_{metric}"] for record in records]
        observed_bars = axis.bar(
            x - width / 2,
            observed,
            width,
            label="Observed approximation",
            color="#0F766E",
        )
        paper_bars = axis.bar(
            x + width / 2,
            paper,
            width,
            label="Paper Table I",
            color="#6B7280",
        )
        add_labels(axis, observed_bars, percentage=percentage)
        add_labels(axis, paper_bars, percentage=percentage)
        axis.set_title(title, loc="left")
        axis.set_xticks(x, labels)
        axis.set_ylim(bottom=0)

    early = [record["early_sharpe"] for record in records]
    late = [record["late_sharpe"] for record in records]
    early_bars = axes[1, 1].bar(
        x - width / 2,
        early,
        width,
        label="First 16 windows",
        color="#1D4ED8",
    )
    late_bars = axes[1, 1].bar(
        x + width / 2,
        late,
        width,
        label="Final 15 windows",
        color="#C2410C",
    )
    add_labels(axes[1, 1], early_bars)
    add_labels(axes[1, 1], late_bars)
    axes[1, 1].axhline(0, color="#111827", linewidth=1)
    axes[1, 1].set_title("All factor models weaken late", loc="left")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylim(bottom=min(early + late + [0]) - 0.65)

    for axis in axes.flat:
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_facecolor("#FFFFFF")
        axis.legend(frameon=False, fontsize=9, loc="upper right")
    figure.patch.set_facecolor("#FFFFFF")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_root = Path("paper_reproduction/output")
    records, returns = load_and_validate(output_root, args.runs)
    summary = build_summary(records, returns)
    analysis_root = output_root / "analysis"
    csv_path = analysis_root / "fourier_5factor_comparison.csv"
    json_path = analysis_root / "fourier_5factor_comparison.json"
    figure_path = analysis_root / "fourier_5factor_comparison.png"
    write_csv(csv_path, records)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_comparison(figure_path, records)
    print(json.dumps(summary, indent=2))
    print(f"comparison table: {csv_path}")
    print(f"comparison summary: {json_path}")
    print(f"comparison figure: {figure_path}")


if __name__ == "__main__":
    main()
