from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ashare_stat_arb.config import DEFAULT_CONFIG
from ashare_stat_arb.data_pipeline import load_panel
from ashare_stat_arb.residual_comparison import compare_residual_definitions
from ashare_stat_arb.signals import rolling_monthly_pca_ou_stock_alpha


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two fixed A-share PCA residual definitions."
    )
    parser.add_argument(
        "--panel",
        default="ashare_stat_arb/data/baostock_csi500_pilot100_2018_2022.npz",
    )
    parser.add_argument(
        "--output",
        default="ashare_stat_arb/output/baostock_pilot100_residual_comparison.json",
    )
    args = parser.parse_args()

    panel = load_panel(args.panel)
    config = DEFAULT_CONFIG
    signal = rolling_monthly_pca_ou_stock_alpha(
        panel.adjusted_returns(),
        panel.dates,
        panel.member,
        n_factors=config.signal.baseline_factor_count,
        covariance_window=config.signal.covariance_window,
        loading_window=config.signal.loading_window,
        residual_lookback=config.signal.residual_lookback,
        entry_threshold=config.signal.ou_entry_threshold,
        min_r_squared=config.signal.ou_min_r_squared,
    )
    start = max(
        int(
            np.searchsorted(
                panel.dates,
                np.datetime64(config.periods.development_start),
            )
        ),
        config.signal.covariance_window + config.signal.residual_lookback - 1,
    )
    comparison = compare_residual_definitions(
        panel.adjusted_returns(),
        panel.dates,
        panel.member,
        signal,
        start=start,
        n_factors=config.signal.baseline_factor_count,
        covariance_window=config.signal.covariance_window,
        loading_window=config.signal.loading_window,
        residual_lookback=config.signal.residual_lookback,
        entry_threshold=config.signal.ou_entry_threshold,
        min_r_squared=config.signal.ou_min_r_squared,
    )
    payload = {
        "label": "a-share-free-data-fixed-residual-definition-comparison",
        "result_scope": "pre-registered diagnostic; no parameter search or holdout",
        "data_fingerprint": panel.fingerprint(),
        "comparison": comparison.to_dict(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
