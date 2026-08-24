from __future__ import annotations

import argparse
import json
from pathlib import Path

from ashare_stat_arb.config import DEFAULT_CONFIG
from ashare_stat_arb.data_pipeline import load_panel
from ashare_stat_arb.residual_predictability import audit_residual_predictability
from ashare_stat_arb.signals import rolling_monthly_pca_ou_stock_alpha


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit model-free predictability in A-share PCA residuals."
    )
    parser.add_argument(
        "--panel",
        default="ashare_stat_arb/data/baostock_csi500_pilot100_2018_2022.npz",
    )
    parser.add_argument(
        "--output",
        default="ashare_stat_arb/output/baostock_pilot100_residual_predictability.json",
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
    audit = audit_residual_predictability(
        signal.residual_returns,
        panel.dates,
        panel.member,
        horizons=(1, 5, 10, 20),
        development_start=config.periods.development_start,
        development_end=config.periods.development_end,
        validation_start=config.periods.validation_start,
        validation_end=config.periods.validation_end,
        minimum_rank_ic=config.admission.minimum_rank_ic,
        minimum_positive_share=0.50,
        minimum_period_days=60,
        required_stable_horizons=2,
    )
    payload = {
        "label": "a-share-free-data-model-free-residual-predictability-audit",
        "result_scope": "descriptive diagnostic; no trading rule, search, or holdout",
        "data_fingerprint": panel.fingerprint(),
        "audit": audit.to_dict(),
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
