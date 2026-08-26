from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ashare_stat_arb.admission import decide_migration, evaluate_period
from ashare_stat_arb.config import DEFAULT_CONFIG
from ashare_stat_arb.data_pipeline import load_panel
from ashare_stat_arb.panel import audit_panel
from ashare_stat_arb.research_backtest import run_long_only_backtest
from ashare_stat_arb.signals import (
    cross_sectional_residual_rank_alpha,
    rolling_monthly_pca_residuals,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen five-day CSI 500 residual-rank migration test."
    )
    parser.add_argument(
        "--panel",
        default="ashare_stat_arb/data/baostock_csi500_pilot100_2018_2022.npz",
    )
    parser.add_argument(
        "--output",
        default="ashare_stat_arb/output/baostock_pilot100_residual_rank5_portfolio.json",
    )
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    panel = load_panel(args.panel)
    residual_result = rolling_monthly_pca_residuals(
        panel.adjusted_returns(),
        panel.dates,
        panel.member,
        n_factors=config.signal.baseline_factor_count,
        covariance_window=config.signal.covariance_window,
        loading_window=config.signal.loading_window,
    )
    horizon = 5
    alpha = cross_sectional_residual_rank_alpha(
        residual_result.residual_returns,
        panel.member,
        horizon=horizon,
        minimum_cross_section=20,
    )
    start_index = config.signal.covariance_window + horizon - 1
    result = run_long_only_backtest(
        panel,
        alpha,
        portfolio=config.portfolio,
        costs=config.costs,
        covariance_window=config.signal.loading_window,
        start=start_index,
    )

    panel_start = str(panel.dates[0])
    panel_end = str(panel.dates[-1])
    development_start = max(panel_start, config.periods.development_start)
    development_end = min(panel_end, config.periods.development_end)
    validation_start = max(panel_start, config.periods.validation_start)
    validation_end = min(panel_end, config.periods.validation_end)
    development = evaluate_period(
        result,
        panel.dates,
        period="development",
        start_date=development_start,
        end_date=development_end,
    )
    validation = evaluate_period(
        result,
        panel.dates,
        period="validation",
        start_date=validation_start,
        end_date=validation_end,
    )
    overall = evaluate_period(
        result,
        panel.dates,
        period="development_plus_validation",
        start_date=development_start,
        end_date=validation_end,
    )
    decision = decide_migration(
        development,
        validation,
        overall,
        config.admission,
    )
    active_alpha = alpha[np.abs(alpha) > 0.0]
    payload = {
        "label": "a-share-free-data-five-day-residual-rank-migration-test",
        "deliverable": config.deliverable,
        "result_scope": (
            "public-data method migration evidence; not a live strategy or "
            "official CSI 500 performance claim"
        ),
        "data_fingerprint": panel.fingerprint(),
        "panel_audit": audit_panel(panel).to_dict(),
        "research_contract": {
            "benchmark": config.portfolio.benchmark,
            "signal": "negative trailing five-day PCA residual sum percentile rank",
            "rank_scaling": "cross-sectional centered percentile, unit variance",
            "factor_count": config.signal.baseline_factor_count,
            "horizon": horizon,
            "universe_snapshot_frequency": config.universe_snapshot_frequency,
            "portfolio_decision_frequency": config.portfolio_decision_frequency,
            "holdout": "2023-2025 sealed and not present in this pilot panel",
            "private_factor_library_used": False,
            "explicit_industry_style_constraints_enforced": False,
            "public_disclosure": "generic product description only",
        },
        "signal_coverage": {
            "nonzero_observations": int(active_alpha.size),
            "nonzero_member_share": float(
                (np.abs(alpha[panel.member]) > 0.0).mean()
            ),
            "pca_refits": len(residual_result.refit_dates),
        },
        "periods": {
            "development": development.to_dict(),
            "validation": validation.to_dict(),
            "overall": overall.to_dict(),
        },
        "admission_decision": decision.to_dict(),
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
