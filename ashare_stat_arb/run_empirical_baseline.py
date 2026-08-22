from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ashare_stat_arb.config import DEFAULT_CONFIG
from ashare_stat_arb.data_pipeline import load_panel
from ashare_stat_arb.panel import audit_panel
from ashare_stat_arb.research_backtest import run_long_only_backtest
from ashare_stat_arb.signals import rolling_monthly_pca_ou_stock_alpha


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an A-share CSI 500 PCA-OU baseline.")
    parser.add_argument("--panel", default="ashare_stat_arb/data/csi500_panel.npz")
    parser.add_argument("--output", default="ashare_stat_arb/output/pca_ou_baseline.json")
    parser.add_argument("--factors", type=int, default=DEFAULT_CONFIG.signal.baseline_factor_count)
    args = parser.parse_args()

    panel = load_panel(args.panel)
    config = DEFAULT_CONFIG
    signal = rolling_monthly_pca_ou_stock_alpha(
        panel.adjusted_returns(),
        panel.dates,
        panel.member,
        n_factors=args.factors,
        covariance_window=config.signal.covariance_window,
        loading_window=config.signal.loading_window,
        residual_lookback=config.signal.residual_lookback,
        entry_threshold=config.signal.ou_entry_threshold,
        min_r_squared=config.signal.ou_min_r_squared,
    )
    development_start = np.datetime64(config.periods.development_start)
    start_index = max(
        int(np.searchsorted(panel.dates, development_start)),
        config.signal.covariance_window + config.signal.residual_lookback - 1,
    )
    result = run_long_only_backtest(
        panel,
        signal.stock_alpha,
        portfolio=config.portfolio,
        costs=config.costs,
        covariance_window=config.signal.loading_window,
        start=start_index,
    )
    panel_path = Path(args.panel)
    manifest_path = panel_path.with_suffix(".manifest.json")
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    active = signal.active_count[signal.active_count > 0]
    payload = {
        "label": "a-share-pca-ou-long-only-feasibility-baseline",
        "result_scope": "research feasibility only; not an investable performance claim",
        "panel_metadata": manifest.get("metadata", {}),
        "data_fingerprint": panel.fingerprint(),
        "panel_audit": audit_panel(panel).to_dict(),
        "factor_count": args.factors,
        "pca_refits": len(signal.refit_dates),
        "average_active_assets": float(active.mean()) if active.size else 0.0,
        "annualized_return": result.annualized_return,
        "annualized_gross_return": result.annualized_gross_return,
        "annualized_benchmark_return": result.annualized_benchmark_return,
        "annualized_sharpe": result.annualized_sharpe,
        "annualized_excess_return": result.annualized_excess_return,
        "annualized_gross_excess_return": result.annualized_gross_excess_return,
        "information_ratio": result.information_ratio,
        "gross_information_ratio": result.gross_information_ratio,
        "annualized_cost_drag": result.annualized_cost_drag,
        "maximum_active_drawdown": result.maximum_active_drawdown,
        "average_two_way_turnover": float(result.two_way_turnover.mean()),
        "total_cost": float(result.costs.sum()),
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
