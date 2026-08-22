from __future__ import annotations

import json

from ashare_stat_arb.panel import audit_panel
from ashare_stat_arb.research_backtest import run_long_only_backtest
from ashare_stat_arb.signals import rolling_monthly_pca_ou_stock_alpha
from ashare_stat_arb.synthetic import make_synthetic_csi500_panel


def main() -> None:
    panel = make_synthetic_csi500_panel()
    signal = rolling_monthly_pca_ou_stock_alpha(
        panel.adjusted_returns(),
        panel.dates,
        panel.member,
        n_factors=3,
        covariance_window=120,
        loading_window=60,
        residual_lookback=30,
        min_r_squared=0.10,
    )
    result = run_long_only_backtest(
        panel,
        signal.stock_alpha,
        covariance_window=60,
        start=150,
    )
    payload = {
        "label": "synthetic-engineering-baseline-not-an-investment-result",
        "panel_audit": audit_panel(panel).to_dict(),
        "pca_refits": len(signal.refit_dates),
        "average_active_assets": float(signal.active_count[signal.active_count > 0].mean()),
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
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
