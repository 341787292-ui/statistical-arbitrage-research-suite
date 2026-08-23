from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ashare_stat_arb.config import DEFAULT_CONFIG
from ashare_stat_arb.data_pipeline import load_panel
from ashare_stat_arb.diagnostics import (
    forward_rank_ic_by_horizon,
    summarize_signal_coverage,
)
from ashare_stat_arb.research_backtest import LongOnlyBacktestResult, run_long_only_backtest
from ashare_stat_arb.signals import rolling_monthly_pca_ou_stock_alpha


def _performance(result: LongOnlyBacktestResult) -> dict[str, float]:
    return {
        "annualized_return": result.annualized_return,
        "annualized_gross_return": result.annualized_gross_return,
        "annualized_benchmark_return": result.annualized_benchmark_return,
        "annualized_excess_return": result.annualized_excess_return,
        "annualized_gross_excess_return": result.annualized_gross_excess_return,
        "annualized_sharpe": result.annualized_sharpe,
        "information_ratio": result.information_ratio,
        "gross_information_ratio": result.gross_information_ratio,
        "annualized_cost_drag": result.annualized_cost_drag,
        "average_two_way_turnover": float(result.two_way_turnover.mean()),
        "maximum_active_drawdown": result.maximum_active_drawdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose A-share PCA-OU signal direction.")
    parser.add_argument(
        "--panel",
        default="ashare_stat_arb/data/baostock_csi500_pilot100_2018_2022.npz",
    )
    parser.add_argument(
        "--output",
        default="ashare_stat_arb/output/baostock_pilot100_signal_diagnostics.json",
    )
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
    start_index = max(
        int(np.searchsorted(panel.dates, np.datetime64(config.periods.development_start))),
        config.signal.covariance_window + config.signal.residual_lookback - 1,
    )
    common = {
        "portfolio": config.portfolio,
        "costs": config.costs,
        "covariance_window": config.signal.loading_window,
        "start": start_index,
    }
    original = run_long_only_backtest(panel, signal.stock_alpha, **common)
    reversed_result = run_long_only_backtest(panel, -signal.stock_alpha, **common)
    neutral = run_long_only_backtest(panel, np.zeros_like(signal.stock_alpha), **common)
    rank_ic = forward_rank_ic_by_horizon(
        signal.stock_alpha,
        panel,
        horizons=(1, 5, 10, 20),
        start=start_index,
    )
    gross_results = {
        "original": original.annualized_gross_excess_return,
        "reversed": reversed_result.annualized_gross_excess_return,
        "neutral": neutral.annualized_gross_excess_return,
    }
    best_direction = max(gross_results, key=gross_results.get)
    payload = {
        "label": "a-share-free-data-ou-direction-diagnostic",
        "result_scope": "diagnostic only; no parameters selected from these results",
        "data_fingerprint": panel.fingerprint(),
        "start_date": str(panel.dates[start_index]),
        "factor_count": args.factors,
        "ou_entry_threshold": config.signal.ou_entry_threshold,
        "ou_min_r_squared": config.signal.ou_min_r_squared,
        "signal_coverage": summarize_signal_coverage(
            signal.stock_alpha,
            panel.member,
            start=start_index,
        ).to_dict(),
        "forward_rank_ic": rank_ic,
        "portfolio_results": {
            "original": _performance(original),
            "reversed": _performance(reversed_result),
            "neutral": _performance(neutral),
        },
        "direction_assessment": {
            "best_gross_excess_variant": best_direction,
            "gross_excess_original_minus_neutral": (
                original.annualized_gross_excess_return
                - neutral.annualized_gross_excess_return
            ),
            "gross_excess_reversed_minus_neutral": (
                reversed_result.annualized_gross_excess_return
                - neutral.annualized_gross_excess_return
            ),
            "all_reported_rank_ic_is_for_original_direction": True,
        },
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
