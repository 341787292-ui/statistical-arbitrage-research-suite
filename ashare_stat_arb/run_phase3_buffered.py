from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ashare_stat_arb.admission import decide_migration, evaluate_period
from ashare_stat_arb.config import DEFAULT_CONFIG
from ashare_stat_arb.data_pipeline import load_panel
from ashare_stat_arb.panel import audit_panel
from ashare_stat_arb.phase2 import conservative_round_trip_cost, validate_development_panel
from ashare_stat_arb.phase3 import PHASE3_MAPPING
from ashare_stat_arb.research_backtest import LongOnlyBacktestResult, run_long_only_backtest
from ashare_stat_arb.signals import buffered_residual_rank_alpha, rolling_monthly_pca_residuals


def _turnover_summary(
    result: LongOnlyBacktestResult,
    dates: np.ndarray,
    start_date: str,
    end_date: str,
) -> dict[str, float | int]:
    trading_dates = np.asarray(dates, dtype="datetime64[D]")
    period = (
        (trading_dates >= np.datetime64(start_date))
        & (trading_dates <= np.datetime64(end_date))
        & np.isfinite(result.strategy_returns)
    )
    daily = result.two_way_turnover[period]
    traded = daily[daily > 1e-12]
    return {
        "decision_count": int((period & result.rebalance_decisions).sum()),
        "mandatory_decision_count": int(
            (period & result.mandatory_rebalance_decisions).sum()
        ),
        "trading_days_with_turnover": int(traded.size),
        "mean_daily_two_way_turnover": float(daily.mean()) if daily.size else 0.0,
        "annualized_two_way_turnover": float(daily.mean() * 252.0) if daily.size else 0.0,
        "mean_turnover_on_trading_days": float(traded.mean()) if traded.size else 0.0,
        "maximum_daily_two_way_turnover": float(traded.max()) if traded.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pre-registered Phase 3 buffered daily mapping."
    )
    parser.add_argument(
        "--panel",
        default="ashare_stat_arb/data/baostock_csi500_pilot100_2018_2022.npz",
    )
    parser.add_argument(
        "--output",
        default="ashare_stat_arb/output/baostock_pilot100_phase3_buffered.json",
    )
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    mapping = PHASE3_MAPPING
    panel = load_panel(args.panel)
    validate_development_panel(panel.dates)
    portfolio = mapping.portfolio(config.portfolio)
    cost_penalty = conservative_round_trip_cost(config.costs)

    residual_result = rolling_monthly_pca_residuals(
        panel.adjusted_returns(),
        panel.dates,
        panel.member,
        n_factors=config.signal.baseline_factor_count,
        covariance_window=config.signal.covariance_window,
        loading_window=config.signal.loading_window,
    )
    alpha = buffered_residual_rank_alpha(
        residual_result.residual_returns,
        panel.member,
        horizon=mapping.signal_horizon,
        positive_entry=mapping.positive_entry,
        positive_exit=mapping.positive_exit,
        negative_entry=mapping.negative_entry,
        negative_exit=mapping.negative_exit,
        minimum_cross_section=20,
    )
    start_index = config.signal.covariance_window + mapping.signal_horizon - 1
    result = run_long_only_backtest(
        panel,
        alpha,
        portfolio=portfolio,
        costs=config.costs,
        covariance_window=config.signal.loading_window,
        start=start_index,
        decision_interval=1,
        turnover_penalty=cost_penalty,
        force_rebalance_on_universe_change=True,
    )

    panel_start = str(panel.dates[0])
    panel_end = str(panel.dates[-1])
    early_start = max(panel_start, config.periods.development_start)
    early_end = min(panel_end, config.periods.development_end)
    later_start = max(panel_start, config.periods.validation_start)
    later_end = min(panel_end, config.periods.validation_end)
    early = evaluate_period(
        result,
        panel.dates,
        period="historical_development_2018_2019",
        start_date=early_start,
        end_date=early_end,
    )
    later = evaluate_period(
        result,
        panel.dates,
        period="historical_development_2020_2022",
        start_date=later_start,
        end_date=later_end,
    )
    overall = evaluate_period(
        result,
        panel.dates,
        period="historical_development_overall",
        start_date=early_start,
        end_date=later_end,
    )
    decision = decide_migration(early, later, overall, config.admission)

    payload = {
        "label": "a-share-phase3-buffered-bounded-development",
        "result_scope": (
            "observed-period buffered-mapping development evidence; not a fresh "
            "validation, live strategy, or official CSI 500 performance claim"
        ),
        "data_fingerprint": panel.fingerprint(),
        "panel_audit": audit_panel(panel).to_dict(),
        "research_contract": {
            "underlying_signal": "five-day reverse PCA-residual rank",
            "positive_entry": mapping.positive_entry,
            "positive_exit": mapping.positive_exit,
            "negative_entry": mapping.negative_entry,
            "negative_exit": mapping.negative_exit,
            "decision_interval_trading_days": 1,
            "maximum_discretionary_two_way_turnover": (
                mapping.maximum_two_way_turnover
            ),
            "turnover_penalty": cost_penalty,
            "parameters_searched": False,
            "holdout": "2023-2025 sealed and rejected by the runner",
        },
        "signal_state_share": {
            "positive": float((alpha[panel.member] > 0.0).mean()),
            "negative": float((alpha[panel.member] < 0.0).mean()),
            "zero": float((alpha[panel.member] == 0.0).mean()),
        },
        "periods": {
            "early_development": early.to_dict(),
            "later_development": later.to_dict(),
            "overall": overall.to_dict(),
        },
        "turnover": {
            "early_development": _turnover_summary(
                result, panel.dates, early_start, early_end
            ),
            "later_development": _turnover_summary(
                result, panel.dates, later_start, later_end
            ),
            "overall": _turnover_summary(result, panel.dates, early_start, later_end),
        },
        "development_gate_check": decision.to_dict(),
        "holdout_eligible": False,
        "holdout_note": (
            "Only a passing full-universe development run plus explicit human "
            "approval can authorize one-time holdout access."
        ),
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
