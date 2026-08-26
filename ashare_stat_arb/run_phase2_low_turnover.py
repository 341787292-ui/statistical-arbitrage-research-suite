from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ashare_stat_arb.admission import decide_migration, evaluate_period
from ashare_stat_arb.config import DEFAULT_CONFIG
from ashare_stat_arb.data_pipeline import load_panel
from ashare_stat_arb.panel import audit_panel
from ashare_stat_arb.phase2 import (
    PHASE2_MAPPING,
    conservative_round_trip_cost,
    validate_development_panel,
)
from ashare_stat_arb.research_backtest import LongOnlyBacktestResult, run_long_only_backtest
from ashare_stat_arb.signals import (
    cross_sectional_residual_rank_alpha,
    rolling_monthly_pca_residuals,
)


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
    decisions = period & result.rebalance_decisions
    mandatory = period & result.mandatory_rebalance_decisions
    return {
        "decision_count": int(decisions.sum()),
        "mandatory_decision_count": int(mandatory.sum()),
        "trading_days_with_turnover": int(traded.size),
        "mean_daily_two_way_turnover": float(daily.mean()) if daily.size else 0.0,
        "annualized_two_way_turnover": float(daily.mean() * 252.0) if daily.size else 0.0,
        "mean_turnover_on_trading_days": (
            float(traded.mean()) if traded.size else 0.0
        ),
        "maximum_daily_two_way_turnover": (
            float(traded.max()) if traded.size else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pre-registered Phase 2 low-turnover mapping."
    )
    parser.add_argument(
        "--panel",
        default="ashare_stat_arb/data/baostock_csi500_pilot100_2018_2022.npz",
    )
    parser.add_argument(
        "--output",
        default="ashare_stat_arb/output/baostock_pilot100_phase2_low_turnover.json",
    )
    parser.add_argument(
        "--stage",
        choices=("bounded", "full-universe"),
        default="bounded",
    )
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    panel = load_panel(args.panel)
    validate_development_panel(panel.dates)
    mapping = PHASE2_MAPPING
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
    alpha = cross_sectional_residual_rank_alpha(
        residual_result.residual_returns,
        panel.member,
        horizon=mapping.signal_horizon,
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
        decision_interval=mapping.decision_interval,
        turnover_penalty=cost_penalty,
        force_rebalance_on_universe_change=True,
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
        period="historical_development_2018_2019",
        start_date=development_start,
        end_date=development_end,
    )
    later_development = evaluate_period(
        result,
        panel.dates,
        period="historical_development_2020_2022",
        start_date=validation_start,
        end_date=validation_end,
    )
    overall = evaluate_period(
        result,
        panel.dates,
        period="historical_development_overall",
        start_date=development_start,
        end_date=validation_end,
    )
    decision = decide_migration(
        development,
        later_development,
        overall,
        config.admission,
    )

    label = (
        "a-share-phase2-bounded-engineering-check"
        if args.stage == "bounded"
        else "a-share-phase2-free-full-universe-development"
    )
    payload = {
        "label": label,
        "result_scope": (
            "observed-period Phase 2 development evidence; not a fresh validation, "
            "live strategy, or official CSI 500 performance claim"
        ),
        "data_fingerprint": panel.fingerprint(),
        "panel_audit": audit_panel(panel).to_dict(),
        "research_contract": {
            "signal": "unchanged five-day reverse PCA-residual percentile rank",
            "signal_horizon": mapping.signal_horizon,
            "decision_interval_trading_days": mapping.decision_interval,
            "maximum_two_way_turnover_per_rebalance": (
                mapping.maximum_two_way_turnover
            ),
            "mandatory_turnover_rule": (
                "point-in-time constituent and eligibility exits are executed "
                "outside the 5% discretionary budget; all costs remain charged"
            ),
            "turnover_penalty": cost_penalty,
            "turnover_penalty_basis": "declared pre-2023 round-trip variable cost",
            "holdout": "2023-2025 sealed and rejected by the runner",
            "parameters_searched": False,
        },
        "periods": {
            "early_development": development.to_dict(),
            "later_development": later_development.to_dict(),
            "overall": overall.to_dict(),
        },
        "turnover": {
            "early_development": _turnover_summary(
                result, panel.dates, development_start, development_end
            ),
            "later_development": _turnover_summary(
                result, panel.dates, validation_start, validation_end
            ),
            "overall": _turnover_summary(
                result, panel.dates, development_start, validation_end
            ),
        },
        "mandatory_rebalances": int(result.mandatory_rebalance_decisions.sum()),
        "development_gate_check": decision.to_dict(),
        "holdout_eligible": False,
        "holdout_note": (
            "A development pass requires data-quality and leakage review plus "
            "explicit human approval before one-time holdout access."
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
