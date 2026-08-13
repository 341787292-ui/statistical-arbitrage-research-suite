from __future__ import annotations

from quant_research_agent.agent.spec import ExperimentPlan, ExperimentStep, PaperResearchSpec


def create_baseline_plan(spec: PaperResearchSpec) -> ExperimentPlan:
    metrics = spec.evaluation_metrics or [
        "Sharpe ratio",
        "annual return",
        "maximum drawdown",
        "turnover",
    ]
    return ExperimentPlan(
        objective="Create a reproducible baseline for the statistical arbitrage research workflow.",
        reproduction_level="Level 1 - Conceptual baseline",
        steps=[
            ExperimentStep(
                step=1,
                task="Load public equity price data",
                tool="load_price_data",
                rationale="Use public data first so the workflow can be reproduced outside a firm environment.",
            ),
            ExperimentStep(
                step=2,
                task="Select candidate related assets",
                tool="select_candidate_pairs",
                rationale="Correlation and cointegration provide a transparent first approximation of related assets.",
            ),
            ExperimentStep(
                step=3,
                task="Construct a residual spread",
                tool="construct_residual_spread",
                rationale="A residual spread maps the paper's arbitrage portfolio idea into a simple baseline.",
            ),
            ExperimentStep(
                step=4,
                task="Generate mean reversion signals",
                tool="generate_zscore_signal",
                rationale="A z-score signal is interpretable and can later be replaced by a deep learning signal module.",
            ),
            ExperimentStep(
                step=5,
                task="Run a backtest and calculate metrics",
                tool="run_stat_arb_backtest",
                rationale="Quant evidence is required before the Agent can produce research judgments.",
            ),
        ],
        metrics=metrics,
        assumptions=[
            "The MVP prioritizes workflow validity over exact paper reproduction.",
            "Deep learning signal extraction is treated as a planned upgrade after the baseline runs end to end.",
        ],
    )
