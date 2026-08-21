from __future__ import annotations


def analyze_experiment_result(experiment_result: dict) -> dict:
    metrics = experiment_result.get("metrics", {})
    diagnostics = experiment_result.get("diagnostics", {})
    sharpe = float(metrics.get("sharpe", 0.0))
    annual_return = float(metrics.get("annual_return", 0.0))
    max_drawdown = float(metrics.get("max_drawdown", 0.0))
    average_turnover = float(diagnostics.get("average_turnover", 0.0))

    observations: list[str] = []
    if sharpe > 1.0:
        observations.append(
            "The baseline produces positive risk-adjusted performance on the deterministic pair sample."
        )
    elif sharpe > 0:
        observations.append(
            "The baseline is positive but not yet strong enough to support a robust research conclusion."
        )
    else:
        observations.append(
            "The baseline does not produce positive risk-adjusted performance under the current settings."
        )

    observations.append(
        f"Annual return is {annual_return:.2%}, Sharpe is {sharpe:.2f}, "
        f"and maximum drawdown is {max_drawdown:.2%}."
    )
    observations.append(
        f"Average turnover is {average_turnover:.2%}, so transaction cost robustness should be checked."
    )

    hypotheses = [
        {
            "hypothesis": "The signal is capturing short-term residual mean reversion.",
            "confidence": "medium" if sharpe > 1.0 else "low",
            "supporting_evidence": [
                "The strategy uses residual spread z-score thresholds.",
                "The baseline remains profitable after applying transaction costs."
                if sharpe > 1.0
                else "Current results do not yet strongly support this mechanism.",
            ],
            "validation_needed": "Run rolling-window performance and signal decay analysis.",
        },
        {
            "hypothesis": "Performance may be sensitive to transaction cost and exit threshold choices.",
            "confidence": "medium",
            "supporting_evidence": [
                "Turnover is non-zero and the strategy enters/exits positions repeatedly."
            ],
            "validation_needed": "Run a grid over transaction costs, entry thresholds, and exit thresholds.",
        },
    ]

    recommended_experiments = [
        {
            "experiment": "Cost sensitivity test",
            "rationale": "Increase transaction cost assumptions and observe whether Sharpe remains positive.",
        },
        {
            "experiment": "Rolling-period analysis",
            "rationale": "Split the sample into subperiods to test whether performance is stable or regime-dependent.",
        },
        {
            "experiment": "Signal-module replacement",
            "rationale": "Replace the z-score signal with a learned signal later while preserving the same backtest interface.",
        },
    ]

    return {
        "observations": observations,
        "hypotheses": hypotheses,
        "recommended_experiments": recommended_experiments,
    }


def reflect_on_validations(
    experiment_result: dict,
    validation_results: list[dict],
) -> dict:
    """Update the research judgment after executable validation tests complete."""
    baseline_sharpe = float(experiment_result.get("metrics", {}).get("sharpe", 0.0))
    evidence: list[str] = [f"Baseline Sharpe is {baseline_sharpe:.2f} on controlled data."]
    limitations = [
        "All experiments currently use deterministic synthetic prices.",
        "The result demonstrates workflow execution, not a tradable real-market edge.",
    ]
    cost_robust = False
    period_robust = False

    for result in validation_results:
        if result.get("experiment") == "cost_sensitivity":
            summary = result.get("summary", {})
            cost_robust = bool(summary.get("survives_highest_cost"))
            evidence.append(
                "The cost test kept positive Sharpe at the highest tested cost."
                if cost_robust
                else "The cost test lost positive Sharpe at the highest tested cost."
            )
        elif result.get("experiment") == "period_stability":
            summary = result.get("summary", {})
            period_robust = bool(
                summary.get("both_periods_positive")
                and summary.get("stable_within_threshold")
            )
            evidence.append(
                "Both controlled subperiods were positive with an acceptable Sharpe gap."
                if period_robust
                else "Subperiod results were unstable or included a non-positive Sharpe."
            )

    if baseline_sharpe > 0 and cost_robust and period_robust:
        verdict = "The controlled baseline supports the mean-reversion mechanism under the tested checks."
        confidence = "medium"
    elif baseline_sharpe > 0:
        verdict = "The controlled baseline is positive, but robustness evidence is incomplete or mixed."
        confidence = "low"
    else:
        verdict = "The controlled baseline does not currently support the proposed mechanism."
        confidence = "low"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "evidence": evidence,
        "limitations": limitations,
        "next_action": (
            "Replace the synthetic pair with public equity data, use a train/test split, "
            "and rerun the same validation protocol."
        ),
    }
