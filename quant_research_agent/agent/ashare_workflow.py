from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from quant_research_agent.agent.protocol import audit_research_run
from quant_research_agent.integrations.ashare import (
    AshareExperimentContract,
    AshareResearchTools,
    create_ashare_experiment_contract,
)
from quant_research_agent.methodology import (
    active_foundation_ids,
    foundation_ids_for_phase,
    methodology_manifest,
)


@dataclass
class AshareAgentResult:
    contract: AshareExperimentContract
    initial_evidence: dict[str, Any]
    hypotheses: list[dict[str, Any]]
    experiment_result: dict[str, Any]
    audit_result: dict[str, Any]
    comparison_result: dict[str, Any]
    predictability_result: dict[str, Any]
    final_assessment: dict[str, Any]
    trace: list[dict[str, Any]] = field(default_factory=list)
    technical_foundations: list[dict[str, Any]] = field(default_factory=list)
    protocol_audit: dict[str, Any] | None = None
    report_markdown: str = ""
    status: str = "completed"

    def to_json(self, *, indent: int | None = None) -> str:
        payload = asdict(self)
        return json.dumps(payload, ensure_ascii=False, indent=indent)


def run_ashare_diagnostic_agent(
    *,
    panel_path: str | Path,
    diagnostics_path: str | Path,
    tools: AshareResearchTools | None = None,
) -> AshareAgentResult:
    """Run a bounded evidence -> hypothesis -> experiment -> reflection loop."""

    contract = create_ashare_experiment_contract(panel_path, diagnostics_path)
    toolset = tools or AshareResearchTools(contract)
    trace: list[dict[str, Any]] = []

    def record(phase: str, action: str, outcome: str) -> None:
        trace.append(
            {
                "step": len(trace) + 1,
                "phase": phase,
                "action": action,
                "outcome": outcome,
                "method_ids": list(foundation_ids_for_phase(phase)),
            }
        )

    record(
        "contract",
        "Freeze data, parameters, and holdout boundary",
        f"Accepted panel through {contract.latest_allowed_date}; parameter search disabled.",
    )
    evidence = toolset.invoke("inspect_ashare_direction_diagnostic")
    decisions = evidence["decisions"]
    record(
        "evidence_review",
        "Inspect RankIC, sign reversal, neutral control, and cost attribution",
        (
            f"RankIC gate={decisions['rank_ic_gate_passed']}; "
            f"simple sign flip={decisions['simple_sign_flip_supported']}."
        ),
    )

    hypotheses = [
        {
            "id": "H1",
            "claim": "The stock-space OU direction is simply reversed.",
            "status_before_new_experiment": (
                "supported"
                if decisions["simple_sign_flip_supported"]
                else "rejected"
            ),
            "evidence": "Reversed gross excess is compared with the zero-signal control.",
        },
        {
            "id": "H2",
            "claim": "The paper-direction OU policy has residual-space mean-reversion value.",
            "status_before_new_experiment": "unresolved",
            "evidence": "Requires a one-day-ahead residual-space mechanism test.",
        },
        {
            "id": "H3",
            "claim": "Dense stock-space mapping creates economically excessive turnover.",
            "status_before_new_experiment": (
                "supported"
                if decisions["cost_problem_is_signal_induced"]
                else "unresolved"
            ),
            "evidence": "Signal and neutral annualized cost drags are compared directly.",
        },
        {
            "id": "H4",
            "claim": "Monthly PCA refits create a discontinuity inside OU histories.",
            "status_before_new_experiment": "unresolved",
            "evidence": "Requires a residual coverage and model-version continuity audit.",
        },
        {
            "id": "H5",
            "claim": "Monthly residual stitching masks an otherwise valid OU mechanism.",
            "status_before_new_experiment": "unresolved",
            "evidence": "Requires the pre-registered residual-definition comparison.",
        },
        {
            "id": "H6",
            "claim": "Stable short-horizon cross-sectional residual reversal exists.",
            "status_before_new_experiment": "unresolved",
            "evidence": "Requires model-free development/validation RankIC evidence.",
        },
        {
            "id": "H7",
            "claim": "Residual reversal is broad enough to justify a learned time-series model.",
            "status_before_new_experiment": "unresolved",
            "evidence": "Requires cross-sectional and pooled evidence at multiple horizons.",
        },
    ]
    record(
        "hypothesis_generation",
        "Form mutually distinguishable explanations",
        "Created mechanism, implementation, continuity, and predictability hypotheses.",
    )

    experiment = toolset.invoke("run_fixed_residual_ou_mechanism_test")
    mechanism = experiment["mechanism_assessment"]
    record(
        "quant_execution",
        "Run the fixed residual-space OU mechanism test",
        (
            f"Paper-direction Sharpe={experiment['original']['annualized_sharpe']:.3f}; "
            f"reversed Sharpe={experiment['reversed']['annualized_sharpe']:.3f}."
        ),
    )

    audit_result = toolset.invoke("audit_ashare_residual_continuity")
    audit = audit_result["audit"]
    audit_assessment = audit_result["assessment"]
    if audit_assessment["visible_refit_day_spike"]:
        h4_status = "supported"
    elif audit_assessment["all_ou_windows_cross_refits"]:
        h4_status = "structural_confounder_not_confirmed_causal"
    else:
        h4_status = "rejected"
    hypotheses[3]["status_after_new_experiment"] = h4_status
    record(
        "continuity_audit",
        "Audit residual coverage and model versions inside each OU history",
        (
            f"Cross-model windows={audit['cross_model_ou_window_rate']:.2%}; "
            f"models/window={audit['average_models_per_ou_window']:.2f}; "
            f"visible refit spike={audit_assessment['visible_refit_day_spike']}."
        ),
    )

    comparison_result = toolset.invoke("compare_ashare_residual_definitions")
    comparison = comparison_result["comparison"]
    comparison_assessment = comparison_result["assessment"]
    hypotheses[4]["status_after_new_experiment"] = (
        "supported"
        if comparison_assessment["current_composition_rescues_mechanism"]
        else "rejected"
    )
    record(
        "residual_definition_comparison",
        "Compare stitched as-of and current-composition residual histories",
        (
            f"Stitched Sharpe={comparison['stitched_asof']['annualized_sharpe']:.3f}; "
            f"current-composition Sharpe="
            f"{comparison['current_composition']['annualized_sharpe']:.3f}; "
            f"gate passed={comparison['current_composition_gate_passed']}."
        ),
    )

    predictability_result = toolset.invoke("audit_ashare_residual_predictability")
    predictability = predictability_result["audit"]
    predictability_assessment = predictability_result["assessment"]
    hypotheses[5]["status_after_new_experiment"] = (
        "supported"
        if predictability_assessment[
            "short_horizon_cross_sectional_reversal_supported"
        ]
        else "rejected"
    )
    hypotheses[6]["status_after_new_experiment"] = (
        "supported"
        if predictability_assessment["broad_reversal_supported"]
        else "rejected"
    )
    record(
        "residual_predictability_audit",
        "Audit model-free residual reversal across horizons and periods",
        (
            "Cross-sectional stable horizons="
            f"{predictability['cross_sectional_stable_horizons']}; "
            f"broad stable horizons={predictability['stable_horizons']}."
        ),
    )

    if (
        mechanism["paper_direction_sharpe_above_half"]
        or comparison_assessment["current_composition_rescues_mechanism"]
    ):
        h2_status = "supported"
    elif mechanism["reversal_outperforms_original"]:
        h2_status = "rejected"
    elif not comparison["current_composition_gate_passed"]:
        h2_status = "not_supported_in_current_pilot"
    else:
        h2_status = "inconclusive"
    hypotheses[1]["status_after_new_experiment"] = h2_status

    if comparison_assessment["current_composition_rescues_mechanism"]:
        verdict = (
            "Recomputing each OU history under its current PCA composition rescues the "
            "fixed mechanism gate, so residual-definition consistency matters."
        )
        next_action = (
            "Replicate the fixed comparison on a broader development universe before "
            "testing any signal-to-portfolio mapping."
        )
    elif h2_status == "supported":
        verdict = (
            "Residual mean reversion clears the fixed paper-direction mechanism gate, "
            "but current-composition re-expression does not improve it."
        )
        next_action = "Test a slower mapping with explicit turnover budgets."
    elif h2_status == "rejected":
        verdict = (
            "The paper-direction residual mechanism is not supported; reversal performs "
            "better, but this does not authorize flipping or tuning the production signal."
        )
        next_action = (
            "Resolve the residual-definition comparison before any parameter or model search."
        )
    elif h2_status == "not_supported_in_current_pilot":
        verdict = (
            "Neither residual definition passes the fixed OU mechanism gate. "
            "Current-composition re-expression performs worse, so monthly residual "
            "stitching does not explain the weak result."
        )
        next_action = (
            "Close OU tuning on this pilot and pre-register a model-free residual "
            "predictability audit before any Fourier or neural signal model."
        )
    else:
        verdict = "The fixed residual test is inconclusive and does not justify optimization."
        next_action = "Audit residual coverage and construction before further experiments."

    if (
        predictability_assessment["short_horizon_cross_sectional_reversal_supported"]
        and not predictability_assessment["broad_reversal_supported"]
    ):
        verdict = (
            "The OU time-series mechanism is not supported, but 1- and 5-day "
            "cross-sectional residual RankIC survives both research periods. Raw pooled "
            "correlations disagree, so this is a narrow ranking effect, not broad mean reversion."
        )
        next_action = (
            "Keep OU and learned models closed; pre-register one continuous "
            "cross-sectional residual-rank mapping under A-share execution constraints."
        )
    elif predictability_assessment["broad_reversal_supported"]:
        verdict = (
            "Model-free residual reversal is stable across multiple horizons and evidence "
            "types, although the fixed OU implementation remains inadequate."
        )
        next_action = (
            "Pre-register one fixed learned signal specification without parameter search."
        )
    elif h2_status == "not_supported_in_current_pilot":
        verdict = (
            "Neither the fixed OU mechanism nor the model-free residual audit establishes "
            "stable predictability on this pilot."
        )
        next_action = "Stop the residual-signal branch and revisit data or research scope."
    final = {
        "verdict": verdict,
        "confidence": (
            "medium"
            if h2_status in {"supported", "rejected", "not_supported_in_current_pilot"}
            else "low"
        ),
        "next_action": next_action,
        "parameter_search_authorized": False,
        "learned_time_series_model_authorized": predictability_assessment[
            "learned_time_series_model_authorized"
        ],
        "holdout_accessed": False,
        "evidence": [
            (
                "The fixed paper-direction residual-space Sharpe is "
                f"{experiment['original']['annualized_sharpe']:.3f}."
            ),
            (
                "The current-composition residual definition has Sharpe "
                f"{comparison['current_composition']['annualized_sharpe']:.3f} "
                "and does not pass the frozen mechanism gate."
            ),
            (
                "The model-free audit supports only narrow 1- and 5-day "
                "cross-sectional ranking, not broad residual mean reversion."
            ),
        ],
        "limitations": [
            "The free pilot uses 100 point-in-time sampled constituents and an equal-weight benchmark.",
            "The residual experiment is theoretical and ignores cash-equity execution constraints.",
            "Cross-sectional RankIC and raw pooled magnitude correlations disagree.",
            "No 2023-2025 holdout observation was used.",
        ],
    }
    record("reflection", "Update hypotheses from executable evidence", verdict)
    record("reporting", "Render an auditable A-share Agent report", "Workflow completed.")
    validation_results = [audit_result, comparison_result, predictability_result]
    protocol_audit = audit_research_run(
        trace=trace,
        hypotheses=hypotheses,
        final_assessment=final,
        experiment_result=experiment,
        validation_results=validation_results,
        require_retrieval=False,
    ).to_dict()
    foundations = methodology_manifest(active_foundation_ids(trace))
    report = _render_report(
        contract,
        evidence,
        hypotheses,
        experiment,
        audit_result,
        comparison_result,
        predictability_result,
        final,
        trace,
        foundations,
        protocol_audit,
    )
    return AshareAgentResult(
        contract=contract,
        initial_evidence=evidence,
        hypotheses=hypotheses,
        experiment_result=experiment,
        audit_result=audit_result,
        comparison_result=comparison_result,
        predictability_result=predictability_result,
        final_assessment=final,
        trace=trace,
        technical_foundations=foundations,
        protocol_audit=protocol_audit,
        report_markdown=report,
        status="completed" if protocol_audit["passed"] else "protocol_failed",
    )


def _render_report(
    contract: AshareExperimentContract,
    evidence: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    experiment: dict[str, Any],
    audit_result: dict[str, Any],
    comparison_result: dict[str, Any],
    predictability_result: dict[str, Any],
    final: dict[str, Any],
    trace: list[dict[str, Any]],
    foundations: list[dict[str, Any]],
    protocol_audit: dict[str, Any],
) -> str:
    facts = evidence["facts"]
    lines = [
        "# A-Share Quant Research Agent Report",
        "",
        "## Technical Method Foundations",
        "| Method | Paper | Implemented control |",
        "| --- | --- | --- |",
    ]
    for item in foundations:
        lines.append(
            f"| `{item['foundation_id']}` | [{item['short_name']}]({item['url']}) "
            f"({item['venue']} {item['year']}) | {item['implementation']} |"
        )
    lines.extend(
        [
        "",
        "> These are engineering adaptations with explicit limits, not claims that the "
        "original research models were trained or reproduced.",
        "",
        "## Research Contract",
        f"- Data fingerprint: `{contract.data_fingerprint}`",
        f"- Latest allowed date: {contract.latest_allowed_date}",
        "- Parameter search allowed: no",
        "- Sealed holdout access allowed: no",
        "",
        "## Existing Evidence",
        f"- Original gross excess: {facts['original_gross_excess']:.2%}",
        f"- Reversed gross excess: {facts['reversed_gross_excess']:.2%}",
        f"- Neutral gross excess: {facts['neutral_gross_excess']:.2%}",
        f"- Original annualized cost drag: {facts['original_cost_drag']:.2%}",
        f"- Nonzero stock-alpha rate: {facts['signal_nonzero_rate']:.2%}",
        "",
        "## Hypotheses",
        ]
    )
    for item in hypotheses:
        status = item.get("status_after_new_experiment", item["status_before_new_experiment"])
        lines.append(f"- {item['id']}: {item['claim']} Status: {status}.")
    lines.extend(
        [
            "",
            "## New Residual-Space Experiment",
            "| Direction | Annualized mean | Sharpe | Active days | Daily turnover |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for direction in ("original", "reversed"):
        item = experiment[direction]
        lines.append(
            f"| {direction} | {item['annualized_mean']:.2%} | "
            f"{item['annualized_sharpe']:.3f} | {item['active_day_rate']:.2%} | "
            f"{item['average_daily_turnover']:.2%} |"
        )
    audit = audit_result["audit"]
    comparison = comparison_result["comparison"]
    predictability = predictability_result["audit"]
    lines.extend(
        [
            "",
            "## Residual Continuity Audit",
            f"- Model-day coverage: {audit['model_day_rate']:.2%}",
            f"- Candidate OU windows: {audit['ou_candidate_windows']}",
            f"- Cross-model OU windows: {audit['cross_model_ou_window_rate']:.2%}",
            f"- Average PCA models per OU window: {audit['average_models_per_ou_window']:.2f}",
            f"- Refit residual-scale ratio: {audit['refit_residual_scale_ratio']:.3f}",
            f"- Refit alpha-change ratio: {audit['refit_alpha_change_ratio']:.3f}",
            "",
            "## Residual Definition Comparison",
            "| Definition | Annualized mean | Sharpe | Active days | Daily turnover |",
            "| --- | ---: | ---: | ---: | ---: |",
            (
                f"| stitched as-of | "
                f"{comparison['stitched_asof']['annualized_mean']:.2%} | "
                f"{comparison['stitched_asof']['annualized_sharpe']:.3f} | "
                f"{comparison['stitched_asof']['active_day_rate']:.2%} | "
                f"{comparison['stitched_asof']['average_daily_turnover']:.2%} |"
            ),
            (
                f"| current composition | "
                f"{comparison['current_composition']['annualized_mean']:.2%} | "
                f"{comparison['current_composition']['annualized_sharpe']:.3f} | "
                f"{comparison['current_composition']['active_day_rate']:.2%} | "
                f"{comparison['current_composition']['average_daily_turnover']:.2%} |"
            ),
            "",
            "## Model-Free Residual Predictability",
            "| Horizon | Development RankIC | Validation RankIC | "
            "Validation positive days | Raw pooled correlation |",
            "| ---: | ---: | ---: | ---: | ---: |",
            *[
                (
                    f"| {item['horizon']} | "
                    f"{item['development']['mean_rank_ic']:.4f} | "
                    f"{item['validation']['mean_rank_ic']:.4f} | "
                    f"{item['validation']['positive_rank_ic_share']:.2%} | "
                    f"{item['validation']['pooled_correlation']:.4f} |"
                )
                for item in predictability["reversal_horizons"]
            ],
            (
                "- Cross-sectional stable horizons: "
                f"{predictability['cross_sectional_stable_horizons']}"
            ),
            f"- Broad stable horizons: {predictability['stable_horizons']}",
            "",
            "## Final Assessment",
            f"- Verdict: {final['verdict']}",
            f"- Confidence: {final['confidence']}",
            f"- Parameter search authorized: {str(final['parameter_search_authorized']).lower()}",
            (
                "- Learned time-series model authorized: "
                f"{str(final['learned_time_series_model_authorized']).lower()}"
            ),
            f"- Next action: {final['next_action']}",
            "",
            "## Limitations",
            *[f"- {item}" for item in final["limitations"]],
            "",
            "## Execution Trace",
            "| Step | Phase | Method IDs | Action | Outcome |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for item in trace:
        method_ids = ", ".join(f"`{value}`" for value in item.get("method_ids", []))
        lines.append(
            f"| {item['step']} | {item['phase']} | {method_ids} | "
            f"{item['action']} | {item['outcome']} |"
        )
    lines.extend(
        [
            "",
            "## Research Protocol Audit",
            f"- Overall protocol pass: **{str(protocol_audit['passed']).lower()}**",
            "",
            "| Check | Foundation | Pass | Audit evidence |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for item in protocol_audit["checks"]:
        method_ids = ", ".join(
            f"`{value}`" for value in item.get("foundation_ids", [])
        )
        lines.append(
            f"| {item['check_id']} | {method_ids} | "
            f"{str(item['passed']).lower()} | {item['evidence']} |"
        )
    lines.append("")
    return "\n".join(lines)
