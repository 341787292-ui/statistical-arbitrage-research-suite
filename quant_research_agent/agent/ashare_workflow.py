from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from quant_research_agent.integrations.ashare import (
    AshareExperimentContract,
    AshareResearchTools,
    create_ashare_experiment_contract,
)


@dataclass
class AshareAgentResult:
    contract: AshareExperimentContract
    initial_evidence: dict[str, Any]
    hypotheses: list[dict[str, Any]]
    experiment_result: dict[str, Any]
    audit_result: dict[str, Any]
    final_assessment: dict[str, Any]
    trace: list[dict[str, Any]] = field(default_factory=list)
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
    ]
    record(
        "hypothesis_generation",
        "Form mutually distinguishable explanations",
        "Created sign, residual-mechanism, turnover, and continuity hypotheses.",
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

    if mechanism["paper_direction_sharpe_above_half"]:
        h2_status = "supported"
    elif mechanism["reversal_outperforms_original"]:
        h2_status = "rejected"
    else:
        h2_status = "inconclusive"
    hypotheses[1]["status_after_new_experiment"] = h2_status

    if h2_status == "supported":
        verdict = (
            "Residual mean reversion clears the fixed paper-direction mechanism gate. "
            "Monthly model mixing remains a separate continuity risk."
        )
        next_action = (
            "Resolve the residual-definition comparison before testing slower "
            "signal-to-portfolio mapping."
        )
    elif h2_status == "rejected":
        verdict = (
            "The paper-direction residual mechanism is not supported; reversal performs "
            "better, but this does not authorize flipping or tuning the production signal."
        )
        next_action = (
            "Resolve the residual-definition comparison before any parameter or model search."
        )
    elif audit_assessment["all_ou_windows_cross_refits"]:
        verdict = (
            "The OU result is inconclusive and every valid 30-day history mixes "
            "multiple monthly PCA models. Refit-day jumps are not visibly larger, "
            "so model mixing is an unresolved confounder rather than a proven cause."
        )
        next_action = (
            "Pre-register one fixed comparison between stitched as-of residuals and "
            "30-day histories recomputed under the current composition matrix."
        )
    else:
        verdict = "The fixed residual test is inconclusive and does not justify optimization."
        next_action = "Audit residual coverage and construction before further experiments."
    final = {
        "verdict": verdict,
        "confidence": "medium" if h2_status in {"supported", "rejected"} else "low",
        "next_action": next_action,
        "parameter_search_authorized": False,
        "holdout_accessed": False,
        "limitations": [
            "The free pilot uses 100 point-in-time sampled constituents and an equal-weight benchmark.",
            "The residual experiment is theoretical and ignores cash-equity execution constraints.",
            "No 2023-2025 holdout observation was used.",
        ],
    }
    record("reflection", "Update hypotheses from executable evidence", verdict)
    record("reporting", "Render an auditable A-share Agent report", "Workflow completed.")
    report = _render_report(
        contract,
        evidence,
        hypotheses,
        experiment,
        audit_result,
        final,
        trace,
    )
    return AshareAgentResult(
        contract=contract,
        initial_evidence=evidence,
        hypotheses=hypotheses,
        experiment_result=experiment,
        audit_result=audit_result,
        final_assessment=final,
        trace=trace,
        report_markdown=report,
    )


def _render_report(
    contract: AshareExperimentContract,
    evidence: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    experiment: dict[str, Any],
    audit_result: dict[str, Any],
    final: dict[str, Any],
    trace: list[dict[str, Any]],
) -> str:
    facts = evidence["facts"]
    lines = [
        "# A-Share Quant Research Agent Report",
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
            "## Final Assessment",
            f"- Verdict: {final['verdict']}",
            f"- Confidence: {final['confidence']}",
            f"- Parameter search authorized: {str(final['parameter_search_authorized']).lower()}",
            f"- Next action: {final['next_action']}",
            "",
            "## Limitations",
            *[f"- {item}" for item in final["limitations"]],
            "",
            "## Execution Trace",
            "| Step | Phase | Action | Outcome |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for item in trace:
        lines.append(
            f"| {item['step']} | {item['phase']} | {item['action']} | {item['outcome']} |"
        )
    lines.append("")
    return "\n".join(lines)
