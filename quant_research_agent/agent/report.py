from __future__ import annotations

from dataclasses import asdict

from quant_research_agent.agent.spec import ExperimentPlan, PaperResearchSpec


def render_report(
    spec: PaperResearchSpec,
    plan: ExperimentPlan,
    experiment_result: dict | None = None,
    result_analysis: dict | None = None,
    validation_results: list[dict] | None = None,
    final_assessment: dict | None = None,
    agent_trace: list[dict] | None = None,
    technical_foundations: list[dict] | None = None,
    protocol_audit: dict | None = None,
) -> str:
    lines: list[str] = ["# Paper Research Spec", ""]
    if technical_foundations:
        lines.extend(
            [
                "## Technical Method Foundations",
                "| Method | Paper | Implemented control |",
                "| --- | --- | --- |",
            ]
        )
        for item in technical_foundations:
            lines.append(
                f"| `{item['foundation_id']}` | [{item['short_name']}]({item['url']}) "
                f"({item['venue']} {item['year']}) | {item['implementation']} |"
            )
        lines.extend(
            [
                "",
                "> These are explicit engineering adaptations. The report does not claim "
                "to reproduce or train the original research models.",
                "",
            ]
        )
    lines.extend(
        [
            "## Research Problem",
            spec.research_problem,
            "",
            "## Financial Hypotheses",
        ]
    )
    lines.extend(_bullets(spec.financial_hypotheses))
    lines.extend(
        [
            "",
            "## Data Requirements",
            *_bullets(spec.data_requirements),
            "",
            "## Method Mapping",
            f"- Portfolio generation: {_method_summary(spec.portfolio_generation)}",
            f"- Signal extraction: {_method_summary(spec.signal_extraction)}",
            f"- Trading policy: {_method_summary(spec.trading_policy)}",
            "",
            "## Evaluation Metrics",
            *_bullets(spec.evaluation_metrics),
            "",
            "## Baseline Reproduction Plan",
            f"- Objective: {plan.objective}",
            f"- Level: {plan.reproduction_level}",
        ]
    )

    for step in plan.steps:
        lines.append(f"- Step {step.step}: {step.task} (`{step.tool}`) - {step.rationale}")

    lines.extend(["", "## Assumptions", *_bullets(plan.assumptions)])
    if experiment_result is not None:
        lines.extend(["", "## Quant Baseline Result"])
        metrics = experiment_result.get("metrics", {})
        diagnostics = experiment_result.get("diagnostics", {})
        lines.extend(
            [
                f"- Pair: {experiment_result.get('pair', 'Unresolved')}",
                f"- Annual return: {_format_number(metrics.get('annual_return'))}",
                f"- Sharpe ratio: {_format_number(metrics.get('sharpe'))}",
                f"- Maximum drawdown: {_format_number(metrics.get('max_drawdown'))}",
                f"- Average turnover: {_format_number(diagnostics.get('average_turnover'))}",
                f"- Number of trades: {diagnostics.get('trade_count', 'Unresolved')}",
            ]
        )
    if result_analysis is not None:
        lines.extend(["", "## Research Interpretation"])
        lines.extend(["", "### Observations"])
        lines.extend(_bullets(result_analysis.get("observations", [])))
        lines.extend(["", "### Hypotheses"])
        for item in result_analysis.get("hypotheses", []):
            lines.append(
                "- "
                f"{item.get('hypothesis', 'Unresolved')} "
                f"(confidence: {item.get('confidence', 'unresolved')})"
            )
            evidence = item.get("supporting_evidence", [])
            if evidence:
                lines.append(f"  Evidence: {' '.join(str(part) for part in evidence)}")
            validation = item.get("validation_needed")
            if validation:
                lines.append(f"  Validation needed: {validation}")
        lines.extend(["", "### Recommended Next Experiments"])
        for item in result_analysis.get("recommended_experiments", []):
            lines.append(
                "- "
                f"{item.get('experiment', 'Unresolved')}: "
                f"{item.get('rationale', 'No rationale provided')}"
            )
    if validation_results:
        lines.extend(["", "## Agent Validation Experiments"])
        for result in validation_results:
            if result.get("experiment") == "cost_sensitivity":
                lines.extend(["", "### Transaction Cost Sensitivity"])
                lines.append("| Cost | Annual Return | Sharpe | Max Drawdown |")
                lines.append("| ---: | ---: | ---: | ---: |")
                for item in result.get("scenarios", []):
                    lines.append(
                        f"| {item['transaction_cost']:.4f} | {item['annual_return']:.4f} | "
                        f"{item['sharpe']:.4f} | {item['max_drawdown']:.4f} |"
                    )
            elif result.get("experiment") == "period_stability":
                lines.extend(["", "### Period Stability"])
                lines.append("| Period | Dates | Annual Return | Sharpe | Max Drawdown |")
                lines.append("| --- | --- | ---: | ---: | ---: |")
                for item in result.get("periods", []):
                    lines.append(
                        f"| {item['period']} | {item['start']} to {item['end']} | "
                        f"{item['annual_return']:.4f} | {item['sharpe']:.4f} | "
                        f"{item['max_drawdown']:.4f} |"
                    )
    if final_assessment:
        lines.extend(
            [
                "",
                "## Final Agent Assessment",
                f"- Verdict: {final_assessment.get('verdict', 'Unresolved')}",
                f"- Confidence: {final_assessment.get('confidence', 'unresolved')}",
                f"- Next action: {final_assessment.get('next_action', 'Unresolved')}",
                "",
                "### Validation Evidence",
                *_bullets(final_assessment.get("evidence", [])),
                "",
                "### Limitations",
                *_bullets(final_assessment.get("limitations", [])),
            ]
        )
    if agent_trace:
        lines.extend(["", "## Agent Execution Trace"])
        lines.append("| Step | Phase | Method IDs | Action | Outcome |")
        lines.append("| ---: | --- | --- | --- | --- |")
        for event in agent_trace:
            method_ids = ", ".join(f"`{item}`" for item in event.get("method_ids", []))
            lines.append(
                f"| {event.get('step')} | {event.get('phase')} | "
                f"{method_ids} | {event.get('action')} | {event.get('outcome')} |"
            )
    if protocol_audit:
        lines.extend(
            [
                "",
                "## Research Protocol Audit",
                f"- Overall protocol pass: **{str(protocol_audit.get('passed', False)).lower()}**",
                "",
                "| Check | Foundation | Pass | Audit evidence |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for item in protocol_audit.get("checks", []):
            foundations = ", ".join(
                f"`{foundation_id}`" for foundation_id in item.get("foundation_ids", [])
            )
            lines.append(
                f"| {item.get('check_id')} | {foundations} | "
                f"{str(item.get('passed', False)).lower()} | {item.get('evidence')} |"
            )
    lines.extend(["", "## Unresolved Items", *_bullets(spec.unresolved_items)])
    lines.extend(["", "## Retrieved Evidence"])

    if spec.evidence:
        for item in spec.evidence:
            score = "" if item.score is None else f", score={item.score:.4f}"
            lines.append(f"- `{item.source_chunk_id}`{score}: {item.claim}")
    else:
        lines.append("- No retrieved evidence was attached.")

    lines.extend(
        [
            "",
            "## Structured Payload",
            "```json",
            _compact_payload(spec, plan),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- Unresolved"]


def _method_summary(method: dict) -> str:
    if not method:
        return "Unresolved"
    name = method.get("method", "Unresolved method")
    notes = method.get("notes", [])
    if notes:
        return f"{name}; {' '.join(str(note) for note in notes)}"
    return str(name)


def _compact_payload(spec: PaperResearchSpec, plan: ExperimentPlan) -> str:
    import json

    return json.dumps(
        {"spec": asdict(spec), "plan": asdict(plan)},
        ensure_ascii=False,
        indent=2,
    )


def _format_number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value) if value is not None else "Unresolved"
