from __future__ import annotations

from pathlib import Path

from quant_research_agent.agent.paper_analyzer import analyze_paper
from quant_research_agent.agent.planner import create_baseline_plan
from quant_research_agent.agent.report import render_report
from quant_research_agent.agent.result_analyzer import (
    analyze_experiment_result,
    reflect_on_validations,
)
from quant_research_agent.agent.spec import PipelineResult
from quant_research_agent.agent.state import ResearchAgentState
from quant_research_agent.quant.tools import QuantToolRegistry
from quant_research_agent.rag.chunking import chunk_text
from quant_research_agent.rag.document_loader import load_document
from quant_research_agent.rag.retriever import LocalTfidfRetriever


VALIDATION_TOOL_MAP = {
    "cost sensitivity test": "run_cost_sensitivity_experiment",
    "rolling-period analysis": "run_period_stability_experiment",
}


def run_baseline_agent(
    *,
    paper_path: Path,
    query: str,
    use_llm: bool = True,
    max_validation_experiments: int = 2,
) -> PipelineResult:
    """Run one auditable paper-to-experiment research loop."""
    state = ResearchAgentState(query=query, paper_path=str(paper_path.resolve()))
    tools = QuantToolRegistry()

    document = load_document(paper_path)
    chunks = chunk_text(document.text, source=document.path)
    retriever = LocalTfidfRetriever(chunks)
    state.record(
        "retrieval",
        "Load and index the research document",
        f"Indexed {len(chunks)} evidence chunks from {document.path.name}.",
    )

    state.spec = analyze_paper(retriever, query=query, use_llm=use_llm)
    state.record(
        "paper_understanding",
        "Extract a structured research specification",
        f"Identified {len(state.spec.financial_hypotheses)} financial hypotheses.",
    )

    state.plan = create_baseline_plan(state.spec)
    state.record(
        "planning",
        "Map the paper to an executable baseline plan",
        f"Created {len(state.plan.steps)} research steps and selected the baseline backtest tool.",
    )

    state.experiment_result = tools.invoke("run_stat_arb_experiment")
    state.record(
        "quant_execution",
        "Run the statistical arbitrage baseline",
        f"Observed Sharpe {state.experiment_result['metrics']['sharpe']:.2f}.",
    )

    state.result_analysis = analyze_experiment_result(state.experiment_result)
    state.record(
        "diagnosis",
        "Generate hypotheses and choose executable validation tests",
        f"Generated {len(state.result_analysis['hypotheses'])} hypotheses.",
    )

    selected_tools = _select_validation_tools(
        state.result_analysis,
        available_tools=set(tools.names),
        limit=max_validation_experiments,
    )
    for tool_name in selected_tools:
        result = tools.invoke(tool_name)
        state.validation_results.append(result)
        state.record(
            "validation",
            f"Invoke {tool_name}",
            _validation_outcome(result),
        )

    state.final_assessment = reflect_on_validations(
        state.experiment_result,
        state.validation_results,
    )
    state.record(
        "reflection",
        "Update the conclusion from validation evidence",
        state.final_assessment["verdict"],
    )

    state.record(
        "reporting",
        "Generate the auditable research report",
        "Baseline agent completed the paper-to-evidence loop.",
    )
    report = render_report(
        state.spec,
        state.plan,
        experiment_result=state.experiment_result,
        result_analysis=state.result_analysis,
        validation_results=state.validation_results,
        final_assessment=state.final_assessment,
        agent_trace=state.trace,
    )

    return PipelineResult(
        spec=state.spec,
        plan=state.plan,
        report_markdown=report,
        experiment_result=state.experiment_result,
        result_analysis=state.result_analysis,
        validation_results=state.validation_results,
        final_assessment=state.final_assessment,
        agent_trace=state.trace,
        status="completed",
    )


def _select_validation_tools(
    analysis: dict,
    *,
    available_tools: set[str],
    limit: int,
) -> list[str]:
    selected: list[str] = []
    for recommendation in analysis.get("recommended_experiments", []):
        experiment = str(recommendation.get("experiment", "")).strip().lower()
        tool_name = VALIDATION_TOOL_MAP.get(experiment)
        if tool_name and tool_name in available_tools and tool_name not in selected:
            selected.append(tool_name)
        if len(selected) >= limit:
            break
    return selected


def _validation_outcome(result: dict) -> str:
    experiment = result.get("experiment")
    summary = result.get("summary", {})
    if experiment == "cost_sensitivity":
        return (
            f"Positive Sharpe in {summary.get('positive_sharpe_scenarios', 0)}/"
            f"{summary.get('total_scenarios', 0)} cost scenarios."
        )
    if experiment == "period_stability":
        return f"Subperiod Sharpe gap was {summary.get('sharpe_gap', 'unresolved')}."
    return "Validation completed."
