from __future__ import annotations

from pathlib import Path

from quant_research_agent.agent.paper_analyzer import analyze_paper
from quant_research_agent.agent.planner import create_baseline_plan
from quant_research_agent.agent.protocol import audit_research_run
from quant_research_agent.agent.report import render_report
from quant_research_agent.agent.result_analyzer import (
    analyze_experiment_result,
    reflect_on_validations,
)
from quant_research_agent.agent.spec import PipelineResult
from quant_research_agent.agent.state import ResearchAgentState
from quant_research_agent.quant.tools import QuantToolRegistry
from quant_research_agent.methodology import active_foundation_ids, methodology_manifest
from quant_research_agent.rag.chunking import chunk_text
from quant_research_agent.rag.document_loader import load_document
from quant_research_agent.rag.retriever import LocalTfidfRetriever


VALIDATION_TOOL_MAP = {
    "cost sensitivity test": "run_cost_sensitivity_experiment",
    "rolling-period analysis": "run_period_stability_experiment",
}

VERIFIED_VALIDATION_TOOL_MAP = {
    "cost sensitivity test": "run_temporally_aligned_cost_sensitivity_experiment",
    "rolling-period analysis": "run_temporally_aligned_period_stability_experiment",
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
    state.protocol_audit = audit_research_run(
        trace=state.trace,
        hypotheses=state.result_analysis.get("hypotheses", []),
        final_assessment=state.final_assessment,
        experiment_result=state.experiment_result,
        validation_results=state.validation_results,
        retrieved_evidence=state.spec.evidence,
        retrieved_evidence_count=len(state.spec.evidence),
        require_retrieval=True,
    ).to_dict()
    foundations = methodology_manifest(active_foundation_ids(state.trace))
    report = render_report(
        state.spec,
        state.plan,
        experiment_result=state.experiment_result,
        result_analysis=state.result_analysis,
        validation_results=state.validation_results,
        final_assessment=state.final_assessment,
        agent_trace=state.trace,
        technical_foundations=foundations,
        protocol_audit=state.protocol_audit,
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
        technical_foundations=foundations,
        protocol_audit=state.protocol_audit,
        status="completed" if state.protocol_audit["passed"] else "protocol_failed",
    )


def run_verified_agent(
    *,
    paper_path: Path,
    query: str,
    use_llm: bool = True,
    max_validation_experiments: int = 2,
    verification_mutation: str | None = None,
) -> PipelineResult:
    """Run the research loop with domain temporal verification attached."""

    from quant_research_agent.verification.adapters import temporal_ir_from_backtest
    from quant_research_agent.verification.benchmark import (
        run_temporal_mutation_benchmark,
    )
    from quant_research_agent.verification.fixtures import apply_named_mutation
    from quant_research_agent.verification.temporal import verify_temporal_causality

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
        f"Created {len(state.plan.steps)} research steps before domain verification.",
    )

    state.experiment_result = tools.invoke(
        "run_temporally_aligned_stat_arb_experiment"
    )
    state.record(
        "quant_execution",
        "Run the instrumented next-open statistical arbitrage baseline",
        f"Observed Sharpe {state.experiment_result['metrics']['sharpe']:.2f}.",
    )

    experiment_ir = temporal_ir_from_backtest(state.experiment_result)
    injected_mutation = None
    if verification_mutation:
        experiment_ir, injected_mutation = apply_named_mutation(
            experiment_ir,
            verification_mutation,
        )
    verification = verify_temporal_causality(experiment_ir)
    state.experiment_ir = experiment_ir.to_dict()
    state.experiment_verification = verification.to_dict()
    state.verification_benchmark = run_temporal_mutation_benchmark()
    state.record(
        "experiment_verification",
        "Verify data, model, signal, execution, return, and holdout ordering",
        (
            f"Checked {len(verification.checked_rules)} rules; "
            f"found {len(verification.findings)} violations."
            + (
                f" Injected benchmark mutation: {injected_mutation.mutation_id}."
                if injected_mutation
                else ""
            )
        ),
    )

    if not verification.passed:
        return _finish_blocked_verified_run(state)

    state.result_analysis = analyze_experiment_result(state.experiment_result)
    state.record(
        "diagnosis",
        "Generate hypotheses after experiment verification",
        f"Generated {len(state.result_analysis['hypotheses'])} hypotheses.",
    )

    selected_tools = _select_validation_tools(
        state.result_analysis,
        available_tools=set(tools.names),
        limit=max_validation_experiments,
        tool_map=VERIFIED_VALIDATION_TOOL_MAP,
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
    state.final_assessment["evidence"].append(
        "The attached StatArb-IR execution trace passed all temporal checks."
    )
    state.final_assessment["limitations"].append(
        "StatArb-IR v0.1 checks temporal provenance only; residual semantics and market rules are not yet verified."
    )
    state.record(
        "reflection",
        "Update the conclusion from verified experiment evidence",
        state.final_assessment["verdict"],
    )
    state.record(
        "reporting",
        "Generate the verified research report",
        "Verified Agent completed the paper-to-evidence loop.",
    )

    state.protocol_audit = audit_research_run(
        trace=state.trace,
        hypotheses=state.result_analysis.get("hypotheses", []),
        final_assessment=state.final_assessment,
        experiment_result=state.experiment_result,
        validation_results=state.validation_results,
        retrieved_evidence=state.spec.evidence,
        retrieved_evidence_count=len(state.spec.evidence),
        require_retrieval=True,
    ).to_dict()
    foundations = methodology_manifest(active_foundation_ids(state.trace))
    report = render_report(
        state.spec,
        state.plan,
        experiment_result=state.experiment_result,
        result_analysis=state.result_analysis,
        validation_results=state.validation_results,
        final_assessment=state.final_assessment,
        agent_trace=state.trace,
        technical_foundations=foundations,
        protocol_audit=state.protocol_audit,
        experiment_ir=state.experiment_ir,
        experiment_verification=state.experiment_verification,
        verification_benchmark=state.verification_benchmark,
    )
    status = (
        "completed"
        if state.protocol_audit["passed"] and state.experiment_verification["passed"]
        else "protocol_failed"
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
        technical_foundations=foundations,
        protocol_audit=state.protocol_audit,
        experiment_ir=state.experiment_ir,
        experiment_verification=state.experiment_verification,
        verification_benchmark=state.verification_benchmark,
        status=status,
    )


def _finish_blocked_verified_run(state: ResearchAgentState) -> PipelineResult:
    findings = (state.experiment_verification or {}).get("findings", [])
    hypotheses = [
        {
            "id": "INVALID-TIMING",
            "claim": "The experiment contains a temporal-causality violation.",
            "evidence": "; ".join(item.get("counterexample", "") for item in findings),
            "status_after_new_experiment": "supported",
        }
    ]
    state.result_analysis = {
        "observations": [item.get("message", "") for item in findings],
        "hypotheses": hypotheses,
        "recommended_experiments": [],
    }
    state.validation_results = [
        {
            "experiment": "temporal_verification",
            "findings": findings,
        }
    ]
    state.final_assessment = {
        "verdict": "Block the research conclusion because temporal causality failed.",
        "confidence": "high",
        "evidence": [item.get("counterexample", "") for item in findings],
        "limitations": [
            "The numerical output may exist, but it is not admissible research evidence."
        ],
        "parameter_search_authorized": False,
        "holdout_accessed": False,
        "next_action": "; ".join(item.get("repair", "") for item in findings),
    }
    state.record(
        "reflection",
        "Reject an experiment with invalid temporal provenance",
        state.final_assessment["verdict"],
    )
    state.record(
        "reporting",
        "Generate a blocked-run report",
        "No strategy conclusion is authorized.",
    )
    state.protocol_audit = audit_research_run(
        trace=state.trace,
        hypotheses=hypotheses,
        final_assessment=state.final_assessment,
        experiment_result=state.experiment_result,
        validation_results=state.validation_results,
        retrieved_evidence=state.spec.evidence,
        retrieved_evidence_count=len(state.spec.evidence),
        require_retrieval=True,
    ).to_dict()
    foundations = methodology_manifest(active_foundation_ids(state.trace))
    report = render_report(
        state.spec,
        state.plan,
        experiment_result=state.experiment_result,
        result_analysis=state.result_analysis,
        validation_results=state.validation_results,
        final_assessment=state.final_assessment,
        agent_trace=state.trace,
        technical_foundations=foundations,
        protocol_audit=state.protocol_audit,
        experiment_ir=state.experiment_ir,
        experiment_verification=state.experiment_verification,
        verification_benchmark=state.verification_benchmark,
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
        technical_foundations=foundations,
        protocol_audit=state.protocol_audit,
        experiment_ir=state.experiment_ir,
        experiment_verification=state.experiment_verification,
        verification_benchmark=state.verification_benchmark,
        status="verification_blocked",
    )


def _select_validation_tools(
    analysis: dict,
    *,
    available_tools: set[str],
    limit: int,
    tool_map: dict[str, str] = VALIDATION_TOOL_MAP,
) -> list[str]:
    selected: list[str] = []
    for recommendation in analysis.get("recommended_experiments", []):
        experiment = str(recommendation.get("experiment", "")).strip().lower()
        tool_name = tool_map.get(experiment)
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
