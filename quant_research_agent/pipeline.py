from __future__ import annotations

from pathlib import Path

from quant_research_agent.agent.paper_analyzer import analyze_paper
from quant_research_agent.agent.planner import create_baseline_plan
from quant_research_agent.agent.report import render_report
from quant_research_agent.agent.spec import PipelineResult
from quant_research_agent.rag.chunking import chunk_text
from quant_research_agent.rag.document_loader import load_document
from quant_research_agent.rag.retriever import LocalTfidfRetriever


def run_paper_pipeline(
    *,
    paper_path: Path,
    query: str,
    use_llm: bool = True,
    run_quant: bool = False,
    run_agent: bool = False,
) -> PipelineResult:
    if run_agent:
        from quant_research_agent.agent.workflow import run_baseline_agent

        return run_baseline_agent(
            paper_path=paper_path,
            query=query,
            use_llm=use_llm,
        )

    document = load_document(paper_path)
    chunks = chunk_text(document.text, source=document.path)
    retriever = LocalTfidfRetriever(chunks)
    spec = analyze_paper(retriever, query=query, use_llm=use_llm)
    plan = create_baseline_plan(spec)
    experiment_result = None
    result_analysis = None
    if run_quant:
        from quant_research_agent.agent.result_analyzer import analyze_experiment_result
        from quant_research_agent.quant.tools import run_stat_arb_experiment

        experiment_result = run_stat_arb_experiment()
        result_analysis = analyze_experiment_result(experiment_result)
    report = render_report(
        spec,
        plan,
        experiment_result=experiment_result,
        result_analysis=result_analysis,
    )
    return PipelineResult(
        spec=spec,
        plan=plan,
        report_markdown=report,
        experiment_result=experiment_result,
        result_analysis=result_analysis,
    )
