from __future__ import annotations

import unittest
from pathlib import Path

from quant_research_agent.pipeline import run_paper_pipeline
from quant_research_agent.quant.tools import run_stat_arb_experiment
from quant_research_agent.quant.tools import run_cost_sensitivity_experiment
from quant_research_agent.rag.chunking import chunk_text
from quant_research_agent.rag.retriever import LocalTfidfRetriever


class PipelineSmokeTest(unittest.TestCase):
    def test_retriever_finds_stat_arb_chunks(self) -> None:
        chunks = chunk_text(
            "Statistical arbitrage uses residual spreads.\n\nCooking recipes are unrelated.",
            source="inline",
            max_chars=80,
            overlap=10,
        )
        retriever = LocalTfidfRetriever(chunks)
        result = retriever.search("residual statistical arbitrage", top_k=1)
        self.assertEqual(len(result), 1)
        self.assertIn("Statistical arbitrage", result[0].chunk.text)

    def test_pipeline_runs_without_openai_key(self) -> None:
        result = run_paper_pipeline(
            paper_path=Path("samples/stat_arb_note.txt"),
            query="Help me reproduce this statistical arbitrage paper.",
            use_llm=False,
        )
        self.assertIn("statistical arbitrage", result.spec.research_problem.lower())
        self.assertGreaterEqual(len(result.plan.steps), 5)
        self.assertIn("# Paper Research Spec", result.report_markdown)

    def test_pipeline_can_attach_quant_result(self) -> None:
        result = run_paper_pipeline(
            paper_path=Path("samples/stat_arb_note.txt"),
            query="Help me reproduce this statistical arbitrage paper.",
            use_llm=False,
            run_quant=True,
        )
        self.assertIsNotNone(result.experiment_result)
        self.assertIsNotNone(result.result_analysis)
        self.assertIn("Quant Baseline Result", result.report_markdown)
        self.assertIn("Research Interpretation", result.report_markdown)

    def test_stat_arb_tool_returns_metrics(self) -> None:
        result = run_stat_arb_experiment()
        self.assertIn("metrics", result)
        self.assertIn("sharpe", result["metrics"])
        self.assertGreater(result["diagnostics"]["trade_count"], 0)
        self.assertGreaterEqual(
            result["diagnostics"]["first_position_index"],
            result["diagnostics"]["calibration_days"],
        )

    def test_cost_sensitivity_uses_real_backtest_outputs(self) -> None:
        result = run_cost_sensitivity_experiment()
        self.assertEqual(len(result["scenarios"]), 4)
        self.assertGreater(
            result["scenarios"][0]["annual_return"],
            result["scenarios"][-1]["annual_return"],
        )

    def test_baseline_agent_completes_research_loop(self) -> None:
        result = run_paper_pipeline(
            paper_path=Path("samples/stat_arb_note.txt"),
            query="Reproduce the paper, test the baseline, and validate the result.",
            use_llm=False,
            run_agent=True,
        )
        phases = {event["phase"] for event in result.agent_trace}
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.validation_results), 2)
        self.assertIn("validation", phases)
        self.assertIn("reflection", phases)
        self.assertIn("reporting", phases)
        self.assertIn("Final Agent Assessment", result.report_markdown)
        self.assertIn("Agent Execution Trace", result.report_markdown)
        self.assertTrue(result.protocol_audit["passed"])
        self.assertGreaterEqual(len(result.technical_foundations), 5)
        foundation_ids = {
            item["foundation_id"] for item in result.technical_foundations
        }
        self.assertIn("rag-2020", foundation_ids)
        self.assertIn("Technical Method Foundations", result.report_markdown)
        self.assertIn("Research Protocol Audit", result.report_markdown)


if __name__ == "__main__":
    unittest.main()
