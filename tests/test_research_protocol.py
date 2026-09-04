from __future__ import annotations

import unittest

from quant_research_agent.agent.protocol import audit_research_run
from quant_research_agent.agent.state import ResearchAgentState
from quant_research_agent.methodology import FOUNDATION_BY_ID, FOUNDATIONS


def _valid_trace() -> list[dict]:
    state = ResearchAgentState(query="test", paper_path="paper.pdf")
    state.record("planning", "Define a fixed experiment", "Plan frozen.")
    state.record("quant_execution", "Run the tool", "Sharpe observed.")
    state.record("validation", "Run a robustness check", "External result observed.")
    state.record("reflection", "Update the decision", "Hypothesis rejected.")
    state.record("reporting", "Render the report", "Report complete.")
    return state.trace


def _hypotheses() -> list[dict]:
    return [
        {
            "id": "H1",
            "claim": "A measurable effect exists.",
            "evidence": "A deterministic tool result is required.",
            "status_after_new_experiment": "rejected",
        }
    ]


def _final() -> dict:
    return {
        "verdict": "The mechanism is not supported.",
        "next_action": "Stop this branch.",
        "evidence": ["The validation result failed the frozen gate."],
        "limitations": ["The sample is bounded."],
        "parameter_search_authorized": False,
        "holdout_accessed": False,
    }


class MethodologyRegistryTests(unittest.TestCase):
    def test_foundation_ids_are_unique_and_have_code_contracts(self) -> None:
        ids = [item.foundation_id for item in FOUNDATIONS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(FOUNDATION_BY_ID))
        for item in FOUNDATIONS:
            self.assertTrue(item.url.startswith("https://"))
            self.assertTrue(item.design_rule)
            self.assertTrue(item.implementation)
            self.assertTrue(item.non_claim)
            self.assertTrue(item.code_surfaces)

    def test_trace_events_record_method_provenance(self) -> None:
        trace = _valid_trace()
        for event in trace:
            self.assertTrue(event["method_ids"])
            self.assertTrue(
                all(item in FOUNDATION_BY_ID for item in event["method_ids"])
            )


class ResearchProtocolAuditTests(unittest.TestCase):
    def test_valid_external_evidence_loop_passes(self) -> None:
        audit = audit_research_run(
            trace=_valid_trace(),
            hypotheses=_hypotheses(),
            final_assessment=_final(),
            experiment_result={"metrics": {"sharpe": 0.1}},
            validation_results=[{"experiment": "fixed_check"}],
            require_retrieval=False,
        )

        self.assertTrue(audit.passed)
        self.assertFalse(audit.critical_failures)

    def test_intrinsic_reflection_without_external_feedback_fails(self) -> None:
        trace = [
            event
            for event in _valid_trace()
            if event["phase"] != "validation"
        ]
        audit = audit_research_run(
            trace=trace,
            hypotheses=_hypotheses(),
            final_assessment=_final(),
            experiment_result={"metrics": {"sharpe": 0.1}},
            validation_results=[],
            require_retrieval=False,
        )

        self.assertFalse(audit.passed)
        self.assertIn("external_feedback_before_reflection", audit.critical_failures)

    def test_unsealed_search_controls_fail(self) -> None:
        final = _final()
        final["parameter_search_authorized"] = True
        audit = audit_research_run(
            trace=_valid_trace(),
            hypotheses=_hypotheses(),
            final_assessment=final,
            experiment_result={"metrics": {"sharpe": 0.1}},
            validation_results=[{"experiment": "fixed_check"}],
            require_retrieval=False,
        )

        self.assertFalse(audit.passed)
        self.assertIn("selection_bias_controls", audit.critical_failures)

    def test_required_retrieval_without_evidence_fails(self) -> None:
        trace = _valid_trace()
        trace.insert(
            0,
            {
                "step": 0,
                "phase": "retrieval",
                "action": "Retrieve paper evidence",
                "outcome": "No evidence found.",
                "method_ids": ["rag-2020", "self-rag-2024"],
            },
        )
        audit = audit_research_run(
            trace=trace,
            hypotheses=_hypotheses(),
            final_assessment=_final(),
            experiment_result={"metrics": {"sharpe": 0.1}},
            validation_results=[{"experiment": "fixed_check"}],
            retrieved_evidence_count=0,
            require_retrieval=True,
        )

        self.assertFalse(audit.passed)
        self.assertIn("retrieval_grounding", audit.critical_failures)

    def test_retrieval_without_source_provenance_fails(self) -> None:
        trace = _valid_trace()
        trace.insert(
            0,
            {
                "step": 0,
                "phase": "retrieval",
                "action": "Retrieve paper evidence",
                "outcome": "One item found.",
                "method_ids": ["rag-2020", "self-rag-2024"],
            },
        )
        audit = audit_research_run(
            trace=trace,
            hypotheses=_hypotheses(),
            final_assessment=_final(),
            experiment_result={"metrics": {"sharpe": 0.1}},
            validation_results=[{"experiment": "fixed_check"}],
            retrieved_evidence=[
                {
                    "claim": "A claim exists.",
                    "source": "paper.pdf",
                    "source_chunk_id": "",
                }
            ],
            require_retrieval=True,
        )

        self.assertFalse(audit.passed)
        self.assertIn("retrieval_grounding", audit.critical_failures)


if __name__ == "__main__":
    unittest.main()
