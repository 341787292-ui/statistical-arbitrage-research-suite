from __future__ import annotations

import json
from typing import Any

from quant_research_agent.agent.protocol import audit_research_run
from quant_research_agent.agent.state import ResearchAgentState
from quant_research_agent.verification.fixtures import (
    build_reference_temporal_ir,
    mutation_cases,
)
from quant_research_agent.verification.ir import StatArbExperimentIR
from quant_research_agent.verification.temporal import verify_temporal_causality


def run_temporal_mutation_benchmark() -> dict[str, Any]:
    """Compare the old artifact audit with domain-aware temporal verification."""

    reference = build_reference_temporal_ir()
    reference_report = verify_temporal_causality(reference)
    cases: list[dict[str, Any]] = []

    for mutation in mutation_cases():
        mutated = mutation.apply(reference)
        old_audit = _run_old_protocol_control(mutated)
        new_report = verify_temporal_causality(mutated)
        finding_ids = sorted({item.rule_id for item in new_report.findings})
        cases.append(
            {
                "mutation_id": mutation.mutation_id,
                "description": mutation.description,
                "expected_rule_id": mutation.expected_rule_id,
                "old_agent_detected": not old_audit["passed"],
                "verified_agent_detected": mutation.expected_rule_id in finding_ids,
                "finding_rule_ids": finding_ids,
            }
        )

    total = len(cases)
    old_detected = sum(item["old_agent_detected"] for item in cases)
    verified_detected = sum(item["verified_agent_detected"] for item in cases)
    return {
        "benchmark": "StatArb temporal mutation benchmark v0.1",
        "scope": (
            "Author-constructed temporal faults. This measures checker behavior, "
            "not real-world fault prevalence."
        ),
        "valid_reference_accepted": reference_report.passed,
        "fault_cases": cases,
        "summary": {
            "total_faults": total,
            "old_agent_faults_detected": old_detected,
            "verified_agent_faults_detected": verified_detected,
            "old_agent_fault_recall": old_detected / total if total else 0.0,
            "verified_agent_fault_recall": verified_detected / total if total else 0.0,
        },
    }


def _run_old_protocol_control(ir: StatArbExperimentIR) -> dict[str, Any]:
    """Run the existing protocol on complete artifacts without semantic IR access."""

    state = ResearchAgentState(query="benchmark", paper_path="paper.pdf")
    state.record("planning", "Freeze an experiment", "Plan frozen.")
    state.record("quant_execution", "Run the experiment", "Result attached.")
    state.record("validation", "Run an external check", "Check attached.")
    state.record("reflection", "Update the conclusion", "Decision updated.")
    state.record("reporting", "Render report", "Report complete.")
    audit = audit_research_run(
        trace=state.trace,
        hypotheses=[
            {
                "id": "H1",
                "claim": "The experiment is valid.",
                "evidence": "A tool result exists.",
                "status_after_new_experiment": "supported",
            }
        ],
        final_assessment={
            "verdict": "Accept the result.",
            "next_action": "Report it.",
            "evidence": ["The expected artifact is present."],
            "limitations": ["No domain temporal verifier is attached."],
            "parameter_search_authorized": False,
            "holdout_accessed": False,
        },
        experiment_result={
            "metrics": {"sharpe": 1.0},
            "experiment_ir": ir.to_dict(),
        },
        validation_results=[{"experiment": "artifact_presence_check"}],
        require_retrieval=False,
    )
    return audit.to_dict()


def main() -> None:
    print(json.dumps(run_temporal_mutation_benchmark(), indent=2))


if __name__ == "__main__":
    main()
