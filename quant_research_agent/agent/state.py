from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_agent.agent.spec import ExperimentPlan, PaperResearchSpec


@dataclass
class ResearchAgentState:
    """Mutable state shared by the nodes in the baseline research workflow."""

    query: str
    paper_path: str
    phase: str = "initialized"
    spec: PaperResearchSpec | None = None
    plan: ExperimentPlan | None = None
    experiment_result: dict | None = None
    result_analysis: dict | None = None
    validation_results: list[dict] = field(default_factory=list)
    final_assessment: dict | None = None
    trace: list[dict] = field(default_factory=list)

    def record(self, phase: str, action: str, outcome: str) -> None:
        self.phase = phase
        self.trace.append(
            {
                "step": len(self.trace) + 1,
                "phase": phase,
                "action": action,
                "outcome": outcome,
            }
        )
