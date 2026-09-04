from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidenceItem:
    claim: str
    source_chunk_id: str
    source: str
    score: float | None = None


@dataclass
class PaperResearchSpec:
    title: str
    research_problem: str
    financial_hypotheses: list[str] = field(default_factory=list)
    data_requirements: list[str] = field(default_factory=list)
    portfolio_generation: dict[str, Any] = field(default_factory=dict)
    signal_extraction: dict[str, Any] = field(default_factory=dict)
    trading_policy: dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: list[str] = field(default_factory=list)
    implementation_requirements: list[str] = field(default_factory=list)
    unresolved_items: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)


@dataclass
class ExperimentStep:
    step: int
    task: str
    tool: str
    rationale: str


@dataclass
class ExperimentPlan:
    objective: str
    reproduction_level: str
    steps: list[ExperimentStep]
    metrics: list[str]
    assumptions: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    spec: PaperResearchSpec
    plan: ExperimentPlan
    report_markdown: str
    experiment_result: dict | None = None
    result_analysis: dict | None = None
    validation_results: list[dict] = field(default_factory=list)
    final_assessment: dict | None = None
    agent_trace: list[dict] = field(default_factory=list)
    technical_foundations: list[dict] = field(default_factory=list)
    protocol_audit: dict | None = None
    status: str = "completed"

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)
