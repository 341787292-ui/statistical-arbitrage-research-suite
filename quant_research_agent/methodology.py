from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class TechnicalFoundation:
    """A paper-to-code contract, not a decorative bibliography entry."""

    foundation_id: str
    short_name: str
    title: str
    authors: str
    venue: str
    year: int
    url: str
    design_rule: str
    implementation: str
    non_claim: str
    code_surfaces: tuple[str, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["code_surfaces"] = list(self.code_surfaces)
        return payload


FOUNDATIONS: tuple[TechnicalFoundation, ...] = (
    TechnicalFoundation(
        foundation_id="rag-2020",
        short_name="RAG",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors="Lewis et al.",
        venue="NeurIPS",
        year=2020,
        url=(
            "https://proceedings.neurips.cc/paper_files/paper/2020/hash/"
            "6b493230205f780e1bc26945df7481e5-Abstract.html"
        ),
        design_rule=(
            "Research claims that depend on a paper must use inspectable external "
            "evidence instead of relying only on model memory."
        ),
        implementation=(
            "The paper pipeline retrieves identified source chunks and carries their "
            "provenance into the research specification and report."
        ),
        non_claim=(
            "This project adapts the retrieval-grounding principle; it does not train "
            "the original end-to-end RAG model."
        ),
        code_surfaces=(
            "quant_research_agent/rag/retriever.py",
            "quant_research_agent/agent/paper_analyzer.py",
        ),
    ),
    TechnicalFoundation(
        foundation_id="react-2023",
        short_name="ReAct",
        title="ReAct: Synergizing Reasoning and Acting in Language Models",
        authors="Yao et al.",
        venue="ICLR",
        year=2023,
        url="https://openreview.net/forum?id=WE_vluYUL-X",
        design_rule=(
            "A research judgment must be separated into a plan, an external action, "
            "an observation, and an updated decision."
        ),
        implementation=(
            "Agent traces record explicit phases, tool actions, and observations. The "
            "protocol audit verifies that quantitative execution precedes reflection."
        ),
        non_claim=(
            "The trace is a structured audit log, not a claim that hidden chain-of-thought "
            "is stored or exposed."
        ),
        code_surfaces=(
            "quant_research_agent/agent/state.py",
            "quant_research_agent/agent/workflow.py",
            "quant_research_agent/agent/ashare_workflow.py",
        ),
    ),
    TechnicalFoundation(
        foundation_id="self-rag-2024",
        short_name="Self-RAG",
        title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        authors="Asai et al.",
        venue="ICLR",
        year=2024,
        url="https://openreview.net/forum?id=hSyW5go0v8",
        design_rule=(
            "Retrieved text is not automatically trustworthy; the system must check "
            "whether evidence is present, relevant, and sufficient for the claim."
        ),
        implementation=(
            "The research protocol audits source grounding and blocks a clean protocol "
            "pass when required retrieved evidence is absent."
        ),
        non_claim=(
            "The project implements an evidence-critique gate inspired by Self-RAG; it "
            "does not train reflection tokens or reproduce the Self-RAG model."
        ),
        code_surfaces=(
            "quant_research_agent/agent/protocol.py",
            "quant_research_agent/agent/report.py",
        ),
    ),
    TechnicalFoundation(
        foundation_id="finqa-2021",
        short_name="FinQA",
        title="FinQA: A Dataset of Numerical Reasoning over Financial Data",
        authors="Chen et al.",
        venue="EMNLP",
        year=2021,
        url="https://aclanthology.org/2021.emnlp-main.300/",
        design_rule=(
            "Financial numerical conclusions should be produced by explicit programs or "
            "tools, with the calculation path available for inspection."
        ),
        implementation=(
            "Returns, Sharpe, drawdown, turnover, RankIC, and robustness checks come from "
            "deterministic Python tools rather than free-form LLM arithmetic."
        ),
        non_claim=(
            "FinQA motivates executable numerical reasoning here; this project is not a "
            "FinQA benchmark submission."
        ),
        code_surfaces=(
            "quant_research_agent/quant/tools.py",
            "quant_research_agent/integrations/ashare.py",
        ),
    ),
    TechnicalFoundation(
        foundation_id="reflexion-2023",
        short_name="Reflexion",
        title="Reflexion: Language Agents with Verbal Reinforcement Learning",
        authors="Shinn et al.",
        venue="NeurIPS",
        year=2023,
        url=(
            "https://proceedings.neurips.cc/paper_files/paper/2023/hash/"
            "1b44b878bb782e6954cd888628510e90-Abstract-Conference.html"
        ),
        design_rule=(
            "Reflection should update the next decision from observed feedback, rather "
            "than merely restating the first answer."
        ),
        implementation=(
            "After quantitative validation, the Agent updates hypothesis states, records "
            "limitations, and issues a stop/go decision."
        ),
        non_claim=(
            "The project adapts feedback-based reflection; it does not train an agent with "
            "verbal reinforcement learning."
        ),
        code_surfaces=(
            "quant_research_agent/agent/result_analyzer.py",
            "quant_research_agent/agent/ashare_workflow.py",
        ),
    ),
    TechnicalFoundation(
        foundation_id="external-feedback-2024",
        short_name="External-feedback correction",
        title="Large Language Models Cannot Self-Correct Reasoning Yet",
        authors="Huang et al.",
        venue="ICLR",
        year=2024,
        url="https://openreview.net/forum?id=IkmD3fKBPQ",
        design_rule=(
            "Do not treat an LLM's unsupported self-critique as validation; require new "
            "external evidence from data, tools, or a human reviewer."
        ),
        implementation=(
            "Protocol reflection passes only after at least one external quantitative "
            "observation is recorded."
        ),
        non_claim=(
            "This is a safety constraint derived from the paper's negative evidence, not "
            "an implementation of a new self-correction model."
        ),
        code_surfaces=("quant_research_agent/agent/protocol.py",),
    ),
    TechnicalFoundation(
        foundation_id="dsr-2014",
        short_name="Deflated Sharpe Ratio",
        title=(
            "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest "
            "Overfitting and Non-Normality"
        ),
        authors="Bailey and Lopez de Prado",
        venue="Journal of Portfolio Management",
        year=2014,
        url="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551",
        design_rule=(
            "A backtest result must disclose search and selection risk. Failed gates must "
            "not be rescued by post-hoc parameter search or premature holdout access."
        ),
        implementation=(
            "The A-share workflow freezes the data fingerprint and parameters, denies "
            "parameter search, and keeps the holdout sealed after failed gates."
        ),
        non_claim=(
            "The current control is an anti-selection protocol. A numerical DSR is not "
            "reported until a valid trial ledger and return series are available."
        ),
        code_surfaces=(
            "quant_research_agent/integrations/ashare.py",
            "quant_research_agent/agent/protocol.py",
        ),
    ),
    TechnicalFoundation(
        foundation_id="agentproof-2026",
        short_name="Agentproof",
        title="Agentproof: Static Verification of Agent Workflow Graphs",
        authors="Xavier et al.",
        venue="arXiv",
        year=2026,
        url="https://arxiv.org/abs/2603.20356",
        design_rule=(
            "Safety properties should be checked mechanically and violations should "
            "include a witness trace, rather than relying on an Agent to critique itself."
        ),
        implementation=(
            "The temporal verifier applies deterministic rules to StatArb-IR and emits "
            "a concrete counterexample plus a repair for every violation."
        ),
        non_claim=(
            "This project adapts static-verification and witness-reporting principles. "
            "It does not reproduce Agentproof's graph extraction or automata algorithms."
        ),
        code_surfaces=(
            "quant_research_agent/verification/temporal.py",
            "quant_research_agent/verification/benchmark.py",
        ),
    ),
    TechnicalFoundation(
        foundation_id="sigil-2026",
        short_name="SIGIL",
        title="SIGIL: Compiling Agent Skills into Typed Harnesses",
        authors="Dantanarayana et al.",
        venue="arXiv",
        year=2026,
        url="https://arxiv.org/abs/2607.27309",
        design_rule=(
            "Procedural requirements that must not be skipped should be represented in "
            "a closed typed form before execution."
        ),
        implementation=(
            "StatArb-IR v0.1 represents data availability, model fitting, signal creation, "
            "execution, return attribution, and holdout selection as typed objects."
        ),
        non_claim=(
            "The project does not implement SIGIL's skill compiler or AG-IR. It applies "
            "the typed-intermediate-representation principle to a narrower finance domain."
        ),
        code_surfaces=("quant_research_agent/verification/ir.py",),
    ),
    TechnicalFoundation(
        foundation_id="prov-o-2013",
        short_name="PROV-O",
        title="PROV-O: The PROV Ontology",
        authors="Lebo, Sahoo, and McGuinness",
        venue="W3C Recommendation",
        year=2013,
        url="https://www.w3.org/TR/prov-o/",
        design_rule=(
            "Research artifacts should retain explicit derivation and activity links so "
            "a decision can be traced back to the data and process that produced it."
        ),
        implementation=(
            "StatArb-IR links derived data, model fits, signals, executions, and selection "
            "boundaries by stable identifiers."
        ),
        non_claim=(
            "StatArb-IR is not a complete PROV-O ontology or RDF serialization; it uses "
            "the standard as a provenance design reference."
        ),
        code_surfaces=(
            "quant_research_agent/verification/ir.py",
            "quant_research_agent/verification/adapters.py",
        ),
    ),
)


FOUNDATION_BY_ID = {item.foundation_id: item for item in FOUNDATIONS}


PHASE_FOUNDATIONS: dict[str, tuple[str, ...]] = {
    "contract": ("dsr-2014",),
    "retrieval": ("rag-2020", "self-rag-2024"),
    "paper_understanding": ("rag-2020", "self-rag-2024"),
    "planning": ("react-2023",),
    "evidence_review": ("self-rag-2024", "finqa-2021"),
    "hypothesis_generation": ("react-2023",),
    "quant_execution": ("react-2023", "finqa-2021"),
    "experiment_verification": (
        "agentproof-2026",
        "sigil-2026",
        "prov-o-2013",
        "external-feedback-2024",
    ),
    "diagnosis": ("react-2023", "self-rag-2024"),
    "validation": (
        "react-2023",
        "finqa-2021",
        "external-feedback-2024",
        "dsr-2014",
    ),
    "continuity_audit": ("finqa-2021", "external-feedback-2024", "dsr-2014"),
    "residual_definition_comparison": (
        "finqa-2021",
        "external-feedback-2024",
        "dsr-2014",
    ),
    "residual_predictability_audit": (
        "finqa-2021",
        "external-feedback-2024",
        "dsr-2014",
    ),
    "reflection": ("reflexion-2023", "external-feedback-2024"),
    "reporting": ("finqa-2021",),
}


def foundation_ids_for_phase(phase: str) -> tuple[str, ...]:
    return PHASE_FOUNDATIONS.get(phase, ())


def methodology_manifest(
    foundation_ids: Iterable[str] | None = None,
) -> list[dict]:
    selected = set(foundation_ids) if foundation_ids is not None else None
    return [
        item.to_dict()
        for item in FOUNDATIONS
        if selected is None or item.foundation_id in selected
    ]


def active_foundation_ids(trace: list[dict]) -> tuple[str, ...]:
    active: list[str] = []
    for event in trace:
        for foundation_id in event.get("method_ids", []):
            if foundation_id not in active:
                active.append(foundation_id)
    return tuple(active)
