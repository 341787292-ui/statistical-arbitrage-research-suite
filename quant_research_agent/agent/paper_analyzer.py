from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from quant_research_agent.agent.prompts import PAPER_SPEC_PROMPT
from quant_research_agent.agent.spec import EvidenceItem, PaperResearchSpec
from quant_research_agent.llm import OpenAITextClient
from quant_research_agent.rag.retriever import LocalTfidfRetriever, RetrievedChunk


RESEARCH_QUERIES = [
    "research problem contribution statistical arbitrage",
    "portfolio generation residual factor model arbitrage portfolio",
    "signal extraction mean reversion CNN Transformer allocation",
    "data requirements returns stock universe training testing",
    "evaluation Sharpe drawdown transaction cost turnover backtest",
]


def analyze_paper(
    retriever: LocalTfidfRetriever,
    *,
    query: str,
    use_llm: bool = True,
) -> PaperResearchSpec:
    evidence_chunks = _collect_evidence(retriever)
    llm = OpenAITextClient()

    if use_llm and llm.available:
        context = _format_context(evidence_chunks)
        prompt = PAPER_SPEC_PROMPT.format(query=query, context=context)
        try:
            response = llm.generate(prompt)
            payload = _extract_json(response.text)
            spec = _spec_from_payload(payload)
            spec.evidence = _evidence_items(evidence_chunks)
            return spec
        except Exception:
            pass

    return _fallback_spec(evidence_chunks)


def _collect_evidence(retriever: LocalTfidfRetriever) -> list[RetrievedChunk]:
    seen: set[str] = set()
    evidence: list[RetrievedChunk] = []
    for research_query in RESEARCH_QUERIES:
        for item in retriever.search(research_query, top_k=3):
            if item.chunk.chunk_id in seen:
                continue
            seen.add(item.chunk.chunk_id)
            evidence.append(item)
    return evidence[:10]


def _format_context(items: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for item in items:
        parts.append(
            f"[{item.chunk.chunk_id} | score={item.score:.3f}]\n{item.chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _spec_from_payload(payload: dict[str, Any]) -> PaperResearchSpec:
    return PaperResearchSpec(
        title=str(payload.get("title") or "Unresolved title"),
        research_problem=str(payload.get("research_problem") or "Unresolved research problem"),
        financial_hypotheses=_string_list(payload.get("financial_hypotheses")),
        data_requirements=_string_list(payload.get("data_requirements")),
        portfolio_generation=_dict(payload.get("portfolio_generation")),
        signal_extraction=_dict(payload.get("signal_extraction")),
        trading_policy=_dict(payload.get("trading_policy")),
        evaluation_metrics=_string_list(payload.get("evaluation_metrics")),
        implementation_requirements=_string_list(payload.get("implementation_requirements")),
        unresolved_items=_string_list(payload.get("unresolved_items")),
    )


def _fallback_spec(items: list[RetrievedChunk]) -> PaperResearchSpec:
    text = "\n".join(item.chunk.text for item in items).lower()
    title = "Deep Learning Statistical Arbitrage" if "statistical arbitrage" in text else "Unresolved title"

    metrics = ["Sharpe ratio", "annual return", "maximum drawdown", "turnover"]
    if "transaction" in text or "cost" in text:
        metrics.append("transaction cost sensitivity")

    signal_method = "Mean reversion baseline"
    signal_notes = [
        "Use spread or residual z-score as the first reproducible signal.",
        "Upgrade path: replace the baseline signal with a CNN/Transformer module once the workflow is stable.",
    ]
    if "transformer" in text or "cnn" in text:
        signal_method = "Deep learning signal extraction with a baseline fallback"
        signal_notes.insert(0, "Paper evidence mentions deep learning signal extraction patterns.")

    return PaperResearchSpec(
        title=title,
        research_problem=(
            "Study a statistical arbitrage workflow that constructs arbitrage portfolios, "
            "extracts trading signals, and evaluates the resulting strategy."
        ),
        financial_hypotheses=[
            "Temporary price deviations among related assets may mean-revert.",
            "Residual-based signals can contain tradable information after common risk components are removed.",
        ],
        data_requirements=[
            "Daily adjusted close prices for a reproducible public equity universe.",
            "Return series for pair selection, spread construction, and backtesting.",
        ],
        portfolio_generation={
            "method": "Baseline pair or residual portfolio construction",
            "notes": [
                "Start with correlation and cointegration for pair selection.",
                "Estimate hedge ratio with linear regression and construct a residual spread.",
            ],
        },
        signal_extraction={"method": signal_method, "notes": signal_notes},
        trading_policy={
            "method": "Rule-based allocation baseline",
            "notes": [
                "Enter long or short spread positions when z-score breaches thresholds.",
                "Close positions when spread reverts near its rolling mean.",
            ],
        },
        evaluation_metrics=metrics,
        implementation_requirements=[
            "A deterministic baseline Quant Engine before adding deep learning modules.",
            "Structured outputs so the Agent can analyze experiment results later.",
        ],
        unresolved_items=[
            "Exact proprietary data sources are unavailable in the local MVP.",
            "Paper-specific hyperparameters should be filled after full PDF extraction.",
        ],
        evidence=_evidence_items(items),
    )


def _evidence_items(items: list[RetrievedChunk]) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            claim=_short_claim(item.chunk.text),
            source_chunk_id=item.chunk.chunk_id,
            source=item.chunk.source,
            score=round(item.score, 4),
        )
        for item in items
    ]


def _short_claim(text: str) -> str:
    clean = " ".join(text.split())
    return clean[:220] + ("..." if len(clean) > 220 else "")


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def spec_to_dict(spec: PaperResearchSpec) -> dict[str, Any]:
    return asdict(spec)
