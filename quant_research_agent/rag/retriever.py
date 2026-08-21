from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from quant_research_agent.rag.chunking import TextChunk


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: TextChunk
    score: float


class LocalTfidfRetriever:
    def __init__(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build retriever with no chunks.")

        self.chunks = chunks
        self.term_counts = [_token_counts(chunk.text) for chunk in chunks]
        doc_freq: Counter[str] = Counter()
        for counts in self.term_counts:
            doc_freq.update(counts.keys())
        total_docs = len(chunks)
        self.idf = {
            term: math.log((1 + total_docs) / (1 + freq)) + 1
            for term, freq in doc_freq.items()
        }
        self.doc_vectors = [self._weight(counts) for counts in self.term_counts]
        self.doc_norms = [_norm(vector) for vector in self.doc_vectors]

    def search(self, query: str, *, top_k: int = 4) -> list[RetrievedChunk]:
        query_vector = self._weight(_token_counts(query))
        query_norm = _norm(query_vector)
        if query_norm == 0:
            return []

        scored: list[RetrievedChunk] = []
        for chunk, vector, norm in zip(self.chunks, self.doc_vectors, self.doc_norms):
            if norm == 0:
                continue
            score = _dot(query_vector, vector) / (query_norm * norm)
            if score > 0:
                scored.append(RetrievedChunk(chunk=chunk, score=score))

        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def _weight(self, counts: Counter[str]) -> dict[str, float]:
        return {term: count * self.idf.get(term, 1.0) for term, count in counts.items()}


def _token_counts(text: str) -> Counter[str]:
    return Counter(token.lower() for token in TOKEN_RE.findall(text))


def _dot(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(term, 0.0) for term, weight in left.items())


def _norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(weight * weight for weight in vector.values()))
