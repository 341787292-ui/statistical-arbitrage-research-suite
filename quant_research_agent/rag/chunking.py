from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    text: str
    source: str
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    source: str | Path,
    *,
    max_chars: int = 1400,
    overlap: int = 200,
) -> list[TextChunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars.")

    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    chunks: list[TextChunk] = []
    start = 0
    source_name = str(source)

    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            paragraph_break = normalized.rfind("\n\n", start, end)
            sentence_break = normalized.rfind(". ", start, end)
            split_at = max(paragraph_break, sentence_break)
            if split_at > start + max_chars // 2:
                end = split_at + 1

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(
                TextChunk(
                    chunk_id=f"chunk-{len(chunks) + 1:04d}",
                    text=chunk,
                    source=source_name,
                    start_char=start,
                    end_char=end,
                )
            )

        if end >= len(normalized):
            break
        start = max(0, end - overlap)
        if start > 0:
            start = _advance_to_boundary(normalized, start)

    return chunks


def _advance_to_boundary(text: str, index: int) -> int:
    while index < len(text) and not text[index].isspace():
        index += 1
    while index < len(text) and text[index].isspace():
        index += 1
    return index
