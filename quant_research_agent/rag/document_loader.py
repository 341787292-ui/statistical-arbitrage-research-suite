from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    text: str


def load_document(path: Path) -> LoadedDocument:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Document not found: {resolved}")

    suffix = resolved.suffix.lower()
    if suffix in {".txt", ".md"}:
        return LoadedDocument(path=resolved, text=resolved.read_text(encoding="utf-8"))
    if suffix == ".pdf":
        return LoadedDocument(path=resolved, text=_load_pdf_text(resolved))

    raise ValueError(f"Unsupported document type: {resolved.suffix}")


def _load_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf to read PDF files: pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"\n\n[Page {index}]\n{text}")
    return "\n".join(pages)
