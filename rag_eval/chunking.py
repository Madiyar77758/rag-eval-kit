"""Chunking strategies for reStructuredText / plain-text documentation."""

from __future__ import annotations

import re
from dataclasses import dataclass

_UNDERLINE = re.compile(r"^([=\-~^\"#*+`])\1{2,}\s*$")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    heading: str
    text: str


def doc_title(text: str) -> str:
    """First reStructuredText heading of a document, or an empty string."""
    for heading, _body in split_sections(text):
        if heading:
            return heading
    return ""


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split on reStructuredText underlined headings. Returns (heading, body) pairs."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = [("", [])]
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and _UNDERLINE.match(nxt) and not _UNDERLINE.match(line) and len(nxt.strip()) >= len(line.strip()) - 1:
            sections.append((line.strip(), []))
            i += 2
            continue
        sections[-1][1].append(line)
        i += 1
    return [(h, "\n".join(body).strip()) for h, body in sections if h or "".join(body).strip()]


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    if len(words) <= size:
        return [words] if words else []
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(words[start : start + size])
        if start + size >= len(words):
            break
    return out


def fixed_chunks(doc_id: str, text: str, *, size: int = 300, overlap: int = 50, title: bool = False) -> list[Chunk]:
    """Sliding word window over the whole document."""
    prefix = doc_title(text) if title else ""
    return [
        Chunk(doc_id, i, prefix, (prefix + "\n" if prefix else "") + " ".join(w))
        for i, w in enumerate(_windows(text.split(), size, overlap))
    ]


def section_chunks(doc_id: str, text: str, *, size: int = 300, overlap: int = 50, title: bool = False) -> list[Chunk]:
    """Heading-aware chunks: never cross a section boundary, window long sections.

    With ``title=True`` every chunk is prefixed with the document title and its section heading, which
    gives short, context-free paragraphs enough vocabulary to be found.
    """
    doc_t = doc_title(text) if title else ""
    chunks: list[Chunk] = []
    for heading, body in split_sections(text):
        context = " / ".join(p for p in (doc_t, heading) if p) if title else ""
        for piece in _windows(body.split(), size, overlap):
            body_text = " ".join(piece)
            chunks.append(Chunk(doc_id, len(chunks), heading, (context + "\n" if context else "") + body_text))
    return chunks


STRATEGIES = {"fixed": fixed_chunks, "section": section_chunks}
