"""Answer synthesis. Uses any OpenAI-compatible chat endpoint when configured, otherwise an extractive fallback.

Environment: LLM_BASE_URL (e.g. https://api.openai.com/v1 or http://localhost:11434/v1 for Ollama),
LLM_API_KEY (optional for local servers) and LLM_MODEL.
"""

from __future__ import annotations

import os
import re

import httpx

from .index import Hit
from .text import tokenize

SYSTEM_PROMPT = (
    "Answer the question using only the numbered passages. Cite passages as [n]. "
    "If the passages do not contain the answer, say you do not know."
)


def _best_sentences(question: str, text: str, limit: int = 2) -> str:
    q_terms = set(tokenize(question))
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if len(s.split()) >= 4]
    ranked = sorted(sentences, key=lambda s: -len(q_terms.intersection(tokenize(s))))
    return " ".join(ranked[:limit]) or text[:300]


def _llm(question: str, hits: list[Hit], transport: httpx.BaseTransport | None = None) -> str | None:
    base, model = os.getenv("LLM_BASE_URL"), os.getenv("LLM_MODEL")
    if not (base and model):
        return None
    passages = "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, start=1))
    headers = {"Authorization": f"Bearer {os.getenv('LLM_API_KEY', '')}"} if os.getenv("LLM_API_KEY") else {}
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Passages:\n{passages}\n\nQuestion: {question}"},
        ],
    }
    with httpx.Client(timeout=60, transport=transport) as client:
        resp = client.post(f"{base.rstrip('/')}/chat/completions", json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def answer(question: str, hits: list[Hit], *, transport: httpx.BaseTransport | None = None) -> dict:
    citations = [{"n": i, "doc_id": h.doc_id, "heading": h.chunk.heading} for i, h in enumerate(hits, start=1)]
    if not hits:
        return {"answer": "No relevant documents found.", "mode": "none", "citations": []}
    try:
        text = _llm(question, hits, transport)
    except httpx.HTTPError:
        text = None  # a failing LLM endpoint must not take retrieval down
    if text is not None:
        return {"answer": text, "mode": "llm", "citations": citations}
    return {"answer": f"{_best_sentences(question, hits[0].chunk.text)} [1]", "mode": "extractive", "citations": citations}
