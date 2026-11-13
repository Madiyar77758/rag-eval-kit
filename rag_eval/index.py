"""Chunk, index and search a document collection with several retrieval strategies."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .bm25 import BM25
from .chunking import STRATEGIES, Chunk
from .fusion import rrf
from .text import tokenize
from .vectors import Embedder, HashedTfidf

RETRIEVERS = ("bm25", "vectors", "hybrid")


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float
    chunk: Chunk


class Index:
    def __init__(
        self,
        docs: dict[str, str],
        *,
        chunking: str = "section",
        size: int = 300,
        overlap: int = 50,
        title: bool = True,
        embedder: Embedder | None = None,
    ):
        chunker = STRATEGIES[chunking]
        started = time.perf_counter()
        self.chunks: list[Chunk] = []
        for doc_id in sorted(docs):
            self.chunks.extend(chunker(doc_id, docs[doc_id], size=size, overlap=overlap, title=title))
        texts = [c.text for c in self.chunks]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.embedder: Embedder = embedder or HashedTfidf()
        self.embedder.fit(texts)
        self.matrix = self.embedder.encode(texts)
        self.build_seconds = time.perf_counter() - started
        self.n_docs = len(docs)

    def _bm25(self, query: str, k: int) -> list[int]:
        return [i for i, _ in self.bm25.search(tokenize(query), k)]

    def _vectors(self, query: str, k: int) -> list[int]:
        scores = self.matrix @ self.embedder.encode([query])[0]
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        return [int(i) for i in top[np.argsort(-scores[top], kind="stable")]]

    def search_chunks(self, query: str, *, retriever: str = "hybrid", k: int = 10, pool: int = 50) -> list[int]:
        if retriever == "bm25":
            return self._bm25(query, k)
        if retriever == "vectors":
            return self._vectors(query, k)
        if retriever == "hybrid":
            return rrf([self._bm25(query, pool), self._vectors(query, pool)], top=k)
        raise ValueError(f"unknown retriever {retriever!r}; choose from {RETRIEVERS}")

    def search_docs(self, query: str, *, retriever: str = "hybrid", k: int = 10, pool: int = 50) -> list[Hit]:
        """Best chunk per document, documents ordered by their best chunk's rank."""
        seen: dict[str, Hit] = {}
        ranked = self.search_chunks(query, retriever=retriever, k=pool, pool=pool)
        for rank, idx in enumerate(ranked, start=1):
            chunk = self.chunks[idx]
            if chunk.doc_id not in seen:
                seen[chunk.doc_id] = Hit(chunk.doc_id, 1.0 / rank, chunk)
            if len(seen) == k:
                break
        return list(seen.values())
