"""BM25 (Okapi) over an in-memory inverted index. No dependencies."""

from __future__ import annotations

import math
from collections import Counter, defaultdict


class BM25:
    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(tokenized_docs)
        self.lengths = [len(d) for d in tokenized_docs]
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, tokens in enumerate(tokenized_docs):
            for term, tf in Counter(tokens).items():
                self.postings[term].append((doc_id, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query_tokens: list[str], k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[doc_id] / self.avg_len)
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
