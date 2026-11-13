"""Sparse-to-dense hashed TF-IDF vectors with word and character n-grams.

This is a lexical-fuzzy retriever, not a neural embedding model: it tolerates typos, inflections and
identifiers such as ``select_related`` that a plain word index splits badly. The ``Embedder`` protocol
lets a neural model be dropped in (see ``rag_eval.embedders``) without touching the index.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from typing import Protocol

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> None: ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


def _features(text: str, ngram: tuple[int, int]) -> list[str]:
    words = tokenize(text)
    feats = list(words) + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
    lo, hi = ngram
    for w in words:
        padded = f"<{w}>"
        for n in range(lo, hi + 1):
            feats.extend(padded[i : i + n] for i in range(max(len(padded) - n + 1, 0)))
    return feats


class HashedTfidf:
    def __init__(self, dim: int = 4096, ngram: tuple[int, int] = (3, 5)):
        self.dim, self.ngram = dim, ngram
        self.idf = np.ones(dim, dtype=np.float32)

    def _slot(self, feat: str) -> tuple[int, float]:
        h = zlib.crc32(feat.encode())
        return h % self.dim, 1.0 if (h >> 31) & 1 else -1.0

    def fit(self, texts: list[str]) -> None:
        df = np.zeros(self.dim, dtype=np.float32)
        for text in texts:
            for slot in {self._slot(f)[0] for f in _features(text, self.ngram)}:
                df[slot] += 1
        n = max(len(texts), 1)
        self.idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, tf in Counter(_features(text, self.ngram)).items():
                slot, sign = self._slot(feat)
                out[row, slot] += sign * (1 + math.log(tf)) * self.idf[slot]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms
