"""Reciprocal rank fusion."""

from __future__ import annotations

from collections import defaultdict


def rrf(rankings: list[list[int]], k: int = 60, top: int | None = None) -> list[int]:
    """Fuse ranked lists of ids. Score of an id is the sum of 1 / (k + rank) over lists (rank starts at 1)."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    fused = sorted(scores, key=lambda i: (-scores[i], i))
    return fused[:top] if top else fused
