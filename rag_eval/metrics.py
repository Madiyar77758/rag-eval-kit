"""Ranking metrics with binary relevance. ``ranked`` is a list of document ids, best first."""

from __future__ import annotations

import math


def hit_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return 1.0 if any(d in gold for d in ranked[:k]) else 0.0


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(gold.intersection(ranked[:k])) / len(gold) if gold else 0.0


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    for rank, doc in enumerate(ranked, start=1):
        if doc in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, d in enumerate(ranked[:k], start=1) if d in gold)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
