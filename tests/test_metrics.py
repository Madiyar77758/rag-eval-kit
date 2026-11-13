import math

import pytest

from rag_eval.metrics import hit_at_k, ndcg_at_k, percentile, recall_at_k, reciprocal_rank

RANKED = ["a", "b", "c", "d"]


def test_hit_and_recall():
    assert hit_at_k(RANKED, {"c"}, 2) == 0.0 and hit_at_k(RANKED, {"c"}, 3) == 1.0
    assert recall_at_k(RANKED, {"a", "z"}, 4) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, {"b"}) == 0.5
    assert reciprocal_rank(RANKED, {"nope"}) == 0.0


def test_ndcg_perfect_and_partial():
    assert ndcg_at_k(RANKED, {"a"}, 4) == pytest.approx(1.0)
    assert ndcg_at_k(RANKED, {"b"}, 4) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(RANKED, set(), 4) == 0.0


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5 and percentile([], 50) == 0.0
