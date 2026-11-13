import numpy as np

from rag_eval.index import Index
from rag_eval.vectors import HashedTfidf


def test_vectors_are_unit_length_and_deterministic():
    v = HashedTfidf(dim=512)
    v.fit(["alpha beta", "beta gamma"])
    a, b = v.encode(["alpha beta"]), v.encode(["alpha beta"])
    assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1.0)


def test_char_ngrams_match_inflections_and_identifiers():
    v = HashedTfidf(dim=2048)
    corpus = ["select_related follows foreign keys", "completely unrelated pizza recipe"]
    v.fit(corpus)
    m = v.encode(corpus)
    q = v.encode(["selecting related objects"])[0]
    assert (m @ q).argmax() == 0


def test_empty_text_encodes_to_zero_vector():
    v = HashedTfidf(dim=64)
    v.fit(["x"])
    assert not v.encode([""]).any()


class FakeEmbedder:
    """Any object with fit/encode can replace the default embedder."""

    def fit(self, texts):
        self.n = len(texts)

    def encode(self, texts):
        out = np.zeros((len(texts), 3), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, 0 if "alpha" in t else 1] = 1.0
        return out


def test_index_accepts_custom_embedder(docs):
    idx = Index({"a.txt": "Alpha\n=====\n\nalpha text here", "b.txt": "Beta\n====\n\nsomething else"}, embedder=FakeEmbedder())
    assert idx.search_docs("alpha", retriever="vectors", k=1)[0].doc_id == "a.txt"
