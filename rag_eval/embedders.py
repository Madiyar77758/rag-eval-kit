"""Optional neural embedder. Not used for the numbers in the README (the build environment had no model hub access)."""

from __future__ import annotations

import numpy as np


class SentenceTransformerEmbedder:
    """Adapter with the same interface as ``HashedTfidf``. Install with ``pip install .[neural]``."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # imported lazily: heavy optional dependency

        self._model = SentenceTransformer(model)

    def fit(self, texts: list[str]) -> None:  # nothing to learn
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
