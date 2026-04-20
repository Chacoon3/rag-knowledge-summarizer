from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.size == 0:
        return array.astype(np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


class EmbeddingBackend:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        model = self.get_model()
        embeddings = model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return normalize_vectors(np.asarray(embeddings, dtype=np.float32))

    def get_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
