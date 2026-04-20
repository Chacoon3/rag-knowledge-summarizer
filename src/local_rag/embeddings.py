from __future__ import annotations

import asyncio
import hashlib
import logging
from threading import Lock
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from local_rag.cache import RedisKeyValueCache


logger = logging.getLogger(__name__)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.size == 0:
        return array.astype(np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


class EmbeddingCache:
    def __init__(
        self,
        cache_backend,
        model_name: str,
        ttl_seconds: int | None = None,
    ) -> None:
        self.cache_backend = cache_backend
        self.model_name = model_name
        self.ttl_seconds = ttl_seconds

    def get_many(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        unique_texts = list(dict.fromkeys(texts))
        if not unique_texts:
            return {}

        key_to_text = {self._build_key(text): text for text in unique_texts}
        payloads = self.cache_backend.get_json_many(list(key_to_text))

        return {
            key_to_text[key]: np.asarray(value["vector"], dtype=np.float32)
            for key, value in payloads.items()
        }

    def set_many(self, embeddings_by_text: dict[str, np.ndarray]) -> None:
        if not embeddings_by_text:
            return

        self.cache_backend.set_json_many(
            {
                self._build_key(text): {
                    "text": text,
                    "vector": np.asarray(vector, dtype=np.float32).tolist(),
                }
                for text, vector in embeddings_by_text.items()
            },
            ttl_seconds=self.ttl_seconds,
        )

    def _build_key(self, text: str) -> str:
        return self.cache_backend.build_key(
            "embedding",
            self.model_name,
            self._hash_text(text),
        )

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingBackend:
    def __init__(
        self,
        model_name: str,
        cache_backend: RedisKeyValueCache | None = None,
        cache_enabled: bool = True,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = None
        self._stats_lock = Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_requests = 0
        self._cache = (
            EmbeddingCache(
                cache_backend=cache_backend,
                model_name=model_name,
                ttl_seconds=cache_ttl_seconds,
            )
            if cache_enabled and cache_backend is not None
            else None
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        normalized_texts = [str(text) for text in texts]
        if self._cache is None:
            return self._encode_batch(normalized_texts)

        unique_texts = list(dict.fromkeys(normalized_texts))
        cached_embeddings = self._cache.get_many(unique_texts)
        missing_texts = [text for text in unique_texts if text not in cached_embeddings]
        if missing_texts:
            missing_vectors = self._encode_batch(missing_texts)
            new_embeddings = {
                text: vector for text, vector in zip(missing_texts, missing_vectors)
            }
            self._cache.set_many(new_embeddings)
            cached_embeddings.update(new_embeddings)

        self._log_cache_stats(
            request_count=len(normalized_texts),
            unique_count=len(unique_texts),
            hit_count=len(unique_texts) - len(missing_texts),
            miss_count=len(missing_texts),
        )

        ordered_embeddings = [cached_embeddings[text] for text in normalized_texts]
        return normalize_vectors(np.asarray(ordered_embeddings, dtype=np.float32))

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        model = self.get_model()
        embeddings = model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return normalize_vectors(np.asarray(embeddings, dtype=np.float32))

    async def encode_async(self, texts: Sequence[str]) -> np.ndarray:
        return await asyncio.to_thread(self.encode, texts)

    def _log_cache_stats(
        self,
        request_count: int,
        unique_count: int,
        hit_count: int,
        miss_count: int,
    ) -> None:
        with self._stats_lock:
            self._cache_requests += 1
            self._cache_hits += hit_count
            self._cache_misses += miss_count
            total_hits = self._cache_hits
            total_misses = self._cache_misses
            total_requests = self._cache_requests

        current_hit_rate = hit_count / unique_count if unique_count else 0.0
        total_lookups = total_hits + total_misses
        cumulative_hit_rate = total_hits / total_lookups if total_lookups else 0.0
        logger.info(
            "Embedding cache model=%s request_texts=%d unique_texts=%d hits=%d misses=%d hit_rate=%.2f cumulative_hits=%d cumulative_misses=%d cumulative_hit_rate=%.2f requests=%d",
            self.model_name,
            request_count,
            unique_count,
            hit_count,
            miss_count,
            current_hit_rate,
            total_hits,
            total_misses,
            cumulative_hit_rate,
            total_requests,
        )

    def get_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
