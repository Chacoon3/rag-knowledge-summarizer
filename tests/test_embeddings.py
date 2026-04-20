from __future__ import annotations

import numpy as np
import pytest

from local_rag.cache import InMemoryKeyValueCache
from local_rag.embeddings import EmbeddingBackend


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(
        self,
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        self.calls.append(list(texts))
        vectors = []
        for index, text in enumerate(texts, start=1):
            vectors.append([float(len(text)), float(index)])
        return np.asarray(vectors, dtype=np.float32)


def test_embedding_cache_reuses_cached_vectors_within_and_across_calls(
    tmp_path,
) -> None:
    backend = EmbeddingBackend(
        "fake-model",
        cache_backend=InMemoryKeyValueCache(),
        cache_enabled=True,
    )
    fake_model = FakeSentenceTransformer()
    backend.get_model = lambda: fake_model

    first = backend.encode(["alpha", "beta", "alpha"])
    second = backend.encode(["beta", "alpha"])

    assert fake_model.calls == [["alpha", "beta"]]
    assert first.shape == (3, 2)
    assert second.shape == (2, 2)
    assert np.allclose(first[0], first[2])
    assert np.allclose(second[0], first[1])


def test_embedding_cache_persists_across_backend_instances(tmp_path) -> None:
    cache_backend = InMemoryKeyValueCache()

    first_backend = EmbeddingBackend(
        "fake-model",
        cache_backend=cache_backend,
        cache_enabled=True,
    )
    first_model = FakeSentenceTransformer()
    first_backend.get_model = lambda: first_model
    first_result = first_backend.encode(["gamma"])

    second_backend = EmbeddingBackend(
        "fake-model",
        cache_backend=cache_backend,
        cache_enabled=True,
    )

    def fail_if_called():
        raise AssertionError("model.encode should not be called when cache is warm")

    second_backend.get_model = fail_if_called
    second_result = second_backend.encode(["gamma"])

    assert first_model.calls == [["gamma"]]
    assert np.allclose(first_result, second_result)


def test_embedding_cache_logs_hit_and_miss_stats(tmp_path, caplog) -> None:
    backend = EmbeddingBackend(
        "fake-model",
        cache_backend=InMemoryKeyValueCache(),
        cache_enabled=True,
    )
    fake_model = FakeSentenceTransformer()
    backend.get_model = lambda: fake_model

    caplog.set_level("INFO")

    backend.encode(["alpha", "beta", "alpha"])
    backend.encode(["beta", "alpha"])

    messages = [record.getMessage() for record in caplog.records]
    assert any("hits=0 misses=2" in message for message in messages)
    assert any("hits=2 misses=0" in message for message in messages)
    assert any(
        "cumulative_hits=2 cumulative_misses=2" in message for message in messages
    )


@pytest.mark.anyio
async def test_embedding_cache_works_for_async_encoding(tmp_path) -> None:
    backend = EmbeddingBackend(
        "fake-model",
        cache_backend=InMemoryKeyValueCache(),
        cache_enabled=True,
    )
    fake_model = FakeSentenceTransformer()
    backend.get_model = lambda: fake_model

    first = await backend.encode_async(["delta", "epsilon"])
    second = await backend.encode_async(["epsilon", "delta"])

    assert fake_model.calls == [["delta", "epsilon"]]
    assert first.shape == (2, 2)
    assert second.shape == (2, 2)
    assert np.allclose(first[0], second[1])
    assert np.allclose(first[1], second[0])
