from __future__ import annotations

import numpy as np
import pytest

from local_rag.models import DocumentChunk, KnowledgeBaseManifest
from local_rag.store import LocalVectorStore


def test_store_roundtrip(tmp_path) -> None:
    store = LocalVectorStore(tmp_path / "storage")
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            source_path="a.md",
            content="第一段内容",
            index=0,
            char_count=5,
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            source_path="b.md",
            content="第二段内容",
            index=0,
            char_count=5,
        ),
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    manifest = KnowledgeBaseManifest(
        source_dir="data/docs",
        embedding_model="fake-model",
        chunk_size=300,
        chunk_overlap=50,
        indexed_at="2026-04-19T00:00:00+00:00",
        document_count=2,
        chunk_count=2,
    )

    store.save(chunks, embeddings, manifest)
    loaded_manifest = store.load_manifest()
    results = store.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=1)

    assert loaded_manifest.embedding_model == "fake-model"
    assert results[0].chunk.source_path == "a.md"
    assert results[0].chunk.content == "第一段内容"


@pytest.mark.anyio
async def test_store_async_roundtrip(tmp_path) -> None:
    store = LocalVectorStore(tmp_path / "storage")
    chunks = [
        DocumentChunk(
            chunk_id="chunk-1",
            source_path="a.md",
            content="第一段内容",
            index=0,
            char_count=5,
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            source_path="b.md",
            content="第二段内容",
            index=0,
            char_count=5,
        ),
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    manifest = KnowledgeBaseManifest(
        source_dir="data/docs",
        embedding_model="fake-model",
        chunk_size=300,
        chunk_overlap=50,
        indexed_at="2026-04-19T00:00:00+00:00",
        document_count=2,
        chunk_count=2,
    )

    await store.save_async(chunks, embeddings, manifest)
    loaded_manifest = await store.load_manifest_async()
    results = await store.search_async(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=1,
    )

    assert loaded_manifest.embedding_model == "fake-model"
    assert results[0].chunk.source_path == "a.md"
