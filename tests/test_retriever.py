from __future__ import annotations

import numpy as np
import pytest

from local_rag.models import DocumentChunk, KnowledgeBaseManifest
from local_rag.retriever import HybridSearchConfig, RerankConfig, Retriever
from local_rag.store import LocalVectorStore


class FakeEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            if "审批" in text or "财务" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "部署" in text or "发布" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def test_retriever_returns_most_relevant_chunk(tmp_path) -> None:
    store = LocalVectorStore(tmp_path / "storage")
    chunks = [
        DocumentChunk(
            chunk_id="finance-1",
            source_path="finance.md",
            content="财务审批需要部门负责人确认。",
            index=0,
            char_count=14,
        ),
        DocumentChunk(
            chunk_id="deploy-1",
            source_path="deploy.md",
            content="部署发布前需要执行回归测试。",
            index=0,
            char_count=14,
        ),
    ]
    embedder = FakeEmbedder()
    embeddings = embedder.encode([chunk.content for chunk in chunks])
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

    retriever = Retriever(store=store, embedder=embedder, threshold=0.1)
    results = retriever.search("财务审批怎么走？", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.source_path == "finance.md"


class FlatEmbedder:
    def encode(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_hybrid_search_uses_keyword_signal_to_rerank(tmp_path) -> None:
    store = LocalVectorStore(tmp_path / "storage")
    chunks = [
        DocumentChunk(
            chunk_id="ops-1",
            source_path="ops.md",
            content="部署发布前需要执行回归测试。",
            index=0,
            char_count=14,
        ),
        DocumentChunk(
            chunk_id="finance-1",
            source_path="finance.md",
            content="财务审批流程需要部门负责人确认并归档。",
            index=0,
            char_count=19,
        ),
    ]
    embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    manifest = KnowledgeBaseManifest(
        source_dir="data/docs",
        embedding_model="flat-model",
        chunk_size=300,
        chunk_overlap=50,
        indexed_at="2026-04-19T00:00:00+00:00",
        document_count=2,
        chunk_count=2,
    )
    store.save(chunks, embeddings, manifest)

    retriever = Retriever(
        store=store,
        embedder=FlatEmbedder(),
        threshold=0.0,
        hybrid_config=HybridSearchConfig(
            mode="hybrid", vector_weight=0.2, keyword_weight=0.8
        ),
    )

    results = retriever.search("财务审批流程", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.source_path == "finance.md"


@pytest.mark.anyio
async def test_hybrid_search_async_uses_keyword_signal_to_rerank(tmp_path) -> None:
    store = LocalVectorStore(tmp_path / "storage")
    chunks = [
        DocumentChunk(
            chunk_id="ops-1",
            source_path="ops.md",
            content="部署发布前需要执行回归测试。",
            index=0,
            char_count=14,
        ),
        DocumentChunk(
            chunk_id="finance-1",
            source_path="finance.md",
            content="财务审批流程需要部门负责人确认并归档。",
            index=0,
            char_count=19,
        ),
    ]
    embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    manifest = KnowledgeBaseManifest(
        source_dir="data/docs",
        embedding_model="flat-model",
        chunk_size=300,
        chunk_overlap=50,
        indexed_at="2026-04-19T00:00:00+00:00",
        document_count=2,
        chunk_count=2,
    )
    await store.save_async(chunks, embeddings, manifest)

    retriever = Retriever(
        store=store,
        embedder=FlatEmbedder(),
        threshold=0.0,
        hybrid_config=HybridSearchConfig(
            mode="hybrid", vector_weight=0.2, keyword_weight=0.8
        ),
    )

    results = await retriever.search_async("财务审批流程", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.source_path == "finance.md"


def test_rerank_reorders_vector_candidates_by_query_coverage(tmp_path) -> None:
    store = LocalVectorStore(tmp_path / "storage")
    chunks = [
        DocumentChunk(
            chunk_id="general-1",
            source_path="general.md",
            content="这是一个通用制度说明文档。",
            index=0,
            char_count=13,
        ),
        DocumentChunk(
            chunk_id="finance-1",
            source_path="finance.md",
            content="财务审批流程需要部门负责人确认并归档。",
            index=0,
            char_count=19,
        ),
    ]
    embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    manifest = KnowledgeBaseManifest(
        source_dir="data/docs",
        embedding_model="flat-model",
        chunk_size=300,
        chunk_overlap=50,
        indexed_at="2026-04-19T00:00:00+00:00",
        document_count=2,
        chunk_count=2,
    )
    store.save(chunks, embeddings, manifest)

    retriever = Retriever(
        store=store,
        embedder=FlatEmbedder(),
        threshold=0.0,
        hybrid_config=HybridSearchConfig(mode="vector", candidate_multiplier=2),
        rerank_config=RerankConfig(
            enabled=True,
            base_weight=0.1,
            keyword_weight=0.5,
            phrase_weight=0.4,
            candidate_multiplier=2,
        ),
    )

    results = retriever.search("财务审批流程", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.source_path == "finance.md"


@pytest.mark.anyio
async def test_rerank_async_reorders_vector_candidates_by_query_coverage(
    tmp_path,
) -> None:
    store = LocalVectorStore(tmp_path / "storage")
    chunks = [
        DocumentChunk(
            chunk_id="general-1",
            source_path="general.md",
            content="这是一个通用制度说明文档。",
            index=0,
            char_count=13,
        ),
        DocumentChunk(
            chunk_id="finance-1",
            source_path="finance.md",
            content="财务审批流程需要部门负责人确认并归档。",
            index=0,
            char_count=19,
        ),
    ]
    embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    manifest = KnowledgeBaseManifest(
        source_dir="data/docs",
        embedding_model="flat-model",
        chunk_size=300,
        chunk_overlap=50,
        indexed_at="2026-04-19T00:00:00+00:00",
        document_count=2,
        chunk_count=2,
    )
    await store.save_async(chunks, embeddings, manifest)

    retriever = Retriever(
        store=store,
        embedder=FlatEmbedder(),
        threshold=0.0,
        hybrid_config=HybridSearchConfig(mode="vector", candidate_multiplier=2),
        rerank_config=RerankConfig(
            enabled=True,
            base_weight=0.1,
            keyword_weight=0.5,
            phrase_weight=0.4,
            candidate_multiplier=2,
        ),
    )

    results = await retriever.search_async("财务审批流程", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.source_path == "finance.md"
