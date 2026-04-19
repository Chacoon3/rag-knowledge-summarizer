from __future__ import annotations

import numpy as np

from local_rag.models import DocumentChunk, KnowledgeBaseManifest
from local_rag.retriever import Retriever
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
