from __future__ import annotations

from local_rag.chunking import chunk_document, chunk_text
from local_rag.models import LoadedDocument


def test_chunk_text_respects_size_and_overlap() -> None:
    text = "本地 RAG 系统用于私有知识检索。" * 40
    chunks = chunk_text(text, chunk_size=60, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk) <= 60 for chunk in chunks)
    assert chunks[0][-10:] in chunks[1]


def test_chunk_document_builds_metadata() -> None:
    document = LoadedDocument(
        source_path="guides/setup.md",
        content="部署说明。" * 30,
    )

    chunks = chunk_document(document, chunk_size=40, chunk_overlap=8)

    assert chunks[0].source_path == "guides/setup.md"
    assert chunks[0].metadata["source_path"] == "guides/setup.md"
    assert chunks[0].chunk_id
