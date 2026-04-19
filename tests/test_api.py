from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from local_rag.api import create_app
from local_rag.service import RagService
from local_rag.settings import Settings


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


def test_index_page_is_served(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "docs",
        storage_dir=tmp_path / "storage",
        enable_generation=False,
        chroma_collection_name="test-ui",
    )
    service = RagService(settings=settings, embedder=FakeEmbedder(), generator=None)
    client = TestClient(create_app(settings=settings, service=service))

    response = client.get("/")

    assert response.status_code == 200
    assert "Local RAG Knowledge Base" in response.text
    assert "上传文档" in response.text


def test_upload_endpoint_ingests_documents(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "docs",
        storage_dir=tmp_path / "storage",
        enable_generation=False,
        chroma_collection_name="test-upload",
    )
    service = RagService(settings=settings, embedder=FakeEmbedder(), generator=None)
    client = TestClient(create_app(settings=settings, service=service))

    response = client.post(
        "/upload",
        files=[
            (
                "files",
                (
                    "guide.md",
                    "财务审批需要部门负责人确认。".encode("utf-8"),
                    "text/markdown",
                ),
            )
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["manifest"]["document_count"] == 1
    assert payload["source_files"] == ["001-guide.md"]

    query_response = client.post(
        "/query",
        json={"question": "财务审批怎么走？", "top_k": 1},
    )

    assert query_response.status_code == 200
    query_payload = query_response.json()
    assert query_payload["matches"][0]["chunk"]["source_path"] == "001-guide.md"
