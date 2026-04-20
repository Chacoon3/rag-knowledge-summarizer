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


def test_upload_endpoint_appends_instead_of_replacing_existing_docs(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "docs",
        storage_dir=tmp_path / "storage",
        enable_generation=False,
        chroma_collection_name="test-upload-append",
    )
    service = RagService(settings=settings, embedder=FakeEmbedder(), generator=None)
    client = TestClient(create_app(settings=settings, service=service))

    first_upload = client.post(
        "/upload",
        files=[
            (
                "files",
                (
                    "finance.md",
                    "财务审批需要部门负责人确认。".encode("utf-8"),
                    "text/markdown",
                ),
            )
        ],
    )
    assert first_upload.status_code == 200

    second_upload = client.post(
        "/upload",
        files=[
            (
                "files",
                (
                    "deploy.md",
                    "部署发布前需要执行回归测试。".encode("utf-8"),
                    "text/markdown",
                ),
            )
        ],
    )
    assert second_upload.status_code == 200

    chunks_response = client.get("/chunks", params={"page": 1, "page_size": 20})
    assert chunks_response.status_code == 200
    chunks_payload = chunks_response.json()
    source_paths = {item["chunk"]["source_path"] for item in chunks_payload["items"]}
    assert "001-finance.md" in source_paths
    assert "001-deploy.md" in source_paths

    manifest_response = client.get("/manifest")
    assert manifest_response.status_code == 200
    manifest_payload = manifest_response.json()
    assert manifest_payload["document_count"] == 2


def test_chunks_endpoint_supports_pagination(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "docs",
        storage_dir=tmp_path / "storage",
        enable_generation=False,
        chroma_collection_name="test-chunks",
        chunk_size=8,
        chunk_overlap=2,
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
                    "财务审批需要部门负责人确认。部署发布前需要执行回归测试。".encode(
                        "utf-8"
                    ),
                    "text/markdown",
                ),
            )
        ],
    )

    assert response.status_code == 200

    page_one = client.get("/chunks", params={"page": 1, "page_size": 2})
    assert page_one.status_code == 200
    payload_one = page_one.json()
    assert payload_one["page"] == 1
    assert payload_one["page_size"] == 2
    assert payload_one["total"] >= 3
    assert payload_one["total_pages"] >= 2
    assert len(payload_one["items"]) == 2

    page_two = client.get("/chunks", params={"page": 2, "page_size": 2})
    assert page_two.status_code == 200
    payload_two = page_two.json()
    assert payload_two["page"] == 2
    assert len(payload_two["items"]) >= 1


def test_delete_chunk_endpoint_removes_item_and_updates_counts(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "docs",
        storage_dir=tmp_path / "storage",
        enable_generation=False,
        chroma_collection_name="test-delete",
        chunk_size=8,
        chunk_overlap=2,
    )
    service = RagService(settings=settings, embedder=FakeEmbedder(), generator=None)
    client = TestClient(create_app(settings=settings, service=service))

    upload_response = client.post(
        "/upload",
        files=[
            (
                "files",
                (
                    "guide.md",
                    "财务审批需要部门负责人确认。部署发布前需要执行回归测试。".encode(
                        "utf-8"
                    ),
                    "text/markdown",
                ),
            )
        ],
    )
    assert upload_response.status_code == 200

    page_response = client.get("/chunks", params={"page": 1, "page_size": 5})
    assert page_response.status_code == 200
    page_payload = page_response.json()
    chunk_id = page_payload["items"][0]["chunk"]["chunk_id"]
    total_before = page_payload["total"]

    delete_response = client.delete(f"/chunks/{chunk_id}")
    assert delete_response.status_code == 200
    delete_payload = delete_response.json()
    assert delete_payload["deleted"] is True
    assert delete_payload["chunk_id"] == chunk_id
    assert delete_payload["remaining_chunks"] == total_before - 1

    refreshed_page = client.get("/chunks", params={"page": 1, "page_size": 10})
    assert refreshed_page.status_code == 200
    refreshed_payload = refreshed_page.json()
    chunk_ids = [item["chunk"]["chunk_id"] for item in refreshed_payload["items"]]
    assert chunk_id not in chunk_ids

    manifest_response = client.get("/manifest")
    assert manifest_response.status_code == 200
    manifest_payload = manifest_response.json()
    assert manifest_payload["chunk_count"] == total_before - 1
