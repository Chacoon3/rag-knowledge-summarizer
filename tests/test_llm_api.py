from __future__ import annotations

from fastapi.testclient import TestClient

from local_rag.api import create_app
from local_rag.service import RagService
from local_rag.settings import Settings


def test_llm_status_endpoint_returns_runtime_info(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "docs",
        storage_dir=tmp_path / "storage",
        enable_generation=False,
    )
    service = RagService(settings=settings, generator=None)
    client = TestClient(create_app(settings=settings, service=service))

    response = client.get("/llm/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["generation_enabled"] is False
    assert payload["configured_provider"] == "auto"
    assert "local_transformers" in payload["available_providers"]


def test_llm_provider_can_be_switched_at_runtime(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "docs",
        storage_dir=tmp_path / "storage",
        enable_generation=True,
        generation_provider="ollama",
    )
    service = RagService(settings=settings, generator=None)
    client = TestClient(create_app(settings=settings, service=service))

    response = client.put("/llm/provider", json={"provider": "local_transformers"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["configured_provider"] == "local_transformers"
    assert payload["provider"] == "local_transformers"
