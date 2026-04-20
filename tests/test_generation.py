from __future__ import annotations

import httpx
from types import SimpleNamespace

import pytest

from local_rag.generation import (
    detect_cuda_available,
    GeminiGenerator,
    get_generator_status,
    LocalTransformersGenerator,
    MultiGenerator,
    OllamaGenerator,
    OpenRouterGenerator,
    create_generator,
)


class FailingGenerator:
    def generate(self, question, matches):
        raise ValueError("failed")


class SuccessGenerator:
    def generate(self, question, matches):
        return "ok"


def test_multi_generator_falls_back_to_next_provider() -> None:
    generator = MultiGenerator([FailingGenerator(), SuccessGenerator()])

    assert generator.generate("q", []) == "ok"


@pytest.mark.anyio
async def test_multi_generator_async_uses_native_async_provider() -> None:
    class AsyncSuccessGenerator:
        async def agenerate(self, question, matches):
            return "ok"

    generator = MultiGenerator([FailingGenerator(), AsyncSuccessGenerator()])

    assert await generator.agenerate("q", []) == "ok"


@pytest.mark.anyio
async def test_ollama_async_generator_uses_httpx_async_client(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "async ok"}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    generator = OllamaGenerator(base_url="http://127.0.0.1:11434", model="qwen")

    assert await generator.agenerate("q", []) == "async ok"


def test_create_generator_auto_prefers_remote_free_models() -> None:
    settings = SimpleNamespace(
        enable_generation=True,
        generation_provider="auto",
        gemini_api_key="gemini-key",
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta",
        gemini_model="gemini-2.0-flash",
        openrouter_api_key="openrouter-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="deepseek/deepseek-chat-v3-0324:free",
        local_llm_model="Qwen/Qwen2.5-1.5B-Instruct",
        local_llm_device="auto",
        local_llm_max_new_tokens=384,
        local_llm_temperature=0.2,
        local_llm_top_p=0.9,
        local_llm_trust_remote_code=False,
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="qwen2.5:7b-instruct",
        request_timeout_seconds=60,
    )

    generator = create_generator(settings)

    assert isinstance(generator, MultiGenerator)
    assert isinstance(generator.generators[0], GeminiGenerator)
    assert isinstance(generator.generators[1], OpenRouterGenerator)
    assert isinstance(generator.generators[2], OllamaGenerator)


def test_create_generator_explicit_provider_requires_config() -> None:
    settings = SimpleNamespace(
        enable_generation=True,
        generation_provider="gemini",
        gemini_api_key="",
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta",
        gemini_model="gemini-2.0-flash",
        openrouter_api_key="",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="deepseek/deepseek-chat-v3-0324:free",
        local_llm_model="Qwen/Qwen2.5-1.5B-Instruct",
        local_llm_device="auto",
        local_llm_max_new_tokens=384,
        local_llm_temperature=0.2,
        local_llm_top_p=0.9,
        local_llm_trust_remote_code=False,
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="qwen2.5:7b-instruct",
        request_timeout_seconds=60,
    )

    with pytest.raises(ValueError, match="Gemini API key 未配置"):
        create_generator(settings)


def test_create_generator_supports_local_transformers_provider() -> None:
    settings = SimpleNamespace(
        enable_generation=True,
        generation_provider="local_transformers",
        gemini_api_key="",
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta",
        gemini_model="gemini-2.0-flash",
        openrouter_api_key="",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="deepseek/deepseek-chat-v3-0324:free",
        local_llm_model="Qwen/Qwen2.5-1.5B-Instruct",
        local_llm_device="auto",
        local_llm_max_new_tokens=384,
        local_llm_temperature=0.2,
        local_llm_top_p=0.9,
        local_llm_trust_remote_code=False,
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="qwen2.5:7b-instruct",
        request_timeout_seconds=60,
    )

    generator = create_generator(settings)

    assert isinstance(generator, LocalTransformersGenerator)


def test_get_generator_status_reports_local_model_state() -> None:
    generator = LocalTransformersGenerator(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device="auto",
    )

    status = get_generator_status(
        generator,
        configured_provider="local_transformers",
        generation_enabled=True,
    )

    assert status["provider"] == "local_transformers"
    assert status["configured_provider"] == "local_transformers"
    assert status["local_model_loaded"] is False
    assert status["local_model_name"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert status["cuda_available"] == detect_cuda_available()
