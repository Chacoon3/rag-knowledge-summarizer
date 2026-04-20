from __future__ import annotations

import asyncio

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from local_rag.models import SearchResult


def build_fallback_answer(matches: list[SearchResult]) -> str:
    if not matches:
        return "知识库中没有检索到相关内容。"

    lines = ["未调用本地 LLM，以下是最相关的知识片段："]
    for match in matches:
        excerpt = " ".join(match.chunk.content.split())[:180]
        lines.append(f"{match.chunk.source_path} | score={match.score:.3f} | {excerpt}")
    return "\n".join(lines)


def build_prompt(question: str, matches: list[SearchResult]) -> str:
    context = build_context(matches)
    return (
        "你是本地 RAG 知识库助手。只能基于提供的上下文作答。"
        "如果上下文不足，请直接说明。回答时尽量给出来源文件。\n\n"
        f"问题：{question}\n\n"
        f"上下文：\n{context}\n\n"
        "请输出中文答案。"
    )


def build_context(matches: list[SearchResult]) -> str:
    blocks = []
    for index, match in enumerate(matches, start=1):
        blocks.append(
            f"[{index}] 来源: {match.chunk.source_path}\n{match.chunk.content}"
        )
    return "\n\n".join(blocks)


class MultiGenerator:
    def __init__(self, generators: list) -> None:
        self.generators = generators
        self.provider_name = "multi"

    def generate(self, question: str, matches: list[SearchResult]) -> str:
        last_error: Exception | None = None
        for generator in self.generators:
            try:
                answer = generator.generate(question, matches)
            except (httpx.HTTPError, RuntimeError, ImportError, ValueError) as exc:
                last_error = exc
                continue
            if answer:
                return answer

        if last_error is not None:
            raise last_error
        return ""

    async def agenerate(self, question: str, matches: list[SearchResult]) -> str:
        last_error: Exception | None = None
        for generator in self.generators:
            try:
                if hasattr(generator, "agenerate"):
                    answer = await generator.agenerate(question, matches)
                else:
                    answer = await asyncio.to_thread(
                        generator.generate, question, matches
                    )
            except (httpx.HTTPError, RuntimeError, ImportError, ValueError) as exc:
                last_error = exc
                continue
            if answer:
                return answer

        if last_error is not None:
            raise last_error
        return ""


class OllamaGenerator:
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.provider_name = "ollama"

    def generate(self, question: str, matches: list[SearchResult]) -> str:
        prompt = build_prompt(question, matches)
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
        response.raise_for_status()
        payload = response.json()
        return payload.get("response", "").strip()

    async def agenerate(self, question: str, matches: list[SearchResult]) -> str:
        prompt = build_prompt(question, matches)
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
        response.raise_for_status()
        payload = response.json()
        return payload.get("response", "").strip()


class GeminiGenerator:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_seconds: int = 60,
    ) -> None:
        if not api_key:
            raise ValueError("Gemini API key 未配置。")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.provider_name = "gemini"

    def generate(self, question: str, matches: list[SearchResult]) -> str:
        prompt = build_prompt(question, matches)
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/models/{self.model}:generateContent",
                params={"key": self.api_key},
                json={
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                            ]
                        }
                    ]
                },
            )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts).strip()

    async def agenerate(self, question: str, matches: list[SearchResult]) -> str:
        prompt = build_prompt(question, matches)
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/models/{self.model}:generateContent",
                params={"key": self.api_key},
                json={
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt},
                            ]
                        }
                    ]
                },
            )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts).strip()


class OpenRouterGenerator:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: int = 60,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key 未配置。")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.provider_name = "openrouter"

    def generate(self, question: str, matches: list[SearchResult]) -> str:
        prompt = build_prompt(question, matches)
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                },
            )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "").strip()

    async def agenerate(self, question: str, matches: list[SearchResult]) -> str:
        prompt = build_prompt(question, matches)
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                },
            )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "").strip()


class LocalTransformersGenerator:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 384,
        temperature: float = 0.2,
        top_p: float = 0.9,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.trust_remote_code = trust_remote_code
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._resolved_device = None
        self.provider_name = "local_transformers"

    def generate(self, question: str, matches: list[SearchResult]) -> str:
        tokenizer, model, torch, resolved_device = self._ensure_model_loaded()
        prompt = build_prompt(question, matches)
        rendered_prompt = prompt
        if hasattr(tokenizer, "apply_chat_template"):
            rendered_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

        model_inputs = tokenizer(rendered_prompt, return_tensors="pt")
        model_inputs = {
            key: value.to(resolved_device) for key, value in model_inputs.items()
        }
        input_length = model_inputs["input_ids"].shape[1]
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p
        else:
            generation_kwargs["do_sample"] = False

        with torch.inference_mode():
            output = model.generate(**model_inputs, **generation_kwargs)
        generated_ids = output[0][input_length:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    async def agenerate(self, question: str, matches: list[SearchResult]) -> str:
        return await asyncio.to_thread(self.generate, question, matches)

    def _ensure_model_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model, self._torch, self._resolved_device

        resolved_device = self._resolve_device(torch)
        torch_dtype = (
            torch.float16 if resolved_device.startswith("cuda") else torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=self.trust_remote_code,
        )
        model.to(resolved_device)
        model.eval()

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._resolved_device = resolved_device
        return tokenizer, model, torch, resolved_device

    def _resolve_device(self, torch) -> str:
        requested = self.device.lower().strip()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("本机未检测到可用 CUDA，无法使用本地 GPU 推理。")
        return requested

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def resolved_device(self) -> str:
        return self._resolved_device or self.device


def detect_cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def get_generator_status(generator, configured_provider: str, generation_enabled: bool):
    provider = (
        getattr(generator, "provider_name", "fallback") if generator else "disabled"
    )
    local_model_loaded = False
    local_model_name = ""
    local_model_device = ""
    message = ""

    if isinstance(generator, LocalTransformersGenerator):
        local_model_loaded = generator.is_loaded
        local_model_name = generator.model_name
        local_model_device = generator.resolved_device
        message = (
            "本地模型已加载"
            if local_model_loaded
            else "本地模型尚未加载，首次问答时会按需加载"
        )
    elif isinstance(generator, MultiGenerator):
        message = "当前使用自动 provider 选择链"
    elif generator is None and not generation_enabled:
        message = "已禁用 LLM 生成"

    return {
        "provider": provider,
        "configured_provider": configured_provider,
        "generation_enabled": generation_enabled,
        "cuda_available": detect_cuda_available(),
        "local_model_loaded": local_model_loaded,
        "local_model_name": local_model_name,
        "local_model_device": local_model_device,
        "available_providers": ["local_transformers", "ollama", "gemini", "openrouter"],
        "message": message,
    }


def create_generator(settings, provider: str | None = None):
    if not settings.enable_generation:
        return None

    provider_name = (provider or settings.generation_provider).lower().strip()
    if provider_name != (provider or settings.generation_provider):
        provider_name = provider_name
    if provider_name == "auto":
        generators = []
        if settings.gemini_api_key:
            generators.append(
                GeminiGenerator(
                    api_key=settings.gemini_api_key,
                    model=settings.gemini_model,
                    base_url=settings.gemini_base_url,
                    timeout_seconds=settings.request_timeout_seconds,
                )
            )
        if settings.openrouter_api_key:
            generators.append(
                OpenRouterGenerator(
                    api_key=settings.openrouter_api_key,
                    model=settings.openrouter_model,
                    base_url=settings.openrouter_base_url,
                    timeout_seconds=settings.request_timeout_seconds,
                )
            )
        generators.append(
            OllamaGenerator(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                timeout_seconds=settings.request_timeout_seconds,
            )
        )
        return MultiGenerator(generators)

    if provider_name == "gemini":
        return GeminiGenerator(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            base_url=settings.gemini_base_url,
            timeout_seconds=settings.request_timeout_seconds,
        )
    if provider_name == "openrouter":
        return OpenRouterGenerator(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            base_url=settings.openrouter_base_url,
            timeout_seconds=settings.request_timeout_seconds,
        )
    if provider_name == "local_transformers":
        return LocalTransformersGenerator(
            model_name=settings.local_llm_model,
            device=settings.local_llm_device,
            max_new_tokens=settings.local_llm_max_new_tokens,
            temperature=settings.local_llm_temperature,
            top_p=settings.local_llm_top_p,
            trust_remote_code=settings.local_llm_trust_remote_code,
        )
    if provider_name == "ollama":
        return OllamaGenerator(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.request_timeout_seconds,
        )

    raise ValueError(
        f"不支持的 LLM provider: {provider or settings.generation_provider}"
    )
