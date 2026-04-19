from __future__ import annotations

import requests

from local_rag.models import SearchResult


def build_fallback_answer(matches: list[SearchResult]) -> str:
    if not matches:
        return "知识库中没有检索到相关内容。"

    lines = ["未调用本地 LLM，以下是最相关的知识片段："]
    for match in matches:
        excerpt = " ".join(match.chunk.content.split())[:180]
        lines.append(f"{match.chunk.source_path} | score={match.score:.3f} | {excerpt}")
    return "\n".join(lines)


class OllamaGenerator:
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(self, question: str, matches: list[SearchResult]) -> str:
        context = self.build_context(matches)
        prompt = (
            "你是本地 RAG 知识库助手。只能基于提供的上下文作答。"
            "如果上下文不足，请直接说明。回答时尽量给出来源文件。\n\n"
            f"问题：{question}\n\n"
            f"上下文：\n{context}\n\n"
            "请输出中文答案。"
        )

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("response", "").strip()

    @staticmethod
    def build_context(matches: list[SearchResult]) -> str:
        blocks = []
        for index, match in enumerate(matches, start=1):
            blocks.append(
                f"[{index}] 来源: {match.chunk.source_path}\n{match.chunk.content}"
            )
        return "\n\n".join(blocks)
