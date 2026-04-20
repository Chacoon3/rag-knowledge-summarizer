from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import re

from local_rag.models import SearchResult
from local_rag.store import LocalVectorStore


@dataclass(slots=True)
class HybridSearchConfig:
    mode: str = "hybrid"
    vector_weight: float = 0.65
    keyword_weight: float = 0.35
    candidate_multiplier: int = 4


@dataclass(slots=True)
class RerankConfig:
    enabled: bool = True
    base_weight: float = 0.55
    keyword_weight: float = 0.3
    phrase_weight: float = 0.15
    candidate_multiplier: int = 3


@dataclass(slots=True)
class RetrievalCacheConfig:
    enabled: bool = True
    ttl_seconds: int = 300


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)


class Retriever:
    def __init__(
        self,
        store: LocalVectorStore,
        embedder,
        threshold: float = 0.0,
        hybrid_config: HybridSearchConfig | None = None,
        rerank_config: RerankConfig | None = None,
        retrieval_cache=None,
        retrieval_cache_config: RetrievalCacheConfig | None = None,
        retrieval_cache_namespace: str = "default",
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.threshold = threshold
        self.hybrid_config = hybrid_config or HybridSearchConfig()
        self.rerank_config = rerank_config or RerankConfig()
        self.retrieval_cache = retrieval_cache
        self.retrieval_cache_config = retrieval_cache_config or RetrievalCacheConfig()
        self.retrieval_cache_namespace = retrieval_cache_namespace

    def search(self, question: str, top_k: int) -> list[SearchResult]:
        cached_results = self._get_cached_results(question, top_k)
        if cached_results is not None:
            return cached_results

        if self.hybrid_config.mode == "vector":
            results = self._vector_search(question, self._candidate_count(top_k))
        else:
            results = self._hybrid_search(question, top_k)

        reranked = self._rerank_results(question, results, top_k)
        self._set_cached_results(question, top_k, reranked)
        return reranked

    def _vector_search(self, question: str, top_k: int) -> list[SearchResult]:
        query_vector = self.embedder.encode([question])[0]
        return self.store.search(
            query_embedding=query_vector,
            top_k=top_k,
            threshold=self.threshold,
        )

    def _hybrid_search(self, question: str, top_k: int) -> list[SearchResult]:
        candidate_count = self._candidate_count(top_k)
        query_vector = self.embedder.encode([question])[0]
        vector_results = self.store.search(
            query_embedding=query_vector,
            top_k=candidate_count,
            threshold=self.threshold,
        )
        keyword_results = self.store.keyword_search(
            query_text=question,
            top_k=candidate_count,
        )
        return self._merge_results(vector_results, keyword_results, top_k)

    async def search_async(self, question: str, top_k: int) -> list[SearchResult]:
        cached_results = await self._get_cached_results_async(question, top_k)
        if cached_results is not None:
            return cached_results

        if self.hybrid_config.mode == "vector":
            if hasattr(self.embedder, "encode_async"):
                query_embeddings = await self.embedder.encode_async([question])
            else:
                query_embeddings = await asyncio.to_thread(
                    self.embedder.encode, [question]
                )
            results = await self.store.search_async(
                query_embedding=query_embeddings[0],
                top_k=self._candidate_count(top_k),
                threshold=self.threshold,
            )
        else:
            candidate_count = self._candidate_count(top_k)
            if hasattr(self.embedder, "encode_async"):
                query_embeddings = await self.embedder.encode_async([question])
            else:
                query_embeddings = await asyncio.to_thread(
                    self.embedder.encode, [question]
                )

            vector_results, keyword_results = await asyncio.gather(
                self.store.search_async(
                    query_embedding=query_embeddings[0],
                    top_k=candidate_count,
                    threshold=self.threshold,
                ),
                self.store.keyword_search_async(
                    query_text=question,
                    top_k=candidate_count,
                ),
            )
            results = self._merge_results(
                vector_results, keyword_results, candidate_count
            )

        reranked = self._rerank_results(question, results, top_k)
        await self._set_cached_results_async(question, top_k, reranked)
        return reranked

    def _merge_results(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        merged: dict[str, SearchResult] = {}
        vector_scores = {
            result.chunk.chunk_id: result.score for result in vector_results
        }
        keyword_scores = {
            result.chunk.chunk_id: result.score for result in keyword_results
        }

        for result in vector_results + keyword_results:
            chunk_id = result.chunk.chunk_id
            vector_score = vector_scores.get(chunk_id, 0.0)
            keyword_score = keyword_scores.get(chunk_id, 0.0)
            hybrid_score = (
                self.hybrid_config.vector_weight * vector_score
                + self.hybrid_config.keyword_weight * keyword_score
            )
            existing = merged.get(chunk_id)
            if existing is None or hybrid_score > existing.score:
                merged[chunk_id] = SearchResult(chunk=result.chunk, score=hybrid_score)

        results = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def _candidate_count(self, top_k: int) -> int:
        candidate_count = max(top_k, top_k * self.hybrid_config.candidate_multiplier)
        if self.rerank_config.enabled:
            candidate_count = max(
                candidate_count,
                top_k * self.rerank_config.candidate_multiplier,
            )
        return candidate_count

    def _rerank_results(
        self,
        question: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if not self.rerank_config.enabled or not results:
            return results[:top_k]

        query_tokens = self._tokenize(question)
        query_phrase = self._collapse_tokens(query_tokens)
        reranked: list[SearchResult] = []
        for result in results:
            keyword_score = self._coverage_score(query_tokens, result.chunk.content)
            phrase_score = self._phrase_score(query_phrase, result.chunk.content)
            final_score = (
                self.rerank_config.base_weight * result.score
                + self.rerank_config.keyword_weight * keyword_score
                + self.rerank_config.phrase_weight * phrase_score
            )
            reranked.append(SearchResult(chunk=result.chunk, score=final_score))

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall((text or "").lower())

    @staticmethod
    def _collapse_tokens(tokens: list[str]) -> str:
        return "".join(tokens)

    @classmethod
    def _coverage_score(cls, query_tokens: list[str], content: str) -> float:
        if not query_tokens:
            return 0.0
        content_tokens = set(cls._tokenize(content))
        if not content_tokens:
            return 0.0
        matched = sum(1 for token in set(query_tokens) if token in content_tokens)
        return matched / len(set(query_tokens))

    @classmethod
    def _phrase_score(cls, query_phrase: str, content: str) -> float:
        if not query_phrase:
            return 0.0
        collapsed_content = cls._collapse_tokens(cls._tokenize(content))
        return 1.0 if query_phrase in collapsed_content else 0.0

    def _get_cached_results(
        self,
        question: str,
        top_k: int,
    ) -> list[SearchResult] | None:
        if not self._retrieval_cache_enabled():
            return None

        key = self._build_cache_key(question, top_k)
        payloads = self.retrieval_cache.get_json_many([key])
        payload = payloads.get(key)
        if payload is None:
            return None
        return [SearchResult.model_validate(item) for item in payload]

    async def _get_cached_results_async(
        self,
        question: str,
        top_k: int,
    ) -> list[SearchResult] | None:
        if not self._retrieval_cache_enabled():
            return None

        key = self._build_cache_key(question, top_k)
        payloads = await self.retrieval_cache.get_json_many_async([key])
        payload = payloads.get(key)
        if payload is None:
            return None
        return [SearchResult.model_validate(item) for item in payload]

    def _set_cached_results(
        self,
        question: str,
        top_k: int,
        results: list[SearchResult],
    ) -> None:
        if not self._retrieval_cache_enabled():
            return

        self.retrieval_cache.set_json_many(
            {
                self._build_cache_key(question, top_k): [
                    result.model_dump() for result in results
                ]
            },
            ttl_seconds=self.retrieval_cache_config.ttl_seconds,
        )

    async def _set_cached_results_async(
        self,
        question: str,
        top_k: int,
        results: list[SearchResult],
    ) -> None:
        if not self._retrieval_cache_enabled():
            return

        await self.retrieval_cache.set_json_many_async(
            {
                self._build_cache_key(question, top_k): [
                    result.model_dump() for result in results
                ]
            },
            ttl_seconds=self.retrieval_cache_config.ttl_seconds,
        )

    def _retrieval_cache_enabled(self) -> bool:
        return (
            self.retrieval_cache is not None
            and self.retrieval_cache_config.enabled
            and self.retrieval_cache_config.ttl_seconds > 0
        )

    def _build_cache_key(self, question: str, top_k: int) -> str:
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
        return self.retrieval_cache.build_key(
            "retrieval",
            self.retrieval_cache_namespace,
            self.hybrid_config.mode,
            f"topk_{top_k}",
            digest,
        )
