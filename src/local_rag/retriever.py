from __future__ import annotations

import asyncio
from dataclasses import dataclass

from local_rag.models import SearchResult
from local_rag.store import LocalVectorStore


@dataclass(slots=True)
class HybridSearchConfig:
    mode: str = "hybrid"
    vector_weight: float = 0.65
    keyword_weight: float = 0.35
    candidate_multiplier: int = 4


class Retriever:
    def __init__(
        self,
        store: LocalVectorStore,
        embedder,
        threshold: float = 0.0,
        hybrid_config: HybridSearchConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.threshold = threshold
        self.hybrid_config = hybrid_config or HybridSearchConfig()

    def search(self, question: str, top_k: int) -> list[SearchResult]:
        if self.hybrid_config.mode == "vector":
            return self._vector_search(question, top_k)

        return self._hybrid_search(question, top_k)

    def _vector_search(self, question: str, top_k: int) -> list[SearchResult]:
        query_vector = self.embedder.encode([question])[0]
        return self.store.search(
            query_embedding=query_vector,
            top_k=top_k,
            threshold=self.threshold,
        )

    def _hybrid_search(self, question: str, top_k: int) -> list[SearchResult]:
        candidate_count = max(top_k, top_k * self.hybrid_config.candidate_multiplier)
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
        if self.hybrid_config.mode == "vector":
            if hasattr(self.embedder, "encode_async"):
                query_embeddings = await self.embedder.encode_async([question])
            else:
                query_embeddings = await asyncio.to_thread(
                    self.embedder.encode, [question]
                )
            return await self.store.search_async(
                query_embedding=query_embeddings[0],
                top_k=top_k,
                threshold=self.threshold,
            )

        candidate_count = max(top_k, top_k * self.hybrid_config.candidate_multiplier)
        if hasattr(self.embedder, "encode_async"):
            query_embeddings = await self.embedder.encode_async([question])
        else:
            query_embeddings = await asyncio.to_thread(self.embedder.encode, [question])

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
        return self._merge_results(vector_results, keyword_results, top_k)

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
