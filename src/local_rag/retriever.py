from __future__ import annotations

from local_rag.models import SearchResult
from local_rag.store import LocalVectorStore


class Retriever:
    def __init__(
        self, store: LocalVectorStore, embedder, threshold: float = 0.0
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.threshold = threshold

    def search(self, question: str, top_k: int) -> list[SearchResult]:
        query_vector = self.embedder.encode([question])[0]
        return self.store.search(
            query_embedding=query_vector,
            top_k=top_k,
            threshold=self.threshold,
        )
