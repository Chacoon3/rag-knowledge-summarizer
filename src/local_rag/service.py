from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from requests import RequestException

from local_rag.chunking import chunk_document
from local_rag.embeddings import EmbeddingBackend
from local_rag.generation import (
    build_fallback_answer,
    create_generator,
    get_generator_status,
)
from local_rag.loaders import load_documents, load_documents_from_paths
from local_rag.models import (
    ChunkPage,
    DeleteChunkResponse,
    IngestStats,
    KnowledgeBaseManifest,
    LlmStatusResponse,
    QueryResponse,
)
from local_rag.retriever import Retriever
from local_rag.settings import Settings
from local_rag.store import LocalVectorStore


class RagService:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder=None,
        store=None,
        generator=None,
    ) -> None:
        self.settings = settings or Settings()
        self.store = store or LocalVectorStore(
            self.settings.storage_dir,
            collection_name=self.settings.chroma_collection_name,
        )
        self.embedder = embedder or EmbeddingBackend(self.settings.embedding_model)

        if generator is None and self.settings.enable_generation:
            generator = create_generator(self.settings)
        self.generator = generator
        self.current_provider = self.settings.generation_provider

    def ingest(self, source_dir: str | Path | None = None) -> IngestStats:
        source_path = Path(source_dir) if source_dir else self.settings.data_dir
        documents = load_documents(source_path)
        return self._ingest_documents(
            documents,
            source_label=source_path.resolve().as_posix(),
            replace_existing=True,
        )

    def ingest_files(
        self,
        file_paths: Sequence[str | Path],
        source_label: str = "uploaded://api",
    ) -> IngestStats:
        documents = load_documents_from_paths(Path(path) for path in file_paths)
        return self._ingest_documents(
            documents,
            source_label=source_label,
            replace_existing=False,
        )

    def _ingest_documents(
        self,
        documents,
        source_label: str,
        replace_existing: bool,
    ) -> IngestStats:
        if not documents:
            raise ValueError(f"没有在 {source_label} 中找到可入库文档。")

        chunks = []
        for document in documents:
            chunks.extend(
                chunk_document(
                    document,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )
            )

        if not chunks:
            raise ValueError("已发现文档，但没有提取到可用文本内容。")

        embeddings = self.embedder.encode([chunk.content for chunk in chunks])
        manifest = KnowledgeBaseManifest(
            source_dir=source_label,
            embedding_model=self.settings.embedding_model,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            indexed_at=datetime.now(timezone.utc).isoformat(),
            document_count=len(documents),
            chunk_count=len(chunks),
        )
        self.store.save(
            chunks,
            embeddings,
            manifest,
            replace=replace_existing,
        )

        return IngestStats(
            manifest=manifest,
            source_files=[document.source_path for document in documents],
        )

    def manifest(self) -> KnowledgeBaseManifest:
        return self.store.load_manifest()

    def list_chunks(self, page: int = 1, page_size: int = 10) -> ChunkPage:
        return self.store.list_chunks(page=page, page_size=page_size)

    def delete_chunk(self, chunk_id: str) -> DeleteChunkResponse:
        return self.store.delete_chunk(chunk_id)

    def set_generation_provider(self, provider: str) -> LlmStatusResponse:
        provider_name = provider.strip().lower()
        if provider_name not in {
            "local_transformers",
            "ollama",
            "gemini",
            "openrouter",
            "auto",
        }:
            raise ValueError(f"不支持的 LLM provider: {provider}")

        self.current_provider = provider_name
        if not self.settings.enable_generation:
            self.generator = None
        else:
            self.generator = create_generator(self.settings, provider=provider_name)
        return self.get_llm_status()

    def get_llm_status(self) -> LlmStatusResponse:
        return LlmStatusResponse(
            **get_generator_status(
                self.generator,
                configured_provider=self.current_provider,
                generation_enabled=self.settings.enable_generation,
            )
        )

    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        retriever = Retriever(
            store=self.store,
            embedder=self.embedder,
            threshold=self.settings.similarity_threshold,
        )
        matches = retriever.search(question, top_k=top_k or self.settings.top_k)
        if not matches:
            return QueryResponse(
                question=question,
                answer="知识库中没有检索到足够相关的内容。",
                used_generator=False,
                matches=[],
            )

        answer = build_fallback_answer(matches)
        used_generator = False
        if self.generator is not None:
            try:
                generated = self.generator.generate(question, matches)
            except (RequestException, RuntimeError, ImportError, ValueError):
                generated = ""
            if generated:
                answer = generated
                used_generator = True

        return QueryResponse(
            question=question,
            answer=answer,
            used_generator=used_generator,
            matches=matches,
        )
