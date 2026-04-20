from __future__ import annotations

import asyncio
from collections import Counter
import math
from pathlib import Path
import re

import chromadb
import numpy as np

from local_rag.models import (
    ChunkPage,
    ChunkPageItem,
    DeleteChunkResponse,
    DocumentChunk,
    KnowledgeBaseManifest,
    SearchResult,
)


class KnowledgeBaseNotFoundError(FileNotFoundError):
    pass


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)


class LocalVectorStore:
    def __init__(
        self,
        storage_dir: Path,
        collection_name: str = "local-rag-kb",
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.collection_name = collection_name
        self.chroma_dir = self.storage_dir / "chroma"
        self.manifest_path = self.storage_dir / "manifest.json"
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.chroma_dir))
        return self._client

    def save(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        manifest: KnowledgeBaseManifest,
        replace: bool = True,
    ) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        collection = self._reset_collection() if replace else self._get_collection()
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[self._serialize_metadata(chunk) for chunk in chunks],
            embeddings=np.asarray(embeddings, dtype=np.float32).tolist(),
        )
        if replace or not self.manifest_path.exists():
            self.manifest_path.write_text(
                manifest.model_dump_json(indent=2), encoding="utf-8"
            )
        else:
            self._merge_manifest(collection, manifest)

    async def save_async(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        manifest: KnowledgeBaseManifest,
        replace: bool = True,
    ) -> None:
        await asyncio.to_thread(self.save, chunks, embeddings, manifest, replace)

    def load_manifest(self) -> KnowledgeBaseManifest:
        if not self.is_ready():
            raise KnowledgeBaseNotFoundError("本地知识库尚未建立，请先执行 ingest。")

        return KnowledgeBaseManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )

    async def load_manifest_async(self) -> KnowledgeBaseManifest:
        return await asyncio.to_thread(self.load_manifest)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        threshold: float = 0.0,
    ) -> list[SearchResult]:
        if not self.is_ready():
            raise KnowledgeBaseNotFoundError("本地知识库尚未建立，请先执行 ingest。")

        collection = self.client.get_collection(self.collection_name)
        payload = collection.query(
            query_embeddings=[np.asarray(query_embedding, dtype=np.float32).tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = payload.get("ids", [[]])[0]
        documents = payload.get("documents", [[]])[0]
        metadatas = payload.get("metadatas", [[]])[0]
        distances = payload.get("distances", [[]])[0]

        results: list[SearchResult] = []
        for chunk_id, content, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            score = 1.0 - float(distance)
            if score < threshold:
                continue

            chunk = self._build_chunk(chunk_id, content, metadata)
            results.append(SearchResult(chunk=chunk, score=score))

        return results

    async def search_async(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        threshold: float = 0.0,
    ) -> list[SearchResult]:
        return await asyncio.to_thread(self.search, query_embedding, top_k, threshold)

    def keyword_search(
        self,
        query_text: str,
        top_k: int,
        threshold: float = 0.0,
    ) -> list[SearchResult]:
        if not self.is_ready():
            raise KnowledgeBaseNotFoundError("本地知识库尚未建立，请先执行 ingest。")

        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return []

        payload = self.client.get_collection(self.collection_name).get(
            include=["documents", "metadatas"]
        )
        ids = payload.get("ids", [])
        documents = payload.get("documents", [])
        metadatas = payload.get("metadatas", [])

        results: list[SearchResult] = []
        for chunk_id, content, metadata in zip(ids, documents, metadatas):
            score = self._keyword_score(query_tokens, content or "")
            if score <= 0 or score < threshold:
                continue
            results.append(
                SearchResult(
                    chunk=self._build_chunk(chunk_id, content, metadata),
                    score=score,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    async def keyword_search_async(
        self,
        query_text: str,
        top_k: int,
        threshold: float = 0.0,
    ) -> list[SearchResult]:
        return await asyncio.to_thread(
            self.keyword_search,
            query_text,
            top_k,
            threshold,
        )

    def list_chunks(self, page: int = 1, page_size: int = 10) -> ChunkPage:
        if page <= 0:
            raise ValueError("page 必须大于 0")
        if page_size <= 0:
            raise ValueError("page_size 必须大于 0")
        if not self.is_ready():
            raise KnowledgeBaseNotFoundError("本地知识库尚未建立，请先执行 ingest。")

        collection = self.client.get_collection(self.collection_name)
        total = collection.count()
        total_pages = max((total + page_size - 1) // page_size, 1)
        page = min(page, total_pages)
        offset = (page - 1) * page_size
        payload = collection.get(
            limit=page_size,
            offset=offset,
            include=["documents", "metadatas"],
        )

        ids = payload.get("ids", [])
        documents = payload.get("documents", [])
        metadatas = payload.get("metadatas", [])
        items: list[ChunkPageItem] = []
        for chunk_id, content, metadata in zip(ids, documents, metadatas):
            chunk = self._build_chunk(chunk_id, content, metadata)
            items.append(ChunkPageItem(chunk=chunk))

        return ChunkPage(
            items=items,
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
        )

    async def list_chunks_async(self, page: int = 1, page_size: int = 10) -> ChunkPage:
        return await asyncio.to_thread(self.list_chunks, page, page_size)

    def delete_chunk(self, chunk_id: str) -> DeleteChunkResponse:
        if not chunk_id.strip():
            raise ValueError("chunk_id 不能为空")
        if not self.is_ready():
            raise KnowledgeBaseNotFoundError("本地知识库尚未建立，请先执行 ingest。")

        collection = self.client.get_collection(self.collection_name)
        payload = collection.get(ids=[chunk_id], include=["metadatas"])
        ids = payload.get("ids", [])
        if not ids:
            raise ValueError(f"未找到 chunk: {chunk_id}")

        collection.delete(ids=[chunk_id])
        remaining_chunks = collection.count()
        remaining_documents = self._refresh_manifest_counts(collection)

        return DeleteChunkResponse(
            deleted=True,
            chunk_id=chunk_id,
            remaining_chunks=remaining_chunks,
            remaining_documents=remaining_documents,
        )

    async def delete_chunk_async(self, chunk_id: str) -> DeleteChunkResponse:
        return await asyncio.to_thread(self.delete_chunk, chunk_id)

    def is_ready(self) -> bool:
        if not self.manifest_path.exists():
            return False

        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception:
            return False

        return collection.count() > 0

    async def is_ready_async(self) -> bool:
        return await asyncio.to_thread(self.is_ready)

    def _reset_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        return self._get_collection()

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _serialize_metadata(
        chunk: DocumentChunk,
    ) -> dict[str, str | int | float | bool]:
        metadata: dict[str, str | int | float | bool] = {
            "source_path": chunk.source_path,
            "index": chunk.index,
            "char_count": chunk.char_count,
        }
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
        return metadata

    @staticmethod
    def _build_chunk(
        chunk_id: str,
        content: str | None,
        metadata,
    ) -> DocumentChunk:
        metadata = dict(metadata or {})
        return DocumentChunk(
            chunk_id=chunk_id,
            source_path=str(metadata.get("source_path", "unknown")),
            content=content or "",
            index=int(metadata.get("index", 0)),
            char_count=int(metadata.get("char_count", len(content or ""))),
            metadata=metadata,
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall((text or "").lower())

    @classmethod
    def _keyword_score(cls, query_tokens: list[str], content: str) -> float:
        content_tokens = cls._tokenize(content)
        if not query_tokens or not content_tokens:
            return 0.0

        query_counter = Counter(query_tokens)
        content_counter = Counter(content_tokens)
        overlap = sum(
            min(query_counter[token], content_counter.get(token, 0))
            for token in query_counter
        )
        if overlap == 0:
            return 0.0

        normalized_score = overlap / math.sqrt(len(query_tokens) * len(content_tokens))
        collapsed_query = "".join(query_tokens)
        collapsed_content = "".join(content_tokens)
        containment_bonus = (
            0.15 if collapsed_query and collapsed_query in collapsed_content else 0.0
        )
        return min(normalized_score + containment_bonus, 1.0)

    def _refresh_manifest_counts(self, collection) -> int:
        if not self.manifest_path.exists():
            return 0

        manifest = KnowledgeBaseManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )
        remaining_chunks = collection.count()
        if remaining_chunks > 0:
            payload = collection.get(include=["metadatas"])
            metadatas = payload.get("metadatas", [])
            source_paths = {
                str(metadata.get("source_path", "unknown"))
                for metadata in metadatas
                if metadata is not None
            }
            manifest.document_count = len(source_paths)
        else:
            manifest.document_count = 0

        manifest.chunk_count = remaining_chunks
        self.manifest_path.write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        return manifest.document_count

    def _merge_manifest(
        self,
        collection,
        latest_manifest: KnowledgeBaseManifest,
    ) -> None:
        manifest = KnowledgeBaseManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )
        payload = collection.get(include=["metadatas"])
        metadatas = payload.get("metadatas", [])
        source_paths = {
            str(metadata.get("source_path", "unknown"))
            for metadata in metadatas
            if metadata is not None
        }
        manifest.source_dir = latest_manifest.source_dir
        manifest.embedding_model = latest_manifest.embedding_model
        manifest.chunk_size = latest_manifest.chunk_size
        manifest.chunk_overlap = latest_manifest.chunk_overlap
        manifest.indexed_at = latest_manifest.indexed_at
        manifest.document_count = len(source_paths)
        manifest.chunk_count = collection.count()
        self.manifest_path.write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
