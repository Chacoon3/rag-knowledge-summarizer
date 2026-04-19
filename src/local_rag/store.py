from __future__ import annotations

from pathlib import Path

import chromadb
import numpy as np

from local_rag.models import (
    ChunkPage,
    ChunkPageItem,
    DocumentChunk,
    KnowledgeBaseManifest,
    SearchResult,
)


class KnowledgeBaseNotFoundError(FileNotFoundError):
    pass


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
    ) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        collection = self._reset_collection()
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[self._serialize_metadata(chunk) for chunk in chunks],
            embeddings=np.asarray(embeddings, dtype=np.float32).tolist(),
        )
        self.manifest_path.write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )

    def load_manifest(self) -> KnowledgeBaseManifest:
        if not self.is_ready():
            raise KnowledgeBaseNotFoundError("本地知识库尚未建立，请先执行 ingest。")

        return KnowledgeBaseManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )

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

            metadata = dict(metadata or {})
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                source_path=str(metadata.get("source_path", "unknown")),
                content=content or "",
                index=int(metadata.get("index", 0)),
                char_count=int(metadata.get("char_count", len(content or ""))),
                metadata=metadata,
            )
            results.append(SearchResult(chunk=chunk, score=score))

        return results

    def list_chunks(self, page: int = 1, page_size: int = 10) -> ChunkPage:
        if page <= 0:
            raise ValueError("page 必须大于 0")
        if page_size <= 0:
            raise ValueError("page_size 必须大于 0")
        if not self.is_ready():
            raise KnowledgeBaseNotFoundError("本地知识库尚未建立，请先执行 ingest。")

        collection = self.client.get_collection(self.collection_name)
        total = collection.count()
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
            metadata = dict(metadata or {})
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                source_path=str(metadata.get("source_path", "unknown")),
                content=content or "",
                index=int(metadata.get("index", 0)),
                char_count=int(metadata.get("char_count", len(content or ""))),
                metadata=metadata,
            )
            items.append(ChunkPageItem(chunk=chunk))

        total_pages = max((total + page_size - 1) // page_size, 1)
        return ChunkPage(
            items=items,
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
        )

    def is_ready(self) -> bool:
        if not self.manifest_path.exists():
            return False

        try:
            collection = self.client.get_collection(self.collection_name)
        except Exception:
            return False

        return collection.count() > 0

    def _reset_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

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
