from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LoadedDocument(BaseModel):
    source_path: str
    content: str


class DocumentChunk(BaseModel):
    chunk_id: str
    source_path: str
    content: str
    index: int
    char_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float


class KnowledgeBaseManifest(BaseModel):
    source_dir: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    indexed_at: str
    document_count: int
    chunk_count: int


class IngestStats(BaseModel):
    manifest: KnowledgeBaseManifest
    source_files: list[str]


class QueryResponse(BaseModel):
    question: str
    answer: str
    used_generator: bool
    matches: list[SearchResult]


class ChunkPageItem(BaseModel):
    chunk: DocumentChunk


class ChunkPage(BaseModel):
    items: list[ChunkPageItem]
    page: int
    page_size: int
    total: int
    total_pages: int


class DeleteChunkResponse(BaseModel):
    deleted: bool
    chunk_id: str
    remaining_chunks: int
    remaining_documents: int


class LlmProviderUpdateRequest(BaseModel):
    provider: str = Field(min_length=1)


class LlmStatusResponse(BaseModel):
    provider: str
    configured_provider: str
    generation_enabled: bool
    cuda_available: bool
    local_model_loaded: bool
    local_model_name: str
    local_model_device: str
    available_providers: list[str]
    message: str = ""
