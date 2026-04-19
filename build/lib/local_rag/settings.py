from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        extra="ignore",
        case_sensitive=False,
    )

    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data" / "docs"
    storage_dir: Path = _PROJECT_ROOT / "storage"
    chroma_collection_name: str = "local-rag-kb"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 4
    similarity_threshold: float = 0.15
    enable_generation: bool = True
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    request_timeout_seconds: int = 60
