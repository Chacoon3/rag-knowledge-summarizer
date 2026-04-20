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
    retrieval_mode: str = "hybrid"
    hybrid_vector_weight: float = 0.65
    hybrid_keyword_weight: float = 0.35
    hybrid_candidate_multiplier: int = 4
    rerank_enabled: bool = True
    rerank_base_weight: float = 0.55
    rerank_keyword_weight: float = 0.3
    rerank_phrase_weight: float = 0.15
    rerank_candidate_multiplier: int = 3
    enable_generation: bool = True
    generation_provider: str = "auto"
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_model: str = "gemini-2.0-flash"
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "deepseek/deepseek-chat-v3-0324:free"
    local_llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_llm_device: str = "auto"
    local_llm_max_new_tokens: int = 384
    local_llm_temperature: float = 0.2
    local_llm_top_p: float = 0.9
    local_llm_trust_remote_code: bool = False
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    request_timeout_seconds: int = 60
