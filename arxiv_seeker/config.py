"""Centralized configuration via pydantic-settings, loaded from .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FAISS_DIR = DATA_DIR / "faiss"
CACHE_DB_PATH = DATA_DIR / "cache" / "arxiv_seeker.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- arXiv / search ---
    arxiv_rate_limit_seconds: float = Field(3.0, description="Delay between arXiv API calls")
    default_max_results: int = 10
    rerank_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.15
    recency_halflife_days: int = 720  # ~2 years

    # --- Cache ---
    cache_ttl_hours: int = 24
    cache_db_path: Path = CACHE_DB_PATH

    # --- RAG ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50
    top_k_chunks: int = 4
    faiss_dir: Path = FAISS_DIR

    # --- LLM backend ---
    llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None  # e.g. for OpenAI-compatible proxies/gateways
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-4-6"

    # --- Optional integrations ---
    semantic_scholar_api_key: Optional[str] = None


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.faiss_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings