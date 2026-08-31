"""Application configuration loaded from environment variables or ``.env``."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings with independent chat and embedding providers."""

    llm_api_key: str = ""
    llm_base_url: str = "http://127.0.0.1:8001/v1"
    llm_model: str = ""
    llm_supports_vision: bool = False
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 2

    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-v4"
    embedding_provider: str = "siliconflow"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = Field(default=10, ge=1, le=100)
    embedding_price_per_1k_cny: float = Field(default=0.0005, ge=0)

    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_chat_model: str = ""
    vision_enabled: bool = True
    siliconflow_vision_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    siliconflow_embedding_model: str = "BAAI/bge-m3"
    siliconflow_vl_embedding_model: str = "Qwen/Qwen3-VL-Embedding-8B"
    vision_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    vision_batch_size: int = Field(default=1, ge=1, le=8)
    modality_text_weight: float = Field(default=1.0, gt=0, le=5)
    modality_table_weight: float = Field(default=0.95, gt=0, le=5)
    modality_image_weight: float = Field(default=0.90, gt=0, le=5)
    rrf_constant: int = Field(default=60, ge=1, le=1000)
    hybrid_vector_weight: float = Field(default=1.0, gt=0, le=5)
    hybrid_graph_weight: float = Field(default=0.85, gt=0, le=5)
    hybrid_max_chunks_per_document: int = Field(default=1, ge=1, le=20)
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidate_k: int = Field(default=40, ge=4, le=200)
    rerank_top_documents: int = Field(default=10, ge=1, le=20)
    rerank_local_candidates_per_query: int = Field(default=6, ge=1, le=50)
    rerank_max_local_candidates: int = Field(default=80, ge=4, le=500)
    rerank_max_chunks_per_document: int = Field(default=2, ge=1, le=20)
    rerank_base_rrf_weight: float = Field(default=1.0, gt=0, le=5)
    rerank_slot_rrf_weight: float = Field(default=1.0, gt=0, le=5)
    rerank_slot_match_top_n: int = Field(default=4, ge=1, le=100)
    rerank_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    rerank_max_retries: int = Field(default=2, ge=0, le=10)
    comparison_tool_enabled: bool = False
    answer_max_context_chunks: int = Field(default=8, ge=2, le=20)
    answer_max_context_chars: int = Field(default=18000, ge=2000, le=100000)

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    chroma_host: str = "localhost"
    chroma_port: int = 8000

    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_doc_changes: str = "doc-changes"
    kafka_consumer_group: str = "agent-knowledge-hub"
    enable_kafka_consumer: bool = True
    cdc_debounce_seconds: float = 2.0
    cdc_max_retries: int = 3

    api_host: str = "0.0.0.0"
    api_port: int = 8080
    upload_dir: str = "./uploads"
    asset_dir: str = "./data/assets"
    catalog_path: str = "./data/catalog.sqlite3"
    checkpoint_path: str = "./data/checkpoints.sqlite3"
    max_upload_mb: int = 50
    max_pdf_pages: int = 500
    max_spreadsheet_rows: int = 100_000
    max_archive_files: int = 10_000
    max_archive_uncompressed_mb: int = 250
    max_json_depth: int = 64
    max_structured_nodes: int = 200_000
    batch_ingest_concurrency: int = 2
    knowledge_extraction_concurrency: int = Field(default=8, ge=1, le=32)

    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 80
    ocr_languages: str = "chi_sim+eng"
    ocr_min_text_chars: int = 40
    default_top_k: int = 8
    max_top_k: int = 20

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_model and self.llm_base_url)

    @property
    def embedding_configured(self) -> bool:
        return bool(self.active_embedding_api_key and self.active_embedding_base_url)

    @property
    def vision_configured(self) -> bool:
        return bool(
            self.vision_enabled
            and self.siliconflow_api_key
            and self.siliconflow_base_url
            and self.active_vision_model
        )

    @property
    def vl_embedding_configured(self) -> bool:
        return bool(self.siliconflow_api_key and self.siliconflow_base_url and self.active_vl_embedding_model)

    @property
    def reranker_configured(self) -> bool:
        return bool(
            not self.rerank_enabled
            or (self.siliconflow_api_key and self.siliconflow_base_url and self.rerank_model)
        )

    @property
    def active_vision_model(self) -> str:
        return self.siliconflow_vision_model or "Qwen/Qwen3-VL-8B-Instruct"

    @property
    def active_vl_embedding_model(self) -> str:
        return self.siliconflow_vl_embedding_model or "Qwen/Qwen3-VL-Embedding-8B"

    @property
    def active_embedding_api_key(self) -> str:
        if self.embedding_provider.lower() == "siliconflow":
            return self.siliconflow_api_key
        return self.embedding_api_key

    @property
    def active_embedding_base_url(self) -> str:
        if self.embedding_provider.lower() == "siliconflow":
            return self.siliconflow_base_url
        return self.embedding_base_url

    @property
    def active_embedding_model(self) -> str:
        if self.embedding_provider.lower() == "siliconflow":
            return self.siliconflow_embedding_model
        return self.embedding_model

    def ensure_runtime_dirs(self) -> None:
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.asset_dir).mkdir(parents=True, exist_ok=True)
        Path(self.catalog_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def _configure_local_proxy_bypass() -> None:
    """Keep local infrastructure out of OS-level HTTP proxies on developer machines."""
    hosts = {"localhost", "127.0.0.1", "::1"}
    llm_host = urlparse(settings.llm_base_url).hostname
    if llm_host in hosts:
        hosts.add(llm_host)
    if settings.chroma_host:
        hosts.add(settings.chroma_host)
    for variable in ("NO_PROXY", "no_proxy"):
        existing = {value.strip() for value in os.environ.get(variable, "").split(",") if value.strip()}
        os.environ[variable] = ",".join(sorted(existing | hosts))


_configure_local_proxy_bypass()
