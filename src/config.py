from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the Constitution RAG system."""

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"
    corpus_path: str = "Data/processed"
    embeddings_path: str = "Data/processed/constitution_embeddings.jsonl"
    index_path: str = "artifacts"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_mode: str = "dense"
    top_k: int = 5
    hybrid_fusion: str = "rrf"
    reranker_enabled: bool = False
    evidence_threshold: float = 0.6
    request_timeout_seconds: int = 30
    log_level: str = "INFO"

    @classmethod
    def from_mapping(cls, values: dict | None = None) -> "Settings":
        mapping = dict(values or {})

        if mapping.get("llm_provider") is None:
            mapping["llm_provider"] = os.getenv("LLM_PROVIDER", "openai")
        if mapping.get("llm_model") is None:
            mapping["llm_model"] = os.getenv("LLM_MODEL", "gpt-4o-mini")
        if mapping.get("llm_api_key") is None:
            mapping["llm_api_key"] = os.getenv("LLM_API_KEY", "")
        if mapping.get("embedding_model") is None:
            mapping["embedding_model"] = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        if mapping.get("corpus_path") is None:
            mapping["corpus_path"] = os.getenv("CORPUS_PATH", "Data/processed")
        if mapping.get("embeddings_path") is None:
            mapping["embeddings_path"] = os.getenv(
                "EMBEDDINGS_PATH",
                "Data/processed/constitution_embeddings.jsonl",
            )
        if mapping.get("index_path") is None:
            mapping["index_path"] = os.getenv("INDEX_PATH", "artifacts")
        if mapping.get("chunk_size") is None:
            mapping["chunk_size"] = int(os.getenv("CHUNK_SIZE", "1200"))
        if mapping.get("chunk_overlap") is None:
            mapping["chunk_overlap"] = int(os.getenv("CHUNK_OVERLAP", "200"))
        if mapping.get("retrieval_mode") is None:
            mapping["retrieval_mode"] = os.getenv("RETRIEVAL_MODE", "dense")
        if mapping.get("top_k") is None:
            mapping["top_k"] = int(os.getenv("TOP_K", "5"))
        if mapping.get("hybrid_fusion") is None:
            mapping["hybrid_fusion"] = os.getenv("HYBRID_FUSION", "rrf")
        if mapping.get("reranker_enabled") is None:
            mapping["reranker_enabled"] = os.getenv("RERANKER_ENABLED", "false").lower() in {"1", "true", "yes"}
        if mapping.get("evidence_threshold") is None:
            mapping["evidence_threshold"] = float(os.getenv("EVIDENCE_THRESHOLD", "0.6"))
        if mapping.get("request_timeout_seconds") is None:
            mapping["request_timeout_seconds"] = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
        if mapping.get("log_level") is None:
            mapping["log_level"] = os.getenv("LOG_LEVEL", "INFO")

        self_valid = cls(**mapping)
        self_valid.validate()
        return self_valid

    @classmethod
    def from_env(cls) -> "Settings":
        return cls.from_mapping()

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.evidence_threshold <= 1.0:
            raise ValueError("evidence_threshold must be between 0.0 and 1.0")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if self.retrieval_mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("retrieval_mode must be one of: dense, bm25, hybrid")
        if self.hybrid_fusion not in {"rrf", "weighted"}:
            raise ValueError("hybrid_fusion must be one of: rrf, weighted")

    @property
    def corpus_dir(self) -> Path:
        return Path(self.corpus_path)

    @property
    def index_dir(self) -> Path:
        return Path(self.index_path)
