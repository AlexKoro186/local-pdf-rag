"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings with safe defaults for a local workstation."""

    project_root: Path
    pdf_dir: Path
    chroma_path: Path
    collection_name: str = "pdf_knowledge_base"
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "embeddinggemma"
    chat_model: str = "gemma3:4b"
    chunk_size: int = 900
    chunk_overlap: int = 150
    embedding_batch_size: int = 32
    top_k: int = 5
    min_relevance: float = 0.20
    max_file_size_mb: int = 50

    def __post_init__(self) -> None:
        if self.chunk_size < 200:
            raise ValueError("RAG_CHUNK_SIZE must be at least 200 characters.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "RAG_CHUNK_OVERLAP must be non-negative and smaller than RAG_CHUNK_SIZE."
            )
        if self.embedding_batch_size < 1:
            raise ValueError("RAG_EMBEDDING_BATCH_SIZE must be positive.")
        if self.top_k < 1:
            raise ValueError("RAG_TOP_K must be positive.")
        if not 0.0 <= self.min_relevance <= 1.0:
            raise ValueError("RAG_MIN_RELEVANCE must be between 0 and 1.")

    @classmethod
    def from_env(cls) -> Settings:
        default_root = Path(__file__).resolve().parents[2]
        load_dotenv(default_root / ".env")
        root = Path(os.getenv("RAG_PROJECT_ROOT", str(default_root))).expanduser().resolve()
        return cls(
            project_root=root,
            pdf_dir=_resolve_path(root, os.getenv("RAG_PDF_DIR", "data/pdfs")),
            chroma_path=_resolve_path(root, os.getenv("RAG_CHROMA_PATH", "chroma_db")),
            collection_name=os.getenv("RAG_COLLECTION_NAME", "pdf_knowledge_base"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "embeddinggemma"),
            chat_model=os.getenv("RAG_CHAT_MODEL", "gemma3:4b"),
            chunk_size=_env_int("RAG_CHUNK_SIZE", 900),
            chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 150),
            embedding_batch_size=_env_int("RAG_EMBEDDING_BATCH_SIZE", 32),
            top_k=_env_int("RAG_TOP_K", 5),
            min_relevance=_env_float("RAG_MIN_RELEVANCE", 0.20),
            max_file_size_mb=_env_int("RAG_MAX_FILE_SIZE_MB", 50),
        )
