"""Domain models shared by ingestion, retrieval, and presentation layers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SourceChunk:
    id: str
    text: str
    source: str
    page: int
    chunk_index: int
    file_hash: str

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "file_hash": self.file_hash,
        }


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    id: str
    text: str
    source: str
    page: int
    relevance: float

    @property
    def citation(self) -> str:
        return f"{self.source}, p. {self.page}"


@dataclass(slots=True)
class IngestionReport:
    files_seen: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    pages_indexed: int = 0
    chunks_indexed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        return (
            f"{self.files_indexed} indexed, {self.files_skipped} unchanged, "
            f"{self.pages_indexed} pages, {self.chunks_indexed} chunks"
        )
