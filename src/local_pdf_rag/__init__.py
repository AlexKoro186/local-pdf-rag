"""Local-first retrieval-augmented generation for PDF documents."""

from .config import Settings
from .domain import IngestionReport, RetrievedChunk, SourceChunk
from .rag import RAGService

__all__ = ["IngestionReport", "RAGService", "RetrievedChunk", "Settings", "SourceChunk"]
__version__ = "1.0.0"
