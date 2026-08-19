"""Dependency construction kept separate from UI and CLI entry points."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .ingestion import PdfIngestor
from .ollama_client import OllamaGateway
from .rag import RAGService
from .vector_store import ChromaRepository


@dataclass(frozen=True, slots=True)
class Components:
    settings: Settings
    gateway: OllamaGateway
    repository: ChromaRepository
    ingestor: PdfIngestor
    rag: RAGService


def create_components(settings: Settings | None = None) -> Components:
    settings = settings or Settings.from_env()
    gateway = OllamaGateway(
        host=settings.ollama_host,
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
    )
    repository = ChromaRepository(
        path=settings.chroma_path,
        collection_name=settings.collection_name,
        embedding_model=settings.embedding_model,
    )
    ingestor = PdfIngestor(settings, gateway, repository)
    rag = RAGService(gateway, repository, gateway)
    return Components(settings, gateway, repository, ingestor, rag)
