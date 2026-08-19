"""Chroma persistence isolated behind a small repository interface."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from .domain import RetrievedChunk, SourceChunk


class ChromaRepository:
    def __init__(self, path: Path, collection_name: str, embedding_model: str) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - exercised in installed app
            raise RuntimeError("Missing dependency 'chromadb'. Run: pip install -e .") from exc

        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._name = collection_name
        self._embedding_model = embedding_model
        self._collection = self._open_collection()

    def _open_collection(self):
        collection = self._client.get_or_create_collection(
            name=self._name,
            metadata={"hnsw:space": "cosine", "embedding_model": self._embedding_model},
        )
        metadata = collection.metadata or {}
        stored_model = metadata.get("embedding_model")
        if stored_model and stored_model != self._embedding_model:
            raise RuntimeError(
                "The knowledge base was built with a different embedding model "
                f"({stored_model}). Clear and re-index it before using {self._embedding_model}."
            )
        return collection

    def count(self) -> int:
        return int(self._collection.count())

    def source_matches_hash(self, source: str, file_hash: str) -> bool:
        result = self._collection.get(where={"source": source}, include=["metadatas"])
        metadatas = result.get("metadatas") or []
        return bool(metadatas) and all(meta.get("file_hash") == file_hash for meta in metadatas)

    def replace_source(
        self, chunks: Sequence[SourceChunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have exactly one embedding.")

        source = chunks[0].source
        self._collection.delete(where={"source": source})
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=[list(vector) for vector in embeddings],
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        available = self.count()
        if available == 0:
            return []

        result: dict[str, Any] = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(top_k, available),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            relevance = max(0.0, min(1.0, 1.0 - float(distance)))
            chunks.append(
                RetrievedChunk(
                    id=str(chunk_id),
                    text=str(text),
                    source=str(metadata["source"]),
                    page=int(metadata["page"]),
                    relevance=relevance,
                )
            )
        return chunks

    def clear(self) -> None:
        with suppress(Exception):
            self._client.delete_collection(self._name)
        self._collection = self._open_collection()
