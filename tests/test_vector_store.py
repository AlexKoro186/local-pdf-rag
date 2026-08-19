from pathlib import Path

import pytest

from local_pdf_rag.domain import SourceChunk
from local_pdf_rag.vector_store import ChromaRepository


def chunk(chunk_id: str, source: str, file_hash: str) -> SourceChunk:
    return SourceChunk(chunk_id, f"evidence {chunk_id}", source, 1, 0, file_hash)


def test_repository_replaces_searches_and_clears_sources(tmp_path: Path) -> None:
    repository = ChromaRepository(tmp_path / "db", "documents", "test-embedding")
    first = chunk("first", "first.pdf", "hash-1")
    second = chunk("second", "second.pdf", "hash-2")

    repository.replace_source([first], [[1.0, 0.0]])
    repository.replace_source([second], [[0.0, 1.0]])

    assert repository.count() == 2
    assert repository.source_matches_hash("first.pdf", "hash-1")
    assert not repository.source_matches_hash("first.pdf", "wrong")
    results = repository.search([1.0, 0.0], top_k=10)
    assert results[0].id == "first"
    assert results[0].relevance == pytest.approx(1.0)

    changed = SourceChunk("changed", "new content", "first.pdf", 2, 0, "hash-3")
    repository.replace_source([changed], [[0.7, 0.3]])
    assert repository.count() == 2
    assert repository.source_matches_hash("first.pdf", "hash-3")

    repository.clear()
    assert repository.count() == 0
    assert repository.search([1.0, 0.0], top_k=5) == []


def test_repository_rejects_wrong_embedding_model(tmp_path: Path) -> None:
    path = tmp_path / "db"
    ChromaRepository(path, "documents", "model-a")

    with pytest.raises(RuntimeError, match="different embedding model"):
        ChromaRepository(path, "documents", "model-b")


def test_repository_validates_embedding_count(tmp_path: Path) -> None:
    repository = ChromaRepository(tmp_path / "db", "documents", "test-embedding")
    with pytest.raises(ValueError, match="exactly one embedding"):
        repository.replace_source([chunk("a", "a.pdf", "hash")], [])
