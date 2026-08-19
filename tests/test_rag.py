import pytest

from local_pdf_rag.domain import RetrievedChunk
from local_pdf_rag.rag import RAGService


class FakeEmbedder:
    def __init__(self) -> None:
        self.inputs = []

    def embed(self, texts):
        self.inputs.append(list(texts))
        return [[0.1, 0.2] for _ in texts]


class FakeRepository:
    def __init__(self, chunks):
        self.chunks = chunks

    def count(self):
        return len(self.chunks)

    def search(self, query_embedding, top_k):
        assert query_embedding == [0.1, 0.2]
        return self.chunks[:top_k]


class FakeGenerator:
    def stream_chat(self, system_prompt, user_prompt):
        assert "source id" in user_prompt
        yield "Grounded "
        yield "answer [1]."


def test_retrieve_filters_low_relevance_and_duplicate_ids() -> None:
    chunks = [
        RetrievedChunk("a", "high", "one.pdf", 1, 0.9),
        RetrievedChunk("a", "duplicate", "one.pdf", 1, 0.8),
        RetrievedChunk("b", "low", "two.pdf", 2, 0.1),
    ]
    embedder = FakeEmbedder()
    service = RAGService(embedder, FakeRepository(chunks), FakeGenerator())

    result = service.retrieve("question", top_k=5, min_relevance=0.2)

    assert [chunk.id for chunk in result] == ["a"]
    assert embedder.inputs == [["question"]]
    assert service.answer("question", result) == "Grounded answer [1]."


def test_empty_index_does_not_call_embedding_model() -> None:
    embedder = FakeEmbedder()
    service = RAGService(embedder, FakeRepository([]), FakeGenerator())

    chunks = service.retrieve("question", top_k=5, min_relevance=0.0)

    assert chunks == []
    assert embedder.inputs == []
    assert "could not find" in service.answer("question", chunks)


def test_empty_question_is_rejected() -> None:
    service = RAGService(FakeEmbedder(), FakeRepository([object()]), FakeGenerator())
    with pytest.raises(ValueError, match="must not be empty"):
        service.retrieve("  ", top_k=3, min_relevance=0.0)
