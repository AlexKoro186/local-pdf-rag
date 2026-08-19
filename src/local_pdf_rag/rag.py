"""Retrieval-augmented answer orchestration."""

from __future__ import annotations

from collections.abc import Iterator

from .domain import RetrievedChunk
from .prompting import SYSTEM_PROMPT, build_user_prompt


class RAGService:
    def __init__(self, embedder, repository, generator) -> None:
        self.embedder = embedder
        self.repository = repository
        self.generator = generator

    def retrieve(self, question: str, top_k: int, min_relevance: float) -> list[RetrievedChunk]:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        if self.repository.count() == 0:
            return []

        query_embedding = self.embedder.embed([question])[0]
        candidates = self.repository.search(query_embedding, top_k)

        results: list[RetrievedChunk] = []
        seen: set[str] = set()
        for chunk in candidates:
            if chunk.relevance >= min_relevance and chunk.id not in seen:
                results.append(chunk)
                seen.add(chunk.id)
        return results

    def stream_answer(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        if not chunks:
            yield "I could not find sufficiently relevant information in the indexed documents."
            return
        prompt = build_user_prompt(question, chunks)
        yield from self.generator.stream_chat(SYSTEM_PROMPT, prompt)

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        return "".join(self.stream_answer(question, chunks))
