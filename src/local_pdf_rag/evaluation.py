"""Small retrieval evaluation harness for repeatable RAG quality checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    question: str
    expected_source: str
    expected_page: int | None = None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    cases: int
    hits: int
    reciprocal_rank_sum: float

    @property
    def hit_rate(self) -> float:
        return self.hits / self.cases if self.cases else 0.0

    @property
    def mean_reciprocal_rank(self) -> float:
        return self.reciprocal_rank_sum / self.cases if self.cases else 0.0


def load_cases(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                cases.append(
                    EvaluationCase(
                        question=str(item["question"]),
                        expected_source=str(item["expected_source"]),
                        expected_page=(
                            int(item["expected_page"])
                            if item.get("expected_page") is not None
                            else None
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid evaluation case on line {line_number}: {exc}") from exc
    if not cases:
        raise ValueError("Evaluation dataset is empty.")
    return cases


def evaluate(rag, cases: list[EvaluationCase], top_k: int) -> EvaluationResult:
    hits = 0
    reciprocal_rank_sum = 0.0
    for case in cases:
        chunks = rag.retrieve(case.question, top_k=top_k, min_relevance=0.0)
        for rank, chunk in enumerate(chunks, start=1):
            source_matches = chunk.source == case.expected_source
            page_matches = case.expected_page is None or chunk.page == case.expected_page
            if source_matches and page_matches:
                hits += 1
                reciprocal_rank_sum += 1.0 / rank
                break
    return EvaluationResult(len(cases), hits, reciprocal_rank_sum)
