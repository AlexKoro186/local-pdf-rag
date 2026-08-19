import json

import pytest

from local_pdf_rag.domain import RetrievedChunk
from local_pdf_rag.evaluation import evaluate, load_cases


class FakeRAG:
    def retrieve(self, question, top_k, min_relevance):
        assert min_relevance == 0.0
        return [
            RetrievedChunk("1", "text", "other.pdf", 1, 0.9),
            RetrievedChunk("2", "text", "target.pdf", 3, 0.8),
        ][:top_k]


def test_load_and_evaluate_cases(tmp_path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "question": "Where is the answer?",
                "expected_source": "target.pdf",
                "expected_page": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate(FakeRAG(), load_cases(dataset), top_k=5)

    assert result.hit_rate == 1.0
    assert result.mean_reciprocal_rank == 0.5


def test_empty_evaluation_dataset_is_rejected(tmp_path) -> None:
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_cases(dataset)
