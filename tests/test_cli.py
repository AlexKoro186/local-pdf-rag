from pathlib import Path
from types import SimpleNamespace

from local_pdf_rag import cli
from local_pdf_rag.domain import IngestionReport, RetrievedChunk


class FakeRAG:
    def retrieve(self, question, top_k, min_relevance):
        return [RetrievedChunk("id", "evidence", "report.pdf", 3, 0.9)]

    def answer(self, question, chunks):
        return "Grounded answer [1]."


def components(online: bool = True):
    return SimpleNamespace(
        settings=SimpleNamespace(
            top_k=5, min_relevance=0.2, chat_model="chat", embedding_model="embed"
        ),
        ingestor=SimpleNamespace(
            ingest=lambda paths: IngestionReport(files_seen=1, files_indexed=1)
        ),
        rag=FakeRAG(),
        gateway=SimpleNamespace(ping=lambda: online),
        repository=SimpleNamespace(count=lambda: 7),
    )


def test_cli_ingest_and_ask(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "create_components", components)

    assert cli.main(["ingest"]) == 0
    assert "Indexing complete" in capsys.readouterr().out

    assert cli.main(["ask", "What", "changed?"]) == 0
    output = capsys.readouterr().out
    assert "Grounded answer [1]." in output
    assert "report.pdf, p. 3" in output


def test_cli_status(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "create_components", lambda: components(online=False))
    assert cli.main(["status"]) == 0
    output = capsys.readouterr().out
    assert "not reachable" in output
    assert "Indexed chunks: 7" in output


def test_cli_evaluate(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "create_components", components)
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        '{"question":"Question?","expected_source":"report.pdf","expected_page":3}\n',
        encoding="utf-8",
    )

    assert cli.main(["evaluate", str(dataset), "--top-k", "5"]) == 0
    output = capsys.readouterr().out
    assert "Hit@5: 100.0%" in output
    assert "MRR@5: 1.000" in output
