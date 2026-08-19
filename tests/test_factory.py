from pathlib import Path

from local_pdf_rag.config import Settings
from local_pdf_rag.factory import create_components


def test_factory_wires_shared_dependencies(monkeypatch, tmp_path: Path) -> None:
    calls = {}

    class Gateway:
        def __init__(self, **kwargs):
            calls["gateway"] = kwargs

    class Repository:
        def __init__(self, **kwargs):
            calls["repository"] = kwargs

    class Ingestor:
        def __init__(self, settings, gateway, repository):
            self.dependencies = (settings, gateway, repository)

    class RAG:
        def __init__(self, embedder, repository, generator):
            self.dependencies = (embedder, repository, generator)

    monkeypatch.setattr("local_pdf_rag.factory.OllamaGateway", Gateway)
    monkeypatch.setattr("local_pdf_rag.factory.ChromaRepository", Repository)
    monkeypatch.setattr("local_pdf_rag.factory.PdfIngestor", Ingestor)
    monkeypatch.setattr("local_pdf_rag.factory.RAGService", RAG)
    settings = Settings(
        project_root=tmp_path,
        pdf_dir=tmp_path / "pdfs",
        chroma_path=tmp_path / "chroma",
    )

    result = create_components(settings)

    assert calls["gateway"]["chat_model"] == settings.chat_model
    assert calls["repository"]["path"] == settings.chroma_path
    assert result.ingestor.dependencies == (settings, result.gateway, result.repository)
    assert result.rag.dependencies == (result.gateway, result.repository, result.gateway)
