from pathlib import Path

import pytest

from local_pdf_rag.config import Settings


def test_settings_resolve_paths_from_project_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAG_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("RAG_PDF_DIR", "documents")
    monkeypatch.setenv("RAG_CHROMA_PATH", "vectors")
    monkeypatch.setenv("RAG_TOP_K", "7")

    settings = Settings.from_env()

    assert settings.pdf_dir == tmp_path / "documents"
    assert settings.chroma_path == tmp_path / "vectors"
    assert settings.top_k == 7


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(100, 10), (500, -1), (500, 500)],
)
def test_settings_validate_chunk_configuration(
    tmp_path: Path, chunk_size: int, overlap: int
) -> None:
    with pytest.raises(ValueError):
        Settings(
            project_root=tmp_path,
            pdf_dir=tmp_path / "pdfs",
            chroma_path=tmp_path / "chroma",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
