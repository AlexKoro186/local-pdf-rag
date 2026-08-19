from pathlib import Path

from local_pdf_rag.config import Settings
from local_pdf_rag.domain import SourceChunk
from local_pdf_rag.ingestion import PdfIngestor, batched, file_sha256


class FakeEmbedder:
    def embed(self, texts):
        return [[float(len(text))] for text in texts]


class FakeRepository:
    def __init__(self, unchanged: bool = False) -> None:
        self.unchanged = unchanged
        self.replacements = []

    def source_matches_hash(self, source, file_hash):
        return self.unchanged

    def replace_source(self, chunks, embeddings):
        self.replacements.append((chunks, embeddings))


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        pdf_dir=tmp_path / "pdfs",
        chroma_path=tmp_path / "chroma",
        embedding_batch_size=1,
    )


def test_batched_and_sha256_are_deterministic(tmp_path: Path) -> None:
    assert list(batched(["a", "b", "c"], 2)) == [["a", "b"], ["c"]]
    path = tmp_path / "document.pdf"
    path.write_bytes(b"same bytes")
    assert file_sha256(path) == file_sha256(path)


def test_ingest_embeds_and_replaces_changed_document(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "document.pdf"
    path.write_bytes(b"%PDF test")
    repository = FakeRepository()
    ingestor = PdfIngestor(make_settings(tmp_path), FakeEmbedder(), repository)
    chunks = [SourceChunk("id", "content", path.name, 1, 0, "hash")]
    monkeypatch.setattr(ingestor, "_extract_chunks", lambda *_: (chunks, 1))

    report = ingestor.ingest([path])

    assert report.succeeded
    assert report.files_indexed == 1
    assert report.pages_indexed == 1
    assert report.chunks_indexed == 1
    assert repository.replacements[0][1] == [[7.0]]


def test_ingest_skips_unchanged_document(tmp_path: Path) -> None:
    path = tmp_path / "document.pdf"
    path.write_bytes(b"%PDF test")
    repository = FakeRepository(unchanged=True)
    ingestor = PdfIngestor(make_settings(tmp_path), FakeEmbedder(), repository)

    report = ingestor.ingest([path])

    assert report.files_skipped == 1
    assert repository.replacements == []


def test_ingest_reports_invalid_input_without_aborting(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not a pdf")
    ingestor = PdfIngestor(make_settings(tmp_path), FakeEmbedder(), FakeRepository())

    report = ingestor.ingest([path])

    assert not report.succeeded
    assert "not a readable PDF" in report.errors[0]
