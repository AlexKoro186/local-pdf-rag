"""PDF extraction, deterministic chunking, embedding, and indexing."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

from pypdf import PdfReader

from .chunking import normalize_text, split_text, stable_chunk_id
from .config import Settings
from .domain import IngestionReport, SourceChunk


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class PdfIngestor:
    def __init__(self, settings: Settings, embedder, repository) -> None:
        self.settings = settings
        self.embedder = embedder
        self.repository = repository

    def discover(self) -> list[Path]:
        self.settings.pdf_dir.mkdir(parents=True, exist_ok=True)
        return sorted(path for path in self.settings.pdf_dir.glob("*.pdf") if path.is_file())

    def _extract_chunks(self, path: Path, file_hash: str) -> tuple[list[SourceChunk], int]:
        reader = PdfReader(str(path))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise ValueError("password-protected PDFs are not supported")

        chunks: list[SourceChunk] = []
        pages_with_text = 0
        for page_number, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if not text:
                continue
            pages_with_text += 1
            for chunk_index, chunk_text in enumerate(
                split_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
            ):
                chunks.append(
                    SourceChunk(
                        id=stable_chunk_id(file_hash, page_number, chunk_index, chunk_text),
                        text=chunk_text,
                        source=path.name,
                        page=page_number,
                        chunk_index=chunk_index,
                        file_hash=file_hash,
                    )
                )
        return chunks, pages_with_text

    def ingest(self, paths: Iterable[Path] | None = None) -> IngestionReport:
        selected_paths = list(paths) if paths is not None else self.discover()
        report = IngestionReport(files_seen=len(selected_paths))

        for path in selected_paths:
            try:
                path = Path(path)
                if path.suffix.lower() != ".pdf" or not path.is_file():
                    raise ValueError("not a readable PDF file")
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > self.settings.max_file_size_mb:
                    raise ValueError(f"file exceeds {self.settings.max_file_size_mb} MB")

                file_hash = file_sha256(path)
                if self.repository.source_matches_hash(path.name, file_hash):
                    report.files_skipped += 1
                    continue

                chunks, page_count = self._extract_chunks(path, file_hash)
                if not chunks:
                    raise ValueError("no extractable text found; scanned PDFs require OCR")

                embeddings: list[list[float]] = []
                texts = [chunk.text for chunk in chunks]
                for batch in batched(texts, self.settings.embedding_batch_size):
                    embeddings.extend(self.embedder.embed(batch))

                self.repository.replace_source(chunks, embeddings)
                report.files_indexed += 1
                report.pages_indexed += page_count
                report.chunks_indexed += len(chunks)
            except Exception as exc:
                report.errors.append(f"{Path(path).name}: {exc}")

        return report
