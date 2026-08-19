import pytest

from local_pdf_rag.chunking import normalize_text, split_text, stable_chunk_id


def test_normalize_text_removes_extraction_noise() -> None:
    assert normalize_text("Hello\x00   world\r\n\r\n\r\nNext") == "Hello world\n\nNext"


def test_split_text_uses_bounded_overlapping_chunks() -> None:
    text = " ".join(f"sentence-{index}." for index in range(100))
    chunks = split_text(text, chunk_size=220, overlap=35)

    assert len(chunks) > 1
    assert all(1 <= len(chunk) <= 220 for chunk in chunks)
    assert chunks == split_text(text, chunk_size=220, overlap=35)


def test_split_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_text("content", chunk_size=200, overlap=200)


def test_chunk_ids_are_stable_and_content_sensitive() -> None:
    first = stable_chunk_id("hash", 1, 0, "text")
    assert first == stable_chunk_id("hash", 1, 0, "text")
    assert first != stable_chunk_id("hash", 1, 0, "changed")
