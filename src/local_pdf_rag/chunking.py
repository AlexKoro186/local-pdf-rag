"""Deterministic text normalization and overlapping chunk generation."""

from __future__ import annotations

import hashlib
import re


def normalize_text(text: str) -> str:
    """Remove extraction noise while preserving paragraph boundaries."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text near natural boundaries with deterministic character overlap."""
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    minimum_chunk = chunk_size // 2

    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end

        if hard_end < len(text):
            search_from = start + minimum_chunk
            candidates = [
                text.rfind(separator, search_from, hard_end)
                for separator in ("\n\n", "\n", ". ", "! ", "? ", "; ", " ")
            ]
            boundary = max(candidates)
            if boundary >= search_from:
                end = boundary + (1 if text[boundary] in ".!?;" else 0)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(end - overlap, start + 1)
        while next_start < end and text[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks


def stable_chunk_id(file_hash: str, page: int, chunk_index: int, text: str) -> str:
    payload = f"{file_hash}:{page}:{chunk_index}:{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]
