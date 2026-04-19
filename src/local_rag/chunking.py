from __future__ import annotations

import hashlib
import re

from local_rag.models import DocumentChunk, LoadedDocument


BOUNDARY_CHARS = ("\n", "。", "！", "？", ".", "!", "?", ";", "；")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    normalized = normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        ideal_end = min(start + chunk_size, text_length)
        end = ideal_end
        if ideal_end < text_length:
            boundary = find_boundary(normalized, start, ideal_end)
            minimum_chunk = max(chunk_size // 3, 1)
            if boundary != -1 and (boundary + 1 - start) >= minimum_chunk:
                end = boundary + 1

        chunk = normalized[start:end].strip()
        if chunk and (not chunks or chunk != chunks[-1]):
            chunks.append(chunk)

        if end >= text_length:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def chunk_document(
    document: LoadedDocument,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    chunks = chunk_text(
        document.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    output: list[DocumentChunk] = []

    for index, content in enumerate(chunks):
        raw_key = f"{document.source_path}:{index}".encode("utf-8")
        chunk_id = hashlib.sha1(raw_key).hexdigest()[:12]
        output.append(
            DocumentChunk(
                chunk_id=chunk_id,
                source_path=document.source_path,
                content=content,
                index=index,
                char_count=len(content),
                metadata={"source_path": document.source_path},
            )
        )

    return output


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def find_boundary(text: str, start: int, ideal_end: int) -> int:
    search_start = max(start + (ideal_end - start) // 2, start)
    boundary = -1
    for marker in BOUNDARY_CHARS:
        boundary = max(boundary, text.rfind(marker, search_start, ideal_end))
    return boundary
