"""Chunking strategies for RAG indexing.

Default strategy is section-aware: it never splits mid-section unless the
section itself exceeds `chunk_size_tokens`, and only then falls back to a
sliding window with overlap. This keeps retrieval units semantically coherent
instead of naive fixed-size slicing across section boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from arxiv_seeker.config import get_settings
from arxiv_seeker.rag.pdf_parser import ParsedPaper, Section

# Rough token estimate without pulling in a tokenizer dependency for this step;
# ~4 chars/token is a standard approximation for English scientific text.
_CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    chunk_id: str
    arxiv_id: str
    heading: str
    text: str
    chunk_index: int


def _tokens_to_chars(tokens: int) -> int:
    return tokens * _CHARS_PER_TOKEN


def _split_long_section(text: str, size_tokens: int, overlap_tokens: int) -> List[str]:
    size_chars = _tokens_to_chars(size_tokens)
    overlap_chars = _tokens_to_chars(overlap_tokens)
    step = max(size_chars - overlap_chars, 1)

    pieces = []
    start = 0
    while start < len(text):
        end = min(start + size_chars, len(text))
        pieces.append(text[start:end])
        if end == len(text):
            break
        start += step
    return pieces


def chunk_paper(parsed: ParsedPaper, size_tokens: int | None = None, overlap_tokens: int | None = None) -> List[Chunk]:
    settings = get_settings()
    size_tokens = size_tokens or settings.chunk_size_tokens
    overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens
    max_chars = _tokens_to_chars(size_tokens)

    chunks: List[Chunk] = []
    idx = 0
    for section in parsed.sections:
        text = section.text.strip()
        if not text:
            continue
        if len(text) <= max_chars:
            chunks.append(
                Chunk(
                    chunk_id=f"{parsed.arxiv_id}::{idx}",
                    arxiv_id=parsed.arxiv_id,
                    heading=section.heading,
                    text=text,
                    chunk_index=idx,
                )
            )
            idx += 1
        else:
            for piece in _split_long_section(text, size_tokens, overlap_tokens):
                chunks.append(
                    Chunk(
                        chunk_id=f"{parsed.arxiv_id}::{idx}",
                        arxiv_id=parsed.arxiv_id,
                        heading=section.heading,
                        text=piece,
                        chunk_index=idx,
                    )
                )
                idx += 1
    return chunks
