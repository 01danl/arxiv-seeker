"""Extracts text (and light structure) from downloaded arXiv PDFs."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pymupdf  # PyMuPDF, imported as `fitz` traditionally but pymupdf works too

logger = logging.getLogger(__name__)

# Common top-level section headers in scientific papers, used to segment text
# so downstream chunking can respect paper structure instead of cutting blindly.
_SECTION_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\s+)?"
    r"(Abstract|Introduction|Related Work|Background|Method(?:ology|s)?|"
    r"Experiments?|Results?|Discussion|Limitations?|Conclusion(?:s)?|"
    r"Acknowledg(?:e)?ments?|References|Appendix)\s*$",
    re.IGNORECASE,
)


@dataclass
class Section:
    heading: str
    text: str
    page_start: int
    page_end: int


@dataclass
class ParsedPaper:
    arxiv_id: str
    full_text: str
    sections: List[Section]
    num_pages: int


def extract_text(pdf_path: str) -> str:
    """Raw text extraction, page-joined."""
    doc = pymupdf.open(pdf_path)
    try:
        return "\n\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def parse_pdf(pdf_path: str, arxiv_id: str) -> ParsedPaper:
    """Extract text and split into coarse sections based on heading heuristics.

    This is intentionally heuristic (no layout model) — good enough to give the
    chunker section boundaries to respect, without pulling in heavy dependencies.
    """
    doc = pymupdf.open(pdf_path)
    try:
        num_pages = doc.page_count
        lines_with_page: List[tuple[str, int]] = []
        for page_idx, page in enumerate(doc):
            for line in page.get_text("text").split("\n"):
                lines_with_page.append((line, page_idx))

        sections: List[Section] = []
        current_heading = "Preamble"
        current_lines: List[str] = []
        current_page_start = 0

        def flush(end_page: int):
            if current_lines:
                sections.append(
                    Section(
                        heading=current_heading,
                        text="\n".join(current_lines).strip(),
                        page_start=current_page_start,
                        page_end=end_page,
                    )
                )

        for line, page_idx in lines_with_page:
            stripped = line.strip()
            match = _SECTION_PATTERN.match(stripped)
            if match and len(stripped) < 60:
                flush(page_idx)
                current_heading = match.group(1).title()
                current_lines = []
                current_page_start = page_idx
            else:
                current_lines.append(line)
        flush(num_pages - 1)

        full_text = "\n\n".join(s.text for s in sections)
        return ParsedPaper(arxiv_id=arxiv_id, full_text=full_text, sections=sections, num_pages=num_pages)
    finally:
        doc.close()
