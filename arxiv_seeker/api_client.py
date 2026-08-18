"""Thin wrapper around the official `arxiv` Python client with rate limiting."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import arxiv

from arxiv_seeker.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Normalized representation of an arXiv paper used across the app."""

    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published: datetime
    updated: datetime
    categories: List[str]
    pdf_url: str
    entry_url: str
    similarity_score: float = 0.0
    citation_count: Optional[int] = None
    final_score: float = 0.0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["published"] = self.published.isoformat()
        d["updated"] = self.updated.isoformat()
        return d


class ArxivClient:
    """Wraps `arxiv.Client` with retry logic and rate limiting."""

    def __init__(self):
        settings = get_settings()
        self._delay = settings.arxiv_rate_limit_seconds
        self._last_call = 0.0
        self._client = arxiv.Client(page_size=100, delay_seconds=self._delay, num_retries=3)

    def _throttle(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_call = time.monotonic()

    def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
    ) -> List[Paper]:
        """Search arXiv and return normalized Paper objects.

        `query` supports arXiv's native syntax, e.g. `cat:cs.LG AND all:attention`.
        `sort_by` is one of: relevance, submitted, lastUpdated.
        """
        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "submitted": arxiv.SortCriterion.SubmittedDate,
            "lastUpdated": arxiv.SortCriterion.LastUpdatedDate,
        }
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_map.get(sort_by, arxiv.SortCriterion.Relevance),
        )

        papers: List[Paper] = []
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                self._throttle()
                for result in self._client.results(search):
                    papers.append(
                        Paper(
                            arxiv_id=result.get_short_id(),
                            title=result.title.strip().replace("\n", " "),
                            abstract=result.summary.strip().replace("\n", " "),
                            authors=[a.name for a in result.authors],
                            published=result.published,
                            updated=result.updated,
                            categories=result.categories,
                            pdf_url=result.pdf_url,
                            entry_url=result.entry_id,
                        )
                    )
                return papers
            except Exception as exc:  # network hiccups, arxiv API errors
                last_error = exc
                logger.warning("arXiv search attempt %d failed: %s", attempt + 1, exc)
                time.sleep(self._delay * (attempt + 1))

        raise RuntimeError(f"arXiv search failed after retries: {last_error}") from last_error

    def get_by_id(self, arxiv_id: str) -> Paper:
        results = self.search(f"id:{arxiv_id}", max_results=1)
        if not results:
            raise ValueError(f"Paper {arxiv_id} not found on arXiv")
        return results[0]

    def download_pdf(self, arxiv_id: str, dest_dir: str) -> str:
        """Download a paper's PDF and return the local file path."""
        self._throttle()
        import os
        import requests
        os.makedirs(dest_dir, exist_ok=True)
        filename = os.path.join(dest_dir, f"{arxiv_id}.pdf")
        paper = self.get_by_id(arxiv_id)
        pdf_url = paper.pdf_url

        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return filename