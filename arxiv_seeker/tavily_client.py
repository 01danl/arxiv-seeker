"""Discovery via Tavily web search, restricted to arxiv.org."""
from __future__ import annotations

import logging
import re
from typing import List

import requests

from arxiv_seeker.config import get_settings

logger = logging.getLogger(__name__)

_ARXIV_ID_PATTERN = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)")


class TavilyClient:
    def __init__(self, api_key: str | None = None):
        settings = get_settings()
        self.api_key = api_key or settings.tavily_api_key
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is not set")

    def search_arxiv_ids(self, query: str, max_results: int = 10) -> List[str]:
        """Search the web (restricted to arxiv.org) and extract arXiv IDs, preserving relevance order."""
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "search_depth": "advanced",
                "include_domains": ["arxiv.org"],
                "max_results": max_results,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        ids: List[str] = []
        seen = set()
        for result in data.get("results", []):
            url = result.get("url", "")
            match = _ARXIV_ID_PATTERN.search(url)
            if match:
                arxiv_id = match.group(1).split("v")[0]  # strip version suffix
                if arxiv_id not in seen:
                    seen.add(arxiv_id)
                    ids.append(arxiv_id)
        logger.info("Tavily found %d arXiv IDs for query=%r", len(ids), query)
        return ids