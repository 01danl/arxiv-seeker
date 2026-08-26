"""Discovery via Tavily web search, restricted to arxiv.org."""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import requests

from arxiv_seeker.config import get_settings

logger = logging.getLogger(__name__)

_ARXIV_ID_PATTERN = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)")

# ---------------------------------------------------------------------------
# Per-domain community sources for general-web discovery.
# When the inferred domain doesn't match any key we fall back to a broad
# set of general-purpose discussion sites.
# ---------------------------------------------------------------------------
_DOMAIN_COMMUNITY_SOURCES: Dict[str, List[str]] = {
    "AI/ML engineering": [
        "reddit.com", "news.ycombinator.com", "github.com",
        "towardsdatascience.com", "sebastianraschka.com",
        "lilianweng.github.io", "karpathy.github.io", "arxiv.org",
    ],
    "biology": [
        "reddit.com", "news.ycombinator.com", "github.com",
        "nature.com", "cell.com", "biologists.com",
        "pubmed.ncbi.nlm.nih.gov", "biorxiv.org",
    ],
    "chemistry": [
        "reddit.com", "news.ycombinator.com", "github.com",
        "nature.com", "pubs.acs.org", "chemistryworld.com",
        "pubs.rsc.org",
    ],
    "physics": [
        "reddit.com", "news.ycombinator.com", "github.com",
        "physicsworld.com", "nature.com", "aps.org",
    ],
    "mathematics": [
        "reddit.com", "news.ycombinator.com", "github.com",
        "mathoverflow.net", "math.stackexchange.com",
    ],
    "economics": [
        "reddit.com", "news.ycombinator.com", "github.com",
        "economist.com", "nber.org", "aeaweb.org",
    ],
    "computer science": [
        "reddit.com", "news.ycombinator.com", "github.com",
        "acm.org", "ieee.org",
    ],
    "medicine": [
        "reddit.com", "news.ycombinator.com",
        "pubmed.ncbi.nlm.nih.gov", "nejm.org", "thelancet.com",
        "bmj.com",
    ],
    "astrophysics": [
        "reddit.com", "news.ycombinator.com",
        "aas.org", "nature.com", "arxiv.org",
    ],
    "neuroscience": [
        "reddit.com", "news.ycombinator.com",
        "nature.com", "jneurosci.org", "pubmed.ncbi.nlm.nih.gov",
    ],
    "quantum computing": [
        "reddit.com", "news.ycombinator.com", "github.com",
        "quantum-journal.org", "arxiv.org",
    ],
    "computational biology": [
        "reddit.com", "news.ycombinator.com", "github.com",
        "nature.com", "pubmed.ncbi.nlm.nih.gov", "biorxiv.org",
    ],
}

# Broad fallback for any domain we don't have a curated list for.
_FALLBACK_COMMUNITY_SOURCES: List[str] = [
    "reddit.com", "news.ycombinator.com", "github.com",
    "stackexchange.com", "wikipedia.org",
]


def _pick_community_sources(domain: str) -> List[str]:
    """Return the best-matching community-domain list for *domain*."""
    domain_lower = domain.lower()
    for key, sources in _DOMAIN_COMMUNITY_SOURCES.items():
        if key.lower() in domain_lower or domain_lower in key.lower():
            return sources
    return _FALLBACK_COMMUNITY_SOURCES


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

    def search_general(
        self,
        query: str,
        max_results: int = 8,
        domain: Optional[str] = None,
    ) -> List[dict]:
        """General web search for community recommendations.

        If *domain* is provided we scope the search to relevant community
        sites for that field; otherwise a broad fallback set is used.
        """
        include_domains = _pick_community_sources(domain) if domain else _FALLBACK_COMMUNITY_SOURCES

        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "include_domains": include_domains,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],
            }
            for r in data.get("results", [])
        ]