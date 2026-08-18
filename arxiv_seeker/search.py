"""Search orchestration: cache lookup, arXiv query, semantic re-ranking."""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from functools import lru_cache
from typing import List, Optional

import numpy as np

from arxiv_seeker.api_client import ArxivClient, Paper
from arxiv_seeker.cache import SearchCache
from arxiv_seeker.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def _get_rerank_model():
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    logger.info("Loading rerank model %s", settings.rerank_model)
    return SentenceTransformer(settings.rerank_model)


class SearchOrchestrator:
    def __init__(self, client: Optional[ArxivClient] = None, cache: Optional[SearchCache] = None):
        self.client = client or ArxivClient()
        self.cache = cache or SearchCache()
        self.settings = get_settings()

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        sort_by: str = "relevance",
        use_cache: bool = True,
        rerank: bool = True,
    ) -> List[Paper]:
        max_results = max_results or self.settings.default_max_results

        if use_cache:
            cached = self.cache.get(query, max_results, sort_by)
            if cached is not None:
                logger.info("Cache hit for query=%r", query)
                return cached

        papers = self.client.search(query, max_results=max_results, sort_by=sort_by)

        if rerank and papers:
            papers = self._rerank(query, papers)
        else:
            for p in papers:
                p.final_score = p.similarity_score

        if use_cache:
            self.cache.set(query, max_results, sort_by, papers)

        return papers

    def _rerank(self, query: str, papers: List[Paper]) -> List[Paper]:
        """Re-rank by semantic similarity (title+abstract) with a recency decay bonus."""
        model = _get_rerank_model()
        texts = [f"{p.title}. {p.abstract}" for p in papers]

        query_emb = model.encode([query], normalize_embeddings=True)[0]
        doc_embs = model.encode(texts, normalize_embeddings=True)

        sims = doc_embs @ query_emb  # cosine similarity since normalized

        now = datetime.now(timezone.utc)
        halflife = self.settings.recency_halflife_days

        for paper, sim in zip(papers, sims):
            paper.similarity_score = float(sim)
            age_days = max((now - paper.published.astimezone(timezone.utc)).days, 0)
            recency_factor = math.exp(-math.log(2) * age_days / halflife) if halflife > 0 else 1.0
            # citation boost is a light multiplicative nudge if available, else neutral
            citation_boost = 1.0
            if paper.citation_count:
                citation_boost = 1.0 + min(math.log1p(paper.citation_count) / 20, 0.25)
            paper.final_score = float(sim * (0.85 + 0.15 * recency_factor) * citation_boost)

        papers = [p for p in papers if p.similarity_score >= self.settings.similarity_threshold] or papers
        papers.sort(key=lambda p: p.final_score, reverse=True)
        return papers
