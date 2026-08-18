"""Loads and caches the embedding model used for RAG chunk retrieval."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

import numpy as np

from arxiv_seeker.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def _get_model():
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    logger.info("Loading embedding model %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


class Embedder:
    """Wraps a sentence-transformers model, returns normalized float32 vectors."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or get_settings().embedding_model
        self._model = _get_model() if model_name is None else __import__(
            "sentence_transformers"
        ).SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]
