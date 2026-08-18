"""FAISS-backed vector store, one index per paper, persisted to disk.

Each paper gets its own `<faiss_dir>/<arxiv_id>/index.faiss` +
`<faiss_dir>/<arxiv_id>/meta.json` (chunk texts + headings, since FAISS only
stores vectors). This keeps indexing/search scoped per-paper, which matches
the product's "chat with this paper" flow, while still being trivial to
extend to a merged multi-paper index later (batch chat roadmap item).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from arxiv_seeker.config import get_settings
from arxiv_seeker.rag.chunker import Chunk


class VectorStore:
    def __init__(self, arxiv_id: str, dimension: int, faiss_dir: str | None = None):
        self.arxiv_id = arxiv_id
        self.dimension = dimension
        base = Path(faiss_dir or get_settings().faiss_dir)
        self.paper_dir = base / arxiv_id.replace("/", "_")
        self.index_path = self.paper_dir / "index.faiss"
        self.meta_path = self.paper_dir / "meta.json"
        self._index = None
        self._meta: List[dict] = []

    def exists(self) -> bool:
        return self.index_path.exists() and self.meta_path.exists()

    def build(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Build (or overwrite) the index for this paper from chunks + their embeddings."""
        assert embeddings.shape[0] == len(chunks), "chunk/embedding count mismatch"
        self.paper_dir.mkdir(parents=True, exist_ok=True)

        index = faiss.IndexFlatIP(self.dimension)  # inner product on normalized vecs = cosine
        index.add(embeddings)

        faiss.write_index(index, str(self.index_path))
        meta = [
            {"chunk_id": c.chunk_id, "heading": c.heading, "text": c.text, "chunk_index": c.chunk_index}
            for c in chunks
        ]
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        self._index = index
        self._meta = meta

    def load(self) -> None:
        if not self.exists():
            raise FileNotFoundError(f"No index found for paper {self.arxiv_id} at {self.paper_dir}")
        self._index = faiss.read_index(str(self.index_path))
        self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Tuple[dict, float]]:
        if self._index is None:
            self.load()
        query = np.asarray([query_embedding], dtype="float32")
        scores, indices = self._index.search(query, min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._meta[idx], float(score)))
        return results
