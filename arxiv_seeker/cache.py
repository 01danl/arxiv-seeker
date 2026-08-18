"""SQLite-backed cache for arXiv search results with TTL."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from typing import List, Optional

from arxiv_seeker.api_client import Paper
from arxiv_seeker.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_cache (
    query_hash TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    results_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class SearchCache:
    def __init__(self, db_path: Optional[str] = None, ttl_hours: Optional[int] = None):
        settings = get_settings()
        self.db_path = str(db_path or settings.cache_db_path)
        self.ttl_seconds = (ttl_hours if ttl_hours is not None else settings.cache_ttl_hours) * 3600
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @staticmethod
    def _key(query: str, max_results: int, sort_by: str) -> str:
        raw = f"{query}|{max_results}|{sort_by}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, query: str, max_results: int, sort_by: str) -> Optional[List[Paper]]:
        key = self._key(query, max_results, sort_by)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT results_json, created_at FROM search_cache WHERE query_hash = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        results_json, created_at = row
        if time.time() - created_at > self.ttl_seconds:
            return None  # stale
        raw_list = json.loads(results_json)
        return [self._paper_from_dict(d) for d in raw_list]

    def set(self, query: str, max_results: int, sort_by: str, papers: List[Paper]) -> None:
        key = self._key(query, max_results, sort_by)
        payload = json.dumps([p.to_dict() for p in papers])
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO search_cache (query_hash, query, results_json, created_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(query_hash) DO UPDATE SET
                       results_json = excluded.results_json,
                       created_at = excluded.created_at""",
                (key, query, payload, time.time()),
            )

    def invalidate(self, query: str, max_results: int, sort_by: str) -> None:
        key = self._key(query, max_results, sort_by)
        with self._connect() as conn:
            conn.execute("DELETE FROM search_cache WHERE query_hash = ?", (key,))

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM search_cache")

    @staticmethod
    def _paper_from_dict(d: dict) -> Paper:
        from datetime import datetime

        d = d.copy()
        d["published"] = datetime.fromisoformat(d["published"])
        d["updated"] = datetime.fromisoformat(d["updated"])
        return Paper(**d)
