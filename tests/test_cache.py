from arxiv_seeker.cache import SearchCache


def test_cache_miss_returns_none(tmp_cache_db):
    cache = SearchCache(db_path=tmp_cache_db, ttl_hours=1)
    assert cache.get("nonexistent query", 10, "relevance") is None


def test_cache_set_and_get(tmp_cache_db, sample_papers):
    cache = SearchCache(db_path=tmp_cache_db, ttl_hours=1)
    cache.set("test query", 10, "relevance", sample_papers)

    result = cache.get("test query", 10, "relevance")
    assert result is not None
    assert len(result) == 2
    assert result[0].arxiv_id == sample_papers[0].arxiv_id
    assert result[0].title == sample_papers[0].title


def test_cache_expires(tmp_cache_db, sample_papers):
    cache = SearchCache(db_path=tmp_cache_db, ttl_hours=0)  # instantly stale
    cache.set("expiring query", 10, "relevance", sample_papers)
    import time

    time.sleep(0.01)
    assert cache.get("expiring query", 10, "relevance") is None


def test_cache_invalidate(tmp_cache_db, sample_papers):
    cache = SearchCache(db_path=tmp_cache_db, ttl_hours=1)
    cache.set("q", 10, "relevance", sample_papers)
    cache.invalidate("q", 10, "relevance")
    assert cache.get("q", 10, "relevance") is None


def test_cache_different_params_different_keys(tmp_cache_db, sample_papers):
    cache = SearchCache(db_path=tmp_cache_db, ttl_hours=1)
    cache.set("q", 10, "relevance", sample_papers)
    assert cache.get("q", 20, "relevance") is None
    assert cache.get("q", 10, "submitted") is None
