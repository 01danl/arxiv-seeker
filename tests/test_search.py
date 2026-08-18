from unittest.mock import MagicMock

from arxiv_seeker.search import SearchOrchestrator


def test_search_uses_cache_when_available(sample_papers, tmp_cache_db):
    from arxiv_seeker.cache import SearchCache

    cache = SearchCache(db_path=tmp_cache_db, ttl_hours=1)
    cache.set("cached query", 10, "relevance", sample_papers)

    mock_client = MagicMock()
    orchestrator = SearchOrchestrator(client=mock_client, cache=cache)

    results = orchestrator.search("cached query", max_results=10, sort_by="relevance")

    assert len(results) == 2
    mock_client.search.assert_not_called()


def test_search_calls_client_on_cache_miss(sample_papers, tmp_cache_db):
    from arxiv_seeker.cache import SearchCache

    cache = SearchCache(db_path=tmp_cache_db, ttl_hours=1)
    mock_client = MagicMock()
    mock_client.search.return_value = sample_papers

    orchestrator = SearchOrchestrator(client=mock_client, cache=cache)
    results = orchestrator.search("fresh query", max_results=10, rerank=False)

    assert len(results) == 2
    mock_client.search.assert_called_once()
