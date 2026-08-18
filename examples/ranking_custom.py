"""Example: compare ranked vs. unranked (raw arXiv order) results side by side."""
from arxiv_seeker.search import SearchOrchestrator

if __name__ == "__main__":
    orchestrator = SearchOrchestrator()
    query = "large language model reasoning"

    raw = orchestrator.search(query, max_results=8, rerank=False, use_cache=False)
    ranked = orchestrator.search(query, max_results=8, rerank=True, use_cache=False)

    print("=== Raw arXiv relevance order ===")
    for p in raw:
        print(f"- {p.title}")

    print("\n=== Semantic re-ranked order ===")
    for p in ranked:
        print(f"[{p.final_score:.3f}] {p.title}")
