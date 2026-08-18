"""Example: search arXiv and print re-ranked results."""
from arxiv_seeker.search import SearchOrchestrator

if __name__ == "__main__":
    orchestrator = SearchOrchestrator()
    papers = orchestrator.search("diffusion models for medical imaging", max_results=5)

    for p in papers:
        print(f"[{p.final_score:.3f}] {p.title} ({p.arxiv_id})")
        print(f"  {p.abstract[:150]}...\n")
