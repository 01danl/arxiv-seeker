"""Example: index a paper for RAG, then ask it questions.

Requires a running LLM backend (Ollama by default — see README).
"""
from arxiv_seeker.rag.chat import PaperIndexer, RagChat

PAPER_ID = "1706.03762"  # "Attention Is All You Need"

if __name__ == "__main__":
    print(f"Indexing {PAPER_ID}...")
    PaperIndexer().index(PAPER_ID)

    session = RagChat(PAPER_ID)
    for question in [
        "What is the main contribution of this paper?",
        "What are the main limitations mentioned?",
    ]:
        result = session.ask(question)
        print(f"\nQ: {question}\nA: {result.answer}")
        print("Sources:", [s.heading for s in result.sources])
