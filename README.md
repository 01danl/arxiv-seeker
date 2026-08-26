# 🔎 ArxivSeeker

**AI-powered research assistant for arXiv** — semantic search, smart ranking, and RAG chat with any paper.

```
arxiv-seeker search "diffusion models medical imaging"
arxiv-seeker rag index --paper-id 1706.03762
arxiv-seeker chat --paper-id 1706.03762 "What is the main contribution?"
```

## Features

- **Semantic search** over arXiv with natural-language queries or native syntax (`cat:cs.LG`)
- **Re-ranking** by semantic similarity + recency decay + optional citation boost (SBERT-based)
- **SQLite cache** (24h TTL by default) to avoid hammering the arXiv API
- **RAG chat** with any paper: download → parse → section-aware chunk → embed → FAISS → LLM answer with cited sources
- **Local-first LLM** via Ollama (no API key needed), or plug in OpenAI/Anthropic
- **Three interfaces**: CLI (`click`), Web UI (`streamlit`), optional REST API (`fastapi`)

## Install

```bash
git clone https://github.com/01danl/arxiv-seeker.git
cd arxiv-seeker
python -m venv venv && source venv/bin/activate
pip install -e ".[dev,rag,web]"
cp .env.example .env
```

By default the RAG chat uses **Ollama** running locally (free, no API key):

```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.1
ollama serve
```

To use OpenAI or Anthropic instead, set `LLM_PROVIDER=openai` (or `anthropic`) and the matching
`*_API_KEY` in `.env`.

## Usage

### CLI

```bash
# Search
arxiv-seeker search "attention is all you need" --max 10 --sort relevance

# Index a paper for RAG (downloads PDF, parses, chunks, embeds, builds FAISS index)
arxiv-seeker rag index --paper-id 1706.03762

# Chat with it
arxiv-seeker chat --paper-id 1706.03762 "What are the main limitations?"
```

### Web UI

```bash
streamlit run arxiv_seeker/web/app.py
```

Search on the left, click "Chat with this paper" on any result, ask questions on the right.
First question triggers indexing automatically if the paper isn't indexed yet.

### Python API

```python
from arxiv_seeker.search import SearchOrchestrator
from arxiv_seeker.rag.chat import PaperIndexer, RagChat

papers = SearchOrchestrator().search("large language model reasoning", max_results=5)

PaperIndexer().index(papers[0].arxiv_id)
answer = RagChat(papers[0].arxiv_id).ask("What datasets were used?")
print(answer.answer, answer.sources)
```

See `examples/` for more.

## Architecture

```
Presentation:  CLI (click) · Web UI (Streamlit) · REST API (FastAPI, optional)
Application:   Search Orchestrator · RAG Engine (chunk/embed/vector store) · SQLite cache
Integration:   arXiv API · PDF parser (PyMuPDF) · Semantic Scholar (optional citations)
```

**RAG pipeline:** PDF → PyMuPDF text extraction with heading-based section detection →
section-aware chunking (falls back to a sliding window with overlap only when a section
exceeds the chunk size) → `BAAI/bge-small-en-v1.5` embeddings → per-paper FAISS `IndexFlatIP`
(cosine similarity via normalized vectors) → top-k retrieval → LLM answer, grounded in the
retrieved excerpts and refusing to answer outside them.

Indexes persist under `data/faiss/<arxiv_id>/`, so re-visiting a paper skips re-indexing.

## Configuration

All settings live in `arxiv_seeker/config.py` (Pydantic Settings, `.env`-driven). Key knobs:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `openai` \| `anthropic` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | RAG chunk embeddings |
| `RERANK_MODEL` | `all-MiniLM-L6-v2` | Search result re-ranking |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `512` / `50` | Chunking |
| `TOP_K_CHUNKS` | `4` | Chunks retrieved per question |
| `CACHE_TTL_HOURS` | `24` | Search cache lifetime |

## Testing

```bash
pytest --cov=arxiv_seeker
```

Tests cover the cache, the chunker (short/long sections, overlap, uniqueness), and search
orchestration (cache hit/miss) with mocked I/O — no network or GPU required to run them.

## Known limitations / next steps

- **arXiv API access**: the official API enforces a 3s rate limit; heavy usage should
  register for a higher tier if available.
- **Re-ranking quality**: `all-MiniLM-L6-v2` is a general-purpose model. For better results on
  scientific text, consider swapping in `allenai/specter2` (see `RERANK_MODEL` in `.env`).
- **Retrieval precision**: a cross-encoder reranking pass on top of the FAISS top-k (currently
  bi-encoder only) would improve answer grounding for multi-part questions — see roadmap.
- **Section detection** is heuristic (regex on common headings), not a layout model — works
  well on typical arXiv LaTeX-generated PDFs, less reliably on unusual templates.

## Roadmap

- Personalized weekly digest emails
- Citation graph visualization (NetworkX)
- Batch chat / compare multiple papers side-by-side
- Zotero/Mendeley export
- Multi-language abstract translation

## License

Apache-2.0
