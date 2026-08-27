<div align="center">
  <img src="https://img.shields.io/badge/version-0.3.0-7c7cf9?style=flat-square" alt="version" />
  <img src="https://img.shields.io/badge/python-≥3.9-7c7cf9?style=flat-square" alt="python" />
  <img src="https://img.shields.io/badge/license-Apache--2.0-7c7cf9?style=flat-square" alt="license" />
  <br />
  <br />
  <h1>🔎 ArxivSeeker</h1>
  <p>
    <strong>AI-powered research assistant for arXiv.</strong><br />
    Find papers with natural language, let an LLM judge relevance,<br />
    then chat with any paper — answers grounded in the actual text.
  </p>
</div>

---

## What is it?

ArxivSeeker is a **two-layer** system:

1. **Agentic search** — you describe what you want to learn (in any language, even
   vaguely), an LLM understands your intent, searches arXiv via
   [Tavily](https://tavily.com), and a second LLM call filters out irrelevant,
   policy, or low-quality papers. No more page 10 of confusing arXiv results.

2. **RAG chat** — pick any paper, index it once (download → parse → chunk → embed),
   then ask questions. Answers cite the exact sections of the paper they came from,
   and the model refuses to guess when the text doesn't cover your question

```
┌───────────────────────────────────────────────────────┐
│  You: "I want to learn gradient descent. What papers? │
│                   should I read?"                      │
└───────────────────────┬───────────────────────────────┘
                        │
           ┌────────────▼────────────┐
           │  IntentAgent (LLM)       │
           │  "gradient descent        │
           │   optimization, SGD       │
           │   convergence, Adam..."  │
           └────────────┬────────────┘
                        │
           ┌────────────▼────────────┐
           │  Tavily web search       │
           │  + arXiv API fetch       │
           └────────────┬────────────┘
                        │
           ┌────────────▼────────────┐
           │  RelevanceJudge (LLM)    │
           │  Keep: 4 papers about    │
           │  gradient descent.       │
           │  Reject: policy paper    │
           │  that mentions "gradient"│
           └────────────┬────────────┘
                        │
           ┌────────────▼────────────┐
           │  📄 Adam: A Method for   │
           │  Stochastic Optimization │
           │  📄 Exact Convergence     │
           │  Rate of Gradient Descent│
           │  📄 Noise of SGD          │
           │  ...                      │
           └──────────────────────────┘
```

---

## Features

- **Agentic natural-language search** — ask "what should I read to become an AI
  engineer?" or "I'm learning CRISPR, key papers?" and get actual *engineering /
  science* papers, not policy reports or bibliometric meta-analyses
- **LLM relevance judge** — a second LLM pass filters the candidate pool;
  rejects papers that are off-topic, mismatched to your level, or outside the
  inferred domain
- **Community discovery** — for vague/beginner queries, searches the general web
  (Reddit, HN, GitHub, domain-specific forums) to find what people actually
  recommend reading
- **Curated seed catalog** — last-resort fallback: ~15 foundational papers per
  domain (AI/ML, biology, chemistry, physics, quantum computing, etc.) so
  common "what to learn" questions always get a good answer
- **Semantic re-ranking** — SBERT (`all-MiniLM-L6-v2`) re-ranks search results
  by cosine similarity to your query, with a recency-decay bonus
- **SQLite cache** — 24h TTL, avoids hammering the arXiv API
- **RAG chat with any paper** — download PDF → parse with PyMuPDF →
  section-aware chunking → `BAAI/bge-small-en-v1.5` embeddings →
  per-paper FAISS index → top-k retrieval → LLM answer with cited sources
- **Local-first LLM** — Ollama by default (no API key, free), plug in OpenAI
  (or any OpenAI-compatible proxy), or Anthropic Claude
- **Three interfaces** — web UI (FastAPI, dark-themed), CLI (click), legacy
  Streamlit UI

---

## Quick start

```bash
git clone https://github.com/01danl/arxiv-seeker.git
cd arxiv-seeker
python -m venv venv && source venv/bin/activate
pip install -e .
cp .env.example .env  # edit with your API keys
```

**You need at least one LLM backend.** The easiest free option is Ollama:

```bash
# Install from https://ollama.com, then:
ollama pull llama3.1
ollama serve
```

For cloud LLMs, set `LLM_PROVIDER=openai` (or `anthropic`) and fill in the
API key in `.env`. An OpenAI-compatible proxy works too — just set
`OPENAI_BASE_URL` to your proxy endpoint.

**For agentic search**, you also need a [Tavily API key](https://tavily.com)
(free tier available). Set `TAVILY_API_KEY` in `.env`.

---

## Usage

### Web UI (recommended)

```bash
arxiv-seeker serve --reload
# → http://127.0.0.1:8000
```

Professional dark-themed interface with:
- Chat with the agent to find papers
- Paper cards with abstracts, reasons, expandable details
- Slide-out RAG panel — chat with a paper, see sources with relevance scores
- All history persisted in localStorage (survives refresh)
- Swagger docs at `/docs`

### CLI

```bash
# Agentic search (natural language)
arxiv-seeker ask "I want to learn gradient descent, important papers?"

# Keyword search (direct arXiv API)
arxiv-seeker search "attention is all you need" --max 10

# Index a paper for RAG
arxiv-seeker rag index --paper-id 1706.03762

# Chat with it
arxiv-seeker chat --paper-id 1706.03762 "What is the main contribution?"
```

### Python API

```python
from arxiv_seeker.search import SearchOrchestrator
from arxiv_seeker.rag.chat import PaperIndexer, RagChat
from arxiv_seeker.agent.orchestrator import AgentSearchOrchestrator

# Option 1: Keyword search + SBERT re-rank
papers = SearchOrchestrator().search("gradient descent optimization", max_results=5)

# Option 2: Agentic search (LLM-understands intent)
result = AgentSearchOrchestrator().run("I'm learning gradient descent, what papers?")
print(result.reply_text, result.papers)

# RAG: index once, then ask questions
PaperIndexer().index(papers[0].arxiv_id)
answer = RagChat(papers[0].arxiv_id).ask("What is the convergence rate?")
print(answer.answer)
for src in answer.sources:
    print(f"  [{src.heading}] (score={src.score:.3f}) {src.text[:120]}")
```

See `examples/` for more.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Interfaces                                               │
│  Web UI (FastAPI)  ·  CLI (click)  ·  Streamlit (legacy) │
├──────────────────────────────────────────────────────────┤
│  Agentic Search Layer                                     │
│  IntentAgent  →  CommunityDiscovery  →  RelevanceJudge   │
│       │                  │                    │          │
│       │          Tavily (web)             Filters         │
│       │          Seed catalog           candidates        │
│       ▼                  ▼                    ▼          │
│  ┌─────────────────────────────────────────────────┐     │
│  │         Search Orchestrator                      │     │
│  │  arXiv API  ·  SBERT re-rank  ·  SQLite cache    │     │
│  └─────────────────────────────────────────────────┘     │
├──────────────────────────────────────────────────────────┤
│  RAG Engine                                               │
│  PDF → PyMuPDF → Sections → Chunks → Embed → FAISS → LLM │
└──────────────────────────────────────────────────────────┘
```

**RAG pipeline detail:** PDF is parsed with PyMuPDF and section headings are
detected via regex heuristics (Abstract, Introduction, Methods, etc.). Text is
chunked section-first — if a section is short enough it stays as one chunk;
only oversized sections are split with a sliding window. Embeddings use
`BAAI/bge-small-en-v1.5` (384-dim). FAISS `IndexFlatIP` stores normalized
vectors for cosine similarity. At query time, top-k chunks are retrieved
and fed to the LLM with a prompt that forbids guessing.

Indexes persist under `data/faiss/<arxiv_id>/` — re-visiting a paper skips
re-indexing.

---

## Configuration

All settings are in `arxiv_seeker/config.py` (Pydantic Settings, `.env`-driven).

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `openai` \| `anthropic` |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name (supports proxies like `deepseek/deepseek-v3.2`) |
| `OPENAI_BASE_URL` | — | OpenAI-compatible proxy endpoint |
| `TAVILY_API_KEY` | — | Required for agentic search (free tier: [tavily.com](https://tavily.com)) |
| `SEARCH_BACKEND` | `tavily` | `tavily` \| `arxiv` (direct arXiv API, no LLM filtering) |
| `RERANK_MODEL` | `all-MiniLM-L6-v2` | SBERT model for semantic re-ranking |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | RAG chunk embeddings |
| `CHUNK_SIZE_TOKENS` | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `50` | Overlap between sliding-window chunks |
| `TOP_K_CHUNKS` | `4` | Chunks retrieved per RAG question |
| `CACHE_TTL_HOURS` | `24` | Search result cache lifetime |
| `AGENT_CANDIDATES_PER_QUERY` | `8` | Papers fetched per generated query |
| `AGENT_FINAL_TOP_N` | `6` | Max papers returned to the user |
| `AGENT_MIN_PAPERS` | `2` | Trigger refinement iteration below this count |

---

## Testing

```bash
pip install -e ".[dev]"
pytest --cov=arxiv_seeker
```

Tests cover the cache, chunker, and search orchestration with mocked I/O — no
network or GPU needed.

---

## Roadmap

- [ ] Streaming responses (SSE) for long agent searches
- [ ] Cross-encoder re-ranking pass on FAISS top-k
- [ ] Multi-paper comparison mode
- [ ] Citation graph visualization
- [ ] Zotero / Mendeley export
- [ ] Personal weekly digest emails
- [ ] Integration with [Semantic Scholar](https://api.semanticscholar.org) for citation counts

---

## License

Apache-2.0 © ArxivSeeker contributors