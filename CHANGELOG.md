# Changelog

## [0.1.0] - 2026-08-18
### Added
- Initial release: arXiv search + SBERT re-ranking + SQLite cache
- RAG engine: PyMuPDF parsing, section-aware chunking, BAAI/bge embeddings, FAISS vector store
- LLM chat backends: Ollama (default), OpenAI, Anthropic
- CLI (`click`) with `search`, `rag index`, `chat` commands
- Streamlit web UI with search + per-paper chat panel
- Test suite for cache, chunker, and search orchestration
