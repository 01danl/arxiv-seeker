"""Orchestrates retrieval + LLM generation for conversational Q&A over a paper."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from arxiv_seeker.api_client import ArxivClient
from arxiv_seeker.config import get_settings
from arxiv_seeker.rag.chunker import chunk_paper
from arxiv_seeker.rag.embedding import Embedder
from arxiv_seeker.rag.pdf_parser import parse_pdf
from arxiv_seeker.rag.vector_store import VectorStore
from arxiv_seeker.llm import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question about a "
    "scientific paper using ONLY the excerpts provided below. If the excerpts "
    "do not contain enough information to answer, say so explicitly instead of "
    "guessing. When you use a fact, mention which section it came from "
    "(e.g. 'per the Results section'). Be concise and technically accurate."
)


@dataclass
class RetrievedChunk:
    heading: str
    text: str
    score: float


@dataclass
class ChatAnswer:
    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)


class PaperIndexer:
    """Handles downloading, parsing, chunking and indexing a paper for RAG."""

    def __init__(self, embedder: Optional[Embedder] = None, client: Optional[ArxivClient] = None):
        self.embedder = embedder or Embedder()
        self.client = client or ArxivClient()

    def index(self, arxiv_id: str, force: bool = False, download_dir: str = "/tmp/arxiv_pdfs") -> VectorStore:
        store = VectorStore(arxiv_id, dimension=self.embedder.dimension)
        if store.exists() and not force:
            logger.info("Index already exists for %s, skipping re-index", arxiv_id)
            store.load()
            return store

        Path(download_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Downloading PDF for %s", arxiv_id)
        pdf_path = self.client.download_pdf(arxiv_id, download_dir)

        logger.info("Parsing PDF for %s", arxiv_id)
        parsed = parse_pdf(pdf_path, arxiv_id)

        logger.info("Chunking paper %s into sections", arxiv_id)
        chunks = chunk_paper(parsed)
        if not chunks:
            raise ValueError(f"No extractable text found for paper {arxiv_id}")

        logger.info("Embedding %d chunks for %s", len(chunks), arxiv_id)
        embeddings = self.embedder.embed([c.text for c in chunks])

        store.build(chunks, embeddings)
        logger.info("Index built for %s (%d chunks)", arxiv_id, len(chunks))
        return store


class RagChat:
    """Retrieval + LLM call for a single paper's Q&A session."""

    def __init__(self, arxiv_id: str, embedder: Optional[Embedder] = None):
        self.llm = LLMClient()
        self.arxiv_id = arxiv_id
        self.settings = get_settings()
        self.embedder = embedder or Embedder()
        self.store = VectorStore(arxiv_id, dimension=self.embedder.dimension)
        if not self.store.exists():
            raise FileNotFoundError(
                f"Paper {arxiv_id} is not indexed yet. Run PaperIndexer().index('{arxiv_id}') first."
            )
        self.store.load()

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        top_k = top_k or self.settings.top_k_chunks
        query_emb = self.embedder.embed_query(question)
        results = self.store.search(query_emb, top_k=top_k)
        return [RetrievedChunk(heading=m["heading"], text=m["text"], score=s) for m, s in results]

    def _build_prompt(self, question: str, chunks: List[RetrievedChunk]) -> str:
        excerpts = "\n\n".join(
            f"[Excerpt {i+1} — {c.heading}]\n{c.text}" for i, c in enumerate(chunks)
        )
        return (
            f"Excerpts from the paper:\n\n{excerpts}\n\n"
            f"Question: {question}\n\n"
            f"Answer based only on the excerpts above."
        )

    def ask(self, question: str, top_k: Optional[int] = None) -> ChatAnswer:
        chunks = self.retrieve(question, top_k=top_k)
        prompt = self._build_prompt(question, chunks)
        answer_text = self._call_llm(prompt)
        return ChatAnswer(answer=answer_text, sources=chunks)

    # --- LLM backends -----------------------------------------------------
    def _call_llm(self, user_prompt: str) -> str:
        return self.llm.chat(_SYSTEM_PROMPT, user_prompt)

    def _call_ollama(self, user_prompt: str) -> str:
        import requests

        resp = requests.post(
            f"{self.settings.ollama_base_url}/api/chat",
            json={
                "model": self.settings.ollama_model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _call_openai(self, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,  # None -> official OpenAI endpoint
        )
        resp = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content

    def _call_anthropic(self, user_prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        resp = client.messages.create(
            model=self.settings.anthropic_model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))