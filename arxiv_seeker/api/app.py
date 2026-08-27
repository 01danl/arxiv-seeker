"""FastAPI backend for ArxivSeeker: agent paper search + RAG paper chat.

Run with:
    uvicorn arxiv_seeker.api.app:app --reload
or:
    arxiv-seeker serve
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from arxiv_seeker.api_client import Paper

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="ArxivSeeker API",
    description="AI-powered research assistant: find, rank and chat with arXiv papers.",
    version="0.3.0",
)


# ---------------------------------------------------------------------------
#  Request / response models
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=2000)


class IndexRequest(BaseModel):
    paper_id: str


class ChatRequest(BaseModel):
    paper_id: str
    question: str = Field(..., min_length=2, max_length=2000)


class PaperOut(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    categories: List[str]
    pdf_url: str
    entry_url: str
    reason: str = ""


class AskResponse(BaseModel):
    reply: str
    papers: List[PaperOut]


class SourceOut(BaseModel):
    heading: str
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceOut]


class IndexResponse(BaseModel):
    status: str
    paper_id: str


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _paper_to_out(p: Paper, reason: str = "") -> PaperOut:
    return PaperOut(
        arxiv_id=p.arxiv_id,
        title=p.title,
        abstract=p.abstract,
        authors=p.authors,
        published=p.published.date().isoformat(),
        categories=p.categories,
        pdf_url=p.pdf_url,
        entry_url=p.entry_url,
        reason=reason,
    )


@lru_cache
def _get_agent_orchestrator():
    """Construct the agent pipeline once and reuse it across requests."""
    from arxiv_seeker.agent.orchestrator import AgentSearchOrchestrator
    from arxiv_seeker.config import get_settings

    settings = get_settings()
    return AgentSearchOrchestrator(
        config_overrides={
            "candidates_per_query": settings.agent_candidates_per_query,
            "final_top_n": settings.agent_final_top_n,
            "min_papers": settings.agent_min_papers,
        }
    )


# ---------------------------------------------------------------------------
#  API routes
# ---------------------------------------------------------------------------
@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Understand a natural-language request, search, rank and filter papers."""
    try:
        orchestrator = _get_agent_orchestrator()
        result = orchestrator.run(req.message)
    except Exception as exc:  # noqa: BLE001 — surface any backend failure cleanly
        logger.exception("Agent search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    papers = [
        _paper_to_out(p, reason=result.reasons.get(p.arxiv_id, ""))
        for p in result.papers
    ]
    return AskResponse(reply=result.reply_text, papers=papers)


@app.post("/api/paper/index", response_model=IndexResponse)
def index_paper(req: IndexRequest) -> IndexResponse:
    """Download, parse, chunk and embed a paper for RAG chat (idempotent)."""
    from arxiv_seeker.rag.chat import PaperIndexer

    try:
        PaperIndexer().index(req.paper_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Indexing failed for %s", req.paper_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IndexResponse(status="ok", paper_id=req.paper_id)


@app.post("/api/paper/chat", response_model=ChatResponse)
def paper_chat(req: ChatRequest) -> ChatResponse:
    """Ask a question about an indexed paper; answer is grounded in its text."""
    from arxiv_seeker.rag.chat import RagChat

    try:
        session = RagChat(req.paper_id)
        result = session.ask(req.question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG chat failed for %s", req.paper_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceOut(heading=s.heading, text=s.text, score=round(s.score, 3))
            for s in result.sources
        ],
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
#  Static frontend (must be mounted AFTER api routes)
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="web")
else:  # pragma: no cover
    logger.warning("Static dir %s not found — API-only mode", STATIC_DIR)