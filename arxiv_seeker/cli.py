"""Command-line interface for ArxivSeeker."""
from __future__ import annotations

import logging

import click

from arxiv_seeker.rag.chat import PaperIndexer, RagChat
from arxiv_seeker.search import SearchOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@click.group()
def cli():
    """ArxivSeeker — search, rank, and chat with arXiv papers."""
    pass


@cli.command()
@click.argument("query")
@click.option("--max", "max_results", default=10, help="Max number of results")
@click.option("--sort", "sort_by", default="relevance", type=click.Choice(["relevance", "submitted", "lastUpdated"]))
@click.option("--no-cache", is_flag=True, help="Bypass cache")
@click.option("--no-rerank", is_flag=True, help="Skip semantic re-ranking")
def search(query, max_results, sort_by, no_cache, no_rerank):
    """Search arXiv, e.g.: arxiv-seeker search "diffusion models medical" --max 10"""
    orchestrator = SearchOrchestrator()
    papers = orchestrator.search(
        query, max_results=max_results, sort_by=sort_by, use_cache=not no_cache, rerank=not no_rerank
    )
    if not papers:
        click.echo("No results found.")
        return
    for i, p in enumerate(papers, 1):
        click.echo(f"\n[{i}] {p.title}")
        click.echo(f"    id: {p.arxiv_id}  score: {p.final_score:.3f}  published: {p.published.date()}")
        click.echo(f"    authors: {', '.join(p.authors[:4])}{' et al.' if len(p.authors) > 4 else ''}")
        click.echo(f"    {p.abstract[:220]}...")


@cli.group()
def rag():
    """RAG indexing and chat commands."""
    pass


@rag.command("index")
@click.option("--paper-id", required=True, help="arXiv paper ID, e.g. 2203.12345")
@click.option("--force", is_flag=True, help="Re-index even if already indexed")
def rag_index(paper_id, force):
    """Download, parse, chunk, and index a paper for RAG."""
    indexer = PaperIndexer()
    indexer.index(paper_id, force=force)
    click.echo(f"Indexed paper {paper_id}.")


@cli.command("chat")
@click.option("--paper-id", required=True, help="arXiv paper ID, e.g. 2203.12345")
@click.argument("question")
@click.option("--top-k", default=None, type=int, help="Number of chunks to retrieve")
def chat(paper_id, question, top_k):
    """Ask a question about an indexed paper.

    arxiv-seeker chat --paper-id 1234.5678 "What is the main contribution?"
    """
    try:
        session = RagChat(paper_id)
    except FileNotFoundError:
        click.echo(f"Paper {paper_id} isn't indexed yet. Indexing now...")
        PaperIndexer().index(paper_id)
        session = RagChat(paper_id)

    result = session.ask(question, top_k=top_k)
    click.echo(f"\n{result.answer}\n")
    click.echo("Sources:")
    for src in result.sources:
        click.echo(f"  - [{src.heading}] (score={src.score:.3f}) {src.text[:120]}...")

@cli.command()
@click.argument("message", nargs=-1, required=True)
def ask(message):
    """Агентный поиск статей на естественном языке."""
    from arxiv_seeker.agent.orchestrator import AgentSearchOrchestrator
    from arxiv_seeker.config import get_settings

    settings = get_settings()
    orchestrator = AgentSearchOrchestrator(
        config_overrides={
            "candidates_per_query": settings.agent_candidates_per_query,
            "final_top_n": settings.agent_final_top_n,
            "min_papers": settings.agent_min_papers,
        }
    )
    user_msg = " ".join(message)
    result = orchestrator.run(user_msg)

    click.echo(result.reply_text)
    if not result.papers:
        return

    for i, paper in enumerate(result.papers, 1):
        click.echo(f"\n{i}. {paper.title}")
        click.echo(f"   {paper.pdf_url}")
        reason = result.reasons.get(paper.arxiv_id, "")
        if reason:
            click.echo(f"   Почему: {reason}")


@cli.command("serve")
@click.option("--host", default="127.0.0.1", help="Bind host")
@click.option("--port", default=8000, help="Bind port")
@click.option("--reload", "reload", is_flag=True, help="Enable auto-reload (dev)")
def serve(host, port, reload):
    """Run the FastAPI backend + web UI.

    arxiv-seeker serve --reload
    """
    import uvicorn

    click.echo(f"🔥 ArxivSeeker UI: http://{host}:{port}")
    uvicorn.run("arxiv_seeker.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    cli()
