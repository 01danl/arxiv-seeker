"""Streamlit UI for ArxivSeeker: search papers, chat with a selected paper."""
from __future__ import annotations

import logging

import streamlit as st

from arxiv_seeker.rag.chat import PaperIndexer, RagChat
from arxiv_seeker.search import SearchOrchestrator
from arxiv_seeker.agent.orchestrator import AgentSearchOrchestrator
from arxiv_seeker.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

st.set_page_config(page_title="ArxivSeeker", page_icon="🔎", layout="wide")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
_defaults = {
    "orchestrator": SearchOrchestrator(),
    "results": [],
    "active_paper": None,
    "chat_history": {},          # arxiv_id -> [(role, text), ...]
    "agent_messages": [],        # AI Assistant conversation
    "agent_result": None,
    "indexed_papers": set(),     # papers already indexed
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.title("🔎 ArxivSeeker")
st.caption("Semantic search over arXiv, with RAG chat on any paper.")

# ---------------------------------------------------------------------------
# Sidebar mode selector
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose mode", ["Search", "AI Assistant"], index=0)

# ===========================================================================
#  MODE 1: Search
# ===========================================================================
if mode == "Search":
    with st.sidebar:
        st.header("Search")
        query = st.text_input(
            "Query", placeholder='e.g. "diffusion models medical imaging"'
        )
        category = st.text_input(
            "Category (optional)", placeholder="e.g. cs.LG, cs.CL",
            help="Leave empty to search all categories.",
        )
        max_results = st.slider("Max results", 5, 30, 10)
        sort_by = st.selectbox("Sort by", ["relevance", "submitted", "lastUpdated"])
        do_search = st.button("Search", type="primary", use_container_width=True)

    if do_search and query:
        arxiv_query = f"cat:{category.strip()} AND {query}" if category.strip() else query
        with st.spinner("Searching arXiv and re-ranking..."):
            st.session_state.results = st.session_state.orchestrator.search(
                arxiv_query, max_results=max_results, sort_by=sort_by
            )
        st.session_state.active_paper = None

    # --- Results ---
    st.subheader("Results")
    if not st.session_state.results:
        st.info("Run a search to see papers here.")
    else:
        for p in st.session_state.results:
            with st.container(border=True):
                st.markdown(f"**{p.title}**")
                st.caption(
                    f"{p.arxiv_id} · {p.published.date()} · score {p.final_score:.3f} · "
                    f"{', '.join(p.categories[:3])}"
                )
                st.write(p.abstract[:280] + ("..." if len(p.abstract) > 280 else ""))
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.link_button("Open on arXiv", p.entry_url, use_container_width=True)
                with c2:
                    if st.button("💬 Chat", key=f"chat_{p.arxiv_id}", use_container_width=True):
                        st.session_state.active_paper = p.arxiv_id
                        st.session_state.chat_history.setdefault(p.arxiv_id, [])
                        st.rerun()

    # --- Chat panel (shown when a paper is selected) ---
    if st.session_state.active_paper:
        st.divider()
        _render_chat_panel(st.session_state.active_paper)

# ===========================================================================
#  MODE 2: AI Assistant
# ===========================================================================
else:
    st.subheader("🤖 AI Assistant")
    st.caption("Ask me about research topics – I will find and recommend relevant arXiv papers.")

    # --- Conversation history ---
    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # --- Chat input ---
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.agent_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your request and searching for papers..."):
                settings = get_settings()
                orch = AgentSearchOrchestrator(
                    config_overrides={
                        "candidates_per_query": settings.agent_candidates_per_query,
                        "final_top_n": settings.agent_final_top_n,
                        "min_papers": settings.agent_min_papers,
                    }
                )
                result = orch.run(prompt)
                st.session_state.agent_result = result

                st.write(result.reply_text)
                if result.papers:
                    for i, paper in enumerate(result.papers, 1):
                        with st.expander(f"{i}. {paper.title}"):
                            st.write(f"**Authors:** {', '.join(paper.authors)}")
                            st.write(f"**Abstract:** {paper.abstract[:500]}...")
                            st.write(f"**URL:** {paper.pdf_url}")
                            reason = result.reasons.get(paper.arxiv_id, "")
                            if reason:
                                st.info(f"**Why?** {reason}")
                            if st.button("💬 Chat with this paper", key=f"agent_chat_{paper.arxiv_id}"):
                                st.session_state.active_paper = paper.arxiv_id
                                st.session_state.chat_history.setdefault(paper.arxiv_id, [])
                                st.rerun()

                    titles = "\n".join(f"- {p.title}" for p in result.papers)
                    st.session_state.agent_messages.append({
                        "role": "assistant",
                        "content": f"{result.reply_text}\n\n{titles}",
                    })
                else:
                    st.session_state.agent_messages.append({
                        "role": "assistant",
                        "content": "No relevant papers found. Try a different query.",
                    })

    # --- Chat panel (shown when a paper is selected from AI Assistant) ---
    if st.session_state.active_paper:
        st.divider()
        _render_chat_panel(st.session_state.active_paper)


# ===========================================================================
#  Shared chat panel (rendered below results in either mode)
# ===========================================================================
def _render_chat_panel(paper_id: str):
    """Render the RAG chat interface for *paper_id*."""
    st.subheader(f"💬 Chat with `{paper_id}`")

    # Index the paper if needed
    if paper_id not in st.session_state.indexed_papers:
        with st.spinner(f"Indexing {paper_id} (download → parse → chunk → embed)..."):
            try:
                PaperIndexer().index(paper_id)
                st.session_state.indexed_papers.add(paper_id)
            except Exception as exc:
                st.error(f"Failed to index paper: {exc}")
                return

    # Show chat history
    history = st.session_state.chat_history.get(paper_id, [])
    for role, text in history:
        with st.chat_message(role):
            st.write(text)

    # Chat input
    question = st.chat_input(f"Ask about {paper_id}...", key=f"rag_input_{paper_id}")
    if question:
        history.append(("user", question))
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                try:
                    session = RagChat(paper_id)
                    result = session.ask(question)
                    st.write(result.answer)
                    with st.expander("📎 Sources"):
                        for src in result.sources:
                            st.markdown(f"**{src.heading}** (score={src.score:.3f})")
                            st.caption(src.text[:300] + "...")
                    history.append(("assistant", result.answer))
                except Exception as exc:
                    st.error(f"Chat failed: {exc}")

        # Persist history
        st.session_state.chat_history[paper_id] = history

    # Close chat button
    if st.button("✕ Close chat", key=f"close_{paper_id}"):
        st.session_state.active_paper = None
        st.rerun()