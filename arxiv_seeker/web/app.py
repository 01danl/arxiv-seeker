"""Streamlit UI for ArxivSeeker — AI Assistant with inline RAG chat."""
from __future__ import annotations

import logging

import streamlit as st

from arxiv_seeker.rag.chat import PaperIndexer, RagChat
from arxiv_seeker.agent.orchestrator import AgentSearchOrchestrator
from arxiv_seeker.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

st.set_page_config(page_title="ArxivSeeker", page_icon="🔎", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [(role, text), ...]
if "papers" not in st.session_state:
    st.session_state.papers = []    # latest search results
if "reasons" not in st.session_state:
    st.session_state.reasons = {}   # arxiv_id -> reason
if "active_paper" not in st.session_state:
    st.session_state.active_paper = None
if "paper_chat" not in st.session_state:
    st.session_state.paper_chat = {}  # arxiv_id -> [(role, text), ...]
if "indexed" not in st.session_state:
    st.session_state.indexed = set()

st.title("🔎 ArxivSeeker")
st.caption("AI-powered research assistant — ask me anything about academic papers.")

# ===========================================================================
#  Main chat — agent conversation
# ===========================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat input ---
if prompt := st.chat_input("What would you like to know?"):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.active_paper = None  # new search → close any open paper chat
    with st.chat_message("user"):
        st.write(prompt)

    # Agent search
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            settings = get_settings()
            orch = AgentSearchOrchestrator(
                config_overrides={
                    "candidates_per_query": settings.agent_candidates_per_query,
                    "final_top_n": settings.agent_final_top_n,
                    "min_papers": settings.agent_min_papers,
                }
            )
            result = orch.run(prompt)

            st.write(result.reply_text)
            st.session_state.papers = result.papers
            st.session_state.reasons = result.reasons

            if result.papers:
                titles = "\n".join(f"- {p.title}" for p in result.papers)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"{result.reply_text}\n\n{titles}",
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "No relevant papers found.",
                })

# ===========================================================================
#  Paper results (clickable)
# ===========================================================================
if st.session_state.papers and not st.session_state.active_paper:
    st.divider()
    st.subheader("📄 Papers found")
    for i, paper in enumerate(st.session_state.papers, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {paper.title}**")
            st.caption(
                f"{paper.arxiv_id} · {paper.published.date()} · "
                f"{', '.join(paper.categories[:3])}"
            )
            st.write(paper.abstract[:300] + ("..." if len(paper.abstract) > 300 else ""))
            reason = st.session_state.reasons.get(paper.arxiv_id, "")
            if reason:
                st.info(f"💡 {reason}")

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.link_button("arXiv", paper.entry_url, use_container_width=True)
            with c2:
                st.link_button("PDF", paper.pdf_url, use_container_width=True)
            with c3:
                if st.button("💬 Chat", key=f"open_chat_{paper.arxiv_id}", use_container_width=True):
                    st.session_state.active_paper = paper.arxiv_id
                    st.session_state.paper_chat.setdefault(paper.arxiv_id, [])
                    st.rerun()

# ===========================================================================
#  RAG chat panel (shown when a paper is selected)
# ===========================================================================
if st.session_state.active_paper:
    paper_id = st.session_state.active_paper
    st.divider()
    st.subheader(f"💬 Chatting with `{paper_id}`")

    # Index if needed
    if paper_id not in st.session_state.indexed:
        with st.spinner(f"Indexing {paper_id}..."):
            try:
                PaperIndexer().index(paper_id)
                st.session_state.indexed.add(paper_id)
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")
                st.stop()

    # History
    history = st.session_state.paper_chat.get(paper_id, [])
    for role, text in history:
        with st.chat_message(role):
            st.write(text)

    # Input
    c1, c2 = st.columns([5, 1])
    with c1:
        q = st.text_input(
            "Question",
            key=f"pq_{paper_id}",
            placeholder="Ask about this paper...",
            label_visibility="collapsed",
        )
    with c2:
        send = st.button("Send", key=f"psend_{paper_id}", use_container_width=True)

    if send and q.strip():
        history.append(("user", q))

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    session = RagChat(paper_id)
                    ans = session.ask(q)
                    st.write(ans.answer)
                    with st.expander("📎 Sources"):
                        for src in ans.sources:
                            st.markdown(f"**{src.heading}** (score={src.score:.3f})")
                            st.caption(src.text[:300] + "...")
                    history.append(("assistant", ans.answer))
                except Exception as exc:
                    st.error(f"Chat failed: {exc}")

        st.session_state.paper_chat[paper_id] = history
        st.rerun()

    # Close
    if st.button("✕ Close", key=f"close_{paper_id}"):
        st.session_state.active_paper = None
        st.rerun()