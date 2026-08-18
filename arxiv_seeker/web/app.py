"""Streamlit UI for ArxivSeeker: search papers, chat with a selected paper."""
from __future__ import annotations

import streamlit as st

from arxiv_seeker.rag.chat import PaperIndexer, RagChat
from arxiv_seeker.search import SearchOrchestrator

st.set_page_config(page_title="ArxivSeeker", page_icon="🔎", layout="wide")

# Инициализация состояния
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = SearchOrchestrator()
if "results" not in st.session_state:
    st.session_state.results = []
if "active_paper" not in st.session_state:
    st.session_state.active_paper = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

st.title("🔎 ArxivSeeker")
st.caption("Semantic search over arXiv, with RAG chat on any paper.")

with st.sidebar:
    st.header("Search")
    # --- Основное поле запроса ---
    query = st.text_input(
        "Query (topic or keywords)",
        placeholder='e.g. "diffusion models medical imaging"'
    )
    # --- НОВОЕ: поле для категории ---
    category = st.text_input(
        "Category (optional)",
        placeholder="e.g. cs.LG, cs.CL, physics",
        help="Leave empty to search all categories. You can also write 'cat:xxx' in the query above."
    )
    max_results = st.slider("Max results", 5, 30, 10)
    sort_by = st.selectbox("Sort by", ["relevance", "submitted", "lastUpdated"])
    do_search = st.button("Search", type="primary", use_container_width=True)

# --- Логика поиска ---
if do_search and query:
    # Формируем запрос для arXiv с учётом категории
    if category.strip():
        arxiv_query = f"cat:{category.strip()} AND {query}"
    else:
        arxiv_query = query

    with st.spinner("Searching arXiv and re-ranking..."):
        st.session_state.results = st.session_state.orchestrator.search(
            arxiv_query, max_results=max_results, sort_by=sort_by
        )
    st.session_state.active_paper = None

# --- Интерфейс: результаты слева, чат справа ---
col_results, col_chat = st.columns([1, 1])

with col_results:
    st.subheader("Results")
    if not st.session_state.results:
        st.info("Run a search to see papers here.")
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
                if st.button("💬 Chat with this paper", key=f"chat_{p.arxiv_id}", use_container_width=True):
                    st.session_state.active_paper = p.arxiv_id
                    st.session_state.chat_history.setdefault(p.arxiv_id, [])

with col_chat:
    st.subheader("RAG Chat")
    paper_id = st.session_state.active_paper
    if not paper_id:
        st.info("Select a paper on the left to start chatting.")
    else:
        st.markdown(f"**Chatting with:** `{paper_id}`")

        index_key = f"indexed_{paper_id}"
        if index_key not in st.session_state:
            with st.spinner(f"Indexing {paper_id} (download → parse → chunk → embed)..."):
                try:
                    PaperIndexer().index(paper_id)
                    st.session_state[index_key] = True
                except Exception as e:
                    st.error(f"Failed to index paper: {e}")
                    st.stop()

        for role, text in st.session_state.chat_history.get(paper_id, []):
            with st.chat_message(role):
                st.write(text)

        question = st.chat_input("Ask about this paper...")
        if question:
            st.session_state.chat_history[paper_id].append(("user", question))
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving relevant excerpts and generating answer..."):
                    try:
                        session = RagChat(paper_id)
                        result = session.ask(question)
                        st.write(result.answer)
                        with st.expander("Sources"):
                            for src in result.sources:
                                st.markdown(f"**{src.heading}** (score={src.score:.3f})")
                                st.caption(src.text[:300] + "...")
                        st.session_state.chat_history[paper_id].append(("assistant", result.answer))
                    except Exception as e:
                        st.error(f"Chat failed: {e}")