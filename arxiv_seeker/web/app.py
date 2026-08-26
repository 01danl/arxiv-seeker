"""Streamlit UI for ArxivSeeker: search papers, chat with a selected paper."""
from __future__ import annotations

import logging

import streamlit as st

from arxiv_seeker.rag.chat import PaperIndexer, RagChat
from arxiv_seeker.search import SearchOrchestrator
from arxiv_seeker.agent.orchestrator import AgentSearchOrchestrator
from arxiv_seeker.config import get_settings

# Show agent diagnostics in the terminal (INTENT, Judge, community discovery, etc.)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

st.set_page_config(page_title="ArxivSeeker", page_icon="🔎", layout="wide")

# --- Инициализация состояния ---
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = SearchOrchestrator()
if "results" not in st.session_state:
    st.session_state.results = []
if "active_paper" not in st.session_state:
    st.session_state.active_paper = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None

st.title("🔎 ArxivSeeker")
st.caption("Semantic search over arXiv, with RAG chat on any paper.")

# --- Боковая панель с выбором режима ---
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choose mode", ["Search", "AI Assistant"], index=0)

# --- Режим 1: обычный поиск (как было) ---
if mode == "Search":
    with st.sidebar:
        st.header("Search")
        query = st.text_input("Query (topic or keywords)", placeholder='e.g. "diffusion models medical imaging"')
        category = st.text_input("Category (optional)", placeholder="e.g. cs.LG, cs.CL, physics",
                                 help="Leave empty to search all categories.")
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
                        st.rerun()

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

# --- Режим 2: AI Assistant (агентный поиск + рекомендации) ---
else:
    st.subheader("🤖 AI Assistant")
    st.caption("Ask me about research topics – I will find and recommend relevant arXiv papers.")

    # Отображаем историю чата (как в обычном чате)
    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Поле ввода
    if prompt := st.chat_input("What would you like to know?"):
        # Добавляем сообщение пользователя
        st.session_state.agent_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Запускаем агента
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your request and searching for papers..."):
                settings = get_settings()
                orchestrator = AgentSearchOrchestrator(
                    config_overrides={
                        "candidates_per_query": settings.agent_candidates_per_query,
                        "final_top_n": settings.agent_final_top_n,
                        "min_papers": settings.agent_min_papers,
                    }
                )
                result = orchestrator.run(prompt)
                st.session_state.agent_result = result

                # Ответ агента
                st.write(result.reply_text)
                if result.papers:
                    for i, paper in enumerate(result.papers, 1):
                        with st.expander(f"{i}. {paper.title}"):
                            st.write(f"**Authors:** {', '.join(paper.authors)}")
                            st.write(f"**Abstract:** {paper.abstract[:500]}...")
                            st.write(f"**URL:** {paper.pdf_url}")
                            reason = result.reasons.get(paper.arxiv_id, "")
                            if reason:
                                st.info(f"**Why this paper?** {reason}")
                            if st.button(f"Chat with paper", key=f"agent_chat_{paper.arxiv_id}"):
                                # Переключаем активную статью и переходим в режим поиска для RAG
                                st.session_state.active_paper = paper.arxiv_id
                                st.session_state.chat_history.setdefault(paper.arxiv_id, [])
                                # Можно сохранить результат, чтобы после переключения не потерять
                                st.rerun()

                    # Сохраняем сообщение ассистента в историю (для отображения при следующем запросе)
                    st.session_state.agent_messages.append({
                        "role": "assistant",
                        "content": result.reply_text + "\n\n" + "\n".join([f"- {p.title}" for p in result.papers])
                    })
                else:
                    st.session_state.agent_messages.append({
                        "role": "assistant",
                        "content": "No relevant papers found. Try a different query."
                    })