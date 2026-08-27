/* ============================================================
   ArxivSeeker — frontend logic
   Chat with the agent, browse paper cards, RAG-chat with a paper.
   ============================================================ */
"use strict";

/* ---------------- state ---------------- */
const STORAGE_KEY = "arxivseeker_v1";

const state = {
  messages: [],          // {role: 'user'|'assistant', content, papers?, error?, retry?}
  paperChats: {},        // paperId -> {title, messages: [{role, content, sources?}]}
  indexedPapers: [],     // paper ids known-indexed this browser session
  busy: false,
  ragBusy: false,
  activePaper: null,     // paperId currently open in the RAG panel
};

/* ---------------- dom ---------------- */
const $ = (id) => document.getElementById(id);
const chatInner = $("chatInner");
const chatScroll = $("chatScroll");
const input = $("input");
const sendBtn = $("sendBtn");
const emptyState = $("emptyState");
const ragPanel = $("ragPanel");
const ragOverlay = $("ragOverlay");
const ragTitle = $("ragTitle");
const ragMessages = $("ragMessages");
const ragEmpty = $("ragEmpty");
const ragInput = $("ragInput");
const ragSendBtn = $("ragSendBtn");
const ragStatus = $("ragStatus");
const toast = $("toast");

/* ---------------- persistence ---------------- */
function save() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        messages: state.messages.filter((m) => !m.error),
        paperChats: state.paperChats,
        indexedPapers: state.indexedPapers,
      })
    );
  } catch (_) {
    /* storage full / disabled — ignore */
  }
}

function load() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    state.messages = Array.isArray(data.messages) ? data.messages : [];
    state.paperChats = data.paperChats || {};
    state.indexedPapers = data.indexedPapers || [];
  } catch (_) {
    /* corrupted storage — start fresh */
  }
}

/* ---------------- helpers ---------------- */
function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

const CAT_PALETTE = [
  ["#f87171", "rgba(248,113,113,.12)"],
  ["#fb923c", "rgba(251,146,60,.12)"],
  ["#fbbf24", "rgba(251,191,36,.12)"],
  ["#34d399", "rgba(52,211,153,.12)"],
  ["#38bdf8", "rgba(56,189,248,.12)"],
  ["#818cf8", "rgba(129,140,248,.12)"],
  ["#c084fc", "rgba(192,132,252,.12)"],
  ["#f472b6", "rgba(244,114,182,.12)"],
];

function catColor(cat) {
  let h = 0;
  for (const ch of cat) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return CAT_PALETTE[h % CAT_PALETTE.length];
}

function catBadge(cat) {
  const [fg, bg] = catColor(cat);
  return `<span class="cat-badge" style="color:${fg};background:${bg};border:1px solid ${fg}33">${escapeHtml(cat)}</span>`;
}

function formatDate(iso) {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch (_) {
    return iso;
  }
}

function scrollChatBottom() {
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function scrollRagBottom() {
  ragMessages.scrollTop = ragMessages.scrollHeight;
}

let toastTimer = null;
function showToast(text, isError = false) {
  toast.textContent = text;
  toast.classList.toggle("error", isError);
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (toast.hidden = true), 3500);
}

/* rotating status line while waiting */
function makeStatusRotator(el, phrases, intervalMs = 3800) {
  let i = 0;
  el.textContent = phrases[0];
  const timer = setInterval(() => {
    i = (i + 1) % phrases.length;
    el.textContent = phrases[i];
  }, intervalMs);
  return () => clearInterval(timer);
}

/* ---------------- api ---------------- */
async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = null;
  try {
    data = await res.json();
  } catch (_) {
    /* non-json error page */
  }
  if (!res.ok) {
    throw new Error(data?.detail || `Request failed (${res.status})`);
  }
  return data;
}

/* ============================================================
   Main chat rendering
   ============================================================ */
function render() {
  // messages
  chatInner.querySelectorAll(".msg, .msg-error-wrap").forEach((n) => n.remove());

  if (state.messages.length === 0) {
    emptyState.style.display = "";
  } else {
    emptyState.style.display = "none";
    for (const m of state.messages) {
      chatInner.appendChild(renderMessage(m));
    }
  }
  save();
}

function renderMessage(m) {
  const wrap = document.createElement("div");

  if (m.error) {
    wrap.className = "msg-error-wrap";
    wrap.innerHTML = `
      <div class="msg assistant">
        <div class="msg-avatar">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round">
            <path d="M12 8v5"/><circle cx="12" cy="16.6" r="0.4" fill="currentColor"/>
            <circle cx="12" cy="12" r="10" opacity=".35"/>
          </svg>
        </div>
        <div class="msg-body">
          <div class="msg-error">
            <span>${escapeHtml(m.content)}</span>
            <button class="retry-btn">Retry</button>
          </div>
        </div>
      </div>`;
    wrap.querySelector(".retry-btn").addEventListener("click", () => {
      state.messages = state.messages.filter((x) => x !== m);
      const lastUser = [...state.messages].reverse().find((x) => x.role === "user");
      if (lastUser) {
        state.messages = state.messages.slice(0, state.messages.indexOf(lastUser) + 1);
        render();
        askAgent(lastUser.content);
      }
    });
    return wrap;
  }

  wrap.className = `msg ${m.role}`;
  const avatar =
    m.role === "user"
      ? `<div class="msg-avatar"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>`
      : `<div class="msg-avatar"><svg viewBox="0 0 32 32" width="15" height="15" fill="none"><path d="M14 7a7 7 0 1 0 0 14 7 7 0 0 0 0-14zm5 12.5l6 6" stroke="#fff" stroke-width="3.4" stroke-linecap="round"/></svg></div>`;

  wrap.innerHTML = `${avatar}<div class="msg-body"></div>`;
  const body = wrap.querySelector(".msg-body");

  if (m.role === "assistant" && m.pending) {
    body.innerHTML = `
      <div class="typing">
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
        <span class="typing-status"></span>
      </div>`;
    const stop = makeStatusRotator(
      body.querySelector(".typing-status"),
      [
        "Understanding your request…",
        "Searching the web & arXiv…",
        "Ranking candidates…",
        "Checking relevance…",
      ]
    );
    wrap._stopStatus = stop;
  } else {
    const textEl = document.createElement("div");
    textEl.className = "msg-text";
    textEl.textContent = m.content;
    body.appendChild(textEl);

    if (m.role === "assistant" && m.papers?.length) {
      body.appendChild(renderPapersBlock(m.papers));
    }
  }
  return wrap;
}

function renderPapersBlock(papers) {
  const block = document.createElement("div");
  block.className = "papers-block";

  const header = document.createElement("div");
  header.className = "papers-header";
  header.innerHTML = `<span>${papers.length} paper${papers.length === 1 ? "" : "s"} found</span>`;
  block.appendChild(header);

  const grid = document.createElement("div");
  grid.className = "papers-grid";
  papers.forEach((p, i) => grid.appendChild(renderPaperCard(p, i)));
  block.appendChild(grid);
  return block;
}

function renderPaperCard(p, idx) {
  const card = document.createElement("article");
  card.className = "paper-card";
  card.style.animation = `fade-up .4s var(--ease) ${idx * 0.06}s both`;

  const cats = (p.categories || []).slice(0, 3).map(catBadge).join("");
  const reason = p.reason
    ? `<div class="paper-reason"><span class="reason-icon">💡</span><span>${escapeHtml(p.reason)}</span></div>`
    : "";
  const authors = (p.authors || []).slice(0, 3).join(", ") + ((p.authors || []).length > 3 ? " et al." : "");
  const abstractLong = (p.abstract || "").length > 220;

  card.innerHTML = `
    <div class="paper-top">
      <h3 class="paper-title">${escapeHtml(p.title)}</h3>
      <span class="paper-index">${escapeHtml(p.arxiv_id)}</span>
    </div>
    <div class="paper-meta">
      <span class="paper-date">${escapeHtml(formatDate(p.published))}</span>
      ${cats}
    </div>
    ${reason}
    <p class="paper-abstract">${escapeHtml(p.abstract)}</p>
    ${abstractLong ? `<button class="paper-more">Show more</button>` : ""}
    <div class="paper-actions">
      <a class="btn" href="${escapeHtml(p.pdf_url)}" target="_blank" rel="noopener">PDF</a>
      <a class="btn" href="${escapeHtml(p.entry_url)}" target="_blank" rel="noopener">arXiv</a>
      <button class="btn primary" data-paper='${escapeHtml(JSON.stringify(p))}'>💬 Discuss</button>
    </div>
    <div class="paper-meta" style="font-size:11.5px;color:var(--text-faint)">${escapeHtml(authors)}</div>
  `;

  const moreBtn = card.querySelector(".paper-more");
  if (moreBtn) {
    moreBtn.addEventListener("click", () => {
      const abs = card.querySelector(".paper-abstract");
      const open = abs.classList.toggle("expanded");
      moreBtn.textContent = open ? "Show less" : "Show more";
    });
  }

  card.querySelector(".btn.primary").addEventListener("click", (e) => {
    const paper = JSON.parse(e.currentTarget.dataset.paper);
    openRag(paper);
  });

  return card;
}

/* ============================================================
   Main chat actions
   ============================================================ */
async function askAgent(message) {
  state.busy = true;
  sendBtn.disabled = true;

  const pendingMsg = { role: "assistant", content: "", pending: true };
  state.messages.push(pendingMsg);
  render();
  scrollChatBottom();

  try {
    const data = await api("/api/ask", { message });
    const i = state.messages.indexOf(pendingMsg);
    state.messages[i] = {
      role: "assistant",
      content: data.reply,
      papers: data.papers,
    };
  } catch (err) {
    const i = state.messages.indexOf(pendingMsg);
    state.messages[i] = { role: "assistant", error: true, content: `Search failed: ${err.message}` };
  } finally {
    state.busy = false;
    sendBtn.disabled = false;
    render();
    scrollChatBottom();
  }
}

function sendMessage() {
  const text = input.value.trim();
  if (!text || state.busy) return;
  input.value = "";
  autoGrow(input);
  state.messages.push({ role: "user", content: text });
  render();
  scrollChatBottom();
  askAgent(text);
}

/* ============================================================
   RAG panel
   ============================================================ */
function openRag(paper) {
  state.activePaper = paper.arxiv_id;
  state.paperChats[paper.arxiv_id] = state.paperChats[paper.arxiv_id] || {
    title: paper.title,
    messages: [],
  };
  ragTitle.textContent = paper.title;
  ragPanel.classList.add("open");
  ragPanel.setAttribute("aria-hidden", "false");
  ragOverlay.hidden = false;
  renderRag();
  ensureIndexed(paper.arxiv_id);
}

function closeRag() {
  state.activePaper = null;
  ragPanel.classList.remove("open");
  ragPanel.setAttribute("aria-hidden", "true");
  ragOverlay.hidden = true;
}

async function ensureIndexed(paperId) {
  const already = state.indexedPapers.includes(paperId);
  if (already) {
    ragInput.disabled = false;
    ragSendBtn.disabled = false;
    return;
  }

  // show indexing status
  ragInput.disabled = true;
  ragSendBtn.disabled = true;
  ragStatus.hidden = false;
  ragStatus.innerHTML = `<span class="spinner"></span><span class="rag-status-text">Preparing paper…</span>`;
  const stop = makeStatusRotator(ragStatus.querySelector(".rag-status-text"), [
    "Downloading PDF…",
    "Parsing & chunking sections…",
    "Embedding text…",
    "Building index…",
  ]);
  scrollRagBottom();

  try {
    await api("/api/paper/index", { paper_id: paperId });
    if (!state.indexedPapers.includes(paperId)) state.indexedPapers.push(paperId);
    save();
    if (state.activePaper === paperId) {
      ragStatus.hidden = true;
      ragInput.disabled = false;
      ragSendBtn.disabled = false;
      ragInput.focus();
    }
  } catch (err) {
    stop();
    if (state.activePaper === paperId) {
      ragStatus.hidden = false;
      ragStatus.innerHTML = `<span>⚠️</span><span>Indexing failed: ${escapeHtml(err.message)}</span>`;
      ragInput.disabled = false;
      ragSendBtn.disabled = false;
    }
  }
}

function renderRag() {
  const chat = state.paperChats[state.activePaper];
  if (!chat) return;

  ragMessages.querySelectorAll(".rag-msg").forEach((n) => n.remove());
  ragEmpty.style.display = chat.messages.length ? "none" : "";

  for (const m of chat.messages) {
    const div = document.createElement("div");
    div.className = `rag-msg ${m.role}`;

    if (m.pending) {
      div.innerHTML = `
        <div class="rag-msg-avatar">
          <svg viewBox="0 0 32 32" width="13" height="13" fill="none"><path d="M14 7a7 7 0 1 0 0 14 7 7 0 0 0 0-14zm5 12.5l6 6" stroke="#fff" stroke-width="3.4" stroke-linecap="round"/></svg>
        </div>
        <div class="rag-bubble">
          <div class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>
        </div>`;
    } else if (m.role === "user") {
      div.innerHTML = `<div class="rag-bubble">${escapeHtml(m.content)}</div>`;
    } else {
      const sources = renderSources(m.sources || []);
      div.innerHTML = `
        <div class="rag-msg-avatar">
          <svg viewBox="0 0 32 32" width="13" height="13" fill="none"><path d="M14 7a7 7 0 1 0 0 14 7 7 0 0 0 0-14zm5 12.5l6 6" stroke="#fff" stroke-width="3.4" stroke-linecap="round"/></svg>
        </div>
        <div class="rag-bubble">
          <div style="white-space:pre-wrap;word-break:break-word">${escapeHtml(m.content)}</div>
          ${sources}
        </div>`;
    }
    ragMessages.appendChild(div);
  }
  save();
  scrollRagBottom();
}

function renderSources(sources) {
  if (!sources.length) return "";
  const items = sources
    .map(
      (s, i) => `
      <div class="source-item">
        <div class="source-head">
          <span class="source-heading">${escapeHtml(s.heading || `Excerpt ${i + 1}`)}</span>
          <span class="source-score">
            <span class="score-track"><span class="score-fill" style="width:${Math.max(4, Math.round(s.score * 100))}%"></span></span>
            ${(s.score || 0).toFixed(3)}
          </span>
        </div>
        <div class="source-text">${escapeHtml(s.text)}</div>
      </div>`
    )
    .join("");

  return `
    <div class="sources">
      <button class="sources-toggle">
        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
        ${sources.length} source${sources.length === 1 ? "" : "s"}
        <svg class="chevron" viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M6 9l6 6 6-6"/></svg>
      </button>
      <div class="sources-body">${items}</div>
    </div>`;
}

async function sendRagQuestion() {
  const paperId = state.activePaper;
  const chat = state.paperChats[paperId];
  const q = ragInput.value.trim();
  if (!q || !chat || state.ragBusy) return;

  ragInput.value = "";
  autoGrow(ragInput);

  const pending = { role: "assistant", pending: true };
  chat.messages.push({ role: "user", content: q }, pending);
  state.ragBusy = true;
  ragSendBtn.disabled = true;
  renderRag();

  try {
    const data = await api("/api/paper/chat", { paper_id: paperId, question: q });
    const i = chat.messages.indexOf(pending);
    chat.messages[i] = { role: "assistant", content: data.answer, sources: data.sources };
  } catch (err) {
    // if index vanished on the server, re-index once and retry
    if (err.message.toLowerCase().includes("not indexed")) {
      try {
        await api("/api/paper/index", { paper_id: paperId });
        const data = await api("/api/paper/chat", { paper_id: paperId, question: q });
        const i = chat.messages.indexOf(pending);
        chat.messages[i] = { role: "assistant", content: data.answer, sources: data.sources };
      } catch (err2) {
        const i = chat.messages.indexOf(pending);
        chat.messages[i] = { role: "assistant", content: `⚠️ ${err2.message}` };
      }
    } else {
      const i = chat.messages.indexOf(pending);
      chat.messages[i] = { role: "assistant", content: `⚠️ ${err.message}` };
    }
  } finally {
    state.ragBusy = false;
    ragSendBtn.disabled = false;
    renderRag();
  }
}

/* ============================================================
   Composer plumbing
   ============================================================ */
function autoGrow(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 180) + "px";
}

function wireComposer(textarea, btn, submitFn) {
  btn.addEventListener("click", submitFn);
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitFn();
    }
  });
  textarea.addEventListener("input", () => autoGrow(textarea));
}

/* ============================================================
   Boot
   ============================================================ */
load();
render();

wireComposer(input, sendBtn, sendMessage);
wireComposer(ragInput, ragSendBtn, sendRagQuestion);

$("ragClose").addEventListener("click", closeRag);
ragOverlay.addEventListener("click", closeRag);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && ragPanel.classList.contains("open")) closeRag();
});

// example chips
document.querySelectorAll(".example-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    input.value = chip.dataset.q;
    autoGrow(input);
    input.focus();
    sendMessage();
  });
});

// backend badge
fetch("/api/health")
  .then(() => ($("backendBadge").textContent = "· API online"))
  .catch(() => ($("backendBadge").textContent = "· API offline"));
