import base64
import html
import os
import re
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv("GRAPHLENS_API_BASE_URL", "http://127.0.0.1:8000/api/v1").rstrip("/")
COLLECTION_NAME = os.getenv("GRAPHLENS_COLLECTION_NAME", "graphlens_chunks")


st.set_page_config(page_title="GraphLens", layout="wide", initial_sidebar_state="collapsed")


def init_state() -> None:
    defaults = {
        "page": "home",
        "content_type": None,
        "content_data": None,
        "content_name": None,
        "scope_type": None,
        "scope_id": None,
        "summary": None,
        "key_topics": [],
        "content_meta": {},
        "messages": [],
        "sources": [],
        "citations": [],
        "graph": None,
        "graph_health": None,
        "last_error": None,
        "use_graph": False,
        "menu_open": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()

if st.query_params.get("page") == "home":
    st.session_state.page = "home"
    st.query_params.clear()


def navigate_to(page_name: str) -> None:
    st.session_state.page = page_name
    st.rerun()


def api_url(path: str) -> str:
    return f"{API_BASE_URL}/{path.lstrip('/')}"


def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(api_url(path), json=payload, timeout=180)
    if not response.ok:
        raise RuntimeError(response.json().get("detail", response.text))
    return response.json()


def post_pdf(file_obj: Any) -> dict[str, Any]:
    files = {"file": (file_obj.name, file_obj.getvalue(), "application/pdf")}
    data = {
        "collection_name": COLLECTION_NAME,
        "force_reindex": "false",
    }
    response = requests.post(api_url("/pdf/index"), files=files, data=data, timeout=240)
    if not response.ok:
        raise RuntimeError(response.json().get("detail", response.text))
    return response.json()


def get_json(path: str, params: dict[str, Any] | None = None, timeout: int = 20) -> dict[str, Any]:
    response = requests.get(api_url(path), params=params, timeout=timeout)
    if not response.ok:
        raise RuntimeError(response.json().get("detail", response.text))
    return response.json()


def refresh_graph_health() -> None:
    try:
        st.session_state.graph_health = get_json("/graph/health", timeout=5)
    except Exception as exc:
        st.session_state.graph_health = {"detail": str(exc)}


def clear_session_content() -> None:
    for key in (
        "content_type",
        "content_data",
        "content_name",
        "scope_type",
        "scope_id",
        "summary",
        "content_meta",
        "sources",
        "citations",
        "graph",
        "last_error",
    ):
        st.session_state[key] = None if key not in {"sources", "citations"} else []
    st.session_state.key_topics = []
    st.session_state.messages = []


def apply_ingest_response(content_type: str, content_data: Any, content_name: str, response: dict[str, Any]) -> None:
    st.session_state.content_type = content_type
    st.session_state.content_data = content_data
    st.session_state.content_name = content_name
    st.session_state.scope_type = response["scope_type"]
    st.session_state.scope_id = response["scope_id"]
    st.session_state.summary = response.get("summary")
    st.session_state.key_topics = response.get("key_topics", [])
    st.session_state.content_meta = response
    st.session_state.sources = []
    st.session_state.citations = []
    st.session_state.graph = None
    st.session_state.last_error = None
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Indexing is complete. Ask a question about the selected source.",
        }
    ]
    navigate_to("session")


@st.dialog("Add a Link")
def link_url_dialog() -> None:
    st.caption("Paste a YouTube URL to learn from it.")
    url = st.text_input("URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")
    force_reindex = st.checkbox("Re-index if this source already exists")
    if st.button("Start Learning", type="primary", use_container_width=True):
        if not url.strip():
            st.warning("Enter a YouTube URL first.")
            return
        with st.spinner("Indexing video. Longer videos can take a minute."):
            try:
                response = post_json(
                    "/youtube/index",
                    {
                        "url": url.strip(),
                        "collection_name": COLLECTION_NAME,
                        "force_reindex": force_reindex,
                        "languages": ["en"],
                    },
                )
                apply_ingest_response("youtube", url.strip(), url.strip(), response)
            except Exception as exc:
                st.error(f"Could not index video: {exc}")


@st.dialog("Upload Document")
def upload_pdf_dialog() -> None:
    st.caption("Upload a PDF document. Citations will use page numbers.")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")
    if st.button("Start Learning", type="primary", use_container_width=True):
        if not uploaded_file:
            st.warning("Choose a PDF first.")
            return
        with st.spinner("Indexing PDF. Large files can take a minute or more."):
            try:
                response = post_pdf(uploaded_file)
                apply_ingest_response("pdf", uploaded_file, uploaded_file.name, response)
            except Exception as exc:
                st.error(f"Could not index PDF: {exc}")


def render_styles() -> None:
    st.markdown(
        """
<style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
    }
    .block-container {
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
        padding: 1rem 2rem 1.1rem !important;
        max-width: 100% !important;
    }
    header { visibility: hidden; }
    .stApp {
        background:
            radial-gradient(circle at 76% 82%, rgba(6, 182, 212, 0.14), transparent 22rem),
            radial-gradient(circle at 28% 82%, rgba(37, 99, 235, 0.12), transparent 24rem),
            linear-gradient(180deg, #07111f 0%, #081524 42%, #0c1724 100%);
        color: #f8fafc; font-family: Inter, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #172a43 0%, #10243c 100%) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    section[data-testid="stSidebar"] h3 { font-size: 1.65rem; margin-bottom: 2rem; }
    .header-logo { display: flex; align-items: center; gap: 14px; min-height: 58px; }
    .hamburger { font-size: 26px; color: #7dd3fc; cursor: pointer; margin-right: 10px; }
    .g-box {
        background: linear-gradient(135deg, #00d9ff, #2563eb);
        color: white; width: 54px; height: 54px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 29px; box-shadow: 0 18px 28px rgba(37, 99, 235, 0.34);
    }
    .brand { font-size: 26px; font-weight: 750; color: white; }
    .custom-divider { margin: 8px 0 0; }
    .signin-pill {
        float: right; display: inline-flex; align-items: center; justify-content: center;
        min-width: 132px; height: 54px; border-radius: 999px;
        background: linear-gradient(90deg, #2f80ed, #06b6d4);
        color: white; font-weight: 800; font-size: 1.05rem;
        box-shadow: 0 18px 32px rgba(37, 99, 235, 0.26);
    }
    .home-spacer { height: clamp(7rem, 22vh, 15rem); }
    .home-title { text-align: center; font-size: clamp(2.6rem, 4.2vw, 4.7rem); font-weight: 850; margin: 0; }
    .home-subtitle { text-align: center; color: #9ca3af; font-size: 1.25rem; margin: 0.65rem 0 3rem; }
    .home-card {
        height: 250px; border-radius: 28px; padding: 30px 28px;
        background: rgba(30, 42, 62, 0.8); border: 1px solid rgba(125, 211, 252, 0.18);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 24px 64px rgba(2, 6, 23, 0.22);
        text-align: center; display: flex; flex-direction: column; justify-content: center; gap: 16px;
    }
    .option-icon {
        width: 84px; height: 84px; border-radius: 22px; margin: 0 auto;
        display: flex; align-items: center; justify-content: center;
        background: rgba(14, 116, 144, 0.32); color: #93c5fd; font-size: 2.5rem;
    }
    .option-title { font-size: 1.45rem; font-weight: 800; color: #f8fafc; }
    .option-subtitle { color: #9ca3af; font-weight: 650; }
    button[kind="primary"] {
        background: linear-gradient(90deg, #2563eb, #06b6d4) !important;
        border: 0 !important; color: white !important; border-radius: 999px !important;
        font-weight: 650 !important;
    }
    button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.82) !important;
        border: 1px solid rgba(6, 182, 212, 0.28) !important;
        color: #f8fafc !important; border-radius: 8px !important;
        min-height: 3.2rem;
    }
    button[kind="secondary"]:has(p:nth-of-type(3)),
    button[kind="secondary"]:has(div[data-testid="stMarkdownContainer"] p:nth-of-type(3)) {
        height: 250px !important;
        border-radius: 28px !important;
        background: rgba(30, 42, 62, 0.8) !important;
        border: 1px solid rgba(125, 211, 252, 0.18) !important;
        font-size: 1.22rem !important;
        font-weight: 800 !important;
        white-space: pre-line !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 24px 64px rgba(2, 6, 23, 0.22) !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"] {
        height: auto !important;
        min-height: 3rem !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }
    button[kind="secondary"]:hover {
        border-color: #22d3ee !important; color: #67e8f9 !important;
        box-shadow: 0 0 18px rgba(6, 182, 212, 0.14) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 23, 42, 0.76) !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 8px !important;
    }
    .metric-strip {
        display: flex; flex-wrap: wrap; gap: 10px; margin: 0.4rem 0 1rem;
    }
    .metric-pill, .topic-chip, .source-badge {
        border: 1px solid rgba(148, 163, 184, 0.24); border-radius: 999px;
        padding: 6px 10px; color: #cbd5e1; background: rgba(15, 23, 42, 0.72);
        font-size: 0.82rem;
    }
    .topic-chip { color: #67e8f9; border-color: rgba(6, 182, 212, 0.32); }
    .source-card {
        border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 8px;
        padding: 12px 14px; margin: 9px 0; background: rgba(2, 6, 23, 0.34);
    }
    .source-meta { color: #94a3b8; font-size: 0.82rem; margin-bottom: 5px; }
    .player-shell {
        height: 460px; border-radius: 20px; overflow: hidden;
        background: #020305; border: 1px solid rgba(6, 182, 212, 0.18);
        display: flex; align-items: center; justify-content: center; text-align: center;
    }
    iframe { max-height: 460px; }
    .placeholder-camera { font-size: 4.4rem; opacity: 0.8; }
    .placeholder-title { color: #9ca3af; font-size: 1.25rem; font-weight: 800; margin-top: 0.8rem; }
    .placeholder-subtitle { color: #6b7280; font-weight: 650; margin-top: 0.7rem; max-width: 520px; }
    .panel-card {
        background: rgba(15, 23, 42, 0.76); border: 1px solid rgba(6, 182, 212, 0.18);
        border-radius: 18px; padding: 22px; margin-bottom: 16px;
    }
    .tool-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .tool-button, .tool-button-wide {
        border: 1px solid rgba(6, 182, 212, 0.22); border-radius: 16px;
        background: rgba(30, 41, 59, 0.7); color: #9ca3af;
        padding: 15px; text-align: center; font-weight: 800;
    }
    .tool-button-wide { grid-column: 1 / -1; }
    .chat-title { font-size: 1.45rem; font-weight: 850; margin-bottom: 8px; }
    div[data-testid="stTextInput"] input {
        background: rgba(30, 41, 59, 0.82) !important;
        border: 1px solid rgba(6, 182, 212, 0.28) !important;
        color: #f8fafc !important;
        border-radius: 14px !important;
        min-height: 48px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 0; background: rgba(15, 23, 42, 0.78); border-radius: 18px 18px 0 0; overflow: hidden; }
    .stTabs [data-baseweb="tab"] { flex: 1; height: 58px; color: #9ca3af; font-weight: 800; }
    .stTabs [aria-selected="true"] { color: #22d3ee !important; border-bottom: 3px solid #06b6d4; background: rgba(8, 47, 73, 0.38); }
    .menu-overlay {
        width: 360px; min-height: calc(100vh - 96px); z-index: 999;
        background: linear-gradient(180deg, #172a43 0%, #10243c 100%);
        padding: 32px 28px; box-shadow: 28px 0 80px rgba(0,0,0,0.45);
    }
    .menu-title { display: flex; justify-content: space-between; align-items: center; font-size: 1.7rem; font-weight: 850; margin-bottom: 2.5rem; }
    .menu-item { color: #f8fafc; font-size: 1.18rem; font-weight: 750; margin: 1.6rem 0; }
    .menu-health { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.12); color: #cbd5e1; }
</style>
        """,
        unsafe_allow_html=True,
    )


render_styles()


with st.sidebar:
    st.markdown("### Menu")
    st.button("Home", use_container_width=True, on_click=lambda: navigate_to("home"))
    st.button("New Learning Page", use_container_width=True, on_click=lambda: navigate_to("home"))
    st.markdown("---")
    st.caption(f"API: {API_BASE_URL}")
    if st.button("Check API Health", use_container_width=True):
        try:
            get_json("/graph/health", timeout=8)
            st.session_state.graph_health = {"neo4j": "connected"}
            st.success("Neo4j connected")
        except Exception as exc:
            st.session_state.graph_health = {"detail": str(exc)}
            st.warning("Neo4j unavailable")
    if st.button("Reset session", use_container_width=True):
        clear_session_content()
        navigate_to("home")


def render_header() -> None:
    menu_col, brand_col, head_col2 = st.columns([0.04, 0.74, 0.22])
    with menu_col:
        if st.button("☰", key="menu_toggle", help="Menu"):
            st.session_state.menu_open = not st.session_state.menu_open
            st.rerun()
    with brand_col:
        st.markdown(
            """
            <div class="header-logo">
                <a href="?page=home" target="_self" style="text-decoration:none; display:flex; align-items:center; gap:14px;">
                    <div class="g-box">G</div>
                    <span class="brand">GraphLens</span>
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with head_col2:
        st.markdown('<div class="signin-pill">Sign in</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    render_menu_overlay()


def render_menu_overlay() -> None:
    if not st.session_state.menu_open:
        return

    menu_cols = st.columns([0.22, 0.78])
    with menu_cols[0]:
        st.markdown('<div class="menu-title"><span>Menu</span></div>', unsafe_allow_html=True)
        if st.button("× Close", use_container_width=True):
            st.session_state.menu_open = False
            st.rerun()
        if st.button("Home", key="menu_home", use_container_width=True):
            st.session_state.menu_open = False
            navigate_to("home")
        st.button("Contact Us", key="menu_contact", use_container_width=True, disabled=True)
        if st.button("New Learning Page", key="menu_new_learning", use_container_width=True):
            st.session_state.menu_open = False
            clear_session_content()
            navigate_to("home")
        st.markdown("---")
        st.caption(f"API: {API_BASE_URL}")
        if st.button("Check API Health", key="menu_api_health", use_container_width=True):
            refresh_graph_health()
            if st.session_state.graph_health and st.session_state.graph_health.get("neo4j") == "connected":
                st.success("Neo4j connected")
            else:
                st.warning("Neo4j unavailable")
        if st.session_state.graph_health:
            if st.session_state.graph_health.get("neo4j") == "connected":
                st.caption("Status: Neo4j connected")
            else:
                st.caption("Status: Neo4j unavailable")


def render_home_page() -> None:
    render_header()
    if st.session_state.menu_open:
        return
    st.markdown('<div class="home-spacer"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="home-title">Initialize Knowledge Graph</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="home-subtitle">Link a YouTube lesson or upload a document to begin neural semantic indexing</div>',
        unsafe_allow_html=True,
    )
    spacer_l, col1, col2, spacer_r = st.columns([1.5, 2.3, 2.3, 1.5], gap="large")
    with col1:
        if st.button("🔗\n\nLink URL\n\nConnect web resources", key="home_link_card", use_container_width=True):
            link_url_dialog()
    with col2:
        if st.button("📄\n\nUpload Document\n\nImport PDF content", key="home_doc_card", use_container_width=True):
            upload_pdf_dialog()


def render_source(source: dict[str, Any], index: int) -> None:
    source_url = source.get("source_url") or ""
    page_or_time = source.get("start_seconds")
    is_pdf = bool(source.get("doc_id")) or st.session_state.scope_type == "document"
    if is_pdf and page_or_time is not None:
        citation = f"Page {int(float(page_or_time))}"
    elif source.get("video_id") and page_or_time is not None:
        citation = f"{int(float(page_or_time))}s"
        source_url = f"https://youtube.com/watch?v={source['video_id']}&t={int(float(page_or_time))}"
    else:
        citation = "Source"
    badge = "Graph" if source.get("expanded") else "Vector"
    source_text = html.escape(source.get("text", ""))
    st.markdown(
        f"""
        <div class="source-card">
            <div class="source-meta">#{index} · {citation} · {badge} · similarity {source.get("similarity", 0):.3f}</div>
            <div>{source_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if source_url and not is_pdf:
        st.link_button("Open timestamp", source_url)


def render_answer_markdown(answer: str) -> None:
    escaped = html.escape(answer)
    rendered = re.sub(
        r"\[(\d+)\]",
        lambda match: f'<sup class="source-badge">[{match.group(1)}]</sup>',
        escaped,
    )
    st.markdown(rendered, unsafe_allow_html=True)


def render_content_preview() -> None:
    if st.session_state.content_type == "youtube":
        st.video(st.session_state.content_data)
    elif st.session_state.content_type == "pdf":
        pdf_bytes = st.session_state.content_data.getvalue()
        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="460" type="application/pdf" style="border:none; border-radius:18px;"></iframe>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="player-shell">
                <div>
                    <div class="placeholder-camera">🎥</div>
                    <div class="placeholder-title">Video Player</div>
                    <div class="placeholder-subtitle">Learn from: {html.escape(str(st.session_state.content_name or "selected source"))}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_summary() -> None:
    meta = st.session_state.content_meta or {}
    st.markdown('<div class="metric-strip">', unsafe_allow_html=True)
    if meta.get("estimated_duration"):
        st.markdown(f'<span class="metric-pill">Duration {meta["estimated_duration"]}</span>', unsafe_allow_html=True)
    if meta.get("page_count"):
        st.markdown(f'<span class="metric-pill">{meta["page_count"]} pages</span>', unsafe_allow_html=True)
    if meta.get("chunks_indexed") is not None:
        st.markdown(f'<span class="metric-pill">{meta["chunks_indexed"]} chunks indexed</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if st.session_state.summary:
        st.write(st.session_state.summary)
    if st.session_state.key_topics:
        chips = " ".join(f'<span class="topic-chip">{html.escape(str(topic))}</span>' for topic in st.session_state.key_topics)
        st.markdown(chips, unsafe_allow_html=True)


def submit_query(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        graph_available = st.session_state.graph_health and st.session_state.graph_health.get("neo4j") == "connected"
        response = post_json(
            "/query",
            {
                "question": prompt,
                "scope_type": st.session_state.scope_type,
                "scope_id": st.session_state.scope_id,
                "collection_name": COLLECTION_NAME,
                "use_graph": bool(st.session_state.use_graph and graph_available),
            },
        )
        st.session_state.sources = response.get("sources", [])
        st.session_state.citations = response.get("citations", [])
        graph_note = ""
        graph_expansion = response.get("graph_expansion") or {}
        if graph_expansion.get("expanded_chunks"):
            graph_note = f"\n\nGraph found {graph_expansion['expanded_chunks']} related sections."
        elif graph_expansion.get("method") == "fallback_plain_rag":
            graph_note = "\n\nGraphRAG is unavailable, so this used plain RAG."
        if response.get("refused"):
            content = response.get("reason") or "I do not have enough relevant evidence in the selected content."
        else:
            content = response.get("answer") or "Here are the most relevant grounded sources from the selected content."
            content += graph_note
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": content,
                "refused": response.get("refused", False),
                "best_similarity": response.get("best_similarity"),
                "model": response.get("model"),
                "citations": response.get("citations", []),
            }
        )
    except Exception as exc:
        st.session_state.messages.append({"role": "assistant", "content": f"Query failed: {exc}", "refused": True})


def load_graph() -> None:
    if not st.session_state.sources:
        st.warning("Ask a question first so GraphLens has source context.")
        return
    text = st.session_state.sources[0].get("text", "")
    concept = " ".join(text.split()[:3]).strip(".,:;") or st.session_state.scope_id
    try:
        st.session_state.graph = get_json("/graph/concept", {"concept": concept, "scope_id": st.session_state.scope_id})
    except Exception as exc:
        st.session_state.graph = {"error": str(exc)}


def render_learning_session() -> None:
    render_header()
    refresh_graph_health()
    if not st.session_state.scope_id:
        st.warning("Index a source first.")
        if st.button("Go home"):
            navigate_to("home")
        return

    left_col, right_col = st.columns([0.58, 0.42], gap="large")
    with left_col:
        render_content_preview()
        st.write("")
        tab1, tab2 = st.tabs(["Summary", "Sources"])
        with tab1:
            with st.container(height=245, border=True):
                render_summary()
        with tab2:
            with st.container(height=245, border=True):
                if st.session_state.sources:
                    citation_indices = [
                        idx for idx in st.session_state.citations
                        if isinstance(idx, int) and 0 <= idx < len(st.session_state.sources)
                    ]
                    indices = citation_indices or list(range(len(st.session_state.sources)))
                    for source_index in indices:
                        render_source(st.session_state.sources[source_index], source_index + 1)
                else:
                    st.caption("Sources will appear after a query.")

    with right_col:
        st.markdown(
            """
            <div class="panel-card">
                <h3 style="margin-top:0;">Intelligence Tools</h3>
                <div class="tool-grid">
                    <div class="tool-button">⌘ &nbsp; View Graph</div>
                    <div class="tool-button">▥ &nbsp; Reliability</div>
                    <div class="tool-button">? &nbsp; Quiz</div>
                    <div class="tool-button">□ &nbsp; Notes</div>
                    <div class="tool-button-wide">▱ &nbsp; Flashcards</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chat-title">Ask Me Anything</div>', unsafe_allow_html=True)
        graph_available = st.session_state.graph_health and st.session_state.graph_health.get("neo4j") == "connected"
        if not graph_available:
            st.session_state.use_graph = False
        st.toggle("Use GraphRAG", key="use_graph", disabled=not bool(graph_available))
        if not graph_available:
            st.caption("Neo4j is not connected. Plain RAG is active.")

        chat_container = st.container(height=255, border=True)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant" and not msg.get("refused"):
                        render_answer_markdown(msg["content"])
                    else:
                        st.write(msg["content"])
                    if msg.get("best_similarity") is not None and not msg.get("refused"):
                        st.caption(f"Best similarity: {msg['best_similarity']:.3f}")
                    if msg.get("model") and not msg.get("refused"):
                        st.caption(f"Model: {msg['model']}")

        with st.form("query_form", clear_on_submit=True):
            prompt = st.text_input(
                "Ask about the indexed source",
                placeholder="Query the grounded graph...",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Send", type="primary", use_container_width=True)
            if submitted and prompt.strip():
                with st.spinner("Retrieving grounded evidence..."):
                    submit_query(prompt.strip())
                st.rerun()


if st.session_state.page == "home":
    render_home_page()
elif st.session_state.page == "session":
    render_learning_session()
