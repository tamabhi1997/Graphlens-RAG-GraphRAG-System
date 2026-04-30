import base64
import hashlib
import html
import os
import re
from typing import Any

import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    from pyvis.network import Network
except Exception:
    Network = None


API_BASE_URL = os.getenv("GRAPHLENS_API_BASE_URL", "http://127.0.0.1:8000/api/v1").rstrip("/")
COLLECTION_NAME = os.getenv("GRAPHLENS_COLLECTION_NAME", "graphlens_chunks")
MIT_COURSE_ID = "mit_6s191"
MIT_COURSE_TITLE = "MIT 6.S191 Introduction to Deep Learning"
MIT_COURSE_FORCE_REINDEX = os.getenv("GRAPHLENS_COURSE_FORCE_REINDEX", "false").lower() == "true"

MIT_COURSE_VIDEOS = [
    {
        "lecture": "Lecture 1",
        "title": "Introduction to Deep Learning",
        "duration": "1h 9m",
        "url": "https://www.youtube.com/watch?v=ErnWZxJovaM",
        "description": "Foundations of deep learning, neural networks, gradient descent, and backpropagation.",
    },
    {
        "lecture": "Lecture 2",
        "title": "Recurrent Neural Networks, Transformers, and Attention",
        "duration": "52m",
        "url": "https://www.youtube.com/watch?v=dqoEU9Ac3ek",
        "description": "Sequence modeling with recurrent networks, attention, and transformer architectures.",
    },
    {
        "lecture": "Lecture 3",
        "title": "Convolutional Neural Networks",
        "duration": "55m",
        "url": "https://www.youtube.com/watch?v=2xqkSUhmmXU",
        "description": "Computer vision foundations, convolutional filters, and deep image models.",
    },
    {
        "lecture": "Lecture 4",
        "title": "Deep Generative Modeling",
        "duration": "56m",
        "url": "https://www.youtube.com/watch?v=Dmm4UG-6jxA",
        "description": "Autoencoders, VAEs, GANs, diffusion ideas, and latent representations.",
    },
    {
        "lecture": "Lecture 5",
        "title": "Reinforcement Learning",
        "duration": "55m",
        "url": "https://www.youtube.com/watch?v=8JVRbHAVCws",
        "description": "Deep reinforcement learning, agents, rewards, policies, and environment feedback.",
    },
    {
        "lecture": "Lecture 6",
        "title": "Language Models and New Frontiers",
        "duration": "48m",
        "url": "https://www.youtube.com/watch?v=N1fbskTpwZ0",
        "description": "Modern language models, frontier AI capabilities, limitations, and active research directions.",
    },
]


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
        "graph_concept": None,
        "graph_health": None,
        "last_question": None,
        "last_error": None,
        "use_graph": False,
        "menu_open": False,
        "course_ingest_results": {},
        "course_video_titles": {},
        "pending_query": None,
        "pending_course_video_id": None,
        "signed_in": False,
        "user_name": None,
        "user_email": None,
        "user_first_name": None,
        "user_last_name": None,
        "user_password_hash": None,
        "open_signup": False,
        "open_signin": False,
        "open_link_dialog": False,
        "open_pdf_dialog": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()

page_param = st.query_params.get("page")
if page_param in {"home", "course"}:
    st.session_state.page = page_param
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


def youtube_video_id(url: str) -> str:
    patterns = [
        r"[?&]v=([^&]+)",
        r"youtu\.be/([^?&/]+)",
        r"youtube\.com/embed/([^?&/]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url.rsplit("/", 1)[-1]


def youtube_thumbnail_url(url: str) -> str:
    return f"https://img.youtube.com/vi/{youtube_video_id(url)}/hqdefault.jpg"


def course_video_by_id(video_id: str) -> dict[str, str] | None:
    return next((video for video in MIT_COURSE_VIDEOS if youtube_video_id(video["url"]) == video_id), None)


def format_timestamp(seconds: Any) -> str:
    total_seconds = int(float(seconds or 0))
    minutes, second = divmod(total_seconds, 60)
    hour, minute = divmod(minutes, 60)
    if hour:
        return f"{hour}:{minute:02d}:{second:02d}"
    return f"{minute}:{second:02d}"


def post_pdf(file_obj: Any) -> dict[str, Any]:
    files = {"file": (file_obj.name, file_obj.getvalue(), "application/pdf")}
    data = {
        "collection_name": COLLECTION_NAME,
        "force_reindex": "true",
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
        "graph_concept",
        "last_question",
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
    st.session_state.graph_concept = None
    st.session_state.last_question = None
    st.session_state.last_error = None
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Quench your curiosity.",
        }
    ]
    navigate_to("session")


def remember_course_video(video: dict[str, str], response: dict[str, Any] | None = None) -> None:
    video_id = response.get("video_id") if response else youtube_video_id(video["url"])
    st.session_state.course_video_titles[video_id] = {
        "title": video["title"],
        "lecture": video["lecture"],
        "url": video["url"],
    }
    if response:
        st.session_state.course_ingest_results[video_id] = response


def ingest_course_video(video: dict[str, str], force_reindex: bool = False) -> dict[str, Any]:
    response = post_json(
        "/youtube/index",
        {
            "url": video["url"],
            "course_id": MIT_COURSE_ID,
            "collection_name": COLLECTION_NAME,
            "force_reindex": force_reindex,
            "languages": ["en"],
        },
    )
    remember_course_video(video, response)
    return response


def load_course_video(video: dict[str, str]) -> None:
    video_id = youtube_video_id(video["url"])
    response = st.session_state.course_ingest_results.get(video_id)
    if not response:
        try:
            response = ingest_course_video(video, MIT_COURSE_FORCE_REINDEX)
        except Exception as exc:
            st.error(f"Could not prepare this lecture: {exc}")
            return
    remember_course_video(video, response)
    st.session_state.content_type = "course"
    st.session_state.content_data = video["url"]
    st.session_state.content_name = f"{video['lecture']}: {video['title']}"
    st.session_state.scope_type = "course"
    st.session_state.scope_id = MIT_COURSE_ID
    st.session_state.summary = response.get("summary") or video["description"]
    st.session_state.key_topics = response.get("key_topics", [])
    st.session_state.content_meta = {
        **response,
        "course_id": MIT_COURSE_ID,
        "course_title": MIT_COURSE_TITLE,
        "lecture": video["lecture"],
        "lecture_title": video["title"],
    }
    st.session_state.sources = []
    st.session_state.citations = []
    st.session_state.graph = None
    st.session_state.graph_concept = None
    st.session_state.last_question = None
    st.session_state.last_error = None
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Quench your curiosity.",
        }
    ]
    navigate_to("session")


@st.dialog("Add a Link")
def link_url_dialog() -> None:
    st.caption("Paste a YouTube URL to learn from it.")
    url = st.text_input("URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")
    if st.button("Start Learning", type="primary", use_container_width=True):
        if not url.strip():
            st.warning("Enter a YouTube URL first.")
            return
        with st.spinner("Loading..."):
            try:
                response = post_json(
                    "/youtube/index",
                    {
                        "url": url.strip(),
                        "collection_name": COLLECTION_NAME,
                        "force_reindex": True,
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
        with st.spinner("Loading... Indexing PDF — extracting text, generating embeddings and building knowledge graph. This takes 1-2 minutes for large documents..."):
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
        min-height: 100vh !important;
    }
    .block-container {
        min-height: 100vh !important;
        padding: 1.2rem 2rem 2.5rem !important;
        max-width: 100% !important;
    }
    header { visibility: hidden; }
    /* Sticky app bar — only the row that contains the GraphLens logo (not drawer/column splits). */
    div[data-testid="stHorizontalBlock"]:has(div.header-logo) {
        position: sticky !important;
        top: 0 !important;
        z-index: 1005 !important;
        margin-left: calc(-2rem + 6px);
        margin-right: calc(-2rem + 6px);
        padding: 10px calc(2rem - 6px) 12px calc(2rem - 6px);
        margin-bottom: 0 !important;
        background: linear-gradient(180deg, rgba(2, 8, 23, 0.97) 0%, rgba(7, 17, 31, 0.95) 100%) !important;
        backdrop-filter: blur(14px);
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        box-shadow: 0 8px 28px rgba(2, 6, 23, 0.35);
    }
    @media (max-width: 900px) {
        div[data-testid="stHorizontalBlock"]:has(div.header-logo) {
            margin-left: calc(-1rem + 6px);
            margin-right: calc(-1rem + 6px);
            padding-left: calc(1rem - 6px);
            padding-right: calc(1rem - 6px);
        }
    }
    section[data-testid="stMain"] {
        scroll-padding-top: 5.5rem;
    }
    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(14, 165, 233, 0.16), transparent 24rem),
            radial-gradient(circle at 85% 12%, rgba(59, 130, 246, 0.12), transparent 24rem),
            linear-gradient(180deg, #020817 0%, #07111f 48%, #08111d 100%);
        color: #f8fafc; font-family: Inter, sans-serif;
    }
    /* Hide built-in Streamlit sidebar; navigation uses the hamburger drawer column instead. */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    .header-logo { display: flex; align-items: center; gap: 12px; min-height: 56px; }
    .hamburger { font-size: 26px; color: #7dd3fc; cursor: pointer; margin-right: 10px; }
    .g-box {
        background: linear-gradient(135deg, #38bdf8, #2563eb);
        color: white; width: 48px; height: 48px; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 850; font-size: 26px; box-shadow: 0 14px 34px rgba(37, 99, 235, 0.26);
    }
    .brand { font-size: 25px; font-weight: 800; color: white; }
    .custom-divider { display: none; }
    .signin-pill {
        float: right; display: inline-flex; align-items: center; justify-content: center;
        min-width: 116px; height: 44px; border-radius: 999px; padding: 0 18px;
        background: rgba(15, 23, 42, 0.82); border: 1px solid rgba(56, 189, 248, 0.32);
        color: #e0f2fe; font-weight: 800; font-size: 0.96rem;
        box-shadow: 0 14px 30px rgba(2, 6, 23, 0.22);
    }
    .user-pill {
        float: right; display: inline-flex; flex-direction: column; justify-content: center;
        min-width: 150px; height: 44px; border-radius: 999px; padding: 0 18px;
        background: rgba(15, 23, 42, 0.82); border: 1px solid rgba(56, 189, 248, 0.32);
        color: #e0f2fe; box-shadow: 0 14px 30px rgba(2, 6, 23, 0.22);
    }
    .user-pill-name { font-weight: 850; font-size: 0.9rem; line-height: 1.1; }
    .user-pill-email { color: #94a3b8; font-size: 0.72rem; line-height: 1.1; }
    .home-spacer { height: clamp(1.5rem, 5vh, 3.5rem); }
    .home-title {
        text-align: center; font-size: clamp(2.5rem, 4vw, 4.4rem); font-weight: 850;
        margin: 0 auto; max-width: 940px; line-height: 1.03;
    }
    .home-subtitle {
        text-align: center; color: #a5b4fc; font-size: 1.12rem; margin: 0.8rem auto 2.1rem;
        max-width: 720px; line-height: 1.5;
    }
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
    .section-kicker {
        color: #38bdf8; font-size: 0.78rem; font-weight: 850;
        letter-spacing: 0.08em; text-transform: uppercase;
    }
    .section-heading {
        color: #f8fafc; font-size: clamp(1.65rem, 2.4vw, 2.35rem);
        font-weight: 850; margin: 0.25rem 0 0.35rem;
    }
    .section-copy {
        color: #94a3b8; font-size: 1rem; max-width: 680px; margin-bottom: 1rem;
    }
    .course-feature {
        max-width: 1180px; margin: 2rem auto 0; border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 8px; background: rgba(15, 23, 42, 0.62); padding: 24px;
        display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr); gap: 24px; align-items: center;
        box-shadow: 0 26px 80px rgba(2, 6, 23, 0.24);
    }
    .course-title {
        color: #f8fafc; font-size: clamp(1.6rem, 2.2vw, 2.1rem); font-weight: 850;
        margin: 0.2rem 0 0.55rem; display: inline-block; text-decoration: none;
        transition: color 0.2s ease, text-shadow 0.2s ease;
    }
    .course-title:hover {
        color: #67e8f9; text-shadow: 0 0 16px rgba(34, 211, 238, 0.3);
    }
    .course-copy { color: #cbd5e1; line-height: 1.6; max-width: 720px; }
    .course-stat-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 1rem; }
    .course-stat {
        border: 1px solid rgba(148, 163, 184, 0.22); border-radius: 999px; color: #cbd5e1;
        padding: 6px 10px; background: rgba(2, 6, 23, 0.28); font-size: 0.82rem; font-weight: 750;
    }
    .course-page-head {
        max-width: 1180px; margin: 1.5rem auto 1rem; display: flex; justify-content: space-between;
        gap: 16px; align-items: flex-end;
    }
    .st-key-course_prepare button[kind="secondary"] {
        background: rgba(15, 23, 42, 0.48) !important;
        border: 0 !important;
        color: #67e8f9 !important;
        font-weight: 800 !important;
    }
    .st-key-course_prepare button[kind="secondary"]:hover {
        background: rgba(8, 47, 73, 0.32) !important;
        color: #a5f3fc !important;
    }
    .lecture-card {
        border: 1px solid rgba(56, 189, 248, 0.18); border-radius: 8px; overflow: hidden;
        background: rgba(15, 23, 42, 0.72); min-height: 402px;
        display: flex; flex-direction: column; height: 100%;
        box-shadow: 0 18px 48px rgba(2, 6, 23, 0.2);
    }
    .lecture-thumb {
        width: 100%; aspect-ratio: 16 / 9; object-fit: cover; display: block;
        border-bottom: 1px solid rgba(148, 163, 184, 0.16);
    }
    .lecture-body { padding: 14px; display: flex; flex-direction: column; flex: 1; }
    .lecture-kicker { color: #67e8f9; font-size: 0.78rem; font-weight: 850; text-transform: uppercase; }
    .lecture-title { color: #f8fafc; font-size: 1.03rem; line-height: 1.28; font-weight: 850; margin: 7px 0; min-height: 2.65rem; }
    .lecture-desc { color: #9ca3af; font-size: 0.88rem; line-height: 1.45; min-height: 76px; }
    .lecture-summary {
        color: #cbd5e1; font-size: 0.84rem; line-height: 1.45; margin-top: auto;
        border-top: 1px solid rgba(148, 163, 184, 0.14); padding-top: 10px;
        min-height: 4.8rem;
    }
    .course-thumb-stack { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .course-thumb-stack img {
        width: 100%; aspect-ratio: 16 / 9; object-fit: cover; border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.18);
    }
    button[kind="primary"] {
        background: linear-gradient(90deg, #2563eb, #06b6d4) !important;
        border: 0 !important; color: white !important; border-radius: 999px !important;
        font-weight: 750 !important; min-height: 2.9rem;
    }
    button[kind="secondary"] {
        background: rgba(15, 23, 42, 0.68) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        color: #f8fafc !important; border-radius: 8px !important;
        min-height: 3.2rem;
    }
    button[kind="secondary"]:has(p:nth-of-type(3)),
    button[kind="secondary"]:has(div[data-testid="stMarkdownContainer"] p:nth-of-type(3)) {
        height: 184px !important;
        border-radius: 8px !important;
        background: rgba(15, 23, 42, 0.68) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        font-size: 1.08rem !important;
        font-weight: 800 !important;
        white-space: pre-line !important;
        box-shadow: 0 18px 48px rgba(2, 6, 23, 0.2) !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"] {
        height: auto !important;
        min-height: 3rem !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }
    button[kind="secondary"]:hover {
        border-color: rgba(56, 189, 248, 0.5) !important; color: #e0f2fe !important;
        box-shadow: 0 18px 50px rgba(8, 47, 73, 0.22) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 23, 42, 0.76) !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 8px !important;
    }
    .metric-strip {
        display: flex; flex-wrap: wrap; gap: 10px; margin: 0.4rem 0 1rem;
    }
    .metric-pill, .source-badge {
        border: 1px solid rgba(148, 163, 184, 0.24); border-radius: 999px;
        padding: 6px 10px; color: #cbd5e1; background: rgba(15, 23, 42, 0.72);
        font-size: 0.82rem;
    }
    .summary-text {
        color: #e2e8f0; line-height: 1.6; margin: 0.4rem 0 1rem;
        white-space: pre-wrap;
    }
    .topics-list { display: grid; gap: 10px; margin-top: 0.85rem; }
    .topic-card {
        border: 1px solid rgba(6, 182, 212, 0.22); border-radius: 8px;
        background: rgba(2, 6, 23, 0.28); padding: 11px 12px;
    }
    .topic-name { color: #f8fafc; font-weight: 850; margin-bottom: 4px; }
    .topic-desc { color: #9ca3af; font-size: 0.88rem; line-height: 1.45; }
    .topic-chip {
        display: inline-flex; border: 1px solid rgba(6, 182, 212, 0.32); border-radius: 999px;
        padding: 6px 10px; color: #67e8f9; background: rgba(15, 23, 42, 0.72);
        font-size: 0.82rem; margin: 0 6px 6px 0;
    }
    .graph-shell {
        height: 520px; border: 1px solid rgba(56, 189, 248, 0.16); border-radius: 8px;
        overflow: hidden; background: rgba(2, 6, 23, 0.62);
    }
    .graph-fallback {
        min-height: 220px; display: flex; flex-wrap: wrap; align-items: center; justify-content: center;
        gap: 12px; padding: 18px; background: rgba(2, 6, 23, 0.42); border-radius: 8px;
    }
    .graph-node-pill {
        display: inline-flex; align-items: center; justify-content: center;
        min-width: 92px; max-width: 210px; min-height: 42px; padding: 8px 12px;
        border-radius: 999px; border: 1px solid rgba(56, 189, 248, 0.28);
        background: rgba(15, 23, 42, 0.88); color: #e2e8f0; font-weight: 800;
        text-align: center; font-size: 0.83rem;
    }
    .graph-node-center {
        min-width: 126px; min-height: 54px; background: rgba(6, 182, 212, 0.22);
        border-color: rgba(34, 211, 238, 0.72); color: #cffafe;
    }
    .chunk-link-card {
        border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 8px;
        padding: 10px 12px; margin: 8px 0; background: rgba(2, 6, 23, 0.3);
    }
    .graph-empty {
        border: 1px dashed rgba(148, 163, 184, 0.28); border-radius: 8px;
        min-height: 180px; display: flex; align-items: center; justify-content: center;
        color: #94a3b8; background: rgba(2, 6, 23, 0.26); text-align: center; padding: 18px;
    }
    .confidence-badge {
        display: inline-flex; align-items: center; gap: 7px; border-radius: 999px;
        padding: 5px 10px; margin: 0.35rem 0 0.15rem; font-size: 0.78rem; font-weight: 850;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .confidence-dot { width: 8px; height: 8px; border-radius: 999px; display: inline-block; }
    .confidence-high { color: #bbf7d0; background: rgba(22, 101, 52, 0.24); border-color: rgba(34, 197, 94, 0.28); }
    .confidence-high .confidence-dot { background: #22c55e; }
    .confidence-medium { color: #fef3c7; background: rgba(133, 77, 14, 0.25); border-color: rgba(234, 179, 8, 0.32); }
    .confidence-medium .confidence-dot { background: #eab308; }
    .confidence-low { color: #fed7aa; background: rgba(154, 52, 18, 0.25); border-color: rgba(249, 115, 22, 0.34); }
    .confidence-low .confidence-dot { background: #f97316; }
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
    .chat-title { font-size: 1.45rem; font-weight: 850; margin-bottom: 8px; }
    /* Chat composer: single row, arrow submit flush with input */
    div[data-testid="stForm"]:has(.st-key-query_send_arrow) {
        padding: 10px 12px 12px !important;
        margin: 0 !important;
        border-top: 1px solid rgba(148, 163, 184, 0.16) !important;
        background: rgba(10, 18, 32, 0.98) !important;
    }
    div[data-testid="stForm"]:has(.st-key-query_send_arrow) div[data-testid="stHorizontalBlock"] {
        gap: 0 !important;
        align-items: flex-end !important;
    }
    div[data-testid="stForm"]:has(.st-key-query_send_arrow) div[data-testid="stTextInput"] input {
        border-radius: 14px 0 0 14px !important;
        border-right: none !important;
    }
    div[data-testid="stForm"]:has(.st-key-query_send_arrow) div[data-testid="stTextInput"] {
        margin-bottom: 0 !important;
    }
    .st-key-query_send_arrow button[kind="primary"] {
        border-radius: 0 14px 14px 0 !important;
        min-width: 52px !important;
        min-height: 48px !important;
        height: 48px !important;
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        padding: 0 10px !important;
    }
    div[data-testid="stForm"] {
        padding: 18px 18px 20px !important;
        margin-top: 1rem !important;
        margin-bottom: 1.25rem !important;
    }
    div[data-testid="stForm"] div[data-testid="stTextInput"] {
        margin-bottom: 0.75rem !important;
    }
    div[data-testid="stTextInput"] input {
        background: rgba(30, 41, 59, 0.82) !important;
        border: 1px solid rgba(6, 182, 212, 0.28) !important;
        color: #f8fafc !important;
        border-radius: 14px !important;
        min-height: 48px;
        padding: 0 18px !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 0; background: rgba(15, 23, 42, 0.68); border-radius: 8px 8px 0 0; overflow: hidden; }
    .stTabs [data-baseweb="tab"] { flex: 1; height: 58px; color: #9ca3af; font-weight: 800; }
    .stTabs [aria-selected="true"] { color: #22d3ee !important; border-bottom: 3px solid #06b6d4; background: rgba(8, 47, 73, 0.38); }
    .auth-switch {
        text-align: center; color: #64748b; font-size: 0.82rem;
        margin: 10px 0 4px; letter-spacing: 0.01em;
    }
    /* make the header sign-in button compact and pill-shaped */
    div[data-testid="column"]:last-child button[kind="secondary"] {
        min-height: 2.4rem !important;
        height: 2.4rem !important;
        font-size: 0.88rem !important;
        border-radius: 999px !important;
        padding: 0 14px !important;
    }
    /* "Create an account" / "Sign in instead" buttons inside dialogs */
    div[data-testid="stDialog"] button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid rgba(56, 189, 248, 0.22) !important;
        color: #7dd3fc !important;
        font-size: 0.85rem !important;
        min-height: 2.4rem !important;
        height: 2.4rem !important;
        border-radius: 999px !important;
    }
    div[data-testid="stDialog"] button[kind="secondary"]:hover {
        background: rgba(56, 189, 248, 0.08) !important;
        border-color: rgba(56, 189, 248, 0.44) !important;
        color: #bae6fd !important;
    }
    @media (max-width: 900px) {
        .block-container { padding: 1rem 1rem 2rem !important; }
        .course-feature { grid-template-columns: 1fr; }
        .course-page-head { display: block; }
        .home-title { font-size: 2.45rem; }
    }
</style>
        """,
        unsafe_allow_html=True,
    )


render_styles()


def render_nav_drawer() -> None:
    with st.container(border=True):
        head_a, head_b = st.columns([0.82, 0.18])
        with head_a:
            st.markdown("### Menu")
        with head_b:
            if st.button("✕", key="nav_drawer_close", help="Close menu"):
                st.session_state.menu_open = False
                st.rerun()
        if st.button("Home", key="nav_drawer_home", use_container_width=True):
            st.session_state.menu_open = False
            navigate_to("home")
        if st.button("MIT Course", key="nav_drawer_course", use_container_width=True):
            st.session_state.menu_open = False
            navigate_to("course")
        st.caption("Add content")
        if st.button("🔗 Link URL", key="nav_drawer_link_url", use_container_width=True):
            st.session_state.open_link_dialog = True
            st.rerun()
        if st.button("📄 Upload PDF", key="nav_drawer_upload_pdf", use_container_width=True):
            st.session_state.open_pdf_dialog = True
            st.rerun()
        st.markdown("---")
        st.caption(f"API: {API_BASE_URL}")
        if st.button("Check API Health", key="nav_drawer_health", use_container_width=True):
            try:
                get_json("/graph/health", timeout=8)
                st.session_state.graph_health = {"neo4j": "connected"}
                st.success("Neo4j connected")
            except Exception as exc:
                st.session_state.graph_health = {"detail": str(exc)}
                st.warning("Neo4j unavailable")
        if st.button("Reset session", key="nav_drawer_reset", use_container_width=True):
            clear_session_content()
            st.session_state.menu_open = False
            navigate_to("home")
        if st.session_state.graph_health:
            if st.session_state.graph_health.get("neo4j") == "connected":
                st.caption("Status: Neo4j connected")
            else:
                st.caption("Status: Neo4j unavailable")


def render_header() -> None:
    menu_col, brand_col, head_col2 = st.columns([0.04, 0.82, 0.14])
    with menu_col:
        if st.button("☰", key="menu_toggle", help="Toggle menu"):
            st.session_state.menu_open = not bool(st.session_state.get("menu_open"))
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
        if st.session_state.signed_in:
            user_name = html.escape(str(st.session_state.user_name or "User"))
            user_email = html.escape(str(st.session_state.user_email or ""))
            st.markdown(
                f"""
                <div class="user-pill">
                    <span class="user-pill-name">{user_name}</span>
                    <span class="user-pill-email">{user_email}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Sign out", key="header_signout", use_container_width=True):
                st.session_state.signed_in = False
                st.session_state.user_name = None
                st.session_state.user_email = None
                st.session_state.user_first_name = None
                st.session_state.user_last_name = None
                st.session_state.user_password_hash = None
                st.rerun()
        elif st.button("Sign in", key="header_signin", use_container_width=True):
            sign_in_dialog()
    if st.session_state.get("open_signup"):
        st.session_state.open_signup = False
        sign_up_dialog()
    if st.session_state.get("open_signin"):
        st.session_state.open_signin = False
        sign_in_dialog()
    if st.session_state.get("open_link_dialog"):
        st.session_state.open_link_dialog = False
        link_url_dialog()
    if st.session_state.get("open_pdf_dialog"):
        st.session_state.open_pdf_dialog = False
        upload_pdf_dialog()


@st.dialog("Sign in")
def sign_in_dialog() -> None:
    st.caption("Welcome back — enter your details to continue.")
    with st.form("sign_in_form", clear_on_submit=False):
        name = st.text_input("Name", placeholder="Jane Doe")
        email = st.text_input("Email", placeholder="you@example.com")
        submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            if not name.strip() or not email.strip():
                st.warning("Enter both name and email.")
                return
            if "@" not in email:
                st.warning("Enter a valid email address.")
                return
            st.session_state.signed_in = True
            st.session_state.user_name = name.strip()
            st.session_state.user_email = email.strip()
            st.rerun()
    st.markdown(
        '<p class="auth-switch">New here?</p>',
        unsafe_allow_html=True,
    )
    if st.button("Create an account →", key="signin_to_signup", use_container_width=True):
        st.session_state.open_signup = True
        st.rerun()


@st.dialog("Create Account")
def sign_up_dialog() -> None:
    st.caption("Join GraphLens — takes less than a minute.")
    with st.form("sign_up_form", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            first_name = st.text_input("First name", placeholder="Jane")
        with col_b:
            last_name = st.text_input("Last name", placeholder="Doe")
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", placeholder="Min. 8 characters", type="password")
        retype = st.text_input("Retype password", placeholder="Re-enter password", type="password")
        submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
        if submitted:
            errors = []
            if not first_name.strip():
                errors.append("First name is required.")
            if not last_name.strip():
                errors.append("Last name is required.")
            if not email.strip() or "@" not in email:
                errors.append("A valid email address is required.")
            if len(password) < 8:
                errors.append("Password must be at least 8 characters.")
            if password != retype:
                errors.append("Passwords do not match.")
            if errors:
                for err in errors:
                    st.warning(err)
            else:
                pw_hash = hashlib.sha256(password.encode()).hexdigest()
                st.session_state.signed_in = True
                st.session_state.user_first_name = first_name.strip()
                st.session_state.user_last_name = last_name.strip()
                st.session_state.user_name = f"{first_name.strip()} {last_name.strip()}"
                st.session_state.user_email = email.strip()
                st.session_state.user_password_hash = pw_hash
                st.rerun()
    st.markdown(
        '<p class="auth-switch">Already have an account?</p>',
        unsafe_allow_html=True,
    )
    if st.button("Sign in instead →", key="signup_to_signin", use_container_width=True):
        st.session_state.open_signin = True
        st.rerun()





def render_course_feature() -> None:
    preview_images = "".join(
        f'<img src="{html.escape(youtube_thumbnail_url(video["url"]))}" alt="">'
        for video in MIT_COURSE_VIDEOS[:4]
    )
    st.markdown(
        f"""
        <div class="course-feature">
            <div>
                <div class="section-kicker">Course Playlist</div>
                <a class="course-title" href="?page=course" target="_self">{html.escape(MIT_COURSE_TITLE)}</a>
                <div class="course-copy">
                    Browse the MIT 6.S191 lecture playlist, prepare the videos with a shared course scope,
                    and ask questions across the full course instead of one video at a time.
                </div>
                <div class="course-stat-row">
                    <span class="course-stat">{len(MIT_COURSE_VIDEOS)} lectures</span>
                    <span class="course-stat">Course scope: {html.escape(MIT_COURSE_ID)}</span>
                    <span class="course-stat">Cross-lecture retrieval</span>
                </div>
            </div>
            <div class="course-thumb-stack">{preview_images}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
def render_course_page() -> None:
    render_header()

    st.markdown('<div class="section-kicker">MIT Course Tile</div>', unsafe_allow_html=True)
    head_col, action_col = st.columns([0.72, 0.28], gap="large")
    with head_col:
        st.markdown(f'<h1 class="section-heading">{html.escape(MIT_COURSE_TITLE)}</h1>', unsafe_allow_html=True)
    with action_col:
        prepare_clicked = st.button(
            "Prepare full course",
            key="course_prepare",
            type="secondary",
            use_container_width=True,
        )
    st.markdown(
        f"""
        <div class="section-copy">
            Select a lecture to watch and query with course-wide retrieval. Use Prepare full course
            to ingest the playlist sequentially under {html.escape(MIT_COURSE_ID)}.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if prepare_clicked:
        progress = st.progress(0, text="Starting course ingest...")
        for index, video in enumerate(MIT_COURSE_VIDEOS, start=1):
            progress.progress((index - 1) / len(MIT_COURSE_VIDEOS), text=f"Indexing {video['lecture']}: {video['title']}")
            try:
                ingest_course_video(video, MIT_COURSE_FORCE_REINDEX)
            except Exception as exc:
                st.error(f"Stopped at {video['lecture']}: {exc}")
                break
        else:
            progress.progress(1.0, text="Course ingest complete.")
            st.success("All course lectures are ready for course-wide questions.")

    pending_video_id = st.session_state.pending_course_video_id
    if pending_video_id:
        pending_video = course_video_by_id(pending_video_id)
        if not pending_video:
            st.session_state.pending_course_video_id = None
            st.warning("Could not find the selected lecture.")
        else:
            st.info(f"Opening {pending_video['lecture']}: {pending_video['title']}...")
            with st.spinner("Preparing lecture and loading the player..."):
                st.session_state.pending_course_video_id = None
                load_course_video(pending_video)

    columns = st.columns(3, gap="medium")
    for index, video in enumerate(MIT_COURSE_VIDEOS):
        remember_course_video(video)
        video_id = youtube_video_id(video["url"])
        ingest_result = st.session_state.course_ingest_results.get(video_id, {})
        thumbnail_url = youtube_thumbnail_url(video["url"])
        summary = ingest_result.get("summary") or video["description"]
        summary = str(summary)
        if len(summary) > 220:
            summary = f"{summary[:217].rstrip()}..."
        status = "Ready" if ingest_result else "Not indexed"
        with columns[index % 3]:
            st.markdown(
                f"""
                <div class="lecture-card">
                    <img class="lecture-thumb" src="{html.escape(thumbnail_url)}" alt="">
                    <div class="lecture-body">
                        <div class="lecture-kicker">{html.escape(video["lecture"])} · {html.escape(video["duration"])} · {status}</div>
                        <div class="lecture-title">{html.escape(video["title"])}</div>
                        <div class="lecture-desc">{html.escape(video["description"])}</div>
                        <div class="lecture-summary">{html.escape(summary)}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Start learning", key=f"course_video_{video_id}", use_container_width=True):
                st.session_state.pending_course_video_id = video_id
                st.rerun()


def render_home_page() -> None:
    render_header()
    st.markdown('<div class="home-spacer"></div>', unsafe_allow_html=True)
    _, hero_col, _ = st.columns([1, 6, 1])
    with hero_col:
        st.markdown('<h1 class="home-title">Your Lecture. Your Graph. Your Answers.</h1>', unsafe_allow_html=True)
        st.markdown(
            '<div class="home-subtitle">GraphLens maps every concept, finds every connection, and cites every claim.</div>',
            unsafe_allow_html=True,
        )
    spacer_l, col1, col2, spacer_r = st.columns([1.5, 2.3, 2.3, 1.5], gap="large")
    with col1:
        if st.button("🔗\n\nLink URL\n\nConnect web resources", key="home_link_card", use_container_width=True):
            link_url_dialog()
    with col2:
        if st.button("📄\n\nUpload Document\n\nImport PDF content", key="home_doc_card", use_container_width=True):
            upload_pdf_dialog()
    render_course_feature()


def render_source(source: dict[str, Any], index: int) -> None:
    source_url = source.get("source_url") or ""
    page_or_time = source.get("start_seconds")
    is_pdf = bool(source.get("doc_id")) or st.session_state.scope_type == "document"
    is_course = st.session_state.scope_type == "course" or bool(source.get("course_id"))
    if is_pdf and page_or_time is not None:
        citation = f"Page {int(float(page_or_time))}"
    elif is_course and source.get("video_id") and page_or_time is not None:
        video_id = source["video_id"]
        lecture_meta = st.session_state.course_video_titles.get(video_id, {})
        lecture_label = lecture_meta.get("lecture", "Lecture")
        lecture_title = lecture_meta.get("title", video_id)
        citation = f"{lecture_label} - {lecture_title}, {format_timestamp(page_or_time)}"
        source_url = f"https://youtube.com/watch?v={video_id}&t={int(float(page_or_time))}"
    elif source.get("video_id") and page_or_time is not None:
        citation = format_timestamp(page_or_time)
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
    if is_pdf and page_or_time is not None:
        page_num = int(float(page_or_time))
        pdf_url = (st.session_state.content_meta or {}).get("pdf_url", "")
        if pdf_url:
            st.link_button(f"Jump to Page {page_num}", f"{pdf_url}#page={page_num}")
    elif source_url and not is_pdf:
        st.link_button("Open timestamp", source_url)


# def render_answer_markdown(answer: str) -> None:
#     escaped = html.escape(answer)
#     rendered = re.sub(
#         r"\[(\d+)\]",
#         lambda match: f'<sup class="source-badge">[{match.group(1)}]</sup>',
#         escaped,
#     )
#     st.markdown(rendered, unsafe_allow_html=True)
def render_answer_markdown(answer: str) -> None:
    sources = st.session_state.sources or []
    pdf_url = (st.session_state.content_meta or {}).get("pdf_url", "")
    is_pdf = st.session_state.scope_type == "document"

    def make_badge(match):
        num = int(match.group(1))
        idx = num - 1  # citations are 1-indexed
        if is_pdf and pdf_url and 0 <= idx < len(sources):
            page = int(float(sources[idx].get("start_seconds", 1)))
            jump_url = f"{pdf_url}#page={page}"
            return (
                    f'<a href="{jump_url}" target="_blank" '
                    f'class="source-badge" style="text-decoration:none;">'
                    f'[{num}]</a>'
                )

        elif not is_pdf and 0 <= idx < len(sources):
            video_id = sources[idx].get("video_id", "")
            t = int(float(sources[idx].get("start_seconds", 0)))
            if video_id:
                url = f"https://youtube.com/watch?v={video_id}&t={t}"
                return (
                    f'<a href="{url}" target="_blank" '
                    f'class="source-badge" style="text-decoration:none;">'
                    f'[{num}]</a>'
                )
        return f'<sup class="source-badge">[{num}]</sup>'

    escaped = html.escape(answer)
    rendered = re.sub(r"\[(\d+)\]", make_badge, escaped)
    st.markdown(rendered, unsafe_allow_html=True)


def confidence_badge_html(confidence: float | None) -> str:
    if confidence is None:
        return ""
    percent = round(max(0.0, min(1.0, confidence)) * 100)
    if confidence >= 0.8:
        label = f"{percent}% confidence"
        class_name = "confidence-high"
    elif confidence >= 0.5:
        label = f"Moderately confident · {percent}%"
        class_name = "confidence-medium"
    else:
        label = f"Treat with caution · {percent}%"
        class_name = "confidence-low"
    return (
        f'<span class="confidence-badge {class_name}" '
        'title="This score reflects how grounded the answer is in the source material.">'
        f'<span class="confidence-dot"></span>{html.escape(label)}</span>'
    )


def parse_topic(topic: str) -> tuple[str, str | None]:
    name, separator, description = topic.partition(":")
    if not separator:
        return topic.strip(), None
    return name.strip(), description.strip()


GRAPH_STOP_WORDS = {
    "about",
    "after",
    "also",
    "answer",
    "are",
    "can",
    "could",
    "does",
    "explain",
    "for",
    "from",
    "give",
    "how",
    "into",
    "is",
    "me",
    "of",
    "show",
    "tell",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "why",
    "with",
    "work",
    "works",
}


def clean_graph_concept(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s_-]+", " ", str(value or "").lower())
    words = [word for word in text.split() if word not in GRAPH_STOP_WORDS]
    return " ".join(words[:4]).strip()


def graph_concept_candidates() -> list[str]:
    candidates: list[str] = []
    if st.session_state.graph_concept:
        candidates.append(str(st.session_state.graph_concept))
    if st.session_state.last_question:
        candidates.append(clean_graph_concept(st.session_state.last_question))
    for topic in st.session_state.key_topics:
        name, _ = parse_topic(str(topic))
        candidates.append(clean_graph_concept(name))

    unique = []
    for candidate in candidates:
        candidate = candidate.strip()
        if len(candidate) >= 2 and candidate not in unique:
            unique.append(candidate)
    return unique


def current_graph_concept() -> str:
    candidates = graph_concept_candidates()
    return candidates[0] if candidates else ""


def render_content_preview() -> None:
    if st.session_state.content_type in {"youtube", "course"}:
        st.video(st.session_state.content_data)
    elif st.session_state.content_type == "pdf":
        pdf_url = (st.session_state.content_meta or {}).get("pdf_url")
        if pdf_url:
            st.markdown(
                f'<iframe src="{pdf_url}" name="pdf-viewer" width="100%" height="460" '
                f'style="border:none; border-radius:18px;"></iframe>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="player-shell">
                    <div>
                        <div class="placeholder-camera">📄</div>
                        <div class="placeholder-title">{html.escape(str(st.session_state.content_name or "Document"))}</div>
                        <div class="placeholder-subtitle">Ask a question below to explore this document</div>
                    </div>
                </div>
                """,
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
        st.markdown(
            f'<div class="summary-text">{html.escape(str(st.session_state.summary))}</div>',
            unsafe_allow_html=True,
        )
    if st.session_state.key_topics:
        topic_items = []
        for topic in st.session_state.key_topics:
            name, description = parse_topic(str(topic))
            if description:
                topic_items.append(
                    '<div class="topic-card">'
                    f'<div class="topic-name">{html.escape(name)}</div>'
                    f'<div class="topic-desc">{html.escape(description)}</div>'
                    "</div>"
                )
            else:
                topic_items.append(f'<span class="topic-chip">{html.escape(name)}</span>')
        st.markdown(f'<div class="topics-list">{"".join(topic_items)}</div>', unsafe_allow_html=True)


def graph_chunk_label(chunk: dict[str, Any]) -> str:
    source_url = str(chunk.get("source_url") or "")
    marker = chunk.get("start_seconds")
    is_pdf = source_url.lower().endswith(".pdf") or st.session_state.content_type == "pdf"
    if marker is None:
        return source_url or chunk.get("chunk_id", "Source")
    if is_pdf:
        return f"{source_url or 'Document'} - Page {int(float(marker))}"
    return f"{source_url or 'Video'} - {format_timestamp(marker)}"


def graph_chunk_url(chunk: dict[str, Any]) -> str | None:
    source_url = str(chunk.get("source_url") or "")
    marker = chunk.get("start_seconds")
    if not source_url or source_url.lower().endswith(".pdf"):
        return None
    if "youtube.com" in source_url or "youtu.be" in source_url:
        if marker is None:
            return source_url
        separator = "&" if "?" in source_url else "?"
        return f"{source_url}{separator}t={int(float(marker))}"
    return source_url


def render_graph_visualization(graph: dict[str, Any]) -> None:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not nodes:
        concept = html.escape(str(graph.get("concept") or st.session_state.graph_concept or "this concept"))
        st.markdown(
            f'<div class="graph-empty">No graph nodes found for <strong>{concept}</strong>. Try a shorter concept such as "neural network" or a suggested topic.</div>',
            unsafe_allow_html=True,
        )
        return

    if Network is None:
        pills = []
        for node in nodes:
            label = html.escape(str(node.get("name") or "Concept"))
            class_name = "graph-node-pill graph-node-center" if node.get("type") == "center" else "graph-node-pill"
            pills.append(f'<span class="{class_name}">{label}</span>')
        st.markdown(f'<div class="graph-fallback">{"".join(pills)}</div>', unsafe_allow_html=True)
        return

    net = Network(
        height="520px",
        width="100%",
        bgcolor="#020817",
        font_color="#e2e8f0",
        directed=True,
        cdn_resources="in_line",
    )
    net.force_atlas_2based(
        gravity=-64,
        central_gravity=0.018,
        spring_length=170,
        spring_strength=0.08,
        damping=0.55,
    )
    center_name = ""
    for node in nodes:
        name = str(node.get("name") or "")
        if not name:
            continue
        is_center = node.get("type") == "center"
        if is_center:
            center_name = name
        net.add_node(
            name,
            label=name,
            title=name,
            color={
                "background": "#06b6d4" if is_center else "#132238",
                "border": "#a5f3fc" if is_center else "#38bdf8",
                "highlight": {"background": "#22d3ee", "border": "#e0f2fe"},
            },
            borderWidth=3 if is_center else 1,
            size=42 if is_center else 27,
            font={"size": 22 if is_center else 16, "face": "Inter", "color": "#f8fafc", "strokeWidth": 0},
        )
    center_name = center_name or str(nodes[0].get("name") or "")
    for edge in edges:
        source = edge.get("from") or edge.get("source")
        target = edge.get("to") or edge.get("target")
        if source and target:
            relation = str(edge.get("type") or "RELATES_TO")
            net.add_edge(
                str(source),
                str(target),
                label=relation,
                title=relation,
                color={"color": "#38bdf8", "highlight": "#67e8f9", "opacity": 0.72},
                width=2,
                arrows={"to": {"enabled": True, "scaleFactor": 0.55}},
            )
    net.set_options(
        """
        var options = {
          "nodes": {
            "shape": "dot",
            "shadow": {"enabled": true, "color": "rgba(34,211,238,0.22)", "size": 14, "x": 0, "y": 0}
          },
          "edges": {
            "smooth": {"type": "continuous", "roundness": 0.35},
            "font": {"size": 12, "color": "#93c5fd", "strokeWidth": 0, "align": "middle"}
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 80,
            "navigationButtons": true,
            "keyboard": true,
            "dragNodes": true
          },
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "stabilization": {"enabled": true, "iterations": 220, "fit": true}
          }
        }
        """
    )
    components.html(net.generate_html(notebook=False), height=540, scrolling=False)


def render_graph_panel() -> None:
    if not st.session_state.sources and not st.session_state.last_question:
        st.caption("Ask a question first so GraphLens can center the concept map on your query.")
        return

    candidates = graph_concept_candidates()
    default_concept = current_graph_concept()
    concept = st.text_input(
        "Concept",
        value=default_concept,
        placeholder="gradient descent",
        help="Use a key topic from the answer or the concept you want to explore.",
        key="graph_concept_input",
    )
    load_col, health_col = st.columns([0.7, 0.3])
    with load_col:
        if st.button("Load concept map", type="primary", use_container_width=True):
            with st.spinner("Fetching concept graph..."):
                load_graph(concept)
            st.rerun()
    with health_col:
        status = "Neo4j connected" if st.session_state.graph_health and st.session_state.graph_health.get("neo4j") == "connected" else "Neo4j unavailable"
        st.caption(status)

    if candidates:
        st.caption("Suggestions")
        cols = st.columns(min(3, len(candidates)))
        for index, candidate in enumerate(candidates[:6]):
            with cols[index % len(cols)]:
                if st.button(candidate, key=f"graph_candidate_{index}_{hashlib.md5(candidate.encode()).hexdigest()[:8]}", use_container_width=True):
                    with st.spinner("Fetching concept graph..."):
                        load_graph(candidate)
                    st.rerun()

    graph = st.session_state.graph
    if not graph:
        st.caption("Load the concept map to render related concepts and source links.")
        return
    if graph.get("error"):
        st.warning(graph["error"])
        return

    node_count = graph.get("node_count", len(graph.get("nodes", [])))
    edge_count = graph.get("edge_count", len(graph.get("edges", [])))
    st.markdown(
        f"""
        <div class="metric-strip">
            <span class="metric-pill">{html.escape(str(graph.get("concept") or st.session_state.graph_concept or concept))}</span>
            <span class="metric-pill">{node_count} nodes</span>
            <span class="metric-pill">{edge_count} edges</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_graph_visualization(graph)
    if graph.get("nodes") and not graph.get("edges"):
        st.caption("The API returned the center concept but no related concept edges for this topic.")

    neighbor_nodes = [node for node in graph.get("nodes", []) if node.get("type") != "center" and node.get("name")]
    if neighbor_nodes:
        st.caption("Re-center")
        cols = st.columns(min(4, len(neighbor_nodes)))
        for index, node in enumerate(neighbor_nodes[:8]):
            name = str(node["name"])
            with cols[index % len(cols)]:
                if st.button(name, key=f"graph_recenter_{index}_{hashlib.md5(name.encode()).hexdigest()[:8]}", use_container_width=True):
                    with st.spinner("Fetching concept graph..."):
                        load_graph(name)
                    st.rerun()

    chunks = [chunk for chunk in graph.get("chunks", []) if chunk and chunk.get("chunk_id")]
    if chunks:
        st.markdown("**Source Links**")
        for chunk in chunks[:6]:
            label = html.escape(graph_chunk_label(chunk))
            chunk_id = html.escape(str(chunk.get("chunk_id", "")))
            st.markdown(
                f'<div class="chunk-link-card"><div>{label}</div><div class="source-meta">{chunk_id}</div></div>',
                unsafe_allow_html=True,
            )
            url = graph_chunk_url(chunk)
            if url:
                st.link_button("Open timestamp", url)


def submit_query(prompt: str) -> None:
    try:
        graph_available = st.session_state.graph_health and st.session_state.graph_health.get("neo4j") == "connected"
        st.session_state.last_question = prompt
        st.session_state.graph_concept = clean_graph_concept(prompt)
        st.session_state.graph = None
        response = post_json(
            "/query",
            {
                "question": prompt,
                "scope_type": st.session_state.scope_type,
                "scope_id": st.session_state.scope_id,
                "collection_name": COLLECTION_NAME,
                "use_graph": bool(st.session_state.scope_type in {"document", "course"} and graph_available)
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
                "confidence": response.get("confidence"),
                "model": response.get("model"),
                "citations": response.get("citations", []),
            }
        )
    except Exception as exc:
        st.session_state.messages.append({"role": "assistant", "content": f"Query failed: {exc}", "refused": True})


def load_graph(concept: str | None = None) -> None:
    if not st.session_state.sources and not st.session_state.last_question:
        st.warning("Ask a question first so GraphLens has source context.")
        return
    concept = clean_graph_concept(concept) or current_graph_concept()
    if not concept:
        st.warning("Enter a concept to visualize.")
        return
    try:
        st.session_state.graph_concept = concept
        graph = get_json("/graph/concept", {"concept": concept, "scope_id": st.session_state.scope_id})
        st.session_state.graph = graph
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
        tab1, tab2, tab3 = st.tabs(["Summary", "Sources", "Concept Map"])
        with tab1:
            with st.container(height=330, border=True):
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
        with tab3:
            with st.container(height=780, border=True):
                render_graph_panel()

    with right_col:
        st.markdown('<div class="chat-title">Ask Me Anything</div>', unsafe_allow_html=True)
        graph_available = st.session_state.graph_health and st.session_state.graph_health.get("neo4j") == "connected"
        if not graph_available:
            st.session_state.use_graph = False
        if st.session_state.scope_type in {"document", "course"}:
            st.caption("● GraphRAG active — knowledge graph expansion enabled")
        else:
            st.caption("● Plain RAG — fast single-video retrieval")
        if not graph_available:
            st.caption("Neo4j is not connected. Plain RAG is active.")

        chat_panel = st.container(height=555, border=True)
        with chat_panel:
            msg_scroll = st.container(height=462, border=False)
            with msg_scroll:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        if msg["role"] == "assistant" and not msg.get("refused"):
                            badge = confidence_badge_html(msg.get("confidence"))
                            if badge:
                                st.markdown(badge, unsafe_allow_html=True)
                            render_answer_markdown(msg["content"])
                        else:
                            st.write(msg["content"])
                        if msg.get("best_similarity") is not None and not msg.get("refused"):
                            st.caption(f"Best similarity: {msg['best_similarity']:.3f}")
                        if msg.get("model") and not msg.get("refused"):
                            st.caption(f"Model: {msg['model']}")
                if st.session_state.pending_query:
                    with st.chat_message("assistant"):
                        with st.spinner("Retrieving grounded evidence..."):
                            st.caption("Working on an answer...")

            with st.form("query_form", clear_on_submit=True):
                q_in, q_send = st.columns([0.87, 0.13], gap="small")
                with q_in:
                    prompt = st.text_input(
                        "Ask about the indexed source",
                        placeholder="Ask your question...",
                        label_visibility="collapsed",
                    )
                with q_send:
                    submitted = st.form_submit_button(
                        "→",
                        type="primary",
                        use_container_width=True,
                        key="query_send_arrow",
                    )
                if submitted and prompt.strip():
                    st.session_state.messages.append({"role": "user", "content": prompt.strip()})
                    st.session_state.pending_query = prompt.strip()
                    st.rerun()

        if st.session_state.pending_query:
            pending_query = st.session_state.pending_query
            submit_query(pending_query)
            st.session_state.pending_query = None
            st.rerun()




def dispatch_page() -> None:
    if st.session_state.page == "home":
        render_home_page()
    elif st.session_state.page == "course":
        render_course_page()
    elif st.session_state.page == "session":
        render_learning_session()


if st.session_state.get("menu_open"):
    drawer_col, main_col = st.columns([0.265, 0.735], gap="medium")
    with drawer_col:
        render_nav_drawer()
    with main_col:
        dispatch_page()
else:
    dispatch_page()
