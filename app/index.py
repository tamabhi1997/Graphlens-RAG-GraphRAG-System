import streamlit as st
import base64
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="GraphLens", layout="wide", initial_sidebar_state="collapsed")

# --- URL ROUTING (Fix for Logo Click) ---
# If the URL contains "?page=home" (triggered by clicking the logo), reset to home
if "page" in st.query_params:
    if st.query_params["page"] == "home":
        st.session_state.page = "home"
        # We clear the query param so it doesn't get stuck in a loop
        st.query_params.clear()

# --- STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'content_type' not in st.session_state:
    st.session_state.content_type = None
if 'content_data' not in st.session_state:
    st.session_state.content_data = None

# Initialize Chat History
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "How does neural cross-encoder reranking improve the precision of this system?"},
        {"role": "assistant", "content": "Cross-encoder reranking acts as a 'deep read' to filter out noise from the initial retrieval processes.", "confidence": "92%"}
    ]

def navigate_to(page_name):
    st.session_state.page = page_name
    st.rerun()

# --- MODAL DIALOGS ---

@st.dialog("Upload File")
def upload_file_dialog():
    st.markdown("Select a Video or PDF file from your device.")
    uploaded_file = st.file_uploader("Choose a file", type=["mp4", "mov", "avi", "pdf"], label_visibility="collapsed")
    if st.button("Start Learning", type="primary", use_container_width=True):
        if uploaded_file:
            st.session_state.content_data = uploaded_file
            if uploaded_file.name.lower().endswith('.pdf'):
                st.session_state.content_type = 'pdf'
            else:
                st.session_state.content_type = 'video'
            navigate_to('session')
        else:
            st.warning("Please upload a file first.")

@st.dialog("Add a Link")
def link_url_dialog():
    st.markdown("Paste a YouTube video or website URL to learn from it")
    url = st.text_input("URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")
    if st.button("Start Learning", type="primary", use_container_width=True):
        if url:
            st.session_state.content_type = 'youtube'
            st.session_state.content_data = url
            navigate_to('session')
        else:
            st.warning("Please enter a link first.")

@st.dialog("Paste Text")
def paste_text_dialog():
    st.markdown("Paste any text content you'd like to learn from")
    text = st.text_area("Text", placeholder="Paste your text here...", height=150, label_visibility="collapsed")
    if st.button("Start Learning", type="primary", use_container_width=True):
        if text:
            st.session_state.content_type = 'text'
            st.session_state.content_data = text
            navigate_to('session')
        else:
            st.warning("Please paste some text first.")

@st.dialog("Record Audio")
def record_lecture_dialog():
    st.markdown("Click the microphone to start recording")
    audio_value = st.audio_input("Ready to record", label_visibility="collapsed")
    if st.button("Start Learning", type="primary", use_container_width=True):
        if audio_value:
            st.session_state.content_type = 'audio'
            st.session_state.content_data = audio_value
            navigate_to('session')
        else:
            st.warning("Please record some audio first.")


# --- CUSTOM CSS ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem !important; padding-bottom: 0rem !important; }
    header { visibility: hidden; }
    .stApp { background-color: #0D1117; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    
    .header-logo { display: flex; align-items: center; gap: 16px; margin-top: 5px; }
    .hamburger { font-size: 24px; color: #8B949E; cursor: pointer; }
    .g-box {
        background: linear-gradient(135deg, #00D9FF, #0077FF);
        color: white; width: 36px; height: 36px; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 20px; box-shadow: 0 0 15px rgba(0, 217, 255, 0.5);
    }
    .brand { font-size: 22px; font-weight: 600; color: white; letter-spacing: 0.5px; }
    
    .custom-divider { border-bottom: 1px solid rgba(255, 255, 255, 0.1); margin-top: -15px; margin-bottom: 35px; }

    button[kind="primary"] {
        background: linear-gradient(90deg, #0077FF, #00D9FF) !important;
        border: none !important; color: white !important; border-radius: 999px !important;
        padding: 0.5rem 2rem !important; box-shadow: 0 0 15px rgba(0, 217, 255, 0.4) !important; font-weight: 600 !important;
    }
    button[kind="secondary"] {
        background: rgba(17, 25, 40, 0.6) !important; border: 1px solid rgba(6, 182, 212, 0.4) !important;
        color: white !important; border-radius: 12px !important; height: auto !important; padding: 20px !important; transition: all 0.3s ease !important;
    }
    button[kind="secondary"]:hover { border-color: #00D9FF !important; box-shadow: 0 0 20px rgba(0, 217, 255, 0.2) !important; color: #00D9FF !important; }
    
    /* Uniform container styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(17, 25, 40, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(6, 182, 212, 0.2) !important; 
        border-radius: 12px !important; 
        padding: 10px !important;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
    }
    
    .glass-panel {
        background: rgba(17, 25, 40, 0.6); backdrop-filter: blur(12px);
        border: 1px solid rgba(6, 182, 212, 0.2); border-radius: 16px; padding: 24px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    
    .browser-dots {
        display: flex; gap: 8px; padding: 12px 16px;
        background: #161B22; border-top-left-radius: 12px; border-top-right-radius: 12px;
        border: 1px solid rgba(6, 182, 212, 0.2); border-bottom: none;
    }
    .dot { width: 12px; height: 12px; border-radius: 50%; }
    .dot.red { background-color: #FF5F56; }
    .dot.yellow { background-color: #FFBD2E; }
    .dot.green { background-color: #27C93F; }
    
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Menu")
    st.button("🏠 Home", use_container_width=True, on_click=lambda: navigate_to('home'))
    st.button("⚙️ Settings", use_container_width=True)
    st.button("📚 My Library", use_container_width=True)

# --- SHARED HEADER ---
def render_header():
    head_col1, head_col2 = st.columns([0.85, 0.15])
    with head_col1:
        # LOGO FIX: Wrapped in an <a> tag targeting "?page=home"
        st.markdown("""
        <div class="header-logo">
            <span class="hamburger" onclick="window.parent.document.querySelector('[data-testid=\\'collapsedControl\\']').click()">☰</span>
            <a href="?page=home" target="_self" style="text-decoration:none; display:flex; align-items:center; gap:16px;">
                <div class="g-box">G</div>
                <span class="brand">GraphLens</span>
            </a>
        </div>
        """, unsafe_allow_html=True)
    with head_col2:
        st.button("Sign in", key="signin_btn", type="primary", use_container_width=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# --- PAGE 1: HOME PAGE ---
# --- PAGE 1: HOME PAGE ---
def render_home_page():
    render_header()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>Initialize Knowledge Graph</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #D1D5DB; font-size: 1.2rem; margin-bottom: 3rem;'>Link, paste, or record long-form media to begin neural semantic indexing</p>", unsafe_allow_html=True)
    
    # Using 5 columns to force the 3 buttons into the center
    # [1.5, 2, 2, 2, 1.5] determines the width ratio of each column
    spacer_left, col1, col2, col3, spacer_right = st.columns([1.5, 2, 2, 2, 1.5], gap="large")
    
    with col1:
        if st.button("🔗\n\n**Link URL**\n\nConnect web resources", key="link_btn"): link_url_dialog()
    with col2:
        if st.button("📄\n\n**Paste Text**\n\nCopy and paste content", key="paste_btn"): paste_text_dialog()
    with col3:
        if st.button("🎤\n\n**Record Lecture**\n\nCapture audio live", key="record_btn"): record_lecture_dialog()
            
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # The search bar is also centered using spacers
    _, search_col, _ = st.columns([2, 6, 2])
    with search_col:
        st.text_input("Ask Me Anything", placeholder="🔍 Ask Me Anything...")

# --- PAGE 2: LEARNING SESSION ---
def render_learning_session():
    render_header()
    
    left_col, right_col = st.columns([0.6, 0.4], gap="large")
    
    with left_col:
        st.markdown("""
        <div class="browser-dots" style="margin-bottom: -15px; position: relative; z-index: 10;">
            <div class="dot red"></div><div class="dot yellow"></div><div class="dot green"></div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container(height=450, border=True):
            if st.session_state.content_type == 'youtube':
                st.video(st.session_state.content_data)
            elif st.session_state.content_type == 'video':
                st.video(st.session_state.content_data)
            elif st.session_state.content_type == 'audio':
                st.markdown("<br><br><br><br><h4 style='text-align:center; color:#00D9FF;'>🎙️ Audio Track Loaded</h4>", unsafe_allow_html=True)
                st.audio(st.session_state.content_data)
            elif st.session_state.content_type == 'text':
                st.markdown("<h4 style='color:#00D9FF;'>📄 Extracted Text Content</h4>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-family: monospace; color: #A5B4FC; line-height: 1.6;'>{st.session_state.content_data}</div>", unsafe_allow_html=True)
            elif st.session_state.content_type == 'pdf':
                base64_pdf = base64.b64encode(st.session_state.content_data.getvalue()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400px" type="application/pdf" style="border:none;"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
            else:
                st.info("No content loaded. Open the sidebar and click Home to try again.")
        
        st.write("")
        
        tab1, tab2 = st.tabs(["Summary", "Chapters"])
        with tab1:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown("<h4 style='color:#00D9FF; margin-top: 0;'>Executive Summary</h4>", unsafe_allow_html=True)
            st.write("This document covers advanced machine learning techniques, focusing heavily on neural network architectures and their direct applications in modern information retrieval...")
            st.markdown('</div>', unsafe_allow_html=True)
            
    with right_col:
        st.markdown("### Intelligence Tools")
        tool_col1, tool_col2 = st.columns(2)
        with tool_col1:
            st.button("🕸️ View Graph", use_container_width=True)
            st.button("❓ Quiz", use_container_width=True)
        with tool_col2:
            st.button("📊 Reliability", use_container_width=True)
            st.button("📝 Notes", use_container_width=True)
            
        st.button("🗂️ Flashcards", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Ask Me Anything")
        
        # CHAT LOGIC: Render dynamic chat history
        chat_container = st.container(height=350, border=True)
        with chat_container:
            for msg in st.session_state.messages:
                # Toggle avatar based on role
                avatar_icon = "👤" if msg["role"] == "user" else "🌌"
                with st.chat_message(msg["role"], avatar=avatar_icon):
                    # Show confidence metric if the assistant provides one
                    if "confidence" in msg:
                        st.markdown(f"Confidence: <span style='color: #00FF00;'>{msg['confidence']}</span>", unsafe_allow_html=True)
                    st.write(msg["content"])

        # CHAT LOGIC: Accept new user input
        if prompt := st.chat_input("Query the grounded graph..."):
            # 1. Append user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 2. Append simulated AI response (In reality, you'd call an LLM API here)
            simulated_response = f"Based on the knowledge graph, I've analyzed your query regarding '{prompt}'. The contextual evidence suggests strong correlation with the uploaded materials."
            st.session_state.messages.append({"role": "assistant", "content": simulated_response, "confidence": "88%"})
            
            # 3. Rerun to instantly show the new messages in the UI
            st.rerun()

# --- ROUTER ---
if st.session_state.page == 'home':
    render_home_page()
elif st.session_state.page == 'session':
    render_learning_session()

