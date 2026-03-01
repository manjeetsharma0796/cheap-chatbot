import warnings
warnings.filterwarnings("ignore")

import os
import uuid
import streamlit as st
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from memory import FileChatMessageHistory, list_sessions

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL   = "mistralai/mixtral-8x7b-instruct-v0.1"

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Chat", page_icon="🤖", layout="wide")

# ── localStorage JS: persist + restore session_id across browser refreshes ────
# Reads localStorage on first load; if no ?session= in URL, redirects with it.
st.components.v1.html("""
<script>
(function() {
    const params = new URLSearchParams(window.parent.location.search);
    const urlSession = params.get('session');
    const stored = localStorage.getItem('ai_chat_session_id');

    if (!urlSession && stored) {
        // Restore last session from localStorage
        params.set('session', stored);
        window.parent.location.search = params.toString();
    } else if (urlSession) {
        // Save current URL session to localStorage
        localStorage.setItem('ai_chat_session_id', urlSession);
    }
})();
</script>
""", height=0)

# ── Session ID: read from query params or generate new ───────────────────────
if "session_id" not in st.session_state:
    qp = st.query_params.get("session")
    st.session_state.session_id = qp if qp else str(uuid.uuid4())[:8]
    if not qp:
        st.query_params["session"] = st.session_state.session_id


def switch_session(new_id: str):
    st.session_state.session_id = new_id
    st.query_params["session"] = new_id


# ── LLM + chain ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_chain():
    llm = ChatNVIDIA(model=MODEL, temperature=0.7, api_key=API_KEY)
    return RunnableWithMessageHistory(llm, FileChatMessageHistory)

chain = get_chain()

# ── Sidebar: chat history like ChatGPT ─────────────────────────────────────────
with st.sidebar:
    st.title("🤖 AI Chat")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        switch_session(str(uuid.uuid4())[:8])
        st.rerun()

    st.divider()
    st.caption("Chat History")

    sessions = list_sessions()
    current_id = st.session_state.session_id

    for s in sessions:
        is_active = s["id"] == current_id
        col1, col2 = st.columns([5, 1])
        with col1:
            label = ("\u25b6 " if is_active else "") + s["title"]
            if st.button(label, key=f"sess_{s['id']}", use_container_width=True,
                         help=f"Session: {s['id']}"):
                switch_session(s["id"])
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{s['id']}", help="Delete"):
                FileChatMessageHistory(s["id"]).clear()
                if is_active:
                    switch_session(str(uuid.uuid4())[:8])
                st.rerun()

    if not sessions:
        st.caption("No chats yet. Start typing!")

# ── Main chat area ─────────────────────────────────────────────────────────────────
st.title("🤖 AI Chatbot")

history = FileChatMessageHistory(st.session_state.session_id)

for msg in history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ── Chat input ───────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Type a message...", key="main_chat_input"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        config = {"configurable": {"session_id": st.session_state.session_id}}
        placeholder = st.empty()
        full_response = ""

        for chunk in chain.stream(prompt, config=config):
            full_response += chunk.content
            placeholder.markdown(full_response + "◌")

        placeholder.markdown(full_response)

    st.rerun()  # Refresh sidebar to update chat title after first message
