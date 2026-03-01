import base64
import mimetypes
import warnings
warnings.filterwarnings("ignore")

import os
import uuid
from types import SimpleNamespace

import streamlit as st
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False
from langchain_core.messages import AIMessage, HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from memory import FileChatMessageHistory, list_sessions

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL   = "mistralai/mixtral-8x7b-instruct-v0.1"

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Chat", page_icon="🤖", layout="wide")

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Attach button: float fixed just above the chat input bar ── */
[data-testid="stMarkdown"]:has(#st-attach-anchor) {
    display: none !important;
}
[data-testid="stMarkdown"]:has(#st-attach-anchor)
  + [data-testid="stHorizontalBlock"] {
    position: fixed !important;
    bottom: 76px !important;
    /* mobile: left edge with safe area */
    left: max(16px, env(safe-area-inset-left, 16px)) !important;
    z-index: 998 !important;
    width: auto !important;
    max-width: 3.5rem !important;
}
/* Desktop: shift right to clear sidebar */
@media (min-width: 768px) {
    [data-testid="stMarkdown"]:has(#st-attach-anchor)
      + [data-testid="stHorizontalBlock"] {
        left: max(360px, calc(336px + 24px)) !important;
    }
}
/* Compact round icon look for the popover trigger button */
[data-testid="stMarkdown"]:has(#st-attach-anchor)
  + [data-testid="stHorizontalBlock"] .stPopover > button,
[data-testid="stMarkdown"]:has(#st-attach-anchor)
  + [data-testid="stHorizontalBlock"] [data-testid="stPopover"] > button {
    border-radius: 50% !important;
    width: 2.8rem !important;
    height: 2.8rem !important;
    min-height: unset !important;
    padding: 0 !important;
    font-size: 1.15rem !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
/* Attached-image badge near input (small pill) */
.attach-badge {
    position: fixed !important;
    bottom: 76px !important;
    left: max(72px, env(safe-area-inset-left, 72px)) !important;
    z-index: 997 !important;
    background: var(--secondary-background-color, #2d2d2d);
    border-radius: 20px;
    padding: 4px 10px;
    font-size: 0.75rem;
    display: flex;
    align-items: center;
    gap: 6px;
    max-width: 180px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
@media (min-width: 768px) {
    .attach-badge {
        left: max(416px, calc(336px + 80px)) !important;
    }
}
/* Sidebar model picker: sticky at top */
[data-testid="stSidebar"] [data-testid="stSelectbox"] {
    position: sticky !important;
    top: 0 !important;
    z-index: 10 !important;
    background: var(--sidebar-background-color, inherit);
    padding-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── localStorage JS ───────────────────────────────────────────────────────────
# JS is the SOLE authority for writing user_id + session_id into the URL on
# initial load.  Python never writes query params on first render — doing so
# would race with JS and overwrite the localStorage-stored session with a new
# random one, losing the user's history.
#
# Flow:
#   1. User visits bare URL (no ?user= or ?session=)
#   2. JS reads localStorage → builds correct params → redirects (<100 ms)
#   3. Python re-runs with full params in URL → loads history as normal
#   4. On refresh: URL already has params (Python set them via switch_session)
#      → JS just syncs localStorage, no redirect needed
st.components.v1.html("""
<script>
(function() {
    const params = new URLSearchParams(window.parent.location.search);
    let changed = false;

    // ── user_id: stable per browser, never changes ───────────────────────────
    let userId = localStorage.getItem('ai_chat_user_id');
    if (!userId) {
        userId = 'u_' + Math.random().toString(36).slice(2, 10);
        localStorage.setItem('ai_chat_user_id', userId);
    }
    if (!params.get('user')) {
        params.set('user', userId);
        changed = true;
    }

    // ── session_id: restore from localStorage or mint a new one ─────────────
    let sessionId = params.get('session');
    if (!sessionId) {
        // No session in URL — check localStorage first, then create fresh
        sessionId = localStorage.getItem('ai_chat_session_id');
        if (!sessionId) {
            sessionId = Math.random().toString(36).slice(2, 10);
        }
        params.set('session', sessionId);
        changed = true;
    }
    // Always keep localStorage in sync with whatever session is active in URL
    localStorage.setItem('ai_chat_session_id', sessionId);

    if (changed) {
        window.parent.location.search = params.toString();
    }
})();
</script>
""", height=0)

# ── Session ID + User ID: read ONLY from URL (JS wrote them above) ────────────
# If params are missing, JS redirect is still in flight — show a spinner and
# halt. The page reloads with correct params in < 200 ms.
if "user_id" not in st.session_state or "session_id" not in st.session_state:
    _qu = st.query_params.get("user")
    _qs = st.query_params.get("session")
    if not _qu or not _qs:
        st.markdown(
            """
            <style>
            #loading-wrap {
                display: flex; flex-direction: column;
                align-items: center; justify-content: center;
                height: 80vh; gap: 16px;
            }
            .spinner {
                width: 40px; height: 40px;
                border: 4px solid rgba(255,255,255,0.15);
                border-top-color: #e63946;
                border-radius: 50%;
                animation: spin 0.7s linear infinite;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
            </style>
            <div id="loading-wrap">
                <div class="spinner"></div>
                <span style="opacity:0.5;font-size:0.9rem;">Starting session…</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()
    st.session_state.user_id    = _qu
    st.session_state.session_id = _qs


def switch_session(new_id: str):
    st.session_state.session_id = new_id
    st.query_params["session"] = new_id
    st.query_params["user"] = st.session_state.user_id


def _stub_model(model_id: str):
    return SimpleNamespace(
        id=model_id,
        model_type="chat",
        supports_tools=False,
        supports_structured_output=False,
        supports_thinking=False,
    )


@st.cache_data(show_spinner=False)
def load_models(api_key: str | None):
    if not api_key:
        return []
    try:
        return ChatNVIDIA.get_available_models(api_key=api_key)
    except Exception as exc:
        warnings.warn(f"Falling back to default model list: {exc}")
        return []


def model_capability_badges(model) -> list[str]:
    badges = []
    if model and "vlm" in str(getattr(model, "model_type", "")):
        badges.append("👁️ Vision")
    if getattr(model, "supports_thinking", False):
        badges.append("🧠 Thinking")
    if getattr(model, "supports_tools", False):
        badges.append("🛠️ Tools")
    if getattr(model, "supports_structured_output", False):
        badges.append("📊 JSON")
    return badges


def format_model_option(model_id: str, lookup: dict[str, object]) -> str:
    model = lookup.get(model_id)
    mtype = str(getattr(model, "model_type", ""))
    if "vlm" in mtype:
        icon = "👁️"
    elif getattr(model, "supports_thinking", False):
        icon = "🧠"
    elif getattr(model, "supports_tools", False):
        icon = "🛠️"
    else:
        icon = "💬"
    return f"{icon}  {model_id}"


def is_vision_model(model) -> bool:
    return "vlm" in str(getattr(model, "model_type", ""))


def encode_image_to_data_url(image: dict) -> str:
    mime = image.get("mime") or "image/png"
    b64 = base64.b64encode(image["data"]).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def render_message_content(message):
    content = message.content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                st.markdown(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    st.markdown(part.get("text", ""))
                elif part.get("type") == "image_url":
                    img_url = part.get("image_url")
                    if isinstance(img_url, dict):
                        img_url = img_url.get("url")
                    if isinstance(img_url, str) and img_url.startswith("data:image"):
                        try:
                            _, data = img_url.split(",", 1)
                            st.image(base64.b64decode(data), use_column_width=True)
                        except Exception:
                            st.markdown("📎 Image attached")
                    elif isinstance(img_url, str) and img_url:
                        st.image(img_url, use_column_width=True)
    else:
        st.markdown(content)


# ── LLM (direct — history managed manually to control image stripping) ──────────
@st.cache_resource(show_spinner=False)
def get_llm(model_id: str, thinking: bool):
    # stream_options=None avoids "Extra inputs are not permitted" on some endpoints
    llm = ChatNVIDIA(model=model_id, temperature=0.7, api_key=API_KEY, stream_options=None)
    if thinking:
        llm = llm.with_thinking_mode(enabled=True)
    return llm


def strip_images(msg):
    """Return msg with image_url blocks removed — keeps only text parts.
    Vision APIs like NVIDIA allow at most 1 image per prompt, so we strip
    images from all historical messages and only keep the latest one.
    """
    content = msg.content
    if not isinstance(content, list):
        return msg
    text_parts = [
        p for p in content
        if isinstance(p, dict) and p.get("type") == "text"
    ]
    text = " ".join(p.get("text", "") for p in text_parts).strip() or "[image]"
    if msg.type == "human":
        return HumanMessage(content=text)
    return AIMessage(content=text)

# ── Sidebar: chat history like ChatGPT ─────────────────────────────────────────
with st.sidebar:
    st.title("🤖 AI Chat")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        switch_session(str(uuid.uuid4())[:8])
        st.rerun()

    st.divider()
    st.caption("Chat History")

    sessions = list_sessions(st.session_state.user_id)
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
                FileChatMessageHistory(st.session_state.user_id, s["id"]).clear()
                if is_active:
                    switch_session(str(uuid.uuid4())[:8])
                st.rerun()

    if not sessions:
        st.caption("No chats yet. Start typing!")

# ── Model chooser lives in sidebar ────────────────────────────────────────────
# Resolved here (before sidebar block closes) so selected_model* is available
# for the rest of the page regardless of sidebar state.
available_models = load_models(API_KEY)
model_lookup = {m.id: m for m in available_models}
model_options = list(model_lookup.keys()) or [MODEL]

if (
    "selected_model_id" not in st.session_state
    or st.session_state.selected_model_id not in model_options
):
    st.session_state.selected_model_id = (
        MODEL if MODEL in model_options else model_options[0]
    )

if "thinking_enabled" not in st.session_state:
    st.session_state.thinking_enabled = False

with st.sidebar:
    st.divider()
    st.caption("🤖 Model")
    selected_model_id = st.selectbox(
        "Supported models",
        options=model_options,
        key="selected_model_id",
        format_func=lambda mid: format_model_option(mid, model_lookup),
        help="All models available under your NVIDIA credentials.",
        label_visibility="collapsed",
    )
    selected_model = model_lookup.get(selected_model_id, _stub_model(selected_model_id))
    supports_vision = is_vision_model(selected_model)
    supports_thinking = bool(getattr(selected_model, "supports_thinking", False))

    _badges = model_capability_badges(selected_model)
    if _badges:
        st.caption(" · ".join(_badges))
    else:
        st.caption("💬 Text only")

    if not supports_thinking:
        st.session_state.thinking_enabled = False
    if supports_thinking:
        st.session_state.thinking_enabled = st.toggle(
            "🧠 Thinking mode",
            value=st.session_state.thinking_enabled,
            help="Enable deeper reasoning (only shown for thinking-capable models).",
        )

if not supports_vision:
    st.session_state.pop("attached_image", None)

llm = get_llm(selected_model_id, st.session_state.thinking_enabled)

# ── Main chat area ─────────────────────────────────────────────────────────────────
st.title("🤖 AI Chatbot")

history = FileChatMessageHistory(st.session_state.user_id, st.session_state.session_id)

for msg in history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        render_message_content(msg)

# ── Attachments (vision models) — fixed icon above chat input ────────────────
if supports_vision:
    # Invisible anchor so CSS knows which sibling to fix
    st.markdown('<span id="st-attach-anchor"></span>', unsafe_allow_html=True)
    attach_col, _ = st.columns([1, 20])
    with attach_col:
        with st.popover("📎"):
            st.markdown("**Attach an image**")
            upload = st.file_uploader(
                "Choose image",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=False,
                label_visibility="collapsed",
            )
            if upload:
                st.session_state.attached_image = {
                    "name": upload.name,
                    "mime": upload.type or mimetypes.guess_type(upload.name)[0] or "image/png",
                    "data": upload.getvalue(),
                }
                st.success(f"✅ {upload.name}")

            if attached := st.session_state.get("attached_image"):
                st.image(attached["data"], caption=attached.get("name"), width=160)
                if st.button("🗑️ Remove", key="remove_image_button"):
                    st.session_state.pop("attached_image", None)
                    st.rerun()

    # Small badge showing filename when an image is queued
    if attached_img := st.session_state.get("attached_image"):
        fname = attached_img.get("name", "image")
        st.markdown(
            f'<div class="attach-badge">🖼️ {fname}</div>',
            unsafe_allow_html=True,
        )

# ── Chat input ───────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Type a message...", key="main_chat_input"):
    content: list[object] = []
    if prompt:
        content.append({"type": "text", "text": prompt})

    image_blob = st.session_state.get("attached_image") if supports_vision else None
    if image_blob:
        content.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(image_blob)}})

    user_message = HumanMessage(content=content if len(content) > 1 else prompt)

    with st.chat_message("user"):
        render_message_content(user_message)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # Build message list: history with images stripped + current message with image
        hist_obj = FileChatMessageHistory(st.session_state.user_id, st.session_state.session_id)
        past = [strip_images(m) for m in hist_obj.messages]
        messages_to_send = past + [user_message]

        try:
            for chunk in llm.stream(messages_to_send):
                full_response += chunk.content
                placeholder.markdown(full_response + "◌")
        except Exception as exc:
            full_response = f"⚠️ {exc}"

        placeholder.markdown(full_response)

        # Save both turns to history
        hist_obj.add_messages([user_message, AIMessage(content=full_response)])

    st.session_state.pop("attached_image", None)
    st.rerun()  # Refresh sidebar to update chat title after first message
