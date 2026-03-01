import json
import os
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict

STORE_DIR = os.path.join(os.path.dirname(__file__), "store")


def list_sessions() -> list[dict]:
    """Return all sessions sorted by last modified time (newest first)."""
    os.makedirs(STORE_DIR, exist_ok=True)
    sessions = []
    for fname in os.listdir(STORE_DIR):
        if not fname.endswith(".json"):
            continue
        session_id = fname[:-5]
        fpath = os.path.join(STORE_DIR, fname)
        title = get_session_title(session_id)
        mtime = os.path.getmtime(fpath)
        sessions.append({"id": session_id, "title": title, "mtime": mtime})
    return sorted(sessions, key=lambda x: x["mtime"], reverse=True)


def get_session_title(session_id: str, max_len: int = 40) -> str:
    """Return the first human message as the session title."""
    fpath = os.path.join(STORE_DIR, f"{session_id}.json")
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for msg in data:
            if msg.get("type") == "human":
                content = msg.get("data", {}).get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                content = content.strip()
                if content:
                    return content[:max_len] + ("..." if len(content) > max_len else "")
    except Exception:
        pass
    return "New Chat"


class FileChatMessageHistory(BaseChatMessageHistory):
    """Chat message history persisted to a JSON file per session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        os.makedirs(STORE_DIR, exist_ok=True)
        self.file_path = os.path.join(STORE_DIR, f"{session_id}.json")

    @property
    def messages(self) -> list[BaseMessage]:
        if not os.path.exists(self.file_path):
            return []
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return messages_from_dict(data)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages)
        all_messages.extend(messages)
        serialized = [message_to_dict(m) for m in all_messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

    def clear(self) -> None:
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
