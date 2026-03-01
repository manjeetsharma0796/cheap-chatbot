# Chat App

## Folder Structure

```
lang_py/
├── app.py                  # Streamlit app entry point
├── memory/
│   ├── __init__.py         # Memory module exports
│   ├── history.py          # File-based chat history implementation
│   └── store/              # Per-session JSON chat history files
├── learning/
│   ├── chat_runnable.py    # Legacy learning script
│   ├── main.py             # Legacy learning script
│   └── nvidia.py           # Legacy learning script
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

## How To Run

### 1) Sync dependencies

```bash
uv sync
```

### 2) Start the app

```bash
uv run streamlit run app.py
```

### 3) Open in browser

```text
http://localhost:8501
```
