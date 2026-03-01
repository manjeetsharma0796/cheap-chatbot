# AI Chatbot with Persistent Memory

A Streamlit-based chatbot frontend for conversing with NVIDIA's ChatNVIDIA LLM model. Features persistent conversation memory across sessions and restarts.

## 📁 Project Structure

```
lang_py/
├── app.py                    # Main Streamlit web application
├── memory/
│   ├── __init__.py          # Module exports
│   ├── history.py           # FileChatMessageHistory implementation
│   └── store/               # Directory where conversation JSON files are stored
│       └── <session_id>.json # One file per conversation session
├── llm/                     # Python virtual environment
├── chat_runnable.py         # Legacy: Multi-user conversation example
├── nvidia.py                # Legacy: Basic LLM integration example
├── pyproject.toml           # Project metadata
└── README.md                # This file
```

## 🔧 Components

### `app.py` (Streamlit UI)
The main application entry point. Provides a web-based chat interface with:
- **Real-time streaming** chat responses with live cursor indicator (`▌`)
- **Session management** sidebar — create, switch, or clear conversation threads
- **Persistent memory** — automatically loads conversation history from JSON files
- **Message display** — user messages and AI responses displayed in chat bubbles

Key features:
- Each session gets a unique ID for tracking separate conversations
- Sessions are isolated — multiple users can have independent chats
- Past messages are displayed when you revisit a session

### `memory/` Module
Handles all conversation history persistence to JSON files.

#### `memory/history.py`
Implements `FileChatMessageHistory` — a custom chat history backend that:
- **Reads** messages from `memory/store/<session_id>.json` via the `messages` property
- **Writes** new messages to the JSON file when `add_messages()` is called
- **Clears** history when `clear()` is called
- Uses LangChain's `BaseChatMessageHistory` interface for compatibility

#### `memory/store/`
A directory containing one `.json` file per conversation session:
```json
[
  {
    "type": "human",
    "data": {
      "content": "Hi I'm Pandit ji.",
      ...
    }
  },
  {
    "type": "ai",
    "data": {
      "content": "Hello Pandit ji! How can I help?",
      ...
    }
  }
]
```
⚠️ **Note:** Each JSON file is named after its `session_id` (e.g., `user_1.json`, `abc123.json`).

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- Virtual environment already set up (included in `llm/`)
- Dependencies installed (LangChain, Streamlit, NVIDIA endpoints)

### Installation

If packages are missing, install them:
```bash
python -m pip install streamlit langchain-core langchain-nvidia-ai-endpoints
```

### Running the App

```bash
python -m streamlit run app.py
```

The app will open automatically at:
```
Local URL: http://localhost:8501
```

## 💬 How Sessions Work

### Creating a Session
- When you first open the app, a random 8-character session ID is generated automatically
- You can type your own session ID in the sidebar to use a custom identifier
- Click **"New Session"** to generate a new conversation thread

### Example Session Usage

**User 1 (Pandit):**
```
Session ID: pandit_1
Message 1: "Hi I'm Pandit ji."
Message 2: "What's the weather?"
→ History saved to: memory/store/pandit_1.json
```

**User 2 (Srijan):**
```
Session ID: srijan_1
Message 1: "Hi I'm Srijan Dubey."
Message 2: "Tell me about AI."
→ History saved to: memory/store/srijan_1.json
```

Both users maintain completely separate conversations.

### Accessing Existing Sessions
1. Enter the session ID in the sidebar (e.g., `pandit_1`)
2. Previous messages will load automatically
3. New messages continue the conversation

### Clearing History
- Click **"Clear History"** in the sidebar to delete the current session's conversation
- Only affects the current session; other sessions remain untouched

## 🔐 Memory Persistence

### How It Works
1. **On each message**, the `FileChatMessageHistory` automatically:
   - Reads all previous messages from the JSON file
   - Adds the new user message and AI response
   - Writes the complete conversation back to JSON

2. **When restarting the app**:
   - Enter the same session ID
   - All previous messages reload from the JSON file
   - You can continue the conversation as if you never left

### Storage Location
```
memory/store/
├── pandit_1.json
├── srijan_1.json
├── user_abc123.json
└── ... (one file per session)
```

## 🤖 LLM Configuration

The app uses NVIDIA's ChatNVIDIA model. To change settings, edit `app.py`:

```python
API_KEY = "your-nvidia-api-key-here"
MODEL   = "mistralai/mixtral-8x7b-instruct-v0.1"  # Or another NVIDIA model
```

### Supported Models
- `mistralai/mixtral-8x7b-instruct-v0.1` (recommended, fast)
- `mistralai/mistral-large-2-instruct` (powerful, slower)
- `meta/llama-3.1-70b-instruct` (good for reasoning)
- See [NVIDIA Endpoints](https://www.nvidia.com/en-us/ai-on-nvidia/) for more

## 📝 Legacy Scripts

### `chat_runnable.py`
Multi-user conversation example without web UI. Demonstrates:
- How to use `RunnableWithMessageHistory` directly
- Managing multiple sessions in Python code
- Tool calling (if model supports it)

### `nvidia.py`
Basic LLM integration example showing:
- Simple prompt → LLM → output pipeline
- Streaming responses
- No memory between calls

## 🛠️ Development

### Adding Features

**Custom session naming:**
```python
config = {"configurable": {"session_id": f"user_{user_id}_{timestamp}"}}
```

**Switching to a different memory backend (e.g., SQLite):**
Replace `FileChatMessageHistory` with a new class implementing LangChain's `BaseChatMessageHistory`.

**Adding tools/functions:**
Use LangGraph for agent-like workflows where the LLM makes decisions and calls functions repeatedly.

## ⚠️ Important Notes

- **Memory is in JSON files** — suitable for development and small deployments. For production, consider:
  - SQLite (simple, file-based)
  - PostgreSQL (scalable)
  - Redis (fast caching)

- **API keys:** Never commit API keys to version control. Use environment variables:
  ```bash
  set NVIDIA_API_KEY=your-key-here
  ```

- **Rate limits:** NVIDIA endpoints have usage limits. Check your account for current quotas.

## 📚 Related Concepts

- **Streaming:** Messages are streamed to the UI in real-time with visual feedback
- **Message History:** Automatically loaded and passed to the LLM for context awareness
- **Session isolation:** Each session is independent; sessions don't interfere with each other
- **Stateless restarts:** Because memory is persistent in JSON, the app is stateless — restarting doesn't lose data

## 🎯 Next Steps

1. **Run the app** and test with different sessions
2. **Inspect the JSON files** in `memory/store/` to see how conversations are stored
3. **Modify the LLM model** or API settings as needed
4. **Deploy** to production using Streamlit Cloud, Docker, or your preferred platform

---

**Questions?** Refer to [LangChain Docs](https://python.langchain.com/) or [Streamlit Docs](https://docs.streamlit.io/).
