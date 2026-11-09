# Price Finder: AI Price Comparison Chatbot

A chatbot that finds the lowest price for a product across Indian retailers (Amazon, Flipkart, Reliance Digital, JioMart), looks up its price history, and gives a buy-now-or-wait estimate. Built with **LangGraph**, **FastAPI**, **Google Gemini**, an **MCP** tool server, and **Firecrawl**.

<!-- Add a screenshot of the chat UI here, e.g. ![Price Finder](docs/screenshot.png) -->

## Features

- **Tool-calling agent.** A LangGraph agent (Gemini) decides when to call tools and answers from live results instead of its outdated training data.
- **MCP tool server.** Two tools run in a separate process over stdio:
  - `lowest_price` finds the cheapest price among approved retailers. It filters out accessories, non-product pages, and coupon or EMI amounts, and returns a structured status (`found`, `variant_unavailable`, `no_match`, `search_error`).
  - `get_price_history` finds all-time-low and average price mentions from price-history sites and adds a buy-or-wait verdict from a scikit-learn model.
- **Human-in-the-loop fallback.** When a product or variant isn't available, the graph pauses (LangGraph `interrupt`) and asks whether to show alternatives. You approve or decline in the UI, and the graph resumes. The pending question is saved in the database, so it survives a page reload or a server restart.
- **Streaming responses.** Answers stream to the browser over Server-Sent Events. Answers keep running on the server if you close the tab, so they are saved to the conversation.
- **Persistent conversations.** Every conversation is a LangGraph thread stored in SQLite. The sidebar lists past chats; open one to continue it, start a new one, or delete one.
- **Single-file UI.** Plain HTML and JavaScript with no build step: safe markdown rendering, dark mode, and a mobile layout.

## How it works

```
Browser (static/index.html)
   │  POST /chat, /chat/resume   ◄── Server-Sent Events (text, status, interrupt, done)
   ▼
FastAPI (app.py) ───────────────► SQLite (LangGraph checkpoints = conversations)
   │
   ▼
LangGraph agent (chatbot.py)
   chat_node (Gemini) ⇄ tools ──► review_fallback (asks you: show alternatives?)
                         │
                         ▼  MCP over stdio
                 MCP server (lowest_price_server.py)
                   ├─ lowest_price ──────► Firecrawl search
                   └─ get_price_history ─► Firecrawl search + recommendation_engine.py (scikit-learn model)
```

**Approval flow.** After the tools run, a `variant_unavailable` or `no_match` result with something to offer routes to the `review_fallback` node, which calls `interrupt(...)`. The API stops streaming and sends an `interrupt` event. When you click **Show alternatives** or **No thanks**, the page calls `POST /chat/resume` and the graph continues from where it paused.

## Project structure

Adjust paths to match your repository.

```
.
├── .env                              # API keys (do not commit)
├── data/
│   └── check_points.db               # conversations (created on first run)
├── src/project_mcp/
│   ├── chatbot.py                    # LangGraph agent: Gemini + MCP tools + approval step
│   └── mcp_server/
│       ├── lowest_price_server.py    # MCP tools: lowest_price, get_price_history
│       ├── recommendation_engine.py  # loads the model, builds features, returns a verdict
│       └── prediction_model/
│           └── price_recommendation_model.pkl
├── app.py                            # FastAPI app
├── static/
│   └── index.html                    # chat UI
└── train_model.py                    # trains the buy-or-wait model
```

## Getting started

### Prerequisites

- Python 3.11 or newer
- A [Google AI Studio](https://aistudio.google.com/) API key (Gemini)
- A [Firecrawl](https://www.firecrawl.dev/) API key

### Install

```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install fastapi uvicorn pydantic python-dotenv \
            langgraph langgraph-checkpoint-sqlite aiosqlite langchain-core \
            langchain-google-genai langchain-mcp-adapters mcp \
            firecrawl-py scikit-learn pandas
```

### Configure

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your-gemini-key
FIRECRAWL_API_KEY=your-firecrawl-key
```

### Run

Run from the project root. The agent starts the MCP server with `python src/project_mcp/mcp_server/lowest_price_server.py`, so that relative path and the `python` on your PATH (your virtual environment) must be right.

```bash
uvicorn app:app --reload --port 8000     # adjust the module path to where app.py lives
```

Open **http://localhost:8000/ui**.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Send a message `{messages, thread_id?}`. Streams Server-Sent Events. Omit `thread_id` to start a new conversation. |
| `POST` | `/chat/resume` | Answer an approval question `{thread_id, approved}`. Streams the continuation. |
| `GET` | `/threads` | List conversations (most recent first). |
| `GET` | `/threads/{id}/messages` | Messages of a conversation, plus any approval question still waiting. |
| `DELETE` | `/threads/{id}` | Delete a conversation permanently. |
| `GET` | `/ui` | The chat page. |
| `GET` | `/` | Health check. |

**Stream events** (each is one JSON object):

| Event | Meaning |
|---|---|
| `{"thread_id": "..."}` | First event; use it to continue the conversation. |
| `{"status": "Checking stores…"}` | The model called a tool; discard any text shown so far. |
| `{"text": "..."}` | A piece of the answer. |
| `{"interrupt": {...}}` | The agent is waiting for your approval. |
| `{"done": true}` / `{"error": "..."}` | End of the turn. |

## Buy-or-wait model

`train_model.py` trains a scikit-learn `HistGradientBoostingClassifier` that predicts whether the price will drop by at least 5% within 30 days. Inputs are the current price, price 7 and 30 days ago, 7 and 30-day rolling means, lowest price so far, discount, month, day of week, a festival-sale flag, category, and store.

Training expects a CSV at `training_data/data.csv` with a `date` column, those feature columns, and the target `wait_for_price_drop`. Run `python train_model.py` and copy the resulting `price_recommendation_model.pkl` into `prediction_model/`. If the file is missing, `recommendation_engine.py` falls back to a simple rule.

> **Important:** the model is a prototype trained on **synthetic demo data**. Its output is a demonstration, not shopping advice. It should be retrained on real recorded prices before being relied on.

## Limitations

- **Prices come from search listings** and can be stale or occasionally wrong. Always confirm on the store page before buying.
- **Price history is approximate.** The recommendation features are estimated from text on price-history pages, not from a continuous price series.
- **Retailer terms apply.** Fetching goes through Firecrawl; check each retailer's terms of use before deploying this publicly.
- **Built for local, single-user use.** SQLite is a good fit here. For many simultaneous users, move the checkpoint store to Postgres.
- **Deleting a conversation is permanent.**

## Roadmap ideas

- Record every lookup as a price snapshot to build a real price history, then retrain the model on it.
- Add tests for price parsing and product matching.
- Add `requirements.txt` with pinned versions and a Dockerfile.

## License

Add a `LICENSE` file (for example MIT) and state it here.
