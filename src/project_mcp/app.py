from fastapi import FastAPI,HTTPException
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel,Field
from project_mcp.chatbot import get_chatbot
from langchain_core.messages import HumanMessage, AIMessage
from contextlib import asynccontextmanager, closing
from fastapi.responses import FileResponse,StreamingResponse
import asyncio
import sqlite3
import uuid
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_FILE = (BASE_DIR.parent / "data" / "check_points.db").resolve()
DB_FILE.parent.mkdir(parents=True, exist_ok=True)
DB_PATH = str(DB_FILE)

class ChatRequest(BaseModel):
    messages: str = Field(description="message to the model")
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class ResumeRequest(BaseModel):                      # NEW: the person's answer to an approval question
    thread_id: str
    approved: bool


agent_container = {}
@asynccontextmanager
async def lifespan(app:FastAPI):
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        await checkpointer.setup()
        agent_container["bot"] = await get_chatbot(checkpointer=checkpointer)
        print("connected to the sqlite database")
        yield
    agent_container.clear()


app = FastAPI(title='low-price',lifespan=lifespan)

@app.get("/")
def health_check():
    return {'status':'ok'}
@app.get('/ui',include_in_schema=False)
def chat_ui():
    return FileResponse(BASE_DIR/"static"/"index.html")


def _get_bot():
    bot = agent_container.get('bot')
    if not bot:
        raise HTTPException(status_code=500,detail="chatbot not initialized")
    return bot


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _text(content) -> str:
    """Message content is a str, or a list of blocks like [{'type': 'text', 'text': '...'}] (thinking blocks are skipped)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else b.get("text", "")
            for b in content
            if isinstance(b, str) or (isinstance(b, dict) and b.get("type") in (None, "text"))
        )
    return ""


def _cfg(thread_id: str) -> dict:
    return {'configurable': {'thread_id': thread_id}}


def _pending(state) -> dict | None:
    """The approval question the graph is paused on, if any (saved in the checkpoint, so it survives restarts)."""
    for task in getattr(state, "tasks", None) or ():
        for item in getattr(task, "interrupts", None) or ():
            if isinstance(item.value, dict):
                return item.value
    return None


# Answers run as background tasks, so closing the tab or reloading mid-answer does not cancel them.
# (A cancelled run loses the whole turn: neither your message nor the answer is saved to the conversation.)
_running: set[asyncio.Task] = set()
_busy_threads: set[str] = set()      # threads with an answer being written right now (can't be deleted)


def _stream_turn(thread_id: str, graph_input) -> StreamingResponse:
    """Run one turn of the graph and stream it as Server-Sent Events:
         {"thread_id"}, then {"status"} / {"text"} ..., then {"interrupt"} if the graph is waiting for approval,
         then {"done"} (or {"error"}).
       graph_input is a new user message, or Command(resume=...) to answer an approval question."""
    bot = _get_bot()
    config = _cfg(thread_id)
    queue: asyncio.Queue = asyncio.Queue()

    async def run():
        waiting_for_tool = False
        _busy_threads.add(thread_id)
        try:
            await queue.put(sse({"thread_id": thread_id}))   # the page needs this to continue the conversation
            async for chunk, metadata in bot.astream(graph_input, config=config, stream_mode="messages"):
                if not isinstance(chunk, AIMessage):
                    continue    # ToolMessage = the tool's raw output. It feeds the model; it is not the answer.
                if chunk.additional_kwargs.get("hitl"):
                    continue    # the approval question: shown as a card, not as streamed text
                if chunk.tool_calls or getattr(chunk, "tool_call_chunks", None):   # plain AIMessages (written by the graph itself) have no tool_call_chunks
                    # the model is calling a tool; anything it said before this ("let me check...") is not the answer
                    if not waiting_for_tool:
                        waiting_for_tool = True
                        await queue.put(sse({"status": "Checking stores…"}))
                    continue
                text = _text(chunk.content)
                if text:
                    waiting_for_tool = False
                    await queue.put(sse({"text": text}))
            pending = _pending(await bot.aget_state(config))
            if pending:                                       # the graph stopped to ask the person something
                await queue.put(sse({"interrupt": pending}))
            await queue.put(sse({"done": True}))   # lets the page tell a finished answer from a dropped connection
        except Exception as e:
            await queue.put(sse({"error": str(e)}))
        finally:
            _busy_threads.discard(thread_id)
            await queue.put(None)

    task = asyncio.create_task(run())
    _running.add(task)                       # keep a reference so the task is not garbage-collected
    task.add_done_callback(_running.discard)

    async def event_generator():
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/chat")
async def chat_bot(payload : ChatRequest):
    _get_bot()
    return _stream_turn(payload.thread_id, {'messages': [HumanMessage(content=payload.messages)]})


@app.post("/chat/resume")      # NEW: Approve / Decline for a conversation that is waiting for approval
async def chat_resume(payload : ResumeRequest):
    bot = _get_bot()
    if payload.thread_id in _busy_threads:
        raise HTTPException(status_code=409, detail="This conversation is still answering.")
    if not _pending(await bot.aget_state(_cfg(payload.thread_id))):
        raise HTTPException(status_code=409, detail="Nothing is waiting for your approval in this conversation.")
    return _stream_turn(payload.thread_id, Command(resume={"approved": payload.approved}))


# ---------------------------------------------------------------------------------------------
# Previous conversations (for the sidebar). Read straight from your existing checkpoint DB:
# every thread_id that has saved messages is a conversation; nothing extra is stored.
# ---------------------------------------------------------------------------------------------
_titles: dict[str, str] = {}   # thread_id -> sidebar title (the first message never changes, so cache it)
MAX_THREADS = 100


def _ckpt_time(checkpoint_id: str) -> float | None:
    """LangGraph checkpoint ids are time-ordered UUIDv6, so they double as a 'last active' timestamp."""
    try:
        h = uuid.UUID(checkpoint_id).hex
        ts = (int(h[:8], 16) << 28) | (int(h[8:12], 16) << 12) | int(h[13:16], 16)
        unix = (ts - 0x01B21DD213814000) / 1e7
        return unix if 1.5e9 < unix < 4.1e9 else None
    except (ValueError, TypeError):
        return None


def _saved_threads() -> list[tuple[str, str]]:
    """(thread_id, latest checkpoint id), most recently active first."""
    try:
        with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
            return conn.execute(
                "SELECT thread_id, MAX(checkpoint_id) FROM checkpoints WHERE checkpoint_ns = '' "
                "GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC LIMIT ?", (MAX_THREADS,)
            ).fetchall()
    except sqlite3.OperationalError:
        return []


async def _load(bot, thread_id: str) -> tuple[list[dict], dict | None]:
    """(your messages + the bot's final answers, the approval question still waiting for you or None).
       Tool calls, tool output and 'let me check…' turns are hidden."""
    state = await bot.aget_state(_cfg(thread_id))
    out = []
    for m in (state.values or {}).get('messages', []):
        text = _text(m.content).strip()
        if m.type == 'human' and text:
            out.append({'role': 'user', 'content': text})
        elif m.type == 'ai' and text and not getattr(m, 'tool_calls', None):
            out.append({'role': 'assistant', 'content': text})
    return out, _pending(state)


@app.get("/threads")
async def list_threads():
    bot = _get_bot()
    threads = []
    for thread_id, checkpoint_id in _saved_threads():
        if thread_id not in _titles:
            msgs, _ = await _load(bot, thread_id)
            first = " ".join(next((m['content'] for m in msgs if m['role'] == 'user'), '').split())
            if not first:
                continue
            _titles[thread_id] = first if len(first) <= 48 else first[:47].rstrip() + "…"
        threads.append({'thread_id': thread_id, 'title': _titles[thread_id],
                        'updated_at': _ckpt_time(checkpoint_id)})
    return {'threads': threads}


@app.get("/threads/{thread_id}/messages")
async def thread_messages(thread_id: str):
    msgs, pending = await _load(_get_bot(), thread_id)
    if not msgs:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {'messages': msgs, 'pending': pending}


def _thread_exists(thread_id: str) -> bool:
    try:
        with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
            return conn.execute("SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1", (thread_id,)).fetchone() is not None
    except sqlite3.OperationalError:
        return False


@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    bot = _get_bot()
    if thread_id in _busy_threads:
        # deleting now would let the unfinished answer re-create the conversation
        raise HTTPException(status_code=409, detail="This conversation is still answering. Try again in a moment.")
    if not _thread_exists(thread_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    deleter = getattr(getattr(bot, "checkpointer", None), "adelete_thread", None)
    if deleter:
        await deleter(thread_id)                       # LangGraph's own cleanup
    else:                                              # older checkpointer versions: remove the rows directly
        with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn, conn:
            conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
    _titles.pop(thread_id, None)
    return {"ok": True}