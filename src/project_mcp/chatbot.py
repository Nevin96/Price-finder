from langgraph.graph.state import StateGraph,START,END
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from typing import TypedDict,Annotated
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,AIMessage,ToolMessage   # CHANGED: + AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.types import interrupt, Command                                                      # NEW
import asyncio
import json                                                                                         # NEW
import re                                                                                           # NEW
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]


# =====================================================================================
# NEW: human approval ("HITL") when the product is not available.
#
# lowest_price returns {"status": "variant_unavailable" | "no_match", ...} when it cannot find the product.
# Instead of letting the model improvise, the graph PAUSES here and asks the person whether they want to see
# alternatives. Their answer (Approve / Decline) arrives through /chat/resume in app.py and the graph continues.
# The pause is saved by the checkpointer, so it survives a page reload or a server restart.
# =====================================================================================
FALLBACK_STATUSES = {"variant_unavailable", "no_match"}
APPROVE_TEXT = "Yes, show me alternatives."      # shown as the person's message in the chat history
DECLINE_TEXT = "No thanks."
DECLINED_REPLY = "No problem, I won't look for alternatives. Tell me if you'd like to search for something else."


def _tool_json(message) -> dict | None:
    """The lowest_price result as a dict (tool content is a str, or a list of text blocks)."""
    content = message.content
    if isinstance(content, list):
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _fallback_payload(state) -> dict | None:
    """A 'not available' result among the tool results of the step that just ran, if any."""
    for m in reversed(state["messages"]):
        if not isinstance(m, ToolMessage):
            break                                   # reached the model's tool-call message: stop
        data = _tool_json(m)
        if data and data.get("status") in FALLBACK_STATUSES:
            return data
    return None


def _size_gb(text: str) -> int | None:
    m = re.search(r"(\d+)\s*(gb|tb)", text, re.IGNORECASE)
    return int(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1) if m else None


def _closest_variant(requested: str, variants: list[str]) -> str:
    """The available storage size nearest to the one asked for (larger wins a tie)."""
    want = _size_gb(requested)
    sized = [(v, _size_gb(v)) for v in variants if _size_gb(v)]
    if not sized:
        return variants[0]
    if want is None:
        return min(sized, key=lambda s: s[1])[0]
    return min(sized, key=lambda s: (abs(s[1] - want), -s[1]))[0]


def _approval_request(data: dict) -> dict:
    """What the person is asked: a message plus the options. Empty options = nothing to offer, so don't ask."""
    if data["status"] == "variant_unavailable":
        requested = data.get("requested_product", "that version")
        base = data.get("base_product") or requested
        variants = [str(v) for v in data.get("available_variants", [])]
        if not variants:
            return {"options": []}
        closest = _closest_variant(requested, variants)
        return {
            "kind": "alternatives", "status": data["status"],
            "message": f"I couldn't find “{requested}”. “{base}” is available in {', '.join(variants)}. "
                       f"Would you like me to check the closest option ({closest}) instead?",
            "options": [f"{base} {v}" for v in variants],
        }
    product = data.get("product", "that product")
    alternatives = [str(a) for a in data.get("alternatives", [])]
    return {
        "kind": "alternatives", "status": data["status"],
        "message": f"I couldn't find an exact match for “{product}” at Amazon, Flipkart, Reliance Digital or JioMart. "
                   "Would you like to see similar products?",
        "options": alternatives,
    }


def _history_text(request: dict) -> str:
    """The same question written into the conversation, so the chat history reads naturally (and the model sees the options)."""
    text = request["message"]
    if request["status"] == "no_match" and request["options"]:
        text += "\nSimilar products I found:\n" + "\n".join(f"- {o}" for o in request["options"])
    return text


def route_after_tools(state: ChatState) -> str:
    data = _fallback_payload(state)
    if data and _approval_request(data)["options"]:
        return "review_fallback"
    return "chat_node"


def review_fallback(state: ChatState):
    request = _approval_request(_fallback_payload(state))
    decision = interrupt(request)       # PAUSE: the graph stops here until the person answers; this returns their answer
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    question = AIMessage(content=_history_text(request), additional_kwargs={"hitl": True})
    if approved:
        return Command(goto="chat_node", update={"messages": [question, HumanMessage(content=APPROVE_TEXT)]})
    return Command(goto=END, update={"messages": [question, HumanMessage(content=DECLINE_TEXT), AIMessage(content=DECLINED_REPLY)]})


async def get_chatbot(checkpointer):
    client = MultiServerMCPClient(
                    {
                    'lowest_price':{
                        'command':'python',
                        'args':['src/project_mcp/mcp_server/lowest_price_server.py'],
                        'transport':'stdio'
                    },
                    }
                )
                
    tools = await client.get_tools()
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    # model = ChatOllama(model='llama3.2:latest')
    model_with_tools = model.bind_tools(tools)
    SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a price comparison assistant. "
        "Your internal training data is outdated regarding recent products and models. "
        "Whenever a user names a product, smartphone, gadget, or asks about prices/availability, "
        "you MUST call the 'lowest_price' tool to fetch current live data before answering. "
        "Never state a product does not exist without calling the tool first. "
        # NEW: how to handle the approval step
        "If a product is not available, the system asks the user whether they want alternatives. "
        "When the user then says yes, call 'lowest_price' for the closest option named in your previous message "
        "(or for up to 2 of the similar products you listed) and present the results briefly. "
        "Never suggest alternatives that were not listed."
    )
    )

    def chat_node(state:ChatState):
        messages = [SYSTEM_PROMPT] +  state['messages']
        response = model_with_tools.invoke(messages)
        return {'messages':[response]}



    graph = StateGraph(ChatState)


    graph.add_node('chat_node',chat_node)
    graph.add_node('tools',ToolNode(tools))
    graph.add_node('review_fallback',review_fallback)                                  # NEW


    graph.add_edge(START,'chat_node')
    graph.add_conditional_edges('chat_node',tools_condition)
    # CHANGED: after the tools run, a 'not available' result goes to the approval step instead of straight back to the model
    graph.add_conditional_edges('tools',route_after_tools,{'review_fallback':'review_fallback','chat_node':'chat_node'})

    return graph.compile(checkpointer=checkpointer)