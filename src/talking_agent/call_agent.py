"""
call_agent.py
LangGraph-based conversational "brain" for the outbound payment-reminder
phone call. Twilio ConversationRelay (see voice_server.py) handles the
actual voice channel -- speech-to-text in, text-to-speech out. This module
only ever deals in plain text: given the conversation so far, decide what
the agent should say next, and expose a few tools the LLM can call to
record how the call ended (customer promised a date, refused, etc).

Requires:
    pip install langgraph langchain-groq langchain-core --break-system-packages
    GROQ_API_KEY set in your environment / .env
"""

import os
import re
import json
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import database as db


# ---------------------------------------------------------------------
# Conversation state carried across the whole call
# ---------------------------------------------------------------------
class CallState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    transaction_id: int
    borrower_name: str
    total_amount: float
    due_date: str
    items_summary: str
    outcome: Optional[str]        # "promised" | "refused" | "ended" | None while ongoing
    promised_date: Optional[str]
    call_should_end: bool


# ---------------------------------------------------------------------
# Tools the LLM can call mid-conversation to record what happened
# ---------------------------------------------------------------------
@tool
def record_promise(promised_date: str, note: str = "") -> str:
    """Call this once the customer commits to a specific date they will pay by.
    promised_date must be in YYYY-MM-DD format."""
    return json.dumps({"outcome": "promised", "promised_date": promised_date, "note": note})


@tool
def record_refusal(reason: str = "") -> str:
    """Call this if the customer clearly refuses to pay or disputes the amount owed."""
    return json.dumps({"outcome": "refused", "note": reason})


@tool
def end_call(summary: str = "") -> str:
    """Call this once the conversation has reached a natural conclusion -- you've
    said your goodbye and it's time to hang up."""
    return json.dumps({"outcome": "ended", "note": summary})


@tool
def lookup_account_details(transaction_id: int) -> str:
    """Look up this borrower's account fresh from the shop's records. Call this
    any time the customer asks a question you're not 100% sure of the exact
    answer to -- e.g. 'what exactly did I buy', 'how much do I owe', 'when
    was this due', 'have I already paid this', or 'has anyone else already
    called/messaged me about this'. Always pass the transaction_id given to
    you in your instructions -- never guess it. Returns the authoritative,
    up-to-date figures so you never make up numbers."""
    for tx in db.get_all_transactions():
        if tx["id"] == transaction_id:
            items = ", ".join(
                f"{it['product_name']} x{it['quantity']} @ Rs.{it['price_per_unit']:,.2f} "
                f"= Rs.{it['item_total']:,.2f}"
                for it in tx["items"]
            )
            return json.dumps({
                "borrower_name": tx["borrower_name"],
                "items_purchased": items,
                "purchase_date": tx["purchase_date"],
                "due_date": tx["due_date"],
                "total_amount": tx["total_amount"],
                "is_already_paid": bool(tx["is_paid"]),
                "whatsapp_reminders_sent": tx["reminder_count"],
                "previous_call_outcome": tx.get("call_outcome"),
                "previous_promised_date": tx.get("call_promised_date"),
            })
    return json.dumps({"error": f"No account found for transaction_id={transaction_id}"})


TOOLS = [record_promise, record_refusal, end_call, lookup_account_details]

# Sent as a synthetic first "customer" turn so the LLM knows to speak first and
# open the call, rather than silently waiting for the other party. Used by both
# generate_greeting() (text-only test harness, see __main__ below) and
# voice_server.py's /chat/completions handler (assistant-speaks-first-with-
# model-generated-message mode), so the greeting behaves identically either way.
CALL_CONNECTED_HINT = "[CALL CONNECTED -- greet the customer and start the conversation]"


SYSTEM_PROMPT_TEMPLATE = """You are Riya, a warm, polite phone assistant calling on \
behalf of a small local shop to remind {borrower_name} about a pending payment \
that is due TODAY.

Account reference (transaction_id): {transaction_id}

Details of the purchase (as of when this call started):
{items_summary}
Total amount due: Rs.{total_amount:,.2f}
Due date: {due_date}

Your goals, in order:
1. Your opening line already asked "am I speaking with {borrower_name}?" --
   read their reply to that first:
   - If they confirm (yes / speaking / that's me), continue to step 2.
   - If they say this is the wrong person, apologize briefly and call
     end_call -- do not discuss the payment with someone who isn't the
     borrower.
   - If they sound unsure or ask who's calling, briefly explain you're
     calling on behalf of the shop about their account, then re-confirm.
2. Remind them of the amount due and briefly what it was for -- summarize
   naturally like a person would, don't read it out like a list.
3. Ask if they're able to pay today. Be understanding, never pushy or
   threatening.
4. If they can't pay today, you MUST ask for a specific date they can
   commit to ("what day works for you?" / "when can you manage it by?")
   before doing anything else. Do not accept a vague answer like "not
   today" or "later" as final -- keep gently asking until you get an actual
   date, then call record_promise with that date. NEVER call end_call after
   a "can't pay today" response without first calling record_promise --
   ending the call without a committed date leaves the shop with no
   follow-up plan at all, which is worse than any date, even a distant one.
5. If they refuse to pay outright or dispute the charge (not the same as
   "can't pay today, but here's when I can"), stay calm and polite, and
   call record_refusal with a short reason.
6. Once you have a clear outcome -- record_promise was called with a date,
   OR record_refusal was called -- thank them warmly and call end_call.
   Never call end_call before one of those two has already been called,
   unless step 1 established you're speaking to the wrong person.

Answering the customer's own questions:
- The customer may ask you things beyond the script -- "what exactly did I
  buy", "how much exactly do I owe", "when was this due", "have I already
  paid", "did someone already message me about this". Never guess or make up
  numbers. Call lookup_account_details with transaction_id={transaction_id}
  to get the current, authoritative answer, then answer them naturally in
  conversation.
- If they ask something totally outside what your account lookup or purchase
  details can answer (e.g. store hours, other products, unrelated
  complaints), be honest that you don't have that information on this call
  and offer to have the shop follow up, rather than guessing.

Style rules (this is a REAL-TIME VOICE CALL, not a chat):
- Keep every response to 1-3 short sentences.
- Speak naturally and warmly, like a real person on the phone, never like a
  script or a bulleted list.
- If the customer sounds confused about who's calling, briefly and kindly
  explain again.
- Never sound robotic, repetitive, or scolding -- the goal is a friendly
  nudge, not a collections threat.
"""


def _get_llm():
    model_name = os.getenv("CALL_AGENT_MODEL", "openai/gpt-oss-120b")
    llm = ChatGroq(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=1,
        # A live phone call can't tolerate the ~15s+ that "medium" reasoning
        # effort added (borrower hangs up from silence long before the
        # model finishes "thinking"). "low" keeps gpt-oss-120b's quality
        # while cutting hidden reasoning tokens down drastically. If turns
        # are still too slow after this, switch CALL_AGENT_MODEL in .env to
        # a fast non-reasoning model instead (e.g. llama-3.1-8b-instant).
        max_tokens=350,   # a 1-3 sentence spoken reply + a tool call rarely needs more
        reasoning_effort="low",
    )
    return llm.bind_tools(TOOLS)


def build_graph():
    llm = _get_llm()

    def call_model(state: CallState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def route(state: CallState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    def apply_tool_results(state: CallState):
        """After a tool runs, read its JSON result and update outcome/end flag."""
        updates = {}
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage):
                try:
                    payload = json.loads(msg.content)
                except (json.JSONDecodeError, TypeError):
                    continue
                if payload.get("outcome") == "promised":
                    updates["outcome"] = "promised"
                    updates["promised_date"] = payload.get("promised_date")
                elif payload.get("outcome") == "refused":
                    updates["outcome"] = "refused"
                elif payload.get("outcome") == "ended":
                    updates["call_should_end"] = True
                break
            if isinstance(msg, AIMessage):
                break
        return updates

    graph = StateGraph(CallState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("apply_tool_results", apply_tool_results)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "apply_tool_results")
    graph.add_edge("apply_tool_results", "agent")

    return graph.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def build_system_prompt(transaction: dict) -> str:
    items_summary = ", ".join(
        f"{it['product_name']} x{it['quantity']}" for it in transaction["items"]
    )
    return SYSTEM_PROMPT_TEMPLATE.format(
        transaction_id=transaction["id"],
        borrower_name=transaction["borrower_name"],
        items_summary=items_summary,
        total_amount=transaction["total_amount"],
        due_date=transaction["due_date"],
    )


def create_initial_state(transaction: dict) -> CallState:
    items_summary = ", ".join(
        f"{it['product_name']} x{it['quantity']}" for it in transaction["items"]
    )
    return {
        "messages": [SystemMessage(content=build_system_prompt(transaction))],
        "transaction_id": transaction["id"],
        "borrower_name": transaction["borrower_name"],
        "total_amount": transaction["total_amount"],
        "due_date": transaction["due_date"],
        "items_summary": items_summary,
        "outcome": None,
        "promised_date": None,
        "call_should_end": False,
    }


def generate_greeting(state: CallState) -> tuple[CallState, str]:
    """
    Kick off the call with an opening line, before the customer says
    anything. Deliberately does NOT call the LLM: this is the single
    moment in the whole call least tolerant of delay -- the borrower is
    holding a silent phone right after picking up, and a slow/reasoning
    model (gpt-oss-120b with reasoning_effort) was taking 15+ seconds here,
    long enough that the borrower hung up before ever hearing a word (see
    the "Call has already ended" / error 90018 failures). A fast, templated
    line goes out instantly and costs nothing; the LLM only takes over
    starting with the borrower's first reply, where a normal conversational
    pause is expected and tolerated.
    """
    greeting = (
        f"Hi, am I speaking with {state['borrower_name']}? "
        f"I'm calling on behalf of the shop about a pending payment."
    )
    # Keep the graph's conversation history consistent: record what was
    # actually said as an AIMessage, so the LLM has full context on the
    # borrower's very first reply.
    state["messages"].append(HumanMessage(content=CALL_CONNECTED_HINT))
    state["messages"].append(AIMessage(content=greeting))
    return state, greeting


def generate_greeting_via_llm(state: CallState) -> tuple[CallState, str]:
    """
    The original LLM-generated greeting, kept for reference / rollback.
    NOT used by default -- see generate_greeting() above for why.
    """
    state["messages"].append(HumanMessage(content=CALL_CONNECTED_HINT))
    result = get_graph().invoke(state)
    return result, _extract_latest_ai_text(result["messages"])


def stream_user_turn(state: CallState, user_text: str):
    """
    Streaming counterpart to handle_user_turn(). Instead of blocking until
    the whole reply is generated, this yields spoken sentences as soon as
    the LLM produces them, so a caller (e.g. telnyx_voice_server.py) can
    start playing TTS on the first sentence immediately instead of waiting
    for the full turn -- including any tool-call round trips inside the
    graph (lookup_account_details, record_promise, etc) -- to finish.

    This is the generator telnyx_voice_server.py's comment referred to as
    "half-finished" -- it was called but never implemented. This is the
    implementation.

    Yields:
        ("sentence", text)  for each complete sentence, in order, as soon
                             as it's formed.
        ("done", (state, should_end))  exactly once, at the end.

    Only the FINAL tool-call-free agent turn is ever spoken -- earlier
    turns in the same graph.invoke() that end in a tool call (e.g. the
    model calling lookup_account_details before answering) produce no
    user-facing text and are silently skipped; the buffer is reset each
    time a tool-call chunk is seen so no partial/abandoned tool-call text
    is ever spoken by mistake.

    NOTE: this relies on LangGraph's stream_mode=["messages", "values"]
    behaviour (each yielded item is a (mode, chunk) tuple; "messages" chunks
    are (message_chunk, metadata) pairs from any node's LLM call; "values"
    chunks are full state snapshots after each step). Verify this shape
    against your installed langgraph version -- the streaming API has
    changed across versions, same caveat the other transport files give for
    their SDKs.
    """
    state["messages"].append(HumanMessage(content=user_text))
    graph = get_graph()

    buffer = ""
    final_state = state

    for stream_mode, chunk in graph.stream(state, stream_mode=["messages", "values"]):
        if stream_mode == "values":
            # Full state snapshot after each graph step -- keep the latest,
            # it's what carries outcome/promised_date/call_should_end once
            # apply_tool_results has run.
            final_state = chunk
            continue

        # stream_mode == "messages": chunk is (message_chunk, metadata)
        message_chunk, metadata = chunk
        if metadata.get("langgraph_node") != "agent":
            continue
        if getattr(message_chunk, "tool_calls", None) or getattr(message_chunk, "tool_call_chunks", None):
            # This agent turn is calling a tool, not speaking to the
            # customer -- discard anything buffered from it (there
            # shouldn't be any in practice, but don't risk speaking a
            # stray fragment) and wait for the turn that actually replies.
            buffer = ""
            continue

        piece = message_chunk.content if isinstance(message_chunk.content, str) else ""
        if not piece:
            continue
        buffer += piece

        while True:
            m = re.search(r"[.!?](\s+|$)", buffer)
            if not m:
                break
            sentence = buffer[: m.end()].strip()
            buffer = buffer[m.end():]
            if sentence:
                yield ("sentence", sentence)

    leftover = buffer.strip()
    if leftover:
        yield ("sentence", leftover)

    should_end = bool(final_state.get("call_should_end", False))
    yield ("done", (final_state, should_end))


def handle_user_turn(state: CallState, user_text: str) -> tuple[CallState, str, bool]:
    """
    Feed one customer utterance into the graph and get back the agent's
    spoken reply plus whether the call should end now.
    """
    state["messages"].append(HumanMessage(content=user_text))
    result = get_graph().invoke(state)
    reply_text = _extract_latest_ai_text(result["messages"])
    should_end = result.get("call_should_end", False)
    return result, reply_text, should_end


def _extract_latest_ai_text(messages) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
        if isinstance(msg, AIMessage) and isinstance(msg.content, list):
            text_parts = [
                b.get("text", "") for b in msg.content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            joined = " ".join(p for p in text_parts if p)
            if joined.strip():
                return joined
    return "Sorry, could you repeat that?"


if __name__ == "__main__":
    # Quick manual test in the terminal (text only, no Twilio) -- requires
    # GROQ_API_KEY to be set.
    sample_transaction = {
        "id": 1,
        "borrower_name": "Aman Kumar",
        "due_date": "2026-08-31",
        "total_amount": 590.0,
        "items": [
            {"product_name": "Rice 5kg bag", "quantity": 2},
            {"product_name": "Soap", "quantity": 3},
        ],
    }
    state = create_initial_state(sample_transaction)
    state, greeting = generate_greeting(state)
    print("AGENT:", greeting)
    while True:
        user_input = input("YOU: ")
        state, reply, ended = handle_user_turn(state, user_input)
        print("AGENT:", reply)
        if ended:
            print(f"--- call ended. outcome={state.get('outcome')} promised_date={state.get('promised_date')} ---")
            break