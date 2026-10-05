"""
voice_server.py
FastAPI app used by Vapi for the outbound "payment due today" phone call.

Exposes:
    POST /chat/completions   Vapi's "custom-llm" model provider calls this on
                              every conversational turn (OpenAI chat-completions
                              request/response format). We run the message
                              history it sends through call_agent.py's existing
                              LangGraph brain -- completely unchanged -- and
                              hand back its reply in the same OpenAI shape.
                              Also used for the opening line, since the
                              assistant is configured with firstMessageMode=
                              "assistant-speaks-first-with-model-generated-
                              message" (see outbound_caller.py).
    POST /vapi/webhook       Receives Vapi's end-of-call-report. Used as a
                              safety net: if a call ends (hangup, silence
                              timeout, error) without record_promise /
                              record_refusal / end_call already having saved
                              an outcome via /chat/completions, this records
                              it as "disconnected" so it isn't silently lost.

Run with (from your project root):
    uv run uvicorn voice_server:app --host 0.0.0.0 --port 8000

Vapi needs a PUBLIC https URL to reach this server -- your own machine isn't
reachable directly. While developing, expose port 8000 with e.g. ngrok:
    ngrok http 8000
Then set PUBLIC_BASE_URL in your .env to the printed https URL, e.g.:
    PUBLIC_BASE_URL=https://abcd1234.ngrok-free.app

Field names below follow Vapi's public API docs (docs.vapi.ai). Vapi's API
surface evolves, so if requests aren't reaching this server as expected,
check the current schema there first -- particularly around the "server"/
"serverUrl" webhook config, which has moved around across API versions.
"""

import os
import time
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import database as db
import call_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice-server")

app = FastAPI()
db.init_db()

VAPI_SERVER_SECRET = os.getenv("VAPI_SERVER_SECRET")


def _load_transaction(transaction_id) -> dict | None:
    if not transaction_id:
        return None
    try:
        transaction_id = int(transaction_id)
    except (TypeError, ValueError):
        return None
    for tx in db.get_all_transactions():
        if tx["id"] == transaction_id:
            return tx
    return None


def _to_langchain_messages(vapi_messages: list) -> list:
    """
    Maps the OpenAI-format messages Vapi sends into LangChain message objects.
    Only system/user/assistant roles are handled -- call_agent.py's own tools
    (record_promise/record_refusal/end_call) run and resolve entirely inside a
    single graph.invoke() call on our side and are never exposed to Vapi, so
    there's no "tool" role coming back from Vapi to translate.
    """
    converted = []
    for m in vapi_messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        # any other role is unexpected from a custom-llm request and is dropped
    return converted


def _openai_chat_response(model: str, text: str) -> dict:
    """Wraps a plain-text reply in a minimal, non-streaming OpenAI
    chat-completions response -- the shape Vapi's custom-llm provider expects.
    (Vapi also accepts a streamed SSE response for lower latency; this project
    keeps it simple with a single JSON response, which Vapi explicitly supports
    too -- see docs.vapi.ai/customization/custom-llm.)"""
    return {
        "id": "chatcmpl-khata",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


@app.post("/chat/completions")
async def chat_completions(request: Request):
    """Vapi's custom-llm model provider hits this on every conversational turn
    (including the very first, to generate the opening line), with the full
    message history so far."""
    body = await request.json()
    vapi_messages = body.get("messages", [])
    call_info = body.get("call") or {}
    metadata = call_info.get("metadata") or body.get("metadata") or {}
    transaction_id = metadata.get("transaction_id")
    model_name = body.get("model", "call-agent-langgraph")

    transaction = _load_transaction(transaction_id)
    if transaction is None:
        logger.error(f"No transaction found for id={transaction_id!r} in custom-llm request")
        return JSONResponse(_openai_chat_response(
            model_name, "Sorry, I couldn't find your account details. Goodbye."
        ))

    lc_messages = _to_langchain_messages(vapi_messages)
    has_conversation_started = any(
        m.get("role") in ("user", "assistant") for m in vapi_messages
    )
    if not has_conversation_started:
        # First turn (greeting): nudge the model to speak first, same trick
        # call_agent.generate_greeting() uses in the text-only test harness.
        lc_messages.append(HumanMessage(content=call_agent.CALL_CONNECTED_HINT))

    state = call_agent.create_initial_state(transaction)
    # Vapi sends the authoritative running transcript (including the system
    # prompt we set at call creation) every turn, so we use that directly
    # instead of layering our own copy on top of it.
    state["messages"] = lc_messages

    result = call_agent.get_graph().invoke(state)
    reply_text = call_agent._extract_latest_ai_text(result["messages"])

    if result.get("outcome") is not None or result.get("call_should_end"):
        outcome = result.get("outcome") or "ended"
        db.save_call_outcome(
            transaction_id=transaction["id"],
            outcome=outcome,
            promised_date=result.get("promised_date"),
        )
        logger.info(f"Saved call outcome for transaction #{transaction['id']}: {outcome}")

    return JSONResponse(_openai_chat_response(model_name, reply_text))


@app.post("/vapi/webhook")
async def vapi_webhook(request: Request):
    """Safety net: if a call ends (hangup, silence timeout, error) without our
    own tools already saving an outcome via /chat/completions, record it as
    'disconnected' so it doesn't silently vanish from the ledger."""
    if VAPI_SERVER_SECRET:
        if request.headers.get("x-vapi-secret") != VAPI_SERVER_SECRET:
            return JSONResponse({"status": "error", "message": "invalid server secret"}, status_code=401)

    payload = await request.json()
    message = payload.get("message", {}) or {}
    if message.get("type") != "end-of-call-report":
        return {"status": "ignored", "type": message.get("type")}

    call_info = message.get("call", {}) or {}
    metadata = call_info.get("metadata") or message.get("metadata") or {}
    transaction_id = metadata.get("transaction_id")
    transaction = _load_transaction(transaction_id)
    if transaction is None:
        return {"status": "error", "message": "no matching transaction"}

    if not transaction.get("call_outcome"):
        ended_reason = message.get("endedReason", "unknown")
        db.save_call_outcome(transaction_id=transaction["id"], outcome="disconnected", promised_date=None)
        logger.info(
            f"Call for transaction #{transaction['id']} ended without a resolved "
            f"outcome (endedReason={ended_reason}); recorded as 'disconnected'."
        )

    return {"status": "ok"}
