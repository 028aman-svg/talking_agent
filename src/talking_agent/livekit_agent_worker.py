"""
livekit_agent_worker.py
Persistent LiveKit Agents worker that actually drives the call's audio:
STT -> call_agent.py's LangGraph "brain" -> TTS. Dispatched into a room by
livekit_voice_server.py's /call/make. call_agent.py itself is UNCHANGED --
same pattern as videosdk_voice_server.py's KhataLLM adapter, ported to
LiveKit's Agents API.

NOTE: LiveKit's Agents SDK API surface moves fast -- double-check class/
method names below (AgentSession, Agent, llm.LLM, ChatContext) against your
installed `livekit-agents` version's docs before relying on this in
production, the same caution the VideoSDK file gives for its SDK.

Requires (see .env): LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET,
DEEPGRAM_API_KEY, LIVEKIT_AGENT_NAME.

Run (separate long-lived process, alongside livekit_voice_server.py):
    uv run livekit_agent_worker.py dev      # local dev
    uv run livekit_agent_worker.py start    # production
"""

import os
import logging
from typing import AsyncIterator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions
from livekit.agents.llm import LLM, LLMStream, ChatContext
from livekit.plugins import deepgram, silero

import database as db
import call_agent

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("livekit-agent-worker")

db.init_db()

AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "khata-collector")


def _find_transaction(transaction_id: int) -> dict | None:
    for tx in db.get_all_transactions():
        if tx["id"] == transaction_id:
            return tx
    return None


class KhataLLM(LLM):
    """Wraps call_agent.py's compiled LangGraph graph so LiveKit's
    AgentSession can drive it turn-by-turn. One instance per call, holding
    that call's running CallState -- same role as videosdk_voice_server.py's
    KhataLLM."""

    def __init__(self, transaction: dict, on_outcome_saved=None):
        super().__init__()
        self._transaction = transaction
        self._state = call_agent.create_initial_state(transaction)
        self._on_outcome_saved = on_outcome_saved

    def chat(self, *, chat_ctx: ChatContext, **kwargs) -> LLMStream:
        latest_user_text = _extract_latest_user_text(chat_ctx) or call_agent.CALL_CONNECTED_HINT

        self._state["messages"].append(HumanMessage(content=latest_user_text))
        result = call_agent.get_graph().invoke(self._state)
        self._state = result

        reply_text = call_agent._extract_latest_ai_text(result["messages"])

        if result.get("outcome") is not None:
            db.save_call_outcome(
                transaction_id=self._transaction["id"],
                outcome=result["outcome"],
                promised_date=result.get("promised_date"),
            )
            logger.info(f"Saved call outcome for transaction #{self._transaction['id']}: {result['outcome']}")
            if self._on_outcome_saved:
                self._on_outcome_saved()

        return _SingleReplyLLMStream(reply_text, should_end=result.get("call_should_end", False))


class _SingleReplyLLMStream(LLMStream):
    """Minimal non-streaming LLMStream wrapping one full reply -- call_agent's
    graph.invoke() is blocking/synchronous per turn, not token-streamed."""

    def __init__(self, text: str, should_end: bool):
        self._text = text
        self.should_end = should_end

    async def __anext__(self):
        raise NotImplementedError("Verify LLMStream's exact chunk-yielding protocol "
                                   "against your installed livekit-agents version.")


def _extract_latest_user_text(chat_ctx: ChatContext) -> str | None:
    for item in reversed(chat_ctx.items):
        if getattr(item, "role", None) == "user":
            return getattr(item, "text_content", None) or str(getattr(item, "content", ""))
    return None


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    transaction_id = int(ctx.job.metadata)
    transaction = _find_transaction(transaction_id)
    if transaction is None:
        logger.error(f"No transaction #{transaction_id} -- shutting down job")
        return

    outcome_saved = {"flag": False}

    def mark_saved():
        outcome_saved["flag"] = True

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3"),
        llm=KhataLLM(transaction=transaction, on_outcome_saved=mark_saved),
        tts=deepgram.TTS(model="aura-asteria-en"),
    )

    agent = Agent(instructions=call_agent.build_system_prompt(transaction))

    async def on_disconnect():
        # Safety net matching telnyx_voice_server.py's call.hangup handler:
        # if the room closes without a resolved outcome, record "disconnected"
        # so it doesn't silently vanish, and due_date still gets pushed
        # forward for a retry (see database.save_call_outcome).
        if not outcome_saved["flag"]:
            db.save_call_outcome(transaction_id=transaction_id, outcome="disconnected", promised_date=None)
            logger.info(f"Call for transaction #{transaction_id} ended without a resolved outcome; recorded as 'disconnected'.")

    ctx.room.on("disconnected", lambda *_: agents.utils.log_exceptions(on_disconnect())) # verify event name/signature

    await session.start(agent=agent, room=ctx.room)
    # Opening line -- mirrors call_agent.generate_greeting()'s role for the
    # other transports. Verify session.generate_reply()'s exact signature.
    await session.generate_reply(instructions=call_agent.CALL_CONNECTED_HINT)


if __name__ == "__main__":
    agents.cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name=AGENT_NAME))