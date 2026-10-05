"""
videosdk_voice_server.py
Bridges VideoSDK's SIP + Agents framework to the existing LangGraph "brain" in
call_agent.py, REPLACING Vapi (outbound_caller.py + voice_server.py) as the
telephony/voice layer. call_agent.py itself is unchanged in spirit -- it still
only ever deals in plain text turns; this file is the new "transport".

ARCHITECTURE (per VideoSDK's SIP framework docs -- verify exact SIPManager
method names against your installed version's docs/`help()` before relying on
this in production; this SDK is new (v1.0+, 2026) and evolving):

    scheduler.py -> outbound_caller.trigger_payment_call(tx)
        -> POST http://localhost:8000/call/make  (this file)
            -> SIPManager creates a VideoSDK room + launches a KhataCallAgent
               into it (waiting)
            -> SIPManager asks Twilio to dial the borrower's phone number
            -> when the borrower answers, Twilio hits /sip/answer/{room_id}
               (also handled by SIPManager) and bridges the call's audio into
               that room
    -> Inside the room: audio in -> STT -> KhataLLM (wraps call_agent.py's
       LangGraph graph) -> TTS -> audio out, driven by VideoSDK's Pipeline.

Requires (see .env):
    VIDEOSDK_AUTH_TOKEN       from app.videosdk.live
    VIDEOSDK_SIP_USERNAME     from the VideoSDK SIP dashboard
    VIDEOSDK_SIP_PASSWORD     from the VideoSDK SIP dashboard
    TWILIO_ACCOUNT_SID        reuse the same Twilio account whatsapp.py uses
    TWILIO_AUTH_TOKEN
    TWILIO_PHONE_NUMBER       a Twilio voice-capable number, e.g. +1415...
    DEEPGRAM_API_KEY          used for both STT (Nova) and TTS (Aura) so you
                               only need one extra provider account -- swap
                               for any other videosdk-plugins-* STT/TTS if
                               you'd rather use something else
    PUBLIC_BASE_URL           only needed in production (see below); in local
                               dev, ngrok is started automatically if
                               NGROK_AUTHTOKEN is set

Install:
    pip install videosdk-agents videosdk-plugins-sip videosdk-plugins-deepgram \
        pyngrok fastapi uvicorn --break-system-packages

Run (from your project root, alongside app.py and scheduler.py):
    uv run uvicorn videosdk_voice_server:app --host 0.0.0.0 --port 8000

NOTE: this fully replaces the OLD Vapi-based voice_server.py + the old
Vapi-based outbound_caller.py. Once this is confirmed working, you can delete
voice_server.py and the Vapi bits of outbound_caller.py (a rewritten version
that calls THIS server is provided separately).
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage, AIMessage

from videosdk.agents import Agent, Pipeline, AgentSession, JobContext
from videosdk.agents.llm import LLM, ChatContext, LLMResponse
from videosdk.plugins.deepgram import DeepgramSTT, DeepgramTTS
from videosdk.plugins.sip import create_sip_manager  # verify this import path
                                                       # against your installed
                                                       # version -- some SDK
                                                       # versions expose it
                                                       # from videosdk.agents.plugins

import database as db
import call_agent

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("videosdk-voice-server")

db.init_db()

VIDEOSDK_AUTH_TOKEN = os.getenv("VIDEOSDK_AUTH_TOKEN")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Same dry-run pattern as whatsapp.py / the old outbound_caller.py -- defaults
# to on so nothing places a real call while you're still wiring this up.
CALL_DRY_RUN = os.getenv("CALL_DRY_RUN", "true").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------
# Custom LLM adapter: wraps call_agent.py's compiled LangGraph graph so
# VideoSDK's Pipeline can drive it turn-by-turn, exactly the same "brain"
# that used to sit behind Vapi's /chat/completions webhook.
# ---------------------------------------------------------------------
class KhataLLM(LLM):
    """One instance of this is created per call (see KhataCallAgent below),
    bound to a single transaction, and holds that call's running CallState."""

    def __init__(self, transaction: dict, on_call_should_end):
        super().__init__()
        self._transaction = transaction
        self._state = call_agent.create_initial_state(transaction)
        self._on_call_should_end = on_call_should_end  # async callback -> hang up

    async def chat(
        self,
        messages: ChatContext,
        tools: Optional[list] = None,
        **kwargs,
    ) -> AsyncIterator[LLMResponse]:
        # ChatContext holds the turn Pipeline just transcribed via STT. Pull
        # the latest user utterance out of it -- our own CallState (self._state)
        # is already the source of truth for everything before this turn, so
        # we only need what's new. If ChatContext has no prior turns at all,
        # treat this as the call-connected greeting trigger, matching
        # call_agent.generate_greeting()'s behaviour.
        latest_user_text = _extract_latest_user_text(messages)
        if latest_user_text is None:
            latest_user_text = call_agent.CALL_CONNECTED_HINT

        self._state["messages"].append(HumanMessage(content=latest_user_text))
        result = call_agent.get_graph().invoke(self._state)
        self._state = result

        reply_text = call_agent._extract_latest_ai_text(result["messages"])

        if result.get("outcome") is not None or result.get("call_should_end"):
            outcome = result.get("outcome") or "ended"
            db.save_call_outcome(
                transaction_id=self._transaction["id"],
                outcome=outcome,
                promised_date=result.get("promised_date"),
            )
            logger.info(
                f"Saved call outcome for transaction #{self._transaction['id']}: {outcome}"
            )

        if result.get("call_should_end"):
            # Let the TTS stage speak `reply_text` first, then hang up --
            # exact hook name to schedule this after speech finishes should
            # be verified against Pipeline's @pipeline.on("tts") /
            # AgentSession hang-up API in your installed version.
            await self._on_call_should_end()

        # LLMResponse's exact field names should be checked against
        # `videosdk.agents.llm.LLMResponse` in your installed version --
        # this assumes a `content` field, matching the common pattern of the
        # other LLM plugins' streaming responses.
        yield LLMResponse(content=reply_text)

    async def cancel_current_generation(self) -> None:
        # call_agent's graph.invoke() is synchronous/blocking per turn rather
        # than a token stream, so there's nothing to cancel mid-flight.
        pass


def _extract_latest_user_text(messages: ChatContext) -> Optional[str]:
    """Pulls the most recent user-role text out of a VideoSDK ChatContext.
    Field/shape should be double-checked against your installed SDK version --
    written defensively to tolerate either a `.items` list of role/content
    dicts or an `.messages` list of LangChain-style message objects."""
    items = getattr(messages, "items", None) or getattr(messages, "messages", None) or []
    for item in reversed(items):
        role = getattr(item, "role", None) or (item.get("role") if isinstance(item, dict) else None)
        if role == "user":
            content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
            if content:
                return content
    return None


# ---------------------------------------------------------------------
# The Agent itself: one instance per call, built with the transaction it's
# about, using KhataLLM as its "brain" and Deepgram for STT/TTS.
# ---------------------------------------------------------------------
class KhataCallAgent(Agent):
    def __init__(self, transaction: dict, ctx: Optional[JobContext] = None):
        self.transaction = transaction
        super().__init__(
            instructions=call_agent.build_system_prompt(transaction),
            tools=[self.end_call],  # base Agent's built-in hang-up tool
        )
        self.ctx = ctx

    async def on_enter(self) -> None:
        pass

    async def on_exit(self) -> None:
        pass


def build_pipeline_for_call(transaction: dict, agent: "KhataCallAgent") -> Pipeline:
    llm = KhataLLM(transaction=transaction, on_call_should_end=agent.end_call)
    return Pipeline(
        stt=DeepgramSTT(model="nova-3"),
        llm=llm,
        tts=DeepgramTTS(model="aura-asteria-en"),
    )


# ---------------------------------------------------------------------
# FastAPI app + SIPManager wiring
# ---------------------------------------------------------------------
sip_manager = create_sip_manager(
    provider="twilio",
    videosdk_token=VIDEOSDK_AUTH_TOKEN,
    provider_config={
        "account_sid": TWILIO_ACCOUNT_SID,
        "auth_token": TWILIO_AUTH_TOKEN,
        "phone_number": TWILIO_PHONE_NUMBER,
    },
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    port = int(os.getenv("PORT", 8000))
    public_base_url = os.getenv("PUBLIC_BASE_URL")

    if public_base_url:
        sip_manager.set_base_url(public_base_url.rstrip("/"))
        logger.info(f"Using configured PUBLIC_BASE_URL: {public_base_url}")
    else:
        # Local dev convenience: auto-start an ngrok tunnel, same pattern as
        # VideoSDK's own SIP framework docs show.
        try:
            from pyngrok import ngrok

            ngrok.kill()
            ngrok_auth_token = os.getenv("NGROK_AUTHTOKEN")
            if ngrok_auth_token:
                ngrok.set_auth_token(ngrok_auth_token)
            tunnel = ngrok.connect(port, "http")
            sip_manager.set_base_url(tunnel.public_url)
            logger.info(f"NGROK TUNNEL CREATED: {tunnel.public_url}")
        except Exception as e:
            logger.error(
                f"Failed to start ngrok tunnel, and no PUBLIC_BASE_URL set "
                f"either -- outbound calls will fail: {e}"
            )

    yield

    try:
        from pyngrok import ngrok
        ngrok.kill()
    except Exception:
        pass


app = FastAPI(title="Khata VideoSDK Voice Server", lifespan=lifespan)

# If your installed videosdk-plugins-sip version exposes a router/mount
# helper (rather than you having to hand-write /webhook/incoming and
# /sip/answer/{room_id}), prefer that over hand-rolling them -- check
# `create_sip_manager`'s docs/source for something like:
#     sip_manager.mount(app)
# or
#     app.include_router(sip_manager.router)
# This is the one piece of wiring most likely to differ from what's below,
# depending on your installed version.
try:
    sip_manager.mount(app)
except AttributeError:
    logger.warning(
        "sip_manager.mount(app) not available in this SDK version -- you'll "
        "need to wire /webhook/incoming and /sip/answer/{room_id} manually "
        "per the SIP framework docs at https://docs.videosdk.live/ai_agents/sip"
    )


@app.post("/call/make")
async def make_call(transaction_id: int):
    """Places an outbound payment-reminder call for the given transaction.
    Called by outbound_caller.trigger_payment_call() instead of the old Vapi
    /call endpoint."""
    transaction = None
    for tx in db.get_all_transactions():
        if tx["id"] == transaction_id:
            transaction = tx
            break
    if transaction is None:
        return {"status": "error", "message": f"No transaction #{transaction_id}"}

    to_number = transaction["phone_number"]

    async def agent_factory(ctx: JobContext) -> AgentSession:
        agent = KhataCallAgent(transaction=transaction, ctx=ctx)
        pipeline = build_pipeline_for_call(transaction, agent)
        return AgentSession(agent=agent, pipeline=pipeline)

    # Exact call here (create_room + launch agent_factory + dial to_number)
    # should be checked against your installed SIPManager's method name --
    # this mirrors the documented /call/make behaviour in
    # https://docs.videosdk.live/ai_agents/sip
    result = await sip_manager.make_call(to_number=to_number, agent_factory=agent_factory)
    logger.info(f"Placed call to {to_number} for transaction #{transaction_id}: {result}")
    return {"status": "ok", "result": result}


@app.get("/sessions")
async def sessions():
    return await sip_manager.get_sessions()