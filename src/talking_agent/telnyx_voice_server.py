"""
telnyx_voice_server.py
Places and drives the outbound "payment due today" phone call entirely on
Telnyx's Voice API (Call Control), REPLACING both the old Vapi-based
voice_server.py and the VideoSDK-based videosdk_voice_server.py as the
telephony/voice layer. call_agent.py's LangGraph "brain" is unchanged -- it
still only ever deals in plain text turns; this file is just the new
"transport" that gets spoken words in and out of it.

ARCHITECTURE (pure Call Control -- no raw media-streaming / audio-websocket
bridge needed):

    scheduler.py -> outbound_caller.trigger_payment_call(tx)
        -> POST http://localhost:8000/call/make?transaction_id=...  (this file)
            -> POST https://api.telnyx.com/v2/calls  (dial the borrower,
               get back a call_control_id immediately)
    -> Telnyx dials the borrower. When they answer, Telnyx POSTs a
       "call.answered" webhook to /webhooks/telnyx (this file):
            -> we call transcription_start on the call (Telnyx runs Deepgram
               ASR on the live audio and pushes us "call.transcription"
               webhook events as the borrower speaks)
            -> we ask call_agent.generate_greeting() for the opening line and
               issue a "speak" command with it (Telnyx TTS plays it into
               the call)
    -> Each "call.transcription" webhook with is_final=true is one customer
       utterance -> call_agent.handle_user_turn() -> the reply is split into
       sentence-sized chunks and pushed onto that call's speak queue; the
       FIRST one is spoken immediately (a "speak" command issued right
       away), and each subsequent queued sentence is spoken only once
       Telnyx confirms the previous one finished (the "call.speak.ended"
       webhook) -- Telnyx can only play one "speak" action at a time per
       call, so sentences are dispatched one-by-one in order. See
       _handle_event()'s "call.transcription" and "call.speak.ended"
       branches, and _advance_speak_queue().
    -> Once the agent's tools mark call_should_end, we hang up once the
       speak queue has fully drained (also driven from "call.speak.ended").

No SIP trunk, no VideoSDK account, and no hand-rolled audio bridge -- Telnyx
Voice API + its built-in transcription_start/speak commands cover the whole
loop. (See https://developers.telnyx.com/docs/voice/programmable-voice for
the underlying API reference.)

Requires (see .env):
    TELNYX_API_KEY          from the Telnyx Mission Control Portal
    TELNYX_CONNECTION_ID    the "Call Control Application" id (a.k.a.
                            connection_id) you created for this app --
                            must have an outbound voice profile attached
    TELNYX_FROM_NUMBER      a Telnyx voice-capable number, e.g. +1415...,
                            in E.164 format (no spaces)
    PUBLIC_BASE_URL         publicly reachable base URL for this server's
                            webhook (e.g. your ngrok URL in dev, or your
                            deployed URL in production) -- Telnyx needs to
                            be able to POST to {PUBLIC_BASE_URL}/webhooks/telnyx
    TELNYX_VOICE            (optional) TTS voice, defaults to "female"
                            (Telnyx's built-in generic voice).

Note: WhatsApp reminders (whatsapp.py) are UNCHANGED and stay on Twilio --
only the phone-call path moves to Telnyx.

Install:
    uv add fastapi "uvicorn[standard]" requests python-dotenv

Run (from your project root, alongside app.py and scheduler.py):
    uv run uvicorn telnyx_voice_server:app --host 0.0.0.0 --port 8000

In the Telnyx Portal, set your Call Control Application's webhook URL to:
    {PUBLIC_BASE_URL}/webhooks/telnyx
"""

import base64
import os
import logging
import threading
from collections import deque
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks

import database as db
import call_agent

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("telnyx-voice-server")

db.init_db()

TELNYX_API_KEY = os.getenv("TELNYX_API_KEY")
TELNYX_CONNECTION_ID = os.getenv("TELNYX_CONNECTION_ID")
# .strip() + drop internal spaces defensively -- a stray space in .env (e.g.
# "+1 272 258 0293" instead of "+12722580293") makes Telnyx reject the call
# with a validation error that otherwise looks like a mysterious 500.
TELNYX_FROM_NUMBER = (os.getenv("TELNYX_FROM_NUMBER") or "").replace(" ", "").strip()
PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").rstrip("/")
TELNYX_VOICE = os.getenv("TELNYX_VOICE", "Telnyx.KokoroTTS.af_heart")
TELNYX_LANGUAGE_CODE = os.getenv("TELNYX_LANGUAGE_CODE", "en-US")

TELNYX_API_BASE = "https://api.telnyx.com/v2"

CALL_DRY_RUN = os.getenv("CALL_DRY_RUN", "true").strip().lower() in ("1", "true", "yes", "on")

logger.info(
    f"telnyx_voice_server.py config -- CALL_DRY_RUN={CALL_DRY_RUN}, "
    f"TELNYX_FROM_NUMBER={TELNYX_FROM_NUMBER!r}, "
    f"TELNYX_CONNECTION_ID={TELNYX_CONNECTION_ID!r}, "
    f"PUBLIC_BASE_URL={PUBLIC_BASE_URL!r}, "
    f"TELNYX_API_KEY set: {bool(TELNYX_API_KEY)}"
)

app = FastAPI(title="Khata Telnyx Voice Server")

SESSIONS: dict[str, dict] = {}


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {TELNYX_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _telnyx_post(path: str, payload: dict) -> dict:
    resp = requests.post(f"{TELNYX_API_BASE}{path}", headers=_headers(), json=payload, timeout=30)
    if not resp.ok:
        # Log AND surface the real Telnyx error body (e.g. "invalid 'from'
        # number", "connection_id not found", "invalid API key") instead of
        # letting it bubble up as an opaque 500 with no detail.
        logger.error(f"Telnyx API error on POST {path}: {resp.status_code} {resp.text}")
        resp.raise_for_status()
    return resp.json()


def _find_transaction(transaction_id: int) -> Optional[dict]:
    for tx in db.get_all_transactions():
        if tx["id"] == transaction_id:
            return tx
    return None


class CallAlreadyEnded(Exception):
    """Raised when Telnyx rejects an action because the call already ended
    (error code 90018). Distinguished from other Telnyx API errors so
    callers can treat it as an expected race, not a real failure."""


def _speak(call_control_id: str, text: str) -> None:
    try:
        _telnyx_post(
            f"/calls/{call_control_id}/actions/speak",
            {
                "payload": text,
                "voice": TELNYX_VOICE,
            },
        )
    except requests.exceptions.HTTPError as e:
        body = e.response.text if e.response is not None else ""
        if e.response is not None and e.response.status_code == 422 and "90018" in body:
            # The borrower (or Telnyx) hung up between us deciding to speak
            # and the request landing -- expected under the lock-free window
            # right at call end, not a bug in itself. Let the caller (see
            # _handle_event) mark the session ended and stop quietly instead
            # of logging a scary traceback for a benign race.
            raise CallAlreadyEnded from e
        raise


def _start_transcription(call_control_id: str) -> None:
    _telnyx_post(
        f"/calls/{call_control_id}/actions/transcription_start",
        {
            "transcription_engine": "Deepgram",
            "transcription_tracks": "inbound",
        },
    )


def _hangup(call_control_id: str) -> None:
    _telnyx_post(f"/calls/{call_control_id}/actions/hangup", {})


def _new_session(transaction: dict) -> dict:
    return {
        "transaction": transaction,
        "state": call_agent.create_initial_state(transaction),
        "greeted": False,
        "pending_hangup": False,
        # Sentences waiting to be spoken. Telnyx can only play one "speak"
        # action at a time per call, so we dispatch them one-by-one --
        # the first as soon as it's ready, each next one only once
        # "call.speak.ended" confirms the previous finished (see
        # _advance_speak_queue()).
        "speak_queue": deque(),
        "speaking": False,
        # Starlette runs background_tasks.add_task() on a SYNC function
        # (like _handle_event) in a thread pool, NOT sequentially -- so if
        # Deepgram's endpointing fires more than one "is_final"
        # transcription for what's really one continuous utterance (common,
        # and visible as back-to-back call.transcription events a second or
        # so apart), two threads can end up running a full LLM turn for the
        # SAME call concurrently, racing to mutate this session dict and
        # racing to call _speak(). Symptoms seen in production: the same
        # call outcome saved multiple times, and a 422 "Call has already
        # ended" crash when a stale in-flight turn tries to speak after
        # call.hangup already popped the session. This lock forces all
        # per-call event handling (transcription turns, hangup, speak-queue
        # advances) to run one at a time per call -- see _handle_event().
        "lock": threading.Lock(),
        # Set once call.hangup fires (or a "call already ended" error comes
        # back from Telnyx) so any turn still in flight can bail out instead
        # of trying to speak into a dead call.
        "ended": False,
    }


def _enqueue_speech(session: dict, call_control_id: str, text: str) -> None:
    """Add a chunk of text to this call's speak queue. If nothing is
    currently playing, speak it right away; otherwise it waits its turn."""
    session["speak_queue"].append(text)
    if not session["speaking"]:
        _advance_speak_queue(session, call_control_id)


def _advance_speak_queue(session: dict, call_control_id: str) -> None:
    """Speak the next queued sentence, if any. If the queue is empty and
    the call is due to end, hang up now that everything's been said."""
    if session["speak_queue"]:
        next_text = session["speak_queue"].popleft()
        session["speaking"] = True
        _speak(call_control_id, next_text)
    else:
        session["speaking"] = False
        if session["pending_hangup"]:
            _hangup(call_control_id)


@app.post("/call/make")
async def make_call(transaction_id: int):
    """Places an outbound payment-reminder call for the given transaction.
    Called by outbound_caller.trigger_payment_call()."""
    transaction = _find_transaction(transaction_id)
    if transaction is None:
        return {"status": "error", "message": f"No transaction #{transaction_id}"}

    to_number = transaction["phone_number"]

    if CALL_DRY_RUN:
        logger.info(
            f"[DRY RUN] Would place Telnyx call to {to_number} for "
            f"transaction #{transaction_id} (set CALL_DRY_RUN=false to "
            f"actually dial)"
        )
        return {"status": "ok", "result": f"DRYRUN_CALL_{transaction_id}"}

    missing = [
        name for name, val in [
            ("TELNYX_API_KEY", TELNYX_API_KEY),
            ("TELNYX_CONNECTION_ID", TELNYX_CONNECTION_ID),
            ("TELNYX_FROM_NUMBER", TELNYX_FROM_NUMBER),
            ("PUBLIC_BASE_URL", PUBLIC_BASE_URL),
        ] if not val
    ]
    if missing:
        return {"status": "error", "message": f"Missing Telnyx config: {', '.join(missing)}"}

    payload = {
        "connection_id": TELNYX_CONNECTION_ID,
        "to": to_number,
        "from": TELNYX_FROM_NUMBER,
        "webhook_url": f"{PUBLIC_BASE_URL}/webhooks/telnyx",
        "client_state": base64.b64encode(str(transaction_id).encode()).decode(),
    }

    # THE ACTUAL FIX: catch the Telnyx API failure here and return its real
    # error message with a 200/JSON response, instead of letting the
    # exception propagate and turn into an opaque "500 Internal Server
    # Error" with no detail on the caller's (outbound_caller.py) side.
    try:
        data = _telnyx_post("/calls", payload)
    except requests.exceptions.HTTPError as e:
        error_body = e.response.text if e.response is not None else str(e)
        logger.error(f"Telnyx rejected the call request: {error_body}")
        return {"status": "error", "message": f"Telnyx API error: {error_body}"}
    except Exception as e:
        logger.exception("Unexpected error placing Telnyx call")
        return {"status": "error", "message": f"Unexpected error: {e}"}

    call_control_id = data["data"]["call_control_id"]

    SESSIONS[call_control_id] = _new_session(transaction)

    logger.info(f"Placed Telnyx call {call_control_id} to {to_number} for transaction #{transaction_id}")
    return {"status": "ok", "result": call_control_id}


@app.get("/sessions")
async def sessions():
    return {
        call_control_id: {
            "transaction_id": s["transaction"]["id"],
            "borrower_name": s["transaction"]["borrower_name"],
            "greeted": s["greeted"],
            "pending_hangup": s["pending_hangup"],
            "speaking": s["speaking"],
            "queued_sentences": len(s["speak_queue"]),
        }
        for call_control_id, s in SESSIONS.items()
    }


@app.post("/webhooks/telnyx")
async def telnyx_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    IMPORTANT: this returns {"received": True} to Telnyx immediately, BEFORE
    doing any of the slow work (calls to Groq for the LLM reply, calls back
    to Telnyx's own /speak and /transcription_start endpoints). Those all run
    afterwards in a background task via _handle_event().

    Why: Telnyx expects a fast HTTP response to its webhook POST. If our
    handler blocks on Groq + two outbound Telnyx API calls before responding
    -- which can easily take several seconds, more under load or a slow
    network -- Telnyx times out, marks the delivery "No Response", and
    retries the SAME event a few seconds later. Each retry used to re-run
    the full call.answered handler (see the "greeted" guard in
    _handle_event() below), so the borrower could end up hearing nothing for
    20-30s while Telnyx silently retried in the background, or hear the
    greeting fire twice. Acking immediately fixes the timeout/retry loop;
    the "greeted" guard fixes the duplicate-greeting risk if a retry still
    slips through.
    """
    body = await request.json()
    data = body.get("data", {})
    event_type = data.get("event_type")
    payload = data.get("payload", {})
    call_control_id = payload.get("call_control_id")

    # TEMP DEBUG: log every event type Telnyx actually sends us. Only 2
    # events per call (instead of answered -> transcription(s) ->
    # speak.ended(s) -> hangup) suggests the call never reaches
    # call.answered -- this makes that visible instead of guessing.
    logger.info(f"Telnyx webhook received: event_type={event_type} call_control_id={call_control_id}")

    session = SESSIONS.get(call_control_id) if call_control_id else None

    if session is None and call_control_id and payload.get("client_state"):
        try:
            transaction_id = int(base64.b64decode(payload["client_state"]).decode())
            transaction = _find_transaction(transaction_id)
            if transaction:
                session = _new_session(transaction)
                SESSIONS[call_control_id] = session
        except Exception:
            pass

    if session is None:
        logger.warning(f"Telnyx event {event_type} for unknown call {call_control_id} -- ignoring")
        return {"received": True}

    # Schedule the actual (slow) work and return to Telnyx right away.
    background_tasks.add_task(_handle_event, event_type, call_control_id, payload)
    return {"received": True}


def _handle_event(event_type: str, call_control_id: str, payload: dict) -> None:
    """Runs in the background, after the webhook has already acked Telnyx.

    IMPORTANT: Starlette runs BackgroundTasks for a sync function like this
    one in a thread pool, not one-at-a-time. If Deepgram's endpointing fires
    two "is_final" transcriptions in quick succession for the same
    continuous utterance (common, and visible in production logs as two
    call.transcription events ~1s apart), two threads would otherwise run a
    full LLM turn for the SAME call at once, racing to mutate `session` and
    racing to call Telnyx's /speak endpoint. Everything below acquires
    session["lock"] for its whole per-call handling so events for one call
    are always processed strictly one at a time, in delivery order.
    """
    session = SESSIONS.get(call_control_id)
    if session is None:
        logger.warning(f"Telnyx event {event_type} for call {call_control_id} vanished before background processing -- ignoring")
        return

    with session["lock"]:
        # The call ended (hangup already processed, or a prior action
        # already came back "call already ended") while this event was
        # waiting for the lock -- nothing left to do for it.
        if session.get("ended") and event_type != "call.hangup":
            logger.info(f"Ignoring {event_type} for {call_control_id} -- call already ended")
            return

        try:
            if event_type == "call.answered":
                # Guard against Telnyx retrying call.answered (e.g. after a slow
                # first response) and us greeting the borrower a second time.
                if session.get("greeted"):
                    logger.info(f"call.answered for {call_control_id} already handled -- skipping duplicate greeting")
                    return
                _start_transcription(call_control_id)
                state, greeting = call_agent.generate_greeting(session["state"])
                session["state"] = state
                session["greeted"] = True
                # Greeting is a single hardcoded line (no LLM involved -- see
                # generate_greeting()'s docstring), so it just goes straight
                # onto the (currently empty) speak queue.
                _enqueue_speech(session, call_control_id, greeting)

            elif event_type == "call.transcription":
                transcription_data = payload.get("transcription_data", {})
                if transcription_data.get("is_final"):
                    user_text = (transcription_data.get("transcript") or "").strip()
                    if user_text:
                        # Stream the reply: sentences get queued for speech
                        # as soon as the LLM produces them, instead of
                        # waiting for the whole turn (including any tool
                        # calls) to finish first. See
                        # call_agent.stream_user_turn()'s docstring.
                        should_end = False
                        for kind, value in call_agent.stream_user_turn(session["state"], user_text):
                            if kind == "sentence":
                                _enqueue_speech(session, call_control_id, value)
                            elif kind == "done":
                                state, should_end = value
                                session["state"] = state
                                # Save the outcome only on the turn it's
                                # newly decided, not every subsequent turn --
                                # state carries "outcome" forward once set,
                                # so checking state.get("outcome") alone
                                # would (and did) re-save it on every later
                                # turn too.
                                if state.get("outcome") is not None and not session.get("outcome_saved"):
                                    db.save_call_outcome(
                                        transaction_id=session["transaction"]["id"],
                                        outcome=state["outcome"],
                                        promised_date=state.get("promised_date"),
                                    )
                                    session["outcome_saved"] = True
                                    logger.info(
                                        f"Saved call outcome for transaction "
                                        f"#{session['transaction']['id']}: {state['outcome']}"
                                    )
                        # Only hang up once everything queued has actually
                        # been spoken -- _advance_speak_queue() (driven by
                        # call.speak.ended) checks this flag once its queue
                        # empties, rather than hanging up right away.
                        session["pending_hangup"] = should_end

            elif event_type == "call.speak.ended":
                _advance_speak_queue(session, call_control_id)

            elif event_type == "call.hangup":
                session["ended"] = True
                transaction = session["transaction"]
                outcome = session["state"].get("outcome")
                hangup_cause = payload.get("hangup_cause")
                logger.info(f"call.hangup for {call_control_id}: hangup_cause={hangup_cause}")
                if outcome is None:
                    # Call ended (no answer, hung up mid-conversation, etc.)
                    # without ever reaching record_promise/record_refusal/
                    # end_call -- record it as "disconnected" so it doesn't
                    # silently vanish, and so save_call_outcome pushes due_date
                    # forward by 2 days for a retry (see database.py).
                    db.save_call_outcome(
                        transaction_id=transaction["id"],
                        outcome="disconnected",
                        promised_date=None,
                    )
                    logger.info(
                        f"Call for transaction #{transaction['id']} ended without a "
                        f"resolved outcome; recorded as 'disconnected' (due date "
                        f"pushed forward for retry)."
                    )
                else:
                    logger.info(
                        f"Call ended for transaction #{transaction['id']} "
                        f"(call_control_id={call_control_id})"
                    )
                SESSIONS.pop(call_control_id, None)

        except CallAlreadyEnded:
            # Telnyx told us the call is gone while we were mid-turn (e.g.
            # the borrower hung up right as we were about to speak the next
            # sentence). Expected race, not a real error -- mark it so any
            # remaining sentences in this turn are skipped instead of each
            # throwing the same thing.
            session["ended"] = True
            logger.info(f"Call {call_control_id} ended mid-turn -- stopping speech for this call")
        except Exception as e:
            logger.exception(f"Error handling {event_type} for call {call_control_id}: {e}")