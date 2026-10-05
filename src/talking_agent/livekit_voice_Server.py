"""
livekit_voice_server.py
Trigger server for the outbound "payment due today" phone call, using
LiveKit for the telephony + conversational layer (REPLACING
telnyx_voice_server.py). This file's job is small: create a LiveKit room,
dial the borrower into it via a SIP trunk, and explicitly dispatch the
call_agent-driven worker (see livekit_agent_worker.py) into that room.
The worker process does all the actual conversation work -- this file
is NOT in the audio path at all, unlike telnyx_voice_server.py.

Requires (see .env): LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET,
LIVEKIT_SIP_TRUNK_ID, LIVEKIT_AGENT_NAME.

You must ALSO have livekit_agent_worker.py running separately (see that
file's docstring) -- this server only places the call, it doesn't answer it.

Run:
    uv run uvicorn livekit_voice_server:app --host 0.0.0.0 --port 8000
"""

import os
import uuid
import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from livekit import api

import database as db

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("livekit-voice-server")

db.init_db()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_TRUNK_ID")
LIVEKIT_AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "khata-collector")

CALL_DRY_RUN = os.getenv("CALL_DRY_RUN", "true").strip().lower() in ("1", "true", "yes", "on")

logger.info(
    f"livekit_voice_server.py config -- CALL_DRY_RUN={CALL_DRY_RUN}, "
    f"LIVEKIT_URL={LIVEKIT_URL!r}, LIVEKIT_SIP_TRUNK_ID={LIVEKIT_SIP_TRUNK_ID!r}, "
    f"LIVEKIT_AGENT_NAME={LIVEKIT_AGENT_NAME!r}"
)

app = FastAPI(title="Khata LiveKit Voice Server")

SESSIONS: dict[str, dict] = {}  # room_name -> {transaction, ...}


def _find_transaction(transaction_id: int) -> dict | None:
    for tx in db.get_all_transactions():
        if tx["id"] == transaction_id:
            return tx
    return None


@app.post("/call/make")
async def make_call(transaction_id: int):
    """Places an outbound payment-reminder call for the given transaction.
    Called by outbound_caller.trigger_payment_call() -- same contract as
    telnyx_voice_server.py's /call/make."""
    transaction = _find_transaction(transaction_id)
    if transaction is None:
        return {"status": "error", "message": f"No transaction #{transaction_id}"}

    to_number = transaction["phone_number"]

    if CALL_DRY_RUN:
        logger.info(
            f"[DRY RUN] Would place LiveKit call to {to_number} for "
            f"transaction #{transaction_id} (set CALL_DRY_RUN=false to actually dial)"
        )
        return {"status": "ok", "result": f"DRYRUN_CALL_{transaction_id}"}

    missing = [
        name for name, val in [
            ("LIVEKIT_URL", LIVEKIT_URL),
            ("LIVEKIT_API_KEY", LIVEKIT_API_KEY),
            ("LIVEKIT_API_SECRET", LIVEKIT_API_SECRET),
            ("LIVEKIT_SIP_TRUNK_ID", LIVEKIT_SIP_TRUNK_ID),
        ] if not val
    ]
    if missing:
        return {"status": "error", "message": f"Missing LiveKit config: {', '.join(missing)}"}

    room_name = f"khata-call-{transaction_id}-{uuid.uuid4().hex[:8]}"

    lk = api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    try:
        # 1. Explicitly dispatch our worker into the room, passing the
        #    transaction_id as job metadata so it knows which borrower/state
        #    to load (see livekit_agent_worker.py's entrypoint).
        await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=LIVEKIT_AGENT_NAME,
                room=room_name,
                metadata=str(transaction_id),
            )
        )

        # 2. Dial the borrower into the same room via the outbound SIP trunk.
        #    Field names here should be checked against your installed
        #    livekit-api version -- this mirrors the documented outbound
        #    calling pattern at https://docs.livekit.io/sip/making-calls/
        participant = await lk.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=LIVEKIT_SIP_TRUNK_ID,
                sip_call_to=to_number,
                room_name=room_name,
                participant_identity=f"borrower-{transaction_id}",
                wait_until_answered=False,
            )
        )
    except Exception as e:
        logger.exception("Failed to place LiveKit call")
        return {"status": "error", "message": f"LiveKit API error: {e}"}
    finally:
        await lk.aclose()

    SESSIONS[room_name] = {
        "transaction_id": transaction_id,
        "borrower_name": transaction["borrower_name"],
        "sip_participant_id": getattr(participant, "participant_id", None),
    }

    logger.info(f"Placed LiveKit call in room {room_name} to {to_number} for transaction #{transaction_id}")
    return {"status": "ok", "result": room_name}


@app.get("/sessions")
async def sessions():
    return SESSIONS