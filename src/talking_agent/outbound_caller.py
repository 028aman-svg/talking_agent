"""
outbound_caller.py
Places the outbound "payment due today" phone call via our own
telnyx_voice_server.py (Telnyx Voice API / Call Control) instead of Vapi or
VideoSDK. The conversation "brain" is unchanged -- it's still call_agent.py's
LangGraph graph -- telnyx_voice_server.py just wraps it in Telnyx's
speak/transcription_start commands instead of Vapi's custom-llm webhook or
VideoSDK's Pipeline.

Requires (see .env):
    VOICE_SERVER_URL   base URL of telnyx_voice_server.py, e.g.
                        http://localhost:8000 if scheduler.py and the voice
                        server run on the same machine (default), or your
                        deployed voice server's URL in production.

The voice server itself (telnyx_voice_server.py) needs its own separate set
of credentials (Telnyx) -- see that file's docstring.

(The previous Vapi-based version of this file is kept as
outbound_caller.vapi.py.bak, if you still have it, for reference / rollback.)
"""

import os
import logging
import requests
from dotenv import load_dotenv

# override=True: without this, load_dotenv() silently refuses to overwrite a
# CALL_DRY_RUN (or any other) value that's already sitting in os.environ from
# an earlier export/session -- which is a very common reason ".env changes
# don't seem to take effect." With override=True, whatever is in .env always
# wins.
load_dotenv(override=True)

logger = logging.getLogger("khata-scheduler")

VOICE_SERVER_URL = os.getenv("VOICE_SERVER_URL", "http://localhost:8000").rstrip("/")

# DRY RUN KNOB (same pattern as whatsapp.py's WHATSAPP_DRY_RUN).
# Defaults to "true" so working on other parts of the app never
# accidentally places a real phone call. Set CALL_DRY_RUN=false in .env
# once you're ready to actually call people. NOTE: telnyx_voice_server.py
# has its own CALL_DRY_RUN knob too -- both must be false to actually dial.
_raw_call_dry_run = os.getenv("CALL_DRY_RUN", "true")
CALL_DRY_RUN = _raw_call_dry_run.strip().lower() in ("1", "true", "yes", "on")

# Visibility: print exactly what was loaded and from where, so a
# misconfigured .env (wrong location, wrong filename, stale shell var) is
# obvious immediately instead of silently no-op'ing every call.
logger.info(
    f"outbound_caller.py loaded CALL_DRY_RUN={_raw_call_dry_run!r} "
    f"-> dry_run={CALL_DRY_RUN}, VOICE_SERVER_URL={VOICE_SERVER_URL!r}"
)


def trigger_payment_call(transaction: dict) -> str:
    """
    Places an outbound call to the borrower about a payment due today, by
    asking telnyx_voice_server.py to place it.
    Returns a call identifier, or a "DRYRUN_CALL_<id>" placeholder if
    CALL_DRY_RUN is on (default) -- nothing is sent to the voice server at all
    in that case.
    """
    to_number = transaction["phone_number"]
    transaction_id = transaction["id"]

    if CALL_DRY_RUN:
        print(
            "[DRY RUN] Would trigger Telnyx call:\n"
            f"  to: {to_number}\n"
            f"  transaction_id: {transaction_id}\n"
            f"  voice server: {VOICE_SERVER_URL}/call/make\n"
            "  (set CALL_DRY_RUN=false in .env -- on BOTH this file and "
            "telnyx_voice_server.py -- to actually place the call)"
        )
        return f"DRYRUN_CALL_{transaction_id}"

    resp = requests.post(
        f"{VOICE_SERVER_URL}/call/make",
        params={"transaction_id": transaction_id},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(
            f"Failed to place call to {to_number} (transaction #{transaction_id}): "
            f"{resp.status_code} {resp.text}"
        )
    data = resp.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"Voice server rejected call for transaction #{transaction_id}: {data}")
    return str(data.get("result", f"CALL_{transaction_id}"))


def check_call_status(transaction_id: int) -> dict:
    """Fetch the current active-session status for a transaction's call, if any."""
    resp = requests.get(f"{VOICE_SERVER_URL}/sessions", timeout=15)
    resp.raise_for_status()
    return resp.json()