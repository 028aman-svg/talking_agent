"""
whatsapp.py
Formats and sends itemized WhatsApp payment reminders via Twilio.
"""

import os
import json
from dotenv import load_dotenv
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

load_dotenv()

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")  # e.g. "whatsapp:+14155238886"

# DRY RUN KNOB
# Set WHATSAPP_DRY_RUN=false in .env (or unset it) to actually send messages.
# Defaults to "true" (dry run / no send) so that working on other parts of
# the codebase never accidentally burns through your limited free WhatsApp
# messages. In dry-run mode, send_whatsapp_reminder() logs/prints what WOULD
# have been sent and returns a fake SID instead of calling Twilio at all.
DRY_RUN = os.getenv("WHATSAPP_DRY_RUN", "true").strip().lower() in ("1", "true", "yes", "on")

# Twilio's pre-approved Sandbox template ("Appointment Reminders") for THIS
# account:
#   "Your appointment is coming up on {{1}} at {{2}}"
# Trial accounts can ONLY send ContentSids that are one of Twilio's own
# sandbox templates shown on the "Try out WhatsApp" console page -- custom
# Content Templates (like the "payment_reminder" one created via Content
# Editor) are rejected with error 21655 until the account is upgraded AND
# a WhatsApp Business Sender is approved. See:
# https://www.twilio.com/docs/usage/trials/try-out-whatsapp
#
# NOTE: this Sandbox template SID is specific to this Twilio account --
# check yours under Messaging > Try it out > Send a WhatsApp message.
SANDBOX_TEMPLATE_SID = "HXfe5ab5f00277942d4d4200328b4d403c"

# Once you upgrade to a paid account and get your own WhatsApp Business
# Sender + approved Content Template, set TWILIO_CONTENT_SID in .env to
# that template's SID (e.g. the "payment_reminder" one,
# HXcff646f616911d1e466eb1c5634ba3b9) and update build_template_variables()
# below to match its variable slots.
CONTENT_SID = os.getenv("TWILIO_CONTENT_SID", SANDBOX_TEMPLATE_SID)


def _get_client():
    if not ACCOUNT_SID or not AUTH_TOKEN:
        raise RuntimeError(
            "Twilio credentials missing. Check your .env file has "
            "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN set."
        )
    return Client(ACCOUNT_SID, AUTH_TOKEN)


def format_reminder_message(transaction: dict) -> str:
    """
    Build a detailed, itemized reminder message text for a transaction dict
    (as returned by database.get_due_today() / get_all_transactions()).
    Kept for reference/preview purposes — actual sending uses a template
    (see build_template_variables) since free-form body text is rejected
    for business-initiated WhatsApp messages.
    """
    lines = [
        f"Hello {transaction['borrower_name']},",
        "",
        "This is a reminder from the shop regarding your pending payment:",
        "",
        "*Items purchased:*",
    ]

    for item in transaction["items"]:
        lines.append(
            f"- {item['product_name']} x{item['quantity']} "
            f"@ Rs.{item['price_per_unit']:,.2f} = Rs.{item['item_total']:,.2f}"
        )

    lines.extend([
        "",
        f"*Purchase Date:* {transaction['purchase_date']}",
        f"*Total Amount Due:* Rs.{transaction['total_amount']:,.2f}",
        f"*Due Date:* {transaction['due_date']} (today)",
        "",
        "Please make the payment at your earliest convenience. "
        "Thank you for your business!",
    ])

    return "\n".join(lines)


def _to_whatsapp_number(phone_number: str) -> str:
    """Ensure the number is in Twilio's 'whatsapp:+<countrycode><number>' format."""
    phone_number = phone_number.strip()
    if phone_number.startswith("whatsapp:"):
        return phone_number
    if not phone_number.startswith("+"):
        raise ValueError(
            f"Phone number '{phone_number}' must include a country code, e.g. +919876543210"
        )
    return f"whatsapp:{phone_number}"


def format_items_summary(transaction: dict) -> str:
    """
    Flatten the item list into one short string for use as a template variable,
    e.g. 'Rice 5kg bag x2, Soap x3'. WhatsApp templates can't render loops,
    so this is the workaround for itemized detail inside a single {{n}} slot.
    """
    parts = [f"{item['product_name']} x{item['quantity']}" for item in transaction["items"]]
    return ", ".join(parts)


def build_template_variables(transaction: dict) -> dict:
    """
    Map transaction data onto the Sandbox 'Appointment Reminders' template's
    two slots: "Your appointment is coming up on {{1}} at {{2}}".
    We repurpose {{1}} for the due date and pack everything else (name,
    items, total) into {{2}} since the sandbox template only gives us two
    slots and trial accounts can't use a custom template with more.

    Once you upgrade the account and get the 'payment_reminder' custom
    template approved for a real WhatsApp Sender, replace this with the
    4-slot mapping: {{1}} name, {{2}} items summary, {{3}} total, {{4}} due
    date, and point CONTENT_SID (via TWILIO_CONTENT_SID in .env) at
    HXcff646f616911d1e466eb1c5634ba3b9.
    """
    items_summary = format_items_summary(transaction)
    return {
        "1": transaction["due_date"],
        "2": (
            f"Rs.{transaction['total_amount']:,.2f} payment due from "
            f"{transaction['borrower_name']} for: {items_summary}"
        ),
    }


def send_whatsapp_reminder(transaction: dict) -> str:
    """
    Sends a reminder WhatsApp message for the given transaction using an
    approved Content Template (required for business-initiated messages
    outside the 24-hour customer session window).
    Returns the Twilio message SID on success. Raises on failure.

    Respects the WHATSAPP_DRY_RUN knob (see top of file): when dry-run is
    on (the default), no Twilio API call is made at all -- this just prints
    what would have been sent and returns a fake SID starting with
    "DRYRUN_". Set WHATSAPP_DRY_RUN=false in .env to send for real.
    """
    to_number = _to_whatsapp_number(transaction["phone_number"])
    variables = build_template_variables(transaction)

    if DRY_RUN:
        preview = format_reminder_message(transaction)
        print(
            "[DRY RUN] Would send WhatsApp message:\n"
            f"  to: {to_number}\n"
            f"  content_sid: {CONTENT_SID}\n"
            f"  content_variables: {json.dumps(variables)}\n"
            f"  --- full itemized preview ---\n{preview}\n"
            "  (set WHATSAPP_DRY_RUN=false in .env to actually send)"
        )
        return f"DRYRUN_{transaction.get('id', 'unknown')}"

    if not WHATSAPP_FROM:
        raise RuntimeError("TWILIO_WHATSAPP_FROM is not set in .env")

    client = _get_client()

    try:
        message = client.messages.create(
            from_=WHATSAPP_FROM,
            to=to_number,
            content_sid=CONTENT_SID,
            content_variables=json.dumps(variables),
        )
        return message.sid
    except TwilioRestException as e:
        raise RuntimeError(f"Failed to send WhatsApp message to {to_number}: {e}") from e


def check_message_status(message_sid: str) -> dict:
    """
    Fetch the current delivery status of a previously sent message.
    Possible statuses: queued, sending, sent, delivered, read, undelivered, failed.
    Returns a dict with status and, if failed, the error info.
    """
    client = _get_client()
    msg = client.messages(message_sid).fetch()
    return {
        "sid": msg.sid,
        "status": msg.status,
        "error_code": msg.error_code,
        "error_message": msg.error_message,
    }


if __name__ == "__main__":
    sample_transaction = {
        "borrower_name": "aman Kumar",
        "phone_number": "+916299587641",
        "purchase_date": "2026-08-30",
        "due_date": "2026-08-31",
        "total_amount": 590.0,
        "items": [
            {"product_name": "Rice 5kg bag", "quantity": 2, "price_per_unit": 250.0, "item_total": 500.0},
            {"product_name": "Soap", "quantity": 3, "price_per_unit": 30.0, "item_total": 90.0},
        ],
    }
    print("Template variables preview:", build_template_variables(sample_transaction))
    print(f"DRY_RUN is currently: {DRY_RUN}")
    # Safe to call any time -- in dry-run mode (default) this never hits
    # Twilio or spends a free message. Set WHATSAPP_DRY_RUN=false in .env
    # when you actually want it to send.
    sid = send_whatsapp_reminder(sample_transaction)
    print("Result SID:", sid)