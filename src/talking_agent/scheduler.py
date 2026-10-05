"""
scheduler.py
Runs two daily background jobs:
  1. WhatsApp reminder -- sent ONE DAY BEFORE the due date (up to
     database.MAX_REMINDERS times per transaction).
  2. Payment call -- an outbound phone call, placed via outbound_caller.py
     (LangGraph agent over Telnyx Call Control), placed ON the due
     date itself.

Run this as a long-lived process alongside the Streamlit app:
    uv run scheduler.py
For production, run it under a process manager (systemd, supervisor, pm2,
or Windows Task Scheduler) so it restarts automatically and keeps running
in the background. The calling agent also needs telnyx_voice_server.py
running separately (see that file's docstring) and reachable via
PUBLIC_BASE_URL.
"""

import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

import database as db
from whatsapp import send_whatsapp_reminder
from outbound_caller import trigger_payment_call

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("khata-scheduler")

# Times of day the two daily checks run (24-hour format). Change as needed.
WHATSAPP_RUN_HOUR = 10
WHATSAPP_RUN_MINUTE = 0

CALL_RUN_HOUR = 11
CALL_RUN_MINUTE = 59
def send_reminders_due_tomorrow():
    """WhatsApp reminder for transactions due TOMORROW, up to MAX_REMINDERS each."""
    logger.info("Running due-tomorrow WhatsApp check...")
    due_tomorrow = db.get_due_tomorrow()

    if not due_tomorrow:
        logger.info("No payments due tomorrow. Nothing to send.")
        return

    logger.info(f"Found {len(due_tomorrow)} payment(s) due tomorrow.")

    for tx in due_tomorrow:
        current_count = db.get_reminder_count(tx["id"])
        if current_count >= db.MAX_REMINDERS:
            logger.info(
                f"Skipping transaction #{tx['id']} — reminder limit "
                f"({db.MAX_REMINDERS}) already reached."
            )
            continue

        try:
            sid = send_whatsapp_reminder(tx)
            new_count = db.increment_reminder_count(tx["id"])
            logger.info(
                f"Sent WhatsApp reminder to {tx['borrower_name']} ({tx['phone_number']}) "
                f"for transaction #{tx['id']} — message SID: {sid} "
                f"({new_count}/{db.MAX_REMINDERS} reminders sent)"
            )
        except Exception as e:
            logger.error(
                f"Failed to send WhatsApp reminder for transaction #{tx['id']} "
                f"({tx['borrower_name']}, {tx['phone_number']}): {e}"
            )


def place_calls_due_today():
    """
    Outbound payment call for transactions due TODAY, up to MAX_CALLS each.
    A transaction can be "due today" either because it was originally due
    today, or because the customer promised on a prior call to pay today
    (see database.save_call_outcome, which moves due_date forward to the
    promised date). Either way, we stop calling once MAX_CALLS is reached --
    the UI then shows the "give up" message instead of a progress bar.
    """
    logger.info("Running due-today call check...")
    due_today = db.get_due_today()

    if not due_today:
        logger.info("No payments due today. No calls to place.")
        return

    logger.info(f"Found {len(due_today)} payment(s) due today.")

    for tx in due_today:
        current_calls = tx.get("call_count", 0)
        if current_calls >= db.MAX_CALLS:
            logger.info(
                f"Skipping transaction #{tx['id']} — call limit "
                f"({db.MAX_CALLS}) already reached."
            )
            continue

        try:
            sid = trigger_payment_call(tx)
            logger.info(
                f"Placed payment call to {tx['borrower_name']} ({tx['phone_number']}) "
                f"for transaction #{tx['id']} — call SID: {sid} "
                f"({current_calls + 1}/{db.MAX_CALLS} calls made)"
            )
        except Exception as e:
            logger.error(
                f"Failed to place call for transaction #{tx['id']} "
                f"({tx['borrower_name']}, {tx['phone_number']}): {e}"
            )


def run_once_now():
    """Manually trigger both checks immediately (useful for testing)."""
    send_reminders_due_tomorrow()
    place_calls_due_today()


def start_scheduler():
    """Start both recurring daily jobs and block, keeping the process alive."""
    db.init_db()
    scheduler = BlockingScheduler()

    scheduler.add_job(
        send_reminders_due_tomorrow,
        trigger=CronTrigger(hour=WHATSAPP_RUN_HOUR, minute=WHATSAPP_RUN_MINUTE),
        id="whatsapp_reminder_due_tomorrow",
        name="Send WhatsApp reminders for payments due tomorrow",
    )
    scheduler.add_job(
        place_calls_due_today,
        trigger=CronTrigger(hour=CALL_RUN_HOUR, minute=CALL_RUN_MINUTE),
        id="call_due_today",
        name="Place payment calls for payments due today",
    )

    logger.info(
        f"Scheduler started. WhatsApp reminders daily at "
        f"{WHATSAPP_RUN_HOUR:02d}:{WHATSAPP_RUN_MINUTE:02d}, "
        f"payment calls daily at {CALL_RUN_HOUR:02d}:{CALL_RUN_MINUTE:02d}. "
        f"Press Ctrl+C to stop."
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--now":
        # Run: `uv run scheduler.py --now` to test both jobs immediately
        db.init_db()
        run_once_now()
    else:
        start_scheduler()