"""
database.py
Handles all SQLite database operations for the Khata (credit book) app.
Supports multiple products per transaction.
"""

import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "khata.db"

MAX_REMINDERS = 3  # cap on how many WhatsApp reminders we'll send per transaction
MAX_CALLS = 2  # cap on how many payment calls we'll place per transaction


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            borrower_name TEXT NOT NULL,
            phone_number TEXT NOT NULL,
            purchase_date TEXT NOT NULL,
            due_date TEXT NOT NULL,
            total_amount REAL NOT NULL DEFAULT 0,
            is_paid INTEGER NOT NULL DEFAULT 0,
            reminder_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transaction_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id INTEGER NOT NULL,
            product_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price_per_unit REAL NOT NULL,
            item_total REAL NOT NULL,
            FOREIGN KEY (transaction_id) REFERENCES transactions (id)
                ON DELETE CASCADE
        )
        """
    )

    # --- Migration: if you already have a khata.db from before this change,
    # it has an old "reminder_sent" column but no "reminder_count" column.
    # This adds reminder_count (defaulting existing rows to 0) without
    # touching your existing data. Safe to run every startup.
    existing_columns = {row["name"] for row in cursor.execute("PRAGMA table_info(transactions)")}
    if "reminder_count" not in existing_columns:
        cursor.execute(
            "ALTER TABLE transactions ADD COLUMN reminder_count INTEGER NOT NULL DEFAULT 0"
        )
    if "call_outcome" not in existing_columns:
        cursor.execute("ALTER TABLE transactions ADD COLUMN call_outcome TEXT")
    if "call_promised_date" not in existing_columns:
        cursor.execute("ALTER TABLE transactions ADD COLUMN call_promised_date TEXT")
    if "call_count" not in existing_columns:
        cursor.execute(
            "ALTER TABLE transactions ADD COLUMN call_count INTEGER NOT NULL DEFAULT 0"
        )

    conn.commit()
    conn.close()


def add_transaction(
    borrower_name: str,
    phone_number: str,
    purchase_date: str,
    due_date: str,
    items: list[dict],
):
    """
    items: list of dicts like
        {"product_name": "Rice 5kg", "quantity": 2, "price_per_unit": 250.0}
    Returns the new transaction id.
    """
    total_amount = sum(item["quantity"] * item["price_per_unit"] for item in items)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO transactions
            (borrower_name, phone_number, purchase_date, due_date, total_amount)
        VALUES (?, ?, ?, ?, ?)
        """,
        (borrower_name, phone_number, purchase_date, due_date, total_amount),
    )
    transaction_id = cursor.lastrowid

    for item in items:
        item_total = item["quantity"] * item["price_per_unit"]
        cursor.execute(
            """
            INSERT INTO transaction_items
                (transaction_id, product_name, quantity, price_per_unit, item_total)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                transaction_id,
                item["product_name"],
                item["quantity"],
                item["price_per_unit"],
                item_total,
            ),
        )

    conn.commit()
    conn.close()
    return transaction_id


def get_items_for_transaction(transaction_id: int):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM transaction_items WHERE transaction_id = ?",
        (transaction_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_transactions():
    """Return all transactions (most recent first), each with its items nested."""
    conn = get_connection()
    tx_rows = conn.execute(
        "SELECT * FROM transactions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    transactions = []
    for row in tx_rows:
        tx = dict(row)
        tx["items"] = get_items_for_transaction(tx["id"])
        transactions.append(tx)
    return transactions


def get_due_today(today: str = None):
    """Unpaid transactions whose due_date is today, each with items nested."""
    if today is None:
        today = date.today().isoformat()

    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM transactions WHERE due_date = ? AND is_paid = 0",
        (today,),
    ).fetchall()
    conn.close()

    due = []
    for row in rows:
        tx = dict(row)
        tx["items"] = get_items_for_transaction(tx["id"])
        due.append(tx)
    return due


def get_due_tomorrow(today: str = None):
    """
    Unpaid transactions whose due_date is tomorrow, each with items nested.
    Used to send the WhatsApp reminder one day before the due date (the
    phone call happens separately, on the actual due date -- see
    get_due_today()).
    """
    if today is None:
        today = date.today()
    else:
        today = date.fromisoformat(today)
    tomorrow = (today + timedelta(days=1)).isoformat()

    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM transactions WHERE due_date = ? AND is_paid = 0",
        (tomorrow,),
    ).fetchall()
    conn.close()

    due = []
    for row in rows:
        tx = dict(row)
        tx["items"] = get_items_for_transaction(tx["id"])
        due.append(tx)
    return due


def mark_as_paid(transaction_id: int):
    conn = get_connection()
    conn.execute("UPDATE transactions SET is_paid = 1 WHERE id = ?", (transaction_id,))
    conn.commit()
    conn.close()


def get_reminder_count(transaction_id: int) -> int:
    """Return how many reminders have been sent so far for this transaction."""
    conn = get_connection()
    row = conn.execute(
        "SELECT reminder_count FROM transactions WHERE id = ?", (transaction_id,)
    ).fetchone()
    conn.close()
    return row["reminder_count"] if row else 0


def increment_reminder_count(transaction_id: int):
    """
    Increment the reminder counter for a transaction, capped at MAX_REMINDERS.
    Returns the new count, or None if the cap was already reached (caller
    should treat None as "don't send").
    """
    current = get_reminder_count(transaction_id)
    if current >= MAX_REMINDERS:
        return None

    conn = get_connection()
    conn.execute(
        "UPDATE transactions SET reminder_count = reminder_count + 1 WHERE id = ?",
        (transaction_id,),
    )
    conn.commit()
    conn.close()
    return current + 1


def get_call_count(transaction_id: int) -> int:
    """Return how many payment calls have been placed so far for this transaction."""
    conn = get_connection()
    row = conn.execute(
        "SELECT call_count FROM transactions WHERE id = ?", (transaction_id,)
    ).fetchone()
    conn.close()
    return row["call_count"] if row else 0


def save_call_outcome(transaction_id: int, outcome: str, promised_date: str = None):
    """
    Record the result of an outbound payment-reminder call.
    outcome: one of "promised", "refused", "ended", "disconnected", "unknown"
             (whatever call_agent.py / voice_server.py decided).
    promised_date: ISO date string if the customer committed to a new date,
                   else None.
    Also bumps call_count so the UI can show how many calls have been made.

    Rescheduling logic for due_date (so get_due_today() knows when to try
    again -- scheduler.py separately enforces MAX_CALLS so this doesn't
    repeat forever):
      - outcome == "promised" and a date was given -> due_date moves to
        that promised date (customer's own commitment takes priority).
      - anything else (refused, ended with no promise, disconnected/no
        answer, unknown) -> due_date moves forward by 2 days, so the agent
        tries again in 2 days rather than never getting picked up again.
    """
    if outcome == "promised" and promised_date:
        next_due_date = promised_date
    else:
        next_due_date = (date.today() + timedelta(days=2)).isoformat()

    conn = get_connection()
    conn.execute(
        """
        UPDATE transactions
        SET call_outcome = ?, call_promised_date = ?, call_count = call_count + 1,
            due_date = ?
        WHERE id = ?
        """,
        (outcome, promised_date, next_due_date, transaction_id),
    )
    conn.commit()
    conn.close()


def update_transaction(
    transaction_id: int,
    borrower_name: str,
    phone_number: str,
    purchase_date: str,
    due_date: str,
    items: list[dict],
):
    """
    Overwrite an existing transaction's fields and its full list of items.
    items: list of dicts like
        {"product_name": "Rice 5kg", "quantity": 2, "price_per_unit": 250.0}
    Recomputes total_amount from the new items. Existing transaction_items
    rows are replaced wholesale (simplest way to handle added/removed/edited
    rows without needing to diff them).
    """
    total_amount = sum(item["quantity"] * item["price_per_unit"] for item in items)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE transactions
        SET borrower_name = ?, phone_number = ?, purchase_date = ?,
            due_date = ?, total_amount = ?
        WHERE id = ?
        """,
        (borrower_name, phone_number, purchase_date, due_date, total_amount, transaction_id),
    )

    cursor.execute("DELETE FROM transaction_items WHERE transaction_id = ?", (transaction_id,))
    for item in items:
        item_total = item["quantity"] * item["price_per_unit"]
        cursor.execute(
            """
            INSERT INTO transaction_items
                (transaction_id, product_name, quantity, price_per_unit, item_total)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                transaction_id,
                item["product_name"],
                item["quantity"],
                item["price_per_unit"],
                item_total,
            ),
        )

    conn.commit()
    conn.close()


def delete_transaction(transaction_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM transaction_items WHERE transaction_id = ?", (transaction_id,))
    conn.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized at:", DB_PATH)