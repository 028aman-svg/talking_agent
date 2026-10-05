"""
app.py
Streamlit frontend for the shopkeeper to log borrower transactions
(multiple products per visit) and view the ledger.
"""

import streamlit as st
from datetime import date, timedelta

import database as db

db.init_db()

MAX_REMINDERS = getattr(db, "MAX_REMINDERS", 3)  # falls back to 3 if db.py doesn't define it
MAX_CALLS = getattr(db, "MAX_CALLS", 3)  # calls are capped the same way as WhatsApp reminders

st.set_page_config(page_title="Khata Book", page_icon="📒", layout="wide")
st.title("📒 Khata Book — Borrower Ledger")

# ---------------------------------------------------------------------
# Session state for dynamically adding product rows before submission
# ---------------------------------------------------------------------
if "cart" not in st.session_state:
    st.session_state.cart = [{"product_name": "", "quantity": 1, "price_per_unit": 0.0}]

def add_item_row():
    st.session_state.cart.append({"product_name": "", "quantity": 1, "price_per_unit": 0.0})

def remove_item_row(index):
    if len(st.session_state.cart) > 1:
        st.session_state.cart.pop(index)

def clear_form():
    st.session_state.cart = [{"product_name": "", "quantity": 1, "price_per_unit": 0.0}]

# ---------------------------------------------------------------------
# Session state for editing an existing transaction in the ledger
# ---------------------------------------------------------------------
if "editing_id" not in st.session_state:
    st.session_state.editing_id = None  # id of the transaction currently being edited, or None

def start_edit(tx):
    st.session_state.editing_id = tx["id"]
    # Seed a separate cart for this edit session, one dict per product row
    st.session_state[f"edit_cart_{tx['id']}"] = [
        {
            "product_name": it["product_name"],
            "quantity": it["quantity"],
            "price_per_unit": it["price_per_unit"],
        }
        for it in tx["items"]
    ]

def cancel_edit(tx_id):
    st.session_state.editing_id = None
    st.session_state.pop(f"edit_cart_{tx_id}", None)

def add_edit_item_row(tx_id):
    st.session_state[f"edit_cart_{tx_id}"].append(
        {"product_name": "", "quantity": 1, "price_per_unit": 0.0}
    )

def remove_edit_item_row(tx_id, index):
    cart = st.session_state[f"edit_cart_{tx_id}"]
    if len(cart) > 1:
        cart.pop(index)

# ---------------------------------------------------------------------
# New transaction form
# ---------------------------------------------------------------------
st.subheader("New Entry")

col1, col2 = st.columns(2)
with col1:
    borrower_name = st.text_input("Borrower Name")
    phone_number = st.text_input("Phone Number (with country code, e.g. +919876543210)")
    purchase_date = st.date_input("Date of Purchase", value=date.today())
with col2:
    due_date = st.date_input("Date He Will Give Money", value=date.today() + timedelta(days=7))

st.markdown("**Products Purchased**")

for i, item in enumerate(st.session_state.cart):
    row1, row2, row3, row4 = st.columns([3, 1, 1.5, 0.5])
    with row1:
        st.session_state.cart[i]["product_name"] = st.text_input(
            "Product Name", value=item["product_name"], key=f"product_{i}"
        )
    with row2:
        st.session_state.cart[i]["quantity"] = st.number_input(
            "Qty", min_value=1, value=item["quantity"], key=f"qty_{i}"
        )
    with row3:
        st.session_state.cart[i]["price_per_unit"] = st.number_input(
            "Price per unit (₹)", min_value=0.0, value=item["price_per_unit"], key=f"price_{i}"
        )
    with row4:
        st.write("")
        st.write("")
        if st.button("🗑️", key=f"remove_{i}"):
            remove_item_row(i)
            st.rerun()

st.button("➕ Add another product", on_click=add_item_row)

# Running total preview
running_total = sum(it["quantity"] * it["price_per_unit"] for it in st.session_state.cart)
st.markdown(f"### Total Amount: ₹{running_total:,.2f}")

if st.button("✅ Save Entry", type="primary"):
    valid_items = [
        it for it in st.session_state.cart
        if it["product_name"].strip() and it["quantity"] > 0
    ]
    if not borrower_name.strip():
        st.error("Please enter the borrower's name.")
    elif not phone_number.strip():
        st.error("Please enter a phone number.")
    elif not valid_items:
        st.error("Please add at least one valid product.")
    elif due_date < purchase_date:
        st.error("Due date cannot be before the purchase date.")
    else:
        tid = db.add_transaction(
            borrower_name=borrower_name.strip(),
            phone_number=phone_number.strip(),
            purchase_date=purchase_date.isoformat(),
            due_date=due_date.isoformat(),
            items=valid_items,
        )
        st.success(f"Saved! Transaction #{tid} for {borrower_name} — ₹{running_total:,.2f}")
        clear_form()
        st.rerun()

st.divider()

# ---------------------------------------------------------------------
# Ledger view
# ---------------------------------------------------------------------
st.subheader("📋 Ledger")

filter_option = st.radio("Show", ["All", "Unpaid only", "Paid only"], horizontal=True)

transactions = db.get_all_transactions()
if filter_option == "Unpaid only":
    transactions = [t for t in transactions if not t["is_paid"]]
elif filter_option == "Paid only":
    transactions = [t for t in transactions if t["is_paid"]]

if not transactions:
    st.info("No entries yet.")
else:
    for tx in transactions:
        status = "✅ Paid" if tx["is_paid"] else "🔴 Unpaid"
        reminder_count = tx.get("reminder_count", 0)  # .get() so this doesn't break before db.py is updated
        call_count = tx.get("call_count", 0)
        header = (
            f"{tx['borrower_name']} — ₹{tx['total_amount']:,.2f} — "
            f"due {tx['due_date']} — {status} — "
            f"🔔 {reminder_count}/{MAX_REMINDERS} reminders — 📞 {call_count}/{MAX_CALLS} calls"
        )
        with st.expander(header):
            is_editing = st.session_state.editing_id == tx["id"]

            if not is_editing:
                # -----------------------------------------------------
                # READ-ONLY VIEW
                # -----------------------------------------------------
                st.write(f"**Phone:** {tx['phone_number']}")
                st.write(f"**Purchase Date:** {tx['purchase_date']}")
                st.write(f"**Due Date:** {tx['due_date']}")

                notif_col1, notif_col2 = st.columns(2)
                with notif_col1:
                    if reminder_count >= MAX_REMINDERS:
                        st.warning(f"🔔 Reminder limit reached ({reminder_count}/{MAX_REMINDERS}) — no more will be sent automatically.")
                    else:
                        st.caption(f"🔔 WhatsApp reminders sent: {reminder_count}/{MAX_REMINDERS}")
                        st.progress(reminder_count / MAX_REMINDERS)
                with notif_col2:
                    if call_count >= MAX_CALLS and not tx["is_paid"]:
                        st.error("📞 MAALIK aap karo paisa wassooli! 😅")
                        st.caption(f"({call_count}/{MAX_CALLS} calls made — agent won't call again automatically.)")
                    elif call_count >= MAX_CALLS:
                        st.caption(f"📞 Payment calls made: {call_count}/{MAX_CALLS}")
                    else:
                        st.caption(f"📞 Payment calls made: {call_count}/{MAX_CALLS}")
                        st.progress(call_count / MAX_CALLS)

                if tx.get("call_outcome"):
                    outcome_label = str(tx["call_outcome"]).replace("_", " ").title()
                    if tx.get("call_promised_date"):
                        st.caption(f"Last call outcome: **{outcome_label}** — promised to pay by {tx['call_promised_date']}")
                    else:
                        st.caption(f"Last call outcome: **{outcome_label}**")

                st.write("**Products:**")
                for prod in tx["items"]:
                    st.write(
                        f"- {prod['product_name']} × {prod['quantity']} "
                        f"@ ₹{prod['price_per_unit']:,.2f} = ₹{prod['item_total']:,.2f}"
                    )

                c1, c2, c3 = st.columns(3)
                with c1:
                    if not tx["is_paid"]:
                        if st.button("Mark as Paid", key=f"paid_{tx['id']}"):
                            db.mark_as_paid(tx["id"])
                            st.rerun()
                with c2:
                    if st.button("✏️ Edit", key=f"edit_{tx['id']}"):
                        start_edit(tx)
                        st.rerun()
                with c3:
                    if st.button("Delete Entry", key=f"del_{tx['id']}"):
                        db.delete_transaction(tx["id"])
                        st.rerun()

            else:
                # -----------------------------------------------------
                # EDIT MODE — all fields editable
                # -----------------------------------------------------
                edit_col1, edit_col2 = st.columns(2)
                with edit_col1:
                    edit_borrower_name = st.text_input(
                        "Borrower Name", value=tx["borrower_name"], key=f"edit_name_{tx['id']}"
                    )
                    edit_phone_number = st.text_input(
                        "Phone Number", value=tx["phone_number"], key=f"edit_phone_{tx['id']}"
                    )
                    edit_purchase_date = st.date_input(
                        "Date of Purchase",
                        value=date.fromisoformat(tx["purchase_date"]),
                        key=f"edit_pdate_{tx['id']}",
                    )
                with edit_col2:
                    edit_due_date = st.date_input(
                        "Date He Will Give Money",
                        value=date.fromisoformat(tx["due_date"]),
                        key=f"edit_ddate_{tx['id']}",
                    )
                    edit_is_paid = st.checkbox(
                        "Paid", value=bool(tx["is_paid"]), key=f"edit_paid_{tx['id']}"
                    )

                st.markdown("**Products Purchased**")
                edit_cart = st.session_state[f"edit_cart_{tx['id']}"]
                for i, item in enumerate(edit_cart):
                    row1, row2, row3, row4 = st.columns([3, 1, 1.5, 0.5])
                    with row1:
                        edit_cart[i]["product_name"] = st.text_input(
                            "Product Name", value=item["product_name"], key=f"edit_product_{tx['id']}_{i}"
                        )
                    with row2:
                        edit_cart[i]["quantity"] = st.number_input(
                            "Qty", min_value=1, value=item["quantity"], key=f"edit_qty_{tx['id']}_{i}"
                        )
                    with row3:
                        edit_cart[i]["price_per_unit"] = st.number_input(
                            "Price per unit (₹)", min_value=0.0, value=item["price_per_unit"], key=f"edit_price_{tx['id']}_{i}"
                        )
                    with row4:
                        st.write("")
                        st.write("")
                        if st.button("🗑️", key=f"edit_remove_{tx['id']}_{i}"):
                            remove_edit_item_row(tx["id"], i)
                            st.rerun()

                st.button(
                    "➕ Add another product",
                    key=f"edit_add_{tx['id']}",
                    on_click=add_edit_item_row,
                    args=(tx["id"],),
                )

                edit_total = sum(it["quantity"] * it["price_per_unit"] for it in edit_cart)
                st.markdown(f"**New Total: ₹{edit_total:,.2f}**")

                save_col, cancel_col = st.columns(2)
                with save_col:
                    if st.button("💾 Save Changes", key=f"save_{tx['id']}", type="primary"):
                        valid_items = [
                            it for it in edit_cart
                            if it["product_name"].strip() and it["quantity"] > 0
                        ]
                        if not edit_borrower_name.strip():
                            st.error("Borrower name can't be empty.")
                        elif not edit_phone_number.strip():
                            st.error("Phone number can't be empty.")
                        elif not valid_items:
                            st.error("Please keep at least one valid product.")
                        elif edit_due_date < edit_purchase_date:
                            st.error("Due date cannot be before the purchase date.")
                        else:
                            db.update_transaction(
                                transaction_id=tx["id"],
                                borrower_name=edit_borrower_name.strip(),
                                phone_number=edit_phone_number.strip(),
                                purchase_date=edit_purchase_date.isoformat(),
                                due_date=edit_due_date.isoformat(),
                                items=valid_items,
                            )
                            # Paid status isn't part of update_transaction; set separately
                            if edit_is_paid and not tx["is_paid"]:
                                db.mark_as_paid(tx["id"])
                            cancel_edit(tx["id"])
                            st.success("Changes saved.")
                            st.rerun()
                with cancel_col:
                    if st.button("❌ Cancel", key=f"cancel_{tx['id']}"):
                        cancel_edit(tx["id"])
                        st.rerun()