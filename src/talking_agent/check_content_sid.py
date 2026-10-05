"""
Quick diagnostic: confirms whether the Content SID your .env resolves to
actually exists under the Twilio account your credentials point to.
Run this locally (not here, no network access in this sandbox) with:
    uv run check_content_sid.py
"""
import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv()

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
CONTENT_SID = os.getenv("TWILIO_CONTENT_SID", "HXcff646f616911d1e466eb1c5634ba3b9")

print(f"Account SID in use : {ACCOUNT_SID}")
print(f"Content SID in use : {CONTENT_SID}")

client = Client(ACCOUNT_SID, AUTH_TOKEN)

try:
    content = client.content.v1.contents(CONTENT_SID).fetch()
    print("FOUND. Friendly name:", content.friendly_name)
except Exception as e:
    print("NOT FOUND under this account. Error:", e)