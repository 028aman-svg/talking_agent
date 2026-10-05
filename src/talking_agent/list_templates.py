"""
list_templates.py
Fetches the Content Templates available in your Twilio account, including
the Sandbox's pre-approved templates. Run this once to find the correct
ContentSid to use in whatsapp.py / .env, instead of guessing.

Usage:
    uv run list_templates.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

if not ACCOUNT_SID or not AUTH_TOKEN:
    raise SystemExit("Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env first.")

response = requests.get(
    "https://content.twilio.com/v1/Content",
    auth=(ACCOUNT_SID, AUTH_TOKEN),
)
response.raise_for_status()
data = response.json()

contents = data.get("contents", [])

if not contents:
    print("No templates found on this account.")
else:
    print(f"Found {len(contents)} template(s):\n")
    for tpl in contents:
        sid = tpl.get("sid")
        friendly_name = tpl.get("friendly_name")
        language = tpl.get("language")
        types = tpl.get("types", {})

        # Pull the body text out of whichever content type this template uses
        body = None
        for type_key, type_val in types.items():
            if isinstance(type_val, dict) and "body" in type_val:
                body = type_val["body"]
                break

        print(f"SID:      {sid}")
        print(f"Name:     {friendly_name}")
        print(f"Language: {language}")
        print(f"Body:     {body}")
        print("-" * 60)