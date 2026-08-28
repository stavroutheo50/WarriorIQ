from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage

from core.config import SETTINGS


def send_transactional_email(recipient: str, subject: str, body: str) -> bool:
    """Send through configured SMTP without persisting secret one-time links."""
    if SETTINGS.email_provider.lower() != "smtp":
        return False
    host = os.getenv("WARRIORIQ_SMTP_HOST", "").strip()
    username = os.getenv("WARRIORIQ_SMTP_USERNAME", "").strip()
    password = os.getenv("WARRIORIQ_SMTP_PASSWORD", "")
    sender = os.getenv("WARRIORIQ_EMAIL_FROM", "").strip()
    if not host or not sender:
        return False
    port = int(os.getenv("WARRIORIQ_SMTP_PORT", "587"))
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=15) as client:
        client.starttls(context=context)
        if username:
            client.login(username, password)
        client.send_message(message)
    return True
